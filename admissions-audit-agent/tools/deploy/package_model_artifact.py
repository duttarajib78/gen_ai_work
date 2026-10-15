"""Build the SageMaker model artifact (``model.tar.gz``) from the reference data.

The artifact is the only thing the endpoint reads at start-up besides the image:

    manifest.json                       sha256 + size of every file (verified at start-up)
    reference/diagnosis_codes.tsv
    reference/achi_codes.tsv
    reference/MBS-XML-YYYYMMDD.XML      latest file in the source directory
    drg/...                             DRG cost-weight reference (for the grouper)

The reference files are loaded and validated here with the same loaders the
service uses, so a wrong file fails now rather than on the endpoint. The tarball
is deterministic (sorted entries, fixed ownership and permissions, no symlinks).

    python -m tools.deploy.package_model_artifact --reference-dir ./reference \
        --drg-reference-dir ./drg_reference --output artifacts/model.tar.gz \
        --s3-uri s3://my-bucket/admissions-audit-agent/2026-10-15/model.tar.gz --kms-key-id alias/my-key

Uploads always use SSE-KMS (the AWS-managed ``aws/s3`` key if no key is given).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

from audit_agent.reference_data import (
    ACHI_FILE, DIAGNOSIS_FILE, MANIFEST_NAME, MANIFEST_SCHEMA_VERSION, ReferenceData, find_latest_mbs_xml, sha256_file,
)

TOOL_NAME = "package_model_artifact/1"


class PackagingError(Exception):
    pass


def build_manifest(root: Path, *, created_at: Optional[str] = None) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PackagingError(f"refusing to package a symlink: {path.relative_to(root)}")
        if path.is_file() and path.name != MANIFEST_NAME:
            rel = path.relative_to(root).as_posix()
            files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL_NAME,
        "files": files,
    }


def stage_artifact(reference_dir: Path, drg_dir: Optional[Path], staging: Path) -> ReferenceData:
    ref = ReferenceData.from_directory(reference_dir)  # validates shape, code types, and the MBS XML
    target = staging / "reference"
    target.mkdir(parents=True)
    shutil.copyfile(reference_dir / DIAGNOSIS_FILE, target / DIAGNOSIS_FILE)
    shutil.copyfile(reference_dir / ACHI_FILE, target / ACHI_FILE)
    mbs = find_latest_mbs_xml(reference_dir)
    shutil.copyfile(mbs, target / mbs.name)

    if drg_dir is not None:
        if not drg_dir.is_dir():
            raise PackagingError("the DRG reference directory does not exist")
        dest = staging / "drg"
        for src in sorted(drg_dir.rglob("*")):
            if src.is_symlink():
                raise PackagingError(f"refusing to package a symlink: {src.relative_to(drg_dir)}")
            rel = src.relative_to(drg_dir)
            if src.is_dir():
                (dest / rel).mkdir(parents=True, exist_ok=True)
            else:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest / rel)
        if not any(dest.rglob("*")):
            raise PackagingError("the DRG reference directory is empty")
    return ref


def write_tarball(staging: Path, output: Path, *, mtime: Optional[int] = None) -> None:
    mtime = int(os.environ.get("SOURCE_DATE_EPOCH", time.time())) if mtime is None else mtime
    output.parent.mkdir(parents=True, exist_ok=True)

    def normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = mtime
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with open(output, "wb") as raw, tarfile.open(fileobj=raw, mode="w:gz", compresslevel=6) as tar:
        for path in sorted(staging.rglob("*")):
            tar.add(path, arcname=path.relative_to(staging).as_posix(), recursive=False, filter=normalise)


def upload(path: Path, s3_uri: str, kms_key_id: Optional[str]) -> None:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise PackagingError("expected --s3-uri s3://bucket/key")
    import boto3  # imported lazily so packaging works offline

    extra = {"ServerSideEncryption": "aws:kms"}
    if kms_key_id:
        extra["SSEKMSKeyId"] = kms_key_id
    boto3.client("s3").upload_file(str(path), parsed.netloc, parsed.path.lstrip("/"), ExtraArgs=extra)


def build_artifact(reference_dir: Path, drg_dir: Optional[Path], output: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "model"
        staging.mkdir()
        ref = stage_artifact(reference_dir, drg_dir, staging)
        manifest = build_manifest(staging)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        write_tarball(staging, output)
    return {"manifest": manifest, "counts": ref.counts(), "sha256": sha256_file(output), "bytes": output.stat().st_size}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", required=True, type=Path,
                   help="directory with diagnosis_codes.tsv, achi_codes.tsv and MBS-XML-*.XML")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--drg-reference-dir", type=Path, help="directory with the DRG cost-weight reference data")
    group.add_argument("--no-drg", action="store_true", help="omit DRG data (only with AUDIT_GROUPER_BACKEND=disabled)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--s3-uri")
    p.add_argument("--kms-key-id", help="KMS key id/ARN/alias for SSE-KMS (default: the aws/s3 managed key)")
    a = p.parse_args(argv)
    try:
        info = build_artifact(a.reference_dir, None if a.no_drg else a.drg_reference_dir, a.output)
        print(f"wrote {a.output} ({info['bytes']} bytes, sha256 {info['sha256']})")
        print(f"reference data: {info['counts']}; {len(info['manifest']['files'])} file(s) in the manifest")
        if a.s3_uri:
            upload(a.output, a.s3_uri, a.kms_key_id)
            print(f"uploaded to {a.s3_uri} with SSE-KMS")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
