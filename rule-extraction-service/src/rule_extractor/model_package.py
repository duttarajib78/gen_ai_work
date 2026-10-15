"""Model artifact (``model.tar.gz``) assembly and integrity verification.

SageMaker extracts the tarball to ``/opt/ml/model``. The manifest lists every file with its SHA-256; the container
verifies it at start-up and refuses to serve on any mismatch, unexpected file, symlink or non-regular file (the corpus
is read by the agent's file tools, so its integrity is a security boundary).

Layout:  manifest.json  corpus/  templates/rule_catalogue.xlsx  [base/executable_rules.jsonl]  [rules/executable_rules.jsonl]
"""
from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from . import __version__
from .errors import IntegrityError
from .util import atomic_write_text, sha256_file, utc_now_iso, validate_object_key

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = 1


def write_manifest(model_dir: Path | str) -> dict[str, Any]:
    root = Path(model_dir)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == MANIFEST_NAME:
            continue
        files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {"format_version": MANIFEST_FORMAT, "builder_version": __version__, "built_at": utc_now_iso(), "files": files}
    atomic_write_text(root / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_model_dir(corpus_dir: Path | str, template_path: Path | str, out_dir: Path | str, *,
                    base_rules_path: Path | str | None = None, evaluation_rules_path: Path | str | None = None,
                    force: bool = False) -> dict[str, Any]:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise IntegrityError(f"{out} is not empty; pass force=True to rebuild it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(corpus_dir, out / "corpus", symlinks=False, ignore=shutil.ignore_patterns(".tmp-*"))
    (out / "templates").mkdir()
    shutil.copy2(template_path, out / "templates" / "rule_catalogue.xlsx")
    if base_rules_path:
        (out / "base").mkdir()
        shutil.copy2(base_rules_path, out / "base" / "executable_rules.jsonl")
    if evaluation_rules_path:
        (out / "rules").mkdir()
        shutil.copy2(evaluation_rules_path, out / "rules" / "executable_rules.jsonl")
    return write_manifest(out)


def verify_model_dir(model_dir: Path | str) -> list[str]:
    """Return problems (empty list == healthy)."""
    root = Path(model_dir)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"{MANIFEST_NAME} missing in {root}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (ValueError, KeyError) as exc:
        return [f"{MANIFEST_NAME} unreadable: {type(exc).__name__}"]
    if manifest.get("format_version") != MANIFEST_FORMAT:
        return [f"unsupported manifest format {manifest.get('format_version')!r}"]
    problems: list[str] = []
    for rel, meta in files.items():
        try:
            validate_object_key(rel)
        except ValueError:
            problems.append(f"unsafe path in manifest: {rel!r}")
            continue
        path = root / rel
        if path.is_symlink() or not path.is_file():
            problems.append(f"missing or not a regular file: {rel}")
        elif path.stat().st_size != meta.get("size") or sha256_file(path) != meta.get("sha256"):
            problems.append(f"checksum mismatch: {rel}")
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            problems.append(f"symlink not allowed: {rel}")
        elif path.is_file() and rel != MANIFEST_NAME and rel not in files:
            problems.append(f"unexpected file: {rel}")
        elif not path.is_file() and not path.is_dir():
            problems.append(f"special file not allowed: {rel}")
    return problems[:50]


def package_tar(model_dir: Path | str, out_path: Path | str) -> str:
    """Deterministic ``model.tar.gz`` (sorted entries, zeroed mtime/owner). Returns its SHA-256."""
    root, out = Path(model_dir), Path(out_path)
    problems = verify_model_dir(root)
    if problems:
        raise IntegrityError("refusing to package an unhealthy model directory", details=problems)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(root.rglob("*")):
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.pax_headers = {}
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if path.is_dir():
                        tar.addfile(info)
                    else:
                        with open(path, "rb") as handle:
                            tar.addfile(info, handle)
    return sha256_file(out)
