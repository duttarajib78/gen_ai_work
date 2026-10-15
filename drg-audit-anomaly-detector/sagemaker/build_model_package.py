#!/usr/bin/env python
"""Build the SageMaker ``model.tar.gz``: verified model bundle + serving code (+ optional offline wheelhouse).

    python sagemaker/build_model_package.py --bundle-dir artifacts/bundle --output dist/model.tar.gz
    python sagemaker/build_model_package.py --model-tar training_output/model.tar.gz --output dist/model.tar.gz

Dependency delivery (``--dependency-mode``):
  wheelhouse  (default, safest) wheels are vendored into the package and installed offline, so the endpoint
              can run with network isolation and nothing is fetched at start-up.
  pypi        ``pip install`` from PyPI when the container starts. Needs outbound network access, so it is
              incompatible with network isolation and widens the supply-chain surface.

Outputs next to ``--output``: ``model.tar.gz.sha256`` and ``package_meta.json`` (consumed by deploy_endpoint.py,
which refuses to deploy a package that does not match its metadata).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _packaging as pk  # noqa: E402  (sibling module; sys.path[0] is this directory)

sys.path.insert(0, str(pk.REPO_ROOT / "src"))

from drg_audit import __version__  # noqa: E402
from drg_audit.archive import safe_extract_tar  # noqa: E402
from drg_audit.artifacts import MANIFEST_NAME, load_bundle  # noqa: E402

CODE_ENTRY = pk.REPO_ROOT / "sagemaker" / "code" / "inference.py"
DEFAULT_REQUIREMENTS = pk.REPO_ROOT / "requirements" / "inference.in"
CONTAINER_WHEELHOUSE = "/opt/ml/model/code/wheelhouse"


def build_package(
    *,
    bundle_dir: Path,
    output: Path,
    dependency_mode: str = "wheelhouse",
    requirements: Path = DEFAULT_REQUIREMENTS,
    python_version: str = "3.12",
    platforms: tuple[str, ...] = pk.DEFAULT_PLATFORMS,
) -> dict[str, Any]:
    if dependency_mode not in ("wheelhouse", "pypi"):
        raise ValueError("dependency_mode must be 'wheelhouse' or 'pypi'")
    bundle = load_bundle(bundle_dir, verify=True)  # refuse to package a tampered or incomplete bundle

    with tempfile.TemporaryDirectory(prefix="model-package-") as tmp:
        stage = Path(tmp) / "package"
        stage.mkdir()
        for name in [*bundle.manifest["files"], MANIFEST_NAME]:
            shutil.copy2(Path(bundle_dir) / name, stage / name)
        code = stage / "code"
        code.mkdir()
        shutil.copy2(CODE_ENTRY, code / "inference.py")
        pk.copy_package(code, serving_only=True)

        wheels: list[dict[str, Any]] = []
        requirement_lines = pk.read_requirements(requirements)
        if dependency_mode == "wheelhouse":
            wheels = pk.vendor_wheels(requirements, code / "wheelhouse", python_version=python_version,
                                      platforms=platforms)
            pk.write_requirements(code / "requirements.txt", requirement_lines, find_links=CONTAINER_WHEELHOUSE)
        else:
            pk.write_requirements(code / "requirements.txt", requirement_lines, find_links=None)
        pk.assert_import_closure(code)
        digest = pk.write_deterministic_targz(stage, output)

    meta = {
        "schema_version": 1,
        "package": output.name,
        "sha256": digest,
        "bytes": output.stat().st_size,
        "bundle_id": bundle.model_version,
        "tokenization": bundle.tokenization.name,
        "dependency_mode": dependency_mode,
        "python_version": python_version,
        "platforms": list(platforms),
        "requirements": requirement_lines,
        "wheelhouse": wheels,
        "package_version": __version__,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output.parent / f"{output.name}.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    (output.parent / "package_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle-dir", type=Path, help="serving bundle directory (output of `drg-audit train`)")
    source.add_argument("--model-tar", type=Path, help="model.tar.gz produced by a SageMaker training job")
    parser.add_argument("--output", type=Path, default=Path("dist/model.tar.gz"))
    parser.add_argument("--dependency-mode", choices=("wheelhouse", "pypi"), default="wheelhouse")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--python-version", default="3.12", help="must match the container (PyTorch 2.6 = 3.12)")
    parser.add_argument("--platform", action="append", dest="platforms",
                        help="wheel platform tag (repeatable); default: manylinux x86_64")
    args = parser.parse_args(argv)

    try:
        with tempfile.TemporaryDirectory(prefix="training-artifact-") as tmp:
            bundle_dir = args.bundle_dir
            if args.model_tar:
                safe_extract_tar(args.model_tar, tmp)
                bundle_dir = Path(tmp)
            meta = build_package(
                bundle_dir=bundle_dir, output=args.output, dependency_mode=args.dependency_mode,
                requirements=args.requirements, python_version=args.python_version,
                platforms=tuple(args.platforms) if args.platforms else pk.DEFAULT_PLATFORMS)
    except Exception as exc:  # surface a clean message; detail is in the exception
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: meta[k] for k in ("package", "sha256", "bytes", "bundle_id", "dependency_mode")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
