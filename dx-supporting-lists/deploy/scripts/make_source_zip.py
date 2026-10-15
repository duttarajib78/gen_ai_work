#!/usr/bin/env python3
"""Zips the project for CodeBuild, leaving out everything that must never leave your machine (licensed data, results, env files, VCS metadata)."""
import sys
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {".git", ".venv", "venv", "data", "data-sample", "output", "compiled", "__pycache__", ".pytest_cache", ".ruff_cache", "docs", "tests"}
EXCLUDE_SUFFIXES = (".pyc", ".env", ".pem", ".zip", ".tar.gz")


def main(out: str) -> int:
    root = Path(__file__).resolve().parents[2]
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if p.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts) or p.name.startswith(".env") or str(p).endswith(EXCLUDE_SUFFIXES):
                continue
            if rel.parts[:2] == ("deploy", "cloudformation"):
                continue
            z.write(p, rel.as_posix())
            n += 1
    print(f"wrote {out} ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "source.zip"))
