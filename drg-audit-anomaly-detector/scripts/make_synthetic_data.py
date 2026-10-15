#!/usr/bin/env python
"""Write a synthetic (PHI-free) dataset to Parquet for smoke tests: ``python scripts/make_synthetic_data.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drg_audit.synthetic import generate_frame, write_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/synthetic/admissions.parquet")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(generate_frame(args.rows, args.seed), str(out))
    print(f"wrote {args.rows} synthetic admissions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
