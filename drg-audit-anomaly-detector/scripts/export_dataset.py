#!/usr/bin/env python
"""Export a curated dataset from the prototype's ``data_prep`` module to the Parquet data contract.

``data_prep.py`` is not part of this repository. Run this from the directory that contains it:

    python scripts/export_dataset.py --split train --output data/train.parquet
    python scripts/export_dataset.py --split inference --output data/inference.parquet

The output is validated against the contract (see drg_audit.data.io) and written owner-readable only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drg_audit.data.io import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, dataframe_to_records  # noqa: E402
from drg_audit.features import normalize_code  # noqa: E402

MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ARRAY_COLUMNS = ("diagnosis_code_array", "procedure_code_array", "medical_claims_array")


def _as_list(value) -> list[str]:
    if value is None or isinstance(value, float):  # None or NaN
        return []
    return [normalize_code(c) for c in value if c is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "inference"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", default="data_prep", help="plain module name, resolved from the current directory")
    parser.add_argument("--limit", type=int, help="export only the first N rows (for trials)")
    parser.add_argument("--max-invalid-fraction", type=float, default=0.01)
    args = parser.parse_args()

    if not MODULE_RE.fullmatch(args.module):
        raise SystemExit("--module must be a plain module name, not a path")
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(args.module)
    df = getattr(module, "prep_train_data" if args.split == "train" else "prep_inference_data")()
    if args.limit:
        df = df.head(args.limit)

    _, report = dataframe_to_records(df, max_invalid_fraction=args.max_invalid_fraction)  # quality gate
    columns = [c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if c in df.columns]
    out = df[columns].copy()
    for column in ARRAY_COLUMNS:
        out[column] = out[column].map(_as_list)
    out["admission_no"] = out["admission_no"].map(
        lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))
    out["drg_vers"] = out["drg_vers"].astype(str)
    out["drg"] = out["drg"].fillna("").astype(str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)  # requires pyarrow
    os.chmod(args.output, 0o600)
    print(json.dumps({**report.to_dict(), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
