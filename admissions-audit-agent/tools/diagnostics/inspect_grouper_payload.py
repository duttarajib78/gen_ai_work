"""Build, without sending, the payload the grouper would receive for one episode.

For diagnosing a rejected grouper request by looking at the payload (especially None values
a strict backend refuses) instead of guessing. Needs the internal au_grouper module on the
path (vendor/). The payload contains demographic fields, so values are hidden unless
--show-values is given; by default only field names, types, and None fields are printed.

    PYTHONPATH=src:vendor python -m tools.diagnostics.inspect_grouper_payload --payload episode.json [--show-values]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from audit_agent.config import Settings
from audit_agent.grouper.corrections import apply_diagnosis_corrections, rows_to_diag_poa
from audit_agent.grouper.drg_version import derive_grouper_version
from audit_agent.schemas import parse_request


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--show-values", action="store_true")
    a = p.parse_args(argv)

    try:
        from au_grouper import FUND_ID_CODE_MAP, _build_au_grouper_payload
    except ImportError as exc:
        print(f"error: au_grouper is not importable ({exc}); add vendor/ to PYTHONPATH", file=sys.stderr)
        return 2

    settings = Settings.from_mapping({"AUDIT_AWS_REGION": "ap-southeast-2", "AUDIT_BEDROCK_MODEL_ID": "au.anthropic.claude-sonnet-5"})
    episode = parse_request(json.loads(a.payload.read_text(encoding="utf-8")), settings).episode
    fund_id = episode.extras.get("fund_id")
    print(f"fund_id mapped: {FUND_ID_CODE_MAP.get(fund_id) is not None}"
          + ("" if FUND_ID_CODE_MAP.get(fund_id) is not None else "   <-- NOT IN THE MAP; this is very likely the problem"))

    rows = apply_diagnosis_corrections(episode, set(), None)
    row = pd.Series(episode.to_row())
    row["new_diag_poa"] = rows_to_diag_poa(rows)
    row = derive_grouper_version(pd.DataFrame([row])).iloc[0]
    payload = _build_au_grouper_payload(row, use_alt_diag=True, new_diag_col="new_diag_poa")

    if a.show_values:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print({k: type(v).__name__ for k, v in payload.items()})
    nulls = [k for k, v in payload.items() if v is None]
    nulls += [f"diagnosis {d.get('code')}: poa" for d in payload.get("diagnosis", []) if d.get("poa") is None]
    print("None fields:", nulls or "none (compare against a known-good admission's payload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
