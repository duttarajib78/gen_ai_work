"""Apply diagnosis corrections to the raw, position-aligned diagnosis arrays.

REMOVAL and RESEQUENCING are different operations:

* a failed ADx is removed outright (the evidence says it should not be coded);
* a failed PDx is kept but demoted to the end of the list while the chosen
  replacement is promoted to the front, so a real condition can still
  contribute a CC/MCC to the regrouped DRG.

Every paired array (POA, onset, index) is carried with its diagnosis; a length
mismatch is refused rather than reordered positionally. A repeated code is
matched by code string, so removal affects every position holding it, while a
resequence that would be ambiguous (a repeated old-PDx or new-principal code)
is refused and routed to a human coder.
"""
from __future__ import annotations

import math
from typing import Any, Collection, Optional

from ..errors import CorrectionError
from ..schemas import Episode

PAIRED_ARRAYS = ("diagnosis_poa_array", "condition_onset_array", "dx_code_idx_array")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _extract_poa(entry: Any, expected_code: str) -> Any:
    """Real data carries POA entries as {"code", "poa"} objects; flat scalars are also accepted.
    The embedded code must match this position's diagnosis code."""
    if isinstance(entry, dict):
        embedded = entry.get("code")
        if embedded is not None and str(embedded).strip() != str(expected_code).strip():
            raise CorrectionError(
                "a diagnosis_poa_array entry does not match the diagnosis at its position; "
                "refusing to trust positional alignment"
            )
        return entry.get("poa")
    return entry


def apply_diagnosis_corrections(
    episode: Episode, removed_codes: Collection[str], pdx_resequence: Optional[tuple[str, str]] = None
) -> list[dict]:
    codes = list(episode.diagnosis_codes)
    n = len(codes)
    paired = {
        "diagnosis_poa_array": episode.diagnosis_poa,
        "condition_onset_array": episode.condition_onset,
        "dx_code_idx_array": episode.dx_code_idx,
    }
    present = {name: arr for name, arr in paired.items() if arr is not None}
    for name, arr in present.items():
        if len(arr) != n:
            raise CorrectionError(f"{name} does not match diagnosis_code_array in length; refusing to reorder")

    removed = set(removed_codes)
    rows: list[dict] = []
    for i, code in enumerate(codes):
        row: dict[str, Any] = {"code": code}
        for name, arr in present.items():
            row[name] = _extract_poa(arr[i], code) if name == "diagnosis_poa_array" else arr[i]
        rows.append(row)
    kept = [r for r in rows if r["code"] not in removed]

    if pdx_resequence is not None:
        old_code, new_code = pdx_resequence
        old_rows = [r for r in kept if r["code"] == old_code]
        new_rows = [r for r in kept if r["code"] == new_code]
        if len(old_rows) != 1:
            raise CorrectionError(
                f"expected exactly one entry for the old principal diagnosis, found {len(old_rows)}; "
                "refusing to resequence against an ambiguous or missing match"
            )
        if len(new_rows) != 1:
            raise CorrectionError(
                f"expected exactly one entry for the new principal diagnosis, found {len(new_rows)}; "
                "refusing to resequence against an ambiguous or missing match"
            )
        rest = [r for r in kept if r["code"] not in (old_code, new_code)]
        kept = new_rows + rest + old_rows  # new principal first, old PDx demoted to last

    if not kept:
        raise CorrectionError(
            "every diagnosis on this episode was flagged unsupported, so no principal diagnosis remains; "
            "this episode needs a human coder"
        )
    return kept


def rows_to_diag_poa(rows: list[dict]) -> list[dict]:
    """The {"code", "poa"} shape the grouper's alternative-diagnosis path expects."""
    return [
        {"code": r["code"], "poa": None if _is_missing(r.get("diagnosis_poa_array")) else r.get("diagnosis_poa_array")}
        for r in rows
    ]


def rows_to_full_diagnosis_fields(rows: list[dict]) -> dict:
    """Back to separate arrays in corrected order, for the audit trail."""
    out: dict[str, list] = {"diagnosis_code_array": [r["code"] for r in rows]}
    for name in PAIRED_ARRAYS:
        if all(name in r for r in rows):
            out[name] = [r[name] for r in rows]
    return out
