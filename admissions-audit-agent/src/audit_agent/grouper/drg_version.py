"""DRG version bucket used to join cost weights.

Exact mirror of ``derive_grouper_version`` in the grouper harness
(drg_cost_weight.py), which must stay in step with the mapping inside
``au_grouper``. Implemented per value in pure Python (the audit only ever
regroups one admission at a time) with a DataFrame wrapper for callers that
expect the original signature.
"""
from __future__ import annotations

import re
from typing import Any

_V60 = {"6X", "6x", "60", "61", "62", "6.0"}
_V100 = {"10", "100"}
_V80 = {"80", "8x", "8X"}
_V90 = {"90", "9x", "92"}
_PASSTHROUGH = {"51", "42", "70", "52", "41", "50", "110", "120"}


def derive_grouper_version_value(drg_vers: Any, service_class_codes: Any) -> str:
    missing = drg_vers is None or (isinstance(drg_vers, float) and drg_vers != drg_vers)
    text = "60" if missing else str(drg_vers)
    text = re.sub(r"^0+", "", text.strip())
    if "CWO" in str(service_class_codes):
        return "100"
    if text in _V60:
        return "60"
    if text in _V100:
        return "100"
    if text in _V80:
        return "80"
    if text in _V90:
        return "90"
    if text in _PASSTHROUGH:
        return text
    return "60"


def derive_grouper_version(df):
    """DataFrame wrapper: adds ``drg_ver_grouper`` and normalises ``drg_vers``."""
    df = df.copy()
    df["drg_vers"] = df["drg_vers"].map(
        lambda v: re.sub(r"^0+", "", str("60" if v is None or (isinstance(v, float) and v != v) else v).strip())
    )
    df["drg_ver_grouper"] = [
        derive_grouper_version_value(v, s)
        for v, s in zip(df["drg_vers"], df["hospital_claims_service_class_code_array"])
    ]
    return df
