"""Turn admission-summary rows (parquet / feather / csv / jsonl) into endpoint payloads.

Hospital extracts often store arrays as strings (``"['3208400']"``) or as numpy
arrays whose text form has no commas. ``coerce_to_list`` handles every shape the
validated prototype met in real data, including the numpy-repr case where
``ast.literal_eval`` would silently glue adjacent quoted codes into one string.
Used by the benchmark, the batch client, and the diagnostics; never by the endpoint.
"""
from __future__ import annotations

import ast
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from audit_agent.schemas import EPISODE_FIELDS

ARRAY_FIELDS = frozenset({
    "diagnosis_code_array", "diagnosis_poa_array", "condition_onset_array", "dx_code_idx_array",
    "procedure_code_array", "procedure_date_array", "medical_claims_array", "hospital_claims_array",
    "hospital_claims_seqno_array", "hospital_claims_service_class_code_array",
})


def coerce_to_list(cell: Any) -> list:
    if cell is None:
        return []
    if isinstance(cell, float):
        return []
    if isinstance(cell, (list, tuple, set)):
        return list(cell)
    if hasattr(cell, "tolist") and hasattr(cell, "__len__"):
        # numpy arrays (and pandas arrays): handled before any str() parsing, because
        # str(np.array(['J211','N179'])) is space-separated and ast.literal_eval would
        # concatenate the adjacent quoted strings into one code without raising.
        return list(cell.tolist())
    s = str(cell).strip()
    if not s or s in ("[]", "()", "{}", "nan", "NaN", "None"):
        return []
    if (s[0], s[-1]) in {("[", "]"), ("(", ")"), ("{", "}")} or s[0] in "\"'":
        inner = s[1:-1] if s[0] in "[({" else s
        # numpy's text form has no commas between elements; detect it before the parsers can mangle it
        if "," not in inner and re.search(r"\s", inner.strip()):
            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
            if quoted:
                return [a or b for a, b in quoted]
            tokens = inner.split()
            if len(tokens) > 1:
                return tokens
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(s)
            except (ValueError, SyntaxError, TypeError):
                continue
            return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    return re.split(r"[;,|/]+", s)


def jsonable(value: Any) -> Any:
    """Plain-JSON form of a pandas/numpy value; missing values become None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):  # includes pandas.Timestamp
        return None if value != value else value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if value.__class__.__name__ in ("NaTType", "NAType"):
        return None
    return value


def row_to_payload(row: Mapping[str, Any], *, mode: str = "full", request_id: Optional[str] = None,
                   options: Optional[dict] = None) -> dict:
    episode: dict[str, Any] = {}
    for name in EPISODE_FIELDS:
        if name not in row:
            continue
        value = row[name]
        episode[name] = [jsonable(x) for x in coerce_to_list(value)] if name in ARRAY_FIELDS else jsonable(value)
    if episode.get("admission_no") is not None:
        episode["admission_no"] = int(episode["admission_no"])
    opts = {"mode": mode, **(options or {})}
    return {"request_id": request_id or f"adm-{episode.get('admission_no', 'unknown')}", "episode": episode,
            "options": opts}


def load_table(path: Path):
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported input type {suffix!r}; use .parquet, .feather, .csv, or .jsonl")
