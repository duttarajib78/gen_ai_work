"""Request contract and validation.

Everything arriving at ``/invocations`` is treated as hostile until it passes
this module:

* the envelope is strict (unknown top-level or option keys are rejected);
* inside ``episode`` only an allowlist of fields is read, anything else is
  dropped and counted in ``warnings`` (the grouper then receives the least data
  it needs);
* every code is matched against a tight character allowlist, which is what keeps
  newlines, quotes, and instruction-like text out of the LLM prompt;
* array lengths, string lengths, and numeric ranges are bounded;
* error messages never echo submitted values.

Parsing is hand-written (no third-party schema library) to keep the runtime
dependency surface small and the behaviour fully explicit.
"""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Optional

from .config import MBS_ARRAY_FIELDS, Settings
from .errors import RequestValidationError
from .textutil import clean_line

MODES = ("full", "verdicts_only")

CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,15}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$")
_SMALL_TOKEN_RE = re.compile(r"^[A-Za-z0-9 _\-/]{1,40}$")
_DRG_RE = re.compile(r"^[A-Za-z0-9]{3,8}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9.]{1,8}$")
_SERVICE_CLASS_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_PROVIDER_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")
_SEX_RE = re.compile(r"^[A-Za-z]{1,8}$")
_SHORT_RE = re.compile(r"^[A-Za-z0-9_\-]{1,8}$")
_ONSET_RE = re.compile(r"^[A-Za-z0-9 _\-]{1,16}$")

_MAX_ISSUES = 25

_ENVELOPE_KEYS = {"request_id", "episode", "options"}
_OPTION_KEYS = {"mode", "include_episode_text", "stream", "deadline_s", "mbs_array_field"}

# Fields that are read from ``episode``. Modelled explicitly:
_MODELLED = {
    "admission_no", "diagnosis_code_array", "procedure_code_array", "procedure_date_array",
    "medical_claims_array", "hospital_claims_array", "dte_admitted", "dte_separated",
    "diagnosis_poa_array", "condition_onset_array", "dx_code_idx_array",
}
# Passed through (validated) to the grouper only; never shown to the LLM.
_EXTRA_SPECS: dict[str, tuple] = {
    "fund_id": ("int", 0, 10**9),
    "provider_id": ("str_re", _PROVIDER_RE),
    "drg": ("str_re", _DRG_RE),
    "drg_vers": ("str_re", _VERSION_RE),
    "los": ("int", 0, 36500),
    "hos_benefit": ("number", 0.0, 1e12),
    "hospital_claims_seqno_array": ("int_list", 0, 10**15, 100),
    "hospital_claims_service_class_code_array": ("str_list_re", _SERVICE_CLASS_RE, 100),
    "urgency_of_admission": ("str_re", _SMALL_TOKEN_RE),
    "mode_of_separation": ("int_or_str", _SHORT_RE),
    "hours_of_mechanical_ventilation": ("number", 0.0, 1e6),
    "minutes_operating_theatre": ("number", 0.0, 1e7),
    "infant_weight": ("number", 0.0, 1e5),
    "source_of_referral": ("int_or_str", _SHORT_RE),
    "member_age": ("int", 0, 150),
    "member_sex": ("str_re", _SEX_RE),
    "member_dob": ("date",),
}
EPISODE_FIELDS = frozenset(_MODELLED | set(_EXTRA_SPECS))
GROUPER_REQUIRED_FIELDS = ("drg", "hos_benefit", "dte_admitted", "dte_separated")


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    admission_no: int
    diagnosis_codes: tuple[str, ...]
    procedure_codes: tuple[str, ...] = ()
    procedure_dates: tuple[Optional[datetime], ...] = ()
    medical_claims: tuple[str, ...] = ()
    hospital_claims: tuple[str, ...] = ()
    admit_dt: Optional[datetime] = None
    separate_dt: Optional[datetime] = None
    diagnosis_poa: Optional[tuple] = None
    condition_onset: Optional[tuple] = None
    dx_code_idx: Optional[tuple] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def mbs_codes(self, array_field: str) -> tuple[str, ...]:
        return self.hospital_claims if array_field == "hospital_claims_array" else self.medical_claims

    def to_row(self) -> dict:
        """Dict shaped like one admission-summary row, for the grouper adapter."""
        row: dict[str, Any] = {
            "admission_no": self.admission_no,
            "diagnosis_code_array": list(self.diagnosis_codes),
            "procedure_code_array": list(self.procedure_codes),
            "procedure_date_array": [d.date().isoformat() if d else None for d in self.procedure_dates],
            "medical_claims_array": list(self.medical_claims),
            "hospital_claims_array": list(self.hospital_claims),
            "hospital_claims_seqno_array": [],
            "hospital_claims_service_class_code_array": [],
            "dte_admitted": self.admit_dt,
            "dte_separated": self.separate_dt,
        }
        if self.diagnosis_poa is not None:
            row["diagnosis_poa_array"] = list(self.diagnosis_poa)
        if self.condition_onset is not None:
            row["condition_onset_array"] = list(self.condition_onset)
        if self.dx_code_idx is not None:
            row["dx_code_idx_array"] = list(self.dx_code_idx)
        for key, value in self.extras.items():
            row[key] = list(value) if isinstance(value, tuple) else value
        return row


@dataclass(frozen=True)
class AuditRequest:
    request_id: str
    episode: Episode
    mode: str = "full"
    include_episode_text: bool = False
    stream: bool = False
    deadline_s: Optional[float] = None
    mbs_array_field: str = "medical_claims_array"
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
class _Issues:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, path: str, message: str) -> None:
        if len(self.items) < _MAX_ISSUES:
            self.items.append({"path": clean_line(path, 80), "message": message})

    def __bool__(self) -> bool:
        return bool(self.items)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_int(path: str, value: Any, issues: _Issues, lo: int, hi: int) -> Optional[int]:
    if _is_int(value) and lo <= value <= hi:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,18}", value.strip()):
        parsed = int(value.strip())
        if lo <= parsed <= hi:
            return parsed
    issues.add(path, f"must be an integer between {lo} and {hi}")
    return None


def _parse_number(path: str, value: Any, issues: _Issues, lo: float, hi: float) -> Optional[float]:
    if _is_number(value) and lo <= value <= hi:
        return float(value) if isinstance(value, float) else value
    issues.add(path, f"must be a finite number between {lo:g} and {hi:g}")
    return None


def _parse_datetime(path: str, value: Any, issues: _Issues) -> Optional[datetime]:
    if not isinstance(value, str) or not (4 <= len(value) <= 40):
        issues.add(path, "must be an ISO-8601 date or datetime string")
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        issues.add(path, "must be an ISO-8601 date or datetime string")
        return None
    # Keep local wall-clock time: converting zones could shift a calendar date.
    return parsed.replace(tzinfo=None)


def _parse_code_list(
    path: str, value: Any, issues: _Issues, *, max_items: int, required: bool = False
) -> tuple[str, ...]:
    if value is None:
        if required:
            issues.add(path, "is required")
        return ()
    if not isinstance(value, list):
        issues.add(path, "must be an array")
        return ()
    if required and not value:
        issues.add(path, "must contain at least one code")
        return ()
    if len(value) > max_items:
        issues.add(path, f"has too many entries (maximum {max_items})")
        return ()
    out: list[str] = []
    for i, item in enumerate(value):
        code = item.strip().upper() if isinstance(item, str) else None
        if code is None or not CODE_RE.match(code):
            issues.add(f"{path}[{i}]", "is not a valid code (letters, digits, '.', '-' only; 1-16 characters)")
            return ()
        out.append(code)
    return tuple(out)


def _parse_poa_entry(path: str, entry: Any, issues: _Issues) -> Any:
    def scalar(v: Any, p: str) -> Any:
        if v is None or _is_int(v):
            return v
        if isinstance(v, str) and _SHORT_RE.match(v):
            return v
        issues.add(p, "must be null, an integer, or a short string")
        return None

    if isinstance(entry, dict):
        unknown = set(entry) - {"code", "poa"}
        if unknown:
            issues.add(path, "may contain only 'code' and 'poa'")
        code = entry.get("code")
        if code is not None:
            norm = code.strip().upper() if isinstance(code, str) else None
            if norm is None or not CODE_RE.match(norm):
                issues.add(f"{path}.code", "is not a valid code")
            code = norm
        return {"code": code, "poa": scalar(entry.get("poa"), f"{path}.poa")}
    return scalar(entry, path)


def _parse_paired(
    path: str, value: Any, n: int, issues: _Issues, item_parser
) -> Optional[tuple]:
    if value is None:
        return None
    if not isinstance(value, list):
        issues.add(path, "must be an array")
        return None
    if len(value) != n:
        issues.add(path, "must have the same length as diagnosis_code_array")
        return None
    return tuple(item_parser(f"{path}[{i}]", item, issues) for i, item in enumerate(value))


def _onset_item(path: str, item: Any, issues: _Issues) -> Any:
    if item is None or (isinstance(item, str) and _ONSET_RE.match(item)):
        return item
    issues.add(path, "must be null or a short string")
    return None


def _idx_item(path: str, item: Any, issues: _Issues) -> Any:
    if item is None or (_is_int(item) and 0 <= item <= 10**6):
        return item
    issues.add(path, "must be null or a non-negative integer")
    return None


def _parse_extra(name: str, value: Any, issues: _Issues) -> Any:
    spec = _EXTRA_SPECS[name]
    path = f"episode.{name}"
    kind = spec[0]
    if value is None:
        return None
    if kind == "int":
        return _parse_int(path, value, issues, spec[1], spec[2])
    if kind == "number":
        return _parse_number(path, value, issues, spec[1], spec[2])
    if kind == "str_re":
        if isinstance(value, str) and spec[1].match(value):
            return value
        issues.add(path, "has an unsupported format")
        return None
    if kind == "int_or_str":
        if _is_int(value) and 0 <= value <= 10**9:
            return value
        if isinstance(value, str) and spec[1].match(value):
            return value
        issues.add(path, "must be a small integer or short string")
        return None
    if kind == "date":
        return _parse_datetime(path, value, issues)
    if kind in ("int_list", "str_list_re"):
        max_items = spec[-1]
        if not isinstance(value, list) or len(value) > max_items:
            issues.add(path, f"must be an array of at most {max_items} items")
            return None
        out = []
        for i, item in enumerate(value):
            if kind == "int_list":
                parsed = _parse_int(f"{path}[{i}]", item, issues, spec[1], spec[2])
            else:
                parsed = item if isinstance(item, str) and spec[1].match(item) else None
                if parsed is None:
                    issues.add(f"{path}[{i}]", "has an unsupported format")
            out.append(parsed)
        return tuple(out)
    raise AssertionError(f"unknown spec kind {kind!r}")  # programming error, not input error


def _parse_episode(raw: Any, settings: Settings, issues: _Issues, mode: str) -> tuple[Optional[Episode], int]:
    if not isinstance(raw, dict):
        issues.add("episode", "must be an object")
        return None, 0

    ignored = len([k for k in raw if k not in EPISODE_FIELDS])

    if "admission_no" not in raw:
        issues.add("episode.admission_no", "is required")
        admission_no = None
    else:
        admission_no = _parse_int("episode.admission_no", raw["admission_no"], issues, 1, 10**18)

    diagnoses = _parse_code_list(
        "episode.diagnosis_code_array", raw.get("diagnosis_code_array"), issues,
        max_items=settings.max_diagnoses, required=True,
    )
    procedures = _parse_code_list(
        "episode.procedure_code_array", raw.get("procedure_code_array"), issues, max_items=settings.max_procedures
    )
    medical = _parse_code_list(
        "episode.medical_claims_array", raw.get("medical_claims_array"), issues, max_items=settings.max_mbs_items
    )
    hospital = _parse_code_list(
        "episode.hospital_claims_array", raw.get("hospital_claims_array"), issues, max_items=settings.max_mbs_items
    )

    proc_dates: tuple = ()
    raw_dates = raw.get("procedure_date_array")
    if raw_dates is not None:
        if not isinstance(raw_dates, list) or len(raw_dates) > settings.max_procedures:
            issues.add("episode.procedure_date_array", "must be an array with at most one entry per procedure")
        else:
            proc_dates = tuple(
                None if d is None else _parse_datetime(f"episode.procedure_date_array[{i}]", d, issues)
                for i, d in enumerate(raw_dates)
            )

    admit = _parse_datetime("episode.dte_admitted", raw["dte_admitted"], issues) if raw.get("dte_admitted") is not None else None
    separate = (
        _parse_datetime("episode.dte_separated", raw["dte_separated"], issues)
        if raw.get("dte_separated") is not None else None
    )

    n = len(diagnoses)
    poa = _parse_paired("episode.diagnosis_poa_array", raw.get("diagnosis_poa_array"), n, issues, _parse_poa_entry) if n else None
    onset = _parse_paired("episode.condition_onset_array", raw.get("condition_onset_array"), n, issues, _onset_item) if n else None
    idx = _parse_paired("episode.dx_code_idx_array", raw.get("dx_code_idx_array"), n, issues, _idx_item) if n else None

    extras: dict[str, Any] = {}
    for name in _EXTRA_SPECS:
        if name in raw:
            parsed = _parse_extra(name, raw[name], issues)
            if parsed is not None:
                extras[name] = parsed

    if mode == "full":
        present = {
            "drg": "drg" in extras,
            "hos_benefit": "hos_benefit" in extras,
            "dte_admitted": admit is not None,
            "dte_separated": separate is not None,
        }
        for name in GROUPER_REQUIRED_FIELDS:
            if not present[name]:
                issues.add(f"episode.{name}", "is required when options.mode is 'full'")

    if issues or admission_no is None:
        return None, ignored
    return (
        Episode(
            admission_no=admission_no, diagnosis_codes=diagnoses, procedure_codes=procedures,
            procedure_dates=proc_dates, medical_claims=medical, hospital_claims=hospital,
            admit_dt=admit, separate_dt=separate, diagnosis_poa=poa, condition_onset=onset,
            dx_code_idx=idx, extras=extras,
        ),
        ignored,
    )


def _parse_options(raw: Any, settings: Settings, issues: _Issues) -> dict:
    defaults = {
        "mode": "full", "include_episode_text": False, "stream": False,
        "deadline_s": None, "mbs_array_field": settings.mbs_array_field,
    }
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        issues.add("options", "must be an object")
        return defaults
    for key in raw:
        if key not in _OPTION_KEYS:
            issues.add(f"options.{key}", "unknown option")
    out = dict(defaults)
    if "mode" in raw:
        if raw["mode"] in MODES:
            out["mode"] = raw["mode"]
        else:
            issues.add("options.mode", f"must be one of {list(MODES)}")
    for flag in ("include_episode_text", "stream"):
        if flag in raw:
            if isinstance(raw[flag], bool):
                out[flag] = raw[flag]
            else:
                issues.add(f"options.{flag}", "must be true or false")
    if raw.get("deadline_s") is not None:
        out["deadline_s"] = _parse_number("options.deadline_s", raw["deadline_s"], issues, 5.0, settings.stream_deadline_s)
    if "mbs_array_field" in raw:
        if raw["mbs_array_field"] in MBS_ARRAY_FIELDS:
            out["mbs_array_field"] = raw["mbs_array_field"]
        else:
            issues.add("options.mbs_array_field", f"must be one of {list(MBS_ARRAY_FIELDS)}")
    return out


def parse_request(payload: Any, settings: Settings) -> AuditRequest:
    issues = _Issues()
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object", details=[{"path": "$", "message": "must be an object"}])

    for key in payload:
        if key not in _ENVELOPE_KEYS:
            issues.add(str(key), "unknown top-level field")

    request_id = payload.get("request_id")
    if request_id is None:
        request_id = uuid.uuid4().hex
    elif not (isinstance(request_id, str) and REQUEST_ID_RE.match(request_id)):
        issues.add("request_id", "must match [A-Za-z0-9._-]{1,64}")
        request_id = uuid.uuid4().hex

    options = _parse_options(payload.get("options"), settings, issues)
    episode, ignored = _parse_episode(payload.get("episode"), settings, issues, options["mode"])

    if issues or episode is None:
        raise RequestValidationError("request failed validation", details=issues.items)

    warnings = (f"ignored_unknown_episode_fields:{ignored}",) if ignored else ()
    return AuditRequest(
        request_id=request_id,
        episode=episode,
        mode=options["mode"],
        include_episode_text=options["include_episode_text"],
        stream=options["stream"],
        deadline_s=options["deadline_s"],
        mbs_array_field=options["mbs_array_field"],
        warnings=warnings,
    )
