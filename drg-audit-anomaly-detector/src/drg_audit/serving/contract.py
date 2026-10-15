"""Request / response contract for the real-time endpoint.

Everything arriving over HTTP is untrusted. Rules enforced here:

* JSON only (no pickle / npy / csv), strict UTF-8, ``NaN`` / ``Infinity`` literals refused
* hard size bounds (bytes, records, codes per array, code length, numeric ranges) - reject, never clamp
* unknown top-level, option and record fields are refused
* request-level problems (bad JSON, too large, wrong shape) fail the whole call with 4xx
* record-level problems yield a per-record ``rejected`` result so one bad record cannot block a batch
* error messages carry field *names* and stable codes only - never submitted values
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from drg_audit.errors import (
    NotAcceptableError,
    PayloadTooLargeError,
    RequestValidationError,
    UnsupportedMediaTypeError,
)
from drg_audit.features import DEFAULT_AGE, normalize_code
from drg_audit.settings import RequestLimits
from drg_audit.schemas import InpatientRecord

JSON_MEDIA_TYPE = "application/json"
_ACCEPT_OK = frozenset({JSON_MEDIA_TYPE, "*/*", "application/*"})

_ID_RE = re.compile(r"[A-Za-z0-9_\-]{1,64}")
_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-:/]*")
_VERSION_RE = re.compile(r"[A-Za-z0-9._\-]{1,16}")
_DRG_RE = re.compile(r"[A-Za-z0-9]{0,8}")

TOP_LEVEL_KEYS = frozenset({"records", "options"})
OPTION_KEYS = frozenset({"confidence_threshold"})
REQUIRED_RECORD_KEYS = ("admission_no", "diagnosis_code_array", "drg_vers", "drg", "los")
RECORD_KEYS = frozenset(
    {
        *REQUIRED_RECORD_KEYS,
        "procedure_code_array",
        "medical_claims_array",
        "minutes_operating_theatre",
        "hours_of_mechanical_ventilation",
        "member_age",
        "drg_benefit",
    }
)


@dataclass(frozen=True)
class ParsedRecord:
    index: int
    record: InpatientRecord
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RejectedRecord:
    index: int
    admission_no: str | None
    reason: str
    field: str | None = None

    @property
    def reason_text(self) -> str:
        return f"{self.reason}:{self.field}" if self.field else self.reason


@dataclass(frozen=True)
class ScoreRequest:
    items: tuple[ParsedRecord | RejectedRecord, ...]
    confidence_threshold: float | None = None

    @property
    def valid(self) -> list[ParsedRecord]:
        return [i for i in self.items if isinstance(i, ParsedRecord)]


class _FieldError(Exception):
    def __init__(self, code: str, field: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.field = field


# --------------------------------------------------------------------------------------
# request parsing
# --------------------------------------------------------------------------------------
def parse_request(body: bytes | bytearray | str, content_type: str | None, limits: RequestLimits) -> ScoreRequest:
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type != JSON_MEDIA_TYPE:
        raise UnsupportedMediaTypeError("Content-Type must be application/json")

    size = len(body) if isinstance(body, (bytes, bytearray)) else len(body.encode("utf-8", errors="replace"))
    if size > limits.max_payload_bytes:
        raise PayloadTooLargeError(f"Payload exceeds the {limits.max_payload_bytes}-byte limit")
    if size == 0:
        raise RequestValidationError("Request body is empty", code="empty_body")

    if isinstance(body, (bytes, bytearray)):
        try:
            text = bytes(body).decode("utf-8")
        except UnicodeDecodeError:
            raise RequestValidationError("Body is not valid UTF-8", code="invalid_encoding") from None
    else:
        text = body

    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise RequestValidationError("Body is not valid JSON", code="invalid_json") from None

    if not isinstance(payload, dict):
        raise RequestValidationError("Top-level JSON value must be an object", code="invalid_shape")
    if set(payload) - TOP_LEVEL_KEYS:
        raise RequestValidationError("Unknown top-level fields; allowed: records, options", code="unknown_fields")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise RequestValidationError("'records' must be a non-empty array", code="invalid_records")
    if len(raw_records) > limits.max_records:
        raise PayloadTooLargeError(f"'records' exceeds the limit of {limits.max_records} records per request")

    threshold = _parse_options(payload.get("options"), limits)

    items: list[ParsedRecord | RejectedRecord] = []
    for index, raw in enumerate(raw_records):
        try:
            items.append(_parse_record(raw, index, limits))
        except _FieldError as err:
            items.append(RejectedRecord(index, _safe_admission_no(raw), err.code, err.field))
    return ScoreRequest(items=tuple(items), confidence_threshold=threshold)


def negotiate_accept(accept: str | None) -> str:
    if not accept:
        return JSON_MEDIA_TYPE
    offered = {part.split(";")[0].strip().lower() for part in accept.split(",")}
    if offered & _ACCEPT_OK:
        return JSON_MEDIA_TYPE
    raise NotAcceptableError("Only application/json responses are supported")


def serialize_response(payload: dict[str, Any]) -> str:
    """Compact JSON. ``allow_nan=False`` guarantees the output is standards-compliant JSON."""
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _reject_constant(_name: str) -> None:
    raise ValueError("non-finite JSON constant")


def _parse_options(raw: Any, limits: RequestLimits) -> float | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RequestValidationError("'options' must be an object", code="invalid_options")
    if set(raw) - OPTION_KEYS:
        raise RequestValidationError("Unknown option; allowed: confidence_threshold", code="invalid_options")
    value = raw.get("confidence_threshold")
    if value is None:
        return None
    try:
        return _number(value, "confidence_threshold", limits.min_confidence_threshold, 1.0)
    except _FieldError:
        raise RequestValidationError(
            f"options.confidence_threshold must be a number in [{limits.min_confidence_threshold}, 1.0]",
            code="invalid_options",
        ) from None


def _safe_admission_no(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("admission_no")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip() if isinstance(value, str) else str(value)
    return text if len(text) <= 64 and _ID_RE.fullmatch(text) else None


def _parse_record(raw: Any, index: int, limits: RequestLimits) -> ParsedRecord:
    if not isinstance(raw, dict):
        raise _FieldError("not_an_object")
    if set(raw) - RECORD_KEYS:
        raise _FieldError("unknown_fields")
    for key in REQUIRED_RECORD_KEYS:
        if raw.get(key) is None:
            raise _FieldError("missing_field", key)

    warnings: list[str] = []
    admission_no = _identifier(raw["admission_no"], "admission_no")
    diagnoses = _code_list(raw["diagnosis_code_array"], "diagnosis_code_array", limits, min_items=1)
    procedures = _code_list(raw.get("procedure_code_array"), "procedure_code_array", limits)
    claims = _code_list(raw.get("medical_claims_array"), "medical_claims_array", limits)
    drg_vers = _string(raw["drg_vers"], "drg_vers", _VERSION_RE)
    drg = _string(raw["drg"], "drg", _DRG_RE)
    los = _integer(raw["los"], "los", 0, limits.max_los_days)
    theatre = _optional_number(raw.get("minutes_operating_theatre"), "minutes_operating_theatre",
                               0.0, limits.max_theatre_minutes, 0.0)
    ventilation = _optional_number(raw.get("hours_of_mechanical_ventilation"), "hours_of_mechanical_ventilation",
                                   0.0, limits.max_ventilation_hours, 0.0)
    if raw.get("member_age") is None:
        age = DEFAULT_AGE
        warnings.append("member_age_missing_defaulted")
    else:
        age = _number(raw["member_age"], "member_age", 0.0, limits.max_age_years)
    if raw.get("drg_benefit") is None:
        benefit = 0.0
        warnings.append("drg_benefit_missing")
    else:
        benefit = _number(raw["drg_benefit"], "drg_benefit", -limits.max_abs_benefit, limits.max_abs_benefit)

    record = InpatientRecord(
        admission_no=admission_no,
        diagnosis_code_array=diagnoses,
        procedure_code_array=procedures,
        medical_claims_array=claims,
        drg_vers=drg_vers,
        drg=drg,
        los=los,
        minutes_operating_theatre=theatre,
        hours_of_mechanical_ventilation=ventilation,
        member_age=age,
        drg_benefit=benefit,
    )
    return ParsedRecord(index=index, record=record, warnings=tuple(warnings))


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise _FieldError("invalid_field", name)
    text = value.strip() if isinstance(value, str) else str(value)
    if len(text) > 64 or not _ID_RE.fullmatch(text):
        raise _FieldError("invalid_field", name)
    return text


def _string(value: Any, name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str):
        raise _FieldError("invalid_field", name)
    text = value.strip()
    if not pattern.fullmatch(text):
        raise _FieldError("invalid_field", name)
    return text


def _code_list(value: Any, name: str, limits: RequestLimits, min_items: int = 0) -> list[str]:
    if value is None:
        if min_items:
            raise _FieldError("missing_field", name)
        return []
    if not isinstance(value, list) or not min_items <= len(value) <= limits.max_codes_per_array:
        raise _FieldError("invalid_field", name)
    codes: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise _FieldError("invalid_field", name)
        code = normalize_code(item)
        if not code or len(code) > limits.max_code_length or not _CODE_RE.fullmatch(code):
            raise _FieldError("invalid_field", name)
        codes.append(code)
    return codes


def _number(value: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _FieldError("invalid_field", name)
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise _FieldError("invalid_field", name) from None
    if not math.isfinite(number) or number < lo or number > hi:
        raise _FieldError("invalid_field", name)
    return number


def _optional_number(value: Any, name: str, lo: float, hi: float, default: float) -> float:
    return default if value is None else _number(value, name, lo, hi)


def _integer(value: Any, name: str, lo: int, hi: int) -> int:
    number = _number(value, name, float(lo), float(hi))
    if number != int(number):
        raise _FieldError("invalid_field", name)
    return int(number)
