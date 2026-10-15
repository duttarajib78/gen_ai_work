"""Hospital-episode record shape (``Episode``) and the claims-schema vocabulary used in rule logic.

The ten core field names are adopted from the user's rule template (they were never supplied as a data
dictionary - confirm them against the real schema and update this module and ``CLAIMS_SCHEMA_FIELDS``
together). The grouper-only fields are only read by grouper adapters, never by the rule interpreters.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_ARRAY_LEN = 200

CLAIMS_SCHEMA_FIELDS: dict[str, tuple[str, str]] = {
    "diagnosis_code_array": ("list of ICD-10-AM codes", "All diagnosis codes for the episode in sequence order; index 0 is the principal diagnosis."),
    "procedure_code_array": ("list of ACHI codes", "All procedure codes (format NNNNN-NN)."),
    "medical_claims_array": ("list of MBS item numbers", "MBS items billed during the episode."),
    "principal_diagnosis_code": ("ICD-10-AM code", "Principal diagnosis; always equal to diagnosis_code_array[0]."),
    "condition_onset_array": ("list of flags", "Condition-onset flag per diagnosis, aligned by index with diagnosis_code_array ('1' = onset during the episode, '2' = not)."),
    "hours_of_mechanical_ventilation": ("number", "Total hours of continuous ventilatory support."),
    "minutes_operating_theatre": ("number", "Minutes spent in the operating theatre."),
    "prosthesis_benefit": ("number", "Prosthesis benefit amount claimed for the episode."),
    "member_sex": ("string", "Member sex code, for example 'M' or 'F'."),
    "member_age": ("number", "Member age in years at admission."),
}
SCALAR_FIELDS: dict[str, str] = {
    "hours_of_mechanical_ventilation": "number",
    "minutes_operating_theatre": "number",
    "prosthesis_benefit": "number",
    "member_age": "number",
    "member_sex": "string",
}
GROUPER_ONLY_FIELDS = (
    "fund_id", "provider_id", "member_dob", "admit_date", "discharge_date", "los",
    "mode_of_separation", "original_drg", "drg_ver_grouper", "urgency_of_admission", "infant_weight",
)
_ARRAY_FIELDS = ("diagnosis_code_array", "procedure_code_array", "medical_claims_array", "condition_onset_array")
_NUMBER_FIELDS = ("hours_of_mechanical_ventilation", "minutes_operating_theatre", "prosthesis_benefit", "member_age")


@dataclass
class Episode:
    episode_id: str = ""
    diagnosis_code_array: list[str] = field(default_factory=list)
    procedure_code_array: list[str] = field(default_factory=list)
    medical_claims_array: list[str] = field(default_factory=list)
    principal_diagnosis_code: str | None = None
    condition_onset_array: list[str] = field(default_factory=list)
    hours_of_mechanical_ventilation: float | None = None
    minutes_operating_theatre: float | None = None
    prosthesis_benefit: float | None = None
    member_sex: str | None = None
    member_age: float | None = None
    # grouper-only
    fund_id: str | None = None
    provider_id: str | None = None
    member_dob: str | None = None
    admit_date: str | None = None
    discharge_date: str | None = None
    los: float | None = None
    mode_of_separation: str | None = None
    original_drg: str | None = None
    drg_ver_grouper: str | None = None
    urgency_of_admission: str | None = None
    infant_weight: float | None = None

    def __post_init__(self) -> None:
        self.normalize_principal()

    def normalize_principal(self) -> None:
        """Keep the invariant: principal diagnosis is index 0 of the diagnosis array."""
        if self.principal_diagnosis_code and self.diagnosis_code_array:
            if self.diagnosis_code_array[0] != self.principal_diagnosis_code:
                if self.principal_diagnosis_code in self.diagnosis_code_array:
                    idx = self.diagnosis_code_array.index(self.principal_diagnosis_code)
                    self.diagnosis_code_array.insert(0, self.diagnosis_code_array.pop(idx))
                    if len(self.condition_onset_array) == len(self.diagnosis_code_array):
                        self.condition_onset_array.insert(0, self.condition_onset_array.pop(idx))
        self.principal_diagnosis_code = self.diagnosis_code_array[0] if self.diagnosis_code_array else self.principal_diagnosis_code

    def copy(self) -> "Episode":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any, *, strict: bool = True) -> "Episode":
        """Validate and coerce an untrusted mapping into an Episode (raises ValueError with every problem)."""
        if not isinstance(raw, dict):
            raise ValueError("episode must be an object")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        errors: list[str] = []
        if strict:
            unknown = sorted(set(raw) - known)
            if unknown:
                errors.append(f"unknown episode field(s): {unknown}")
        kwargs: dict[str, Any] = {}
        for key in known:
            if key not in raw or raw[key] is None:
                continue
            value = raw[key]
            if key in _ARRAY_FIELDS:
                if not isinstance(value, list) or len(value) > MAX_ARRAY_LEN or not all(isinstance(v, (str, int)) for v in value):
                    errors.append(f"{key} must be a list of at most {MAX_ARRAY_LEN} strings")
                    continue
                kwargs[key] = [str(v).strip() for v in value]
            elif key in _NUMBER_FIELDS or key in ("los", "infant_weight"):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"{key} must be a number")
                    continue
                kwargs[key] = float(value)
            else:
                if not isinstance(value, (str, int, float)) or len(str(value)) > 200:
                    errors.append(f"{key} must be a short string")
                    continue
                kwargs[key] = str(value).strip()
        onset, diags = kwargs.get("condition_onset_array"), kwargs.get("diagnosis_code_array", [])
        if onset and len(onset) != len(diags):
            errors.append("condition_onset_array must be the same length as diagnosis_code_array")
        if errors:
            raise ValueError("; ".join(errors))
        return cls(**kwargs)
