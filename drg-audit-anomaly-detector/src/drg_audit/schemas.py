"""Typed schemas shared by training, batch scoring and the endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from drg_audit.errors import ArtifactError, ConfigError

HEADS = ("version", "mdc", "base", "split")


@dataclass(slots=True)
class InpatientRecord:
    """One hospital admission. ``diagnosis_code_array[0]`` is the principal diagnosis (PDx)."""

    admission_no: str
    diagnosis_code_array: list[str]
    procedure_code_array: list[str]
    medical_claims_array: list[str]
    drg_vers: str
    drg: str
    los: int
    minutes_operating_theatre: float = 0.0
    hours_of_mechanical_ventilation: float = 0.0
    member_age: float = 50.0
    drg_benefit: float = 0.0


@dataclass(frozen=True)
class TokenizationSpec:
    """How a record becomes a token sequence. Stored in the model bundle so serving always matches training.

    ``standard-v1``        claims sorted, ``[SEP]`` always kept on truncation (used by all newly trained models).
    ``legacy-training-v1`` claims in source order, ``[SEP]`` dropped on truncation. This reproduces how the
                           prototype *trained* its weights, and is what converted legacy checkpoints default to.
    """

    max_len: int = 128
    sort_medical_claims: bool = True
    keep_sep_on_truncation: bool = True
    name: str = "standard-v1"

    def validate(self) -> None:
        if not 8 <= self.max_len <= 512:
            raise ConfigError("tokenization.max_len must be between 8 and 512")

    @classmethod
    def standard(cls, max_len: int = 128) -> TokenizationSpec:
        return cls(max_len=max_len)

    @classmethod
    def legacy_training(cls, max_len: int = 128) -> TokenizationSpec:
        return cls(max_len=max_len, sort_medical_claims=False, keep_sep_on_truncation=False,
                   name="legacy-training-v1")


@dataclass(frozen=True)
class LabelMaps:
    """Label-to-index maps for the version embedding and the three classification heads."""

    version: dict[str, int]
    mdc: dict[str, int]
    base: dict[str, int]
    split: dict[str, int]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LabelMaps:
        maps: dict[str, dict[str, int]] = {}
        for head in HEADS:
            raw = data.get(head)
            if not isinstance(raw, Mapping) or not raw:
                raise ArtifactError(f"label map '{head}' is missing or empty")
            parsed = {str(k): int(v) for k, v in raw.items()}
            if sorted(parsed.values()) != list(range(len(parsed))):
                raise ArtifactError(f"label map '{head}' must use contiguous indices 0..N-1")
            maps[head] = parsed
        return cls(**maps)

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {head: dict(getattr(self, head)) for head in HEADS}

    def sizes(self) -> dict[str, int]:
        return {head: len(getattr(self, head)) for head in HEADS}


@dataclass(slots=True)
class AuditResult:
    """Per-record scoring outcome. ``status`` is ``ok``, ``skipped`` (cannot be scored) or ``rejected`` (invalid input)."""

    admission_no: str | None
    status: str = "ok"
    reason: str | None = None
    assigned_drg: str | None = None
    predicted_mdc: str | None = None
    predicted_base_drg: str | None = None
    predicted_complexity_split: str | None = None
    mdc_confidence: float | None = None
    base_confidence: float | None = None
    split_confidence: float | None = None
    prob_assigned_split: float | None = None
    mdc_discrepancy_flag: bool | None = None
    base_discrepancy_flag: bool | None = None
    split_inflation_flag: bool | None = None
    is_billing_anomaly: bool | None = None
    audit_priority_score: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def skipped(cls, admission_no: str | None, reason: str, assigned_drg: str | None = None) -> AuditResult:
        return cls(admission_no=admission_no, status="skipped", reason=reason, assigned_drg=assigned_drg)

    @classmethod
    def rejected(cls, admission_no: str | None, reason: str) -> AuditResult:
        return cls(admission_no=admission_no, status="rejected", reason=reason)
