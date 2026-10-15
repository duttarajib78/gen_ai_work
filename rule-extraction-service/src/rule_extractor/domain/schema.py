"""Rule schema matched to the user-supplied template (``rule_catalogue`` sheet, 11 columns).

Migrated from the original ``schema.py`` with the same controlled vocabularies and invariants:

* ``confidence`` is a *review state*, not a statistical score. Agents can never write ``verified``;
  only a human reviewer promotes a row after checking its citation.
* ``recommended_action`` is one of four short tokens; anything else degrades to the safe
  ``flag_for_clinical_review`` (never auto-correct).

Additions for production: length limits, text normalisation of untrusted model output, richer
provenance (prompt version, job/unit ids), and structured validation errors.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..util import clean_text

RECOMMENDED_ACTIONS = [
    "remove_code",
    "resequence_principal_diagnosis",
    "correct_code",
    "flag_for_clinical_review",
]
MATERIALITY_LEVELS = ["high", "medium", "low"]
CONFIDENCE_STATES = ["needs_review", "disputed", "verified"]
EXTRACTION_METHODS = ["LLM-Synthesis", "LLM-Web", "Deterministic-XML", "Manual-Seed"]
RULE_SHAPES = [
    "redundancy",
    "inherent_component",
    "sequencing",
    "evidence_requirement",
    "must_never_assign",
    "once_only",
    "demographic_implausibility",
]

EXCEL_COLUMNS = [
    "rule_id",
    "rule_name",
    "rule_logic_plain_english",
    "icd10am_codes",
    "achi_codes",
    "mbs_codes",
    "upcoding_mechanism",
    "recommended_action",
    "source_reference",
    "materiality",
    "confidence",
]

# Tracked for audit/resume/provenance; NOT part of the delivered workbook.
INTERNAL_ONLY_COLUMNS = [
    "extraction_method",
    "source_model",
    "date_extracted",
    "topic",
    "rule_shape",
    "prompt_version",
    "job_id",
    "unit_id",
    "recommended_action_raw",
    "validation_warnings",
]

FIELD_LIMITS = {
    "rule_name": 200,
    "rule_logic_plain_english": 3000,
    "upcoding_mechanism": 3000,
    "source_reference": 1500,
    "icd10am_codes": 800,
    "achi_codes": 800,
    "mbs_codes": 800,
    "topic": 200,
}
REQUIRED_FIELDS = [
    "rule_name",
    "rule_logic_plain_english",
    "upcoding_mechanism",
    "recommended_action",
    "source_reference",
]


class RuleValidationError(ValueError):
    """Raised with *every* problem found so the model (or caller) can fix them in one pass."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Rule:
    rule_name: str
    rule_logic_plain_english: str
    upcoding_mechanism: str
    recommended_action: str
    source_reference: str
    icd10am_codes: str = "N/A"
    achi_codes: str = "N/A"
    mbs_codes: str = "N/A"
    materiality: str = "medium"
    confidence: str = "needs_review"   # never "verified" from an agent - see module docstring
    extraction_method: str = "LLM-Synthesis"
    source_model: str = ""
    topic: str = ""
    rule_shape: str = ""
    prompt_version: str = ""
    job_id: str = ""
    unit_id: str = ""
    recommended_action_raw: str = ""
    validation_warnings: list[str] = field(default_factory=list)
    date_extracted: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    rule_id: str = ""

    def __post_init__(self) -> None:
        for name in ("rule_name", "rule_logic_plain_english", "upcoding_mechanism", "source_reference",
                     "icd10am_codes", "achi_codes", "mbs_codes", "topic", "recommended_action"):
            setattr(self, name, clean_text(getattr(self, name)))
        for name in ("icd10am_codes", "achi_codes", "mbs_codes"):
            if not getattr(self, name):
                setattr(self, name, "N/A")
        if self.recommended_action not in RECOMMENDED_ACTIONS:
            # Keep the original text for audit but never let it masquerade as another category.
            self.recommended_action_raw = self.recommended_action_raw or self.recommended_action
            self.recommended_action = "flag_for_clinical_review"
            self.validation_warnings = list(self.validation_warnings) + ["recommended_action_coerced"]
        if self.materiality not in MATERIALITY_LEVELS:
            self.materiality = "medium"
        if self.confidence == "verified":
            # An agent asserting "verified" would be exactly the false claim the AI-content policy
            # exists to prevent - only a human review step may write this value.
            self.confidence = "needs_review"
        elif self.confidence not in CONFIDENCE_STATES:
            self.confidence = "needs_review"
        if self.extraction_method not in EXTRACTION_METHODS:
            self.extraction_method = "LLM-Synthesis"
        if self.rule_shape and self.rule_shape not in RULE_SHAPES:
            self.validation_warnings = list(self.validation_warnings) + [f"unknown_rule_shape:{self.rule_shape}"]
            self.rule_shape = ""
        if not self.rule_id:
            self.rule_id = self._make_id()

    def _make_id(self) -> str:
        basis = f"{self.rule_name}|{self.rule_logic_plain_english}".encode("utf-8")
        return "RULE_" + hashlib.sha1(basis).hexdigest()[:8].upper()  # identifier only, not security-relevant

    def as_row(self) -> dict[str, str]:
        data = asdict(self)
        return {col: data.get(col, "") for col in EXCEL_COLUMNS}

    def as_ledger_record(self) -> dict[str, Any]:
        data = asdict(self)
        return {col: data.get(col, "") for col in EXCEL_COLUMNS + INTERNAL_ONLY_COLUMNS}

    def dedupe_key(self) -> str:
        return " ".join(self.rule_logic_plain_english.lower().split())


def validate_rule_dict(raw: dict[str, Any], *, enforce_limits: bool = True) -> Rule:
    """Build a :class:`Rule` from untrusted input or raise :class:`RuleValidationError` listing all problems."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise RuleValidationError(["rule must be a JSON object"])

    cleaned: dict[str, Any] = {}
    for name in REQUIRED_FIELDS:
        value = clean_text(raw.get(name))
        if not value:
            errors.append(f"{name} is required and must be non-empty text")
        cleaned[name] = value
    if enforce_limits:
        for name, limit in FIELD_LIMITS.items():
            value = clean_text(raw.get(name, cleaned.get(name, "")))
            if len(value) > limit:
                errors.append(f"{name} is {len(value)} characters; the maximum is {limit} - shorten it")
    if errors:
        raise RuleValidationError(errors)

    kwargs: dict[str, Any] = dict(cleaned)
    for name in ("icd10am_codes", "achi_codes", "mbs_codes", "materiality", "confidence", "extraction_method",
                 "source_model", "topic", "rule_shape", "prompt_version", "job_id", "unit_id",
                 "recommended_action_raw", "date_extracted", "rule_id"):
        if raw.get(name) not in (None, ""):
            kwargs[name] = raw[name] if name not in ("icd10am_codes", "achi_codes", "mbs_codes") else clean_text(raw[name])
    warnings = raw.get("validation_warnings")
    if isinstance(warnings, list):
        kwargs["validation_warnings"] = [clean_text(w) for w in warnings][:20]
    return Rule(**kwargs)


def rule_from_ledger_record(record: dict[str, Any]) -> Rule:
    """Rehydrate a stored rule (limits not re-enforced so historical rows always load)."""
    return validate_rule_dict(record, enforce_limits=False)
