"""Strict request models for the ``/invocations`` API (no unknown fields, bounded sizes, safe charsets)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..errors import RequestValidationError
from ..settings import SWEEP_DOCUMENT_KEYS
from ..util import canonical_json, clean_text, sha256_bytes

TOPIC_RE = re.compile(r"^\d{4}$")
IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9_.:\-]{8,128}$")
WEB_TOPIC_RE = re.compile(r"^[A-Za-z0-9 ,.;:()'/&\-]{3,200}$")
JOB_KINDS = ("extraction", "contract_check")


def _reject_unknown(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise RequestValidationError(f"unknown field(s) in {where}", details=[f"unknown field: {k}" for k in unknown])


def _opt_int(obj: dict[str, Any], key: str, lo: int, hi: int, errors: list[str]) -> int | None:
    if obj.get(key) is None:
        return None
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        errors.append(f"{key} must be an integer between {lo} and {hi}")
        return None
    return value


def _opt_bool(obj: dict[str, Any], key: str, default: bool, errors: list[str]) -> bool:
    if key not in obj or obj[key] is None:
        return default
    if not isinstance(obj[key], bool):
        errors.append(f"{key} must be a boolean")
        return default
    return obj[key]


def _opt_str_list(obj: dict[str, Any], key: str, pattern: re.Pattern[str], max_items: int, errors: list[str]) -> list[str] | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not value or len(value) > max_items:
        errors.append(f"{key} must be a non-empty list with at most {max_items} items")
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not pattern.match(item):
            errors.append(f"{key} contains an invalid entry: {str(item)[:40]!r}")
            continue
        out.append(item)
    return out


@dataclass(frozen=True)
class JobOptions:
    """Mirror of the original CLI flags (``RunOptions``) plus service-level guards."""

    kind: str = "extraction"
    topics: tuple[str, ...] | None = None
    limit_topics: int | None = None
    include_acs_topics: bool = True
    include_tabular_sweep: bool = True
    sweep_documents: tuple[str, ...] = SWEEP_DOCUMENT_KEYS
    limit_sweep_windows: int | None = None
    include_web_research: bool = False
    web_research_topics: tuple[str, ...] | None = None
    max_cost_usd: float | None = None

    _FIELDS = {
        "kind", "topics", "limit_topics", "include_acs_topics", "include_tabular_sweep", "sweep_documents",
        "limit_sweep_windows", "include_web_research", "web_research_topics", "max_cost_usd",
    }

    @classmethod
    def from_dict(cls, raw: Any) -> "JobOptions":
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise RequestValidationError("options must be an object")
        _reject_unknown(raw, set(cls._FIELDS), "options")
        errors: list[str] = []

        kind = raw.get("kind", "extraction")
        if kind not in JOB_KINDS:
            errors.append(f"kind must be one of {JOB_KINDS}")
            kind = "extraction"
        topics = _opt_str_list(raw, "topics", TOPIC_RE, 200, errors)
        sweep_docs = raw.get("sweep_documents")
        if sweep_docs is not None:
            if not isinstance(sweep_docs, list) or not sweep_docs or any(d not in SWEEP_DOCUMENT_KEYS for d in sweep_docs):
                errors.append(f"sweep_documents must be a non-empty subset of {list(SWEEP_DOCUMENT_KEYS)}")
                sweep_docs = None
        web_topics = _opt_str_list(raw, "web_research_topics", WEB_TOPIC_RE, 10, errors)
        max_cost = raw.get("max_cost_usd")
        if max_cost is not None:
            if isinstance(max_cost, bool) or not isinstance(max_cost, (int, float)) or not 0 < max_cost <= 100_000:
                errors.append("max_cost_usd must be a positive number")
                max_cost = None
        options = cls(
            kind=kind,
            topics=tuple(topics) if topics else None,
            limit_topics=_opt_int(raw, "limit_topics", 1, 1000, errors),
            include_acs_topics=_opt_bool(raw, "include_acs_topics", True, errors),
            include_tabular_sweep=_opt_bool(raw, "include_tabular_sweep", True, errors),
            sweep_documents=tuple(sweep_docs) if sweep_docs else SWEEP_DOCUMENT_KEYS,
            limit_sweep_windows=_opt_int(raw, "limit_sweep_windows", 1, 1000, errors),
            include_web_research=_opt_bool(raw, "include_web_research", False, errors),
            web_research_topics=tuple(clean_text(t) for t in web_topics) if web_topics else None,
            max_cost_usd=float(max_cost) if max_cost is not None else None,
        )
        if errors:
            raise RequestValidationError("invalid options", details=errors)
        return options

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "topics": list(self.topics) if self.topics else None,
            "limit_topics": self.limit_topics,
            "include_acs_topics": self.include_acs_topics,
            "include_tabular_sweep": self.include_tabular_sweep,
            "sweep_documents": list(self.sweep_documents),
            "limit_sweep_windows": self.limit_sweep_windows,
            "include_web_research": self.include_web_research,
            "web_research_topics": list(self.web_research_topics) if self.web_research_topics else None,
            "max_cost_usd": self.max_cost_usd,
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))


def parse_idempotency_key(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not IDEMPOTENCY_RE.match(raw):
        raise RequestValidationError("idempotency_key must match ^[A-Za-z0-9_.:-]{8,128}$")
    return raw


@dataclass(frozen=True)
class Envelope:
    """The outer ``/invocations`` request: ``{"operation": ..., ...operation-specific fields...}``."""

    operation: str
    body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any, *, known_operations: tuple[str, ...]) -> "Envelope":
        if not isinstance(raw, dict):
            raise RequestValidationError("request body must be a JSON object")
        operation = raw.get("operation")
        if not isinstance(operation, str) or operation not in known_operations:
            raise RequestValidationError("unknown or missing operation", details=[f"operation must be one of {list(known_operations)}"])
        body = {k: v for k, v in raw.items() if k != "operation"}
        return cls(operation=operation, body=body)
