"""Turn job options into an ordered list of units of work (the original CLI's "plan").

Unit kinds: ``acs_topic`` (one ACS standard, cross-document research), ``sweep`` (a page window of a Tabular List or
Alphabetic Index), ``web_research`` (allow-listed WebFetch, opt-in), ``contract_check`` (SDK/Bedrock diagnostics).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..domain.api import JobOptions
from ..errors import PolicyError, RequestValidationError, UnavailableError
from ..ingestion.chunking import sliding_windows
from ..ingestion.corpus import CorpusIndex
from ..ingestion.sources import resolve_sources
from ..settings import SWEEP_DOCUMENT_KEYS, Settings


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    kind: str
    order: int
    spec: dict[str, Any]


@dataclass
class Plan:
    units: list[UnitSpec] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "topic"


def build_plan(options: JobOptions, settings: Settings, corpus: CorpusIndex | None) -> Plan:
    if options.kind == "contract_check":
        return Plan([UnitSpec("contract-check", "contract_check", 0, {})], {"units_total": 1, "contract_check": 1})
    if corpus is None:
        raise UnavailableError("the corpus is not available on this instance; build and package it first (rule-extractor corpus build)")

    units: list[UnitSpec] = []
    summary: dict[str, Any] = {"acs_topics": 0, "sweep_windows": {}, "web_research": 0}

    def add(unit_id: str, kind: str, spec: dict[str, Any]) -> None:
        units.append(UnitSpec(unit_id, kind, len(units), spec))

    if options.include_acs_topics:
        standards = corpus.standards()
        if options.topics:
            known = {s.code for s in standards}
            unknown = sorted(set(options.topics) - known)
            if unknown:
                raise RequestValidationError("unknown ACS topic code(s)", details=[f"not in the corpus: {unknown}"])
            standards = [s for s in standards if s.code in set(options.topics)]
        standards = sorted(standards, key=lambda s: s.code)
        if options.limit_topics:
            standards = standards[: options.limit_topics]
        for s in standards:
            add(f"acs-{s.code}", "acs_topic", {"code": s.code, "title": s.title, "start_page": s.start_page, "end_page": s.end_page})
        summary["acs_topics"] = len(standards)

    if options.include_tabular_sweep:
        sources = resolve_sources(settings.extras.get("sources"))
        for doc_key in (k for k in SWEEP_DOCUMENT_KEYS if k in options.sweep_documents):
            kind = sources[doc_key].kind
            ck = settings.chunking
            ppc, ov = ((ck.tabular_pages_per_chunk, ck.tabular_page_overlap) if kind == "tabular"
                       else (ck.alphabetic_pages_per_chunk, ck.alphabetic_page_overlap))
            windows = sliding_windows(corpus.page_count(doc_key), ppc, ov)
            if options.limit_sweep_windows:
                windows = windows[: options.limit_sweep_windows]
            for idx, (start, end) in enumerate(windows):
                add(f"sweep-{doc_key}-w{idx:04d}", "sweep",
                    {"doc_key": doc_key, "doc_kind": kind, "window_index": idx, "start_page": start, "end_page": end})
            summary["sweep_windows"][doc_key] = len(windows)

    if options.include_web_research:
        if not settings.web.enabled:
            raise PolicyError("web research is disabled on this deployment (it needs outbound access and is a prompt-injection surface)")
        topics = list(options.web_research_topics or settings.web.default_topics)
        if not topics:
            raise RequestValidationError("web research requires web_research_topics (none configured by default)")
        for topic in topics:
            add(f"web-{_slug(topic)}-{hashlib.sha1(topic.encode()).hexdigest()[:6]}", "web_research", {"topic": topic})
        summary["web_research"] = len(topics)

    if not units:
        raise RequestValidationError("the plan is empty: enable at least one of ACS topics, the sweep, or web research")
    if len(units) > settings.service.max_units_per_job:
        raise RequestValidationError(f"plan has {len(units)} units; the limit is {settings.service.max_units_per_job}")
    summary["sweep_windows_total"] = sum(summary["sweep_windows"].values())
    summary["units_total"] = len(units)
    return Plan(units, summary)
