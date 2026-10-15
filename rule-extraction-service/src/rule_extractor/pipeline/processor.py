"""Process ONE unit of work: build prompts + tool profile, run the agent with retries, persist the rules."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..agents.collector import RuleCollector
from ..agents.contract_check import run_contract_check
from ..agents.prompts import PROMPT_VERSION, PromptLibrary
from ..agents.types import AgentRunner, AgentRunStats, AgentTask, ToolProfile
from ..domain.jobs import JobRecord, UnitRecord
from ..errors import AgentError, ConfigError, UnavailableError
from ..ingestion.corpus import CorpusIndex
from ..llm.bedrock import resolved_models
from ..logging_setup import event
from ..output.jsonl import dumps_line
from ..settings import Settings
from ..storage.interfaces import ArtifactStore

log = logging.getLogger(__name__)
MAX_BACKOFF_SECONDS = 120.0


@dataclass
class UnitOutcome:
    rules_key: str | None
    rules_count: int = 0
    rejected_count: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = "success"
    duration_s: float = 0.0
    model: str = ""


def unit_rules_key(job_id: str, unit_id: str) -> str:
    return f"jobs/{job_id}/units/{unit_id}.jsonl"


class UnitProcessor:
    def __init__(self, settings: Settings, corpus: CorpusIndex | None, runner: AgentRunner, artifacts: ArtifactStore,
                 prompts: PromptLibrary | None = None, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 rng: Callable[[], float] = random.random) -> None:
        self._s, self._corpus, self._runner, self._artifacts = settings, corpus, runner, artifacts
        self._prompts = prompts or PromptLibrary()
        self._sleep, self._rng = sleep, rng

    # ------------------------------------------------------------------ public
    async def process(self, job: JobRecord, unit: UnitRecord) -> UnitOutcome:
        if unit.kind == "contract_check":
            return await self._contract_check(job)
        agent = self._s.agent
        attempts_allowed = agent.max_retries + 1
        max_turns = min(agent.max_turns, agent.max_turns_cap)
        total_cost = 0.0
        for attempt in range(1, attempts_allowed + 1):
            task, collector, model = self._build(job, unit, max_turns)
            try:
                stats = await self._runner.run(task)
            except AgentError as exc:
                total_cost += _cost(exc.stats)
                exc.cost_usd = total_cost  # type: ignore[attr-defined]
                event(log, logging.WARNING, "agent attempt failed", attempt=attempt, reason=exc.reason, transient=exc.transient,
                      final=(not exc.transient or attempt >= attempts_allowed))
                if not exc.transient or attempt >= attempts_allowed:
                    raise
                if exc.reason == "max_turns":     # retries escalate the turn budget instead of repeating a too-small one
                    max_turns = min(int(max_turns * 1.5) + 1, agent.max_turns_cap)
                await self._sleep(self._backoff(attempt))
                continue
            total_cost += _cost(stats)
            return self._persist(job, unit, collector, stats, total_cost, model)
        raise AssertionError("unreachable")  # pragma: no cover

    # ----------------------------------------------------------------- internals
    def _backoff(self, attempt: int) -> float:
        base = self._s.agent.retry_base_delay_seconds
        return min(MAX_BACKOFF_SECONDS, base * (2 ** (attempt - 1)) * (0.5 + self._rng()))

    def _timeout(self, max_turns: int) -> float:
        a = self._s.agent
        return float(max(a.turn_timeout_seconds, max_turns * a.seconds_per_turn_budget))

    def _need_corpus(self) -> CorpusIndex:
        if self._corpus is None:
            raise UnavailableError("corpus not available")
        return self._corpus

    def _build(self, job: JobRecord, unit: UnitRecord, max_turns: int) -> tuple[AgentTask, RuleCollector, str]:
        s, spec, models = self._s, unit.spec, resolved_models(self._s)
        common = dict(job_id=job.job_id, unit_id=unit.unit_id, prompt_version=PROMPT_VERSION, max_rules=s.agent.max_rules_per_unit)
        if unit.kind == "acs_topic":
            corpus = self._need_corpus()
            prompts = self._prompts.acs_topic(code=spec["code"], title=spec["title"], start_page=spec["start_page"],
                                              end_page=spec["end_page"],
                                              standard_text=corpus.window_text("acs", spec["start_page"], spec["end_page"]))
            model = models["strong"]
            profile = ToolProfile("corpus", builtin_tools=("Read", "Grep", "Glob"), file_root=Path(corpus.root).resolve())
            collector = RuleCollector(topic=f"ACS {spec['code']} {spec['title']}", source_model=model,
                                      registry=corpus.known_codes(), **common)
            untrusted = True
        elif unit.kind == "sweep":
            corpus = self._need_corpus()
            label = corpus.doc_label(spec["doc_key"])
            prompts = self._prompts.sweep(doc_kind=spec["doc_kind"], doc_key=spec["doc_key"], doc_label=label,
                                          start_page=spec["start_page"], end_page=spec["end_page"],
                                          window_text=corpus.window_text(spec["doc_key"], spec["start_page"], spec["end_page"]))
            model = models["fast"] if spec["doc_kind"] == "alphabetic" else models["strong"]
            profile = ToolProfile("sweep")
            collector = RuleCollector(topic=f"{label} pp.{spec['start_page']}-{spec['end_page']}", source_model=model,
                                      registry=corpus.known_codes(), **common)
            untrusted = True
        elif unit.kind == "web_research":
            if not s.web.enabled:
                raise ConfigError("web research is disabled by configuration")
            prompts = self._prompts.web_research(topic=spec["topic"], allowed_hosts=s.web.allowed_hosts)
            model = models["web"]
            profile = ToolProfile("web", builtin_tools=("WebFetch",), web_allowed_hosts=s.web.allowed_hosts)
            collector = RuleCollector(topic=f"web: {spec['topic']}", source_model=model, extraction_method="LLM-Web",
                                      required_url_hosts=s.web.allowed_hosts, **common)
            untrusted = False
        else:
            raise ConfigError(f"unknown unit kind {unit.kind!r}")
        task = AgentTask(unit_id=unit.unit_id, kind=unit.kind, system_prompt=prompts.system, user_prompt=prompts.user,
                         model=model, profile=profile, tools=[collector.tool_spec()], max_turns=max_turns,
                         timeout_seconds=self._timeout(max_turns), max_budget_usd=s.agent.max_budget_usd_per_unit,
                         untrusted_text_in_prompt=untrusted)
        return task, collector, model

    def _persist(self, job: JobRecord, unit: UnitRecord, collector: RuleCollector, stats: AgentRunStats, cost: float,
                 model: str) -> UnitOutcome:
        lines = [dumps_line({"rule": e.rule.as_ledger_record(), "executable": e.executable.to_dict() if e.executable else None})
                 for e in collector.emitted]
        key = unit_rules_key(job.job_id, unit.unit_id)
        self._artifacts.put_bytes(key, ("".join(line + "\n" for line in lines)).encode("utf-8"), content_type="application/x-ndjson")
        event(log, logging.INFO, "unit complete", rules=len(lines), rejected=len(collector.rejected), turns=stats.turns,
              cost_usd=round(cost, 4), duration_s=round(stats.duration_s, 1), denied=len(stats.denied_tools))
        return UnitOutcome(rules_key=key, rules_count=len(lines), rejected_count=len(collector.rejected), cost_usd=cost,
                           turns=stats.turns, usage=stats.usage, stop_reason=stats.stop_reason,
                           duration_s=stats.duration_s, model=model)

    async def _contract_check(self, job: JobRecord) -> UnitOutcome:
        report = await run_contract_check(self._runner, self._s)
        key = f"jobs/{job.job_id}/outputs/contract_report.json"
        self._artifacts.put_bytes(key, json.dumps(report.to_dict(), indent=2).encode("utf-8"), content_type="application/json")
        cost = sum(c.cost_usd for c in report.checks)
        if not report.passed:
            exc = AgentError("contract check failed: " + ", ".join(report.failed_names), transient=False, reason="contract_failed")
            exc.cost_usd = cost  # type: ignore[attr-defined]
            raise exc
        return UnitOutcome(rules_key=None, cost_usd=cost, duration_s=sum(c.duration_s for c in report.checks), model="contract")


def _cost(stats: Any) -> float:
    value = getattr(stats, "cost_usd", None)
    return float(value) if isinstance(value, (int, float)) else 0.0
