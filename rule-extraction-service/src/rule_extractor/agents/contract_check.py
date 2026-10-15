"""Live SDK contract check - the productionised successor of ``minimal_repro.py``.

Runs three tiny sessions against Bedrock with the *production* permission settings and reports:
  simple_tool   a tool with a plain string parameter is dispatched and its handler runs
  nested_tool   a tool with a nested-object parameter (the ``emit_rule`` shape) is dispatched
  deny_outside  a Read outside the corpus root is denied by the policy gate and never executed
It also proves both model tiers (fast + strong) are reachable with the configured inference profiles and IAM role.
Run it after every deploy (``submit_job`` with ``{"kind": "contract_check"}``) and after every SDK/CLI upgrade.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..llm.bedrock import resolved_models
from ..logging_setup import redact
from ..settings import Settings
from .types import AgentRunner, AgentRunStats, AgentTask, ToolProfile, ToolResult, ToolSpec


@dataclass
class CheckResult:
    name: str
    status: str            # pass | fail | inconclusive
    detail: str = ""
    model: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0


@dataclass
class ContractReport:
    checks: list[CheckResult] = field(default_factory=list)
    sdk_version: str = "unknown"
    region: str = ""
    models: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.status != "fail" for c in self.checks)

    @property
    def failed_names(self) -> list[str]:
        return [c.name for c in self.checks if c.status == "fail"]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "sdk_version": self.sdk_version, "region": self.region, "models": self.models,
                "checks": [c.__dict__ for c in self.checks]}


def _task(unit: str, instruction: str, model: str, profile: ToolProfile, tools: list[ToolSpec]) -> AgentTask:
    return AgentTask(unit_id=unit, kind="contract_check", system_prompt=instruction,
                     user_prompt="Follow the system instructions now.", model=model, profile=profile, tools=tools,
                     max_turns=6, timeout_seconds=150, max_budget_usd=0.5)


async def _case(name: str, model: str, runner: AgentRunner, task: AgentTask,
                verify: Callable[[AgentRunStats], tuple[str, str]]) -> CheckResult:
    started = time.monotonic()
    try:
        stats = await runner.run(task)
        status, detail = verify(stats)
        return CheckResult(name, status, detail, model, stats.cost_usd or 0.0, time.monotonic() - started)
    except Exception as exc:  # report, never raise: the point is to surface the failure
        stats = getattr(exc, "stats", None)
        return CheckResult(name, "fail", redact(f"{type(exc).__name__}: {str(exc)[:300]}"), model,
                           (getattr(stats, "cost_usd", None) or 0.0), time.monotonic() - started)


async def run_contract_check(runner: AgentRunner, settings: Settings) -> ContractReport:
    models = resolved_models(settings)
    report = ContractReport(region=settings.bedrock.region, models=models)
    try:
        from importlib import metadata
        report.sdk_version = metadata.version("claude-agent-sdk")
    except Exception:
        pass

    simple_calls: list[dict] = []
    nested_calls: list[dict] = []
    read_probe: list[dict] = []

    def record_simple(args: dict) -> ToolResult:
        simple_calls.append(args)
        return ToolResult("Recorded.")

    def record_nested(args: dict) -> ToolResult:
        nested_calls.append(args)
        return ToolResult("Recorded.")

    simple = ToolSpec("record_simple", "Record a plain text message", {
        "type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}, record_simple)
    nested = ToolSpec("record_nested", "Record a message with a nested detail object", {
        "type": "object", "properties": {"message": {"type": "string"}, "detail": {"type": "object"}},
        "required": ["message"]}, record_nested)
    done = ToolSpec("record_done", "Call once after the read attempt", {
        "type": "object", "properties": {"note": {"type": "string"}}}, lambda a: (read_probe.append(a) or ToolResult("ok")))

    report.checks.append(await _case(
        "simple_tool", models["fast"], runner,
        _task("contract-simple", "You must call the record_simple tool exactly once with message='hello world', then stop.",
              models["fast"], ToolProfile("contract-simple"), [simple]),
        lambda s: ("pass", "handler invoked once") if len(simple_calls) == 1 and simple_calls[0].get("message") == "hello world"
        else ("fail", f"handler invoked {len(simple_calls)} time(s); args={simple_calls[:1]}")))

    report.checks.append(await _case(
        "nested_tool", models["strong"], runner,
        _task("contract-nested", "You must call the record_nested tool exactly once with message='hello world' and "
                                 "detail={'a': 1, 'b': [1, 2, 3]}, then stop.",
              models["strong"], ToolProfile("contract-nested"), [nested]),
        lambda s: ("pass", "nested object delivered") if len(nested_calls) == 1 and isinstance(nested_calls[0].get("detail"), dict)
        else ("fail", f"handler invoked {len(nested_calls)} time(s); args={nested_calls[:1]}")))

    with tempfile.TemporaryDirectory(prefix="contract-root-") as tmp:
        profile = ToolProfile("contract-read", builtin_tools=("Read",), file_root=Path(tmp))

        def verify_deny(stats: AgentRunStats) -> tuple[str, str]:
            if "Read" in stats.denied_tools:
                return "pass", "read outside the corpus root was denied by the policy gate"
            if stats.tool_calls.get("Read"):
                return "fail", "a Read outside the corpus root was NOT denied - review permission settings immediately"
            return "inconclusive", "the model did not attempt the read"

        report.checks.append(await _case(
            "deny_outside_root", models["strong"], runner,
            _task("contract-deny", "Call the Read tool once with file_path='/etc/hostname'. If it is refused, do not retry; "
                                   "call record_done once and stop.",
                  models["strong"], profile, [done]),
            verify_deny))
    return report
