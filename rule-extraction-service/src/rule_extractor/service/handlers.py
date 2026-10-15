"""Operation handlers for ``POST /invocations``. Strict schemas: unknown fields are rejected, sizes are bounded.

Request bodies are NEVER logged. Episode data (``evaluate_episodes``) is only echoed back as ``episode_id``.
"""
from __future__ import annotations

from typing import Any, Callable

from .. import __version__
from ..domain.api import JobOptions, parse_idempotency_key
from ..domain.jobs import JobStatus, UnitStatus
from ..engine.compare import run_episode_against_rule
from ..engine.episode import Episode
from ..errors import PolicyError, RequestValidationError
from .container import ServiceContainer

Handler = Callable[[ServiceContainer, dict[str, Any]], dict[str, Any]]


def _only(body: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestValidationError("unknown field(s)", details=[f"unknown field: {k}" for k in unknown])


def _int(body: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise RequestValidationError(f"{key} must be an integer between {lo} and {hi}")
    return value


def op_plan(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"options"})
    return c.jobs.plan(JobOptions.from_dict(body.get("options")))


def op_submit_job(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"options", "idempotency_key"})
    return c.jobs.submit(JobOptions.from_dict(body.get("options")), parse_idempotency_key(body.get("idempotency_key")))


def op_get_job(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"job_id", "include_units", "unit_status", "limit"})
    include = body.get("include_units", False)
    if not isinstance(include, bool):
        raise RequestValidationError("include_units must be a boolean")
    status = body.get("unit_status")
    if status is not None and status not in {s.value for s in UnitStatus}:
        raise RequestValidationError(f"unit_status must be one of {[s.value for s in UnitStatus]}")
    return c.jobs.get(body.get("job_id"), include_units=include, unit_status=status, limit=_int(body, "limit", 200, 1, 600))


def op_list_jobs(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"status", "limit"})
    status = body.get("status")
    if status is not None and status not in {s.value for s in JobStatus}:
        raise RequestValidationError(f"status must be one of {[s.value for s in JobStatus]}")
    return c.jobs.list_jobs(status, _int(body, "limit", 20, 1, 50))


def op_get_rules(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"job_id", "offset", "limit"})
    return c.jobs.get_rules(body.get("job_id"), _int(body, "offset", 0, 0, 10_000_000), _int(body, "limit", 50, 1, 100))


def op_get_results(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"job_id"})
    return c.jobs.get_results(body.get("job_id"))


def op_cancel_job(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"job_id"})
    return c.jobs.cancel(body.get("job_id"))


def op_retry_failed(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"job_id"})
    return c.jobs.retry_failed(body.get("job_id"))


def op_health(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, set())
    queue: dict[str, Any] = {}
    try:
        visible, in_flight = c.backends.queue.depth()
        queue = {"visible": visible, "in_flight": in_flight}
    except Exception:
        queue = {"error": "unavailable"}
    return {"status": "ok", "service_version": __version__, "settings": c.settings.public_summary(), "queue": queue,
            "corpus_available": c.corpus is not None, "versions": c.jobs.version_info()}


def op_evaluate_episodes(c: ServiceContainer, body: dict[str, Any]) -> dict[str, Any]:
    _only(body, {"episodes", "rule_ids"})
    raw = body.get("episodes")
    limit = c.settings.service.max_episodes_per_request
    if not isinstance(raw, list) or not 1 <= len(raw) <= limit:
        raise RequestValidationError(f"episodes must be a list of 1-{limit} objects")
    episodes, problems = [], []
    for i, item in enumerate(raw):
        try:
            episodes.append(Episode.from_dict(item, strict=True))
        except ValueError as exc:
            problems.append(f"episodes[{i}]: {exc}")
    if problems:
        raise RequestValidationError("invalid episode(s)", details=problems[:20])
    rules = c.executable_rules()
    wanted = body.get("rule_ids")
    if wanted is not None:
        if not isinstance(wanted, list) or not all(isinstance(x, str) for x in wanted) or len(wanted) > 1000:
            raise RequestValidationError("rule_ids must be a list of strings")
        rules = [r for r in rules if r.rule_id in set(wanted)]
    grouper = c.grouper()
    results, is_mock = [], True
    for ep in episodes:
        flagged = []
        for rule in rules:
            res = run_episode_against_rule(ep, rule, grouper)
            if not res.was_flagged:
                continue
            is_mock = bool((res.before and res.before.is_mock) or (res.after and res.after.is_mock) or (not res.corrected and is_mock))
            flagged.append({"rule_id": rule.rule_id, "corrected": res.corrected, "automatable": res.automatable,
                            "cost_weight_delta": res.cost_weight_delta,
                            "before_drg": res.before.drg_code if res.before else None, "after_drg": res.after.drg_code if res.after else None,
                            "notes": res.notes})
        results.append({"episode_id": ep.episode_id, "rules_evaluated": len(rules), "flagged": flagged})
    return {"results": results, "preview_only": True, "grouper": {"is_mock": is_mock}}


OPERATIONS: dict[str, Handler] = {
    "plan": op_plan, "submit_job": op_submit_job, "get_job": op_get_job, "list_jobs": op_list_jobs, "get_rules": op_get_rules,
    "get_results": op_get_results, "cancel_job": op_cancel_job, "retry_failed": op_retry_failed, "health": op_health,
    "evaluate_episodes": op_evaluate_episodes,
}


def dispatch(c: ServiceContainer, operation: str, body: dict[str, Any]) -> dict[str, Any]:
    if operation not in c.settings.service.allowed_operations:
        raise PolicyError(f"operation {operation!r} is disabled on this deployment")
    return OPERATIONS[operation](c, body)
