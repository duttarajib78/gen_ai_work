"""Job lifecycle: submit / get / cancel / retry / finalize. Pure orchestration over the storage ports."""
from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from .. import __version__
from ..agents.prompts import PROMPT_VERSION, prompt_digest
from ..domain.api import JobOptions
from ..domain.jobs import JOB_ID_RE, JobRecord, JobStatus, UnitRecord, UnitStatus, WorkItem
from ..errors import ConfigError, ConflictError, NotFoundError, RequestValidationError
from ..ingestion.corpus import CorpusIndex
from ..llm.bedrock import resolved_models
from ..output.jsonl import iter_jsonl
from ..pipeline.finalize import Finalizer
from ..pipeline.planner import build_plan
from ..settings import Settings
from ..storage.interfaces import Backends
from ..util import truncate

log = logging.getLogger(__name__)
FAILURE_PREVIEW = 20


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobService:
    def __init__(self, settings: Settings, backends: Backends, corpus: CorpusIndex | None, *,
                 finalizer: Finalizer | None = None, clock: Callable[[], float] = time.time) -> None:
        self.settings, self.corpus, self.finalizer, self._clock = settings, corpus, finalizer, clock
        self.state, self.artifacts, self.queue = backends.state, backends.artifacts, backends.queue
        self._version: dict[str, Any] | None = None

    # -------------------------------------------------------------------- helpers
    def version_info(self) -> dict[str, Any]:
        if self._version is None:
            try:
                from importlib import metadata
                sdk = metadata.version("claude-agent-sdk")
            except Exception:
                sdk = "unknown"
            try:
                models = resolved_models(self.settings)
            except ConfigError:
                models = {}
            self._version = {"service": __version__, "prompt_version": PROMPT_VERSION, "prompt_digest": prompt_digest(),
                             "models": models, "sdk_version": sdk, "corpus_digest": self.corpus.digest() if self.corpus else None,
                             "inference_profile_prefix": self.settings.bedrock.inference_profile_prefix}
        return self._version

    def _touch(self, job: JobRecord) -> None:
        now = self._clock()
        job.updated_epoch, job.updated_at = now, iso(now)

    def _require(self, job_id: Any) -> JobRecord:
        if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
            raise RequestValidationError("job_id must look like 'job-<hex>'")
        job = self.state.get_job(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job

    def _view(self, job: JobRecord, units: list[UnitRecord] | None, *, include_units: bool = False,
              unit_status: str | None = None, limit: int = 200) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": job.job_id, "kind": job.kind, "status": job.status, "created_at": job.created_at,
            "updated_at": job.updated_at, "units_total": job.units_total, "cost_usd": round(job.cost_usd, 4),
            "plan": job.plan, "artifacts": job.artifacts, "error": job.error, "cancel_requested": job.cancel_requested,
            "options": job.options, "summary": {k: v for k, v in job.summary.items() if k not in ("versions", "artifacts", "failures")},
        }
        if units is not None:
            counts = Counter(u.status for u in units)
            out["progress"] = {**{s.value.lower(): counts.get(s.value, 0) for s in UnitStatus},
                               "done": sum(counts.get(s, 0) for s in ("SUCCEEDED", "FAILED", "CANCELLED")), "total": len(units)}
            out["failures"] = [{"unit_id": u.unit_id, "error": truncate(u.error or "", 300)} for u in units
                               if u.status == UnitStatus.FAILED.value][:FAILURE_PREVIEW]
            if include_units:
                rows = [u for u in units if unit_status is None or u.status == unit_status][:limit]
                out["units"] = [{"unit_id": u.unit_id, "kind": u.kind, "status": u.status, "attempts": u.attempts,
                                 "rules": u.rules_count, "cost_usd": round(u.cost_usd, 4), "turns": u.turns,
                                 "error": truncate(u.error or "", 200)} for u in rows]
        return out

    # ------------------------------------------------------------------------ API
    def plan(self, options: JobOptions) -> dict[str, Any]:
        plan = build_plan(options, self.settings, self.corpus)
        return {"summary": plan.summary, "units_preview": [{"unit_id": u.unit_id, "kind": u.kind} for u in plan.units[:25]],
                "options": options.to_dict()}

    def submit(self, options: JobOptions, idempotency_key: str | None = None) -> dict[str, Any]:
        plan = build_plan(options, self.settings, self.corpus)
        digest = options.digest()
        job_id = "job-" + uuid.uuid4().hex[:20]
        if idempotency_key:
            existing = self.state.reserve_idempotency(idempotency_key, job_id, digest)
            if existing:
                existing_id, existing_digest = existing
                if existing_digest != digest:
                    raise ConflictError("idempotency_key was already used with different options")
                job = self.state.get_job(existing_id)
                if job is None:
                    raise ConflictError("a previous submission with this idempotency_key did not complete; use a new key")
                view = self._view(job, self.state.list_units(existing_id))
                view["deduplicated"] = True
                return view
        now = self._clock()
        job = JobRecord(job_id=job_id, kind=options.kind, status=JobStatus.SUBMITTING.value, options=options.to_dict(),
                        options_digest=digest, idempotency_key=idempotency_key, created_at=iso(now), updated_at=iso(now),
                        created_epoch=now, updated_epoch=now, deadline_epoch=now + self.settings.service.job_max_runtime_seconds,
                        units_total=len(plan.units), plan=plan.summary, version_info=self.version_info())
        units = [UnitRecord(job_id, u.unit_id, u.kind, u.order, u.spec) for u in plan.units]
        self.state.put_job(job)
        self.state.put_units(units)
        self.queue.send_batch([WorkItem(job_id, u.unit_id).to_json() for u in units])
        self.activate(job_id)
        log.info("job submitted", extra={"fields": {"job_id": job_id, "units": len(units), "kind": options.kind}})
        view = self._view(self.state.get_job(job_id) or job, None)
        view.update(deduplicated=False, progress={"total": len(units), "done": 0, "pending": len(units)})
        return view

    def activate(self, job_id: str) -> None:
        def mutate(j: JobRecord) -> bool:
            if j.status != JobStatus.SUBMITTING.value:
                return False
            j.status = JobStatus.RUNNING.value
            self._touch(j)
            return True
        self.state.update_job(job_id, mutate)

    def get(self, job_id: Any, *, include_units: bool = False, unit_status: str | None = None, limit: int = 200) -> dict[str, Any]:
        job = self._require(job_id)
        return self._view(job, self.state.list_units(job.job_id), include_units=include_units, unit_status=unit_status, limit=limit)

    def list_jobs(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        statuses = [status] if status else [s.value for s in JobStatus]
        jobs: list[JobRecord] = []
        for s in statuses:
            jobs.extend(self.state.list_jobs_by_status(s, limit))
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return {"jobs": [{"job_id": j.job_id, "kind": j.kind, "status": j.status, "created_at": j.created_at,
                          "units_total": j.units_total, "cost_usd": round(j.cost_usd, 4)} for j in jobs[:limit]]}

    def cancel(self, job_id: Any) -> dict[str, Any]:
        self._require(job_id)

        def mutate(j: JobRecord) -> bool:
            if j.is_terminal or j.cancel_requested:
                return False
            j.cancel_requested = True
            self._touch(j)
            return True
        _, job = self.state.update_job(job_id, mutate)
        return self._view(job, self.state.list_units(job_id)) if job else {}

    def retry_failed(self, job_id: Any) -> dict[str, Any]:
        job = self._require(job_id)
        if job.status not in (JobStatus.PARTIAL.value, JobStatus.FAILED.value):
            raise ConflictError(f"only PARTIAL or FAILED jobs can be retried (job is {job.status})")
        failed = [u for u in self.state.list_units(job_id) if u.status == UnitStatus.FAILED.value]
        if not failed:
            raise ConflictError("the job has no failed units to retry")

        def reset(u: UnitRecord) -> bool:
            if u.status != UnitStatus.FAILED.value:
                return False
            u.status, u.attempts, u.error, u.lease_owner, u.lease_expires = UnitStatus.PENDING.value, 0, None, None, 0.0
            u.finished_at = None
            return True

        # Order matters: reset units while the job is still terminal (the janitor ignores terminal jobs), then flip the
        # job to RUNNING, then enqueue - so no worker can see a message for a unit whose job is still "finished".
        for unit in failed:
            self.state.update_unit(job_id, unit.unit_id, reset)

        def reopen(j: JobRecord) -> bool:
            j.status, j.error = JobStatus.RUNNING.value, None
            j.deadline_epoch = self._clock() + self.settings.service.job_max_runtime_seconds
            self._touch(j)
            return True
        self.state.update_job(job_id, reopen)
        self.queue.send_batch([WorkItem(job_id, u.unit_id).to_json() for u in failed])
        return self.get(job_id)

    def resume(self, job_id: str, *, reset_running: bool = False) -> int:
        """Re-enqueue unfinished units (local restart, or the janitor repairing a half-submitted job)."""
        enqueue: list[str] = []
        for unit in self.state.list_units(job_id):
            if unit.is_terminal:
                continue
            if unit.status == UnitStatus.RUNNING.value:
                if not reset_running:
                    continue

                def reset(u: UnitRecord) -> bool:
                    if u.status != UnitStatus.RUNNING.value:
                        return False
                    u.status, u.lease_owner, u.lease_expires = UnitStatus.PENDING.value, None, 0.0
                    return True
                self.state.update_unit(job_id, unit.unit_id, reset)
            enqueue.append(WorkItem(job_id, unit.unit_id).to_json())
        if enqueue:
            self.queue.send_batch(enqueue)
        return len(enqueue)

    def get_rules(self, job_id: Any, offset: int, limit: int) -> dict[str, Any]:
        job = self._require(job_id)
        key = f"jobs/{job.job_id}/outputs/rules_final.jsonl"
        if not job.is_terminal or not self.artifacts.exists(key):
            raise ConflictError(f"rules are not available yet (job is {job.status})")
        rows = list(iter_jsonl(self.artifacts.get_bytes(key)))
        page = rows[offset:offset + limit]
        nxt = offset + len(page)
        return {"job_id": job.job_id, "total": len(rows), "offset": offset, "rules": page, "next_offset": nxt if nxt < len(rows) else None}

    def get_results(self, job_id: Any) -> dict[str, Any]:
        job = self._require(job_id)
        return {"job_id": job.job_id, "status": job.status, "artifacts": job.artifacts, "summary": job.summary, "error": job.error}

    # ------------------------------------------------------------ worker/janitor hooks
    def add_cost(self, job_id: str, cost: float) -> None:
        if cost:
            def mutate(j: JobRecord) -> bool:
                j.cost_usd = round(j.cost_usd + cost, 6)
                return True
            self.state.update_job(job_id, mutate)

    def maybe_finalize(self, job_id: str) -> bool:
        """Finalize when every unit is terminal. Exactly one caller wins the RUNNING -> FINALIZING transition."""
        if self.finalizer is None:
            return False
        job = self.state.get_job(job_id)
        if job is None or job.status != JobStatus.RUNNING.value:
            return False
        units = self.state.list_units(job_id)
        if not units or any(not u.is_terminal for u in units):
            return False

        def claim(j: JobRecord) -> bool:
            if j.status != JobStatus.RUNNING.value:
                return False
            j.status = JobStatus.FINALIZING.value
            self._touch(j)
            return True
        won, job = self.state.update_job(job_id, claim)
        if not won or job is None:
            return False
        try:
            result = self.finalizer.finalize(job, units)
        except Exception as exc:  # last-resort: never leave a job stuck in FINALIZING by a deterministic error
            log.exception("finalizer crashed")
            result = None
            err = f"finalization crashed: {type(exc).__name__}: {truncate(str(exc), 300)}"

        def finish(j: JobRecord) -> bool:
            if result is None:
                j.status, j.error = JobStatus.FAILED.value, err
            else:
                j.status, j.artifacts, j.summary, j.error = result.status, result.artifacts, result.summary, result.error
            self._touch(j)
            return True
        self.state.update_job(job_id, finish)
        log.info("job finalized", extra={"fields": {"job_id": job_id, "status": (result.status if result else "FAILED")}})
        return True
