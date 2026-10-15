"""Periodic repair of states a crashed instance can leave behind. Every action is an idempotent compare-and-swap."""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Callable

from ..domain.jobs import JobRecord, JobStatus, UnitRecord, UnitStatus
from ..settings import Settings
from .jobs import JobService, iso

log = logging.getLogger(__name__)
SUBMIT_STALE_SECONDS = 300
FINALIZE_STALE_SECONDS = 900


class Janitor:
    def __init__(self, settings: Settings, jobs: JobService, *, clock: Callable[[], float] = time.time) -> None:
        self._s, self._jobs, self._state, self._clock = settings, jobs, jobs.state, clock

    def _fail_unit(self, job_id: str, unit: UnitRecord, status: UnitStatus, reason: str) -> bool:
        now = self._clock()

        def mutate(u: UnitRecord) -> bool:
            if u.is_terminal:
                return False
            u.status, u.error, u.finished_at, u.lease_owner = status.value, reason, iso(now), None
            return True
        applied, _ = self._state.update_unit(job_id, unit.unit_id, mutate)
        return applied

    def sweep_once(self) -> dict[str, int]:
        now, stats = self._clock(), Counter()
        grace = self._s.service.lease_seconds
        max_claims = self._s.agent.max_retries + 1

        for job in self._state.list_jobs_by_status(JobStatus.SUBMITTING.value):
            if now - job.updated_epoch > SUBMIT_STALE_SECONDS:
                self._jobs.resume(job.job_id)
                self._jobs.activate(job.job_id)
                stats["resubmitted"] += 1

        for job in self._state.list_jobs_by_status(JobStatus.RUNNING.value):
            units = self._state.list_units(job.job_id)
            aborting = job.cancel_requested or (job.deadline_epoch > 0 and now > job.deadline_epoch)
            for unit in units:
                if unit.is_terminal:
                    continue
                lease_dead = unit.status == UnitStatus.RUNNING.value and unit.lease_expires + grace < now
                if aborting and (unit.status == UnitStatus.PENDING.value or lease_dead):
                    if job.cancel_requested:
                        done = self._fail_unit(job.job_id, unit, UnitStatus.CANCELLED, "cancelled")
                    else:
                        done = self._fail_unit(job.job_id, unit, UnitStatus.FAILED, "job deadline exceeded")
                    stats["aborted_units"] += int(done)
                elif lease_dead and unit.attempts >= max_claims:
                    stats["lease_exhausted"] += int(self._fail_unit(
                        job.job_id, unit, UnitStatus.FAILED, f"lease expired after {unit.attempts} attempts (instance crash loop?)"))
            if self._jobs.maybe_finalize(job.job_id):
                stats["finalized"] += 1

        for job in self._state.list_jobs_by_status(JobStatus.FINALIZING.value):
            if now - job.updated_epoch > FINALIZE_STALE_SECONDS:
                def reopen(j: JobRecord) -> bool:
                    if j.status != JobStatus.FINALIZING.value:
                        return False
                    j.status = JobStatus.RUNNING.value
                    j.updated_epoch = now
                    return True
                self._state.update_job(job.job_id, reopen)
                stats["finalize_retried"] += int(self._jobs.maybe_finalize(job.job_id))
        if stats:
            log.info("janitor sweep", extra={"fields": dict(stats)})
        return dict(stats)
