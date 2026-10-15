"""Queue-consuming worker: claim -> heartbeat -> process -> complete/fail -> finalize.

Correctness rests on three properties:
* SQS delivery is at-least-once, so every unit is *claimed* with a compare-and-swap lease; duplicates are harmless.
* A crashed instance's unit becomes claimable again when its lease (and the SQS visibility timeout, extended in
  lock-step by the heartbeat) expires.
* On SIGTERM the worker stops receiving, cancels in-flight sessions and hands each unit straight back
  (``PENDING`` + visibility 0) so another instance resumes it within seconds, well inside SageMaker's 30 s grace.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..domain.jobs import JobRecord, UnitRecord, UnitStatus, WorkItem
from ..logging_setup import log_context
from ..pipeline.processor import UnitOutcome, UnitProcessor
from ..settings import Settings
from ..storage.interfaces import QueueMessage
from ..util import truncate
from .health import touch_heartbeat
from .janitor import Janitor
from .jobs import JobService, iso
from .metrics import MetricsSink, NullMetrics

log = logging.getLogger(__name__)
HEARTBEAT_FILE_INTERVAL = 10.0


class Worker:
    def __init__(self, *, settings: Settings, jobs: JobService, processor: UnitProcessor, metrics: MetricsSink | None = None,
                 janitor: Janitor | None = None, run_dir: Path | None = None, worker_id: str | None = None,
                 clock: Callable[[], float] = time.time, poll_wait: float = 5.0, idle_sleep: float = 0.5,
                 janitor_interval: float = 60.0, flush_interval: float = 60.0, lease_seconds: int | None = None,
                 heartbeat_interval: float | None = None) -> None:
        self._s, self._jobs, self._processor = settings, jobs, processor
        self._state, self._queue = jobs.state, jobs.queue
        self._metrics = metrics or NullMetrics()
        self._janitor, self._run_dir, self._clock = janitor, run_dir, clock
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._slots = settings.agent.max_concurrent_sessions
        self._lease = lease_seconds or settings.service.lease_seconds
        self._hb_interval = heartbeat_interval or max(1.0, self._lease / 3)
        self._max_claims = settings.agent.max_retries + 1
        self._poll_wait, self._idle_sleep = poll_wait, idle_sleep
        self._janitor_interval, self._flush_interval = janitor_interval, flush_interval
        self._grace = settings.service.shutdown_grace_seconds
        self._active: dict[int, asyncio.Task] = {}
        self._abort: dict[str, str] = {}

    # ------------------------------------------------------------------ main loop
    async def run(self, stop: asyncio.Event) -> None:
        aux = [asyncio.create_task(self._periodic(stop, HEARTBEAT_FILE_INTERVAL, self._beat, immediate=True)),
               asyncio.create_task(self._periodic(stop, self._flush_interval, self._publish_metrics))]
        if self._janitor is not None:
            aux.append(asyncio.create_task(self._periodic(stop, self._janitor_interval, self._sweep)))
        log.info("worker started", extra={"fields": {"worker_id": self.worker_id, "slots": self._slots}})
        try:
            while not stop.is_set():
                free = self._slots - len(self._active)
                if free <= 0:
                    await self._wait(stop, 0.1)
                    continue
                try:
                    messages = await asyncio.to_thread(self._queue.receive, free, self._poll_wait, self._lease)
                except Exception:
                    log.exception("queue receive failed")
                    await self._wait(stop, 5.0)
                    continue
                if stop.is_set():                     # shutting down: hand anything just received straight back
                    for m in messages:
                        await self._safe(self._queue.extend, m, 0)
                    break
                for m in messages:
                    task = asyncio.create_task(self._handle(m))
                    self._active[id(task)] = task
                    task.add_done_callback(lambda t: self._active.pop(id(t), None))
                if not messages:
                    await self._wait(stop, self._idle_sleep)
        finally:
            await self._drain()
            for t in aux:
                t.cancel()
            await asyncio.gather(*aux, return_exceptions=True)
            await self._safe(self._metrics.flush)
            log.info("worker stopped", extra={"fields": {"worker_id": self.worker_id}})

    async def _drain(self) -> None:
        tasks = list(self._active.values())
        for t in tasks:
            t.cancel()
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=self._grace)
            except asyncio.TimeoutError:
                log.warning("some units did not hand back within the grace period; leases will expire instead")

    @staticmethod
    async def _wait(stop: asyncio.Event, seconds: float) -> bool:
        """Sleep up to ``seconds``; True if ``stop`` was set meanwhile."""
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def _safe(self, fn: Callable[..., Any], *args: Any) -> None:
        try:
            await asyncio.to_thread(fn, *args)
        except Exception:
            log.exception("background call failed: %s", getattr(fn, "__name__", fn))

    async def _periodic(self, stop: asyncio.Event, interval: float, fn: Callable[[], Any], *, immediate: bool = False) -> None:
        if not immediate and await self._wait(stop, interval):
            return
        while not stop.is_set():
            await self._safe(fn)
            if await self._wait(stop, interval):
                return

    def _beat(self) -> None:
        if self._run_dir is not None:
            touch_heartbeat(self._run_dir)

    def _sweep(self) -> None:
        if self._janitor is not None:
            self._janitor.sweep_once()

    def _publish_metrics(self) -> None:
        self._metrics.emit("ActiveUnits", len(self._active))
        self._metrics.emit("WorkerSlotUtilization", 100.0 * len(self._active) / max(1, self._slots), "Percent")
        try:
            visible, in_flight = self._queue.depth()
            self._metrics.emit("QueueVisible", visible)
            self._metrics.emit("QueueInFlight", in_flight)
        except Exception:
            pass
        self._metrics.flush()

    # ----------------------------------------------------------------- one message
    async def _handle(self, msg: QueueMessage) -> None:
        try:
            item = WorkItem.from_json(msg.body)
        except (ValueError, TypeError):
            log.error("poison message ignored (will reach the DLQ after the redrive limit)", extra={"fields": {"message_id": msg.message_id}})
            self._metrics.emit("PoisonMessages", 1)
            await self._safe(self._queue.extend, msg, 900)
            return
        with log_context(job_id=item.job_id, unit_id=item.unit_id, worker=self.worker_id):
            try:
                await self._process(msg, item)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("unexpected error while handling a unit")
                await self._safe(self._queue.extend, msg, 60)

    async def _ack_and_finalize(self, msg: QueueMessage, job_id: str) -> None:
        await self._safe(self._queue.ack, msg)
        await self._safe(self._jobs.maybe_finalize, job_id)

    async def _finish_unit(self, item: WorkItem, status: UnitStatus, error: str | None, *, outcome: UnitOutcome | None = None,
                           cost: float = 0.0, owner_only: bool = False) -> bool:
        now = self._clock()

        def mutate(u: UnitRecord) -> bool:
            if u.is_terminal or (owner_only and u.lease_owner != self.worker_id):
                return False
            u.status, u.error, u.finished_at, u.lease_owner, u.lease_expires = status.value, error, iso(now), None, 0.0
            u.cost_usd = round(u.cost_usd + cost, 6)
            if outcome is not None:
                u.rules_key, u.rules_count, u.rejected_count = outcome.rules_key, outcome.rules_count, outcome.rejected_count
                u.cost_usd, u.turns, u.usage = round(outcome.cost_usd, 6), outcome.turns, outcome.usage
                u.stop_reason, u.duration_s, u.model = outcome.stop_reason, outcome.duration_s, outcome.model
            return True
        applied, _ = await asyncio.to_thread(self._state.update_unit, item.job_id, item.unit_id, mutate)
        return applied

    async def _give_back(self, msg: QueueMessage, item: WorkItem) -> None:
        def mutate(u: UnitRecord) -> bool:
            if u.status != UnitStatus.RUNNING.value or u.lease_owner != self.worker_id:
                return False
            u.status, u.lease_owner, u.lease_expires, u.attempts = UnitStatus.PENDING.value, None, 0.0, max(0, u.attempts - 1)
            return True
        await self._safe(self._state.update_unit, item.job_id, item.unit_id, mutate)
        await self._safe(self._queue.extend, msg, 0)

    async def _process(self, msg: QueueMessage, item: WorkItem) -> None:
        job: JobRecord | None = await asyncio.to_thread(self._state.get_job, item.job_id)
        if job is None:                                         # job expired (TTL): orphaned message
            await self._safe(self._queue.ack, msg)
            return
        now = self._clock()
        budget = job.options.get("max_cost_usd")
        refusal: tuple[UnitStatus, str] | None = None
        if job.cancel_requested:
            refusal = (UnitStatus.CANCELLED, "cancelled")
        elif job.is_terminal:
            refusal = (UnitStatus.CANCELLED, f"job already {job.status}")
        elif job.deadline_epoch > 0 and now > job.deadline_epoch:
            refusal = (UnitStatus.FAILED, "job deadline exceeded")
        elif budget and job.cost_usd >= budget:
            refusal = (UnitStatus.CANCELLED, f"job budget of ${budget} reached")
        if refusal:
            await self._finish_unit(item, refusal[0], refusal[1])
            await self._ack_and_finalize(msg, item.job_id)
            return

        def claim(u: UnitRecord) -> bool:
            if u.is_terminal:
                return False
            if u.status == UnitStatus.RUNNING.value and u.lease_owner != self.worker_id and u.lease_expires > now:
                return False
            if u.attempts >= self._max_claims:
                u.status, u.lease_owner, u.finished_at = UnitStatus.FAILED.value, None, iso(now)
                u.error = f"gave up after {u.attempts} claims (crashes or redeliveries)"
                return True
            u.status, u.attempts, u.lease_owner, u.lease_expires = UnitStatus.RUNNING.value, u.attempts + 1, self.worker_id, now + self._lease
            u.started_at = u.started_at or iso(now)
            return True
        applied, unit = await asyncio.to_thread(self._state.update_unit, item.job_id, item.unit_id, claim)
        if unit is None:                                        # unknown unit: nothing to do
            await self._safe(self._queue.ack, msg)
            return
        if not applied:
            if unit.is_terminal:                                # duplicate delivery of finished work
                await self._safe(self._queue.ack, msg)
            else:                                               # another worker holds a live lease: look again later
                await self._safe(self._queue.extend, msg, min(self._lease, 120))
            return
        if unit.status == UnitStatus.FAILED.value:              # claim budget exhausted
            self._metrics.emit("UnitsFailed", 1)
            await self._ack_and_finalize(msg, item.job_id)
            return

        key = f"{item.job_id}/{item.unit_id}"
        proc_task = asyncio.create_task(self._processor.process(job, unit))
        hb_stop = asyncio.Event()
        hb_task = asyncio.create_task(self._lease_heartbeat(msg, item, hb_stop, proc_task, key))
        try:
            outcome = await proc_task
        except asyncio.CancelledError:
            reason = self._abort.pop(key, None)
            if reason is None:                                  # worker shutdown
                await asyncio.shield(self._give_back(msg, item))
                raise
            status = UnitStatus.CANCELLED if reason == "cancelled" else UnitStatus.FAILED
            await self._finish_unit(item, status, f"aborted: {reason}", owner_only=True)
            await self._ack_and_finalize(msg, item.job_id)
            return
        except Exception as exc:
            cost = float(getattr(exc, "cost_usd", 0.0) or 0.0)
            log.warning("unit failed: %s: %s", type(exc).__name__, truncate(str(exc), 200))
            await self._finish_unit(item, UnitStatus.FAILED, truncate(f"{type(exc).__name__}: {exc}", 500), cost=cost, owner_only=True)
            await self._safe(self._jobs.add_cost, item.job_id, cost)
            self._metrics.emit("UnitsFailed", 1)
            await self._ack_and_finalize(msg, item.job_id)
            return
        finally:
            hb_stop.set()
            await asyncio.gather(hb_task, return_exceptions=True)

        await self._finish_unit(item, UnitStatus.SUCCEEDED, None, outcome=outcome, owner_only=False)
        await self._safe(self._jobs.add_cost, item.job_id, outcome.cost_usd)
        for name, value, unit_name in (("UnitsSucceeded", 1, "Count"), ("UnitDurationSeconds", outcome.duration_s, "Seconds"),
                                       ("UnitCostUSD", outcome.cost_usd, "None"), ("RulesEmitted", outcome.rules_count, "Count"),
                                       ("TokensInput", float(outcome.usage.get("input_tokens", 0) or 0), "Count"),
                                       ("TokensOutput", float(outcome.usage.get("output_tokens", 0) or 0), "Count")):
            self._metrics.emit(name, value, unit_name)
        await self._ack_and_finalize(msg, item.job_id)

    async def _lease_heartbeat(self, msg: QueueMessage, item: WorkItem, stop: asyncio.Event, proc_task: asyncio.Task, key: str) -> None:
        while not await self._wait(stop, self._hb_interval):
            now = self._clock()

            def extend(u: UnitRecord) -> bool:
                if u.lease_owner != self.worker_id or u.status != UnitStatus.RUNNING.value:
                    return False
                u.lease_expires = now + self._lease
                return True
            try:
                await asyncio.to_thread(self._state.update_unit, item.job_id, item.unit_id, extend)
                await asyncio.to_thread(self._queue.extend, msg, self._lease)
                job = await asyncio.to_thread(self._state.get_job, item.job_id)
            except Exception:
                log.warning("lease heartbeat failed; will retry", exc_info=True)
                continue
            if job is None:
                continue
            reason = "cancelled" if job.cancel_requested else ("deadline exceeded" if job.deadline_epoch > 0 and now > job.deadline_epoch else None)
            if reason:
                self._abort[key] = reason
                proc_task.cancel()
                return
