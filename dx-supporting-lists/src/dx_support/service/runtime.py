"""The long-lived state of one serving process.

One process owns ONE asyncio event loop (in a background thread) that runs every model-calling job, so:
  * Bedrock concurrency is bounded per process by a single shared limiter, however many HTTP requests arrive;
  * a rolling per-hour spend budget acts as a circuit breaker for the whole instance;
  * jobs keep running after the HTTP response that started them has gone (a 202 or a dropped stream): they are resumable and idempotent, so the
    caller simply asks again and is answered from what was stored.

Duplicate work is avoided with a per-category lock: if two requests need the same category, the second waits, then finds it already done and
spends nothing. Across INSTANCES there is no lock (objects are keyed by content, so concurrent writers produce equivalent objects); a rare
duplicate computation costs money but never correctness. See the README, "Known limitations".
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import socket
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional

from dx_support import __version__
from dx_support.constants import DEFAULT_FAMILIES, FAMILIES, SUPPORT_TIERS
from dx_support.indexing.universe import Universe, load_universe
from dx_support.llm.bedrock import BedrockBackend
from dx_support.llm.fake import FakeBackend
from dx_support.llm.gate import AdaptiveLimiter, LLMGate, RollingBudget
from dx_support.llm.types import StopRequested
from dx_support.pipeline import prompts as P
from dx_support.pipeline.context import RunContext
from dx_support.pipeline.fingerprint import run_fingerprint
from dx_support.pipeline.stages import families_done, process_category
from dx_support.service.results import assemble_category
from dx_support.settings import Settings
from dx_support.storage.checkpoint import Checkpoint
from dx_support.storage.objectstore import ObjectStore, namespace, open_store

logger = logging.getLogger(__name__)


@dataclass
class JobProgress:
    category: str
    families: tuple[str, ...]
    state: str = "queued"                      # queued -> running -> done
    outcome: str = ""                          # complete | partial | cached | stopped | failed | ...
    error: str = ""
    gate: Optional[LLMGate] = None
    started: float = field(default_factory=time.monotonic)
    finished: float = 0.0

    def snapshot(self) -> dict:
        g = self.gate
        return {"state": self.state, "outcome": self.outcome or None, "llm_calls": g.n_calls if g else 0,
                "cost_usd": round(g.total_cost, 4) if g else 0.0, "elapsed_s": round((self.finished or time.monotonic()) - self.started, 1)}


@dataclass
class JobHandle:
    progress: JobProgress
    future: Future


class ServiceRuntime:
    def __init__(self, settings: Settings, *, universe: Optional[Universe] = None, backend: Any = None, base_store: Optional[ObjectStore] = None):
        self.settings = settings
        self._universe, self._backend, self._base_store = universe, backend, base_store
        self.ready = False
        self.init_error = ""
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._active: dict[int, JobHandle] = {}
        self.counters: Counter = Counter()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._cat_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "ServiceRuntime":
        """Blocking, and deliberately loud: any failure here (missing data, no store access, bad model name) must stop the process at start-up."""
        s = self.settings
        try:
            self.universe = self._universe or load_universe(s.data_dir, s.include_coarse_prostheses)
            self.backend = self._backend or (FakeBackend() if s.backend == "fake" else BedrockBackend(s))
            self.fingerprint = run_fingerprint(s, self.universe, mode=s.mode, backend=s.backend)
            base = self._base_store or open_store(s.store_uri, region=s.aws_region, kms_key_id=s.store_kms_key_id)
            self.store = namespace(base, "serve", self.fingerprint)
            self._probe_store()
            self.ckpt = Checkpoint(self.store, io_workers=16)
            self.limiter = AdaptiveLimiter(s.llm_concurrency)
            self.budget = RollingBudget(s.max_cost_usd_per_hour)
            self.leaf_map = self.universe.leaf_to_category()
            self.models_in_force = (self.backend.describe(s.stage_models()) if hasattr(self.backend, "describe")
                                    else {stage: {"alias": name, "model_id": name} for stage, name in s.stage_models().items()})
            placeholder = LLMGate(self.backend, self.limiter, 0.0)
            self.base_ctx = RunContext(categories=self.universe.categories, indices=self.universe.indices, gate=placeholder, ckpt=self.ckpt, mode=s.mode,
                                       models=s.stage_models(), thinking=s.thinking, include_context_chapters=s.include_context_chapters,
                                       lexical_rescue=s.lexical_rescue, max_budget_usd_per_call=s.max_budget_usd_per_call, max_tokens=dict(s.max_tokens),
                                       fingerprint=self.fingerprint, vocab=P.vocabulary_hint(self.universe.indices))
            self._start_loop()
        except Exception as e:                       # noqa: BLE001
            self.init_error = f"{type(e).__name__}: {e}"[:300]
            logger.error("service initialisation failed: %s", self.init_error)
            raise
        self.ready = True
        logger.info("service ready", extra={"fingerprint": self.fingerprint, "backend": s.backend, "region": s.aws_region, "version": __version__,
                                            "models": {k: v.get("model_id", v.get("alias")) for k, v in self.models_in_force.items()},
                                            "categories": len(self.universe.categories), "store": self.store.uri})
        return self

    def _probe_store(self) -> None:
        """Fails fast if the store is unreachable or not writable (a missing S3 permission should stop start-up, not the first request)."""
        key = "probe/" + hashlib.sha1(socket.gethostname().encode()).hexdigest()[:12] + ".json"      # noqa: S324 - not a security use
        self.store.put(key, b"{}")
        if self.store.get(key) != b"{}":
            raise RuntimeError("the result store did not return what was just written")

    def _start_loop(self) -> None:
        started = threading.Event()

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.call_soon(started.set)
            try:
                loop.run_forever()
            finally:
                loop.close()
        self._thread = threading.Thread(target=run, name="dx-loop", daemon=True)
        self._thread.start()
        if not started.wait(10):
            raise RuntimeError("the event-loop thread did not start")

    def shutdown(self, timeout: float = 20.0) -> None:
        self.ready = False
        with self._lock:
            handles = list(self._active.values())
        for h in handles:
            if h.progress.gate is not None:
                h.progress.gate.request_stop("service shutting down")       # no NEW model calls; calls in flight finish
        deadline = time.monotonic() + timeout
        for h in handles:
            try:
                h.future.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:                                               # noqa: BLE001, S110 - best effort while shutting down
                pass
        try:
            self.ckpt.flush_run_log()
        except Exception:                                                   # noqa: BLE001, S110
            pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        close = getattr(self.backend, "close", None)
        if close:
            close()

    # ------------------------------------------------------------------ request-thread API
    def refresh_status(self, category: str) -> dict:
        return self.ckpt.refresh_status(category)

    def is_cached(self, category: str, families: tuple[str, ...]) -> bool:
        return set(families) <= families_done(self.refresh_status(category), self.settings.mode)

    def assemble(self, category: str, families, merge_tiers, prosthesis_extract: str, include_audit: bool) -> dict:
        self.refresh_status(category)
        return assemble_category(ckpt=self.ckpt, universe=self.universe, category=category, families=families, merge_tiers=merge_tiers,
                                 prosthesis_extract=prosthesis_extract, include_audit=include_audit, mode=self.settings.mode)

    def submit(self, category: str, families: tuple[str, ...]) -> JobHandle:
        assert self._loop is not None
        progress = JobProgress(category, tuple(families))
        fut = asyncio.run_coroutine_threadsafe(self._job(progress), self._loop)
        handle = JobHandle(progress, fut)
        with self._lock:
            self._active[id(handle)] = handle
            self.counters["jobs_submitted"] += 1
        fut.add_done_callback(lambda f, h=handle: self._job_done(h, f))
        return handle

    def _job_done(self, handle: JobHandle, fut: Future) -> None:
        with self._lock:
            self._active.pop(id(handle), None)
            self.counters[f"jobs_{handle.progress.outcome or 'failed'}"] += 1
        if fut.cancelled() or fut.exception() is not None:
            logger.error("job ended abnormally", extra={"category": handle.progress.category})

    # ------------------------------------------------------------------ the job (runs on the loop thread)
    async def _job(self, progress: JobProgress) -> dict:
        cat, fams, s = progress.category, progress.families, self.settings
        lock = self._cat_locks.setdefault(cat, asyncio.Lock())
        async with lock:                                                    # one computation per category at a time in this process
            progress.state = "running"
            try:
                await self.ckpt.ahydrate(cat, force=True)                   # pick up anything another request or instance finished meanwhile
                todo = [f for f in fams if f not in families_done(self.ckpt.status.get(cat), s.mode)]
                if not todo:
                    progress.outcome = "cached"
                else:
                    gate = LLMGate(self.backend, self.limiter, s.max_cost_usd_per_category, run_log=self.ckpt.log_run, backoff_base_s=1.5,
                                   max_retries=s.max_call_retries, max_throttle_retries=s.max_throttle_retries, budget=self.budget,
                                   call_timeout_s=s.llm_call_timeout_s + 30)
                    progress.gate = gate
                    ctx = dataclasses.replace(self.base_ctx, gate=gate, families=tuple(todo), tier_counts={t: 0 for t in SUPPORT_TIERS})
                    progress.outcome = await process_category(ctx, cat)
                    if gate.fatal_error:
                        progress.error = "model call failed: " + gate.fatal_error[:150]
            except StopRequested as e:
                progress.outcome, progress.error = "stopped", str(e)[:150]
            except asyncio.CancelledError:
                raise
            except Exception as e:                                          # noqa: BLE001
                progress.outcome, progress.error = "failed", type(e).__name__
        progress.state = "done"
        progress.finished = time.monotonic()
        return {"outcome": progress.outcome, "error": progress.error}

    # ------------------------------------------------------------------ introspection
    def snapshot(self) -> dict:
        s = self.settings
        with self._lock:
            active = len(self._active)
            counters = dict(self.counters)
        return {
            "version": __version__, "fingerprint": getattr(self, "fingerprint", None), "ready": self.ready, "backend": s.backend, "region": s.aws_region,
            "mode": s.mode, "models": getattr(self, "models_in_force", {}), "thinking": s.thinking, "families": list(FAMILIES), "default_families": list(DEFAULT_FAMILIES),
            "limits": {"max_categories_per_request": s.max_categories_per_request, "max_concurrent_requests": s.max_concurrent_requests,
                       "sync_deadline_s": s.sync_deadline_s, "stream_max_s": s.stream_max_s, "compute_enabled": s.allow_compute},
            "load": {"active_jobs": active, "model_concurrency_limit": getattr(self, "limiter", None) and self.limiter.limit,
                     "model_calls_in_flight": getattr(self, "limiter", None) and self.limiter.in_flight,
                     "spend_last_hour_usd": round(self.budget.spent(), 4) if hasattr(self, "budget") else None,
                     "hourly_budget_usd": s.max_cost_usd_per_hour, "counters": counters},
            "uptime_s": int(time.time() - self.started_at),
        }
