"""The single choke point for model calls.

LLMGate
  * bounds in-flight calls through a limiter that can be SHARED by many gates (the service runs one gate per category job and one limiter per
    process, so Bedrock concurrency is bounded no matter how many requests arrive);
  * retries transient failures with jittered back-off, throttling with a longer one, and shrinks concurrency while Bedrock is throttling
    (AIMD: halve on a throttle, +1 after a run of successes);
  * adds the cost of EVERY attempt, failed ones included, to the running total and stops launching new calls once the cap is reached. Calls
    already in flight finish, so the worst-case overshoot is about `concurrency x max_budget_usd_per_call`;
  * also honours an optional process-wide rolling budget (a circuit breaker for the service: spend per hour per instance);
  * stops the whole run on a fatal error (bad credentials, unknown model, a request the model rejects) instead of burning through every
    remaining call to fail the same way;
  * appends one record per attempt to the run log (no prompts, no secrets).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import threading
import time
from collections import deque
from typing import Callable, Optional, Union

from dx_support.llm.types import LLMBackend, LLMCallError, LLMRequest, LLMResult, StopRequested

logger = logging.getLogger(__name__)
call_logger = logging.getLogger("dx_support.calls")

MAX_CALL_RETRIES = 2
MAX_THROTTLE_RETRIES = 6


class AdaptiveLimiter:
    """Concurrency limit with additive-increase / multiplicative-decrease. Must be used from ONE event loop."""

    def __init__(self, max_concurrency: int, min_concurrency: int = 1, recover_after: int = 20, cooldown_s: float = 5.0):
        self.max = max(1, int(max_concurrency))
        self.min = max(1, min(int(min_concurrency), self.max))
        self.limit = self.max
        self.recover_after = recover_after
        self.cooldown_s = cooldown_s
        self.in_flight = 0
        self.n_throttles = 0
        self._waiters: deque[asyncio.Future] = deque()
        self._successes = 0
        self._last_cut = float("-inf")

    async def acquire(self) -> None:
        while self.in_flight >= self.limit:
            fut = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await fut
            except asyncio.CancelledError:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(fut)
                raise
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1
        self._wake()

    def _wake(self) -> None:
        for _ in range(max(0, self.limit - self.in_flight)):
            while self._waiters:
                fut = self._waiters.popleft()
                if not fut.done():
                    fut.set_result(None)
                    break

    @contextlib.asynccontextmanager
    async def slot(self):
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def on_throttle(self) -> None:
        self.n_throttles += 1
        self._successes = 0
        now = time.monotonic()
        if now - self._last_cut >= self.cooldown_s and self.limit > self.min:
            self._last_cut = now
            self.limit = max(self.min, self.limit // 2)
            logger.warning("Bedrock is throttling: model-call concurrency reduced to %d (max %d)", self.limit, self.max)

    def on_success(self) -> None:
        if self.limit >= self.max:
            return
        self._successes += 1
        if self._successes >= self.recover_after:
            self._successes = 0
            self.limit += 1
            self._wake()


class RollingBudget:
    """Spend over a sliding window (default one hour). Thread-safe."""

    def __init__(self, cap_usd: float, window_s: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        self.cap_usd = cap_usd
        self.window_s = window_s
        self._clock = clock
        self._events: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window_s:
            self._events.popleft()

    def charge(self, cost: float) -> None:
        if cost:
            with self._lock:
                now = self._clock()
                self._purge(now)
                self._events.append((now, cost))

    def spent(self) -> float:
        with self._lock:
            self._purge(self._clock())
            return sum(c for _, c in self._events)

    @property
    def exhausted(self) -> bool:
        return self.spent() >= self.cap_usd


class LLMGate:
    def __init__(self, backend: LLMBackend, concurrency: Union[int, AdaptiveLimiter], max_total_cost_usd: float,
                 run_log: Optional[Callable[[dict], None]] = None, backoff_base_s: float = 1.5, max_retries: int = MAX_CALL_RETRIES,
                 max_throttle_retries: int = MAX_THROTTLE_RETRIES, budget: Optional[RollingBudget] = None,
                 call_timeout_s: Optional[float] = None):
        self.backend = backend
        self.limiter = concurrency if isinstance(concurrency, AdaptiveLimiter) else AdaptiveLimiter(concurrency)
        self.max_total_cost_usd = max_total_cost_usd
        self.run_log = run_log
        self.backoff_base_s = backoff_base_s
        self.max_retries = max_retries
        self.max_throttle_retries = max_throttle_retries
        self.budget = budget
        self.call_timeout_s = call_timeout_s
        self.total_cost = 0.0
        self.stop = False
        self.stop_reason = ""
        self.fatal_error = ""
        self.n_calls = 0
        self.n_failed_attempts = 0
        self.by_stage: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def request_stop(self, reason: str) -> None:
        if not self.stop:
            self.stop, self.stop_reason = True, reason
            logger.warning("Stopping new work: %s", reason)

    @property
    def cost_cap_hit(self) -> bool:
        return self.total_cost >= self.max_total_cost_usd

    @property
    def budget_exhausted(self) -> bool:
        return bool(self.budget and self.budget.exhausted)

    def _check_stop(self) -> None:
        if not self.stop and self.budget_exhausted:
            self.request_stop(f"the rolling budget of ${self.budget.cap_usd:,.2f} per {self.budget.window_s / 3600:g} h is used up")   # type: ignore[union-attr]
        if self.stop:
            raise StopRequested(self.stop_reason)

    def _account(self, req: LLMRequest, cost: Optional[float], ok: bool, diag: dict, error: str = "") -> None:
        c = cost or 0.0
        self.total_cost += c
        if self.budget:
            self.budget.charge(c)
        st = self.by_stage.setdefault(req.stage, {"calls": 0, "failed": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0})
        st["calls"] += 1
        st["cost_usd"] += c
        if not ok:
            st["failed"] += 1
            self.n_failed_attempts += 1
        st["input_tokens"] += diag.get("input_tokens") or 0
        st["output_tokens"] += diag.get("output_tokens") or 0
        if self.run_log:
            self.run_log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": req.stage, "category": req.category, "family": req.family, "label": req.label,
                "model": req.model, "thinking": req.thinking, "ok": ok, "cost_usd": c, "input_tokens": diag.get("input_tokens"),
                "output_tokens": diag.get("output_tokens"), "thinking_tokens": diag.get("thinking_tokens"),
                "cache_read_tokens": diag.get("cache_read_tokens"), "cache_creation_tokens": diag.get("cache_creation_tokens"),
                "duration_ms": diag.get("duration_ms"), "stop_reason": diag.get("stop_reason"), "error": error[:300],
            })
        if call_logger.isEnabledFor(logging.INFO):
            call_logger.info("model call", extra={"stage": req.stage, "label": req.label, "ok": ok, "cost_usd": round(c, 5),
                                                  "input_tokens": diag.get("input_tokens"), "output_tokens": diag.get("output_tokens"),
                                                  "duration_ms": diag.get("duration_ms"), "error": error[:120] if error else None})
        if self.cost_cap_hit:
            self.request_stop(f"cumulative cost ${self.total_cost:.2f} reached the cap ${self.max_total_cost_usd:.2f}")

    async def _sleep(self, attempt: int, throttle: bool) -> None:
        if throttle:
            delay = min(30.0, 2.0 * (2 ** (attempt - 1))) * (self.backoff_base_s / 1.5)
        else:
            delay = self.backoff_base_s * (2 ** (attempt - 1))
        await asyncio.sleep(delay + random.uniform(0, 0.25 * max(delay, 0.0)))

    # ------------------------------------------------------------------
    async def call(self, req: LLMRequest) -> LLMResult:
        self._check_stop()
        async with self.limiter.slot():
            attempt = throttle_attempt = 0
            while True:
                self._check_stop()
                self.n_calls += 1
                try:
                    coro = self.backend.complete(req)
                    res = await (asyncio.wait_for(coro, self.call_timeout_s) if self.call_timeout_s else coro)
                except LLMCallError as e:
                    self._account(req, e.cost_usd, False, {}, str(e))
                    if e.kind == "fatal":
                        self.fatal_error = str(e)
                        self.request_stop(f"fatal error: {e}")
                        raise
                    if e.kind == "budget":
                        raise
                    if e.kind == "throttle":
                        self.limiter.on_throttle()
                        if throttle_attempt >= self.max_throttle_retries:
                            raise
                        throttle_attempt += 1
                        await self._sleep(throttle_attempt, True)
                        continue
                    if attempt >= self.max_retries:
                        raise
                    attempt += 1
                    await self._sleep(attempt, False)
                    continue
                except (StopRequested, asyncio.CancelledError):
                    raise
                except asyncio.TimeoutError:
                    self._account(req, None, False, {}, "call timed out")
                    if attempt >= self.max_retries:
                        raise LLMCallError("call timed out", kind="transient") from None
                    attempt += 1
                    await self._sleep(attempt, False)
                    continue
                except Exception as e:                      # noqa: BLE001
                    # Not a model-call failure: a configuration problem (a missing dependency, an option refused) or a bug. Retrying cannot fix
                    # it and every remaining call would fail the same way -- stop the whole run (fail closed).
                    msg = f"{type(e).__name__}: {e}"
                    self._account(req, None, False, {}, msg)
                    self.fatal_error = msg
                    self.request_stop(f"fatal error: {msg}")
                    raise LLMCallError(msg, kind="fatal") from e
                if res.structured is None:
                    # A reply with no usable structured output is a failed attempt (it still cost money): retry.
                    self._account(req, res.cost_usd, False, res.diag, "no structured output")
                    if attempt >= self.max_retries:
                        raise LLMCallError("no structured output after retries", cost_usd=None, kind="transient")
                    attempt += 1
                    await self._sleep(attempt, False)
                    continue
                self._account(req, res.cost_usd, True, res.diag)
                self.limiter.on_success()
                return res
