"""Per-request time budget.

SageMaker real-time endpoints cut a normal invocation at 60 seconds and a
streamed one at 8 minutes. Rather than let the platform return an opaque
timeout, every unit of slow work (an LLM attempt, a grouper call) asks the
deadline first, and the HTTP layer answers with a structured response before
the platform limit is reached.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .errors import DeadlineExceededError, RequestCancelledError


class Deadline:
    def __init__(self, budget_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self._budget = float(budget_s)
        self._cancelled = threading.Event()

    @property
    def budget_s(self) -> float:
        return self._budget

    def elapsed(self) -> float:
        return self._clock() - self._start

    def remaining(self) -> float:
        return max(0.0, self._budget - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def check(self, min_remaining: float = 0.0, *, what: Optional[str] = None) -> None:
        """Raise unless there is still at least ``min_remaining`` seconds left."""
        if self.cancelled:
            raise RequestCancelledError("request was cancelled before this step started")
        if self.remaining() <= 0.0 or self.remaining() < min_remaining:
            step = f" before starting {what}" if what else ""
            raise DeadlineExceededError(
                f"time budget of {self._budget:.0f}s exhausted{step}",
                details={"budget_s": self._budget, "elapsed_s": round(self.elapsed(), 1)},
            )
