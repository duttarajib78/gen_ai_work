"""Request / result / error types shared by every backend."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Optional, Protocol

from dx_support.constants import DEFAULT_THINKING


@dataclass
class LLMRequest:
    stage: str                      # "brief" | "route" | "judge" (logs, cost attribution, the stub backend)
    system_prompt: str
    prompt: str
    schema: dict                    # JSON Schema of the structured answer
    model: str                      # catalog alias, or an explicit Bedrock model / inference-profile id
    thinking: str = DEFAULT_THINKING
    max_budget_usd: float = 0.50    # a call whose WORST-CASE cost exceeds this is refused before it is sent
    max_tokens: int = 8000          # output cap, thinking included
    cache_prefix_chars: int = 0     # leading characters of `prompt` that are constant across calls (prompt-cache candidate); 0 = none
    label: str = ""                 # e.g. "H25/achi/route-L1" -- logs only
    category: str = ""
    family: str = ""
    meta: dict = field(default_factory=dict)   # structured side-info for the stub backend; ignored by the real one


@dataclass
class LLMResult:
    structured: Optional[dict]
    cost_usd: Optional[float]
    diag: dict = field(default_factory=dict)
    text: str = ""


class LLMCallError(Exception):
    """A failed model call. `cost_usd` is what the failed attempt still cost, so the cost cap never undercounts.

    kind: "transient" (retry) | "throttle" (retry with longer back-off; shrinks concurrency) |
          "budget" (retrying cannot help; the call is skipped) | "fatal" (stop the whole run: bad credentials, unknown model, bad request shape).
    """

    def __init__(self, message: str, cost_usd: Optional[float] = None, kind: str = "transient"):
        super().__init__(message)
        self.cost_usd = cost_usd
        self.kind = kind


class StopRequested(Exception):
    """Raised instead of making a new call once the run has been told to stop (cost cap reached, fatal error, shutdown)."""


class LLMBackend(Protocol):
    name: str

    def complete(self, req: LLMRequest) -> Awaitable[LLMResult]: ...


def extract_json_fallback(text: str) -> Optional[dict]:
    """Last-resort parse if no tool call arrived: the outermost {...} in the free text."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj: Any = json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
