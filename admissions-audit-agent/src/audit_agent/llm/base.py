"""Provider-neutral LLM types.

The audit logic depends only on ``LlmClient``; Amazon Bedrock is one
implementation. Tests use scripted fakes through the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

from ..deadline import Deadline


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0


@dataclass(frozen=True)
class ToolCallResult:
    tool_name: str
    arguments: Mapping[str, Any]
    usage: LlmUsage = field(default_factory=LlmUsage)
    stop_reason: str = "tool_use"


@dataclass
class UsageMeter:
    """Accumulates LLM usage for one request (single-threaded use)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0

    def add(self, usage: LlmUsage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.latency_ms += usage.latency_ms

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "llm_latency_ms": self.latency_ms,
        }


class LlmClient(Protocol):
    def invoke_tool(
        self,
        *,
        system_prompt: str,
        user_text: str,
        tool: ToolSpec,
        deadline: Deadline,
        max_output_tokens: Optional[int] = None,
        meter: Optional["UsageMeter"] = None,
    ) -> ToolCallResult:
        """Run one model call that must end in a call to ``tool``.

        Transport-level retries (throttling, timeouts, 5xx) happen inside; a
        response that is not a usable tool call raises ``LlmOutputError`` and the
        caller decides whether to ask again. Token usage is added to ``meter``
        for every model response, including rejected ones."""
        ...
