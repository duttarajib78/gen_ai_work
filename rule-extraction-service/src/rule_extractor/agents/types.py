"""Runner-agnostic task/result types so the pipeline can be tested without the SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Union


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool = False


ToolHandler = Callable[[dict], Union[ToolResult, Awaitable[ToolResult]]]


@dataclass(frozen=True)
class ToolSpec:
    """A custom in-process tool offered to the model (served by the SDK's in-process MCP server)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


@dataclass(frozen=True)
class ToolProfile:
    """The complete tool surface for a task. Anything not listed here is denied."""

    name: str
    builtin_tools: tuple[str, ...] = ()
    file_root: Path | None = None            # Read/Grep/Glob are confined to this directory
    web_allowed_hosts: tuple[str, ...] = ()  # WebFetch is confined to these hosts (https only)


@dataclass
class AgentTask:
    unit_id: str
    kind: str
    system_prompt: str
    user_prompt: str
    model: str
    profile: ToolProfile
    tools: list[ToolSpec]
    max_turns: int
    timeout_seconds: float
    max_budget_usd: float | None = None
    untrusted_text_in_prompt: bool = False


@dataclass
class AgentRunStats:
    turns: int = 0
    cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    stop_reason: str = "success"
    session_id: str | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    denied_tools: list[str] = field(default_factory=list)
    final_text: str = ""


class AgentRunner(Protocol):
    async def run(self, task: AgentTask) -> AgentRunStats: ...
