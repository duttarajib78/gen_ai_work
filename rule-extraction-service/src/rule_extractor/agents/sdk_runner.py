"""Claude Agent SDK runner for Claude on Amazon Bedrock.

One ``query()`` per unit of work (stateless, trivially retryable, embarrassingly parallel). Security posture:

* ``tools=`` restricts the built-ins; ``disallowed_tools`` removes every other dangerous built-in;
  ``permission_mode="default"`` plus a fail-closed ``can_use_tool`` gate handles anything not pre-approved
  (``bypassPermissions`` is never used). Only the in-process ``emit_rule`` tool is in ``allowed_tools`` so the
  gate is never shadowed for file/web tools.
* ``verbatim_prompts=True`` (SDK >= 0.2.158): untrusted document text inlined in a prompt cannot trigger ``@path``
  file expansion or slash-command dispatch.
* Options that carry security meaning are REQUIRED: if the installed SDK lacks one, we refuse to run (fail closed)
  instead of silently dropping it.
* Every session gets a private, throw-away config/HOME directory (no transcript or document text left behind).
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import logging
import re
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator

from ..errors import AgentError, ConfigError
from ..llm.bedrock import build_sdk_env
from ..logging_setup import redact
from ..settings import Settings
from .permissions import SERVER_NAME, deny_list_for, make_sdk_gate, mcp_tool_name
from .types import AgentRunStats, AgentTask, ToolResult, ToolSpec

log = logging.getLogger(__name__)

REQUIRED_OPTION_FIELDS = frozenset({
    "model", "system_prompt", "mcp_servers", "tools", "allowed_tools", "disallowed_tools",
    "permission_mode", "can_use_tool", "max_turns", "cwd", "env", "setting_sources",
})
_REQUIRED_SDK_ATTRS = ("ClaudeAgentOptions", "query", "tool", "create_sdk_mcp_server",
                       "PermissionResultAllow", "PermissionResultDeny")
_TRANSIENT = re.compile(
    r"(?i)(throttl|rate.?limit|too many requests|\b429\b|overloaded|\b529\b|timed? ?out|timeout|econnreset|etimedout|"
    r"temporar|service unavailable|\b50[0234]\b|internal server error|capacity|connection (reset|refused|error)|"
    r"model is currently|try again)")
_TRANSIENT_EXCEPTIONS = ("CLIConnectionError", "ProcessError", "CLIJSONDecodeError", "ResultError")
_NON_TRANSIENT_EXCEPTIONS = ("CLINotFoundError",)


async def _prompt_stream(text: str) -> AsyncIterator[dict[str, Any]]:
    """Streaming-input form: works with ``can_use_tool`` and in-process MCP servers on every SDK version."""
    yield {"type": "user", "message": {"role": "user", "content": text}, "parent_tool_use_id": None, "session_id": "default"}


class ClaudeSdkRunner:
    def __init__(self, settings: Settings, *, sdk: Any = None, work_root: Path | str | None = None) -> None:
        self._settings = settings
        self._sdk = sdk
        self._work_root = Path(work_root or settings.paths.work_dir)

    # ------------------------------------------------------------------ SDK access
    def sdk(self) -> Any:
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("claude_agent_sdk")
            except ImportError as exc:
                raise ConfigError("claude-agent-sdk is not installed in this environment") from exc
        return self._sdk

    def _option_fields(self, sdk: Any) -> set[str]:
        try:
            return {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
        except TypeError as exc:
            raise ConfigError("ClaudeAgentOptions is not a dataclass in this SDK version; cannot verify required options") from exc

    def preflight(self) -> dict[str, Any]:
        """Fail fast (at container start) if the SDK cannot honour the security-relevant options."""
        sdk = self.sdk()
        missing_attrs = [a for a in _REQUIRED_SDK_ATTRS if not hasattr(sdk, a)]
        if missing_attrs:
            raise ConfigError(f"installed claude-agent-sdk lacks required API: {missing_attrs}")
        fields = self._option_fields(sdk)
        missing = sorted(REQUIRED_OPTION_FIELDS - fields)
        if missing:
            raise ConfigError(f"installed claude-agent-sdk lacks required option(s) {missing}; upgrade the pinned version")
        if self._settings.agent.require_verbatim_prompts and "verbatim_prompts" not in fields:
            raise ConfigError("verbatim_prompts is required (prompt-injection defence) but the installed SDK predates it; "
                              "use claude-agent-sdk>=0.2.158 or set RULE_EXTRACTOR_REQUIRE_VERBATIM_PROMPTS=false (not recommended)")
        try:
            from importlib import metadata
            version = metadata.version("claude-agent-sdk")
        except Exception:
            version = "unknown"
        return {"sdk_version": version, "verbatim_prompts": "verbatim_prompts" in fields,
                "budget_cap": "max_budget_usd" in fields}

    # ------------------------------------------------------------------------ run
    async def run(self, task: AgentTask) -> AgentRunStats:
        sdk = self.sdk()
        self._work_root.mkdir(parents=True, exist_ok=True)
        session_dir = Path(tempfile.mkdtemp(prefix="sess-", dir=str(self._work_root)))
        try:
            return await self._run(sdk, task, session_dir)
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)

    def _wrap_tool(self, sdk: Any, spec: ToolSpec) -> Any:
        async def call(args: dict[str, Any]) -> dict[str, Any]:
            try:
                out = spec.handler(args)
                if inspect.isawaitable(out):
                    out = await out
                result = out if isinstance(out, ToolResult) else ToolResult(str(out))
            except Exception as exc:  # a tool bug must never crash the session or leak a stack trace to the model
                log.exception("tool handler failed: %s", spec.name)
                result = ToolResult(f"{spec.name} failed internally ({type(exc).__name__}).", is_error=True)
            payload: dict[str, Any] = {"content": [{"type": "text", "text": result.text[:4000]}]}
            if result.is_error:
                payload["is_error"] = True
            return payload

        return sdk.tool(spec.name, spec.description, spec.input_schema)(call)

    def _build_options(self, sdk: Any, task: AgentTask, server: Any, gate: Any, env: dict[str, str], cwd: str,
                       stderr_cb: Any) -> Any:
        fields = self._option_fields(sdk)
        missing = sorted(REQUIRED_OPTION_FIELDS - fields)
        if missing:
            raise ConfigError(f"installed claude-agent-sdk lacks required option(s) {missing}")
        kwargs: dict[str, Any] = dict(
            model=task.model,
            system_prompt=task.system_prompt,
            mcp_servers={SERVER_NAME: server},
            tools=list(task.profile.builtin_tools),
            allowed_tools=[mcp_tool_name(spec.name) for spec in task.tools],
            disallowed_tools=list(deny_list_for(task.profile)),
            permission_mode="default",
            can_use_tool=gate,
            max_turns=task.max_turns,
            cwd=cwd,
            env=env,
            setting_sources=[],
        )
        optional: dict[str, Any] = {"include_partial_messages": False, "stderr": stderr_cb}
        if task.max_budget_usd:
            optional["max_budget_usd"] = task.max_budget_usd
        if task.untrusted_text_in_prompt or self._settings.agent.require_verbatim_prompts:
            if "verbatim_prompts" in fields:
                optional["verbatim_prompts"] = True
            elif self._settings.agent.require_verbatim_prompts:
                raise ConfigError("verbatim_prompts is required but unsupported by the installed claude-agent-sdk")
        for name, value in optional.items():
            if name in fields:
                kwargs[name] = value
            else:
                log.warning("claude-agent-sdk does not support option %s; continuing without it", name)
        return sdk.ClaudeAgentOptions(**kwargs)

    async def _run(self, sdk: Any, task: AgentTask, session_dir: Path) -> AgentRunStats:
        stats = AgentRunStats()
        started = time.monotonic()
        env = build_sdk_env(self._settings, config_dir=session_dir)
        server = sdk.create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0",
                                           tools=[self._wrap_tool(sdk, spec) for spec in task.tools])
        gate = make_sdk_gate(sdk, task.profile, {mcp_tool_name(s.name) for s in task.tools}, stats.denied_tools)
        stderr_tail: deque[str] = deque(maxlen=20)
        options = self._build_options(sdk, task, server, gate, env,
                                      cwd=str(task.profile.file_root or session_dir),
                                      stderr_cb=lambda line: stderr_tail.append(redact(str(line))[:500]))
        result: Any = None
        last_text = ""

        async def consume() -> None:
            nonlocal result, last_text
            async for message in sdk.query(prompt=_prompt_stream(task.user_prompt), options=options):
                kind = type(message).__name__
                if kind == "AssistantMessage":
                    for block in getattr(message, "content", None) or []:
                        bkind = type(block).__name__
                        if bkind == "ToolUseBlock":
                            name = str(getattr(block, "name", "?"))
                            stats.tool_calls[name] = stats.tool_calls.get(name, 0) + 1
                        elif bkind == "TextBlock":
                            last_text = str(getattr(block, "text", ""))
                elif kind == "ResultMessage":
                    result = message

        try:
            await asyncio.wait_for(consume(), timeout=task.timeout_seconds)
        except asyncio.TimeoutError:
            stats.duration_s = time.monotonic() - started
            raise AgentError(f"agent attempt exceeded {task.timeout_seconds:.0f}s", transient=True, reason="timeout", stats=stats) from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stats.duration_s = time.monotonic() - started
            if result is None:
                err = self._classify_exception(exc, stderr_tail)
                err.stats = stats
                raise err from exc
            log.debug("SDK raised %s after delivering a result; using the result", type(exc).__name__)

        stats.duration_s = time.monotonic() - started
        stats.final_text = last_text[:500]
        if result is None:
            raise AgentError("agent ended without a result message", transient=True, reason="no_result", stats=stats)
        self._apply_result(result, stats)
        return stats

    @staticmethod
    def _classify_exception(exc: BaseException, stderr_tail: deque[str]) -> AgentError:
        name = type(exc).__name__
        if name in _NON_TRANSIENT_EXCEPTIONS:
            return AgentError(f"{name}: the Claude CLI bundled with claude-agent-sdk is unavailable", transient=False, reason="cli_not_found")
        text = f"{exc} {' '.join(stderr_tail)}"
        transient = name in _TRANSIENT_EXCEPTIONS or bool(_TRANSIENT.search(text))
        return AgentError(redact(f"{name}: {str(exc)[:300]}"), transient=transient, reason="sdk_error")

    @staticmethod
    def _apply_result(result: Any, stats: AgentRunStats) -> None:
        subtype = str(getattr(result, "subtype", "") or "")
        stats.turns = int(getattr(result, "num_turns", 0) or 0)
        cost = getattr(result, "total_cost_usd", None)
        stats.cost_usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
        usage = getattr(result, "usage", None)
        stats.usage = dict(usage) if isinstance(usage, dict) else {}
        stats.session_id = getattr(result, "session_id", None)
        if subtype == "error_max_turns":
            stats.stop_reason = "max_turns"
            raise AgentError("agent hit the turn limit", transient=True, reason="max_turns", stats=stats)
        if subtype == "error_max_budget_usd":
            stats.stop_reason = "budget"
            raise AgentError("agent hit its per-unit budget", transient=False, reason="budget", stats=stats)
        if bool(getattr(result, "is_error", False)) or subtype.startswith("error"):
            text = str(getattr(result, "result", "") or "")
            stats.stop_reason = "error"
            raise AgentError(redact(f"agent error ({subtype or 'error'}): {text[:300]}"),
                             transient=bool(_TRANSIENT.search(text)), reason="api_error", stats=stats)
        stats.stop_reason = "success"
