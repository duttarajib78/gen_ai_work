"""Least-privilege tool policy.

Layers (any one of which blocks a call):
  1. ``tools=``            only the profile's built-in tools exist in the session
  2. ``disallowed_tools``  every other dangerous built-in is removed from the model's context
  3. ``can_use_tool``      this module's ``decide`` is called for anything not pre-approved and denies by default:
                           corpus reads are confined to one directory, web fetches to an allow-list of https hosts
``bypassPermissions`` is never used. ``decide`` is a pure function so the policy is unit-testable without the SDK.
"""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any, Collection, Mapping
from urllib.parse import urlsplit

from ..util import is_within
from .types import ToolProfile

SERVER_NAME = "rules"
READ_ONLY_TOOLS = ("Read", "Grep", "Glob")
# Built-ins we never want available; WebFetch is removed from this list only for profiles that include it.
DENY_TOOLS = (
    "Bash", "BashOutput", "KillBash", "KillShell", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebSearch",
    "WebFetch", "Agent", "Task", "Skill", "SlashCommand", "TodoWrite", "AskUserQuestion", "ExitPlanMode",
    "EnterPlanMode", "ListMcpResources", "ReadMcpResource",
)
_PATH_KEYS = ("file_path", "path", "notebook_path")


def mcp_tool_name(name: str) -> str:
    return f"mcp__{SERVER_NAME}__{name}"


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str = ""


def deny_list_for(profile: ToolProfile) -> tuple[str, ...]:
    return tuple(t for t in DENY_TOOLS if t not in profile.builtin_tools)


def decide(profile: ToolProfile, tool_name: str, tool_input: Mapping[str, Any], *, custom_tools: Collection[str]) -> Decision:
    if tool_name in custom_tools:
        return Decision(True)
    if tool_name not in profile.builtin_tools:
        return Decision(False, f"tool {tool_name!r} is not available for this task")
    if tool_name in READ_ONLY_TOOLS:
        return _check_paths(profile, tool_name, tool_input)
    if tool_name == "WebFetch":
        return _check_url(profile, tool_input)
    return Decision(False, f"tool {tool_name!r} is not permitted")


def _check_paths(profile: ToolProfile, tool_name: str, tool_input: Mapping[str, Any]) -> Decision:
    root = profile.file_root
    if root is None:
        return Decision(False, "file access is not enabled for this task")
    candidates: list[str] = []
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or "\x00" in value:
            return Decision(False, f"invalid {key}")
        candidates.append(value)
    if tool_name == "Read" and "file_path" not in tool_input:
        return Decision(False, "Read requires file_path")
    pattern = tool_input.get("pattern")
    if tool_name == "Glob" and isinstance(pattern, str):
        if ".." in pattern.replace("\\", "/").split("/") or "\x00" in pattern:
            return Decision(False, "glob patterns may not contain '..'")
        if os.path.isabs(pattern):
            candidates.append(pattern.split("*", 1)[0].split("?", 1)[0] or "/")
    for raw in candidates:
        resolved = raw if os.path.isabs(raw) else os.path.join(str(root), raw)
        if not is_within(root, resolved):
            return Decision(False, "path is outside the corpus directory")
    return Decision(True)


def _check_url(profile: ToolProfile, tool_input: Mapping[str, Any]) -> Decision:
    url = tool_input.get("url")
    if not isinstance(url, str) or len(url) > 2048:
        return Decision(False, "invalid url")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return Decision(False, "invalid url")
    if parts.scheme != "https" or parts.username or parts.password:
        return Decision(False, "only plain https URLs are allowed")
    if port not in (None, 443):
        return Decision(False, "non-standard ports are not allowed")
    try:
        ipaddress.ip_address(host)
        return Decision(False, "IP-literal hosts are not allowed")
    except ValueError:
        pass
    allowed = tuple(h.lower() for h in profile.web_allowed_hosts)
    if host in allowed or any(a.startswith(".") and host.endswith(a) for a in allowed):
        return Decision(True)
    return Decision(False, f"host {host!r} is not on the allow-list")


def make_sdk_gate(sdk: Any, profile: ToolProfile, custom_tools: Collection[str], audit: list[str]):
    """Adapt :func:`decide` to the SDK's ``can_use_tool`` callback signature."""

    async def gate(tool_name: str, input_data: dict[str, Any], context: Any = None):
        decision = decide(profile, tool_name, input_data or {}, custom_tools=custom_tools)
        if decision.allow:
            return sdk.PermissionResultAllow()
        audit.append(tool_name)
        return sdk.PermissionResultDeny(message=decision.reason)

    return gate
