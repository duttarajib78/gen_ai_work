"""A fake ``claude_agent_sdk`` module: just enough surface for the runner, with a scriptable model."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

FULL_FIELDS = ("model", "system_prompt", "mcp_servers", "tools", "allowed_tools", "disallowed_tools", "permission_mode",
               "can_use_tool", "max_turns", "cwd", "env", "setting_sources", "include_partial_messages", "stderr",
               "max_budget_usd", "verbatim_prompts")


@dataclasses.dataclass
class PermissionResultAllow:
    updated_input: Any = None


@dataclasses.dataclass
class PermissionResultDeny:
    message: str = ""
    interrupt: bool = False


class TextBlock:
    def __init__(self, text): self.text = text


class ToolUseBlock:
    def __init__(self, name, input=None): self.name, self.input = name, input or {}


class AssistantMessage:
    def __init__(self, content): self.content = content


class ResultMessage:
    def __init__(self, subtype="success", is_error=False, num_turns=3, total_cost_usd=0.02, usage=None, result="DONE", session_id="s1"):
        self.subtype, self.is_error, self.num_turns = subtype, is_error, num_turns
        self.total_cost_usd, self.usage, self.result, self.session_id = total_cost_usd, usage or {"input_tokens": 10, "output_tokens": 5}, result, session_id


class CLINotFoundError(Exception):
    pass


class ProcessError(Exception):
    pass


@dataclasses.dataclass
class SdkTool:
    name: str
    description: str
    input_schema: dict
    handler: Callable


class FakeSdk:
    PermissionResultAllow = PermissionResultAllow
    PermissionResultDeny = PermissionResultDeny

    def __init__(self, script, *, fields=FULL_FIELDS):
        self.ClaudeAgentOptions = dataclasses.make_dataclass(
            "ClaudeAgentOptions", [(f, Any, dataclasses.field(default=None)) for f in fields])
        self._script = script
        self.last_options = None
        self.last_prompt = None
        self.queries = 0

    def tool(self, name, description, input_schema):
        def deco(fn):
            return SdkTool(name, description, input_schema, fn)
        return deco

    def create_sdk_mcp_server(self, name, version, tools):
        return {"name": name, "version": version, "tools": {t.name: t for t in tools}}

    async def query(self, *, prompt, options=None):
        self.queries += 1
        self.last_options = options
        first = await prompt.__anext__()
        self.last_prompt = first["message"]["content"]
        async for message in self._script(self, options):
            yield message
