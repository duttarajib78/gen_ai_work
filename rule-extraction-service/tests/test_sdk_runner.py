import asyncio
import tempfile
import unittest
from pathlib import Path

from rule_extractor.agents.collector import RuleCollector
from rule_extractor.agents.permissions import deny_list_for
from rule_extractor.agents.sdk_runner import ClaudeSdkRunner
from rule_extractor.agents.types import AgentTask, ToolProfile, ToolResult, ToolSpec
from rule_extractor.errors import AgentError, ConfigError
from rule_extractor.settings import Settings

from .fake_sdk import (FULL_FIELDS, AssistantMessage, CLINotFoundError, FakeSdk, ProcessError, ResultMessage, TextBlock,
                       ToolUseBlock)


def run(coro):
    return asyncio.run(coro)


def make_settings(tmp, **env):
    base = {"AWS_REGION": "ap-southeast-2", "RULE_EXTRACTOR_WORK_DIR": str(Path(tmp) / "work")}
    base.update(env)
    return Settings.load(env=base)


def make_task(spec_handler=None, profile=None, **kw):
    spec = ToolSpec("emit_rule", "emit", {"type": "object"}, spec_handler or (lambda a: ToolResult("ok")))
    base = dict(unit_id="u1", kind="sweep", system_prompt="SYS", user_prompt="USER TEXT", model="au.anthropic.claude-sonnet-5",
                profile=profile or ToolProfile("none"), tools=[spec], max_turns=5, timeout_seconds=5, max_budget_usd=1.5,
                untrusted_text_in_prompt=True)
    base.update(kw)
    return AgentTask(**base)


GOOD_RULE = dict(rule_name="n", rule_logic_plain_english="l", upcoding_mechanism="m", recommended_action="remove_code",
                 source_reference="acs printed p.1", rule_shape="redundancy")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = make_settings(self.tmp.name)

    def test_success_path_enforces_tool_policy_and_collects_rules(self):
        collector = RuleCollector(job_id="job-aaaaaaaaaaaa", unit_id="u1", topic="t", source_model="m", prompt_version="v")
        denied = {}

        async def script(sdk, options):
            tool = options.mcp_servers["rules"]["tools"]["emit_rule"]
            out = await tool.handler(GOOD_RULE)
            yield AssistantMessage([ToolUseBlock("mcp__rules__emit_rule", GOOD_RULE), TextBlock("recorded")])
            denied["bash"] = await options.can_use_tool("Bash", {"command": "id"}, None)
            denied["read"] = await options.can_use_tool("Read", {"file_path": "/etc/passwd"}, None)
            denied["emit"] = out
            yield AssistantMessage([TextBlock("DONE")])
            yield ResultMessage(num_turns=4, total_cost_usd=0.07)

        sdk = FakeSdk(script)
        runner = ClaudeSdkRunner(self.settings, sdk=sdk)
        stats = run(runner.run(make_task(collector.handle, untrusted_text_in_prompt=True)))
        self.assertEqual((stats.turns, stats.cost_usd, stats.stop_reason), (4, 0.07, "success"))
        self.assertEqual(len(collector.emitted), 1)
        self.assertEqual(stats.tool_calls, {"mcp__rules__emit_rule": 1})
        self.assertEqual(stats.final_text, "DONE")
        self.assertEqual(stats.denied_tools, ["Bash", "Read"])
        self.assertEqual(type(denied["bash"]).__name__, "PermissionResultDeny")
        self.assertNotIn("is_error", denied["emit"])
        o = sdk.last_options
        self.assertEqual(o.permission_mode, "default")
        self.assertEqual(o.setting_sources, [])
        self.assertEqual(o.tools, [])
        self.assertEqual(o.allowed_tools, ["mcp__rules__emit_rule"])        # no built-ins pre-approved -> gate never shadowed
        self.assertTrue({"Bash", "Write", "WebFetch", "WebSearch"} <= set(o.disallowed_tools))
        self.assertTrue(o.verbatim_prompts)
        self.assertEqual(o.max_budget_usd, 1.5)
        self.assertEqual(o.env["CLAUDE_CODE_USE_BEDROCK"], "1")
        self.assertEqual(o.env["AWS_REGION"], "ap-southeast-2")
        self.assertEqual(o.model, "au.anthropic.claude-sonnet-5")
        self.assertEqual(sdk.last_prompt, "USER TEXT")
        self.assertEqual(list(Path(self.tmp.name, "work").glob("sess-*")), [])   # session dir removed

    def test_corpus_profile_sets_cwd_and_builtins(self):
        root = Path(self.tmp.name) / "corpus"
        root.mkdir()
        profile = ToolProfile("corpus", builtin_tools=("Read", "Grep", "Glob"), file_root=root)

        async def script(sdk, options):
            yield ResultMessage()

        sdk = FakeSdk(script)
        run(ClaudeSdkRunner(self.settings, sdk=sdk).run(make_task(profile=profile)))
        self.assertEqual(sdk.last_options.cwd, str(root))
        self.assertEqual(sdk.last_options.tools, ["Read", "Grep", "Glob"])
        self.assertNotIn("Read", sdk.last_options.disallowed_tools)
        self.assertNotIn("Read", sdk.last_options.allowed_tools)

    def test_missing_security_options_fail_closed(self):
        sdk = FakeSdk(lambda *_: None, fields=tuple(f for f in FULL_FIELDS if f != "can_use_tool"))
        with self.assertRaises(ConfigError):
            run(ClaudeSdkRunner(self.settings, sdk=sdk).run(make_task()))
        with self.assertRaises(ConfigError):
            ClaudeSdkRunner(self.settings, sdk=sdk).preflight()

    def test_verbatim_prompts_required_by_default_but_can_be_waived(self):
        fields = tuple(f for f in FULL_FIELDS if f != "verbatim_prompts")

        async def script(sdk, options):
            yield ResultMessage()

        with self.assertRaises(ConfigError):
            run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script, fields=fields)).run(make_task()))
        waived = make_settings(self.tmp.name, RULE_EXTRACTOR_REQUIRE_VERBATIM_PROMPTS="false")
        stats = run(ClaudeSdkRunner(waived, sdk=FakeSdk(script, fields=fields)).run(make_task(untrusted_text_in_prompt=False)))
        self.assertEqual(stats.stop_reason, "success")

    def test_preflight_reports_capabilities(self):
        info = ClaudeSdkRunner(self.settings, sdk=FakeSdk(lambda *_: None)).preflight()
        self.assertTrue(info["verbatim_prompts"] and info["budget_cap"])

    def test_max_turns_is_transient_and_carries_stats(self):
        async def script(sdk, options):
            yield ResultMessage(subtype="error_max_turns", is_error=True, num_turns=40, total_cost_usd=0.5)

        with self.assertRaises(AgentError) as cm:
            run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task()))
        self.assertTrue(cm.exception.transient)
        self.assertEqual(cm.exception.reason, "max_turns")
        self.assertEqual(cm.exception.stats.cost_usd, 0.5)

    def test_budget_error_is_not_transient(self):
        async def script(sdk, options):
            yield ResultMessage(subtype="error_max_budget_usd", is_error=True)

        with self.assertRaises(AgentError) as cm:
            run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task()))
        self.assertFalse(cm.exception.transient)

    def test_api_error_text_classification(self):
        for text, transient in (("ThrottlingException: Too many requests", True), ("API Error 429", True),
                                ("ValidationException: model id invalid", False)):
            async def script(sdk, options, text=text):
                yield ResultMessage(subtype="error_during_execution", is_error=True, result=text)

            with self.assertRaises(AgentError) as cm:
                run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task()))
            self.assertEqual(cm.exception.transient, transient, text)

    def test_exception_after_result_is_tolerated(self):
        async def script(sdk, options):
            yield ResultMessage()
            raise ProcessError("exit code 1 after result")

        stats = run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task()))
        self.assertEqual(stats.stop_reason, "success")

    def test_exception_without_result_is_classified(self):
        async def cli_missing(sdk, options):
            raise CLINotFoundError("no cli")
            yield  # pragma: no cover

        async def crash(sdk, options):
            raise ProcessError("Command failed with exit code 1")
            yield  # pragma: no cover

        async def bug(sdk, options):
            raise ValueError("programming error")
            yield  # pragma: no cover

        for script, transient in ((cli_missing, False), (crash, True), (bug, False)):
            with self.assertRaises(AgentError) as cm:
                run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task()))
            self.assertEqual(cm.exception.transient, transient, script.__name__)
            self.assertIsNotNone(cm.exception.stats)

    def test_missing_result_message_is_transient(self):
        async def script(sdk, options):
            yield AssistantMessage([TextBlock("hi")])

        with self.assertRaises(AgentError) as cm:
            run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task()))
        self.assertEqual(cm.exception.reason, "no_result")

    def test_timeout(self):
        async def script(sdk, options):
            await asyncio.sleep(5)
            yield ResultMessage()

        with self.assertRaises(AgentError) as cm:
            run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task(timeout_seconds=0.1)))
        self.assertEqual(cm.exception.reason, "timeout")
        self.assertTrue(cm.exception.transient)
        self.assertEqual(list(Path(self.tmp.name, "work").glob("sess-*")), [])

    def test_cancellation_propagates_and_cleans_up(self):
        async def script(sdk, options):
            await asyncio.sleep(30)
            yield ResultMessage()

        async def main():
            t = asyncio.create_task(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task(timeout_seconds=60)))
            await asyncio.sleep(0.1)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t

        run(main())
        self.assertEqual(list(Path(self.tmp.name, "work").glob("sess-*")), [])

    def test_tool_handler_exceptions_become_error_results(self):
        out = {}

        def boom(args):
            raise RuntimeError("secret internal detail")

        async def script(sdk, options):
            out["r"] = await options.mcp_servers["rules"]["tools"]["emit_rule"].handler({})
            yield ResultMessage()

        run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task(boom)))
        self.assertTrue(out["r"]["is_error"])
        self.assertNotIn("secret internal detail", out["r"]["content"][0]["text"])

    def test_async_tool_handlers_are_awaited(self):
        out = {}

        async def handler(args):
            return ToolResult("async ok")

        async def script(sdk, options):
            out["r"] = await options.mcp_servers["rules"]["tools"]["emit_rule"].handler({})
            yield ResultMessage()

        run(ClaudeSdkRunner(self.settings, sdk=FakeSdk(script)).run(make_task(handler)))
        self.assertEqual(out["r"]["content"][0]["text"], "async ok")

    def test_missing_sdk_is_a_config_error(self):
        import sys
        saved = sys.modules.pop("claude_agent_sdk", None)
        try:
            with self.assertRaises(ConfigError):
                ClaudeSdkRunner(self.settings).preflight()
        finally:
            if saved:
                sys.modules["claude_agent_sdk"] = saved

    def test_deny_list_excludes_profile_builtins(self):
        self.assertNotIn("WebFetch", deny_list_for(ToolProfile("w", builtin_tools=("WebFetch",))))


if __name__ == "__main__":
    unittest.main()
