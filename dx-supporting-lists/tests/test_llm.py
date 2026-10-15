"""LLM layer: catalog resolution, thinking mapping, Bedrock request/response handling, error mapping, and the gate."""
from __future__ import annotations

import asyncio
import unittest

from dx_support.llm.bedrock import STRUCTURED_OUTPUT_NOTE, TOOL_NAME, BedrockBackend, translate_error
from dx_support.llm.catalog import CatalogError, load_catalog, resolve_model, thinking_fields
from dx_support.llm.gate import AdaptiveLimiter, LLMGate, RollingBudget
from dx_support.llm.types import LLMCallError, LLMRequest, LLMResult, StopRequested, extract_json_fallback
from dx_support.settings import Settings

CAT = load_catalog()
SCHEMA = {"type": "object", "properties": {"selected": {"type": "array", "items": {"type": "string"}}}, "required": ["selected"]}


def req(**kw) -> LLMRequest:
    base = dict(stage="route", system_prompt="SYS", prompt="PROMPT", schema=SCHEMA, model="claude-sonnet-5", thinking="low", max_tokens=2000)
    base.update(kw)
    return LLMRequest(**base)


class ClientError(Exception):
    """Looks like botocore's ClientError (duck-typed by translate_error)."""

    def __init__(self, code: str, message: str = "m", status: int = 400):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeConverse:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def tool_response(payload, usage=None, stop="tool_use", thinking_first=True):
    content = []
    if thinking_first:
        content.append({"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}})
    content.append({"toolUse": {"toolUseId": "t1", "name": TOOL_NAME, "input": payload}})
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop,
            "usage": usage or {"inputTokens": 1000, "outputTokens": 200}, "metrics": {"latencyMs": 1234}}


class CatalogTests(unittest.TestCase):
    def test_alias_gets_profile_prefix_and_geo_premium(self):
        r = resolve_model("claude-sonnet-5", "au", CAT)
        self.assertEqual(r.model_id, "au.anthropic.claude-sonnet-5")
        self.assertAlmostEqual(r.price_multiplier, 1.10)
        g = resolve_model("claude-sonnet-5", "global", CAT)
        self.assertEqual((g.model_id, g.price_multiplier), ("global.anthropic.claude-sonnet-5", 1.0))
        self.assertEqual(resolve_model("claude-sonnet-5", "none", CAT).model_id, "anthropic.claude-sonnet-5")

    def test_unlisted_profile_prefix_warns_but_resolves(self):
        r = resolve_model("claude-sonnet-5-5", "au", CAT)
        self.assertEqual(r.model_id, "au.anthropic.claude-sonnet-5-5")
        self.assertIn("does not list", r.warning)

    def test_explicit_ids_are_used_verbatim_and_typos_are_refused(self):
        r = resolve_model("global.anthropic.claude-sonnet-5-5", "au", CAT)
        self.assertEqual((r.model_id, r.spec.alias, r.spec.known), ("global.anthropic.claude-sonnet-5-5", "claude-sonnet-5-5", True))
        r2 = resolve_model("us.anthropic.claude-future-9", "au", CAT)
        self.assertFalse(r2.spec.known)
        self.assertIn("not in the model catalog", r2.warning)
        with self.assertRaises(CatalogError):
            resolve_model("claude-sonet-5", "au", CAT)                  # a typo must never become a different model
        with self.assertRaises(CatalogError):
            resolve_model("", "au", CAT)

    def test_thinking_mapping_per_model(self):
        s5, s55, h45 = CAT.models["claude-sonnet-5"], CAT.models["claude-sonnet-5-5"], CAT.models["claude-haiku-4-5-20251001"]
        fable = CAT.models["claude-fable-5-1"]
        self.assertEqual(thinking_fields(s5, "low")[0], {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})
        self.assertEqual(thinking_fields(s5, "adaptive")[0], {"thinking": {"type": "adaptive"}})
        self.assertEqual(thinking_fields(s5, "disabled")[0], {"thinking": {"type": "disabled"}})
        self.assertEqual(thinking_fields(s55, "disabled")[0], {"thinking": {"type": "between_tools"}})   # 5.5 rejects `disabled`
        extra, warn = thinking_fields(fable, "disabled")
        self.assertEqual(extra["output_config"], {"effort": "low"})
        self.assertIn("cannot turn thinking off", warn)
        self.assertEqual(thinking_fields(h45, "low")[0], {"thinking": {"type": "enabled", "budget_tokens": 1024}})
        self.assertEqual(thinking_fields(h45, "disabled")[0], {})
        with self.assertRaises(CatalogError):
            thinking_fields(s5, "turbo")


class BedrockBackendTests(unittest.TestCase):
    def backend(self, responses, **settings_kw):
        client = FakeConverse(responses)
        s = Settings.from_env({"DX_ENV": "test"}).with_overrides(**settings_kw)
        return BedrockBackend(s, client=client), client

    def test_request_shape(self):
        be, _ = self.backend([])
        kw = be.build_request(req())
        self.assertEqual(kw["modelId"], "au.anthropic.claude-sonnet-5")
        self.assertEqual(kw["inferenceConfig"], {"maxTokens": 2000})                      # no temperature / top_p: Claude 5.x rejects them
        self.assertEqual(kw["toolConfig"]["toolChoice"], {"auto": {}})                    # forced tool choice is rejected on Claude 5.x
        tool = kw["toolConfig"]["tools"][0]["toolSpec"]
        self.assertEqual((tool["name"], tool["inputSchema"]["json"]), (TOOL_NAME, SCHEMA))
        self.assertIn(STRUCTURED_OUTPUT_NOTE, kw["system"][0]["text"])
        self.assertEqual(kw["additionalModelRequestFields"], {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})
        self.assertNotIn("guardrailConfig", kw)
        self.assertEqual(kw["messages"], [{"role": "user", "content": [{"text": "PROMPT"}]}])

    def test_budget_thinking_raises_the_output_floor_and_guardrail_is_passed_through(self):
        be, _ = self.backend([], guardrail_id="gr-1", guardrail_version="3")
        kw = be.build_request(req(model="claude-haiku-4-5-20251001", max_tokens=1500))
        self.assertGreaterEqual(kw["inferenceConfig"]["maxTokens"], 1024 + 1024)           # budget must stay below max_tokens
        self.assertEqual(kw["guardrailConfig"]["guardrailIdentifier"], "gr-1")

    def test_prompt_caching_adds_cache_points_only_for_large_constant_prefixes(self):
        be, _ = self.backend([], prompt_caching=True)
        big = "x" * 3000
        kw = be.build_request(req(prompt=big + "TAIL", cache_prefix_chars=3000))
        self.assertEqual(kw["system"][1], {"cachePoint": {"type": "default"}})
        self.assertEqual([list(b)[0] for b in kw["messages"][0]["content"]], ["text", "cachePoint", "text"])
        self.assertEqual(kw["messages"][0]["content"][2]["text"], "TAIL")
        small = be.build_request(req(prompt="short", cache_prefix_chars=5))
        self.assertEqual(small["messages"][0]["content"], [{"text": "short"}])
        off, _ = self.backend([], prompt_caching=False)
        self.assertEqual(len(off.build_request(req(prompt=big, cache_prefix_chars=3000))["system"]), 1)

    def test_tool_use_is_found_by_type_not_position_and_cost_uses_usage(self):
        be, client = self.backend([tool_response({"selected": ["a"]}, {"inputTokens": 1_000_000, "outputTokens": 100_000})])
        res = asyncio.run(be.complete(req(max_tokens=2000, max_budget_usd=100)))
        self.assertEqual(res.structured, {"selected": ["a"]})
        self.assertAlmostEqual(res.cost_usd, (2.0 + 1.0) * 1.10, places=6)                 # $2/M in + $10/M out, +10% geo premium
        self.assertEqual((res.diag["stop_reason"], res.diag["duration_ms"], res.diag["model_id"]), ("tool_use", 1234, "au.anthropic.claude-sonnet-5"))
        self.assertEqual(len(client.calls), 1)

    def test_cache_tokens_are_priced_separately(self):
        be, _ = self.backend([tool_response({"selected": []}, {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 1_000_000,
                                                             "cacheWriteInputTokens": 1_000_000})])
        res = asyncio.run(be.complete(req(max_budget_usd=100)))
        self.assertAlmostEqual(res.cost_usd, (0.1 * 2.0 + 1.25 * 2.0) * 1.10, places=6)

    def test_plain_text_json_is_a_fallback_and_no_json_means_no_structured_output(self):
        txt = {"output": {"message": {"content": [{"text": 'Here: {"selected": ["z"]} done'}]}}, "stopReason": "end_turn", "usage": {}}
        be, _ = self.backend([txt, {"output": {"message": {"content": [{"text": "sorry"}]}}, "stopReason": "end_turn", "usage": {}}])
        self.assertEqual(asyncio.run(be.complete(req())).structured, {"selected": ["z"]})
        self.assertIsNone(asyncio.run(be.complete(req())).structured)
        self.assertEqual(extract_json_fallback("no braces"), None)

    def test_truncation_and_guardrail_stops(self):
        cut = {"output": {"message": {"content": [{"text": "thinking out loud"}]}}, "stopReason": "max_tokens", "usage": {"outputTokens": 2000}}
        blocked = {"output": {"message": {"content": []}}, "stopReason": "guardrail_intervened", "usage": {}}
        be, _ = self.backend([cut, blocked])
        with self.assertRaises(LLMCallError) as c1:
            asyncio.run(be.complete(req()))
        self.assertEqual(c1.exception.kind, "transient")
        self.assertGreater(c1.exception.cost_usd, 0)
        with self.assertRaises(LLMCallError) as c2:
            asyncio.run(be.complete(req()))
        self.assertEqual(c2.exception.kind, "budget")                                       # blocked again if retried

    def test_a_call_that_could_exceed_the_per_call_budget_is_refused_before_it_is_sent(self):
        be, client = self.backend([])
        with self.assertRaises(LLMCallError) as c:
            asyncio.run(be.complete(req(model="claude-fable-5-1", max_tokens=16000, max_budget_usd=0.10)))
        self.assertEqual(c.exception.kind, "budget")
        self.assertEqual(client.calls, [])

    def test_error_translation(self):
        k = lambda e: translate_error(e).kind          # noqa: E731
        self.assertEqual(k(ClientError("ThrottlingException", status=429)), "throttle")
        self.assertEqual(k(ClientError("ServiceQuotaExceededException", status=400)), "throttle")
        self.assertEqual(k(ClientError("ModelTimeoutException", status=408)), "transient")
        self.assertEqual(k(ClientError("InternalServerException", status=500)), "transient")
        self.assertEqual(k(ClientError("AccessDeniedException", status=403)), "fatal")
        self.assertEqual(k(ClientError("ResourceNotFoundException", status=404)), "fatal")
        self.assertEqual(k(ClientError("ValidationException", "Invocation of model ID x with on-demand throughput isn't supported")), "fatal")
        self.assertEqual(k(ClientError("ValidationException", "Input is too long for requested model")), "budget")
        self.assertEqual(k(type("ReadTimeoutError", (Exception,), {})("t")), "transient")
        self.assertEqual(k(type("NoCredentialsError", (Exception,), {})("no creds")), "fatal")
        e = translate_error(ClientError("AccessDeniedException", "key AKIA" "ABCDEFGHIJKLMNOP is not allowed", 403))
        self.assertNotIn("AKIA" "ABCDEFGHIJKLMNOP", str(e))                                  # nothing credential-shaped leaves the process

    def test_backend_translates_client_exceptions(self):
        be, _ = self.backend([ClientError("ThrottlingException", status=429)])
        with self.assertRaises(LLMCallError) as c:
            asyncio.run(be.complete(req()))
        self.assertEqual(c.exception.kind, "throttle")


class Scripted:
    name = "scripted"

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def complete(self, r):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


def run(coro):
    return asyncio.run(coro)


def jreq():
    return LLMRequest(stage="judge", system_prompt="s", prompt="p", schema={}, model="m")


class GateTests(unittest.TestCase):
    def test_retry_counts_failed_cost_and_stops(self):
        b = Scripted([LLMCallError("overloaded", cost_usd=0.01), LLMResult({"ok": 1}, 0.02)])
        g = LLMGate(b, 2, 10, backoff_base_s=0)
        self.assertEqual(run(g.call(jreq())).structured, {"ok": 1})
        self.assertAlmostEqual(g.total_cost, 0.03)
        self.assertEqual((g.n_failed_attempts, b.calls), (1, 2))
        # budget errors are not retried
        b = Scripted([LLMCallError("too big", cost_usd=0.5, kind="budget")])
        g = LLMGate(b, 2, 10, backoff_base_s=0)
        with self.assertRaises(LLMCallError):
            run(g.call(jreq()))
        self.assertEqual((b.calls, g.total_cost), (1, 0.5))
        # fatal error stops the whole run
        b = Scripted([LLMCallError("denied", kind="fatal")])
        g = LLMGate(b, 2, 10, backoff_base_s=0)
        with self.assertRaises(LLMCallError):
            run(g.call(jreq()))
        self.assertTrue(g.stop and g.fatal_error)
        with self.assertRaises(StopRequested):
            run(g.call(jreq()))
        # an unexpected non-model exception is fatal (fail closed)
        g = LLMGate(Scripted([ValueError("bug")]), 2, 10, backoff_base_s=0)
        with self.assertRaises(LLMCallError) as c:
            run(g.call(jreq()))
        self.assertEqual(c.exception.kind, "fatal")

    def test_cost_cap_lets_the_crossing_call_finish_and_refuses_the_next(self):
        b = Scripted([LLMResult({"a": 1}, 0.03)] * 3)
        g = LLMGate(b, 2, 0.05, backoff_base_s=0)
        run(g.call(jreq()))
        run(g.call(jreq()))
        self.assertTrue(g.stop and g.cost_cap_hit)
        with self.assertRaises(StopRequested):
            run(g.call(jreq()))
        self.assertEqual(b.calls, 2)

    def test_empty_structured_output_is_retried_then_fails(self):
        b = Scripted([LLMResult(None, 0.01)] * 3)
        g = LLMGate(b, 2, 10, backoff_base_s=0, max_retries=2)
        with self.assertRaises(LLMCallError):
            run(g.call(jreq()))
        self.assertEqual((b.calls, round(g.total_cost, 6)), (3, 0.03))

    def test_throttling_retries_longer_and_shrinks_concurrency(self):
        lim = AdaptiveLimiter(8, cooldown_s=0)
        b = Scripted([LLMCallError("slow down", kind="throttle")] * 3 + [LLMResult({"ok": 1}, 0.01)])
        g = LLMGate(b, lim, 10, backoff_base_s=0, max_retries=0, max_throttle_retries=5)
        self.assertEqual(run(g.call(jreq())).structured, {"ok": 1})
        self.assertEqual(b.calls, 4)
        self.assertLess(lim.limit, 8)                      # halved on throttling
        b = Scripted([LLMCallError("slow down", kind="throttle")] * 3)
        g = LLMGate(b, AdaptiveLimiter(4), 10, backoff_base_s=0, max_throttle_retries=1)
        with self.assertRaises(LLMCallError):
            run(g.call(jreq()))
        self.assertEqual(b.calls, 2)

    def test_limiter_recovers_after_a_run_of_successes_and_bounds_in_flight(self):
        lim = AdaptiveLimiter(4, recover_after=3, cooldown_s=0)
        lim.on_throttle()
        self.assertEqual(lim.limit, 2)
        for _ in range(3):
            lim.on_success()
        self.assertEqual(lim.limit, 3)

        peak = 0

        class Slow:
            name = "slow"

            async def complete(self, r):
                nonlocal peak
                peak = max(peak, lim2.in_flight)
                await asyncio.sleep(0.01)
                return LLMResult({"ok": 1}, 0.0)

        lim2 = AdaptiveLimiter(3)
        g = LLMGate(Slow(), lim2, 10, backoff_base_s=0)

        async def many():
            await asyncio.gather(*[g.call(jreq()) for _ in range(20)])
        run(many())
        self.assertLessEqual(peak, 3)
        self.assertEqual(lim2.in_flight, 0)

    def test_rolling_budget_stops_new_calls(self):
        now = [0.0]
        rb = RollingBudget(0.05, window_s=100, clock=lambda: now[0])
        g = LLMGate(Scripted([LLMResult({"a": 1}, 0.06)] * 2), 2, 10, backoff_base_s=0, budget=rb)
        run(g.call(jreq()))
        self.assertTrue(rb.exhausted)
        with self.assertRaises(StopRequested):
            run(g.call(jreq()))
        now[0] = 500.0                                     # the window slides: spend is forgotten
        self.assertFalse(rb.exhausted)

    def test_run_log_records_every_attempt_without_prompts(self):
        recs = []
        b = Scripted([LLMCallError("x", cost_usd=0.01), LLMResult({"ok": 1}, 0.02, {"input_tokens": 5, "output_tokens": 7})])
        g = LLMGate(b, 1, 10, run_log=recs.append, backoff_base_s=0)
        run(g.call(jreq()))
        self.assertEqual([r["ok"] for r in recs], [False, True])
        self.assertTrue(all("prompt" not in r and "system_prompt" not in r for r in recs))


if __name__ == "__main__":
    unittest.main()
