import json
import unittest

from audit_agent.deadline import Deadline
from audit_agent.errors import (
    DeadlineExceededError, LlmAccessError, LlmOutputError, LlmRefusalError, LlmRequestError, LlmThrottledError,
)
from audit_agent.llm.base import UsageMeter
from audit_agent.llm.bedrock import BedrockClaudeClient
from audit_agent.prompts import VERDICTS_TOOL
from tests.fakes import FakeBedrockRuntime, FakeClientError, FakeClock, converse_response

MODEL = "global.anthropic.claude-sonnet-5-5"
ARGS = {"verdicts": [{"code": "J189"}]}


def make_client(runtime, clock=None, **kw):
    clock = clock or FakeClock()
    kw.setdefault("max_transport_retries", 2)
    return BedrockClaudeClient(runtime, model_id=MODEL, sleep=clock.sleep, clock=clock, rand=lambda: 1.0, **kw), clock


def call(client, deadline=None, meter=None):
    return client.invoke_tool(system_prompt="S", user_text="U", tool=VERDICTS_TOOL,
                              deadline=deadline or Deadline(50), meter=meter)


class RequestShapeTests(unittest.TestCase):
    def test_request_respects_sonnet_5_5_constraints(self):
        client, _ = make_client(FakeBedrockRuntime(), effort="high")
        req = client.build_request(system_prompt="S", user_text="U", tool=VERDICTS_TOOL)
        self.assertEqual(req["modelId"], MODEL)
        self.assertEqual(req["inferenceConfig"], {"maxTokens": 16000})  # no temperature/top_p/top_k: they return 400
        self.assertEqual(req["toolConfig"]["toolChoice"], {"auto": {}})  # forced tool choice returns 400
        self.assertEqual(req["additionalModelRequestFields"],
                         {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}})
        self.assertEqual(req["system"], [{"text": "S"}])
        self.assertEqual(req["messages"], [{"role": "user", "content": [{"text": "U"}]}])
        spec = req["toolConfig"]["tools"][0]["toolSpec"]
        self.assertEqual(spec["name"], "submit_diagnosis_verdicts")
        self.assertEqual(spec["inputSchema"]["json"]["required"], ["verdicts"])
        self.assertNotIn("guardrailConfig", req)
        json.dumps(req)  # must be serialisable as-is

    def test_optional_features(self):
        client, _ = make_client(
            FakeBedrockRuntime(), effort="medium", thinking_mode="between_tools", prompt_caching=True,
            guardrail_id="gr-1", guardrail_version="2", max_output_tokens=9000,
            extra_request_fields={"output_config": {"effort": "low"}},
        )
        req = client.build_request(system_prompt="S", user_text="U", tool=VERDICTS_TOOL, max_output_tokens=20000)
        self.assertEqual(req["system"][1], {"cachePoint": {"type": "default"}})
        self.assertEqual(req["guardrailConfig"], {"guardrailIdentifier": "gr-1", "guardrailVersion": "2"})
        self.assertEqual(req["additionalModelRequestFields"]["thinking"], {"type": "between_tools"})
        self.assertEqual(req["additionalModelRequestFields"]["output_config"], {"effort": "low"})  # operator override wins
        self.assertEqual(req["inferenceConfig"]["maxTokens"], 20000)


class ResponseParsingTests(unittest.TestCase):
    def test_reasoning_block_before_tool_call_is_ignored_and_usage_recorded(self):
        meter = UsageMeter()
        client, _ = make_client(FakeBedrockRuntime(converse_response(ARGS)))
        result = call(client, meter=meter)
        self.assertEqual(dict(result.arguments), ARGS)
        self.assertEqual((meter.calls, meter.input_tokens, meter.output_tokens), (1, 1000, 500))
        self.assertEqual(result.usage.latency_ms, 1234)

    def test_cache_token_usage_is_mapped(self):
        resp = converse_response(ARGS, usage={"inputTokens": 5, "outputTokens": 6, "cacheReadInputTokens": 700,
                                              "cacheWriteInputTokens": 80})
        meter = UsageMeter()
        call(make_client(FakeBedrockRuntime(resp))[0], meter=meter)
        self.assertEqual((meter.cache_read_tokens, meter.cache_write_tokens), (700, 80))

    def test_tool_input_given_as_json_string_is_parsed(self):
        result = call(make_client(FakeBedrockRuntime(converse_response(json.dumps(ARGS))))[0])
        self.assertEqual(dict(result.arguments), ARGS)

    def test_unusable_answers_raise_output_errors_but_still_count_usage(self):
        cases = {
            "truncated": (converse_response(ARGS, stop="max_tokens"), "llm_output_truncated"),
            "text only": ({"output": {"message": {"content": [{"text": "I think..."}]}}, "stopReason": "end_turn"}, "llm_output_invalid"),
            "wrong tool": (converse_response(ARGS, tool_name="other_tool"), "llm_output_invalid"),
            "two calls": ({"output": {"message": {"content": [
                {"toolUse": {"name": "submit_diagnosis_verdicts", "input": ARGS}},
                {"toolUse": {"name": "submit_diagnosis_verdicts", "input": ARGS}}]}}, "stopReason": "tool_use"}, "llm_output_invalid"),
            "bad input": (converse_response("not json"), "llm_output_invalid"),
            "malformed": (converse_response(ARGS, stop="malformed_tool_use"), "llm_output_invalid"),
        }
        for name, (response, code) in cases.items():
            with self.subTest(name), self.assertRaises(LlmOutputError) as ctx:
                meter = UsageMeter()
                try:
                    call(make_client(FakeBedrockRuntime(response))[0], meter=meter)
                finally:
                    self.assertEqual(meter.calls, 1)
            self.assertEqual(ctx.exception.code, code)

    def test_refusals_are_not_retried_as_output_errors(self):
        for stop in ("refusal", "guardrail_intervened", "content_filtered"):
            with self.subTest(stop), self.assertRaises(LlmRefusalError):
                call(make_client(FakeBedrockRuntime(converse_response(ARGS, stop=stop)))[0])


class TransportRetryTests(unittest.TestCase):
    def test_throttling_is_retried_with_backoff_then_succeeds(self):
        runtime = FakeBedrockRuntime(FakeClientError("ThrottlingException", 429), FakeClientError("ThrottlingException", 429),
                                     converse_response(ARGS))
        client, clock = make_client(runtime)
        call(client)
        self.assertEqual(len(runtime.requests), 3)
        self.assertEqual(clock.sleeps, [0.5, 1.0])

    def test_retries_are_bounded_and_the_final_error_is_typed(self):
        runtime = FakeBedrockRuntime(*[FakeClientError("ThrottlingException", 429)] * 3)
        with self.assertRaises(LlmThrottledError) as ctx:
            call(make_client(runtime)[0])
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.details["aws_error_code"], "ThrottlingException")
        self.assertEqual(len(runtime.requests), 3)

    def test_service_errors_and_network_errors_are_transient(self):
        for exc in (FakeClientError("ServiceUnavailableException", 503), FakeClientError("InternalServerException", 500),
                    FakeClientError("ModelTimeoutException", 408), TimeoutError("read timed out")):
            with self.subTest(type(exc).__name__):
                runtime = FakeBedrockRuntime(exc, converse_response(ARGS))
                call(make_client(runtime)[0])
                self.assertEqual(len(runtime.requests), 2)

    def test_access_and_request_errors_are_not_retried(self):
        runtime = FakeBedrockRuntime(FakeClientError("AccessDeniedException", 403, "model is not available for this account"))
        with self.assertRaises(LlmAccessError) as ctx:
            call(make_client(runtime)[0])
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(runtime.requests), 1)
        runtime = FakeBedrockRuntime(FakeClientError("ValidationException", 400, "temperature is not supported"))
        with self.assertRaises(LlmRequestError) as ctx:
            call(make_client(runtime)[0])
        self.assertIn("temperature is not supported", str(ctx.exception))
        self.assertEqual(len(runtime.requests), 1)

    def test_programming_errors_are_not_disguised_as_outages(self):
        with self.assertRaises(KeyError):
            call(make_client(FakeBedrockRuntime(KeyError("bug")))[0])

    def test_deadline_prevents_a_call_that_cannot_finish(self):
        runtime = FakeBedrockRuntime(converse_response(ARGS))
        client, _ = make_client(runtime, min_attempt_budget_s=8.0)
        with self.assertRaises(DeadlineExceededError):
            call(client, deadline=Deadline(5.0))
        self.assertEqual(runtime.requests, [])

    def test_no_retry_when_the_deadline_cannot_afford_another_attempt(self):
        clock = FakeClock()
        runtime = FakeBedrockRuntime(FakeClientError("ThrottlingException", 429), converse_response(ARGS))
        client, _ = make_client(runtime, clock=clock, min_attempt_budget_s=8.0)
        with self.assertRaises(LlmThrottledError):
            call(client, deadline=Deadline(8.4, clock=clock))  # 8.4 left < 0.5 backoff + 8 minimum attempt
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(clock.sleeps, [])

    def test_cancelled_request_does_not_start_new_calls(self):
        deadline = Deadline(50)
        deadline.cancel()
        runtime = FakeBedrockRuntime(converse_response(ARGS))
        with self.assertRaises(Exception) as ctx:
            call(make_client(runtime)[0], deadline=deadline)
        self.assertEqual(ctx.exception.code, "request_cancelled")
        self.assertEqual(runtime.requests, [])


if __name__ == "__main__":
    unittest.main()
