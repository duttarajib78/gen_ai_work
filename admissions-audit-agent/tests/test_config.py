import unittest

from audit_agent.config import Settings
from audit_agent.errors import ConfigError
from tests.fakes import BASE_ENV, make_settings


class SettingsTests(unittest.TestCase):
    def test_minimal_valid_environment(self):
        s = make_settings()
        self.assertEqual(s.llm_effort, "high")
        self.assertEqual(s.request_deadline_s, 50.0)
        self.assertEqual(s.low_confidence_weak_threshold, 0.5)
        self.assertEqual(s.routing_scope, "global")
        self.assertFalse(s.log_raw_identifiers)

    def test_missing_required_values_are_all_reported_together(self):
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_mapping({})
        text = str(ctx.exception)
        self.assertIn("AUDIT_AWS_REGION", text)
        self.assertIn("AUDIT_BEDROCK_MODEL_ID", text)

    def test_region_falls_back_to_aws_region(self):
        s = Settings.from_mapping({"AWS_REGION": "ap-southeast-2", "AUDIT_BEDROCK_MODEL_ID": "au.anthropic.claude-sonnet-5"})
        self.assertEqual(s.aws_region, "ap-southeast-2")
        self.assertEqual(s.routing_scope, "geo:au")

    def test_global_profile_requires_explicit_opt_in(self):
        env = {**BASE_ENV}
        del env["AUDIT_ALLOW_GLOBAL_INFERENCE"]
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_mapping(env)
        self.assertIn("AUDIT_ALLOW_GLOBAL_INFERENCE", str(ctx.exception))

    def test_cross_region_bedrock_requires_explicit_opt_in(self):
        with self.assertRaises(ConfigError):
            make_settings(AUDIT_BEDROCK_REGION="us-east-1", AUDIT_BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-5-5")
        s = make_settings(
            AUDIT_BEDROCK_REGION="us-east-1", AUDIT_BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-5-5",
            AUDIT_ALLOW_CROSS_REGION_BEDROCK="true",
        )
        self.assertEqual(s.bedrock_region, "us-east-1")

    def test_inference_profile_arn_accepted_and_flagged_unknown_scope(self):
        arn = "arn:aws:bedrock:ap-southeast-2:123456789012:application-inference-profile/abc123"
        self.assertEqual(make_settings(AUDIT_BEDROCK_MODEL_ID=arn).routing_scope, "unknown-arn")

    def test_rejects_non_claude_or_malformed_model_ids(self):
        for bad in ("gpt-4", "global.anthropic.claude sonnet", "anthropic.claude-sonnet-5-5; drop"):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                make_settings(AUDIT_BEDROCK_MODEL_ID=bad)

    def test_value_validation(self):
        cases = {
            "AUDIT_LLM_EFFORT": "ultra",
            "AUDIT_LOW_CONFIDENCE_WEAK_THRESHOLD": "1.5",
            "AUDIT_REQUEST_DEADLINE_S": "120",
            "AUDIT_STREAM_DEADLINE_S": "900",
            "AUDIT_MAX_REQUEST_BYTES": "10",
            "AUDIT_LLM_MAX_OUTPUT_ATTEMPTS": "nine",
            "AUDIT_ALLOW_GLOBAL_INFERENCE": "maybe",
            "AUDIT_GROUPER_BACKEND": "other",
        }
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaises(ConfigError) as ctx:
                make_settings(**{key: value})
            self.assertIn(key, str(ctx.exception))

    def test_between_tools_is_limited_to_effort_high_or_below(self):
        make_settings(AUDIT_LLM_THINKING_MODE="between_tools", AUDIT_LLM_EFFORT="medium")
        with self.assertRaises(ConfigError):
            make_settings(AUDIT_LLM_THINKING_MODE="between_tools", AUDIT_LLM_EFFORT="max")

    def test_output_token_ceiling_must_cover_default(self):
        with self.assertRaises(ConfigError):
            make_settings(AUDIT_LLM_MAX_OUTPUT_TOKENS="20000", AUDIT_LLM_MAX_OUTPUT_TOKENS_CEILING="10000")

    def test_extra_request_fields_must_be_a_json_object(self):
        s = make_settings(AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON='{"thinking": {"type": "between_tools"}}')
        self.assertEqual(s.llm_extra_request_fields["thinking"]["type"], "between_tools")
        for bad in ("[1]", "not json", '"x"'):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                make_settings(AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON=bad)

    def test_guardrail_id_and_version_come_together(self):
        with self.assertRaises(ConfigError):
            make_settings(AUDIT_BEDROCK_GUARDRAIL_ID="gr-1")
        s = make_settings(AUDIT_BEDROCK_GUARDRAIL_ID="gr-1", AUDIT_BEDROCK_GUARDRAIL_VERSION="3")
        self.assertEqual(s.guardrail_version, "3")

    def test_describe_contains_no_free_form_values(self):
        d = make_settings(AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON='{"k": "secret-ish"}').describe()
        self.assertNotIn("secret-ish", str(d))


if __name__ == "__main__":
    unittest.main()
