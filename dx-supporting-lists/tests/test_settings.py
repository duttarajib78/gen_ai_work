"""Configuration parsing and the production guards."""
from __future__ import annotations

import unittest

from dx_support.settings import ConfigError, Settings, environment_warnings

PROD = {"DX_ENV": "production", "DX_STORE_URI": "s3://my-bucket/store"}


class SettingsTests(unittest.TestCase):
    def test_defaults_are_safe_and_documented(self):
        s = Settings.from_env({})
        self.assertEqual((s.env, s.backend, s.model, s.thinking, s.inference_profile_prefix, s.aws_region), ("development", "bedrock", "claude-sonnet-5", "low", "au", "ap-southeast-2"))
        self.assertFalse(s.prompt_caching)
        self.assertTrue(s.allow_compute)
        self.assertEqual(s.stage_models(), {"brief": "claude-sonnet-5", "route": "claude-sonnet-5", "judge": "claude-sonnet-5"})
        self.assertTrue(s.store_uri.startswith("file://"))

    def test_values_are_parsed_and_stage_overrides_apply(self):
        s = Settings.from_env({"DX_MODEL": "claude-sonnet-5-5", "DX_JUDGE_MODEL": "claude-haiku-5-5", "DX_THINKING": "HIGH", "DX_LLM_CONCURRENCY": "32", "DX_PROMPT_CACHING": "yes",
                               "DX_MAX_OUTPUT_TOKENS_JUDGE": "20000", "DX_INFERENCE_PROFILE_PREFIX": "global", "AWS_REGION": "us-east-1"})
        self.assertEqual((s.thinking, s.llm_concurrency, s.prompt_caching, s.max_tokens["judge"], s.inference_profile_prefix, s.aws_region), ("high", 32, True, 20000, "global", "us-east-1"))
        self.assertEqual(s.stage_models(), {"brief": "claude-sonnet-5-5", "route": "claude-sonnet-5-5", "judge": "claude-haiku-5-5"})
        self.assertEqual(Settings.from_env({"DX_AWS_REGION": "ap-southeast-4", "AWS_REGION": "us-east-1"}).aws_region, "ap-southeast-4")

    def test_bad_values_fail_closed_with_the_variable_named_and_never_its_value(self):
        for env, name in [({"DX_THINKING": "turbo"}, "DX_THINKING"), ({"DX_LLM_CONCURRENCY": "0"}, "DX_LLM_CONCURRENCY"), ({"DX_LLM_CONCURRENCY": "lots"}, "DX_LLM_CONCURRENCY"),
                          ({"DX_BACKEND": "openai"}, "DX_BACKEND"), ({"DX_PROMPT_CACHING": "maybe"}, "DX_PROMPT_CACHING"), ({"DX_STORE_URI": "ftp://x"}, "DX_STORE_URI"),
                          ({"DX_GUARDRAIL_ID": "g-1"}, "DX_GUARDRAIL_ID"), ({"DX_SYNC_DEADLINE_S": "75"}, "DX_SYNC_DEADLINE_S"), ({"DX_LOG_LEVEL": "LOUD"}, "DX_LOG_LEVEL"),
                          ({"DX_MAX_OUTPUT_TOKENS_ROUTE": "5"}, "DX_MAX_OUTPUT_TOKENS_ROUTE"), ({"DX_INFERENCE_PROFILE_PREFIX": "moon"}, "DX_INFERENCE_PROFILE_PREFIX")]:
            with self.assertRaises(ConfigError, msg=name) as e:
                Settings.from_env(env)
            self.assertIn(name, str(e.exception))
        with self.assertRaises(ConfigError) as e:
            Settings.from_env({"DX_THINKING": "super-secret-looking-value"})
        self.assertNotIn("super-secret-looking-value", str(e.exception))

    def test_production_is_stricter(self):
        ok = Settings.from_env(PROD)
        self.assertTrue(ok.is_production)
        for env, needle in [({**PROD, "DX_BACKEND": "fake"}, "stub backend"), ({"DX_ENV": "production"}, "s3://"), ({**PROD, "DX_STORE_URI": "file:///tmp/x"}, "s3://"),
                            ({**PROD, "AWS_ACCESS_KEY_ID": "AKIA" "XXXXXXXXXXXXXXXX", "AWS_SECRET_ACCESS_KEY": "x"}, "Static AWS credentials"),
                            ({**PROD, "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://elsewhere"}, "endpoint override")]:
            with self.assertRaises(ConfigError, msg=env) as e:
                Settings.from_env(env)
            self.assertIn(needle, str(e.exception))
        self.assertNotIn("AKIA" "XXXXXXXXXXXXXXXX", str(ConfigError("Static AWS credentials are set in the environment (['AWS_ACCESS_KEY_ID'])")))
        Settings.from_env({**PROD, "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "https://gateway.internal", "DX_ALLOW_ENDPOINT_OVERRIDE": "true"})
        Settings.from_env({"DX_ENV": "production", "DX_STORE_URI": "file:///tmp/x", "DX_ALLOW_LOCAL_STORE": "true"})
        Settings.from_env({"DX_ENV": "development", "DX_BACKEND": "fake", "AWS_ACCESS_KEY_ID": "AKIA" "XXXXXXXXXXXXXXXX"})      # development tolerates both

    def test_warnings_name_variables_but_never_values(self):
        w = environment_warnings({"AWS_ACCESS_KEY_ID": "AKIA" "SECRETSECRETSECR", "AWS_ENDPOINT_URL": "http://x"})
        text = " ".join(w)
        self.assertIn("AWS_ACCESS_KEY_ID", text)
        self.assertIn("AWS_ENDPOINT_URL", text)
        self.assertNotIn("AKIA" "SECRETSECRETSECR", text)
        self.assertEqual(environment_warnings({}), [])


if __name__ == "__main__":
    unittest.main()
