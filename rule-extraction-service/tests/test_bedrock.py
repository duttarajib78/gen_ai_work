import unittest
from pathlib import Path

from rule_extractor.errors import ConfigError
from rule_extractor.llm.bedrock import BLANKED_ENV, build_sdk_env, resolve_model_id, resolved_models
from rule_extractor.settings import Settings


def settings(**env):
    base = {"AWS_REGION": "ap-southeast-2"}
    base.update(env)
    return Settings.load(env=base)


class BedrockTests(unittest.TestCase):
    def test_prefix_applied_once(self):
        self.assertEqual(resolve_model_id("anthropic.claude-sonnet-5", "au"), "au.anthropic.claude-sonnet-5")
        self.assertEqual(resolve_model_id("global.anthropic.claude-sonnet-5", "au"), "global.anthropic.claude-sonnet-5")
        self.assertEqual(resolve_model_id("anthropic.claude-haiku-4-5-20251001-v1:0", ""), "anthropic.claude-haiku-4-5-20251001-v1:0")

    def test_arn_passthrough_and_validation(self):
        arn = "arn:aws:bedrock:ap-southeast-2:123456789012:application-inference-profile/abc123"
        self.assertEqual(resolve_model_id(arn, "au"), arn)
        for bad in ("", "x", "model id with spaces", "a;b", "$(id)", "arn:aws:s3:::bucket"):
            with self.assertRaises(ConfigError):
                resolve_model_id(bad, "au")

    def test_mantle_ids_are_not_prefixed(self):
        self.assertEqual(resolve_model_id("anthropic.claude-sonnet-5", "au", "mantle"), "anthropic.claude-sonnet-5")

    def test_default_models_resolve_to_au_profiles(self):
        m = resolved_models(settings())
        self.assertEqual(m["strong"], "au.anthropic.claude-sonnet-5")
        self.assertEqual(m["fast"], "au.anthropic.claude-haiku-4-5-20251001-v1:0")

    def test_env_routes_to_bedrock_and_hardens_cli(self):
        env = build_sdk_env(settings(), config_dir=Path("/tmp/x"))
        self.assertEqual(env["CLAUDE_CODE_USE_BEDROCK"], "1")
        self.assertEqual(env["CLAUDE_CODE_USE_MANTLE"], "")
        self.assertEqual(env["AWS_REGION"], "ap-southeast-2")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertEqual(env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "1")
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "/tmp/x")
        for name in BLANKED_ENV:
            self.assertEqual(env[name], "")
        self.assertNotIn("AWS_BEARER_TOKEN_BEDROCK", build_sdk_env(settings(), config_dir=Path("/tmp/x")))

    def test_prod_blanks_bedrock_api_key_and_mantle_switches_flag(self):
        env = build_sdk_env(settings(RULE_EXTRACTOR_ENV="prod", RULE_EXTRACTOR_BEDROCK_PROVIDER="mantle"), config_dir=Path("/tmp/x"))
        self.assertEqual(env["AWS_BEARER_TOKEN_BEDROCK"], "")
        self.assertEqual(env["CLAUDE_CODE_USE_MANTLE"], "1")
        self.assertEqual(env["CLAUDE_CODE_USE_BEDROCK"], "")

    def test_region_required_and_validated(self):
        with self.assertRaises(ConfigError):
            build_sdk_env(Settings.load(env={}), config_dir=Path("/tmp/x"))
        with self.assertRaises(ConfigError):
            build_sdk_env(settings(AWS_REGION="us east"), config_dir=Path("/tmp/x"))

    def test_no_credentials_are_ever_added(self):
        env = build_sdk_env(settings(), config_dir=Path("/tmp/x"))
        for key, value in env.items():
            if key in BLANKED_ENV or key == "AWS_BEARER_TOKEN_BEDROCK":
                continue
            self.assertFalse(any(t in key for t in ("SECRET", "TOKEN", "PASSWORD", "ACCESS_KEY")), key)


if __name__ == "__main__":
    unittest.main()
