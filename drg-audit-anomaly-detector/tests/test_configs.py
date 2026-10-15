import json
import unittest
from dataclasses import fields

from tests.helpers import ROOT  # noqa: I001

import _awscfg as cfglib
import deploy_endpoint as dep
import launch_training_job as ltj

from drg_audit.config import FinetuneConfig, PretrainConfig, load_config
from drg_audit.postprocess.alerts import AlertsConfig
from drg_audit.schemas import AuditResult
from drg_audit.serving.contract import ParsedRecord, parse_request
from drg_audit.settings import RequestLimits


class ShippedConfigTests(unittest.TestCase):
    def test_stage_configs_equal_the_documented_defaults(self):
        self.assertEqual(load_config(PretrainConfig, ROOT / "configs" / "pretrain.yaml"), PretrainConfig())
        self.assertEqual(load_config(FinetuneConfig, ROOT / "configs" / "finetune.yaml"), FinetuneConfig())

    def test_cli_style_overrides(self):
        cfg = load_config(PretrainConfig, ROOT / "configs" / "pretrain.yaml",
                          ["epochs=3", "runtime.require_gpu=false", "learning_rate=2e-5", "runtime.checkpoint_dir=/tmp/c"])
        self.assertEqual((cfg.epochs, cfg.runtime.require_gpu, cfg.learning_rate), (3, False, 2e-5))
        self.assertEqual(cfg.runtime.checkpoint_dir, "/tmp/c")

    def test_alerts_config_loads_but_is_not_runnable_until_edited(self):
        cfg = AlertsConfig.from_yaml(ROOT / "configs" / "alerts.yaml")
        self.assertEqual(cfg.rule.alert_instance_id, 62206)
        self.assertEqual(cfg.eligibility.fund_ids, [102, 103])
        self.assertTrue(cfglib.find_placeholders(cfg))

    def test_deploy_example_parses_and_is_not_deployable_as_is(self):
        cfg = dep.load_deploy_config(ROOT / "configs" / "deploy.example.yaml")
        placeholders = cfglib.find_placeholders(cfg)
        for expected in ("region", "execution_role_arn", "model.kms_key_arn", "tags.Owner"):
            self.assertIn(expected, placeholders)
        self.assertTrue(cfg.security.enable_network_isolation)
        self.assertFalse(cfg.data_capture.enabled)
        self.assertFalse(cfg.security.allow_no_vpc)

    def test_train_job_example_parses_and_is_not_launchable_as_is(self):
        cfg = ltj.load_train_config(ROOT / "configs" / "train_job.example.yaml")
        self.assertIn("role_arn", cfglib.find_placeholders(cfg))
        self.assertTrue(cfg.security.enable_network_isolation)

    def test_serving_section_matches_request_limit_defaults(self):
        serving = load_config(FinetuneConfig, ROOT / "configs" / "finetune.yaml").serving
        self.assertEqual(serving.limits, RequestLimits())


class ExamplePayloadTests(unittest.TestCase):
    def test_example_request_is_valid(self):
        body = (ROOT / "examples" / "request.json").read_text()
        request = parse_request(body, "application/json", RequestLimits())
        self.assertEqual(len(request.valid), 2)
        self.assertTrue(all(isinstance(i, ParsedRecord) for i in request.items))
        self.assertEqual(request.confidence_threshold, 0.85)

    def test_example_response_has_the_documented_shape(self):
        response = json.loads((ROOT / "examples" / "response.json").read_text())
        self.assertEqual(list(response), ["request_id", "model_version", "confidence_threshold", "summary", "results"])
        self.assertEqual(set(response["summary"]),
                         {"n_records", "n_scored", "n_anomalies", "n_skipped", "n_rejected"})
        expected_fields = {f.name for f in fields(AuditResult)}
        for result in response["results"]:
            self.assertEqual(set(result), expected_fields)


if __name__ == "__main__":
    unittest.main()
