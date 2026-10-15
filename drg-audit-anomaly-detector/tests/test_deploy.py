import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_labels, make_vocab  # noqa: I001

import _awscfg as cfglib
import _packaging as pk
import build_model_package as bmp
import deploy_endpoint as dep

from drg_audit._mapping import dataclass_from_mapping
from drg_audit.artifacts import write_bundle
from drg_audit.errors import ConfigError
from drg_audit.schemas import TokenizationSpec

ACCOUNT = "123456789012"
VALID = {
    "region": "ap-southeast-2", "stage": "dev", "endpoint_name": "drg-audit-test",
    "execution_role_arn": f"arn:aws:iam::{ACCOUNT}:role/drg-audit-endpoint",
    "model": {"s3_bucket": "example-model-bucket", "expected_bucket_owner": ACCOUNT,
              "kms_key_arn": f"arn:aws:kms:ap-southeast-2:{ACCOUNT}:key/11111111-2222-3333-4444-555555555555"},
    "container": {"image_uri": f"{ACCOUNT}.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:test"},
    "security": {"vpc": {"subnet_ids": ["subnet-0123456789abcdef0", "subnet-0fedcba9876543210"],
                         "security_group_ids": ["sg-0123456789abcdef0"]}},
    "tags": {"Owner": "ml-platform"},
}


def config(**overrides):
    data = json.loads(json.dumps(VALID))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return dataclass_from_mapping(dep.DeployConfig, data)


class ClientError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


class Waiter:
    def __init__(self, fail=False):
        self.fail = fail

    def wait(self, **kwargs):
        if self.fail:
            raise RuntimeError("waiter timeout")


class FakeSageMaker:
    def __init__(self, existing=False, rolled_back=False):
        self.calls, self.exists, self.rolled_back = [], existing, rolled_back
        self.config_name = "previous-config"

    def _rec(self, name, kwargs):
        self.calls.append((name, kwargs))

    def names(self):
        return [n for n, _ in self.calls]

    def args(self, name):
        return next(kw for n, kw in self.calls if n == name)

    def describe_endpoint(self, **kw):
        if not self.exists:
            raise ClientError("ValidationException", f'Could not find endpoint "{kw["EndpointName"]}".')
        return {"EndpointStatus": "InService", "EndpointConfigName": self.config_name,
                "ProductionVariants": [{"VariantName": "AllTraffic", "CurrentInstanceCount": 2,
                                        "DesiredInstanceCount": 2}]}

    def describe_endpoint_config(self, **kw):
        return {"ProductionVariants": [{"ModelName": "model-x"}]}

    def create_model(self, **kw):
        self._rec("create_model", kw)

    def create_endpoint_config(self, **kw):
        self._rec("create_endpoint_config", kw)

    def create_endpoint(self, **kw):
        self._rec("create_endpoint", kw)
        self.exists, self.config_name = True, kw["EndpointConfigName"]

    def update_endpoint(self, **kw):
        self._rec("update_endpoint", kw)
        if not self.rolled_back:
            self.config_name = kw["EndpointConfigName"]

    def delete_endpoint(self, **kw):
        self._rec("delete_endpoint", kw)

    def delete_endpoint_config(self, **kw):
        self._rec("delete_endpoint_config", kw)

    def delete_model(self, **kw):
        self._rec("delete_model", kw)

    def get_waiter(self, name):
        self._rec("get_waiter", {"name": name})
        return Waiter()


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(**kw):
            self.calls.append((name, kw))
        return call

    def args(self, name):
        return [kw for n, kw in self.calls if n == name]


class FakeS3:
    def __init__(self, objects=None):
        self.objects, self.uploads = dict(objects or {}), []

    def head_object(self, Bucket, Key, ExpectedBucketOwner):
        if Key not in self.objects:
            raise ClientError("404")
        return {"Metadata": {"sha256": self.objects[Key]}}

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads.append((key, ExtraArgs))
        self.objects[key] = ExtraArgs["Metadata"]["sha256"]


class FakeRuntime:
    def invoke_endpoint(self, **kw):
        body = {"model_version": "abc", "summary": {"n_records": 2, "n_anomalies": 1}, "results": [{"x": 1}]}
        return {"Body": io.BytesIO(json.dumps(body).encode())}


class Sts:
    def __init__(self, account=ACCOUNT):
        self.account = account

    def get_caller_identity(self):
        return {"Account": self.account, "Arn": f"arn:aws:sts::{self.account}:assumed-role/deployer/session"}


class DeployTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        vocab = make_vocab(["a"])
        bundle = write_bundle(self.root / "bundle", write_weights=lambda p: p.write_bytes(b"w"),
                              encoder_config={"vocab_size": len(vocab)}, classifier_config={"hidden_dim": 8},
                              vocab=vocab, labels=make_labels(), inference_config={},
                              tokenization=TokenizationSpec.standard())

        def fake_vendor(requirements, wheelhouse, *, python_version, platforms):
            wheelhouse.mkdir(parents=True)
            (wheelhouse / "x-1-py3-none-any.whl").write_bytes(b"w")
            return [{"file": "x-1-py3-none-any.whl", "sha256": "x", "bytes": 1}]

        with mock.patch.object(pk, "vendor_wheels", side_effect=fake_vendor):
            self.meta = bmp.build_package(bundle_dir=bundle, output=self.root / "dist" / "model.tar.gz")
        self.package = self.root / "dist" / "model.tar.gz"
        self.sm, self.aas, self.cw, self.s3 = FakeSageMaker(), Recorder(), Recorder(), FakeS3()

    def deployer(self, cfg=None, sm=None, s3=None, sts=None):
        clients = dep.AwsClients(sagemaker=sm or self.sm, runtime=FakeRuntime(), s3=s3 or self.s3,
                                 autoscaling=self.aas, cloudwatch=self.cw, sts=sts or Sts())
        return dep.Deployer(cfg or config(), clients)

    # ---- happy paths --------------------------------------------------------------
    def test_create_flow_applies_secure_defaults(self):
        result = self.deployer().deploy(self.package)
        self.assertEqual(result["mode"], "create")
        model = self.sm.args("create_model")
        self.assertTrue(model["EnableNetworkIsolation"])
        self.assertEqual(model["VpcConfig"]["Subnets"], VALID["security"]["vpc"]["subnet_ids"])
        self.assertEqual(model["PrimaryContainer"]["Environment"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(model["PrimaryContainer"]["Environment"]["SAGEMAKER_PROGRAM"], "inference.py")
        self.assertTrue(model["PrimaryContainer"]["ModelDataUrl"].startswith("s3://example-model-bucket/"))
        self.assertIn(self.meta["bundle_id"], model["PrimaryContainer"]["ModelDataUrl"])
        epc = self.sm.args("create_endpoint_config")
        self.assertEqual(epc["KmsKeyId"], VALID["model"]["kms_key_arn"])
        self.assertNotIn("DataCaptureConfig", epc)
        variant = epc["ProductionVariants"][0]
        self.assertEqual((variant["InstanceType"], variant["InitialInstanceCount"]), ("ml.m5.xlarge", 2))
        self.assertIn("create_endpoint", self.sm.names())
        self.assertNotIn("update_endpoint", self.sm.names())
        key, extra = self.s3.uploads[0]
        self.assertEqual(extra["ServerSideEncryption"], "aws:kms")
        self.assertEqual(extra["ExpectedBucketOwner"], ACCOUNT)
        self.assertEqual(extra["Metadata"]["sha256"], self.meta["sha256"])

    def test_alarms_and_autoscaling(self):
        self.deployer().deploy(self.package)
        alarms = {kw["AlarmName"]: kw for kw in self.cw.args("put_metric_alarm")}
        self.assertEqual(set(alarms), {"drg-audit-test-5xx", "drg-audit-test-latency-p95", "drg-audit-test-memory"})
        self.assertEqual(alarms["drg-audit-test-latency-p95"]["Threshold"], 500_000.0)  # ms -> microseconds
        self.assertEqual(alarms["drg-audit-test-latency-p95"]["ExtendedStatistic"], "p95")
        self.assertNotIn("Statistic", alarms["drg-audit-test-latency-p95"])
        target = self.aas.args("register_scalable_target")[0]
        self.assertEqual(target["ResourceId"], "endpoint/drg-audit-test/variant/AllTraffic")
        policy = self.aas.args("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        self.assertEqual(policy["PredefinedMetricSpecification"]["PredefinedMetricType"],
                         "SageMakerVariantInvocationsPerInstance")

    def test_update_is_blue_green_with_alarm_rollback(self):
        sm = FakeSageMaker(existing=True)
        result = self.deployer(sm=sm).deploy(self.package)
        self.assertEqual(result["mode"], "blue_green_update")
        deployment = sm.args("update_endpoint")["DeploymentConfig"]
        traffic = deployment["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]
        self.assertEqual(traffic["Type"], "CANARY")
        self.assertEqual(traffic["CanarySize"], {"Type": "CAPACITY_PERCENT", "Value": 25})
        self.assertEqual([a["AlarmName"] for a in deployment["AutoRollbackConfiguration"]["Alarms"]],
                         ["drg-audit-test-5xx", "drg-audit-test-latency-p95"])
        self.assertNotIn("create_endpoint", sm.names())

    def test_alarm_driven_rollback_is_reported_as_failure(self):
        sm = FakeSageMaker(existing=True, rolled_back=True)
        with self.assertRaisesRegex(dep.DeployError, "rollback"):
            self.deployer(sm=sm).deploy(self.package)

    def test_smoke_test_returns_only_aggregates(self):
        payload = self.root / "req.json"
        payload.write_text("{}")
        result = self.deployer().deploy(self.package, smoke_test=payload)
        self.assertEqual(result["smoke_test"], {"model_version": "abc", "summary": {"n_records": 2, "n_anomalies": 1}})

    def test_data_capture_is_opt_in_and_encrypted(self):
        cfg = config(data_capture={"enabled": True, "s3_destination": "s3://example-capture-bucket/capture"})
        self.deployer(cfg).deploy(self.package)
        capture = self.sm.args("create_endpoint_config")["DataCaptureConfig"]
        self.assertEqual(capture["KmsKeyId"], VALID["model"]["kms_key_arn"])
        self.assertEqual({o["CaptureMode"] for o in capture["CaptureOptions"]}, {"Input", "Output"})

    # ---- policy ---------------------------------------------------------------------
    def test_upload_is_content_addressed_and_idempotent(self):
        self.deployer().deploy(self.package)
        key = self.s3.uploads[0][0]
        self.assertIn(self.meta["sha256"][:16], key)
        s3 = FakeS3({key: self.meta["sha256"]})
        self.deployer(s3=s3, sm=FakeSageMaker()).deploy(self.package)
        self.assertEqual(s3.uploads, [])  # identical object already present: no re-upload

    def test_digest_collision_at_key_is_refused(self):
        self.deployer().deploy(self.package)
        key = self.s3.uploads[0][0]
        with self.assertRaisesRegex(dep.DeployError, "different digest"):
            self.deployer(s3=FakeS3({key: "0" * 64}), sm=FakeSageMaker()).deploy(self.package)

    def test_modified_package_is_refused_before_any_aws_call(self):
        self.package.write_bytes(self.package.read_bytes() + b"x")
        sm = FakeSageMaker()
        with self.assertRaisesRegex(dep.DeployError, "package_meta"):
            self.deployer(sm=sm).deploy(self.package)
        self.assertEqual(sm.calls, [])

    def test_placeholders_block_deployment(self):
        cfg = dataclass_from_mapping(dep.DeployConfig, {**VALID, "execution_role_arn": "REPLACE_ME"})
        with self.assertRaisesRegex(ConfigError, "execution_role_arn"):
            self.deployer(cfg).deploy(self.package)

    def test_vpc_is_required_unless_explicitly_waived(self):
        no_vpc = config(security={"vpc": {"subnet_ids": [], "security_group_ids": []}})
        with self.assertRaisesRegex(ConfigError, "allow_no_vpc"):
            self.deployer(no_vpc).deploy(self.package)
        waived = config(security={"allow_no_vpc": True, "vpc": {"subnet_ids": [], "security_group_ids": []}})
        self.deployer(waived).deploy(self.package)
        self.assertNotIn("VpcConfig", self.sm.args("create_model"))

    def test_pypi_package_cannot_use_network_isolation(self):
        meta_path = self.package.parent / "package_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["dependency_mode"] = "pypi"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ConfigError, "network isolation"):
            self.deployer().deploy(self.package)

    def test_prod_policy(self):
        base = {"stage": "prod", "endpoint_name": "drg-audit-prod"}
        with self.assertRaisesRegex(ConfigError, "alarms.sns_topic_arn"):
            self.deployer(config(**base)).deploy(self.package)
        one_az = config(**base, alarms={"sns_topic_arn": f"arn:aws:sns:ap-southeast-2:{ACCOUNT}:alerts"},
                        security={"vpc": {"subnet_ids": ["subnet-0123456789abcdef0"],
                                          "security_group_ids": ["sg-0123456789abcdef0"]}})
        with self.assertRaisesRegex(ConfigError, "Availability Zones"):
            self.deployer(one_az).deploy(self.package)
        ok = config(**base, alarms={"sns_topic_arn": f"arn:aws:sns:ap-southeast-2:{ACCOUNT}:alerts"})
        self.deployer(ok).deploy(self.package)

    def test_role_from_another_account_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "different account"):
            self.deployer(sts=Sts("999999999999")).deploy(self.package)

    def test_missing_image_uri_without_sdk_is_a_clear_error(self):
        cfg = config(container={"image_uri": None})
        with mock.patch.dict(sys.modules, {"sagemaker": None}), self.assertRaisesRegex(ConfigError, "image_uri"):
            self.deployer(cfg).deploy(self.package)

    # ---- other commands -------------------------------------------------------------
    def test_rollback_and_status_and_delete(self):
        sm = FakeSageMaker(existing=True)
        deployer = self.deployer(sm=sm)
        self.assertEqual(deployer.status()["status"], "InService")
        deployer.rollback("previous-config")
        self.assertEqual(sm.args("update_endpoint")["EndpointConfigName"], "previous-config")
        deployer.delete(delete_models=True)
        self.assertIn("delete_endpoint", sm.names())
        self.assertEqual(sm.args("delete_model"), {"ModelName": "model-x"})
        self.assertEqual(len(self.aas.args("deregister_scalable_target")), 1)
        self.assertEqual(len(self.cw.args("delete_alarms")[0]["AlarmNames"]), 3)

    def test_cli_refuses_delete_without_confirmation(self):
        cfg_path = self.root / "deploy.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(VALID))
        with mock.patch.object(cfglib, "make_session"), mock.patch.object(dep.AwsClients, "from_session"):
            self.assertEqual(dep.main(["delete", "--config", str(cfg_path)]), 1)


class ConfigValidationTests(unittest.TestCase):
    def test_structural_validation(self):
        bad = [
            {"endpoint_name": "x" * 40}, {"endpoint_name": "bad_name"}, {"stage": "qa"}, {"region": "mars"},
            {"execution_role_arn": "arn:aws:iam::1:role/x"},
            {"endpoint": {"initial_instance_count": 9}},  # outside autoscaling max
            {"blue_green": {"canary_size_percent": 80}},
            {"data_capture": {"enabled": True}},
            {"model": {"s3_prefix": "../escape"}},
            {"security": {"vpc": {"subnet_ids": ["not-a-subnet"]}}},
        ]
        for override in bad:
            with self.subTest(override=override), self.assertRaises(ConfigError):
                config(**override)

    def test_placeholder_scan(self):
        cfg = dataclass_from_mapping(dep.DeployConfig, {})
        found = cfglib.find_placeholders(cfg)
        self.assertIn("region", found)
        self.assertIn("model.kms_key_arn", found)


if __name__ == "__main__":
    unittest.main()
