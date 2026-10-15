import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.helpers  # noqa: F401,I001  (puts src/ on sys.path)

import _packaging as pk
import launch_training_job as ltj

from drg_audit._mapping import dataclass_from_mapping
from drg_audit.errors import ConfigError

ACCOUNT = "123456789012"
KMS = f"arn:aws:kms:ap-southeast-2:{ACCOUNT}:key/11111111-2222-3333-4444-555555555555"
VALID = {
    "region": "ap-southeast-2",
    "role_arn": f"arn:aws:iam::{ACCOUNT}:role/drg-audit-training",
    "image_uri": f"{ACCOUNT}.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:test",
    "inputs": {"train_s3_uri": "s3://example-data/train/", "source_s3_prefix": "s3://example-data/source",
               "expected_bucket_owner": ACCOUNT},
    "output": {"s3_output_path": "s3://example-data/output"},
    "kms": {"output_key_arn": KMS, "volume_key_arn": KMS},
    "security": {"vpc": {"subnet_ids": ["subnet-0123456789abcdef0"],
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
    return dataclass_from_mapping(ltj.TrainJobConfig, data)


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "get_waiter":
                return mock.Mock()
            if name == "describe_training_job":
                return {"TrainingJobStatus": "Completed",
                        "ModelArtifacts": {"S3ModelArtifacts": "s3://example-data/output/job/output/model.tar.gz"}}
            return None
        return call

    def kwargs(self, name):
        return next(kw for n, _, kw in self.calls if n == name)


class Sts:
    def __init__(self, account=ACCOUNT):
        self.account = account

    def get_caller_identity(self):
        return {"Account": self.account}


class SourceBundleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def names(self, bundle):
        with tarfile.open(bundle["path"]) as tar:
            return {m.name for m in tar.getmembers()}, tar

    def test_contents_and_exclusions(self):
        bundle = ltj.build_source_bundle(self.out, dependency_mode="pypi", python_version="3.12")
        with tarfile.open(bundle["path"]) as tar:
            names = {m.name for m in tar.getmembers()}
            requirements = tar.extractfile("requirements.txt").read().decode()
        for expected in ("train_entry.py", "configs/pretrain.yaml", "configs/finetune.yaml", "requirements.txt",
                         "src/drg_audit/training/pretrain.py", "src/drg_audit/cli.py"):
            self.assertIn(expected, names)
        for forbidden in ("configs/deploy.example.yaml", "configs/alerts.yaml", "configs/train_job.example.yaml"):
            self.assertNotIn(forbidden, names)
        self.assertNotIn("torch", requirements)  # the training container already provides torch
        self.assertIn("pandas", requirements)
        self.assertEqual(bundle["sha256"], pk.sha256_file(bundle["path"]))

    def test_wheelhouse_uses_a_relative_find_links(self):
        def fake_vendor(requirements, wheelhouse, *, python_version, platforms):
            wheelhouse.mkdir(parents=True)
            (wheelhouse / "x-1-py3-none-any.whl").write_bytes(b"w")
            return [{"file": "x-1-py3-none-any.whl", "sha256": "x", "bytes": 1}]

        with mock.patch.object(pk, "vendor_wheels", side_effect=fake_vendor):
            bundle = ltj.build_source_bundle(self.out, dependency_mode="wheelhouse", python_version="3.12")
        with tarfile.open(bundle["path"]) as tar:
            text = tar.extractfile("requirements.txt").read().decode()
            self.assertIn("wheelhouse/x-1-py3-none-any.whl", {m.name for m in tar.getmembers()})
        self.assertIn("--no-index", text)
        self.assertIn("--find-links=wheelhouse", text)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        archive = Path(self._tmp.name) / "sourcedir.tar.gz"
        archive.write_bytes(b"source")
        self.source = {"path": archive, "sha256": "ab" * 32, "wheelhouse": []}
        self.sm, self.s3 = Recorder(), Recorder()

    def launcher(self, cfg=None, sts=None):
        return ltj.TrainingLauncher(cfg or config(), sagemaker=self.sm, s3=self.s3, sts=sts or Sts())

    def test_job_parameters_apply_secure_defaults(self):
        result = self.launcher().launch(self.source)
        params = self.sm.kwargs("create_training_job")
        self.assertTrue(params["EnableNetworkIsolation"])
        self.assertTrue(params["EnableInterContainerTrafficEncryption"])
        self.assertEqual(params["OutputDataConfig"]["KmsKeyId"], KMS)
        self.assertEqual(params["ResourceConfig"]["VolumeKmsKeyId"], KMS)
        self.assertEqual(params["ResourceConfig"]["InstanceCount"], 1)
        self.assertEqual(params["VpcConfig"]["SecurityGroupIds"], ["sg-0123456789abcdef0"])
        channels = {c["ChannelName"]: c for c in params["InputDataConfig"]}
        self.assertEqual(set(channels), {"train", "code"})
        self.assertTrue(channels["code"]["DataSource"]["S3DataSource"]["S3Uri"].endswith("/sourcedir.tar.gz"))
        hyper = params["HyperParameters"]
        self.assertEqual(json.loads(hyper["sagemaker_program"]), "train_entry.py")
        self.assertEqual(json.loads(hyper["sagemaker_submit_directory"]), ltj.CONTAINER_CODE_PATH)
        self.assertEqual(json.loads(hyper["stage"]), "all")
        self.assertEqual(params["Environment"]["HF_HUB_OFFLINE"], "1")
        upload = self.s3.kwargs("upload_file")["ExtraArgs"]
        self.assertEqual((upload["ServerSideEncryption"], upload["SSEKMSKeyId"]), ("aws:kms", KMS))
        self.assertEqual(upload["ExpectedBucketOwner"], ACCOUNT)
        self.assertTrue(result["training_job_name"].startswith("drg-audit-train-"))

    def test_spot_training_adds_checkpointing(self):
        cfg = config(spot={"enabled": True}, output={"checkpoint_s3_uri": "s3://example-data/ckpt"})
        self.launcher(cfg).launch(self.source)
        params = self.sm.kwargs("create_training_job")
        self.assertTrue(params["EnableManagedSpotTraining"])
        self.assertEqual(params["StoppingCondition"]["MaxWaitTimeInSeconds"], 172_800)
        self.assertEqual(params["CheckpointConfig"]["LocalPath"], "/opt/ml/checkpoints")

    def test_pretrained_channel_for_finetune_only(self):
        cfg = config(inputs={"pretrained_s3_uri": "s3://example-data/pretrained/"},
                     hyperparameters={"stage": "finetune"})
        self.launcher(cfg).launch(self.source)
        names = [c["ChannelName"] for c in self.sm.kwargs("create_training_job")["InputDataConfig"]]
        self.assertIn("pretrained", names)

    def test_wait_returns_the_model_artifact(self):
        result = self.launcher().launch(self.source, wait=True)
        self.assertEqual(result["status"], "Completed")
        self.assertTrue(result["model_artifact"].endswith("model.tar.gz"))

    def test_policy_failures_happen_before_any_aws_call(self):
        cases = [
            (dataclass_from_mapping(ltj.TrainJobConfig, {**VALID, "role_arn": "REPLACE_ME"}), "role_arn"),
            (config(security={"vpc": {"subnet_ids": [], "security_group_ids": []}}), "allow_no_vpc"),
            (config(dependency_mode="pypi"), "network isolation"),
        ]
        for cfg, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ConfigError, message):
                self.launcher(cfg).launch(self.source)
        self.assertEqual(self.sm.calls, [])
        self.assertEqual(self.s3.calls, [])

    def test_role_account_mismatch(self):
        with self.assertRaisesRegex(ConfigError, "different account"):
            self.launcher(sts=Sts("999999999999")).launch(self.source)

    def test_structural_validation(self):
        for override in ({"spot": {"enabled": True}}, {"job_name_prefix": "bad_name"}, {"dependency_mode": "curl"},
                         {"inputs": {"train_s3_uri": "https://not-s3"}}, {"volume_size_gb": 0}):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                config(**override)


if __name__ == "__main__":
    unittest.main()
