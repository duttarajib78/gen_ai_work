#!/usr/bin/env python
"""Launch the training pipeline as a SageMaker Training Job (boto3 only).

    python sagemaker/launch_training_job.py --config configs/train_job.dev.yaml --wait

Builds a source bundle (train_entry.py + configs + the full drg_audit package + dependencies), uploads it with
SSE-KMS, and starts a job with secure defaults: network isolation, VPC attachment, KMS-encrypted output and
volume, inter-container traffic encryption. The job writes the serving bundle to its model.tar.gz; feed that to
``sagemaker/build_model_package.py --model-tar``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _awscfg as cfglib  # noqa: E402  (also puts src/ on sys.path)
import _packaging as pk
from _awscfg import SecuritySettings, tags_to_aws

from drg_audit._mapping import dataclass_from_mapping  # noqa: E402
from drg_audit.config import load_yaml  # noqa: E402
from drg_audit.errors import ConfigError  # noqa: E402
from drg_audit.logging_utils import configure_logging, get_logger, log_event  # noqa: E402

logger = get_logger("train_job")
JOB_PREFIX_RE = re.compile(r"[A-Za-z0-9](?:-?[A-Za-z0-9]){0,39}")
CONTAINER_CODE_PATH = "/opt/ml/input/data/code/sourcedir.tar.gz"
TRAIN_REQUIREMENTS = pk.REPO_ROOT / "requirements" / "train_job.in"
TRAINING_CONFIGS = ("pretrain.yaml", "finetune.yaml")


@dataclass(frozen=True)
class SpotSettings:
    enabled: bool = False
    max_wait_seconds: int = 172_800


@dataclass(frozen=True)
class InputSettings:
    train_s3_uri: str = cfglib.PLACEHOLDER
    pretrained_s3_uri: str | None = None
    source_s3_prefix: str = cfglib.PLACEHOLDER
    expected_bucket_owner: str = cfglib.PLACEHOLDER


@dataclass(frozen=True)
class OutputSettings:
    s3_output_path: str = cfglib.PLACEHOLDER
    checkpoint_s3_uri: str | None = None


@dataclass(frozen=True)
class KmsSettings:
    output_key_arn: str = cfglib.PLACEHOLDER
    volume_key_arn: str = cfglib.PLACEHOLDER


@dataclass(frozen=True)
class TrainJobConfig:
    region: str = cfglib.PLACEHOLDER
    role_arn: str = cfglib.PLACEHOLDER
    job_name_prefix: str = "drg-audit-train"
    image_uri: str | None = None
    framework_version: str = "2.6.0"
    py_version: str = "py312"
    instance_type: str = "ml.g5.xlarge"
    volume_size_gb: int = 200
    max_runtime_seconds: int = 86_400
    spot: SpotSettings = field(default_factory=SpotSettings)
    inputs: InputSettings = field(default_factory=InputSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    kms: KmsSettings = field(default_factory=KmsSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    dependency_mode: str = "wheelhouse"
    python_version: str = "3.12"
    hyperparameters: dict[str, str] = field(default_factory=lambda: {"stage": "all"})
    tags: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not JOB_PREFIX_RE.fullmatch(self.job_name_prefix):
            raise ConfigError("job_name_prefix must be <= 40 characters: letters, digits and single hyphens")
        cfglib.check_pattern(self.region, cfglib.REGION_RE, "region")
        cfglib.check_pattern(self.role_arn, cfglib.ROLE_ARN_RE, "role_arn")
        cfglib.check_pattern(self.kms.output_key_arn, cfglib.KMS_ARN_RE, "kms.output_key_arn")
        cfglib.check_pattern(self.kms.volume_key_arn, cfglib.KMS_ARN_RE, "kms.volume_key_arn")
        cfglib.check_pattern(self.inputs.expected_bucket_owner, cfglib.ACCOUNT_RE, "inputs.expected_bucket_owner")
        for label, uri in (("inputs.train_s3_uri", self.inputs.train_s3_uri),
                           ("inputs.source_s3_prefix", self.inputs.source_s3_prefix),
                           ("output.s3_output_path", self.output.s3_output_path)):
            cfglib.check_pattern(uri, cfglib.S3_URI_RE, label)
        if self.dependency_mode not in ("wheelhouse", "pypi"):
            raise ConfigError("dependency_mode must be 'wheelhouse' or 'pypi'")
        if self.spot.enabled and not self.output.checkpoint_s3_uri:
            raise ConfigError("spot training requires output.checkpoint_s3_uri so interrupted jobs can resume")
        if self.spot.enabled and self.spot.max_wait_seconds < self.max_runtime_seconds:
            raise ConfigError("spot.max_wait_seconds must be >= max_runtime_seconds")
        if not 1 <= self.volume_size_gb <= 16_384 or self.max_runtime_seconds < 60:
            raise ConfigError("volume_size_gb or max_runtime_seconds is out of range")

    def enforce_policy(self) -> None:
        cfglib.require_no_placeholders(self)
        self.security.enforce_policy()
        if self.dependency_mode == "pypi" and self.security.enable_network_isolation:
            raise ConfigError("dependency_mode 'pypi' needs outbound network access and cannot be combined with "
                              "network isolation; use the default 'wheelhouse' mode (recommended)")


def load_train_config(path: str | Path) -> TrainJobConfig:
    return dataclass_from_mapping(TrainJobConfig, load_yaml(path))


def build_source_bundle(
    out_dir: Path, *, dependency_mode: str, python_version: str, platforms: tuple[str, ...] = pk.DEFAULT_PLATFORMS
) -> dict[str, Any]:
    """sourcedir.tar.gz = train_entry.py + training configs + full package (+ requirements / wheelhouse)."""
    with tempfile.TemporaryDirectory(prefix="train-source-") as tmp:
        stage = Path(tmp) / "source"
        stage.mkdir()
        shutil.copy2(pk.REPO_ROOT / "sagemaker" / "train_entry.py", stage / "train_entry.py")
        (stage / "configs").mkdir()
        for name in TRAINING_CONFIGS:  # never ship deploy / alert configs into the training container
            shutil.copy2(pk.REPO_ROOT / "configs" / name, stage / "configs" / name)
        pk.copy_package(stage / "src", serving_only=False)
        lines = pk.read_requirements(TRAIN_REQUIREMENTS)
        wheels: list[dict[str, Any]] = []
        if dependency_mode == "wheelhouse":
            wheels = pk.vendor_wheels(TRAIN_REQUIREMENTS, stage / "wheelhouse", python_version=python_version,
                                      platforms=platforms)
            pk.write_requirements(stage / "requirements.txt", lines, find_links="wheelhouse")
        else:
            pk.write_requirements(stage / "requirements.txt", lines, find_links=None)
        pk.assert_import_closure(stage / "src")
        digest = pk.write_deterministic_targz(stage, out_dir / "sourcedir.tar.gz")
    return {"path": out_dir / "sourcedir.tar.gz", "sha256": digest, "wheelhouse": wheels}


class TrainingLauncher:
    def __init__(self, cfg: TrainJobConfig, *, sagemaker: Any, s3: Any, sts: Any, clock=None) -> None:
        self.cfg, self.sm, self.s3, self.sts = cfg, sagemaker, s3, sts
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def launch(self, source: dict[str, Any], *, wait: bool = False) -> dict[str, Any]:
        cfg = self.cfg
        cfg.enforce_policy()
        caller = self.sts.get_caller_identity()
        if caller["Account"] != cfglib.account_of(cfg.role_arn, cfglib.ROLE_ARN_RE):
            raise ConfigError("the training role belongs to a different account than the credentials in use")
        image = cfglib.resolve_image_uri(
            explicit=cfg.image_uri, region=cfg.region, framework_version=cfg.framework_version,
            py_version=cfg.py_version, instance_type=cfg.instance_type, scope="training")
        source_uri = self._upload_source(source)
        name = f"{cfg.job_name_prefix}-{self._clock().strftime('%Y%m%d%H%M%S')}"
        self.sm.create_training_job(**self.job_params(name, image, source_uri))
        log_event(logger, "training_job_started", job=name)
        result: dict[str, Any] = {"training_job_name": name, "source_s3_uri": source_uri,
                                  "source_sha256": source["sha256"]}
        if wait:
            self.sm.get_waiter("training_job_completed_or_stopped").wait(
                TrainingJobName=name, WaiterConfig={"Delay": 60, "MaxAttempts": 1440})
            desc = self.sm.describe_training_job(TrainingJobName=name)
            result.update(status=desc.get("TrainingJobStatus"), failure_reason=desc.get("FailureReason"),
                          model_artifact=desc.get("ModelArtifacts", {}).get("S3ModelArtifacts"))
        return result

    def _upload_source(self, source: dict[str, Any]) -> str:
        match = re.fullmatch(r"s3://([^/]+)/?(.*)", self.cfg.inputs.source_s3_prefix)
        bucket, prefix = match.group(1), match.group(2).strip("/")
        key = f"{prefix}/{source['sha256'][:16]}/sourcedir.tar.gz".lstrip("/")
        self.s3.upload_file(
            str(source["path"]), bucket, key,
            ExtraArgs={"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.cfg.kms.output_key_arn,
                       "ExpectedBucketOwner": self.cfg.inputs.expected_bucket_owner,
                       "Metadata": {"sha256": source["sha256"]}})
        return f"s3://{bucket}/{key}"

    @staticmethod
    def _channel(name: str, uri: str) -> dict[str, Any]:
        return {"ChannelName": name, "InputMode": "File", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}}

    def job_params(self, name: str, image: str, source_uri: str) -> dict[str, Any]:
        cfg = self.cfg
        channels = [self._channel("train", cfg.inputs.train_s3_uri), self._channel("code", source_uri)]
        if cfg.inputs.pretrained_s3_uri:
            channels.append(self._channel("pretrained", cfg.inputs.pretrained_s3_uri))
        hyperparameters = {
            "sagemaker_program": "train_entry.py", "sagemaker_submit_directory": CONTAINER_CODE_PATH,
            "sagemaker_region": cfg.region, "sagemaker_container_log_level": "20", **cfg.hyperparameters}
        stopping: dict[str, Any] = {"MaxRuntimeInSeconds": cfg.max_runtime_seconds}
        params: dict[str, Any] = {
            "TrainingJobName": name,
            "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
            "RoleArn": cfg.role_arn,
            "InputDataConfig": channels,
            "OutputDataConfig": {"S3OutputPath": cfg.output.s3_output_path, "KmsKeyId": cfg.kms.output_key_arn},
            "ResourceConfig": {"InstanceType": cfg.instance_type, "InstanceCount": 1,
                               "VolumeSizeInGB": cfg.volume_size_gb, "VolumeKmsKeyId": cfg.kms.volume_key_arn},
            "StoppingCondition": stopping,
            "EnableNetworkIsolation": cfg.security.enable_network_isolation,
            "EnableInterContainerTrafficEncryption": True,
            "HyperParameters": {k: json.dumps(v) for k, v in hyperparameters.items()},  # JSON-encoded like the SDK
            "Environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "Tags": tags_to_aws({"Application": "drg-audit", **cfg.tags}),
        }
        vpc = cfg.security.vpc_config()
        if vpc:
            params["VpcConfig"] = vpc
        if cfg.spot.enabled:
            params["EnableManagedSpotTraining"] = True
            stopping["MaxWaitTimeInSeconds"] = cfg.spot.max_wait_seconds
        if cfg.output.checkpoint_s3_uri:
            params["CheckpointConfig"] = {"S3Uri": cfg.output.checkpoint_s3_uri, "LocalPath": "/opt/ml/checkpoints"}
        return params


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--wait", action="store_true", help="block until the job finishes")
    args = parser.parse_args(argv)
    try:
        cfg = load_train_config(args.config)
        cfg.enforce_policy()  # fail before building or uploading anything
        session = cfglib.make_session(cfg.region, args.profile)
        with tempfile.TemporaryDirectory(prefix="train-bundle-") as tmp:
            source = build_source_bundle(Path(tmp), dependency_mode=cfg.dependency_mode,
                                         python_version=cfg.python_version)
            launcher = TrainingLauncher(cfg, sagemaker=session.client("sagemaker"), s3=session.client("s3"),
                                        sts=session.client("sts"))
            result = launcher.launch(source, wait=args.wait)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
