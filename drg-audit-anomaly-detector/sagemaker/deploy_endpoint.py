#!/usr/bin/env python
"""Deploy, update, roll back, inspect or delete the real-time endpoint. Uses boto3 only (no SageMaker SDK).

    python sagemaker/deploy_endpoint.py deploy   --config configs/deploy.dev.yaml --smoke-test examples/request.json
    python sagemaker/deploy_endpoint.py status   --config configs/deploy.dev.yaml
    python sagemaker/deploy_endpoint.py rollback --config configs/deploy.dev.yaml --endpoint-config-name <name>
    python sagemaker/deploy_endpoint.py delete   --config configs/deploy.dev.yaml --yes

Policy enforced BEFORE any AWS call (fail closed; see README "Security"):
  * no REPLACE_ME left in the config; execution role must belong to the calling account
  * VPC attachment required unless ``security.allow_no_vpc`` is explicitly set
  * the package must match package_meta.json (SHA-256); a ``pypi``-mode package cannot use network isolation
  * ``stage: prod`` additionally requires >= 2 instances, >= 2 subnets, network isolation and an alarm topic
Deployments to an existing endpoint are blue/green with canary traffic shifting and automatic rollback on alarms.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _awscfg as cfglib  # noqa: E402  (also puts src/ on sys.path)
import _packaging as pk
from _awscfg import SecuritySettings, error_code, error_message, tags_to_aws

from drg_audit._mapping import dataclass_from_mapping  # noqa: E402
from drg_audit.config import load_yaml  # noqa: E402
from drg_audit.errors import ConfigError, DrgAuditError  # noqa: E402
from drg_audit.logging_utils import configure_logging, get_logger, log_event  # noqa: E402

logger = get_logger("deploy")
ENDPOINT_NAME_RE = re.compile(r"[A-Za-z0-9](?:-?[A-Za-z0-9]){0,35}")  # <= 36 chars: leaves room for suffixes
NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}
STAGES = ("dev", "staging", "prod")


class DeployError(DrgAuditError):
    """A deployment step failed."""


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelLocation:
    package_path: str = "dist/model.tar.gz"
    s3_bucket: str = cfglib.PLACEHOLDER
    s3_prefix: str = "drg-audit/models"
    kms_key_arn: str = cfglib.PLACEHOLDER
    expected_bucket_owner: str = cfglib.PLACEHOLDER

    def validate(self) -> None:
        cfglib.check_pattern(self.kms_key_arn, cfglib.KMS_ARN_RE, "model.kms_key_arn")
        cfglib.check_pattern(self.expected_bucket_owner, cfglib.ACCOUNT_RE, "model.expected_bucket_owner")
        if self.s3_prefix.startswith("/") or ".." in self.s3_prefix.split("/"):
            raise ConfigError("model.s3_prefix must be a relative key prefix")


@dataclass(frozen=True)
class ContainerConfig:
    image_uri: str | None = None
    framework_version: str = "2.6.0"
    py_version: str = "py312"


@dataclass(frozen=True)
class EndpointSettings:
    instance_type: str = "ml.m5.xlarge"
    initial_instance_count: int = 2
    variant_name: str = "AllTraffic"
    container_startup_timeout_seconds: int = 600
    model_data_download_timeout_seconds: int = 600

    def validate(self) -> None:
        if self.initial_instance_count < 1 or not self.instance_type.startswith("ml."):
            raise ConfigError("endpoint.initial_instance_count must be >= 1 and instance_type must start with 'ml.'")
        for value in (self.container_startup_timeout_seconds, self.model_data_download_timeout_seconds):
            if not 60 <= value <= 3600:
                raise ConfigError("endpoint timeouts must be between 60 and 3600 seconds")


@dataclass(frozen=True)
class AutoscalingSettings:
    enabled: bool = True
    min_capacity: int = 2
    max_capacity: int = 6
    target_invocations_per_instance_per_minute: float = 1000.0
    scale_in_cooldown_seconds: int = 300
    scale_out_cooldown_seconds: int = 60

    def validate(self) -> None:
        if not 1 <= self.min_capacity <= self.max_capacity or self.target_invocations_per_instance_per_minute <= 0:
            raise ConfigError("autoscaling: need 1 <= min_capacity <= max_capacity and a positive target")


@dataclass(frozen=True)
class AlarmSettings:
    sns_topic_arn: str | None = None
    error_5xx_threshold: float = 5.0
    model_latency_p95_ms: float = 500.0
    memory_utilization_percent: float = 85.0

    def validate(self) -> None:
        if self.sns_topic_arn and not cfglib.is_placeholder(self.sns_topic_arn) \
                and not self.sns_topic_arn.startswith("arn:aws"):
            raise ConfigError("alarms.sns_topic_arn is malformed")


@dataclass(frozen=True)
class DataCaptureSettings:
    """Captures request/response payloads, which contain clinical data. Off by default."""

    enabled: bool = False
    s3_destination: str | None = None
    sampling_percentage: int = 10

    def validate(self) -> None:
        if not 1 <= self.sampling_percentage <= 100:
            raise ConfigError("data_capture.sampling_percentage must be between 1 and 100")
        if self.enabled and not (self.s3_destination and (cfglib.is_placeholder(self.s3_destination)
                                                          or cfglib.S3_URI_RE.fullmatch(self.s3_destination))):
            raise ConfigError("data_capture.s3_destination must be an s3:// URI when capture is enabled")


@dataclass(frozen=True)
class BlueGreenSettings:
    canary_size_percent: int = 25
    wait_seconds: int = 300
    termination_wait_seconds: int = 120
    max_execution_timeout_seconds: int = 1800

    def validate(self) -> None:
        if not (1 <= self.canary_size_percent <= 50 and 0 <= self.wait_seconds <= 3600
                and 0 <= self.termination_wait_seconds <= 3600 and 600 <= self.max_execution_timeout_seconds <= 14400):
            raise ConfigError("blue_green settings are outside the ranges accepted by SageMaker")


@dataclass(frozen=True)
class DeployConfig:
    region: str = cfglib.PLACEHOLDER
    stage: str = "dev"
    endpoint_name: str = "drg-audit-anomaly-dev"
    execution_role_arn: str = cfglib.PLACEHOLDER
    model: ModelLocation = field(default_factory=ModelLocation)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    endpoint: EndpointSettings = field(default_factory=EndpointSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    container_environment: dict[str, str] = field(default_factory=dict)
    autoscaling: AutoscalingSettings = field(default_factory=AutoscalingSettings)
    alarms: AlarmSettings = field(default_factory=AlarmSettings)
    data_capture: DataCaptureSettings = field(default_factory=DataCaptureSettings)
    blue_green: BlueGreenSettings = field(default_factory=BlueGreenSettings)
    tags: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ConfigError(f"stage must be one of {STAGES}")
        if not ENDPOINT_NAME_RE.fullmatch(self.endpoint_name):
            raise ConfigError("endpoint_name must be 1-36 characters: letters, digits and single hyphens")
        cfglib.check_pattern(self.region, cfglib.REGION_RE, "region")
        cfglib.check_pattern(self.execution_role_arn, cfglib.ROLE_ARN_RE, "execution_role_arn")
        a, e = self.autoscaling, self.endpoint
        if a.enabled and not a.min_capacity <= e.initial_instance_count <= a.max_capacity:
            raise ConfigError("endpoint.initial_instance_count must lie within autoscaling min/max capacity")

    def enforce_policy(self, package_meta: dict[str, Any]) -> None:
        """Every check that must pass before anything is created in AWS."""
        cfglib.require_no_placeholders(self)
        self.security.enforce_policy()
        if package_meta.get("dependency_mode") == "pypi" and self.security.enable_network_isolation:
            raise ConfigError(
                "this package was built with --dependency-mode pypi, which installs packages at container "
                "start-up and cannot run with network isolation; rebuild with the default wheelhouse mode "
                "(recommended) or disable security.enable_network_isolation after security approval")
        if self.stage == "prod":
            problems = []
            if self.endpoint.initial_instance_count < 2 or (self.autoscaling.enabled
                                                            and self.autoscaling.min_capacity < 2):
                problems.append("at least 2 instances are required for availability")
            if len(self.security.vpc.subnet_ids) < 2:
                problems.append("subnets in at least 2 Availability Zones are required")
            if not self.security.enable_network_isolation:
                problems.append("network isolation must stay enabled")
            if not self.alarms.sns_topic_arn:
                problems.append("alarms.sns_topic_arn is required so failures reach a human")
            if problems:
                raise ConfigError("stage 'prod' policy: " + "; ".join(problems))


def load_deploy_config(path: str | Path) -> DeployConfig:
    return dataclass_from_mapping(DeployConfig, load_yaml(path))


# --------------------------------------------------------------------------------------
# package verification
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PackageInfo:
    path: Path
    sha256: str
    meta: dict[str, Any]


def load_package(path: str | Path) -> PackageInfo:
    package = Path(path)
    meta_path = package.parent / "package_meta.json"
    if not package.is_file() or not meta_path.is_file():
        raise DeployError("package or package_meta.json not found; build it with sagemaker/build_model_package.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = pk.sha256_file(package)
    if meta.get("sha256") != digest or meta.get("bytes") != package.stat().st_size:
        raise DeployError("package does not match package_meta.json (modified or stale); rebuild it")
    return PackageInfo(package, digest, meta)


# --------------------------------------------------------------------------------------
# AWS operations
# --------------------------------------------------------------------------------------
@dataclass
class AwsClients:
    sagemaker: Any
    runtime: Any
    s3: Any
    autoscaling: Any
    cloudwatch: Any
    sts: Any

    @classmethod
    def from_session(cls, session: Any) -> AwsClients:
        return cls(
            sagemaker=session.client("sagemaker"), runtime=session.client("sagemaker-runtime"),
            s3=session.client("s3"), autoscaling=session.client("application-autoscaling"),
            cloudwatch=session.client("cloudwatch"), sts=session.client("sts"))


class Deployer:
    def __init__(self, cfg: DeployConfig, clients: AwsClients, *, clock=None) -> None:
        self.cfg = cfg
        self.aws = clients
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ---- commands -----------------------------------------------------------------
    def deploy(self, package: str | Path, *, smoke_test: str | Path | None = None) -> dict[str, Any]:
        info = load_package(package)
        self.cfg.enforce_policy(info.meta)
        caller = self._verify_account()
        image = cfglib.resolve_image_uri(
            explicit=self.cfg.container.image_uri, region=self.cfg.region,
            framework_version=self.cfg.container.framework_version, py_version=self.cfg.container.py_version,
            instance_type=self.cfg.endpoint.instance_type, scope="inference")

        suffix = f"{info.meta['bundle_id'][:8]}-{self._clock().strftime('%Y%m%d%H%M%S')}"
        model_name = f"{self.cfg.endpoint_name}-m-{suffix}"
        config_name = f"{self.cfg.endpoint_name}-c-{suffix}"

        model_url = self._upload(info)
        self._create_model(model_name, image, model_url)
        self._create_endpoint_config(config_name, model_name)
        rollback_alarms = self._put_alarms()
        updating = self._endpoint_exists()
        if updating:
            self._update_endpoint(config_name, rollback_alarms)
        else:
            self._create_endpoint(config_name)
        self._wait_for_service(config_name)
        self._configure_autoscaling()

        result: dict[str, Any] = {
            "endpoint_name": self.cfg.endpoint_name, "mode": "blue_green_update" if updating else "create",
            "model_name": model_name, "endpoint_config_name": config_name, "model_data_url": model_url,
            "bundle_id": info.meta["bundle_id"], "package_sha256": info.sha256, "deployed_by": caller,
        }
        if smoke_test:
            result["smoke_test"] = self.smoke_test(smoke_test)
        log_event(logger, "deployed", endpoint=self.cfg.endpoint_name, bundle_id=info.meta["bundle_id"],
                  mode=result["mode"])
        return result

    def status(self) -> dict[str, Any]:
        desc = self.aws.sagemaker.describe_endpoint(EndpointName=self.cfg.endpoint_name)
        return {
            "endpoint_name": self.cfg.endpoint_name, "status": desc.get("EndpointStatus"),
            "endpoint_config_name": desc.get("EndpointConfigName"), "failure_reason": desc.get("FailureReason"),
            "variants": [{"name": v.get("VariantName"), "current_instances": v.get("CurrentInstanceCount"),
                          "desired_instances": v.get("DesiredInstanceCount")}
                         for v in desc.get("ProductionVariants", [])],
        }

    def rollback(self, endpoint_config_name: str) -> dict[str, Any]:
        self.aws.sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)  # must exist
        self._update_endpoint(endpoint_config_name, self._alarm_names())
        self._wait_for_service(endpoint_config_name)
        return {"endpoint_name": self.cfg.endpoint_name, "endpoint_config_name": endpoint_config_name}

    def delete(self, *, delete_models: bool = False) -> dict[str, Any]:
        ep, variant = self.cfg.endpoint_name, self.cfg.endpoint.variant_name
        resource_id = f"endpoint/{ep}/variant/{variant}"
        dimension = "sagemaker:variant:DesiredInstanceCount"
        for call, kwargs in (
            (self.aws.autoscaling.delete_scaling_policy, {"PolicyName": f"{ep}-invocations-target"}),
            (self.aws.autoscaling.deregister_scalable_target, {}),
        ):
            try:
                call(ServiceNamespace="sagemaker", ResourceId=resource_id, ScalableDimension=dimension, **kwargs)
            except Exception as exc:
                if error_code(exc) != "ObjectNotFoundException":
                    raise
        self.aws.cloudwatch.delete_alarms(AlarmNames=[f"{ep}-{s}" for s in ("5xx", "latency-p95", "memory")])
        desc = self.aws.sagemaker.describe_endpoint(EndpointName=ep)
        config_name = desc.get("EndpointConfigName")
        model_names = []
        if delete_models and config_name:
            cfg_desc = self.aws.sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
            model_names = [v["ModelName"] for v in cfg_desc.get("ProductionVariants", [])]
        self.aws.sagemaker.delete_endpoint(EndpointName=ep)
        if delete_models and config_name:
            self.aws.sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            for name in model_names:
                self.aws.sagemaker.delete_model(ModelName=name)
        return {"deleted_endpoint": ep, "deleted_models": model_names if delete_models else []}

    def smoke_test(self, payload_path: str | Path) -> dict[str, Any]:
        """Invoke the live endpoint once. Only aggregate counts are returned (never record data)."""
        response = self.aws.runtime.invoke_endpoint(
            EndpointName=self.cfg.endpoint_name, ContentType="application/json", Accept="application/json",
            Body=Path(payload_path).read_bytes())
        data = json.loads(response["Body"].read())
        if "results" not in data or "summary" not in data:
            raise DeployError("smoke test: the endpoint returned a malformed response")
        return {"model_version": data.get("model_version"), "summary": data["summary"]}

    # ---- steps --------------------------------------------------------------------
    def _verify_account(self) -> str:
        caller = self.aws.sts.get_caller_identity()
        role_account = cfglib.account_of(self.cfg.execution_role_arn, cfglib.ROLE_ARN_RE)
        if caller["Account"] != role_account:
            raise ConfigError("the execution role belongs to a different account than the credentials in use")
        return str(caller.get("Arn", caller["Account"]))

    def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return self.aws.s3.head_object(Bucket=self.cfg.model.s3_bucket, Key=key,
                                           ExpectedBucketOwner=self.cfg.model.expected_bucket_owner)
        except Exception as exc:
            if error_code(exc) in NOT_FOUND_CODES:
                return None
            raise

    def _upload(self, info: PackageInfo) -> str:
        m = self.cfg.model
        key = f"{m.s3_prefix.strip('/')}/{info.meta['bundle_id']}/{info.sha256[:16]}/model.tar.gz"
        head = self._head(key)
        if head is None:
            self.aws.s3.upload_file(
                str(info.path), m.s3_bucket, key,
                ExtraArgs={"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": m.kms_key_arn,
                           "ExpectedBucketOwner": m.expected_bucket_owner, "Metadata": {"sha256": info.sha256}})
            head = self._head(key)
            if head is None:
                raise DeployError("model package upload could not be verified")
        if head.get("Metadata", {}).get("sha256") != info.sha256:
            raise DeployError("an object with a different digest already exists at the content-addressed key")
        return f"s3://{m.s3_bucket}/{key}"

    def _container_env(self) -> dict[str, str]:
        env = {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        env.update(self.cfg.container_environment)
        return env

    def _tags(self) -> list[dict[str, str]]:
        return tags_to_aws({"Application": "drg-audit", "Stage": self.cfg.stage, **self.cfg.tags})

    def _create_model(self, name: str, image: str, model_url: str) -> None:
        params: dict[str, Any] = {
            "ModelName": name, "ExecutionRoleArn": self.cfg.execution_role_arn,
            "PrimaryContainer": {"Image": image, "ModelDataUrl": model_url, "Environment": self._container_env()},
            "EnableNetworkIsolation": self.cfg.security.enable_network_isolation, "Tags": self._tags(),
        }
        vpc = self.cfg.security.vpc_config()
        if vpc:
            params["VpcConfig"] = vpc
        self.aws.sagemaker.create_model(**params)

    def _create_endpoint_config(self, name: str, model_name: str) -> None:
        e, capture = self.cfg.endpoint, self.cfg.data_capture
        params: dict[str, Any] = {
            "EndpointConfigName": name,
            "ProductionVariants": [{
                "VariantName": e.variant_name, "ModelName": model_name, "InstanceType": e.instance_type,
                "InitialInstanceCount": e.initial_instance_count, "InitialVariantWeight": 1.0,
                "ContainerStartupHealthCheckTimeoutInSeconds": e.container_startup_timeout_seconds,
                "ModelDataDownloadTimeoutInSeconds": e.model_data_download_timeout_seconds}],
            "KmsKeyId": self.cfg.model.kms_key_arn,
            "Tags": self._tags(),
        }
        if capture.enabled:
            params["DataCaptureConfig"] = {
                "EnableCapture": True, "InitialSamplingPercentage": capture.sampling_percentage,
                "DestinationS3Uri": capture.s3_destination, "KmsKeyId": self.cfg.model.kms_key_arn,
                "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]}}
        self.aws.sagemaker.create_endpoint_config(**params)

    def _alarm_names(self) -> list[str]:
        return [f"{self.cfg.endpoint_name}-5xx", f"{self.cfg.endpoint_name}-latency-p95"]

    def _put_alarms(self) -> list[str]:
        ep, a = self.cfg.endpoint_name, self.cfg.alarms
        dimensions = [{"Name": "EndpointName", "Value": ep},
                      {"Name": "VariantName", "Value": self.cfg.endpoint.variant_name}]
        actions = [a.sns_topic_arn] if a.sns_topic_arn else []
        specs = {
            "5xx": {"Namespace": "AWS/SageMaker", "MetricName": "Invocation5XXErrors", "Statistic": "Sum",
                    "Threshold": a.error_5xx_threshold},
            "latency-p95": {"Namespace": "AWS/SageMaker", "MetricName": "ModelLatency", "ExtendedStatistic": "p95",
                            "Threshold": a.model_latency_p95_ms * 1000.0},  # metric is in microseconds
            "memory": {"Namespace": "/aws/sagemaker/Endpoints", "MetricName": "MemoryUtilization",
                       "Statistic": "Average", "Threshold": a.memory_utilization_percent},
        }
        for suffix, spec in specs.items():
            self.aws.cloudwatch.put_metric_alarm(
                AlarmName=f"{ep}-{suffix}", AlarmDescription=f"drg-audit {suffix} alarm for {ep}",
                ActionsEnabled=bool(actions), AlarmActions=actions, Dimensions=dimensions, Period=60,
                EvaluationPeriods=5, DatapointsToAlarm=3, ComparisonOperator="GreaterThanOrEqualToThreshold",
                TreatMissingData="notBreaching", Tags=self._tags(), **spec)
        return self._alarm_names()

    def _deployment_config(self, alarms: list[str]) -> dict[str, Any]:
        bg = self.cfg.blue_green
        return {
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": {
                    "Type": "CANARY", "WaitIntervalInSeconds": bg.wait_seconds,
                    "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": bg.canary_size_percent}},
                "TerminationWaitInSeconds": bg.termination_wait_seconds,
                "MaximumExecutionTimeoutInSeconds": bg.max_execution_timeout_seconds},
            "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": name} for name in alarms]},
        }

    def _endpoint_exists(self) -> bool:
        try:
            self.aws.sagemaker.describe_endpoint(EndpointName=self.cfg.endpoint_name)
            return True
        except Exception as exc:
            if error_code(exc) == "ValidationException" and "Could not find" in error_message(exc):
                return False
            raise

    def _create_endpoint(self, config_name: str) -> None:
        self.aws.sagemaker.create_endpoint(EndpointName=self.cfg.endpoint_name, EndpointConfigName=config_name,
                                           Tags=self._tags())

    def _update_endpoint(self, config_name: str, alarms: list[str]) -> None:
        self.aws.sagemaker.update_endpoint(
            EndpointName=self.cfg.endpoint_name, EndpointConfigName=config_name,
            DeploymentConfig=self._deployment_config(alarms))

    def _wait_for_service(self, expected_config: str) -> None:
        ep = self.cfg.endpoint_name
        try:
            self.aws.sagemaker.get_waiter("endpoint_in_service").wait(
                EndpointName=ep, WaiterConfig={"Delay": 30, "MaxAttempts": 120})
        except Exception as exc:
            reason = self.aws.sagemaker.describe_endpoint(EndpointName=ep).get("FailureReason", "unknown")
            raise DeployError(f"endpoint did not reach InService: {reason}") from exc
        if self.aws.sagemaker.describe_endpoint(EndpointName=ep).get("EndpointConfigName") != expected_config:
            raise DeployError(
                "the endpoint is serving a different configuration than the one just deployed; an alarm-driven "
                "blue/green rollback probably occurred - inspect the CloudWatch alarms and endpoint logs")

    def _configure_autoscaling(self) -> None:
        a = self.cfg.autoscaling
        if not a.enabled:
            return
        ep = self.cfg.endpoint_name
        resource_id = f"endpoint/{ep}/variant/{self.cfg.endpoint.variant_name}"
        dimension = "sagemaker:variant:DesiredInstanceCount"
        self.aws.autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker", ResourceId=resource_id, ScalableDimension=dimension,
            MinCapacity=a.min_capacity, MaxCapacity=a.max_capacity)
        self.aws.autoscaling.put_scaling_policy(
            PolicyName=f"{ep}-invocations-target", ServiceNamespace="sagemaker", ResourceId=resource_id,
            ScalableDimension=dimension, PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": a.target_invocations_per_instance_per_minute,
                "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"},
                "ScaleInCooldown": a.scale_in_cooldown_seconds, "ScaleOutCooldown": a.scale_out_cooldown_seconds})


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("deploy", "status", "rollback", "delete"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--profile", help="AWS named profile (default: standard credential chain)")
    sub.choices["deploy"].add_argument("--package", help="default: model.package_path from the config")
    sub.choices["deploy"].add_argument("--smoke-test", help="JSON request file to invoke after deployment")
    sub.choices["rollback"].add_argument("--endpoint-config-name", required=True)
    sub.choices["delete"].add_argument("--yes", action="store_true", help="confirm deletion")
    sub.choices["delete"].add_argument("--delete-models", action="store_true",
                                       help="also delete the endpoint config and model")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_deploy_config(args.config)
        if cfglib.is_placeholder(cfg.region):
            raise ConfigError("set `region` in the deployment config")
        deployer = Deployer(cfg, AwsClients.from_session(cfglib.make_session(cfg.region, args.profile)))
        if args.command == "deploy":
            result = deployer.deploy(args.package or cfg.model.package_path, smoke_test=args.smoke_test)
        elif args.command == "status":
            result = deployer.status()
        elif args.command == "rollback":
            result = deployer.rollback(args.endpoint_config_name)
        else:
            if not args.yes:
                print(f"refusing to delete endpoint '{cfg.endpoint_name}' without --yes", file=sys.stderr)
                return 1
            result = deployer.delete(delete_models=args.delete_models)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
