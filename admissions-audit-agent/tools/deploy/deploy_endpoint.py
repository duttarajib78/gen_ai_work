"""Create or update the SageMaker real-time endpoint, with autoscaling and monitoring.

    python -m tools.deploy.deploy_endpoint --config tools/deploy/config/prod.yaml --dry-run
    python -m tools.deploy.deploy_endpoint --config tools/deploy/config/prod.yaml --smoke-test

What it does, in order (``--dry-run`` prints exactly these calls and changes nothing):

 1. validates the configuration, including the container environment with the same
    code the service runs at start-up (a setup that would crash on the endpoint is
    rejected here, for example a ``global.`` profile without the residency opt-in);
 2. creates a new, uniquely named SageMaker model and endpoint configuration
    (models and configurations are immutable, so every release gets its own);
 3. creates CloudWatch alarms (they also act as the automatic-rollback triggers);
 4. creates the endpoint, or updates it with a blue/green canary and automatic rollback;
 5. waits for ``InService``;
 6. registers target-tracking autoscaling on invocations per instance;
 7. creates log metric filters for business metrics (episodes, alerts, failures, tokens).

Secure defaults enforced here: VPC required, storage encryption key required, at least
two instances, no ``latest`` image tag, no secrets in the container environment, network
isolation off (the container must reach Bedrock, so isolation cannot be enabled) and
data capture off (it would write request payloads to S3).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from audit_agent.config import Settings
from audit_agent.errors import ConfigError

_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
_ROLE_RE = re.compile(r"^arn:aws[a-z\-]*:iam::\d{12}:role/[\w+=,.@\-/]+$")
_IMAGE_RE = re.compile(
    r"^\d{12}\.dkr\.ecr\.[a-z0-9\-]+\.amazonaws\.com(?:\.cn)?/[a-z0-9._/\-]+(?::[\w.\-]{1,128}|@sha256:[0-9a-f]{64})$"
)
_PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]*>")
_SECRET_KEY_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|PRIVATE)", re.I)
VARIANT = "AllTraffic"


@dataclass(frozen=True)
class Step:
    service: str      # sagemaker | cloudwatch | application-autoscaling | logs | wait
    operation: str
    params: dict
    note: str = ""
    optional: bool = False  # failure is reported but does not abort the deployment


@dataclass
class Problems(Exception):
    items: list = field(default_factory=list)

    def __str__(self) -> str:
        return "invalid deployment configuration:\n  - " + "\n  - ".join(self.items)


# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        return json.loads(text)
    import yaml  # PyYAML is a deployment-tool dependency only

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise Problems(["the configuration file must contain a mapping"])
    return data


def find_placeholders(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in find_placeholders(v, f"{path}.{k}" if path else str(k))]
    if isinstance(node, (list, tuple)):
        return [p for i, v in enumerate(node) for p in find_placeholders(v, f"{path}[{i}]")]
    if isinstance(node, str) and _PLACEHOLDER_RE.search(node):
        return [path]
    return []


def container_environment(cfg: dict) -> dict[str, str]:
    env = {"AUDIT_AWS_REGION": cfg["region"]}
    for key, value in (cfg.get("environment") or {}).items():
        env[str(key)] = ("true" if value else "false") if isinstance(value, bool) else str(value)
    return env


def validate_config(
    cfg: dict, *, allow_no_vpc: bool = False, allow_no_kms: bool = False, allow_single_instance: bool = False
) -> list[str]:
    problems: list[str] = []
    for key in ("endpoint_name", "region", "image_uri", "model_data_url", "execution_role_arn", "instance", "environment"):
        if key not in cfg:
            problems.append(f"missing required key: {key}")
    if problems:
        return problems

    problems += [f"placeholder not replaced at {p}" for p in find_placeholders(cfg)]

    if not _NAME_RE.match(str(cfg["endpoint_name"])) or len(str(cfg["endpoint_name"])) > 40:
        problems.append("endpoint_name must be 1-40 characters: letters, digits, hyphens")
    if not _IMAGE_RE.match(str(cfg["image_uri"])):
        problems.append("image_uri must be an ECR image URI with a tag or digest")
    elif str(cfg["image_uri"]).endswith(":latest"):
        problems.append("image_uri must not use the mutable 'latest' tag")
    if not (str(cfg["model_data_url"]).startswith("s3://") and str(cfg["model_data_url"]).endswith(".tar.gz")):
        problems.append("model_data_url must be an s3:// URI of a .tar.gz artifact")
    if not _ROLE_RE.match(str(cfg["execution_role_arn"])):
        problems.append("execution_role_arn is not a valid IAM role ARN")

    vpc = cfg.get("vpc") or {}
    if not allow_no_vpc:
        if len(vpc.get("subnets") or []) < 2 or not vpc.get("security_group_ids"):
            problems.append("vpc needs at least two subnets (different Availability Zones) and one security group")
    if not cfg.get("kms_key_id") and not allow_no_kms:
        problems.append("kms_key_id is required to encrypt the endpoint's storage volume")

    inst = cfg["instance"] or {}
    count = inst.get("count", 0)
    if not inst.get("type", "").startswith("ml."):
        problems.append("instance.type must be a SageMaker instance type such as ml.m5.large")
    if not isinstance(count, int) or count < 1 or (count < 2 and not allow_single_instance):
        problems.append("instance.count must be at least 2 (use --allow-single-instance for non-production)")

    auto = cfg.get("autoscaling")
    if auto:
        lo, hi = auto.get("min_capacity", 0), auto.get("max_capacity", 0)
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= count <= hi):
            problems.append("autoscaling requires 1 <= min_capacity <= instance.count <= max_capacity")
        if not auto.get("target_invocations_per_instance", 0) > 0:
            problems.append("autoscaling.target_invocations_per_instance must be positive")

    env = container_environment(cfg)
    for key in env:
        if not key.startswith("AUDIT_"):
            problems.append(f"environment key {key!r} is not an AUDIT_* setting")
        if _SECRET_KEY_RE.search(key):
            problems.append(f"environment key {key!r} looks like a secret; secrets must not be passed as environment variables")
    try:  # the exact validation the container performs at start-up
        Settings.from_mapping({**env, "AUDIT_MODEL_DIR": "/opt/ml/model"})
    except ConfigError as exc:
        problems.append("environment would fail service start-up: " + str(exc).replace("\n", " "))
    return problems


# ---------------------------------------------------------------------------
def _tags(cfg: dict) -> list[dict]:
    return [{"Key": str(k), "Value": str(v)} for k, v in sorted((cfg.get("tags") or {}).items())]


def names(cfg: dict, release: str) -> dict:
    base = f"{cfg['endpoint_name']}-{release}"
    if not _NAME_RE.match(base):
        raise Problems([f"release {release!r} produces an invalid resource name"])
    return {"model": base, "config": base, "endpoint": cfg["endpoint_name"]}


def _alarm(cfg: dict, suffix: str, **kw) -> dict:
    mon = cfg.get("monitoring") or {}
    params = {"AlarmName": f"{cfg['endpoint_name']}-{suffix}", "TreatMissingData": "notBreaching",
              "ComparisonOperator": "GreaterThanOrEqualToThreshold", "ActionsEnabled": bool(mon.get("sns_topic_arn")), **kw}
    if mon.get("sns_topic_arn"):
        params["AlarmActions"] = [mon["sns_topic_arn"]]
    return params


def build_plan(cfg: dict, *, release: str, endpoint_exists: bool) -> list[Step]:
    n = names(cfg, release)
    ep = cfg["endpoint_name"]
    mon = cfg.get("monitoring") or {}
    vpc = cfg.get("vpc") or {}
    inst = cfg["instance"]
    namespace = f"AdmissionsAuditAgent/{ep}"
    dims = [{"Name": "EndpointName", "Value": ep}, {"Name": "VariantName", "Value": VARIANT}]
    plan: list[Step] = []

    model: dict[str, Any] = {
        "ModelName": n["model"], "ExecutionRoleArn": cfg["execution_role_arn"],
        "PrimaryContainer": {"Image": cfg["image_uri"], "Mode": "SingleModel", "ModelDataUrl": cfg["model_data_url"],
                             "Environment": container_environment(cfg)},
        "EnableNetworkIsolation": False,  # the container must reach Bedrock
        "Tags": _tags(cfg),
    }
    if vpc:
        model["VpcConfig"] = {"Subnets": list(vpc["subnets"]), "SecurityGroupIds": list(vpc["security_group_ids"])}
    plan.append(Step("sagemaker", "create_model", model, "immutable model for this release"))

    variant: dict[str, Any] = {
        "VariantName": VARIANT, "ModelName": n["model"], "InstanceType": inst["type"],
        "InitialInstanceCount": inst["count"], "InitialVariantWeight": 1.0,
        "ContainerStartupHealthCheckTimeoutInSeconds": int(cfg.get("startup_health_check_timeout_s", 600)),
        "ModelDataDownloadTimeoutInSeconds": int(cfg.get("model_data_download_timeout_s", 600)),
        "RoutingConfig": {"RoutingStrategy": cfg.get("routing_strategy", "LEAST_OUTSTANDING_REQUESTS")},
    }
    config: dict[str, Any] = {"EndpointConfigName": n["config"], "ProductionVariants": [variant], "Tags": _tags(cfg)}
    if cfg.get("kms_key_id"):
        config["KmsKeyId"] = cfg["kms_key_id"]  # no DataCaptureConfig: it would persist request payloads
    plan.append(Step("sagemaker", "create_endpoint_config", config))

    alarms = [
        _alarm(cfg, "5xx", Namespace="AWS/SageMaker", MetricName="Invocation5XXErrors", Dimensions=dims,
               Statistic="Sum", Period=60, EvaluationPeriods=5, DatapointsToAlarm=3,
               Threshold=float(mon.get("alarm_5xx_threshold", 5)),
               AlarmDescription="Endpoint is returning 5XX responses (also triggers automatic rollback)"),
        _alarm(cfg, "latency-p99", Namespace="AWS/SageMaker", MetricName="ModelLatency", Dimensions=dims,
               ExtendedStatistic="p99", Period=300, EvaluationPeriods=3, DatapointsToAlarm=3,
               Threshold=float(mon.get("p99_model_latency_ms", 55000)) * 1000.0,  # metric unit is microseconds
               AlarmDescription="p99 model latency is near the invocation limit"),
        _alarm(cfg, "deadline-exceeded", Namespace=namespace, MetricName="DeadlineExceeded", Statistic="Sum", Period=300,
               EvaluationPeriods=1, Threshold=float(mon.get("deadline_exceeded_threshold", 5)),
               AlarmDescription="Audits are running out of time budget; use streaming or lower the reasoning effort"),
        _alarm(cfg, "llm-throttled", Namespace=namespace, MetricName="LlmThrottled", Statistic="Sum", Period=300,
               EvaluationPeriods=1, Threshold=float(mon.get("llm_throttled_threshold", 20)),
               AlarmDescription="Amazon Bedrock is throttling; check quotas"),
    ]
    for alarm in alarms:
        plan.append(Step("cloudwatch", "put_metric_alarm", alarm, "alarm"))

    if endpoint_exists:
        bg = cfg.get("blue_green") or {}
        plan.append(Step("sagemaker", "update_endpoint", {
            "EndpointName": ep, "EndpointConfigName": n["config"],
            "DeploymentConfig": {
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "CANARY",
                        "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": int(bg.get("canary_percent", 10))},
                        "WaitIntervalInSeconds": int(bg.get("canary_wait_s", 300)),
                    },
                    "TerminationWaitInSeconds": int(bg.get("termination_wait_s", 600)),
                    "MaximumExecutionTimeoutInSeconds": int(bg.get("max_execution_s", 1800)),
                },
                "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": alarms[0]["AlarmName"]}, {"AlarmName": alarms[1]["AlarmName"]}]},
            },
        }, "blue/green canary with automatic rollback"))
    else:
        plan.append(Step("sagemaker", "create_endpoint",
                         {"EndpointName": ep, "EndpointConfigName": n["config"], "Tags": _tags(cfg)}, "first deployment"))
    plan.append(Step("wait", "endpoint_in_service", {"EndpointName": ep, "WaiterConfig": {"Delay": 30, "MaxAttempts": 120}},
                     "wait for InService"))

    auto = cfg.get("autoscaling")
    if auto:
        resource = f"endpoint/{ep}/variant/{VARIANT}"
        plan.append(Step("application-autoscaling", "register_scalable_target", {
            "ServiceNamespace": "sagemaker", "ResourceId": resource, "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
            "MinCapacity": auto["min_capacity"], "MaxCapacity": auto["max_capacity"]}))
        plan.append(Step("application-autoscaling", "put_scaling_policy", {
            "PolicyName": f"{ep}-invocations-per-instance", "ServiceNamespace": "sagemaker", "ResourceId": resource,
            "ScalableDimension": "sagemaker:variant:DesiredInstanceCount", "PolicyType": "TargetTrackingScaling",
            "TargetTrackingScalingPolicyConfiguration": {
                "TargetValue": float(auto["target_invocations_per_instance"]),
                "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"},
                "ScaleInCooldown": int(auto.get("scale_in_cooldown_s", 600)),
                "ScaleOutCooldown": int(auto.get("scale_out_cooldown_s", 120))}}))

    log_group = f"/aws/sagemaker/Endpoints/{ep}"
    filters = [
        ("episodes-completed", '{ $.event = "episode_completed" }', "EpisodesCompleted", "1"),
        ("episodes-failed", '{ $.event = "episode_failed" }', "EpisodesFailed", "1"),
        ("alerts-flagged", '{ ($.event = "episode_completed") && ($.flagged IS TRUE) }', "AlertsFlagged", "1"),
        ("deadline-exceeded", '{ $.error_code = "deadline_exceeded" }', "DeadlineExceeded", "1"),
        ("llm-throttled", '{ ($.event = "llm_call_failed") && ($.error_code = "llm_throttled") }', "LlmThrottled", "1"),
        ("llm-input-tokens", '{ $.event = "llm_call" }', "LlmInputTokens", "$.input_tokens"),
        ("llm-output-tokens", '{ $.event = "llm_call" }', "LlmOutputTokens", "$.output_tokens"),
    ]
    for suffix, pattern, metric, value in filters:
        plan.append(Step("logs", "put_metric_filter", {
            "logGroupName": log_group, "filterName": f"{ep}-{suffix}", "filterPattern": pattern,
            "metricTransformations": [{"metricName": metric, "metricNamespace": namespace, "metricValue": value}]},
            "business metric", optional=True))
    return plan


# ---------------------------------------------------------------------------
def endpoint_exists(sm: Any, name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except Exception as exc:  # noqa: BLE001
        if "ValidationException" in str(exc) or "Could not find" in str(exc):
            return False
        raise


def apply_plan(plan: Sequence[Step], session: Any, log=print) -> None:
    clients: dict[str, Any] = {}

    def client(service: str) -> Any:
        clients.setdefault(service, session.client(service))
        return clients[service]

    for i, step in enumerate(plan, start=1):
        label = f"[{i}/{len(plan)}] {step.service}.{step.operation}" + (f" ({step.note})" if step.note else "")
        log(label)
        try:
            if step.service == "wait":
                client("sagemaker").get_waiter(step.operation).wait(**step.params)
            else:
                getattr(client(step.service), step.operation)(**step.params)
        except Exception as exc:  # noqa: BLE001
            if step.optional:
                log(f"    warning: {type(exc).__name__} (continuing): {str(exc)[:200]}")
                continue
            if step.operation == "endpoint_in_service":
                reason = client("sagemaker").describe_endpoint(EndpointName=step.params["EndpointName"]).get("FailureReason")
                raise RuntimeError(f"endpoint did not reach InService: {reason}") from exc
            raise


def smoke_test(session: Any, cfg: dict, log=print) -> bool:
    from tools.client.sagemaker_client import invoke

    payload = json.loads((Path(__file__).parent / "smoke_payload.json").read_text(encoding="utf-8"))
    runtime = session.client("sagemaker-runtime")
    result = invoke(runtime, cfg["endpoint_name"], payload)
    ok = result.ok and len(result.body.get("verdicts", [])) == 2 and "provenance" in result.body
    log(f"smoke test: http {result.http_status}, status {result.body.get('status')}, "
        f"{'PASSED' if ok else 'FAILED: ' + json.dumps(result.body.get('error'))}")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--release", default=time.strftime("%Y%m%d%H%M%S", time.gmtime()),
                   help="suffix for the new model and endpoint-config names (default: UTC timestamp)")
    p.add_argument("--dry-run", action="store_true", help="print the planned calls and exit")
    p.add_argument("--assume-existing", action="store_true", help="with --dry-run: plan an update instead of a create")
    p.add_argument("--smoke-test", action="store_true", help="invoke the endpoint once after it is InService")
    p.add_argument("--allow-no-vpc", action="store_true", help="NOT recommended: permit an endpoint outside a VPC")
    p.add_argument("--allow-no-kms", action="store_true", help="NOT recommended: permit an unencrypted storage volume")
    p.add_argument("--allow-single-instance", action="store_true", help="non-production only")
    a = p.parse_args(argv)

    try:
        cfg = load_config(a.config)
        problems = validate_config(cfg, allow_no_vpc=a.allow_no_vpc, allow_no_kms=a.allow_no_kms,
                                   allow_single_instance=a.allow_single_instance)
        if problems:
            raise Problems(problems)
        if a.dry_run:
            plan = build_plan(cfg, release=a.release, endpoint_exists=a.assume_existing)
            print(json.dumps([{"service": s.service, "operation": s.operation, "params": s.params, "note": s.note}
                              for s in plan], indent=2, default=str))
            return 0

        import boto3

        session = boto3.Session(region_name=cfg["region"])
        exists = endpoint_exists(session.client("sagemaker"), cfg["endpoint_name"])
        plan = build_plan(cfg, release=a.release, endpoint_exists=exists)
        print(f"{'Updating' if exists else 'Creating'} endpoint {cfg['endpoint_name']} (release {a.release})")
        apply_plan(plan, session)
        if a.smoke_test and not smoke_test(session, cfg):
            return 3
        print("done")
        return 0
    except Problems as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
