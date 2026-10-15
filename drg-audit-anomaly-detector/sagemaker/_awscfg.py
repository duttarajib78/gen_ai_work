"""Shared pieces for the SageMaker operations scripts: config fragments, placeholder checks, image URIs."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from drg_audit.errors import ConfigError  # noqa: E402

PLACEHOLDER = "REPLACE_ME"
REGION_RE = re.compile(r"[a-z]{2}(?:-[a-z]+)+-\d")
ROLE_ARN_RE = re.compile(r"arn:aws[a-z-]*:iam::(\d{12}):role/[\w+=,.@/-]+")
KMS_ARN_RE = re.compile(r"arn:aws[a-z-]*:kms:[a-z0-9-]+:(\d{12}):(?:key|alias)/[\w/-]+")
SUBNET_RE = re.compile(r"subnet-[0-9a-f]{8,17}")
SECURITY_GROUP_RE = re.compile(r"sg-[0-9a-f]{8,17}")
S3_URI_RE = re.compile(r"s3://[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9](?:/\S*)?")
ACCOUNT_RE = re.compile(r"\d{12}")


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and PLACEHOLDER in value


def find_placeholders(obj: Any, path: str = "") -> list[str]:
    """Dotted paths of every string that still contains the REPLACE_ME marker."""
    if is_placeholder(obj):
        return [path or "<root>"]
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found += find_placeholders(value, f"{path}.{key}" if path else str(key))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found += find_placeholders(value, f"{path}[{i}]")
    elif hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            found += find_placeholders(getattr(obj, name), f"{path}.{name}" if path else name)
    return found


def require_no_placeholders(cfg: Any) -> None:
    placeholders = find_placeholders(cfg)
    if placeholders:
        raise ConfigError("replace every REPLACE_ME value before running; still set: " + ", ".join(placeholders))


def account_of(arn: str, pattern: re.Pattern[str]) -> str:
    match = pattern.fullmatch(arn)
    if not match:
        raise ConfigError("malformed ARN in configuration")
    return match.group(1)


def check_pattern(value: str, pattern: re.Pattern[str], what: str) -> None:
    if not is_placeholder(value) and not pattern.fullmatch(value):
        raise ConfigError(f"{what} is malformed")


@dataclass(frozen=True)
class VpcSettings:
    subnet_ids: list[str] = field(default_factory=list)
    security_group_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        for subnet in self.subnet_ids:
            check_pattern(subnet, SUBNET_RE, "vpc.subnet_ids entry")
        for group in self.security_group_ids:
            check_pattern(group, SECURITY_GROUP_RE, "vpc.security_group_ids entry")


@dataclass(frozen=True)
class SecuritySettings:
    enable_network_isolation: bool = True
    allow_no_vpc: bool = False
    vpc: VpcSettings = field(default_factory=VpcSettings)

    def enforce_policy(self) -> None:
        """Secure by default: a VPC is mandatory unless explicitly (and knowingly) waived."""
        if self.vpc_config() is None and not self.allow_no_vpc:
            raise ConfigError(
                "security.vpc.subnet_ids and security.vpc.security_group_ids are required; "
                "set security.allow_no_vpc: true only with explicit security approval")

    def vpc_config(self) -> dict[str, list[str]] | None:
        if self.vpc.subnet_ids and self.vpc.security_group_ids:
            return {"Subnets": list(self.vpc.subnet_ids), "SecurityGroupIds": list(self.vpc.security_group_ids)}
        return None


def tags_to_aws(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in sorted(tags.items())]


def resolve_image_uri(
    *, explicit: str | None, region: str, framework_version: str, py_version: str, instance_type: str, scope: str
) -> str:
    """Explicit image URI wins. Otherwise ask the SageMaker SDK (v2 or v3) for the AWS Deep Learning Container."""
    if explicit:
        return explicit
    try:
        from sagemaker import image_uris  # SDK v2
    except ImportError:
        try:
            from sagemaker.core import image_uris  # SDK v3
        except ImportError:
            raise ConfigError(
                "container.image_uri is not set and the 'sagemaker' package is not installed; set image_uri "
                "explicitly (AWS Deep Learning Containers: pytorch-inference / pytorch-training)") from None
    return image_uris.retrieve(
        framework="pytorch", region=region, version=framework_version, py_version=py_version,
        instance_type=instance_type, image_scope=scope)


def make_session(region: str, profile: str | None):
    import boto3

    return boto3.Session(profile_name=profile, region_name=region)


def error_code(exc: BaseException) -> str:
    """Duck-typed botocore ClientError code (keeps this module importable without botocore)."""
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))


def error_message(exc: BaseException) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Message", exc))
