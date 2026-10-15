"""Least-privilege IAM policy generators.

Generating policies from parameters (instead of editing JSON by hand) keeps the
resource scoping exact. Nothing here calls AWS.

    python -m tools.deploy.iam bedrock --region ap-southeast-2 --account 123456789012 \
        --model-id global.anthropic.claude-sonnet-5-5
    python -m tools.deploy.iam execution-role --region ... --account ... --endpoint-name ... \
        --ecr-repository admissions-audit-agent --artifact-s3-uri s3://bucket/key/model.tar.gz \
        --kms-key-arn arn:aws:kms:... --model-id global.anthropic.claude-sonnet-5-5
    python -m tools.deploy.iam trust
    python -m tools.deploy.iam deployer ...

Bedrock access follows the AWS documentation for inference profiles:

* a ``global.`` profile needs three statements (the profile, the foundation model
  in the calling Region, and the Region-less global model ARN, whose
  ``aws:RequestedRegion`` is ``unspecified``), all pinned to the one profile ARN;
* a geography profile (``us.`` / ``eu.`` / ``au.`` ...) needs the profile plus the
  foundation model in *each destination Region*. Get the destinations from
  ``aws bedrock get-inference-profile --region <source> --inference-profile-identifier <id>``
  and pass them with ``--destination-region``. Do not rely on the prefix alone.

Always validate a generated policy with a real test invocation (and, if you use
SCPs, confirm they allow the destination Regions).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Optional, Sequence
from urllib.parse import urlparse

_ACCOUNT_RE = re.compile(r"^\d{12}$")
_PROFILE_RE = re.compile(r"^(?P<prefix>[a-z]{2,10})\.(?P<base>anthropic\.claude-[a-z0-9.\-]+(?::\d+)?)$")
_BASE_RE = re.compile(r"^anthropic\.claude-[a-z0-9.\-]+(?::\d+)?$")


def _split_model_id(model_id: str) -> tuple[Optional[str], str]:
    m = _PROFILE_RE.match(model_id)
    if m:
        return m.group("prefix"), m.group("base")
    if _BASE_RE.match(model_id):
        return None, model_id
    raise ValueError(f"unsupported model id: {model_id!r}")


def bedrock_invoke_statements(
    *, region: str, account_id: str, model_id: str, destination_regions: Optional[Sequence[str]] = None
) -> list[dict]:
    if not _ACCOUNT_RE.match(account_id):
        raise ValueError("account id must be 12 digits")
    prefix, base = _split_model_id(model_id)
    if prefix is None:
        raise ValueError(
            "this model is invoked through an inference profile (global./us./eu./au. ...); pass the profile id"
        )
    profile_arn = f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{model_id}"
    action = "bedrock:InvokeModel"  # also authorises the Converse API

    if prefix == "global":
        return [
            {"Sid": "InvokeGlobalProfile", "Effect": "Allow", "Action": action, "Resource": [profile_arn],
             "Condition": {"StringEquals": {"aws:RequestedRegion": region}}},
            {"Sid": "InvokeModelInCallingRegionViaProfile", "Effect": "Allow", "Action": action,
             "Resource": [f"arn:aws:bedrock:{region}::foundation-model/{base}"],
             "Condition": {"StringEquals": {"aws:RequestedRegion": region, "bedrock:InferenceProfileArn": profile_arn}}},
            {"Sid": "InvokeGlobalModelViaProfile", "Effect": "Allow", "Action": action,
             "Resource": [f"arn:aws:bedrock:::foundation-model/{base}"],
             "Condition": {"StringEquals": {"aws:RequestedRegion": "unspecified", "bedrock:InferenceProfileArn": profile_arn}}},
        ]

    if not destination_regions:
        raise ValueError(
            "a geography profile needs its destination Regions: run `aws bedrock get-inference-profile` "
            "from the source Region and pass each one with --destination-region"
        )
    return [
        {"Sid": "InvokeGeoProfile", "Effect": "Allow", "Action": action, "Resource": [profile_arn]},
        {"Sid": "InvokeModelInDestinationRegionsViaProfile", "Effect": "Allow", "Action": action,
         "Resource": [f"arn:aws:bedrock:{r}::foundation-model/{base}" for r in sorted(set(destination_regions))],
         "Condition": {"StringEquals": {"bedrock:InferenceProfileArn": profile_arn}}},
    ]


def bedrock_invoke_policy(**kwargs) -> dict:
    return {"Version": "2012-10-17", "Statement": bedrock_invoke_statements(**kwargs)}


def trust_policy() -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("expected an s3://bucket/key URI")
    return parsed.netloc, parsed.path.lstrip("/")


def execution_role_policy(
    *, region: str, account_id: str, endpoint_name: str, ecr_repository: str, artifact_s3_uri: str,
    kms_key_arn: str, model_id: str, destination_regions: Optional[Sequence[str]] = None, vpc: bool = True,
) -> dict:
    bucket, key = _s3_parts(artifact_s3_uri)
    prefix = key.rsplit("/", 1)[0] if "/" in key else ""
    log_group = f"arn:aws:logs:{region}:{account_id}:log-group:/aws/sagemaker/Endpoints/{endpoint_name}"
    statements = bedrock_invoke_statements(
        region=region, account_id=account_id, model_id=model_id, destination_regions=destination_regions
    ) + [
        {"Sid": "PullImage", "Effect": "Allow",
         "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
         "Resource": [f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repository}"]},
        {"Sid": "EcrToken", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Sid": "ReadModelArtifact", "Effect": "Allow", "Action": "s3:GetObject",
         "Resource": [f"arn:aws:s3:::{bucket}/{key}"]},
        {"Sid": "ListArtifactPrefix", "Effect": "Allow", "Action": "s3:ListBucket",
         "Resource": [f"arn:aws:s3:::{bucket}"], "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*" if prefix else "*"]}}},
        {"Sid": "DecryptArtifact", "Effect": "Allow", "Action": ["kms:Decrypt", "kms:DescribeKey"], "Resource": [kms_key_arn]},
        {"Sid": "EndpointLogs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": [log_group, f"{log_group}:*"]},
        {"Sid": "EndpointMetrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
         "Condition": {"StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/Endpoints"}}},
    ]
    if vpc:
        statements.append({
            "Sid": "VpcNetworking", "Effect": "Allow", "Resource": "*",
            "Action": [
                "ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface",
                "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces", "ec2:DescribeVpcs",
                "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups",
            ],
        })
    return {"Version": "2012-10-17", "Statement": statements}


def deployer_policy(
    *, region: str, account_id: str, endpoint_name: str, execution_role_arn: str, kms_key_arn: str,
    ecr_repository: str, artifact_bucket: str,
) -> dict:
    """What the human or pipeline role that runs tools/deploy needs (not the endpoint's own role)."""
    ep = f"arn:aws:sagemaker:{region}:{account_id}"
    name = f"{endpoint_name}*"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ManageEndpoint", "Effect": "Allow",
             "Action": ["sagemaker:CreateModel", "sagemaker:CreateEndpointConfig", "sagemaker:CreateEndpoint",
                        "sagemaker:UpdateEndpoint", "sagemaker:DescribeModel", "sagemaker:DescribeEndpointConfig",
                        "sagemaker:DescribeEndpoint", "sagemaker:AddTags", "sagemaker:InvokeEndpoint"],
             "Resource": [f"{ep}:model/{name}", f"{ep}:endpoint-config/{name}", f"{ep}:endpoint/{name}"]},
            {"Sid": "PassEndpointRole", "Effect": "Allow", "Action": "iam:PassRole", "Resource": [execution_role_arn],
             "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
            {"Sid": "EndpointVolumeKey", "Effect": "Allow", "Action": ["kms:CreateGrant", "kms:DescribeKey"],
             "Resource": [kms_key_arn]},
            {"Sid": "AutoScaling", "Effect": "Allow",
             "Action": ["application-autoscaling:RegisterScalableTarget", "application-autoscaling:PutScalingPolicy",
                        "application-autoscaling:DescribeScalableTargets", "application-autoscaling:DescribeScalingPolicies"],
             "Resource": "*"},
            {"Sid": "ServiceLinkedRoleForAutoScaling", "Effect": "Allow", "Action": "iam:CreateServiceLinkedRole",
             "Resource": f"arn:aws:iam::{account_id}:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/*",
             "Condition": {"StringEquals": {"iam:AWSServiceName": "sagemaker.application-autoscaling.amazonaws.com"}}},
            {"Sid": "Alarms", "Effect": "Allow", "Action": ["cloudwatch:PutMetricAlarm", "cloudwatch:DescribeAlarms"],
             "Resource": f"arn:aws:cloudwatch:{region}:{account_id}:alarm:{endpoint_name}*"},
            {"Sid": "LogMetricFilters", "Effect": "Allow", "Action": ["logs:PutMetricFilter", "logs:DescribeLogGroups"],
             "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/sagemaker/Endpoints/{endpoint_name}:*"},
            {"Sid": "PushImage", "Effect": "Allow",
             "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:DescribeImages"],
             "Resource": f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repository}"},
            {"Sid": "EcrToken", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {"Sid": "PublishArtifact", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"],
             "Resource": f"arn:aws:s3:::{artifact_bucket}/*"},
            {"Sid": "EncryptArtifact", "Effect": "Allow", "Action": ["kms:GenerateDataKey", "kms:Encrypt", "kms:Decrypt"],
             "Resource": [kms_key_arn]},
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--region", required=True)
        p.add_argument("--account", required=True, dest="account_id")

    b = sub.add_parser("bedrock", help="policy allowing invocation of one Bedrock inference profile")
    common(b)
    b.add_argument("--model-id", required=True)
    b.add_argument("--destination-region", action="append", dest="destination_regions")

    e = sub.add_parser("execution-role", help="permissions for the endpoint's execution role")
    common(e)
    e.add_argument("--endpoint-name", required=True)
    e.add_argument("--ecr-repository", required=True)
    e.add_argument("--artifact-s3-uri", required=True)
    e.add_argument("--kms-key-arn", required=True)
    e.add_argument("--model-id", required=True)
    e.add_argument("--destination-region", action="append", dest="destination_regions")
    e.add_argument("--no-vpc", action="store_true")

    sub.add_parser("trust", help="trust policy for the execution role")

    d = sub.add_parser("deployer", help="permissions for whoever runs tools/deploy")
    common(d)
    d.add_argument("--endpoint-name", required=True)
    d.add_argument("--execution-role-arn", required=True)
    d.add_argument("--kms-key-arn", required=True)
    d.add_argument("--ecr-repository", required=True)
    d.add_argument("--artifact-bucket", required=True)

    a = parser.parse_args(argv)
    try:
        if a.command == "bedrock":
            doc = bedrock_invoke_policy(region=a.region, account_id=a.account_id, model_id=a.model_id,
                                        destination_regions=a.destination_regions)
        elif a.command == "execution-role":
            doc = execution_role_policy(
                region=a.region, account_id=a.account_id, endpoint_name=a.endpoint_name, ecr_repository=a.ecr_repository,
                artifact_s3_uri=a.artifact_s3_uri, kms_key_arn=a.kms_key_arn, model_id=a.model_id,
                destination_regions=a.destination_regions, vpc=not a.no_vpc)
        elif a.command == "trust":
            doc = trust_policy()
        else:
            doc = deployer_policy(
                region=a.region, account_id=a.account_id, endpoint_name=a.endpoint_name,
                execution_role_arn=a.execution_role_arn, kms_key_arn=a.kms_key_arn,
                ecr_repository=a.ecr_repository, artifact_bucket=a.artifact_bucket)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
