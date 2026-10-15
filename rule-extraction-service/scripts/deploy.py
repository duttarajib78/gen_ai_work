#!/usr/bin/env python3
"""Deployment helper (boto3 + CloudFormation). Uses your current AWS credentials; stores and prints no secrets.

  deploy.py stack --stack-name N --template T [--param K=V ...]     create/update a stack and wait
  deploy.py outputs --stack-name N                                  print stack outputs as JSON
  deploy.py upload-model --file dist/model.tar.gz --foundation-stack N   upload to the artifact bucket (SSE-KMS)
  deploy.py scale --endpoint E --instances N                        change the live instance count
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _cfn(region):
    import boto3
    return boto3.client("cloudformation", region_name=region)


def outputs(cfn, stack_name: str) -> dict:
    stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


try:
    from botocore.exceptions import ClientError
except ImportError:  # keeps the module importable (and testable) without boto3 installed
    ClientError = Exception  # type: ignore[misc,assignment]


def deploy_stack(cfn, stack_name: str, template: str, params: dict) -> dict:
    body = Path(template).read_text(encoding="utf-8")
    kwargs = dict(StackName=stack_name, TemplateBody=body, Capabilities=["CAPABILITY_NAMED_IAM"],
                  Parameters=[{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()])
    try:
        cfn.describe_stacks(StackName=stack_name)
        exists = True
    except ClientError:
        exists = False
    try:
        if exists:
            cfn.update_stack(**kwargs)
            cfn.get_waiter("stack_update_complete").wait(StackName=stack_name)
        else:
            cfn.create_stack(**kwargs, EnableTerminationProtection=True)
            cfn.get_waiter("stack_create_complete").wait(StackName=stack_name)
    except ClientError as exc:
        if "No updates are to be performed" not in str(exc):
            raise
        print("no changes to deploy", file=sys.stderr)
    return outputs(cfn, stack_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", help="defaults to your configured AWS region")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stack")
    s.add_argument("--stack-name", required=True)
    s.add_argument("--template", required=True)
    s.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    o = sub.add_parser("outputs")
    o.add_argument("--stack-name", required=True)
    u = sub.add_parser("upload-model")
    u.add_argument("--file", required=True)
    u.add_argument("--foundation-stack", required=True)
    u.add_argument("--key-prefix", default="models")
    sc = sub.add_parser("scale")
    sc.add_argument("--endpoint", required=True)
    sc.add_argument("--instances", type=int, required=True)
    args = parser.parse_args()

    if args.cmd == "stack":
        params = dict(p.split("=", 1) for p in args.param)
        print(json.dumps(deploy_stack(_cfn(args.region), args.stack_name, args.template, params), indent=2))
    elif args.cmd == "outputs":
        print(json.dumps(outputs(_cfn(args.region), args.stack_name), indent=2))
    elif args.cmd == "upload-model":
        import boto3
        out = outputs(_cfn(args.region), args.foundation_stack)
        digest = hashlib.sha256(Path(args.file).read_bytes()).hexdigest()
        key = f"{args.key_prefix}/model-{digest[:16]}.tar.gz"      # content-addressed: immutable per artifact
        boto3.client("s3", region_name=args.region).upload_file(
            args.file, out["ArtifactBucket"], key,
            ExtraArgs={"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": out["KmsKeyArn"]})
        print(json.dumps({"ModelDataUrl": f"s3://{out['ArtifactBucket']}/{key}", "ModelVersion": digest[:12], "sha256": digest}, indent=2))
    elif args.cmd == "scale":
        import boto3
        boto3.client("sagemaker", region_name=args.region).update_endpoint_weights_and_capacities(
            EndpointName=args.endpoint,
            DesiredWeightsAndCapacities=[{"VariantName": "AllTraffic", "DesiredInstanceCount": args.instances}])
        print(f"requested {args.instances} instance(s) for {args.endpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
