"""Pre-warming the endpoint: run the batch build as a SageMaker Processing job that is configured FROM the live endpoint.

The endpoint serves what it finds under serve/<fingerprint>/, and the fingerprint covers the model, thinking setting, prompts, tunables and reference
data. A batch build only warms the endpoint if it runs with exactly the same settings, so this module reads them from the endpoint's own SageMaker model
(image, environment, role, VPC, reference-data artifact) instead of asking you to repeat them: they cannot drift apart.
"""
from __future__ import annotations

import shlex
import time
from typing import Any, Optional


def build_processing_request(model: dict, build_args: list[str], *, instance_type: str = "ml.m5.xlarge", volume_gb: int = 30, max_runtime_s: int = 86400,
                             job_name: Optional[str] = None) -> dict:
    """The CreateProcessingJob request, from a DescribeModel response and the arguments for `dx-support build` (always run with --for-service and --yes)."""
    container = model["PrimaryContainer"]
    if not container.get("ModelDataUrl"):
        raise ValueError("the endpoint's model has no ModelDataUrl, so there is no reference-data artifact to build from")
    job_name = job_name or time.strftime("dx-prewarm-%Y%m%d%H%M%S")
    script = ("set -eu; mkdir -p /tmp/d; tar -xzf /opt/ml/processing/input/model.tar.gz -C /tmp/d; export DX_DATA_DIR=/tmp/d/data; "
              "exec dx-support build --for-service --yes " + " ".join(shlex.quote(a) for a in build_args))
    req: dict[str, Any] = {
        "ProcessingJobName": job_name,
        "RoleArn": model["ExecutionRoleArn"],
        "AppSpecification": {"ImageUri": container["Image"], "ContainerEntrypoint": ["/bin/sh", "-c"], "ContainerArguments": [script]},
        "Environment": dict(container.get("Environment") or {}),
        "ProcessingInputs": [{"InputName": "reference-data", "S3Input": {"S3Uri": container["ModelDataUrl"], "LocalPath": "/opt/ml/processing/input", "S3DataType": "S3Prefix",
                                                                    "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}}],
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": instance_type, "VolumeSizeInGB": volume_gb}},
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_s},
        "Tags": [{"Key": "project", "Value": "dx-support"}, {"Key": "purpose", "Value": "prewarm"}],
    }
    if model.get("VpcConfig"):
        req["NetworkConfig"] = {"EnableNetworkIsolation": False, "VpcConfig": {"SecurityGroupIds": list(model["VpcConfig"]["SecurityGroupIds"]),
                                                                              "Subnets": list(model["VpcConfig"]["Subnets"])}}
    return req


def describe_endpoint_model(sm: Any, endpoint: str) -> dict:
    ep = sm.describe_endpoint(EndpointName=endpoint)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    return sm.describe_model(ModelName=cfg["ProductionVariants"][0]["ModelName"])
