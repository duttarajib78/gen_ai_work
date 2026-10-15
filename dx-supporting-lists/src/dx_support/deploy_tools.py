"""Helpers for deployment: the exact IAM resources the endpoint needs for Bedrock."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from dx_support.llm.catalog import KNOWN_PREFIXES

# Regions an inference profile routes to, where this project knows them. Everything else must be given explicitly or discovered from Bedrock
# (`aws bedrock get-inference-profile` lists the exact foundation-model ARNs of every destination region).
DEFAULT_DESTINATIONS = {"au": ("ap-southeast-2", "ap-southeast-4")}


def split_profile_id(model_id: str) -> tuple[str, str]:
    """'au.anthropic.claude-sonnet-5' -> ('au', 'anthropic.claude-sonnet-5'); a base id -> ('', base id)."""
    head, _, rest = model_id.partition(".")
    return (head, rest) if head in KNOWN_PREFIXES and rest.startswith("anthropic.") else ("", model_id)


def build_arns(partition: str, region: str, account: str, model_ids: list[str], destinations: Optional[Mapping[str, list[str]]] = None) -> list[str]:
    """The ARNs `bedrock:InvokeModel*` must be allowed on, for each model id the endpoint will send as `modelId`.

    A base model id needs only its foundation-model ARN in the calling region. An inference profile needs the PROFILE's ARN (in the calling region and
    account) plus the foundation-model ARN in EVERY region the profile can route to; a global profile also needs the region-less foundation-model ARN.
    Granting the profile alone is not enough, and granting wildcards is broader than necessary."""
    arns: list[str] = []
    for mid in model_ids:
        prefix, base = split_profile_id(mid)
        if not prefix:
            arns.append(f"arn:{partition}:bedrock:{region}::foundation-model/{base}")
            continue
        arns.append(f"arn:{partition}:bedrock:{region}:{account}:inference-profile/{mid}")
        if prefix == "global":
            arns += [f"arn:{partition}:bedrock:::foundation-model/{base}", f"arn:{partition}:bedrock:{region}::foundation-model/{base}"]
        else:
            for dest in (destinations or {}).get(mid) or DEFAULT_DESTINATIONS.get(prefix) or (region,):
                arns.append(f"arn:{partition}:bedrock:{dest}::foundation-model/{base}")
    return list(dict.fromkeys(arns))


def discover_destinations(bedrock_client: Any, model_ids: list[str]) -> dict[str, list[str]]:
    """Asks Bedrock which regions each profile routes to (needs bedrock:GetInferenceProfile). Returns {profile id: [region, ...]}."""
    out: dict[str, list[str]] = {}
    for mid in model_ids:
        if not split_profile_id(mid)[0]:
            continue
        resp = bedrock_client.get_inference_profile(inferenceProfileIdentifier=mid)
        regions = []
        for m in resp.get("models", []):
            arn = str(m.get("modelArn", ""))
            parts = arn.split(":")
            if len(parts) > 3 and parts[3]:
                regions.append(parts[3])
        if regions:
            out[mid] = sorted(set(regions))
    return out
