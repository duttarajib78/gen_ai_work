"""Check Bedrock access and the exact request shape the service sends.

Reads the same AUDIT_* environment as the service (region, model id, effort, residency
opt-ins), makes one small tool-use call, and reports the outcome. Run it with the
credentials the endpoint will use (or as close as you can) before deploying.

    export AUDIT_AWS_REGION=ap-southeast-2 AUDIT_BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-5-5 \
           AUDIT_ALLOW_GLOBAL_INFERENCE=true
    python -m tools.diagnostics.smoke_test_bedrock

Exit code 0 means the call succeeded and the model answered through the tool.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from audit_agent.config import Settings
from audit_agent.deadline import Deadline
from audit_agent.errors import AuditError, ConfigError
from audit_agent.llm.base import ToolSpec, UsageMeter
from audit_agent.llm.bedrock import BedrockClaudeClient

READY_TOOL = ToolSpec(
    name="report_ready", description="Report that you are ready.",
    input_schema={"type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"],
                  "additionalProperties": False},
)

HINTS = {
    "llm_access_denied": [
        "Enable the model for this account in the Bedrock console (model access / Marketplace agreement).",
        "Check the role's policy: python -m tools.deploy.iam bedrock --region ... --account ... --model-id ...",
        "A global. profile also needs the Region-less foundation-model statement; check SCPs for 'unspecified'.",
        "Confirm the profile id exists in this Region: aws bedrock list-inference-profiles --region ...",
    ],
    "llm_request_rejected": [
        "Read the Bedrock message above: Sonnet 5.5 rejects temperature/top_p/top_k, forced tool choice, "
        "and disabled or budgeted thinking.",
        "If you set AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON, try without it.",
    ],
    "llm_throttled": ["Check Bedrock quotas for this model and Region."],
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"model: {settings.bedrock_model_id}  region: {settings.bedrock_region}  routing: {settings.routing_scope}")
    print(f"effort: {settings.llm_effort}  thinking: {settings.llm_thinking_mode}")
    client = BedrockClaudeClient.from_settings(settings)
    meter = UsageMeter()
    try:
        result = client.invoke_tool(
            system_prompt="You are a connectivity check. Answer only by calling the tool.",
            user_text="Call report_ready with ready=true.", tool=READY_TOOL, deadline=Deadline(90), meter=meter)
    except AuditError as exc:
        print(f"FAILED: {exc.code}: {exc}", file=sys.stderr)
        for hint in HINTS.get(exc.code, []):
            print(f"  - {hint}", file=sys.stderr)
        return 1
    print(f"OK: tool={result.tool_name} args={dict(result.arguments)} stop={result.stop_reason}")
    print(f"tokens in/out: {meter.input_tokens}/{meter.output_tokens}  latency: {meter.latency_ms} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
