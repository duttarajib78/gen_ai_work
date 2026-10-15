"""Route the Claude Agent SDK (the bundled Claude Code CLI) to Amazon Bedrock.

Verified against the Claude Code Bedrock documentation (October 2026):
  * ``CLAUDE_CODE_USE_BEDROCK=1`` selects the bedrock-runtime (InvokeModel) path; ``CLAUDE_CODE_USE_MANTLE=1``
    selects the Messages-API "Mantle" path. Default here is bedrock-runtime because it supports Bedrock model
    invocation logging and Guardrails (Mantle does not).
  * ``AWS_REGION`` must be set; credentials come from the standard AWS chain (the SageMaker execution role).
  * Newer Claude models are reachable only through inference profiles (``<geo>.anthropic.<model>``); the geo
    prefix controls where inference may run (``au`` keeps it inside Australia, ``global`` allows any region).
  * WebSearch is NOT available on Bedrock.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..errors import ConfigError
from ..settings import INFERENCE_PROFILE_PREFIXES, Settings

_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{2,200}$")
_ARN_RE = re.compile(r"^arn:aws[a-z\-]*:bedrock:[a-z0-9\-]+:\d{12}:(?:application-)?inference-profile/[A-Za-z0-9._:\-]+$")
_KNOWN_PREFIXES = tuple(p for p in INFERENCE_PROFILE_PREFIXES if p)

# Variables that must never steer a Bedrock deployment to another auth path.
BLANKED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL")


def resolve_model_id(model: str, prefix: str, provider: str = "invoke") -> str:
    """Apply the inference-profile geo prefix to a base model id; pass ARNs / already-prefixed ids through."""
    value = (model or "").strip()
    if _ARN_RE.match(value):
        return value
    if value.lower().startswith("arn:") or not _MODEL_ID_RE.match(value):
        raise ConfigError(f"invalid Bedrock model id: {value[:60]!r}")
    if provider == "mantle":      # Mantle uses unprefixed ids such as 'anthropic.claude-sonnet-5'
        return value
    if prefix and value.split(".", 1)[0] not in _KNOWN_PREFIXES:
        return f"{prefix}.{value}"
    return value


def resolved_models(settings: Settings) -> dict[str, str]:
    b = settings.bedrock
    return {
        "strong": resolve_model_id(b.model_strong, b.inference_profile_prefix, b.provider),
        "fast": resolve_model_id(b.model_fast, b.inference_profile_prefix, b.provider),
        "web": resolve_model_id(b.model_web, b.inference_profile_prefix, b.provider),
    }


def build_sdk_env(settings: Settings, *, config_dir: Path) -> dict[str, str]:
    """Environment overlay for one SDK session (merged over ``os.environ`` by the SDK)."""
    b = settings.bedrock
    if not _REGION_RE.match(b.region or ""):
        raise ConfigError(f"AWS region is missing or malformed: {b.region!r}")
    models = resolved_models(settings)
    env: dict[str, str] = {name: "" for name in BLANKED_ENV}      # empty string == unset for provider selection
    if settings.service.environment == "prod":
        env["AWS_BEARER_TOKEN_BEDROCK"] = ""                       # production uses the IAM role, never a long-lived key
    env.update({
        "CLAUDE_CODE_USE_BEDROCK": "1" if b.provider == "invoke" else "",
        "CLAUDE_CODE_USE_MANTLE": "1" if b.provider == "mantle" else "",
        "AWS_REGION": b.region,
        "AWS_DEFAULT_REGION": b.region,
        "ANTHROPIC_MODEL": models["strong"],
        "ANTHROPIC_DEFAULT_SONNET_MODEL": models["strong"],
        "ANTHROPIC_DEFAULT_OPUS_MODEL": models["strong"],
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": models["fast"],
        "ANTHROPIC_SMALL_FAST_MODEL": models["fast"],
        "ANTHROPIC_BEDROCK_SERVICE_TIER": b.service_tier,
        # Outbound-traffic and local-state hygiene for an isolated container:
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_BUG_COMMAND": "1",
        "DISABLE_COST_WARNINGS": "1",
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "HOME": str(config_dir),
    })
    if b.effort:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = b.effort
    return env
