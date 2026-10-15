"""Runtime configuration.

All configuration comes from environment variables (set on the SageMaker model
as ``Environment``). Nothing is read from files, no ``.env`` loading, and no
credentials are accepted here: AWS access comes from the endpoint execution
role through the default boto3 credential chain.

Settings are validated once at startup; every problem is reported together so a
bad deployment fails fast with one actionable message instead of one error per
restart.

Two settings are deliberately *fail closed* because they decide where patient-
derived prompts may be processed:

* ``AUDIT_ALLOW_GLOBAL_INFERENCE``      required to use a ``global.`` inference
                                        profile (routing can leave the geography)
* ``AUDIT_ALLOW_CROSS_REGION_BEDROCK``  required when the Bedrock client region
                                        differs from the endpoint region
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from . import __version__
from .errors import ConfigError

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
THINKING_MODES = ("adaptive", "between_tools")
GROUPER_BACKENDS = ("au_drg", "disabled")
MBS_ARRAY_FIELDS = ("medical_claims_array", "hospital_claims_array")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# e.g. anthropic.claude-sonnet-5-5, global.anthropic.claude-sonnet-5-5, us.anthropic....
_MODEL_ID_RE = re.compile(r"^(?:[a-z]{2,10}\.)?anthropic\.claude-[a-z0-9.\-]+(?::\d+)?$")
# system-defined or application inference profile ARN
_PROFILE_ARN_RE = re.compile(
    r"^arn:aws[a-z\-]*:bedrock:[a-z0-9\-]+:\d{12}:(?:application-)?inference-profile/[A-Za-z0-9._:\-]+$"
)
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


@dataclass(frozen=True)
class Settings:
    # ---- runtime / data ----------------------------------------------------
    aws_region: str
    bedrock_region: str
    model_dir: Path
    require_manifest: bool
    mbs_array_field: str

    # ---- LLM (Amazon Bedrock, Converse API) ----------------------------------
    bedrock_model_id: str
    allow_global_inference: bool
    allow_cross_region_bedrock: bool
    llm_effort: str
    llm_thinking_mode: str
    llm_max_output_tokens: int
    llm_max_output_tokens_ceiling: int
    llm_max_output_attempts: int
    llm_max_transport_retries: int
    llm_connect_timeout_s: float
    llm_read_timeout_s: float
    llm_min_attempt_budget_s: float
    llm_prompt_caching: bool
    llm_extra_request_fields: Mapping[str, Any]
    guardrail_id: Optional[str]
    guardrail_version: Optional[str]

    # ---- decision rule -------------------------------------------------------
    low_confidence_weak_threshold: float

    # ---- grouper -------------------------------------------------------------
    grouper_backend: str
    grouper_stack_name: str
    grouper_serialize_calls: bool
    min_grouper_budget_s: float

    # ---- alert record --------------------------------------------------------
    rule_id: int

    # ---- serving -------------------------------------------------------------
    request_deadline_s: float
    stream_deadline_s: float
    stream_heartbeat_s: float
    max_request_bytes: int
    max_concurrent_requests: int
    max_diagnoses: int
    max_procedures: int
    max_mbs_items: int

    # ---- logging -------------------------------------------------------------
    log_level: str
    log_raw_identifiers: bool

    service_version: str = __version__

    # ------------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, env: Mapping[str, str]) -> "Settings":
        r = _Reader(env)

        aws_region = r.string("AUDIT_AWS_REGION", default=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION"), required=True)
        if aws_region and not _REGION_RE.match(aws_region):
            r.error("AUDIT_AWS_REGION", f"{aws_region!r} is not a valid AWS region name")
        bedrock_region = r.string("AUDIT_BEDROCK_REGION", default=aws_region) or aws_region

        model_id = r.string("AUDIT_BEDROCK_MODEL_ID", required=True)
        if model_id and not (_MODEL_ID_RE.match(model_id) or _PROFILE_ARN_RE.match(model_id)):
            r.error(
                "AUDIT_BEDROCK_MODEL_ID",
                "must be a Claude model or inference-profile id (e.g. 'global.anthropic.claude-sonnet-5-5') "
                "or an inference-profile ARN",
            )

        allow_global = r.boolean("AUDIT_ALLOW_GLOBAL_INFERENCE", default=False)
        allow_cross_region = r.boolean("AUDIT_ALLOW_CROSS_REGION_BEDROCK", default=False)
        if model_id and model_id.startswith("global.") and not allow_global:
            r.error(
                "AUDIT_BEDROCK_MODEL_ID",
                "a 'global.' inference profile can route prompts to any commercial AWS Region. Set "
                "AUDIT_ALLOW_GLOBAL_INFERENCE=true only after your data-governance owner has approved that "
                "(see README, 'Decisions required before go-live'), or choose a geography-scoped profile.",
            )
        if aws_region and bedrock_region != aws_region and not allow_cross_region:
            r.error(
                "AUDIT_BEDROCK_REGION",
                "differs from the endpoint region, which sends prompts across Regions. Set "
                "AUDIT_ALLOW_CROSS_REGION_BEDROCK=true only if that has been approved.",
            )
        if bedrock_region and not _REGION_RE.match(bedrock_region):
            r.error("AUDIT_BEDROCK_REGION", f"{bedrock_region!r} is not a valid AWS region name")

        effort = r.choice("AUDIT_LLM_EFFORT", EFFORT_LEVELS, default="high")
        thinking = r.choice("AUDIT_LLM_THINKING_MODE", THINKING_MODES, default="adaptive")
        if thinking == "between_tools" and effort in ("xhigh", "max"):
            r.error("AUDIT_LLM_THINKING_MODE", "'between_tools' is only valid with effort low, medium, or high")

        max_tokens = r.integer("AUDIT_LLM_MAX_OUTPUT_TOKENS", default=16000, minimum=1024, maximum=128000)
        ceiling = r.integer("AUDIT_LLM_MAX_OUTPUT_TOKENS_CEILING", default=64000, minimum=1024, maximum=128000)
        if max_tokens > ceiling:
            r.error("AUDIT_LLM_MAX_OUTPUT_TOKENS_CEILING", "must be >= AUDIT_LLM_MAX_OUTPUT_TOKENS")

        guardrail_id = r.string("AUDIT_BEDROCK_GUARDRAIL_ID")
        guardrail_version = r.string("AUDIT_BEDROCK_GUARDRAIL_VERSION")
        if bool(guardrail_id) != bool(guardrail_version):
            r.error("AUDIT_BEDROCK_GUARDRAIL_ID", "guardrail id and version must be set together")

        extra_raw = r.string("AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON")
        extra: dict = {}
        if extra_raw:
            try:
                parsed = json.loads(extra_raw)
                if not isinstance(parsed, dict):
                    raise ValueError("not an object")
                extra = parsed
            except ValueError:
                r.error("AUDIT_BEDROCK_EXTRA_REQUEST_FIELDS_JSON", "must be a JSON object")

        threshold = r.number("AUDIT_LOW_CONFIDENCE_WEAK_THRESHOLD", default=0.5, minimum=0.0, maximum=1.0)

        grouper_backend = r.choice("AUDIT_GROUPER_BACKEND", GROUPER_BACKENDS, default="au_drg")
        stack = r.string("AUDIT_GROUPER_STACK_NAME", default="hambs-shed") or "hambs-shed"
        if not _NAME_RE.match(stack):
            r.error("AUDIT_GROUPER_STACK_NAME", "contains unsupported characters")

        request_deadline = r.number("AUDIT_REQUEST_DEADLINE_S", default=50.0, minimum=5.0, maximum=58.0)
        stream_deadline = r.number("AUDIT_STREAM_DEADLINE_S", default=420.0, minimum=30.0, maximum=470.0)

        model_dir = Path(r.string("AUDIT_MODEL_DIR", default="/opt/ml/model") or "/opt/ml/model")

        mbs_field = r.choice("AUDIT_MBS_ARRAY_FIELD", MBS_ARRAY_FIELDS, default="medical_claims_array")
        log_level = r.choice("AUDIT_LOG_LEVEL", LOG_LEVELS, default="INFO", upper=True)

        settings = cls(
            aws_region=aws_region or "",
            bedrock_region=bedrock_region or "",
            model_dir=model_dir,
            require_manifest=r.boolean("AUDIT_REQUIRE_MANIFEST", default=True),
            mbs_array_field=mbs_field,
            bedrock_model_id=model_id or "",
            allow_global_inference=allow_global,
            allow_cross_region_bedrock=allow_cross_region,
            llm_effort=effort,
            llm_thinking_mode=thinking,
            llm_max_output_tokens=max_tokens,
            llm_max_output_tokens_ceiling=ceiling,
            llm_max_output_attempts=r.integer("AUDIT_LLM_MAX_OUTPUT_ATTEMPTS", default=3, minimum=1, maximum=5),
            llm_max_transport_retries=r.integer("AUDIT_LLM_MAX_TRANSPORT_RETRIES", default=2, minimum=0, maximum=5),
            llm_connect_timeout_s=r.number("AUDIT_LLM_CONNECT_TIMEOUT_S", default=5.0, minimum=1.0, maximum=30.0),
            llm_read_timeout_s=r.number("AUDIT_LLM_READ_TIMEOUT_S", default=120.0, minimum=10.0, maximum=3600.0),
            llm_min_attempt_budget_s=r.number("AUDIT_LLM_MIN_ATTEMPT_BUDGET_S", default=8.0, minimum=1.0, maximum=60.0),
            llm_prompt_caching=r.boolean("AUDIT_BEDROCK_PROMPT_CACHING", default=False),
            llm_extra_request_fields=extra,
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            low_confidence_weak_threshold=threshold,
            grouper_backend=grouper_backend,
            grouper_stack_name=stack,
            grouper_serialize_calls=r.boolean("AUDIT_GROUPER_SERIALIZE_CALLS", default=True),
            min_grouper_budget_s=r.number("AUDIT_MIN_GROUPER_BUDGET_S", default=5.0, minimum=1.0, maximum=60.0),
            rule_id=r.integer("AUDIT_RULE_ID", default=62248, minimum=1, maximum=2**31 - 1),
            request_deadline_s=request_deadline,
            stream_deadline_s=stream_deadline,
            stream_heartbeat_s=r.number("AUDIT_STREAM_HEARTBEAT_S", default=10.0, minimum=0.01, maximum=60.0),
            max_request_bytes=r.integer("AUDIT_MAX_REQUEST_BYTES", default=1_048_576, minimum=4096, maximum=8_388_608),
            max_concurrent_requests=r.integer("AUDIT_MAX_CONCURRENT_REQUESTS", default=16, minimum=1, maximum=256),
            max_diagnoses=r.integer("AUDIT_MAX_DIAGNOSES", default=100, minimum=1, maximum=500),
            max_procedures=r.integer("AUDIT_MAX_PROCEDURES", default=200, minimum=0, maximum=1000),
            max_mbs_items=r.integer("AUDIT_MAX_MBS_ITEMS", default=300, minimum=0, maximum=2000),
            log_level=log_level,
            log_raw_identifiers=r.boolean("AUDIT_LOG_RAW_IDENTIFIERS", default=False),
        )
        r.raise_if_any()
        return settings

    # ------------------------------------------------------------------------
    @property
    def routing_scope(self) -> str:
        """Best-effort statement of where prompts may be processed, for logs and provenance."""
        mid = self.bedrock_model_id
        if mid.startswith("global."):
            return "global"
        if mid.startswith("arn:"):
            return "unknown-arn"
        prefix = mid.split(".", 1)[0] if mid.count(".") >= 2 else ""
        return f"geo:{prefix}" if prefix else "in-region"

    def describe(self) -> dict:
        """Startup log line: non-sensitive operating parameters only."""
        return {
            "version": self.service_version,
            "aws_region": self.aws_region,
            "bedrock_region": self.bedrock_region,
            "model_id": self.bedrock_model_id,
            "routing_scope": self.routing_scope,
            "effort": self.llm_effort,
            "thinking_mode": self.llm_thinking_mode,
            "threshold": self.low_confidence_weak_threshold,
            "grouper_backend": self.grouper_backend,
            "request_deadline_s": self.request_deadline_s,
            "stream_deadline_s": self.stream_deadline_s,
        }


@dataclass
class _Reader:
    env: Mapping[str, str]
    errors: list = field(default_factory=list)

    def error(self, key: str, message: str) -> None:
        self.errors.append(f"{key}: {message}")

    def raise_if_any(self) -> None:
        if self.errors:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(self.errors))

    def string(self, key: str, *, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        value = self.env.get(key)
        if value is None or value.strip() == "":
            value = default
        if value is None or str(value).strip() == "":
            if required:
                self.error(key, "is required")
            return None
        return str(value).strip()

    def boolean(self, key: str, *, default: bool) -> bool:
        raw = self.env.get(key)
        if raw is None or raw.strip() == "":
            return default
        v = raw.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        self.error(key, f"expected true/false, got {raw!r}")
        return default

    def integer(self, key: str, *, default: int, minimum: int, maximum: int) -> int:
        raw = self.env.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            value = int(raw.strip())
        except ValueError:
            self.error(key, f"expected an integer, got {raw!r}")
            return default
        if not minimum <= value <= maximum:
            self.error(key, f"must be between {minimum} and {maximum}, got {value}")
            return default
        return value

    def number(self, key: str, *, default: float, minimum: float, maximum: float) -> float:
        raw = self.env.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            value = float(raw.strip())
        except ValueError:
            self.error(key, f"expected a number, got {raw!r}")
            return default
        if not (minimum <= value <= maximum) or value != value:
            self.error(key, f"must be between {minimum} and {maximum}, got {raw!r}")
            return default
        return value

    def choice(self, key: str, options: tuple, *, default: str, upper: bool = False) -> str:
        raw = self.env.get(key)
        if raw is None or raw.strip() == "":
            return default
        value = raw.strip().upper() if upper else raw.strip().lower()
        if value not in options:
            self.error(key, f"must be one of {list(options)}, got {raw!r}")
            return default
        return value
