"""Typed, validated runtime configuration.

Sources, lowest to highest precedence:
  1. built-in defaults
  2. optional YAML file (structured defaults baked into the image: ``configs/base.yaml``)
  3. environment variables (``RULE_EXTRACTOR_*`` plus the standard ``AWS_REGION``)

No secrets are read, stored, or logged here. Credentials come from the IAM role attached to the
SageMaker endpoint (or your developer role) through the default AWS credential chain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PREFIX = "RULE_EXTRACTOR_"

ALL_OPERATIONS = (
    "plan", "submit_job", "get_job", "list_jobs", "get_rules", "get_results",
    "cancel_job", "retry_failed", "health", "evaluate_episodes",
)
DEFAULT_ENABLED_OPERATIONS = tuple(op for op in ALL_OPERATIONS if op != "evaluate_episodes")
SWEEP_DOCUMENT_KEYS = ("icd10am_tabular", "achi_tabular", "icd10am_alphabetic", "achi_alphabetic")
INFERENCE_PROFILE_PREFIXES = ("", "au", "apac", "us", "eu", "jp", "global")


@dataclass(frozen=True)
class PathSettings:
    model_dir: Path
    corpus_dir: Path
    source_docs_dir: Path
    work_dir: Path
    output_dir: Path
    state_dir: Path
    template_path: Path | None = None       # workbook the writer copies from (default: model_dir/templates/... or source_docs/...)
    base_rules_path: Path | None = None     # executable rules already in the base workbook (default: model_dir/base/...)


@dataclass(frozen=True)
class BedrockSettings:
    region: str
    inference_profile_prefix: str
    model_strong: str
    model_fast: str
    model_web: str
    provider: str            # "invoke" (bedrock-runtime) | "mantle"
    service_tier: str
    effort: str | None


@dataclass(frozen=True)
class AgentSettings:
    max_concurrent_sessions: int
    max_retries: int
    retry_base_delay_seconds: float
    turn_timeout_seconds: int
    seconds_per_turn_budget: int
    max_turns: int
    max_turns_cap: int
    max_budget_usd_per_unit: float | None
    max_rules_per_unit: int
    require_verbatim_prompts: bool


@dataclass(frozen=True)
class ChunkingSettings:
    tabular_pages_per_chunk: int
    tabular_page_overlap: int
    alphabetic_pages_per_chunk: int
    alphabetic_page_overlap: int
    min_acs_standards: int


@dataclass(frozen=True)
class AwsSettings:
    state_table: str | None
    work_queue_url: str | None
    artifact_bucket: str | None
    artifact_prefix: str
    kms_key_id: str | None
    metrics_namespace: str
    endpoint_name: str


@dataclass(frozen=True)
class ServiceSettings:
    backend: str                      # "aws" | "local"
    environment: str                  # "dev" | "prod"
    allowed_operations: tuple[str, ...]
    max_request_bytes: int
    max_units_per_job: int
    job_retention_days: int
    lease_seconds: int
    job_max_runtime_seconds: int
    max_episodes_per_request: int
    run_as_uid: int
    web_workers: int
    web_threads: int
    shutdown_grace_seconds: int


@dataclass(frozen=True)
class WebResearchSettings:
    enabled: bool
    allowed_hosts: tuple[str, ...]
    default_topics: tuple[str, ...]


@dataclass(frozen=True)
class OutputSettings:
    excel_sheet_name: str
    footer_enabled: bool
    footer_lines: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    paths: PathSettings
    bedrock: BedrockSettings
    agent: AgentSettings
    chunking: ChunkingSettings
    aws: AwsSettings
    service: ServiceSettings
    web: WebResearchSettings
    output: OutputSettings
    grouper_factory: str | None = None
    enable_evaluate: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, env: Mapping[str, str] | None = None, config_file: str | Path | None = None) -> "Settings":
        env = os.environ if env is None else env
        cfg_path = config_file or env.get(ENV_PREFIX + "CONFIG_FILE")
        yaml_cfg: dict[str, Any] = {}
        if cfg_path:
            yaml_cfg = _read_yaml(Path(cfg_path))
        reader = _Reader(env, yaml_cfg)

        model_dir = Path(reader.str("MODEL_DIR", "/opt/ml/model"))
        paths = PathSettings(
            model_dir=model_dir,
            corpus_dir=Path(reader.str("CORPUS_DIR", str(model_dir / "corpus"))),
            source_docs_dir=Path(reader.str("SOURCE_DOCS_DIR", str(PROJECT_ROOT / "source_docs"))),
            work_dir=Path(reader.str("WORK_DIR", "/tmp/rule-extractor")),
            output_dir=Path(reader.str("OUTPUT_DIR", str(PROJECT_ROOT / "sample_output"))),
            state_dir=Path(reader.str("STATE_DIR", str(PROJECT_ROOT / "checkpoints"))),
            template_path=Path(reader.str("TEMPLATE_PATH", "")) if reader.str("TEMPLATE_PATH", "") else None,
            base_rules_path=Path(reader.str("BASE_RULES_PATH", "")) if reader.str("BASE_RULES_PATH", "") else None,
        )

        region = env.get(ENV_PREFIX + "BEDROCK_REGION") or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or ""
        prefix = reader.str("BEDROCK_PROFILE_PREFIX", "au").strip().lower()
        if prefix not in INFERENCE_PROFILE_PREFIXES:
            reader.fail(f"BEDROCK_PROFILE_PREFIX must be one of {INFERENCE_PROFILE_PREFIXES}, got {prefix!r}")
        strong = reader.str("MODEL_STRONG", "anthropic.claude-sonnet-5")
        bedrock = BedrockSettings(
            region=region,
            inference_profile_prefix=prefix,
            model_strong=strong,
            model_fast=reader.str("MODEL_FAST", "anthropic.claude-haiku-4-5-20251001-v1:0"),
            model_web=reader.str("MODEL_WEB", strong),
            provider=reader.choice("BEDROCK_PROVIDER", "invoke", ("invoke", "mantle")),
            service_tier=reader.choice("BEDROCK_SERVICE_TIER", "default", ("default", "flex", "priority")),
            effort=reader.str("EFFORT", "") or None,
        )

        budget = reader.float("MAX_BUDGET_USD_PER_UNIT", 5.0, minimum=0.0)
        agent = AgentSettings(
            max_concurrent_sessions=reader.int("CONCURRENCY", 3, minimum=1, maximum=32),
            max_retries=reader.int("MAX_RETRIES", 4, minimum=0, maximum=10),
            retry_base_delay_seconds=reader.float("RETRY_BASE_DELAY_SECONDS", 5.0, minimum=0.0),
            turn_timeout_seconds=reader.int("AGENT_TIMEOUT_SECONDS", 480, minimum=30),
            seconds_per_turn_budget=reader.int("SECONDS_PER_TURN", 25, minimum=1),
            max_turns=reader.int("MAX_TURNS", 40, minimum=1),
            max_turns_cap=reader.int("MAX_TURNS_CAP", 100, minimum=1),
            max_budget_usd_per_unit=budget if budget > 0 else None,
            max_rules_per_unit=reader.int("MAX_RULES_PER_UNIT", 40, minimum=1, maximum=500),
            require_verbatim_prompts=reader.bool("REQUIRE_VERBATIM_PROMPTS", True),
        )
        chunking = ChunkingSettings(
            tabular_pages_per_chunk=reader.int("TABULAR_PAGES_PER_CHUNK", 6, minimum=1),
            tabular_page_overlap=reader.int("TABULAR_PAGE_OVERLAP", 1, minimum=0),
            alphabetic_pages_per_chunk=reader.int("ALPHABETIC_PAGES_PER_CHUNK", 10, minimum=1),
            alphabetic_page_overlap=reader.int("ALPHABETIC_PAGE_OVERLAP", 1, minimum=0),
            min_acs_standards=reader.int("MIN_ACS_STANDARDS", 100, minimum=1),
        )
        if chunking.tabular_page_overlap >= chunking.tabular_pages_per_chunk:
            reader.fail("TABULAR_PAGE_OVERLAP must be smaller than TABULAR_PAGES_PER_CHUNK")
        if chunking.alphabetic_page_overlap >= chunking.alphabetic_pages_per_chunk:
            reader.fail("ALPHABETIC_PAGE_OVERLAP must be smaller than ALPHABETIC_PAGES_PER_CHUNK")

        aws = AwsSettings(
            state_table=reader.str("STATE_TABLE", "") or None,
            work_queue_url=reader.str("QUEUE_URL", "") or None,
            artifact_bucket=reader.str("ARTIFACT_BUCKET", "") or None,
            artifact_prefix=reader.str("ARTIFACT_PREFIX", "rule-extractor").strip("/"),
            kms_key_id=reader.str("KMS_KEY_ID", "") or None,
            metrics_namespace=reader.str("METRICS_NAMESPACE", "RuleExtractor"),
            endpoint_name=reader.str("ENDPOINT_NAME", "rule-extractor"),
        )

        enabled_ops = reader.csv("ALLOWED_OPERATIONS", DEFAULT_ENABLED_OPERATIONS)
        unknown_ops = [op for op in enabled_ops if op not in ALL_OPERATIONS]
        if unknown_ops:
            reader.fail(f"ALLOWED_OPERATIONS contains unknown operations: {unknown_ops}")
        enable_evaluate = reader.bool("ENABLE_EVALUATE", False)
        if enable_evaluate and "evaluate_episodes" not in enabled_ops:
            enabled_ops = tuple(enabled_ops) + ("evaluate_episodes",)
        if not enable_evaluate:
            enabled_ops = tuple(op for op in enabled_ops if op != "evaluate_episodes")

        service = ServiceSettings(
            backend=reader.choice("BACKEND", "local", ("local", "aws")),
            environment=reader.choice("ENV", "dev", ("dev", "prod")),
            allowed_operations=tuple(enabled_ops),
            max_request_bytes=reader.int("MAX_REQUEST_BYTES", 1_048_576, minimum=1024, maximum=25 * 1024 * 1024),
            max_units_per_job=reader.int("MAX_UNITS_PER_JOB", 600, minimum=1),
            job_retention_days=reader.int("JOB_RETENTION_DAYS", 90, minimum=1),
            lease_seconds=reader.int("LEASE_SECONDS", 600, minimum=60, maximum=3600),
            job_max_runtime_seconds=reader.int("JOB_MAX_RUNTIME_SECONDS", 12 * 3600, minimum=600),
            max_episodes_per_request=reader.int("MAX_EPISODES_PER_REQUEST", 50, minimum=1, maximum=1000),
            run_as_uid=reader.int("RUN_AS_UID", 10001, minimum=0),
            web_workers=reader.int("WEB_WORKERS", 2, minimum=1, maximum=8),
            web_threads=reader.int("WEB_THREADS", 4, minimum=1, maximum=32),
            shutdown_grace_seconds=reader.int("SHUTDOWN_GRACE_SECONDS", 20, minimum=1, maximum=25),
        )

        web_cfg = yaml_cfg.get("web_research", {}) or {}
        hosts = reader.csv("WEB_ALLOWED_HOSTS", tuple(web_cfg.get("allowed_hosts", ("www.ihacpa.gov.au", "ihacpa.gov.au"))))
        web = WebResearchSettings(
            enabled=reader.bool("ENABLE_WEB_RESEARCH", False),
            allowed_hosts=tuple(h.strip().lower() for h in hosts if h.strip()),
            default_topics=tuple(str(t) for t in web_cfg.get("default_topics", ())),
        )

        out_cfg = yaml_cfg.get("output", {}) or {}
        footer_cfg = out_cfg.get("excel_footer", {}) or {}
        output = OutputSettings(
            excel_sheet_name=reader.str("EXCEL_SHEET_NAME", str(out_cfg.get("excel_sheet_name", "rule_catalogue"))),
            footer_enabled=reader.bool("EXCEL_FOOTER_ENABLED", bool(footer_cfg.get("enabled", True))),
            footer_lines=tuple(str(x) for x in footer_cfg.get("lines", ())),
        )

        reader.raise_if_errors()
        settings = cls(
            paths=paths, bedrock=bedrock, agent=agent, chunking=chunking, aws=aws, service=service,
            web=web, output=output, grouper_factory=reader.str("GROUPER_FACTORY", "") or None,
            enable_evaluate=enable_evaluate,
            extras={"sources": yaml_cfg.get("sources", {}) or {}},
        )
        return settings

    # ----------------------------------------------------------------- validation
    def validate_for_aws_backend(self) -> None:
        """Fail fast when the SageMaker deployment is missing required resources."""
        missing = []
        if not self.aws.state_table:
            missing.append("STATE_TABLE")
        if not self.aws.work_queue_url:
            missing.append("QUEUE_URL")
        if not self.aws.artifact_bucket:
            missing.append("ARTIFACT_BUCKET")
        if not self.bedrock.region:
            missing.append("AWS_REGION (or RULE_EXTRACTOR_BEDROCK_REGION)")
        if missing:
            raise ConfigError("missing required configuration for the aws backend: " + ", ".join(missing))

    def validate_for_agents(self) -> None:
        if not self.bedrock.region:
            raise ConfigError("an AWS region is required to call Bedrock (set AWS_REGION or RULE_EXTRACTOR_BEDROCK_REGION)")

    def public_summary(self) -> dict[str, Any]:
        """Non-sensitive configuration summary for the health endpoint."""
        return {
            "environment": self.service.environment,
            "backend": self.service.backend,
            "bedrock": {
                "region": self.bedrock.region,
                "provider": self.bedrock.provider,
                "inference_profile_prefix": self.bedrock.inference_profile_prefix,
            },
            "agent": {
                "max_concurrent_sessions": self.agent.max_concurrent_sessions,
                "max_turns": self.agent.max_turns,
                "max_retries": self.agent.max_retries,
            },
            "web_research_enabled": self.web.enabled,
            "evaluate_enabled": self.enable_evaluate,
            "allowed_operations": list(self.service.allowed_operations),
        }


# ---------------------------------------------------------------------- helpers
def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # PyYAML
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError("PyYAML is required to read a config file") from exc
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return data


class _Reader:
    """Typed environment reader that collects every problem before raising."""

    def __init__(self, env: Mapping[str, str], yaml_cfg: Mapping[str, Any]) -> None:
        self._env = env
        self._yaml = yaml_cfg
        self._errors: list[str] = []

    def fail(self, message: str) -> None:
        self._errors.append(message)

    def raise_if_errors(self) -> None:
        if self._errors:
            raise ConfigError("invalid configuration: " + "; ".join(self._errors), details=list(self._errors))

    def _raw(self, name: str) -> str | None:
        value = self._env.get(ENV_PREFIX + name)
        if value is None or value == "":
            return None
        return value

    def str(self, name: str, default: str) -> str:
        raw = self._raw(name)
        return default if raw is None else raw.strip()

    def choice(self, name: str, default: str, allowed: tuple[str, ...]) -> str:
        value = self.str(name, default).lower()
        if value not in allowed:
            self.fail(f"{ENV_PREFIX}{name} must be one of {allowed}, got {value!r}")
            return default
        return value

    def int(self, name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            self.fail(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}")
            return default
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            self.fail(f"{ENV_PREFIX}{name}={value} outside [{minimum}, {maximum}]")
            return default
        return value

    def float(self, name: str, default: float, *, minimum: float | None = None) -> float:
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            self.fail(f"{ENV_PREFIX}{name} must be a number, got {raw!r}")
            return default
        if minimum is not None and value < minimum:
            self.fail(f"{ENV_PREFIX}{name}={value} must be >= {minimum}")
            return default
        return value

    def bool(self, name: str, default: bool) -> bool:
        raw = self._raw(name)
        if raw is None:
            return default
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        self.fail(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")
        return default

    def csv(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = self._raw(name)
        if raw is None:
            return tuple(default)
        return tuple(part.strip() for part in raw.split(",") if part.strip())
