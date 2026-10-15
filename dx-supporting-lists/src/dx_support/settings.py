"""Runtime configuration.

Everything is read from environment variables prefixed ``DX_`` (plus the standard ``AWS_*`` variables). There are NO credentials here:
the model and the storage are reached with the IAM role of the SageMaker endpoint (or your own AWS identity when you run the CLI).
Nothing secret is ever read from a file, and nothing in this module logs a value.

``Settings.from_env()`` validates eagerly and fails closed: a misconfiguration stops the process at start-up instead of surfacing as
a confusing failure on the first request. Production (``DX_ENV=production``) is stricter than development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional

from dx_support.constants import DEFAULT_THINKING, THINKING_PRESETS

ENVIRONMENTS = ("development", "test", "production")
BACKENDS = ("bedrock", "fake")
PROFILE_PREFIXES = ("au", "apac", "us", "eu", "jp", "global", "none")
MODES = ("routed", "exhaustive")
STAGES = ("brief", "route", "judge")

DEFAULT_MAX_TOKENS = {"brief": 4000, "route": 6000, "judge": 16000}

# Endpoint overrides redirect prompts (and the signed requests) to another host; static keys are what leaks. Both are refused in production.
_ENDPOINT_OVERRIDE_VARS = ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_ENDPOINT_URL_BEDROCK", "AWS_ENDPOINT_URL_S3")
_STATIC_CREDENTIAL_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


class ConfigError(ValueError):
    """The configuration is invalid. The message names the variable, never its value."""


def _get(env: Mapping[str, str], name: str) -> Optional[str]:
    v = env.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    v = _get(env, name)
    if v is None:
        return default
    low = v.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false")


def _num(env: Mapping[str, str], name: str, default, cast, lo=None, hi=None):
    v = _get(env, name)
    if v is None:
        return default
    try:
        out = cast(v)
    except ValueError:
        raise ConfigError(f"{name} must be a {cast.__name__}") from None
    if (lo is not None and out < lo) or (hi is not None and out > hi):
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return out


def _choice(env: Mapping[str, str], name: str, default: str, choices: tuple) -> str:
    v = (_get(env, name) or default).lower()
    if v not in choices:
        raise ConfigError(f"{name} must be one of {list(choices)}")
    return v


@dataclass(frozen=True)
class Settings:
    # ---- environment / backend ----
    env: str = "development"
    backend: str = "bedrock"
    aws_region: str = "ap-southeast-2"
    # ---- models ----
    model: str = "claude-sonnet-5"
    brief_model: Optional[str] = None
    route_model: Optional[str] = None
    judge_model: Optional[str] = None
    inference_profile_prefix: str = "au"
    model_catalog_path: Optional[Path] = None
    thinking: str = DEFAULT_THINKING
    max_tokens: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_TOKENS))
    prompt_caching: bool = False
    guardrail_id: Optional[str] = None
    guardrail_version: Optional[str] = None
    # ---- call control ----
    llm_concurrency: int = 16
    llm_call_timeout_s: float = 240.0
    max_call_retries: int = 3
    max_throttle_retries: int = 6
    max_budget_usd_per_call: float = 0.50
    max_cost_usd_per_category: float = 3.00
    max_cost_usd_per_hour: float = 100.00
    # ---- data / storage ----
    data_dir: Path = Path("data")
    store_uri: str = ""
    store_kms_key_id: Optional[str] = None
    # ---- pipeline ----
    mode: str = "routed"
    lexical_rescue: bool = True
    include_coarse_prostheses: bool = False
    include_context_chapters: bool = False
    cache_salt: str = ""
    # ---- service ----
    max_categories_per_request: int = 3
    max_concurrent_requests: int = 8
    max_request_bytes: int = 64 * 1024
    sync_deadline_s: float = 50.0
    stream_heartbeat_s: float = 5.0
    stream_max_s: float = 450.0
    allow_compute: bool = True
    # ---- logging / output ----
    log_level: str = "INFO"
    log_json: bool = True
    log_calls: bool = False
    notice_line_1: str = "AI-assisted screening output: a verdict is a screening signal, not a coding determination."
    notice_line_2: str = "A qualified clinical coder must review these lists before any production use."

    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def stage_models(self) -> dict[str, str]:
        return {"brief": self.brief_model or self.model, "route": self.route_model or self.model, "judge": self.judge_model or self.model}

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *, cwd: Optional[Path] = None) -> "Settings":
        env = os.environ if env is None else env
        cwd = cwd or Path.cwd()
        deploy_env = _choice(env, "DX_ENV", "development", ENVIRONMENTS)
        data_dir = Path(_get(env, "DX_DATA_DIR") or (cwd / "data")).expanduser()
        store_uri = _get(env, "DX_STORE_URI") or ("" if deploy_env == "production" else (cwd / "output").resolve().as_uri())
        max_tokens = {s: _num(env, f"DX_MAX_OUTPUT_TOKENS_{s.upper()}", DEFAULT_MAX_TOKENS[s], int, 256, 128_000) for s in STAGES}
        model_catalog = _get(env, "DX_MODEL_CATALOG")
        s = cls(
            env=deploy_env,
            backend=_choice(env, "DX_BACKEND", "bedrock", BACKENDS),
            aws_region=_get(env, "DX_AWS_REGION") or _get(env, "AWS_REGION") or _get(env, "AWS_DEFAULT_REGION") or "ap-southeast-2",
            model=_get(env, "DX_MODEL") or "claude-sonnet-5",
            brief_model=_get(env, "DX_BRIEF_MODEL"), route_model=_get(env, "DX_ROUTE_MODEL"), judge_model=_get(env, "DX_JUDGE_MODEL"),
            inference_profile_prefix=_choice(env, "DX_INFERENCE_PROFILE_PREFIX", "au", PROFILE_PREFIXES),
            model_catalog_path=Path(model_catalog).expanduser() if model_catalog else None,
            thinking=_choice(env, "DX_THINKING", DEFAULT_THINKING, THINKING_PRESETS),
            max_tokens=max_tokens,
            prompt_caching=_bool(env, "DX_PROMPT_CACHING", False),
            guardrail_id=_get(env, "DX_GUARDRAIL_ID"), guardrail_version=_get(env, "DX_GUARDRAIL_VERSION"),
            llm_concurrency=_num(env, "DX_LLM_CONCURRENCY", 16, int, 1, 256),
            llm_call_timeout_s=_num(env, "DX_LLM_CALL_TIMEOUT_S", 240.0, float, 10, 900),
            max_call_retries=_num(env, "DX_MAX_CALL_RETRIES", 3, int, 0, 10),
            max_throttle_retries=_num(env, "DX_MAX_THROTTLE_RETRIES", 6, int, 0, 20),
            max_budget_usd_per_call=_num(env, "DX_MAX_BUDGET_USD_PER_CALL", 0.50, float, 0.001, 100),
            max_cost_usd_per_category=_num(env, "DX_MAX_COST_USD_PER_CATEGORY", 3.00, float, 0.01, 1000),
            max_cost_usd_per_hour=_num(env, "DX_MAX_COST_USD_PER_HOUR", 100.00, float, 0.01, 1_000_000),
            data_dir=data_dir, store_uri=store_uri, store_kms_key_id=_get(env, "DX_STORE_KMS_KEY_ID"),
            mode=_choice(env, "DX_MODE", "routed", MODES),
            lexical_rescue=_bool(env, "DX_LEXICAL_RESCUE", True),
            include_coarse_prostheses=_bool(env, "DX_INCLUDE_COARSE_PROSTHESES", False),
            include_context_chapters=_bool(env, "DX_INCLUDE_CONTEXT_CHAPTERS", False),
            cache_salt=_get(env, "DX_CACHE_SALT") or "",
            max_categories_per_request=_num(env, "DX_MAX_CATEGORIES_PER_REQUEST", 3, int, 1, 50),
            max_concurrent_requests=_num(env, "DX_MAX_CONCURRENT_REQUESTS", 8, int, 1, 128),
            max_request_bytes=_num(env, "DX_MAX_REQUEST_BYTES", 64 * 1024, int, 1024, 1024 * 1024),
            sync_deadline_s=_num(env, "DX_SYNC_DEADLINE_S", 50.0, float, 1, 58),
            stream_heartbeat_s=_num(env, "DX_STREAM_HEARTBEAT_S", 5.0, float, 0.2, 30),
            stream_max_s=_num(env, "DX_STREAM_MAX_S", 450.0, float, 10, 475),
            allow_compute=_bool(env, "DX_ALLOW_COMPUTE", True),
            log_level=(_get(env, "DX_LOG_LEVEL") or "INFO").upper(),
            log_json=_bool(env, "DX_LOG_JSON", True), log_calls=_bool(env, "DX_LOG_CALLS", False),
            notice_line_1=_get(env, "DX_OUTPUT_NOTICE_LINE_1") or cls.notice_line_1,
            notice_line_2=_get(env, "DX_OUTPUT_NOTICE_LINE_2") or cls.notice_line_2,
        )
        s.validate(env)
        return s

    # ------------------------------------------------------------------
    def validate(self, env: Optional[Mapping[str, str]] = None) -> None:
        """Cross-field and production rules. Raises ConfigError."""
        env = os.environ if env is None else env
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigError("DX_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
        if bool(self.guardrail_id) != bool(self.guardrail_version):
            raise ConfigError("DX_GUARDRAIL_ID and DX_GUARDRAIL_VERSION must be set together")
        if self.stream_max_s <= self.sync_deadline_s:
            raise ConfigError("DX_STREAM_MAX_S must be larger than DX_SYNC_DEADLINE_S")
        scheme = self.store_uri.split("://", 1)[0] if "://" in self.store_uri else ("file" if self.store_uri else "")
        if self.store_uri and scheme not in ("file", "s3"):
            raise ConfigError("DX_STORE_URI must be s3://bucket/prefix or file:///path")
        if self.is_production:
            if self.backend != "bedrock":
                raise ConfigError("DX_BACKEND must be 'bedrock' in production: the stub backend produces non-clinical verdicts")
            if scheme != "s3" and not _bool(env, "DX_ALLOW_LOCAL_STORE", False):
                raise ConfigError("DX_STORE_URI must be an s3:// URI in production (a local store is lost when an instance is replaced "
                                  "and is not shared between instances); set DX_ALLOW_LOCAL_STORE=true only for a single-instance trial")
            leaked = [n for n in _STATIC_CREDENTIAL_VARS if env.get(n)]
            if leaked and not _bool(env, "DX_ALLOW_STATIC_CREDENTIALS", False):
                raise ConfigError(f"Static AWS credentials are set in the environment ({leaked}); production uses the endpoint's IAM role only. "
                                  "Remove them (set DX_ALLOW_STATIC_CREDENTIALS=true only if you accept the risk)")
            redirected = [n for n in _ENDPOINT_OVERRIDE_VARS if env.get(n)]
            if redirected and not _bool(env, "DX_ALLOW_ENDPOINT_OVERRIDE", False):
                raise ConfigError(f"AWS endpoint override variables are set ({redirected}); they can redirect prompts and signed requests "
                                  "to another host. Remove them or set DX_ALLOW_ENDPOINT_OVERRIDE=true if you deliberately use a private gateway")


def environment_warnings(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Human-readable warnings about the surrounding environment. Variable NAMES only, never values."""
    env = os.environ if env is None else env
    out: list[str] = []
    redirected = [n for n in _ENDPOINT_OVERRIDE_VARS if env.get(n)]
    if redirected:
        out.append(f"AWS endpoint override variables are set ({redirected}): requests may not go to the AWS service endpoints you expect.")
    leaked = [n for n in _STATIC_CREDENTIAL_VARS if env.get(n)]
    if leaked:
        out.append(f"Static AWS credentials are present in the environment ({leaked}). Prefer role-based credentials and rotate any key that has been shared.")
    return out
