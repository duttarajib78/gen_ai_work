"""Serving-time settings. Defaults ship inside the model bundle (``inference_config.json``);
selected values can be overridden with ``DRG_AUDIT_*`` environment variables on the SageMaker model."""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from drg_audit._mapping import dataclass_from_mapping
from drg_audit.errors import ConfigError

VALID_DEVICES = ("auto", "cpu", "cuda")
VALID_PRECISIONS = ("fp32", "fp16", "bf16")
VALID_VERSION_POLICIES = ("fallback", "reject")


@dataclass(frozen=True)
class RequestLimits:
    """Hard bounds applied to every request. Anything outside is rejected, never clamped."""

    max_payload_bytes: int = 8 * 1024 * 1024
    max_records: int = 256
    max_codes_per_array: int = 512
    max_code_length: int = 32
    max_los_days: int = 3650
    max_age_years: float = 130.0
    max_theatre_minutes: float = 100_000.0
    max_ventilation_hours: float = 8_760.0
    max_abs_benefit: float = 1.0e9
    min_confidence_threshold: float = 0.50

    def validate(self) -> None:
        if self.max_payload_bytes < 1024 or self.max_records < 1 or self.max_codes_per_array < 1:
            raise ConfigError("limits: payload, record and code-array bounds must be positive")
        if not 0.0 < self.min_confidence_threshold < 1.0:
            raise ConfigError("limits.min_confidence_threshold must be in (0, 1)")


@dataclass(frozen=True)
class DecisionConfig:
    confidence_threshold: float = 0.85

    def validate(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ConfigError("decision.confidence_threshold must be in (0, 1]")


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "auto"
    precision: str = "fp32"
    max_batch_size: int = 64
    torch_threads: int | None = None
    dynamic_padding: bool = True
    unknown_version_policy: str = "fallback"

    def validate(self) -> None:
        if self.device not in VALID_DEVICES:
            raise ConfigError(f"runtime.device must be one of {VALID_DEVICES}")
        if self.precision not in VALID_PRECISIONS:
            raise ConfigError(f"runtime.precision must be one of {VALID_PRECISIONS}")
        if self.unknown_version_policy not in VALID_VERSION_POLICIES:
            raise ConfigError(f"runtime.unknown_version_policy must be one of {VALID_VERSION_POLICIES}")
        if not 1 <= self.max_batch_size <= 4096:
            raise ConfigError("runtime.max_batch_size must be between 1 and 4096")
        if self.torch_threads is not None and self.torch_threads < 1:
            raise ConfigError("runtime.torch_threads must be >= 1")

    def with_env(self, env: Mapping[str, str] | None = None) -> RuntimeSettings:
        env = os.environ if env is None else env
        updates: dict[str, Any] = {}
        if "DRG_AUDIT_DEVICE" in env:
            updates["device"] = env["DRG_AUDIT_DEVICE"].strip().lower()
        if "DRG_AUDIT_PRECISION" in env:
            updates["precision"] = env["DRG_AUDIT_PRECISION"].strip().lower()
        if "DRG_AUDIT_MAX_BATCH_SIZE" in env:
            updates["max_batch_size"] = _env_int(env, "DRG_AUDIT_MAX_BATCH_SIZE")
        if "DRG_AUDIT_TORCH_THREADS" in env:
            updates["torch_threads"] = _env_int(env, "DRG_AUDIT_TORCH_THREADS")
        if "DRG_AUDIT_DYNAMIC_PADDING" in env:
            updates["dynamic_padding"] = env["DRG_AUDIT_DYNAMIC_PADDING"].strip().lower() in {"1", "true", "yes"}
        if "DRG_AUDIT_UNKNOWN_VERSION_POLICY" in env:
            updates["unknown_version_policy"] = env["DRG_AUDIT_UNKNOWN_VERSION_POLICY"].strip().lower()
        merged = dataclasses.replace(self, **updates)
        merged.validate()
        return merged


@dataclass(frozen=True)
class ServingConfig:
    """Contents of ``inference_config.json`` inside a model bundle."""

    decision: DecisionConfig = field(default_factory=DecisionConfig)
    limits: RequestLimits = field(default_factory=RequestLimits)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ServingConfig:
        return dataclass_from_mapping(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def with_env(self, env: Mapping[str, str] | None = None) -> ServingConfig:
        """Apply the ``DRG_AUDIT_*`` environment overrides and re-validate."""
        env = os.environ if env is None else env
        limits = self.limits
        updates: dict[str, Any] = {}
        if "DRG_AUDIT_MAX_PAYLOAD_BYTES" in env:
            updates["max_payload_bytes"] = _env_int(env, "DRG_AUDIT_MAX_PAYLOAD_BYTES")
        if "DRG_AUDIT_MAX_RECORDS" in env:
            updates["max_records"] = _env_int(env, "DRG_AUDIT_MAX_RECORDS")
        limits = dataclasses.replace(limits, **updates)
        limits.validate()
        decision = self.decision
        if "DRG_AUDIT_CONFIDENCE_THRESHOLD" in env:
            decision = DecisionConfig(confidence_threshold=_env_float(env, "DRG_AUDIT_CONFIDENCE_THRESHOLD"))
            decision.validate()
        return ServingConfig(decision=decision, limits=limits, runtime=self.runtime.with_env(env))


def _env_int(env: Mapping[str, str], key: str) -> int:
    try:
        return int(env[key])
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer") from exc


def _env_float(env: Mapping[str, str], key: str) -> float:
    try:
        value = float(env[key])
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{key} must be finite")
    return value
