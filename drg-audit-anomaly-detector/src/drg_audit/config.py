"""Typed configuration for the two training stages. Defaults equal the prototype's hyper-parameters.

YAML is read with ``yaml.safe_load`` only. Unknown keys are rejected (see ``_mapping``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from drg_audit._mapping import dataclass_from_mapping
from drg_audit.errors import ConfigError
from drg_audit.schemas import TokenizationSpec
from drg_audit.settings import ServingConfig

T = TypeVar("T")


@dataclass(frozen=True)
class EncoderSpec:
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    attention_probs_dropout_prob: float = 0.20
    hidden_dropout_prob: float = 0.20
    position_embedding_type: str = "relative_key_query"
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    initializer_range: float = 0.02

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ConfigError("encoder.hidden_size must be divisible by encoder.num_attention_heads")

    def to_encoder_config(self, vocab_size: int, pad_token_id: int) -> dict[str, Any]:
        return {**asdict(self), "vocab_size": vocab_size, "pad_token_id": pad_token_id}


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    require_gpu: bool = True      # training the full model on CPU is impractically slow; fail fast
    amp: bool = True              # fp16 autocast on CUDA
    num_workers: int = 4
    checkpoint_dir: str | None = None  # set to /opt/ml/checkpoints on SageMaker for spot-safe resume


@dataclass(frozen=True)
class PretrainConfig:
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    tokenization: TokenizationSpec = field(default_factory=TokenizationSpec)
    min_code_frequency: int = 10
    val_fraction: float = 0.10
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    temperature: float = 0.30
    mask_probability: float = 0.15
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    uniformity_sample_size: int = 5000
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        _check_common(self)


@dataclass(frozen=True)
class LossWeights:
    contrastive: float = 0.5
    mdc: float = 1.0
    base: float = 1.2
    split: float = 0.8


@dataclass(frozen=True)
class FinetuneConfig:
    val_fraction: float = 0.10
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 100
    encoder_learning_rate: float = 2e-5
    classifier_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    temperature: float = 0.30
    mask_probability: float = 0.15
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 6
    classifier_hidden_dim: int = 128
    loss_weights: LossWeights = field(default_factory=LossWeights)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def validate(self) -> None:
        _check_common(self)
        if self.classifier_hidden_dim < 8:
            raise ConfigError("classifier_hidden_dim must be >= 8")


def _check_common(cfg: Any) -> None:
    if not 0.0 < cfg.val_fraction < 0.5:
        raise ConfigError("val_fraction must be in (0, 0.5)")
    if cfg.batch_size < 2 or cfg.eval_batch_size < 2:
        raise ConfigError("batch sizes must be >= 2 (contrastive loss needs in-batch negatives)")
    if cfg.epochs < 1 or cfg.early_stopping_patience < 1:
        raise ConfigError("epochs and early_stopping_patience must be >= 1")
    if not 0.0 <= cfg.mask_probability < 1.0 or not 0.0 <= cfg.warmup_ratio < 1.0:
        raise ConfigError("mask_probability and warmup_ratio must be in [0, 1)")
    if cfg.temperature <= 0.0 or cfg.grad_clip_norm <= 0.0:
        raise ConfigError("temperature and grad_clip_norm must be positive")


# ------------------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml  # PyYAML is a training / ops dependency, not needed by the endpoint

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML value must be a mapping")
    return data


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides. Values are parsed as YAML scalars (``0.5``, ``true``, ``null``)."""
    import yaml

    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"override '{item}' must look like dotted.key=value")
        target = data
        *parents, leaf = key.split(".")
        for part in parents:
            nxt = target.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"override '{key}': '{part}' is not a section")
            target = nxt
        target[leaf] = yaml.safe_load(raw)
    return data


def load_config(cls: type[T], path: str | Path | None, overrides: Sequence[str] = ()) -> T:
    data = load_yaml(path) if path else {}
    return dataclass_from_mapping(cls, apply_overrides(data, overrides))


def to_mapping(cfg: Any, *, exclude: Sequence[str] = ()) -> Mapping[str, Any]:
    """JSON-serialisable view of a config (used for provenance in the manifest)."""
    return {k: v for k, v in asdict(cfg).items() if k not in exclude}
