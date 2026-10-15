"""Transformer encoder: maps a masked-PDx admission context to a single embedding.

Module and parameter names (``roberta``, ``pooler_ln``, ``projector``) are identical to the
prototype so weights trained there load without renaming.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel

from drg_audit.errors import ArtifactError

# Keys that define the architecture and the state-dict shapes. These are what encoder_config.json stores.
ENCODER_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "attention_probs_dropout_prob",
    "hidden_dropout_prob",
    "position_embedding_type",
    "pad_token_id",
    "max_position_embeddings",
    "type_vocab_size",
    "layer_norm_eps",
    "hidden_act",
    "initializer_range",
)
_REQUIRED_KEYS = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                  "intermediate_size", "position_embedding_type", "pad_token_id")


def build_roberta_config(cfg: Mapping[str, Any]) -> RobertaConfig:
    """Build a RobertaConfig from a plain dict. Eager attention is required for relative position embeddings."""
    missing = [key for key in _REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ArtifactError(f"encoder config is missing keys: {missing}")
    kwargs = {key: cfg[key] for key in ENCODER_CONFIG_KEYS if key in cfg}
    return RobertaConfig(**kwargs, attn_implementation="eager")


class ClinicalSimCSEModel(nn.Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.roberta = RobertaModel(config)
        self.pooler_ln = nn.LayerNorm(config.hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projector(self.pooler_ln(cls_output))
