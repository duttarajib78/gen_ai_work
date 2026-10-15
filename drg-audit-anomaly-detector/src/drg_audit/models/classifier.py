"""Cascaded AR-DRG classifier: MDC -> base DRG (conditioned on MDC) -> complexity split (conditioned on base).

Parameter names are identical to the prototype so trained weights load unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from drg_audit.features import META_DIM

VERSION_EMBEDDING_DIM = 16


class CascadedARDRGClassifier(nn.Module):
    def __init__(
        self,
        num_versions: int,
        num_mdcs: int,
        num_bases: int,
        num_splits: int,
        emb_dim: int = 256,
        meta_dim: int = META_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.version_embedding = nn.Embedding(num_versions, VERSION_EMBEDDING_DIM)
        combined_input_dim = emb_dim + meta_dim + VERSION_EMBEDDING_DIM

        self.shared_network = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.mdc_head = nn.Linear(hidden_dim, num_mdcs)
        self.base_head = nn.Linear(hidden_dim + num_mdcs, num_bases)
        self.split_head = nn.Linear(hidden_dim + num_bases, num_splits)

    def forward(
        self, embedding: torch.Tensor, meta: torch.Tensor, version_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_emb = self.version_embedding(version_id)
        shared = self.shared_network(torch.cat([embedding, meta, v_emb], dim=-1))

        mdc_logits = self.mdc_head(shared)
        base_logits = self.base_head(torch.cat([shared, F.softmax(mdc_logits, dim=-1)], dim=-1))
        split_logits = self.split_head(torch.cat([shared, F.softmax(base_logits, dim=-1)], dim=-1))
        return mdc_logits, base_logits, split_logits
