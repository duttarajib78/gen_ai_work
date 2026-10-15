"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SimCSELoss(nn.Module):
    """In-batch InfoNCE over interleaved views ``[clean_0, masked_0, clean_1, masked_1, ...]``.

    Row ``i`` of the clean view must retrieve row ``i`` of the masked view among all masked views in the batch.
    """

    def __init__(self, temperature: float = 0.30) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        clean = F.normalize(embeddings[0::2], p=2, dim=-1)
        masked = F.normalize(embeddings[1::2], p=2, dim=-1)
        similarity = torch.mm(clean, masked.t()) / self.temperature
        labels = torch.arange(similarity.size(0), device=embeddings.device)
        return F.cross_entropy(similarity, labels)
