"""Encoder + cascaded classifier as one module, with safe (pickle-free) weight export and import."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import nn

from drg_audit.artifacts import ModelBundle, load_tensors
from drg_audit.errors import ArtifactError
from drg_audit.models.classifier import CascadedARDRGClassifier
from drg_audit.models.encoder import ClinicalSimCSEModel, build_roberta_config

ENCODER_PREFIX = "encoder."
CLASSIFIER_PREFIX = "classifier."
# Buffers that older transformers versions persisted in checkpoints but newer ones recreate on init.
_IGNORED_ENCODER_SUFFIXES = ("embeddings.position_ids", "embeddings.token_type_ids")


def filter_encoder_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop non-persistent buffers that some transformers versions wrote into checkpoints."""
    return {k: v for k, v in state.items() if not k.endswith(_IGNORED_ENCODER_SUFFIXES)}


class ModelOutput(NamedTuple):
    embeddings: torch.Tensor
    mdc_logits: torch.Tensor
    base_logits: torch.Tensor
    split_logits: torch.Tensor


class DrgAuditModel(nn.Module):
    def __init__(self, encoder: ClinicalSimCSEModel, classifier: CascadedARDRGClassifier) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta: torch.Tensor,
        version_ids: torch.Tensor,
    ) -> ModelOutput:
        embeddings = self.encoder(input_ids, attention_mask)
        mdc_logits, base_logits, split_logits = self.classifier(embeddings, meta, version_ids)
        return ModelOutput(embeddings, mdc_logits, base_logits, split_logits)

    # ---- construction ---------------------------------------------------------------
    @classmethod
    def build(
        cls,
        encoder_config: Mapping[str, Any],
        label_sizes: Mapping[str, int],
        classifier_hidden_dim: int,
    ) -> DrgAuditModel:
        config = build_roberta_config(encoder_config)
        encoder = ClinicalSimCSEModel(config)
        classifier = CascadedARDRGClassifier(
            num_versions=label_sizes["version"],
            num_mdcs=label_sizes["mdc"],
            num_bases=label_sizes["base"],
            num_splits=label_sizes["split"],
            emb_dim=config.hidden_size,
            hidden_dim=classifier_hidden_dim,
        )
        return cls(encoder, classifier)

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> DrgAuditModel:
        model = cls.build(bundle.encoder_config, bundle.labels.sizes(), int(bundle.classifier_config["hidden_dim"]))
        model.load_exported(load_tensors(bundle.weights_path))
        return model

    # ---- (de)serialization ----------------------------------------------------------
    def export_tensors(self) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for prefix, module in ((ENCODER_PREFIX, self.encoder), (CLASSIFIER_PREFIX, self.classifier)):
            for key, value in module.state_dict().items():
                tensors[prefix + key] = value.detach().cpu().contiguous()
        return tensors

    def load_exported(self, tensors: Mapping[str, torch.Tensor]) -> None:
        encoder_state: dict[str, torch.Tensor] = {}
        classifier_state: dict[str, torch.Tensor] = {}
        for key, value in tensors.items():
            if key.startswith(ENCODER_PREFIX):
                encoder_state[key[len(ENCODER_PREFIX):]] = value
            elif key.startswith(CLASSIFIER_PREFIX):
                classifier_state[key[len(CLASSIFIER_PREFIX):]] = value
            else:
                raise ArtifactError("weights file contains tensors outside the encoder./classifier. namespaces")
        try:
            self.encoder.load_state_dict(filter_encoder_state(encoder_state), strict=True)
            self.classifier.load_state_dict(classifier_state, strict=True)
        except RuntimeError as exc:
            raise ArtifactError("weights do not match the architecture declared in the bundle") from exc
