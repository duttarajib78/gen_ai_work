"""In-process scoring engine shared by the SageMaker endpoint, offline batch scoring and tests."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from drg_audit.artifacts import ModelBundle, load_bundle
from drg_audit.errors import ArtifactError, ConfigError
from drg_audit.features import CLS, MASK, META_DIM, PAD, SEP, EncodedBatch, encode_records, split_drg
from drg_audit.inference.decision import InverseLabels, decide
from drg_audit.models.joint import DrgAuditModel
from drg_audit.schemas import AuditResult, InpatientRecord
from drg_audit.settings import RuntimeSettings, ServingConfig

_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ConfigError("runtime.device is 'cuda' but no CUDA device is available")
    return torch.device(name)


class AuditEngine:
    """Loads a verified bundle once and scores batches of ``InpatientRecord`` objects.

    Inference is deterministic (eval mode, no dropout, no random masking) and independent of batch
    composition. Padding is trimmed to the longest sequence per batch, which is numerically equivalent
    because padded positions are masked.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        settings: RuntimeSettings | None = None,
        default_threshold: float = 0.85,
    ) -> None:
        self.settings = settings or RuntimeSettings()
        self.settings.validate()
        self.bundle = bundle
        self.model_version = bundle.model_version
        self.default_threshold = float(default_threshold)
        self.device = resolve_device(self.settings.device)
        self._autocast_dtype = self._resolve_precision()
        if self.settings.torch_threads:
            torch.set_num_threads(self.settings.torch_threads)
        self.model = DrgAuditModel.from_bundle(bundle).to(self.device).eval()
        self.inverse = InverseLabels.from_label_maps(bundle.labels)
        self._self_check()

    @classmethod
    def from_bundle_dir(
        cls,
        path: str | Path,
        settings: RuntimeSettings | None = None,
        default_threshold: float | None = None,
    ) -> AuditEngine:
        bundle = load_bundle(path, verify=True)
        serving = ServingConfig.from_mapping(bundle.inference_config)
        return cls(
            bundle,
            settings=settings or serving.runtime,
            default_threshold=serving.decision.confidence_threshold
            if default_threshold is None
            else default_threshold,
        )

    # ------------------------------------------------------------------------------------
    def predict(
        self, records: Sequence[InpatientRecord], confidence_threshold: float | None = None
    ) -> list[AuditResult]:
        threshold = self.default_threshold if confidence_threshold is None else float(confidence_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")

        results: list[AuditResult | None] = [None] * len(records)
        pending: list[tuple[int, InpatientRecord, tuple[str, str, str]]] = []
        reject_unknown = self.settings.unknown_version_policy == "reject"
        for i, record in enumerate(records):
            parts = split_drg(record.drg)
            if parts is None:
                results[i] = AuditResult.skipped(record.admission_no, "invalid_assigned_drg",
                                                 assigned_drg=(record.drg or "").strip() or None)
            elif reject_unknown and str(record.drg_vers) not in self.bundle.labels.version:
                results[i] = AuditResult.skipped(record.admission_no, "unknown_drg_version",
                                                 assigned_drg=record.drg.strip())
            else:
                pending.append((i, record, parts))

        step = self.settings.max_batch_size
        for start in range(0, len(pending), step):
            chunk = pending[start : start + step]
            batch = encode_records(
                [record for _, record, _ in chunk],
                self.bundle.vocab,
                self.bundle.labels.version,
                self.bundle.tokenization,
                pad_to_longest=self.settings.dynamic_padding,
            )
            mdc_probs, base_probs, split_probs = self._probabilities(batch)
            for k, (i, record, parts) in enumerate(chunk):
                results[i] = decide(
                    admission_no=record.admission_no,
                    assigned_drg=record.drg.strip(),
                    assigned_parts=parts,
                    benefit=record.drg_benefit,
                    mdc_probs=mdc_probs[k],
                    base_probs=base_probs[k],
                    split_probs=split_probs[k],
                    inverse=self.inverse,
                    split_index=self.bundle.labels.split,
                    threshold=threshold,
                    warnings=["unknown_drg_version_fallback"] if batch.unknown_version[k] else [],
                )
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------------------------
    @torch.inference_mode()
    def _probabilities(self, batch: EncodedBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = torch.from_numpy(batch.input_ids).long().to(self.device)
        mask = torch.from_numpy(batch.attention_mask).long().to(self.device)
        meta = torch.from_numpy(batch.meta).float().to(self.device)
        versions = torch.from_numpy(batch.version_ids).long().to(self.device)
        with self._autocast():
            out = self.model(ids, mask, meta, versions)
        return tuple(  # type: ignore[return-value]
            torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for logits in (out.mdc_logits, out.base_logits, out.split_logits)
        )

    def _autocast(self):
        if self._autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype)

    def _resolve_precision(self) -> torch.dtype | None:
        precision = self.settings.precision
        if precision == "fp32":
            return None
        if precision == "fp16" and self.device.type != "cuda":
            raise ConfigError("runtime.precision 'fp16' requires a CUDA device")
        return _AUTOCAST_DTYPES[precision]

    def _self_check(self) -> None:
        """Fail fast at load time if the model produces non-finite or non-normalised probabilities."""
        vocab = self.bundle.vocab
        ids = np.full((2, 8), vocab[PAD], dtype=np.int64)
        ids[:, 0], ids[:, 1], ids[:, 2] = vocab[CLS], vocab[MASK], vocab[SEP]
        batch = EncodedBatch(ids, np.zeros((2, META_DIM), np.float32), np.zeros(2, np.int64),
                             np.zeros(2, bool), vocab[PAD])
        for probs in self._probabilities(batch):
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-3):
                raise ArtifactError("model self-check failed: non-finite or non-normalised probabilities")
