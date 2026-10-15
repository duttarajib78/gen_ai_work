"""Feature engineering shared by training, batch scoring and the real-time endpoint.

This module is the single source of truth for turning an ``InpatientRecord`` into model inputs.
The prototype duplicated this logic in three scripts, which had drifted apart (medical-claim
ordering and truncation differed between training and inference). Depends only on the standard
library and NumPy so it can be tested without PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from drg_audit.schemas import InpatientRecord, TokenizationSpec

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, CLS, SEP, MASK)
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
META_DIM = 4
DEFAULT_SPLIT = "D"
DEFAULT_AGE = 50.0


def normalize_code(value: object) -> str:
    """Canonical string form of a clinical or claim code (whitespace trimmed; case is preserved)."""
    return str(value).strip()


def record_tokens(record: InpatientRecord, spec: TokenizationSpec) -> list[str]:
    """Token strings for one record. The principal diagnosis is replaced by ``[MASK]`` so the model
    must infer the DRG from secondary diagnoses, procedures and claims (no label leakage)."""
    secondary = sorted(normalize_code(c) for c in record.diagnosis_code_array[1:])
    procedures = sorted(normalize_code(c) for c in record.procedure_code_array)
    claims = [normalize_code(c) for c in record.medical_claims_array]
    if spec.sort_medical_claims:
        claims.sort()
    return [MASK, *secondary, *procedures, *claims]


def encode_tokens(tokens: Sequence[str], vocab: Mapping[str, int], spec: TokenizationSpec) -> list[int]:
    """``[CLS] tokens [SEP]`` as vocabulary ids, truncated to ``spec.max_len``."""
    unk = vocab[UNK]
    if spec.keep_sep_on_truncation:
        body = [vocab.get(t, unk) for t in tokens[: spec.max_len - 2]]
        return [vocab[CLS], *body, vocab[SEP]]
    full = [vocab[CLS], *(vocab.get(t, unk) for t in tokens), vocab[SEP]]
    return full[: spec.max_len]


def stay_type_bucket(los: float) -> float:
    """Ordinal length-of-stay bucket. Legacy behaviour is preserved: LOS == 1 maps to 0.0 and any
    LOS below 1 (including 0) falls into the 0.25 bucket. Changing this requires retraining."""
    if los == 1:
        return 0.0
    if los <= 3:
        return 0.25
    if los <= 7:
        return 0.50
    if los <= 14:
        return 0.75
    return 1.0


def _non_negative(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else 0.0


def meta_features(record: InpatientRecord) -> list[float]:
    age = float(record.member_age)
    age = age if math.isfinite(age) else DEFAULT_AGE
    return [
        stay_type_bucket(float(record.los)),
        math.log1p(_non_negative(record.minutes_operating_theatre)),
        math.log1p(_non_negative(record.hours_of_mechanical_ventilation)),
        age / 100.0,
    ]


def split_drg(drg: str | None) -> tuple[str, str, str] | None:
    """Split an AR-DRG such as ``F62A`` into (MDC ``F``, base ``62``, split ``A``).
    Returns ``None`` when the code is too short to carry an MDC and base DRG."""
    code = (drg or "").strip()
    if len(code) < 3:
        return None
    return code[0], code[1:3], code[3] if len(code) > 3 else DEFAULT_SPLIT


@dataclass(frozen=True)
class EncodedBatch:
    """Model-ready numpy arrays for a batch of records."""

    input_ids: np.ndarray        # [B, L] integer ids
    meta: np.ndarray             # [B, META_DIM] float32
    version_ids: np.ndarray      # [B] int64
    unknown_version: np.ndarray  # [B] bool, True where the DRG version was not seen in training
    pad_id: int

    @property
    def attention_mask(self) -> np.ndarray:
        return (self.input_ids != self.pad_id).astype(np.int64)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def encode_records(
    records: Sequence[InpatientRecord],
    vocab: Mapping[str, int],
    version_map: Mapping[str, int],
    spec: TokenizationSpec,
    *,
    pad_to_longest: bool = False,
    id_dtype: type = np.int64,
) -> EncodedBatch:
    """Encode records. ``pad_to_longest`` trims padding to the longest sequence in the batch, which is
    numerically equivalent (padding is masked) but much faster for short admissions."""
    n = len(records)
    sequences = [encode_tokens(record_tokens(r, spec), vocab, spec) for r in records]
    width = max((len(s) for s in sequences), default=2) if pad_to_longest else spec.max_len
    pad_id = vocab[PAD]
    input_ids = np.full((n, width), pad_id, dtype=id_dtype)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = seq
    meta = np.asarray([meta_features(r) for r in records], dtype=np.float32).reshape(n, META_DIM)
    versions = [str(r.drg_vers) for r in records]
    unknown = np.fromiter((v not in version_map for v in versions), dtype=bool, count=n)
    version_ids = np.fromiter((version_map.get(v, 0) for v in versions), dtype=np.int64, count=n)
    return EncodedBatch(input_ids, meta, version_ids, unknown, pad_id)
