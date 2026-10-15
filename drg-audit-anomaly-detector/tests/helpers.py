"""Shared test helpers. Importing this module puts ``src`` and the SageMaker code dir on ``sys.path``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DRG_AUDIT_LOG_LEVEL", "WARNING")  # keep test output quiet; assertLogs raises the level

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "sagemaker" / "code", ROOT / "sagemaker"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from drg_audit.features import SPECIAL_TOKENS  # noqa: E402
from drg_audit.schemas import InpatientRecord, LabelMaps  # noqa: E402


def make_vocab(codes=()):
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for code in sorted(codes):
        vocab[code] = len(vocab)
    return vocab


def make_labels() -> LabelMaps:
    return LabelMaps(
        version={"10.0": 0, "11.0": 1},
        mdc={"F": 0, "G": 1},
        base={"10": 0, "62": 1},
        split={"A": 0, "B": 1, "C": 2, "D": 3},
    )


def make_record(**overrides) -> InpatientRecord:
    values = dict(
        admission_no="1001",
        diagnosis_code_array=["I21.4", "E11.9", "I10"],
        procedure_code_array=["38497-00"],
        medical_claims_array=["55114", "38200"],
        drg_vers="10.0",
        drg="F62A",
        los=5,
        minutes_operating_theatre=0.0,
        hours_of_mechanical_ventilation=0.0,
        member_age=67.0,
        drg_benefit=1000.0,
    )
    values.update(overrides)
    return InpatientRecord(**values)


def valid_payload_record(**overrides) -> dict:
    record = {
        "admission_no": "1001",
        "diagnosis_code_array": ["I21.4", "E11.9"],
        "procedure_code_array": ["38497-00"],
        "medical_claims_array": ["55114", 38200],
        "drg_vers": "10.0",
        "drg": "F62A",
        "los": 5,
        "minutes_operating_theatre": 0,
        "hours_of_mechanical_ventilation": 0,
        "member_age": 67,
        "drg_benefit": 1250.5,
    }
    record.update(overrides)
    return record
