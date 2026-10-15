"""Anomaly decision rules (framework independent; NumPy only).

A record is a *billing anomaly* when the model confidently disagrees with the assigned DRG:
  * MDC discrepancy      predicted MDC != assigned MDC            and  p(pred MDC)  > threshold
  * base-DRG discrepancy predicted base != assigned base          and  p(pred base) > threshold
  * split inflation      predicted split ranks BELOW the assigned and  p(pred split)> threshold
    (A is the most complex split; a lower predicted complexity suggests the assigned DRG is inflated)

``audit_priority_score = drg_benefit * (1 - p(assigned split))`` for anomalies, else 0.
These semantics are identical to the prototype's batch inference script.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from drg_audit.schemas import AuditResult, LabelMaps

SPLIT_RANK: dict[str, int] = {"A": 3, "B": 2, "C": 1, "D": 0}


@dataclass(frozen=True)
class InverseLabels:
    """Index -> label lookups for the three classification heads."""

    mdc: tuple[str, ...]
    base: tuple[str, ...]
    split: tuple[str, ...]

    @classmethod
    def from_label_maps(cls, labels: LabelMaps) -> InverseLabels:
        def invert(mapping: Mapping[str, int]) -> tuple[str, ...]:
            ordered = [""] * len(mapping)
            for label, idx in mapping.items():
                ordered[idx] = label
            return tuple(ordered)

        return cls(invert(labels.mdc), invert(labels.base), invert(labels.split))


def decide(
    *,
    admission_no: str,
    assigned_drg: str,
    assigned_parts: tuple[str, str, str],
    benefit: float,
    mdc_probs: Sequence[float] | np.ndarray,
    base_probs: Sequence[float] | np.ndarray,
    split_probs: Sequence[float] | np.ndarray,
    inverse: InverseLabels,
    split_index: Mapping[str, int],
    threshold: float,
    warnings: Sequence[str] = (),
) -> AuditResult:
    assigned_mdc, assigned_base, assigned_split = assigned_parts

    m_idx, b_idx, s_idx = int(np.argmax(mdc_probs)), int(np.argmax(base_probs)), int(np.argmax(split_probs))
    m_conf, b_conf, s_conf = float(mdc_probs[m_idx]), float(base_probs[b_idx]), float(split_probs[s_idx])
    pred_mdc, pred_base, pred_split = inverse.mdc[m_idx], inverse.base[b_idx], inverse.split[s_idx]

    mdc_discrepancy = pred_mdc != assigned_mdc and m_conf > threshold
    base_discrepancy = pred_base != assigned_base and b_conf > threshold
    split_inflation = (
        SPLIT_RANK.get(pred_split, 0) < SPLIT_RANK.get(assigned_split, 0) and s_conf > threshold
    )

    assigned_split_idx = split_index.get(assigned_split)
    prob_assigned_split = float(split_probs[assigned_split_idx]) if assigned_split_idx is not None else 0.0

    is_anomaly = bool(mdc_discrepancy or base_discrepancy or split_inflation)
    safe_benefit = float(benefit) if math.isfinite(benefit) else 0.0
    score = safe_benefit * (1.0 - prob_assigned_split) if is_anomaly else 0.0

    return AuditResult(
        admission_no=admission_no,
        status="ok",
        assigned_drg=assigned_drg,
        predicted_mdc=pred_mdc,
        predicted_base_drg=pred_base,
        predicted_complexity_split=pred_split,
        mdc_confidence=round(m_conf, 6),
        base_confidence=round(b_conf, 6),
        split_confidence=round(s_conf, 6),
        prob_assigned_split=round(prob_assigned_split, 6),
        mdc_discrepancy_flag=bool(mdc_discrepancy),
        base_discrepancy_flag=bool(base_discrepancy),
        split_inflation_flag=bool(split_inflation),
        is_billing_anomaly=is_anomaly,
        audit_priority_score=round(score, 4),
        warnings=list(warnings),
    )
