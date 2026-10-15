"""Confusion-matrix metrics shared by the benchmark and the threshold sweep.

"Unsupported" (True) is the positive class throughout: this is a rare-positive
screening task, so precision, recall, and F1 describe how well the audit catches
that class, not the clean majority.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

SMALL_SAMPLE_THRESHOLD = 30


def confusion(pairs: Iterable[tuple[bool, bool]]) -> dict:
    """pairs: (ground_truth_unsupported, predicted_unsupported)."""
    tp = fp = fn = tn = 0
    for truth, predicted in pairs:
        if truth and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def compute_metrics(counts: dict) -> dict:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n = tp + fp + fn + tn
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n_scored": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": _ratio(tp + tn, n), "specificity": _ratio(tn, tn + fp),
        "mcc": ((tp * tn - fp * fn) / denom) if denom else None,
        "prevalence": _ratio(tp + fn, n),
    }


def fmt(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, float) else ("n/a" if value is None else str(value))


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
