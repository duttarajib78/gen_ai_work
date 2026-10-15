"""The alert decision rule, as a pure function.

A diagnosis (PDx or ADx) FAILS when the model says its ACHI/MBS support is:

* ``none``                                                  -> fails
* ``weak`` with confidence below the threshold              -> fails
* ``weak`` with confidence at or above the threshold        -> passes
* ``clear`` (any confidence)                                -> passes

An episode is a potential alert when the PDx or at least one ADx fails. The
threshold is an argument (from ``Settings``), never module state, so the rule
used for a response is always the one recorded in its provenance.

The default threshold of 0.5 is provisional: the prompt defines 0.5 as "a coin
flip". It was set for a different model, and confidence calibration differs
between models, so re-validate it with ``tools/evaluation`` before relying on it.
"""
from __future__ import annotations

from typing import Optional

RULE_VERSION = "any-verdict:none-or-lowconf-weak/1"

SUPPORT_LEVELS = ("none", "weak", "clear")


def diagnosis_fails(support_level: str, confidence: float, threshold: float) -> bool:
    if support_level == "none":
        return True
    if support_level == "weak":
        return confidence < threshold
    return False


def flag_reason(support_level: str, confidence: float, threshold: float) -> Optional[str]:
    """Label for a failing verdict (``none`` or ``low_confidence_weak``); None if it passes."""
    if not diagnosis_fails(support_level, confidence, threshold):
        return None
    return "none" if support_level == "none" else "low_confidence_weak"
