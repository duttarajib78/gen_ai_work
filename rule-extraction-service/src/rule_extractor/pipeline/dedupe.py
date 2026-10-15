"""Collapse duplicate rules (same normalised logic text) before writing.

First occurrence wins (stable, deterministic order); if a later duplicate is executable and the kept one is not, the
executable version replaces it so no automation is lost. Near-duplicates across passes are NOT merged (known limit).
"""
from __future__ import annotations

from typing import Sequence

from ..domain.schema import Rule
from ..engine.rules_store import ExecutableRule

Pair = tuple[Rule, "ExecutableRule | None"]


def dedupe_rules(pairs: Sequence[Pair]) -> tuple[list[Pair], int]:
    kept: list[Pair] = []
    index: dict[str, int] = {}
    dropped = 0
    for rule, executable in pairs:
        key = rule.dedupe_key()
        if key not in index:
            index[key] = len(kept)
            kept.append((rule, executable))
            continue
        dropped += 1
        pos = index[key]
        if kept[pos][1] is None and executable is not None:
            kept[pos] = (rule, executable)
    return kept, dropped
