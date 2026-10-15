"""Structures the pipeline reviews: UNITS (what a verdict is given to), NODES (a routing tree over the units), and a per-family index."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dx_support.indexing.bm25 import BM25
from dx_support.text import truncate


@dataclass(frozen=True)
class Unit:
    unit_id: str
    family: str
    text: str                       # judge-facing text: sanitised, length-bounded
    codes: tuple[str, ...]          # billing codes this unit stands for (1 for ACHI / MBS)


@dataclass
class Node:
    node_id: str
    level: int                      # 0-based routing level
    label: str                      # routing-facing text: sanitised, length-bounded
    n_units: int
    children: list[str] = field(default_factory=list)   # next-level node ids ([] at the last level)
    unit_ids: list[str] = field(default_factory=list)   # units directly under the node (last level only)


@dataclass
class FamilyIndex:
    family: str
    units: dict[str, Unit]
    nodes: dict[str, Node]
    root_ids: list[str]
    level_names: list[str]
    excluded: dict[str, str]                        # billing code -> reason it is not reviewed
    code_text: dict[str, str]                       # billing code -> original description / label (for outputs)
    code_meta: dict[str, dict] = field(default_factory=dict)   # prostheses: suffixes, latest-extract flag, ...
    _bm25: Optional[BM25] = field(default=None, repr=False)

    @property
    def n_levels(self) -> int:
        return len(self.level_names)

    @property
    def n_codes(self) -> int:
        return sum(len(u.codes) for u in self.units.values())

    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25({uid: u.text for uid, u in self.units.items()})
        return self._bm25

    def unit_codes(self, unit_id: str) -> tuple[str, ...]:
        return self.units[unit_id].codes

    def root_overview(self, max_chars_per_line: int = 120) -> list[str]:
        """One short line per top-level node -- the 'vocabulary hint' shown to the model so the search terms it writes use this file's wording."""
        return [truncate(self.nodes[r].label, max_chars_per_line) for r in self.root_ids]
