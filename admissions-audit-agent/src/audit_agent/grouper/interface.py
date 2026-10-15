"""Grouper contract used by the engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Protocol

from ..deadline import Deadline
from ..schemas import Episode


@dataclass(frozen=True)
class RegroupResult:
    codes_removed: tuple[str, ...]
    pdx_resequence: Optional[tuple[str, str]]  # (old_pdx_code, new_principal_code)
    original_drg: Optional[str]
    original_drg_desc: Optional[str]
    original_cost_weight: Optional[float]
    regrouped_drg: Optional[str]
    regrouped_drg_desc: Optional[str]
    regrouped_cost_weight: Optional[float]
    drg_changed: bool
    cost_weight_drop: Optional[float]  # original - regrouped; positive means the weight dropped
    possible_recovery: Optional[float]
    drg_ver_grouper: Optional[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["codes_removed"] = list(self.codes_removed)
        data["pdx_resequence"] = list(self.pdx_resequence) if self.pdx_resequence else None
        return data


class Grouper(Protocol):
    def regroup(
        self,
        episode: Episode,
        *,
        removed_codes: frozenset,
        pdx_resequence: Optional[tuple[str, str]],
        deadline: Deadline,
    ) -> RegroupResult:
        ...
