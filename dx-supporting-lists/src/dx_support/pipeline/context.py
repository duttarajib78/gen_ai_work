"""Everything one run (or one service job) needs, in one object."""
from __future__ import annotations

from dataclasses import dataclass, field

from dx_support.constants import DEFAULT_FAMILIES, DEFAULT_THINKING, SUPPORT_TIERS
from dx_support.indexing.models import FamilyIndex
from dx_support.llm.gate import LLMGate
from dx_support.reference.categories import CategoryInfo, neighbour_titles
from dx_support.settings import DEFAULT_MAX_TOKENS
from dx_support.storage.checkpoint import Checkpoint


@dataclass
class RunContext:
    categories: dict[str, CategoryInfo]
    indices: dict[str, FamilyIndex]
    gate: LLMGate
    ckpt: Checkpoint
    families: tuple[str, ...] = DEFAULT_FAMILIES
    mode: str = "routed"                       # "routed" (production) | "exhaustive" (validation / certainty)
    models: dict[str, str] = field(default_factory=lambda: {"brief": "claude-sonnet-5", "route": "claude-sonnet-5", "judge": "claude-sonnet-5"})
    thinking: str = DEFAULT_THINKING
    include_context_chapters: bool = False
    brief_skip: bool = False                   # opt-in: skip categories the brief marks as having no procedural evidence
    lexical_rescue: bool = True
    max_budget_usd_per_call: float = 0.50
    max_tokens: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_TOKENS))
    fingerprint: str = ""
    vocab: dict[str, list[str]] = field(default_factory=dict)
    tier_counts: dict[str, int] = field(default_factory=lambda: {t: 0 for t in SUPPORT_TIERS})
    _neighbours: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def neighbours(self, cat: str) -> list[tuple[str, str]]:
        if cat not in self._neighbours:
            self._neighbours[cat] = neighbour_titles(self.categories, cat)
        return self._neighbours[cat]

    def category_text(self, info: CategoryInfo) -> str:
        return f"{info.title} {info.descriptor}"
