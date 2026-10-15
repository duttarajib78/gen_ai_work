"""The 3-character ICD-10-AM categories supporting lists are built FOR, derived from the leaf-code file."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional

from dx_support.reference.chapters import icd10am_chapter
from dx_support.text import sanitize_prompt_text, truncate

LARGE_CATEGORY_MIN_MEMBERS = 20   # categories with more member codes than this get member-level scoping


@dataclass(frozen=True)
class CategoryInfo:
    category: str                              # 3-character category, e.g. "H25"
    chapter: Optional[str]
    members: tuple[tuple[str, str], ...]       # (leaf code, description), sorted by code
    title: str
    descriptor: str                            # multi-line, grouped member descriptions (for prompts)
    has_own_row: bool                          # the file lists the 3-character code itself

    @property
    def is_large(self) -> bool:
        return len(self.members) > LARGE_CATEGORY_MIN_MEMBERS

    @property
    def member_codes(self) -> tuple[str, ...]:
        return tuple(c for c, _ in self.members)


def _split_head_tail(desc: str) -> tuple[str, str]:
    head, sep, tail = desc.rpartition(", ")
    if sep and len(head) >= 8 and 0 < len(tail) <= 45:
        return head, tail
    return desc, ""


def _head_groups(members: Iterable[tuple[str, str]]) -> "OrderedDict[str, list]":
    """Groups member descriptions by their common leading text, collapsing the trailing site/variant phrase into a per-group list of "tails".

    Phase 1 splits at the last comma, but ONLY when the split is corroborated inside the same category (the head or the tail occurs on at
    least two members) -- otherwise "Infantile, juvenile and presenile cataract" would be mis-split. Phase 2 groups the rest by a shared word
    prefix when at least three members share it ("Pain in joint <site>")."""
    descs = [d for _c, d in members]
    cands = [_split_head_tail(d) for d in descs]
    head_n = Counter(h.lower() for h, t in cands if t)
    tail_n = Counter(t.lower() for h, t in cands if t)
    split: list[tuple[str, str]] = []
    for d, (h, t) in zip(descs, cands):
        split.append((h, t) if t and (head_n[h.lower()] >= 2 or tail_n[t.lower()] >= 2) else (d, ""))

    by_prefix: dict[str, list[int]] = {}
    for i, (h, t) in enumerate(split):
        words = h.split()
        if not t and len(words) >= 4:
            by_prefix.setdefault(" ".join(words[:3]).lower(), []).append(i)
    for idxs in by_prefix.values():
        if len(idxs) < 3:
            continue
        word_lists = [split[i][0].split() for i in idxs]
        lcp = 0
        for col in zip(*word_lists):
            if len({w.lower() for w in col}) == 1:
                lcp += 1
            else:
                break
        if lcp < 2:
            continue
        for i, words in zip(idxs, word_lists):
            tail_words = words[lcp:]
            if 0 < len(tail_words) <= 5:
                split[i] = (" ".join(words[:lcp]), " ".join(tail_words))

    groups: "OrderedDict[str, list]" = OrderedDict()
    for h, t in split:
        groups.setdefault(h.lower(), [h, []])[1].append(t)
    return groups


def build_descriptor(members: tuple[tuple[str, str], ...], max_lines: int = 14, max_chars: int = 1500) -> str:
    """Compact picture of what a category contains. The diagnosis file has no category titles, so the meaning of a category is shown through
    its member descriptions, grouped by their common leading text; the trailing ", <site>"-style differences are collapsed into a short list."""
    lines = []
    for head, tails in _head_groups(members).values():
        shown_tails = [t for t in dict.fromkeys(tails) if t]
        n = len(tails)
        head_s = sanitize_prompt_text(head, 160)
        if n == 1 and not shown_tails:
            lines.append(f"- {head_s}")
        elif shown_tails:
            more = "; ..." if len(shown_tails) > 4 else ""
            lines.append(f"- {head_s} [{n} codes: {'; '.join(sanitize_prompt_text(t, 45) for t in shown_tails[:4])}{more}]")
        else:
            lines.append(f"- {head_s} [{n} codes]")
    if len(lines) > max_lines:
        extra = len(lines) - (max_lines - 1)
        lines = lines[: max_lines - 1] + [f"- (+{extra} further descriptions)"]
    return truncate("\n".join(lines), max_chars)


def build_categories(diagnosis: dict[str, str]) -> dict[str, CategoryInfo]:
    """Group leaf codes by their first three characters."""
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for code, desc in diagnosis.items():
        by_cat.setdefault(code[:3], []).append((code, desc))
    out: dict[str, CategoryInfo] = {}
    for cat in sorted(by_cat):
        members = tuple(sorted(by_cat[cat]))
        own = diagnosis.get(cat)
        if own:
            title = own
        else:
            heads = [h for h, _ in _head_groups(members).values()]
            title = "; ".join(sanitize_prompt_text(h, 70) for h in heads[:3]) + (" ..." if len(heads) > 3 else "")
        out[cat] = CategoryInfo(
            category=cat, chapter=icd10am_chapter(cat), members=members, title=truncate(title, 200),
            descriptor=build_descriptor(members), has_own_row=own is not None,
        )
    return out


def category_heads(info: CategoryInfo, n: int = 8) -> list[str]:
    """The leading wording of a category's members (the same grouping the descriptor uses) -- short phrases suitable as lexical-search seeds."""
    return [sanitize_prompt_text(h, 80) for h, _ in _head_groups(info.members).values()][:n]


def neighbour_titles(categories: dict[str, CategoryInfo], cat: str, n_each_side: int = 2) -> list[tuple[str, str]]:
    """Titles of the categories either side of `cat` in code order -- context for "Other ..." / "unspecified" categories, whose meaning is
    partly "whatever the neighbouring categories are not"."""
    keys = list(categories)
    i = keys.index(cat)
    window = keys[max(0, i - n_each_side): i] + keys[i + 1: i + 1 + n_each_side]
    return [(k, categories[k].title) for k in window]
