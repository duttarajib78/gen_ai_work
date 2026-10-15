"""Turns the code families into the structures the pipeline reviews.

  * UNITS      -- the things a verdict is given to. ACHI and MBS: one unit per code. Prostheses: one unit per DEVICE TYPE (a distinct category >
                  sub-category > product group > product sub-group path) covering every billing code under it, because the file has no product
                  names: all codes in a group carry the same label, so judging them one by one would repeat one verdict hundreds of times.
  * NODES      -- a routing tree over the units. A model is shown one level at a time and picks the branches worth opening, so a category is
                  never compared with all ~12,400 units. Fan-out is small at every level (ACHI <= 10 children per node).
  * EXCLUSIONS -- units that cannot indicate a diagnosis was treated, investigated or examined (professional attendances, anaesthesia-schedule
                  items, content-free placeholder codes, coarse-label-only prostheses). Each excluded code keeps its REASON; nothing is silently dropped.

No clinical knowledge is hardcoded here: no synonym tables, no code-to-diagnosis mappings. The only clinical-ish rules are the generic-item
exclusions, each verified against the real files and reported with counts by `dx-support estimate --show-universe`.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

from dx_support.indexing.models import FamilyIndex, Node, Unit
from dx_support.reference.categories import CategoryInfo
from dx_support.reference.chapters import CHAPTER_TITLES, CONTEXT_ONLY_CHAPTERS
from dx_support.reference.loaders import MbsItem, ProsRecord
from dx_support.text import sanitize_prompt_text, truncate


def sample_texts(texts: list[str], k: int, width: int) -> list[str]:
    """k evenly spaced, de-duplicated, sanitised snippets from `texts` (assumed ordered by code) -- the only 'title' a node without a real
    title can have."""
    n = len(texts)
    if n == 0:
        return []
    idxs = list(range(n)) if n <= k else sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    out, seen = [], set()
    for i in idxs:
        t = sanitize_prompt_text(texts[i], width)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _short_hash(*parts: str, n: int = 10) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:n]   # noqa: S324 - identifier, not a security control


# ---------------------------------------------------------------------------
# ACHI
# ---------------------------------------------------------------------------
ACHI_PLACEHOLDER_RE = re.compile(r"^(?:Emergency|Provisional) use of (\d{5})-(\d{2})\b", re.I)
ACHI_PREFIX_LENGTHS = (2, 3, 4)          # routing levels: 2-digit group > 3-digit > 4-digit block; units are the 7-digit codes
_ACHI_LABEL_SAMPLES = ((7, 55), (5, 60), (4, 70))   # (snippets, chars each) per level


def achi_exclusion_reason(code: str, description: str) -> Optional[str]:
    """'Emergency/Provisional use of NNNNN-NN [....]' codes carry no clinical content -- in the current file all 195 cite THEMSELVES."""
    m = ACHI_PLACEHOLDER_RE.match(description)
    if not m:
        return None
    return "placeholder_self_citing" if m.group(1) + m.group(2) == code else "placeholder_cites_other_code"


def build_achi_index(descriptions: dict[str, str]) -> FamilyIndex:
    units: dict[str, Unit] = {}
    excluded: dict[str, str] = {}
    for code, desc in descriptions.items():
        reason = achi_exclusion_reason(code, desc)
        if reason:
            excluded[code] = reason
        else:
            units[code] = Unit(code, "achi", sanitize_prompt_text(desc, 220), (code,))

    nodes: dict[str, Node] = {}
    groups: list[dict[str, list[str]]] = []
    for k in ACHI_PREFIX_LENGTHS:
        g: dict[str, list[str]] = defaultdict(list)
        for code in sorted(units):
            g[code[:k]].append(code)
        groups.append(g)
    for level, (k, g) in enumerate(zip(ACHI_PREFIX_LENGTHS, groups)):
        n_snip, width = _ACHI_LABEL_SAMPLES[level]
        last = level == len(ACHI_PREFIX_LENGTHS) - 1
        for prefix, codes in g.items():
            snippets = sample_texts([units[c].text for c in codes], n_snip, width)
            node = Node(node_id=f"achi:{k}:{prefix}", level=level, n_units=len(codes),
                        label=f"{prefix}* ({len(codes)} codes): " + "; ".join(snippets))
            if last:
                node.unit_ids = list(codes)
            else:
                nxt = ACHI_PREFIX_LENGTHS[level + 1]
                node.children = sorted({f"achi:{nxt}:{c[:nxt]}" for c in codes})
            nodes[node.node_id] = node
    root_ids = sorted(n.node_id for n in nodes.values() if n.level == 0)
    return FamilyIndex(family="achi", units=units, nodes=nodes, root_ids=root_ids,
                       level_names=["2-digit group", "3-digit subgroup", "4-digit block"], excluded=excluded, code_text=dict(descriptions))


# ---------------------------------------------------------------------------
# MBS
# ---------------------------------------------------------------------------
# Category 3 items written on the anaesthesia schedule (basic-unit and time-band wording). Verified against the current file: 488 items, all of
# them anaesthesia services -- none identifies the diagnosis any better than the operation it accompanies.
MBS_ANAES_SCHEDULE_RE = re.compile(r"\bbasic units?\b|^(?:initiation of the management of anaesthesia|management of anaesthesia|anaesthesia)\b", re.I)
_MBS_LABEL_SAMPLES = ((5, 80), (4, 80))


def mbs_exclusion_reason(item: MbsItem) -> Optional[str]:
    if item.ended:
        return "ended_item"
    if item.category == "1":
        return "category_1_professional_attendance"
    if item.category == "3" and MBS_ANAES_SCHEDULE_RE.search(item.description):
        return "anaesthesia_schedule_item"
    return None


def build_mbs_index(items: Iterable[MbsItem]) -> FamilyIndex:
    units: dict[str, Unit] = {}
    excluded: dict[str, str] = {}
    code_text: dict[str, str] = {}
    kept: list[MbsItem] = []
    for it in items:
        code_text[it.code] = it.description
        reason = mbs_exclusion_reason(it)
        if reason:
            excluded[it.code] = reason
        else:
            kept.append(it)
            units[it.code] = Unit(it.code, "mbs", sanitize_prompt_text(it.description, 300), (it.code,))

    def _num(code: str) -> tuple[int, str]:
        head = code.split("-")[0]
        return (int(head) if head.isdigit() else 0, code)

    by_group: dict[tuple[str, str], dict[str, list[MbsItem]]] = defaultdict(lambda: defaultdict(list))
    for it in sorted(kept, key=lambda i: _num(i.code)):
        by_group[(it.category, it.group)][it.subgroup or "-"].append(it)

    nodes: dict[str, Node] = {}
    for (cat, grp), subs in sorted(by_group.items()):
        g_items = [i for lst in subs.values() for i in lst]
        gid = f"mbs:g:{cat}|{grp}"
        g_node = Node(node_id=gid, level=0, n_units=len(g_items),
                      label=f"Category {cat} / Group {grp} ({len(g_items)} items): "
                            + "; ".join(sample_texts([i.description for i in g_items], *_MBS_LABEL_SAMPLES[0])))
        for sub, lst in sorted(subs.items()):
            sid = f"mbs:s:{cat}|{grp}|{sub}"
            s_label_sub = "" if sub == "-" else f" / SubGroup {sub}"
            nodes[sid] = Node(node_id=sid, level=1, n_units=len(lst), unit_ids=[i.code for i in lst],
                              label=f"Category {cat} / Group {grp}{s_label_sub} ({len(lst)} items): "
                                    + "; ".join(sample_texts([i.description for i in lst], *_MBS_LABEL_SAMPLES[1])))
            g_node.children.append(sid)
        nodes[gid] = g_node
    root_ids = sorted(n.node_id for n in nodes.values() if n.level == 0)
    return FamilyIndex(family="mbs", units=units, nodes=nodes, root_ids=root_ids, level_names=["category / group", "sub-group"],
                       excluded=excluded, code_text=code_text)


# ---------------------------------------------------------------------------
# Prostheses
# ---------------------------------------------------------------------------
def _path_label(path: tuple[str, str, str, str]) -> str:
    return " > ".join(p for p in path if p)


def build_prosthesis_index(records: dict[str, ProsRecord], include_coarse: bool = False) -> FamilyIndex:
    """One unit per device type (distinct label path). Coarse-label-only codes (no sub-category) are excluded unless include_coarse=True, in
    which case they form one low-resolution unit per coarse category."""
    excluded: dict[str, str] = {}
    code_text: dict[str, str] = {}
    code_meta: dict[str, dict] = {}
    by_path: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for code, rec in sorted(records.items()):
        code_text[code] = _path_label(rec.path)
        code_meta[code] = {"suffixes": ", ".join(rec.suffixes), "in_latest_extract": rec.in_latest_extract,
                           "n_label_variants": rec.n_label_variants, "label_resolution": "full" if rec.labelled else "coarse"}
        if rec.labelled:
            by_path[rec.path].append(code)
        elif include_coarse:
            by_path[(rec.path[0], "", "", "")].append(code)
        else:
            excluded[code] = "coarse_label_only"

    units: dict[str, Unit] = {}
    unit_path: dict[str, tuple[str, str, str, str]] = {}
    for path, codes in sorted(by_path.items()):
        uid = "pg:" + _short_hash(*path)
        coarse = not path[1]
        text = sanitize_prompt_text(_path_label(path), 250) + (
            f" (LOW-RESOLUTION: only this coarse category is recorded; {len(codes)} billing codes)" if coarse else f" ({len(codes)} billing codes)")
        units[uid] = Unit(uid, "prosthesis", text, tuple(sorted(codes)))
        unit_path[uid] = path

    # level 0: sub-category (or the coarse category); level 1: product group
    sub_units: dict[tuple[str, str], list[str]] = defaultdict(list)
    for uid, p in unit_path.items():
        sub_units[(p[0], p[1])].append(uid)
    nodes: dict[str, Node] = {}
    for (cat, sub), uids in sorted(sub_units.items()):
        by_grp: dict[str, list[str]] = defaultdict(list)
        for uid in uids:
            by_grp[unit_path[uid][2]].append(uid)
        n_codes = sum(len(units[u].codes) for u in uids)
        sid = "pros:s:" + _short_hash(cat, sub)
        grp_names = [g or "(no product group)" for g in sorted(by_grp)]
        s_node = Node(node_id=sid, level=0, n_units=len(uids),
                      label=truncate(f"{cat} > {sub or '(coarse category only)'} ({n_codes} billing codes, {len(by_grp)} product groups"
                                     + (": " + "; ".join(sanitize_prompt_text(g, 60) for g in grp_names[:6]) + ("; ..." if len(grp_names) > 6 else "") if sub else "")
                                     + ")", 420))
        for grp, g_uids in sorted(by_grp.items()):
            gid = "pros:g:" + _short_hash(cat, sub, grp)
            g_codes = sum(len(units[u].codes) for u in g_uids)
            ssg = [unit_path[u][3] for u in g_uids if unit_path[u][3]]
            snippets = "; ".join(sanitize_prompt_text(x, 70) for x in ssg[:4]) + ("; ..." if len(ssg) > 4 else "")
            nodes[gid] = Node(node_id=gid, level=1, n_units=len(g_uids), unit_ids=sorted(g_uids),
                              label=truncate(f"{cat} > {sub} > {grp or '(no product group)'} ({g_codes} billing codes, {len(g_uids)} device types)"
                                             + (f": {snippets}" if snippets else ""), 420))
            s_node.children.append(gid)
        nodes[sid] = s_node
    root_ids = sorted(n.node_id for n in nodes.values() if n.level == 0)
    return FamilyIndex(family="prosthesis", units=units, nodes=nodes, root_ids=root_ids, level_names=["sub-category", "product group"],
                       excluded=excluded, code_text=code_text, code_meta=code_meta)


# ---------------------------------------------------------------------------
# Diagnoses (the "which OTHER diagnoses make a target plausible" family)
# ---------------------------------------------------------------------------
def build_dx_index(categories: dict[str, CategoryInfo]) -> FamilyIndex:
    """Units are the 3-character categories themselves. Chapters XX and XXII are not candidates, for the same reason they are not targets.
    Routing is two levels: chapter, then block (letter + tens digit: at most ten categories). A category is never its own supporting
    diagnosis -- the pipeline removes the target from its own candidates."""
    units: dict[str, Unit] = {}
    excluded: dict[str, str] = {}
    code_text: dict[str, str] = {}
    by_chapter: dict[str, list[str]] = defaultdict(list)
    for cat, info in categories.items():
        code_text[cat] = info.members[0][1]
        if info.chapter in CONTEXT_ONLY_CHAPTERS:
            excluded[cat] = "context_only_chapter"
            continue
        units[cat] = Unit(cat, "dx", sanitize_prompt_text(info.title, 200), (cat,))
        by_chapter[info.chapter or "?"].append(cat)
    nodes: dict[str, Node] = {}
    for ch, cats in sorted(by_chapter.items()):
        cats.sort()
        blocks: dict[str, list[str]] = defaultdict(list)
        for c in cats:
            blocks[c[:2]].append(c)
        title = CHAPTER_TITLES.get(ch, "")
        ch_node = Node(node_id=f"dx:c:{ch}", level=0, n_units=len(cats),
                       label=f"Chapter {ch}{' ' + title if title else ''} ({len(cats)} categories, {cats[0]}-{cats[-1]}): "
                             + "; ".join(sample_texts([units[c].text for c in cats], 6, 60)))
        for b, bc in sorted(blocks.items()):
            bid = f"dx:b:{ch}|{b}"
            nodes[bid] = Node(node_id=bid, level=1, n_units=len(bc), unit_ids=list(bc),
                              label=f"{bc[0]}-{bc[-1]} ({len(bc)} categories): " + "; ".join(sample_texts([units[c].text for c in bc], 5, 60)))
            ch_node.children.append(bid)
        nodes[ch_node.node_id] = ch_node
    root_ids = sorted(n.node_id for n in nodes.values() if n.level == 0)
    return FamilyIndex(family="dx", units=units, nodes=nodes, root_ids=root_ids, level_names=["chapter", "block (letter + tens digit)"],
                       excluded=excluded, code_text=code_text)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------
def expand_to_units(index: FamilyIndex, leaf_node_ids: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for nid in leaf_node_ids:
        for uid in index.nodes[nid].unit_ids:
            seen.setdefault(uid, None)
    return list(seen)


def index_stats(index: FamilyIndex) -> dict:
    """Facts about one family used by `dx-support estimate` and the tests."""
    by_level = Counter(n.level for n in index.nodes.values())
    fan = [len(n.children) for n in index.nodes.values() if n.children]
    unit_chars = [len(u.text) for u in index.units.values()]
    root_chars = sum(len(index.nodes[r].label) for r in index.root_ids)
    return {
        "family": index.family, "n_units": len(index.units), "n_codes_in_units": index.n_codes, "n_excluded": len(index.excluded),
        "excluded_by_reason": dict(Counter(index.excluded.values())),
        "nodes_by_level": {index.level_names[lv]: by_level[lv] for lv in sorted(by_level)},
        "max_fanout": max(fan) if fan else 0,
        "mean_unit_chars": (sum(unit_chars) / len(unit_chars)) if unit_chars else 0.0,
        "root_label_chars": root_chars,
        "mean_label_chars": (sum(len(n.label) for n in index.nodes.values()) / max(len(index.nodes), 1)),
        "max_units_per_leaf_node": max((len(n.unit_ids) for n in index.nodes.values() if n.unit_ids), default=0),
    }


def check_index_integrity(index: FamilyIndex) -> list[str]:
    """Structural invariants -- returns a list of problems (empty = healthy)."""
    problems: list[str] = []
    reachable: set[str] = set()
    for rid in index.root_ids:
        stack = [rid]
        while stack:
            nid = stack.pop()
            node = index.nodes.get(nid)
            if node is None:
                problems.append(f"{index.family}: dangling node id {nid}")
                continue
            reachable.update(node.unit_ids)
            stack.extend(node.children)
    missing = set(index.units) - reachable
    if missing:
        problems.append(f"{index.family}: {len(missing)} unit(s) unreachable from the roots, e.g. {sorted(missing)[:3]}")
    extra = reachable - set(index.units)
    if extra:
        problems.append(f"{index.family}: tree references {len(extra)} unknown unit(s)")
    covered = {c for u in index.units.values() for c in u.codes}
    overlap = covered & set(index.excluded)
    if overlap:
        problems.append(f"{index.family}: {len(overlap)} code(s) are both reviewed and excluded")
    return problems
