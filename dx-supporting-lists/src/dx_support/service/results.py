"""Turns a category's stored verdicts into the response a caller receives.

A verdict is given to a UNIT (an ACHI code, an MBS item, a prosthesis DEVICE TYPE, a diagnosis category). The supporting list is every code of every
unit whose tier is in `merge_tiers` (default core + broad); a prosthesis verdict therefore expands to every billing code in its device type. Every
verdict, merged or not, can be returned with `include_audit` for coder review.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from dx_support.constants import SUPPORT_TIERS
from dx_support.indexing.universe import Universe
from dx_support.pipeline.stages import families_done
from dx_support.reference.chapters import CONTEXT_ONLY_CHAPTERS
from dx_support.storage.checkpoint import Checkpoint

TIER_RANK = {t: i for i, t in enumerate(SUPPORT_TIERS)}
MAX_AUDIT_ROWS = 5000


def code_sort_key(code: str) -> tuple:
    return (0, int(code), "") if code.isdigit() else (1, 0, code)


def assemble_category(*, ckpt: Checkpoint, universe: Universe, category: str, families: Iterable[str], merge_tiers: Iterable[str],
                      prosthesis_extract: str = "all", include_audit: bool = False, mode: str = "routed") -> dict:
    families, merge = tuple(families), set(merge_tiers)
    info = universe.categories[category]
    status = ckpt.status.get(category, {})
    done = families_done(status, mode)
    fam_status = status.get("families") or {}
    verdicts = ckpt.read_verdicts(category)
    out_fams: dict[str, dict] = {}
    audit: list[dict] = []
    for fam in families:
        index = universe.indices[fam]
        items: list[dict] = []
        counts: Counter = Counter()
        reviewed = 0
        for (_c, f, uid), rec in verdicts.items():
            if f != fam:
                continue
            unit = index.units.get(uid)
            tier = rec.get("support_tier")
            if unit is None or tier not in SUPPORT_TIERS:
                continue                                            # a verdict for a unit the current index does not know is never served
            reviewed += 1
            scope = rec.get("member_scope", "all")
            if include_audit and len(audit) < MAX_AUDIT_ROWS:
                audit.append({"family": fam, "unit_id": uid, "unit_text": unit.text, "n_codes": len(unit.codes), "support_tier": tier,
                              "evidence_type": rec.get("evidence_type"), "rationale": rec.get("rationale", ""), "member_scope": scope,
                              "source": rec.get("source", ""), "model": rec.get("model", ""), "merged": tier in merge})
            if tier not in merge:
                continue
            codes = unit.codes
            if fam == "prosthesis" and prosthesis_extract == "latest":
                codes = tuple(c for c in codes if index.code_meta[c]["in_latest_extract"])
            for code in codes:
                item = {"code": code, "description": index.code_text.get(code, ""), "support_tier": tier, "evidence_type": rec.get("evidence_type"),
                        "member_scope": scope, "rationale": rec.get("rationale", ""), "unit_id": uid, "source": rec.get("source", "")}
                if fam == "prosthesis":
                    m = index.code_meta[code]
                    item.update(device_type=index.code_text.get(code, ""), suffixes=m["suffixes"], in_latest_extract=m["in_latest_extract"],
                                label_resolution=m["label_resolution"], n_label_variants=m["n_label_variants"])
                items.append(item)
                counts[tier] += 1
        items.sort(key=lambda i: (TIER_RANK[i["support_tier"]], code_sort_key(i["code"])))
        fs = fam_status.get(fam, {})
        out_fams[fam] = {"state": "complete" if fam in done else ("partial" if fam in fam_status else "pending"), "units_reviewed": reviewed,
                         "n_candidates": fs.get("n_candidates"), "n_items": len(items), "counts": {t: counts[t] for t in SUPPORT_TIERS if counts[t]},
                         "candidates_truncated": bool(fs.get("truncated_nodes") or fs.get("family_truncated")), "items": items}
    if status.get("status") == "skipped_context_only" or (info.chapter in CONTEXT_ONLY_CHAPTERS and not status):
        state = "skipped_context_only"
    elif set(families) <= done:
        state = "complete"
    else:
        state = "partial" if status else "pending"
    res = {"status": state, "title": info.title, "chapter": info.chapter, "n_member_codes": len(info.members), "families": out_fams}
    if status.get("error"):
        res["error"] = str(status["error"])[:200]
    if include_audit:
        audit.sort(key=lambda a: (a["family"], TIER_RANK[a["support_tier"]], code_sort_key(a["unit_id"]) if a["unit_id"].isdigit() else (1, 0, a["unit_id"])))
        res["audit"] = audit
    return res
