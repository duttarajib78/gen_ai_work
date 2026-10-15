"""Scope / cost projection -- shared by `dx-support estimate` and the confirmation gate of `dx-support build`.

ORDER-OF-MAGNITUDE only: it is computed from the real universe sizes and explicit assumptions, and no model call is made. Prices come from the model
catalog (point-in-time, not fetched live). After a pilot, prefer `calibrate_from_pilot()`: measured per-call costs replace the assumptions.
"""
from __future__ import annotations

import math
import statistics
from typing import Callable, Iterable, Optional

from dx_support.constants import DEFAULT_FAMILIES
from dx_support.indexing.families import index_stats
from dx_support.indexing.models import FamilyIndex
from dx_support.llm.catalog import CatalogError, load_catalog, resolve_model
from dx_support.pipeline import prompts as P
from dx_support.pipeline.constants import (ASSUMED_SELECT_FRACTION, CALL_OVERHEAD_INPUT_TOKENS, CHARS_PER_TOKEN, COST_UNCERTAINTY,
                                           DEFAULT_CATEGORY_CONCURRENCY, EST_OUTPUT_TOKENS_PER_VERDICT, EST_THINKING_TOKENS, JUDGE_BATCH_SIZE, LEXICAL_TOP_K,
                                           MAX_UNITS_PER_FAMILY, MAX_UNITS_PER_LEAF_NODE, ROUTE_MAX_NODES_PER_CALL, SECONDS_PER_CALL_ESTIMATE,
                                           TOKENIZER_ADJUST, VOCAB_FAMILIES)
from dx_support.reference.categories import CategoryInfo
from dx_support.storage.checkpoint import Checkpoint

PriceFn = Callable[[str], tuple[float, float, bool]]


def make_price_fn(prefix: str = "au", catalog_path=None) -> PriceFn:
    """(input $/M, output $/M, known) for a model name, including the geo premium of the configured inference-profile prefix."""
    catalog = load_catalog(catalog_path)

    def price(model: str) -> tuple[float, float, bool]:
        try:
            r = resolve_model(model, prefix, catalog)
        except CatalogError:
            s = catalog.default_spec()
            return s.price_in, s.price_out, False
        return r.spec.price_in * r.price_multiplier, r.spec.price_out * r.price_multiplier, r.spec.known
    return price


def _tokens(chars: float) -> float:
    return chars / CHARS_PER_TOKEN * TOKENIZER_ADJUST


def fmt_usd(x: float) -> str:
    """Cents for small amounts, whole dollars for large ones -- a one-category projection must not print as '$0'."""
    return f"${x:,.2f}" if x < 100 else f"${x:,.0f}"


def fmt_hours(h: float) -> str:
    return f"{h * 3600:,.0f} s" if h * 60 < 1 else f"{h * 60:,.0f} min" if h < 1 else f"{h:,.1f} h"


def estimate_hours(calls: float, depth: float, n_categories: int, concurrency: int, category_concurrency: int,
                   seconds_per_call: float = SECONDS_PER_CALL_ESTIMATE) -> float:
    """Wall-clock estimate bounded two ways: total throughput (calls / concurrency) and chain length (each category's stages run in sequence --
    brief, then one call per routing level, then the judge -- and only `category_concurrency` categories are in flight at once). The slower wins."""
    by_calls = calls * seconds_per_call / max(min(concurrency, calls), 1)
    by_chain = n_categories * depth * seconds_per_call / max(min(category_concurrency, n_categories), 1)
    return max(by_calls, by_chain) / 3600


def _levels_profile(index: FamilyIndex) -> list[dict]:
    """Per routing level: node count, mean label chars, mean children per parent."""
    prof = []
    for lvl in range(index.n_levels):
        nodes = [n for n in index.nodes.values() if n.level == lvl]
        prof.append({"n_nodes": len(nodes), "mean_label_chars": sum(len(n.label) for n in nodes) / max(len(nodes), 1),
                     "mean_units": sum(n.n_units for n in nodes) / max(len(nodes), 1)})
    for lvl in range(index.n_levels):
        prof[lvl]["fanout_from_parent"] = (prof[lvl]["n_nodes"] / prof[lvl - 1]["n_nodes"]) if lvl > 0 else float(prof[0]["n_nodes"])
    return prof


def project_scope(
    categories: list[CategoryInfo], n_skipped_context: int, indices: dict[str, FamilyIndex], families: Iterable[str], models: dict[str, str],
    thinking: str = "low", concurrency: int = 20, select_fraction: tuple[float, float] = ASSUMED_SELECT_FRACTION, lexical_rescue: bool = True,
    category_concurrency: int = DEFAULT_CATEGORY_CONCURRENCY, call_overhead_tokens: int = CALL_OVERHEAD_INPUT_TOKENS, n_cached_briefs: int = 0,
    price_fn: Optional[PriceFn] = None,
) -> dict:
    price_fn = price_fn or make_price_fn()

    def call_cost(model: str, in_chars: float, out_tokens: float, overhead: float = 0.0) -> float:
        pin, pout, _ = price_fn(model)
        return ((_tokens(in_chars) + overhead) / 1e6) * pin + ((out_tokens + EST_THINKING_TOKENS[thinking]) / 1e6) * pout

    families = tuple(families)
    n_cat = len(categories)
    mean_cat_chars = (statistics.mean(len(c.descriptor) + len(c.title) for c in categories) if categories else 300) + 900   # + neighbours + brief
    big_cat_chars = 90 * 75   # member list added for 'large' categories
    frac_large = (sum(1 for c in categories if c.is_large) / n_cat) if n_cat else 0.0
    per_family: dict[str, dict] = {}
    totals = {"exh_calls": 0.0, "exh_cost": 0.0, "lo_calls": 0.0, "hi_calls": 0.0, "lo_cost": 0.0, "hi_cost": 0.0}

    vocab_chars = sum(len(line) for fam in families if fam in VOCAB_FAMILIES for line in indices[fam].root_overview(150 if fam == "achi" else 110))
    brief_cost = call_cost(models["brief"], len(P.SYSTEM_PROMPT_BRIEF) + vocab_chars + mean_cat_chars, 450, call_overhead_tokens)
    brief_calls = max(n_cat - n_cached_briefs, 0)      # briefs already checkpointed are reused, not re-bought
    brief_total = brief_calls * brief_cost

    for fam in families:
        idx = indices[fam]
        st = index_stats(idx)
        prof = _levels_profile(idx)
        batch = JUDGE_BATCH_SIZE[fam]
        n_units = st["n_units"]

        def judge_cost(n: float, st=st) -> float:
            judge_in = len(P.SYSTEM_PROMPT_JUDGE) + mean_cat_chars + frac_large * big_cat_chars + n * (st["mean_unit_chars"] + 30)
            return call_cost(models["judge"], judge_in, n * EST_OUTPUT_TOKENS_PER_VERDICT, call_overhead_tokens)

        exh_chunks = math.ceil(n_units / batch)
        exh_cost_per_cat = sum(judge_cost(min(batch, n_units - i * batch)) for i in range(exh_chunks))

        scenarios = {}
        for tag, frac in zip(("lo", "hi"), select_fraction):
            calls = 0.0
            cost = 0.0
            frontier = float(prof[0]["n_nodes"])
            selected = 0.0
            for lvl, p in enumerate(prof):
                n_calls = max(1, math.ceil(frontier / ROUTE_MAX_NODES_PER_CALL)) if frontier > 0 else 0
                chunk_nodes = frontier / max(n_calls, 1)
                per_call = call_cost(models["route"], len(P.SYSTEM_PROMPT_ROUTE) + mean_cat_chars + 600 + chunk_nodes * (p["mean_label_chars"] + 12),
                                     60 + 14 * chunk_nodes * frac, call_overhead_tokens)
                calls += n_calls
                cost += n_calls * per_call
                selected = frontier * frac
                if lvl + 1 < len(prof):
                    frontier = selected * prof[lvl + 1]["fanout_from_parent"]
            leaf = prof[-1]
            route_units = min(selected * min(leaf["mean_units"], MAX_UNITS_PER_LEAF_NODE), MAX_UNITS_PER_FAMILY[fam])
            lex = LEXICAL_TOP_K[fam] * 0.5 if lexical_rescue else 0
            cand = min(route_units + lex, MAX_UNITS_PER_FAMILY[fam], n_units)
            j_chunks = max(1, math.ceil(cand / batch))
            j_cost = sum(judge_cost(cand / j_chunks) for _ in range(j_chunks))
            scenarios[tag] = {"route_calls": calls, "judge_calls": j_chunks, "candidates": cand, "cost_per_cat": cost + j_cost}
        per_family[fam] = {
            "n_units": n_units, "n_codes": st["n_codes_in_units"], "n_excluded": st["n_excluded"], "excluded_by_reason": st["excluded_by_reason"],
            "nodes_by_level": st["nodes_by_level"], "judge_batch": batch, "exhaustive_calls_per_category": exh_chunks,
            "exhaustive_cost_per_category": exh_cost_per_cat, "routed_lo": scenarios["lo"], "routed_hi": scenarios["hi"],
        }
        totals["exh_calls"] += n_cat * exh_chunks
        totals["exh_cost"] += n_cat * exh_cost_per_cat
        totals["lo_calls"] += n_cat * (scenarios["lo"]["route_calls"] + scenarios["lo"]["judge_calls"])
        totals["hi_calls"] += n_cat * (scenarios["hi"]["route_calls"] + scenarios["hi"]["judge_calls"])
        totals["lo_cost"] += n_cat * scenarios["lo"]["cost_per_cat"]
        totals["hi_cost"] += n_cat * scenarios["hi"]["cost_per_cat"]

    lo_calls, hi_calls = totals["lo_calls"] + brief_calls, totals["hi_calls"] + brief_calls
    lo_cost, hi_cost = (totals["lo_cost"] + brief_total) * COST_UNCERTAINTY[0], (totals["hi_cost"] + brief_total) * COST_UNCERTAINTY[1]
    depth = 1 + max((indices[f].n_levels + 1 for f in families), default=1)     # brief + routing levels + judge, in sequence
    known = {m: price_fn(m)[2] for m in set(models.values())}
    return {
        "n_categories": n_cat, "n_skipped_context": n_skipped_context, "per_family": per_family, "brief_calls": brief_calls, "brief_cost": brief_total,
        "routed": {"calls_low": lo_calls, "calls_high": hi_calls, "cost_low": lo_cost, "cost_high": hi_cost, "depth": depth,
                   "hours_low": estimate_hours(lo_calls, depth, n_cat, concurrency, category_concurrency),
                   "hours_high": estimate_hours(hi_calls, depth, n_cat, concurrency, category_concurrency)},
        "exhaustive": {"calls": totals["exh_calls"] + brief_calls, "cost": (totals["exh_cost"] + brief_total) * 1.0,
                       "cost_low": (totals["exh_cost"] + brief_total) * COST_UNCERTAINTY[0], "cost_high": (totals["exh_cost"] + brief_total) * COST_UNCERTAINTY[1],
                       "pairs": n_cat * sum(per_family[f]["n_units"] for f in families),
                       "hours": estimate_hours(totals["exh_calls"] + brief_calls, 2, n_cat, concurrency, category_concurrency)},
        "unknown_price_models": [m for m, ok in known.items() if not ok],
        "assumptions": {"select_fraction": select_fraction, "lexical_rescue": lexical_rescue, "thinking": thinking, "mean_category_block_chars": mean_cat_chars,
                        "call_overhead_tokens": call_overhead_tokens, "concurrency": concurrency, "category_concurrency": category_concurrency},
    }


def calibration_applies(groups: dict, families: Iterable[str]) -> bool:
    """A measured per-category average describes a category that still needs EVERY requested family. It cannot describe categories that need only
    some (for example five categories awaiting a prosthesis re-judge): it would overstate that work several times over."""
    return set(groups) <= {tuple(families)}


def describe_groups(groups: dict) -> str:
    """{families still needed: n categories} -> 'dx for 64 categories; achi+mbs+prosthesis+dx for 148 categories'."""
    if not groups:
        return "nothing"
    return "; ".join(f"{'+'.join(need)} for {n:,} categor{'y' if n == 1 else 'ies'}" for need, n in groups.items())


def project_scope_remaining(categories: list, status: dict, families: Iterable[str], indices: dict, models: dict, thinking: str = "low",
                            concurrency: int = 20, brief_cached: Iterable[str] = (), **kw) -> dict:
    """project_scope(), but only for the families each category STILL NEEDS: a category finished for the code families but not for dx is projected
    for dx alone. Returns project_scope's dict for the whole request, with the routed / exhaustive TOTALS replaced by the sums over the groups, plus
    `groups` = {families still needed: n categories}. Extra keyword arguments go to project_scope."""
    families = tuple(families)
    cached = set(brief_cached)
    groups: dict[tuple, list] = {}
    for info in categories:
        st = status.get(info.category, {})
        done = set() if str(st.get("status", "")).startswith("skipped") else set(st.get("families_complete", []))
        need = tuple(f for f in families if f not in done)
        if need:
            groups.setdefault(need, []).append(info)

    def n_cached(cs: list) -> int:
        return sum(1 for c in cs if c.category in cached)

    full = project_scope(categories, 0, indices, families, models, thinking, concurrency, n_cached_briefs=n_cached(categories), **kw)
    if not groups:
        return {**full, "n_categories": 0, "groups": {}}
    parts = [project_scope(g, 0, indices, need, models, thinking, concurrency, n_cached_briefs=n_cached(g), **kw) for need, g in groups.items()]
    n_cat = sum(len(g) for g in groups.values())
    cc = kw.get("category_concurrency", DEFAULT_CATEGORY_CONCURRENCY)
    r = {k: sum(p["routed"][k] for p in parts) for k in ("calls_low", "calls_high", "cost_low", "cost_high")}
    depth = max(p["routed"]["depth"] for p in parts)
    r.update(depth=depth, hours_low=estimate_hours(r["calls_low"], depth, n_cat, concurrency, cc), hours_high=estimate_hours(r["calls_high"], depth, n_cat, concurrency, cc))
    e = {k: sum(p["exhaustive"][k] for p in parts) for k in ("calls", "cost", "cost_low", "cost_high", "pairs")}
    e["hours"] = estimate_hours(e["calls"], 2, n_cat, concurrency, cc)
    return {**full, "n_categories": n_cat, "routed": r, "exhaustive": e, "groups": {need: len(g) for need, g in groups.items()},
            "unknown_price_models": sorted({m for p in parts for m in p["unknown_price_models"]})}


def calibrate_from_pilot(ckpt: Checkpoint, families: Iterable[str] = DEFAULT_FAMILIES, min_categories: int = 3) -> Optional[dict]:
    """Replace assumptions with MEASURED per-category call counts, costs and call durations from an earlier run's logs.

    Only categories that COMPLETED ALL the requested families are used, so a one-family smoke test cannot drag the per-category average down; and
    only the requested families' calls are counted. The brief is counted only when a code family is requested -- a diagnosis-only run reuses the
    briefs. Returns None if fewer than `min_categories` qualify. Besides the mean it returns the spread (min / median / max cost per category)."""
    want = set(families)
    count_brief = bool(want & set(DEFAULT_FAMILIES))
    status = ckpt.load_status_all()
    full = {c for c, r in status.items() if r.get("status") == "complete" and want <= set(r.get("families_complete", []))}
    if len(full) < min_categories:
        return None
    by_stage: dict[str, dict] = {}
    per_cat_cost: dict[str, float] = {c: 0.0 for c in full}
    for r in ckpt.iter_run_log():
        cat = r.get("category")
        if cat not in full:
            continue
        fam, stage = r.get("family") or "", r.get("stage", "?")
        if (fam and fam not in want) or (stage == "brief" and not count_brief):
            continue
        st = by_stage.setdefault(stage, {"attempts": 0, "cost": 0.0, "dur_ms": 0.0, "n_dur": 0})
        cost = r.get("cost_usd") or 0.0
        st["attempts"] += 1
        st["cost"] += cost
        per_cat_cost[cat] += cost
        if r.get("ok") and r.get("duration_ms"):
            st["dur_ms"] += r["duration_ms"]
            st["n_dur"] += 1
    if not by_stage:
        return None
    n_cat = len(full)
    per_cat = {s: {"calls_per_category": v["attempts"] / n_cat, "cost_per_call": v["cost"] / max(v["attempts"], 1)} for s, v in by_stage.items()}
    costs = sorted(per_cat_cost.values())
    n_dur = sum(v["n_dur"] for v in by_stage.values())
    seconds = (sum(v["dur_ms"] for v in by_stage.values()) / n_dur / 1000) if n_dur else None
    return {"n_categories_observed": n_cat, "by_stage": per_cat, "seconds_per_call": seconds, "cost_per_category": sum(costs) / n_cat, "cost_min": costs[0],
            "cost_p50": statistics.median(costs), "cost_max": costs[-1], "calls_per_category": sum(v["calls_per_category"] for v in per_cat.values())}
