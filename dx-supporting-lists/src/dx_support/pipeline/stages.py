"""The pipeline, per category and per code family, in three stages:

  0. BRIEF   (1 call/category)   the model writes a short clinical work-up brief and plain search phrases. One brief drives every family, which
                                 keeps their lists consistent with each other.
  1. ROUTE   (~2-4 calls/family) the code family is a small-fan-out tree. The model sees one level at a time and picks every branch that could
                                 hold a supporting code ("be inclusive"). In PARALLEL a lexical (BM25) search over all units, driven by the brief's
                                 search phrases, adds candidates the tree walk may have missed ("lexical rescue").
  2. JUDGE   (1-3 calls/family)  every candidate is classified core / broad / unclear / not_supporting, with an evidence type and a one-sentence rationale.

Routing can lose recall that an exhaustive review would not. That is MEASURED, not assumed (`dx-support validate-retrieval`); `--mode exhaustive`
runs the full-universe review for a category you want to be certain about.

Governance: a verdict is a screening signal, not a coding determination. Every verdict (merged or not) stays in the audit. The model has no tools,
reads only prompt text built here, and every id it returns is validated against the ids it was sent. Everything is resumable: each brief, routing
decision and verdict is checkpointed the moment it exists, and a re-run never re-pays for a decision already made.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from dx_support.indexing.models import FamilyIndex
from dx_support.llm.types import LLMCallError, LLMRequest, StopRequested
from dx_support.pipeline import prompts as P
from dx_support.pipeline.constants import (DEFAULT_CATEGORY_CONCURRENCY, JUDGE_BATCH_SIZE, JUDGE_RETRY_BATCHES, LEXICAL_TOP_K, MAX_UNITS_PER_FAMILY,
                                           MAX_UNITS_PER_LEAF_NODE, ROUTE_MAX_NODES_PER_CALL)
from dx_support.pipeline.context import RunContext
from dx_support.pipeline.parsing import parse_brief, parse_judge, parse_route
from dx_support.reference.categories import CategoryInfo, category_heads
from dx_support.reference.chapters import CONTEXT_ONLY_CHAPTERS
from dx_support.storage.checkpoint import now_ts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def split_even(seq: list, max_size: int) -> list[list]:
    if not seq:
        return []
    n_chunks = math.ceil(len(seq) / max_size)
    size = math.ceil(len(seq) / n_chunks)
    return [seq[i: i + size] for i in range(0, len(seq), size)]


def route_key(family: str, level: int, node_ids: list[str]) -> str:
    return hashlib.sha1(f"{family}|{level}|{','.join(sorted(node_ids))}".encode("utf-8")).hexdigest()[:16]   # noqa: S324 - identifier only


def lexical_query(info: CategoryInfo, brief: dict, family: str = "achi") -> str:
    if family == "dx":     # related DIAGNOSES share wording with the target itself, not with its work-up
        return " ; ".join([info.title] + category_heads(info, 8))
    parts = (list(brief.get("search_terms", [])) + list(brief.get("treatment", [])) + list(brief.get("investigation", []))
             + list(brief.get("devices", [])) + list(brief.get("examination", [])) + category_heads(info, 6))
    return " ; ".join(parts)


def category_done_from_status(status: dict, cat: str, families: Iterable[str], mode: str = "routed") -> bool:
    """True if `cat` already finished ALL the requested families in this mode (`status` = category -> last status record). A category complete for
    only one family (e.g. an ACHI-only smoke test) is NOT done for a three-family run."""
    st = status.get(cat, {})
    return st.get("status") == "complete" and set(families) <= set(st.get("families_complete", [])) and st.get("mode", mode) == mode


def families_done(status_rec: Optional[dict], mode: str = "routed") -> set[str]:
    st = status_rec or {}
    if str(st.get("status", "")).startswith("skipped") or st.get("mode", mode) != mode:
        return set()
    return set(st.get("families_complete", []))


def concurrency_hint(concurrency: int, category_concurrency: int, n_families: int) -> Optional[str]:
    """A category has about one call in flight per family at a time, so category_concurrency x families bounds the calls in flight. When that is
    below --concurrency the gate can never fill -- typical of a single-family run such as `--families dx`."""
    in_flight = category_concurrency * max(n_families, 1)
    if in_flight >= concurrency:
        return None
    need = math.ceil(concurrency / max(n_families, 1))
    return (f"only about {in_flight} calls can be in flight (--category-concurrency {category_concurrency} x {n_families} "
            f"famil{'y' if n_families == 1 else 'ies'}) but --concurrency is {concurrency}: raise --category-concurrency to about {need} to use it.")


# ---------------------------------------------------------------------------
# Stage 0: brief
# ---------------------------------------------------------------------------
async def get_brief(ctx: RunContext, cat: str) -> dict:
    cached = ctx.ckpt.briefs.get(cat)
    if cached:
        return cached
    info = ctx.categories[cat]
    if not ctx.vocab:
        ctx.vocab = P.vocabulary_hint(ctx.indices)
    prompt = P.build_brief_prompt(ctx.vocab, info, ctx.neighbours(cat))
    req = LLMRequest(
        stage="brief", system_prompt=P.SYSTEM_PROMPT_BRIEF, prompt=prompt, schema=P.brief_schema(), model=ctx.models["brief"], thinking=ctx.thinking,
        max_budget_usd=ctx.max_budget_usd_per_call, max_tokens=ctx.max_tokens["brief"], cache_prefix_chars=P.cache_prefix_len(prompt),
        label=f"{cat}/brief", category=cat, meta={"category_text": ctx.category_text(info)},
    )
    res = await ctx.gate.call(req)
    brief = parse_brief(res.structured)
    if brief is None:
        raise LLMCallError(f"unusable brief for {cat}", cost_usd=0.0, kind="transient")
    await ctx.ckpt.write_brief(cat, brief)
    return brief


# ---------------------------------------------------------------------------
# Stage 1: candidate generation (routing + lexical rescue)
# ---------------------------------------------------------------------------
@dataclass
class CandidateSet:
    unit_sources: dict[str, set[str]] = field(default_factory=dict)   # unit_id -> {"route","lexical","exhaustive"}
    n_route_failures: int = 0
    n_truncated_nodes: int = 0
    family_truncated: bool = False


async def _route_chunk(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex, level: int, chunk: list[str]) -> list[str]:
    key = route_key(index.family, level, chunk)
    cached = ctx.ckpt.routes.get((cat, index.family, key))
    if cached is not None:
        valid = set(chunk)
        return [i for i in cached if i in valid]
    nodes = [index.nodes[i] for i in chunk]
    prompt = P.build_route_prompt(index.family, level, index.n_levels, index.level_names[level], nodes, info, ctx.neighbours(cat), brief)
    req = LLMRequest(
        stage="route", system_prompt=P.route_system_prompt(index.family), prompt=prompt, schema=P.route_schema(), model=ctx.models["route"],
        thinking=ctx.thinking, max_budget_usd=ctx.max_budget_usd_per_call, max_tokens=ctx.max_tokens["route"], cache_prefix_chars=P.cache_prefix_len(prompt),
        label=f"{cat}/{index.family}/route-L{level + 1}", category=cat, family=index.family,
        meta={"valid_ids": list(chunk), "node_texts": {n.node_id: n.label for n in nodes}, "category_text": ctx.category_text(info),
              "search_terms": brief.get("search_terms", [])},
    )
    res = await ctx.gate.call(req)
    selected = parse_route(res.structured, set(chunk))
    if selected is None:
        raise LLMCallError(f"unusable routing response for {cat}/{index.family}/L{level + 1}", cost_usd=0.0, kind="transient")
    await ctx.ckpt.write_route(cat, index.family, level, key, selected)
    return selected


async def _select_nodes(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex, level: int,
                        frontier: list[str]) -> tuple[list[str], int]:
    chunks = split_even(frontier, ROUTE_MAX_NODES_PER_CALL)
    results = await asyncio.gather(*[_route_chunk(ctx, cat, info, brief, index, level, c) for c in chunks], return_exceptions=True)
    picked: list[str] = []
    failed = 0
    stop: Optional[BaseException] = None
    for chunk, r in zip(chunks, results):
        if isinstance(r, StopRequested):
            stop = r
        elif isinstance(r, asyncio.CancelledError):
            raise r
        elif isinstance(r, BaseException):
            failed += 1
            logger.error("Routing chunk failed for %s/%s level %d (%d groups): %s", cat, index.family, level + 1, len(chunk), type(r).__name__)
        else:
            picked.extend(r)
    if stop is not None:
        raise stop
    return list(dict.fromkeys(picked)), failed


async def route_family(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex) -> tuple[list[str], int]:
    """Walk the family's tree. Returns (leaf-level node ids selected, number of failed routing chunks).

    A branch with exactly one child is passed through without asking (the parent's selection already covers it). An empty selection at any
    level ends that walk -- 'nothing here supports this category' is a real answer."""
    frontier, auto = list(index.root_ids), []
    last = index.n_levels - 1
    failures = 0
    for level in range(index.n_levels):
        picked: list[str] = []
        if frontier:
            picked, failed = await _select_nodes(ctx, cat, info, brief, index, level, frontier)
            failures += failed
        picked = list(dict.fromkeys(picked + auto))
        if level == last:
            return picked, failures
        frontier, auto = [], []
        for nid in picked:
            children = index.nodes[nid].children
            (auto if len(children) == 1 else frontier).extend(children)
        frontier = list(dict.fromkeys(frontier))
        if not frontier and not auto:
            return [], failures
    return [], failures


def _expand_with_cap(index: FamilyIndex, leaf_ids: list[str], query: str) -> tuple[list[str], int]:
    """Units under the selected leaf nodes. A leaf node bigger than MAX_UNITS_PER_LEAF_NODE keeps its best BM25 matches (topped up in code order)
    and is COUNTED as truncated -- surfaced in the status record and the QA sheet, never silent."""
    out: dict[str, None] = {}
    truncated = 0
    for nid in leaf_ids:
        uids = index.nodes[nid].unit_ids
        if len(uids) > MAX_UNITS_PER_LEAF_NODE:
            truncated += 1
            ranked = [u for u, _ in index.bm25().search(query, len(uids), allowed=set(uids))]
            rest = [u for u in uids if u not in set(ranked)]
            uids = (ranked + rest)[:MAX_UNITS_PER_LEAF_NODE]
        for u in uids:
            out.setdefault(u, None)
    return list(out), truncated


async def generate_candidates(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex) -> CandidateSet:
    cs = CandidateSet()
    if ctx.mode == "exhaustive":
        for uid in sorted(index.units):
            cs.unit_sources[uid] = {"exhaustive"}
        if index.family == "dx":
            cs.unit_sources.pop(cat, None)             # a category is never its own supporting diagnosis
        return cs
    leaf_ids, cs.n_route_failures = await route_family(ctx, cat, info, brief, index)
    query = lexical_query(info, brief, index.family)
    route_units, cs.n_truncated_nodes = _expand_with_cap(index, leaf_ids, query)
    for uid in route_units:
        cs.unit_sources.setdefault(uid, set()).add("route")
    if ctx.lexical_rescue:
        for uid, _score in index.bm25().search(query, LEXICAL_TOP_K[index.family]):
            cs.unit_sources.setdefault(uid, set()).add("lexical")
    if index.family == "dx":
        cs.unit_sources.pop(cat, None)
    cap = MAX_UNITS_PER_FAMILY[index.family]
    if len(cs.unit_sources) > cap:
        cs.family_truncated = True
        ranked = [u for u, _ in index.bm25().search(query, len(index.units), allowed=set(cs.unit_sources))]
        keep = (ranked + [u for u in sorted(cs.unit_sources) if u not in set(ranked)])[:cap]
        cs.unit_sources = {u: cs.unit_sources[u] for u in keep}
    return cs


# ---------------------------------------------------------------------------
# Stage 2: judge
# ---------------------------------------------------------------------------
async def _judge_call(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex, unit_ids: list[str],
                      label_suffix: str = "", retry: bool = False) -> dict[str, dict]:
    units = [index.units[u] for u in unit_ids]
    req = LLMRequest(
        stage="judge", system_prompt=P.judge_system_prompt(index.family),
        prompt=P.build_judge_prompt(index.family, units, info, ctx.neighbours(cat), brief, retry),
        schema=P.judge_schema(info.is_large, index.family), model=ctx.models["judge"], thinking=ctx.thinking,
        max_budget_usd=ctx.max_budget_usd_per_call, max_tokens=ctx.max_tokens["judge"],
        label=f"{cat}/{index.family}/judge{label_suffix}", category=cat, family=index.family,
        meta={"unit_texts": {u.unit_id: u.text for u in units}, "category_text": ctx.category_text(info), "search_terms": brief.get("search_terms", []),
              "member_codes": list(info.member_codes), "include_member_scope": info.is_large},
    )
    res = await ctx.gate.call(req)
    return parse_judge(res.structured, set(unit_ids), set(info.member_codes), index.family)


async def _judge_chunk(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex, chunk: list[str],
                       sources: dict[str, set[str]]) -> int:
    """Judges one chunk, re-judges skipped ids in smaller and smaller batches (JUDGE_RETRY_BATCHES), checkpoints what it got. Returns how many
    units are STILL without a verdict. Raises StopRequested; other failures propagate to the caller."""
    verdicts = await _judge_call(ctx, cat, info, brief, index, chunk)
    missing = [u for u in chunk if u not in verdicts]
    for size in JUDGE_RETRY_BATCHES:
        if not missing or ctx.gate.stop:
            break
        subs = split_even(missing, size)
        results = await asyncio.gather(*[_judge_call(ctx, cat, info, brief, index, sub, "-retry", True) for sub in subs], return_exceptions=True)
        stop: Optional[BaseException] = None
        for sub, r in zip(subs, results):
            if isinstance(r, StopRequested):
                stop = r
            elif isinstance(r, asyncio.CancelledError):
                raise r
            elif isinstance(r, BaseException):      # keep what the other passes produced
                logger.warning("Re-judging %d skipped unit(s) failed for %s/%s: %s", len(sub), cat, index.family, type(r).__name__)
            else:
                verdicts.update(r)
        if stop is not None:
            raise stop
        missing = [u for u in chunk if u not in verdicts]
    records = []
    ts = now_ts()
    for uid, v in verdicts.items():
        records.append({
            "category": cat, "family": index.family, "unit_id": uid, "support_tier": v["support_tier"], "evidence_type": v["evidence_type"],
            "rationale": v["rationale"], "member_scope": v["member_scope"], "source": "+".join(sorted(sources.get(uid, {"unknown"}))),
            "model": ctx.models["judge"], "ts": ts,
        })
        ctx.tier_counts[v["support_tier"]] += 1
    if records:
        await ctx.ckpt.write_verdicts(records)
    return len(missing)


async def judge_units(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex, unit_sources: dict[str, set[str]]) -> dict:
    pending = [u for u in sorted(unit_sources) if (cat, index.family, u) not in ctx.ckpt.verdict_keys]
    chunks = split_even(pending, JUDGE_BATCH_SIZE[index.family])
    results = await asyncio.gather(*[_judge_chunk(ctx, cat, info, brief, index, c, unit_sources) for c in chunks], return_exceptions=True)
    n_missing = failed_chunks = 0
    stop: Optional[BaseException] = None
    for chunk, r in zip(chunks, results):
        if isinstance(r, StopRequested):
            stop = r
            n_missing += len(chunk)
        elif isinstance(r, asyncio.CancelledError):
            raise r
        elif isinstance(r, BaseException):
            failed_chunks += 1
            n_missing += len(chunk)
            logger.error("Judge chunk failed for %s/%s (%d units): %s", cat, index.family, len(chunk), type(r).__name__)
        else:
            n_missing += r
    if stop is not None:
        raise stop
    return {"n_pending_at_start": len(pending), "n_missing": n_missing, "n_failed_chunks": failed_chunks}


async def judge_batch(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, index: FamilyIndex, unit_ids: Iterable[str]) -> dict[str, dict]:
    """Judges `unit_ids` and RETURNS the verdicts without writing anything to the checkpoint -- for the validation scripts, which must never
    pollute a production checkpoint with experimental verdicts."""
    ids = sorted(set(unit_ids))
    chunks = split_even(ids, JUDGE_BATCH_SIZE[index.family])
    results = await asyncio.gather(*[_judge_call(ctx, cat, info, brief, index, c, "-val") for c in chunks], return_exceptions=True)
    out: dict[str, dict] = {}
    for chunk, r in zip(chunks, results):
        if isinstance(r, StopRequested):
            raise r
        if isinstance(r, BaseException):
            logger.error("Validation judge chunk failed for %s/%s (%d units): %s", cat, index.family, len(chunk), type(r).__name__)
            continue
        out.update(r)
    return out


# ---------------------------------------------------------------------------
# Per-category pipeline + run loop
# ---------------------------------------------------------------------------
async def process_family(ctx: RunContext, cat: str, info: CategoryInfo, brief: dict, family: str) -> dict:
    index = ctx.indices[family]
    cs = await generate_candidates(ctx, cat, info, brief, index)
    stats = await judge_units(ctx, cat, info, brief, index, cs.unit_sources)
    stats.update({
        "n_candidates": len(cs.unit_sources), "n_from_route": sum(1 for s in cs.unit_sources.values() if "route" in s),
        "n_lexical_only": sum(1 for s in cs.unit_sources.values() if s == {"lexical"}), "route_failures": cs.n_route_failures,
        "truncated_nodes": cs.n_truncated_nodes, "family_truncated": cs.family_truncated,
    })
    stats["complete"] = stats["n_missing"] == 0 and cs.n_route_failures == 0
    return stats


async def process_category(ctx: RunContext, cat: str) -> str:
    """Returns an outcome label: complete | partial | skipped_context_only | skipped_by_brief | already_complete."""
    info = ctx.categories[cat]
    await ctx.ckpt.ahydrate(cat)
    prev = ctx.ckpt.status.get(cat, {})
    context_only = info.chapter in CONTEXT_ONLY_CHAPTERS
    if context_only and not ctx.include_context_chapters:
        if prev.get("status") != "skipped_context_only":
            await ctx.ckpt.write_status({"category": cat, "status": "skipped_context_only", "chapter": info.chapter, "families_complete": []})
        return "skipped_context_only"
    done = [] if prev.get("status", "").startswith("skipped") else list(prev.get("families_complete", []))
    if prev.get("mode", ctx.mode) != ctx.mode:
        done = []
    todo = [f for f in ctx.families if f not in done]
    if not todo:
        return "already_complete"
    # keep earlier families' stats when a later run (e.g. --families dx) adds another family to the same category
    prev_fams = {} if (prev.get("status", "").startswith("skipped") or prev.get("mode", ctx.mode) != ctx.mode) else dict(prev.get("families") or {})

    cost0, calls0 = ctx.gate.total_cost, ctx.gate.n_calls
    try:
        brief = await get_brief(ctx, cat)
    except StopRequested:
        raise
    except Exception as e:                      # noqa: BLE001 - recorded in the status, never swallowed silently
        await ctx.ckpt.write_status({"category": cat, "status": "partial", "mode": ctx.mode, "families_complete": done,
                                     "error": f"brief: {type(e).__name__}: {str(e)[:160]}", "chapter": info.chapter, "fingerprint": ctx.fingerprint})
        return "partial"
    if ctx.brief_skip and brief.get("typically_no_procedural_evidence"):
        await ctx.ckpt.write_status({"category": cat, "status": "skipped_by_brief", "mode": ctx.mode, "families_complete": [], "chapter": info.chapter,
                                     "brief_summary": brief["clinical_summary"][:200]})
        return "skipped_by_brief"

    results = await asyncio.gather(*[process_family(ctx, cat, info, brief, f) for f in todo], return_exceptions=True)
    fam_stats: dict[str, dict] = {}
    complete = list(done)
    stopped: Optional[BaseException] = None
    for fam, r in zip(todo, results):
        if isinstance(r, StopRequested):
            stopped = r
            fam_stats[fam] = {"complete": False, "error": "stopped"}
        elif isinstance(r, asyncio.CancelledError):
            raise r
        elif isinstance(r, BaseException):
            fam_stats[fam] = {"complete": False, "error": f"{type(r).__name__}: {str(r)[:160]}"}
            logger.error("Family %s failed for %s: %s", fam, cat, type(r).__name__)
        else:
            fam_stats[fam] = r
            if r["complete"]:
                complete.append(fam)
    full = set(ctx.families) <= set(complete)
    await ctx.ckpt.write_status({
        "category": cat, "status": "complete" if full else "partial", "mode": ctx.mode, "chapter": info.chapter, "families_complete": sorted(set(complete)),
        "families": {**prev_fams, **fam_stats}, "calls": ctx.gate.n_calls - calls0, "cost_usd_approx": round(ctx.gate.total_cost - cost0, 5),
        "fingerprint": ctx.fingerprint,
    })
    if stopped is not None and not full:
        raise stopped
    return "complete" if full else "partial"


async def run_build(ctx: RunContext, category_ids: list[str], category_concurrency: int = DEFAULT_CATEGORY_CONCURRENCY) -> dict:
    """Processes categories through a bounded pool. Model calls are bounded separately by the gate's limiter, so category_concurrency only controls
    how many categories are in flight (memory / checkpoint granularity). Safe to interrupt at any point and re-run."""
    queue: asyncio.Queue = asyncio.Queue()
    for c in category_ids:
        queue.put_nowait(c)
    outcomes: dict[str, int] = {}
    errors = 0
    total = len(category_ids)
    done_count = 0
    t0 = time.monotonic()

    async def worker() -> None:
        nonlocal errors, done_count
        while True:
            if ctx.gate.stop:
                return
            try:
                cat = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                outcome = await process_category(ctx, cat)
            except StopRequested:
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:                  # noqa: BLE001
                errors += 1
                logger.error("Category %s failed: %s", cat, type(e).__name__)
                continue
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            done_count += 1
            if outcome in ("complete", "partial") or done_count % 100 == 0:
                logger.info("[%d/%d] %s %s | calls %d | cost $%.2f | %.0fs", done_count, total, cat, outcome, ctx.gate.n_calls, ctx.gate.total_cost,
                            time.monotonic() - t0)

    try:
        await asyncio.gather(*[worker() for _ in range(max(1, min(category_concurrency, max(total, 1))))])
    finally:
        await ctx.ckpt.aflush_run_log()
    return {
        "n_categories_requested": total, "outcomes": outcomes, "n_category_errors": errors, "cost_usd_this_run": ctx.gate.total_cost,
        "n_calls_this_run": ctx.gate.n_calls, "n_failed_attempts": ctx.gate.n_failed_attempts, "by_stage": ctx.gate.by_stage, "stopped": ctx.gate.stop,
        "stop_reason": ctx.gate.stop_reason, "fatal_error": ctx.gate.fatal_error, "stopped_on_cost_cap": ctx.gate.stop and ctx.gate.cost_cap_hit,
        "tier_counts_this_run": dict(ctx.tier_counts),
    }
