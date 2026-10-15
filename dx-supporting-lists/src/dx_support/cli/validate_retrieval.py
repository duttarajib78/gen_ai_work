"""dx-support validate-retrieval -- Pilot: how much does ROUTED candidate generation lose compared with an EXHAUSTIVE review, measured on real categories
before anything depends on it at scale. The production pipeline only judges the codes that tree routing and lexical rescue put in front of the judge, so
any supporting code those two stages never surface is lost silently. This puts a number on that.

Method (per sampled category x code family)
  1. REFERENCE: the judge reviews EVERY reviewable unit in the family (exhaustive mode, ~100 judge calls per category across the 3 families). Units it
     calls core or broad are the reference positives.
  2. ROUTED: the production candidate generator (brief -> tree routing -> lexical rescue) runs on the same category. No judging -- only the candidate set.
  3. RECALL = share of the reference positives inside the routed candidate set; also for tree routing alone and lexical rescue alone, plus the work saved.
  4. BASELINE: pure lexical retrieval from the category wording only (no brief, no model) at the SAME candidate budget -- how much the model-planned
     retrieval adds over plain text matching.

Read this before trusting the numbers: recall is measured against the SAME judge's exhaustive verdicts, so it isolates what routing loses, not whether the
judge is clinically right (that is `validate-judge`, against a clinician-reviewed gold file). The judge is also not perfectly repeatable -- a few reference
positives may flip between runs -- so small differences are noise. With a handful of categories the interval is wide: widen --n-categories across chapters.

Cost: roughly the exhaustive projection for the sampled categories (`dx-support estimate --categories ...`). The cumulative cap (--max-total-cost-usd,
default $50) applies. Nothing here writes to a production namespace: it uses its own (validation/retrieval).

    dx-support validate-retrieval --dry-run                          # harness check, no credentials, no spend
    dx-support validate-retrieval --n-categories 6 --yes
    dx-support validate-retrieval --categories H25,I48,K35 --families achi,mbs
    dx-support validate-retrieval --route-model claude-haiku-5-5 --categories H25,I48,K35   # does a cheaper router lose recall?
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from dx_support.cli.common import add_run_args, load_settings, make_backend, open_run
from dx_support.constants import AT_LEAST_BROAD, DEFAULT_FAMILIES, DEFAULT_THINKING, THINKING_PRESETS, parse_families
from dx_support.indexing.universe import load_universe
from dx_support.llm.gate import AdaptiveLimiter, LLMGate
from dx_support.pipeline import projection as proj
from dx_support.pipeline import stages as S
from dx_support.pipeline.context import RunContext
from dx_support.reference.categories import category_heads
from dx_support.reference.chapters import CONTEXT_ONLY_CHAPTERS, parse_chapters
from dx_support.storage.locks import RunLock

logger = logging.getLogger("dx_support.validate_retrieval")


def sample_categories(categories, requested, chapter, n, seed) -> list[str]:
    """Spread across chapters: one category per chapter in random chapter order, cycling until n are chosen -- a pilot that sits in one chapter says little about the rest."""
    chs = None if chapter is None else ({chapter} if isinstance(chapter, str) else set(chapter))
    eligible = [c for c, i in categories.items() if i.chapter not in CONTEXT_ONLY_CHAPTERS and (chs is None or i.chapter in chs)]
    if requested:
        want = {c.strip().upper() for c in requested}
        return [c for c in eligible if c in want]
    rng = random.Random(seed)
    by_ch: dict = defaultdict(list)
    for c in eligible:
        by_ch[categories[c].chapter].append(c)
    for lst in by_ch.values():
        rng.shuffle(lst)
    chapters = list(by_ch)
    rng.shuffle(chapters)
    picked: list[str] = []
    while len(picked) < min(n, len(eligible)):
        progressed = False
        for ch in chapters:
            if by_ch[ch] and len(picked) < n:
                picked.append(by_ch[ch].pop())
                progressed = True
        if not progressed:
            break
    return sorted(picked)


def metrics_for(index, cs, ref_verdicts: dict, baseline_ids: set[str]) -> dict:
    pos = {u for u, v in ref_verdicts.items() if v[0] in AT_LEAST_BROAD}
    core = {u for u, v in ref_verdicts.items() if v[0] == "core"}
    cand = set(cs.unit_sources)
    route = {u for u, s in cs.unit_sources.items() if "route" in s}
    lex = {u for u, s in cs.unit_sources.items() if "lexical" in s}

    def rec(a, b):
        return (len(a & b) / len(a)) if a else None
    return {
        "n_units": len(index.units), "n_ref_judged": len(ref_verdicts), "n_ref_positive": len(pos), "n_ref_core": len(core), "n_candidates": len(cand),
        "n_from_route": len(route), "n_lexical_only": len(cand - route), "recall_positive": rec(pos, cand), "recall_core": rec(core, cand),
        "recall_route_only": rec(pos, route), "recall_lexical_only": rec(pos, lex), "baseline_recall_same_budget": rec(pos, baseline_ids),
        "reduction_factor": (len(index.units) / len(cand)) if cand else None, "route_failures": cs.n_route_failures, "truncated_nodes": cs.n_truncated_nodes,
        "family_truncated": cs.family_truncated, "_pos": pos, "_cand": cand,
    }


async def run_validation(ctx_ex: RunContext, ctx_rt: RunContext, cats: list[str], families: tuple[str, ...], category_concurrency: int) -> list[dict]:
    rows: list[dict] = []
    sem = asyncio.Semaphore(category_concurrency)

    async def one(cat: str):
        async with sem:
            info = ctx_ex.categories[cat]
            await ctx_ex.ckpt.ahydrate(cat)
            brief = await S.get_brief(ctx_ex, cat)
            for fam in families:
                idx = ctx_ex.indices[fam]
                cs_ex = await S.generate_candidates(ctx_ex, cat, info, brief, idx)               # every unit
                await S.judge_units(ctx_ex, cat, info, brief, idx, cs_ex.unit_sources)           # reference review (stored + resumable)
                cs_rt = await S.generate_candidates(ctx_rt, cat, info, brief, idx)               # production candidate generation
                ref_v = {u: (r["support_tier"],) for (_c, _f, u), r in ctx_ex.ckpt.read_verdicts(cat, fam).items()}
                base_q = " ; ".join([info.title] + category_heads(info, 6))     # descriptor-only lexical baseline at the same budget as the routed candidate set
                baseline = {u for u, _ in idx.bm25().search(base_q, max(len(cs_rt.unit_sources), 1))}
                m = metrics_for(idx, cs_rt, ref_v, baseline)
                m.update({"category": cat, "family": fam, "chapter": info.chapter})
                rows.append(m)

    results = await asyncio.gather(*[one(c) for c in cats], return_exceptions=True)
    for c, r in zip(cats, results):
        if isinstance(r, BaseException):
            logger.error("Category %s failed during validation: %s", c, type(r).__name__)
    await ctx_ex.ckpt.aflush_run_log()
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dx-support validate-retrieval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p)
    p.add_argument("--n-categories", type=int, default=6)
    p.add_argument("--categories", type=str, default=None, help="Comma-separated exact categories (bypasses sampling).")
    p.add_argument("--chapter", "--chapters", type=parse_chapters, default=None, metavar="CHAPTERS", help="1-22 or Roman numerals: one, a list, or a range.")
    p.add_argument("--families", type=str, default=",".join(DEFAULT_FAMILIES), help="Default achi,mbs,prosthesis; add dx for the diagnosis family; `all` = every family.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--route-model", type=str, default=None, help="Model for brief+routing (default --model); the judge always uses --judge-model/--model.")
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--thinking", choices=list(THINKING_PRESETS), default=None, help=f"Default DX_THINKING, else '{DEFAULT_THINKING}'.")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--category-concurrency", type=int, default=3)
    p.add_argument("--max-budget-usd-per-call", type=float, default=None)
    p.add_argument("--max-total-cost-usd", type=float, default=50.0)
    p.add_argument("--no-lexical-rescue", action="store_true")
    p.add_argument("--output-dir", type=str, default="output/validation")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)
    try:
        families = parse_families(args.families)
    except ValueError as e:
        sys.exit(str(e))
    settings = load_settings(args)
    over = {k: v for k, v in {"model": args.model, "judge_model": args.judge_model, "thinking": args.thinking, "llm_concurrency": args.concurrency,
                              "max_budget_usd_per_call": args.max_budget_usd_per_call, "lexical_rescue": False if args.no_lexical_rescue else None}.items() if v is not None}
    if args.route_model:
        over["brief_model"] = over["route_model"] = args.route_model
    settings = settings.with_overrides(**over)
    universe = load_universe(settings.data_dir, settings.include_coarse_prostheses)
    cats = sample_categories(universe.categories, args.categories.split(",") if args.categories else None, args.chapter, args.n_categories, args.seed)
    if not cats:
        sys.exit("No eligible categories.")
    models = settings.stage_models()
    price_fn = proj.make_price_fn(settings.inference_profile_prefix, settings.model_catalog_path)
    projection = proj.project_scope([universe.categories[c] for c in cats], 0, universe.indices, families, models, settings.thinking, settings.llm_concurrency, price_fn=price_fn)
    print("=" * 72)
    print(f"Categories ({len(cats)}): {cats}")
    print(f"Reference = EXHAUSTIVE review: ~{projection['exhaustive']['calls']:,.0f} calls, {proj.fmt_usd(projection['exhaustive']['cost_low'])} - {proj.fmt_usd(projection['exhaustive']['cost_high'])}"
          + (" (stub: no spend)" if args.dry_run else ""))
    print(f"Models: brief/route={models['route']} judge={models['judge']} | thinking={settings.thinking} | cap ${args.max_total_cost_usd:,.0f}")
    print("=" * 72)
    if not args.yes and not args.dry_run and input("Type 'yes' to proceed: ").strip().lower() != "yes":
        sys.exit("Aborted -- nothing was sent to the model.")

    run = open_run(settings, "validation/dry_run" if args.dry_run else "validation/retrieval")
    backend = make_backend(settings, args.dry_run)
    gate = LLMGate(backend, AdaptiveLimiter(settings.llm_concurrency), args.max_total_cost_usd, run_log=run.ckpt.log_run, backoff_base_s=0.0 if args.dry_run else 1.5,
                   max_retries=settings.max_call_retries, max_throttle_retries=settings.max_throttle_retries, call_timeout_s=None if args.dry_run else settings.llm_call_timeout_s + 30)
    common = dict(categories=universe.categories, indices=universe.indices, gate=gate, ckpt=run.ckpt, families=families, models=models, thinking=settings.thinking,
                  lexical_rescue=settings.lexical_rescue, max_budget_usd_per_call=settings.max_budget_usd_per_call, max_tokens=dict(settings.max_tokens))
    ctx_ex = RunContext(mode="exhaustive", **common)
    ctx_rt = RunContext(mode="routed", **common)
    try:
        with RunLock(run.store):
            rows = asyncio.run(run_validation(ctx_ex, ctx_rt, cats, families, args.category_concurrency))
    finally:
        run.close()
        close = getattr(backend, "close", None)
        if close:
            close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("retrieval_validation_dryrun.csv" if args.dry_run else "retrieval_validation.csv")
    cols = ["category", "family", "chapter", "n_units", "n_ref_judged", "n_ref_positive", "n_ref_core", "n_candidates", "n_from_route", "n_lexical_only", "recall_positive",
            "recall_core", "recall_route_only", "recall_lexical_only", "baseline_recall_same_budget", "reduction_factor", "route_failures", "truncated_nodes", "family_truncated"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def pooled(sel):
        hit = tot = 0
        for r in sel:
            hit += len(r["_pos"] & r["_cand"])
            tot += len(r["_pos"])
        return (100.0 * hit / tot) if tot else float("nan"), tot

    print("=" * 72)
    if args.dry_run:
        print("DRY RUN -- offline stub, NOT real model behaviour. Sanity-checks the harness only.")
    print(f"Spend: ${gate.total_cost:,.2f} over {gate.n_calls:,} calls | failed attempts: {gate.n_failed_attempts}")
    for fam in families:
        sel = [r for r in rows if r["family"] == fam]
        if not sel:
            continue
        rec, n_pos = pooled(sel)
        base_hit = sum(len(r["_pos"]) * (r["baseline_recall_same_budget"] or 0) for r in sel)
        base = 100.0 * base_hit / n_pos if n_pos else float("nan")
        red = sum(r["reduction_factor"] or 0 for r in sel) / len(sel)
        print(f"{fam:>10}: RECALL of reference positives (core+broad) in the routed candidate set = {rec:5.1f}%  (n={n_pos} positives over {len(sel)} categories) "
              f"| lexical-only baseline at same budget {base:5.1f}% | work reduction ~{red:,.0f}x")
    rec, n_pos = pooled(rows)
    print("-" * 72)
    print(f"OVERALL recall: {rec:.1f}% of {n_pos} reference positives. Categories with recall < 80% (look at these first):")
    for r in sorted(rows, key=lambda r: (r["recall_positive"] if r["recall_positive"] is not None else 2)):
        if r["recall_positive"] is not None and r["recall_positive"] < 0.8:
            print(f"   {r['category']} {r['family']}: {r['recall_positive']:.0%} ({len(r['_pos'] & r['_cand'])}/{len(r['_pos'])}) -- consider --mode exhaustive for this category")
    print(f"Per-category detail: {out_csv}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
