"""dx-support estimate -- RUN THIS BEFORE `dx-support build`. Makes no model calls and needs no AWS credentials (unless --calibrate reads an S3 store): it
loads the reference files and projects the run's size, cost and time from the REAL universe (code counts, tree sizes, text lengths) and a few stated
assumptions.

What it can and cannot tell you: the universe sizes and the exhaustive-review projection are computed from the data and are reliable to within the
token/price assumptions. The ROUTED projection also depends on how much of each tree level a model keeps (--select-fraction), which cannot be known
without calling the model. Use the range to decide whether to pilot, then run a pilot and re-run with --calibrate to replace the assumptions with
measured calls-per-category and cost-per-call. Prices come from the model catalog and are point-in-time -- verify before budgeting.

    dx-support estimate                          # everything, routed + exhaustive side by side
    dx-support estimate --chapter IX             # one chapter
    dx-support estimate --max-categories 5       # what a small pilot costs
    dx-support estimate --show-universe          # exclusions, tree sizes, category facts -- and stop
    dx-support estimate --calibrate              # use measured numbers from a pilot in the store
    dx-support estimate --route-model claude-haiku-5-5
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from dx_support.cli.common import add_run_args, load_settings, namespace_path, open_run
from dx_support.constants import DEFAULT_FAMILIES, DEFAULT_THINKING, FAMILIES, THINKING_PRESETS, parse_families
from dx_support.indexing.families import index_stats
from dx_support.indexing.universe import load_universe, select_categories
from dx_support.pipeline import projection as proj
from dx_support.pipeline import stages as S
from dx_support.pipeline.constants import (ASSUMED_SELECT_FRACTION, CALL_OVERHEAD_INPUT_TOKENS, DEFAULT_CATEGORY_CONCURRENCY, MAX_UNITS_PER_LEAF_NODE,
                                           SECONDS_PER_CALL_ESTIMATE)
from dx_support.reference.categories import LARGE_CATEGORY_MIN_MEMBERS
from dx_support.reference.chapters import CHAPTER_TITLES, CONTEXT_ONLY_CHAPTERS, VALID_CHAPTER_IDS, parse_chapters

logger = logging.getLogger("dx_support.estimate")


def show_universe(universe) -> None:
    categories, indices, report = universe.categories, universe.indices, universe.report
    print("=" * 72)
    print("REFERENCE UNIVERSE (computed from the data directory)")
    print("-" * 72)
    chap = Counter(i.chapter for i in categories.values())
    ctx_n = sum(n for c, n in chap.items() if c in CONTEXT_ONLY_CHAPTERS)
    print(f"ICD-10-AM: {sum(len(i.members) for i in categories.values()):,} leaf codes -> {len(categories):,} three-character categories (derived; the file has no category rows)")
    print(f"  with their own row in the file: {sum(i.has_own_row for i in categories.values())} | large (>{LARGE_CATEGORY_MIN_MEMBERS} member codes, member-level scoping): {sum(i.is_large for i in categories.values())}")
    print(f"  context-only chapters {CONTEXT_ONLY_CHAPTERS}: {ctx_n} categories skipped by default")
    print("  by chapter: " + ", ".join(f"{c or '?'}={n}" for c, n in sorted(chap.items(), key=lambda kv: (kv[0] is None, VALID_CHAPTER_IDS.index(kv[0]) if kv[0] in VALID_CHAPTER_IDS else 99))))
    for fam in FAMILIES:
        st = index_stats(indices[fam])
        print("-" * 72)
        print(f"{fam.upper()}: {st['n_units']:,} reviewable units covering {st['n_codes_in_units']:,} codes | excluded: {st['n_excluded']:,} {st['excluded_by_reason']}")
        print(f"  routing tree: {st['nodes_by_level']} | max children per node: {st['max_fanout']} | largest leaf branch: {st['max_units_per_leaf_node']} units (cap {MAX_UNITS_PER_LEAF_NODE})")
        print(f"  mean unit text: {st['mean_unit_chars']:.0f} chars | top-level catalogue: {st['root_label_chars']:,} chars")
    r = report["prostheses"]
    print("-" * 72)
    print(f"PROSTHESES file cleaning: {r['rows']:,} rows -> {r['unique_codes']:,} unique codes ({r['labelled_codes']:,} with a device-type hierarchy, {r['coarse_only_codes']} coarse-label only)")
    print(f"  column-shifted rows recovered: {r['rows_column_shifted_recovered']} | unrecoverable rows: {r['rows_unrecoverable']} | codes whose labels differ between extracts: "
          f"{r['codes_with_conflicting_labels']} | codes in the latest extract: {r['codes_in_latest_extract']:,} (detected: {r['latest_extract_detected']})")
    print("=" * 72)


def list_chapters(categories) -> None:
    by_ch: dict = {}
    for cat, info in categories.items():
        by_ch.setdefault(info.chapter, []).append(cat)
    print("=" * 72)
    print(f"{'No.':>3}  {'Roman':<6} {'Categories in file':<19} {'Count':>5}  {'Default':<8} Standard ICD-10 chapter name (convenience only)")
    for n, cid in enumerate(VALID_CHAPTER_IDS, start=1):
        cs = sorted(by_ch.get(cid, []))
        rng = f"{cs[0]}-{cs[-1]}" if cs else "-"
        print(f"{n:>3}  {cid:<6} {rng:<19} {len(cs):>5}  {'SKIP' if cid in CONTEXT_ONLY_CHAPTERS else 'build':<8} {CHAPTER_TITLES[cid]}")
    print("-" * 72)
    print("Use either form:  dx-support build --chapter 9   or   --chapter IX; several at once:  --chapter 9,10   or   --chapter 9-11. SKIP = context-only (circumstances, not conditions).")
    print("=" * 72)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dx-support estimate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p)
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--chapter", "--chapters", type=parse_chapters, default=None, metavar="CHAPTERS", help="1-22 or Roman numerals: one, a list, or a range (9, 9,10, IX,X, 9-11).")
    p.add_argument("--max-categories", type=int, default=None)
    p.add_argument("--families", type=str, default=",".join(DEFAULT_FAMILIES), help="Default achi,mbs,prosthesis; add dx for the diagnosis family; `all` = every family.")
    p.add_argument("--model", type=str, default=None); p.add_argument("--brief-model", type=str, default=None)
    p.add_argument("--route-model", type=str, default=None); p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--thinking", choices=list(THINKING_PRESETS), default=None, help=f"Default DX_THINKING, else '{DEFAULT_THINKING}'.")
    p.add_argument("--concurrency", type=int, default=20, help="Max in-flight model calls (as for `build`).")
    p.add_argument("--category-concurrency", type=int, default=DEFAULT_CATEGORY_CONCURRENCY, help="Max categories in flight. Each category's stages run in sequence, so this -- not just --concurrency -- bounds wall-clock time.")
    p.add_argument("--call-overhead-tokens", type=int, default=CALL_OVERHEAD_INPUT_TOKENS, help="Fixed input-token overhead assumed per call (0 on Bedrock Converse; the earlier Agent SDK build measured ~20,000).")
    p.add_argument("--select-fraction", type=str, default=",".join(str(x) for x in ASSUMED_SELECT_FRACTION), help="LOW,HIGH share of each routing frontier the model keeps (assumption).")
    p.add_argument("--include-context-chapters", action="store_true"); p.add_argument("--include-coarse-prostheses", action="store_true")
    p.add_argument("--no-lexical-rescue", action="store_true")
    p.add_argument("--list-chapters", action="store_true", help="Print the chapters (number, Roman numeral, categories, default) and stop.")
    p.add_argument("--show-universe", action="store_true", help="Print the universe facts (exclusions, trees, categories) and stop.")
    p.add_argument("--calibrate", action="store_true", help="Also project from MEASURED numbers in the store (needs a pilot of >= 3 categories).")
    args = p.parse_args(argv)
    try:
        families = parse_families(args.families)
    except ValueError as e:
        sys.exit(str(e))
    settings = load_settings(args, quiet=True)
    over = {k: v for k, v in {"model": args.model, "brief_model": args.brief_model, "route_model": args.route_model, "judge_model": args.judge_model, "thinking": args.thinking}.items() if v}
    settings = settings.with_overrides(**over)
    universe = load_universe(settings.data_dir, settings.include_coarse_prostheses)
    categories, indices = universe.categories, universe.indices
    if args.list_chapters:
        list_chapters(categories)
        return 0
    if args.show_universe:
        show_universe(universe)
        return 0

    selected, missing = select_categories(categories, args.categories.split(",") if args.categories else None, args.chapter, args.max_categories)
    if missing:
        print(f"WARNING: not in diagnosis_codes.tsv, skipped: {missing}")
    skipped_ctx = [] if args.include_context_chapters else [c for c in selected if categories[c].chapter in CONTEXT_ONLY_CHAPTERS]
    in_scope = [categories[c] for c in selected if c not in set(skipped_ctx)]
    lo, hi = (float(x) for x in args.select_fraction.split(","))
    models = settings.stage_models()
    price_fn = proj.make_price_fn(settings.inference_profile_prefix, settings.model_catalog_path)

    status: dict = {}
    cached_briefs: set = set()
    run = None
    try:
        run = open_run(settings, namespace_path(args.namespace, args.dry_run))
        status = run.ckpt.load_status_all()
        cached_briefs = run.ckpt.cached_brief_ids()
    except Exception as e:                                           # noqa: BLE001 - an unreachable store must not stop a projection
        print(f"NOTE: could not read the result store ({type(e).__name__}); projecting as if nothing had been built yet.")
    done_ids = {i.category for i in in_scope if S.category_done_from_status(status, i.category, families)}
    in_scope = [i for i in in_scope if i.category not in done_ids]
    if not in_scope:
        print(f"Nothing left to build: all {len(done_ids):,} categories in scope are already complete for {','.join(families)}.")
        return 0
    projection = proj.project_scope_remaining(in_scope, status, families, indices, models, settings.thinking, args.concurrency, brief_cached=cached_briefs,
                                              select_fraction=(lo, hi), lexical_rescue=not args.no_lexical_rescue, category_concurrency=args.category_concurrency,
                                              call_overhead_tokens=args.call_overhead_tokens, price_fn=price_fn)
    n_cached = sum(1 for i in in_scope if i.category in cached_briefs)
    print("=" * 72)
    print(f"Categories selected: {len(selected):,} | skipped as context-only chapters: {len(skipped_ctx):,} | to build: {projection['n_categories']:,} | already complete: {len(done_ids):,} (not projected)")
    print("Reviewable units (after exclusions): " + " | ".join(f"{f} {projection['per_family'][f]['n_units']:,}" for f in families)
          + "   (excluded: " + ", ".join(f"{f} {projection['per_family'][f]['n_excluded']:,}" for f in families) + ")")
    print(f"Models: brief={models['brief']} route={models['route']} judge={models['judge']} | thinking={settings.thinking} | inference profile prefix={settings.inference_profile_prefix}")
    if n_cached:
        print(f"Briefs already stored for {n_cached} of {len(in_scope)} categories: reused, not re-bought.")
    print("Work remaining: " + proj.describe_groups(projection["groups"]))
    print("-" * 72)
    e = projection["exhaustive"]
    print(f"[EXHAUSTIVE -- every category x every unit]  {e['pairs']:,} (category, unit) pairs")
    print(f"  ~{e['calls']:,.0f} LLM calls | {proj.fmt_usd(e['cost_low'])} - {proj.fmt_usd(e['cost_high'])} USD | ~{proj.fmt_hours(e['hours'])} at concurrency {args.concurrency}")
    print("-" * 72)
    r = projection["routed"]
    print("[ROUTED -- brief + tree routing + lexical rescue + judge]   (production default)")
    print(f"  ~{r['calls_low']:,.0f} - {r['calls_high']:,.0f} LLM calls | {proj.fmt_usd(r['cost_low'])} - {proj.fmt_usd(r['cost_high'])} USD")
    for c, cc in ((10, 4), (20, 8), (40, 16)):
        lo_h = proj.estimate_hours(r["calls_low"], r["depth"], projection["n_categories"], c, cc)
        hi_h = proj.estimate_hours(r["calls_high"], r["depth"], projection["n_categories"], c, cc)
        print(f"  --concurrency {c:>2} --category-concurrency {cc:>2}: ~{proj.fmt_hours(lo_h)} - {proj.fmt_hours(hi_h)} wall-clock "
              f"({SECONDS_PER_CALL_ESTIMATE} s/call assumed; {r['depth']} sequential calls per category; mind your Bedrock quotas)")
    print(f"  assumptions: model keeps {lo:.0%}-{hi:.0%} of each routing frontier; lexical rescue {'on' if not args.no_lexical_rescue else 'off'}; "
          f"{projection['assumptions']['mean_category_block_chars']:.0f}-char category block; {args.call_overhead_tokens:,}-token per-call overhead")
    print("  per family (routed, low -> high scenario): ")
    for f in families:
        pf = projection["per_family"][f]
        print(f"    {f:>10}: routing calls {pf['routed_lo']['route_calls']:.0f}-{pf['routed_hi']['route_calls']:.0f}/category, candidates {pf['routed_lo']['candidates']:.0f}-"
              f"{pf['routed_hi']['candidates']:.0f}/category, judge calls {pf['routed_lo']['judge_calls']}-{pf['routed_hi']['judge_calls']}/category "
              f"(exhaustive would be {pf['exhaustive_calls_per_category']} judge calls/category)")
    if e["cost_low"] > 0:
        ratio_lo, ratio_hi = e["cost_low"] / max(r["cost_high"], 1e-9), e["cost_high"] / max(r["cost_low"], 1e-9)
        if ratio_lo >= 1:
            print(f"  routed is roughly {ratio_lo:,.1f}x - {ratio_hi:,.1f}x cheaper than exhaustive -- at the price of routing recall, which `dx-support validate-retrieval` measures.")
        else:
            print(f"  routed vs exhaustive cost ratio: {ratio_lo:,.1f}x - {ratio_hi:,.1f}x (routing only pays off on a universe this large or larger than the real one).")
    if projection["unknown_price_models"]:
        print(f"  NOTE: no price on file for {projection['unknown_price_models']} -- priced as the default model.")

    cal = proj.calibrate_from_pilot(run.ckpt, families=families) if run else None
    if cal is not None and not proj.calibration_applies(projection["groups"], families):
        print("-" * 72)
        print("[CALIBRATION] not shown: part of the remaining work is only some of the requested families, which a per-category average cannot represent.")
    elif cal is not None or args.calibrate:
        print("-" * 72)
        if cal is None:
            print("[CALIBRATION] no usable pilot found (need >= 3 categories that finished ALL the requested families). Run a pilot first, e.g.\n"
                  "  dx-support build --categories H25,I48,M54,K35 --max-total-cost-usd 10")
        else:
            remaining = in_scope
            tot = cal["cost_per_category"] * len(remaining)
            print(f"[CALIBRATED from {cal['n_categories_observed']} pilot categories] {cal['calls_per_category']:.1f} calls and ${cal['cost_per_category']:.3f} per category")
            for stage, d in cal["by_stage"].items():
                print(f"    {stage:>6}: {d['calls_per_category']:.1f} calls/category x ${d['cost_per_call']:.4f}/call")
            print(f"  -> {len(remaining):,} remaining categories: ~{cal['calls_per_category'] * len(remaining):,.0f} calls, ~{proj.fmt_usd(tot)} (pilot categories may not be representative -- widen the pilot across chapters)")
            print(f"     measured cost per category: min {proj.fmt_usd(cal['cost_min'])}, median {proj.fmt_usd(cal['cost_p50'])}, max {proj.fmt_usd(cal['cost_max'])} "
                  f"-> remaining at the median ~{proj.fmt_usd(cal['cost_p50'] * len(remaining))}; if every one cost the max ~{proj.fmt_usd(cal['cost_max'] * len(remaining))}")
            if cal.get("seconds_per_call"):
                hrs = proj.estimate_hours(cal["calls_per_category"] * len(remaining), r["depth"], len(remaining), args.concurrency, args.category_concurrency, cal["seconds_per_call"])
                print(f"     measured {cal['seconds_per_call']:.0f} s per call -> ~{proj.fmt_hours(hrs)} at --concurrency {args.concurrency} --category-concurrency {args.category_concurrency}")
    print("=" * 72)
    if run:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
