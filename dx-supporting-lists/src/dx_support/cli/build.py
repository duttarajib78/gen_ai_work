"""dx-support build -- builds the supporting lists (ACHI codes, MBS items, prosthesis billing codes, optionally diagnoses) for each 3-character
ICD-10-AM category, using Claude on Amazon Bedrock. See pipeline/stages.py for the method (brief -> tree routing + lexical rescue -> judge).

ALWAYS run `dx-support estimate` first, or at least read the projection this command prints before confirming. Nothing is spent without an interactive
"yes" or an explicit --yes, and the default cumulative cap is a modest $50 that must be raised deliberately.

Resumable by design: re-running the same command after an interruption (network blip, Ctrl-C, cost cap, a stopped job) continues from what the store
holds -- briefs, routing decisions and verdicts already stored are never re-sent, so never re-billed.

Exit codes (so a shell loop over chapters can stop at the first problem): 0 finished cleanly; 2 fatal error; 3 cost cap reached; 4 finished but some
categories are partial -- re-run the same command.

PRE-WARMING THE ENDPOINT: `--for-service` writes into the result store the endpoint reads (serve/<fingerprint>/), so categories built here are served
instantly. Run it with the SAME DX_* settings as the endpoint (same model, thinking, data) -- otherwise the fingerprint differs and nothing is shared.

    dx-support build --dry-run --max-categories 5                      # offline plumbing check, no credentials, no cost
    dx-support build --categories H25,I48 --max-total-cost-usd 5       # a named pilot
    dx-support build --chapter 9 --max-total-cost-usd 150 --yes        # one whole chapter (9 or IX)
    dx-support build --chapter 9 --families dx                         # the diagnosis family
    dx-support build --for-service --chapter 7-10 --max-total-cost-usd 400 --yes
    dx-support build --mode exhaustive --categories H25                # full-universe review of one category (certainty / validation)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dx_support.constants import DEFAULT_FAMILIES, DEFAULT_THINKING, THINKING_PRESETS, parse_families
from dx_support.indexing.universe import load_universe, select_categories
from dx_support.llm.gate import AdaptiveLimiter, LLMGate
from dx_support.pipeline import projection as proj
from dx_support.pipeline import stages as S
from dx_support.pipeline.constants import DEFAULT_CATEGORY_CONCURRENCY
from dx_support.pipeline.context import RunContext
from dx_support.pipeline import prompts as P
from dx_support.pipeline.fingerprint import PIPELINE_VERSION, resolved_models, run_fingerprint
from dx_support.reference.chapters import CONTEXT_ONLY_CHAPTERS, parse_chapters
from dx_support.storage.locks import RunLock
from dx_support.cli.common import RUN_CONFIG_KEY, add_run_args, fmt_ts, install_graceful_shutdown, load_settings, make_backend, namespace_path, open_run

logger = logging.getLogger("dx_support.build")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="dx-support build", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p)
    p.add_argument("--for-service", action="store_true", help="Write into the store the endpoint reads (serve/<fingerprint>/) to pre-warm it. Use the endpoint's own DX_* settings.")
    p.add_argument("--categories", type=str, default=None, help="Comma-separated 3-character categories (e.g. H25,I48). Overrides --chapter/--max-categories.")
    p.add_argument("--chapter", "--chapters", type=parse_chapters, default=None, metavar="CHAPTERS", help="Only these ICD-10-AM chapters: 1-22 or Roman numerals, one, a list, or a range (9, 9,10, IX,X, 9-11).")
    p.add_argument("--max-categories", type=int, default=None, help="Only the first N categories (code order) -- for a pilot. Composes with --chapter (chapter applies first).")
    p.add_argument("--families", type=str, default=",".join(DEFAULT_FAMILIES), help=f"Comma-separated families (default {','.join(DEFAULT_FAMILIES)}); add dx for the diagnosis family; `all` = every family.")
    p.add_argument("--mode", choices=["routed", "exhaustive"], default="routed", help="routed (default): tree routing + lexical rescue. exhaustive: review EVERY unit (~100 calls/category) into <namespace>/exhaustive.")
    p.add_argument("--model", type=str, default=None, help="Catalog alias or explicit Bedrock id (default DX_MODEL, else claude-sonnet-5). NOT re-validated for other models: pilot first.")
    p.add_argument("--brief-model", type=str, default=None); p.add_argument("--route-model", type=str, default=None); p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--thinking", choices=list(THINKING_PRESETS), default=None, help=f"Default DX_THINKING, else '{DEFAULT_THINKING}'.")
    p.add_argument("--concurrency", type=int, default=None, help="Max in-flight model calls (default DX_LLM_CONCURRENCY).")
    p.add_argument("--category-concurrency", type=int, default=DEFAULT_CATEGORY_CONCURRENCY, help="Max categories in flight.")
    p.add_argument("--max-budget-usd-per-call", type=float, default=None, help="A call whose worst-case cost exceeds this is refused before it is sent.")
    p.add_argument("--max-total-cost-usd", type=float, default=50.0, help="Hard cumulative spend cap for this run (default $50, deliberately modest).")
    p.add_argument("--include-context-chapters", action="store_true", help="Also build chapters XX and XXII (circumstances, not conditions).")
    p.add_argument("--include-coarse-prostheses", action="store_true", help="Also review prosthesis codes that carry only a coarse category. Low resolution.")
    p.add_argument("--brief-skip", action="store_true", help="Skip categories whose brief says no procedural evidence is typical. Off by default (recall first).")
    p.add_argument("--no-lexical-rescue", action="store_true", help="Disable the BM25 'lexical rescue' candidates (A/B comparison).")
    p.add_argument("--accept-config-change", action="store_true", help="Continue a run whose model / prompts / data changed since it started (mixes results; not recommended).")
    p.add_argument("--fake-fail-rate", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--fake-delay-s", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--yes", action="store_true", help="Skip the interactive confirmation (unattended runs). The projection is still printed.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        families = parse_families(args.families)
    except ValueError as e:
        sys.exit(str(e))
    settings = load_settings(args)
    over = {k: v for k, v in {"model": args.model, "brief_model": args.brief_model, "route_model": args.route_model, "judge_model": args.judge_model,
                              "thinking": args.thinking, "llm_concurrency": args.concurrency, "max_budget_usd_per_call": args.max_budget_usd_per_call,
                              "lexical_rescue": False if args.no_lexical_rescue else None,
                              "include_context_chapters": True if args.include_context_chapters else None}.items() if v is not None}
    settings = settings.with_overrides(**over)
    backend_name = "fake" if args.dry_run else "bedrock"
    price_fn = proj.make_price_fn(settings.inference_profile_prefix, settings.model_catalog_path)

    universe = load_universe(settings.data_dir, settings.include_coarse_prostheses)
    categories, indices = universe.categories, universe.indices
    fingerprint = run_fingerprint(settings, universe, mode=args.mode, backend=backend_name)
    if args.for_service:
        if args.mode != "routed":
            sys.exit("--for-service only applies to the routed mode (the endpoint serves routed results).")
        ns_path = f"serve/{run_fingerprint(settings, universe, mode='routed', backend=backend_name)}"
    else:
        ns_path = namespace_path(args.namespace, args.dry_run, "exhaustive" if args.mode == "exhaustive" else None)
    run = open_run(settings, ns_path)
    ckpt = run.ckpt
    exit_code = 0
    try:
        with RunLock(run.store):
            requested = args.categories.split(",") if args.categories else None
            selected, missing = select_categories(categories, requested, args.chapter, args.max_categories)
            if missing:
                logger.warning("These requested categories are not in diagnosis_codes.tsv, skipping: %s", missing)
            if not selected:
                sys.exit("No categories left to build (check --categories / --chapter / --max-categories).")
            skipped_ctx = [] if settings.include_context_chapters else [c for c in selected if categories[c].chapter in CONTEXT_ONLY_CHAPTERS]
            in_scope = [categories[c] for c in selected if c not in set(skipped_ctx)]

            # Configuration guard: a run continues only under the configuration it started with.
            cfg = ckpt.read_json(RUN_CONFIG_KEY) or {}
            if cfg.get("fingerprint") and cfg["fingerprint"] != fingerprint and not args.accept_config_change:
                sys.exit(f"This namespace ({run.store.uri}) was started with a different configuration (fingerprint {cfg['fingerprint']}, now {fingerprint}): the model, thinking "
                         "setting, prompts, tunables or reference data changed, and mixing results would make the lists meaningless. Use a new --namespace, or pass "
                         "--accept-config-change if you accept the mix.")
            cfg.update({"fingerprint": fingerprint, "pipeline_version": PIPELINE_VERSION, "mode": args.mode, "families": list(families),
                        "include_coarse_prostheses": settings.include_coarse_prostheses, "dry_run": args.dry_run, "thinking": settings.thinking,
                        "last_models": resolved_models(settings, backend_name), "data_digest": universe.digest, "updated": fmt_ts()})
            ckpt.write_json(RUN_CONFIG_KEY, cfg)

            status = ckpt.load_status_all([c for c in selected])
            done_ids = {c for c in selected if S.category_done_from_status(status, c, families, args.mode)}
            in_scope = [i for i in in_scope if i.category not in done_ids]
            if not in_scope:
                print("=" * 72)
                print(f"Nothing to build: all {len(selected) - len(skipped_ctx)} selected categories are already complete"
                      + (f" ({len(skipped_ctx)} context-only skipped)" if skipped_ctx else "") + ".")
                print(f"Next: dx-support compile{' --dry-run' if args.dry_run else ''}{' --tag exhaustive' if args.mode == 'exhaustive' else ''}"
                      + (f" --chapter {','.join(args.chapter)}" if args.chapter else ""))
                print("=" * 72)
                return 0
            models = settings.stage_models()
            cached = ckpt.cached_brief_ids()
            n_cached_briefs = sum(1 for i in in_scope if i.category in cached)
            projection = proj.project_scope_remaining(in_scope, status, families, indices, models, settings.thinking, settings.llm_concurrency, brief_cached=cached,
                                                      lexical_rescue=settings.lexical_rescue, category_concurrency=args.category_concurrency, price_fn=price_fn)
            print("=" * 72)
            print(f"Categories selected: {len(selected)} | context-only chapters skipped: {len(skipped_ctx)} | to build: {len(in_scope)} | already complete: {len(done_ids)}")
            print(f"Store: {run.store.uri}  (fingerprint {fingerprint})")
            print(f"Families: {', '.join(families)} | mode: {args.mode} | reviewable units: " + ", ".join(f"{f}={projection['per_family'][f]['n_units']:,}" for f in families))
            print("Work remaining: " + proj.describe_groups(projection["groups"]))
            if args.mode == "exhaustive":
                e = projection["exhaustive"]
                print(f"Projection (EXHAUSTIVE): ~{e['calls']:,.0f} LLM calls, {proj.fmt_usd(e['cost_low'])} - {proj.fmt_usd(e['cost_high'])}, ~{proj.fmt_hours(e['hours'])} wall-clock")
            else:
                r = projection["routed"]
                print(f"Projection (ROUTED): ~{r['calls_low']:,.0f} - {r['calls_high']:,.0f} LLM calls, {proj.fmt_usd(r['cost_low'])} - {proj.fmt_usd(r['cost_high'])}, "
                      f"~{proj.fmt_hours(r['hours_low'])} - {proj.fmt_hours(r['hours_high'])} wall-clock (order of magnitude -- see `dx-support estimate`; calibrate after a pilot)")
            if n_cached_briefs:
                print(f"  ({n_cached_briefs} of {len(in_scope)} briefs are already stored and are reused, not re-bought)")
            cal = proj.calibrate_from_pilot(ckpt, families=families)
            if cal and not proj.calibration_applies(projection["groups"], families):
                print("  (measured per-category cost not shown: the remaining work is only some of the requested families, which an average cannot represent)")
                cal = None
            if cal:
                remaining = [i for i in in_scope if status.get(i.category, {}).get("status") != "complete"]
                print(f"CALIBRATED (measured on {cal['n_categories_observed']} completed categories): {len(remaining)} remaining x {proj.fmt_usd(cal['cost_per_category'])} avg "
                      f"(min {proj.fmt_usd(cal['cost_min'])}, max {proj.fmt_usd(cal['cost_max'])}) => ~{proj.fmt_usd(cal['cost_per_category'] * len(remaining))} expected, "
                      f"~{proj.fmt_usd(cal['cost_max'] * len(remaining))} if every one costs the max")
            print(f"This run's hard cumulative cap: ${args.max_total_cost_usd:,.0f} (stops launching new calls once reached; completed work stays stored).")
            print(f"Models: brief={models['brief']} route={models['route']} judge={models['judge']} | thinking={settings.thinking} | concurrency={settings.llm_concurrency}"
                  + (" | DRY RUN (offline stub, no spend)" if args.dry_run else f" | region={settings.aws_region} profile={settings.inference_profile_prefix}"))
            hint = S.concurrency_hint(settings.llm_concurrency, args.category_concurrency, len(families))
            if hint:
                print("NOTE: " + hint)
            if projection["unknown_price_models"]:
                print(f"NOTE: no price on file for {projection['unknown_price_models']} -- projection uses the default model's price.")
            print("=" * 72)
            if not args.yes and not args.dry_run and input("Type 'yes' to proceed: ").strip().lower() != "yes":
                sys.exit("Aborted -- nothing was sent to the model.")

            backend = make_backend(settings, args.dry_run, args.fake_fail_rate, args.fake_delay_s)
            gate = LLMGate(backend, AdaptiveLimiter(settings.llm_concurrency), args.max_total_cost_usd, run_log=ckpt.log_run, backoff_base_s=0.0 if args.dry_run else 1.5,
                           max_retries=settings.max_call_retries, max_throttle_retries=settings.max_throttle_retries,
                           call_timeout_s=None if args.dry_run else settings.llm_call_timeout_s + 30)
            ctx = RunContext(categories=categories, indices=indices, gate=gate, ckpt=ckpt, families=families, mode=args.mode, models=models, thinking=settings.thinking,
                             include_context_chapters=settings.include_context_chapters, brief_skip=args.brief_skip, lexical_rescue=settings.lexical_rescue,
                             max_budget_usd_per_call=settings.max_budget_usd_per_call, max_tokens=dict(settings.max_tokens), fingerprint=fingerprint,
                             vocab=P.vocabulary_hint(indices))
            install_graceful_shutdown()
            try:
                stats = asyncio.run(S.run_build(ctx, selected, args.category_concurrency))
            except KeyboardInterrupt:
                ckpt.flush_run_log()
                sys.exit("\nInterrupted. Everything stored so far is kept -- re-run the same command to continue.")
            finally:
                close = getattr(backend, "close", None)
                if close:
                    close()

            print("=" * 72)
            print(f"Outcomes: {stats['outcomes']} | category errors: {stats['n_category_errors']}")
            print(f"Calls this run: {stats['n_calls_this_run']:,} (failed attempts: {stats['n_failed_attempts']:,}) | cost this run: ${stats['cost_usd_this_run']:,.2f}")
            for stage, d in stats["by_stage"].items():
                print(f"  {stage:>6}: {d['calls']:,} calls, ${d['cost_usd']:,.2f}")
            print(f"Verdicts this run by tier: {stats['tier_counts_this_run']}")
            if stats["fatal_error"]:
                print(f"STOPPED on a fatal error: {stats['fatal_error']}")
            elif stats["stopped_on_cost_cap"]:
                print(f"Stopped early: hit --max-total-cost-usd={args.max_total_cost_usd:,.0f}. Re-run the same command to continue.")
            n_partial = stats["outcomes"].get("partial", 0)
            if n_partial:
                print(f"{n_partial} categor{'y is' if n_partial == 1 else 'ies are'} PARTIAL (a call kept failing, or the run stopped): re-run the same command to finish them.")
            print(f"Next: dx-support compile{' --dry-run' if args.dry_run else ''}{' --tag exhaustive' if args.mode == 'exhaustive' else ''}")
            print("=" * 72)
            exit_code = 2 if stats["fatal_error"] else 3 if stats["stopped_on_cost_cap"] else 4 if n_partial else 0
    finally:
        run.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
