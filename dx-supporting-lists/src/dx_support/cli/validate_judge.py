"""dx-support validate-judge -- Pilot: how good is the JUDGE stage (the model's core / broad / unclear / not_supporting verdicts), measured before the lists
are used for anything?

Two modes
  1. GOLD mode (--gold FILE.csv) -- the real accuracy test. FILE.csv holds clinician-reviewed answers, one row per (category, code):
         icd10_category,family,code,gold_tier
         H25,achi,4270100,core
     gold_tier is core | broad | unclear | not_supporting; family is achi | mbs | prosthesis | dx. (For prostheses give any billing code in the device type;
     if codes of one device type disagree, the majority wins and the conflict is reported -- verdicts are per device type.) The judge re-judges exactly
     those pairs and the tool reports, overall and per family: SUPPORTING-vs-NOT precision / recall / F1 (supporting = core or broad), exact 4-tier
     agreement and Cohen's kappa, a confusion matrix, and a CSV of every disagreement. Build the gold set from a clinical coder's review of
     full_audit.csv. Include negatives: a gold file of only positives cannot measure false positives.
  2. REPEATABILITY mode (no --gold) -- not accuracy: it re-judges a stratified sample of pairs ALREADY in a run and reports how often the second verdict
     agrees with the first. Low agreement means the verdicts are unstable and any single run should be treated cautiously (try --thinking high, or a
     stronger --model, and compare).

The re-judge uses the brief already stored by the source run when there is one, so only the judge's own noise is measured. Nothing here writes to a
production namespace: verdicts are judged and compared in memory (the validation namespace holds only a run log and briefs it had to create).

    dx-support validate-judge --dry-run
    dx-support validate-judge --gold gold.csv
    dx-support validate-judge --n-pairs 300                          # repeatability on a sample of the default run's verdicts
    dx-support validate-judge --gold gold.csv --thinking high --model claude-sonnet-5-5   # compare configurations
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dx_support.cli.common import add_run_args, load_settings, make_backend, namespace_path, open_run
from dx_support.constants import AT_LEAST_BROAD, DEFAULT_THINKING, FAMILIES, SUPPORT_TIERS, THINKING_PRESETS
from dx_support.indexing.universe import load_universe
from dx_support.llm.gate import AdaptiveLimiter, LLMGate
from dx_support.pipeline import stages as S
from dx_support.pipeline.context import RunContext
from dx_support.storage.checkpoint import Checkpoint, latest_by_key
from dx_support.storage.locks import RunLock
from dx_support.text import safe_cell, display_text

logger = logging.getLogger("dx_support.validate_judge")
_CAT_RE = re.compile(r"^[A-Z]\d{2}$")


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[lb] / n) * (cb[lb] / n) for lb in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def prf(pairs: list[tuple[str, str]], positive=AT_LEAST_BROAD) -> dict:
    """pairs = (gold_tier, predicted_tier). Supporting = tier in `positive`."""
    tp = sum(1 for g, p in pairs if g in positive and p in positive)
    fp = sum(1 for g, p in pairs if g not in positive and p in positive)
    fn = sum(1 for g, p in pairs if g in positive and p not in positive)
    tn = len(pairs) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
    return {"n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / len(pairs) if pairs else float("nan")}


def read_gold(path, categories, indices) -> tuple[dict[tuple[str, str, str], str], dict]:
    """(category, family, unit_id) -> gold tier, plus a report of what was skipped / conflicted."""
    code_to_unit = {f: {c: uid for uid, u in idx.units.items() for c in u.codes} for f, idx in indices.items()}
    votes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    skipped = Counter()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        need = {"icd10_category", "family", "code", "gold_tier"}
        if not need <= set(rd.fieldnames or []):
            sys.exit(f"{path} must have columns {sorted(need)}; found {rd.fieldnames}.")
        for row in rd:
            cat, fam, code, tier = ((row["icd10_category"] or "").strip().upper(), (row["family"] or "").strip().lower(), (row["code"] or "").strip(), (row["gold_tier"] or "").strip().lower())
            if not _CAT_RE.match(cat) or cat not in categories:
                skipped["unknown_category"] += 1
                continue
            if fam not in indices:
                skipped["unknown_family"] += 1
                continue
            if tier not in SUPPORT_TIERS:
                skipped["invalid_gold_tier"] += 1
                continue
            uid = code_to_unit[fam].get(code)
            if uid is None:
                skipped["code_not_reviewable (excluded or unknown)"] += 1
                continue
            votes[(cat, fam, uid)].append(tier)
    gold, conflicts = {}, 0
    for key, v in votes.items():
        c = Counter(v).most_common()
        if len(c) > 1:
            conflicts += 1
        gold[key] = c[0][0]
    return gold, {"skipped": dict(skipped), "unit_level_conflicts": conflicts, "n_rows_used": sum(len(v) for v in votes.values())}


def sample_from_checkpoint(ckpt: Checkpoint, n: int, seed: int) -> dict[tuple[str, str, str], str]:
    """Stratified sample (half supporting, half not) of the newest stored verdicts."""
    rng = random.Random(seed)
    pos, neg = [], []
    for (cat, fam, uid), r in latest_by_key(ckpt.iter_verdicts()).items():
        (pos if r.get("support_tier") in AT_LEAST_BROAD else neg).append(((cat, fam, uid), r["support_tier"]))
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen = pos[: n // 2] + neg[: n - min(n // 2, len(pos))]
    return dict(chosen)


async def judge_pairs(ctx: RunContext, keys: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict]:
    by_cf: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cat, fam, uid in keys:
        by_cf[(cat, fam)].append(uid)
    out: dict[tuple[str, str, str], dict] = {}

    async def one(cat: str, fam: str, uids: list[str]):
        info = ctx.categories[cat]
        await ctx.ckpt.ahydrate(cat)
        brief = await S.get_brief(ctx, cat)
        got = await S.judge_batch(ctx, cat, info, brief, ctx.indices[fam], uids)
        for uid, v in got.items():
            out[(cat, fam, uid)] = v

    res = await asyncio.gather(*[one(c, f, u) for (c, f), u in by_cf.items()], return_exceptions=True)
    for ((c, f), _), r in zip(by_cf.items(), res):
        if isinstance(r, BaseException):
            logger.error("Judging %s/%s failed: %s", c, f, type(r).__name__)
    await ctx.ckpt.aflush_run_log()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dx-support validate-judge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, tag=True)
    p.add_argument("--gold", type=str, default=None, help="Clinician-reviewed CSV (icd10_category,family,code,gold_tier). Omit for repeatability mode.")
    p.add_argument("--n-pairs", type=int, default=300, help="Repeatability mode: how many stored verdicts to re-judge.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--thinking", choices=list(THINKING_PRESETS), default=None, help=f"Default DX_THINKING, else '{DEFAULT_THINKING}'.")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-budget-usd-per-call", type=float, default=None)
    p.add_argument("--max-total-cost-usd", type=float, default=20.0)
    p.add_argument("--output-dir", type=str, default="output/validation")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)
    settings = load_settings(args)
    over = {k: v for k, v in {"model": args.model, "thinking": args.thinking, "llm_concurrency": args.concurrency, "max_budget_usd_per_call": args.max_budget_usd_per_call}.items() if v is not None}
    settings = settings.with_overrides(**over)
    universe = load_universe(settings.data_dir, False)
    categories, indices = universe.categories, universe.indices

    source = open_run(settings, args.namespace if args.namespace.startswith(("runs/", "serve/")) else namespace_path(args.namespace, args.dry_run, args.tag))
    try:
        if args.gold:
            gold, report = read_gold(args.gold, categories, indices)
            reference, mode = gold, "GOLD"
            print(f"Gold rows used: {report['n_rows_used']:,} -> {len(gold):,} (category, unit) pairs | skipped: {report['skipped']} | unit-level conflicts resolved by majority: {report['unit_level_conflicts']}")
            if sum(1 for t in gold.values() if t in AT_LEAST_BROAD) == 0 or sum(1 for t in gold.values() if t not in AT_LEAST_BROAD) == 0:
                print("WARNING: the gold file needs BOTH supporting and not-supporting examples for precision/recall to mean anything.")
        else:
            reference = sample_from_checkpoint(source.ckpt, args.n_pairs, args.seed)
            mode = "REPEATABILITY"
            print(f"Sampled {len(reference):,} stored verdicts from {source.store.uri}")
        if not reference:
            sys.exit("No pairs to judge.")
        stored_briefs = {r["category"]: r["brief"] for r in source.ckpt.iter_briefs() if isinstance(r.get("brief"), dict)}
    finally:
        source.close()

    cats = sorted({k[0] for k in reference})
    models = {"brief": settings.model, "route": settings.model, "judge": settings.model}
    n_calls = len({(c, f) for c, f, _ in reference}) + len(cats)
    print("=" * 72)
    print(f"{mode} mode | {len(reference):,} pairs over {len(cats)} categories | ~{n_calls} calls | model {settings.model} | thinking {settings.thinking}" + (" | STUB (no spend)" if args.dry_run else ""))
    print("=" * 72)
    if not args.yes and not args.dry_run and input("Type 'yes' to proceed: ").strip().lower() != "yes":
        sys.exit("Aborted -- nothing was sent to the model.")

    run = open_run(settings, "validation/dry_run" if args.dry_run else "validation/judge")
    for c in cats:                                       # re-judge with the brief the source run used: only the judge's own noise is measured
        if c in stored_briefs:
            run.ckpt.briefs[c] = stored_briefs[c]
    backend = make_backend(settings, args.dry_run)
    gate = LLMGate(backend, AdaptiveLimiter(settings.llm_concurrency), args.max_total_cost_usd, run_log=run.ckpt.log_run, backoff_base_s=0.0 if args.dry_run else 1.5,
                   max_retries=settings.max_call_retries, max_throttle_retries=settings.max_throttle_retries, call_timeout_s=None if args.dry_run else settings.llm_call_timeout_s + 30)
    ctx = RunContext(categories=categories, indices=indices, gate=gate, ckpt=run.ckpt, models=models, thinking=settings.thinking,
                     max_budget_usd_per_call=settings.max_budget_usd_per_call, max_tokens=dict(settings.max_tokens))
    try:
        with RunLock(run.store):
            judged = asyncio.run(judge_pairs(ctx, sorted(reference)))
    finally:
        run.close()
        close = getattr(backend, "close", None)
        if close:
            close()

    pairs_all, rows, per_family = [], [], defaultdict(list)
    for key, ref_tier in sorted(reference.items()):
        v = judged.get(key)
        if v is None:
            continue
        pairs_all.append((ref_tier, v["support_tier"]))
        per_family[key[1]].append((ref_tier, v["support_tier"]))
        if ref_tier != v["support_tier"]:
            u = indices[key[1]].units[key[2]]
            rows.append([key[0], key[1], key[2], display_text(u.text)[:200], ref_tier, v["support_tier"], v["evidence_type"], v["rationale"]])
    ref_name = "gold_tier" if mode == "GOLD" else "stored_tier"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("judge_disagreements_dryrun.csv" if args.dry_run else "judge_disagreements.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["icd10_category", "family", "unit_id", "unit_text", ref_name, "judge_tier", "judge_evidence_type", "judge_rationale"])
        for r in rows:
            w.writerow([safe_cell(x) for x in r])

    print("=" * 72)
    if args.dry_run:
        print("DRY RUN -- offline stub, NOT real model behaviour. Sanity-checks the harness only.")
    print(f"Spend: ${gate.total_cost:,.2f} over {gate.n_calls} calls | judged {len(pairs_all):,} of {len(reference):,} pairs ({len(reference) - len(pairs_all)} got no verdict)")
    label = "gold" if mode == "GOLD" else "first-run"
    for name, pairs in [("ALL", pairs_all)] + [(f, per_family[f]) for f in FAMILIES if per_family[f]]:
        m = prf(pairs)
        exact = sum(1 for g, q in pairs if g == q) / len(pairs) if pairs else float("nan")
        print(f"{name:>10}: n={m['n']:>5} | supporting-vs-not: precision {m['precision']:.1%} recall {m['recall']:.1%} F1 {m['f1']:.1%} | exact tier agreement {exact:.1%} | kappa {cohen_kappa(pairs):.2f}")
    print("-" * 72)
    print(f"Confusion ({label} tier down, judge tier across):")
    cm = Counter(pairs_all)
    print(f"{'':>16}" + "".join(f"{t:>16}" for t in SUPPORT_TIERS))
    for g in SUPPORT_TIERS:
        print(f"{g:>16}" + "".join(f"{cm.get((g, q), 0):>16}" for q in SUPPORT_TIERS))
    if mode == "REPEATABILITY":
        print("NOTE: repeatability, not accuracy. Agreement with a previous model run says nothing about clinical correctness; use --gold for that.")
    print(f"Every disagreement: {out_csv}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
