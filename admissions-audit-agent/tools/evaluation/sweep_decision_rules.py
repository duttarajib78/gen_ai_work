"""Would a different per-verdict decision rule score better against the same ground truth?

Uses the per-verdict ``benchmark_results.csv`` a benchmark run already produced, so it
needs no new model calls. Each rule decides which VERDICTS count as failing; an
EPISODE is flagged when ANY of its verdicts fails (the aggregation the service uses and
this sweep holds fixed). Ground truth is per episode.

Reading the output:

* if every rule scores about as poorly as "none only", the support_level/confidence
  signal does not track the ground truth at any threshold. Fix it upstream (evidence,
  prompt, model, effort), not by moving the line.
* if one rule clearly beats the current one, treat it as a candidate only. Choosing a
  threshold by searching the data it is scored on overstates its benefit (multiple
  comparisons); confirm on a fresh slice before changing AUDIT_LOW_CONFIDENCE_WEAK_THRESHOLD.

Not swept: separate thresholds for PDx and ADx, and aggregations other than "any".

    python -m tools.evaluation.sweep_decision_rules output/benchmark/benchmark_results.csv [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import pandas as pd

from audit_agent.rules import diagnosis_fails

from .metrics import compute_metrics, confusion, fmt

REQUIRED = {"admission_no", "role", "support_level", "confidence", "ground_truth_unsupported", "error"}
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
STRICT_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def build_rules(df: pd.DataFrame, current_threshold: float) -> dict[str, pd.Series]:
    level, conf = df["support_level"], df["confidence"]
    rules = {'"none" only': level == "none",
             '"none" OR "weak", any confidence': level.isin(["none", "weak"])}
    for t in SWEEP:
        name = f'"none" OR ("weak" AND confidence<{t})'
        if abs(t - current_threshold) < 1e-9:
            name = f"CURRENT: {name}"
        rules[name] = pd.Series([diagnosis_fails(l, c, t) for l, c in zip(level, conf)], index=df.index)
    if all(abs(t - current_threshold) > 1e-9 for t in SWEEP):
        rules[f'CURRENT: "none" OR ("weak" AND confidence<{current_threshold})'] = pd.Series(
            [diagnosis_fails(l, c, current_threshold) for l, c in zip(level, conf)], index=df.index)
    for t in STRICT_SWEEP:
        rules[f'"none" AND confidence>={t}'] = (level == "none") & (conf >= t)
    for t in STRICT_SWEEP:
        rules[f'"none" OR ("weak" AND confidence>={t})'] = (level == "none") | ((level == "weak") & (conf >= t))
    return rules


def episode_level(df: pd.DataFrame, verdict_fails: pd.Series) -> tuple[pd.Series, pd.Series]:
    tmp = df.assign(_fails=verdict_fails)
    return tmp.groupby("admission_no")["_fails"].any(), tmp.groupby("admission_no")["ground_truth_unsupported"].first()


def evaluate(df: pd.DataFrame, current_threshold: float) -> list[tuple[str, dict]]:
    rows = []
    for name, mask in build_rules(df, current_threshold).items():
        predicted, truth = episode_level(df, mask)
        rows.append((name, compute_metrics(confusion(zip(truth.astype(bool), predicted.astype(bool))))))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--threshold", type=float, default=0.5, help="the threshold currently deployed (marks the CURRENT row)")
    a = p.parse_args(argv)

    df = pd.read_csv(a.path)
    missing = REQUIRED - set(df.columns)
    if missing:
        print(f"error: {a.path} is missing column(s) {sorted(missing)}; expected a per-verdict benchmark_results.csv",
              file=sys.stderr)
        return 2
    df["ground_truth_unsupported"] = df["ground_truth_unsupported"].astype(bool)
    scored = df[df["error"].isna() | (df["error"] == "")].dropna(subset=["support_level"]).copy()
    print(f"{len(scored)} verdict row(s) from {scored['admission_no'].nunique()} episode(s); "
          f"{len(df) - len(scored)} errored row(s) excluded.")

    varying = scored.groupby("admission_no")["ground_truth_unsupported"].nunique()
    if (varying > 1).any():
        print(f"WARNING: ground truth varies within {int((varying > 1).sum())} episode(s); it must be per episode. "
              "Taking the first value; do not trust these results until that is understood.")

    rows = evaluate(scored, a.threshold)
    width = max(len(n) for n, _ in rows) + 2
    header = f"{'rule':{width}s} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} {'precision':>10} {'recall':>8} {'f1':>8} {'mcc':>8}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:{width}s} {m['tp']:4d} {m['fp']:4d} {m['fn']:4d} {m['tn']:4d} "
              f"{fmt(m['precision']):>10} {fmt(m['recall']):>8} {fmt(m['f1']):>8} {fmt(m['mcc']):>8}")

    best = max(rows, key=lambda r: r[1]["f1"] if r[1]["f1"] is not None else -1)
    current = next((m for n, m in rows if n.startswith("CURRENT")), None)
    print(f"\nBest F1: {best[0]!r} ({fmt(best[1]['f1'])}); current rule F1: {fmt(current['f1']) if current else 'n/a'}.")
    if current and best[1]["f1"] and current["f1"] and best[1]["f1"] > current["f1"] * 1.5:
        print("The gap is large enough to investigate, but confirm on a fresh slice before changing the threshold.")
    else:
        print("No rule here meaningfully beats the current one: the per-verdict signal itself is the limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
