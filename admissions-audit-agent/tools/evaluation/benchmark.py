"""Benchmark the audit against human coding-expert review outcomes.

Ground truth is a boolean column (default ``Recommended``): True means a human
coder found the episode unsupported. The prediction is the service's
``potential_alert`` (phase 1: PDx or any ADx failing the decision rule). The same
``AuditService`` that serves production traffic is exercised, either in-process
(direct Bedrock calls, needs the AUDIT_* environment) or through a deployed endpoint.

Why this matters for a model change: confidence calibration differs between
models, and the 0.5 low-confidence threshold was set for a different one. Run this
on a labelled slice, then ``sweep_decision_rules`` on its per-verdict output,
before relying on any threshold. Judge it on a *fresh* slice: a threshold picked on
the data it is scored on looks better than it is.

    # in-process (reads AUDIT_AWS_REGION, AUDIT_BEDROCK_MODEL_ID, ... from the environment)
    python -m tools.evaluation.benchmark --input labelled.parquet --reference-dir ./reference
    # through a deployed endpoint
    python -m tools.evaluation.benchmark --input labelled.parquet --endpoint-name NAME --region ap-southeast-2

Outputs in --out-dir: benchmark_results.csv (one row per verdict), benchmark_episodes.csv,
benchmark_metrics.json, benchmark_report.md. Outputs contain model reasoning derived from
clinical codes: store them with the same care as the input.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

from tools.tabular import load_table, row_to_payload

from .metrics import SMALL_SAMPLE_THRESHOLD, compute_metrics, confusion, fmt, percentile

Invoker = Callable[[dict], tuple[int, dict]]
VERDICT_COLUMNS = ["admission_no", "role", "code", "support_level", "confidence", "failed", "flag_reason",
                   "ground_truth_unsupported", "error"]


def run_benchmark(rows: list[dict], invoke: Invoker, *, label_column: str = "Recommended", workers: int = 4) -> list[dict]:
    def one(row: dict) -> dict:
        payload = row_to_payload(row, mode="verdicts_only")
        truth = bool(row[label_column])
        started = time.monotonic()
        try:
            status, body = invoke(payload)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the run
            status, body = 500, {"status": "failed", "error": {"code": type(exc).__name__, "message": "invocation failed"}}
        return {
            "admission_no": payload["episode"].get("admission_no"), "ground_truth": truth, "http_status": status,
            "body": body, "wall_ms": int((time.monotonic() - started) * 1000),
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, rows))


def summarise(results: list[dict]) -> dict:
    scored = [r for r in results if r["body"].get("status") == "completed"]
    metrics = compute_metrics(confusion((r["ground_truth"], bool(r["body"]["potential_alert"])) for r in scored))
    metrics["n_total"] = len(results)
    metrics["n_errored"] = len(results) - len(scored)
    metrics["error_codes"] = dict(Counter((r["body"].get("error") or {}).get("code", "unknown")
                                          for r in results if r["body"].get("status") != "completed"))
    usage = [r["body"].get("usage") or {} for r in scored]
    metrics["input_tokens"] = sum(u.get("input_tokens", 0) for u in usage)
    metrics["output_tokens"] = sum(u.get("output_tokens", 0) for u in usage)
    elapsed = [float(u["elapsed_ms"]) for u in usage if "elapsed_ms" in u]
    metrics["elapsed_ms_p50"] = percentile(elapsed, 50)
    metrics["elapsed_ms_p95"] = percentile(elapsed, 95)
    metrics["provenance"] = next((r["body"]["provenance"] for r in scored if r["body"].get("provenance")), None)
    return metrics


def write_outputs(results: list[dict], out_dir: Path, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=VERDICT_COLUMNS)
        writer.writeheader()
        for r in results:
            body = r["body"]
            if body.get("status") != "completed":
                writer.writerow({"admission_no": r["admission_no"], "ground_truth_unsupported": r["ground_truth"],
                                 "error": (body.get("error") or {}).get("code", "unknown")})
                continue
            for v in body["verdicts"]:
                writer.writerow({"admission_no": r["admission_no"], "role": v["role"], "code": v["code"],
                                 "support_level": v["support_level"], "confidence": v["confidence"],
                                 "failed": v["failed"], "flag_reason": v["flag_reason"] or "",
                                 "ground_truth_unsupported": r["ground_truth"], "error": ""})
    with open(out_dir / "benchmark_episodes.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["admission_no", "ground_truth_unsupported", "predicted_unsupported", "correct", "status", "error"])
        for r in results:
            body = r["body"]
            ok = body.get("status") == "completed"
            pred = bool(body["potential_alert"]) if ok else ""
            writer.writerow([r["admission_no"], r["ground_truth"], pred, (pred == r["ground_truth"]) if ok else "",
                             body.get("status"), "" if ok else (body.get("error") or {}).get("code", "")])
    (out_dir / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out_dir / "benchmark_report.md").write_text(render_report(results, metrics), encoding="utf-8")


def render_report(results: list[dict], m: dict) -> str:
    prov = m.get("provenance") or {}
    scored = [r for r in results if r["body"].get("status") == "completed"]
    lines = ["# Audit benchmark report", ""]
    lines += ["## Run", "",
              f"- Model: {prov.get('model_id', 'n/a')} (routing: {prov.get('routing_scope', 'n/a')})",
              f"- Reasoning effort: {prov.get('reasoning_effort', 'n/a')}; thinking mode: {prov.get('thinking_mode', 'n/a')}",
              f"- Prompt version: {prov.get('prompt_version', 'n/a')}; rule version: {prov.get('rule_version', 'n/a')}",
              f"- Low-confidence weak threshold: {prov.get('low_confidence_weak_threshold', 'n/a')}",
              f"- Episodes: {m['n_total']} submitted, {m['n_scored']} scored, {m['n_errored']} errored (excluded from every metric)",
              ""]
    if m["n_scored"] < SMALL_SAMPLE_THRESHOLD:
        lines += [f"> Caution: only {m['n_scored']} scored episodes (fewer than {SMALL_SAMPLE_THRESHOLD}). "
                  "These figures cannot support a decision.", ""]
    lines += ["## Headline metrics (positive class = unsupported)", "", "| Metric | Value |", "|---|---|"]
    for key in ("precision", "recall", "f1", "accuracy", "specificity", "mcc", "prevalence"):
        lines.append(f"| {key} | {fmt(m[key])} |")
    lines += ["", "## Confusion matrix", "", "| | Audit flagged | Audit did not flag |", "|---|---|---|",
              f"| Human found unsupported | {m['tp']} (TP) | {m['fn']} (FN) |",
              f"| Human found supported | {m['fp']} (FP) | {m['tn']} (TN) |", ""]

    by_level: Counter = Counter()
    by_reason: Counter = Counter()
    for r in scored:
        for v in r["body"]["verdicts"]:
            by_level[(v["role"], v["support_level"])] += 1
            if v["flag_reason"]:
                by_reason[(v["role"], v["flag_reason"])] += 1
    lines += ["## Verdicts by role and support level", "", "| Role | Support level | Count |", "|---|---|---|"]
    lines += [f"| {role} | {level} | {n} |" for (role, level), n in sorted(by_level.items())]
    lines += ["", "## Failing verdicts by reason", "", "| Role | Reason | Count |", "|---|---|---|"]
    lines += [f"| {role} | {reason} | {n} |" for (role, reason), n in sorted(by_reason.items())] or ["| - | - | 0 |"]
    lines += ["", "## Cost and latency", "",
              f"- Input tokens: {m['input_tokens']}; output tokens: {m['output_tokens']}",
              f"- Elapsed per episode: p50 {fmt(m['elapsed_ms_p50'])} ms, p95 {fmt(m['elapsed_ms_p95'])} ms", ""]
    if m["error_codes"]:
        lines += ["## Errors", "", "| Code | Count |", "|---|---|"] + [f"| {c} | {n} |" for c, n in sorted(m["error_codes"].items())] + [""]
    lines += ["## Reading these results", "",
              "- The audit sees only diagnosis, ACHI, and MBS descriptions; human reviewers may have used more evidence.",
              "- Agreement with human review measures consistency, not clinical truth.",
              "- Choose thresholds on one slice and confirm on a fresh one (see sweep_decision_rules).", ""]
    return "\n".join(lines)


def local_invoker(reference_dir: Path) -> Invoker:
    """In-process: the real service with direct Bedrock calls and no grouper."""
    from audit_agent.config import Settings
    from audit_agent.deadline import Deadline
    from audit_agent.engine import AuditEngine
    from audit_agent.errors import RequestValidationError
    from audit_agent.llm.bedrock import BedrockClaudeClient
    from audit_agent.reference_data import ReferenceData
    from audit_agent.service import AuditService

    settings = Settings.from_env()
    reference = ReferenceData.from_directory(reference_dir)
    engine = AuditEngine(reference=reference, llm=BedrockClaudeClient.from_settings(settings), grouper=None, settings=settings)
    service = AuditService(settings=settings, engine=engine, reference=reference)

    def invoke(payload: dict) -> tuple[int, dict]:
        try:
            request = service.validate(payload)
        except RequestValidationError as exc:
            return 400, {"status": "failed", "error": exc.to_error_body()}
        result = service.audit(request, Deadline(settings.stream_deadline_s))
        return result.status_code, result.body

    return invoke


def endpoint_invoker(endpoint_name: str, region: str, stream: bool) -> Invoker:
    from tools.client.sagemaker_client import invoke_with_retry, make_runtime_client

    runtime = make_runtime_client(region, streaming=stream)

    def invoke(payload: dict) -> tuple[int, dict]:
        result, _ = invoke_with_retry(runtime, endpoint_name, payload, stream=stream)
        return result.http_status, result.body

    return invoke


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--label-column", default="Recommended")
    p.add_argument("--n-samples", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=Path("output/benchmark"))
    p.add_argument("--reference-dir", type=Path, help="in-process mode: directory with the three reference files")
    p.add_argument("--endpoint-name", help="endpoint mode")
    p.add_argument("--region")
    p.add_argument("--stream", action="store_true", help="endpoint mode: use response streaming (recommended for high effort)")
    a = p.parse_args(argv)

    df = load_table(a.input)
    if a.label_column not in df.columns:
        print(f"error: label column {a.label_column!r} not found; columns: {list(df.columns)[:20]}", file=sys.stderr)
        return 2
    if a.n_samples:
        df = df.head(a.n_samples)
    if a.endpoint_name:
        if not a.region:
            print("error: --region is required with --endpoint-name", file=sys.stderr)
            return 2
        invoke = endpoint_invoker(a.endpoint_name, a.region, a.stream)
    elif a.reference_dir:
        invoke = local_invoker(a.reference_dir)
    else:
        print("error: give --reference-dir (in-process) or --endpoint-name (deployed endpoint)", file=sys.stderr)
        return 2

    rows = df.to_dict("records")
    print(f"Benchmarking {len(rows)} episode(s) with {a.workers} worker(s)...")
    results = run_benchmark(rows, invoke, label_column=a.label_column, workers=a.workers)
    metrics = summarise(results)
    write_outputs(results, a.out_dir, metrics)
    print(f"scored {metrics['n_scored']}/{metrics['n_total']}; precision={fmt(metrics['precision'])} "
          f"recall={fmt(metrics['recall'])} f1={fmt(metrics['f1'])} mcc={fmt(metrics['mcc'])}")
    print(f"wrote {a.out_dir}/benchmark_report.md and companions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
