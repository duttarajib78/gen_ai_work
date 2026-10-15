"""Run many episodes through the endpoint with retries and resumable checkpoints.

    python -m tools.client.batch_invoke --input episodes.parquet --endpoint-name admissions-audit-agent-prod \
        --region ap-southeast-2 --out-dir output/batch --mode full --workers 4 --stream

* Each episode is one request. Retryable failures (throttling, 5xx, deadline) are retried
  with backoff; the audit is read-only, so retries are safe.
* ``results.jsonl`` holds every completed result and is the checkpoint: re-running the same
  command skips episodes already completed. Failures go to ``failures.jsonl`` and are retried
  on the next run.
* ``<concept>_json.txt`` is rebuilt from ``results.jsonl`` on every run (so it never holds
  duplicates) with one alert per line in the schema downstream consumers already read:
  claim_seqno, alert_info, recovery_amount, alert_instance_id.
* Use --stream for effort high or long episodes (regular calls are cut at 60 seconds).
* Output files hold clinical reasoning: they are created readable by the owner only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

from tools.tabular import load_table, row_to_payload

from .sagemaker_client import invoke_with_retry, make_runtime_client


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def private_append(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "a", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue  # a torn final line from an interrupted run
    return out


def write_alert_file(results: list[dict], path: Path) -> int:
    seen: dict[Any, dict] = {}
    for rec in results:  # last record per admission wins
        body = rec.get("body") or {}
        if body.get("flagged") and body.get("alert"):
            seen[rec["admission_no"]] = body["alert"]
        else:
            seen.pop(rec["admission_no"], None)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for alert in seen.values():
            fh.write(json.dumps(alert) + "\n")
    return len(seen)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("output/batch"))
    p.add_argument("--mode", choices=("full", "verdicts_only"), default="full")
    p.add_argument("--workers", type=int, default=4, help="concurrent requests (keep within your Bedrock quota)")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--concept-name", default="llm_pdx_adx_clinical_support_audit")
    a = p.parse_args(argv)

    out = private_dir(a.out_dir)
    results_path, failures_path = out / "results.jsonl", out / "failures.jsonl"
    done = {r["admission_no"] for r in read_jsonl(results_path)}

    df = load_table(a.input)
    if a.limit:
        df = df.head(a.limit)
    todo = []
    for row in df.to_dict("records"):
        payload = row_to_payload(row, mode=a.mode)
        if payload["episode"].get("admission_no") not in done:
            todo.append(payload)
    print(f"{len(df)} episode(s) loaded; {len(done)} already completed; {len(todo)} to run.")

    runtime = make_runtime_client(a.region, streaming=a.stream)
    lock = threading.Lock()
    counts: Counter = Counter()
    results_fh, failures_fh = private_append(results_path), private_append(failures_path)

    def work(payload: dict) -> None:
        result, attempts = invoke_with_retry(runtime, a.endpoint_name, payload, stream=a.stream)
        adm = payload["episode"].get("admission_no")
        with lock:
            if result.ok:
                results_fh.write(json.dumps({"admission_no": adm, "http_status": result.http_status, "body": result.body}) + "\n")
                results_fh.flush()
                counts[result.body.get("disposition", "unknown")] += 1
            else:
                err = result.body.get("error") or {}
                failures_fh.write(json.dumps({"admission_no": adm, "http_status": result.http_status,
                                              "error": {"code": err.get("code"), "retryable": err.get("retryable")},
                                              "attempts": attempts}) + "\n")
                failures_fh.flush()
                counts[f"failed:{err.get('code', 'unknown')}"] += 1

    try:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            list(pool.map(work, todo))
    finally:
        results_fh.close()
        failures_fh.close()

    n_alerts = write_alert_file(read_jsonl(results_path), out / f"{a.concept_name}_json.txt")
    print("Run summary:", dict(counts) or "nothing to do")
    print(f"{n_alerts} alert(s) in {out / (a.concept_name + '_json.txt')}")
    failed = sum(v for k, v in counts.items() if k.startswith("failed:"))
    if failed:
        print(f"{failed} episode(s) failed; see {failures_path}. Re-run the same command to retry only those.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
