#!/usr/bin/env python3
"""Client for the deployed endpoint (SigV4 via your AWS credentials). Every call returns within seconds.

  invoke_endpoint.py --endpoint E health
  invoke_endpoint.py --endpoint E plan   [--topics 0110,0002] [--limit-topics 3] [--limit-sweep-windows 3] [--no-tabular-sweep]
  invoke_endpoint.py --endpoint E submit [same flags] [--idempotency-key KEY] [--max-cost-usd 50]
  invoke_endpoint.py --endpoint E status JOB_ID [--units]
  invoke_endpoint.py --endpoint E wait   JOB_ID [--timeout 14400] [--interval 30]
  invoke_endpoint.py --endpoint E rules  JOB_ID [--offset 0] [--limit 50]
  invoke_endpoint.py --endpoint E results JOB_ID | cancel JOB_ID | retry JOB_ID
  invoke_endpoint.py --endpoint E raw < request.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def client(region: str | None):
    import boto3
    from botocore.config import Config
    # 70 s socket timeout: SageMaker answers within 60 s or returns an error.
    return boto3.client("sagemaker-runtime", region_name=region,
                        config=Config(read_timeout=70, connect_timeout=5, retries={"max_attempts": 3, "mode": "standard"}))


def invoke(rt, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    from botocore.exceptions import ClientError
    try:
        resp = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Accept="application/json",
                                  Body=json.dumps(payload).encode("utf-8"))
    except ClientError as exc:
        msg = exc.response.get("Message") or exc.response.get("Error", {}).get("Message", str(exc))
        raise SystemExit(f"endpoint error: {msg}") from exc
    body = json.loads(resp["Body"].read())
    if not body.get("ok", False):
        raise SystemExit(json.dumps(body, indent=2))
    return body["data"]


def options_from(args: argparse.Namespace) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if args.topics:
        opts["topics"] = args.topics.split(",")
    if args.limit_topics:
        opts["limit_topics"] = args.limit_topics
    if args.limit_sweep_windows:
        opts["limit_sweep_windows"] = args.limit_sweep_windows
    if args.no_tabular_sweep:
        opts["include_tabular_sweep"] = False
    if args.no_acs:
        opts["include_acs_topics"] = False
    if getattr(args, "max_cost_usd", None):
        opts["max_cost_usd"] = args.max_cost_usd
    return opts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--region")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "submit"):
        s = sub.add_parser(name)
        s.add_argument("--topics")
        s.add_argument("--limit-topics", type=int)
        s.add_argument("--limit-sweep-windows", type=int)
        s.add_argument("--no-tabular-sweep", action="store_true")
        s.add_argument("--no-acs", action="store_true")
        if name == "submit":
            s.add_argument("--idempotency-key")
            s.add_argument("--max-cost-usd", type=float)
    sub.add_parser("health")
    sub.add_parser("raw")
    st = sub.add_parser("status")
    st.add_argument("job_id")
    st.add_argument("--units", action="store_true")
    w = sub.add_parser("wait")
    w.add_argument("job_id")
    w.add_argument("--timeout", type=int, default=4 * 3600)
    w.add_argument("--interval", type=int, default=30)
    r = sub.add_parser("rules")
    r.add_argument("job_id")
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--limit", type=int, default=50)
    for name in ("results", "cancel", "retry"):
        sub.add_parser(name).add_argument("job_id")
    args = p.parse_args()
    rt = client(args.region)

    if args.cmd == "raw":
        out = invoke(rt, args.endpoint, json.load(sys.stdin))
    elif args.cmd == "health":
        out = invoke(rt, args.endpoint, {"operation": "health"})
    elif args.cmd == "plan":
        out = invoke(rt, args.endpoint, {"operation": "plan", "options": options_from(args)})
    elif args.cmd == "submit":
        payload: dict[str, Any] = {"operation": "submit_job", "options": options_from(args)}
        if args.idempotency_key:
            payload["idempotency_key"] = args.idempotency_key
        out = invoke(rt, args.endpoint, payload)
    elif args.cmd == "status":
        out = invoke(rt, args.endpoint, {"operation": "get_job", "job_id": args.job_id, "include_units": args.units})
    elif args.cmd == "wait":
        deadline = time.monotonic() + args.timeout
        while True:
            out = invoke(rt, args.endpoint, {"operation": "get_job", "job_id": args.job_id})
            prog = out.get("progress", {})
            print(f"{out['status']}: {prog.get('done', 0)}/{prog.get('total', '?')} units, ${out.get('cost_usd', 0):.2f}", file=sys.stderr)
            if out["status"] in ("SUCCEEDED", "PARTIAL", "FAILED", "CANCELLED"):
                break
            if time.monotonic() > deadline:
                raise SystemExit("timed out waiting for the job (it keeps running; poll again)")
            time.sleep(args.interval)
    elif args.cmd == "rules":
        out = invoke(rt, args.endpoint, {"operation": "get_rules", "job_id": args.job_id, "offset": args.offset, "limit": args.limit})
    else:
        op = {"results": "get_results", "cancel": "cancel_job", "retry": "retry_failed"}[args.cmd]
        out = invoke(rt, args.endpoint, {"operation": op, "job_id": args.job_id})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
