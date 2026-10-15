"""Invoke the endpoint once from the command line.

    python -m tools.client.invoke_endpoint --endpoint-name admissions-audit-agent-prod \
        --region ap-southeast-2 --payload episode.json [--stream]

The payload file is the full request body (see README, "Request"). Use --mode to
override ``options.mode`` without editing the file. The response body is printed
as JSON and may contain clinical reasoning: handle the output accordingly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .sagemaker_client import invoke_with_retry, make_runtime_client


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--stream", action="store_true", help="use response streaming (up to 8 minutes)")
    p.add_argument("--mode", choices=("full", "verdicts_only"))
    p.add_argument("--max-attempts", type=int, default=4)
    a = p.parse_args(argv)

    payload = json.loads(a.payload.read_text(encoding="utf-8"))
    if a.mode:
        payload.setdefault("options", {})["mode"] = a.mode
    runtime = make_runtime_client(a.region, streaming=a.stream)
    result, attempts = invoke_with_retry(runtime, a.endpoint_name, payload, stream=a.stream, max_attempts=a.max_attempts)
    print(json.dumps({"http_status": result.http_status, "attempts": attempts, "body": result.body}, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
