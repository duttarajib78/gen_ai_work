"""`dx-support <command>` -- one entry point for every tool."""
from __future__ import annotations

import importlib
import sys

from dx_support import __version__

COMMANDS = {
    "build": ("dx_support.cli.build", "main", "Build supporting lists (resumable, cost-capped). Use --for-service to pre-warm the endpoint."),
    "estimate": ("dx_support.cli.estimate", "main", "Project calls, cost and time before a build (no model calls)."),
    "compile": ("dx_support.cli.compile", "main", "Turn stored results into CSV lists, the audit and workbooks."),
    "status": ("dx_support.cli.status", "main", "Read-only progress report for a build."),
    "validate-retrieval": ("dx_support.cli.validate_retrieval", "main", "Measure how much recall routing loses versus an exhaustive review."),
    "validate-judge": ("dx_support.cli.validate_judge", "main", "Judge accuracy against a clinician-reviewed gold file (or repeatability)."),
    "doctor": ("dx_support.cli.doctor", "main", "Preflight: configuration, data, store, IAM and a real probe of every model."),
    "bedrock-arns": ("dx_support.cli.tools", "bedrock_arns", "Print the exact IAM resource ARNs the endpoint needs for your configured models."),
    "prewarm": ("dx_support.cli.tools", "prewarm", "Pre-warm the endpoint: build lists into its result store with a SageMaker Processing job."),
    "invoke": ("dx_support.cli.tools", "invoke", "Call a deployed endpoint (waits on the streaming API for uncached work)."),
    "make-sample-data": ("dx_support.cli.tools", "make_sample_data", "Write invented, non-clinical sample reference files."),
    "serve-local": ("dx_support.cli.tools", "serve_local", "Run the HTTP service locally (development server)."),
}


def usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = [f"dx-support {__version__}", "", "usage: dx-support <command> [options]   (add -h after a command for its options)", "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_m, _f, desc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0 if argv else 2
    if argv[0] in ("-V", "--version", "version"):
        print(__version__)
        return 0
    if argv[0] not in COMMANDS:
        print(f"unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module, func, _ = COMMANDS[argv[0]]
    rc = getattr(importlib.import_module(module), func)(argv[1:])
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
