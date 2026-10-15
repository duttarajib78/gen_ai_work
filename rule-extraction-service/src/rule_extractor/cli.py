"""``rule-extractor`` command line (successor of ``run_extraction.py``)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .domain.api import JobOptions
from .errors import RuleExtractorError
from .logging_setup import configure_logging
from .settings import SWEEP_DOCUMENT_KEYS, Settings


def _add_job_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--topics", help="Comma-separated ACS codes (e.g. 0110,0002); default: all standards.")
    p.add_argument("--limit-topics", type=int, help="Cap the ACS topic count (cheap smoke test).")
    p.add_argument("--no-acs", action="store_true", help="Skip the ACS-topic pass.")
    p.add_argument("--no-tabular-sweep", action="store_true", help="Skip the page-window sweep over the Tabular Lists and Indexes.")
    p.add_argument("--sweep-documents", help=f"Comma-separated subset of {','.join(SWEEP_DOCUMENT_KEYS)}.")
    p.add_argument("--limit-sweep-windows", type=int, help="Cap sweep windows per document (cheap smoke test).")
    p.add_argument("--web-research", action="store_true", help="Enable the allow-listed web research pass (off by default; needs egress).")
    p.add_argument("--web-topics", help="Semicolon-separated topics for web research.")
    p.add_argument("--max-cost-usd", type=float, help="Stop dispatching new units once this estimated spend is reached.")


def _options(args: argparse.Namespace) -> JobOptions:
    raw: dict[str, Any] = {
        "topics": args.topics.split(",") if args.topics else None,
        "limit_topics": args.limit_topics,
        "include_acs_topics": not args.no_acs,
        "include_tabular_sweep": not args.no_tabular_sweep,
        "sweep_documents": args.sweep_documents.split(",") if args.sweep_documents else None,
        "limit_sweep_windows": args.limit_sweep_windows,
        "include_web_research": args.web_research,
        "web_research_topics": [t.strip() for t in args.web_topics.split(";")] if args.web_topics else None,
        "max_cost_usd": args.max_cost_usd,
    }
    return JobOptions.from_dict({k: v for k, v in raw.items() if v is not None})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rule-extractor", description="Upcoding / coding-compliance rule extraction (Claude on Amazon Bedrock).")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--log-level", default=os.environ.get("RULE_EXTRACTOR_LOG_LEVEL", "INFO"))
    p.add_argument("--plain-logs", action="store_true", help="Human-readable logs instead of JSON.")
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Print the plan (no API calls).")
    _add_job_flags(plan)
    run = sub.add_parser("run", help="Run a job locally against Bedrock (resumable).")
    _add_job_flags(run)
    run.add_argument("--no-resume", action="store_true", help="Ignore the checkpoint state and start a new job.")
    run.add_argument("--dry-run", action="store_true", help="Same as `plan`.")

    corpus = sub.add_parser("corpus", help="Build or verify the page-indexed corpus.")
    csub = corpus.add_subparsers(dest="corpus_command", required=True)
    build = csub.add_parser("build", help="Extract source PDFs/XML into corpus/ (incremental).")
    build.add_argument("--force", action="store_true")
    build.add_argument("--workers", type=int, default=2)
    build.add_argument("--only", help="Comma-separated document keys.")
    csub.add_parser("verify", help="Check the corpus is complete and print its statistics.")

    pkg = sub.add_parser("package", help="Assemble model.tar.gz (corpus + template + manifest) for SageMaker.")
    pkg.add_argument("--out", default="dist/model.tar.gz")
    pkg.add_argument("--model-dir", default="build/model", help="Staging directory (rebuilt each time).")
    pkg.add_argument("--template", help="Catalogue workbook (default: source_docs/rule_catalogue.xlsx).")
    pkg.add_argument("--base-rules", help="executable_rules.jsonl already represented in the template workbook.")
    pkg.add_argument("--evaluation-rules", help="executable_rules.jsonl served by the optional evaluate_episodes operation.")
    chk = sub.add_parser("check-model", help="Verify a model directory against its manifest.")
    chk.add_argument("--model-dir")

    sub.add_parser("serve", help="Container entry point: supervise the web and worker processes.")
    sub.add_parser("worker", help="Run only the queue worker.")

    ev = sub.add_parser("evaluate", help="Evaluate episodes (JSONL) against executable rules (JSONL).")
    ev.add_argument("--episodes", required=True)
    ev.add_argument("--rules", required=True)
    sub.add_parser("contract-check", help="Live SDK/Bedrock contract check (calls the models; costs a few cents).")
    return p


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_logs=not args.plain_logs,
                      stream=sys.stdout if args.command in ("serve", "worker") else sys.stderr)
    try:
        return _dispatch(args)
    except RuleExtractorError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        for line in exc.details:
            print(f"  - {line}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    settings = Settings.load()
    cmd = args.command
    if cmd in ("plan", "run"):
        options = _options(args)
        if cmd == "plan" or getattr(args, "dry_run", False):
            from .pipeline.planner import build_plan
            from .service.container import open_corpus
            plan = build_plan(options, settings, open_corpus(settings))
            _print({"summary": plan.summary, "units": [{"unit_id": u.unit_id, "kind": u.kind} for u in plan.units]})
            return 0
        from .pipeline.local_run import run_local
        view = asyncio.run(run_local(options, settings, resume=not args.no_resume,
                                     on_progress=lambda v: print(f"\r{v['progress']['done']}/{v['progress']['total']} units "
                                                                 f"(${v['cost_usd']:.2f})", end="", file=sys.stderr, flush=True)))
        print(file=sys.stderr)
        _print({k: view.get(k) for k in ("job_id", "status", "cost_usd", "progress", "summary", "failures", "artifacts", "elapsed_seconds")})
        return 0 if view["status"] in ("SUCCEEDED", "PARTIAL") else 1
    if cmd == "corpus":
        from .ingestion.corpus import build_corpus, verify_corpus
        if args.corpus_command == "build":
            manifest = build_corpus(settings, force=args.force, workers=args.workers, only=args.only.split(",") if args.only else None)
            _print({"documents": {k: v["pages"] for k, v in manifest["documents"].items()}, "known_codes": manifest["known_codes"], "mbs": manifest["mbs"]})
            return 0
        problems = verify_corpus(settings.paths.corpus_dir, min_standards=settings.chunking.min_acs_standards)
        from .ingestion.corpus import CorpusIndex
        if not problems:
            m = CorpusIndex(settings.paths.corpus_dir).manifest
            _print({"ok": True, "documents": {k: v["pages"] for k, v in m["documents"].items()}, "known_codes": m["known_codes"], "mbs": m["mbs"]})
            return 0
        _print({"ok": False, "problems": problems})
        return 1
    if cmd == "package":
        from .ingestion.corpus import verify_corpus
        from .model_package import build_model_dir, package_tar
        problems = verify_corpus(settings.paths.corpus_dir, min_standards=settings.chunking.min_acs_standards)
        if problems:
            _print({"ok": False, "corpus_problems": problems})
            return 1
        template = Path(args.template) if args.template else settings.paths.source_docs_dir / "rule_catalogue.xlsx"
        manifest = build_model_dir(settings.paths.corpus_dir, template, args.model_dir, base_rules_path=args.base_rules,
                                   evaluation_rules_path=args.evaluation_rules, force=True)
        digest = package_tar(args.model_dir, args.out)
        _print({"ok": True, "artifact": args.out, "sha256": digest, "bytes": Path(args.out).stat().st_size, "files": len(manifest["files"])})
        return 0
    if cmd == "check-model":
        from .model_package import verify_model_dir
        problems = verify_model_dir(args.model_dir or settings.paths.model_dir)
        _print({"ok": not problems, "problems": problems})
        return 0 if not problems else 1
    if cmd == "serve":
        from .service.supervisor import run_supervisor
        return run_supervisor(settings)
    if cmd == "worker":
        from .service.worker_main import main as worker_main
        return worker_main()
    if cmd == "evaluate":
        from .engine.compare import run_episode_against_ruleset
        from .engine.episode import Episode
        from .engine.grouper import load_grouper
        from .engine.rules_store import load_all
        rules, grouper = load_all(args.rules), load_grouper(settings.grouper_factory)
        for line in Path(args.episodes).read_text(encoding="utf-8").splitlines():
            if line.strip():
                episode = Episode.from_dict(json.loads(line))
                flagged = [r for r in run_episode_against_ruleset(episode, rules, grouper) if r.was_flagged]
                _print({"episode_id": episode.episode_id,
                        "flagged": [{"rule_id": r.rule_id, "corrected": r.corrected, "cost_weight_delta": r.cost_weight_delta, "notes": r.notes} for r in flagged]})
        return 0
    if cmd == "contract-check":
        from .agents.contract_check import run_contract_check
        from .agents.sdk_runner import ClaudeSdkRunner
        settings.validate_for_agents()
        report = asyncio.run(run_contract_check(ClaudeSdkRunner(settings), settings))
        _print(report.to_dict())
        return 0 if report.passed else 1
    return 2
