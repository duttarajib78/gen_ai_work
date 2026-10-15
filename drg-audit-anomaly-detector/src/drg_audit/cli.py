"""Command-line interface: ``drg-audit <command>`` (or ``python -m drg_audit.cli <command>``).

Heavy dependencies (torch, pandas, boto3) are imported inside the commands that need them, so
``verify-bundle`` and ``--help`` work in a minimal environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from drg_audit import __version__
from drg_audit.errors import DrgAuditError
from drg_audit.logging_utils import configure_logging


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------------------
def cmd_validate_data(args: argparse.Namespace) -> int:
    from drg_audit.data.io import read_table, dataframe_to_records

    _, report = dataframe_to_records(read_table(args.input), max_invalid_fraction=args.max_invalid_fraction)
    _print_json(report.to_dict())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from drg_audit.config import FinetuneConfig, PretrainConfig, load_config
    from drg_audit.training.finetune import run_finetuning
    from drg_audit.training.pretrain import run_pretraining

    out = Path(args.output_dir)
    pretrained_dir = Path(args.pretrained_dir) if args.pretrained_dir else out / "pretrain"
    if args.stage in ("pretrain", "all"):
        cfg = load_config(PretrainConfig, args.pretrain_config, args.pretrain_set)
        run_pretraining(cfg, input_path=args.input, output_dir=out / "pretrain")
    if args.stage in ("finetune", "all"):
        cfg_ft = load_config(FinetuneConfig, args.finetune_config, args.finetune_set)
        run_finetuning(cfg_ft, input_path=args.input, pretrained_dir=pretrained_dir, output_dir=out / "bundle")
    return 0


def cmd_verify_bundle(args: argparse.Namespace) -> int:
    from drg_audit.artifacts import load_bundle

    bundle = load_bundle(args.bundle_dir, verify=True)
    _print_json({
        "status": "ok", "model_version": bundle.model_version, "tokenization": bundle.tokenization.name,
        "label_sizes": bundle.labels.sizes(), "vocab_size": len(bundle.vocab),
        "created_at_utc": bundle.manifest.get("created_at_utc"), "library_versions": bundle.manifest.get("library_versions"),
    })
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    import pandas as pd

    from drg_audit.client import EndpointClient, record_to_payload
    from drg_audit.data.io import load_records

    records, report = load_records(args.input)
    if args.bundle_dir:
        from drg_audit.inference.engine import AuditEngine

        engine = AuditEngine.from_bundle_dir(args.bundle_dir)
        results = [r.to_dict() for r in engine.predict(records, args.threshold)]
    else:
        client = EndpointClient(args.endpoint_name, args.region)
        results = list(client.score_stream((record_to_payload(r) for r in records), batch_size=args.batch_size,
                                           max_workers=args.workers, confidence_threshold=args.threshold))
    frame = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)
    os.chmod(out, 0o600)  # predictions are sensitive: owner-only
    status_counts = frame["status"].value_counts().to_dict() if "status" in frame else {}
    anomalies = int(frame["is_billing_anomaly"].fillna(False).astype(bool).sum()) if "is_billing_anomaly" in frame else 0
    _print_json({"n_records": len(frame), "n_anomalies": anomalies, "status_counts": status_counts,
                 "invalid_input_rows": report.n_invalid, "output": str(out)})
    return 0


def cmd_alerts(args: argparse.Namespace) -> int:
    from drg_audit.postprocess.alerts import AlertsConfig, generate_alerts

    _print_json(generate_alerts(AlertsConfig.from_yaml(args.config), run_id=args.run_id))
    return 0


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drg-audit", description="AR-DRG billing anomaly detection")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data", help="check a dataset against the data contract")
    p.add_argument("--input", required=True)
    p.add_argument("--max-invalid-fraction", type=float, default=0.01)
    p.set_defaults(func=cmd_validate_data)

    p = sub.add_parser("train", help="pre-train the encoder and/or fine-tune the classifier")
    p.add_argument("--stage", choices=("pretrain", "finetune", "all"), default="all")
    p.add_argument("--input", required=True, help="Parquet/Feather file or directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pretrained-dir", help="encoder checkpoint for --stage finetune (default: <output>/pretrain)")
    p.add_argument("--pretrain-config", default="configs/pretrain.yaml")
    p.add_argument("--finetune-config", default="configs/finetune.yaml")
    p.add_argument("--pretrain-set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--finetune-set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("verify-bundle", help="verify a model bundle's integrity and print its metadata")
    p.add_argument("--bundle-dir", required=True)
    p.set_defaults(func=cmd_verify_bundle)

    p = sub.add_parser("score", help="score a dataset with a local bundle or a deployed endpoint")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help=".parquet or .csv")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--bundle-dir")
    target.add_argument("--endpoint-name")
    p.add_argument("--region")
    p.add_argument("--threshold", type=float)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("alerts", help="generate alert JSON lines from scored admissions")
    p.add_argument("--config", default="configs/alerts.yaml")
    p.add_argument("--run-id")
    p.set_defaults(func=cmd_alerts)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DrgAuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
