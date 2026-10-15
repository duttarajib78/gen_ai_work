"""SageMaker Training Job entry point (script mode).

Runs inside the training container. Hyper-parameters arrive as CLI arguments:
    --stage {pretrain,finetune,all}            default: all
    --pretrain_config / --finetune_config      config file names inside the bundled configs/ directory
    --pretrain_overrides / --finetune_overrides  comma-separated dotted.key=value pairs (no commas inside values)

Channels:  train (required, Parquet/Feather)   pretrained (only for --stage finetune; encoder checkpoint or the
           model.tar.gz of a previous pretrain job)
Outputs:   the serving bundle is written at the ROOT of /opt/ml/model, so the job's model.tar.gz can be passed
           straight to sagemaker/build_model_package.py --model-tar. With --stage all the intermediate encoder
           checkpoint goes to /opt/ml/output/data/pretrain.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _candidate in (HERE / "src", HERE.parent / "src"):  # bundle layout: ./src   repo layout: ../src
    if (_candidate / "drg_audit").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from drg_audit.archive import safe_extract_tar  # noqa: E402
from drg_audit.config import FinetuneConfig, PretrainConfig, load_config  # noqa: E402
from drg_audit.logging_utils import configure_logging  # noqa: E402

CHECKPOINT_DIR = Path("/opt/ml/checkpoints")


def _config_path(name: str) -> str:
    if Path(name).name != name:
        raise SystemExit("config names must be plain file names inside the bundled configs/ directory")
    for base in (HERE / "configs", HERE.parent / "configs"):
        if (base / name).is_file():
            return str(base / name)
    raise SystemExit(f"config '{name}' was not found in the bundled configs/ directory")


def _split(csv: str) -> list[str]:
    return [part.strip() for part in csv.split(",") if part.strip()]


def _resolve_pretrained(scratch: Path) -> Path:
    channel = os.environ.get("SM_CHANNEL_PRETRAINED")
    if not channel:
        raise SystemExit("--stage finetune needs a 'pretrained' input channel")
    archive = Path(channel) / "model.tar.gz"
    if archive.is_file():
        safe_extract_tar(archive, scratch)  # artifacts from another job are untrusted input
        return scratch
    return Path(channel)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("pretrain", "finetune", "all"), default="all")
    parser.add_argument("--pretrain_config", default="pretrain.yaml")
    parser.add_argument("--finetune_config", default="finetune.yaml")
    parser.add_argument("--pretrain_overrides", default="")
    parser.add_argument("--finetune_overrides", default="")
    args, _ = parser.parse_known_args(argv)  # tolerate framework-injected sagemaker_* arguments

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    checkpoint_override = [f"runtime.checkpoint_dir={CHECKPOINT_DIR}"] if CHECKPOINT_DIR.is_dir() else []
    pretrain_out = output_dir / "pretrain" if args.stage == "all" else model_dir

    if args.stage in ("pretrain", "all"):
        from drg_audit.training.pretrain import run_pretraining

        cfg = load_config(PretrainConfig, _config_path(args.pretrain_config),
                          [*checkpoint_override, *_split(args.pretrain_overrides)])
        run_pretraining(cfg, input_path=train_dir, output_dir=pretrain_out)

    if args.stage in ("finetune", "all"):
        from drg_audit.training.finetune import run_finetuning

        with tempfile.TemporaryDirectory() as scratch:
            pretrained = pretrain_out if args.stage == "all" else _resolve_pretrained(Path(scratch))
            cfg_ft = load_config(FinetuneConfig, _config_path(args.finetune_config),
                                 [*checkpoint_override, *_split(args.finetune_overrides)])
            run_finetuning(cfg_ft, input_path=train_dir, pretrained_dir=pretrained, output_dir=model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
