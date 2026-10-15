#!/usr/bin/env python
"""Convert a prototype checkpoint into a verified serving bundle (no retraining needed).

    python scripts/convert_legacy_checkpoint.py \
        --legacy-checkpoint model_artefacts/integrated_drg_pipeline.pth \
        --vocab model_artefacts/vocab.json \
        --output-dir artifacts/bundle

Tokenization (--tokenization):
  legacy-training (default)  claims in source order and [SEP] dropped on truncation: how the prototype TRAINED
                             these weights, so serving matches training. Predictions can differ from the old
                             inference script, which sorted claims and kept [SEP] (train/serve skew).
  standard                   the spec used by newly trained models. Only choose this after comparing predictions.

Security: the legacy file is a pickle. It is loaded with weights_only=True and an allow-list containing only
RobertaConfig. On torch < 2.4 the script refuses unless --allow-unsafe-pickle is given; use that ONLY for a
file you produced yourself, because unpickling can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import RobertaConfig  # noqa: E402

from drg_audit.artifacts import load_bundle, sha256_file, tensors_writer, validate_vocab, write_bundle  # noqa: E402
from drg_audit.config import FinetuneConfig, load_config  # noqa: E402
from drg_audit.errors import ArtifactError  # noqa: E402
from drg_audit.features import CLS, SEP  # noqa: E402
from drg_audit.models.encoder import ENCODER_CONFIG_KEYS  # noqa: E402
from drg_audit.models.joint import DrgAuditModel  # noqa: E402
from drg_audit.schemas import LabelMaps, TokenizationSpec  # noqa: E402

REQUIRED_KEYS = ("encoder_config", "encoder_state", "classifier_state", "classification_maps")


def load_legacy_payload(path: Path, allow_unsafe: bool) -> dict[str, Any]:
    serialization = torch.serialization
    if hasattr(serialization, "safe_globals"):
        with serialization.safe_globals([RobertaConfig]):
            return torch.load(path, map_location="cpu", weights_only=True)
    if hasattr(serialization, "add_safe_globals"):
        serialization.add_safe_globals([RobertaConfig])
        return torch.load(path, map_location="cpu", weights_only=True)
    if not allow_unsafe:
        raise SystemExit(
            "torch >= 2.4 is required to load the legacy checkpoint safely (weights_only with an allow-listed "
            "RobertaConfig). Upgrade torch, or re-run with --allow-unsafe-pickle ONLY for a file you created.")
    print("WARNING: loading a pickle without restrictions; only do this for files you trust.", file=sys.stderr)
    return torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614


def extract_encoder_config(config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ENCODER_CONFIG_KEYS:
        if hasattr(config, key):
            value = getattr(config, key)
            if not isinstance(value, (int, float, str, bool)):
                raise ArtifactError(f"legacy encoder config value for '{key}' has an unexpected type")
            out[key] = value
    return out


def parity_check(a: DrgAuditModel, b: DrgAuditModel, vocab: dict[str, int]) -> float:
    """Both models must produce identical outputs on a fixed random batch (serialization round trip)."""
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(5, len(vocab), (4, 32), generator=gen)
    ids[:, 0], ids[:, -1] = vocab[CLS], vocab[SEP]
    mask = torch.ones_like(ids)
    meta = torch.rand(4, 4, generator=gen)
    versions = torch.zeros(4, dtype=torch.long)
    worst = 0.0
    with torch.inference_mode():
        for x, y in zip(a.eval()(ids, mask, meta, versions), b.eval()(ids, mask, meta, versions)):
            worst = max(worst, float((x - y).abs().max()))
    if worst > 1e-6:
        raise ArtifactError(f"round-trip parity check failed (max abs diff {worst:.3e})")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True, help="integrated_drg_pipeline.pth")
    parser.add_argument("--vocab", type=Path, required=True, help="vocab.json from the same training run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenization", choices=("legacy-training", "standard"), default="legacy-training")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--finetune-config", default="configs/finetune.yaml", help="source of the serving defaults")
    parser.add_argument("--allow-unsafe-pickle", action="store_true")
    args = parser.parse_args()

    payload = load_legacy_payload(args.legacy_checkpoint, args.allow_unsafe_pickle)
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise SystemExit(f"legacy checkpoint is missing keys: {missing}")

    vocab = validate_vocab(json.loads(args.vocab.read_text(encoding="utf-8")))
    encoder_config = extract_encoder_config(payload["encoder_config"])
    if encoder_config.get("vocab_size") != len(vocab):
        raise SystemExit("vocab.json does not belong to this checkpoint (vocab_size mismatch)")
    labels = LabelMaps.from_mapping(payload["classification_maps"])
    state = payload["classifier_state"]
    hidden_dim = int(state["mdc_head.weight"].shape[1])
    sizes = labels.sizes()
    shapes = {"version": state["version_embedding.weight"].shape[0], "mdc": state["mdc_head.weight"].shape[0],
              "base": state["base_head.weight"].shape[0], "split": state["split_head.weight"].shape[0]}
    if shapes != sizes:
        raise SystemExit(f"label maps {sizes} do not match classifier shapes {shapes}")

    model = DrgAuditModel.build(encoder_config, sizes, hidden_dim)
    tensors = {f"encoder.{k}": v for k, v in payload["encoder_state"].items()}
    tensors.update({f"classifier.{k}": v for k, v in state.items()})
    model.load_exported(tensors)  # strict: any missing / unexpected key aborts

    spec = (TokenizationSpec.legacy_training(args.max_len) if args.tokenization == "legacy-training"
            else TokenizationSpec.standard(args.max_len))
    serving = load_config(FinetuneConfig, args.finetune_config).serving
    write_bundle(
        args.output_dir, write_weights=tensors_writer(model.export_tensors()), encoder_config=encoder_config,
        classifier_config={"hidden_dim": hidden_dim}, vocab=vocab, labels=labels,
        inference_config=serving.to_dict(), tokenization=spec,
        training_info={"converted_from": "legacy_checkpoint", "legacy_sha256": sha256_file(args.legacy_checkpoint),
                       "tokenization_choice": args.tokenization})
    bundle = load_bundle(args.output_dir, verify=True)
    drift = parity_check(model, DrgAuditModel.from_bundle(bundle), vocab)
    print(json.dumps({"bundle_id": bundle.model_version, "tokenization": spec.name, "label_sizes": sizes,
                      "classifier_hidden_dim": hidden_dim, "round_trip_max_abs_diff": drift}, indent=2))
    print("Next: compare predictions with the legacy pipeline on a sample, then run "
          "sagemaker/build_model_package.py --bundle-dir", args.output_dir, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
