"""Stage 2: joint fine-tuning of the encoder and the cascaded AR-DRG classifier.

Objective: ``w_c * contrastive + w_m * CE(MDC) + w_b * CE(base) + w_s * CE(split)`` with sqrt-inverse-frequency
class weights. Model selection and early stopping use the macro-F1 averaged over the three heads, computed
on CLEAN validation views (what the endpoint sees); the prototype validated on randomly masked views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from drg_audit.artifacts import load_encoder_checkpoint, load_tensors, tensors_writer, write_bundle
from drg_audit.config import FinetuneConfig, to_mapping
from drg_audit.data.datasets import Batch, ContrastiveCollator, EvalCollator, encode_dataset, make_loader
from drg_audit.data.io import load_records, split_records
from drg_audit.data.vocab import build_label_maps, class_weights
from drg_audit.features import MASK, PAD, split_drg
from drg_audit.logging_utils import get_logger, log_event
from drg_audit.models.classifier import CascadedARDRGClassifier
from drg_audit.models.encoder import ClinicalSimCSEModel, build_roberta_config
from drg_audit.models.joint import DrgAuditModel, filter_encoder_state
from drg_audit.training.common import (
    CheckpointManager, EarlyStopper, autocast_context, make_grad_scaler, resolve_training_device, set_seed,
)
from drg_audit.training.losses import SimCSELoss
from drg_audit.training.pretrain import check_position_capacity

logger = get_logger("finetune")
HEADS = ("mdc", "base", "split")


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(*(t.to(device, non_blocking=True) for t in batch))


@torch.no_grad()
def evaluate_f1(model: DrgAuditModel, loader: DataLoader, device: torch.device, use_amp: bool) -> dict[str, float]:
    """Macro-F1 (in percent) per head plus their mean as ``joint``."""
    model.eval()
    preds: list[list[np.ndarray]] = [[], [], []]
    truth: list[list[np.ndarray]] = [[], [], []]
    for batch in loader:
        batch = _to_device(batch, device)
        with autocast_context(device, use_amp):
            out = model(batch.input_ids, batch.attention_mask, batch.meta, batch.version_ids)
        for h, logits in enumerate((out.mdc_logits, out.base_logits, out.split_logits)):
            preds[h].append(logits.argmax(dim=-1).cpu().numpy())
            truth[h].append(batch.targets[:, h].cpu().numpy())
    scores = {
        head: float(f1_score(np.concatenate(truth[h]), np.concatenate(preds[h]), average="macro",
                             zero_division=0)) * 100.0
        for h, head in enumerate(HEADS)
    }
    scores["joint"] = sum(scores[h] for h in HEADS) / len(HEADS)
    return scores


def run_finetuning(
    cfg: FinetuneConfig, *, input_path: str | Path, pretrained_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    runtime = cfg.runtime
    set_seed(runtime.seed)
    device = resolve_training_device(runtime.require_gpu)
    use_amp = runtime.amp and device.type == "cuda"

    pretrained = load_encoder_checkpoint(pretrained_dir)  # integrity-verified
    vocab, spec = pretrained.vocab, pretrained.tokenization
    check_position_capacity(spec.max_len, pretrained.encoder_config)

    records, report = load_records(input_path)
    labelled = [r for r in records if split_drg(r.drg) is not None]
    del records
    labels = build_label_maps(labelled)
    train_records, val_records = split_records(labelled, cfg.val_fraction, runtime.seed)
    del labelled
    train_ds = encode_dataset(train_records, vocab, labels, spec)
    val_ds = encode_dataset(val_records, vocab, labels, spec)
    train_rows, val_rows = len(train_ds), len(val_ds)
    del train_records, val_records

    pin = device.type == "cuda"
    train_loader = make_loader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=runtime.num_workers, pin_memory=pin,
        collate_fn=ContrastiveCollator(vocab[MASK], vocab[PAD], cfg.mask_probability))
    val_loader = make_loader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=runtime.num_workers, pin_memory=pin,
        collate_fn=EvalCollator(vocab[PAD]))

    encoder_config = pretrained.encoder_config
    encoder = ClinicalSimCSEModel(build_roberta_config(encoder_config))
    encoder.load_state_dict(filter_encoder_state(load_tensors(pretrained.weights_path)), strict=True)
    sizes = labels.sizes()
    classifier = CascadedARDRGClassifier(
        sizes["version"], sizes["mdc"], sizes["base"], sizes["split"],
        emb_dim=int(encoder_config["hidden_size"]), hidden_dim=cfg.classifier_hidden_dim)
    model = DrgAuditModel(encoder, classifier).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.encoder_learning_rate, "weight_decay": cfg.weight_decay},
        {"params": model.classifier.parameters(), "lr": cfg.classifier_learning_rate,
         "weight_decay": cfg.weight_decay},
    ])
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps)
    scaler = make_grad_scaler(device, use_amp)
    contrastive = SimCSELoss(cfg.temperature)
    criteria = [
        nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights(train_ds.targets[:, h], sizes[head])).to(device))
        for h, head in enumerate(HEADS)
    ]
    weights = cfg.loss_weights
    head_weights = (weights.mdc, weights.base, weights.split)
    stopper = EarlyStopper(cfg.early_stopping_patience, mode="max")
    checkpoints = CheckpointManager(runtime.checkpoint_dir, "finetune")

    history: list[dict[str, float]] = []
    best_tensors: dict[str, torch.Tensor] | None = None
    best_epoch, start_epoch = 0, 0
    resumed = checkpoints.load(device)
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        stopper.load(resumed["stopper"])
        best_tensors, best_epoch = resumed["best_tensors"], int(resumed["best_epoch"])
        history, start_epoch = list(resumed["history"]), int(resumed["epoch"])
        log_event(logger, "resumed_from_checkpoint", epoch=start_epoch)

    log_event(logger, "finetuning_started", train_rows=train_rows, val_rows=val_rows, label_sizes=sizes,
              epochs=cfg.epochs, device=str(device), amp=use_amp)

    for epoch in range(start_epoch, cfg.epochs):
        if stopper.bad_epochs >= cfg.early_stopping_patience:
            break
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                out = model(batch.input_ids, batch.attention_mask, batch.meta, batch.version_ids)
                head_losses = [
                    crit(logits, batch.targets[:, h])
                    for h, (crit, logits) in enumerate(zip(criteria, (out.mdc_logits, out.base_logits,
                                                                      out.split_logits)))
                ]
                joint = weights.contrastive * contrastive(out.embeddings) + sum(
                    w * loss for w, loss in zip(head_weights, head_losses))
            scaler.scale(joint).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += joint.item()

        scores = evaluate_f1(model, val_loader, device, use_amp)
        entry = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1),
                 **{f"f1_{k}": v for k, v in scores.items()}}
        history.append(entry)
        improved, should_stop = stopper.update(scores["joint"])
        if improved:
            best_tensors, best_epoch = {k: v.clone() for k, v in model.export_tensors().items()}, epoch + 1
        log_event(logger, "finetune_epoch", improved=improved, stall=stopper.bad_epochs, **entry)

        checkpoints.save({
            "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "stopper": stopper.state(),
            "best_tensors": best_tensors, "best_epoch": best_epoch, "history": history,
        })
        if should_stop:
            log_event(logger, "early_stop", epoch=epoch + 1, best_epoch=best_epoch)
            break

    if best_tensors is None:
        raise RuntimeError("fine-tuning finished without producing a checkpoint")
    best = next(h for h in history if h["epoch"] == best_epoch)
    training_info = {
        "best_epoch": best_epoch, "epochs_run": len(history), "best_metrics": best,
        "dataset": {"train_rows": train_rows, "val_rows": val_rows, "n_rows": report.n_rows,
                    "n_invalid": report.n_invalid},
        "pretrained_bundle_id": pretrained.manifest["bundle_id"],
        "hyperparameters": to_mapping(cfg, exclude=("serving",)),
    }
    write_bundle(
        output_dir,
        write_weights=tensors_writer(best_tensors),
        encoder_config=encoder_config,
        classifier_config={"hidden_dim": cfg.classifier_hidden_dim},
        vocab=vocab,
        labels=labels,
        inference_config=cfg.serving.to_dict(),
        tokenization=spec,
        training_info=training_info,
    )
    log_event(logger, "finetuning_finished", best_epoch=best_epoch, best_joint_f1=best["f1_joint"])
    return training_info
