"""Stage 1: contrastive (SimCSE-style) pre-training of the clinical encoder.

Two views of every admission (clean, and randomly token-masked) must embed close together while all
other admissions in the batch are pushed apart. The principal diagnosis is always masked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from drg_audit.artifacts import tensors_writer, write_encoder_checkpoint
from drg_audit.config import PretrainConfig, to_mapping
from drg_audit.data.datasets import Batch, ContrastiveCollator, encode_dataset, make_loader
from drg_audit.data.io import load_records, split_records
from drg_audit.data.vocab import build_vocab
from drg_audit.errors import ConfigError
from drg_audit.features import MASK, PAD
from drg_audit.logging_utils import get_logger, log_event
from drg_audit.models.encoder import ClinicalSimCSEModel, build_roberta_config
from drg_audit.training.common import (
    CheckpointManager, EarlyStopper, autocast_context, cpu_clone, make_grad_scaler, resolve_training_device,
    set_seed,
)
from drg_audit.training.losses import SimCSELoss

logger = get_logger("pretrain")


def check_position_capacity(max_len: int, encoder_config: dict[str, Any]) -> None:
    """RoBERTa position ids start at ``pad_token_id + 1``; the table must be large enough."""
    if max_len + int(encoder_config["pad_token_id"]) + 1 > int(encoder_config["max_position_embeddings"]):
        raise ConfigError("tokenization.max_len does not fit in encoder.max_position_embeddings")


def _inputs(batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch.input_ids.to(device, non_blocking=True), batch.attention_mask.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, criterion: SimCSELoss, device: torch.device,
                  use_amp: bool) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        ids, mask = _inputs(batch, device)
        with autocast_context(device, use_amp):
            total += criterion(model(ids, mask)).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_space_quality(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool,
                           sample_size: int) -> tuple[float, float]:
    """Alignment (stability under dropout noise; lower is better) and uniformity (spread on the unit
    sphere; more negative is better), as in Wang & Isola (2020). Dropout is deliberately active."""
    first, second = [], []
    model.train()
    for batch in loader:
        ids, mask = _inputs(batch, device)
        with autocast_context(device, use_amp):
            v1, v2 = model(ids, mask), model(ids, mask)
        first.append(F.normalize(v1.float(), p=2, dim=-1))
        second.append(F.normalize(v2.float(), p=2, dim=-1))
    model.eval()
    x, y = torch.cat(first), torch.cat(second)
    alignment = torch.mean((x - y) ** 2).item()
    sub = x[: min(len(x), sample_size)]
    if len(sub) < 2:
        return alignment, 0.0
    uniformity = torch.log(torch.mean(torch.exp(-2 * torch.pdist(sub, p=2) ** 2)) + 1e-9).item()
    return alignment, uniformity


def run_pretraining(cfg: PretrainConfig, *, input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    runtime = cfg.runtime
    set_seed(runtime.seed)
    device = resolve_training_device(runtime.require_gpu)
    use_amp = runtime.amp and device.type == "cuda"

    records, report = load_records(input_path)
    train_records, val_records = split_records(records, cfg.val_fraction, runtime.seed)
    del records
    vocab = build_vocab(train_records, cfg.tokenization, cfg.min_code_frequency)
    encoder_config = cfg.encoder.to_encoder_config(len(vocab), vocab[PAD])
    check_position_capacity(cfg.tokenization.max_len, encoder_config)

    train_ds = encode_dataset(train_records, vocab, None, cfg.tokenization)
    val_ds = encode_dataset(val_records, vocab, None, cfg.tokenization)
    train_rows, val_rows = len(train_ds), len(val_ds)
    del train_records, val_records

    pin = device.type == "cuda"
    train_loader = make_loader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=runtime.num_workers, pin_memory=pin,
        collate_fn=ContrastiveCollator(vocab[MASK], vocab[PAD], cfg.mask_probability))
    val_loader = make_loader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=runtime.num_workers, pin_memory=pin,
        collate_fn=ContrastiveCollator(vocab[MASK], vocab[PAD], cfg.mask_probability,
                                       deterministic_seed=runtime.seed))

    model = ClinicalSimCSEModel(build_roberta_config(encoder_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps)
    scaler = make_grad_scaler(device, use_amp)
    criterion = SimCSELoss(cfg.temperature)
    stopper = EarlyStopper(cfg.early_stopping_patience, mode="min")
    checkpoints = CheckpointManager(runtime.checkpoint_dir, "pretrain")

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, start_epoch = 0, 0
    resumed = checkpoints.load(device)
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        stopper.load(resumed["stopper"])
        best_state, best_epoch = resumed["best_state"], int(resumed["best_epoch"])
        history, start_epoch = list(resumed["history"]), int(resumed["epoch"])
        log_event(logger, "resumed_from_checkpoint", epoch=start_epoch)

    log_event(logger, "pretraining_started", train_rows=train_rows, val_rows=val_rows, vocab_size=len(vocab),
              epochs=cfg.epochs, device=str(device), amp=use_amp)

    for epoch in range(start_epoch, cfg.epochs):
        if stopper.bad_epochs >= cfg.early_stopping_patience:
            break
        model.train()
        running = 0.0
        for batch in train_loader:
            ids, mask = _inputs(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                loss = criterion(model(ids, mask))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip real gradients, not loss-scaled ones (prototype bug)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val_loss = evaluate_loss(model, val_loader, criterion, device, use_amp)
        alignment, uniformity = evaluate_space_quality(model, val_loader, device, use_amp, cfg.uniformity_sample_size)
        entry = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1), "val_loss": val_loss,
                 "alignment": alignment, "uniformity": uniformity, "lr": scheduler.get_last_lr()[0]}
        history.append(entry)
        improved, should_stop = stopper.update(val_loss)
        if improved:
            best_state, best_epoch = cpu_clone(model.state_dict()), epoch + 1
        log_event(logger, "pretrain_epoch", improved=improved, stall=stopper.bad_epochs, **entry)

        checkpoints.save({
            "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "stopper": stopper.state(),
            "best_state": best_state, "best_epoch": best_epoch, "history": history,
        })
        if should_stop:
            log_event(logger, "early_stop", epoch=epoch + 1, best_epoch=best_epoch)
            break

    if best_state is None:
        raise RuntimeError("pre-training finished without producing a checkpoint")
    best = next(h for h in history if h["epoch"] == best_epoch)
    metrics = {
        "best_epoch": best_epoch, "epochs_run": len(history), "best": best,
        "dataset": {"train_rows": train_rows, "val_rows": val_rows, **{
            k: v for k, v in report.to_dict().items() if k in ("n_rows", "n_invalid")}},
        "hyperparameters": to_mapping(cfg),
    }
    write_encoder_checkpoint(output_dir, write_weights=tensors_writer(best_state), encoder_config=encoder_config,
                             vocab=vocab, tokenization=cfg.tokenization, metrics=metrics)
    log_event(logger, "pretraining_finished", best_epoch=best_epoch, best_val_loss=best["val_loss"])
    return metrics
