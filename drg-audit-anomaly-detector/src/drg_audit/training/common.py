"""Shared training utilities."""

from __future__ import annotations

import contextlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drg_audit.errors import ConfigError


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_training_device(require_gpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise ConfigError("runtime.require_gpu is true but no GPU was detected; refusing to train on CPU")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool):
    if not (enabled and device.type == "cuda"):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


class EarlyStopper:
    """Tracks the best metric; ``update`` returns ``(improved, should_stop)``."""

    def __init__(self, patience: int, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience, self.mode = patience, mode
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        better = self.best is None or (value < self.best if self.mode == "min" else value > self.best)
        if better:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return better, self.bad_epochs >= self.patience

    def state(self) -> dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load(self, state: dict[str, Any]) -> None:
        self.best, self.bad_epochs = state["best"], int(state["bad_epochs"])


class CheckpointManager:
    """Atomic, resumable training-state checkpoints (spot-instance safe).

    State is loaded with ``weights_only=True`` so a tampered checkpoint cannot execute code.
    """

    def __init__(self, directory: str | None, name: str) -> None:
        self.path = Path(directory) / f"{name}.pt" if directory else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, device: torch.device) -> dict[str, Any] | None:
        if self.path is None or not self.path.exists():
            return None
        try:
            return torch.load(self.path, map_location=device, weights_only=True)
        except Exception as exc:  # corrupt / incompatible checkpoint: fail loudly, never silently restart
            raise ConfigError(f"cannot resume from checkpoint {self.path.name}; delete it to start fresh") from exc

    def save(self, state: dict[str, Any]) -> None:
        if self.path is None:
            return
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(state, tmp_name)
            os.replace(tmp_name, self.path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)


def cpu_clone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detached CPU copy (``clone`` matters on CPU, where ``.cpu()`` would alias live parameters)."""
    return {k: v.detach().cpu().clone() for k, v in state.items()}
