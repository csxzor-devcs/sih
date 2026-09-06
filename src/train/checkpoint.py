"""Save/load checkpoints. All paths relative to SIH_ARTIFACTS_DIR or caller-supplied."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    vocab: dict | None,
    epoch: int,
    val_loss: float,
    config: dict | None = None,
) -> None:
    """Persist model + optimizer + scaler + vocab + epoch + val_loss to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scaler": scaler,
        "vocab": vocab,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "config": config,
    }, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Load a checkpoint produced by `save_checkpoint` into `model` (and `optimizer` if given)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
