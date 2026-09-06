"""Dependency injection: load the model, scaler, vocab, schema once at startup."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import artifacts_dir, load_config, load_schema
from src.model.latent_dynamics_model import LatentDynamicsModel


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()


@lru_cache(maxsize=1)
def get_schema() -> dict[str, Any]:
    return load_schema()


@lru_cache(maxsize=1)
def get_model() -> LatentDynamicsModel:
    """Load the model. Falls back to a randomly-initialized model in the demo
    if no checkpoint is on disk -- the API is still up so the dashboard renders."""
    import torch  # lazy import: src/api/ is not in the torch import allow-list

    cfg = get_config()
    schema = get_schema()
    V_p, V_s, V_t = 17, 12, 5
    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    ckpt = artifacts_dir() / "checkpoints" / "0" / "best.pt"
    if ckpt.exists():
        from src.train.checkpoint import load_checkpoint
        load_checkpoint(ckpt, model)
    model.eval()
    return model


def get_device() -> "torch.device":
    """Return the torch device. Lazy import keeps torch out of module top-level."""
    import torch  # lazy import: src/api/ is not in the torch import allow-list

    cfg = get_config()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
