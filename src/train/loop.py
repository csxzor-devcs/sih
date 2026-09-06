"""Training loop. Stage 1: direct end-to-end with L_total.

L_total = L_onset + L_class + L_present + lambda * L_transition.

The transition gets TWO signals: end-to-end through heads, and L_transition
regression to the future encoder (the future encoder is detached).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import L_transition_weight, device_for
from src.model.latent_dynamics_model import LatentDynamicsModel
from src.model.losses import class_loss, onset_loss, present_loss, transition_loss
from src.model.rollout import rollout
from src.train.checkpoint import save_checkpoint
from src.train.seed import seed_everything


def train_one_epoch(
    model: LatentDynamicsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_transition: float = 0.1,
) -> dict[str, float]:
    """Run one pass over `loader`, returning mean per-component losses."""
    model.train()
    totals = {"L_onset": 0.0, "L_class": 0.0, "L_present": 0.0, "L_transition": 0.0, "L_total": 0.0}
    n = 0
    for batch in loader:
        x_t = batch["x_t"].to(device)
        y_onset = {k: v.to(device) for k, v in batch["y_onset"].items()}
        y_class = {k: v.to(device) for k, v in batch["y_class"].items()}
        y_present = batch["y_present"].to(device)
        x_future = {k: v.to(device) for k, v in batch["x_future"].items()}

        # Ruling R: rollout_steps=1 so heads and L_transition see the rolled z,
        # not z0. Without this L_transition collapses to ||z0 - z_actual||.
        out = model(x_t, rollout_steps=1)
        # L_onset
        l_onset = sum(onset_loss(out["onset_logits"][k], y_onset[k]) for k in model.k_steps)
        # L_class
        l_class = sum(class_loss(out["class_logits"][k], y_class[k]) for k in model.k_steps)
        # L_present
        l_present = present_loss(out["present_logits"], y_present)
        # L_transition: future windows, encoder detached on the future side
        l_transition = 0.0
        for k in model.k_steps:
            e_f = model.per_bin_encoder(x_future[k]).detach()
            z_actual = model.window_encoder(e_f)
            z_hat = out["z_rolled"][k]
            l_transition = l_transition + transition_loss(z_hat, z_actual)
        l_transition = l_transition / len(model.k_steps)

        l_total = l_onset + l_class + l_present + lambda_transition * l_transition
        optimizer.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x_t.size(0)
        totals["L_onset"] += float(l_onset.item()) * bs
        totals["L_class"] += float(l_class.item()) * bs
        totals["L_present"] += float(l_present.item()) * bs
        totals["L_transition"] += float(l_transition.item()) * bs
        totals["L_total"] += float(l_total.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(
    model: LatentDynamicsModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    artifact_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Top-level training entry: seeds, builds optimizer/scheduler, runs epochs, saves best."""
    seed_everything(seed)
    device = device_for(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
    )
    lambda_transition = L_transition_weight(config)
    best_val = float("inf")
    best_path = artifact_dir / "checkpoints" / str(seed) / "best.pt"
    for epoch in range(int(config["training"]["epochs"])):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, lambda_transition)
        val_metrics = train_one_epoch(model, val_loader, optimizer, device, lambda_transition)
        scheduler.step(val_metrics["L_onset"])
        if val_metrics["L_onset"] < best_val:
            best_val = val_metrics["L_onset"]
            save_checkpoint(best_path, model, optimizer, None, None, epoch, best_val, config)
    return {"best_val": best_val, "best_path": str(best_path)}
