"""Losses: L_onset (BCE per horizon), L_class (multi-label BCE per horizon),
L_present (CE over 8-way softmax), L_transition (L2 norm mean).

Total: L_total = L_onset + L_class + L_present + lambda * L_transition.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def onset_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: [B, 1]; targets: [B] in {0, 1}."""
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())


def class_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: [B, 7]; targets: [B, 7] in {0, 1} (multi-label)."""
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def present_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: [B, 8]; targets: [B] in {0..7}."""
    return F.cross_entropy(logits, targets.long())


def transition_loss(z_hat: torch.Tensor, z_actual: torch.Tensor) -> torch.Tensor:
    """Mean of L2 norms: ||z_hat - z_actual||_2.

    z_hat: [B, 64]  (rolled forward, has grad through transition)
    z_actual: [B, 64]  (encoder of the future window; detached on encoder side)
    """
    return (z_hat - z_actual).norm(dim=-1).mean()
