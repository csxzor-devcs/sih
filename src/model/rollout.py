"""K-step latent rollout. Free-running; differentiable end-to-end."""
from __future__ import annotations

import torch
import torch.nn as nn


def rollout(f: nn.Module, z0: torch.Tensor, k: int) -> torch.Tensor:
    """Apply f k times starting from z0.

    z0: [B, dim]  →  z_k: [B, dim]
    The rollout is differentiable end-to-end (no .detach()).
    """
    z = z0
    for _ in range(k):
        z = f(z)
    return z
