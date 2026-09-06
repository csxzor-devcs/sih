"""GRU window encoder + deterministic MLP latent transition."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUWindowEncoder(nn.Module):
    def __init__(self, in_size: int = 64, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_size, hidden_size=hidden, num_layers=num_layers, batch_first=True
        )
        self.hidden = hidden

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, L, in_size] → z_0: [B, hidden]"""
        _, h = self.gru(e)
        return h.squeeze(0)  # [B, hidden]


class LatentTransitionMLP(nn.Module):
    """Deterministic, free-running latent transition: z_{t+1} = f_θ(z_t)."""

    def __init__(self, dim: int = 64, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
