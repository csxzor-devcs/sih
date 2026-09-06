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
        """e: [B, L, in_size] → z_0: [B, hidden]

        Returns the last layer's final hidden state, which is the standard
        "summary" of a stacked GRU. Correct for any num_layers >= 1.
        """
        _, h = self.gru(e)  # h: [num_layers, B, hidden]
        return h[-1]  # last layer's final state → [B, hidden]


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
