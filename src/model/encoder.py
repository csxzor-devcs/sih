"""Per-bin MLP encoder: F_entity → 128 → 64 with ReLU + Dropout(0.1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class PerBinEncoder(nn.Module):
    def __init__(self, F_in: int, hidden: int = 128, out: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, F_in] → [B, L, out]"""
        B, L, F = x.shape
        out = self.net(x.reshape(B * L, F))
        return out.reshape(B, L, -1)
