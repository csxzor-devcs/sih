"""Forecast heads. Each horizon reads from z_{t+k}."""
from __future__ import annotations

import torch
import torch.nn as nn


class OnsetHead(nn.Module):
    """Per-horizon onset head: scalar logit (sigmoid outside the loss)."""

    def __init__(self, trunk: tuple[int, int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk[0], trunk[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk[1], 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, 64] → [B, 1]"""
        return self.net(z)


class ClassHead(nn.Module):
    """Per-horizon multi-label class head: 7 logits (sigmoid outside the loss)."""

    def __init__(self, trunk: tuple[int, int] = (64, 32), dropout: float = 0.2, n: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk[0], trunk[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk[1], n),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, 64] → [B, 7]"""
        return self.net(z)


class PresentHead(nn.Module):
    """Diagnostic 8-way softmax presence head (BENIGN=0)."""

    def __init__(self, trunk: tuple[int, int] = (64, 16), n: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk[0], trunk[1]),
            nn.ReLU(),
            nn.Linear(trunk[1], n),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, 64] → [B, 8] (logits)"""
        return self.net(z)
