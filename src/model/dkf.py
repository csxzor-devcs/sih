"""Deep Kalman Filter transition as a drop-in replacement for LatentTransitionMLP.

Per Tier 3: tests whether deterministic MLP is enough or whether stochastic
dynamics is materially better. Same interface as LatentTransitionMLP:
  z_next = transition(z)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class DKFTransition(nn.Module):
    """Stochastic transition: z_{t+1} ~ N(μ_θ(z_t), σ_θ(z_t))."""
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        return self.mu_head(h), self.logvar_head(h)

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
