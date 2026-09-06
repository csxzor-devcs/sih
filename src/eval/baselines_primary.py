"""Primary system reference baselines.

Wraps the trained LatentDynamicsModel to evaluate it with different
rollout lengths. The difference between rollout=0 and rollout=K is
what the transition function earns.
"""
from __future__ import annotations

import numpy as np

from .baselines import BaselineOutput


def primary_with_rollout(model, x, rollout_steps: int = 0) -> BaselineOutput:
    """Run the primary system with `rollout_steps` latent rollouts.

    model: LatentDynamicsModel
    x: (B, L, F_entity)
    """
    # Lazy import: src/eval/ is not in the global torch-import allow-list.
    import torch

    model.eval()
    with torch.no_grad():
        # Run per-bin encoder + window encoder to get z_L.
        e = model.per_bin_encoder(x)          # (B, L, 64)
        z = model.window_encoder(e)            # (B, 64)
        # Optionally rollout the latent transition K times.
        for _ in range(rollout_steps):
            z = model.transition(z)            # (B, 64)
        # Heads
        onset_logit = model.heads.onset(z).squeeze(-1)   # (B,)
        class_logit = model.heads.class_head(z)           # (B, 7)
        present_logit = model.heads.present(z)            # (B, 8)
        # To numpy with appropriate activations.
        onset = torch.sigmoid(onset_logit).cpu().numpy()
        cls = torch.sigmoid(class_logit).cpu().numpy()  # multi-label
        present = torch.softmax(present_logit, dim=-1).cpu().numpy()
    name = "primary_rollout" + str(rollout_steps)
    return BaselineOutput(onset, cls, present, name=name)
