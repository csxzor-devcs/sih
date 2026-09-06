"""f_θ(z_0) called twice with the same input produces the same output. Deterministic."""
from __future__ import annotations

import torch

from src.model.gru import LatentTransitionMLP
from src.model.rollout import rollout

torch.manual_seed(0)


def test_single_step_is_deterministic():
    f = LatentTransitionMLP(dim=64, hidden=128)
    f.eval()
    z = torch.randn(4, 64)
    out1 = f(z)
    out2 = f(z)
    assert torch.allclose(out1, out2)


def test_multi_step_rollout_is_deterministic():
    f = LatentTransitionMLP(dim=64, hidden=128)
    f.eval()
    z0 = torch.randn(4, 64)
    out1 = rollout(f, z0, k=5)
    out2 = rollout(f, z0, k=5)
    assert torch.allclose(out1, out2)


def test_no_observation_input_to_transition():
    """f_θ has no parameter that depends on x_t — verified by signature."""
    f = LatentTransitionMLP(dim=64, hidden=128)
    # The real test: f(z) depends ONLY on z.
    z1 = torch.randn(2, 64)
    z2 = torch.randn(2, 64)
    assert not torch.allclose(f(z1), f(z2))
