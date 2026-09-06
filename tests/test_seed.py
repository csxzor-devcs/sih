"""seed_everything makes random sequences reproducible."""
from __future__ import annotations

import random

import numpy as np
import torch

from src.train.seed import seed_everything


def test_seed_makes_torch_random_reproducible():
    seed_everything(42)
    a1 = torch.randn(10)
    seed_everything(42)
    a2 = torch.randn(10)
    assert torch.allclose(a1, a2)


def test_seed_makes_numpy_reproducible():
    seed_everything(42)
    a1 = np.random.rand(10)
    seed_everything(42)
    a2 = np.random.rand(10)
    assert np.allclose(a1, a2)


def test_seed_sets_cudnn_flags():
    seed_everything(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
