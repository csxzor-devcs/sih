"""Evaluation runner: paired bootstrap CI is correct on synthetic data."""
from __future__ import annotations

import numpy as np

from src.eval.runner import evaluate, paired_bootstrap_diff


def test_paired_bootstrap_diff_zero_when_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = a.copy()
    lo, hi = paired_bootstrap_diff(a, b, n_boot=200, seed=0)
    # Identical distributions should give diff ~ 0
    assert abs(lo) < 0.1 and abs(hi) < 0.1


def test_paired_bootstrap_diff_positive_when_a_better():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.2, 200)  # mostly 1
    b = rng.normal(0.3, 0.2, 200)  # mostly 0
    lo, hi = paired_bootstrap_diff(a, b, n_boot=200, seed=0)
    assert lo > 0.0
