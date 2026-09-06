"""Metrics and paired bootstrap CI are correct on synthetic data."""
from __future__ import annotations

import numpy as np

from src.eval.metrics import auroc, auprc, brier, ece, paired_bootstrap_ci


def test_auroc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, s) == 1.0


def test_auroc_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    s = rng.uniform(0, 1, size=1000)
    assert 0.4 < auroc(y, s) < 0.6


def test_auprc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auprc(y, s) == 1.0


def test_brier_perfect():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert brier(y, p) == 0.0


def test_ece_well_calibrated():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.05] * 50 + [0.95] * 50)
    assert ece(y, p, n_bins=10) < 0.05


def test_paired_bootstrap_ci_includes_true_mean():
    rng = np.random.default_rng(0)
    delta = rng.normal(0.1, 1.0, size=500)
    lo, hi = paired_bootstrap_ci(delta, n_samples=2000, rng=rng)
    assert lo < 0.1 < hi
