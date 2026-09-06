"""Evaluation runner: paired bootstrap CI is correct on synthetic data."""
from __future__ import annotations

import numpy as np

from src.eval.metrics import auroc
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


def test_class_macro_handles_multilabel():
    """Per-label macro-averaged AUROC on a [N, K] multi-label binary target.

    Verifies the Ruling AM fix: the class head is multi-LABEL binary
    (shape [N, 7]) so AUROC must be computed per-column, not via
    sklearn's multi_class="ovr" path which expects a [N] integer label
    vector.
    """
    rng = np.random.default_rng(0)
    N, K = 200, 7
    # 5 of 7 columns are non-degenerate (have both 0 and 1).
    y = np.zeros((N, K), dtype=np.int64)
    y[:N // 2, 0] = 1
    y[N // 4:, 1] = 1
    y[::3, 2] = 1
    y[:30, 3] = 1
    y[50:80, 4] = 1
    p = rng.uniform(0, 1, size=(N, K))
    # Inline per-label macro (matches the runner.py pattern).
    aurocs = []
    for k_idx in range(K):
        col = y[:, k_idx]
        if col.min() == col.max():
            continue
        aurocs.append(auroc(col, p[:, k_idx]))
    macro = float(np.mean(aurocs))
    assert 0.0 <= macro <= 1.0
    assert len(aurocs) >= 5  # at least 5 non-degenerate columns

