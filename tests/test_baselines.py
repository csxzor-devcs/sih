"""Non-learned baselines (mean, persistence) behave as specified on synthetic inputs."""
from __future__ import annotations

import numpy as np

from src.eval.baselines import baseline_mean, baseline_persistence


def test_baseline_mean_predicts_majority_class():
    rng = np.random.default_rng(0)
    # 10 benign, 2 attack windows
    present = np.array([0]*10 + [1]*2)
    out = baseline_mean(present, num_classes_present=8)
    # Most-frequent present label is 0 (BENIGN)
    assert out.preds_present.shape == (12, 8)
    assert np.all(out.preds_present[:, 0] == 1.0)


def test_baseline_persistence_repeats_last_window():
    rng = np.random.default_rng(0)
    # Last window has present = [0,0,0,0,0,0,0,1] (BENIGN=0, attack1 class=7)
    last = np.zeros((1, 8))
    last[0, 7] = 1.0
    history = np.zeros((11, 8))  # 11 prior windows
    history_full = np.concatenate([history, last], axis=0)  # L=12
    out = baseline_persistence(history_full, num_classes_present=8)
    # All predictions should equal last
    assert np.array_equal(out.preds_present, np.tile(last, (12, 1)))
