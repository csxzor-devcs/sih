"""Tests for classical ML baselines (LogReg, RandomForest, XGBoost)."""
import numpy as np
import pytest

from src.eval.baselines_ml import baseline_logreg, baseline_random_forest


def test_logreg_learns_simple_boundary():
    rng = np.random.default_rng(0)
    # 50 windows, 12-min history, 8 features
    X = rng.normal(0, 1, (50, 12 * 8)).astype(np.float32)
    # Label = positive if first feature > 0
    y = (X[:, 0] > 0).astype(np.int64)
    out = baseline_logreg(X, y, num_classes_present=2)
    # Output shape
    assert out.preds_present.shape == (50, 2)
    # Should beat 50% on training set (sanity)
    preds = out.preds_present.argmax(axis=1)
    assert (preds == y).mean() > 0.7


def test_random_forest_handles_multiclass():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 12 * 4)).astype(np.float32)
    y = rng.integers(0, 5, 80)
    out = baseline_random_forest(X, y, num_classes_present=5)
    assert out.preds_present.shape == (80, 5)
    # At minimum, predictions are valid class indices
    assert out.preds_present.sum(axis=1).mean() == pytest.approx(1.0, abs=0.01)
