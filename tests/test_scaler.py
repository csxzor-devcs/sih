"""log1p + StandardScaler produces zero-mean unit-variance features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_schema
from src.data.scaler import LOG1P_COLS, apply_scaler, fit_scaler


def make_fake_per_bin(n: int = 200):
    return pd.DataFrame({
        c: np.random.exponential(1.0, size=n) for c in LOG1P_COLS
    })


def test_fit_and_apply_produces_correct_shape():
    schema = load_schema()
    df = make_fake_per_bin(200)
    scaler = fit_scaler(df, schema)
    X = apply_scaler(df, scaler)
    assert X.shape == (200, 18)


def test_scaled_features_have_near_zero_mean():
    df = make_fake_per_bin(2000)
    scaler = fit_scaler(df, load_schema())
    X = apply_scaler(df, scaler)
    means = X.mean(axis=0)
    assert np.all(np.abs(means) < 0.1)
