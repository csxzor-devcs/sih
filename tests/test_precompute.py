"""Precompute produces a parquet with pred__ and gt__ column groups that are disjoint."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.demo.precompute import GT_PREFIX, PRED_PREFIX, build_predictions_parquet


def test_pred_and_gt_columns_disjoint():
    with tempfile.TemporaryDirectory() as d:
        out = build_predictions_parquet(
            model=None, agg=pd.DataFrame(), config={}, artifact_dir=Path(d)
        )
        df = pd.read_parquet(out)
        pred_cols = [c for c in df.columns if c.startswith(PRED_PREFIX)]
        gt_cols = [c for c in df.columns if c.startswith(GT_PREFIX)]
        assert len(pred_cols) > 0
        assert len(gt_cols) > 0
        # Disjoint: no column is in both groups
        assert set(pred_cols).isdisjoint(set(gt_cols))


def test_no_label_substring_in_columns():
    with tempfile.TemporaryDirectory() as d:
        out = build_predictions_parquet(
            model=None, agg=pd.DataFrame(), config={}, artifact_dir=Path(d)
        )
        df = pd.read_parquet(out)
        for c in df.columns:
            low = c.lower()
            assert "label" not in low, "Column " + c + " contains 'label'"
            assert "ground_truth" not in low
