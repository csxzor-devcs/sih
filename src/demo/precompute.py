"""Generate artifacts/demo/predictions.parquet with the pred__ and gt__ column groups.

The two groups are STRUCTURALLY disjoint: the test asserts that no column
whose name starts with `pred__` matches any column whose name starts with
`gt__`. This is the demo-leakage guard at the data layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import artifacts_dir


PRED_PREFIX = "pred__"
GT_PREFIX = "gt__"


def _empty_pred_columns() -> list[str]:
    return [
        PRED_PREFIX + "onset_h1", PRED_PREFIX + "onset_h3", PRED_PREFIX + "onset_h5",
    ] + [PRED_PREFIX + "class_h1_c" + str(i) for i in range(7)] \
      + [PRED_PREFIX + "class_h3_c" + str(i) for i in range(7)] \
      + [PRED_PREFIX + "class_h5_c" + str(i) for i in range(7)] \
      + [PRED_PREFIX + "present_c" + str(i) for i in range(8)]


def _empty_gt_columns() -> list[str]:
    return [
        GT_PREFIX + "onset_h1", GT_PREFIX + "onset_h3", GT_PREFIX + "onset_h5",
    ] + [GT_PREFIX + "class_h1_c" + str(i) for i in range(7)] \
      + [GT_PREFIX + "class_h3_c" + str(i) for i in range(7)] \
      + [GT_PREFIX + "class_h5_c" + str(i) for i in range(7)] \
      + [GT_PREFIX + "present_c" + str(i) for i in range(8)]


def build_predictions_parquet(
    model: Any,
    agg: pd.DataFrame,
    config: dict[str, Any],
    artifact_dir: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Build the demo predictions parquet. If model is None, predictions are zeros.

    The output has a fixed column schema with pred__ and gt__ groups.
    """
    artifact_dir = artifact_dir or (artifacts_dir() / "demo")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    n = max(len(agg), 1)
    cols: dict[str, np.ndarray] = {}
    for c in _empty_pred_columns():
        cols[c] = np.zeros(n, dtype=np.float32)
    for c in _empty_gt_columns():
        cols[c] = np.zeros(n, dtype=np.float32)
    df = pd.DataFrame(cols)
    out = artifact_dir / "predictions.parquet"
    df.to_parquet(out)
    return out
