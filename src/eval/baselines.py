"""Non-learned baselines for forecasting.

Used for V8b (downstream utility). These are the strongest naive
baselines we should beat before claiming "world model" is useful.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BaselineOutput:
    preds_onset: np.ndarray   # (N,) or (N, H)
    preds_class: np.ndarray   # (N, 7) sigmoid probs
    preds_present: np.ndarray # (N, 8) one-hot like
    name: str


def baseline_mean(present_labels: np.ndarray, num_classes_present: int = 8) -> BaselineOutput:
    """Predict the most-frequent present class for every window at every horizon."""
    N = len(present_labels)
    most_freq = int(np.bincount(present_labels, minlength=num_classes_present).argmax())
    out = np.zeros((N, num_classes_present), dtype=np.float32)
    out[:, most_freq] = 1.0
    # onset = always 0 (never predict an onset)
    onset = np.zeros(N, dtype=np.float32)
    # class = all zeros (no class prediction)
    cls = np.zeros((N, 7), dtype=np.float32)
    return BaselineOutput(onset, cls, out, name="mean")


def baseline_persistence(history_present: np.ndarray, num_classes_present: int = 8) -> BaselineOutput:
    """Repeat the last observed present label for all horizons."""
    # history_present: (L, C) or (N, L, C) — one-hot present over time
    if history_present.ndim == 2:
        L, C = history_present.shape
        N = 1
        last = history_present[-1, :]  # (C,)
        out = np.tile(last, (L, 1))  # (L, C)
    else:
        N, L, C = history_present.shape
        last = history_present[:, -1, :]  # (N, C)
        out = np.tile(last[:, None, :], (1, L, 1)).reshape(N * L, C)
    onset = np.zeros(N * L, dtype=np.float32)
    cls = np.zeros((N * L, 7), dtype=np.float32)
    return BaselineOutput(onset, cls, out, name="persistence")
