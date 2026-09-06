"""Sliding-window construction: 12 bins × 60 s = 12 min. Stride = 1 bin (60 s).

The window is dense at 1-bin stride. Each window has L=12 bins. For each
window, the targets (y_onset, y_class, y_present) are computed by looking
at the future bins t+1, ..., t+k for k in {1, 3, 5}.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import CANONICAL_LABELS, ATTACK_CLASS_INDICES


def build_windows(agg: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build sliding windows of L=12 bins per (host, direction).

    Returns a list of dicts with keys: host, direction, start_bin, L,
    indices (list of bin indices into agg for this window).
    """
    L = int(config["data"]["sequence_length"])
    if agg.empty:
        return []
    windows: list[dict[str, Any]] = []
    for (host, direction), group in agg.groupby(["host", "direction"]):
        bins = sorted(group["bin"].unique())
        if len(bins) < L:
            continue
        for start_idx in range(len(bins) - L + 1):
            window_bins = bins[start_idx:start_idx + L]
            windows.append({
                "host": host,
                "direction": direction,
                "start_bin": int(bins[start_idx]),
                "end_bin": int(bins[start_idx + L - 1]),
                "L": L,
                "bin_indices": window_bins,
            })
    return windows


def _attack_classes_in_bin(agg: pd.DataFrame, host: str, direction: str, bin_idx: int) -> set[int]:
    """Return the set of attack classes present in (host, direction, bin).

    Looks up the per-flow canonical-class column on `agg` if it is present.
    """
    label_col = "attack_label"
    if label_col not in agg.columns:
        return set()
    sel = agg[
        (agg["host"] == host)
        & (agg["direction"] == direction)
        & (agg["bin"] == bin_idx)
    ]
    if sel.empty:
        return set()
    return {int(x) for x in sel[label_col].unique() if int(x) != 0}


def targets_for_window(
    window: dict[str, Any],
    agg: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute y_onset, y_class, y_present for a single window.

    Onset is "first appearance" of a class in the future horizon
    [start_bin + L, start_bin + L + k]. The onset rule is strict: the class
    must NOT be present in the current window [start_bin, start_bin + L - 1].
    """
    L = int(config["data"]["sequence_length"])
    k_steps: list[int] = list(config["model"]["rollout"]["k_steps"])
    start = window["start_bin"]
    end = start + L - 1
    host = window["host"]
    direction = window["direction"]

    # Current-window class set
    current_classes: set[int] = set()
    for b in range(start, end + 1):
        current_classes |= _attack_classes_in_bin(agg, host, direction, b)

    # y_present: argmax over class counts in the current window.
    # If all are 0, return 0 (BENIGN).
    y_present = 0
    if current_classes:
        # Tie-break: lowest class index.
        y_present = min(current_classes)

    y_onset = {k: 0 for k in k_steps}
    y_class = {k: {c: 0 for c in ATTACK_CLASS_INDICES} for k in k_steps}

    for k in k_steps:
        # Onset window: future bins in (end, end + k]
        onset_classes: set[int] = set()
        for b in range(end + 1, end + 1 + k):
            onset_classes |= _attack_classes_in_bin(agg, host, direction, b)
        new_classes = onset_classes - current_classes
        y_onset[k] = 1 if new_classes else 0
        for c in ATTACK_CLASS_INDICES:
            y_class[k][c] = 1 if c in new_classes else 0

    return {
        "host": host,
        "direction": direction,
        "start_bin": start,
        "y_onset": y_onset,
        "y_class": y_class,
        "y_present": y_present,
    }
