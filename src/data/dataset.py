"""Torch Dataset wrapping the per-(host, bin, direction) aggregation.

Each sample corresponds to ONE (host, direction) sliding window. The dataset
yields:
  - X_t:           [L, F_entity]         — current window
  - X_t_plus_k:    dict[k -> [L, F_entity]]  — future windows for L_transition
  - y_onset:       dict[k -> int]        — onset label for each k
  - y_class:       dict[k -> np.ndarray of shape (7,)]
  - y_present:     int                   — current-window argmax
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.scaler import apply_scaler
from src.data.window import build_windows, targets_for_window

DIRECTIONS = ("IN", "OUT")
NUM_ATTACK_CLASSES = 7


class WindowDataset(Dataset):
    def __init__(
        self,
        agg: pd.DataFrame,
        config: dict[str, Any],
        scaler,
        schema: dict[str, Any],
        vocab: dict[str, dict[str, int]],
        include_future_windows: bool = True,
    ):
        self.agg = agg
        self.config = config
        self.scaler = scaler
        self.schema = schema
        self.vocab = vocab
        self.L = int(config["data"]["sequence_length"])
        self.k_steps: list[int] = list(config["model"]["rollout"]["k_steps"])
        self.include_future_windows = include_future_windows
        self.windows = build_windows(agg, config)

        # Pre-compute targets for all windows
        self.targets: list[dict[str, Any]] = [
            targets_for_window(w, agg, config) for w in self.windows
        ]

        # Pre-compute the scaled per-bin tensor per (host, direction)
        # This is the [F_per_direction] vector for each (host, direction, bin).
        # In this MVP we represent the per-bin features as a single row per
        # (host, direction, bin) and look them up by index.
        self.per_bin_index: dict[tuple[str, str, int], np.ndarray] = {}
        for _, row in agg.iterrows():
            key = (row["host"], row["direction"], int(row["bin"]))
            self.per_bin_index[key] = row  # store the raw row; scaler applied in __getitem__

    def __len__(self) -> int:
        return len(self.windows)

    def _build_window_tensor(self, host: str, direction: str, start_bin: int) -> np.ndarray:
        """Return [L, F_entity] for the given (host, direction) starting at start_bin.

        F_entity = 2 * F_per_direction. For each of the L bins we concat the
        IN and OUT feature rows (the other direction may not exist for a
        given bin — we zero-fill).
        """
        raise NotImplementedError  # TODO(impl): wire up F_entity with histograms
