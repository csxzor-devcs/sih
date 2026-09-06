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

from src.config import F_entity
from src.data.scaler import apply_scaler
from src.data.window import build_windows, targets_for_window

DIRECTIONS = ("IN", "OUT")
NUM_ATTACK_CLASSES = 7


def collate(batch: list[Any]) -> Any:
    """Collate a list of per-window samples into a batch.

    Pads variable-length host lists and stacks tensors. Full implementation
    lands in Task 9 once F_entity is wired up with histograms.
    """
    raise NotImplementedError  # TODO(impl): Task 9 — pad/stack per-sample dicts


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

        # Vocabulary sizes are read from the schema so F_entity is stable
        # for downstream consumers (model config, tests). Task 9 will use
        # this to wire up the per-bin encoder.
        vocabs = schema["vocabularies"]
        V_p = int(vocabs["protocols"]["max_size"])
        V_s = int(vocabs["services"]["max_size"])
        V_t = int(vocabs["tcp_states"]["max_size"])
        self.F_entity: int = F_entity(schema, V_p, V_s, V_t)

        # Pre-compute targets for all windows
        self.targets: list[dict[str, Any]] = [
            targets_for_window(w, agg, config) for w in self.windows
        ]

        # Per-(host, direction, bin) feature rows indexed for fast lookup by
        # _build_window_tensor (Task 9). Stores a pd.Series per key; consumed
        # by `_build_window_tensor` (Task 9).
        self.per_bin_index: dict[tuple[str, str, int], pd.Series] = {}
        for _, row in agg.iterrows():
            key = (row["host"], row["direction"], int(row["bin"]))
            self.per_bin_index[key] = row  # store the raw row; scaler applied in __getitem__

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample: X_t, X_t_plus_k, y_onset, y_class, y_present.

        Full per-sample assembly lands in Task 9 once F_entity and histograms
        are wired up. This stub exists so callers can rely on the Dataset
        contract.
        """
        raise NotImplementedError  # TODO(impl): Task 9 — assemble per-sample tensors

    def _build_window_tensor(self, host: str, direction: str, start_bin: int) -> np.ndarray:
        """Return [L, F_entity] for the given (host, direction) starting at start_bin.

        F_entity = 2 * F_per_direction. For each of the L bins we concat the
        IN and OUT feature rows (the other direction may not exist for a
        given bin — we zero-fill).
        """
        raise NotImplementedError  # TODO(impl): wire up F_entity with histograms
