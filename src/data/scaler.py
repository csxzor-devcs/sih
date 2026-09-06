"""Per-feature scaler. log1p on raw counts, then StandardScaler."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOG1P_COLS: list[str] = [
    "flow_count", "total_bytes", "total_packets", "byte_rate", "packet_rate",
    "dur_mean", "dur_std", "dur_p99", "iat_mean", "iat_std", "iat_max",
    "unique_peer_ports", "unique_peer_ips",
    "pkt_size_mean", "pkt_size_std", "pkt_size_p99",
    "fwd_bwd_pkt_ratio", "small_pkt_frac",
]


def fit_scaler(per_bin_df: pd.DataFrame, schema: dict[str, Any]) -> StandardScaler:
    """Fit a StandardScaler on the log1p of the LOG1P_COLS."""
    X = per_bin_df[LOG1P_COLS].astype(np.float64).copy()
    X = np.log1p(X.clip(lower=0))
    scaler = StandardScaler()
    scaler.fit(X.to_numpy())
    return scaler


def apply_scaler(per_bin_df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Apply the fitted scaler to new data. Returns float32 array [N, 18]."""
    X = per_bin_df[LOG1P_COLS].astype(np.float64).copy()
    X = np.log1p(X.clip(lower=0))
    return scaler.transform(X.to_numpy()).astype(np.float32)
