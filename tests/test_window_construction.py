"""Window covers exactly 12 bins × 60 s = 12 min; bin stride is 60 s; L = 12."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.window import build_windows


def make_aggregated(n_bins: int = 24, n_hosts: int = 3):
    """Return a per-(host, bin, direction) DataFrame with n_hosts × n_bins × 2 rows."""
    rows = []
    for h in range(n_hosts):
        for b in range(n_bins):
            for d in ["IN", "OUT"]:
                rows.append({
                    "host": f"host_{h}",
                    "bin": b,
                    "direction": d,
                    "flow_count": int(np.random.randint(0, 100)),
                    "total_bytes": int(np.random.randint(0, 100_000)),
                    "total_packets": int(np.random.randint(0, 1000)),
                    "byte_rate": 0.0,
                    "packet_rate": 0.0,
                    "dur_mean": 0.0, "dur_std": 0.0, "dur_p99": 0.0,
                    "iat_mean": 0.0, "iat_std": 0.0, "iat_max": 0.0,
                    "unique_peer_ports": 0, "unique_peer_ips": 0,
                    "pkt_size_mean": 0.0, "pkt_size_std": 0.0, "pkt_size_p99": 0.0,
                    "fwd_bwd_pkt_ratio": 0.0, "small_pkt_frac": 0.0,
                })
    return pd.DataFrame(rows)


def test_window_covers_12_bins():
    cfg = load_config()
    agg = make_aggregated(n_bins=24)
    windows = build_windows(agg, cfg)
    # Each window must have L=12 rows per (host, direction).
    for w in windows:
        assert w["L"] == 12


def test_window_strides_by_one_bin():
    cfg = load_config()
    agg = make_aggregated(n_bins=24)
    windows = build_windows(agg, cfg)
    # Adjacent windows for the same (host, direction) must shift by exactly 1 bin.
    by_key = {}
    for w in windows:
        by_key.setdefault((w["host"], w["direction"]), []).append(w["start_bin"])
    for key, starts in by_key.items():
        starts.sort()
        for a, b in zip(starts, starts[1:]):
            assert b - a == 1, f"Stride is {b - a} bins, expected 1"
