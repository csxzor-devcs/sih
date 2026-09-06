"""The 5 packet-level features are derived correctly from per-flow statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.aggregate import compute_packet_features


def make_fake_flows(fwd_max, bwd_max, fwd_pkts, bwd_pkts):
    return pd.DataFrame({
        "Fwd Pkt Len Max": fwd_max,
        "Bwd Pkt Len Max": bwd_max,
        "Total Fwd Packet": fwd_pkts,
        "Total Bwd packet": bwd_pkts,
    })


def test_pkt_size_p99_uses_percentile():
    # fwd_max holds 100 values 1..100; bwd_max is all-NaN (no bwd flows). The
    # implementation concatenates Fwd Pkt Len Max + Bwd Pkt Len Max and drops
    # NaN entries, so the 99th percentile of the pooled sizes must match
    # np.percentile(1..100, 99).
    flows = make_fake_flows(
        fwd_max=np.arange(1, 101, dtype=np.float64),
        bwd_max=np.full(100, np.nan),
        fwd_pkts=np.full(100, 1),
        bwd_pkts=np.full(100, 1),
    )
    feats = compute_packet_features(flows)
    assert feats["pkt_size_p99"] == np.percentile(np.arange(1, 101), 99)


def test_fwd_bwd_pkt_ratio_uses_safe_max():
    flows = make_fake_flows(
        fwd_max=np.array([100.0, 200.0]),
        bwd_max=np.array([50.0, 75.0]),
        fwd_pkts=np.array([10, 20]),
        bwd_pkts=np.array([0, 5]),
    )
    feats = compute_packet_features(flows)
    # fwd_pkts sum = 30, bwd_pkts sum = 5 → 30 / max(5, 1) = 30 / 5 = 6
    assert feats["fwd_bwd_pkt_ratio"] == 6.0


def test_small_pkt_frac_uses_64_threshold():
    # All arrays must be the same length so DataFrame() can build. 2 flows:
    # flow 0 has fwd_max=32, bwd_max=10; flow 1 has fwd_max=64, bwd_max=1000.
    # sizes pooled = [32, 10, 64, 1000] → small (<64) = 2/4 = 0.5
    flows = make_fake_flows(
        fwd_max=np.array([32.0, 64.0]),
        bwd_max=np.array([10.0, 1000.0]),
        fwd_pkts=np.array([1, 1]),
        bwd_pkts=np.array([1, 1]),
    )
    feats = compute_packet_features(flows)
    sizes = np.array([32.0, 10.0, 64.0, 1000.0])
    expected = float(np.mean(sizes < 64.0))
    assert abs(feats["small_pkt_frac"] - expected) < 1e-6


def test_empty_flows_returns_zeros():
    flows = make_fake_flows(fwd_max=np.array([]), bwd_max=np.array([]), fwd_pkts=np.array([]), bwd_pkts=np.array([]))
    feats = compute_packet_features(flows)
    assert feats["pkt_size_mean"] == 0.0
    assert feats["fwd_bwd_pkt_ratio"] == 0.0
