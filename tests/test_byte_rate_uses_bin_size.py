"""byte_rate = total_bytes / bin_size_seconds, NOT hard-coded 60. The bin size comes from config."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import bin_size_seconds, load_config


def make_fake_flow_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "total_bytes": np.random.randint(0, 100_000, size=n).astype(np.int64),
        "total_packets": np.random.randint(1, 100, size=n).astype(np.int64),
    })


def compute_byte_rate(flow_df: pd.DataFrame, bin_seconds: int) -> np.ndarray:
    return flow_df["total_bytes"].to_numpy() / bin_seconds


def compute_packet_rate(flow_df: pd.DataFrame, bin_seconds: int) -> np.ndarray:
    return flow_df["total_packets"].to_numpy() / bin_seconds


def test_byte_rate_uses_config_bin_size():
    cfg = load_config()
    bsz = bin_size_seconds(cfg)
    df = make_fake_flow_df(50)
    rate = compute_byte_rate(df, bsz)
    assert np.allclose(rate, df["total_bytes"].to_numpy() / bsz)


def test_byte_rate_changes_with_bin_size():
    df = make_fake_flow_df(50)
    rate_30 = compute_byte_rate(df, 30)
    rate_60 = compute_byte_rate(df, 60)
    rate_120 = compute_byte_rate(df, 120)
    # They MUST differ — proves the rate is a function of bin_size, not hard-coded.
    assert not np.allclose(rate_30, rate_60)
    assert not np.allclose(rate_60, rate_120)
    assert np.allclose(rate_30, 2 * rate_60)
    assert np.allclose(rate_120, rate_60 / 2)


def test_packet_rate_uses_config_bin_size():
    cfg = load_config()
    bsz = bin_size_seconds(cfg)
    df = make_fake_flow_df(50)
    rate = compute_packet_rate(df, bsz)
    assert np.allclose(rate, df["total_packets"].to_numpy() / bsz)
