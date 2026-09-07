"""Tests for V10/V11 ablations added per audit Issue 4/9."""
from __future__ import annotations

from src.eval.ablations import ABLATION_REGISTRY


def test_v10_lambda_sweep_present():
    assert "V10_lambda_sweep" in ABLATION_REGISTRY
    cfg = ABLATION_REGISTRY["V10_lambda_sweep"]
    # Per audit, sweep at minimum {0.0, 0.05, 0.1, 0.5, 1.0}
    assert 0.1 in cfg["overrides"]["lambda_values"]


def test_v11_packet_feature_ablation_present():
    assert "V11_packet_features_off" in ABLATION_REGISTRY
    cfg = ABLATION_REGISTRY["V11_packet_features_off"]
    assert cfg["overrides"]["use_packet_features"] is False
