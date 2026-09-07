"""Tests for the ablation registry."""
from __future__ import annotations

from src.eval.ablations import ABLATION_REGISTRY


def test_registry_has_required_ablations():
    # Per §22
    assert "V1_per_bin_encoder" in ABLATION_REGISTRY
    assert "V4_window_size" in ABLATION_REGISTRY
    assert "V6_no_histograms" in ABLATION_REGISTRY
    assert "V7_horizons" in ABLATION_REGISTRY
    assert "V8a_transition_validity" in ABLATION_REGISTRY
    assert "V8b_downstream_utility" in ABLATION_REGISTRY


def test_ablation_config_overrides_window_size():
    cfg = ABLATION_REGISTRY["V4_window_size"]
    assert cfg["overrides"]["window_size_seconds"] == 360  # half the default
