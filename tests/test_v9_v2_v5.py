# tests/test_v9_v2_v5.py
from src.eval.ablations import ABLATION_REGISTRY


def test_v9_latent_dim_sweep_present():
    assert "V9_latent_dim_sweep" in ABLATION_REGISTRY
    cfg = ABLATION_REGISTRY["V9_latent_dim_sweep"]
    # Default latent dim is 64
    assert 64 in cfg["overrides"]["latent_dim_values"]


def test_v2_no_per_bin_encoder_present():
    assert "V2_no_per_bin_encoder" in ABLATION_REGISTRY


def test_v5_scalar_features_only_present():
    assert "V5_scalar_features_only" in ABLATION_REGISTRY
    cfg = ABLATION_REGISTRY["V5_scalar_features_only"]
    assert cfg["overrides"]["use_histograms"] is False
    assert cfg["overrides"]["use_packet_features"] is False
