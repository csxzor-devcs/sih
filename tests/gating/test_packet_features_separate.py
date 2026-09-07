# tests/gating/test_packet_features_separate.py
"""Gate: V11 packet-feature ablation is separate from V6 histograms."""
from src.eval.ablations import ABLATION_REGISTRY


def test_v11_is_separate_from_v6():
    assert "V11_packet_features_off" in ABLATION_REGISTRY
    assert "V6_no_histograms" in ABLATION_REGISTRY
    v6 = ABLATION_REGISTRY["V6_no_histograms"]
    v11 = ABLATION_REGISTRY["V11_packet_features_off"]
    # They must NOT both flip the same flag
    assert v6["overrides"].get("use_histograms") is False
    assert v11["overrides"].get("use_packet_features") is False
    assert "use_packet_features" not in v6["overrides"]
    assert "use_histograms" not in v11["overrides"]
