# tests/gating/test_lambda_sweep_present.py
"""Gate: V10 lambda sweep must include λ=0.1 and at least 4 other values."""
from src.eval.ablations import ABLATION_REGISTRY


def test_v10_sweep_contains_default_lambda():
    cfg = ABLATION_REGISTRY["V10_lambda_sweep"]
    assert 0.1 in cfg["overrides"]["lambda_values"]


def test_v10_sweep_has_at_least_five_values():
    cfg = ABLATION_REGISTRY["V10_lambda_sweep"]
    assert len(cfg["overrides"]["lambda_values"]) >= 5
