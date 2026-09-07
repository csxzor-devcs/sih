# tests/gating/test_v8a_v8b_split.py
"""Gate: V8a and V8b must be registered as separate, non-overlapping experiments."""
from src.eval.ablations import ABLATION_REGISTRY


def test_v8a_evaluates_transition_validity():
    cfg = ABLATION_REGISTRY["V8a_transition_validity"]
    # V8a is a TRAINING ablation (L_transition_weight=0.0 → "does the transition function do what it claims?")
    assert cfg["kind"] == "training"
    assert cfg["overrides"]["L_transition_weight"] == 0.0


def test_v8b_evaluates_downstream_utility():
    cfg = ABLATION_REGISTRY["V8b_downstream_utility"]
    # V8b is an EVALUATION ablation (rollout_steps=0 → "is the latent dynamics useful?")
    assert cfg["kind"] == "evaluation"
    assert cfg["overrides"]["rollout_steps"] == 0


def test_v8a_and_v8b_have_non_overlapping_metrics():
    # Per Issue 6: V8a measures latent-space reconstruction, V8b measures downstream forecasting.
    # Their metrics must NOT be the same.
    a = ABLATION_REGISTRY["V8a_transition_validity"]
    b = ABLATION_REGISTRY["V8b_downstream_utility"]
    assert a["kind"] != b["kind"]
    assert "L_transition_weight" in a["overrides"]
    assert "rollout_steps" in b["overrides"]
