"""Tests for the primary system reference baseline (zero / full rollout)."""
import torch

from src.config import load_config, load_schema
from src.model.latent_dynamics_model import LatentDynamicsModel
from src.eval.baselines_primary import primary_with_rollout


def test_primary_with_zero_rollout():
    cfg = load_config()
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    x = torch.randn(2, 12, 104)
    out = primary_with_rollout(model, x, rollout_steps=0)
    assert out.preds_present.shape == (2, 8)
    assert out.preds_class.shape == (2, 7)
    assert out.preds_onset.shape == (2,)


def test_primary_with_rollout_uses_transition():
    cfg = load_config()
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    x = torch.randn(2, 12, 104)
    out = primary_with_rollout(model, x, rollout_steps=3)
    assert out.preds_present.shape == (2, 8)
