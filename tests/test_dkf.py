# tests/test_dkf.py
import torch
from src.model.dkf import DKFTransition


def test_dkf_transition_output_shapes():
    tr = DKFTransition(latent_dim=64)
    z = torch.randn(4, 64)
    mu, logvar = tr(z)
    assert mu.shape == (4, 64)
    assert logvar.shape == (4, 64)


def test_dkf_sample_shape_and_finite():
    tr = DKFTransition(latent_dim=64)
    z = torch.randn(4, 64)
    z_next = tr.sample(z)
    assert z_next.shape == (4, 64)
    assert torch.isfinite(z_next).all()
