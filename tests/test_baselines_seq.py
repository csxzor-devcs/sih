"""Tests for sequence-model baselines (GRU no-encoder, Transformer)."""
import torch

from src.eval.baselines_seq import GRUNoEncoder, TransformerBaseline


def test_gru_no_encoder_forward_shape():
    model = GRUNoEncoder(F_entity=104, L=12, num_classes_present=8)
    x = torch.randn(4, 12, 104)
    out = model(x)
    # onset, class, present
    assert out["onset"].shape == (4,)
    assert out["class"].shape == (4, 7)
    assert out["present"].shape == (4, 8)


def test_transformer_baseline_forward_shape():
    model = TransformerBaseline(F_entity=104, L=12, num_classes_present=8, d_model=32, nhead=2, num_layers=1)
    x = torch.randn(4, 12, 104)
    out = model(x)
    assert out["onset"].shape == (4,)
    assert out["class"].shape == (4, 7)
    assert out["present"].shape == (4, 8)
