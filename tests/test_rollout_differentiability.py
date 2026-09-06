"""Gradient flows from heads back through rollout to encoder."""
from __future__ import annotations

import torch

from src.config import F_entity, load_schema
from src.model.encoder import PerBinEncoder
from src.model.gru import GRUWindowEncoder, LatentTransitionMLP
from src.model.heads import OnsetHead
from src.model.rollout import rollout

torch.manual_seed(0)


def test_gradient_flows_through_rollout_to_encoder():
    schema = load_schema()
    F_in = F_entity(schema, 32, 16, 8)  # 148 — matches PerBinEncoder input dim
    encoder = PerBinEncoder(F_in=F_in, hidden=128, out=64)
    gru = GRUWindowEncoder(in_size=64, hidden=64)
    transition = LatentTransitionMLP(dim=64, hidden=128)
    head = OnsetHead(trunk=(64, 32), dropout=0.0)

    x = torch.randn(2, 12, F_in, requires_grad=False)
    e = encoder(x)
    z0 = gru(e)
    z5 = rollout(transition, z0, k=5)
    logit = head(z5).sum()
    logit.backward()

    # Every parameter must have a non-None grad.
    for name, p in encoder.named_parameters():
        assert p.grad is not None, "encoder." + name + " has no grad"
    for name, p in gru.named_parameters():
        assert p.grad is not None, "gru." + name + " has no grad"
    for name, p in transition.named_parameters():
        assert p.grad is not None, "transition." + name + " has no grad"
    for name, p in head.named_parameters():
        assert p.grad is not None, "head." + name + " has no grad"


def test_no_detach_in_rollout():
    """rollout() must not call .detach() — gradients must flow."""
    f = LatentTransitionMLP(dim=64, hidden=128)
    z0 = torch.randn(2, 64, requires_grad=True)
    z5 = rollout(f, z0, k=5)
    z5.sum().backward()
    assert z0.grad is not None
