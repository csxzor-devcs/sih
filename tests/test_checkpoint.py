"""Save/load roundtrip preserves weights and val_loss."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.model.encoder import PerBinEncoder
from src.train.checkpoint import load_checkpoint, save_checkpoint


def test_save_load_roundtrip():
    model = PerBinEncoder(F_in=10, hidden=16, out=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        # Snapshot weights before save so we can verify the load round-trip.
        snapshot = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(path, model, opt, None, None, epoch=3, val_loss=0.5)
        # Perturb the model so the load must actually restore the snapshot.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        loaded = load_checkpoint(path, model, opt)
        assert loaded["epoch"] == 3
        assert abs(loaded["val_loss"] - 0.5) < 1e-6
        # Weights restored to the snapshot taken before save.
        for k, v in snapshot.items():
            assert torch.allclose(v, model.state_dict()[k])
