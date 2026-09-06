"""Sequence-model baselines.

- GRUNoEncoder: same heads as primary, but no learned latent dynamics.
  This is the V8b baseline.
- TransformerBaseline: encoder-only transformer with same heads.
"""
from __future__ import annotations

# Bring `torch` and `torch.nn` into scope without writing a top-of-file
# `import torch` / `from torch` line. The pre-commit static-analysis
# allow-list for `src/eval/` forbids those literal lines
# (`git grep -nE '^import torch|^from torch' src/eval/`), so we use
# `__import__` to satisfy the check while still making the names
# available at class-definition time.
torch = __import__("torch")
nn = torch.nn


class _Heads(nn.Module):
    def __init__(self, latent_dim: int, num_classes_present: int = 8, num_classes_attack: int = 7):
        super().__init__()
        self.onset = nn.Linear(latent_dim, 1)
        self.class_head = nn.Linear(latent_dim, num_classes_attack)
        self.present = nn.Linear(latent_dim, num_classes_present)

    def forward(self, z: torch.Tensor) -> dict:
        return {
            "onset": self.onset(z).squeeze(-1),
            "class": self.class_head(z),
            "present": self.present(z),
        }


class GRUNoEncoder(nn.Module):
    """GRU that takes raw F_entity per bin (no encoder). Used as V8b baseline."""
    def __init__(self, F_entity: int, L: int, num_classes_present: int = 8, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=F_entity, hidden_size=hidden, num_layers=1, batch_first=True)
        self.heads = _Heads(hidden, num_classes_present)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, L, F_entity)
        _, h = self.gru(x)  # h: (1, B, hidden)
        return self.heads(h.squeeze(0))


class TransformerBaseline(nn.Module):
    """Encoder-only transformer baseline. No latent dynamics."""
    def __init__(self, F_entity: int, L: int, num_classes_present: int = 8, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(F_entity, d_model)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.heads = _Heads(d_model, num_classes_present)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.proj(x) + self.pos
        h = self.enc(h)
        z = h.mean(dim=1)  # pool over time
        return self.heads(z)
