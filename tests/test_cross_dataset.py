"""V3 cross-dataset evaluation smoke test (tests/test_cross_dataset.py).

The brief specified a tiny smoke test that runs ``src.eval.runner.evaluate``
against a hand-rolled ``DataLoader`` and asserts the returned metrics dict
contains the ``onset_auroc`` key. The verbatim test in the brief contains
two interface mismatches with the real codebase and was adjusted to make
it runnable; see the task-30 report "Concerns" for the deviation list.
"""
from __future__ import annotations

import numpy as np
from src.eval.runner import evaluate


def test_cross_dataset_eval_produces_metrics():
    # Smoke test with a tiny model
    import torch
    from src.config import F_entity, load_config, load_schema
    from src.model.latent_dynamics_model import LatentDynamicsModel
    from torch.utils.data import DataLoader, TensorDataset

    cfg = load_config()
    schema = load_schema()
    vocabs = schema["vocabularies"]
    V_p = int(vocabs["protocols"]["max_size"])
    V_s = int(vocabs["services"]["max_size"])
    V_t = int(vocabs["tcp_states"]["max_size"])
    F_ent = F_entity(schema, V_p, V_s, V_t)
    L = int(cfg["data"]["sequence_length"])

    class _DS:
        def __init__(self): self.F_entity = F_ent; self.L = L; self._n = 8
        def __len__(self): return self._n
        def __getitem__(self, i):
            return {
                "x": torch.randn(self.L, self.F_entity),
                "y_onset": {1: torch.tensor(0.0), 3: torch.tensor(0.0), 5: torch.tensor(0.0)},
                "y_class": {1: torch.zeros(7), 3: torch.zeros(7), 5: torch.zeros(7)},
                "y_present": torch.tensor(i % 8).long(),
            }
    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    loader = DataLoader(_DS(), batch_size=4)
    metrics = evaluate(model, loader)
    assert "onset_auroc" in metrics
