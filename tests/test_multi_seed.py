"""aggregate_seeds computes mean/std/n across per-seed metric dicts."""
from __future__ import annotations

import numpy as np

from src.eval.multi_seed import aggregate_seeds


def test_aggregate_seeds_computes_mean_std():
    # Three "seeds" with AUROC values
    results = [{"onset_auroc": 0.7, "present_auroc_macro": 0.6},
               {"onset_auroc": 0.72, "present_auroc_macro": 0.62},
               {"onset_auroc": 0.71, "present_auroc_macro": 0.61}]
    agg = aggregate_seeds(results)
    assert abs(agg["onset_auroc"]["mean"] - 0.71) < 1e-6
    assert agg["onset_auroc"]["std"] > 0
    assert agg["onset_auroc"]["n"] == 3
