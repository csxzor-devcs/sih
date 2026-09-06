"""Multi-seed training evaluation per spec §23.1.

Distinguishes:
- Training variance: 5 seeds × full training run (5 distinct models)
- Evaluation variance: 1,000 paired bootstrap resamples per model

We report mean ± std across the 5 training seeds, and per-baseline
paired bootstrap CIs against the primary system.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    """Aggregate per-seed metric dicts into mean/std/n.

    Returns nested dict: {metric_name: {"mean": float, "std": float, "n": int}}.
    """
    if not per_seed_results:
        return {}
    keys = per_seed_results[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_seed_results], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "values": [float(v) for v in vals],
        }
    return out
