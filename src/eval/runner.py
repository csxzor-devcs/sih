"""Evaluation runner.

Per the project spec (paired bootstrap on identical
example, baseline_pred, primary_pred triples so primary and baseline are
scored on the same windows), this module exposes two functions:

- ``paired_bootstrap_diff``: a 95% bootstrap CI for (primary - baseline)
  on matched per-example score arrays.
- ``evaluate``: runs the model over a DataLoader and returns a
  JSON-serializable dict of per-head metrics.

The ``evaluate`` function reads the model's configured rollout horizons
via ``model.k_steps`` and reports metrics at the first horizon
(``model.k_steps[0]``). The present head is a diagnostic that always
reads from z0, so it is reported independently of the rollout horizon.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .metrics import auroc, auprc, brier, ece


def paired_bootstrap_diff(
    primary_scores: np.ndarray,
    baseline_scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap CI for (primary - baseline) on matched windows.

    Returns (ci_low, ci_high) at the 95% level. If both endpoints are
    positive, primary is significantly better than baseline; if both are
    negative, primary is significantly worse.

    Args:
        primary_scores: 1-D array of per-example primary scores (length N).
        baseline_scores: 1-D array of per-example baseline scores (length N),
            aligned element-wise with primary_scores.
        n_boot: Number of bootstrap resamples (default 1000 per spec).
        seed: Seed for the numpy random Generator (default 42).

    Returns:
        Tuple (lo, hi) of CI bounds as Python floats.
    """
    rng = np.random.default_rng(seed)
    n = len(primary_scores)
    diffs = primary_scores - baseline_scores
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = diffs[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate(
    model: Any,
    loader: Any,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run model on loader, return per-head metrics.

    The model is run with ``rollout_steps=0`` (i.e. z_rolled[k] = z0 for
    every configured horizon k), so all heads evaluate from the encoded
    window latent. Onset and class metrics are reported at the model's
    first configured horizon (``model.k_steps[0]``); present metrics are
    reported for the z0-diagnostic head. All metrics are computed in
    ``torch.no_grad()`` mode.

    Args:
        model: A LatentDynamicsModel (or any object exposing forward(x,
            rollout_steps=...) returning a dict with onset_logits,
            class_logits, present_logits).
        loader: A DataLoader yielding dicts with keys "x", "y_onset",
            "y_present", "y_class".
        device: Device to move input batches to (default "cpu").

    Returns:
        JSON-serializable dict with onset / present / class metrics and
        the number of evaluated examples.
    """
    # Lazy torch import: src/eval/ is not in the torch allow-list.
    import torch

    model.eval()
    k = model.k_steps[0]

    onset_p: list[np.ndarray] = []
    onset_y: list[np.ndarray] = []
    present_p: list[np.ndarray] = []
    present_y: list[np.ndarray] = []
    class_p_list: list[np.ndarray] = []
    class_y_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x, rollout_steps=0)
            onset_p.append(torch.sigmoid(out["onset_logits"][k]).squeeze(-1).cpu().numpy())
            onset_y.append(batch["y_onset"][k].numpy())
            present_p.append(torch.softmax(out["present_logits"], dim=-1).cpu().numpy())
            present_y.append(batch["y_present"].numpy())
            class_p_list.append(torch.sigmoid(out["class_logits"][k]).cpu().numpy())
            class_y_list.append(batch["y_class"][k].numpy())

    onset_p_arr = np.concatenate(onset_p)
    onset_y_arr = np.concatenate(onset_y)
    present_p_arr = np.concatenate(present_p)
    present_y_arr = np.concatenate(present_y)
    class_p_arr = np.concatenate(class_p_list)
    class_y_arr = np.concatenate(class_y_list)

    # Per-label macro for the multi-label class head. class_y_arr is a
    # [N, 7] 0/1 indicator matrix (multi-LABEL binary), not a [N] integer
    # class label vector, so sklearn's multi_class="ovr" path is invalid
    # here. We score each of the 7 columns as a 1-D binary problem and
    # macro-average. Labels whose column is constant (all 0 or all 1) are
    # skipped because roc_auc_score raises on degenerate targets.
    class_aurocs: list[float] = []
    class_auprcs: list[float] = []
    for label_idx in range(class_y_arr.shape[1]):
        y_col = class_y_arr[:, label_idx]
        p_col = class_p_arr[:, label_idx]
        if y_col.min() == y_col.max():
            continue
        class_aurocs.append(auroc(y_col, p_col))
        class_auprcs.append(auprc(y_col, p_col))
    class_auroc_macro = float(np.mean(class_aurocs)) if class_aurocs else float("nan")
    class_auprc_macro = float(np.mean(class_auprcs)) if class_auprcs else float("nan")

    return {
        "onset_auroc": auroc(onset_y_arr, onset_p_arr),
        "onset_auprc": auprc(onset_y_arr, onset_p_arr),
        "onset_brier": brier(onset_y_arr, onset_p_arr),
        "onset_ece": ece(onset_y_arr, onset_p_arr),
        "present_auroc_macro": auroc(present_y_arr, present_p_arr, multi_class="ovr"),
        "present_auprc_macro": auprc(present_y_arr, present_p_arr, multi_class="ovr"),
        "class_auroc_macro": class_auroc_macro,
        "class_auprc_macro": class_auprc_macro,
        "n": int(len(onset_y_arr)),
    }


def main() -> None:
    """Smoke entry point: print a JSON dump of an empty report schema."""
    print(json.dumps({"status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
