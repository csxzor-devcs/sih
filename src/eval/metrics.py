"""AUROC, AUPRC, Brier, ECE, and paired bootstrap CIs.

All metrics operate on 1-D numpy arrays. The paired bootstrap CI is for
the *delta* per evaluation example (e.g., rollout_err - no_trans_err for
V8a, or the per-example rank-statistic for V8b).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    multi_class: str | None = None,
) -> float:
    """Compute the area under the ROC curve.

    Args:
        y_true: Binary ground-truth labels (0/1) for the 1-D path; integer
            class labels (0..K-1) for the multi-class path.
        y_score: Predicted scores or probabilities for the positive class
            (1-D) when multi_class is None; 2-D probability-like matrix
            of shape (N, K) when multi_class is provided.
        multi_class: If None (default), the 1-D binary behaviour is used.
            If "ovr" or "ovo", forwards to sklearn's roc_auc_score for the
            multi-class case.

    Returns:
        AUROC as a float in [0, 1].
    """
    if multi_class is None:
        return float(roc_auc_score(y_true, y_score))
    return float(roc_auc_score(y_true, y_score, multi_class=multi_class))


def auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    multi_class: str | None = None,
) -> float:
    """Compute the area under the precision-recall curve (average precision).

    Args:
        y_true: Binary ground-truth labels (0/1) for the 1-D path; integer
            class labels (0..K-1) for the multi-class path.
        y_score: Predicted scores or probabilities for the positive class
            (1-D) when multi_class is None; 2-D probability-like matrix
            of shape (N, K) when multi_class is provided.
        multi_class: If None (default), the 1-D binary behaviour is used.
            If "ovr", the macro-averaged multi-class average precision is
            computed via sklearn's average_precision_score (sklearn >= 1.3).

    Returns:
        AUPRC as a float in [0, 1].
    """
    if multi_class is None:
        return float(average_precision_score(y_true, y_score))
    # multi_class="ovr" => macro-averaged multi-class average precision.
    if multi_class == "ovr":
        return float(average_precision_score(y_true, y_score, average="macro"))
    # Fallback: forward to sklearn and let it decide.
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score (mean squared error of probabilities).

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Brier score as a non-negative float (0 is perfect).
    """
    return float(np.mean((y_prob - y_true) ** 2))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equal-width probability bins in [0, 1].

    Returns:
        ECE as a non-negative float (0 is perfectly calibrated).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece_val += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece_val)


def paired_bootstrap_ci(
    delta_per_example: np.ndarray,
    n_samples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Paired bootstrap CI on the per-example delta.

    Returns (lo, hi) such that P(lo < mean(delta) < hi) is approximately 1 - alpha.
    The bootstrap is paired because it resamples the *same* indices across
    examples, preserving any per-example correlation structure.

    Args:
        delta_per_example: 1-D array of per-example deltas.
        n_samples: Number of bootstrap resamples to draw.
        alpha: Miscoverage rate; the returned CI has nominal coverage 1 - alpha.
        rng: Optional numpy random Generator for reproducibility. If None,
            a default-seeded Generator is used.

    Returns:
        Tuple (lo, hi) of CI bounds.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(delta_per_example)
    means = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        means[i] = delta_per_example[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi
