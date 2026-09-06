"""Classical ML baselines (LogReg, RandomForest, XGBoost).

These are flattened-window classifiers. They DO NOT model temporal
dynamics — that's the point. If our primary system doesn't beat them,
the latent dynamics is not earning its complexity.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .baselines import BaselineOutput


def _to_baseline_output(preds_proba: np.ndarray, name: str) -> BaselineOutput:
    N, C = preds_proba.shape
    onset = np.zeros(N, dtype=np.float32)
    cls = np.zeros((N, 7), dtype=np.float32)
    return BaselineOutput(onset, cls, preds_proba.astype(np.float32), name=name)


def baseline_logreg(X: np.ndarray, y_present: np.ndarray, num_classes_present: int) -> BaselineOutput:
    """Logistic regression on flattened window features."""
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y_present)
    proba = clf.predict_proba(X)
    # Pad to num_classes_present if some classes are missing
    if proba.shape[1] < num_classes_present:
        pad = np.zeros((proba.shape[0], num_classes_present - proba.shape[1]))
        proba = np.concatenate([proba, pad], axis=1)
    return _to_baseline_output(proba, "logreg")


def baseline_random_forest(X: np.ndarray, y_present: np.ndarray, num_classes_present: int) -> BaselineOutput:
    """Random forest on flattened window features."""
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X, y_present)
    proba = clf.predict_proba(X)
    if proba.shape[1] < num_classes_present:
        pad = np.zeros((proba.shape[0], num_classes_present - proba.shape[1]))
        proba = np.concatenate([proba, pad], axis=1)
    return _to_baseline_output(proba, "rf")


def baseline_xgboost(X: np.ndarray, y_present: np.ndarray, num_classes_present: int) -> BaselineOutput:
    """XGBoost on flattened window features."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=num_classes_present,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X, y_present)
    proba = clf.predict_proba(X)
    return _to_baseline_output(proba, "xgboost")
