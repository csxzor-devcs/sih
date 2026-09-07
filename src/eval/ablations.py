"""Ablation registry per spec §22.

Each entry is a config override applied to a fresh training/eval run.
Per Issue 6: V8a and V8b are now separate, non-overlapping experiments.
- V8a: transition validity — does z_{t+K} ≈ encode(x_{t+K})?
- V8b: downstream utility — does the latent help forecasting vs. no-rollout baseline?
"""
from __future__ import annotations

ABALATIONS = []  # populated below


def _entry(name, kind, overrides, hypothesis):
    return {"name": name, "kind": kind, "overrides": overrides, "hypothesis": hypothesis}


ABALATIONS.append(_entry(
    "V1_per_bin_encoder",
    "training",
    {"latent_transition_mlp": {"layers": [64, 64]}},
    "Ablate the deeper 64→128→64 MLP. Expectation: AUROC drops ≥2 points on present_auroc.",
))

ABALATIONS.append(_entry(
    "V4_window_size",
    "training",
    {"window_size_seconds": 360},
    "Halve the history window from 12 to 6 bins. Expectation: onset_auroc drops ≥3 points.",
))

ABALATIONS.append(_entry(
    "V6_no_histograms",
    "training",
    {"use_histograms": False},
    "Drop the 3 histograms per direction (proto/service/state). Expectation: class_auroc drops ≥2 points.",
))

ABALATIONS.append(_entry(
    "V7_horizons",
    "training",
    {"horizons_minutes": [1]},
    "Single horizon (1 min) instead of {1,3,5}. Expectation: 3/5-min onset_auprc drops sharply.",
))

ABALATIONS.append(_entry(
    "V8a_transition_validity",
    "training",
    {"latent_transition_mlp": {"layers": [64, 64]}, "L_transition_weight": 0.0},
    "Disable the transition supervision. Per §22: V8a checks whether the transition function is doing what it claims. Metric: ||z_{t+K} - encode(x_{t+K})||^2 in latent space.",
))

ABALATIONS.append(_entry(
    "V8b_downstream_utility",
    "evaluation",
    {"rollout_steps": 0},
    "Rollout 0 steps (no transition used at inference) vs. full rollout. Per §22: V8b checks whether the latent dynamics earns its keep on downstream forecasting.",
))

ABALATIONS.append(_entry(
    "V10_lambda_sweep",
    "training",
    {"lambda_values": [0.0, 0.05, 0.1, 0.5, 1.0]},
    "Sweep the L_transition_weight to characterize the loss balance. Per audit: the chosen value λ=0.1 must be inside the sweep and supported by a short rationale.",
))

ABALATIONS.append(_entry(
    "V11_packet_features_off",
    "training",
    {"use_packet_features": False},
    "Drop the 5 packet-derived features per direction. Per audit: separate ablation from V6 (histograms) so the contribution of packet features is independently measured.",
))


ABLATION_REGISTRY = {a["name"]: a for a in ABALATIONS}  # overwrite
