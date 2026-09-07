"""Full evaluation: primary system + baselines + paired bootstrap CI.

Usage (real run, on the dev box):
    python scripts/eval.py \\
        --ckpt artifacts/checkpoints/seed_0/model.pt \\
        --out artifacts/eval/seed_0.json

Dev-box smoke path: mirrors ``scripts/train.py`` and uses a synthetic
batch in place of a real ``WindowDataset`` / ``DataLoader``. The
``Real-Dataset-Wiring`` block marks where the real loader is plugged in
once ``WindowDataset.__getitem__`` lands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import F_entity, load_config, load_schema
from src.eval.runner import evaluate, paired_bootstrap_diff
from src.model.latent_dynamics_model import LatentDynamicsModel


def _resolve_device(arg: str) -> str:
    """Resolve ``--device auto`` to ``cuda`` or ``cpu`` based on availability."""
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for paired_bootstrap_diff")
    ap.add_argument("--seed", type=int, default=42, help="rng seed for the bootstrap")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"[eval] device={device} ckpt={args.ckpt} out={args.out}")

    # Real-Dataset-Wiring: when WindowDataset.__getitem__ lands, replace
    # the synthetic block below with:
    #   ds = WindowDataset(agg, cfg, scaler, schema, vocab)
    #   loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]),
    #                       shuffle=False, num_workers=int(cfg["hardware"]["num_workers"]))
    #   primary_metrics = evaluate(model, loader, device=device)
    # For now, the dev-box smoke uses a synthetic batch to verify the
    # evaluate() path is importable + runnable. Real evaluation on the
    # dev box replaces this synthetic block with the DataLoader call.

    # ---- synthetic-batch block (dev-box smoke) -------------------------
    cfg = load_config()
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    F_ent = F_entity(schema, V_p, V_s, V_t)
    L = int(cfg["data"]["sequence_length"])
    B = 64  # matches the brief's batch_size=64 for the real loader

    bundle = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    model.load_state_dict(bundle["model"])
    model.to(device).eval()

    # Synthesize a batch compatible with what a real DataLoader would
    # produce. The evaluate() helper iterates over the loader; the
    # synthetic "loader" yields one batch whose shape matches
    # collate(WindowDataset) per the spec.
    torch.manual_seed(0)
    x = torch.randn(B, L, F_ent, device=device)
    y_onset = {1: torch.randint(0, 2, (B,), device=device),
               3: torch.randint(0, 2, (B,), device=device),
               5: torch.randint(0, 2, (B,), device=device)}
    y_class = {1: torch.randint(0, 2, (B, 7), device=device),
               3: torch.randint(0, 2, (B, 7), device=device),
               5: torch.randint(0, 2, (B, 7), device=device)}
    y_present = torch.randint(0, 8, (B,), device=device)

    class _SyntheticLoader:
        """One-batch loader. Mirrors what the real DataLoader yields."""

        def __init__(self) -> None:
            self.batch = {"x": x, "y_onset": y_onset, "y_class": y_class, "y_present": y_present}

        def __iter__(self):
            return iter([self.batch])

        def __len__(self) -> int:
            return 1

    loader = _SyntheticLoader()
    primary_metrics = evaluate(model, loader, device=device)
    # -------------------------------------------------------------------

    # Baselines in the smoke path are recorded as synthetic placeholders
    # so the JSON output has the full key set. Real baselines on real
    # data come from src/eval/baselines*.py per the brief's spec §21
    # once the real DataLoader is wired in.
    baseline_metrics = {
        "mean": {"synthetic": True, "loss": 0.0},
        "persistence": {"synthetic": True, "loss": 0.0},
        "logreg": {"synthetic": True, "loss": 0.0},
        "random_forest": {"synthetic": True, "loss": 0.0},
        "xgboost": {"synthetic": True, "loss": 0.0},
        "gru_no_encoder": {"synthetic": True, "loss": 0.0},
        "transformer": {"synthetic": True, "loss": 0.0},
        "primary_rollout": {"synthetic": True, "loss": 0.0},
    }

    # Paired bootstrap CI: on a 1-batch synthetic run, primary and
    # baseline scores have length B with degenerate variance. We still
    # compute the CI for shape-completeness; a no-warning numpy run
    # confirms the call signature matches the spec.
    rng = np.random.default_rng(args.seed)
    primary_scores = rng.uniform(0, 1, B)
    baseline_scores = rng.uniform(0, 1, B)
    ci_lo, ci_hi = paired_bootstrap_diff(
        primary_scores, baseline_scores, n_boot=args.n_boot, seed=args.seed
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary": primary_metrics,
        "baselines": baseline_metrics,
        "paired_bootstrap_diff": {"ci_low": ci_lo, "ci_high": ci_hi, "n_boot": args.n_boot, "seed": args.seed},
        "synthetic": True,
        "F_entity": int(F_ent),
        "L": int(L),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
