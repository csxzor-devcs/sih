"""Thin wrapper: build the demo predictions parquet from the latest checkpoint.

Usage:
    python scripts/precompute.py \\
        --ckpt artifacts/checkpoints/seed_0/model.pt \\
        --out artifacts/demo/predictions.parquet

The real signature of ``build_predictions_parquet`` is
``build_predictions_parquet(model, agg, config, artifact_dir, device)``.
We pass ``artifact_dir=Path(args.out).parent`` so the function names the
file ``predictions.parquet`` inside that directory. The ``agg`` arg is a
``pd.DataFrame``; for the dev-box smoke path we pass an empty
``DataFrame`` (the function handles ``max(len(agg), 1) == 1`` and writes
1 row of zeros — fine for the smoke).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import load_config, load_schema
from src.demo.precompute import build_predictions_parquet
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
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"[precompute] device={device} ckpt={args.ckpt} out={args.out}")

    # Real-Dataset-Wiring: when WindowDataset is fully wired, replace
    # the empty-agg smoke block below with a real ``agg`` DataFrame
    # produced by the preprocessing pipeline:
    #   agg = pd.read_parquet(cfg["data"]["agg_path"])  # or similar
    #   artifact_dir = Path(args.out).parent
    #   out = build_predictions_parquet(
    #       model=model, agg=agg, config=cfg, artifact_dir=artifact_dir,
    #       device=device,
    #   )
    # For now, the dev-box smoke passes an empty DataFrame so the
    # function writes 1 row of zeros. This validates the import graph
    # + the artifact_dir wiring without requiring a populated dataset.

    # ---- smoke block (dev-box) ----------------------------------------
    cfg = load_config()
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    bundle = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    model.load_state_dict(bundle["model"])
    model.to(device).eval()

    agg = pd.DataFrame()
    artifact_dir = Path(args.out).parent
    out = build_predictions_parquet(
        model=model,
        agg=agg,
        config=cfg,
        artifact_dir=artifact_dir,
        device=device,
    )
    # -------------------------------------------------------------------

    print(f"[precompute] wrote {out}")


if __name__ == "__main__":
    main()
