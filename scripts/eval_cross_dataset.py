"""V3 cross-dataset evaluation: model trained on CIC-IDS-2017, evaluated on UNSW-NB15.

Reports the same metric set as Tier 2 evaluation. The drop (if any)
quantifies generalization to a different dataset.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.model.latent_dynamics_model import LatentDynamicsModel
from src.eval.runner import evaluate
from src.data.unsw_nb15 import load_unsw_nb15
from src.data.dataset import WindowDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unsw-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = bundle["config"]

    # Build UNSW-NB15 window dataset (reusing WindowDataset with mapped features)
    # This is the dataset-specific glue: convert UNSW-NB15 → 18-feature-per-direction
    # format, then build windows.
    unsw = load_unsw_nb15(args.unsw_csv)
    # NOTE: full feature alignment is non-trivial; for V3 we re-use the
    # column subset REQUIRED_COLS and pad to F_entity=104.
    # Implementation deferred; this script is the entry point.
    print(f"[v3] loaded {len(unsw)} UNSW-NB15 rows")
    print(f"[v3] WIP: full feature alignment in Task 30 step 4")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"note": "V3 cross-dataset; see spec §22 for full method"}, f, indent=2)


if __name__ == "__main__":
    main()
