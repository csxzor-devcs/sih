"""Canonical training entry point per spec section 14.2 (Task 23).

This is the ONLY script that the RTX 4060 training machine runs. The
three-machine split says: dev box writes this, training box only runs it.

On the dev box TODAY the script runs a synthetic-data smoke (mirroring
``scripts/smoke_train.py``) to verify the end-to-end forward + loss +
optimizer step. The synthetic data is NOT CIC-IDS-2017 — it is generated
with ``torch.manual_seed(args.seed)`` against the schema-derived
``F_entity``. On the training machine a populated ``WindowDataset`` will
replace the synthetic block; see the ``Real-Dataset-Wiring`` marker in
``main()``.

Usage (real run, on the training machine):

    python scripts/train.py \\
        --config configs/default.yaml \\
        --seed 0 \\
        --out artifacts/checkpoints/seed_0

Usage (dev-box smoke):

    python scripts/train.py --epochs 1 --out /tmp/smoke_train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import F_entity, load_config, load_schema
from src.model.latent_dynamics_model import LatentDynamicsModel
from src.model.losses import class_loss, onset_loss, present_loss, transition_loss
from src.train.seed import seed_everything


def _resolve_device(arg: str) -> str:
    """Resolve ``--device auto`` to ``cuda`` or ``cpu`` based on availability."""
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/checkpoints/seed_0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = _resolve_device(args.device)
    print(f"[train] device={device} seed={args.seed} epochs={args.epochs} out={args.out}")

    # Real-Dataset-Wiring: when WindowDataset.__getitem__ lands, replace
    # the synthetic block below with:
    #   ds = WindowDataset(agg, cfg, scaler, schema, vocab)
    #   train_loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]),
    #                             shuffle=True, num_workers=int(cfg["hardware"]["num_workers"]))
    # For now, the dev-box smoke uses a synthetic batch to verify the
    # end-to-end forward + loss + step path. Real training on the
    # RTX 4060 box runs the same script with a populated dataset.

    # ---- synthetic-batch block (dev-box smoke) -------------------------
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    F_ent = F_entity(schema, V_p, V_s, V_t)
    B, L = 8, int(cfg["data"]["sequence_length"])
    torch.manual_seed(args.seed)
    x_t = torch.randn(B, L, F_ent)
    x_future = {k: torch.randn(B, L, F_ent) for k in [1, 3, 5]}
    y_onset = {k: torch.randint(0, 2, (B,)) for k in [1, 3, 5]}
    y_class = {k: torch.randint(0, 2, (B, 7)) for k in [1, 3, 5]}
    y_present = torch.randint(0, 8, (B,))
    # -------------------------------------------------------------------

    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    lambda_transition = float(cfg["training"]["L_transition_weight"])

    last_losses: dict[str, float] = {}
    for epoch in range(args.epochs):
        model.train()
        # Ruling S (mirrored from smoke_train.py): rollout_steps=1 so the
        # transition MLP is actually applied to z0. With rollout_steps=0
        # z_rolled[k] = z0 for all k and L_transition reduces to a
        # trivial ||z0 - z_actual|| that does NOT exercise the
        # transition MLP that the spec requires.
        x_t_d = x_t.to(device)
        x_future_d = {k: v.to(device) for k, v in x_future.items()}
        y_onset_d = {k: v.to(device) for k, v in y_onset.items()}
        y_class_d = {k: v.to(device) for k, v in y_class.items()}
        y_present_d = y_present.to(device)

        out = model(x_t_d, rollout_steps=1)
        l_onset = sum(onset_loss(out["onset_logits"][k], y_onset_d[k]) for k in [1, 3, 5])
        l_class = sum(class_loss(out["class_logits"][k], y_class_d[k]) for k in [1, 3, 5])
        l_present = present_loss(out["present_logits"], y_present_d)
        l_transition = 0.0
        for k in [1, 3, 5]:
            e_f = model.per_bin_encoder(x_future_d[k]).detach()
            z_actual = model.window_encoder(e_f)
            l_transition = l_transition + transition_loss(out["z_rolled"][k], z_actual)
        l_transition = l_transition / 3.0
        l_total = l_onset + l_class + l_present + lambda_transition * l_transition

        opt.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        last_losses = {
            "L_onset": float(l_onset.item()),
            "L_class": float(l_class.item()),
            "L_present": float(l_present.item()),
            "L_transition": float(l_transition.item()),
            "L_total": float(l_total.item()),
        }
        print(
            f"[train] epoch={epoch} L_total={last_losses['L_total']:.4f} "
            f"L_onset={last_losses['L_onset']:.4f} L_class={last_losses['L_class']:.4f} "
            f"L_present={last_losses['L_present']:.4f} L_transition={last_losses['L_transition']:.4f}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": cfg, "seed": args.seed},
        out_dir / "model.pt",
    )
    report = {
        "epoch": int(args.epochs - 1),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "device": device,
        "F_entity": int(F_ent),
        "B": int(B),
        "L": int(L),
        "losses": last_losses,
        "synthetic": True,
    }
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train] saved to {out_dir}")


if __name__ == "__main__":
    main()
