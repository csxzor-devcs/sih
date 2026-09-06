"""Smoke training run: 1 epoch on a synthetic batch. Used to verify the
forward pass, the loss, and the optimizer step all work end-to-end on the
dev machine before the 5-seed full training on the training machine.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import F_entity, load_config, load_schema
from src.model.latent_dynamics_model import LatentDynamicsModel
from src.model.losses import class_loss, onset_loss, present_loss, transition_loss


def main() -> None:
    """Run one synthetic forward + loss + optimizer step and write the report."""
    cfg = load_config()
    schema = load_schema()
    # Synthesize a vocab of typical CIC-IDS-2017 sizes for the smoke run.
    V_p, V_s, V_t = 17, 12, 5
    F_ent = F_entity(schema, V_p, V_s, V_t)
    B, L = 8, 12

    torch.manual_seed(0)
    x_t = torch.randn(B, L, F_ent)
    x_future = {k: torch.randn(B, L, F_ent) for k in [1, 3, 5]}
    y_onset = {k: torch.randint(0, 2, (B,)) for k in [1, 3, 5]}
    y_class = {k: torch.randint(0, 2, (B, 7)) for k in [1, 3, 5]}
    y_present = torch.randint(0, 8, (B,))

    model = LatentDynamicsModel(cfg, schema, V_p, V_s, V_t)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Ruling S: rollout_steps=1 so the transition MLP is actually applied to z0.
    # With rollout_steps=0 (the model's default), z_rolled[k] = z0 for all k, and
    # L_transition reduces to ||z0 - z_actual||, which does NOT exercise the
    # transition MLP that the spec requires.
    out = model(x_t, rollout_steps=1)
    l_onset = sum(onset_loss(out["onset_logits"][k], y_onset[k]) for k in [1, 3, 5])
    l_class = sum(class_loss(out["class_logits"][k], y_class[k]) for k in [1, 3, 5])
    l_present = present_loss(out["present_logits"], y_present)
    l_transition = 0.0
    for k in [1, 3, 5]:
        e_f = model.per_bin_encoder(x_future[k]).detach()
        z_actual = model.window_encoder(e_f)
        l_transition = l_transition + transition_loss(out["z_rolled"][k], z_actual)
    l_transition = l_transition / 3.0

    l_total = l_onset + l_class + l_present + 0.1 * l_transition
    opt.zero_grad()
    l_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    report = {
        "L_onset": float(l_onset.item()),
        "L_class": float(l_class.item()),
        "L_present": float(l_present.item()),
        "L_transition": float(l_transition.item()),
        "L_total": float(l_total.item()),
        "F_entity": int(F_ent),
        "B": int(B),
        "L": int(L),
    }
    out_path = Path("artifacts/smoke_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
