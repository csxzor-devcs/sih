"""Full model: per-bin encoder + GRU window encoder + transition + heads."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.config import F_entity
from src.model.encoder import PerBinEncoder
from src.model.gru import GRUWindowEncoder, LatentTransitionMLP
from src.model.heads import ClassHead, OnsetHead, PresentHead
from src.model.rollout import rollout


class _Heads(nn.Module):
    """Single submodule-grouping for the three heads.

    Exposes onset / class_head / present as direct attributes so callers
    (e.g. primary_with_rollout in Task 18) can do model.heads.onset(z).
    """

    def __init__(self, onset_cfg: dict[str, Any], class_cfg: dict[str, Any], present_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.onset = OnsetHead(tuple(onset_cfg["trunk"]), dropout=onset_cfg["dropout"])
        self.class_head = ClassHead(
            tuple(class_cfg["trunk"]), dropout=class_cfg["dropout"], n=class_cfg["output"]
        )
        self.present = PresentHead(tuple(present_cfg["trunk"]), n=present_cfg["output"])


class LatentDynamicsModel(nn.Module):
    """Composes per-bin encoder, GRU window encoder, deterministic MLP transition,
    and three heads (onset, class, present).

    forward(x, rollout_steps=0) contract:
      rollout_steps=0  -> use z0 directly (no transition rollout).
      rollout_steps=K  -> rollout(self.transition, z0, K) then run heads on z_K.
    Heads are single submodules (model.heads.onset, .class_head, .present),
    NOT per-k ModuleLists. The present head always reads from z0 (diagnostic).
    """

    def __init__(
        self,
        config: dict[str, Any],
        schema: dict[str, Any],
        V_p: int,
        V_s: int,
        V_t: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.F_ent = F_entity(schema, V_p, V_s, V_t)
        self.k_steps: list[int] = list(config["model"]["rollout"]["k_steps"])

        per_bin_cfg = config["model"]["per_bin_encoder"]
        self.per_bin_encoder = PerBinEncoder(
            F_in=self.F_ent,
            hidden=per_bin_cfg["layers"][1],
            out=per_bin_cfg["layers"][2],
            dropout=per_bin_cfg["dropout"],
        )

        gru_cfg = config["model"]["gru_window_encoder"]
        self.window_encoder = GRUWindowEncoder(
            in_size=gru_cfg["input_size"],
            hidden=gru_cfg["hidden_size"],
            num_layers=gru_cfg["num_layers"],
        )

        trans_cfg = config["model"]["latent_transition_mlp"]
        self.transition = LatentTransitionMLP(
            dim=trans_cfg["layers"][0],
            hidden=trans_cfg["layers"][1],
            dropout=trans_cfg["dropout"],
        )

        onset_cfg = config["model"]["heads"]["onset"]
        class_cfg = config["model"]["heads"]["class_conditional"]
        present_cfg = config["model"]["heads"]["attack_present"]
        self.heads = _Heads(onset_cfg, class_cfg, present_cfg)

    def forward(self, x: torch.Tensor, rollout_steps: int = 0) -> dict[str, Any]:
        """x: [B, L, F_entity] -> dict with z0, z_rolled, onset_logits,
        class_logits, present_logits (logits).

        rollout_steps=0: present-only; z_rolled/onset_logits/class_logits all
        evaluate the single heads on z0 (no transition applied).
        rollout_steps=K for K in self.k_steps: rollout(self.transition, z0, K)
        is used; outputs are indexed by the rollout horizon k.
        """
        e = self.per_bin_encoder(x)              # [B, L, 64]
        z0 = self.window_encoder(e)             # [B, 64]
        if rollout_steps == 0:
            z_rolled = {k: z0 for k in self.k_steps}
        else:
            z_rolled = {k: rollout(self.transition, z0, k) for k in self.k_steps}
        onset_logits = {k: self.heads.onset(z_rolled[k]) for k in self.k_steps}
        class_logits = {k: self.heads.class_head(z_rolled[k]) for k in self.k_steps}
        present_logits = self.heads.present(z0)  # [B, 8]
        return {
            "z0": z0,
            "z_rolled": z_rolled,
            "onset_logits": onset_logits,
            "class_logits": class_logits,
            "present_logits": present_logits,
        }
