"""FastAPI app. The /predict handler has no path to the ground-truth store.

The static check in test_demo_no_leakage.py enforces this.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.deps import get_config, get_device, get_model, get_schema
from src.api.schemas import ForecastResponse, WindowPayload
from src.config import F_entity

app = FastAPI(title="SIH 26153 Latent-Dynamics Network Attack Forecasting")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe. Reports device and model version."""
    import torch  # lazy import: src/api/ is not in the torch import allow-list

    cfg = get_config()
    return {
        "status": "ok",
        "device": str(get_device()),
        "model_version": cfg.get("model", {}).get("version", "0.1.0"),
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: WindowPayload) -> ForecastResponse:
    """Run the model on a window of features and return onset/class/present probs."""
    import torch  # lazy import: src/api/ is not in the torch import allow-list

    schema = get_schema()
    cfg = get_config()
    L = int(cfg["data"]["sequence_length"])
    V_p, V_s, V_t = 17, 12, 5
    F_ent = F_entity(schema, V_p, V_s, V_t)

    if len(payload.features) != L:
        raise HTTPException(
            status_code=422,
            detail="features must have L=" + str(L) + " rows",
        )
    for row in payload.features:
        if len(row) != F_ent:
            raise HTTPException(
                status_code=422,
                detail="each feature row must have F_entity=" + str(F_ent) + " cols",
            )

    x = torch.tensor([payload.features], dtype=torch.float32, device=get_device())
    model = get_model().to(get_device())
    with torch.no_grad():
        out = model(x)
    p_onset = {str(k): float(torch.sigmoid(out["onset_logits"][k]).squeeze().item()) for k in [1, 3, 5]}
    p_class = {str(k): torch.sigmoid(out["class_logits"][k]).squeeze().cpu().tolist() for k in [1, 3, 5]}
    p_present = torch.softmax(out["present_logits"], dim=-1).squeeze().cpu().tolist()
    return ForecastResponse(
        p_onset=p_onset,
        p_class=p_class,
        p_attack_present=p_present,
        model_version=cfg.get("model", {}).get("version", "0.1.0"),
        device=str(get_device()),
    )
