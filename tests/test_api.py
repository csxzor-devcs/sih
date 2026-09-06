"""/predict accepts a valid payload and rejects malformed ones with 422."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "device" in r.json()


def test_predict_accepts_valid_payload():
    # 12 bins x 104 features
    L = 12
    F = 104
    payload = {
        "host": "test",
        "direction": "BOTH",
        "features": [[0.0] * F for _ in range(L)],
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body["p_onset"].keys()) == {"1", "3", "5"}
    assert all(isinstance(v, float) for v in body["p_onset"].values())
    for k in ["1", "3", "5"]:
        assert len(body["p_class"][k]) == 7
    assert len(body["p_attack_present"]) == 8


def test_predict_rejects_wrong_length():
    r = client.post("/predict", json={"features": [[0.0] * 104 for _ in range(11)]})
    assert r.status_code == 422


def test_predict_rejects_wrong_width():
    r = client.post("/predict", json={"features": [[0.0] * 50 for _ in range(12)]})
    assert r.status_code == 422
