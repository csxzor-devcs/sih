from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_metrics_endpoint_returns_prometheus_format():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    # Prometheus format
    assert "requests_total" in body
    assert "request_latency_seconds" in body
