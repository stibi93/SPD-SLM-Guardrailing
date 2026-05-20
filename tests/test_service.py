import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from spd.categories import CATEGORIES


MOCK_PREDICT_RESULT = {
    "scores": {cat: 0.01 for cat in CATEGORIES} | {"health": 0.92},
    "categories_triggered": ["health"],
    "spd_detected": True,
    "latency_ms": 18.4,
}


@pytest.fixture(scope="module")
def client():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = MOCK_PREDICT_RESULT

    with patch("spd.service.SPDPredictor") as MockCls, \
         patch("spd.service.load_thresholds", return_value={c: 0.5 for c in CATEGORIES}):
        MockCls.return_value = mock_predictor
        from spd.service import app
        with TestClient(app) as client:
            yield client


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_spd_check_200(client):
    r = client.post("/v1/spd-check", json={"text": "Cukorbeteg vagyok."})
    assert r.status_code == 200


def test_spd_check_response_keys(client):
    r = client.post("/v1/spd-check", json={"text": "teszt"})
    body = r.json()
    assert "request_id" in body
    assert "model_version" in body
    assert "scores" in body
    assert "spd_detected" in body
    assert "latency_ms" in body


def test_spd_check_scores_has_all_categories(client):
    r = client.post("/v1/spd-check", json={"text": "teszt"})
    assert set(r.json()["scores"].keys()) == set(CATEGORIES)


def test_spd_detected_true(client):
    r = client.post("/v1/spd-check", json={"text": "irrelevant — mock returns health=0.92"})
    assert r.json()["spd_detected"] is True
    assert "health" in r.json()["categories_triggered"]


def test_text_too_long_rejected(client):
    r = client.post("/v1/spd-check", json={"text": "x" * 9000})
    assert r.status_code == 422


def test_info_endpoint(client):
    r = client.get("/v1/info")
    assert r.status_code == 200
    assert "model_version" in r.json()
    assert "thresholds" in r.json()


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "spd_requests_total" in r.text
