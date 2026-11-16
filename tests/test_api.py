import time

from fastapi.testclient import TestClient

from pidgin_senti.api import app
from pidgin_senti.predictor import BasePredictor, RawPrediction, ServingSettings
from pidgin_senti.schemas import SentimentLabel


class StubPredictor(BasePredictor):
    version = "stub-v1"

    def predict_one(self, text: str) -> RawPrediction:
        return RawPrediction(
            label=SentimentLabel.POSITIVE,
            confidence=0.88,
            probabilities={
                SentimentLabel.POSITIVE: 0.88,
                SentimentLabel.NEUTRAL: 0.04,
                SentimentLabel.NEGATIVE: 0.08,
            },
        )


def test_predict_and_metrics_contract() -> None:
    app.state.settings = ServingSettings()
    app.state.predictor = StubPredictor()

    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["model_version"] == "stub-v1"

    t0 = time.perf_counter()
    resp = client.post("/predict", json={"text": "No wahala, dis app sweet well well 🔥"})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert body["needs_human_review"] is False
    assert elapsed_ms < 50.0  # API serialization + contract latency gate

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "pidgin_predictions_total" in metrics_resp.text


def test_empty_text_rejected() -> None:
    app.state.settings = ServingSettings()
    app.state.predictor = StubPredictor()
    client = TestClient(app)

    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422
