import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app

from pidgin_senti.predictor import BasePredictor, ServingSettings, load_predictor
from pidgin_senti.preprocessing import normalize_pidgin_social_text
from pidgin_senti.schemas import PredictionRequest, PredictionResponse

PREDICTIONS_TOTAL = Counter(
    "pidgin_predictions_total",
    "Total sentiment predictions served",
    ["label", "needs_human_review", "model_version"],
)
INFERENCE_LATENCY_SECONDS = Histogram(
    "pidgin_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
PREDICTION_CONFIDENCE = Histogram(
    "pidgin_prediction_confidence",
    "Model confidence distribution for drift monitoring",
    buckets=(0.35, 0.45, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0),
)
SUBWORD_FRAGMENTATION = Histogram(
    "pidgin_subword_per_word_ratio",
    "Subword-to-word ratio (detects emerging Pidgin slang OOV drift)",
    buckets=(1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = ServingSettings()
    app.state.settings = settings
    app.state.predictor = load_predictor(settings)
    yield


app = FastAPI(
    title="Pidgin Sentiment Analysis Service",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    predictor: BasePredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": predictor.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor: BasePredictor = request.app.state.predictor
    settings: ServingSettings = request.app.state.settings

    t0 = time.perf_counter()
    raw = predictor.predict_one(payload.text)
    elapsed_s = time.perf_counter() - t0
    latency_ms = elapsed_s * 1000.0

    needs_review = raw.confidence < settings.confidence_threshold

    INFERENCE_LATENCY_SECONDS.observe(elapsed_s)
    PREDICTION_CONFIDENCE.observe(raw.confidence)
    PREDICTIONS_TOTAL.labels(
        label=raw.label.value,
        needs_human_review=str(needs_review).lower(),
        model_version=predictor.version,
    ).inc()
    if raw.subword_per_word_ratio is not None:
        SUBWORD_FRAGMENTATION.observe(raw.subword_per_word_ratio)

    log_path = settings.prediction_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "clean_text": normalize_pidgin_social_text(payload.text),
        "label": raw.label.value,
        "confidence": round(raw.confidence, 6),
        "probabilities": {k.value: v for k, v in raw.probabilities.items()},
        "needs_human_review": needs_review,
        "subword_per_word_ratio": raw.subword_per_word_ratio,
        "latency_ms": round(latency_ms, 3),
        "model_version": predictor.version,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return PredictionResponse(
        label=raw.label,
        confidence=round(raw.confidence, 6),
        probabilities=raw.probabilities,
        needs_human_review=needs_review,
        latency_ms=round(latency_ms, 3),
        model_version=predictor.version,
    )
