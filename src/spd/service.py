# src/spd/service.py
import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from spd.categories import CATEGORIES
from spd.inference import SPDPredictor

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("spd_requests_total", "Total SPD check requests")
LATENCY_HIST = Histogram("spd_latency_seconds", "SPD check latency")
DETECTED_COUNT = Counter("spd_detected_total", "Requests where SPD was detected")

MODEL_VERSION = "spd-hubert-v1.0.0"

_predictor: SPDPredictor | None = None
_thresholds: dict[str, float] = {}


def load_thresholds(path: str) -> dict[str, float]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    default = cfg.get("defaults", {}).get("threshold", 0.5)
    return {cat: cfg.get("per_category", {}).get(cat, default) for cat in CATEGORIES}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _thresholds
    model_path = os.environ.get("MODEL_PATH", "artifacts/model.onnx")
    tokenizer_name = os.environ.get("TOKENIZER_NAME", "SZTAKI-HLT/hubert-base-cc")
    threshold_path = os.environ.get("THRESHOLD_PATH", "configs/thresholds.yaml")
    _thresholds = load_thresholds(threshold_path)
    _predictor = SPDPredictor(model_path, tokenizer_name, _thresholds)
    yield


app = FastAPI(lifespan=lifespan)


class CheckRequest(BaseModel):
    text: str = Field(..., max_length=8192)
    lang: str = "hu"
    request_id: str | None = None


class CheckResponse(BaseModel):
    request_id: str
    model_version: str
    lang_detected: str
    scores: dict[str, float]
    categories_triggered: list[str]
    spd_detected: bool
    latency_ms: float


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.get("/v1/info")
def info() -> dict:
    return {"model_version": MODEL_VERSION, "thresholds": _thresholds}


@app.post("/v1/spd-check", response_model=CheckResponse)
def spd_check(req: CheckRequest) -> CheckResponse:
    REQUEST_COUNT.inc()
    request_id = req.request_id or str(uuid.uuid4())

    with LATENCY_HIST.time():
        result = _predictor.predict(req.text)

    if result["spd_detected"]:
        DETECTED_COUNT.inc()

    logger.info(
        str({
            "request_id": request_id,
            "model_version": MODEL_VERSION,
            "lang_detected": "hu",
            "scores": result["scores"],
            "categories_triggered": result["categories_triggered"],
            "spd_detected": result["spd_detected"],
            "latency_ms": result["latency_ms"],
            "text_sha256": hashlib.sha256(req.text.encode()).hexdigest(),
            "text_len": len(req.text),
        })
    )

    return CheckResponse(
        request_id=request_id,
        model_version=MODEL_VERSION,
        lang_detected="hu",
        **result,
    )
