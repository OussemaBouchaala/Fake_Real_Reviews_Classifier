"""HTTP inference API for the fake review classifier.

Run from the app/ folder:  uvicorn api:app --port 8000   (docs at http://localhost:8000/docs)

/health says the process is alive (Kubernetes livenessProbe);
/ready says the model is loaded and requests can be served (readinessProbe).
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import REVIEW_PIPELINE
from text_features import Classifier, Label, predict_label

# config.json paths are relative to the app/ folder
MODEL_PATH = Path(__file__).parent / REVIEW_PIPELINE

model: Classifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global model
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(title="Fake Review Detector API", lifespan=lifespan)


class Review(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, examples=["Great quality, arrived on time."])


class Prediction(BaseModel):
    label: Label


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/predict")
def predict(review: Review) -> Prediction:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not review.text.strip():
        raise HTTPException(status_code=422, detail="review text is empty")
    return Prediction(label=predict_label(model, review.text))
