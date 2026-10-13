"""FastAPI service for the trained fraud-detection pipeline."""
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(__import__("os").environ.get("MODEL_PATH", ROOT / "models" / "fraud_pipeline.joblib"))

app = FastAPI(title="Online Fraud Detection API", version="1.0.0")
model = None
feature_names = None


@app.on_event("startup")
def load_model():
    global model, feature_names
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        # sklearn Pipeline trained with a DataFrame retains feature_names_in_.
        feature_names = list(getattr(model, "feature_names_in_", [])) or None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Flexible fields allow this example API to work with arbitrary dataset schemas.
    # At runtime we enforce the trained feature schema below.
    transaction: dict[str, Any] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=422, detail="Provide a JSON object containing model features.")
    if "transaction" in payload and isinstance(payload["transaction"], dict):
        payload = payload["transaction"]
    expected = list(feature_names or [])
    if expected:
        missing = [c for c in expected if c not in payload]
        extra = [c for c in payload if c not in expected]
        if missing:
            raise HTTPException(status_code=422, detail={"missing_fields": missing})
        if extra:
            raise HTTPException(status_code=422, detail={"unexpected_fields": extra})
        row = {c: payload[c] for c in expected}
    else:
        row = payload
    try:
        X = pd.DataFrame([row])
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not score input: {exc}") from exc
    return {"prediction": prediction, "label": "fraud" if prediction == 1 else "normal", "fraud_probability": probability}
