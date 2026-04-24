from typing import List, Dict
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Fake Review Detector API")


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    predictions: List[str]
    scores: List[float] = None


# load default model
MODEL_PATH = os.getenv("MODEL_PATH", "models/tfidf_lr.joblib")
if not os.path.exists(MODEL_PATH):
    # try a common alternative
    MODEL_PATH = os.path.join("models", os.path.basename(MODEL_PATH))

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    texts = req.texts
    try:
        # if model supports predict_proba
        try:
            probs = model.predict_proba(texts)
            if probs.shape[1] > 1:
                scores = [float(p[1]) for p in probs]
            else:
                scores = [float(p[0]) for p in probs]
        except Exception:
            scores = None

        preds = model.predict(texts)
        preds_list = [str(p) for p in preds]
        return PredictResponse(predictions=preds_list, scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
