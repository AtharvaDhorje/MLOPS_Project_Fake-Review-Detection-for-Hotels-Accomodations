import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import mlflow

from src.monitor import population_stability_index, ks_test_series


def _score_distribution(model, texts):
    # use predict_proba if available
    try:
        probs = model.predict_proba(texts)
        # take probability of positive class (index 1)
        if probs.shape[1] > 1:
            scores = probs[:, 1]
        else:
            scores = probs[:, 0]
    except Exception:
        # fallback to predicted labels
        scores = model.predict(texts)
    return np.array(scores)


def run_monitoring(train_pkl: str, new_pkl: str, model_path: str, out_dir: str = "outputs", run_id: str = None) -> str:
    """Compute PSI and KS for model score distributions and text-length between train and new data.

    Saves `monitoring.json` to out_dir and logs metrics to MLflow under a monitoring run.
    """
    with open(train_pkl, "rb") as f:
        train = pd.read_pickle(f)
    with open(new_pkl, "rb") as f:
        new = pd.read_pickle(f)

    model = joblib.load(model_path)

    train_texts = train["text"].astype(str)
    new_texts = new["text"].astype(str)

    train_scores = _score_distribution(model, train_texts)
    new_scores = _score_distribution(model, new_texts)

    # PSI on score distributions
    psi_score = population_stability_index(pd.Series(train_scores), pd.Series(new_scores), buckets=10)

    # KS on text length
    train_len = train_texts.str.len()
    new_len = new_texts.str.len()
    ks_stat, ks_p = ks_test_series(train_len, new_len) or (None, None)

    results = {
        "psi_score": float(psi_score),
        "ks_text_length_stat": float(ks_stat) if ks_stat is not None else None,
        "ks_text_length_p": float(ks_p) if ks_p is not None else None,
        "train_count": int(len(train_texts)),
        "new_count": int(len(new_texts)),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "monitoring.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # log monitoring metrics to MLflow (create a monitoring run)
    mlflow.set_experiment("fake-review-monitoring")
    with mlflow.start_run(nested=False) as mrun:
        mlflow.log_metrics(results)
        if run_id:
            mlflow.set_tag("train_run_id", run_id)
        mlflow.log_artifact(out_path)

    # simple alerting rule
    alert = False
    alert_reasons = []
    if psi_score > 0.2:
        alert = True
        alert_reasons.append(f"PSI score {psi_score:.3f} > 0.2")
    if ks_stat is not None and ks_p is not None and ks_p < 0.01:
        alert = True
        alert_reasons.append(f"KS p-value {ks_p:.4f} < 0.01")

    if alert:
        results["alert"] = True
        results["alert_reasons"] = alert_reasons
    else:
        results["alert"] = False

    # update file with alert info
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return out_path
