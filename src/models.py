import os
import pickle
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(params: Dict):
    ngram_range = (params.get("ngram_min", 1), params.get("ngram_max", 2))
    tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=params.get("max_features", 20000), stop_words="english")
    clf = LogisticRegression(max_iter=params.get("max_iter", 1000))
    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
    return pipeline


def train_and_log(train_pkl: str, model_out: str, params: Dict, experiment_name: str = "fake-review-hotel"):
    """Train the pipeline on a pickled train split and log to MLflow.

    Args:
        train_pkl: path to pickled train DataFrame with `text` and `label` columns.
        model_out: path to save the trained model (joblib).
        params: training parameters dict.
        experiment_name: MLflow experiment name.
    """
    with open(train_pkl, "rb") as f:
        train = pickle.load(f)

    X_train = train["text"]
    y_train = train["label"]

    pipeline = build_pipeline(params)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params({"ngram_range": (params.get("ngram_min"), params.get("ngram_max")), "max_features": params.get("max_features"), "max_iter": params.get("max_iter")})
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(train_pkl, artifact_path="train_data")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipeline, model_out)
    return model_out
