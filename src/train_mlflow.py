import argparse
import os
import pickle

import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_mlflow(train_pkl: str, model_out: str, params: dict):
    with open(train_pkl, "rb") as f:
        train = pickle.load(f)

    X_train = train["text"]
    y_train = train["label"]

    ngram_range = (params.get("ngram_min", 1), params.get("ngram_max", 2))
    tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=params.get("max_features", 20000), stop_words="english")
    clf = LogisticRegression(max_iter=params.get("max_iter", 1000))
    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])

    mlflow.set_experiment("fake-review-hotel")
    with mlflow.start_run() as run:
        mlflow.log_params({"ngram_range": ngram_range, "max_features": params.get("max_features"), "max_iter": params.get("max_iter")})
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(train_pkl, artifact_path="train_data")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", dest="train_pkl", required=True)
    parser.add_argument("--model-out", type=str, default="models/tfidf_lr.joblib")
    parser.add_argument("--params", dest="params_file", type=str, default="params.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.params_file) as f:
        params = yaml.safe_load(f).get("train", {})

    train_mlflow(args.train_pkl, args.model_out, params)
