import argparse
import os
import pickle

from src.models import train_and_log


def train_mlflow(train_pkl: str, model_out: str, params: dict):
    model_path = train_and_log(train_pkl, model_out, params)
    print(f"Model saved to {model_path}")


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
