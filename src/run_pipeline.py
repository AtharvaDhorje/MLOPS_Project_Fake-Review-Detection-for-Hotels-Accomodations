import argparse
import os
import yaml

from src.preprocess import preprocess, save_splits
from src.models import train_and_log


def run_pipeline(data_path: str, params_file: str = "params.yaml", model_out: str = "models/tfidf_lr.joblib"):
    with open(params_file) as f:
        params = yaml.safe_load(f).get("train", {})

    # Preprocess and create splits
    import pandas as pd

    df = pd.read_csv(data_path)
    df2 = preprocess(df)
    save_splits(df2, out_dir="data/processed", test_size=params.get("test_size", 0.2), random_state=params.get("random_state", 42))

    train_pkl = os.path.join("data/processed", "train.pkl")
    model_path = train_and_log(train_pkl, model_out, params)

    print("Pipeline completed. Model at:", model_path)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="deceptive-opinion.csv")
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--model-out", type=str, default="models/tfidf_lr.joblib")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.params, args.model_out)
