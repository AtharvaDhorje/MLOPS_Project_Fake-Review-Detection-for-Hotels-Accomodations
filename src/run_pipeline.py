import argparse
import os
import sys
import yaml
import mlflow

# Ensure MLflow uses a SQL-backed store by default so the UI Overview tab works.
# If the environment hasn't set a tracking URI, fallback to a local SQLite DB.
if not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

# Ensure project root is on PYTHONPATH so `src` imports work when called by DVC
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import preprocess, save_splits
from src.models import train_and_log
from src.evaluate_pickled import evaluate_pickled


def run_pipeline(data_path: str, params_file: str = "params.yaml", model_out: str = "models/tfidf_lr.joblib"):
    with open(params_file) as f:
        params = yaml.safe_load(f).get("train", {})

    # Preprocess and create splits
    import pandas as pd

    df = pd.read_csv(data_path)
    df2 = preprocess(df)
    save_splits(df2, out_dir="data/processed", test_size=params.get("test_size", 0.2), random_state=params.get("random_state", 42))

    train_pkl = os.path.join("data/processed", "train.pkl")
    model_path, run_id = train_and_log(train_pkl, model_out, params)

    # Evaluate on the test split
    test_pkl = os.path.join("data/processed", "test.pkl")
    metrics_path = evaluate_pickled(test_pkl, model_path, out_dir="outputs", run_id=run_id)

    # Run monitoring comparing train vs test
    from src.monitor_service import run_monitoring

    monitor_path = run_monitoring(train_pkl, test_pkl, model_path, out_dir="outputs", run_id=run_id)

    # Archive run outputs into run-specific folder and update outputs/latest
    import shutil

    run_folder = os.path.join("outputs", f"runs", run_id)
    latest_folder = os.path.join("outputs", "latest")
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(latest_folder, exist_ok=True)

    # files to archive
    candidates = [metrics_path, monitor_path, os.path.join("outputs", "confusion_matrix.png")]
    # copy model too
    candidates.append(model_path)

    for p in candidates:
        if p and os.path.exists(p):
            shutil.copy(p, run_folder)
            shutil.copy(p, latest_folder)

    print("Pipeline completed. Model at:", model_path)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="deceptive-opinion.csv")
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--model-out", type=str, default="models/tfidf_lr.joblib")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.params, args.model_out)
