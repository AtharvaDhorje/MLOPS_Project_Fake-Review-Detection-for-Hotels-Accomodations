import argparse
import os
import pickle

import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_pickled(test_pkl: str, model_path: str, out_dir: str = "outputs", run_id: str = None):
    with open(test_pkl, "rb") as f:
        test = pickle.load(f)

    X_test = test["text"]
    y_test = test["label"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path)

    # save metrics JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    import json

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Confusion matrix saved to {out_path}")
    print(f"Metrics saved to {metrics_path}")
    # Log evaluation metrics/artifacts to MLflow if a run_id is provided or an experiment exists
    try:
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run()
        # log summary metrics
        # classification_report has 'accuracy' and 'macro avg' keys
        if isinstance(report, dict):
            if "accuracy" in report:
                mlflow.log_metric("accuracy", float(report.get("accuracy", 0)))
            macro = report.get("macro avg") or report.get("macro_avg")
            if macro and isinstance(macro, dict):
                if "f1-score" in macro:
                    mlflow.log_metric("f1_macro", float(macro.get("f1-score")))
                if "precision" in macro:
                    mlflow.log_metric("precision_macro", float(macro.get("precision")))
                if "recall" in macro:
                    mlflow.log_metric("recall_macro", float(macro.get("recall")))
        # log JSON and image artifacts
        mlflow.log_artifact(metrics_path, artifact_path="evaluation")
        mlflow.log_artifact(out_path, artifact_path="evaluation")
    except Exception:
        # don't fail evaluation if MLflow logging fails
        pass
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass

    return metrics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pkl", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()
    evaluate_pickled(args.test_pkl, args.model_path, args.out_dir)
