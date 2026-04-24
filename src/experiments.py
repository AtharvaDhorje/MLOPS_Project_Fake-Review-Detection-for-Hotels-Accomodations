import os
import json
import yaml
import mlflow
from pprint import pprint

from src.models import train_and_log
from src.evaluate_pickled import evaluate_pickled


# Ensure MLflow uses a SQL-backed store by default so experiment overview works
if not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f).get("train", {})


def run_experiments():
    base = load_params()
    # ensure splits exist
    train_pkl = os.path.join("data", "processed", "train.pkl")
    test_pkl = os.path.join("data", "processed", "test.pkl")
    if not os.path.exists(train_pkl) or not os.path.exists(test_pkl):
        raise RuntimeError("Train/test splits not found. Run run_pipeline first.")

    experiments = []
    grid = []
    # small grid for quick comparison
    for ngram_max in [1, 2]:
        for max_features in [5000, 20000]:
            for C in [0.1, 1.0]:
                grid.append({"ngram_min": 1, "ngram_max": ngram_max, "max_features": max_features, "C": C, "max_iter": base.get("max_iter", 1000)})

    results = []
    os.makedirs("outputs/experiments", exist_ok=True)
    for i, params in enumerate(grid, 1):
        name = f"exp_{i}_ng{params['ngram_max']}_mf{params['max_features']}_C{params['C']}"
        model_out = os.path.join("models", f"tfidf_lr_{name}.joblib")
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        print("Running", name)
        model_path, run_id = train_and_log(train_pkl, model_out, params, experiment_name="fake-review-experiments")
        metrics_path = evaluate_pickled(test_pkl, model_path, out_dir=os.path.join("outputs", "experiments", name), run_id=run_id)
        # load metrics
        with open(metrics_path) as f:
            report = json.load(f)
        acc = report.get("accuracy")
        f1_macro = None
        macro = report.get("macro avg") or report.get("macro_avg")
        if macro and isinstance(macro, dict):
            f1_macro = macro.get("f1-score") or macro.get("f1_score")

        results.append({"name": name, "model_path": model_path, "run_id": run_id, "accuracy": acc, "f1_macro": f1_macro, "params": params})

    # save results
    out_csv = os.path.join("outputs", "experiments", "results.json")
    with open(out_csv, "w") as f:
        json.dump(results, f, indent=2)

    # print summary table
    print("\nExperiment results summary:")
    for r in results:
        print(r["name"], "accuracy=", r["accuracy"], "f1_macro=", r["f1_macro"], "run_id=", r["run_id"]) 

    # create simple comparison plot
    try:
        import matplotlib.pyplot as plt

        names = [r["name"] for r in results]
        accs = [float(r["accuracy"]) if r["accuracy"] is not None else 0.0 for r in results]
        f1s = [float(r["f1_macro"]) if r["f1_macro"] is not None else 0.0 for r in results]

        x = range(len(names))
        plt.figure(figsize=(10, 5))
        plt.bar(x, accs, width=0.4, label="accuracy")
        plt.bar([i + 0.4 for i in x], f1s, width=0.4, label="f1_macro")
        plt.xticks([i + 0.2 for i in x], names, rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join("outputs", "experiments", "comparison.png")
        plt.legend()
        plt.savefig(plot_path)
        print("Comparison plot saved to", plot_path)
    except Exception as e:
        print("Failed to create plot:", e)


if __name__ == "__main__":
    run_experiments()
