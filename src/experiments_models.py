import os
import json
import yaml
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

from src.evaluate_pickled import evaluate_pickled


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f).get("train", {})


def build_pipeline_for(estimator, params):
    ngram_range = (params.get("ngram_min", 1), params.get("ngram_max", 2))
    tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=params.get("max_features", 20000), stop_words="english")
    return Pipeline([("tfidf", tfidf), ("clf", estimator)])


def run():
    params = load_params()
    train_pkl = os.path.join("data", "processed", "train.pkl")
    test_pkl = os.path.join("data", "processed", "test.pkl")
    if not os.path.exists(train_pkl) or not os.path.exists(test_pkl):
        raise RuntimeError("Train/test splits not found. Run run_pipeline first.")

    models = {
        "logistic": LogisticRegression(C=params.get("C", 1.0), max_iter=params.get("max_iter", 1000)),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "multinomial_nb": MultinomialNB(),
        "sgd_log": SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42),
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/experiments_models", exist_ok=True)

    results = []
    mlflow.set_experiment("fake-review-multi-models")

    for name, estimator in models.items():
        print("Training", name)
        pipeline = build_pipeline_for(estimator, params)
        with mlflow.start_run() as run:
            mlflow.log_param("model", name)
            mlflow.log_params({"ngram_min": params.get("ngram_min"), "ngram_max": params.get("ngram_max"), "max_features": params.get("max_features")})
            # fit on train
            import pickle
            with open(train_pkl, "rb") as f:
                train = pickle.load(f)
            X_train = train["text"]
            y_train = train["label"]
            pipeline.fit(X_train, y_train)

            model_path = os.path.join("models", f"{name}.joblib")
            joblib.dump(pipeline, model_path)
            mlflow.sklearn.log_model(pipeline, "model")

            run_id = run.info.run_id

        # Evaluate and log artifacts (evaluate_pickled will attach to MLflow run if run_id passed)
        out_dir = os.path.join("outputs", "experiments_models", name)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = evaluate_pickled(test_pkl, model_path, out_dir=out_dir, run_id=run_id)
        with open(metrics_path) as f:
            report = json.load(f)
        acc = report.get("accuracy")
        macro = report.get("macro avg") or report.get("macro_avg")
        f1_macro = macro.get("f1-score") if macro else None

        results.append({"name": name, "run_id": run_id, "model_path": model_path, "accuracy": acc, "f1_macro": f1_macro})
        print(name, "done — accuracy=", acc, "f1_macro=", f1_macro, "run_id=", run_id)

    # save results
    res_path = os.path.join("outputs", "experiments_models", "results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # plot comparison
    try:
        import matplotlib.pyplot as plt
        names = [r["name"] for r in results]
        accs = [float(r["accuracy"]) if r["accuracy"] is not None else 0.0 for r in results]
        f1s = [float(r["f1_macro"]) if r["f1_macro"] is not None else 0.0 for r in results]
        x = range(len(names))
        plt.figure(figsize=(8, 4))
        plt.bar(x, accs, width=0.4, label='accuracy')
        plt.bar([i+0.4 for i in x], f1s, width=0.4, label='f1_macro')
        plt.xticks([i+0.2 for i in x], names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join("outputs", "experiments_models", "comparison.png")
        plt.savefig(plot_path)
        print("Comparison plot saved to", plot_path)
    except Exception as e:
        print("Plot failed:", e)


if __name__ == '__main__':
    run()
