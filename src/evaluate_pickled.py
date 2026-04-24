import argparse
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_pickled(test_pkl: str, model_path: str, out_dir: str = "outputs"):
    with open(test_pkl, "rb") as f:
        test = pickle.load(f)

    X_test = test["text"]
    y_test = test["label"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)

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
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pkl", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()
    evaluate_pickled(args.test_pkl, args.model_path, args.out_dir)
