import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.data import load_dataset, prepare_data


def evaluate(data_path: str, model_path: str, out_dir: str = "outputs"):
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--data-path", type=str, default="deceptive-opinion.csv")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()
    evaluate(args.data_path, args.model_path, args.out_dir)
