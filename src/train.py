import argparse
import os
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_dataset, prepare_data


def train_baseline(data_path: str, model_out: str):
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english"),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    out_dir = Path(model_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="deceptive-opinion.csv")
    parser.add_argument("--model-out", type=str, default="models/tfidf_lr.joblib")
    args = parser.parse_args()
    train_baseline(args.data_path, args.model_out)
