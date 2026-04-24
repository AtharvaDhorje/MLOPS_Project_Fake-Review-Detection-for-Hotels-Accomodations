import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _detect_text_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    # heuristics to find text and label columns
    text_cols = [c for c in df.columns if c.lower() in ("text", "review", "content", "opinion")]
    label_cols = [c for c in df.columns if c.lower() in ("label", "class", "target", "deceptive", "truthful")]
    if not text_cols:
        # fallback to largest text-like column
        text_cols = sorted(df.columns, key=lambda c: df[c].astype(str).map(len).median(), reverse=True)[:1]
    if not label_cols:
        # try binary columns with small unique values
        candidates = [c for c in df.columns if df[c].nunique() <= 5]
        label_cols = candidates[:1]
    return text_cols[0], label_cols[0]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    text_col, label_col = _detect_text_label_columns(df)
    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    # normalize labels to 0/1 if possible
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.lower()
        mapping = None
        if set(df["label"]) <= {"deceptive", "truthful"}:
            mapping = {"deceptive": 1, "truthful": 0}
        elif set(df["label"]) <= {"fake", "real"}:
            mapping = {"fake": 1, "real": 0}
        if mapping:
            df["label"] = df["label"].map(mapping)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"] if df["label"].nunique() > 1 else None
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="deceptive-opinion.csv")
    args = parser.parse_args()
    df = load_dataset(args.data)
    print("Loaded", len(df), "rows")