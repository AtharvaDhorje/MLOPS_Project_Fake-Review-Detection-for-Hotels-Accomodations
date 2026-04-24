import argparse
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame):
    # basic cleaning: dropna, strip
    df = df.dropna()
    text_col = [c for c in df.columns if c.lower() in ("text", "review", "content", "opinion")]
    label_col = [c for c in df.columns if c.lower() in ("label", "class", "target", "deceptive", "truthful")]
    if not text_col:
        text = df.iloc[:, 0].astype(str)
    else:
        text = df[text_col[0]].astype(str)
    if not label_col:
        label = df.iloc[:, 1]
    else:
        label = df[label_col[0]]
    df2 = pd.DataFrame({"text": text.str.strip(), "label": label})
    return df2


def save_splits(df, out_dir="data/processed", test_size=0.2, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)
    train = pd.DataFrame({"text": X_train, "label": y_train})
    test = pd.DataFrame({"text": X_test, "label": y_test})
    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test, f)
    print(f"Saved train/test to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df2 = preprocess(df)
    save_splits(df2, args.out_dir, args.test_size, args.random_state)
