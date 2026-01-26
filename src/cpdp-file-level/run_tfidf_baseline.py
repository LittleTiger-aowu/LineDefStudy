"""
Run TF-IDF baseline on raw source code.

Example:
  python script/cpdp-file-level/run_tfidf_baseline.py --task groovy_to_lucene
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def load_split(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "g_mean": float(np.sqrt(tpr * tnr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cpdp/all_files.parquet")
    parser.add_argument("--splits_dir", default="data/cpdp/splits")
    parser.add_argument("--task", required=True)
    parser.add_argument("--max_features", type=int, default=20000)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    splits_dir = Path(args.splits_dir)
    task_dir = splits_dir / args.task

    train_df = load_split(task_dir / "train.jsonl")
    val_df = load_split(task_dir / "val.jsonl")
    test_df = load_split(task_dir / "test.jsonl")

    train = train_df.merge(df, on="uid")
    val = val_df.merge(df, on="uid")
    test = test_df.merge(df, on="uid")

    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train = vectorizer.fit_transform(train["src"].tolist())
    X_val = vectorizer.transform(val["src"].tolist())
    X_test = vectorizer.transform(test["src"].tolist())

    y_train = train["y"].to_numpy()
    y_val = val["y"].to_numpy()
    y_test = test["y"].to_numpy() if "y" in test.columns else None

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    if y_test is not None:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        metrics.update({"task": args.task, "model": "tfidf_logreg"})

        output_dir = Path("data/cpdp/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"baselines_{args.task}.json").open("w", encoding="utf-8") as f:
            json.dump([metrics], f, indent=2)
        summary_path = output_dir / "summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df = pd.concat([summary_df, pd.DataFrame([metrics])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([metrics])
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
