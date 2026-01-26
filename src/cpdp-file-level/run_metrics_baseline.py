"""
Run classical baselines on extracted metrics.

Example:
  python script/cpdp-file-level/run_metrics_baseline.py --task groovy_to_lucene
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier


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


def train_logreg(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def train_xgboost(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    pos = max(y.sum(), 1)
    neg = max(len(y) - pos, 1)
    scale_pos_weight = neg / pos
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y)
    return model


def run_task(
    task: str,
    metrics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_dir: Path,
) -> list[dict[str, float]]:
    task_dir = splits_dir / task
    train_df = load_split(task_dir / "train.jsonl")
    val_df = load_split(task_dir / "val.jsonl")
    test_df = load_split(task_dir / "test.jsonl")

    train = train_df.merge(metrics_df, left_on="uid", right_index=True)
    val = val_df.merge(metrics_df, left_on="uid", right_index=True)
    test = test_df.merge(metrics_df, left_on="uid", right_index=True)
    test = test.merge(labels_df, on="uid", how="left")

    feature_cols = [c for c in metrics_df.columns if c != "uid"]
    X_train = train[feature_cols].to_numpy()
    y_train = train["y"].to_numpy()
    X_val = val[feature_cols].to_numpy()
    y_val = val["y"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["y"].to_numpy() if "y" in test.columns else None

    results = []

    logreg = train_logreg(X_train, y_train)
    logreg_prob = logreg.predict_proba(X_test)[:, 1]
    if y_test is not None:
        metrics = compute_metrics(y_test, logreg_prob)
        metrics.update({"task": task, "model": "logreg"})
        results.append(metrics)

    xgb = train_xgboost(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    if y_test is not None:
        metrics = compute_metrics(y_test, xgb_prob)
        metrics.update({"task": task, "model": "xgboost"})
        results.append(metrics)

    output_dir = Path("data/cpdp/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"baselines_{task}.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="data/cpdp/all_metrics.parquet")
    parser.add_argument("--files", default="data/cpdp/all_files.parquet")
    parser.add_argument("--splits_dir", default="data/cpdp/splits")
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    metrics_df = pd.read_parquet(args.metrics)
    labels_df = pd.read_parquet(args.files)[["uid", "y"]]
    splits_dir = Path(args.splits_dir)
    tasks = [args.task] if args.task else [p.name for p in splits_dir.iterdir() if p.is_dir()]

    all_results = []
    for task in tasks:
        all_results.extend(run_task(task, metrics_df, labels_df, splits_dir))

    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = Path("data/cpdp/results/summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
