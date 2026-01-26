"""
Create CPDP source->target splits.

Example:
  python script/cpdp-file-level/make_cpdp_splits.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def write_jsonl(path: Path, df: pd.DataFrame, include_y: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {"uid": row["uid"]}
            if include_y:
                record["y"] = int(row["y"])
            f.write(pd.Series(record).to_json())
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cpdp/all_files.parquet")
    parser.add_argument("--output_dir", default="data/cpdp/splits")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    projects = sorted(df["project"].unique().tolist())
    rng = np.random.default_rng(args.seed)

    for source in projects:
        for target in projects:
            if source == target:
                continue
            source_df = df[df["project"] == source].copy()
            target_df = df[df["project"] == target].copy()

            indices = rng.permutation(len(source_df))
            split_idx = int(len(indices) * (1.0 - args.val_ratio))
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
            train_df = source_df.iloc[train_idx]
            val_df = source_df.iloc[val_idx]

            source_sha = set(pd.concat([train_df, val_df])["sha1"].tolist())
            target_df = target_df[~target_df["sha1"].isin(source_sha)].copy()

            task_dir = Path(args.output_dir) / f"{source}_to_{target}"
            write_jsonl(task_dir / "train.jsonl", train_df, include_y=True)
            write_jsonl(task_dir / "val.jsonl", val_df, include_y=True)
            write_jsonl(task_dir / "test.jsonl", target_df, include_y=False)
            print(
                f"{source}_to_{target}: train={len(train_df)} val={len(val_df)} test={len(target_df)}"
            )


if __name__ == "__main__":
    main()
