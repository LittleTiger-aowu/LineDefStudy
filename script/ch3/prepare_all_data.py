"""
Prepare Chapter 3 parquet data.

Example:
  python script/ch3/prepare_all_data.py --csv_glob "data/**/*.csv"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from glob import glob
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<stem>.+?)(?:_ground-truth-files_dataset|_ground-truth-files|_groundtruth-files_dataset)$"
)


def parse_project_version(csv_path: Path) -> tuple[str, str]:
    stem = csv_path.stem
    match = FILENAME_RE.match(stem)
    if match:
        stem = match.group("stem")
    if "-" in stem:
        project, version = stem.split("-", 1)
    else:
        parts = stem.split("_")
        if len(parts) >= 2:
            version = parts[-1]
            project = "_".join(parts[:-1])
        else:
            project = stem
            version = "unknown"
    return project.lower(), version


def canonicalize_src(src: str) -> str:
    if not isinstance(src, str):
        return ""
    normalized = src.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    stripped = [line.rstrip() for line in lines]
    return "\n".join(stripped)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def count_loc(src: str) -> int:
    return sum(1 for line in src.splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_glob", default="data/**/*_ground-truth-files_dataset.csv")
    parser.add_argument("--output", default="data/processed/all_files.parquet")
    parser.add_argument("--project_vocab", default="data/processed/project_vocab.json")
    parser.add_argument("--lang", default="java")
    args = parser.parse_args()

    if Path(args.csv_glob).is_absolute():
        paths = [Path(p) for p in glob(args.csv_glob)]
    else:
        paths = [Path(p) for p in Path(".").glob(args.csv_glob)]
    if not paths:
        raise FileNotFoundError(f"No CSV files found for glob: {args.csv_glob}")

    records = []
    projects = []
    for csv_path in paths:
        project, version = parse_project_version(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin1")
        expected_cols = {"File", "Bug", "SRC"}
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        for _, row in df.iterrows():
            file_path = str(row["File"])
            src = str(row["SRC"])
            canonical_src = canonicalize_src(src)
            uid = f"{project}::{version}::{file_path}"
            records.append(
                {
                    "uid": uid,
                    "project": project,
                    "version": version,
                    "file_path": file_path,
                    "y": int(row["Bug"]),
                    "src": src,
                    "sha1": sha1_text(canonical_src),
                    "lang": args.lang,
                    "src_len": len(src),
                    "loc": count_loc(src),
                }
            )
        projects.append(project)

    out_df = pd.DataFrame.from_records(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    unique_projects = sorted(out_df["project"].unique().tolist())
    project_vocab = {name: idx for idx, name in enumerate(unique_projects)}
    vocab_path = Path(args.project_vocab)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(project_vocab, f, indent=2)

    print(f"Saved {len(out_df)} rows to {output_path}")
    print(f"Saved project vocab with {len(project_vocab)} entries to {vocab_path}")


if __name__ == "__main__":
    main()
