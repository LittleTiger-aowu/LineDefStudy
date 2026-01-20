from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .tpsm import Block, extract_blocks


@dataclass
class FileSample:
    uid: str
    project: str
    version: str
    file_path: str
    y: int
    src: str
    sha1: str


class FileDataset:
    def __init__(
        self,
        parquet_path: str,
        project_vocab: dict,
        dedup_by_sha1: str = "within_project",
        max_files: Optional[int] = None,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        required = {"uid", "project", "version", "file_path", "y", "src", "sha1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in parquet: {missing}")
        self.raw_len = len(df)
        if dedup_by_sha1 == "within_project":
            df = df.drop_duplicates(subset=["project", "sha1"])
        if max_files is not None:
            df = df.head(max_files)
        self.df = df.reset_index(drop=True)
        self.project_vocab = project_vocab
        self.dedup_rule = dedup_by_sha1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> FileSample:
        row = self.df.iloc[idx]
        return FileSample(
            uid=row["uid"],
            project=row["project"],
            version=row["version"],
            file_path=row["file_path"],
            y=int(row["y"]),
            src=str(row["src"]),
            sha1=row["sha1"],
        )


def iter_blocks(src: str, lang: str = "java", win_size_lines: int = 20) -> List[Block]:
    return extract_blocks(src, lang=lang, win_size_lines=win_size_lines)
