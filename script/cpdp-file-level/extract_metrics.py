"""
Extract lightweight metrics for CPDP files.

Example:
  python script/cpdp-file-level/extract_metrics.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import lizard
import pandas as pd

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||[^\s]")
KEYWORDS = ["if", "for", "while", "catch", "return"]


def count_keywords(src: str) -> dict[str, int]:
    counts = {}
    for kw in KEYWORDS:
        counts[f"kw_{kw}"] = len(re.findall(rf"\b{kw}\b", src))
    return counts


def analyze_lizard(src: str) -> dict[str, float]:
    try:
        analysis = lizard.analyze_file.analyze_source_code("sample", src)
        functions = analysis.function_list
        if not functions:
            return {
                "num_functions": 0,
                "sum_ccn": 0,
                "avg_ccn": 0,
                "max_ccn": 0,
                "avg_params": 0,
                "max_params": 0,
                "sum_nloc_func": 0,
                "avg_nloc_func": 0,
                "max_nloc_func": 0,
                "parse_ok": 1,
            }
        ccn_vals = [f.cyclomatic_complexity for f in functions]
        params_vals = [len(f.parameters) for f in functions]
        nloc_vals = [f.nloc for f in functions]
        return {
            "num_functions": len(functions),
            "sum_ccn": sum(ccn_vals),
            "avg_ccn": sum(ccn_vals) / len(ccn_vals),
            "max_ccn": max(ccn_vals),
            "avg_params": sum(params_vals) / len(params_vals),
            "max_params": max(params_vals),
            "sum_nloc_func": sum(nloc_vals),
            "avg_nloc_func": sum(nloc_vals) / len(nloc_vals),
            "max_nloc_func": max(nloc_vals),
            "parse_ok": 1,
        }
    except Exception:
        return {
            "num_functions": 0,
            "sum_ccn": 0,
            "avg_ccn": 0,
            "max_ccn": 0,
            "avg_params": 0,
            "max_params": 0,
            "sum_nloc_func": 0,
            "avg_nloc_func": 0,
            "max_nloc_func": 0,
            "parse_ok": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cpdp/all_files.parquet")
    parser.add_argument("--output", default="data/cpdp/all_metrics.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    rows = []
    for _, row in df.iterrows():
        src = str(row["src"])
        loc = len(src.splitlines())
        token_count = len(TOKEN_RE.findall(src))
        metrics = {
            "uid": row["uid"],
            "loc": loc,
            "token_count": token_count,
        }
        metrics.update(count_keywords(src))
        metrics.update(analyze_lizard(src))
        if metrics["parse_ok"] == 0:
            for key in list(metrics.keys()):
                if key in {"uid", "parse_ok"}:
                    continue
                if isinstance(metrics[key], (int, float)):
                    metrics[key] = 0
        rows.append(metrics)

    out_df = pd.DataFrame(rows).set_index("uid")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)
    print(f"Saved metrics for {len(out_df)} files to {output_path}")


if __name__ == "__main__":
    main()
