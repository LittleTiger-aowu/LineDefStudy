"""
Train Graph-CodeBERT-CPDP model.

Example:
  python script/cpdp-file-level/train_deep_cpdp.py --task groovy_to_lucene --enable_dann 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from models.file_cpdp_model import GraphCodeBertCPDP


class FileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, include_label: bool) -> None:
        self.df = df.reset_index(drop=True)
        self.include_label = include_label

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        record = {
            "uid": row["uid"],
            "src": row["src"],
        }
        if self.include_label:
            record["y"] = int(row["y"])
        return record


def collate_batch(
    batch: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_lines: int,
    max_tokens: int,
) -> dict[str, Any]:
    uids = [item["uid"] for item in batch]
    lines_per_file = []
    line_ptr = [0]
    for item in batch:
        lines = str(item["src"]).splitlines()
        if not lines:
            lines = [""]
        lines = lines[:max_lines]
        lines_per_file.append(lines)
        line_ptr.append(line_ptr[-1] + len(lines))

    flat_lines = [line for lines in lines_per_file for line in lines]
    encoded = tokenizer(
        flat_lines,
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )

    labels = None
    if "y" in batch[0]:
        labels = torch.tensor([item["y"] for item in batch], dtype=torch.long)

    return {
        "uids": uids,
        "flat_input_ids": encoded["input_ids"],
        "flat_attention_mask": encoded["attention_mask"],
        "line_ptr": torch.tensor(line_ptr, dtype=torch.long),
        "file_labels": labels,
        "lines_per_file": lines_per_file,
    }


def load_split(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
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


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(y_prob)
    best_thr = 0.5
    best_mcc = -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr
    return best_thr


def infinite_loader(loader: DataLoader) -> Iterable[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", default="data/cpdp/all_files.parquet")
    parser.add_argument("--splits_dir", default="data/cpdp/splits")
    parser.add_argument("--output_dir", default="data/cpdp/results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_lines", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--shared_dim", type=int, default=128)
    parser.add_argument("--private_dim", type=int, default=128)
    parser.add_argument("--lambda_dom", type=float, default=0.1)
    parser.add_argument("--lambda_ortho", type=float, default=0.1)
    parser.add_argument("--enable_dann", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.input)
    splits_dir = Path(args.splits_dir)
    task_dir = splits_dir / args.task

    train_df = load_split(task_dir / "train.jsonl")
    val_df = load_split(task_dir / "val.jsonl")
    test_df = load_split(task_dir / "test.jsonl")

    train = train_df.merge(df, on="uid")
    val = val_df.merge(df, on="uid")
    test = test_df.merge(df, on="uid")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = FileDataset(train, include_label=True)
    val_dataset = FileDataset(val, include_label=True)
    test_dataset = FileDataset(test, include_label=True)

    class_counts = train["y"].value_counts().to_dict()
    pos_weight = class_counts.get(0, 1) / max(class_counts.get(1, 1), 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)

    sample_weights = train["y"].map(lambda y: pos_weight if y == 1 else 1.0).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    collate = lambda batch: collate_batch(batch, tokenizer, args.max_lines, args.max_tokens)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphCodeBertCPDP(
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        private_dim=args.private_dim,
        enable_dann=bool(args.enable_dann),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    dom_loss_fn = nn.CrossEntropyLoss()

    target_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    target_iter = infinite_loader(target_loader)

    for epoch in range(args.epochs):
        model.train()
        grl_lambda = (epoch + 1) / args.epochs
        for batch_s in train_loader:
            batch_t = next(target_iter)

            input_ids_s = batch_s["flat_input_ids"].to(device)
            mask_s = batch_s["flat_attention_mask"].to(device)
            line_ptr_s = batch_s["line_ptr"].to(device)
            labels_s = batch_s["file_labels"].to(device)

            input_ids_t = batch_t["flat_input_ids"].to(device)
            mask_t = batch_t["flat_attention_mask"].to(device)
            line_ptr_t = batch_t["line_ptr"].to(device)

            out_s = model(input_ids_s, mask_s, line_ptr_s, labels_s, grl_lambda=grl_lambda)
            out_t = model(input_ids_t, mask_t, line_ptr_t, None, grl_lambda=grl_lambda)

            cls_loss = cls_loss_fn(out_s["logits_cls"], labels_s)
            ortho_loss = 0.5 * (out_s["losses"]["ortho"] + out_t["losses"]["ortho"])

            dom_loss = torch.tensor(0.0, device=device)
            if model.enable_dann:
                dom_logits_s = out_s["domain_logits"]
                dom_logits_t = out_t["domain_logits"]
                dom_labels_s = torch.zeros(dom_logits_s.size(0), dtype=torch.long, device=device)
                dom_labels_t = torch.ones(dom_logits_t.size(0), dtype=torch.long, device=device)
                dom_loss = dom_loss_fn(dom_logits_s, dom_labels_s) + dom_loss_fn(dom_logits_t, dom_labels_t)

            loss = cls_loss + args.lambda_dom * dom_loss + args.lambda_ortho * ortho_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["flat_input_ids"].to(device)
            mask = batch["flat_attention_mask"].to(device)
            line_ptr = batch["line_ptr"].to(device)
            labels = batch["file_labels"].to(device)
            out = model(input_ids, mask, line_ptr, None, grl_lambda=1.0)
            probs = torch.softmax(out["logits_cls"], dim=1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    threshold = find_best_threshold(val_labels, val_probs)

    test_probs = []
    test_labels = []
    test_line_info = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["flat_input_ids"].to(device)
            mask = batch["flat_attention_mask"].to(device)
            line_ptr = batch["line_ptr"].to(device)
            labels = batch["file_labels"]
            out = model(input_ids, mask, line_ptr, None, grl_lambda=1.0)
            probs = torch.softmax(out["logits_cls"], dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_labels.extend(labels.numpy())
            test_line_info.append(
                (
                    batch["uids"],
                    labels.numpy(),
                    batch["lines_per_file"],
                    line_ptr.cpu(),
                    out["att_weights"].cpu(),
                )
            )

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    metrics = compute_metrics(test_labels, test_probs, threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"deep_{args.task}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    explain_path = output_dir / f"explain_{args.task}.jsonl"
    with explain_path.open("w", encoding="utf-8") as f:
        for uids, labels, lines_per_file, line_ptr, att_weights in test_line_info:
            for i, uid in enumerate(uids):
                if labels[i] != 1:
                    continue
                start = int(line_ptr[i])
                end = int(line_ptr[i + 1])
                weights = att_weights[start:end]
                lines = lines_per_file[i]
                topk = torch.topk(weights, k=min(5, len(weights)))
                for rank, (weight, idx) in enumerate(zip(topk.values, topk.indices), start=1):
                    record = {
                        "uid": uid,
                        "rank": rank,
                        "line_no": int(idx.item()) + 1,
                        "weight": float(weight.item()),
                        "text": lines[int(idx.item())][:200],
                    }
                    f.write(json.dumps(record))
                    f.write("\n")

    print(f"Saved metrics to {output_dir / f'deep_{args.task}.json'}")


if __name__ == "__main__":
    main()
