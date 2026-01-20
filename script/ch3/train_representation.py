"""
Train Chapter 3 representation model.

Example:
  python script/ch3/train_representation.py --data-parquet data/processed/all_files.parquet --project-vocab data/processed/project_vocab.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rep.collate import collate_batch
from src.rep.dataset import FileDataset
from src.rep.encoder import CodeBertBlockEncoder, build_tokenizer
from src.rep.model import RepresentationModel
from src.rep.struct_prior import StatsMLP, TypeEmbedding, num_block_types


def alpha_entropy(alpha: torch.Tensor, blk_ptr: torch.Tensor, parse_ok: torch.Tensor) -> float:
    entropies = []
    eps = 1e-8
    for i in range(len(blk_ptr) - 1):
        if not bool(parse_ok[i]):
            continue
        start = int(blk_ptr[i].item())
        end = int(blk_ptr[i + 1].item())
        segment = alpha[start:end]
        if segment.numel() == 0:
            continue
        ent = -torch.sum(segment * torch.log(segment + eps)).item()
        entropies.append(ent)
    if not entropies:
        return 0.0
    return float(np.mean(entropies))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parquet", required=True)
    parser.add_argument("--project-vocab", required=True)
    parser.add_argument("--output-dir", default="outputs/ch3_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=128)
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--win-size-lines", type=int, default=20)
    parser.add_argument("--d-h", type=int, default=256)
    parser.add_argument("--d-sh", type=int, default=128)
    parser.add_argument("--d-pr", type=int, default=128)
    parser.add_argument("--d-t", type=int, default=32)
    parser.add_argument("--d-p", type=int, default=32)
    parser.add_argument("--lambda-pr", type=float, default=1.0)
    parser.add_argument("--lambda-ortho", type=float, default=0.1)
    parser.add_argument("--dedup-by-sha1", default="within_project")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--lr-codebert", type=float, default=2e-5)
    parser.add_argument("--lr-other", type=float, default=1e-3)
    parser.add_argument("--codebert-path", default="E:\\project\\WYP\\CPDP\\CodeBert")
    parser.add_argument("--local-files-only", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.project_vocab, "r", encoding="utf-8") as f:
        project_vocab = json.load(f)

    dataset = FileDataset(
        args.data_parquet,
        project_vocab,
        dedup_by_sha1=args.dedup_by_sha1,
        max_files=args.max_files,
    )

    tokenizer = build_tokenizer(args.codebert_path, local_files_only=bool(args.local_files_only))

    def _collate(batch):
        return collate_batch(
            batch,
            tokenizer=tokenizer,
            tmax=args.tmax,
            win_size_lines=args.win_size_lines,
            window=args.w,
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CodeBertBlockEncoder(args.codebert_path, local_files_only=bool(args.local_files_only))
    type_embed = TypeEmbedding(num_block_types(), args.d_t)
    stats_mlp = StatsMLP(3, args.d_p)
    model = RepresentationModel(
        d_h=args.d_h,
        d_sh=args.d_sh,
        d_pr=args.d_pr,
        num_projects=len(project_vocab),
        input_dim=768 + args.d_t + args.d_p,
    )

    encoder.to(device)
    type_embed.to(device)
    stats_mlp.to(device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.lr_codebert},
            {"params": type_embed.parameters(), "lr": args.lr_other},
            {"params": stats_mlp.parameters(), "lr": args.lr_other},
            {"params": model.parameters(), "lr": args.lr_other},
        ]
    )

    bug_loss_fn = nn.BCEWithLogitsLoss()
    pr_loss_fn = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    best_loss = None

    for epoch in range(args.epochs):
        encoder.train()
        type_embed.train()
        stats_mlp.train()
        model.train()

        loss_total_acc = []
        loss_bug_acc = []
        loss_pr_acc = []
        loss_ortho_acc = []
        alpha_entropy_acc = []
        parse_fail_acc = []

        for batch in loader:
            input_ids = batch["flat_input_ids"].to(device)
            attention_mask = batch["flat_attention_mask"].to(device)
            blk_ptr = batch["blk_ptr"].to(device)
            struct_type_ids = batch["struct_type_ids"].to(device)
            struct_stats = batch["struct_stats"].to(device)
            edge_indices = batch["edge_indices"]
            parse_ok = batch["file_parse_ok"].to(device)

            meta = batch["meta"]
            y = torch.tensor(meta["y"], dtype=torch.float, device=device)
            proj_ids = torch.tensor(
                [project_vocab[p] for p in meta["project"]],
                dtype=torch.long,
                device=device,
            )

            h_sem = encoder(input_ids, attention_mask)
            type_emb = type_embed(struct_type_ids)
            stats_emb = stats_mlp(struct_stats)
            e_struct = torch.cat([type_emb, stats_emb], dim=1)

            outputs = model(h_sem, e_struct, blk_ptr, edge_indices=edge_indices)

            loss_bug = bug_loss_fn(outputs["logit_bug"].squeeze(-1), y)
            loss_pr = pr_loss_fn(outputs["logits_pr_dom"], proj_ids)
            loss_ortho = outputs["loss_ortho"]
            loss_total = loss_bug + args.lambda_pr * loss_pr + args.lambda_ortho * loss_ortho

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            loss_total_acc.append(loss_total.item())
            loss_bug_acc.append(loss_bug.item())
            loss_pr_acc.append(loss_pr.item())
            loss_ortho_acc.append(loss_ortho.item())
            alpha_entropy_acc.append(alpha_entropy(outputs["alpha_values"], blk_ptr, parse_ok))
            parse_fail_acc.append(1.0 - float(parse_ok.float().mean().item()))

        epoch_log = {
            "epoch": epoch + 1,
            "loss_total": float(np.mean(loss_total_acc)),
            "loss_bug": float(np.mean(loss_bug_acc)),
            "loss_pr_dom": float(np.mean(loss_pr_acc)),
            "loss_ortho": float(np.mean(loss_ortho_acc)),
            "alpha_entropy_mean": float(np.mean(alpha_entropy_acc)),
            "ortho_fro_batch": float(np.mean(loss_ortho_acc)),
            "parse_fail_ratio_batch": float(np.mean(parse_fail_acc)),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_log))
            f.write("\n")

        ckpt = {
            "encoder": encoder.state_dict(),
            "type_embed": type_embed.state_dict(),
            "stats_mlp": stats_mlp.state_dict(),
            "model": model.state_dict(),
            "project_vocab": project_vocab,
            "config": vars(args),
        }

        torch.save(ckpt, output_dir / "last.pt")
        if best_loss is None or epoch_log["loss_total"] < best_loss:
            best_loss = epoch_log["loss_total"]
            torch.save(ckpt, output_dir / "best.pt")

        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_log['loss_total']:.4f}")


if __name__ == "__main__":
    main()
