"""
Dump Chapter 3 representations and interpretability artifacts.

Example:
  python src/ch3/dump_representations.py --data-parquet data/processed/all_files.parquet --ckpt outputs/ch3_ckpt/best.pt --outdir outputs/ch3_dump/run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rep.collate import collate_batch
from src.rep.dataset import FileDataset
from src.rep.encoder import CodeBertBlockEncoder, build_tokenizer
from src.rep.model import RepresentationModel
from src.rep.struct_prior import StatsMLP, TypeEmbedding, num_block_types


ALPHA_SUM_TOL = 1e-3


def coverage_overlap(spans):
    if not spans:
        return 0.0, 0.0
    total_lines = max(end for _, end in spans)
    if total_lines <= 0:
        return 0.0, 0.0
    counts = [0] * total_lines
    for start, end in spans:
        for line in range(start, end + 1):
            counts[line - 1] += 1
    covered = sum(1 for c in counts if c > 0)
    overlap = sum(1 for c in counts if c > 1)
    return covered / total_lines, overlap / total_lines


def alpha_entropy(alpha: np.ndarray, start: int, end: int) -> float:
    segment = alpha[start:end]
    if segment.size == 0:
        return 0.0
    eps = 1e-8
    return float(-np.sum(segment * np.log(segment + eps)))


def assert_non_overlap(spans, uid: str) -> None:
    if not spans:
        return
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    for i in range(1, len(spans_sorted)):
        prev_start, prev_end = spans_sorted[i - 1]
        cur_start, _ = spans_sorted[i]
        if prev_end >= cur_start:
            raise ValueError(f"Overlapping spans in {uid}: {spans_sorted[i - 1]} vs {spans_sorted[i]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parquet", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=128)
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--win-size-lines", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--dedup-by-sha1", default="within_project")
    parser.add_argument("--dump-block-text", type=int, default=0)
    parser.add_argument("--codebert-path", default="E:\\project\\WYP\\CPDP\\CodeBert")
    parser.add_argument("--local-files-only", type=int, default=1)
    parser.add_argument("--max-blocks-per-file", type=int, default=128)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    project_vocab = ckpt.get("project_vocab")
    if project_vocab is None:
        raise ValueError("Checkpoint missing project_vocab")
    config = ckpt.get("config", {})

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
            include_block_text=bool(args.dump_block_text),
            max_blocks_per_file=args.max_blocks_per_file,
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CodeBertBlockEncoder(args.codebert_path, local_files_only=bool(args.local_files_only))
    type_embed = TypeEmbedding(num_block_types(), config.get("d_t", 32))
    stats_mlp = StatsMLP(3, config.get("d_p", 32))
    model = RepresentationModel(
        d_h=config.get("d_h", 256),
        d_sh=config.get("d_sh", 128),
        d_pr=config.get("d_pr", 128),
        num_projects=len(project_vocab),
        input_dim=768 + config.get("d_t", 32) + config.get("d_p", 32),
    )

    encoder.load_state_dict(ckpt["encoder"])
    type_embed.load_state_dict(ckpt["type_embed"])
    stats_mlp.load_state_dict(ckpt["stats_mlp"])
    model.load_state_dict(ckpt["model"])

    encoder.to(device)
    type_embed.to(device)
    stats_mlp.to(device)
    model.to(device)

    encoder.eval()
    type_embed.eval()
    stats_mlp.eval()
    model.eval()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "project_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(project_vocab, f, indent=2)

    meta_path = outdir / "meta.jsonl"
    topk_path = outdir / "topk_blocks.jsonl"

    H_file_list = []
    Z_sh_list = []
    Z_pr_list = []
    proj_id_list = []
    y_list = []
    H_blk_list = []
    alpha_list = []
    blk_ptr_global = [0]

    blocks_per_file = []
    parse_ok_list = []
    parse_error_list = []
    coverage_list = []
    overlap_list = []
    alpha_entropy_list = []
    alpha_sum_dev = []
    ortho_fro_list = []

    with meta_path.open("w", encoding="utf-8") as meta_f, topk_path.open("w", encoding="utf-8") as topk_f:
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["flat_input_ids"].to(device)
                attention_mask = batch["flat_attention_mask"].to(device)
                blk_ptr = batch["blk_ptr"].to(device)
                struct_type_ids = batch["struct_type_ids"].to(device)
                struct_stats = batch["struct_stats"].to(device)
                edge_indices = batch["edge_indices"]
                parse_ok = batch["file_parse_ok"].cpu().numpy().tolist()
                parse_has_error = batch["file_parse_has_error"].cpu().numpy().tolist()
                meta = batch["meta"]

                h_sem = encoder(input_ids, attention_mask)
                type_emb = type_embed(struct_type_ids)
                stats_emb = stats_mlp(struct_stats)
                e_struct = torch.cat([type_emb, stats_emb], dim=1)

                outputs = model(h_sem, e_struct, blk_ptr, edge_indices=edge_indices)

                H_file_list.append(outputs["H_file"].detach().cpu().numpy())
                Z_sh_list.append(outputs["Z_sh"].detach().cpu().numpy())
                Z_pr_list.append(outputs["Z_pr"].detach().cpu().numpy())
                H_blk_list.append(outputs["H_blk"].detach().cpu().numpy())
                alpha_vals = outputs["alpha_values"].detach().cpu().numpy()
                alpha_list.append(alpha_vals)
                ortho_fro_list.append(float(outputs["loss_ortho"].item()))

                blk_ptr_list = blk_ptr.cpu().numpy().tolist()
                for i in range(len(blk_ptr_list) - 1):
                    num_blocks = int(blk_ptr_list[i + 1] - blk_ptr_list[i])
                    blk_ptr_global.append(blk_ptr_global[-1] + num_blocks)

                for i, uid in enumerate(meta["uid"]):
                    proj = meta["project"][i]
                    proj_id = project_vocab[proj]
                    proj_id_list.append(proj_id)
                    y_list.append(int(meta["y"][i]))

                    spans = [tuple(span) for span in meta["spans"][i]]
                    type_ids = meta["type_ids"][i]
                    type_strs = meta["type_strs"][i]
                    num_blocks = meta["num_blocks"][i]
                    parse_flag = bool(parse_ok[i])
                    parse_error_flag = bool(parse_has_error[i])

                    assert_non_overlap(spans, uid)

                    record = {
                        "uid": uid,
                        "project": proj,
                        "version": meta["version"][i],
                        "file_path": meta["file_path"][i],
                        "y": int(meta["y"][i]),
                        "sha1": meta["sha1"][i],
                        "num_blocks": int(num_blocks),
                        "parse_ok": parse_flag,
                        "parse_has_error": parse_error_flag,
                        "spans": [list(span) for span in spans],
                        "type_ids": [int(t) for t in type_ids],
                        "type_strs": type_strs,
                    }
                    if meta.get("block_texts") is not None:
                        record["block_texts"] = meta["block_texts"][i]
                    meta_f.write(json.dumps(record))
                    meta_f.write("\n")

                    start = int(blk_ptr_list[i])
                    end = int(blk_ptr_list[i + 1])
                    alpha_seg = alpha_vals[start:end]
                    blocks_per_file.append(num_blocks)
                    parse_ok_list.append(parse_flag)
                    parse_error_list.append(parse_error_flag)
                    cov, ov = coverage_overlap(spans)
                    coverage_list.append(cov)
                    overlap_list.append(ov)
                    if parse_flag:
                        alpha_entropy_list.append(alpha_entropy(alpha_vals, start, end))
                        alpha_sum_dev.append(float(abs(alpha_seg.sum() - 1.0)))
                        if alpha_seg.size > 0 and abs(alpha_seg.sum() - 1.0) > ALPHA_SUM_TOL:
                            raise ValueError(f"alpha sum check failed for {uid}: {alpha_seg.sum():.6f}")

                    if alpha_seg.size > 0:
                        topk = min(args.topk, alpha_seg.size)
                        top_indices = np.argsort(-alpha_seg)[:topk]
                        topk_rec = {
                            "uid": uid,
                            "y": int(meta["y"][i]),
                            "topk": [],
                        }
                        for rank, idx in enumerate(top_indices, start=1):
                            span = spans[int(idx)]
                            topk_rec["topk"].append(
                                {
                                    "rank": rank,
                                    "block_id": int(idx),
                                    "alpha": float(alpha_seg[int(idx)]),
                                    "span": [int(span[0]), int(span[1])],
                                    "type": type_strs[int(idx)],
                                }
                            )
                        topk_f.write(json.dumps(topk_rec))
                        topk_f.write("\n")

    H_file = np.concatenate(H_file_list, axis=0).astype(np.float32)
    Z_sh = np.concatenate(Z_sh_list, axis=0).astype(np.float32)
    Z_pr = np.concatenate(Z_pr_list, axis=0).astype(np.float32)
    H_blk = np.concatenate(H_blk_list, axis=0).astype(np.float32)
    alpha_values = np.concatenate(alpha_list, axis=0).astype(np.float32)
    proj_id = np.array(proj_id_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.int64)
    blk_ptr = np.array(blk_ptr_global, dtype=np.int64)

    if H_file.shape[0] != len(proj_id) or H_file.shape[0] != len(y):
        raise ValueError("Mismatch between file-level arrays and metadata length")
    if blk_ptr.shape[0] != H_file.shape[0] + 1:
        raise ValueError("blk_ptr length mismatch")
    if blk_ptr[0] != 0 or blk_ptr[-1] != H_blk.shape[0]:
        raise ValueError("blk_ptr endpoints mismatch")
    if alpha_values.shape[0] != H_blk.shape[0]:
        raise ValueError("alpha_values length mismatch")

    np.savez(
        outdir / "repr.npz",
        H_file=H_file,
        Z_sh=Z_sh,
        Z_pr=Z_pr,
        proj_id=proj_id,
        y=y,
        H_blk=H_blk,
        alpha_values=alpha_values,
        blk_ptr=blk_ptr,
    )

    projects = {}
    for proj in proj_id_list:
        name = [k for k, v in project_vocab.items() if v == proj][0]
        projects[name] = projects.get(name, 0) + 1

    stats = {
        "N_raw": int(dataset.raw_len),
        "N_dedup": int(len(dataset)),
        "dedup_rule": f"{dataset.dedup_rule}_sha1" if dataset.dedup_rule != "off" else "off",
        "projects": projects,
        "pos_ratio": float(np.mean(y) if y.size else 0.0),
        "blocks_per_file": {
            "mean": float(np.mean(blocks_per_file)) if blocks_per_file else 0.0,
            "p50": float(np.percentile(blocks_per_file, 50)) if blocks_per_file else 0.0,
            "p90": float(np.percentile(blocks_per_file, 90)) if blocks_per_file else 0.0,
        },
        "parse_fail_ratio": float(1.0 - np.mean(parse_ok_list)) if parse_ok_list else 0.0,
        "parse_error_ratio": float(np.mean(parse_error_list)) if parse_error_list else 0.0,
        "coverage_ratio": {
            "mean": float(np.mean(coverage_list)) if coverage_list else 0.0,
            "p50": float(np.percentile(coverage_list, 50)) if coverage_list else 0.0,
        },
        "overlap_ratio": {
            "mean": float(np.mean(overlap_list)) if overlap_list else 0.0,
            "p50": float(np.percentile(overlap_list, 50)) if overlap_list else 0.0,
        },
        "alpha_entropy": {
            "mean": float(np.mean(alpha_entropy_list)) if alpha_entropy_list else 0.0,
            "p50": float(np.percentile(alpha_entropy_list, 50)) if alpha_entropy_list else 0.0,
        },
        "alpha_sum_abs_dev": {
            "mean": float(np.mean(alpha_sum_dev)) if alpha_sum_dev else 0.0,
            "p50": float(np.percentile(alpha_sum_dev, 50)) if alpha_sum_dev else 0.0,
        },
        "ortho_fro": {
            "mean": float(np.mean(ortho_fro_list)) if ortho_fro_list else 0.0,
            "p50": float(np.percentile(ortho_fro_list, 50)) if ortho_fro_list else 0.0,
        },
    }

    with (outdir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved dump to {outdir}")


if __name__ == "__main__":
    main()
