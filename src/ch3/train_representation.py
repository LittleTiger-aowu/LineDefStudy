"""
Train Chapter 3 representation model.

Example:
  python src/ch3/train_representation.py --data-parquet data/processed/all_files.parquet --project-vocab data/processed/project_vocab.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
import logging
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
    parser.add_argument("--beta-bug-file", type=float, default=0.2)
    parser.add_argument("--dedup-by-sha1", default="within_project")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--lr-codebert", type=float, default=2e-5)
    parser.add_argument("--lr-other", type=float, default=1e-3)
    parser.add_argument("--codebert-path", default="E:\\project\\WYP\\CPDP\\CodeBert")
    parser.add_argument("--local-files-only", type=int, default=1)
    parser.add_argument("--use-amp", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--freeze-encoder", type=int, default=1)
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--max-blocks-per-file", type=int, default=128)
    parser.add_argument("--max-total-blocks", type=int, default=512)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--log-uid", type=int, default=0)
    parser.add_argument("--log-mem", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-template", type=int, default=0)
    parser.add_argument("--skip-missing-src", type=int, default=0)
    parser.add_argument("--debug-ortho", type=int, default=0)
    parser.add_argument("--debug-ortho-every", type=int, default=200)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--log-file", type=int, default=0)
    parser.add_argument("--dump-config-every", type=int, default=0)
    parser.add_argument("--metrics-file", type=int, default=0)
    parser.add_argument("--no-pr-dom", type=int, default=0)
    parser.add_argument("--no-ortho", type=int, default=0)
    parser.add_argument("--no-gcn", type=int, default=0)
    parser.add_argument("--no-struct", type=int, default=0)
    parser.add_argument("--balanced-batch", type=int, default=0)
    parser.add_argument("--projects-per-batch", type=int, default=2)
    parser.add_argument("--lambda-ortho-warmup-epochs", type=int, default=0)
    parser.add_argument("--block-cache-dir", default=None)
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
    block_cache = None
    if args.block_cache_dir:
        cache_dir = Path(args.block_cache_dir)
        index_path = cache_dir / "index.json"
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        block_cache = {uid: str(cache_dir / path) for uid, path in index.items()}

    def _collate(batch):
        return collate_batch(
            batch,
            tokenizer=tokenizer,
            tmax=args.tmax,
            win_size_lines=args.win_size_lines,
            window=args.w,
            max_blocks_per_file=args.max_blocks_per_file,
            skip_template=bool(args.skip_template),
            skip_missing_src=bool(args.skip_missing_src),
            block_cache=block_cache,
        )

    if args.balanced_batch:
        proj_to_indices = {}
        for idx, proj in enumerate(dataset.df["project"].tolist()):
            proj_to_indices.setdefault(proj, []).append(idx)

        class BalancedBatchSampler:
            def __init__(self, proj_indices, batch_size, projects_per_batch, epoch_batches, seed=0):
                self.proj_indices = proj_indices
                self.projects = list(proj_indices.keys())
                self.batch_size = int(batch_size)
                self.projects_per_batch = max(1, projects_per_batch)
                self.epoch_batches = max(1, epoch_batches)
                self.seed = int(seed)
                self.epoch = 0

            def set_epoch(self, epoch: int) -> None:
                self.epoch = int(epoch)

            def __iter__(self):
                rng = random.Random(self.seed + 10007 * self.epoch)
                for _ in range(self.epoch_batches):
                    picks = rng.sample(self.projects, min(self.projects_per_batch, len(self.projects)))
                    per_proj = max(1, self.batch_size // max(1, len(picks)))
                    batch = []
                    for proj in picks:
                        choices = self.proj_indices[proj]
                        batch.extend(rng.choices(choices, k=per_proj))
                    if len(batch) < self.batch_size:
                        all_indices = [i for v in self.proj_indices.values() for i in v]
                        batch.extend(rng.choices(all_indices, k=self.batch_size - len(batch)))
                    yield batch[: self.batch_size]

            def __len__(self):
                return self.epoch_batches

        epoch_batches = len(dataset) // max(args.batch_size, 1)
        loader = DataLoader(
            dataset,
            batch_sampler=BalancedBatchSampler(
                proj_to_indices,
                args.batch_size,
                args.projects_per_batch,
                epoch_batches,
                seed=args.seed,
            ),
            collate_fn=_collate,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=bool(args.shuffle),
            collate_fn=_collate,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_device = torch.device(args.encoder_device)

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

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
    encoder.to(encoder_device)
    type_embed.to(device)
    stats_mlp.to(device)
    model.to(device)

    optim_params = [
        {"params": type_embed.parameters(), "lr": args.lr_other},
        {"params": stats_mlp.parameters(), "lr": args.lr_other},
        {"params": model.parameters(), "lr": args.lr_other},
    ]
    if not args.freeze_encoder:
        optim_params.insert(0, {"params": encoder.parameters(), "lr": args.lr_codebert})
    optimizer = torch.optim.AdamW(optim_params)

    bug_loss_fn = nn.BCEWithLogitsLoss()
    pr_loss_fn = nn.CrossEntropyLoss()
    use_amp = bool(args.use_amp) and device.type == "cuda"
    encoder_amp = use_amp and encoder_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config_path = output_dir / f"run_config_{ts}.json"
    run_config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    metrics_path = output_dir / f"metrics_{ts}.jsonl"
    config_snap_path = output_dir / f"config_snapshots_{ts}.jsonl"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if args.log_file:
        log_file = output_dir / f"train_run_{ts}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_file)

    def emit(msg: str) -> None:
        logger.info("%s", msg)

    def write_metrics(payload: dict) -> None:
        if not args.metrics_file:
            return
        payload["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload))
            f.write("\n")

    best_loss = None

    global_step = 0
    for epoch in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        if args.lambda_ortho_warmup_epochs > 0:
            scale = min((epoch + 1) / args.lambda_ortho_warmup_epochs, 1.0)
            lambda_ortho = args.lambda_ortho * scale
        else:
            lambda_ortho = args.lambda_ortho
        if not args.freeze_encoder:
            encoder.train()
        type_embed.train()
        stats_mlp.train()
        model.train()

        loss_total_acc = []
        loss_bug_acc = []
        loss_bug_sh_acc = []
        loss_bug_file_acc = []
        loss_pr_acc = []
        loss_ortho_acc = []
        alpha_entropy_acc = []
        parse_fail_acc = []
        pr_acc_acc = []
        bug_acc_acc = []
        pos_ratio_acc = []
        zsh_std_acc = []
        zpr_std_acc = []

        optimizer.zero_grad()
        iter_loader = loader
        if args.progress:
            try:
                from tqdm import tqdm

                iter_loader = tqdm(loader, total=len(loader), desc=f"epoch {epoch+1}")
            except Exception:
                iter_loader = loader
        for step, batch in enumerate(iter_loader, start=1):
            step_start = time.perf_counter()
            if batch is None:
                continue
            global_step += 1
            input_ids = batch["flat_input_ids"]
            attention_mask = batch["flat_attention_mask"]
            blk_ptr = batch["blk_ptr"].to(device)
            struct_type_ids = batch["struct_type_ids"].to(device)
            struct_stats = batch["struct_stats"].to(device)
            edge_indices = batch["edge_indices"]
            parse_ok = batch["file_parse_ok"].to(device)
            parse_has_error = batch["file_parse_has_error"].to(device)

            meta = batch["meta"]
            y = torch.tensor(meta["y"], dtype=torch.float, device=device)
            proj_ids = torch.tensor(
                [project_vocab[p] for p in meta["project"]],
                dtype=torch.long,
                device=device,
            )
            pos_ratio = float(y.mean().item()) if y.numel() > 0 else 0.0
            proj_unique = int(torch.unique(proj_ids).numel()) if proj_ids.numel() > 0 else 0

            if batch["h_sem"] is not None:
                h_sem = batch["h_sem"].to(device)
            else:
                input_ids = input_ids.to(encoder_device)
                attention_mask = attention_mask.to(encoder_device)
                if args.freeze_encoder:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=encoder_amp):
                            h_sem = encoder(input_ids, attention_mask)
                else:
                    with torch.cuda.amp.autocast(enabled=encoder_amp):
                        h_sem = encoder(input_ids, attention_mask)
                if encoder_device != device:
                    h_sem = h_sem.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.no_struct:
                    type_emb = torch.zeros((struct_type_ids.size(0), args.d_t), device=device)
                    stats_emb = torch.zeros((struct_stats.size(0), args.d_p), device=device)
                else:
                    type_emb = type_embed(struct_type_ids)
                    stats_emb = stats_mlp(struct_stats)
                e_struct = torch.cat([type_emb, stats_emb], dim=1)

                outputs = model(
                    h_sem,
                    e_struct,
                    blk_ptr,
                    edge_indices=edge_indices,
                    use_gcn=not args.no_gcn,
                )

                loss_bug_sh = bug_loss_fn(outputs["logit_bug_sh"].squeeze(-1), y)
                loss_bug_file = bug_loss_fn(outputs["logit_bug_file"].squeeze(-1), y)
                loss_bug = loss_bug_sh + args.beta_bug_file * loss_bug_file
                loss_pr = pr_loss_fn(outputs["logits_pr_dom"], proj_ids)
                loss_ortho = outputs["loss_ortho"]
                if args.no_pr_dom:
                    loss_pr = torch.zeros((), device=device)
                if args.no_ortho:
                    loss_ortho = torch.zeros((), device=device)
                loss_total = loss_bug + args.lambda_pr * loss_pr + lambda_ortho * loss_ortho
                loss_scaled = loss_total / max(args.grad_accum_steps, 1)

            if args.debug_ortho and step % max(args.debug_ortho_every, 1) == 0:
                # Skip grad inspection if ortho is disabled / detached
                grad_norm = 0.0
                if loss_ortho.requires_grad and loss_ortho.grad_fn is not None:
                    params = list(model.f_sh.parameters()) + list(model.f_pr.parameters())
                    grads = torch.autograd.grad(loss_ortho, params, retain_graph=True, allow_unused=True)
                    grad_norms = [g.norm().item() for g in grads if g is not None]
                    grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
                with torch.no_grad():
                    z_sh = outputs["Z_sh"]
                    z_pr = outputs["Z_pr"]
                    cos_vals = torch.nn.functional.cosine_similarity(z_sh, z_pr, dim=1).float()
                    cos_mean = float(cos_vals.mean().item()) if cos_vals.numel() > 0 else 0.0
                    cos_p50 = float(torch.quantile(cos_vals, 0.5).item()) if cos_vals.numel() > 0 else 0.0
                    cos_p90 = float(torch.quantile(cos_vals, 0.9).item()) if cos_vals.numel() > 0 else 0.0
                    if cos_vals.numel() > 0:
                        z_sh_norm = torch.nn.functional.normalize(z_sh.float(), p=2, dim=1)
                        z_pr_norm = torch.nn.functional.normalize(z_pr.float(), p=2, dim=1)
                        batch_size = max(z_sh_norm.size(0), 1)
                        c = (z_sh_norm.transpose(0, 1) @ z_pr_norm) / batch_size
                        c_fro = float(torch.norm(c, p="fro").item())
                        c_max = float(c.abs().max().item()) if c.numel() > 0 else 0.0
                        zsh_std = float(z_sh.std(dim=0).mean().item())
                        zpr_std = float(z_pr.std(dim=0).mean().item())
                        pr_acc = float((outputs["logits_pr_dom"].argmax(-1) == proj_ids).float().mean().item())
                    else:
                        c_fro = 0.0
                        c_max = 0.0
                        zsh_std = 0.0
                        zpr_std = 0.0
                        pr_acc = 0.0
                emit(
                    f"debug_ortho step={step} loss_ortho={loss_ortho.item():.6f} "
                    f"grad_norm_f_sh_pr={grad_norm:.6e} cos_mean={cos_mean:.4f} "
                    f"cos_p50={cos_p50:.4f} cos_p90={cos_p90:.4f} "
                    f"C_fro={c_fro:.4f} C_maxabs={c_max:.4f} "
                    f"zsh_std={zsh_std:.4f} zpr_std={zpr_std:.4f} pr_acc={pr_acc:.4f} "
                    f"proj_unique={proj_unique}"
                )

            with torch.no_grad():
                pr_acc = float((outputs["logits_pr_dom"].argmax(-1) == proj_ids).float().mean().item())
                bug_pred = (torch.sigmoid(outputs["logit_bug_sh"].squeeze(-1)) > 0.5).float()
                bug_acc = float((bug_pred == y).float().mean().item()) if y.numel() > 0 else 0.0
                zsh_std = float(outputs["Z_sh"].std(dim=0).mean().item()) if outputs["Z_sh"].numel() > 0 else 0.0
                zpr_std = float(outputs["Z_pr"].std(dim=0).mean().item()) if outputs["Z_pr"].numel() > 0 else 0.0

            scaler.scale(loss_scaled).backward()

            if step % max(args.grad_accum_steps, 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_total_acc.append(loss_total.item())
            loss_bug_acc.append(loss_bug.item())
            loss_bug_sh_acc.append(loss_bug_sh.item())
            loss_bug_file_acc.append(loss_bug_file.item())
            loss_pr_acc.append(loss_pr.item())
            loss_ortho_acc.append(loss_ortho.item())
            alpha_entropy_acc.append(alpha_entropy(outputs["alpha_values"], blk_ptr, parse_ok))
            parse_fail_acc.append(1.0 - float(parse_ok.float().mean().item()))
            pr_acc_acc.append(pr_acc)
            bug_acc_acc.append(bug_acc)
            pos_ratio_acc.append(pos_ratio)
            zsh_std_acc.append(zsh_std)
            zpr_std_acc.append(zpr_std)
            parse_error_acc = float(parse_has_error.float().mean().item())
            if step % max(args.log_every, 1) == 0:
                step_time = time.perf_counter() - step_start
                total_blocks = int(blk_ptr[-1].item())
                k_max = int((blk_ptr[1:] - blk_ptr[:-1]).max().item())
                parse_ok_ratio = float(parse_ok.float().mean().item())
                seq_len = int(input_ids.shape[1]) if input_ids is not None else 0
                tok_counts = attention_mask.sum(dim=1).float() if attention_mask is not None else torch.zeros(0)
                tok_max = int(tok_counts.max().item()) if tok_counts.numel() > 0 else 0
                tok_p95 = int(torch.quantile(tok_counts, 0.95).item()) if tok_counts.numel() > 0 else 0
                uid0 = meta["uid"][0] if args.log_uid and "uid" in meta else None
                msg = (
                    f"batch {step} total_blocks={total_blocks} k_max={k_max} "
                    f"seq_len={seq_len} tok_max={tok_max} tok_p95={tok_p95} "
                    f"parse_ok={parse_ok_ratio:.3f} parse_err={parse_error_acc:.3f} "
                    f"proj_unique={proj_unique}"
                )
                if uid0:
                    msg = f"{msg} uid={uid0}"
                msg = f"{msg} step_time={step_time:.3f}s"
                if args.log_mem and device.type == "cuda":
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    resv = torch.cuda.memory_reserved() / 1024**3
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    msg = (
                        f"{msg} [cuda] allocated={alloc:.2f}G reserved={resv:.2f}G max_alloc={peak:.2f}G"
                    )
                emit(msg)
                write_metrics(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": float(loss_total.item()),
                        "loss_ortho": float(loss_ortho.item()),
                        "pr_acc": float(pr_acc),
                        "bug_acc": float(bug_acc),
                        "step_time": float(step_time),
                        "proj_unique": int(proj_unique),
                    }
                )
                if args.progress and hasattr(iter_loader, "set_postfix"):
                    iter_loader.set_postfix(
                        loss=f"{loss_total.item():.4f}",
                        ortho=f"{float(loss_ortho.item()):.4f}",
                        pr_acc=f"{float(pr_acc):.3f}",
                        t=f"{step_time:.2f}s",
                    )
                if args.dump_config_every > 0 and global_step % args.dump_config_every == 0:
                    snap = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "step": global_step,
                        "epoch": epoch + 1,
                        "args": vars(args),
                    }
                    with config_snap_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(snap))
                        f.write("\n")
            total_blocks = int(blk_ptr[-1].item())
            if args.max_total_blocks > 0 and total_blocks > args.max_total_blocks:
                if args.log_uid and "uid" in meta:
                    print(f"skip batch {step} uid={meta['uid'][0]} total_blocks={total_blocks}")
                else:
                    print(f"skip batch {step} total_blocks={total_blocks}")
                continue

        epoch_time = time.perf_counter() - epoch_start
        epoch_log = {
            "epoch": epoch + 1,
            "loss_total": float(np.mean(loss_total_acc)),
            "loss_bug": float(np.mean(loss_bug_acc)),
            "loss_bug_sh": float(np.mean(loss_bug_sh_acc)),
            "loss_bug_file": float(np.mean(loss_bug_file_acc)),
            "loss_pr_dom": float(np.mean(loss_pr_acc)),
            "loss_ortho": float(np.mean(loss_ortho_acc)),
            "alpha_entropy_mean": float(np.mean(alpha_entropy_acc)),
            "ortho_fro_batch": float(np.mean(loss_ortho_acc)),
            "parse_fail_ratio_batch": float(np.mean(parse_fail_acc)),
            "parse_error_ratio_batch": parse_error_acc,
            "pr_acc": float(np.mean(pr_acc_acc)) if pr_acc_acc else 0.0,
            "bug_acc": float(np.mean(bug_acc_acc)) if bug_acc_acc else 0.0,
            "pos_ratio": float(np.mean(pos_ratio_acc)) if pos_ratio_acc else 0.0,
            "zsh_std": float(np.mean(zsh_std_acc)) if zsh_std_acc else 0.0,
            "zpr_std": float(np.mean(zpr_std_acc)) if zpr_std_acc else 0.0,
            "epoch_time_sec": float(epoch_time),
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

        emit(f"Epoch {epoch+1}/{args.epochs} loss={epoch_log['loss_total']:.4f}")



if __name__ == "__main__":
    main()
