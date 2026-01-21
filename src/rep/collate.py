from __future__ import annotations

from typing import Any, List, Optional

import torch

from .graph import build_sliding_window_edges
from .struct_prior import build_struct_features
from .tpsm import Block, extract_blocks


def collate_batch(
    samples: List[Any],
    tokenizer,
    tmax: int = 128,
    win_size_lines: int = 20,
    lang: str = "java",
    window: int = 2,
    include_block_text: bool = False,
    max_blocks_per_file: Optional[int] = None,
) -> dict:
    blocks_per_file: List[List[Block]] = []
    blk_ptr = [0]
    file_parse_ok = []
    file_parse_has_error = []
    meta = {
        "uid": [],
        "project": [],
        "version": [],
        "file_path": [],
        "y": [],
        "sha1": [],
        "spans": [],
        "type_ids": [],
        "type_strs": [],
        "num_blocks": [],
        "block_texts": [] if include_block_text else None,
    }
    for sample in samples:
        blocks = extract_blocks(sample.src, lang=lang, win_size_lines=win_size_lines)
        if max_blocks_per_file is not None:
            blocks = _cap_blocks(blocks, max_blocks_per_file)
        blocks_per_file.append(blocks)
        blk_ptr.append(blk_ptr[-1] + len(blocks))
        file_parse_ok.append(bool(blocks[0].parse_ok))
        file_parse_has_error.append(bool(blocks[0].parse_has_error))
        meta["uid"].append(sample.uid)
        meta["project"].append(sample.project)
        meta["version"].append(sample.version)
        meta["file_path"].append(sample.file_path)
        meta["y"].append(sample.y)
        meta["sha1"].append(sample.sha1)
        meta["spans"].append([list(block.span) for block in blocks])
        meta["type_ids"].append([block.type_id for block in blocks])
        meta["type_strs"].append([block.type_str for block in blocks])
        meta["num_blocks"].append(len(blocks))
        if include_block_text:
            meta["block_texts"].append([block.text for block in blocks])

    all_blocks = [block for blocks in blocks_per_file for block in blocks]
    assert blk_ptr[-1] == len(all_blocks), "blk_ptr must end at TotalBlocks"

    block_texts = [block.text for block in all_blocks]
    tokenized = tokenizer(
        block_texts,
        padding="max_length",
        truncation=True,
        max_length=tmax,
        return_tensors="pt",
    )

    type_ids, stats_vecs = build_struct_features(all_blocks)
    struct_type_ids = torch.tensor(type_ids, dtype=torch.long)
    struct_stats = torch.tensor(stats_vecs, dtype=torch.float)

    edge_indices = [build_sliding_window_edges(len(blocks), window=window) for blocks in blocks_per_file]

    return {
        "flat_input_ids": tokenized["input_ids"],
        "flat_attention_mask": tokenized["attention_mask"],
        "blk_ptr": torch.tensor(blk_ptr, dtype=torch.long),
        "struct_type_ids": struct_type_ids,
        "struct_stats": struct_stats,
        "edge_indices": edge_indices,
        "file_parse_ok": torch.tensor(file_parse_ok, dtype=torch.bool),
        "file_parse_has_error": torch.tensor(file_parse_has_error, dtype=torch.bool),
        "meta": meta,
    }


def _cap_blocks(blocks: List[Block], max_blocks: int) -> List[Block]:
    if max_blocks <= 0 or len(blocks) <= max_blocks:
        return blocks
    anchors = [b for b in blocks if b.stats.get("anchor_flag", 0) == 1]
    windows = [b for b in blocks if b.stats.get("anchor_flag", 0) != 1]
    anchors_sorted = sorted(anchors, key=lambda b: b.span[0])
    if len(anchors_sorted) >= max_blocks:
        return anchors_sorted[:max_blocks]
    remaining = max_blocks - len(anchors_sorted)
    if not windows:
        return anchors_sorted
    step = len(windows) / remaining
    pick = []
    seen = set()
    for i in range(remaining):
        idx = int(i * step)
        if idx >= len(windows):
            idx = len(windows) - 1
        if idx in seen:
            continue
        pick.append(windows[idx])
        seen.add(idx)
    if len(pick) < remaining:
        for i, b in enumerate(windows):
            if i in seen:
                continue
            pick.append(b)
            seen.add(i)
            if len(pick) >= remaining:
                break
    return sorted(anchors_sorted + pick, key=lambda b: b.span[0])
