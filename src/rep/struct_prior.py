from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .tpsm import TYPE_TO_ID, Block


class TypeEmbedding(nn.Module):
    def __init__(self, num_types: int, d_t: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_types, d_t)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(type_ids)


class StatsMLP(nn.Module):
    def __init__(self, in_dim: int, d_p: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d_p), nn.ReLU(), nn.Linear(d_p, d_p))

    def forward(self, stats_vec: torch.Tensor) -> torch.Tensor:
        return self.net(stats_vec)


def build_struct_features(blocks: List[Block]) -> Tuple[List[int], List[List[float]]]:
    type_ids: List[int] = []
    stats_vecs: List[List[float]] = []
    for block in blocks:
        type_ids.append(block.type_id)
        span_len = float(block.stats.get("span_len", block.span[1] - block.span[0] + 1))
        is_fallback = float(block.stats.get("is_fallback", 0))
        anchor_flag = float(block.stats.get("anchor_flag", 0))
        stats_vecs.append([span_len, is_fallback, anchor_flag])
    return type_ids, stats_vecs


def num_block_types() -> int:
    return max(TYPE_TO_ID.values()) + 1

