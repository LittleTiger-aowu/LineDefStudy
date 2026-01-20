from __future__ import annotations

from typing import List

import torch


def build_sliding_window_edges(num_nodes: int, window: int) -> torch.Tensor:
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    edges = []
    for i in range(num_nodes):
        start = max(0, i - window)
        end = min(num_nodes - 1, i + window)
        for j in range(start, end + 1):
            if i == j:
                continue
            edges.append((i, j))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def batch_merge_edges(edge_indices: List[torch.Tensor], num_nodes: List[int]) -> torch.Tensor:
    if not edge_indices:
        return torch.zeros((2, 0), dtype=torch.long)
    merged = []
    offset = 0
    for edge_index, n_nodes in zip(edge_indices, num_nodes):
        if edge_index.numel() > 0:
            merged.append(edge_index + offset)
        offset += n_nodes
    if not merged:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.cat(merged, dim=1)

