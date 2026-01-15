"""Feature disentanglement heads."""
from __future__ import annotations

import torch
from torch import nn


class DisentangleHeads(nn.Module):
    def __init__(self, in_dim: int, shared_dim: int, private_dim: int) -> None:
        super().__init__()
        self.shared_head = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.private_head = nn.Sequential(
            nn.Linear(in_dim, private_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_shared = self.shared_head(x)
        z_private = self.private_head(x)
        return z_shared, z_private
