"""AM-Softmax classifier head."""
from __future__ import annotations

import torch
from torch import nn


class AMSoftmax(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, margin: float = 0.35, scale: float = 30.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [N, D]
        # labels: [N]
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        logits = torch.matmul(features, weight.t())
        if labels is not None:
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            logits = logits - one_hot * self.margin
        logits = logits * self.scale
        return logits
