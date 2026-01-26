"""Gradient reversal layer."""
from __future__ import annotations

import torch


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grl(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GRL.apply(x, lambda_)
