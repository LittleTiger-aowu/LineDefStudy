from __future__ import annotations

import torch
from torch import nn


def orthogonality_loss(z_sh: torch.Tensor, z_pr: torch.Tensor) -> torch.Tensor:
    # z_sh: [B, d_sh], z_pr: [B, d_pr]
    z_sh_norm = torch.nn.functional.normalize(z_sh, p=2, dim=1)
    z_pr_norm = torch.nn.functional.normalize(z_pr, p=2, dim=1)
    c = torch.matmul(z_sh_norm.transpose(0, 1), z_pr_norm)
    return torch.norm(c, p="fro")


def pr_domain_loss(logits_pr_dom: torch.Tensor, project_id: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits_pr_dom, project_id)


def bug_loss(logit_bug: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(logit_bug.squeeze(-1), y.float())

