from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from .graph import batch_merge_edges, build_sliding_window_edges
from .losses import orthogonality_loss


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, dim)
        self.context = nn.Parameter(torch.zeros(dim))

    def forward(self, h_blk: torch.Tensor, blk_ptr: torch.Tensor) -> torch.Tensor:
        # h_blk: [TotalBlocks, d_h], blk_ptr: [B+1]
        assert h_blk.dim() == 2, "h_blk must be 2D"
        assert blk_ptr.dim() == 1, "blk_ptr must be 1D"
        assert int(blk_ptr[0].item()) == 0, "blk_ptr must start at 0"
        assert int(blk_ptr[-1].item()) == h_blk.size(0), "blk_ptr must end at TotalBlocks"
        scores = torch.tanh(self.score(h_blk))
        scores = torch.matmul(scores, self.context)
        alpha = torch.zeros_like(scores)
        h_file = []
        for i in range(len(blk_ptr) - 1):
            start = int(blk_ptr[i].item())
            end = int(blk_ptr[i + 1].item())
            segment = scores[start:end]
            weights = torch.softmax(segment, dim=0) if segment.numel() > 0 else segment
            alpha[start:end] = weights
            h_file.append((weights.unsqueeze(-1) * h_blk[start:end]).sum(dim=0))
        h_file = torch.stack(h_file, dim=0) if h_file else torch.zeros((0, h_blk.size(1)), device=h_blk.device)
        return h_file, alpha


class RepresentationModel(nn.Module):
    def __init__(
        self,
        d_h: int,
        d_sh: int,
        d_pr: int,
        num_projects: int,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.fusion = nn.Linear(input_dim, d_h)
        self.gcn1 = GCNConv(d_h, d_h)
        self.gcn2 = GCNConv(d_h, d_h)
        self.att_pool = AttentionPool(d_h)
        self.f_sh = nn.Sequential(nn.Linear(d_h, d_sh), nn.ReLU(), nn.Linear(d_sh, d_sh))
        self.f_pr = nn.Sequential(nn.Linear(d_h, d_pr), nn.ReLU(), nn.Linear(d_pr, d_pr))
        self.d_pr = nn.Linear(d_pr, num_projects)
        self.bug_head = nn.Linear(d_h, 1)

    def forward(
        self,
        h_sem: torch.Tensor,
        e_struct: torch.Tensor,
        blk_ptr: torch.Tensor,
        edge_indices: Optional[list[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # h_sem: [TotalBlocks, 768], e_struct: [TotalBlocks, d_s]
        assert h_sem.dim() == 2 and e_struct.dim() == 2, "inputs must be 2D"
        assert h_sem.size(0) == e_struct.size(0), "h_sem and e_struct must align"
        h0 = torch.cat([h_sem, e_struct], dim=1)
        h0 = self.fusion(h0)
        if edge_indices is None:
            edge_indices = []
            for i in range(len(blk_ptr) - 1):
                k = int(blk_ptr[i + 1] - blk_ptr[i])
                edge_indices.append(build_sliding_window_edges(k, window=2))
        num_nodes = [int(blk_ptr[i + 1] - blk_ptr[i]) for i in range(len(blk_ptr) - 1)]
        edge_index = batch_merge_edges(edge_indices, num_nodes).to(h0.device)
        h1 = self.gcn1(h0, edge_index)
        h1 = torch.relu(h1)
        h_blk = self.gcn2(h1, edge_index)
        h_file, alpha = self.att_pool(h_blk, blk_ptr)
        z_sh = self.f_sh(h_file)
        z_pr = self.f_pr(h_file)
        logits_pr_dom = self.d_pr(z_pr)
        logit_bug = self.bug_head(h_file)
        loss_ortho = orthogonality_loss(z_sh, z_pr)
        return {
            "H_blk": h_blk,
            "alpha_values": alpha,
            "H_file": h_file,
            "Z_sh": z_sh,
            "Z_pr": z_pr,
            "logits_pr_dom": logits_pr_dom,
            "logit_bug": logit_bug,
            "loss_ortho": loss_ortho,
        }
