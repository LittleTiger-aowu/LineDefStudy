"""Graph-CodeBERT-CPDP model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from transformers import AutoModel

from .amsoftmax import AMSoftmax
from .disentangle import DisentangleHeads
from .domain_discriminator import DomainDiscriminator
from .grl import grl


def build_line_graph(line_ptr: torch.Tensor, window: int = 2) -> torch.Tensor:
    # line_ptr: [N+1] prefix sum of lines per file.
    edges = []
    for i in range(len(line_ptr) - 1):
        start = int(line_ptr[i])
        end = int(line_ptr[i + 1])
        for idx in range(start, end):
            for offset in range(1, window + 1):
                j = idx + offset
                if j < end:
                    edges.append((idx, j))
                    edges.append((j, idx))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def attention_pool(h_node: torch.Tensor, line_ptr: torch.Tensor, attn_fc: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    # h_node: [Total_Lines, H]
    # line_ptr: [N+1]
    att_logits = attn_fc(h_node).squeeze(-1)
    att_weights = torch.zeros_like(att_logits)
    pooled = []
    for i in range(len(line_ptr) - 1):
        start = int(line_ptr[i])
        end = int(line_ptr[i + 1])
        if start == end:
            pooled.append(torch.zeros(h_node.size(1), device=h_node.device))
            continue
        weights = torch.softmax(att_logits[start:end], dim=0)
        att_weights[start:end] = weights
        pooled.append((weights.unsqueeze(-1) * h_node[start:end]).sum(dim=0))
    h_file = torch.stack(pooled, dim=0)
    return h_file, att_weights


def orthogonality_loss(z_shared: torch.Tensor, z_private: torch.Tensor) -> torch.Tensor:
    z_s = torch.nn.functional.normalize(z_shared, p=2, dim=1)
    z_p = torch.nn.functional.normalize(z_private, p=2, dim=1)
    corr = torch.matmul(z_s.t(), z_p)
    return torch.norm(corr, p="fro")


class GraphCodeBertCPDP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        shared_dim: int = 128,
        private_dim: int = 128,
        enable_dann: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.gcn1 = GCNConv(768, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.disentangle = DisentangleHeads(hidden_dim, shared_dim, private_dim)
        self.classifier = AMSoftmax(shared_dim, num_classes=2)
        self.enable_dann = enable_dann
        self.domain_disc = DomainDiscriminator(shared_dim, hidden_dim=shared_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ptr: torch.Tensor,
        file_labels: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # input_ids: [Total_Lines, T_max]
        # attention_mask: [Total_Lines, T_max]
        # line_ptr: [N+1]
        assert input_ids.dim() == 2, "input_ids must be [Total_Lines, T_max]"
        assert attention_mask.shape == input_ids.shape, "attention_mask shape mismatch"
        assert line_ptr.dim() == 1, "line_ptr must be [N+1]"

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        line_emb = outputs.last_hidden_state[:, 0, :]  # [Total_Lines, 768]

        edge_index = build_line_graph(line_ptr)
        edge_index = edge_index.to(input_ids.device)

        h_node = torch.relu(self.gcn1(line_emb, edge_index))
        h_node = self.gcn2(h_node, edge_index)  # [Total_Lines, H]

        h_file, att_weights = attention_pool(h_node, line_ptr, self.attn_fc)
        z_shared, z_private = self.disentangle(h_file)

        logits_cls = self.classifier(z_shared, file_labels)
        losses = {
            "ortho": orthogonality_loss(z_shared, z_private),
        }

        domain_logits = None
        if self.enable_dann:
            domain_logits = self.domain_disc(grl(z_shared, grl_lambda))

        return {
            "logits_cls": logits_cls,
            "h_file": h_file,
            "z_shared": z_shared,
            "z_private": z_private,
            "att_weights": att_weights,
            "domain_logits": domain_logits,
            "losses": losses,
        }
