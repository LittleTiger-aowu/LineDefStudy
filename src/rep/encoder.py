from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class CodeBertBlockEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "E:\\project\\WYP\\CPDP\\CodeBert",
        pooling: Literal["cls", "mean"] = "cls",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            masked = hidden * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            return masked.sum(dim=1) / denom
        return hidden[:, 0, :]


def build_tokenizer(model_name: str = "E:\\project\\WYP\\CPDP\\CodeBert", local_files_only: bool = True):
    return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
