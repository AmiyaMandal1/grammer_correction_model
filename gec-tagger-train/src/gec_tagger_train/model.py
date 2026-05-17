from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class DebertaTagger(nn.Module):
    def __init__(
        self,
        *,
        num_tags: int,
        pretrained: str = "microsoft/deberta-v3-base",
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained)
        self.encoder = AutoModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=pad_id)
        self.num_tags = num_tags
        self.pad_id = pad_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()
        logits = self.classifier(self.dropout(hidden))
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.num_tags),
                labels.view(-1),
            )
            out["loss"] = loss
        return out
