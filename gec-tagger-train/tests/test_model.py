import torch

from gec_tagger_train.model import DebertaTagger


def test_forward_returns_logits_and_loss() -> None:
    model = DebertaTagger(num_tags=10, pretrained="microsoft/deberta-v3-base")
    input_ids = torch.tensor([[101, 200, 300, 102]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[0, 2, 3, 0]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert "logits" in out
    assert "loss" in out
    assert out["logits"].shape == (1, 4, 10)
    assert out["loss"].dim() == 0


def test_forward_without_labels_returns_logits_only() -> None:
    model = DebertaTagger(num_tags=10, pretrained="microsoft/deberta-v3-base")
    input_ids = torch.tensor([[101, 200, 102]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert "logits" in out
    assert "loss" not in out
