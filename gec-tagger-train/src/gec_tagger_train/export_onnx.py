from __future__ import annotations

from pathlib import Path

import torch

from gec_tagger_train.model import DebertaTagger


def export_tagger_to_onnx(
    *,
    model: DebertaTagger,
    out_path: Path,
    max_length: int = 128,
    opset: int = 17,
) -> Path:
    model.eval()
    dummy_input_ids = torch.zeros((1, max_length), dtype=torch.long)
    dummy_mask = torch.ones((1, max_length), dtype=torch.long)

    class _Wrapper(torch.nn.Module):
        def __init__(self, m: DebertaTagger) -> None:
            super().__init__()
            self.m = m

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            result: dict[str, torch.Tensor] = self.m(
                input_ids=input_ids, attention_mask=attention_mask
            )
            return result["logits"]

    wrapper = _Wrapper(model)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_mask),
        str(out_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        dynamo=False,
    )
    return out_path
