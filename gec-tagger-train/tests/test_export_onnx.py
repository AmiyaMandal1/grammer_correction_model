from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer

from gec_tagger_train.export_onnx import export_tagger_to_onnx
from gec_tagger_train.model import DebertaTagger


def test_export_produces_loadable_onnx(tmp_out: Path) -> None:
    model = DebertaTagger(num_tags=8)
    out = tmp_out / "tagger.onnx"
    export_tagger_to_onnx(model=model, out_path=out, max_length=16, opset=17)
    assert out.exists()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    enc = tok(
        "hello world", return_tensors="np", padding="max_length", max_length=16, truncation=True
    )
    out_np = sess.run(
        None,
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
    )
    assert out_np[0].shape == (1, 16, 8)
