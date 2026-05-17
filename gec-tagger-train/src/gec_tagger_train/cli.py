from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from transformers import AutoTokenizer

from gec_tagger_train.config import StageConfig
from gec_tagger_train.dataset import GECTaggerDataset
from gec_tagger_train.tag_vocab import TagVocab, build_tag_vocab

app = typer.Typer(no_args_is_help=True)


@app.command("build-vocab")
def build_vocab(
    jsonl: Path = typer.Option(..., exists=True, help="gec_tagger.jsonl input"),
    out: Path = typer.Option(..., help="Output tags.json path"),
    min_count: int = typer.Option(5, help="Minimum frequency to retain a tag"),
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab = build_tag_vocab(jsonl=jsonl, min_count=min_count, out=out)
    typer.echo(f"wrote {len(vocab)} tags to {out}")


@app.command("train")
def train(
    stage: int = typer.Option(..., min=1, max=3),
    jsonl: Path = typer.Option(..., exists=True),
    tags: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    max_steps: int = typer.Option(-1),
    num_epochs: int = typer.Option(1),
    per_device_batch_size: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-5),
    max_length: int = typer.Option(128),
) -> None:
    from gec_tagger_train.trainer import run_training

    out.mkdir(parents=True, exist_ok=True)
    vocab = TagVocab.load(tags)
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    ds = GECTaggerDataset(jsonl=jsonl, tokenizer=tok, vocab=vocab, max_length=max_length)
    cfg = StageConfig(
        name=f"stage{stage}",
        train_jsonl=jsonl,
        output_dir=out,
        max_length=max_length,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        warmup_ratio=0.1 if max_steps < 0 else 0.0,
    )
    metrics = run_training(cfg=cfg, dataset=ds, vocab=vocab, tokenizer=tok)
    typer.echo(f"trained stage {stage}: {metrics}")


@app.command("export")
def export(
    checkpoint: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    max_length: int = typer.Option(128),
    opset: int = typer.Option(17),
) -> None:
    from gec_tagger_train.export_onnx import export_tagger_to_onnx
    from gec_tagger_train.model import DebertaTagger

    state = _load_state_dict(checkpoint)
    num_tags = int(state["classifier.weight"].shape[0])
    model = DebertaTagger(num_tags=num_tags)
    model.load_state_dict(state, strict=False)
    export_tagger_to_onnx(model=model, out_path=out, max_length=max_length, opset=opset)
    typer.echo(f"exported ONNX to {out}")


def _load_state_dict(checkpoint: Path) -> dict[str, Any]:
    import torch

    bin_path = checkpoint / "pytorch_model.bin"
    safe_path = checkpoint / "model.safetensors"
    if safe_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safe_path))
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)  # type: ignore[no-any-return]
    raise FileNotFoundError(f"no model weights found under {checkpoint}")


if __name__ == "__main__":
    app()
