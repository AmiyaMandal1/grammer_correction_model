from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("train")
def train(
    jsonl: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    base_model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct"),
    max_steps: int = typer.Option(-1),
    num_epochs: int = typer.Option(1),
    per_device_batch_size: int = typer.Option(1),
    gradient_accumulation_steps: int = typer.Option(8),
    learning_rate: float = typer.Option(5e-5),
    max_seq_length: int = typer.Option(2048),
    warmup_ratio: float = typer.Option(0.1),
) -> None:
    """Fine-tune the base model with LoRA adapters."""
    from style_llm_train.config import TrainConfig
    from style_llm_train.trainer import run_sft_training

    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        jsonl=jsonl,
        output_dir=out,
        base_model=base_model,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        warmup_ratio=warmup_ratio,
    )
    metrics = run_sft_training(cfg=cfg)
    typer.echo(f"trained: {metrics}")


@app.command("merge")
def merge(
    adapter: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    base_model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct"),
) -> None:
    """Merge a LoRA adapter into the base model and save the full HF checkpoint."""
    from style_llm_train.merge import merge_adapter

    merge_adapter(adapter_dir=adapter, out_dir=out, base_model=base_model)
    typer.echo(f"merged checkpoint written to {out}")


@app.command("quantize")
def quantize(
    hf_checkpoint: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    llama_cpp_dir: Path = typer.Option(..., exists=True),
    quant: str = typer.Option("Q4_K_M"),
) -> None:
    """Convert an HF checkpoint to GGUF and quantize."""
    from style_llm_train.gguf_export import convert_and_quantize

    convert_and_quantize(
        hf_checkpoint=hf_checkpoint, out=out, llama_cpp_dir=llama_cpp_dir, quant=quant
    )
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
