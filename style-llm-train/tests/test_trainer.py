from pathlib import Path

from style_llm_train.config import TrainConfig
from style_llm_train.trainer import run_sft_training


def test_smoke_sft_runs_one_step(fixtures_dir: Path, tmp_out: Path) -> None:
    cfg = TrainConfig(
        jsonl=fixtures_dir / "tiny_sft.jsonl",
        output_dir=tmp_out / "ckpt",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_epochs=1,
        max_steps=1,
        max_seq_length=128,
        warmup_ratio=0.0,
        fp16=False,
        bf16=False,
    )
    metrics = run_sft_training(cfg=cfg)
    assert "train_runtime" in metrics
    assert (tmp_out / "ckpt").is_dir()
