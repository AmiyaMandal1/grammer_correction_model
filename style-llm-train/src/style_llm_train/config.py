from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    jsonl: Path
    output_dir: Path
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 1
    max_steps: int = -1
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
