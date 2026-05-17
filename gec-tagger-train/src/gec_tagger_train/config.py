from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class StageConfig:
    name: str
    train_jsonl: Path
    output_dir: Path
    max_length: int = 128
    per_device_batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
