from __future__ import annotations

from peft import LoraConfig, TaskType


def build_lora_config(
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> LoraConfig:
    """LoRA config targeting Qwen2.5 attention + MLP projections."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
