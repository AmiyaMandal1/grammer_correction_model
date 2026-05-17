from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

from style_llm_train.config import TrainConfig
from style_llm_train.dataset import load_sft_records
from style_llm_train.lora import build_lora_config


def _format_chatml(record: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    text = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def run_sft_training(*, cfg: TrainConfig) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = list(load_sft_records(cfg.jsonl))
    formatted = [_format_chatml(r, tokenizer) for r in records]
    ds = Dataset.from_list(formatted)

    # Apple Silicon MPS + fp16 + Qwen RMSNorm/rotary embeds = kIOGPUCommandBuffer
    # errors that put the Metal queue into "ignore" mode mid-training. Load the
    # base in fp32 on MPS and skip gradient checkpointing (the other half of the
    # crash trigger). bf16 is also unsafe under the same combo on M2 family.
    import platform

    on_mps = platform.system() == "Darwin" and platform.machine() == "arm64"
    dtype = torch.float32 if on_mps else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    if not on_mps:
        try:
            base.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
        except Exception:
            pass

    model = get_peft_model(base, build_lora_config())

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        max_length=cfg.max_seq_length,  # TRL >=1.0 uses max_length
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        report_to=[],
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,  # type: ignore[arg-type]
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    out = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return dict(out.metrics)
