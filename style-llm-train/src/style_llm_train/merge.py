from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(
    *,
    adapter_dir: Path,
    out_dir: Path,
    base_model: str,
) -> Path:
    """Load base + LoRA adapter, merge weights, save full HF checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_model)
    peft = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
