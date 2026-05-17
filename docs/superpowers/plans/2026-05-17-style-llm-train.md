# Style LLM Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package `style-llm-train/` that LoRA-fine-tunes `Qwen2.5-3B-Instruct` on the `style_sft.jsonl` produced by `data-pipeline/`, merges the LoRA adapters back into the base model, and emits a GGUF file (target quantization Q4_K_M) ready for the Rust `gec-engine` runtime via `llama.cpp`.

**Architecture:** A SFT dataset reads ChatML records from `style_sft.jsonl` and applies Qwen2.5's chat template via the tokenizer. The trainer wraps TRL's `SFTTrainer` with PEFT LoRA adapters targeting `q/k/v/o + gate/up/down` projections, r=16, alpha=32, dropout 0.05. Training runs in fp16 on the base model (full QLoRA via bitsandbytes is CUDA-only; on Apple Silicon we keep the base in fp16 and only the adapters in fp32 — a working compromise documented in the README). After training, `peft.PeftModel.merge_and_unload` produces a merged HF checkpoint, which is then handed to a `llama.cpp` `convert_hf_to_gguf.py` subprocess plus a `llama-quantize` step. A typer CLI exposes `train`, `merge`, `quantize`, and `train-and-export` (the convenience wrapper).

**Tech Stack:** Python 3.11, `uv`, `torch>=2.4` (MPS backend), `transformers>=4.44`, `tokenizers`, `peft>=0.12`, `trl>=0.10`, `accelerate>=0.33`, `datasets>=2.20`, `evaluate>=0.4`, `sacrebleu>=2.4`, `bert-score>=0.3.13`, `typer>=0.12`, `pytest`, `ruff`, `mypy`.

External tools (operator must clone separately):
- `llama.cpp` repo (for `convert_hf_to_gguf.py` + `llama-quantize`). README documents the path.

---

## Layout

```
style-llm-train/
├── pyproject.toml
├── README.md
├── .python-version
├── .gitignore
├── src/style_llm_train/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── lora.py
│   ├── trainer.py
│   ├── merge.py
│   ├── gguf_export.py
│   ├── eval_bleu.py
│   └── cli.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   └── tiny_sft.jsonl
    ├── test_dataset.py
    ├── test_lora.py
    ├── test_trainer.py
    ├── test_merge.py
    ├── test_gguf_export.py
    ├── test_eval_bleu.py
    └── test_cli_smoke.py
```

`models/`, `outputs/`, and `.checkpoints/` are gitignored.

---

## Task 1: Bootstrap project

**Files:**
- Create: `style-llm-train/pyproject.toml`
- Create: `style-llm-train/.python-version`
- Create: `style-llm-train/.gitignore`
- Create: `style-llm-train/README.md`
- Create: `style-llm-train/src/style_llm_train/__init__.py`
- Create: `style-llm-train/tests/conftest.py`
- Create: `style-llm-train/tests/fixtures/.gitkeep`

- [ ] **Step 1: `pyproject.toml`**

```toml
[project]
name = "style-llm-train"
version = "0.1.0"
description = "Qwen2.5-3B LoRA fine-tuning for style/tone rewriting; GGUF export"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.4",
    "transformers>=4.44",
    "tokenizers>=0.19",
    "datasets>=2.20",
    "accelerate>=0.33",
    "peft>=0.12",
    "trl>=0.10",
    "evaluate>=0.4",
    "sacrebleu>=2.4",
    "bert-score>=0.3.13",
    "safetensors>=0.4",
    "typer>=0.12",
    "rich>=13.7",
    "sentencepiece>=0.2.1",
    "tiktoken>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
style-llm-train = "style_llm_train.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/style_llm_train"]

[tool.uv]
constraint-dependencies = ["numpy>=1.25,<2"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]
ignore = ["B008"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
```

- [ ] **Step 2: `.python-version`**

```
3.11
```

- [ ] **Step 3: `.gitignore`**

```
models/
outputs/
.checkpoints/
data/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
dist/
*.egg-info/
*.gguf
*.safetensors
wandb/
```

- [ ] **Step 4: `README.md`**

````markdown
# style-llm-train

LoRA fine-tunes `Qwen2.5-3B-Instruct` for style/tone rewriting. Outputs a GGUF model ready for the Rust runtime.

## Setup

```
cd style-llm-train
uv sync --extra dev
```

You must also clone `llama.cpp` somewhere on disk and note the path:

```
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp && cmake -B build && cmake --build build --target llama-quantize
```

## Run

```
# Train (creates LoRA adapter under .checkpoints/style)
uv run style-llm-train train \
    --jsonl ../data-pipeline/data/processed/style_sft.jsonl \
    --out .checkpoints/style \
    --max-steps 1000

# Merge LoRA into base + write HF checkpoint to outputs/qwen-style-merged
uv run style-llm-train merge \
    --adapter .checkpoints/style \
    --out outputs/qwen-style-merged

# Convert + quantize to Q4_K_M
uv run style-llm-train quantize \
    --hf-checkpoint outputs/qwen-style-merged \
    --out models/qwen-style-q4.gguf \
    --llama-cpp-dir /tmp/llama.cpp \
    --quant Q4_K_M
```

## Apple Silicon notes

QLoRA's 4-bit base via `bitsandbytes` is CUDA-only. On M-series we keep the base model in fp16 and only the LoRA adapters in fp32. Memory is comfortable on a 32GB Mac; 16GB Macs may need `--per-device-batch-size 1 --gradient-accumulation-steps 8`.

The actual 4-bit quantization happens *after* training, when `llama-quantize` produces the Q4_K_M GGUF.
````

- [ ] **Step 5: `src/style_llm_train/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 6: `tests/conftest.py`**

```python
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out
```

- [ ] **Step 7: `tests/fixtures/.gitkeep`** — empty file.

- [ ] **Step 8: Install + verify**

```
cd style-llm-train
uv sync --extra dev
uv run pytest -q
```

Expected: `uv sync` succeeds (~120-150 packages). `pytest -q` exits 5 (no tests). Allow `trl` and `bert-score` to resolve — they pull `nltk`, `numpy`, and other transitive deps.

- [ ] **Step 9: Commit**

```
git add style-llm-train/
git commit -m "feat(style-llm-train): bootstrap project skeleton"
```

---

## Task 2: SFT dataset

**Files:**
- Create: `style-llm-train/src/style_llm_train/dataset.py`
- Create: `style-llm-train/tests/test_dataset.py`
- Create: `style-llm-train/tests/fixtures/tiny_sft.jsonl`

- [ ] **Step 1: Fixture**

Create `style-llm-train/tests/fixtures/tiny_sft.jsonl`:

```
{"messages":[{"role":"system","content":"Rewrite formally."},{"role":"user","content":"yo whats up"},{"role":"assistant","content":"Hello, how are you?"}],"meta":{"source":"gyafc","style_target":"formal"}}
{"messages":[{"role":"system","content":"Rewrite simply."},{"role":"user","content":"He is an erudite gentleman."},{"role":"assistant","content":"He is a smart man."}],"meta":{"source":"wiki_auto","style_target":"simplify"}}
{"messages":[{"role":"system","content":"Detoxify."},{"role":"user","content":"that is dumb"},{"role":"assistant","content":"that is not helpful"}],"meta":{"source":"paradetox","style_target":"detoxify"}}
```

- [ ] **Step 2: Failing test**

Create `style-llm-train/tests/test_dataset.py`:

```python
from pathlib import Path

from style_llm_train.dataset import load_sft_records


def test_load_sft_records_yields_chatml(fixtures_dir: Path) -> None:
    records = list(load_sft_records(fixtures_dir / "tiny_sft.jsonl"))
    assert len(records) == 3
    assert records[0]["messages"][0]["role"] == "system"
    assert records[0]["messages"][1]["role"] == "user"
    assert records[0]["messages"][2]["role"] == "assistant"
    assert records[0]["messages"][2]["content"] == "Hello, how are you?"
    assert records[0]["meta"]["style_target"] == "formal"


def test_load_sft_records_skips_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"messages":[{"role":"system","content":"x"}]}\n'  # only 1 msg, malformed
        '{"messages":[{"role":"system","content":"a"},{"role":"user","content":"b"},{"role":"assistant","content":"c"}]}\n'
        "\n"  # blank line, ignored
    )
    records = list(load_sft_records(bad))
    assert len(records) == 1
```

- [ ] **Step 3: Run, expect failure**

```
cd style-llm-train && uv run pytest tests/test_dataset.py -v
```

- [ ] **Step 4: Implementation**

Create `style-llm-train/src/style_llm_train/dataset.py`:

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_sft_records(jsonl: Path) -> Iterator[dict[str, Any]]:
    """Yield `{messages, meta}` records from a ChatML SFT JSONL file.

    Records lacking the system+user+assistant trio are skipped silently.
    """
    with jsonl.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            roles = [m.get("role") for m in msgs]
            if roles[:3] != ["system", "user", "assistant"]:
                continue
            yield rec
```

- [ ] **Step 5: Run, expect 2 PASS**

```
uv run pytest tests/test_dataset.py -v
```

- [ ] **Step 6: ruff + mypy clean**

- [ ] **Step 7: Commit**

```
git add style-llm-train/src/style_llm_train/dataset.py style-llm-train/tests/test_dataset.py style-llm-train/tests/fixtures/tiny_sft.jsonl
git commit -m "feat(style-llm-train): SFT record loader with ChatML validation"
```

---

## Task 3: LoRA config builder

**Files:**
- Create: `style-llm-train/src/style_llm_train/lora.py`
- Create: `style-llm-train/tests/test_lora.py`

- [ ] **Step 1: Failing test**

Create `style-llm-train/tests/test_lora.py`:

```python
from peft import LoraConfig

from style_llm_train.lora import build_lora_config


def test_lora_config_uses_paper_defaults() -> None:
    cfg = build_lora_config()
    assert isinstance(cfg, LoraConfig)
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(cfg.target_modules) == expected_targets


def test_lora_config_overrides() -> None:
    cfg = build_lora_config(r=8, alpha=16, dropout=0.1)
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implementation**

Create `style-llm-train/src/style_llm_train/lora.py`:

```python
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
```

- [ ] **Step 4: Run, expect 2 PASS**

- [ ] **Step 5: Commit**

```
git add style-llm-train/src/style_llm_train/lora.py style-llm-train/tests/test_lora.py
git commit -m "feat(style-llm-train): LoRA config for Qwen2.5 attention + MLP projections"
```

---

## Task 4: TRL SFT trainer wrapper

**Files:**
- Create: `style-llm-train/src/style_llm_train/config.py`
- Create: `style-llm-train/src/style_llm_train/trainer.py`
- Create: `style-llm-train/tests/test_trainer.py`

- [ ] **Step 1: Failing test**

Create `style-llm-train/tests/test_trainer.py`:

```python
from pathlib import Path

from style_llm_train.config import TrainConfig
from style_llm_train.trainer import run_sft_training


def test_smoke_sft_runs_one_step(fixtures_dir: Path, tmp_out: Path) -> None:
    cfg = TrainConfig(
        jsonl=fixtures_dir / "tiny_sft.jsonl",
        output_dir=tmp_out / "ckpt",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model for the smoke test
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implementation**

Create `style-llm-train/src/style_llm_train/config.py`:

```python
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
```

Create `style-llm-train/src/style_llm_train/trainer.py`:

```python
from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from style_llm_train.config import TrainConfig
from style_llm_train.dataset import load_sft_records
from style_llm_train.lora import build_lora_config


def _format_chatml(record: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    text = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def run_sft_training(*, cfg: TrainConfig) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = list(load_sft_records(cfg.jsonl))
    formatted = [_format_chatml(r, tokenizer) for r in records]
    ds = Dataset.from_list(formatted)

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16
    )
    base.gradient_checkpointing_enable()
    model = get_peft_model(base, build_lora_config())

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        max_seq_length=cfg.max_seq_length,
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
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    out = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return dict(out.metrics)
```

- [ ] **Step 4: Run, expect 1 PASS**

This is the slowest test (~3-6 minutes, downloads + loads Qwen2.5-0.5B + one optimization step).

```
uv run pytest tests/test_trainer.py -v
```

- [ ] **Step 5: ruff + mypy clean**

Known potential snags:
- `SFTConfig` may have renamed `max_seq_length`. If the test fails with `TypeError`, search the installed `trl` version for the right kwarg.
- `processing_class` is the transformers 5.x name; `tokenizer` is the older alias. Try `processing_class` first.
- If `gradient_checkpointing_enable` errors, drop the call — smoke test doesn't need it.

- [ ] **Step 6: Commit**

```
git add style-llm-train/src/style_llm_train/config.py style-llm-train/src/style_llm_train/trainer.py style-llm-train/tests/test_trainer.py
git commit -m "feat(style-llm-train): TRL SFTTrainer wrapper with LoRA + ChatML formatting"
```

---

## Task 5: Adapter merge

**Files:**
- Create: `style-llm-train/src/style_llm_train/merge.py`
- Create: `style-llm-train/tests/test_merge.py`

- [ ] **Step 1: Failing test**

Create `style-llm-train/tests/test_merge.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from style_llm_train.merge import merge_adapter


def test_merge_adapter_invokes_peft_and_saves(tmp_out: Path) -> None:
    adapter_dir = tmp_out / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    out_dir = tmp_out / "merged"

    merged_model = MagicMock()
    merged_model.save_pretrained = MagicMock()
    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged_model

    fake_base = MagicMock()
    fake_tokenizer = MagicMock()

    with patch("style_llm_train.merge.AutoModelForCausalLM.from_pretrained", return_value=fake_base) as m_base, \
         patch("style_llm_train.merge.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as m_tok, \
         patch("style_llm_train.merge.PeftModel.from_pretrained", return_value=peft_model) as m_peft:
        merge_adapter(
            adapter_dir=adapter_dir,
            out_dir=out_dir,
            base_model="Qwen/Qwen2.5-3B-Instruct",
        )

    m_base.assert_called_once()
    m_peft.assert_called_once_with(fake_base, str(adapter_dir))
    peft_model.merge_and_unload.assert_called_once()
    merged_model.save_pretrained.assert_called_once_with(str(out_dir))
    fake_tokenizer.save_pretrained.assert_called_once_with(str(out_dir))
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implementation**

Create `style-llm-train/src/style_llm_train/merge.py`:

```python
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
```

- [ ] **Step 4: Run, expect 1 PASS**

- [ ] **Step 5: Commit**

```
git add style-llm-train/src/style_llm_train/merge.py style-llm-train/tests/test_merge.py
git commit -m "feat(style-llm-train): LoRA adapter merge into base HF checkpoint"
```

---

## Task 6: GGUF export wrapper

**Files:**
- Create: `style-llm-train/src/style_llm_train/gguf_export.py`
- Create: `style-llm-train/tests/test_gguf_export.py`

- [ ] **Step 1: Failing test**

Create `style-llm-train/tests/test_gguf_export.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from style_llm_train.gguf_export import convert_and_quantize


def test_convert_and_quantize_invokes_subprocesses(tmp_out: Path) -> None:
    hf_dir = tmp_out / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    llama_dir = tmp_out / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_hf_to_gguf.py").write_text("# stub")
    build_dir = llama_dir / "build" / "bin"
    build_dir.mkdir(parents=True)
    (build_dir / "llama-quantize").write_text("# stub")

    out = tmp_out / "out.gguf"

    def fake_run(args, check, cwd=None):  # noqa: ANN001
        # Simulate the converter producing the f16 intermediate.
        if "convert_hf_to_gguf.py" in args[1]:
            (out.with_suffix(".f16.gguf")).write_text("stub")
        return None

    with patch("style_llm_train.gguf_export.subprocess.run", side_effect=fake_run) as run:
        convert_and_quantize(
            hf_checkpoint=hf_dir,
            out=out,
            llama_cpp_dir=llama_dir,
            quant="Q4_K_M",
        )
    assert run.call_count == 2


def test_missing_llama_cpp_dir_raises(tmp_out: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_and_quantize(
            hf_checkpoint=tmp_out,
            out=tmp_out / "x.gguf",
            llama_cpp_dir=tmp_out / "no-such-dir",
            quant="Q4_K_M",
        )
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implementation**

Create `style-llm-train/src/style_llm_train/gguf_export.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path


def convert_and_quantize(
    *,
    hf_checkpoint: Path,
    out: Path,
    llama_cpp_dir: Path,
    quant: str = "Q4_K_M",
) -> Path:
    """Run llama.cpp convert + quantize as subprocesses.

    Produces an intermediate `<out>.f16.gguf` then a final quantized GGUF.
    Requires `llama_cpp_dir/convert_hf_to_gguf.py` and a built
    `llama_cpp_dir/build/bin/llama-quantize` binary.
    """
    if not llama_cpp_dir.is_dir():
        raise FileNotFoundError(f"llama.cpp dir not found: {llama_cpp_dir}")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not convert_script.is_file():
        raise FileNotFoundError(f"missing converter: {convert_script}")
    if not quantize_bin.is_file():
        raise FileNotFoundError(
            f"missing quantize binary (build llama.cpp first): {quantize_bin}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    intermediate = out.with_suffix(".f16.gguf")

    subprocess.run(
        [
            "python",
            str(convert_script),
            str(hf_checkpoint),
            "--outfile",
            str(intermediate),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    subprocess.run(
        [str(quantize_bin), str(intermediate), str(out), quant],
        check=True,
    )
    return out
```

- [ ] **Step 4: Run, expect 2 PASS**

- [ ] **Step 5: Commit**

```
git add style-llm-train/src/style_llm_train/gguf_export.py style-llm-train/tests/test_gguf_export.py
git commit -m "feat(style-llm-train): GGUF convert+quantize wrapper around llama.cpp tools"
```

---

## Task 7: BLEU evaluation helper

**Files:**
- Create: `style-llm-train/src/style_llm_train/eval_bleu.py`
- Create: `style-llm-train/tests/test_eval_bleu.py`

- [ ] **Step 1: Failing test**

Create `style-llm-train/tests/test_eval_bleu.py`:

```python
from style_llm_train.eval_bleu import compute_corpus_bleu


def test_identical_corpus_scores_one() -> None:
    score = compute_corpus_bleu(
        hyp=["Hello, how are you?"], ref=["Hello, how are you?"]
    )
    assert score["bleu"] > 0.9


def test_mismatched_lengths_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_corpus_bleu(hyp=["a", "b"], ref=["a"])
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implementation**

Create `style-llm-train/src/style_llm_train/eval_bleu.py`:

```python
from __future__ import annotations

from sacrebleu import corpus_bleu


def compute_corpus_bleu(*, hyp: list[str], ref: list[str]) -> dict[str, float]:
    """Single-reference corpus BLEU via sacrebleu."""
    if len(hyp) != len(ref):
        raise ValueError(f"length mismatch: {len(hyp)} hyp vs {len(ref)} ref")
    score = corpus_bleu(hyp, [ref])
    return {"bleu": float(score.score) / 100.0}
```

- [ ] **Step 4: Run, expect 2 PASS**

- [ ] **Step 5: Commit**

```
git add style-llm-train/src/style_llm_train/eval_bleu.py style-llm-train/tests/test_eval_bleu.py
git commit -m "feat(style-llm-train): sacrebleu corpus BLEU helper"
```

---

## Task 8: Typer CLI

**Files:**
- Create: `style-llm-train/src/style_llm_train/cli.py`
- Create: `style-llm-train/tests/test_cli_smoke.py`

- [ ] **Step 1: Failing test**

Create `style-llm-train/tests/test_cli_smoke.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from style_llm_train.cli import app


def test_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "merge" in result.output
    assert "quantize" in result.output


def test_train_command_smoke(fixtures_dir: Path, tmp_out: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--jsonl",
            str(fixtures_dir / "tiny_sft.jsonl"),
            "--out",
            str(tmp_out / "ckpt"),
            "--base-model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--max-steps",
            "1",
            "--per-device-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--max-seq-length",
            "128",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_out / "ckpt").is_dir()
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implementation**

Create `style-llm-train/src/style_llm_train/cli.py`:

```python
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
```

- [ ] **Step 4: Run, expect 2 PASS**

- [ ] **Step 5: Commit**

```
git add style-llm-train/src/style_llm_train/cli.py style-llm-train/tests/test_cli_smoke.py
git commit -m "feat(style-llm-train): typer CLI (train, merge, quantize)"
```

---

## Task 9: Lint/type/coverage gates

**Files:** none new.

- [ ] **Step 1: ruff**

```
cd style-llm-train && uv run ruff check src tests
```

- [ ] **Step 2: mypy strict**

```
cd style-llm-train && uv run mypy src
```

- [ ] **Step 3: pytest with coverage**

```
cd style-llm-train && uv run pytest --cov=style_llm_train --cov-report=term
```

Target: ≥ 70% on `src/style_llm_train` (lower than data-pipeline's 85% because slow ML tests are necessarily sparse).

- [ ] **Step 4: Commit cleanup if any**

```
git status
git diff
git add -A
git diff --cached --quiet && echo "clean" || git commit -m "chore(style-llm-train): lint/type/coverage cleanup"
```

---

## Notes

- The 4-bit quantization happens at *export* time via `llama-quantize`, not during training. This is the practical Apple-Silicon equivalent of QLoRA: fp16 training + Q4_K_M serving.
- Training runs are not part of CI. The `--max-steps 1` smoke is the only thing tested.
- The full multi-tone curriculum (formality → simplification → detoxification, etc.) is an operator playbook, not a single CLI command.
