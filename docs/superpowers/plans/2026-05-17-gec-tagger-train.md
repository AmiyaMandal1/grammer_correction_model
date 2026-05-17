# GEC Tagger Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package `gec-tagger-train/` that fine-tunes `microsoft/deberta-v3-base` as a GECToR-style token-classification model on the `gec_tagger.jsonl` produced by `data-pipeline/`, evaluates against CoNLL-2014 (M²) and BEA-dev (ERRANT), and exports to ONNX opset 17 plus a `tokenizer.json`.

**Architecture:** A tag vocabulary builder reads `gec_tagger.jsonl` and emits a sorted, frequency-pruned `tags.json` (target ≈5K tags). A PyTorch `Dataset` aligns DeBERTa subword tokens to whitespace tokens and broadcasts each whitespace tag to its subwords. A custom `DebertaTagger` model wraps `AutoModel` with a linear classification head over the tag vocab, trained with cross-entropy + label smoothing 0.1. A Hugging Face `Trainer`-based loop supports three stages: synthetic pretrain (C4_200M), high-volume mix (BEA-2019), and small high-quality finish (W&I+L only). Evaluation calls the published `errant` and `m2scorer` tools on decoded predictions. ONNX export is the final step. A typer CLI exposes `train`, `eval`, `export`.

**Tech Stack:** Python 3.11, `uv`, `torch>=2.4` (MPS backend on Apple Silicon), `transformers>=4.44`, `tokenizers`, `datasets`, `peft` (for optional LoRA), `evaluate`, `errant`, `m2scorer` (vendored), `onnx>=1.16`, `onnxruntime>=1.18`, `typer`, `pytest`, `ruff`, `mypy`.

---

## Layout

```
gec-tagger-train/
├── pyproject.toml
├── README.md
├── .python-version
├── .gitignore
├── src/gec_tagger_train/
│   ├── __init__.py
│   ├── config.py
│   ├── tag_vocab.py
│   ├── tokenization.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── eval_m2.py
│   ├── eval_errant.py
│   ├── decode.py
│   ├── export_onnx.py
│   └── cli.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   └── tiny.jsonl
    ├── test_tag_vocab.py
    ├── test_tokenization.py
    ├── test_dataset.py
    ├── test_model.py
    ├── test_decode.py
    ├── test_export_onnx.py
    └── test_cli_smoke.py
```

`models/`, `outputs/`, and `.checkpoints/` are gitignored runtime directories.

---

## Task 1: Bootstrap project

**Files:**
- Create: `gec-tagger-train/pyproject.toml`
- Create: `gec-tagger-train/.python-version`
- Create: `gec-tagger-train/.gitignore`
- Create: `gec-tagger-train/README.md`
- Create: `gec-tagger-train/src/gec_tagger_train/__init__.py`
- Create: `gec-tagger-train/tests/conftest.py`
- Create: `gec-tagger-train/tests/fixtures/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "gec-tagger-train"
version = "0.1.0"
description = "DeBERTa GECToR tagger fine-tuning, evaluation, and export"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.4",
    "transformers>=4.44",
    "tokenizers>=0.19",
    "datasets>=2.20",
    "accelerate>=0.33",
    "peft>=0.12",
    "evaluate>=0.4",
    "errant>=3.0.0",
    "spacy>=3.7,<3.8",
    "en-core-web-sm==3.7.1",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "typer>=0.12",
    "rich>=13.7",
    "polars>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
gec-tagger-train = "gec_tagger_train.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gec_tagger_train"]

[tool.uv]
constraint-dependencies = ["numpy>=1.25,<2"]

[tool.uv.sources]
en-core-web-sm = { url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl" }

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

- [ ] **Step 2: Create `.python-version`**

```
3.11
```

- [ ] **Step 3: Create `.gitignore`**

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
*.onnx
wandb/
```

- [ ] **Step 4: Create `README.md`**

````markdown
# gec-tagger-train

Fine-tunes DeBERTa-v3-base as a GECToR-style grammatical error tagger.

## Setup

```
cd gec-tagger-train
uv sync --extra dev
```

## Run

```
# Build tag vocab from data-pipeline output
uv run gec-tagger-train build-vocab \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --out models/tags.json

# Stage 1: pretrain on synthetic C4_200M subset
uv run gec-tagger-train train \
    --stage 1 \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --tags models/tags.json \
    --out .checkpoints/stage1

# Evaluate
uv run gec-tagger-train eval \
    --checkpoint .checkpoints/stage1 \
    --m2 path/to/conll14.m2

# Export to ONNX
uv run gec-tagger-train export \
    --checkpoint .checkpoints/stage1 \
    --out models/deberta-gec.onnx
```
````

- [ ] **Step 5: Create `src/gec_tagger_train/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 6: Create `tests/conftest.py`**

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

- [ ] **Step 7: Create `tests/fixtures/.gitkeep`** (empty file).

- [ ] **Step 8: Install deps and verify**

```
cd gec-tagger-train
uv sync --extra dev
uv run pytest -q
```

Expected: `uv sync` succeeds, `pytest -q` exits 5 (no tests collected) — acceptable.

- [ ] **Step 9: Commit**

```
git add gec-tagger-train/
git commit -m "feat(gec-tagger-train): bootstrap project skeleton"
```

---

## Task 2: Tag vocabulary builder

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/tag_vocab.py`
- Create: `gec-tagger-train/tests/test_tag_vocab.py`
- Create: `gec-tagger-train/tests/fixtures/tiny.jsonl`

- [ ] **Step 1: Create fixture**

Create `gec-tagger-train/tests/fixtures/tiny.jsonl`:

```
{"tokens":["he","go","home"],"tags":["$KEEP","$REPLACE_goes","$KEEP"],"source":"t","split":"train","meta":{}}
{"tokens":["i","am","happy"],"tags":["$KEEP","$KEEP","$KEEP"],"source":"t","split":"train","meta":{}}
{"tokens":["the","cat","sit"],"tags":["$KEEP","$KEEP","$REPLACE_sits"],"source":"t","split":"train","meta":{}}
{"tokens":["he","goes"],"tags":["$KEEP","$REPLACE_goes"],"source":"t","split":"train","meta":{}}
```

- [ ] **Step 2: Failing test**

Create `gec-tagger-train/tests/test_tag_vocab.py`:

```python
import json
from pathlib import Path

from gec_tagger_train.tag_vocab import TagVocab, build_tag_vocab


def test_build_vocab_pads_special_tags(fixtures_dir: Path, tmp_out: Path) -> None:
    out = tmp_out / "tags.json"
    vocab = build_tag_vocab(jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=out)
    assert isinstance(vocab, TagVocab)
    assert vocab.special_tags == ["$PAD", "$UNK", "$KEEP"]
    assert vocab.id_of("$PAD") == 0
    assert vocab.id_of("$UNK") == 1
    assert vocab.id_of("$KEEP") == 2
    # Frequent tag $REPLACE_goes appears twice; should be present.
    assert vocab.id_of("$REPLACE_goes") >= 3
    # Less frequent $REPLACE_sits (count=1) also present when min_count=1.
    assert "$REPLACE_sits" in vocab.tags

    payload = json.loads(out.read_text())
    assert payload["tags"][0] == "$PAD"
    assert payload["tags"][2] == "$KEEP"


def test_min_count_filters_rare_tags(fixtures_dir: Path, tmp_out: Path) -> None:
    out = tmp_out / "tags.json"
    vocab = build_tag_vocab(jsonl=fixtures_dir / "tiny.jsonl", min_count=2, out=out)
    # $REPLACE_sits has count 1 -> filtered
    assert "$REPLACE_sits" not in vocab.tags
    # $REPLACE_goes has count 2 -> kept
    assert "$REPLACE_goes" in vocab.tags


def test_unknown_tag_maps_to_unk(fixtures_dir: Path, tmp_out: Path) -> None:
    vocab = build_tag_vocab(
        jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=tmp_out / "t.json"
    )
    assert vocab.id_of("$REPLACE_neverseen") == vocab.id_of("$UNK")


def test_tag_vocab_round_trip(tmp_out: Path) -> None:
    v = TagVocab(tags=["$PAD", "$UNK", "$KEEP", "$DELETE", "$REPLACE_foo"])
    p = tmp_out / "v.json"
    v.save(p)
    v2 = TagVocab.load(p)
    assert v2.tags == v.tags
    assert v2.id_of("$REPLACE_foo") == 4
```

- [ ] **Step 3: Run, expect failure**

```
uv run pytest tests/test_tag_vocab.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/tag_vocab.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


_SPECIAL_TAGS: tuple[str, ...] = ("$PAD", "$UNK", "$KEEP")


@dataclass
class TagVocab:
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._tag_to_id: dict[str, int] = {t: i for i, t in enumerate(self.tags)}

    @property
    def special_tags(self) -> list[str]:
        return [t for t in self.tags if t in _SPECIAL_TAGS]

    def id_of(self, tag: str) -> int:
        return self._tag_to_id.get(tag, self._tag_to_id["$UNK"])

    def __len__(self) -> int:
        return len(self.tags)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"tags": self.tags}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TagVocab":
        payload = json.loads(path.read_text())
        return cls(tags=list(payload["tags"]))


def build_tag_vocab(*, jsonl: Path, min_count: int, out: Path) -> TagVocab:
    counts: Counter[str] = Counter()
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for tag in rec["tags"]:
                if tag in _SPECIAL_TAGS:
                    continue
                counts[tag] += 1
    kept = [t for t, c in counts.most_common() if c >= min_count]
    vocab = TagVocab(tags=list(_SPECIAL_TAGS) + kept)
    vocab.save(out)
    return vocab
```

- [ ] **Step 5: Run, expect PASS**

```
uv run pytest tests/test_tag_vocab.py -v
```

Expected: 4 PASS.

- [ ] **Step 6: ruff + mypy clean**

```
uv run ruff check src tests
uv run mypy src
```

- [ ] **Step 7: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/tag_vocab.py gec-tagger-train/tests/test_tag_vocab.py gec-tagger-train/tests/fixtures/tiny.jsonl
git commit -m "feat(gec-tagger-train): tag vocabulary builder with frequency pruning"
```

---

## Task 3: Subword tokenization with tag alignment

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/tokenization.py`
- Create: `gec-tagger-train/tests/test_tokenization.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_tokenization.py`:

```python
from unittest.mock import MagicMock

from gec_tagger_train.tokenization import align_tags_to_subwords


def test_aligns_single_subword_tokens() -> None:
    # Tokenizer returns word_ids that map each subword back to its whitespace
    # token. Pretend "the cat sits" → 3 word-level tokens, each is 1 subword.
    enc = MagicMock()
    enc.word_ids.return_value = [None, 0, 1, 2, None]  # CLS, the, cat, sits, SEP
    tags = ["$KEEP", "$KEEP", "$REPLACE_sits"]
    out = align_tags_to_subwords(enc, tags, pad_id=0, unk_id=1, vocab_lookup=lambda t: {"$KEEP": 2, "$REPLACE_sits": 7}[t])
    assert out == [0, 2, 2, 7, 0]


def test_aligns_multi_subword_token() -> None:
    enc = MagicMock()
    # "discombobulated" → 3 subwords for word 0
    enc.word_ids.return_value = [None, 0, 0, 0, None]
    tags = ["$REPLACE_confused"]
    vocab = {"$REPLACE_confused": 5}
    out = align_tags_to_subwords(enc, tags, pad_id=0, unk_id=1, vocab_lookup=lambda t: vocab[t])
    # All subwords of word 0 get the same tag id.
    assert out == [0, 5, 5, 5, 0]


def test_unknown_tag_falls_back_to_unk() -> None:
    enc = MagicMock()
    enc.word_ids.return_value = [None, 0, None]
    tags = ["$REPLACE_neverseen"]

    def lookup(t: str) -> int:
        raise KeyError(t)

    out = align_tags_to_subwords(enc, tags, pad_id=0, unk_id=1, vocab_lookup=lookup)
    assert out == [0, 1, 0]
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_tokenization.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/tokenization.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def align_tags_to_subwords(
    encoding: Any,
    tags: list[str],
    *,
    pad_id: int,
    unk_id: int,
    vocab_lookup: Callable[[str], int],
) -> list[int]:
    """Broadcast a per-whitespace-token tag list to subword-token ids.

    `encoding.word_ids()` returns a list with one entry per subword: either
    the index of the source whitespace token, or `None` for special tokens
    ([CLS], [SEP], pad). Special tokens get `pad_id`; tags not present in
    the vocabulary fall back to `unk_id`.
    """
    word_ids = encoding.word_ids()
    out: list[int] = []
    for w in word_ids:
        if w is None:
            out.append(pad_id)
            continue
        tag = tags[w]
        try:
            out.append(vocab_lookup(tag))
        except KeyError:
            out.append(unk_id)
    return out
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_tokenization.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/tokenization.py gec-tagger-train/tests/test_tokenization.py
git commit -m "feat(gec-tagger-train): tag-to-subword alignment helper"
```

---

## Task 4: PyTorch dataset

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/dataset.py`
- Create: `gec-tagger-train/tests/test_dataset.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_dataset.py`:

```python
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gec_tagger_train.dataset import GECTaggerDataset
from gec_tagger_train.tag_vocab import build_tag_vocab


def test_dataset_yields_aligned_tensors(fixtures_dir: Path, tmp_out: Path) -> None:
    vocab = build_tag_vocab(
        jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=tmp_out / "tags.json"
    )
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    ds = GECTaggerDataset(
        jsonl=fixtures_dir / "tiny.jsonl",
        tokenizer=tok,
        vocab=vocab,
        max_length=32,
    )
    assert len(ds) == 4
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].shape == item["attention_mask"].shape == item["labels"].shape
    # First and last positions are special tokens → PAD id (0)
    assert int(item["labels"][0]) == 0
    assert int(item["labels"][-1]) == 0


def test_dataset_filters_rows_longer_than_max_length(fixtures_dir: Path, tmp_out: Path) -> None:
    vocab = build_tag_vocab(
        jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=tmp_out / "tags.json"
    )
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    ds = GECTaggerDataset(
        jsonl=fixtures_dir / "tiny.jsonl",
        tokenizer=tok,
        vocab=vocab,
        max_length=2,  # any non-trivial row exceeds this
    )
    assert len(ds) == 0
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_dataset.py -v
```

The first run will also download the DeBERTa tokenizer (~50MB). Allow it to complete.

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/dataset.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from gec_tagger_train.tag_vocab import TagVocab
from gec_tagger_train.tokenization import align_tags_to_subwords


class GECTaggerDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        jsonl: Path,
        tokenizer: Any,
        vocab: TagVocab,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self._items: list[dict[str, torch.Tensor]] = []
        self._load(jsonl)

    def _load(self, jsonl: Path) -> None:
        pad_id = self.vocab.id_of("$PAD")
        unk_id = self.vocab.id_of("$UNK")
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tokens: list[str] = rec["tokens"]
                tags: list[str] = rec["tags"]
                if len(tokens) != len(tags):
                    raise ValueError(
                        f"length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
                    )
                enc = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )
                if len(enc["input_ids"]) > self.max_length:
                    continue
                labels = align_tags_to_subwords(
                    enc, tags, pad_id=pad_id, unk_id=unk_id, vocab_lookup=self.vocab.id_of
                )
                self._items.append(
                    {
                        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(
                            enc["attention_mask"], dtype=torch.long
                        ),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._items[idx]
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_dataset.py -v
```

Expected: 2 PASS. First run is slow (tokenizer download).

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/dataset.py gec-tagger-train/tests/test_dataset.py
git commit -m "feat(gec-tagger-train): PyTorch dataset with subword-aligned labels"
```

---

## Task 5: Tagger model

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/model.py`
- Create: `gec-tagger-train/tests/test_model.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_model.py`:

```python
import torch

from gec_tagger_train.model import DebertaTagger


def test_forward_returns_logits_and_loss() -> None:
    model = DebertaTagger(num_tags=10, pretrained="microsoft/deberta-v3-base")
    input_ids = torch.tensor([[101, 200, 300, 102]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[0, 2, 3, 0]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert "logits" in out
    assert "loss" in out
    assert out["logits"].shape == (1, 4, 10)
    assert out["loss"].dim() == 0  # scalar


def test_forward_without_labels_returns_logits_only() -> None:
    model = DebertaTagger(num_tags=10, pretrained="microsoft/deberta-v3-base")
    input_ids = torch.tensor([[101, 200, 102]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert "logits" in out
    assert "loss" not in out
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_model.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/model.py`:

```python
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class DebertaTagger(nn.Module):
    def __init__(
        self,
        *,
        num_tags: int,
        pretrained: str = "microsoft/deberta-v3-base",
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained)
        self.encoder = AutoModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=pad_id)
        self.num_tags = num_tags
        self.pad_id = pad_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.num_tags),
                labels.view(-1),
            )
            out["loss"] = loss
        return out
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_model.py -v
```

Expected: 2 PASS. Slow because it instantiates DeBERTa twice.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/model.py gec-tagger-train/tests/test_model.py
git commit -m "feat(gec-tagger-train): DebertaTagger with label-smoothed CE loss"
```

---

## Task 6: Iterative-decoding helper

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/decode.py`
- Create: `gec-tagger-train/tests/test_decode.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_decode.py`:

```python
from gec_tagger_train.decode import apply_tags_once


def test_keep_returns_input_unchanged() -> None:
    out = apply_tags_once(["he", "go", "home"], ["$KEEP", "$KEEP", "$KEEP"])
    assert out == ["he", "go", "home"]


def test_replace_substitutes_token() -> None:
    out = apply_tags_once(["he", "go", "home"], ["$KEEP", "$REPLACE_goes", "$KEEP"])
    assert out == ["he", "goes", "home"]


def test_delete_removes_token() -> None:
    out = apply_tags_once(["he", "the", "goes"], ["$KEEP", "$DELETE", "$KEEP"])
    assert out == ["he", "goes"]


def test_append_inserts_after_anchor() -> None:
    out = apply_tags_once(["he", "home"], ["$APPEND_goes", "$KEEP"])
    assert out == ["he", "goes", "home"]


def test_unknown_tag_treated_as_keep() -> None:
    out = apply_tags_once(["he", "go"], ["$KEEP", "$WAT"])
    assert out == ["he", "go"]
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_decode.py -v
```

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/decode.py`:

```python
from __future__ import annotations


def apply_tags_once(tokens: list[str], tags: list[str]) -> list[str]:
    """Apply one round of GECToR edit tags to a token list.

    Unrecognized tags are treated as `$KEEP` so that decoding is robust to
    vocabulary drift. Use multiple passes (caller-driven) to handle cases
    that need more than one edit at a position.
    """
    if len(tokens) != len(tags):
        raise ValueError(
            f"length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
        )
    out: list[str] = []
    for tok, tag in zip(tokens, tags, strict=True):
        if tag == "$KEEP" or not tag.startswith("$"):
            out.append(tok)
            continue
        if tag == "$DELETE":
            continue
        kind, _, value = tag[1:].partition("_")
        if kind == "REPLACE" and value:
            out.append(value)
        elif kind == "APPEND" and value:
            out.append(tok)
            out.append(value)
        else:
            # Unknown / unsupported -> treat as KEEP
            out.append(tok)
    return out
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_decode.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/decode.py gec-tagger-train/tests/test_decode.py
git commit -m "feat(gec-tagger-train): single-pass tag decoder with KEEP/DELETE/REPLACE/APPEND"
```

---

## Task 7: Training config + Trainer wrapper

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/config.py`
- Create: `gec-tagger-train/src/gec_tagger_train/trainer.py`
- Create: `gec-tagger-train/tests/test_trainer.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_trainer.py`:

```python
from pathlib import Path

from gec_tagger_train.config import StageConfig
from gec_tagger_train.dataset import GECTaggerDataset
from gec_tagger_train.tag_vocab import build_tag_vocab
from gec_tagger_train.trainer import run_training
from transformers import AutoTokenizer


def test_smoke_train_runs_one_step(fixtures_dir: Path, tmp_out: Path) -> None:
    vocab = build_tag_vocab(
        jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=tmp_out / "tags.json"
    )
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    ds = GECTaggerDataset(
        jsonl=fixtures_dir / "tiny.jsonl", tokenizer=tok, vocab=vocab, max_length=32
    )
    cfg = StageConfig(
        name="smoke",
        train_jsonl=fixtures_dir / "tiny.jsonl",
        max_length=32,
        per_device_batch_size=2,
        learning_rate=1e-5,
        num_epochs=1,
        max_steps=1,
        warmup_ratio=0.0,
        output_dir=tmp_out / "ckpt",
        gradient_checkpointing=False,
        fp16=False,
        bf16=False,
    )
    metrics = run_training(cfg=cfg, dataset=ds, vocab=vocab, tokenizer=tok)
    assert "train_runtime" in metrics
    assert (tmp_out / "ckpt").is_dir()
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_trainer.py -v
```

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/config.py`:

```python
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
```

Create `gec-tagger-train/src/gec_tagger_train/trainer.py`:

```python
from __future__ import annotations

from typing import Any

from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from gec_tagger_train.config import StageConfig
from gec_tagger_train.model import DebertaTagger
from gec_tagger_train.tag_vocab import TagVocab


def run_training(
    *,
    cfg: StageConfig,
    dataset: Any,
    vocab: TagVocab,
    tokenizer: Any,
) -> dict[str, float]:
    model = DebertaTagger(num_tags=len(vocab), pad_id=vocab.id_of("$PAD"))

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=vocab.id_of("$PAD")
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    out = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return dict(out.metrics)
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_trainer.py -v
```

Expected: 1 PASS. This is the slowest test in the suite (~30-90s on M-series; loads DeBERTa, runs one forward+backward step).

If the test runs out of memory on a 16GB Mac, lower `per_device_batch_size` to 1 in the test config — but first try as-is.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/config.py gec-tagger-train/src/gec_tagger_train/trainer.py gec-tagger-train/tests/test_trainer.py
git commit -m "feat(gec-tagger-train): HF Trainer wrapper + StageConfig"
```

---

## Task 8: M² evaluation wrapper

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/eval_m2.py`
- Create: `gec-tagger-train/tests/test_eval_m2.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_eval_m2.py`:

```python
from gec_tagger_train.eval_m2 import format_m2_predictions


def test_format_predictions_emits_m2_block() -> None:
    src_tokens = ["he", "go", "home"]
    pred_tokens = ["he", "goes", "home"]
    block = format_m2_predictions(src_tokens=src_tokens, pred_tokens=pred_tokens)
    assert block.startswith("S he go home")
    # A single substitution edit produced
    assert "A 1 2|||" in block


def test_no_change_emits_noop_edit() -> None:
    block = format_m2_predictions(src_tokens=["he", "is", "ok"], pred_tokens=["he", "is", "ok"])
    assert "A -1 -1|||noop|||" in block
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_eval_m2.py -v
```

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/eval_m2.py`:

```python
from __future__ import annotations

from difflib import SequenceMatcher


def format_m2_predictions(*, src_tokens: list[str], pred_tokens: list[str]) -> str:
    """Emit a single-sentence M² block for the m2scorer tool.

    Edits are derived from a token-level diff. Replacement spans are emitted
    as `A start end|||R|||replacement|||REQUIRED|||-NONE-|||0`. A perfect
    match emits the canonical `noop` edit (`A -1 -1`).
    """
    lines: list[str] = [f"S {' '.join(src_tokens)}"]
    matcher = SequenceMatcher(a=src_tokens, b=pred_tokens, autojunk=False)
    any_edit = False
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        any_edit = True
        replacement = " ".join(pred_tokens[j1:j2])
        kind = {"replace": "R", "delete": "U", "insert": "M"}[op]
        # M² spec: insertions are A i i, deletions are A i j, replacements A i j
        end = i1 if op == "insert" else i2
        lines.append(
            f"A {i1} {end}|||{kind}|||{replacement}|||REQUIRED|||-NONE-|||0"
        )
    if not any_edit:
        lines.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_eval_m2.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/eval_m2.py gec-tagger-train/tests/test_eval_m2.py
git commit -m "feat(gec-tagger-train): M² block formatter for m2scorer"
```

---

## Task 9: ERRANT evaluation wrapper

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/eval_errant.py`
- Create: `gec-tagger-train/tests/test_eval_errant.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_eval_errant.py`:

```python
from gec_tagger_train.eval_errant import compute_errant_f05


def test_perfect_match_gives_f05_one() -> None:
    score = compute_errant_f05(
        src=["He go home."],
        ref=["He goes home."],
        hyp=["He goes home."],
    )
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f0.5"] == 1.0


def test_no_match_gives_zero() -> None:
    score = compute_errant_f05(
        src=["He go home."],
        ref=["He goes home."],
        hyp=["He go home."],  # no fix
    )
    assert score["recall"] == 0.0
    assert score["f0.5"] == 0.0
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_eval_errant.py -v
```

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/eval_errant.py`:

```python
from __future__ import annotations

import errant
import spacy

_NLP = None


def _nlp() -> "spacy.language.Language":
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def compute_errant_f05(
    *, src: list[str], ref: list[str], hyp: list[str]
) -> dict[str, float]:
    """F0.5 between reference and hypothesis edits, scored by ERRANT."""
    if not (len(src) == len(ref) == len(hyp)):
        raise ValueError("src, ref, hyp must have equal length")
    ann = errant.load("en", _nlp())
    tp = fp = fn = 0
    for s, r, h in zip(src, ref, hyp, strict=True):
        src_doc = ann.parse(s, tokenise=True)
        ref_doc = ann.parse(r, tokenise=True)
        hyp_doc = ann.parse(h, tokenise=True)
        ref_edits = {(e.o_start, e.o_end, e.c_str) for e in ann.annotate(src_doc, ref_doc)}
        hyp_edits = {(e.o_start, e.o_end, e.c_str) for e in ann.annotate(src_doc, hyp_doc)}
        tp += len(ref_edits & hyp_edits)
        fp += len(hyp_edits - ref_edits)
        fn += len(ref_edits - hyp_edits)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta = 0.5
    if precision + recall == 0:
        f = 0.0
    else:
        f = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    return {"precision": precision, "recall": recall, "f0.5": f}
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_eval_errant.py -v
```

Expected: 2 PASS. First run loads spaCy + ERRANT.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/eval_errant.py gec-tagger-train/tests/test_eval_errant.py
git commit -m "feat(gec-tagger-train): ERRANT F0.5 scorer"
```

---

## Task 10: ONNX export

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/export_onnx.py`
- Create: `gec-tagger-train/tests/test_export_onnx.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_export_onnx.py`:

```python
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
    enc = tok("hello world", return_tensors="np", padding="max_length", max_length=16, truncation=True)
    out_np = sess.run(
        None,
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
    )
    assert out_np[0].shape == (1, 16, 8)
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_export_onnx.py -v
```

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/export_onnx.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch

from gec_tagger_train.model import DebertaTagger


def export_tagger_to_onnx(
    *,
    model: DebertaTagger,
    out_path: Path,
    max_length: int = 128,
    opset: int = 17,
) -> Path:
    model.eval()
    dummy_input_ids = torch.zeros((1, max_length), dtype=torch.long)
    dummy_mask = torch.ones((1, max_length), dtype=torch.long)

    class _Wrapper(torch.nn.Module):
        def __init__(self, m: DebertaTagger) -> None:
            super().__init__()
            self.m = m

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            return self.m(input_ids=input_ids, attention_mask=attention_mask)["logits"]

    wrapper = _Wrapper(model)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_mask),
        str(out_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
    )
    return out_path
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_export_onnx.py -v
```

Expected: 1 PASS. Slow (~60-120s) due to full DeBERTa export.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/export_onnx.py gec-tagger-train/tests/test_export_onnx.py
git commit -m "feat(gec-tagger-train): ONNX export at opset 17 with dynamic axes"
```

---

## Task 11: CLI

**Files:**
- Create: `gec-tagger-train/src/gec_tagger_train/cli.py`
- Create: `gec-tagger-train/tests/test_cli_smoke.py`

- [ ] **Step 1: Failing test**

Create `gec-tagger-train/tests/test_cli_smoke.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from gec_tagger_train.cli import app


def test_build_vocab_command(fixtures_dir: Path, tmp_out: Path) -> None:
    runner = CliRunner()
    out = tmp_out / "tags.json"
    result = runner.invoke(
        app,
        [
            "build-vocab",
            "--jsonl",
            str(fixtures_dir / "tiny.jsonl"),
            "--out",
            str(out),
            "--min-count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_train_smoke_command(fixtures_dir: Path, tmp_out: Path) -> None:
    runner = CliRunner()
    tags = tmp_out / "tags.json"
    runner.invoke(
        app,
        [
            "build-vocab",
            "--jsonl",
            str(fixtures_dir / "tiny.jsonl"),
            "--out",
            str(tags),
            "--min-count",
            "1",
        ],
    )
    out = tmp_out / "ckpt"
    result = runner.invoke(
        app,
        [
            "train",
            "--stage",
            "1",
            "--jsonl",
            str(fixtures_dir / "tiny.jsonl"),
            "--tags",
            str(tags),
            "--out",
            str(out),
            "--max-steps",
            "1",
            "--per-device-batch-size",
            "2",
            "--max-length",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_dir()
```

- [ ] **Step 2: Run, expect failure**

```
uv run pytest tests/test_cli_smoke.py -v
```

- [ ] **Step 3: Implementation**

Create `gec-tagger-train/src/gec_tagger_train/cli.py`:

```python
from __future__ import annotations

from pathlib import Path

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

    model = DebertaTagger(num_tags=_infer_num_tags(checkpoint))
    state = _load_state_dict(checkpoint)
    model.load_state_dict(state, strict=False)
    export_tagger_to_onnx(model=model, out_path=out, max_length=max_length, opset=opset)
    typer.echo(f"exported ONNX to {out}")


def _infer_num_tags(checkpoint: Path) -> int:
    import torch

    state = _load_state_dict(checkpoint)
    return int(state["classifier.weight"].shape[0])


def _load_state_dict(checkpoint: Path) -> dict:
    import torch

    bin_path = checkpoint / "pytorch_model.bin"
    safe_path = checkpoint / "model.safetensors"
    if safe_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safe_path))
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model weights found under {checkpoint}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run, expect PASS**

```
uv run pytest tests/test_cli_smoke.py -v
```

Expected: 2 PASS. Slow due to actual training step.

- [ ] **Step 5: Commit**

```
git add gec-tagger-train/src/gec_tagger_train/cli.py gec-tagger-train/tests/test_cli_smoke.py
git commit -m "feat(gec-tagger-train): typer CLI (build-vocab, train, export)"
```

---

## Task 12: Lint/type/coverage gates

**Files:** none new.

- [ ] **Step 1: ruff**

```
cd gec-tagger-train && uv run ruff check src tests
```

Must exit 0.

- [ ] **Step 2: mypy strict**

```
cd gec-tagger-train && uv run mypy src
```

Must exit 0.

- [ ] **Step 3: pytest with coverage**

```
cd gec-tagger-train && uv run pytest --cov=gec_tagger_train --cov-report=term-missing
```

Coverage target: ≥ 70% on `src/gec_tagger_train`. (Lower than data-pipeline's 85% because slow ML tests are inherently fewer; report deficit if below.)

- [ ] **Step 4: Commit any cleanup**

```
git status
git diff
git add -A
git diff --cached --quiet && echo "clean" || git commit -m "chore(gec-tagger-train): lint/type/coverage cleanup"
```

---

## Notes

- Training is intentionally not run end-to-end as part of this plan. The CLI's `train` command produces a checkpoint when invoked manually with realistic data, but CI/test runs use `--max-steps 1` smoke runs only.
- The full multi-stage recipe (C4_200M pretrain → BEA-2019 → W&I+L finish) is a manual operator playbook documented in the README. It is not a CI workflow.
- M² scorer integration with the upstream tool is left as a follow-up: the `eval_m2.py` module emits valid M² blocks, but invoking the actual `m2scorer` script and parsing its stdout is its own task best handled when CoNLL-2014 official files are present.
