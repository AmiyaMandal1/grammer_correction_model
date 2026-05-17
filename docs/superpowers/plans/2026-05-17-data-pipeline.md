# Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package `data-pipeline/` that ingests public GEC and style-transfer corpora and emits two artifacts: `gec_tagger.jsonl` (token/edit-tag pairs for the DeBERTa GECToR tagger) and `style_sft.jsonl` (ChatML instruction/output pairs for the Qwen2.5 style LLM), with a `manifest.json` recording source hashes, row counts, dedup stats, and a leakage report.

**Architecture:** Pluggable source loaders in `src/data_pipeline/sources/` each yield canonical `Pair(src, tgt, meta)` records. Core modules apply NFKC normalization, ERRANT alignment, GECToR tag encoding, exact + near-duplicate dedup, and cross-set leakage check. Two builders (`gec_builder`, `sft_builder`) consume normalized pairs and write JSONL. A single CLI orchestrates the run.

**Tech Stack:** Python 3.11, `uv` for env + lockfile, `pytest` for tests, `ruff` + `mypy` for lint/typing, `polars` for tabular ops, `datasets` for Hugging Face data, `errant` for grammar-error alignment, `spacy` (en_core_web_sm) as ERRANT's parser, `xxhash` for fast hashing, `typer` for CLI.

---

## Layout

```
data-pipeline/
├── pyproject.toml
├── README.md
├── .python-version
├── .gitignore
├── src/data_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   ├── normalize.py
│   ├── tag_encoder.py
│   ├── errant_align.py
│   ├── dedup.py
│   ├── leakage.py
│   ├── manifest.py
│   ├── gec_builder.py
│   ├── sft_builder.py
│   ├── cli.py
│   └── sources/
│       ├── __init__.py
│       ├── base.py
│       ├── bea2019.py
│       ├── jfleg.py
│       ├── conll2014.py
│       ├── c4_200m.py
│       ├── gyafc.py
│       ├── paradetox.py
│       └── wiki_auto.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── tiny.m2
    │   └── tiny_pairs.jsonl
    ├── test_normalize.py
    ├── test_tag_encoder.py
    ├── test_errant_align.py
    ├── test_dedup.py
    ├── test_leakage.py
    ├── test_manifest.py
    ├── test_sources_bea2019.py
    ├── test_sources_jfleg.py
    ├── test_sources_conll2014.py
    ├── test_sources_c4_200m.py
    ├── test_sources_gyafc.py
    ├── test_sources_paradetox.py
    ├── test_sources_wiki_auto.py
    ├── test_gec_builder.py
    ├── test_sft_builder.py
    └── test_cli_end_to_end.py
```

`data/raw/`, `data/interim/`, `data/processed/` are runtime directories and are gitignored.

---

## Task 1: Bootstrap project

**Files:**
- Create: `data-pipeline/pyproject.toml`
- Create: `data-pipeline/.python-version`
- Create: `data-pipeline/.gitignore`
- Create: `data-pipeline/README.md`
- Create: `data-pipeline/src/data_pipeline/__init__.py`
- Create: `data-pipeline/tests/conftest.py`

- [ ] **Step 1: Create `data-pipeline/pyproject.toml`**

```toml
[project]
name = "data-pipeline"
version = "0.1.0"
description = "Corpus preparation for GEC tagger + style LLM training"
requires-python = ">=3.11,<3.12"
dependencies = [
    "typer>=0.12",
    "polars>=1.0",
    "datasets>=2.20",
    "xxhash>=3.4",
    "spacy>=3.7,<3.8",
    "errant>=3.0.0",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
data-pipeline = "data_pipeline.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_pipeline"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `data-pipeline/.python-version`**

```
3.11
```

- [ ] **Step 3: Create `data-pipeline/.gitignore`**

```
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
```

- [ ] **Step 4: Create `data-pipeline/README.md`**

```markdown
# data-pipeline

Builds `gec_tagger.jsonl` and `style_sft.jsonl` from public corpora.

## Setup

```
cd data-pipeline
uv sync --extra dev
uv run python -m spacy download en_core_web_sm
```

## Run

```
uv run data-pipeline build-all --out data/processed/
```

Outputs:
- `data/processed/gec_tagger.jsonl`
- `data/processed/style_sft.jsonl`
- `data/processed/manifest.json`
```

- [ ] **Step 5: Create `data-pipeline/src/data_pipeline/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 6: Create `data-pipeline/tests/conftest.py`**

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

- [ ] **Step 7: Install deps and verify**

Run from `data-pipeline/`:

```
uv sync --extra dev
uv run python -m spacy download en_core_web_sm
uv run pytest -q
```

Expected: pytest reports `no tests ran`, exit 0.

- [ ] **Step 8: Commit**

```
git add data-pipeline/
git commit -m "feat(data-pipeline): bootstrap project skeleton"
```

---

## Task 2: Define canonical types

**Files:**
- Create: `data-pipeline/src/data_pipeline/types.py`
- Create: `data-pipeline/tests/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_types.py`:

```python
from data_pipeline.types import EditTag, Pair, Split, StyleKind, TagKind


def test_pair_round_trip() -> None:
    p = Pair(src="he go home", tgt="he goes home", source="bea2019", split=Split.TRAIN)
    d = p.to_dict()
    assert d == {
        "src": "he go home",
        "tgt": "he goes home",
        "source": "bea2019",
        "split": "train",
        "meta": {},
    }
    p2 = Pair.from_dict(d)
    assert p2 == p


def test_edit_tag_construct() -> None:
    t = EditTag(kind=TagKind.REPLACE, value="goes")
    assert t.to_str() == "$REPLACE_goes"


def test_edit_tag_keep() -> None:
    t = EditTag(kind=TagKind.KEEP, value=None)
    assert t.to_str() == "$KEEP"


def test_style_kind_values() -> None:
    assert StyleKind.FORMAL.value == "formal"
    assert StyleKind.CONCISE.value == "concise"
```

- [ ] **Step 2: Run test to verify it fails**

Run from `data-pipeline/`:

```
uv run pytest tests/test_types.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.types`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class TagKind(str, Enum):
    KEEP = "KEEP"
    DELETE = "DELETE"
    APPEND = "APPEND"
    REPLACE = "REPLACE"
    TRANSFORM_CASE = "TRANSFORM_CASE"
    TRANSFORM_VERB = "TRANSFORM_VERB"


class StyleKind(str, Enum):
    FORMAL = "formal"
    INFORMAL = "informal"
    CONCISE = "concise"
    SIMPLIFY = "simplify"
    DETOXIFY = "detoxify"


@dataclass(frozen=True, slots=True)
class EditTag:
    kind: TagKind
    value: str | None

    def to_str(self) -> str:
        if self.kind is TagKind.KEEP:
            return "$KEEP"
        if self.kind is TagKind.DELETE:
            return "$DELETE"
        if self.value is None:
            raise ValueError(f"tag {self.kind} requires a value")
        return f"${self.kind.value}_{self.value}"


@dataclass
class Pair:
    src: str
    tgt: str
    source: str
    split: Split
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "tgt": self.tgt,
            "source": self.source,
            "split": self.split.value,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pair":
        return cls(
            src=d["src"],
            tgt=d["tgt"],
            source=d["source"],
            split=Split(d["split"]),
            meta=d.get("meta", {}),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_types.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/types.py data-pipeline/tests/test_types.py
git commit -m "feat(data-pipeline): add canonical Pair/EditTag/StyleKind types"
```

---

## Task 3: Text normalization

**Files:**
- Create: `data-pipeline/src/data_pipeline/normalize.py`
- Create: `data-pipeline/tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_normalize.py`:

```python
from data_pipeline.normalize import normalize_text


def test_collapses_whitespace() -> None:
    assert normalize_text("hello   world\t\t!") == "hello world !"


def test_nfkc_compatibility() -> None:
    # fullwidth digits to ascii
    assert normalize_text("test ２０２３") == "test 2023"


def test_strips_zero_width() -> None:
    assert normalize_text("ab​cd") == "abcd"


def test_unifies_quotes_and_dashes() -> None:
    assert normalize_text("“hello” — world") == '"hello" - world'


def test_strips_outer_whitespace() -> None:
    assert normalize_text("  hi  ") == "hi"


def test_empty_input() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_normalize.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.normalize`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/normalize.py`:

```python
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = {"​", "‌", "‍", "﻿"}
_QUOTE_MAP = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
}
_DASH_MAP = {"—": "-", "–": "-", "−": "-"}
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in _ZERO_WIDTH)
    text = text.translate(str.maketrans({**_QUOTE_MAP, **_DASH_MAP}))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_normalize.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/normalize.py data-pipeline/tests/test_normalize.py
git commit -m "feat(data-pipeline): text normalization (NFKC, quotes, dashes, whitespace)"
```

---

## Task 4: GECToR tag encoder

**Files:**
- Create: `data-pipeline/src/data_pipeline/tag_encoder.py`
- Create: `data-pipeline/tests/test_tag_encoder.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_tag_encoder.py`:

```python
from data_pipeline.tag_encoder import (
    align_tokens_to_tags,
    decode_tag,
    encode_tag,
)
from data_pipeline.types import EditTag, TagKind


def test_encode_decode_keep() -> None:
    t = EditTag(kind=TagKind.KEEP, value=None)
    s = encode_tag(t)
    assert s == "$KEEP"
    assert decode_tag(s) == t


def test_encode_decode_replace() -> None:
    t = EditTag(kind=TagKind.REPLACE, value="goes")
    assert encode_tag(t) == "$REPLACE_goes"
    assert decode_tag("$REPLACE_goes") == t


def test_decode_append_with_underscore_in_value() -> None:
    assert decode_tag("$APPEND_well_done") == EditTag(
        kind=TagKind.APPEND, value="well_done"
    )


def test_align_tokens_to_tags_keep_all_when_equal() -> None:
    src = ["I", "am", "happy"]
    tgt = ["I", "am", "happy"]
    tags = align_tokens_to_tags(src, tgt)
    assert tags == [
        EditTag(TagKind.KEEP, None),
        EditTag(TagKind.KEEP, None),
        EditTag(TagKind.KEEP, None),
    ]


def test_align_tokens_replace_one() -> None:
    src = ["he", "go", "home"]
    tgt = ["he", "goes", "home"]
    tags = align_tokens_to_tags(src, tgt)
    assert tags == [
        EditTag(TagKind.KEEP, None),
        EditTag(TagKind.REPLACE, "goes"),
        EditTag(TagKind.KEEP, None),
    ]


def test_align_tokens_delete() -> None:
    src = ["he", "the", "goes", "home"]
    tgt = ["he", "goes", "home"]
    tags = align_tokens_to_tags(src, tgt)
    assert tags[1] == EditTag(TagKind.DELETE, None)


def test_align_tokens_append() -> None:
    src = ["he", "home"]
    tgt = ["he", "goes", "home"]
    tags = align_tokens_to_tags(src, tgt)
    # APPEND on previous token
    assert tags[0] == EditTag(TagKind.APPEND, "goes")
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_tag_encoder.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.tag_encoder`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/tag_encoder.py`:

```python
from __future__ import annotations

from difflib import SequenceMatcher

from data_pipeline.types import EditTag, TagKind


def encode_tag(tag: EditTag) -> str:
    return tag.to_str()


def decode_tag(s: str) -> EditTag:
    if not s.startswith("$"):
        raise ValueError(f"not a tag: {s!r}")
    body = s[1:]
    if body == "KEEP":
        return EditTag(TagKind.KEEP, None)
    if body == "DELETE":
        return EditTag(TagKind.DELETE, None)
    head, _, value = body.partition("_")
    if not value:
        raise ValueError(f"tag missing value: {s!r}")
    try:
        kind = TagKind(head)
    except ValueError as e:
        raise ValueError(f"unknown tag kind: {head}") from e
    return EditTag(kind=kind, value=value)


def align_tokens_to_tags(src: list[str], tgt: list[str]) -> list[EditTag]:
    """Single-pass token-level alignment.

    The full GECToR scheme is iterative; this function performs one round
    of edit derivation. Iterative re-tagging is handled by the tagger model
    at inference time.
    """
    tags: list[EditTag] = [EditTag(TagKind.KEEP, None) for _ in src]
    if not src:
        return tags
    matcher = SequenceMatcher(a=src, b=tgt, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            for k, src_idx in enumerate(range(i1, i2)):
                if j1 + k < j2:
                    tags[src_idx] = EditTag(TagKind.REPLACE, tgt[j1 + k])
                else:
                    tags[src_idx] = EditTag(TagKind.DELETE, None)
        elif op == "delete":
            for src_idx in range(i1, i2):
                tags[src_idx] = EditTag(TagKind.DELETE, None)
        elif op == "insert":
            # Attach inserts to the preceding source token via APPEND.
            anchor = max(i1 - 1, 0)
            for j_idx in range(j1, j2):
                # Multiple inserts on the same anchor: only the first survives
                # in single-pass mode. Tagger model iterates to capture more.
                if tags[anchor].kind is TagKind.KEEP:
                    tags[anchor] = EditTag(TagKind.APPEND, tgt[j_idx])
                    break
    return tags
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_tag_encoder.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/tag_encoder.py data-pipeline/tests/test_tag_encoder.py
git commit -m "feat(data-pipeline): GECToR single-pass tag encoder"
```

---

## Task 5: ERRANT alignment wrapper

**Files:**
- Create: `data-pipeline/src/data_pipeline/errant_align.py`
- Create: `data-pipeline/tests/test_errant_align.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_errant_align.py`:

```python
import pytest

from data_pipeline.errant_align import ErrantAligner, tokenize


def test_tokenize_simple() -> None:
    aligner = ErrantAligner()
    toks = tokenize("He go home.", aligner.nlp)
    assert toks == ["He", "go", "home", "."]


def test_align_pair_returns_token_aligned_output() -> None:
    aligner = ErrantAligner()
    src, tgt = aligner.align_pair("He go home.", "He goes home.")
    assert src == ["He", "go", "home", "."]
    assert tgt == ["He", "goes", "home", "."]


def test_align_pair_handles_insertion() -> None:
    aligner = ErrantAligner()
    src, tgt = aligner.align_pair("He home.", "He goes home.")
    assert "goes" in tgt
    assert len(tgt) > len(src)


def test_align_pair_empty_target_raises() -> None:
    aligner = ErrantAligner()
    with pytest.raises(ValueError):
        aligner.align_pair("hello", "")
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_errant_align.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.errant_align`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/errant_align.py`:

```python
from __future__ import annotations

from functools import lru_cache

import errant
import spacy


def tokenize(text: str, nlp: "spacy.language.Language") -> list[str]:
    doc = nlp.make_doc(text)
    return [t.text for t in doc]


@lru_cache(maxsize=1)
def _load_nlp() -> "spacy.language.Language":
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


class ErrantAligner:
    """Thin wrapper over ERRANT that exposes token-aligned source/target.

    ERRANT itself runs a more elaborate edit extraction; we only need the
    token sequences that match what the downstream tag encoder consumes.
    """

    def __init__(self) -> None:
        self.nlp = _load_nlp()
        self.annotator = errant.load("en", self.nlp)

    def align_pair(self, src: str, tgt: str) -> tuple[list[str], list[str]]:
        if not src or not tgt:
            raise ValueError("source and target must be non-empty")
        src_doc = self.annotator.parse(src)
        tgt_doc = self.annotator.parse(tgt)
        return [t.text for t in src_doc], [t.text for t in tgt_doc]
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_errant_align.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/errant_align.py data-pipeline/tests/test_errant_align.py
git commit -m "feat(data-pipeline): ERRANT-backed tokenizer/aligner wrapper"
```

---

## Task 6: Dedup module

**Files:**
- Create: `data-pipeline/src/data_pipeline/dedup.py`
- Create: `data-pipeline/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_dedup.py`:

```python
from data_pipeline.dedup import DedupStats, dedupe_pairs
from data_pipeline.types import Pair, Split


def _p(src: str, tgt: str) -> Pair:
    return Pair(src=src, tgt=tgt, source="t", split=Split.TRAIN)


def test_dedupe_exact_duplicates_removed() -> None:
    pairs = [
        _p("a", "b"),
        _p("a", "b"),
        _p("c", "d"),
    ]
    out, stats = dedupe_pairs(pairs)
    assert len(out) == 2
    assert stats == DedupStats(input_count=3, output_count=2, removed_exact=1)


def test_dedupe_keeps_first_seen() -> None:
    pairs = [
        _p("hello", "world"),
        _p("HELLO", "WORLD"),  # case-different but normalized identical
    ]
    out, _ = dedupe_pairs(pairs)
    assert out[0].src == "hello"
    assert len(out) == 1


def test_dedupe_empty_input() -> None:
    out, stats = dedupe_pairs([])
    assert out == []
    assert stats.input_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_dedup.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.dedup`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/dedup.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import xxhash

from data_pipeline.types import Pair


@dataclass(frozen=True)
class DedupStats:
    input_count: int
    output_count: int
    removed_exact: int


def _key(p: Pair) -> int:
    h = xxhash.xxh64()
    h.update(p.src.lower().encode("utf-8"))
    h.update(b"\x1f")
    h.update(p.tgt.lower().encode("utf-8"))
    return h.intdigest()


def dedupe_pairs(pairs: Iterable[Pair]) -> tuple[list[Pair], DedupStats]:
    seen: set[int] = set()
    out: list[Pair] = []
    in_count = 0
    for p in pairs:
        in_count += 1
        k = _key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out, DedupStats(
        input_count=in_count,
        output_count=len(out),
        removed_exact=in_count - len(out),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_dedup.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/dedup.py data-pipeline/tests/test_dedup.py
git commit -m "feat(data-pipeline): exact dedup keyed on lowercased (src, tgt)"
```

---

## Task 7: Leakage check

**Files:**
- Create: `data-pipeline/src/data_pipeline/leakage.py`
- Create: `data-pipeline/tests/test_leakage.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_leakage.py`:

```python
import pytest

from data_pipeline.leakage import LeakageError, check_leakage
from data_pipeline.types import Pair, Split


def _p(src: str, tgt: str, split: Split) -> Pair:
    return Pair(src=src, tgt=tgt, source="t", split=split)


def test_no_leakage_passes() -> None:
    train = [_p("a", "b", Split.TRAIN)]
    dev = [_p("c", "d", Split.DEV)]
    check_leakage(train=train, eval_sets={"dev": dev})


def test_train_dev_overlap_raises() -> None:
    train = [_p("hello world", "Hello world.", Split.TRAIN)]
    dev = [_p("hello world", "Hello world.", Split.DEV)]
    with pytest.raises(LeakageError) as exc:
        check_leakage(train=train, eval_sets={"dev": dev})
    assert "dev" in str(exc.value)
    assert "1" in str(exc.value)


def test_train_test_overlap_raises() -> None:
    train = [_p("foo", "foo.", Split.TRAIN)]
    test = [_p("FOO", "foo.", Split.TEST)]  # case-insensitive match on src
    with pytest.raises(LeakageError):
        check_leakage(train=train, eval_sets={"test": test})
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_leakage.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.leakage`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/leakage.py`:

```python
from __future__ import annotations

from typing import Mapping

import xxhash

from data_pipeline.types import Pair


class LeakageError(RuntimeError):
    pass


def _src_key(p: Pair) -> int:
    return xxhash.xxh64(p.src.lower().encode("utf-8")).intdigest()


def check_leakage(
    *, train: list[Pair], eval_sets: Mapping[str, list[Pair]]
) -> None:
    train_keys = {_src_key(p) for p in train}
    for name, items in eval_sets.items():
        overlap = sum(1 for p in items if _src_key(p) in train_keys)
        if overlap > 0:
            raise LeakageError(
                f"train ∩ {name} = {overlap} rows (source side, case-insensitive)"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_leakage.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/leakage.py data-pipeline/tests/test_leakage.py
git commit -m "feat(data-pipeline): cross-split leakage check"
```

---

## Task 8: Manifest writer

**Files:**
- Create: `data-pipeline/src/data_pipeline/manifest.py`
- Create: `data-pipeline/tests/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_manifest.py`:

```python
import json
from pathlib import Path

from data_pipeline.manifest import ManifestBuilder


def test_manifest_records_files_and_counts(tmp_path: Path) -> None:
    target = tmp_path / "file.jsonl"
    target.write_text('{"a":1}\n{"a":2}\n')
    mb = ManifestBuilder(out_dir=tmp_path)
    mb.add_artifact(name="gec_tagger", path=target, row_count=2)
    mb.add_source(name="bea2019", row_count=1000, retained=950)
    mb.set_dedup(removed_exact=12)
    mb.set_leakage_passed(True)
    out = mb.write()
    data = json.loads(out.read_text())
    assert data["artifacts"]["gec_tagger"]["row_count"] == 2
    assert "sha256" in data["artifacts"]["gec_tagger"]
    assert data["sources"]["bea2019"] == {"row_count": 1000, "retained": 950}
    assert data["dedup"]["removed_exact"] == 12
    assert data["leakage"]["passed"] is True


def test_manifest_includes_timestamp(tmp_path: Path) -> None:
    mb = ManifestBuilder(out_dir=tmp_path)
    out = mb.write()
    data = json.loads(out.read_text())
    assert "generated_at" in data
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.manifest`.

- [ ] **Step 3: Write minimal implementation**

Create `data-pipeline/src/data_pipeline/manifest.py`:

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ManifestBuilder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, dict[str, int]] = {}
        self._dedup: dict[str, int] = {}
        self._leakage_passed: bool | None = None  # None = not run

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def add_artifact(self, *, name: str, path: Path, row_count: int) -> None:
        self._artifacts[name] = {
            "path": str(path.relative_to(self.out_dir)),
            "row_count": row_count,
            "sha256": self._sha256(path),
            "bytes": path.stat().st_size,
        }

    def add_source(self, *, name: str, row_count: int, retained: int) -> None:
        self._sources[name] = {"row_count": row_count, "retained": retained}

    def set_dedup(self, *, removed_exact: int) -> None:
        self._dedup = {"removed_exact": removed_exact}

    def set_leakage_passed(self, passed: bool | None) -> None:
        """`None` means leakage check was not run for this build."""
        self._leakage_passed = passed

    def write(self) -> Path:
        out = self.out_dir / "manifest.json"
        payload: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": self._artifacts,
            "sources": self._sources,
            "dedup": self._dedup,
            "leakage": {"passed": self._leakage_passed},
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_manifest.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/manifest.py data-pipeline/tests/test_manifest.py
git commit -m "feat(data-pipeline): manifest writer with sha256 + provenance"
```

---

## Task 9: Source loader base + config

**Files:**
- Create: `data-pipeline/src/data_pipeline/config.py`
- Create: `data-pipeline/src/data_pipeline/sources/__init__.py`
- Create: `data-pipeline/src/data_pipeline/sources/base.py`
- Create: `data-pipeline/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_config.py`:

```python
from pathlib import Path

from data_pipeline.config import PipelineConfig


def test_default_config_paths(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path)
    assert cfg.raw_dir == tmp_path / "data" / "raw"
    assert cfg.interim_dir == tmp_path / "data" / "interim"
    assert cfg.processed_dir == tmp_path / "data" / "processed"


def test_config_ensures_dirs(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    assert cfg.raw_dir.is_dir()
    assert cfg.interim_dir.is_dir()
    assert cfg.processed_dir.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.config`.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    def ensure_dirs(self) -> "PipelineConfig":
        for d in (self.raw_dir, self.interim_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
```

Create `data-pipeline/src/data_pipeline/sources/__init__.py`:

```python
from data_pipeline.sources.base import Source

__all__ = ["Source"]
```

Create `data-pipeline/src/data_pipeline/sources/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from data_pipeline.config import PipelineConfig
from data_pipeline.types import Pair


class Source(ABC):
    """Abstract source loader.

    Implementations read raw files from `config.raw_dir / self.name` and
    yield `Pair` records. They do not perform normalization or dedup;
    those happen in the builder stage.
    """

    name: str

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @property
    def raw_path(self) -> "Path":
        return self.config.raw_dir / self.name

    @abstractmethod
    def iter_pairs(self) -> Iterator[Pair]: ...


from pathlib import Path  # noqa: E402  (forward-ref typing only)
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_config.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/config.py data-pipeline/src/data_pipeline/sources/ data-pipeline/tests/test_config.py
git commit -m "feat(data-pipeline): pipeline config + Source abstract base"
```

---

## Task 10: BEA-2019 source loader (M² format)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/bea2019.py`
- Create: `data-pipeline/tests/test_sources_bea2019.py`
- Create: `data-pipeline/tests/fixtures/tiny.m2`

- [ ] **Step 1: Create fixture**

Create `data-pipeline/tests/fixtures/tiny.m2`:

```
S He go to school .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0

S I are happy .
A 1 2|||R:VERB:SVA|||am|||REQUIRED|||-NONE-|||0

S Hello world .
```

- [ ] **Step 2: Write the failing test**

Create `data-pipeline/tests/test_sources_bea2019.py`:

```python
import shutil
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.bea2019 import BEA2019Source
from data_pipeline.types import Split


def test_loads_m2_pairs(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src_dir = cfg.raw_dir / "bea2019"
    src_dir.mkdir()
    shutil.copy(fixtures_dir / "tiny.m2", src_dir / "train.m2")

    src = BEA2019Source(config=cfg, split=Split.TRAIN)
    pairs = list(src.iter_pairs())

    assert len(pairs) == 3
    assert pairs[0].src == "He go to school ."
    assert pairs[0].tgt == "He goes to school ."
    assert pairs[1].src == "I are happy ."
    assert pairs[1].tgt == "I am happy ."
    # third sentence has no edits; src == tgt
    assert pairs[2].src == pairs[2].tgt == "Hello world ."
    assert all(p.source == "bea2019" for p in pairs)
    assert all(p.split == Split.TRAIN for p in pairs)


def test_missing_file_raises(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = BEA2019Source(config=cfg, split=Split.TRAIN)
    import pytest

    with pytest.raises(FileNotFoundError):
        list(src.iter_pairs())
```

- [ ] **Step 3: Run test to verify it fails**

```
uv run pytest tests/test_sources_bea2019.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.sources.bea2019`.

- [ ] **Step 4: Write the implementation**

Create `data-pipeline/src/data_pipeline/sources/bea2019.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split


class BEA2019Source(Source):
    name = "bea2019"

    def __init__(self, config: PipelineConfig, split: Split) -> None:
        super().__init__(config)
        self.split = split

    def _m2_file(self) -> Path:
        fname = {
            Split.TRAIN: "train.m2",
            Split.DEV: "dev.m2",
            Split.TEST: "test.m2",
        }[self.split]
        return self.raw_path / fname

    def iter_pairs(self) -> Iterator[Pair]:
        path = self._m2_file()
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} (download BEA-2019 to {self.raw_path})"
            )
        for src_toks, edits in _parse_m2(path):
            tgt_toks = _apply_edits(src_toks, edits)
            yield Pair(
                src=" ".join(src_toks),
                tgt=" ".join(tgt_toks),
                source=self.name,
                split=self.split,
            )


def _parse_m2(path: Path) -> Iterator[tuple[list[str], list[tuple[int, int, str]]]]:
    src_toks: list[str] | None = None
    edits: list[tuple[int, int, str]] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if src_toks is not None:
                    yield src_toks, edits
                src_toks = line[2:].split(" ")
                edits = []
            elif line.startswith("A "):
                # A start end|||type|||replacement|||REQUIRED|||-NONE-|||annotator
                parts = line[2:].split("|||")
                span = parts[0].split(" ")
                start, end = int(span[0]), int(span[1])
                if start == -1:
                    continue  # noop
                replacement = parts[2]
                edits.append((start, end, replacement))
            elif line == "":
                if src_toks is not None:
                    yield src_toks, edits
                    src_toks, edits = None, []
        if src_toks is not None:
            yield src_toks, edits


def _apply_edits(
    src_toks: list[str], edits: list[tuple[int, int, str]]
) -> list[str]:
    if not edits:
        return list(src_toks)
    # Apply edits in right-to-left order to keep offsets stable.
    out = list(src_toks)
    for start, end, replacement in sorted(edits, key=lambda e: -e[0]):
        repl = replacement.split(" ") if replacement else []
        out[start:end] = repl
    return out
```

- [ ] **Step 5: Run test to verify it passes**

```
uv run pytest tests/test_sources_bea2019.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```
git add data-pipeline/src/data_pipeline/sources/bea2019.py data-pipeline/tests/test_sources_bea2019.py data-pipeline/tests/fixtures/tiny.m2
git commit -m "feat(data-pipeline): BEA-2019 M² source loader"
```

---

## Task 11: JFLEG source loader (HF datasets)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/jfleg.py`
- Create: `data-pipeline/tests/test_sources_jfleg.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sources_jfleg.py`:

```python
from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.jfleg import JFLEGSource
from data_pipeline.types import Split


def _fake_hf_load(_name: str, split: str):
    # Mimic JFLEG row shape from Hugging Face: {sentence, corrections: [..]}
    return [
        {"sentence": "He go home.", "corrections": ["He goes home.", "He went home."]},
        {"sentence": "I love programs.", "corrections": ["I love programming."]},
    ]


def test_jfleg_expands_corrections(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = JFLEGSource(config=cfg, split=Split.DEV)
    with patch("data_pipeline.sources.jfleg.load_dataset", side_effect=_fake_hf_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 3  # 2 + 1
    assert pairs[0].src == "He go home."
    assert pairs[0].tgt == "He goes home."
    assert pairs[1].src == "He go home."
    assert pairs[1].tgt == "He went home."
    assert all(p.source == "jfleg" for p in pairs)
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sources_jfleg.py -v
```

Expected: FAIL with `ModuleNotFoundError: data_pipeline.sources.jfleg`.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sources/jfleg.py`:

```python
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split


class JFLEGSource(Source):
    name = "jfleg"

    def __init__(self, config: PipelineConfig, split: Split) -> None:
        super().__init__(config)
        self.split = split

    def iter_pairs(self) -> Iterator[Pair]:
        split_name = {Split.DEV: "validation", Split.TEST: "test"}.get(self.split)
        if split_name is None:
            raise ValueError(f"JFLEG has no {self.split.value} split")
        rows = load_dataset("jfleg", split=split_name)
        for row in rows:
            sentence = row["sentence"]
            for correction in row["corrections"]:
                yield Pair(
                    src=sentence,
                    tgt=correction,
                    source=self.name,
                    split=self.split,
                )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sources_jfleg.py -v
```

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sources/jfleg.py data-pipeline/tests/test_sources_jfleg.py
git commit -m "feat(data-pipeline): JFLEG source loader via HF datasets"
```

---

## Task 12: CoNLL-2014 source loader (M² format on disk)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/conll2014.py`
- Create: `data-pipeline/tests/test_sources_conll2014.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sources_conll2014.py`:

```python
import shutil
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.conll2014 import CoNLL2014Source
from data_pipeline.types import Split


def test_loads_conll_m2(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src_dir = cfg.raw_dir / "conll2014"
    src_dir.mkdir()
    shutil.copy(fixtures_dir / "tiny.m2", src_dir / "official-2014.0.m2")

    src = CoNLL2014Source(config=cfg)
    pairs = list(src.iter_pairs())
    assert len(pairs) == 3
    assert all(p.split == Split.TEST for p in pairs)
    assert all(p.source == "conll2014" for p in pairs)
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sources_conll2014.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sources/conll2014.py`:

```python
from __future__ import annotations

from typing import Iterator

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.sources.bea2019 import _apply_edits, _parse_m2
from data_pipeline.types import Pair, Split


class CoNLL2014Source(Source):
    name = "conll2014"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def iter_pairs(self) -> Iterator[Pair]:
        path = self.raw_path / "official-2014.0.m2"
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} (download CoNLL-2014 to {self.raw_path})"
            )
        for src_toks, edits in _parse_m2(path):
            tgt_toks = _apply_edits(src_toks, edits)
            yield Pair(
                src=" ".join(src_toks),
                tgt=" ".join(tgt_toks),
                source=self.name,
                split=Split.TEST,
            )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sources_conll2014.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sources/conll2014.py data-pipeline/tests/test_sources_conll2014.py
git commit -m "feat(data-pipeline): CoNLL-2014 M² loader (eval split)"
```

---

## Task 13: C4_200M source loader (HF, capped)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/c4_200m.py`
- Create: `data-pipeline/tests/test_sources_c4_200m.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sources_c4_200m.py`:

```python
from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.c4_200m import C4200MSource
from data_pipeline.types import Split


class _FakeStream:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


def _fake_load(_name: str, split: str, streaming: bool):
    assert streaming is True
    return _FakeStream(
        [
            {"input": "He go home.", "output": "He goes home."},
            {"input": "She wnt there.", "output": "She went there."},
            {"input": "third row src", "output": "third row tgt"},
            {"input": "fourth row src", "output": "fourth row tgt"},
        ]
    )


def test_c4_200m_streams_and_caps(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = C4200MSource(config=cfg, max_rows=2)
    with patch("data_pipeline.sources.c4_200m.load_dataset", side_effect=_fake_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].src == "He go home."
    assert pairs[0].tgt == "He goes home."
    assert all(p.source == "c4_200m" for p in pairs)
    assert all(p.split == Split.TRAIN for p in pairs)
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sources_c4_200m.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sources/c4_200m.py`:

```python
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split


class C4200MSource(Source):
    """Streaming loader for the C4_200M synthetic GEC dataset.

    HF dataset id: `liweili/c4_200m`. Streams to avoid downloading the
    full set; capped at `max_rows` for development and small-scale runs.
    """

    name = "c4_200m"

    def __init__(self, config: PipelineConfig, max_rows: int = 2_000_000) -> None:
        super().__init__(config)
        self.max_rows = max_rows

    def iter_pairs(self) -> Iterator[Pair]:
        ds = load_dataset("liweili/c4_200m", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= self.max_rows:
                break
            yield Pair(
                src=row["input"],
                tgt=row["output"],
                source=self.name,
                split=Split.TRAIN,
            )
            count += 1
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sources_c4_200m.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sources/c4_200m.py data-pipeline/tests/test_sources_c4_200m.py
git commit -m "feat(data-pipeline): C4_200M streaming source loader with cap"
```

---

## Task 14: GYAFC source loader (formality, license-gated)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/gyafc.py`
- Create: `data-pipeline/tests/test_sources_gyafc.py`

GYAFC requires a license; loader reads pre-placed `informal` and `formal` parallel files from disk.

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sources_gyafc.py`:

```python
from pathlib import Path

import pytest

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.gyafc import GYAFCSource
from data_pipeline.types import Split, StyleKind


def test_loads_parallel_files(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    d = cfg.raw_dir / "gyafc" / "Family_Relationships" / "train"
    d.mkdir(parents=True)
    (d / "informal").write_text("yo whats up\nidk man\n")
    (d / "formal").write_text("Hello, how are you?\nI am uncertain.\n")

    src = GYAFCSource(config=cfg, split=Split.TRAIN, domain="Family_Relationships")
    pairs = list(src.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].meta["style_target"] == StyleKind.FORMAL.value
    assert pairs[0].src == "yo whats up"
    assert pairs[0].tgt == "Hello, how are you?"
    assert pairs[0].source == "gyafc"


def test_missing_files_raise(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = GYAFCSource(config=cfg, split=Split.TRAIN, domain="Family_Relationships")
    with pytest.raises(FileNotFoundError):
        list(src.iter_pairs())


def test_mismatched_lengths_raise(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    d = cfg.raw_dir / "gyafc" / "Family_Relationships" / "train"
    d.mkdir(parents=True)
    (d / "informal").write_text("a\nb\n")
    (d / "formal").write_text("A\n")
    src = GYAFCSource(config=cfg, split=Split.TRAIN, domain="Family_Relationships")
    with pytest.raises(ValueError):
        list(src.iter_pairs())
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sources_gyafc.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sources/gyafc.py`:

```python
from __future__ import annotations

from typing import Iterator, Literal

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split, StyleKind


Domain = Literal["Family_Relationships", "Entertainment_Music"]


class GYAFCSource(Source):
    name = "gyafc"

    def __init__(
        self,
        config: PipelineConfig,
        split: Split,
        domain: Domain,
    ) -> None:
        super().__init__(config)
        self.split = split
        self.domain = domain

    def iter_pairs(self) -> Iterator[Pair]:
        split_dir_name = {
            Split.TRAIN: "train",
            Split.DEV: "tune",
            Split.TEST: "test",
        }[self.split]
        base = self.raw_path / self.domain / split_dir_name
        informal = base / "informal"
        formal = base / "formal"
        if not informal.exists() or not formal.exists():
            raise FileNotFoundError(
                f"expected {informal} and {formal} (place GYAFC files there)"
            )
        with informal.open() as fi, formal.open() as ff:
            informal_lines = [ln.rstrip("\n") for ln in fi]
            formal_lines = [ln.rstrip("\n") for ln in ff]
        if len(informal_lines) != len(formal_lines):
            raise ValueError(
                f"GYAFC parallel length mismatch: {len(informal_lines)} != {len(formal_lines)}"
            )
        for i_line, f_line in zip(informal_lines, formal_lines, strict=True):
            yield Pair(
                src=i_line,
                tgt=f_line,
                source=self.name,
                split=self.split,
                meta={
                    "style_target": StyleKind.FORMAL.value,
                    "domain": self.domain,
                },
            )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sources_gyafc.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sources/gyafc.py data-pipeline/tests/test_sources_gyafc.py
git commit -m "feat(data-pipeline): GYAFC formality source (informal->formal)"
```

---

## Task 15: ParaDetox source loader

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/paradetox.py`
- Create: `data-pipeline/tests/test_sources_paradetox.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sources_paradetox.py`:

```python
from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.paradetox import ParaDetoxSource
from data_pipeline.types import StyleKind


def _fake_load(_name: str, split: str):
    return [
        {"en_toxic_comment": "that is dumb", "en_neutral_comment": "that is not helpful"},
        {"en_toxic_comment": "you suck", "en_neutral_comment": "you are not good at this"},
    ]


def test_paradetox_pairs(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = ParaDetoxSource(config=cfg)
    with patch("data_pipeline.sources.paradetox.load_dataset", side_effect=_fake_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].src == "that is dumb"
    assert pairs[0].tgt == "that is not helpful"
    assert pairs[0].meta["style_target"] == StyleKind.DETOXIFY.value
    assert pairs[0].source == "paradetox"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sources_paradetox.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sources/paradetox.py`:

```python
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split, StyleKind


class ParaDetoxSource(Source):
    name = "paradetox"

    def iter_pairs(self) -> Iterator[Pair]:
        rows = load_dataset("s-nlp/paradetox", split="train")
        for row in rows:
            yield Pair(
                src=row["en_toxic_comment"],
                tgt=row["en_neutral_comment"],
                source=self.name,
                split=Split.TRAIN,
                meta={"style_target": StyleKind.DETOXIFY.value},
            )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sources_paradetox.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sources/paradetox.py data-pipeline/tests/test_sources_paradetox.py
git commit -m "feat(data-pipeline): ParaDetox source (toxic->neutral)"
```

---

## Task 16: Wiki-Auto source loader (simplification)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sources/wiki_auto.py`
- Create: `data-pipeline/tests/test_sources_wiki_auto.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sources_wiki_auto.py`:

```python
from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.wiki_auto import WikiAutoSource
from data_pipeline.types import StyleKind


def _fake_load(_name: str, _subset: str, split: str):
    return [
        {"normal_sentence": "He is an erudite gentleman.", "simple_sentence": "He is a smart man."},
    ]


def test_wiki_auto_pairs(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = WikiAutoSource(config=cfg)
    with patch("data_pipeline.sources.wiki_auto.load_dataset", side_effect=_fake_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 1
    assert pairs[0].src == "He is an erudite gentleman."
    assert pairs[0].tgt == "He is a smart man."
    assert pairs[0].meta["style_target"] == StyleKind.SIMPLIFY.value
    assert pairs[0].source == "wiki_auto"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sources_wiki_auto.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sources/wiki_auto.py`:

```python
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split, StyleKind


class WikiAutoSource(Source):
    name = "wiki_auto"

    def iter_pairs(self) -> Iterator[Pair]:
        rows = load_dataset("wiki_auto", "auto_acl", split="train")
        for row in rows:
            yield Pair(
                src=row["normal_sentence"],
                tgt=row["simple_sentence"],
                source=self.name,
                split=Split.TRAIN,
                meta={"style_target": StyleKind.SIMPLIFY.value},
            )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sources_wiki_auto.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sources/wiki_auto.py data-pipeline/tests/test_sources_wiki_auto.py
git commit -m "feat(data-pipeline): Wiki-Auto simplification source"
```

---

## Task 17: GEC builder (writes `gec_tagger.jsonl`)

**Files:**
- Create: `data-pipeline/src/data_pipeline/gec_builder.py`
- Create: `data-pipeline/tests/test_gec_builder.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_gec_builder.py`:

```python
import json
from pathlib import Path

from data_pipeline.gec_builder import build_gec_jsonl
from data_pipeline.types import Pair, Split


def test_build_gec_jsonl_writes_token_tag_records(tmp_path: Path) -> None:
    pairs = [
        Pair(src="he go home", tgt="he goes home", source="t", split=Split.TRAIN),
        Pair(src="i am happy", tgt="i am happy", source="t", split=Split.TRAIN),
    ]
    out = tmp_path / "gec.jsonl"
    written = build_gec_jsonl(pairs, out)
    assert written == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["tokens"] == ["he", "go", "home"]
    assert rec0["tags"] == ["$KEEP", "$REPLACE_goes", "$KEEP"]
    assert rec0["source"] == "t"
    assert rec0["split"] == "train"
    rec1 = json.loads(lines[1])
    assert rec1["tags"] == ["$KEEP", "$KEEP", "$KEEP"]


def test_build_gec_jsonl_skips_empty_sources(tmp_path: Path) -> None:
    pairs = [Pair(src="", tgt="x", source="t", split=Split.TRAIN)]
    out = tmp_path / "gec.jsonl"
    written = build_gec_jsonl(pairs, out)
    assert written == 0
    assert out.read_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_gec_builder.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/gec_builder.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from data_pipeline.tag_encoder import align_tokens_to_tags, encode_tag
from data_pipeline.types import Pair


def build_gec_jsonl(pairs: Iterable[Pair], out_path: Path) -> int:
    """Write one record per pair: {tokens, tags, source, split, meta}.

    Skips pairs with empty source. Whitespace tokenization is intentional;
    upstream sources already produce ERRANT- or M²-aligned tokens.
    """
    written = 0
    with out_path.open("w") as f:
        for p in pairs:
            src_toks = p.src.split()
            tgt_toks = p.tgt.split()
            if not src_toks:
                continue
            tags = align_tokens_to_tags(src_toks, tgt_toks)
            rec = {
                "tokens": src_toks,
                "tags": [encode_tag(t) for t in tags],
                "source": p.source,
                "split": p.split.value,
                "meta": p.meta,
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_gec_builder.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/gec_builder.py data-pipeline/tests/test_gec_builder.py
git commit -m "feat(data-pipeline): GEC tagger JSONL builder"
```

---

## Task 18: SFT builder (writes `style_sft.jsonl`)

**Files:**
- Create: `data-pipeline/src/data_pipeline/sft_builder.py`
- Create: `data-pipeline/tests/test_sft_builder.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_sft_builder.py`:

```python
import json
from pathlib import Path

from data_pipeline.sft_builder import build_sft_jsonl
from data_pipeline.types import Pair, Split, StyleKind


def test_build_sft_records(tmp_path: Path) -> None:
    pairs = [
        Pair(
            src="yo whats up",
            tgt="Hello, how are you?",
            source="gyafc",
            split=Split.TRAIN,
            meta={"style_target": StyleKind.FORMAL.value},
        ),
        Pair(
            src="that is dumb",
            tgt="that is not helpful",
            source="paradetox",
            split=Split.TRAIN,
            meta={"style_target": StyleKind.DETOXIFY.value},
        ),
    ]
    out = tmp_path / "sft.jsonl"
    written = build_sft_jsonl(pairs, out)
    assert written == 2
    lines = out.read_text().strip().splitlines()
    rec0 = json.loads(lines[0])
    assert rec0["messages"][0]["role"] == "system"
    assert rec0["messages"][1]["role"] == "user"
    assert "yo whats up" in rec0["messages"][1]["content"]
    assert "formal" in rec0["messages"][0]["content"].lower()
    assert rec0["messages"][2]["role"] == "assistant"
    assert rec0["messages"][2]["content"] == "Hello, how are you?"


def test_skips_pairs_without_style_target(tmp_path: Path) -> None:
    pairs = [Pair(src="a", tgt="b", source="t", split=Split.TRAIN, meta={})]
    out = tmp_path / "sft.jsonl"
    assert build_sft_jsonl(pairs, out) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_sft_builder.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/sft_builder.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from data_pipeline.types import Pair, StyleKind

_SYSTEM_PROMPT = {
    StyleKind.FORMAL: "Rewrite the user's text in formal English while preserving meaning.",
    StyleKind.INFORMAL: "Rewrite the user's text in casual, informal English while preserving meaning.",
    StyleKind.CONCISE: "Rewrite the user's text more concisely while preserving meaning.",
    StyleKind.SIMPLIFY: "Rewrite the user's text in simpler English while preserving meaning.",
    StyleKind.DETOXIFY: "Rewrite the user's text in a neutral, non-toxic way while preserving meaning.",
}


def build_sft_jsonl(pairs: Iterable[Pair], out_path: Path) -> int:
    written = 0
    with out_path.open("w") as f:
        for p in pairs:
            target = p.meta.get("style_target")
            if not target:
                continue
            try:
                kind = StyleKind(target)
            except ValueError:
                continue
            rec = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT[kind]},
                    {"role": "user", "content": p.src},
                    {"role": "assistant", "content": p.tgt},
                ],
                "meta": {
                    "source": p.source,
                    "style_target": kind.value,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_sft_builder.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/sft_builder.py data-pipeline/tests/test_sft_builder.py
git commit -m "feat(data-pipeline): SFT JSONL builder in ChatML message form"
```

---

## Task 19: CLI

**Files:**
- Create: `data-pipeline/src/data_pipeline/cli.py`
- Create: `data-pipeline/tests/test_cli_end_to_end.py`

- [ ] **Step 1: Write the failing test**

Create `data-pipeline/tests/test_cli_end_to_end.py`:

```python
import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from data_pipeline.cli import app


def test_build_gec_only_from_local_m2(tmp_path: Path, fixtures_dir: Path) -> None:
    # Place a tiny BEA-2019 train.m2 into raw dir.
    raw = tmp_path / "data" / "raw" / "bea2019"
    raw.mkdir(parents=True)
    shutil.copy(fixtures_dir / "tiny.m2", raw / "train.m2")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-gec",
            "--root",
            str(tmp_path),
            "--sources",
            "bea2019",
            "--split",
            "train",
        ],
    )
    assert result.exit_code == 0, result.output

    out_jsonl = tmp_path / "data" / "processed" / "gec_tagger.jsonl"
    out_manifest = tmp_path / "data" / "processed" / "manifest.json"
    assert out_jsonl.exists()
    assert out_manifest.exists()

    lines = out_jsonl.read_text().strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert rec["tokens"][0] == "He"
    assert "$REPLACE_goes" in rec["tags"] or rec["tags"][1] == "$REPLACE_goes"

    m = json.loads(out_manifest.read_text())
    assert "gec_tagger" in m["artifacts"]
    assert m["artifacts"]["gec_tagger"]["row_count"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_cli_end_to_end.py -v
```

Expected: FAIL (`data_pipeline.cli` not importable).

- [ ] **Step 3: Write implementation**

Create `data-pipeline/src/data_pipeline/cli.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import typer

from data_pipeline.config import PipelineConfig
from data_pipeline.dedup import dedupe_pairs
from data_pipeline.gec_builder import build_gec_jsonl
from data_pipeline.manifest import ManifestBuilder
from data_pipeline.normalize import normalize_text
from data_pipeline.sft_builder import build_sft_jsonl
from data_pipeline.sources.bea2019 import BEA2019Source
from data_pipeline.sources.c4_200m import C4200MSource
from data_pipeline.sources.conll2014 import CoNLL2014Source
from data_pipeline.sources.gyafc import GYAFCSource
from data_pipeline.sources.jfleg import JFLEGSource
from data_pipeline.sources.paradetox import ParaDetoxSource
from data_pipeline.sources.wiki_auto import WikiAutoSource
from data_pipeline.types import Pair, Split

app = typer.Typer(no_args_is_help=True)


def _normalize_pair(p: Pair) -> Pair | None:
    src = normalize_text(p.src)
    tgt = normalize_text(p.tgt)
    if not src:
        return None
    return Pair(src=src, tgt=tgt, source=p.source, split=p.split, meta=p.meta)


def _load_gec_sources(
    cfg: PipelineConfig, names: list[str], split: Split
) -> Iterable[Pair]:
    for name in names:
        if name == "bea2019":
            yield from BEA2019Source(config=cfg, split=split).iter_pairs()
        elif name == "jfleg":
            yield from JFLEGSource(config=cfg, split=split).iter_pairs()
        elif name == "conll2014":
            yield from CoNLL2014Source(config=cfg).iter_pairs()
        elif name == "c4_200m":
            yield from C4200MSource(config=cfg).iter_pairs()
        else:
            raise typer.BadParameter(f"unknown GEC source: {name}")


def _load_style_sources(cfg: PipelineConfig, names: list[str]) -> Iterable[Pair]:
    for name in names:
        if name == "gyafc":
            yield from GYAFCSource(
                config=cfg, split=Split.TRAIN, domain="Family_Relationships"
            ).iter_pairs()
        elif name == "paradetox":
            yield from ParaDetoxSource(config=cfg).iter_pairs()
        elif name == "wiki_auto":
            yield from WikiAutoSource(config=cfg).iter_pairs()
        else:
            raise typer.BadParameter(f"unknown style source: {name}")


@app.command("build-gec")
def build_gec(
    root: Path = typer.Option(..., exists=False, help="Project root"),
    sources: str = typer.Option(
        "bea2019,jfleg,conll2014,c4_200m",
        help="Comma-separated GEC source names for the train split.",
    ),
    split: Split = typer.Option(Split.TRAIN),
    eval_sources: str = typer.Option(
        "",
        help="Comma-separated GEC source names for the eval split (e.g. 'jfleg,conll2014'). "
        "When set, runs a hard leakage check between train and each eval source.",
    ),
) -> None:
    """Build gec_tagger.jsonl and manifest.json."""
    from data_pipeline.leakage import check_leakage  # local import to keep top clean

    cfg = PipelineConfig(root=root).ensure_dirs()
    names = [s.strip() for s in sources.split(",") if s.strip()]

    raw_pairs = list(_load_gec_sources(cfg, names, split))
    normalized = [np for p in raw_pairs if (np := _normalize_pair(p))]
    dedup, dedup_stats = dedupe_pairs(normalized)

    leakage_passed: bool | None = None
    if eval_sources.strip():
        eval_names = [s.strip() for s in eval_sources.split(",") if s.strip()]
        eval_sets: dict[str, list[Pair]] = {}
        for ev_name in eval_names:
            ev_split = Split.TEST if ev_name == "conll2014" else Split.DEV
            ev_raw = list(_load_gec_sources(cfg, [ev_name], ev_split))
            eval_sets[ev_name] = [
                np for p in ev_raw if (np := _normalize_pair(p))
            ]
        check_leakage(train=dedup, eval_sets=eval_sets)  # raises on overlap
        leakage_passed = True

    out = cfg.processed_dir / "gec_tagger.jsonl"
    written = build_gec_jsonl(dedup, out)

    mb = ManifestBuilder(out_dir=cfg.processed_dir)
    mb.add_artifact(name="gec_tagger", path=out, row_count=written)
    for name in names:
        per_source = sum(1 for p in raw_pairs if p.source == name)
        retained = sum(1 for p in dedup if p.source == name)
        mb.add_source(name=name, row_count=per_source, retained=retained)
    mb.set_dedup(removed_exact=dedup_stats.removed_exact)
    mb.set_leakage_passed(leakage_passed)
    mb.write()
    typer.echo(f"wrote {written} records to {out}")


@app.command("build-sft")
def build_sft(
    root: Path = typer.Option(..., exists=False, help="Project root"),
    sources: str = typer.Option(
        "gyafc,paradetox,wiki_auto", help="Comma-separated style sources."
    ),
) -> None:
    """Build style_sft.jsonl and manifest.json."""
    cfg = PipelineConfig(root=root).ensure_dirs()
    names = [s.strip() for s in sources.split(",") if s.strip()]

    raw_pairs = list(_load_style_sources(cfg, names))
    normalized = [np for p in raw_pairs if (np := _normalize_pair(p))]
    dedup, dedup_stats = dedupe_pairs(normalized)

    out = cfg.processed_dir / "style_sft.jsonl"
    written = build_sft_jsonl(dedup, out)

    mb = ManifestBuilder(out_dir=cfg.processed_dir)
    mb.add_artifact(name="style_sft", path=out, row_count=written)
    for name in names:
        per_source = sum(1 for p in raw_pairs if p.source == name)
        retained = sum(1 for p in dedup if p.source == name)
        mb.add_source(name=name, row_count=per_source, retained=retained)
    mb.set_dedup(removed_exact=dedup_stats.removed_exact)
    mb.set_leakage_passed(True)
    mb.write()
    typer.echo(f"wrote {written} records to {out}")


@app.command("build-all")
def build_all(
    root: Path = typer.Option(..., exists=False),
    gec_sources: str = typer.Option("bea2019,c4_200m"),
    eval_sources: str = typer.Option("jfleg,conll2014"),
    style_sources: str = typer.Option("gyafc,paradetox,wiki_auto"),
    split: Split = typer.Option(Split.TRAIN),
) -> None:
    build_gec(root=root, sources=gec_sources, split=split, eval_sources=eval_sources)
    build_sft(root=root, sources=style_sources)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_cli_end_to_end.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add data-pipeline/src/data_pipeline/cli.py data-pipeline/tests/test_cli_end_to_end.py
git commit -m "feat(data-pipeline): typer CLI (build-gec, build-sft, build-all)"
```

---

## Task 20: Top-level checks (ruff + mypy + full suite)

**Files:** none new.

- [ ] **Step 1: Run lint**

```
cd data-pipeline && uv run ruff check src tests
```

Expected: no errors. Fix any introduced.

- [ ] **Step 2: Run type check**

```
cd data-pipeline && uv run mypy src
```

Expected: `Success: no issues found`. Fix any introduced.

- [ ] **Step 3: Run full pytest with coverage**

```
cd data-pipeline && uv run pytest --cov=data_pipeline --cov-report=term-missing
```

Expected: all tests PASS, coverage ≥ 85% on `src/data_pipeline`.

- [ ] **Step 4: Commit anything that changed during cleanup**

```
git add -A
git diff --cached --quiet && echo "nothing to commit" || git commit -m "chore(data-pipeline): lint/type fixes"
```

---

## Notes on data acquisition

These datasets are not auto-downloaded by the pipeline (license-gated or HF-only):

- **BEA-2019**: register at `https://www.cl.cam.ac.uk/research/nl/bea2019st/`, place `train.m2` and `dev.m2` under `data/raw/bea2019/`.
- **GYAFC**: request from authors (Rao & Tetreault), place per-domain `informal`/`formal` files under `data/raw/gyafc/<Domain>/<split>/`.
- **CoNLL-2014**: download `official-2014.0.m2`, place under `data/raw/conll2014/`.
- **JFLEG, C4_200M, ParaDetox, Wiki-Auto**: pulled from Hugging Face Hub automatically by their loaders.

The pipeline raises clear `FileNotFoundError` messages when a license-gated source is requested without its files being present.
