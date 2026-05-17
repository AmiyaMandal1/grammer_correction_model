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
        max_length=2,
    )
    assert len(ds) == 0
