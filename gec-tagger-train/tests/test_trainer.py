from pathlib import Path

from transformers import AutoTokenizer

from gec_tagger_train.config import StageConfig
from gec_tagger_train.dataset import GECTaggerDataset
from gec_tagger_train.tag_vocab import build_tag_vocab
from gec_tagger_train.trainer import run_training


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
