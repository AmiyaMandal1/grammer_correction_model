import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from gec_tagger_train.cli import app
from gec_tagger_train.model import DebertaTagger
from gec_tagger_train.tag_vocab import build_tag_vocab


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


def test_eval_cmd(fixtures_dir: Path, tmp_out: Path) -> None:
    """Eval command loads an untrained checkpoint, runs inference, and prints JSON metrics."""
    # Build vocab from tiny fixture
    vocab_path = tmp_out / "tags.json"
    vocab = build_tag_vocab(jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=vocab_path)

    # Save an untrained model checkpoint
    ckpt_dir = tmp_out / "ckpt_eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = DebertaTagger(num_tags=len(vocab))
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")

    # Write a tiny 2-pair dev JSONL
    dev_path = tmp_out / "dev.jsonl"
    dev_pairs = [
        {"src": "He go home .", "ref": "He goes home ."},
        {"src": "I are happy .", "ref": "I am happy ."},
    ]
    dev_path.write_text("\n".join(json.dumps(p) for p in dev_pairs) + "\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "--checkpoint",
            str(ckpt_dir),
            "--tags",
            str(vocab_path),
            "--dev",
            str(dev_path),
            "--max-length",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output
    # Progress bars use \r so JSON may be preceded by \r-terminated lines; extract it.
    # Find the last substring that starts with '{' and ends with '}'.
    out = result.output
    start = out.rfind("{")
    end = out.rfind("}") + 1
    assert start != -1 and end > start, f"No JSON found in output: {out!r}"
    metrics = json.loads(out[start:end])
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f0.5" in metrics
