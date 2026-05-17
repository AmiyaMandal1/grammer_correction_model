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
