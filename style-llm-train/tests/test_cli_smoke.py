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
