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
        '{"messages":[{"role":"system","content":"x"}]}\n'
        '{"messages":[{"role":"system","content":"a"},{"role":"user","content":"b"},{"role":"assistant","content":"c"}]}\n'
        "\n"
    )
    records = list(load_sft_records(bad))
    assert len(records) == 1
