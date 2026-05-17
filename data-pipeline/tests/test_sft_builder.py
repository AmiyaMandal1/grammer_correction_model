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
