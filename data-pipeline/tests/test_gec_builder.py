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
