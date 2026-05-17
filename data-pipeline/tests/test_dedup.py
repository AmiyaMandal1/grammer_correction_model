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
