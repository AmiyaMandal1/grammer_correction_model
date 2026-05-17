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
