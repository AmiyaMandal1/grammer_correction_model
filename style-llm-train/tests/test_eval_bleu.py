from style_llm_train.eval_bleu import compute_corpus_bleu


def test_identical_corpus_scores_one() -> None:
    score = compute_corpus_bleu(
        hyp=["Hello, how are you?"], ref=["Hello, how are you?"]
    )
    assert score["bleu"] > 0.9


def test_mismatched_lengths_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_corpus_bleu(hyp=["a", "b"], ref=["a"])
