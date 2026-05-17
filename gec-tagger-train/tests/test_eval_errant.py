from gec_tagger_train.eval_errant import compute_errant_f05


def test_perfect_match_gives_f05_one() -> None:
    score = compute_errant_f05(
        src=["He go home."],
        ref=["He goes home."],
        hyp=["He goes home."],
    )
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f0.5"] == 1.0


def test_no_match_gives_zero() -> None:
    score = compute_errant_f05(
        src=["He go home."],
        ref=["He goes home."],
        hyp=["He go home."],
    )
    assert score["recall"] == 0.0
    assert score["f0.5"] == 0.0
