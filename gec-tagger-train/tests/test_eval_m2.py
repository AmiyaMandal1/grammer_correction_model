from gec_tagger_train.eval_m2 import format_m2_predictions


def test_format_predictions_emits_m2_block() -> None:
    src_tokens = ["he", "go", "home"]
    pred_tokens = ["he", "goes", "home"]
    block = format_m2_predictions(src_tokens=src_tokens, pred_tokens=pred_tokens)
    assert block.startswith("S he go home")
    assert "A 1 2|||" in block


def test_no_change_emits_noop_edit() -> None:
    block = format_m2_predictions(src_tokens=["he", "is", "ok"], pred_tokens=["he", "is", "ok"])
    assert "A -1 -1|||noop|||" in block
