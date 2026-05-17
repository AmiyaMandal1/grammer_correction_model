from gec_tagger_train.decode import apply_tags_once


def test_keep_returns_input_unchanged() -> None:
    out = apply_tags_once(["he", "go", "home"], ["$KEEP", "$KEEP", "$KEEP"])
    assert out == ["he", "go", "home"]


def test_replace_substitutes_token() -> None:
    out = apply_tags_once(["he", "go", "home"], ["$KEEP", "$REPLACE_goes", "$KEEP"])
    assert out == ["he", "goes", "home"]


def test_delete_removes_token() -> None:
    out = apply_tags_once(["he", "the", "goes"], ["$KEEP", "$DELETE", "$KEEP"])
    assert out == ["he", "goes"]


def test_append_inserts_after_anchor() -> None:
    out = apply_tags_once(["he", "home"], ["$APPEND_goes", "$KEEP"])
    assert out == ["he", "goes", "home"]


def test_unknown_tag_treated_as_keep() -> None:
    out = apply_tags_once(["he", "go"], ["$KEEP", "$WAT"])
    assert out == ["he", "go"]
