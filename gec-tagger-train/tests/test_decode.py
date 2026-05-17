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


def test_transform_case_lower() -> None:
    out = apply_tags_once(["HELLO", "World"], ["$TRANSFORM_CASE_LOWER", "$KEEP"])
    assert out == ["hello", "World"]


def test_transform_case_upper() -> None:
    out = apply_tags_once(["hello"], ["$TRANSFORM_CASE_UPPER"])
    assert out == ["HELLO"]


def test_transform_case_capital() -> None:
    out = apply_tags_once(["hello"], ["$TRANSFORM_CASE_CAPITAL"])
    assert out == ["Hello"]


def test_transform_verb_falls_through_as_keep() -> None:
    out = apply_tags_once(["go"], ["$TRANSFORM_VERB_VB_VBZ"])
    assert out == ["go"]
