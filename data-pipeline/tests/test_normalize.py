from data_pipeline.normalize import normalize_text


def test_collapses_whitespace() -> None:
    assert normalize_text("hello   world\t\t!") == "hello world !"


def test_nfkc_compatibility() -> None:
    # fullwidth digits to ascii
    assert normalize_text("test ２０２３") == "test 2023"  # noqa: RUF001


def test_strips_zero_width() -> None:
    assert normalize_text("ab​cd") == "abcd"


def test_unifies_quotes_and_dashes() -> None:
    assert normalize_text("“hello” — world") == '"hello" - world'


def test_strips_outer_whitespace() -> None:
    assert normalize_text("  hi  ") == "hi"


def test_empty_input() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
