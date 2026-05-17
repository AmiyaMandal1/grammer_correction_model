from data_pipeline.types import EditTag, Pair, Split, StyleKind, TagKind


def test_pair_round_trip() -> None:
    p = Pair(src="he go home", tgt="he goes home", source="bea2019", split=Split.TRAIN)
    d = p.to_dict()
    assert d == {
        "src": "he go home",
        "tgt": "he goes home",
        "source": "bea2019",
        "split": "train",
        "meta": {},
    }
    p2 = Pair.from_dict(d)
    assert p2 == p


def test_edit_tag_construct() -> None:
    t = EditTag(kind=TagKind.REPLACE, value="goes")
    assert t.to_str() == "$REPLACE_goes"


def test_edit_tag_keep() -> None:
    t = EditTag(kind=TagKind.KEEP, value=None)
    assert t.to_str() == "$KEEP"


def test_style_kind_values() -> None:
    assert StyleKind.FORMAL.value == "formal"
    assert StyleKind.CONCISE.value == "concise"
