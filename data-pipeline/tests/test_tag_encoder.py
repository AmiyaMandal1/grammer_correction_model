from data_pipeline.tag_encoder import (
    align_tokens_to_tags,
    decode_tag,
    encode_tag,
)
from data_pipeline.types import EditTag, TagKind


def test_encode_decode_keep() -> None:
    t = EditTag(kind=TagKind.KEEP, value=None)
    s = encode_tag(t)
    assert s == "$KEEP"
    assert decode_tag(s) == t


def test_encode_decode_replace() -> None:
    t = EditTag(kind=TagKind.REPLACE, value="goes")
    assert encode_tag(t) == "$REPLACE_goes"
    assert decode_tag("$REPLACE_goes") == t


def test_decode_append_with_underscore_in_value() -> None:
    assert decode_tag("$APPEND_well_done") == EditTag(
        kind=TagKind.APPEND, value="well_done"
    )


def test_align_tokens_to_tags_keep_all_when_equal() -> None:
    src = ["I", "am", "happy"]
    tgt = ["I", "am", "happy"]
    tags = align_tokens_to_tags(src, tgt)
    assert tags == [
        EditTag(TagKind.KEEP, None),
        EditTag(TagKind.KEEP, None),
        EditTag(TagKind.KEEP, None),
    ]


def test_align_tokens_replace_one() -> None:
    src = ["he", "go", "home"]
    tgt = ["he", "goes", "home"]
    tags = align_tokens_to_tags(src, tgt)
    assert tags == [
        EditTag(TagKind.KEEP, None),
        EditTag(TagKind.REPLACE, "goes"),
        EditTag(TagKind.KEEP, None),
    ]


def test_align_tokens_delete() -> None:
    src = ["he", "the", "goes", "home"]
    tgt = ["he", "goes", "home"]
    tags = align_tokens_to_tags(src, tgt)
    assert tags[1] == EditTag(TagKind.DELETE, None)


def test_align_tokens_append() -> None:
    src = ["he", "home"]
    tgt = ["he", "goes", "home"]
    tags = align_tokens_to_tags(src, tgt)
    # APPEND on previous token
    assert tags[0] == EditTag(TagKind.APPEND, "goes")
