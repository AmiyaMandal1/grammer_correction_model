import json
from pathlib import Path

from gec_tagger_train.tag_vocab import TagVocab, build_tag_vocab


def test_build_vocab_pads_special_tags(fixtures_dir: Path, tmp_out: Path) -> None:
    out = tmp_out / "tags.json"
    vocab = build_tag_vocab(jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=out)
    assert isinstance(vocab, TagVocab)
    assert vocab.special_tags == ["$PAD", "$UNK", "$KEEP"]
    assert vocab.id_of("$PAD") == 0
    assert vocab.id_of("$UNK") == 1
    assert vocab.id_of("$KEEP") == 2
    assert vocab.id_of("$REPLACE_goes") >= 3
    assert "$REPLACE_sits" in vocab.tags

    payload = json.loads(out.read_text())
    assert payload["tags"][0] == "$PAD"
    assert payload["tags"][2] == "$KEEP"


def test_min_count_filters_rare_tags(fixtures_dir: Path, tmp_out: Path) -> None:
    out = tmp_out / "tags.json"
    vocab = build_tag_vocab(jsonl=fixtures_dir / "tiny.jsonl", min_count=2, out=out)
    assert "$REPLACE_sits" not in vocab.tags
    assert "$REPLACE_goes" in vocab.tags


def test_unknown_tag_maps_to_unk(fixtures_dir: Path, tmp_out: Path) -> None:
    vocab = build_tag_vocab(
        jsonl=fixtures_dir / "tiny.jsonl", min_count=1, out=tmp_out / "t.json"
    )
    assert vocab.id_of("$REPLACE_neverseen") == vocab.id_of("$UNK")


def test_tag_vocab_round_trip(tmp_out: Path) -> None:
    v = TagVocab(tags=["$PAD", "$UNK", "$KEEP", "$DELETE", "$REPLACE_foo"])
    p = tmp_out / "v.json"
    v.save(p)
    v2 = TagVocab.load(p)
    assert v2.tags == v.tags
    assert v2.id_of("$REPLACE_foo") == 4
