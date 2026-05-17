from unittest.mock import MagicMock

from gec_tagger_train.tokenization import align_tags_to_subwords


def test_aligns_single_subword_tokens() -> None:
    enc = MagicMock()
    enc.word_ids.return_value = [None, 0, 1, 2, None]
    tags = ["$KEEP", "$KEEP", "$REPLACE_sits"]
    vocab = {"$KEEP": 2, "$REPLACE_sits": 7}
    out = align_tags_to_subwords(enc, tags, pad_id=0, unk_id=1, vocab_lookup=lambda t: vocab[t])
    assert out == [0, 2, 2, 7, 0]


def test_aligns_multi_subword_token() -> None:
    enc = MagicMock()
    enc.word_ids.return_value = [None, 0, 0, 0, None]
    tags = ["$REPLACE_confused"]
    vocab = {"$REPLACE_confused": 5}
    out = align_tags_to_subwords(enc, tags, pad_id=0, unk_id=1, vocab_lookup=lambda t: vocab[t])
    assert out == [0, 5, 5, 5, 0]


def test_unknown_tag_falls_back_to_unk() -> None:
    enc = MagicMock()
    enc.word_ids.return_value = [None, 0, None]
    tags = ["$REPLACE_neverseen"]

    def lookup(t: str) -> int:
        raise KeyError(t)

    out = align_tags_to_subwords(enc, tags, pad_id=0, unk_id=1, vocab_lookup=lookup)
    assert out == [0, 1, 0]
