from __future__ import annotations

from collections.abc import Callable
from typing import Any


def align_tags_to_subwords(
    encoding: Any,
    tags: list[str],
    *,
    pad_id: int,
    unk_id: int,
    vocab_lookup: Callable[[str], int],
) -> list[int]:
    """Broadcast a per-whitespace-token tag list to subword-token ids.

    `encoding.word_ids()` returns a list with one entry per subword: either
    the index of the source whitespace token, or `None` for special tokens
    ([CLS], [SEP], pad). Special tokens get `pad_id`; tags not present in
    the vocabulary fall back to `unk_id`.
    """
    word_ids = encoding.word_ids()
    out: list[int] = []
    for w in word_ids:
        if w is None:
            out.append(pad_id)
            continue
        tag = tags[w]
        try:
            out.append(vocab_lookup(tag))
        except KeyError:
            out.append(unk_id)
    return out
