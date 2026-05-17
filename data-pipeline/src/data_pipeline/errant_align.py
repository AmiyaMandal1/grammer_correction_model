from __future__ import annotations

from functools import lru_cache

import errant
import spacy


def tokenize(text: str, nlp: spacy.language.Language) -> list[str]:
    doc = nlp.make_doc(text)
    return [t.text for t in doc]


@lru_cache(maxsize=1)
def _load_nlp() -> spacy.language.Language:
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


class ErrantAligner:
    """Thin wrapper over ERRANT that exposes token-aligned source/target.

    ERRANT itself runs a more elaborate edit extraction; we only need the
    token sequences that match what the downstream tag encoder consumes.
    """

    def __init__(self) -> None:
        self.nlp = _load_nlp()
        self.annotator = errant.load("en", self.nlp)

    def align_pair(self, src: str, tgt: str) -> tuple[list[str], list[str]]:
        if not src or not tgt:
            raise ValueError("source and target must be non-empty")
        src_doc = self.annotator.parse(src, tokenise=True)
        tgt_doc = self.annotator.parse(tgt, tokenise=True)
        return [t.text for t in src_doc], [t.text for t in tgt_doc]
