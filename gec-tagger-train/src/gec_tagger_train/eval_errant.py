from __future__ import annotations

import errant
import spacy

_NLP: spacy.language.Language | None = None


def _nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def compute_errant_f05(
    *, src: list[str], ref: list[str], hyp: list[str]
) -> dict[str, float]:
    """F0.5 between reference and hypothesis edits, scored by ERRANT."""
    if not (len(src) == len(ref) == len(hyp)):
        raise ValueError("src, ref, hyp must have equal length")
    ann = errant.load("en", _nlp())
    tp = fp = fn = 0
    for s, r, h in zip(src, ref, hyp, strict=True):
        src_doc = ann.parse(s, tokenise=True)
        ref_doc = ann.parse(r, tokenise=True)
        hyp_doc = ann.parse(h, tokenise=True)
        ref_edits = {(e.o_start, e.o_end, e.c_str) for e in ann.annotate(src_doc, ref_doc)}
        hyp_edits = {(e.o_start, e.o_end, e.c_str) for e in ann.annotate(src_doc, hyp_doc)}
        tp += len(ref_edits & hyp_edits)
        fp += len(hyp_edits - ref_edits)
        fn += len(ref_edits - hyp_edits)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta = 0.5
    if precision + recall == 0:
        f = 0.0
    else:
        f = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    return {"precision": precision, "recall": recall, "f0.5": f}
