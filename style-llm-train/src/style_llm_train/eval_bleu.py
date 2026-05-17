from __future__ import annotations

from sacrebleu import corpus_bleu


def compute_corpus_bleu(*, hyp: list[str], ref: list[str]) -> dict[str, float]:
    """Single-reference corpus BLEU via sacrebleu."""
    if len(hyp) != len(ref):
        raise ValueError(f"length mismatch: {len(hyp)} hyp vs {len(ref)} ref")
    score = corpus_bleu(hyp, [ref])
    return {"bleu": float(score.score) / 100.0}
