from __future__ import annotations


def apply_tags_once(tokens: list[str], tags: list[str]) -> list[str]:
    """Apply one round of GECToR edit tags to a token list.

    Unrecognized tags are treated as `$KEEP` so decoding is robust to
    vocabulary drift. Use multiple passes (caller-driven) for cases that
    need more than one edit at a position.
    """
    if len(tokens) != len(tags):
        raise ValueError(
            f"length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
        )
    out: list[str] = []
    for tok, tag in zip(tokens, tags, strict=True):
        if tag == "$KEEP" or not tag.startswith("$"):
            out.append(tok)
            continue
        if tag == "$DELETE":
            continue
        kind, _, value = tag[1:].partition("_")
        if kind == "REPLACE" and value:
            out.append(value)
        elif kind == "APPEND" and value:
            out.append(tok)
            out.append(value)
        else:
            out.append(tok)
    return out
