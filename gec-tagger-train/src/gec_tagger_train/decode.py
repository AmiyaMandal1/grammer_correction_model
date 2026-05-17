from __future__ import annotations


def apply_tags_once(tokens: list[str], tags: list[str]) -> list[str]:
    """Apply one round of GECToR edit tags to a token list.

    Supported tags: $KEEP, $DELETE, $REPLACE_<value>, $APPEND_<value>,
    $TRANSFORM_CASE_<LOWER|UPPER|CAPITAL>. $TRANSFORM_VERB_* is reserved
    and currently treated as $KEEP (proper handling needs lemminflect).
    Unrecognized tags are treated as $KEEP so decoding is robust to
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
        body = tag[1:]
        if body.startswith("REPLACE_"):
            value = body[len("REPLACE_") :]
            if value:
                out.append(value)
            else:
                out.append(tok)
            continue
        if body.startswith("APPEND_"):
            value = body[len("APPEND_") :]
            if value:
                out.append(tok)
                out.append(value)
            else:
                out.append(tok)
            continue
        if body.startswith("TRANSFORM_CASE_"):
            kind = body[len("TRANSFORM_CASE_") :]
            out.append(_apply_case(tok, kind))
            continue
        # TRANSFORM_VERB_* and anything else: treat as KEEP for now.
        out.append(tok)
    return out


def _apply_case(token: str, kind: str) -> str:
    if kind == "LOWER":
        return token.lower()
    if kind == "UPPER":
        return token.upper()
    if kind == "CAPITAL":
        return token.capitalize()
    return token
