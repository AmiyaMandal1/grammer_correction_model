from __future__ import annotations

from difflib import SequenceMatcher

from data_pipeline.types import EditTag, TagKind


def encode_tag(tag: EditTag) -> str:
    return tag.to_str()


def decode_tag(s: str) -> EditTag:
    if not s.startswith("$"):
        raise ValueError(f"not a tag: {s!r}")
    body = s[1:]
    if body == "KEEP":
        return EditTag(TagKind.KEEP, None)
    if body == "DELETE":
        return EditTag(TagKind.DELETE, None)
    head, _, value = body.partition("_")
    if not value:
        raise ValueError(f"tag missing value: {s!r}")
    try:
        kind = TagKind(head)
    except ValueError as e:
        raise ValueError(f"unknown tag kind: {head}") from e
    return EditTag(kind=kind, value=value)


def align_tokens_to_tags(src: list[str], tgt: list[str]) -> list[EditTag]:
    """Single-pass token-level alignment.

    The full GECToR scheme is iterative; this function performs one round
    of edit derivation. Iterative re-tagging is handled by the tagger model
    at inference time.
    """
    tags: list[EditTag] = [EditTag(TagKind.KEEP, None) for _ in src]
    if not src:
        return tags
    matcher = SequenceMatcher(a=src, b=tgt, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            for k, src_idx in enumerate(range(i1, i2)):
                if j1 + k < j2:
                    tags[src_idx] = EditTag(TagKind.REPLACE, tgt[j1 + k])
                else:
                    tags[src_idx] = EditTag(TagKind.DELETE, None)
        elif op == "delete":
            for src_idx in range(i1, i2):
                tags[src_idx] = EditTag(TagKind.DELETE, None)
        elif op == "insert":
            # Attach inserts to the preceding source token via APPEND.
            anchor = max(i1 - 1, 0)
            for j_idx in range(j1, j2):
                # Multiple inserts on the same anchor: only the first survives
                # in single-pass mode. Tagger model iterates to capture more.
                if tags[anchor].kind is TagKind.KEEP:
                    tags[anchor] = EditTag(TagKind.APPEND, tgt[j_idx])
                    break
    return tags
