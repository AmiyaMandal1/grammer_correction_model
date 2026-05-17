from __future__ import annotations

from difflib import SequenceMatcher


def format_m2_predictions(*, src_tokens: list[str], pred_tokens: list[str]) -> str:
    """Emit a single-sentence M² block for the m2scorer tool.

    Edits are derived from a token-level diff. Replacement spans are emitted
    as `A start end|||R|||replacement|||REQUIRED|||-NONE-|||0`. A perfect
    match emits the canonical `noop` edit (`A -1 -1`).
    """
    lines: list[str] = [f"S {' '.join(src_tokens)}"]
    matcher = SequenceMatcher(a=src_tokens, b=pred_tokens, autojunk=False)
    any_edit = False
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        any_edit = True
        replacement = " ".join(pred_tokens[j1:j2])
        kind = {"replace": "R", "delete": "U", "insert": "M"}[op]
        end = i1 if op == "insert" else i2
        lines.append(
            f"A {i1} {end}|||{kind}|||{replacement}|||REQUIRED|||-NONE-|||0"
        )
    if not any_edit:
        lines.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
    return "\n".join(lines) + "\n"
