from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from data_pipeline.types import Pair, StyleKind

_SYSTEM_PROMPT = {
    StyleKind.FORMAL: "Rewrite the user's text in formal English while preserving meaning.",
    StyleKind.INFORMAL: (
        "Rewrite the user's text in casual, informal English while preserving meaning."
    ),
    StyleKind.CONCISE: "Rewrite the user's text more concisely while preserving meaning.",
    StyleKind.SIMPLIFY: "Rewrite the user's text in simpler English while preserving meaning.",
    StyleKind.DETOXIFY: (
        "Rewrite the user's text in a neutral, non-toxic way while preserving meaning."
    ),
}


def build_sft_jsonl(pairs: Iterable[Pair], out_path: Path) -> int:
    written = 0
    with out_path.open("w") as f:
        for p in pairs:
            target = p.meta.get("style_target")
            if not target:
                continue
            try:
                kind = StyleKind(target)
            except ValueError:
                continue
            rec = {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT[kind]},
                    {"role": "user", "content": p.src},
                    {"role": "assistant", "content": p.tgt},
                ],
                "meta": {
                    "source": p.source,
                    "style_target": kind.value,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written
