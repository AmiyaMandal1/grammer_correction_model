from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from data_pipeline.tag_encoder import align_tokens_to_tags, encode_tag
from data_pipeline.types import Pair


def build_gec_jsonl(pairs: Iterable[Pair], out_path: Path) -> int:
    """Write one record per pair: {tokens, tags, source, split, meta}.

    Skips pairs with empty source. Whitespace tokenization is intentional;
    upstream sources already produce ERRANT- or M²-aligned tokens.
    """
    written = 0
    with out_path.open("w") as f:
        for p in pairs:
            src_toks = p.src.split()
            tgt_toks = p.tgt.split()
            if not src_toks:
                continue
            tags = align_tokens_to_tags(src_toks, tgt_toks)
            rec = {
                "tokens": src_toks,
                "tags": [encode_tag(t) for t in tags],
                "source": p.source,
                "split": p.split.value,
                "meta": p.meta,
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written
