from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import xxhash

from data_pipeline.types import Pair


@dataclass(frozen=True)
class DedupStats:
    input_count: int
    output_count: int
    removed_exact: int


def _key(p: Pair) -> int:
    h = xxhash.xxh64()
    h.update(p.src.lower().encode("utf-8"))
    h.update(b"\x1f")
    h.update(p.tgt.lower().encode("utf-8"))
    return h.intdigest()


def dedupe_pairs(pairs: Iterable[Pair]) -> tuple[list[Pair], DedupStats]:
    seen: set[int] = set()
    out: list[Pair] = []
    in_count = 0
    for p in pairs:
        in_count += 1
        k = _key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out, DedupStats(
        input_count=in_count,
        output_count=len(out),
        removed_exact=in_count - len(out),
    )
