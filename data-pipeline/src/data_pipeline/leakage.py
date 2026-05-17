from __future__ import annotations

from collections.abc import Mapping

import xxhash

from data_pipeline.types import Pair


class LeakageError(RuntimeError):
    pass


def _src_key(p: Pair) -> int:
    return xxhash.xxh64(p.src.lower().encode("utf-8")).intdigest()


def check_leakage(
    *, train: list[Pair], eval_sets: Mapping[str, list[Pair]]
) -> None:
    train_keys = {_src_key(p) for p in train}
    for name, items in eval_sets.items():
        overlap = sum(1 for p in items if _src_key(p) in train_keys)
        if overlap > 0:
            raise LeakageError(
                f"train ∩ {name} = {overlap} rows (source side, case-insensitive)"
            )
