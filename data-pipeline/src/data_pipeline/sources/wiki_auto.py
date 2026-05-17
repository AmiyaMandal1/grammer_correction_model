from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split, StyleKind


class WikiAutoSource(Source):
    name = "wiki_auto"

    def iter_pairs(self) -> Iterator[Pair]:
        rows = load_dataset("wiki_auto", "auto_acl", split="train")
        for row in rows:
            yield Pair(
                src=row["normal_sentence"],
                tgt=row["simple_sentence"],
                source=self.name,
                split=Split.TRAIN,
                meta={"style_target": StyleKind.SIMPLIFY.value},
            )
