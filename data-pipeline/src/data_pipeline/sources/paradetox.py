from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split, StyleKind


class ParaDetoxSource(Source):
    name = "paradetox"

    def iter_pairs(self) -> Iterator[Pair]:
        rows = load_dataset("s-nlp/paradetox", split="train")
        for row in rows:
            yield Pair(
                src=row["en_toxic_comment"],
                tgt=row["en_neutral_comment"],
                source=self.name,
                split=Split.TRAIN,
                meta={"style_target": StyleKind.DETOXIFY.value},
            )
