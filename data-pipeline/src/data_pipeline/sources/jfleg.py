from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split


class JFLEGSource(Source):
    name = "jfleg"

    def __init__(self, config: PipelineConfig, split: Split) -> None:
        super().__init__(config)
        self.split = split

    def iter_pairs(self) -> Iterator[Pair]:
        split_name = {Split.DEV: "validation", Split.TEST: "test"}.get(self.split)
        if split_name is None:
            raise ValueError(f"JFLEG has no {self.split.value} split")
        rows = load_dataset("jfleg", split=split_name)
        for row in rows:
            sentence = row["sentence"]
            for correction in row["corrections"]:
                yield Pair(
                    src=sentence,
                    tgt=correction,
                    source=self.name,
                    split=self.split,
                )
