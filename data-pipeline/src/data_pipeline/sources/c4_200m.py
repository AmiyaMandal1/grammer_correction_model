from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split


class C4200MSource(Source):
    """Streaming loader for the C4_200M synthetic GEC dataset.

    HF dataset id: `liweili/c4_200m`. Streams to avoid downloading the
    full set; capped at `max_rows` for development and small-scale runs.
    """

    name = "c4_200m"

    def __init__(self, config: PipelineConfig, max_rows: int = 2_000_000) -> None:
        super().__init__(config)
        self.max_rows = max_rows

    def iter_pairs(self) -> Iterator[Pair]:
        ds = load_dataset("liweili/c4_200m", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= self.max_rows:
                break
            yield Pair(
                src=row["input"],
                tgt=row["output"],
                source=self.name,
                split=Split.TRAIN,
            )
            count += 1
