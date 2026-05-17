from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split, StyleKind

Domain = Literal["Family_Relationships", "Entertainment_Music"]


class GYAFCSource(Source):
    name = "gyafc"

    def __init__(
        self,
        config: PipelineConfig,
        split: Split,
        domain: Domain,
    ) -> None:
        super().__init__(config)
        self.split = split
        self.domain = domain

    def iter_pairs(self) -> Iterator[Pair]:
        split_dir_name = {
            Split.TRAIN: "train",
            Split.DEV: "tune",
            Split.TEST: "test",
        }[self.split]
        base = self.raw_path / self.domain / split_dir_name
        informal = base / "informal"
        formal = base / "formal"
        if not informal.exists() or not formal.exists():
            raise FileNotFoundError(
                f"expected {informal} and {formal} (place GYAFC files there)"
            )
        with informal.open() as fi, formal.open() as ff:
            informal_lines = [ln.rstrip("\n") for ln in fi]
            formal_lines = [ln.rstrip("\n") for ln in ff]
        if len(informal_lines) != len(formal_lines):
            raise ValueError(
                f"GYAFC parallel length mismatch: {len(informal_lines)} != {len(formal_lines)}"
            )
        for i_line, f_line in zip(informal_lines, formal_lines, strict=True):
            yield Pair(
                src=i_line,
                tgt=f_line,
                source=self.name,
                split=self.split,
                meta={
                    "style_target": StyleKind.FORMAL.value,
                    "domain": self.domain,
                },
            )
