from __future__ import annotations

from collections.abc import Iterator

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.sources.bea2019 import _apply_edits, _parse_m2
from data_pipeline.types import Pair, Split


class CoNLL2014Source(Source):
    name = "conll2014"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def iter_pairs(self) -> Iterator[Pair]:
        path = self.raw_path / "official-2014.0.m2"
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} (download CoNLL-2014 to {self.raw_path})"
            )
        for src_toks, edits in _parse_m2(path):
            tgt_toks = _apply_edits(src_toks, edits)
            yield Pair(
                src=" ".join(src_toks),
                tgt=" ".join(tgt_toks),
                source=self.name,
                split=Split.TEST,
            )
