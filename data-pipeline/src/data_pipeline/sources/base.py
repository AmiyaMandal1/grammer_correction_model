from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.types import Pair


class Source(ABC):
    """Abstract source loader.

    Implementations read raw files from `config.raw_dir / self.name` and
    yield `Pair` records. They do not perform normalization or dedup;
    those happen in the builder stage.
    """

    name: str

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    @property
    def raw_path(self) -> Path:
        return self.config.raw_dir / self.name

    @abstractmethod
    def iter_pairs(self) -> Iterator[Pair]: ...
