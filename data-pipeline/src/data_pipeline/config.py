from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    def ensure_dirs(self) -> PipelineConfig:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
