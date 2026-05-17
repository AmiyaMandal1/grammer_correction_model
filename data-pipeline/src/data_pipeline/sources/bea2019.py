from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split


class BEA2019Source(Source):
    name = "bea2019"

    def __init__(self, config: PipelineConfig, split: Split) -> None:
        super().__init__(config)
        self.split = split

    def _m2_file(self) -> Path:
        fname = {
            Split.TRAIN: "train.m2",
            Split.DEV: "dev.m2",
            Split.TEST: "test.m2",
        }[self.split]
        return self.raw_path / fname

    def iter_pairs(self) -> Iterator[Pair]:
        path = self._m2_file()
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} (download BEA-2019 to {self.raw_path})"
            )
        for src_toks, edits in _parse_m2(path):
            tgt_toks = _apply_edits(src_toks, edits)
            yield Pair(
                src=" ".join(src_toks),
                tgt=" ".join(tgt_toks),
                source=self.name,
                split=self.split,
            )


def _parse_m2(path: Path) -> Iterator[tuple[list[str], list[tuple[int, int, str]]]]:
    src_toks: list[str] | None = None
    edits: list[tuple[int, int, str]] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("S "):
                if src_toks is not None:
                    yield src_toks, edits
                src_toks = line[2:].split(" ")
                edits = []
            elif line.startswith("A "):
                # A start end|||type|||replacement|||REQUIRED|||-NONE-|||annotator
                parts = line[2:].split("|||")
                span = parts[0].split(" ")
                start, end = int(span[0]), int(span[1])
                if start == -1:
                    continue  # noop
                replacement = parts[2]
                edits.append((start, end, replacement))
            elif line == "":
                if src_toks is not None:
                    yield src_toks, edits
                    src_toks, edits = None, []
        if src_toks is not None:
            yield src_toks, edits


def _apply_edits(
    src_toks: list[str], edits: list[tuple[int, int, str]]
) -> list[str]:
    if not edits:
        return list(src_toks)
    # Apply edits in right-to-left order to keep offsets stable.
    out = list(src_toks)
    for start, end, replacement in sorted(edits, key=lambda e: -e[0]):
        repl = replacement.split(" ") if replacement else []
        out[start:end] = repl
    return out
