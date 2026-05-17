from pathlib import Path

import pytest

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.gyafc import GYAFCSource
from data_pipeline.types import Split, StyleKind


def test_loads_parallel_files(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    d = cfg.raw_dir / "gyafc" / "Family_Relationships" / "train"
    d.mkdir(parents=True)
    (d / "informal").write_text("yo whats up\nidk man\n")
    (d / "formal").write_text("Hello, how are you?\nI am uncertain.\n")

    src = GYAFCSource(config=cfg, split=Split.TRAIN, domain="Family_Relationships")
    pairs = list(src.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].meta["style_target"] == StyleKind.FORMAL.value
    assert pairs[0].src == "yo whats up"
    assert pairs[0].tgt == "Hello, how are you?"
    assert pairs[0].source == "gyafc"


def test_missing_files_raise(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = GYAFCSource(config=cfg, split=Split.TRAIN, domain="Family_Relationships")
    with pytest.raises(FileNotFoundError):
        list(src.iter_pairs())


def test_mismatched_lengths_raise(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    d = cfg.raw_dir / "gyafc" / "Family_Relationships" / "train"
    d.mkdir(parents=True)
    (d / "informal").write_text("a\nb\n")
    (d / "formal").write_text("A\n")
    src = GYAFCSource(config=cfg, split=Split.TRAIN, domain="Family_Relationships")
    with pytest.raises(ValueError):
        list(src.iter_pairs())
