import shutil
from pathlib import Path

import pytest

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.bea2019 import BEA2019Source
from data_pipeline.types import Split


def test_loads_m2_pairs(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src_dir = cfg.raw_dir / "bea2019"
    src_dir.mkdir()
    shutil.copy(fixtures_dir / "tiny.m2", src_dir / "train.m2")

    src = BEA2019Source(config=cfg, split=Split.TRAIN)
    pairs = list(src.iter_pairs())

    assert len(pairs) == 3
    assert pairs[0].src == "He go to school ."
    assert pairs[0].tgt == "He goes to school ."
    assert pairs[1].src == "I are happy ."
    assert pairs[1].tgt == "I am happy ."
    # third sentence has no edits; src == tgt
    assert pairs[2].src == pairs[2].tgt == "Hello world ."
    assert all(p.source == "bea2019" for p in pairs)
    assert all(p.split == Split.TRAIN for p in pairs)


def test_missing_file_raises(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = BEA2019Source(config=cfg, split=Split.TRAIN)
    with pytest.raises(FileNotFoundError):
        list(src.iter_pairs())
