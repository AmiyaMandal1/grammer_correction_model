import shutil
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.conll2014 import CoNLL2014Source
from data_pipeline.types import Split


def test_loads_conll_m2(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src_dir = cfg.raw_dir / "conll2014"
    src_dir.mkdir()
    shutil.copy(fixtures_dir / "tiny.m2", src_dir / "official-2014.0.m2")

    src = CoNLL2014Source(config=cfg)
    pairs = list(src.iter_pairs())
    assert len(pairs) == 3
    assert all(p.split == Split.TEST for p in pairs)
    assert all(p.source == "conll2014" for p in pairs)
