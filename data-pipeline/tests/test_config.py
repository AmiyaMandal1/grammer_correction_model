from pathlib import Path

from data_pipeline.config import PipelineConfig


def test_default_config_paths(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path)
    assert cfg.raw_dir == tmp_path / "data" / "raw"
    assert cfg.interim_dir == tmp_path / "data" / "interim"
    assert cfg.processed_dir == tmp_path / "data" / "processed"


def test_config_ensures_dirs(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    assert cfg.raw_dir.is_dir()
    assert cfg.interim_dir.is_dir()
    assert cfg.processed_dir.is_dir()
