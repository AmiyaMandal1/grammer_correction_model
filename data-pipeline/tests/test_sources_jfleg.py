from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.jfleg import JFLEGSource
from data_pipeline.types import Split


def _fake_hf_load(_name: str, split: str):
    # Mimic JFLEG row shape from Hugging Face: {sentence, corrections: [..]}
    return [
        {"sentence": "He go home.", "corrections": ["He goes home.", "He went home."]},
        {"sentence": "I love programs.", "corrections": ["I love programming."]},
    ]


def test_jfleg_expands_corrections(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = JFLEGSource(config=cfg, split=Split.DEV)
    with patch("data_pipeline.sources.jfleg.load_dataset", side_effect=_fake_hf_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 3  # 2 + 1
    assert pairs[0].src == "He go home."
    assert pairs[0].tgt == "He goes home."
    assert pairs[1].src == "He go home."
    assert pairs[1].tgt == "He went home."
    assert all(p.source == "jfleg" for p in pairs)
