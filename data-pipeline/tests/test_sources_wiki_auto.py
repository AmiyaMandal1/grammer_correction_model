from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.wiki_auto import WikiAutoSource
from data_pipeline.types import StyleKind


def _fake_load(_name: str, _subset: str, split: str):
    return [
        {"normal_sentence": "He is an erudite gentleman.", "simple_sentence": "He is a smart man."},
    ]


def test_wiki_auto_pairs(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = WikiAutoSource(config=cfg)
    with patch("data_pipeline.sources.wiki_auto.load_dataset", side_effect=_fake_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 1
    assert pairs[0].src == "He is an erudite gentleman."
    assert pairs[0].tgt == "He is a smart man."
    assert pairs[0].meta["style_target"] == StyleKind.SIMPLIFY.value
    assert pairs[0].source == "wiki_auto"
