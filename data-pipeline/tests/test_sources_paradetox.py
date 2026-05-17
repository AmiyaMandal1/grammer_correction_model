from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.paradetox import ParaDetoxSource
from data_pipeline.types import StyleKind


def _fake_load(_name: str, split: str):
    return [
        {"en_toxic_comment": "that is dumb", "en_neutral_comment": "that is not helpful"},
        {"en_toxic_comment": "you suck", "en_neutral_comment": "you are not good at this"},
    ]


def test_paradetox_pairs(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = ParaDetoxSource(config=cfg)
    with patch("data_pipeline.sources.paradetox.load_dataset", side_effect=_fake_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].src == "that is dumb"
    assert pairs[0].tgt == "that is not helpful"
    assert pairs[0].meta["style_target"] == StyleKind.DETOXIFY.value
    assert pairs[0].source == "paradetox"
