from pathlib import Path
from unittest.mock import patch

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.c4_200m import C4200MSource
from data_pipeline.types import Split


class _FakeStream:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


def _fake_load(_name: str, split: str, streaming: bool):
    assert streaming is True
    return _FakeStream(
        [
            {"input": "He go home.", "output": "He goes home."},
            {"input": "She wnt there.", "output": "She went there."},
            {"input": "third row src", "output": "third row tgt"},
            {"input": "fourth row src", "output": "fourth row tgt"},
        ]
    )


def test_c4_200m_streams_and_caps(tmp_path: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src = C4200MSource(config=cfg, max_rows=2)
    with patch("data_pipeline.sources.c4_200m.load_dataset", side_effect=_fake_load):
        pairs = list(src.iter_pairs())
    assert len(pairs) == 2
    assert pairs[0].src == "He go home."
    assert pairs[0].tgt == "He goes home."
    assert all(p.source == "c4_200m" for p in pairs)
    assert all(p.split == Split.TRAIN for p in pairs)
