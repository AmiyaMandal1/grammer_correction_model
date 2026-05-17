import pytest

from data_pipeline.leakage import LeakageError, check_leakage
from data_pipeline.types import Pair, Split


def _p(src: str, tgt: str, split: Split) -> Pair:
    return Pair(src=src, tgt=tgt, source="t", split=split)


def test_no_leakage_passes() -> None:
    train = [_p("a", "b", Split.TRAIN)]
    dev = [_p("c", "d", Split.DEV)]
    check_leakage(train=train, eval_sets={"dev": dev})


def test_train_dev_overlap_raises() -> None:
    train = [_p("hello world", "Hello world.", Split.TRAIN)]
    dev = [_p("hello world", "Hello world.", Split.DEV)]
    with pytest.raises(LeakageError) as exc:
        check_leakage(train=train, eval_sets={"dev": dev})
    assert "dev" in str(exc.value)
    assert "1" in str(exc.value)


def test_train_test_overlap_raises() -> None:
    train = [_p("foo", "foo.", Split.TRAIN)]
    test = [_p("FOO", "foo.", Split.TEST)]  # case-insensitive match on src
    with pytest.raises(LeakageError):
        check_leakage(train=train, eval_sets={"test": test})
