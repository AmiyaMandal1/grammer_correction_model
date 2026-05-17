from pathlib import Path
from unittest.mock import patch

import pytest

from style_llm_train.gguf_export import convert_and_quantize


def test_convert_and_quantize_invokes_subprocesses(tmp_out: Path) -> None:
    hf_dir = tmp_out / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    llama_dir = tmp_out / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_hf_to_gguf.py").write_text("# stub")
    build_dir = llama_dir / "build" / "bin"
    build_dir.mkdir(parents=True)
    (build_dir / "llama-quantize").write_text("# stub")

    out = tmp_out / "out.gguf"

    def fake_run(args, check, cwd=None):  # type: ignore[no-untyped-def]
        if "convert_hf_to_gguf.py" in args[1]:
            (out.with_suffix(".f16.gguf")).write_text("stub")
        return None

    with patch("style_llm_train.gguf_export.subprocess.run", side_effect=fake_run) as run:
        convert_and_quantize(
            hf_checkpoint=hf_dir,
            out=out,
            llama_cpp_dir=llama_dir,
            quant="Q4_K_M",
        )
    assert run.call_count == 2


def test_missing_llama_cpp_dir_raises(tmp_out: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_and_quantize(
            hf_checkpoint=tmp_out,
            out=tmp_out / "x.gguf",
            llama_cpp_dir=tmp_out / "no-such-dir",
            quant="Q4_K_M",
        )
