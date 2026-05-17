from __future__ import annotations

import subprocess
from pathlib import Path


def convert_and_quantize(
    *,
    hf_checkpoint: Path,
    out: Path,
    llama_cpp_dir: Path,
    quant: str = "Q4_K_M",
) -> Path:
    """Run llama.cpp convert + quantize as subprocesses.

    Produces an intermediate `<out>.f16.gguf` then a final quantized GGUF.
    Requires `llama_cpp_dir/convert_hf_to_gguf.py` and a built
    `llama_cpp_dir/build/bin/llama-quantize` binary.
    """
    if not llama_cpp_dir.is_dir():
        raise FileNotFoundError(f"llama.cpp dir not found: {llama_cpp_dir}")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not convert_script.is_file():
        raise FileNotFoundError(f"missing converter: {convert_script}")
    if not quantize_bin.is_file():
        raise FileNotFoundError(
            f"missing quantize binary (build llama.cpp first): {quantize_bin}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    intermediate = out.with_suffix(".f16.gguf")

    subprocess.run(
        [
            "python",
            str(convert_script),
            str(hf_checkpoint),
            "--outfile",
            str(intermediate),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    subprocess.run(
        [str(quantize_bin), str(intermediate), str(out), quant],
        check=True,
    )
    return out
