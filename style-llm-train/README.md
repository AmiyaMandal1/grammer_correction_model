# style-llm-train

LoRA fine-tunes `Qwen2.5-3B-Instruct` for style/tone rewriting. Outputs a GGUF model ready for the Rust runtime.

## Setup

```
cd style-llm-train
uv sync --extra dev
```

You must also clone `llama.cpp` and build `llama-quantize`:

```
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp && cmake -B build && cmake --build build --target llama-quantize
```

## Run

```
uv run style-llm-train train \
    --jsonl ../data-pipeline/data/processed/style_sft.jsonl \
    --out .checkpoints/style \
    --max-steps 1000

uv run style-llm-train merge \
    --adapter .checkpoints/style \
    --out outputs/qwen-style-merged

uv run style-llm-train quantize \
    --hf-checkpoint outputs/qwen-style-merged \
    --out models/qwen-style-q4.gguf \
    --llama-cpp-dir /tmp/llama.cpp \
    --quant Q4_K_M
```

## Apple Silicon notes

`bitsandbytes` 4-bit is CUDA-only. On M-series we keep the base model in fp16 and LoRA adapters in fp32; 4-bit quantization is applied at export time via `llama-quantize`. Memory is comfortable on 32GB Macs; 16GB Macs need `--per-device-batch-size 1 --gradient-accumulation-steps 8`.
