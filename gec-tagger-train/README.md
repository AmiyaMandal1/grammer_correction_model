# gec-tagger-train

Fine-tunes DeBERTa-v3-base as a GECToR-style grammatical error tagger.

## Setup

```
cd gec-tagger-train
uv sync --extra dev
```

## Run

```
uv run gec-tagger-train build-vocab \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --out models/tags.json

uv run gec-tagger-train train \
    --stage 1 \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --tags models/tags.json \
    --out .checkpoints/stage1

uv run gec-tagger-train export \
    --checkpoint .checkpoints/stage1 \
    --out models/deberta-gec.onnx
```
