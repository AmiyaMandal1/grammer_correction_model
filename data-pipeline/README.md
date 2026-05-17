# data-pipeline

Builds `gec_tagger.jsonl` and `style_sft.jsonl` from public corpora.

## Setup

```
cd data-pipeline
uv sync --extra dev
```

The spaCy model `en_core_web_sm` is locked as a dependency, so no manual download is required.

## Run

```
uv run data-pipeline build-all --root .
```

Outputs:
- `data/processed/gec_tagger.jsonl`
- `data/processed/style_sft.jsonl`
- `data/processed/manifest.json`
