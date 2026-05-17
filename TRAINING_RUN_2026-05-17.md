# Training Run — 2026-05-17

End-to-end smoke training across the data pipeline, GEC tagger, and style LLM, executed on macOS-arm64 (Apple M2 Max, 32 GB unified memory).

## Data sources

License-gated corpora (BEA-2019, GYAFC, CoNLL-2014) were not placed under `data/raw/`, so the run used only the publicly downloadable Hugging Face sources.

| Source | Records | Notes |
|---|---|---|
| JFLEG (dev + test) | 2,492 | Used as GEC training data |
| C4_200M | 0 | Hugging Face removed script-based datasets; loader fails with `RuntimeError: Dataset scripts are no longer supported, but found c4_200m.py` |
| Wiki-Auto | 0 | Same script-based-dataset error |
| ParaDetox | 18,993 | Detoxify SFT pairs |
| BEA-2019 / GYAFC / CoNLL-2014 | 0 | License-gated, files not placed |

A held-out 100-pair `jfleg_test.jsonl` was extracted under `gec-tagger-train/eval_data/`.

## GEC tagger run

Two training runs on `gec_tagger.jsonl` produced from JFLEG.

| Run | Vocab cutoff | Tags | Steps | Train loss | F0.5 (JFLEG test) | F0.5 (synthetic dev) |
|---|---|---|---|---|---|---|
| `.checkpoints/jfleg/ckpt` | min-count 2 | 915 | 500 (1.6 epochs) | 3.401 | 0.0 | 0.0 |
| `.checkpoints/jfleg2/ckpt` | min-count 5 | 302 | 3000 (9.6 epochs) | 2.972 | 0.0 | 0.0 |

Both checkpoints collapsed to all-`$KEEP` predictions, so `apply_tags_once` produces no edits and ERRANT records zero true positives.

### Why the tagger collapsed

JFLEG references are full-sentence paraphrases, not minimal grammatical edits. After `align_tokens_to_tags` derives per-token edit tags, the vocabulary blows up with one-off paraphrase replacements: most appear only once. After frequency pruning, those become `$UNK`, and the only well-supported tag in training is `$KEEP`. The model learns the dominant class.

GECToR's published F0.5 numbers were obtained on BEA-2019 train.m2 (W&I + L + NUCLE + FCE + Lang-8) which contains millions of minimal-edit annotations. JFLEG alone cannot replicate that without different data preprocessing (e.g., back-translating each correction into a minimal edit chain).

### Reproduction commands

```
# Data
cd data-pipeline
uv run data-pipeline build-gec --root . --sources jfleg --split dev
uv run data-pipeline build-sft --root . --sources paradetox

# Tagger
cd ../gec-tagger-train
uv run gec-tagger-train build-vocab \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --out .checkpoints/jfleg2/tags.json --min-count 5

uv run gec-tagger-train train --stage 1 \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --tags .checkpoints/jfleg2/tags.json \
    --out .checkpoints/jfleg2/ckpt \
    --max-steps 3000 --per-device-batch-size 8 --learning-rate 5e-5 \
    --max-length 128 --num-epochs 30

uv run gec-tagger-train eval \
    --checkpoint .checkpoints/jfleg2/ckpt \
    --tags .checkpoints/jfleg2/tags.json \
    --dev eval_data/jfleg_test.jsonl
```

Training runtime: 1620 s (≈27 min) for 3000 steps on a single M2 Max with `pin_memory` disabled (MPS does not support it).

## Style LLM run

### First attempt — failure (fp16 + MPS)

| Item | Value |
|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` |
| Steps | 200 |
| Train loss | 0 (all NaN, aggregator returns 0) |
| Entropy | NaN |
| Mean token accuracy | 0.0115 (random) |
| GPU recoveryCount | 15 |
| Adapter saved? | yes (35 MB) but no weight updates after step ≈44 |

The training nominally ran 200/200 steps but every Metal command buffer after step ~44 was rejected with:

```
Error: command buffer exited with error status.
Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)
```

Root cause: fp16 base + `gradient_checkpointing_enable()` on Apple MPS produces overflow in the rotary embedding + RMSNorm path. The Metal driver hits 15 GPU resets and then places the queue in "ignore" mode for the rest of the session. Hugging Face's NaN-safe loss aggregator reports `0.0`, so the run looks superficially successful.

### Fix

Patch in `style-llm-train/src/style_llm_train/trainer.py`:

- Detect Apple Silicon at runtime.
- On MPS, load the base in `torch.float32` and skip `gradient_checkpointing_enable()`.
- Keep fp16 + gradient checkpointing on CUDA/CPU.

```python
on_mps = platform.system() == "Darwin" and platform.machine() == "arm64"
dtype = torch.float32 if on_mps else torch.float16
base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, ...)
base.config.use_cache = False
if not on_mps:
    try:
        base.gradient_checkpointing_enable()
    except Exception:
        pass
```

### Second attempt — success

Same hyperparameters, fp32 base on MPS, gradient checkpointing off.

| Metric | Value |
|---|---|
| Train runtime | 228.6 s (≈4 min) |
| Train loss | 1.61 |
| Entropy | 1.506 |
| Mean token accuracy | 0.7293 |
| GPU recovery count | 0 (no Metal resets) |
| Adapter | `style-llm-train/.checkpoints/style/adapter_model.safetensors` (35 MB) |

Generation smoke test with the merged adapter (greedy, max_new_tokens=40):

| Input | Output |
|---|---|
| `that is dumb` | `That is not good` |
| `you suck at this` | `You are not good at this` |
| `he is a fucking idiot` | `He is a fool` |

The model learned the ParaDetox distribution. Not production quality (200 steps × 8 grad-accum on a 0.5 B base sees only ~1,600 records out of 18,993) but the detoxification signal is unambiguous and the fp32 fix is confirmed to resolve the MPS NaN failure.

## Known limitations

The end-to-end pipeline is functional but smoke-trained models are not production-grade. Real numbers require:

1. **License-gated corpora**: register and place BEA-2019 train.m2 (≈8 M minimal-edit pairs) under `data-pipeline/data/raw/bea2019/`. The same for GYAFC and CoNLL-2014.
2. **C4_200M alternative**: the upstream HF dataset is no longer loadable via the `datasets` library. Use a parquet-mirror (e.g. `nbroad/c4_200m_gec_train_clean`) once the loader is updated.
3. **TRANSFORM_VERB decoding**: the decoder currently treats verb-form transforms as `$KEEP`. Add `lemminflect` and a conversion table.
4. **CoNLL-2014 multi-annotator handling**: parser merges both annotators. Filter to one or yield separately.
5. **mlx-lm for the style LLM**: faster + native on M-series than PyTorch+MPS. Switch when scaling beyond 0.5 B params.

Open follow-ups already tracked in each sub-project's `FOLLOWUPS.md`.
