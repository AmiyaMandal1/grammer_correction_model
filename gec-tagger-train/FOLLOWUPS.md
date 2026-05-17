# GEC Tagger Train — Known Follow-ups

Items raised by the final review that were intentionally deferred.

## High priority

### 1. Streaming dataset for C4_200M

`GECTaggerDataset._load` materializes every row into RAM at construction time. The C4_200M synthetic-GEC corpus is hundreds of millions of rows and will OOM long before stage-1 training starts.

**Fix:** switch to `IterableDataset` (the `datasets` package is already a declared dependency and supports streaming JSONL), tokenize on-the-fly in `__iter__`, and adjust `Trainer` configuration accordingly.

**Blocks:** any real-scale training run.

### 2. `eval` CLI command

The plan and the package's README mention `train`, `eval`, `export` as the three CLI commands. Only `train` and `export` are implemented. `eval` should:
- Load a checkpoint
- Iteratively apply tags (`apply_tags_once` loop, ≤3 iterations)
- Emit M² blocks via `eval_m2.format_m2_predictions`
- Optionally invoke the `m2scorer` binary as a subprocess and parse its output
- Optionally call `eval_errant.compute_errant_f05` on a (src, ref, hyp) triple

**Blocks:** spec §9.2 model-release quality gates.

### 3. Iterative decoding loop

`apply_tags_once` is a single round. GECToR runs ≤3 iterations and stops on a no-op pass. A wrapper `apply_tags_iterative(tokens, model_fn, max_iter=3)` is needed for accurate eval. Build this when `eval` lands.

### 4. `$TRANSFORM_VERB_*` decoding

`decode.apply_tags_once` handles `$TRANSFORM_CASE_*` but treats `$TRANSFORM_VERB_*` as `$KEEP`. Proper verb-form transformation requires a morphological inflector (e.g., `lemminflect`). Add the dependency and implement the conversion table from the GECToR paper when this affects quality.

## Medium priority

### 5. `StageConfig.from_checkpoint`

The multi-stage recipe (C4_200M → BEA-2019 → W&I+L) requires loading the prior stage's checkpoint before training the next. `StageConfig` and the `train` CLI currently always start from `microsoft/deberta-v3-base` pretrained weights; the `--stage` flag is cosmetic. Add a `from_checkpoint: Path | None` field and pass to `DebertaTagger`/`Trainer`.

### 6. M² fine-grained error codes

`eval_m2.format_m2_predictions` emits coarse codes (`R`, `U`, `M`). The CoNLL-2014 leaderboard requires ERRANT-style fine-grained codes (e.g., `R:VERB:SVA`). When `eval` lands, derive the code via ERRANT's classifier on the (src, hyp) diff.

### 7. Thread/process safety in `eval_errant`

`eval_errant._NLP` is a module-level global. spaCy pipelines are not fork-safe. If `compute_errant_f05` is called from `DataLoader` workers with `num_workers > 0`, results may corrupt or raise. For now `num_workers=0` is the safe default; document this if multi-worker eval is added later.

### 8. CLI tests for `export`

`test_cli_smoke.py` covers `build-vocab` and `train`. `export` has unit-level coverage via `test_export_onnx.py` but no end-to-end CLI test exercising the new tokenizer + tags.json copy paths. Add a smoke test that trains 1 step, then runs `export`, then asserts that `out.parent` contains `tags.json` and `tokenizer.json`.

## Lower priority

### 9. Unused dependencies

`pyproject.toml` lists `peft`, `evaluate`, and `datasets` as runtime dependencies. None are imported in `src/` today. `peft` is the LoRA dependency planned for the style-LLM project, not this one — should be moved when that plan lands. `evaluate` is planned for built-in metric integration. `datasets` will be needed for streaming (follow-up #1). Leave for now; revisit when the dependents are written.

### 10. `safetensors` is implicit

`cli._load_state_dict` imports `safetensors.torch` but the package is not declared in `pyproject.toml`. It arrives transitively via `accelerate`. Declare it explicitly or gate the import behind a clearer error.

### 11. `fp16` encoder upcast

`model.DebertaTagger.forward` upcasts the encoder's last hidden state to `float32` before the classifier head, working around a dtype mismatch observed when DeBERTa weights are loaded in fp16. The upcast is harmless but worth documenting; consider forcing the encoder to load in `float32` explicitly in `__init__`.
