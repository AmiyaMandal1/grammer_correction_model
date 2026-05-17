# Data Pipeline — Known Follow-ups

These items were flagged in the final code review of `feat/data-pipeline` and intentionally deferred. They should be tracked before downstream training plans (`gec-tagger-train`, `style-llm-train`) begin work that depends on them.

## High-priority follow-ups

### 1. CoNLL-2014 multi-annotator handling

`_parse_m2` in `sources/bea2019.py` collects every `A` line regardless of the annotator index (field 6 of the M² line). CoNLL-2014 has two annotators per sentence and applying both as a single combined edit set can produce a target string that matches neither annotator. BEA-2019 is single-annotator and unaffected.

**Fix:** parse the annotator id, then either yield one `Pair` per annotator or filter to annotator 0 only. Decide which is better for the GEC tagger's training signal during `gec-tagger-train`.

**Risk if ignored:** CoNLL-2014 eval JSONL produces wrong M² scores.

### 2. Sentence-initial APPEND semantics

`align_tokens_to_tags` in `tag_encoder.py` clamps the anchor to `src[0]` when inserting before the first token. GECToR conventionally treats `APPEND` as "insert *after* this token," so a sentence-initial insertion is recorded with the wrong position. Three fixes are possible:

- Use a leading sentinel `$START` token and anchor sentence-initial inserts to it.
- Introduce a `$PREPEND` tag for position-zero inserts.
- Document and accept the limitation if the downstream tagger model treats `APPEND` on `src[0]` as a prepend convention.

The choice should be made together with the tag vocabulary used in `gec-tagger-train`.

### 3. Near-duplicate dedup

`dedup.py` performs only exact dedup (xxhash on lowercased `(src, tgt)`). The plan's architecture summary promises "exact + near-duplicate dedup." C4_200M in particular contains many near-paraphrases that will inflate training-time recall on in-distribution eval.

**Fix:** add a MinHash pass over token 3-grams with a Jaccard threshold around 0.8, applied after exact dedup. Extend `DedupStats` with `removed_near`.

### 4. `build-all` previously overwrote `manifest.json` — fixed

`build-gec` now writes `manifest_gec.json` and `build-sft` writes `manifest_sft.json`. The combined `build-all` run produces both files side by side. No `manifest.json` is created. Downstream consumers should read whichever file matches the artifact they need.

The manifest now carries `"schema_version": "1"` so downstream loaders can pin to a compatible shape.

## Medium-priority follow-ups

### 5. ERRANT alignment is dead code in production

`errant_align.py` exists, is tested, and is exported, but no module in `data_pipeline` actually calls `ErrantAligner` in the production path. The builders use whitespace `.split()` on src/tgt. For M²-derived sources (BEA-2019, CoNLL-2014) the input is already token-aligned so whitespace split is correct; for paraphrastic sources (C4_200M, JFLEG) it produces token sequences that don't align to ERRANT's tokenization, which can confuse the tag encoder.

**Fix:** either wire `ErrantAligner` into the builder for paraphrastic sources, or delete the module if the whitespace approach is sufficient. Decision should consider whether the tagger model expects ERRANT-tokenized input.

### 6. GYAFC `Entertainment_Music` domain is silently dropped

`cli.py:_load_style_sources` hard-codes `domain="Family_Relationships"`. The `Entertainment_Music` domain is never loaded. Either expose a `--gyafc-domains` flag on `build-sft` or document the scope decision.

### 7. JSONL schemas should be typed

`gec_builder.py` emits `{tokens, tags, source, split, meta}` and `sft_builder.py` emits ChatML messages. Neither has a `TypedDict` or schema docstring. Add type definitions so downstream training scripts can import them as the contract.

### 8. Missing CLI e2e tests

`tests/test_cli_end_to_end.py` covers only `build-gec`. Add tests for `build-sft` (with mocked HF sources) and `build-all` (which now writes two manifests).

## Lower-priority follow-ups

### 9. Source dispatch should be a registry

`cli.py` uses `if/elif` for source name resolution. Replace with a `dict[str, type[Source]]` in `sources/__init__.py` so new sources do not require CLI edits.

### 10. `decode_tag` error paths lack tests

`tag_encoder.py:decode_tag` raises `ValueError` on three malformed-input paths. Add tests for each.

### 11. `polars` removed

Was declared but never imported. Removed from `pyproject.toml` dependencies in this branch.

### 12. Additional GEC sources from the spec

The spec §6.1 lists NUCLE, W&I+L, FCE, Lang-8, Newsela. The plan intentionally scoped to BEA-2019 (which already contains W&I+L, NUCLE, FCE, Lang-8 in the published train.m2 distribution), JFLEG, CoNLL-2014, C4_200M, GYAFC, ParaDetox, Wiki-Auto. Newsela is license-gated and was deferred. Add it as a source loader when the simplification corpus needs more data.
