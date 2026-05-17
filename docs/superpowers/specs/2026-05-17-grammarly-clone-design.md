# Grammarly-Clone Local Engine — Design

**Date:** 2026-05-17
**Status:** Approved (brainstorming output)
**Owner:** amiyamandal

## 1. Goal

Build a local, open-source, Rust-deployable text-correction engine covering English grammar, spelling, punctuation, and tone/style rewriting. Two-pass design: a fast encoder-based corrector for inline edits, plus a deeper LLM-based rewriter for tone and style. Train on Apple Silicon; serve in Rust with Metal-accelerated inference.

## 2. Non-goals

- No browser extension, desktop UI, or editor integrations in this spec. The engine exposes a library, CLI, and HTTP API; clients are out of scope.
- No user accounts, billing, telemetry-to-cloud, or multi-tenant features.
- No multilingual support. English only.
- No from-scratch pretraining. Reuse public checkpoints.
- No retrieval, plagiarism check, or document-level coherence scoring.

## 3. Constraints

- Hardware: Apple Silicon (M-series), 16-32 GB unified RAM.
- Training: PyTorch + MPS, single device. LoRA / QLoRA only for the LLM. Full fine-tune acceptable for the encoder.
- Runtime: Rust, single binary preferred. Metal-accelerated LLM via `llama.cpp` bindings; encoder via `ort` (ONNX Runtime) or `candle`.
- Latency: fast pass < 150 ms per sentence (p95); deep pass < 3 s per 200-word paragraph (p95).
- Context: LLM default 16,384 tokens, configurable to 32,768 via YaRN. Encoder hard-capped at 512 tokens; long documents handled by sentence splitting and chunking.

## 4. Approach

Approach A from brainstorming: GECToR-style encoder tagger for fast correction, plus a LoRA-tuned small instruction LLM for tone and style. Edits from both passes converge through a single `Edit` type.

Alternatives considered and rejected:

- **B — T5 seq2seq for fast pass**: simpler paradigm but slower for inline use and lacks span output.
- **C — Single unified small LLM**: simpler stack but worse grammar precision and higher inline latency.

## 5. Architecture

```
Client (CLI / HTTP / future UI)
        │
        ▼
Rust serving layer (axum)
  /correct   /rewrite   /healthz
        │
   ┌────┴────┐
   ▼         ▼
FastPass   DeepPass
DeBERTa    Qwen2.5-3B GGUF Q4_K_M
ort/candle llama-cpp-2 (Metal)
   │         │
   └────┬────┘
        ▼
  Edit aligner + post-processor
        ▼
   Vec<Edit> response
```

Single Rust binary loads both models at startup, keeps them resident. Expected resident memory ~5 GB total at the default 16K context.

Repository layout:

- `data-pipeline/` — Python; dataset preparation.
- `gec-tagger-train/` — Python; DeBERTa GECToR fine-tune.
- `style-llm-train/` — Python; Qwen2.5 LoRA fine-tune.
- `gec-engine/` — Rust Cargo workspace; runtime (library, CLI, HTTP).

## 6. Components

### 6.1 `data-pipeline/`

Sources:

- **GEC:** BEA-2019 (train + dev), NUCLE, W&I+L, FCE, Lang-8, JFLEG (eval only), CoNLL-2014 (eval only), C4_200M synthetic GEC.
- **Style/tone:** GYAFC (formality), ParaDetox (toxicity rewrite), Wiki-Auto (simplification), Newsela.
- **Synthetic:** seed prompts plus GPT-4o-mini or Claude-generated pairs for tones not covered (concise, professional, friendly, confident).

Outputs:

- GEC tagger training set: `(tokens, edit_tags)` JSONL, produced via ERRANT alignment and the GECToR tag encoder.
- LLM SFT set: `{instruction, input, output}` JSONL in ChatML format.

Tooling: `datasets`, `errant`, `spacy`, `polars`. Pipeline emits a `manifest.json` with source hashes, row counts, and dedup statistics. Cross-set leakage check runs as a hard gate.

### 6.2 `gec-tagger-train/`

- Base model: `microsoft/deberta-v3-base`.
- Head: linear layer over a vocabulary of approximately 5,000 edit tags as in the GECToR paper. Tag categories cover `$KEEP`, `$DELETE`, `$APPEND_<tok>`, `$REPLACE_<tok>`, `$TRANSFORM_CASE_*`, `$TRANSFORM_VERB_*`, and punctuation operations.
- Loss: token-level cross-entropy with label smoothing 0.1.
- Training stages: (1) synthetic C4_200M pretrain, (2) BEA-2019 plus NUCLE, (3) high-quality W&I+L only.
- Export: HF checkpoint → ONNX (opset 17) plus `tokenizer.json`.
- Evaluation: M² scorer on CoNLL-2014, ERRANT on BEA-dev.

### 6.3 `style-llm-train/`

- Base model: `Qwen2.5-3B-Instruct`.
- Method: QLoRA — 4-bit base, LoRA r=16, alpha=32, dropout 0.05, targets `q/k/v/o` plus `gate/up/down`.
- Trainer: TRL `SFTTrainer`, sequence packing on, sequence length 2,048.
- Output: merge LoRA weights into the base, then `llama.cpp/convert_hf_to_gguf.py` and quantize to Q4_K_M.
- Evaluation: held-out style pairs, BLEU plus BERTScore, and G-Eval scoring via a local Qwen-2.5 judge.

### 6.4 `gec-engine/` (Rust workspace)

Crates:

- `gec-core` — shared types: `Edit`, `CorrectionRequest`, `CorrectionResponse`, `Tone`, `EditCategory`.
- `gec-tokenizer` — thin wrapper over the `tokenizers` crate; both BPE and SentencePiece variants.
- `gec-fastpass` — ONNX Runtime (`ort`) loads DeBERTa, runs iterative tagging up to 3 iterations, decodes tags into `Edit[]`.
- `gec-deeppass` — `llama-cpp-2` loads the GGUF, applies prompt templates per tone, streams generation, diffs output to produce `Edit[]`.
- `gec-aligner` — merges fast-pass and deep-pass edits, deduplicates overlapping spans, ranks by severity.
- `gec-cli` — `clap`-based CLI; reads stdin or file; emits JSON or annotated text.
- `gec-server` — `axum`; routes `/correct`, `/rewrite`, `/healthz`; OpenAPI via `utoipa`.

Public traits (mockable):

```rust
pub trait Corrector {
    fn correct(&self, text: &str, opts: CorrectOpts) -> Result<Vec<Edit>, EngineError>;
}

pub trait Rewriter {
    fn rewrite(&self, text: &str, tone: Tone) -> Result<String, EngineError>;
}
```

## 7. Data flow

### 7.1 Training-time

```
Raw corpora ─► data-pipeline ─┬─► gec_tagger.jsonl ─► gec-tagger-train ─► deberta-gec.onnx + vocab
                              │
                              └─► style_sft.jsonl  ─► style-llm-train  ─► qwen-style-q4.gguf
```

Artifacts are versioned under `models/v{N}/` together with a `manifest.json` recording file hashes, evaluation metrics, and the training commit SHA.

### 7.2 Inference — `/correct` (fast pass)

```
POST /correct {text}
  → sentence split (rust-tokenizers / unicode-segmentation)
  → batch (max 8 sentences)
  → gec-fastpass.tag(batch), iterate ≤3 times, stop on no-op
  → decode tags to Edit[]
  → post-process (dedup, remap byte offsets to original text)
  → JSON { edits, corrected_text }
```

### 7.3 Inference — `/rewrite` (deep pass)

```
POST /rewrite {text, tone}
  → optional fast-pass clean-up to reduce LLM noise
  → prompt_build(tone, cleaned_text) using ChatML template
  → gec-deeppass.generate(prompt, stream=true)
  → diff(original, rewritten) → Edit[] with category=Style(tone)
  → SSE stream: edits as produced; final corrected_text frame
```

### 7.4 Context-length policy

- Encoder hard cap 512 tokens. Always sentence-split. Long documents use batching, never truncation.
- LLM default context 16,384; configurable to 32,768 via YaRN. KV cache budget: ~2 GB at 16K, ~4 GB at 32K.
- Document chunker: paragraph boundaries via `unicode-segmentation` plus heuristic, overlap 128 tokens, stitch outputs at boundaries, drop overlap from the second chunk.
- SSE streaming emits per-chunk edits so the client can render incrementally on multi-page documents.

Runtime config (`config.toml`):

```toml
[llm]
ctx_size = 16384
n_batch = 512
rope_scaling = "yarn"
yarn_factor = 4.0

[chunker]
max_chunk_tokens = 12000
overlap_tokens = 128
```

### 7.5 Edit type (cross-cutting)

```rust
pub struct Edit {
    pub span: Range<usize>,         // byte offsets in original text
    pub original: String,
    pub replacement: String,
    pub category: EditCategory,     // Grammar | Spelling | Punctuation | Style(Tone)
    pub confidence: f32,            // tagger softmax max for fast pass; diff-derived heuristic for deep pass
}

pub struct CorrectionResponse {
    pub edits: Vec<Edit>,
    pub corrected_text: String,
    pub degraded: bool,             // true if deep pass fell back to fast-pass only
    pub partial: bool,              // true if a chunk was cut off (LLM timeout)
    pub request_id: String,
    pub stats: ResponseStats,       // fast_ms, deep_ms, n_chunks, tokens_generated
}
```

The `/rewrite` SSE stream emits `event: edit` frames containing one `Edit` each, then a terminating `event: done` frame containing the full `CorrectionResponse`.

## 8. Error handling

### 8.1 Failure taxonomy

| Failure | Stage | Strategy |
|---|---|---|
| Tokenizer OOV / malformed UTF-8 | input | NFKC normalize, replace invalid bytes with `U+FFFD`, log warn, continue |
| Empty / whitespace input | input | Return `{edits: [], corrected_text: ""}`, HTTP 200 |
| Input over `max_doc_size` (10 MB default) | input | HTTP 413, `input_too_large` |
| ONNX session load failure | startup | Fatal, exit 1 |
| GGUF load failure | startup | Fatal, exit 1 |
| Encoder inference OOM | runtime | Halve batch, retry once; persistent → HTTP 503 `model_overloaded` |
| LLM context overflow | runtime | Chunker bug; log, fall back to fast-pass-only response, mark `degraded: true` |
| LLM generation timeout (> 60 s / chunk) | runtime | Cancel; return partial output with `partial: true` |
| Tagger non-convergence (still tagging after 3 iterations) | runtime | Stop, return current state, log warn with text hash |
| Diff produces zero edits but LLM rewrote | runtime | Log; return original text; increment a metric |
| Concurrent burst | runtime | Bounded `tokio::sync::Semaphore` (4 fast, 1 deep); excess → HTTP 429 with `Retry-After` |
| Model file corruption (hash mismatch) | startup | Fatal, exit 1 |

### 8.2 Error type

```rust
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("input too large: {0} bytes > {1}")]
    InputTooLarge(usize, usize),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("encoder inference: {0}")]
    Encoder(#[from] ort::Error),
    #[error("llm inference: {0}")]
    Llm(#[from] llama_cpp_2::LLamaCppError),
    #[error("model overloaded")]
    Overloaded,
    #[error("generation timeout after {0:?}")]
    Timeout(std::time::Duration),
    #[error("internal: {0}")]
    Internal(String),
}
```

HTTP layer maps variants to status codes via `axum::IntoResponse`. JSON body: `{error_code, message, request_id}`.

### 8.3 Observability

- `tracing` with structured fields: `request_id`, `text_len`, `n_chunks`, `fast_ms`, `deep_ms`, `degraded`, `partial`.
- Optional `tracing-opentelemetry` exporter, env-gated, off by default.
- Counters: `requests_total`, `errors_total{code}`, `model_overload_total`, `timeout_total`, `degraded_total`.
- Histograms: `latency_ms{stage=fast|deep}`, `tokens_generated`.
- `/healthz` performs a deep check: tokenize, one-token encoder pass, one-token LLM pass.

### 8.4 Resource guards

- `mlock` model files to avoid macOS swap thrash.
- Startup memory check via `sysinfo`: refuse to load if free RAM is below model size × 1.3.
- Graceful shutdown on `SIGTERM`: drain in-flight requests, flush metrics, unload contexts.

### 8.5 Deliberate non-features

- No automatic retry on LLM errors (deterministic seed; retries hide bugs).
- No silent fallback to a smaller model.
- No persistent state. Restart is the recovery path.

## 9. Testing

### 9.1 Layers

| Layer | Scope | Tools | Frequency |
|---|---|---|---|
| Unit (Python) | tokenization, tag encode/decode, ERRANT align, dataset filters | `pytest` | per PR |
| Unit (Rust) | edit decoder, diff, chunker, prompt builder, error mapping | `cargo test` | per PR |
| Integration (Rust) | full pipeline with tiny stub models (5 MB ONNX, 50 MB GGUF) | `cargo test --features integration` | per PR |
| Model quality | M² on CoNLL-2014, ERRANT F0.5 on BEA-dev, JFLEG GLEU, GYAFC style accuracy | `eval/` Python | every checkpoint |
| End-to-end | spin server, hit endpoints, assert latency budget | `cargo test --test e2e` | nightly |
| Regression (golden) | curated 500-sentence corpus with expected edits | `cargo test --test golden` | per PR |
| Property tests | input/output invariants | `proptest` | per PR |
| Benchmark | latency p50/p95, throughput, RAM ceiling | `criterion` | nightly |

### 9.2 Model-release quality gates

| Metric | Target |
|---|---|
| BEA-dev ERRANT F0.5 | ≥ 0.65 |
| CoNLL-2014 M² F0.5 | ≥ 0.60 |
| JFLEG GLEU | ≥ 0.58 |
| GYAFC formal ↔ informal accuracy | ≥ 0.80 |
| False-positive rate on clean Wikipedia sample | ≤ 2% |
| p95 fast-pass latency (mean-length sentence) | < 150 ms |
| p95 deep-pass latency (200-word paragraph) | < 3 s |

A release is blocked if any threshold is missed.

### 9.3 Property invariants

- `correct(correct(t)) == correct(t)` (idempotent).
- All `edit.span` values are byte-valid UTF-8 boundaries within `[0, text.len)`.
- `apply_edits(text, edits) == corrected_text`.
- No edit `replacement` introduces tokens absent from the source vocabulary (no hallucinated Unicode).
- Empty input returns empty output, never an error.
- Fast pass on clean prose returns zero edits at least 98% of the time.

### 9.4 Test data

- Public eval sets cached in `eval/data/`, gitignored, hash-checked against a manifest.
- Golden corpus at `tests/golden/corpus.jsonl`, versioned and reviewed.
- Adversarial set: code blocks, URLs, emoji, mixed scripts, intentional artistic deviations (Joyce, Dickens) — engine must leave these alone.

### 9.5 CI

- GitHub Actions matrix `{macos-14, ubuntu-22} × {stable, beta}`.
- Heavy jobs gated behind a label or a nightly cron.
- `cargo audit` and `cargo-deny` for license and CVE checks.
- `ruff` and `mypy` for the Python side.

### 9.6 Explicit skips

- No live network in tests; all models offline.
- No human evaluation in CI; rely on G-Eval with a local judge model.
- No fuzzing; the input is plain text, parser-bug surface area is small.

## 10. Open items deferred

- Choice between `ort` and `candle` for the encoder — benchmark during implementation.
- Whether to ship a separate small "style classifier" before the LLM to skip unnecessary deep passes — revisit after baseline metrics.
- Speculative decoding for the LLM via `llama.cpp` draft model — performance tweak, not a design concern.

## 11. Out of scope for this spec, in scope later

- Browser extension or editor plugin clients (separate spec).
- Multilingual support (separate effort; requires new datasets and likely a separate model line).
- User-personalised corrections / preference learning (requires storage and identity, out of this spec).
