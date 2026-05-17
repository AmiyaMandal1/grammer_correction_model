# gec-engine — Known Follow-ups

## Runtime integration

### 1. `ort-runtime` feature wiring

`gec_fastpass::ort_runtime::OrtEncoder::forward` returns an error today. Wire the real `ort::Session::run` call: load `*.onnx`, prepare `Value` inputs for `input_ids` and `attention_mask`, run, extract `logits` as `ndarray::Array2<f32>`. Iterate at most three times, stopping on a no-op. Requires shipping ONNX Runtime as a sibling binary (macOS-arm64) and pointing `ort` at it via `ORT_DYLIB_PATH`.

### 2. `llama-runtime` feature wiring

`gec_deeppass::llama_runtime::LlamaCppLlm::generate` returns an error today. Wire `llama_cpp_2::Llama` for the model load plus `LlamaContext` per request worker. Use `add_to_batch` + `decode` loop, sample via `llama_cpp_2::sampling`. Honor `max_tokens`. Stream tokens back through a channel for SSE on `/rewrite`.

### 3. `gec-server` actually wires both passes

The server currently returns stub responses with `degraded: true`. Plumb the fast pass and the deep pass behind handlers using an `Arc<AppState>` that holds loaded models + tokenizer. Use a bounded `Semaphore` per pass (4 fast, 1 deep per the design spec).

## Decoding rigor

### 4. Sentence splitting before the fast pass

The fast pass should split inputs on sentence boundaries before tokenization to respect the 512-token cap. Use `unicode-segmentation` for grapheme-aware splitting plus a sentence-boundary heuristic.

### 5. Chunker for the deep pass

The design spec sets `ctx_size = 16384` and chunks longer documents at paragraph boundaries with 128-token overlap. Implement in `gec-deeppass::chunker` once the LLM runtime is live.

## Observability

### 6. Tracing fields per request

The design spec lists `request_id`, `text_len`, `n_chunks`, `fast_ms`, `deep_ms`, `degraded`, `partial`. Only `request_id` is emitted today (in the response payload). Add `tracing::info_span!` around each pass and propagate `request_id` via `tracing::Span::current()`.

### 7. Counters and histograms

Spec calls for `requests_total`, `errors_total{code}`, `latency_ms{stage}`, `tokens_generated`. Use `metrics` + `metrics-exporter-prometheus` behind an optional feature once the integrations land.

## Validation

### 8. Hash check on model files

The design spec calls for sha256 verification of model files against the manifest at startup. Deferred until #1 and #2 land.

### 9. Smoke runbook

Add `scripts/smoke.sh` once the runtimes are live: load a tiny ONNX + GGUF, hit `/correct` and `/rewrite`, assert HTTP 200 plus a non-degraded response.
