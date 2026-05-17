# GEC Engine (Rust) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rust `gec-engine/` workspace from the design spec §6.4: a single-binary serving runtime exposing `/correct` and `/rewrite` HTTP routes plus a CLI, loading the DeBERTa ONNX tagger (`ort` crate) and the Qwen2.5 GGUF style model (`llama-cpp-2`), and emitting a unified `Edit` stream.

**Architecture:** Cargo workspace with seven library crates and two binaries. `gec-core` holds shared types (`Edit`, `Tone`, `EditCategory`, request/response). `gec-tokenizer` wraps the `tokenizers` crate. `gec-fastpass` runs the ONNX encoder via `ort`, decodes tags into `Edit`s, and iterates up to three times. `gec-deeppass` wraps `llama-cpp-2` and produces edits by diffing rewritten output against the source. `gec-aligner` merges/dedupes outputs of both passes. `gec-cli` is a `clap` binary; `gec-server` is an `axum` HTTP service.

**Tech Stack:** Rust stable (edition 2021), `tokio`, `axum`, `clap`, `ort` (ONNX Runtime), `tokenizers`, `llama-cpp-2`, `serde`, `serde_json`, `tracing`, `thiserror`, `anyhow`, `proptest` (for the aligner's invariants).

External tools at runtime:
- ONNX model file (`deberta-gec.onnx` from `gec-tagger-train` export)
- `tags.json` and `tokenizer.json` (co-located by the tagger's export command)
- GGUF file (`qwen-style-q4.gguf` from `style-llm-train` quantize)
- Built `libllama.dylib` (or static lib) from `llama.cpp` for the `llama-cpp-2` crate's link step

---

## Layout

```
gec-engine/
├── Cargo.toml                # workspace manifest
├── README.md
├── rust-toolchain.toml
├── .gitignore
├── crates/
│   ├── gec-core/
│   │   ├── Cargo.toml
│   │   ├── src/lib.rs
│   │   └── tests/types.rs
│   ├── gec-tokenizer/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── gec-fastpass/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── gec-deeppass/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── gec-aligner/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── gec-cli/
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── gec-server/
│       ├── Cargo.toml
│       └── src/main.rs
```

`models/`, `target/`, `*.onnx`, `*.gguf` are gitignored.

---

## Task 1: Workspace bootstrap

**Files:**
- Create: `gec-engine/Cargo.toml` (workspace)
- Create: `gec-engine/rust-toolchain.toml`
- Create: `gec-engine/.gitignore`
- Create: `gec-engine/README.md`

- [ ] **Step 1: `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = [
    "crates/gec-core",
    "crates/gec-tokenizer",
    "crates/gec-fastpass",
    "crates/gec-deeppass",
    "crates/gec-aligner",
    "crates/gec-cli",
    "crates/gec-server",
]

[workspace.package]
edition = "2021"
rust-version = "1.80"
license = "Apache-2.0"
publish = false

[workspace.dependencies]
anyhow = "1"
thiserror = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1.40", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
axum = "0.7"
clap = { version = "4.5", features = ["derive"] }
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
ort = { version = "2.0.0-rc.4", default-features = false, features = ["ndarray"] }
ndarray = "0.16"
llama-cpp-2 = "0.1"
proptest = "1.5"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

- [ ] **Step 2: `rust-toolchain.toml`**

```toml
[toolchain]
channel = "stable"
components = ["clippy", "rustfmt"]
```

- [ ] **Step 3: `.gitignore`**

```
target/
*.onnx
*.gguf
models/
.cargo/
*.swp
```

- [ ] **Step 4: `README.md`**

````markdown
# gec-engine

Rust serving runtime for the GEC tagger + style LLM. Single binary exposing CLI and HTTP.

## Build

```
cd gec-engine
cargo build --release
```

## Run

```
./target/release/gec-server \
    --encoder-onnx /path/to/deberta-gec.onnx \
    --encoder-tags /path/to/tags.json \
    --encoder-tokenizer /path/to/tokenizer.json \
    --llm-gguf /path/to/qwen-style-q4.gguf \
    --bind 127.0.0.1:8080
```

## Endpoints

- `POST /correct` — grammar/spelling/punctuation. Body: `{"text": "..."}`.
- `POST /rewrite` — tone/style rewrite. Body: `{"text": "...", "tone": "formal|informal|concise|simplify|detoxify"}`.
- `GET /healthz` — deep health check.
````

- [ ] **Step 5: Verify and commit**

From the repo root:

```
cd gec-engine && cargo metadata --no-deps --format-version 1 > /dev/null
```

Should exit 0 (no members yet but the workspace manifest parses).

```
git add gec-engine/
git commit -m "feat(gec-engine): bootstrap Rust workspace"
```

---

## Task 2: `gec-core` types

**Files:**
- Create: `gec-engine/crates/gec-core/Cargo.toml`
- Create: `gec-engine/crates/gec-core/src/lib.rs`
- Create: `gec-engine/crates/gec-core/tests/types.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-core"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-core/tests/types.rs`:

```rust
use gec_core::{Edit, EditCategory, Tone, CorrectionResponse, ResponseStats};

#[test]
fn edit_json_round_trip() {
    let edit = Edit {
        span: 4..6,
        original: "go".into(),
        replacement: "goes".into(),
        category: EditCategory::Grammar,
        confidence: 0.93,
    };
    let s = serde_json::to_string(&edit).unwrap();
    let parsed: Edit = serde_json::from_str(&s).unwrap();
    assert_eq!(parsed.span, 4..6);
    assert_eq!(parsed.replacement, "goes");
    assert_eq!(parsed.category, EditCategory::Grammar);
}

#[test]
fn tone_serialization_lowercase() {
    assert_eq!(serde_json::to_string(&Tone::Formal).unwrap(), "\"formal\"");
    let parsed: Tone = serde_json::from_str("\"detoxify\"").unwrap();
    assert!(matches!(parsed, Tone::Detoxify));
}

#[test]
fn correction_response_default_flags_false() {
    let resp = CorrectionResponse {
        edits: vec![],
        corrected_text: "".into(),
        degraded: false,
        partial: false,
        request_id: "abc".into(),
        stats: ResponseStats::default(),
    };
    assert!(!resp.degraded);
    assert!(!resp.partial);
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-core/src/lib.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Tone {
    Formal,
    Informal,
    Concise,
    Simplify,
    Detoxify,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "tone", rename_all = "lowercase")]
pub enum EditCategory {
    Grammar,
    Spelling,
    Punctuation,
    Style(Tone),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edit {
    pub span: Range<usize>,
    pub original: String,
    pub replacement: String,
    pub category: EditCategory,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseStats {
    pub fast_ms: u64,
    pub deep_ms: u64,
    pub n_chunks: u32,
    pub tokens_generated: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionResponse {
    pub edits: Vec<Edit>,
    pub corrected_text: String,
    pub degraded: bool,
    pub partial: bool,
    pub request_id: String,
    pub stats: ResponseStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionRequest {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteRequest {
    pub text: String,
    pub tone: Tone,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("input too large: {0} bytes > {1}")]
    InputTooLarge(usize, usize),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("encoder inference: {0}")]
    Encoder(String),
    #[error("llm inference: {0}")]
    Llm(String),
    #[error("model overloaded")]
    Overloaded,
    #[error("generation timeout after {0} ms")]
    Timeout(u64),
    #[error("internal: {0}")]
    Internal(String),
}
```

- [ ] **Step 4: Test, expect PASS**

```
cd gec-engine && cargo test -p gec-core
```

Expected: 3 PASS.

- [ ] **Step 5: clippy + fmt**

```
cargo clippy -p gec-core --all-targets -- -D warnings
cargo fmt -p gec-core --check
```

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-core/
git commit -m "feat(gec-engine): gec-core shared types + JSON round-trip"
```

---

## Task 3: `gec-tokenizer` wrapper

**Files:**
- Create: `gec-engine/crates/gec-tokenizer/Cargo.toml`
- Create: `gec-engine/crates/gec-tokenizer/src/lib.rs`
- Create: `gec-engine/crates/gec-tokenizer/tests/wrap.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-tokenizer"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
gec-core = { path = "../gec-core" }
tokenizers = { workspace = true }
anyhow = { workspace = true }
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-tokenizer/tests/wrap.rs`:

```rust
use gec_tokenizer::Tokenizer;

#[test]
fn loads_from_json_string() {
    // A minimal tokenizers JSON producing simple word-level tokenization.
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#;
    let tok = Tokenizer::from_json(json).expect("loads");
    let enc = tok.encode("hello world", false).expect("encodes");
    let ids = enc.get_ids();
    assert_eq!(ids, &[1, 2]);
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-tokenizer/src/lib.rs`:

```rust
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;
use tokenizers::Encoding;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("from_file: {e}"))?;
        Ok(Self { inner })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let inner: HfTokenizer = serde_json::from_str(json)
            .context("parse tokenizer JSON")?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Encoding> {
        self.inner
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
```

- [ ] **Step 4: Test**

```
cargo test -p gec-tokenizer
```

- [ ] **Step 5: clippy + fmt**

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-tokenizer/
git commit -m "feat(gec-engine): gec-tokenizer thin wrapper over HF tokenizers"
```

---

## Task 4: `gec-aligner` edit merger

We do this before fastpass and deeppass so the merge target is well-defined. Pure-logic crate, fully testable without models.

**Files:**
- Create: `gec-engine/crates/gec-aligner/Cargo.toml`
- Create: `gec-engine/crates/gec-aligner/src/lib.rs`
- Create: `gec-engine/crates/gec-aligner/tests/merge.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-aligner"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
gec-core = { path = "../gec-core" }

[dev-dependencies]
proptest = { workspace = true }
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-aligner/tests/merge.rs`:

```rust
use gec_aligner::{merge_edits, apply_edits};
use gec_core::{Edit, EditCategory};

fn ed(span: std::ops::Range<usize>, orig: &str, repl: &str, cat: EditCategory, conf: f32) -> Edit {
    Edit {
        span,
        original: orig.into(),
        replacement: repl.into(),
        category: cat,
        confidence: conf,
    }
}

#[test]
fn merge_drops_lower_confidence_overlap() {
    let a = ed(4..6, "go", "goes", EditCategory::Grammar, 0.9);
    let b = ed(4..6, "go", "went", EditCategory::Grammar, 0.6);
    let merged = merge_edits(vec![a, b]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].replacement, "goes");
}

#[test]
fn merge_keeps_non_overlapping_edits() {
    let a = ed(0..2, "he", "He", EditCategory::Punctuation, 0.8);
    let b = ed(4..6, "go", "goes", EditCategory::Grammar, 0.9);
    let merged = merge_edits(vec![a, b]);
    assert_eq!(merged.len(), 2);
}

#[test]
fn apply_edits_substitutes_in_order() {
    let text = "he go home";
    let edits = vec![
        ed(3..5, "go", "goes", EditCategory::Grammar, 0.9),
        ed(0..2, "he", "He", EditCategory::Punctuation, 0.7),
    ];
    let out = apply_edits(text, &edits);
    assert_eq!(out, "He goes home");
}

#[test]
fn apply_edits_handles_length_changing_replacements() {
    let text = "he go";
    let edits = vec![
        ed(3..5, "go", "is going", EditCategory::Grammar, 0.8),
    ];
    let out = apply_edits(text, &edits);
    assert_eq!(out, "he is going");
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-aligner/src/lib.rs`:

```rust
use gec_core::Edit;

/// Merge edits from multiple passes, dropping overlapping lower-confidence
/// edits. Stable order (sorted by span start).
pub fn merge_edits(mut edits: Vec<Edit>) -> Vec<Edit> {
    // Sort by start ascending, confidence descending so higher-confidence
    // edits win on overlap.
    edits.sort_by(|a, b| {
        a.span
            .start
            .cmp(&b.span.start)
            .then(b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut kept: Vec<Edit> = Vec::new();
    for e in edits {
        if let Some(last) = kept.last() {
            if e.span.start < last.span.end {
                // Overlap: lower-confidence (or equal) loses.
                if e.confidence > last.confidence {
                    kept.pop();
                    kept.push(e);
                }
                continue;
            }
        }
        kept.push(e);
    }
    kept
}

/// Apply non-overlapping edits to a byte string, right-to-left so earlier
/// offsets stay valid.
pub fn apply_edits(text: &str, edits: &[Edit]) -> String {
    let mut sorted: Vec<&Edit> = edits.iter().collect();
    sorted.sort_by(|a, b| b.span.start.cmp(&a.span.start));
    let mut out = text.to_string();
    for e in sorted {
        out.replace_range(e.span.clone(), &e.replacement);
    }
    out
}
```

- [ ] **Step 4: Test**

```
cargo test -p gec-aligner
```

Expected: 4 PASS.

- [ ] **Step 5: clippy + fmt**

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-aligner/
git commit -m "feat(gec-engine): gec-aligner merges + applies non-overlapping edits"
```

---

## Task 5: `gec-fastpass` — encoder runner

We can't easily run a real ONNX model in unit tests without shipping one. Instead, the public API takes an `EncoderSession` trait we can mock. The real `ort`-backed implementation is wrapped behind a feature flag.

**Files:**
- Create: `gec-engine/crates/gec-fastpass/Cargo.toml`
- Create: `gec-engine/crates/gec-fastpass/src/lib.rs`
- Create: `gec-engine/crates/gec-fastpass/tests/decode.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-fastpass"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
gec-core = { path = "../gec-core" }
gec-tokenizer = { path = "../gec-tokenizer" }
anyhow = { workspace = true }
serde_json = { workspace = true }
ndarray = { workspace = true }
ort = { workspace = true, optional = true }

[features]
default = []
ort-runtime = ["dep:ort"]
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-fastpass/tests/decode.rs`:

```rust
use gec_fastpass::{decode_tag_string, FastPass, TagVocab, EncoderSession};
use gec_core::{Edit, EditCategory};
use ndarray::Array2;

struct StubEncoder {
    // For each call, returns this fixed [seq, num_tags] argmax.
    fixed_logits: Array2<f32>,
}

impl EncoderSession for StubEncoder {
    fn forward(&self, _input_ids: &[i64], _attention_mask: &[i64]) -> anyhow::Result<Array2<f32>> {
        Ok(self.fixed_logits.clone())
    }
}

#[test]
fn decode_tag_string_handles_keep() {
    let (kind, value) = decode_tag_string("$KEEP");
    assert_eq!(kind, "KEEP");
    assert_eq!(value, "");
}

#[test]
fn decode_tag_string_handles_replace() {
    let (kind, value) = decode_tag_string("$REPLACE_goes");
    assert_eq!(kind, "REPLACE");
    assert_eq!(value, "goes");
}

#[test]
fn decode_tag_string_handles_value_with_underscores() {
    let (kind, value) = decode_tag_string("$APPEND_well_done");
    assert_eq!(kind, "APPEND");
    assert_eq!(value, "well_done");
}

#[test]
fn tag_vocab_load_from_json() {
    let json = r#"{"tags":["$PAD","$UNK","$KEEP","$REPLACE_goes"]}"#;
    let vocab: TagVocab = serde_json::from_str(json).unwrap();
    assert_eq!(vocab.tags.len(), 4);
    assert_eq!(vocab.tags[3], "$REPLACE_goes");
}

#[test]
fn fastpass_decodes_tags_to_edits() {
    // 3 source tokens, 3 tag classes; argmax picks tag 1 ($REPLACE_goes) on token 2.
    let logits = Array2::from_shape_vec((3, 3), vec![
        0.9, 0.1, 0.0,   // tok 0 -> KEEP
        0.0, 0.9, 0.1,   // tok 1 -> REPLACE_goes
        0.9, 0.1, 0.0,   // tok 2 -> KEEP
    ]).unwrap();
    let stub = StubEncoder { fixed_logits: logits };
    let vocab = TagVocab { tags: vec!["$KEEP".into(), "$REPLACE_goes".into(), "$DELETE".into()] };
    let edits: Vec<Edit> = FastPass::with_encoder(stub, vocab).edits_for_tokens(&["he", "go", "home"]).unwrap();
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].replacement, "goes");
    assert_eq!(edits[0].category, EditCategory::Grammar);
    assert_eq!(edits[0].original, "go");
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-fastpass/src/lib.rs`:

```rust
use anyhow::Result;
use gec_core::{Edit, EditCategory};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Encoder forward: input ids + attention mask in, [seq, num_tags] logits out.
pub trait EncoderSession {
    fn forward(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Array2<f32>>;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TagVocab {
    pub tags: Vec<String>,
}

pub struct FastPass<E: EncoderSession> {
    encoder: E,
    vocab: TagVocab,
}

impl<E: EncoderSession> FastPass<E> {
    pub fn with_encoder(encoder: E, vocab: TagVocab) -> Self {
        Self { encoder, vocab }
    }

    /// Token-level entry point. Real callers will pre-tokenize text via the
    /// HF tokenizer; this signature lets us unit test without one.
    pub fn edits_for_tokens(&self, tokens: &[&str]) -> Result<Vec<Edit>> {
        // Pretend input_ids are 0..N; the stub encoder ignores them.
        let n = tokens.len() as i64;
        let ids: Vec<i64> = (0..n).collect();
        let mask: Vec<i64> = vec![1; n as usize];
        let logits = self.encoder.forward(&ids, &mask)?;
        let mut edits = Vec::new();
        let mut byte_offset = 0usize;
        for (idx, tok) in tokens.iter().enumerate() {
            let row = logits.row(idx);
            let (tag_idx, &conf) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            let tag = self.vocab.tags.get(tag_idx).map(String::as_str).unwrap_or("$KEEP");
            let (kind, value) = decode_tag_string(tag);
            let tok_len = tok.len();
            let span = byte_offset..byte_offset + tok_len;
            match kind.as_str() {
                "KEEP" => {}
                "DELETE" => edits.push(Edit {
                    span: span.clone(),
                    original: (*tok).into(),
                    replacement: String::new(),
                    category: EditCategory::Grammar,
                    confidence: conf,
                }),
                "REPLACE" => edits.push(Edit {
                    span: span.clone(),
                    original: (*tok).into(),
                    replacement: value,
                    category: EditCategory::Grammar,
                    confidence: conf,
                }),
                "APPEND" => edits.push(Edit {
                    span: span.end..span.end,
                    original: String::new(),
                    replacement: format!(" {}", value),
                    category: EditCategory::Grammar,
                    confidence: conf,
                }),
                _ => {}
            }
            byte_offset += tok_len + 1; // +1 for separating space
        }
        Ok(edits)
    }
}

/// Decode `$REPLACE_goes` -> ("REPLACE", "goes"). `$KEEP` and `$DELETE`
/// produce empty value. Unknown forms return the head with empty value.
pub fn decode_tag_string(s: &str) -> (String, String) {
    let body = s.strip_prefix('$').unwrap_or(s);
    if body == "KEEP" {
        return ("KEEP".into(), String::new());
    }
    if body == "DELETE" {
        return ("DELETE".into(), String::new());
    }
    match body.split_once('_') {
        Some((head, value)) => (head.to_string(), value.to_string()),
        None => (body.to_string(), String::new()),
    }
}

#[cfg(feature = "ort-runtime")]
pub mod ort_runtime {
    use super::{EncoderSession, Result};
    use ndarray::Array2;
    use std::path::Path;

    pub struct OrtEncoder {
        // Real ort session lives here. Implementation deferred to the
        // operator runbook because we can't ship a working ONNX file in
        // a test fixture; the trait above is the seam.
        _path: std::path::PathBuf,
    }

    impl OrtEncoder {
        pub fn from_file(path: &Path) -> Result<Self> {
            Ok(Self { _path: path.to_path_buf() })
        }
    }

    impl EncoderSession for OrtEncoder {
        fn forward(&self, _input_ids: &[i64], _attention_mask: &[i64]) -> Result<Array2<f32>> {
            // TODO: wire ort::Session::run. Document in FOLLOWUPS.md.
            anyhow::bail!("ort-runtime not wired yet — see gec-engine/FOLLOWUPS.md")
        }
    }
}
```

- [ ] **Step 4: Test**

```
cargo test -p gec-fastpass
```

Expected: 5 PASS.

- [ ] **Step 5: clippy + fmt**

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-fastpass/
git commit -m "feat(gec-engine): gec-fastpass tag decoder + EncoderSession trait"
```

---

## Task 6: `gec-deeppass` — LLM diff wrapper

Same pattern as fastpass: a `LlmSession` trait we can mock, with the real `llama-cpp-2` impl gated behind a feature flag.

**Files:**
- Create: `gec-engine/crates/gec-deeppass/Cargo.toml`
- Create: `gec-engine/crates/gec-deeppass/src/lib.rs`
- Create: `gec-engine/crates/gec-deeppass/tests/diff.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-deeppass"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
gec-core = { path = "../gec-core" }
anyhow = { workspace = true }
llama-cpp-2 = { workspace = true, optional = true }

[features]
default = []
llama-runtime = ["dep:llama-cpp-2"]
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-deeppass/tests/diff.rs`:

```rust
use gec_deeppass::{diff_to_edits, DeepPass, LlmSession};
use gec_core::{EditCategory, Tone};

struct StubLlm {
    out: String,
}

impl LlmSession for StubLlm {
    fn generate(&self, _prompt: &str, _max_tokens: usize) -> anyhow::Result<String> {
        Ok(self.out.clone())
    }
}

#[test]
fn diff_to_edits_substitution() {
    let edits = diff_to_edits("he go home", "he goes home", Tone::Formal);
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].original, "go");
    assert_eq!(edits[0].replacement, "goes");
    assert!(matches!(edits[0].category, EditCategory::Style(Tone::Formal)));
}

#[test]
fn diff_to_edits_no_change_returns_empty() {
    assert!(diff_to_edits("same text", "same text", Tone::Formal).is_empty());
}

#[test]
fn deeppass_rewrites_via_llm() {
    let llm = StubLlm { out: "He goes home.".into() };
    let dp = DeepPass::with_llm(llm);
    let (text, edits) = dp.rewrite("he go home.", Tone::Formal).unwrap();
    assert_eq!(text, "He goes home.");
    assert!(!edits.is_empty());
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-deeppass/src/lib.rs`:

```rust
use anyhow::Result;
use gec_core::{Edit, EditCategory, Tone};

pub trait LlmSession {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}

pub struct DeepPass<L: LlmSession> {
    llm: L,
}

impl<L: LlmSession> DeepPass<L> {
    pub fn with_llm(llm: L) -> Self {
        Self { llm }
    }

    pub fn rewrite(&self, text: &str, tone: Tone) -> Result<(String, Vec<Edit>)> {
        let prompt = build_prompt(text, &tone);
        let rewritten = self.llm.generate(&prompt, 1024)?;
        let edits = diff_to_edits(text, &rewritten, tone);
        Ok((rewritten, edits))
    }
}

fn build_prompt(text: &str, tone: &Tone) -> String {
    let system = match tone {
        Tone::Formal => "Rewrite the user's text in formal English while preserving meaning.",
        Tone::Informal => "Rewrite the user's text in casual, informal English.",
        Tone::Concise => "Rewrite the user's text more concisely.",
        Tone::Simplify => "Rewrite the user's text in simpler English.",
        Tone::Detoxify => "Rewrite the user's text in a neutral, non-toxic way.",
    };
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Token-level diff between original and rewritten text. Each contiguous
/// mismatch becomes one `Edit` with the source span and the rewritten span.
pub fn diff_to_edits(original: &str, rewritten: &str, tone: Tone) -> Vec<Edit> {
    let src_toks: Vec<&str> = original.split_whitespace().collect();
    let tgt_toks: Vec<&str> = rewritten.split_whitespace().collect();
    if src_toks == tgt_toks {
        return Vec::new();
    }
    // Naive whole-string replacement edit. A finer diff lives in a follow-up.
    vec![Edit {
        span: 0..original.len(),
        original: original.to_string(),
        replacement: rewritten.to_string(),
        category: EditCategory::Style(tone),
        confidence: 0.5,
    }]
}

#[cfg(feature = "llama-runtime")]
pub mod llama_runtime {
    use super::{LlmSession, Result};
    use std::path::Path;

    pub struct LlamaCppLlm {
        _path: std::path::PathBuf,
    }

    impl LlamaCppLlm {
        pub fn from_file(path: &Path) -> Result<Self> {
            Ok(Self { _path: path.to_path_buf() })
        }
    }

    impl LlmSession for LlamaCppLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
            // TODO: wire llama-cpp-2::Llama / ::Context. Documented in
            // gec-engine/FOLLOWUPS.md as the runtime integration milestone.
            anyhow::bail!("llama-runtime not wired yet — see gec-engine/FOLLOWUPS.md")
        }
    }
}
```

- [ ] **Step 4: Test**

```
cargo test -p gec-deeppass
```

Expected: 3 PASS.

- [ ] **Step 5: clippy + fmt**

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-deeppass/
git commit -m "feat(gec-engine): gec-deeppass diff helper + LlmSession trait"
```

---

## Task 7: `gec-cli` binary

**Files:**
- Create: `gec-engine/crates/gec-cli/Cargo.toml`
- Create: `gec-engine/crates/gec-cli/src/main.rs`
- Create: `gec-engine/crates/gec-cli/tests/help.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-cli"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[[bin]]
name = "gec-cli"
path = "src/main.rs"

[dependencies]
gec-core = { path = "../gec-core" }
gec-aligner = { path = "../gec-aligner" }
clap = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-cli/tests/help.rs`:

```rust
use std::process::Command;

#[test]
fn cli_help_prints_subcommands() {
    let exe = env!("CARGO_BIN_EXE_gec-cli");
    let out = Command::new(exe).arg("--help").output().expect("ran");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("apply-edits"), "stdout: {stdout}");
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-cli/src/main.rs`:

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};
use gec_aligner::apply_edits;
use gec_core::Edit;

#[derive(Parser)]
#[command(name = "gec-cli", about = "GEC engine command-line interface")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Read text on stdin and a JSON edits array from --edits, write
    /// the corrected text to stdout.
    ApplyEdits {
        #[arg(long)]
        edits: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ApplyEdits { edits } => {
            let edits: Vec<Edit> = serde_json::from_str(&edits)?;
            let mut input = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
            print!("{}", apply_edits(&input, &edits));
            Ok(())
        }
    }
}
```

- [ ] **Step 4: Test**

```
cargo test -p gec-cli
```

- [ ] **Step 5: clippy + fmt**

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-cli/
git commit -m "feat(gec-engine): gec-cli with apply-edits subcommand"
```

---

## Task 8: `gec-server` binary

**Files:**
- Create: `gec-engine/crates/gec-server/Cargo.toml`
- Create: `gec-engine/crates/gec-server/src/main.rs`
- Create: `gec-engine/crates/gec-server/tests/healthz.rs`

- [ ] **Step 1: `Cargo.toml`**

```toml
[package]
name = "gec-server"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true

[[bin]]
name = "gec-server"
path = "src/main.rs"

[dependencies]
gec-core = { path = "../gec-core" }
axum = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
clap = { workspace = true }
anyhow = { workspace = true }

[dev-dependencies]
tower = { version = "0.5", features = ["util"] }
http-body-util = "0.1"
```

- [ ] **Step 2: Failing test**

`gec-engine/crates/gec-server/tests/healthz.rs`:

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use gec_server::router;
use tower::ServiceExt;

#[tokio::test]
async fn healthz_returns_ok() {
    let app = router();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn correct_returns_empty_edits_for_empty_text() {
    let app = router();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/correct")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(body_str.contains("\"edits\""));
    assert!(body_str.contains("\"corrected_text\""));
}
```

- [ ] **Step 3: Implementation**

`gec-engine/crates/gec-server/src/main.rs`:

```rust
use anyhow::Result;
use axum::{
    extract::Json as JsonExtract,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use gec_core::{
    CorrectionRequest, CorrectionResponse, ResponseStats, RewriteRequest,
};

#[derive(Parser, Debug)]
#[command(name = "gec-server")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long)]
    encoder_onnx: Option<String>,
    #[arg(long)]
    encoder_tags: Option<String>,
    #[arg(long)]
    encoder_tokenizer: Option<String>,
    #[arg(long)]
    llm_gguf: Option<String>,
}

pub fn router() -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/correct", post(correct))
        .route("/rewrite", post(rewrite))
}

async fn healthz() -> &'static str {
    "ok"
}

async fn correct(
    JsonExtract(req): JsonExtract<CorrectionRequest>,
) -> Result<Json<CorrectionResponse>, (StatusCode, String)> {
    // Stub implementation: returns empty edits. Wired model paths land
    // in the runtime-integration follow-up.
    Ok(Json(CorrectionResponse {
        edits: vec![],
        corrected_text: req.text,
        degraded: true,
        partial: false,
        request_id: uuid_v4(),
        stats: ResponseStats::default(),
    }))
}

async fn rewrite(
    JsonExtract(req): JsonExtract<RewriteRequest>,
) -> Result<Json<CorrectionResponse>, (StatusCode, String)> {
    Ok(Json(CorrectionResponse {
        edits: vec![],
        corrected_text: req.text,
        degraded: true,
        partial: false,
        request_id: uuid_v4(),
        stats: ResponseStats::default(),
    }))
}

fn uuid_v4() -> String {
    // Lightweight non-cryptographic id: timestamp + nanos.
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}-{:x}", dur.as_secs(), dur.subsec_nanos())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let app = router();
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    tracing::info!("listening on {}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
```

The binary's `router` function is `pub` so the integration test can import it. To make that work, expose the function via a library target. The simplest path: split into a small `lib.rs` that re-exports `router`. Adjust this when implementing.

Actual layout to use:

`gec-engine/crates/gec-server/src/lib.rs`:

```rust
mod handlers;
pub use handlers::router;
```

`gec-engine/crates/gec-server/src/handlers.rs`:

```rust
// Move the router() + handlers from main.rs here.
```

`gec-engine/crates/gec-server/src/main.rs` becomes just:

```rust
use anyhow::Result;
use clap::Parser;
use gec_server::router;

// ...Args + #[tokio::main] body...
```

Update the Cargo.toml accordingly:

```toml
[lib]
name = "gec_server"
path = "src/lib.rs"

[[bin]]
name = "gec-server"
path = "src/main.rs"
```

- [ ] **Step 4: Test**

```
cargo test -p gec-server
```

Expected: 2 PASS.

- [ ] **Step 5: clippy + fmt**

- [ ] **Step 6: Commit**

```
git add gec-engine/crates/gec-server/
git commit -m "feat(gec-engine): gec-server axum routes (stub correct/rewrite)"
```

---

## Task 9: Workspace gates + FOLLOWUPS

**Files:**
- Create: `gec-engine/FOLLOWUPS.md`

- [ ] **Step 1: clippy across the workspace**

```
cd gec-engine && cargo clippy --workspace --all-targets -- -D warnings
```

Must exit 0.

- [ ] **Step 2: rustfmt across the workspace**

```
cargo fmt --all --check
```

Must exit 0.

- [ ] **Step 3: full test suite**

```
cargo test --workspace
```

Must exit 0. Expected total: 17+ tests (3 core + 1 tokenizer + 4 aligner + 5 fastpass + 3 deeppass + 1 cli + 2 server).

- [ ] **Step 4: build release binary**

```
cargo build --release --workspace
```

Must exit 0. Validates that both binaries link.

- [ ] **Step 5: Write `FOLLOWUPS.md`**

```markdown
# gec-engine — Known Follow-ups

## Runtime integration

### 1. `ort-runtime` feature wiring
`gec_fastpass::ort_runtime::OrtEncoder::forward` returns an error today. Wire the real `ort::Session::run` call: load `*.onnx`, prepare `Value` inputs for `input_ids` and `attention_mask`, run, extract `logits` as `ndarray::Array2<f32>`. Iterate ≤3 times stopping on no-op. Requires shipping ONNX Runtime as a sibling binary (macOS-arm64) and pointing `ort` at it via `ORT_DYLIB_PATH`.

### 2. `llama-runtime` feature wiring
`gec_deeppass::llama_runtime::LlamaCppLlm::generate` returns an error today. Wire `llama_cpp_2::Llama` model load + `LlamaContext` per request worker. Use `add_to_batch` + `decode` loop, sampling via `llama_cpp_2::sampling`. Honor max_tokens. Stream tokens back through a channel for SSE on `/rewrite`.

### 3. `gec-server` actually wires both passes
The server currently returns stub responses. Plumb fast pass and deep pass behind handlers using an `Arc<AppState>` holding loaded models + tokenizer. Use a bounded `Semaphore` per pass (4 fast / 1 deep per the design spec).

## Decoding rigor

### 4. Token-level diff in `diff_to_edits`
Current implementation emits a single whole-string replacement on any mismatch. Replace with a proper token-level diff (e.g., `dissimilar` crate) so individual span edits land in the response.

### 5. Sentence splitting before the fast pass
The fast pass should split inputs on sentence boundaries before tokenization to respect the 512-token cap. Use `unicode-segmentation` for grapheme-aware splitting plus a sentence-boundary heuristic.

### 6. Chunker for the deep pass
The design spec sets ctx_size 16384 and chunks longer documents at paragraph boundaries with 128-token overlap. Implement in `gec-deeppass::chunker` once the LLM runtime is live.

## Observability

### 7. Tracing fields per request
The design spec lists `request_id`, `text_len`, `n_chunks`, `fast_ms`, `deep_ms`, `degraded`, `partial`. Currently only `request_id` is emitted (in the response payload). Add `tracing::info_span!` around each pass and propagate `request_id` via `tracing::Span::current()`.

### 8. Counters and histograms
Spec calls for `requests_total`, `errors_total{code}`, `latency_ms{stage}`, `tokens_generated`. Use `metrics-rs` (Prometheus exporter, optional feature) once the integrations land.

## Validation

### 9. Hash check on model files
The design spec calls for sha256 verification of model files against the manifest at startup. The bootstrap loads no models, so this is deferred until #1 and #2 land.

### 10. Smoke runbook
Add a `scripts/smoke.sh` once the runtimes are live: download a tiny ONNX + GGUF, hit `/correct` and `/rewrite`, assert HTTP 200 + non-error response.
```

- [ ] **Step 6: Commit**

```
git add gec-engine/FOLLOWUPS.md
git commit -m "docs(gec-engine): record runtime-integration followups"
```

---

## Notes

- The trait-based seams (`EncoderSession`, `LlmSession`) keep the workspace tests pure-Rust with zero model-file dependency. The feature-gated `ort_runtime` and `llama_runtime` modules are scaffolds for the real integration, which is non-trivial and intentionally deferred.
- The server returns stub responses today. This is intentional and documented in `FOLLOWUPS.md`. Wiring the models is the next milestone.
- Final coverage target: ≥ 70% on `crates/gec-*` (matches the Python side; lower than 85% because the runtime integration layer is intentionally untested at unit level).
