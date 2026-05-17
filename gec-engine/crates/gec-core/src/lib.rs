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
