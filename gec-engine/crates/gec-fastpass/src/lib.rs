use anyhow::Result;
use gec_core::{Edit, EditCategory};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

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

    pub fn edits_for_tokens(&self, tokens: &[&str]) -> Result<Vec<Edit>> {
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
            let tag = self
                .vocab
                .tags
                .get(tag_idx)
                .map(String::as_str)
                .unwrap_or("$KEEP");
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
            byte_offset += tok_len + 1;
        }
        Ok(edits)
    }
}

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
        _path: std::path::PathBuf,
    }

    impl OrtEncoder {
        pub fn from_file(path: &Path) -> Result<Self> {
            Ok(Self {
                _path: path.to_path_buf(),
            })
        }
    }

    impl EncoderSession for OrtEncoder {
        fn forward(&self, _input_ids: &[i64], _attention_mask: &[i64]) -> Result<Array2<f32>> {
            anyhow::bail!("ort-runtime not wired yet — see gec-engine/FOLLOWUPS.md")
        }
    }
}
