use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;
use tokenizers::Encoding;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("from_file: {e}"))?;
        Ok(Self { inner })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let inner: HfTokenizer = serde_json::from_str(json).context("parse tokenizer JSON")?;
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
