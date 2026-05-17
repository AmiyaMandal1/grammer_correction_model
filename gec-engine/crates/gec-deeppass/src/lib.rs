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

/// Token-level diff between original and rewritten text using LCS.
/// Collects contiguous mismatch regions and emits one `Edit` per region,
/// with byte spans pointing into `original`.
pub fn diff_to_edits(original: &str, rewritten: &str, tone: Tone) -> Vec<Edit> {
    // Collect tokens with their byte offsets from the original.
    let src_tokens: Vec<(&str, usize)> = token_spans(original);
    let tgt_tokens: Vec<&str> = rewritten.split_whitespace().collect();

    let src_words: Vec<&str> = src_tokens.iter().map(|(w, _)| *w).collect();
    if src_words == tgt_tokens {
        return Vec::new();
    }

    // Build LCS table.
    let m = src_words.len();
    let n = tgt_tokens.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if src_words[i - 1] == tgt_tokens[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Back-track to get the diff ops: (src_idx, tgt_idx, op) where op is
    // Keep, Delete-src, or Insert-tgt. We only care about mismatched runs.
    let mut ops: Vec<DiffOp> = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && src_words[i - 1] == tgt_tokens[j - 1] {
            ops.push(DiffOp::Keep(i - 1, ()));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push(DiffOp::Insert(j - 1));
            j -= 1;
        } else {
            ops.push(DiffOp::Delete(i - 1));
            i -= 1;
        }
    }
    ops.reverse();

    // Group consecutive non-Keep ops into replacement regions.
    let mut edits = Vec::new();
    let mut idx = 0;
    while idx < ops.len() {
        if matches!(ops[idx], DiffOp::Keep(..)) {
            idx += 1;
            continue;
        }
        // Start of a mismatch region: collect all consecutive non-Keep ops.
        let start = idx;
        while idx < ops.len() && !matches!(ops[idx], DiffOp::Keep(..)) {
            idx += 1;
        }
        let region = &ops[start..idx];

        // Src tokens deleted from this region.
        let deleted: Vec<usize> = region
            .iter()
            .filter_map(|op| {
                if let DiffOp::Delete(si) = op {
                    Some(*si)
                } else {
                    None
                }
            })
            .collect();

        // Tgt tokens inserted in this region.
        let inserted: Vec<usize> = region
            .iter()
            .filter_map(|op| {
                if let DiffOp::Insert(ti) = op {
                    Some(*ti)
                } else {
                    None
                }
            })
            .collect();

        if deleted.is_empty() && inserted.is_empty() {
            continue;
        }

        // Compute byte span in original: from start of first deleted token to
        // end of last deleted token (or a zero-width span after the previous
        // kept token for pure insertions).
        let (span_start, span_end, orig_text) = if !deleted.is_empty() {
            let first_si = *deleted.first().unwrap();
            let last_si = *deleted.last().unwrap();
            let byte_start = src_tokens[first_si].1;
            let byte_end = src_tokens[last_si].1 + src_tokens[last_si].0.len();
            let orig = original[byte_start..byte_end].to_string();
            (byte_start, byte_end, orig)
        } else {
            // Pure insertion: zero-width span after previous kept token.
            // Find the last Keep op before this region.
            let anchor = if start > 0 {
                if let DiffOp::Keep(si, _) = ops[start - 1] {
                    src_tokens[si].1 + src_tokens[si].0.len()
                } else {
                    0
                }
            } else {
                0
            };
            (anchor, anchor, String::new())
        };

        let replacement = inserted
            .iter()
            .map(|&ti| tgt_tokens[ti])
            .collect::<Vec<_>>()
            .join(" ");

        edits.push(Edit {
            span: span_start..span_end,
            original: orig_text,
            replacement,
            category: EditCategory::Style(tone.clone()),
            confidence: 0.5,
        });
    }

    edits
}

#[derive(Debug)]
enum DiffOp {
    Keep(usize, ()),
    Delete(usize),
    Insert(usize),
}

/// Returns (token_str, byte_offset) pairs for each whitespace-separated token.
fn token_spans(text: &str) -> Vec<(&str, usize)> {
    let mut result = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some(&(start, ch)) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        // Consume non-whitespace run.
        let tok_start = start;
        while chars
            .peek()
            .map(|(_, c)| !c.is_whitespace())
            .unwrap_or(false)
        {
            chars.next();
        }
        // End byte offset: next char start or end of string.
        let tok_end = chars.peek().map(|(i, _)| *i).unwrap_or(text.len());
        result.push((&text[tok_start..tok_end], tok_start));
    }
    result
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
            Ok(Self {
                _path: path.to_path_buf(),
            })
        }
    }

    impl LlmSession for LlamaCppLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
            anyhow::bail!("llama-runtime not wired yet — see gec-engine/FOLLOWUPS.md")
        }
    }
}
