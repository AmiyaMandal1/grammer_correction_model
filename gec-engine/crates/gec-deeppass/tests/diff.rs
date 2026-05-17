use gec_core::{EditCategory, Tone};
use gec_deeppass::{diff_to_edits, DeepPass, LlmSession};

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
    assert!(matches!(
        edits[0].category,
        EditCategory::Style(Tone::Formal)
    ));
}

#[test]
fn diff_to_edits_no_change_returns_empty() {
    assert!(diff_to_edits("same text", "same text", Tone::Formal).is_empty());
}

#[test]
fn deeppass_rewrites_via_llm() {
    let llm = StubLlm {
        out: "He goes home.".into(),
    };
    let dp = DeepPass::with_llm(llm);
    let (text, edits) = dp.rewrite("he go home.", Tone::Formal).unwrap();
    assert_eq!(text, "He goes home.");
    assert!(!edits.is_empty());
}
