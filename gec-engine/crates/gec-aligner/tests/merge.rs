use gec_aligner::{apply_edits, merge_edits};
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
    let edits = vec![ed(3..5, "go", "is going", EditCategory::Grammar, 0.8)];
    let out = apply_edits(text, &edits);
    assert_eq!(out, "he is going");
}
