use gec_core::Edit;

/// Merge edits from multiple passes, dropping overlapping lower-confidence
/// edits. Stable output sorted by span start.
pub fn merge_edits(mut edits: Vec<Edit>) -> Vec<Edit> {
    edits.sort_by(|a, b| {
        a.span.start.cmp(&b.span.start).then(
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    let mut kept: Vec<Edit> = Vec::new();
    for e in edits {
        if let Some(last) = kept.last() {
            if e.span.start < last.span.end {
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
