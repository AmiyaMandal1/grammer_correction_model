use gec_core::{CorrectionResponse, Edit, EditCategory, ResponseStats, Tone};

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
