use gec_core::{Edit, EditCategory};
use gec_fastpass::{decode_tag_string, EncoderSession, FastPass, TagVocab};
use ndarray::Array2;

struct StubEncoder {
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
    let logits =
        Array2::from_shape_vec((3, 3), vec![0.9, 0.1, 0.0, 0.0, 0.9, 0.1, 0.9, 0.1, 0.0]).unwrap();
    let stub = StubEncoder {
        fixed_logits: logits,
    };
    let vocab = TagVocab {
        tags: vec!["$KEEP".into(), "$REPLACE_goes".into(), "$DELETE".into()],
    };
    let edits: Vec<Edit> = FastPass::with_encoder(stub, vocab)
        .edits_for_tokens(&["he", "go", "home"])
        .unwrap();
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].replacement, "goes");
    assert_eq!(edits[0].category, EditCategory::Grammar);
    assert_eq!(edits[0].original, "go");
}
