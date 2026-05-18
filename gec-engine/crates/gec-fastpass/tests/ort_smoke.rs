// Smoke test for the ort-backed encoder.
//
// Skipped unless both env vars are set:
//   ORT_DYLIB_PATH      — path to libonnxruntime.{dylib,so,dll}
//   GEC_TAGGER_ONNX     — path to the exported DeBERTa tagger ONNX file
//
// Run with:
//   ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
//   GEC_TAGGER_ONNX=/path/to/model.onnx \
//   cargo test -p gec-fastpass --features ort-runtime -- --nocapture

#![cfg(feature = "ort-runtime")]

use std::path::Path;

use gec_fastpass::ort_runtime::OrtEncoder;
use gec_fastpass::EncoderSession;

fn skip_if_unset(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("skip: {name} not set");
            None
        }
    }
}

#[test]
fn ort_encoder_returns_logits_with_expected_shape() {
    let Some(model_path) = skip_if_unset("GEC_TAGGER_ONNX") else {
        return;
    };
    if skip_if_unset("ORT_DYLIB_PATH").is_none() {
        return;
    }

    let enc = OrtEncoder::from_file(Path::new(&model_path))
        .expect("OrtEncoder::from_file");

    // [CLS] and [SEP] are deberta-v3 special tokens 1 and 2; padding is 0.
    let ids: Vec<i64> = vec![1, 100, 200, 300, 2];
    let mask: Vec<i64> = vec![1; ids.len()];

    let logits = enc.forward(&ids, &mask).expect("forward");
    assert_eq!(logits.shape()[0], ids.len(), "seq dim must match input");
    assert!(logits.shape()[1] > 1, "tag dim should exceed 1");
    eprintln!("logits shape: {:?}", logits.shape());
}
