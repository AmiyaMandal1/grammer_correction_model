use gec_tokenizer::Tokenizer;

#[test]
fn loads_from_json_string() {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#;
    let tok = Tokenizer::from_json(json).expect("loads");
    let enc = tok.encode("hello world", false).expect("encodes");
    let ids = enc.get_ids();
    assert_eq!(ids, &[1, 2]);
}
