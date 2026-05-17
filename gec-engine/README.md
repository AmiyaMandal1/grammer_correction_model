# gec-engine

Rust serving runtime for the GEC tagger + style LLM. Single binary exposing CLI and HTTP.

## Build

```
cd gec-engine
cargo build --release
```

## Run

```
./target/release/gec-server \
    --encoder-onnx /path/to/deberta-gec.onnx \
    --encoder-tags /path/to/tags.json \
    --encoder-tokenizer /path/to/tokenizer.json \
    --llm-gguf /path/to/qwen-style-q4.gguf \
    --bind 127.0.0.1:8080
```

## Endpoints

- `POST /correct` — grammar/spelling/punctuation. Body: `{"text": "..."}`.
- `POST /rewrite` — tone/style rewrite. Body: `{"text": "...", "tone": "formal|informal|concise|simplify|detoxify"}`.
- `GET /healthz` — deep health check.
