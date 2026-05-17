use anyhow::Result;
use clap::Parser;
use gec_server::router;

#[derive(Parser, Debug)]
#[command(name = "gec-server")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long)]
    encoder_onnx: Option<String>,
    #[arg(long)]
    encoder_tags: Option<String>,
    #[arg(long)]
    encoder_tokenizer: Option<String>,
    #[arg(long)]
    llm_gguf: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let app = router();
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    tracing::info!("listening on {}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
