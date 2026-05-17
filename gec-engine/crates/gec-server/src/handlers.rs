use axum::{
    extract::Json as JsonExtract,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use gec_core::{CorrectionRequest, CorrectionResponse, ResponseStats, RewriteRequest};

pub fn router() -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/correct", post(correct))
        .route("/rewrite", post(rewrite))
}

async fn healthz() -> &'static str {
    "ok"
}

async fn correct(
    JsonExtract(req): JsonExtract<CorrectionRequest>,
) -> Result<Json<CorrectionResponse>, (StatusCode, String)> {
    Ok(Json(CorrectionResponse {
        edits: vec![],
        corrected_text: req.text,
        degraded: true,
        partial: false,
        request_id: request_id(),
        stats: ResponseStats::default(),
    }))
}

async fn rewrite(
    JsonExtract(req): JsonExtract<RewriteRequest>,
) -> Result<Json<CorrectionResponse>, (StatusCode, String)> {
    Ok(Json(CorrectionResponse {
        edits: vec![],
        corrected_text: req.text,
        degraded: true,
        partial: false,
        request_id: request_id(),
        stats: ResponseStats::default(),
    }))
}

fn request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}-{:x}", dur.as_secs(), dur.subsec_nanos())
}
