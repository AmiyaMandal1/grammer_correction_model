use axum::body::Body;
use axum::http::{Request, StatusCode};
use gec_server::router;
use tower::ServiceExt;

#[tokio::test]
async fn healthz_returns_ok() {
    let app = router();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn correct_returns_empty_edits_for_empty_text() {
    let app = router();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/correct")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(body_str.contains("\"edits\""));
    assert!(body_str.contains("\"corrected_text\""));
}
