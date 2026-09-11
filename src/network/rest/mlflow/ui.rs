//! MLflow UI serving (TD-MLOPS-3): the vendored stock MLflow web bundle
//! (Apache-2.0, third_party/mlflow-ui/<version>/build/) served same-origin
//! behind the `PROXIMADB_MLFLOW_COMPAT_ENABLE` gate. Compiled in only with
//! the non-default `mlflow-ui` cargo feature (binary-size guardrail).
//!
//! Architecture (the mount-is-the-contract rule, TD-MLOPS-3): statics AND
//! the ajax-api aliases mount on the same base_router, BEFORE the
//! tenant/auth middleware layers — a future OIDC middleware gates UI and
//! API together in one place. NEVER implement MLflow's `users/*`/`roles/*`
//! endpoints (shadow identity); NEVER merge this bundle into the admin
//! React app (forks upstream updates).

use axum::Router;
#[cfg(feature = "mlflow-ui")]
use axum::body::Body;
#[cfg(feature = "mlflow-ui")]
use axum::http::{HeaderMap, StatusCode, header};
#[cfg(feature = "mlflow-ui")]
use axum::response::{IntoResponse, Response};
#[cfg(feature = "mlflow-ui")]
use axum::routing::get;
#[cfg(feature = "mlflow-ui")]
use include_dir::{Dir, include_dir};

#[cfg(feature = "mlflow-ui")]
static UI_BUILD: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/third_party/mlflow-ui/3.16.0/build");

/// Content types the bundle actually serves (mime_guess covers the rest).
#[cfg(feature = "mlflow-ui")]
const STATIC_CACHE_MAX_AGE: u64 = 2_419_200; // 28 days — Flask parity for hashed assets

/// The UI router: `/` (index.html) + `/static-files/{*path}` (build tree
/// with the prefix STRIPPED — MLflow Flask's send_from_directory semantics).
#[cfg(feature = "mlflow-ui")]
pub fn ui_routes() -> Router<MlflowState> {
    Router::new()
        .route("/", get(serve_index))
        .route("/static-files/{*path}", get(serve_static_file))
        .fallback(serve_index)
}

/// Public handler for the trailing-slash mount (server.rs); no-feature
/// stub keeps the default build compiling.
#[cfg(not(feature = "mlflow-ui"))]
pub async fn serve_index_route() -> impl axum::response::IntoResponse {
    axum::http::StatusCode::NOT_FOUND
}

#[cfg(feature = "mlflow-ui")]
pub async fn serve_index_route() -> Response {
    serve_index().await
}

#[cfg(feature = "mlflow-ui")]
async fn serve_index() -> Response {
    serve_file_from("index.html", false)
}

#[cfg(feature = "mlflow-ui")]
async fn serve_static_file(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    serve_file_from(&path, true)
}

#[cfg(feature = "mlflow-ui")]
fn serve_file_from(path: &str, cache: bool) -> Response {
    let Some(file) = UI_BUILD.get_file(path) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let mut headers = HeaderMap::new();
    if let Ok(v) = header::HeaderValue::from_str(mime.as_ref()) {
        headers.insert(header::CONTENT_TYPE, v);
    }
    if cache {
        if let Ok(v) =
            header::HeaderValue::from_str(&format!("public, max-age={STATIC_CACHE_MAX_AGE}"))
        {
            headers.insert(header::CACHE_CONTROL, v);
        }
    }
    (headers, Body::from(file.contents())).into_response()
}

/// The full `/mlflow-ui` mount: statics + the ajax-api nests (same-origin
/// relative URLs resolve against `/mlflow-ui/`). Requires the shared
/// `MlflowState` for the nested API routers.
pub fn ui_mount_routes() -> Router<MlflowState> {
    #[cfg(feature = "mlflow-ui")]
    {
        Router::new()
            .merge(ui_routes())
            .nest("/ajax-api/2.0/mlflow", super::mlflow_routes())
            .nest(
                "/ajax-api/2.0/mlflow-artifacts",
                super::artifacts_router_relative(),
            )
    }
    #[cfg(not(feature = "mlflow-ui"))]
    {
        Router::new()
    }
}

/// Log line when the gate is on but the UI wasn't compiled in.
pub fn log_feature_status() {
    #[cfg(not(feature = "mlflow-ui"))]
    tracing::info!(
        "ℹ️ MLflow UI not compiled in (rebuild with --features mlflow-ui to serve /mlflow-ui)"
    );
    #[cfg(feature = "mlflow-ui")]
    tracing::info!("✅ MLflow UI served at /mlflow-ui");
}

use super::MlflowState;
