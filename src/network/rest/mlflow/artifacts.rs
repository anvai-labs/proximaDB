//! MLflow artifacts proxy (TD-MLOPS-1 slice 4).
//!
//! Real MLflow clients never fetch artifacts via `/api/2.0/mlflow/*`: the
//! server returns `artifact_location = mlflow-artifacts:/<exp>` and the
//! client resolves it against the tracking host's PROXY family at
//! `/api/2.0/mlflow-artifacts/artifacts/...` (PUT bytes to log, GET to
//! download / list with `?path=`, DELETE to remove).
//!
//! Slice-4 storage backend: local files under
//! `<data_dir>/mlflow_artifacts/<tenant>/...` — the honest default for a
//! single-node deployment; the S3-backed repository (object storage via the
//! platform's storage locations) is the tracked follow-up. Paths are
//! segment-sanitized (no `..`, no absolute escapes) and tenant-scoped.

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::network::middleware::tenant::TenantContext;

use super::{MlflowError, MlflowResult, MlflowState};

pub fn artifacts_routes() -> Router<MlflowState> {
    Router::new()
        .route(
            "/api/2.0/mlflow-artifacts/artifacts",
            axum::routing::get(artifact_root_list).delete(artifact_root_delete),
        )
        .route(
            "/api/2.0/mlflow-artifacts/artifacts/{*path}",
            any(artifact_proxy),
        )
}

/// The client's directory LIST hits the bare `/artifacts` root with the
/// target encoded in `?path=<exp>/<run>/artifacts[...]`.
async fn artifact_root_list(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Query(list): Query<ListParams>,
) -> MlflowResult<Json<Value>> {
    let sub = list.path.clone().unwrap_or_default();
    let segments = sanitize_segments(&sub)?;
    let root = tenant_artifact_root(&state, &tenant);
    let target = segments.iter().fold(root, |acc, s| acc.join(s));
    if !target.exists() {
        return Err(MlflowError::not_found(format!(
            "artifact path '{sub}' does not exist"
        )));
    }
    if !target.is_dir() {
        // Listing a FILE path yields no entries (the MLflow server
        // convention) — the client then treats the path itself as the file.
        return Ok(Json(json!({ "files": [] })));
    }
    let mut entries = tokio::fs::read_dir(&target)
        .await
        .map_err(|e| MlflowError::internal(format!("read artifact dir: {e}")))?;
    let mut files = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| MlflowError::internal(format!("iterate artifacts: {e}")))?
    {
        let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
        let size = if is_dir {
            0
        } else {
            entry.metadata().await.map(|m| m.len()).unwrap_or(0)
        };
        files.push(json!({
            "path": entry.file_name().to_string_lossy(),
            "is_dir": is_dir,
            "file_size": size,
        }));
    }
    Ok(Json(json!({ "files": files })))
}

async fn artifact_root_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Query(list): Query<ListParams>,
) -> MlflowResult<Response> {
    let sub = list.path.clone().unwrap_or_default();
    if sub.is_empty() {
        return Err(MlflowError::invalid("artifact root delete requires ?path="));
    }
    let segments = sanitize_segments(&sub)?;
    let root = tenant_artifact_root(&state, &tenant);
    let target = segments.iter().fold(root, |acc, s| acc.join(s));
    if target.is_dir() {
        tokio::fs::remove_dir_all(&target)
            .await
            .map_err(|e| MlflowError::internal(format!("delete artifacts: {e}")))?;
    } else if target.exists() {
        tokio::fs::remove_file(&target)
            .await
            .map_err(|e| MlflowError::internal(format!("delete artifact: {e}")))?;
    }
    Ok(StatusCode::OK.into_response())
}

fn sanitize_segments(path: &str) -> MlflowResult<Vec<String>> {
    let mut out = Vec::new();
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." || segment.contains('\\') || segment.contains('\0') {
            return Err(MlflowError::invalid(format!(
                "invalid artifact path segment '{segment}'"
            )));
        }
        out.push(segment.to_string());
    }
    if out.is_empty() {
        return Err(MlflowError::invalid("artifact path must not be empty"));
    }
    Ok(out)
}

fn tenant_artifact_root(state: &MlflowState, tenant: &TenantContext) -> std::path::PathBuf {
    state
        .data_dir
        .join("mlflow_artifacts")
        .join(sanitize(&tenant.tenant_id))
}

fn sanitize(component: &str) -> String {
    component
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Default, Deserialize)]
struct ListParams {
    #[serde(default)]
    path: Option<String>,
}

async fn artifact_proxy(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(path): Path<String>,
    Query(list): Query<ListParams>,
    method: axum::http::Method,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> MlflowResult<Response> {
    let segments = sanitize_segments(&path)?;
    let root = tenant_artifact_root(&state, &tenant);
    // Segments are traversal-checked verbatim (no .., backslash, NUL) —
    // file names must round-trip byte-identically.
    let target = segments.iter().fold(root.clone(), |acc, s| acc.join(s));

    match method {
        axum::http::Method::PUT => {
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| MlflowError::internal(format!("create artifact dir: {e}")))?;
            }
            tokio::fs::write(&target, &body)
                .await
                .map_err(|e| MlflowError::internal(format!("write artifact: {e}")))?;
            Ok(StatusCode::OK.into_response())
        }
        axum::http::Method::GET => {
            if !target.exists() {
                return Err(MlflowError::not_found(format!(
                    "artifact '{path}' does not exist"
                )));
            }
            if target.is_dir() {
                // Directory LIST: MLflow shape {files: [{path, is_dir,
                // file_size}]} relative to the requested directory.
                let list_root = match &list.path {
                    Some(sub) => {
                        let mut p = target.clone();
                        for segment in sanitize_segments(sub)? {
                            p = p.join(segment);
                        }
                        p
                    }
                    None => target,
                };
                let mut entries = tokio::fs::read_dir(&list_root)
                    .await
                    .map_err(|e| MlflowError::internal(format!("read artifact dir: {e}")))?;
                let mut files = Vec::new();
                while let Some(entry) = entries
                    .next_entry()
                    .await
                    .map_err(|e| MlflowError::internal(format!("iterate artifacts: {e}")))?
                {
                    let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
                    let size = if is_dir {
                        0
                    } else {
                        entry.metadata().await.map(|m| m.len()).unwrap_or(0)
                    };
                    files.push(json!({
                        "path": entry.file_name().to_string_lossy(),
                        "is_dir": is_dir,
                        "file_size": size,
                    }));
                }
                Ok(Json(json!({"files": files})).into_response())
            } else {
                let bytes = tokio::fs::read(&target)
                    .await
                    .map_err(|e| MlflowError::internal(format!("read artifact: {e}")))?;
                let mut response = Response::new(axum::body::Body::from(bytes));
                if let Some(mime) = headers.get("accept").and_then(|v| v.to_str().ok()) {
                    if !mime.is_empty() && mime != "*/*" {
                        response
                            .headers_mut()
                            .insert(axum::http::header::CONTENT_TYPE, mime.parse().unwrap());
                    }
                }
                Ok(response)
            }
        }
        axum::http::Method::DELETE => {
            if target.is_dir() {
                tokio::fs::remove_dir_all(&target)
                    .await
                    .map_err(|e| MlflowError::internal(format!("delete artifacts: {e}")))?;
            } else if target.exists() {
                tokio::fs::remove_file(&target)
                    .await
                    .map_err(|e| MlflowError::internal(format!("delete artifact: {e}")))?;
            }
            Ok(StatusCode::OK.into_response())
        }
        other => Err(MlflowError::invalid(format!(
            "unsupported artifact method {other}"
        ))),
    }
}

/// The artifact URI clients resolve against the tracking host:
/// `mlflow-artifacts:/<experiment_id>/<run_id>/artifacts`.
pub(super) fn run_artifact_uri(experiment_id: u64, run_id: &str) -> String {
    format!("mlflow-artifacts:/{experiment_id}/{run_id}/artifacts")
}

/// Default experiment artifact location.
pub(super) fn experiment_artifact_location(experiment_id: u64) -> String {
    format!("mlflow-artifacts:/{experiment_id}")
}

#[allow(dead_code)]
fn _assert_send(_f: impl Fn() -> MlflowResult<Response>) {}
