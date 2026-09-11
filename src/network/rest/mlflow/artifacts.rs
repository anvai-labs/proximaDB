//! MLflow artifacts proxy (TD-MLOPS-1 slice 4; ported to the
//! [`ArtifactRepository`] seam by the TD-MLOPS-2 opening — audit #4).
//!
//! Real MLflow clients never fetch artifacts via `/api/2.0/mlflow/*`: the
//! server returns `artifact_location = mlflow-artifacts:/<exp>` and the
//! client resolves it against the tracking host's PROXY family at
//! `/api/2.0/mlflow-artifacts/artifacts/...` (PUT bytes to log, GET to
//! download / list with `?path=`, DELETE to remove). This handler is a
//! pure codec over the port — the local-fs backend is one implementation,
//! the tracked S3-backed repository is another.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::network::middleware::tenant::TenantContext;

use super::{MlflowError, MlflowResult, MlflowState};

/// Artifact uploads carry model weights — allow the legacy-path cap
/// (64 MiB) instead of axum's silent 2 MiB default; the rest of the API
/// surface keeps the platform defaults.
const ARTIFACT_BODY_LIMIT: usize = 64 * 1024 * 1024;

pub fn artifacts_routes() -> Router<MlflowState> {
    artifacts_routes_at("/api/2.0")
}

/// Same routes at an arbitrary prefix (the MLflow UI's ajax-api mount uses
/// `/mlflow-ui/ajax-api/2.0`); the absolute form stays canonical.
pub fn artifacts_routes_at(prefix: &str) -> Router<MlflowState> {
    Router::new()
        .route(
            &format!("{prefix}/mlflow-artifacts/artifacts"),
            axum::routing::get(artifact_root_list).delete(artifact_root_delete),
        )
        .route(
            &format!("{prefix}/mlflow-artifacts/artifacts/{{*path}}"),
            any(artifact_proxy),
        )
        .layer(axum::extract::DefaultBodyLimit::max(ARTIFACT_BODY_LIMIT))
}

/// Routes RELATIVE to an artifacts mount (just `/artifacts/...`, no
/// `mlflow-artifacts` prefix) — for nesting under a mount that already
/// carries the full prefix (the UI's `/ajax-api/2.0/mlflow-artifacts`).
pub fn artifacts_routes_relative() -> Router<MlflowState> {
    Router::new()
        .route(
            "/artifacts",
            axum::routing::get(artifact_root_list).delete(artifact_root_delete),
        )
        .route("/artifacts/{*path}", any(artifact_proxy))
        .layer(axum::extract::DefaultBodyLimit::max(ARTIFACT_BODY_LIMIT))
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

/// The port path for a request: tenant prefix + the repo-root-relative
/// path (structural isolation without a per-tenant repository instance).
fn port_path(tenant: &TenantContext, segments: &[String]) -> String {
    let mut path = sanitize(&tenant.tenant_id);
    for segment in segments {
        path.push('/');
        path.push_str(segment);
    }
    path
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

/// Entry names for directory listings are relative to the RUN's artifact
/// root (`<exp>/<run>/artifacts`), not to the listed subdirectory — the
/// client passes `file.path` verbatim as the next remote path.
fn list_entry_prefix(segments: &[String]) -> String {
    let anchor = segments
        .iter()
        .position(|s| s == "artifacts")
        .map(|i| i + 1)
        .unwrap_or(0);
    segments[anchor..].join("/")
}

fn entries_to_json(entries: &[proximadb_catalog::run_store::ArtifactEntry], prefix: &str) -> Value {
    json!({
        "files": entries
            .iter()
            .map(|entry| {
                let path = if prefix.is_empty() {
                    entry.path.clone()
                } else {
                    format!("{prefix}/{}", entry.path)
                };
                json!({
                    "path": path,
                    "is_dir": entry.is_dir,
                    "file_size": entry.size_bytes,
                })
            })
            .collect::<Vec<_>>()
    })
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
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> MlflowResult<Response> {
    let segments = sanitize_segments(&path)?;
    let prefix = list_entry_prefix(&segments);
    let repo_path = port_path(&tenant, &segments);
    match method {
        axum::http::Method::PUT => {
            state
                .artifacts
                .put(&repo_path, &body)
                .await
                .map_err(MlflowError::internal)?;
            Ok(StatusCode::OK.into_response())
        }
        axum::http::Method::GET => {
            match state
                .artifacts
                .get(&repo_path)
                .await
                .map_err(MlflowError::internal)?
            {
                Some(bytes) => Ok((
                    [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
                    bytes,
                )
                    .into_response()),
                None => {
                    // Not a FILE. A DIRECTORY at this path lists its
                    // entries; a MISSING path is a download miss — real
                    // MLflow 404s (RESOURCE_DOES_NOT_EXIST).
                    let entries = state
                        .artifacts
                        .list(&repo_path)
                        .await
                        .map_err(MlflowError::internal)?;
                    if entries.is_empty() {
                        return Err(MlflowError::not_found(format!(
                            "artifact '{path}' does not exist"
                        )));
                    }
                    Ok(Json(entries_to_json(&entries, &prefix)).into_response())
                }
            }
        }
        axum::http::Method::DELETE => {
            state
                .artifacts
                .delete(&repo_path)
                .await
                .map_err(MlflowError::internal)?;
            Ok(StatusCode::OK.into_response())
        }
        other => Err(MlflowError::invalid(format!(
            "unsupported artifact method {other}"
        ))),
    }
}

/// The client's directory LIST hits the bare `/artifacts` root with the
/// target encoded in `?path=<exp>/<run>/artifacts[...]`.
async fn artifact_root_list(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Query(list): Query<ListParams>,
) -> MlflowResult<Json<Value>> {
    let sub = list.path.clone().unwrap_or_default();
    // Real MLflow returns an EMPTY listing for any path that is not a
    // directory (missing or a file) — the client relies on [] for fresh
    // runs and post-delete walks; a 404 breaks both. The empty `?path=`
    // form lists the tenant root.
    let segments = if sub.is_empty() {
        Vec::new()
    } else {
        sanitize_segments(&sub)?
    };
    let prefix = list_entry_prefix(&segments);
    let repo_path = port_path(&tenant, &segments);
    let entries = state
        .artifacts
        .list(&repo_path)
        .await
        .map_err(MlflowError::internal)?;
    Ok(Json(entries_to_json(&entries, &prefix)))
}

async fn artifact_root_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Query(list): Query<ListParams>,
) -> MlflowResult<Response> {
    // The bare-root DELETE with no ?path= would wipe the TENANT's whole
    // artifact tree — require an explicit path (the client's
    // delete_artifacts(None) form targets a run's subtree and always
    // carries one).
    let sub = list.path.clone().unwrap_or_default();
    if sub.is_empty() {
        return Err(MlflowError::invalid(
            "artifact delete requires an explicit ?path=",
        ));
    }
    let segments = sanitize_segments(&sub)?;
    let repo_path = port_path(&tenant, &segments);
    state
        .artifacts
        .delete(&repo_path)
        .await
        .map_err(MlflowError::internal)?;
    Ok(StatusCode::OK.into_response())
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
