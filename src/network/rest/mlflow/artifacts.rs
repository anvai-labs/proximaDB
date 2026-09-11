//! MLflow artifacts proxy (TD-MLOPS-1 slice 4).
//!
//! Real MLflow clients never fetch artifacts via `/api/2.0/mlflow/*`: the
//! server returns `artifact_location = mlflow-artifacts:/<exp>` and the
//! client resolves it against the tracking host's PROXY family at
//! `/api/2.0/mlflow-artifacts/artifacts/...` (PUT bytes to log, GET to
//! download / list with `?path=`, DELETE to remove).
//!
//! Slice-4 storage backend: local files under an injectively encoded tenant
//! root below `<data_dir>/mlflow_artifacts_v2/` — the honest default for a
//! single-node deployment; the S3-backed repository (object storage via the
//! platform's storage locations) is the tracked follow-up. Paths are
//! segment-sanitized (no `..`, no absolute escapes) and tenant-scoped. The
//! lossy legacy `mlflow_artifacts/<sanitized-tenant>` layout is intentionally
//! not read because distinct tenants may already share one legacy directory.

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
/// Keeps the encoded tenant portion portable while still accommodating UUIDs
/// and ordinary account/workspace identifiers. Artifact routes revalidate at
/// this storage boundary instead of relying solely on middleware construction.
const MAX_ARTIFACT_TENANT_ID_BYTES: usize = 64;

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
        .layer(axum::extract::DefaultBodyLimit::max(ARTIFACT_BODY_LIMIT))
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
    let prefix = list_entry_prefix(&segments);
    let root = tenant_artifact_root(&state, &tenant)?;
    let target = segments.iter().fold(root, |acc, s| acc.join(s));
    let Some(kind) = artifact_kind(&target, "inspect artifact path").await? else {
        return Err(MlflowError::not_found(format!(
            "artifact path '{sub}' does not exist"
        )));
    };
    if kind != ArtifactKind::Directory {
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
        let name = entry.file_name().to_string_lossy().to_string();
        let path = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        files.push(json!({
            "path": path,
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
    let root = tenant_artifact_root(&state, &tenant)?;
    let target = segments.iter().fold(root, |acc, s| acc.join(s));
    if let Some(kind) = artifact_kind(&target, "inspect artifact delete target").await? {
        if kind == ArtifactKind::Directory {
            tokio::fs::remove_dir_all(&target)
                .await
                .map_err(|e| MlflowError::internal(format!("delete artifacts: {e}")))?;
        } else {
            tokio::fs::remove_file(&target)
                .await
                .map_err(|e| MlflowError::internal(format!("delete artifact: {e}")))?;
        }
    }
    Ok(StatusCode::OK.into_response())
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

fn tenant_artifact_root(
    state: &MlflowState,
    tenant: &TenantContext,
) -> MlflowResult<std::path::PathBuf> {
    Ok(state
        .data_dir
        .join(tenant_artifact_relative_root(&tenant.tenant_id)?))
}

/// Encode every tenant id byte-for-byte under a clean versioned root.
///
/// The byte length separates tenant roots before the encoded leaf, so a
/// tenant whose encoded id is a prefix of another tenant cannot own an ancestor
/// of that tenant's artifact tree. The 64-byte boundary keeps the hex leaf below
/// common filesystem component limits. There is deliberately no legacy read
/// fallback: the lossy v1 sanitizer made ownership of an existing colliding
/// directory unknowable, so exposing it to any claimant would preserve the bug.
fn tenant_artifact_relative_root(tenant_id: &str) -> MlflowResult<std::path::PathBuf> {
    proximadb_tenant::validate_request_tenant(tenant_id)
        .map_err(|error| MlflowError::invalid(format!("invalid artifact tenant: {error}")))?;
    if tenant_id.len() > MAX_ARTIFACT_TENANT_ID_BYTES {
        return Err(MlflowError::invalid(format!(
            "artifact tenant id exceeds {MAX_ARTIFACT_TENANT_ID_BYTES} UTF-8 bytes"
        )));
    }

    let encoded = hex::encode(tenant_id.as_bytes());
    let mut root = std::path::PathBuf::from("mlflow_artifacts_v2");
    root.push(tenant_id.len().to_string());
    root.push(encoded);
    Ok(root)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactKind {
    File,
    Directory,
}

async fn artifact_kind(
    target: &std::path::Path,
    operation: &str,
) -> MlflowResult<Option<ArtifactKind>> {
    match tokio::fs::metadata(target).await {
        Ok(metadata) if metadata.is_dir() => Ok(Some(ArtifactKind::Directory)),
        Ok(_) => Ok(Some(ArtifactKind::File)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(MlflowError::internal(format!("{operation}: {error}"))),
    }
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
    let root = tenant_artifact_root(&state, &tenant)?;
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
            let Some(kind) = artifact_kind(&target, "inspect artifact").await? else {
                return Err(MlflowError::not_found(format!(
                    "artifact '{path}' does not exist"
                )));
            };
            if kind == ArtifactKind::Directory {
                // Directory LIST. Entry names are repo-root-relative
                // (relative to <exp>/<run>/artifacts) — the client feeds
                // file.path verbatim into the next remote GET.
                let prefix = list_entry_prefix(&segments);
                let list_root = target.clone();
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
                    let name = entry.file_name().to_string_lossy().to_string();
                    let entry_path = if prefix.is_empty() {
                        name
                    } else {
                        format!("{prefix}/{name}")
                    };
                    files.push(json!({
                        "path": entry_path,
                        "is_dir": is_dir,
                        "file_size": size,
                    }));
                }
                Ok(Json(json!({"files": files})).into_response())
            } else {
                let bytes = tokio::fs::read(&target)
                    .await
                    .map_err(|e| MlflowError::internal(format!("read artifact: {e}")))?;
                // Octet-stream: the proxy carries bytes; echoing the
                // request's Accept header as Content-Type is semantically
                // wrong and could yield an invalid MIME.
                Ok((
                    [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
                    bytes,
                )
                    .into_response())
            }
        }
        axum::http::Method::DELETE => {
            if let Some(kind) = artifact_kind(&target, "inspect artifact delete target").await? {
                if kind == ArtifactKind::Directory {
                    tokio::fs::remove_dir_all(&target)
                        .await
                        .map_err(|e| MlflowError::internal(format!("delete artifacts: {e}")))?;
                } else {
                    tokio::fs::remove_file(&target)
                        .await
                        .map_err(|e| MlflowError::internal(format!("delete artifact: {e}")))?;
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_artifact_components_are_injective_on_common_filesystems() {
        let punctuation_variants = ["acme:prod", "acme?prod", "acme_prod"];
        let encoded: std::collections::HashSet<_> = punctuation_variants
            .iter()
            .map(|tenant| {
                tenant_artifact_relative_root(tenant)
                    .ok()
                    .expect("valid test tenant must encode")
            })
            .collect();
        assert_eq!(encoded.len(), punctuation_variants.len());

        assert_ne!(
            tenant_artifact_relative_root("Tenant")
                .ok()
                .expect("valid test tenant must encode")
                .to_string_lossy()
                .to_ascii_lowercase(),
            tenant_artifact_relative_root("tenant")
                .ok()
                .expect("valid test tenant must encode")
                .to_string_lossy()
                .to_ascii_lowercase(),
            "tenant artifact roots must remain distinct on case-insensitive filesystems"
        );

        assert!(tenant_artifact_relative_root("").is_err());
        assert!(tenant_artifact_relative_root("a".repeat(65).as_str()).is_err());
    }

    #[tokio::test]
    async fn encoded_tenant_roots_never_reuse_legacy_collision_bytes() {
        let temp = tempfile::tempdir().expect("create artifact isolation test directory");
        let legacy = temp.path().join("mlflow_artifacts/acme_prod/secret.bin");
        tokio::fs::create_dir_all(legacy.parent().expect("legacy path has parent"))
            .await
            .expect("create legacy collision directory");
        tokio::fs::write(&legacy, b"co-mingled legacy bytes")
            .await
            .expect("seed legacy collision bytes");

        for tenant in ["acme:prod", "acme?prod", "acme_prod", "Tenant", "tenant"] {
            let root = temp.path().join(
                tenant_artifact_relative_root(tenant)
                    .ok()
                    .expect("valid test tenant must encode"),
            );
            let kind = artifact_kind(&root, "inspect isolated tenant root").await;
            assert!(kind.is_ok(), "isolated tenant root inspection must succeed");
            assert_eq!(
                kind.ok().flatten(),
                None,
                "tenant {tenant} must not inherit bytes from the ambiguous legacy layout"
            );
        }
    }

    #[test]
    fn async_artifact_handlers_do_not_use_synchronous_path_probes() {
        let source = include_str!("artifacts.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or_default();
        for forbidden in [
            [".", "exists()"].concat(),
            ["target", ".", "is_dir()"].concat(),
            [".", "is_file()"].concat(),
            ["std", "::", "fs", "::"].concat(),
        ] {
            assert!(
                !production.contains(&forbidden),
                "async artifact handlers contain synchronous path probe {forbidden}"
            );
        }
    }
}

#[allow(dead_code)]
fn _assert_send(_f: impl Fn() -> MlflowResult<Response>) {}
