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
//! Local artifact operations fail closed on Windows until every filesystem
//! operation in the capability stack is handle-relative on that platform.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::{Extension, Json, Router};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{Read, Write};

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

#[derive(Clone)]
struct ArtifactStore {
    data_dir: std::path::PathBuf,
    tenant_root: std::path::PathBuf,
}

struct ArtifactListEntry {
    name: String,
    is_dir: bool,
    size: u64,
}

enum StoredArtifact {
    File(Vec<u8>),
    Directory(Vec<ArtifactListEntry>),
}

fn windows_artifact_operations_supported() -> bool {
    false
}

fn ensure_local_artifact_operations_supported() -> std::io::Result<()> {
    if cfg!(windows) && !windows_artifact_operations_supported() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "local artifact operations are unavailable on Windows without fully handle-relative filesystem support",
        ));
    }
    Ok(())
}

impl ArtifactStore {
    fn new(state: &MlflowState, tenant: &TenantContext) -> MlflowResult<Self> {
        Ok(Self {
            data_dir: state.data_dir.clone(),
            tenant_root: tenant_artifact_relative_root(&tenant.tenant_id)?,
        })
    }

    async fn run<T, F>(&self, operation: &'static str, task: F) -> MlflowResult<T>
    where
        T: Send + 'static,
        F: FnOnce(Self) -> std::io::Result<T> + Send + 'static,
    {
        let store = self.clone();
        tokio::task::spawn_blocking(move || task(store))
            .await
            .map_err(|error| MlflowError::internal(format!("{operation} task failed: {error}")))?
            .map_err(|error| map_artifact_io_error(operation, error))
    }

    async fn read(&self, segments: Vec<String>) -> MlflowResult<Option<StoredArtifact>> {
        self.run("read artifact", move |store| store.read_sync(&segments))
            .await
    }

    async fn write(&self, segments: Vec<String>, bytes: Vec<u8>) -> MlflowResult<()> {
        self.run("write artifact", move |store| {
            store.write_sync(&segments, &bytes)
        })
        .await
    }

    async fn list(&self, segments: Vec<String>) -> MlflowResult<Vec<ArtifactListEntry>> {
        self.run("list artifacts", move |store| store.list_sync(&segments))
            .await
    }

    async fn delete(&self, segments: Vec<String>) -> MlflowResult<()> {
        self.run("delete artifact", move |store| store.delete_sync(&segments))
            .await
    }

    fn open_tenant_dir(&self, create: bool) -> std::io::Result<Option<Dir>> {
        if create {
            Dir::create_ambient_dir_all(&self.data_dir, ambient_authority())?;
        }
        let mut current = match Dir::open_ambient_dir(&self.data_dir, ambient_authority()) {
            Ok(dir) => dir,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        for component in self.tenant_root.components() {
            let Some(next) = open_child_dir(&current, component.as_os_str(), create)? else {
                return Ok(None);
            };
            current = next;
        }
        Ok(Some(current))
    }

    fn open_parent(
        &self,
        segments: &[String],
        create: bool,
    ) -> std::io::Result<Option<(Dir, String)>> {
        let Some((leaf, parents)) = segments.split_last() else {
            return Err(unsafe_artifact_path("artifact path must not be empty"));
        };
        let Some(mut current) = self.open_tenant_dir(create)? else {
            return Ok(None);
        };
        for component in parents {
            let Some(next) = open_child_dir(&current, component, create)? else {
                return Ok(None);
            };
            current = next;
        }
        Ok(Some((current, leaf.clone())))
    }

    fn read_sync(&self, segments: &[String]) -> std::io::Result<Option<StoredArtifact>> {
        ensure_local_artifact_operations_supported()?;
        let Some((parent, leaf)) = self.open_parent(segments, false)? else {
            return Ok(None);
        };
        let metadata = match parent.symlink_metadata(&leaf) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if metadata.file_type().is_symlink() {
            return Err(unsafe_artifact_path(
                "artifact path contains a symbolic link",
            ));
        }
        if metadata.is_dir() {
            let directory = parent
                .open_dir_nofollow(&leaf)
                .map_err(|error| nofollow_error(&leaf, error))?;
            Ok(Some(StoredArtifact::Directory(read_directory_entries(
                &directory,
            )?)))
        } else {
            let mut options = OpenOptions::new();
            options.read(true).follow(FollowSymlinks::No);
            let mut file = parent
                .open_with(&leaf, &options)
                .map_err(|error| nofollow_error(&leaf, error))?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            Ok(Some(StoredArtifact::File(bytes)))
        }
    }

    fn list_sync(&self, segments: &[String]) -> std::io::Result<Vec<ArtifactListEntry>> {
        ensure_local_artifact_operations_supported()?;
        let directory = if let Some((leaf, parents)) = segments.split_last() {
            let Some(mut current) = self.open_tenant_dir(false)? else {
                return Ok(Vec::new());
            };
            for component in parents {
                let Some(next) = open_child_dir(&current, component, false)? else {
                    return Ok(Vec::new());
                };
                current = next;
            }

            let metadata = match current.symlink_metadata(leaf) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(Vec::new());
                }
                Err(error) => return Err(error),
            };
            if metadata.file_type().is_symlink() {
                return Err(unsafe_artifact_path(
                    "artifact path contains a symbolic link",
                ));
            }
            if !metadata.is_dir() {
                return Ok(Vec::new());
            }
            match current.open_dir_nofollow(leaf) {
                Ok(directory) => directory,
                // LIST is intentionally empty for a missing path. Preserve
                // that contract if the directory disappears after metadata
                // but before the capability-relative open.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(Vec::new());
                }
                Err(error) => return Err(nofollow_error(leaf, error)),
            }
        } else {
            let Some(directory) = self.open_tenant_dir(false)? else {
                return Ok(Vec::new());
            };
            directory
        };

        read_directory_entries(&directory)
    }

    fn write_sync(&self, segments: &[String], bytes: &[u8]) -> std::io::Result<()> {
        ensure_local_artifact_operations_supported()?;
        let Some((parent, leaf)) = self.open_parent(segments, true)? else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "tenant artifact root disappeared during creation",
            ));
        };
        match parent.symlink_metadata(&leaf) {
            Ok(metadata) if metadata.file_type().is_symlink() || metadata.is_dir() => {
                return Err(unsafe_artifact_path(
                    "artifact destination is not a regular file",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create(true)
            .truncate(true)
            .follow(FollowSymlinks::No);
        let mut file = parent
            .open_with(&leaf, &options)
            .map_err(|error| nofollow_error(&leaf, error))?;
        file.write_all(bytes)
    }

    fn delete_sync(&self, segments: &[String]) -> std::io::Result<()> {
        ensure_local_artifact_operations_supported()?;
        let Some((parent, leaf)) = self.open_parent(segments, false)? else {
            return Ok(());
        };
        let metadata = match parent.symlink_metadata(&leaf) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        if metadata.file_type().is_symlink() {
            return Err(unsafe_artifact_path(
                "artifact path contains a symbolic link",
            ));
        }
        let result = if metadata.is_dir() {
            parent.remove_dir_all(&leaf)
        } else {
            parent.remove_file(&leaf)
        };
        match result {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

fn open_child_dir(
    parent: &Dir,
    component: impl AsRef<std::path::Path>,
    create: bool,
) -> std::io::Result<Option<Dir>> {
    let component = component.as_ref();
    match parent.symlink_metadata(component) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(unsafe_artifact_path(
                "artifact path component is not a real directory",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
            match parent.create_dir(component) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    }
    match parent.open_dir_nofollow(component) {
        Ok(directory) => Ok(Some(directory)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(nofollow_error(component, error)),
    }
}

fn read_directory_entries(directory: &Dir) -> std::io::Result<Vec<ArtifactListEntry>> {
    let mut entries = Vec::new();
    for entry in directory.entries()? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(unsafe_artifact_path(
                "artifact directory contains a symbolic link",
            ));
        }
        let is_dir = file_type.is_dir();
        let size = if is_dir { 0 } else { entry.metadata()?.len() };
        entries.push(ArtifactListEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            is_dir,
            size,
        });
    }
    Ok(entries)
}

fn unsafe_artifact_path(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}

fn nofollow_error(path: impl AsRef<std::path::Path>, error: std::io::Error) -> std::io::Error {
    if matches!(
        error.kind(),
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::NotADirectory
    ) || is_symlink_loop(&error)
        || is_windows_stopped_on_symlink(&error)
    {
        unsafe_artifact_path(format!(
            "artifact path '{}' could not be opened without following links: {error}",
            path.as_ref().display()
        ))
    } else {
        error
    }
}

const WINDOWS_ERROR_STOPPED_ON_SYMLINK: i32 = 681;

fn is_windows_stopped_on_symlink_code(raw_os_error: Option<i32>) -> bool {
    raw_os_error == Some(WINDOWS_ERROR_STOPPED_ON_SYMLINK)
}

fn is_windows_stopped_on_symlink(error: &std::io::Error) -> bool {
    cfg!(windows) && is_windows_stopped_on_symlink_code(error.raw_os_error())
}

#[cfg(unix)]
fn is_symlink_loop(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(libc::ELOOP)
}

#[cfg(not(unix))]
fn is_symlink_loop(_error: &std::io::Error) -> bool {
    false
}

fn map_artifact_io_error(operation: &str, error: std::io::Error) -> MlflowError {
    if error.kind() == std::io::ErrorKind::InvalidInput {
        MlflowError::invalid(error.to_string())
    } else if error.kind() == std::io::ErrorKind::NotFound {
        MlflowError::not_found(format!("{operation}: artifact path does not exist"))
    } else {
        MlflowError::internal(format!("{operation}: {error}"))
    }
}

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

/// The client's directory LIST hits the bare `/artifacts` root with the
/// target encoded in `?path=<exp>/<run>/artifacts[...]`.
async fn artifact_root_list(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Query(list): Query<ListParams>,
) -> MlflowResult<Json<Value>> {
    let sub = list.path.clone().unwrap_or_default();
    let segments = if sub.is_empty() {
        Vec::new()
    } else {
        sanitize_segments(&sub)?
    };
    let prefix = list_entry_prefix(&segments);
    let store = ArtifactStore::new(&state, &tenant)?;
    let entries = store.list(segments).await?;
    Ok(Json(json!({
        "files": artifact_entries_json(entries, &prefix)
    })))
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
    ArtifactStore::new(&state, &tenant)?
        .delete(segments)
        .await?;
    Ok(StatusCode::OK.into_response())
}

fn artifact_entries_json(entries: Vec<ArtifactListEntry>, prefix: &str) -> Vec<Value> {
    entries
        .into_iter()
        .map(|entry| {
            let path = if prefix.is_empty() {
                entry.name
            } else {
                format!("{prefix}/{}", entry.name)
            };
            json!({
                "path": path,
                "is_dir": entry.is_dir,
                "file_size": entry.size,
            })
        })
        .collect()
}

/// Capability-safe directory listing for sibling MLflow handlers. Keeping
/// this behind the artifact module prevents callers from reconstructing a
/// host path or bypassing the tenant capability and no-follow checks.
pub(super) async fn list_directory(
    state: &MlflowState,
    tenant: &TenantContext,
    path: &str,
) -> MlflowResult<Vec<Value>> {
    let segments = sanitize_segments(path)?;
    let entries = ArtifactStore::new(state, tenant)?.list(segments).await?;
    Ok(artifact_entries_json(entries, ""))
}

/// Capability-safe recursive removal for an owner-derived artifact directory.
/// Callers provide only server-owned path components, but this boundary still
/// performs the same traversal and symlink validation as the public proxy.
pub(super) async fn delete_directory(
    state: &MlflowState,
    tenant: &TenantContext,
    path: &str,
) -> MlflowResult<()> {
    let segments = sanitize_segments(path)?;
    ArtifactStore::new(state, tenant)?.delete(segments).await
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
    let store = ArtifactStore::new(&state, &tenant)?;

    match method {
        axum::http::Method::PUT => {
            store.write(segments, body.to_vec()).await?;
            Ok(StatusCode::OK.into_response())
        }
        axum::http::Method::GET => {
            let Some(artifact) = store.read(segments.clone()).await? else {
                return Err(MlflowError::not_found(format!(
                    "artifact '{path}' does not exist"
                )));
            };
            match artifact {
                StoredArtifact::Directory(entries) => {
                    // Directory LIST. Entry names are repo-root-relative
                    // (relative to <exp>/<run>/artifacts) — the client feeds
                    // file.path verbatim into the next remote GET.
                    let prefix = list_entry_prefix(&segments);
                    Ok(Json(json!({
                        "files": artifact_entries_json(entries, &prefix)
                    }))
                    .into_response())
                }
                StoredArtifact::File(bytes) => {
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
        }
        axum::http::Method::DELETE => {
            store.delete(segments).await?;
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

/// The logged-model artifact root (minted on create by the wire; the
/// client round-trips it through this capability-safe proxy).
pub(super) fn model_artifact_uri(experiment_id: u64, model_id: &str) -> String {
    format!("mlflow-artifacts:/{experiment_id}/models/{model_id}/artifacts")
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::middleware::tenant::TenantIdSource;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn tenant_ctx(name: &str) -> TenantContext {
        TenantContext::new(name, TenantIdSource::Default)
    }

    fn test_state(data_dir: &std::path::Path) -> MlflowState {
        use proximadb_catalog::run_store::conformance_tests::InMemoryRunStoreFactory;

        MlflowState::new(
            Arc::new(InMemoryRunStoreFactory::new()),
            Arc::new(
                proximadb_catalog::model_registry_service::CatalogModelRegistryService::new(
                    Arc::new(crate::catalog::CatalogManager::new()),
                ),
            ),
            data_dir.to_path_buf(),
        )
    }

    fn test_router(data_dir: &std::path::Path, tenant: &str, relative: bool) -> Router {
        let routes = if relative {
            artifacts_routes_relative()
        } else {
            artifacts_routes()
        };
        routes
            .with_state(test_state(data_dir))
            .layer(axum::Extension(tenant_ctx(tenant)))
    }

    async fn request(router: &Router, method: &str, uri: &str, body: &'static [u8]) -> Response {
        router
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::from(body))
                    .expect("artifact test request must build"),
            )
            .await
            .expect("artifact test router must respond")
    }

    #[cfg(not(windows))]
    async fn assert_empty_listing(response: Response, context: &str) {
        assert_eq!(response.status(), StatusCode::OK, "{context}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("artifact listing body must be readable");
        let body: Value =
            serde_json::from_slice(&bytes).expect("artifact listing body must be valid JSON");
        assert_eq!(body, json!({ "files": [] }), "{context}");
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn root_list_preserves_empty_missing_and_file_contract() {
        let temp = tempfile::tempdir().expect("create artifact list contract directory");
        for (relative, prefix) in [(false, "/api/2.0/mlflow-artifacts"), (true, "")] {
            let tenant = if relative {
                "list-contract-relative"
            } else {
                "list-contract-canonical"
            };
            let router = test_router(temp.path(), tenant, relative);
            let root = format!("{prefix}/artifacts");
            let file = format!("{prefix}/artifacts/exp/run/artifacts/model.bin");

            assert_empty_listing(
                request(&router, "GET", &root, b"").await,
                "an absent tenant root must list as empty",
            )
            .await;
            assert_empty_listing(
                request(
                    &router,
                    "GET",
                    &format!("{root}?path=exp/run/artifacts/missing"),
                    b"",
                )
                .await,
                "a missing list path must list as empty",
            )
            .await;

            assert_eq!(
                request(&router, "PUT", &file, b"model-bytes")
                    .await
                    .status(),
                StatusCode::OK
            );
            assert_empty_listing(
                request(
                    &router,
                    "GET",
                    &format!("{root}?path=exp/run/artifacts/model.bin"),
                    b"",
                )
                .await,
                "listing a file must return an empty listing",
            )
            .await;
        }
    }

    #[test]
    fn nofollow_mapping_preserves_operational_io_failures() {
        let permission = nofollow_error(
            "model.bin",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        assert_eq!(permission.kind(), std::io::ErrorKind::PermissionDenied);

        let exhausted = nofollow_error(
            "model.bin",
            std::io::Error::new(std::io::ErrorKind::StorageFull, "full"),
        );
        assert_eq!(exhausted.kind(), std::io::ErrorKind::StorageFull);

        let unsafe_component = nofollow_error(
            "model.bin",
            std::io::Error::new(std::io::ErrorKind::NotADirectory, "not a directory"),
        );
        assert_eq!(
            unsafe_component.kind(),
            std::io::ErrorKind::InvalidInput,
            "a no-follow path violation remains a client error"
        );

        #[cfg(unix)]
        assert_eq!(
            nofollow_error("model.bin", std::io::Error::from_raw_os_error(libc::ELOOP),).kind(),
            std::io::ErrorKind::InvalidInput,
            "a Unix ELOOP remains a client-visible no-follow violation"
        );

        assert!(is_windows_stopped_on_symlink_code(Some(681)));
        assert!(!is_windows_stopped_on_symlink_code(Some(5)));
        assert!(!is_windows_stopped_on_symlink_code(None));

        assert!(
            !windows_artifact_operations_supported(),
            "local artifact operations must remain fail-closed on Windows until every path operation is handle-relative"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_ambient_capability_operations_fail_closed() {
        let temp = tempfile::tempdir().expect("create Windows delete test directory");
        let state = test_state(temp.path());
        let store = ArtifactStore::new(&state, &tenant_ctx("windows-delete-tenant"))
            .ok()
            .expect("construct artifact store");
        let file = ["tree", "model.bin"].map(str::to_owned);
        let operations = [
            store.read_sync(&file).map(|_| ()),
            store.list_sync(&["tree".to_owned()]).map(|_| ()),
            store.write_sync(&file, b"model"),
            store.delete_sync(&file),
        ];
        for result in operations {
            assert!(matches!(
                result,
                Err(error) if error.kind() == std::io::ErrorKind::Unsupported
            ));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn artifact_handlers_reject_symlinked_version_root() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("create artifact root test directory");
        let outside = temp.path().join("outside");
        tokio::fs::create_dir(&outside)
            .await
            .expect("create outside directory");
        symlink(&outside, temp.path().join("mlflow_artifacts_v2"))
            .expect("replace artifact version root with symlink");
        let router = test_router(temp.path(), "root-link-tenant", false);

        for (method, uri, body) in [
            (
                "PUT",
                "/api/2.0/mlflow-artifacts/artifacts/model.bin",
                &b"escaped-write"[..],
            ),
            (
                "GET",
                "/api/2.0/mlflow-artifacts/artifacts/model.bin",
                &b""[..],
            ),
            (
                "GET",
                "/api/2.0/mlflow-artifacts/artifacts?path=model.bin",
                &b""[..],
            ),
            (
                "DELETE",
                "/api/2.0/mlflow-artifacts/artifacts/model.bin",
                &b""[..],
            ),
        ] {
            assert_eq!(
                request(&router, method, uri, body).await.status(),
                StatusCode::BAD_REQUEST,
                "{method} {uri} must reject a symlinked version root"
            );
        }
        assert!(
            tokio::fs::read_dir(&outside)
                .await
                .expect("outside directory remains readable")
                .next_entry()
                .await
                .expect("outside directory iteration succeeds")
                .is_none(),
            "no request may create bytes outside the configured artifact root"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn deleting_directory_with_symlink_preserves_external_target() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("create artifact delete test directory");
        let outside = temp.path().join("outside.bin");
        tokio::fs::write(&outside, b"outside-secret")
            .await
            .expect("seed outside file");
        let tenant = "delete-link-tenant";
        let root = temp.path().join(
            tenant_artifact_relative_root(tenant)
                .ok()
                .expect("test tenant must encode"),
        );
        let directory = root.join("exp/run/artifacts/tree");
        tokio::fs::create_dir_all(&directory)
            .await
            .expect("create artifact directory");
        symlink(&outside, directory.join("external-link")).expect("create contained symlink");

        let router = test_router(temp.path(), tenant, false);
        let response = request(
            &router,
            "DELETE",
            "/api/2.0/mlflow-artifacts/artifacts/exp/run/artifacts/tree",
            b"",
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            tokio::fs::read(&outside)
                .await
                .expect("external target must survive"),
            b"outside-secret"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn artifact_handlers_reject_intermediate_symlink_escape_for_all_methods() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("create artifact test directory");
        let outside = temp.path().join("outside");
        tokio::fs::create_dir(&outside)
            .await
            .expect("create outside directory");
        let outside_secret = outside.join("secret.bin");
        tokio::fs::write(&outside_secret, b"outside-secret")
            .await
            .expect("seed outside secret");

        let tenant = "symlink-tenant";
        let root = temp.path().join(
            tenant_artifact_relative_root(tenant)
                .ok()
                .expect("test tenant must encode"),
        );
        tokio::fs::create_dir_all(&root)
            .await
            .expect("create tenant root");
        symlink(&outside, root.join("escape")).expect("create intermediate symlink");

        let router = test_router(temp.path(), tenant, false);
        for (method, uri, body) in [
            (
                "PUT",
                "/api/2.0/mlflow-artifacts/artifacts/escape/new.bin",
                &b"escaped-write"[..],
            ),
            (
                "GET",
                "/api/2.0/mlflow-artifacts/artifacts/escape/secret.bin",
                &b""[..],
            ),
            (
                "GET",
                "/api/2.0/mlflow-artifacts/artifacts?path=escape",
                &b""[..],
            ),
            (
                "DELETE",
                "/api/2.0/mlflow-artifacts/artifacts/escape/secret.bin",
                &b""[..],
            ),
        ] {
            let response = request(&router, method, uri, body).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{method} {uri} must reject a symlinked path"
            );
        }

        assert_eq!(
            tokio::fs::read(&outside_secret)
                .await
                .expect("outside secret must survive"),
            b"outside-secret"
        );
        assert!(
            tokio::fs::symlink_metadata(outside.join("new.bin"))
                .await
                .is_err(),
            "PUT must not create outside bytes"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn artifact_handlers_reject_final_symlinks() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("create artifact test directory");
        let outside = temp.path().join("outside.bin");
        tokio::fs::write(&outside, b"outside-secret")
            .await
            .expect("seed outside secret");
        let tenant = "final-link-tenant";
        let root = temp.path().join(
            tenant_artifact_relative_root(tenant)
                .ok()
                .expect("test tenant must encode"),
        );
        tokio::fs::create_dir_all(&root)
            .await
            .expect("create tenant root");
        symlink(&outside, root.join("link.bin")).expect("create final symlink");
        let router = test_router(temp.path(), tenant, false);

        for method in ["PUT", "GET", "DELETE"] {
            let response = request(
                &router,
                method,
                "/api/2.0/mlflow-artifacts/artifacts/link.bin",
                b"replacement",
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{method}");
        }
        assert_eq!(
            tokio::fs::read(&outside)
                .await
                .expect("outside file must survive"),
            b"outside-secret"
        );
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn canonical_and_ui_relative_routes_enforce_identical_validation() {
        let temp = tempfile::tempdir().expect("create artifact test directory");
        let overlong = "a".repeat(65);
        for (relative, prefix) in [(false, "/api/2.0/mlflow-artifacts"), (true, "")] {
            let valid = test_router(temp.path(), "route-parity", relative);
            let put_uri = format!("{prefix}/artifacts/exp/run/artifacts/model.bin");
            assert_eq!(
                request(&valid, "PUT", &put_uri, b"model").await.status(),
                StatusCode::OK
            );
            assert_eq!(
                request(&valid, "GET", &put_uri, b"").await.status(),
                StatusCode::OK
            );
            assert_eq!(
                request(
                    &valid,
                    "GET",
                    &format!("{prefix}/artifacts?path=exp/run/artifacts"),
                    b"",
                )
                .await
                .status(),
                StatusCode::OK
            );

            for tenant in ["", overlong.as_str()] {
                let invalid = test_router(temp.path(), tenant, relative);
                let response = request(
                    &invalid,
                    "PUT",
                    &format!("{prefix}/artifacts/model.bin"),
                    b"model",
                )
                .await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            }
            let traversal = request(
                &valid,
                "GET",
                &format!("{prefix}/artifacts?path=../outside"),
                b"",
            )
            .await;
            assert_eq!(traversal.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                request(&valid, "DELETE", &put_uri, b"").await.status(),
                StatusCode::OK
            );
        }
    }

    #[cfg(all(feature = "mlflow-ui", not(windows)))]
    #[tokio::test]
    async fn actual_ui_mount_serves_nested_artifact_proxy_routes() {
        let temp = tempfile::tempdir().expect("create UI artifact route directory");
        let router = Router::new()
            .nest(
                "/mlflow-ui",
                crate::network::rest::mlflow::ui::ui_mount_routes(),
            )
            .with_state(test_state(temp.path()))
            .layer(axum::Extension(tenant_ctx("ui-mount-tenant")));
        let file = "/mlflow-ui/ajax-api/2.0/mlflow-artifacts/artifacts/exp/run/artifacts/model.bin";
        assert_eq!(
            request(&router, "PUT", file, b"ui-model").await.status(),
            StatusCode::OK
        );
        let response = request(&router, "GET", file, b"").await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read nested UI artifact response");
        assert_eq!(&body[..], b"ui-model");
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn artifact_routes_isolate_tenant_put_get_list_and_delete() {
        let temp = tempfile::tempdir().expect("create artifact isolation directory");
        let tenant_a = test_router(temp.path(), "tenant-a", false);
        let tenant_b = test_router(temp.path(), "tenant-b", false);
        let file = "/api/2.0/mlflow-artifacts/artifacts/exp/run/artifacts/model.bin";
        let list = "/api/2.0/mlflow-artifacts/artifacts?path=exp/run/artifacts";

        assert_eq!(
            request(&tenant_a, "PUT", file, b"tenant-a-model")
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            request(&tenant_b, "GET", file, b"").await.status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            request(&tenant_b, "GET", list, b"").await.status(),
            StatusCode::OK
        );
        assert_eq!(
            request(&tenant_b, "DELETE", file, b"").await.status(),
            StatusCode::OK
        );
        assert_eq!(
            request(&tenant_a, "GET", file, b"").await.status(),
            StatusCode::OK,
            "another tenant's idempotent delete must not remove these bytes"
        );
        assert_eq!(
            request(&tenant_a, "GET", list, b"").await.status(),
            StatusCode::OK
        );
        assert_eq!(
            request(&tenant_a, "DELETE", file, b"").await.status(),
            StatusCode::OK
        );
        assert_eq!(
            request(&tenant_a, "GET", file, b"").await.status(),
            StatusCode::NOT_FOUND
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_artifact_handlers_fail_closed_without_creating_storage() {
        let temp = tempfile::tempdir().expect("create Windows artifact handler directory");
        let router = test_router(temp.path(), "windows-handler-tenant", false);
        let file = "/api/2.0/mlflow-artifacts/artifacts/exp/run/artifacts/model.bin";
        let list = "/api/2.0/mlflow-artifacts/artifacts?path=exp/run/artifacts";

        for (method, uri, body) in [
            ("PUT", file, &b"model"[..]),
            ("GET", file, &b""[..]),
            ("GET", list, &b""[..]),
            ("DELETE", file, &b""[..]),
        ] {
            assert_eq!(
                request(&router, method, uri, body).await.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "{method} {uri} must fail closed on Windows"
            );
        }

        assert!(
            tokio::fs::symlink_metadata(temp.path().join("mlflow_artifacts_v2"))
                .await
                .is_err(),
            "a rejected Windows artifact operation must not create storage"
        );
    }

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
            assert!(
                tokio::fs::symlink_metadata(&root).await.is_err(),
                "tenant {tenant} must not inherit bytes from the ambiguous legacy layout"
            );
        }
    }

    #[test]
    fn artifact_handlers_offload_capability_filesystem_work() {
        let source = include_str!("artifacts.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or_default();
        assert!(production.contains("tokio::task::spawn_blocking"));
        for forbidden in [
            [".", "exists()"].concat(),
            ["target", ".", "is_dir()"].concat(),
            [".", "is_file()"].concat(),
            ["tokio", "::", "fs", "::"].concat(),
        ] {
            assert!(
                !production.contains(&forbidden),
                "async artifact handlers contain ambient async filesystem access {forbidden}"
            );
        }
    }
}

#[allow(dead_code)]
fn _assert_send(_f: impl Fn() -> MlflowResult<Response>) {}
