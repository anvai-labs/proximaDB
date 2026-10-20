//! Iceberg REST Catalog server — Axum route handlers
//!
//! Implements the Apache Iceberg REST Catalog specification v1 so external engines
//! (Trino, Spark, DuckDB, PyIceberg, Flink) can connect to ProximaDB as their catalog
//! without a custom connector.
//!
//! Mount point: `/iceberg/v1`
//!
//! ## Client configuration examples
//!
//! **PyIceberg**
//! ```python
//! from pyiceberg.catalog.rest import RestCatalog
//! cat = RestCatalog("proximadb", uri="http://localhost:5678/iceberg/v1")
//! ```
//!
//! **Spark**
//! ```text
//! spark.conf.set("spark.sql.catalog.proximadb", "org.apache.iceberg.spark.SparkCatalog")
//! spark.conf.set("spark.sql.catalog.proximadb.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
//! spark.conf.set("spark.sql.catalog.proximadb.uri", "http://localhost:5678/iceberg/v1")
//! ```
//!
//! **Trino**
//! ```properties
//! connector.name=iceberg
//! iceberg.catalog.type=rest
//! iceberg.rest-catalog.uri=http://localhost:5678/iceberg/v1
//! ```
//!
//! **DuckDB**
//! ```sql
//! INSTALL iceberg; LOAD iceberg;
//! ATTACH 'http://localhost:5678/iceberg/v1' AS proximadb (TYPE ICEBERG);
//! ```

use std::sync::Arc;

use axum::{
    Extension, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::catalog::CatalogManager;
use crate::catalog::iceberg_rest_service::{
    IcebergCommitTableRequest, IcebergCreateNamespaceRequest, IcebergCreateTableRequest,
    IcebergErrorResponse, IcebergRegisterTableRequest, IcebergRestService,
    IcebergUpdateNamespacePropertiesRequest,
};
use crate::network::middleware::tenant::TenantContext;

/// Shared state for the Iceberg REST catalog server
#[derive(Clone)]
pub struct IcebergRestState {
    pub service: Arc<IcebergRestService>,
    pub segment_registry: Arc<crate::catalog::SegmentRegistry>,
}

impl IcebergRestState {
    pub fn new(
        catalog_manager: Arc<CatalogManager>,
        warehouse: impl Into<String>,
        flight_endpoint: impl Into<String>,
        server_base_url: impl Into<String>,
    ) -> Self {
        Self {
            service: Arc::new(IcebergRestService::new(
                catalog_manager,
                warehouse,
                flight_endpoint,
                server_base_url,
            )),
            segment_registry: Arc::new(crate::catalog::SegmentRegistry::new()),
        }
    }

    pub fn with_defaults(catalog_manager: Arc<CatalogManager>) -> Self {
        Self::new(
            catalog_manager,
            "proximadb",
            "grpc://localhost:5680",
            "http://localhost:5678/iceberg/v1",
        )
    }

    /// Replace the default registry with the shared one from `SharedServices`.
    pub fn with_segment_registry(mut self, registry: Arc<crate::catalog::SegmentRegistry>) -> Self {
        // Rebuild the service with the registry wired in.
        let inner = Arc::try_unwrap(self.service)
            .unwrap_or_else(|arc| (*arc).clone())
            .with_segment_registry(registry.clone());
        self.service = Arc::new(inner);
        self.segment_registry = registry;
        self
    }

    /// Wire the warehouse object-store bridge so `load_table` persists and serves a real
    /// `metadata.json` with a manifest-log-driven snapshot history (TD-119).
    pub fn with_object_store_bridge(
        mut self,
        bridge: Arc<dyn proximadb_storage_common::object_store_bridge::ObjectStoreBridge>,
    ) -> Self {
        let inner = Arc::try_unwrap(self.service)
            .unwrap_or_else(|arc| (*arc).clone())
            .with_object_store_bridge(bridge);
        self.service = Arc::new(inner);
        self
    }
}

// ============================================================================
// Router
// ============================================================================

/// Create the Iceberg REST catalog router. Mount at `/iceberg/v1`.
pub fn create_iceberg_rest_router() -> Router<IcebergRestState> {
    Router::new()
        // Config
        .route("/config", get(get_config))
        // OAuth2 (returns 501 when auth is disabled)
        .route("/oauth/tokens", post(oauth_tokens))
        // Namespaces
        .route("/namespaces", get(list_namespaces).post(create_namespace))
        .route(
            "/namespaces/{namespace}",
            get(get_namespace)
                .head(namespace_exists)
                .delete(drop_namespace),
        )
        .route(
            "/namespaces/{namespace}/properties",
            post(update_namespace_properties),
        )
        // Tables
        .route(
            "/namespaces/{namespace}/tables",
            get(list_tables).post(create_table),
        )
        .route(
            "/namespaces/{namespace}/tables/{table}",
            get(load_table)
                .head(table_exists)
                .delete(drop_table)
                .post(commit_table),
        )
        // Serve a persisted metadata.json so `metadata-location` resolves (TD-119)
        .route(
            "/namespaces/{namespace}/tables/{table}/metadata/{file}",
            get(load_table_metadata_file),
        )
        // Register existing table
        .route("/namespaces/{namespace}/register", post(register_table))
        // Views (v2 stub — returns empty list)
        .route("/namespaces/{namespace}/views", get(list_views))
}

// ============================================================================
// Query params
// ============================================================================

#[derive(Debug, Deserialize)]
struct ListNamespacesParams {
    parent: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DropTableParams {
    purge: Option<bool>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse multi-level namespace from a URL path segment.
/// Iceberg uses the unit separator (0x1F) as level delimiter in paths.
fn parse_namespace(raw: &str) -> Vec<String> {
    if raw.contains('\x1f') {
        raw.split('\x1f').map(String::from).collect()
    } else if raw.contains('.') {
        raw.split('.').map(String::from).collect()
    } else {
        vec![raw.to_string()]
    }
}

/// Convert an `anyhow::Error` to an appropriate Iceberg JSON error response.
fn err_to_response(err: anyhow::Error) -> Response {
    let msg = err.to_string();
    if msg.starts_with("Bad request:") {
        return (
            StatusCode::BAD_REQUEST,
            Json(IcebergErrorResponse::bad_request(msg)),
        )
            .into_response();
    }
    if msg.contains("not found") || msg.contains("NoSuch") || msg.contains("does not exist") {
        (
            StatusCode::NOT_FOUND,
            Json(IcebergErrorResponse::not_found(msg)),
        )
            .into_response()
    } else if msg.contains("already exists") || msg.contains("AlreadyExists") {
        (
            StatusCode::CONFLICT,
            Json(IcebergErrorResponse::already_exists(msg)),
        )
            .into_response()
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(IcebergErrorResponse::internal(msg)),
        )
            .into_response()
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /config — return catalog configuration
async fn get_config(State(state): State<IcebergRestState>) -> impl IntoResponse {
    let config = state.service.get_config();
    Json(config)
}

/// POST /oauth/tokens — OAuth2 token exchange (returns 501 if not configured)
async fn oauth_tokens() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(IcebergErrorResponse {
            error: crate::catalog::iceberg_rest_service::IcebergError {
                message: "OAuth2 token exchange not configured. Use header-based auth or no-auth."
                    .to_string(),
                error_type: "NotAuthorizedException".to_string(),
                code: 501,
                stack: vec![],
            },
        }),
    )
}

/// GET /namespaces — list namespaces
async fn list_namespaces(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Query(params): Query<ListNamespacesParams>,
) -> Response {
    debug!("Iceberg REST: list_namespaces parent={:?}", params.parent);
    match state
        .service
        .list_namespaces(&tenant.tenant_id, params.parent.as_deref())
        .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// POST /namespaces — create namespace
async fn create_namespace(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<IcebergCreateNamespaceRequest>,
) -> Response {
    info!("Iceberg REST: create_namespace {:?}", req.namespace);
    match state.service.create_namespace(&tenant.tenant_id, req).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// GET /namespaces/:namespace — load namespace properties
async fn get_namespace(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    debug!("Iceberg REST: get_namespace {:?}", namespace);
    match state
        .service
        .get_namespace(&tenant.tenant_id, namespace)
        .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// HEAD /namespaces/:namespace — check namespace existence
async fn namespace_exists(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
) -> StatusCode {
    let namespace = parse_namespace(&namespace_raw);
    match state
        .service
        .namespace_exists(&tenant.tenant_id, namespace)
        .await
    {
        Ok(true) => StatusCode::NO_CONTENT,
        Ok(false) => StatusCode::NOT_FOUND,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// DELETE /namespaces/:namespace — drop namespace
async fn drop_namespace(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    info!("Iceberg REST: drop_namespace {:?}", namespace);
    match state
        .service
        .drop_namespace(&tenant.tenant_id, namespace)
        .await
    {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(IcebergErrorResponse::not_found("Namespace not found")),
        )
            .into_response(),
        Err(e) => err_to_response(e),
    }
}

/// POST /namespaces/:namespace/properties — update namespace properties
async fn update_namespace_properties(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
    Json(req): Json<IcebergUpdateNamespacePropertiesRequest>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    match state
        .service
        .update_namespace_properties(&tenant.tenant_id, namespace, req)
        .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// GET /namespaces/:namespace/tables — list tables
async fn list_tables(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    debug!("Iceberg REST: list_tables {:?}", namespace);
    match state
        .service
        .list_tables(&tenant.tenant_id, namespace)
        .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// POST /namespaces/:namespace/tables — create table
async fn create_table(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
    Json(req): Json<IcebergCreateTableRequest>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    info!(
        "Iceberg REST: create_table {}.{}",
        namespace.join("."),
        req.name
    );
    match state
        .service
        .create_table(&tenant.tenant_id, namespace, req)
        .await
    {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// GET /namespaces/:namespace/tables/:table — load table metadata
async fn load_table(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace_raw, table)): Path<(String, String)>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    debug!("Iceberg REST: load_table {}.{}", namespace.join("."), table);
    match state
        .service
        .load_table(&tenant.tenant_id, namespace, table)
        .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// GET /namespaces/:namespace/tables/:table/metadata/:file — serve a persisted
/// `metadata.json` so the `metadata-location` returned by load_table resolves (TD-119).
async fn load_table_metadata_file(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace_raw, table, file)): Path<(String, String, String)>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    let version = match file
        .strip_prefix('v')
        .and_then(|s| s.strip_suffix(".metadata.json"))
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(v) => v,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid metadata file name: {file}"),
            )
                .into_response();
        }
    };
    match state
        .service
        .read_table_metadata_file(&tenant.tenant_id, namespace, table, version)
        .await
    {
        Ok(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Err(e) => err_to_response(e),
    }
}

/// HEAD /namespaces/:namespace/tables/:table — check table existence
async fn table_exists(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace_raw, table)): Path<(String, String)>,
) -> StatusCode {
    let namespace = parse_namespace(&namespace_raw);
    match state
        .service
        .table_exists(&tenant.tenant_id, namespace, table)
        .await
    {
        Ok(true) => StatusCode::NO_CONTENT,
        Ok(false) => StatusCode::NOT_FOUND,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// DELETE /namespaces/:namespace/tables/:table — drop table
async fn drop_table(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace_raw, table)): Path<(String, String)>,
    Query(params): Query<DropTableParams>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    let purge = params.purge.unwrap_or(false);
    info!(
        "Iceberg REST: drop_table {}.{} purge={}",
        namespace.join("."),
        table,
        purge
    );
    match state
        .service
        .drop_table(&tenant.tenant_id, namespace, table, purge)
        .await
    {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(IcebergErrorResponse::not_found("Table not found")),
        )
            .into_response(),
        Err(e) => err_to_response(e),
    }
}

/// POST /namespaces/:namespace/tables/:table — commit table changes
async fn commit_table(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace_raw, table)): Path<(String, String)>,
    Json(req): Json<IcebergCommitTableRequest>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    debug!(
        "Iceberg REST: commit_table {}.{}",
        namespace.join("."),
        table
    );
    match state
        .service
        .commit_table(&tenant.tenant_id, namespace, table, req)
        .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// POST /namespaces/:namespace/register — register an existing table
async fn register_table(
    State(state): State<IcebergRestState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace_raw): Path<String>,
    Json(req): Json<IcebergRegisterTableRequest>,
) -> Response {
    let namespace = parse_namespace(&namespace_raw);
    info!(
        "Iceberg REST: register_table {}.{}",
        namespace.join("."),
        req.name
    );
    match state
        .service
        .register_table(&tenant.tenant_id, namespace, req)
        .await
    {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => err_to_response(e),
    }
}

/// GET /namespaces/:namespace/views — list views (v2 stub, returns empty list)
#[derive(Serialize)]
struct EmptyViewsResponse {
    identifiers: Vec<serde_json::Value>,
}

async fn list_views(Path(_namespace_raw): Path<String>) -> impl IntoResponse {
    Json(EmptyViewsResponse {
        identifiers: vec![],
    })
}
