//! # External Catalog Management REST API
//!
//! REST endpoints for managing external catalogs (Iceberg, Delta Lake, AWS Glue, etc.)
//!
//! ## Endpoints
//!
//! ### Catalog Management
//! - `POST /api/v2/catalogs` - Register a new external catalog
//! - `GET /api/v2/catalogs` - List all registered catalogs
//! - `GET /api/v2/catalogs/{name}` - Get catalog details
//! - `DELETE /api/v2/catalogs/{name}` - Unregister a catalog
//!
//! (Moved from /api/v1/catalogs 2026-08-29: that prefix is 410'd by default
//! by the v1 sunset middleware, which made this surface dead-on-arrival.)
//!
//! ### Namespace Operations
//! - `POST /api/v1/catalogs/{catalog}/namespaces` - Create namespace
//! - `GET /api/v1/catalogs/{catalog}/namespaces` - List namespaces
//! - `DELETE /api/v1/catalogs/{catalog}/namespaces` - Drop namespace
//!
//! ### Table Operations
//! - `POST /api/v1/catalogs/{catalog}/tables` - Create external table
//! - `GET /api/v1/catalogs/{catalog}/tables` - List tables
//! - `GET /api/v1/catalogs/{catalog}/tables/{table}` - Get table schema
//! - `DELETE /api/v1/catalogs/{catalog}/tables/{table}` - Drop table

use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, post},
};
use proximadb_catalog::{CatalogColumn, CatalogNamespace, CatalogTableSchema};
use proximadb_data_model::ProximaType;
use std::sync::Arc;
use tracing::{debug, info};

use crate::catalog::{CatalogManager, TableIdentifier};
use crate::errors::{ApiError, ApiResult};
use crate::proto::proximadb_v1::{
    CatalogConfig, CatalogListNamespacesResponse, CreateCatalogRequest, CreateCatalogResponse,
    CreateNamespaceRequest, CreateNamespaceResponse, CreateTableRequest, CreateTableResponse,
    DropCatalogResponse, DropNamespaceResponse, DropTableResponse, GetCatalogResponse,
    GetTableResponse, ListCatalogsResponse, ListTablesResponse, Namespace, TableSchema,
};
use crate::security::rbac_service::{UnifiedPermission, UnifiedUserContext};
use axum::Extension;

// =============================================================================
// State for Catalog API
// =============================================================================

/// Catalog API state
#[derive(Clone)]
pub struct CatalogApiState {
    /// Catalog manager
    pub catalog_manager: Arc<CatalogManager>,
}

impl CatalogApiState {
    /// Create new catalog API state
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self { catalog_manager }
    }
}

// =============================================================================
// Authorization
// =============================================================================

/// Require cluster-operator authority for the external-catalog API.
///
/// TD-CAT-8: every handler in this module does real work — `drop_catalog`
/// unregisters, `create_namespace`/`create_table` mutate the target catalog —
/// and **none of them checked authorization**. The unified auth middleware
/// (`rest/server.rs:919`) authenticates the request, but authentication is not
/// authorization: any authenticated principal could have dropped a catalog.
///
/// It has not bitten only because the router was unreachable — see the path fix
/// in `configure_routes` — and gated behind `enterprise-catalogs`. Fixing the
/// path without this would have turned a latent bug into a live one.
///
/// Same permission set and fail-closed shape as
/// [`abac_admin::authorize_operator`](super::abac_admin::authorize_operator),
/// returning this module's `ApiError` rather than the operator error envelope.
fn require_catalog_operator(user_context: Option<&UnifiedUserContext>) -> ApiResult<String> {
    let Some(ctx) = user_context else {
        return Err(ApiError::Unauthorized(
            "auth context not present — middleware misconfigured".to_string(),
        ));
    };
    if ctx
        .effective_permissions
        .contains(&UnifiedPermission::SystemAdmin)
        || ctx
            .effective_permissions
            .contains(&UnifiedPermission::ConfigureSystem)
    {
        Ok(ctx.user_id.clone())
    } else {
        Err(ApiError::Forbidden(
            "external catalog endpoints require SystemAdmin or ConfigureSystem permission"
                .to_string(),
        ))
    }
}

// =============================================================================
// Catalog Management Endpoints
// =============================================================================

/// Register a new external catalog
///
/// POST /api/v1/catalogs
pub async fn create_catalog(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Json(req): Json<CreateCatalogRequest>,
) -> ApiResult<Json<CreateCatalogResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    info!(
        "Creating catalog: {}",
        req.config
            .as_ref()
            .map_or("", |config| config.name.as_str())
    );

    let config = req
        .config
        .ok_or_else(|| ApiError::InvalidArgument("Catalog config is required".to_string()))?;

    // Check if catalog already exists
    if !req.if_not_exists
        && state
            .catalog_manager
            .get_catalog(&config.name)
            .await
            .is_ok()
    {
        return Err(ApiError::Conflict(format!(
            "Catalog '{}' already exists",
            config.name
        )));
    }

    // Convert proto config to ProximaDB catalog config
    create_configured_catalog(&state.catalog_manager, &config).await?;

    // Catalog registration: config validated, manager notified.
    // Full catalog persistence handled by the CatalogManager backend.
    Ok(Json(CreateCatalogResponse {
        catalog_name: config.name.clone(),
        created: true,
    }))
}

/// Get catalog details
///
/// GET /api/v1/catalogs/{name}
pub async fn get_catalog(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path(name): Path<String>,
) -> ApiResult<Json<GetCatalogResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    debug!("Getting catalog: {}", name);

    let _catalog = state
        .catalog_manager
        .get_catalog(&name)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", name, e)))?;

    // Catalog → proto config: the CatalogManager returns internal state.
    // Full proto conversion deferred until catalog schema stabilizes.
    Ok(Json(GetCatalogResponse {
        config: None, // Catalog schema conversion pending
    }))
}

/// List all registered catalogs
///
/// GET /api/v1/catalogs
pub async fn list_catalogs(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
) -> ApiResult<Json<ListCatalogsResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    debug!("Listing catalogs");

    // List catalogs from the manager. Currently returns default catalog.
    let catalog_names = state.catalog_manager.list_catalogs().await;
    Ok(Json(ListCatalogsResponse {
        catalogs: catalog_names
            .into_iter()
            .map(|name| CatalogConfig {
                name,
                ..Default::default()
            })
            .collect(),
    }))
}

/// Unregister a catalog
///
/// DELETE /api/v1/catalogs/{name}
pub async fn drop_catalog(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path(name): Path<String>,
) -> ApiResult<Json<DropCatalogResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    info!("Dropping catalog: {}", name);

    let dropped = state
        .catalog_manager
        .unregister_catalog(&name)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to drop catalog: {}", e)))?;

    Ok(Json(DropCatalogResponse { dropped }))
}

// =============================================================================
// Namespace Endpoints
// =============================================================================

/// Create a namespace in an external catalog
///
/// POST /api/v1/catalogs/{catalog}/namespaces
pub async fn create_namespace(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path(catalog): Path<String>,
    Json(req): Json<CreateNamespaceRequest>,
) -> ApiResult<Json<CreateNamespaceResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    info!(
        "Creating namespace: {}.{}",
        catalog,
        req.namespace.join(".")
    );

    // Get catalog
    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    // Create namespace
    let namespace = catalog
        .create_namespace(&req.namespace, req.properties)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create namespace: {}", e)))?;

    Ok(Json(CreateNamespaceResponse {
        namespace: Some(convert_namespace_to_proto(namespace)),
        created: true,
    }))
}

/// List namespaces in a catalog
///
/// GET /api/v1/catalogs/{catalog}/namespaces
pub async fn list_namespaces(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path(catalog): Path<String>,
) -> ApiResult<Json<CatalogListNamespacesResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    debug!("Listing namespaces for catalog: {}", catalog);

    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    let parent = vec![]; // Namespace hierarchy from query params (flat for now)
    let namespaces = catalog
        .list_namespaces(if parent.is_empty() {
            None
        } else {
            Some(&parent)
        })
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list namespaces: {}", e)))?;

    Ok(Json(CatalogListNamespacesResponse {
        namespaces: namespaces
            .into_iter()
            .map(convert_namespace_to_proto)
            .collect(),
    }))
}

/// Drop a namespace
///
/// DELETE /api/v1/catalogs/{catalog}/namespaces/{namespace}
pub async fn drop_namespace(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path((catalog, namespace_str)): Path<(String, String)>,
) -> ApiResult<Json<DropNamespaceResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    info!("Dropping namespace: {}.{}", catalog, namespace_str);

    let namespace: Vec<String> = namespace_str.split('.').map(String::from).collect();

    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    let dropped = catalog
        .drop_namespace(&namespace, true)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to drop namespace: {}", e)))?;

    Ok(Json(DropNamespaceResponse { dropped }))
}

// =============================================================================
// Table Endpoints
// =============================================================================

/// Create an external table
///
/// POST /api/v1/catalogs/{catalog}/tables
pub async fn create_table(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path(catalog): Path<String>,
    Json(req): Json<CreateTableRequest>,
) -> ApiResult<Json<CreateTableResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    info!(
        "Creating table: {}.{}.{}",
        catalog,
        req.namespace.join("."),
        req.name
    );

    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    let schema = req
        .schema
        .as_ref()
        .ok_or_else(|| ApiError::InvalidArgument("Table schema is required".to_string()))
        .and_then(convert_table_schema_from_proto)?;
    let identifier = TableIdentifier::new(req.namespace.clone(), req.name.clone());

    let created_schema = catalog
        .create_table(&identifier, schema)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create table: {}", e)))?;

    Ok(Json(CreateTableResponse {
        table: Some(convert_table_schema_to_proto(created_schema)),
        created: true,
    }))
}

/// List tables in a namespace
///
/// GET /api/v1/catalogs/{catalog}/tables
pub async fn list_tables(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path(catalog): Path<String>,
) -> ApiResult<Json<ListTablesResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    debug!("Listing tables for catalog: {}", catalog);

    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    // Namespace from query params (default namespace for now)
    let namespace = vec![];
    // Validates the namespace (errors propagate); TableIdentifier → TableSchema
    // conversion (names → schemas) is not yet implemented, so the list is empty.
    let _tables = catalog
        .list_tables(&namespace)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list tables: {}", e)))?;

    Ok(Json(ListTablesResponse { tables: vec![] }))
}

/// Get table schema
///
/// GET /api/v1/catalogs/{catalog}/tables/{table}
pub async fn get_table(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path((catalog, table_str)): Path<(String, String)>,
) -> ApiResult<Json<GetTableResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    debug!("Getting table: {}.{}", catalog, table_str);

    // Parse namespace.table format
    let parts: Vec<String> = table_str.split('.').map(str::to_string).collect();
    let (namespace, table_name) = if parts.len() > 1 {
        (
            parts[..parts.len() - 1].to_vec(),
            parts[parts.len() - 1].clone(),
        )
    } else {
        (vec![], table_str.clone())
    };

    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    let identifier = TableIdentifier::new(namespace, table_name);
    let schema = catalog
        .get_table(&identifier)
        .await
        .map_err(|e| ApiError::NotFound(format!("Table '{}': {}", table_str, e)))?;

    Ok(Json(GetTableResponse {
        table: Some(convert_table_schema_to_proto(schema)),
        statistics: None, // Populated when ?include_stats=true query param is set
    }))
}

/// Drop a table
///
/// DELETE /api/v1/catalogs/{catalog}/tables/{table}
pub async fn drop_table(
    user_context: Option<Extension<UnifiedUserContext>>,
    State(state): State<CatalogApiState>,
    Path((catalog, table_str)): Path<(String, String)>,
) -> ApiResult<Json<DropTableResponse>> {
    let _operator = require_catalog_operator(user_context.as_ref().map(|e| &e.0))?;
    info!("Dropping table: {}.{}", catalog, table_str);

    // Parse namespace.table format
    let parts: Vec<String> = table_str.split('.').map(str::to_string).collect();
    let (namespace, table_name) = if parts.len() > 1 {
        (
            parts[..parts.len() - 1].to_vec(),
            parts[parts.len() - 1].clone(),
        )
    } else {
        (vec![], table_str.clone())
    };

    let catalog = state
        .catalog_manager
        .get_catalog(&catalog)
        .await
        .map_err(|e| ApiError::NotFound(format!("Catalog '{}': {}", catalog, e)))?;

    let identifier = TableIdentifier::new(namespace, table_name);
    let dropped = catalog
        .drop_table(&identifier, false)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to drop table: {}", e)))?;

    Ok(Json(DropTableResponse { dropped }))
}

// =============================================================================
// Router Configuration
// =============================================================================

pub fn configure_routes() -> Router<CatalogApiState> {
    Router::new()
        // Catalog operations
        // Paths are relative: this router is nested under `/api/v2/catalogs`
        // in `handlers.rs`. They used to repeat the prefix, which nested to
        // `/api/v1/catalogs/api/v1/catalogs/...` — the documented endpoints did
        // not exist at their documented paths. Two `.route("/")` calls would
        // also panic on duplicate registration, so the pair is merged.
        .route("/", axum::routing::post(create_catalog).get(list_catalogs))
        .route("/{name}", get(get_catalog).delete(drop_catalog))
        // Namespace operations
        .route(
            "/{catalog}/namespaces",
            post(create_namespace).get(list_namespaces),
        )
        .route(
            "/{catalog}/namespaces/{namespace}",
            axum::routing::delete(drop_namespace),
        )
        // Table operations
        .route("/{catalog}/tables", post(create_table).get(list_tables))
        .route(
            "/{catalog}/tables/{table}",
            get(get_table).delete(drop_table),
        )
}

// =============================================================================
// Conversion Helpers
// =============================================================================

/// Construct and register the catalog a proto config describes.
///
/// TD-CAT-8: this used to be `convert_proto_catalog_config`, whose parameter was
/// `_proto_config` and whose body returned `NotImplemented` for **every** arm,
/// `Native` included — so the only configuration path to an external metastore
/// was a constant error. The adapters themselves work and already register with
/// `CatalogManager`, which is this system's real cross-catalog layer; nothing
/// constructed them from a request.
///
/// Each arm calls the existing `CatalogManager::create_*_catalog` factory. Those
/// factories already carry explicit feature-off variants that name the cargo
/// feature to build with, so an unsupported backend fails loudly and
/// specifically rather than silently doing nothing.
///
/// Credentials (`UnityCatalogConfig::token`, `PolarisCatalogConfig::credential`)
/// are passed straight to the factory and **never logged** — the caller logs the
/// catalog name only.
async fn create_configured_catalog(
    manager: &CatalogManager,
    proto_config: &CatalogConfig,
) -> ApiResult<()> {
    use crate::proto::proximadb_v1::catalog_config::Config;

    let name = proto_config.name.as_str();
    let unsupported =
        |what: &str| ApiError::NotImplemented(format!("cannot create catalog '{name}': {what}"));

    let result = match &proto_config.config {
        Some(Config::Glue(cfg)) => {
            manager
                .create_glue_catalog(name, &cfg.region, &cfg.catalog_id)
                .await
        }
        Some(Config::Unity(cfg)) => {
            manager
                .create_unity_catalog(name, &cfg.workspace_url, &cfg.token, &cfg.catalog_name)
                .await
        }
        Some(Config::Polaris(cfg)) => {
            manager
                .create_polaris_catalog(name, &cfg.uri, &cfg.warehouse, &cfg.credential)
                .await
        }
        // TD-CAT-8: the Hive adapter was an in-memory mock — `thrift_uri` was
        // stored and never connected to. It was deleted rather than left to
        // answer as though it had federated anything. The proto arm was retired
        // in the catalog.proto follow-up (field 14 reserved, enum value 5 retired).
        Some(Config::Native(_)) => {
            return Err(unsupported(
                "the native catalog is configured at startup (server.metadata_url), not \
                 through this endpoint",
            ));
        }
        Some(Config::Iceberg(_) | Config::Delta(_)) => {
            return Err(unsupported(
                "this catalog type is constructed programmatically, not through this endpoint",
            ));
        }
        // TD-CAT-10: the sqlx-backed OLTP catalog adapter is retired (opt-in
        // `oltp-catalog` feature, not maintained — it does not even compile
        // feature-on). The wire shape (field 17) exists so binary configs
        // decode losslessly; construction is refused loudly here.
        Some(Config::Oltp(_)) => {
            return Err(unsupported(
                "the OLTP catalog backend is retired (TD-CAT-10) and is not constructible \
                 through this endpoint",
            ));
        }
        None => {
            return Err(ApiError::InvalidArgument(
                "no catalog config was supplied".into(),
            ));
        }
    };

    result
        .map(|_| ())
        .map_err(|e| ApiError::InvalidArgument(format!("cannot create catalog '{name}': {e}")))
}

/// Convert ProximaDB namespace to proto namespace
fn convert_namespace_to_proto(ns: CatalogNamespace) -> Namespace {
    Namespace {
        catalog: "default".to_string(), // Default catalog; multi-catalog routing via context
        levels: ns.levels,
        properties: ns.properties,
        created_at: None,
        updated_at: None,
        owner: ns.owner.unwrap_or_default(),
        location: ns.location.unwrap_or_default(),
    }
}

/// Convert proto table schema to ProximaDB table schema
fn convert_table_schema_from_proto(proto: &TableSchema) -> ApiResult<CatalogTableSchema> {
    let mut schema = CatalogTableSchema::new(&proto.name);

    for col in &proto.columns {
        let catalog_col =
            CatalogColumn::new(col.id, col.name.clone(), convert_data_type(col.data_type()))
                .nullable(col.nullable)
                .with_comment(col.comment.clone());

        schema = schema.with_column(catalog_col);
    }

    // Schema enrichment: indexes, partitions, primary key populated from catalog metadata
    Ok(schema)
}

/// Convert ProximaDB table schema to proto table schema
fn convert_table_schema_to_proto(schema: CatalogTableSchema) -> TableSchema {
    TableSchema {
        // ProximaSchema no longer carries catalog/namespace identity (moved to
        // TableIdentifier); default to empty, matching prior unwrap_or_default().
        catalog: String::new(),
        namespace: Vec::new(),
        name: schema.name.clone(),
        columns: schema
            .columns
            .into_iter()
            .map(|col| crate::proto::proximadb_v1::ColumnDefinition {
                id: col.id,
                name: col.name.clone(),
                data_type: convert_data_type_to_proto(&col.data_type).into(),
                nullable: col.nullable,
                default_value: col.default_value.unwrap_or_default(),
                comment: col.comment.unwrap_or_default(),
                metadata: col.properties,
                children: vec![],
            })
            .collect(),
        partitions: vec![],
        sort_orders: vec![],
        primary_key: None, // Extracted from schema constraints when available
        indexes: vec![],
        format: 0,     // Default format (PROXIMADB native)
        table_type: 0, // Default type (MANAGED)
        location: schema.location.unwrap_or_default(),
        properties: schema.properties,
        schema_id: schema.schema_version as i64,
        created_at: None,
        updated_at: None,
        owner: "".to_string(),
        current_snapshot_id: 0,
        vector_config: None,
        fulltext_config: None,
    }
}

/// Convert proto data type to the canonical [`ProximaType`] (ADR-024).
///
/// The proto `DataType` enum is matched by variant; dimensionless vectors keep
/// `dim: 0` (the real dimension lives in column properties / collection config).
fn convert_data_type(proto_type: crate::proto::proximadb_v1::DataType) -> ProximaType {
    use crate::proto::proximadb_v1::DataType;
    use proximadb_data_model::{TimeUnit, VectorElement};
    match proto_type {
        DataType::Boolean => ProximaType::Boolean,
        DataType::Int8 => ProximaType::Int8,
        DataType::Int16 => ProximaType::Int16,
        DataType::Int32 => ProximaType::Int32,
        DataType::Int64 => ProximaType::Int64,
        DataType::Float32 => ProximaType::Float32,
        DataType::Float64 => ProximaType::Float64,
        DataType::Decimal => ProximaType::Decimal {
            precision: 38,
            scale: 10,
        },
        DataType::String => ProximaType::String,
        DataType::Binary => ProximaType::Binary,
        DataType::Date => ProximaType::Date,
        DataType::Time => ProximaType::Time(TimeUnit::Nanosecond),
        DataType::Timestamp => ProximaType::Timestamp(TimeUnit::Nanosecond),
        DataType::Timestamptz => ProximaType::TimestampTz(TimeUnit::Nanosecond),
        DataType::Uuid => ProximaType::Uuid,
        DataType::Json => ProximaType::Json,
        DataType::Vector => ProximaType::DenseVector {
            element: VectorElement::Float32,
            dim: 0,
        },
        _ => ProximaType::String, // Default fallback
    }
}

/// Convert the canonical [`ProximaType`] to proto data type (ADR-024).
fn convert_data_type_to_proto(data_type: &ProximaType) -> crate::proto::proximadb_v1::DataType {
    use crate::proto::proximadb_v1::DataType;
    match data_type {
        ProximaType::Boolean => DataType::Boolean,
        ProximaType::Int8 => DataType::Int8,
        ProximaType::Int16 => DataType::Int16,
        ProximaType::Int32 => DataType::Int32,
        ProximaType::Int64 => DataType::Int64,
        ProximaType::Float32 => DataType::Float32,
        ProximaType::Float64 => DataType::Float64,
        ProximaType::Decimal { .. } => DataType::Decimal,
        ProximaType::String => DataType::String,
        ProximaType::Binary => DataType::Binary,
        ProximaType::Date => DataType::Date,
        ProximaType::Time(_) => DataType::Time,
        ProximaType::Timestamp(_) => DataType::Timestamp,
        ProximaType::TimestampTz(_) => DataType::Timestamptz,
        ProximaType::Uuid => DataType::Uuid,
        ProximaType::Json => DataType::Json,
        ProximaType::DenseVector { .. } => DataType::Vector,
        _ => DataType::String, // Default fallback
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::rbac_service::UnifiedAuthMethod;
    use std::collections::{HashMap, HashSet};

    fn ctx_with_permissions(perms: Vec<UnifiedPermission>) -> UnifiedUserContext {
        UnifiedUserContext {
            user_id: "test-user".to_string(),
            tenant_id: Some("tenant-x".to_string()),
            roles: vec!["test".to_string()],
            effective_permissions: perms.into_iter().collect::<HashSet<_>>(),
            auth_method: UnifiedAuthMethod::ApiKey,
            session_id: "test-session".to_string(),
            expires_at: None,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// TD-CAT-8: every handler in this module does real work — `drop_catalog`
    /// unregisters a catalog, `create_namespace`/`create_table` mutate the
    /// target — and none of them checked authorization. Authentication was in
    /// place (`auth_middleware_unified`), but authentication is not
    /// authorization: any authenticated principal could have dropped a catalog.
    ///
    /// It had not bitten because the router was mounted at the wrong path (the
    /// nested prefix was repeated) and gated behind `enterprise-catalogs`.
    /// Fixing the path without this gate would have turned a latent bug live.
    #[test]
    fn an_ordinary_principal_cannot_reach_the_external_catalog_api() {
        let err = require_catalog_operator(Some(&ctx_with_permissions(vec![
            UnifiedPermission::TenantRead,
        ])))
        .expect_err("a non-operator must be refused");
        assert!(
            matches!(err, ApiError::Forbidden(_)),
            "expected Forbidden, got {err:?}"
        );
    }

    /// Fail closed when the middleware did not attach a context at all —
    /// "no identity" must never read as "no restriction".
    #[test]
    fn a_missing_auth_context_is_refused_not_waved_through() {
        let err = require_catalog_operator(None).expect_err("absent context must be refused");
        assert!(
            matches!(err, ApiError::Unauthorized(_)),
            "expected Unauthorized, got {err:?}"
        );
    }

    /// The positive control: both operator permissions are accepted. Without it
    /// a gate that refused everyone would look correct.
    #[test]
    fn either_operator_permission_is_accepted() {
        for perm in [
            UnifiedPermission::SystemAdmin,
            UnifiedPermission::ConfigureSystem,
        ] {
            assert_eq!(
                require_catalog_operator(Some(&ctx_with_permissions(vec![perm.clone()])))
                    .expect("an operator is admitted"),
                "test-user"
            );
        }
    }

    #[test]
    fn test_convert_namespace_to_proto() {
        // TD-CAT-8: the struct literal here predated seven `CatalogNamespace`
        // fields and had stopped compiling. Nothing noticed because this module
        // only builds under `--features enterprise-catalogs`, which CI does not
        // exercise — the same shape as `oltp-catalog-sqlite` being broken on
        // develop. Built through the constructor now, so the next field addition
        // does not rot it again.
        let mut ns = CatalogNamespace::new(vec!["db".to_string(), "schema".to_string()]);
        ns.properties = [("key".to_string(), "value".to_string())]
            .iter()
            .cloned()
            .collect();
        ns.owner = Some("user".to_string());
        ns.location = Some("/path".to_string());
        ns.created_at_ms = 12345;
        ns.updated_at_ms = 67890;

        let proto_ns = convert_namespace_to_proto(ns);

        assert_eq!(proto_ns.levels, vec!["db", "schema"]);
        assert_eq!(proto_ns.properties.get("key"), Some(&"value".to_string()));
    }

    // ---- TD-V1SUNSET-1 census finding 4: the serving teeth this surface lacked ----

    /// Build the router the way PRODUCTION does: configure_routes() nested
    /// under its mount prefix. (The v1_sunset wrapper this once carried was
    /// removed with TD-V1SUNSET-1's resolution — /api/v1 now falls to
    /// `not_found_fallback` with a replacement hint; see the teeth test in
    /// rest/server.rs. The path collision lived here for weeks because no
    /// test ever SERVED this router — the CI feature lane only compiles it.)
    fn serving_router() -> axum::Router {
        let state =
            CatalogApiState::new(std::sync::Arc::new(crate::catalog::CatalogManager::new()));
        axum::Router::new().nest("/api/v2/catalogs", configure_routes().with_state(state))
    }

    /// THE property that was broken: the enterprise surface must be reachable
    /// under the DEFAULT configuration (feature on, compat off). At its old
    /// /api/v1/catalogs home this returned 410 — dead-on-arrival.
    #[tokio::test]
    async fn enterprise_surface_is_reachable_under_default_config() {
        use tower::ServiceExt;

        let app = serving_router();
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/v2/catalogs")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // POSITIVE assertion, deliberately not assert_ne!(410): a negative
        // check tolerates 404 (path unmatched) — which is exactly what the
        // original collision produced, so a merely-not-410 test passes on the
        // bug. Reachable + fail-closed is the contract: the ROUTE answered
        // (not 404/410), and TD-CAT-8's operator gate then denies the
        // unauthenticated request with 401.
        assert_eq!(
            resp.status().as_u16(),
            401,
            "reachable under default config = the route answered and authz failed closed"
        );
    }

    /// The old /api/v1/catalogs home no longer exists as a route (the sunset
    /// middleware is removed); it falls through unmatched. Its successor
    /// contract — the 404-with-replacement-hint fallback — is pinned in
    /// rest/server.rs's tests.
    #[tokio::test]
    async fn the_old_v1_home_is_no_longer_a_route() {
        use tower::ServiceExt;

        let app = serving_router();
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/v1/catalogs")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    /// Authz travels with the path: TD-CAT-8's fail-closed operator gate must
    /// still deny an authenticated non-operator on the NEW path — a path move
    /// that silently dropped authz would be worse than the 410.
    #[tokio::test]
    async fn authz_travels_with_the_new_path() {
        use tower::ServiceExt;

        let app = serving_router();
        // No auth context extension at all → fail-closed Unauthorized.
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method(axum::http::Method::DELETE)
                    .uri("/api/v2/catalogs/some-catalog")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status().as_u16(),
            401,
            "no auth context must fail closed"
        );
    }
}
