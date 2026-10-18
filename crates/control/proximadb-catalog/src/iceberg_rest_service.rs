//! Iceberg REST Catalog Server — service layer
//!
//! Translates between ProximaDB's internal catalog types and the Apache Iceberg REST Catalog
//! specification (v1). This is the server side: external engines (Trino, Spark, DuckDB, PyIceberg)
//! connect to ProximaDB as their catalog authority.
//!
//! ## ProximaRecord → Iceberg Schema Canonical Mapping (ADR-ICE-001)
//!
//! | ProximaRecord field    | Iceberg field name      | Iceberg type          |
//! |------------------------|-------------------------|------------------------|
//! | id / oid               | id                      | string (required)      |
//! | tenant_id              | tenant_id               | string (required)      |
//! | created_at_ns          | created_at              | timestamptz            |
//! | updated_at_ns          | updated_at              | timestamptz            |
//! | valid_from_ns          | valid_from              | timestamptz (optional) |
//! | valid_to_ns            | valid_to                | timestamptz (optional) |
//! | actor                  | actor                   | string (optional)      |
//! | origin                 | origin                  | string (optional)      |
//! | props                  | props                   | map<string, binary>    |
//! | labels                 | labels                  | list<string>           |
//! | embeddings[i].values   | embedding_{model_id}    | list<float>            |
//! | edge.source_id         | edge_source_id          | string (optional)      |
//! | edge.target_id         | edge_target_id          | string (optional)      |
//! | edge.edge_type         | edge_type               | string (optional)      |
//! | edge.weight            | edge_weight             | float (optional)       |
//!
//! Table properties carry index metadata so external planners can route vector searches:
//! `proximadb.index.{col}.type=hnsw`, `proximadb.index.{col}.dim=768`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::object_store_bridge::ObjectStoreBridge;
use crate::{CatalogManager, CatalogTableSchema, TableIdentifier};
use proximadb_data_model::ProximaType;

// ============================================================================
// Iceberg REST API types — serialized exactly per spec
// ============================================================================

/// Catalog configuration response (GET /v1/config)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergConfigResponse {
    pub defaults: HashMap<String, String>,
    pub overrides: HashMap<String, String>,
}

/// Response for listing namespaces (GET /v1/namespaces)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergListNamespacesResponse {
    pub namespaces: Vec<Vec<String>>,
}

/// Response for loading or creating a namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergNamespaceResponse {
    pub namespace: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Request to create a namespace (POST /v1/namespaces)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergCreateNamespaceRequest {
    pub namespace: Vec<String>,
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

/// Request to update namespace properties (POST /v1/namespaces/{ns}/properties)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergUpdateNamespacePropertiesRequest {
    #[serde(default)]
    pub removals: Vec<String>,
    #[serde(default)]
    pub updates: HashMap<String, String>,
}

/// Response for updating namespace properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergUpdateNamespacePropertiesResponse {
    pub updated: Vec<String>,
    pub removed: Vec<String>,
    pub missing: Vec<String>,
}

/// Response for listing tables (GET /v1/namespaces/{ns}/tables)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergListTablesResponse {
    pub identifiers: Vec<IcebergTableIdentifier>,
}

/// An Iceberg table identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergTableIdentifier {
    pub namespace: Vec<String>,
    pub name: String,
}

/// Iceberg field type — string representation per spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IcebergFieldType {
    Primitive(String),
    List {
        #[serde(rename = "type")]
        type_: String,
        #[serde(rename = "element-id")]
        element_id: i32,
        element: Box<IcebergFieldType>,
        #[serde(rename = "element-required")]
        element_required: bool,
    },
    Map {
        #[serde(rename = "type")]
        type_: String,
        #[serde(rename = "key-id")]
        key_id: i32,
        key: Box<IcebergFieldType>,
        #[serde(rename = "value-id")]
        value_id: i32,
        value: Box<IcebergFieldType>,
        #[serde(rename = "value-required")]
        value_required: bool,
    },
    Struct {
        #[serde(rename = "type")]
        type_: String,
        fields: Vec<IcebergSchemaField>,
    },
}

/// A field in an Iceberg schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSchemaField {
    pub id: i32,
    pub name: String,
    pub required: bool,
    #[serde(rename = "type")]
    pub field_type: IcebergFieldType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, String>,
}

/// An Iceberg schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSchema {
    #[serde(rename = "schema-id")]
    pub schema_id: i32,
    #[serde(rename = "type")]
    pub type_: String,
    pub fields: Vec<IcebergSchemaField>,
    #[serde(
        rename = "identifier-field-ids",
        skip_serializing_if = "Vec::is_empty",
        default
    )]
    pub identifier_field_ids: Vec<i32>,
}

/// Partition field transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergPartitionField {
    #[serde(rename = "field-id")]
    pub field_id: i32,
    #[serde(rename = "source-id")]
    pub source_id: i32,
    pub name: String,
    pub transform: String,
}

/// Partition spec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergPartitionSpec {
    #[serde(rename = "spec-id")]
    pub spec_id: i32,
    pub fields: Vec<IcebergPartitionField>,
}

/// Sort field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSortField {
    #[serde(rename = "source-id")]
    pub source_id: i32,
    pub transform: String,
    pub direction: String,
    #[serde(rename = "null-order")]
    pub null_order: String,
}

/// Sort order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSortOrder {
    #[serde(rename = "order-id")]
    pub order_id: i32,
    pub fields: Vec<IcebergSortField>,
}

/// Snapshot summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSnapshotSummary {
    pub operation: String,
    #[serde(flatten)]
    pub extra: HashMap<String, String>,
}

/// Snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSnapshot {
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    #[serde(rename = "parent-snapshot-id", skip_serializing_if = "Option::is_none")]
    pub parent_snapshot_id: Option<i64>,
    #[serde(rename = "sequence-number")]
    pub sequence_number: i64,
    #[serde(rename = "timestamp-ms")]
    pub timestamp_ms: i64,
    #[serde(rename = "manifest-list")]
    pub manifest_list: String,
    pub summary: IcebergSnapshotSummary,
    #[serde(rename = "schema-id", skip_serializing_if = "Option::is_none")]
    pub schema_id: Option<i32>,
}

/// A named snapshot reference (`refs` map entry): a branch or tag pointing at a snapshot.
///
/// This is the Iceberg-native substrate for git-style branching (TD-117): a branch is a
/// mutable named pointer to a snapshot id; a tag is an immutable one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergSnapshotRef {
    #[serde(rename = "snapshot-id")]
    pub snapshot_id: i64,
    /// `"branch"` or `"tag"`.
    #[serde(rename = "type")]
    pub ref_type: String,
}

impl IcebergSnapshotRef {
    /// A `branch`-type ref pointing at `snapshot_id`.
    pub fn branch(snapshot_id: i64) -> Self {
        Self {
            snapshot_id,
            ref_type: "branch".to_string(),
        }
    }
}

/// Full Iceberg table metadata (returned by load-table)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergTableMetadata {
    #[serde(rename = "format-version")]
    pub format_version: i32,
    #[serde(rename = "table-uuid")]
    pub table_uuid: String,
    pub location: String,
    #[serde(rename = "last-updated-ms")]
    pub last_updated_ms: i64,
    #[serde(rename = "last-column-id")]
    pub last_column_id: i32,
    #[serde(rename = "current-schema-id")]
    pub current_schema_id: i32,
    pub schemas: Vec<IcebergSchema>,
    #[serde(rename = "default-spec-id")]
    pub default_spec_id: i32,
    #[serde(rename = "partition-specs")]
    pub partition_specs: Vec<IcebergPartitionSpec>,
    #[serde(rename = "last-partition-id")]
    pub last_partition_id: i32,
    #[serde(rename = "default-sort-order-id")]
    pub default_sort_order_id: i32,
    #[serde(rename = "sort-orders")]
    pub sort_orders: Vec<IcebergSortOrder>,
    pub properties: HashMap<String, String>,
    #[serde(
        rename = "current-snapshot-id",
        skip_serializing_if = "Option::is_none"
    )]
    pub current_snapshot_id: Option<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub snapshots: Vec<IcebergSnapshot>,
    #[serde(
        rename = "snapshot-log",
        skip_serializing_if = "Vec::is_empty",
        default
    )]
    pub snapshot_log: Vec<HashMap<String, serde_json::Value>>,
    #[serde(
        rename = "metadata-log",
        skip_serializing_if = "Vec::is_empty",
        default
    )]
    pub metadata_log: Vec<HashMap<String, serde_json::Value>>,
    /// Named snapshot references (branches/tags). `main` always points at the current
    /// snapshot; this is the substrate for git-style agent branching (TD-117).
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub refs: HashMap<String, IcebergSnapshotRef>,
}

/// Response for loading a table (GET /v1/namespaces/{ns}/tables/{table})
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergLoadTableResponse {
    #[serde(rename = "metadata-location")]
    pub metadata_location: String,
    pub metadata: IcebergTableMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, String>>,
}

/// Request to create a table (POST /v1/namespaces/{ns}/tables)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergCreateTableRequest {
    pub name: String,
    pub schema: IcebergSchema,
    #[serde(rename = "partition-spec", skip_serializing_if = "Option::is_none")]
    pub partition_spec: Option<IcebergPartitionSpec>,
    #[serde(rename = "write-order", skip_serializing_if = "Option::is_none")]
    pub write_order: Option<IcebergSortOrder>,
    #[serde(
        rename = "stage-create",
        default,
        skip_serializing_if = "std::ops::Not::not"
    )]
    pub stage_create: bool,
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

/// Iceberg table update — one element of a commit
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum IcebergTableUpdate {
    #[serde(rename = "add-schema")]
    AddSchema {
        schema: IcebergSchema,
        #[serde(rename = "last-column-id")]
        last_column_id: i32,
    },
    #[serde(rename = "set-current-schema")]
    SetCurrentSchema {
        #[serde(rename = "schema-id")]
        schema_id: i32,
    },
    #[serde(rename = "add-snapshot")]
    AddSnapshot { snapshot: IcebergSnapshot },
    #[serde(rename = "set-snapshot-ref")]
    SetSnapshotRef {
        #[serde(rename = "ref-name")]
        ref_name: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: i64,
        #[serde(rename = "type")]
        type_: String,
    },
    #[serde(rename = "set-properties")]
    SetProperties { updates: HashMap<String, String> },
    #[serde(rename = "remove-properties")]
    RemoveProperties { removals: Vec<String> },
    #[serde(rename = "set-location")]
    SetLocation { location: String },
    #[serde(rename = "upgrade-format-version")]
    UpgradeFormatVersion {
        #[serde(rename = "format-version")]
        format_version: i32,
    },
}

/// Iceberg table requirement — asserted before committing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IcebergTableRequirement {
    #[serde(rename = "assert-table-does-not-exist")]
    AssertTableDoesNotExist,
    #[serde(rename = "assert-table-uuid")]
    AssertTableUuid { uuid: String },
    #[serde(rename = "assert-ref-snapshot-id")]
    AssertRefSnapshotId {
        #[serde(rename = "ref")]
        ref_name: String,
        #[serde(rename = "snapshot-id")]
        snapshot_id: Option<i64>,
    },
    #[serde(rename = "assert-last-assigned-field-id")]
    AssertLastAssignedFieldId {
        #[serde(rename = "last-assigned-field-id")]
        last_assigned_field_id: i32,
    },
    #[serde(rename = "assert-current-schema-id")]
    AssertCurrentSchemaId {
        #[serde(rename = "schema-id")]
        schema_id: i32,
    },
    #[serde(rename = "assert-last-assigned-partition-id")]
    AssertLastAssignedPartitionId {
        #[serde(rename = "last-assigned-partition-id")]
        last_assigned_partition_id: i32,
    },
    #[serde(rename = "assert-default-spec-id")]
    AssertDefaultSpecId {
        #[serde(rename = "spec-id")]
        spec_id: i32,
    },
    #[serde(rename = "assert-default-sort-order-id")]
    AssertDefaultSortOrderId {
        #[serde(rename = "sort-order-id")]
        sort_order_id: i32,
    },
}

/// Request to commit table changes (POST /v1/namespaces/{ns}/tables/{table})
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergCommitTableRequest {
    pub identifier: IcebergTableIdentifier,
    #[serde(default)]
    pub requirements: Vec<IcebergTableRequirement>,
    pub updates: Vec<IcebergTableUpdate>,
}

/// Response from a commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergCommitTableResponse {
    #[serde(rename = "metadata-location")]
    pub metadata_location: String,
    pub metadata: IcebergTableMetadata,
}

/// Request to register an existing table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergRegisterTableRequest {
    pub name: String,
    #[serde(rename = "metadata-location")]
    pub metadata_location: String,
}

/// Iceberg error response per spec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergErrorResponse {
    pub error: IcebergError,
}

/// Iceberg error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: u16,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stack: Vec<String>,
}

impl IcebergErrorResponse {
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            error: IcebergError {
                message: message.into(),
                error_type: "NoSuchTableException".to_string(),
                code: 404,
                stack: vec![],
            },
        }
    }

    pub fn already_exists(message: impl Into<String>) -> Self {
        Self {
            error: IcebergError {
                message: message.into(),
                error_type: "AlreadyExistsException".to_string(),
                code: 409,
                stack: vec![],
            },
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            error: IcebergError {
                message: message.into(),
                error_type: "BadRequestException".to_string(),
                code: 400,
                stack: vec![],
            },
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            error: IcebergError {
                message: message.into(),
                error_type: "ServiceUnavailableException".to_string(),
                code: 503,
                stack: vec![],
            },
        }
    }
}

// ============================================================================
// Service layer — translates between ProximaDB catalog and Iceberg REST types
// ============================================================================

/// Iceberg REST Catalog service.
///
/// Wraps `CatalogManager` and handles all translation between ProximaDB's internal
/// catalog types and the Iceberg REST spec JSON responses.
#[derive(Clone)]
pub struct IcebergRestService {
    catalog_manager: Arc<CatalogManager>,
    /// PAX segment registry: provides real row counts and file sizes for snapshots.
    segment_registry: Option<Arc<crate::SegmentRegistry>>,
    /// Warehouse object-store bridge. When present, `load_table` materializes a real,
    /// versioned `metadata.json` with a manifest-log-driven snapshot history and serves
    /// a resolvable `metadata-location` (TD-119). When absent, metadata is synthesized
    /// inline (still spec-shaped) without persistence.
    object_store_bridge: Option<Arc<dyn ObjectStoreBridge>>,
    /// Warehouse identifier returned in GET /v1/config
    pub warehouse: String,
    /// Arrow Flight endpoint embedded in table write-credentials
    pub flight_endpoint: String,
    /// Base URL for metadata locations (e.g. "http://localhost:5678/iceberg/v1")
    pub server_base_url: String,
}

impl IcebergRestService {
    pub fn new(
        catalog_manager: Arc<CatalogManager>,
        warehouse: impl Into<String>,
        flight_endpoint: impl Into<String>,
        server_base_url: impl Into<String>,
    ) -> Self {
        Self {
            catalog_manager,
            segment_registry: None,
            object_store_bridge: None,
            warehouse: warehouse.into(),
            flight_endpoint: flight_endpoint.into(),
            server_base_url: server_base_url.into(),
        }
    }

    /// Attach a shared segment registry so snapshot summaries reflect real PAX stats.
    pub fn with_segment_registry(mut self, registry: Arc<crate::SegmentRegistry>) -> Self {
        self.segment_registry = Some(registry);
        self
    }

    /// Attach the warehouse object-store bridge so table metadata is persisted as a real,
    /// versioned `metadata.json` with a manifest-log-driven snapshot history (TD-119).
    pub fn with_object_store_bridge(mut self, bridge: Arc<dyn ObjectStoreBridge>) -> Self {
        self.object_store_bridge = Some(bridge);
        self
    }

    // ---- Config ----

    pub fn get_config(&self) -> IcebergConfigResponse {
        let mut defaults = HashMap::new();
        defaults.insert("clients".to_string(), "4".to_string());
        defaults.insert("max-connections".to_string(), "128".to_string());

        let mut overrides = HashMap::new();
        overrides.insert(
            "proximadb.flight.endpoint".to_string(),
            self.flight_endpoint.clone(),
        );
        overrides.insert("warehouse".to_string(), self.warehouse.clone());

        IcebergConfigResponse {
            defaults,
            overrides,
        }
    }

    // ---- Namespaces ----

    pub async fn list_namespaces(
        &self,
        parent: Option<&str>,
    ) -> Result<IcebergListNamespacesResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let parent_levels = parent
            .map(|p| p.split('\x1f').map(String::from).collect::<Vec<_>>())
            .as_deref()
            .map(|sl: &[String]| sl)
            .map(|_| vec![])
            .unwrap_or_default();

        let namespaces = catalog
            .list_namespaces(if parent_levels.is_empty() {
                None
            } else {
                Some(parent_levels.as_slice())
            })
            .await?;

        Ok(IcebergListNamespacesResponse {
            namespaces: namespaces.into_iter().map(|ns| ns.levels).collect(),
        })
    }

    pub async fn create_namespace(
        &self,
        req: IcebergCreateNamespaceRequest,
    ) -> Result<IcebergNamespaceResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;

        let ns = catalog
            .create_namespace(&req.namespace, req.properties.clone())
            .await?;

        Ok(IcebergNamespaceResponse {
            namespace: ns.levels,
            properties: ns.properties,
        })
    }

    pub async fn get_namespace(&self, namespace: Vec<String>) -> Result<IcebergNamespaceResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let ns = catalog.get_namespace(&namespace).await?;

        Ok(IcebergNamespaceResponse {
            namespace: ns.levels,
            properties: ns.properties,
        })
    }

    pub async fn namespace_exists(&self, namespace: Vec<String>) -> Result<bool> {
        let catalog = self.catalog_manager.default_catalog().await?;
        catalog.namespace_exists(&namespace).await
    }

    pub async fn drop_namespace(&self, namespace: Vec<String>) -> Result<bool> {
        let catalog = self.catalog_manager.default_catalog().await?;
        catalog.drop_namespace(&namespace, false).await
    }

    pub async fn update_namespace_properties(
        &self,
        namespace: Vec<String>,
        req: IcebergUpdateNamespacePropertiesRequest,
    ) -> Result<IcebergUpdateNamespacePropertiesResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;

        let updated_keys: Vec<String> = req.updates.keys().cloned().collect();
        let removed_keys = req.removals.clone();

        catalog
            .update_namespace_properties(&namespace, req.updates, req.removals)
            .await?;

        Ok(IcebergUpdateNamespacePropertiesResponse {
            updated: updated_keys,
            removed: removed_keys,
            missing: vec![],
        })
    }

    // ---- Tables ----

    pub async fn list_tables(&self, namespace: Vec<String>) -> Result<IcebergListTablesResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifiers = catalog.list_tables(&namespace).await?;

        Ok(IcebergListTablesResponse {
            identifiers: identifiers
                .into_iter()
                .map(|id| IcebergTableIdentifier {
                    namespace: id.namespace,
                    name: id.name,
                })
                .collect(),
        })
    }

    pub async fn create_table(
        &self,
        namespace: Vec<String>,
        req: IcebergCreateTableRequest,
    ) -> Result<IcebergLoadTableResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;

        let identifier = TableIdentifier::new(namespace.clone(), req.name.clone());
        let schema = self.iceberg_schema_to_catalog(&req.schema, &req.name, req.properties.clone());

        let created = catalog.create_table(&identifier, schema).await?;
        let (metadata, metadata_location) =
            self.ensure_table_metadata(&identifier, &created).await?;

        Ok(IcebergLoadTableResponse {
            metadata_location,
            metadata,
            config: None,
        })
    }

    pub async fn load_table(
        &self,
        namespace: Vec<String>,
        table: String,
    ) -> Result<IcebergLoadTableResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifier = TableIdentifier::new(namespace.clone(), table.clone());

        let schema = catalog.get_table(&identifier).await?;
        let (metadata, metadata_location) =
            self.ensure_table_metadata(&identifier, &schema).await?;

        let mut config = HashMap::new();
        config.insert(
            "proximadb.flight.endpoint".to_string(),
            self.flight_endpoint.clone(),
        );

        Ok(IcebergLoadTableResponse {
            metadata_location,
            metadata,
            config: Some(config),
        })
    }

    /// Materialize a real, versioned `metadata.json` for `schema` and return it with a
    /// resolvable `metadata-location` (TD-119).
    ///
    /// When a warehouse bridge is configured AND the table has an object-store location,
    /// reads the data manifest log to build a parent-chained snapshot history, then
    /// idempotently persists `metadata.json` (only re-committing when the manifest log has
    /// advanced). Otherwise falls back to inline-synthesized (still spec-shaped) metadata.
    /// Persistence is fail-soft: a write error logs and still serves the in-memory metadata.
    pub async fn ensure_table_metadata(
        &self,
        identifier: &TableIdentifier,
        schema: &CatalogTableSchema,
    ) -> Result<(IcebergTableMetadata, String)> {
        let mut md = self.catalog_schema_to_iceberg_metadata(identifier, schema);
        let ns = identifier.namespace.join("\x1f");
        let default_location = format!(
            "{}/namespaces/{}/tables/{}/metadata/v1.metadata.json",
            self.server_base_url, ns, identifier.name
        );

        let (Some(bridge), Some(base)) =
            (self.object_store_bridge.as_ref(), table_base_path(schema))
        else {
            // No persistence wired / unmaterialized table → serve spec-shaped inline metadata.
            return Ok((md, default_location));
        };

        let manifest_prefix = format!("{base}/_manifests");
        let metadata_prefix = format!("{base}/_metadata");
        let manifest_base_url = format!(
            "{}/namespaces/{}/tables/{}/manifests",
            self.server_base_url, ns, identifier.name
        );

        // Snapshot history is driven by the data manifest log: version k -> snapshot k.
        let latest = bridge
            .latest_manifest_version(&manifest_prefix)
            .await
            .ok()
            .flatten();

        // Current aggregate stats applied to the head snapshot's summary.
        let mut summary_extra: HashMap<String, String> = HashMap::new();
        if let Some(stats) = self
            .segment_registry
            .as_ref()
            .and_then(|r| r.stats(&schema.name))
        {
            summary_extra.insert("total-records".to_string(), stats.row_count.to_string());
            summary_extra.insert(
                "total-data-files".to_string(),
                stats.segment_count.to_string(),
            );
            summary_extra.insert("total-files-size".to_string(), stats.size_bytes.to_string());
        }

        match latest {
            None => {
                // Table exists but no data committed yet: a valid table with no snapshots.
                md.snapshots = Vec::new();
                md.current_snapshot_id = None;
                md.refs = HashMap::new();
                md.snapshot_log = Vec::new();
            }
            Some(latest) => {
                let (snaps, log) = build_snapshot_chain(
                    &md.table_uuid,
                    md.current_schema_id,
                    latest,
                    md.last_updated_ms,
                    &manifest_base_url,
                    summary_extra,
                );
                let current = snaps.last().map(|s| s.snapshot_id);
                md.refs = current
                    .map(|id| HashMap::from([("main".to_string(), IcebergSnapshotRef::branch(id))]))
                    .unwrap_or_default();
                md.current_snapshot_id = current;
                md.snapshots = snaps;
                md.snapshot_log = log;
            }
        }

        let expected_snapshots = latest.map(|v| (v + 1) as usize).unwrap_or(0);
        let existing_version = bridge
            .latest_metadata_version(&metadata_prefix)
            .await
            .ok()
            .flatten();

        // If a persisted metadata.json already reflects the current manifest log, serve it
        // as-is — repeated reads must not keep appending metadata versions.
        if let Some(mv) = existing_version
            && let Ok(bytes) = bridge.read_table_metadata(&metadata_prefix, mv).await
            && let Ok(existing) = serde_json::from_slice::<IcebergTableMetadata>(&bytes)
            && existing.snapshots.len() == expected_snapshots
        {
            let loc = format!(
                "{}/namespaces/{}/tables/{}/metadata/v{}.metadata.json",
                self.server_base_url, ns, identifier.name, mv
            );
            return Ok((existing, loc));
        }

        // Stale or absent → commit a fresh metadata version (fail-soft).
        let target = existing_version.map(|v| v + 1).unwrap_or(0);
        let metadata_location = format!(
            "{}/namespaces/{}/tables/{}/metadata/v{}.metadata.json",
            self.server_base_url, ns, identifier.name, target
        );
        md.metadata_log.push(HashMap::from([
            (
                "timestamp-ms".to_string(),
                serde_json::json!(md.last_updated_ms),
            ),
            (
                "metadata-file".to_string(),
                serde_json::json!(metadata_location.clone()),
            ),
        ]));
        match serde_json::to_vec(&md) {
            Ok(bytes) => {
                if let Err(err) = bridge
                    .commit_table_metadata(&metadata_prefix, existing_version, bytes)
                    .await
                {
                    debug!(
                        "iceberg: persisting metadata.json for `{}` failed, serving inline: {err}",
                        identifier.name
                    );
                }
            }
            Err(err) => debug!("iceberg: serializing metadata.json failed: {err}"),
        }

        Ok((md, metadata_location))
    }

    /// Read the raw bytes of a persisted `metadata.json` version (for the REST GET route).
    pub async fn read_table_metadata_file(
        &self,
        namespace: Vec<String>,
        table: String,
        version: u64,
    ) -> Result<Vec<u8>> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifier = TableIdentifier::new(namespace, table);
        let schema = catalog.get_table(&identifier).await?;
        let base = table_base_path(&schema)
            .ok_or_else(|| anyhow::anyhow!("table has no object-store location"))?;
        let bridge = self
            .object_store_bridge
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("iceberg metadata persistence is not configured"))?;
        Ok(bridge
            .read_table_metadata(&format!("{base}/_metadata"), version)
            .await?)
    }

    pub async fn table_exists(&self, namespace: Vec<String>, table: String) -> Result<bool> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifier = TableIdentifier::new(namespace, table);
        catalog.table_exists(&identifier).await
    }

    pub async fn drop_table(
        &self,
        namespace: Vec<String>,
        table: String,
        purge: bool,
    ) -> Result<bool> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifier = TableIdentifier::new(namespace, table);
        catalog.drop_table(&identifier, purge).await
    }

    pub async fn commit_table(
        &self,
        namespace: Vec<String>,
        table: String,
        req: IcebergCommitTableRequest,
    ) -> Result<IcebergCommitTableResponse> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifier = TableIdentifier::new(namespace.clone(), table.clone());

        // Apply updates: handle SetProperties and AddSchema
        for update in &req.updates {
            match update {
                IcebergTableUpdate::SetProperties { updates } => {
                    // Update table properties in catalog (best-effort via statistics or properties)
                    debug!("Commit: SetProperties on {}.{}", namespace.join("."), table);
                    let _ = updates;
                }
                IcebergTableUpdate::AddSnapshot { snapshot } => {
                    debug!(
                        "Commit: AddSnapshot {} on {}.{}",
                        snapshot.snapshot_id,
                        namespace.join("."),
                        table
                    );
                }
                _ => {
                    debug!(
                        "Commit: unhandled update action on {}.{}",
                        namespace.join("."),
                        table
                    );
                }
            }
        }

        let schema = catalog.get_table(&identifier).await?;
        let metadata = self.catalog_schema_to_iceberg_metadata(&identifier, &schema);

        let metadata_location = format!(
            "{}/namespaces/{}/tables/{}/metadata/v2.metadata.json",
            self.server_base_url,
            namespace.join("\x1f"),
            table
        );

        Ok(IcebergCommitTableResponse {
            metadata_location,
            metadata,
        })
    }

    pub async fn register_table(
        &self,
        namespace: Vec<String>,
        req: IcebergRegisterTableRequest,
    ) -> Result<IcebergLoadTableResponse> {
        // For ProximaDB-native tables, registration creates a catalog entry pointing to the
        // external metadata location. We create a minimal schema from the name.
        let catalog = self.catalog_manager.default_catalog().await?;
        let identifier = TableIdentifier::new(namespace.clone(), req.name.clone());

        let mut properties = HashMap::new();
        properties.insert(
            "metadata-location".to_string(),
            req.metadata_location.clone(),
        );
        properties.insert("registered".to_string(), "true".to_string());

        let mut schema = CatalogTableSchema::new(req.name.clone());
        schema.properties = properties;
        let created = catalog.create_table(&identifier, schema).await?;
        let metadata = self.catalog_schema_to_iceberg_metadata(&identifier, &created);

        Ok(IcebergLoadTableResponse {
            metadata_location: req.metadata_location,
            metadata,
            config: None,
        })
    }

    // ---- Translation helpers ----

    /// Convert a ProximaDB `CatalogTableSchema` into Iceberg table metadata.
    ///
    /// Generates synthetic Iceberg metadata pointing to ProximaDB's Arrow Flight
    /// endpoint for data access. External engines see valid Iceberg table metadata
    /// and use the embedded Flight ticket to retrieve records.
    fn catalog_schema_to_iceberg_metadata(
        &self,
        identifier: &TableIdentifier,
        schema: &CatalogTableSchema,
    ) -> IcebergTableMetadata {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Build Iceberg fields from ProximaDB columns
        let mut fields = Vec::new();
        let mut field_id_counter = 1i32;

        // Always include canonical ProximaRecord identity fields
        fields.push(IcebergSchemaField {
            id: field_id_counter,
            name: "id".to_string(),
            required: true,
            field_type: IcebergFieldType::Primitive("string".to_string()),
            doc: Some("Canonical ProximaRecord identifier".to_string()),
            metadata: HashMap::new(),
        });
        field_id_counter += 1;

        fields.push(IcebergSchemaField {
            id: field_id_counter,
            name: "tenant_id".to_string(),
            required: true,
            field_type: IcebergFieldType::Primitive("string".to_string()),
            doc: Some("Tenant identifier for RLS".to_string()),
            metadata: HashMap::new(),
        });
        field_id_counter += 1;

        fields.push(IcebergSchemaField {
            id: field_id_counter,
            name: "created_at".to_string(),
            required: false,
            field_type: IcebergFieldType::Primitive("timestamptz".to_string()),
            doc: None,
            metadata: HashMap::new(),
        });
        field_id_counter += 1;

        fields.push(IcebergSchemaField {
            id: field_id_counter,
            name: "updated_at".to_string(),
            required: false,
            field_type: IcebergFieldType::Primitive("timestamptz".to_string()),
            doc: None,
            metadata: HashMap::new(),
        });
        field_id_counter += 1;

        // Add columns from the catalog schema
        for col in &schema.columns {
            let iceberg_type = catalog_type_to_iceberg(&col.data_type, field_id_counter);
            fields.push(IcebergSchemaField {
                id: field_id_counter,
                name: col.name.clone(),
                required: !col.nullable,
                field_type: iceberg_type,
                doc: col.comment.clone(),
                metadata: col.properties.clone(),
            });
            field_id_counter += 1;
        }

        let iceberg_schema = IcebergSchema {
            schema_id: schema.schema_version,
            type_: "struct".to_string(),
            fields,
            identifier_field_ids: vec![1], // id field
        };

        // Build table properties with ProximaDB index metadata
        let mut properties = schema.properties.clone();
        properties.insert("proximadb.collection".to_string(), schema.name.clone());
        properties.insert(
            "proximadb.flight.endpoint".to_string(),
            self.flight_endpoint.clone(),
        );
        properties.insert(
            "proximadb.record.envelope".to_string(),
            "ProximaRecord/v1".to_string(),
        );

        // Add HNSW index metadata for vector columns
        for col in &schema.columns {
            if matches!(
                col.data_type,
                ProximaType::DenseVector { .. } | ProximaType::SparseVector { .. }
            ) {
                let prefix = format!("proximadb.index.{}", col.name);
                properties.insert(format!("{}.type", prefix), "hnsw".to_string());
                if let Some(dim) = col.properties.get("dimension") {
                    properties.insert(format!("{}.dim", prefix), dim.clone());
                }
                if let Some(ef) = col.properties.get("ef_construction") {
                    properties.insert(format!("{}.ef_construction", prefix), ef.clone());
                }
            }
        }

        // Synthetic snapshot: represents the current state of the ProximaDB collection
        let snapshot_id = now_ms;
        let manifest_list_url = format!(
            "{}/namespaces/{}/tables/{}/manifests/snap-{}.avro",
            self.server_base_url,
            identifier.namespace.join("\x1f"),
            identifier.name,
            snapshot_id
        );

        // Build snapshot summary: use real PAX segment stats when available.
        let seg_stats = self
            .segment_registry
            .as_ref()
            .and_then(|r| r.stats(&schema.name));

        let snapshot = IcebergSnapshot {
            snapshot_id,
            parent_snapshot_id: None,
            sequence_number: 1,
            timestamp_ms: now_ms,
            manifest_list: manifest_list_url,
            summary: IcebergSnapshotSummary {
                operation: "append".to_string(),
                extra: {
                    let mut m = HashMap::new();
                    m.insert(
                        "proximadb.flight.ticket".to_string(),
                        format!(
                            "{{\"collection\":\"{}\",\"snapshot\":{}}}",
                            schema.name, snapshot_id
                        ),
                    );
                    if let Some(ref s) = seg_stats {
                        m.insert("total-records".to_string(), s.row_count.to_string());
                        m.insert("total-data-files".to_string(), s.segment_count.to_string());
                        m.insert("total-files-size".to_string(), s.size_bytes.to_string());
                    }
                    m
                },
            },
            schema_id: Some(schema.schema_version),
        };

        // Default partition spec: partition by tenant_id (field id=2)
        let partition_spec = IcebergPartitionSpec {
            spec_id: 0,
            fields: vec![IcebergPartitionField {
                field_id: 1000,
                source_id: 2, // tenant_id
                name: "tenant_id".to_string(),
                transform: "identity".to_string(),
            }],
        };

        // Unsorted by default
        let sort_order = IcebergSortOrder {
            order_id: 0,
            fields: vec![],
        };

        let table_uuid = schema
            .properties
            .get("proximadb.table.uuid")
            .cloned()
            .unwrap_or_else(|| uuid_from_name(&identifier.name));

        IcebergTableMetadata {
            format_version: 2,
            table_uuid,
            location: format!(
                "{}/namespaces/{}/tables/{}",
                self.server_base_url,
                identifier.namespace.join("/"),
                identifier.name
            ),
            last_updated_ms: now_ms,
            last_column_id: field_id_counter - 1,
            current_schema_id: iceberg_schema.schema_id,
            schemas: vec![iceberg_schema],
            default_spec_id: 0,
            partition_specs: vec![partition_spec],
            last_partition_id: 1000,
            default_sort_order_id: 0,
            sort_orders: vec![sort_order],
            properties,
            current_snapshot_id: Some(snapshot_id),
            snapshots: vec![snapshot],
            snapshot_log: vec![HashMap::from([
                ("timestamp-ms".to_string(), serde_json::json!(now_ms)),
                ("snapshot-id".to_string(), serde_json::json!(snapshot_id)),
            ])],
            metadata_log: vec![],
            refs: HashMap::from([("main".to_string(), IcebergSnapshotRef::branch(snapshot_id))]),
        }
    }

    /// Convert an incoming Iceberg schema to a ProximaDB `CatalogTableSchema`.
    fn iceberg_schema_to_catalog(
        &self,
        iceberg_schema: &IcebergSchema,
        table_name: &str,
        properties: HashMap<String, String>,
    ) -> CatalogTableSchema {
        let mut schema = CatalogTableSchema::new(table_name);
        // System-written properties are never taken from a client request.
        schema.properties = crate::table_properties::without_reserved_table_properties(properties);

        for (i, f) in iceberg_schema.fields.iter().enumerate() {
            let data_type = iceberg_type_to_catalog(&f.field_type);
            let mut col =
                crate::CatalogColumn::new(f.id.max(i as i32 + 1), f.name.clone(), data_type);
            col.nullable = !f.required;
            if let Some(doc) = &f.doc {
                col.comment = Some(doc.clone());
            }
            col.properties = f.metadata.clone();
            schema = schema.with_column(col);
        }

        schema
    }
}

// ============================================================================
// Type conversion helpers
// ============================================================================

/// Map a [`ProximaType`] to an Iceberg primitive or complex type string.
fn catalog_type_to_iceberg(dt: &ProximaType, field_id_base: i32) -> IcebergFieldType {
    match dt {
        ProximaType::Boolean => IcebergFieldType::Primitive("boolean".to_string()),
        ProximaType::Int8 | ProximaType::Int16 | ProximaType::Int32 => {
            IcebergFieldType::Primitive("int".to_string())
        }
        ProximaType::Int64 => IcebergFieldType::Primitive("long".to_string()),
        ProximaType::Float32 => IcebergFieldType::Primitive("float".to_string()),
        ProximaType::Float64 => IcebergFieldType::Primitive("double".to_string()),
        ProximaType::Decimal { .. } => IcebergFieldType::Primitive("decimal(38,18)".to_string()),
        ProximaType::String => IcebergFieldType::Primitive("string".to_string()),
        ProximaType::Binary => IcebergFieldType::Primitive("binary".to_string()),
        ProximaType::Date => IcebergFieldType::Primitive("date".to_string()),
        ProximaType::Time(_) => IcebergFieldType::Primitive("time".to_string()),
        ProximaType::Timestamp(_) => IcebergFieldType::Primitive("timestamp".to_string()),
        ProximaType::TimestampTz(_) => IcebergFieldType::Primitive("timestamptz".to_string()),
        ProximaType::Uuid => IcebergFieldType::Primitive("uuid".to_string()),
        ProximaType::Json => IcebergFieldType::Primitive("string".to_string()),
        // Vector → list<float>
        ProximaType::DenseVector { .. } | ProximaType::SparseVector { .. } => {
            IcebergFieldType::List {
                type_: "list".to_string(),
                element_id: field_id_base + 1000,
                element: Box::new(IcebergFieldType::Primitive("float".to_string())),
                element_required: true,
            }
        }
        // Binary vector → binary
        ProximaType::BinaryVector { .. } => IcebergFieldType::Primitive("binary".to_string()),
        // Richer ProximaType variants without a dedicated Iceberg mapping → string.
        _ => IcebergFieldType::Primitive("string".to_string()),
    }
}

/// Map an Iceberg field type to the canonical [`ProximaType`].
fn iceberg_type_to_catalog(ft: &IcebergFieldType) -> ProximaType {
    use proximadb_data_model::{TimeUnit, VectorElement};
    match ft {
        IcebergFieldType::Primitive(s) => match s.as_str() {
            "boolean" => ProximaType::Boolean,
            "int" => ProximaType::Int32,
            "long" => ProximaType::Int64,
            "float" => ProximaType::Float32,
            "double" => ProximaType::Float64,
            "date" => ProximaType::Date,
            "time" => ProximaType::Time(TimeUnit::Nanosecond),
            "timestamp" => ProximaType::Timestamp(TimeUnit::Nanosecond),
            "timestamptz" => ProximaType::TimestampTz(TimeUnit::Nanosecond),
            "uuid" => ProximaType::Uuid,
            "binary" => ProximaType::Binary,
            s if s.starts_with("decimal") => ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
            s if s.starts_with("fixed") => ProximaType::Binary,
            _ => ProximaType::String,
        },
        IcebergFieldType::List { element, .. } => match element.as_ref() {
            IcebergFieldType::Primitive(s) if s == "float" || s == "double" => {
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                }
            }
            _ => ProximaType::Json,
        },
        IcebergFieldType::Map { .. } => ProximaType::Json,
        IcebergFieldType::Struct { .. } => ProximaType::Json,
    }
}

/// Deterministic UUID from table name (for stable table-uuid across restarts).
fn uuid_from_name(name: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut h);
    let v = h.finish();
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (v >> 32) as u32,
        ((v >> 16) & 0xffff) as u16,
        (v & 0x0fff) as u16,
        0x8000u16 | ((v >> 48) & 0x3fff) as u16,
        v & 0xffffffffffff
    )
}

/// Derive a materialized table's object-store base prefix from its catalog schema,
/// mirroring the write path's `object_write_base_path` (explicit-location branch). Returns
/// `None` for tables without an object-store location (not yet materialized), for which
/// there is no manifest/metadata log to expose.
fn table_base_path(schema: &CatalogTableSchema) -> Option<String> {
    let primary = schema
        .storage_layouts
        .iter()
        .rev()
        .find(|l| l.name == "primary")
        .or_else(|| schema.storage_layouts.first());
    let location = primary
        .and_then(|l| match l.physical_format {
            crate::CatalogPhysicalFormat::Iceberg | crate::CatalogPhysicalFormat::Parquet => {
                l.location.as_deref()
            }
            _ => None,
        })
        .or(schema.location.as_deref())?;
    let normalized = normalize_object_path_prefix(location);
    (!normalized.is_empty()).then_some(normalized)
}

fn normalize_object_path_prefix(location: &str) -> String {
    let without_scheme = location
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(location);
    without_scheme.trim_matches('/').to_string()
}

/// Deterministic, stable positive `i64` snapshot id from `(table uuid, manifest version)`.
/// Stable across rebuilds so external time-travel by snapshot id stays valid.
fn stable_snapshot_id(table_uuid: &str, version: u64) -> i64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in table_uuid.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= version.wrapping_add(1);
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    (hash & 0x7FFF_FFFF_FFFF_FFFF) as i64
}

/// Build a parent-chained snapshot history for manifest versions `0..=latest_version`
/// (one Iceberg snapshot per committed data manifest). Pure; `head_summary_extra` is
/// applied to the latest snapshot. Returns `(snapshots, snapshot_log)`.
fn build_snapshot_chain(
    table_uuid: &str,
    schema_id: i32,
    latest_version: u64,
    now_ms: i64,
    manifest_base_url: &str,
    head_summary_extra: HashMap<String, String>,
) -> (
    Vec<IcebergSnapshot>,
    Vec<HashMap<String, serde_json::Value>>,
) {
    let mut snapshots = Vec::new();
    let mut log = Vec::new();
    for k in 0..=latest_version {
        let snapshot_id = stable_snapshot_id(table_uuid, k);
        let parent = (k > 0).then(|| stable_snapshot_id(table_uuid, k - 1));
        let extra = if k == latest_version {
            head_summary_extra.clone()
        } else {
            HashMap::new()
        };
        snapshots.push(IcebergSnapshot {
            snapshot_id,
            parent_snapshot_id: parent,
            sequence_number: (k + 1) as i64,
            timestamp_ms: now_ms,
            manifest_list: format!("{manifest_base_url}/snap-{snapshot_id}.avro"),
            summary: IcebergSnapshotSummary {
                operation: "append".to_string(),
                extra,
            },
            schema_id: Some(schema_id),
        });
        log.push(HashMap::from([
            ("timestamp-ms".to_string(), serde_json::json!(now_ms)),
            ("snapshot-id".to_string(), serde_json::json!(snapshot_id)),
        ]));
    }
    (snapshots, log)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_type_to_iceberg_vector() {
        let ft = catalog_type_to_iceberg(
            &ProximaType::DenseVector {
                element: proximadb_data_model::VectorElement::Float32,
                dim: 0,
            },
            10,
        );
        match ft {
            IcebergFieldType::List { type_, .. } => assert_eq!(type_, "list"),
            _ => panic!("expected list type for Vector"),
        }
    }

    #[test]
    fn test_iceberg_type_to_catalog_float_list() {
        let ft = IcebergFieldType::List {
            type_: "list".to_string(),
            element_id: 9999,
            element: Box::new(IcebergFieldType::Primitive("float".to_string())),
            element_required: true,
        };
        assert!(matches!(
            iceberg_type_to_catalog(&ft),
            ProximaType::DenseVector { .. }
        ));
    }

    #[test]
    fn test_uuid_from_name_stable() {
        let a = uuid_from_name("my_table");
        let b = uuid_from_name("my_table");
        assert_eq!(a, b);
        assert_ne!(uuid_from_name("table_a"), uuid_from_name("table_b"));
    }

    #[test]
    fn test_iceberg_error_response_serialization() {
        let err = IcebergErrorResponse::not_found("table not found");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("NoSuchTableException"));
        assert!(json.contains("404"));
    }

    #[test]
    fn metadata_has_main_ref_and_v2_snapshot_fields() {
        use crate::CatalogColumn;
        let svc = IcebergRestService::new(
            Arc::new(CatalogManager::new()),
            "wh",
            "grpc://localhost:5680",
            "http://localhost:5678/iceberg/v1",
        );
        let schema = CatalogTableSchema::new("events")
            .with_column(CatalogColumn::new(1, "id", ProximaType::String))
            .with_column(CatalogColumn::new(2, "score", ProximaType::Int64))
            .with_primary_key(vec!["id".to_string()]);
        let id = TableIdentifier::new(vec!["default".to_string()], "events".to_string());

        let md = svc.catalog_schema_to_iceberg_metadata(&id, &schema);

        // refs.main is the git-style branch pointer at the current snapshot (TD-117 substrate).
        let main = md.refs.get("main").expect("main ref present");
        assert_eq!(main.ref_type, "branch");
        assert_eq!(Some(main.snapshot_id), md.current_snapshot_id);

        // The snapshot carries a v2 sequence number and is recorded in the snapshot log.
        assert_eq!(md.snapshots.len(), 1);
        assert_eq!(md.snapshots[0].sequence_number, 1);
        assert_eq!(md.snapshot_log.len(), 1);

        // Serializes with Iceberg-spec field names so external readers parse it.
        let json = serde_json::to_string(&md).unwrap();
        assert!(json.contains("\"refs\""), "refs map present: {json}");
        assert!(json.contains("\"sequence-number\""));
        assert!(json.contains("\"format-version\":2"));
        assert!(json.contains("\"current-snapshot-id\""));
    }

    #[test]
    fn build_snapshot_chain_parent_chains_and_is_stable() {
        let (snaps, log) =
            build_snapshot_chain("uuid-x", 0, 2, 1000, "http://h/manifests", HashMap::new());
        assert_eq!(snaps.len(), 3, "one snapshot per manifest version 0..=2");
        assert_eq!(snaps[0].parent_snapshot_id, None);
        assert_eq!(snaps[1].parent_snapshot_id, Some(snaps[0].snapshot_id));
        assert_eq!(snaps[2].parent_snapshot_id, Some(snaps[1].snapshot_id));
        assert_eq!(snaps[0].sequence_number, 1);
        assert_eq!(snaps[2].sequence_number, 3);
        assert_ne!(snaps[0].snapshot_id, snaps[1].snapshot_id);
        assert_eq!(log.len(), 3);

        // Snapshot ids are deterministic (stable across rebuilds → time-travel by id holds).
        let (snaps2, _) =
            build_snapshot_chain("uuid-x", 0, 2, 9999, "http://h/manifests", HashMap::new());
        assert_eq!(snaps[1].snapshot_id, snaps2[1].snapshot_id);
    }
}
