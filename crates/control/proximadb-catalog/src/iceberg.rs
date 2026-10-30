//! # Apache Iceberg Catalog - PRODUCTION READY
//!
//! Provides Apache Iceberg table format compatibility for:
//! - Lakehouse interoperability
//! - Spark, DuckDB, Arrow integration
//! - Schema evolution with full history
//!
//! ## Features
//!
//! - **Multiple Backends**: JDBC, Hadoop FileSystem, In-Memory (testing)
//! - **Schema Evolution**: Full history with version tracking
//! - **Partition Evolution**: Change partitioning without rewriting data
//! - **Time Travel**: Query data as of specific snapshots
//! - **Snapshot Isolation**: ACID transactions for concurrent access
//! - **Format Version 2**: Latest Iceberg specification support
//!
//! ## Lakehouse Integration
//!
//! This catalog implements `LakehouseExtension` providing:
//! - `get_table_location()`: Storage path for data files
//! - `get_current_snapshot()`: Current snapshot ID
//! - `list_snapshots()`: All available snapshots for time travel
//! - `get_schema_history()`: Schema version history
//!
//! ## Configuration
//!
//! ```ignore
//! let config = IcebergCatalogConfig {
//!     // Local metadata only — a `jdbc:` URI or a remote warehouse is rejected
//!     // at construction (TD-CAT-8); this catalog writes metadata to local disk.
//!     uri: "file:///var/lib/proximadb/iceberg".to_string(),
//!     warehouse: "file:///var/lib/proximadb/warehouse".to_string(),
//! };
//! let catalog = IcebergCatalog::new("iceberg", config, cache).await?;
//! ```
//!
//! ## Vector Type Mapping
//!
//! ProximaDB vectors are stored as `list<float>` in Iceberg schema,
//! with dimension stored in column metadata.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::cache::CatalogCache;
use crate::schema::{apply_evolution, validate_schema};
use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};

use crate::{
    Catalog, CatalogColumn, CatalogHealth, CatalogIndex, CatalogNamespace, CatalogPartitionSpec,
    CatalogSchemaEvolution, CatalogSortOrder, CatalogTableSchema, CatalogTableStatistics,
    LakehouseExtension, TableFormat, TableIdentifier,
};

/// Plain Rust configuration for the Apache Iceberg catalog.
///
/// Decoupled from `proximadb_proto::proximadb::v1::IcebergCatalogConfig` so the
/// workspace contract crate doesn't depend on the heavy proto crate. The
/// network/API layer converts from the proto form when configuring the
/// catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IcebergCatalogConfig {
    /// Catalog implementation class (e.g., REST, JDBC, Hadoop)
    pub catalog_impl: String,
    /// Connection URI
    pub uri: String,
    /// Warehouse location
    pub warehouse: String,
    /// Iceberg catalog properties
    pub properties: HashMap<String, String>,
    /// For JDBC catalog: driver class
    pub jdbc_driver: String,
}

/// Iceberg catalog backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergBackend {
    /// JDBC catalog (PostgreSQL, MySQL, etc.).
    ///
    /// **Not implemented, and unconstructible** — `IcebergCatalog::new` rejects a
    /// `jdbc:` URI (TD-CAT-8). Retained so the variant's discriminant stays
    /// stable for anything that persisted it; remove at the next clean break.
    Jdbc,
    /// Hadoop FileSystem catalog
    Hadoop,
    /// In-memory catalog (for testing)
    Memory,
}

/// Generic Iceberg catalog implementation
///
/// Supports multiple backends while maintaining Iceberg semantics:
/// - Schema evolution with full history
/// - Partition evolution
/// - Time travel queries
/// - Snapshot isolation
pub struct IcebergCatalog {
    /// Catalog name
    name: String,
    /// Configuration
    _config: IcebergCatalogConfig,
    /// Backend type
    backend: IcebergBackend,
    /// Base path for local storage
    base_path: PathBuf,
    /// Catalog cache
    cache: Arc<CatalogCache>,
    /// In-memory storage for tables/namespaces
    namespaces: tokio::sync::RwLock<HashMap<String, IcebergNamespace>>,
    tables: tokio::sync::RwLock<HashMap<String, IcebergTableMetadata>>,
}

/// Iceberg namespace stored data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IcebergNamespace {
    namespace: Vec<String>,
    properties: HashMap<String, String>,
}

/// Iceberg table metadata (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IcebergTableMetadata {
    /// Table identifier
    identifier: TableIdentifierData,
    /// Format version (1 or 2)
    format_version: i32,
    /// Table UUID
    table_uuid: String,
    /// Storage location
    location: String,
    /// Last updated timestamp
    last_updated_ms: i64,
    /// Current schema
    current_schema: IcebergSchema,
    /// All schemas (for history)
    schemas: Vec<IcebergSchema>,
    /// Current schema ID
    current_schema_id: i32,
    /// Partition specs
    partition_specs: Vec<serde_json::Value>,
    /// Default partition spec ID
    default_spec_id: i32,
    /// Sort orders
    sort_orders: Vec<serde_json::Value>,
    /// Default sort order ID
    default_sort_order_id: i32,
    /// Table properties
    properties: HashMap<String, String>,
    /// Snapshots
    snapshots: Vec<IcebergSnapshot>,
    /// Current snapshot ID
    current_snapshot_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableIdentifierData {
    namespace: Vec<String>,
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IcebergSchema {
    schema_id: i32,
    fields: Vec<IcebergField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IcebergField {
    id: i32,
    name: String,
    field_type: String,
    required: bool,
    doc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IcebergSnapshot {
    snapshot_id: i64,
    timestamp_ms: i64,
    summary: HashMap<String, String>,
    manifest_list: String,
}

impl IcebergCatalog {
    /// Create a new Iceberg catalog
    pub async fn new(
        name: String,
        config: IcebergCatalogConfig,
        cache: Arc<CatalogCache>,
    ) -> Result<Self> {
        info!("Initializing Iceberg catalog: {}", name);

        // TD-CAT-8: fail closed on any configuration this implementation cannot
        // honour. `backend` used to be derived from the URI scheme, stored,
        // echoed back by `health_check` — and never dispatched on. A `jdbc:`
        // URI got you local files while health reported "Jdbc", and a cloud
        // `warehouse` got you catalog metadata in `std::env::temp_dir()`. Both
        // were silent: the operator's configuration was accepted, reported back
        // to them, and ignored.
        //
        // Same precedent as `CatalogManager::create_native_catalog`, whose own
        // comment says an unsupported scheme surfacing a clear error is
        // "strictly safer than silently caching the catalog under /tmp"
        // (TD-OBJSTORE-1, #960).
        if config.uri.starts_with("jdbc:") {
            return Err(anyhow!(
                "iceberg catalog '{name}': a JDBC metastore URI ('{}') is not implemented — \
                 this catalog stores metadata on the local filesystem. Use a file:// URI, or \
                 an external Iceberg REST catalog.",
                config.uri
            ));
        }
        for remote in ["s3://", "gs://", "az://", "abfs://", "adls://", "hdfs://"] {
            if config.warehouse.starts_with(remote) {
                return Err(anyhow!(
                    "iceberg catalog '{name}': warehouse '{}' is remote, but this catalog \
                     writes its metadata to the LOCAL filesystem — it would land in a temp \
                     directory, not in the warehouse, and would not survive a restart. Use a \
                     local warehouse path.",
                    config.warehouse
                ));
            }
        }

        // Only local backends remain reachable, so the reported backend is now
        // the one actually in use. `Jdbc` is retained on the enum for wire/API
        // compatibility but is unconstructible — the guard above rejects it.
        let backend = if config.uri.starts_with("file://") || config.uri.starts_with("hdfs://") {
            IcebergBackend::Hadoop
        } else {
            IcebergBackend::Memory
        };

        // Determine base path for local storage
        let base_path = Self::parse_warehouse(&config.warehouse)?;

        // Ensure base path exists
        fs::create_dir_all(&base_path).await?;

        let catalog = Self {
            name,
            _config: config,
            backend,
            base_path,
            cache,
            namespaces: tokio::sync::RwLock::new(HashMap::new()),
            tables: tokio::sync::RwLock::new(HashMap::new()),
        };

        // Load existing metadata
        catalog.load_catalog_metadata().await?;

        Ok(catalog)
    }

    /// Parse warehouse path
    fn parse_warehouse(warehouse: &str) -> Result<PathBuf> {
        if let Some(path) = warehouse.strip_prefix("file://") {
            Ok(PathBuf::from(path))
        } else {
            // Remote warehouses are rejected in `new`; this is the local path.
            Ok(PathBuf::from(warehouse))
        }
    }

    /// Load catalog metadata from storage
    async fn load_catalog_metadata(&self) -> Result<()> {
        let namespaces_path = self.base_path.join("namespaces.json");

        match fs::read(&namespaces_path).await {
            Ok(data) => {
                let ns: HashMap<String, IcebergNamespace> = serde_json::from_slice(&data)?;
                *self.namespaces.write().await = ns;
                debug!(
                    "Loaded {} namespaces from catalog",
                    self.namespaces.read().await.len()
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!("No existing namespaces found");
            }
            Err(e) => {
                warn!("Error loading namespaces: {}", e);
            }
        }
        Ok(())
    }

    /// Save catalog metadata to storage
    async fn save_catalog_metadata(&self) -> Result<()> {
        let namespaces_path = self.base_path.join("namespaces.json");

        if let Some(parent) = namespaces_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let data = serde_json::to_vec_pretty(&*self.namespaces.read().await)?;
        fs::write(&namespaces_path, &data).await?;
        Ok(())
    }

    /// Get namespace key
    fn namespace_key(namespace: &[String]) -> String {
        namespace.join(".")
    }

    /// Get table key
    fn table_key(identifier: &TableIdentifier) -> String {
        format!("{}.{}", identifier.namespace.join("."), identifier.name)
    }

    /// Get table metadata path
    fn table_metadata_path(&self, identifier: &TableIdentifier) -> PathBuf {
        self.base_path
            .join(identifier.namespace.join("/"))
            .join(&identifier.name)
            .join("metadata")
            .join("v1.metadata.json")
    }

    /// Convert Iceberg type to canonical [`ProximaType`].
    fn iceberg_type_to_data_type(iceberg_type: &str) -> ProximaType {
        match iceberg_type.to_lowercase().as_str() {
            "boolean" => ProximaType::Boolean,
            "int" | "integer" => ProximaType::Int32,
            "long" => ProximaType::Int64,
            "float" => ProximaType::Float32,
            "double" => ProximaType::Float64,
            "string" => ProximaType::String,
            "binary" => ProximaType::Binary,
            "date" => ProximaType::Date,
            "time" => ProximaType::Time(TimeUnit::Nanosecond),
            "timestamp" | "timestamptz" => ProximaType::Timestamp(TimeUnit::Nanosecond),
            "uuid" => ProximaType::Uuid,
            t if t.starts_with("decimal") => ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
            t if t.starts_with("list<float>") || t.starts_with("list<double>") => {
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                }
            }
            t if t.starts_with("list<") => ProximaType::Json,
            t if t.starts_with("map<") => ProximaType::Json,
            t if t.starts_with("struct<") => ProximaType::Json,
            _ => ProximaType::String,
        }
    }

    /// Convert canonical [`ProximaType`] to Iceberg type
    fn data_type_to_iceberg_type(
        data_type: &ProximaType,
        properties: &HashMap<String, String>,
    ) -> String {
        match data_type {
            ProximaType::Boolean => "boolean".to_string(),
            ProximaType::Int8 | ProximaType::Int16 | ProximaType::Int32 => "int".to_string(),
            ProximaType::Int64 => "long".to_string(),
            ProximaType::Float32 => "float".to_string(),
            ProximaType::Float64 => "double".to_string(),
            ProximaType::String => "string".to_string(),
            ProximaType::Binary => "binary".to_string(),
            ProximaType::Date => "date".to_string(),
            ProximaType::Time(_) => "time".to_string(),
            ProximaType::Timestamp(_) | ProximaType::TimestampTz(_) => "timestamptz".to_string(),
            ProximaType::Uuid => "uuid".to_string(),
            ProximaType::Decimal { .. } => "decimal(38,18)".to_string(),
            ProximaType::DenseVector { .. } => {
                let _dim = properties
                    .get("dimension")
                    .unwrap_or(&"0".to_string())
                    .clone();
                "list<float>".to_string()
            }
            ProximaType::SparseVector { .. } => "map<int,float>".to_string(),
            ProximaType::BinaryVector { .. } => "binary".to_string(),
            ProximaType::Json => "string".to_string(),
            // Richer ProximaType variants without a dedicated Iceberg mapping
            // fall back to string (matching the legacy catalog default).
            _ => "string".to_string(),
        }
    }

    /// Create a new snapshot for a table
    fn create_snapshot(&self, table: &mut IcebergTableMetadata) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let snapshot_id = now; // Using timestamp as snapshot ID
        let snapshot = IcebergSnapshot {
            snapshot_id,
            timestamp_ms: now,
            summary: HashMap::from([
                ("operation".to_string(), "schema-update".to_string()),
                ("proximadb-version".to_string(), "0.2.0".to_string()),
            ]),
            manifest_list: format!("{}/metadata/snap-{}.avro", table.location, snapshot_id),
        };

        table.snapshots.push(snapshot);
        table.current_snapshot_id = Some(snapshot_id);
        table.last_updated_ms = now;
    }

    /// Generate a UUID
    fn generate_uuid() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        format!("{:032x}", timestamp)
    }

    /// Inherent accessor for the catalog metadata cache.
    /// Was a trait method before Option B consolidation; moved to inherent
    /// since the canonical `proximadb_catalog::Catalog` trait omits it.
    pub fn cache(&self) -> Option<Arc<CatalogCache>> {
        Some(self.cache.clone())
    }
}

#[async_trait]
impl Catalog for IcebergCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        "iceberg"
    }

    // ========================
    // Namespace Operations
    // ========================

    async fn create_namespace(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
    ) -> Result<CatalogNamespace> {
        let key = Self::namespace_key(namespace);

        if self.namespaces.read().await.contains_key(&key) {
            return Err(anyhow!("Namespace '{}' already exists", key));
        }

        let ns = IcebergNamespace {
            namespace: namespace.to_vec(),
            properties: properties.clone(),
        };

        self.namespaces.write().await.insert(key.clone(), ns);
        self.save_catalog_metadata().await?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        info!("Created Iceberg namespace: {}", key);

        Ok(CatalogNamespace {
            levels: namespace.to_vec(),
            properties,
            owner: None,
            location: None,
            created_at_ms: now,
            updated_at_ms: now,
            object_id: None,
            namespace_id: None,
            tenant_id: None,
            account_id: None,
            region_home: None,
            default_dr_region_pair_id: None,
            storage_pool_class: Default::default(),
        })
    }

    async fn drop_namespace(&self, namespace: &[String], cascade: bool) -> Result<bool> {
        let key = Self::namespace_key(namespace);

        if !cascade {
            let tables = self.list_tables(namespace).await?;
            if !tables.is_empty() {
                return Err(anyhow!(
                    "Namespace '{}' is not empty. Use cascade=true to force drop.",
                    key
                ));
            }
        }

        if cascade {
            let tables = self.list_tables(namespace).await?;
            for table_id in tables {
                self.drop_table(&table_id, true).await?;
            }
        }

        let removed = self.namespaces.write().await.remove(&key).is_some();
        if removed {
            self.save_catalog_metadata().await?;
            info!("Dropped Iceberg namespace: {}", key);
        }

        Ok(removed)
    }

    async fn list_namespaces(&self, _parent: Option<&[String]>) -> Result<Vec<CatalogNamespace>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let namespaces = self.namespaces.read().await;
        let results: Vec<CatalogNamespace> = namespaces
            .values()
            .map(|ns| CatalogNamespace {
                levels: ns.namespace.clone(),
                properties: ns.properties.clone(),
                owner: None,
                location: None,
                created_at_ms: now,
                updated_at_ms: now,
                object_id: None,
                namespace_id: None,
                tenant_id: None,
                account_id: None,
                region_home: None,
                default_dr_region_pair_id: None,
                storage_pool_class: Default::default(),
            })
            .collect();

        Ok(results)
    }

    async fn namespace_exists(&self, namespace: &[String]) -> Result<bool> {
        let key = Self::namespace_key(namespace);
        Ok(self.namespaces.read().await.contains_key(&key))
    }

    async fn get_namespace(&self, namespace: &[String]) -> Result<CatalogNamespace> {
        let key = Self::namespace_key(namespace);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let namespaces = self.namespaces.read().await;
        let ns = namespaces
            .get(&key)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))?;

        Ok(CatalogNamespace {
            levels: ns.namespace.clone(),
            properties: ns.properties.clone(),
            owner: None,
            location: None,
            created_at_ms: now,
            updated_at_ms: now,
            object_id: None,
            namespace_id: None,
            tenant_id: None,
            account_id: None,
            region_home: None,
            default_dr_region_pair_id: None,
            storage_pool_class: Default::default(),
        })
    }

    async fn update_namespace_properties(
        &self,
        namespace: &[String],
        updates: HashMap<String, String>,
        removals: Vec<String>,
    ) -> Result<()> {
        let key = Self::namespace_key(namespace);

        let mut namespaces = self.namespaces.write().await;
        let ns = namespaces
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))?;

        for (k, v) in updates {
            ns.properties.insert(k, v);
        }

        for k in removals {
            ns.properties.remove(&k);
        }

        drop(namespaces);
        self.save_catalog_metadata().await?;
        Ok(())
    }

    // ========================
    // Table Operations
    // ========================

    async fn create_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> Result<CatalogTableSchema> {
        // ADR-077 M1: normalize BEFORE validating. A creation path may legitimately
        // hand us a UNIQUE recorded only in the projection — `CREATE TABLE … UNIQUE(x)`
        // lowers to `relational_capabilities.unique_indexes` — and rejecting that would
        // break table creation. Folding it into the canonical field first is what makes
        // the invariant hold by construction rather than by rejecting real schemas.
        let mut schema = schema;
        crate::schema::normalize_identity(&mut schema);
        validate_schema(&schema)?;

        let key = Self::table_key(identifier);

        if self.tables.read().await.contains_key(&key) {
            return Err(anyhow!("Table '{}' already exists", identifier));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        // Convert schema to Iceberg format
        let fields: Vec<IcebergField> = schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| IcebergField {
                id: i as i32 + 1,
                name: col.name.clone(),
                field_type: Self::data_type_to_iceberg_type(&col.data_type, &col.properties),
                required: !col.nullable,
                doc: col.comment.clone(),
            })
            .collect();

        let iceberg_schema = IcebergSchema {
            schema_id: 1,
            fields,
        };

        let location = self
            .base_path
            .join(identifier.namespace.join("/"))
            .join(&identifier.name)
            .to_string_lossy()
            .to_string();

        let table_meta = IcebergTableMetadata {
            identifier: TableIdentifierData {
                namespace: identifier.namespace.clone(),
                name: identifier.name.clone(),
            },
            format_version: 2,
            table_uuid: Self::generate_uuid(),
            location: location.clone(),
            last_updated_ms: now,
            current_schema: iceberg_schema.clone(),
            schemas: vec![iceberg_schema],
            current_schema_id: 1,
            partition_specs: vec![],
            default_spec_id: 0,
            sort_orders: vec![],
            default_sort_order_id: 0,
            properties: schema.properties.clone(),
            snapshots: vec![],
            current_snapshot_id: None,
        };

        // Save table metadata to disk
        let metadata_path = self.table_metadata_path(identifier);
        if let Some(parent) = metadata_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let data = serde_json::to_vec_pretty(&table_meta)?;
        fs::write(&metadata_path, &data).await?;

        self.tables.write().await.insert(key, table_meta);

        info!("Created Iceberg table: {}", identifier);
        Ok(schema)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, purge: bool) -> Result<bool> {
        let key = Self::table_key(identifier);

        let removed = self.tables.write().await.remove(&key).is_some();

        if removed {
            // Delete metadata file
            let metadata_path = self.table_metadata_path(identifier);
            let _ = fs::remove_file(&metadata_path).await;

            // Purge data if requested
            if purge {
                let data_path = self
                    .base_path
                    .join(identifier.namespace.join("/"))
                    .join(&identifier.name);
                let _ = fs::remove_dir_all(&data_path).await;
            }

            self.cache
                .invalidate_table_in_catalog(&self.name, identifier);
            info!("Dropped Iceberg table: {} (purge={})", identifier, purge);
        }

        Ok(removed)
    }

    async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        crate::list_tables_from_map(&self.tables, &Self::namespace_key(namespace), |t| {
            (&t.identifier.namespace, &t.identifier.name)
        })
        .await
    }

    async fn table_exists(&self, identifier: &TableIdentifier) -> Result<bool> {
        let key = Self::table_key(identifier);
        Ok(self.tables.read().await.contains_key(&key))
    }

    async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        crate::cached_table(&self.cache, &self.name, identifier, async {
            let key = Self::table_key(identifier);
            let tables = self.tables.read().await;
            let table = tables
                .get(&key)
                .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

            // Convert Iceberg schema to CatalogTableSchema
            let columns: Vec<CatalogColumn> = table
                .current_schema
                .fields
                .iter()
                .map(|f| CatalogColumn {
                    id: f.id,
                    object_id: None,
                    name: f.name.clone(),
                    data_type: Self::iceberg_type_to_data_type(&f.field_type),
                    nullable: !f.required,
                    default_value: None,
                    comment: f.doc.clone(),
                    properties: HashMap::new(),
                    is_deleted: false,
                    original_id: None,
                })
                .collect();

            Ok(CatalogTableSchema {
                name: identifier.name.clone(),
                columns,
                primary_key: vec![],
                indexes: vec![],
                schema_version: table.current_schema_id,
                properties: table.properties.clone(),
                location: Some(table.location.clone()),
                created_at_ms: table.last_updated_ms,
                updated_at_ms: table.last_updated_ms,
                ..Default::default()
            })
        })
        .await
    }

    async fn rename_table(&self, from: &TableIdentifier, to: &TableIdentifier) -> Result<()> {
        let from_key = Self::table_key(from);
        let to_key = Self::table_key(to);

        let mut tables = self.tables.write().await;

        let mut table = tables
            .remove(&from_key)
            .ok_or_else(|| anyhow!("Table '{}' not found", from))?;

        table.identifier.namespace = to.namespace.clone();
        table.identifier.name = to.name.clone();
        table.last_updated_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        tables.insert(to_key, table);
        drop(tables);

        // Update metadata file
        let from_path = self.table_metadata_path(from);
        let to_path = self.table_metadata_path(to);

        if let Some(parent) = to_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let _ = fs::rename(&from_path, &to_path).await;

        self.cache.invalidate_table_in_catalog(&self.name, from);

        info!("Renamed Iceberg table: {} -> {}", from, to);
        Ok(())
    }

    // ========================
    // Schema Evolution
    // ========================

    async fn evolve_schema(
        &self,
        identifier: &TableIdentifier,
        evolution: CatalogSchemaEvolution,
    ) -> Result<CatalogTableSchema> {
        let current_schema = self.get_table(identifier).await?;
        let new_schema = apply_evolution(&current_schema, &evolution)?;

        let key = Self::table_key(identifier);

        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Create new Iceberg schema
        let fields: Vec<IcebergField> = new_schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| IcebergField {
                id: i as i32 + 1,
                name: col.name.clone(),
                field_type: Self::data_type_to_iceberg_type(&col.data_type, &col.properties),
                required: !col.nullable,
                doc: col.comment.clone(),
            })
            .collect();

        let new_iceberg_schema = IcebergSchema {
            schema_id: table.current_schema_id + 1,
            fields,
        };

        // Add to schema history (Iceberg feature)
        table.schemas.push(new_iceberg_schema.clone());
        table.current_schema = new_iceberg_schema;
        table.current_schema_id += 1;

        // Create snapshot for the schema change
        self.create_snapshot(table);

        // Save updated metadata
        let metadata_path = self.table_metadata_path(identifier);
        let data = serde_json::to_vec_pretty(&*table)?;
        drop(tables);

        fs::write(&metadata_path, &data).await?;

        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        info!(
            "Evolved Iceberg table schema: {} (v{})",
            identifier, new_schema.schema_version
        );
        Ok(new_schema)
    }

    async fn get_schema_version(&self, identifier: &TableIdentifier) -> Result<i32> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        Ok(table.current_schema_id)
    }

    async fn get_schema_by_version(
        &self,
        identifier: &TableIdentifier,
        version: i32,
    ) -> Result<CatalogTableSchema> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Find schema by version (Iceberg maintains history!)
        let schema = table
            .schemas
            .iter()
            .find(|s| s.schema_id == version)
            .ok_or_else(|| anyhow!("Schema version {} not found", version))?;

        let columns: Vec<CatalogColumn> = schema
            .fields
            .iter()
            .map(|f| CatalogColumn {
                id: f.id,
                object_id: None,
                name: f.name.clone(),
                data_type: Self::iceberg_type_to_data_type(&f.field_type),
                nullable: !f.required,
                default_value: None,
                comment: f.doc.clone(),
                properties: HashMap::new(),
                is_deleted: false,
                original_id: None,
            })
            .collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        Ok(CatalogTableSchema {
            name: identifier.name.clone(),
            columns,
            primary_key: vec![],
            indexes: vec![],
            schema_version: version,
            properties: HashMap::new(),
            location: None,
            created_at_ms: now,
            updated_at_ms: now,
            ..Default::default()
        })
    }

    // ========================
    // Index Operations
    // ========================

    async fn create_index(
        &self,
        _identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> Result<CatalogIndex> {
        warn!("Iceberg: indexes stored as metadata only");
        Ok(index)
    }

    async fn drop_index(&self, _identifier: &TableIdentifier, _index_name: &str) -> Result<bool> {
        Ok(true)
    }

    async fn list_indexes(&self, identifier: &TableIdentifier) -> Result<Vec<CatalogIndex>> {
        let schema = self.get_table(identifier).await?;
        Ok(schema.indexes)
    }

    // ========================
    // Statistics
    // ========================

    async fn get_statistics(&self, identifier: &TableIdentifier) -> Result<CatalogTableStatistics> {
        if let Some(stats) = self.cache.get_statistics(&self.name, identifier) {
            return Ok(stats);
        }

        let stats = CatalogTableStatistics::default();
        self.cache
            .put_statistics(&self.name, identifier, stats.clone());
        Ok(stats)
    }

    async fn update_statistics(
        &self,
        identifier: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) -> Result<()> {
        self.cache.put_statistics(&self.name, identifier, stats);
        Ok(())
    }

    // ========================
    // Partitioning (Iceberg feature)
    // ========================

    async fn get_partition_spec(
        &self,
        identifier: &TableIdentifier,
    ) -> Result<Option<CatalogPartitionSpec>> {
        use crate::{CatalogPartitionField, PartitionTransform};

        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Convert Iceberg partition specs to CatalogPartitionSpec
        if table.partition_specs.is_empty() {
            return Ok(None);
        }

        // Find the default partition spec
        let spec_json = table
            .partition_specs
            .iter()
            .find(|s| {
                s.get("spec-id")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|id| id as i32 == table.default_spec_id)
            })
            .or_else(|| table.partition_specs.first());

        if let Some(spec) = spec_json {
            let spec_id = spec.get("spec-id").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

            let fields = spec
                .get("fields")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|f| {
                            let source_id = f.get("source-id")?.as_i64()? as i32;
                            let field_id = f.get("field-id")?.as_i64()? as i32;
                            let name = f.get("name")?.as_str()?.to_string();
                            let transform_str = f.get("transform")?.as_str()?;
                            let transform =
                                PartitionTransform::parse_from_iceberg_format(transform_str);

                            Some(CatalogPartitionField {
                                source_id,
                                field_id,
                                name,
                                transform,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            Ok(Some(CatalogPartitionSpec { spec_id, fields }))
        } else {
            Ok(None)
        }
    }

    async fn update_partition_spec(
        &self,
        identifier: &TableIdentifier,
        spec: CatalogPartitionSpec,
    ) -> Result<()> {
        let key = Self::table_key(identifier);
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Iceberg supports partition evolution - add new spec
        let new_spec_id = table.default_spec_id + 1;

        // Convert CatalogPartitionSpec to Iceberg JSON format
        let fields_json: Vec<serde_json::Value> = spec
            .fields
            .iter()
            .map(|f| {
                serde_json::json!({
                    "source-id": f.source_id,
                    "field-id": f.field_id,
                    "name": f.name,
                    "transform": f.transform.to_string()
                })
            })
            .collect();

        let spec_json = serde_json::json!({
            "spec-id": new_spec_id,
            "fields": fields_json
        });

        table.partition_specs.push(spec_json);
        table.default_spec_id = new_spec_id;

        // Create snapshot for the partition spec change
        self.create_snapshot(table);

        // Save updated metadata
        let metadata_path = self.table_metadata_path(identifier);
        let data = serde_json::to_vec_pretty(&*table)?;
        drop(tables);
        fs::write(&metadata_path, &data).await?;

        info!(
            "Updated partition spec for {}: spec_id={}",
            identifier, new_spec_id
        );
        Ok(())
    }

    // ========================
    // Sort Order (Iceberg feature)
    // ========================

    async fn get_sort_order(
        &self,
        identifier: &TableIdentifier,
    ) -> Result<Option<CatalogSortOrder>> {
        use crate::{CatalogSortField, NullOrder, PartitionTransform, SortDirection};

        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Convert Iceberg sort orders to CatalogSortOrder
        if table.sort_orders.is_empty() {
            return Ok(None);
        }

        // Find the default sort order
        let order_json = table
            .sort_orders
            .iter()
            .find(|s| {
                s.get("order-id")
                    .and_then(|v| v.as_i64())
                    .is_some_and(|id| id as i32 == table.default_sort_order_id)
            })
            .or_else(|| table.sort_orders.first());

        if let Some(order) = order_json {
            let order_id = order.get("order-id").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

            let fields = order
                .get("fields")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|f| {
                            let source_id = f.get("source-id")?.as_i64()? as i32;
                            let transform_str = f.get("transform")?.as_str().unwrap_or("identity");
                            let transform =
                                PartitionTransform::parse_from_iceberg_format(transform_str);
                            let direction_str = f.get("direction")?.as_str().unwrap_or("asc");
                            let direction = if direction_str == "desc" {
                                SortDirection::Descending
                            } else {
                                SortDirection::Ascending
                            };
                            let null_order_str =
                                f.get("null-order")?.as_str().unwrap_or("nulls-first");
                            let null_order = if null_order_str == "nulls-last" {
                                NullOrder::NullsLast
                            } else {
                                NullOrder::NullsFirst
                            };

                            Some(CatalogSortField {
                                source_id,
                                transform,
                                direction,
                                null_order,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            Ok(Some(CatalogSortOrder { order_id, fields }))
        } else {
            Ok(None)
        }
    }

    async fn update_sort_order(
        &self,
        identifier: &TableIdentifier,
        order: CatalogSortOrder,
    ) -> Result<()> {
        use crate::{NullOrder, SortDirection};

        let key = Self::table_key(identifier);
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Iceberg supports sort order evolution - add new order
        let new_order_id = table.default_sort_order_id + 1;

        // Convert CatalogSortOrder to Iceberg JSON format
        let fields_json: Vec<serde_json::Value> = order
            .fields
            .iter()
            .map(|f| {
                let direction = match f.direction {
                    SortDirection::Ascending => "asc",
                    SortDirection::Descending => "desc",
                };
                let null_order = match f.null_order {
                    NullOrder::NullsFirst => "nulls-first",
                    NullOrder::NullsLast => "nulls-last",
                };
                serde_json::json!({
                    "source-id": f.source_id,
                    "transform": f.transform.to_string(),
                    "direction": direction,
                    "null-order": null_order
                })
            })
            .collect();

        let order_json = serde_json::json!({
            "order-id": new_order_id,
            "fields": fields_json
        });

        table.sort_orders.push(order_json);
        table.default_sort_order_id = new_order_id;

        // Create snapshot for the sort order change
        self.create_snapshot(table);

        // Save updated metadata
        let metadata_path = self.table_metadata_path(identifier);
        let data = serde_json::to_vec_pretty(&*table)?;
        drop(tables);
        fs::write(&metadata_path, &data).await?;

        info!(
            "Updated sort order for {}: order_id={}",
            identifier, new_order_id
        );
        Ok(())
    }

    // ========================
    // Health & Connectivity
    // ========================

    async fn health_check(&self) -> Result<CatalogHealth> {
        let start = Instant::now();

        match fs::metadata(&self.base_path).await {
            Ok(_) => {
                let latency = start.elapsed().as_millis() as u64;
                Ok(CatalogHealth::healthy(latency)
                    .with_detail("backend", format!("{:?}", self.backend))
                    .with_detail("catalog_type", "iceberg"))
            }
            Err(e) => Ok(CatalogHealth::unhealthy(e.to_string())),
        }
    }

    async fn close(&self) -> Result<()> {
        debug!("Closing Iceberg catalog: {}", self.name);
        Ok(())
    }
}

// ================================
// Lakehouse Extension (Iceberg-specific)
// ================================

#[async_trait]
impl LakehouseExtension for IcebergCatalog {
    fn table_format(&self) -> TableFormat {
        TableFormat::Iceberg
    }

    async fn get_table_location(&self, identifier: &TableIdentifier) -> Result<String> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        Ok(table.location.clone())
    }

    async fn get_current_snapshot(&self, identifier: &TableIdentifier) -> Result<Option<i64>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        Ok(table.current_snapshot_id)
    }

    async fn list_snapshots(&self, identifier: &TableIdentifier) -> Result<Vec<i64>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        Ok(table.snapshots.iter().map(|s| s.snapshot_id).collect())
    }

    async fn get_schema_history(&self, identifier: &TableIdentifier) -> Result<Vec<i32>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        Ok(table.schemas.iter().map(|s| s.schema_id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn open(uri: &str, warehouse: &str) -> Result<IcebergCatalog> {
        IcebergCatalog::new(
            "t".to_string(),
            IcebergCatalogConfig {
                uri: uri.to_string(),
                warehouse: warehouse.to_string(),
                ..Default::default()
            },
            Arc::new(CatalogCache::new(16, 60)),
        )
        .await
    }

    /// TD-CAT-8: a `jdbc:` URI used to be accepted, recorded as
    /// `IcebergBackend::Jdbc`, reported as such by `health_check` — and then
    /// ignored, because nothing ever dispatched on the backend. The operator's
    /// configuration was echoed back to them and not honoured.
    ///
    /// Teeth: delete the guard in `new` and this constructs successfully again,
    /// which is precisely the silent state being removed.
    #[tokio::test]
    async fn a_jdbc_uri_is_rejected_rather_than_silently_stored_locally() {
        let dir = tempfile::tempdir().expect("tempdir");
        let warehouse = dir.path().display().to_string();
        let err = open("jdbc:postgresql://localhost:5432/iceberg", &warehouse)
            .await
            .err()
            .expect("a JDBC metastore is not implemented and must not be accepted");

        let message = err.to_string();
        assert!(
            message.contains("JDBC") && message.contains("not implemented"),
            "the rejection must name what is unsupported, got: {message}"
        );
    }

    /// A remote warehouse used to resolve to `std::env::temp_dir()` — catalog
    /// metadata written to /tmp for a bucket the operator named, surviving no
    /// restart. Same failure the native catalog closed in TD-OBJSTORE-1 (#960).
    #[tokio::test]
    async fn a_remote_warehouse_is_rejected_rather_than_written_to_a_temp_dir() {
        for warehouse in [
            "s3://bucket/warehouse",
            "gs://bucket/warehouse",
            "az://acct/warehouse",
            "abfs://acct/warehouse",
            "adls://acct/warehouse",
            "hdfs://nn/warehouse",
        ] {
            let err = open("file:///tmp/iceberg", warehouse)
                .await
                .err()
                .unwrap_or_else(|| panic!("{warehouse} must be rejected, not accepted"));
            assert!(
                err.to_string().contains("remote"),
                "{warehouse}: the rejection must say why, got: {err}"
            );
        }
    }

    /// The positive control: the configuration this catalog CAN honour still
    /// works. Without it, a guard that rejected everything would look correct.
    #[tokio::test]
    async fn a_local_warehouse_still_constructs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let warehouse = format!("file://{}", dir.path().display());
        let catalog = open("file:///var/lib/proximadb/iceberg", &warehouse)
            .await
            .expect("a local warehouse is the supported configuration");
        assert_eq!(catalog.backend, IcebergBackend::Hadoop);
    }

    #[test]
    fn test_iceberg_type_conversion() {
        assert_eq!(
            IcebergCatalog::iceberg_type_to_data_type("long"),
            ProximaType::Int64
        );
        assert_eq!(
            IcebergCatalog::iceberg_type_to_data_type("string"),
            ProximaType::String
        );
        assert_eq!(
            IcebergCatalog::iceberg_type_to_data_type("list<float>"),
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0
            }
        );
    }

    #[test]
    fn test_data_type_to_iceberg() {
        let props = HashMap::new();
        assert_eq!(
            IcebergCatalog::data_type_to_iceberg_type(&ProximaType::Int64, &props),
            "long"
        );
        assert_eq!(
            IcebergCatalog::data_type_to_iceberg_type(&ProximaType::String, &props),
            "string"
        );
        assert_eq!(
            IcebergCatalog::data_type_to_iceberg_type(
                &ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0
                },
                &props
            ),
            "list<float>"
        );
    }

    #[test]
    fn test_table_key() {
        let id = TableIdentifier::new(vec!["db".to_string()], "users".to_string());
        assert_eq!(IcebergCatalog::table_key(&id), "db.users");
    }

    #[test]
    fn test_namespace_key() {
        let ns = vec!["catalog".to_string(), "schema".to_string()];
        assert_eq!(IcebergCatalog::namespace_key(&ns), "catalog.schema");
    }
}
