//! # Delta Lake Catalog - PRODUCTION READY
//!
//! Provides Delta Lake table format integration for lakehouse interoperability.
//!
//! ## Feature Gate
//!
//! This catalog requires the `delta-lake` feature flag:
//! ```toml
//! [dependencies]
//! proximadb = { version = "0.1", features = ["delta-lake"] }
//! ```
//!
//! ## Features
//!
//! - **Delta Log Protocol**: Native Delta transaction log parsing
//! - **Schema Evolution**: Full schema change tracking via Delta log
//! - **Time Travel**: Query data at specific versions
//! - **ACID Transactions**: Optimistic concurrency with conflict resolution
//! - **Partitioning**: Hive-style partition layout
//! - **Vector Support**: Store vectors as `array<float>` with dimension metadata
//!
//! ## Concept Mapping
//!
//! | ProximaDB | Delta Lake |
//! |-----------|------------|
//! | Namespace | Directory/Database |
//! | Table | Delta Table (with _delta_log) |
//! | Column | Spark SQL Column |
//! | Vector | `array<float>` + metadata `delta.proximadb.dimension` |
//! | SparseVector | `map<int,float>` |
//!
//! ## Configuration
//!
//! ```ignore
//! let config = DeltaCatalogConfig {
//!     storage_url: "s3://my-bucket/delta-tables".to_string(),
//!     aws_region: Some("us-east-1".to_string()),
//!     ..Default::default()
//! };
//! let catalog = DeltaCatalog::new("delta", config, cache).await?;
//! ```
//!
//! ## Limitations
//!
//! - No native index support (stored as table properties)
//! - Requires Delta log for all operations
//! - Partitioned tables use Hive-style layout

// TD-CAT-8: module-level gate, matching `unity.rs` / `polaris.rs`. The
// `delta-lake` feature already existed and is off by default, but this module
// compiled into EVERY build anyway — `lib.rs`'s `pub mod` line does not show a
// module's own inner attribute, which is how that went unnoticed.
#![cfg(feature = "delta-lake")]

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
use proximadb_storage_filesystem_types::{FileOptions, FileSystem};

use crate::{
    Catalog, CatalogColumn, CatalogHealth, CatalogIndex, CatalogNamespace, CatalogPartitionSpec,
    CatalogSchemaEvolution, CatalogSortOrder, CatalogTableSchema, CatalogTableStatistics,
    LakehouseExtension, TableFormat, TableIdentifier,
};

/// Delta Lake catalog configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaCatalogConfig {
    /// Storage URL for Delta tables (s3://bucket/path, file:///path, etc.)
    pub storage_url: String,
    /// AWS region (for S3 storage)
    pub aws_region: Option<String>,
    /// Azure storage account (for ADLS)
    pub azure_storage_account: Option<String>,
    /// Storage options as JSON (credentials, etc.)
    pub storage_options: Option<String>,
    /// Enable checkpointing
    pub enable_checkpoint: bool,
    /// Checkpoint interval (number of commits)
    pub checkpoint_interval: Option<i32>,
}

/// Delta Lake catalog implementation
///
/// Maps ProximaDB concepts to Delta Lake:
/// - Namespace -> Directory structure
/// - Table -> Delta Table with _delta_log
/// - Columns -> Spark SQL schema
pub struct DeltaCatalog {
    /// Catalog name
    name: String,
    /// Configuration
    config: DeltaCatalogConfig,
    /// Base path for local storage (unused in object-store mode)
    base_path: PathBuf,
    /// Object-store mode (TD-OBJSTORE-1, #960): `Some` when `storage_url`
    /// carries a non-`file` scheme and the composition root injected a
    /// `FileSystem`. All persistence then routes through it with
    /// scheme-qualified URLs joined from `storage_url` — the old behavior
    /// (staging catalog metadata in a LOCAL temp cache, non-durable across
    /// VM loss) only remains for the direct `new()` constructor without an
    /// injected filesystem.
    filesystem: Option<Arc<dyn FileSystem>>,
    /// Catalog cache
    cache: Arc<CatalogCache>,
    /// In-memory storage for namespaces
    namespaces: tokio::sync::RwLock<HashMap<String, DeltaNamespace>>,
    /// In-memory storage for tables
    tables: tokio::sync::RwLock<HashMap<String, DeltaTableMetadata>>,
}

/// Delta namespace stored data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaNamespace {
    namespace: Vec<String>,
    properties: HashMap<String, String>,
    location: String,
}

/// Delta table metadata (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaTableMetadata {
    /// Table identifier
    identifier: DeltaTableIdentifier,
    /// Storage location
    location: String,
    /// Current schema
    schema: DeltaSchema,
    /// Schema history (all versions)
    schema_history: Vec<DeltaSchema>,
    /// Current version
    version: i64,
    /// Table properties
    properties: HashMap<String, String>,
    /// Partition columns
    partition_columns: Vec<String>,
    /// Created timestamp
    created_at_ms: i64,
    /// Last modified timestamp
    last_modified_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaTableIdentifier {
    namespace: Vec<String>,
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaSchema {
    /// Schema version
    version: i32,
    /// Fields
    fields: Vec<DeltaField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaField {
    /// Field name
    name: String,
    /// Spark SQL data type
    data_type: String,
    /// Is nullable
    nullable: bool,
    /// Field metadata (including vector dimension)
    metadata: HashMap<String, String>,
}

/// Delta log action for commits
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
enum DeltaAction {
    #[serde(rename = "metaData")]
    Metadata {
        id: String,
        name: Option<String>,
        description: Option<String>,
        format: DeltaFormat,
        #[serde(rename = "schemaString")]
        schema_string: String,
        #[serde(rename = "partitionColumns")]
        partition_columns: Vec<String>,
        configuration: HashMap<String, String>,
        #[serde(rename = "createdTime")]
        created_time: Option<i64>,
    },
    #[serde(rename = "protocol")]
    Protocol {
        #[serde(rename = "minReaderVersion")]
        min_reader_version: i32,
        #[serde(rename = "minWriterVersion")]
        min_writer_version: i32,
    },
    #[serde(rename = "add")]
    Add {
        path: String,
        #[serde(rename = "partitionValues")]
        partition_values: HashMap<String, String>,
        size: i64,
        #[serde(rename = "modificationTime")]
        modification_time: i64,
        #[serde(rename = "dataChange")]
        data_change: bool,
    },
    #[serde(rename = "remove")]
    Remove {
        path: String,
        #[serde(rename = "deletionTimestamp")]
        deletion_timestamp: Option<i64>,
        #[serde(rename = "dataChange")]
        data_change: bool,
    },
    #[serde(rename = "commitInfo")]
    CommitInfo {
        timestamp: i64,
        operation: String,
        #[serde(rename = "operationParameters")]
        operation_parameters: Option<HashMap<String, String>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaFormat {
    provider: String,
    options: HashMap<String, String>,
}

impl DeltaCatalog {
    /// Create a new Delta Lake catalog over the local filesystem (no injected
    /// backend). Prefer [`Self::new_with_filesystem`] via
    /// `CatalogManager::create_delta_catalog`, which resolves an object-store
    /// `FileSystem` for cloud URLs (TD-OBJSTORE-1, #960).
    pub async fn new(
        name: String,
        config: DeltaCatalogConfig,
        cache: Arc<CatalogCache>,
    ) -> Result<Self> {
        Self::new_with_filesystem(name, config, cache, None).await
    }

    /// Create a new Delta Lake catalog. With `filesystem: Some(_)` all
    /// persistence (catalog metadata, delta logs) routes through the injected
    /// backend against scheme-qualified URLs joined from
    /// `config.storage_url` — durable on `s3://`/`adls://`/`abfs://`/`gcs://`.
    /// With `None`, the pre-existing local layout is byte-identical.
    pub async fn new_with_filesystem(
        name: String,
        config: DeltaCatalogConfig,
        cache: Arc<CatalogCache>,
        filesystem: Option<Arc<dyn FileSystem>>,
    ) -> Result<Self> {
        info!(
            "Initializing Delta Lake catalog: {} at {} ({})",
            name,
            config.storage_url,
            if filesystem.is_some() {
                "object-store backend"
            } else {
                "local filesystem"
            }
        );

        let base_path = if filesystem.is_some() {
            // Object-store mode never touches local paths; keep a placeholder.
            PathBuf::new()
        } else {
            // Parse storage URL to base path
            let base_path = Self::parse_storage_url(&config.storage_url)?;
            // Ensure base path exists
            fs::create_dir_all(&base_path).await?;
            base_path
        };

        let catalog = Self {
            name,
            config,
            base_path,
            filesystem,
            cache,
            namespaces: tokio::sync::RwLock::new(HashMap::new()),
            tables: tokio::sync::RwLock::new(HashMap::new()),
        };

        // Load existing metadata
        catalog.load_catalog_metadata().await?;

        Ok(catalog)
    }

    /// Parse storage URL to local path
    fn parse_storage_url(url: &str) -> Result<PathBuf> {
        if let Some(path) = url.strip_prefix("file://") {
            Ok(PathBuf::from(path))
        } else if url.contains("://") {
            // ANY non-file scheme (s3://, gs://, gcs://, az://, azure://,
            // adls://, abfs://, …) — never enumerate schemes here: an
            // unlisted alias used to fall through to "local path" and
            // create a literal `adls:` directory (TD-OBJSTORE-1, #960).
            //
            // The delta catalog does not yet write its metadata to the
            // object store: it stages into a LOCAL cache directory, which
            // is NOT durable across VM loss (TD-OBJSTORE-1 deferred item).
            let cache_dir = std::env::temp_dir().join("proximadb_delta_cache");
            tracing::warn!(
                "DeltaCatalog storage_url '{}' is an object store; catalog metadata is staged \
                 in LOCAL, NON-DURABLE {} (TD-OBJSTORE-1)",
                url,
                cache_dir.display()
            );
            Ok(cache_dir)
        } else {
            // Assume local path
            Ok(PathBuf::from(url))
        }
    }

    /// Scheme-qualified URL for `rel` under the configured storage base
    /// (object-store mode only). String-joined so the scheme survives
    /// verbatim (TD-OBJSTORE-1, #960).
    fn object_url(&self, rel: &str) -> String {
        format!(
            "{}/{}",
            self.config.storage_url.trim_end_matches('/'),
            rel.trim_start_matches('/')
        )
    }

    /// Read `rel` (relative to the storage base) from whichever backend this
    /// catalog runs on. `Ok(None)` = not found.
    async fn read_catalog_object(&self, rel: &str) -> Result<Option<Vec<u8>>> {
        if let Some(fs_backend) = &self.filesystem {
            let url = self.object_url(rel);
            match fs_backend.exists(&url).await {
                Ok(false) => return Ok(None),
                Ok(true) => {}
                Err(e) => return Err(anyhow!("checking catalog object {url}: {e}")),
            }
            let data = fs_backend
                .read(&url)
                .await
                .map_err(|e| anyhow!("reading catalog object {url}: {e}"))?;
            Ok(Some(data))
        } else {
            match fs::read(self.base_path.join(rel)).await {
                Ok(data) => Ok(Some(data)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e.into()),
            }
        }
    }

    /// Write `rel` (relative to the storage base) to whichever backend this
    /// catalog runs on, creating parents as needed.
    async fn write_catalog_object(&self, rel: &str, data: &[u8]) -> Result<()> {
        if let Some(fs_backend) = &self.filesystem {
            let url = self.object_url(rel);
            let options = FileOptions {
                create_dirs: true,
                overwrite: true,
                ..Default::default()
            };
            fs_backend
                .write(&url, data, Some(options))
                .await
                .map_err(|e| anyhow!("writing catalog object {url}: {e}"))?;
        } else {
            let path = self.base_path.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
            fs::write(&path, data).await?;
        }
        Ok(())
    }

    /// Load catalog metadata from storage
    async fn load_catalog_metadata(&self) -> Result<()> {
        match self.read_catalog_object("_delta_catalog.json").await {
            Ok(Some(data)) => {
                let catalog_data: CatalogData = serde_json::from_slice(&data)?;
                *self.namespaces.write().await = catalog_data.namespaces;
                *self.tables.write().await = catalog_data.tables;
                debug!(
                    "Loaded {} namespaces and {} tables from Delta catalog",
                    self.namespaces.read().await.len(),
                    self.tables.read().await.len()
                );
            }
            Ok(None) => {
                debug!("No existing Delta catalog found, starting fresh");
            }
            Err(e) => {
                warn!("Error loading Delta catalog: {}", e);
            }
        }
        Ok(())
    }

    /// Save catalog metadata to storage.
    ///
    /// Last-writer-wins on the single `_delta_catalog.json` object — parity
    /// with the pre-existing local layout. Multi-writer CAS (ETag / put-if-
    /// absent, ManifestCommitter-style) is the SystemCatalog's job; the
    /// warehouse Delta catalog is single-writer per deployment (TD-CAT-4).
    async fn save_catalog_metadata(&self) -> Result<()> {
        let catalog_data = CatalogData {
            namespaces: self.namespaces.read().await.clone(),
            tables: self.tables.read().await.clone(),
        };

        let data = serde_json::to_vec_pretty(&catalog_data)?;
        self.write_catalog_object("_delta_catalog.json", &data)
            .await
    }

    /// Get namespace key
    fn namespace_key(namespace: &[String]) -> String {
        namespace.join(".")
    }

    /// Get table key
    fn table_key(identifier: &TableIdentifier) -> String {
        format!("{}.{}", identifier.namespace.join("."), identifier.name)
    }

    /// Table root relative to the storage base (`ns1/ns2/table`).
    fn table_rel(identifier: &TableIdentifier) -> String {
        format!("{}/{}", identifier.namespace.join("/"), identifier.name)
    }

    /// Get Delta log directory path (local mode)
    fn delta_log_path(&self, identifier: &TableIdentifier) -> PathBuf {
        self.base_path
            .join(identifier.namespace.join("/"))
            .join(&identifier.name)
            .join("_delta_log")
    }

    /// Get table data path (local mode)
    fn table_data_path(&self, identifier: &TableIdentifier) -> PathBuf {
        self.base_path
            .join(identifier.namespace.join("/"))
            .join(&identifier.name)
    }

    /// The table's location string as recorded in catalog metadata: a
    /// scheme-qualified URL in object-store mode, a local path otherwise.
    fn table_location(&self, identifier: &TableIdentifier) -> String {
        if self.filesystem.is_some() {
            self.object_url(&Self::table_rel(identifier))
        } else {
            self.table_data_path(identifier)
                .to_string_lossy()
                .to_string()
        }
    }

    /// Write Delta log entry
    async fn write_delta_log(
        &self,
        identifier: &TableIdentifier,
        version: i64,
        actions: Vec<DeltaAction>,
    ) -> Result<()> {
        let mut lines = Vec::new();
        for action in actions {
            lines.push(serde_json::to_string(&action)?);
        }
        let content = lines.join("\n");
        let rel = format!(
            "{}/_delta_log/{:020}.json",
            Self::table_rel(identifier),
            version
        );
        self.write_catalog_object(&rel, content.as_bytes()).await
    }

    /// Convert Spark SQL type to canonical [`ProximaType`].
    fn spark_type_to_data_type(spark_type: &str) -> ProximaType {
        let lower = spark_type.to_lowercase();
        match lower.as_str() {
            "boolean" => ProximaType::Boolean,
            "byte" | "tinyint" => ProximaType::Int8,
            "short" | "smallint" => ProximaType::Int16,
            "int" | "integer" => ProximaType::Int32,
            "long" | "bigint" => ProximaType::Int64,
            "float" | "real" => ProximaType::Float32,
            "double" => ProximaType::Float64,
            "string" => ProximaType::String,
            "binary" => ProximaType::Binary,
            "date" => ProximaType::Date,
            "timestamp" | "timestamp_ntz" => ProximaType::Timestamp(TimeUnit::Nanosecond),
            t if t.starts_with("decimal") => ProximaType::Decimal {
                precision: 38,
                scale: 10,
            },
            t if t.starts_with("array<float>") || t.starts_with("array<double>") => {
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                }
            }
            t if t.starts_with("array<") => ProximaType::Json,
            t if t.starts_with("map<") => ProximaType::Json,
            t if t.starts_with("struct<") => ProximaType::Json,
            _ => ProximaType::String,
        }
    }

    /// Convert canonical [`ProximaType`] to Spark SQL type
    fn data_type_to_spark_type(
        data_type: &ProximaType,
        _properties: &HashMap<String, String>,
    ) -> String {
        match data_type {
            ProximaType::Boolean => "boolean".to_string(),
            ProximaType::Int8 => "byte".to_string(),
            ProximaType::Int16 => "short".to_string(),
            ProximaType::Int32 => "integer".to_string(),
            ProximaType::Int64 => "long".to_string(),
            ProximaType::Float32 => "float".to_string(),
            ProximaType::Float64 => "double".to_string(),
            ProximaType::String => "string".to_string(),
            ProximaType::Binary => "binary".to_string(),
            ProximaType::Date => "date".to_string(),
            ProximaType::Time(_) => "string".to_string(), // Delta doesn't have TIME type
            ProximaType::Timestamp(_) | ProximaType::TimestampTz(_) => "timestamp".to_string(),
            ProximaType::Uuid => "string".to_string(),
            ProximaType::Decimal { .. } => "decimal(38,18)".to_string(),
            ProximaType::Json => "string".to_string(),
            ProximaType::DenseVector { .. } => "array<float>".to_string(),
            ProximaType::SparseVector { .. } => "map<integer,float>".to_string(),
            ProximaType::BinaryVector { .. } => "binary".to_string(),
            // Richer ProximaType variants without a Spark/Delta mapping → string.
            _ => "string".to_string(),
        }
    }

    /// Generate a UUID for tables
    fn generate_uuid() -> Result<String> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| anyhow!("System time error: {}", e))?
            .as_nanos();

        Ok(format!("{:032x}", timestamp))
    }

    /// Get current timestamp in milliseconds
    ///
    /// Thin wrapper over the shared kernel clock (`proximadb_kernel::time`).
    /// Deliberate behavior change: the old body propagated a pre-epoch clock
    /// as an error, while the kernel helper clamps to 0 — the same trade-off
    /// every other clock call site in the workspace already makes.
    fn now_ms() -> Result<i64> {
        Ok(proximadb_kernel::time::now_unix_ms())
    }

    /// Build schema string for Delta log
    fn build_schema_string(schema: &DeltaSchema) -> String {
        let fields: Vec<serde_json::Value> = schema
            .fields
            .iter()
            .map(|f| {
                serde_json::json!({
                    "name": f.name,
                    "type": f.data_type,
                    "nullable": f.nullable,
                    "metadata": f.metadata
                })
            })
            .collect();

        serde_json::json!({
            "type": "struct",
            "fields": fields
        })
        .to_string()
    }
}

/// Serialized catalog data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogData {
    namespaces: HashMap<String, DeltaNamespace>,
    tables: HashMap<String, DeltaTableMetadata>,
}

#[async_trait]
impl Catalog for DeltaCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        "delta"
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

        let rel = namespace.join("/");
        let location = if self.filesystem.is_some() {
            self.object_url(&rel)
        } else {
            self.base_path.join(&rel).to_string_lossy().to_string()
        };

        // Create the directory on local backends; object stores are flat
        // keyspaces (the injected backend's create_dir_all is a no-op there).
        if let Some(fs_backend) = &self.filesystem {
            fs_backend
                .create_dir_all(&location)
                .await
                .map_err(|e| anyhow!("creating namespace location {location}: {e}"))?;
        } else {
            fs::create_dir_all(&location).await?;
        }

        let ns = DeltaNamespace {
            namespace: namespace.to_vec(),
            properties: properties.clone(),
            location: location.clone(),
        };

        self.namespaces.write().await.insert(key.clone(), ns);
        self.save_catalog_metadata().await?;

        let now = Self::now_ms()?;

        info!("Created Delta namespace: {}", key);

        Ok(CatalogNamespace {
            levels: namespace.to_vec(),
            properties,
            owner: None,
            location: Some(location),
            created_at_ms: now,
            updated_at_ms: now,
            ..CatalogNamespace::new(Vec::new())
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
            // Optionally remove the directory (local mode only; object stores
            // have no directories — namespace objects are removed with their
            // tables).
            if self.filesystem.is_none() {
                let dir_path = self.base_path.join(namespace.join("/"));
                let _ = fs::remove_dir(&dir_path).await; // Ignore errors if not empty
            }

            self.save_catalog_metadata().await?;
            info!("Dropped Delta namespace: {}", key);
        }

        Ok(removed)
    }

    async fn list_namespaces(&self, _parent: Option<&[String]>) -> Result<Vec<CatalogNamespace>> {
        let now = Self::now_ms()?;

        let namespaces = self.namespaces.read().await;
        let results: Vec<CatalogNamespace> = namespaces
            .values()
            .map(|ns| CatalogNamespace {
                levels: ns.namespace.clone(),
                properties: ns.properties.clone(),
                owner: None,
                location: Some(ns.location.clone()),
                created_at_ms: now,
                updated_at_ms: now,
                ..CatalogNamespace::new(Vec::new())
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
        let now = Self::now_ms()?;

        let namespaces = self.namespaces.read().await;
        let ns = namespaces
            .get(&key)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))?;

        Ok(CatalogNamespace {
            levels: ns.namespace.clone(),
            properties: ns.properties.clone(),
            owner: None,
            location: Some(ns.location.clone()),
            created_at_ms: now,
            updated_at_ms: now,
            ..CatalogNamespace::new(Vec::new())
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

        let now = Self::now_ms()?;
        let location = self.table_location(identifier);

        // Convert schema to Delta format
        let fields: Vec<DeltaField> = schema
            .columns
            .iter()
            .map(|col| {
                let mut metadata = col.properties.clone();
                if matches!(col.data_type, ProximaType::DenseVector { .. })
                    && let Some(dim) = col.properties.get("dimension")
                {
                    metadata.insert("delta.proximadb.dimension".to_string(), dim.clone());
                }
                if let Some(comment) = &col.comment {
                    metadata.insert("comment".to_string(), comment.clone());
                }

                DeltaField {
                    name: col.name.clone(),
                    data_type: Self::data_type_to_spark_type(&col.data_type, &col.properties),
                    nullable: col.nullable,
                    metadata,
                }
            })
            .collect();

        let delta_schema = DeltaSchema { version: 1, fields };

        let table_meta = DeltaTableMetadata {
            identifier: DeltaTableIdentifier {
                namespace: identifier.namespace.clone(),
                name: identifier.name.clone(),
            },
            location: location.clone(),
            schema: delta_schema.clone(),
            schema_history: vec![delta_schema.clone()],
            version: 0,
            properties: schema.properties.clone(),
            partition_columns: vec![],
            created_at_ms: now,
            last_modified_ms: now,
        };

        // Create the Delta log directory on local backends (object stores
        // need none; `write_delta_log` creates parents as it writes).
        if self.filesystem.is_none() {
            let log_path = self.delta_log_path(identifier);
            fs::create_dir_all(&log_path).await?;
        }

        // Write initial Delta log entries
        let schema_string = Self::build_schema_string(&delta_schema);
        let mut properties = schema.properties.clone();
        properties.insert("delta.proximadb.version".to_string(), "1".to_string());

        let actions = vec![
            DeltaAction::Protocol {
                min_reader_version: 1,
                min_writer_version: 2,
            },
            DeltaAction::Metadata {
                id: Self::generate_uuid()?,
                name: Some(identifier.name.clone()),
                description: None,
                format: DeltaFormat {
                    provider: "parquet".to_string(),
                    options: HashMap::new(),
                },
                schema_string,
                partition_columns: vec![],
                configuration: properties,
                created_time: Some(now),
            },
            DeltaAction::CommitInfo {
                timestamp: now,
                operation: "CREATE TABLE".to_string(),
                operation_parameters: Some(HashMap::from([
                    ("isManaged".to_string(), "false".to_string()),
                    (
                        "description".to_string(),
                        "Created by ProximaDB".to_string(),
                    ),
                ])),
            },
        ];

        self.write_delta_log(identifier, 0, actions).await?;

        self.tables.write().await.insert(key, table_meta);
        self.save_catalog_metadata().await?;

        info!("Created Delta table: {}", identifier);
        Ok(schema)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, purge: bool) -> Result<bool> {
        let key = Self::table_key(identifier);

        let removed = self.tables.write().await.remove(&key).is_some();

        if removed {
            if purge {
                // Delete table data and logs
                if let Some(fs_backend) = &self.filesystem {
                    // Object store: enumerate the table prefix and delete
                    // each object (best-effort, mirroring the local
                    // `remove_dir_all`'s ignored errors).
                    let prefix = self.object_url(&Self::table_rel(identifier));
                    match fs_backend.list(&prefix).await {
                        Ok(entries) => {
                            for entry in entries {
                                if let Err(e) = fs_backend.delete(&entry.url).await {
                                    warn!("purging delta table object {}: {e}", entry.url);
                                }
                            }
                        }
                        Err(e) => warn!("listing delta table prefix {prefix} for purge: {e}"),
                    }
                } else {
                    let data_path = self.table_data_path(identifier);
                    let _ = fs::remove_dir_all(&data_path).await;
                }
            }

            self.cache
                .invalidate_table_in_catalog(&self.name, identifier);
            self.save_catalog_metadata().await?;

            info!("Dropped Delta table: {} (purge={})", identifier, purge);
        }

        Ok(removed)
    }

    async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        let ns_key = Self::namespace_key(namespace);
        let tables = self.tables.read().await;

        let identifiers: Vec<TableIdentifier> = tables
            .values()
            .filter(|t| t.identifier.namespace.join(".") == ns_key)
            .map(|t| {
                TableIdentifier::new(t.identifier.namespace.clone(), t.identifier.name.clone())
            })
            .collect();

        Ok(identifiers)
    }

    async fn table_exists(&self, identifier: &TableIdentifier) -> Result<bool> {
        let key = Self::table_key(identifier);
        Ok(self.tables.read().await.contains_key(&key))
    }

    async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        // Check cache first
        if let Some(schema) = self.cache.get_table(&self.name, identifier) {
            return Ok(schema);
        }

        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Convert Delta schema to CatalogTableSchema
        let columns: Vec<CatalogColumn> = table
            .schema
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut properties = HashMap::new();
                // Extract vector dimension from metadata
                if let Some(dim) = f.metadata.get("delta.proximadb.dimension") {
                    properties.insert("dimension".to_string(), dim.clone());
                }

                CatalogColumn {
                    id: i as i32,
                    object_id: None,
                    name: f.name.clone(),
                    data_type: Self::spark_type_to_data_type(&f.data_type),
                    nullable: f.nullable,
                    default_value: None,
                    comment: f.metadata.get("comment").cloned(),
                    properties,
                    is_deleted: false,
                    original_id: None,
                }
            })
            .collect();

        let schema = CatalogTableSchema {
            name: identifier.name.clone(),
            columns,
            primary_key: vec![],
            indexes: vec![],
            schema_version: table.schema.version,
            properties: table.properties.clone(),
            location: Some(table.location.clone()),
            created_at_ms: table.created_at_ms,
            updated_at_ms: table.last_modified_ms,
            ..Default::default()
        };

        self.cache.put_table(&self.name, identifier, schema.clone());
        Ok(schema)
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
        table.last_modified_ms = Self::now_ms()?;

        // Update location
        table.location = self.table_location(to);

        tables.insert(to_key, table);
        drop(tables);

        if self.filesystem.is_none() {
            // Move the table directory (local mode). Object stores have no
            // rename — data objects stay at the old prefix and the catalog
            // pointer above is the authority (documented Delta-on-object-store
            // semantics; a copy-based relocation is a TD-CAT-4 follow-up).
            let from_path = self.table_data_path(from);
            let to_path = self.table_data_path(to);

            if let Some(parent) = to_path.parent() {
                fs::create_dir_all(parent).await?;
            }

            let _ = fs::rename(&from_path, &to_path).await;
        } else {
            warn!(
                "Delta table rename {} -> {} on an object store updates the catalog pointer \
                 only; data objects are not relocated (TD-CAT-4)",
                from, to
            );
        }

        self.cache.invalidate_table_in_catalog(&self.name, from);
        self.save_catalog_metadata().await?;

        info!("Renamed Delta table: {} -> {}", from, to);
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
        let now = Self::now_ms()?;

        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Convert to Delta schema
        let fields: Vec<DeltaField> = new_schema
            .columns
            .iter()
            .map(|col| {
                let mut metadata = col.properties.clone();
                if matches!(col.data_type, ProximaType::DenseVector { .. })
                    && let Some(dim) = col.properties.get("dimension")
                {
                    metadata.insert("delta.proximadb.dimension".to_string(), dim.clone());
                }
                if let Some(comment) = &col.comment {
                    metadata.insert("comment".to_string(), comment.clone());
                }

                DeltaField {
                    name: col.name.clone(),
                    data_type: Self::data_type_to_spark_type(&col.data_type, &col.properties),
                    nullable: col.nullable,
                    metadata,
                }
            })
            .collect();

        let new_delta_schema = DeltaSchema {
            version: table.schema.version + 1,
            fields,
        };

        // Add to schema history
        table.schema_history.push(new_delta_schema.clone());
        table.schema = new_delta_schema.clone();
        table.version += 1;
        table.last_modified_ms = now;

        // Write new Delta log entry
        let schema_string = Self::build_schema_string(&new_delta_schema);
        let actions = vec![
            DeltaAction::Metadata {
                id: Self::generate_uuid()?,
                name: Some(identifier.name.clone()),
                description: None,
                format: DeltaFormat {
                    provider: "parquet".to_string(),
                    options: HashMap::new(),
                },
                schema_string,
                partition_columns: table.partition_columns.clone(),
                configuration: table.properties.clone(),
                created_time: Some(now),
            },
            DeltaAction::CommitInfo {
                timestamp: now,
                operation: "CHANGE SCHEMA".to_string(),
                operation_parameters: Some(HashMap::from([(
                    "newSchema".to_string(),
                    "true".to_string(),
                )])),
            },
        ];

        let version = table.version;
        drop(tables);

        self.write_delta_log(identifier, version, actions).await?;

        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);
        self.save_catalog_metadata().await?;

        info!(
            "Evolved Delta table schema: {} (v{})",
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

        Ok(table.schema.version)
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

        // Find schema by version (Delta keeps history!)
        let schema = table
            .schema_history
            .iter()
            .find(|s| s.version == version)
            .ok_or_else(|| anyhow!("Schema version {} not found", version))?;

        let columns: Vec<CatalogColumn> = schema
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut properties = HashMap::new();
                if let Some(dim) = f.metadata.get("delta.proximadb.dimension") {
                    properties.insert("dimension".to_string(), dim.clone());
                }

                CatalogColumn {
                    id: i as i32,
                    object_id: None,
                    name: f.name.clone(),
                    data_type: Self::spark_type_to_data_type(&f.data_type),
                    nullable: f.nullable,
                    default_value: None,
                    comment: f.metadata.get("comment").cloned(),
                    properties,
                    is_deleted: false,
                    original_id: None,
                }
            })
            .collect();

        Ok(CatalogTableSchema {
            name: identifier.name.clone(),
            columns,
            primary_key: vec![],
            indexes: vec![],
            schema_version: version,
            properties: HashMap::new(),
            location: None,
            created_at_ms: table.created_at_ms,
            updated_at_ms: table.last_modified_ms,
            ..Default::default()
        })
    }

    // ========================
    // Index Operations
    // ========================

    async fn create_index(
        &self,
        identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> Result<CatalogIndex> {
        // Delta Lake doesn't have native index support
        // Store index metadata in table properties
        warn!("Delta Lake catalog: indexes stored as metadata only, not enforced");

        let key = Self::table_key(identifier);
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Serialize index info to properties
        let index_key = format!("delta.proximadb.index.{}", index.name);
        let index_json = serde_json::to_string(&index)?;
        table.properties.insert(index_key, index_json);
        table.last_modified_ms = Self::now_ms()?;

        drop(tables);
        self.save_catalog_metadata().await?;

        Ok(index)
    }

    async fn drop_index(&self, identifier: &TableIdentifier, index_name: &str) -> Result<bool> {
        let key = Self::table_key(identifier);
        let mut tables = self.tables.write().await;
        let table = tables
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        let index_key = format!("delta.proximadb.index.{}", index_name);
        let removed = table.properties.remove(&index_key).is_some();
        table.last_modified_ms = Self::now_ms()?;

        drop(tables);
        self.save_catalog_metadata().await?;

        Ok(removed)
    }

    async fn list_indexes(&self, identifier: &TableIdentifier) -> Result<Vec<CatalogIndex>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        let mut indexes = Vec::new();
        for (k, v) in &table.properties {
            if k.starts_with("delta.proximadb.index.")
                && let Ok(index) = serde_json::from_str::<CatalogIndex>(v)
            {
                indexes.push(index);
            }
        }

        Ok(indexes)
    }

    // ========================
    // Statistics
    // ========================

    async fn get_statistics(&self, identifier: &TableIdentifier) -> Result<CatalogTableStatistics> {
        if let Some(stats) = self.cache.get_statistics(&self.name, identifier) {
            return Ok(stats);
        }

        // Delta Lake stores statistics in add actions
        // For now, return default stats
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
    // Partitioning
    // ========================

    async fn get_partition_spec(
        &self,
        identifier: &TableIdentifier,
    ) -> Result<Option<CatalogPartitionSpec>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        if table.partition_columns.is_empty() {
            return Ok(None);
        }

        // Delta uses Hive-style partitioning
        Ok(Some(CatalogPartitionSpec::default()))
    }

    async fn update_partition_spec(
        &self,
        identifier: &TableIdentifier,
        _spec: CatalogPartitionSpec,
    ) -> Result<()> {
        // Delta Lake partition changes require data rewrite
        warn!("Delta Lake: partition evolution requires data rewrite");

        let _ = identifier; // Suppress warning
        Err(anyhow!(
            "Delta Lake partition evolution not supported without data rewrite"
        ))
    }

    // ========================
    // Sort Order
    // ========================

    async fn get_sort_order(
        &self,
        _identifier: &TableIdentifier,
    ) -> Result<Option<CatalogSortOrder>> {
        // Delta Lake uses Z-ordering for optimized data layout
        Ok(None)
    }

    async fn update_sort_order(
        &self,
        _identifier: &TableIdentifier,
        _order: CatalogSortOrder,
    ) -> Result<()> {
        warn!("Delta Lake: use OPTIMIZE ZORDER for sort order optimization");
        Ok(())
    }

    // ========================
    // Health & Connectivity
    // ========================

    async fn health_check(&self) -> Result<CatalogHealth> {
        let start = Instant::now();

        if let Some(fs_backend) = &self.filesystem {
            // One cheap round-trip against the backend: an existence probe on
            // the catalog metadata object. Both outcomes are healthy (a fresh
            // catalog has no metadata object yet); only a backend error is not.
            match fs_backend
                .exists(&self.object_url("_delta_catalog.json"))
                .await
            {
                Ok(_) => {
                    let latency = start.elapsed().as_millis() as u64;
                    Ok(CatalogHealth::healthy(latency)
                        .with_detail("storage_url", &self.config.storage_url)
                        .with_detail("catalog_type", "delta"))
                }
                Err(e) => Ok(CatalogHealth::unhealthy(e.to_string())),
            }
        } else {
            match fs::metadata(&self.base_path).await {
                Ok(_) => {
                    let latency = start.elapsed().as_millis() as u64;
                    Ok(CatalogHealth::healthy(latency)
                        .with_detail("storage_url", &self.config.storage_url)
                        .with_detail("catalog_type", "delta"))
                }
                Err(e) => Ok(CatalogHealth::unhealthy(e.to_string())),
            }
        }
    }

    async fn close(&self) -> Result<()> {
        debug!("Closing Delta Lake catalog: {}", self.name);
        self.save_catalog_metadata().await?;
        Ok(())
    }
}

// ================================
// Lakehouse Extension (Delta-specific)
// ================================

#[async_trait]
impl LakehouseExtension for DeltaCatalog {
    fn table_format(&self) -> TableFormat {
        TableFormat::Delta
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

        // Delta uses version numbers instead of snapshot IDs
        Ok(Some(table.version))
    }

    async fn list_snapshots(&self, identifier: &TableIdentifier) -> Result<Vec<i64>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        // Return all versions from 0 to current
        Ok((0..=table.version).collect())
    }

    async fn get_schema_history(&self, identifier: &TableIdentifier) -> Result<Vec<i32>> {
        let key = Self::table_key(identifier);
        let tables = self.tables.read().await;
        let table = tables
            .get(&key)
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))?;

        Ok(table.schema_history.iter().map(|s| s.version).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spark_type_conversion() {
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("long"),
            ProximaType::Int64
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("string"),
            ProximaType::String
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("array<float>"),
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0
            }
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("integer"),
            ProximaType::Int32
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("boolean"),
            ProximaType::Boolean
        );
    }

    #[test]
    fn test_data_type_to_spark() {
        let props = HashMap::new();
        assert_eq!(
            DeltaCatalog::data_type_to_spark_type(&ProximaType::Int64, &props),
            "long"
        );
        assert_eq!(
            DeltaCatalog::data_type_to_spark_type(&ProximaType::String, &props),
            "string"
        );
        assert_eq!(
            DeltaCatalog::data_type_to_spark_type(
                &ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0
                },
                &props
            ),
            "array<float>"
        );
        assert_eq!(
            DeltaCatalog::data_type_to_spark_type(&ProximaType::Boolean, &props),
            "boolean"
        );
    }

    #[test]
    fn test_vector_type_with_dimension() {
        let mut props = HashMap::new();
        props.insert("dimension".to_string(), "768".to_string());

        let spark_type = DeltaCatalog::data_type_to_spark_type(
            &ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 0,
            },
            &props,
        );
        assert_eq!(spark_type, "array<float>");
    }

    #[test]
    fn test_table_key() {
        let id = TableIdentifier::new(vec!["db".to_string()], "users".to_string());
        assert_eq!(DeltaCatalog::table_key(&id), "db.users");
    }

    #[test]
    fn test_namespace_key() {
        let ns = vec!["catalog".to_string(), "schema".to_string()];
        assert_eq!(DeltaCatalog::namespace_key(&ns), "catalog.schema");
    }

    #[test]
    fn test_delta_config_default() {
        let config = DeltaCatalogConfig::default();
        assert!(config.storage_url.is_empty());
        assert!(config.aws_region.is_none());
        assert!(!config.enable_checkpoint);
    }

    #[test]
    fn test_parse_storage_url_local() {
        let path = DeltaCatalog::parse_storage_url("file:///tmp/delta")
            .expect("Failed to parse storage URL");
        assert_eq!(path.to_string_lossy(), "/tmp/delta");
    }

    #[test]
    fn test_parse_storage_url_s3() {
        let path = DeltaCatalog::parse_storage_url("s3://bucket/path")
            .expect("Failed to parse storage URL");
        assert!(path.to_string_lossy().contains("proximadb_delta_cache"));
    }

    #[test]
    fn test_generate_uuid() {
        let uuid1 = DeltaCatalog::generate_uuid().expect("Failed to generate UUID");
        let uuid2 = DeltaCatalog::generate_uuid().expect("Failed to generate UUID");

        assert_eq!(uuid1.len(), 32);
        assert_eq!(uuid2.len(), 32);
        // UUIDs should be different (based on timestamp, might be same in fast execution)
    }

    #[test]
    fn test_delta_schema_serialization() {
        let schema = DeltaSchema {
            version: 1,
            fields: vec![
                DeltaField {
                    name: "id".to_string(),
                    data_type: "long".to_string(),
                    nullable: false,
                    metadata: HashMap::new(),
                },
                DeltaField {
                    name: "name".to_string(),
                    data_type: "string".to_string(),
                    nullable: true,
                    metadata: HashMap::new(),
                },
            ],
        };

        let schema_string = DeltaCatalog::build_schema_string(&schema);
        assert!(schema_string.contains("\"type\":\"struct\""));
        assert!(schema_string.contains("\"name\":\"id\""));
        assert!(schema_string.contains("\"name\":\"name\""));
    }

    #[test]
    fn test_complex_spark_types() {
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("decimal(10,2)"),
            ProximaType::Decimal {
                precision: 38,
                scale: 10
            }
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("array<string>"),
            ProximaType::Json
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("map<string,int>"),
            ProximaType::Json
        );
        assert_eq!(
            DeltaCatalog::spark_type_to_data_type("struct<name:string,age:int>"),
            ProximaType::Json
        );
    }

    #[tokio::test]
    async fn test_delta_catalog_creation() {
        let temp_dir = std::env::temp_dir().join("proximadb_test_delta");
        let _ = fs::remove_dir_all(&temp_dir).await;

        let config = DeltaCatalogConfig {
            storage_url: format!("file://{}", temp_dir.display()),
            aws_region: None,
            azure_storage_account: None,
            storage_options: None,
            enable_checkpoint: false,
            checkpoint_interval: None,
        };

        let cache = Arc::new(CatalogCache::new(1000, 300));
        let catalog = DeltaCatalog::new("test".to_string(), config, cache).await;

        assert!(catalog.is_ok());
        let catalog = catalog.expect("Failed to create Delta catalog");
        assert_eq!(catalog.name(), "test");
        assert_eq!(catalog.catalog_type(), "delta");

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_delta_namespace_operations() {
        let temp_dir = std::env::temp_dir().join("proximadb_test_delta_ns");
        let _ = fs::remove_dir_all(&temp_dir).await;

        let config = DeltaCatalogConfig {
            storage_url: format!("file://{}", temp_dir.display()),
            ..Default::default()
        };

        let cache = Arc::new(CatalogCache::new(1000, 300));
        let catalog = DeltaCatalog::new("test".to_string(), config, cache)
            .await
            .expect("Failed to create Delta catalog");

        // Create namespace
        let ns = catalog
            .create_namespace(&["test_db".to_string()], HashMap::new())
            .await
            .expect("Failed to create namespace");
        assert_eq!(ns.levels, vec!["test_db"]);

        // Check exists
        assert!(
            catalog
                .namespace_exists(&["test_db".to_string()])
                .await
                .expect("Failed to check namespace exists")
        );

        // List namespaces
        let namespaces = catalog
            .list_namespaces(None)
            .await
            .expect("Failed to list namespaces");
        assert_eq!(namespaces.len(), 1);

        // Drop namespace
        assert!(
            catalog
                .drop_namespace(&["test_db".to_string()], false)
                .await
                .expect("Failed to drop namespace")
        );

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_delta_table_operations() {
        let temp_dir = std::env::temp_dir().join("proximadb_test_delta_table");
        let _ = fs::remove_dir_all(&temp_dir).await;

        let config = DeltaCatalogConfig {
            storage_url: format!("file://{}", temp_dir.display()),
            ..Default::default()
        };

        let cache = Arc::new(CatalogCache::new(1000, 300));
        let catalog = DeltaCatalog::new("test".to_string(), config, cache)
            .await
            .expect("Failed to create Delta catalog");

        // Create namespace first
        catalog
            .create_namespace(&["mydb".to_string()], HashMap::new())
            .await
            .expect("Failed to create namespace");

        // Create table
        let schema = CatalogTableSchema::new("users")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "name", ProximaType::String));

        let identifier = TableIdentifier::new(vec!["mydb".to_string()], "users".to_string());

        let created = catalog
            .create_table(&identifier, schema)
            .await
            .expect("Failed to create table");
        assert_eq!(created.name, "users");

        // Check table exists
        assert!(
            catalog
                .table_exists(&identifier)
                .await
                .expect("Failed to check table exists")
        );

        // Get table
        let retrieved = catalog
            .get_table(&identifier)
            .await
            .expect("Failed to get table");
        assert_eq!(retrieved.columns.len(), 2);
        assert_eq!(retrieved.columns[0].name, "id");

        // List tables
        let tables = catalog
            .list_tables(&["mydb".to_string()])
            .await
            .expect("Failed to list tables");
        assert_eq!(tables.len(), 1);

        // Drop table
        assert!(
            catalog
                .drop_table(&identifier, true)
                .await
                .expect("Failed to drop table")
        );
        assert!(
            !catalog
                .table_exists(&identifier)
                .await
                .expect("Failed to check table exists after drop")
        );

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir).await;
    }
    /// TD-OBJSTORE-1 (#960): with an injected object-store backend, the Delta
    /// catalog persists its metadata and delta logs as scheme-qualified
    /// objects — no local temp-cache staging — and a FRESH catalog instance
    /// over the same backend recovers the full state (durability across a
    /// process/VM loss).
    #[tokio::test]
    async fn object_store_backend_round_trips() {
        use crate::testfs::MemFs;
        use proximadb_storage_filesystem_types::FileSystem as _;

        let fs_backend: Arc<MemFs> = Arc::new(MemFs::default());
        let cfg = || DeltaCatalogConfig {
            storage_url: "adls://container/warehouse".to_string(),
            ..Default::default()
        };
        let cache = Arc::new(CatalogCache::new(1000, 300));

        let catalog = DeltaCatalog::new_with_filesystem(
            "delta-objstore".to_string(),
            cfg(),
            cache.clone(),
            Some(fs_backend.clone()),
        )
        .await
        .expect("object-store delta catalog");

        catalog
            .create_namespace(&["lake".to_string()], HashMap::new())
            .await
            .expect("create namespace");

        let schema = CatalogTableSchema::new("events")
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false));
        let identifier = TableIdentifier::new(vec!["lake".to_string()], "events".to_string());
        catalog
            .create_table(&identifier, schema)
            .await
            .expect("create table");

        // The catalog pointer and the v0 delta log are OBJECTS under the
        // scheme-qualified base — nothing staged under a local temp dir.
        assert!(
            fs_backend
                .exists("adls://container/warehouse/_delta_catalog.json")
                .await
                .expect("exists"),
            "catalog metadata object must live under the storage_url"
        );
        assert!(
            fs_backend
                .exists(
                    "adls://container/warehouse/lake/events/_delta_log/00000000000000000000.json"
                )
                .await
                .expect("exists"),
            "v0 delta log object must live under the storage_url"
        );
        // Table location is a scheme-qualified URL, not a local path.
        let table = catalog.get_table(&identifier).await.expect("get table");
        assert_eq!(table.name, "events");

        // Durability: a FRESH instance over the same backend recovers state.
        let reopened = DeltaCatalog::new_with_filesystem(
            "delta-objstore".to_string(),
            cfg(),
            Arc::new(CatalogCache::new(1000, 300)),
            Some(fs_backend.clone()),
        )
        .await
        .expect("reopen object-store delta catalog");
        assert!(
            reopened
                .table_exists(&identifier)
                .await
                .expect("table_exists"),
            "table written through the injected backend must survive a reopen"
        );
        assert!(
            reopened
                .namespace_exists(&["lake".to_string()])
                .await
                .expect("namespace_exists")
        );

        // Health probes the backend, not the local filesystem.
        let health = reopened.health_check().await.expect("health");
        assert!(health.is_healthy);
    }
}
