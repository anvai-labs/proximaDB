//! DDL (Data Definition Language) Service
//!
//! Provides SQL DDL operations that integrate with the catalog system:
//! - CREATE TABLE / CREATE COLLECTION
//! - DROP TABLE / DROP COLLECTION
//! - ALTER TABLE (schema evolution)
//! - CREATE INDEX / DROP INDEX
//! - CREATE NAMESPACE / DROP NAMESPACE

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use proximadb_catalog::{
    CatalogAuthorityMode, CatalogColumn, CatalogIndex, CatalogIndexType, CatalogPhysicalFormat,
    CatalogProjection, CatalogProjectionKind, CatalogSchemaEvolution, CatalogStorageLayout,
    CatalogStorageLayoutKind, CatalogStorageSpecialization, CatalogTableSchema,
    CatalogWorkloadProfile, ColumnConstraint, ProjectionFreshness, RelationalCapabilities,
    SchemaChange,
};
use proximadb_data_model::ProximaType;
use tracing::{info, warn};

use crate::catalog::CatalogManager;

/// DDL Statement types
#[derive(Debug, Clone)]
pub enum DdlStatement {
    /// CREATE TABLE [IF NOT EXISTS] table_name (columns...)
    CreateTable {
        /// Name of the table to create.
        table_name: String,
        /// Column definitions for the new table.
        columns: Vec<DdlColumnDefinition>,
        /// Table-level relational constraints.
        constraints: Vec<TableConstraint>,
        /// When `true`, silently succeeds if the table already exists.
        if_not_exists: bool,
        /// Additional table-level properties (e.g., storage options).
        properties: HashMap<String, String>,
    },
    /// DROP TABLE [IF EXISTS] table_name
    DropTable {
        /// Name of the table to drop.
        table_name: String,
        /// When `true`, silently succeeds if the table does not exist.
        if_exists: bool,
        /// When `true`, also removes all underlying data files.
        purge: bool,
    },
    /// ALTER TABLE table_name ADD COLUMN / DROP COLUMN / etc.
    AlterTable {
        /// Name of the table to alter.
        table_name: String,
        /// Schema changes to apply.
        changes: Vec<AlterTableChange>,
    },
    /// `ALTER TABLE <name> MATERIALIZE` — ProximaDB extension: publish the table's
    /// current rows as a Parquet snapshot on object storage and mark it
    /// Parquet-backed, so OLAP SELECTs over it route to DataFusion.
    MaterializeTable {
        /// Name of the table to materialize.
        name: String,
    },
    /// CREATE INDEX [IF NOT EXISTS] index_name ON table_name (columns)
    CreateIndex {
        /// Name of the index to create.
        index_name: String,
        /// Table on which the index is created.
        table_name: String,
        /// Columns included in the index.
        columns: Vec<String>,
        /// Index algorithm and its configuration.
        index_type: IndexType,
        /// When `true`, silently succeeds if the index already exists.
        if_not_exists: bool,
    },
    /// DROP INDEX [IF EXISTS] index_name ON table_name
    DropIndex {
        /// Name of the index to drop.
        index_name: String,
        /// Table that owns the index.
        table_name: String,
        /// When `true`, silently succeeds if the index does not exist.
        if_exists: bool,
    },
    /// CREATE NAMESPACE [IF NOT EXISTS] namespace_name
    CreateNamespace {
        /// Hierarchical namespace path components.
        namespace: Vec<String>,
        /// When `true`, silently succeeds if the namespace already exists.
        if_not_exists: bool,
        /// Additional namespace-level properties.
        properties: HashMap<String, String>,
    },
    /// DROP NAMESPACE [IF EXISTS] namespace_name
    DropNamespace {
        /// Hierarchical namespace path components.
        namespace: Vec<String>,
        /// When `true`, silently succeeds if the namespace does not exist.
        if_exists: bool,
        /// When `true`, also drops all tables contained in the namespace.
        cascade: bool,
    },
    /// CREATE COLLECTION (ProximaDB-specific)
    CreateCollection {
        /// Name of the vector collection to create.
        collection_name: String,
        /// Vector dimensionality for this collection.
        dimension: u32,
        /// Optional storage engine override (e.g., "HELIX", "VIPER").
        engine: Option<String>,
        /// When `true`, silently succeeds if the collection already exists.
        if_not_exists: bool,
        /// Additional collection-level properties.
        properties: HashMap<String, String>,
    },
    /// `CREATE RANK PROFILE [IF NOT EXISTS] name AS '<toml>'` (ProximaDB-specific).
    ///
    /// Lowers to [`crate::services::RankProfileStore::install`] with the
    /// caller-provided TOML body; the spec is re-validated + compiled into
    /// the live `RankServices` registry before the DDL response is sent.
    CreateRankProfile {
        /// Profile name (catalog key).
        name: String,
        /// Raw TOML body of the profile spec.
        spec_toml: String,
        /// When `true`, silently succeeds (without bumping version) if the
        /// profile already exists.
        if_not_exists: bool,
    },
    /// `DROP RANK PROFILE [IF EXISTS] name` (ProximaDB-specific).
    ///
    /// Lowers to [`crate::services::RankProfileStore::remove`].
    DropRankProfile {
        /// Profile name to remove.
        name: String,
        /// When `true`, silently succeeds if the profile does not exist.
        if_exists: bool,
    },
    /// `CREATE [OR REPLACE] FUNCTION name(params) RETURNS ty AS '<expr>'` (F5) — a
    /// SQL-expression-bodied scalar user function. The body is lowered to an engine-neutral
    /// `Expr` and registered in the shared function registry, so it runs on BOTH the Volcano
    /// and DataFusion engines without per-engine reimplementation.
    CreateFunction {
        /// Function name (registry key).
        name: String,
        /// Parameters in order: `(name, type)`. Parameter `i` is referenced in the body as
        /// column ordinal `i`.
        params: Vec<(String, ProximaType)>,
        /// Declared return type.
        return_ty: ProximaType,
        /// Raw SQL body — a scalar expression over the parameters.
        body: String,
        /// `CREATE OR REPLACE`: when `false`, fail if the name is already registered.
        or_replace: bool,
    },
}

/// Backwards-compat alias for [`DdlColumnDefinition`].
pub type ColumnDefinition = DdlColumnDefinition;

/// Column definition for CREATE TABLE
#[derive(Debug, Clone)]
pub struct DdlColumnDefinition {
    /// Column name.
    pub name: String,
    /// SQL data type of the column.
    pub data_type: SqlDataType,
    /// Whether the column accepts NULL values.
    pub nullable: bool,
    /// SQL expression used as the column's default value, if any.
    pub default_value: Option<String>,
    /// Optional human-readable description of the column.
    pub comment: Option<String>,
    /// Whether this column is part of the primary key.
    pub primary_key: bool,
}

/// SQL data types
#[derive(Debug, Clone)]
pub enum SqlDataType {
    /// SQL BOOLEAN type.
    Boolean,
    /// 8-bit signed integer (TINYINT).
    TinyInt,
    /// 16-bit signed integer (SMALLINT).
    SmallInt,
    /// 32-bit signed integer (INT).
    Int,
    /// 64-bit signed integer (BIGINT).
    BigInt,
    /// 32-bit IEEE 754 floating-point (FLOAT).
    Float,
    /// 64-bit IEEE 754 floating-point (DOUBLE).
    Double,
    /// Fixed-precision decimal number.
    Decimal {
        /// Total number of significant digits.
        precision: u32,
        /// Number of digits after the decimal point.
        scale: u32,
    },
    /// Variable-length character string.
    Varchar {
        /// Optional maximum number of characters.
        max_length: Option<u32>,
    },
    /// Unbounded text string.
    Text,
    /// Fixed-size binary data.
    Binary,
    /// Variable-size binary large object.
    Blob,
    /// Calendar date (year, month, day).
    Date,
    /// Time of day (hours, minutes, seconds).
    Time,
    /// Timestamp without time zone.
    Timestamp,
    /// Timestamp with time zone.
    TimestampTz,
    /// Universally unique identifier (UUID).
    Uuid,
    /// JSON document.
    Json,
    /// Binary JSON document.
    Jsonb,
    /// Dense vector type: VECTOR(dimension)
    Vector {
        /// Number of dimensions in the vector.
        dimension: u32,
    },
    /// Sparse vector: SPARSE_VECTOR(dimension)
    SparseVector {
        /// Number of dimensions in the sparse vector.
        dimension: u32,
    },
    /// Binary vector: BINARY_VECTOR(dimension)
    BinaryVector {
        /// Number of dimensions in the binary vector.
        dimension: u32,
    },
}

/// ALTER TABLE change types
#[derive(Debug, Clone)]
pub enum AlterTableChange {
    /// ADD COLUMN column_def
    AddColumn(DdlColumnDefinition),
    /// DROP COLUMN column_name
    DropColumn(String),
    /// RENAME COLUMN old_name TO new_name
    RenameColumn {
        /// Current name of the column.
        old_name: String,
        /// New name for the column.
        new_name: String,
    },
    /// ALTER COLUMN column_name SET DATA TYPE type
    ChangeType {
        /// Name of the column whose type is being changed.
        column_name: String,
        /// New SQL data type for the column.
        new_type: SqlDataType,
    },
    /// ALTER COLUMN column_name SET NOT NULL / DROP NOT NULL
    SetNullable {
        /// Name of the column to modify.
        column_name: String,
        /// New nullability setting (`true` = allow NULL, `false` = NOT NULL).
        nullable: bool,
    },
    /// ALTER COLUMN column_name SET DEFAULT value / DROP DEFAULT
    SetDefault {
        /// Name of the column to modify.
        column_name: String,
        /// New default value expression, or `None` to drop the default.
        default_value: Option<String>,
    },
    /// COMMENT ON COLUMN column_name IS 'comment'
    SetComment {
        /// Name of the column to annotate.
        column_name: String,
        /// Human-readable description to attach to the column.
        comment: String,
    },
    /// Move column position: FIRST or AFTER another_column
    MoveColumn {
        /// Name of the column to reposition.
        column_name: String,
        /// Target position within the column list.
        position: ColumnPosition,
    },
    /// ADD CONSTRAINT
    AddConstraint {
        /// Optional name for the new constraint.
        constraint_name: Option<String>,
        /// Constraint definition to add.
        constraint: TableConstraint,
    },
    /// DROP CONSTRAINT
    DropConstraint {
        /// Name of the constraint to remove.
        constraint_name: String,
    },
    /// PROMOTE PROPS KEY key TYPE type — promote a high-frequency props key to a typed column.
    /// The column is named `props__<key>` and stored in the next available ID ≥ 100.
    PromotePropsKey {
        key: String,
        column_type: SqlDataType,
        comment: Option<String>,
    },
    /// SET (key = 'value') — table-level option, e.g. `props_auto_promotion = 'enabled'`.
    SetTableOption { key: String, value: String },
}

/// Column position for ALTER TABLE ... MODIFY
#[derive(Debug, Clone)]
pub enum ColumnPosition {
    /// Move column to first position
    First,
    /// Move column after specified column
    After(String),
}

/// Table-level constraint for ALTER TABLE
#[derive(Debug, Clone)]
pub enum TableConstraint {
    /// UNIQUE constraint
    Unique {
        /// Columns that must be unique together.
        columns: Vec<String>,
    },
    /// CHECK constraint
    Check {
        /// SQL boolean expression that each row must satisfy.
        expression: String,
    },
    /// FOREIGN KEY constraint
    ForeignKey {
        /// Columns in this table that form the foreign key.
        columns: Vec<String>,
        /// Referenced table name.
        references_table: String,
        /// Columns in the referenced table.
        references_columns: Vec<String>,
        /// `ON DELETE` referential action, if specified.
        on_delete: Option<proximadb_catalog::ReferentialAction>,
        /// `ON UPDATE` referential action, if specified.
        on_update: Option<proximadb_catalog::ReferentialAction>,
    },
}

/// Index types for CREATE INDEX
#[derive(Debug, Clone)]
pub enum IndexType {
    /// B-tree index (default for scalar columns)
    BTree,
    /// Hash index
    Hash,
    /// Full-text search index
    FullText,
    /// GIN index for JSONB/document projection columns
    Gin,
    /// HNSW vector index (default for vector columns)
    Hnsw {
        /// Maximum number of bi-directional links per node (`M` parameter).
        m: Option<u32>,
        /// Size of the dynamic candidate list during construction.
        ef_construction: Option<u32>,
    },
    /// IVF vector index
    Ivf {
        /// Number of Voronoi cells (inverted lists).
        nlist: Option<u32>,
    },
    /// Product quantization
    Pq {
        /// Number of sub-quantizers.
        m: Option<u32>,
        /// Bits per sub-quantizer code.
        nbits: Option<u32>,
    },
}

/// Result of a DDL operation
#[derive(Debug, Clone)]
pub struct DdlResult {
    /// Was the operation successful?
    pub success: bool,
    /// Number of affected objects (tables, indexes, etc.)
    pub affected_count: u32,
    /// Message describing the result
    pub message: String,
    /// Warnings (if any)
    pub warnings: Vec<String>,
}

impl DdlResult {
    /// Create a successful DDL result
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            affected_count: 1,
            message: message.into(),
            warnings: Vec::new(),
        }
    }

    /// Create a result indicating an object already exists
    pub fn already_exists(object_type: &str, name: &str) -> Self {
        Self {
            success: true,
            affected_count: 0,
            message: format!("{} '{}' already exists", object_type, name),
            warnings: Vec::new(),
        }
    }

    /// Create a result indicating an object was not found
    pub fn not_found(object_type: &str, name: &str) -> Self {
        Self {
            success: true,
            affected_count: 0,
            message: format!("{} '{}' does not exist", object_type, name),
            warnings: Vec::new(),
        }
    }

    /// Add a warning to this result
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

/// DDL Service for executing DDL statements
/// Capability to materialize a table to a Parquet snapshot on object storage (the
/// warehouse publish op behind `ALTER TABLE … MATERIALIZE`).
///
/// Injected into [`DdlService`] so the DDL path can trigger materialization without
/// `DdlService` depending on `DmlService` or an object-store bridge directly. The
/// implementor (boot-wired with a configured warehouse store) scans the table's
/// current rows, writes the snapshot, and flips the catalog layout. Returns the
/// published object-store location.
#[async_trait::async_trait]
pub trait TableMaterializer: Send + Sync {
    /// Materialize `table_name`'s current rows; returns the published location.
    ///
    /// `tenant` is the request/connection tenant (TD-064). It scopes the table
    /// snapshot and the tenant-isolated object prefix; `None` means single-tenant
    /// / embedded mode. Threading it (vs the old hardcoded `None`) is what makes
    /// warehouse materialization tenant-isolated (TD-113).
    async fn materialize(&self, table_name: &str, tenant: Option<&str>) -> Result<String>;
}

pub struct DdlService {
    /// Catalog manager for metadata operations
    catalog_manager: Arc<CatalogManager>,
    /// Optional table materializer. Required for `ALTER TABLE … MATERIALIZE`;
    /// absent for paths without a configured warehouse object store (those get a
    /// clean error when the statement is issued).
    materializer: Option<Arc<dyn TableMaterializer>>,
    /// Optional rank-profile catalog. Required for `CREATE RANK PROFILE` /
    /// `DROP RANK PROFILE`; absent for embedded paths that never see those
    /// statements.
    rank_profile_store: Option<Arc<dyn crate::services::RankProfileStore>>,
    /// Optional rank-services singleton. When present, `CREATE RANK PROFILE`
    /// also compiles + installs the profile into the live registry so SQL
    /// `RERANK(...)` sees it immediately without waiting for the next boot.
    rank_services: Option<Arc<crate::network::rest::canonical::rank::RankServices>>,
    /// Optional durable function catalog (F5). When present, `CREATE FUNCTION`
    /// persists the definition so it is re-registered after a restart; absent
    /// for embedded/test paths (the in-process registration still happens).
    function_store: Option<Arc<dyn crate::services::FunctionStore>>,
    /// Partition-lease manager for per-collection write authority. When present
    /// (alongside a registry + pod id), collection/table-scoped DDL consults the
    /// lease before executing so writes are split-brain-free. `None` ⇒ inert.
    partition_lease_manager: Option<Arc<crate::cluster::partition_lease::PartitionLeaseManager>>,
    /// Primary-pod registry for write routing; the in-memory fallback consulted
    /// by the lease check (and the authority when no lease manager is attached).
    primary_pod_registry: Option<Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>>,
    /// This pod's identity for write-routing comparisons.
    self_pod_id: Option<String>,
    /// Canonical table-record store (shared with DML). When present, `DROP
    /// TABLE` also purges the table's record rows + WAL partition so a table
    /// recreated with the same name starts empty (SQL-row record oids are the
    /// PK text — without the purge the first re-INSERT hits INSERT_CONFLICT,
    /// or scans silently serve the dropped rows). Absent ⇒ warn-only legacy.
    record_store: Option<Arc<dyn crate::services::record_store::TableRecordStore>>,
}

impl DdlService {
    /// Create a new DDL service
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self {
            catalog_manager,
            materializer: None,
            rank_profile_store: None,
            rank_services: None,
            function_store: None,
            partition_lease_manager: None,
            primary_pod_registry: None,
            self_pod_id: None,
            record_store: None,
        }
    }

    /// Attach the table materializer. Required by `ALTER TABLE … MATERIALIZE`;
    /// callers that don't wire it get a clean error when that statement is issued.
    pub fn with_materializer(mut self, materializer: Arc<dyn TableMaterializer>) -> Self {
        self.materializer = Some(materializer);
        self
    }

    /// Attach the canonical table-record store (the SAME instance DML uses) so
    /// `DROP TABLE` can purge the table's record rows and WAL partition.
    pub fn with_record_store(
        mut self,
        record_store: Arc<dyn crate::services::record_store::TableRecordStore>,
    ) -> Self {
        self.record_store = Some(record_store);
        self
    }

    /// Attach the durable function catalog (F5). When present, every successful
    /// `CREATE FUNCTION` is persisted so it survives a restart (boot re-registers it).
    pub fn with_function_store(mut self, store: Arc<dyn crate::services::FunctionStore>) -> Self {
        self.function_store = Some(store);
        self
    }

    /// Attach the rank-profile catalog. Required by `CREATE RANK PROFILE` /
    /// `DROP RANK PROFILE`; callers that don't wire it will get a clean error
    /// message when those statements are issued.
    pub fn with_rank_profile_store(
        mut self,
        store: Arc<dyn crate::services::RankProfileStore>,
    ) -> Self {
        self.rank_profile_store = Some(store);
        self
    }

    /// Attach the live `RankServices` registry. When present, every
    /// successful `CREATE RANK PROFILE` is compiled + installed into the
    /// in-process registry so SQL `RERANK(...)` picks up the change without
    /// requiring a server restart.
    pub fn with_rank_services(
        mut self,
        services: Arc<crate::network::rest::canonical::rank::RankServices>,
    ) -> Self {
        self.rank_services = Some(services);
        self
    }

    /// Attach the partition-lease manager for per-collection write authority.
    /// When the lease system is on (a future registered gate; none exists yet), the
    /// `SharedServices` composition root attaches this so collection/table-scoped
    /// DDL consults the lease before executing. Inert by default.
    pub fn with_partition_lease_manager(
        mut self,
        manager: Arc<crate::cluster::partition_lease::PartitionLeaseManager>,
    ) -> Self {
        self.partition_lease_manager = Some(manager);
        self
    }

    /// Wire the complete write-lease authority onto an already-assembled DDL
    /// service. Pgwire builds rank/materializer decorators per connection, so a
    /// mutating setter avoids rebuilding the service and dropping those handles.
    pub fn set_write_lease_authority(
        &mut self,
        manager: Arc<crate::cluster::partition_lease::PartitionLeaseManager>,
        registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
        self_pod_id: String,
    ) {
        self.partition_lease_manager = Some(manager);
        self.primary_pod_registry = Some(registry);
        self.self_pod_id = Some(self_pod_id);
    }

    /// Attach the primary-pod registry used for write-routing decisions.
    pub fn with_primary_pod_registry(
        mut self,
        registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
    ) -> Self {
        self.primary_pod_registry = Some(registry);
        self
    }

    /// Attach this pod's identity for write-routing comparisons.
    pub fn with_self_pod_id(mut self, pod_id: String) -> Self {
        self.self_pod_id = Some(pod_id);
        self
    }

    /// Consult the write lease before a collection/table-scoped DDL.
    ///
    /// Returns `Ok(())` when the write is allowed, or when lease enforcement is
    /// not wired (registry or pod-id is `None`). The lease manager is optional:
    /// present ⇒ lease acquire/renew; absent ⇒ in-memory registry lookup only.
    /// Returns `Err` naming the target pod when another pod holds authority.
    ///
    /// `SystemCatalog` generation-fences catalog snapshot publication, but it
    /// does not fence external side effects such as a Parquet MATERIALIZE write.
    /// The renewable partition lease is therefore the authority for these
    /// table-scoped operations; the post-operation check detects displacement
    /// before success/catalog-cache publication.
    async fn check_lease_for_write(&self, tenant: Option<&str>, collection_id: &str) -> Result<()> {
        use crate::cluster::primary_pod_registry::{
            WriteRoutingDecision, consult_for_write_leased,
        };
        // Inert unless a registry + pod id are wired (the essentials for any
        // routing decision). The lease manager is optional: present ⇒ lease
        // acquire/renew; absent ⇒ in-memory registry lookup only.
        let (registry, pod_id) = match (&self.primary_pod_registry, &self.self_pod_id) {
            (Some(r), Some(p)) => (r, p),
            _ => return Ok(()),
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let decision = consult_for_write_leased(
            registry,
            self.partition_lease_manager.as_deref(),
            pod_id,
            tenant.unwrap_or("default"),
            collection_id,
            now_ms,
        )
        .await
        .map_err(|e| anyhow!("DDL lease check failed: {e}"))?;
        match decision {
            WriteRoutingDecision::Allow => Ok(()),
            WriteRoutingDecision::Misrouted { target_pod } => Err(anyhow!(
                "DDL misrouted: write for collection '{collection_id}' must go to pod '{target_pod}'"
            )),
        }
    }

    /// Execute a DDL statement
    ///
    /// This is the main entry point for DDL operations. It dispatches to the appropriate
    /// handler based on the statement type.
    /// Execute a DDL statement (single-tenant / unscoped).
    pub async fn execute(&self, statement: DdlStatement) -> Result<DdlResult> {
        self.execute_scoped(statement, None).await
    }

    /// Execute a DDL statement within a tenant scope (TD-064). The tenant
    /// scopes table-targeting DDL (CREATE/DROP/ALTER TABLE, CREATE/DROP INDEX)
    /// and explicit namespace creation onto the same tenant-prefixed catalog
    /// namespace the DML path resolves, so a tenant's CREATE-then-INSERT
    /// address one schema row. `None` ⇒ single-tenant, identical to the legacy
    /// path.
    pub async fn execute_scoped(
        &self,
        statement: DdlStatement,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        match statement {
            DdlStatement::CreateTable {
                table_name,
                columns,
                constraints,
                if_not_exists,
                properties,
            } => {
                self.check_lease_for_write(tenant, &table_name).await?;
                self.create_table(
                    &table_name,
                    columns,
                    constraints,
                    if_not_exists,
                    properties,
                    tenant,
                )
                .await
            }
            DdlStatement::DropTable {
                table_name,
                if_exists,
                purge,
            } => {
                self.check_lease_for_write(tenant, &table_name).await?;
                self.drop_table(&table_name, if_exists, purge, tenant).await
            }
            DdlStatement::AlterTable {
                table_name,
                changes,
            } => {
                self.check_lease_for_write(tenant, &table_name).await?;
                self.alter_table(&table_name, changes, tenant).await
            }
            DdlStatement::MaterializeTable { name } => {
                self.check_lease_for_write(tenant, &name).await?;
                let materializer = self.materializer.as_ref().ok_or_else(|| {
                    anyhow!(
                        "ALTER TABLE … MATERIALIZE requires a configured warehouse object \
                         store (no table materializer is wired)"
                    )
                })?;
                let location = materializer.materialize(&name, tenant).await?;
                // MATERIALIZE can outlive a lease TTL. Revalidate before
                // publishing success so a displaced pod cannot bless stale work.
                self.check_lease_for_write(tenant, &name).await?;
                // TD-OLAP-4: a re-MATERIALIZE republishes the base at this
                // location, so any cached table-OPEN discovery (schema/splits/
                // sizes) for it is now stale — drop it so reads re-discover.
                #[cfg(feature = "datafusion-integration")]
                crate::datafusion::engine_adapters::table_open_cache::invalidate_location(
                    &location,
                );
                Ok(DdlResult::success(format!(
                    "Materialized table '{name}' to '{location}'"
                )))
            }
            DdlStatement::CreateIndex {
                index_name,
                table_name,
                columns,
                index_type,
                if_not_exists,
            } => {
                self.check_lease_for_write(tenant, &table_name).await?;
                let result = self
                    .create_index(
                        &index_name,
                        &table_name,
                        columns,
                        index_type,
                        if_not_exists,
                        tenant,
                    )
                    .await?;
                self.check_lease_for_write(tenant, &table_name).await?;
                Ok(result)
            }
            DdlStatement::DropIndex {
                index_name,
                table_name,
                if_exists,
            } => {
                self.check_lease_for_write(tenant, &table_name).await?;
                let result = self
                    .drop_index(&index_name, &table_name, if_exists, tenant)
                    .await?;
                self.check_lease_for_write(tenant, &table_name).await?;
                Ok(result)
            }
            DdlStatement::CreateNamespace {
                namespace,
                if_not_exists,
                properties,
            } => {
                self.create_namespace(&namespace, if_not_exists, properties, tenant)
                    .await
            }
            DdlStatement::DropNamespace {
                namespace,
                if_exists,
                cascade,
            } => {
                self.drop_namespace(&namespace, if_exists, cascade, tenant)
                    .await
            }
            DdlStatement::CreateCollection {
                collection_name,
                dimension,
                engine,
                if_not_exists,
                properties,
            } => {
                self.create_collection(
                    &collection_name,
                    dimension,
                    engine,
                    if_not_exists,
                    properties,
                )
                .await
            }
            DdlStatement::CreateRankProfile {
                name,
                spec_toml,
                if_not_exists,
            } => {
                self.create_rank_profile(&name, &spec_toml, if_not_exists)
                    .await
            }
            DdlStatement::CreateFunction {
                name,
                params,
                return_ty,
                body,
                or_replace,
            } => {
                self.create_function(&name, params, return_ty, &body, or_replace)
                    .await
            }
            DdlStatement::DropRankProfile { name, if_exists } => {
                self.drop_rank_profile(&name, if_exists).await
            }
        }
    }

    // ========================
    // Rank Profile Operations
    // ========================

    /// Install (or replace) a rank profile in the durable catalog and the
    /// live `RankServices` registry. Requires both
    /// `with_rank_profile_store(...)` and `with_rank_services(...)` to have
    /// been wired; otherwise returns a clean error so SQL clients see a
    /// readable failure rather than a panic.
    async fn create_rank_profile(
        &self,
        name: &str,
        spec_toml: &str,
        if_not_exists: bool,
    ) -> Result<DdlResult> {
        use proximadb_rank_profile::{CompiledRankProfile, dsl::parse_single};

        let store = self.rank_profile_store.as_ref().ok_or_else(|| {
            anyhow!(
                "CREATE RANK PROFILE: rank-profile catalog is not configured on this DDL service"
            )
        })?;
        let services = self.rank_services.as_ref().ok_or_else(|| {
            anyhow!(
                "CREATE RANK PROFILE: rank-services registry is not configured on this DDL service"
            )
        })?;

        if if_not_exists && store.get(name).await?.is_some() {
            return Ok(DdlResult::already_exists("RANK PROFILE", name));
        }

        // Validate + compile up-front so the operator gets a clear parse /
        // validation error before anything is persisted to the catalog.
        let spec = parse_single(name, spec_toml)
            .map_err(|e| anyhow!("CREATE RANK PROFILE '{}': invalid spec: {}", name, e))?;
        let compiled = CompiledRankProfile::compile(spec, services.blueprint_factory.clone())
            .map_err(|e| {
                // Record the failure for ops dashboards before bubbling up.
                services.record_profile_reload_error(name);
                anyhow!("CREATE RANK PROFILE '{}': compile failed: {}", name, e)
            })?;

        store
            .install(name, spec_toml.to_string(), None, None)
            .await
            .map_err(|e| {
                anyhow!(
                    "CREATE RANK PROFILE '{}': catalog write failed: {}",
                    name,
                    e
                )
            })?;

        services.install_profile(compiled);

        info!(profile = %name, "Created rank profile");
        Ok(DdlResult::success(format!("CREATE RANK PROFILE {}", name)))
    }

    /// Remove a rank profile from the durable catalog and the live
    /// `RankServices` registry.
    async fn drop_rank_profile(&self, name: &str, if_exists: bool) -> Result<DdlResult> {
        let store = self.rank_profile_store.as_ref().ok_or_else(|| {
            anyhow!("DROP RANK PROFILE: rank-profile catalog is not configured on this DDL service")
        })?;

        let removed = store
            .remove(name)
            .await
            .map_err(|e| anyhow!("DROP RANK PROFILE '{}': catalog delete failed: {}", name, e))?;
        if !removed {
            if if_exists {
                return Ok(DdlResult::not_found("RANK PROFILE", name));
            }
            return Err(anyhow!("Rank profile '{}' does not exist", name));
        }

        if let Some(services) = self.rank_services.as_ref() {
            services.profile_registry.remove(name);
        }

        info!(profile = %name, "Dropped rank profile");
        Ok(DdlResult::success(format!("DROP RANK PROFILE {}", name)))
    }

    /// `CREATE [OR REPLACE] FUNCTION` (F5): lower the SQL body to an engine-neutral `Expr`
    /// (parameters as ordinals) and register a SQL-expression-bodied scalar in the shared
    /// function registry, so the user function runs on BOTH engines (Volcano dispatch +
    /// DataFusion `ScalarUDF` adapter) reusing native machinery — not a bespoke interpreter.
    async fn create_function(
        &self,
        name: &str,
        params: Vec<(String, ProximaType)>,
        return_ty: ProximaType,
        body: &str,
        or_replace: bool,
    ) -> Result<DdlResult> {
        let key = name.to_ascii_lowercase();
        if !or_replace
            && proximadb_functions::builtins()
                .lookup_scalar(&key)
                .is_some()
        {
            return Err(anyhow!(
                "function '{name}' already exists (use CREATE OR REPLACE FUNCTION)"
            ));
        }
        let stored = crate::services::StoredFunction {
            name: key.clone(),
            params,
            return_ty,
            body: body.to_string(),
            created_at_ms: crate::services::function_store::now_ms(),
        };
        // Register live (this also validates the body via the shared frontend lowering) ...
        crate::services::function_store::register_stored_function(&stored)
            .map_err(|e| anyhow!("CREATE FUNCTION {name}: {e}"))?;
        // ... and persist to the durable catalog when one is configured, so it survives a
        // restart (boot recovery replays the catalog through `register_stored_function`).
        if let Some(store) = &self.function_store {
            store
                .put(stored)
                .await
                .map_err(|e| anyhow!("CREATE FUNCTION {name}: persisting to catalog: {e}"))?;
        }
        info!(function = %name, "Created SQL function");
        Ok(DdlResult::success(format!("Function '{name}' created")))
    }

    // ========================
    // Table Operations
    // ========================

    /// Create a new table with the given columns and properties in the catalog.
    async fn create_table(
        &self,
        table_name: &str,
        columns: Vec<DdlColumnDefinition>,
        constraints: Vec<TableConstraint>,
        if_not_exists: bool,
        properties: HashMap<String, String>,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        // Check if table exists
        if catalog.table_exists(&table_id).await? {
            if if_not_exists {
                return Ok(DdlResult::already_exists("Table", table_name));
            } else {
                return Err(anyhow!("Table '{}' already exists", table_name));
            }
        }

        // Build catalog schema
        let schema = self.build_catalog_schema(&table_id.name, columns, constraints, properties)?;

        // PostgreSQL clients routinely create unqualified tables without
        // explicitly creating a schema first. Keep the native catalog path
        // aligned with the OLTP catalog by materializing the resolved namespace.
        if !catalog.namespace_exists(&table_id.namespace).await? {
            // Record the owning tenant on the auto-created namespace (TD-113) so it
            // is DR-addressable and the warehouse path resolver can assert tenant
            // ownership / route by storage pool.
            catalog
                .create_namespace_for_tenant(&table_id.namespace, HashMap::new(), tenant)
                .await?;
        }

        // Create the table
        catalog.create_table(&table_id, schema).await?;

        info!(table = %table_name, "Created table");
        Ok(DdlResult::success(format!(
            "Created table '{}'",
            table_name
        )))
    }

    /// Drop a table from the catalog, optionally purging its data.
    async fn drop_table(
        &self,
        table_name: &str,
        if_exists: bool,
        purge: bool,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        // Check if table exists
        if !catalog.table_exists(&table_id).await? {
            if if_exists {
                return Ok(DdlResult::not_found("Table", table_name));
            } else {
                return Err(anyhow!("Table '{}' does not exist", table_name));
            }
        }

        // Snapshot the schema BEFORE the catalog drop: the record purge keys
        // partitions off the schema's stable identity + name.
        let dropped_schema = catalog.get_table(&table_id).await.ok();

        // Drop the table
        catalog.drop_table(&table_id, purge).await?;

        // TD-CONV-2: purge the record tier so a table recreated with the same
        // name starts empty. Best-effort — the catalog drop already committed;
        // stores without drop support warn (trait default).
        if let (Some(record_store), Some(schema)) = (&self.record_store, dropped_schema.as_ref())
            && let Err(e) = record_store
                .drop_table_records(
                    schema,
                    tenant
                        .map(crate::storage::tenant::context::TenantContext::for_tenant_id)
                        .as_ref(),
                )
                .await
        {
            warn!(
                table = %table_name,
                error = %e,
                "record-tier purge after DROP TABLE failed; a recreated table may see dropped rows"
            );
        }

        info!(table = %table_name, purge = purge, "Dropped table");
        Ok(DdlResult::success(format!(
            "Dropped table '{}'",
            table_name
        )))
    }

    /// Apply schema evolution changes (add/drop columns, rename, etc.) to an existing table.
    async fn alter_table(
        &self,
        table_name: &str,
        changes: Vec<AlterTableChange>,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        // Check if table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{}' does not exist", table_name));
        }

        // Convert changes to catalog schema evolution
        let evolution = self.build_schema_evolution(changes)?;

        // Apply evolution
        catalog.evolve_schema(&table_id, evolution).await?;

        // Bump corpus_version: a schema change definitionally
        // invalidates cached plans — predicate types may shift,
        // new columns add predicate surface, dropped columns
        // invalidate predicates against them. Same tenant-extraction
        // convention as create_index/drop_index (namespace[0]).
        if let Some(tenant_id) = table_id.namespace.first() {
            let version = crate::catalog::CorpusVersionRegistry::global()
                .bump(tenant_id, &table_id.name)
                .await;
            tracing::debug!(
                table = %table_name,
                tenant = %tenant_id,
                version,
                "🔄 corpus_version bumped after evolve_schema"
            );
        }

        info!(table = %table_name, "Altered table");
        Ok(DdlResult::success(format!(
            "Altered table '{}'",
            table_name
        )))
    }

    // ========================
    // Index Operations
    // ========================

    /// Create a new index on the specified table columns with the given type.
    async fn create_index(
        &self,
        index_name: &str,
        table_name: &str,
        columns: Vec<String>,
        index_type: IndexType,
        if_not_exists: bool,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        // Check if table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{}' does not exist", table_name));
        }

        // Check if index already exists
        let existing_indexes = catalog.list_indexes(&table_id).await?;
        if existing_indexes.iter().any(|idx| idx.name == index_name) {
            if if_not_exists {
                return Ok(DdlResult::already_exists("Index", index_name));
            } else {
                return Err(anyhow!(
                    "Index '{}' already exists on table '{}'",
                    index_name,
                    table_name
                ));
            }
        }

        // Convert index type
        let catalog_index_type = self.convert_index_type(&index_type);

        // Build index with properties
        let mut index = CatalogIndex::new(index_name.to_string(), columns, catalog_index_type);
        index.properties = self.get_index_properties(&index_type);

        // Create the index
        catalog.create_index(&table_id, index).await?;

        // Bump corpus_version: a new index changes which routes the
        // planner can pick (e.g. an HNSW index unlocks a different
        // route choice than the lexical-only path). Extract tenant
        // from namespace[0] per the catalog convention (tenant_id is
        // the first namespace segment in multi-tenant tables); skip
        // when the table isn't tenant-scoped.
        if let Some(tenant_id) = table_id.namespace.first() {
            let version = crate::catalog::CorpusVersionRegistry::global()
                .bump(tenant_id, &table_id.name)
                .await;
            tracing::debug!(
                index = %index_name,
                table = %table_name,
                tenant = %tenant_id,
                version,
                "🔄 corpus_version bumped after create_index"
            );
        }

        info!(index = %index_name, table = %table_name, "Created index");
        Ok(DdlResult::success(format!(
            "Created index '{}' on table '{}'",
            index_name, table_name
        )))
    }

    /// Remove an existing index from the specified table.
    async fn drop_index(
        &self,
        index_name: &str,
        table_name: &str,
        if_exists: bool,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        // Check if table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{}' does not exist", table_name));
        }

        // Check if index exists
        let existing_indexes = catalog.list_indexes(&table_id).await?;
        if !existing_indexes.iter().any(|idx| idx.name == index_name) {
            if if_exists {
                return Ok(DdlResult::not_found("Index", index_name));
            } else {
                return Err(anyhow!(
                    "Index '{}' does not exist on table '{}'",
                    index_name,
                    table_name
                ));
            }
        }

        // Drop the index
        catalog.drop_index(&table_id, index_name).await?;

        // Bump corpus_version — dropping an index changes the
        // available routes (FullPrecisionGraph → forced when the
        // quantized index is gone, etc.). Same tenant-extraction
        // convention as create_index.
        if let Some(tenant_id) = table_id.namespace.first() {
            let version = crate::catalog::CorpusVersionRegistry::global()
                .bump(tenant_id, &table_id.name)
                .await;
            tracing::debug!(
                index = %index_name,
                table = %table_name,
                tenant = %tenant_id,
                version,
                "🔄 corpus_version bumped after drop_index"
            );
        }

        info!(index = %index_name, table = %table_name, "Dropped index");
        Ok(DdlResult::success(format!(
            "Dropped index '{}' from table '{}'",
            index_name, table_name
        )))
    }

    // ========================
    // Namespace Operations
    // ========================

    fn tenant_scoped_namespace(namespace: &[String], tenant: Option<&str>) -> Result<Vec<String>> {
        let Some(tenant) = tenant.filter(|tenant| !tenant.is_empty()) else {
            return Ok(namespace.to_vec());
        };
        if tenant.starts_with('_') {
            anyhow::bail!(
                "tenant '{tenant}' is invalid: tenant identifiers must not begin with '_' \
                 (reserved for system/control-plane catalog subtrees)"
            );
        }
        if namespace.first().is_some_and(|segment| segment == tenant) {
            return Ok(namespace.to_vec());
        }

        let mut scoped = Vec::with_capacity(namespace.len() + 1);
        scoped.push(tenant.to_string());
        scoped.extend(namespace.iter().cloned());
        Ok(scoped)
    }

    /// Create a new namespace (schema grouping) in the default catalog.
    async fn create_namespace(
        &self,
        namespace: &[String],
        if_not_exists: bool,
        properties: HashMap<String, String>,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let namespace = Self::tenant_scoped_namespace(namespace, tenant)?;
        let ns_name = namespace.join(".");

        // Check if namespace exists
        if catalog.namespace_exists(&namespace).await? {
            if if_not_exists {
                return Ok(DdlResult::already_exists("Namespace", &ns_name));
            } else {
                return Err(anyhow!("Namespace '{}' already exists", ns_name));
            }
        }

        // Create the namespace. In a tenant-scoped request, persist the owning
        // tenant so downstream DrPathBuilder users can fail closed on
        // cross-tenant materialize/read paths instead of treating the namespace
        // as legacy/unowned.
        match tenant {
            Some(tenant) if !tenant.is_empty() => {
                catalog
                    .create_namespace_for_tenant(&namespace, properties, Some(tenant))
                    .await?;
            }
            _ => {
                catalog.create_namespace(&namespace, properties).await?;
            }
        }

        info!(namespace = %ns_name, "Created namespace");
        Ok(DdlResult::success(format!(
            "Created namespace '{}'",
            ns_name
        )))
    }

    /// Drop a namespace, optionally cascading to all contained tables.
    async fn drop_namespace(
        &self,
        namespace: &[String],
        if_exists: bool,
        cascade: bool,
        tenant: Option<&str>,
    ) -> Result<DdlResult> {
        let catalog = self.catalog_manager.default_catalog().await?;
        let namespace = Self::tenant_scoped_namespace(namespace, tenant)?;
        let ns_name = namespace.join(".");

        // Check if namespace exists
        if !catalog.namespace_exists(&namespace).await? {
            if if_exists {
                return Ok(DdlResult::not_found("Namespace", &ns_name));
            } else {
                return Err(anyhow!("Namespace '{}' does not exist", ns_name));
            }
        }

        // Drop the namespace
        catalog.drop_namespace(&namespace, cascade).await?;

        info!(namespace = %ns_name, cascade = cascade, "Dropped namespace");
        Ok(DdlResult::success(format!(
            "Dropped namespace '{}'",
            ns_name
        )))
    }

    // ========================
    // Collection Operations (ProximaDB-specific)
    // ========================

    /// Create a ProximaDB vector collection with auto-generated HNSW index.
    async fn create_collection(
        &self,
        collection_name: &str,
        dimension: u32,
        engine: Option<String>,
        if_not_exists: bool,
        mut properties: HashMap<String, String>,
    ) -> Result<DdlResult> {
        let (catalog, table_id) = self.catalog_manager.resolve_table(collection_name).await?;

        // Check if collection exists
        if catalog.table_exists(&table_id).await? {
            if if_not_exists {
                return Ok(DdlResult::already_exists("Collection", collection_name));
            } else {
                return Err(anyhow!("Collection '{}' already exists", collection_name));
            }
        }

        // Add ProximaDB-specific properties
        properties.insert("type".to_string(), "vector_collection".to_string());
        properties.insert("dimension".to_string(), dimension.to_string());
        if let Some(eng) = engine {
            properties.insert("engine".to_string(), eng);
        }

        // Build schema with standard vector collection columns
        let columns = vec![
            DdlColumnDefinition {
                name: "id".to_string(),
                data_type: SqlDataType::Varchar {
                    max_length: Some(255),
                },
                nullable: false,
                default_value: None,
                comment: Some("Vector record ID".to_string()),
                primary_key: true,
            },
            DdlColumnDefinition {
                name: "vector".to_string(),
                data_type: SqlDataType::Vector { dimension },
                nullable: false,
                default_value: None,
                comment: Some("Vector embedding".to_string()),
                primary_key: false,
            },
            DdlColumnDefinition {
                name: "metadata".to_string(),
                data_type: SqlDataType::Json,
                nullable: true,
                default_value: None,
                comment: Some("JSON metadata".to_string()),
                primary_key: false,
            },
            DdlColumnDefinition {
                name: "timestamp".to_string(),
                data_type: SqlDataType::Timestamp,
                nullable: true,
                default_value: None,
                comment: Some("Creation timestamp".to_string()),
                primary_key: false,
            },
        ];

        let schema = self.build_catalog_schema(&table_id.name, columns, Vec::new(), properties)?;

        // Create the collection/table
        catalog.create_table(&table_id, schema).await?;

        // Auto-create HNSW index on vector column
        let index = CatalogIndex::new(
            format!("{}_vector_hnsw", table_id.name),
            vec!["vector".to_string()],
            CatalogIndexType::Hnsw,
        );
        catalog.create_index(&table_id, index).await?;

        info!(collection = %collection_name, dimension = dimension, "Created collection");
        Ok(DdlResult::success(format!(
            "Created collection '{}' with dimension {}",
            collection_name, dimension
        )))
    }

    // ========================
    // Helper Methods
    // ========================

    /// Build a `CatalogTableSchema` from column definitions and table properties.
    fn build_catalog_schema(
        &self,
        table_name: &str,
        columns: Vec<DdlColumnDefinition>,
        constraints: Vec<TableConstraint>,
        properties: HashMap<String, String>,
    ) -> Result<CatalogTableSchema> {
        let mut schema = CatalogTableSchema::new(table_name);
        schema.properties = properties;

        let mut primary_key_cols = Vec::new();
        let mut has_json = false;
        let mut has_vector = false;

        for (idx, col) in columns.into_iter().enumerate() {
            let (data_type, col_properties) = self.sql_to_catalog_type(&col.data_type)?;
            has_json |= matches!(data_type, ProximaType::Json);
            has_vector |= matches!(
                data_type,
                ProximaType::DenseVector { .. }
                    | ProximaType::SparseVector { .. }
                    | ProximaType::BinaryVector { .. }
            );

            let catalog_col = CatalogColumn {
                id: idx as i32 + 1,
                object_id: None,
                name: col.name.clone(),
                data_type,
                nullable: col.nullable,
                default_value: col.default_value,
                comment: col.comment,
                properties: col_properties,
                is_deleted: false,
                original_id: None,
            };

            if col.primary_key {
                primary_key_cols.push(col.name);
            }

            schema = schema.with_column(catalog_col);
        }

        if !primary_key_cols.is_empty() {
            schema = schema.with_primary_key(primary_key_cols);
        }

        let relational_capabilities =
            self.build_relational_capabilities(&schema.primary_key, constraints);
        if relational_capabilities.has_enforced_semantics() {
            schema = schema.with_relational_capabilities(relational_capabilities);
        }

        schema
            .properties
            .entry("schema_kind".to_string())
            .or_insert_with(|| match (has_json, has_vector) {
                (true, true) => "mixed_relational_document_vector".to_string(),
                (true, false) => "relational_document".to_string(),
                (false, true) => "relational_vector".to_string(),
                (false, false) => "relational".to_string(),
            });

        // Honor `WITH (canonical_embedding_precision = '<fp32|fp16|bf16|int8|uint8>')`
        // on CREATE TABLE so pgwire SQL DDL clients can opt into a
        // non-fp32 collection. The property name matches what the
        // proto / REST handler expose. Same string-label dispatch as
        // `apply_proto_enum_workarounds` in
        // `crates/platform/proximadb-api/src/rest/canonical/catalog.rs` so
        // mixed-protocol clients see consistent semantics. Unknown
        // values fall back to Fp32 with a warn-level trace rather
        // than failing the CREATE — the legacy fp32 path is the
        // safe default.
        if let Some(raw) = schema.properties.get("canonical_embedding_precision") {
            use proximadb_records::EmbeddingScalarType as P;
            let normalised = raw.trim().to_ascii_lowercase();
            let stripped = normalised
                .strip_prefix("embedding_precision_")
                .unwrap_or(&normalised);
            let target = match stripped {
                "fp32" | "f32" | "float32" | "" => Some(P::Fp32),
                "fp16" | "f16" | "float16" | "half" => Some(P::Fp16),
                "bf16" | "bfloat16" | "brain_float16" => Some(P::Bf16),
                "int8" | "i8" | "int8_scalar" => Some(P::Int8Scalar),
                "uint8" | "u8" | "uint8_scalar" => Some(P::UInt8Scalar),
                _ => {
                    tracing::warn!(
                        raw = %raw,
                        table = %table_name,
                        "CREATE TABLE WITH (canonical_embedding_precision=...): \
                         unrecognized value; defaulting to fp32"
                    );
                    None
                }
            };
            if let Some(t) = target {
                schema.canonical_embedding_precision = t;
            }
        }

        self.apply_storage_profile(&mut schema, has_json, has_vector)?;

        Ok(schema)
    }

    fn apply_storage_profile(
        &self,
        schema: &mut CatalogTableSchema,
        has_json: bool,
        has_vector: bool,
    ) -> Result<()> {
        let profile = schema
            .properties
            .get("workload")
            .or_else(|| schema.properties.get("workload_profile"))
            .or_else(|| schema.properties.get("profile"))
            .and_then(|value| CatalogWorkloadProfile::parse(value))
            .unwrap_or_else(|| infer_workload_profile(has_json, has_vector));

        let specialization = schema
            .properties
            .get("layout")
            .or_else(|| schema.properties.get("storage_layout"))
            .or_else(|| schema.properties.get("specialization"))
            .or_else(|| schema.properties.get("storage_engine"))
            .and_then(|value| CatalogStorageSpecialization::parse(value))
            .unwrap_or_else(|| default_specialization_for_workload(profile, has_json, has_vector));

        schema.workload_profile = profile;
        schema.storage_specialization = specialization;
        schema
            .properties
            .insert("workload_profile".to_string(), profile.as_str().to_string());
        schema.properties.insert(
            "storage_specialization".to_string(),
            specialization.as_str().to_string(),
        );
        normalize_route_knob_properties(schema);

        schema.storage_layouts = vec![primary_layout_for_specialization(specialization, profile)];

        add_specialty_projection_layouts(schema, has_json, has_vector);
        apply_projection_freshness_options(schema);
        add_open_table_layouts(schema)?;

        Ok(())
    }

    fn build_relational_capabilities(
        &self,
        primary_key: &[String],
        constraints: Vec<TableConstraint>,
    ) -> RelationalCapabilities {
        let mut capabilities = RelationalCapabilities {
            primary_key: primary_key.to_vec(),
            ..Default::default()
        };

        for constraint in constraints {
            match constraint {
                TableConstraint::Unique { columns } => {
                    let index_name = format!("unique_{}", columns.join("_"));
                    let index = CatalogIndex::new(index_name, columns, CatalogIndexType::BTree);
                    capabilities.unique_indexes.push(index);
                }
                TableConstraint::Check { expression } => {
                    capabilities
                        .constraints
                        .push(ColumnConstraint::Check { expression });
                }
                TableConstraint::ForeignKey {
                    columns,
                    references_table,
                    references_columns,
                    on_delete,
                    on_update,
                } => {
                    capabilities.constraints.push(ColumnConstraint::ForeignKey {
                        columns,
                        references_table,
                        references_columns,
                        on_delete,
                        on_update,
                    });
                }
            }
        }

        capabilities
    }

    /// Convert a SQL data type to its catalog equivalent and extract type properties.
    fn sql_to_catalog_type(
        &self,
        sql_type: &SqlDataType,
    ) -> Result<(ProximaType, HashMap<String, String>)> {
        use proximadb_data_model::{TimeUnit, VectorElement};
        let mut properties = HashMap::new();

        let catalog_type = match sql_type {
            SqlDataType::Boolean => ProximaType::Boolean,
            SqlDataType::TinyInt => ProximaType::Int8,
            SqlDataType::SmallInt => ProximaType::Int16,
            SqlDataType::Int => ProximaType::Int32,
            SqlDataType::BigInt => ProximaType::Int64,
            SqlDataType::Float => ProximaType::Float32,
            SqlDataType::Double => ProximaType::Float64,
            SqlDataType::Decimal { precision, scale } => {
                properties.insert("precision".to_string(), precision.to_string());
                properties.insert("scale".to_string(), scale.to_string());
                // Placeholder precision/scale; the authoritative values live in
                // column properties (matching the legacy CatalogDataType::Decimal
                // dimensionless mapping).
                ProximaType::Decimal {
                    precision: 38,
                    scale: 10,
                }
            }
            SqlDataType::Varchar { max_length } => {
                if let Some(len) = max_length {
                    properties.insert("max_length".to_string(), len.to_string());
                }
                ProximaType::String
            }
            SqlDataType::Text => ProximaType::String,
            SqlDataType::Binary | SqlDataType::Blob => ProximaType::Binary,
            SqlDataType::Date => ProximaType::Date,
            SqlDataType::Time => ProximaType::Time(TimeUnit::Nanosecond),
            SqlDataType::Timestamp => ProximaType::Timestamp(TimeUnit::Nanosecond),
            SqlDataType::TimestampTz => ProximaType::TimestampTz(TimeUnit::Nanosecond),
            SqlDataType::Uuid => ProximaType::Uuid,
            SqlDataType::Json => {
                properties.insert("json_encoding".to_string(), "json".to_string());
                ProximaType::Json
            }
            SqlDataType::Jsonb => {
                properties.insert("json_encoding".to_string(), "jsonb".to_string());
                ProximaType::Json
            }
            SqlDataType::Vector { dimension } => {
                properties.insert("dimension".to_string(), dimension.to_string());
                // Dimensionless placeholder; real dimension lives in properties.
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                }
            }
            SqlDataType::SparseVector { dimension } => {
                properties.insert("dimension".to_string(), dimension.to_string());
                ProximaType::SparseVector {
                    element: VectorElement::Float32,
                }
            }
            SqlDataType::BinaryVector { dimension } => {
                properties.insert("dimension".to_string(), dimension.to_string());
                ProximaType::BinaryVector { dim: 0 }
            }
        };

        Ok((catalog_type, properties))
    }

    /// Translate ALTER TABLE changes into a `CatalogSchemaEvolution` for the catalog layer.
    fn build_schema_evolution(
        &self,
        changes: Vec<AlterTableChange>,
    ) -> Result<CatalogSchemaEvolution> {
        let mut schema_changes = Vec::new();

        for change in changes {
            let catalog_change = match change {
                AlterTableChange::AddColumn(col) => {
                    let (data_type, _) = self.sql_to_catalog_type(&col.data_type)?;
                    SchemaChange::AddColumn {
                        name: col.name,
                        data_type,
                        nullable: col.nullable,
                        default_value: col.default_value,
                        comment: col.comment,
                        after: None,
                    }
                }
                AlterTableChange::DropColumn(name) => SchemaChange::DropColumn { name },
                AlterTableChange::RenameColumn { old_name, new_name } => {
                    SchemaChange::RenameColumn { old_name, new_name }
                }
                AlterTableChange::ChangeType {
                    column_name,
                    new_type,
                } => {
                    let (data_type, _) = self.sql_to_catalog_type(&new_type)?;
                    SchemaChange::ChangeType {
                        name: column_name,
                        new_type: data_type,
                    }
                }
                AlterTableChange::SetNullable {
                    column_name,
                    nullable,
                } => {
                    if nullable {
                        SchemaChange::MakeNullable { name: column_name }
                    } else {
                        // Now we support SET NOT NULL via MakeNotNullable
                        SchemaChange::MakeNotNullable { name: column_name }
                    }
                }
                AlterTableChange::SetDefault {
                    column_name,
                    default_value,
                } => {
                    if let Some(val) = default_value {
                        SchemaChange::SetDefault {
                            name: column_name,
                            default_value: val,
                        }
                    } else {
                        SchemaChange::DropDefault { name: column_name }
                    }
                }
                AlterTableChange::SetComment {
                    column_name,
                    comment,
                } => SchemaChange::UpdateComment {
                    name: column_name,
                    comment,
                },
                AlterTableChange::MoveColumn {
                    column_name,
                    position,
                } => SchemaChange::MoveColumn {
                    name: column_name,
                    after: match position {
                        ColumnPosition::First => None,
                        ColumnPosition::After(col) => Some(col),
                    },
                },
                AlterTableChange::AddConstraint {
                    constraint_name,
                    constraint,
                } => {
                    let catalog_constraint = match constraint {
                        TableConstraint::Unique { columns } => ColumnConstraint::Unique { columns },
                        TableConstraint::Check { expression } => {
                            ColumnConstraint::Check { expression }
                        }
                        TableConstraint::ForeignKey {
                            columns,
                            references_table,
                            references_columns,
                            on_delete,
                            on_update,
                        } => ColumnConstraint::ForeignKey {
                            columns,
                            references_table,
                            references_columns,
                            on_delete,
                            on_update,
                        },
                    };
                    SchemaChange::AddConstraint {
                        constraint_name,
                        constraint: catalog_constraint,
                    }
                }
                AlterTableChange::DropConstraint { constraint_name } => {
                    SchemaChange::DropConstraint { constraint_name }
                }
                AlterTableChange::PromotePropsKey {
                    key,
                    column_type,
                    comment,
                } => {
                    let (catalog_type, _) = self.sql_to_catalog_type(&column_type)?;
                    SchemaChange::PromotePropsKey {
                        key,
                        column_type: catalog_type,
                        comment,
                    }
                }
                AlterTableChange::SetTableOption { key, value } => {
                    SchemaChange::SetTableOption { key, value }
                }
            };
            schema_changes.push(catalog_change);
        }

        Ok(CatalogSchemaEvolution {
            changes: schema_changes,
        })
    }

    /// Map a SQL-level index type to its catalog enum variant.
    fn convert_index_type(&self, index_type: &IndexType) -> CatalogIndexType {
        match index_type {
            IndexType::BTree => CatalogIndexType::BTree,
            IndexType::Hash => CatalogIndexType::Hash,
            IndexType::FullText => CatalogIndexType::FullText,
            IndexType::Gin => CatalogIndexType::Gin,
            IndexType::Hnsw { .. } => CatalogIndexType::Hnsw,
            IndexType::Ivf { .. } => CatalogIndexType::Ivf,
            IndexType::Pq { .. } => CatalogIndexType::Pq,
        }
    }

    /// Extract type-specific properties (e.g., m, ef_construction, nlist) from an index type.
    fn get_index_properties(&self, index_type: &IndexType) -> HashMap<String, String> {
        let mut props = HashMap::new();

        match index_type {
            IndexType::Hnsw { m, ef_construction } => {
                if let Some(m_val) = m {
                    props.insert("m".to_string(), m_val.to_string());
                }
                if let Some(ef) = ef_construction {
                    props.insert("ef_construction".to_string(), ef.to_string());
                }
            }
            IndexType::Ivf { nlist: Some(n) } => {
                props.insert("nlist".to_string(), n.to_string());
            }
            IndexType::Ivf { .. } => {}
            IndexType::Pq { m, nbits } => {
                if let Some(m_val) = m {
                    props.insert("m".to_string(), m_val.to_string());
                }
                if let Some(nb) = nbits {
                    props.insert("nbits".to_string(), nb.to_string());
                }
            }
            _ => {}
        }

        props
    }
}

fn infer_workload_profile(has_json: bool, has_vector: bool) -> CatalogWorkloadProfile {
    match (has_json, has_vector) {
        (true, true) => CatalogWorkloadProfile::Mixed,
        (false, true) => CatalogWorkloadProfile::Vector,
        (true, false) => CatalogWorkloadProfile::Document,
        (false, false) => CatalogWorkloadProfile::Htap,
    }
}

fn default_specialization_for_workload(
    profile: CatalogWorkloadProfile,
    has_json: bool,
    has_vector: bool,
) -> CatalogStorageSpecialization {
    match profile {
        CatalogWorkloadProfile::Oltp => CatalogStorageSpecialization::PaxOltp,
        CatalogWorkloadProfile::Olap => CatalogStorageSpecialization::PaxOlap,
        CatalogWorkloadProfile::Htap | CatalogWorkloadProfile::Mixed => {
            CatalogStorageSpecialization::PaxRowFamily
        }
        CatalogWorkloadProfile::Vector if has_vector => CatalogStorageSpecialization::VectorAnn,
        CatalogWorkloadProfile::Document if has_json => CatalogStorageSpecialization::DocumentJson,
        CatalogWorkloadProfile::Graph => CatalogStorageSpecialization::GraphTopology,
        CatalogWorkloadProfile::Observability => {
            CatalogStorageSpecialization::ObservabilityTimeSeries
        }
        CatalogWorkloadProfile::Vector | CatalogWorkloadProfile::Document => {
            CatalogStorageSpecialization::PaxRowFamily
        }
    }
}

fn primary_layout_for_specialization(
    specialization: CatalogStorageSpecialization,
    profile: CatalogWorkloadProfile,
) -> CatalogStorageLayout {
    let mut layout = match specialization {
        CatalogStorageSpecialization::PaxOltp => {
            CatalogStorageLayout::proxima_authoritative_pax("primary")
        }
        CatalogStorageSpecialization::PaxOlap => {
            CatalogStorageLayout::proxima_authoritative_pax("primary")
        }
        CatalogStorageSpecialization::LsmWriteOptimized => {
            CatalogStorageLayout::internal("primary", CatalogStorageLayoutKind::LsmRecord)
        }
        CatalogStorageSpecialization::ColumnarAnalytics => {
            CatalogStorageLayout::internal("primary", CatalogStorageLayoutKind::Columnar)
        }
        CatalogStorageSpecialization::ExternalOpenTable => CatalogStorageLayout {
            name: "primary".to_string(),
            authority: CatalogAuthorityMode::FederatedRead,
            layout_kind: CatalogStorageLayoutKind::ExternalTable,
            physical_format: CatalogPhysicalFormat::External("unknown".to_string()),
            write_mode: proximadb_catalog::CatalogWriteMode::ReadOnly,
            snapshot_semantics: Some("external-snapshot".to_string()),
            policy_enforced_in_proxima: false,
            ..Default::default()
        },
        CatalogStorageSpecialization::GenericRelational
        | CatalogStorageSpecialization::PaxRowFamily
        | CatalogStorageSpecialization::VectorAnn
        | CatalogStorageSpecialization::DocumentJson
        | CatalogStorageSpecialization::GraphTopology
        | CatalogStorageSpecialization::ObservabilityTimeSeries => {
            CatalogStorageLayout::proxima_authoritative_pax("primary")
        }
    };

    match specialization {
        CatalogStorageSpecialization::PaxOltp => {
            layout.properties.insert("pax_mode".into(), "Oltp".into());
        }
        CatalogStorageSpecialization::PaxOlap => {
            layout.properties.insert("pax_mode".into(), "Olap".into());
        }
        CatalogStorageSpecialization::PaxRowFamily
        | CatalogStorageSpecialization::VectorAnn
        | CatalogStorageSpecialization::DocumentJson
        | CatalogStorageSpecialization::GraphTopology
        | CatalogStorageSpecialization::ObservabilityTimeSeries
        | CatalogStorageSpecialization::GenericRelational => {
            layout.properties.insert("pax_mode".into(), "Pax".into());
        }
        CatalogStorageSpecialization::LsmWriteOptimized => {
            layout.physical_format = CatalogPhysicalFormat::Sst;
            layout.snapshot_semantics = Some("mvcc-lsm".to_string());
        }
        CatalogStorageSpecialization::ColumnarAnalytics => {
            layout.physical_format = CatalogPhysicalFormat::ProximaBlock;
            layout.snapshot_semantics = Some("mvcc-columnar".to_string());
        }
        CatalogStorageSpecialization::ExternalOpenTable => {}
    }

    layout
        .properties
        .insert("workload_profile".into(), profile.as_str().into());
    layout.properties.insert(
        "storage_specialization".into(),
        specialization.as_str().into(),
    );
    layout
}

fn add_specialty_projection_layouts(
    schema: &mut CatalogTableSchema,
    has_json: bool,
    has_vector: bool,
) {
    if has_vector
        || matches!(
            schema.storage_specialization,
            CatalogStorageSpecialization::VectorAnn
        )
    {
        schema
            .storage_layouts
            .push(CatalogStorageLayout::specialty_projection(
                "vector_ann",
                CatalogStorageLayoutKind::VectorAnn,
                CatalogPhysicalFormat::ProximaBlock,
            ));
        schema.projections.push(CatalogProjection {
            name: "vector_ann".to_string(),
            kind: CatalogProjectionKind::VectorAnn,
            physical_format: CatalogPhysicalFormat::ProximaBlock,
            rebuild_source: "primary".to_string(),
            freshness: ProjectionFreshness::BoundedLag,
            freshness_state: Default::default(),
            max_lag_ms: Some(1_000),
            source_range: None,
            last_included_position: None,
            rebuild_rto: Some(proximadb_catalog::RebuildRtoSpec::hnsw_benchmarked()),
            rebuildable: true,
            invalidation_policy: Some("mark-stale".to_string()),
            policy_boundary: Some("engine-enforced".to_string()),
            lossy: true,
            benchmark_gate: Some("hybrid-vector-smoke".to_string()),
            support_status: "experimental".to_string(),
            location: None,
            tier: None,
            properties: HashMap::new(),
        });
    }

    if has_json
        || matches!(
            schema.storage_specialization,
            CatalogStorageSpecialization::DocumentJson
        )
    {
        schema
            .storage_layouts
            .push(CatalogStorageLayout::specialty_projection(
                "json_path",
                CatalogStorageLayoutKind::Columnar,
                CatalogPhysicalFormat::ProximaBlock,
            ));
        schema.projections.push(CatalogProjection {
            name: "json_path".to_string(),
            kind: CatalogProjectionKind::JsonPath,
            physical_format: CatalogPhysicalFormat::ProximaBlock,
            rebuild_source: "primary".to_string(),
            freshness: ProjectionFreshness::Lazy,
            freshness_state: Default::default(),
            max_lag_ms: None,
            source_range: None,
            last_included_position: None,
            rebuild_rto: None,
            rebuildable: true,
            invalidation_policy: Some("mark-stale".to_string()),
            policy_boundary: Some("engine-enforced".to_string()),
            lossy: false,
            benchmark_gate: Some("json-path-smoke".to_string()),
            support_status: "experimental".to_string(),
            location: None,
            tier: None,
            properties: HashMap::new(),
        });
    }

    match schema.storage_specialization {
        CatalogStorageSpecialization::GraphTopology => {
            schema
                .storage_layouts
                .push(CatalogStorageLayout::specialty_projection(
                    "graph_topology",
                    CatalogStorageLayoutKind::GraphTopology,
                    CatalogPhysicalFormat::GraphAr,
                ));
        }
        CatalogStorageSpecialization::ObservabilityTimeSeries => {
            schema
                .storage_layouts
                .push(CatalogStorageLayout::specialty_projection(
                    "time_series",
                    CatalogStorageLayoutKind::TimeSeriesBlock,
                    CatalogPhysicalFormat::ProximaBlock,
                ));
        }
        _ => {}
    }
}

fn normalize_route_knob_properties(schema: &mut CatalogTableSchema) {
    if let Some(route) = first_property(
        schema,
        &["compute_route", "preferred_compute_route", "compute"],
    ) {
        schema
            .properties
            .insert("compute_route".to_string(), canonical_option_value(&route));
    }

    if let Some(partitioning) = first_property(
        schema,
        &["partitioning", "partition_key", "distribution_key"],
    ) {
        schema
            .properties
            .insert("partitioning".to_string(), partitioning);
    }

    if let Some(isolation) = first_property(
        schema,
        &["isolation_profile", "isolation", "transaction_profile"],
    ) {
        schema
            .properties
            .insert("isolation_profile".to_string(), isolation.clone());
        if schema.relational_capabilities.transaction_profile.is_none() {
            schema.relational_capabilities.transaction_profile = Some(isolation);
        }
    }

    if let Some(freshness) = first_property(
        schema,
        &["freshness_sla", "projection_freshness", "freshness"],
    ) {
        schema
            .properties
            .insert("freshness_sla".to_string(), freshness);
    }

    if let Some(policy_boundary) =
        first_property(schema, &["policy_boundary", "rls_boundary", "policy"])
    {
        schema
            .properties
            .insert("policy_boundary".to_string(), policy_boundary);
    }

    if let Some(authority) = first_property(schema, &["authority_mode", "authority", "ownership"])
        .and_then(|value| parse_authority_mode(&value))
    {
        schema.properties.insert(
            "authority_mode".to_string(),
            authority.ownership_mode_name().to_string(),
        );
    }
}

fn apply_projection_freshness_options(schema: &mut CatalogTableSchema) {
    let Some(freshness) = first_property(schema, &["projection_freshness", "freshness"]) else {
        return;
    };
    let Some(freshness) = parse_projection_freshness(&freshness) else {
        return;
    };
    let max_lag_ms = first_property(schema, &["projection_max_lag_ms", "max_lag_ms"])
        .and_then(|value| value.parse::<i64>().ok());

    for projection in &mut schema.projections {
        projection.freshness = freshness;
        if max_lag_ms.is_some() {
            projection.max_lag_ms = max_lag_ms;
        }
    }
}

fn add_open_table_layouts(schema: &mut CatalogTableSchema) -> Result<()> {
    let Some(format) = schema
        .properties
        .get("open_table_format")
        .or_else(|| schema.properties.get("table_format"))
        .or_else(|| schema.properties.get("format"))
        .and_then(|value| parse_physical_format(value))
    else {
        return Ok(());
    };

    let location = schema
        .properties
        .get("location")
        .or_else(|| schema.properties.get("external_location"))
        .or_else(|| schema.properties.get("path"))
        .cloned()
        .unwrap_or_default();

    let ownership = schema
        .properties
        .get("ownership")
        .or_else(|| schema.properties.get("authority"))
        .or_else(|| schema.properties.get("authority_mode"))
        .and_then(|value| parse_authority_mode(value))
        .unwrap_or(CatalogAuthorityMode::ProjectionPublication);

    // Path Isolation (CRITICAL mandate): a table that reads an operator/tenant
    // supplied external `location` reaches outside `DrPathBuilder`'s
    // `data/{tenant}/{namespace}/…` tree. Gate it fail-closed behind an operator
    // allowlist of URI-prefix roots (`PROXIMADB_EXTERNAL_TABLE_ROOTS`) so a
    // tenant cannot point a table at an arbitrary path. The publication
    // authorities below synthesize an internal `proximadb://…` location and are
    // exempt; only the externally-located authorities are validated.
    let reads_external_location = matches!(
        ownership,
        CatalogAuthorityMode::ExternalAuthoritative
            | CatalogAuthorityMode::ImportedSnapshot
            | CatalogAuthorityMode::FederatedRead
    );
    if reads_external_location && !external_location_allowed(&location) {
        return Err(anyhow!(
            "external table '{}': location '{}' is not within an allowlisted root — \
             set PROXIMADB_EXTERNAL_TABLE_ROOTS to a comma-separated list of permitted \
             URI-prefix roots to opt in (Path Isolation, fail-closed default)",
            schema.name,
            location
        ));
    }

    let mut layout = match ownership {
        CatalogAuthorityMode::ExternalAuthoritative => {
            if location.is_empty() {
                return Err(anyhow!(
                    "External authoritative table '{}' requires LOCATION or external_location",
                    schema.name
                ));
            }
            CatalogStorageLayout::external_authoritative("external", format, location)
        }
        CatalogAuthorityMode::ImportedSnapshot => {
            if location.is_empty() {
                return Err(anyhow!(
                    "Imported snapshot table '{}' requires LOCATION or external_location",
                    schema.name
                ));
            }
            CatalogStorageLayout::imported_snapshot("imported_snapshot", format, location)
        }
        CatalogAuthorityMode::FederatedRead => {
            if location.is_empty() {
                return Err(anyhow!(
                    "Federated table '{}' requires LOCATION or external_location",
                    schema.name
                ));
            }
            CatalogStorageLayout::federated_read("federated", format, location)
        }
        CatalogAuthorityMode::InternalCanonical
        | CatalogAuthorityMode::ProximaAuthoritative
        | CatalogAuthorityMode::ExportedPublication
        | CatalogAuthorityMode::ProjectionPublication
        | CatalogAuthorityMode::RebuildableProjection => {
            let publication_location = if location.is_empty() {
                format!("proximadb://{}/publications/open_table", schema.name)
            } else {
                location
            };
            CatalogStorageLayout::projection_publication("open_table", format, publication_location)
        }
    };

    if let Some(snapshot) = schema
        .properties
        .get("snapshot_semantics")
        .or_else(|| schema.properties.get("snapshot"))
    {
        layout.snapshot_semantics = Some(snapshot.clone());
    }
    if let Some(refresh) = schema
        .properties
        .get("refresh")
        .or_else(|| schema.properties.get("refresh_mode"))
    {
        layout.properties.insert("refresh".into(), refresh.clone());
    }
    if let Some(schema_mode) = schema.properties.get("schema_mode") {
        layout
            .properties
            .insert("schema_mode".into(), schema_mode.clone());
    }

    schema.storage_layouts.push(layout);
    Ok(())
}

/// Fail-closed allowlist for external-table `location`s (Path Isolation mandate).
///
/// A location is permitted only when the operator has set
/// `PROXIMADB_EXTERNAL_TABLE_ROOTS` to a comma-separated list of URI-prefix roots
/// and the location falls under one of them. Empty/unset allowlist ⇒ every
/// external location is rejected (the feature is off by default). Locations
/// containing `..` are always rejected (no traversal-escape). Matching is on a
/// path boundary — a root `file:///data/ext` matches `file:///data/ext/a.parquet`
/// but NOT `file:///data/ext-evil/…`.
fn external_location_allowed(location: &str) -> bool {
    let loc = location.trim();
    if loc.is_empty() || loc.contains("..") {
        return false;
    }
    let Ok(roots) = std::env::var("PROXIMADB_EXTERNAL_TABLE_ROOTS") else {
        return false;
    };
    roots
        .split(',')
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .any(|root| {
            // Normalize the root's trailing slash so `file:///x/` and `file:///x`
            // both denote the same directory: a location is allowed when it IS
            // the root or sits strictly beneath it (path boundary).
            let root_norm = root.trim_end_matches('/');
            loc == root_norm || loc.starts_with(&format!("{root_norm}/"))
        })
}

fn parse_physical_format(value: &str) -> Option<CatalogPhysicalFormat> {
    Some(match value.trim().to_ascii_lowercase().as_str() {
        "proximablock" | "proxima_block" | "pax" => CatalogPhysicalFormat::ProximaBlock,
        "sst" | "lsm" => CatalogPhysicalFormat::Sst,
        "arrow" | "arrow_ipc" => CatalogPhysicalFormat::Arrow,
        "csv" => CatalogPhysicalFormat::Csv,
        "json" | "jsonl" | "jsonlines" => CatalogPhysicalFormat::Json,
        "xml" => CatalogPhysicalFormat::Xml,
        "avro" => CatalogPhysicalFormat::Avro,
        "parquet" => CatalogPhysicalFormat::Parquet,
        "orc" => CatalogPhysicalFormat::Orc,
        "iceberg" => CatalogPhysicalFormat::Iceberg,
        "delta" | "deltalake" => CatalogPhysicalFormat::Delta,
        "hudi" => CatalogPhysicalFormat::Hudi,
        "graphar" => CatalogPhysicalFormat::GraphAr,
        other if !other.is_empty() => CatalogPhysicalFormat::External(other.to_string()),
        _ => return None,
    })
}

fn parse_authority_mode(value: &str) -> Option<CatalogAuthorityMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "proxima"
        | "proximadb"
        | "internal"
        | "canonical"
        | "proxima_authoritative"
        | "proximaauthoritative" => Some(CatalogAuthorityMode::ProximaAuthoritative),
        "projection"
        | "publication"
        | "projection_publication"
        | "projectionpublication"
        | "export" => Some(CatalogAuthorityMode::ProjectionPublication),
        "import" | "imported" | "imported_snapshot" | "importedsnapshot" => {
            Some(CatalogAuthorityMode::ImportedSnapshot)
        }
        "external" | "external_authoritative" | "externalauthoritative" => {
            Some(CatalogAuthorityMode::ExternalAuthoritative)
        }
        "federated" | "federated_read" | "federatedread" => {
            Some(CatalogAuthorityMode::FederatedRead)
        }
        _ => None,
    }
}

fn parse_projection_freshness(value: &str) -> Option<ProjectionFreshness> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "sync" | "synchronous" => Some(ProjectionFreshness::Synchronous),
        "bounded" | "bounded_lag" | "boundedlag" => Some(ProjectionFreshness::BoundedLag),
        "lazy" | "async" | "asynchronous" => Some(ProjectionFreshness::Lazy),
        "manual" | "explicit" => Some(ProjectionFreshness::Manual),
        _ => None,
    }
}

fn first_property(schema: &CatalogTableSchema, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| schema.properties.get(*key).cloned())
}

fn canonical_option_value(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddl_result_success() {
        let result = DdlResult::success("Operation completed");
        assert!(result.success);
        assert_eq!(result.affected_count, 1);
    }

    // --- DDL write-lease enforcement (check_lease_for_write, ADR-032) ---
    // The lease manager is optional; these exercise the in-memory registry path
    // (manager = None), which is the routing fallback consult_for_write_leased
    // uses when no manager is attached.
    fn lease_ddl() -> DdlService {
        DdlService::new(std::sync::Arc::new(CatalogManager::new()))
    }

    #[tokio::test]
    async fn check_lease_for_write_is_noop_when_not_wired() {
        // No registry / pod id attached ⇒ inert; every DDL is allowed.
        let ddl = lease_ddl();
        ddl.check_lease_for_write(Some("tenant-a"), "coll-1")
            .await
            .expect("inert when lease not wired");
    }

    #[tokio::test]
    async fn check_lease_for_write_allows_when_self_is_primary() {
        use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};
        let registry = std::sync::Arc::new(PrimaryPodRegistry::new());
        registry.assign("tenant-a", "coll-1", "pod-self", AssignmentReason::Create);
        let ddl = lease_ddl()
            .with_primary_pod_registry(registry)
            .with_self_pod_id("pod-self".to_string());
        ddl.check_lease_for_write(Some("tenant-a"), "coll-1")
            .await
            .expect("allowed when self holds the binding");
    }

    #[tokio::test]
    async fn check_lease_for_write_rejects_misrouted_ddl() {
        use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};
        let registry = std::sync::Arc::new(PrimaryPodRegistry::new());
        registry.assign("tenant-a", "coll-1", "pod-other", AssignmentReason::Create);
        let ddl = lease_ddl()
            .with_primary_pod_registry(registry)
            .with_self_pod_id("pod-self".to_string());
        let err = ddl
            .check_lease_for_write(Some("tenant-a"), "coll-1")
            .await
            .expect_err("misrouted DDL must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("misrouted"), "unexpected error: {msg}");
        assert!(
            msg.contains("pod-other"),
            "error should name the target pod: {msg}"
        );
    }

    #[tokio::test]
    async fn materialize_and_index_arms_all_apply_the_lease_gate() {
        use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};
        let registry = std::sync::Arc::new(PrimaryPodRegistry::new());
        registry.assign("tenant-a", "coll-1", "pod-other", AssignmentReason::Create);
        let ddl = lease_ddl()
            .with_primary_pod_registry(registry)
            .with_self_pod_id("pod-self".to_string());

        let statements = [
            DdlStatement::MaterializeTable {
                name: "coll-1".to_string(),
            },
            DdlStatement::CreateIndex {
                index_name: "idx".to_string(),
                table_name: "coll-1".to_string(),
                columns: vec!["id".to_string()],
                index_type: IndexType::BTree,
                if_not_exists: false,
            },
            DdlStatement::DropIndex {
                index_name: "idx".to_string(),
                table_name: "coll-1".to_string(),
                if_exists: false,
            },
        ];

        for statement in statements {
            let err = ddl
                .execute_scoped(statement, Some("tenant-a"))
                .await
                .expect_err("every table-scoped arm must reject a foreign lease");
            assert!(err.to_string().contains("pod-other"), "{err}");
        }
    }

    struct LeaseTakeoverMaterializer {
        contender: std::sync::Arc<crate::cluster::partition_lease::PartitionLeaseManager>,
    }

    #[async_trait::async_trait]
    impl TableMaterializer for LeaseTakeoverMaterializer {
        async fn materialize(&self, table_name: &str, tenant: Option<&str>) -> Result<String> {
            let future_ms = chrono::Utc::now().timestamp_millis() + 100;
            assert!(
                self.contender
                    .acquire(tenant.unwrap_or("default"), table_name, future_ms)
                    .await?,
                "the second pod should acquire after the first lease expires"
            );
            Ok("file:///tmp/materialized".to_string())
        }
    }

    #[tokio::test]
    async fn materialize_revalidates_after_lease_expiry_and_takeover() {
        use crate::cluster::partition_lease::{PartitionLeaseManager, PartitionLeaseStore};
        use crate::cluster::primary_pod_registry::PrimaryPodRegistry;
        use object_store::memory::InMemory;
        use proximadb_object_store::ProximaObjectStore;

        let backing: std::sync::Arc<dyn object_store::ObjectStore> =
            std::sync::Arc::new(InMemory::new());
        let store = std::sync::Arc::new(PartitionLeaseStore::new(
            ProximaObjectStore::new(backing),
            "_operator/leases",
        ));
        let registry_a = std::sync::Arc::new(PrimaryPodRegistry::new());
        let registry_b = std::sync::Arc::new(PrimaryPodRegistry::new());
        let manager_a = std::sync::Arc::new(PartitionLeaseManager::new(
            store.clone(),
            registry_a.clone(),
            "pod-a",
            10,
        ));
        let manager_b =
            std::sync::Arc::new(PartitionLeaseManager::new(store, registry_b, "pod-b", 10));
        let ddl = lease_ddl()
            .with_primary_pod_registry(registry_a)
            .with_self_pod_id("pod-a".to_string())
            .with_partition_lease_manager(manager_a)
            .with_materializer(std::sync::Arc::new(LeaseTakeoverMaterializer {
                contender: manager_b,
            }));

        let err = ddl
            .execute_scoped(
                DdlStatement::MaterializeTable {
                    name: "coll-1".to_string(),
                },
                Some("tenant-a"),
            )
            .await
            .expect_err("post-operation lease validation must detect takeover");
        assert!(err.to_string().contains("pod-b"), "{err}");
    }

    #[tokio::test]
    async fn check_lease_for_write_with_manager_acquires_and_allows() {
        // With a PartitionLeaseManager attached (the production wiring when the
        // lease system is on), check_lease_for_write acquires the lease and
        // allows the write when there is no contention — the manager-attached
        // path the registry-only tests above do not cover.
        use crate::cluster::partition_lease::{PartitionLeaseManager, PartitionLeaseStore};
        use crate::cluster::primary_pod_registry::PrimaryPodRegistry;
        use object_store::memory::InMemory;
        use proximadb_object_store::ProximaObjectStore;

        let backing: std::sync::Arc<dyn object_store::ObjectStore> =
            std::sync::Arc::new(InMemory::new());
        let store = PartitionLeaseStore::new(ProximaObjectStore::new(backing), "_operator/leases");
        let registry = std::sync::Arc::new(PrimaryPodRegistry::new());
        let manager = std::sync::Arc::new(PartitionLeaseManager::new(
            std::sync::Arc::new(store),
            registry.clone(),
            "pod-self",
            10_000,
        ));
        let ddl = lease_ddl()
            .with_primary_pod_registry(registry)
            .with_self_pod_id("pod-self".to_string())
            .with_partition_lease_manager(manager);
        ddl.check_lease_for_write(Some("tenant-a"), "coll-1")
            .await
            .expect("manager acquires the lease (no contention) => Allow");
    }

    // External-location allowlist (Path Isolation, fail-closed). Serialized by
    // the shared env var; nextest's process-per-test isolation keeps these from
    // racing other tests.
    #[test]
    fn external_location_allowlist_is_fail_closed_and_boundary_aware() {
        // Unset ⇒ everything rejected (feature off by default).
        unsafe { std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS") };
        assert!(!external_location_allowed("file:///data/ext/a.parquet"));

        unsafe {
            std::env::set_var(
                "PROXIMADB_EXTERNAL_TABLE_ROOTS",
                "file:///data/ext/, s3://bucket/warehouse/",
            )
        };
        // Within an allowlisted root.
        assert!(external_location_allowed("file:///data/ext/a.parquet"));
        assert!(external_location_allowed(
            "s3://bucket/warehouse/hits/part-0.parquet"
        ));
        // Exact-root match is allowed.
        assert!(external_location_allowed("file:///data/ext"));
        // Path-boundary escape must NOT match a prefix of a sibling dir.
        assert!(!external_location_allowed(
            "file:///data/ext-evil/secret.parquet"
        ));
        // Outside any root.
        assert!(!external_location_allowed("file:///etc/passwd"));
        // Traversal is always rejected, even under an allowlisted root.
        assert!(!external_location_allowed(
            "file:///data/ext/../../etc/passwd"
        ));
        // Empty is rejected.
        assert!(!external_location_allowed("  "));

        unsafe { std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS") };
    }

    #[test]
    fn test_ddl_result_already_exists() {
        let result = DdlResult::already_exists("Table", "users");
        assert!(result.success);
        assert_eq!(result.affected_count, 0);
        assert!(result.message.contains("already exists"));
    }

    #[test]
    fn test_column_definition() {
        let col = DdlColumnDefinition {
            name: "id".to_string(),
            data_type: SqlDataType::BigInt,
            nullable: false,
            default_value: None,
            comment: Some("Primary key".to_string()),
            primary_key: true,
        };

        assert_eq!(col.name, "id");
        assert!(!col.nullable);
        assert!(col.primary_key);
    }

    #[test]
    fn test_sql_data_types() {
        let vector_type = SqlDataType::Vector { dimension: 768 };
        match vector_type {
            SqlDataType::Vector { dimension } => assert_eq!(dimension, 768),
            _ => panic!("Expected Vector type"),
        }
    }

    #[test]
    fn test_sql_to_catalog_type_maps_json_and_jsonb_to_json_catalog_type() {
        let service = DdlService::new(Arc::new(CatalogManager::new()));

        let mapping_inputs = [SqlDataType::Json, SqlDataType::Jsonb];

        for sql_type in mapping_inputs {
            let (catalog_type, props) = service
                .sql_to_catalog_type(&sql_type)
                .expect("mapping json/jsonb should succeed");
            assert_eq!(catalog_type, ProximaType::Json);
            assert_eq!(props.len(), 1, "json/jsonb should preserve encoding");
        }
    }

    #[tokio::test]
    async fn test_agentic_pgwire_ddl_executes_into_catalog_schema() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let service = DdlService::new(manager.clone());
        service
            .execute(DdlStatement::CreateNamespace {
                namespace: vec!["default".to_string()],
                if_not_exists: true,
                properties: HashMap::new(),
            })
            .await
            .expect("create default namespace");

        let parser = crate::query::sql_frontend::SqlFrontendParser::new();
        let create_table = parser
            .parse_ddl(
                "CREATE TABLE IF NOT EXISTS \"agent_store\" (
                    \"record_id\" TEXT NOT NULL,
                    \"tenant_id\" TEXT NOT NULL,
                    \"payload\" JSONB NOT NULL DEFAULT '{}'::jsonb,
                    \"metadata\" JSONB NOT NULL DEFAULT '{}'::jsonb,
                    \"embedding\" VECTOR(384),
                    PRIMARY KEY (\"record_id\")
                ) WITH (
                    storage_engine = 'SST',
                    layout = 'hybrid',
                    xcatalog_namespace = 'agentic.demo',
                    schema_kind = 'agentic_mixed'
                );",
            )
            .expect("parse create table")
            .expect("ddl statement");
        service
            .execute(create_table)
            .await
            .expect("execute create table");

        for index_sql in [
            "CREATE INDEX idx_agent_store_payload_gin ON agent_store USING GIN (payload);",
            "CREATE INDEX idx_agent_store_embedding_hnsw ON agent_store USING HNSW (embedding);",
        ] {
            let statement = parser
                .parse_ddl(index_sql)
                .expect("parse create index")
                .expect("index ddl statement");
            service
                .execute(statement)
                .await
                .expect("execute create index");
        }

        let (catalog, table_id) = manager
            .resolve_table("agent_store")
            .await
            .expect("resolve table");
        let schema = catalog.get_table(&table_id).await.expect("get schema");

        assert_eq!(schema.primary_key, vec!["record_id".to_string()]);
        assert_eq!(
            schema.properties.get("storage_engine").map(String::as_str),
            Some("SST")
        );
        assert_eq!(
            schema.properties.get("layout").map(String::as_str),
            Some("hybrid")
        );
        assert_eq!(
            schema
                .properties
                .get("xcatalog_namespace")
                .map(String::as_str),
            Some("agentic.demo")
        );
        assert_eq!(
            schema.properties.get("schema_kind").map(String::as_str),
            Some("agentic_mixed")
        );

        let payload = schema
            .columns
            .iter()
            .find(|column| column.name == "payload")
            .expect("payload column");
        assert_eq!(payload.data_type, ProximaType::Json);
        assert_eq!(
            payload.properties.get("json_encoding").map(String::as_str),
            Some("jsonb")
        );

        let embedding = schema
            .columns
            .iter()
            .find(|column| column.name == "embedding")
            .expect("embedding column");
        assert_eq!(
            embedding.data_type,
            ProximaType::DenseVector {
                element: proximadb_data_model::VectorElement::Float32,
                dim: 0
            }
        );
        assert_eq!(
            embedding.properties.get("dimension").map(String::as_str),
            Some("384")
        );

        let indexes = catalog.list_indexes(&table_id).await.expect("list indexes");
        assert!(
            indexes
                .iter()
                .any(|index| index.index_type == CatalogIndexType::Gin)
        );
        assert!(
            indexes
                .iter()
                .any(|index| index.index_type == CatalogIndexType::Hnsw)
        );
    }

    #[tokio::test]
    async fn scoped_create_namespace_records_owning_tenant() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let ddl = DdlService::new(manager.clone());
        ddl.execute_scoped(
            DdlStatement::CreateNamespace {
                namespace: vec!["analytics".to_string()],
                if_not_exists: false,
                properties: HashMap::new(),
            },
            Some("acmecorp"),
        )
        .await
        .expect("tenant-scoped create namespace");

        let catalog = manager.default_catalog().await.expect("default catalog");
        let ns = catalog
            .get_namespace(&["acmecorp".to_string(), "analytics".to_string()])
            .await
            .expect("get tenant-scoped namespace");

        assert_eq!(ns.tenant_id.as_deref(), Some("acmecorp"));
        assert!(ns.namespace_id.is_some());
        assert!(ns.is_dr_addressable(), "namespace must be DR-addressable");
    }

    #[tokio::test]
    async fn scoped_drop_namespace_uses_tenant_prefixed_namespace() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let ddl = DdlService::new(manager.clone());
        ddl.execute_scoped(
            DdlStatement::CreateNamespace {
                namespace: vec!["analytics".to_string()],
                if_not_exists: false,
                properties: HashMap::new(),
            },
            Some("acmecorp"),
        )
        .await
        .expect("tenant-scoped create namespace");

        ddl.execute_scoped(
            DdlStatement::DropNamespace {
                namespace: vec!["analytics".to_string()],
                if_exists: false,
                cascade: false,
            },
            Some("acmecorp"),
        )
        .await
        .expect("tenant-scoped drop namespace");

        let catalog = manager.default_catalog().await.expect("default catalog");
        assert!(
            !catalog
                .namespace_exists(&["acmecorp".to_string(), "analytics".to_string()])
                .await
                .expect("namespace_exists"),
            "tenant-scoped DROP NAMESPACE must remove the tenant-prefixed namespace"
        );
    }

    #[tokio::test]
    async fn test_pgwire_table_options_shape_oltp_olap_htap_and_open_table_layouts() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let service = DdlService::new(manager.clone());
        let parser = crate::query::sql_frontend::SqlFrontendParser::new();
        let create_table = parser
            .parse_ddl(
                "CREATE TABLE facts (
                    id BIGINT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    payload JSONB,
                    embedding VECTOR(4)
                ) WITH (
                    workload = 'htap',
                    layout = 'pax',
                    compute_route = 'datafusion-local',
                    partitioning = 'tenant_id,bucket',
                    isolation = 'snapshot-isolation',
                    freshness_sla = '5s',
                    projection_freshness = 'bounded_lag',
                    projection_max_lag_ms = 250,
                    policy_boundary = 'engine-enforced',
                    open_table_format = 'iceberg',
                    ownership = 'projection',
                    location = 's3://warehouse/facts',
                    refresh = 'manual'
                );",
            )
            .expect("parse create table")
            .expect("ddl statement");

        service
            .execute(create_table)
            .await
            .expect("execute create table");

        let (catalog, table_id) = manager.resolve_table("facts").await.expect("resolve table");
        let schema = catalog.get_table(&table_id).await.expect("get schema");

        assert_eq!(schema.workload_profile, CatalogWorkloadProfile::Htap);
        assert_eq!(
            schema.storage_specialization,
            CatalogStorageSpecialization::PaxRowFamily
        );
        assert_eq!(
            schema.properties.get("compute_route").map(String::as_str),
            Some("datafusion-local")
        );
        assert_eq!(
            schema.properties.get("partitioning").map(String::as_str),
            Some("tenant_id,bucket")
        );
        assert_eq!(
            schema
                .relational_capabilities
                .transaction_profile
                .as_deref(),
            Some("snapshot-isolation")
        );
        assert_eq!(
            schema.properties.get("freshness_sla").map(String::as_str),
            Some("5s")
        );
        assert_eq!(
            schema.properties.get("policy_boundary").map(String::as_str),
            Some("engine-enforced")
        );
        assert!(
            schema
                .storage_layouts
                .iter()
                .any(|layout| layout.name == "primary"
                    && layout.layout_kind == CatalogStorageLayoutKind::Pax)
        );
        assert!(
            schema
                .storage_layouts
                .iter()
                .any(|layout| layout.name == "vector_ann"
                    && layout.layout_kind == CatalogStorageLayoutKind::VectorAnn)
        );
        assert!(
            schema
                .storage_layouts
                .iter()
                .any(|layout| layout.name == "json_path"
                    && layout.layout_kind == CatalogStorageLayoutKind::Columnar)
        );
        assert!(schema.storage_layouts.iter().any(|layout| {
            layout.name == "open_table"
                && layout.authority == CatalogAuthorityMode::ProjectionPublication
                && layout.physical_format == CatalogPhysicalFormat::Iceberg
                && layout.location.as_deref() == Some("s3://warehouse/facts")
        }));
        assert!(schema.projections.iter().any(|projection| {
            projection.name == "vector_ann"
                && projection.freshness == ProjectionFreshness::BoundedLag
                && projection.max_lag_ms == Some(250)
        }));
    }

    #[tokio::test]
    async fn test_pgwire_options_can_select_sst_for_legacy_vector_specialty() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("native", temp_dir.path().to_string_lossy().as_ref())
            .await
            .expect("native catalog");

        let service = DdlService::new(manager.clone());
        let parser = crate::query::sql_frontend::SqlFrontendParser::new();
        let create_table = parser
            .parse_ddl(
                "CREATE TABLE vec_sst (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR(8)
                ) WITH (
                    workload = 'vector',
                    storage_engine = 'sst'
                );",
            )
            .expect("parse create table")
            .expect("ddl statement");

        service
            .execute(create_table)
            .await
            .expect("execute create table");

        let (catalog, table_id) = manager
            .resolve_table("vec_sst")
            .await
            .expect("resolve table");
        let schema = catalog.get_table(&table_id).await.expect("get schema");

        assert_eq!(schema.workload_profile, CatalogWorkloadProfile::Vector);
        assert_eq!(
            schema.storage_specialization,
            CatalogStorageSpecialization::LsmWriteOptimized
        );
        assert!(schema.storage_layouts.iter().any(|layout| {
            layout.name == "primary"
                && layout.layout_kind == CatalogStorageLayoutKind::LsmRecord
                && layout.physical_format == CatalogPhysicalFormat::Sst
        }));
        assert!(
            schema
                .storage_layouts
                .iter()
                .any(|layout| layout.name == "vector_ann")
        );
    }

    // ── canonical_embedding_precision via WITH (...) properties ───────────────
    //
    // pgwire SQL clients write
    //   CREATE TABLE t (...) WITH (canonical_embedding_precision = 'fp16')
    // which lands on DdlStatement::CreateTable.properties. The handler
    // path tested below is what `apply_proto_enum_workarounds` is for
    // the REST/gRPC route, and what
    // `services::collection::manager.rs::catalog_schema_from_collection`
    // is for the proto CollectionConfig path.

    async fn create_with_precision_property(value: &str) -> proximadb_records::EmbeddingScalarType {
        let manager = Arc::new(CatalogManager::new());
        // Unique tempdir per test (SOLID test-hygiene mandate 17c): the four
        // precision-variant tests run as THREADS under plain `cargo test`, and
        // a shared /tmp catalog path made whoever created it second panic —
        // a schedule-sensitive flake. Leak the dir: the catalog outlives this fn.
        let tmp = tempfile::tempdir().expect("tempdir");
        let url = format!("file://{}", tmp.path().display());
        std::mem::forget(tmp);
        manager
            .create_native_catalog("default", &url)
            .await
            .expect("create catalog");
        let service = DdlService::new(manager.clone());

        let mut properties = HashMap::new();
        properties.insert(
            "canonical_embedding_precision".to_string(),
            value.to_string(),
        );

        let stmt = DdlStatement::CreateTable {
            table_name: format!(
                "ddl_precision_{}_{}",
                value.replace('-', "_"),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            columns: vec![DdlColumnDefinition {
                name: "id".to_string(),
                data_type: SqlDataType::BigInt,
                nullable: false,
                default_value: None,
                comment: None,
                primary_key: true,
            }],
            constraints: vec![],
            if_not_exists: false,
            properties,
        };

        let table_name = match &stmt {
            DdlStatement::CreateTable { table_name, .. } => table_name.clone(),
            _ => unreachable!(),
        };

        service.execute(stmt).await.expect("execute create table");

        let (catalog, table_id) = manager
            .resolve_table(&table_name)
            .await
            .expect("resolve table");
        catalog
            .get_table(&table_id)
            .await
            .expect("get schema")
            .canonical_embedding_precision
    }

    #[tokio::test]
    async fn create_table_with_canonical_embedding_precision_fp16() {
        assert_eq!(
            create_with_precision_property("fp16").await,
            proximadb_records::EmbeddingScalarType::Fp16
        );
    }

    #[tokio::test]
    async fn create_table_with_canonical_embedding_precision_bf16() {
        assert_eq!(
            create_with_precision_property("bf16").await,
            proximadb_records::EmbeddingScalarType::Bf16
        );
    }

    #[tokio::test]
    async fn create_table_with_canonical_embedding_precision_int8() {
        assert_eq!(
            create_with_precision_property("int8").await,
            proximadb_records::EmbeddingScalarType::Int8Scalar
        );
    }

    #[tokio::test]
    async fn create_table_with_canonical_embedding_precision_uint8() {
        assert_eq!(
            create_with_precision_property("uint8").await,
            proximadb_records::EmbeddingScalarType::UInt8Scalar
        );
    }

    #[tokio::test]
    async fn create_table_with_canonical_embedding_precision_screaming_label() {
        // The proto-generated SCREAMING form ("EMBEDDING_PRECISION_FP16")
        // should normalize the same way as "fp16".
        assert_eq!(
            create_with_precision_property("EMBEDDING_PRECISION_FP16").await,
            proximadb_records::EmbeddingScalarType::Fp16
        );
    }

    #[tokio::test]
    async fn create_table_with_canonical_embedding_precision_aliases() {
        // SDKs and users send a variety of shorthand — accept all.
        assert_eq!(
            create_with_precision_property("f16").await,
            proximadb_records::EmbeddingScalarType::Fp16
        );
        assert_eq!(
            create_with_precision_property("half").await,
            proximadb_records::EmbeddingScalarType::Fp16
        );
        assert_eq!(
            create_with_precision_property("float16").await,
            proximadb_records::EmbeddingScalarType::Fp16
        );
    }

    #[tokio::test]
    async fn create_table_unknown_precision_value_falls_back_to_fp32_silently() {
        // Don't fail CREATE TABLE on a typo — log a warn and use the
        // legacy fp32 path. This matches the proto enum behavior
        // (Unspecified → Fp32 default).
        assert_eq!(
            create_with_precision_property("bogus_precision").await,
            proximadb_records::EmbeddingScalarType::Fp32
        );
    }

    #[tokio::test]
    async fn create_table_without_precision_property_defaults_to_fp32() {
        // No WITH option set → legacy fp32 (no behavior change for
        // existing CREATE TABLE statements).
        let manager = Arc::new(CatalogManager::new());
        // Unique tempdir (mandate 17c) — see create_with_precision_property.
        let tmp = tempfile::tempdir().expect("tempdir");
        let url = format!("file://{}", tmp.path().display());
        std::mem::forget(tmp);
        manager
            .create_native_catalog("default", &url)
            .await
            .expect("create catalog");
        let service = DdlService::new(manager.clone());

        let stmt = DdlStatement::CreateTable {
            table_name: format!(
                "ddl_precision_default_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            columns: vec![DdlColumnDefinition {
                name: "id".to_string(),
                data_type: SqlDataType::BigInt,
                nullable: false,
                default_value: None,
                comment: None,
                primary_key: true,
            }],
            constraints: vec![],
            if_not_exists: false,
            properties: HashMap::new(),
        };

        let table_name = match &stmt {
            DdlStatement::CreateTable { table_name, .. } => table_name.clone(),
            _ => unreachable!(),
        };

        service.execute(stmt).await.expect("execute create table");

        let (catalog, table_id) = manager.resolve_table(&table_name).await.expect("resolve");
        let schema = catalog.get_table(&table_id).await.expect("get schema");
        assert_eq!(
            schema.canonical_embedding_precision,
            proximadb_records::EmbeddingScalarType::Fp32
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // CREATE / DROP RANK PROFILE (commit 4/5 of R-7c production wiring)
    // ─────────────────────────────────────────────────────────────────

    fn rank_pipeline_for_tests() -> (
        Arc<dyn crate::services::RankProfileStore>,
        Arc<crate::network::rest::canonical::rank::RankServices>,
    ) {
        use crate::core::search::hybrid::FusionStrategy;
        use crate::network::rest::canonical::rank::{
            HybridCoordinatorAdapter, MockRangeCandidateProvider, RankServices,
        };
        use crate::services::record_store::TableWalAppender;
        use crate::services::{CanonicalWalRankProfileStore, MemoryTableWalAppender};

        // In-memory rank-profile store backed by `MemoryTableWalAppender` so
        // the DDL service can install + remove profiles without a temp dir.
        let appender: Arc<dyn TableWalAppender> = Arc::new(MemoryTableWalAppender::new());
        let store: Arc<dyn crate::services::RankProfileStore> =
            Arc::new(CanonicalWalRankProfileStore::new(appender));

        // RankServices with a fixed-range candidate provider so the
        // `with_rank_services` builder has a working stub to attach. The
        // candidates aren't exercised by these DDL tests; only the registry
        // + blueprint factory matter.
        let candidates: Arc<dyn crate::network::rest::canonical::rank::CandidateProvider> =
            Arc::new(MockRangeCandidateProvider::default());
        // The HybridCoordinatorAdapter isn't used here but matches the
        // production wiring shape so the test stays close to production.
        let _adapter = Arc::new(HybridCoordinatorAdapter::new(
            FusionStrategy::ReciprocalRank { k: 60 },
            Arc::new(NoopHybridBackend),
        ));
        let services = Arc::new(RankServices::new(candidates));
        (store, services)
    }

    struct NoopHybridBackend;

    #[async_trait::async_trait]
    impl crate::network::rest::canonical::rank::HybridSearchBackend for NoopHybridBackend {
        async fn bm25_search(
            &self,
            _collection: &str,
            _query: &str,
        ) -> proximadb_rank_core::RankResult<Vec<crate::core::search::hybrid::BM25Result>> {
            Ok(Vec::new())
        }

        async fn vector_search(
            &self,
            _collection: &str,
            _vector: &[f32],
        ) -> proximadb_rank_core::RankResult<Vec<crate::core::search::hybrid::VectorResult>>
        {
            Ok(Vec::new())
        }
    }

    const VALID_PROFILE_TOML: &str = "[first_phase]\nexpression = \"1.0\"\nheap_size = 50\n";
    const BROKEN_PROFILE_TOML: &str = "[first_phase]\nexpression = \"definitely_not_a_feature(\\\"missing\\\")\"\nheap_size = 50\n";

    #[tokio::test]
    async fn create_rank_profile_installs_into_store_and_registry() {
        let (store, services) = rank_pipeline_for_tests();
        let ddl = DdlService::new(Arc::new(CatalogManager::new()))
            .with_rank_profile_store(store.clone())
            .with_rank_services(services.clone());

        let result = ddl
            .execute(DdlStatement::CreateRankProfile {
                name: "basic".to_string(),
                spec_toml: VALID_PROFILE_TOML.to_string(),
                if_not_exists: false,
            })
            .await
            .expect("CREATE RANK PROFILE should succeed");
        assert!(result.success);
        assert_eq!(result.message, "CREATE RANK PROFILE basic");

        assert!(store.get("basic").await.unwrap().is_some());
        assert!(services.profile_registry.get("basic").is_some());
    }

    #[tokio::test]
    async fn create_rank_profile_if_not_exists_is_idempotent() {
        let (store, services) = rank_pipeline_for_tests();
        let ddl = DdlService::new(Arc::new(CatalogManager::new()))
            .with_rank_profile_store(store.clone())
            .with_rank_services(services.clone());

        ddl.execute(DdlStatement::CreateRankProfile {
            name: "idempotent".to_string(),
            spec_toml: VALID_PROFILE_TOML.to_string(),
            if_not_exists: false,
        })
        .await
        .unwrap();
        let first = store.get("idempotent").await.unwrap().unwrap();

        let result = ddl
            .execute(DdlStatement::CreateRankProfile {
                name: "idempotent".to_string(),
                spec_toml: VALID_PROFILE_TOML.to_string(),
                if_not_exists: true,
            })
            .await
            .expect("IF NOT EXISTS should not error on existing");
        assert!(result.success);
        assert_eq!(result.affected_count, 0);

        let second = store.get("idempotent").await.unwrap().unwrap();
        assert_eq!(
            first.version, second.version,
            "IF NOT EXISTS must not bump version when the profile already exists"
        );
    }

    #[tokio::test]
    async fn create_rank_profile_rejects_uncompilable_spec_without_persisting() {
        let (store, services) = rank_pipeline_for_tests();
        let ddl = DdlService::new(Arc::new(CatalogManager::new()))
            .with_rank_profile_store(store.clone())
            .with_rank_services(services.clone());

        let err = ddl
            .execute(DdlStatement::CreateRankProfile {
                name: "broken".to_string(),
                spec_toml: BROKEN_PROFILE_TOML.to_string(),
                if_not_exists: false,
            })
            .await
            .expect_err("CREATE RANK PROFILE with unresolvable feature must fail");
        assert!(err.to_string().contains("compile failed"));

        // Catalog must NOT contain a partial install.
        assert!(
            store.get("broken").await.unwrap().is_none(),
            "uncompilable profile must not be written to the catalog"
        );
        assert!(services.profile_registry.get("broken").is_none());
    }

    #[tokio::test]
    async fn create_rank_profile_without_store_returns_clean_error() {
        let ddl = DdlService::new(Arc::new(CatalogManager::new()));
        let err = ddl
            .execute(DdlStatement::CreateRankProfile {
                name: "x".to_string(),
                spec_toml: VALID_PROFILE_TOML.to_string(),
                if_not_exists: false,
            })
            .await
            .expect_err("DDL service without rank store should reject the statement");
        assert!(err.to_string().contains("not configured"));
    }

    #[tokio::test]
    async fn drop_rank_profile_removes_from_store_and_registry() {
        let (store, services) = rank_pipeline_for_tests();
        let ddl = DdlService::new(Arc::new(CatalogManager::new()))
            .with_rank_profile_store(store.clone())
            .with_rank_services(services.clone());

        ddl.execute(DdlStatement::CreateRankProfile {
            name: "doomed".to_string(),
            spec_toml: VALID_PROFILE_TOML.to_string(),
            if_not_exists: false,
        })
        .await
        .unwrap();

        let result = ddl
            .execute(DdlStatement::DropRankProfile {
                name: "doomed".to_string(),
                if_exists: false,
            })
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.message, "DROP RANK PROFILE doomed");
        assert!(store.get("doomed").await.unwrap().is_none());
        assert!(services.profile_registry.get("doomed").is_none());
    }

    #[tokio::test]
    async fn drop_rank_profile_if_exists_is_noop_for_missing() {
        let (store, services) = rank_pipeline_for_tests();
        let ddl = DdlService::new(Arc::new(CatalogManager::new()))
            .with_rank_profile_store(store)
            .with_rank_services(services);

        let result = ddl
            .execute(DdlStatement::DropRankProfile {
                name: "ghost".to_string(),
                if_exists: true,
            })
            .await
            .expect("IF EXISTS must not error on missing profile");
        assert!(result.success);
        assert_eq!(result.affected_count, 0);
    }

    #[tokio::test]
    async fn drop_rank_profile_without_if_exists_errors_for_missing() {
        let (store, services) = rank_pipeline_for_tests();
        let ddl = DdlService::new(Arc::new(CatalogManager::new()))
            .with_rank_profile_store(store)
            .with_rank_services(services);

        let err = ddl
            .execute(DdlStatement::DropRankProfile {
                name: "ghost".to_string(),
                if_exists: false,
            })
            .await
            .expect_err("DROP without IF EXISTS must fail for missing profile");
        assert!(err.to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn create_function_registers_sql_bodied_scalar() {
        // F5 slice 3: CREATE FUNCTION lowers the body and registers a SQL-bodied scalar in the
        // shared registry — so the user function is then dispatchable (and runs on both engines).
        let ddl = DdlService::new(Arc::new(CatalogManager::new()));
        ddl.execute(DdlStatement::CreateFunction {
            name: "triple".to_string(),
            params: vec![("x".to_string(), ProximaType::Int64)],
            return_ty: ProximaType::Int64,
            body: "x * 3".to_string(),
            or_replace: false,
        })
        .await
        .expect("CREATE FUNCTION triple");

        use proximadb_data_model::ProximaValue;
        let def = proximadb_functions::builtins()
            .lookup_scalar("triple")
            .expect("triple registered");
        let out = (def.kernel)(&[ProximaValue::Int64(7)]).expect("triple eval");
        assert_eq!(out, ProximaValue::Int64(21));

        // Re-creating without OR REPLACE fails.
        assert!(
            ddl.execute(DdlStatement::CreateFunction {
                name: "triple".to_string(),
                params: vec![("x".to_string(), ProximaType::Int64)],
                return_ty: ProximaType::Int64,
                body: "x * 4".to_string(),
                or_replace: false,
            })
            .await
            .is_err()
        );
    }

    #[test]
    fn create_function_persists_to_durable_catalog() {
        // F5: with a durable catalog attached, CREATE FUNCTION persists the definition (so boot
        // recovery can re-register it) in addition to the live registration.
        //
        // Hang-prone under the multi-threaded `#[tokio::test]` runtime (CI worker-thread
        // shutdown race) — runs on an isolated single-threaded runtime with a bounded
        // timeout via `test_runtime` (see CLAUDE.md #11 streaming-hang flake).
        crate::query::execution::test_runtime::run_with_timeout(30, async {
            use crate::services::canonical_wal::FramedTableWalAppender;
            let dir = tempfile::tempdir().unwrap();
            let appender = Arc::new(
                FramedTableWalAppender::open(dir.path().join("fns.wal"))
                    .await
                    .unwrap(),
            );
            let store = Arc::new(crate::services::CanonicalWalFunctionStore::new(appender));
            let ddl =
                DdlService::new(Arc::new(CatalogManager::new())).with_function_store(store.clone());

            ddl.execute(DdlStatement::CreateFunction {
                name: "quad".to_string(),
                params: vec![("x".to_string(), ProximaType::Int64)],
                return_ty: ProximaType::Int64,
                body: "x * 4".to_string(),
                or_replace: false,
            })
            .await
            .expect("CREATE FUNCTION quad");

            use crate::services::FunctionStore;
            let persisted = store.list_all().await.unwrap();
            assert_eq!(persisted.len(), 1);
            assert_eq!(persisted[0].name, "quad");
            assert_eq!(persisted[0].body, "x * 4");
        });
    }
}
