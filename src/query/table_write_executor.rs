//! Routed table-write execution contract.
//!
//! `table_write_plan` decides which backend/access method should execute a
//! table-to-table write. This module owns the next boundary: taking a routed
//! plan and executing it through native record writers, DataFusion, or an
//! external open-table commit protocol. The first implementation is deliberately
//! planned-only so pgwire/DML can depend on a stable executor trait before the
//! concrete readers and writers are wired in.

use std::{
    fmt,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
#[cfg(feature = "abac-policy")]
use proximadb_abac::{ReadContext, SystemReadReason};
use proximadb_catalog::relational::{RelationalConstraintKind, RelationalConstraintViolation};
use proximadb_catalog::{
    CatalogPhysicalFormat, CatalogStorageLayout, CatalogTableSchema, ColumnConstraint,
};
use proximadb_records::ProximaRecord;
use proximadb_storage_common::object_store_bridge::{BridgeObjectPath as Path, ObjectStoreBridge};

use crate::query::table_write_plan::{
    ComputeBackend, ExecutionGuard, ReadSource, RoutedExecutionPlan, WriteMode,
};
use crate::services::WriteLane;
use crate::services::record_store::{
    TableRecordMutation, TableRecordMutationKind, TableRecordScanRequest, TableRecordStore,
};

/// Execution status for a routed table-write plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableWriteExecutionStatus {
    /// The plan was validated and routed, but no physical execution happened.
    PlannedOnly,
    /// The plan completed and committed through the selected writer.
    Completed,
}

/// Request passed to the table-write executor.
#[derive(Debug, Clone)]
pub struct TableWriteExecutionRequest<'a> {
    /// Resolved xCatalog target schema.
    pub target_schema: &'a CatalogTableSchema,
    /// Resolved xCatalog source schema when the source is another catalog table.
    pub source_schema: Option<&'a CatalogTableSchema>,
    /// Routed plan selected by `DmlWritePlanner` / `TableWriteRouter`.
    pub routed_plan: RoutedExecutionPlan,
    /// Tenant context for execution, used for path isolation and billing attribution.
    pub tenant_context: Option<&'a crate::storage::tenant::context::TenantContext>,
}

/// Result returned by a table-write executor implementation.
#[derive(Debug, Clone)]
pub struct TableWriteExecutionResult {
    /// Whether execution only reached the planned boundary or actually committed.
    pub status: TableWriteExecutionStatus,
    /// Number of rows committed. Planned-only executions report zero.
    pub rows_written: u64,
    /// Human-readable backend/access-method summary.
    pub route_summary: String,
    /// Guards validated or still required before physical execution.
    pub guards: Vec<ExecutionGuard>,
}

impl TableWriteExecutionResult {
    pub fn planned(routed_plan: &RoutedExecutionPlan) -> Self {
        Self {
            status: TableWriteExecutionStatus::PlannedOnly,
            rows_written: 0,
            route_summary: format!(
                "backend={:?}, access_method={:?}",
                routed_plan.backend, routed_plan.selected_path.access_method
            ),
            guards: routed_plan.required_guards.clone(),
        }
    }
}

/// Execution boundary for `INSERT ... SELECT`, `INSERT OVERWRITE`, CTAS, and MERGE.
#[async_trait]
pub trait TableWriteExecutor: Send + Sync {
    /// Execute a routed table-write plan.
    async fn execute(
        &self,
        request: TableWriteExecutionRequest<'_>,
    ) -> Result<TableWriteExecutionResult>;
}

/// Structured source-reader failure class.
///
/// Keep the class name in user-facing errors until protocol error payloads can
/// carry structured codes end to end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableRecordSourceErrorClass {
    RetryableReadFailure,
    PermanentReadFailure,
    PolicyViolation,
    TypeViolation,
    StaleSnapshot,
}

impl fmt::Display for TableRecordSourceErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Error returned by source readers before data becomes canonical records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRecordSourceError {
    pub class: TableRecordSourceErrorClass,
    pub message: String,
}

impl TableRecordSourceError {
    pub fn new(class: TableRecordSourceErrorClass, message: impl Into<String>) -> Self {
        Self {
            class,
            message: message.into(),
        }
    }
}

impl fmt::Display for TableRecordSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class, self.message)
    }
}

impl std::error::Error for TableRecordSourceError {}

/// Canonical source reader used by native table-write execution.
///
/// DataFusion, native table scans, Arrow Flight, and external file readers can
/// all adapt into this trait by yielding catalog-validated `ProximaRecord`
/// batches. The executor owns write semantics; source adapters only read.
#[async_trait]
pub trait TableRecordSourceReader: Send + Sync {
    /// Read the next source batch. `Ok(None)` means end-of-stream.
    async fn next_batch(
        &self,
        source: &ReadSource,
        source_schema: Option<&CatalogTableSchema>,
        target_schema: &CatalogTableSchema,
        tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
        cursor: &mut TableRecordSourceCursor,
    ) -> Result<Option<Vec<ProximaRecord>>>;
}

/// Per-execution source cursor.
///
/// Source readers are shared services, so executor-local cursor state prevents
/// one `INSERT ... SELECT` execution from consuming state belonging to another.
#[derive(Debug, Default)]
pub struct TableRecordSourceCursor {
    buffered_records: Option<Vec<ProximaRecord>>,
    offset: usize,
}

impl TableRecordSourceCursor {
    fn take_next(&mut self, batch_size: usize) -> Option<Vec<ProximaRecord>> {
        let records = self.buffered_records.as_ref()?;
        if self.offset >= records.len() {
            return None;
        }

        let end = self.offset.saturating_add(batch_size).min(records.len());
        let batch = records[self.offset..end].to_vec();
        self.offset = end;
        Some(batch)
    }
}

/// Native source reader backed by the canonical `TableRecordStore` scan method.
///
/// This reader supports catalog-table sources. General `SELECT` queries with
/// projection/filter/join/aggregate semantics should route through DataFusion
/// or another compute adapter and yield canonical record batches from there.
pub struct TableRecordStoreSourceReader {
    record_store: Arc<dyn TableRecordStore>,
    batch_size: usize,
}

impl TableRecordStoreSourceReader {
    pub fn new(record_store: Arc<dyn TableRecordStore>) -> Self {
        Self::with_batch_size(record_store, 1024)
    }

    pub fn with_batch_size(record_store: Arc<dyn TableRecordStore>, batch_size: usize) -> Self {
        Self {
            record_store,
            batch_size: batch_size.max(1),
        }
    }
}

#[async_trait]
impl TableRecordSourceReader for TableRecordStoreSourceReader {
    async fn next_batch(
        &self,
        source: &ReadSource,
        source_schema: Option<&CatalogTableSchema>,
        target_schema: &CatalogTableSchema,
        tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
        cursor: &mut TableRecordSourceCursor,
    ) -> Result<Option<Vec<ProximaRecord>>> {
        let ReadSource::CatalogTable { table, .. } = source else {
            return Err(TableRecordSourceError::new(
                TableRecordSourceErrorClass::PermanentReadFailure,
                format!(
                    "Native table-write source reader supports catalog-table sources only; source {:?} requires a compute adapter",
                    source
                ),
            )
            .into());
        };
        let scan_schema = source_schema.unwrap_or(target_schema);
        if scan_schema.name != table.name {
            return Err(TableRecordSourceError::new(
                TableRecordSourceErrorClass::TypeViolation,
                format!(
                    "Catalog-table source '{}' does not match resolved schema '{}'",
                    table.qualified_name(),
                    scan_schema.name
                ),
            )
            .into());
        }

        if cursor.buffered_records.is_none() {
            let records = self
                .record_store
                .scan_records(
                    scan_schema,
                    TableRecordScanRequest {
                        filter: None,
                        table_id: table.qualified_name(),
                        limit: None,
                        include_vector: true,
                        include_props: true,
                    },
                    // TD-113 family: scope the SELECT-source scan to the tenant's
                    // record partition (was `None` → unscoped/cross-tenant read).
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(
                        SystemReadReason::Statistics,
                        "table_write_executor [CLIENT-PLACEHOLDER]",
                    ),
                )
                .await?;
            cursor.buffered_records = Some(records);
        }

        Ok(cursor.take_next(self.batch_size))
    }
}

/// Resolved parent-table information for FOREIGN KEY enforcement on the
/// INSERT-SELECT / native write path.
#[derive(Debug, Clone)]
pub struct ResolvedParentTable {
    /// The parent table's catalog schema (carries its primary key + columns).
    pub schema: CatalogTableSchema,
    /// Physical table id used to address the parent in the record store.
    pub table_id_name: String,
}

/// Catalog lookup port used by [`NativeTableWriteExecutor`] to resolve the
/// parent tables referenced by FOREIGN KEY constraints.
///
/// The native executor has no catalog handle of its own, so DmlService backs
/// this with its `CatalogManager`; tests can supply an in-memory stub. This
/// mirrors `DmlService::enforce_foreign_keys` so every write path enforces FK
/// references identically (TD-110).
#[async_trait]
pub trait ParentTableResolver: Send + Sync {
    /// Resolve the parent table named by a FOREIGN KEY's `REFERENCES <table>`.
    ///
    /// `Ok(None)` means the parent table does not exist (a reference violation);
    /// `Err` means resolution itself failed (misconfigured catalog).
    async fn resolve_parent_table(
        &self,
        references_table: &str,
        tenant: Option<&str>,
    ) -> Result<Option<ResolvedParentTable>>;
}

/// Native executor that commits canonical source batches through `TableRecordStore`.
pub struct NativeTableWriteExecutor {
    source_reader: Arc<dyn TableRecordSourceReader>,
    record_store: Arc<dyn TableRecordStore>,
    parent_table_resolver: Option<Arc<dyn ParentTableResolver>>,
}

impl NativeTableWriteExecutor {
    pub fn new(
        source_reader: Arc<dyn TableRecordSourceReader>,
        record_store: Arc<dyn TableRecordStore>,
    ) -> Self {
        Self {
            source_reader,
            record_store,
            parent_table_resolver: None,
        }
    }

    /// Enable FOREIGN KEY enforcement on the native write path by supplying a
    /// catalog lookup port. Without it, FK constraints are not checked here
    /// (the row-local catalog validator no longer fails them closed).
    pub fn with_parent_table_resolver(mut self, resolver: Arc<dyn ParentTableResolver>) -> Self {
        self.parent_table_resolver = Some(resolver);
        self
    }
}

#[async_trait]
impl TableWriteExecutor for NativeTableWriteExecutor {
    async fn execute(
        &self,
        request: TableWriteExecutionRequest<'_>,
    ) -> Result<TableWriteExecutionResult> {
        validate_required_guards(request.target_schema, &request.routed_plan)?;
        if request.routed_plan.backend != ComputeBackend::Native {
            return Ok(TableWriteExecutionResult::planned(&request.routed_plan));
        }
        validate_native_write_lane(request.target_schema, &request.routed_plan)?;

        let mutation_kind = mutation_kind_for_write_mode(&request.routed_plan.plan.write_mode)?;
        let mut rows_written = 0;
        let mut cursor = TableRecordSourceCursor::default();

        while let Some(batch) = self
            .source_reader
            .next_batch(
                &request.routed_plan.plan.source,
                request.source_schema,
                request.target_schema,
                request.tenant_context,
                &mut cursor,
            )
            .await?
        {
            if batch.is_empty() {
                continue;
            }
            // TD-110: enforce non-PK UNIQUE on the INSERT-SELECT / native path.
            // Within-batch dedup + cross-existing probe per unique set; cross-BATCH
            // duplicates are caught because each prior batch is committed before the
            // next batch's probe. (PK duplicates are rejected by write_mutations'
            // Insert-conflict check.)
            let primary_key =
                crate::services::record_store::schema_oid_key_column(request.target_schema);
            let candidate_sets = crate::services::record_store::build_unique_candidate_sets(
                request.target_schema,
                &batch,
                primary_key.as_deref(),
            )?;
            if !candidate_sets.is_empty()
                && let Some(conflict) = self
                    .record_store
                    .check_unique_conflict(
                        request.target_schema,
                        &request.target_schema.name,
                        primary_key.as_deref(),
                        &candidate_sets,
                        &std::collections::HashSet::new(),
                        // The rows land in the writing tenant's partition; probe it.
                        request.tenant_context,
                    )
                    .await?
            {
                return Err(RelationalConstraintViolation::new(
                    RelationalConstraintKind::Unique,
                    format!(
                        "duplicate key value violates unique constraint on ({}) for table '{}': ({}) already exists",
                        conflict.columns.join(", "),
                        request.target_schema.name,
                        conflict.tuple.join(", ")
                    ),
                )
                .into());
            }
            // TD-110: enforce FOREIGN KEY references on the INSERT-SELECT / native
            // path. Requires a catalog lookup port to resolve parent tables; when
            // unset (e.g. planned-only test wiring), FK is not checked here.
            if let Some(resolver) = self.parent_table_resolver.as_ref() {
                enforce_foreign_keys_for_batch(
                    resolver,
                    &self.record_store,
                    request.target_schema,
                    &batch,
                    request.tenant_context,
                )
                .await?;
            }
            let mutations = batch
                .into_iter()
                .map(|record| TableRecordMutation::new(mutation_kind, record))
                .collect::<Vec<_>>();
            // TD-113 family: thread the tenant so the bulk-append lands in the
            // tenant's record partition (was `None` → unscoped/cross-tenant write).
            let result = self
                .record_store
                .write_mutations(request.target_schema, mutations, request.tenant_context)
                .await?;
            if !result.success {
                return Err(anyhow!(
                    "Native table write failed for '{}': {:?}",
                    request.target_schema.name,
                    result.errors
                ));
            }
            rows_written += result.record_ids.len() as u64;
        }

        Ok(TableWriteExecutionResult {
            status: TableWriteExecutionStatus::Completed,
            rows_written,
            route_summary: format!(
                "backend={:?}, access_method={:?}",
                request.routed_plan.backend, request.routed_plan.selected_path.access_method
            ),
            guards: request.routed_plan.required_guards,
        })
    }
}

/// Planned-only executor used until concrete DataFusion/native/open-table paths exist.
#[derive(Debug, Default)]
pub struct PlannedOnlyTableWriteExecutor;

impl PlannedOnlyTableWriteExecutor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TableWriteExecutor for PlannedOnlyTableWriteExecutor {
    async fn execute(
        &self,
        request: TableWriteExecutionRequest<'_>,
    ) -> Result<TableWriteExecutionResult> {
        validate_required_guards(request.target_schema, &request.routed_plan)?;
        Ok(TableWriteExecutionResult::planned(&request.routed_plan))
    }
}

/// Enforce FOREIGN KEY references for one source batch against parent tables
/// in the same partition. Mirrors `DmlService::enforce_foreign_keys` through
/// the shared resolver (`services::referential`): an FK (single- or
/// multi-column) referencing the parent's declared key or one of its UNIQUE
/// sets, in full and in order. NULL FK values are exempt (MATCH SIMPLE); any
/// other referenced column set is cleanly rejected rather than silently
/// accepted.
async fn enforce_foreign_keys_for_batch(
    resolver: &Arc<dyn ParentTableResolver>,
    record_store: &Arc<dyn TableRecordStore>,
    target_schema: &CatalogTableSchema,
    batch: &[ProximaRecord],
    tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
) -> Result<()> {
    for constraint in &target_schema.relational_capabilities.constraints {
        let ColumnConstraint::ForeignKey {
            columns,
            references_table,
            references_columns,
            ..
        } = constraint
        else {
            continue;
        };
        if columns.len() != references_columns.len() {
            return Err(anyhow!(
                "FOREIGN KEY ({}) on table '{}' has mismatched column counts ({} local vs {} referenced)",
                columns.join(", "),
                target_schema.name,
                columns.len(),
                references_columns.len()
            ));
        }

        let Some(parent) = resolver
            .resolve_parent_table(
                references_table,
                tenant_context.map(|tc| tc.tenant_id.as_str()),
            )
            .await?
        else {
            return Err(anyhow!(
                "FOREIGN KEY ({}) on table '{}' references missing table '{}'",
                columns.join(", "),
                target_schema.name,
                references_table
            ));
        };
        // TD-RELIDEQ-2: the parent's declared key (its oid) or a UNIQUE set —
        // the same resolution the DML path uses.
        let Some(referenced_key) = crate::services::referential::resolve_referenced_key(
            &parent.schema,
            references_columns,
        ) else {
            return Err(anyhow!(
                "FOREIGN KEY ({}) REFERENCES {}({}) on table '{}' must reference the parent's primary key or a UNIQUE constraint, in full and in order",
                columns.join(", "),
                references_table,
                references_columns.join(", "),
                target_schema.name
            ));
        };

        for record in batch {
            // MATCH SIMPLE: a NULL/absent FK column → no reference required.
            let Some(values) = crate::services::referential::record_key_values(record, columns)
            else {
                continue;
            };
            let referenced_exists = crate::services::referential::referenced_parent_exists(
                record_store.as_ref(),
                &parent.schema,
                &parent.table_id_name,
                referenced_key,
                references_columns,
                &values,
                // The parent row is looked up in the writing tenant's partition.
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(
                    SystemReadReason::ForeignKeyResolution,
                    "table_write_executor",
                ),
            )
            .await?;
            if !referenced_exists {
                return Err(RelationalConstraintViolation::new(
                    RelationalConstraintKind::ForeignKey,
                    format!(
                        "FOREIGN KEY ({}) on table '{}' violates reference: ({}) is not present in {}({})",
                        columns.join(", "),
                        target_schema.name,
                        values
                            .iter()
                            .map(crate::services::record_store::proxima_value_to_unique_text)
                            .collect::<Vec<_>>()
                            .join(", "),
                        references_table,
                        references_columns.join(", ")
                    ),
                )
                .into());
            }
        }
    }
    Ok(())
}

fn mutation_kind_for_write_mode(write_mode: &WriteMode) -> Result<TableRecordMutationKind> {
    match write_mode {
        WriteMode::Append | WriteMode::InsertOnly => Ok(TableRecordMutationKind::Insert),
        WriteMode::Upsert => Ok(TableRecordMutationKind::Upsert),
        WriteMode::OverwriteTable => Ok(TableRecordMutationKind::OverwriteSnapshot),
        WriteMode::ReplacePartitions(_) => Ok(TableRecordMutationKind::ReplacePartitions),
        WriteMode::Merge => Err(anyhow!(
            "MERGE table-write execution requires merge predicate support"
        )),
    }
}

fn validate_required_guards(
    target_schema: &CatalogTableSchema,
    routed_plan: &RoutedExecutionPlan,
) -> Result<()> {
    let guards = &routed_plan.required_guards;
    for required in [
        ExecutionGuard::PinSourceSnapshot,
        ExecutionGuard::CheckTargetWriteCapabilities,
    ] {
        if !guards.contains(&required) {
            return Err(anyhow!(
                "Routed table-write plan for '{}' is missing required guard {:?}",
                target_schema.name,
                required
            ));
        }
    }
    Ok(())
}

fn validate_native_write_lane(
    target_schema: &CatalogTableSchema,
    routed_plan: &RoutedExecutionPlan,
) -> Result<()> {
    if routed_plan.write_lane_decision.lane == WriteLane::WalCurrentState {
        return Ok(());
    }

    Err(anyhow!(
        "Native table-write executor for '{}' can commit {:?} only after the dedicated {:?} commit protocol is implemented",
        target_schema.name,
        routed_plan.plan.write_mode,
        routed_plan.write_lane_decision.lane
    ))
}

fn is_datafusion_backend(backend: &ComputeBackend) -> bool {
    matches!(
        backend,
        ComputeBackend::DataFusionLocal | ComputeBackend::DataFusionDistributed
    )
}

fn validate_datafusion_write_lane(
    target_schema: &CatalogTableSchema,
    routed_plan: &RoutedExecutionPlan,
) -> Result<()> {
    if matches!(
        routed_plan.write_lane_decision.lane,
        WriteLane::WalCurrentState | WriteLane::BulkAppendCommit
    ) {
        return Ok(());
    }

    Err(anyhow!(
        "DataFusion table-write executor for '{}' cannot commit {:?} through {:?}",
        target_schema.name,
        routed_plan.plan.write_mode,
        routed_plan.write_lane_decision.lane
    ))
}

fn primary_layout(schema: &CatalogTableSchema) -> Option<&CatalogStorageLayout> {
    schema
        .storage_layouts
        .iter()
        .rev()
        .find(|layout| layout.name == "primary")
        .or_else(|| schema.storage_layouts.first())
}

fn object_write_base_path(schema: &CatalogTableSchema, tenant: Option<&str>) -> String {
    primary_layout(schema)
        .and_then(|layout| match layout.physical_format {
            CatalogPhysicalFormat::Iceberg | CatalogPhysicalFormat::Parquet => {
                layout.location.as_deref()
            }
            _ => None,
        })
        .or(schema.location.as_deref())
        .map(normalize_object_path_prefix)
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| {
            // No explicit (materialize-set, already tenant-scoped) location → derive a
            // fallback. Tenant-scope it (TD-113 family) so two tenants' same-named
            // tables don't write/commit to a shared `tables/{name}` prefix. Route
            // through DrPathBuilder (not a raw `data/{..}` literal) so the segments
            // are validated and the path-resolver guard is satisfied; same shape
            // (`data/{tenant}/tables/{name}`).
            let table = sanitize_object_path_segment(&schema.name);
            match tenant.filter(|t| !t.is_empty()) {
                Some(t) => {
                    crate::storage::trait_components::path_resolver::DrPathBuilder::build_from_parts(
                        t,
                        "tables",
                        &table,
                        Default::default(),
                    )
                    .map(|resolved| resolved.root_prefix().trim_end_matches('/').to_string())
                    .unwrap_or_else(|_| format!("tables/{table}"))
                }
                None => format!("tables/{table}"),
            }
        })
}

fn normalize_object_path_prefix(location: &str) -> String {
    let without_scheme = location
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(location);
    without_scheme.trim_matches('/').to_string()
}

fn sanitize_object_path_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn write_mode_label(write_mode: &WriteMode) -> &'static str {
    match write_mode {
        WriteMode::Append => "append",
        WriteMode::InsertOnly => "insert-only",
        WriteMode::Upsert => "upsert",
        WriteMode::OverwriteTable => "overwrite",
        WriteMode::ReplacePartitions(_) => "replace-partitions",
        WriteMode::Merge => "merge",
    }
}

fn object_write_path(
    schema: &CatalogTableSchema,
    routed_plan: &RoutedExecutionPlan,
    batch_index: usize,
    tenant: Option<&str>,
) -> Path {
    let base = object_write_base_path(schema, tenant);
    let table = sanitize_object_path_segment(&schema.name);
    let sequence = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Path::from(format!(
        "{base}/data/{table}-{}-{sequence}-{batch_index:05}.parquet",
        write_mode_label(&routed_plan.plan.write_mode)
    ))
}

/// Upper bound on optimistic-concurrency manifest-commit retries. A healthy
/// system rebases past a handful of concurrent committers within a few attempts;
/// hitting this ceiling means the conflict is persistent (a wedged committer or a
/// `latest` that never lets this writer win the CAS), which we surface as an error
/// rather than spinning forever.
const MAX_MANIFEST_COMMIT_ATTEMPTS: usize = 32;

/// DataFusion-based executor for OLAP/table-to-table write workloads.
///
/// This executor materializes query output into `ProximaRecord` batches and
/// commits through `TableRecordStore`. It is not the direct authority for
/// PostgreSQL-style OLTP writes; constraint-sensitive mutations must still pass
/// through the native WAL/row-delta commit path selected by xCatalog routing.
pub struct DataFusionTableWriteExecutor {
    source_reader: Arc<dyn TableRecordSourceReader>,
    record_store: Arc<dyn TableRecordStore>,
    object_store_bridge: Option<Arc<dyn ObjectStoreBridge>>,
}

impl DataFusionTableWriteExecutor {
    pub fn new(
        source_reader: Arc<dyn TableRecordSourceReader>,
        record_store: Arc<dyn TableRecordStore>,
    ) -> Self {
        Self {
            source_reader,
            record_store,
            object_store_bridge: None,
        }
    }

    pub fn with_object_store_bridge(mut self, bridge: Arc<dyn ObjectStoreBridge>) -> Self {
        self.object_store_bridge = Some(bridge);
        self
    }
}

#[async_trait]
impl TableWriteExecutor for DataFusionTableWriteExecutor {
    async fn execute(
        &self,
        request: TableWriteExecutionRequest<'_>,
    ) -> Result<TableWriteExecutionResult> {
        use proximadb_storage_common::object_store_bridge::CommitOutcome;

        validate_required_guards(request.target_schema, &request.routed_plan)?;
        if !is_datafusion_backend(&request.routed_plan.backend) {
            return Ok(TableWriteExecutionResult::planned(&request.routed_plan));
        }
        validate_datafusion_write_lane(request.target_schema, &request.routed_plan)?;

        let mutation_kind = mutation_kind_for_write_mode(&request.routed_plan.plan.write_mode)?;
        let mut rows_written = 0;
        let mut cursor = TableRecordSourceCursor::default();
        let is_bulk_append =
            request.routed_plan.write_lane_decision.lane == WriteLane::BulkAppendCommit;
        let mut batch_index = 0;
        let mut wrote_any_parquet = false;

        while let Some(batch) = self
            .source_reader
            .next_batch(
                &request.routed_plan.plan.source,
                request.source_schema,
                request.target_schema,
                request.tenant_context,
                &mut cursor,
            )
            .await?
        {
            if batch.is_empty() {
                continue;
            }
            if is_bulk_append {
                let path = object_write_path(
                    request.target_schema,
                    &request.routed_plan,
                    batch_index,
                    request.tenant_context.map(|tc| tc.tenant_id.as_str()),
                );
                batch_index += 1;

                let bridge = self.object_store_bridge.as_ref().ok_or_else(|| {
                    anyhow!(
                        "DataFusion bulk append for '{}' requires an ObjectStoreBridge",
                        request.target_schema.name
                    )
                })?;
                let tenant_id = request.tenant_context.map(|tc| tc.tenant_id.as_str());
                bridge
                    .write_records_to_parquet(&path, &batch, tenant_id)
                    .await
                    .map_err(|err| {
                        anyhow!(
                            "DataFusion Parquet write failed for '{}' at '{}': {err}",
                            request.target_schema.name,
                            path
                        )
                    })?;
                rows_written += batch.len() as u64;
                wrote_any_parquet = true;
                continue;
            }

            let mutations = batch
                .into_iter()
                .map(|record| TableRecordMutation::new(mutation_kind, record))
                .collect::<Vec<_>>();
            // TD-113 family: thread the tenant so the non-bulk-append DataFusion
            // route writes into the tenant's record partition (was `None`).
            let result = self
                .record_store
                .write_mutations(request.target_schema, mutations, request.tenant_context)
                .await?;
            if !result.success {
                return Err(anyhow!(
                    "DataFusion table write failed for '{}': {:?}",
                    request.target_schema.name,
                    result.errors
                ));
            }
            rows_written += result.record_ids.len() as u64;
        }

        if is_bulk_append && wrote_any_parquet {
            let bridge = self.object_store_bridge.as_ref().ok_or_else(|| {
                anyhow!(
                    "DataFusion bulk append for '{}' wrote Parquet without an ObjectStoreBridge",
                    request.target_schema.name
                )
            })?;
            let base = object_write_base_path(
                request.target_schema,
                request.tenant_context.map(|tc| tc.tenant_id.as_str()),
            );
            let data_prefix = format!("{base}/data");
            let manifest_prefix = format!("{base}/_manifests");

            // Optimistic-concurrency commit: rebase onto the latest snapshot and retry
            // on conflict, bounded so a persistent conflict surfaces as an error instead
            // of spinning forever.
            let mut parent = bridge.latest_manifest_version(&manifest_prefix).await?;
            let mut committed = false;
            for _ in 0..MAX_MANIFEST_COMMIT_ATTEMPTS {
                match bridge
                    .publish_snapshot(&Path::from(data_prefix.as_str()), &manifest_prefix, parent)
                    .await?
                {
                    CommitOutcome::Committed(_) => {
                        committed = true;
                        break;
                    }
                    CommitOutcome::Conflict { latest } => parent = latest,
                }
            }
            if !committed {
                return Err(anyhow!(
                    "DataFusion manifest commit for '{}' failed after {} attempts due to \
                     persistent snapshot conflict",
                    request.target_schema.name,
                    MAX_MANIFEST_COMMIT_ATTEMPTS
                ));
            }
        }

        Ok(TableWriteExecutionResult {
            status: TableWriteExecutionStatus::Completed,
            rows_written,
            route_summary: format!(
                "backend={:?}, access_method={:?}, batches={}",
                request.routed_plan.backend,
                request.routed_plan.selected_path.access_method,
                batch_index
            ),
            guards: request.routed_plan.required_guards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_write_base_path_tenant_isolates_the_fallback() {
        // No explicit location → fallback; the tenant scopes it so two tenants'
        // same-named tables don't share a `tables/{name}` prefix (TD-113 family).
        let schema = CatalogTableSchema::new("facts");
        assert_eq!(object_write_base_path(&schema, None), "tables/facts");
        assert_eq!(
            object_write_base_path(&schema, Some("acmecorp")),
            "data/acmecorp/tables/facts"
        );
        assert_ne!(
            object_write_base_path(&schema, Some("acmecorp")),
            object_write_base_path(&schema, Some("globexco")),
        );
    }

    use crate::query::table_write_plan::{
        CopyIntoPlan, CostEstimate, LogicalTableRef, TableWriteRouter, WriteIntentOverrides,
        WriteMode,
    };
    use crate::services::operations::OperationMetrics;
    use crate::services::record_store::{
        TableRecordGetRequest, TableRecordGetResponse, TableRecordScanRequest,
        TableRecordWriteResult,
    };
    use crate::services::{DEFAULT_BULK_BYTES_THRESHOLD, DEFAULT_BULK_ROW_THRESHOLD};
    use crate::storage::tenant::context::TenantContext;
    use arrow_array::RecordBatch;
    use arrow_schema::Schema as ArrowSchema;
    use futures::stream::BoxStream;
    use proximadb_catalog::{
        CatalogPhysicalFormat, CatalogStorageLayout, CatalogStorageSpecialization,
        CatalogTableSchema, CatalogWorkloadProfile, ColumnConstraint,
    };
    use proximadb_kernel::error::StorageError;
    use proximadb_storage_common::object_store_bridge::{
        BridgeInMemoryObjectStore as InMemory, BridgeObjectStore as ObjectStore, ObjectStoreBridge,
    };
    use std::sync::Mutex;

    struct VecSourceReader {
        batches: Mutex<Vec<Vec<ProximaRecord>>>,
    }

    impl VecSourceReader {
        fn new(mut batches: Vec<Vec<ProximaRecord>>) -> Self {
            batches.reverse();
            Self {
                batches: Mutex::new(batches),
            }
        }
    }

    #[async_trait]
    impl TableRecordSourceReader for VecSourceReader {
        async fn next_batch(
            &self,
            _source: &ReadSource,
            _source_schema: Option<&CatalogTableSchema>,
            _target_schema: &CatalogTableSchema,
            _tenant_context: Option<&crate::storage::tenant::context::TenantContext>,
            _cursor: &mut TableRecordSourceCursor,
        ) -> Result<Option<Vec<ProximaRecord>>> {
            Ok(self.batches.lock().unwrap().pop())
        }
    }

    struct CapturingRecordStore {
        writes: Mutex<Vec<TableRecordMutation>>,
    }

    #[async_trait]
    impl TableRecordStore for CapturingRecordStore {
        async fn write_mutations(
            &self,
            _table_schema: &CatalogTableSchema,
            mutations: Vec<TableRecordMutation>,
            _tenant_context: Option<&TenantContext>,
        ) -> Result<TableRecordWriteResult> {
            let ids = mutations
                .iter()
                .map(|mutation| mutation.record.oid.clone())
                .collect::<Vec<_>>();
            self.writes.lock().unwrap().extend(mutations);
            Ok(TableRecordWriteResult {
                success: true,
                record_ids: ids,
                metrics: OperationMetrics::default(),
                errors: Vec::new(),
                error_code: None,
            })
        }

        async fn get_by_key(
            &self,
            _table_schema: &CatalogTableSchema,
            _request: TableRecordGetRequest,
            _tenant_context: Option<&TenantContext>,
            #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
        ) -> Result<TableRecordGetResponse> {
            Ok(None)
        }
    }

    struct CapturingObjectStoreBridge {
        store: Arc<dyn ObjectStore>,
        writes: Mutex<Vec<(Path, Vec<String>)>>,
        commits: Mutex<Vec<(Path, String, Option<u64>)>>,
        /// When set, `publish_snapshot` always reports a conflict (recording each
        /// attempt in `commits`) so tests can exercise the bounded-retry ceiling.
        always_conflict: bool,
    }

    impl CapturingObjectStoreBridge {
        fn new() -> Self {
            Self {
                store: Arc::new(InMemory::new()),
                writes: Mutex::new(Vec::new()),
                commits: Mutex::new(Vec::new()),
                always_conflict: false,
            }
        }

        fn new_always_conflict() -> Self {
            Self {
                always_conflict: true,
                ..Self::new()
            }
        }
    }

    #[async_trait]
    impl ObjectStoreBridge for CapturingObjectStoreBridge {
        fn inner_store(&self) -> Arc<dyn ObjectStore> {
            self.store.clone()
        }

        async fn read_parquet_batches(
            &self,
            _path: &Path,
            _schema: Arc<ArrowSchema>,
            _batch_size: usize,
            _tenant_id: Option<&str>,
        ) -> std::result::Result<
            BoxStream<'static, std::result::Result<RecordBatch, StorageError>>,
            StorageError,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }

        async fn write_records_to_parquet(
            &self,
            path: &Path,
            records: &[ProximaRecord],
            _tenant_id: Option<&str>,
        ) -> std::result::Result<(), StorageError> {
            self.writes.lock().unwrap().push((
                path.clone(),
                records.iter().map(|record| record.oid.clone()).collect(),
            ));
            Ok(())
        }

        async fn fetch_vector_segment(
            &self,
            _path: &Path,
            _tenant_id: Option<&str>,
        ) -> std::result::Result<Vec<u8>, StorageError> {
            Ok(Vec::new())
        }

        async fn persist_vector_segment(
            &self,
            _path: &Path,
            _data: &[u8],
            _tenant_id: Option<&str>,
        ) -> std::result::Result<(), StorageError> {
            Ok(())
        }

        async fn latest_manifest_version(
            &self,
            _manifest_prefix: &str,
        ) -> std::result::Result<Option<u64>, StorageError> {
            Ok(self.commits.lock().unwrap().last().and_then(|c| c.2))
        }

        async fn publish_snapshot(
            &self,
            data_prefix: &Path,
            manifest_prefix: &str,
            parent: Option<u64>,
        ) -> std::result::Result<
            proximadb_storage_common::object_store_bridge::CommitOutcome,
            StorageError,
        > {
            use proximadb_storage_common::object_store_bridge::CommitOutcome;
            if self.always_conflict {
                self.commits.lock().unwrap().push((
                    data_prefix.clone(),
                    manifest_prefix.to_string(),
                    parent,
                ));
                return Ok(CommitOutcome::Conflict { latest: parent });
            }
            let next = parent.map(|p| p + 1).unwrap_or(0);
            self.commits.lock().unwrap().push((
                data_prefix.clone(),
                manifest_prefix.to_string(),
                Some(next),
            ));
            Ok(CommitOutcome::Committed(next))
        }
    }

    struct ScanRecordStore {
        records: Vec<ProximaRecord>,
    }

    #[async_trait]
    impl TableRecordStore for ScanRecordStore {
        async fn write_mutations(
            &self,
            _table_schema: &CatalogTableSchema,
            _mutations: Vec<TableRecordMutation>,
            _tenant_context: Option<&TenantContext>,
        ) -> Result<TableRecordWriteResult> {
            Ok(TableRecordWriteResult {
                success: true,
                record_ids: Vec::new(),
                metrics: OperationMetrics::default(),
                errors: Vec::new(),
                error_code: None,
            })
        }

        async fn get_by_key(
            &self,
            _table_schema: &CatalogTableSchema,
            _request: TableRecordGetRequest,
            _tenant_context: Option<&TenantContext>,
            #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
        ) -> Result<TableRecordGetResponse> {
            Ok(None)
        }

        async fn scan_records(
            &self,
            _table_schema: &CatalogTableSchema,
            _request: TableRecordScanRequest,
            _tenant_context: Option<&TenantContext>,
            #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
        ) -> Result<Vec<ProximaRecord>> {
            Ok(self.records.clone())
        }
    }

    fn test_record(id: &str) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            local_id: Some(id.to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn planned_executor_reports_selected_route_without_committing() {
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        let result = PlannedOnlyTableWriteExecutor::new()
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: routed,
                tenant_context: None,
            })
            .await
            .unwrap();

        assert_eq!(result.status, TableWriteExecutionStatus::PlannedOnly);
        assert_eq!(result.rows_written, 0);
        assert!(result.route_summary.contains("backend=Native"));
    }

    #[tokio::test]
    async fn planned_executor_rejects_routes_missing_core_guards() {
        let schema = CatalogTableSchema::new("orders");
        let mut routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &CopyIntoPlan {
                source: crate::query::table_write_plan::ReadSource::QuerySql(
                    "SELECT * FROM staging".to_string(),
                ),
                target: LogicalTableRef::new("orders"),
                write_mode: WriteMode::Append,
                conflict_policy: Default::default(),
                distribution: Default::default(),
            },
        });
        routed.required_guards.clear();
        routed.selected_path.guards.clear();

        let err = PlannedOnlyTableWriteExecutor::new()
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: RoutedExecutionPlan {
                    backend: ComputeBackend::Native,
                    estimated_cost: CostEstimate {
                        rows: None,
                        bytes: None,
                        relative_cost: 1.0,
                        reason: "test".to_string(),
                    },
                    ..routed
                },
                tenant_context: None,
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("missing required guard"));
    }

    #[tokio::test]
    async fn native_source_reader_classifies_unsupported_sources() {
        let reader = TableRecordStoreSourceReader::new(Arc::new(ScanRecordStore {
            records: Vec::new(),
        }));
        let schema = CatalogTableSchema::new("orders");
        let mut cursor = TableRecordSourceCursor::default();

        let err = reader
            .next_batch(
                &ReadSource::QuerySql("SELECT * FROM staging".to_string()),
                None,
                &schema,
                None,
                &mut cursor,
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("PermanentReadFailure"));
        assert!(err.to_string().contains("requires a compute adapter"));
    }

    #[tokio::test]
    async fn native_source_reader_classifies_schema_mismatch() {
        let reader = TableRecordStoreSourceReader::new(Arc::new(ScanRecordStore {
            records: Vec::new(),
        }));
        let target_schema = CatalogTableSchema::new("orders");
        let source_schema = CatalogTableSchema::new("staging");
        let mut cursor = TableRecordSourceCursor::default();

        let err = reader
            .next_batch(
                &ReadSource::CatalogTable {
                    table: LogicalTableRef::new("orders"),
                    snapshot: Default::default(),
                },
                Some(&source_schema),
                &target_schema,
                None,
                &mut cursor,
            )
            .await
            .unwrap_err();

        assert!(err.to_string().contains("TypeViolation"));
        assert!(err.to_string().contains("does not match resolved schema"));
    }

    #[tokio::test]
    async fn native_executor_writes_source_batches_through_record_store() {
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        assert_eq!(routed.backend, ComputeBackend::Native);

        let source = Arc::new(VecSourceReader::new(vec![
            vec![test_record("r1"), test_record("r2")],
            vec![test_record("r3")],
        ]));
        let store = Arc::new(CapturingRecordStore {
            writes: Mutex::new(Vec::new()),
        });

        let result = NativeTableWriteExecutor::new(source, store.clone())
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: routed,
                tenant_context: None,
            })
            .await
            .unwrap();

        assert_eq!(result.status, TableWriteExecutionStatus::Completed);
        assert_eq!(result.rows_written, 3);
        let writes = store.writes.lock().unwrap();
        assert_eq!(writes.len(), 3);
        assert!(
            writes
                .iter()
                .all(|mutation| mutation.kind == TableRecordMutationKind::Insert)
        );
    }

    /// Stub catalog port returning a fixed parent table for FK tests.
    struct StubParentResolver {
        parent_schema: CatalogTableSchema,
        parent_table_id: String,
    }

    #[async_trait]
    impl ParentTableResolver for StubParentResolver {
        async fn resolve_parent_table(
            &self,
            _references_table: &str,
            _tenant: Option<&str>,
        ) -> Result<Option<ResolvedParentTable>> {
            Ok(Some(ResolvedParentTable {
                schema: self.parent_schema.clone(),
                table_id_name: self.parent_table_id.clone(),
            }))
        }
    }

    /// Record store that reports which parent keys exist (for FK probes) and
    /// captures committed mutations.
    struct FkAwareRecordStore {
        existing_parent_keys: std::collections::HashSet<String>,
        writes: Mutex<Vec<TableRecordMutation>>,
    }

    #[async_trait]
    impl TableRecordStore for FkAwareRecordStore {
        async fn write_mutations(
            &self,
            _table_schema: &CatalogTableSchema,
            mutations: Vec<TableRecordMutation>,
            _tenant_context: Option<&TenantContext>,
        ) -> Result<TableRecordWriteResult> {
            let ids = mutations
                .iter()
                .map(|mutation| mutation.record.oid.clone())
                .collect::<Vec<_>>();
            self.writes.lock().unwrap().extend(mutations);
            Ok(TableRecordWriteResult {
                success: true,
                record_ids: ids,
                metrics: OperationMetrics::default(),
                errors: Vec::new(),
                error_code: None,
            })
        }

        async fn get_by_key(
            &self,
            _table_schema: &CatalogTableSchema,
            request: TableRecordGetRequest,
            _tenant_context: Option<&TenantContext>,
            #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
        ) -> Result<TableRecordGetResponse> {
            Ok(self.existing_parent_keys.contains(&request.key).then(|| {
                crate::services::operations::vectors::RichSearchResult {
                    id: request.key,
                    score: 0.0,
                    similarity: None,
                    vector: Vec::new(),
                    props: std::collections::HashMap::new(),
                    version: None,
                    timestamp: None,
                    source: None,
                }
            }))
        }
    }

    /// A child `orders` record carrying a `customer_id` FK value in its props.
    fn fk_child_record(id: &str, customer_id: &str) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            local_id: Some(id.to_string()),
            props: proximadb_records::ProximaTree::from([(
                "customer_id".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(customer_id.to_string()),
                ),
            )]),
            ..Default::default()
        }
    }

    /// Build a routed Native plan + a child `orders` schema whose single-column
    /// `customer_id` FK references `customers(id)`.
    fn fk_child_schema() -> CatalogTableSchema {
        let mut schema = CatalogTableSchema::new("orders")
            .with_primary_key(vec!["id".to_string()])
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        schema
            .relational_capabilities
            .constraints
            .push(ColumnConstraint::ForeignKey {
                columns: vec!["customer_id".to_string()],
                references_table: "customers".to_string(),
                references_columns: vec!["id".to_string()],
                on_delete: None,
                on_update: None,
            });
        schema
    }

    fn fk_routed_plan(schema: &CatalogTableSchema) -> RoutedExecutionPlan {
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        assert_eq!(routed.backend, ComputeBackend::Native);
        routed
    }

    #[tokio::test]
    async fn native_executor_enforces_foreign_key_references() {
        // TD-110: the native INSERT-SELECT path now enforces single-column FK
        // references when a catalog lookup port is supplied. Existing parent →
        // commit; missing parent → reject; NULL FK → exempt.
        let schema = fk_child_schema();
        let parent_schema =
            CatalogTableSchema::new("customers").with_primary_key(vec!["id".to_string()]);
        let resolver: Arc<dyn ParentTableResolver> = Arc::new(StubParentResolver {
            parent_schema,
            parent_table_id: "customers".to_string(),
        });

        // Existing parent ('c1') + a NULL-FK child ('o3', no customer_id) commit.
        let store = Arc::new(FkAwareRecordStore {
            existing_parent_keys: std::collections::HashSet::from(["c1".to_string()]),
            writes: Mutex::new(Vec::new()),
        });
        let source = Arc::new(VecSourceReader::new(vec![vec![
            fk_child_record("o1", "c1"),
            test_record("o3"),
        ]]));
        let result = NativeTableWriteExecutor::new(source, store.clone())
            .with_parent_table_resolver(resolver.clone())
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: fk_routed_plan(&schema),
                tenant_context: None,
            })
            .await
            .unwrap();
        assert_eq!(result.rows_written, 2);
        assert_eq!(store.writes.lock().unwrap().len(), 2);

        // A child referencing a missing parent ('c99') is rejected; nothing commits.
        let store = Arc::new(FkAwareRecordStore {
            existing_parent_keys: std::collections::HashSet::from(["c1".to_string()]),
            writes: Mutex::new(Vec::new()),
        });
        let source = Arc::new(VecSourceReader::new(vec![vec![fk_child_record(
            "o2", "c99",
        )]]));
        let err = NativeTableWriteExecutor::new(source, store.clone())
            .with_parent_table_resolver(resolver)
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: fk_routed_plan(&schema),
                tenant_context: None,
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("violates reference"), "{err}");
        assert_eq!(store.writes.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn native_executor_skips_foreign_keys_without_resolver() {
        // Without a catalog lookup port, FK is not enforced here (the row-local
        // validator no longer fails it closed) — the write proceeds.
        let schema = fk_child_schema();
        let store = Arc::new(FkAwareRecordStore {
            existing_parent_keys: std::collections::HashSet::new(),
            writes: Mutex::new(Vec::new()),
        });
        let source = Arc::new(VecSourceReader::new(vec![vec![fk_child_record(
            "o1", "c99",
        )]]));
        let result = NativeTableWriteExecutor::new(source, store.clone())
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: fk_routed_plan(&schema),
                tenant_context: None,
            })
            .await
            .unwrap();
        assert_eq!(result.rows_written, 1);
        assert_eq!(store.writes.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn native_executor_rejects_within_batch_unique_duplicate() {
        // TD-110: the INSERT-SELECT / native write path now enforces UNIQUE — two
        // source rows sharing a UNIQUE value in one batch are rejected, and nothing
        // is committed. (Previously this path bypassed UNIQUE entirely.)
        use proximadb_catalog::{
            CatalogColumn, CatalogIndex, CatalogIndexType, RelationalCapabilities,
        };
        use proximadb_data_model::{ProximaType, ProximaValue};
        use proximadb_records::ProximaTreeNode;

        let schema = CatalogTableSchema::new("members")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp)
            .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(2, "email", ProximaType::String))
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                unique_indexes: vec![
                    CatalogIndex::new(
                        "uq_email",
                        vec!["email".to_string()],
                        CatalogIndexType::BTree,
                    )
                    .unique(),
                ],
                ..Default::default()
            });
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("members"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        assert_eq!(routed.backend, ComputeBackend::Native);

        let with_email = |id: &str, email: &str| {
            let mut record = test_record(id);
            record.props.insert(
                "email".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(email.to_string())),
            );
            record
        };
        let source = Arc::new(VecSourceReader::new(vec![vec![
            with_email("r1", "dup@x.com"),
            with_email("r2", "dup@x.com"),
        ]]));
        let store = Arc::new(CapturingRecordStore {
            writes: Mutex::new(Vec::new()),
        });

        let err = NativeTableWriteExecutor::new(source, store.clone())
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: routed,
                tenant_context: None,
            })
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("appears more than once"),
            "unexpected error: {err}"
        );
        assert!(
            store.writes.lock().unwrap().is_empty(),
            "no rows should be committed when a batch violates UNIQUE"
        );
    }

    #[tokio::test]
    async fn table_record_store_source_reader_batches_catalog_table_scans() {
        let source_schema = CatalogTableSchema::new("staging")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let target_schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let source = Arc::new(ScanRecordStore {
            records: vec![test_record("r1"), test_record("r2"), test_record("r3")],
        });
        let reader = TableRecordStoreSourceReader::with_batch_size(source, 2);
        let read_source = ReadSource::CatalogTable {
            table: crate::query::table_write_plan::LogicalTableRef::new("staging"),
            snapshot: Default::default(),
        };
        let mut cursor = TableRecordSourceCursor::default();

        let first = reader
            .next_batch(
                &read_source,
                Some(&source_schema),
                &target_schema,
                None,
                &mut cursor,
            )
            .await
            .unwrap()
            .expect("first batch");
        let second = reader
            .next_batch(
                &read_source,
                Some(&source_schema),
                &target_schema,
                None,
                &mut cursor,
            )
            .await
            .unwrap()
            .expect("second batch");
        let done = reader
            .next_batch(
                &read_source,
                Some(&source_schema),
                &target_schema,
                None,
                &mut cursor,
            )
            .await
            .unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 1);
        assert!(done.is_none());
    }

    #[tokio::test]
    async fn native_executor_rejects_merge_until_predicates_exist() {
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let mut plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        plan.write_mode = WriteMode::Merge;
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        let err = NativeTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]])),
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("MERGE table-write execution"));
    }

    struct FailingRecordStore;

    #[async_trait]
    impl TableRecordStore for FailingRecordStore {
        async fn write_mutations(
            &self,
            _table_schema: &CatalogTableSchema,
            _mutations: Vec<TableRecordMutation>,
            _tenant_context: Option<&TenantContext>,
        ) -> Result<TableRecordWriteResult> {
            Ok(TableRecordWriteResult {
                success: false,
                record_ids: Vec::new(),
                metrics: OperationMetrics::default(),
                errors: vec!["simulated write failure".to_string()],
                error_code: None,
            })
        }

        async fn get_by_key(
            &self,
            _table_schema: &CatalogTableSchema,
            _request: TableRecordGetRequest,
            _tenant_context: Option<&TenantContext>,
            #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
        ) -> Result<TableRecordGetResponse> {
            Ok(None)
        }

        async fn scan_records(
            &self,
            _table_schema: &CatalogTableSchema,
            _request: TableRecordScanRequest,
            _tenant_context: Option<&TenantContext>,
            #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
        ) -> Result<Vec<ProximaRecord>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn native_executor_empty_source_completes_with_zero_rows() {
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        let result = NativeTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![])),
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap();

        assert_eq!(result.status, TableWriteExecutionStatus::Completed);
        assert_eq!(result.rows_written, 0);
    }

    #[tokio::test]
    async fn native_executor_propagates_store_write_failure() {
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        let err = NativeTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]])),
            Arc::new(FailingRecordStore),
        )
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("Native table write failed"));
        assert!(err.to_string().contains("simulated write failure"));
    }

    #[tokio::test]
    async fn native_executor_rejects_direct_commit_lanes_until_implemented() {
        let schema = CatalogTableSchema::new("facts")
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let overrides = WriteIntentOverrides {
            row_count_hint: Some(DEFAULT_BULK_ROW_THRESHOLD),
            estimated_bytes: Some(DEFAULT_BULK_BYTES_THRESHOLD),
            batch_local_constraints_sufficient: Some(true),
            ..Default::default()
        };
        let mut routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: Some(&overrides),
            plan: &plan,
        });
        assert_eq!(routed.write_lane_decision.lane, WriteLane::BulkAppendCommit);
        routed.backend = ComputeBackend::Native;

        let err = NativeTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]])),
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("BulkAppendCommit"));
    }

    #[tokio::test]
    async fn datafusion_executor_writes_source_batches_through_record_store() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        assert_eq!(routed.backend, ComputeBackend::DataFusionLocal);

        let source = Arc::new(VecSourceReader::new(vec![
            vec![test_record("r1"), test_record("r2")],
            vec![test_record("r3")],
        ]));
        let store = Arc::new(CapturingRecordStore {
            writes: Mutex::new(Vec::new()),
        });

        let result = DataFusionTableWriteExecutor::new(source, store.clone())
            .execute(TableWriteExecutionRequest {
                target_schema: &schema,
                source_schema: None,
                routed_plan: routed,
                tenant_context: None,
            })
            .await
            .unwrap();

        assert_eq!(result.status, TableWriteExecutionStatus::Completed);
        assert_eq!(result.rows_written, 3);
        assert!(result.route_summary.contains("DataFusionLocal"));
        assert_eq!(store.writes.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn datafusion_executor_allows_bulk_append_object_store_lane() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_layout(CatalogStorageLayout::projection_publication(
                "primary",
                CatalogPhysicalFormat::Iceberg,
                "warehouse/facts",
            ))
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let overrides = WriteIntentOverrides {
            row_count_hint: Some(DEFAULT_BULK_ROW_THRESHOLD),
            estimated_bytes: Some(DEFAULT_BULK_BYTES_THRESHOLD),
            batch_local_constraints_sufficient: Some(true),
            ..Default::default()
        };
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: Some(&overrides),
            plan: &plan,
        });
        assert_eq!(routed.backend, ComputeBackend::DataFusionLocal);
        assert_eq!(routed.write_lane_decision.lane, WriteLane::BulkAppendCommit);

        let store = Arc::new(CapturingRecordStore {
            writes: Mutex::new(Vec::new()),
        });
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        let result = DataFusionTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]])),
            store.clone(),
        )
        .with_object_store_bridge(bridge.clone())
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap();

        assert_eq!(result.status, TableWriteExecutionStatus::Completed);
        assert_eq!(result.rows_written, 1);
        assert_eq!(store.writes.lock().unwrap().len(), 0);
        let writes = bridge.writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].1, vec!["r1".to_string()]);
        assert!(
            writes[0]
                .0
                .as_ref()
                .starts_with("warehouse/facts/data/facts-append-")
        );
        assert!(writes[0].0.as_ref().ends_with(".parquet"));
    }

    #[tokio::test]
    async fn datafusion_bulk_append_requires_object_store_bridge() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let overrides = WriteIntentOverrides {
            row_count_hint: Some(DEFAULT_BULK_ROW_THRESHOLD),
            estimated_bytes: Some(DEFAULT_BULK_BYTES_THRESHOLD),
            batch_local_constraints_sufficient: Some(true),
            ..Default::default()
        };
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: Some(&overrides),
            plan: &plan,
        });
        assert_eq!(routed.write_lane_decision.lane, WriteLane::BulkAppendCommit);

        let err = DataFusionTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]])),
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("ObjectStoreBridge"));
    }

    #[tokio::test]
    async fn datafusion_bulk_append_batches_parquet_and_commits_manifest() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_layout(CatalogStorageLayout::projection_publication(
                "primary",
                CatalogPhysicalFormat::Iceberg,
                "warehouse/facts",
            ))
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let overrides = WriteIntentOverrides {
            row_count_hint: Some(DEFAULT_BULK_ROW_THRESHOLD),
            batch_local_constraints_sufficient: Some(true),
            ..Default::default()
        };
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: Some(&overrides),
            plan: &plan,
        });

        // Two source batches
        let source = Arc::new(VecSourceReader::new(vec![
            vec![test_record("r1"), test_record("r2")],
            vec![test_record("r3")],
        ]));
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        let result = DataFusionTableWriteExecutor::new(
            source,
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .with_object_store_bridge(bridge.clone())
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap();

        assert_eq!(result.status, TableWriteExecutionStatus::Completed);
        assert_eq!(result.rows_written, 3);

        // Verify multiple Parquet files were written with distinct paths (batch index)
        let writes = bridge.writes.lock().unwrap();
        assert_eq!(writes.len(), 2);
        assert!(writes[0].0.as_ref().contains("-00000.parquet"));
        assert!(writes[1].0.as_ref().contains("-00001.parquet"));
        assert_eq!(writes[0].1, vec!["r1", "r2"]);
        assert_eq!(writes[1].1, vec!["r3"]);

        // Verify manifest was committed exactly once
        let commits = bridge.commits.lock().unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].1, "warehouse/facts/_manifests");
        assert_eq!(commits[0].2, Some(0)); // First version
    }

    #[tokio::test]
    async fn datafusion_bulk_append_errors_on_persistent_manifest_conflict() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_layout(CatalogStorageLayout::projection_publication(
                "primary",
                CatalogPhysicalFormat::Iceberg,
                "warehouse/facts",
            ))
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let overrides = WriteIntentOverrides {
            row_count_hint: Some(DEFAULT_BULK_ROW_THRESHOLD),
            batch_local_constraints_sufficient: Some(true),
            ..Default::default()
        };
        let routed = TableWriteRouter.route(crate::query::table_write_plan::RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: Some(&overrides),
            plan: &plan,
        });

        let source = Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]]));
        // Bridge that never lets the writer win the CAS — the commit loop must give
        // up after MAX_MANIFEST_COMMIT_ATTEMPTS instead of spinning forever.
        let bridge = Arc::new(CapturingObjectStoreBridge::new_always_conflict());
        let err = DataFusionTableWriteExecutor::new(
            source,
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .with_object_store_bridge(bridge.clone())
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("persistent snapshot conflict"));
        // The data file was written, then exactly MAX attempts were made before bailing.
        assert_eq!(bridge.writes.lock().unwrap().len(), 1);
        assert_eq!(
            bridge.commits.lock().unwrap().len(),
            MAX_MANIFEST_COMMIT_ATTEMPTS
        );
    }
}
