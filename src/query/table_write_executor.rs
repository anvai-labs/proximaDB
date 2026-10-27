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
use proximadb_kernel::error::StorageError;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::object_store_bridge::{
    BridgeObjectPath as Path, ObjectStoreBridge, deterministic_table_uuid, iceberg_publishes_v3,
    iceberg_snapshot_fields,
};

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
    /// The warehouse-root-relative object prefix MATERIALIZE publishes this table
    /// under (`data/{tenant}/{namespace_id}/{table}`), resolved by the caller
    /// through the same resolver. The bulk-append lane keys its Iceberg metadata
    /// log with it — the one-log-per-table invariant — because a layout URL's
    /// scheme-stripped path is a *different physical key* than the root-relative
    /// log. `None` (tests, direct executor use) derives the root from the
    /// schema's own paths; a scheme-carrying location then refuses.
    pub warehouse_prefix: Option<&'a str>,
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

/// The bridge-relative key of the table's engine metadata log root (the parent
/// of its `metadata/` directory), for the bulk-append commit.
///
/// Precedence: (1) the caller-resolved warehouse prefix — the same
/// root-relative prefix MATERIALIZE publishes under, which is the only key that
/// can be derived for a scheme-carrying layout location (its scheme-stripped
/// path is a *different physical key*, so using it would write a second log for
/// the table — refuse instead). (2) The schema's own paths: a storage-layout
/// location ending in `/data` is the data directory, so the root is its parent;
/// a `schema.location` (the dml-stamped warehouse prefix) or a fallback base is
/// already a table root and is NEVER stripped — two tables whose path ends in
/// `data` must not share their parent's log.
fn engine_metadata_root(
    request: &TableWriteExecutionRequest<'_>,
    fallback_base: &str,
) -> Result<String> {
    if let Some(prefix) = request.warehouse_prefix {
        return Ok(normalize_object_path_prefix(prefix));
    }
    let layout = primary_layout(request.target_schema).and_then(|l| match l.physical_format {
        CatalogPhysicalFormat::Iceberg | CatalogPhysicalFormat::Parquet => l.location.as_deref(),
        _ => None,
    });
    if let Some(location) = layout {
        if location.contains("://") {
            return Err(anyhow!(
                "DataFusion bulk append for '{}' cannot key its engine metadata log from the \
                 URL location '{location}' without the warehouse prefix (refusing rather than \
                 writing a second log for the table)",
                request.target_schema.name
            ));
        }
        let normalized = normalize_object_path_prefix(location);
        return Ok(normalized
            .strip_suffix("/data")
            .unwrap_or(normalized.as_str())
            .to_string());
    }
    if request
        .target_schema
        .location
        .as_deref()
        .is_some_and(|l| l.contains("://"))
    {
        return Err(anyhow!(
            "DataFusion bulk append for '{}' cannot key its engine metadata log from the URL \
             location '{}' without the warehouse prefix (refusing rather than writing a second \
             log for the table)",
            request.target_schema.name,
            request
                .target_schema
                .location
                .as_deref()
                .unwrap_or_default()
        ));
    }
    Ok(fallback_base.to_string())
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
        let mut part_keys: Vec<Path> = Vec::new();

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
                part_keys.push(path.clone());

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
            // One Iceberg writer, one metadata log per table (ADR-097 D3 /
            // TD-SEAM-1 slice 2). The log key is the warehouse-root-relative
            // prefix MATERIALIZE publishes under when the caller resolved one
            // (a layout URL's scheme-stripped path is a different physical key,
            // so it must NOT be used); without it the root is derived from the
            // schema's own paths, refusing a scheme-carrying location rather
            // than silently writing a second log.
            let table_root = engine_metadata_root(&request, &base)?;
            let metadata_prefix = format!("{table_root}/metadata");
            let data_dir = format!("{base}/data");
            let table_uuid = deterministic_table_uuid(&table_root);
            let fields = iceberg_snapshot_fields(&request.target_schema.to_arrow_schema());

            // Part names keep the nanosecond clock (uniqueness is all they
            // need). Snapshot ids join the MILLISECOND scale — the engine
            // requires a successor id to exceed its parent's, MATERIALIZE
            // mints millisecond ids, and a nanosecond id here would starve
            // every later MATERIALIZE of a valid id (review finding 4).
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or_default();

            // Optimistic-concurrency commit: rebase onto the latest metadata
            // version and retry on conflict, bounded so a persistent conflict
            // surfaces as an error instead of spinning forever. Every attempt
            // recomputes the carried set, so a publisher that took the slot between
            // attempts has its files carried instead of dropped. The engine
            // additionally refuses to downgrade an already-v3 table (format
            // policy), so the requested version is only ever a floor.
            let mut committed = false;
            for attempt in 0..MAX_MANIFEST_COMMIT_ATTEMPTS {
                let parent = bridge
                    .latest_metadata_version(&metadata_prefix)
                    .await
                    .map_err(|err| {
                        anyhow!(
                            "DataFusion bulk append for '{}' failed to read the metadata log \
                             at '{metadata_prefix}': {err}",
                            request.target_schema.name
                        )
                    })?;
                // Bumped past the parent's current snapshot id when the clock
                // does not move (two publications in one millisecond stay
                // monotone; the attempt adds margin for rebased retries). An
                // unparseable parent reads as 0 and the engine's own
                // advancement check remains the backstop.
                let parent_snap_id = match parent {
                    Some(version) => bridge
                        .read_table_metadata(&metadata_prefix, version)
                        .await
                        .ok()
                        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
                        .and_then(|meta| meta.get("current-snapshot-id").and_then(|v| v.as_i64()))
                        .unwrap_or(0),
                    None => 0,
                };
                let snapshot_id = now_ms.max(parent_snap_id.saturating_add(1)) + attempt as i64;
                // The carried set: the latest engine snapshot's files (and its
                // deletion vectors — P2.4 slice 3b: a successor that names the
                // carried files without their vectors resurrects every deleted
                // row) when one exists; otherwise the retired list-based
                // publisher's semantics — every `*.parquet` in the data dir
                // except this statement's own parts — which adopts stray parts
                // of failed statements and pre-migration trees. A failed listing
                // is an error, never an empty set: an empty seed would publish a
                // snapshot that drops every existing file.
                let (mut snapshot_keys, carried_deletes) = match parent {
                    Some(version) => {
                        let snapshot = bridge
                            .read_iceberg_snapshot_files(&metadata_prefix, version)
                            .await
                            .map_err(|err| {
                                anyhow!(
                                    "DataFusion bulk append for '{}' failed to read snapshot \
                                     v{version} at '{metadata_prefix}': {err}",
                                    request.target_schema.name
                                )
                            })?;
                        (snapshot.data_files, snapshot.deletion_vectors)
                    }
                    None => {
                        let listed = bridge
                            .list_objects(&Path::from(data_dir.as_str()))
                            .await
                            .map_err(|err| {
                                anyhow!(
                                    "DataFusion bulk append for '{}' failed to list '{data_dir}': \
                                     {err}",
                                    request.target_schema.name
                                )
                            })?;
                        (
                            listed
                                .into_iter()
                                .filter(|key| {
                                    key.as_ref().ends_with(".parquet") && !part_keys.contains(key)
                                })
                                .collect(),
                            Vec::new(),
                        )
                    }
                };
                snapshot_keys.extend(part_keys.iter().cloned());
                snapshot_keys.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
                snapshot_keys.dedup_by(|a, b| a.as_ref() == b.as_ref());

                let v3 = iceberg_publishes_v3(&fields, parent.is_none());
                let publish = if carried_deletes.is_empty() {
                    bridge.publish_iceberg_files(
                        &snapshot_keys,
                        parent,
                        &metadata_prefix,
                        &table_uuid,
                        &table_root,
                        &fields,
                        snapshot_id,
                        now_ms,
                        v3,
                    )
                } else {
                    bridge.publish_iceberg_files_deleting(
                        &snapshot_keys,
                        &carried_deletes,
                        parent,
                        &metadata_prefix,
                        &table_uuid,
                        &table_root,
                        &fields,
                        snapshot_id,
                        now_ms,
                    )
                };
                match publish.await {
                    Ok(_) => {
                        committed = true;
                        break;
                    }
                    // Lost the race for the successor slot: rebase onto the
                    // winner and retry with a fresh parent and carried set.
                    Err(StorageError::AlreadyExists(_)) => continue,
                    Err(err) => {
                        // The parts stay (durable data, only the commit is
                        // missing): the next statement's carried set adopts them.
                        // Returning the error keeps this statement atomic.
                        return Err(anyhow!(
                            "DataFusion Iceberg commit for '{}' at '{metadata_prefix}' failed: \
                             {err}",
                            request.target_schema.name
                        ));
                    }
                }
            }
            if !committed {
                return Err(anyhow!(
                    "DataFusion Iceberg commit for '{}' failed after {} attempts due to \
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
        BridgeInMemoryObjectStore as InMemory, BridgeObjectStore as ObjectStore,
        IcebergSnapshotDeletion, IcebergSnapshotField, IcebergSnapshotFiles, ObjectStoreBridge,
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

    /// Test-only `TableRecordWriteResult` constructors (the four mocks below
    /// shared one literal until `max_lsn` grew the clone pairs past the
    /// duplication ratchet).
    fn mock_write_result(record_ids: Vec<String>) -> TableRecordWriteResult {
        TableRecordWriteResult {
            max_lsn: None,
            success: true,
            record_ids,
            metrics: OperationMetrics::default(),
            errors: Vec::new(),
            error_code: None,
        }
    }

    fn mock_write_failure() -> TableRecordWriteResult {
        TableRecordWriteResult {
            max_lsn: None,
            success: false,
            record_ids: Vec::new(),
            metrics: OperationMetrics::default(),
            errors: vec!["simulated write failure".to_string()],
            error_code: None,
        }
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
            Ok(mock_write_result(ids))
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
        /// Engine-log publishes (TD-SEAM-1 slice 2): (metadata_prefix, parent, files).
        iceberg_commits: Mutex<Vec<(String, Option<u64>, Vec<Path>)>>,
        /// Total `publish_iceberg_files` attempts, including ones refused with
        /// `AlreadyExists` (conflict-retry observability).
        iceberg_attempts: Mutex<usize>,
        /// When set, every `publish_iceberg_files` loses the CAS so tests can
        /// exercise the bounded-retry ceiling.
        always_conflict: bool,
        /// Refuse the next `publish_iceberg_files` with `AlreadyExists` exactly
        /// once, then let it commit (the rebase-then-retry path).
        conflict_once: std::sync::atomic::AtomicBool,
        /// When the `conflict_once` refusal fires, first commit this part as a
        /// COMPETING publisher winning the slot (the executor's retry must
        /// rebase onto it and carry its file).
        competitor_part: Mutex<Option<Path>>,
        /// Keys `list_objects` reports besides the in-memory store (the mock's
        /// parquet writes only record, so the legacy-seed path sees these).
        planted_parquet: Mutex<Vec<Path>>,
        /// Per-version deletion vectors `read_iceberg_snapshot_files` reports
        /// (indexed like the engine-log versions); empty by default.
        snapshot_deletes: Mutex<Vec<Vec<IcebergSnapshotDeletion>>>,
        /// `publish_iceberg_files_deleting` commits (carry observability):
        /// (metadata_prefix, parent, files, carried deletes).
        deleting_commits:
            Mutex<Vec<(String, Option<u64>, Vec<Path>, Vec<IcebergSnapshotDeletion>)>>,
    }

    impl CapturingObjectStoreBridge {
        fn new() -> Self {
            Self {
                store: Arc::new(InMemory::new()),
                writes: Mutex::new(Vec::new()),
                iceberg_commits: Mutex::new(Vec::new()),
                iceberg_attempts: Mutex::new(0),
                always_conflict: false,
                conflict_once: std::sync::atomic::AtomicBool::new(false),
                competitor_part: Mutex::new(None),
                planted_parquet: Mutex::new(Vec::new()),
                snapshot_deletes: Mutex::new(Vec::new()),
                deleting_commits: Mutex::new(Vec::new()),
            }
        }

        fn new_always_conflict() -> Self {
            Self {
                always_conflict: true,
                ..Self::new()
            }
        }

        fn new_conflict_once() -> Self {
            let b = Self::new();
            b.conflict_once
                .store(true, std::sync::atomic::Ordering::SeqCst);
            b
        }

        /// The one-shot conflict loses to a competitor that commits `part`
        /// before us, so the retry path rebase-and-carry is observable.
        fn new_conflict_once_losing_to(part: Path) -> Self {
            let b = Self::new_conflict_once();
            *b.competitor_part.lock().unwrap() = Some(part);
            b
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

        async fn list_objects(
            &self,
            prefix: &Path,
        ) -> std::result::Result<Vec<Path>, StorageError> {
            Ok(self
                .planted_parquet
                .lock()
                .unwrap()
                .iter()
                .filter(|p| p.as_ref().starts_with(prefix.as_ref()))
                .cloned()
                .collect())
        }

        async fn latest_metadata_version(
            &self,
            metadata_prefix: &str,
        ) -> std::result::Result<Option<u64>, StorageError> {
            let commits = self.iceberg_commits.lock().unwrap();
            let count = commits.iter().filter(|c| c.0 == metadata_prefix).count();
            Ok((count > 0).then(|| count as u64 - 1))
        }

        async fn read_iceberg_table(
            &self,
            metadata_prefix: &str,
            version: u64,
        ) -> std::result::Result<Vec<Path>, StorageError> {
            Ok(self
                .iceberg_commits
                .lock()
                .unwrap()
                .iter()
                .filter(|c| c.0 == metadata_prefix)
                .nth(version as usize)
                .map(|c| c.2.clone())
                .unwrap_or_default())
        }

        async fn read_iceberg_snapshot_files(
            &self,
            metadata_prefix: &str,
            version: u64,
        ) -> std::result::Result<IcebergSnapshotFiles, StorageError> {
            Ok(IcebergSnapshotFiles {
                data_files: self.read_iceberg_table(metadata_prefix, version).await?,
                deletion_vectors: self
                    .snapshot_deletes
                    .lock()
                    .unwrap()
                    .get(version as usize)
                    .cloned()
                    .unwrap_or_default(),
            })
        }

        async fn publish_iceberg_files(
            &self,
            files: &[Path],
            parent: Option<u64>,
            metadata_prefix: &str,
            _table_uuid: &str,
            _table_location: &str,
            _fields: &[IcebergSnapshotField],
            _snapshot_id: i64,
            _timestamp_ms: i64,
            _v3: bool,
        ) -> std::result::Result<u64, StorageError> {
            *self.iceberg_attempts.lock().unwrap() += 1;
            if self.always_conflict
                || self
                    .conflict_once
                    .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                // The configured competitor wins the slot first: its commit
                // lands before our refusal, so the executor's retry must rebase
                // onto it and carry its file.
                if let Some(part) = self.competitor_part.lock().unwrap().take() {
                    let mut commits = self.iceberg_commits.lock().unwrap();
                    let version = commits.iter().filter(|c| c.0 == metadata_prefix).count() as u64;
                    commits.push((metadata_prefix.to_string(), parent, vec![part]));
                }
                return Err(StorageError::AlreadyExists(
                    "mock: lost the successor slot".into(),
                ));
            }
            let mut commits = self.iceberg_commits.lock().unwrap();
            let version = commits.iter().filter(|c| c.0 == metadata_prefix).count() as u64;
            commits.push((metadata_prefix.to_string(), parent, files.to_vec()));
            Ok(version)
        }

        async fn publish_iceberg_files_deleting(
            &self,
            files: &[Path],
            deletes: &[IcebergSnapshotDeletion],
            parent: Option<u64>,
            metadata_prefix: &str,
            _table_uuid: &str,
            _table_location: &str,
            _fields: &[IcebergSnapshotField],
            _snapshot_id: i64,
            _timestamp_ms: i64,
        ) -> std::result::Result<u64, StorageError> {
            let version = self
                .publish_iceberg_files(
                    files,
                    parent,
                    metadata_prefix,
                    "",
                    "",
                    &[],
                    _snapshot_id,
                    _timestamp_ms,
                    true,
                )
                .await?;
            self.deleting_commits.lock().unwrap().push((
                metadata_prefix.to_string(),
                parent,
                files.to_vec(),
                deletes.to_vec(),
            ));
            Ok(version)
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
            Ok(mock_write_result(Vec::new()))
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
        let mut record = ProximaRecord {
            oid: id.to_string(),
            local_id: Some(id.to_string()),
            ..Default::default()
        };
        // One typed prop so the record survives a real Arrow batch (the engine
        // writer cannot encode an empty-schema row); mock bridges ignore it.
        record.props.insert(
            "qty".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_records::ProximaValue::Int64(1)),
        );
        record
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
                warehouse_prefix: None,
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
                warehouse_prefix: None,
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
                warehouse_prefix: None,
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
            Ok(mock_write_result(ids))
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
                warehouse_prefix: None,
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
                warehouse_prefix: None,
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
                warehouse_prefix: None,
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
                warehouse_prefix: None,
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
            warehouse_prefix: None,
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
            Ok(mock_write_failure())
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
            warehouse_prefix: None,
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
            warehouse_prefix: None,
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
            warehouse_prefix: None,
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
                warehouse_prefix: None,
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
            warehouse_prefix: None,
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
            warehouse_prefix: None,
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
            warehouse_prefix: None,
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

        // Verify the engine snapshot committed exactly once, carrying exactly this
        // statement's parts into the table's metadata log (TD-SEAM-1 slice 2: the
        // legacy `{base}/_manifests` tree receives no new commits).
        let mut written: Vec<Path> = writes.iter().map(|w| w.0.clone()).collect();
        written.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        drop(writes);
        let commits = bridge.iceberg_commits.lock().unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].0, "warehouse/facts/metadata");
        assert_eq!(commits[0].1, None); // first version
        assert_eq!(commits[0].2, written);
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
            warehouse_prefix: None,
        })
        .await
        .unwrap_err();

        assert!(err.to_string().contains("persistent snapshot conflict"));
        // The data file was written, then exactly MAX attempts were made before bailing.
        assert_eq!(bridge.writes.lock().unwrap().len(), 1);
        assert_eq!(
            *bridge.iceberg_attempts.lock().unwrap(),
            MAX_MANIFEST_COMMIT_ATTEMPTS
        );
    }

    /// Shared harness: an OLAP "facts" table on the bulk-append lane, one batch.
    /// Generic over the bridge so the same harness runs against the mock and the
    /// real engine writer.
    async fn run_bulk_append<B: ObjectStoreBridge + 'static>(
        bridge: Arc<B>,
        batch: Vec<ProximaRecord>,
    ) -> Result<TableWriteExecutionResult> {
        run_bulk_append_as(bridge, "warehouse/facts", None, batch).await
    }

    /// Same, with an explicit primary-layout location (raw, as the catalog
    /// records it) and warehouse prefix (as dml resolves it), for the
    /// metadata-log key-derivation cases.
    async fn run_bulk_append_as<B: ObjectStoreBridge + 'static>(
        bridge: Arc<B>,
        layout_location: &str,
        warehouse_prefix: Option<&str>,
        batch: Vec<ProximaRecord>,
    ) -> Result<TableWriteExecutionResult> {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_layout(CatalogStorageLayout::projection_publication(
                "primary",
                CatalogPhysicalFormat::Iceberg,
                layout_location,
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
        DataFusionTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![batch])),
            Arc::new(CapturingRecordStore {
                writes: Mutex::new(Vec::new()),
            }),
        )
        .with_object_store_bridge(bridge)
        .execute(TableWriteExecutionRequest {
            target_schema: &schema,
            source_schema: None,
            routed_plan: routed,
            tenant_context: None,
            warehouse_prefix,
        })
        .await
    }

    /// A second bulk-append statement carries the previous snapshot's files: the
    /// successor snapshot names both generations, so no published row disappears.
    #[tokio::test]
    async fn datafusion_bulk_append_carries_previous_snapshot_files() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        run_bulk_append(bridge.clone(), vec![test_record("r1"), test_record("r2")])
            .await
            .unwrap();
        run_bulk_append(bridge.clone(), vec![test_record("r3")])
            .await
            .unwrap();

        let commits = bridge.iceberg_commits.lock().unwrap();
        assert_eq!(commits.len(), 2, "two statements, two snapshots");
        assert_eq!(commits[1].1, Some(0), "the second commits on the first");
        assert_eq!(commits[1].2.len(), 2, "carried + new part");
        assert!(
            commits[1]
                .2
                .iter()
                .any(|p| p.as_ref().contains("-00000.parquet")),
            "the first statement's part is carried: {:?}",
            commits[1].2
        );
    }

    /// P2.4 slice 3b: when the parent snapshot carries deletion vectors, the
    /// successor publish must go through `publish_iceberg_files_deleting` and
    /// name the carried vectors — a snapshot that carries the files but not
    /// their vectors resurrects every deleted row.
    #[tokio::test]
    async fn datafusion_bulk_append_carries_previous_snapshot_deletes() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        run_bulk_append(bridge.clone(), vec![test_record("r1"), test_record("r2")])
            .await
            .unwrap();
        // v0 now carries a deletion vector on its (recorded) part, as if a
        // delete had been published against it.
        let carried_file = bridge.iceberg_commits.lock().unwrap()[0]
            .2
            .first()
            .cloned()
            .expect("v0 names its part");
        *bridge.snapshot_deletes.lock().unwrap() = vec![vec![IcebergSnapshotDeletion {
            data_file: carried_file.clone(),
            puffin_file: Path::from("warehouse/facts/data/dv-0.puffin"),
            content_offset: 0,
            content_size_in_bytes: 64,
            puffin_file_size_in_bytes: 128,
            cardinality: 2,
        }]];

        run_bulk_append(bridge.clone(), vec![test_record("r3")])
            .await
            .unwrap();

        let deleting = bridge.deleting_commits.lock().unwrap();
        assert_eq!(
            deleting.len(),
            1,
            "the successor publish carries deletes: {deleting:?}"
        );
        assert_eq!(deleting[0].1, Some(0), "committed on the DV snapshot");
        assert_eq!(deleting[0].3.len(), 1, "the parent's vector is carried");
        assert_eq!(deleting[0].3[0].data_file, carried_file);
        assert!(
            deleting[0].2.len() >= 2,
            "carried files + the new part: {:?}",
            deleting[0].2
        );
    }

    /// First engine commit on a data dir holding stray parquet (a failed statement
    /// or a pre-migration `_manifests` tree): the legacy list-based semantics adopt
    /// it, so the seed publish cannot drop existing files.
    #[tokio::test]
    async fn datafusion_bulk_append_legacy_seed_adopts_stray_parquet() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        bridge.planted_parquet.lock().unwrap().push(Path::from(
            "warehouse/facts/data/facts-append-legacy-0.parquet",
        ));
        run_bulk_append(bridge.clone(), vec![test_record("r1")])
            .await
            .unwrap();

        let commits = bridge.iceberg_commits.lock().unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(
            commits[0].2.len(),
            2,
            "the stray part and this statement's part: {:?}",
            commits[0].2
        );
        assert!(
            commits[0]
                .2
                .iter()
                .any(|p| p.as_ref().contains("legacy-0.parquet"))
        );
    }

    /// Losing the successor slot once is not an error: the executor rebases onto
    /// the WINNER (a competitor that committed while we raced) and retries with a
    /// fresh parent and carried set — the competitor's file must be carried, not
    /// dropped by the rebase.
    #[tokio::test]
    async fn datafusion_bulk_append_retries_after_a_conflict() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new_conflict_once_losing_to(
            Path::from("warehouse/facts/data/competitor-0.parquet"),
        ));
        let result = run_bulk_append(bridge.clone(), vec![test_record("r1")])
            .await
            .unwrap();
        assert_eq!(result.status, TableWriteExecutionStatus::Completed);
        assert_eq!(*bridge.iceberg_attempts.lock().unwrap(), 2);

        let commits = bridge.iceberg_commits.lock().unwrap();
        assert_eq!(commits.len(), 2, "the competitor's commit, then ours");
        assert_eq!(commits[0].1, None, "the competitor committed first");
        assert_eq!(commits[0].2.len(), 1);
        assert_eq!(
            commits[1].1,
            Some(0),
            "our retry rebased onto the competitor's version"
        );
        assert_eq!(commits[1].2.len(), 2, "carried winner + our part");
        assert!(
            commits[1]
                .2
                .iter()
                .any(|p| p.as_ref().contains("competitor-0.parquet"))
        );
    }

    /// A scheme-carrying layout location keys its log from the warehouse prefix
    /// (the same root-relative key MATERIALIZE uses), never from the
    /// scheme-stripped URL path — that is a different physical key, and using it
    /// would write a second log for the table. Real engine writer.
    #[tokio::test]
    async fn datafusion_bulk_append_keys_the_log_from_the_warehouse_prefix() {
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;
        use proximadb_object_store::ProximaObjectStore;

        let bridge = Arc::new(IcebergObjectStoreBridge::new(ProximaObjectStore::new(
            Arc::new(InMemory::new()),
        )));
        run_bulk_append_as(
            bridge.clone(),
            "file:///wh/data/acme/ns1/facts/data",
            Some("data/acme/ns1/facts"),
            vec![test_record("r1")],
        )
        .await
        .unwrap();

        assert_eq!(
            bridge
                .latest_metadata_version("data/acme/ns1/facts/metadata")
                .await
                .unwrap(),
            Some(0),
            "the log is keyed by the warehouse-root-relative prefix"
        );
        assert_eq!(
            bridge
                .latest_metadata_version("wh/data/acme/ns1/facts/metadata")
                .await
                .unwrap(),
            None,
            "the URL-path-derived key (the pre-fix derivation) stays empty — no second log"
        );
    }

    /// The same URL layout WITHOUT a warehouse prefix cannot be keyed at all:
    /// refuse the statement rather than silently forking the table's log.
    #[tokio::test]
    async fn datafusion_bulk_append_refuses_a_url_location_without_the_warehouse_prefix() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        let err = run_bulk_append_as(
            bridge.clone(),
            "file:///wh/data/acme/ns1/facts/data",
            None,
            vec![test_record("r1")],
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("without the warehouse prefix (refusing rather than writing a second"),
            "{err}"
        );
        assert!(
            bridge.iceberg_commits.lock().unwrap().is_empty(),
            "nothing committed"
        );
    }

    /// A fallback base that merely ENDS in `/data` (a table named "data") is a
    /// table root, not a data directory: its log sits at `{root}/metadata`, never
    /// stripped to the parent — sibling tables must not share one log.
    #[tokio::test]
    async fn datafusion_bulk_append_does_not_strip_a_data_suffixed_fallback_root() {
        let schema = CatalogTableSchema::new("data")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("data"), "SELECT * FROM staging");
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
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        DataFusionTableWriteExecutor::new(
            Arc::new(VecSourceReader::new(vec![vec![test_record("r1")]])),
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
            warehouse_prefix: None,
        })
        .await
        .unwrap();

        let commits = bridge.iceberg_commits.lock().unwrap();
        assert_eq!(
            commits.len(),
            1,
            "the fallback table named 'data' commits exactly once"
        );
        assert_eq!(
            commits[0].0, "tables/data/metadata",
            "the fallback root is used as-is — no /data strip to the parent 'tables'"
        );
        assert!(
            commits[0]
                .2
                .iter()
                .all(|p| p.as_ref().starts_with("tables/data/data/")),
            "parts still land under the table's own data dir: {:?}",
            commits[0].2
        );
    }

    /// End to end through the REAL engine writer (no mock): parts written by the
    /// bridge, a spec-shaped `v*.metadata.json` at `{table_root}/metadata`, and
    /// the snapshot naming exactly the parts this statement wrote (the same
    /// writer/readers the MATERIALIZE interop gate exercises externally).
    #[tokio::test]
    async fn datafusion_bulk_append_commits_through_the_real_engine_writer() {
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;
        use proximadb_object_store::ProximaObjectStore;

        let bridge = Arc::new(IcebergObjectStoreBridge::new(ProximaObjectStore::new(
            Arc::new(InMemory::new()),
        )));
        run_bulk_append(bridge.clone(), vec![test_record("r1"), test_record("r2")])
            .await
            .unwrap();

        let version = bridge
            .latest_metadata_version("warehouse/facts/metadata")
            .await
            .unwrap()
            .expect("one committed engine version");
        assert_eq!(version, 0);
        let files = bridge
            .read_iceberg_table("warehouse/facts/metadata", version)
            .await
            .unwrap();
        assert_eq!(files.len(), 1, "one batch ⇒ one part: {files:?}");
        assert!(
            files[0].as_ref().starts_with("warehouse/facts/data/"),
            "the part sits under the table's data dir: {}",
            files[0]
        );
        let meta = bridge
            .read_table_metadata("warehouse/facts/metadata", version)
            .await
            .unwrap();
        let md: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(md["format-version"], 2, "no _ns column ⇒ v2 floor");
        assert_eq!(
            md["location"].as_str().unwrap(),
            "warehouse/facts",
            "the table location is the table root"
        );
    }
}
