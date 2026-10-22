//! DML (Data Manipulation Language) Service
//!
//! Provides SQL DML operations that integrate with the catalog and storage system:
//! - INSERT INTO ... VALUES (...)
//! - UPDATE ... SET ... WHERE ...
//! - DELETE FROM ... WHERE ...
//! - UPSERT / INSERT ... ON CONFLICT ...
//!
//! ## Cross-Model Transactions (TD-133)
//!
//! For atomic multi-modal writes (node + embedding + edges), use the
//! [`CrossModelTransactionCoordinator`](crate::services::transaction::CrossModelTransactionCoordinator):
//!
//! ```ignore
//! // Coordinator is constructed with a graph engine, record store, and the
//! // embedding collection id (see CrossModelTransactionCoordinator::new).
//! let result = coordinator
//!     .write_symbol_atomically(node, embedding, edges, &tenant_ctx)
//!     .await?;
//!
//! match result {
//!     TransactionOutcome::Committed { node_oid } => {
//!         println!("Symbol committed: {}", node_oid);
//!     }
//!     TransactionOutcome::RolledBack { reason } => {
//!         eprintln!("Transaction rolled back: {}", reason);
//!     }
//!     TransactionOutcome::Disabled => {
//!         // Fall back to legacy separate-write path
//!     }
//! }
//! ```
//!
//! The coordinator is behind the `PROXIMADB_CROSS_MODEL_TX_ENABLED` flag and
//! provides atomicity guarantees across graph and vector storage engines.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
#[cfg(feature = "abac-policy")]
use proximadb_abac::{ReadContext, SystemReadReason};
#[cfg(feature = "abac-policy")]
use proximadb_catalog::fc_metamodel::{SubjectId, Target};
use proximadb_catalog::{
    CatalogColumn, CatalogStorageLayout, CatalogTableSchema, CatalogTableStatistics,
    relational::{
        CatalogRow, RelationalConstraintKind, RelationalConstraintViolation,
        RelationalMutationKind, RelationalRecordOptions, RelationalWriteProfile,
        encode_primary_key_tuple, split_primary_key_tuple,
    },
};
use proximadb_data_model::ProximaType;
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
use proximadb_relational_types::ExprError;
use proximadb_runtime::PortIdentity;
use tracing::{debug, info, warn};

use crate::catalog::CatalogManager;
use crate::cluster::partition_lease::{DmlLockGuard, DmlLockScope, DmlLockService, LockIntent};
use crate::query::execution::ExecutionControls;
use crate::query::table_write_executor::{
    DataFusionTableWriteExecutor, NativeTableWriteExecutor, ParentTableResolver,
    PlannedOnlyTableWriteExecutor, ResolvedParentTable, TableRecordStoreSourceReader,
    TableWriteExecutionRequest, TableWriteExecutionStatus, TableWriteExecutor,
};
use crate::query::table_write_plan::{
    ConflictPolicy, CopyIntoPlan, DistributionMode, DmlWritePlanRequest, DmlWritePlanner,
    LogicalTableRef, ReadSource, RoutedExecutionPlan, TableWriteRouteExplanation,
    WriteIntentOverrides, WriteMode,
};
use crate::services::layer_a_snapshot::LayerARowAccess;
use crate::services::operations::VectorOps;
use crate::services::operations::vectors::RichSearchResult;
use crate::services::record_store::{
    CatalogRoutingTableRecordStore, DirectWalTableRecordStore, ObjectStoreIcebergRecordStore,
    ObjectStoreVectorRecordStore, TableRecordGetRequest, TableRecordMutation,
    TableRecordMutationKind, TableRecordScanRequest, TableRecordStore, VectorOpsTableRecordStore,
    proxima_value_to_unique_text,
};
use crate::services::{
    WriteDurabilityRequirement, WriteIntent, WriteLane, WriteLaneDecision, WriteLaneRouter,
    WriteOperationKind,
};
use crate::storage::tenant::context::TenantContext;
use crate::storage::trait_components::path_resolver::DrPathBuilder;
use proximadb_catalog::TableIdentifier;
use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;
use proximadb_storage_common::parquet_multipart::MultipartParquetWriter;
use proximadb_storage_common::parquet_write_policy::{ParquetLayoutHints, ParquetWritePolicy};
use proximadb_storage_common::proxima_arrow::proxima_records_to_record_batch;

/// Count of Iceberg snapshot publishes that failed during `ALTER TABLE …
/// MATERIALIZE` (TD-USUB-9).
///
/// This exists so an interop publish failure is **alertable** rather than living
/// only in a log line. The publish is deliberately non-fatal — the Parquet data
/// and catalog layout are correct without it — so a counter is the signal that
/// tells an operator the Iceberg manifests have drifted from the published data.
///
/// Registered lazily and tolerantly, exactly like the relational memtable gauge:
/// observability must never take down a write path, so a construction error
/// yields `None` and a duplicate registration leaves a working-but-unexported
/// counter. No `.expect()` — this is `src/` production code (mandate #4).
static ICEBERG_PUBLISH_FAILURES: std::sync::LazyLock<Option<prometheus::IntCounter>> =
    std::sync::LazyLock::new(|| {
        let counter = prometheus::IntCounter::new(
            "proximadb_warehouse_iceberg_publish_failures_total",
            "Iceberg snapshot publishes that failed during warehouse materialization",
        )
        .ok()?;
        let _ = prometheus::register(Box::new(counter.clone()));
        Some(counter)
    });

/// Placeholder tenant used by warehouse materialization when no `TenantContext`
/// reaches it. This path does NOT yet enforce tenant isolation (see the note in
/// [`DmlService::materialize_table_to_parquet`]); the placeholder is named and
/// centralized so the gap is greppable and the eventual fix has one call site.
pub(crate) const DEFAULT_TENANT_PLACEHOLDER: &str = "default_tenant";

// The single-tenant default namespace id now lives on the path contract
// (DrPathBuilder, ADR-074 S1) — the local const duplicate was removed so the
// default has ONE source.

/// Resolve the tenant-isolated object prefix (no trailing `/`) for a warehouse
/// snapshot: the single canonical **DrPath** layout
/// `data/{tenant}/{namespace_id}/{table}` via [`DrPathBuilder::build_from_parts`]
/// (rename-stable opaque ids; per-segment validated — tenant, namespace_id, AND
/// table are all injection-guarded by `validate_id`).
///
/// There is exactly one layout. Every namespace carries a `namespace_id` — real
/// namespaces get one at create (`NativeCatalog`), and the embedded / single-tenant
/// path uses the well-known [`DrPathBuilder::DEFAULT_NAMESPACE_ID`] (`ns_default`), the same id the
/// record-store write path already uses. The caller resolves `namespace_id`
/// (real-or-`ns_default`) and passes it here, so single-tenant is just a degenerate
/// multi-tenant — no legacy `data/{tenant}/{ns.join}/{table}` fork, no special case.
///
/// If the namespace carries an explicit owning `tenant_id` that differs from the
/// request `tenant_id`, the materialize is refused (cross-tenant).
/// ADR-031: master gate for keying the materialized object PATH by the stable
/// `object_id` instead of the mutable table name (default OFF). Mixed-read-safe
/// WITHOUT a read fallback: the materialized `location` is persisted per-table in
/// the catalog storage layout, so tables published before the flip keep their
/// name-path and tables (re)published after it get the oid-path — each read uses
/// its own stored location. A re-materialize is a full atomic snapshot, so the
/// first post-flip publish is complete at the oid-path (the old name-path files
/// are orphaned, never read). Independent of the WAL/memtable gate (separate
/// physical layer), mirroring the per-layer catalog gate `PROXIMADB_CATALOG_OBJECT_ID_PATHS`.
fn materialize_object_id_paths_enabled() -> bool {
    std::env::var("PROXIMADB_MATERIALIZE_OBJECT_ID_PATHS")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"))
}

/// The object-path segment for a materialized table: the stable `object_id`
/// (decimal text) when oid paths are on AND the table carries one, else the bare
/// name (legacy). Pure (gate passed in) so it unit-tests without the process env.
fn materialize_path_segment(table: &str, object_id: Option<u64>, oid_paths: bool) -> String {
    if oid_paths && let Some(oid) = object_id {
        return oid.to_string();
    }
    table.to_string()
}

/// TD-OLAP-6: resolve the cluster key for a materialized publication.
///
/// Precedence: the explicit `cluster_key` table property (must name an
/// existing column, else it is ignored — never a materialize failure), then
/// the first DATE / TIMESTAMP / TIMESTAMPTZ column, else no clustering.
pub(crate) fn resolve_cluster_key(schema: &CatalogTableSchema) -> Option<String> {
    if let Some(explicit) = schema.properties.get("cluster_key") {
        if schema.columns.iter().any(|c| &c.name == explicit) {
            return Some(explicit.clone());
        }
        tracing::warn!(
            "cluster_key property '{explicit}' names no column of '{}' — ignoring",
            schema.name
        );
    }
    schema
        .columns
        .iter()
        .find(|c| {
            matches!(
                c.data_type,
                ProximaType::Date | ProximaType::Timestamp(_) | ProximaType::TimestampTz(_)
            )
        })
        .map(|c| c.name.clone())
}

/// TD-OLAP-2 (A2): fold the materialized records into per-column HLL sketches
/// and publish them to the ADR-037 resident statistics registry, keyed by the
/// table name — the SAME key `ObjectStoreParquetTable::statistics()` passes to
/// `statistics_from_splits` as `collection` (and the empty-delta OLAP-merge
/// fast path falls back to that same reader, so the read-mostly benchmark path
/// is covered). `statistics_from_splits` then overlays
/// `distinct_count = Inexact(ndv)` from the registry, giving DataFusion's
/// `EnforceDistribution` rule the NDV it needs to estimate join cardinality
/// and choose a PARTITIONED (multi-core) hash-join build instead of falling
/// back to a single-partition build when `distinct_count` is `Absent` (the
/// measured 100-700× gap vs DuckDB at SF=0.1).
///
/// This is the proper fix for the reverted A1 `num_rows`-as-NDV proxy, which
/// overestimated low-cardinality columns (e.g. `l_returnflag` NDV=2 → proxy=
/// 60K) and selected an exploding intermediate join plan at SF=0.01. The HLL
/// carries the TRUE per-column distinct count (±~2% standard error at a fixed
/// 16 KB/column).
///
/// Co-design (TD-OLAP-2 / ADR-037): the dominant cost term moved here is
/// COMPUTE (the single-threaded join build), and the lever is 1 core → N cores
/// by feeding the optimizer its missing input — accurate NDV. The sketch is
/// computed at the write boundary, chunk by chunk as the rows stream to the
/// Parquet writer, so it adds ZERO scans: O(rows × cols) HLL
/// inserts (~1 s at SF=0.1 = 600 K rows × 16 cols), acceptable for a
/// materialize DDL.
///
/// Mixed-read-safe + best-effort: the registry is in-memory and a miss degrades
/// to `Absent` (the pre-A2 behavior), never a wrong answer. The summary is
/// REPLACED (`put`, not `update`) on each materialize because materialize is a
/// full-overwrite snapshot — the old records are gone, so the old sketches must
/// not survive. (The SST flush path for vector collections is a follow-on; the
/// DF/OLAP join path only reads parquet-backed tables, which are always
/// materialized, so this seam covers the join-parallelism goal.)
///
/// Streaming: the per-column HLL sketches fold one chunk at a time, so a
/// bounded-memory MATERIALIZE never holds the table.
pub(crate) struct NdvAccumulator {
    summary: crate::core::statistics::StatisticsSummary,
    /// `(name, type-name)` per column. The type name only feeds
    /// `is_numeric_type` (quantile tracking); the HLL distinct count is type-
    /// agnostic (`canonical_string` keys every value), so NDV is correct even
    /// for types outside the numeric set.
    cols: Vec<(String, serde_json::Value)>,
    rows: u64,
}

impl NdvAccumulator {
    pub(crate) fn new(table_name: &str, tenant: Option<&str>, schema: &CatalogTableSchema) -> Self {
        Self {
            summary: crate::core::statistics::StatisticsSummary::new(
                crate::core::statistics::table_statistics_key(tenant, table_name),
            ),
            cols: schema
                .columns
                .iter()
                .map(|c| {
                    (
                        c.name.clone(),
                        serde_json::Value::String(format!("{:?}", c.data_type)),
                    )
                })
                .collect(),
            rows: 0,
        }
    }

    pub(crate) fn observe(&mut self, records: &[ProximaRecord]) {
        use proximadb_search_types::sql_value_filter::proxima_value_to_filter_literal;
        self.rows += records.len() as u64;
        for rec in records {
            for (name, ty) in &self.cols {
                // NULL/absent/object nodes observe as `None` (counted in the null rate, not the HLL).
                let json_val = rec.props.get(name).and_then(|node| match node {
                    ProximaTreeNode::Value(v) => Some(proxima_value_to_filter_literal(v)),
                    _ => None,
                });
                self.summary.observe_field(name, ty, json_val.as_ref());
            }
        }
    }

    pub(crate) fn publish(mut self) {
        self.summary.set_record_count(self.rows);
        crate::core::statistics::statistics_registry().put(self.summary);
    }
}

/// TD-USUB-4: merge `incoming` into `existing` by layout `name`, rather than
/// replacing the whole list.
///
/// `MATERIALIZE` published with `set_storage_layouts(&id, vec![layout])`, which
/// **discards every other layout a table had**. A table could therefore hold exactly
/// one physical representation, which makes cost-based routing across representations
/// *inexpressible* — the router cannot choose between alternatives the catalog has no
/// way to record (ADR-094 Decision 2).
///
/// `CatalogStorageLayout::name` is documented as the stable layout identifier, so it
/// is the upsert key: re-publishing the same layout replaces it in place (no
/// duplicates on repeat `MATERIALIZE`), while a layout of a different name is
/// preserved.
///
/// Order is stable — the replaced layout keeps its original position, and a genuinely
/// new one is appended — so a reader that treats the first match as the default sees
/// no reordering churn across republications.
pub(crate) fn upsert_storage_layout(
    mut existing: Vec<CatalogStorageLayout>,
    incoming: CatalogStorageLayout,
) -> Vec<CatalogStorageLayout> {
    if let Some(slot) = existing.iter_mut().find(|l| l.name == incoming.name) {
        *slot = incoming;
    } else {
        existing.push(incoming);
    }
    existing
}

pub(crate) fn resolve_materialize_prefix(
    tenant_id: &str,
    namespace_id: &str,
    namespace_tenant_id: Option<&str>,
    storage_pool_class: proximadb_catalog::StoragePoolClass,
    table: &str,
    table_object_id: Option<u64>,
) -> Result<String> {
    if let Some(owner) = namespace_tenant_id
        && owner != tenant_id
    {
        return Err(anyhow!(
            "refusing materialize: namespace is owned by tenant {owner:?} but the request \
                 tenant is {tenant_id:?} (cross-tenant materialize)"
        ));
    }
    // The path segment is the stable object_id (gate on + present) or the name. Both
    // materialize call sites route through here, so the primary snapshot and the
    // warehouse bulk-append target always agree on the key (a divergence would write
    // appended rows to a path the published `location` doesn't point at).
    let segment = materialize_path_segment(
        table,
        table_object_id,
        materialize_object_id_paths_enabled(),
    );
    let resolved =
        DrPathBuilder::build_from_parts(tenant_id, namespace_id, &segment, storage_pool_class)
            .map_err(|e| anyhow!("refusing materialize: DrPathBuilder rejected path: {e}"))?;
    // root_prefix() carries a trailing '/'; the caller appends `/data/...`.
    Ok(resolved.root_prefix().trim_end_matches('/').to_string())
}

/// Comparison operators supported by the lightweight catalog-table SELECT path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalSelectPredicateOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// Catalog-shaped predicate condition for simple relational scans.
#[derive(Debug, Clone)]
pub enum RelationalSelectPredicateCondition {
    Comparison {
        operator: RelationalSelectPredicateOperator,
        literal: String,
    },
    In {
        literals: Vec<String>,
        negated: bool,
    },
    Like {
        pattern: String,
        negated: bool,
    },
    IsNull {
        negated: bool,
    },
}

/// Catalog-resolved predicate used by simple pgwire relational scans.
#[derive(Debug, Clone)]
pub struct RelationalSelectPredicate {
    pub column: proximadb_catalog::CatalogColumn,
    pub condition: RelationalSelectPredicateCondition,
}

/// Resolved boolean predicate tree for UPDATE/DELETE `WHERE` evaluation. Leaves
/// are the same catalog-resolved [`RelationalSelectPredicate`] used by SELECT;
/// the And/Or/Not nodes let UPDATE/DELETE honor full boolean `WHERE` clauses
/// (OR, nested groups, NOT BETWEEN) while reusing the exact same catalog-aware
/// leaf comparison (`compare_catalog_value`, PK-by-oid resolution).
#[derive(Debug, Clone)]
enum RelationalPredicateTree {
    Leaf(RelationalSelectPredicate),
    And(Vec<RelationalPredicateTree>),
    Or(Vec<RelationalPredicateTree>),
    Not(Box<RelationalPredicateTree>),
}

impl RelationalPredicateTree {
    /// Number of leaf (column) predicates in the tree — used as the
    /// `predicate_count` route-metadata signal for tree-driven SELECTs (the
    /// tree analogue of the flat path's `predicates.len()`).
    fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::And(children) | Self::Or(children) => children.iter().map(Self::leaf_count).sum(),
            Self::Not(child) => child.leaf_count(),
        }
    }
}

/// Map an Arrow column type (the materialized Parquet's physical type) to the Iceberg
/// primitive type name for the published `TableMetadata` schema. Kept in lock-step with
/// the Parquet physical types so an external Iceberg reader's schema matches the file.
/// Timestamps map to the v3 nanosecond types (ProximaDB is nanosecond-native).
fn iceberg_type_for(arrow: &arrow_schema::DataType) -> String {
    use arrow_schema::{DataType as D, TimeUnit};
    match arrow {
        D::Boolean => "boolean".into(),
        D::Int8 | D::Int16 | D::Int32 => "int".into(),
        D::Int64 | D::UInt8 | D::UInt16 | D::UInt32 | D::UInt64 => "long".into(),
        D::Float16 | D::Float32 => "float".into(),
        D::Float64 => "double".into(),
        D::Date32 | D::Date64 => "date".into(),
        D::Decimal128(p, s) | D::Decimal256(p, s) => format!("decimal({p},{s})"),
        // Iceberg's v2 `timestamp` is microseconds; nanoseconds need v3's `_ns` types
        // (see `iceberg_format_version_for`).
        D::Timestamp(TimeUnit::Nanosecond, Some(_)) => "timestamptz_ns".into(),
        D::Timestamp(TimeUnit::Nanosecond, None) => "timestamp_ns".into(),
        D::Timestamp(_, Some(_)) => "timestamptz".into(),
        D::Timestamp(_, None) => "timestamp".into(),
        D::Utf8 | D::LargeUtf8 => "string".into(),
        D::FixedSizeBinary(n) => format!("fixed[{n}]"),
        D::Binary | D::LargeBinary => "binary".into(),
        // JSON columns materialize as Utf8 today; v3 `variant` is a follow-up.
        _ => "string".into(),
    }
}

/// ADR-097 D3: publish Iceberg **v2** — the version external readers (PyIceberg,
/// DuckDB, Trino) verifiably read — unless a column needs a v3-only type
/// (nanosecond timestamps). v3 deletion vectors / row lineage / `variant` follow
/// once an external v3 reader is in the read-back gate.
fn iceberg_needs_v3(
    fields: &[proximadb_storage_common::object_store_bridge::IcebergSnapshotField],
) -> bool {
    fields.iter().any(|f| f.type_name.ends_with("_ns"))
}

/// A stable, well-formed UUID string derived deterministically from the table's object
/// prefix, so an Iceberg table keeps the same `table-uuid` across re-materializations
/// (no `uuid` v5 dependency; two seeded hashes fill the 16 bytes).
fn deterministic_table_uuid(prefix: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    prefix.hash(&mut h1);
    let a = h1.finish();
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    (prefix, 0x50524f58_4944425fu64).hash(&mut h2); // "PROX_IDB_" salt
    let b = h2.finish();
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (a >> 32) as u32,
        (a >> 16) as u16,
        a as u16,
        (b >> 48) as u16,
        b & 0x0000_ffff_ffff_ffff
    )
}

/// Parse a SQL literal into a JSON value for the pushdown filter: integers and
/// floats become JSON numbers (so i64/f64 zone-map pruning fires); everything
/// else stays a string (string-equality hash pruning).
fn literal_to_json(literal: &str) -> serde_json::Value {
    if let Ok(i) = literal.parse::<i64>() {
        serde_json::Value::from(i)
    } else if let Ok(f) = literal.parse::<f64>() {
        serde_json::json!(f)
    } else {
        serde_json::Value::String(literal.to_string())
    }
}

/// Lower a resolved [`RelationalPredicateTree`] to a canonical `FilterExpression`
/// for **block/row-group pushdown** in the object-store vector store. Only the
/// operators the zone-map pruner can use are converted; the row-exact closure
/// still filters, so this is purely a coarse pre-filter.
///
/// Soundness (no false negatives when used for pruning): an unconvertible `AND`
/// conjunct is dropped (only weakens pruning); an `OR` with any unconvertible
/// branch returns `None` (cannot soundly prune); `NOT` and unsupported operators
/// (`!=`, `NOT IN`, `LIKE`, `IS NULL`) return `None`.
fn predicate_tree_to_filter_expression(
    tree: &RelationalPredicateTree,
) -> Option<proximadb_filter_expression::FilterExpression> {
    use proximadb_filter_expression::{ComparisonOperator as FxOp, FilterExpression as Fx};
    match tree {
        RelationalPredicateTree::Leaf(pred) => {
            let field = pred.column.name.clone();
            match &pred.condition {
                RelationalSelectPredicateCondition::Comparison { operator, literal } => {
                    let op = match operator {
                        RelationalSelectPredicateOperator::Equal => FxOp::Equals,
                        RelationalSelectPredicateOperator::LessThan => FxOp::LessThan,
                        RelationalSelectPredicateOperator::LessThanOrEqual => FxOp::LessThanOrEqual,
                        RelationalSelectPredicateOperator::GreaterThan => FxOp::GreaterThan,
                        RelationalSelectPredicateOperator::GreaterThanOrEqual => {
                            FxOp::GreaterThanOrEqual
                        }
                        RelationalSelectPredicateOperator::NotEqual => return None,
                    };
                    Some(Fx::Comparison {
                        field,
                        operator: op,
                        value: literal_to_json(literal),
                    })
                }
                RelationalSelectPredicateCondition::In {
                    literals,
                    negated: false,
                } => Some(Fx::Comparison {
                    field,
                    operator: FxOp::In,
                    value: serde_json::Value::Array(
                        literals.iter().map(|l| literal_to_json(l)).collect(),
                    ),
                }),
                _ => None,
            }
        }
        RelationalPredicateTree::And(children) => {
            let parts: Vec<_> = children
                .iter()
                .filter_map(predicate_tree_to_filter_expression)
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(Fx::And(parts))
            }
        }
        RelationalPredicateTree::Or(children) => {
            // Every disjunct must convert, else a block matching the dropped
            // branch could be wrongly skipped.
            let parts: Option<Vec<_>> = children
                .iter()
                .map(predicate_tree_to_filter_expression)
                .collect();
            parts.map(Fx::Or)
        }
        // Negation cannot be soundly pruned by a min/max zone map.
        RelationalPredicateTree::Not(_) => None,
    }
}

/// Syntax-level predicate input from a SQL/protocol facade.
///
/// DML resolves this against xCatalog before route selection or evaluation.
#[derive(Debug, Clone)]
pub struct RelationalSelectPredicateInput {
    pub column_name: String,
    pub condition: RelationalSelectPredicateCondition,
}

/// Read access path selected for a simple relational SELECT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalSelectAccessPath {
    PrimaryKeyLookup,
    /// TD-127: a single-column equality / `IN`-list on a non-PK secondary-indexed
    /// column was answered by probing the OLTP secondary index for candidate oids
    /// (each re-checked against the full predicate) instead of a full scan.
    SecondaryIndexLookup,
    TableScan,
}

/// Catalog and route metadata for a simple relational SELECT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalSelectRouteMetadata {
    pub access_path: RelationalSelectAccessPath,
    pub authority_mode: String,
    pub workload_profile: String,
    pub storage_specialization: String,
    pub policy_boundary: String,
    pub predicate_count: usize,
    pub projected_column_count: usize,
    pub limit: Option<usize>,
}

/// Catalog-resolved result for a simple relational SELECT.
#[derive(Debug, Clone)]
pub struct RelationalSelectResult {
    pub selected_columns: Vec<CatalogColumn>,
    pub route_metadata: RelationalSelectRouteMetadata,
    pub rows: Vec<Vec<ProximaValue>>,
}

/// DML Statement types
#[derive(Debug, Clone)]
pub enum DmlStatement {
    /// INSERT INTO table (columns) VALUES (values), ...
    Insert {
        /// Target table name.
        table_name: String,
        /// Column names in insertion order.
        columns: Vec<String>,
        /// Rows to insert; each inner `Vec` is one row of values.
        values: Vec<Vec<SqlValueLiteral>>,
    },
    /// UPDATE table SET col = val, ... WHERE condition
    Update {
        /// Target table name.
        table_name: String,
        /// Column-value pairs to set.
        assignments: Vec<(String, SqlValueLiteral)>,
        /// Optional filter restricting which rows are updated.
        where_clause: Option<WhereClause>,
    },
    /// DELETE FROM table WHERE condition
    Delete {
        /// Target table name.
        table_name: String,
        /// Optional filter restricting which rows are deleted.
        where_clause: Option<WhereClause>,
    },
    /// INSERT INTO ... ON CONFLICT DO UPDATE
    Upsert {
        /// Target table name.
        table_name: String,
        /// Column names in insertion order.
        columns: Vec<String>,
        /// Rows to insert or update; each inner `Vec` is one row of values.
        values: Vec<Vec<SqlValueLiteral>>,
        /// Columns that define the conflict (unique key).
        conflict_columns: Vec<String>,
        /// Column-value pairs applied when a conflict is detected.
        update_assignments: Vec<(String, SqlValueLiteral)>,
    },
    /// INSERT INTO target SELECT ... FROM source.
    InsertSelect {
        /// Logical copy/write plan for the statement.
        plan: CopyIntoPlan,
        /// Optional target column list.
        columns: Vec<String>,
    },
    /// INSERT OVERWRITE target SELECT ... FROM source.
    InsertOverwrite {
        /// Logical overwrite plan for the statement.
        plan: CopyIntoPlan,
        /// Optional target column list.
        columns: Vec<String>,
    },
}

impl DmlStatement {
    /// Return the logical target table for all DML variants.
    pub fn target_table_name(&self) -> &str {
        match self {
            Self::Insert { table_name, .. }
            | Self::Update { table_name, .. }
            | Self::Delete { table_name, .. }
            | Self::Upsert { table_name, .. } => table_name,
            Self::InsertSelect { plan, .. } | Self::InsertOverwrite { plan, .. } => {
                &plan.target.name
            }
        }
    }
}

/// SQL value literals for DML operations
#[derive(Debug, Clone)]
pub enum SqlValueLiteral {
    /// NULL value
    Null,
    /// Boolean literal
    Boolean(bool),
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Binary data (hex or base64 encoded)
    Binary(Vec<u8>),
    /// Array literal (for vectors)
    Array(Vec<SqlValueLiteral>),
    /// JSON object
    Json(serde_json::Value),
    /// Parameter placeholder ($1, $2, etc.)
    Parameter(usize),
    /// Column reference (for UPDATE SET col = other_col)
    Column(String),
    /// DEFAULT keyword
    Default,
    /// Function call (e.g., NOW(), CURRENT_TIMESTAMP)
    Function {
        /// SQL function name (case-insensitive).
        name: String,
        /// Arguments passed to the function.
        args: Vec<SqlValueLiteral>,
    },
}

/// WHERE clause for UPDATE/DELETE
#[derive(Debug, Clone)]
pub struct WhereClause {
    /// Individual conditions that make up the clause.
    pub conditions: Vec<Condition>,
    /// How conditions are combined (AND / OR).
    pub operator: LogicalOperator,
}

/// Condition in WHERE clause
#[derive(Debug, Clone)]
pub enum Condition {
    /// Simple comparison: column op value
    Comparison {
        /// Column name to compare.
        column: String,
        /// Comparison operator to apply.
        operator: ComparisonOperator,
        /// Right-hand side value.
        value: SqlValueLiteral,
    },
    /// IN list: column IN (values)
    In {
        /// Column name to test.
        column: String,
        /// Set of values to test membership against.
        values: Vec<SqlValueLiteral>,
        /// When `true`, the condition is `NOT IN`.
        negated: bool,
    },
    /// BETWEEN: column BETWEEN low AND high
    Between {
        /// Column name to test.
        column: String,
        /// Lower bound (inclusive).
        low: SqlValueLiteral,
        /// Upper bound (inclusive).
        high: SqlValueLiteral,
        /// When `true`, the condition is `NOT BETWEEN`.
        negated: bool,
    },
    /// IS NULL / IS NOT NULL
    IsNull {
        /// Column name to test.
        column: String,
        /// When `true`, the condition is `IS NOT NULL`.
        negated: bool,
    },
    /// LIKE pattern match
    Like {
        /// Column name to test.
        column: String,
        /// SQL LIKE pattern (supports `%` and `_` wildcards).
        pattern: String,
        /// When `true`, the condition is `NOT LIKE`.
        negated: bool,
    },
    /// Nested conditions with AND/OR
    Nested {
        /// Inner conditions to combine.
        conditions: Vec<Condition>,
        /// Logical operator applied to inner conditions.
        operator: LogicalOperator,
    },
}

/// Comparison operators
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOperator {
    /// `=`
    Equal,
    /// `<>` or `!=`
    NotEqual,
    /// `<`
    LessThan,
    /// `<=`
    LessThanOrEqual,
    /// `>`
    GreaterThan,
    /// `>=`
    GreaterThanOrEqual,
}

/// Logical operators for combining conditions
#[derive(Debug, Clone, Copy)]
pub enum LogicalOperator {
    /// All conditions must be satisfied (SQL `AND`).
    And,
    /// At least one condition must be satisfied (SQL `OR`).
    Or,
}

/// Result of a DML operation
#[derive(Debug, Clone)]
pub struct DmlResult {
    /// Was the operation successful?
    pub success: bool,
    /// Number of rows affected
    pub rows_affected: u64,
    /// Message describing the result
    pub message: String,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Warnings (if any)
    pub warnings: Vec<String>,
    /// Inserted IDs (for INSERT operations)
    pub inserted_ids: Vec<String>,
}

impl DmlResult {
    /// Create a successful DML result
    pub fn success(rows_affected: u64, message: impl Into<String>) -> Self {
        Self {
            success: true,
            rows_affected,
            message: message.into(),
            execution_time_us: 0,
            warnings: Vec::new(),
            inserted_ids: Vec::new(),
        }
    }

    /// Set the execution time for this result
    pub fn with_execution_time(mut self, time_us: u64) -> Self {
        self.execution_time_us = time_us;
        self
    }

    /// Set the inserted IDs for this result
    pub fn with_inserted_ids(mut self, ids: Vec<String>) -> Self {
        self.inserted_ids = ids;
        self
    }

    /// Add a warning to this result
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

/// `CatalogManager`-backed [`ParentTableResolver`] so the native INSERT-SELECT
/// executor can resolve FOREIGN KEY parent tables (TD-110). Resolution failures
/// surface as `Err`; a resolvable-but-missing parent surfaces as `Ok(None)`,
/// which the executor reports as a reference violation — matching the
/// row-by-row `DmlService::enforce_foreign_keys` behavior.
struct CatalogManagerParentResolver {
    catalog_manager: Arc<CatalogManager>,
}

impl CatalogManagerParentResolver {
    fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self { catalog_manager }
    }
}

#[async_trait::async_trait]
impl ParentTableResolver for CatalogManagerParentResolver {
    async fn resolve_parent_table(
        &self,
        references_table: &str,
        tenant: Option<&str>,
    ) -> Result<Option<ResolvedParentTable>> {
        let (parent_catalog, parent_table_id) = self
            .catalog_manager
            .resolve_table_scoped(references_table, tenant)
            .await
            .map_err(|err| {
                anyhow!("FOREIGN KEY references table '{references_table}' which cannot be resolved: {err}")
            })?;
        if !parent_catalog.table_exists(&parent_table_id).await? {
            return Ok(None);
        }
        let schema = parent_catalog.get_table(&parent_table_id).await?;
        Ok(Some(ResolvedParentTable {
            schema,
            table_id_name: parent_table_id.name.clone(),
        }))
    }
}

/// Which row-producing statement `enforce_row_write_constraints` runs for
/// (TD-DML-1): INSERT and UPSERT share one enforcement sequence and differ only
/// in what a live PK holder means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowWriteConstraintMode {
    /// A live PK holder is a duplicate (23505).
    Insert,
    /// A live PK holder is the row being replaced.
    Upsert,
}

/// DML Service for executing DML statements
pub struct DmlService {
    /// Catalog manager for metadata operations
    catalog_manager: Arc<CatalogManager>,
    /// Canonical table-record store for DML mutations.
    record_store: Arc<dyn TableRecordStore>,
    /// Routed table-write executor for INSERT SELECT / OVERWRITE / CTAS / MERGE.
    table_write_executor: Arc<dyn TableWriteExecutor>,
    /// Optional tenant-scoped DML lock service. When absent, DML executes with
    /// the legacy local behavior.
    dml_lock_service: Option<Arc<DmlLockService>>,
    /// F2 (ADR-072 D3/D5): when present, composite-PK uniqueness is enforced by a
    /// fenced `put_if_absent` at write time instead of the check-then-act
    /// `get_by_key` probe. `None` (the default) keeps the legacy probe path;
    /// a store is wired only at the composition root under the `oltp-integrity`
    /// feature, so default builds are behaviorally unchanged.
    conditional_key_store: Option<Arc<dyn proximadb_storage_ports::ConditionalKeyStore>>,
    /// FA-c Phase 2: optional ABAC enforcer. When present (behind `abac-policy`)
    /// and a client subject is supplied at a read boundary, relational scans
    /// filter rows by the subject's policy. Absent by default.
    #[cfg(feature = "abac-policy")]
    abac_enforcer: Option<Arc<crate::security::rls::AbacEnforcer>>,
    /// ADR-095 D3 transaction fork: the table op-locks the COMMIT routine holds
    /// for the whole commit. `Some` ⇒ statements must NOT re-acquire them (the
    /// registry is non-reentrant) and any table outside the set fails closed —
    /// never written without its lock. `None` ⇒ acquire per statement (legacy).
    held_op_locks: Option<Arc<std::collections::HashSet<TableIdentifier>>>,
    /// ADR-095 review F8: in a transaction fork, catalog-statistics bumps are
    /// queued here instead of applied, so a transaction that fails at COMMIT
    /// never moves row counts / min-max / NDV. `None` ⇒ apply immediately.
    deferred_stats: Option<Arc<parking_lot::Mutex<Vec<DeferredStatBump>>>>,
    /// ADR-095 review (S4a) F5: in a transaction fork, the cross-pod DML lease
    /// guards a statement took are RETAINED here instead of released when the
    /// statement finishes staging. The commit task releases them only after the
    /// durable write, preserving the TD-110 guarantee that a cascade's child
    /// leases are held across the parent's write. `None` ⇒ release immediately.
    retained_dml_guards: Option<Arc<parking_lot::Mutex<Vec<DmlLockGuard>>>>,
}

/// A catalog-statistics update captured during a staged transaction, applied
/// by [`DmlService::apply_deferred_stats`] only once the commit is durable.
#[derive(Debug, Clone)]
pub enum DeferredStatBump {
    RowCount {
        table: String,
        /// The tenant the write acted as; the table resolves within it.
        tenant: Option<String>,
        delta: i64,
    },
    NullCounts {
        table: String,
        /// The tenant the write acted as; the table resolves within it.
        tenant: Option<String>,
        columns: Vec<String>,
        null_counts: Vec<u64>,
    },
    MinMax {
        table: String,
        /// The tenant the write acted as; the table resolves within it.
        tenant: Option<String>,
        column_minmax: HashMap<String, (String, String)>,
    },
    Ndv {
        table: String,
        /// The tenant the write acted as; the table resolves within it.
        tenant: Option<String>,
        ndv_per_column: HashMap<String, u64>,
    },
}

/// ADR-025: `DmlService` is the authoritative post-snapshot delta source for the
/// OLAP read-merge — it owns both the canonical-WAL change-feed and the live
/// point-read path, so the merge reconciles the cold Parquet base against exactly
/// the state any pgwire write produced.
#[async_trait::async_trait]
impl crate::query::execution::olap_delta_merge::OlapDeltaSource for DmlService {
    async fn changed_oids_since(
        &self,
        table: &str,
        snapshot_lsn: u64,
        tenant: Option<&str>,
    ) -> anyhow::Result<Vec<String>> {
        // Tenant-isolated canonical-WAL change-feed after the snapshot LSN; each
        // ChangeRow.key is the canonical record oid (= PK text). The feed is scoped
        // by tenant_id (the WAL collection_id is not tenant-unique under name-keying).
        //
        // ADR-031 O2 (mixed-read-safe dual-read): a table's WAL entries may be keyed
        // by the bare name (legacy) OR the stable object_id (post-cutover), so we
        // union both. With the object_id-first design this is the path that lets the
        // tenant predicate drop out at O4 (object_id is globally unique).
        let mut oids: std::collections::HashSet<String> = self
            .record_store
            .read_changes_since_scoped(table, tenant, snapshot_lsn)
            .await?
            .into_iter()
            .map(|c| c.key)
            .collect();
        if let Ok((schema, _)) = self.resolve_select_table(table, tenant).await
            && let Some(oid) = schema.object_id
        {
            let oid_key = oid.to_string();
            if oid_key != table {
                for c in self
                    .record_store
                    .read_changes_since_scoped(&oid_key, tenant, snapshot_lsn)
                    .await?
                {
                    oids.insert(c.key);
                }
            }
        }
        Ok(oids.into_iter().collect())
    }

    async fn current_records(
        &self,
        table: &str,
        oids: &[String],
        tenant: Option<&str>,
    ) -> anyhow::Result<(CatalogTableSchema, Vec<ProximaRecord>)> {
        let tenant_ctx = tenant.map(TenantContext::for_tenant_id);
        let (schema, _table_id) = self.resolve_select_table(table, tenant).await?;
        let col_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let mut records = Vec::with_capacity(oids.len());
        for oid in oids {
            // get_by_key applies the canonical dead-record predicate, so a deleted
            // or TTL-expired oid yields None and contributes no append.
            if let Some(row) = self
                .point_lookup_relational(table, oid, tenant_ctx.as_ref())
                .await?
            {
                records.push(Self::value_row_to_relational_record(oid, &col_names, row));
            }
        }
        Ok((schema, records))
    }
}

/// TD-USUB-2: native relational rows for DataFusion registration.
///
/// Rows come from [`DmlService::scan_table_relational`], which resolves the ABAC row
/// filter itself — a `Denied` subject yields zero rows, `Restricted(p)` applies the
/// predicate, and an identity/storage tenant mismatch errors. So the rows returned here
/// are already governed; registering them adds no enforcement bypass.
#[async_trait::async_trait]
impl crate::query::execution::native_table_provider::NativeTableSource for DmlService {
    async fn full_table_records(
        &self,
        table: &str,
        tenant: Option<&str>,
        identity: PortIdentity<'_>,
        row_cap: usize,
    ) -> anyhow::Result<Option<(CatalogTableSchema, Vec<ProximaRecord>)>> {
        let tenant_ctx = tenant.map(TenantContext::for_tenant_id);
        // Fetch one MORE than the cap so an over-cap table is detected exactly rather
        // than silently truncated — a truncated scan is a wrong answer, which is
        // strictly worse than declining (mandate #1: fail closed, never silently wrong).
        let (schema, rows) = self
            .scan_table_relational(
                table,
                None,
                None,
                Some(row_cap.saturating_add(1)),
                tenant_ctx.as_ref(),
                identity,
            )
            .await?;
        if rows.len() > row_cap {
            tracing::debug!(
                target: "proximadb::compute_route",
                table,
                row_cap,
                "native table exceeds the registration row cap; declining to register"
            );
            return Ok(None);
        }
        let col_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let records: Vec<ProximaRecord> = rows
            .into_iter()
            .enumerate()
            .map(|(i, row)| Self::value_row_to_relational_record(&i.to_string(), &col_names, row))
            .collect();
        Ok(Some((schema, records)))
    }

    async fn layer_a_snapshot(
        &self,
        table: &str,
        tenant: Option<&str>,
        identity: PortIdentity<'_>,
    ) -> anyhow::Result<Option<crate::services::layer_a_snapshot::NativeLayerASnapshot>> {
        use crate::services::layer_a_snapshot::NativeLayerASnapshot;
        let tenant_ctx = tenant.map(TenantContext::for_tenant_id);
        let (schema, _table_id, access) = self
            .relational_read_prelude(table, tenant_ctx.as_ref(), identity)
            .await?;
        // Denied: the table is empty for this caller, and no partition is read.
        let Some(access) = access else {
            return Ok(Some(NativeLayerASnapshot {
                schema,
                snapshot: None,
                access: LayerARowAccess::Unrestricted,
            }));
        };
        let Some(snapshot) = self
            .record_store
            .layer_a_snapshot(
                &schema,
                tenant_ctx.as_ref(),
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
            )
            .await?
        else {
            return Ok(None);
        };
        Ok(Some(NativeLayerASnapshot {
            schema,
            snapshot: Some(snapshot),
            access,
        }))
    }
}

impl DmlService {
    /// Create a new DML service
    pub fn new(catalog_manager: Arc<CatalogManager>, vector_ops: Arc<VectorOps>) -> Self {
        Self::with_record_store(
            catalog_manager,
            Arc::new(CatalogRoutingTableRecordStore::with_vector_compatibility(
                vector_ops,
            )),
        )
    }

    /// Create a DML service over an explicit canonical table-record store.
    pub fn with_record_store(
        catalog_manager: Arc<CatalogManager>,
        record_store: Arc<dyn TableRecordStore>,
    ) -> Self {
        Self::with_record_store_and_table_write_executor(
            catalog_manager,
            record_store,
            Arc::new(PlannedOnlyTableWriteExecutor::new()),
        )
    }

    /// Create a DML service with the native direct canonical writer enabled.
    ///
    /// Mutable Proxima-owned relational/OLTP/HTAP tables route through
    /// `DirectWalTableRecordStore` into canonical WAL plus the `RecordStorage`
    /// row/delta spine. OLAP/open-format publication is a projection of that
    /// state unless xCatalog declares an explicit external authority. Legacy
    /// vector/LSM-specialized tables still route through the VectorOps
    /// compatibility adapter until those formats are retired or migrated.
    pub fn with_direct_record_storage(
        catalog_manager: Arc<CatalogManager>,
        vector_ops: Arc<VectorOps>,
        canonical_store: Arc<DirectWalTableRecordStore>,
    ) -> Self {
        let legacy_store = Arc::new(VectorOpsTableRecordStore::new(vector_ops));
        let routed_store: Arc<dyn TableRecordStore> = Arc::new(
            // Temporary wiring during migration: canonical acts as the native
            // row/delta path, legacy acts as the vector-specialized path. The
            // canonical store is shared across all pgwire connections so its
            // per-(tenant, collection) partitions hold one authoritative state.
            CatalogRoutingTableRecordStore::new(
                canonical_store,
                legacy_store.clone(),
                legacy_store,
            ),
        );
        let source_reader = Arc::new(TableRecordStoreSourceReader::new(routed_store.clone()));
        // TD-110: thread a catalog lookup port so the native INSERT-SELECT path
        // enforces FOREIGN KEY references like the row-by-row INSERT/UPDATE path.
        let parent_resolver: Arc<dyn ParentTableResolver> =
            Arc::new(CatalogManagerParentResolver::new(catalog_manager.clone()));
        let table_write_executor = Arc::new(
            NativeTableWriteExecutor::new(source_reader, routed_store.clone())
                .with_parent_table_resolver(parent_resolver),
        );

        Self::with_record_store_and_table_write_executor(
            catalog_manager,
            routed_store,
            table_write_executor,
        )
    }

    /// Create a DML service with the decoupled object-store routes enabled.
    ///
    /// Relational/analytics table writes route to the Parquet/Iceberg object
    /// store path, vector-specialized writes route to the PAX/vector object
    /// store path, and legacy vector/LSM tables keep the VectorOps adapter.
    pub fn with_object_store_bridge(
        catalog_manager: Arc<CatalogManager>,
        vector_ops: Arc<VectorOps>,
        bridge: Arc<dyn ObjectStoreBridge>,
    ) -> Self {
        let iceberg_store = Arc::new(ObjectStoreIcebergRecordStore::new(bridge.clone()));
        let vector_store = Arc::new(ObjectStoreVectorRecordStore::new(bridge.clone()));
        let legacy_store = Arc::new(VectorOpsTableRecordStore::new(vector_ops));
        let routed_store: Arc<dyn TableRecordStore> = Arc::new(
            CatalogRoutingTableRecordStore::new(iceberg_store, vector_store, legacy_store),
        );
        let source_reader = Arc::new(TableRecordStoreSourceReader::new(routed_store.clone()));
        let table_write_executor = Arc::new(
            DataFusionTableWriteExecutor::new(source_reader, routed_store.clone())
                .with_object_store_bridge(bridge),
        );

        Self::with_record_store_and_table_write_executor(
            catalog_manager,
            routed_store,
            table_write_executor,
        )
    }

    /// Create a DML service over explicit DML and table-write executors.
    pub fn with_record_store_and_table_write_executor(
        catalog_manager: Arc<CatalogManager>,
        record_store: Arc<dyn TableRecordStore>,
        table_write_executor: Arc<dyn TableWriteExecutor>,
    ) -> Self {
        Self {
            catalog_manager,
            record_store,
            table_write_executor,
            dml_lock_service: None,
            conditional_key_store: None,
            #[cfg(feature = "abac-policy")]
            abac_enforcer: None,
            held_op_locks: None,
            deferred_stats: None,
            retained_dml_guards: None,
        }
    }

    /// ADR-095 D3: a copy of this service for executing one transaction's
    /// statements against `record_store` (the staging store) while the caller
    /// holds `held_op_locks`. Everything else (catalog, DML leases, ABAC) is
    /// shared. The table-write executor is kept but never reached from a fork:
    /// INSERT … SELECT is refused inside an atomic transaction (pgwire buffer
    /// time) and runs on the direct path in autocommit (it would bypass staging).
    pub fn fork_for_transaction(
        &self,
        record_store: Arc<dyn TableRecordStore>,
        held_op_locks: std::collections::HashSet<TableIdentifier>,
    ) -> Self {
        Self {
            catalog_manager: self.catalog_manager.clone(),
            record_store,
            table_write_executor: self.table_write_executor.clone(),
            dml_lock_service: self.dml_lock_service.clone(),
            conditional_key_store: self.conditional_key_store.clone(),
            #[cfg(feature = "abac-policy")]
            abac_enforcer: self.abac_enforcer.clone(),
            held_op_locks: Some(Arc::new(held_op_locks)),
            deferred_stats: Some(Arc::new(parking_lot::Mutex::new(Vec::new()))),
            retained_dml_guards: Some(Arc::new(parking_lot::Mutex::new(Vec::new()))),
        }
    }

    /// Drain the statistics bumps this transaction fork queued (empty when not
    /// a fork). Hand them to [`Self::apply_deferred_stats`] after the commit is
    /// durable; drop them if it fails.
    pub fn take_deferred_stats(&self) -> Vec<DeferredStatBump> {
        self.deferred_stats
            .as_ref()
            .map(|queue| std::mem::take(&mut *queue.lock()))
            .unwrap_or_default()
    }

    /// Drain the DML lease guards this transaction fork retained; release them
    /// after the commit is durable (or on failure).
    pub fn take_retained_dml_guards(&self) -> Vec<DmlLockGuard> {
        self.retained_dml_guards
            .as_ref()
            .map(|guards| std::mem::take(&mut *guards.lock()))
            .unwrap_or_default()
    }

    /// End-of-statement handling for DML lease guards: release them now, or —
    /// in a transaction fork — retain them until the durable commit.
    async fn finish_dml_guards(&self, guards: impl IntoIterator<Item = DmlLockGuard>) {
        match &self.retained_dml_guards {
            Some(retained) => retained.lock().extend(guards),
            None => {
                for guard in guards {
                    guard.release().await;
                }
            }
        }
    }

    /// Apply statistics bumps queued by a committed transaction, in order.
    /// Best-effort, like every statistics update (planner-only, never data).
    pub async fn apply_deferred_stats(&self, bumps: Vec<DeferredStatBump>) {
        for bump in bumps {
            match bump {
                DeferredStatBump::RowCount {
                    table,
                    tenant,
                    delta,
                } => {
                    self.bump_row_count_stats(&table, tenant.as_deref(), delta)
                        .await
                }
                DeferredStatBump::NullCounts {
                    table,
                    tenant,
                    columns,
                    null_counts,
                } => {
                    self.bump_column_null_counts(&table, tenant.as_deref(), &columns, &null_counts)
                        .await
                }
                DeferredStatBump::MinMax {
                    table,
                    tenant,
                    column_minmax,
                } => {
                    self.bump_column_minmax(&table, tenant.as_deref(), column_minmax)
                        .await
                }
                DeferredStatBump::Ndv {
                    table,
                    tenant,
                    ndv_per_column,
                } => {
                    self.bump_column_ndv(&table, tenant.as_deref(), ndv_per_column)
                        .await
                }
            }
        }
    }

    /// Queue `bump` when this is a transaction fork; `false` ⇒ apply it now.
    fn defer_stat(&self, bump: impl FnOnce() -> DeferredStatBump) -> bool {
        match &self.deferred_stats {
            Some(queue) => {
                queue.lock().push(bump());
                true
            }
            None => false,
        }
    }

    /// Whether a fenced `ConditionalKeyStore` is wired (`oltp-integrity`). Its
    /// claims live in a second log, so a transaction cannot commit atomically
    /// with it yet (TD-PGOLTP-3) — callers fail closed.
    pub fn has_conditional_key_store(&self) -> bool {
        self.conditional_key_store.is_some()
    }

    /// The op-lock set a transaction touching `tables` must hold: each table
    /// plus its ON DELETE cascade closure, resolved in each entry's tenant scope.
    pub async fn transaction_op_lock_set(
        &self,
        tables: &[(String, Option<TenantContext>, bool)],
    ) -> Result<std::collections::HashSet<TableIdentifier>> {
        let mut set = std::collections::HashSet::new();
        // The bool asks for the ON DELETE cascade closure — only DELETE walks
        // it; INSERT/UPDATE/UPSERT hold just their target (`ensure_held` fails
        // closed if that is ever too small). Skipping it avoids a tenant-wide
        // catalog scan per statement (review S4a F7).
        for (table_name, tenant_context, with_cascade_closure) in tables {
            let (catalog, table_id) = self
                .catalog_manager
                .resolve_table_scoped(
                    table_name,
                    tenant_context.as_ref().map(|t| t.tenant_id.as_str()),
                )
                .await?;
            if *with_cascade_closure && catalog.table_exists(&table_id).await? {
                let schema = catalog.get_table(&table_id).await?;
                for (namespace, name) in self
                    .discover_cascade_child_set(
                        &catalog,
                        &table_id,
                        &schema,
                        tenant_context.as_ref(),
                    )
                    .await?
                {
                    set.insert(TableIdentifier { namespace, name });
                }
            }
            set.insert(table_id);
        }
        Ok(set)
    }

    /// Acquire a transaction's whole op-lock set, in the registry's deterministic
    /// order (the same order cascade DELETE uses, so no lock-order inversion).
    pub async fn acquire_transaction_op_locks(
        &self,
        tables: &std::collections::HashSet<TableIdentifier>,
    ) -> Result<Vec<tokio::sync::OwnedSemaphorePermit>> {
        self.catalog_manager
            .op_locks()
            .acquire_sorted(tables.iter().cloned().collect())
            .await
    }

    /// Acquire `table_id`'s op-lock, or — in a transaction fork — confirm the
    /// caller already holds it (fail closed otherwise).
    async fn op_lock(
        &self,
        table_id: &TableIdentifier,
    ) -> Result<Option<tokio::sync::OwnedSemaphorePermit>> {
        match &self.held_op_locks {
            Some(held) => {
                Self::ensure_held(held, std::slice::from_ref(table_id))?;
                Ok(None)
            }
            None => Ok(Some(
                self.catalog_manager.op_locks().acquire(table_id).await?,
            )),
        }
    }

    /// Sorted multi-table form of [`Self::op_lock`].
    async fn op_locks_sorted(
        &self,
        tables: Vec<TableIdentifier>,
    ) -> Result<Vec<tokio::sync::OwnedSemaphorePermit>> {
        match &self.held_op_locks {
            Some(held) => {
                Self::ensure_held(held, &tables)?;
                Ok(Vec::new())
            }
            None => self.catalog_manager.op_locks().acquire_sorted(tables).await,
        }
    }

    fn ensure_held(
        held: &std::collections::HashSet<TableIdentifier>,
        tables: &[TableIdentifier],
    ) -> Result<()> {
        if let Some(missing) = tables.iter().find(|table| !held.contains(*table)) {
            return Err(anyhow!(
                "transaction commit does not hold the operation lock for table '{}' \
                 (its lock set changed while committing); refusing to write it unlocked",
                missing.name
            ));
        }
        Ok(())
    }

    /// F2 (ADR-072 D3/D5): wire a fenced `ConditionalKeyStore` so composite-PK
    /// uniqueness is enforced atomically at write time rather than by the
    /// check-then-act probe. Injected at the composition root under the
    /// `oltp-integrity` feature; absent by default.
    pub fn with_conditional_key_store(
        mut self,
        store: Arc<dyn proximadb_storage_ports::ConditionalKeyStore>,
    ) -> Self {
        self.conditional_key_store = Some(store);
        self
    }

    /// T4.1: Get the canonical table-record store.
    ///
    /// Provides access to the record store for components that need direct
    /// read/write access to table records, such as the `RelationalExpander`
    /// in the cross-modal fusion seam (F-D).
    pub fn record_store(&self) -> Arc<dyn TableRecordStore> {
        self.record_store.clone()
    }

    /// Attach the tenant-scoped DML lock service.
    pub fn with_dml_lock_service(mut self, dml_lock_service: Arc<DmlLockService>) -> Self {
        self.dml_lock_service = Some(dml_lock_service);
        self
    }

    /// FA-c Phase 2: wire an ABAC enforcer so relational scans filter rows by the
    /// requesting subject's policy. Behind `abac-policy`; absent (no enforcement)
    /// by default. The enforcer is a self-contained policy (holds authority +
    /// predicate store + epochs + bindings).
    #[cfg(feature = "abac-policy")]
    pub fn with_abac_enforcer(mut self, enforcer: Arc<crate::security::rls::AbacEnforcer>) -> Self {
        self.abac_enforcer = Some(enforcer);
        self
    }

    /// Attach the process-shared ABAC enforcer after construction.
    ///
    /// The pgwire protocol builds its per-connection DML façade only after the
    /// socket is accepted, so it cannot use the consuming builder without
    /// rebuilding other already-attached dependencies. The composition root
    /// calls this exactly once while its `Arc<DmlService>` is still unique.
    #[cfg(feature = "abac-policy")]
    pub fn set_abac_enforcer(&mut self, enforcer: Arc<crate::security::rls::AbacEnforcer>) {
        self.abac_enforcer = Some(enforcer);
    }

    /// TD-ABAC-10c: whether row-level enforcement is armed on this service.
    ///
    /// Read-paths that CANNOT enforce (they lack the row-filter wiring
    /// [`Self::abac_row_filter`] feeds) use this to fail closed rather than
    /// serve unenforced rows — see the legacy single-table SELECT fallback in
    /// the pgwire protocol. Deliberately not "enforce here too": a second
    /// enforcement implementation for the same rows is the drift seam ADR-087
    /// exists to remove.
    #[cfg(feature = "abac-policy")]
    pub fn has_abac_enforcer(&self) -> bool {
        self.abac_enforcer.is_some()
    }

    /// Execute a DML statement (single-tenant / unscoped).
    pub async fn execute(&self, statement: DmlStatement) -> Result<DmlResult> {
        self.execute_scoped(statement, None).await
    }

    /// Execute a DML statement within a tenant scope (TD-064). The tenant
    /// context selects the catalog schema row (via `resolve_table_scoped`) and
    /// the record partition (threaded into `write_mutations` / FK / unique
    /// checks). `None` ⇒ single-tenant, identical to the legacy path.
    pub async fn execute_scoped(
        &self,
        statement: DmlStatement,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        let start = std::time::Instant::now();

        let result = match statement {
            DmlStatement::Insert {
                table_name,
                columns,
                values,
            } => {
                self.execute_insert(&table_name, &columns, values, tenant_context)
                    .await?
            }
            DmlStatement::Update {
                table_name,
                assignments,
                where_clause,
            } => {
                self.execute_update(&table_name, assignments, where_clause, tenant_context)
                    .await?
            }
            DmlStatement::Delete {
                table_name,
                where_clause,
            } => {
                self.execute_delete(&table_name, where_clause, tenant_context)
                    .await?
            }
            DmlStatement::Upsert {
                table_name,
                columns,
                values,
                conflict_columns,
                update_assignments,
            } => {
                self.execute_upsert(
                    &table_name,
                    &columns,
                    values,
                    &conflict_columns,
                    update_assignments,
                    tenant_context,
                )
                .await?
            }
            DmlStatement::InsertSelect { plan, columns }
            | DmlStatement::InsertOverwrite { plan, columns } => {
                // TD-113 family: thread the connection tenant so the target table
                // resolves within the tenant scope and the bulk-append writes into
                // the tenant's partition (was `None` → cross-tenant / unscoped).
                self.plan_table_write(&plan, &columns, tenant_context)
                    .await?
            }
        };

        Ok(result.with_execution_time(start.elapsed().as_micros() as u64))
    }

    /// TD-064 write-authz gate: does `table_name` resolve to an existing table
    /// within the tenant scope? pgwire uses this pre-execute to deny cross-tenant
    /// writes with `42P01` (relation does not exist) — never leaking existence of
    /// another tenant's table.
    pub async fn table_visible_for_tenant(
        &self,
        table_name: &str,
        tenant: Option<&str>,
    ) -> Result<bool> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;
        catalog.table_exists(&table_id).await
    }

    async fn acquire_table_dml_lock(
        &self,
        table_id: &TableIdentifier,
        tenant_context: Option<&TenantContext>,
        intent: LockIntent,
    ) -> Result<Option<DmlLockGuard>> {
        let Some(lock_service) = &self.dml_lock_service else {
            return Ok(None);
        };

        let tenant_id = tenant_context
            .map(|tc| tc.tenant_id.as_str())
            .unwrap_or(DEFAULT_TENANT_PLACEHOLDER);
        let namespace_id = Self::dml_lock_namespace_id(table_id, tenant_context);
        let scope = DmlLockScope::Table {
            schema_name: namespace_id.clone(),
            table_name: table_id.name.clone(),
        };

        let guard = lock_service
            .acquire_dml_lock_guard(
                tenant_id,
                Some(namespace_id.as_str()),
                scope,
                intent,
                Self::now_unix_ms(),
            )
            .await;
        match guard {
            Ok(g) => Ok(Some(g)),
            Err(e) => {
                // A lock conflict → typed ProximaDBError so protocol layers can
                // map it (pgwire SQLSTATE 55P03 / gRPC ABORTED). Walk the chain
                // (anyhow downcast_ref only sees the top error) to find the
                // cluster-local conflict detail.
                use crate::cluster::partition_lease::DmlLockAcquireError;
                if let Some(conflict) = e
                    .chain()
                    .find_map(|s| s.downcast_ref::<DmlLockAcquireError>())
                {
                    let (resource, holder) = conflict.resource_holder();
                    return Err(crate::core::errors::ProximaDBError::DmlLockConflict {
                        resource,
                        holder,
                    }
                    .into());
                }
                // Non-conflict failure — propagate with context.
                Err(e).with_context(|| {
                    format!(
                        "acquiring DML lock for tenant '{tenant_id}', namespace '{namespace_id}', table '{}'",
                        table_id.name
                    )
                })
            }
        }
    }

    fn dml_lock_namespace_id(
        table_id: &TableIdentifier,
        tenant_context: Option<&TenantContext>,
    ) -> String {
        let namespace = if let Some(tenant) = tenant_context {
            table_id
                .namespace
                .strip_prefix(std::slice::from_ref(&tenant.tenant_id))
                .unwrap_or(table_id.namespace.as_slice())
        } else {
            table_id.namespace.as_slice()
        };

        if namespace.is_empty() {
            DrPathBuilder::DEFAULT_NAMESPACE_ID.to_string()
        } else {
            namespace.join(".")
        }
    }

    /// Scan current visible records for a cataloged table through the shared
    /// table-record store boundary.
    pub async fn scan_table_records(
        &self,
        table_name: &str,
        limit: Option<usize>,
    ) -> Result<(CatalogTableSchema, Vec<ProximaRecord>)> {
        self.scan_table_records_with_predicates(table_name, limit, &[])
            .await
    }

    /// Scan current visible records and apply simple catalog-shaped predicates
    /// behind the DML/catalog boundary.
    pub async fn scan_table_records_with_predicates(
        &self,
        table_name: &str,
        limit: Option<usize>,
        predicates: &[RelationalSelectPredicateInput],
    ) -> Result<(CatalogTableSchema, Vec<ProximaRecord>)> {
        self.select_table_records(table_name, limit, predicates)
            .await
    }

    /// CDC change-feed (P2): row-level changes for `table_name` with WAL sequence number
    /// strictly greater than `since_lsn`, oldest first, for the table `tenant` owns.
    ///
    /// The table is resolved within `tenant`'s namespace first; `Ok(None)` when it
    /// is not visible to `tenant` (absent and another tenant's table look the same).
    /// The feed then keeps only WAL entries written by `tenant`, across the table's
    /// partition keys (stable object id and legacy bare name). Entries written
    /// before tenant stamping (no `tenant_id`) belong to the default tenant.
    pub async fn changes_since_for_tenant(
        &self,
        table_name: &str,
        tenant: &str,
        since_lsn: u64,
    ) -> Result<Option<Vec<crate::services::record_store::ChangeRow>>> {
        if tenant.is_empty() {
            return Ok(None);
        }
        let is_default = tenant == proximadb_tenant::DEFAULT_TENANT;
        // The resolved table must sit in the tenant's own namespace.
        match self
            .catalog_manager
            .resolve_table_scoped(table_name, Some(tenant))
            .await
        {
            Ok((_, id)) if id.namespace.first().map(String::as_str) == Some(tenant) => {}
            _ => return Ok(None),
        }
        let schema = match self.resolve_select_table(table_name, Some(tenant)).await {
            Ok((schema, _)) => schema,
            Err(error) => {
                tracing::debug!(table = %table_name, %error, "change feed: table not visible to tenant");
                return Ok(None);
            }
        };
        let (primary, legacy) = crate::services::record_store::memtable_partition_keys(&schema);
        let mut keys = vec![primary];
        keys.extend(legacy);
        let mut tenants = vec![Some(tenant)];
        if is_default {
            tenants.push(None);
        }
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for key in &keys {
            for scope in &tenants {
                for row in self
                    .record_store
                    .read_changes_since_scoped(key, *scope, since_lsn)
                    .await?
                {
                    if seen.insert(row.lsn) {
                        out.push(row);
                    }
                }
            }
        }
        out.sort_by_key(|row| row.lsn);
        Ok(Some(out))
    }

    /// Select current visible records and resolve projection columns inside the
    /// DML/catalog boundary.
    pub async fn select_table_records_with_projection(
        &self,
        table_name: &str,
        projection_column_names: &[String],
        limit: Option<usize>,
        predicates: &[RelationalSelectPredicateInput],
        tenant_context: Option<&TenantContext>,
    ) -> Result<RelationalSelectResult> {
        let (table_schema, table_id_name) = self
            .resolve_select_table(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;
        let selected_columns =
            Self::resolve_select_projection(&table_schema, projection_column_names)?;
        let predicates = Self::resolve_select_predicates(&table_schema, predicates)?;
        let (access_path, records) = self
            .select_table_records_with_resolved_predicates(
                &table_schema,
                &table_id_name,
                limit,
                &predicates,
                tenant_context,
            )
            .await?;
        let rows = Self::project_select_rows(&records, &table_schema, &selected_columns)?;
        let route_metadata = Self::select_route_metadata(
            &table_schema,
            &selected_columns,
            predicates.len(),
            limit,
            access_path,
        );

        Ok(RelationalSelectResult {
            selected_columns,
            route_metadata,
            rows,
        })
    }

    /// Select current visible records for a catalog table using the same faithful
    /// boolean [`WhereClause`] IR that UPDATE/DELETE use — so OR / mixed-AND-OR /
    /// grouped SELECT predicates push into the scan instead of degrading to a full
    /// table scan. `where_clause = None` means no `WHERE` (scan all, capped by limit).
    ///
    /// This converges the pgwire SELECT WHERE path onto [`RelationalPredicateTree`]
    /// (built by [`Self::where_clause_to_predicate_tree`]) and reuses the exact
    /// OR-safe PK fast-path ([`Self::extract_pk_candidate_ids`]) + tree evaluator
    /// ([`Self::eval_predicate_tree`]) that drive UPDATE/DELETE. The flat
    /// [`Self::select_table_records_with_projection`] is retained unchanged for
    /// callers that still pass a flat predicate Vec.
    pub async fn select_table_records_with_projection_where(
        &self,
        table_name: &str,
        projection_column_names: &[String],
        limit: Option<usize>,
        where_clause: Option<&WhereClause>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<RelationalSelectResult> {
        let (table_schema, table_id_name) = self
            .resolve_select_table(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;
        let selected_columns =
            Self::resolve_select_projection(&table_schema, projection_column_names)?;

        let (access_path, records, predicate_count) = match where_clause {
            Some(where_clause) => {
                let tree = self.where_clause_to_predicate_tree(where_clause, &table_schema)?;
                let predicate_count = tree.leaf_count();
                let (access_path, records) = self
                    .select_table_records_with_tree(
                        &table_schema,
                        &table_id_name,
                        limit,
                        where_clause,
                        &tree,
                        tenant_context,
                    )
                    .await?;
                (access_path, records, predicate_count)
            }
            None => {
                // No WHERE: scan all rows up to the limit (no predicate pushed).
                let records = self
                    .record_store
                    .scan_records_filtered(
                        &table_schema,
                        TableRecordScanRequest {
                            filter: None,
                            table_id: table_id_name.clone(),
                            limit,
                            include_vector: true,
                            include_props: true,
                        },
                        None,
                        tenant_context,
                        #[cfg(feature = "abac-policy")]
                        &ReadContext::system(
                            SystemReadReason::Statistics,
                            "dml [CLIENT-PLACEHOLDER]",
                        ),
                    )
                    .await?;
                (RelationalSelectAccessPath::TableScan, records, 0)
            }
        };

        let rows = Self::project_select_rows(&records, &table_schema, &selected_columns)?;
        let route_metadata = Self::select_route_metadata(
            &table_schema,
            &selected_columns,
            predicate_count,
            limit,
            access_path,
        );

        Ok(RelationalSelectResult {
            selected_columns,
            route_metadata,
            rows,
        })
    }

    /// Resolve a catalog table's schema for the relational pipeline (PATH B).
    /// Used by the pipeline's schema-only prefetch (the sync `CatalogLookup`
    /// can't await xCatalog), so the actual rows can be fetched lazily per scan.
    pub async fn resolve_relational_schema(
        &self,
        table_name: &str,
        tenant: Option<&str>,
    ) -> Result<CatalogTableSchema> {
        let (table_schema, _table_id_name) = self.resolve_select_table(table_name, tenant).await?;
        Ok(table_schema)
    }

    /// FA-c Phase 2: resolve the ABAC row filter for a relational scan. Returns
    /// `None` when enforcement does not apply (no enforcer wired, or no client
    /// subject — e.g. an internal read). Otherwise resolves the subject's
    /// authorization against the held policy; the caller short-circuits a
    /// denied result to zero rows and AND-combines `Restricted` into the user
    /// predicate. Fail-closed: a legacy table lacking stable object/namespace
    /// ids, or a tenant without a stable id, yields `Denied`.
    #[cfg(feature = "abac-policy")]
    pub(crate) fn relational_abac_required(&self, identity: PortIdentity<'_>) -> bool {
        !matches!(
            proximadb_tenant::read_enforcement_composition(
                self.abac_enforcer.is_some(),
                identity.subject.is_some(),
                identity.tenant_stable_id.is_some(),
                identity.auth_class,
            ),
            proximadb_tenant::ReadEnforcementComposition::Passthrough
        )
    }

    #[cfg(feature = "abac-policy")]
    fn abac_row_filter(
        &self,
        identity: PortIdentity<'_>,
        table_schema: &CatalogTableSchema,
    ) -> Option<crate::security::rls::AbacScanResult> {
        match proximadb_tenant::read_enforcement_composition(
            self.abac_enforcer.is_some(),
            identity.subject.is_some(),
            identity.tenant_stable_id.is_some(),
            identity.auth_class,
        ) {
            proximadb_tenant::ReadEnforcementComposition::Passthrough => return None,
            proximadb_tenant::ReadEnforcementComposition::DenyMissingStableId => {
                tracing::warn!(
                    target: "proximadb::tenant_audit",
                    subject = identity.subject,
                    auth_class = %identity.auth_class,
                    table = table_schema.name,
                    "ABAC denied a relational subject whose tenant has no stable policy key"
                );
                return Some(crate::security::rls::AbacScanResult::Denied(
                    proximadb_abac::DenyReason::NoApplicablePolicy,
                ));
            }
            proximadb_tenant::ReadEnforcementComposition::ResolvePolicy => {}
        }
        let enforcer = self.abac_enforcer.as_ref()?;
        let subject = SubjectId(identity.subject?.to_string());
        // Enforcer + client subject are present ⇒ enforce. Fail-closed: a legacy
        // table lacking stable object/namespace ids, or an unknown tenant stable
        // id, denies rather than reading unfiltered.
        let (Some(namespace), Some(table)) =
            (table_schema.stable_namespace_id, table_schema.object_id)
        else {
            return Some(crate::security::rls::AbacScanResult::Denied(
                proximadb_abac::DenyReason::NoApplicablePolicy,
            ));
        };
        let tenant = identity.tenant_stable_id?;
        // B3: build the Target fallibly. `table` is a u64 catalog object id but
        // `Target.table` is u32, and `scope_covers` matches with `==` — a
        // truncating `as u32` could WRAP an id onto a different, legitimately
        // bound table and authorize this read under someone else's policy.
        // Fold the unrepresentable case into the same fail-closed deny the
        // missing-id case above already uses.
        let Some(target) = Target::try_from_object_ids(namespace, table) else {
            return Some(crate::security::rls::AbacScanResult::Denied(
                proximadb_abac::DenyReason::NoApplicablePolicy,
            ));
        };
        Some(enforcer.predicate_for(&subject, tenant, target))
    }

    /// Every visible record of a relational table as a bounded-memory, ordered
    /// chunk stream (P2.4): strictly ascending `oid`, each once, chunks of at
    /// most `chunk_rows`, the caller's ABAC row access applied per chunk. Returns
    /// the resolved catalog schema with it. Records keep their real `oid`;
    /// project them with [`Self::project_one_record`] for column-ordered values.
    pub async fn scan_table_records_chunked(
        &self,
        table_name: &str,
        chunk_rows: usize,
        tenant_context: Option<&TenantContext>,
        identity: PortIdentity<'_>,
    ) -> Result<(
        CatalogTableSchema,
        proximadb_records::chunked::RecordChunkStream<'static>,
    )> {
        let (table_schema, table_id_name, access) = self
            .relational_read_prelude(table_name, tenant_context, identity)
            .await?;
        let Some(access) = access else {
            let empty: proximadb_records::chunked::RecordChunkStream<'static> =
                Box::pin(futures::stream::empty());
            return Ok((table_schema, empty));
        };
        let chunks = self.record_store.clone().scan_records_chunked(
            table_schema.clone(),
            TableRecordScanRequest {
                filter: None,
                table_id: table_id_name,
                limit: None,
                include_vector: true,
                include_props: true,
            },
            chunk_rows,
            tenant_context.cloned(),
        );
        let chunks = match access {
            LayerARowAccess::Unrestricted => chunks,
            LayerARowAccess::Restricted(keep) => {
                proximadb_records::chunked::filter_chunks(chunks, move |r| keep(r))
            }
        };
        Ok((table_schema, chunks))
    }

    /// The governed-read prelude every relational read of a catalog table runs
    /// before touching rows: the identity/storage tenant cross-check, table
    /// resolution, and the ABAC row filter (FA-c Phase 2).
    ///
    /// Returns the schema, the resolved table id, and the caller's row access —
    /// `None` when the subject is denied every row. Shared by
    /// [`Self::scan_table_relational`] and the Layer A snapshot, so the
    /// streaming provider cannot drift from the materialized scan on who may
    /// see what.
    async fn relational_read_prelude(
        &self,
        table_name: &str,
        tenant_context: Option<&TenantContext>,
        identity: PortIdentity<'_>,
    ) -> Result<(CatalogTableSchema, String, Option<LayerARowAccess>)> {
        if identity.subject.is_some() {
            let identity_tenant = identity.tenant_id.ok_or_else(|| {
                anyhow!("governed relational identity is missing its tenant string")
            })?;
            let storage_tenant = tenant_context.ok_or_else(|| {
                anyhow!("governed relational identity is missing its storage tenant context")
            })?;
            if identity_tenant != storage_tenant.tenant_id {
                return Err(anyhow!(
                    "relational identity/storage tenant mismatch: identity '{}' != partition '{}'",
                    identity_tenant,
                    storage_tenant.tenant_id
                ));
            }
            if let Some(storage_stable_id) = storage_tenant.tenant_stable_id
                && identity.tenant_stable_id != Some(storage_stable_id)
            {
                return Err(anyhow!(
                    "relational identity/storage stable-id mismatch for tenant '{}'",
                    identity_tenant
                ));
            }
        }
        let (table_schema, table_id_name) = self
            .resolve_select_table(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;
        #[cfg(feature = "abac-policy")]
        let access = match self.abac_row_filter(identity, &table_schema) {
            None | Some(crate::security::rls::AbacScanResult::Unrestricted) => {
                Some(LayerARowAccess::Unrestricted)
            }
            Some(crate::security::rls::AbacScanResult::Restricted(p)) => {
                Some(LayerARowAccess::Restricted(p))
            }
            Some(crate::security::rls::AbacScanResult::Denied(_)) => None,
        };
        #[cfg(not(feature = "abac-policy"))]
        let access = {
            let _ = identity;
            Some(LayerARowAccess::Unrestricted)
        };
        Ok((table_schema, table_id_name, access))
    }

    /// Scan a catalog table for the relational pipeline (PATH B), pushing the
    /// caller's row predicate + limit into the record-store scan and projecting
    /// matching records into `output_columns` order (or all columns when `None`).
    ///
    /// The predicate is `Expr`-agnostic: callers pass a closure over a FULL row
    /// (all columns in `schema.columns` order — matching how the relational
    /// `Expr` binds its column ordinals), so the relational reader can supply
    /// `|row| expr.eval(row) == true` without coupling DmlService to the algebra
    /// crate. `MemtableRecordStorage::scan_records_filtered` iterate-filters and
    /// early-stops at `limit`, so only matching rows are cloned/materialized.
    pub async fn scan_table_relational(
        &self,
        table_name: &str,
        output_columns: Option<&[String]>,
        full_row_predicate: Option<
            &(dyn Fn(&[ProximaValue]) -> Result<bool, ExprError> + Send + Sync),
        >,
        limit: Option<usize>,
        tenant_context: Option<&TenantContext>,
        identity: PortIdentity<'_>,
    ) -> Result<(CatalogTableSchema, Vec<Vec<ProximaValue>>)> {
        let (table_schema, table_id_name, access) = self
            .relational_read_prelude(table_name, tenant_context, identity)
            .await?;
        // Full column set — predicates are evaluated against a complete row.
        let all_columns: Vec<String> = table_schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let full_selected = Self::resolve_select_projection(&table_schema, &all_columns)?;
        // Output column set the caller wants emitted (defaults to all columns).
        let output_selected = match output_columns {
            Some(cols) => Self::resolve_select_projection(&table_schema, cols)?,
            None => full_selected.clone(),
        };

        // FA-c Phase 2: a denied (or fail-closed) subject sees zero rows.
        let abac_pred = match access {
            None => return Ok((table_schema, Vec::new())),
            Some(LayerARowAccess::Unrestricted) => None,
            Some(LayerARowAccess::Restricted(p)) => Some(p),
        };

        // Push the predicate INTO the store scan: project each record to a full
        // row and apply the caller's full-row predicate. The scan-predicate API is
        // `Fn(&ProximaRecord) -> bool` (no error channel), so a predicate eval error
        // is captured here and surfaced after the scan as a hard error — NEVER
        // coerced to `false` (which would silently drop rows). ADR-043 Invariant 1:
        // a predicate the engine cannot evaluate fails loudly so the OLAP route can
        // serve it, instead of returning a silently-wrong empty result.
        let pred_err: std::sync::Mutex<Option<ExprError>> = std::sync::Mutex::new(None);
        let record_pred = |record: &ProximaRecord| -> bool {
            match Self::project_one_record(record, &table_schema, &full_selected) {
                Ok(full_row) => match full_row_predicate {
                    Some(p) => match p(&full_row) {
                        Ok(keep) => keep,
                        Err(e) => {
                            let mut slot = pred_err.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            false
                        }
                    },
                    None => true,
                },
                Err(_) => false,
            }
        };
        // FA-c Phase 2: AND-combine the ABAC row filter with the user predicate.
        let combined_pred = move |record: &ProximaRecord| -> bool {
            record_pred(record) && abac_pred.as_ref().is_none_or(|p| p(record))
        };
        let predicate: Option<&proximadb_records::RecordScanPredicate<'_>> = Some(&combined_pred);

        let records = self
            .record_store
            .scan_records_filtered(
                &table_schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: table_id_name.clone(),
                    limit,
                    include_vector: true,
                    include_props: true,
                },
                predicate,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
            )
            .await?;

        if let Some(e) = pred_err.lock().unwrap_or_else(|p| p.into_inner()).take() {
            return Err(anyhow!("predicate evaluation failed: {e}"));
        }

        let rows = Self::project_select_rows(&records, &table_schema, &output_selected)?;
        Ok((table_schema, rows))
    }

    /// Execute an exact vector search against a catalog table's authoritative
    /// `TableRecordStore` state.
    ///
    /// SQL tables with vector columns keep their rows in the relational primary
    /// store; the vector layout is a rebuildable projection. Searching the
    /// legacy vector WAL here would therefore read a different authority and can
    /// return a false empty result immediately after a successful SQL INSERT.
    /// `Ok(None)` means the identifier is not a catalog table, allowing the query
    /// router to try the native collection authority instead.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_table_vectors_exact(
        &self,
        table_name: &str,
        vector_column: &str,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&proximadb_filter_expression::FilterExpression>,
        metric: proximadb_distance_types::DistanceMetric,
        metadata_columns: Option<&[String]>,
        tenant_context: Option<&TenantContext>,
        identity: PortIdentity<'_>,
        controls: &ExecutionControls,
    ) -> Result<Option<Vec<proximadb_search_types::results::OptimizedSearchRecord>>> {
        let started = std::time::Instant::now();
        if top_k == 0 {
            return Err(anyhow!("vector-search top_k must be greater than zero"));
        }
        if query_vector.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!("vector-search query components must be finite"));
        }
        let tenant = tenant_context.map(|context| context.tenant_id.as_str());
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;
        if !catalog.table_exists(&table_id).await? {
            return Ok(None);
        }
        let schema = catalog.get_table(&table_id).await?;
        let vector_index = schema
            .columns
            .iter()
            .position(|column| column.name.eq_ignore_ascii_case(vector_column))
            .ok_or_else(|| {
                anyhow!(
                    "vector column '{}' does not exist on table '{}'",
                    vector_column,
                    table_name
                )
            })?;
        let vector_schema = &schema.columns[vector_index];
        if !matches!(vector_schema.data_type, ProximaType::DenseVector { .. }) {
            return Err(anyhow!(
                "ORDER BY vector operator requires a dense-vector column; '{}.{}' is {:?}",
                table_name,
                vector_column,
                vector_schema.data_type
            ));
        }
        if let Some(expected) = vector_schema
            .properties
            .get("dimension")
            .and_then(|value| value.parse::<usize>().ok())
            && expected != query_vector.len()
        {
            return Err(anyhow!(
                "query vector has dimension {}, but '{}.{}' requires {}",
                query_vector.len(),
                table_name,
                vector_column,
                expected
            ));
        }
        if let Some(columns) = metadata_columns {
            for column in columns {
                if !schema
                    .columns
                    .iter()
                    .any(|candidate| candidate.name.eq_ignore_ascii_case(column))
                {
                    return Err(anyhow!(
                        "column '{}' does not exist on table '{}'",
                        column,
                        table_name
                    ));
                }
            }
        }

        let filter_columns = schema.columns.clone();
        let filter_predicate = |row: &[ProximaValue]| -> Result<bool, ExprError> {
            let Some(filter) = filter else {
                return Ok(true);
            };
            let props = filter_columns
                .iter()
                .zip(row)
                .map(|(column, value)| (column.name.clone(), ProximaTreeNode::Value(value.clone())))
                .collect();
            Ok(crate::core::search::sql_value_filter::evaluate_filter_proxima(filter, &props))
        };
        let (scan_schema, rows) = self
            .scan_table_relational(
                table_name,
                None,
                Some(&filter_predicate),
                None,
                tenant_context,
                identity,
            )
            .await?;
        let scanned_rows = rows.len();

        let id_columns = if scan_schema.primary_key.is_empty() {
            scan_schema
                .columns
                .iter()
                .find(|column| {
                    column.name.eq_ignore_ascii_case("id")
                        || column.name.eq_ignore_ascii_case("record_id")
                })
                .map(|column| vec![column.name.clone()])
                .ok_or_else(|| {
                    anyhow!(
                        "vector-search table '{}' has no record identifier",
                        table_name
                    )
                })?
        } else {
            scan_schema.primary_key.clone()
        };
        let id_indices = id_columns
            .iter()
            .map(|id_column| {
                scan_schema
                    .columns
                    .iter()
                    .position(|column| column.name.eq_ignore_ascii_case(id_column))
                    .ok_or_else(|| {
                        anyhow!(
                            "identifier column '{}' is absent from scan schema",
                            id_column
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let kernel_metric = match metric {
            proximadb_distance_types::DistanceMetric::L2 => {
                proximadb_distance_kernel::DistanceMetric::Euclidean
            }
            proximadb_distance_types::DistanceMetric::Cosine => {
                proximadb_distance_kernel::DistanceMetric::Cosine
            }
            proximadb_distance_types::DistanceMetric::InnerProduct => {
                proximadb_distance_kernel::DistanceMetric::DotProduct
            }
            proximadb_distance_types::DistanceMetric::L1 => {
                proximadb_distance_kernel::DistanceMetric::Manhattan
            }
        };
        let distance =
            proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(kernel_metric);
        let mut results = proximadb_search_types::bounded_queue::BoundedPriorityQueue::new(top_k);
        for (row_index, row) in rows.into_iter().enumerate() {
            if row_index % 256 == 0 {
                controls.check_cancelled().map_err(|error| anyhow!(error))?;
            }
            let vector = match row.get(vector_index) {
                Some(ProximaValue::DenseVector(vector)) => vector,
                Some(ProximaValue::Null) | None => continue,
                Some(other) => {
                    return Err(anyhow!(
                        "vector column '{}.{}' produced non-vector value {:?}",
                        table_name,
                        vector_column,
                        other
                    ));
                }
            };
            if vector.len() != query_vector.len() {
                return Err(anyhow!(
                    "stored vector in '{}.{}' has dimension {}, expected {}",
                    table_name,
                    vector_column,
                    vector.len(),
                    query_vector.len()
                ));
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(anyhow!(
                    "stored vector in '{}.{}' contains a non-finite component",
                    table_name,
                    vector_column
                ));
            }
            let semantic = distance.calculate_distance(query_vector, vector, &kernel_metric);
            if !results.would_accept(semantic.normalized_score) {
                continue;
            }
            let id_values = id_indices
                .iter()
                .map(|index| {
                    row.get(*index).cloned().ok_or_else(|| {
                        anyhow!("identifier column is absent from vector-search row")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let id = encode_primary_key_tuple(&id_values)?;
            let metadata = scan_schema
                .columns
                .iter()
                .zip(&row)
                .filter(|(column, _)| {
                    !matches!(
                        column.data_type,
                        ProximaType::DenseVector { .. }
                            | ProximaType::SparseVector { .. }
                            | ProximaType::BinaryVector { .. }
                    )
                })
                .filter(|(column, _)| {
                    metadata_columns.is_none_or(|requested| {
                        requested
                            .iter()
                            .any(|name| column.name.eq_ignore_ascii_case(name))
                    })
                })
                .map(|(column, value)| (column.name.clone(), value.clone()))
                .collect();
            results.try_insert(proximadb_search_types::results::OptimizedSearchRecord {
                id: id.clone(),
                vector_id: Some(id),
                score: semantic.normalized_score,
                similarity: Some(semantic.normalized_score),
                vector: None,
                semantic_similarity: Some(semantic),
                metadata,
                ..Default::default()
            });
        }
        controls.check_cancelled().map_err(|error| anyhow!(error))?;
        let results = results.into_sorted_vec();
        debug!(
            target: "proximadb::query_route",
            authority = "catalog_table",
            route = "exact_vector_scan",
            table = table_name,
            vector_column,
            scanned_rows,
            returned_rows = results.len(),
            elapsed_us = started.elapsed().as_micros() as u64,
            "completed catalog-table vector search"
        );
        Ok(Some(results))
    }

    /// Build a relational `ProximaRecord` from a full column-ordered value row:
    /// every non-NULL column lands in `props` keyed by name (the warehouse
    /// materializer's record shape). Shared by `MATERIALIZE` (cold base) and the
    /// ADR-025 OLAP read-merge (live appends) so both encode identically through
    /// `proxima_records_to_record_batch`.
    pub(crate) fn value_row_to_relational_record(
        oid: &str,
        col_names: &[String],
        row: Vec<ProximaValue>,
    ) -> ProximaRecord {
        let mut rec = ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        };
        for (name, value) in col_names.iter().zip(row) {
            if !matches!(value, ProximaValue::Null) {
                rec.props
                    .insert(name.clone(), ProximaTreeNode::Value(value));
            }
        }
        rec
    }

    /// Materialize a relational table's current rows as a Parquet snapshot on object
    /// storage and flip its catalog storage layout to `Parquet` /
    /// `ProjectionPublication`, so the OLAP router's `catalog_table_is_parquet_backed`
    /// check passes and SELECTs over the table route to DataFusion.
    ///
    /// This is the explicit-publish half of the dual-path design (course-correction
    /// §6 P3): OLTP rows stay authoritative in RecordStorage; this publishes a
    /// read-optimized Parquet projection of the current snapshot. It is triggered
    /// explicitly (e.g. `ALTER TABLE … MATERIALIZE`), not on every write.
    ///
    /// `bridge` is the object store to write into; `warehouse_root_url` is the URL the
    /// OLAP reader reopens that same physical store from, so the published
    /// `location = {warehouse_root_url}/{tenant-isolated prefix}/data` resolves back to
    /// the data files the snapshot's Iceberg manifest names. Returns the published
    /// `location`.
    ///
    /// P2.4: a full rewrite in bounded memory. The rows stream from an ordered,
    /// chunked scan; the cluster-key sort spills past its memory budget; the output
    /// is multi-part (`{generation}-{nnnnn}.parquet`, sized by
    /// `ParquetWritePolicy::target_file_*`); the Iceberg snapshot names exactly
    /// those parts and is committed BEFORE the catalog layout flips, so no reader
    /// is ever pointed at a generation without a manifest. Files of generations
    /// older than the previous version are then deleted.
    pub async fn materialize_table_to_parquet(
        &self,
        bridge: &dyn ObjectStoreBridge,
        warehouse_root_url: &str,
        table_name: &str,
        tenant_context: Option<&TenantContext>,
    ) -> Result<String> {
        self.materialize_table_to_parquet_fenced(
            bridge,
            warehouse_root_url,
            table_name,
            tenant_context,
            &crate::services::ddl::NoPublishFence,
        )
        .await
    }

    /// [`Self::materialize_table_to_parquet`] with a fenced advance (P2.4):
    ///
    /// * one publication per table at a time in this process (a per-table
    ///   publish lock held from before the snapshot LSN is captured to after the
    ///   catalog flip);
    /// * the Iceberg parent version is read before the scan, so a publication
    ///   that another process committed meanwhile makes this commit's
    ///   create-only CAS lose (an error, never a silent overwrite);
    /// * `fence` is re-checked immediately before that commit;
    /// * the catalog flip is forward-only: it never replaces a layout that
    ///   already records a newer Iceberg metadata version.
    pub async fn materialize_table_to_parquet_fenced(
        &self,
        bridge: &dyn ObjectStoreBridge,
        warehouse_root_url: &str,
        table_name: &str,
        tenant_context: Option<&TenantContext>,
        fence: &dyn crate::services::ddl::PublishFence,
    ) -> Result<String> {
        use futures::{SinkExt, StreamExt};

        let _publishing = materialize_stream::publish_lock(
            tenant_context.map(|t| t.tenant_id.as_str()),
            table_name,
        )
        .await;

        // ADR-025: capture the snapshot LSN BEFORE scanning, so any write landing
        // during/after the scan has a strictly greater LSN and is caught by the OLAP
        // read-merge delta (`read_changes_since(.., snapshot_lsn)`). The capture also
        // pins the WAL above that LSN until the catalog records it below (TD-SEAM-12):
        // until then nothing else stops truncation from removing that delta.
        let snapshot = self.record_store.capture_snapshot_lsn(table_name).await?;
        let snapshot_lsn = snapshot.lsn;

        // 1. The table's current rows as an ordered, bounded chunk stream (P2.4):
        //    nothing below holds more than a chunk of records at once.
        let (schema, chunks) = self
            .scan_table_records_chunked(
                table_name,
                materialize_stream::scan_chunk_rows(),
                tenant_context,
                PortIdentity::anonymous(),
            )
            .await?;

        // 2. TD-OLAP-6: cluster the snapshot by the table's cluster key before
        //    publication, so Parquet row groups carry tight min/max bounds and
        //    zone-map / runtime-filter pruning can actually skip (the snapshot
        //    scan otherwise discards row order — every row group spans the whole
        //    domain, measured 0% skip in TPC_PERF_GATE_EVIDENCE_V2_2026_07_04).
        //    Key resolution: explicit `cluster_key` table property, else the
        //    first DATE/TIMESTAMP column (the dominant analytical access shape).
        //    Ordering is reader-neutral (mixed-read-safe); NULL keys sort last.
        let cluster_col = resolve_cluster_key(&schema);

        // 3. Tenant-isolated object prefix (DrPathBuilder mandate: data/{tenant}/{ns}/{table}).
        //    NOTE (tracked tech-debt): this path does NOT yet route through
        //    `DrPathBuilder::build`. The native catalog leaves `namespace_id`/`tenant_id`
        //    unset (pending the P0.5 backfill), so the builder would fail-closed here, and
        //    the DDL `TableMaterializer` trait does not yet thread a `TenantContext`
        //    (production callers pass `None` -> `DEFAULT_TENANT_PLACEHOLDER`). Until that
        //    multi-layer fix lands, isolation on this path is best-effort. We still apply
        //    DrPathBuilder's canonical per-segment validation so a crafted tenant /
        //    namespace / table id cannot inject `..` or path separators into the prefix.
        let scope_tenant = tenant_context.map(|tc| tc.tenant_id.as_str());
        let tenant_id = scope_tenant.unwrap_or(DEFAULT_TENANT_PLACEHOLDER);
        // Resolve the table under the SAME tenant scope the snapshot scan used: a
        // tenant-scoped table lives under a tenant-prefixed namespace, so the
        // unscoped resolve would look in `default` and miss it.
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, scope_tenant)
            .await?;
        // Best-effort namespace metadata (looked up by the FULL scoped namespace):
        // supplies the rename-stable `namespace_id` (DrPath layout) and the owning
        // `tenant_id` (cross-tenant assertion). A miss (embedded / single-tenant with
        // no catalog namespace) uses the well-known `ns_default`.
        let ns_meta = catalog.get_namespace(&table_id.namespace).await.ok();
        let prefix = resolve_materialize_prefix(
            tenant_id,
            ns_meta
                .as_ref()
                .and_then(|n| n.namespace_id.as_deref())
                .unwrap_or(DrPathBuilder::DEFAULT_NAMESPACE_ID),
            ns_meta.as_ref().and_then(|n| n.tenant_id.as_deref()),
            ns_meta
                .as_ref()
                .map(|n| n.storage_pool_class)
                .unwrap_or_default(),
            &schema.name,
            schema.object_id,
        )?;

        // 4. Stream the rows into generation-named parts under `{prefix}/data/`, in
        //    the CATALOG-AUTHORITATIVE schema (not record inference) so the files'
        //    columns/types/nullability match the catalog exactly — including
        //    all-null columns inference would drop. Each chunk is projected to the
        //    full column set (a row that cannot be projected is skipped, as the
        //    relational scan does), folded into the NDV sketches, encoded, and
        //    handed to the sort/writer side over a bounded channel.
        //    The Iceberg parent is fixed BEFORE the rows are read, and the parts are
        //    named for the version this publication commits as (see
        //    `generation_stem`): if another publisher commits meanwhile, the commit
        //    below loses its create-only CAS rather than racing GC.
        let metadata_prefix = format!("{prefix}/metadata");
        let parent = bridge.latest_metadata_version(&metadata_prefix).await?;
        let successor = parent.map_or(0, |p| p.saturating_add(1));
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let stem = materialize_stream::generation_stem(successor);
        let data_dir = format!("{prefix}/data");
        let full_columns = Self::resolve_select_projection(&schema, &[])?;
        let col_names: Vec<String> = full_columns.iter().map(|c| c.name.clone()).collect();
        let arrow_schema = proxima_records_to_record_batch(&[], &schema)?.schema();
        // The cluster sort is declared as Parquet `sorting_columns`, and the
        // warehouse write policy adds the page index + spec SBBF Bloom filters.
        let sort_columns: Vec<String> = cluster_col.iter().cloned().collect();
        let mut writer = MultipartParquetWriter::new(
            bridge,
            data_dir.as_str(),
            stem.as_str(),
            arrow_schema.clone(),
            ParquetWritePolicy::from_env(),
            ParquetLayoutHints {
                sort_columns: sort_columns.clone(),
            },
        );
        let mut ndv = NdvAccumulator::new(
            table_name,
            tenant_context.map(|t| t.tenant_id.as_str()),
            &schema,
        );
        let (tx, rx) = futures::channel::mpsc::channel::<Result<arrow_array::RecordBatch>>(2);
        let produce = async {
            let mut tx = tx;
            let mut chunks = chunks;
            while let Some(chunk) = chunks.next().await {
                let records: Vec<ProximaRecord> = chunk?
                    .iter()
                    .filter_map(|record| {
                        Self::project_one_record(record, &schema, &full_columns)
                            .ok()
                            .map(|row| {
                                Self::value_row_to_relational_record(&record.oid, &col_names, row)
                            })
                    })
                    .collect();
                if records.is_empty() {
                    continue;
                }
                ndv.observe(&records);
                let batch = proxima_records_to_record_batch(&records, &schema)?;
                if tx.send(Ok(batch)).await.is_err() {
                    break; // the writer side failed; its error is reported
                }
            }
            anyhow::Ok(())
        };
        let consume = async {
            let mut batches = match &cluster_col {
                Some(column) => {
                    materialize_stream::sort_batches(rx.boxed(), arrow_schema.clone(), column)?
                }
                None => rx.boxed(),
            };
            while let Some(batch) = batches.next().await {
                writer.write(&batch?).await?;
            }
            anyhow::Ok(())
        };
        if let Err(e) = futures::try_join!(produce, consume) {
            let _ = writer.abort().await;
            return Err(e);
        }
        let parts = writer.finish().await?;
        let part_keys: Vec<object_store::path::Path> =
            parts.iter().map(|p| p.path.clone()).collect();

        // 5. Commit the spec-shaped Iceberg snapshot (Avro manifest + manifest list +
        //    TableMetadata under `{prefix}/metadata`) naming exactly these parts,
        //    BEFORE the catalog flip. ProximaDB's own readers select files from this
        //    manifest (they no longer list `data/`), and so do external Iceberg
        //    engines, so a failure here fails the MATERIALIZE: the parts are
        //    removed and the previous publication stays current.
        let iceberg_fields: Vec<
            proximadb_storage_common::object_store_bridge::IcebergSnapshotField,
        > = schema
            .to_arrow_schema()
            .fields()
            .iter()
            .enumerate()
            .map(
                |(i, f)| proximadb_storage_common::object_store_bridge::IcebergSnapshotField {
                    id: (i + 1) as i32,
                    name: f.name().clone(),
                    type_name: iceberg_type_for(f.data_type()),
                    required: !f.is_nullable(),
                },
            )
            .collect();
        let table_uuid = deterministic_table_uuid(&prefix);
        // The Iceberg table `location` is the table ROOT (`data/` and `metadata/` are
        // its children), unlike the catalog layout location below, which is the
        // directory of the data files.
        let table_root = format!("{}/{prefix}", warehouse_root_url.trim_end_matches('/'));
        // Fenced advance: the authority to publish is re-checked right before the
        // commit, which is the point after which the new snapshot is visible.
        if let Err(e) = fence.check().await {
            Self::delete_parts(bridge, &part_keys).await;
            return Err(e.context(format!(
                "MATERIALIZE {table_name}: publish fence refused the commit"
            )));
        }
        let committed = bridge
            .publish_iceberg_files(
                &part_keys,
                parent,
                &metadata_prefix,
                &table_uuid,
                &table_root,
                &iceberg_fields,
                now_ms, // snapshot id (unique per materialize)
                now_ms, // timestamp ms
                iceberg_needs_v3(&iceberg_fields),
            )
            .await;
        let version = match committed {
            Ok(version) => version,
            Err(e) => {
                if let Some(counter) = ICEBERG_PUBLISH_FAILURES.as_ref() {
                    counter.inc();
                }
                // The error may have come after the commit landed: only parts that
                // no version names are deleted.
                match materialize_stream::check_commit(
                    bridge,
                    &metadata_prefix,
                    successor,
                    &part_keys,
                )
                .await
                {
                    materialize_stream::CommitCheck::Landed(version) => {
                        warn!(
                            table = %table_name,
                            version,
                            "MATERIALIZE: the Iceberg commit reported an error but landed: {e}"
                        );
                        version
                    }
                    check => {
                        if check == materialize_stream::CommitCheck::NotLanded {
                            Self::delete_parts(bridge, &part_keys).await;
                        }
                        return Err(anyhow!(
                            "MATERIALIZE {table_name}: Iceberg snapshot commit failed; the \
                             previous publication stays current: {e}"
                        ));
                    }
                }
            }
        };

        // 6. Flip the catalog layout to a published Parquet projection at the location.
        // ADR-059 first-principles: the `location` is the directory where the
        // data files are IMMEDIATE children (Hive/Unity/Polaris model) — NOT the
        // table base dir with a `data/` subpath.
        let location = format!("{}/{data_dir}", warehouse_root_url.trim_end_matches('/'));
        let layout = CatalogStorageLayout {
            name: "parquet-snapshot".to_string(),
            authority: proximadb_catalog::CatalogAuthorityMode::ProjectionPublication,
            physical_format: proximadb_catalog::CatalogPhysicalFormat::Parquet,
            location: Some(location.clone()),
            // ADR-025: record the snapshot LSN so the OLAP read-merge can reconcile
            // this cold base against post-snapshot writes (`read_changes_since`).
            properties: std::collections::HashMap::from([
                ("snapshot_lsn".to_string(), snapshot_lsn.to_string()),
                // Forward-only flips compare this (see `layout_is_newer`).
                (
                    materialize_stream::METADATA_VERSION_PROPERTY.to_string(),
                    version.to_string(),
                ),
            ]),
            ..Default::default()
        };
        // TD-USUB-4: additive publication. Read the table's current layouts and upsert
        // this one by name, instead of replacing the list. Passing `vec![layout]` here
        // dropped every other representation the table had, which is what made
        // cost-based routing across representations inexpressible (ADR-094 Decision 2).
        //
        // Read fresh from the catalog rather than reusing the pre-write `schema`, so a
        // layout added between the snapshot scan and here is not lost. A failed read
        // degrades to "no existing layouts" — the pre-TD-USUB-4 behaviour — rather than
        // failing the materialize.
        let existing_layouts = catalog
            .get_table(&table_id)
            .await
            .map(|s| s.storage_layouts)
            .unwrap_or_default();
        //
        // Forward-only (P2.4): a publication whose commit was overtaken — another
        // publisher committed a later metadata version and already flipped the
        // catalog to it — must not move the layout (and its `snapshot_lsn`) back.
        let overtaken = materialize_stream::layout_is_newer(&existing_layouts, &layout);
        if overtaken {
            info!(
                table = %table_name,
                version,
                "MATERIALIZE: a newer publication already flipped the catalog; keeping it"
            );
        } else {
            let layouts = upsert_storage_layout(existing_layouts, layout);
            catalog.set_storage_layouts(&table_id, layouts).await?;
        }
        // The catalog now records `snapshot_lsn`; published snapshots hold the WAL
        // truncation floor from here on.
        drop(snapshot);

        // 6b. TD-OLAP-2 (A2): publish per-column NDV sketches (HLL), folded chunk by
        //     chunk during the scan, to the ADR-037 resident statistics registry,
        //     keyed by table name. The DF reader's `statistics()` overlays
        //     `distinct_count` from this registry, so `EnforceDistribution` can
        //     pick a partitioned (multi-core) join build. Only after the commit +
        //     catalog flip, so stats describe data that actually landed — and not
        //     when the flip was overtaken, so they never replace the newer
        //     publication's.
        if !overtaken {
            ndv.publish();
        }

        // 7. Garbage-collect generations older than the previous version. Best
        //    effort: a leftover file is unreferenced (readers select from the
        //    manifest), so a failure here only costs storage until the next run.
        let keep_current: std::collections::HashSet<String> =
            part_keys.iter().map(|p| p.to_string()).collect();
        let keep_previous: std::collections::HashSet<String> = match parent {
            Some(previous) => match bridge.read_iceberg_table(&metadata_prefix, previous).await {
                Ok(files) => files.iter().map(|p| p.to_string()).collect(),
                Err(e) => {
                    warn!(table = %table_name, "MATERIALIZE GC skipped: {e}");
                    return Ok(location);
                }
            },
            None => Default::default(),
        };
        match materialize_stream::gc_older_generations(
            bridge,
            &object_store::path::Path::from(data_dir.as_str()),
            &keep_current,
            &keep_previous,
            &stem,
        )
        .await
        {
            Ok(deleted) => debug!(table = %table_name, version, deleted, "MATERIALIZE GC"),
            Err(e) => warn!(table = %table_name, "MATERIALIZE GC incomplete: {e}"),
        }

        Ok(location)
    }

    /// Remove the parts of a publication that failed to commit (best effort:
    /// they are unreferenced either way).
    async fn delete_parts(bridge: &dyn ObjectStoreBridge, parts: &[object_store::path::Path]) {
        let store = bridge.inner_store();
        for part in parts {
            let _ = object_store::ObjectStoreExt::delete(
                store.as_ref(),
                &bridge.object_store_path(part),
            )
            .await;
        }
    }

    /// Point-lookup a single relational row by primary key, projected into the
    /// FULL `schema.columns` order (the executor re-applies any projection).
    ///
    /// Extends the canonical [`DmlService`]; the OLTP point-read fast path for the
    /// relational pipeline (PATH B) — the storage-side backing for
    /// `ScanAccess::PkLookup`. Single-column PK only (`get_by_key` keys on
    /// `record.oid`, matching the SELECT PK fast-path's `primary_key.first()`).
    /// ADR-018 Phase 2 (pgwire SQL parity); TD-076. Reuses the exact primitives the
    /// SELECT PK fast-path uses (`get_by_key` + `rich_result_to_record`).
    pub async fn point_lookup_relational(
        &self,
        table_name: &str,
        key: &str,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<Vec<ProximaValue>>> {
        let (table_schema, table_id_name) = self
            .resolve_select_table(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;
        let all_columns: Vec<String> = table_schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let full_selected = Self::resolve_select_projection(&table_schema, &all_columns)?;
        let record = self
            .record_store
            .get_by_key(
                &table_schema,
                TableRecordGetRequest {
                    table_id: table_id_name,
                    key: key.to_string(),
                    include_vector: true,
                    include_props: true,
                },
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
            )
            .await?;
        match record {
            Some(rich) => {
                let record = Self::rich_result_to_record(rich);
                let row = Self::project_one_record(&record, &table_schema, &full_selected)?;
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    /// TD-128: discrete multi-key OLTP point-read for the Volcano relational
    /// pipeline. Resolves each PK in `keys` via [`TableRecordStore::get_by_key`]
    /// (which applies the canonical dead-record filter — invariant #16), and
    /// returns the FULL projected row for every key that hits a live record.
    /// Missing / dead keys are simply absent — the result is NOT positionally
    /// aligned with `keys`. Single-column PK only (mirrors
    /// [`Self::point_lookup_relational`]); reuses the same `get_by_key` path the
    /// legacy native DML PK fast-path uses.
    pub async fn point_lookup_batch_relational(
        &self,
        table_name: &str,
        keys: &[String],
        tenant_context: Option<&TenantContext>,
    ) -> Result<Vec<Vec<ProximaValue>>> {
        let (table_schema, table_id_name) = self
            .resolve_select_table(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;
        let all_columns: Vec<String> = table_schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let full_selected = Self::resolve_select_projection(&table_schema, &all_columns)?;
        let mut rows = Vec::with_capacity(keys.len());
        for key in keys {
            let record = self
                .record_store
                .get_by_key(
                    &table_schema,
                    TableRecordGetRequest {
                        table_id: table_id_name.clone(),
                        key: key.clone(),
                        include_vector: true,
                        include_props: true,
                    },
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
                )
                .await?;
            if let Some(rich) = record {
                let record = Self::rich_result_to_record(rich);
                rows.push(Self::project_one_record(
                    &record,
                    &table_schema,
                    &full_selected,
                )?);
            }
        }
        Ok(rows)
    }

    /// TD-127: secondary-index point-read for the Volcano relational pipeline.
    /// Probes the store's single-column non-PK hash index for the rows whose
    /// `column` value (canonical text) is in `values`, then returns the FULL
    /// projected row for each live candidate. Reuses the exact
    /// [`TableRecordStore::lookup_secondary`] + [`TableRecordStore::get_by_key`]
    /// path the legacy native DML secondary fast-path uses (so both agree on
    /// which columns are indexed and on dead-record filtering — invariant #16).
    ///
    /// Returns `Ok(None)` when the store has no built index for `column` (the
    /// caller falls back to a full scan); `Ok(Some(rows))` (possibly empty) when
    /// the index answered. The index only NARROWS — the caller's residual filter
    /// re-checks every candidate.
    pub async fn secondary_lookup_relational(
        &self,
        table_name: &str,
        column: &str,
        values: &[String],
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<Vec<Vec<ProximaValue>>>> {
        let (table_schema, table_id_name) = self
            .resolve_select_table(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;
        let value_set: std::collections::HashSet<String> = values.iter().cloned().collect();
        let Some(candidate_oids) = self
            .record_store
            .lookup_secondary(&table_schema, column, &value_set, tenant_context)
            .await?
        else {
            return Ok(None);
        };
        let all_columns: Vec<String> = table_schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let full_selected = Self::resolve_select_projection(&table_schema, &all_columns)?;
        let mut rows = Vec::with_capacity(candidate_oids.len());
        for oid in candidate_oids {
            let record = self
                .record_store
                .get_by_key(
                    &table_schema,
                    TableRecordGetRequest {
                        table_id: table_id_name.clone(),
                        key: oid,
                        include_vector: true,
                        include_props: true,
                    },
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
                )
                .await?;
            if let Some(rich) = record {
                let record = Self::rich_result_to_record(rich);
                rows.push(Self::project_one_record(
                    &record,
                    &table_schema,
                    &full_selected,
                )?);
            }
        }
        Ok(Some(rows))
    }

    /// Records-returning, limit-honoring twin of [`Self::resolve_matching_ids`] for
    /// the SELECT path: PK fast-path (OR-safe via [`Self::extract_pk_candidate_ids`],
    /// re-checked against the full tree) else a predicate scan with the limit pushed in.
    async fn select_table_records_with_tree(
        &self,
        table_schema: &CatalogTableSchema,
        table_id_name: &str,
        limit: Option<usize>,
        where_clause: &WhereClause,
        tree: &RelationalPredicateTree,
        tenant_context: Option<&TenantContext>,
    ) -> Result<(RelationalSelectAccessPath, Vec<ProximaRecord>)> {
        let primary_key = table_schema.primary_key.first().map(String::as_str);

        // PK fast-path: only fires under a top-level conjunction (extract_pk_candidate_ids
        // returns empty under top-level OR, so `id = 5 OR status = 'x'` correctly scans).
        // Each candidate is re-checked against the FULL tree before counting as a match.
        let pk_candidates = self.extract_pk_candidate_ids(where_clause, table_schema)?;
        // TD-RELIDEQ-1: with no declared key the "key" is the `id`/`record_id`
        // fallback, which equals the record oid only for collections written
        // through the record API. A keyless pgwire row has a generated oid, so a
        // miss there proves nothing: fall through to the predicate scan.
        let key_is_declared = Self::table_declares_primary_key(table_schema);
        let mut fallback_key_missed = false;
        if !pk_candidates.is_empty() {
            let cap = limit.unwrap_or(usize::MAX);
            let mut records = Vec::new();
            for candidate in pk_candidates {
                if records.len() >= cap {
                    break;
                }
                let Some(rich) = self
                    .record_store
                    .get_by_key(
                        table_schema,
                        TableRecordGetRequest {
                            table_id: table_id_name.to_string(),
                            key: candidate,
                            include_vector: true,
                            include_props: true,
                        },
                        tenant_context,
                        #[cfg(feature = "abac-policy")]
                        &ReadContext::system(
                            SystemReadReason::Statistics,
                            "dml [CLIENT-PLACEHOLDER]",
                        ),
                    )
                    .await?
                else {
                    fallback_key_missed |= !key_is_declared;
                    continue;
                };
                let record = Self::rich_result_to_record(rich);
                if Self::eval_predicate_tree(&record, tree, primary_key) {
                    records.push(record);
                }
            }
            if !fallback_key_missed {
                return Ok((RelationalSelectAccessPath::PrimaryKeyLookup, records));
            }
        }

        // TD-127 secondary-index fast-path: a single-column equality / IN-list on
        // a non-PK secondary-indexed column → probe the index for candidate oids,
        // then re-check each against the FULL tree (the index only narrows; the
        // tree still decides). `None` from `lookup_secondary` (no built index /
        // kill-switch) falls through to the scan below.
        if let Some((column, values)) =
            self.extract_secondary_index_probe(where_clause, table_schema)?
        {
            let value_set: std::collections::HashSet<String> = values.into_iter().collect();
            if let Some(candidate_oids) = self
                .record_store
                .lookup_secondary(table_schema, &column, &value_set, tenant_context)
                .await?
            {
                let cap = limit.unwrap_or(usize::MAX);
                let mut records = Vec::new();
                for oid in candidate_oids {
                    if records.len() >= cap {
                        break;
                    }
                    let Some(rich) = self
                        .record_store
                        .get_by_key(
                            table_schema,
                            TableRecordGetRequest {
                                table_id: table_id_name.to_string(),
                                key: oid,
                                include_vector: true,
                                include_props: true,
                            },
                            tenant_context,
                            #[cfg(feature = "abac-policy")]
                            &ReadContext::system(
                                SystemReadReason::Statistics,
                                "dml [CLIENT-PLACEHOLDER]",
                            ),
                        )
                        .await?
                    else {
                        continue;
                    };
                    let record = Self::rich_result_to_record(rich);
                    if Self::eval_predicate_tree(&record, tree, primary_key) {
                        records.push(record);
                    }
                }
                return Ok((RelationalSelectAccessPath::SecondaryIndexLookup, records));
            }
        }

        // No usable PK predicate: push the full tree into the store scan + limit.
        let pred = |record: &ProximaRecord| Self::eval_predicate_tree(record, tree, primary_key);
        let predicate: Option<&proximadb_records::RecordScanPredicate<'_>> = Some(&pred);
        let records = self
            .record_store
            .scan_records_filtered(
                table_schema,
                TableRecordScanRequest {
                    // Block/row-group pushdown for object-store vector tables; the
                    // `predicate` closure remains the row-exact authority.
                    filter: predicate_tree_to_filter_expression(tree),
                    table_id: table_id_name.to_string(),
                    limit,
                    include_vector: true,
                    include_props: true,
                },
                predicate,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
            )
            .await?;
        Ok((RelationalSelectAccessPath::TableScan, records))
    }

    /// Select current visible records for simple catalog-table reads.
    ///
    /// This is the relational read routing boundary for pgwire's current
    /// compatibility SELECT path: it chooses point lookup when catalog-shaped
    /// predicates contain primary-key equality, otherwise it uses the
    /// table-record scan contract.
    pub async fn select_table_records(
        &self,
        table_name: &str,
        limit: Option<usize>,
        predicates: &[RelationalSelectPredicateInput],
    ) -> Result<(CatalogTableSchema, Vec<ProximaRecord>)> {
        // Not on the pgwire tenant read path (pgwire SELECT uses the
        // `_with_projection*` methods); resolve unscoped.
        let (table_schema, table_id_name) = self.resolve_select_table(table_name, None).await?;
        let predicates = Self::resolve_select_predicates(&table_schema, predicates)?;
        let (_, records) = self
            .select_table_records_with_resolved_predicates(
                &table_schema,
                &table_id_name,
                limit,
                &predicates,
                None,
            )
            .await?;
        Ok((table_schema, records))
    }

    /// Resolve a table name to its canonical collection id (`table_id.name`) —
    /// the key the write/flush path stamps statistics under (so an EXPLAIN
    /// statistics lookup uses the same key). `None` if the table can't be
    /// resolved. Used by the relational EXPLAIN selectivity disclosure (TD-174).
    pub(crate) async fn resolve_collection_id(
        &self,
        table_name: &str,
        tenant: Option<&str>,
    ) -> Option<String> {
        self.resolve_select_table(table_name, tenant)
            .await
            .ok()
            .map(|(_, collection_id)| collection_id)
    }

    async fn resolve_select_table(
        &self,
        table_name: &str,
        tenant: Option<&str>,
    ) -> Result<(CatalogTableSchema, String)> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{table_name}' does not exist"));
        }

        Ok((catalog.get_table(&table_id).await?, table_id.name.clone()))
    }

    async fn select_table_records_with_resolved_predicates(
        &self,
        table_schema: &CatalogTableSchema,
        table_id_name: &str,
        limit: Option<usize>,
        predicates: &[RelationalSelectPredicate],
        tenant_context: Option<&TenantContext>,
    ) -> Result<(RelationalSelectAccessPath, Vec<ProximaRecord>)> {
        if let Some(primary_key_value) = Self::primary_key_lookup_value(table_schema, predicates) {
            let record = self
                .record_store
                .get_by_key(
                    table_schema,
                    TableRecordGetRequest {
                        table_id: table_id_name.to_string(),
                        key: primary_key_value,
                        include_vector: true,
                        include_props: true,
                    },
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
                )
                .await?;
            let primary_key = table_schema.primary_key.first().map(String::as_str);
            return Ok((
                RelationalSelectAccessPath::PrimaryKeyLookup,
                record
                    .map(Self::rich_result_to_record)
                    .filter(|record| {
                        Self::record_matches_select_predicates(record, predicates, primary_key)
                    })
                    .into_iter()
                    .take(limit.unwrap_or(usize::MAX))
                    .collect(),
            ));
        }

        // Push the resolved `WHERE` predicate + limit INTO the scan so the store
        // filters during iteration and stops at the limit — instead of cloning
        // the whole table (embeddings included) and filtering in memory here.
        let primary_key = table_schema.primary_key.first().map(String::as_str);
        let pred = |record: &ProximaRecord| {
            Self::record_matches_select_predicates(record, predicates, primary_key)
        };
        let predicate: Option<&proximadb_records::RecordScanPredicate<'_>> =
            if predicates.is_empty() {
                None
            } else {
                Some(&pred)
            };
        let records = self
            .record_store
            .scan_records_filtered(
                table_schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: table_id_name.to_string(),
                    limit,
                    include_vector: true,
                    include_props: true,
                },
                predicate,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
            )
            .await?;

        Ok((RelationalSelectAccessPath::TableScan, records))
    }

    /// Explain table-to-table write routing without executing the write.
    pub async fn explain_table_write(
        &self,
        statement: DmlStatement,
    ) -> Result<TableWriteRouteExplanation> {
        self.explain_table_write_with_overrides(statement, None)
            .await
    }

    /// Plan the write and then execute it, returning the route explanation enriched with actual
    /// wall-clock time and rows written. Used for `EXPLAIN ANALYZE <DML>`.
    pub async fn explain_analyze_table_write(
        &self,
        statement: DmlStatement,
    ) -> Result<TableWriteRouteExplanation> {
        let mut explanation = self.explain_table_write(statement.clone()).await?;
        let t0 = std::time::Instant::now();
        let result = self.execute(statement).await?;
        let elapsed_us = t0.elapsed().as_micros() as u64;
        explanation.execution_elapsed_us = Some(elapsed_us);
        explanation.execution_rows_written = Some(result.rows_affected);
        Ok(explanation)
    }

    /// Explain table-to-table write routing with protocol/session write intent hints.
    pub async fn explain_table_write_with_overrides(
        &self,
        statement: DmlStatement,
        write_intent_overrides: Option<&WriteIntentOverrides>,
    ) -> Result<TableWriteRouteExplanation> {
        match statement {
            DmlStatement::InsertSelect { plan, columns }
            | DmlStatement::InsertOverwrite { plan, columns } => {
                let routed = self
                    .route_table_write_plan_with_overrides(&plan, &columns, write_intent_overrides)
                    .await?;
                Ok(routed.route_explanation())
            }
            // VALUES-based DML: synthesize a CopyIntoPlan for route planning.
            // ReadSource::QuerySql acts as a sentinel for inline-values writes;
            // the row count hint drives bulk-vs-WAL lane selection.
            DmlStatement::Insert {
                table_name,
                columns,
                values,
            } => {
                let row_count = values.len() as u64;
                let plan = CopyIntoPlan::insert_select(LogicalTableRef::new(&table_name), "VALUES");
                let mut overrides = write_intent_overrides.cloned().unwrap_or_default();
                overrides.row_count_hint = Some(row_count);
                let plan = CopyIntoPlan {
                    write_mode: WriteMode::InsertOnly,
                    ..plan
                };
                let routed = self
                    .route_table_write_plan_with_overrides(&plan, &columns, Some(&overrides))
                    .await?;
                Ok(routed.route_explanation())
            }
            DmlStatement::Upsert {
                table_name,
                columns,
                values,
                ..
            } => {
                let row_count = values.len() as u64;
                let mut overrides = write_intent_overrides.cloned().unwrap_or_default();
                overrides.row_count_hint = Some(row_count);
                let plan = CopyIntoPlan {
                    source: ReadSource::QuerySql("VALUES".to_string()),
                    target: LogicalTableRef::new(&table_name),
                    write_mode: WriteMode::Upsert,
                    conflict_policy: ConflictPolicy::Upsert,
                    distribution: DistributionMode::Auto,
                };
                let routed = self
                    .route_table_write_plan_with_overrides(&plan, &columns, Some(&overrides))
                    .await?;
                Ok(routed.route_explanation())
            }
            DmlStatement::Update { table_name, .. } => {
                let plan = CopyIntoPlan {
                    source: ReadSource::QuerySql("UPDATE".to_string()),
                    target: LogicalTableRef::new(&table_name),
                    write_mode: WriteMode::Upsert,
                    conflict_policy: ConflictPolicy::Upsert,
                    distribution: DistributionMode::Auto,
                };
                let routed = self
                    .route_table_write_plan_with_overrides(&plan, &[], write_intent_overrides)
                    .await?;
                Ok(routed.route_explanation())
            }
            DmlStatement::Delete { table_name, .. } => {
                let plan = CopyIntoPlan {
                    source: ReadSource::QuerySql("DELETE".to_string()),
                    target: LogicalTableRef::new(&table_name),
                    write_mode: WriteMode::Append,
                    conflict_policy: ConflictPolicy::Error,
                    distribution: DistributionMode::Auto,
                };
                let routed = self
                    .route_table_write_plan_with_overrides(&plan, &[], write_intent_overrides)
                    .await?;
                Ok(routed.route_explanation())
            }
        }
    }

    /// Resolve catalog metadata and route table-to-table writes before execution.
    async fn plan_table_write(
        &self,
        plan: &CopyIntoPlan,
        target_columns: &[String],
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        let (table_schema, target_stats) = self
            .resolve_table_metadata(
                &plan.target.qualified_name(),
                tenant_context.map(|tc| tc.tenant_id.as_str()),
            )
            .await?;
        let (_catalog, target_table_id) = self
            .catalog_manager
            .resolve_table_scoped(
                &plan.target.qualified_name(),
                tenant_context.map(|tc| tc.tenant_id.as_str()),
            )
            .await?;
        let dml_lock_guard = self
            .acquire_table_dml_lock(&target_table_id, tenant_context, LockIntent::Write)
            .await?;
        let source_metadata = self
            .resolve_table_write_source_metadata(
                plan,
                tenant_context.map(|tc| tc.tenant_id.as_str()),
            )
            .await?;
        let source_schema = source_metadata.as_ref().map(|(schema, _)| schema);
        let source_stats = source_metadata.as_ref().map(|(_, stats)| stats);
        let routed = self.route_table_write_with_schemas(
            plan,
            target_columns,
            &table_schema,
            Some(&target_stats),
            source_schema,
            source_stats,
            None,
        )?;

        // Fold the namespace/DrPath-aware prefix into the warehouse bulk-append write
        // so its object layout matches the materialize path
        // (`data/{tenant}/{namespace}/{table}`) instead of the flat
        // `data/{tenant}/tables/{name}` fallback the executor derives without
        // namespace context. Only for the bulk-append lane, only under a tenant
        // scope, and only when no location is already pinned (a materialize-set
        // primary-layout location keeps precedence in the executor → no
        // double-prefix). Set on the local, non-persisted schema copy.
        let mut table_schema = table_schema;
        if routed.write_lane_decision.lane == WriteLane::BulkAppendCommit
            && table_schema.location.is_none()
            && let Some(prefix) = self
                .resolve_warehouse_object_prefix(
                    &plan.target.qualified_name(),
                    tenant_context.map(|tc| tc.tenant_id.as_str()),
                )
                .await?
        {
            table_schema.location = Some(prefix);
        }

        let execution = self
            .table_write_executor
            .execute(TableWriteExecutionRequest {
                target_schema: &table_schema,
                source_schema,
                routed_plan: routed,
                tenant_context,
            })
            .await?;

        match execution.status {
            TableWriteExecutionStatus::Completed => {
                if let Some(guard) = dml_lock_guard {
                    guard.release().await;
                }
                Ok(DmlResult::success(
                    execution.rows_written,
                    format!("Table write completed through {}", execution.route_summary),
                ))
            }
            TableWriteExecutionStatus::PlannedOnly => Err(anyhow!(
                "INSERT ... SELECT and INSERT OVERWRITE execution is not implemented yet; planned route: {}, guards={:?}",
                execution.route_summary,
                execution.guards
            )),
        }
    }

    async fn route_table_write_plan_with_overrides(
        &self,
        plan: &CopyIntoPlan,
        target_columns: &[String],
        write_intent_overrides: Option<&WriteIntentOverrides>,
    ) -> Result<RoutedExecutionPlan> {
        let target_table_name = plan.target.qualified_name();
        // Route/EXPLAIN planning is not a write boundary; resolve unscoped (the
        // execution path `plan_table_write` resolves scoped via the tenant).
        let (table_schema, target_stats) = self
            .resolve_table_metadata(&target_table_name, None)
            .await?;
        let source_metadata = self.resolve_table_write_source_metadata(plan, None).await?;
        let source_schema = source_metadata.as_ref().map(|(schema, _)| schema);
        let source_stats = source_metadata.as_ref().map(|(_, stats)| stats);
        self.route_table_write_with_schemas(
            plan,
            target_columns,
            &table_schema,
            Some(&target_stats),
            source_schema,
            source_stats,
            write_intent_overrides,
        )
    }

    async fn resolve_table_metadata(
        &self,
        table_name: &str,
        tenant: Option<&str>,
    ) -> Result<(CatalogTableSchema, CatalogTableStatistics)> {
        // Resolve within the tenant scope (TD-064/TD-113) so a tenant-scoped target
        // table is addressed in the tenant's namespace, not the default one.
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await?;

        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{table_name}' does not exist"));
        }

        let schema = catalog.get_table(&table_id).await?;
        let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
        Ok((schema, stats))
    }

    /// Resolve the tenant/namespace/DrPath-aware object base prefix for a warehouse
    /// bulk-append target (`data/{tenant}/{namespace}/{table}`), so an
    /// `INSERT ... SELECT` lands under the SAME object layout the materialize path
    /// publishes instead of the flat `data/{tenant}/tables/{name}` fallback the
    /// executor derives when it has no namespace context. Returns `None` when there
    /// is no tenant scope (single-tenant / embedded keeps the legacy fallback).
    ///
    /// Reuses the canonical [`resolve_materialize_prefix`] so the DrPath-vs-legacy
    /// layout selection (by `namespace_id`), the cross-tenant ownership assertion, and
    /// per-segment injection validation stay single-sourced with the materialize path.
    /// The
    /// caller sets this on the (local, non-persisted) target-schema `location`,
    /// where the executor consumes it at second priority — a materialize-set primary
    /// layout location still wins, so this never double-prefixes a published table.
    async fn resolve_warehouse_object_prefix(
        &self,
        table_name: &str,
        tenant: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(tenant_id) = tenant.filter(|t| !t.is_empty()) else {
            return Ok(None);
        };
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, Some(tenant_id))
            .await?;
        // Namespace metadata supplies the rename-stable `namespace_id` (DrPath layout)
        // and the owning `tenant_id` (cross-tenant assertion). A miss (embedded /
        // single-tenant with no catalog namespace) uses the well-known `ns_default`.
        let ns_meta = catalog.get_namespace(&table_id.namespace).await.ok();
        // The object_id keys the path in lockstep with the primary materialize so
        // bulk-appended rows land under the SAME prefix the published `location`
        // points at (a miss here would orphan them from reads). A catalog lookup
        // failure falls back to name-keying (Ok-or-None → None object_id).
        let table_object_id = catalog
            .get_table(&table_id)
            .await
            .ok()
            .and_then(|schema| schema.object_id);
        let prefix = resolve_materialize_prefix(
            tenant_id,
            ns_meta
                .as_ref()
                .and_then(|n| n.namespace_id.as_deref())
                .unwrap_or(DrPathBuilder::DEFAULT_NAMESPACE_ID),
            ns_meta.as_ref().and_then(|n| n.tenant_id.as_deref()),
            ns_meta
                .as_ref()
                .map(|n| n.storage_pool_class)
                .unwrap_or_default(),
            &table_id.name,
            table_object_id,
        )?;
        Ok(Some(prefix))
    }

    fn route_table_write_with_schemas(
        &self,
        plan: &CopyIntoPlan,
        target_columns: &[String],
        target_schema: &CatalogTableSchema,
        target_stats: Option<&CatalogTableStatistics>,
        source_schema: Option<&CatalogTableSchema>,
        source_stats: Option<&CatalogTableStatistics>,
        write_intent_overrides: Option<&WriteIntentOverrides>,
    ) -> Result<RoutedExecutionPlan> {
        DmlWritePlanner::default().plan(DmlWritePlanRequest {
            target_schema,
            target_stats,
            source_schema,
            source_stats,
            write_intent_overrides,
            plan,
            target_columns,
        })
    }

    fn route_row_dml_write_intent(
        table_schema: &CatalogTableSchema,
        operation_kind: WriteOperationKind,
        row_count: usize,
    ) -> (WriteIntent, WriteLaneDecision) {
        let mut intent = WriteIntent::new(table_schema.name.clone(), operation_kind)
            .with_durability(WriteDurabilityRequirement::WalRequired)
            .with_row_count_hint(row_count as u64);

        if operation_kind.requires_row_level_mvcc() {
            intent = intent.with_row_level_semantics(true);
        }

        let decision = WriteLaneRouter::new().route(&intent);
        (intent, decision)
    }

    fn trace_row_dml_write_lane(intent: &WriteIntent, decision: &WriteLaneDecision) {
        debug!(
            table = %intent.target_table,
            operation = ?intent.operation_kind,
            durability = ?intent.durability,
            write_lane = ?decision.lane,
            guards = ?decision.required_guards,
            rejected_lanes = ?decision.rejected_lanes,
            "Routed row-level DML write intent"
        );
    }

    /// Strip a leading table qualifier (`t.col` / alias `k.col` → `col`). On a
    /// single-table SELECT the qualifier is unambiguous, so the unqualified suffix
    /// is the column name to resolve against the schema.
    fn unqualified_column(name: &str) -> &str {
        name.rsplit('.').next().unwrap_or(name)
    }

    /// TD-OLAP-18: resolve a SQL column spelling to the table's DECLARED name —
    /// an exact match keeps its own spelling; otherwise a unique
    /// case-insensitive match resolves (ambiguous case-variants stay
    /// unresolved, mirroring the frontend's fold semantics). Anything else is
    /// the same "does not exist" error the validators raise.
    fn resolve_declared_column_name(
        raw: &str,
        table_schema: &CatalogTableSchema,
    ) -> Result<String> {
        let bare = Self::unqualified_column(raw);
        let matches: Vec<&CatalogColumn> = table_schema
            .columns
            .iter()
            .filter(|column| column.name.eq_ignore_ascii_case(bare))
            .collect();
        match matches.as_slice() {
            [declared] => Ok(declared.name.clone()),
            _ => Err(anyhow!(
                "Column '{}' does not exist in table '{}'",
                bare,
                table_schema.name
            )),
        }
    }

    fn resolve_select_predicates(
        table_schema: &CatalogTableSchema,
        predicates: &[RelationalSelectPredicateInput],
    ) -> Result<Vec<RelationalSelectPredicate>> {
        predicates
            .iter()
            .map(|predicate| {
                let bare = Self::unqualified_column(&predicate.column_name);
                let column = table_schema
                    .columns
                    .iter()
                    .find(|column| column.name.eq_ignore_ascii_case(bare))
                    .cloned()
                    .ok_or_else(|| {
                        anyhow!(
                            "Column '{}' does not exist in table '{}'",
                            predicate.column_name,
                            table_schema.name
                        )
                    })?;
                Ok(RelationalSelectPredicate {
                    column,
                    condition: predicate.condition.clone(),
                })
            })
            .collect()
    }

    fn resolve_select_projection(
        table_schema: &CatalogTableSchema,
        projection_column_names: &[String],
    ) -> Result<Vec<CatalogColumn>> {
        if projection_column_names.is_empty() {
            return Ok(table_schema.columns.clone());
        }

        projection_column_names
            .iter()
            .map(|column_name| {
                let bare = Self::unqualified_column(column_name);
                table_schema
                    .columns
                    .iter()
                    .find(|column| column.name.eq_ignore_ascii_case(bare))
                    .cloned()
                    .ok_or_else(|| {
                        anyhow!(
                            "Column '{}' does not exist in table '{}'",
                            column_name,
                            table_schema.name
                        )
                    })
            })
            .collect()
    }

    fn select_route_metadata(
        table_schema: &CatalogTableSchema,
        selected_columns: &[CatalogColumn],
        predicate_count: usize,
        limit: Option<usize>,
        access_path: RelationalSelectAccessPath,
    ) -> RelationalSelectRouteMetadata {
        RelationalSelectRouteMetadata {
            access_path,
            authority_mode: Self::select_authority_mode(table_schema),
            workload_profile: table_schema.workload_profile.as_str().to_string(),
            storage_specialization: table_schema.storage_specialization.as_str().to_string(),
            policy_boundary: Self::select_policy_boundary(table_schema),
            predicate_count,
            projected_column_count: selected_columns.len(),
            limit,
        }
    }

    fn select_authority_mode(table_schema: &CatalogTableSchema) -> String {
        Self::select_primary_layout(table_schema)
            .map(|layout| layout.authority.ownership_mode_name().to_string())
            .unwrap_or_else(|| "ProximaAuthoritative".to_string())
    }

    fn select_policy_boundary(table_schema: &CatalogTableSchema) -> String {
        if let Some(boundary) = table_schema
            .properties
            .get("policy_boundary")
            .or_else(|| table_schema.properties.get("rls_boundary"))
        {
            return boundary.clone();
        }

        match Self::select_primary_layout(table_schema) {
            Some(layout) if layout.policy_enforced_in_proxima => "engine-enforced".to_string(),
            Some(layout) if layout.authority.is_external_authoritative() => {
                "external-policy".to_string()
            }
            Some(_) => "unsupported".to_string(),
            None => "engine-enforced".to_string(),
        }
    }

    fn select_primary_layout(table_schema: &CatalogTableSchema) -> Option<&CatalogStorageLayout> {
        table_schema
            .storage_layouts
            .iter()
            .rev()
            .find(|layout| layout.name == "primary")
            .or_else(|| table_schema.storage_layouts.first())
    }

    fn primary_key_lookup_value(
        table_schema: &CatalogTableSchema,
        predicates: &[RelationalSelectPredicate],
    ) -> Option<String> {
        let primary_key = table_schema.primary_key.first()?;
        predicates.iter().find_map(|predicate| {
            if !predicate.column.name.eq_ignore_ascii_case(primary_key) {
                return None;
            }
            let RelationalSelectPredicateCondition::Comparison {
                operator: RelationalSelectPredicateOperator::Equal,
                literal,
            } = &predicate.condition
            else {
                return None;
            };
            Some(literal.clone())
        })
    }

    fn rich_result_to_record(result: RichSearchResult) -> ProximaRecord {
        let embeddings = if result.vector.is_empty() {
            Vec::new()
        } else {
            vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                result.vector.len() as u32,
                result.vector,
            )]
        };

        ProximaRecord {
            oid: result.id.clone(),
            local_id: Some(result.id),
            props: result
                .props
                .into_iter()
                .map(|(key, value)| (key, ProximaTreeNode::Value(value)))
                .collect(),
            embeddings,
            record_version: result.version.map(u64::from).unwrap_or_default(),
            created_at_ns: result
                .timestamp
                .map(|timestamp_ms| timestamp_ms.saturating_mul(1_000_000))
                .unwrap_or_default(),
            origin: result.source,
            ..ProximaRecord::default()
        }
    }

    fn project_select_rows(
        records: &[ProximaRecord],
        table_schema: &CatalogTableSchema,
        selected_columns: &[CatalogColumn],
    ) -> Result<Vec<Vec<ProximaValue>>> {
        records
            .iter()
            .map(|record| Self::project_one_record(record, table_schema, selected_columns))
            .collect()
    }

    /// Project a single record into `selected_columns` order. Extracted so the
    /// relational-pipeline scan can project per-record inside a scan-filter
    /// closure (predicate evaluation against a full row) without batching.
    fn project_one_record(
        record: &ProximaRecord,
        table_schema: &CatalogTableSchema,
        selected_columns: &[CatalogColumn],
    ) -> Result<Vec<ProximaValue>> {
        let primary_key = table_schema.primary_key.first().map(String::as_str);
        selected_columns
            .iter()
            .map(|column| {
                Self::record_column_value_for_select(record, column, primary_key, table_schema)
            })
            .collect()
    }

    fn record_column_value_for_select(
        record: &ProximaRecord,
        column: &CatalogColumn,
        primary_key: Option<&str>,
        table_schema: &CatalogTableSchema,
    ) -> Result<ProximaValue> {
        if let Some(proximadb_records::ProximaTreeNode::Value(value)) =
            record.props.get(&column.name)
        {
            return Ok(value.clone());
        }
        if primary_key.is_some_and(|primary_key| column.name.eq_ignore_ascii_case(primary_key)) {
            return Self::primary_key_string_to_proxima_value(
                &column.name,
                &record.oid,
                table_schema,
            );
        }
        if matches!(
            column.data_type,
            ProximaType::DenseVector { .. }
                | ProximaType::SparseVector { .. }
                | ProximaType::BinaryVector { .. }
        ) && let Some(embedding) = record.embeddings.first()
        {
            return Ok(ProximaValue::DenseVector(embedding.values.to_fp32_owned()));
        }
        Ok(ProximaValue::Null)
    }

    /// Evaluate simple catalog-shaped predicates against a canonical record.
    pub fn record_matches_select_predicates(
        record: &ProximaRecord,
        predicates: &[RelationalSelectPredicate],
        primary_key: Option<&str>,
    ) -> bool {
        predicates
            .iter()
            .all(|predicate| Self::eval_predicate_leaf(record, predicate, primary_key))
    }

    /// Evaluate a single catalog-resolved leaf predicate against a record
    /// (catalog-type-aware comparison, PK resolved via `record.oid`). Shared by
    /// the flat SELECT matcher and the UPDATE/DELETE predicate tree.
    fn eval_predicate_leaf(
        record: &ProximaRecord,
        predicate: &RelationalSelectPredicate,
        primary_key: Option<&str>,
    ) -> bool {
        let value = Self::record_column_value_for_predicate(record, &predicate.column, primary_key);
        match &predicate.condition {
            RelationalSelectPredicateCondition::Comparison { operator, literal } => {
                Self::compare_catalog_value(&value, literal, *operator, &predicate.column.data_type)
            }
            RelationalSelectPredicateCondition::In { literals, negated } => {
                let matches = literals.iter().any(|literal| {
                    Self::compare_catalog_value(
                        &value,
                        literal,
                        RelationalSelectPredicateOperator::Equal,
                        &predicate.column.data_type,
                    )
                });
                if *negated { !matches } else { matches }
            }
            RelationalSelectPredicateCondition::Like { pattern, negated } => {
                let matches = Self::sql_like_matches(&value, pattern);
                if *negated { !matches } else { matches }
            }
            RelationalSelectPredicateCondition::IsNull { negated } => {
                let matches = value.is_empty();
                if *negated { !matches } else { matches }
            }
        }
    }

    /// Recursively evaluate a boolean predicate tree against a record.
    fn eval_predicate_tree(
        record: &ProximaRecord,
        tree: &RelationalPredicateTree,
        primary_key: Option<&str>,
    ) -> bool {
        match tree {
            RelationalPredicateTree::Leaf(predicate) => {
                Self::eval_predicate_leaf(record, predicate, primary_key)
            }
            RelationalPredicateTree::And(children) => children
                .iter()
                .all(|child| Self::eval_predicate_tree(record, child, primary_key)),
            RelationalPredicateTree::Or(children) => children
                .iter()
                .any(|child| Self::eval_predicate_tree(record, child, primary_key)),
            RelationalPredicateTree::Not(child) => {
                !Self::eval_predicate_tree(record, child, primary_key)
            }
        }
    }

    /// Resolve and evaluate syntax-level predicate inputs for tests and simple
    /// read adapters that already have a catalog schema.
    pub fn record_matches_select_predicate_inputs(
        record: &ProximaRecord,
        table_schema: &CatalogTableSchema,
        predicates: &[RelationalSelectPredicateInput],
    ) -> Result<bool> {
        let predicates = Self::resolve_select_predicates(table_schema, predicates)?;
        Ok(Self::record_matches_select_predicates(
            record,
            &predicates,
            table_schema.primary_key.first().map(String::as_str),
        ))
    }

    fn record_column_value_for_predicate(
        record: &ProximaRecord,
        column: &proximadb_catalog::CatalogColumn,
        primary_key: Option<&str>,
    ) -> String {
        if let Some(proximadb_records::ProximaTreeNode::Value(value)) =
            record.props.get(&column.name)
        {
            return Self::proxima_value_to_predicate_text(value);
        }
        if primary_key.is_some_and(|primary_key| column.name.eq_ignore_ascii_case(primary_key)) {
            return record.oid.clone();
        }
        // Vector-typed columns are not meaningfully comparable via SQL scalar
        // predicates; emitting a CSV stringification per row (potentially KB-sized)
        // is wasted work and the comparison can never succeed semantically. Return
        // empty so IsNull behaves correctly and all ordered/equality predicates
        // evaluate to false.
        String::new()
    }

    fn proxima_value_to_predicate_text(value: &ProximaValue) -> String {
        // Single source of truth shared with the record store (TD-110 Slice C),
        // so the value rendered for a UNIQUE/PK index/probe matches the value
        // rendered for predicate evaluation.
        proxima_value_to_unique_text(value)
    }

    fn compare_catalog_value(
        value: &str,
        literal: &str,
        operator: RelationalSelectPredicateOperator,
        data_type: &ProximaType,
    ) -> bool {
        match data_type {
            ProximaType::Boolean => {
                let left = Self::normalize_bool_literal(value);
                let right = Self::normalize_bool_literal(literal);
                Self::compare_ordered_values(left, right, operator)
            }
            ProximaType::Int8
            | ProximaType::Int16
            | ProximaType::Int32
            | ProximaType::Int64
            | ProximaType::Float32
            | ProximaType::Float64
            | ProximaType::Decimal { .. } => {
                let Ok(left) = value.parse::<f64>() else {
                    return false;
                };
                let Ok(right) = literal.parse::<f64>() else {
                    return false;
                };
                match operator {
                    RelationalSelectPredicateOperator::Equal => (left - right).abs() < f64::EPSILON,
                    RelationalSelectPredicateOperator::NotEqual => {
                        (left - right).abs() >= f64::EPSILON
                    }
                    RelationalSelectPredicateOperator::LessThan => left < right,
                    RelationalSelectPredicateOperator::LessThanOrEqual => left <= right,
                    RelationalSelectPredicateOperator::GreaterThan => left > right,
                    RelationalSelectPredicateOperator::GreaterThanOrEqual => left >= right,
                }
            }
            _ => Self::compare_ordered_values(value.to_string(), literal.to_string(), operator),
        }
    }

    fn normalize_bool_literal(value: &str) -> String {
        match value.trim().to_ascii_lowercase().as_str() {
            "t" | "true" | "1" | "yes" | "on" => "t".to_string(),
            "f" | "false" | "0" | "no" | "off" => "f".to_string(),
            other => other.to_string(),
        }
    }

    fn compare_ordered_values(
        left: String,
        right: String,
        operator: RelationalSelectPredicateOperator,
    ) -> bool {
        match operator {
            RelationalSelectPredicateOperator::Equal => left == right,
            RelationalSelectPredicateOperator::NotEqual => left != right,
            RelationalSelectPredicateOperator::LessThan => left < right,
            RelationalSelectPredicateOperator::LessThanOrEqual => left <= right,
            RelationalSelectPredicateOperator::GreaterThan => left > right,
            RelationalSelectPredicateOperator::GreaterThanOrEqual => left >= right,
        }
    }

    fn sql_like_matches(value: &str, pattern: &str) -> bool {
        // Fast paths for common patterns. Only applied for pure-ASCII patterns
        // where byte-indexed slicing is equivalent to char-indexed slicing; this
        // preserves the char-based semantics of the DP path below for multi-byte
        // patterns. `_` is not handled here because it requires per-position
        // wildcard checking.
        if pattern.is_ascii() && !pattern.contains('_') {
            let bytes = pattern.as_bytes();
            let pct_count = bytes.iter().filter(|&&b| b == b'%').count();
            match pct_count {
                0 => return value == pattern,
                1 => {
                    if bytes.first() == Some(&b'%') {
                        return value.ends_with(&pattern[1..]);
                    }
                    if bytes.last() == Some(&b'%') {
                        return value.starts_with(&pattern[..pattern.len() - 1]);
                    }
                    // Single `%` in the middle: fall through to DP.
                }
                2 if bytes.first() == Some(&b'%') && bytes.last() == Some(&b'%') => {
                    return value.contains(&pattern[1..pattern.len() - 1]);
                }
                _ => {}
            }
        }

        Self::sql_like_matches_dp(value, pattern)
    }

    fn sql_like_matches_dp(value: &str, pattern: &str) -> bool {
        let value: Vec<char> = value.chars().collect();
        let pattern: Vec<char> = pattern.chars().collect();
        let plen = pattern.len();
        let vlen = value.len();

        // Rolling-row DP: O(plen) extra memory instead of O(vlen * plen).
        let mut prev = vec![false; plen + 1];
        let mut cur = vec![false; plen + 1];
        prev[0] = true;
        for p in 1..=plen {
            if pattern[p - 1] == '%' {
                prev[p] = prev[p - 1];
            }
        }

        for v in 1..=vlen {
            cur[0] = false;
            for p in 1..=plen {
                cur[p] = match pattern[p - 1] {
                    '%' => cur[p - 1] || prev[p],
                    '_' => prev[p - 1],
                    ch => ch == value[v - 1] && prev[p - 1],
                };
            }
            std::mem::swap(&mut prev, &mut cur);
        }

        prev[plen]
    }

    async fn resolve_table_write_source_metadata(
        &self,
        plan: &CopyIntoPlan,
        tenant: Option<&str>,
    ) -> Result<Option<(CatalogTableSchema, CatalogTableStatistics)>> {
        let ReadSource::CatalogTable { table, .. } = &plan.source else {
            return Ok(None);
        };

        let source_table_name = table.qualified_name();
        // Resolve the SELECT source within the tenant scope (TD-064/TD-113) so a
        // tenant reads its own source table, not the default namespace's.
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(&source_table_name, tenant)
            .await?;
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!(
                "Source table '{}' does not exist for table write into '{}'",
                source_table_name,
                plan.target.qualified_name()
            ));
        }

        let schema = catalog.get_table(&table_id).await?;
        let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
        Ok(Some((schema, stats)))
    }

    /// Execute INSERT statement
    async fn execute_insert(
        &self,
        table_name: &str,
        columns: &[String],
        values: Vec<Vec<SqlValueLiteral>>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;

        // Verify table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{table_name}' does not exist"));
        }

        // Get table schema for column mapping
        let table_schema = catalog.get_table(&table_id).await?;
        let dml_lock_guard = self
            .acquire_table_dml_lock(&table_id, tenant_context, LockIntent::Write)
            .await?;
        // TD-110 S1: non-reentrant in-process operation lock for this table,
        // held for the whole INSERT. Pairs with the same lock a cascading
        // parent DELETE takes on its child set, so a child INSERT concurrent
        // with a parent DELETE serializes (blocks) rather than orphaning.
        let _op_lock = self.op_lock(&table_id).await?;
        let (write_intent, write_lane_decision) = Self::route_row_dml_write_intent(
            &table_schema,
            WriteOperationKind::Insert,
            values.len(),
        );
        Self::trace_row_dml_write_lane(&write_intent, &write_lane_decision);

        // Compute per-column null counts before consuming values (for T8 column stats).
        let null_counts_per_column: Vec<u64> = (0..columns.len())
            .map(|idx| {
                values
                    .iter()
                    .filter(|row| {
                        row.get(idx)
                            .is_none_or(|v| matches!(v, SqlValueLiteral::Null))
                    })
                    .count() as u64
            })
            .collect();

        // Convert SQL literals into canonical ProximaRecord envelopes.
        let mut records = Vec::new();
        let mut inserted_ids = Vec::new();

        for row in values {
            let record = self.build_mutation_record(
                columns,
                &row,
                &table_schema,
                RelationalMutationKind::Insert,
            )?;
            inserted_ids.push(record.oid.clone());
            records.push(record);
        }

        // TD-110 / TD-DML-1: PRIMARY KEY (fenced CKS claim or point probe),
        // secondary UNIQUE and FOREIGN KEY — the ONE enforcement sequence every
        // row-producing DML shares (UPSERT runs the same one). CHECK / NOT NULL /
        // types were already enforced row-locally by `CatalogRow::validate`.
        self.enforce_row_write_constraints(
            &table_schema,
            &table_id,
            &records,
            RowWriteConstraintMode::Insert,
            tenant_context,
        )
        .await?;

        // Compute per-column min/max and NDV from the canonical record props before moving records
        // into mutations. Only orderable types (String, integers) tracked for min/max; floats and
        // booleans are additionally included in the NDV pass.
        let column_minmax = Self::compute_column_minmax_from_records(&records);
        let column_ndv = Self::compute_column_ndv_from_records(&records);

        // Insert canonical records through the table-record boundary. The current
        // implementation may adapt to legacy vector paths behind that trait.
        let num_records = records.len();
        let mutations = records
            .into_iter()
            .map(|record| TableRecordMutation::new(TableRecordMutationKind::Insert, record))
            .collect::<Vec<_>>();
        let batch_result = self
            .record_store
            .write_mutations(&table_schema, mutations, tenant_context)
            .await?;
        if !batch_result.success {
            let detail = batch_result
                .errors
                .first()
                .cloned()
                .unwrap_or_else(|| "unknown error".to_string());
            // A store-level key conflict is a duplicate primary key: keep it typed
            // so every surface reports 23505, even if a pre-check missed it.
            if batch_result.error_code.as_deref() == Some("INSERT_CONFLICT") {
                return Err(RelationalConstraintViolation::new(
                    RelationalConstraintKind::Unique,
                    format!("duplicate key value violates primary key: {detail}"),
                )
                .into());
            }
            return Err(anyhow!("Insert failed: {detail}"));
        }

        info!(
            table = %table_name,
            rows = num_records,
            "Inserted rows"
        );

        let stats_tenant = tenant_context.map(|t| t.tenant_id.as_str());
        self.bump_row_count_stats(table_name, stats_tenant, num_records as i64)
            .await;
        self.bump_column_null_counts(table_name, stats_tenant, columns, &null_counts_per_column)
            .await;
        self.bump_column_minmax(table_name, stats_tenant, column_minmax)
            .await;
        self.bump_column_ndv(table_name, stats_tenant, column_ndv)
            .await;

        let result =
            DmlResult::success(num_records as u64, format!("Inserted {} rows", num_records))
                .with_inserted_ids(inserted_ids);
        self.finish_dml_guards(dml_lock_guard).await;
        Ok(result)
    }

    /// Execute UPDATE statement
    ///
    /// Note: UPDATE operations require full table scan with WHERE clause evaluation.
    /// For vector databases, updates are typically done by delete + insert.
    async fn execute_update(
        &self,
        table_name: &str,
        assignments: Vec<(String, SqlValueLiteral)>,
        where_clause: Option<WhereClause>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;

        // Verify table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{table_name}' does not exist"));
        }

        let table_schema = catalog.get_table(&table_id).await?;
        let dml_lock_guard = self
            .acquire_table_dml_lock(&table_id, tenant_context, LockIntent::Write)
            .await?;
        // TD-110 S1: non-reentrant in-process operation lock (see execute_insert).
        let _op_lock = self.op_lock(&table_id).await?;
        let ids_to_update = if let Some(ref wc) = where_clause {
            self.resolve_matching_ids(&table_schema, &table_id.name, wc, tenant_context)
                .await?
        } else {
            return Err(anyhow!("UPDATE without WHERE clause is not allowed"));
        };
        if ids_to_update.is_empty() {
            return Ok(DmlResult::success(0, "No rows matched WHERE clause"));
        }
        let (write_intent, write_lane_decision) = Self::route_row_dml_write_intent(
            &table_schema,
            WriteOperationKind::Update,
            ids_to_update.len(),
        );
        Self::trace_row_dml_write_lane(&write_intent, &write_lane_decision);
        Self::validate_update_assignments(&assignments, &table_schema)?;

        let mut records = Vec::new();
        let mut warnings = Vec::new();
        for record_id in &ids_to_update {
            let Some(existing) = self
                .record_store
                .get_by_key(
                    &table_schema,
                    TableRecordGetRequest {
                        table_id: table_id.name.clone(),
                        key: record_id.clone(),
                        include_vector: true,
                        include_props: true,
                    },
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
                )
                .await?
            else {
                warnings.push(format!(
                    "Record '{}' was not found in table '{}'",
                    record_id, table_schema.name
                ));
                continue;
            };

            records.push(self.build_updated_proxima_record(
                existing,
                &assignments,
                &table_schema,
            )?);
        }

        if records.is_empty() {
            return Ok(DmlResult::success(0, "No rows matched WHERE clause"));
        }

        // TD-110: enforce non-PK UNIQUE constraints on UPDATE. The rows' NEW
        // values must not collide with OTHER rows (or each other). The updated
        // rows are excluded so a row keeping or vacating its own unique value is
        // not flagged as a self-conflict. (PK is immutable on UPDATE — rejected by
        // validate_update_assignments — so only non-PK UNIQUE sets apply here.)
        let primary_key = Self::primary_key_column(&table_schema);
        let candidate_sets =
            Self::build_unique_candidate_sets(&table_schema, &records, primary_key.as_deref())?;
        if !candidate_sets.is_empty() {
            let exclude_oids: std::collections::HashSet<String> =
                records.iter().map(|record| record.oid.clone()).collect();
            if let Some(conflict) = self
                .record_store
                .check_unique_conflict(
                    &table_schema,
                    &table_id.name,
                    primary_key.as_deref(),
                    &candidate_sets,
                    &exclude_oids,
                    tenant_context,
                )
                .await?
            {
                return Err(Self::unique_violation(&table_schema, &conflict));
            }
        }

        // TD-110: an UPDATE may change a FOREIGN KEY column — re-verify references.
        self.enforce_foreign_keys(&table_schema, &records, tenant_context)
            .await?;

        let updated_count = records.len();
        let mutations = records
            .into_iter()
            .map(|record| TableRecordMutation::new(TableRecordMutationKind::Update, record))
            .collect::<Vec<_>>();
        let batch_result = self
            .record_store
            .write_mutations(&table_schema, mutations, tenant_context)
            .await?;
        if !batch_result.success {
            return Err(anyhow!(
                "Update failed: {}",
                batch_result
                    .errors
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        info!(
            table = %table_name,
            rows = updated_count,
            "Updated rows"
        );

        let mut result = DmlResult::success(
            updated_count as u64,
            format!("Updated {} rows", updated_count),
        );
        result.warnings = warnings;
        self.finish_dml_guards(dml_lock_guard).await;
        Ok(result)
    }

    /// Execute DELETE statement
    ///
    /// Note: DELETE by ID is the primary supported operation.
    async fn execute_delete(
        &self,
        table_name: &str,
        where_clause: Option<WhereClause>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        self.execute_delete_rows(table_name, where_clause, false, tenant_context)
            .await
    }

    /// `TRUNCATE`: delete every row of `table_name`, resolved within
    /// `tenant_context`, through the same path (locks, referential actions,
    /// tombstones, statistics) as a DELETE.
    pub async fn truncate_table(
        &self,
        table_name: &str,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        self.execute_delete_rows(table_name, None, true, tenant_context)
            .await
    }

    /// DELETE rows matching `where_clause`, or — only when `all_rows` is set
    /// (TRUNCATE) — every row. A DELETE with neither is refused.
    async fn execute_delete_rows(
        &self,
        table_name: &str,
        where_clause: Option<WhereClause>,
        all_rows: bool,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;

        // Verify table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{table_name}' does not exist"));
        }

        let table_schema = catalog.get_table(&table_id).await?;
        let dml_lock_guard = self
            .acquire_table_dml_lock(&table_id, tenant_context, LockIntent::Write)
            .await?;

        // Get IDs to delete based on WHERE clause
        let ids_to_delete = if let Some(ref wc) = where_clause {
            self.resolve_matching_ids(&table_schema, &table_id.name, wc, tenant_context)
                .await?
        } else if all_rows {
            self.record_store
                .scan_records_filtered(
                    &table_schema,
                    TableRecordScanRequest {
                        filter: None,
                        table_id: table_id.name.clone(),
                        limit: None,
                        include_vector: false,
                        include_props: false,
                    },
                    None,
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "dml truncate"),
                )
                .await?
                .into_iter()
                .map(|record| record.oid)
                .collect()
        } else {
            return Err(anyhow!(
                "DELETE without WHERE clause is not allowed. Use WHERE primary key IN (...) to delete specific rows."
            ));
        };

        if ids_to_delete.is_empty() {
            return Ok(DmlResult::success(0, "No rows matched WHERE clause"));
        }

        // TD-110 S1: lock the transitive child set for the FULL critical section
        // (cascade scan → child mutations → parent tombstone) so a concurrent
        // writer cannot slip a child row in mid-flight and orphan it. Two layers:
        //   • in-process op-locks (`TableOpLockRegistry`, non-reentrant, blocking):
        //     serialize concurrent CONNECTIONS on a single (embedded) pod. The
        //     durable DML lock below is pod-level + re-entrant, so without this an
        //     intra-pod child INSERT during the parent DELETE could orphan.
        //   • durable DML write locks (cross-pod, non-blocking): serialize the same
        //     section across pods; a conflict surfaces as a typed `DmlLockConflict`
        //     (pgwire 55P03 / gRPC ABORTED) for client retry.
        // Acquired in deterministic (namespace, name) order (op-locks block, so the
        // global order avoids self-deadlock). Guards are bound to the end of this
        // function — released only after the parent tombstone is written.
        let child_closure = self
            .discover_cascade_child_set(&catalog, &table_id, &table_schema, tenant_context)
            .await?;
        let mut op_lock_tables: Vec<TableIdentifier> = child_closure
            .iter()
            .map(|(namespace, name)| TableIdentifier {
                namespace: namespace.clone(),
                name: name.clone(),
            })
            .collect();
        op_lock_tables.push(table_id.clone()); // parent is in the critical section too
        let _cascade_op_locks = self.op_locks_sorted(op_lock_tables).await?;
        let cascade_child_dml_locks = self
            .acquire_cascade_child_dml_locks(&child_closure, tenant_context)
            .await?;

        // TD-110: apply ON DELETE referential actions (RESTRICT/CASCADE/SET NULL)
        // BEFORE removing the parent rows — RESTRICT may abort the DELETE, and
        // CASCADE/SET NULL mutate child tables in the same tenant scope.
        self.enforce_delete_referential_actions(
            &catalog,
            &table_id,
            &table_schema,
            &ids_to_delete,
            tenant_context,
        )
        .await?;

        let (write_intent, write_lane_decision) = Self::route_row_dml_write_intent(
            &table_schema,
            WriteOperationKind::Delete,
            ids_to_delete.len(),
        );
        Self::trace_row_dml_write_lane(&write_intent, &write_lane_decision);

        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let tombstones = ids_to_delete
            .iter()
            .map(|id| Self::build_delete_tombstone_record(id, &table_schema, now_ns))
            .collect::<Result<Vec<_>>>()?;
        // F5: capture the fenced-store PK identities from these tombstone records
        // (same insert-path builder → matching tenant_id + typed PK), applied
        // after the record store confirms the delete. Full PK tuple (TD-DML-2).
        let cks_tombstone_ids: Vec<proximadb_storage_ports::Identity> =
            if self.conditional_key_store.is_some()
                && Self::table_declares_primary_key(&table_schema)
            {
                // TD-DML-2: composite PKs are fenced on their full tuple, so
                // their deletes must release the full-tuple identity too.
                let pk_cols = Self::primary_key_columns(&table_schema);
                tombstones
                    .iter()
                    .filter_map(|rec| {
                        let pk_values = Self::record_primary_key_values(rec, &pk_cols)?;
                        proximadb_data_model::key_codec::encode_identity(
                            &rec.tenant_id,
                            &table_schema.name,
                            &pk_values,
                        )
                        .ok()
                        .map(proximadb_storage_ports::Identity::from_bytes)
                    })
                    .collect()
            } else {
                Vec::new()
            };
        let deleted_count = ids_to_delete.len();
        let mutations = tombstones
            .into_iter()
            .map(|record| TableRecordMutation::new(TableRecordMutationKind::Delete, record))
            .collect::<Vec<_>>();
        let batch_result = self
            .record_store
            .write_mutations(&table_schema, mutations, tenant_context)
            .await?;
        if !batch_result.success {
            return Err(anyhow!(
                "Delete failed: {}",
                batch_result
                    .errors
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        // F5 (ADR-072): keep the fenced ConditionalKeyStore consistent with the
        // delete — tombstone each removed row's PK Identity so FK existence
        // (`cks.get`, the insert-path slice) and PK re-insert (`put_if_absent`)
        // reflect the deletion. Without it a deleted parent stays "live": a child
        // FK to it would wrongly succeed, and re-inserting the same PK would
        // wrongly conflict. The identities are derived from the SAME tombstone
        // records above (which go through the insert-path record builder), so the
        // `tenant_id` + typed PK tuple match the CKS registration exactly
        // (single- and multi-column PKs alike — TD-DML-2).
        if let Some(cks) = &self.conditional_key_store {
            for id in &cks_tombstone_ids {
                cks.tombstone(id, proximadb_storage_ports::Generation(0))
                    .await?;
            }
        }

        info!(
            table = %table_name,
            rows = deleted_count,
            "Deleted rows"
        );

        self.bump_row_count_stats(
            table_name,
            tenant_context.map(|t| t.tenant_id.as_str()),
            -(deleted_count as i64),
        )
        .await;

        let result = DmlResult::success(
            deleted_count as u64,
            format!("Deleted {} rows", deleted_count),
        );
        self.finish_dml_guards(dml_lock_guard.into_iter().chain(cascade_child_dml_locks))
            .await;
        Ok(result)
    }

    /// Execute UPSERT statement
    async fn execute_upsert(
        &self,
        table_name: &str,
        columns: &[String],
        values: Vec<Vec<SqlValueLiteral>>,
        _conflict_columns: &[String],
        _update_assignments: Vec<(String, SqlValueLiteral)>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<DmlResult> {
        let (catalog, table_id) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant_context.map(|t| t.tenant_id.as_str()))
            .await?;

        // Verify table exists
        if !catalog.table_exists(&table_id).await? {
            return Err(anyhow!("Table '{table_name}' does not exist"));
        }

        // Get table schema
        let table_schema = catalog.get_table(&table_id).await?;
        let dml_lock_guard = self
            .acquire_table_dml_lock(&table_id, tenant_context, LockIntent::Write)
            .await?;
        // TD-DML-1: the same non-reentrant in-process op-lock INSERT holds (see
        // execute_insert) — the constraint probes below and the write must not
        // interleave with a concurrent writer or a cascading parent DELETE.
        let _op_lock = self.op_lock(&table_id).await?;
        let (write_intent, write_lane_decision) = Self::route_row_dml_write_intent(
            &table_schema,
            WriteOperationKind::Upsert,
            values.len(),
        );
        Self::trace_row_dml_write_lane(&write_intent, &write_lane_decision);

        let mut records = Vec::new();
        let mut inserted_ids = Vec::new();

        for row in values {
            let record = self.build_mutation_record(
                columns,
                &row,
                &table_schema,
                RelationalMutationKind::Upsert,
            )?;
            inserted_ids.push(record.oid.clone());
            records.push(record);
        }

        // TD-DML-1: an UPSERT writes rows exactly like INSERT does, so it runs
        // the SAME integrity sequence (PK claim, secondary UNIQUE, FOREIGN KEY);
        // CHECK / NOT NULL ran row-locally in `build_mutation_record`. The
        // Upsert mode differs only where the semantics do: a live PK holder is
        // the row being replaced (not a duplicate), and replaced rows vacate
        // their own UNIQUE values (no self-conflict).
        let replaced = self
            .enforce_row_write_constraints(
                &table_schema,
                &table_id,
                &records,
                RowWriteConstraintMode::Upsert,
                tenant_context,
            )
            .await?;
        let new_rows = replaced.iter().filter(|existed| !**existed).count();

        let num_records = records.len();
        let mutations = records
            .into_iter()
            .map(|record| TableRecordMutation::new(TableRecordMutationKind::Upsert, record))
            .collect::<Vec<_>>();
        let batch_result = self
            .record_store
            .write_mutations(&table_schema, mutations, tenant_context)
            .await?;
        if !batch_result.success {
            return Err(anyhow!(
                "Upsert failed: {}",
                batch_result
                    .errors
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        info!(
            table = %table_name,
            rows = num_records,
            "Upserted rows"
        );

        // Only rows that did not exist grow the table; replaced rows do not.
        self.bump_row_count_stats(
            table_name,
            tenant_context.map(|t| t.tenant_id.as_str()),
            new_rows as i64,
        )
        .await;

        let result =
            DmlResult::success(num_records as u64, format!("Upserted {} rows", num_records))
                .with_inserted_ids(inserted_ids);
        self.finish_dml_guards(dml_lock_guard).await;
        Ok(result)
    }

    // ========================
    // Helper Methods
    // ========================

    /// Current wall-clock time in milliseconds since the Unix epoch, used to mark
    /// `CatalogTableStatistics.last_analyzed_ms` after a stats update so the planner
    /// can detect stale statistics via `CatalogTableStatistics::is_stale`.
    fn now_unix_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    /// Increment or decrement the row count in catalog statistics after a successful write.
    /// Errors are non-fatal: they are logged as warnings and do not fail the operation.
    /// Called by DML paths and by fast-lane (gRPC/REST/Arrow Flight) write paths for
    /// collections that are registered as relational tables in xCatalog.
    pub(crate) async fn bump_row_count_stats(
        &self,
        table_name: &str,
        tenant: Option<&str>,
        delta: i64,
    ) {
        if self.defer_stat(|| DeferredStatBump::RowCount {
            table: table_name.to_string(),
            tenant: tenant.map(str::to_string),
            delta,
        }) {
            return;
        }
        let Ok((catalog, table_id)) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await
        else {
            return;
        };
        let mut stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
        stats.row_count = if delta >= 0 {
            stats.row_count.saturating_add(delta as u64)
        } else {
            stats.row_count.saturating_sub(delta.unsigned_abs())
        };
        stats.last_analyzed_ms = Some(Self::now_unix_ms());
        if let Err(e) = catalog.update_statistics(&table_id, stats).await {
            warn!(table = %table_name, error = %e, "Failed to update row-count statistics after DML");
        } else {
            Self::bump_corpus_version_for_stats(&table_id).await;
        }
    }

    /// Bump the process-wide corpus_version after a successful stats
    /// refresh so cached planner outputs invalidate against the new
    /// selectivity numbers. Extracts tenant from `namespace[0]` per
    /// the catalog convention; skips when the table isn't
    /// tenant-scoped.
    async fn bump_corpus_version_for_stats(table_id: &crate::catalog::TableIdentifier) {
        let Some(tenant_id) = table_id.namespace.first() else {
            return;
        };
        let version = crate::catalog::CorpusVersionRegistry::global()
            .bump(tenant_id, &table_id.name)
            .await;
        tracing::debug!(
            table = %table_id.name,
            tenant = %tenant_id,
            version,
            "🔄 corpus_version bumped after stats refresh"
        );
    }

    /// Increment per-column null counts in catalog statistics after a successful INSERT.
    /// `null_counts[i]` is the number of NULL values for `columns[i]` in the inserted batch.
    /// Errors are non-fatal. Called only from the INSERT path (null counts grow monotonically).
    async fn bump_column_null_counts(
        &self,
        table_name: &str,
        tenant: Option<&str>,
        columns: &[String],
        null_counts: &[u64],
    ) {
        if columns.is_empty() || null_counts.iter().all(|&c| c == 0) {
            return;
        }
        if self.defer_stat(|| DeferredStatBump::NullCounts {
            table: table_name.to_string(),
            tenant: tenant.map(str::to_string),
            columns: columns.to_vec(),
            null_counts: null_counts.to_vec(),
        }) {
            return;
        }
        let Ok((catalog, table_id)) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await
        else {
            return;
        };
        let mut stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
        for (col_name, &null_delta) in columns.iter().zip(null_counts.iter()) {
            if null_delta > 0 {
                let col_stats = stats.column_stats.entry(col_name.clone()).or_default();
                col_stats.null_count =
                    Some(col_stats.null_count.unwrap_or(0).saturating_add(null_delta));
            }
        }
        stats.last_analyzed_ms = Some(Self::now_unix_ms());
        if let Err(e) = catalog.update_statistics(&table_id, stats).await {
            warn!(table = %table_name, error = %e, "Failed to update column-null statistics after INSERT");
        } else {
            Self::bump_corpus_version_for_stats(&table_id).await;
        }
    }

    /// Convert a `ProximaValue` to a lexicographically sortable string for min/max tracking.
    /// Returns `None` for types without natural total order (Null, Float, Binary, Vector, etc.).
    fn value_to_minmax_string(value: &ProximaValue) -> Option<String> {
        match value {
            // Integers: zero-padded with explicit sign for correct lex order.
            ProximaValue::Int8(v) => Some(format!("{:+020}", *v as i64)),
            ProximaValue::Int16(v) => Some(format!("{:+020}", *v as i64)),
            ProximaValue::Int32(v) => Some(format!("{:+020}", *v as i64)),
            ProximaValue::Int64(v) => Some(format!("{:+020}", v)),
            // Strings: use directly (lexicographic order is natural for text columns).
            ProximaValue::String(s) => Some(s.clone()),
            // Dates/timestamps stored in ISO 8601 are naturally sortable.
            ProximaValue::Date(d) => Some(format!("{d}")),
            // Floats, Null, Binary, Vector, Struct, etc.: skip.
            _ => None,
        }
    }

    /// Compute per-column min/max sortable strings from a record batch before the records
    /// are consumed. Returns a map of column → (min_string, max_string).
    fn compute_column_minmax_from_records(
        records: &[ProximaRecord],
    ) -> HashMap<String, (String, String)> {
        let mut minmax: HashMap<String, (String, String)> = HashMap::new();
        for record in records {
            for (col, node) in record.props.iter() {
                let ProximaTreeNode::Value(val) = node else {
                    continue;
                };
                let Some(s) = Self::value_to_minmax_string(val) else {
                    continue;
                };
                match minmax.get_mut(col) {
                    Some((min, max)) => {
                        if s < *min {
                            *min = s.clone();
                        }
                        if s > *max {
                            *max = s;
                        }
                    }
                    None => {
                        minmax.insert(col.clone(), (s.clone(), s));
                    }
                }
            }
        }
        minmax
    }

    /// Merge per-column batch min/max into `CatalogTableStatistics.column_stats[col].min_value`
    /// / `max_value`. Errors are non-fatal (logged as warnings).
    async fn bump_column_minmax(
        &self,
        table_name: &str,
        tenant: Option<&str>,
        column_minmax: HashMap<String, (String, String)>,
    ) {
        if column_minmax.is_empty() {
            return;
        }
        let column_minmax = match &self.deferred_stats {
            Some(queue) => {
                queue.lock().push(DeferredStatBump::MinMax {
                    table: table_name.to_string(),
                    tenant: tenant.map(str::to_string),
                    column_minmax,
                });
                return;
            }
            None => column_minmax,
        };
        let Ok((catalog, table_id)) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await
        else {
            return;
        };
        let mut stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
        for (col, (batch_min, batch_max)) in column_minmax {
            let col_stats = stats.column_stats.entry(col).or_default();
            col_stats.min_value = Some(match col_stats.min_value.take() {
                Some(existing) if existing <= batch_min => existing,
                _ => batch_min,
            });
            col_stats.max_value = Some(match col_stats.max_value.take() {
                Some(existing) if existing >= batch_max => existing,
                _ => batch_max,
            });
        }
        stats.last_analyzed_ms = Some(Self::now_unix_ms());
        if let Err(e) = catalog.update_statistics(&table_id, stats).await {
            warn!(table = %table_name, error = %e, "Failed to update column min/max statistics after INSERT");
        } else {
            Self::bump_corpus_version_for_stats(&table_id).await;
        }
    }

    /// Convert a `ProximaValue` to a string key suitable for distinct-value counting.
    /// Covers a wider set than `value_to_minmax_string` (includes Float, Bool).
    fn value_to_ndv_key(value: &ProximaValue) -> Option<String> {
        match value {
            ProximaValue::Int8(v) => Some(format!("{v}")),
            ProximaValue::Int16(v) => Some(format!("{v}")),
            ProximaValue::Int32(v) => Some(format!("{v}")),
            ProximaValue::Int64(v) => Some(format!("{v}")),
            ProximaValue::Float32(v) => Some(format!("{v}")),
            ProximaValue::Float64(v) => Some(format!("{v}")),
            ProximaValue::String(s) => Some(s.clone()),
            ProximaValue::Boolean(b) => Some(if *b { "1" } else { "0" }.to_string()),
            ProximaValue::Date(d) => Some(format!("{d}")),
            _ => None,
        }
    }

    /// Count distinct values per column in a batch.
    /// Returns a map of column → distinct count within this batch.
    fn compute_column_ndv_from_records(records: &[ProximaRecord]) -> HashMap<String, u64> {
        // HLL sketch per column instead of an exact HashSet<String> holding
        // EVERY distinct value (unbounded memory for high-cardinality
        // columns on large batches). The consumer (`bump_column_ndv`)
        // already merges batches additively as an estimate — "seed NDV
        // before ANALYZE" — so the sketch's small relative error is within
        // the design's approximation.
        use crate::core::statistics::sketches::HyperLogLog;
        let mut sketches: HashMap<String, HyperLogLog> = HashMap::new();
        for record in records {
            for (col, node) in record.props.iter() {
                let ProximaTreeNode::Value(val) = node else {
                    continue;
                };
                let Some(key) = Self::value_to_ndv_key(val) else {
                    continue;
                };
                sketches
                    .entry(col.clone())
                    .or_default()
                    .insert_bytes(key.as_bytes());
            }
        }
        sketches
            .into_iter()
            .map(|(col, sketch)| (col, sketch.estimate()))
            .collect()
    }

    /// Merge per-column distinct counts from a batch into `CatalogColumnStatistics.distinct_count`.
    ///
    /// Uses an additive estimate (assumes no overlap between batches), capped at the current
    /// table row count. This matches how simple columnar stats systems seed NDV before ANALYZE.
    /// Errors are non-fatal (logged as warnings).
    async fn bump_column_ndv(
        &self,
        table_name: &str,
        tenant: Option<&str>,
        ndv_per_column: HashMap<String, u64>,
    ) {
        if ndv_per_column.is_empty() {
            return;
        }
        let ndv_per_column = match &self.deferred_stats {
            Some(queue) => {
                queue.lock().push(DeferredStatBump::Ndv {
                    table: table_name.to_string(),
                    tenant: tenant.map(str::to_string),
                    ndv_per_column,
                });
                return;
            }
            None => ndv_per_column,
        };
        let Ok((catalog, table_id)) = self
            .catalog_manager
            .resolve_table_scoped(table_name, tenant)
            .await
        else {
            return;
        };
        let mut stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
        let row_count_cap = stats.row_count.max(1);
        for (col, batch_ndv) in ndv_per_column {
            let col_stats = stats.column_stats.entry(col).or_default();
            let new_ndv = col_stats
                .distinct_count
                .unwrap_or(0)
                .saturating_add(batch_ndv);
            col_stats.distinct_count = Some(new_ndv.min(row_count_cap));
        }
        stats.last_analyzed_ms = Some(Self::now_unix_ms());
        if let Err(e) = catalog.update_statistics(&table_id, stats).await {
            warn!(table = %table_name, error = %e, "Failed to update column NDV statistics after INSERT");
        } else {
            Self::bump_corpus_version_for_stats(&table_id).await;
        }
    }

    /// Validate a batch of `ProximaRecord`s against the xCatalog schema for `collection_name`.
    ///
    /// Designed for fast-lane REST/gRPC/Arrow Flight writes that arrive as pre-built records
    /// rather than SQL literals. Silently skips validation if the collection is not registered
    /// as a relational table (non-relational collections remain unrestricted). Returns `Err`
    /// containing the column constraint violation message on the first failing record.
    pub async fn validate_record_batch_against_schema(
        &self,
        collection_name: &str,
        tenant: Option<&str>,
        records: &[ProximaRecord],
    ) -> Result<()> {
        let Ok((catalog, table_id)) = self
            .catalog_manager
            .resolve_table_scoped(collection_name, tenant)
            .await
        else {
            return Ok(());
        };
        let Ok(table_schema) = catalog.get_table(&table_id).await else {
            return Ok(());
        };
        // v2 record API path: skip relational schema validation for records that are
        // vector-shaped (carry ≥1 embedding cell) OR document-facade records (labelled
        // `document`, ADR-009 convergence). Both are v2 record-API shapes — a vector ingest
        // or a schemaless NF² document projection — NOT SQL DML rows. Relational schema
        // constraints (`reject_unknown_columns`, missing-required-column, type strictness)
        // reject perfectly valid vector-API batches — including ones that carry filter/doc
        // metadata in `props` — because the auto-registered schema is `id` + `vector` only
        // and treats anything else as unknown. Document records are the vectorless case: a
        // metadata-only document has no embedding but must not be validated against the
        // vector-collection's relational schema. (Vector path reconciled 2026-05-28 for the
        // v0.2 v2 INSERT→SEARCH gap; document label added for ADR-009.)
        let all_records_are_v2_record_api = !records.is_empty()
            && records.iter().all(|r| {
                !r.embeddings.is_empty()
                    || r.labels.contains(proximadb_document::DOCUMENT_RECORD_LABEL)
            });
        if all_records_are_v2_record_api {
            return Ok(());
        }
        // Determine which schema column (if any) maps to the record's canonical
        // identifier (`oid`). Auto-registered vector-collection schemas declare
        // either `id` or `record_id` for this — line up with the same convention
        // used in `dml_field_mapping` below. Without this projection, REST/gRPC
        // v2 INSERT records (whose OID lives outside `props`) fail validation
        // with "Missing required column 'id'" even though the OID was provided.
        let id_column_name: Option<String> = table_schema
            .columns
            .iter()
            .find(|c| c.name == "id" || c.name == "record_id")
            .map(|c| c.name.clone());
        let profile = RelationalWriteProfile::fast_lane();
        for record in records {
            let mut values: HashMap<String, ProximaValue> = record
                .props
                .iter()
                .filter_map(|(k, node)| {
                    if let proximadb_records::ProximaTreeNode::Value(v) = node {
                        Some((k.clone(), v.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            if let Some(id_col) = id_column_name.as_ref() {
                values
                    .entry(id_col.clone())
                    .or_insert_with(|| ProximaValue::String(record.oid.clone()));
            }
            CatalogRow::validate(&table_schema, values, &profile).with_context(|| {
                format!(
                    "record '{}' violates schema '{}'",
                    record.oid, collection_name
                )
            })?;
        }
        Ok(())
    }

    /// Build a canonical ProximaRecord from catalog schema and SQL literals.
    fn build_mutation_record(
        &self,
        columns: &[String],
        values: &[SqlValueLiteral],
        table_schema: &CatalogTableSchema,
        mutation_kind: RelationalMutationKind,
    ) -> Result<ProximaRecord> {
        // `INSERT INTO t VALUES (...)` without an explicit column
        // list is standard SQL — the values are mapped to the table's
        // declared columns in order. Synthesize the column list
        // here when it's empty so the validation + per-column
        // mapping below behaves uniformly. The arity check then
        // catches actual mismatches (too many/few values vs the
        // schema's column count) with a clearer error.
        let synthesized: Vec<String>;
        let columns: &[String] = if columns.is_empty() && !values.is_empty() {
            synthesized = table_schema
                .columns
                .iter()
                .map(|c| c.name.clone())
                .collect();
            &synthesized
        } else {
            columns
        };
        // TD-OLAP-18: normalize the provided spellings to the DECLARED column
        // names (exact match keeps its spelling; otherwise a unique
        // case-insensitive match resolves — same fold semantics as the read
        // path and the SELECT predicate/projection resolvers above). Every
        // downstream step (validation, per-column literal typing, row_values
        // keys, default-fill, CatalogRow::validate) is then keyed by declared
        // names, exactly as if the query had spelled them that way.
        let folded: Vec<String>;
        let columns: &[String] = if columns
            .iter()
            .all(|c| table_schema.columns.iter().any(|sc| &sc.name == c))
        {
            columns
        } else {
            folded = columns
                .iter()
                .map(|c| Self::resolve_declared_column_name(c, table_schema))
                .collect::<Result<_>>()?;
            &folded
        };
        if columns.len() != values.len() {
            return Err(anyhow!(
                "Column count ({}) doesn't match value count ({})",
                columns.len(),
                values.len()
            ));
        }

        self.validate_insert_columns(columns, values, table_schema)?;

        let mut row_values = HashMap::new();
        let mut created_at_ns = None;

        for (col, val) in columns.iter().zip(values.iter()) {
            let effective_value = self.effective_insert_literal(col, val, table_schema)?;
            let proxima_value =
                self.literal_to_proxima_value_for_column(col, &effective_value, table_schema)?;

            row_values.insert(col.clone(), proxima_value.clone());

            if col == "timestamp" {
                created_at_ns = self
                    .literal_to_timestamp(&effective_value)?
                    .map(|timestamp_ms| timestamp_ms.saturating_mul(1_000_000));
                continue;
            }
        }

        for column in &table_schema.columns {
            if columns.iter().any(|provided| provided == &column.name) {
                continue;
            }
            let Some(default_value) = &column.default_value else {
                continue;
            };
            let default_literal = Self::parse_default_literal(default_value)?;
            let proxima_value = self.literal_to_proxima_value_for_column(
                &column.name,
                &default_literal,
                table_schema,
            )?;
            row_values.insert(column.name.clone(), proxima_value.clone());
        }

        let catalog_row =
            CatalogRow::validate(table_schema, row_values, &RelationalWriteProfile::oltp())?;
        let record = catalog_row.to_mutation_record(
            table_schema,
            mutation_kind,
            RelationalRecordOptions {
                method: Some(Self::mutation_method(mutation_kind).to_string()),
                created_at_ns,
                ..RelationalRecordOptions::default()
            },
        )?;
        Ok(record)
    }

    fn mutation_method(kind: RelationalMutationKind) -> &'static str {
        match kind {
            RelationalMutationKind::Insert => "sql_insert",
            RelationalMutationKind::Upsert => "sql_upsert",
            RelationalMutationKind::Update => "sql_update",
            RelationalMutationKind::Delete => "sql_delete",
        }
    }

    /// Build a canonical tombstone record for a primary-key-targeted DELETE.
    /// Composite-PK aware (TD-110 S2): the `key_value` is the
    /// `\u{1f}`-joined encoded PK tuple; it is split into per-column values and
    /// each is parsed by the type-aware `primary_key_string_to_proxima_value`.
    fn build_delete_tombstone_record(
        key_value: &str,
        table_schema: &CatalogTableSchema,
        now_ns: i64,
    ) -> Result<ProximaRecord> {
        // TD-RELIDEQ-1: a table with no declared key addresses its rows by
        // `oid` (here `key_value`); there is no key tuple to decode.
        if !Self::table_declares_primary_key(table_schema) {
            let row = CatalogRow {
                table: table_schema.name.clone(),
                values: HashMap::new(),
            };
            return row.to_mutation_record(
                table_schema,
                RelationalMutationKind::Delete,
                RelationalRecordOptions {
                    method: Some("sql_delete".to_string()),
                    origin: Some("delete".to_string()),
                    created_at_ns: Some(now_ns),
                    updated_at_ns: Some(now_ns),
                    valid_to_ns: Some(0),
                    include_vector_embeddings: false,
                    record_id: Some(key_value.to_string()),
                    ..RelationalRecordOptions::default()
                },
            );
        }
        let primary_key_columns = Self::primary_key_columns(table_schema);
        if primary_key_columns.is_empty() {
            return Err(anyhow!(
                "Table '{}' has no primary key/id column for DELETE",
                table_schema.name
            ));
        }
        let parts = split_primary_key_tuple(key_value);
        if parts.len() != primary_key_columns.len() {
            return Err(anyhow!(
                "DELETE key '{}' for table '{}' decoded to {} part(s) but its primary key \
                 ({}) has {} column(s)",
                key_value,
                table_schema.name,
                parts.len(),
                primary_key_columns.join(", "),
                primary_key_columns.len()
            ));
        }
        let mut key_values = HashMap::with_capacity(primary_key_columns.len());
        for (column, part) in primary_key_columns.iter().zip(parts.iter()) {
            let value = Self::primary_key_string_to_proxima_value(column, part, table_schema)?;
            key_values.insert(column.clone(), value);
        }
        let catalog_row = CatalogRow::validate_primary_key(table_schema, key_values)?;

        catalog_row.to_mutation_record(
            table_schema,
            RelationalMutationKind::Delete,
            RelationalRecordOptions {
                method: Some("sql_delete".to_string()),
                origin: Some("delete".to_string()),
                created_at_ns: Some(now_ns),
                updated_at_ns: Some(now_ns),
                valid_to_ns: Some(0),
                include_vector_embeddings: false,
                ..RelationalRecordOptions::default()
            },
        )
    }

    fn primary_key_string_to_proxima_value(
        column_name: &str,
        key_value: &str,
        table_schema: &CatalogTableSchema,
    ) -> Result<ProximaValue> {
        let Some(column) = table_schema
            .columns
            .iter()
            .find(|column| column.name == column_name)
        else {
            return Ok(ProximaValue::String(key_value.to_string()));
        };

        match &column.data_type {
            ProximaType::Int8 => key_value
                .parse::<i8>()
                .map(ProximaValue::Int8)
                .map_err(|e| anyhow!("Invalid primary key value '{}': {}", key_value, e)),
            ProximaType::Int16 => key_value
                .parse::<i16>()
                .map(ProximaValue::Int16)
                .map_err(|e| anyhow!("Invalid primary key value '{}': {}", key_value, e)),
            ProximaType::Int32 => key_value
                .parse::<i32>()
                .map(ProximaValue::Int32)
                .map_err(|e| anyhow!("Invalid primary key value '{}': {}", key_value, e)),
            ProximaType::Int64 => key_value
                .parse::<i64>()
                .map(ProximaValue::Int64)
                .map_err(|e| anyhow!("Invalid primary key value '{}': {}", key_value, e)),
            ProximaType::Uuid => Ok(ProximaValue::String(key_value.to_string())),
            ProximaType::String => Ok(ProximaValue::String(key_value.to_string())),
            other => Err(anyhow!(
                "Primary key column '{}' with type {:?} is not supported for DELETE key extraction",
                column_name,
                other
            )),
        }
    }

    fn validate_insert_columns(
        &self,
        columns: &[String],
        values: &[SqlValueLiteral],
        table_schema: &CatalogTableSchema,
    ) -> Result<()> {
        for column in columns {
            if !table_schema
                .columns
                .iter()
                .any(|schema_column| schema_column.name == *column)
            {
                return Err(anyhow!(
                    "Column '{}' does not exist in table '{}'",
                    column,
                    table_schema.name
                ));
            }
        }

        for schema_column in &table_schema.columns {
            if schema_column.nullable || schema_column.default_value.is_some() {
                continue;
            }
            let Some(position) = columns
                .iter()
                .position(|column| column == &schema_column.name)
            else {
                return Err(anyhow!(
                    "Column '{}' is required for table '{}'",
                    schema_column.name,
                    table_schema.name
                ));
            };
            if values.get(position).is_some_and(Self::literal_is_null) {
                return Err(anyhow!(
                    "Column '{}' cannot be NULL for table '{}'",
                    schema_column.name,
                    table_schema.name
                ));
            }
        }

        Ok(())
    }

    fn validate_update_assignments(
        assignments: &[(String, SqlValueLiteral)],
        table_schema: &CatalogTableSchema,
    ) -> Result<()> {
        if assignments.is_empty() {
            return Err(anyhow!("UPDATE requires at least one assignment"));
        }

        let primary_key_column = Self::primary_key_column(table_schema);
        for (column_name, value) in assignments {
            let Some(column) = table_schema
                .columns
                .iter()
                .find(|column| column.name == *column_name)
            else {
                return Err(anyhow!(
                    "Column '{}' does not exist in table '{}'",
                    column_name,
                    table_schema.name
                ));
            };
            if primary_key_column.as_deref() == Some(column_name.as_str())
                && Self::table_declares_primary_key(table_schema)
            {
                return Err(anyhow!(
                    "UPDATE cannot modify primary key column '{}'",
                    column_name
                ));
            }
            if Self::literal_is_null(value) && !column.nullable {
                return Err(anyhow!(
                    "Column '{}' cannot be NULL for table '{}'",
                    column.name,
                    table_schema.name
                ));
            }
            if matches!(value, SqlValueLiteral::Default) && column.default_value.is_none() {
                return Err(anyhow!("Column '{}' has no DEFAULT value", column_name));
            }
        }

        Ok(())
    }

    fn build_updated_proxima_record(
        &self,
        existing: RichSearchResult,
        assignments: &[(String, SqlValueLiteral)],
        table_schema: &CatalogTableSchema,
    ) -> Result<ProximaRecord> {
        let mut row_values = Self::row_values_from_existing(&existing, table_schema)?;
        let mut created_at_ns = existing
            .timestamp
            .map(|timestamp_ms| timestamp_ms.saturating_mul(1_000_000));
        let mut updated_at_ns = chrono::Utc::now()
            .timestamp_millis()
            .saturating_mul(1_000_000);

        for (column_name, value) in assignments {
            let effective_value =
                self.effective_insert_literal(column_name, value, table_schema)?;
            let proxima_value = self.literal_to_proxima_value_for_column(
                column_name,
                &effective_value,
                table_schema,
            )?;

            if column_name == "timestamp"
                && let Some(timestamp_ms) = self.literal_to_timestamp(&effective_value)?
            {
                let timestamp_ns = timestamp_ms.saturating_mul(1_000_000);
                created_at_ns = Some(timestamp_ns);
                updated_at_ns = timestamp_ns;
            }

            row_values.insert(column_name.clone(), proxima_value);
        }

        let catalog_row =
            CatalogRow::validate(table_schema, row_values, &RelationalWriteProfile::oltp())?;
        catalog_row.to_mutation_record(
            table_schema,
            RelationalMutationKind::Update,
            RelationalRecordOptions {
                method: Some("sql_dml_update".to_string()),
                created_at_ns,
                updated_at_ns: Some(updated_at_ns),
                record_version: existing.version.map(|version| version as u64 + 1),
                // TD-RELIDEQ-1: a keyless row keeps its oid (ignored when the
                // table declares a key, whose value is the oid).
                record_id: Some(existing.id.clone()),
                ..RelationalRecordOptions::default()
            },
        )
    }

    fn row_values_from_existing(
        existing: &RichSearchResult,
        table_schema: &CatalogTableSchema,
    ) -> Result<HashMap<String, ProximaValue>> {
        let primary_key_column = Self::primary_key_column(table_schema);
        let mut row_values = HashMap::new();

        for column in &table_schema.columns {
            // TD-RELIDEQ-1: only a declared key lives in the oid; a keyless
            // table's `id`/`record_id` fallback column is an ordinary column.
            if primary_key_column.as_deref() == Some(column.name.as_str())
                && Self::table_declares_primary_key(table_schema)
            {
                row_values.insert(
                    column.name.clone(),
                    Self::primary_key_string_to_proxima_value(
                        &column.name,
                        &existing.id,
                        table_schema,
                    )?,
                );
                continue;
            }

            if matches!(column.data_type, ProximaType::DenseVector { .. }) {
                if existing.vector.is_empty() && column.nullable {
                    row_values.insert(column.name.clone(), ProximaValue::Null);
                } else {
                    row_values.insert(
                        column.name.clone(),
                        ProximaValue::DenseVector(existing.vector.clone()),
                    );
                }
                continue;
            }

            if column.name == "timestamp" {
                if let Some(timestamp_ms) = existing.timestamp {
                    row_values.insert(
                        column.name.clone(),
                        match &column.data_type {
                            ProximaType::TimestampTz(_) => ProximaValue::TimestampTz(
                                timestamp_ms,
                                proximadb_data_model::TimeUnit::Millisecond,
                            ),
                            _ => ProximaValue::Timestamp(
                                timestamp_ms,
                                proximadb_data_model::TimeUnit::Millisecond,
                            ),
                        },
                    );
                }
                continue;
            }

            if let Some(value) = existing.props.get(&column.name) {
                row_values.insert(column.name.clone(), value.clone());
            }
        }

        Ok(row_values)
    }

    fn literal_is_null(value: &SqlValueLiteral) -> bool {
        matches!(value, SqlValueLiteral::Null)
    }

    fn primary_key_column(table_schema: &CatalogTableSchema) -> Option<String> {
        // Shared with the record-store index (TD-110 Slice C) so candidate and
        // indexed tuples derive their PK column identically.
        crate::services::record_store::schema_primary_key_column(table_schema)
    }

    /// TD-110 S2: all primary-key columns (composite-aware), in catalog order.
    /// Used by the composite-FK "references parent PK" check and the composite-PK
    /// tombstone decode. Shared with the record-store layer for parity.
    fn primary_key_columns(table_schema: &CatalogTableSchema) -> Vec<String> {
        crate::services::record_store::schema_primary_key_columns(table_schema)
    }

    /// TD-110: the column sets that carry a UNIQUE guarantee — cataloged unique
    /// indexes plus inline `UNIQUE (...)` column constraints. Delegates to the
    /// shared store-layer helper so DmlService candidates and the store's index
    /// enforce exactly the same sets.
    #[allow(dead_code)] // store-layer parity helper; wired by upcoming slice
    fn unique_column_sets(table_schema: &CatalogTableSchema) -> Vec<Vec<String>> {
        crate::services::record_store::schema_unique_column_sets(table_schema)
    }

    /// The record's typed PK tuple, one value per `pk_columns` entry in order —
    /// the input `encode_identity` keys the fenced ConditionalKeyStore on.
    /// `None` when any PK column is absent or non-scalar.
    fn record_primary_key_values(
        record: &ProximaRecord,
        pk_columns: &[String],
    ) -> Option<Vec<ProximaValue>> {
        pk_columns
            .iter()
            .map(|column| match record.props.get(column) {
                Some(ProximaTreeNode::Value(value)) => Some(value.clone()),
                _ => None,
            })
            .collect()
    }

    /// The typed 23505 error for a secondary-UNIQUE conflict (shared by the
    /// INSERT/UPSERT enforcement sequence and UPDATE).
    fn unique_violation(
        table_schema: &CatalogTableSchema,
        conflict: &crate::services::record_store::UniqueConflict,
    ) -> anyhow::Error {
        RelationalConstraintViolation::new(
            RelationalConstraintKind::Unique,
            format!(
                "duplicate key value violates unique constraint on ({}) for table '{}': ({}) already exists",
                conflict.columns.join(", "),
                table_schema.name,
                conflict.tuple.join(", ")
            ),
        )
        .into()
    }

    /// TD-DML-1: the ONE pre-write integrity sequence for row-producing DML
    /// (INSERT and every UPSERT spelling). Runs, in order:
    ///
    /// 1. PRIMARY KEY — within-statement repeat, then per row the F2 fenced
    ///    `ConditionalKeyStore::put_if_absent` claim when wired (ADR-072 D3),
    ///    else the point `get_by_key` probe;
    /// 2. secondary UNIQUE — `build_unique_candidate_sets` +
    ///    `check_unique_conflict` (TD-110);
    /// 3. FOREIGN KEY — `enforce_foreign_keys` (TD-110).
    ///
    /// CHECK / NOT NULL / type constraints are row-local and already ran in
    /// `CatalogRow::validate` when the records were built.
    ///
    /// Returns, per record in order, whether its PK already had a live row. For
    /// `Insert` a live holder is a 23505 error, so every entry is `false`; for
    /// `Upsert` it marks the row being replaced. If a later step rejects the
    /// statement, PK claims this call made in the fenced store are released so
    /// a rejected statement leaves no claimed-but-rowless key behind.
    async fn enforce_row_write_constraints(
        &self,
        table_schema: &CatalogTableSchema,
        table_id: &TableIdentifier,
        records: &[ProximaRecord],
        mode: RowWriteConstraintMode,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Vec<bool>> {
        let mut fresh_claims: Vec<proximadb_storage_ports::Identity> = Vec::new();
        let outcome = self
            .enforce_row_write_constraints_inner(
                table_schema,
                table_id,
                records,
                mode,
                tenant_context,
                &mut fresh_claims,
            )
            .await;
        if outcome.is_err()
            && let Some(cks) = &self.conditional_key_store
        {
            for id in &fresh_claims {
                if let Err(release_err) = cks
                    .tombstone(id, proximadb_storage_ports::Generation(0))
                    .await
                {
                    warn!(
                        table = %table_schema.name,
                        error = %release_err,
                        "failed to release a fenced PK claim after a rejected write"
                    );
                }
            }
        }
        outcome
    }

    async fn enforce_row_write_constraints_inner(
        &self,
        table_schema: &CatalogTableSchema,
        table_id: &TableIdentifier,
        records: &[ProximaRecord],
        mode: RowWriteConstraintMode,
        tenant_context: Option<&TenantContext>,
        fresh_claims: &mut Vec<proximadb_storage_ports::Identity>,
    ) -> Result<Vec<bool>> {
        let mut pk_live = vec![false; records.len()];

        // TD-110 Slice B: PRIMARY KEY — reject a key that repeats within this
        // statement OR (INSERT) already exists as a committed row. Composite-PK
        // aware (TD-DML-2): the fenced identity is built from the FULL PK tuple.
        let pk_columns = Self::primary_key_columns(table_schema);
        if !pk_columns.is_empty() {
            let pk_column = pk_columns.join(", ");
            let mut batch_keys: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for (index, record) in records.iter().enumerate() {
                let key = record.oid.as_str();
                if !batch_keys.insert(key) {
                    return Err(match mode {
                        RowWriteConstraintMode::Insert => RelationalConstraintViolation::new(
                            RelationalConstraintKind::Unique,
                            format!(
                                "duplicate key value violates primary key '{}' on table '{}': '{}' appears more than once in this INSERT",
                                pk_column, table_schema.name, key
                            ),
                        ),
                        RowWriteConstraintMode::Upsert => RelationalConstraintViolation::new(
                            RelationalConstraintKind::UpsertAffectsRowTwice,
                            format!(
                                "ON CONFLICT DO UPDATE command cannot affect row a second time: primary key '{}' value '{}' appears more than once in this statement on table '{}'",
                                pk_column, key, table_schema.name
                            ),
                        ),
                    }
                    .into());
                }
                // F2 (ADR-072 D3): when a fenced ConditionalKeyStore is wired,
                // claim the PK atomically (oid == primary_key_string, so the row
                // is keyed by its own PK) — a live holder means the key exists.
                // Otherwise fall back to the legacy check-then-act `get_by_key`
                // probe. The op-lock is held by the caller for cascade ordering
                // (D3). NB: the claim and the later `write_mutations` row write
                // are not yet a single atomic apply — that is the D11
                // append-only-MVCC follow-up.
                // TD-RELIDEQ-1: CKS fences the DECLARED key's value. A keyless
                // table's identity is its oid (a generated id), so claiming by
                // the `id`/`record_id` fallback value would fence an identity
                // nothing ever releases — take the legacy probe instead.
                let is_live = if let Some(cks) = self
                    .conditional_key_store
                    .as_ref()
                    .filter(|_| Self::table_declares_primary_key(table_schema))
                {
                    use proximadb_storage_ports::{Generation, Identity, Oid, PutOutcome};
                    // Key on the F0 typed, order-preserving codec built from the
                    // row's typed PK tuple (every PK column, in catalog order —
                    // TD-DML-2) and its tenant/keyspace scope — not the legacy
                    // oid string (ADR-072 D9).
                    let pk_values = Self::record_primary_key_values(record, &pk_columns)
                        .ok_or_else(|| {
                            anyhow!(
                                "primary key ({}) missing from record for '{}' on table '{}'",
                                pk_column,
                                key,
                                table_schema.name
                            )
                        })?;
                    let id =
                        Identity::from_bytes(proximadb_data_model::key_codec::encode_identity(
                            &record.tenant_id,
                            &table_schema.name,
                            &pk_values,
                        )?);
                    match cks
                        .put_if_absent(&id, &Oid(key.to_string()), Generation(0))
                        .await?
                    {
                        PutOutcome::Conflict { .. } => true,
                        PutOutcome::Committed { .. } => {
                            fresh_claims.push(id);
                            false
                        }
                    }
                } else {
                    self.record_store
                        .get_by_key(
                            table_schema,
                            TableRecordGetRequest {
                                table_id: table_id.name.clone(),
                                key: key.to_string(),
                                include_vector: false,
                                include_props: false,
                            },
                            // The tenant's partition — the one this INSERT writes.
                            // Probing with `None` looked in the default partition,
                            // missed the existing row in every tenant-scoped
                            // session, and let the duplicate surface later as an
                            // untyped store conflict (42P01 instead of 23505).
                            tenant_context,
                            #[cfg(feature = "abac-policy")]
                            &ReadContext::system(
                                SystemReadReason::ForeignKeyResolution,
                                "dml::fk_check",
                            ),
                        )
                        .await?
                        .is_some()
                };
                if is_live && mode == RowWriteConstraintMode::Insert {
                    return Err(RelationalConstraintViolation::new(
                        RelationalConstraintKind::Unique,
                        format!(
                            "duplicate key value violates primary key '{}' on table '{}': '{}' already exists",
                            pk_column, table_schema.name, key
                        ),
                    )
                    .into());
                }
                pk_live[index] = is_live;
            }
        }

        // TD-110: secondary UNIQUE constraints/indexes — within-statement dedup +
        // cross-existing probe (O(1) on index-backed stores, Slice C). INSERT
        // excludes nothing (every row is new). UPSERT excludes the statement's
        // own oids: a replaced row vacates its old UNIQUE values (exactly as
        // UPDATE excludes its target rows), and a new row's oid is not stored.
        let primary_key = Self::primary_key_column(table_schema);
        let candidate_sets =
            Self::build_unique_candidate_sets(table_schema, records, primary_key.as_deref())?;
        if !candidate_sets.is_empty() {
            let exclude_oids: std::collections::HashSet<String> = match mode {
                RowWriteConstraintMode::Insert => std::collections::HashSet::new(),
                RowWriteConstraintMode::Upsert => {
                    records.iter().map(|record| record.oid.clone()).collect()
                }
            };
            if let Some(conflict) = self
                .record_store
                .check_unique_conflict(
                    table_schema,
                    &table_id.name,
                    primary_key.as_deref(),
                    &candidate_sets,
                    &exclude_oids,
                    tenant_context,
                )
                .await?
            {
                return Err(Self::unique_violation(table_schema, &conflict));
            }
        }

        // TD-110: FOREIGN KEY references (same-partition cross-table).
        self.enforce_foreign_keys(table_schema, records, tenant_context)
            .await?;

        Ok(pk_live)
    }

    /// TD-110: build the per-set candidate tuples for `records`, rejecting a tuple
    /// that repeats within this statement (NULL tuples exempt). Shared by INSERT
    /// and UPDATE; the caller passes the result to `check_unique_conflict`.
    fn build_unique_candidate_sets(
        table_schema: &CatalogTableSchema,
        records: &[ProximaRecord],
        primary_key: Option<&str>,
    ) -> Result<Vec<crate::services::record_store::UniqueCandidateSet>> {
        // Shared with the INSERT-SELECT native executor so every write path
        // builds UNIQUE candidates identically (TD-110).
        crate::services::record_store::build_unique_candidate_sets(
            table_schema,
            records,
            primary_key,
        )
    }

    /// Maximum CASCADE recursion depth (TD-110 S1). Bounds N-level cascades so a
    /// pathological (or accidentally wide) FK graph cannot exhaust the stack or
    /// spin; a chain this deep is almost certainly a schema error, so it trips a
    /// typed error rather than running.
    const MAX_CASCADE_DEPTH: u32 = 16;

    /// TD-110: enforce ON DELETE referential actions for the rows about to be
    /// removed from `parent_table_id`. For every sibling table in the same
    /// namespace whose single-column FK references this parent's primary key,
    /// apply the catalogued action to the referencing child rows:
    ///   * `NO ACTION` / `RESTRICT` (and absent-action default) → reject the
    ///     DELETE (error) if any referencing child rows remain.
    ///   * `CASCADE` → delete the referencing child rows, **recursing** into
    ///     their children so N-level chains cascade fully (S1).
    ///   * `SET NULL` → clear the child FK column on the referencing rows.
    ///   * `SET DEFAULT` → rejected as unsupported.
    ///
    /// S1: before any mutation, discover the transitive child closure and acquire
    /// DML write locks on it in deterministic `(namespace, name)` order — this
    /// closes the TOCTOU (a concurrent child `INSERT` after the restrict-check /
    /// cascade-scan would orphan) and avoids deadlock between concurrent DELETEs
    /// over overlapping child sets. The parent table's lock is already held by
    /// `execute_delete` for the duration of this call. CASCADE recursion carries
    /// a `recursion_stack` (cycle detection — a cyclic CASCADE chain is a schema
    /// error and is rejected) and a depth guard (`MAX_CASCADE_DEPTH`). Composite
    /// FKs and self-references are skipped; cross-namespace/deferrable semantics
    /// are out of scope. All child resolution is scoped to the SAME
    /// `tenant_context` as the parent DELETE (FKs are intra-tenant).
    async fn enforce_delete_referential_actions(
        &self,
        parent_catalog: &Arc<dyn crate::catalog::Catalog>,
        parent_table_id: &crate::catalog::TableIdentifier,
        parent_schema: &CatalogTableSchema,
        deleted_keys: &[String],
        tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        let Some(_parent_pk) = Self::primary_key_column(parent_schema) else {
            return Ok(()); // No PK → nothing can reference these rows.
        };

        // Reject a cyclic CASCADE chain up front, before any mutation, so the
        // error leaves nothing partially deleted. The transitive write locks
        // (in-process op-locks + durable DML locks) for the child set are
        // acquired by the caller, `execute_delete`, and held across this cascade
        // AND the subsequent parent-tombstone write — closing the cross-table
        // TOCTOU for the full critical section (cross-pod via the durable locks,
        // intra-pod via the non-reentrant op-locks).
        self.assert_no_cascade_cycle(
            parent_catalog,
            parent_table_id,
            parent_schema,
            tenant_context,
        )
        .await?;

        self.enforce_delete_referential_actions_inner(
            parent_catalog,
            parent_table_id,
            parent_schema,
            deleted_keys,
            tenant_context,
            std::collections::HashSet::new(),
            0,
        )
        .await
    }

    /// TD-110 S1: acquire the cross-pod DML write locks for the transitive child
    /// set (a `BTreeSet`, so already in deterministic `(namespace, name)` order)
    /// to close the cross-table TOCTOU across pods. Non-blocking: a conflict
    /// surfaces as a typed `ProximaDBError::DmlLockConflict` (pgwire 55P03 / gRPC
    /// ABORTED) for client retry, and already-acquired guards are released first
    /// so no lock is leaked. Returns an empty vec when no DML lock service is
    /// configured (embedded/test paths — intra-pod serialization is then provided
    /// by the in-process op-locks).
    async fn acquire_cascade_child_dml_locks(
        &self,
        child_set: &std::collections::BTreeSet<(Vec<String>, String)>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Vec<DmlLockGuard>> {
        if self.dml_lock_service.is_none() || child_set.is_empty() {
            return Ok(Vec::new());
        }
        // BTreeSet iterates in sorted (namespace, name) order — acquire in that
        // order so concurrent DELETEs over overlapping child sets can't deadlock.
        let mut guards: Vec<DmlLockGuard> = Vec::with_capacity(child_set.len());
        for (namespace, name) in child_set {
            let lock_target = TableIdentifier {
                namespace: namespace.clone(),
                name: name.clone(),
            };
            match self
                .acquire_table_dml_lock(&lock_target, tenant_context, LockIntent::Write)
                .await
            {
                Ok(Some(guard)) => guards.push(guard),
                Ok(None) => {} // No lock service configured → locking is a no-op.
                Err(e) => {
                    for guard in guards {
                        guard.release().await;
                    }
                    return Err(e);
                }
            }
        }
        Ok(guards)
    }

    /// TD-110: the child tables (anywhere in the tenant's namespace subtree —
    /// cross-namespace within the tenant, S3) whose (single- or multi-column) FK
    /// references `parent_id`'s full primary key, with the FK columns and action.
    /// Factored out so both the discovery pass and the recursive worker share one
    /// copy of the FK-matching logic. Self-references are excluded; only FKs that
    /// reference the parent's FULL PK (in order) are returned — partial / non-PK
    /// references remain unsupported (S2 scope).
    async fn child_tables_referencing(
        &self,
        catalog: &Arc<dyn crate::catalog::Catalog>,
        parent_id: &crate::catalog::TableIdentifier,
        parent_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<
        Vec<(
            crate::catalog::TableIdentifier,
            CatalogTableSchema,
            Vec<String>,
            Option<proximadb_catalog::ReferentialAction>,
        )>,
    > {
        let parent_pk_cols = Self::primary_key_columns(parent_schema);
        if parent_pk_cols.is_empty() {
            return Ok(Vec::new());
        }
        // TD-110 S3: enumerate candidate children across the tenant's WHOLE
        // namespace subtree (not just the parent's namespace), so a child living
        // in another namespace whose FK references this parent is found. Scoped
        // to the tenant (`Some([tenant])`) to keep child discovery intra-tenant;
        // cross-tenant children are never enumerated. The per-child reference
        // match below still requires the FK to resolve to THIS parent (name AND
        // namespace), so a same-named table in another namespace never matches.
        let tenant_scope = tenant_context.map(|t| vec![t.tenant_id.clone()]);
        let sibling_ids = catalog
            .list_all_tables_in_scope(tenant_scope.as_deref())
            .await?;
        let mut out = Vec::new();
        for child_id in sibling_ids {
            if child_id.name == parent_id.name && child_id.namespace == parent_id.namespace {
                continue; // Self-referencing FK on the same table: out of scope.
            }
            let child_schema = catalog.get_table(&child_id).await?;
            for constraint in &child_schema.relational_capabilities.constraints {
                let proximadb_catalog::ColumnConstraint::ForeignKey {
                    columns,
                    references_table,
                    references_columns,
                    on_delete,
                    ..
                } = constraint
                else {
                    continue;
                };
                if columns.len() != references_columns.len() {
                    continue; // Malformed FK (mismatched arities): skip.
                }
                // Does this FK reference `parent_id`? Resolve under the SAME
                // tenant scope so cross-tenant FKs never match.
                let Ok((_, referenced_id)) = self
                    .catalog_manager
                    .resolve_table_scoped(
                        references_table,
                        tenant_context.map(|t| t.tenant_id.as_str()),
                    )
                    .await
                else {
                    continue;
                };
                if referenced_id.name != parent_id.name
                    || referenced_id.namespace != parent_id.namespace
                {
                    continue;
                }
                // Only FKs referencing the parent's FULL primary key (in order)
                // are enforced; partial / non-PK references are not (S2 scope).
                if references_columns.as_slice() != parent_pk_cols.as_slice() {
                    continue;
                }
                out.push((
                    child_id.clone(),
                    child_schema.clone(),
                    columns.clone(),
                    *on_delete,
                ));
            }
        }
        Ok(out)
    }

    /// TD-110 S1: the transitive closure of child tables reachable from
    /// `parent_id` via any single-column FK on the parent PK (any action). Used
    /// to size the DML write-lock set before the cascade. Cycle- and
    /// diamond-safe (each table visited once); excludes the parent itself.
    async fn discover_cascade_child_set(
        &self,
        catalog: &Arc<dyn crate::catalog::Catalog>,
        parent_id: &crate::catalog::TableIdentifier,
        parent_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<std::collections::BTreeSet<(Vec<String>, String)>> {
        let mut result: std::collections::BTreeSet<(Vec<String>, String)> =
            std::collections::BTreeSet::new();
        let mut visited: std::collections::HashSet<(Vec<String>, String)> =
            std::collections::HashSet::new();
        visited.insert((parent_id.namespace.clone(), parent_id.name.clone()));
        let mut frontier: Vec<(crate::catalog::TableIdentifier, CatalogTableSchema)> =
            vec![(parent_id.clone(), parent_schema.clone())];
        while let Some((tid, schema)) = frontier.pop() {
            for (child_id, child_schema, _fk_column, _on_delete) in self
                .child_tables_referencing(catalog, &tid, &schema, tenant_context)
                .await?
            {
                let key = (child_id.namespace.clone(), child_id.name.clone());
                if visited.insert(key.clone()) {
                    result.insert(key);
                    frontier.push((child_id, child_schema));
                }
                // Already visited (cycle / diamond) → skip; the cascade itself
                // is guarded by the recursion stack + depth limit.
            }
        }
        Ok(result)
    }

    /// TD-110 S1: detect a CASCADE cycle reachable from `parent_id` BEFORE any
    /// lock or mutation, so a cyclic CASCADE chain (a schema error) is rejected
    /// cleanly with no partial deletion. DFS over CASCADE edges only — RESTRICT
    /// / SET NULL don't recurse, so they can't form a cascade cycle. `gray` =
    /// on the current DFS path (a repeat ⇒ cycle); `black` = fully explored
    /// (skip). The recursion-stack check in the inner worker stays as a
    /// defense-in-depth backstop.
    async fn assert_no_cascade_cycle(
        &self,
        catalog: &Arc<dyn crate::catalog::Catalog>,
        parent_id: &crate::catalog::TableIdentifier,
        parent_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        let mut gray: std::collections::HashSet<(Vec<String>, String)> =
            std::collections::HashSet::new();
        let mut black: std::collections::HashSet<(Vec<String>, String)> =
            std::collections::HashSet::new();
        Box::pin(self.assert_no_cascade_cycle_visit(
            catalog,
            parent_id,
            parent_schema,
            tenant_context,
            &mut gray,
            &mut black,
        ))
        .await
    }

    async fn assert_no_cascade_cycle_visit(
        &self,
        catalog: &Arc<dyn crate::catalog::Catalog>,
        id: &crate::catalog::TableIdentifier,
        schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
        gray: &mut std::collections::HashSet<(Vec<String>, String)>,
        black: &mut std::collections::HashSet<(Vec<String>, String)>,
    ) -> Result<()> {
        let key = (id.namespace.clone(), id.name.clone());
        if black.contains(&key) {
            return Ok(());
        }
        if gray.contains(&key) {
            return Err(anyhow!(
                "ON DELETE cascade cycle detected at table '{}.{}': cyclic CASCADE chains are not supported",
                id.namespace.join("."),
                id.name
            ));
        }
        gray.insert(key.clone());
        for (child_id, child_schema, _fk_column, on_delete) in self
            .child_tables_referencing(catalog, id, schema, tenant_context)
            .await?
        {
            if matches!(
                on_delete,
                Some(proximadb_catalog::ReferentialAction::Cascade)
            ) {
                Box::pin(self.assert_no_cascade_cycle_visit(
                    catalog,
                    &child_id,
                    &child_schema,
                    tenant_context,
                    gray,
                    black,
                ))
                .await?;
            }
        }
        gray.remove(&key);
        black.insert(key);
        Ok(())
    }

    /// TD-110 S1: recursive worker that applies ON DELETE actions for one level
    /// and, for CASCADE, descends into the deleted children. `ancestors` is the
    /// set of `(namespace, table)` on the path from the root EXCLUDING the
    /// current table — a repeat of the current table on the path is a cycle.
    /// Diamond topologies (one table reached via two parents) are NOT cycles:
    /// the first cascade tombstones the rows, so a later reach scans empty and
    /// naturally skips (idempotent tombstones).
    async fn enforce_delete_referential_actions_inner(
        &self,
        catalog: &Arc<dyn crate::catalog::Catalog>,
        table_id: &crate::catalog::TableIdentifier,
        table_schema: &CatalogTableSchema,
        deleted_keys: &[String],
        tenant_context: Option<&TenantContext>,
        ancestors: std::collections::HashSet<(Vec<String>, String)>,
        depth: u32,
    ) -> Result<()> {
        let self_key = (table_id.namespace.clone(), table_id.name.clone());
        if ancestors.contains(&self_key) {
            return Err(anyhow!(
                "ON DELETE cascade cycle detected at table '{}.{}': cyclic CASCADE chains are not supported",
                table_id.namespace.join("."),
                table_id.name
            ));
        }
        if depth >= Self::MAX_CASCADE_DEPTH {
            return Err(anyhow!(
                "ON DELETE cascade exceeded the maximum depth of {} at table '{}' — likely a schema error",
                Self::MAX_CASCADE_DEPTH,
                table_id.name
            ));
        }
        let Some(_pk) = Self::primary_key_column(table_schema) else {
            return Ok(()); // No PK → nothing can reference these rows.
        };
        let deleted: std::collections::HashSet<&str> =
            deleted_keys.iter().map(String::as_str).collect();

        for (child_id, child_schema, fk_columns, on_delete) in self
            .child_tables_referencing(catalog, table_id, table_schema, tenant_context)
            .await?
        {
            // Find child rows referencing any deleted parent key (tuple). The FK
            // columns' typed values are extracted from the record and encoded
            // with the SAME encoder that built the deleted parent oids
            // (`encode_primary_key_tuple` / `stable_value_string`), so a probe
            // always matches the oid exactly — for both single- and multi-column
            // FKs. MATCH SIMPLE: any NULL/absent/non-scalar FK column → exempt.
            let fk_for_pred = fk_columns.clone();
            let deleted_ref = &deleted;
            let pred = move |record: &ProximaRecord| {
                let mut values: Vec<ProximaValue> = Vec::with_capacity(fk_for_pred.len());
                for col in &fk_for_pred {
                    match record.props.get(col) {
                        Some(ProximaTreeNode::Value(value))
                            if !matches!(value, ProximaValue::Null) =>
                        {
                            values.push(value.clone());
                        }
                        _ => return false, // NULL/absent/non-scalar FK → MATCH SIMPLE exempt.
                    }
                }
                match encode_primary_key_tuple(&values) {
                    Ok(encoded) => deleted_ref.contains(encoded.as_str()),
                    Err(_) => false,
                }
            };
            let predicate: Option<&proximadb_records::RecordScanPredicate<'_>> = Some(&pred);
            let referencing = self
                .record_store
                .scan_records_filtered(
                    &child_schema,
                    TableRecordScanRequest {
                        filter: None,
                        table_id: child_id.name.clone(),
                        limit: None,
                        include_vector: true,
                        include_props: true,
                    },
                    predicate,
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(
                        SystemReadReason::ForeignKeyResolution,
                        "dml::referencing_check",
                    ),
                )
                .await?;
            if referencing.is_empty() {
                continue;
            }

            match on_delete {
                None
                | Some(proximadb_catalog::ReferentialAction::Restrict)
                | Some(proximadb_catalog::ReferentialAction::NoAction) => {
                    return Err(RelationalConstraintViolation::new(
                        RelationalConstraintKind::ForeignKey,
                        format!(
                            "DELETE on table '{}' violates FOREIGN KEY ({}) on table '{}': {} referencing row(s) remain (ON DELETE NO ACTION)",
                            table_id.name,
                            fk_columns.join(", "),
                            child_id.name,
                            referencing.len()
                        ),
                    )
                    .into());
                }
                Some(proximadb_catalog::ReferentialAction::Cascade) => {
                    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                    // The child OIDs being tombstoned become the deleted-key set
                    // for the next cascade level (grandchildren, etc.).
                    let child_deleted_keys: Vec<String> = referencing
                        .iter()
                        .map(|record| record.oid.clone())
                        .collect();
                    let tombstones = referencing
                        .iter()
                        .map(|record| {
                            Self::build_delete_tombstone_record(&record.oid, &child_schema, now_ns)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let cascaded = tombstones.len();
                    let mutations = tombstones
                        .into_iter()
                        .map(|record| {
                            TableRecordMutation::new(TableRecordMutationKind::Delete, record)
                        })
                        .collect::<Vec<_>>();
                    let result = self
                        .record_store
                        .write_mutations(&child_schema, mutations, tenant_context)
                        .await?;
                    if !result.success {
                        return Err(anyhow!(
                            "ON DELETE CASCADE failed for child table '{}': {:?}",
                            child_id.name,
                            result.errors
                        ));
                    }
                    self.bump_row_count_stats(
                        &child_id.name,
                        tenant_context.map(|t| t.tenant_id.as_str()),
                        -(cascaded as i64),
                    )
                    .await;
                    // Recurse so grandchildren (and deeper) cascade too. Locks
                    // for the whole closure are already held by the entry. The
                    // call is boxed because async recursion requires it (the
                    // future's size is otherwise unbounded); the depth guard
                    // bounds the recursion to `MAX_CASCADE_DEPTH`.
                    if !child_deleted_keys.is_empty() {
                        let mut child_ancestors = ancestors.clone();
                        child_ancestors.insert(self_key.clone());
                        Box::pin(self.enforce_delete_referential_actions_inner(
                            catalog,
                            &child_id,
                            &child_schema,
                            &child_deleted_keys,
                            tenant_context,
                            child_ancestors,
                            depth + 1,
                        ))
                        .await?;
                    }
                }
                Some(proximadb_catalog::ReferentialAction::SetNull) => {
                    let mut updated = Vec::with_capacity(referencing.len());
                    for mut record in referencing {
                        // Null ALL FK columns (standard SET NULL on a composite FK).
                        for fk_col in &fk_columns {
                            record
                                .props
                                .insert(fk_col.clone(), ProximaTreeNode::Value(ProximaValue::Null));
                        }
                        updated.push(record);
                    }
                    let mutations = updated
                        .into_iter()
                        .map(|record| {
                            TableRecordMutation::new(TableRecordMutationKind::Update, record)
                        })
                        .collect::<Vec<_>>();
                    let result = self
                        .record_store
                        .write_mutations(&child_schema, mutations, tenant_context)
                        .await?;
                    if !result.success {
                        return Err(anyhow!(
                            "ON DELETE SET NULL failed for child table '{}': {:?}",
                            child_id.name,
                            result.errors
                        ));
                    }
                }
                Some(proximadb_catalog::ReferentialAction::SetDefault) => {
                    return Err(anyhow!(
                        "ON DELETE SET DEFAULT on table '{}' (FOREIGN KEY {}) is not supported",
                        child_id.name,
                        fk_columns.join(", ")
                    ));
                }
            }
        }
        Ok(())
    }

    /// TD-110: enforce FOREIGN KEY references for `records` against parent tables
    /// in the same partition (cross-table state the row-local catalog validator
    /// cannot check). Supported shape (S2): a single- or multi-column FK
    /// referencing the parent's FULL primary key (in order) — verified by a point
    /// `get_by_key` on the parent using the encoded PK tuple. MATCH SIMPLE: any
    /// NULL/absent FK column exempts the row. Partial / non-PK-referencing FKs
    /// are cleanly rejected rather than silently accepted.
    async fn enforce_foreign_keys(
        &self,
        table_schema: &CatalogTableSchema,
        records: &[ProximaRecord],
        tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        for constraint in &table_schema.relational_capabilities.constraints {
            let proximadb_catalog::ColumnConstraint::ForeignKey {
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
                    table_schema.name,
                    columns.len(),
                    references_columns.len()
                ));
            }
            let fk_columns = columns;

            let (parent_catalog, parent_table_id) = self
                .catalog_manager
                .resolve_table_scoped(references_table, tenant_context.map(|t| t.tenant_id.as_str()))
                .await
                .map_err(|err| {
                    anyhow!(
                        "FOREIGN KEY ({}) on table '{}' references table '{}' which cannot be resolved: {err}",
                        fk_columns.join(", "),
                        table_schema.name,
                        references_table
                    )
                })?;
            if !parent_catalog.table_exists(&parent_table_id).await? {
                return Err(anyhow!(
                    "FOREIGN KEY ({}) on table '{}' references missing table '{}'",
                    fk_columns.join(", "),
                    table_schema.name,
                    references_table
                ));
            }
            let parent_schema = parent_catalog.get_table(&parent_table_id).await?;
            let parent_pk_cols = Self::primary_key_columns(&parent_schema);
            // Only FKs referencing the parent's FULL primary key (in order) are
            // enforced — partial / non-PK references are not (S2 scope).
            if parent_pk_cols.is_empty()
                || references_columns.as_slice() != parent_pk_cols.as_slice()
            {
                return Err(anyhow!(
                    "FOREIGN KEY ({}) REFERENCES {}({}) on table '{}' is only supported when it references the parent primary key ({}) in full, in order",
                    fk_columns.join(", "),
                    references_table,
                    references_columns.join(", "),
                    table_schema.name,
                    parent_pk_cols.join(", ")
                ));
            }

            for record in records {
                // Extract the FK columns' typed values. MATCH SIMPLE: any NULL,
                // absent, or non-scalar FK column → exempt (no reference required).
                let mut values: Vec<ProximaValue> = Vec::with_capacity(fk_columns.len());
                let mut exempt = false;
                for col in fk_columns {
                    match record.props.get(col) {
                        Some(ProximaTreeNode::Value(value))
                            if !matches!(value, ProximaValue::Null) =>
                        {
                            values.push(value.clone());
                        }
                        _ => {
                            // NULL / absent / non-scalar FK column → MATCH SIMPLE exempt.
                            exempt = true;
                            break;
                        }
                    }
                }
                if exempt {
                    continue;
                }
                // Encode with the same encoder that built the parent oids so the
                // point lookup matches exactly (single- or multi-column).
                let key = encode_primary_key_tuple(&values)?;
                // F5 (ADR-072): when the fenced ConditionalKeyStore is wired,
                // resolve FK existence against the SAME store that fences the
                // parent's PK uniqueness (its `get` was designed for exactly this
                // — "foreign-key existence checks (phase 2) ride this method"),
                // instead of a check-then-act probe on the record store. This
                // makes the child-insert see the authoritative uniqueness index
                // and closes the TOCTOU with a concurrent parent delete. The FK
                // references the parent's FULL PK in order (checked above), and
                // the parent claim is keyed on that full tuple (TD-DML-2), so
                // single- and multi-column FKs both resolve through the CKS.
                // TD-RELIDEQ-1: only a parent with a DECLARED key is fenced in
                // the CKS; a keyless parent resolves through the record store.
                let referenced_exists = if let Some(cks) = self
                    .conditional_key_store
                    .as_ref()
                    .filter(|_| Self::table_declares_primary_key(&parent_schema))
                {
                    let id = proximadb_storage_ports::Identity::from_bytes(
                        proximadb_data_model::key_codec::encode_identity(
                            &record.tenant_id,
                            &parent_schema.name,
                            &values,
                        )?,
                    );
                    cks.get(&id).await?.is_some()
                } else {
                    self.record_store
                        .get_by_key(
                            &parent_schema,
                            TableRecordGetRequest {
                                table_id: parent_table_id.name.clone(),
                                key: key.clone(),
                                include_vector: false,
                                include_props: false,
                            },
                            tenant_context,
                            #[cfg(feature = "abac-policy")]
                            &ReadContext::system(
                                SystemReadReason::ForeignKeyResolution,
                                "dml::fk_check",
                            ),
                        )
                        .await?
                        .is_some()
                };
                if !referenced_exists {
                    return Err(RelationalConstraintViolation::new(
                        RelationalConstraintKind::ForeignKey,
                        format!(
                            "FOREIGN KEY ({}) on table '{}' violates reference: '{}' is not present in {}({})",
                            fk_columns.join(", "),
                            table_schema.name,
                            key,
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

    /// Extract IDs from WHERE clause using the catalog primary key.
    /// Resolve the canonical OIDs of the rows an UPDATE/DELETE `WHERE` clause
    /// targets, honoring the FULL predicate (any catalog column), not just the
    /// primary key. Reuses the same predicate evaluator + scan push-down as
    /// SELECT, so mixed `pk = x AND col = y` no longer mutates rows that fail the
    /// non-PK part (the prior silent bug), and non-PK `WHERE` now works.
    async fn resolve_matching_ids(
        &self,
        table_schema: &CatalogTableSchema,
        table_id_name: &str,
        where_clause: &WhereClause,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Vec<String>> {
        let tree = self.where_clause_to_predicate_tree(where_clause, table_schema)?;
        let primary_key = table_schema.primary_key.first().map(String::as_str);

        // PK fast-path: when the WHERE pins primary-key `=`/`IN` candidates (only
        // sound under a top-level conjunction — see `extract_pk_candidate_ids`),
        // fetch them by key and keep only those that ALSO satisfy the full tree.
        let pk_candidates = self.extract_pk_candidate_ids(where_clause, table_schema)?;
        // TD-RELIDEQ-1: see `select_table_records_with_tree` — a miss on the
        // `id`/`record_id` fallback key falls through to the predicate scan.
        let key_is_declared = Self::table_declares_primary_key(table_schema);
        let mut fallback_key_missed = false;
        if !pk_candidates.is_empty() {
            let mut ids = Vec::new();
            for candidate in pk_candidates {
                let Some(rich) = self
                    .record_store
                    .get_by_key(
                        table_schema,
                        TableRecordGetRequest {
                            table_id: table_id_name.to_string(),
                            key: candidate,
                            include_vector: true,
                            include_props: true,
                        },
                        tenant_context,
                        #[cfg(feature = "abac-policy")]
                        &ReadContext::system(
                            SystemReadReason::Statistics,
                            "dml [CLIENT-PLACEHOLDER]",
                        ),
                    )
                    .await?
                else {
                    fallback_key_missed |= !key_is_declared;
                    continue;
                };
                let record = Self::rich_result_to_record(rich);
                if Self::eval_predicate_tree(&record, &tree, primary_key) {
                    ids.push(record.oid);
                }
            }
            if !fallback_key_missed {
                return Ok(ids);
            }
        }

        // No PK predicate: push the full predicate tree into the store scan.
        let pred = |record: &ProximaRecord| Self::eval_predicate_tree(record, &tree, primary_key);
        let predicate: Option<&proximadb_records::RecordScanPredicate<'_>> = Some(&pred);
        let records = self
            .record_store
            .scan_records_filtered(
                table_schema,
                TableRecordScanRequest {
                    filter: predicate_tree_to_filter_expression(&tree),
                    table_id: table_id_name.to_string(),
                    limit: None,
                    include_vector: true,
                    include_props: true,
                },
                predicate,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "dml [CLIENT-PLACEHOLDER]"),
            )
            .await?;
        Ok(records.into_iter().map(|record| record.oid).collect())
    }

    /// Extract primary-key `=`/`IN` candidate OIDs from a WHERE clause. Returns
    /// an empty Vec (NOT an error) when the clause has no usable PK predicate, so
    /// the caller can fall back to a predicate scan. Candidates are still
    /// re-checked against the full predicate by [`Self::resolve_matching_ids`].
    /// TD-127: extract a single-column equality / non-negated `IN`-list on a
    /// secondary-indexed non-PK column from `where_clause`, returning
    /// `(column, value-texts)` to probe the OLTP secondary index. Value text is
    /// rendered through the SAME [`proxima_value_to_unique_text`] the index uses,
    /// so the probe text matches the indexed text exactly. Returns `None` (scan
    /// fallback) when no such leaf exists.
    ///
    /// OR-safety: only sound under a top-level conjunction — `name = 'x' OR ...`
    /// would under-bound the match set — mirroring [`Self::extract_pk_candidate_ids`].
    fn extract_secondary_index_probe(
        &self,
        where_clause: &WhereClause,
        table_schema: &CatalogTableSchema,
    ) -> Result<Option<(String, Vec<String>)>> {
        if matches!(where_clause.operator, LogicalOperator::Or) && where_clause.conditions.len() > 1
        {
            return Ok(None);
        }
        let indexed = crate::services::record_store::schema_secondary_index_columns(table_schema);
        if indexed.is_empty() {
            return Ok(None);
        }
        let is_indexed = |column: &String| indexed.iter().any(|c| c == column);
        for condition in &where_clause.conditions {
            match condition {
                Condition::Comparison {
                    column,
                    operator,
                    value,
                } if matches!(operator, ComparisonOperator::Equal)
                    && is_indexed(column)
                    && !Self::literal_is_null(value) =>
                {
                    return Ok(Some((
                        column.clone(),
                        vec![self.literal_to_secondary_text(value)?],
                    )));
                }
                Condition::In {
                    column,
                    values,
                    negated,
                } if !*negated && is_indexed(column) && !values.is_empty() => {
                    let mut texts = Vec::with_capacity(values.len());
                    for value in values {
                        if Self::literal_is_null(value) {
                            continue; // NULL never equals anything; skip the term
                        }
                        texts.push(self.literal_to_secondary_text(value)?);
                    }
                    if !texts.is_empty() {
                        return Ok(Some((column.clone(), texts)));
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }

    /// Render a WHERE literal as secondary-index probe text, going through the
    /// same `ProximaValue` → [`proxima_value_to_unique_text`] path the index uses
    /// for stored values so probe text and indexed text are byte-identical. (TD-127.)
    fn literal_to_secondary_text(&self, val: &SqlValueLiteral) -> Result<String> {
        let value = self.literal_to_proxima_value(val)?;
        Ok(crate::services::record_store::proxima_value_to_unique_text(
            &value,
        ))
    }

    /// Whether the table declares a primary key. Without one,
    /// `schema_primary_key_column` falls back to a column named `id` or
    /// `record_id`, which is NOT the record oid for a keyless relational row
    /// (TD-RELIDEQ-1).
    fn table_declares_primary_key(table_schema: &CatalogTableSchema) -> bool {
        !table_schema.primary_key.is_empty()
            || !table_schema.relational_capabilities.primary_key.is_empty()
    }

    fn extract_pk_candidate_ids(
        &self,
        where_clause: &WhereClause,
        table_schema: &CatalogTableSchema,
    ) -> Result<Vec<String>> {
        // Only sound under a top-level conjunction: under OR a PK leaf does not
        // bound the match set (`id = 5 OR status = 'x'` matches any id), so force
        // the full scan path. A PK leaf nested under an inner OR yields no
        // top-level candidate below → also scans.
        if matches!(where_clause.operator, LogicalOperator::Or) && where_clause.conditions.len() > 1
        {
            return Ok(Vec::new());
        }
        // The fast-path builds single-key `get_by_key` candidates, which is only
        // sound for a single-column PK. A composite PK would yield a partial
        // candidate (one column's value) that never matches the encoded oid, so
        // return empty and let `resolve_matching_ids` use the predicate scan
        // (which evaluates the full WHERE, incl. every PK column, from props).
        if Self::primary_key_columns(table_schema).len() != 1 {
            return Ok(Vec::new());
        }
        let Some(primary_key_column) = Self::primary_key_column(table_schema) else {
            return Ok(Vec::new());
        };
        let mut ids = Vec::new();
        for condition in &where_clause.conditions {
            match condition {
                Condition::Comparison {
                    column,
                    operator,
                    value,
                } if column == &primary_key_column
                    && matches!(operator, ComparisonOperator::Equal) =>
                {
                    ids.push(self.literal_to_string(value)?);
                }
                Condition::In {
                    column,
                    values,
                    negated,
                } if column == &primary_key_column && !negated => {
                    for v in values {
                        ids.push(self.literal_to_string(v)?);
                    }
                }
                _ => {}
            }
        }
        Ok(ids)
    }

    /// Convert an UPDATE/DELETE `WhereClause` into a resolved boolean predicate
    /// tree. Columns are resolved against the catalog ONCE here (so per-record
    /// evaluation is infallible); supports AND/OR/nested/BETWEEN/NOT BETWEEN.
    fn where_clause_to_predicate_tree(
        &self,
        where_clause: &WhereClause,
        table_schema: &CatalogTableSchema,
    ) -> Result<RelationalPredicateTree> {
        self.combine_conditions(
            &where_clause.conditions,
            where_clause.operator,
            table_schema,
        )
    }

    /// Combine a list of conditions with the given logical operator.
    #[allow(clippy::expect_used)] // infallible: guarded by len==1 check above
    fn combine_conditions(
        &self,
        conditions: &[Condition],
        operator: LogicalOperator,
        table_schema: &CatalogTableSchema,
    ) -> Result<RelationalPredicateTree> {
        if conditions.is_empty() {
            return Err(anyhow!("WHERE clause has no conditions for UPDATE/DELETE"));
        }
        let mut children = conditions
            .iter()
            .map(|condition| self.condition_to_predicate_tree(condition, table_schema))
            .collect::<Result<Vec<_>>>()?;
        Ok(if children.len() == 1 {
            children.pop().expect("len == 1")
        } else {
            match operator {
                LogicalOperator::And => RelationalPredicateTree::And(children),
                LogicalOperator::Or => RelationalPredicateTree::Or(children),
            }
        })
    }

    /// Lower one `Condition` into a resolved predicate tree node. `BETWEEN`
    /// expands to `>= low AND <= high` (negated → `Not`); `Nested` recurses.
    #[allow(clippy::expect_used)] // infallible: guarded by len/shape check above
    fn condition_to_predicate_tree(
        &self,
        condition: &Condition,
        table_schema: &CatalogTableSchema,
    ) -> Result<RelationalPredicateTree> {
        // Resolve a single column-leaf condition into a tree `Leaf`.
        let leaf = |column: &str,
                    cond: RelationalSelectPredicateCondition|
         -> Result<RelationalPredicateTree> {
            let resolved = Self::resolve_select_predicates(
                table_schema,
                &[RelationalSelectPredicateInput {
                    column_name: column.to_string(),
                    condition: cond,
                }],
            )?;
            Ok(RelationalPredicateTree::Leaf(
                resolved
                    .into_iter()
                    .next()
                    .expect("one input → one predicate"),
            ))
        };
        match condition {
            Condition::Comparison {
                column,
                operator,
                value,
            } => leaf(
                column,
                RelationalSelectPredicateCondition::Comparison {
                    operator: Self::map_comparison_operator(*operator),
                    literal: self.literal_to_string(value)?,
                },
            ),
            Condition::In {
                column,
                values,
                negated,
            } => {
                let literals = values
                    .iter()
                    .map(|value| self.literal_to_string(value))
                    .collect::<Result<Vec<_>>>()?;
                leaf(
                    column,
                    RelationalSelectPredicateCondition::In {
                        literals,
                        negated: *negated,
                    },
                )
            }
            Condition::Like {
                column,
                pattern,
                negated,
            } => leaf(
                column,
                RelationalSelectPredicateCondition::Like {
                    pattern: pattern.clone(),
                    negated: *negated,
                },
            ),
            Condition::IsNull { column, negated } => leaf(
                column,
                RelationalSelectPredicateCondition::IsNull { negated: *negated },
            ),
            Condition::Between {
                column,
                low,
                high,
                negated,
            } => {
                // col BETWEEN low AND high  ≡  col >= low AND col <= high
                let ge = leaf(
                    column,
                    RelationalSelectPredicateCondition::Comparison {
                        operator: RelationalSelectPredicateOperator::GreaterThanOrEqual,
                        literal: self.literal_to_string(low)?,
                    },
                )?;
                let le = leaf(
                    column,
                    RelationalSelectPredicateCondition::Comparison {
                        operator: RelationalSelectPredicateOperator::LessThanOrEqual,
                        literal: self.literal_to_string(high)?,
                    },
                )?;
                let between = RelationalPredicateTree::And(vec![ge, le]);
                Ok(if *negated {
                    RelationalPredicateTree::Not(Box::new(between))
                } else {
                    between
                })
            }
            Condition::Nested {
                conditions,
                operator,
            } => self.combine_conditions(conditions, *operator, table_schema),
        }
    }

    fn map_comparison_operator(operator: ComparisonOperator) -> RelationalSelectPredicateOperator {
        match operator {
            ComparisonOperator::Equal => RelationalSelectPredicateOperator::Equal,
            ComparisonOperator::NotEqual => RelationalSelectPredicateOperator::NotEqual,
            ComparisonOperator::LessThan => RelationalSelectPredicateOperator::LessThan,
            ComparisonOperator::LessThanOrEqual => {
                RelationalSelectPredicateOperator::LessThanOrEqual
            }
            ComparisonOperator::GreaterThan => RelationalSelectPredicateOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqual => {
                RelationalSelectPredicateOperator::GreaterThanOrEqual
            }
        }
    }

    fn effective_insert_literal(
        &self,
        column_name: &str,
        value: &SqlValueLiteral,
        table_schema: &CatalogTableSchema,
    ) -> Result<SqlValueLiteral> {
        if !matches!(value, SqlValueLiteral::Default) {
            return Ok(value.clone());
        }

        let Some(column) = table_schema
            .columns
            .iter()
            .find(|column| column.name == column_name)
        else {
            return Err(anyhow!("Column '{}' does not exist", column_name));
        };
        let Some(default_value) = &column.default_value else {
            return Err(anyhow!("Column '{}' has no DEFAULT value", column_name));
        };

        Self::parse_default_literal(default_value)
    }

    fn parse_default_literal(default_value: &str) -> Result<SqlValueLiteral> {
        let without_cast = default_value
            .split_once("::")
            .map(|(value, _)| value)
            .unwrap_or(default_value)
            .trim();
        let trimmed = without_cast
            .strip_prefix('(')
            .and_then(|value| value.strip_suffix(')'))
            .unwrap_or(without_cast)
            .trim();

        if trimmed.eq_ignore_ascii_case("NULL") {
            return Ok(SqlValueLiteral::Null);
        }
        if trimmed.eq_ignore_ascii_case("TRUE") {
            return Ok(SqlValueLiteral::Boolean(true));
        }
        if trimmed.eq_ignore_ascii_case("FALSE") {
            return Ok(SqlValueLiteral::Boolean(false));
        }
        if trimmed.eq_ignore_ascii_case("NOW()")
            || trimmed.eq_ignore_ascii_case("CURRENT_TIMESTAMP")
            || trimmed.eq_ignore_ascii_case("CURRENT_TIMESTAMP()")
        {
            return Ok(SqlValueLiteral::Function {
                name: "CURRENT_TIMESTAMP".to_string(),
                args: Vec::new(),
            });
        }

        if let Some(unquoted) = Self::unquote_sql_string(trimmed) {
            let value = unquoted?;
            if (value.starts_with('{') || value.starts_with('['))
                && let Ok(json) = serde_json::from_str(&value)
            {
                return Ok(SqlValueLiteral::Json(json));
            }
            return Ok(SqlValueLiteral::String(value));
        }

        if let Ok(value) = trimmed.parse::<i64>() {
            return Ok(SqlValueLiteral::Integer(value));
        }
        if let Ok(value) = trimmed.parse::<f64>() {
            return Ok(SqlValueLiteral::Float(value));
        }
        if (trimmed.starts_with('{') || trimmed.starts_with('['))
            && let Ok(json) = serde_json::from_str(trimmed)
        {
            return Ok(SqlValueLiteral::Json(json));
        }

        Ok(SqlValueLiteral::String(trimmed.to_string()))
    }

    fn unquote_sql_string(value: &str) -> Option<Result<String>> {
        if !(value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'')) {
            return None;
        }

        let mut output = String::new();
        let mut chars = value[1..value.len() - 1].chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\'' {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    output.push('\'');
                } else {
                    return Some(Err(anyhow!("Invalid SQL string literal: {}", value)));
                }
            } else {
                output.push(ch);
            }
        }
        Some(Ok(output))
    }

    /// Convert SqlValueLiteral to string
    fn literal_to_string(&self, val: &SqlValueLiteral) -> Result<String> {
        match val {
            SqlValueLiteral::String(s) => Ok(s.clone()),
            SqlValueLiteral::Integer(i) => Ok(i.to_string()),
            SqlValueLiteral::Float(f) => Ok(f.to_string()),
            SqlValueLiteral::Boolean(b) => Ok(b.to_string()),
            SqlValueLiteral::Null => Err(anyhow!("Cannot convert NULL to string")),
            _ => Err(anyhow!("Unsupported value type for string conversion")),
        }
    }

    /// Convert SqlValueLiteral to vector — the ONE non-finite policy:
    /// every query boundary rejects inf/NaN literals, so persisting one
    /// here writes a poison row no query can ever express.
    fn literal_to_vector(&self, val: &SqlValueLiteral) -> Result<Vec<f32>> {
        let ensure_finite = |f: f32| {
            if f.is_finite() {
                Ok(f)
            } else {
                Err(anyhow!("vector elements must be finite"))
            }
        };
        match val {
            SqlValueLiteral::Array(arr) => {
                // NOTE: empty vectors are permitted on the WRITE path (a
                // schema DEFAULT '[]' is legal config; every QUERY boundary
                // rejects empty — the asymmetry is deliberate, develop-
                // compatible, and avoids breaking whole tables).
                arr.iter()
                    .map(|v| match v {
                        SqlValueLiteral::Float(f) => ensure_finite(*f as f32),
                        SqlValueLiteral::Integer(i) => ensure_finite(*i as f32),
                        _ => Err(anyhow!("Vector elements must be numeric")),
                    })
                    .collect()
            }
            SqlValueLiteral::String(value) => {
                // Lazy iterator (an intermediate Vec cost ~36KB per
                // 1536-dim row on the INSERT coercion path); emptiness is
                // checked on the RESULT.
                value
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| {
                        let f = part
                            .trim()
                            .parse::<f32>()
                            .map_err(|e| anyhow!("Invalid vector element '{}': {}", part, e))?;
                        ensure_finite(f)
                    })
                    .collect()
            }
            _ => Err(anyhow!("Vector column expects array value")),
        }
    }

    /// Convert SqlValueLiteral to timestamp
    fn literal_to_timestamp(&self, val: &SqlValueLiteral) -> Result<Option<i64>> {
        match val {
            SqlValueLiteral::Null => Ok(None),
            SqlValueLiteral::Integer(i) => Ok(Some(*i)),
            SqlValueLiteral::String(s) => {
                // Accept RFC3339 (with TZ) first, then standard SQL naive timestamp
                // spellings (`YYYY-MM-DD HH:MM:SS[.fff]` or with a `T` separator,
                // treated as UTC) and a bare date. `TIMESTAMP '2016-01-01 00:00:00'`
                // literals (no TZ) are standard SQL and dense in time-series data.
                use chrono::{DateTime, NaiveDate, NaiveDateTime};
                let s = s.trim();
                if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                    return Ok(Some(dt.timestamp_millis()));
                }
                for fmt in [
                    "%Y-%m-%d %H:%M:%S%.f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S%.f",
                    "%Y-%m-%dT%H:%M:%S",
                ] {
                    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
                        return Ok(Some(ndt.and_utc().timestamp_millis()));
                    }
                }
                if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                    return Ok(Some(
                        d.and_hms_opt(0, 0, 0)
                            .ok_or_else(|| anyhow!("internal: bad midnight"))?
                            .and_utc()
                            .timestamp_millis(),
                    ));
                }
                Err(anyhow!("Invalid timestamp format: {s}"))
            }
            SqlValueLiteral::Function { name, .. } if name.eq_ignore_ascii_case("NOW") => {
                Ok(Some(chrono::Utc::now().timestamp_millis()))
            }
            _ => Err(anyhow!("Invalid timestamp value")),
        }
    }

    fn literal_to_proxima_value_for_column(
        &self,
        column_name: &str,
        val: &SqlValueLiteral,
        table_schema: &CatalogTableSchema,
    ) -> Result<ProximaValue> {
        let Some(column) = table_schema
            .columns
            .iter()
            .find(|column| column.name == column_name)
        else {
            return self.literal_to_proxima_value(val);
        };

        match &column.data_type {
            ProximaType::Boolean => match val {
                SqlValueLiteral::Boolean(value) => Ok(ProximaValue::Boolean(*value)),
                SqlValueLiteral::Null if column.nullable => Ok(ProximaValue::Null),
                _ => Err(anyhow!("Column '{}' expects boolean", column_name)),
            },
            ProximaType::Int8 => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_i64(val)
                    .map(|v| ProximaValue::Int8(v as i8))
            }
            ProximaType::Int16 => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_i64(val)
                    .map(|v| ProximaValue::Int16(v as i16))
            }
            ProximaType::Int32 => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_i64(val)
                    .map(|v| ProximaValue::Int32(v as i32))
            }
            ProximaType::Int64 => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_i64(val).map(ProximaValue::Int64)
            }
            ProximaType::Float32 => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_f64(val)
                    .map(|v| ProximaValue::Float32(v as f32))
            }
            ProximaType::Float64 => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_f64(val).map(ProximaValue::Float64)
            }
            ProximaType::String | ProximaType::Uuid => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_string(val).map(ProximaValue::String)
            }
            ProximaType::Json => {
                let json = match val {
                    SqlValueLiteral::Json(value) => value.clone(),
                    SqlValueLiteral::String(value) => serde_json::from_str(value).map_err(|e| {
                        anyhow!("Column '{}' expects valid JSON/JSONB: {}", column_name, e)
                    })?,
                    SqlValueLiteral::Null if column.nullable => serde_json::Value::Null,
                    _ => self.literal_to_json(val)?,
                };
                if column.properties.get("json_encoding").map(String::as_str) == Some("jsonb") {
                    Ok(ProximaValue::Jsonb(json))
                } else {
                    Ok(ProximaValue::Json(json))
                }
            }
            ProximaType::DenseVector { .. } => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    Ok(ProximaValue::Null)
                } else {
                    self.literal_to_vector(val).map(ProximaValue::DenseVector)
                }
            }
            ProximaType::Binary | ProximaType::BinaryVector { .. } => match val {
                SqlValueLiteral::Binary(value) => Ok(ProximaValue::Binary(value.clone())),
                SqlValueLiteral::Null if column.nullable => Ok(ProximaValue::Null),
                _ => Err(anyhow!("Column '{}' expects binary", column_name)),
            },
            ProximaType::Date => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_date_days(val).map(ProximaValue::Date)
            }
            ProximaType::Time(_) => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                self.literal_to_i64(val).map(|value| {
                    ProximaValue::Time(value, proximadb_data_model::TimeUnit::Millisecond)
                })
            }
            ProximaType::Timestamp(_) => self.literal_to_timestamp(val).map(|value| {
                value
                    .map(|timestamp| {
                        ProximaValue::Timestamp(
                            timestamp,
                            proximadb_data_model::TimeUnit::Millisecond,
                        )
                    })
                    .unwrap_or(ProximaValue::Null)
            }),
            ProximaType::TimestampTz(_) => self.literal_to_timestamp(val).map(|value| {
                value
                    .map(|timestamp| {
                        ProximaValue::TimestampTz(
                            timestamp,
                            proximadb_data_model::TimeUnit::Millisecond,
                        )
                    })
                    .unwrap_or(ProximaValue::Null)
            }),
            ProximaType::Decimal { precision, scale } => {
                if matches!(val, SqlValueLiteral::Null) && column.nullable {
                    return Ok(ProximaValue::Null);
                }
                // Store the canonical text at the declared scale (`12.5` →
                // `12.50` for DECIMAL(10,2)) so every read path — pgwire rows,
                // Parquet `Decimal128`, Iceberg — agrees, and reject a value
                // that is not a number or overflows the declared precision.
                // Numeric literals reach here as f64 (exact to 15 significant
                // digits; TD-SEAM-1 residual for wider decimals).
                let text = self.literal_to_string(val)?;
                proximadb_storage_common::proxima_arrow::canonical_decimal_text(
                    &text, *precision, *scale,
                )
                .map(ProximaValue::Decimal)
                .ok_or_else(|| {
                    anyhow!(
                        "value '{text}' for column '{column_name}' is not a valid \
                         DECIMAL({precision},{scale})"
                    )
                })
            }
            ProximaType::SparseVector { .. } => Err(anyhow!(
                "Sparse vector DML literal lowering is not implemented for column '{}'",
                column_name
            )),
            // Richer ProximaType variants (unsigned ints, Float16, Symbol,
            // Jsonb, Array, Map, Struct, Interval, Duration, ULID, geo, Null)
            // are not produced by the SQL/catalog type surface; reject DML
            // literal lowering for them rather than silently coercing.
            other => Err(anyhow!(
                "Column '{}' has type {:?} which is not supported for DML literal lowering",
                column_name,
                other
            )),
        }
    }

    fn literal_to_proxima_value(&self, val: &SqlValueLiteral) -> Result<ProximaValue> {
        match val {
            SqlValueLiteral::Null => Ok(ProximaValue::Null),
            SqlValueLiteral::Boolean(value) => Ok(ProximaValue::Boolean(*value)),
            SqlValueLiteral::Integer(value) => Ok(ProximaValue::Int64(*value)),
            SqlValueLiteral::Float(value) => Ok(ProximaValue::Float64(*value)),
            SqlValueLiteral::String(value) => Ok(ProximaValue::String(value.clone())),
            SqlValueLiteral::Binary(value) => Ok(ProximaValue::Binary(value.clone())),
            SqlValueLiteral::Json(value) => Ok(ProximaValue::Json(value.clone())),
            SqlValueLiteral::Array(values) => values
                .iter()
                .map(|value| self.literal_to_proxima_value(value))
                .collect::<Result<Vec<_>>>()
                .map(ProximaValue::Array),
            SqlValueLiteral::Function { name, .. }
                if name.eq_ignore_ascii_case("NOW")
                    || name.eq_ignore_ascii_case("CURRENT_TIMESTAMP") =>
            {
                Ok(ProximaValue::TimestampTz(
                    chrono::Utc::now().timestamp_millis(),
                    proximadb_data_model::TimeUnit::Millisecond,
                ))
            }
            SqlValueLiteral::Default => Err(anyhow!("DEFAULT value is not supported yet")),
            SqlValueLiteral::Parameter(_) => {
                Err(anyhow!("Unbound parameter in literal conversion"))
            }
            SqlValueLiteral::Column(_) => {
                Err(anyhow!("Column reference not supported in value context"))
            }
            SqlValueLiteral::Function { name, .. } => Err(anyhow!("Unsupported function: {name}")),
        }
    }

    fn literal_to_i64(&self, val: &SqlValueLiteral) -> Result<i64> {
        match val {
            SqlValueLiteral::Integer(value) => Ok(*value),
            SqlValueLiteral::String(value) => value
                .parse()
                .map_err(|e| anyhow!("Invalid integer literal '{}': {}", value, e)),
            SqlValueLiteral::Null => Err(anyhow!("Cannot convert NULL to integer")),
            _ => Err(anyhow!("Expected integer literal")),
        }
    }

    /// Coerce a DML literal to a DATE stored as days since the Unix epoch.
    /// Accepts an integer (already a day count) or an ISO-8601 `YYYY-MM-DD`
    /// string (the form `DATE '1995-02-15'` literals lower to). Parsing to a
    /// real day count — rather than stuffing the string in — keeps the column a
    /// true Date32 through materialization, so DataFusion compares it correctly
    /// against `DATE '...'` predicates.
    fn literal_to_date_days(&self, val: &SqlValueLiteral) -> Result<i32> {
        match val {
            SqlValueLiteral::Integer(value) => Ok(*value as i32),
            SqlValueLiteral::String(value) => {
                let date = chrono::NaiveDate::parse_from_str(value.trim(), "%Y-%m-%d")
                    .map_err(|e| anyhow!("Invalid DATE literal '{}': {}", value, e))?;
                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                    .ok_or_else(|| anyhow!("internal: bad epoch date"))?;
                Ok(date.signed_duration_since(epoch).num_days() as i32)
            }
            SqlValueLiteral::Null => Err(anyhow!("Cannot convert NULL to date")),
            _ => Err(anyhow!(
                "Expected date literal (integer days or 'YYYY-MM-DD')"
            )),
        }
    }

    fn literal_to_f64(&self, val: &SqlValueLiteral) -> Result<f64> {
        match val {
            SqlValueLiteral::Float(value) => Ok(*value),
            SqlValueLiteral::Integer(value) => Ok(*value as f64),
            SqlValueLiteral::String(value) => value
                .parse()
                .map_err(|e| anyhow!("Invalid float literal '{}': {}", value, e)),
            SqlValueLiteral::Null => Err(anyhow!("Cannot convert NULL to float")),
            _ => Err(anyhow!("Expected numeric literal")),
        }
    }

    /// Convert SqlValueLiteral to JSON value
    fn literal_to_json(&self, val: &SqlValueLiteral) -> Result<serde_json::Value> {
        // Allow recursive calls - this is intentional for array processing
        let _ = val; // Suppress unused warning while implementation is pending
        match val {
            SqlValueLiteral::Null => Ok(serde_json::Value::Null),
            SqlValueLiteral::Boolean(b) => Ok(serde_json::Value::Bool(*b)),
            SqlValueLiteral::Integer(i) => Ok(serde_json::Value::Number((*i).into())),
            SqlValueLiteral::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| anyhow!("Invalid float value")),
            SqlValueLiteral::String(s) => Ok(serde_json::Value::String(s.clone())),
            SqlValueLiteral::Json(j) => Ok(j.clone()),
            SqlValueLiteral::Array(arr) => {
                let json_arr: Result<Vec<_>> =
                    arr.iter().map(|v| self.literal_to_json(v)).collect();
                Ok(serde_json::Value::Array(json_arr?))
            }
            _ => Err(anyhow!("Cannot convert to JSON")),
        }
    }
}

/// Adapts [`DmlService`] + an object-store bridge into the DDL-layer
/// [`TableMaterializer`](crate::services::ddl::TableMaterializer) capability, so
/// `ALTER TABLE … MATERIALIZE` can run without `DdlService` depending on the DML
/// service or object storage directly. Boot wires one of these (with the configured
/// warehouse store) into the `DdlService`.
pub struct DmlTableMaterializer {
    dml: Arc<DmlService>,
    bridge: Arc<dyn ObjectStoreBridge>,
    warehouse_root_url: String,
}

impl DmlTableMaterializer {
    /// Build a materializer over `dml`, writing snapshots into `bridge` and publishing
    /// catalog locations rooted at `warehouse_root_url` (the URL the OLAP reader reopens
    /// the same store from).
    pub fn new(
        dml: Arc<DmlService>,
        bridge: Arc<dyn ObjectStoreBridge>,
        warehouse_root_url: impl Into<String>,
    ) -> Self {
        Self {
            dml,
            bridge,
            warehouse_root_url: warehouse_root_url.into(),
        }
    }
}

#[async_trait::async_trait]
impl crate::services::ddl::TableMaterializer for DmlTableMaterializer {
    async fn materialize(
        &self,
        table_name: &str,
        tenant: Option<&str>,
        fence: &dyn crate::services::ddl::PublishFence,
    ) -> Result<String> {
        // Thread the request/connection tenant (TD-113) so the snapshot is scoped
        // and the object prefix is tenant-isolated instead of landing under the
        // DEFAULT_TENANT_PLACEHOLDER. Empty tenant ⇒ None (single-tenant/embedded).
        let tenant_ctx = tenant
            .filter(|t| !t.is_empty())
            .map(TenantContext::for_tenant_id);
        self.dml
            .materialize_table_to_parquet_fenced(
                &*self.bridge,
                &self.warehouse_root_url,
                table_name,
                tenant_ctx.as_ref(),
                fence,
            )
            .await
    }
}

mod materialize_stream;

#[cfg(test)]
mod tests;
