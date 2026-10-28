//! Logical table-to-table write planning and compute routing.
//!
//! These types are intentionally independent of DataFusion, Polars, DuckDB, or
//! any concrete storage writer. SQL surfaces lower `INSERT ... SELECT`,
//! `INSERT OVERWRITE ... SELECT`, CTAS, and MERGE into this shape. A later
//! execution layer can lower the read side into a compute backend and the write
//! side into `TableRecordStore` or an open-table commit protocol.

use proximadb_catalog::{
    CatalogAuthorityMode, CatalogPhysicalFormat, CatalogStorageSpecialization, CatalogTableSchema,
    CatalogTableStatistics, CatalogWorkloadProfile, ColumnConstraint, ProjectionFreshnessState,
};
use serde::Serialize;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};

use crate::services::{
    ProjectionFreshnessRequirement, RejectedWriteLane, WriteDurabilityRequirement, WriteGuard,
    WriteIntent, WriteIsolationRequirement, WriteLaneDecision, WriteLaneRouter, WriteOperationKind,
};

/// Logical table identifier used by write plans before catalog resolution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LogicalTableRef {
    /// Optional namespace/schema path.
    pub namespace: Vec<String>,
    /// Table name.
    pub name: String,
}

impl LogicalTableRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            namespace: Vec::new(),
            name: name.into(),
        }
    }

    pub fn qualified_name(&self) -> String {
        if self.namespace.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.namespace.join("."), self.name)
        }
    }
}

/// Snapshot, branch, or freshness reference pinned for a read.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum SnapshotRef {
    #[default]
    Latest,
    SnapshotId(String),
    Branch(String),
    TimestampNs(i64),
}

/// External/open format referenced by a read or write plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFormatRef {
    pub format: CatalogPhysicalFormat,
    pub location: String,
}

/// Source side of a copy/write plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadSource {
    CatalogTable {
        table: LogicalTableRef,
        snapshot: SnapshotRef,
    },
    ExternalLocation {
        format: ExternalFormatRef,
        schema_hint: Option<String>,
    },
    QuerySql(String),
    ArrowFlightTicket(String),
}

/// Target write behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteMode {
    Append,
    InsertOnly,
    Upsert,
    OverwriteTable,
    ReplacePartitions(Vec<String>),
    Merge,
}

/// Conflict behavior for key collisions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ConflictPolicy {
    #[default]
    Error,
    Ignore,
    Upsert,
    Merge,
}

/// Distribution preference requested by the SQL/frontend planner.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DistributionMode {
    #[default]
    Auto,
    LocalOnly,
    PseudoDistributed,
    Distributed,
}

/// Logical copy/write plan for INSERT SELECT, INSERT OVERWRITE, CTAS, and MERGE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyIntoPlan {
    pub source: ReadSource,
    pub target: LogicalTableRef,
    pub write_mode: WriteMode,
    pub conflict_policy: ConflictPolicy,
    pub distribution: DistributionMode,
}

impl CopyIntoPlan {
    pub fn insert_select(target: LogicalTableRef, source_sql: impl Into<String>) -> Self {
        Self {
            source: ReadSource::QuerySql(source_sql.into()),
            target,
            write_mode: WriteMode::Append,
            conflict_policy: ConflictPolicy::Error,
            distribution: DistributionMode::Auto,
        }
    }

    pub fn insert_overwrite(target: LogicalTableRef, source_sql: impl Into<String>) -> Self {
        Self {
            source: ReadSource::QuerySql(source_sql.into()),
            target,
            write_mode: WriteMode::OverwriteTable,
            conflict_policy: ConflictPolicy::Upsert,
            distribution: DistributionMode::Auto,
        }
    }
}

/// Compute/write backend selected for a copy plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeBackend {
    Native,
    DataFusionLocal,
    DataFusionDistributed,
    PolarsLocal,
    DuckDbCompat,
    ExternalDelegated(String),
}

/// Physical/access-method family represented by a candidate path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessMethodFamily {
    NativeRecord,
    Pax,
    Lsm,
    Columnar,
    VectorAnn,
    DocumentJson,
    GraphTopology,
    ObservabilityTimeSeries,
    ExternalOpenTable,
}

/// Pushdown capabilities advertised by a candidate path.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PushdownCapabilities {
    pub projection: bool,
    pub filter: bool,
    pub aggregate: bool,
    pub join: bool,
    pub limit: bool,
    pub vector_topk: bool,
    pub graph_pattern: bool,
    pub json_path: bool,
    pub requires_proxima_recheck: bool,
}

/// Access-method cost hints. Initial values are static; xCatalog statistics and
/// engine feedback should populate them over time.
#[derive(Debug, Clone, PartialEq)]
pub struct AccessMethodCostHints {
    pub row_lookup_cost: f64,
    pub scan_setup_cost: f64,
    pub sequential_scan_cost_per_mb: f64,
    pub remote_read_cost_per_mb: f64,
    pub write_amplification: f64,
    pub compaction_debt: f64,
    pub projection_lag_penalty: f64,
}

impl Default for AccessMethodCostHints {
    fn default() -> Self {
        Self {
            row_lookup_cost: 1.0,
            scan_setup_cost: 1.0,
            sequential_scan_cost_per_mb: 1.0,
            remote_read_cost_per_mb: 4.0,
            write_amplification: 1.0,
            compaction_debt: 0.0,
            projection_lag_penalty: 0.0,
        }
    }
}

/// Capability/semantic guard that must be enforced before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionGuard {
    PinSourceSnapshot,
    CheckTargetWriteCapabilities,
    EnforceRlsInProxima,
    RequireExternalAtomicCommit,
    RequireIdempotencyKey,
    PreservePreviousSnapshot,
}

/// Backwards-compat alias for [`TableWriteCostEstimate`].
pub type CostEstimate = TableWriteCostEstimate;

/// Lightweight cost estimate. The first router is rule-based, but it returns
/// comparable cost fields so xCatalog statistics can drive CBO later.
#[derive(Debug, Clone, PartialEq)]
pub struct TableWriteCostEstimate {
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
    pub relative_cost: f64,
    pub reason: String,
}

impl TableWriteCostEstimate {
    fn new(relative_cost: f64, reason: impl Into<String>) -> Self {
        Self {
            rows: None,
            bytes: None,
            relative_cost,
            reason: reason.into(),
        }
    }
}

/// Lightweight candidate path, inspired by PostgreSQL's path-vs-plan split.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateWritePath {
    pub backend: ComputeBackend,
    pub access_method: AccessMethodFamily,
    pub pushdown: PushdownCapabilities,
    pub cost_hints: AccessMethodCostHints,
    pub estimated_cost: TableWriteCostEstimate,
    pub guards: Vec<ExecutionGuard>,
}

/// Candidate backend that was considered but rejected before cost selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedCandidatePath {
    pub backend: ComputeBackend,
    pub access_method: AccessMethodFamily,
    pub reason: String,
    pub required_guards: Vec<ExecutionGuard>,
}

/// Explainable routing inputs derived from xCatalog metadata and table properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecisionMetadata {
    pub authority_mode: CatalogAuthorityMode,
    pub workload_profile: CatalogWorkloadProfile,
    pub storage_specialization: CatalogStorageSpecialization,
    pub primary_format: Option<CatalogPhysicalFormat>,
    pub preferred_compute_route: Option<String>,
    pub partitioning: Option<String>,
    pub isolation_profile: Option<String>,
    pub freshness_sla: Option<String>,
    pub projection_freshness_state: Option<ProjectionFreshnessState>,
    pub projection_metadata: Vec<ProjectionRouteMetadata>,
    pub policy_boundary: String,
    pub constraint_enforcement: String,
    pub constraint_gaps: Vec<String>,
}

/// Explainable projection/publication freshness and repair metadata from xCatalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionRouteMetadata {
    pub name: String,
    pub kind: String,
    pub physical_format: String,
    pub rebuild_source: String,
    pub freshness: String,
    pub freshness_state: ProjectionFreshnessState,
    pub max_lag_ms: Option<i64>,
    pub source_range: Option<String>,
    pub last_included_position: Option<String>,
    pub rebuildable: bool,
    pub invalidation_policy: Option<String>,
    pub policy_boundary: Option<String>,
    pub lossy: bool,
    pub support_status: String,
    pub benchmark_gate: Option<String>,
}

/// Stable, API-facing route metadata for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableWriteRouteMetadataExplanation {
    pub authority_mode: String,
    pub workload_profile: String,
    pub storage_specialization: String,
    pub primary_format: Option<String>,
    pub preferred_compute_route: Option<String>,
    pub partitioning: Option<String>,
    pub isolation_profile: Option<String>,
    pub freshness_sla: Option<String>,
    pub projection_freshness_state: Option<String>,
    pub projection_metadata: Vec<ProjectionRouteMetadataExplanation>,
    pub policy_boundary: String,
    pub constraint_enforcement: String,
    pub constraint_gaps: Vec<String>,
}

/// Stable, API-facing projection/publication metadata for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProjectionRouteMetadataExplanation {
    pub name: String,
    pub kind: String,
    pub physical_format: String,
    pub rebuild_source: String,
    pub freshness: String,
    pub freshness_state: String,
    pub max_lag_ms: Option<i64>,
    pub source_range: Option<String>,
    pub last_included_position: Option<String>,
    pub rebuildable: bool,
    pub invalidation_policy: Option<String>,
    pub policy_boundary: Option<String>,
    pub lossy: bool,
    pub support_status: String,
    pub benchmark_gate: Option<String>,
}

/// Stable, API-facing cost estimate for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableWriteCostExplanation {
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
    pub relative_cost: f64,
    pub reason: String,
}

/// Stable, API-facing data-movement estimate for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TableWriteDataMovementExplanation {
    pub source_rows: Option<u64>,
    pub source_bytes: Option<u64>,
    pub target_rows_before_write: Option<u64>,
    pub target_bytes_before_write: Option<u64>,
    pub estimated_read_bytes: Option<u64>,
    pub estimated_write_bytes: Option<u64>,
    pub estimated_rewrite_bytes: Option<u64>,
    pub estimate_source: String,
    pub source_last_analyzed_ms: Option<i64>,
    pub target_last_analyzed_ms: Option<i64>,
    pub source_stats_age_ms: Option<u64>,
    pub target_stats_age_ms: Option<u64>,
    pub freshness_sla_ms: Option<u64>,
    pub stats_freshness: String,
}

/// Stable, API-facing candidate path for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableWriteCandidateExplanation {
    pub backend: String,
    pub access_method: String,
    pub estimated_cost: TableWriteCostExplanation,
    pub required_guards: Vec<String>,
    pub pushdown: Vec<String>,
}

/// Stable, API-facing rejected path for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TableWriteRejectedPathExplanation {
    pub backend: String,
    pub access_method: String,
    pub reason: String,
    pub required_guards: Vec<String>,
}

/// Stable, API-facing rejected write lane for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TableWriteRejectedLaneExplanation {
    pub lane: String,
    pub reason: String,
}

/// Stable, API-facing write intent summary for table-write EXPLAIN output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TableWriteIntentExplanation {
    pub target_table: String,
    pub operation_kind: String,
    pub durability: String,
    pub isolation: String,
    pub projection_freshness: String,
    pub tenant_id: Option<String>,
    pub actor: Option<String>,
    pub idempotency_key: Option<String>,
    pub catalog_schema_version: Option<u64>,
    pub row_count_hint: Option<u64>,
    pub estimated_bytes: Option<u64>,
    pub requires_row_level_semantics: bool,
    pub batch_local_constraints_sufficient: bool,
}

/// Stable, API-facing route explanation for table-write planning.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableWriteRouteExplanation {
    pub target_table: String,
    pub source: String,
    pub write_mode: String,
    pub distribution: String,
    pub write_intent: TableWriteIntentExplanation,
    pub write_lane: String,
    pub write_lane_reason: String,
    pub write_lane_required_guards: Vec<String>,
    pub rejected_write_lanes: Vec<TableWriteRejectedLaneExplanation>,
    pub selected_backend: String,
    pub selected_access_method: String,
    pub estimated_cost: TableWriteCostExplanation,
    pub data_movement: TableWriteDataMovementExplanation,
    pub required_guards: Vec<String>,
    pub route_metadata: TableWriteRouteMetadataExplanation,
    pub candidate_paths: Vec<TableWriteCandidateExplanation>,
    pub rejected_paths: Vec<TableWriteRejectedPathExplanation>,
    /// Set only for EXPLAIN ANALYZE: wall-clock time of the actual write in microseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_elapsed_us: Option<u64>,
    /// Set only for EXPLAIN ANALYZE: actual rows written by the executed statement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_rows_written: Option<u64>,
}

/// Result of routing a logical copy/write plan.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedExecutionPlan {
    pub backend: ComputeBackend,
    pub plan: CopyIntoPlan,
    pub write_intent: WriteIntent,
    pub write_lane_decision: WriteLaneDecision,
    pub estimated_cost: TableWriteCostEstimate,
    pub required_guards: Vec<ExecutionGuard>,
    pub selected_path: CandidateWritePath,
    pub candidate_paths: Vec<CandidateWritePath>,
    pub rejected_paths: Vec<RejectedCandidatePath>,
    pub route_metadata: RouteDecisionMetadata,
    pub data_movement: TableWriteDataMovementExplanation,
}

impl RoutedExecutionPlan {
    /// Convert the internal router result into a stable EXPLAIN/debug payload.
    pub fn route_explanation(&self) -> TableWriteRouteExplanation {
        TableWriteRouteExplanation {
            target_table: self.plan.target.qualified_name(),
            source: read_source_name(&self.plan.source),
            write_mode: format!("{:?}", self.plan.write_mode),
            distribution: format!("{:?}", self.plan.distribution),
            write_intent: write_intent_explanation(&self.write_intent),
            write_lane: format!("{:?}", self.write_lane_decision.lane),
            write_lane_reason: self.write_lane_decision.reason.clone(),
            write_lane_required_guards: write_guard_names(
                &self.write_lane_decision.required_guards,
            ),
            rejected_write_lanes: self
                .write_lane_decision
                .rejected_lanes
                .iter()
                .map(rejected_write_lane_explanation)
                .collect(),
            selected_backend: backend_name(&self.backend),
            selected_access_method: access_method_name(&self.selected_path.access_method),
            estimated_cost: cost_explanation(&self.estimated_cost),
            data_movement: self.data_movement.clone(),
            required_guards: guard_names(&self.required_guards),
            route_metadata: route_metadata_explanation(&self.route_metadata),
            candidate_paths: self
                .candidate_paths
                .iter()
                .map(candidate_explanation)
                .collect(),
            rejected_paths: self
                .rejected_paths
                .iter()
                .map(rejected_path_explanation)
                .collect(),
            execution_elapsed_us: None,
            execution_rows_written: None,
        }
    }
}

/// Inputs available to the first routing pass.
#[derive(Debug, Clone)]
pub struct RoutingContext<'a> {
    pub target_schema: &'a CatalogTableSchema,
    pub target_stats: Option<&'a CatalogTableStatistics>,
    pub source_schema: Option<&'a CatalogTableSchema>,
    pub source_stats: Option<&'a CatalogTableStatistics>,
    pub write_intent_overrides: Option<&'a WriteIntentOverrides>,
    pub plan: &'a CopyIntoPlan,
}

/// Protocol-provided write-intent hints applied after catalog inference.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteIntentOverrides {
    pub tenant_id: Option<String>,
    pub actor: Option<String>,
    pub idempotency_key: Option<String>,
    pub row_count_hint: Option<u64>,
    pub estimated_bytes: Option<u64>,
    pub requires_row_level_semantics: Option<bool>,
    pub batch_local_constraints_sufficient: Option<bool>,
}

/// Catalog-resolved request passed from DML analysis into routing.
///
/// The SQL parser owns syntax only. This planner boundary owns catalog-aware
/// validation and route selection before a later executor lowers the plan into
/// DataFusion/native/open-table writes.
#[derive(Debug, Clone)]
pub struct DmlWritePlanRequest<'a> {
    pub target_schema: &'a CatalogTableSchema,
    /// The table's published parent snapshot carries deletion vectors — the
    /// bulk-append lane must refuse (see write_intent.rs).
    pub published_parent_has_deletes: bool,
    pub target_stats: Option<&'a CatalogTableStatistics>,
    pub source_schema: Option<&'a CatalogTableSchema>,
    pub source_stats: Option<&'a CatalogTableStatistics>,
    pub write_intent_overrides: Option<&'a WriteIntentOverrides>,
    pub plan: &'a CopyIntoPlan,
    pub target_columns: &'a [String],
}

/// Catalog-aware DML planner for table-to-table writes.
#[derive(Debug, Clone, Default)]
pub struct DmlWritePlanner {
    router: TableWriteRouter,
}

impl DmlWritePlanner {
    pub fn new(router: TableWriteRouter) -> Self {
        Self { router }
    }

    pub fn plan(&self, request: DmlWritePlanRequest<'_>) -> Result<RoutedExecutionPlan> {
        validate_target_table(request.target_schema, request.plan)?;
        validate_target_columns(request.target_schema, request.target_columns)?;

        Ok(self.router.route_fenced(
            RoutingContext {
                target_schema: request.target_schema,
                target_stats: request.target_stats,
                source_schema: request.source_schema,
                source_stats: request.source_stats,
                write_intent_overrides: request.write_intent_overrides,
                plan: request.plan,
            },
            request.published_parent_has_deletes,
        ))
    }
}

/// Rule-based router that encodes the architecture while leaving room for CBO.
#[derive(Debug, Clone, Default)]
pub struct TableWriteRouter;

impl TableWriteRouter {
    pub fn route(&self, context: RoutingContext<'_>) -> RoutedExecutionPlan {
        self.route_fenced(context, false)
    }

    /// Route with the fence-eligibility fact the DML service computed from
    /// the published state: when the published parent snapshot carries
    /// deletion vectors, the bulk-append lane must refuse (the visibility
    /// fence contract's FORBID — the intent fact named
    /// `published_parent_has_deletes`). `route` keeps the plain signature so
    /// the dozens of literal RoutingContext test constructions stay valid.
    pub fn route_fenced(
        &self,
        context: RoutingContext<'_>,
        parent_has_deletes: bool,
    ) -> RoutedExecutionPlan {
        let mut candidate_paths = self.candidate_paths(&context);
        let rejected_paths = self.rejected_paths(&context);
        candidate_paths.sort_by(|left, right| {
            left.estimated_cost
                .relative_cost
                .partial_cmp(&right.estimated_cost.relative_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected_path = candidate_paths
            .first()
            .cloned()
            .unwrap_or_else(|| self.native_candidate(&context));
        let write_intent = self.write_intent(&context, parent_has_deletes);
        let write_lane_decision = WriteLaneRouter::new().route(&write_intent);

        RoutedExecutionPlan {
            backend: selected_path.backend.clone(),
            plan: context.plan.clone(),
            write_intent,
            write_lane_decision,
            estimated_cost: selected_path.estimated_cost.clone(),
            required_guards: selected_path.guards.clone(),
            selected_path,
            candidate_paths,
            rejected_paths,
            route_metadata: route_metadata_for_schema(context.target_schema),
            data_movement: data_movement_for_context(&context),
        }
    }

    pub fn candidate_paths(&self, context: &RoutingContext<'_>) -> Vec<CandidateWritePath> {
        let mut paths = vec![self.native_candidate(context)];
        if self.datafusion_applicable(context) {
            paths.push(self.datafusion_candidate(context));
        }
        if self.external_commit_applicable(context) {
            paths.push(self.external_candidate(context));
        }
        paths
    }

    pub fn rejected_paths(&self, context: &RoutingContext<'_>) -> Vec<RejectedCandidatePath> {
        let mut rejected = Vec::new();

        if !self.datafusion_applicable(context) {
            rejected.push(RejectedCandidatePath {
                backend: ComputeBackend::DataFusionLocal,
                access_method: access_method_for_schema(context.target_schema),
                reason: datafusion_rejection_reason(context),
                required_guards: self.required_guards(context),
            });
        }

        if !self.external_commit_applicable(context) {
            rejected.push(RejectedCandidatePath {
                backend: ComputeBackend::ExternalDelegated("open_table".to_string()),
                access_method: AccessMethodFamily::ExternalOpenTable,
                reason: external_rejection_reason(context.target_schema),
                required_guards: vec![ExecutionGuard::RequireExternalAtomicCommit],
            });
        }

        if policy_boundary_for_schema(context.target_schema) == "unsupported" {
            rejected.push(RejectedCandidatePath {
                backend: ComputeBackend::ExternalDelegated("policy_boundary".to_string()),
                access_method: AccessMethodFamily::ExternalOpenTable,
                reason: "Policy boundary is unsupported; delegated/external routes are unsafe"
                    .to_string(),
                required_guards: vec![ExecutionGuard::EnforceRlsInProxima],
            });
        }

        rejected
    }

    fn native_candidate(&self, context: &RoutingContext<'_>) -> CandidateWritePath {
        let backend = ComputeBackend::Native;
        CandidateWritePath {
            backend: backend.clone(),
            access_method: access_method_for_schema(context.target_schema),
            pushdown: native_pushdown_for_schema(context.target_schema),
            cost_hints: cost_hints_for_schema(context.target_schema),
            estimated_cost: self.estimate_cost(context, &backend),
            guards: self.required_guards(context),
        }
    }

    fn datafusion_candidate(&self, context: &RoutingContext<'_>) -> CandidateWritePath {
        let backend = if matches!(context.plan.distribution, DistributionMode::Distributed) {
            ComputeBackend::DataFusionDistributed
        } else {
            ComputeBackend::DataFusionLocal
        };
        CandidateWritePath {
            backend: backend.clone(),
            access_method: access_method_for_schema(context.target_schema),
            pushdown: PushdownCapabilities {
                projection: true,
                filter: true,
                aggregate: true,
                join: true,
                limit: true,
                ..Default::default()
            },
            cost_hints: cost_hints_for_schema(context.target_schema),
            estimated_cost: self.estimate_cost(context, &backend),
            guards: self.required_guards(context),
        }
    }

    fn external_candidate(&self, context: &RoutingContext<'_>) -> CandidateWritePath {
        let backend = target_primary_format(context.target_schema)
            .map(|format| ComputeBackend::ExternalDelegated(format!("{format:?}")))
            .unwrap_or_else(|| ComputeBackend::ExternalDelegated("external".to_string()));
        CandidateWritePath {
            backend: backend.clone(),
            access_method: AccessMethodFamily::ExternalOpenTable,
            pushdown: PushdownCapabilities {
                projection: true,
                filter: true,
                aggregate: true,
                limit: true,
                requires_proxima_recheck: true,
                ..Default::default()
            },
            cost_hints: cost_hints_for_schema(context.target_schema),
            estimated_cost: self.estimate_cost(context, &backend),
            guards: self.required_guards(context),
        }
    }

    fn datafusion_applicable(&self, context: &RoutingContext<'_>) -> bool {
        let target = context.target_schema;
        if projection_route_block_reason(target).is_some() {
            return false;
        }
        if requires_native_row_delta_commit(context) {
            return false;
        }
        matches!(context.plan.distribution, DistributionMode::Distributed)
            || preferred_compute_route(target).is_some_and(|route| {
                matches!(
                    route.as_str(),
                    "datafusion" | "datafusion-local" | "datafusion-distributed" | "distributed"
                )
            })
            || matches!(
                target.workload_profile,
                CatalogWorkloadProfile::Olap
                    | CatalogWorkloadProfile::Htap
                    | CatalogWorkloadProfile::Mixed
            )
            || matches!(
                target.storage_specialization,
                CatalogStorageSpecialization::PaxOlap
                    | CatalogStorageSpecialization::ColumnarAnalytics
            )
    }

    fn external_commit_applicable(&self, context: &RoutingContext<'_>) -> bool {
        let target = context.target_schema;
        if !target_authority(target).is_external_authoritative() {
            return false;
        }
        let Some(format) = target_primary_format(target) else {
            return false;
        };
        matches!(
            format,
            CatalogPhysicalFormat::Iceberg
                | CatalogPhysicalFormat::Delta
                | CatalogPhysicalFormat::Hudi
        ) && primary_layout(target).is_none_or(|layout| layout.lossy_type_mappings.is_empty())
    }

    fn required_guards(&self, context: &RoutingContext<'_>) -> Vec<ExecutionGuard> {
        let mut guards = vec![
            ExecutionGuard::PinSourceSnapshot,
            ExecutionGuard::CheckTargetWriteCapabilities,
        ];

        if matches!(
            context.plan.write_mode,
            WriteMode::OverwriteTable | WriteMode::ReplacePartitions(_)
        ) {
            guards.push(ExecutionGuard::PreservePreviousSnapshot);
            guards.push(ExecutionGuard::RequireIdempotencyKey);
        }

        if target_authority(context.target_schema).is_external_authoritative() {
            guards.push(ExecutionGuard::RequireExternalAtomicCommit);
        } else {
            guards.push(ExecutionGuard::EnforceRlsInProxima);
        }

        guards
    }

    fn estimate_cost(
        &self,
        context: &RoutingContext<'_>,
        backend: &ComputeBackend,
    ) -> TableWriteCostEstimate {
        let write_mode_penalty = match context.plan.write_mode {
            WriteMode::Append | WriteMode::InsertOnly | WriteMode::Upsert => 1.0,
            WriteMode::OverwriteTable | WriteMode::ReplacePartitions(_) => 3.0,
            WriteMode::Merge => 4.0,
        };
        let backend_cost = match backend {
            ComputeBackend::Native => 1.0,
            ComputeBackend::DataFusionLocal => {
                if matches!(
                    context.target_schema.workload_profile,
                    CatalogWorkloadProfile::Olap
                ) || matches!(
                    context.target_schema.storage_specialization,
                    CatalogStorageSpecialization::PaxOlap
                        | CatalogStorageSpecialization::ColumnarAnalytics
                ) {
                    0.8
                } else if matches!(
                    context.plan.write_mode,
                    WriteMode::Append | WriteMode::InsertOnly | WriteMode::Upsert
                ) && matches!(
                    context.target_schema.workload_profile,
                    CatalogWorkloadProfile::Oltp
                ) {
                    3.0
                } else {
                    1.2
                }
            }
            ComputeBackend::DataFusionDistributed => {
                if matches!(context.plan.distribution, DistributionMode::Distributed) {
                    0.9
                } else {
                    2.0
                }
            }
            ComputeBackend::PolarsLocal => 1.4,
            ComputeBackend::DuckDbCompat => 2.5,
            ComputeBackend::ExternalDelegated(_) => {
                if target_authority(context.target_schema).is_external_authoritative() {
                    0.6
                } else {
                    3.0
                }
            }
        };
        let mut estimate = TableWriteCostEstimate::new(
            write_mode_penalty * backend_cost,
            format!(
                "rule-based route for {:?} {:?}",
                context.target_schema.workload_profile, context.plan.write_mode
            ),
        );
        let movement = data_movement_for_context(context);
        estimate.rows = movement.source_rows;
        estimate.bytes = match context.plan.write_mode {
            WriteMode::OverwriteTable | WriteMode::ReplacePartitions(_) => movement
                .estimated_read_bytes
                .zip(movement.estimated_rewrite_bytes)
                .map(|(read, rewrite)| read.saturating_add(rewrite))
                .or(movement.estimated_read_bytes)
                .or(movement.estimated_rewrite_bytes),
            _ => movement
                .estimated_read_bytes
                .or(movement.estimated_write_bytes),
        };
        estimate
    }

    fn write_intent(&self, context: &RoutingContext<'_>, parent_has_deletes: bool) -> WriteIntent {
        let mut intent = WriteIntent::new(
            context.plan.target.qualified_name(),
            write_operation_kind(&context.plan.write_mode),
        )
        .with_durability(write_durability_for_context(context))
        .with_isolation(write_isolation_for_schema(context.target_schema))
        .with_projection_freshness(projection_freshness_for_schema(context.target_schema));

        if let Some(version) = catalog_schema_version(context.target_schema) {
            intent = intent.with_catalog_schema_version(version);
        }

        if matches!(
            context.plan.write_mode,
            WriteMode::Upsert | WriteMode::Merge
        ) {
            intent = intent.with_row_level_semantics(true);
        }

        if batch_local_constraints_sufficient(context) {
            intent = intent.with_batch_local_constraints_sufficient(true);
        }

        if parent_has_deletes {
            intent = intent.with_published_parent_has_deletes(true);
        }

        if let Some(overrides) = context.write_intent_overrides {
            intent = apply_write_intent_overrides(intent, overrides);
        }

        intent
    }
}

fn read_source_name(source: &ReadSource) -> String {
    match source {
        ReadSource::CatalogTable { table, snapshot } => {
            format!("catalog:{}@{:?}", table.qualified_name(), snapshot)
        }
        ReadSource::ExternalLocation {
            format,
            schema_hint,
        } => match schema_hint {
            Some(schema_hint) => format!(
                "external:{:?}:{} schema={}",
                format.format, format.location, schema_hint
            ),
            None => format!("external:{:?}:{}", format.format, format.location),
        },
        ReadSource::QuerySql(sql) => format!("sql:{}", sql),
        ReadSource::ArrowFlightTicket(ticket) => format!("arrow-flight:{}", ticket),
    }
}

fn backend_name(backend: &ComputeBackend) -> String {
    match backend {
        ComputeBackend::ExternalDelegated(name) => format!("ExternalDelegated({name})"),
        other => format!("{other:?}"),
    }
}

fn access_method_name(access_method: &AccessMethodFamily) -> String {
    format!("{access_method:?}")
}

fn guard_names(guards: &[ExecutionGuard]) -> Vec<String> {
    guards.iter().map(|guard| format!("{guard:?}")).collect()
}

fn write_guard_names(guards: &[WriteGuard]) -> Vec<String> {
    guards.iter().map(|guard| format!("{guard:?}")).collect()
}

fn write_intent_explanation(intent: &WriteIntent) -> TableWriteIntentExplanation {
    TableWriteIntentExplanation {
        target_table: intent.target_table.clone(),
        operation_kind: format!("{:?}", intent.operation_kind),
        durability: format!("{:?}", intent.durability),
        isolation: format!("{:?}", intent.isolation),
        projection_freshness: format!("{:?}", intent.projection_freshness),
        tenant_id: intent.tenant_id.clone(),
        actor: intent.actor.clone(),
        idempotency_key: intent.idempotency_key.clone(),
        catalog_schema_version: intent.catalog_schema_version,
        row_count_hint: intent.row_count_hint,
        estimated_bytes: intent.estimated_bytes,
        requires_row_level_semantics: intent.requires_row_level_semantics,
        batch_local_constraints_sufficient: intent.batch_local_constraints_sufficient,
    }
}

fn cost_explanation(cost: &TableWriteCostEstimate) -> TableWriteCostExplanation {
    TableWriteCostExplanation {
        rows: cost.rows,
        bytes: cost.bytes,
        relative_cost: cost.relative_cost,
        reason: cost.reason.clone(),
    }
}

fn pushdown_names(pushdown: &PushdownCapabilities) -> Vec<String> {
    let mut names = Vec::new();
    if pushdown.projection {
        names.push("projection".to_string());
    }
    if pushdown.filter {
        names.push("filter".to_string());
    }
    if pushdown.aggregate {
        names.push("aggregate".to_string());
    }
    if pushdown.join {
        names.push("join".to_string());
    }
    if pushdown.limit {
        names.push("limit".to_string());
    }
    if pushdown.vector_topk {
        names.push("vector_topk".to_string());
    }
    if pushdown.graph_pattern {
        names.push("graph_pattern".to_string());
    }
    if pushdown.json_path {
        names.push("json_path".to_string());
    }
    if pushdown.requires_proxima_recheck {
        names.push("requires_proxima_recheck".to_string());
    }
    names
}

fn route_metadata_explanation(
    metadata: &RouteDecisionMetadata,
) -> TableWriteRouteMetadataExplanation {
    TableWriteRouteMetadataExplanation {
        authority_mode: metadata.authority_mode.ownership_mode_name().to_string(),
        workload_profile: metadata.workload_profile.as_str().to_string(),
        storage_specialization: metadata.storage_specialization.as_str().to_string(),
        primary_format: metadata
            .primary_format
            .as_ref()
            .map(|format| format!("{format:?}")),
        preferred_compute_route: metadata.preferred_compute_route.clone(),
        partitioning: metadata.partitioning.clone(),
        isolation_profile: metadata.isolation_profile.clone(),
        freshness_sla: metadata.freshness_sla.clone(),
        projection_freshness_state: metadata
            .projection_freshness_state
            .map(|state| format!("{state:?}")),
        projection_metadata: metadata
            .projection_metadata
            .iter()
            .map(projection_metadata_explanation)
            .collect(),
        policy_boundary: metadata.policy_boundary.clone(),
        constraint_enforcement: metadata.constraint_enforcement.clone(),
        constraint_gaps: metadata.constraint_gaps.clone(),
    }
}

fn projection_metadata_explanation(
    metadata: &ProjectionRouteMetadata,
) -> ProjectionRouteMetadataExplanation {
    ProjectionRouteMetadataExplanation {
        name: metadata.name.clone(),
        kind: metadata.kind.clone(),
        physical_format: metadata.physical_format.clone(),
        rebuild_source: metadata.rebuild_source.clone(),
        freshness: metadata.freshness.clone(),
        freshness_state: format!("{:?}", metadata.freshness_state),
        max_lag_ms: metadata.max_lag_ms,
        source_range: metadata.source_range.clone(),
        last_included_position: metadata.last_included_position.clone(),
        rebuildable: metadata.rebuildable,
        invalidation_policy: metadata.invalidation_policy.clone(),
        policy_boundary: metadata.policy_boundary.clone(),
        lossy: metadata.lossy,
        support_status: metadata.support_status.clone(),
        benchmark_gate: metadata.benchmark_gate.clone(),
    }
}

fn candidate_explanation(path: &CandidateWritePath) -> TableWriteCandidateExplanation {
    TableWriteCandidateExplanation {
        backend: backend_name(&path.backend),
        access_method: access_method_name(&path.access_method),
        estimated_cost: cost_explanation(&path.estimated_cost),
        required_guards: guard_names(&path.guards),
        pushdown: pushdown_names(&path.pushdown),
    }
}

fn rejected_path_explanation(path: &RejectedCandidatePath) -> TableWriteRejectedPathExplanation {
    TableWriteRejectedPathExplanation {
        backend: backend_name(&path.backend),
        access_method: access_method_name(&path.access_method),
        reason: path.reason.clone(),
        required_guards: guard_names(&path.required_guards),
    }
}

fn rejected_write_lane_explanation(
    rejected: &RejectedWriteLane,
) -> TableWriteRejectedLaneExplanation {
    TableWriteRejectedLaneExplanation {
        lane: format!("{:?}", rejected.lane),
        reason: rejected.reason.clone(),
    }
}

fn apply_write_intent_overrides(
    mut intent: WriteIntent,
    overrides: &WriteIntentOverrides,
) -> WriteIntent {
    if let Some(tenant_id) = &overrides.tenant_id {
        intent = intent.with_tenant_id(tenant_id.clone());
    }
    if let Some(actor) = &overrides.actor {
        intent = intent.with_actor(actor.clone());
    }
    if let Some(idempotency_key) = &overrides.idempotency_key {
        intent = intent.with_idempotency_key(idempotency_key.clone());
    }
    if let Some(row_count_hint) = overrides.row_count_hint {
        intent = intent.with_row_count_hint(row_count_hint);
    }
    if let Some(estimated_bytes) = overrides.estimated_bytes {
        intent = intent.with_estimated_bytes(estimated_bytes);
    }
    if let Some(requires_row_level_semantics) = overrides.requires_row_level_semantics {
        intent = intent.with_row_level_semantics(requires_row_level_semantics);
    }
    if let Some(batch_local_constraints_sufficient) = overrides.batch_local_constraints_sufficient {
        intent = intent.with_batch_local_constraints_sufficient(batch_local_constraints_sufficient);
    }
    intent
}

fn data_movement_for_context(context: &RoutingContext<'_>) -> TableWriteDataMovementExplanation {
    let source_rows = known_row_count(context.source_stats);
    let source_bytes = known_size_bytes(context.source_stats);
    let target_rows = known_row_count(context.target_stats);
    let target_bytes = known_size_bytes(context.target_stats);
    let source_last_analyzed_ms = context
        .source_stats
        .and_then(|stats| stats.last_analyzed_ms);
    let target_last_analyzed_ms = context
        .target_stats
        .and_then(|stats| stats.last_analyzed_ms);
    let now_ms = current_epoch_ms();
    let source_stats_age_ms = stats_age_ms(source_last_analyzed_ms, now_ms);
    let target_stats_age_ms = stats_age_ms(target_last_analyzed_ms, now_ms);
    let freshness_sla_ms =
        freshness_sla_for_schema(context.target_schema).and_then(|sla| parse_duration_ms(&sla));
    let estimated_read_bytes = source_bytes;
    let estimated_write_bytes = source_bytes;
    let estimated_rewrite_bytes = match context.plan.write_mode {
        WriteMode::OverwriteTable | WriteMode::ReplacePartitions(_) => target_bytes,
        _ => None,
    };

    let estimate_source = match (
        stats_are_known(context.source_stats),
        stats_are_known(context.target_stats),
    ) {
        (true, true) => "xcatalog-source-and-target-statistics",
        (true, false) => "xcatalog-source-statistics",
        (false, true) => "xcatalog-target-statistics",
        (false, false) => "unknown",
    }
    .to_string();
    let stats_freshness = stats_freshness_for_context(
        context,
        source_last_analyzed_ms,
        target_last_analyzed_ms,
        source_stats_age_ms,
        target_stats_age_ms,
        freshness_sla_ms,
    );

    TableWriteDataMovementExplanation {
        source_rows,
        source_bytes,
        target_rows_before_write: target_rows,
        target_bytes_before_write: target_bytes,
        estimated_read_bytes,
        estimated_write_bytes,
        estimated_rewrite_bytes,
        estimate_source,
        source_last_analyzed_ms,
        target_last_analyzed_ms,
        source_stats_age_ms,
        target_stats_age_ms,
        freshness_sla_ms,
        stats_freshness,
    }
}

fn stats_freshness_for_context(
    context: &RoutingContext<'_>,
    source_last_analyzed_ms: Option<i64>,
    target_last_analyzed_ms: Option<i64>,
    source_stats_age_ms: Option<u64>,
    target_stats_age_ms: Option<u64>,
    freshness_sla_ms: Option<u64>,
) -> String {
    let source_known = stats_are_known(context.source_stats);
    let target_known = stats_are_known(context.target_stats);
    if !source_known && !target_known {
        return "unknown".to_string();
    }

    let known_stats_missing_analyze_time = (source_known && source_last_analyzed_ms.is_none())
        || (target_known && target_last_analyzed_ms.is_none());
    if known_stats_missing_analyze_time {
        return "counters-without-analyze-time".to_string();
    }

    let Some(sla_ms) = freshness_sla_ms else {
        return "analyzed-no-freshness-sla".to_string();
    };

    let source_stale = source_known && source_stats_age_ms.is_some_and(|age| age > sla_ms);
    let target_stale = target_known && target_stats_age_ms.is_some_and(|age| age > sla_ms);
    if source_stale || target_stale {
        "stale-for-freshness-sla".to_string()
    } else {
        "fresh-within-freshness-sla".to_string()
    }
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn stats_age_ms(last_analyzed_ms: Option<i64>, now_ms: u64) -> Option<u64> {
    let analyzed_ms = last_analyzed_ms?;
    if analyzed_ms <= 0 {
        return Some(now_ms);
    }
    Some(now_ms.saturating_sub(analyzed_ms as u64))
}

fn parse_duration_ms(value: &str) -> Option<u64> {
    let trimmed = value.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    let number_len = trimmed
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(index, ch)| index + ch.len_utf8())
        .last()?;
    let amount = trimmed[..number_len].parse::<u64>().ok()?;
    let unit = trimmed[number_len..].trim();
    match unit {
        "" | "ms" | "millisecond" | "milliseconds" => Some(amount),
        "s" | "sec" | "secs" | "second" | "seconds" => amount.checked_mul(1_000),
        "m" | "min" | "mins" | "minute" | "minutes" => amount.checked_mul(60_000),
        "h" | "hr" | "hrs" | "hour" | "hours" => amount.checked_mul(3_600_000),
        "d" | "day" | "days" => amount.checked_mul(86_400_000),
        _ => None,
    }
}

fn stats_are_known(stats: Option<&CatalogTableStatistics>) -> bool {
    stats.is_some_and(|stats| {
        stats.row_count > 0
            || stats.size_bytes > 0
            || stats.file_count > 0
            || stats.last_analyzed_ms.is_some()
            || !stats.column_stats.is_empty()
    })
}

fn known_row_count(stats: Option<&CatalogTableStatistics>) -> Option<u64> {
    stats.and_then(|stats| stats_are_known(Some(stats)).then_some(stats.row_count))
}

fn known_size_bytes(stats: Option<&CatalogTableStatistics>) -> Option<u64> {
    stats.and_then(|stats| stats_are_known(Some(stats)).then_some(stats.size_bytes))
}

fn write_operation_kind(write_mode: &WriteMode) -> WriteOperationKind {
    match write_mode {
        WriteMode::Append => WriteOperationKind::Append,
        WriteMode::InsertOnly => WriteOperationKind::Insert,
        WriteMode::Upsert => WriteOperationKind::Upsert,
        WriteMode::OverwriteTable => WriteOperationKind::OverwriteTable,
        WriteMode::ReplacePartitions(_) => WriteOperationKind::OverwritePartitions,
        WriteMode::Merge => WriteOperationKind::Merge,
    }
}

fn write_durability_for_context(context: &RoutingContext<'_>) -> WriteDurabilityRequirement {
    if target_authority(context.target_schema).is_external_authoritative() {
        return WriteDurabilityRequirement::ExternalAuthoritative;
    }

    if requires_native_row_delta_commit(context) {
        return WriteDurabilityRequirement::WalRequired;
    }

    if matches!(
        context.plan.write_mode,
        WriteMode::Append
            | WriteMode::InsertOnly
            | WriteMode::OverwriteTable
            | WriteMode::ReplacePartitions(_)
    ) && supports_direct_snapshot_commit(context.target_schema)
    {
        return WriteDurabilityRequirement::DirectCommitAllowed;
    }

    WriteDurabilityRequirement::WalRequired
}

fn requires_native_row_delta_commit(context: &RoutingContext<'_>) -> bool {
    is_values_source(&context.plan.source)
        || matches!(
            context.plan.write_mode,
            WriteMode::Upsert | WriteMode::Merge
        )
}

fn is_values_source(source: &ReadSource) -> bool {
    matches!(source, ReadSource::QuerySql(sql) if sql.trim().eq_ignore_ascii_case("VALUES"))
}

fn supports_direct_snapshot_commit(schema: &CatalogTableSchema) -> bool {
    matches!(
        schema.storage_specialization,
        CatalogStorageSpecialization::PaxOlap
            | CatalogStorageSpecialization::ColumnarAnalytics
            | CatalogStorageSpecialization::ExternalOpenTable
    ) || matches!(
        target_primary_format(schema),
        Some(
            CatalogPhysicalFormat::Iceberg
                | CatalogPhysicalFormat::Delta
                | CatalogPhysicalFormat::Hudi
                | CatalogPhysicalFormat::Parquet
        )
    )
}

fn write_isolation_for_schema(schema: &CatalogTableSchema) -> WriteIsolationRequirement {
    let profile = schema
        .relational_capabilities
        .transaction_profile
        .clone()
        .or_else(|| property_value(schema, &["isolation_profile", "isolation"]));
    let Some(profile) = profile.map(|profile| profile.trim().to_ascii_lowercase()) else {
        return WriteIsolationRequirement::ReadCommitted;
    };

    if profile.contains("serializable") {
        WriteIsolationRequirement::Serializable
    } else if profile.contains("snapshot") {
        WriteIsolationRequirement::Snapshot
    } else {
        WriteIsolationRequirement::ReadCommitted
    }
}

fn projection_freshness_for_schema(schema: &CatalogTableSchema) -> ProjectionFreshnessRequirement {
    let Some(freshness) = freshness_sla_for_schema(schema)
        .map(|value| value.trim().to_ascii_lowercase().replace(['_', '-'], ""))
    else {
        return ProjectionFreshnessRequirement::None;
    };

    if freshness.contains("synchronous") || freshness.contains("sync") {
        ProjectionFreshnessRequirement::Synchronous
    } else if freshness.contains("read") && freshness.contains("write") {
        ProjectionFreshnessRequirement::ReadYourWrites
    } else if freshness.contains("besteffort") || freshness.contains("async") {
        ProjectionFreshnessRequirement::BestEffort
    } else {
        ProjectionFreshnessRequirement::None
    }
}

fn catalog_schema_version(schema: &CatalogTableSchema) -> Option<u64> {
    property_value(schema, &["schema_version", "catalog_schema_version"])
        .and_then(|version| version.parse::<u64>().ok())
}

fn batch_local_constraints_sufficient(context: &RoutingContext<'_>) -> bool {
    matches!(
        context.plan.write_mode,
        WriteMode::Append | WriteMode::InsertOnly
    ) && !matches!(
        context.plan.conflict_policy,
        ConflictPolicy::Upsert | ConflictPolicy::Merge
    )
}

fn access_method_for_schema(schema: &CatalogTableSchema) -> AccessMethodFamily {
    match schema.storage_specialization {
        CatalogStorageSpecialization::GenericRelational => AccessMethodFamily::NativeRecord,
        CatalogStorageSpecialization::PaxRowFamily
        | CatalogStorageSpecialization::PaxOltp
        | CatalogStorageSpecialization::PaxOlap => AccessMethodFamily::Pax,
        CatalogStorageSpecialization::LsmWriteOptimized => AccessMethodFamily::Lsm,
        CatalogStorageSpecialization::ColumnarAnalytics => AccessMethodFamily::Columnar,
        CatalogStorageSpecialization::VectorAnn => AccessMethodFamily::VectorAnn,
        CatalogStorageSpecialization::DocumentJson => AccessMethodFamily::DocumentJson,
        CatalogStorageSpecialization::GraphTopology => AccessMethodFamily::GraphTopology,
        CatalogStorageSpecialization::ObservabilityTimeSeries => {
            AccessMethodFamily::ObservabilityTimeSeries
        }
        CatalogStorageSpecialization::ExternalOpenTable => AccessMethodFamily::ExternalOpenTable,
    }
}

fn native_pushdown_for_schema(schema: &CatalogTableSchema) -> PushdownCapabilities {
    PushdownCapabilities {
        projection: true,
        filter: true,
        limit: true,
        vector_topk: matches!(
            schema.storage_specialization,
            CatalogStorageSpecialization::VectorAnn
        ),
        graph_pattern: matches!(
            schema.storage_specialization,
            CatalogStorageSpecialization::GraphTopology
        ),
        json_path: matches!(
            schema.storage_specialization,
            CatalogStorageSpecialization::DocumentJson
        ),
        ..Default::default()
    }
}

fn cost_hints_for_schema(schema: &CatalogTableSchema) -> AccessMethodCostHints {
    let mut hints = AccessMethodCostHints::default();
    match schema.storage_specialization {
        CatalogStorageSpecialization::PaxOltp | CatalogStorageSpecialization::GenericRelational => {
            hints.row_lookup_cost = 0.8;
            hints.write_amplification = 1.1;
        }
        CatalogStorageSpecialization::PaxOlap | CatalogStorageSpecialization::ColumnarAnalytics => {
            hints.scan_setup_cost = 0.7;
            hints.sequential_scan_cost_per_mb = 0.5;
            hints.write_amplification = 2.0;
        }
        CatalogStorageSpecialization::LsmWriteOptimized => {
            hints.row_lookup_cost = 1.2;
            hints.write_amplification = 0.7;
            hints.compaction_debt = 0.4;
        }
        CatalogStorageSpecialization::ExternalOpenTable => {
            hints.remote_read_cost_per_mb = 5.0;
            hints.write_amplification = 3.0;
        }
        _ => {}
    }
    hints
}

fn target_authority(schema: &CatalogTableSchema) -> CatalogAuthorityMode {
    primary_layout(schema)
        .map(|layout| layout.authority)
        .unwrap_or_default()
}

fn target_primary_format(schema: &CatalogTableSchema) -> Option<CatalogPhysicalFormat> {
    primary_layout(schema).map(|layout| layout.physical_format.clone())
}

fn primary_layout(schema: &CatalogTableSchema) -> Option<&proximadb_catalog::CatalogStorageLayout> {
    schema
        .storage_layouts
        .iter()
        .rev()
        .find(|layout| layout.name == "primary")
        .or_else(|| schema.storage_layouts.first())
}

fn route_metadata_for_schema(schema: &CatalogTableSchema) -> RouteDecisionMetadata {
    RouteDecisionMetadata {
        authority_mode: target_authority(schema),
        workload_profile: schema.workload_profile,
        storage_specialization: schema.storage_specialization,
        primary_format: target_primary_format(schema),
        preferred_compute_route: preferred_compute_route(schema),
        partitioning: property_value(
            schema,
            &["partitioning", "partition_key", "distribution_key"],
        ),
        isolation_profile: schema
            .relational_capabilities
            .transaction_profile
            .clone()
            .or_else(|| property_value(schema, &["isolation_profile", "isolation"])),
        freshness_sla: freshness_sla_for_schema(schema),
        projection_freshness_state: projection_freshness_state_for_schema(schema),
        projection_metadata: projection_metadata_for_schema(schema),
        policy_boundary: policy_boundary_for_schema(schema),
        constraint_enforcement: constraint_enforcement_for_schema(schema),
        constraint_gaps: constraint_gaps_for_schema(schema),
    }
}

fn property_value(schema: &CatalogTableSchema, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| schema.properties.get(*key).cloned())
}

fn preferred_compute_route(schema: &CatalogTableSchema) -> Option<String> {
    property_value(schema, &["compute_route", "preferred_compute_route"])
        .map(|route| route.trim().to_ascii_lowercase().replace('_', "-"))
}

fn freshness_sla_for_schema(schema: &CatalogTableSchema) -> Option<String> {
    property_value(schema, &["freshness_sla", "projection_freshness"]).or_else(|| {
        schema
            .projections
            .iter()
            .map(|projection| {
                projection
                    .max_lag_ms
                    .map(|max_lag_ms| format!("{max_lag_ms}ms"))
                    .unwrap_or_else(|| format!("{:?}", projection.freshness))
            })
            .next()
    })
}

fn projection_freshness_state_for_schema(
    schema: &CatalogTableSchema,
) -> Option<ProjectionFreshnessState> {
    schema
        .projections
        .iter()
        .map(|projection| projection.freshness_state)
        .max_by_key(|state| projection_state_severity(*state))
}

fn projection_state_severity(state: ProjectionFreshnessState) -> u8 {
    match state {
        ProjectionFreshnessState::Fresh => 0,
        ProjectionFreshnessState::ExternalSnapshotRegistered => 1,
        ProjectionFreshnessState::Updating => 2,
        ProjectionFreshnessState::Stale => 3,
        ProjectionFreshnessState::RebuildRequired => 4,
        ProjectionFreshnessState::Unavailable => 5,
    }
}

fn projection_metadata_for_schema(schema: &CatalogTableSchema) -> Vec<ProjectionRouteMetadata> {
    schema
        .projections
        .iter()
        .map(|projection| ProjectionRouteMetadata {
            name: projection.name.clone(),
            kind: format!("{:?}", projection.kind),
            physical_format: format!("{:?}", projection.physical_format),
            rebuild_source: projection.rebuild_source.clone(),
            freshness: format!("{:?}", projection.freshness),
            freshness_state: projection.freshness_state,
            max_lag_ms: projection.max_lag_ms,
            source_range: projection.source_range.clone(),
            last_included_position: projection.last_included_position.clone(),
            rebuildable: projection.rebuildable,
            invalidation_policy: projection.invalidation_policy.clone(),
            policy_boundary: projection.policy_boundary.clone(),
            lossy: projection.lossy,
            support_status: projection.support_status.clone(),
            benchmark_gate: projection.benchmark_gate.clone(),
        })
        .collect()
}

fn projection_route_block_reason(schema: &CatalogTableSchema) -> Option<String> {
    if property_value(schema, &["allow_stale_projection", "stale_projection_ok"]).is_some_and(
        |value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "true" | "1" | "yes"
            )
        },
    ) {
        return None;
    }

    let state = projection_freshness_state_for_schema(schema)?;
    match state {
        ProjectionFreshnessState::Fresh | ProjectionFreshnessState::ExternalSnapshotRegistered => {
            None
        }
        ProjectionFreshnessState::Updating => Some(
            "Projection freshness state is Updating; route must wait, use canonical records, or opt into stale projection reads"
                .to_string(),
        ),
        ProjectionFreshnessState::Stale => Some(
            "Projection freshness state is Stale; current-snapshot DataFusion/projection route is unsafe"
                .to_string(),
        ),
        ProjectionFreshnessState::RebuildRequired => Some(
            "Projection freshness state is RebuildRequired; projection route must be rejected until rebuild completes"
                .to_string(),
        ),
        ProjectionFreshnessState::Unavailable => Some(
            "Projection freshness state is Unavailable; projection route cannot be used"
                .to_string(),
        ),
    }
}

fn datafusion_rejection_reason(context: &RoutingContext<'_>) -> String {
    if let Some(reason) = projection_route_block_reason(context.target_schema) {
        return reason;
    }
    if requires_native_row_delta_commit(context) {
        return "Row-level/VALUES DML requires the native WAL + row/delta commit path before OLAP publication".to_string();
    }
    "DataFusion route is reserved for OLAP/HTAP/mixed, columnar/PAX-OLAP, or explicitly distributed writes".to_string()
}

fn policy_boundary_for_schema(schema: &CatalogTableSchema) -> String {
    if let Some(boundary) = property_value(schema, &["policy_boundary", "rls_boundary"]) {
        return boundary;
    }

    match primary_layout(schema) {
        Some(layout) if layout.policy_enforced_in_proxima => "engine-enforced".to_string(),
        Some(layout) if layout.authority.is_external_authoritative() => {
            "external-policy".to_string()
        }
        Some(layout) if matches!(layout.authority, CatalogAuthorityMode::FederatedRead) => {
            "connector-enforced".to_string()
        }
        Some(_) => "unsupported".to_string(),
        None => "engine-enforced".to_string(),
    }
}

fn constraint_enforcement_for_schema(schema: &CatalogTableSchema) -> String {
    let mut enforced = vec![
        "type".to_string(),
        "not_null".to_string(),
        "defaults".to_string(),
    ];

    if has_primary_key(schema) {
        enforced.push("primary_key_identity".to_string());
    }
    if has_check_constraints(schema) {
        enforced.push("check".to_string());
    }
    if has_unique_constraints(schema) {
        enforced.push("unique_non_null_fail_closed".to_string());
    }
    if has_foreign_key_constraints(schema) {
        enforced.push("foreign_key_non_null_fail_closed".to_string());
    }

    if constraint_gaps_for_schema(schema).is_empty() {
        format!("native_enforced:{}", enforced.join(","))
    } else {
        format!(
            "partial_native_enforced:{}; cataloged_gaps_present",
            enforced.join(",")
        )
    }
}

fn constraint_gaps_for_schema(schema: &CatalogTableSchema) -> Vec<String> {
    let mut gaps = Vec::new();

    if !schema.relational_capabilities.unique_indexes.is_empty() {
        gaps.push("unique_indexes_cataloged_not_enforced".to_string());
    }

    // TD-110: single-column FK references (INSERT parent-exists) and ON DELETE
    // referential actions are now enforced. Still unenforced — and therefore
    // still surfaced as a gap — are composite FKs and any ON UPDATE action.
    let mut has_unenforced_foreign_key = false;
    for constraint in &schema.relational_capabilities.constraints {
        match constraint {
            ColumnConstraint::Check { .. } => {}
            ColumnConstraint::ForeignKey {
                columns, on_update, ..
            } => {
                if columns.len() != 1 || on_update.is_some() {
                    has_unenforced_foreign_key = true;
                }
            }
            ColumnConstraint::Unique { .. } => {
                gaps.push("unique_constraints_cataloged_not_enforced".to_string());
            }
        }
    }

    if has_unenforced_foreign_key {
        gaps.push("foreign_keys_cataloged_not_enforced".to_string());
    }

    gaps.sort();
    gaps.dedup();
    gaps
}

fn has_primary_key(schema: &CatalogTableSchema) -> bool {
    !schema.primary_key.is_empty() || !schema.relational_capabilities.primary_key.is_empty()
}

fn has_check_constraints(schema: &CatalogTableSchema) -> bool {
    schema
        .relational_capabilities
        .constraints
        .iter()
        .any(|constraint| matches!(constraint, ColumnConstraint::Check { .. }))
}

fn has_unique_constraints(schema: &CatalogTableSchema) -> bool {
    !schema.relational_capabilities.unique_indexes.is_empty()
        || schema
            .relational_capabilities
            .constraints
            .iter()
            .any(|constraint| matches!(constraint, ColumnConstraint::Unique { .. }))
}

fn has_foreign_key_constraints(schema: &CatalogTableSchema) -> bool {
    schema
        .relational_capabilities
        .constraints
        .iter()
        .any(|constraint| matches!(constraint, ColumnConstraint::ForeignKey { .. }))
}

fn external_rejection_reason(schema: &CatalogTableSchema) -> String {
    let authority = target_authority(schema);
    if !authority.is_external_authoritative() {
        return format!(
            "Target authority is {}; external delegated commits require ExternalAuthoritative",
            authority.ownership_mode_name()
        );
    }

    let Some(format) = target_primary_format(schema) else {
        return "No primary external format is cataloged for delegated commit".to_string();
    };
    if !matches!(
        format,
        CatalogPhysicalFormat::Iceberg | CatalogPhysicalFormat::Delta | CatalogPhysicalFormat::Hudi
    ) {
        return format!("External delegated commit does not support format {format:?}");
    }

    if primary_layout(schema).is_some_and(|layout| !layout.lossy_type_mappings.is_empty()) {
        return "External delegated commit rejected because primary layout has lossy ProximaType mappings".to_string();
    }

    "External delegated commit is not applicable".to_string()
}

fn validate_target_table(schema: &CatalogTableSchema, plan: &CopyIntoPlan) -> Result<()> {
    if plan.target.name != schema.name {
        return Err(anyhow!(
            "DML target '{}' does not match resolved catalog table '{}'",
            plan.target.qualified_name(),
            schema.name
        ));
    }
    Ok(())
}

fn validate_target_columns(schema: &CatalogTableSchema, columns: &[String]) -> Result<()> {
    if columns.is_empty() {
        return Ok(());
    }

    let known_columns: HashSet<&str> = schema
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    for column in columns {
        if !known_columns.contains(column.as_str()) {
            return Err(anyhow!(
                "Unknown target column '{}' for table '{}'",
                column,
                schema.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::{DEFAULT_BULK_BYTES_THRESHOLD, DEFAULT_BULK_ROW_THRESHOLD};
    use proximadb_catalog::{
        CatalogColumn, CatalogIndex, CatalogIndexType, CatalogProjection, CatalogProjectionKind,
        CatalogStorageLayout, CatalogStorageSpecialization, ProjectionFreshnessState,
        RelationalCapabilities,
    };
    use proximadb_data_model::ProximaType;

    #[test]
    fn routes_oltp_pax_append_to_native() {
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");
        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(routed.backend, ComputeBackend::Native);
        assert_eq!(routed.selected_path.access_method, AccessMethodFamily::Pax);
        assert!(
            routed
                .required_guards
                .contains(&ExecutionGuard::PinSourceSnapshot)
        );
    }

    #[test]
    fn routes_distributed_olap_to_datafusion_distributed() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let mut plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM source");
        plan.distribution = DistributionMode::Distributed;

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(routed.backend, ComputeBackend::DataFusionDistributed);
        assert!(
            routed
                .candidate_paths
                .iter()
                .any(|path| path.backend == ComputeBackend::DataFusionDistributed)
        );
    }

    #[test]
    fn routes_external_iceberg_overwrite_to_external_commit() {
        let schema = CatalogTableSchema::new("iceberg_facts")
            .with_storage_layout(CatalogStorageLayout::external_authoritative(
                "primary",
                CatalogPhysicalFormat::Iceberg,
                "s3://warehouse/facts",
            ))
            .with_storage_specialization(CatalogStorageSpecialization::ExternalOpenTable);
        let plan = CopyIntoPlan::insert_overwrite(
            LogicalTableRef::new("iceberg_facts"),
            "SELECT * FROM source",
        );

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(
            routed.backend,
            ComputeBackend::ExternalDelegated("Iceberg".to_string())
        );
        assert_eq!(
            routed.selected_path.access_method,
            AccessMethodFamily::ExternalOpenTable
        );
        assert_eq!(
            format!("{:?}", routed.write_lane_decision.lane),
            "OverwriteSnapshotCommit"
        );
        assert_eq!(
            routed.write_intent.durability,
            WriteDurabilityRequirement::ExternalAuthoritative
        );
        assert!(
            routed
                .required_guards
                .contains(&ExecutionGuard::RequireExternalAtomicCommit)
        );
        assert!(
            routed
                .required_guards
                .contains(&ExecutionGuard::PreservePreviousSnapshot)
        );
        assert_eq!(
            format!("{:?}", routed.write_lane_decision.lane),
            "OverwriteSnapshotCommit"
        );
    }

    #[test]
    fn dml_write_planner_rejects_unknown_target_column() {
        let schema = CatalogTableSchema::new("orders")
            .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false));
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT id FROM staging");

        let err = DmlWritePlanner::default()
            .plan(DmlWritePlanRequest {
                target_schema: &schema,
                published_parent_has_deletes: false,
                target_stats: None,
                source_schema: None,
                source_stats: None,
                write_intent_overrides: None,
                plan: &plan,
                target_columns: &["missing".to_string()],
            })
            .unwrap_err();

        assert!(err.to_string().contains("Unknown target column"));
    }

    #[test]
    fn dml_write_planner_routes_catalog_resolved_overwrite() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_overwrite(LogicalTableRef::new("facts"), "SELECT * FROM staging");

        let routed = DmlWritePlanner::default()
            .plan(DmlWritePlanRequest {
                target_schema: &schema,
                published_parent_has_deletes: false,
                target_stats: None,
                source_schema: None,
                source_stats: None,
                write_intent_overrides: None,
                plan: &plan,
                target_columns: &[],
            })
            .unwrap();

        assert_eq!(routed.backend, ComputeBackend::DataFusionLocal);
        assert!(
            routed
                .required_guards
                .contains(&ExecutionGuard::PreservePreviousSnapshot)
        );
    }

    #[test]
    fn route_metadata_carries_catalog_knobs() {
        let mut schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics)
            .with_storage_layout(CatalogStorageLayout::proxima_authoritative_pax("primary"))
            .with_projection(
                CatalogProjection::rebuildable(
                    "facts_iceberg_publication",
                    CatalogProjectionKind::Columnar,
                    "primary",
                )
                .with_bounded_lag(5_000)
                .with_lineage("wal:1..42", "wal:42")
                .with_policy_and_gate("engine-enforced", "projection-publication-smoke"),
            )
            .with_relational_capabilities(RelationalCapabilities {
                transaction_profile: Some("snapshot-isolation".to_string()),
                ..Default::default()
            });
        schema
            .properties
            .insert("compute_route".to_string(), "datafusion-local".to_string());
        schema
            .properties
            .insert("partitioning".to_string(), "tenant_id,bucket".to_string());
        schema
            .properties
            .insert("freshness_sla".to_string(), "5s".to_string());
        schema
            .properties
            .insert("policy_boundary".to_string(), "engine-enforced".to_string());
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(
            routed.route_metadata.authority_mode,
            CatalogAuthorityMode::ProximaAuthoritative
        );
        assert_eq!(
            routed.route_metadata.workload_profile,
            CatalogWorkloadProfile::Olap
        );
        assert_eq!(
            routed.route_metadata.storage_specialization,
            CatalogStorageSpecialization::ColumnarAnalytics
        );
        assert_eq!(
            routed.route_metadata.primary_format,
            Some(CatalogPhysicalFormat::ProximaBlock)
        );
        assert_eq!(
            routed.route_metadata.preferred_compute_route.as_deref(),
            Some("datafusion-local")
        );
        assert_eq!(
            routed.route_metadata.partitioning.as_deref(),
            Some("tenant_id,bucket")
        );
        assert_eq!(
            routed.route_metadata.isolation_profile.as_deref(),
            Some("snapshot-isolation")
        );
        assert_eq!(routed.route_metadata.freshness_sla.as_deref(), Some("5s"));
        assert_eq!(
            routed.route_metadata.projection_freshness_state,
            Some(ProjectionFreshnessState::Fresh)
        );
        assert_eq!(routed.route_metadata.projection_metadata.len(), 1);
        let projection = &routed.route_metadata.projection_metadata[0];
        assert_eq!(projection.name, "facts_iceberg_publication");
        assert_eq!(projection.kind, "Columnar");
        assert_eq!(projection.rebuild_source, "primary");
        assert_eq!(projection.freshness, "BoundedLag");
        assert_eq!(projection.freshness_state, ProjectionFreshnessState::Fresh);
        assert_eq!(projection.max_lag_ms, Some(5_000));
        assert_eq!(projection.source_range.as_deref(), Some("wal:1..42"));
        assert_eq!(projection.last_included_position.as_deref(), Some("wal:42"));
        assert_eq!(
            projection.policy_boundary.as_deref(),
            Some("engine-enforced")
        );
        assert_eq!(
            projection.benchmark_gate.as_deref(),
            Some("projection-publication-smoke")
        );
        assert_eq!(routed.route_metadata.policy_boundary, "engine-enforced");
        assert!(
            routed
                .route_metadata
                .constraint_enforcement
                .starts_with("native_enforced:")
        );
        assert!(routed.route_metadata.constraint_gaps.is_empty());
    }

    #[test]
    fn route_metadata_discloses_cataloged_constraint_gaps() {
        let schema = CatalogTableSchema::new("orders")
            .with_primary_key(vec!["id".to_string()])
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["id".to_string()],
                unique_indexes: vec![
                    CatalogIndex::new(
                        "unique_email",
                        vec!["email".to_string()],
                        CatalogIndexType::BTree,
                    )
                    .unique(),
                ],
                constraints: vec![
                    ColumnConstraint::Check {
                        expression: "amount > 0".to_string(),
                    },
                    ColumnConstraint::ForeignKey {
                        columns: vec!["customer_id".to_string()],
                        references_table: "customers".to_string(),
                        references_columns: vec!["id".to_string()],
                        on_delete: None,
                        // ON UPDATE actions are still unenforced → still a gap.
                        on_update: Some(proximadb_catalog::ReferentialAction::Cascade),
                    },
                ],
                ..Default::default()
            });
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        let explanation = routed.route_explanation();

        assert!(
            explanation
                .route_metadata
                .constraint_enforcement
                .starts_with("partial_native_enforced:")
        );
        assert!(
            explanation
                .route_metadata
                .constraint_enforcement
                .contains("primary_key_identity")
        );
        assert!(
            explanation
                .route_metadata
                .constraint_enforcement
                .contains("check")
        );
        assert!(
            explanation
                .route_metadata
                .constraint_enforcement
                .contains("unique_non_null_fail_closed")
        );
        assert!(
            explanation
                .route_metadata
                .constraint_enforcement
                .contains("foreign_key_non_null_fail_closed")
        );
        assert_eq!(
            explanation.route_metadata.constraint_gaps,
            vec![
                "foreign_keys_cataloged_not_enforced".to_string(),
                "unique_indexes_cataloged_not_enforced".to_string(),
            ]
        );
    }

    #[test]
    fn route_explanation_serializes_selected_and_rejected_paths() {
        let schema = CatalogTableSchema::new("orders")
            .with_workload_profile(CatalogWorkloadProfile::Oltp)
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        let explanation = routed.route_explanation();

        assert_eq!(explanation.target_table, "orders");
        assert_eq!(explanation.selected_backend, "Native");
        assert_eq!(explanation.selected_access_method, "Pax");
        assert!(
            explanation
                .candidate_paths
                .iter()
                .any(|path| path.backend == "Native")
        );
        assert!(
            explanation
                .rejected_paths
                .iter()
                .any(|path| path.backend == "DataFusionLocal")
        );
        let json = serde_json::to_value(&explanation).expect("explanation should serialize");
        assert_eq!(json["route_metadata"]["workload_profile"], "oltp");
        assert_eq!(json["route_metadata"]["storage_specialization"], "pax_oltp");
        assert!(json["required_guards"].is_array());
        assert_eq!(json["write_intent"]["operation_kind"], "Append");
        assert_eq!(json["write_intent"]["durability"], "WalRequired");
        assert_eq!(json["write_lane"], "WalCurrentState");
        assert!(json["write_lane_required_guards"].is_array());
        assert!(json["rejected_write_lanes"].is_array());
    }

    #[test]
    fn route_explanation_includes_codegen_guardrail_fields() {
        let mut schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        schema
            .properties
            .insert("policy_boundary".to_string(), "engine-enforced".to_string());
        schema
            .properties
            .insert("compute_route".to_string(), "datafusion-local".to_string());
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });
        let json = serde_json::to_value(routed.route_explanation()).unwrap();

        for field in [
            "target_table",
            "source",
            "write_mode",
            "write_lane",
            "selected_backend",
            "selected_access_method",
            "estimated_cost",
            "data_movement",
            "required_guards",
            "route_metadata",
            "candidate_paths",
            "rejected_paths",
        ] {
            assert!(json.get(field).is_some(), "missing route field {field}");
        }
        assert_eq!(
            json["route_metadata"]["preferred_compute_route"],
            "datafusion-local"
        );
        assert_eq!(json["route_metadata"]["policy_boundary"], "engine-enforced");
        assert!(
            json["candidate_paths"]
                .as_array()
                .unwrap()
                .iter()
                .any(|path| {
                    path["backend"] == "DataFusionLocal"
                        && path["pushdown"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|capability| capability == "aggregate")
                })
        );
    }

    #[test]
    fn bounded_projection_freshness_drives_data_movement_sla() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics)
            .with_projection(
                CatalogProjection::rebuildable(
                    "facts_columnar",
                    CatalogProjectionKind::Columnar,
                    "facts.primary",
                )
                .with_bounded_lag(7_500)
                .with_freshness_state(ProjectionFreshnessState::Stale),
            );
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(
            routed.route_metadata.freshness_sla.as_deref(),
            Some("7500ms")
        );
        assert_eq!(
            routed.route_metadata.projection_freshness_state,
            Some(ProjectionFreshnessState::Stale)
        );
        assert!(
            routed
                .rejected_paths
                .iter()
                .any(|path| path.backend == ComputeBackend::DataFusionLocal
                    && path.reason.contains("Projection freshness state is Stale"))
        );
        assert_eq!(routed.data_movement.freshness_sla_ms, Some(7_500));
        let explanation = routed.route_explanation();
        assert_eq!(explanation.data_movement.freshness_sla_ms, Some(7_500));
        assert_eq!(
            explanation
                .route_metadata
                .projection_freshness_state
                .as_deref(),
            Some("Stale")
        );
    }

    #[test]
    fn write_intent_overrides_are_applied_before_lane_routing() {
        let schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("facts"), "SELECT * FROM staging");
        let overrides = WriteIntentOverrides {
            tenant_id: Some("tenant-a".to_string()),
            actor: Some("loader".to_string()),
            idempotency_key: Some("load-001".to_string()),
            row_count_hint: Some(DEFAULT_BULK_ROW_THRESHOLD),
            estimated_bytes: Some(DEFAULT_BULK_BYTES_THRESHOLD),
            requires_row_level_semantics: Some(false),
            batch_local_constraints_sufficient: Some(true),
        };

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: Some(&overrides),
            plan: &plan,
        });

        assert_eq!(routed.write_intent.tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(routed.write_intent.actor.as_deref(), Some("loader"));
        assert_eq!(
            routed.write_intent.idempotency_key.as_deref(),
            Some("load-001")
        );
        assert_eq!(
            routed.write_intent.row_count_hint,
            Some(DEFAULT_BULK_ROW_THRESHOLD)
        );
        assert_eq!(
            format!("{:?}", routed.write_lane_decision.lane),
            "BulkAppendCommit"
        );

        let explanation = routed.route_explanation();
        assert_eq!(
            explanation.write_intent.tenant_id.as_deref(),
            Some("tenant-a")
        );
        assert_eq!(
            explanation.write_intent.idempotency_key.as_deref(),
            Some("load-001")
        );
        assert_eq!(explanation.write_lane, "BulkAppendCommit");
    }

    #[test]
    fn route_explanation_carries_xcatalog_data_movement_estimates() {
        let mut schema = CatalogTableSchema::new("facts")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        schema
            .properties
            .insert("freshness_sla".to_string(), "5s".to_string());
        let source_schema = CatalogTableSchema::new("staging");
        let now_ms = current_epoch_ms() as i64;
        let target_stats = CatalogTableStatistics {
            row_count: 10_000,
            size_bytes: 4_000_000,
            file_count: 4,
            last_analyzed_ms: Some(now_ms),
            ..Default::default()
        };
        let source_stats = CatalogTableStatistics {
            row_count: 1_000,
            size_bytes: 512_000,
            file_count: 1,
            last_analyzed_ms: Some(now_ms.saturating_sub(10_000)),
            ..Default::default()
        };
        let plan =
            CopyIntoPlan::insert_overwrite(LogicalTableRef::new("facts"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: Some(&target_stats),
            source_schema: Some(&source_schema),
            source_stats: Some(&source_stats),
            write_intent_overrides: None,
            plan: &plan,
        });
        let explanation = routed.route_explanation();

        assert_eq!(explanation.data_movement.source_rows, Some(1_000));
        assert_eq!(explanation.data_movement.source_bytes, Some(512_000));
        assert_eq!(
            explanation.data_movement.target_bytes_before_write,
            Some(4_000_000)
        );
        assert_eq!(
            explanation.data_movement.estimated_read_bytes,
            Some(512_000)
        );
        assert_eq!(
            explanation.data_movement.estimated_write_bytes,
            Some(512_000)
        );
        assert_eq!(
            explanation.data_movement.estimated_rewrite_bytes,
            Some(4_000_000)
        );
        assert_eq!(
            explanation.data_movement.estimate_source,
            "xcatalog-source-and-target-statistics"
        );
        assert_eq!(
            explanation.data_movement.source_last_analyzed_ms,
            source_stats.last_analyzed_ms
        );
        assert_eq!(
            explanation.data_movement.target_last_analyzed_ms,
            target_stats.last_analyzed_ms
        );
        assert_eq!(explanation.data_movement.freshness_sla_ms, Some(5_000));
        assert_eq!(
            explanation.data_movement.stats_freshness,
            "stale-for-freshness-sla"
        );
        assert!(
            explanation
                .data_movement
                .source_stats_age_ms
                .is_some_and(|age| age >= 10_000)
        );
        assert_eq!(explanation.estimated_cost.rows, Some(1_000));
        assert_eq!(explanation.estimated_cost.bytes, Some(4_512_000));
    }

    #[test]
    fn write_intent_for_oltp_append_defaults_to_wal_current_state() {
        let schema = CatalogTableSchema::new("orders")
            .with_workload_profile(CatalogWorkloadProfile::Oltp)
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(
            routed.write_intent.operation_kind,
            WriteOperationKind::Append
        );
        assert_eq!(
            routed.write_intent.durability,
            WriteDurabilityRequirement::WalRequired
        );
        assert_eq!(
            format!("{:?}", routed.write_lane_decision.lane),
            "WalCurrentState"
        );
    }

    #[test]
    fn compute_route_property_adds_datafusion_candidate_for_oltp() {
        let mut schema = CatalogTableSchema::new("orders")
            .with_workload_profile(CatalogWorkloadProfile::Oltp)
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        schema
            .properties
            .insert("compute_route".to_string(), "datafusion-local".to_string());
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("orders"), "SELECT * FROM staging");

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert!(
            routed
                .candidate_paths
                .iter()
                .any(|path| path.backend == ComputeBackend::DataFusionLocal)
        );
        assert_eq!(
            routed.route_metadata.preferred_compute_route.as_deref(),
            Some("datafusion-local")
        );
    }

    #[test]
    fn values_dml_to_olap_table_stays_native_wal() {
        let mut schema = CatalogTableSchema::new("metrics")
            .with_workload_profile(CatalogWorkloadProfile::Olap)
            .with_storage_specialization(CatalogStorageSpecialization::ColumnarAnalytics);
        schema
            .properties
            .insert("compute_route".to_string(), "datafusion-local".to_string());
        let plan = CopyIntoPlan {
            source: ReadSource::QuerySql("VALUES".to_string()),
            target: LogicalTableRef::new("metrics"),
            write_mode: WriteMode::InsertOnly,
            conflict_policy: ConflictPolicy::Error,
            distribution: DistributionMode::Auto,
        };

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(routed.backend, ComputeBackend::Native);
        assert_eq!(
            routed.write_intent.durability,
            WriteDurabilityRequirement::WalRequired
        );
        assert!(
            routed
                .rejected_paths
                .iter()
                .any(|path| path.backend == ComputeBackend::DataFusionLocal
                    && path.reason.contains("row/delta commit path")),
            "expected TD-110 DataFusion rejection, got {:?}",
            routed.rejected_paths
        );
    }

    #[test]
    fn rejects_external_commit_when_type_mapping_is_lossy() {
        let mut layout = CatalogStorageLayout::external_authoritative(
            "primary",
            CatalogPhysicalFormat::Iceberg,
            "s3://warehouse/facts",
        );
        layout
            .lossy_type_mappings
            .push("Decimal(38,10)".to_string());
        let schema = CatalogTableSchema::new("iceberg_facts")
            .with_storage_layout(layout)
            .with_storage_specialization(CatalogStorageSpecialization::ExternalOpenTable);
        let plan = CopyIntoPlan::insert_overwrite(
            LogicalTableRef::new("iceberg_facts"),
            "SELECT * FROM source",
        );

        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert!(!matches!(
            routed.backend,
            ComputeBackend::ExternalDelegated(_)
        ));
        assert!(routed.rejected_paths.iter().any(|path| {
            matches!(path.backend, ComputeBackend::ExternalDelegated(_))
                && path.reason.contains("lossy ProximaType mappings")
        }));
    }

    #[test]
    fn routes_vector_ann_to_native_vector_ann_access_method() {
        let schema = CatalogTableSchema::new("embeddings")
            .with_storage_specialization(CatalogStorageSpecialization::VectorAnn);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("embeddings"), "SELECT * FROM src");
        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(routed.backend, ComputeBackend::Native);
        assert_eq!(
            routed.selected_path.access_method,
            AccessMethodFamily::VectorAnn
        );
        assert!(routed.selected_path.pushdown.vector_topk);
        // WAL lane is preserved even for vector-native path
        assert_eq!(
            format!("{:?}", routed.write_lane_decision.lane),
            "WalCurrentState"
        );
    }

    #[test]
    fn routes_lsm_write_optimized_to_native_lsm_access_method() {
        let schema = CatalogTableSchema::new("timeseries")
            .with_storage_specialization(CatalogStorageSpecialization::LsmWriteOptimized);
        let plan =
            CopyIntoPlan::insert_select(LogicalTableRef::new("timeseries"), "SELECT * FROM src");
        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert_eq!(routed.backend, ComputeBackend::Native);
        assert_eq!(routed.selected_path.access_method, AccessMethodFamily::Lsm);
        // Low write amplification characteristic of LSM
        assert!(routed.selected_path.cost_hints.write_amplification < 1.0);
        assert_eq!(
            format!("{:?}", routed.write_lane_decision.lane),
            "WalCurrentState"
        );
    }

    #[test]
    fn routes_external_delta_to_external_delegated() {
        let schema = CatalogTableSchema::new("delta_facts")
            .with_storage_layout(CatalogStorageLayout::external_authoritative(
                "primary",
                CatalogPhysicalFormat::Delta,
                "s3://warehouse/delta-facts",
            ))
            .with_storage_specialization(CatalogStorageSpecialization::ExternalOpenTable);
        let plan = CopyIntoPlan::insert_overwrite(
            LogicalTableRef::new("delta_facts"),
            "SELECT * FROM source",
        );
        let routed = TableWriteRouter.route(RoutingContext {
            target_schema: &schema,
            target_stats: None,
            source_schema: None,
            source_stats: None,
            write_intent_overrides: None,
            plan: &plan,
        });

        assert!(matches!(
            routed.backend,
            ComputeBackend::ExternalDelegated(_)
        ));
        if let ComputeBackend::ExternalDelegated(ref name) = routed.backend {
            assert_eq!(name.as_str(), "Delta");
        }
    }
}
