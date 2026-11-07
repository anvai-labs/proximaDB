//! Orchestration-level EXPLAIN plan structures for SQL queries.
//!
//! This module provides:
//! - `ExplainPlan`: Basic EXPLAIN output with orchestration steps and hints
//! - `EnhancedExplainPlan`: Detailed execution plan explanation including:
//!   - RL planner decision details
//!   - Optimization rules applied
//!   - Per-model cost estimates
//!   - Parallelization opportunities
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::query::explain::{EnhancedExplainPlan, OptimizationRule, ParallelStage};
//!
//! // Get enhanced explain plan
//! let plan = EnhancedExplainPlan::builder()
//!     .with_rl_explanation(rl_explanation)
//!     .with_rule(OptimizationRule::predicate_pushdown("filter pushed to vector scan"))
//!     .with_model_cost(ModelType::Vector, 10.5)
//!     .with_parallel_stage(stage)
//!     .build();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use proximadb_catalog::{
    CatalogCompressionRejectedCandidate, CatalogCompressionStatsProfile, CatalogPhysicalFormat,
    CatalogProjection, CatalogStorageLayout, CatalogTableSchema, RelationalCapabilities,
};

use crate::query::multimodal::plan::{
    PlanContext, ResolvedCompressionRejectedCandidateContext,
    ResolvedCompressionStatsProfileContext, ResolvedObjectContext, ResolvedProjectionContext,
    ResolvedStorageLayoutContext,
};
// TODO: Move to proximadb-graph crate
// For now, use local definitions
use crate::graph::query::planner::{GraphQueryPlan, GraphStatistics, PlanStepType};

use crate::storage::multimodel::ModelType;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplainPlan {
    pub orchestration_steps: Vec<String>,
    pub vector_hints: Option<VectorHints>,
    pub graph_hints: Option<GraphHints>,
    pub join_costs: Option<JoinExplainCostEstimate>,
    pub query_stats: Option<AnalyzeMetrics>,
    pub execution_strategy: Option<String>,
    pub estimated_total_cost: Option<f64>,
    /// Per-operation cost breakdown from the cost-based optimizer
    pub cost_breakdown: Option<Vec<ExplainCostEstimate>>,
    /// Join strategy chosen by the optimizer and the reasoning behind it
    pub join_strategy: Option<JoinStrategyExplanation>,
    /// Fusion strategy chosen by the optimizer and the reasoning behind it
    pub fusion_strategy: Option<FusionStrategyExplanation>,
    /// xCatalog storage authority, projection freshness, and relational capability metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_authority: Option<StorageAuthorityExplanation>,
    /// Vector object-storage route economics for SST/centroid/IVF-like projections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_object_economy: Option<VectorObjectEconomyExplain>,
}

/// Cost estimate for a single operation in the query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainCostEstimate {
    /// Human-readable operation name (e.g. "VectorSearch(products)")
    pub operation: String,
    /// Estimated cost units for this operation
    pub estimated_cost: f64,
    /// Estimated output row count
    pub estimated_rows: u64,
    /// Estimated fraction of input rows passing this operator's predicate
    /// (ADR-004's `estimated_selectivity` slot). Producer: the ADR-037 statistics
    /// envelope's per-field distinct/cardinality (see
    /// `core::statistics::StatisticsSummary::equality_selectivity`). `None` when
    /// no resident statistics are available for the predicate's field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_selectivity: Option<f64>,
    /// Optional free-form notes (e.g. "HNSW index used")
    pub notes: Option<String>,
}

/// Explanation of the join strategy selected by the cost-based optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinStrategyExplanation {
    /// Selected strategy name (HashJoin, NestedLoopJoin, IndexJoin)
    pub strategy: String,
    /// Estimated left-side cardinality that influenced the decision
    pub left_rows: u64,
    /// Estimated right-side cardinality that influenced the decision
    pub right_rows: u64,
    /// Human-readable explanation of why this strategy was chosen
    pub reason: String,
}

/// Explanation of the fusion strategy selected by the cost-based optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStrategyExplanation {
    /// Selected strategy name (Rrf, Intersection, Union, WeightedSum)
    pub strategy: String,
    /// Human-readable explanation of why this strategy was chosen
    pub reason: String,
}

impl ExplainPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an EXPLAIN plan with orchestration steps
    pub fn with_steps(steps: Vec<String>) -> Self {
        Self {
            orchestration_steps: steps,
            ..Default::default()
        }
    }

    /// Add vector hints to the plan
    pub fn with_vector_hints(mut self, hints: VectorHints) -> Self {
        self.vector_hints = Some(hints);
        self
    }

    /// Add graph hints to the plan
    pub fn with_graph_hints(mut self, hints: GraphHints) -> Self {
        self.graph_hints = Some(hints);
        self
    }

    /// Add join cost estimates
    pub fn with_join_costs(mut self, costs: JoinExplainCostEstimate) -> Self {
        self.join_costs = Some(costs);
        self
    }

    /// Add ANALYZE metrics
    pub fn with_analyze_metrics(mut self, metrics: AnalyzeMetrics) -> Self {
        self.query_stats = Some(metrics);
        self
    }

    /// Set the overall execution strategy
    pub fn with_execution_strategy(mut self, strategy: String) -> Self {
        self.execution_strategy = Some(strategy);
        self
    }

    /// Set estimated total cost
    pub fn with_total_cost(mut self, cost: f64) -> Self {
        self.estimated_total_cost = Some(cost);
        self
    }

    /// Set per-operation cost breakdown
    pub fn with_cost_breakdown(mut self, breakdown: Vec<ExplainCostEstimate>) -> Self {
        self.cost_breakdown = Some(breakdown);
        self
    }

    /// Set join strategy explanation
    pub fn with_join_strategy(mut self, explanation: JoinStrategyExplanation) -> Self {
        self.join_strategy = Some(explanation);
        self
    }

    /// Set fusion strategy explanation
    pub fn with_fusion_strategy(mut self, explanation: FusionStrategyExplanation) -> Self {
        self.fusion_strategy = Some(explanation);
        self
    }

    /// Add xCatalog storage authority and projection metadata to the plan.
    pub fn with_storage_authority(mut self, authority: StorageAuthorityExplanation) -> Self {
        self.storage_authority = Some(authority);
        self
    }

    /// Add object-storage vector route economics to the plan.
    pub fn with_vector_object_economy(mut self, explain: VectorObjectEconomyExplain) -> Self {
        self.vector_object_economy = Some(explain);
        self
    }
}

/// Vector object-storage route metadata for EXPLAIN and trace events.
///
/// This describes an access-method projection over canonical storage. It does
/// not make the projection authoritative; `authority_mode` and
/// `freshness_mode` must remain explicit so callers can audit stale or lossy
/// routes.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VectorObjectEconomyExplain {
    pub route_kind: String,
    pub authority_mode: String,
    pub freshness_mode: String,
    pub freshness_watermark_lsn: Option<u64>,
    pub policy_boundary: String,
    pub cache_status: String,
    pub cache_affinity_key: Option<String>,
    pub selected_files: usize,
    pub selected_blocks: usize,
    pub pruned_blocks: usize,
    pub estimated_object_gets: usize,
    pub actual_object_gets: Option<usize>,
    pub estimated_remote_bytes: u64,
    pub actual_remote_bytes: Option<u64>,
    pub overfetch_bytes: u64,
    pub wal_delta_searched: bool,
    pub rejected_route_reasons: Vec<String>,
    // ── Phase 5 Slice 5.5 — delta-merge observability ───────────────────
    /// Number of WAL/memtable delta records scanned for the query when
    /// the strong/bounded-stale route required a delta merge. `None`
    /// when no scan ran (StaleOk, or watermark already covered the WAL).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wal_delta_records_scanned: Option<u64>,
    /// Bytes read from WAL/memtable during the delta scan. `None` when
    /// no scan ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wal_delta_bytes: Option<u64>,
    /// Effective freshness mode chosen for this query, mirroring
    /// [`VectorFreshnessMode::explain_label`]. Stays separate from the
    /// existing `freshness_mode` (which is the projection-side label).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness_mode_used: Option<String>,
    /// Current WAL cursor observed at query plan time. Compared
    /// against `freshness_watermark_lsn` to decide whether a delta
    /// scan was needed. `None` when no source was wired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lsn_at_query: Option<u64>,
}

impl VectorObjectEconomyExplain {
    pub fn from_index_stats(stats: &crate::core::service_types::ServiceIndexStats) -> Self {
        Self {
            route_kind: "vector_object_economy".to_string(),
            authority_mode: "projection_over_canonical_records".to_string(),
            freshness_mode: "strong".to_string(),
            policy_boundary: "proxima_internal_policy".to_string(),
            cache_status: "unknown".to_string(),
            selected_blocks: stats.object_selected_blocks.max(0) as usize,
            pruned_blocks: stats.object_pruned_blocks.max(0) as usize,
            estimated_object_gets: stats.object_estimated_gets.max(0) as usize,
            actual_object_gets: Some(stats.object_actual_gets.max(0) as usize),
            estimated_remote_bytes: stats.object_estimated_remote_bytes.max(0) as u64,
            actual_remote_bytes: Some(stats.object_actual_remote_bytes.max(0) as u64),
            overfetch_bytes: stats.object_overfetch_bytes.max(0) as u64,
            ..Default::default()
        }
    }

    pub fn with_cache_status(mut self, cache_status: impl Into<String>) -> Self {
        self.cache_status = cache_status.into();
        self
    }

    pub fn with_selected_files(mut self, selected_files: usize) -> Self {
        self.selected_files = selected_files;
        self
    }

    /// Record the result of a WAL/memtable delta scan for this query
    /// (Phase 5 Slice 5.5). Populates the four delta-merge observability
    /// fields in a single call so the service path doesn't have to
    /// touch them individually.
    ///
    /// * `mode` — the freshness mode the service honored.
    /// * `current_lsn` — WAL cursor read at query plan time.
    /// * `scanned_records` / `scanned_bytes` — `None` when no scan ran
    ///   (StaleOk, or watermark already covered the WAL); `Some(0)` if
    ///   the scan ran but returned nothing.
    pub fn record_wal_delta_scan(
        mut self,
        mode: &crate::core::search::VectorFreshnessMode,
        current_lsn: Option<u64>,
        scanned_records: Option<u64>,
        scanned_bytes: Option<u64>,
    ) -> Self {
        self.freshness_mode_used = Some(mode.explain_label().to_string());
        // `None` = the WAL cursor was not read (e.g. no manifest service);
        // never a fabricated 0.
        self.current_lsn_at_query = current_lsn;
        self.wal_delta_records_scanned = scanned_records;
        self.wal_delta_bytes = scanned_bytes;
        // `wal_delta_searched` is the existing bool field; flip it
        // whenever any scan-related metric is populated so callers
        // reading just the bool still see the right state.
        self.wal_delta_searched = scanned_records.is_some();
        self
    }

    /// Record the outcome of a per-collection directory load. Pushes a
    /// reason into `rejected_route_reasons` when the load was degraded so
    /// EXPLAIN reflects the fallback to embedded SST index reads.
    pub fn record_directory_load(
        mut self,
        status: &crate::storage::engines::sst::object_economy_directory::DirectoryLoadStatus,
        sidecar_path: &str,
    ) -> Self {
        if status.is_degraded() {
            self.rejected_route_reasons
                .push(status.reason(sidecar_path));
        }
        self
    }
}

/// Storage authority metadata surfaced by EXPLAIN from xCatalog.
///
/// This keeps query plans honest about whether a scan uses canonical
/// ProximaRecord storage, an externally authoritative table, or a rebuildable
/// projection/access method. The fields are descriptive and do not change
/// planning behavior by themselves.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageAuthorityExplanation {
    /// Cataloged physical layouts that may satisfy the plan.
    pub layouts: Vec<StorageLayoutExplanation>,
    /// Cataloged projections/access methods considered or available.
    pub projections: Vec<ProjectionExplanation>,
    /// Optional relational integrity and transaction capabilities.
    pub relational_capabilities: RelationalCapabilityExplanation,
    /// Cataloged codec/layout profiling records available to the planner.
    #[serde(default)]
    pub compression_profiles: Vec<CompressionProfileExplanation>,
    /// Fallback behavior when a preferred projection is stale, missing, or lossy.
    pub fallback_behavior: String,
}

impl StorageAuthorityExplanation {
    /// Build EXPLAIN metadata from planner-native resolved object context.
    pub fn from_plan_context(context: &PlanContext) -> Option<Self> {
        if context.resolved_objects.is_empty() {
            return None;
        }

        let layouts = context
            .resolved_objects
            .iter()
            .flat_map(|object| object.storage_layouts.iter())
            .map(StorageLayoutExplanation::from)
            .collect();
        let projections = context
            .resolved_objects
            .iter()
            .flat_map(|object| object.projections.iter())
            .map(ProjectionExplanation::from)
            .collect();
        let compression_profiles = context
            .resolved_objects
            .iter()
            .flat_map(|object| object.compression_stats_profiles.iter())
            .map(CompressionProfileExplanation::from)
            .collect();
        let fallback_behavior = fallback_behavior_from_resolved_objects(&context.resolved_objects);

        Some(Self {
            layouts,
            projections,
            relational_capabilities: RelationalCapabilityExplanation::default(),
            compression_profiles,
            fallback_behavior,
        })
    }

    /// Build EXPLAIN metadata for one resolved source.
    pub fn from_resolved_object_context(object: &ResolvedObjectContext) -> Self {
        Self {
            layouts: object
                .storage_layouts
                .iter()
                .map(StorageLayoutExplanation::from)
                .collect(),
            projections: object
                .projections
                .iter()
                .map(ProjectionExplanation::from)
                .collect(),
            relational_capabilities: RelationalCapabilityExplanation::default(),
            compression_profiles: object
                .compression_stats_profiles
                .iter()
                .map(CompressionProfileExplanation::from)
                .collect(),
            fallback_behavior: object.fallback_behavior.clone(),
        }
    }

    /// Build EXPLAIN metadata from a catalog table schema.
    pub fn from_catalog_table_schema(schema: &CatalogTableSchema) -> Self {
        Self::from_catalog_metadata(
            &schema.storage_layouts,
            &schema.projections,
            &schema.compression_stats_profiles,
            &schema.relational_capabilities,
        )
    }

    /// Build EXPLAIN metadata from xCatalog layout/projection/capability records.
    pub fn from_catalog_metadata(
        layouts: &[CatalogStorageLayout],
        projections: &[CatalogProjection],
        compression_profiles: &[CatalogCompressionStatsProfile],
        relational_capabilities: &RelationalCapabilities,
    ) -> Self {
        let fallback_behavior = if projections.iter().any(|projection| !projection.rebuildable) {
            "planner must verify non-rebuildable projection freshness before use".to_string()
        } else if projections.iter().any(|projection| projection.lossy) {
            "planner should fall back to canonical records when exact recall is required"
                .to_string()
        } else if layouts
            .iter()
            .any(|layout| !layout.policy_enforced_in_proxima)
        {
            "planner must apply ProximaDB policy/RLS after external reads".to_string()
        } else {
            "canonical records remain the fallback for stale or unavailable projections".to_string()
        };

        Self {
            layouts: layouts.iter().map(StorageLayoutExplanation::from).collect(),
            projections: projections
                .iter()
                .map(ProjectionExplanation::from)
                .collect(),
            relational_capabilities: RelationalCapabilityExplanation::from(relational_capabilities),
            compression_profiles: compression_profiles
                .iter()
                .map(CompressionProfileExplanation::from)
                .collect(),
            fallback_behavior,
        }
    }

    /// Returns true when all authority metadata preserves ProximaDB policy semantics.
    pub fn policy_safe_inside_proxima(&self) -> bool {
        self.layouts
            .iter()
            .all(|layout| layout.policy_enforced_in_proxima)
    }
}

fn fallback_behavior_from_resolved_objects(objects: &[ResolvedObjectContext]) -> String {
    if objects
        .iter()
        .any(ResolvedObjectContext::requires_policy_boundary)
    {
        "planner must apply ProximaDB policy/RLS after external reads".to_string()
    } else if objects.iter().any(|object| {
        object
            .projections
            .iter()
            .any(|projection| !projection.rebuildable)
    }) {
        "planner must verify non-rebuildable projection freshness before use".to_string()
    } else if objects
        .iter()
        .any(|object| object.projections.iter().any(|projection| projection.lossy))
    {
        "planner should fall back to canonical records when exact recall is required".to_string()
    } else if objects.len() == 1 {
        objects[0].fallback_behavior.clone()
    } else {
        "canonical records remain the fallback for stale or unavailable projections".to_string()
    }
}

/// Physical layout authority row for EXPLAIN output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLayoutExplanation {
    pub name: String,
    pub authority: String,
    pub layout_kind: String,
    pub physical_format: String,
    pub write_mode: String,
    pub location: Option<String>,
    pub snapshot_semantics: Option<String>,
    pub policy_enforced_in_proxima: bool,
    pub lossy_type_mappings: Vec<String>,
}

impl From<&CatalogStorageLayout> for StorageLayoutExplanation {
    fn from(layout: &CatalogStorageLayout) -> Self {
        Self {
            name: layout.name.clone(),
            authority: format!("{:?}", layout.authority),
            layout_kind: format!("{:?}", layout.layout_kind),
            physical_format: physical_format_label(&layout.physical_format),
            write_mode: format!("{:?}", layout.write_mode),
            location: layout.location.clone(),
            snapshot_semantics: layout.snapshot_semantics.clone(),
            policy_enforced_in_proxima: layout.policy_enforced_in_proxima,
            lossy_type_mappings: layout.lossy_type_mappings.clone(),
        }
    }
}

impl From<&ResolvedStorageLayoutContext> for StorageLayoutExplanation {
    fn from(layout: &ResolvedStorageLayoutContext) -> Self {
        Self {
            name: layout.name.clone(),
            authority: format!("{:?}", layout.authority),
            layout_kind: layout.layout_kind.clone(),
            physical_format: layout.physical_format.clone(),
            write_mode: layout.write_mode.clone(),
            location: layout.location.clone(),
            snapshot_semantics: layout.snapshot_semantics.clone(),
            policy_enforced_in_proxima: layout.policy_enforced_in_proxima,
            lossy_type_mappings: layout.lossy_type_mappings.clone(),
        }
    }
}

/// Projection/access-method row for EXPLAIN output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionExplanation {
    pub name: String,
    pub kind: String,
    pub physical_format: String,
    pub rebuild_source: String,
    pub freshness: String,
    pub max_lag_ms: Option<i64>,
    pub rebuildable: bool,
    pub lossy: bool,
    pub support_status: String,
    /// `ProjectionFreshnessState` variant name. Absent when the catalog entry
    /// predates the freshness-state field or the projection is external-authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness_state: Option<String>,
    /// Rebuild rate from `RebuildRtoSpec` in seconds per 10 GiB. Absent when
    /// no RTO estimate has been benchmarked or cataloged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rebuild_rto_seconds_per_10gb: Option<f64>,
}

/// Codec/layout profiling feedback row for EXPLAIN output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionProfileExplanation {
    pub profile_id: String,
    pub layout_name: Option<String>,
    pub projection_id: Option<String>,
    pub selected_scheme: String,
    pub raw_bytes: u64,
    pub encoded_bytes: u64,
    pub value_count: u64,
    pub measured_ratio: f64,
    pub exact_reconstruction: bool,
    pub encode_cpu_ms_per_block: Option<f64>,
    pub decode_ns_per_value: Option<f64>,
    #[serde(default)]
    pub rejected_candidates: Vec<CompressionRejectedCandidateExplanation>,
}

impl CompressionProfileExplanation {
    pub fn bytes_per_value(&self) -> f64 {
        if self.value_count == 0 {
            0.0
        } else {
            self.encoded_bytes as f64 / self.value_count as f64
        }
    }
}

impl From<&CatalogCompressionStatsProfile> for CompressionProfileExplanation {
    fn from(profile: &CatalogCompressionStatsProfile) -> Self {
        Self {
            profile_id: profile.profile_id.clone(),
            layout_name: profile.layout_name.clone(),
            projection_id: profile.projection_id.clone(),
            selected_scheme: profile.selected_scheme.clone(),
            raw_bytes: profile.raw_bytes,
            encoded_bytes: profile.encoded_bytes,
            value_count: profile.value_count,
            measured_ratio: profile.measured_ratio,
            exact_reconstruction: profile.exact_reconstruction,
            encode_cpu_ms_per_block: profile.encode_cpu_ms_per_block,
            decode_ns_per_value: profile.decode_ns_per_value,
            rejected_candidates: profile
                .rejected_candidates
                .iter()
                .map(CompressionRejectedCandidateExplanation::from)
                .collect(),
        }
    }
}

impl From<&ResolvedCompressionStatsProfileContext> for CompressionProfileExplanation {
    fn from(profile: &ResolvedCompressionStatsProfileContext) -> Self {
        Self {
            profile_id: profile.profile_id.clone(),
            layout_name: profile.layout_name.clone(),
            projection_id: profile.projection_id.clone(),
            selected_scheme: profile.selected_scheme.clone(),
            raw_bytes: profile.raw_bytes,
            encoded_bytes: profile.encoded_bytes,
            value_count: profile.value_count,
            measured_ratio: profile.measured_ratio,
            exact_reconstruction: profile.exact_reconstruction,
            encode_cpu_ms_per_block: profile.encode_cpu_ms_per_block,
            decode_ns_per_value: profile.decode_ns_per_value,
            rejected_candidates: profile
                .rejected_candidates
                .iter()
                .map(CompressionRejectedCandidateExplanation::from)
                .collect(),
        }
    }
}

/// Rejected codec candidate surfaced in EXPLAIN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionRejectedCandidateExplanation {
    pub scheme: String,
    pub reason: String,
    pub expected_ratio: Option<f32>,
}

impl From<&CatalogCompressionRejectedCandidate> for CompressionRejectedCandidateExplanation {
    fn from(candidate: &CatalogCompressionRejectedCandidate) -> Self {
        Self {
            scheme: candidate.scheme.clone(),
            reason: candidate.reason.clone(),
            expected_ratio: candidate.expected_ratio,
        }
    }
}

impl From<&ResolvedCompressionRejectedCandidateContext>
    for CompressionRejectedCandidateExplanation
{
    fn from(candidate: &ResolvedCompressionRejectedCandidateContext) -> Self {
        Self {
            scheme: candidate.scheme.clone(),
            reason: candidate.reason.clone(),
            expected_ratio: candidate.expected_ratio,
        }
    }
}

impl From<&CatalogProjection> for ProjectionExplanation {
    fn from(projection: &CatalogProjection) -> Self {
        Self {
            name: projection.name.clone(),
            kind: format!("{:?}", projection.kind),
            physical_format: physical_format_label(&projection.physical_format),
            rebuild_source: projection.rebuild_source.clone(),
            freshness: format!("{:?}", projection.freshness),
            max_lag_ms: projection.max_lag_ms,
            rebuildable: projection.rebuildable,
            lossy: projection.lossy,
            support_status: projection.support_status.clone(),
            freshness_state: Some(format!("{:?}", projection.freshness_state)),
            rebuild_rto_seconds_per_10gb: projection
                .rebuild_rto
                .as_ref()
                .map(|rto| rto.rebuild_seconds_per_10gb),
        }
    }
}

impl From<&ResolvedProjectionContext> for ProjectionExplanation {
    fn from(projection: &ResolvedProjectionContext) -> Self {
        Self {
            name: projection.name.clone(),
            kind: projection.kind.clone(),
            physical_format: projection.physical_format.clone(),
            rebuild_source: projection.rebuild_source.clone(),
            freshness: projection.freshness.clone(),
            max_lag_ms: projection.max_lag_ms,
            rebuildable: projection.rebuildable,
            lossy: projection.lossy,
            support_status: projection.support_status.clone(),
            freshness_state: projection.freshness_state.clone(),
            rebuild_rto_seconds_per_10gb: projection.rebuild_rto_seconds_per_10gb,
        }
    }
}

/// Optional relational semantics surfaced by EXPLAIN.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationalCapabilityExplanation {
    pub has_enforced_semantics: bool,
    pub primary_key: Vec<String>,
    pub unique_index_count: usize,
    pub secondary_index_count: usize,
    pub constraint_count: usize,
    pub materialized_view_count: usize,
    pub transaction_profile: Option<String>,
    pub schema_evolution_policy: Option<String>,
}

impl From<&RelationalCapabilities> for RelationalCapabilityExplanation {
    fn from(capabilities: &RelationalCapabilities) -> Self {
        Self {
            has_enforced_semantics: capabilities.has_enforced_semantics(),
            primary_key: capabilities.primary_key.clone(),
            unique_index_count: capabilities.unique_indexes.len(),
            secondary_index_count: capabilities.secondary_indexes.len(),
            constraint_count: capabilities.constraints.len(),
            materialized_view_count: capabilities.materialized_views.len(),
            transaction_profile: capabilities.transaction_profile.clone(),
            schema_evolution_policy: capabilities.schema_evolution_policy.clone(),
        }
    }
}

fn physical_format_label(format: &CatalogPhysicalFormat) -> String {
    match format {
        CatalogPhysicalFormat::External(label) => label.clone(),
        other => format!("{:?}", other),
    }
}

/// Lightweight vector-side hints surfaced from VectorOperationsService when available.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorHints {
    pub cache_hit: bool,
    pub pruned_files: Option<usize>,
    pub ef_search: Option<usize>,
    pub nprobe: Option<usize>,
    pub candidates: Option<usize>,
    pub progressive_stages: Option<Vec<String>>,
    pub recall_estimates: Option<Vec<f32>>,
    pub index_type: Option<String>,
    pub quantization_level: Option<String>,
    pub estimated_io_cost: Option<f64>,
    pub estimated_compute_cost: Option<f64>,
    /// ADR-011 ANN filtering mode chosen by the planner: "PreFilter", "Inline", or "PostFilter".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ann_filtering_mode: Option<String>,
    /// Estimated fraction of records matching the scalar filter that drove
    /// ADR-011 route selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ann_filtering_selectivity: Option<f64>,
    /// Source of the selectivity estimate, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ann_filtering_selectivity_source: Option<String>,
    /// Why the planner chose this mode (selectivity estimate, policy override, degradation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ann_mode_reason: Option<String>,
    /// TurboQuant EXPLAIN hints (Phase F — Quantization Trait Convergence
    /// Plan). Populated when the search ran through a `ReadTime`-lifecycle
    /// quantization method (TurboQuant today). The value is the
    /// `TurboQuantExplainHints` struct serialized via its own
    /// `to_explain_value()` method (`src/index/turboquant_bridge.rs:298`),
    /// so the wire shape stays stable across REST / gRPC / Arrow Flight /
    /// pgwire per ADR-004 §"Plan Operators".
    ///
    /// The 9 fields surfaced (when present) are: `quantization`,
    /// `calibration_mode`, `rotation_seed`, `encoded_epoch`,
    /// `current_epoch`, `mask_pushed_to_kernel`, `kernel_arch`,
    /// `blocks_skipped_by_mask`, `length_renorm_applied`, plus optional
    /// `candidate_set_size` and `n_vectors_scanned`. See the
    /// `TurboQuantExplainHints` doc in the bridge for field semantics
    /// and the `current_epoch == None` rendering convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turboquant: Option<serde_json::Value>,
}

impl VectorHints {
    /// Builder-style setter for the TurboQuant EXPLAIN hint payload.
    /// Accepts the bridge's already-shipped `TurboQuantExplainHints`
    /// builder output and serializes it via `to_explain_value()` so the
    /// hint round-trips cleanly across every protocol surface (per
    /// ADR-004). Returns `self` for chaining at the constructor site.
    ///
    /// Available behind `experimental-turboquant` to mirror the bridge
    /// type's feature gate; non-TurboQuant builds carry no `serde_json`
    /// payload here.
    #[cfg(feature = "experimental-turboquant")]
    pub fn with_turboquant_hints(
        mut self,
        hints: &crate::index::turboquant_bridge::TurboQuantExplainHints,
    ) -> Self {
        self.turboquant = Some(hints.to_explain_value());
        // Mirror the quantization name into the existing top-level
        // `quantization_level` field so legacy EXPLAIN consumers that
        // only read that one slot still see the right value.
        self.quantization_level = Some(hints.quantization.clone());
        self
    }
}

impl From<&crate::services::operations::vectors::SearchPlanHints> for VectorHints {
    fn from(h: &crate::services::operations::vectors::SearchPlanHints) -> Self {
        let ann_mode_reason = match (
            h.ann_filtering_mode.as_deref(),
            h.ann_filtering_selectivity,
            h.ann_filtering_selectivity_source.as_deref(),
        ) {
            (Some(mode), Some(selectivity), Some(source)) => Some(format!(
                "{mode} selected from {source} selectivity={selectivity:.6}"
            )),
            (Some(mode), Some(selectivity), None) => {
                Some(format!("{mode} selected from selectivity={selectivity:.6}"))
            }
            (Some(mode), None, _) => Some(format!("{mode} selected without selectivity estimate")),
            _ => None,
        };

        Self {
            cache_hit: h.cache_hit,
            pruned_files: h.pruned_files,
            ef_search: h.ef_search,
            nprobe: h.nprobe,
            candidates: h.candidates,
            progressive_stages: h.progressive_stages.clone(),
            recall_estimates: h.recall_estimates.clone(),
            ann_filtering_mode: h.ann_filtering_mode.clone(),
            ann_filtering_selectivity: h.ann_filtering_selectivity,
            ann_filtering_selectivity_source: h.ann_filtering_selectivity_source.clone(),
            ann_mode_reason,
            // Phase J: propagate the TurboQuant EXPLAIN payload from the
            // caller-side hint bag straight into the wire-facing struct.
            // Cloned because `serde_json::Value` is owned. `None` is the
            // common case (most searches don't run through TurboQuant);
            // the `skip_serializing_if` on the field hides it from the
            // wire shape when absent.
            turboquant: h.turboquant.clone(),
            // index_type, quantization_level, estimated_*_cost populated by engine layer
            ..Default::default()
        }
    }
}

/// Graph-side hints from graph query planning and execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphHints {
    /// Graph traversal algorithm used
    pub traversal_algorithm: Option<String>,
    /// Maximum traversal depth
    pub max_depth: Option<u32>,
    /// Starting nodes for traversal
    pub start_nodes: Option<usize>,
    /// Index usage in graph operations
    pub index_usage: Vec<GraphIndexUsage>,
    /// Estimated nodes to visit
    pub estimated_nodes_visited: Option<usize>,
    /// Estimated edges to traverse
    pub estimated_edges_traversed: Option<usize>,
    /// Graph statistics used in planning
    pub graph_stats: Option<GraphPlannerStats>,
    /// Edge filters applied
    pub edge_filters: Option<usize>,
    /// Node filters applied
    pub node_filters: Option<usize>,
    /// Memory estimate for graph operation
    pub estimated_memory_mb: Option<f64>,
    /// Estimated I/O cost for graph operations
    pub estimated_io_cost: Option<f64>,
}

/// Information about index usage in graph operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIndexUsage {
    /// Index name
    pub index_name: String,
    /// Index type (label_index, property_index, composite_index)
    pub index_type: String,
    /// Estimated selectivity (0.0 to 1.0)
    pub selectivity: f64,
    /// Whether index was actually used
    pub used: bool,
    /// Reason if index was not used
    pub skip_reason: Option<String>,
}

/// Graph planner statistics used for cost estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPlannerStats {
    /// Total node count in graph
    pub total_nodes: usize,
    /// Total edge count in graph
    pub total_edges: usize,
    /// Average node degree
    pub avg_node_degree: f64,
    /// Label selectivity map
    pub label_selectivity: HashMap<String, usize>,
    /// Property cardinality estimates
    pub property_cardinality: HashMap<String, usize>,
}

/// Join cost estimation for hybrid queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JoinExplainCostEstimate {
    /// Join algorithm used
    pub join_algorithm: String,
    /// Estimated cost of the join
    pub estimated_cost: f64,
    /// Left input cardinality estimate
    pub left_cardinality: usize,
    /// Right input cardinality estimate  
    pub right_cardinality: usize,
    /// Join selectivity estimate
    pub join_selectivity: f64,
    /// Memory requirements in MB
    pub memory_mb: f64,
    /// Expected output cardinality
    pub output_cardinality: usize,
    /// Join key information
    pub join_keys: Vec<String>,
}

/// ANALYZE metrics from actual query execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyzeMetrics {
    /// Actual execution time in milliseconds
    pub actual_execution_time_ms: u64,
    /// Actual rows returned
    pub actual_rows: usize,
    /// Actual memory usage in MB
    pub actual_memory_mb: f64,
    /// Cache hit rates
    pub cache_statistics: ExplainCacheStatistics,
    /// I/O statistics
    pub io_statistics: IOStatistics,
    /// Operator timing breakdown
    pub operator_timings: Vec<OperatorTiming>,
    /// Resource utilization
    pub resource_usage: ExplainResourceUsage,
}

/// Cache statistics for ANALYZE
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplainCacheStatistics {
    /// Vector cache hit rate
    pub vector_cache_hit_rate: f64,
    /// Graph cache hit rate  
    pub graph_cache_hit_rate: f64,
    /// Plan cache hit
    pub plan_cache_hit: bool,
    /// Total cache requests
    pub total_cache_requests: usize,
    /// Total cache hits
    pub total_cache_hits: usize,
}

/// I/O statistics for ANALYZE
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IOStatistics {
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Number of disk seeks
    pub disk_seeks: usize,
    /// Files accessed
    pub files_accessed: usize,
    /// Average I/O latency in microseconds
    pub avg_io_latency_us: f64,
}

/// Individual operator timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorTiming {
    /// Operator name
    pub operator: String,
    /// Time spent in milliseconds
    pub time_ms: u64,
    /// Rows processed
    pub rows_processed: usize,
    /// Memory used in MB
    pub memory_mb: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplainResourceUsage {
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// CPU time in milliseconds
    pub cpu_time_ms: u64,
    /// Number of threads used
    pub threads_used: usize,
    /// GPU utilization if applicable
    pub gpu_utilization: Option<f64>,
}

impl GraphHints {
    /// Create GraphHints from a graph query plan
    pub fn from_query_plan(plan: &GraphQueryPlan, stats: Option<&GraphStatistics>) -> Self {
        let mut hints = GraphHints::default();

        // Extract information from plan steps
        for step in &plan.steps {
            match &step.step_type {
                PlanStepType::IndexSeek { index_name, .. } => {
                    hints.index_usage.push(GraphIndexUsage {
                        index_name: index_name.clone(),
                        index_type: "label_index".to_string(),
                        selectivity: 1.0 / (step.cost.total_cost + 1.0), // estimate from cost
                        used: true,
                        skip_reason: None,
                    });
                }
                PlanStepType::Traverse {
                    algorithm,
                    max_depth,
                    ..
                } => {
                    hints.traversal_algorithm = Some(format!("{:?}", algorithm));
                    hints.max_depth = Some(*max_depth as u32);
                }
                _ => {}
            }
        }

        // Estimate costs and cardinalities
        hints.estimated_nodes_visited = Some(plan.estimated_result_size);
        hints.estimated_memory_mb = Some(plan.estimated_cost.memory_cost);
        hints.estimated_io_cost = Some(plan.estimated_cost.io_cost);

        // Add graph statistics if available
        if let Some(stats) = stats {
            hints.graph_stats = Some(GraphPlannerStats {
                total_nodes: stats.node_count as usize,
                total_edges: stats.edge_count as usize,
                avg_node_degree: stats.avg_node_degree,
                label_selectivity: stats
                    .label_selectivity
                    .iter()
                    .map(|(k, v)| (k.clone(), *v as usize))
                    .collect(),
                property_cardinality: HashMap::new(), // Deferred: Add property stats
            });
        }

        hints
    }
}

impl JoinExplainCostEstimate {
    /// Create a join cost estimate for vector-graph hybrid queries
    pub fn for_hybrid_join(
        vector_cardinality: usize,
        graph_cardinality: usize,
        join_selectivity: f64,
    ) -> Self {
        let estimated_cost = (vector_cardinality as f64) * (graph_cardinality as f64) * 0.001; // Simple cost model
        let output_cardinality =
            ((vector_cardinality as f64) * (graph_cardinality as f64) * join_selectivity) as usize;
        let memory_mb = ((vector_cardinality + graph_cardinality) as f64 * 0.001).max(1.0); // Rough estimate

        JoinExplainCostEstimate {
            join_algorithm: "hybrid_hash_join".to_string(),
            estimated_cost,
            left_cardinality: vector_cardinality,
            right_cardinality: graph_cardinality,
            join_selectivity,
            memory_mb,
            output_cardinality,
            join_keys: vec!["id".to_string()], // Typical join key
        }
    }
}

impl AnalyzeMetrics {
    /// Create minimal ANALYZE metrics for testing
    pub fn minimal(execution_time_ms: u64, rows: usize) -> Self {
        AnalyzeMetrics {
            actual_execution_time_ms: execution_time_ms,
            actual_rows: rows,
            actual_memory_mb: 1.0,
            cache_statistics: ExplainCacheStatistics::default(),
            io_statistics: IOStatistics::default(),
            operator_timings: vec![],
            resource_usage: ExplainResourceUsage::default(),
        }
    }
}

// ============================================================================
// A4: Enhanced Query Explanation
// ============================================================================

/// Serializable model type for use in enhanced explain plans
/// (Wraps the internal ModelType which doesn't derive Serde traits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplainModelType {
    /// Vector embeddings storage
    Vector,
    /// Semi-structured JSON documents
    Document,
    /// Graph nodes and edges
    Graph,
    /// Relational tables
    Relational,
    /// Observability data: logs, metrics, traces
    Observability,
    /// Time-series data
    TimeSeries,
    /// Event sourcing
    Event,
}

impl From<ModelType> for ExplainModelType {
    fn from(model: ModelType) -> Self {
        match model {
            ModelType::Vector => ExplainModelType::Vector,
            ModelType::Document => ExplainModelType::Document,
            ModelType::Graph => ExplainModelType::Graph,
            ModelType::Relational => ExplainModelType::Relational,
            ModelType::Observability => ExplainModelType::Observability,
            ModelType::TimeSeries => ExplainModelType::TimeSeries,
            ModelType::Event => ExplainModelType::Event,
        }
    }
}

impl From<ExplainModelType> for ModelType {
    fn from(model: ExplainModelType) -> Self {
        match model {
            ExplainModelType::Vector => ModelType::Vector,
            ExplainModelType::Document => ModelType::Document,
            ExplainModelType::Graph => ModelType::Graph,
            ExplainModelType::Relational => ModelType::Relational,
            ExplainModelType::Observability => ModelType::Observability,
            ExplainModelType::TimeSeries => ModelType::TimeSeries,
            ExplainModelType::Event => ModelType::Event,
        }
    }
}

/// Enhanced EXPLAIN plan with detailed debugging information.
///
/// Provides comprehensive query execution plan explanation including:
/// - RL planner decision details (what the planner chose and why)
/// - Optimization rules applied (predicate pushdown, join reordering, etc.)
/// - Estimated costs per model type (vector, graph, document, observability)
/// - Parallelization opportunities identified
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnhancedExplainPlan {
    /// Basic explain plan (orchestration steps, hints, etc.)
    pub base_plan: ExplainPlan,

    /// RL planner decision explanation (if RL planner was used)
    pub rl_planner_decision: Option<RLPlannerExplanation>,

    /// List of optimization rules that were applied to the query
    pub optimization_rules_applied: Vec<OptimizationRule>,

    /// Estimated cost breakdown per data model type
    pub estimated_costs_per_model: HashMap<ExplainModelType, f64>,

    /// Identified parallelization opportunities in the plan
    pub parallelization_opportunities: Vec<ParallelStage>,

    /// Query complexity analysis
    pub complexity_analysis: Option<ComplexityAnalysis>,

    /// Warnings and recommendations for query improvement
    pub warnings: Vec<PlanWarning>,
}

impl EnhancedExplainPlan {
    /// Create a new builder for EnhancedExplainPlan
    pub fn builder() -> EnhancedExplainPlanBuilder {
        EnhancedExplainPlanBuilder::default()
    }

    /// Create from a basic ExplainPlan
    pub fn from_base(base_plan: ExplainPlan) -> Self {
        Self {
            base_plan,
            ..Default::default()
        }
    }

    /// Get total estimated cost across all models
    pub fn total_estimated_cost(&self) -> f64 {
        self.estimated_costs_per_model.values().sum()
    }

    /// Check if the plan involves multiple data models
    pub fn is_cross_model(&self) -> bool {
        self.estimated_costs_per_model.len() > 1
    }

    /// Get the dominant (highest cost) model type
    pub fn dominant_model(&self) -> Option<(ExplainModelType, f64)> {
        self.estimated_costs_per_model
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (*k, *v))
    }

    /// Get parallel execution speedup estimate
    pub fn estimated_parallel_speedup(&self) -> f64 {
        if self.parallelization_opportunities.is_empty() {
            return 1.0;
        }

        // Calculate speedup based on parallel stages
        let max_parallel_work: f64 = self
            .parallelization_opportunities
            .iter()
            .map(|s| s.estimated_speedup)
            .fold(1.0, |acc, s| acc * s);

        max_parallel_work.max(1.0)
    }

    /// Generate a human-readable summary of the plan
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push("=== Enhanced Query Execution Plan ===".to_string());
        lines.push(String::new());

        // RL Planner Decision
        if let Some(ref rl) = self.rl_planner_decision {
            lines.push("RL Planner Decision:".to_string());
            lines.push(format!("  Action: {}", rl.selected_action));
            lines.push(format!("  Confidence: {:.2}%", rl.confidence * 100.0));
            if let Some(ref reason) = rl.selection_reason {
                lines.push(format!("  Reason: {}", reason));
            }
            lines.push(String::new());
        }

        // Optimization Rules
        if !self.optimization_rules_applied.is_empty() {
            lines.push("Optimization Rules Applied:".to_string());
            for rule in &self.optimization_rules_applied {
                lines.push(format!("  - {} ({:?})", rule.rule_name, rule.rule_type));
                if let Some(ref desc) = rule.description {
                    lines.push(format!("    {}", desc));
                }
            }
            lines.push(String::new());
        }

        // Cost per Model
        if !self.estimated_costs_per_model.is_empty() {
            lines.push("Estimated Cost per Model:".to_string());
            let mut costs: Vec<_> = self.estimated_costs_per_model.iter().collect();
            costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (model, cost) in costs {
                lines.push(format!("  {:?}: {:.2}", model, cost));
            }
            lines.push(format!("  Total: {:.2}", self.total_estimated_cost()));
            lines.push(String::new());
        }

        // Parallelization
        if !self.parallelization_opportunities.is_empty() {
            lines.push("Parallelization Opportunities:".to_string());
            for stage in &self.parallelization_opportunities {
                lines.push(format!(
                    "  Stage '{}': {} parallel units, {:.1}x speedup",
                    stage.stage_name, stage.parallel_units, stage.estimated_speedup
                ));
            }
            lines.push(format!(
                "  Total Speedup: {:.1}x",
                self.estimated_parallel_speedup()
            ));
            lines.push(String::new());
        }

        // Warnings
        if !self.warnings.is_empty() {
            lines.push("Warnings:".to_string());
            for warning in &self.warnings {
                lines.push(format!("  [{:?}] {}", warning.severity, warning.message));
                if let Some(ref rec) = warning.recommendation {
                    lines.push(format!("    Recommendation: {}", rec));
                }
            }
        }

        lines.join("\n")
    }
}

/// Builder for EnhancedExplainPlan
#[derive(Debug, Default)]
pub struct EnhancedExplainPlanBuilder {
    plan: EnhancedExplainPlan,
}

impl EnhancedExplainPlanBuilder {
    /// Set the base explain plan
    pub fn base_plan(mut self, plan: ExplainPlan) -> Self {
        self.plan.base_plan = plan;
        self
    }

    /// Set the RL planner explanation
    pub fn with_rl_explanation(mut self, explanation: RLPlannerExplanation) -> Self {
        self.plan.rl_planner_decision = Some(explanation);
        self
    }

    /// Add an optimization rule that was applied
    pub fn with_rule(mut self, rule: OptimizationRule) -> Self {
        self.plan.optimization_rules_applied.push(rule);
        self
    }

    /// Add multiple optimization rules
    pub fn with_rules(mut self, rules: Vec<OptimizationRule>) -> Self {
        self.plan.optimization_rules_applied.extend(rules);
        self
    }

    /// Set cost for a specific model type
    pub fn with_model_cost(mut self, model: ExplainModelType, cost: f64) -> Self {
        self.plan.estimated_costs_per_model.insert(model, cost);
        self
    }

    /// Set cost for a specific model type using ModelType (convenience method)
    pub fn with_model_cost_from_type(mut self, model: ModelType, cost: f64) -> Self {
        self.plan
            .estimated_costs_per_model
            .insert(ExplainModelType::from(model), cost);
        self
    }

    /// Set all model costs at once
    pub fn with_model_costs(mut self, costs: HashMap<ExplainModelType, f64>) -> Self {
        self.plan.estimated_costs_per_model = costs;
        self
    }

    /// Add a parallelization opportunity
    pub fn with_parallel_stage(mut self, stage: ParallelStage) -> Self {
        self.plan.parallelization_opportunities.push(stage);
        self
    }

    /// Add multiple parallelization opportunities
    pub fn with_parallel_stages(mut self, stages: Vec<ParallelStage>) -> Self {
        self.plan.parallelization_opportunities.extend(stages);
        self
    }

    /// Set complexity analysis
    pub fn with_complexity(mut self, analysis: ComplexityAnalysis) -> Self {
        self.plan.complexity_analysis = Some(analysis);
        self
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: PlanWarning) -> Self {
        self.plan.warnings.push(warning);
        self
    }

    /// Build the enhanced explain plan
    pub fn build(self) -> EnhancedExplainPlan {
        self.plan
    }
}

/// Explanation of RL planner decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLPlannerExplanation {
    /// The action selected by the RL planner
    pub selected_action: String,

    /// Confidence score (0.0 to 1.0) in the selected action
    pub confidence: f32,

    /// Human-readable reason for the selection
    pub selection_reason: Option<String>,

    /// Alternative actions that were considered
    pub alternatives_considered: Vec<AlternativeAction>,

    /// State features that influenced the decision
    pub influential_features: Vec<InfluentialFeature>,

    /// Whether exploration or exploitation was used
    pub exploration_mode: ExplorationMode,

    /// Historical performance of this action in similar contexts
    pub historical_performance: Option<HistoricalPerformance>,
}

impl Default for RLPlannerExplanation {
    fn default() -> Self {
        Self {
            selected_action: "Unknown".to_string(),
            confidence: 0.0,
            selection_reason: None,
            alternatives_considered: Vec::new(),
            influential_features: Vec::new(),
            exploration_mode: ExplorationMode::Exploitation,
            historical_performance: None,
        }
    }
}

impl RLPlannerExplanation {
    /// Create new explanation with selected action
    pub fn new(action: impl Into<String>, confidence: f32) -> Self {
        Self {
            selected_action: action.into(),
            confidence,
            ..Default::default()
        }
    }

    /// Builder-style method to add reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.selection_reason = Some(reason.into());
        self
    }

    /// Builder-style method to add alternative
    pub fn with_alternative(mut self, alt: AlternativeAction) -> Self {
        self.alternatives_considered.push(alt);
        self
    }

    /// Builder-style method to add influential feature
    pub fn with_feature(mut self, feature: InfluentialFeature) -> Self {
        self.influential_features.push(feature);
        self
    }

    /// Builder-style method to set exploration mode
    pub fn with_exploration_mode(mut self, mode: ExplorationMode) -> Self {
        self.exploration_mode = mode;
        self
    }

    /// Builder-style method to add historical performance
    pub fn with_history(mut self, history: HistoricalPerformance) -> Self {
        self.historical_performance = Some(history);
        self
    }
}

/// Alternative action that was considered by the RL planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeAction {
    /// Action name/description
    pub action: String,
    /// Expected reward/score for this action
    pub expected_reward: f32,
    /// Reason it was not selected
    pub rejection_reason: Option<String>,
}

impl AlternativeAction {
    /// Create new alternative action
    pub fn new(action: impl Into<String>, expected_reward: f32) -> Self {
        Self {
            action: action.into(),
            expected_reward,
            rejection_reason: None,
        }
    }

    /// Builder-style method to add rejection reason
    pub fn with_rejection_reason(mut self, reason: impl Into<String>) -> Self {
        self.rejection_reason = Some(reason.into());
        self
    }
}

/// Feature that influenced the RL planner decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluentialFeature {
    /// Feature name
    pub feature_name: String,
    /// Feature value
    pub value: f64,
    /// Influence weight (positive = favored action, negative = disfavored)
    pub influence_weight: f64,
    /// Human-readable interpretation
    pub interpretation: Option<String>,
}

impl InfluentialFeature {
    /// Create new influential feature
    pub fn new(name: impl Into<String>, value: f64, weight: f64) -> Self {
        Self {
            feature_name: name.into(),
            value,
            influence_weight: weight,
            interpretation: None,
        }
    }

    /// Builder-style method to add interpretation
    pub fn with_interpretation(mut self, interp: impl Into<String>) -> Self {
        self.interpretation = Some(interp.into());
        self
    }
}

/// Exploration vs exploitation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplorationMode {
    /// Selected action based on learned policy (best expected reward)
    Exploitation,
    /// Selected action for exploration (trying less-visited actions)
    Exploration,
    /// Thompson sampling (probabilistic selection)
    ThompsonSampling,
    /// Epsilon-greedy with random selection
    EpsilonGreedy,
}

/// Historical performance statistics for an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPerformance {
    /// Number of times this action was selected in similar contexts
    pub execution_count: u64,
    /// Average reward achieved
    pub average_reward: f32,
    /// Average latency in milliseconds
    pub average_latency_ms: f64,
    /// Average recall achieved
    pub average_recall: f32,
    /// Success rate (fraction of executions that met targets)
    pub success_rate: f32,
}

impl HistoricalPerformance {
    /// Create new historical performance record
    pub fn new(count: u64, avg_reward: f32) -> Self {
        Self {
            execution_count: count,
            average_reward: avg_reward,
            average_latency_ms: 0.0,
            average_recall: 0.0,
            success_rate: 0.0,
        }
    }

    /// Builder-style method to add latency
    pub fn with_latency(mut self, latency_ms: f64) -> Self {
        self.average_latency_ms = latency_ms;
        self
    }

    /// Builder-style method to add recall
    pub fn with_recall(mut self, recall: f32) -> Self {
        self.average_recall = recall;
        self
    }

    /// Builder-style method to add success rate
    pub fn with_success_rate(mut self, rate: f32) -> Self {
        self.success_rate = rate;
        self
    }
}

/// Optimization rule that was applied to the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRule {
    /// Name of the optimization rule
    pub rule_name: String,
    /// Type/category of the rule
    pub rule_type: OptimizationRuleType,
    /// Description of what the rule did
    pub description: Option<String>,
    /// Estimated cost reduction from applying this rule
    pub estimated_cost_reduction: Option<f64>,
    /// Nodes in the plan that were affected
    pub affected_nodes: Vec<String>,
}

impl OptimizationRule {
    /// Create new optimization rule
    pub fn new(name: impl Into<String>, rule_type: OptimizationRuleType) -> Self {
        Self {
            rule_name: name.into(),
            rule_type,
            description: None,
            estimated_cost_reduction: None,
            affected_nodes: Vec::new(),
        }
    }

    /// Create predicate pushdown rule
    pub fn predicate_pushdown(description: impl Into<String>) -> Self {
        Self {
            rule_name: "PredicatePushdown".to_string(),
            rule_type: OptimizationRuleType::PredicatePushdown,
            description: Some(description.into()),
            estimated_cost_reduction: None,
            affected_nodes: Vec::new(),
        }
    }

    /// Create join reordering rule
    pub fn join_reordering(description: impl Into<String>) -> Self {
        Self {
            rule_name: "JoinReordering".to_string(),
            rule_type: OptimizationRuleType::JoinReordering,
            description: Some(description.into()),
            estimated_cost_reduction: None,
            affected_nodes: Vec::new(),
        }
    }

    /// Create projection pushdown rule
    pub fn projection_pushdown(description: impl Into<String>) -> Self {
        Self {
            rule_name: "ProjectionPushdown".to_string(),
            rule_type: OptimizationRuleType::ProjectionPushdown,
            description: Some(description.into()),
            estimated_cost_reduction: None,
            affected_nodes: Vec::new(),
        }
    }

    /// Create index selection rule
    pub fn index_selection(description: impl Into<String>) -> Self {
        Self {
            rule_name: "IndexSelection".to_string(),
            rule_type: OptimizationRuleType::IndexSelection,
            description: Some(description.into()),
            estimated_cost_reduction: None,
            affected_nodes: Vec::new(),
        }
    }

    /// Builder-style method to add cost reduction
    pub fn with_cost_reduction(mut self, reduction: f64) -> Self {
        self.estimated_cost_reduction = Some(reduction);
        self
    }

    /// Builder-style method to add affected nodes
    pub fn with_affected_nodes(mut self, nodes: Vec<String>) -> Self {
        self.affected_nodes = nodes;
        self
    }
}

/// Types of optimization rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationRuleType {
    /// Push predicates (filters) closer to data sources
    PredicatePushdown,
    /// Reorder joins for better performance
    JoinReordering,
    /// Push projections to read only needed columns
    ProjectionPushdown,
    /// Select optimal index for query
    IndexSelection,
    /// Eliminate common subexpressions
    CommonSubexpressionElimination,
    /// Rewrite subqueries for better execution
    SubqueryUnnesting,
    /// Apply constant folding
    ConstantFolding,
    /// Remove dead/unreachable code
    DeadCodeElimination,
    /// Combine multiple aggregations
    AggregationPushdown,
    /// Partition pruning based on predicates
    PartitionPruning,
    /// Rewrite for parallelism
    ParallelizationRewrite,
    /// Model-specific optimization (vector, graph, etc.)
    ModelSpecific,
    /// Custom/other optimization rule
    Custom,
}

/// A stage in the query that can be parallelized
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelStage {
    /// Name/identifier of the stage
    pub stage_name: String,
    /// Node IDs in this parallel stage
    pub node_ids: Vec<usize>,
    /// Number of parallel execution units
    pub parallel_units: usize,
    /// Type of parallelism
    pub parallelism_type: ParallelismType,
    /// Estimated speedup factor from parallelization
    pub estimated_speedup: f64,
    /// Data dependencies that limit parallelism
    pub dependencies: Vec<String>,
    /// Resource requirements per unit
    pub resource_per_unit: Option<ExplainResourceRequirements>,
}

impl ParallelStage {
    /// Create new parallel stage
    pub fn new(name: impl Into<String>, units: usize, parallelism: ParallelismType) -> Self {
        Self {
            stage_name: name.into(),
            node_ids: Vec::new(),
            parallel_units: units,
            parallelism_type: parallelism,
            estimated_speedup: units as f64 * 0.8, // Assume 80% efficiency by default
            dependencies: Vec::new(),
            resource_per_unit: None,
        }
    }

    /// Builder-style method to add node IDs
    pub fn with_nodes(mut self, nodes: Vec<usize>) -> Self {
        self.node_ids = nodes;
        self
    }

    /// Builder-style method to set speedup
    pub fn with_speedup(mut self, speedup: f64) -> Self {
        self.estimated_speedup = speedup;
        self
    }

    /// Builder-style method to add dependencies
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Builder-style method to add resource requirements
    pub fn with_resources(mut self, resources: ExplainResourceRequirements) -> Self {
        self.resource_per_unit = Some(resources);
        self
    }
}

/// Type of parallelism used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismType {
    /// Data parallelism (same operation on different data partitions)
    DataParallel,
    /// Pipeline parallelism (different stages run concurrently)
    Pipeline,
    /// Task parallelism (independent operations run concurrently)
    TaskParallel,
    /// Intra-operator parallelism (within a single operator)
    IntraOperator,
    /// SIMD vectorization
    Simd,
}

/// Backwards-compat alias for [`ExplainResourceRequirements`].
pub type ResourceRequirements = ExplainResourceRequirements;

/// Resource requirements for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainResourceRequirements {
    /// Memory requirement in MB
    pub memory_mb: f64,
    /// CPU cores needed
    pub cpu_cores: f32,
    /// Expected I/O bandwidth in MB/s
    pub io_bandwidth_mbps: f64,
    /// GPU memory if applicable
    pub gpu_memory_mb: Option<f64>,
}

impl Default for ExplainResourceRequirements {
    fn default() -> Self {
        Self {
            memory_mb: 16.0,
            cpu_cores: 1.0,
            io_bandwidth_mbps: 100.0,
            gpu_memory_mb: None,
        }
    }
}

/// Query complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysis {
    /// Estimated time complexity (e.g., "O(n log n)")
    pub time_complexity: String,
    /// Estimated space complexity
    pub space_complexity: String,
    /// Number of data models involved
    pub models_involved: usize,
    /// Number of joins in the query
    pub join_count: usize,
    /// Maximum join depth
    pub max_join_depth: usize,
    /// Total number of plan nodes
    pub total_plan_nodes: usize,
    /// Whether the query requires sorting
    pub requires_sorting: bool,
    /// Whether the query requires aggregation
    pub requires_aggregation: bool,
    /// Estimated selectivity of all filters combined
    pub combined_selectivity: f64,
}

impl Default for ComplexityAnalysis {
    fn default() -> Self {
        Self {
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(1)".to_string(),
            models_involved: 1,
            join_count: 0,
            max_join_depth: 0,
            total_plan_nodes: 1,
            requires_sorting: false,
            requires_aggregation: false,
            combined_selectivity: 1.0,
        }
    }
}

/// Warning or recommendation about the query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanWarning {
    /// Warning severity
    pub severity: WarningSeverity,
    /// Warning message
    pub message: String,
    /// Optional recommendation for improvement
    pub recommendation: Option<String>,
    /// Code for programmatic handling
    pub warning_code: String,
}

impl PlanWarning {
    /// Create new warning
    pub fn new(severity: WarningSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            recommendation: None,
            warning_code: "UNKNOWN".to_string(),
        }
    }

    /// Create info-level warning
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(WarningSeverity::Info, message)
    }

    /// Create low-severity warning
    pub fn low(message: impl Into<String>) -> Self {
        Self::new(WarningSeverity::Low, message)
    }

    /// Create medium-severity warning
    pub fn medium(message: impl Into<String>) -> Self {
        Self::new(WarningSeverity::Medium, message)
    }

    /// Create high-severity warning
    pub fn high(message: impl Into<String>) -> Self {
        Self::new(WarningSeverity::High, message)
    }

    /// Builder-style method to add recommendation
    pub fn with_recommendation(mut self, rec: impl Into<String>) -> Self {
        self.recommendation = Some(rec.into());
        self
    }

    /// Builder-style method to set warning code
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.warning_code = code.into();
        self
    }
}

/// Warning severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    /// Informational only
    Info,
    /// Minor issue, query will work but could be improved
    Low,
    /// Moderate issue, may affect performance
    Medium,
    /// Serious issue, likely to cause poor performance
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::{
        CatalogCompressionRejectedCandidate, CatalogCompressionStatsProfile, CatalogPhysicalFormat,
        CatalogProjection, CatalogProjectionKind, CatalogStorageLayout, CatalogStorageLayoutKind,
        CatalogTableSchema, RelationalCapabilities,
    };

    #[test]
    fn test_enhanced_explain_plan_builder() {
        let plan = EnhancedExplainPlan::builder()
            .base_plan(ExplainPlan::with_steps(vec!["Step 1".to_string()]))
            .with_model_cost(ExplainModelType::Vector, 10.0)
            .with_model_cost(ExplainModelType::Graph, 5.0)
            .with_rule(OptimizationRule::predicate_pushdown(
                "Pushed filter to vector scan",
            ))
            .with_parallel_stage(ParallelStage::new(
                "VectorSearch",
                4,
                ParallelismType::DataParallel,
            ))
            .build();

        assert_eq!(plan.total_estimated_cost(), 15.0);
        assert!(plan.is_cross_model());
        assert_eq!(plan.optimization_rules_applied.len(), 1);
        assert_eq!(plan.parallelization_opportunities.len(), 1);
    }

    #[test]
    fn vector_hints_from_search_plan_hints_preserves_ann_reason() {
        let hints = crate::services::operations::vectors::SearchPlanHints {
            ann_filtering_mode: Some("PostFilter".to_string()),
            ann_filtering_selectivity: Some(0.9),
            ann_filtering_selectivity_source: Some("cost_analysis".to_string()),
            ..Default::default()
        };

        let vector_hints = VectorHints::from(&hints);

        assert_eq!(
            vector_hints.ann_filtering_mode.as_deref(),
            Some("PostFilter")
        );
        assert_eq!(vector_hints.ann_filtering_selectivity, Some(0.9));
        assert_eq!(
            vector_hints.ann_filtering_selectivity_source.as_deref(),
            Some("cost_analysis")
        );
        assert_eq!(
            vector_hints.ann_mode_reason.as_deref(),
            Some("PostFilter selected from cost_analysis selectivity=0.900000")
        );
    }

    #[test]
    fn vector_hints_from_search_plan_hints_propagates_turboquant_payload() {
        // Phase J wire test: a `SearchPlanHints` carrying a TurboQuant
        // JSON payload (built by `score_turboquant` in the caller-side
        // shim) must propagate that payload into the wire-facing
        // `VectorHints.turboquant` slot unchanged. Without this
        // propagation, the EXPLAIN payload would be silently dropped
        // before reaching REST / gRPC / Arrow Flight / pgwire.
        let payload = serde_json::json!({
            "quantization": "turboquant_4bit",
            "calibration_mode": "tq_plus",
            "rotation_seed": "1234567",
            "encoded_epoch": 7,
            "mask_pushed_to_kernel": true,
            "kernel_arch": "scalar",
            "blocks_skipped_by_mask": 42,
            "length_renorm_applied": true,
        });
        let hints = crate::services::operations::vectors::SearchPlanHints {
            turboquant: Some(payload.clone()),
            ..Default::default()
        };
        let vh = VectorHints::from(&hints);
        assert_eq!(vh.turboquant.as_ref(), Some(&payload));
    }

    #[test]
    fn vector_hints_from_search_plan_hints_without_turboquant_yields_none() {
        // The default path (no TurboQuant) must NOT synthesize a stale
        // turboquant payload — that would leak a misleading "TurboQuant
        // ran" signal into the EXPLAIN wire shape.
        let hints = crate::services::operations::vectors::SearchPlanHints::default();
        let vh = VectorHints::from(&hints);
        assert!(vh.turboquant.is_none());
    }

    #[test]
    fn vector_hints_default_skips_turboquant_field_in_serialized_wire_shape() {
        // The `turboquant` field is `Option<serde_json::Value>` with
        // `skip_serializing_if = "Option::is_none"`. Default-constructed
        // VectorHints must not surface a stale-null `turboquant` key —
        // EXPLAIN consumers that read JSON would otherwise infer
        // "TurboQuant ran but had nothing to say" from a `null`. Lock
        // the shape with a literal-presence assertion.
        let h = VectorHints::default();
        let s = serde_json::to_string(&h).expect("VectorHints serializes");
        assert!(
            !s.contains("\"turboquant\""),
            "wire shape leaked `turboquant` key: {s}",
        );
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn vector_hints_with_turboquant_hints_serializes_full_payload() {
        // Build a TurboQuantExplainHints, run it through the builder,
        // and verify the EXPLAIN JSON carries every load-bearing field
        // under the `turboquant` key. This is the wire-contract test
        // for ADR-004 EXPLAIN integration of TurboQuant — every protocol
        // (REST / gRPC / Arrow Flight / pgwire) reads the same key
        // shape.
        use crate::index::turboquant_bridge::{KernelArch, TurboQuantExplainHints};
        use proximadb_quantization_types::CalibrationMode;
        use proximadb_vector::quantization::turboquant::TurboQuantStore;

        let store = TurboQuantStore::new(64, 4, CalibrationMode::Identity, 0xabcd).unwrap();
        let tq = TurboQuantExplainHints::for_search(&store)
            .with_encoded_epoch(11)
            .with_current_epoch(11)
            .with_mask_pushed(true)
            .with_kernel_arch(KernelArch::Scalar)
            .with_blocks_skipped(42)
            .with_candidate_set_size(128)
            .with_n_vectors_scanned(2048);

        let hints = VectorHints::default().with_turboquant_hints(&tq);

        // Top-level mirror: legacy consumers reading just
        // `quantization_level` get the right value.
        assert_eq!(hints.quantization_level.as_deref(), Some("turboquant_4bit"));

        // Full payload under the new key.
        let v = serde_json::to_value(&hints).expect("VectorHints serializes");
        let tq_obj = v
            .get("turboquant")
            .expect("turboquant key present")
            .as_object()
            .expect("turboquant is a JSON object");
        assert_eq!(
            tq_obj.get("quantization").and_then(|v| v.as_str()),
            Some("turboquant_4bit")
        );
        assert_eq!(
            tq_obj.get("calibration_mode").and_then(|v| v.as_str()),
            Some("identity")
        );
        assert_eq!(
            tq_obj.get("encoded_epoch").and_then(|v| v.as_u64()),
            Some(11)
        );
        assert_eq!(
            tq_obj.get("current_epoch").and_then(|v| v.as_u64()),
            Some(11)
        );
        assert_eq!(
            tq_obj
                .get("mask_pushed_to_kernel")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            tq_obj.get("kernel_arch").and_then(|v| v.as_str()),
            Some("scalar")
        );
        assert_eq!(
            tq_obj
                .get("blocks_skipped_by_mask")
                .and_then(|v| v.as_u64()),
            Some(42)
        );
        assert_eq!(
            tq_obj
                .get("length_renorm_applied")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            tq_obj.get("candidate_set_size").and_then(|v| v.as_u64()),
            Some(128)
        );
        assert_eq!(
            tq_obj.get("n_vectors_scanned").and_then(|v| v.as_u64()),
            Some(2048)
        );
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn vector_hints_with_turboquant_hints_round_trips_through_serde() {
        // Round-trip through to_string + from_str must preserve every
        // field — otherwise a REST → server → REST shape change is
        // silent. The contract is locked.
        use crate::index::turboquant_bridge::TurboQuantExplainHints;
        use proximadb_quantization_types::CalibrationMode;
        use proximadb_vector::quantization::turboquant::TurboQuantStore;

        let store = TurboQuantStore::new(64, 2, CalibrationMode::Identity, 0x1234).unwrap();
        let tq = TurboQuantExplainHints::for_search(&store)
            .with_encoded_epoch(1)
            .with_mask_pushed(false)
            .with_blocks_skipped(0);
        let hints = VectorHints::default().with_turboquant_hints(&tq);
        let s = serde_json::to_string(&hints).expect("serialize");
        let back: VectorHints = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back.quantization_level, hints.quantization_level);
        assert_eq!(back.turboquant, hints.turboquant);
    }

    #[test]
    fn test_rl_planner_explanation() {
        let explanation = RLPlannerExplanation::new("HNSW(ef=100)", 0.85)
            .with_reason("Best historical performance")
            .with_exploration_mode(ExplorationMode::Exploitation)
            .with_alternative(
                AlternativeAction::new("IVF(nprobe=16)", 0.75)
                    .with_rejection_reason("Lower expected reward"),
            )
            .with_feature(
                InfluentialFeature::new("collection_size", 0.8, 0.6)
                    .with_interpretation("Large collection favors HNSW"),
            );

        assert_eq!(explanation.confidence, 0.85);
        assert_eq!(explanation.alternatives_considered.len(), 1);
        assert_eq!(explanation.influential_features.len(), 1);
        assert_eq!(explanation.exploration_mode, ExplorationMode::Exploitation);
    }

    #[test]
    fn test_optimization_rule_creation() {
        let rule = OptimizationRule::predicate_pushdown("Filter moved to scan")
            .with_cost_reduction(50.0)
            .with_affected_nodes(vec!["scan_1".to_string(), "filter_1".to_string()]);

        assert_eq!(rule.rule_type, OptimizationRuleType::PredicatePushdown);
        assert_eq!(rule.estimated_cost_reduction, Some(50.0));
        assert_eq!(rule.affected_nodes.len(), 2);
    }

    #[test]
    fn test_parallel_stage() {
        let stage = ParallelStage::new("HashJoin", 8, ParallelismType::DataParallel)
            .with_nodes(vec![1, 2, 3])
            .with_speedup(6.4)
            .with_dependencies(vec!["scan_1".to_string()]);

        assert_eq!(stage.parallel_units, 8);
        assert_eq!(stage.estimated_speedup, 6.4);
        assert_eq!(stage.node_ids.len(), 3);
        assert_eq!(stage.dependencies.len(), 1);
    }

    #[test]
    fn test_plan_warning() {
        let warning = PlanWarning::high("Missing index on column 'id'")
            .with_recommendation("Create index on 'id' for better performance")
            .with_code("MISSING_INDEX");

        assert_eq!(warning.severity, WarningSeverity::High);
        assert!(warning.recommendation.is_some());
        assert_eq!(warning.warning_code, "MISSING_INDEX");
    }

    #[test]
    fn test_explain_model_type_conversion() {
        // Test conversion from ModelType to ExplainModelType
        assert_eq!(
            ExplainModelType::from(ModelType::Vector),
            ExplainModelType::Vector
        );
        assert_eq!(
            ExplainModelType::from(ModelType::Graph),
            ExplainModelType::Graph
        );
        assert_eq!(
            ExplainModelType::from(ModelType::Document),
            ExplainModelType::Document
        );

        // Test conversion from ExplainModelType to ModelType
        assert_eq!(ModelType::from(ExplainModelType::Vector), ModelType::Vector);
    }

    #[test]
    fn test_enhanced_explain_plan_summary() {
        let plan = EnhancedExplainPlan::builder()
            .with_rl_explanation(RLPlannerExplanation::new("HNSW(ef=100)", 0.9))
            .with_model_cost(ExplainModelType::Vector, 10.0)
            .with_rule(OptimizationRule::predicate_pushdown("Filter pushed"))
            .with_parallel_stage(ParallelStage::new(
                "VectorSearch",
                4,
                ParallelismType::DataParallel,
            ))
            .with_warning(PlanWarning::info("Query may benefit from caching"))
            .build();

        let summary = plan.summary();
        assert!(summary.contains("Enhanced Query Execution Plan"));
        assert!(summary.contains("RL Planner Decision"));
        assert!(summary.contains("HNSW(ef=100)"));
        assert!(summary.contains("Optimization Rules Applied"));
        assert!(summary.contains("Estimated Cost per Model"));
        assert!(summary.contains("Parallelization Opportunities"));
        assert!(summary.contains("Warnings"));
    }

    #[test]
    fn test_dominant_model() {
        let plan = EnhancedExplainPlan::builder()
            .with_model_cost(ExplainModelType::Vector, 100.0)
            .with_model_cost(ExplainModelType::Graph, 50.0)
            .with_model_cost(ExplainModelType::Document, 25.0)
            .build();

        let dominant = plan.dominant_model();
        assert!(dominant.is_some());
        let (model, cost) = dominant.unwrap();
        assert_eq!(model, ExplainModelType::Vector);
        assert_eq!(cost, 100.0);
    }

    #[test]
    fn test_historical_performance() {
        let history = HistoricalPerformance::new(100, 0.85)
            .with_latency(10.5)
            .with_recall(0.98)
            .with_success_rate(0.95);

        assert_eq!(history.execution_count, 100);
        assert_eq!(history.average_reward, 0.85);
        assert_eq!(history.average_latency_ms, 10.5);
        assert_eq!(history.average_recall, 0.98);
        assert_eq!(history.success_rate, 0.95);
    }

    #[test]
    fn test_complexity_analysis() {
        let complexity = ComplexityAnalysis {
            time_complexity: "O(n log n)".to_string(),
            space_complexity: "O(n)".to_string(),
            models_involved: 3,
            join_count: 2,
            max_join_depth: 2,
            total_plan_nodes: 7,
            requires_sorting: true,
            requires_aggregation: false,
            combined_selectivity: 0.15,
        };

        assert_eq!(complexity.time_complexity, "O(n log n)");
        assert_eq!(complexity.models_involved, 3);
        assert!(complexity.requires_sorting);
    }

    #[test]
    fn test_serialization() {
        let plan = EnhancedExplainPlan::builder()
            .with_model_cost(ExplainModelType::Vector, 10.0)
            .with_rule(OptimizationRule::predicate_pushdown("Test"))
            .build();

        // Serialize to JSON
        let json = serde_json::to_string(&plan).expect("Failed to serialize");
        assert!(json.contains("Vector"));
        assert!(json.contains("PredicatePushdown"));

        // Deserialize from JSON
        let deserialized: EnhancedExplainPlan =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.optimization_rules_applied.len(), 1);
    }

    #[test]
    fn test_explain_storage_authority_from_catalog_metadata() {
        let mut parquet_lake = CatalogStorageLayout::external_authoritative(
            "iceberg_lake",
            CatalogPhysicalFormat::Iceberg,
            "s3://bucket/table",
        );
        parquet_lake.snapshot_semantics = Some("iceberg-snapshot".to_string());

        let mut vector_projection = CatalogProjection::rebuildable(
            "semantic_ann",
            CatalogProjectionKind::VectorAnn,
            "primary",
        );
        vector_projection.lossy = true;

        let schema = CatalogTableSchema::new("events")
            .with_storage_layout(CatalogStorageLayout::internal(
                "pax_hot",
                CatalogStorageLayoutKind::Pax,
            ))
            .with_storage_layout(parquet_lake)
            .with_projection(vector_projection)
            .with_compression_stats_profile(
                CatalogCompressionStatsProfile::new(
                    "bench/vector/base_xor",
                    "VectorBaseXorEntropy",
                    1024,
                    256,
                    128,
                    true,
                )
                .with_layout_name("pax_hot")
                .with_projection_id("semantic_ann"),
            )
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["event_id".to_string()],
                transaction_profile: Some("mvcc".to_string()),
                ..Default::default()
            });

        let authority = StorageAuthorityExplanation::from_catalog_table_schema(&schema);

        assert_eq!(authority.layouts.len(), 3);
        assert_eq!(authority.projections.len(), 1);
        assert_eq!(authority.compression_profiles.len(), 1);
        assert_eq!(
            authority.compression_profiles[0].selected_scheme,
            "VectorBaseXorEntropy"
        );
        assert_eq!(authority.compression_profiles[0].bytes_per_value(), 2.0);
        assert!(!authority.policy_safe_inside_proxima());
        assert!(authority.fallback_behavior.contains("canonical records"));
        assert!(authority.relational_capabilities.has_enforced_semantics);

        let plan = ExplainPlan::new().with_storage_authority(authority);
        assert!(plan.storage_authority.is_some());
    }

    #[test]
    fn test_explain_storage_authority_from_plan_context() {
        use crate::query::multimodal::plan::{
            PlanContext, ResolvedAuthorityMode, ResolvedObjectContext, ResolvedProjectionContext,
            ResolvedStorageLayoutContext,
        };

        let mut object =
            ResolvedObjectContext::internal_canonical("vectors", "vector", "default.vectors");
        object.storage_layouts.push(ResolvedStorageLayoutContext {
            name: "pax_hot".to_string(),
            authority: ResolvedAuthorityMode::InternalCanonical,
            layout_kind: "Pax".to_string(),
            physical_format: "ProximaBlock".to_string(),
            write_mode: "Mutable".to_string(),
            location: None,
            snapshot_semantics: Some("mvcc".to_string()),
            policy_enforced_in_proxima: true,
            lossy_type_mappings: Vec::new(),
        });
        object.projections.push(ResolvedProjectionContext {
            name: "vectors_hnsw".to_string(),
            kind: "VectorAnn".to_string(),
            physical_format: "ProximaBlock".to_string(),
            rebuild_source: "pax_hot".to_string(),
            freshness: "Lazy".to_string(),
            max_lag_ms: None,
            rebuildable: true,
            lossy: false,
            support_status: "experimental".to_string(),
            freshness_state: Some("Fresh".to_string()),
            rebuild_rto_seconds_per_10gb: Some(45.0),
        });
        object.compression_stats_profiles.push(
            crate::query::multimodal::plan::ResolvedCompressionStatsProfileContext {
                profile_id: "bench/vector/base_xor".to_string(),
                layout_name: Some("pax_hot".to_string()),
                projection_id: Some("vectors_hnsw".to_string()),
                selected_scheme: "VectorBaseXorEntropy".to_string(),
                raw_bytes: 1024,
                encoded_bytes: 256,
                value_count: 128,
                measured_ratio: 4.0,
                exact_reconstruction: true,
                encode_cpu_ms_per_block: None,
                decode_ns_per_value: Some(12.0),
                rejected_candidates: Vec::new(),
            },
        );

        let mut context = PlanContext::default();
        context.resolved_objects.push(object);

        let authority = StorageAuthorityExplanation::from_plan_context(&context)
            .expect("resolved object context should produce EXPLAIN authority metadata");

        assert_eq!(authority.layouts.len(), 1);
        assert_eq!(authority.layouts[0].layout_kind, "Pax");
        assert_eq!(authority.projections[0].kind, "VectorAnn");
        assert_eq!(authority.compression_profiles[0].measured_ratio, 4.0);
        assert!(authority.compression_profiles[0].exact_reconstruction);
        assert!(authority.policy_safe_inside_proxima());
    }

    #[test]
    fn test_explain_compression_profiles_preserve_rejected_candidates() {
        let mut profile = CatalogCompressionStatsProfile::new(
            "bench/json/path_dictionary",
            "Dictionary",
            4096,
            1024,
            256,
            true,
        )
        .with_layout_name("json_shape_order")
        .with_projection_id("json_paths");
        profile
            .rejected_candidates
            .push(CatalogCompressionRejectedCandidate {
                scheme: "Raw".to_string(),
                reason: "CompressionTargetMiss".to_string(),
                expected_ratio: Some(1.0),
            });

        let explanation = CompressionProfileExplanation::from(&profile);

        assert_eq!(explanation.profile_id, "bench/json/path_dictionary");
        assert_eq!(explanation.measured_ratio, 4.0);
        assert_eq!(explanation.rejected_candidates.len(), 1);
        assert_eq!(
            explanation.rejected_candidates[0].reason,
            "CompressionTargetMiss"
        );
    }

    #[test]
    fn test_explain_storage_authority_from_plan_context_external_boundary() {
        use crate::query::multimodal::plan::{
            PlanContext, ResolvedAuthorityMode, ResolvedObjectContext, ResolvedStorageLayoutContext,
        };

        let mut object =
            ResolvedObjectContext::internal_canonical("lake_docs", "document", "lake.docs");
        object.authority = ResolvedAuthorityMode::ExternalAuthoritative;
        object.external_policy_boundary = true;
        object.storage_layouts.push(ResolvedStorageLayoutContext {
            name: "iceberg".to_string(),
            authority: ResolvedAuthorityMode::ExternalAuthoritative,
            layout_kind: "ExternalTable".to_string(),
            physical_format: "Iceberg".to_string(),
            write_mode: "ExternalRefresh".to_string(),
            location: Some("s3://warehouse/docs".to_string()),
            snapshot_semantics: Some("iceberg-snapshot".to_string()),
            policy_enforced_in_proxima: false,
            lossy_type_mappings: vec!["timestamp_tz".to_string()],
        });

        let mut context = PlanContext::default();
        context.resolved_objects.push(object);

        let authority = StorageAuthorityExplanation::from_plan_context(&context)
            .expect("external resolved object context should produce EXPLAIN metadata");

        assert!(!authority.policy_safe_inside_proxima());
        assert!(authority.fallback_behavior.contains("policy/RLS"));
        assert_eq!(
            authority.layouts[0].lossy_type_mappings,
            vec!["timestamp_tz".to_string()]
        );
    }

    // ========================================================================
    // COST-BASED OPTIMIZER EXPLAIN TESTS
    // ========================================================================

    #[test]
    fn test_cost_estimate_struct() {
        let estimate = ExplainCostEstimate {
            operation: "VectorSearch(products)".to_string(),
            estimated_cost: 3.5,
            estimated_rows: 100,
            estimated_selectivity: None,
            notes: Some("HNSW index used".to_string()),
        };
        assert_eq!(estimate.estimated_rows, 100);
        assert!(estimate.notes.is_some());
    }

    #[test]
    fn test_join_strategy_explanation() {
        let explanation = JoinStrategyExplanation {
            strategy: "HashJoin".to_string(),
            left_rows: 50_000,
            right_rows: 10_000,
            reason: "Both sides exceed 1000 rows".to_string(),
        };
        let json = serde_json::to_string(&explanation).expect("Failed to serialize");
        assert!(json.contains("HashJoin"));
        assert!(json.contains("50000"));
    }

    #[test]
    fn test_fusion_strategy_explanation() {
        let explanation = FusionStrategyExplanation {
            strategy: "Rrf".to_string(),
            reason: "Heterogeneous score scales from vector and graph".to_string(),
        };
        let json = serde_json::to_string(&explanation).expect("Failed to serialize");
        assert!(json.contains("Rrf"));
    }

    #[test]
    fn test_explain_plan_with_cost_breakdown() {
        let plan = ExplainPlan::new()
            .with_cost_breakdown(vec![
                ExplainCostEstimate {
                    operation: "VectorSearch(products)".to_string(),
                    estimated_cost: 3.5,
                    estimated_rows: 100,
                    estimated_selectivity: None,
                    notes: None,
                },
                ExplainCostEstimate {
                    operation: "GraphTraversal(knowledge)".to_string(),
                    estimated_cost: 6.0,
                    estimated_rows: 500,
                    estimated_selectivity: None,
                    notes: Some("depth=3".to_string()),
                },
            ])
            .with_join_strategy(JoinStrategyExplanation {
                strategy: "HashJoin".to_string(),
                left_rows: 100,
                right_rows: 500,
                reason: "Both sides exceed threshold".to_string(),
            })
            .with_fusion_strategy(FusionStrategyExplanation {
                strategy: "Rrf".to_string(),
                reason: "Mixed vector and graph scores".to_string(),
            })
            .with_total_cost(9.5);

        assert!(plan.cost_breakdown.is_some());
        assert_eq!(plan.cost_breakdown.as_ref().map_or(0, |b| b.len()), 2);
        assert!(plan.join_strategy.is_some());
        assert!(plan.fusion_strategy.is_some());
        assert_eq!(plan.estimated_total_cost, Some(9.5));
    }

    #[test]
    fn test_vector_object_economy_explain_from_index_stats() {
        let stats = crate::core::service_types::ServiceIndexStats {
            object_selected_blocks: 8,
            object_pruned_blocks: 24,
            object_estimated_gets: 2,
            object_actual_gets: 2,
            object_estimated_remote_bytes: 128 * 1024,
            object_actual_remote_bytes: 128 * 1024,
            object_overfetch_bytes: 4096,
            ..Default::default()
        };

        let explain = VectorObjectEconomyExplain::from_index_stats(&stats)
            .with_cache_status("cold")
            .with_selected_files(1);

        assert_eq!(explain.route_kind, "vector_object_economy");
        assert_eq!(explain.authority_mode, "projection_over_canonical_records");
        assert_eq!(explain.selected_files, 1);
        assert_eq!(explain.selected_blocks, 8);
        assert_eq!(explain.pruned_blocks, 24);
        assert_eq!(explain.estimated_object_gets, 2);
        assert_eq!(explain.actual_object_gets, Some(2));
        assert_eq!(explain.estimated_remote_bytes, 128 * 1024);
        assert_eq!(explain.overfetch_bytes, 4096);
        assert_eq!(explain.cache_status, "cold");
    }

    #[test]
    fn vector_object_economy_explain_records_wal_delta_scan_metadata() {
        use crate::core::search::VectorFreshnessMode;

        let explain = VectorObjectEconomyExplain::default().record_wal_delta_scan(
            &VectorFreshnessMode::Strong,
            /*current_lsn*/ Some(250),
            /*scanned_records*/ Some(7),
            /*scanned_bytes*/ Some(4096),
        );

        assert_eq!(explain.freshness_mode_used.as_deref(), Some("strong"));
        assert_eq!(explain.current_lsn_at_query, Some(250));
        assert_eq!(explain.wal_delta_records_scanned, Some(7));
        assert_eq!(explain.wal_delta_bytes, Some(4096));
        assert!(explain.wal_delta_searched, "bool flips when scan ran");
    }

    #[test]
    fn vector_object_economy_explain_skipped_scan_leaves_metrics_none() {
        use crate::core::search::VectorFreshnessMode;

        let explain = VectorObjectEconomyExplain::default().record_wal_delta_scan(
            &VectorFreshnessMode::StaleOk,
            /*current_lsn*/ Some(250),
            /*scanned_records*/ None,
            /*scanned_bytes*/ None,
        );

        assert_eq!(explain.freshness_mode_used.as_deref(), Some("stale_ok"));
        assert_eq!(explain.current_lsn_at_query, Some(250));
        assert!(explain.wal_delta_records_scanned.is_none());
        assert!(explain.wal_delta_bytes.is_none());
        assert!(!explain.wal_delta_searched);
    }

    #[test]
    fn vector_object_economy_explain_delta_fields_round_trip_through_json() {
        use crate::core::search::VectorFreshnessMode;

        let explain = VectorObjectEconomyExplain::default().record_wal_delta_scan(
            &VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 5_000,
            },
            Some(42),
            Some(3),
            Some(128),
        );
        let json = serde_json::to_string(&explain).expect("serialize");
        let decoded: VectorObjectEconomyExplain = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(
            decoded.freshness_mode_used.as_deref(),
            Some("bounded_stale")
        );
        assert_eq!(decoded.current_lsn_at_query, Some(42));
        assert_eq!(decoded.wal_delta_records_scanned, Some(3));
        assert_eq!(decoded.wal_delta_bytes, Some(128));
        assert!(decoded.wal_delta_searched);
    }

    #[test]
    fn vector_object_economy_explain_records_directory_degradation_reason() {
        use crate::storage::engines::sst::object_economy_directory::DirectoryLoadStatus;

        let path = "s3://bucket/coll/oedir/v5.bin";
        let loaded = VectorObjectEconomyExplain::default()
            .record_directory_load(&DirectoryLoadStatus::Loaded, path);
        assert!(loaded.rejected_route_reasons.is_empty());

        let missing = VectorObjectEconomyExplain::default()
            .record_directory_load(&DirectoryLoadStatus::Missing, path);
        assert_eq!(missing.rejected_route_reasons.len(), 1);
        assert!(missing.rejected_route_reasons[0].starts_with("object_economy_directory_missing:"));

        let mismatch = VectorObjectEconomyExplain::default().record_directory_load(
            &DirectoryLoadStatus::Mismatch {
                expected_collection: "coll".into(),
                found_collection: "other".into(),
            },
            path,
        );
        assert_eq!(mismatch.rejected_route_reasons.len(), 1);
        assert!(mismatch.rejected_route_reasons[0].contains("collection_mismatch"));
    }

    #[test]
    fn test_explain_plan_carries_vector_object_economy_metadata() {
        let plan = ExplainPlan::new().with_vector_object_economy(VectorObjectEconomyExplain {
            selected_blocks: 3,
            estimated_object_gets: 1,
            ..Default::default()
        });

        let json = serde_json::to_string(&plan).expect("serialize explain");

        assert!(json.contains("vector_object_economy"));
        assert!(json.contains("selected_blocks"));
        assert_eq!(
            plan.vector_object_economy
                .as_ref()
                .map(|explain| explain.selected_blocks),
            Some(3)
        );
    }
}
