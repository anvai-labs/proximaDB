//! MultiModelPlan v1 - Unified Query Contract for Vectorized Cross-Model Execution
//!
//! Also re-exports `compute_plan` — the serializable physical `ComputePlan` / `PlanNode`
//! IR used by the compute scheduler and provider layer.
//!
//! This module implements the MultiModelPlan contract that enables unified query
//! execution across all ProximaDB storage engines (SST, HELIX, NOVA, SWIFT, NOVA, RAPTOR).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    MultiModelPlan v1                        │
//! │  - Unified operator contract for all storage engines        │
//! │  - Zero-copy operations with selection vectors             │
//! │  - Cross-model joins and federated aggregation            │
//! └──────────────────────┬────────────────────────────────────────┘
//!                        │
//!                        ▼
//!     ┌─────────────────────────────────────────┐
//!     │         Operator Pipeline                │
//!     ├─────────────────────────────────────────┤
//!     │ Scan → Filter → Project → Join → Agg    │
//!     │         ↓         ↓         ↓         ↓  │
//!     │    Selection vectors (zero-copy)        │
//!     └─────────────────────────────────────────┘
//!                        │
//!                        ▼
//!     ┌─────────────────────────────────────────┐
//!     │      Storage Engine Dispatch            │
//!     ├─────────────────────────────────────────┤
//!     │ SST │ HELIX │ NOVA │ SWIFT │ NOVA │... │
//!     └─────────────────────────────────────────┘
//! ```
//!
//! ## Key Features
//!
//! - **Unified Operators**: Scan, Filter, Project, Join, Aggregate, Sort, TopK
//! - **Zero-Copy**: Selection vectors enable efficient operator chaining
//! - **Cross-Model**: Join data from different storage engines
//! - **Pushdown**: Filter and projection pushdown to storage engines
//! - **Vectorized**: All operators use Arrow compute kernels
//!
//! ## Design Principles
//!
//! 1. **Composability**: Operators can be combined in any order
//! 2. **Zero-Copy**: Selection vectors avoid row copying
//! 3. **Storage Agnostic**: Same operators work for all engines
//! 4. **Extensible**: Easy to add new operators and optimizations
//! 5. **Serializable**: Plans can be serialized for distributed execution

use anyhow::Result;
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_pipeline_operator::PipelineOperator as ComputeOperator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, trace};

use proximadb_data_model::DataModel;

/// MultiModelPlan v1 - Unified query execution plan
///
/// Represents a complete query execution plan that can be executed
/// across multiple storage engines with vectorized operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModelPlan {
    /// Unique plan identifier
    pub plan_id: String,

    /// Plan version for serialization compatibility
    pub version: u32,

    /// Pipeline of operators to execute
    pub operators: Vec<Operator>,

    /// Execution context and metadata
    pub context: PlanContext,

    /// Optimization hints
    pub hints: PlanHints,
}

impl MultiModelPlan {
    /// Create a new MultiModelPlan
    pub fn new(operators: Vec<Operator>, context: PlanContext) -> Self {
        let plan_id = format!("plan_{}", uuid::Uuid::new_v4());

        Self {
            plan_id,
            version: 1,
            operators,
            context,
            hints: PlanHints::default(),
        }
    }

    /// Get the number of operators in the plan
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Check if the plan is empty
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Add an operator to the end of the pipeline
    pub fn add_operator(&mut self, operator: Operator) {
        self.operators.push(operator);
    }

    /// Get plan statistics
    pub fn stats(&self) -> PlanStats {
        let mut stats = PlanStats::default();

        for operator in &self.operators {
            match operator {
                Operator::Scan { .. } => stats.scan_count += 1,
                Operator::Filter { .. } => stats.filter_count += 1,
                Operator::Project { .. } => stats.project_count += 1,
                Operator::Join { .. } => stats.join_count += 1,
                Operator::Aggregate { .. } => stats.aggregate_count += 1,
                Operator::Sort { .. } => stats.sort_count += 1,
                Operator::TopK { .. } | Operator::VectorTopK { .. } => stats.topk_count += 1,
                Operator::Union { .. } => stats.union_count += 1,
                // Spec §7 extended operators counted generically
                Operator::Limit { .. }
                | Operator::HybridTraverse { .. }
                | Operator::PatternMatch { .. }
                | Operator::CrossModelJoin { .. }
                | Operator::ModulationOp { .. }
                | Operator::MatrixOp { .. }
                | Operator::SemanticJoin { .. }
                | Operator::ModelConvert { .. } => {
                    stats.operator_count += 0; // counted via len() below
                }
            }
        }

        stats.operator_count = self.operators.len();
        stats
    }

    /// Validate the plan for correctness
    pub fn validate(&self) -> Result<PlanValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check if plan is empty
        if self.is_empty() {
            warnings.push("Empty plan - no operators to execute".to_string());
        }

        // Validate operator sequence
        let mut has_scan = false;

        for (idx, operator) in self.operators.iter().enumerate() {
            match operator {
                Operator::Scan { source, .. } => {
                    has_scan = true;
                    if source.is_empty() {
                        errors.push(format!("Operator {}: Scan has empty source", idx));
                    }
                }
                Operator::Join { .. } if !has_scan => {
                    errors.push(format!("Operator {}: Join before any Scan operator", idx));
                }
                Operator::Join { .. } => {}
                Operator::Aggregate { .. } => {
                    // Aggregate validation - currently no specific rules
                }
                Operator::Filter { expression } => {
                    // Validate filter expression
                    if let Err(e) = self.validate_filter_expression(expression) {
                        errors.push(format!("Operator {}: Invalid filter: {}", idx, e));
                    }
                }
                Operator::Project { columns } if columns.is_empty() => {
                    warnings.push(format!("Operator {}: Project with no columns", idx));
                }
                Operator::Project { .. } => {}
                _ => {}
            }
        }

        // Check if plan has at least one scan
        if !has_scan && !self.is_empty() {
            warnings.push("Plan has no Scan operator - may not produce results".to_string());
        }

        // Check for invalid operator sequences
        for (idx, (current, next)) in self
            .operators
            .iter()
            .zip(self.operators.iter().skip(1))
            .enumerate()
        {
            // Check if aggregate comes after join (potentially inefficient)
            if matches!(current, Operator::Join { .. })
                && matches!(next, Operator::Aggregate { .. })
            {
                warnings.push(format!(
                    "Operator {}: Aggregate immediately after Join - consider reordering",
                    idx
                ));
            }
        }

        Ok(PlanValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Validate a filter expression
    fn validate_filter_expression(&self, expression: &FilterExpression) -> Result<()> {
        use proximadb_filter_expression::FilterExpression::*;

        match expression {
            Comparison {
                field,
                operator,
                value,
            } => {
                if field.is_empty() {
                    return Err(anyhow::anyhow!("Filter has empty field name"));
                }
                // Validate value based on operator
                match operator {
                    ComparisonOperator::In => {
                        if let Some(arr) = value.as_array()
                            && arr.is_empty()
                        {
                            return Err(anyhow::anyhow!("IN filter has empty array"));
                        }
                    }
                    ComparisonOperator::Between => {
                        if let Some(arr) = value.as_array()
                            && arr.len() != 2
                        {
                            return Err(anyhow::anyhow!(
                                "BETWEEN filter requires exactly 2 values"
                            ));
                        }
                    }
                    _ => {}
                }
                Ok(())
            }
            And(exprs) | Or(exprs) => {
                for expr in exprs {
                    self.validate_filter_expression(expr)?;
                }
                Ok(())
            }
            Not(expr) => self.validate_filter_expression(expr),
        }
    }

    /// Optimize the plan
    ///
    /// Applies various optimization passes to improve performance:
    /// - Filter pushdown to storage engines
    /// - Projection pushdown to reduce data transfer
    /// - Operator reordering for efficiency
    pub fn optimize(&mut self) -> Result<PlanOptimizationResult> {
        let mut optimizations_applied = Vec::new();
        let original_stats = self.stats();

        // Optimization 1: Filter pushdown
        let filters_pushed = self.pushdown_filters();
        if filters_pushed > 0 {
            optimizations_applied.push(format!("Pushed down {} filters", filters_pushed));
        }

        // Optimization 2: Projection pushdown
        let projections_pushed = self.pushdown_projections();
        if projections_pushed > 0 {
            optimizations_applied.push(format!("Pushed down {} projections", projections_pushed));
        }

        // Optimization 3: Operator reordering
        let reordered = self.reorder_operators()?;
        if reordered {
            optimizations_applied.push("Reordered operators for efficiency".to_string());
        }

        let optimized_stats = self.stats();

        Ok(PlanOptimizationResult {
            optimizations_applied,
            original_stats,
            optimized_stats,
        })
    }

    /// Push down filters to storage engines
    fn pushdown_filters(&mut self) -> usize {
        let pushdown_count = 0;

        // For now, this is a placeholder. In production, you would:
        // 1. Identify filters that can be pushed to Scan operators
        // 2. Move them earlier in the pipeline
        // 3. Combine multiple filters if possible
        // 4. Validate storage engine capabilities

        trace!("Filter pushdown optimization: {} filters", pushdown_count);
        pushdown_count
    }

    /// Push down projections to reduce data transfer
    fn pushdown_projections(&mut self) -> usize {
        let pushdown_count = 0;

        // For now, this is a placeholder. In production, you would:
        // 1. Identify which columns are needed for the final result
        // 2. Trace column usage through all operators
        // 3. Add projection operators early in the pipeline
        // 4. Remove unnecessary columns as early as possible

        trace!(
            "Projection pushdown optimization: {} projections",
            pushdown_count
        );
        pushdown_count
    }

    /// Reorder operators for better performance
    fn reorder_operators(&mut self) -> Result<bool> {
        let reordered = false;

        // For now, this is a placeholder. In production, you would:
        // 1. Analyze operator costs and selectivities
        // 2. Move selective filters early
        // 3. Reorder joins based on table sizes
        // 4. Consider pushing aggregates before joins when possible

        trace!("Operator reordering optimization: {}", reordered);
        Ok(reordered)
    }

    /// Convert to compute pipeline operators for execution
    ///
    /// This bridges the MultiModelPlan to the PipelineExecutor for actual execution.
    /// Federated operators (Join, Aggregate) are handled separately.
    pub fn to_compute_operators(&self) -> Vec<ComputeOperator> {
        self.operators
            .iter()
            .filter_map(|op| self.operator_to_compute(op))
            .collect()
    }

    /// Convert a single operator to compute operator
    fn operator_to_compute(&self, operator: &Operator) -> Option<ComputeOperator> {
        match operator {
            Operator::Scan { source, .. } => Some(ComputeOperator::Scan {
                source: source.clone(),
            }),
            Operator::Filter { expression } => Some(ComputeOperator::Filter {
                expression: expression.clone(),
            }),
            Operator::Project { columns } => Some(ComputeOperator::Project {
                columns: columns.clone(),
            }),
            Operator::Sort {
                column,
                ascending,
                limit,
            } => Some(ComputeOperator::Sort {
                column: column.clone(),
                ascending: *ascending,
                limit: *limit,
            }),
            Operator::TopK { k, sort_column } => Some(ComputeOperator::TopK {
                k: *k,
                sort_column: sort_column.clone(),
            }),
            // Federated and spec §7 operators don't map to low-level compute operators
            Operator::Join { .. }
            | Operator::Aggregate { .. }
            | Operator::Union { .. }
            | Operator::Limit { .. }
            | Operator::VectorTopK { .. }
            | Operator::HybridTraverse { .. }
            | Operator::PatternMatch { .. }
            | Operator::CrossModelJoin { .. }
            | Operator::ModulationOp { .. }
            | Operator::MatrixOp { .. }
            | Operator::SemanticJoin { .. }
            | Operator::ModelConvert { .. } => None,
        }
    }
}

/// MultiModel operator for unified query execution
///
/// Extends the compute PipelineOperator with federated operators
/// like Join and Aggregate that work across storage engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operator {
    /// Scan operator - read data from a data model's service layer
    ///
    /// Engine selection is deferred to execution time via factory.rs config,
    /// not hardcoded per data model. The executor resolves the appropriate
    /// storage engine based on the collection's configuration.
    Scan {
        /// Logical data model (Vector, Document, Graph, Observability)
        /// The executor routes to the correct service layer, not a storage engine.
        data_model: DataModel,
        /// Source identifier (collection ID, graph name, log namespace, etc.)
        source: String,
        /// Optional column projection at scan level
        columns: Option<Vec<String>>,
        /// Optional filter at scan level
        filter: Option<FilterExpression>,
    },

    /// Filter operator - apply filter predicate
    Filter {
        /// Filter expression
        expression: FilterExpression,
    },

    /// Project operator - select specific columns
    Project {
        /// Column names to project
        columns: Vec<String>,
    },

    /// Join operator - combine data from multiple sources
    Join {
        /// Join type (inner, left, right, full)
        join_type: JoinType,
        /// Left side plan
        left_plan: Box<MultiModelPlan>,
        /// Right side plan
        right_plan: Box<MultiModelPlan>,
        /// Join condition
        condition: JoinCondition,
        /// Optional alias for the joined result
        alias: Option<String>,
    },

    /// Aggregate operator - group by and aggregation
    Aggregate {
        /// Group by columns
        group_by: Vec<String>,
        /// Aggregate expressions
        aggregates: Vec<AggregateExpression>,
        /// Optional alias for the result
        alias: Option<String>,
    },

    /// Sort operator - sort by specified column
    Sort {
        /// Sort column name
        column: String,
        /// Ascending or descending
        ascending: bool,
        /// Optional limit on number of results
        limit: Option<usize>,
    },

    /// TopK operator - select top K results
    TopK {
        /// K value
        k: usize,
        /// Sort column for ranking
        sort_column: String,
    },

    /// Union operator - combine results from multiple plans
    Union {
        /// Plans to union
        plans: Vec<MultiModelPlan>,
        /// Remove duplicates
        distinct: bool,
    },

    // ── Spec §7 extended operators ──────────────────────────────────────────
    /// Limit operator (spec §7 — Limit).
    Limit { n: usize, offset: usize },

    /// Filter-aware vector similarity search (spec §7 — VectorTopK, ACORN/NaviX).
    ///
    /// Distinct from the generic `TopK` because it carries an explicit query
    /// vector and distance metric so the executor can route to HNSW indexes.
    VectorTopK {
        query_vector: Vec<f32>,
        k: usize,
        metric: VectorMetric,
        /// Optional predicate pushed into the HNSW navigator (Phase C).
        predicate: Option<FilterExpression>,
    },

    /// Combined graph + vector traversal (spec §7 — HybridTraverse, GredoDB Alg.1).
    HybridTraverse { edge_pattern: EdgePattern },

    /// Cypher-style path pattern match (spec §7 — PatternMatch).
    PatternMatch {
        /// Serialized Cypher path pattern, e.g. `(a)-[:KNOWS]->(b)`.
        pattern: String,
    },

    /// Cross-modality Multi-Stage Hash Join (spec §7 — CrossModelJoin, M2 MSHJ).
    CrossModelJoin {
        left_modality: DataModel,
        right_modality: DataModel,
        condition: JoinCondition,
    },

    /// Flexvec modulation operations (spec §7 — ModulationOp).
    ModulationOp { ops: Vec<String> },

    /// PreVision matrix operation (spec §7 — MatrixOp).
    MatrixOp { op: MatrixOpKind },

    /// NLP-predicate semantic join (spec §7 — SemanticJoin, arXiv:2510.08489).
    SemanticJoin {
        /// Natural language join condition evaluated by the semantic layer.
        nl_predicate: String,
    },

    /// Cross-modality record conversion (spec §7 — ModelConvert).
    ModelConvert {
        source_modality: DataModel,
        target_modality: DataModel,
    },
}

// ── Supporting types for spec §7 operators ──────────────────────────────────

/// Distance metric for vector search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMetric {
    #[default]
    Cosine,
    L2,
    DotProduct,
    Manhattan,
}

/// Edge pattern for graph traversal (HybridTraverse spec §7).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgePattern {
    /// Optional edge type label constraint (e.g. "KNOWS").
    pub edge_type: Option<String>,
    /// Minimum hop count. `0` includes ANN seed nodes in traversal output.
    pub min_hops: u32,
    /// Maximum hop count. `None` = unbounded.
    pub max_hops: Option<u32>,
    /// Direction constraint.
    pub direction: TraversalDirection,
}

impl Default for EdgePattern {
    fn default() -> Self {
        Self {
            edge_type: None,
            min_hops: 1,
            max_hops: Some(1),
            direction: TraversalDirection::Outgoing,
        }
    }
}

/// Graph traversal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Matrix operation kind for PreVision integration (spec §7 — MatrixOp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixOpKind {
    /// Element-wise multiply (Hadamard).
    Hadamard,
    /// Matrix-vector product.
    MatVec,
    /// Outer product.
    Outer,
    /// Cosine similarity matrix.
    CosineSimilarityMatrix,
}

/// Join type specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Join condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinCondition {
    /// Equijoin on single column
    On(String, String),
    /// Equijoin on multiple columns
    OnMultiple(Vec<(String, String)>),
    /// Complex expression join
    Expression(FilterExpression),
}

/// Aggregate expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateExpression {
    /// Aggregate function type
    pub function: AggregateFunction,
    /// Input column
    pub column: String,
    /// Optional alias for the result
    pub alias: Option<String>,
    /// DISTINCT modifier
    pub distinct: bool,
}

/// Aggregate function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    StdDev,
    Variance,
    ArrayAgg,
}

/// Plan execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanContext {
    /// Plan creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Query timeout (milliseconds).
    pub timeout_ms: Option<u64>,
    /// Memory limit (bytes).
    pub memory_limit_bytes: Option<usize>,
    /// Enable distributed execution.
    pub enable_distributed: bool,
    /// Execution priority.
    pub priority: ExecutionPriority,
    /// Catalog-resolved storage authority, layout, and projection metadata for sources in this plan.
    #[serde(default)]
    pub resolved_objects: Vec<ResolvedObjectContext>,
}

impl Default for PlanContext {
    fn default() -> Self {
        Self {
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            memory_limit_bytes: None,
            enable_distributed: false,
            priority: ExecutionPriority::Normal,
            resolved_objects: Vec::new(),
        }
    }
}

/// Source-of-truth mode visible to the planner after xCatalog resolution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResolvedAuthorityMode {
    /// ProximaRecord plus WAL/log/manifest own durable truth.
    InternalCanonical,
    /// An external table/source owns durable truth; ProximaDB maps and governs access.
    ExternalAuthoritative,
    /// Point-in-time import from an external source.
    ImportedSnapshot,
    /// Publication generated from canonical records.
    ExportedPublication,
    /// Rebuildable structure derived from canonical records or events.
    RebuildableProjection,
}

/// Catalog-resolved object metadata carried through lowerers, optimizers, and EXPLAIN.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedObjectContext {
    /// Source name as referenced by the query.
    pub source: String,
    /// Optional alias from the query.
    pub alias: Option<String>,
    /// Logical model requested by the query, e.g. vector, document, graph, observability.
    pub data_model: String,
    /// Resolved catalog namespace/table path.
    pub table_identifier: String,
    /// Source-of-truth role for the resolved object.
    pub authority: ResolvedAuthorityMode,
    /// Physical layouts cataloged for the object.
    pub storage_layouts: Vec<ResolvedStorageLayoutContext>,
    /// Rebuildable projections/access methods cataloged for the object.
    pub projections: Vec<ResolvedProjectionContext>,
    /// Measured codec/layout profiling feedback cataloged for this object.
    #[serde(default)]
    pub compression_stats_profiles: Vec<ResolvedCompressionStatsProfileContext>,
    /// Whether the plan crosses an external policy/RLS boundary.
    pub external_policy_boundary: bool,
    /// Planner-visible fallback behavior if a projection or external mapping is unavailable.
    pub fallback_behavior: String,
}

impl ResolvedObjectContext {
    pub fn internal_canonical(
        source: impl Into<String>,
        data_model: impl Into<String>,
        table_identifier: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            alias: None,
            data_model: data_model.into(),
            table_identifier: table_identifier.into(),
            authority: ResolvedAuthorityMode::InternalCanonical,
            storage_layouts: Vec::new(),
            projections: Vec::new(),
            compression_stats_profiles: Vec::new(),
            external_policy_boundary: false,
            fallback_behavior: "read canonical ProximaRecord storage".to_string(),
        }
    }

    pub fn is_external_authority(&self) -> bool {
        matches!(self.authority, ResolvedAuthorityMode::ExternalAuthoritative)
    }

    pub fn requires_policy_boundary(&self) -> bool {
        self.external_policy_boundary
            || self.storage_layouts.iter().any(|layout| {
                layout.authority == ResolvedAuthorityMode::ExternalAuthoritative
                    && !layout.policy_enforced_in_proxima
            })
    }
}

/// Planner-visible physical layout metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolvedStorageLayoutContext {
    pub name: String,
    pub authority: ResolvedAuthorityMode,
    pub layout_kind: String,
    pub physical_format: String,
    pub write_mode: String,
    pub location: Option<String>,
    pub snapshot_semantics: Option<String>,
    pub policy_enforced_in_proxima: bool,
    pub lossy_type_mappings: Vec<String>,
}

/// Planner-visible rebuildable projection/access-method metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedProjectionContext {
    pub name: String,
    pub kind: String,
    pub physical_format: String,
    pub rebuild_source: String,
    pub freshness: String,
    pub max_lag_ms: Option<i64>,
    pub rebuildable: bool,
    pub lossy: bool,
    pub support_status: String,
    /// `ProjectionFreshnessState` variant name (e.g. "Fresh", "Stale", "RebuildRequired").
    /// Surfaced in EXPLAIN so planners and operators can assess projection health.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness_state: Option<String>,
    /// Rebuild rate from `RebuildRtoSpec` in seconds per 10 GiB of L1 data.
    /// None means no RTO estimate has been cataloged for this projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rebuild_rto_seconds_per_10gb: Option<f64>,
}

/// Planner-visible compression/layout profile feedback from xCatalog.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedCompressionStatsProfileContext {
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
    pub rejected_candidates: Vec<ResolvedCompressionRejectedCandidateContext>,
}

/// Planner-visible rejected codec candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedCompressionRejectedCandidateContext {
    pub scheme: String,
    pub reason: String,
    pub expected_ratio: Option<f32>,
}

/// Execution priority
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub enum ExecutionPriority {
    Low,
    #[default]
    Normal,
    High,
    Urgent,
}

/// Plan optimization hints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanHints {
    /// Suggest using index for specific column
    pub use_index: Option<String>,

    /// Suggest join order
    pub join_order: Option<Vec<String>>,

    /// Estimated row count
    pub estimated_rows: Option<usize>,

    /// Enable result caching
    pub enable_cache: bool,

    /// Custom optimization hints
    pub custom_hints: HashMap<String, serde_json::Value>,
}

/// Plan validation result
#[derive(Debug, Clone)]
pub struct PlanValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Plan optimization result
#[derive(Debug, Clone)]
pub struct PlanOptimizationResult {
    pub optimizations_applied: Vec<String>,
    pub original_stats: PlanStats,
    pub optimized_stats: PlanStats,
}

/// Plan statistics
#[derive(Debug, Clone, Default)]
pub struct PlanStats {
    pub operator_count: usize,
    pub scan_count: usize,
    pub filter_count: usize,
    pub project_count: usize,
    pub join_count: usize,
    pub aggregate_count: usize,
    pub sort_count: usize,
    pub topk_count: usize,
    pub union_count: usize,
}

/// Operator contract trait for operator validation
pub trait OperatorContract {
    /// Validate the operator
    fn validate(&self) -> Result<()>;

    /// Get the operator's schema requirements
    fn required_columns(&self) -> Vec<String>;

    /// Get the operator's schema output
    fn output_columns(&self) -> Vec<String>;

    /// Estimate operator cost
    fn estimate_cost(&self, input_rows: usize) -> f64;
}

impl OperatorContract for Operator {
    fn validate(&self) -> Result<()> {
        match self {
            Operator::Scan {
                data_model, source, ..
            } => {
                if source.is_empty() {
                    return Err(anyhow::anyhow!("Scan operator has empty source"));
                }
                debug!("Validated Scan operator: {} for {:?}", source, data_model);
                Ok(())
            }
            Operator::Filter { expression } => {
                // Validate filter expression structure
                match expression {
                    FilterExpression::Comparison { field, .. } => {
                        if field.is_empty() {
                            return Err(anyhow::anyhow!("Filter has empty field name"));
                        }
                    }
                    FilterExpression::And(exprs) | FilterExpression::Or(exprs) => {
                        if exprs.is_empty() {
                            return Err(anyhow::anyhow!("And/Or filter has no expressions"));
                        }
                    }
                    FilterExpression::Not(expr) => {
                        if matches!(
                            expr.as_ref(),
                            FilterExpression::And(_) | FilterExpression::Or(_)
                        ) {
                            return Err(anyhow::anyhow!("Not(And/Or) not supported"));
                        }
                    }
                }
                debug!("Validated Filter operator");
                Ok(())
            }
            Operator::Project { columns } => {
                if columns.is_empty() {
                    return Err(anyhow::anyhow!("Project operator has no columns"));
                }
                debug!("Validated Project operator: {} columns", columns.len());
                Ok(())
            }
            Operator::Sort { column, .. } => {
                if column.is_empty() {
                    return Err(anyhow::anyhow!("Sort operator has empty column name"));
                }
                debug!("Validated Sort operator: {}", column);
                Ok(())
            }
            Operator::TopK { k, sort_column } => {
                if *k == 0 {
                    return Err(anyhow::anyhow!("TopK operator has k=0"));
                }
                if sort_column.is_empty() {
                    return Err(anyhow::anyhow!("TopK operator has empty sort column"));
                }
                debug!("Validated TopK operator: k={}, column={}", k, sort_column);
                Ok(())
            }
            Operator::Join {
                join_type,
                condition,
                ..
            } => {
                match condition {
                    JoinCondition::On(left, right) => {
                        if left.is_empty() || right.is_empty() {
                            return Err(anyhow::anyhow!("Join condition has empty column name"));
                        }
                    }
                    JoinCondition::OnMultiple(pairs) => {
                        if pairs.is_empty() {
                            return Err(anyhow::anyhow!("Join has empty condition"));
                        }
                    }
                    JoinCondition::Expression(expr) => {
                        if matches!(expr, FilterExpression::And(_) | FilterExpression::Or(_)) {
                            return Err(anyhow::anyhow!("Join condition cannot be And/Or"));
                        }
                    }
                }
                debug!("Validated Join operator: {:?}", join_type);
                Ok(())
            }
            Operator::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                if group_by.is_empty() && aggregates.is_empty() {
                    return Err(anyhow::anyhow!(
                        "Aggregate operator has no group_by or aggregates"
                    ));
                }
                for agg in aggregates {
                    if agg.column.is_empty() {
                        return Err(anyhow::anyhow!("Aggregate has empty column name"));
                    }
                }
                debug!(
                    "Validated Aggregate operator: {} groups, {} aggregates",
                    group_by.len(),
                    aggregates.len()
                );
                Ok(())
            }
            Operator::Union { plans, .. } => {
                if plans.is_empty() {
                    return Err(anyhow::anyhow!("Union operator has no plans"));
                }
                debug!("Validated Union operator: {} plans", plans.len());
                Ok(())
            }
            // ── Spec §7 extended operators ──────────────────────────────────
            Operator::Limit { n, .. } => {
                if *n == 0 {
                    return Err(anyhow::anyhow!("Limit n must be > 0"));
                }
                Ok(())
            }
            Operator::VectorTopK {
                k, query_vector, ..
            } => {
                if *k == 0 {
                    return Err(anyhow::anyhow!("VectorTopK k must be > 0"));
                }
                if query_vector.is_empty() {
                    return Err(anyhow::anyhow!("VectorTopK query_vector is empty"));
                }
                Ok(())
            }
            Operator::HybridTraverse { edge_pattern } => {
                if let Some(max_hops) = edge_pattern.max_hops
                    && max_hops < edge_pattern.min_hops
                {
                    return Err(anyhow::anyhow!(
                        "HybridTraverse max_hops must be >= min_hops"
                    ));
                }
                Ok(())
            }
            Operator::PatternMatch { pattern } => {
                if pattern.is_empty() {
                    return Err(anyhow::anyhow!("PatternMatch pattern is empty"));
                }
                Ok(())
            }
            Operator::CrossModelJoin {
                left_modality,
                right_modality,
                ..
            } => {
                if left_modality == right_modality {
                    return Err(anyhow::anyhow!(
                        "CrossModelJoin requires distinct modalities"
                    ));
                }
                Ok(())
            }
            Operator::ModulationOp { ops } => {
                if ops.is_empty() {
                    return Err(anyhow::anyhow!("ModulationOp has no ops"));
                }
                Ok(())
            }
            Operator::MatrixOp { .. } => Ok(()),
            Operator::SemanticJoin { nl_predicate } => {
                if nl_predicate.is_empty() {
                    return Err(anyhow::anyhow!("SemanticJoin nl_predicate is empty"));
                }
                Ok(())
            }
            Operator::ModelConvert {
                source_modality,
                target_modality,
            } => {
                if source_modality == target_modality {
                    return Err(anyhow::anyhow!(
                        "ModelConvert source and target modalities are the same"
                    ));
                }
                Ok(())
            }
        }
    }

    fn required_columns(&self) -> Vec<String> {
        match self {
            Operator::Scan { columns, .. } => columns.clone().unwrap_or_default(),
            Operator::Filter { expression } => extract_columns_from_filter(expression),
            Operator::Project { columns } => columns.clone(),
            Operator::Sort { column, .. } => vec![column.clone()],
            Operator::TopK { sort_column, .. } => vec![sort_column.clone()],
            Operator::Join { condition, .. } => match condition {
                JoinCondition::On(left, right) => vec![left.clone(), right.clone()],
                JoinCondition::OnMultiple(pairs) => pairs
                    .iter()
                    .flat_map(|(l, r)| vec![l.clone(), r.clone()])
                    .collect(),
                JoinCondition::Expression(expr) => extract_columns_from_filter(expr),
            },
            Operator::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                let mut cols = group_by.clone();
                cols.extend(aggregates.iter().map(|a| a.column.clone()));
                cols
            }
            Operator::Union { .. } => vec![],
            // Spec §7 extended operators
            Operator::Limit { .. } => vec![],
            Operator::VectorTopK { predicate, .. } => predicate
                .as_ref()
                .map(extract_columns_from_filter)
                .unwrap_or_default(),
            Operator::HybridTraverse { .. } => vec![],
            Operator::PatternMatch { .. } => vec![],
            Operator::CrossModelJoin { condition, .. } => match condition {
                JoinCondition::On(l, r) => vec![l.clone(), r.clone()],
                JoinCondition::OnMultiple(pairs) => pairs
                    .iter()
                    .flat_map(|(l, r)| vec![l.clone(), r.clone()])
                    .collect(),
                JoinCondition::Expression(expr) => extract_columns_from_filter(expr),
            },
            Operator::ModulationOp { .. } => vec![],
            Operator::MatrixOp { .. } => vec![],
            Operator::SemanticJoin { .. } => vec![],
            Operator::ModelConvert { .. } => vec![],
        }
    }

    fn output_columns(&self) -> Vec<String> {
        match self {
            Operator::Scan { columns, .. } => columns.clone().unwrap_or_default(),
            Operator::Filter { .. } => vec![], // Filter preserves all columns
            Operator::Project { columns } => columns.clone(),
            Operator::Sort { .. } => vec![], // Sort preserves all columns
            Operator::TopK { .. } => vec![], // TopK preserves all columns
            Operator::Join { alias, .. } => {
                // Output columns depend on join schema - placeholder
                alias.clone().map(|a| vec![a]).unwrap_or_default()
            }
            Operator::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                let mut cols = group_by.clone();
                cols.extend(aggregates.iter().filter_map(|a| a.alias.clone()));
                cols
            }
            Operator::Union { .. } => vec![], // Union output depends on unioned schemas
            // Spec §7 extended operators
            Operator::Limit { .. } => vec![],
            Operator::VectorTopK { .. } => vec!["id".to_string(), "score".to_string()],
            Operator::HybridTraverse { .. } => {
                vec![
                    "id".to_string(),
                    "score".to_string(),
                    "hop_depth".to_string(),
                ]
            }
            Operator::PatternMatch { .. } => vec![],
            Operator::CrossModelJoin { .. } => vec![],
            Operator::ModulationOp { .. } => vec![],
            Operator::MatrixOp { .. } => vec![],
            Operator::SemanticJoin { .. } => vec![],
            Operator::ModelConvert { .. } => vec![],
        }
    }

    fn estimate_cost(&self, input_rows: usize) -> f64 {
        match self {
            Operator::Scan { .. } => input_rows as f64,
            Operator::Filter { .. } => input_rows as f64 * 0.5,
            Operator::Project { .. } => input_rows as f64 * 0.1,
            Operator::Sort { .. } => input_rows as f64 * (input_rows as f64).log2(),
            Operator::TopK { k, .. } => {
                input_rows as f64 + (*k as f64 * (input_rows as f64).log2())
            }
            Operator::Join { .. } => input_rows as f64 * input_rows as f64,
            Operator::Aggregate { .. } => input_rows as f64 * 1.5,
            Operator::Union { .. } => input_rows as f64,
            // Spec §7 extended operators
            Operator::Limit { n, .. } => (*n).min(input_rows) as f64,
            // HNSW is O(log n * ef) ≈ O(log² n) with predicate overhead
            Operator::VectorTopK { k, .. } => (input_rows as f64).log2().powi(2) * *k as f64,
            // Graph traversal is O(V + E) in the worst case
            Operator::HybridTraverse { .. } => input_rows as f64 * 2.0,
            Operator::PatternMatch { .. } => input_rows as f64 * 3.0,
            // MSHJ: O(n + m) with hash table
            Operator::CrossModelJoin { .. } => input_rows as f64 * 1.8,
            Operator::ModulationOp { ops, .. } => input_rows as f64 * ops.len() as f64 * 0.1,
            Operator::MatrixOp { .. } => input_rows as f64 * input_rows as f64 * 0.001,
            Operator::SemanticJoin { .. } => input_rows as f64 * input_rows as f64 * 0.5,
            Operator::ModelConvert { .. } => input_rows as f64 * 0.2,
        }
    }
}

/// Extract column names from a filter expression
fn extract_columns_from_filter(expression: &FilterExpression) -> Vec<String> {
    match expression {
        FilterExpression::Comparison { field, .. } => vec![field.clone()],
        FilterExpression::And(exprs) | FilterExpression::Or(exprs) => {
            exprs.iter().flat_map(extract_columns_from_filter).collect()
        }
        FilterExpression::Not(expr) => extract_columns_from_filter(expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_simple_plan() {
        let operators = vec![
            Operator::Scan {
                data_model: DataModel::Vector,
                source: "test_collection".to_string(),
                columns: None,
                filter: None,
            },
            Operator::Filter {
                expression: FilterExpression::Comparison {
                    field: "score".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: serde_json::json!(0.5),
                },
            },
        ];

        let context = PlanContext::default();
        let plan = MultiModelPlan::new(operators, context);

        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_plan_context_carries_resolved_object_authority() {
        let mut context = PlanContext::default();
        assert!(context.resolved_objects.is_empty());

        context
            .resolved_objects
            .push(ResolvedObjectContext::internal_canonical(
                "docs",
                "document",
                "default.docs",
            ));

        let resolved = &context.resolved_objects[0];
        assert_eq!(resolved.source, "docs");
        assert_eq!(resolved.authority, ResolvedAuthorityMode::InternalCanonical);
        assert!(!resolved.requires_policy_boundary());
    }

    #[test]
    fn test_plan_context_marks_external_authority_as_policy_boundary() {
        let mut resolved =
            ResolvedObjectContext::internal_canonical("lake_docs", "document", "lake.docs");
        resolved.authority = ResolvedAuthorityMode::ExternalAuthoritative;
        resolved.storage_layouts.push(ResolvedStorageLayoutContext {
            name: "iceberg".to_string(),
            authority: ResolvedAuthorityMode::ExternalAuthoritative,
            layout_kind: "ExternalTable".to_string(),
            physical_format: "Iceberg".to_string(),
            write_mode: "ExternalRefresh".to_string(),
            location: Some("s3://warehouse/docs".to_string()),
            snapshot_semantics: Some("iceberg-snapshot".to_string()),
            policy_enforced_in_proxima: false,
            lossy_type_mappings: Vec::new(),
        });

        assert!(resolved.is_external_authority());
        assert!(resolved.requires_policy_boundary());
    }

    #[test]
    fn test_plan_context_carries_compression_stats_profiles() {
        let mut resolved =
            ResolvedObjectContext::internal_canonical("vectors", "vector", "default.vectors");
        resolved
            .compression_stats_profiles
            .push(ResolvedCompressionStatsProfileContext {
                profile_id: "bench/vector/base_xor".to_string(),
                layout_name: Some("pax_vector_spatial".to_string()),
                projection_id: Some("embedding_exact".to_string()),
                selected_scheme: "VectorBaseXorEntropy".to_string(),
                raw_bytes: 1024,
                encoded_bytes: 256,
                value_count: 128,
                measured_ratio: 4.0,
                exact_reconstruction: true,
                encode_cpu_ms_per_block: Some(0.9),
                decode_ns_per_value: Some(12.0),
                rejected_candidates: vec![ResolvedCompressionRejectedCandidateContext {
                    scheme: "Raw".to_string(),
                    reason: "CompressionTargetMiss".to_string(),
                    expected_ratio: Some(1.0),
                }],
            });

        assert_eq!(resolved.compression_stats_profiles.len(), 1);
        assert!(resolved.compression_stats_profiles[0].exact_reconstruction);
        assert_eq!(
            resolved.compression_stats_profiles[0].rejected_candidates[0].scheme,
            "Raw"
        );
    }

    #[test]
    fn test_plan_context_serde_is_backward_compatible_without_resolved_objects() {
        let json = serde_json::json!({
            "created_at": "2026-05-15T00:00:00Z",
            "timeout_ms": null,
            "memory_limit_bytes": null,
            "enable_distributed": false,
            "priority": "Normal"
        });

        let context: PlanContext =
            serde_json::from_value(json).expect("PlanContext should deserialize legacy payloads");
        assert!(context.resolved_objects.is_empty());
    }

    #[test]
    fn test_plan_validation() {
        let operators = vec![
            Operator::Scan {
                data_model: DataModel::Vector,
                source: "test_collection".to_string(),
                columns: None,
                filter: None,
            },
            Operator::Project {
                columns: vec!["id".to_string(), "score".to_string()],
            },
        ];

        let context = PlanContext::default();
        let plan = MultiModelPlan::new(operators, context);

        let validation = plan.validate().unwrap();
        assert!(validation.is_valid);
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn test_plan_validation_empty_source() {
        let operators = vec![Operator::Scan {
            data_model: DataModel::Vector,
            source: "".to_string(), // Empty source - invalid
            columns: None,
            filter: None,
        }];

        let context = PlanContext::default();
        let plan = MultiModelPlan::new(operators, context);

        let validation = plan.validate().unwrap();
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
    }

    #[test]
    fn test_operator_contract_validation() {
        let scan_op = Operator::Scan {
            data_model: DataModel::Vector,
            source: "test".to_string(),
            columns: None,
            filter: None,
        };

        assert!(scan_op.validate().is_ok());

        let invalid_scan = Operator::Scan {
            data_model: DataModel::Vector,
            source: "".to_string(), // Invalid
            columns: None,
            filter: None,
        };

        assert!(invalid_scan.validate().is_err());
    }

    #[test]
    fn test_join_operator() {
        let left_plan = MultiModelPlan::new(
            vec![Operator::Scan {
                data_model: DataModel::Vector,
                source: "users".to_string(),
                columns: None,
                filter: None,
            }],
            PlanContext::default(),
        );

        let right_plan = MultiModelPlan::new(
            vec![Operator::Scan {
                data_model: DataModel::Document,
                source: "orders".to_string(),
                columns: None,
                filter: None,
            }],
            PlanContext::default(),
        );

        let join_op = Operator::Join {
            join_type: JoinType::Inner,
            left_plan: Box::new(left_plan),
            right_plan: Box::new(right_plan),
            condition: JoinCondition::On("user_id".to_string(), "id".to_string()),
            alias: Some("user_orders".to_string()),
        };

        assert!(join_op.validate().is_ok());
    }

    #[test]
    fn test_aggregate_operator() {
        let agg_op = Operator::Aggregate {
            group_by: vec!["category".to_string()],
            aggregates: vec![
                AggregateExpression {
                    function: AggregateFunction::Count,
                    column: "*".to_string(),
                    alias: Some("count".to_string()),
                    distinct: false,
                },
                AggregateExpression {
                    function: AggregateFunction::Avg,
                    column: "score".to_string(),
                    alias: Some("avg_score".to_string()),
                    distinct: false,
                },
            ],
            alias: None,
        };

        assert!(agg_op.validate().is_ok());
    }

    #[test]
    fn test_plan_stats() {
        let operators = vec![
            Operator::Scan {
                data_model: DataModel::Vector,
                source: "test".to_string(),
                columns: None,
                filter: None,
            },
            Operator::Filter {
                expression: FilterExpression::Comparison {
                    field: "score".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: serde_json::json!(0.5),
                },
            },
            Operator::TopK {
                k: 10,
                sort_column: "score".to_string(),
            },
        ];

        let context = PlanContext::default();
        let plan = MultiModelPlan::new(operators, context);

        let stats = plan.stats();
        assert_eq!(stats.operator_count, 3);
        assert_eq!(stats.scan_count, 1);
        assert_eq!(stats.filter_count, 1);
        assert_eq!(stats.topk_count, 1);
    }

    #[test]
    fn test_to_compute_operators() {
        let operators = vec![
            Operator::Scan {
                data_model: DataModel::Vector,
                source: "test".to_string(),
                columns: None,
                filter: None,
            },
            Operator::Project {
                columns: vec!["id".to_string(), "name".to_string()],
            },
        ];

        let context = PlanContext::default();
        let plan = MultiModelPlan::new(operators, context);

        let compute_ops = plan.to_compute_operators();
        assert_eq!(compute_ops.len(), 2);
    }

    #[test]
    fn test_operator_cost_estimation() {
        let scan_op = Operator::Scan {
            data_model: DataModel::Vector,
            source: "test".to_string(),
            columns: None,
            filter: None,
        };

        let cost = scan_op.estimate_cost(1000);
        assert_eq!(cost, 1000.0);

        let sort_op = Operator::Sort {
            column: "score".to_string(),
            ascending: false,
            limit: Some(10),
        };

        let sort_cost = sort_op.estimate_cost(1000);
        assert!(sort_cost > 1000.0); // Sort is more expensive than scan
    }

    // ── Spec §7 extended operator TDD tests ──────────────────────────────────

    #[test]
    fn test_limit_operator_validates() {
        let op = Operator::Limit { n: 10, offset: 0 };
        assert!(op.validate().is_ok());

        let zero = Operator::Limit { n: 0, offset: 0 };
        assert!(zero.validate().is_err(), "n=0 must fail");
    }

    #[test]
    fn test_vector_topk_operator_validates() {
        let op = Operator::VectorTopK {
            query_vector: vec![0.1, 0.2, 0.3],
            k: 10,
            metric: VectorMetric::Cosine,
            predicate: None,
        };
        assert!(op.validate().is_ok());

        let zero_k = Operator::VectorTopK {
            query_vector: vec![0.1],
            k: 0,
            metric: VectorMetric::L2,
            predicate: None,
        };
        assert!(zero_k.validate().is_err(), "k=0 must fail");

        let empty_vec = Operator::VectorTopK {
            query_vector: vec![],
            k: 5,
            metric: VectorMetric::Cosine,
            predicate: None,
        };
        assert!(
            empty_vec.validate().is_err(),
            "empty query_vector must fail"
        );
    }

    #[test]
    fn test_vector_topk_cost_is_sublinear_in_input() {
        let op = Operator::VectorTopK {
            query_vector: vec![0.1, 0.2],
            k: 10,
            metric: VectorMetric::Cosine,
            predicate: None,
        };
        // HNSW cost is O(log² n * k), not O(n)
        let cost_1000 = op.estimate_cost(1000);
        let cost_1_000_000 = op.estimate_cost(1_000_000);
        assert!(
            cost_1_000_000 < 1000.0 * cost_1000,
            "VectorTopK should be sublinear vs linear scan"
        );
    }

    #[test]
    fn test_hybrid_traverse_validates() {
        let op = Operator::HybridTraverse {
            edge_pattern: EdgePattern::default(),
        };
        assert!(op.validate().is_ok());

        let bad = Operator::HybridTraverse {
            edge_pattern: EdgePattern {
                min_hops: 2,
                max_hops: Some(1),
                ..EdgePattern::default()
            },
        };
        assert!(bad.validate().is_err(), "max_hops < min_hops must fail");

        let seed_inclusive = Operator::HybridTraverse {
            edge_pattern: EdgePattern {
                min_hops: 0,
                max_hops: Some(0),
                ..EdgePattern::default()
            },
        };
        assert!(
            seed_inclusive.validate().is_ok(),
            "min_hops=0 is valid for ANN seed output"
        );
    }

    #[test]
    fn test_pattern_match_validates() {
        let op = Operator::PatternMatch {
            pattern: "(a)-[:KNOWS]->(b)".to_string(),
        };
        assert!(op.validate().is_ok());

        let empty = Operator::PatternMatch {
            pattern: "".to_string(),
        };
        assert!(empty.validate().is_err());
    }

    #[test]
    fn test_cross_model_join_requires_distinct_modalities() {
        let ok = Operator::CrossModelJoin {
            left_modality: DataModel::Vector,
            right_modality: DataModel::Graph,
            condition: JoinCondition::On("id".to_string(), "entity_id".to_string()),
        };
        assert!(ok.validate().is_ok());

        let same = Operator::CrossModelJoin {
            left_modality: DataModel::Vector,
            right_modality: DataModel::Vector,
            condition: JoinCondition::On("id".to_string(), "id".to_string()),
        };
        assert!(same.validate().is_err(), "same modality must fail");
    }

    #[test]
    fn test_modulation_op_validates() {
        let op = Operator::ModulationOp {
            ops: vec!["scale".to_string()],
        };
        assert!(op.validate().is_ok());

        let empty = Operator::ModulationOp { ops: vec![] };
        assert!(empty.validate().is_err());
    }

    #[test]
    fn test_matrix_op_always_valid() {
        let op = Operator::MatrixOp {
            op: MatrixOpKind::Hadamard,
        };
        assert!(op.validate().is_ok());
    }

    #[test]
    fn test_semantic_join_validates() {
        let op = Operator::SemanticJoin {
            nl_predicate: "items semantically related to user interests".to_string(),
        };
        assert!(op.validate().is_ok());

        let empty = Operator::SemanticJoin {
            nl_predicate: "".to_string(),
        };
        assert!(empty.validate().is_err());
    }

    #[test]
    fn test_model_convert_requires_distinct_modalities() {
        let ok = Operator::ModelConvert {
            source_modality: DataModel::Vector,
            target_modality: DataModel::Document,
        };
        assert!(ok.validate().is_ok());

        let same = Operator::ModelConvert {
            source_modality: DataModel::Graph,
            target_modality: DataModel::Graph,
        };
        assert!(same.validate().is_err(), "same modality must fail");
    }

    #[test]
    fn test_vector_topk_with_predicate_in_plan() {
        let ops = vec![
            Operator::Scan {
                data_model: DataModel::Vector,
                source: "embeddings".to_string(),
                columns: None,
                filter: None,
            },
            Operator::VectorTopK {
                query_vector: vec![0.1, 0.2, 0.3, 0.4],
                k: 20,
                metric: VectorMetric::Cosine,
                predicate: Some(FilterExpression::Comparison {
                    field: "tenant_id".to_string(),
                    operator: ComparisonOperator::Equals,
                    value: serde_json::json!("acme"),
                }),
            },
        ];
        let plan = MultiModelPlan::new(ops, PlanContext::default());
        assert_eq!(plan.len(), 2);
        assert!(plan.validate().unwrap().is_valid);
    }
}

/// Physical compute plan IR — serializable provider-agnostic plan used by the
/// compute scheduler, providers, and pipeline executor.
pub mod compute_plan;
