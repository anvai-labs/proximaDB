//! # Cross-Model Query Optimizer
//!
//! Optimizes federated queries across multiple data models.
//!
//! ## Optimization Strategies
//!
//! - **Predicate pushdown**: Push filters to storage engines
//! - **Join reordering**: Optimize cross-model join order
//! - **Model-aware cost estimation**: Different cost models per data model
//! - **Parallel execution planning**: Identify parallelizable operations

use anyhow::{Result, anyhow};
use proximadb_graph_subset::describe_supported_graph_query;
use std::collections::HashMap;

use super::parser::{
    FederatedQuery, QueryTarget, QueryType, SqlExtension, TargetModelType, VectorQuery,
};
use crate::query::capability::{Capability, CapabilitySet};
use crate::storage::multimodel::ModelType;
use proximadb_kernel::error::VectorDBError;

mod sql_parsing;
pub(crate) mod vector_query_parsing;

pub mod cached_plan_builder;
pub mod filter_strategy;
// Moved to proximadb-query-fusion (TD-DECOMP-17)
pub use proximadb_query_fusion::gls;
pub mod plan_builder;
pub use proximadb_query_fusion::plan_calibration;
pub mod plan_inference_gate;
pub mod plan_quality;
pub mod plan_v2_inference;
pub mod plan_v2_training;
pub mod predicate_normalizer;
pub mod selectivity;
pub mod trace_replay;

/// Physical plan node types
#[derive(Debug, Clone)]
pub enum PlanNodeType {
    /// Scan a table/collection
    Scan {
        /// Target table or collection name
        target: String,
        /// Data model type
        model_type: ModelType,
        /// Predicates to apply during scan
        predicates: Vec<Predicate>,
    },
    /// Vector similarity search
    VectorSearch {
        /// Collection to search
        collection: String,
        /// Number of nearest neighbors to return
        top_k: usize,
        /// Source of the query vector
        query_vector_source: VectorSource,
    },
    /// Multi-phase rank pipeline (RERANK SRF surface).
    ///
    /// R-7c.4c.1: parser-recognized RERANK(...) lowers into this
    /// node. The optimizer doesn't break it down further — execution
    /// dispatches to the rank pipeline (first-phase retrieval +
    /// optional second-phase rescorer per `rank_profile`).
    RerankSearch {
        /// Collection to retrieve from
        collection: String,
        /// Free-text query (threaded into the rank pipeline's
        /// `QueryContext.query_text` for tokenized scorers)
        query_text: String,
        /// Source of the dense embedding for first-phase retrieval
        query_vector_source: VectorSource,
        /// First-phase k (cut-off before second-phase rescoring)
        k: usize,
        /// Optional rank profile name. `None` = retrieval-only path
        /// (skip second phase).
        rank_profile: Option<String>,
    },
    /// Graph traversal
    GraphTraversal {
        /// Cypher query to execute
        cypher: String,
        /// Optional starting node IDs
        start_nodes: Option<Vec<String>>,
        /// SQL alias for the source when present
        source_alias: Option<String>,
    },
    /// Document query
    DocumentQuery {
        /// Document collection name
        collection: String,
        /// Optional filter expression
        filter: Option<String>,
        /// SQL alias for the source when present
        source_alias: Option<String>,
    },
    /// Observability query (logs/metrics)
    ObservabilityQuery {
        /// Observability namespace
        namespace: String,
        /// Type of observability data to query
        query_type: ObservabilityQueryType,
        /// Optional time range filter
        time_range: Option<TimeRange>,
    },
    /// Hash join
    HashJoin {
        /// Left input plan
        left: Box<PlanNode>,
        /// Right input plan
        right: Box<PlanNode>,
        /// Join key pairs (left_col, right_col)
        join_keys: Vec<(String, String)>,
        /// Type of join
        join_type: JoinType,
    },
    /// Nested loop join (for LATERAL)
    NestedLoopJoin {
        /// Outer (driving) plan
        outer: Box<PlanNode>,
        /// Inner (probed) plan
        inner: Box<PlanNode>,
        /// Correlated columns
        correlation: Vec<String>,
    },
    /// Index join
    IndexJoin {
        /// Left input plan
        left: Box<PlanNode>,
        /// Right input plan
        right: Box<PlanNode>,
        /// Index to use for the lookup
        index_lookup: String,
    },
    /// Filter operation
    Filter {
        /// Input plan to filter
        input: Box<PlanNode>,
        /// Predicate to apply
        predicate: Predicate,
    },
    /// Project columns
    Project {
        /// Input plan to project
        input: Box<PlanNode>,
        /// Column names to project
        columns: Vec<String>,
    },
    /// Distinct rows across the projected output
    Distinct {
        /// Input plan to deduplicate
        input: Box<PlanNode>,
    },
    /// Sort/Order By
    Sort {
        /// Input plan to sort
        input: Box<PlanNode>,
        /// Sort ordering clauses
        order_by: Vec<OrderByClause>,
    },
    /// Limit
    Limit {
        /// Input plan to limit
        input: Box<PlanNode>,
        /// Maximum number of rows
        limit: usize,
        /// Number of rows to skip
        offset: usize,
    },
    /// Aggregate
    Aggregate {
        /// Input plan to aggregate
        input: Box<PlanNode>,
        /// Group by column names
        group_by: Vec<String>,
        /// Aggregate expressions to compute
        aggregates: Vec<AggregateExpr>,
    },
    /// Union
    Union {
        /// Input plans to union
        inputs: Vec<PlanNode>,
        /// Whether to include all rows (UNION ALL)
        all: bool,
    },
}

/// Source of a query vector
#[derive(Debug, Clone)]
pub enum VectorSource {
    /// Literal vector value
    Literal(Vec<f32>),
    /// Column reference from another table
    ColumnRef {
        /// Source table name
        table: String,
        /// Column name in the source table
        column: String,
    },
    /// Raw SQL expression that could not be reduced during parsing
    Expression(String),
    /// Subquery result
    Subquery(Box<PlanNode>),
}

/// Observability query type
#[derive(Debug, Clone)]
pub enum ObservabilityQueryType {
    /// Log entries query
    Logs,
    /// Metrics query
    Metrics,
    /// Distributed traces query
    Traces,
}

/// Time range for observability queries
#[derive(Debug, Clone)]
pub struct TimeRange {
    /// Start time in nanoseconds since epoch
    pub start_ns: i64,
    /// End time in nanoseconds since epoch
    pub end_ns: i64,
}

/// Join type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join
    Inner,
    /// Left outer join
    Left,
    /// Right outer join
    Right,
    /// Full outer join
    Full,
    /// Cross join (cartesian product)
    Cross,
    /// Lateral join (correlated subquery)
    Lateral,
}

/// Predicate for filtering
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Column name to filter on
    pub column: String,
    /// Comparison operator
    pub op: PredicateOp,
    /// Value to compare against
    pub value: PredicateValue,
}

/// Predicate operators
#[derive(Debug, Clone)]
pub enum PredicateOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// LIKE pattern match
    Like,
    /// IN set membership
    In,
    /// IS NULL check
    IsNull,
    /// IS NOT NULL check
    IsNotNull,
    /// BETWEEN range check
    Between,
}

/// Predicate values
#[derive(Debug, Clone)]
pub enum PredicateValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// List of values (for IN operator)
    List(Vec<PredicateValue>),
    /// Null value
    Null,
}

/// Order by clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    /// Column name to sort by
    pub column: String,
    /// True for ascending, false for descending
    pub ascending: bool,
    /// Whether nulls sort first
    pub nulls_first: bool,
}

/// Aggregate expression
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// Aggregate function to apply
    pub function: AggregateFunction,
    /// Column to aggregate (None for COUNT(*))
    pub column: Option<String>,
    /// Output alias name
    pub alias: String,
}

/// Aggregate functions
#[derive(Debug, Clone)]
pub enum AggregateFunction {
    /// Count of rows
    Count,
    /// Sum of values
    Sum,
    /// Average of values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of distinct values
    CountDistinct,
}

/// Query execution plan
#[derive(Debug, Clone)]
pub struct PlanNode {
    /// Node ID for tracking
    pub id: usize,
    /// Node type with parameters
    pub node_type: PlanNodeType,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated row count
    pub estimated_rows: u64,
    /// Output columns
    pub output_columns: Vec<String>,
    /// Required capabilities for this plan node
    pub required_capabilities: CapabilitySet,
}

impl PlanNode {
    /// Create a new PlanNode with capabilities inferred from node type
    ///
    /// This is the preferred way to create PlanNodes as it automatically
    /// determines the required capabilities based on the node type.
    pub fn with_inferred_capabilities(
        id: usize,
        node_type: PlanNodeType,
        estimated_cost: f64,
        estimated_rows: u64,
        output_columns: Vec<String>,
    ) -> Self {
        let mut node = Self {
            id,
            node_type,
            estimated_cost,
            estimated_rows,
            output_columns,
            required_capabilities: CapabilitySet::new(),
        };
        node.required_capabilities = node.infer_capabilities_from_node_type();
        node
    }

    /// Infer capabilities required for this plan node based on its type
    ///
    /// This method analyzes the PlanNodeType and determines what capabilities
    /// are required to execute this node. Used for query validation and planning.
    pub fn infer_capabilities(&self) -> CapabilitySet {
        // Start with the already-inferred capabilities
        if !self.required_capabilities.is_empty() {
            return self.required_capabilities.clone();
        }

        // Infer based on node type
        self.infer_capabilities_from_node_type()
    }

    /// Infer capabilities from the node type
    fn infer_capabilities_from_node_type(&self) -> CapabilitySet {
        let mut capabilities = CapabilitySet::new();

        match &self.node_type {
            PlanNodeType::Scan {
                target,
                model_type,
                predicates,
            } => {
                // All scans require basic scan capability
                capabilities.add(Capability::Scan);

                // If there are predicates, add filter capability
                if !predicates.is_empty() {
                    capabilities.add(Capability::Filter);
                    capabilities.add(Capability::PredicatePushdown);
                }

                // Model-specific capabilities
                match model_type {
                    ModelType::Vector => {
                        capabilities.add(Capability::VectorSearch);
                        // Add all distance metrics as options
                        capabilities.add(Capability::CosineDistance);
                        capabilities.add(Capability::EuclideanDistance);
                        capabilities.add(Capability::DotProduct);
                    }
                    ModelType::Graph => {
                        capabilities.add(Capability::GraphQuery);
                        capabilities.add(Capability::GraphTraversal);
                    }
                    ModelType::Document => {
                        capabilities.add(Capability::DocumentQuery);
                        capabilities.add(Capability::FullTextSearch);
                        capabilities.add(Capability::JSONPathQueries);
                    }
                    ModelType::Observability => {
                        // Add observability capabilities based on target
                        if target.contains("log") {
                            capabilities.add(Capability::LogQuery);
                            capabilities.add(Capability::LogAggregation);
                        } else if target.contains("metric") {
                            capabilities.add(Capability::MetricsQuery);
                            capabilities.add(Capability::MetricAggregation);
                        } else {
                            capabilities.add(Capability::LogQuery);
                            capabilities.add(Capability::MetricsQuery);
                        }
                    }
                    ModelType::Relational => {
                        // Relational tables support standard SQL operations
                        capabilities.add(Capability::Aggregate);
                        capabilities.add(Capability::Join);
                        capabilities.add(Capability::Sort);
                    }
                    ModelType::TimeSeries => {
                        capabilities.add(Capability::TimeSeriesQuery);
                        capabilities.add(Capability::Aggregate);
                    }
                    ModelType::Event => {
                        capabilities.add(Capability::EventSourcingQuery);
                    }
                }
            }

            PlanNodeType::VectorSearch {
                collection: _,
                top_k: _,
                query_vector_source,
            } => {
                capabilities.add(Capability::VectorSearch);
                capabilities.add(Capability::Scan);

                // Add all distance metrics as options
                capabilities.add(Capability::CosineDistance);
                capabilities.add(Capability::EuclideanDistance);
                capabilities.add(Capability::DotProduct);

                // If query vector comes from a column reference, we need column access
                if matches!(query_vector_source, VectorSource::ColumnRef { .. }) {
                    capabilities.add(Capability::Project);
                }
            }

            PlanNodeType::RerankSearch {
                query_vector_source,
                ..
            } => {
                // R-7c.4c.1: rerank requires vector retrieval; the
                // optional second-phase rescorer is profile-driven and
                // doesn't surface as a separate capability here.
                capabilities.add(Capability::VectorSearch);
                capabilities.add(Capability::Scan);
                capabilities.add(Capability::CosineDistance);
                capabilities.add(Capability::EuclideanDistance);
                capabilities.add(Capability::DotProduct);
                if matches!(query_vector_source, VectorSource::ColumnRef { .. }) {
                    capabilities.add(Capability::Project);
                }
            }

            PlanNodeType::GraphTraversal {
                cypher,
                start_nodes: _,
                source_alias: _,
            } => {
                capabilities.add(Capability::GraphQuery);
                capabilities.add(Capability::GraphTraversal);
                capabilities.add(Capability::PatternMatching);
                capabilities.add(Capability::Scan);

                // If Cypher query contains function calls, add function capability
                if cypher.contains('(') && cypher.contains(')') {
                    capabilities.add(Capability::CypherFunctions);
                }
            }

            PlanNodeType::DocumentQuery {
                collection: _,
                filter,
                source_alias: _,
            } => {
                capabilities.add(Capability::DocumentQuery);
                capabilities.add(Capability::Scan);
                capabilities.add(Capability::FullTextSearch);
                capabilities.add(Capability::JSONPathQueries);

                // If there's a filter, add filter capability
                if filter.is_some() {
                    capabilities.add(Capability::Filter);
                    capabilities.add(Capability::PredicatePushdown);
                }
            }

            PlanNodeType::ObservabilityQuery {
                namespace: _,
                query_type,
                time_range,
            } => {
                capabilities.add(Capability::Scan);

                match query_type {
                    ObservabilityQueryType::Logs => {
                        capabilities.add(Capability::LogQuery);
                        capabilities.add(Capability::LogAggregation);
                    }
                    ObservabilityQueryType::Metrics => {
                        capabilities.add(Capability::MetricsQuery);
                        capabilities.add(Capability::MetricAggregation);
                    }
                    ObservabilityQueryType::Traces => {
                        capabilities.add(Capability::LogQuery);
                    }
                }

                // If there's a time range filter, add filter capability
                if time_range.is_some() {
                    capabilities.add(Capability::Filter);
                }
            }

            PlanNodeType::HashJoin {
                left,
                right,
                join_keys: _,
                join_type: _,
            } => {
                capabilities.add(Capability::Join);
                capabilities.add(Capability::Scan);

                // Add capabilities from child nodes
                capabilities = capabilities.union(&left.required_capabilities);
                capabilities = capabilities.union(&right.required_capabilities);

                // For cross-model joins, add cross-model capability
                if left.has_model(ModelType::Vector) && right.has_model(ModelType::Document) {
                    capabilities.add(Capability::CrossModelJoin);
                }
            }

            PlanNodeType::NestedLoopJoin {
                outer,
                inner,
                correlation: _,
            } => {
                capabilities.add(Capability::Join);
                capabilities.add(Capability::Scan);

                // Nested loop joins are often used for cross-model queries
                capabilities.add(Capability::CrossModelJoin);

                // Add capabilities from child nodes
                capabilities = capabilities.union(&outer.required_capabilities);
                capabilities = capabilities.union(&inner.required_capabilities);
            }

            PlanNodeType::IndexJoin {
                left,
                right,
                index_lookup: _,
            } => {
                capabilities.add(Capability::Join);
                capabilities.add(Capability::Scan);
                capabilities.add(Capability::Filter);

                // Add capabilities from child nodes
                capabilities = capabilities.union(&left.required_capabilities);
                capabilities = capabilities.union(&right.required_capabilities);
            }

            PlanNodeType::Filter {
                input,
                predicate: _,
            } => {
                capabilities.add(Capability::Filter);
                capabilities.add(Capability::PredicatePushdown);

                capabilities = capabilities.union(&input.required_capabilities);
            }

            PlanNodeType::Project { input, columns: _ } => {
                capabilities.add(Capability::Project);
                capabilities.add(Capability::Scan);

                capabilities = capabilities.union(&input.required_capabilities);
            }

            PlanNodeType::Distinct { input } => {
                capabilities.add(Capability::Aggregate);
                capabilities.add(Capability::Scan);

                capabilities = capabilities.union(&input.required_capabilities);
            }

            PlanNodeType::Sort { input, order_by: _ } => {
                capabilities.add(Capability::Sort);
                capabilities.add(Capability::Scan);

                capabilities = capabilities.union(&input.required_capabilities);
            }

            PlanNodeType::Limit {
                input,
                limit: _,
                offset: _,
            } => {
                capabilities.add(Capability::Limit);
                capabilities.add(Capability::Scan);

                capabilities = capabilities.union(&input.required_capabilities);
            }

            PlanNodeType::Aggregate {
                input,
                group_by: _,
                aggregates: _,
            } => {
                capabilities.add(Capability::Aggregate);
                capabilities.add(Capability::Scan);

                capabilities = capabilities.union(&input.required_capabilities);
            }

            PlanNodeType::Union { inputs, all: _ } => {
                capabilities.add(Capability::Scan);

                for input in inputs {
                    capabilities = capabilities.union(&input.required_capabilities);
                }
            }
        }

        capabilities
    }

    /// Check if this plan node involves a specific model type
    fn has_model(&self, model_type: ModelType) -> bool {
        self.node_type.has_model(model_type)
    }

    /// Get honest capabilities by intersecting claimed capabilities with actual engine capabilities
    ///
    /// This method closes the "honesty gap" by ensuring that the plan node only claims
    /// capabilities that are actually supported by the underlying storage engine.
    ///
    /// # Arguments
    /// * `actual_engine_capabilities` - The actual capabilities supported by the storage engine
    ///
    /// # Returns
    /// A new CapabilitySet containing only the capabilities that are both claimed and actually available
    pub fn get_honest_capabilities(
        &self,
        actual_engine_capabilities: &CapabilitySet,
    ) -> CapabilitySet {
        self.required_capabilities
            .intersection(actual_engine_capabilities)
    }

    /// Validate that this plan node's capabilities are supported by the storage engine
    ///
    /// Returns an error if the plan node requires capabilities that the storage engine doesn't support.
    ///
    /// # Arguments
    /// * `engine_capabilities` - The actual capabilities supported by the storage engine
    /// * `node_description` - Human-readable description of the node for error messages
    pub fn validate_capabilities(
        &self,
        engine_capabilities: &CapabilitySet,
        node_description: &str,
    ) -> Result<()> {
        let honest_caps = self.get_honest_capabilities(engine_capabilities);

        // Check if we're missing any required capabilities
        let missing_caps = self.required_capabilities.difference(&honest_caps);

        if !missing_caps.is_empty() {
            let missing_list: Vec<String> =
                missing_caps.iter().map(|c| format!("{:?}", c)).collect();

            return Err(anyhow!(
                "Plan node '{}' requires capabilities that the storage engine doesn't support: {:?}. \
                 Please either use a different storage engine or modify the query.",
                node_description,
                missing_list
            ));
        }

        Ok(())
    }

    /// Get a capability honesty report for this plan node
    ///
    /// Returns a detailed report showing the gap between claimed and actual capabilities.
    /// This is useful for debugging and optimization.
    ///
    /// # Arguments
    /// * `actual_engine_capabilities` - The actual capabilities supported by the storage engine
    ///
    /// # Returns
    /// A tuple of (honest_capabilities, missing_capabilities, extra_capabilities)
    /// * honest_capabilities: Capabilities that are both claimed and available
    /// * missing_capabilities: Capabilities that are claimed but not available (the gap)
    /// * extra_capabilities: Capabilities that are available but not claimed (opportunities)
    pub fn get_capability_honesty_report(
        &self,
        actual_engine_capabilities: &CapabilitySet,
    ) -> (CapabilitySet, CapabilitySet, CapabilitySet) {
        let required = if self.required_capabilities.is_empty() {
            self.infer_capabilities()
        } else {
            self.required_capabilities.clone()
        };
        let honest_capabilities = required.intersection(actual_engine_capabilities);
        let missing_capabilities = required.difference(&honest_capabilities);
        let extra_capabilities = actual_engine_capabilities.difference(&honest_capabilities);

        (
            honest_capabilities,
            missing_capabilities,
            extra_capabilities,
        )
    }
}

impl PlanNodeType {
    /// Check if this node type involves a specific model
    fn has_model(&self, model_type: ModelType) -> bool {
        match self {
            PlanNodeType::Scan { model_type: mt, .. } => *mt == model_type,
            PlanNodeType::VectorSearch { .. } => model_type == ModelType::Vector,
            PlanNodeType::RerankSearch { .. } => model_type == ModelType::Vector,
            PlanNodeType::GraphTraversal { .. } => model_type == ModelType::Graph,
            PlanNodeType::DocumentQuery { .. } => model_type == ModelType::Document,
            PlanNodeType::ObservabilityQuery { .. } => model_type == ModelType::Observability,
            PlanNodeType::HashJoin { left, right, .. } => {
                left.has_model(model_type) || right.has_model(model_type)
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                outer.has_model(model_type) || inner.has_model(model_type)
            }
            PlanNodeType::IndexJoin { left, right, .. } => {
                left.has_model(model_type) || right.has_model(model_type)
            }
            PlanNodeType::Filter { input, .. } => input.has_model(model_type),
            PlanNodeType::Project { input, .. } => input.has_model(model_type),
            PlanNodeType::Distinct { input } => input.has_model(model_type),
            PlanNodeType::Sort { input, .. } => input.has_model(model_type),
            PlanNodeType::Limit { input, .. } => input.has_model(model_type),
            PlanNodeType::Aggregate { input, .. } => input.has_model(model_type),
            PlanNodeType::Union { inputs, .. } => {
                inputs.iter().any(|input| input.has_model(model_type))
            }
        }
    }
}

/// Cross-model federated query plan: PlanNode tree + cost + metadata.
///
/// Naming note: this type used to be called `QueryPlan` and collided with
/// the graph/distributed/orchestration/proto `QueryPlan` types. Renamed
/// to make the federation-layer specialisation explicit at call sites.
/// `src/query/mod.rs` previously re-exported it as `FederatedQueryPlan`;
/// the rename makes that alias redundant.
#[derive(Debug, Clone)]
pub struct FederatedQueryPlan {
    /// Root of the plan tree
    pub root: PlanNode,
    /// Total estimated cost
    pub total_cost: f64,
    /// Plan metadata
    pub metadata: PlanMetadata,
}

/// Plan metadata
#[derive(Debug, Default, Clone)]
pub struct PlanMetadata {
    /// Models involved in the plan
    pub involved_models: Vec<ModelType>,
    /// Whether this is a cross-model join
    pub is_cross_model: bool,
    /// Parallelizable stages
    pub parallel_stages: Vec<Vec<usize>>,
    /// Optimization hints applied
    pub hints_applied: Vec<String>,
}

/// Result of attempting to push a predicate into a node
struct PredicatePushResult {
    node: PlanNode,
    predicate_pushed: bool,
}

struct SelectItem {
    expression: String,
    alias: Option<String>,
}

#[derive(Clone, Copy)]
enum QuerySourceRef<'a> {
    Extension {
        extension: &'a SqlExtension,
        ordinal: usize,
    },
    Target(&'a QueryTarget),
}

struct GraphQueryPlanShape {
    graph_name: String,
    output_columns: Vec<String>,
}

impl GraphQueryPlanShape {
    fn from_cypher(cypher: &str) -> Self {
        match describe_supported_graph_query(cypher, None, Some("default")) {
            Ok(descriptor) => Self {
                graph_name: descriptor.graph_id().to_string(),
                output_columns: descriptor.output_columns().to_vec(),
            },
            Err(_) => Self {
                graph_name: "default".to_string(),
                output_columns: vec![
                    "node_id".to_string(),
                    "label".to_string(),
                    "properties".to_string(),
                ],
            },
        }
    }
}

/// Fallback selectivity policy for federated SQL predicates when stats are absent.
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateSelectivityPolicy {
    pub eq: f64,
    pub ne: f64,
    pub range: f64,
    pub like: f64,
    pub in_list: f64,
    pub is_null: f64,
    pub is_not_null: f64,
    pub between: f64,
}

impl Default for PredicateSelectivityPolicy {
    fn default() -> Self {
        Self {
            eq: 0.1,
            ne: 0.9,
            range: 0.3,
            like: 0.2,
            in_list: 0.15,
            is_null: 0.05,
            is_not_null: 0.95,
            between: 0.25,
        }
    }
}

impl PredicateSelectivityPolicy {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("eq", self.eq),
            ("ne", self.ne),
            ("range", self.range),
            ("like", self.like),
            ("in_list", self.in_list),
            ("is_null", self.is_null),
            ("is_not_null", self.is_not_null),
            ("between", self.between),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                anyhow::bail!(
                    "predicate selectivity policy {name} must be finite and between 0.0 and 1.0, got {value}"
                );
            }
        }
        Ok(())
    }
}

fn validate_unit_interval(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        anyhow::bail!("{name} must be finite and between 0.0 and 1.0, got {value}");
    }
    Ok(())
}

/// Cost model for different data models
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Base cost per row scan
    pub row_scan_cost: f64,
    /// Index lookup cost
    pub index_lookup_cost: f64,
    /// Network transfer cost per byte
    pub network_cost_per_byte: f64,
    /// CPU cost per operation
    pub cpu_cost_per_op: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            row_scan_cost: 1.0,
            index_lookup_cost: 0.1,
            network_cost_per_byte: 0.001,
            cpu_cost_per_op: 0.01,
        }
    }
}

// ============================================================================
// STATISTICS PROVIDER AND MODEL STATISTICS
// ============================================================================

/// Statistics for a vector collection
#[derive(Debug, Clone, Default)]
pub struct VectorCollectionStats {
    /// Number of vectors in the collection
    pub vector_count: u64,
    /// Dimension of vectors
    pub dimension: usize,
    /// Whether an HNSW index is available
    pub has_hnsw_index: bool,
    /// Whether an IVF index is available
    pub has_ivf_index: bool,
    /// Number of IVF clusters (if IVF index exists)
    pub ivf_clusters: Option<usize>,
    /// Average query latency in milliseconds (from historical data)
    pub avg_query_latency_ms: Option<f64>,
}

/// Backwards-compat alias for [`FederatedGraphStats`].
pub type GraphStats = FederatedGraphStats;

/// Statistics for a graph
#[derive(Debug, Clone, Default)]
pub struct FederatedGraphStats {
    /// Total number of nodes
    pub node_count: u64,
    /// Total number of edges
    pub edge_count: u64,
    /// Average degree (edges per node)
    pub avg_degree: f64,
    /// Maximum depth for BFS/DFS operations
    pub max_depth: Option<usize>,
    /// Whether the graph is indexed by label
    pub has_label_index: bool,
}

/// Statistics for a document collection
#[derive(Debug, Clone, Default)]
pub struct DocumentCollectionStats {
    /// Number of documents
    pub document_count: u64,
    /// Average document size in bytes
    pub avg_document_size_bytes: u64,
    /// Number of indexed fields
    pub indexed_fields: Vec<String>,
    /// Field cardinalities (distinct values per field)
    pub field_cardinalities: HashMap<String, u64>,
}

/// Statistics for observability data (logs/metrics)
#[derive(Debug, Clone, Default)]
pub struct ObservabilityStats {
    /// Total number of data points
    pub data_point_count: u64,
    /// Time range coverage in seconds
    pub time_range_seconds: u64,
    /// Average data points per second
    pub avg_points_per_second: f64,
    /// Whether time-based partitioning is used
    pub has_time_partitioning: bool,
}

/// Unified statistics for any model type
#[derive(Debug, Clone)]
pub enum ModelStatistics {
    Vector(VectorCollectionStats),
    Graph(FederatedGraphStats),
    Document(DocumentCollectionStats),
    Observability(ObservabilityStats),
    Relational { row_count: u64, avg_row_size: usize },
}

impl ModelStatistics {
    /// Get the estimated row/item count
    pub fn estimated_count(&self) -> u64 {
        match self {
            ModelStatistics::Vector(s) => s.vector_count,
            ModelStatistics::Graph(s) => s.node_count,
            ModelStatistics::Document(s) => s.document_count,
            ModelStatistics::Observability(s) => s.data_point_count,
            ModelStatistics::Relational { row_count, .. } => *row_count,
        }
    }
}

/// Trait for providing statistics to the optimizer
pub trait StatisticsProvider: Send + Sync {
    /// Get statistics for a collection/table by name
    fn get_statistics(&self, name: &str) -> Option<ModelStatistics>;

    /// Get statistics for a specific model type
    fn get_model_statistics(&self, name: &str, model_type: ModelType) -> Option<ModelStatistics>;
}

/// Default statistics provider with cached statistics
#[derive(Debug, Default)]
pub struct CachedStatisticsProvider {
    /// Cached statistics by name
    stats_cache: HashMap<String, ModelStatistics>,
}

impl CachedStatisticsProvider {
    /// Create a new cached statistics provider
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update statistics for a collection
    pub fn set_statistics(&mut self, name: String, stats: ModelStatistics) {
        self.stats_cache.insert(name, stats);
    }

    /// Create default statistics for a vector collection
    pub fn default_vector_stats(vector_count: u64, dimension: usize) -> ModelStatistics {
        ModelStatistics::Vector(VectorCollectionStats {
            vector_count,
            dimension,
            has_hnsw_index: vector_count > 10000,
            has_ivf_index: vector_count > 100000,
            ivf_clusters: if vector_count > 100000 {
                Some(((vector_count as f64).sqrt() as usize).max(16))
            } else {
                None
            },
            avg_query_latency_ms: None,
        })
    }

    /// Create default statistics for a graph
    pub fn default_graph_stats(node_count: u64, edge_count: u64) -> ModelStatistics {
        ModelStatistics::Graph(FederatedGraphStats {
            node_count,
            edge_count,
            avg_degree: if node_count > 0 {
                edge_count as f64 / node_count as f64
            } else {
                0.0
            },
            max_depth: Some(6), // Default max depth for traversals
            has_label_index: true,
        })
    }
}

impl StatisticsProvider for CachedStatisticsProvider {
    fn get_statistics(&self, name: &str) -> Option<ModelStatistics> {
        self.stats_cache.get(name).cloned()
    }

    fn get_model_statistics(&self, name: &str, _model_type: ModelType) -> Option<ModelStatistics> {
        self.stats_cache.get(name).cloned()
    }
}

// ============================================================================
// PER-MODEL COST FUNCTIONS
// ============================================================================

/// Advanced cost estimator with per-model cost functions
#[derive(Debug, Clone)]
pub struct AdvancedCostEstimator {
    /// Base cost model parameters
    base_cost_model: CostModel,
    /// CPU cycles per distance calculation (vector search)
    cpu_cycles_per_distance: f64,
    /// I/O cost per page read
    io_cost_per_page: f64,
    /// Memory access cost factor
    memory_access_cost: f64,
    /// Smoothing factor for feedback-driven cost model updates.
    feedback_ema_alpha: f64,
}

impl Default for AdvancedCostEstimator {
    fn default() -> Self {
        Self {
            base_cost_model: CostModel::default(),
            cpu_cycles_per_distance: 0.001,
            io_cost_per_page: 1.0,
            memory_access_cost: 0.1,
            feedback_ema_alpha: 0.2,
        }
    }
}

impl AdvancedCostEstimator {
    /// Create a new advanced cost estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an advanced cost estimator with explicit feedback EMA smoothing.
    pub fn with_feedback_ema_alpha(mut self, alpha: f64) -> Self {
        debug_assert!(
            validate_unit_interval("feedback_ema_alpha", alpha).is_ok(),
            "invalid feedback_ema_alpha: {alpha}"
        );
        self.feedback_ema_alpha = alpha;
        self
    }

    /// Validate advanced cost estimator tuning.
    pub fn validate(&self) -> Result<()> {
        validate_unit_interval("feedback_ema_alpha", self.feedback_ema_alpha)
    }

    /// Calculate cost for vector search based on collection size and top_k
    ///
    /// Cost model:
    /// - With HNSW: O(log n * top_k * d * ef_search)
    /// - With IVF: O(nprobe * vectors_per_cluster * d + top_k * log(top_k))
    /// - Brute force: O(n * d)
    pub fn vector_search_cost(&self, stats: &VectorCollectionStats, top_k: usize) -> f64 {
        let n = stats.vector_count as f64;
        let d = stats.dimension as f64;

        if n == 0.0 {
            return 0.0;
        }

        // HNSW index: logarithmic search complexity
        if stats.has_hnsw_index {
            let ef_search = (top_k * 2).max(64) as f64; // Typical ef_search
            let hnsw_cost = n.log2() * top_k as f64 * d * ef_search * self.cpu_cycles_per_distance;
            // Add memory access cost for graph traversal
            let memory_cost = n.log2() * 10.0 * self.memory_access_cost;
            return hnsw_cost + memory_cost;
        }

        // IVF index: probe multiple clusters
        if stats.has_ivf_index {
            let nprobe = 10.min(stats.ivf_clusters.unwrap_or(16)) as f64;
            let vectors_per_cluster = n / stats.ivf_clusters.unwrap_or(16) as f64;
            let ivf_cost = nprobe * vectors_per_cluster * d * self.cpu_cycles_per_distance;
            // Add sorting cost for final top_k
            let sort_cost = (top_k as f64) * (top_k as f64).log2() * self.cpu_cycles_per_distance;
            return ivf_cost + sort_cost;
        }

        // Brute force: scan all vectors
        let brute_force_cost = n * d * self.cpu_cycles_per_distance;
        // Add I/O cost for loading vectors
        let pages = (n * d * 4.0 / 4096.0).ceil(); // Assuming 4-byte floats, 4KB pages
        let io_cost = pages * self.io_cost_per_page;

        brute_force_cost + io_cost
    }

    /// Calculate cost for graph traversal based on edge count and depth
    ///
    /// Cost model:
    /// - BFS/DFS: O(V + E) for full traversal, O(branching_factor^depth) for limited depth
    /// - With label index: reduces initial node lookup to O(log V)
    pub fn graph_traversal_cost(&self, stats: &FederatedGraphStats, max_depth: usize) -> f64 {
        let v = stats.node_count as f64;
        let _e = stats.edge_count as f64; // Used in future sophisticated cost models

        if v == 0.0 {
            return 0.0;
        }

        let avg_degree = stats.avg_degree.max(1.0);

        // Limited depth traversal: estimate nodes visited
        let estimated_nodes_visited = if max_depth == 0 {
            1.0
        } else {
            // Approximate: branching_factor^depth, capped at total nodes
            let branching = avg_degree.min(10.0); // Cap branching factor
            let depth_factor = branching.powf(max_depth as f64);
            depth_factor.min(v)
        };

        // Cost components:
        // 1. Node lookup cost
        let lookup_cost = if stats.has_label_index {
            v.log2() * self.base_cost_model.index_lookup_cost
        } else {
            v * self.base_cost_model.row_scan_cost * 0.1
        };

        // 2. Edge traversal cost
        let edges_per_node = avg_degree;
        let edge_traversal_cost =
            estimated_nodes_visited * edges_per_node * self.base_cost_model.cpu_cost_per_op;

        // 3. Memory access for CSR format
        let memory_cost = estimated_nodes_visited * self.memory_access_cost;

        lookup_cost + edge_traversal_cost + memory_cost
    }

    /// Calculate cost for document query based on filter complexity
    ///
    /// Cost model:
    /// - Indexed field: O(log n * selectivity * n)
    /// - Non-indexed field: O(n)
    /// - Complex filter (AND/OR): product/sum of individual selectivities
    pub fn document_query_cost(
        &self,
        stats: &DocumentCollectionStats,
        filter_fields: &[String],
        filter_complexity: usize,
    ) -> f64 {
        let n = stats.document_count as f64;

        if n == 0.0 {
            return 0.0;
        }

        // Base scan cost
        let mut total_cost = 0.0;
        let mut combined_selectivity = 1.0;

        for field in filter_fields {
            let is_indexed = stats.indexed_fields.contains(field);
            let cardinality = stats
                .field_cardinalities
                .get(field)
                .copied()
                .unwrap_or(n as u64);
            let selectivity = 1.0 / cardinality.max(1) as f64;

            if is_indexed {
                // Index lookup + scan matching rows
                let index_cost = n.log2() * self.base_cost_model.index_lookup_cost;
                let scan_cost = n * selectivity * self.base_cost_model.row_scan_cost;
                total_cost += index_cost + scan_cost;
            } else {
                // Full scan with filter
                total_cost += n * self.base_cost_model.row_scan_cost;
            }

            combined_selectivity *= selectivity;
        }

        // Add complexity penalty for nested filters
        let complexity_factor = 1.0 + (filter_complexity as f64 * 0.1);

        // Add document deserialization cost
        let avg_doc_size = stats.avg_document_size_bytes as f64;
        let deser_cost = n * combined_selectivity * avg_doc_size * 0.00001;

        (total_cost + deser_cost) * complexity_factor
    }

    /// Calculate cost for observability query (logs/metrics)
    pub fn observability_query_cost(
        &self,
        stats: &ObservabilityStats,
        time_range_seconds: Option<u64>,
    ) -> f64 {
        let total_points = stats.data_point_count as f64;

        if total_points == 0.0 {
            return 0.0;
        }

        // Calculate fraction of data to scan based on time range
        let scan_fraction = if let Some(range) = time_range_seconds {
            if stats.time_range_seconds > 0 {
                (range as f64 / stats.time_range_seconds as f64).min(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };

        let points_to_scan = total_points * scan_fraction;

        // With time partitioning, we can skip irrelevant partitions
        let partition_benefit = if stats.has_time_partitioning {
            0.5
        } else {
            1.0
        };

        points_to_scan * self.base_cost_model.row_scan_cost * partition_benefit
    }

    /// Update cost model parameters from observed execution feedback.
    /// Uses exponential moving average to smooth parameter updates.
    pub fn update_from_feedback(&mut self, feedback: &ExecutionFeedback) {
        let alpha = self.feedback_ema_alpha;

        if let Some(observed_cpu_per_distance) = feedback.observed_cpu_per_distance {
            self.cpu_cycles_per_distance =
                self.cpu_cycles_per_distance * (1.0 - alpha) + observed_cpu_per_distance * alpha;
        }
        if let Some(observed_io_per_page) = feedback.observed_io_per_page {
            self.io_cost_per_page =
                self.io_cost_per_page * (1.0 - alpha) + observed_io_per_page * alpha;
        }
        if let Some(observed_mem_cost) = feedback.observed_memory_cost {
            self.memory_access_cost =
                self.memory_access_cost * (1.0 - alpha) + observed_mem_cost * alpha;
        }
    }
}

// ============================================================================
// RUNTIME STATISTICS FEEDBACK
// ============================================================================

/// Feedback from a single query execution, used to calibrate cost models
#[derive(Debug, Clone, Default)]
pub struct ExecutionFeedback {
    /// The operation key (e.g., "vector_search:embeddings:top10")
    pub operation_key: String,
    /// Estimated cardinality at plan time
    pub estimated_cardinality: u64,
    /// Actual cardinality observed at execution time
    pub actual_cardinality: u64,
    /// Estimated cost at plan time
    pub estimated_cost: f64,
    /// Actual execution latency in milliseconds
    pub actual_latency_ms: f64,
    /// Observed CPU cost per distance computation (if measurable)
    pub observed_cpu_per_distance: Option<f64>,
    /// Observed I/O cost per page read (if measurable)
    pub observed_io_per_page: Option<f64>,
    /// Observed memory access cost factor (if measurable)
    pub observed_memory_cost: Option<f64>,
    /// Number of rows scanned (for selectivity calibration)
    pub rows_scanned: Option<u64>,
    /// Number of I/O pages read
    pub pages_read: Option<u64>,
}

/// Thread-safe runtime statistics collector that accumulates execution feedback
/// and uses it to calibrate the query optimizer's cost model and cardinality estimates.
///
/// Usage pattern:
/// 1. Before execution: optimizer produces a plan with estimated costs
/// 2. After execution: caller records feedback via `record_feedback()`
/// 3. On next optimization: calibrated estimates incorporate learned corrections
pub struct RuntimeStatisticsCollector {
    /// Accumulated per-operation cardinality ratios (estimated vs actual)
    cardinality_history: parking_lot::RwLock<HashMap<String, CardinalityHistory>>,
    /// Accumulated per-operation latency observations
    latency_history: parking_lot::RwLock<HashMap<String, LatencyHistory>>,
    /// Selectivity observations per filter pattern
    selectivity_history: parking_lot::RwLock<HashMap<String, SelectivityHistory>>,
    /// Maximum number of history entries per operation before compaction
    max_history_per_op: usize,
}

/// Tracks cardinality estimation accuracy for a specific operation type
#[derive(Debug, Clone)]
struct CardinalityHistory {
    /// Rolling ratio of actual/estimated cardinality (EMA)
    correction_ratio: f64,
    /// Number of observations
    sample_count: u64,
    /// Recent observation timestamps for decay
    last_updated_ms: u64,
}

/// Tracks latency observations for cost model calibration
#[derive(Debug, Clone)]
struct LatencyHistory {
    /// Exponential moving average of latency in ms
    avg_latency_ms: f64,
    /// P95 approximation (tracks the max of recent window)
    p95_latency_ms: f64,
    /// Number of observations
    sample_count: u64,
    /// Cost-to-latency ratio for this operation type
    cost_latency_ratio: f64,
}

/// Tracks observed selectivities for filter predicates
#[derive(Debug, Clone)]
struct SelectivityHistory {
    /// Rolling selectivity estimate (fraction of rows passing filter)
    avg_selectivity: f64,
    /// Number of observations
    sample_count: u64,
}

impl Default for RuntimeStatisticsCollector {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl RuntimeStatisticsCollector {
    /// Create a new collector with the given max history entries per operation
    pub fn new(max_history_per_op: usize) -> Self {
        Self {
            cardinality_history: parking_lot::RwLock::new(HashMap::new()),
            latency_history: parking_lot::RwLock::new(HashMap::new()),
            selectivity_history: parking_lot::RwLock::new(HashMap::new()),
            max_history_per_op,
        }
    }

    /// Record execution feedback and update internal models
    pub fn record_feedback(&self, feedback: &ExecutionFeedback) {
        self.update_cardinality_history(feedback);
        self.update_latency_history(feedback);
        if let Some(rows_scanned) = feedback.rows_scanned
            && rows_scanned > 0
            && feedback.actual_cardinality > 0
        {
            self.update_selectivity_history(
                &feedback.operation_key,
                feedback.actual_cardinality as f64 / rows_scanned as f64,
            );
        }
    }

    fn update_cardinality_history(&self, feedback: &ExecutionFeedback) {
        const ALPHA: f64 = 0.3; // Faster adaptation for cardinality
        let mut history = self.cardinality_history.write();

        let ratio = if feedback.estimated_cardinality > 0 {
            feedback.actual_cardinality as f64 / feedback.estimated_cardinality as f64
        } else {
            1.0
        };

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let entry = history
            .entry(feedback.operation_key.clone())
            .or_insert(CardinalityHistory {
                correction_ratio: 1.0,
                sample_count: 0,
                last_updated_ms: now_ms,
            });

        entry.correction_ratio = entry.correction_ratio * (1.0 - ALPHA) + ratio * ALPHA;
        entry.sample_count += 1;
        entry.last_updated_ms = now_ms;

        self.compact_if_needed(&mut history);
    }

    fn update_latency_history(&self, feedback: &ExecutionFeedback) {
        const ALPHA: f64 = 0.2;
        let mut history = self.latency_history.write();

        let entry = history
            .entry(feedback.operation_key.clone())
            .or_insert(LatencyHistory {
                avg_latency_ms: feedback.actual_latency_ms,
                p95_latency_ms: feedback.actual_latency_ms,
                sample_count: 0,
                cost_latency_ratio: if feedback.estimated_cost > 0.0 {
                    feedback.actual_latency_ms / feedback.estimated_cost
                } else {
                    1.0
                },
            });

        entry.avg_latency_ms =
            entry.avg_latency_ms * (1.0 - ALPHA) + feedback.actual_latency_ms * ALPHA;
        // Approximate P95 by tracking 95th percentile via exponential max decay
        if feedback.actual_latency_ms > entry.p95_latency_ms {
            entry.p95_latency_ms = feedback.actual_latency_ms;
        } else {
            entry.p95_latency_ms = entry.p95_latency_ms * 0.99 + feedback.actual_latency_ms * 0.01;
        }
        entry.sample_count += 1;
        if feedback.estimated_cost > 0.0 {
            let new_ratio = feedback.actual_latency_ms / feedback.estimated_cost;
            entry.cost_latency_ratio = entry.cost_latency_ratio * (1.0 - ALPHA) + new_ratio * ALPHA;
        }
    }

    fn update_selectivity_history(&self, operation_key: &str, selectivity: f64) {
        const ALPHA: f64 = 0.25;
        let mut history = self.selectivity_history.write();

        let entry = history
            .entry(operation_key.to_string())
            .or_insert(SelectivityHistory {
                avg_selectivity: selectivity,
                sample_count: 0,
            });

        entry.avg_selectivity = entry.avg_selectivity * (1.0 - ALPHA) + selectivity * ALPHA;
        entry.sample_count += 1;
    }

    fn compact_if_needed(&self, history: &mut HashMap<String, CardinalityHistory>) {
        if history.len() > self.max_history_per_op * 2 {
            // Remove entries with fewest samples
            let mut entries: Vec<_> = history
                .iter()
                .map(|(k, v)| (k.clone(), v.sample_count))
                .collect();
            entries.sort_by_key(|(_, count)| *count);
            let to_remove = history.len() - self.max_history_per_op;
            for (key, _) in entries.into_iter().take(to_remove) {
                history.remove(&key);
            }
        }
    }

    /// Get the calibrated cardinality correction ratio for an operation
    pub fn cardinality_correction(&self, operation_key: &str) -> Option<f64> {
        self.cardinality_history
            .read()
            .get(operation_key)
            .map(|h| h.correction_ratio)
    }

    /// Get the average observed latency for an operation
    pub fn avg_latency(&self, operation_key: &str) -> Option<f64> {
        self.latency_history
            .read()
            .get(operation_key)
            .map(|h| h.avg_latency_ms)
    }

    /// Get the cost-to-latency ratio for translating costs to expected ms
    pub fn cost_latency_ratio(&self, operation_key: &str) -> Option<f64> {
        self.latency_history
            .read()
            .get(operation_key)
            .map(|h| h.cost_latency_ratio)
    }

    /// Get calibrated selectivity for a filter pattern
    pub fn calibrated_selectivity(&self, operation_key: &str) -> Option<f64> {
        self.selectivity_history
            .read()
            .get(operation_key)
            .map(|h| h.avg_selectivity)
    }

    /// Check if a cached plan should be invalidated based on performance regression.
    /// Returns true if actual latency exceeds 3x the expected latency.
    pub fn should_invalidate_plan(
        &self,
        operation_key: &str,
        estimated_cost: f64,
        actual_latency_ms: f64,
    ) -> bool {
        if let Some(ratio) = self.cost_latency_ratio(operation_key) {
            let expected_latency = estimated_cost * ratio;
            actual_latency_ms > expected_latency * 3.0
        } else {
            false
        }
    }

    /// Generate a snapshot of all tracked statistics for diagnostics
    pub fn snapshot(&self) -> RuntimeStatsSnapshot {
        let cardinality = self.cardinality_history.read();
        let latency = self.latency_history.read();
        let selectivity = self.selectivity_history.read();

        RuntimeStatsSnapshot {
            tracked_operations: cardinality.len() + latency.len(),
            cardinality_entries: cardinality.len(),
            latency_entries: latency.len(),
            selectivity_entries: selectivity.len(),
            total_observations: cardinality.values().map(|h| h.sample_count).sum::<u64>()
                + latency.values().map(|h| h.sample_count).sum::<u64>(),
        }
    }
}

/// Diagnostic snapshot of runtime statistics state
#[derive(Debug, Clone)]
pub struct RuntimeStatsSnapshot {
    /// Total unique operations tracked
    pub tracked_operations: usize,
    /// Number of cardinality history entries
    pub cardinality_entries: usize,
    /// Number of latency history entries
    pub latency_entries: usize,
    /// Number of selectivity history entries
    pub selectivity_entries: usize,
    /// Total observation count across all entries
    pub total_observations: u64,
}

// ============================================================================
// CARDINALITY ESTIMATION
// ============================================================================

/// Cardinality estimator for cross-model operations
#[derive(Debug, Clone, Default)]
pub struct CardinalityEstimator {
    /// Historical cardinality data for calibration
    historical_ratios: HashMap<String, f64>,
}

impl CardinalityEstimator {
    /// Create a new cardinality estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate output cardinality for a vector search
    pub fn estimate_vector_search_cardinality(
        &self,
        top_k: usize,
        _stats: &VectorCollectionStats,
    ) -> u64 {
        // Vector search always returns at most top_k results
        top_k as u64
    }

    /// Estimate output cardinality for a graph traversal
    pub fn estimate_graph_traversal_cardinality(
        &self,
        stats: &FederatedGraphStats,
        max_depth: usize,
    ) -> u64 {
        let avg_degree = stats.avg_degree.max(1.0);

        // Estimate based on branching factor and depth
        let estimate = if max_depth == 0 {
            1
        } else {
            let branching = avg_degree.min(10.0);
            (branching.powf(max_depth as f64) as u64).min(stats.node_count)
        };

        estimate.max(1)
    }

    /// Estimate output cardinality for a document query
    pub fn estimate_document_query_cardinality(
        &self,
        stats: &DocumentCollectionStats,
        filter_selectivity: f64,
    ) -> u64 {
        let estimate = (stats.document_count as f64 * filter_selectivity) as u64;
        estimate.max(1)
    }

    /// Estimate cardinality for a join operation
    pub fn estimate_join_cardinality(
        &self,
        left_cardinality: u64,
        right_cardinality: u64,
        join_type: &JoinType,
        join_selectivity: Option<f64>,
    ) -> u64 {
        let selectivity = join_selectivity.unwrap_or(0.1); // Default 10% join selectivity

        let estimate = match join_type {
            JoinType::Inner => {
                // Inner join: product * selectivity
                ((left_cardinality as f64) * (right_cardinality as f64) * selectivity) as u64
            }
            JoinType::Left => {
                // Left join: at least left_cardinality
                let inner =
                    ((left_cardinality as f64) * (right_cardinality as f64) * selectivity) as u64;
                inner.max(left_cardinality)
            }
            JoinType::Right => {
                // Right join: at least right_cardinality
                let inner =
                    ((left_cardinality as f64) * (right_cardinality as f64) * selectivity) as u64;
                inner.max(right_cardinality)
            }
            JoinType::Full => {
                // Full join: max of both plus potential unmatched
                left_cardinality + right_cardinality
            }
            JoinType::Cross => {
                // Cross join: full product
                left_cardinality * right_cardinality
            }
            JoinType::Lateral => {
                // Lateral join: left * average right rows per left row
                let avg_right_per_left = (right_cardinality as f64 * selectivity).max(1.0);
                (left_cardinality as f64 * avg_right_per_left) as u64
            }
        };

        estimate.max(1)
    }

    /// Estimate cardinality after applying a filter
    pub fn estimate_filter_cardinality(&self, input_cardinality: u64, selectivity: f64) -> u64 {
        let estimate = (input_cardinality as f64 * selectivity) as u64;
        estimate.max(1)
    }

    /// Update historical ratio for calibration
    pub fn record_actual_cardinality(
        &mut self,
        operation_key: String,
        estimated: u64,
        actual: u64,
    ) {
        if estimated > 0 {
            let ratio = actual as f64 / estimated as f64;
            self.historical_ratios.insert(operation_key, ratio);
        }
    }

    /// Get calibrated estimate using historical data
    pub fn calibrated_estimate(&self, operation_key: &str, base_estimate: u64) -> u64 {
        if let Some(&ratio) = self.historical_ratios.get(operation_key) {
            (base_estimate as f64 * ratio) as u64
        } else {
            base_estimate
        }
    }
}

// ============================================================================
// DYNAMIC PROGRAMMING JOIN ORDER OPTIMIZATION
// ============================================================================

/// Represents a set of relations for join ordering
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RelationSet {
    /// Bitmask representing which relations are in this set
    mask: u64,
}

#[allow(dead_code)]
impl RelationSet {
    fn new(mask: u64) -> Self {
        Self { mask }
    }

    fn singleton(index: usize) -> Self {
        Self {
            mask: 1u64 << index,
        }
    }

    fn union(&self, other: &RelationSet) -> RelationSet {
        RelationSet::new(self.mask | other.mask)
    }

    fn intersects(&self, other: &RelationSet) -> bool {
        (self.mask & other.mask) != 0
    }

    fn is_subset_of(&self, other: &RelationSet) -> bool {
        (self.mask & other.mask) == self.mask
    }

    fn count(&self) -> usize {
        self.mask.count_ones() as usize
    }

    fn iter_subsets(&self) -> impl Iterator<Item = RelationSet> {
        let mask = self.mask;
        (1..mask)
            .filter(move |&s| (s & mask) == s)
            .map(RelationSet::new)
    }
}

/// Entry in the DP memo table for join ordering
#[derive(Debug, Clone)]
struct DPEntry {
    /// Best plan for this relation set
    plan: PlanNode,
    /// Cost of the best plan
    cost: f64,
    /// Estimated cardinality
    cardinality: u64,
}

/// Join order optimizer using dynamic programming (Selinger-style)
pub struct JoinOrderOptimizer {
    /// Cost estimator (reserved for future use with more sophisticated cost models)
    #[allow(dead_code)]
    cost_estimator: AdvancedCostEstimator,
    /// Cardinality estimator
    cardinality_estimator: CardinalityEstimator,
    /// Maximum number of relations for DP (use greedy for larger)
    dp_threshold: usize,
}

impl Default for JoinOrderOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl JoinOrderOptimizer {
    /// Create a new join order optimizer
    pub fn new() -> Self {
        Self {
            cost_estimator: AdvancedCostEstimator::new(),
            cardinality_estimator: CardinalityEstimator::new(),
            dp_threshold: 10, // Use DP for up to 10 relations
        }
    }

    /// Find optimal join order using dynamic programming
    ///
    /// For n relations, considers all 2^n subsets and finds the optimal
    /// way to join each subset by trying all binary partitions.
    pub fn find_optimal_join_order(
        &self,
        relations: &[PlanNode],
        join_predicates: &[(usize, usize, Vec<(String, String)>)], // (left_idx, right_idx, join_keys)
        next_id: &std::sync::atomic::AtomicUsize,
    ) -> Result<PlanNode> {
        let n = relations.len();

        if n == 0 {
            return Err(anyhow!("No relations to join"));
        }
        if n == 1 {
            return Ok(relations[0].clone());
        }

        // Use greedy for large number of relations
        if n > self.dp_threshold {
            return self.greedy_join_order(relations, join_predicates, next_id);
        }

        // Initialize DP table with single relations
        let mut dp: HashMap<RelationSet, DPEntry> = HashMap::new();

        for (i, rel) in relations.iter().enumerate() {
            let set = RelationSet::singleton(i);
            dp.insert(
                set,
                DPEntry {
                    plan: rel.clone(),
                    cost: rel.estimated_cost,
                    cardinality: rel.estimated_rows,
                },
            );
        }

        // Build up solutions for larger sets
        for size in 2..=n {
            let sets_of_size: Vec<RelationSet> = (0..(1u64 << n))
                .filter(|&mask| (mask.count_ones() as usize) == size)
                .map(RelationSet::new)
                .collect();

            for set in sets_of_size {
                let mut best_entry: Option<DPEntry> = None;

                // Try all ways to partition into two non-empty subsets
                for left_set in set.iter_subsets() {
                    let right_mask = set.mask & !left_set.mask;
                    if right_mask == 0 {
                        continue;
                    }
                    let right_set = RelationSet::new(right_mask);

                    // Check if we have solutions for both subsets
                    let (left_entry, right_entry) = match (dp.get(&left_set), dp.get(&right_set)) {
                        (Some(l), Some(r)) => (l, r),
                        _ => continue,
                    };

                    // Find applicable join predicates
                    let join_keys = self.find_join_keys(&left_set, &right_set, join_predicates);

                    // Calculate join cost and cardinality
                    let join_selectivity = if join_keys.is_empty() { 1.0 } else { 0.1 };
                    let join_cardinality = self.cardinality_estimator.estimate_join_cardinality(
                        left_entry.cardinality,
                        right_entry.cardinality,
                        &JoinType::Inner,
                        Some(join_selectivity),
                    );

                    // Cost model: hash join = build + probe
                    let build_cost = left_entry.cardinality as f64 * 0.01;
                    let probe_cost = right_entry.cardinality as f64 * 0.001;
                    let output_cost = join_cardinality as f64 * 0.001;
                    let join_cost =
                        left_entry.cost + right_entry.cost + build_cost + probe_cost + output_cost;

                    // Update best if this is better
                    let is_better = best_entry.as_ref().is_none_or(|e| join_cost < e.cost);

                    if is_better {
                        let plan = PlanNode {
                            id: next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                            node_type: PlanNodeType::HashJoin {
                                left: Box::new(left_entry.plan.clone()),
                                right: Box::new(right_entry.plan.clone()),
                                join_keys: if join_keys.is_empty() {
                                    vec![("id".to_string(), "id".to_string())]
                                } else {
                                    join_keys
                                },
                                join_type: JoinType::Inner,
                            },
                            estimated_cost: join_cost,
                            estimated_rows: join_cardinality,
                            output_columns: vec!["*".to_string()],
                            required_capabilities: CapabilitySet::new(), // Will be inferred on first access
                        };

                        best_entry = Some(DPEntry {
                            plan,
                            cost: join_cost,
                            cardinality: join_cardinality,
                        });
                    }
                }

                if let Some(entry) = best_entry {
                    dp.insert(set, entry);
                }
            }
        }

        // Get the solution for the full set
        let full_set = RelationSet::new((1u64 << n) - 1);
        dp.get(&full_set)
            .map(|e| e.plan.clone())
            .ok_or_else(|| anyhow!("Failed to find optimal join order"))
    }

    /// Find join keys applicable between two relation sets
    fn find_join_keys(
        &self,
        left_set: &RelationSet,
        right_set: &RelationSet,
        join_predicates: &[(usize, usize, Vec<(String, String)>)],
    ) -> Vec<(String, String)> {
        let mut keys = Vec::new();

        for (left_idx, right_idx, join_keys) in join_predicates {
            let left_in = (left_set.mask >> left_idx) & 1 == 1;
            let right_in = (right_set.mask >> right_idx) & 1 == 1;

            if left_in && (right_in || (right_set.mask >> right_idx) & 1 == 1) {
                keys.extend(join_keys.clone());
            }
        }

        keys
    }

    /// Greedy join ordering for large numbers of relations
    fn greedy_join_order(
        &self,
        relations: &[PlanNode],
        _join_predicates: &[(usize, usize, Vec<(String, String)>)],
        next_id: &std::sync::atomic::AtomicUsize,
    ) -> Result<PlanNode> {
        let mut remaining: Vec<PlanNode> = relations.to_vec();

        // Sort by cost (ascending)
        remaining.sort_by(|a, b| {
            a.estimated_cost
                .partial_cmp(&b.estimated_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedily join from cheapest
        let mut result = remaining.remove(0);

        while !remaining.is_empty() {
            // Find the cheapest relation to join next
            let (best_idx, best_cost) = remaining
                .iter()
                .enumerate()
                .map(|(i, rel)| {
                    let join_cost = result.estimated_cost
                        + rel.estimated_cost
                        + (result.estimated_rows as f64 * 0.01);
                    (i, join_cost)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| VectorDBError::Internal("No viable join order found".to_string()))?;

            let next = remaining.remove(best_idx);
            let join_cardinality = self.cardinality_estimator.estimate_join_cardinality(
                result.estimated_rows,
                next.estimated_rows,
                &JoinType::Inner,
                Some(0.1),
            );

            result = PlanNode {
                id: next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                node_type: PlanNodeType::HashJoin {
                    left: Box::new(result),
                    right: Box::new(next),
                    join_keys: vec![("id".to_string(), "id".to_string())],
                    join_type: JoinType::Inner,
                },
                estimated_cost: best_cost,
                estimated_rows: join_cardinality,
                output_columns: vec!["*".to_string()],
                required_capabilities: CapabilitySet::new(),
            };
        }

        Ok(result)
    }
}

// ============================================================================
// PLAN CACHE
// ============================================================================

/// Cache key for query plans
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanCacheKey {
    /// Normalized SQL (with literals replaced by placeholders)
    normalized_sql: String,
    /// Query type
    query_type: String,
    /// Involved collections/tables
    targets: Vec<String>,
}

impl PlanCacheKey {
    /// Create a cache key from a federated query
    pub fn from_query(query: &FederatedQuery) -> Self {
        use crate::query::federated::parser::SqlExtension;
        let normalized_sql = Self::normalize_sql(&query.sql);
        let query_type = format!("{:?}", query.query_type);
        let mut targets: Vec<String> = query.targets.iter().map(|t| t.name.clone()).collect();
        // Also include collection names from SQL extensions
        for ext in &query.extensions {
            match ext {
                SqlExtension::VectorSearch { collection, .. }
                | SqlExtension::DocumentQuery { collection, .. }
                    if !targets.contains(collection) =>
                {
                    targets.push(collection.clone());
                }
                _ => {}
            }
        }
        targets.sort();

        Self {
            normalized_sql,
            query_type,
            targets,
        }
    }

    /// Normalize SQL by replacing literals with placeholders
    fn normalize_sql(sql: &str) -> String {
        let mut result = String::with_capacity(sql.len());
        let mut in_string = false;
        let mut in_number = false;
        let mut string_char = '"';

        for c in sql.chars() {
            if !in_string && (c == '"' || c == '\'') {
                in_string = true;
                string_char = c;
                result.push('?');
            } else if in_string && c == string_char {
                in_string = false;
            } else if !in_string && !in_number && c.is_ascii_digit() {
                in_number = true;
                result.push('?');
            } else if in_number && !c.is_ascii_digit() && c != '.' {
                in_number = false;
                result.push(c);
            } else if !in_string && !in_number {
                result.push(c);
            }
        }

        result
    }
}

/// Cached plan entry with metadata
#[derive(Debug, Clone)]
pub struct CachedPlan {
    /// The cached query plan
    pub plan: FederatedQueryPlan,
    /// When the plan was created
    pub created_at: std::time::Instant,
    /// Number of times this plan was used
    pub hit_count: u64,
    /// Average execution time when using this plan
    pub avg_execution_time_ms: Option<f64>,
}

/// Plan cache for repeated query patterns
pub struct PlanCache {
    /// Cached plans
    cache: parking_lot::RwLock<HashMap<PlanCacheKey, CachedPlan>>,
    /// Maximum number of cached plans
    max_entries: usize,
    /// Time-to-live for cached plans
    ttl: std::time::Duration,
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(1000, std::time::Duration::from_secs(300))
    }
}

impl PlanCache {
    /// Create a new plan cache
    pub fn new(max_entries: usize, ttl: std::time::Duration) -> Self {
        Self {
            cache: parking_lot::RwLock::new(HashMap::new()),
            max_entries,
            ttl,
        }
    }

    /// Get a cached plan if available
    pub fn get(&self, key: &PlanCacheKey) -> Option<FederatedQueryPlan> {
        let mut cache = self.cache.write();

        if let Some(entry) = cache.get_mut(key) {
            // Check if still valid
            if entry.created_at.elapsed() < self.ttl {
                entry.hit_count += 1;
                return Some(entry.plan.clone());
            } else {
                // Expired, remove it
                cache.remove(key);
            }
        }

        None
    }

    /// Cache a plan
    pub fn put(&self, key: PlanCacheKey, plan: FederatedQueryPlan) {
        let mut cache = self.cache.write();

        // Evict if at capacity
        if cache.len() >= self.max_entries {
            self.evict_lru(&mut cache);
        }

        cache.insert(
            key,
            CachedPlan {
                plan,
                created_at: std::time::Instant::now(),
                hit_count: 0,
                avg_execution_time_ms: None,
            },
        );
    }

    /// Record execution time for a cached plan
    pub fn record_execution_time(&self, key: &PlanCacheKey, execution_time_ms: f64) {
        let mut cache = self.cache.write();

        if let Some(entry) = cache.get_mut(key) {
            let current_avg = entry.avg_execution_time_ms.unwrap_or(execution_time_ms);
            let new_avg = (current_avg * 0.9) + (execution_time_ms * 0.1); // Exponential moving average
            entry.avg_execution_time_ms = Some(new_avg);
        }
    }

    /// Invalidate plans for a specific target
    pub fn invalidate_for_target(&self, target: &str) {
        let mut cache = self.cache.write();
        cache.retain(|key, _| !key.targets.contains(&target.to_string()));
    }

    /// Clear all cached plans
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> FederatedPlanCacheStats {
        let cache = self.cache.read();
        let total_hits: u64 = cache.values().map(|e| e.hit_count).sum();

        FederatedPlanCacheStats {
            cached_plans: cache.len(),
            total_hits,
            avg_plan_age_ms: cache
                .values()
                .map(|e| e.created_at.elapsed().as_millis() as f64)
                .sum::<f64>()
                / cache.len().max(1) as f64,
        }
    }

    /// Evict least recently used entries
    fn evict_lru(&self, cache: &mut HashMap<PlanCacheKey, CachedPlan>) {
        // Collect entries with their keys for sorting
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.hit_count, v.created_at))
            .collect();

        // Sort by hit count (ascending), then by age (descending)
        entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.cmp(&a.2)));

        // Remove bottom 10%
        let to_remove = (entries.len() / 10).max(1);
        for (key, _, _) in entries.into_iter().take(to_remove) {
            cache.remove(&key);
        }
    }
}

/// Federated optimizer plan-cache statistics with plan-age tracking.
///
/// Naming note: this type used to be called `PlanCacheStats` and collided
/// with three other `PlanCacheStats` types (proximadb-query optimizer
/// support, src/query/cache observability, src/query/execution). Renamed
/// because the field set is federation-specific — only this variant tracks
/// average plan age. The canonical `proximadb_query::PlanCacheStats`
/// (re-exported via the proximadb-query crate root) is what generic
/// callers should reach for.
#[derive(Debug, Clone)]
pub struct FederatedPlanCacheStats {
    /// Number of cached plans
    pub cached_plans: usize,
    /// Total cache hits
    pub total_hits: u64,
    /// Average age of cached plans in milliseconds
    pub avg_plan_age_ms: f64,
}

/// Cross-model query optimizer
pub struct CrossModelOptimizer {
    /// Cost models per data model (used for basic cost estimation and model-aware optimization)
    #[allow(dead_code)]
    cost_models: HashMap<ModelType, CostModel>,
    /// Next plan node ID
    next_node_id: std::sync::atomic::AtomicUsize,
    /// Advanced cost estimator for per-model cost functions
    advanced_cost_estimator: AdvancedCostEstimator,
    /// Cardinality estimator for cross-model operations
    cardinality_estimator: CardinalityEstimator,
    /// Join order optimizer using dynamic programming
    join_order_optimizer: JoinOrderOptimizer,
    /// Plan cache for repeated query patterns
    plan_cache: std::sync::Arc<PlanCache>,
    /// Statistics provider (optional, for enhanced cost estimation)
    #[allow(dead_code)]
    statistics_provider: Option<std::sync::Arc<dyn StatisticsProvider>>,
    /// Runtime statistics collector for adaptive cost model tuning
    runtime_stats: std::sync::Arc<RuntimeStatisticsCollector>,
    /// Predicate selectivity policy used when statistics are unavailable.
    predicate_selectivity_policy: PredicateSelectivityPolicy,
}

impl CrossModelOptimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        let mut cost_models = HashMap::new();

        // Vector search is fast with index
        cost_models.insert(
            ModelType::Vector,
            CostModel {
                row_scan_cost: 0.5,
                index_lookup_cost: 0.05,
                network_cost_per_byte: 0.001,
                cpu_cost_per_op: 0.02, // More CPU for distance calculations
            },
        );

        // Graph traversal varies by depth
        cost_models.insert(
            ModelType::Graph,
            CostModel {
                row_scan_cost: 0.3,      // CSR is efficient
                index_lookup_cost: 0.01, // Very fast node lookup
                network_cost_per_byte: 0.001,
                cpu_cost_per_op: 0.005,
            },
        );

        // Document queries depend on indexes
        cost_models.insert(
            ModelType::Document,
            CostModel {
                row_scan_cost: 1.0,
                index_lookup_cost: 0.2,
                network_cost_per_byte: 0.002, // Larger documents
                cpu_cost_per_op: 0.01,
            },
        );

        // RDBMS is baseline
        cost_models.insert(ModelType::Relational, CostModel::default());

        // Observability is optimized for time-range
        cost_models.insert(
            ModelType::Observability,
            CostModel {
                row_scan_cost: 0.2, // Partitioned by time
                index_lookup_cost: 0.1,
                network_cost_per_byte: 0.001,
                cpu_cost_per_op: 0.005,
            },
        );

        Self {
            cost_models,
            next_node_id: std::sync::atomic::AtomicUsize::new(0),
            advanced_cost_estimator: AdvancedCostEstimator::new(),
            cardinality_estimator: CardinalityEstimator::new(),
            join_order_optimizer: JoinOrderOptimizer::new(),
            plan_cache: std::sync::Arc::new(PlanCache::default()),
            statistics_provider: None,
            runtime_stats: std::sync::Arc::new(RuntimeStatisticsCollector::default()),
            predicate_selectivity_policy: PredicateSelectivityPolicy::default(),
        }
    }

    /// Create an optimizer with a custom statistics provider (builder pattern)
    pub fn with_statistics_provider(
        mut self,
        provider: std::sync::Arc<dyn StatisticsProvider>,
    ) -> Self {
        self.statistics_provider = Some(provider);
        self
    }

    /// Create an optimizer with a custom runtime statistics collector
    pub fn with_runtime_stats(mut self, stats: std::sync::Arc<RuntimeStatisticsCollector>) -> Self {
        self.runtime_stats = stats;
        self
    }

    /// Create an optimizer with explicit predicate selectivity policy.
    pub fn with_predicate_selectivity_policy(mut self, policy: PredicateSelectivityPolicy) -> Self {
        debug_assert!(
            policy.validate().is_ok(),
            "invalid predicate selectivity policy: {:?}",
            policy.validate().err()
        );
        self.predicate_selectivity_policy = policy;
        self
    }

    /// Get a reference to the runtime statistics collector for sharing with executors
    pub fn runtime_stats(&self) -> &std::sync::Arc<RuntimeStatisticsCollector> {
        &self.runtime_stats
    }

    /// Record execution feedback and adaptively tune the cost model.
    ///
    /// Call this after each query execution with observed metrics.
    /// The optimizer will:
    /// 1. Update cardinality correction ratios
    /// 2. Update cost model parameters (cpu, io, memory costs)
    /// 3. Invalidate cached plans that show performance regression
    pub fn record_execution_feedback(&mut self, feedback: ExecutionFeedback) {
        // 1. Update runtime stats collector
        self.runtime_stats.record_feedback(&feedback);

        // 2. Update cardinality estimator with actual vs estimated
        self.cardinality_estimator.record_actual_cardinality(
            feedback.operation_key.clone(),
            feedback.estimated_cardinality,
            feedback.actual_cardinality,
        );

        // 3. Update cost model parameters from hardware observations
        self.advanced_cost_estimator.update_from_feedback(&feedback);

        // 4. Invalidate cached plans if performance regressed significantly
        if self.runtime_stats.should_invalidate_plan(
            &feedback.operation_key,
            feedback.estimated_cost,
            feedback.actual_latency_ms,
        ) {
            // Extract collection/target segments from the operation key and
            // invalidate any cached plan that references them
            for segment in feedback.operation_key.split(':').skip(1) {
                self.plan_cache.invalidate_for_target(segment);
            }
        }
    }

    /// Get calibrated cardinality estimate incorporating runtime feedback
    pub fn calibrated_cardinality(&self, operation_key: &str, base_estimate: u64) -> u64 {
        // First check runtime stats for correction ratio
        if let Some(correction) = self.runtime_stats.cardinality_correction(operation_key) {
            let calibrated = (base_estimate as f64 * correction) as u64;
            return calibrated.max(1);
        }
        // Fall back to cardinality estimator's historical data
        self.cardinality_estimator
            .calibrated_estimate(operation_key, base_estimate)
    }

    fn ordered_query_sources<'a>(query: &'a FederatedQuery) -> Vec<QuerySourceRef<'a>> {
        let sql_upper = query.sql.to_ascii_uppercase();
        let mut sources = Vec::new();

        for (ordinal, extension) in query.extensions.iter().enumerate() {
            sources.push((
                Self::query_source_position(
                    query,
                    &sql_upper,
                    QuerySourceRef::Extension { extension, ordinal },
                ),
                ordinal,
                QuerySourceRef::Extension { extension, ordinal },
            ));
        }

        for (ordinal, target) in query.targets.iter().enumerate() {
            if !matches!(
                target.model_type,
                TargetModelType::Table | TargetModelType::Unknown
            ) {
                continue;
            }

            sources.push((
                Self::query_source_position(query, &sql_upper, QuerySourceRef::Target(target)),
                query.extensions.len() + ordinal,
                QuerySourceRef::Target(target),
            ));
        }

        sources.sort_by_key(|(position, ordinal, _)| (*position, *ordinal));
        sources
            .into_iter()
            .map(|(_, _, source)| source)
            .collect::<Vec<_>>()
    }

    fn query_source_position(
        query: &FederatedQuery,
        sql_upper: &str,
        source: QuerySourceRef<'_>,
    ) -> usize {
        match source {
            QuerySourceRef::Extension { ordinal, .. } => query
                .extension_positions
                .get(ordinal)
                .copied()
                .unwrap_or(usize::MAX),
            QuerySourceRef::Target(target) => Self::target_position(sql_upper, target),
        }
    }

    fn extension_alias(query: &FederatedQuery, ordinal: usize) -> Option<String> {
        query.extension_aliases.get(ordinal).cloned().flatten()
    }

    fn target_position(sql_upper: &str, target: &QueryTarget) -> usize {
        let target_upper = target.name.to_uppercase();
        [
            format!("FROM {}", target_upper),
            format!("JOIN {}", target_upper),
            format!(", {}", target_upper),
        ]
        .into_iter()
        .filter_map(|needle| sql_upper.find(&needle))
        .min()
        .or_else(|| sql_upper.find(&target_upper))
        .unwrap_or(usize::MAX)
    }

    fn collect_correlations(plan: &PlanNode) -> Vec<String> {
        let mut correlations = Vec::new();
        Self::collect_correlations_into(plan, &mut correlations);
        correlations
    }

    fn collect_correlations_into(plan: &PlanNode, correlations: &mut Vec<String>) {
        match &plan.node_type {
            PlanNodeType::VectorSearch {
                query_vector_source,
                ..
            }
            | PlanNodeType::RerankSearch {
                query_vector_source,
                ..
            } => Self::collect_vector_source_correlations(query_vector_source, correlations),
            PlanNodeType::HashJoin { left, right, .. }
            | PlanNodeType::IndexJoin { left, right, .. } => {
                Self::collect_correlations_into(left, correlations);
                Self::collect_correlations_into(right, correlations);
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                Self::collect_correlations_into(outer, correlations);
                Self::collect_correlations_into(inner, correlations);
            }
            PlanNodeType::Filter { input, .. }
            | PlanNodeType::Project { input, .. }
            | PlanNodeType::Distinct { input }
            | PlanNodeType::Sort { input, .. }
            | PlanNodeType::Limit { input, .. }
            | PlanNodeType::Aggregate { input, .. } => {
                Self::collect_correlations_into(input, correlations);
            }
            PlanNodeType::Union { inputs, .. } => {
                for input in inputs {
                    Self::collect_correlations_into(input, correlations);
                }
            }
            PlanNodeType::Scan { .. }
            | PlanNodeType::GraphTraversal { .. }
            | PlanNodeType::DocumentQuery { .. }
            | PlanNodeType::ObservabilityQuery { .. } => {}
        }
    }

    fn collect_vector_source_correlations(source: &VectorSource, correlations: &mut Vec<String>) {
        match source {
            VectorSource::ColumnRef { table, column } => {
                let reference = format!("{}.{}", table, column);
                if !correlations.iter().any(|existing| existing == &reference) {
                    correlations.push(reference);
                }
            }
            VectorSource::Subquery(plan) => Self::collect_correlations_into(plan, correlations),
            VectorSource::Literal(_) | VectorSource::Expression(_) => {}
        }
    }

    /// Set statistics provider on an existing optimizer instance
    pub fn set_statistics_provider(&mut self, provider: std::sync::Arc<dyn StatisticsProvider>) {
        self.statistics_provider = Some(provider);
    }

    /// Create an optimizer with a custom plan cache
    pub fn with_plan_cache(mut self, cache: std::sync::Arc<PlanCache>) -> Self {
        self.plan_cache = cache;
        self
    }

    /// Get the plan cache
    pub fn plan_cache(&self) -> &std::sync::Arc<PlanCache> {
        &self.plan_cache
    }

    /// Get the cardinality estimator
    pub fn cardinality_estimator(&self) -> &CardinalityEstimator {
        &self.cardinality_estimator
    }

    /// Get the advanced cost estimator
    pub fn cost_estimator(&self) -> &AdvancedCostEstimator {
        &self.advanced_cost_estimator
    }

    /// Optimize a federated query
    ///
    /// If a statistics provider is configured, uses cardinality-aware
    /// cost estimation for better plan selection.
    pub fn optimize(&self, query: &FederatedQuery) -> Result<FederatedQueryPlan> {
        // If we have a statistics provider, collect stats for referenced collections
        // and delegate to optimize_with_statistics for cardinality-aware planning
        if let Some(ref provider) = self.statistics_provider {
            use crate::query::federated::parser::SqlExtension;
            let mut stats = HashMap::new();
            for ext in &query.extensions {
                let collection_name = match ext {
                    SqlExtension::VectorSearch { collection, .. } => Some(collection.clone()),
                    SqlExtension::DocumentQuery { collection, .. } => Some(collection.clone()),
                    _ => None,
                };
                if let Some(name) = collection_name
                    && let Some(model_stats) = provider.get_statistics(&name)
                {
                    stats.insert(name, model_stats);
                }
            }
            // Also check query targets
            for target in &query.targets {
                if let Some(model_stats) = provider.get_statistics(&target.name) {
                    stats.insert(target.name.clone(), model_stats);
                }
            }
            if !stats.is_empty() {
                return self.optimize_with_statistics(query, &stats);
            }
        }

        // Check plan cache first
        let cache_key = PlanCacheKey::from_query(query);
        if let Some(cached_plan) = self.plan_cache.get(&cache_key) {
            tracing::debug!("Using cached plan for query");
            return Ok(cached_plan);
        }

        // Build initial logical plan
        let logical_plan = self.build_logical_plan(query)?;

        // Apply optimizations
        let optimized = self.apply_optimizations(logical_plan)?;

        // Build physical plan
        let physical_plan = self.build_physical_plan(optimized)?;

        // Cache the plan
        self.plan_cache.put(cache_key, physical_plan.clone());

        Ok(physical_plan)
    }

    /// Optimize a federated query with statistics (enhanced cost estimation)
    pub fn optimize_with_statistics(
        &self,
        query: &FederatedQuery,
        stats: &HashMap<String, ModelStatistics>,
    ) -> Result<FederatedQueryPlan> {
        // Check plan cache first
        let cache_key = PlanCacheKey::from_query(query);
        if let Some(cached_plan) = self.plan_cache.get(&cache_key) {
            return Ok(cached_plan);
        }

        // Build initial logical plan with statistics-aware costs
        let logical_plan = self.build_logical_plan_with_statistics(query, stats)?;

        // Apply optimizations including DP-based join ordering
        let optimized = self.apply_optimizations_with_statistics(logical_plan, stats)?;

        // Build physical plan
        let physical_plan = self.build_physical_plan(optimized)?;

        // Cache the plan
        self.plan_cache.put(cache_key, physical_plan.clone());

        Ok(physical_plan)
    }

    /// Build logical plan with statistics-aware cost estimation
    fn build_logical_plan_with_statistics(
        &self,
        query: &FederatedQuery,
        stats: &HashMap<String, ModelStatistics>,
    ) -> Result<PlanNode> {
        match query.query_type {
            QueryType::VectorSearch => self.plan_vector_search_with_statistics(query, stats),
            QueryType::GraphQuery => self.plan_graph_query_with_statistics(query, stats),
            QueryType::DocumentQuery => self.plan_document_query_with_statistics(query, stats),
            QueryType::Federated => self.plan_federated_query_with_statistics(query, stats),
            _ => self.build_logical_plan(query),
        }
    }

    /// Plan vector search with statistics
    fn plan_vector_search_with_statistics(
        &self,
        query: &FederatedQuery,
        stats: &HashMap<String, ModelStatistics>,
    ) -> Result<PlanNode> {
        // Extract collection name and top_k from extensions
        let (collection, query_vector_source, top_k) = query
            .extensions
            .iter()
            .find_map(|ext| {
                if let SqlExtension::VectorSearch {
                    collection,
                    query_vector,
                    top_k,
                } = ext
                {
                    Some((
                        collection.clone(),
                        vector_query_parsing::vector_source_from_query(query_vector),
                        *top_k,
                    ))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ("unknown".to_string(), VectorSource::Literal(vec![]), 10));

        // Get statistics for this collection
        let (cost, rows) = if let Some(ModelStatistics::Vector(vec_stats)) = stats.get(&collection)
        {
            let cost = self
                .advanced_cost_estimator
                .vector_search_cost(vec_stats, top_k);
            let rows = self
                .cardinality_estimator
                .estimate_vector_search_cardinality(top_k, vec_stats);
            (cost, rows)
        } else {
            // Default estimation
            (100.0, top_k as u64)
        };

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::VectorSearch {
                collection: collection.clone(),
                top_k,
                query_vector_source,
            },
            estimated_cost: cost,
            estimated_rows: rows,
            output_columns: vec!["id".to_string(), "score".to_string()],
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan graph query with statistics
    fn plan_graph_query_with_statistics(
        &self,
        query: &FederatedQuery,
        stats: &HashMap<String, ModelStatistics>,
    ) -> Result<PlanNode> {
        // Extract cypher query
        let cypher = query
            .extensions
            .iter()
            .find_map(|ext| {
                if let SqlExtension::GraphQuery { cypher } = ext {
                    Some(cypher.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let graph_shape = GraphQueryPlanShape::from_cypher(&cypher);

        // Estimate max depth from cypher (simple heuristic)
        let max_depth = cypher.matches("->").count().max(1);
        let graph_name = graph_shape.graph_name.as_str();

        // Get statistics
        let (cost, rows) = if let Some(ModelStatistics::Graph(graph_stats)) = stats.get(graph_name)
        {
            let cost = self
                .advanced_cost_estimator
                .graph_traversal_cost(graph_stats, max_depth);
            let rows = self
                .cardinality_estimator
                .estimate_graph_traversal_cardinality(graph_stats, max_depth);
            (cost, rows)
        } else {
            // Default estimation
            let default_stats = FederatedGraphStats {
                node_count: 10000,
                edge_count: 50000,
                avg_degree: 5.0,
                max_depth: Some(6),
                has_label_index: true,
            };
            let cost = self
                .advanced_cost_estimator
                .graph_traversal_cost(&default_stats, max_depth);
            (cost, 100)
        };

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::GraphTraversal {
                cypher,
                start_nodes: None, // Will be populated from query parsing
                source_alias: None,
            },
            estimated_cost: cost,
            estimated_rows: rows,
            output_columns: graph_shape.output_columns,
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan document query with statistics
    fn plan_document_query_with_statistics(
        &self,
        query: &FederatedQuery,
        stats: &HashMap<String, ModelStatistics>,
    ) -> Result<PlanNode> {
        // Extract collection and filter
        let (collection, filter) = query
            .extensions
            .iter()
            .find_map(|ext| {
                if let SqlExtension::DocumentQuery { collection, filter } = ext {
                    Some((collection.clone(), filter.clone()))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ("unknown".to_string(), None));

        // Estimate filter complexity and fields
        let filter_complexity = filter.as_ref().map_or(1, |f| {
            f.matches("AND").count() + f.matches("OR").count() + 1
        });
        let filter_fields: Vec<String> = vec![]; // Would need proper filter parsing

        // Get statistics
        let (cost, rows) =
            if let Some(ModelStatistics::Document(doc_stats)) = stats.get(&collection) {
                let cost = self.advanced_cost_estimator.document_query_cost(
                    doc_stats,
                    &filter_fields,
                    filter_complexity,
                );
                let selectivity = 0.1_f64.powi(filter_complexity as i32);
                let rows = self
                    .cardinality_estimator
                    .estimate_document_query_cardinality(doc_stats, selectivity);
                (cost, rows)
            } else {
                (100.0, 100)
            };

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::DocumentQuery {
                collection,
                filter,
                source_alias: None,
            },
            estimated_cost: cost,
            estimated_rows: rows,
            output_columns: vec!["id".to_string(), "document".to_string()],
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan federated query with statistics and DP join ordering
    fn plan_federated_query_with_statistics(
        &self,
        query: &FederatedQuery,
        stats: &HashMap<String, ModelStatistics>,
    ) -> Result<PlanNode> {
        // Build sub-plans for each extension/target
        let mut sub_plans = Vec::new();

        for source in Self::ordered_query_sources(query) {
            let sub_plan = match source {
                QuerySourceRef::Extension {
                    extension:
                        SqlExtension::VectorSearch {
                            collection,
                            query_vector,
                            top_k,
                        },
                    ..
                } => {
                    let vec_stats = stats.get(collection).and_then(|s| {
                        if let ModelStatistics::Vector(vs) = s {
                            Some(vs)
                        } else {
                            None
                        }
                    });
                    let (cost, rows) = if let Some(vs) = vec_stats {
                        (
                            self.advanced_cost_estimator.vector_search_cost(vs, *top_k),
                            self.cardinality_estimator
                                .estimate_vector_search_cardinality(*top_k, vs),
                        )
                    } else {
                        (100.0, *top_k as u64)
                    };

                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::VectorSearch {
                            collection: collection.clone(),
                            top_k: *top_k,
                            query_vector_source: vector_query_parsing::vector_source_from_query(
                                query_vector,
                            ),
                        },
                        estimated_cost: cost,
                        estimated_rows: rows,
                        output_columns: vec!["id".to_string(), "score".to_string()],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Extension {
                    extension: SqlExtension::GraphQuery { cypher },
                    ordinal,
                } => {
                    let graph_shape = GraphQueryPlanShape::from_cypher(cypher);
                    let max_depth = cypher.matches("->").count().max(1);
                    // Use default stats for graph
                    let default_stats = FederatedGraphStats::default();
                    let cost = self
                        .advanced_cost_estimator
                        .graph_traversal_cost(&default_stats, max_depth);

                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::GraphTraversal {
                            cypher: cypher.clone(),
                            start_nodes: None, // Will be populated from query parsing
                            source_alias: Self::extension_alias(query, ordinal),
                        },
                        estimated_cost: cost,
                        estimated_rows: 100,
                        output_columns: graph_shape.output_columns,
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Extension {
                    extension: SqlExtension::DocumentQuery { collection, filter },
                    ordinal,
                } => {
                    let doc_stats = stats.get(collection).and_then(|s| {
                        if let ModelStatistics::Document(ds) = s {
                            Some(ds)
                        } else {
                            None
                        }
                    });
                    let (cost, rows) = if let Some(ds) = doc_stats {
                        (
                            self.advanced_cost_estimator.document_query_cost(ds, &[], 1),
                            ds.document_count,
                        )
                    } else {
                        (100.0, 1000)
                    };

                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::DocumentQuery {
                            collection: collection.clone(),
                            filter: filter.clone(),
                            source_alias: Self::extension_alias(query, ordinal),
                        },
                        estimated_cost: cost,
                        estimated_rows: rows,
                        output_columns: vec!["id".to_string(), "document".to_string()],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Extension {
                    extension: SqlExtension::Logs { namespace },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::ObservabilityQuery {
                        namespace: namespace.clone(),
                        query_type: ObservabilityQueryType::Logs,
                        time_range: None,
                    },
                    estimated_cost: 20.0,
                    estimated_rows: 1000,
                    output_columns: vec![
                        "timestamp".to_string(),
                        "level".to_string(),
                        "message".to_string(),
                    ],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension: SqlExtension::Metrics { namespace },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::ObservabilityQuery {
                        namespace: namespace.clone(),
                        query_type: ObservabilityQueryType::Metrics,
                        time_range: None,
                    },
                    estimated_cost: 20.0,
                    estimated_rows: 500,
                    output_columns: vec![
                        "timestamp".to_string(),
                        "metric_name".to_string(),
                        "value".to_string(),
                    ],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension: SqlExtension::Traces { namespace },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::ObservabilityQuery {
                        namespace: namespace.clone(),
                        query_type: ObservabilityQueryType::Traces,
                        time_range: None,
                    },
                    estimated_cost: 20.0,
                    estimated_rows: 500,
                    output_columns: vec![
                        "trace_id".to_string(),
                        "span_id".to_string(),
                        "operation".to_string(),
                        "duration_ns".to_string(),
                    ],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension:
                        SqlExtension::VectorDistance {
                            left_column,
                            right_literal,
                        },
                    ..
                } => {
                    let target = query
                        .targets
                        .first()
                        .map_or("default".to_string(), |t| t.name.clone());
                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::VectorSearch {
                            collection: target,
                            top_k: 10,
                            query_vector_source: vector_query_parsing::vector_source_from_literal(
                                right_literal,
                            ),
                        },
                        estimated_cost: 10.0,
                        estimated_rows: 10,
                        output_columns: vec!["id".to_string(), left_column.clone()],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Target(target) => {
                    let rows = stats
                        .get(&target.name)
                        .map_or(1000, |s| s.estimated_count());

                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::Scan {
                            target: target.name.clone(),
                            model_type: ModelType::Relational,
                            predicates: vec![],
                        },
                        estimated_cost: rows as f64 * 0.1,
                        estimated_rows: rows,
                        output_columns: vec!["*".to_string()],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Extension {
                    extension:
                        SqlExtension::RerankSearch {
                            collection,
                            query_text,
                            query_vector,
                            k,
                            rank_profile,
                        },
                    ..
                } => {
                    // R-7c.4c.1: emit RerankSearch plan node; execution
                    // dispatches to the rank pipeline (first-phase
                    // retrieval + optional second-phase rescore).
                    let vec_stats = stats.get(collection).and_then(|s| {
                        if let ModelStatistics::Vector(vs) = s {
                            Some(vs)
                        } else {
                            None
                        }
                    });
                    let (cost, rows) = if let Some(vs) = vec_stats {
                        (
                            self.advanced_cost_estimator.vector_search_cost(vs, *k),
                            self.cardinality_estimator
                                .estimate_vector_search_cardinality(*k, vs),
                        )
                    } else {
                        (100.0, *k as u64)
                    };
                    // Second-phase rescoring adds latency on top of
                    // retrieval — bump cost when a profile is set.
                    let rerank_cost_factor = if rank_profile.is_some() { 2.5 } else { 1.0 };
                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::RerankSearch {
                            collection: collection.clone(),
                            query_text: query_text.clone(),
                            query_vector_source: vector_query_parsing::vector_source_from_query(
                                query_vector,
                            ),
                            k: *k,
                            rank_profile: rank_profile.clone(),
                        },
                        estimated_cost: cost * rerank_cost_factor,
                        estimated_rows: rows,
                        output_columns: vec![
                            "id".to_string(),
                            "score".to_string(),
                            "phase".to_string(),
                            "match_features".to_string(),
                            "summary_features".to_string(),
                        ],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
            };
            sub_plans.push(sub_plan);
        }

        if sub_plans.is_empty() {
            return Err(anyhow!("No sub-plans generated for federated query"));
        }

        if sub_plans.len() == 1 {
            return Ok(sub_plans.remove(0));
        }

        if query.sql.to_uppercase().contains("LATERAL") {
            let mut result = sub_plans.remove(0);
            for right in sub_plans {
                let correlation = Self::collect_correlations(&right);
                let estimated_cost = result.estimated_cost + right.estimated_cost + 200.0;
                let estimated_rows = result
                    .estimated_rows
                    .saturating_mul(right.estimated_rows.max(1));
                result = PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::NestedLoopJoin {
                        outer: Box::new(result),
                        inner: Box::new(right),
                        correlation,
                    },
                    estimated_cost,
                    estimated_rows,
                    output_columns: vec!["*".to_string()],
                    required_capabilities: CapabilitySet::new(),
                };
            }
            return Ok(result);
        }

        // Use DP-based join order optimization
        let join_predicates: Vec<(usize, usize, Vec<(String, String)>)> = vec![];
        self.join_order_optimizer.find_optimal_join_order(
            &sub_plans,
            &join_predicates,
            &self.next_node_id,
        )
    }

    /// Apply optimizations with statistics
    fn apply_optimizations_with_statistics(
        &self,
        plan: PlanNode,
        _stats: &HashMap<String, ModelStatistics>,
    ) -> Result<PlanNode> {
        // Apply standard optimizations
        let optimized = self.apply_optimizations(plan)?;
        Ok(optimized)
    }

    /// Build initial logical plan from parsed query
    fn build_logical_plan(&self, query: &FederatedQuery) -> Result<PlanNode> {
        let plan = match query.query_type {
            QueryType::Sql => self.plan_sql_query(query),
            QueryType::VectorSearch => self.plan_vector_search(query),
            QueryType::GraphQuery => self.plan_graph_query(query),
            QueryType::DocumentQuery => self.plan_document_query(query),
            QueryType::LogQuery | QueryType::MetricQuery | QueryType::TraceQuery => {
                self.plan_observability_query(query)
            }
            QueryType::Federated => self.plan_federated_query(query),
            // R-7c.4c.1: RERANK(...) SRF is expressed as a
            // SqlExtension::RerankSearch source. Route through
            // plan_federated_query so the per-source match arm fires
            // and emits PlanNodeType::RerankSearch (instead of the
            // legacy Scan/VectorSearch fallback this branch used
            // before the variant existed).
            QueryType::RerankSearch => self.plan_federated_query(query),
        }?;

        self.apply_sql_clauses(plan, query)
    }

    #[allow(clippy::nonminimal_bool)]
    fn apply_sql_clauses(&self, mut plan: PlanNode, query: &FederatedQuery) -> Result<PlanNode> {
        let has_distinct = sql_parsing::select_has_distinct(&query.sql);
        if sql_parsing::find_top_level_keyword(&query.sql, "UNION").is_some() {
            return Err(anyhow!(
                "UNION queries are not yet supported in federated SQL execution"
            ));
        }
        if sql_parsing::find_top_level_keyword(&query.sql, "HAVING").is_some() {
            return Err(anyhow!(
                "HAVING is not yet supported in federated SQL execution"
            ));
        }

        if let Some(predicate) = sql_parsing::extract_where_predicate(&query.sql) {
            let output_columns = plan.output_columns.clone();
            let estimated_cost = plan.estimated_cost * 1.05;
            let estimated_rows = plan.estimated_rows;
            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Filter {
                    input: Box::new(plan),
                    predicate,
                },
                estimated_cost,
                estimated_rows,
                output_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        let select_items = sql_parsing::extract_select_items(&query.sql);
        let group_by = sql_parsing::extract_group_by_columns(&query.sql);
        let aggregates = select_items
            .iter()
            .filter_map(sql_parsing::parse_aggregate_expr)
            .collect::<Vec<_>>();
        let has_group_by = !group_by.is_empty();
        let has_aggregate_projection = !aggregates.is_empty();
        if has_group_by {
            let invalid_group_projection = select_items
                .iter()
                .filter(|item| sql_parsing::parse_aggregate_expr(item).is_none())
                .any(|item| !group_by.contains(&item.expression));

            if invalid_group_projection {
                return Err(anyhow!(
                    "GROUP BY select items must reference grouped columns or aggregate expressions"
                ));
            }
        }

        if has_group_by || has_aggregate_projection {
            let aggregate_columns = group_by
                .iter()
                .cloned()
                .chain(aggregates.iter().map(|aggregate| aggregate.alias.clone()))
                .collect::<Vec<_>>();
            let estimated_rows = if has_group_by {
                ((plan.estimated_rows as f64) * 0.5).ceil() as u64
            } else {
                1
            };
            let estimated_rows = estimated_rows.max(u64::from(plan.estimated_rows > 0));
            let estimated_cost = if has_group_by {
                plan.estimated_cost * 1.15
            } else {
                plan.estimated_cost * 1.1
            };

            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Aggregate {
                    input: Box::new(plan),
                    group_by,
                    aggregates,
                },
                estimated_cost,
                estimated_rows,
                output_columns: aggregate_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        let projection_sources = select_items
            .iter()
            .map(|item| {
                sql_parsing::parse_aggregate_expr(item)
                    .map_or_else(|| item.expression.clone(), |aggregate| aggregate.alias)
            })
            .collect::<Vec<_>>();
        let projection_output_columns = select_items
            .iter()
            .map(|item| {
                item.alias
                    .clone()
                    .unwrap_or_else(|| item.expression.clone())
            })
            .collect::<Vec<_>>();
        if !has_aggregate_projection
            && !has_group_by
            && !projection_sources.is_empty()
            && !(projection_sources.len() == 1 && projection_sources[0] == "*")
            && !projection_sources
                .iter()
                .any(|column| column.ends_with(".*"))
        {
            let estimated_cost = plan.estimated_cost;
            let estimated_rows = plan.estimated_rows;
            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Project {
                    input: Box::new(plan),
                    columns: projection_sources.clone(),
                },
                estimated_cost,
                estimated_rows,
                output_columns: projection_output_columns.clone(),
                required_capabilities: CapabilitySet::new(),
            };
        }

        if has_group_by
            && !projection_sources.is_empty()
            && !(projection_sources.len() == 1 && projection_sources[0] == "*")
            && !projection_sources
                .iter()
                .any(|column| column.ends_with(".*"))
            && (projection_sources != plan.output_columns
                || projection_output_columns != plan.output_columns)
        {
            let estimated_cost = plan.estimated_cost;
            let estimated_rows = plan.estimated_rows;
            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Project {
                    input: Box::new(plan),
                    columns: projection_sources,
                },
                estimated_cost,
                estimated_rows,
                output_columns: projection_output_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        if has_distinct {
            let output_columns = plan.output_columns.clone();
            let estimated_cost = plan.estimated_cost * 1.05;
            let estimated_rows = ((plan.estimated_rows as f64) * 0.8).ceil() as u64;
            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Distinct {
                    input: Box::new(plan),
                },
                estimated_cost,
                estimated_rows,
                output_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        let order_by = sql_parsing::extract_order_by(&query.sql);
        if !order_by.is_empty() {
            let output_columns = plan.output_columns.clone();
            let estimated_cost = plan.estimated_cost * 1.1;
            let estimated_rows = plan.estimated_rows;
            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Sort {
                    input: Box::new(plan),
                    order_by,
                },
                estimated_cost,
                estimated_rows,
                output_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        let (limit, offset) = sql_parsing::extract_limit_offset(&query.sql);
        if let Some(limit) = limit {
            let output_columns = plan.output_columns.clone();
            let estimated_cost = plan.estimated_cost;
            let estimated_rows = plan.estimated_rows.min(limit as u64);
            plan = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::Limit {
                    input: Box::new(plan),
                    limit,
                    offset,
                },
                estimated_cost,
                estimated_rows,
                output_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        Ok(plan)
    }

    /// Plan a simple SQL query
    fn plan_sql_query(&self, query: &FederatedQuery) -> Result<PlanNode> {
        let target = query
            .targets
            .first()
            .map_or_else(|| "unknown".to_string(), |t| t.name.clone());

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::Scan {
                target: target.clone(),
                model_type: ModelType::Relational,
                predicates: vec![],
            },
            estimated_cost: 100.0,
            estimated_rows: 1000,
            output_columns: vec!["*".to_string()],
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan a vector search query
    fn plan_vector_search(&self, query: &FederatedQuery) -> Result<PlanNode> {
        let (collection, query_vector_source, top_k) = query
            .extensions
            .iter()
            .find_map(|ext| match ext {
                SqlExtension::VectorSearch {
                    collection,
                    query_vector,
                    top_k,
                } => Some((
                    collection.clone(),
                    vector_query_parsing::vector_source_from_query(query_vector),
                    *top_k,
                )),
                SqlExtension::VectorDistance { right_literal, .. } => {
                    // Extract from query targets
                    query.targets.first().map(|t| {
                        (
                            t.name.clone(),
                            vector_query_parsing::vector_source_from_literal(right_literal),
                            10,
                        )
                    })
                }
                _ => None,
            })
            .unwrap_or(("default".to_string(), VectorSource::Literal(vec![]), 10));

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::VectorSearch {
                collection,
                top_k,
                query_vector_source,
            },
            estimated_cost: 10.0,
            estimated_rows: top_k as u64,
            output_columns: vec!["id".to_string(), "score".to_string()],
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan a graph query
    fn plan_graph_query(&self, query: &FederatedQuery) -> Result<PlanNode> {
        let cypher = query
            .extensions
            .iter()
            .find_map(|ext| match ext {
                SqlExtension::GraphQuery { cypher } => Some(cypher.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let graph_shape = GraphQueryPlanShape::from_cypher(&cypher);

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::GraphTraversal {
                cypher,
                start_nodes: None,
                source_alias: None,
            },
            estimated_cost: 50.0,
            estimated_rows: 100,
            output_columns: graph_shape.output_columns,
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan a document query
    fn plan_document_query(&self, query: &FederatedQuery) -> Result<PlanNode> {
        let (collection, filter) = query
            .extensions
            .iter()
            .find_map(|ext| match ext {
                SqlExtension::DocumentQuery { collection, filter } => {
                    Some((collection.clone(), filter.clone()))
                }
                _ => None,
            })
            .unwrap_or(("default".to_string(), None));

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::DocumentQuery {
                collection,
                filter,
                source_alias: None,
            },
            estimated_cost: 30.0,
            estimated_rows: 500,
            output_columns: vec!["id".to_string(), "document".to_string()],
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan an observability query
    fn plan_observability_query(&self, query: &FederatedQuery) -> Result<PlanNode> {
        let (namespace, query_type) = query
            .extensions
            .iter()
            .find_map(|ext| match ext {
                SqlExtension::Logs { namespace } => {
                    Some((namespace.clone(), ObservabilityQueryType::Logs))
                }
                SqlExtension::Metrics { namespace } => {
                    Some((namespace.clone(), ObservabilityQueryType::Metrics))
                }
                SqlExtension::Traces { namespace } => {
                    Some((namespace.clone(), ObservabilityQueryType::Traces))
                }
                _ => None,
            })
            .unwrap_or(("default".to_string(), ObservabilityQueryType::Logs));
        let output_columns = match &query_type {
            ObservabilityQueryType::Logs => vec![
                "timestamp".to_string(),
                "level".to_string(),
                "message".to_string(),
            ],
            ObservabilityQueryType::Metrics => vec![
                "timestamp".to_string(),
                "metric_name".to_string(),
                "value".to_string(),
            ],
            ObservabilityQueryType::Traces => vec![
                "trace_id".to_string(),
                "span_id".to_string(),
                "operation".to_string(),
                "duration_ns".to_string(),
            ],
        };

        Ok(PlanNode {
            id: self.next_id(),
            node_type: PlanNodeType::ObservabilityQuery {
                namespace,
                query_type,
                time_range: None, // Deferred: Extract from WHERE clause
            },
            estimated_cost: 20.0,
            estimated_rows: 1000,
            output_columns,
            required_capabilities: CapabilitySet::new(),
        })
    }

    /// Plan a federated (cross-model) query
    fn plan_federated_query(&self, query: &FederatedQuery) -> Result<PlanNode> {
        // Build sub-plans for each extension
        let mut sub_plans: Vec<PlanNode> = Vec::new();

        for source in Self::ordered_query_sources(query) {
            let sub_plan = match source {
                QuerySourceRef::Extension {
                    extension:
                        SqlExtension::VectorSearch {
                            collection,
                            query_vector,
                            top_k,
                        },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::VectorSearch {
                        collection: collection.clone(),
                        top_k: *top_k,
                        query_vector_source: vector_query_parsing::vector_source_from_query(
                            query_vector,
                        ),
                    },
                    estimated_cost: 10.0,
                    estimated_rows: *top_k as u64,
                    output_columns: vec!["id".to_string(), "score".to_string()],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension: SqlExtension::GraphQuery { cypher },
                    ordinal,
                } => {
                    let graph_shape = GraphQueryPlanShape::from_cypher(cypher);
                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::GraphTraversal {
                            cypher: cypher.clone(),
                            start_nodes: None,
                            source_alias: Self::extension_alias(query, ordinal),
                        },
                        estimated_cost: 50.0,
                        estimated_rows: 100,
                        output_columns: graph_shape.output_columns,
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Extension {
                    extension: SqlExtension::DocumentQuery { collection, filter },
                    ordinal,
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::DocumentQuery {
                        collection: collection.clone(),
                        filter: filter.clone(),
                        source_alias: Self::extension_alias(query, ordinal),
                    },
                    estimated_cost: 30.0,
                    estimated_rows: 500,
                    output_columns: vec!["id".to_string(), "document".to_string()],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension: SqlExtension::Logs { namespace },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::ObservabilityQuery {
                        namespace: namespace.clone(),
                        query_type: ObservabilityQueryType::Logs,
                        time_range: None,
                    },
                    estimated_cost: 20.0,
                    estimated_rows: 1000,
                    output_columns: vec![
                        "timestamp".to_string(),
                        "level".to_string(),
                        "message".to_string(),
                    ],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension: SqlExtension::Metrics { namespace },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::ObservabilityQuery {
                        namespace: namespace.clone(),
                        query_type: ObservabilityQueryType::Metrics,
                        time_range: None,
                    },
                    estimated_cost: 20.0,
                    estimated_rows: 500,
                    output_columns: vec![
                        "timestamp".to_string(),
                        "metric_name".to_string(),
                        "value".to_string(),
                    ],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension: SqlExtension::Traces { namespace },
                    ..
                } => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::ObservabilityQuery {
                        namespace: namespace.clone(),
                        query_type: ObservabilityQueryType::Traces,
                        time_range: None,
                    },
                    estimated_cost: 20.0,
                    estimated_rows: 500,
                    output_columns: vec![
                        "trace_id".to_string(),
                        "span_id".to_string(),
                        "operation".to_string(),
                        "duration_ns".to_string(),
                    ],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension:
                        SqlExtension::VectorDistance {
                            left_column,
                            right_literal,
                        },
                    ..
                } => {
                    let target = query
                        .targets
                        .first()
                        .map_or("default".to_string(), |t| t.name.clone());
                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::VectorSearch {
                            collection: target,
                            top_k: 10,
                            query_vector_source: vector_query_parsing::vector_source_from_literal(
                                right_literal,
                            ),
                        },
                        estimated_cost: 10.0,
                        estimated_rows: 10,
                        output_columns: vec!["id".to_string(), left_column.clone()],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
                QuerySourceRef::Target(target) => PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Scan {
                        target: target.name.clone(),
                        model_type: ModelType::Relational,
                        predicates: vec![],
                    },
                    estimated_cost: 100.0,
                    estimated_rows: 1000,
                    output_columns: vec!["*".to_string()],
                    required_capabilities: CapabilitySet::new(),
                },
                QuerySourceRef::Extension {
                    extension:
                        SqlExtension::RerankSearch {
                            collection,
                            query_text,
                            query_vector,
                            k,
                            rank_profile,
                        },
                    ..
                } => {
                    // R-7c.4c.1: parallel arm to the
                    // plan_federated_query_with_statistics path, but
                    // without a stats-table lookup (cost falls back to
                    // the default vector-search heuristic).
                    let rerank_cost_factor = if rank_profile.is_some() { 2.5 } else { 1.0 };
                    PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::RerankSearch {
                            collection: collection.clone(),
                            query_text: query_text.clone(),
                            query_vector_source: vector_query_parsing::vector_source_from_query(
                                query_vector,
                            ),
                            k: *k,
                            rank_profile: rank_profile.clone(),
                        },
                        estimated_cost: 10.0 * rerank_cost_factor,
                        estimated_rows: *k as u64,
                        output_columns: vec![
                            "id".to_string(),
                            "score".to_string(),
                            "phase".to_string(),
                            "match_features".to_string(),
                            "summary_features".to_string(),
                        ],
                        required_capabilities: CapabilitySet::new(),
                    }
                }
            };
            sub_plans.push(sub_plan);
        }

        // If we have multiple sub-plans, join them
        if sub_plans.len() >= 2 {
            let mut result = sub_plans.remove(0);
            for right in sub_plans {
                let correlation = Self::collect_correlations(&right);
                result = PlanNode {
                    id: self.next_id(),
                    node_type: if query.sql.to_uppercase().contains("LATERAL") {
                        PlanNodeType::NestedLoopJoin {
                            outer: Box::new(result),
                            inner: Box::new(right),
                            correlation,
                        }
                    } else {
                        PlanNodeType::HashJoin {
                            left: Box::new(result),
                            right: Box::new(right),
                            join_keys: vec![("id".to_string(), "id".to_string())],
                            join_type: JoinType::Inner,
                        }
                    },
                    estimated_cost: 200.0,
                    estimated_rows: 100,
                    output_columns: vec!["*".to_string()],
                    required_capabilities: CapabilitySet::new(),
                };
            }
            Ok(result)
        } else if sub_plans.len() == 1 {
            Ok(sub_plans.remove(0))
        } else {
            Err(anyhow!("No valid plan nodes generated"))
        }
    }

    /// Apply optimization rules in order:
    /// 1. Predicate pushdown - push filters closer to data sources
    /// 2. Join reordering - reorder joins based on cost estimation
    /// 3. Projection pushdown - only read needed columns
    /// 4. Parallel execution identification - mark independent subqueries
    fn apply_optimizations(&self, plan: PlanNode) -> Result<PlanNode> {
        let plan = self.push_predicates(plan)?;
        let plan = self.reorder_joins(plan)?;
        let plan = self.push_projections(plan)?;
        let plan = self.identify_parallelism(plan)?;
        Ok(plan)
    }

    // ========================================================================
    // PREDICATE PUSHDOWN
    // ========================================================================

    /// Push predicates (filters) as close to data sources as possible.
    /// This reduces the amount of data flowing through the plan.
    fn push_predicates(&self, plan: PlanNode) -> Result<PlanNode> {
        match plan.node_type {
            PlanNodeType::Filter { input, predicate } => {
                // Try to push the predicate into the input node
                let pushed = self.try_push_predicate_into(*input, predicate.clone())?;
                if pushed.predicate_pushed {
                    Ok(pushed.node)
                } else {
                    // Could not push, keep filter as-is but optimize input
                    let optimized_input = self.push_predicates(pushed.node)?;
                    Ok(PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::Filter {
                            input: Box::new(optimized_input),
                            predicate,
                        },
                        estimated_cost: plan.estimated_cost,
                        estimated_rows: plan.estimated_rows,
                        output_columns: plan.output_columns,
                        required_capabilities: CapabilitySet::new(),
                    })
                }
            }
            PlanNodeType::HashJoin {
                left,
                right,
                join_keys,
                join_type,
            } => {
                // Recursively optimize children
                let left = self.push_predicates(*left)?;
                let right = self.push_predicates(*right)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::HashJoin {
                        left: Box::new(left),
                        right: Box::new(right),
                        join_keys,
                        join_type,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::NestedLoopJoin {
                outer,
                inner,
                correlation,
            } => {
                let outer = self.push_predicates(*outer)?;
                let inner = self.push_predicates(*inner)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::NestedLoopJoin {
                        outer: Box::new(outer),
                        inner: Box::new(inner),
                        correlation,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::IndexJoin {
                left,
                right,
                index_lookup,
            } => {
                let left = self.push_predicates(*left)?;
                let right = self.push_predicates(*right)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::IndexJoin {
                        left: Box::new(left),
                        right: Box::new(right),
                        index_lookup,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Project { input, columns } => {
                let input = self.push_predicates(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Project {
                        input: Box::new(input),
                        columns,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Distinct { input } => {
                let input = self.push_predicates(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Distinct {
                        input: Box::new(input),
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Sort { input, order_by } => {
                let input = self.push_predicates(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Sort {
                        input: Box::new(input),
                        order_by,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Limit {
                input,
                limit,
                offset,
            } => {
                let input = self.push_predicates(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Limit {
                        input: Box::new(input),
                        limit,
                        offset,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let input = self.push_predicates(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Aggregate {
                        input: Box::new(input),
                        group_by,
                        aggregates,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Union { inputs, all } => {
                let inputs: Result<Vec<_>> = inputs
                    .into_iter()
                    .map(|input| self.push_predicates(input))
                    .collect();
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Union {
                        inputs: inputs?,
                        all,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            // Leaf nodes - no further pushdown possible
            _ => Ok(plan),
        }
    }

    /// Try to push a predicate into a plan node
    fn try_push_predicate_into(
        &self,
        node: PlanNode,
        predicate: Predicate,
    ) -> Result<PredicatePushResult> {
        match &node.node_type {
            PlanNodeType::Scan {
                target,
                model_type,
                predicates,
            } => {
                // Check if predicate references columns from this scan
                if self.predicate_references_target(&predicate, target) {
                    // Merge predicate into scan
                    let mut new_predicates = predicates.clone();
                    new_predicates.push(predicate);

                    // Estimate reduced row count based on predicate selectivity
                    let last_predicate = new_predicates.last().ok_or_else(|| {
                        VectorDBError::Internal("No predicates found after push".to_string())
                    })?;
                    let selectivity = self.estimate_predicate_selectivity(last_predicate);
                    let new_rows = ((node.estimated_rows as f64) * selectivity) as u64;

                    Ok(PredicatePushResult {
                        node: PlanNode {
                            id: self.next_id(),
                            node_type: PlanNodeType::Scan {
                                target: target.clone(),
                                model_type: *model_type,
                                predicates: new_predicates,
                            },
                            estimated_cost: node.estimated_cost * selectivity,
                            estimated_rows: new_rows.max(1),
                            output_columns: node.output_columns.clone(),
                            required_capabilities: CapabilitySet::new(),
                        },
                        predicate_pushed: true,
                    })
                } else {
                    Ok(PredicatePushResult {
                        node,
                        predicate_pushed: false,
                    })
                }
            }
            PlanNodeType::VectorSearch { collection, .. } => {
                let _ = collection;
                Ok(PredicatePushResult {
                    node,
                    predicate_pushed: false,
                })
            }
            PlanNodeType::DocumentQuery {
                collection,
                filter,
                source_alias,
            } => {
                // Document queries can have filters pushed into them
                if self.predicate_references_target(&predicate, collection) {
                    let new_filter = match filter {
                        Some(existing) => Some(format!(
                            "({}) AND ({})",
                            existing,
                            self.predicate_to_string(&predicate)
                        )),
                        None => Some(self.predicate_to_string(&predicate)),
                    };
                    let selectivity = self.estimate_predicate_selectivity(&predicate);
                    Ok(PredicatePushResult {
                        node: PlanNode {
                            id: self.next_id(),
                            node_type: PlanNodeType::DocumentQuery {
                                collection: collection.clone(),
                                filter: new_filter,
                                source_alias: source_alias.clone(),
                            },
                            estimated_cost: node.estimated_cost * selectivity,
                            estimated_rows: ((node.estimated_rows as f64) * selectivity) as u64,
                            output_columns: node.output_columns.clone(),
                            required_capabilities: CapabilitySet::new(),
                        },
                        predicate_pushed: true,
                    })
                } else {
                    Ok(PredicatePushResult {
                        node,
                        predicate_pushed: false,
                    })
                }
            }
            PlanNodeType::HashJoin {
                left,
                right,
                join_keys,
                join_type,
            } => {
                let predicate_column = predicate.column.clone();
                let left_can_match = self.column_from_left(&predicate_column, left)
                    && !self.column_from_right(&predicate_column, right);
                let right_can_match = self.column_from_right(&predicate_column, right)
                    && !self.column_from_left(&predicate_column, left);

                if left_can_match {
                    let left_pushed =
                        self.try_push_predicate_into(*left.clone(), predicate.clone())?;
                    if left_pushed.predicate_pushed {
                        return Ok(PredicatePushResult {
                            node: PlanNode {
                                id: self.next_id(),
                                node_type: PlanNodeType::HashJoin {
                                    left: Box::new(left_pushed.node),
                                    right: right.clone(),
                                    join_keys: join_keys.clone(),
                                    join_type: join_type.clone(),
                                },
                                estimated_cost: node.estimated_cost * 0.8,
                                estimated_rows: node.estimated_rows,
                                output_columns: node.output_columns.clone(),
                                required_capabilities: CapabilitySet::new(),
                            },
                            predicate_pushed: true,
                        });
                    }
                }

                if right_can_match {
                    let right_pushed = self.try_push_predicate_into(*right.clone(), predicate)?;
                    if right_pushed.predicate_pushed {
                        return Ok(PredicatePushResult {
                            node: PlanNode {
                                id: self.next_id(),
                                node_type: PlanNodeType::HashJoin {
                                    left: left.clone(),
                                    right: Box::new(right_pushed.node),
                                    join_keys: join_keys.clone(),
                                    join_type: join_type.clone(),
                                },
                                estimated_cost: node.estimated_cost * 0.8,
                                estimated_rows: node.estimated_rows,
                                output_columns: node.output_columns.clone(),
                                required_capabilities: CapabilitySet::new(),
                            },
                            predicate_pushed: true,
                        });
                    }
                }

                Ok(PredicatePushResult {
                    node,
                    predicate_pushed: false,
                })
            }
            _ => Ok(PredicatePushResult {
                node,
                predicate_pushed: false,
            }),
        }
    }

    /// Check if a predicate references a specific target
    fn predicate_references_target(&self, predicate: &Predicate, target: &str) -> bool {
        // Simple heuristic: check if column name starts with target or contains no qualifier
        predicate.column.starts_with(target)
            || predicate.column.starts_with(&format!("{}.", target))
            || !predicate.column.contains('.')
    }

    /// Estimate selectivity of a predicate (fraction of rows that match)
    fn estimate_predicate_selectivity(&self, predicate: &Predicate) -> f64 {
        let policy = &self.predicate_selectivity_policy;
        match predicate.op {
            PredicateOp::Eq => policy.eq,
            PredicateOp::Ne => policy.ne,
            PredicateOp::Lt | PredicateOp::Le | PredicateOp::Gt | PredicateOp::Ge => policy.range,
            PredicateOp::Like => policy.like,
            PredicateOp::In => policy.in_list,
            PredicateOp::IsNull => policy.is_null,
            PredicateOp::IsNotNull => policy.is_not_null,
            PredicateOp::Between => policy.between,
        }
    }

    /// Convert predicate to string representation
    fn predicate_to_string(&self, predicate: &Predicate) -> String {
        let op_str = match predicate.op {
            PredicateOp::Eq => "=",
            PredicateOp::Ne => "!=",
            PredicateOp::Lt => "<",
            PredicateOp::Le => "<=",
            PredicateOp::Gt => ">",
            PredicateOp::Ge => ">=",
            PredicateOp::Like => "LIKE",
            PredicateOp::In => "IN",
            PredicateOp::IsNull => "IS NULL",
            PredicateOp::IsNotNull => "IS NOT NULL",
            PredicateOp::Between => "BETWEEN",
        };

        let value_str = match &predicate.value {
            PredicateValue::String(s) => format!("'{}'", s),
            PredicateValue::Int(i) => i.to_string(),
            PredicateValue::Float(f) => f.to_string(),
            PredicateValue::Bool(b) => b.to_string(),
            PredicateValue::Null => "NULL".to_string(),
            PredicateValue::List(values) => {
                let items: Vec<String> = values
                    .iter()
                    .map(|v| match v {
                        PredicateValue::String(s) => format!("'{}'", s),
                        PredicateValue::Int(i) => i.to_string(),
                        PredicateValue::Float(f) => f.to_string(),
                        PredicateValue::Bool(b) => b.to_string(),
                        _ => "?".to_string(),
                    })
                    .collect();
                format!("({})", items.join(", "))
            }
        };

        format!("{} {} {}", predicate.column, op_str, value_str)
    }

    // ========================================================================
    // JOIN REORDERING
    // ========================================================================

    /// Reorder joins based on cost estimation.
    /// Uses a greedy approach: always join the two relations with lowest combined cost first.
    fn reorder_joins(&self, plan: PlanNode) -> Result<PlanNode> {
        match plan.node_type {
            PlanNodeType::HashJoin {
                left,
                right,
                join_keys,
                join_type,
            } => {
                // First, recursively optimize children
                let left = self.reorder_joins(*left)?;
                let right = self.reorder_joins(*right)?;

                // Extract all leaf relations from both sides
                let mut left_leaves = self.extract_join_leaves(&left);
                let mut right_leaves = self.extract_join_leaves(&right);

                // If we have multiple leaves to join, apply greedy reordering
                if left_leaves.len() + right_leaves.len() > 2 {
                    let mut all_leaves: Vec<PlanNode> = left_leaves
                        .drain(..)
                        .chain(right_leaves.drain(..))
                        .collect();
                    let reordered =
                        self.greedy_join_order(&mut all_leaves, &join_keys, &join_type)?;
                    return Ok(reordered);
                }

                // Simple case: just swap if right is cheaper than left
                let left_cost = self.calculate_total_cost(&left);
                let right_cost = self.calculate_total_cost(&right);

                if right_cost < left_cost && join_type == JoinType::Inner {
                    // Swap for inner joins (commutative)
                    let swapped_keys: Vec<(String, String)> = join_keys
                        .iter()
                        .map(|(l, r)| (r.clone(), l.clone()))
                        .collect();
                    Ok(PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::HashJoin {
                            left: Box::new(right),
                            right: Box::new(left),
                            join_keys: swapped_keys,
                            join_type,
                        },
                        estimated_cost: plan.estimated_cost,
                        estimated_rows: plan.estimated_rows,
                        output_columns: plan.output_columns,
                        required_capabilities: CapabilitySet::new(),
                    })
                } else {
                    Ok(PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::HashJoin {
                            left: Box::new(left),
                            right: Box::new(right),
                            join_keys,
                            join_type,
                        },
                        estimated_cost: plan.estimated_cost,
                        estimated_rows: plan.estimated_rows,
                        output_columns: plan.output_columns,
                        required_capabilities: CapabilitySet::new(),
                    })
                }
            }
            PlanNodeType::NestedLoopJoin {
                outer,
                inner,
                correlation,
            } => {
                let outer = self.reorder_joins(*outer)?;
                let inner = self.reorder_joins(*inner)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::NestedLoopJoin {
                        outer: Box::new(outer),
                        inner: Box::new(inner),
                        correlation,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Filter { input, predicate } => {
                let input = self.reorder_joins(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Filter {
                        input: Box::new(input),
                        predicate,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Project { input, columns } => {
                let input = self.reorder_joins(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Project {
                        input: Box::new(input),
                        columns,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Distinct { input } => {
                let input = self.reorder_joins(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Distinct {
                        input: Box::new(input),
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Sort { input, order_by } => {
                let input = self.reorder_joins(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Sort {
                        input: Box::new(input),
                        order_by,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Limit {
                input,
                limit,
                offset,
            } => {
                let input = self.reorder_joins(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Limit {
                        input: Box::new(input),
                        limit,
                        offset,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let input = self.reorder_joins(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Aggregate {
                        input: Box::new(input),
                        group_by,
                        aggregates,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Union { inputs, all } => {
                let inputs: Result<Vec<_>> = inputs
                    .into_iter()
                    .map(|input| self.reorder_joins(input))
                    .collect();
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Union {
                        inputs: inputs?,
                        all,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            _ => Ok(plan),
        }
    }

    /// Extract leaf nodes from a join tree
    fn extract_join_leaves(&self, plan: &PlanNode) -> Vec<PlanNode> {
        match &plan.node_type {
            PlanNodeType::HashJoin { left, right, .. } => {
                let mut leaves = self.extract_join_leaves(left);
                leaves.extend(self.extract_join_leaves(right));
                leaves
            }
            _ => vec![plan.clone()],
        }
    }

    /// Apply greedy join ordering to a set of relations
    fn greedy_join_order(
        &self,
        relations: &mut Vec<PlanNode>,
        _join_keys: &[(String, String)],
        join_type: &JoinType,
    ) -> Result<PlanNode> {
        if relations.is_empty() {
            return Err(anyhow!("No relations to join"));
        }
        if relations.len() == 1 {
            return Ok(relations.remove(0));
        }

        // Sort by estimated cost (ascending)
        relations.sort_by(|a, b| {
            a.estimated_cost
                .partial_cmp(&b.estimated_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedily build join tree from cheapest to most expensive
        let mut result = relations.remove(0);

        while !relations.is_empty() {
            let next = relations.remove(0);
            let join_cost = self.estimate_join_cost(&result, &next);
            let output_columns: Vec<String> = result
                .output_columns
                .iter()
                .chain(next.output_columns.iter())
                .cloned()
                .collect();

            result = PlanNode {
                id: self.next_id(),
                node_type: PlanNodeType::HashJoin {
                    left: Box::new(result),
                    right: Box::new(next),
                    join_keys: vec![("id".to_string(), "id".to_string())], // Default join key
                    join_type: join_type.clone(),
                },
                estimated_cost: join_cost,
                estimated_rows: 100, // Simplified estimate
                output_columns,
                required_capabilities: CapabilitySet::new(),
            };
        }

        Ok(result)
    }

    /// Estimate cost of joining two plans
    fn estimate_join_cost(&self, left: &PlanNode, right: &PlanNode) -> f64 {
        // Hash join cost model: O(left) + O(right) build + O(left * right) probe
        // Simplified to: left_cost + right_cost + (left_rows * 0.01)
        left.estimated_cost + right.estimated_cost + (left.estimated_rows as f64 * 0.01)
    }

    // ========================================================================
    // PROJECTION PUSHDOWN
    // ========================================================================

    /// Push projections down to read only needed columns.
    /// This reduces I/O and memory usage.
    fn push_projections(&self, plan: PlanNode) -> Result<PlanNode> {
        // Collect required columns from the root
        let required_columns = self.collect_required_columns(&plan);
        self.push_projections_with_required(plan, &required_columns)
    }

    /// Push projections with a set of required columns
    fn push_projections_with_required(
        &self,
        plan: PlanNode,
        required: &[String],
    ) -> Result<PlanNode> {
        match plan.node_type {
            PlanNodeType::Project { input, columns } => {
                // Intersect required columns with projected columns
                let new_required: Vec<String> = if required.contains(&"*".to_string()) {
                    columns.clone()
                } else {
                    columns
                        .iter()
                        .zip(plan.output_columns.iter())
                        .filter(|(column, output)| {
                            required.contains(*column)
                                || required.contains(*output)
                                || **column == "*"
                        })
                        .map(|(column, _)| column.clone())
                        .collect()
                };

                let input = self.push_projections_with_required(*input, &new_required)?;
                let preserves_names = columns == plan.output_columns;

                // If all columns are required and the projection is not renaming columns,
                // remove it entirely.
                if (new_required.len() == columns.len() || columns.contains(&"*".to_string()))
                    && preserves_names
                {
                    Ok(input)
                } else {
                    let filtered_output_columns: Vec<String> =
                        if required.contains(&"*".to_string()) {
                            plan.output_columns.clone()
                        } else {
                            columns
                                .iter()
                                .zip(plan.output_columns.iter())
                                .filter(|(column, output)| {
                                    new_required.contains(column)
                                        || required.contains(*output)
                                        || **column == "*"
                                })
                                .map(|(_, output)| output.clone())
                                .collect()
                        };

                    Ok(PlanNode {
                        id: self.next_id(),
                        node_type: PlanNodeType::Project {
                            input: Box::new(input),
                            columns: new_required,
                        },
                        estimated_cost: plan.estimated_cost * 0.9, // Slight cost reduction
                        estimated_rows: plan.estimated_rows,
                        output_columns: filtered_output_columns,
                        required_capabilities: CapabilitySet::new(),
                    })
                }
            }
            PlanNodeType::Scan {
                target,
                model_type,
                predicates,
            } => {
                // Apply column pruning to scan
                let pruned_columns = Self::filter_output_columns(&plan.output_columns, required);

                // Estimate cost reduction from reading fewer columns
                let column_ratio = if !plan.output_columns.is_empty() {
                    pruned_columns.len() as f64 / plan.output_columns.len() as f64
                } else {
                    1.0
                };

                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Scan {
                        target,
                        model_type,
                        predicates,
                    },
                    estimated_cost: plan.estimated_cost * column_ratio.max(0.5),
                    estimated_rows: plan.estimated_rows,
                    output_columns: pruned_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::HashJoin {
                left,
                right,
                join_keys,
                join_type,
            } => {
                // Collect columns needed from each side
                let mut left_required: Vec<String> = if required.contains(&"*".to_string()) {
                    left.output_columns.clone()
                } else {
                    required
                        .iter()
                        .filter_map(|column| self.join_output_to_left_column(column, &left))
                        .collect()
                };
                left_required.extend(join_keys.iter().map(|(left_key, _)| left_key.clone()));
                left_required.sort();
                left_required.dedup();

                let mut right_required: Vec<String> = if required.contains(&"*".to_string()) {
                    right.output_columns.clone()
                } else {
                    required
                        .iter()
                        .filter_map(|column| {
                            self.join_output_to_right_column(column, &left, &right)
                        })
                        .collect()
                };
                right_required.extend(join_keys.iter().map(|(_, right_key)| right_key.clone()));
                right_required.sort();
                right_required.dedup();

                let left = self.push_projections_with_required(*left, &left_required)?;
                let right = self.push_projections_with_required(*right, &right_required)?;

                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::HashJoin {
                        left: Box::new(left),
                        right: Box::new(right),
                        join_keys,
                        join_type,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::NestedLoopJoin {
                outer,
                inner,
                correlation,
            } => {
                let outer = self.push_projections_with_required(*outer, required)?;
                let inner = self.push_projections_with_required(*inner, required)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::NestedLoopJoin {
                        outer: Box::new(outer),
                        inner: Box::new(inner),
                        correlation,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Filter { input, predicate } => {
                // Include columns from predicate in required set
                let mut extended_required = required.to_vec();
                if !extended_required.contains(&predicate.column) {
                    extended_required.push(predicate.column.clone());
                }

                let input = self.push_projections_with_required(*input, &extended_required)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Filter {
                        input: Box::new(input),
                        predicate,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Distinct { input } => {
                let input = self.push_projections_with_required(*input, required)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Distinct {
                        input: Box::new(input),
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Sort { input, order_by } => {
                // Include sort columns in required set
                let mut extended_required = required.to_vec();
                for clause in &order_by {
                    if !extended_required.contains(&clause.column) {
                        extended_required.push(clause.column.clone());
                    }
                }

                let input = self.push_projections_with_required(*input, &extended_required)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Sort {
                        input: Box::new(input),
                        order_by,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Limit {
                input,
                limit,
                offset,
            } => {
                let input = self.push_projections_with_required(*input, required)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Limit {
                        input: Box::new(input),
                        limit,
                        offset,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                // Collect all columns needed for grouping and aggregation
                let mut agg_required: Vec<String> = group_by.clone();
                for agg in &aggregates {
                    if let Some(col) = &agg.column
                        && !agg_required.contains(col)
                    {
                        agg_required.push(col.clone());
                    }
                }

                let input = self.push_projections_with_required(*input, &agg_required)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Aggregate {
                        input: Box::new(input),
                        group_by,
                        aggregates,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Union { inputs, all } => {
                let inputs: Result<Vec<_>> = inputs
                    .into_iter()
                    .map(|input| self.push_projections_with_required(input, required))
                    .collect();
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Union {
                        inputs: inputs?,
                        all,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: Self::filter_output_columns(&plan.output_columns, required),
                    required_capabilities: CapabilitySet::new(),
                })
            }
            // For other node types, pass through required columns
            _ => Ok(PlanNode {
                output_columns: Self::filter_output_columns(&plan.output_columns, required),
                required_capabilities: CapabilitySet::new(),
                ..plan
            }),
        }
    }

    /// Collect all columns required by a plan
    fn collect_required_columns(&self, plan: &PlanNode) -> Vec<String> {
        let mut columns = Vec::new();

        match &plan.node_type {
            PlanNodeType::Project {
                columns: proj_cols,
                input,
            } => {
                columns.extend(proj_cols.clone());
                columns.extend(self.collect_required_columns(input));
            }
            PlanNodeType::Distinct { .. } => {
                // DISTINCT operates on the visible row shape at this point in the plan.
                // Pulling through child columns here can incorrectly widen the distinct key
                // and strip away a preceding projection during pushdown.
                columns.extend(plan.output_columns.clone());
            }
            PlanNodeType::Filter { predicate, input } => {
                columns.push(predicate.column.clone());
                columns.extend(self.collect_required_columns(input));
            }
            PlanNodeType::HashJoin { .. } => columns.extend(plan.output_columns.clone()),
            PlanNodeType::Sort { order_by, input } => {
                for clause in order_by {
                    columns.push(clause.column.clone());
                }
                columns.extend(self.collect_required_columns(input));
            }
            PlanNodeType::Aggregate { .. } => columns.extend(plan.output_columns.clone()),
            _ => {
                columns.extend(plan.output_columns.clone());
            }
        }

        // Deduplicate
        columns.sort();
        columns.dedup();
        columns
    }

    /// Check if column comes from left side of join
    fn column_from_left(&self, column: &str, left: &PlanNode) -> bool {
        // Check if column has a table qualifier (e.g., "users.name")
        if let Some((qualifier, _col)) = column.split_once('.') {
            // Qualified column: match only if scan target matches the qualifier
            if let PlanNodeType::Scan { target, .. } = &left.node_type {
                return target == qualifier;
            }
            // Non-scan nodes with wildcard can accept qualified columns
            return left.output_columns.contains(&column.to_string());
        }
        left.output_columns.contains(&column.to_string())
            || left.output_columns.contains(&"*".to_string())
    }

    /// Check if column comes from right side of join
    fn column_from_right(&self, column: &str, right: &PlanNode) -> bool {
        // Check if column has a table qualifier (e.g., "orders.id")
        if let Some((qualifier, _col)) = column.split_once('.') {
            // Qualified column: match only if scan target matches the qualifier
            if let PlanNodeType::Scan { target, .. } = &right.node_type {
                return target == qualifier;
            }
            // Non-scan nodes with wildcard can accept qualified columns
            return right.output_columns.contains(&column.to_string());
        }
        right.output_columns.contains(&column.to_string())
            || right.output_columns.contains(&"*".to_string())
    }

    fn filter_output_columns(output_columns: &[String], required: &[String]) -> Vec<String> {
        if required.contains(&"*".to_string()) {
            return output_columns.to_vec();
        }

        let filtered = output_columns
            .iter()
            .filter(|column| required.contains(*column))
            .cloned()
            .collect::<Vec<_>>();

        if filtered.is_empty() && output_columns.is_empty() {
            required.to_vec()
        } else if filtered.is_empty() {
            output_columns.to_vec()
        } else {
            filtered
        }
    }

    fn normalize_column_name(column: &str) -> &str {
        column.rsplit('.').next().unwrap_or(column).trim()
    }

    fn resolve_output_column_name(&self, column: &str, plan: &PlanNode) -> Option<String> {
        let normalized = Self::normalize_column_name(column);
        plan.output_columns
            .iter()
            .find(|candidate| {
                candidate.eq_ignore_ascii_case(column) || candidate.eq_ignore_ascii_case(normalized)
            })
            .cloned()
    }

    fn join_output_to_left_column(&self, column: &str, left: &PlanNode) -> Option<String> {
        self.resolve_output_column_name(column, left)
    }

    fn join_output_to_right_column(
        &self,
        column: &str,
        left: &PlanNode,
        right: &PlanNode,
    ) -> Option<String> {
        if let Some(stripped) = column.strip_prefix("right_") {
            return self.resolve_output_column_name(stripped, right);
        }

        let right_column = self.resolve_output_column_name(column, right)?;
        if self.resolve_output_column_name(column, left).is_some() {
            None
        } else {
            Some(right_column)
        }
    }

    // ========================================================================
    // PARALLEL EXECUTION IDENTIFICATION
    // ========================================================================

    /// Identify opportunities for parallel execution.
    /// Marks independent subqueries that can run concurrently.
    fn identify_parallelism(&self, plan: PlanNode) -> Result<PlanNode> {
        // Collect parallel stages
        let _parallel_stages = self.find_parallel_stages(&plan);

        // For now, we just mark the plan with parallelism info
        // The actual parallel execution would be done by the executor

        // Recursively process children
        match plan.node_type {
            PlanNodeType::HashJoin {
                left,
                right,
                join_keys,
                join_type,
            } => {
                // Both sides of a hash join can be executed in parallel
                let left = self.identify_parallelism(*left)?;
                let right = self.identify_parallelism(*right)?;

                // Mark this as a parallelizable point
                let parallel_cost = left.estimated_cost.max(right.estimated_cost);

                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::HashJoin {
                        left: Box::new(left),
                        right: Box::new(right),
                        join_keys,
                        join_type,
                    },
                    estimated_cost: parallel_cost + plan.estimated_cost * 0.5, // Parallel execution bonus
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Union { inputs, all } => {
                // All union inputs can run in parallel
                let inputs: Result<Vec<_>> = inputs
                    .into_iter()
                    .map(|input| self.identify_parallelism(input))
                    .collect();
                let inputs = inputs?;

                // Parallel cost is max of all inputs
                let max_cost = inputs
                    .iter()
                    .map(|i| i.estimated_cost)
                    .fold(0.0f64, |a, b| a.max(b));

                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Union { inputs, all },
                    estimated_cost: max_cost + plan.estimated_cost * 0.1, // Union overhead
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::NestedLoopJoin {
                outer,
                inner,
                correlation,
            } => {
                let outer = self.identify_parallelism(*outer)?;
                let inner = self.identify_parallelism(*inner)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::NestedLoopJoin {
                        outer: Box::new(outer),
                        inner: Box::new(inner),
                        correlation,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Filter { input, predicate } => {
                let input = self.identify_parallelism(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Filter {
                        input: Box::new(input),
                        predicate,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Project { input, columns } => {
                let input = self.identify_parallelism(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Project {
                        input: Box::new(input),
                        columns,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Distinct { input } => {
                let input = self.identify_parallelism(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Distinct {
                        input: Box::new(input),
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Sort { input, order_by } => {
                let input = self.identify_parallelism(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Sort {
                        input: Box::new(input),
                        order_by,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Limit {
                input,
                limit,
                offset,
            } => {
                let input = self.identify_parallelism(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Limit {
                        input: Box::new(input),
                        limit,
                        offset,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            PlanNodeType::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let input = self.identify_parallelism(*input)?;
                Ok(PlanNode {
                    id: self.next_id(),
                    node_type: PlanNodeType::Aggregate {
                        input: Box::new(input),
                        group_by,
                        aggregates,
                    },
                    estimated_cost: plan.estimated_cost,
                    estimated_rows: plan.estimated_rows,
                    output_columns: plan.output_columns,
                    required_capabilities: CapabilitySet::new(),
                })
            }
            _ => Ok(plan),
        }
    }

    /// Find stages that can execute in parallel
    fn find_parallel_stages(&self, plan: &PlanNode) -> Vec<Vec<usize>> {
        let mut stages: Vec<Vec<usize>> = Vec::new();

        match &plan.node_type {
            PlanNodeType::HashJoin { left, right, .. } => {
                // Left and right can execute in parallel
                let mut stage = vec![left.id, right.id];
                stage.sort();
                stages.push(stage);

                // Recursively find more parallel stages
                stages.extend(self.find_parallel_stages(left));
                stages.extend(self.find_parallel_stages(right));
            }
            PlanNodeType::Union { inputs, .. } => {
                // All union inputs can execute in parallel
                let mut stage: Vec<usize> = inputs.iter().map(|i| i.id).collect();
                stage.sort();
                if stage.len() > 1 {
                    stages.push(stage);
                }

                for input in inputs {
                    stages.extend(self.find_parallel_stages(input));
                }
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                stages.extend(self.find_parallel_stages(outer));
                stages.extend(self.find_parallel_stages(inner));
            }
            PlanNodeType::Filter { input, .. }
            | PlanNodeType::Project { input, .. }
            | PlanNodeType::Distinct { input, .. }
            | PlanNodeType::Sort { input, .. }
            | PlanNodeType::Limit { input, .. }
            | PlanNodeType::Aggregate { input, .. } => {
                stages.extend(self.find_parallel_stages(input));
            }
            _ => {}
        }

        stages
    }

    /// Build physical plan from logical plan
    fn build_physical_plan(&self, plan: PlanNode) -> Result<FederatedQueryPlan> {
        let total_cost = self.calculate_total_cost(&plan);
        let involved_models = self.collect_involved_models(&plan);

        Ok(FederatedQueryPlan {
            root: plan,
            total_cost,
            metadata: PlanMetadata {
                involved_models: involved_models.clone(),
                is_cross_model: involved_models.len() > 1,
                parallel_stages: vec![],
                hints_applied: vec![],
            },
        })
    }

    /// Calculate total cost of a plan
    fn calculate_total_cost(&self, plan: &PlanNode) -> f64 {
        let mut cost = plan.estimated_cost;

        match &plan.node_type {
            PlanNodeType::HashJoin { left, right, .. } => {
                cost += self.calculate_total_cost(left);
                cost += self.calculate_total_cost(right);
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                cost += self.calculate_total_cost(outer);
                cost += self.calculate_total_cost(inner) * outer.estimated_rows as f64;
            }
            PlanNodeType::IndexJoin { left, right, .. } => {
                cost += self.calculate_total_cost(left);
                cost += self.calculate_total_cost(right);
            }
            PlanNodeType::Filter { input, .. } => {
                cost += self.calculate_total_cost(input);
            }
            PlanNodeType::Project { input, .. } => {
                cost += self.calculate_total_cost(input);
            }
            PlanNodeType::Distinct { input, .. } => {
                cost += self.calculate_total_cost(input);
            }
            PlanNodeType::Sort { input, .. } => {
                cost += self.calculate_total_cost(input);
            }
            PlanNodeType::Limit { input, .. } => {
                cost += self.calculate_total_cost(input);
            }
            PlanNodeType::Aggregate { input, .. } => {
                cost += self.calculate_total_cost(input);
            }
            PlanNodeType::Union { inputs, .. } => {
                for input in inputs {
                    cost += self.calculate_total_cost(input);
                }
            }
            _ => {}
        }

        cost
    }

    /// Collect all model types involved in the plan
    fn collect_involved_models(&self, plan: &PlanNode) -> Vec<ModelType> {
        let mut models = Vec::new();

        match &plan.node_type {
            PlanNodeType::Scan { model_type, .. } => {
                if !models.contains(model_type) {
                    models.push(*model_type);
                }
            }
            PlanNodeType::VectorSearch { .. } | PlanNodeType::RerankSearch { .. } => {
                if !models.contains(&ModelType::Vector) {
                    models.push(ModelType::Vector);
                }
            }
            PlanNodeType::GraphTraversal { .. } => {
                if !models.contains(&ModelType::Graph) {
                    models.push(ModelType::Graph);
                }
            }
            PlanNodeType::DocumentQuery { .. } => {
                if !models.contains(&ModelType::Document) {
                    models.push(ModelType::Document);
                }
            }
            PlanNodeType::ObservabilityQuery { .. } => {
                if !models.contains(&ModelType::Observability) {
                    models.push(ModelType::Observability);
                }
            }
            PlanNodeType::HashJoin { left, right, .. } => {
                models.extend(self.collect_involved_models(left));
                models.extend(self.collect_involved_models(right));
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                models.extend(self.collect_involved_models(outer));
                models.extend(self.collect_involved_models(inner));
            }
            PlanNodeType::IndexJoin { left, right, .. } => {
                models.extend(self.collect_involved_models(left));
                models.extend(self.collect_involved_models(right));
            }
            PlanNodeType::Filter { input, .. }
            | PlanNodeType::Project { input, .. }
            | PlanNodeType::Distinct { input, .. }
            | PlanNodeType::Sort { input, .. }
            | PlanNodeType::Limit { input, .. }
            | PlanNodeType::Aggregate { input, .. } => {
                models.extend(self.collect_involved_models(input));
            }
            PlanNodeType::Union { inputs, .. } => {
                for input in inputs {
                    models.extend(self.collect_involved_models(input));
                }
            }
        }

        // Deduplicate
        models.sort_by_key(|m| format!("{:?}", m));
        models.dedup();
        models
    }

    /// Get next node ID
    fn next_id(&self) -> usize {
        self.next_node_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
}

impl Default for CrossModelOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// JOIN STRATEGY SELECTION (Cost-Based Optimizer)
// ============================================================================

/// Join execution strategy selected by the cost-based optimizer
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinStrategy {
    /// Build a hash table on the smaller side, probe with the larger
    HashJoin,
    /// Iterate outer rows, scan inner for each (small datasets)
    NestedLoopJoin,
    /// Use an existing index on the inner side for lookups
    IndexJoin,
}

/// Select the optimal join strategy based on estimated cardinalities
///
/// Decision logic:
/// - **IndexJoin**: when an index exists and the right side is small (<1000 rows),
///   exploiting the index avoids a full scan.
/// - **HashJoin**: when both sides exceed 1000 rows, hash-based joining
///   gives O(n + m) cost vs O(n * m) for nested loop.
/// - **NestedLoopJoin**: fallback for small inputs where hash-table
///   construction overhead is not worthwhile.
pub fn select_join_strategy(left_rows: u64, right_rows: u64, has_index: bool) -> JoinStrategy {
    if has_index && right_rows < 1000 {
        JoinStrategy::IndexJoin
    } else if left_rows > 1000 && right_rows > 1000 {
        JoinStrategy::HashJoin
    } else {
        JoinStrategy::NestedLoopJoin
    }
}

#[cfg(test)]
mod tests;
