//! # Federated Query Executor
//!
//! Executes federated query plans across multiple data models.
//!
//! ## Features
//!
//! - **Arrow-based vectorized execution**: Efficient batch processing
//! - **Cross-model join operators**: Hash join, nested loop, index join
//! - **Streaming results**: Memory-efficient result streaming
//! - **Parallel execution**: Execute independent branches concurrently
//!
//! See also [`source_executors`] for per-model trait-based abstractions.

pub mod source_executors;

use anyhow::{Result, anyhow};
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, ListArray, RecordBatch,
    StringArray, UInt32Builder,
    builder::{Float32Builder, ListBuilder},
    new_null_array,
};
use arrow::compute::{concat, take};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_buffer::NullBuffer;
use proximadb_graph_subset::{discover_default_graph_id, legacy_graph_row_to_node};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::optimizer::{
    FederatedQueryPlan, JoinType, ObservabilityQueryType, PlanNode, PlanNodeType, PredicateOp,
    PredicateValue, VectorSource,
};
use crate::core::search::SearchParams;
use crate::graph::{Node, PropertyValue, property_value};
use crate::proto::proximadb_v1::{
    Collection, DocFilterCondition, DocFilterOperator, DocumentFilter, SqlObject, SqlValue,
    sql_value,
};
use crate::query::graph_lowering::lower_supported_graph_query_expr;
use crate::query::graph_runtime::execute_graph_query_expr_with_start_nodes;
use crate::storage::multimodel::{ModelType, MultiModelStorageFacade};
use crate::storage::traits::{
    DocumentRecord, DocumentStorageOperations, MetricAggregationParams,
    ObservabilityStorageOperations, StorageQueryContext,
};
pub use proximadb_query_fusion::execution_result::{
    ExecutionResult, FederatedExecutionResult, FederatedExecutionStats,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKey {
    Utf8(String),
    Int64(i64),
    Int32(i32),
    UInt64(u64),
    UInt32(u32),
    Boolean(bool),
}

enum DirectVectorResolution {
    Missing,
    SkipRow,
    Resolved(Vec<f32>),
}

#[derive(Debug, Clone)]
enum AggregateValue {
    Int64(Option<i64>),
    Int32(Option<i32>),
    UInt64(Option<u64>),
    UInt32(Option<u32>),
    Float64(Option<f64>),
    Float32(Option<f32>),
    Utf8(Option<String>),
    Boolean(Option<bool>),
}

impl AggregateValue {
    fn data_type(&self) -> DataType {
        match self {
            Self::Int64(_) => DataType::Int64,
            Self::Int32(_) => DataType::Int32,
            Self::UInt64(_) => DataType::UInt64,
            Self::UInt32(_) => DataType::UInt32,
            Self::Float64(_) => DataType::Float64,
            Self::Float32(_) => DataType::Float32,
            Self::Utf8(_) => DataType::Utf8,
            Self::Boolean(_) => DataType::Boolean,
        }
    }

    fn is_nullable(&self) -> bool {
        match self {
            Self::Int64(value) => value.is_none(),
            Self::Int32(value) => value.is_none(),
            Self::UInt64(value) => value.is_none(),
            Self::UInt32(value) => value.is_none(),
            Self::Float64(value) => value.is_none(),
            Self::Float32(value) => value.is_none(),
            Self::Utf8(value) => value.is_none(),
            Self::Boolean(value) => value.is_none(),
        }
    }

    fn into_array(self) -> ArrayRef {
        match self {
            Self::Int64(value) => Arc::new(Int64Array::from(vec![value])),
            Self::Int32(value) => Arc::new(arrow::array::Int32Array::from(vec![value])),
            Self::UInt64(value) => Arc::new(arrow::array::UInt64Array::from(vec![value])),
            Self::UInt32(value) => Arc::new(arrow::array::UInt32Array::from(vec![value])),
            Self::Float64(value) => Arc::new(arrow::array::Float64Array::from(vec![value])),
            Self::Float32(value) => Arc::new(Float32Array::from(vec![value])),
            Self::Utf8(value) => Arc::new(StringArray::from(vec![value.as_deref()])),
            Self::Boolean(value) => Arc::new(arrow::array::BooleanArray::from(vec![value])),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NativeVectorLayout {
    FixedSize(usize),
    Variable,
}

#[derive(Clone, Copy)]
enum ProjectedGraphColumnType {
    Boolean,
    Int64,
    Float64,
    Utf8,
}

impl ProjectedGraphColumnType {
    fn arrow_type(self) -> DataType {
        match self {
            Self::Boolean => DataType::Boolean,
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::Utf8 => DataType::Utf8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeVectorColumnSpec {
    output_name: String,
    source_path: String,
    path: Vec<String>,
    layout: NativeVectorLayout,
}

const VECTOR_SOURCE_PATH_METADATA_KEY: &str = "proximadb.federated.vector_source_path";
const VECTOR_SOURCE_ALIAS_METADATA_KEY: &str = "proximadb.federated.vector_source_alias";

/// Federated query executor
pub struct FederatedExecutor {
    /// Multi-model storage facade
    storage: Arc<MultiModelStorageFacade>,
    /// Collection metadata resolver for storage assignments and engine details
    /// (Task #76 migration: was `Option<Arc<CollectionService>>`).
    collection_port: Option<Arc<dyn proximadb_runtime::CollectionPort>>,
    /// Reuse the existing vector service so SQL VECTOR_SEARCH follows the
    /// same engine resolution, WAL visibility, and scoring path as direct search.
    vector_operations_service:
        Option<Arc<crate::services::operations::vectors::VectorOperationsService>>,
    /// Rank-pipeline singleton (R-7c.4c.2). When wired, RERANK(...) SRF
    /// dispatches second-phase rescoring via the registered profile +
    /// scorer; when `None`, RERANK degrades to first-phase retrieval
    /// only (the R-7c.4c.1 stub path).
    rank_services: Option<Arc<crate::network::rest::canonical::rank::RankServices>>,
    /// Execution configuration
    config: ExecutionConfig,
}

/// Execution configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum batch size
    pub batch_size: usize,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Maximum parallel tasks
    pub max_parallel_tasks: usize,
    /// Spill threshold in bytes
    pub spill_threshold_bytes: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            parallel_execution: true,
            max_parallel_tasks: 4,
            spill_threshold_bytes: 100 * 1024 * 1024, // 100MB
        }
    }
}

impl FederatedExecutor {
    /// Create a new federated executor
    pub fn new(storage: Arc<MultiModelStorageFacade>) -> Self {
        Self {
            storage,
            collection_port: None,
            vector_operations_service: None,
            rank_services: None,
            config: ExecutionConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(storage: Arc<MultiModelStorageFacade>, config: ExecutionConfig) -> Self {
        Self {
            storage,
            collection_port: None,
            vector_operations_service: None,
            rank_services: None,
            config,
        }
    }

    /// Reuse live collection metadata instead of synthesizing collection configs.
    pub fn with_collection_port(
        mut self,
        collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
    ) -> Self {
        self.collection_port = Some(collection_port);
        self
    }

    /// Reuse the live vector operations service instead of bypassing it and
    /// talking directly to the raw storage engine from federated SQL.
    pub fn with_vector_operations(
        mut self,
        vector_operations_service: Arc<
            crate::services::operations::vectors::VectorOperationsService,
        >,
    ) -> Self {
        self.vector_operations_service = Some(vector_operations_service);
        self
    }

    /// Wire the rank-pipeline singleton (R-7c.4c.2). When set, RERANK(...)
    /// SRF dispatches second-phase rescoring via the registered profile +
    /// scorer. Without it, RERANK degrades to first-phase retrieval only
    /// (the R-7c.4c.1 stub path).
    pub fn with_rank_services(
        mut self,
        rank_services: Arc<crate::network::rest::canonical::rank::RankServices>,
    ) -> Self {
        self.rank_services = Some(rank_services);
        self
    }

    /// Execute a query plan
    pub async fn execute(&self, plan: FederatedQueryPlan) -> Result<FederatedExecutionResult> {
        let start = std::time::Instant::now();

        let result = self.execute_node(&plan.root, true).await?;

        let mut stats = result.stats.clone();
        stats.execution_time_us = start.elapsed().as_micros() as u64;
        stats.models_queried = plan.metadata.involved_models;

        Ok(FederatedExecutionResult {
            batches: result.batches,
            schema: result.schema,
            stats,
        })
    }

    /// Execute a single plan node
    fn execute_node<'a>(
        &'a self,
        node: &'a PlanNode,
        is_root: bool,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<FederatedExecutionResult>> + Send + 'a>,
    > {
        Box::pin(async move {
            let result = match &node.node_type {
                PlanNodeType::Scan {
                    target,
                    model_type,
                    predicates,
                } => self.execute_scan(target, model_type, predicates).await,
                PlanNodeType::VectorSearch {
                    collection,
                    top_k,
                    query_vector_source,
                } => {
                    self.execute_vector_search(collection, query_vector_source, *top_k)
                        .await
                }
                PlanNodeType::RerankSearch {
                    collection,
                    query_text,
                    query_vector_source,
                    k,
                    rank_profile,
                } => {
                    self.execute_rerank_search(
                        collection,
                        query_text,
                        query_vector_source,
                        *k,
                        rank_profile.as_deref(),
                    )
                    .await
                }
                PlanNodeType::GraphTraversal {
                    cypher,
                    start_nodes,
                    source_alias,
                } => {
                    self.execute_graph_traversal(
                        cypher,
                        start_nodes.as_ref(),
                        source_alias.as_deref(),
                    )
                    .await
                }
                PlanNodeType::DocumentQuery {
                    collection,
                    filter,
                    source_alias,
                } => {
                    self.execute_document_query(
                        collection,
                        filter.as_ref(),
                        source_alias.as_deref(),
                    )
                    .await
                }
                PlanNodeType::ObservabilityQuery {
                    namespace,
                    query_type,
                    time_range: _,
                } => {
                    self.execute_observability_query(namespace, query_type)
                        .await
                }
                PlanNodeType::HashJoin {
                    left,
                    right,
                    join_keys,
                    join_type,
                } => {
                    self.execute_hash_join(left, right, join_keys, join_type)
                        .await
                }
                PlanNodeType::NestedLoopJoin {
                    outer,
                    inner,
                    correlation,
                } => {
                    self.execute_nested_loop_join(outer, inner, correlation)
                        .await
                }
                PlanNodeType::IndexJoin {
                    left,
                    right,
                    index_lookup,
                } => self.execute_index_join(left, right, index_lookup).await,
                PlanNodeType::Filter { input, predicate } => {
                    self.execute_filter(input, predicate).await
                }
                PlanNodeType::Project { input, columns } => {
                    self.execute_project(input, columns, &node.output_columns)
                        .await
                }
                PlanNodeType::Distinct { input } => self.execute_distinct(input).await,
                PlanNodeType::Sort { input, order_by } => self.execute_sort(input, order_by).await,
                PlanNodeType::Limit {
                    input,
                    limit,
                    offset,
                } => self.execute_limit(input, *limit, *offset).await,
                PlanNodeType::Aggregate {
                    input,
                    group_by,
                    aggregates,
                } => self.execute_aggregate(input, group_by, aggregates).await,
                PlanNodeType::Union { inputs, all } => self.execute_union(inputs, *all).await,
            }?;

            self.project_result_to_output_columns(result, &node.output_columns, !is_root)
        })
    }

    /// Execute a table/collection scan
    async fn execute_scan(
        &self,
        target: &str,
        model_type: &ModelType,
        predicates: &[super::optimizer::Predicate],
    ) -> Result<FederatedExecutionResult> {
        match model_type {
            ModelType::Document => {
                // Convert optimizer predicates to a DocumentFilter
                let doc_filter = if predicates.is_empty() {
                    None
                } else {
                    let conditions: Vec<DocFilterCondition> = predicates
                        .iter()
                        .filter_map(|p| {
                            let op = match p.op {
                                PredicateOp::Eq => DocFilterOperator::Eq,
                                PredicateOp::Ne => DocFilterOperator::Ne,
                                PredicateOp::Lt => DocFilterOperator::Lt,
                                PredicateOp::Le => DocFilterOperator::Lte,
                                PredicateOp::Gt => DocFilterOperator::Gt,
                                PredicateOp::Ge => DocFilterOperator::Gte,
                                PredicateOp::Like => DocFilterOperator::Contains,
                                // IsNull / IsNotNull / In / Between not yet mapped
                                _ => return None,
                            };
                            let sql_val = match &p.value {
                                PredicateValue::String(s) => Some(SqlValue {
                                    value: Some(sql_value::Value::StringValue(s.clone())),
                                }),
                                PredicateValue::Int(i) => Some(SqlValue {
                                    value: Some(sql_value::Value::Int64Value(*i)),
                                }),
                                PredicateValue::Float(f) => Some(SqlValue {
                                    value: Some(sql_value::Value::NumberValue(*f)),
                                }),
                                PredicateValue::Bool(b) => Some(SqlValue {
                                    value: Some(sql_value::Value::BoolValue(*b)),
                                }),
                                PredicateValue::Null => None,
                                PredicateValue::List(_) => None,
                            };
                            Some(DocFilterCondition {
                                path: p.column.clone(),
                                operator: op as i32,
                                value: sql_val,
                                values: vec![],
                            })
                        })
                        .collect();

                    if conditions.is_empty() {
                        None
                    } else {
                        Some(DocumentFilter {
                            conditions,
                            or_filters: vec![],
                            and_filters: vec![],
                        })
                    }
                };

                let doc_store = self.storage.get_document_store().ok_or_else(|| {
                    anyhow!(
                        "Document store is not configured for collection '{}'",
                        target
                    )
                })?;

                let documents = doc_store
                    .query_documents(target, doc_filter, self.config.batch_size, 0)
                    .await?;

                if documents.is_empty() {
                    return Ok(FederatedExecutionResult::empty_with_schema(
                        Self::document_query_schema(&[], None),
                    ));
                }

                let batch = Self::build_document_record_batch(&documents, None)?;
                Ok(FederatedExecutionResult::from_batch(batch))
            }

            ModelType::Graph => {
                // For graph scans, fetch all nodes from the graph store
                let graph_store = self.storage.get_graph_store().ok_or_else(|| {
                    anyhow!("Graph store is not configured for target '{}'", target)
                })?;

                let nodes = graph_store.fetch_all_nodes().await?;
                if nodes.is_empty() {
                    return Ok(FederatedExecutionResult::empty_with_schema(
                        Self::graph_query_schema(&[], None),
                    ));
                }

                let batch = Self::build_graph_node_batch(&nodes, None)?;
                Ok(FederatedExecutionResult::from_batch(batch))
            }

            ModelType::Vector => Err(anyhow!(
                "Full vector collection scans are not supported for '{}'; use VECTOR_SEARCH('{}', <query_vector>, <top_k>) to search by similarity or apply metadata filters via DOCUMENT_QUERY",
                target,
                target
            )),

            _ => Err(anyhow!(
                "Scan execution is not supported for model type '{:?}' on target '{}'; use the appropriate SQL extension (VECTOR_SEARCH, GRAPH_QUERY, DOCUMENT_QUERY, LOGS, or METRICS)",
                model_type,
                target
            )),
        }
    }

    /// Execute vector similarity search
    async fn execute_vector_search(
        &self,
        collection: &str,
        query_vector_source: &VectorSource,
        top_k: usize,
    ) -> Result<FederatedExecutionResult> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("score", DataType::Float32, false),
        ]));

        let query_vector = self.resolve_query_vector(query_vector_source)?;

        if query_vector.is_empty() {
            return Err(anyhow!(
                "Vector search requires a non-empty query vector for collection '{}'",
                collection
            ));
        }

        if let Some(vector_ops) = &self.vector_operations_service {
            let collection_id = if let Some(collection_port) = &self.collection_port {
                collection_port
                    .get_collection(collection, None)
                    .await?
                    .map(|resolved| resolved.id)
                    .unwrap_or_else(|| collection.to_string())
            } else {
                collection.to_string()
            };

            let request = crate::proto::proximadb_v1::VectorSearchRequest {
                collection_id,
                queries: vec![crate::proto::proximadb_v1::SearchQuery {
                    vector: query_vector,
                    filters: std::collections::HashMap::new(),
                    advanced_filter: None,
                }],
                top_k: top_k as u32,
                // Match the v2 record-search default path. Supplying
                // include_fields with both vector and metadata disabled can
                // bypass WAL-visible rows in the legacy compatibility search.
                include_fields: None,
                search_params: None,
                distance_metric_override: None,
                search_optimization: None,
            };

            let response = vector_ops
                .search_v1(request, proximadb_runtime::PortIdentity::anonymous())
                .await?;
            let records = response
                .results
                .map(|result| result.results)
                .unwrap_or_default();

            if records.is_empty() {
                return Ok(FederatedExecutionResult::empty_with_schema(schema));
            }

            let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
            let scores: Vec<f32> = records
                .iter()
                .map(|r| r.similarity.unwrap_or(r.score as f32))
                .collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(ids)) as ArrayRef,
                    Arc::new(Float32Array::from(scores)) as ArrayRef,
                ],
            )?;

            return Ok(FederatedExecutionResult::from_batch(batch));
        }

        let vector_store = self
            .storage
            .get_vector_store()
            .ok_or_else(|| anyhow!("Vector store is not configured"))?;
        let engine = vector_store
            .primary_engine()
            .ok_or_else(|| anyhow!("Vector store has no primary engine"))?;

        use crate::proto::proximadb_v1::CollectionConfig;
        let collection_config = if let Some(collection_port) = &self.collection_port {
            match collection_port.get_collection(collection, None).await? {
                Some(mut resolved) => {
                    let mut config = resolved.config.take().unwrap_or_else(|| CollectionConfig {
                        name: collection.to_string(),
                        dimension: query_vector.len() as u32,
                        ..Default::default()
                    });

                    if config.name.is_empty() {
                        config.name = collection.to_string();
                    }
                    if config.dimension == 0 {
                        config.dimension = query_vector.len() as u32;
                    }

                    resolved.config = Some(config);
                    Arc::new(resolved)
                }
                None => Arc::new(Collection {
                    id: collection.to_string(),
                    config: Some(CollectionConfig {
                        name: collection.to_string(),
                        dimension: query_vector.len() as u32,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            }
        } else {
            Arc::new(Collection {
                id: collection.to_string(),
                config: Some(CollectionConfig {
                    name: collection.to_string(),
                    dimension: query_vector.len() as u32,
                    ..Default::default()
                }),
                ..Default::default()
            })
        };

        let search_params = Arc::new(SearchParams {
            top_k: Some(top_k as u16),
            vector: Some(query_vector),
            ..Default::default()
        });
        let query_context = StorageQueryContext::new(search_params, collection_config);
        let results = engine.search_vectors_unified(&query_context).await?;

        if results.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(schema));
        }

        let ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        let scores: Vec<f32> = results
            .iter()
            .map(|r| r.similarity.unwrap_or(r.score))
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(Float32Array::from(scores)) as ArrayRef,
            ],
        )?;

        Ok(FederatedExecutionResult::from_batch(batch))
    }

    /// Execute RERANK SRF.
    ///
    /// R-7c.4c.1 landed the first-phase-only stub; R-7c.4c.2 wires the
    /// real second-phase dispatch through [`RankServices`]. The two
    /// paths still share the same 5-column output schema
    /// (`id` / `score` / `phase` / `match_features` / `summary_features`),
    /// so downstream SELECTs are stable regardless of which path runs.
    ///
    /// Phase column values:
    /// - `"first"` — retrieval-only (no `rank_profile`)
    /// - `"first_only"` — profile requested but [`RankServices`] not
    ///   wired (degraded path)
    /// - `"second"` — full rank pipeline ran (profile + rank_services)
    async fn execute_rerank_search(
        &self,
        collection: &str,
        query_text: &str,
        query_vector_source: &VectorSource,
        k: usize,
        rank_profile: Option<&str>,
    ) -> Result<FederatedExecutionResult> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("score", DataType::Float32, false),
            Field::new("phase", DataType::Utf8, false),
            Field::new("match_features", DataType::Utf8, true),
            Field::new("summary_features", DataType::Utf8, true),
        ]));

        // R-7c.4c.2: full rank pipeline when rank_services is wired AND
        // a profile is requested. Other cases fall through to the
        // first-phase-only path below.
        if let (Some(rank_services), Some(profile_name)) = (&self.rank_services, rank_profile) {
            return self
                .execute_rerank_full_pipeline(
                    rank_services.as_ref(),
                    collection,
                    query_text,
                    query_vector_source,
                    k,
                    profile_name,
                    schema,
                )
                .await;
        }

        // Reuse the dense-retrieval path for the first phase.
        let retrieval = self
            .execute_vector_search(collection, query_vector_source, k)
            .await?;
        if retrieval.batches.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(schema));
        }

        // Without RankServices, profile + first-phase-only is the
        // degraded path; without profile it's retrieval-only.
        let phase_label = if rank_profile.is_some() {
            "first_only"
        } else {
            "first"
        };

        // Coalesce retrieval batches into a single 5-column batch.
        let id_chunks: Vec<ArrayRef> = retrieval
            .batches
            .iter()
            .map(|b| b.column(0).clone())
            .collect();
        let score_chunks: Vec<ArrayRef> = retrieval
            .batches
            .iter()
            .map(|b| b.column(1).clone())
            .collect();
        let id_refs: Vec<&dyn Array> = id_chunks.iter().map(|a| a.as_ref()).collect();
        let score_refs: Vec<&dyn Array> = score_chunks.iter().map(|a| a.as_ref()).collect();
        let id_concat = concat(&id_refs)?;
        let score_concat = concat(&score_refs)?;
        let total_rows = id_concat.len();
        let phase_array: ArrayRef =
            Arc::new(StringArray::from(vec![phase_label.to_string(); total_rows]));
        let null_strings: Vec<Option<String>> = vec![None; total_rows];
        let match_array: ArrayRef = Arc::new(StringArray::from(null_strings.clone()));
        let summary_array: ArrayRef = Arc::new(StringArray::from(null_strings));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                id_concat,
                score_concat,
                phase_array,
                match_array,
                summary_array,
            ],
        )?;

        // `query_text` is unused in the stub path; the full-pipeline
        // path above threads it into the rank pipeline's
        // QueryContext.query_text.
        let _ = query_text;

        Ok(FederatedExecutionResult::from_batch(batch))
    }

    /// Run the full rank pipeline through [`RankServices`] (R-7c.4c.2).
    ///
    /// Delegates to [`handle_rank_search`] so REST + SQL dispatch share
    /// one code path (profile lookup, blueprint materialisation,
    /// blocking phase runner, optional global cross-modal rescoring,
    /// truncation, identifier rendering). The response shape
    /// (`Vec<ScoredHitDto>`) is then projected back into the canonical
    /// 5-column rerank Arrow schema so RERANK(...) consumers see the
    /// same shape regardless of which path produced the rows.
    #[allow(clippy::too_many_arguments)]
    async fn execute_rerank_full_pipeline(
        &self,
        rank_services: &crate::network::rest::canonical::rank::RankServices,
        collection: &str,
        query_text: &str,
        query_vector_source: &VectorSource,
        k: usize,
        profile_name: &str,
        schema: Arc<Schema>,
    ) -> Result<FederatedExecutionResult> {
        let query_vector = self.resolve_query_vector(query_vector_source)?;
        let req = crate::network::rest::canonical::rank::RankSearchRequest {
            collection: collection.to_string(),
            query_vector,
            query_text: if query_text.is_empty() {
                None
            } else {
                Some(query_text.to_string())
            },
            k,
            rank_profile: Some(profile_name.to_string()),
            rank_overrides: None,
        };

        // Resolve the per-profile second-phase scorer (may be absent —
        // matches `handle_rank_search`'s contract that a missing scorer
        // skips the second phase).
        let scorer = rank_services
            .second_phase_scorers
            .get(profile_name)
            .map(|entry| entry.value().clone());

        let response = crate::network::rest::canonical::rank::handle_rank_search(
            req,
            rank_services.profile_registry.as_ref(),
            rank_services.candidate_provider.as_ref(),
            rank_services.blueprint_factory.clone(),
            scorer,
        )
        .await
        .map_err(|err| anyhow!("rerank pipeline error: {err}"))?;

        Self::rerank_hits_to_batch(&response.hits, schema)
    }

    /// Project the wire-shape rerank hits into the canonical 5-column
    /// Arrow batch. Extracted as a pure helper so the reshape is
    /// testable without standing up the full [`RankServices`] +
    /// pipeline scaffolding (R-7c.4c.2).
    ///
    /// `phase = "second"` flags rows produced by this path; the
    /// first-phase-only stub path emits `"first"` or `"first_only"`
    /// (see `execute_rerank_search`).
    fn rerank_hits_to_batch(
        hits: &[crate::network::rest::canonical::rank::ScoredHitDto],
        schema: Arc<Schema>,
    ) -> Result<FederatedExecutionResult> {
        if hits.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(schema));
        }
        let n = hits.len();
        let mut ids: Vec<String> = Vec::with_capacity(n);
        let mut scores: Vec<f32> = Vec::with_capacity(n);
        let mut match_feats: Vec<Option<String>> = Vec::with_capacity(n);
        let mut summary_feats: Vec<Option<String>> = Vec::with_capacity(n);
        for hit in hits.iter() {
            ids.push(hit.id.clone());
            scores.push(hit.score);
            match_feats.push(if hit.match_features.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&hit.match_features).unwrap_or_default())
            });
            summary_feats.push(if hit.summary_features.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&hit.summary_features).unwrap_or_default())
            });
        }
        let phase: ArrayRef = Arc::new(StringArray::from(vec!["second".to_string(); n]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(Float32Array::from(scores)) as ArrayRef,
                phase,
                Arc::new(StringArray::from(match_feats)) as ArrayRef,
                Arc::new(StringArray::from(summary_feats)) as ArrayRef,
            ],
        )?;
        Ok(FederatedExecutionResult::from_batch(batch))
    }

    /// Execute graph traversal
    async fn execute_graph_traversal(
        &self,
        cypher: &str,
        start_nodes: Option<&Vec<String>>,
        source_alias: Option<&str>,
    ) -> Result<FederatedExecutionResult> {
        let graph_store = self
            .storage
            .get_graph_store()
            .ok_or_else(|| anyhow!("Graph store is not configured"))?;

        if let Some(graph_service) = graph_store.service() {
            let default_graph = discover_default_graph_id(graph_service.as_ref()).await;
            if let Ok(graph_query) =
                lower_supported_graph_query_expr(cypher, None, default_graph.as_deref())
            {
                let executed = execute_graph_query_expr_with_start_nodes(
                    graph_service.as_ref(),
                    &graph_query,
                    start_nodes.map(Vec::as_slice),
                )
                .await?;
                if graph_query.uses_legacy_node_rows {
                    // `legacy_graph_row_to_node` (proto-typed graph subset) yields
                    // proto nodes; the batch builder speaks the neutral model.
                    let nodes = executed
                        .rows
                        .iter()
                        .map(|row| {
                            legacy_graph_row_to_node(row)
                                .map(|n| std::sync::Arc::new((*n).clone().into()))
                        })
                        .collect::<Result<Vec<std::sync::Arc<Node>>>>()?;

                    if nodes.is_empty() {
                        return Ok(FederatedExecutionResult::empty_with_schema(
                            Self::graph_query_schema(&[], source_alias),
                        ));
                    }

                    let batch = Self::build_graph_node_batch(&nodes, source_alias)?;
                    return Ok(FederatedExecutionResult::from_batch(batch));
                }

                let batch = Self::build_projected_graph_result_batch(
                    &executed.rows,
                    &graph_query.output_columns,
                )?;
                return Ok(FederatedExecutionResult::from_batch(batch));
            }
        }

        let nodes = if let Some(start_node_ids) = start_nodes {
            let mut result_nodes = Vec::new();
            for node_id in start_node_ids {
                if let Some(node) = graph_store.fetch_node(node_id).await? {
                    result_nodes.push(node);
                    if cypher.contains("-->") || cypher.contains("-[") {
                        result_nodes.extend(graph_store.fetch_neighbors(node_id).await?);
                    }
                }
            }
            result_nodes
        } else if cypher.contains(':') {
            let label = cypher
                .split(':')
                .nth(1)
                .and_then(|s| s.split([')', ' ', '{']).next())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());

            if let Some(label) = label {
                graph_store.fetch_nodes_by_label(label).await?
            } else {
                graph_store.fetch_all_nodes().await?
            }
        } else {
            graph_store.fetch_all_nodes().await?
        };

        if nodes.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(
                Self::graph_query_schema(&[], source_alias),
            ));
        }

        let batch = Self::build_graph_node_batch(&nodes, source_alias)?;

        Ok(FederatedExecutionResult::from_batch(batch))
    }
    fn build_projected_graph_result_batch(
        rows: &[JsonValue],
        output_columns: &[String],
    ) -> Result<RecordBatch> {
        let inferred_types = output_columns
            .iter()
            .map(|column| Self::infer_projected_graph_column_type(rows, column))
            .collect::<Vec<_>>();

        let schema = Arc::new(Schema::new(
            output_columns
                .iter()
                .zip(inferred_types.iter())
                .map(|(column, column_type)| Field::new(column, column_type.arrow_type(), true))
                .collect::<Vec<_>>(),
        ));

        if output_columns.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        let columns = output_columns
            .iter()
            .zip(inferred_types.iter())
            .map(|(column, column_type)| {
                Self::build_projected_graph_column(rows, column, *column_type)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, columns).map_err(Into::into)
    }

    fn infer_projected_graph_column_type(
        rows: &[JsonValue],
        column: &str,
    ) -> ProjectedGraphColumnType {
        let mut saw_float = false;
        let mut saw_int = false;
        let mut saw_bool = false;
        let mut saw_string = false;
        let mut saw_complex = false;

        for row in rows {
            let Some(value) = row
                .as_object()
                .and_then(|object| object.get(column))
                .filter(|value| !value.is_null())
            else {
                continue;
            };

            match value {
                JsonValue::Bool(_) => saw_bool = true,
                JsonValue::Number(number) => {
                    if number.is_i64() || number.is_u64() {
                        saw_int = true;
                    } else {
                        saw_float = true;
                    }
                }
                JsonValue::String(_) => saw_string = true,
                JsonValue::Array(_) | JsonValue::Object(_) => saw_complex = true,
                JsonValue::Null => {}
            }
        }

        let scalar_kinds = [saw_bool, saw_int, saw_float, saw_string]
            .into_iter()
            .filter(|seen| *seen)
            .count();
        if saw_complex || scalar_kinds > 1 {
            ProjectedGraphColumnType::Utf8
        } else if saw_bool {
            ProjectedGraphColumnType::Boolean
        } else if saw_float {
            ProjectedGraphColumnType::Float64
        } else if saw_int {
            ProjectedGraphColumnType::Int64
        } else {
            ProjectedGraphColumnType::Utf8
        }
    }

    fn build_projected_graph_column(
        rows: &[JsonValue],
        column: &str,
        column_type: ProjectedGraphColumnType,
    ) -> Result<ArrayRef> {
        match column_type {
            ProjectedGraphColumnType::Boolean => Ok(Arc::new(arrow::array::BooleanArray::from(
                rows.iter()
                    .map(|row| {
                        row.as_object()
                            .and_then(|object| object.get(column))
                            .and_then(JsonValue::as_bool)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef),
            ProjectedGraphColumnType::Int64 => Ok(Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| {
                        row.as_object()
                            .and_then(|object| object.get(column))
                            .and_then(JsonValue::as_i64)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef),
            ProjectedGraphColumnType::Float64 => Ok(Arc::new(arrow::array::Float64Array::from(
                rows.iter()
                    .map(|row| {
                        row.as_object()
                            .and_then(|object| object.get(column))
                            .and_then(JsonValue::as_f64)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef),
            ProjectedGraphColumnType::Utf8 => Ok(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| {
                        row.as_object()
                            .and_then(|object| object.get(column))
                            .and_then(Self::projected_graph_utf8_value)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef),
        }
    }

    fn projected_graph_utf8_value(value: &JsonValue) -> Option<String> {
        match value {
            JsonValue::Null => None,
            JsonValue::String(value) => Some(value.clone()),
            other => serde_json::to_string(other).ok(),
        }
    }

    /// Execute document query
    async fn execute_document_query(
        &self,
        collection: &str,
        filter: Option<&String>,
        source_alias: Option<&str>,
    ) -> Result<FederatedExecutionResult> {
        let doc_store = self
            .storage
            .get_document_store()
            .ok_or_else(|| anyhow!("Document store is not configured"))?;

        let doc_filter = match filter {
            Some(filter) => Self::parse_document_query_filter(filter)?,
            None => None,
        };

        let documents = doc_store
            .query_documents(collection, doc_filter, self.config.batch_size, 0)
            .await?;

        if documents.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(
                Self::document_query_schema(&[], source_alias),
            ));
        }

        let batch = Self::build_document_record_batch(&documents, source_alias)?;

        Ok(FederatedExecutionResult::from_batch(batch))
    }

    fn parse_document_query_filter(filter: &str) -> Result<Option<DocumentFilter>> {
        let trimmed = filter.trim();
        if trimmed.is_empty() || trimmed == "1=1" {
            return Ok(None);
        }
        if Self::split_document_filter_keyword(trimmed, "OR").len() > 1 {
            return Err(anyhow!(
                "Unsupported DOCUMENT_QUERY filter clause '{}'; OR filters are not yet supported",
                trimmed
            ));
        }

        let mut conditions = Vec::new();
        for clause in Self::split_document_filter_and_clauses(trimmed) {
            let Some(condition) = Self::parse_document_filter_condition(clause)? else {
                return Err(anyhow!(
                    "Unsupported DOCUMENT_QUERY filter clause '{}'; supported clauses are simple AND comparisons with =, !=, <, <=, >, >=, or CONTAINS",
                    clause
                ));
            };
            conditions.push(condition);
        }

        if conditions.is_empty() {
            return Ok(None);
        }

        Ok(Some(DocumentFilter {
            conditions,
            or_filters: vec![],
            and_filters: vec![],
        }))
    }

    fn split_document_filter_and_clauses(filter: &str) -> Vec<&str> {
        Self::split_document_filter_keyword(filter, "AND")
            .into_iter()
            .map(str::trim)
            .filter(|clause| !clause.is_empty())
            .collect()
    }

    fn split_document_filter_keyword<'a>(filter: &'a str, keyword: &str) -> Vec<&'a str> {
        let mut clauses = Vec::new();
        let mut start = 0usize;
        let mut in_single_quote = false;
        let mut in_double_quote = false;
        let keyword_len = keyword.len();

        for (idx, ch) in filter.char_indices() {
            match ch {
                '\'' if !in_double_quote => in_single_quote = !in_single_quote,
                '"' if !in_single_quote => in_double_quote = !in_double_quote,
                _ => {}
            }

            if in_single_quote || in_double_quote {
                continue;
            }

            let Some(candidate) = filter.get(idx..idx + keyword_len) else {
                continue;
            };
            if !candidate.eq_ignore_ascii_case(keyword) {
                continue;
            }

            let before = filter[..idx].chars().next_back();
            let after = filter[idx + keyword_len..].chars().next();
            if before.is_some_and(|c| !c.is_ascii_whitespace())
                || after.is_some_and(|c| !c.is_ascii_whitespace())
            {
                continue;
            }

            clauses.push(&filter[start..idx]);
            start = idx + keyword_len;
        }

        clauses.push(&filter[start..]);
        clauses
    }

    fn parse_document_filter_condition(clause: &str) -> Result<Option<DocFilterCondition>> {
        for (operator_text, operator) in [
            (" CONTAINS ", DocFilterOperator::Contains),
            (">=", DocFilterOperator::Gte),
            ("<=", DocFilterOperator::Lte),
            ("!=", DocFilterOperator::Ne),
            ("<>", DocFilterOperator::Ne),
            (">", DocFilterOperator::Gt),
            ("<", DocFilterOperator::Lt),
            ("=", DocFilterOperator::Eq),
        ] {
            let Some((path, value)) = Self::split_document_filter_operator(clause, operator_text)
            else {
                continue;
            };
            let path = path.trim();
            let value = value.trim();
            if path.is_empty() || value.is_empty() {
                return Ok(None);
            }

            return Ok(Some(DocFilterCondition {
                path: path.strip_prefix("$.").unwrap_or(path).to_string(),
                operator: operator as i32,
                value: Some(Self::document_filter_sql_value(value)?),
                values: vec![],
            }));
        }

        Ok(None)
    }

    fn split_document_filter_operator<'a>(
        clause: &'a str,
        operator_text: &str,
    ) -> Option<(&'a str, &'a str)> {
        if operator_text
            .chars()
            .all(|ch| ch.is_ascii_alphabetic() || ch.is_ascii_whitespace())
        {
            let needle = operator_text.trim();
            for (idx, _) in clause.char_indices() {
                let candidate = clause.get(idx..idx + needle.len())?;
                if !candidate.eq_ignore_ascii_case(needle) {
                    continue;
                }
                let before = clause[..idx].chars().next_back();
                let after = clause[idx + needle.len()..].chars().next();
                if before.is_some_and(|c| !c.is_ascii_whitespace())
                    || after.is_some_and(|c| !c.is_ascii_whitespace())
                {
                    continue;
                }
                return Some((&clause[..idx], &clause[idx + needle.len()..]));
            }
            return None;
        }

        clause.split_once(operator_text)
    }

    fn document_filter_sql_value(raw: &str) -> Result<SqlValue> {
        let trimmed = raw.trim();
        let unquoted = trimmed
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .or_else(|| {
                trimmed
                    .strip_prefix('\'')
                    .and_then(|value| value.strip_suffix('\''))
            });

        if let Some(value) = unquoted {
            // Decode SQL-standard doubled quotes (round 33 added this to
            // the sibling parse_predicate_value — the filter value with an
            // apostrophe silently matched 0 rows here).
            let decoded = if trimmed.starts_with('\'') {
                value.replace("''", "'")
            } else {
                value.replace("\"\"", "\"")
            };
            return Ok(SqlValue {
                value: Some(sql_value::Value::StringValue(decoded)),
            });
        }

        if trimmed.eq_ignore_ascii_case("true") || trimmed.eq_ignore_ascii_case("false") {
            return Ok(SqlValue {
                value: Some(sql_value::Value::BoolValue(
                    trimmed.eq_ignore_ascii_case("true"),
                )),
            });
        }

        if let Ok(value) = trimmed.parse::<i64>() {
            return Ok(SqlValue {
                value: Some(sql_value::Value::Int64Value(value)),
            });
        }

        if let Ok(value) = trimmed.parse::<f64>() {
            return Ok(SqlValue {
                value: Some(sql_value::Value::NumberValue(value)),
            });
        }

        Ok(SqlValue {
            value: Some(sql_value::Value::StringValue(trimmed.to_string())),
        })
    }

    fn document_query_schema(
        vector_columns: &[NativeVectorColumnSpec],
        source_alias: Option<&str>,
    ) -> Arc<Schema> {
        let mut fields = vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document", DataType::Utf8, true),
        ];
        fields.extend(
            vector_columns
                .iter()
                .map(|spec| Self::native_vector_field(spec, source_alias)),
        );
        Arc::new(Schema::new(fields))
    }

    fn graph_query_schema(
        vector_columns: &[NativeVectorColumnSpec],
        source_alias: Option<&str>,
    ) -> Arc<Schema> {
        let mut fields = vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, true),
            Field::new("properties", DataType::Utf8, true),
        ];
        fields.extend(
            vector_columns
                .iter()
                .map(|spec| Self::native_vector_field(spec, source_alias)),
        );
        Arc::new(Schema::new(fields))
    }

    fn native_vector_field(spec: &NativeVectorColumnSpec, source_alias: Option<&str>) -> Field {
        let data_type = match spec.layout {
            NativeVectorLayout::FixedSize(dimension) => DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dimension as i32,
            ),
            NativeVectorLayout::Variable => {
                DataType::List(Arc::new(Field::new("item", DataType::Float32, false)))
            }
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            VECTOR_SOURCE_PATH_METADATA_KEY.to_string(),
            spec.source_path.clone(),
        );
        if let Some(source_alias) = source_alias {
            metadata.insert(
                VECTOR_SOURCE_ALIAS_METADATA_KEY.to_string(),
                source_alias.to_string(),
            );
        }
        Field::new(&spec.output_name, data_type, true).with_metadata(metadata)
    }

    fn build_document_record_batch(
        documents: &[DocumentRecord],
        source_alias: Option<&str>,
    ) -> Result<RecordBatch> {
        let vector_columns = Self::collect_document_vector_columns(documents);
        let schema = Self::document_query_schema(&vector_columns, source_alias);
        let ids: Vec<String> = documents
            .iter()
            .map(|document| document.id.clone())
            .collect();
        let docs: Vec<String> = documents
            .iter()
            .map(|document| {
                serde_json::to_string(&document.document).unwrap_or_else(|_| "{}".to_string())
            })
            .collect();

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(docs)) as ArrayRef,
        ];
        for spec in &vector_columns {
            columns.push(Self::build_native_vector_array(
                documents,
                spec,
                |document, path| Self::extract_sql_object_vector(&document.document, path),
            )?);
        }

        RecordBatch::try_new(schema, columns).map_err(Into::into)
    }

    fn build_graph_node_batch(
        nodes: &[Arc<Node>],
        source_alias: Option<&str>,
    ) -> Result<RecordBatch> {
        let vector_columns = Self::collect_graph_vector_columns(nodes);
        let schema = Self::graph_query_schema(&vector_columns, source_alias);
        let node_ids: Vec<String> = nodes.iter().map(|node| node.id.clone()).collect();
        let labels: Vec<Option<String>> = nodes
            .iter()
            .map(|node| node.labels.first().cloned())
            .collect();
        let properties: Vec<String> = nodes
            .iter()
            .map(|node| {
                serde_json::to_string(&node.properties).unwrap_or_else(|_| "{}".to_string())
            })
            .collect();

        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(node_ids)) as ArrayRef,
            Arc::new(StringArray::from(labels)) as ArrayRef,
            Arc::new(StringArray::from(properties)) as ArrayRef,
        ];
        for spec in &vector_columns {
            columns.push(Self::build_native_vector_array(
                nodes,
                spec,
                |node, path| Self::extract_property_vector(&node.properties, path),
            )?);
        }

        RecordBatch::try_new(schema, columns).map_err(Into::into)
    }

    fn collect_document_vector_columns(
        documents: &[DocumentRecord],
    ) -> Vec<NativeVectorColumnSpec> {
        let mut columns = BTreeMap::new();
        for document in documents {
            Self::collect_sql_object_vector_columns_into(&document.document, &[], &mut columns);
        }
        columns.into_values().collect()
    }

    fn collect_graph_vector_columns(nodes: &[Arc<Node>]) -> Vec<NativeVectorColumnSpec> {
        let mut columns = BTreeMap::new();
        for node in nodes {
            for (name, value) in &node.properties {
                let path = vec![name.clone()];
                Self::collect_property_vector_columns_into(value, &path, &mut columns);
            }
        }
        columns.into_values().collect()
    }

    fn collect_sql_object_vector_columns_into(
        object: &SqlObject,
        prefix: &[String],
        columns: &mut BTreeMap<String, NativeVectorColumnSpec>,
    ) {
        for (name, value) in &object.fields {
            let mut path = prefix.to_vec();
            path.push(name.clone());
            Self::collect_sql_value_vector_columns_into(value, &path, columns);
        }
    }

    fn collect_sql_value_vector_columns_into(
        value: &SqlValue,
        path: &[String],
        columns: &mut BTreeMap<String, NativeVectorColumnSpec>,
    ) {
        if let Some(vector) = Self::sql_value_to_vector(value) {
            Self::register_vector_column(
                columns,
                Self::document_vector_output_name(path),
                Self::document_vector_source_path(path),
                path.to_vec(),
                vector.len(),
            );
            return;
        }

        if let Some(sql_value::Value::ObjectValue(object)) = value.value.as_ref() {
            Self::collect_sql_object_vector_columns_into(object, path, columns);
        }
    }

    fn collect_property_vector_columns_into(
        value: &PropertyValue,
        path: &[String],
        columns: &mut BTreeMap<String, NativeVectorColumnSpec>,
    ) {
        if let Some(vector) = Self::property_value_to_vector(value) {
            Self::register_vector_column(
                columns,
                Self::graph_vector_output_name(path),
                Self::graph_vector_source_path(path),
                path.to_vec(),
                vector.len(),
            );
            return;
        }

        if let Some(property_value::Value::ObjectValue(object)) = value.value.as_ref() {
            for (name, nested) in &object.fields {
                let mut nested_path = path.to_vec();
                nested_path.push(name.clone());
                Self::collect_property_vector_columns_into(nested, &nested_path, columns);
            }
        }
    }

    fn register_vector_column(
        columns: &mut BTreeMap<String, NativeVectorColumnSpec>,
        output_name: String,
        source_path: String,
        path: Vec<String>,
        dimension: usize,
    ) {
        if dimension == 0 {
            return;
        }

        columns
            .entry(output_name.clone())
            .and_modify(|existing| {
                if existing.path != path {
                    return;
                }
                match existing.layout {
                    NativeVectorLayout::FixedSize(existing_dimension)
                        if existing_dimension != dimension =>
                    {
                        existing.layout = NativeVectorLayout::Variable;
                    }
                    NativeVectorLayout::FixedSize(_) | NativeVectorLayout::Variable => {}
                }
            })
            .or_insert(NativeVectorColumnSpec {
                output_name,
                source_path,
                path,
                layout: NativeVectorLayout::FixedSize(dimension),
            });
    }

    fn document_vector_output_name(path: &[String]) -> String {
        if path.len() == 1 {
            path[0].clone()
        } else {
            format!("document.{}", path.join("."))
        }
    }

    fn graph_vector_output_name(path: &[String]) -> String {
        if path.len() == 1 {
            path[0].clone()
        } else {
            format!("properties.{}", path.join("."))
        }
    }

    fn document_vector_source_path(path: &[String]) -> String {
        format!("document.{}", path.join("."))
    }

    fn graph_vector_source_path(path: &[String]) -> String {
        format!("properties.{}", path.join("."))
    }

    fn build_native_vector_array<T, F>(
        items: &[T],
        spec: &NativeVectorColumnSpec,
        mut extractor: F,
    ) -> Result<ArrayRef>
    where
        F: FnMut(&T, &[String]) -> Option<Vec<f32>>,
    {
        match spec.layout {
            NativeVectorLayout::FixedSize(dimension) => {
                let mut values = Vec::with_capacity(items.len() * dimension);
                let mut validity = Vec::with_capacity(items.len());

                for item in items {
                    if let Some(vector) = extractor(item, &spec.path)
                        && vector.len() == dimension
                    {
                        values.extend(vector);
                        validity.push(true);
                    } else {
                        values.resize(values.len() + dimension, 0.0);
                        validity.push(false);
                    }
                }

                let values_array = Arc::new(Float32Array::from(values)) as ArrayRef;
                let nulls = (!validity.iter().all(|value| *value))
                    .then(|| NullBuffer::from(validity.clone()));
                let list = FixedSizeListArray::try_new(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dimension as i32,
                    values_array,
                    nulls,
                )
                .map_err(|error| {
                    anyhow!(
                        "Failed to build native vector column '{}': {}",
                        spec.output_name,
                        error
                    )
                })?;
                Ok(Arc::new(list) as ArrayRef)
            }
            NativeVectorLayout::Variable => {
                let item_field = Arc::new(Field::new("item", DataType::Float32, false));
                let mut builder = ListBuilder::new(Float32Builder::new()).with_field(item_field);
                for item in items {
                    if let Some(vector) = extractor(item, &spec.path) {
                        for value in vector {
                            builder.values().append_value(value);
                        }
                        builder.append(true);
                    } else {
                        builder.append(false);
                    }
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }
        }
    }

    fn extract_sql_object_vector(object: &SqlObject, path: &[String]) -> Option<Vec<f32>> {
        let (first, rest) = path.split_first()?;
        let value = object.fields.get(first)?;
        Self::extract_sql_value_vector(value, rest)
    }

    fn extract_sql_value_vector(value: &SqlValue, path: &[String]) -> Option<Vec<f32>> {
        if path.is_empty() {
            return Self::sql_value_to_vector(value);
        }

        match value.value.as_ref()? {
            sql_value::Value::ObjectValue(object) => Self::extract_sql_object_vector(object, path),
            // TD-PROTO-2 round 6: JSONB documents traverse the same paths —
            // the wildcard silently dropped/zeroed federated vector rows
            // whose path crossed a JSONB field.
            sql_value::Value::JsonbValue(bytes) => {
                // Round 11: delegate to the shared walker (was a verbatim
                // third copy of the object.get walk).
                let doc = proximadb_data_model::ProximaValue::jsonb_to_json_lossy(bytes);
                let current = proximadb_search_types::sql_value_filter::json_get_path(
                    &doc,
                    path.iter().map(String::as_str),
                )?;
                Self::json_to_vector(current)
            }
            _ => None,
        }
    }

    /// Extract a vector from a JSON array of numbers (inside a JSONB doc).
    fn json_to_vector(value: &serde_json::Value) -> Option<Vec<f32>> {
        match value {
            serde_json::Value::Array(items) => items
                .iter()
                .map(|item| match item {
                    // The ONE non-finite policy: the f32 narrowing can
                    // overflow (1e300 → inf) — reject, don't dispatch.
                    serde_json::Value::Number(n) => {
                        n.as_f64().and_then(crate::core::utils::finite_f32)
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .filter(|vector| !vector.is_empty()),
            _ => None,
        }
    }

    fn extract_property_vector(
        properties: &HashMap<String, PropertyValue>,
        path: &[String],
    ) -> Option<Vec<f32>> {
        let (first, rest) = path.split_first()?;
        let value = properties.get(first)?;
        Self::extract_property_value_vector(value, rest)
    }

    fn extract_property_value_vector(value: &PropertyValue, path: &[String]) -> Option<Vec<f32>> {
        if path.is_empty() {
            return Self::property_value_to_vector(value);
        }

        match value.value.as_ref()? {
            property_value::Value::ObjectValue(object) => {
                Self::extract_property_vector(&object.fields, path)
            }
            _ => None,
        }
    }

    fn sql_value_to_vector(value: &SqlValue) -> Option<Vec<f32>> {
        match value.value.as_ref()? {
            sql_value::Value::ArrayValue(array) => array
                .values
                .iter()
                .map(|value| match value.value.as_ref()? {
                    sql_value::Value::NumberValue(number) => {
                        crate::core::utils::finite_f32(*number)
                    }
                    sql_value::Value::Int64Value(number) => {
                        crate::core::utils::finite_f32(*number as f64)
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .filter(|vector| !vector.is_empty()),
            // A JSONB document carrying a bare JSON array of numbers.
            sql_value::Value::JsonbValue(bytes) => Self::json_to_vector(
                &proximadb_data_model::ProximaValue::jsonb_to_json_lossy(bytes),
            ),
            _ => None,
        }
    }

    fn property_value_to_vector(value: &PropertyValue) -> Option<Vec<f32>> {
        match value.value.as_ref()? {
            property_value::Value::VectorValue(vector) => {
                (!vector.is_empty()).then(|| vector.clone())
            }
            property_value::Value::ArrayValue(array) => array
                .values
                .iter()
                .map(|value| match value.value.as_ref()? {
                    property_value::Value::DoubleValue(number) => {
                        crate::core::utils::finite_f32(*number)
                    }
                    property_value::Value::IntValue(number) => {
                        crate::core::utils::finite_f32(*number as f64)
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .filter(|vector| !vector.is_empty()),
            _ => None,
        }
    }

    /// Execute observability query
    async fn execute_observability_query(
        &self,
        namespace: &str,
        query_type: &ObservabilityQueryType,
    ) -> Result<FederatedExecutionResult> {
        let schema = match query_type {
            ObservabilityQueryType::Logs => Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("level", DataType::Utf8, false),
                Field::new("message", DataType::Utf8, false),
            ])),
            ObservabilityQueryType::Metrics => Arc::new(Schema::new(vec![
                Field::new("timestamp", DataType::Int64, false),
                Field::new("metric_name", DataType::Utf8, false),
                Field::new("value", DataType::Float32, false),
            ])),
            ObservabilityQueryType::Traces => Arc::new(Schema::new(vec![
                Field::new("trace_id", DataType::Utf8, false),
                Field::new("span_id", DataType::Utf8, false),
                Field::new("operation", DataType::Utf8, false),
                Field::new("duration_ns", DataType::Int64, false),
            ])),
        };

        let obs_store = self
            .storage
            .get_observability_store()
            .ok_or_else(|| anyhow!("Observability store is not configured"))?;
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let hour_ago_ns = now_ns - (3600 * 1_000_000_000);

        match query_type {
            ObservabilityQueryType::Logs => {
                let result = obs_store
                    .query_logs(namespace, hour_ago_ns, now_ns, None, 1000)
                    .await?;
                if result.logs.is_empty() {
                    return Ok(FederatedExecutionResult::empty_with_schema(schema));
                }

                use crate::proto::proximadb_v1::Severity;
                let timestamps: Vec<i64> = result.logs.iter().map(|l| l.timestamp_ns).collect();
                let levels: Vec<String> = result
                    .logs
                    .iter()
                    .map(|l| {
                        match Severity::try_from(l.severity).unwrap_or(Severity::Info) {
                            Severity::Debug => "DEBUG",
                            Severity::Info => "INFO",
                            Severity::Warn => "WARN",
                            Severity::Error => "ERROR",
                            Severity::Fatal => "FATAL",
                            _ => "UNKNOWN",
                        }
                        .to_string()
                    })
                    .collect();
                let messages: Vec<String> = result.logs.iter().map(|l| l.message.clone()).collect();

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(timestamps)) as ArrayRef,
                        Arc::new(StringArray::from(levels)) as ArrayRef,
                        Arc::new(StringArray::from(messages)) as ArrayRef,
                    ],
                )?;

                Ok(FederatedExecutionResult::from_batch(batch))
            }
            ObservabilityQueryType::Metrics => {
                use crate::proto::proximadb_v1::MetricAggregation;
                let params = MetricAggregationParams {
                    metric_name: "*".to_string(),
                    start_time_ns: hour_ago_ns,
                    end_time_ns: now_ns,
                    aggregation: MetricAggregation::Avg,
                    step_seconds: 60,
                    label_filters: HashMap::new(),
                    group_by: vec![],
                };
                let result = obs_store.aggregate_metrics(namespace, params).await?;
                let mut timestamps = Vec::new();
                let mut names = Vec::new();
                let mut values = Vec::new();

                for series in &result.series {
                    let series_name = series
                        .labels
                        .get("__name__")
                        .cloned()
                        .unwrap_or_else(|| "metric".to_string());
                    for point in &series.points {
                        timestamps.push(point.timestamp_ns);
                        names.push(series_name.clone());
                        values.push(point.value as f32);
                    }
                }

                if timestamps.is_empty() {
                    return Ok(FederatedExecutionResult::empty_with_schema(schema));
                }

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(timestamps)) as ArrayRef,
                        Arc::new(StringArray::from(names)) as ArrayRef,
                        Arc::new(Float32Array::from(values)) as ArrayRef,
                    ],
                )?;

                Ok(FederatedExecutionResult::from_batch(batch))
            }
            ObservabilityQueryType::Traces => {
                let traces = obs_store
                    .query_traces(namespace, hour_ago_ns, now_ns, None, None, 100)
                    .await?;

                if traces.is_empty() {
                    return Ok(FederatedExecutionResult::empty_with_schema(schema));
                }

                let trace_ids: Vec<String> = traces.iter().map(|t| t.trace_id.clone()).collect();
                let span_ids: Vec<String> = traces.iter().map(|t| t.span_id.clone()).collect();
                let operations: Vec<String> = traces.iter().map(|t| t.name.clone()).collect();
                let durations: Vec<i64> = traces
                    .iter()
                    .map(|t| t.end_time_ns - t.start_time_ns)
                    .collect();

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(trace_ids)) as ArrayRef,
                        Arc::new(StringArray::from(span_ids)) as ArrayRef,
                        Arc::new(StringArray::from(operations)) as ArrayRef,
                        Arc::new(Int64Array::from(durations)) as ArrayRef,
                    ],
                )?;

                Ok(FederatedExecutionResult::from_batch(batch))
            }
        }
    }

    fn resolve_query_vector(&self, source: &VectorSource) -> Result<Vec<f32>> {
        match source {
            VectorSource::Literal(vector) => Ok(vector.clone()),
            VectorSource::Expression(expr) => Self::parse_vector_literal(expr).ok_or_else(|| {
                anyhow!(
                    "Unsupported vector expression '{}' in federated executor; provide a literal vector for now",
                    expr
                )
            }),
            VectorSource::ColumnRef { table, column } => Err(anyhow!(
                "Correlated vector source '{}.{}' is not yet executable in the federated executor",
                table,
                column
            )),
            VectorSource::Subquery(_) => Err(anyhow!(
                "Subquery-derived vector sources are not yet executable in the federated executor"
            )),
        }
    }

    fn resolve_query_vector_for_row(
        &self,
        source: &VectorSource,
        outer_batch: &RecordBatch,
        outer_row: usize,
    ) -> Result<Option<Vec<f32>>> {
        match source {
            VectorSource::Literal(vector) => Ok(Some(vector.clone())),
            VectorSource::Expression(expr) => {
                Self::parse_vector_literal(expr).map(Some).ok_or_else(|| {
                    anyhow!(
                        "Unsupported vector expression '{}' in federated executor; provide a literal vector for now",
                        expr
                    )
                })
            }
            VectorSource::ColumnRef { table, column } => self
                .resolve_vector_from_outer_column_optional(outer_batch, outer_row, table, column),
            VectorSource::Subquery(_) => Err(anyhow!(
                "Subquery-derived vector sources are not yet executable in the federated executor"
            )),
        }
    }

    #[cfg(test)]
    fn resolve_vector_from_outer_column(
        &self,
        outer_batch: &RecordBatch,
        outer_row: usize,
        table: &str,
        column_path: &str,
    ) -> Result<Vec<f32>> {
        self.resolve_vector_from_outer_column_optional(outer_batch, outer_row, table, column_path)?
            .ok_or_else(|| {
                anyhow!(
                    "Correlated vector source '{}.{}' was null or missing for outer row {}",
                    table,
                    column_path,
                    outer_row
                )
            })
    }

    fn resolve_vector_from_outer_column_optional(
        &self,
        outer_batch: &RecordBatch,
        outer_row: usize,
        table: &str,
        column_path: &str,
    ) -> Result<Option<Vec<f32>>> {
        let mut path_segments = column_path
            .split('.')
            .map(str::trim)
            .filter(|segment| !segment.is_empty());
        let base_column = path_segments.next().ok_or_else(|| {
            anyhow!(
                "Correlated vector source '{}.{}' did not include a column name",
                table,
                column_path
            )
        })?;
        let nested_path = path_segments.collect::<Vec<_>>();

        // ── Arrow-native fast path (TD-032) ─────────────────────────────────
        // Document and graph executors materialize vector-bearing fields as
        // native Arrow list columns, including nested document paths like
        // `document.profile.embedding` and graph property paths like
        // `properties.embedding`. Probe the exact correlated path first, then
        // use the leaf-name fallback only for one-level aliases such as
        // `document.embedding` -> `embedding`.
        if !nested_path.is_empty() {
            let exact_candidates = [
                column_path.to_string(),
                format!("{}.{}", table, column_path),
            ];
            if let Some(resolution) = Self::resolve_metadata_vector_candidate(
                outer_batch,
                outer_row,
                table,
                column_path,
                &exact_candidates,
            )? {
                match resolution {
                    DirectVectorResolution::Resolved(vector) => return Ok(Some(vector)),
                    DirectVectorResolution::SkipRow => return Ok(None),
                    DirectVectorResolution::Missing => {}
                }
            }
            match Self::resolve_direct_vector_candidate(outer_batch, outer_row, &exact_candidates)?
            {
                DirectVectorResolution::Resolved(vector) => return Ok(Some(vector)),
                DirectVectorResolution::SkipRow => return Ok(None),
                DirectVectorResolution::Missing => {}
            }

            if nested_path.len() == 1 {
                let leaf_column = nested_path[0];
                let leaf_candidates = [
                    leaf_column.to_string(),
                    format!("{}.{}", table, leaf_column),
                ];
                match Self::resolve_direct_vector_candidate(
                    outer_batch,
                    outer_row,
                    &leaf_candidates,
                )? {
                    DirectVectorResolution::Resolved(vector) => return Ok(Some(vector)),
                    DirectVectorResolution::SkipRow => return Ok(None),
                    DirectVectorResolution::Missing => {}
                }
            }
        }

        let requested = format!("{}.{}", table, base_column);
        let column_index = Self::resolve_column_index(outer_batch.schema().as_ref(), &requested)
            .or_else(|| Self::resolve_column_index(outer_batch.schema().as_ref(), base_column))
            .ok_or_else(|| {
                anyhow!(
                    "Correlated column '{}' was not found in outer schema {:?}",
                    requested,
                    outer_batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| field.name().clone())
                        .collect::<Vec<_>>()
                )
            })?;

        let array = outer_batch.column(column_index);
        if array.is_null(outer_row) {
            return Ok(None);
        }

        // Fast path: try direct Arrow extraction (no JSON parsing needed)
        if nested_path.is_empty()
            && let Some(vector) = Self::try_extract_vector_from_arrow(array.as_ref(), outer_row)
        {
            return Ok(Some(vector));
        }

        match array.data_type() {
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Utf8 correlated column"))?;
                if nested_path.is_empty() {
                    if let Some(vector) = Self::parse_vector_literal(values.value(outer_row)) {
                        Ok(Some(vector))
                    } else if values.value(outer_row).trim().eq_ignore_ascii_case("null") {
                        Ok(None)
                    } else {
                        Err(anyhow!(
                            "Correlated vector source '{}' did not contain a vector literal",
                            requested
                        ))
                    }
                } else {
                    Self::parse_nested_vector_from_serialized_value(
                        values.value(outer_row),
                        &requested,
                        &nested_path,
                    )
                }
            }
            DataType::FixedSizeList(_, _) | DataType::List(_) => {
                // LOUD error, symmetric with the Utf8 arm below: the same
                // poison value must not behave differently by storage
                // layout, and a silent Ok(None) here returned 0 rows for
                // typo-class nested paths (try_extract is gated on
                // nested_path.is_empty()). A poison-row SKIP policy is
                // TD-tracked — it needs a distinguishing signal between
                // stored-data poison and user typos.
                Err(anyhow!(
                    "Correlated vector source '{}' has Arrow list type but could not extract finite, non-empty Float32 values",
                    requested
                ))
            }
            other => Err(anyhow!(
                "Correlated vector source '{}' uses unsupported outer column type {:?}",
                requested,
                other
            )),
        }
    }

    /// Try to extract a vector directly from Arrow array types without JSON parsing.
    /// Returns Some(Vec<f32>) if the column contains native Arrow list/fixed-size-list data,
    /// or None if it needs to fall through to the JSON parsing path.
    fn try_extract_vector_from_arrow(array: &dyn Array, row: usize) -> Option<Vec<f32>> {
        // The ONE non-finite policy: the string-decode arm of the resolver
        // rejects inf/NaN text — the binary fast path must not re-admit
        // them (NaN similarities order arbitrarily downstream).
        // The ONE non-finite AND empty policies (a 0-dimension or inf
        // vector must not dispatch to the kernels).
        let finite = |values: Vec<f32>| {
            (!values.is_empty() && values.iter().all(|v| v.is_finite())).then_some(values)
        };
        // Try FixedSizeList<Float32> first (most common for embeddings)
        if let Some(fsl) = array.as_any().downcast_ref::<FixedSizeListArray>()
            && !fsl.is_null(row)
        {
            let values = fsl.value(row);
            if let Some(float_array) = values.as_any().downcast_ref::<Float32Array>() {
                return finite(float_array.values().to_vec());
            }
            // Try Float64 list and convert to f32
            if let Some(f64_array) = values.as_any().downcast_ref::<arrow::array::Float64Array>() {
                return finite(f64_array.values().iter().map(|&v| v as f32).collect());
            }
        }

        // Try List<Float32>
        if let Some(list) = array.as_any().downcast_ref::<ListArray>()
            && !list.is_null(row)
        {
            let values = list.value(row);
            if let Some(float_array) = values.as_any().downcast_ref::<Float32Array>() {
                return finite(float_array.values().to_vec());
            }
            // Try Float64 list and convert to f32
            if let Some(f64_array) = values.as_any().downcast_ref::<arrow::array::Float64Array>() {
                return finite(f64_array.values().iter().map(|&v| v as f32).collect());
            }
        }

        None
    }

    fn resolve_direct_vector_candidate(
        batch: &RecordBatch,
        row: usize,
        candidates: &[String],
    ) -> Result<DirectVectorResolution> {
        for candidate in candidates {
            let Some(idx) = Self::resolve_column_index(batch.schema().as_ref(), candidate) else {
                continue;
            };
            return Self::resolve_direct_vector_index(batch, row, idx, candidate);
        }

        Ok(DirectVectorResolution::Missing)
    }

    fn resolve_metadata_vector_candidate(
        batch: &RecordBatch,
        row: usize,
        table: &str,
        column_path: &str,
        candidates: &[String],
    ) -> Result<Option<DirectVectorResolution>> {
        let matching_indices = batch
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(idx, field)| {
                field
                    .metadata()
                    .get(VECTOR_SOURCE_PATH_METADATA_KEY)
                    .filter(|source_path| source_path.as_str() == column_path)
                    .map(|_| idx)
            })
            .collect::<Vec<_>>();

        if matching_indices.is_empty() {
            return Ok(None);
        }

        let alias_matches = matching_indices
            .iter()
            .copied()
            .filter(|idx| {
                let schema = batch.schema();
                schema
                    .field(*idx)
                    .metadata()
                    .get(VECTOR_SOURCE_ALIAS_METADATA_KEY)
                    .map(|source_alias| Self::source_alias_matches(source_alias, table))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        let has_alias_metadata = matching_indices.iter().any(|idx| {
            batch
                .schema()
                .field(*idx)
                .metadata()
                .contains_key(VECTOR_SOURCE_ALIAS_METADATA_KEY)
        });
        if has_alias_metadata && alias_matches.is_empty() {
            return Err(anyhow!(
                "Correlated vector source '{}.{}' did not match any outer source alias",
                table,
                column_path
            ));
        }

        let candidate_pool = if alias_matches.is_empty() {
            matching_indices
        } else {
            alias_matches
        };

        let named_matches = candidate_pool
            .iter()
            .copied()
            .filter(|idx| {
                let schema = batch.schema();
                let field_name = schema.field(*idx).name().clone();
                candidates
                    .iter()
                    .any(|candidate| candidate.as_str() == field_name.as_str())
            })
            .collect::<Vec<_>>();

        let selected_index = match named_matches.as_slice() {
            [idx] => Some(*idx),
            [] if candidate_pool.len() == 1 => Some(candidate_pool[0]),
            _ => None,
        };

        selected_index
            .map(|idx| {
                let source = batch.schema().field(idx).name().clone();
                Self::resolve_direct_vector_index(batch, row, idx, &source)
            })
            .transpose()
    }

    fn source_alias_matches(source_alias: &str, table: &str) -> bool {
        let source_alias = source_alias.trim();
        let table = table.trim();

        match (
            Self::quoted_identifier_contents(source_alias),
            Self::quoted_identifier_contents(table),
        ) {
            (Some(left), Some(right)) => left == right,
            (None, None) => source_alias.eq_ignore_ascii_case(table),
            _ => false,
        }
    }

    fn quoted_identifier_contents(identifier: &str) -> Option<&str> {
        if identifier.len() >= 2 && identifier.starts_with('"') && identifier.ends_with('"') {
            Some(&identifier[1..identifier.len() - 1])
        } else {
            None
        }
    }

    fn resolve_direct_vector_index(
        batch: &RecordBatch,
        row: usize,
        index: usize,
        source: &str,
    ) -> Result<DirectVectorResolution> {
        let array = batch.column(index);
        if array.is_null(row) {
            return Ok(DirectVectorResolution::SkipRow);
        }
        if let Some(vector) = Self::try_extract_vector_from_arrow(array.as_ref(), row) {
            return Ok(DirectVectorResolution::Resolved(vector));
        }

        match array.data_type() {
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Utf8 correlated column"))?;
                let raw = values.value(row);
                if raw.trim().eq_ignore_ascii_case("null") {
                    return Ok(DirectVectorResolution::SkipRow);
                }
                Self::parse_vector_from_serialized_value(raw, source, &[])
                    .map(DirectVectorResolution::Resolved)
            }
            DataType::FixedSizeList(_, _) | DataType::List(_) => Err(anyhow!(
                "Correlated vector source '{}' has Arrow list type but could not extract finite, non-empty Float32 values",
                source
            )),
            other => Err(anyhow!(
                "Correlated vector source '{}' uses unsupported outer column type {:?}",
                source,
                other
            )),
        }
    }

    fn parse_vector_from_serialized_value(
        raw: &str,
        source: &str,
        nested_path: &[&str],
    ) -> Result<Vec<f32>> {
        if nested_path.is_empty() {
            return Self::parse_vector_literal(raw).ok_or_else(|| {
                anyhow!(
                    "Correlated vector source '{}' did not contain a vector literal",
                    source
                )
            });
        }

        let json: serde_json::Value = serde_json::from_str(raw).map_err(|error| {
            anyhow!(
                "Correlated vector source '{}' is not valid JSON for nested path resolution: {}",
                source,
                error
            )
        })?;
        let mut current = &json;
        for segment in nested_path {
            // Try direct path first, then fall back to looking under "fields" key
            // (SqlObject serializes with a "fields" wrapper)
            current = current
                .get(*segment)
                .or_else(|| current.get("fields").and_then(|f| f.get(*segment)))
                .ok_or_else(|| {
                    anyhow!(
                        "Nested path '{}.{}' was not found in correlated source",
                        source,
                        nested_path.join(".")
                    )
                })?;
        }

        Self::parse_vector_from_json_value(current, source, nested_path)
    }

    fn parse_nested_vector_from_serialized_value(
        raw: &str,
        source: &str,
        nested_path: &[&str],
    ) -> Result<Option<Vec<f32>>> {
        let json: serde_json::Value = serde_json::from_str(raw).map_err(|error| {
            anyhow!(
                "Correlated vector source '{}' is not valid JSON for nested path resolution: {}",
                source,
                error
            )
        })?;
        let mut current = &json;
        for segment in nested_path {
            let Some(next) = current.get(*segment).or_else(|| {
                current
                    .get("fields")
                    .and_then(|fields| fields.get(*segment))
            }) else {
                return Ok(None);
            };
            current = next;
        }

        if current.is_null() {
            return Ok(None);
        }

        Self::parse_vector_from_json_value(current, source, nested_path).map(Some)
    }

    fn parse_vector_from_json_value(
        value: &serde_json::Value,
        source: &str,
        nested_path: &[&str],
    ) -> Result<Vec<f32>> {
        match value {
            serde_json::Value::Array(values) => {
                // The ONE empty-vector policy (a 0-dimension query is a
                // deferred kernel error or garbage scores).
                if values.is_empty() {
                    return Err(anyhow!(
                        "Nested vector source '{}.{}' is empty",
                        source,
                        nested_path.join(".")
                    ));
                }
                values
                    .iter()
                    .map(|value| Self::parse_vector_component(value, source, nested_path))
                    .collect()
            }
            serde_json::Value::String(value) => {
                Self::parse_vector_literal(value).ok_or_else(|| {
                    anyhow!(
                        "Nested vector source '{}.{}' did not contain a vector literal string",
                        source,
                        nested_path.join(".")
                    )
                })
            }
            serde_json::Value::Object(object) => {
                // Handle proto SqlValue wrapper: { "value": { "arrayValue": { "values": [...] } } }
                if let Some(inner) = object.get("value") {
                    return Self::parse_vector_from_json_value(inner, source, nested_path);
                }
                for wrapper_key in ["array_value", "arrayValue", "values", "object_value"] {
                    if let Some(inner) = object.get(wrapper_key) {
                        return Self::parse_vector_from_json_value(inner, source, nested_path);
                    }
                }
                for scalar_key in ["number_value", "int64_value", "string_value"] {
                    if let Some(inner) = object.get(scalar_key) {
                        return Self::parse_vector_from_json_value(inner, source, nested_path);
                    }
                }
                Err(anyhow!(
                    "Nested vector source '{}.{}' resolved to an unsupported object {:?}",
                    source,
                    nested_path.join("."),
                    object
                ))
            }
            serde_json::Value::Number(number) => Ok(vec![number.as_f64().ok_or_else(|| {
                anyhow!(
                    "Nested vector source '{}.{}' contains a non-finite number",
                    source,
                    nested_path.join(".")
                )
            })? as f32]),
            other => Err(anyhow!(
                "Nested vector source '{}.{}' resolved to unsupported JSON value {:?}",
                source,
                nested_path.join("."),
                other
            )),
        }
    }

    fn parse_vector_component(
        value: &serde_json::Value,
        source: &str,
        nested_path: &[&str],
    ) -> Result<f32> {
        match value {
            serde_json::Value::Number(number) => {
                // The ONE non-finite policy (shared finite_f32 helper):
                // serde_json numbers are always finite as f64, but the f32
                // NARROWING can overflow to inf (1e300).
                number
                    .as_f64()
                    .and_then(crate::core::utils::finite_f32)
                    .ok_or_else(|| {
                        anyhow!(
                            "Nested vector source '{}.{}' contains a component outside the finite f32 range",
                            source,
                            nested_path.join(".")
                        )
                    })
            }
            serde_json::Value::Object(object) => {
                if let Some(inner) = object
                    .get("number_value")
                    .or_else(|| object.get("int64_value"))
                {
                    return Self::parse_vector_component(inner, source, nested_path);
                }
                Err(anyhow!(
                    "Nested vector source '{}.{}' contains a non-numeric element {:?}",
                    source,
                    nested_path.join("."),
                    object
                ))
            }
            other => Err(anyhow!(
                "Nested vector source '{}.{}' contains a non-numeric element {:?}",
                source,
                nested_path.join("."),
                other
            )),
        }
    }

    // Delegate to the ONE filtered parser (non-finite components reject)
    // — the local unfiltered copy re-admitted the values the optimizer's
    // parse rejects whenever the Expression fallback re-parsed raw SQL.
    fn parse_vector_literal(raw: &str) -> Option<Vec<f32>> {
        crate::query::federated::optimizer::vector_query_parsing::parse_vector_literal(raw)
    }

    fn merge_batches(&self, result: &FederatedExecutionResult) -> Result<RecordBatch> {
        if result.batches.is_empty() {
            return Ok(RecordBatch::new_empty(result.schema.clone()));
        }

        if result.batches.len() == 1 {
            return Ok(result.batches[0].clone());
        }

        let columns = (0..result.schema.fields().len())
            .map(|column_idx| {
                let arrays: Vec<&dyn Array> = result
                    .batches
                    .iter()
                    .map(|batch| batch.column(column_idx).as_ref())
                    .collect();
                concat(&arrays)
            })
            .collect::<arrow::error::Result<Vec<_>>>()?;

        Ok(RecordBatch::try_new(result.schema.clone(), columns)?)
    }

    fn normalize_column_name(column: &str) -> &str {
        column.rsplit('.').next().unwrap_or(column).trim()
    }

    fn resolve_column_index(schema: &Schema, requested: &str) -> Option<usize> {
        let normalized = Self::normalize_column_name(requested);

        schema
            .fields()
            .iter()
            .position(|field| field.name().eq_ignore_ascii_case(requested))
            .or_else(|| {
                schema
                    .fields()
                    .iter()
                    .position(|field| field.name().eq_ignore_ascii_case(normalized))
            })
            .or_else(|| {
                schema.fields().iter().position(|field| {
                    Self::normalize_column_name(field.name()).eq_ignore_ascii_case(normalized)
                })
            })
            .or_else(|| {
                if normalized.eq_ignore_ascii_case("id") {
                    schema
                        .fields()
                        .iter()
                        .position(|field| field.name().eq_ignore_ascii_case("node_id"))
                } else {
                    None
                }
            })
    }

    fn predicate_matches_row(
        batch: &RecordBatch,
        row: usize,
        predicate: &super::optimizer::Predicate,
    ) -> Result<bool> {
        let column_index = Self::resolve_column_index(batch.schema().as_ref(), &predicate.column)
            .ok_or_else(|| {
            anyhow!(
                "Column '{}' was not found in schema {:?}",
                predicate.column,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect::<Vec<_>>()
            )
        })?;
        let array = batch.column(column_index);

        if matches!(predicate.op, super::optimizer::PredicateOp::IsNull) {
            return Ok(array.is_null(row));
        }
        if matches!(predicate.op, super::optimizer::PredicateOp::IsNotNull) {
            return Ok(!array.is_null(row));
        }
        if array.is_null(row) {
            return Ok(false);
        }

        match array.data_type() {
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Utf8 predicate column"))?;
                let value = values.value(row);
                match (&predicate.op, &predicate.value) {
                    (
                        super::optimizer::PredicateOp::Eq,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(value == expected),
                    (
                        super::optimizer::PredicateOp::Ne,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(value != expected),
                    (
                        super::optimizer::PredicateOp::Like,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(Self::like_matches(value, expected)),
                    (
                        super::optimizer::PredicateOp::Gt,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(value > expected.as_str()),
                    (
                        super::optimizer::PredicateOp::Ge,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(value >= expected.as_str()),
                    (
                        super::optimizer::PredicateOp::Lt,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(value < expected.as_str()),
                    (
                        super::optimizer::PredicateOp::Le,
                        super::optimizer::PredicateValue::String(expected),
                    ) => Ok(value <= expected.as_str()),
                    _ => Err(anyhow!(
                        "Predicate {:?} with value {:?} is not supported for Utf8 column '{}'",
                        predicate.op,
                        predicate.value,
                        predicate.column
                    )),
                }
            }
            DataType::Int64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int64 predicate column"))?;
                let value = values.value(row);
                Self::compare_numeric_predicate(value as f64, predicate)
            }
            DataType::Int32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int32 predicate column"))?;
                let value = values.value(row);
                Self::compare_numeric_predicate(value as f64, predicate)
            }
            DataType::UInt64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt64 predicate column"))?;
                let value = values.value(row);
                Self::compare_numeric_predicate(value as f64, predicate)
            }
            DataType::UInt32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt32 predicate column"))?;
                let value = values.value(row);
                Self::compare_numeric_predicate(value as f64, predicate)
            }
            DataType::Float32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float32 predicate column"))?;
                let value = values.value(row);
                Self::compare_numeric_predicate(value as f64, predicate)
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float64 predicate column"))?;
                let value = values.value(row);
                Self::compare_numeric_predicate(value, predicate)
            }
            DataType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Boolean predicate column"))?;
                let value = values.value(row);
                match (&predicate.op, &predicate.value) {
                    (
                        super::optimizer::PredicateOp::Eq,
                        super::optimizer::PredicateValue::Bool(expected),
                    ) => Ok(value == *expected),
                    (
                        super::optimizer::PredicateOp::Ne,
                        super::optimizer::PredicateValue::Bool(expected),
                    ) => Ok(value != *expected),
                    _ => Err(anyhow!(
                        "Predicate {:?} with value {:?} is not supported for Boolean column '{}'",
                        predicate.op,
                        predicate.value,
                        predicate.column
                    )),
                }
            }
            other => Err(anyhow!(
                "Filtering on data type {:?} is not yet supported for column '{}'",
                other,
                predicate.column
            )),
        }
    }

    fn compare_numeric_predicate(
        actual: f64,
        predicate: &super::optimizer::Predicate,
    ) -> Result<bool> {
        let expected = match &predicate.value {
            super::optimizer::PredicateValue::Int(value) => *value as f64,
            super::optimizer::PredicateValue::Float(value) => *value,
            _ => {
                return Err(anyhow!(
                    "Predicate {:?} requires a numeric literal for column '{}'",
                    predicate.op,
                    predicate.column
                ));
            }
        };

        Ok(match predicate.op {
            super::optimizer::PredicateOp::Eq => actual == expected,
            super::optimizer::PredicateOp::Ne => actual != expected,
            super::optimizer::PredicateOp::Lt => actual < expected,
            super::optimizer::PredicateOp::Le => actual <= expected,
            super::optimizer::PredicateOp::Gt => actual > expected,
            super::optimizer::PredicateOp::Ge => actual >= expected,
            _ => {
                return Err(anyhow!(
                    "Numeric predicate {:?} is not supported for column '{}'",
                    predicate.op,
                    predicate.column
                ));
            }
        })
    }

    fn like_matches(value: &str, pattern: &str) -> bool {
        if pattern == "%" {
            return true;
        }
        if let Some(inner) = pattern
            .strip_prefix('%')
            .and_then(|trimmed| trimmed.strip_suffix('%'))
        {
            return value.contains(inner);
        }
        if let Some(suffix) = pattern.strip_prefix('%') {
            return value.ends_with(suffix);
        }
        if let Some(prefix) = pattern.strip_suffix('%') {
            return value.starts_with(prefix);
        }
        value == pattern
    }

    fn compare_rows(
        batch: &RecordBatch,
        left_row: usize,
        right_row: usize,
        order_by: &[super::optimizer::OrderByClause],
    ) -> Result<std::cmp::Ordering> {
        for clause in order_by {
            let column_index = Self::resolve_column_index(batch.schema().as_ref(), &clause.column)
                .ok_or_else(|| anyhow!("Sort column '{}' was not found", clause.column))?;
            let array = batch.column(column_index);
            let left_null = array.is_null(left_row);
            let right_null = array.is_null(right_row);

            let ordering = match (left_null, right_null) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => {
                    if clause.nulls_first {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                }
                (false, true) => {
                    if clause.nulls_first {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    }
                }
                (false, false) => match array.data_type() {
                    DataType::Utf8 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| anyhow!("Failed to downcast Utf8 sort column"))?;
                        values.value(left_row).cmp(values.value(right_row))
                    }
                    DataType::Int64 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| anyhow!("Failed to downcast Int64 sort column"))?;
                        values.value(left_row).cmp(&values.value(right_row))
                    }
                    DataType::Int32 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<arrow::array::Int32Array>()
                            .ok_or_else(|| anyhow!("Failed to downcast Int32 sort column"))?;
                        values.value(left_row).cmp(&values.value(right_row))
                    }
                    DataType::UInt64 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<arrow::array::UInt64Array>()
                            .ok_or_else(|| anyhow!("Failed to downcast UInt64 sort column"))?;
                        values.value(left_row).cmp(&values.value(right_row))
                    }
                    DataType::UInt32 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                            .ok_or_else(|| anyhow!("Failed to downcast UInt32 sort column"))?;
                        values.value(left_row).cmp(&values.value(right_row))
                    }
                    DataType::Float32 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| anyhow!("Failed to downcast Float32 sort column"))?;
                        values
                            .value(left_row)
                            .partial_cmp(&values.value(right_row))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                    DataType::Float64 => {
                        let values = array
                            .as_any()
                            .downcast_ref::<arrow::array::Float64Array>()
                            .ok_or_else(|| anyhow!("Failed to downcast Float64 sort column"))?;
                        values
                            .value(left_row)
                            .partial_cmp(&values.value(right_row))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                    DataType::Boolean => {
                        let values = array
                            .as_any()
                            .downcast_ref::<arrow::array::BooleanArray>()
                            .ok_or_else(|| anyhow!("Failed to downcast Boolean sort column"))?;
                        values.value(left_row).cmp(&values.value(right_row))
                    }
                    other => {
                        return Err(anyhow!(
                            "Sorting on data type {:?} is not yet supported for column '{}'",
                            other,
                            clause.column
                        ));
                    }
                },
            };

            let ordering = if clause.ascending {
                ordering
            } else {
                ordering.reverse()
            };

            if ordering != std::cmp::Ordering::Equal {
                return Ok(ordering);
            }
        }

        Ok(std::cmp::Ordering::Equal)
    }

    fn resolve_aggregate_column(
        batch: &RecordBatch,
        aggregate: &super::optimizer::AggregateExpr,
    ) -> Result<usize> {
        let column = aggregate.column.as_ref().ok_or_else(|| {
            anyhow!(
                "Aggregate {:?} requires a column reference",
                aggregate.function
            )
        })?;

        Self::resolve_column_index(batch.schema().as_ref(), column).ok_or_else(|| {
            anyhow!(
                "Aggregate column '{}' was not found in schema {:?}",
                column,
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect::<Vec<_>>()
            )
        })
    }

    fn scalar_distinct_key(array: &dyn Array, row: usize) -> Result<String> {
        let key = match array.data_type() {
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Utf8 aggregate column"))?;
                format!("s:{}", values.value(row))
            }
            DataType::Int64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int64 aggregate column"))?;
                format!("i64:{}", values.value(row))
            }
            DataType::Int32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int32 aggregate column"))?;
                format!("i32:{}", values.value(row))
            }
            DataType::UInt64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt64 aggregate column"))?;
                format!("u64:{}", values.value(row))
            }
            DataType::UInt32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt32 aggregate column"))?;
                format!("u32:{}", values.value(row))
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float64 aggregate column"))?;
                format!("f64:{:x}", values.value(row).to_bits())
            }
            DataType::Float32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float32 aggregate column"))?;
                format!("f32:{:x}", values.value(row).to_bits())
            }
            DataType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Boolean aggregate column"))?;
                format!("b:{}", values.value(row))
            }
            other => {
                return Err(anyhow!(
                    "COUNT DISTINCT is not yet supported for data type {:?}",
                    other
                ));
            }
        };

        Ok(key)
    }

    fn count_distinct_values(array: &ArrayRef) -> Result<i64> {
        let mut distinct = HashSet::new();
        for row in 0..array.len() {
            if array.is_null(row) {
                continue;
            }
            distinct.insert(Self::scalar_distinct_key(array.as_ref(), row)?);
        }
        Ok(distinct.len() as i64)
    }

    fn row_distinct_key(batch: &RecordBatch, row: usize) -> Result<Vec<String>> {
        batch
            .columns()
            .iter()
            .map(|column| {
                if column.is_null(row) {
                    Ok("null".to_string())
                } else {
                    Self::scalar_distinct_key(column.as_ref(), row)
                }
            })
            .collect()
    }

    fn row_key_for_columns(
        batch: &RecordBatch,
        column_indices: &[usize],
        row: usize,
    ) -> Result<Vec<String>> {
        column_indices
            .iter()
            .map(|column_index| {
                let column = batch.column(*column_index);
                if column.is_null(row) {
                    Ok("null".to_string())
                } else {
                    Self::scalar_distinct_key(column.as_ref(), row)
                }
            })
            .collect()
    }

    fn take_batch_rows(batch: &RecordBatch, row_indices: &[usize]) -> Result<RecordBatch> {
        let take_indices =
            Self::build_take_indices(&row_indices.iter().copied().map(Some).collect::<Vec<_>>());
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &take_indices, None))
            .collect::<arrow::error::Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }

    fn aggregate_output_data_type(
        batch: &RecordBatch,
        aggregate: &super::optimizer::AggregateExpr,
    ) -> Result<DataType> {
        use super::optimizer::AggregateFunction;

        match aggregate.function {
            AggregateFunction::Count | AggregateFunction::CountDistinct => Ok(DataType::Int64),
            AggregateFunction::Sum | AggregateFunction::Avg => Ok(DataType::Float64),
            AggregateFunction::Min | AggregateFunction::Max => {
                let column_index = Self::resolve_aggregate_column(batch, aggregate)?;
                Ok(batch.column(column_index).data_type().clone())
            }
        }
    }

    fn aggregate_output_nullable(aggregate: &super::optimizer::AggregateExpr) -> bool {
        use super::optimizer::AggregateFunction;

        !matches!(
            aggregate.function,
            AggregateFunction::Count | AggregateFunction::CountDistinct
        )
    }

    fn aggregate_values_to_array(
        values: Vec<AggregateValue>,
        data_type: &DataType,
    ) -> Result<ArrayRef> {
        match data_type {
            DataType::Int64 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::Int64(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected Int64 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(Int64Array::from(values)))
            }
            DataType::Int32 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::Int32(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected Int32 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(arrow::array::Int32Array::from(values)))
            }
            DataType::UInt64 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::UInt64(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected UInt64 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(arrow::array::UInt64Array::from(values)))
            }
            DataType::UInt32 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::UInt32(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected UInt32 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(arrow::array::UInt32Array::from(values)))
            }
            DataType::Float64 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::Float64(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected Float64 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(arrow::array::Float64Array::from(values)))
            }
            DataType::Float32 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::Float32(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected Float32 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(Float32Array::from(values)))
            }
            DataType::Utf8 => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::Utf8(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected Utf8 aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let value_refs = values
                    .iter()
                    .map(|value| value.as_deref())
                    .collect::<Vec<_>>();
                Ok(Arc::new(StringArray::from(value_refs)))
            }
            DataType::Boolean => {
                let values = values
                    .into_iter()
                    .map(|value| match value {
                        AggregateValue::Boolean(value) => Ok(value),
                        other => Err(anyhow!(
                            "Expected Boolean aggregate values, got {:?}",
                            other.data_type()
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(arrow::array::BooleanArray::from(values)))
            }
            other => Err(anyhow!(
                "Grouped aggregate output is not supported for data type {:?}",
                other
            )),
        }
    }

    fn aggregate_numeric_values(array: &ArrayRef) -> Result<Vec<f64>> {
        match array.data_type() {
            DataType::Int64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int64 aggregate column"))?;
                Ok((0..values.len())
                    .filter(|row| !values.is_null(*row))
                    .map(|row| values.value(row) as f64)
                    .collect())
            }
            DataType::Int32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int32 aggregate column"))?;
                Ok((0..values.len())
                    .filter(|row| !values.is_null(*row))
                    .map(|row| values.value(row) as f64)
                    .collect())
            }
            DataType::UInt64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt64 aggregate column"))?;
                Ok((0..values.len())
                    .filter(|row| !values.is_null(*row))
                    .map(|row| values.value(row) as f64)
                    .collect())
            }
            DataType::UInt32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt32 aggregate column"))?;
                Ok((0..values.len())
                    .filter(|row| !values.is_null(*row))
                    .map(|row| values.value(row) as f64)
                    .collect())
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float64 aggregate column"))?;
                Ok((0..values.len())
                    .filter(|row| !values.is_null(*row))
                    .map(|row| values.value(row))
                    .collect())
            }
            DataType::Float32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float32 aggregate column"))?;
                Ok((0..values.len())
                    .filter(|row| !values.is_null(*row))
                    .map(|row| values.value(row) as f64)
                    .collect())
            }
            other => Err(anyhow!(
                "Aggregate numeric functions are not supported for data type {:?}",
                other
            )),
        }
    }

    fn aggregate_extremum(array: &ArrayRef, choose_min: bool) -> Result<AggregateValue> {
        match array.data_type() {
            DataType::Utf8 => {
                let values = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Utf8 aggregate column"))?;
                let mut best: Option<String> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best.as_deref() {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value.to_string());
                    }
                }
                Ok(AggregateValue::Utf8(best))
            }
            DataType::Int64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int64 aggregate column"))?;
                let mut best: Option<i64> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::Int64(best))
            }
            DataType::Int32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Int32 aggregate column"))?;
                let mut best: Option<i32> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::Int32(best))
            }
            DataType::UInt64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt64 aggregate column"))?;
                let mut best: Option<u64> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::UInt64(best))
            }
            DataType::UInt32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast UInt32 aggregate column"))?;
                let mut best: Option<u32> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::UInt32(best))
            }
            DataType::Float64 => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float64 aggregate column"))?;
                let mut best: Option<f64> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::Float64(best))
            }
            DataType::Float32 => {
                let values = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("Failed to downcast Float32 aggregate column"))?;
                let mut best: Option<f32> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                value < current
                            } else {
                                value > current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::Float32(best))
            }
            DataType::Boolean => {
                let values = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .ok_or_else(|| anyhow!("Failed to downcast Boolean aggregate column"))?;
                let mut best: Option<bool> = None;
                for row in 0..values.len() {
                    if values.is_null(row) {
                        continue;
                    }
                    let value = values.value(row);
                    let replace = match best {
                        None => true,
                        Some(current) => {
                            if choose_min {
                                !value && current
                            } else {
                                value && !current
                            }
                        }
                    };
                    if replace {
                        best = Some(value);
                    }
                }
                Ok(AggregateValue::Boolean(best))
            }
            other => Err(anyhow!(
                "Aggregate MIN/MAX is not yet supported for data type {:?}",
                other
            )),
        }
    }

    fn compute_aggregate_value(
        batch: &RecordBatch,
        aggregate: &super::optimizer::AggregateExpr,
    ) -> Result<AggregateValue> {
        use super::optimizer::AggregateFunction;

        match aggregate.function {
            AggregateFunction::Count => {
                let count = if let Some(column) = &aggregate.column {
                    let column_index = Self::resolve_column_index(batch.schema().as_ref(), column)
                        .ok_or_else(|| anyhow!("Aggregate column '{}' was not found", column))?;
                    let array = batch.column(column_index);
                    (0..array.len()).filter(|row| !array.is_null(*row)).count() as i64
                } else {
                    batch.num_rows() as i64
                };
                Ok(AggregateValue::Int64(Some(count)))
            }
            AggregateFunction::CountDistinct => {
                let column_index = Self::resolve_aggregate_column(batch, aggregate)?;
                let array = batch.column(column_index).clone();
                Ok(AggregateValue::Int64(Some(Self::count_distinct_values(
                    &array,
                )?)))
            }
            AggregateFunction::Sum => {
                let column_index = Self::resolve_aggregate_column(batch, aggregate)?;
                let array = batch.column(column_index).clone();
                let values = Self::aggregate_numeric_values(&array)?;
                Ok(AggregateValue::Float64(if values.is_empty() {
                    None
                } else {
                    Some(values.iter().sum())
                }))
            }
            AggregateFunction::Avg => {
                let column_index = Self::resolve_aggregate_column(batch, aggregate)?;
                let array = batch.column(column_index).clone();
                let values = Self::aggregate_numeric_values(&array)?;
                Ok(AggregateValue::Float64(if values.is_empty() {
                    None
                } else {
                    Some(values.iter().sum::<f64>() / values.len() as f64)
                }))
            }
            AggregateFunction::Min => {
                let column_index = Self::resolve_aggregate_column(batch, aggregate)?;
                let array = batch.column(column_index).clone();
                Self::aggregate_extremum(&array, true)
            }
            AggregateFunction::Max => {
                let column_index = Self::resolve_aggregate_column(batch, aggregate)?;
                let array = batch.column(column_index).clone();
                Self::aggregate_extremum(&array, false)
            }
        }
    }

    fn extract_join_key(
        batch: &RecordBatch,
        row: usize,
        key_indices: &[usize],
    ) -> Result<Option<Vec<JoinKey>>> {
        let mut key = Vec::with_capacity(key_indices.len());

        for &index in key_indices {
            let array = batch.column(index);
            let value = match array.data_type() {
                DataType::Utf8 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow!("Failed to downcast Utf8 join column"))?;
                    if values.is_null(row) {
                        return Ok(None);
                    }
                    JoinKey::Utf8(values.value(row).to_string())
                }
                DataType::Int64 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| anyhow!("Failed to downcast Int64 join column"))?;
                    if values.is_null(row) {
                        return Ok(None);
                    }
                    JoinKey::Int64(values.value(row))
                }
                DataType::Int32 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<arrow::array::Int32Array>()
                        .ok_or_else(|| anyhow!("Failed to downcast Int32 join column"))?;
                    if values.is_null(row) {
                        return Ok(None);
                    }
                    JoinKey::Int32(values.value(row))
                }
                DataType::UInt64 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<arrow::array::UInt64Array>()
                        .ok_or_else(|| anyhow!("Failed to downcast UInt64 join column"))?;
                    if values.is_null(row) {
                        return Ok(None);
                    }
                    JoinKey::UInt64(values.value(row))
                }
                DataType::UInt32 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<arrow::array::UInt32Array>()
                        .ok_or_else(|| anyhow!("Failed to downcast UInt32 join column"))?;
                    if values.is_null(row) {
                        return Ok(None);
                    }
                    JoinKey::UInt32(values.value(row))
                }
                DataType::Boolean => {
                    let values = array
                        .as_any()
                        .downcast_ref::<arrow::array::BooleanArray>()
                        .ok_or_else(|| anyhow!("Failed to downcast Boolean join column"))?;
                    if values.is_null(row) {
                        return Ok(None);
                    }
                    JoinKey::Boolean(values.value(row))
                }
                other => {
                    return Err(anyhow!(
                        "Join keys of type {:?} are not yet supported in federated execution",
                        other
                    ));
                }
            };

            key.push(value);
        }

        Ok(Some(key))
    }

    fn build_take_indices(indices: &[Option<usize>]) -> arrow::array::UInt32Array {
        let mut builder = UInt32Builder::with_capacity(indices.len());
        for index in indices {
            match index {
                Some(index) => builder.append_value(*index as u32),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }

    fn build_join_schema(&self, left: &Schema, right: &Schema) -> Arc<Schema> {
        let mut fields = Vec::with_capacity(left.fields().len() + right.fields().len());
        let mut seen = HashSet::new();

        for field in left.fields() {
            seen.insert(field.name().to_string());
            fields.push(field.as_ref().clone());
        }

        for field in right.fields() {
            let mut name = field.name().to_string();
            while seen.contains(&name) {
                name = format!("right_{}", name);
            }
            seen.insert(name.clone());
            fields.push(Self::clone_field_with_name(field.as_ref(), &name));
        }

        Arc::new(Schema::new(fields))
    }

    fn clone_field_with_name(field: &Field, name: &str) -> Field {
        if field.name() == name {
            field.clone()
        } else {
            Field::new(name, field.data_type().clone(), field.is_nullable())
                .with_metadata(field.metadata().clone())
        }
    }

    fn plan_output_schema(&self, node: &PlanNode) -> Result<Arc<Schema>> {
        match &node.node_type {
            PlanNodeType::Scan { .. } => Ok(Arc::new(Schema::new(
                node.output_columns
                    .iter()
                    .map(|column| Field::new(column, DataType::Utf8, true))
                    .collect::<Vec<_>>(),
            ))),
            PlanNodeType::VectorSearch { .. } => Ok(Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("score", DataType::Float32, false),
            ]))),
            PlanNodeType::RerankSearch { .. } => Ok(Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("score", DataType::Float32, false),
                Field::new("phase", DataType::Utf8, false),
                Field::new("match_features", DataType::Utf8, true),
                Field::new("summary_features", DataType::Utf8, true),
            ]))),
            PlanNodeType::GraphTraversal { .. } => Ok(Arc::new(Schema::new(vec![
                Field::new("node_id", DataType::Utf8, false),
                Field::new("label", DataType::Utf8, true),
                Field::new("properties", DataType::Utf8, false),
            ]))),
            PlanNodeType::DocumentQuery { .. } => Ok(Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("document", DataType::Utf8, false),
            ]))),
            PlanNodeType::ObservabilityQuery { query_type, .. } => Ok(match query_type {
                ObservabilityQueryType::Logs => Arc::new(Schema::new(vec![
                    Field::new("timestamp", DataType::Int64, false),
                    Field::new("level", DataType::Utf8, false),
                    Field::new("message", DataType::Utf8, false),
                ])),
                ObservabilityQueryType::Metrics => Arc::new(Schema::new(vec![
                    Field::new("timestamp", DataType::Int64, false),
                    Field::new("metric_name", DataType::Utf8, false),
                    Field::new("value", DataType::Float32, false),
                ])),
                ObservabilityQueryType::Traces => Arc::new(Schema::new(vec![
                    Field::new("trace_id", DataType::Utf8, false),
                    Field::new("span_id", DataType::Utf8, false),
                    Field::new("operation", DataType::Utf8, false),
                    Field::new("duration_ns", DataType::Int64, false),
                ])),
            }),
            PlanNodeType::HashJoin { left, right, .. }
            | PlanNodeType::IndexJoin { left, right, .. } => {
                let left_schema = self.plan_output_schema(left)?;
                let right_schema = self.plan_output_schema(right)?;
                Ok(self.build_join_schema(left_schema.as_ref(), right_schema.as_ref()))
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                let outer_schema = self.plan_output_schema(outer)?;
                let inner_schema = self.plan_output_schema(inner)?;
                Ok(self.build_join_schema(outer_schema.as_ref(), inner_schema.as_ref()))
            }
            PlanNodeType::Filter { input, .. }
            | PlanNodeType::Project { input, .. }
            | PlanNodeType::Distinct { input }
            | PlanNodeType::Sort { input, .. }
            | PlanNodeType::Limit { input, .. }
            | PlanNodeType::Aggregate { input, .. } => self.plan_output_schema(input),
            PlanNodeType::Union { inputs, .. } => inputs
                .first()
                .map(|input| self.plan_output_schema(input))
                .transpose()?
                .ok_or_else(|| anyhow!("Union plan has no inputs")),
        }
    }

    fn resolve_correlations_in_node(
        &self,
        node: &mut PlanNode,
        outer_batch: &RecordBatch,
        outer_row: usize,
    ) -> Result<bool> {
        match &mut node.node_type {
            PlanNodeType::VectorSearch {
                query_vector_source,
                ..
            }
            | PlanNodeType::RerankSearch {
                query_vector_source,
                ..
            } => {
                let Some(resolved) =
                    self.resolve_query_vector_for_row(query_vector_source, outer_batch, outer_row)?
                else {
                    return Ok(false);
                };
                *query_vector_source = VectorSource::Literal(resolved);
            }
            PlanNodeType::HashJoin { left, right, .. }
            | PlanNodeType::IndexJoin { left, right, .. } => {
                if !self.resolve_correlations_in_node(left, outer_batch, outer_row)? {
                    return Ok(false);
                }
                if !self.resolve_correlations_in_node(right, outer_batch, outer_row)? {
                    return Ok(false);
                }
            }
            PlanNodeType::NestedLoopJoin { outer, inner, .. } => {
                if !self.resolve_correlations_in_node(outer, outer_batch, outer_row)? {
                    return Ok(false);
                }
                if !self.resolve_correlations_in_node(inner, outer_batch, outer_row)? {
                    return Ok(false);
                }
            }
            PlanNodeType::Filter { input, .. }
            | PlanNodeType::Project { input, .. }
            | PlanNodeType::Distinct { input }
            | PlanNodeType::Sort { input, .. }
            | PlanNodeType::Limit { input, .. }
            | PlanNodeType::Aggregate { input, .. } => {
                if !self.resolve_correlations_in_node(input, outer_batch, outer_row)? {
                    return Ok(false);
                }
            }
            PlanNodeType::Union { inputs, .. } => {
                for input in inputs {
                    if !self.resolve_correlations_in_node(input, outer_batch, outer_row)? {
                        return Ok(false);
                    }
                }
            }
            PlanNodeType::Scan { .. }
            | PlanNodeType::GraphTraversal { .. }
            | PlanNodeType::DocumentQuery { .. }
            | PlanNodeType::ObservabilityQuery { .. } => {}
        }

        Ok(true)
    }

    fn join_batches(
        &self,
        left_batch: &RecordBatch,
        right_batch: &RecordBatch,
        left_indices: &[Option<usize>],
        right_indices: &[Option<usize>],
    ) -> Result<RecordBatch> {
        let joined_schema =
            self.build_join_schema(left_batch.schema().as_ref(), right_batch.schema().as_ref());
        if left_indices.is_empty() && right_indices.is_empty() {
            let arrays = joined_schema
                .fields()
                .iter()
                .map(|field| new_null_array(field.data_type(), 0))
                .collect();
            return Ok(RecordBatch::try_new(joined_schema, arrays)?);
        }

        let left_take = Self::build_take_indices(left_indices);
        let right_take = Self::build_take_indices(right_indices);
        let mut columns = Vec::with_capacity(
            left_batch.schema().fields().len() + right_batch.schema().fields().len(),
        );

        for column in left_batch.columns() {
            columns.push(take(column.as_ref(), &left_take, None)?);
        }
        for column in right_batch.columns() {
            columns.push(take(column.as_ref(), &right_take, None)?);
        }

        Ok(RecordBatch::try_new(joined_schema, columns)?)
    }

    /// Execute hash join
    async fn execute_hash_join(
        &self,
        left: &PlanNode,
        right: &PlanNode,
        join_keys: &[(String, String)],
        join_type: &JoinType,
    ) -> Result<FederatedExecutionResult> {
        let left_result = self.execute_node(left, false).await?;
        let right_result = self.execute_node(right, false).await?;

        let left_batch = self.merge_batches(&left_result)?;
        let right_batch = self.merge_batches(&right_result)?;
        let joined_schema =
            self.build_join_schema(left_batch.schema().as_ref(), right_batch.schema().as_ref());

        if *join_type == JoinType::Cross {
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            for left_row in 0..left_batch.num_rows() {
                for right_row in 0..right_batch.num_rows() {
                    left_indices.push(Some(left_row));
                    right_indices.push(Some(right_row));
                }
            }

            if left_indices.is_empty() {
                return Ok(FederatedExecutionResult::empty_with_schema(joined_schema));
            }

            let left_take = Self::build_take_indices(&left_indices);
            let right_take = Self::build_take_indices(&right_indices);
            let mut columns = Vec::with_capacity(
                left_batch.schema().fields().len() + right_batch.schema().fields().len(),
            );

            for column in left_batch.columns() {
                columns.push(take(column.as_ref(), &left_take, None)?);
            }
            for column in right_batch.columns() {
                columns.push(take(column.as_ref(), &right_take, None)?);
            }

            let batch = RecordBatch::try_new(joined_schema.clone(), columns)?;
            return Ok(FederatedExecutionResult::from_batch(batch));
        }

        if join_keys.is_empty() {
            return Err(anyhow!(
                "Hash join requires at least one join key unless join type is Cross"
            ));
        }

        let left_key_indices = join_keys
            .iter()
            .map(|(left_key, _)| {
                Self::resolve_column_index(left_batch.schema().as_ref(), left_key).ok_or_else(
                    || {
                        anyhow!(
                            "Join key '{}' was not found in left schema {:?}",
                            left_key,
                            left_batch
                                .schema()
                                .fields()
                                .iter()
                                .map(|field| field.name().clone())
                                .collect::<Vec<_>>()
                        )
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let right_key_indices = join_keys
            .iter()
            .map(|(_, right_key)| {
                Self::resolve_column_index(right_batch.schema().as_ref(), right_key).ok_or_else(
                    || {
                        anyhow!(
                            "Join key '{}' was not found in right schema {:?}",
                            right_key,
                            right_batch
                                .schema()
                                .fields()
                                .iter()
                                .map(|field| field.name().clone())
                                .collect::<Vec<_>>()
                        )
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mut right_index: HashMap<Vec<JoinKey>, Vec<usize>> = HashMap::new();
        for row in 0..right_batch.num_rows() {
            if let Some(key) = Self::extract_join_key(&right_batch, row, &right_key_indices)? {
                right_index.entry(key).or_default().push(row);
            }
        }

        let mut matched_right_rows = vec![false; right_batch.num_rows()];
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for left_row in 0..left_batch.num_rows() {
            let Some(key) = Self::extract_join_key(&left_batch, left_row, &left_key_indices)?
            else {
                if matches!(join_type, JoinType::Left | JoinType::Full) {
                    left_indices.push(Some(left_row));
                    right_indices.push(None);
                }
                continue;
            };

            if let Some(matches) = right_index.get(&key) {
                for &right_row in matches {
                    matched_right_rows[right_row] = true;
                    left_indices.push(Some(left_row));
                    right_indices.push(Some(right_row));
                }
            } else if matches!(join_type, JoinType::Left | JoinType::Full) {
                left_indices.push(Some(left_row));
                right_indices.push(None);
            }
        }

        if matches!(join_type, JoinType::Right | JoinType::Full) {
            for (right_row, matched) in matched_right_rows.iter().enumerate() {
                if !matched {
                    left_indices.push(None);
                    right_indices.push(Some(right_row));
                }
            }
        }

        if left_indices.is_empty() && right_indices.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(joined_schema));
        }

        let left_take = Self::build_take_indices(&left_indices);
        let right_take = Self::build_take_indices(&right_indices);
        let mut columns = Vec::with_capacity(
            left_batch.schema().fields().len() + right_batch.schema().fields().len(),
        );

        for column in left_batch.columns() {
            columns.push(take(column.as_ref(), &left_take, None)?);
        }
        for column in right_batch.columns() {
            columns.push(take(column.as_ref(), &right_take, None)?);
        }

        let batch = if left_indices.is_empty() {
            let arrays = joined_schema
                .fields()
                .iter()
                .map(|field| new_null_array(field.data_type(), 0))
                .collect();
            RecordBatch::try_new(joined_schema.clone(), arrays)?
        } else {
            RecordBatch::try_new(joined_schema.clone(), columns)?
        };

        Ok(FederatedExecutionResult::from_batch(batch))
    }

    /// Execute nested loop join (for LATERAL)
    async fn execute_nested_loop_join(
        &self,
        outer: &PlanNode,
        inner: &PlanNode,
        correlation: &[String],
    ) -> Result<FederatedExecutionResult> {
        let outer_result = self.execute_node(outer, false).await?;
        let outer_batch = self.merge_batches(&outer_result)?;
        let inner_schema = self.plan_output_schema(inner)?;
        let joined_schema =
            self.build_join_schema(outer_batch.schema().as_ref(), inner_schema.as_ref());

        if outer_batch.num_rows() == 0 {
            return Ok(FederatedExecutionResult::empty_with_schema(joined_schema));
        }

        let mut batches = Vec::new();
        let mut rows_produced = 0usize;

        for outer_row in 0..outer_batch.num_rows() {
            let mut resolved_inner = inner.clone();
            let should_execute_inner = self
                .resolve_correlations_in_node(&mut resolved_inner, &outer_batch, outer_row)
                .map_err(|error| {
                    anyhow!(
                        "Failed to resolve lateral join correlations {:?} for outer row {}: {}",
                        correlation,
                        outer_row,
                        error
                    )
                })?;
            if !should_execute_inner {
                continue;
            }

            let inner_result = self.execute_node(&resolved_inner, false).await?;
            let inner_batch = self.merge_batches(&inner_result)?;
            if inner_batch.num_rows() == 0 {
                continue;
            }

            let left_indices = vec![Some(outer_row); inner_batch.num_rows()];
            let right_indices = (0..inner_batch.num_rows()).map(Some).collect::<Vec<_>>();
            let batch =
                self.join_batches(&outer_batch, &inner_batch, &left_indices, &right_indices)?;
            rows_produced += batch.num_rows();
            batches.push(batch);
        }

        if batches.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(joined_schema));
        }

        Ok(FederatedExecutionResult {
            batches,
            schema: joined_schema,
            stats: FederatedExecutionStats {
                rows_produced,
                ..Default::default()
            },
        })
    }

    /// Execute index join
    async fn execute_index_join(
        &self,
        left: &PlanNode,
        right: &PlanNode,
        _index_lookup: &str,
    ) -> Result<FederatedExecutionResult> {
        // Similar to hash join but uses index
        self.execute_hash_join(left, right, &[], &JoinType::Inner)
            .await
    }

    /// Execute filter
    async fn execute_filter(
        &self,
        input: &PlanNode,
        predicate: &super::optimizer::Predicate,
    ) -> Result<FederatedExecutionResult> {
        let result = self.execute_node(input, false).await?;
        let batch = self.merge_batches(&result)?;

        if batch.num_rows() == 0 {
            return Ok(FederatedExecutionResult::empty_with_schema(batch.schema()));
        }

        let matched_indices = (0..batch.num_rows())
            .filter_map(
                |row| match Self::predicate_matches_row(&batch, row, predicate) {
                    Ok(true) => Some(Ok(Some(row))),
                    Ok(false) => None,
                    Err(error) => Some(Err(error)),
                },
            )
            .collect::<Result<Vec<_>>>()?;

        if matched_indices.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(batch.schema()));
        }

        let take_indices = Self::build_take_indices(&matched_indices);
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &take_indices, None))
            .collect::<arrow::error::Result<Vec<_>>>()?;

        let filtered = RecordBatch::try_new(batch.schema(), columns)?;
        Ok(FederatedExecutionResult::from_batch(filtered))
    }

    /// Execute projection
    async fn execute_project(
        &self,
        input: &PlanNode,
        columns: &[String],
        output_columns: &[String],
    ) -> Result<FederatedExecutionResult> {
        let result = self.execute_node(input, false).await?;
        if columns.iter().any(|column| column == "*") {
            return Ok(result);
        }

        let batch = self.merge_batches(&result)?;
        let projected_indices = columns
            .iter()
            .map(|column| {
                Self::resolve_column_index(batch.schema().as_ref(), column)
                    .ok_or_else(|| anyhow!("Projection column '{}' was not found", column))
            })
            .collect::<Result<Vec<_>>>()?;

        let batch_schema = batch.schema();
        let projected_fields = projected_indices
            .iter()
            .enumerate()
            .map(|(position, index)| {
                let field = batch_schema.field(*index);
                let requested_name = output_columns
                    .get(position)
                    .or_else(|| columns.get(position))
                    .map_or_else(|| field.name(), String::as_str);

                if field.name() == requested_name {
                    field.as_ref().clone()
                } else {
                    Self::clone_field_with_name(field, requested_name)
                }
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(projected_fields));

        if batch.num_rows() == 0 {
            return Ok(FederatedExecutionResult::empty_with_schema(schema));
        }

        let projected_columns = projected_indices
            .iter()
            .map(|index| batch.column(*index).clone())
            .collect::<Vec<_>>();
        let projected = RecordBatch::try_new(schema, projected_columns)?;
        Ok(FederatedExecutionResult::from_batch(projected))
    }

    fn project_result_to_output_columns(
        &self,
        result: FederatedExecutionResult,
        output_columns: &[String],
        allow_native_vector_passthrough: bool,
    ) -> Result<FederatedExecutionResult> {
        if output_columns.is_empty() || output_columns.iter().any(|column| column == "*") {
            return Ok(result);
        }

        let schema_matches = result.schema.fields().len() == output_columns.len()
            && result
                .schema
                .fields()
                .iter()
                .zip(output_columns.iter())
                .all(|(field, requested)| field.name() == requested);
        if schema_matches {
            return Ok(result);
        }

        let batch = self.merge_batches(&result)?;
        let mut projected_indices = output_columns
            .iter()
            .map(|column| {
                Self::resolve_column_index(batch.schema().as_ref(), column)
                    .ok_or_else(|| anyhow!("Projection column '{}' was not found", column))
            })
            .collect::<Result<Vec<_>>>()?;

        // TD-032: Preserve any FixedSizeList/List columns (vector data) that were
        // dynamically added by execute_document_query but aren't in the optimizer's
        // output_columns. This allows Arrow-native vector columns to pass through
        // to LATERAL join resolution without JSON parsing.
        if allow_native_vector_passthrough {
            let bs = batch.schema();
            for (idx, field) in bs.fields().iter().enumerate() {
                if !projected_indices.contains(&idx)
                    && matches!(
                        field.data_type(),
                        DataType::FixedSizeList(_, _) | DataType::List(_)
                    )
                {
                    projected_indices.push(idx);
                }
            }
        }

        let batch_schema = batch.schema();
        let projected_fields = projected_indices
            .iter()
            .enumerate()
            .map(|(position, index)| {
                let field = batch_schema.field(*index);
                // For dynamically added columns (beyond output_columns), use the field's own name
                let requested_name = output_columns
                    .get(position)
                    .map_or_else(|| field.name(), String::as_str);

                if field.name() == requested_name {
                    field.as_ref().clone()
                } else {
                    Self::clone_field_with_name(field, requested_name)
                }
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(projected_fields));

        if batch.num_rows() == 0 {
            return Ok(FederatedExecutionResult {
                batches: vec![],
                schema,
                stats: result.stats,
            });
        }

        let projected_columns = projected_indices
            .iter()
            .map(|index| batch.column(*index).clone())
            .collect::<Vec<_>>();
        let projected = RecordBatch::try_new(schema.clone(), projected_columns)?;
        Ok(FederatedExecutionResult {
            batches: vec![projected],
            schema,
            stats: result.stats,
        })
    }

    /// Execute DISTINCT on the current result schema.
    async fn execute_distinct(&self, input: &PlanNode) -> Result<FederatedExecutionResult> {
        let result = self.execute_node(input, false).await?;
        let batch = self.merge_batches(&result)?;

        if batch.num_rows() <= 1 {
            return Ok(FederatedExecutionResult::from_batch(batch));
        }

        let mut seen = HashSet::new();
        let mut distinct_rows = Vec::new();

        for row in 0..batch.num_rows() {
            let key = Self::row_distinct_key(&batch, row)?;
            if seen.insert(key) {
                distinct_rows.push(Some(row));
            }
        }

        let take_indices = Self::build_take_indices(&distinct_rows);
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &take_indices, None))
            .collect::<arrow::error::Result<Vec<_>>>()?;

        let distinct = RecordBatch::try_new(batch.schema(), columns)?;
        Ok(FederatedExecutionResult::from_batch(distinct))
    }

    /// Execute sort
    async fn execute_sort(
        &self,
        input: &PlanNode,
        order_by: &[super::optimizer::OrderByClause],
    ) -> Result<FederatedExecutionResult> {
        let result = self.execute_node(input, false).await?;
        if order_by.is_empty() {
            return Ok(result);
        }

        let batch = self.merge_batches(&result)?;
        if batch.num_rows() <= 1 {
            return Ok(FederatedExecutionResult::from_batch(batch));
        }

        for clause in order_by {
            let column_index = Self::resolve_column_index(batch.schema().as_ref(), &clause.column)
                .ok_or_else(|| anyhow!("Sort column '{}' was not found", clause.column))?;
            match batch.column(column_index).data_type() {
                DataType::Utf8
                | DataType::Int64
                | DataType::Int32
                | DataType::UInt64
                | DataType::UInt32
                | DataType::Float32
                | DataType::Float64
                | DataType::Boolean => {}
                other => {
                    return Err(anyhow!(
                        "Sorting on data type {:?} is not yet supported for column '{}'",
                        other,
                        clause.column
                    ));
                }
            }
        }

        let mut row_indices: Vec<usize> = (0..batch.num_rows()).collect();
        row_indices.sort_by(|left, right| {
            match Self::compare_rows(&batch, *left, *right, order_by) {
                Ok(ordering) => ordering,
                Err(e) => {
                    tracing::error!("Failed to compare rows during sort: {}", e);
                    // Fallback to maintaining original order on error
                    std::cmp::Ordering::Equal
                }
            }
        });

        let take_indices = Self::build_take_indices(
            &row_indices
                .into_iter()
                .map(Some)
                .collect::<Vec<Option<usize>>>(),
        );
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &take_indices, None))
            .collect::<arrow::error::Result<Vec<_>>>()?;

        let sorted = RecordBatch::try_new(batch.schema(), columns)?;
        Ok(FederatedExecutionResult::from_batch(sorted))
    }

    /// Execute limit
    async fn execute_limit(
        &self,
        input: &PlanNode,
        limit: usize,
        offset: usize,
    ) -> Result<FederatedExecutionResult> {
        let result = self.execute_node(input, false).await?;

        // Apply limit/offset to batches
        let mut remaining_offset = offset;
        let mut remaining_limit = limit;
        let mut output_batches = Vec::new();

        for batch in &result.batches {
            if remaining_offset >= batch.num_rows() {
                remaining_offset -= batch.num_rows();
                continue;
            }

            let start = remaining_offset;
            let end = (start + remaining_limit).min(batch.num_rows());
            remaining_offset = 0;

            if end > start {
                let sliced = batch.slice(start, end - start);
                remaining_limit -= sliced.num_rows();
                output_batches.push(sliced);
            }

            if remaining_limit == 0 {
                break;
            }
        }

        Ok(FederatedExecutionResult {
            batches: output_batches,
            schema: result.schema,
            stats: FederatedExecutionStats {
                rows_produced: limit.saturating_sub(remaining_limit),
                ..Default::default()
            },
        })
    }

    /// Execute aggregate
    async fn execute_aggregate(
        &self,
        input: &PlanNode,
        group_by: &[String],
        aggregates: &[super::optimizer::AggregateExpr],
    ) -> Result<FederatedExecutionResult> {
        let result = self.execute_node(input, false).await?;
        let batch = self.merge_batches(&result)?;

        if !group_by.is_empty() {
            let group_indices = group_by
                .iter()
                .map(|column| {
                    Self::resolve_column_index(batch.schema().as_ref(), column).ok_or_else(|| {
                        anyhow!(
                            "GROUP BY column '{}' was not found in the federated schema",
                            column
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let aggregate_types = aggregates
                .iter()
                .map(|aggregate| Self::aggregate_output_data_type(&batch, aggregate))
                .collect::<Result<Vec<_>>>()?;

            let group_fields = group_indices
                .iter()
                .map(|index| batch.schema().field(*index).as_ref().clone())
                .collect::<Vec<_>>();
            let aggregate_fields = aggregates
                .iter()
                .zip(aggregate_types.iter())
                .map(|(aggregate, data_type)| {
                    Field::new(
                        &aggregate.alias,
                        data_type.clone(),
                        Self::aggregate_output_nullable(aggregate),
                    )
                })
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(
                group_fields
                    .into_iter()
                    .chain(aggregate_fields)
                    .collect::<Vec<_>>(),
            ));

            if batch.num_rows() == 0 {
                return Ok(FederatedExecutionResult::empty_with_schema(schema));
            }

            let mut group_lookup: HashMap<Vec<String>, usize> = HashMap::new();
            let mut grouped_rows: Vec<Vec<usize>> = Vec::new();
            for row in 0..batch.num_rows() {
                let key = Self::row_key_for_columns(&batch, &group_indices, row)?;
                if let Some(existing_group) = group_lookup.get(&key) {
                    grouped_rows[*existing_group].push(row);
                } else {
                    group_lookup.insert(key, grouped_rows.len());
                    grouped_rows.push(vec![row]);
                }
            }

            let first_rows = grouped_rows
                .iter()
                .map(|rows| rows.first().copied())
                .collect::<Vec<_>>();
            let first_row_take = Self::build_take_indices(&first_rows);
            let mut columns = group_indices
                .iter()
                .map(|index| take(batch.column(*index).as_ref(), &first_row_take, None))
                .collect::<arrow::error::Result<Vec<_>>>()?;

            for (aggregate_index, aggregate) in aggregates.iter().enumerate() {
                let mut values = Vec::with_capacity(grouped_rows.len());
                for rows in &grouped_rows {
                    let grouped_batch = Self::take_batch_rows(&batch, rows)?;
                    values.push(Self::compute_aggregate_value(&grouped_batch, aggregate)?);
                }
                columns.push(Self::aggregate_values_to_array(
                    values,
                    &aggregate_types[aggregate_index],
                )?);
            }

            let aggregated = RecordBatch::try_new(schema.clone(), columns)?;
            return Ok(FederatedExecutionResult {
                batches: vec![aggregated],
                schema,
                stats: FederatedExecutionStats {
                    rows_produced: grouped_rows.len(),
                    ..Default::default()
                },
            });
        }

        let aggregate_values = aggregates
            .iter()
            .map(|aggregate| Self::compute_aggregate_value(&batch, aggregate))
            .collect::<Result<Vec<_>>>()?;

        let fields = aggregates
            .iter()
            .zip(aggregate_values.iter())
            .map(|(aggregate, value)| {
                Field::new(&aggregate.alias, value.data_type(), value.is_nullable())
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields));

        if aggregates.is_empty() {
            return Ok(FederatedExecutionResult::empty_with_schema(schema));
        }

        let columns = aggregate_values
            .into_iter()
            .map(AggregateValue::into_array)
            .collect::<Vec<_>>();
        let aggregated = RecordBatch::try_new(schema.clone(), columns)?;

        Ok(FederatedExecutionResult {
            batches: vec![aggregated],
            schema,
            stats: FederatedExecutionStats {
                rows_produced: 1,
                ..Default::default()
            },
        })
    }

    /// Execute union
    async fn execute_union(
        &self,
        inputs: &[PlanNode],
        _all: bool,
    ) -> Result<FederatedExecutionResult> {
        let mut all_batches = Vec::new();
        let mut schema = None;

        for input in inputs {
            let result = self.execute_node(input, false).await?;
            if schema.is_none() {
                schema = Some(result.schema.clone());
            }
            all_batches.extend(result.batches);
        }

        Ok(FederatedExecutionResult {
            batches: all_batches,
            schema: schema.unwrap_or_else(|| Arc::new(Schema::empty())),
            stats: FederatedExecutionStats::default(),
        })
    }
}

#[cfg(test)]
mod tests;
