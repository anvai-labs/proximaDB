// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! # Cross-modal source bridge (Track B — the §8 "zero-ETL multimodal" moat)
//!
//! Turns a **vector-search result set** into a DataFusion-joinable `(id, score)`
//! table, so a SINGLE SQL plan can join vector similarity against relational (and,
//! later, graph/document) data over the one canonical `ProximaRecord` spine. This is
//! the substrate of ProximaDB's durable differentiation per
//! `docs/12-design/DATA_WAREHOUSE_AND_ENGINEERING_COURSE_CORRECTION_2026_06_04.adoc`
//! §8: no competitor lets you filter-by-vector-similarity ⋈ relational-aggregate in
//! one query.
//!
//! ## Scope
//! All three non-relational modalities are now joinable relations, live-backed and reachable
//! from the pgwire DataFusion path (registered by `build_session_context`), so a four-way
//! `relational ⋈ vector ⋈ timeseries ⋈ graph` join executes in ONE data-local plan instead of a
//! client-side fan-out + intersect (ADR-053):
//!   * `vector_search(collection, query, k)` → `(id, score)` — via [`VectorSearchTableFunction`].
//!   * `timeseries_range(collection, start_ms, end_ms)` → `(timestamp, metric, value)` — via
//!     [`TimeseriesRangeTableFunction`].
//!   * `graph_traverse(graph_id, start_id, edge_type, max_depth)` → `(node_id, depth)` — via
//!     [`GraphTraverseTableFunction`], over the variable-length reachability traversal.
//!
//! Next slices (ADR-053): a single depth-carrying graph pass (vs the derived-depth `*k..k`
//! passes), and a frontend source node that lowers (via the P4 `logical_lowering`) into the
//! shared logical plane. All reuse the `*_to_batch` bridges below.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use proximadb_graph_query::service::GraphQueryReadService;
use proximadb_query::graph_lowering::parse_supported_graph_query;
use proximadb_query::graph_runtime::{
    execute_supported_graph_query_with_start_nodes, graph_query_row_id,
};
use proximadb_vector_query::{VectorSearchExpr, VectorSearchParams};

// TD-XMODAL-4 S2: only the test mocks' (v1) `search()` impls reference these proto
// types now — the production UDTF scan uses the v2 `unified_search_native` kernel.
#[cfg(test)]
use crate::proto::proximadb_v1::{SearchVectorRecord, VectorSearchRequest};
use crate::services::timeseries_service::{TimeSeriesService, TsPoint};

/// Hard cap on `graph_traverse` depth — bounds the number of `*k..k` traversal passes
/// (each is O(V+E)) regardless of a user-supplied `max_depth`.
const MAX_GRAPH_TRAVERSE_DEPTH: i64 = 64;

/// The Arrow schema a vector-search source exposes: `(id, score, metadata)`
/// (TD-XMODAL-4 S1). `id` joins against a relational key; `score` is the similarity
/// the SQL can rank/filter on (Float64 to match the pgwire `<->` operator path's
/// Float8); `metadata` is the record's stored payload as a JSON string, so a single
/// `SELECT * FROM vector_search(...)` is useful without a self-join.
pub fn vector_matches_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("metadata", DataType::Utf8, false),
    ]))
}

/// Convert vector-search results into an `(id, score, metadata)` [`RecordBatch`]
/// DataFusion can register as a table and JOIN against relational data on `id`.
/// The bridge from the vector modality into the shared (DataFusion) query plane.
///
/// `metadata` is rendered as JSON. TD-XMODAL-4 S2: the input is now the **v2
/// canonical [`OptimizedSearchRecord`]** (whose `metadata` is already
/// `ProximaValue`-typed, serde-native) — both the pgvector `<->` operator and this
/// UDTF flow through the one `unified_search_native` kernel, so no `SqlValue`
/// conversion is needed.
pub fn vector_matches_to_batch(
    results: &[crate::core::search::results::OptimizedSearchRecord],
) -> Result<RecordBatch, ArrowError> {
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    let scores: Vec<f64> = results.iter().map(|r| r.score as f64).collect();
    let metadata: Vec<String> = results
        .iter()
        .map(|r| serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".to_string()))
        .collect();
    RecordBatch::try_new(
        vector_matches_schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Float64Array::from(scores)),
            Arc::new(StringArray::from(metadata)),
        ],
    )
}

/// A DataFusion [`TableProvider`] whose `scan` runs a **live** vector search through
/// [`VectorOpsPort`] and exposes the `(id, score)` matches as a table — so a single
/// SQL plan can join vector similarity against relational data (§8 moat). This is the
/// production-backed counterpart of [`vector_matches_to_batch`]: register it in a
/// `SessionContext` and the query planner can scan/join/order it like any table.
pub struct VectorSearchTableProvider {
    vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
    search: VectorSearchExpr,
    identity: proximadb_runtime::OwnedPortIdentity,
}

impl VectorSearchTableProvider {
    /// Build a provider for one parameterized similarity search.
    pub fn new(
        vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
        search: VectorSearchExpr,
        identity: proximadb_runtime::OwnedPortIdentity,
    ) -> Self {
        Self {
            vector_ops,
            search,
            identity,
        }
    }
}

// Manual `Debug` (required by `TableProvider`): the `VectorOpsPort` trait object is
// not `Debug`, so print only the query parameters.
impl std::fmt::Debug for VectorSearchTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorSearchTableProvider")
            .field("collection_id", &self.search.collection)
            .field("vector_column", &self.search.vector_column)
            .field("top_k", &self.search.top_k)
            .field("tenant_id", &self.identity.tenant_id)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for VectorSearchTableProvider {
    fn schema(&self) -> SchemaRef {
        vector_matches_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // TD-XMODAL-4 S2: run the live similarity search through the ONE canonical
        // native kernel `VectorOpsPort::unified_search_native` (the same kernel the
        // pgvector `<->` operator uses) — fail-closed tenant-scoped by the impl —
        // and bridge the v2 `OptimizedSearchRecord`s into the plane. Metadata-filter
        // pushdown from the SQL `filters` is a follow-up (S3); pass `None` for now.
        let results = self
            .vector_ops
            .unified_search_native(&self.search, self.identity.as_borrowed())
            .await
            .map_err(|e| DataFusionError::Execution(format!("vector search: {e}")))?;
        let batch = vector_matches_to_batch(&results).map_err(DataFusionError::from)?;
        // Delegate to a `MemTable` so projection/filter/limit are honored uniformly.
        let mem = MemTable::try_new(vector_matches_schema(), vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// DataFusion table-valued function `vector_search(collection, query, k)` returning a
/// [`VectorSearchTableProvider`] — makes the cross-modal join expressible directly in
/// SQL (and so reachable from the pgwire DataFusion path):
/// `SELECT d.title, v.score
///  FROM docs d JOIN vector_search('docs_vec', '[0.1,0.2,0.3]', 10) v ON d.id = v.id`.
/// Register once per `SessionContext`:
/// `ctx.register_udtf("vector_search", Arc::new(VectorSearchTableFunction::new(ops)))`.
pub struct VectorSearchTableFunction {
    vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
    /// The complete connection identity forwarded to the canonical vector-search
    /// port. ABAC must not be reduced to a tenant-only approximation here.
    identity: proximadb_runtime::OwnedPortIdentity,
}

impl VectorSearchTableFunction {
    /// Capture the live vector service the function will search (no tenant scope).
    pub fn new(vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>) -> Self {
        Self {
            vector_ops,
            identity: proximadb_runtime::OwnedPortIdentity::default(),
        }
    }

    /// Capture the vector service and the complete request identity.
    pub fn with_identity(
        vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
        identity: proximadb_runtime::OwnedPortIdentity,
    ) -> Self {
        Self {
            vector_ops,
            identity,
        }
    }
}

impl std::fmt::Debug for VectorSearchTableFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorSearchTableFunction")
            .finish_non_exhaustive()
    }
}

impl TableFunctionImpl for VectorSearchTableFunction {
    fn call(&self, args: &[Expr]) -> DFResult<Arc<dyn TableProvider>> {
        let collection = arg_string(args, 0).ok_or_else(|| {
            DataFusionError::Plan(
                "vector_search(collection, query, k): arg 1 must be a collection-name string"
                    .into(),
            )
        })?;
        let query_text = arg_string(args, 1).ok_or_else(|| {
            DataFusionError::Plan(
                "vector_search: arg 2 must be a '[..]' query-vector string".into(),
            )
        })?;
        let top_k = arg_i64(args, 2).ok_or_else(|| {
            DataFusionError::Plan("vector_search: arg 3 must be an integer top_k".into())
        })?;
        let query_vector = parse_vector_literal(&query_text).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "vector_search: cannot parse query vector {query_text:?}"
            ))
        })?;
        if collection.trim().is_empty() {
            return Err(DataFusionError::Plan(
                "vector_search: collection must not be empty".into(),
            ));
        }
        if query_vector.is_empty() {
            return Err(DataFusionError::Plan(
                "vector_search: query vector must contain at least one dimension".into(),
            ));
        }
        if top_k <= 0 {
            return Err(DataFusionError::Plan(
                "vector_search: top_k must be greater than zero".into(),
            ));
        }
        let top_k = u32::try_from(top_k).map_err(|_| {
            DataFusionError::Plan("vector_search: top_k exceeds the supported u32 range".into())
        })?;
        // Shared server top_k cap (TD-SECSCAN-1 §5).
        proximadb_search_types::top_k_limit::check_top_k(u64::from(top_k))
            .map_err(|e| DataFusionError::Plan(format!("vector_search: {e}")))?;
        Ok(Arc::new(VectorSearchTableProvider::new(
            self.vector_ops.clone(),
            VectorSearchExpr {
                collection,
                vector_column: None,
                query_vector,
                top_k,
                threshold: None,
                metric: proximadb_distance_types::DistanceMetric::L2,
                filter: None,
                params: VectorSearchParams::default(),
            },
            self.identity.clone(),
        )))
    }
}

/// Extract a string-literal argument at position `i`.
fn arg_string(args: &[Expr], i: usize) -> Option<String> {
    match args.get(i)? {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Some(s.clone()),
        _ => None,
    }
}

/// Extract an integer-literal argument at position `i`.
fn arg_i64(args: &[Expr], i: usize) -> Option<i64> {
    match args.get(i)? {
        Expr::Literal(ScalarValue::Int64(Some(n)), _) => Some(*n),
        Expr::Literal(ScalarValue::Int32(Some(n)), _) => Some(*n as i64),
        Expr::Literal(ScalarValue::UInt64(Some(n)), _) => i64::try_from(*n).ok(),
        _ => None,
    }
}

/// Parse a pgvector-style text literal `[0.1, 0.2, 0.3]` into `Vec<f32>`.
fn parse_vector_literal(text: &str) -> Option<Vec<f32>> {
    let inner = text.trim().strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner.split(',').try_fold(Vec::new(), |mut values, part| {
        let value = part.trim().parse::<f32>().ok()?;
        if !value.is_finite() {
            return None;
        }
        values.push(value);
        Some(values)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeseries slice (§8 moat, second modality) — `timeseries_range(collection, start_ms,
// end_ms)` as a joinable table. The timeseries analogue of `vector_search`: it turns a
// time-range read into a DataFusion `(timestamp, metric, value)` relation so a SINGLE SQL
// plan can join a timeseries anomaly-scan against relational/graph data — the graph⋈timeseries
// root-cause join the vertical copilots do client-side today becomes one data-local query.
// ─────────────────────────────────────────────────────────────────────────────

/// The Arrow schema a timeseries-range source exposes for joins: `(timestamp, metric, value)`
/// in long form — one row per (point, value-column), so a collection with several value
/// columns projects generically and SQL can `WHERE metric = '…'` + bucket/aggregate.
pub fn timeseries_range_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

/// Convert time-series points into a `(timestamp, metric, value)` [`RecordBatch`]. Value columns
/// are emitted in sorted name order for deterministic output.
pub fn timeseries_points_to_batch(points: &[TsPoint]) -> Result<RecordBatch, ArrowError> {
    let mut timestamps: Vec<i64> = Vec::new();
    let mut metrics: Vec<String> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for point in points {
        let mut cols: Vec<(&String, &f64)> = point.values.iter().collect();
        cols.sort_by(|a, b| a.0.cmp(b.0));
        for (metric, value) in cols {
            timestamps.push(point.timestamp);
            metrics.push(metric.clone());
            values.push(*value);
        }
    }
    RecordBatch::try_new(
        timeseries_range_schema(),
        vec![
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(metrics)),
            Arc::new(Float64Array::from(values)),
        ],
    )
}

/// Port yielding time-series points for a `[start_ms, end_ms]` range of one collection — the
/// timeseries analogue of `VectorOpsPort`. Kept as a trait so the table source is unit-testable
/// with a fixed set of points, independent of the process time-series service.
#[async_trait]
pub trait TimeseriesScanPort: Send + Sync {
    /// Read the points of `tenant`'s `collection` in `[start_ms, end_ms]` (epoch millis). The
    /// tenant scopes the read structurally (the service selects the tenant's engine); the
    /// collection name is tenant-clean.
    async fn range(
        &self,
        tenant: &str,
        collection: &str,
        start_ms: i64,
        end_ms: i64,
    ) -> anyhow::Result<Vec<TsPoint>>;
}

/// Production adapter: the process [`TimeSeriesService`] as a [`TimeseriesScanPort`].
struct TimeSeriesServiceScan(Arc<TimeSeriesService>);

#[async_trait]
impl TimeseriesScanPort for TimeSeriesServiceScan {
    async fn range(
        &self,
        tenant: &str,
        collection: &str,
        start_ms: i64,
        end_ms: i64,
    ) -> anyhow::Result<Vec<TsPoint>> {
        self.0
            .query(tenant, collection, start_ms, end_ms, None)
            .await
    }
}

/// A DataFusion [`TableProvider`] whose `scan` reads a live time range through a
/// [`TimeseriesScanPort`] and exposes `(timestamp, metric, value)` rows — so a single SQL plan
/// can join a timeseries scan against relational/graph data.
pub struct TimeseriesRangeTableProvider {
    scan: Arc<dyn TimeseriesScanPort>,
    tenant: String,
    collection: String,
    start_ms: i64,
    end_ms: i64,
}

impl TimeseriesRangeTableProvider {
    /// Build a provider for one parameterized time-range read, scoped to `tenant`.
    pub fn new(
        scan: Arc<dyn TimeseriesScanPort>,
        tenant: impl Into<String>,
        collection: impl Into<String>,
        start_ms: i64,
        end_ms: i64,
    ) -> Self {
        Self {
            scan,
            tenant: tenant.into(),
            collection: collection.into(),
            start_ms,
            end_ms,
        }
    }
}

impl std::fmt::Debug for TimeseriesRangeTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeseriesRangeTableProvider")
            .field("tenant", &self.tenant)
            .field("collection", &self.collection)
            .field("start_ms", &self.start_ms)
            .field("end_ms", &self.end_ms)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for TimeseriesRangeTableProvider {
    fn schema(&self) -> SchemaRef {
        timeseries_range_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let points = self
            .scan
            .range(&self.tenant, &self.collection, self.start_ms, self.end_ms)
            .await
            .map_err(|e| DataFusionError::Execution(format!("timeseries range: {e}")))?;
        let batch = timeseries_points_to_batch(&points).map_err(DataFusionError::from)?;
        // Delegate to a `MemTable` so projection/filter/limit are honored uniformly.
        let mem = MemTable::try_new(timeseries_range_schema(), vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// DataFusion table-valued function `timeseries_range(collection, start_ms, end_ms)` returning a
/// [`TimeseriesRangeTableProvider`] — makes the timeseries⋈relational/graph join expressible
/// directly in SQL (and so reachable from the pgwire DataFusion path):
/// `SELECT t.metric, MAX(t.value)
///  FROM timeseries_range('sensor', 1700000000000, 1700003600000) t GROUP BY t.metric`.
/// Register once per `SessionContext`:
/// `ctx.register_udtf("timeseries_range", Arc::new(TimeseriesRangeTableFunction::new(port)))`.
pub struct TimeseriesRangeTableFunction {
    scan: Arc<dyn TimeseriesScanPort>,
    /// The connection tenant, passed structurally to the scan port (which selects the tenant's
    /// engine) — NOT folded into the collection name. `None` ⇒ the canonical `DEFAULT_TENANT`.
    tenant: Option<String>,
}

impl TimeseriesRangeTableFunction {
    /// Capture the scan port the function will read (no tenant scope).
    pub fn new(scan: Arc<dyn TimeseriesScanPort>) -> Self {
        Self { scan, tenant: None }
    }

    /// Capture the scan port AND the connection tenant, so the range read folds the collection
    /// key the same way the REST write path does (TD-XMODAL-6).
    pub fn with_tenant(scan: Arc<dyn TimeseriesScanPort>, tenant: Option<String>) -> Self {
        Self { scan, tenant }
    }

    /// Build the function backed by the process time-series service (no tenant scope).
    pub fn from_service(service: Arc<TimeSeriesService>) -> Self {
        Self {
            scan: Arc::new(TimeSeriesServiceScan(service)),
            tenant: None,
        }
    }

    /// Build the function backed by the process time-series service, scoped to the connection
    /// tenant — the pgwire OLAP route wiring that makes `timeseries_range` read REST-written data.
    pub fn from_service_with_tenant(
        service: Arc<TimeSeriesService>,
        tenant: Option<String>,
    ) -> Self {
        Self {
            scan: Arc::new(TimeSeriesServiceScan(service)),
            tenant,
        }
    }
}

impl std::fmt::Debug for TimeseriesRangeTableFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeseriesRangeTableFunction")
            .finish_non_exhaustive()
    }
}

impl TableFunctionImpl for TimeseriesRangeTableFunction {
    fn call(&self, args: &[Expr]) -> DFResult<Arc<dyn TableProvider>> {
        let collection = arg_string(args, 0).ok_or_else(|| {
            DataFusionError::Plan(
                "timeseries_range(collection, start_ms, end_ms): arg 1 must be a collection-name string"
                    .into(),
            )
        })?;
        let start_ms = arg_i64(args, 1).ok_or_else(|| {
            DataFusionError::Plan("timeseries_range: arg 2 must be an integer start_ms".into())
        })?;
        let end_ms = arg_i64(args, 2).ok_or_else(|| {
            DataFusionError::Plan("timeseries_range: arg 3 must be an integer end_ms".into())
        })?;
        if collection.trim().is_empty() {
            return Err(DataFusionError::Plan(
                "timeseries_range: collection must not be empty".into(),
            ));
        }
        if end_ms < start_ms {
            return Err(DataFusionError::Plan(
                "timeseries_range: end_ms must be >= start_ms".into(),
            ));
        }
        // Structural tenant scoping: pass the connection tenant (defaulting to the one canonical
        // DEFAULT_TENANT) + the tenant-CLEAN collection name to the service, which selects the
        // tenant's engine. No name-folding — the read hits the same per-tenant engine the REST
        // ingest wrote to.
        let tenant = proximadb_tenant::resolve_request_tenant(self.tenant.as_deref());
        Ok(Arc::new(TimeseriesRangeTableProvider::new(
            self.scan.clone(),
            tenant,
            collection,
            start_ms,
            end_ms,
        )))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph slice (§8 moat, third modality) — `graph_traverse(graph_id, start_id, edge_type,
// max_depth)` as a joinable table. The graph analogue of vector_search / timeseries_range:
// it turns a variable-length reachability traversal into a DataFusion `(node_id, depth)`
// relation, so a SINGLE SQL plan can join graph reachability against relational / vector /
// timeseries data — the four-way zero-ETL cross-modal join over the pgwire OLAP route.
// ─────────────────────────────────────────────────────────────────────────────

/// The Arrow schema a graph-traversal source exposes for joins: `(node_id, depth)`.
pub fn graph_traverse_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::Utf8, false),
        Field::new("depth", DataType::Int32, false),
    ]))
}

/// Convert `(node_id, depth)` reachable-node rows into a [`RecordBatch`].
pub fn graph_nodes_to_batch(nodes: &[(String, i32)]) -> Result<RecordBatch, ArrowError> {
    let ids: Vec<&str> = nodes.iter().map(|(id, _)| id.as_str()).collect();
    let depths: Vec<i32> = nodes.iter().map(|(_, d)| *d).collect();
    RecordBatch::try_new(
        graph_traverse_schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Int32Array::from(depths)),
        ],
    )
}

/// A DataFusion [`TableProvider`] whose `scan` walks a graph's variable-length reachability
/// (`start_id -[:edge_type*1..max_depth]-> x`) live through a [`GraphQueryReadService`] and
/// exposes the reachable `(node_id, depth)` set — so a single SQL plan can join graph
/// reachability against relational / vector / timeseries data.
pub struct GraphTraverseTableProvider {
    graph_ops: Arc<dyn GraphQueryReadService>,
    graph_id: String,
    start_id: String,
    edge_type: String,
    max_depth: i64,
    /// The connection tenant. `scan` composes the SAME structural scope the graph write path
    /// uses (`{tenant}/{graph_id}` via [`crate::graph::scoped_graph_id`]), so a traverse reads
    /// exactly the tenant's graph. `None` ⇒ the canonical default tenant.
    tenant: Option<String>,
}

impl GraphTraverseTableProvider {
    /// Build a provider for one parameterized reachability traversal, scoped to `tenant`.
    pub fn new(
        graph_ops: Arc<dyn GraphQueryReadService>,
        graph_id: impl Into<String>,
        start_id: impl Into<String>,
        edge_type: impl Into<String>,
        max_depth: i64,
        tenant: Option<String>,
    ) -> Self {
        Self {
            graph_ops,
            graph_id: graph_id.into(),
            start_id: start_id.into(),
            edge_type: edge_type.into(),
            max_depth,
            tenant,
        }
    }
}

impl std::fmt::Debug for GraphTraverseTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphTraverseTableProvider")
            .field("graph_id", &self.graph_id)
            .field("start_id", &self.start_id)
            .field("edge_type", &self.edge_type)
            .field("max_depth", &self.max_depth)
            .field("tenant", &self.tenant)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for GraphTraverseTableProvider {
    fn schema(&self) -> SchemaRef {
        graph_traverse_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Derived-depth traversal: one `*k..k` reachability pass per depth k in 1..=max, tagging
        // each first-seen node with its (shortest) depth. The inner BFS is visited-set bounded,
        // so each pass is O(V+E); a single depth-carrying pass is a follow-on TD.
        let max_depth = self.max_depth.clamp(1, MAX_GRAPH_TRAVERSE_DEPTH);
        // Structural tenant scope: read the SAME `{tenant}/{graph_id}` key the graph write path
        // composes (`TenantGraphOps`/`for_tenant`), so a cross-modal traverse sees exactly this
        // tenant's graph and never another's. `None` ⇒ the canonical default tenant.
        let tenant = proximadb_tenant::resolve_request_tenant(self.tenant.as_deref());
        let scoped_graph_id = crate::graph::scoped_graph_id(&tenant, &self.graph_id)
            .map_err(|e| DataFusionError::Execution(format!("graph_traverse tenant scope: {e}")))?;
        let mut rows: Vec<(String, i32)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for k in 1..=max_depth {
            let cypher = format!(
                "MATCH (a)-[:{edge}*{k}..{k}]->(x) RETURN x.id AS node_id",
                edge = self.edge_type
            );
            let parsed = parse_supported_graph_query(&cypher, None, Some(&scoped_graph_id))
                .map_err(|e| DataFusionError::Execution(format!("graph_traverse parse: {e}")))?;
            let executed = execute_supported_graph_query_with_start_nodes(
                &*self.graph_ops,
                &parsed,
                Some(std::slice::from_ref(&self.start_id)),
            )
            .await
            .map_err(|e| DataFusionError::Execution(format!("graph_traverse: {e}")))?;
            for row in &executed.rows {
                let id = graph_query_row_id(row, 0);
                if !id.is_empty() && seen.insert(id.clone()) {
                    rows.push((id, k as i32));
                }
            }
        }
        let batch = graph_nodes_to_batch(&rows).map_err(DataFusionError::from)?;
        let mem = MemTable::try_new(graph_traverse_schema(), vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// DataFusion table-valued function `graph_traverse(graph_id, start_id, edge_type, max_depth)`
/// returning a [`GraphTraverseTableProvider`] — makes the reachability ⋈ relational join
/// expressible directly in SQL (reachable from the pgwire DataFusion path):
/// `SELECT a.tier, g.node_id, g.depth
///  FROM accounts a JOIN graph_traverse('aml','acct-7','sent_to',4) g ON a.id = g.node_id`.
/// Register once per `SessionContext`:
/// `ctx.register_udtf("graph_traverse", Arc::new(GraphTraverseTableFunction::new(graph_ops)))`.
pub struct GraphTraverseTableFunction {
    graph_ops: Arc<dyn GraphQueryReadService>,
    /// The connection tenant, forwarded to each provider so the traverse reads the same
    /// `{tenant}/{graph_id}` scope the graph write path composes. `None` ⇒ the canonical
    /// default tenant (matches a default-tenant write).
    tenant: Option<String>,
}

impl GraphTraverseTableFunction {
    /// Capture the live graph read service the function will traverse (no tenant scope ⇒
    /// the canonical default tenant). Prefer [`with_tenant`](Self::with_tenant) on the pgwire
    /// path so the traverse reads the connection's tenant.
    pub fn new(graph_ops: Arc<dyn GraphQueryReadService>) -> Self {
        Self {
            graph_ops,
            tenant: None,
        }
    }

    /// Capture the graph read service AND the connection tenant — so the traverse reads the
    /// SAME structural scope (`{tenant}/{graph_id}`) the graph write path (`TenantGraphOps` on
    /// gRPC/Flight) composed. Without this the read would hit a different (or default) scope and
    /// return 0 rows for tenant-scoped graphs (closes TD-XMODAL-6 for the graph modality).
    pub fn with_tenant(graph_ops: Arc<dyn GraphQueryReadService>, tenant: Option<String>) -> Self {
        Self { graph_ops, tenant }
    }
}

impl std::fmt::Debug for GraphTraverseTableFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphTraverseTableFunction")
            .field("tenant", &self.tenant)
            .finish_non_exhaustive()
    }
}

/// A bare identifier (edge type / relationship name) safe to interpolate into the internal
/// Cypher: non-empty ASCII alphanumerics + `_`. (The start-node id is passed out-of-band via
/// `execute_supported_graph_query_with_start_nodes`, so it never enters the query string.)
fn is_bare_identifier(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl TableFunctionImpl for GraphTraverseTableFunction {
    fn call(&self, args: &[Expr]) -> DFResult<Arc<dyn TableProvider>> {
        let graph_id = arg_string(args, 0).ok_or_else(|| {
            DataFusionError::Plan(
                "graph_traverse(graph_id, start_id, edge_type, max_depth): arg 1 must be a graph-id string"
                    .into(),
            )
        })?;
        let start_id = arg_string(args, 1).ok_or_else(|| {
            DataFusionError::Plan("graph_traverse: arg 2 must be a start-node-id string".into())
        })?;
        let edge_type = arg_string(args, 2).ok_or_else(|| {
            DataFusionError::Plan("graph_traverse: arg 3 must be an edge-type string".into())
        })?;
        let max_depth = arg_i64(args, 3).ok_or_else(|| {
            DataFusionError::Plan("graph_traverse: arg 4 must be an integer max_depth".into())
        })?;
        if graph_id.trim().is_empty() {
            return Err(DataFusionError::Plan(
                "graph_traverse: graph_id must not be empty".into(),
            ));
        }
        if start_id.trim().is_empty() {
            return Err(DataFusionError::Plan(
                "graph_traverse: start_id must not be empty".into(),
            ));
        }
        if !is_bare_identifier(&edge_type) {
            return Err(DataFusionError::Plan(
                "graph_traverse: edge_type must be a bare identifier (letters, digits, underscore)"
                    .into(),
            ));
        }
        if max_depth <= 0 {
            return Err(DataFusionError::Plan(
                "graph_traverse: max_depth must be greater than zero".into(),
            ));
        }
        Ok(Arc::new(GraphTraverseTableProvider::new(
            self.graph_ops.clone(),
            graph_id,
            start_id,
            edge_type,
            max_depth,
            self.tenant.clone(),
        )))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Document slice (ADR-055 P-DFSource) — `documents(collection)` as a joinable SQL relation over
// the converged document store. Phase 1 is correctness-first: it exposes `(id, props)` (props as a
// lossless JSON string) via the same MemTable-delegation the other cross-modal sources use, so a
// single SQL plan can join documents against relational / vector / timeseries / graph data. The
// PAX-native pruning payoff (shredded-column pushdown) is a later phase (P-Pushdown); this ADDS a
// SQL surface and does NOT reroute the existing REST/gRPC `query_documents` API.
// ─────────────────────────────────────────────────────────────────────────────

/// The Arrow schema a `documents(collection)` source exposes: `(id, props)` where `props` is the
/// document body serialized to a JSON string (lossless; shredding into typed columns is P-Pushdown).
pub fn documents_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("props", DataType::Utf8, true),
    ]))
}

/// Convert `(id, props_json)` document rows into a [`RecordBatch`].
pub fn document_rows_to_batch(rows: &[(String, String)]) -> Result<RecordBatch, ArrowError> {
    let ids: Vec<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
    let props: Vec<&str> = rows.iter().map(|(_, p)| p.as_str()).collect();
    RecordBatch::try_new(
        documents_schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(props)),
        ],
    )
}

/// Port yielding a collection's documents as `(id, props_json)` rows — the document analogue of
/// [`TimeseriesScanPort`]. Kept as a trait so the table source is unit-testable with a fixed set of
/// documents, independent of the process document service.
#[async_trait]
pub trait DocumentScanPort: Send + Sync {
    /// Read `tenant`'s `collection` as `(id, props_json)` rows. The tenant scopes the read
    /// structurally; `collection` is tenant-clean. Dead/TTL-expired documents are already filtered
    /// by the underlying `query_documents`.
    async fn scan(&self, tenant: &str, collection: &str) -> anyhow::Result<Vec<(String, String)>>;
}

/// Production adapter: the process [`DocumentService`] as a [`DocumentScanPort`].
struct DocumentServiceScan(Arc<crate::storage::document::DocumentService>);

#[async_trait]
impl DocumentScanPort for DocumentServiceScan {
    async fn scan(&self, tenant: &str, collection: &str) -> anyhow::Result<Vec<(String, String)>> {
        // Scope the collection key the way the REST/middleware path does: a named tenant reads
        // `{tenant}/{collection}`; the canonical `DEFAULT_TENANT` uses the bare key.
        let scoped = if tenant == proximadb_tenant::DEFAULT_TENANT {
            collection.to_string()
        } else {
            format!("{tenant}/{collection}")
        };
        let result = self
            .0
            .query_documents(
                &scoped,
                crate::storage::document::DocumentQueryParams::default(),
            )
            .await?;
        Ok(result
            .documents
            .into_iter()
            .map(|doc| {
                (
                    doc.id,
                    crate::core::search::sql_value_filter::proxima_tree_to_canonical_json_string(
                        &doc.props,
                    ),
                )
            })
            .collect())
    }
}

/// A DataFusion [`TableProvider`] whose `scan` reads a collection's documents through a
/// [`DocumentScanPort`] and exposes `(id, props)` rows — so a single SQL plan can join documents
/// against relational / vector / timeseries / graph data.
pub struct DocumentTableProvider {
    scan: Arc<dyn DocumentScanPort>,
    tenant: String,
    collection: String,
}

impl DocumentTableProvider {
    /// Build a provider for one collection read, scoped to `tenant`.
    pub fn new(
        scan: Arc<dyn DocumentScanPort>,
        tenant: impl Into<String>,
        collection: impl Into<String>,
    ) -> Self {
        Self {
            scan,
            tenant: tenant.into(),
            collection: collection.into(),
        }
    }
}

impl std::fmt::Debug for DocumentTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentTableProvider")
            .field("tenant", &self.tenant)
            .field("collection", &self.collection)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for DocumentTableProvider {
    fn schema(&self) -> SchemaRef {
        documents_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let rows = self
            .scan
            .scan(&self.tenant, &self.collection)
            .await
            .map_err(|e| DataFusionError::Execution(format!("documents scan: {e}")))?;
        let batch = document_rows_to_batch(&rows).map_err(DataFusionError::from)?;
        // Delegate to a `MemTable` so projection/filter/limit are honored uniformly.
        let mem = MemTable::try_new(documents_schema(), vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// DataFusion table-valued function `documents(collection)` returning a [`DocumentTableProvider`] —
/// makes the document⋈relational/vector/timeseries/graph join expressible directly in SQL (and so
/// reachable from the pgwire DataFusion path):
/// `SELECT d.id FROM documents('orders') d JOIN vector_search('orders', $q, 10) v ON d.id = v.id`.
/// Register once per `SessionContext`:
/// `ctx.register_udtf("documents", Arc::new(DocumentsTableFunction::new(port)))`.
pub struct DocumentsTableFunction {
    scan: Arc<dyn DocumentScanPort>,
    /// The connection tenant, passed structurally to the scan port (which scopes the collection
    /// key) — NOT folded into the collection name. `None` ⇒ the canonical `DEFAULT_TENANT`.
    tenant: Option<String>,
    /// The canonical record route (TD-DOC-PUSHDOWN-1): when present AND the collection resolves
    /// `pax_scan_inputs`, `documents()` serves a storage-inclusive PAX-pruned scan (flushed
    /// segments + unflushed WAL delta + dead-filter/dedup) that also exposes the shredded
    /// promoted columns as typed columns. `None` ⇒ the `(id, props)` MemTable path only.
    record_route: Option<Arc<dyn proximadb_runtime::RecordRoutePort>>,
    /// The process filesystem factory used to read `.pax` segments for the pushdown scan.
    filesystem_factory: Option<Arc<crate::storage::persistence::filesystem::FilesystemFactory>>,
}

impl DocumentsTableFunction {
    /// Capture the scan port the function will read (no tenant scope, no PAX pushdown).
    pub fn new(scan: Arc<dyn DocumentScanPort>) -> Self {
        Self {
            scan,
            tenant: None,
            record_route: None,
            filesystem_factory: None,
        }
    }

    /// Capture the scan port AND the connection tenant, so the read scopes the collection key the
    /// same way the REST write path does (TD-XMODAL-6). No PAX pushdown (MemTable path only).
    pub fn with_tenant(scan: Arc<dyn DocumentScanPort>, tenant: Option<String>) -> Self {
        Self {
            scan,
            tenant,
            record_route: None,
            filesystem_factory: None,
        }
    }

    /// Build the function backed by the process document service, scoped to the connection tenant —
    /// the pgwire OLAP route wiring that makes `documents` read REST/gRPC-written data. Also wires
    /// the record route + filesystem factory so a converged collection with `.pax` segments is
    /// served through the correct storage-inclusive PAX pushdown provider (TD-DOC-PUSHDOWN-1),
    /// falling back to the `(id, props)` MemTable path when neither is available.
    pub fn from_service_with_tenant(
        service: Arc<crate::storage::document::DocumentService>,
        tenant: Option<String>,
    ) -> Self {
        let record_route = service.record_route_port();
        Self {
            scan: Arc::new(DocumentServiceScan(service)),
            tenant,
            record_route,
            filesystem_factory: crate::services::document_service::filesystem_factory(),
        }
    }

    /// Try to build the storage-inclusive PAX pushdown provider for `collection`. Returns `None`
    /// (⇒ MemTable fallback) when the record route / filesystem factory is unwired or the
    /// collection isn't PAX-pushdown-eligible (`pax_scan_inputs` ⇒ `None`). Resolving the scan
    /// inputs is async but `call()` is sync, so it runs on the current multi-thread runtime via
    /// `block_in_place` (the server runtime is multi-thread; tests use `flavor = "multi_thread"`).
    fn try_pax_provider(&self, collection: &str) -> Option<Arc<dyn TableProvider>> {
        let route = self.record_route.clone()?;
        let filesystem_factory = self.filesystem_factory.clone()?;
        let tenant = self.tenant.clone();
        let collection_owned = collection.to_string();
        let inputs = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::try_current().ok().and_then(|h| {
                h.block_on(route.pax_scan_inputs(&collection_owned, tenant.as_deref()))
            })
        })?;
        Some(Arc::new(
            crate::datafusion::document_pax_provider::DocumentPaxPushdownProvider::new(
                route,
                filesystem_factory,
                self.tenant.clone(),
                collection.to_string(),
                inputs,
            ),
        ))
    }
}

impl std::fmt::Debug for DocumentsTableFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentsTableFunction")
            .finish_non_exhaustive()
    }
}

impl TableFunctionImpl for DocumentsTableFunction {
    fn call(&self, args: &[Expr]) -> DFResult<Arc<dyn TableProvider>> {
        let collection = arg_string(args, 0).ok_or_else(|| {
            DataFusionError::Plan(
                "documents(collection): arg 1 must be a collection-name string".into(),
            )
        })?;
        if collection.trim().is_empty() {
            return Err(DataFusionError::Plan(
                "documents: collection must not be empty".into(),
            ));
        }
        // TD-DOC-PUSHDOWN-1: prefer the storage-inclusive PAX pushdown provider when the
        // collection is converged (has resolvable `pax_scan_inputs`). It reads the SAME rows the
        // MemTable path would (byte-identical) plus the shredded typed columns, pruning the
        // flushed segments off the wire. Any collection that isn't pushdown-eligible falls back to
        // the `(id, props)` MemTable path below — mixed-read-safe, no flag-day.
        if let Some(provider) = self.try_pax_provider(&collection) {
            return Ok(provider);
        }

        // Structural tenant scoping: pass the connection tenant (defaulting to the one canonical
        // DEFAULT_TENANT) + the tenant-CLEAN collection name to the scan port, which scopes the key
        // the same way the REST ingest wrote it. No name-folding at the SQL layer.
        let tenant = proximadb_tenant::resolve_request_tenant(self.tenant.as_deref());
        Ok(Arc::new(DocumentTableProvider::new(
            self.scan.clone(),
            tenant,
            collection,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;

    fn ov(id: &str, score: f32) -> crate::core::search::results::OptimizedSearchRecord {
        crate::core::search::results::OptimizedSearchRecord {
            id: id.to_string(),
            score,
            ..Default::default()
        }
    }

    #[test]
    fn vector_matches_batch_has_id_score_metadata_schema() {
        let batch = vector_matches_to_batch(&[ov("a", 0.9), ov("b", 0.5)]).unwrap();
        assert_eq!(batch.num_rows(), 2);
        // TD-XMODAL-4 S1: (id, score, metadata) — payload rides with id+score so a
        // standalone `SELECT * FROM vector_search(...)` needs no self-join.
        assert_eq!(batch.schema().field(0).name(), "id");
        assert_eq!(batch.schema().field(1).name(), "score");
        assert_eq!(
            batch.schema().field(1).data_type(),
            &DataType::Float64,
            "score is Float64 to match the pgwire <-> operator path"
        );
        assert_eq!(batch.schema().field(2).name(), "metadata");
        // Empty metadata (the sv helper) serializes to an empty JSON object.
        let meta = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(meta.value(0), "{}");
    }

    /// The moat proof: vector-search results JOIN a relational table in ONE
    /// DataFusion SQL plan (filter-by-similarity ⋈ relational), ordered by score.
    #[tokio::test]
    async fn vector_matches_join_relational_in_one_sql_plan() {
        let ctx = SessionContext::new();

        // Vector modality → joinable table (would come from the live VectorOpsPort
        // in the next slice; here we feed a fixed result set through the bridge).
        let matches = vector_matches_to_batch(&[ov("a", 0.95), ov("b", 0.80), ov("c", 0.70)])
            .expect("matches batch");
        ctx.register_table(
            "vmatches",
            Arc::new(MemTable::try_new(vector_matches_schema(), vec![vec![matches]]).unwrap()),
        )
        .unwrap();

        // Relational modality.
        let docs_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
        ]));
        let docs = RecordBatch::try_new(
            docs_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "z"])),
                Arc::new(StringArray::from(vec!["Alpha", "Bravo", "Zulu"])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "docs",
            Arc::new(MemTable::try_new(docs_schema, vec![vec![docs]]).unwrap()),
        )
        .unwrap();

        // One SQL plan joining vector similarity with relational rows.
        let df = ctx
            .sql(
                "SELECT d.id, d.title, m.score \
                 FROM docs d JOIN vmatches m ON d.id = m.id \
                 ORDER BY m.score DESC",
            )
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        // a(Alpha,0.95) + b(Bravo,0.80); c has no doc, z has no vector match.
        assert_eq!(rows, 2);
    }

    use crate::proto::proximadb_v1::{SearchResult, VectorBatchRequest, VectorOperationResponse};

    /// A fixed `VectorOpsPort` that returns a canned similarity result — stands in for
    /// the live vector service so the provider's `scan` path is exercised.
    struct FixedVectorOps {
        matches: Vec<(String, f64)>,
    }

    #[async_trait]
    impl proximadb_runtime::VectorOpsPort for FixedVectorOps {
        async fn search(
            &self,
            _request: VectorSearchRequest,
            _identity: proximadb_runtime::PortIdentity<'_>,
        ) -> anyhow::Result<VectorOperationResponse> {
            let results = self
                .matches
                .iter()
                .map(|(id, score)| SearchVectorRecord {
                    id: id.clone(),
                    score: *score,
                    ..Default::default()
                })
                .collect();
            Ok(VectorOperationResponse {
                results: Some(SearchResult {
                    results,
                    ..Default::default()
                }),
                ..Default::default()
            })
        }
        // TD-XMODAL-4 S2: the provider now scans via this kernel — return the fixed
        // matches as v2 OptimizedSearchRecords.
        async fn unified_search_native(
            &self,
            _search: &proximadb_vector_query::VectorSearchExpr,
            _identity: proximadb_runtime::PortIdentity<'_>,
        ) -> anyhow::Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
            Ok(self
                .matches
                .iter()
                .map(
                    |(id, score)| crate::core::search::results::OptimizedSearchRecord {
                        id: id.clone(),
                        score: *score as f32,
                        ..Default::default()
                    },
                )
                .collect())
        }
        async fn batch_upsert(
            &self,
            _r: VectorBatchRequest,
            _t: Option<&str>,
        ) -> anyhow::Result<VectorOperationResponse> {
            unimplemented!()
        }
        async fn get_vector(
            &self,
            _c: &str,
            _v: &str,
            _iv: bool,
            _im: bool,
            _t: Option<&str>,
        ) -> anyhow::Result<VectorOperationResponse> {
            unimplemented!()
        }
        async fn flush_all(&self) -> anyhow::Result<()> {
            Ok(())
        }
        async fn metrics(&self) -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
    }

    /// The live-backed moat: a `VectorSearchTableProvider` (running a search through a
    /// `VectorOpsPort`) registered as a table, scanned AND joined with relational data
    /// in one SQL plan.
    #[tokio::test]
    async fn vector_search_provider_scans_and_joins() {
        let ops: Arc<dyn proximadb_runtime::VectorOpsPort> = Arc::new(FixedVectorOps {
            matches: vec![("a".into(), 0.95), ("b".into(), 0.80), ("c".into(), 0.70)],
        });
        let provider = VectorSearchTableProvider::new(
            ops,
            VectorSearchExpr {
                collection: "docs_vec".to_string(),
                vector_column: None,
                query_vector: vec![0.1, 0.2, 0.3],
                top_k: 10,
                threshold: None,
                metric: proximadb_distance_types::DistanceMetric::L2,
                filter: None,
                params: VectorSearchParams::default(),
            },
            proximadb_runtime::OwnedPortIdentity::default(),
        );

        let ctx = SessionContext::new();
        ctx.register_table("vsearch", Arc::new(provider)).unwrap();

        // (1) Scan the live-backed provider directly.
        let scanned: usize = ctx
            .sql("SELECT id, score FROM vsearch ORDER BY score DESC")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(scanned, 3);

        // (2) Join it with a relational table in ONE plan.
        let docs_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
        ]));
        let docs = RecordBatch::try_new(
            docs_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "z"])),
                Arc::new(StringArray::from(vec!["Alpha", "Bravo", "Zulu"])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "docs",
            Arc::new(MemTable::try_new(docs_schema, vec![vec![docs]]).unwrap()),
        )
        .unwrap();
        let joined: usize = ctx
            .sql(
                "SELECT d.title, v.score FROM docs d JOIN vsearch v ON d.id = v.id \
                 ORDER BY v.score DESC",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(joined, 2); // a + b match docs; c has no doc, z has no match.
    }

    #[test]
    fn parses_pgvector_text_literal() {
        assert_eq!(
            parse_vector_literal("[0.1, 0.2, 0.3]"),
            Some(vec![0.1, 0.2, 0.3])
        );
        assert_eq!(parse_vector_literal("[]"), Some(vec![]));
        assert_eq!(parse_vector_literal("0.1,0.2"), None); // missing brackets
    }

    /// The customer-facing moat: a single SQL statement (via the `vector_search` UDTF)
    /// joins vector similarity with relational data — the shape a pgwire client writes.
    #[tokio::test]
    async fn vector_search_udtf_joins_relational_in_sql() {
        let ops: Arc<dyn proximadb_runtime::VectorOpsPort> = Arc::new(FixedVectorOps {
            matches: vec![("a".into(), 0.95), ("b".into(), 0.80), ("c".into(), 0.70)],
        });
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "vector_search",
            Arc::new(VectorSearchTableFunction::new(ops)),
        );

        let docs_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
        ]));
        let docs = RecordBatch::try_new(
            docs_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "z"])),
                Arc::new(StringArray::from(vec!["Alpha", "Bravo", "Zulu"])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "docs",
            Arc::new(MemTable::try_new(docs_schema, vec![vec![docs]]).unwrap()),
        )
        .unwrap();

        let n: usize = ctx
            .sql(
                "SELECT d.title, v.score \
                 FROM docs d JOIN vector_search('docs_vec', '[0.1,0.2,0.3]', 10) v \
                   ON d.id = v.id \
                 ORDER BY v.score DESC",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n, 2); // a + b
    }

    #[tokio::test]
    async fn vector_search_udtf_rejects_invalid_inputs() {
        let ops: Arc<dyn proximadb_runtime::VectorOpsPort> =
            Arc::new(FixedVectorOps { matches: vec![] });
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "vector_search",
            Arc::new(VectorSearchTableFunction::new(ops)),
        );

        for (sql, expected) in [
            (
                "SELECT * FROM vector_search('', '[0.1,0.2]', 10)",
                "collection must not be empty",
            ),
            (
                "SELECT * FROM vector_search('docs_vec', '[]', 10)",
                "query vector must contain at least one dimension",
            ),
            (
                "SELECT * FROM vector_search('docs_vec', '[0.1,0.2]', 0)",
                "top_k must be greater than zero",
            ),
            (
                "SELECT * FROM vector_search('docs_vec', '[NaN,0.2]', 10)",
                "cannot parse query vector",
            ),
            (
                "SELECT * FROM vector_search('docs_vec', '[0.1,0.2]', 4294967296)",
                "top_k exceeds the supported u32 range",
            ),
        ] {
            let error = ctx.sql(sql).await.expect_err("invalid vector_search");
            assert!(
                error.to_string().contains(expected),
                "expected {expected:?} in {error}"
            );
        }
    }

    #[tokio::test]
    async fn live_session_context_registers_vector_search() {
        // F4: the live session-context builder registers `vector_search` itself, so the
        // cross-modal table function is available over the DataFusion path WITHOUT a manual
        // register_udtf — this is exactly how the pgwire OLAP route wires it.
        let ops: Arc<dyn proximadb_runtime::VectorOpsPort> = Arc::new(FixedVectorOps {
            matches: vec![("a".into(), 0.95), ("b".into(), 0.80)],
        });
        let ctx = crate::datafusion::create_session_context_with_vector_ops(ops).unwrap();
        let n: usize = ctx
            .sql("SELECT id, score FROM vector_search('docs_vec', '[0.1,0.2]', 10)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n, 2);
    }

    // ── timeseries slice ────────────────────────────────────────────────────────
    use std::collections::HashMap;

    fn tp(timestamp: i64, metric: &str, value: f64) -> TsPoint {
        TsPoint {
            timestamp,
            values: HashMap::from([(metric.to_string(), value)]),
            tags: HashMap::new(),
        }
    }

    /// A fixed `TimeseriesScanPort` returning canned points — stands in for the process
    /// time-series service so the provider's `scan` path is exercised without engine state.
    struct FixedTimeseriesScan {
        points: Vec<TsPoint>,
    }

    #[async_trait]
    impl TimeseriesScanPort for FixedTimeseriesScan {
        async fn range(
            &self,
            _tenant: &str,
            _collection: &str,
            _start_ms: i64,
            _end_ms: i64,
        ) -> anyhow::Result<Vec<TsPoint>> {
            Ok(self.points.clone())
        }
    }

    #[test]
    fn timeseries_batch_has_timestamp_metric_value_schema() {
        let batch =
            timeseries_points_to_batch(&[tp(10, "amount", 1.5), tp(20, "amount", 2.5)]).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().field(0).name(), "timestamp");
        assert_eq!(batch.schema().field(1).name(), "metric");
        assert_eq!(batch.schema().field(2).name(), "value");
    }

    /// The moat proof for the second modality: a timeseries scan JOINs a relational table in
    /// ONE DataFusion SQL plan — the shape a pgwire client (or a copilot) writes instead of
    /// fanning out N per-entity reads and intersecting them client-side.
    #[tokio::test]
    async fn timeseries_range_udtf_joins_relational_in_sql() {
        let port: Arc<dyn TimeseriesScanPort> = Arc::new(FixedTimeseriesScan {
            points: vec![
                tp(1_000, "amount", 100.0),
                tp(2_000, "amount", 9_000.0),
                tp(3_000, "amount", 50.0),
            ],
        });
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "timeseries_range",
            Arc::new(TimeseriesRangeTableFunction::new(port)),
        );

        // Relational side: an entity dimension the timeseries joins onto by metric name.
        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let dim = RecordBatch::try_new(
            dim_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["amount"])),
                Arc::new(StringArray::from(vec!["claimed"])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "metrics",
            Arc::new(MemTable::try_new(dim_schema, vec![vec![dim]]).unwrap()),
        )
        .unwrap();

        // One SQL plan: the timeseries anomaly-scan (MAX per metric) ⋈ the relational dimension.
        let batches = ctx
            .sql(
                "SELECT m.label, MAX(t.value) AS peak \
                 FROM timeseries_range('ts_clmt_007', 0, 9999999) t \
                 JOIN metrics m ON t.metric = m.metric \
                 GROUP BY m.label",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1); // one metric group: 'claimed' with peak 9000
        let peak = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(peak, 9_000.0);
    }

    // ─── Document slice (ADR-055 P-DFSource) ────────────────────────────────

    /// Fixed `DocumentScanPort` returning canned `(id, props_json)` rows — stands in for the
    /// process document service so the `documents(...)` table source is unit-testable.
    struct FixedDocScan {
        rows: Vec<(String, String)>,
    }

    #[async_trait]
    impl DocumentScanPort for FixedDocScan {
        async fn scan(
            &self,
            _tenant: &str,
            _collection: &str,
        ) -> anyhow::Result<Vec<(String, String)>> {
            Ok(self.rows.clone())
        }
    }

    fn doc_rows() -> Vec<(String, String)> {
        vec![
            ("d1".to_string(), r#"{"status":"open","n":1}"#.to_string()),
            ("d2".to_string(), r#"{"status":"closed","n":2}"#.to_string()),
            ("d3".to_string(), r#"{"status":"open","n":3}"#.to_string()),
        ]
    }

    fn ids_of(batches: &[RecordBatch]) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            if let Some(col) = b.column(0).as_any().downcast_ref::<StringArray>() {
                for i in 0..b.num_rows() {
                    out.push(col.value(i).to_string());
                }
            }
        }
        out
    }

    #[test]
    fn document_batch_has_id_props_schema() {
        let batch = document_rows_to_batch(&doc_rows()).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.schema().field(0).name(), "id");
        assert_eq!(batch.schema().field(1).name(), "props");
    }

    /// Parity: `SELECT id FROM documents('coll')` returns EXACTLY the ids the scan port yields
    /// (i.e. what `query_documents` would return for the same set), and a `WHERE id = ...`
    /// predicate is honored by the MemTable delegation.
    #[tokio::test]
    async fn documents_udtf_select_id_matches_scan() {
        let port: Arc<dyn DocumentScanPort> = Arc::new(FixedDocScan { rows: doc_rows() });
        let ctx = SessionContext::new();
        ctx.register_udtf("documents", Arc::new(DocumentsTableFunction::new(port)));

        // All ids, in order — must equal the scan port's ids exactly.
        let all = ctx
            .sql("SELECT id FROM documents('coll') ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(ids_of(&all), vec!["d1", "d2", "d3"]);

        // Predicate pushed through the MemTable delegation.
        let one = ctx
            .sql("SELECT id FROM documents('coll') WHERE id = 'd2'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(ids_of(&one), vec!["d2"]);
    }

    /// The document slice's moat proof: `documents(...)` joins a relational table in ONE SQL plan.
    #[tokio::test]
    async fn documents_udtf_joins_relational_in_sql() {
        let port: Arc<dyn DocumentScanPort> = Arc::new(FixedDocScan { rows: doc_rows() });
        let ctx = SessionContext::new();
        ctx.register_udtf("documents", Arc::new(DocumentsTableFunction::new(port)));

        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let dim = RecordBatch::try_new(
            dim_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["d1", "d3"])),
                Arc::new(StringArray::from(vec!["A", "B"])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "dim",
            Arc::new(MemTable::try_new(dim_schema, vec![vec![dim]]).unwrap()),
        )
        .unwrap();

        let batches = ctx
            .sql("SELECT d.id FROM documents('coll') d JOIN dim ON d.id = dim.id ORDER BY d.id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(ids_of(&batches), vec!["d1", "d3"]);
    }

    #[tokio::test]
    async fn documents_udtf_rejects_empty_collection() {
        let port: Arc<dyn DocumentScanPort> = Arc::new(FixedDocScan { rows: vec![] });
        let ctx = SessionContext::new();
        ctx.register_udtf("documents", Arc::new(DocumentsTableFunction::new(port)));
        assert!(
            ctx.sql("SELECT id FROM documents('')").await.is_err(),
            "empty collection name must be rejected"
        );
    }

    #[tokio::test]
    async fn timeseries_range_udtf_rejects_invalid_inputs() {
        let port: Arc<dyn TimeseriesScanPort> = Arc::new(FixedTimeseriesScan { points: vec![] });
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "timeseries_range",
            Arc::new(TimeseriesRangeTableFunction::new(port)),
        );

        for (sql, expected) in [
            (
                "SELECT * FROM timeseries_range('', 0, 10)",
                "collection must not be empty",
            ),
            (
                "SELECT * FROM timeseries_range('ts', 10, 0)",
                "end_ms must be >= start_ms",
            ),
        ] {
            let error = ctx.sql(sql).await.expect_err("invalid timeseries_range");
            assert!(
                error.to_string().contains(expected),
                "expected {expected:?} in {error}"
            );
        }
    }

    /// TD-XMODAL-7 repro: a UDTF wrapped in an aggregating derived table that is then JOINed with
    /// a base table (`FROM base o CROSS JOIN (SELECT max(value) FROM timeseries_range(...)) ts`)
    /// must not blow the stack. A top-level UDTF (even two joined) plans fine; only the
    /// join-to-a-derived-table-over-a-UDTF shape recursed over pgwire's DataFusion fallback.
    #[tokio::test]
    async fn udtf_in_aggregating_subquery_does_not_recurse() {
        let port: Arc<dyn TimeseriesScanPort> = Arc::new(FixedTimeseriesScan {
            points: vec![tp(1, "amount", 5.0), tp(2, "amount", 9.0)],
        });
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "timeseries_range",
            Arc::new(TimeseriesRangeTableFunction::new(port)),
        );
        // A base relation to join against, mirroring the pgwire `orders` table.
        let base_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let base = RecordBatch::try_new(
            base_schema.clone(),
            vec![Arc::new(StringArray::from(vec!["acct-7"]))],
        )
        .unwrap();
        ctx.register_table(
            "orders",
            Arc::new(MemTable::try_new(base_schema, vec![vec![base]]).unwrap()),
        )
        .unwrap();
        let n: usize = ctx
            .sql(
                "SELECT o.id, ts.peak \
                 FROM orders o \
                 CROSS JOIN (SELECT max(value) AS peak FROM timeseries_range('c', 0, 9)) ts \
                 WHERE o.id = 'acct-7' \
                 GROUP BY o.id, ts.peak",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n, 1);
    }

    /// A `TimeseriesScanPort` that records the `(tenant, collection)` it was asked to read — used
    /// to prove `timeseries_range` passes the tenant STRUCTURALLY and keeps the collection name
    /// tenant-clean (no `{tenant}::` folding).
    struct RecordingTimeseriesScan {
        seen: Arc<std::sync::Mutex<Vec<(String, String)>>>,
    }

    #[async_trait]
    impl TimeseriesScanPort for RecordingTimeseriesScan {
        async fn range(
            &self,
            tenant: &str,
            collection: &str,
            _start_ms: i64,
            _end_ms: i64,
        ) -> anyhow::Result<Vec<TsPoint>> {
            self.seen
                .lock()
                .unwrap()
                .push((tenant.to_string(), collection.to_string()));
            Ok(vec![tp(1_000, "amount", 1.0)])
        }
    }

    /// Structural isolation: `timeseries_range` passes the connection tenant to the scan port
    /// (which selects the tenant's engine) and keeps the collection name tenant-CLEAN — no
    /// `{tenant}::` folding. An empty/absent tenant resolves to the one canonical `DEFAULT_TENANT`.
    #[tokio::test]
    async fn timeseries_range_udtf_passes_tenant_structurally_with_clean_name() {
        for (tenant, expected_tenant) in [
            (Some("acme".to_string()), "acme"),
            (Some(String::new()), "default"),
            (None, "default"),
        ] {
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let port: Arc<dyn TimeseriesScanPort> = Arc::new(RecordingTimeseriesScan {
                seen: Arc::clone(&seen),
            });
            let ctx = SessionContext::new();
            ctx.register_udtf(
                "timeseries_range",
                Arc::new(TimeseriesRangeTableFunction::with_tenant(port, tenant)),
            );
            let _ = ctx
                .sql("SELECT * FROM timeseries_range('sensor', 0, 9999)")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            // The collection stays "sensor" (no `::`); the tenant is passed as a distinct arg.
            assert_eq!(
                seen.lock().unwrap().as_slice(),
                &[(expected_tenant.to_string(), "sensor".to_string())]
            );
        }
    }

    /// A `VectorOpsPort` that records the complete identity at its search seam.
    struct RecordingVectorOps {
        seen: Arc<std::sync::Mutex<Vec<proximadb_runtime::OwnedPortIdentity>>>,
        seen_intents: Arc<std::sync::Mutex<Vec<proximadb_vector_query::VectorSearchExpr>>>,
        seen_metrics: Arc<std::sync::Mutex<Vec<Option<proximadb_distance_types::DistanceMetric>>>>,
    }

    #[async_trait]
    impl proximadb_runtime::VectorOpsPort for RecordingVectorOps {
        async fn search(
            &self,
            _request: VectorSearchRequest,
            identity: proximadb_runtime::PortIdentity<'_>,
        ) -> anyhow::Result<VectorOperationResponse> {
            self.seen.lock().unwrap().push(identity.into_owned());
            Ok(VectorOperationResponse {
                results: Some(SearchResult {
                    results: vec![],
                    ..Default::default()
                }),
                ..Default::default()
            })
        }
        // TD-XMODAL-4 S2: the UDTF forwards identity through THIS kernel.
        async fn unified_search_native(
            &self,
            search: &proximadb_vector_query::VectorSearchExpr,
            identity: proximadb_runtime::PortIdentity<'_>,
        ) -> anyhow::Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
            self.seen.lock().unwrap().push(identity.into_owned());
            self.seen_intents.lock().unwrap().push(search.clone());
            self.seen_metrics.lock().unwrap().push(Some(search.metric));
            Ok(vec![])
        }
        async fn batch_upsert(
            &self,
            _r: VectorBatchRequest,
            _t: Option<&str>,
        ) -> anyhow::Result<VectorOperationResponse> {
            unimplemented!()
        }
        async fn get_vector(
            &self,
            _c: &str,
            _v: &str,
            _iv: bool,
            _im: bool,
            _t: Option<&str>,
        ) -> anyhow::Result<VectorOperationResponse> {
            unimplemented!()
        }
        async fn flush_all(&self) -> anyhow::Result<()> {
            Ok(())
        }
        async fn metrics(&self) -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
    }

    /// TD-XMODAL-6 / ADR-087: `vector_search` forwards one complete identity;
    /// tenant scoping and ABAC subject resolution must not diverge by SQL syntax.
    #[tokio::test]
    async fn vector_search_udtf_forwards_tenant_to_search() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_intents = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_metrics = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ops: Arc<dyn proximadb_runtime::VectorOpsPort> = Arc::new(RecordingVectorOps {
            seen: Arc::clone(&seen),
            seen_intents: Arc::clone(&seen_intents),
            seen_metrics: Arc::clone(&seen_metrics),
        });
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "vector_search",
            Arc::new(VectorSearchTableFunction::with_identity(
                ops,
                proximadb_runtime::OwnedPortIdentity {
                    tenant_id: Some("acme".to_string()),
                    subject: Some("alice".to_string()),
                    tenant_stable_id: Some(7),
                    auth_class: proximadb_tenant::AuthClass::Authenticated,
                },
            )),
        );
        let _ = ctx
            .sql("SELECT * FROM vector_search('docs_vec', '[0.1,0.2]', 5)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[proximadb_runtime::OwnedPortIdentity {
                tenant_id: Some("acme".to_string()),
                subject: Some("alice".to_string()),
                tenant_stable_id: Some(7),
                auth_class: proximadb_tenant::AuthClass::Authenticated,
            }]
        );
        assert_eq!(
            seen_metrics.lock().unwrap().as_slice(),
            &[Some(proximadb_distance_types::DistanceMetric::L2)],
            "the UDTF must not discard the metric carried by its canonical vector intent"
        );
        let intents = seen_intents.lock().unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].collection, "docs_vec");
        assert_eq!(intents[0].query_vector, vec![0.1, 0.2]);
        assert_eq!(intents[0].top_k, 5);
        assert_eq!(intents[0].threshold, None);
        assert_eq!(intents[0].vector_column, None);
        assert_eq!(intents[0].filter, None);
        assert_eq!(intents[0].params, VectorSearchParams::default());
    }

    // ── graph slice ───────────────────────────────────────────────────────────────
    use crate::proto::proximadb_v1::{Edge, EdgeQuery, Node, NodeQuery};
    use proximadb_graph_query::service::GraphQueryResult;

    /// A fixed in-memory graph read service — a linear chain over one edge type, enough to
    /// exercise the variable-length `*k..k` traversal the `graph_traverse` provider runs.
    #[derive(Default)]
    struct FixedGraphOps {
        nodes: HashMap<String, Arc<Node>>,
        edges: Vec<Arc<Edge>>,
    }

    impl FixedGraphOps {
        fn chain(ids: &[&str], edge_type: &str) -> Self {
            let mut s = Self::default();
            for id in ids {
                s.nodes.insert(
                    (*id).to_string(),
                    Arc::new(Node {
                        id: (*id).to_string(),
                        labels: vec!["N".to_string()],
                        ..Default::default()
                    }),
                );
            }
            for w in ids.windows(2) {
                s.edges.push(Arc::new(Edge {
                    id: format!("{}->{}", w[0], w[1]),
                    from_node_id: w[0].to_string(),
                    to_node_id: w[1].to_string(),
                    edge_type: edge_type.to_string(),
                    ..Default::default()
                }));
            }
            s
        }
    }

    #[async_trait]
    impl GraphQueryReadService for FixedGraphOps {
        async fn list_graphs(&self) -> GraphQueryResult<Vec<String>> {
            Ok(vec!["g".to_string()])
        }
        async fn get_node(&self, _g: &str, id: &str) -> GraphQueryResult<Option<Arc<Node>>> {
            Ok(self.nodes.get(id).cloned())
        }
        async fn query_nodes(&self, _g: &str, _q: NodeQuery) -> GraphQueryResult<Vec<Arc<Node>>> {
            Ok(self.nodes.values().cloned().collect())
        }
        async fn query_edges(&self, _g: &str, q: EdgeQuery) -> GraphQueryResult<Vec<Arc<Edge>>> {
            Ok(self
                .edges
                .iter()
                .filter(|e| {
                    q.from_node_id
                        .as_deref()
                        .is_none_or(|f| e.from_node_id == f)
                        && (q.edge_types.is_empty()
                            || q.edge_types.iter().any(|t| t == &e.edge_type))
                })
                .cloned()
                .collect())
        }
    }

    /// The moat proof for the third modality: a graph reachability set JOINs a relational table
    /// in ONE DataFusion SQL plan — the shape a pgwire client writes for `relational ⋈ graph`.
    #[tokio::test]
    async fn graph_traverse_udtf_joins_relational_in_sql() {
        // n0 -> n1 -> n2 -> n3 over LINK.
        let graph: Arc<dyn GraphQueryReadService> =
            Arc::new(FixedGraphOps::chain(&["n0", "n1", "n2", "n3"], "LINK"));
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "graph_traverse",
            Arc::new(GraphTraverseTableFunction::new(graph)),
        );

        // `*1..3` from n0 reaches n1 (d1), n2 (d2), n3 (d3).
        let rows: usize = ctx
            .sql("SELECT node_id, depth FROM graph_traverse('g','n0','LINK',3) ORDER BY depth")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 3);

        // Join reachability with a relational dimension.
        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("tier", DataType::Utf8, false),
        ]));
        let dim = RecordBatch::try_new(
            dim_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["n1", "n3", "z"])),
                Arc::new(StringArray::from(vec!["hot", "hot", "cold"])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "acct",
            Arc::new(MemTable::try_new(dim_schema, vec![vec![dim]]).unwrap()),
        )
        .unwrap();
        let joined: usize = ctx
            .sql(
                "SELECT a.tier, g.depth FROM acct a \
                 JOIN graph_traverse('g','n0','LINK',4) g ON a.node_id = g.node_id",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(joined, 2); // n1 + n3 reachable & in acct; z has no reachable node
    }

    #[tokio::test]
    async fn graph_traverse_udtf_rejects_invalid_inputs() {
        let graph: Arc<dyn GraphQueryReadService> = Arc::new(FixedGraphOps::default());
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "graph_traverse",
            Arc::new(GraphTraverseTableFunction::new(graph)),
        );

        for (sql, expected) in [
            (
                "SELECT * FROM graph_traverse('','n0','LINK',3)",
                "graph_id must not be empty",
            ),
            (
                "SELECT * FROM graph_traverse('g','','LINK',3)",
                "start_id must not be empty",
            ),
            (
                "SELECT * FROM graph_traverse('g','n0','bad edge',3)",
                "edge_type must be a bare identifier",
            ),
            (
                "SELECT * FROM graph_traverse('g','n0','LINK',0)",
                "max_depth must be greater than zero",
            ),
        ] {
            let error = ctx.sql(sql).await.expect_err("invalid graph_traverse");
            assert!(
                error.to_string().contains(expected),
                "expected {expected:?} in {error}"
            );
        }
    }

    /// Records the graph id each read is issued against — so we can assert the `graph_traverse`
    /// UDTF composes the SAME `{tenant}/{graph_id}` structural scope the graph write path uses.
    #[derive(Default)]
    struct RecordingGraphOps {
        seen_graph_ids: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl GraphQueryReadService for RecordingGraphOps {
        async fn list_graphs(&self) -> GraphQueryResult<Vec<String>> {
            Ok(Vec::new())
        }
        async fn get_node(&self, g: &str, _id: &str) -> GraphQueryResult<Option<Arc<Node>>> {
            self.seen_graph_ids.lock().unwrap().push(g.to_string());
            Ok(None)
        }
        async fn query_nodes(&self, g: &str, _q: NodeQuery) -> GraphQueryResult<Vec<Arc<Node>>> {
            self.seen_graph_ids.lock().unwrap().push(g.to_string());
            Ok(Vec::new())
        }
        async fn query_edges(&self, g: &str, _q: EdgeQuery) -> GraphQueryResult<Vec<Arc<Edge>>> {
            self.seen_graph_ids.lock().unwrap().push(g.to_string());
            Ok(Vec::new())
        }
    }

    /// TD-XMODAL-6 (graph modality): a tenant-scoped `graph_traverse` reads the SAME
    /// `{tenant}/{graph_id}` structural scope the graph write path (`TenantGraphOps`) composes —
    /// never the raw `graph_id` — so a cross-modal traverse sees exactly the tenant's graph and
    /// the composed scope carries no `::` fold.
    #[tokio::test]
    async fn graph_traverse_udtf_scopes_read_by_tenant() {
        let ops = Arc::new(RecordingGraphOps::default());
        let ops_dyn: Arc<dyn GraphQueryReadService> = ops.clone();
        let ctx = SessionContext::new();
        ctx.register_udtf(
            "graph_traverse",
            Arc::new(GraphTraverseTableFunction::with_tenant(
                ops_dyn,
                Some("tenantA".to_string()),
            )),
        );
        // The logical graph_id in SQL is tenant-CLEAN ("shared"); the tenant is applied by the UDTF.
        let _ = ctx
            .sql("SELECT node_id, depth FROM graph_traverse('shared','n0','LINK',1)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let seen = ops.seen_graph_ids.lock().unwrap();
        assert!(!seen.is_empty(), "the read path was exercised");
        assert!(
            seen.iter().all(|g| g == "tenantA/shared"),
            "every graph read is scoped to the tenant: {seen:?}"
        );
        assert!(
            seen.iter().all(|g| !g.contains("::")),
            "structural scope carries no `::` fold: {seen:?}"
        );
    }
}
