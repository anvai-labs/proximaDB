//! Unified multi-model query port for `proximadb-runtime`.
//!
//! `UnifiedQueryPort` is the stable seam between REST/gRPC protocol adapters
//! and the root-crate `QueryFacadeAdapter` + `FederatedQueryContext`.  It uses
//! `serde_json::Value` for query payloads and results because no dedicated proto
//! message types exist yet for unified/federated queries.
//!
//! **One decision path (ADR-098 D5 / TD-SPINE-6):** every SQL-bearing entry on
//! this port carries an explicit [`UnifiedQueryPortContext`] — the caller's
//! identity as *data* plus the requested freshness — and the implementation
//! delegates plain SQL to the same `ComputeScheduler` decision path that serves
//! pgwire and `api_handlers.execute_sql`. The transport-edge task-local scope
//! remains only as a fallback for callers that cannot construct an identity.
//!
//! **Phase 9.9 blocker**: the root-crate implementation of this trait depends on
//! `CollectionService`, `DocumentService`, `ObservabilityService`, and
//! `QueryFacadeAdapter` being fully extracted.  Until then, `proximadb-api`
//! handlers return `501 Not Implemented` for all methods.

use anyhow::Result;
use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_search_types::search_params::VectorFreshnessMode;

use crate::service_ports::PortIdentity;

/// A request reached the unified query port but its client-supplied query
/// shape or values are invalid. Protocol adapters map this error to their
/// invalid-argument status instead of reporting an internal server failure.
#[derive(Debug)]
pub struct InvalidQueryInput(pub String);

/// A query reached the unified query port with neither an explicit identity
/// tenant nor an in-force request scope (TD-SPINE-6 fail-closed gate).
/// Protocol adapters map this to their unauthorized status (401): the caller
/// did not establish who is asking.
#[derive(Debug)]
pub struct MissingQueryIdentity(pub &'static str);

/// A unified-query-port execution exceeded the port's deadline
/// ([`PORT_QUERY_DEADLINE`]). Protocol adapters map this to their
/// deadline/gateway-timeout status (HTTP 504). The deadline stops WAITING for
/// the execution (the future is dropped); work already handed to spawned or
/// blocking tasks is not cancelled by it.
#[derive(Debug)]
pub struct QueryDeadlineExceeded(pub std::time::Duration);

/// The deadline every `UnifiedQueryPort` SQL execution runs under — the same
/// default as `/api/v2/sql` (TD-SPINE-6 round 3, L3). Enforced inside the
/// port, so every surface over it (REST v2 query, `/api/v2/unified/*`,
/// canonical `/sql`, MCP) carries it.
pub const PORT_QUERY_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);

/// Explicit per-request execution context for `UnifiedQueryPort` calls
/// (ADR-098 D5 / TD-SPINE-6).
///
/// * `identity` — the caller's identity travels as **data** on the call
///   (tenant scope + ABAC principal) instead of relying on the transport-edge
///   task-local scope. `tenant_id: None` means "no identity supplied"; the
///   implementation then falls back to the task-local request scope and fails
///   closed when neither is present.
/// * `freshness` — honored at every result cache the delegated decision path
///   consults: `Strong` bypasses the OLAP result-cache lookup entirely
///   (CLAUDE.md mandate #16c — gate the lookup on `!Strong`); weaker modes may
///   consult it under the existing LSN-pinned eligibility, which is never
///   staler than the requested mode.
#[derive(Debug, Clone)]
pub struct UnifiedQueryPortContext<'a> {
    pub identity: PortIdentity<'a>,
    pub freshness: VectorFreshnessMode,
}

impl<'a> UnifiedQueryPortContext<'a> {
    /// The caller's identity with the default (Strong) freshness.
    pub fn for_identity(identity: PortIdentity<'a>) -> Self {
        Self {
            identity,
            freshness: VectorFreshnessMode::default(),
        }
    }

    /// Override the freshness mode (identity unchanged).
    pub fn with_freshness(mut self, freshness: VectorFreshnessMode) -> Self {
        self.freshness = freshness;
        self
    }
}

impl std::fmt::Display for InvalidQueryInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InvalidQueryInput {}

impl std::fmt::Display for MissingQueryIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for MissingQueryIdentity {}

impl std::fmt::Display for QueryDeadlineExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "query exceeded the {:?} deadline", self.0)
    }
}

impl std::error::Error for QueryDeadlineExceeded {}

/// Port for cross-model unified and federated query execution.
///
/// Implemented by root-crate services that own `QueryFacadeAdapter`.  Injected
/// into `proximadb-api` as `Arc<dyn UnifiedQueryPort>` so the API layer carries
/// zero dependency on root-crate concrete types.
///
/// **Read-only contract (TD-SPINE-6 round 2, dated 2026-11-25):** plain SQL
/// on this port is READ-ONLY. Raw user text is classified on the statement
/// AST (a single pure `SELECT` — no data-modifying set-expression bodies, no
/// data-modifying CTEs, no `SELECT … INTO`; EXPLAIN is not served) and
/// executes through the authority's READ-ONLY entry, which structurally
/// cannot reach the EXPLAIN-DML explainer or the relational write dispatch.
/// Writes are served by `/api/v2/sql` (guarded), gRPC `ExecuteQuery`, and
/// pgwire. Port-lowered (UQL / multi-model) extension SQL keeps its federated
/// lowering. Query bodies are capped at 1 MiB and executions run under
/// [`PORT_QUERY_DEADLINE`]; rejections are typed (`InvalidQueryInput` → 400,
/// `MissingQueryIdentity` → 401, `QueryDeadlineExceeded` → 504).
#[async_trait]
pub trait UnifiedQueryPort: Send + Sync {
    // ── Core query execution ──────────────────────────────────────────────────

    /// Execute a unified SQL-like query (may contain multi-model extensions).
    ///
    /// Plain relational SQL delegates to the one SQL authority's read-only
    /// entry (`ComputeScheduler::route_select` decision path — ADR-098 D5 /
    /// TD-SPINE-6); multi-model extensions keep their federated lowering.
    async fn execute_unified_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        limit: Option<u32>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;

    /// Execute a multi-model structured query.
    ///
    /// Accepts a JSON object with optional `vector`, `graph`, `document`, and
    /// `observability` sub-queries plus a `fusion_strategy` field.
    async fn execute_multi_model_query(
        &self,
        request: serde_json::Value,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;

    /// Execute a federated SQL query with multi-model extensions such as
    /// `VECTOR_SEARCH(...)`, `GRAPH_QUERY(...)`, and `DOCUMENT_QUERY(...)`.
    async fn execute_federated_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;

    /// Execute a distributed query across shards or remote nodes.
    async fn execute_distributed_query(
        &self,
        request: serde_json::Value,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;

    // ── Query explanation ─────────────────────────────────────────────────────

    /// Explain the execution plan for a unified query.
    ///
    /// Returns a JSON representation of the query plan including estimated costs
    /// and storage authority decisions.
    async fn explain_unified_query(
        &self,
        query: String,
        collection: Option<String>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;

    // ── Prepared statements ───────────────────────────────────────────────────

    /// Parse and cache a prepared statement; returns the statement ID.
    async fn prepare_statement(
        &self,
        name: Option<String>,
        query: String,
        cache_results: bool,
        ttl_seconds: Option<u64>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<String>;

    /// Execute a previously prepared statement with the given parameter bindings.
    async fn execute_prepared(
        &self,
        statement_id: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;

    /// Delete a previously prepared statement.
    async fn delete_prepared(
        &self,
        statement_id: String,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<()>;

    /// Return cache/execution statistics for the given statement IDs.
    async fn get_prepared_stats(
        &self,
        statement_ids: Vec<String>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value>;
}
