//! Unified multi-model query port for `proximadb-runtime`.
//!
//! `UnifiedQueryPort` is the stable seam between REST/gRPC protocol adapters
//! and the root-crate `QueryFacadeAdapter` + `FederatedQueryContext`.  It uses
//! `serde_json::Value` for query payloads and results because no dedicated proto
//! message types exist yet for unified/federated queries.
//!
//! **Phase 9.9 blocker**: the root-crate implementation of this trait depends on
//! `CollectionService`, `DocumentService`, `ObservabilityService`, and
//! `QueryFacadeAdapter` being fully extracted.  Until then, `proximadb-api`
//! handlers return `501 Not Implemented` for all methods.

use anyhow::Result;
use async_trait::async_trait;
use proximadb_data_model::ProximaValue;

/// A request reached the unified query port but its client-supplied query
/// shape or values are invalid. Protocol adapters map this error to their
/// invalid-argument status instead of reporting an internal server failure.
#[derive(Debug)]
pub struct InvalidQueryInput(pub String);

impl std::fmt::Display for InvalidQueryInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InvalidQueryInput {}

/// Port for cross-model unified and federated query execution.
///
/// Implemented by root-crate services that own `QueryFacadeAdapter`.  Injected
/// into `proximadb-api` as `Arc<dyn UnifiedQueryPort>` so the API layer carries
/// zero dependency on root-crate concrete types.
#[async_trait]
pub trait UnifiedQueryPort: Send + Sync {
    // ── Core query execution ──────────────────────────────────────────────────

    /// Execute a unified SQL-like query (may contain multi-model extensions).
    ///
    /// Routes through `QueryFacadeAdapter::federated_query()` when the adapter
    /// is configured; falls back to the internal decomposer/executor pipeline.
    async fn execute_unified_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        limit: Option<u32>,
    ) -> Result<serde_json::Value>;

    /// Execute a multi-model structured query.
    ///
    /// Accepts a JSON object with optional `vector`, `graph`, `document`, and
    /// `observability` sub-queries plus a `fusion_strategy` field.
    async fn execute_multi_model_query(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value>;

    /// Execute a federated SQL query with multi-model extensions such as
    /// `VECTOR_SEARCH(...)`, `GRAPH_QUERY(...)`, and `DOCUMENT_QUERY(...)`.
    async fn execute_federated_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
    ) -> Result<serde_json::Value>;

    /// Execute a distributed query across shards or remote nodes.
    async fn execute_distributed_query(
        &self,
        request: serde_json::Value,
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
    ) -> Result<serde_json::Value>;

    // ── Prepared statements ───────────────────────────────────────────────────

    /// Parse and cache a prepared statement; returns the statement ID.
    async fn prepare_statement(
        &self,
        name: Option<String>,
        query: String,
        cache_results: bool,
        ttl_seconds: Option<u64>,
    ) -> Result<String>;

    /// Execute a previously prepared statement with the given parameter bindings.
    async fn execute_prepared(
        &self,
        statement_id: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
    ) -> Result<serde_json::Value>;

    /// Delete a previously prepared statement.
    async fn delete_prepared(&self, statement_id: String) -> Result<()>;

    /// Return cache/execution statistics for the given statement IDs.
    async fn get_prepared_stats(&self, statement_ids: Vec<String>) -> Result<serde_json::Value>;
}
