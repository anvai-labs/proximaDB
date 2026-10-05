//! Service composition port traits for `proximadb-runtime`.
//!
//! These traits define the stable contracts that `UnifiedHandlers` uses to call into
//! the concrete service layer (collection management, vector operations, query routing).
//! They use only proto types and standard Rust types so `proximadb-runtime` can express
//! the full composition surface without importing any root-crate concrete types.
//!
//! ## Tenant contract
//!
//! Every tenant-sensitive method accepts `tenant_id: Option<&str>`.  The implementing
//! service is responsible for resolving that string to its internal `TenantContext` and
//! enforcing access control.  No tenant context type leaks through the port boundary.

use crate::port::CollectionSchemaColumn;
use anyhow::Result;
use async_trait::async_trait;
use proximadb_proto::v1::{
    Collection, CollectionConfig, HybridSearchRequest, HybridSearchResponse, SqlValue,
    VectorBatchRequest, VectorOperationResponse, VectorSearchRequest,
};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};

// ── Collection management ──────────────────────────────────────────────────────

/// Port for collection lifecycle operations.
///
/// Implemented by root-crate `CollectionService`.  Each method receives an optional
/// `tenant_id`; the implementation resolves tenant context and enforces isolation
/// internally so no tenant type crosses the port boundary.
#[async_trait]
pub trait CollectionPort: Send + Sync {
    /// Fetch a single collection by name or ID, honouring tenant scope.
    async fn get_collection(
        &self,
        identifier: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<Collection>>;

    /// Create a collection from a proto config, honouring tenant scope.
    async fn create_collection(
        &self,
        config: CollectionConfig,
        tenant_id: Option<&str>,
    ) -> Result<Collection>;

    /// Update collection configuration, honouring tenant scope.
    async fn update_collection(
        &self,
        id: &str,
        config: CollectionConfig,
        tenant_id: Option<&str>,
    ) -> Result<Collection>;

    /// Delete a collection, honouring tenant scope. Returns true if deleted.
    async fn delete_collection(&self, id: &str, tenant_id: Option<&str>) -> Result<bool>;

    /// List all collections visible to the given tenant (or all if tenant_id is None).
    async fn list_collections(&self, tenant_id: Option<&str>) -> Result<Vec<Collection>>;

    /// Resolve a collection name or UUID to its canonical internal ID.
    async fn resolve_collection_id(&self, identifier: &str) -> Result<Option<String>>;

    /// Persist the canonical ProximaType columns for a collection (ADR-047 /
    /// TD-TBL-1 authority). The narrow v1 `CollectionConfig` cannot represent
    /// the full `ProximaType` vocabulary, so these are stored as a catalog-asset
    /// sidecar. An empty slice clears them. The default no-op keeps ports without
    /// a catalog (mocks, `NoopCollectionPort`) compiling; `CollectionService`
    /// overrides to actually persist.
    async fn set_collection_schema_columns(
        &self,
        _id: &str,
        _columns: &[CollectionSchemaColumn],
        _tenant_id: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }

    /// Read back the canonical ProximaType columns, or `None` when the collection
    /// has none (legacy collection → caller falls back to the narrow-derived view).
    /// Default no-op returns `None`.
    async fn get_collection_schema_columns(
        &self,
        _id: &str,
        _tenant_id: Option<&str>,
    ) -> Result<Option<Vec<CollectionSchemaColumn>>> {
        Ok(None)
    }
}

// ── Vector operations ─────────────────────────────────────────────────────────

/// The pre-resolution caller identity carried across port seams (TD-ABAC-7).
///
/// This is the RAW identity captured at the network boundary — not an
/// authorization result (`AuthorizedReadContext` in `proximadb-abac` is what
/// the enforcement seam *resolves from* it). Fields:
///
/// * `tenant_id` — the authoritative ADR-0083 string identity, used for
///   tenant-scoped catalog/name resolution and `DrPathBuilder` paths.
/// * `subject` — the authenticated principal (ABAC); `None` = unauthenticated
///   / internal caller, which the seam treats as passthrough (no policy
///   evaluation).
/// * `tenant_stable_id` — the derived numeric projection of `tenant_id`
///   (widened `account_u32`), the ABAC policy-lookup key. Never a second
///   source of truth: the string stays authoritative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PortIdentity<'a> {
    pub tenant_id: Option<&'a str>,
    pub subject: Option<&'a str>,
    pub tenant_stable_id: Option<u64>,
    /// How the identity was established (ADR-087) — audit-only at the seam:
    /// enforcement stays deny-biased for any present subject regardless of
    /// class; provenance-conditional policy is a future knob.
    pub auth_class: proximadb_tenant::AuthClass,
}

/// Owned form of [`PortIdentity`] for planners/readers that must retain the
/// identity after the protocol call frame. Convert back with
/// [`Self::as_borrowed`] at the enforcement seam; fields remain one canonical
/// carrier rather than parallel tenant/subject parameters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OwnedPortIdentity {
    pub tenant_id: Option<String>,
    pub subject: Option<String>,
    pub tenant_stable_id: Option<u64>,
    pub auth_class: proximadb_tenant::AuthClass,
}

impl OwnedPortIdentity {
    pub fn as_borrowed(&self) -> PortIdentity<'_> {
        PortIdentity {
            tenant_id: self.tenant_id.as_deref(),
            subject: self.subject.as_deref(),
            tenant_stable_id: self.tenant_stable_id,
            auth_class: self.auth_class,
        }
    }
}

impl<'a> PortIdentity<'a> {
    /// No identity at all — internal/system callers; enforcement passthrough.
    pub const fn anonymous() -> Self {
        Self {
            tenant_id: None,
            subject: None,
            tenant_stable_id: None,
            auth_class: proximadb_tenant::AuthClass::Anonymous,
        }
    }

    /// Tenant-scoped but with no authenticated ABAC principal.
    pub const fn for_tenant(tenant_id: &'a str) -> Self {
        Self {
            tenant_id: Some(tenant_id),
            subject: None,
            tenant_stable_id: None,
            auth_class: proximadb_tenant::AuthClass::Anonymous,
        }
    }

    pub fn into_owned(self) -> OwnedPortIdentity {
        OwnedPortIdentity {
            tenant_id: self.tenant_id.map(str::to_string),
            subject: self.subject.map(str::to_string),
            tenant_stable_id: self.tenant_stable_id,
            auth_class: self.auth_class,
        }
    }
}

/// ADR-087: the borrowed port-seam projection of the one foundation identity.
impl<'a> From<&'a proximadb_tenant::ResolvedRequestIdentity> for PortIdentity<'a> {
    fn from(identity: &'a proximadb_tenant::ResolvedRequestIdentity) -> Self {
        Self {
            tenant_id: Some(identity.tenant.as_str()),
            subject: identity.subject.as_deref(),
            tenant_stable_id: identity.tenant_stable_id,
            auth_class: identity.auth_class,
        }
    }
}

/// Port for vector CRUD and search operations.
///
/// Implemented by root-crate `VectorOperationsService`.
#[async_trait]
pub trait VectorOpsPort: Send + Sync {
    /// Execute a vector search, tenant-scoped when `identity.tenant_id` is
    /// provided; `identity.subject` + `identity.tenant_stable_id` drive ABAC
    /// enforcement at the shared search seam (`unified_search_v1_inner`).
    async fn search(
        &self,
        request: VectorSearchRequest,
        identity: PortIdentity<'_>,
    ) -> Result<VectorOperationResponse>;

    /// TD-XMODAL-4 S2: the **single canonical native vector-search kernel** — the
    /// v2 path shared by the pgvector `<->` operator and the `vector_search(...)`
    /// UDTF, returning [`OptimizedSearchRecord`]s for internal (Rust) callers.
    /// Tenant-scoped + **fail-closed** when `tenant_id` is provided (the impl
    /// validates collection access for the tenant before searching). The default
    /// fails closed so a missing implementation cannot masquerade as an empty result.
    ///
    /// (This is the one port method that exposes the v2 search-result type rather
    /// than a proto type — a deliberate exception so both SQL surfaces share one
    /// kernel instead of diverging; the types are foundation crates, so
    /// `proximadb-runtime` stays monolith-independent.)
    async fn unified_search_native(
        &self,
        _search: &proximadb_vector_query::VectorSearchExpr,
        _identity: PortIdentity<'_>,
    ) -> Result<Vec<proximadb_search_types::results::OptimizedSearchRecord>> {
        Err(anyhow::anyhow!(
            "native vector search is not implemented by this runtime port"
        ))
    }

    /// Execute a batch upsert/delete, tenant-scoped when `tenant_id` is provided.
    async fn batch_upsert(
        &self,
        request: VectorBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<VectorOperationResponse>;

    /// Fetch a single vector by ID.
    async fn get_vector(
        &self,
        collection_id: &str,
        vector_id: &str,
        include_vector: bool,
        include_metadata: bool,
        tenant_id: Option<&str>,
    ) -> Result<VectorOperationResponse>;

    /// Force-flush all pending WAL entries to the storage engine.
    async fn flush_all(&self) -> Result<()>;

    /// Return engine-level metrics as a JSON blob.
    async fn metrics(&self) -> Result<JsonValue>;

    /// Resolve `collection_id` and return the set of record ids whose property
    /// tree satisfies the v1 simple equality `filters` map, read from the
    /// authoritative record set (WAL memtable + flushed storage) and evaluated
    /// with the canonical filter. Used by hybrid search to enforce metadata
    /// filters on text-only (BM25) candidates, which carry no metadata of their
    /// own and so cannot be filtered by the retrieval engine.
    ///
    /// Fail-closed by contract: callers MUST drop any candidate absent from the
    /// returned set, and the default implementation returns an empty set so a
    /// port that does not back this path can never widen a filter to fail-open
    /// (which would re-open a cross-tenant disclosure).
    async fn record_ids_matching_filter(
        &self,
        _collection_id: &str,
        _filters: &HashMap<String, SqlValue>,
        _tenant_id: Option<&str>,
    ) -> Result<HashSet<String>> {
        Ok(HashSet::new())
    }

    /// Resolve a user-facing collection identifier to its canonical object id
    /// (recall-tune/diagnostics support). Fail-closed default: only the
    /// production impl resolves real ids.
    async fn resolve_collection_object_id(
        &self,
        _collection_id: &str,
    ) -> Result<proximadb_kernel::stable_id::CollectionObjectId> {
        Err(anyhow::anyhow!(
            "collection object-id resolution is not implemented by this runtime port"
        ))
    }

    /// List every visibility-filtered record (WAL memtable + flushed storage)
    /// for recall-tune reads. Fail-closed default.
    async fn list_records_for_recall_tune(
        &self,
        _internal_collection_id: &str,
        _tenant_id: Option<&str>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        Err(anyhow::anyhow!(
            "recall-tune record listing is not implemented by this runtime port"
        ))
    }
}

// ── Query facade ──────────────────────────────────────────────────────────────

/// Protocol-neutral result of executing one SQL statement.
///
/// SQL values remain canonical [`proximadb_data_model::ProximaValue`] instances
/// until a transport adapter serializes them. The v1 [`SqlValue`] type is a
/// compatibility wire type and must not leak into this result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SqlExecutionResult {
    /// Ordered column names; every row uses the same ordinal layout.
    pub columns: Vec<String>,
    /// Canonical logical type labels aligned with `columns`.
    pub column_types: Vec<String>,
    /// Ordered, typed result cells.
    pub rows: Vec<Vec<proximadb_data_model::ProximaValue>>,
    /// Present for DDL/DML and absent for row-producing statements.
    pub rows_affected: Option<u64>,
    pub rows_scanned: u64,
    pub execution_time_ms: u64,
}

/// Port for unified query routing (SQL, hybrid, vector-via-facade).
///
/// Implemented by root-crate `QueryFacadeAdapter` when the `unified-facade-routing`
/// feature is active.
#[async_trait]
pub trait QueryAdapterPort: Send + Sync {
    /// Route a vector search through the unified query planner.
    async fn vector_search(&self, request: VectorSearchRequest) -> Result<VectorOperationResponse>;

    /// Execute a hybrid (vector + keyword) query.
    async fn execute_hybrid(&self, request: HybridSearchRequest) -> Result<HybridSearchResponse>;

    /// Execute a SQL statement through the unified facade.
    ///
    /// Returns canonical typed rows. JSON and legacy v1 `SqlValue` conversion
    /// belong exclusively to their respective transport adapters.
    ///
    /// `identity` is the canonical ADR-087 carrier. Its string tenant scopes
    /// storage/catalog resolution while subject + stable key drive ABAC at the
    /// relational read seam; callers must not flatten or re-derive these fields.
    async fn execute_sql(
        &self,
        query: String,
        collection: Option<String>,
        identity: PortIdentity<'_>,
    ) -> Result<SqlExecutionResult>;
}

// ============================================================================
// Data-plane capability (auth contract)
// ============================================================================

#[derive(Debug, Clone)]
pub struct DataPlaneCapability {
    pub capability_type: String,
    pub collection: Option<String>,
    pub operation: Option<String>,
    pub protocol: Option<String>,
    pub mode: Option<String>,
    pub scopes: Vec<String>,
    pub max_records: Option<u64>,
    pub max_bytes: Option<u64>,
}

impl DataPlaneCapability {
    /// Build from a metadata map (ADR-094: the root middleware adapts its
    /// `UnifiedUserContext` to this call — the struct lives here so every
    /// transport shares one capability type).
    pub fn from_metadata(metadata: &std::collections::HashMap<String, String>) -> Option<Self> {
        let capability_type = metadata.get("capability_type")?.clone();
        Some(Self {
            capability_type,
            collection: metadata.get("collection").cloned(),
            operation: metadata.get("operation").cloned(),
            protocol: metadata.get("protocol").cloned(),
            mode: metadata.get("mode").cloned(),
            scopes: metadata
                .get("scopes")
                .map(|scopes| {
                    scopes
                        .split_whitespace()
                        .map(ToOwned::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            max_records: metadata.get("max_records").and_then(|value| value.parse::<u64>().ok()),
            max_bytes: metadata.get("max_bytes").and_then(|value| value.parse::<u64>().ok()),
        })
    }

    pub fn ensure_record_count(&self, count: usize) -> Result<(), String> {
        if let Some(max_records) = self.max_records
            && count as u64 > max_records
        {
            return Err(format!(
                "Request has {} records, exceeding capability limit {}",
                count, max_records
            ));
        }
        Ok(())
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|candidate| candidate == scope)
    }
}
