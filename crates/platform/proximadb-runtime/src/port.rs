//! API handlers port — the seam between protocol adapters and business logic.
//!
//! Protocol handlers in `proximadb-api` depend on this trait rather than on root-crate
//! concrete types. The root crate implements the trait on its `UnifiedHandlers` and
//! injects the `Arc<dyn ApiHandlersPort>` at server startup.
//!
//! Request/response envelopes that are still explicitly v1-compatible use proto
//! types, but value parameters crossing this runtime seam use the canonical
//! `ProximaValue` model. Protocol adapters are responsible for converting
//! legacy wire values at the edge.

use crate::service_ports::PortIdentity;
use anyhow::Result;
use async_trait::async_trait;
use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_proto::v1::{
    CollectionRequest, CollectionResponse, ExecuteQueryResponse, HybridSearchRequest,
    HybridSearchResponse, VectorBatchRequest, VectorOperationResponse, VectorSearchRequest,
};
use serde::{Deserialize, Serialize};

/// Runtime-native schema metadata for v2 collection/schema handlers.
///
/// This is intentionally not a proto envelope. The underlying collection store
/// may still adapt to older persisted metadata, but protocol adapters should
/// depend on this v2-shaped contract instead of constructing v1 collection
/// operation requests.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchemaMetadata {
    pub collection_id: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub schema_id: Option<String>,
    pub schema_version: Option<String>,
    pub enforcement: Option<CollectionSchemaEnforcement>,
    pub auto_evolve: bool,
    pub enabled: bool,
    pub columns: Vec<CollectionSchemaColumn>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchemaUpdate {
    pub schema_id: String,
    pub schema_version: String,
    pub enforcement: CollectionSchemaEnforcement,
    pub auto_evolve: bool,
    pub columns: Vec<CollectionSchemaColumn>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchemaColumn {
    pub name: String,
    pub data_type: ProximaType,
    pub nullable: bool,
    pub indexed: bool,
    pub filterable: bool,
    pub text_storage: Option<CollectionTextStorage>,
    pub max_length: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectionTextStorage {
    Inline,
    Large,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectionSchemaEnforcement {
    Strict,
    Flexible,
    #[default]
    Hybrid,
}

/// Port trait that protocol adapters use to dispatch API requests.
///
/// The root crate's `UnifiedHandlers` implements this trait. `proximadb-api` gRPC stubs
/// hold an `Arc<dyn ApiHandlersPort>` so the real implementation is swapped in without
/// `proximadb-api` importing any root-crate concrete types.
#[async_trait]
pub trait ApiHandlersPort: Send + Sync {
    // ── Collection ────────────────────────────────────────────────────────────

    async fn handle_collection_operation_for_tenant(
        &self,
        request: CollectionRequest,
        tenant_id: Option<&str>,
    ) -> Result<CollectionResponse>;

    /// Fetch v2 schema metadata for a collection without exposing v1 operation
    /// envelopes to protocol adapters.
    async fn get_collection_schema_metadata(
        &self,
        collection_id: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<CollectionSchemaMetadata>>;

    /// Persist v2 schema metadata for a collection. Implementations may adapt to
    /// legacy storage internally, but callers pass only the runtime-native shape.
    async fn update_collection_schema_metadata(
        &self,
        collection_id: &str,
        update: CollectionSchemaUpdate,
        tenant_id: Option<&str>,
    ) -> Result<CollectionSchemaMetadata>;

    /// [`Self::update_collection_schema_metadata`] with the record mode
    /// pinned: `Some(v)` forces the persisted `enable_proxima_record` to `v`
    /// (the gRPC create path's canonical persist step — persistence, not a
    /// mode switch); `None` keeps schema-update semantics (mode on).
    /// TD-GRPCSCHEMA-2. The default ignores the pin (mocks).
    async fn update_collection_schema_metadata_with_record_mode(
        &self,
        collection_id: &str,
        update: CollectionSchemaUpdate,
        tenant_id: Option<&str>,
        record_mode: Option<bool>,
    ) -> Result<CollectionSchemaMetadata> {
        let _ = record_mode;
        self.update_collection_schema_metadata(collection_id, update, tenant_id)
            .await
    }

    // ── Vector ────────────────────────────────────────────────────────────────

    /// `identity` carries the tenant scope + authenticated principal for ABAC
    /// enforcement at the shared search seam (TD-ABAC-7);
    /// [`PortIdentity::anonymous`] = no policy evaluation.
    async fn handle_vector_search_v1_for_tenant(
        &self,
        request: VectorSearchRequest,
        identity: PortIdentity<'_>,
    ) -> Result<VectorOperationResponse>;

    async fn handle_vector_search_v1(
        &self,
        request: VectorSearchRequest,
    ) -> Result<VectorOperationResponse>;

    async fn handle_vector_batch_v1_for_tenant(
        &self,
        request: VectorBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<VectorOperationResponse>;

    async fn handle_vector_v1_for_tenant(
        &self,
        collection_id: &str,
        vector_id: &str,
        include_vector: bool,
        include_metadata: bool,
        tenant_id: Option<&str>,
    ) -> Result<VectorOperationResponse>;

    // ── Hybrid ────────────────────────────────────────────────────────────────

    /// DEPRECATED (TD-143): v1 hybrid query — dormant (v1 gRPC surface removed, TD-V1SUNSET-1),
    /// not tenant-scoped, owns its own ranking. Use v2 FusionSearch instead.
    #[deprecated(note = "v1 ExecuteHybridQuery is deprecated; use v2 FusionSearch. See TD-143.")]
    async fn execute_hybrid_query(
        &self,
        request: HybridSearchRequest,
    ) -> Result<HybridSearchResponse>;

    // ── SQL ───────────────────────────────────────────────────────────────────

    /// Execute SQL while retaining canonical `ProximaValue` cells across the
    /// application boundary. New transports must use this method.
    async fn execute_sql(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        identity: crate::service_ports::PortIdentity<'_>,
    ) -> Result<crate::service_ports::SqlExecutionResult>;

    /// Legacy v1 protobuf compatibility adapter.
    async fn execute_sql_v1(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        // ADR-087: one canonical caller identity. Tenant scopes the partition;
        // subject + stable key + auth class reach the relational ABAC seam.
        identity: crate::service_ports::PortIdentity<'_>,
    ) -> Result<ExecuteQueryResponse>;
}
