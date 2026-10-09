// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Arrow Flight service implementation
//!
//! This implements the FlightService trait and delegates to UnifiedHandlers,
//! following the same pattern as REST and gRPC protocols.
//!
//! Note: This uses tonic 0.13 (via arrow-flight), which is different from
//! the main codebase's tonic 0.10. Types are carefully managed to avoid conflicts.

use anyhow::{Context, Result};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
    decode::FlightRecordBatchStream, encode::FlightDataEncoderBuilder, error::FlightError,
    flight_service_server::FlightService,
};
use arrow_schema::Schema;
use futures::{Stream, StreamExt, stream};
use proximadb_embedding::{
    EmbeddingService,
    scheduler::IngestMode,
    service::{EmbedBatch, EmbedRecord},
};
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::catalog::CatalogManager;
use crate::network::auth::middleware::DataPlaneCapability;
use crate::security::{AuthenticationData, ClientCertificateData, SecurityCoordinator};
use crate::services::operations::{
    BatchOperationResult, BulkWriteMode, CatalogBulkWriteResult, CatalogBulkWriteService,
};
use chrono::Utc;

use super::codec::{
    ArrowProtoCodec, FlightFilter, FlightSearchTicket, FlightWriteOperation, WriteMode,
};
use super::file_export::{
    ArrowFileExportHandler, ArrowFileRequest, ArrowFileTicket, FlightCompression,
};
use super::multimodal_codec::relational_schema_from_catalog;

// Type aliases using tonic from arrow-flight's dependency tree
// This avoids conflicts with the main codebase's tonic 0.10
type TonicRequest<T> = tonic::Request<T>;
type TonicResponse<T> = tonic::Response<T>;
type TonicStatus = tonic::Status;
type TonicStreaming<T> = tonic::Streaming<T>;

type TonicResult<T> = std::result::Result<TonicResponse<T>, TonicStatus>;
type TonicStream<T> = Pin<Box<dyn Stream<Item = std::result::Result<T, TonicStatus>> + Send>>;

/// Total wire bytes a `do_get` response will egress: the encoded header + body +
/// app-metadata across every `FlightData` frame. Used to meter KOU result-egress
/// on the actual (post-compression) bytes leaving the engine.
fn flight_data_wire_bytes(flight_data: &[FlightData]) -> u64 {
    flight_data
        .iter()
        .map(|fd| (fd.data_header.len() + fd.data_body.len() + fd.app_metadata.len()) as u64)
        .sum()
}

/// Optional control frame for the `bulk_search` DoExchange (TD-FLIGHT-1).
/// Carried as JSON in a `FlightData.app_metadata` message ahead of the query
/// batches; sets top_k / filters / include_vector for the queries that follow.
/// Absent ⇒ defaults (top_k = 10, no filters), matching the pre-TD-FLIGHT-1
/// behavior but now on the canonical v2 search path.
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct BulkSearchControl {
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    include_vector: bool,
    #[serde(default)]
    filters: Vec<FlightFilter>,
}

/// Build a control-only `FlightData` frame (no body) carrying `metadata` as
/// app-metadata — used for per-query error frames and the bulk_search
/// completion frame.
fn control_flight_data(metadata: Vec<u8>) -> FlightData {
    FlightData {
        flight_descriptor: None,
        data_header: Default::default(),
        app_metadata: metadata.into(),
        data_body: Default::default(),
    }
}

#[derive(Debug, Clone)]
struct AuthenticatedFlightContext {
    tenant_id: String,
    /// The authenticated principal's user id (TD-ABAC-6), threaded as the ABAC
    /// subject on the v2 search path. `None` on the trust-asserted path (no
    /// credential) or when no subject was resolved.
    user_id: Option<String>,
    /// ADR-087: the tenant's stable u64 (ABAC policy key), stamped ONCE by the
    /// identity orchestrator — never re-resolved per handler.
    tenant_stable_id: Option<u64>,
    /// ADR-087: trust provenance of this identity (audit-only at the seam).
    auth_class: proximadb_tenant::AuthClass,
    capability: Option<DataPlaneCapability>,
}

impl AuthenticatedFlightContext {
    /// The owned foundation identity for spawned/streaming paths (bulk_search).
    fn owned_identity(&self) -> proximadb_tenant::ResolvedRequestIdentity {
        proximadb_tenant::ResolvedRequestIdentity {
            tenant: self.tenant_id.clone(),
            subject: self.user_id.clone(),
            auth_class: self.auth_class,
            tenant_stable_id: self.tenant_stable_id,
        }
    }
}

/// ADR-087: the borrowed port-seam projection of the Flight identity.
impl<'a> From<&'a AuthenticatedFlightContext> for proximadb_runtime::PortIdentity<'a> {
    fn from(ctx: &'a AuthenticatedFlightContext) -> Self {
        Self {
            tenant_id: Some(ctx.tenant_id.as_str()),
            subject: ctx.user_id.as_deref(),
            tenant_stable_id: ctx.tenant_stable_id,
            auth_class: ctx.auth_class,
        }
    }
}

/// DoGet ticket for the batched columnar graph export path. JSON-encoded in the
/// Flight `Ticket` bytes; `model` selects nodes vs edges and the optional fields
/// scope the query (label / edge type / endpoints / limit).
#[derive(Debug, Clone, serde::Deserialize)]
struct GraphTicket {
    /// `"graph_nodes"` or `"graph_edges"`.
    model: String,
    graph_id: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    edge_type: Option<String>,
    #[serde(default)]
    from_node_id: Option<String>,
    #[serde(default)]
    to_node_id: Option<String>,
    #[serde(default)]
    limit: Option<u32>,
}

/// `RecordBatch` source item for the streaming graph export (the input to the
/// Flight `FlightDataEncoder`).
type GraphBatchResult = std::result::Result<arrow_array::RecordBatch, FlightError>;

/// Pagination state for the streaming node export.
enum NodePage {
    /// Pre-fetched first page — also used to fix the stream's schema dimension.
    First(Vec<crate::graph::Node>),
    /// Fetch the page starting at this offset next.
    More(usize),
    /// No more pages.
    Done,
}

/// Map a graph/codec error into a Flight stream error.
fn graph_flight_err(e: anyhow::Error) -> FlightError {
    FlightError::ProtocolError(e.to_string())
}

/// ProximaDB Flight service implementation
///
/// Thin wrapper around UnifiedHandlers that converts Arrow Flight messages
/// to/from ProximaDB proto types.
///
/// ## Supported Operations
///
/// - **Vector Search** (DoGet): Execute vector similarity search
/// - **Vector Insert** (DoPut): Bulk insert vectors via Arrow IPC
/// - **File Export** (DoGet with arrow_file ticket): Stream .arrow or .parquet files directly
/// - **List Files** (GetFlightInfo): List available .arrow and .parquet files in a collection
///
/// ## Supported File Formats
///
/// - **.arrow**: Arrow IPC files (from SST, HELIX engines)
/// - **.parquet**: Parquet files (from Nova, VIPER engines)
pub struct ProximaFlightService {
    // TD-104 S3 / TD-FLIGHT-1: the Flight service depends on ports + the
    // concrete services it actually uses, not a monolithic handler. Canonical
    // v2 vector search goes through `RecordSearchPort`, record-batch ingest
    // through `RecordOpsPort`; the vector-ops/collection services are held
    // directly (same Arcs as before). (TD-FLIGHT-2: the deprecated v1
    // `api_port`/`ApiHandlersPort` search field was removed once #1351 landed —
    // the v1 method stays on `ApiHandlersPort` for REST `/progressive/search`.)
    record_port: Arc<dyn proximadb_runtime::RecordOpsPort>,
    record_search_port: Arc<dyn proximadb_runtime::RecordSearchPort>,
    vector_operations_service: Arc<crate::services::VectorOperationsService>,
    collection_service: Arc<crate::services::CollectionService>,
    security_coordinator: Option<Arc<SecurityCoordinator>>,
    /// TD-TENANT-1: the deployment's bare `x-tenant-id` trust policy,
    /// enforced through the ONE shared primitive in
    /// `authenticated_flight_context`. Default `Open` (legacy behavior).
    tenant_header_trust: proximadb_tenant::HeaderTrustPolicy,
    /// TD-TENANT-3 S2 / ADR-0053 W8: the deployment's `x-tenant-tier` claim
    /// trust policy, the Flight sibling of the REST and gRPC gates.
    tier_header_trust: proximadb_tenant::HeaderTrustPolicy,
    /// Whether missing request identity may resolve to a configured default.
    tenant_deployment_mode: proximadb_tenant::TenantDeploymentMode,
    catalog_manager: Option<Arc<CatalogManager>>,
    /// TD-TENANT-1: catalog-backed tenant stable-id resolver, mirroring the
    /// REST `TenantExtractor::with_stable_id_resolver` seam. When present,
    /// `handle_v2_search` stamps the resolved stable u64 into the io_trace
    /// boundary (so Flight search is attributable per-tenant like REST).
    stable_id_resolver: Option<Arc<dyn proximadb_tenant::TenantStableIdResolver>>,
    /// R-7c.4b: when present, the `rank_features_export` Flight action
    /// drives the multi-phase ranking pipeline through this singleton.
    /// Absent means the action returns `Unimplemented` (deployments that
    /// didn't opt into the ranking framework don't pay for it).
    rank_services: Option<Arc<crate::network::rest::canonical::rank::RankServices>>,
    /// Slice 6.2: primary-pod write router. Same shape as the gRPC v2
    /// service's `primary_pod_gate` — when present, `do_put` consults
    /// the registry before any storage work and rejects misrouted
    /// writes with `failed_precondition` + trailing metadata. Covers
    /// Insert/Upsert/Delete (every mutation, not just Insert) because
    /// any of them landing on the wrong pod's memtable would be
    /// invisible to the readers on the primary pod.
    primary_pod_gate: Option<FlightPrimaryPodGate>,
    /// Graph backing service for the batched columnar graph path
    /// (`graph_nodes`/`graph_edges` DoExchange + DoGet). Held directly (like the
    /// vector-ops service) so bulk graph ingest lands in the live graph engine
    /// rather than the generic record store. `None` outside production wiring
    /// (e.g. the port-only test constructor), in which case the graph Flight
    /// routes return `Unimplemented`.
    graph_service: Option<Arc<crate::graph::GraphOperationsService>>,
    _codec: ArrowProtoCodec,
    file_export_handler: ArrowFileExportHandler,
}

/// Slice 6.2 gate-input bundle. Distinct from the gRPC v2 service's
/// `PrimaryPodGate` only because module privacy keeps each surface
/// self-contained; the wire contract is identical.
#[derive(Clone)]
struct FlightPrimaryPodGate {
    registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
    self_pod_id: String,
}

/// Slice 6.2 testable gate check. Same logic as the gRPC v2
/// `check_primary_pod_gate` — free function so unit tests can call
/// it without constructing the full `ProximaFlightService`.
fn check_flight_primary_pod_gate(
    gate: &Option<FlightPrimaryPodGate>,
    tenant_id: &str,
    collection_id: &str,
) -> Result<(), TonicStatus> {
    let Some(gate) = gate else {
        return Ok(());
    };
    match crate::cluster::primary_pod_registry::consult_for_write(
        &gate.registry,
        &gate.self_pod_id,
        tenant_id,
        collection_id,
    ) {
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Allow => {
            if gate.registry.is_assigned(tenant_id, collection_id) {
                crate::metrics::primary_pod_metrics::record_allowed_bound(tenant_id);
            } else {
                crate::metrics::primary_pod_metrics::record_allowed_unbounded(tenant_id);
            }
            Ok(())
        }
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Misrouted { target_pod } => {
            crate::metrics::primary_pod_metrics::record_misrouted(tenant_id);
            tracing::warn!(
                target = "proximadb.primary_pod.misroute",
                self_pod = %gate.self_pod_id,
                target_pod = %target_pod,
                tenant_id = %tenant_id,
                collection_id = %collection_id,
                "Arrow Flight do_put misrouted — client should retry against the primary pod"
            );
            let api_err = crate::errors::ApiError::Misdirected {
                target_pod,
                tenant_id: tenant_id.to_string(),
                collection_id: collection_id.to_string(),
            };
            Err(api_err.into())
        }
    }
}

impl ProximaFlightService {
    /// Create a new Arrow Flight service from injected ports + concrete services.
    ///
    /// TD-104 S3-c: the Flight service no longer depends on the concrete root
    /// `UnifiedHandlers`. Vector search goes through the `ApiHandlersPort`;
    /// record-batch ingest goes through the canonical `RecordOpsPort` (backed
    /// directly by the runtime `RecordOpsService`, so the write path no longer
    /// routes through ROOT). The vector-ops/collection services and the storage
    /// locations are passed in by the boot wiring (multi_server / ArrowFlightServer).
    pub fn new(
        record_port: Arc<dyn proximadb_runtime::RecordOpsPort>,
        record_search_port: Arc<dyn proximadb_runtime::RecordSearchPort>,
        vector_operations_service: Arc<crate::services::VectorOperationsService>,
        collection_service: Arc<crate::services::CollectionService>,
        storage_locations: Vec<String>,
    ) -> Self {
        Self {
            record_port,
            record_search_port,
            vector_operations_service,
            collection_service,
            security_coordinator: None,
            tenant_header_trust: proximadb_tenant::HeaderTrustPolicy::default(),
            tier_header_trust: proximadb_tenant::HeaderTrustPolicy::default(),
            tenant_deployment_mode: proximadb_tenant::TenantDeploymentMode::single_tenant_default(),
            catalog_manager: None,
            stable_id_resolver: None,
            rank_services: None,
            primary_pod_gate: None,
            graph_service: None,
            _codec: ArrowProtoCodec,
            file_export_handler: ArrowFileExportHandler::new(storage_locations),
        }
    }

    /// Attach the graph backing service so the `graph_nodes`/`graph_edges`
    /// DoExchange ingest and DoGet export routes are live. Same
    /// `GraphOperationsService` the gRPC/REST graph surfaces use.
    pub fn with_graph_service(
        mut self,
        graph_service: Arc<crate::graph::GraphOperationsService>,
    ) -> Self {
        self.graph_service = Some(graph_service);
        self
    }

    /// Boot adapter (TD-104 S3-c/S3-e): build a `ProximaFlightService` from the
    /// runtime ports plus the concrete services it needs, all passed directly
    /// (no root `UnifiedHandlers` indirection). `storage_locations` is derived
    /// from the collection service's storage config — the same read the former
    /// root `storage_config()` delegated to.
    pub fn from_services(
        record_port: Arc<dyn proximadb_runtime::RecordOpsPort>,
        record_search_port: Arc<dyn proximadb_runtime::RecordSearchPort>,
        vector_operations_service: Arc<crate::services::VectorOperationsService>,
        collection_service: Arc<crate::services::CollectionService>,
        graph_service: Arc<crate::graph::GraphOperationsService>,
    ) -> Self {
        let storage_locations: Vec<String> = collection_service
            .storage_config()
            .storage_locations
            .iter()
            .map(|loc| loc.url.clone())
            .collect();

        Self::new(
            record_port,
            record_search_port,
            vector_operations_service,
            collection_service,
            storage_locations,
        )
        .with_graph_service(graph_service)
    }

    /// Slice 6.2: attach the primary-pod write router. Once set,
    /// `do_put` rejects misrouted Insert/Upsert/Delete batches with
    /// `failed_precondition` + `x-primary-pod` trailing metadata
    /// before consuming a single record from the stream.
    pub fn with_primary_pod_gate(
        mut self,
        registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
        self_pod_id: String,
    ) -> Self {
        self.primary_pod_gate = Some(FlightPrimaryPodGate {
            registry,
            self_pod_id,
        });
        self
    }

    /// R-7c.4b: attach the shared RankServices singleton so the
    /// `rank_features_export` action can drive the rank pipeline. Same
    /// `Arc<RankServices>` the REST and gRPC paths hold — single source
    /// of truth across all three protocols.
    pub fn with_rank_services(
        mut self,
        rank_services: Option<Arc<crate::network::rest::canonical::rank::RankServices>>,
    ) -> Self {
        self.rank_services = rank_services;
        self
    }

    /// Attach the shared security coordinator used by other network surfaces.
    pub fn with_security_coordinator(
        mut self,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
    ) -> Self {
        self.security_coordinator = security_coordinator;
        self
    }

    /// Set the deployment's bare `x-tenant-id` trust policy (TD-TENANT-1) —
    /// the same policy the REST middleware, gRPC auth layer, and pgwire
    /// resolve seams enforce through the shared primitive.
    pub fn with_tenant_header_trust(mut self, policy: proximadb_tenant::HeaderTrustPolicy) -> Self {
        self.tenant_header_trust = policy;
        self
    }

    /// Set the deployment's `x-tenant-tier` claim trust policy
    /// (TD-TENANT-3 S2 / ADR-0053 W8) — the same policy REST, gRPC, and pgwire
    /// enforce through the shared `resolve_tier_claim` primitive.
    pub fn with_tier_header_trust(mut self, policy: proximadb_tenant::HeaderTrustPolicy) -> Self {
        self.tier_header_trust = policy;
        self
    }

    /// Set the request-tenant presence contract independently of authentication.
    pub fn with_tenant_deployment_mode(
        mut self,
        mode: proximadb_tenant::TenantDeploymentMode,
    ) -> Self {
        self.tenant_deployment_mode = mode;
        self
    }

    fn resolve_tenant_for_mode(
        tenant_id: Option<&str>,
        mode: &proximadb_tenant::TenantDeploymentMode,
    ) -> std::result::Result<String, TonicStatus> {
        proximadb_tenant::resolve_request_tenant_for_mode(tenant_id, mode).map_err(|error| {
            match error {
                proximadb_tenant::ResolveRequestTenantError::MissingTenant => {
                    TonicStatus::unauthenticated(error.to_string())
                }
                proximadb_tenant::ResolveRequestTenantError::InvalidTenant(_) => {
                    TonicStatus::invalid_argument(error.to_string())
                }
            }
        })
    }

    /// Attach xCatalog metadata for relational/table Flight schema resolution.
    pub fn with_catalog_manager(mut self, catalog_manager: Option<Arc<CatalogManager>>) -> Self {
        self.catalog_manager = catalog_manager;
        self
    }

    /// Wire the catalog-backed tenant stable-id resolver (TD-TENANT-1), mirroring
    /// REST's `TenantExtractor::with_stable_id_resolver`. When set, Flight search
    /// stamps the resolved stable id into the io_trace boundary.
    pub fn with_stable_id_resolver(
        mut self,
        resolver: Option<Arc<dyn proximadb_tenant::TenantStableIdResolver>>,
    ) -> Self {
        self.stable_id_resolver = resolver;
        self
    }

    fn batch_result_app_metadata(result: &BatchOperationResult) -> Result<Vec<u8>> {
        serde_json::to_vec(result).map_err(Into::into)
    }

    fn batch_progress_app_metadata(
        operation: FlightWriteOperation,
        batch: u64,
        batch_rows: usize,
        cumulative_records: u64,
        result: &BatchOperationResult,
    ) -> Result<Vec<u8>> {
        let failed_count = result.metrics.failed_count.max(result.errors.len() as i64);
        let progress = serde_json::json!({
            "type": "progress",
            "operation": operation.as_str(),
            "batch": batch,
            "batch_rows": batch_rows,
            "total_records": cumulative_records,
            "success": result.success,
            "successful_count": result.metrics.successful_count,
            "failed_count": failed_count,
            "record_ids": result.vector_ids,
            "errors": result.errors,
            "error_code": result.error_code,
        });

        serde_json::to_vec(&progress).map_err(Into::into)
    }

    fn bulk_insert_complete_app_metadata(
        operation: FlightWriteOperation,
        total_batches: u64,
        total_records: u64,
        total_failed: u64,
        success: bool,
    ) -> Result<Vec<u8>> {
        let final_status = serde_json::json!({
            "type": "complete",
            "operation": operation.as_str(),
            "total_batches": total_batches,
            "total_records": total_records,
            "total_failed": total_failed,
            "success": success,
        });

        serde_json::to_vec(&final_status).map_err(Into::into)
    }

    fn schema_result_from_arrow_schema(schema: &Arc<Schema>) -> Result<SchemaResult> {
        use arrow_ipc::writer::IpcWriteOptions;

        let write_options = IpcWriteOptions::default();
        let mut schema_bytes = Vec::new();
        {
            let mut writer = arrow_ipc::writer::FileWriter::try_new_with_options(
                &mut schema_bytes,
                schema,
                write_options,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create schema writer: {}", e))?;
            writer
                .finish()
                .map_err(|e| anyhow::anyhow!("Failed to write schema: {}", e))?;
        }

        Ok(SchemaResult {
            schema: schema_bytes.into(),
        })
    }

    async fn catalog_arrow_schema_for_descriptor(
        catalog_manager: Option<&Arc<CatalogManager>>,
        descriptor: &FlightDescriptor,
    ) -> Result<Option<Arc<Schema>>> {
        let Some(catalog_manager) = catalog_manager else {
            return Ok(None);
        };
        let Some(table_fqn) = Self::table_fqn_from_descriptor(descriptor)? else {
            return Ok(None);
        };

        let (catalog, table_id) = catalog_manager.resolve_table(&table_fqn).await?;
        let table_schema = catalog.get_table(&table_id).await?;
        Ok(Some(relational_schema_from_catalog(&table_schema)))
    }

    fn table_fqn_from_descriptor(descriptor: &FlightDescriptor) -> Result<Option<String>> {
        let path_model = descriptor.path.first().map(String::as_str);
        if matches!(path_model, Some("relational" | "table" | "sql")) {
            return Ok(descriptor.path.get(1).cloned());
        }

        if descriptor.cmd.is_empty() {
            return Ok(None);
        }

        let cmd: serde_json::Value = serde_json::from_slice(&descriptor.cmd)
            .context("Invalid FlightDescriptor command for schema lookup")?;
        let model = cmd
            .get("model_type")
            .or_else(|| cmd.get("model"))
            .or_else(|| cmd.get("schema_mode"))
            .and_then(|value| value.as_str());
        let is_relational = matches!(model, Some("relational" | "table" | "sql"));
        if !is_relational {
            return Ok(None);
        }

        Ok(cmd
            .get("table_fqn")
            .or_else(|| cmd.get("table_name"))
            .or_else(|| cmd.get("table"))
            .or_else(|| cmd.get("collection_id"))
            .or_else(|| cmd.get("collection"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned))
    }

    /// TD-TENANT-3: the shared claim vocabulary. Flight historically accepted
    /// four tenant spellings while REST and gRPC accepted one, so a claim
    /// honored here was *silently ignored* there. The alias list now lives in
    /// TD-TENANT-3 S4 item 1 (honoring removed 2026-08-29): Flight reads the
    /// CANONICAL name only, exactly like REST and gRPC — a legacy alias no
    /// longer grants a tenant (the client resolves to its credential/default
    /// tenant). Presence of a legacy name is still DETECTED: warn once per
    /// process + count every attempt, so the behavior change is observable
    /// rather than silent. The names are deleted at the next release boundary.
    fn tenant_id_from_metadata(metadata: &tonic::metadata::MetadataMap) -> Option<String> {
        if let Some(alias) = proximadb_tenant::legacy_alias_present(|name| {
            metadata.get(name).and_then(|value| value.to_str().ok())
        }) {
            crate::network::middleware::claim_metrics::record_deprecated_claim_use("flight", alias);
            Self::warn_deprecated_tenant_alias(alias);
        }
        proximadb_tenant::tenant_claim(|name| {
            metadata.get(name).and_then(|value| value.to_str().ok())
        })
        .map(|hit| hit.value.to_owned())
    }

    /// Warn once per deprecated alias (not per request) — a per-request warn on
    /// a bulk Flight path would be its own throughput problem.
    fn warn_deprecated_tenant_alias(alias: &'static str) {
        static WARNED: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashSet<&'static str>>,
        > = std::sync::OnceLock::new();
        let warned = WARNED.get_or_init(Default::default);
        let Ok(mut seen) = warned.lock() else {
            return;
        };
        if seen.insert(alias) {
            tracing::warn!(
                target: "proximadb::tenant_audit",
                surface = "flight",
                alias,
                canonical = proximadb_tenant::TENANT_CLAIM_HEADER,
                "deprecated tenant-claim alias accepted on Arrow Flight; \
                 migrate to the canonical header (TD-TENANT-3 S4 removes it)"
            );
        }
    }

    /// Read and gate the Arrow Flight tier entitlement claim (TD-TENANT-3 S2).
    ///
    /// Flight previously had no tier surface at all, so a Flight client always
    /// ran at the default tier — the inverted priority, since Flight is the
    /// bulk/columnar path and therefore the highest cache-pressure surface in
    /// the system, exactly where per-tenant floors and weights matter most.
    ///
    /// Semantics match REST and gRPC exactly: a rejected claim is DROPPED and
    /// warned, never an error, and the stamp is insert-only.
    fn stamp_gated_tier_claim(
        metadata: &tonic::metadata::MetadataMap,
        tenant_id: &str,
        binding: Option<&proximadb_tenant::AuthenticatedTenantBinding>,
        policy: proximadb_tenant::HeaderTrustPolicy,
    ) {
        if tenant_id.is_empty() {
            return;
        }
        let claim = proximadb_tenant::tier_claim(|name| {
            metadata.get(name).and_then(|value| value.to_str().ok())
        });
        match proximadb_tenant::resolve_tier_claim(claim, binding, policy) {
            Ok(Some(tier)) => {
                crate::services::record_store::set_tenant_tier(tenant_id, tier);
            }
            Ok(None) => {}
            Err(rejection) => {
                crate::network::middleware::tenant::warn_tier_claim_dropped(
                    "flight", tenant_id, &rejection,
                );
            }
        }
    }

    fn auth_data_from_metadata(
        metadata: &tonic::metadata::MetadataMap,
    ) -> std::result::Result<Option<AuthenticationData>, TonicStatus> {
        if let Some(auth_header) = metadata
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            // TD-ABAC-6: the shared credential parser — this branch was a
            // verbatim copy of gRPC's `auth_data_from_headers` and REST's
            // `map_header_to_auth_data` (same Bearer/API-Key/raw logic).
            return Ok(Some(
                crate::security::request_identity::parse_authorization(auth_header),
            ));
        }

        for key in ["x-api-key", "api-key"] {
            if let Some(api_key) = metadata
                .get(key)
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                return Ok(Some(AuthenticationData::ApiKey(api_key.to_string())));
            }
        }

        Ok(None)
    }

    fn auth_data_from_peer_certificate_der(cert_der: &[u8]) -> Option<AuthenticationData> {
        if cert_der.is_empty() {
            return None;
        }

        let now = Utc::now();
        Some(AuthenticationData::ClientCertificate(
            ClientCertificateData {
                subject: String::new(),
                issuer: String::new(),
                serial_number: String::new(),
                not_before: now,
                not_after: now,
                raw_cert_der: Some(cert_der.to_vec()),
            },
        ))
    }

    fn auth_data_from_peer_certs(
        peer_certs: Option<Arc<Vec<tonic::transport::CertificateDer<'static>>>>,
    ) -> Option<AuthenticationData> {
        peer_certs.as_ref().and_then(|certs| {
            certs
                .first()
                .and_then(|cert| Self::auth_data_from_peer_certificate_der(cert.as_ref()))
        })
    }

    async fn authenticated_flight_context(
        &self,
        metadata: &tonic::metadata::MetadataMap,
        peer_certs: Option<Arc<Vec<tonic::transport::CertificateDer<'static>>>>,
    ) -> std::result::Result<AuthenticatedFlightContext, TonicStatus> {
        let requested_tenant_id = Self::tenant_id_from_metadata(metadata);
        let credential = Self::auth_data_from_metadata(metadata)?
            .or_else(|| Self::auth_data_from_peer_certs(peer_certs));
        let coordinator = self.security_coordinator.as_deref();

        // An auth-wired deployment (coordinator present) requires a credential —
        // token or mTLS peer cert. No credential ⇒ unauthenticated.
        if coordinator.is_some() && credential.is_none() {
            return Err(TonicStatus::unauthenticated(
                "Arrow Flight authentication required",
            ));
        }

        // TD-ABAC-6: the ONE identity resolver (authenticate → tenant/subject
        // trust-gate → mode gate). Arrow has no subject-assertion surface, so
        // the subject comes from the credential (authenticated) or is absent
        // (trust-asserted). The two former branches — no-coordinator (bare
        // assertion) and coordinator (authenticated binding) — are exactly the
        // orchestrator's trust-asserted vs authenticated paths.
        let resolved = crate::security::request_identity::resolve_request_identity(
            coordinator,
            credential,
            requested_tenant_id.as_deref(),
            None,
            self.tenant_header_trust,
            &self.tenant_deployment_mode,
            self.stable_id_resolver.as_deref(),
        )
        .await
        .map_err(|err| Self::identity_error_to_flight_status(err, self.tenant_header_trust))?;

        // TD-TENANT-3 S2: the tier hook Flight was missing. The binding comes
        // from the resolved credential (Flight's authenticated principal); the
        // stamp lands on the ACTING tenant, matching REST and gRPC.
        let binding = resolved
            .user_context
            .as_ref()
            .and_then(|ctx| ctx.tenant_id.as_ref())
            .map(|tenant_id| proximadb_tenant::AuthenticatedTenantBinding {
                tenant_id: tenant_id.clone(),
                is_gateway_principal: resolved
                    .user_context
                    .as_ref()
                    .is_some_and(|ctx| ctx.is_gateway_principal()),
            });
        Self::stamp_gated_tier_claim(
            metadata,
            &resolved.identity.tenant,
            binding.as_ref(),
            self.tier_header_trust,
        );

        Ok(AuthenticatedFlightContext {
            tenant_id: resolved.identity.tenant,
            // #1309: surface the authenticated principal so the vector/ANN read
            // path can thread it as the ABAC subject (previously dropped here).
            user_id: resolved.identity.subject,
            tenant_stable_id: resolved.identity.tenant_stable_id,
            auth_class: resolved.identity.auth_class,
            capability: resolved
                .user_context
                .as_ref()
                .and_then(DataPlaneCapability::from_user_context),
        })
    }

    /// Map the unified [`IdentityError`] onto a Flight gRPC status, preserving
    /// the per-surface `tenant_audit` trail for assertion rejections
    /// (TD-TENANT-1). The orchestrator never logs; each surface owns its audit.
    fn identity_error_to_flight_status(
        error: crate::security::request_identity::IdentityError,
        trust: proximadb_tenant::HeaderTrustPolicy,
    ) -> TonicStatus {
        use crate::security::request_identity::IdentityError;
        match error {
            IdentityError::Authentication(message) => {
                TonicStatus::unauthenticated(format!("Authentication failed: {message}"))
            }
            IdentityError::Assertion(error) => {
                tracing::warn!(
                    target: "proximadb::tenant_audit",
                    surface = "arrow_flight",
                    policy = %trust,
                    %error,
                    "rejected x-tenant-id under tenant trust policy"
                );
                TonicStatus::permission_denied(error.to_string())
            }
            IdentityError::TenantResolution(error) => match error {
                proximadb_tenant::ResolveRequestTenantError::MissingTenant => {
                    TonicStatus::unauthenticated(error.to_string())
                }
                proximadb_tenant::ResolveRequestTenantError::InvalidTenant(_) => {
                    TonicStatus::invalid_argument(error.to_string())
                }
            },
        }
    }

    fn validate_flight_write_capability(
        capability: Option<&DataPlaneCapability>,
        collection_id: &str,
        operation: FlightWriteOperation,
        record_count: usize,
    ) -> std::result::Result<(), TonicStatus> {
        let Some(capability) = capability else {
            return Ok(());
        };
        if capability.protocol.as_deref() != Some("arrow_flight") {
            return Err(TonicStatus::permission_denied(
                "Capability token is not valid for Arrow Flight",
            ));
        }
        if capability.collection.as_deref() != Some(collection_id) {
            return Err(TonicStatus::permission_denied(
                "Capability token collection does not match Flight descriptor",
            ));
        }
        if !matches!(
            operation,
            FlightWriteOperation::Insert | FlightWriteOperation::Upsert
        ) {
            return Err(TonicStatus::permission_denied(
                "Capability token does not allow this Flight write operation",
            ));
        }
        if capability.operation.as_deref() != Some("ingest")
            || !capability
                .scopes
                .iter()
                .any(|scope| scope == "records:write")
        {
            return Err(TonicStatus::permission_denied(
                "Capability token lacks ingest scope",
            ));
        }
        capability
            .ensure_record_count(record_count)
            .map_err(TonicStatus::resource_exhausted)
    }

    fn validate_flight_search_capability(
        capability: Option<&DataPlaneCapability>,
        collection_id: &str,
    ) -> std::result::Result<(), TonicStatus> {
        let Some(capability) = capability else {
            return Ok(());
        };
        if capability.protocol.as_deref() != Some("arrow_flight") {
            return Err(TonicStatus::permission_denied(
                "Capability token is not valid for Arrow Flight",
            ));
        }
        if capability.collection.as_deref() != Some(collection_id) {
            return Err(TonicStatus::permission_denied(
                "Capability token collection does not match Flight ticket",
            ));
        }
        if capability.operation.as_deref() != Some("search")
            || !capability
                .scopes
                .iter()
                .any(|scope| scope == "search:execute" || scope == "records:read")
        {
            return Err(TonicStatus::permission_denied(
                "Capability token lacks search scope",
            ));
        }
        Ok(())
    }

    fn record_batch_stream(
        first_msg: FlightData,
        stream: TonicStreaming<FlightData>,
    ) -> FlightRecordBatchStream {
        let first = if first_msg.data_header.is_empty() {
            Vec::new()
        } else {
            vec![Ok(first_msg)]
        };
        let first_stream = stream::iter(first);
        let remaining_stream = stream.filter_map(|message| async move {
            match message {
                Ok(data) if data.data_header.is_empty() => None,
                Ok(data) => Some(Ok(data)),
                Err(status) => Some(Err(arrow_flight::error::FlightError::from(status))),
            }
        });
        FlightRecordBatchStream::new_from_flight_data(first_stream.chain(remaining_stream))
    }

    fn empty_batch_result() -> BatchOperationResult {
        BatchOperationResult::success(Vec::new(), Default::default())
    }

    fn merge_batch_result(total: &mut BatchOperationResult, result: BatchOperationResult) {
        total.success &= result.success;
        total.vector_ids.extend(result.vector_ids);
        total.errors.extend(result.errors);
        if total.error_code.is_none() {
            total.error_code = result.error_code;
        }
        total.metrics.total_processed += result.metrics.total_processed;
        total.metrics.successful_count += result.metrics.successful_count;
        total.metrics.failed_count += result.metrics.failed_count;
        total.metrics.updated_count += result.metrics.updated_count;
        total.metrics.processing_time_us += result.metrics.processing_time_us;
        total.metrics.wal_write_time_us += result.metrics.wal_write_time_us;
        total.metrics.index_update_time_us += result.metrics.index_update_time_us;
    }

    fn parse_exchange_descriptor(
        descriptor: &FlightDescriptor,
    ) -> Result<(String, String, Option<FlightWriteOperation>)> {
        if let Some(exchange_type) = descriptor.path.first() {
            let collection_id = descriptor
                .path
                .get(1)
                .cloned()
                .context("Exchange descriptor path is missing collection id")?;
            let operation = FlightWriteOperation::from_token(exchange_type);
            return Ok((exchange_type.clone(), collection_id, operation));
        }

        // Command descriptors are accepted for parity with DoPut and for
        // clients that cannot easily populate descriptor path segments.
        let cmd: serde_json::Value = serde_json::from_slice(&descriptor.cmd)
            .context("Invalid exchange descriptor command")?;
        let exchange_type = cmd
            .get("exchange_type")
            .or_else(|| cmd.get("operation"))
            .or_else(|| cmd.get("write_operation"))
            .and_then(|value| value.as_str())
            .context("Exchange descriptor command is missing exchange_type/operation")?
            .to_string();
        let operation = FlightWriteOperation::from_token(&exchange_type);
        let exchange_type = match (exchange_type.as_str(), operation) {
            ("insert", Some(FlightWriteOperation::Insert)) => "bulk_insert".to_string(),
            ("upsert", Some(FlightWriteOperation::Upsert)) => "bulk_upsert".to_string(),
            ("delete", Some(FlightWriteOperation::Delete)) => "bulk_delete".to_string(),
            _ => exchange_type,
        };
        let collection_id = cmd
            .get("collection_id")
            .or_else(|| cmd.get("collection"))
            .and_then(|value| value.as_str())
            .context("Exchange descriptor command is missing collection_id")?
            .to_string();

        Ok((exchange_type, collection_id, operation))
    }

    /// Handle Arrow file export (DoGet with arrow_file ticket)
    async fn handle_arrow_file_export(
        &self,
        file_ticket: ArrowFileTicket,
    ) -> Result<Vec<arrow_array::RecordBatch>> {
        info!(
            collection_id = %file_ticket.collection_id,
            "Arrow Flight file export"
        );

        let collection = self
            .collection_service
            .collection(&file_ticket.collection_id)
            .await?
            .with_context(|| format!("Collection not found: {}", file_ticket.collection_id))?;

        self.file_export_handler
            .read_collection_file(&collection, &file_ticket.file_path)
    }

    async fn trigger_collection_compaction(&self, collection_id: &str) -> Result<()> {
        let collection = self
            .collection_service
            .collection(collection_id)
            .await?
            .with_context(|| format!("Collection not found: {collection_id}"))?;
        let storage_engine = self.vector_operations_service.unified_engine();
        storage_engine
            .compact_collection(&collection.id, Some(&collection))
            .await
            .with_context(|| format!("Failed to compact collection '{}'", collection_id))?;
        Ok(())
    }

    fn insert_request_conflict_result(
        records: &[ProximaRecord],
        seen_ids: &mut HashSet<String>,
    ) -> Option<BatchOperationResult> {
        for record in records {
            if record.oid.is_empty() {
                return Some(BatchOperationResult::failure(
                    "Insert requires non-empty record id".to_string(),
                    "INVALID_RECORD_ID".to_string(),
                ));
            }

            if !seen_ids.insert(record.oid.clone()) {
                return Some(BatchOperationResult::failure(
                    format!(
                        "Record '{}' appears more than once in insert request",
                        record.oid
                    ),
                    "INSERT_CONFLICT".to_string(),
                ));
            }
        }

        None
    }

    fn insert_conflict_result(
        records: &[ProximaRecord],
        seen_ids: &mut HashSet<String>,
    ) -> Option<BatchOperationResult> {
        Self::insert_request_conflict_result(records, seen_ids)
    }

    fn catalog_bulk_write_mode(
        operation: FlightWriteOperation,
        write_mode: WriteMode,
    ) -> BulkWriteMode {
        match (operation, write_mode) {
            (_, WriteMode::Direct) => BulkWriteMode::Append,
            (FlightWriteOperation::Insert, _) => BulkWriteMode::InsertIfNotExists,
            (FlightWriteOperation::Upsert, _) => BulkWriteMode::Upsert,
            (FlightWriteOperation::Delete, _) => BulkWriteMode::Append,
        }
    }

    /// Phase 1 native-embedding dispatch.
    ///
    /// Walks `records` and identifies those with empty `embeddings` (the Flight
    /// text-only schema variant). Extracts text from the `text` property (or
    /// falls back to `body` / `title` when those are present), batches by
    /// tenant, and calls [`EmbeddingService::embed_sync`]. Populates the
    /// resulting vectors back onto each record as the default modality cell.
    ///
    /// Records with non-empty `embeddings` pass through unchanged. If the
    /// embedding singleton hasn't been initialized (defensive), we leave the
    /// records as-is and let the existing catalog validation reject them —
    /// no silent data loss.
    ///
    /// `pub(crate)` so the REST `/api/v2/documents` handler can reuse the same
    /// dispatch logic without duplicating it.
    pub(crate) async fn embed_text_only_records(
        records: &mut [ProximaRecord],
        tenant_id: Option<&str>,
    ) -> Result<()> {
        // Find indices that need embedding.
        let to_embed: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.embeddings.is_empty())
            .map(|(i, _)| i)
            .collect();

        if to_embed.is_empty() {
            return Ok(());
        }

        // Singleton may be absent in unit tests that don't call
        // EmbeddingService::initialize(); silently no-op in that case.
        let Some(service) = EmbeddingService::try_global() else {
            warn!(
                count = to_embed.len(),
                "embedding singleton not initialized — leaving records without vectors"
            );
            return Ok(());
        };

        let mut batch_records: Vec<EmbedRecord> = Vec::with_capacity(to_embed.len());
        for &idx in &to_embed {
            let rec = &records[idx];
            let text = Self::extract_record_text(rec).unwrap_or_default();
            batch_records.push(EmbedRecord {
                id: rec.oid.clone(),
                text,
                tenant_id: tenant_id
                    .map(str::to_string)
                    .unwrap_or_else(|| rec.tenant_id.clone()),
            });
        }

        let batch = EmbedBatch {
            records: batch_records,
            mode: IngestMode::Async,
        };
        let result = service
            .embed_sync(batch)
            .await
            .map_err(|e| anyhow::anyhow!("embedding dispatch failed: {}", e))?;

        // Defensive: vectors.len() must match to_embed.len() per the contract.
        if result.vectors.len() != to_embed.len() {
            anyhow::bail!(
                "embedding returned {} vectors for {} records",
                result.vectors.len(),
                to_embed.len()
            );
        }

        let dim = result.route.dimension() as u32;
        for (slot, vector) in to_embed.iter().zip(result.vectors) {
            records[*slot].embeddings.push(EmbeddingCell {
                model_id: "native".to_string(),
                modality: "dense_vector".to_string(),
                dim,
                values: proximadb_records::EmbeddingValues::Fp32(vector),
                ..Default::default()
            });
        }
        Ok(())
    }

    pub(crate) fn extract_record_text(record: &ProximaRecord) -> Option<String> {
        for key in ["text", "body", "title"] {
            if let Some(ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(s))) =
                record.props.get(key)
            {
                return Some(s.clone());
            }
        }
        None
    }

    async fn records_for_write_batches(
        catalog_manager: Option<&Arc<CatalogManager>>,
        table_fqn: Option<&str>,
        operation: FlightWriteOperation,
        write_mode: WriteMode,
        batches: &[arrow_array::RecordBatch],
    ) -> Result<(Vec<ProximaRecord>, Option<CatalogBulkWriteResult>)> {
        if let (Some(catalog_manager), Some(table_fqn)) = (catalog_manager, table_fqn) {
            let service = CatalogBulkWriteService::with_defaults(catalog_manager.clone());
            let (records, result) = service
                .prepare_bulk_write(
                    table_fqn,
                    batches,
                    Self::catalog_bulk_write_mode(operation, write_mode),
                )
                .await?;
            return Ok((records, Some(result)));
        }

        ArrowProtoCodec::batches_to_proxima_records(batches.to_vec()).map(|records| (records, None))
    }

    /// Convert and commit one Arrow write batch.
    async fn handle_record_write_batch(
        &self,
        collection_id: &str,
        table_fqn: Option<&str>,
        operation: FlightWriteOperation,
        write_mode: WriteMode,
        tenant_id: Option<&str>,
        batch: arrow_array::RecordBatch,
        insert_seen_ids: Option<&mut HashSet<String>>,
    ) -> Result<BatchOperationResult> {
        let batch_rows = batch.num_rows();
        debug!(
            collection_id = %collection_id,
            write_mode = ?write_mode,
            tenant_id = ?tenant_id,
            batch_rows = batch_rows,
            "Arrow IPC write batch"
        );

        // Convert Arrow batches to canonical ProximaRecord envelopes so rich
        // scalar fields and modality columns survive the Flight boundary.
        // Relational/table descriptors additionally validate through xCatalog.
        let batches = [batch];
        let (mut records, catalog_result) = Self::records_for_write_batches(
            self.catalog_manager.as_ref(),
            table_fqn,
            operation,
            write_mode,
            &batches,
        )
        .await?;

        // Phase 1 native-embedding intercept (Approach A inline).
        //
        // When the Flight client sent a text-only schema variant (vector column
        // absent), records arrive with empty `embeddings`. Dispatch them through
        // the EmbeddingService singleton — same process, Arc-shared model, two-tier
        // priority scheduler — and populate the vector before WAL/index commit.
        // Records that already carry an embedding bypass embedding entirely.
        //
        // Approach B (true async via WAL pending_embed flag + background drainer)
        // is wired by adding a header-driven branch here; the WAL field and
        // catalog nullable variant are already in place.
        Self::embed_text_only_records(&mut records, tenant_id).await?;

        info!(
            collection_id = %collection_id,
            table_fqn = ?table_fqn,
            records = records.len(),
            "Converted Arrow batch to ProximaRecords"
        );
        if let Some(catalog_result) = catalog_result {
            debug!(
                table_created = catalog_result.table_created,
                schema_evolved = catalog_result.schema_evolved,
                records_prepared = catalog_result.records_written,
                "Arrow Flight catalog bulk write preparation completed"
            );
        }

        if operation == FlightWriteOperation::Insert {
            let mut local_seen_ids;
            let seen_ids = match insert_seen_ids {
                Some(seen_ids) => seen_ids,
                None => {
                    local_seen_ids = HashSet::with_capacity(records.len());
                    &mut local_seen_ids
                }
            };
            if let Some(result) = Self::insert_conflict_result(&records, seen_ids) {
                return Ok(result);
            }
        }

        // Build shared WriteIntent so all Flight operations route through the
        // canonical lane contract (ADR-009/ADR-010/blueprint).
        let op_kind = crate::services::WriteOperationKind::from(operation);
        let durability = crate::services::WriteDurabilityRequirement::from(write_mode);
        let intent = crate::services::WriteIntent::new(collection_id, op_kind)
            .with_durability(durability)
            .with_row_count_hint(records.len() as u64);
        let lane_decision = crate::services::WriteLaneRouter::new().route(&intent);

        debug!(
            collection_id = %collection_id,
            operation = ?op_kind,
            write_lane = ?lane_decision.lane,
            guards = ?lane_decision.required_guards,
            "Arrow Flight write-lane decision"
        );
        // BulkAppendCommit and OverwriteSnapshotCommit are not yet wired to a
        // direct segment/manifest commit path; reject them explicitly here
        // rather than silently falling through to WAL (T16 will add that path).
        lane_decision.require_wal_lane("Arrow Flight DoPut")?;

        // Route to WAL-current-state or bulk-append based on lane decision.
        // BulkAppendCommit defers to WAL while direct-commit is not yet wired.
        let result = if operation == FlightWriteOperation::Insert {
            self.record_port
                .insert_record_batch(collection_id, records, tenant_id)
                .await?
        } else {
            self.record_port
                .upsert_record_batch(collection_id, records, tenant_id)
                .await?
        };

        Ok(result)
    }

    async fn handle_record_delete_batch(
        &self,
        collection_id: &str,
        tenant_id: Option<&str>,
        batch: arrow_array::RecordBatch,
    ) -> Result<BatchOperationResult> {
        let batch_rows = batch.num_rows();
        debug!(
            collection_id = %collection_id,
            tenant_id = ?tenant_id,
            batch_rows = batch_rows,
            "Arrow IPC delete batch"
        );

        let record_ids = ArrowProtoCodec::batches_to_record_ids(vec![batch])?;

        self.record_port
            .delete_record_batch(collection_id, record_ids, tenant_id)
            .await
    }

    /// Canonical v2 vector search (TD-FLIGHT-1). Routes through the
    /// `RecordSearchPort` — the same `RecordOpsService::handle_record_search_for_tenant`
    /// authority REST v2 and gRPC v2 use — so Flight inherits typed filters,
    /// WAL delta-merge, MVCC/tombstone filtering, Strong-freshness cache
    /// behavior, and the tenant-collection-access check. Wrapped in the same
    /// query-scoped I/O-trace boundary as REST v2 (route
    /// `arrow_flight.v2.records.search`) so object GETs/bytes stay attributable
    /// per query. The tenant stable id (TD-TENANT-1) is resolved via the wired
    /// `TenantStableIdResolver` when present, else `None`.
    async fn handle_v2_search(
        &self,
        ticket: FlightSearchTicket,
        identity: proximadb_runtime::PortIdentity<'_>,
    ) -> Result<Vec<arrow_array::RecordBatch>> {
        let include_vector = ticket.include_vector;
        let request = ticket.to_rich_request()?;
        debug!(
            collection_id = %request.collection_id,
            top_k = request.top_k,
            include_vector,
            "Arrow Flight canonical v2 vector search"
        );

        // ADR-087: the stable id arrives on the identity, stamped ONCE by the
        // identity orchestrator — no per-handler re-resolution.
        let response = crate::observability::io_trace::instrument_with_stable_tenant(
            identity.tenant_id.map(str::to_string),
            identity.tenant_stable_id,
            "arrow_flight.v2.records.search",
            crate::observability::predicate_diagnostics::scope(async {
                self.record_search_port
                    .search_record(request, identity)
                    .await
            }),
        )
        .await?;

        // Always emit one batch of the canonical v2 schema — even when empty
        // (acceptance: empty results / optional vectors must not panic).
        let batch =
            ArrowProtoCodec::rich_search_results_to_batch(&response.results, include_vector)?;
        Ok(vec![batch])
    }
}

#[tonic::async_trait]
impl FlightService for ProximaFlightService {
    type HandshakeStream = TonicStream<HandshakeResponse>;
    type ListFlightsStream = TonicStream<FlightInfo>;
    type DoGetStream = TonicStream<FlightData>;
    type DoPutStream = TonicStream<PutResult>;
    type DoActionStream = TonicStream<arrow_flight::Result>;
    type ListActionsStream = TonicStream<ActionType>;
    type DoExchangeStream = TonicStream<FlightData>;

    async fn handshake(
        &self,
        _request: TonicRequest<TonicStreaming<HandshakeRequest>>,
    ) -> TonicResult<Self::HandshakeStream> {
        // No authentication for now (future enhancement)
        Ok(TonicResponse::new(Box::pin(stream::empty())))
    }

    async fn list_flights(
        &self,
        request: TonicRequest<Criteria>,
    ) -> TonicResult<Self::ListFlightsStream> {
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let tenant_context =
            crate::storage::tenant::TenantContext::for_tenant_id(&auth_context.tenant_id);
        let criteria = request.into_inner();

        // Parse collection_id from criteria expression
        // Format: collection_id as bytes, or JSON with collection_id field
        let collection_id = if criteria.expression.is_empty() {
            // List all collections
            None
        } else {
            // Try to parse as UTF-8 collection ID
            String::from_utf8(criteria.expression.to_vec()).ok()
        };

        debug!(collection_id = ?collection_id, "list_flights request");

        // Get collections to list
        let collections = if let Some(cid) = collection_id {
            // Get specific collection
            match self
                .collection_service
                .get_collection_with_tenant_context(&cid, Some(&tenant_context))
                .await
            {
                Ok(Some(c)) => vec![c],
                Ok(None) => {
                    return Err(TonicStatus::not_found(format!(
                        "Collection not found: {}",
                        cid
                    )));
                }
                Err(e) => {
                    return Err(TonicStatus::internal(format!(
                        "Failed to get collection: {}",
                        e
                    )));
                }
            }
        } else {
            // List all collections
            self.collection_service
                .list_collections_with_tenant_context(Some(&tenant_context))
                .await
                .map_err(|e| TonicStatus::internal(format!("Failed to list collections: {}", e)))?
        };

        // Build FlightInfo for each collection's arrow files
        let mut flight_infos = Vec::new();
        for collection in collections {
            let files = self
                .file_export_handler
                .list_arrow_files(&collection, None, Some(100))
                .map_err(|e| TonicStatus::internal(format!("Failed to list arrow files: {}", e)))?;

            if !files.is_empty() {
                let flight_info = self
                    .file_export_handler
                    .create_flight_info(&collection, &files, "grpc://localhost:5680")
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to create flight info: {}", e))
                    })?;
                flight_infos.push(flight_info);
            }
        }

        info!(
            num_flights = flight_infos.len(),
            "Returning list of available Arrow/Parquet file flights"
        );

        let stream = stream::iter(flight_infos.into_iter().map(Ok));
        Ok(TonicResponse::new(Box::pin(stream)))
    }

    async fn get_flight_info(
        &self,
        request: TonicRequest<FlightDescriptor>,
    ) -> TonicResult<FlightInfo> {
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let tenant_context =
            crate::storage::tenant::TenantContext::for_tenant_id(&auth_context.tenant_id);
        let descriptor = request.into_inner();

        // Parse request from descriptor
        let file_request = ArrowFileRequest::from_descriptor(&descriptor)
            .map_err(|e| TonicStatus::invalid_argument(format!("Invalid descriptor: {}", e)))?;

        debug!(
            collection_id = %file_request.collection_id,
            file_pattern = ?file_request.file_pattern,
            "get_flight_info request"
        );

        // Get collection
        let collection = self
            .collection_service
            .get_collection_with_tenant_context(&file_request.collection_id, Some(&tenant_context))
            .await
            .map_err(|e| TonicStatus::internal(format!("Failed to get collection: {}", e)))?
            .ok_or_else(|| {
                TonicStatus::not_found(format!(
                    "Collection not found: {}",
                    file_request.collection_id
                ))
            })?;

        // List arrow files matching pattern
        let files = self
            .file_export_handler
            .list_arrow_files(
                &collection,
                file_request.file_pattern.as_deref(),
                file_request.limit,
            )
            .map_err(|e| TonicStatus::internal(format!("Failed to list arrow files: {}", e)))?;

        info!(
            collection_id = %file_request.collection_id,
            num_files = files.len(),
            "Found Arrow/Parquet files for export"
        );

        // Create FlightInfo with endpoints for each file
        self.file_export_handler
            .create_flight_info(&collection, &files, "grpc://localhost:5680")
            .map_err(|e| TonicStatus::internal(format!("Failed to create flight info: {}", e)))
            .map(TonicResponse::new)
    }

    async fn poll_flight_info(
        &self,
        _request: TonicRequest<FlightDescriptor>,
    ) -> TonicResult<arrow_flight::PollInfo> {
        Err(TonicStatus::unimplemented(
            "poll_flight_info not implemented",
        ))
    }

    async fn get_schema(
        &self,
        request: TonicRequest<FlightDescriptor>,
    ) -> TonicResult<SchemaResult> {
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let tenant_context =
            crate::storage::tenant::TenantContext::for_tenant_id(&auth_context.tenant_id);
        let descriptor = request.into_inner();

        if let Some(schema) =
            Self::catalog_arrow_schema_for_descriptor(self.catalog_manager.as_ref(), &descriptor)
                .await
                .map_err(|e| {
                    TonicStatus::internal(format!("Failed to resolve catalog schema: {}", e))
                })?
        {
            let result = Self::schema_result_from_arrow_schema(&schema)
                .map_err(|e| TonicStatus::internal(e.to_string()))?;
            return Ok(TonicResponse::new(result));
        }

        // Parse collection_id from descriptor
        let metadata = ArrowProtoCodec::parse_descriptor(&descriptor)
            .map_err(|e| TonicStatus::invalid_argument(format!("Invalid descriptor: {}", e)))?;

        // Get collection to determine dimension
        let collection = self
            .collection_service
            .get_collection_with_tenant_context(&metadata.collection_id, Some(&tenant_context))
            .await
            .map_err(|e| TonicStatus::internal(format!("Failed to get collection: {}", e)))?
            .ok_or_else(|| TonicStatus::not_found("Collection not found"))?;

        let dimension = collection
            .config
            .as_ref()
            .map(|c| c.dimension)
            .ok_or_else(|| TonicStatus::internal("Collection config missing"))?;

        let schema = ArrowProtoCodec::create_vector_schema(dimension as usize);

        let result = Self::schema_result_from_arrow_schema(&schema)
            .map_err(|e| TonicStatus::internal(e.to_string()))?;
        Ok(TonicResponse::new(result))
    }

    async fn do_get(&self, request: TonicRequest<Ticket>) -> TonicResult<Self::DoGetStream> {
        // KOU result-egress: classify the client's peer IP once for this request
        // (no remote addr ⇒ unspecified ⇒ Local/free). Drives both the egress
        // meter and the egress-aware shaping decision below.
        let edge = crate::metrics::consumption_metrics::EdgePolicyContext::classify(
            request
                .remote_addr()
                .map(|a| a.ip())
                .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)),
        );
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let ticket = request.into_inner();

        // Check if this is an arrow file export ticket
        if ArrowFileTicket::is_arrow_file_ticket(&ticket) {
            if auth_context.capability.is_some() {
                return Err(TonicStatus::permission_denied(
                    "Capability tokens cannot export Arrow files",
                ));
            }
            let file_ticket = ArrowFileTicket::from_ticket(&ticket).map_err(|e| {
                TonicStatus::invalid_argument(format!("Failed to parse file ticket: {}", e))
            })?;

            // Get compression setting from ticket
            let compression = file_ticket
                .compression
                .unwrap_or(FlightCompression::None)
                .to_arrow_compression();

            debug!(
                collection_id = %file_ticket.collection_id,
                compression = ?compression,
                "Arrow file export via do_get"
            );

            // Stream Arrow file contents
            let collection_id = file_ticket.collection_id.clone();
            let batches = self
                .handle_arrow_file_export(file_ticket)
                .await
                .map_err(|error| {
                    warn!(%collection_id, %error, "Arrow file export rejected");
                    TonicStatus::not_found("Export file is unavailable")
                })?;

            // Convert batches to FlightData stream with compression
            // Use the new compression-aware encoder for all batches
            let flight_data =
                ArrowProtoCodec::batches_to_flight_data_with_compression(&batches, compression)
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to encode batches: {}", e))
                    })?;

            // KOU result-egress: meter the actual encoded FlightData bytes (the
            // client picked the compression via the ticket).
            edge.record_result_egress(
                Some(auth_context.tenant_id.as_str()),
                flight_data_wire_bytes(&flight_data),
            );

            let stream = stream::iter(flight_data.into_iter().map(Ok));

            return Ok(TonicResponse::new(Box::pin(stream)));
        }

        // Batched columnar graph export: a graph ticket reads nodes/edges from
        // the live graph engine and streams them, paginated, as columnar Arrow
        // batches — the server never materializes the whole result set.
        if let Some(graph_ticket) = Self::parse_graph_ticket(&ticket) {
            Self::validate_flight_search_capability(
                auth_context.capability.as_ref(),
                &graph_ticket.graph_id,
            )?;
            let stream = self
                .graph_export_flight_stream(
                    graph_ticket,
                    Some(auth_context.tenant_id.clone()),
                    edge,
                )
                .await?;
            return Ok(TonicResponse::new(stream));
        }

        // Canonical v2 vector search (TD-FLIGHT-1): a self-describing JSON
        // ticket discriminated by "type":"vector_search". This replaces the
        // deprecated v1 `VectorSearchRequest` fallback — Flight search now runs
        // the same canonical search authority (RecordSearchPort) as REST v2 /
        // gRPC v2, with full-fidelity props and the tenant-collection-access
        // check.
        if let Some(search_ticket) = FlightSearchTicket::from_ticket(&ticket) {
            Self::validate_flight_search_capability(
                auth_context.capability.as_ref(),
                &search_ticket.collection_id,
            )?;

            let batches = self
                .handle_v2_search(
                    search_ticket,
                    proximadb_runtime::PortIdentity::from(&auth_context),
                )
                .await
                .map_err(|e| TonicStatus::internal(format!("Search failed: {}", e)))?;

            // Egress-aware shaping (co-design D2): for a chargeable (far)
            // client, encode the result batches with ZSTD — a lossless
            // byte-minimization the Arrow reader decompresses transparently.
            // Near/free clients stay uncompressed (save CPU). Then meter the
            // ACTUAL encoded bytes so the bill reflects the compressed egress.
            let compression = if edge.shape_policy().compress {
                FlightCompression::Zstd.to_arrow_compression()
            } else {
                None
            };
            let flight_data =
                ArrowProtoCodec::batches_to_flight_data_with_compression(&batches, compression)
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to encode batches: {}", e))
                    })?;
            edge.record_result_egress(
                Some(auth_context.tenant_id.as_str()),
                flight_data_wire_bytes(&flight_data),
            );
            let stream = stream::iter(flight_data.into_iter().map(Ok));
            return Ok(TonicResponse::new(Box::pin(stream)));
        }

        // Unknown ticket shape — no longer silently coerced into v1 search.
        Err(TonicStatus::invalid_argument(
            "Unrecognized Flight DoGet ticket: expected an arrow_file, graph, \
             or vector_search (type:\"vector_search\") ticket",
        ))
    }

    async fn do_put(
        &self,
        request: TonicRequest<TonicStreaming<FlightData>>,
    ) -> TonicResult<Self::DoPutStream> {
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let mut stream = request.into_inner();

        // First message should contain descriptor
        let first_msg = stream
            .message()
            .await
            .map_err(|e| TonicStatus::internal(format!("Failed to read first message: {}", e)))?
            .ok_or_else(|| TonicStatus::invalid_argument("Empty stream"))?;

        // Parse descriptor
        let descriptor = first_msg
            .flight_descriptor
            .clone()
            .ok_or_else(|| TonicStatus::invalid_argument("Missing FlightDescriptor"))?;

        let metadata = ArrowProtoCodec::parse_descriptor(&descriptor)
            .map_err(|e| TonicStatus::invalid_argument(format!("Invalid descriptor: {}", e)))?;
        let table_fqn = Self::table_fqn_from_descriptor(&descriptor).map_err(|e| {
            TonicStatus::invalid_argument(format!("Invalid table descriptor: {}", e))
        })?;
        let write_target = table_fqn
            .clone()
            .unwrap_or_else(|| metadata.collection_id.clone());

        let operation = metadata.operation;
        Self::validate_flight_write_capability(
            auth_context.capability.as_ref(),
            &write_target,
            operation,
            0,
        )?;

        // Slice 6.2: primary-pod write-router gate. Runs AFTER auth +
        // capability validation (so a malformed/unauthorized request
        // still gets the right error) and BEFORE the streaming write
        // loop (so a misroute never touches the WAL on this pod).
        // Covers Insert/Upsert/Delete — all mutations participate;
        // pure reads use do_get and don't pass through here.
        check_flight_primary_pod_gate(
            &self.primary_pod_gate,
            &auth_context.tenant_id,
            &write_target,
        )?;

        let mut batch_stream = Self::record_batch_stream(first_msg, stream);
        let mut total_rows = 0usize;
        let mut total_batches = 0u64;
        let mut result = Self::empty_batch_result();
        let mut insert_seen_ids = HashSet::new();

        while let Some(batch) = batch_stream.next().await {
            let batch = batch.map_err(|e| TonicStatus::internal(format!("Stream error: {}", e)))?;
            let batch_rows = batch.num_rows();
            total_rows += batch_rows;
            total_batches += 1;

            Self::validate_flight_write_capability(
                auth_context.capability.as_ref(),
                &write_target,
                operation,
                total_rows,
            )?;

            let batch_result = match operation {
                FlightWriteOperation::Upsert | FlightWriteOperation::Insert => {
                    self.handle_record_write_batch(
                        &write_target,
                        table_fqn.as_deref(),
                        operation,
                        metadata.write_mode,
                        Some(auth_context.tenant_id.as_str()),
                        batch,
                        (operation == FlightWriteOperation::Insert).then_some(&mut insert_seen_ids),
                    )
                    .await
                }
                FlightWriteOperation::Delete => {
                    self.handle_record_delete_batch(
                        &write_target,
                        Some(auth_context.tenant_id.as_str()),
                        batch,
                    )
                    .await
                }
            }
            .map_err(|e| TonicStatus::internal(format!("{} failed: {}", operation.as_str(), e)))?;
            Self::merge_batch_result(&mut result, batch_result);
        }

        debug!(
            collection_id = %metadata.collection_id,
            total_batches = total_batches,
            total_rows = total_rows,
            "Processed Flight DoPut stream"
        );

        if metadata.trigger_compaction {
            info!(
                collection_id = %write_target,
                "Arrow Flight: triggering compaction after DoPut"
            );
            self.trigger_collection_compaction(&write_target)
                .await
                .map_err(|e| TonicStatus::internal(format!("Compaction failed: {}", e)))?;
        }

        // Return rich batch result metadata to Flight clients.
        let result_bytes = Self::batch_result_app_metadata(&result)
            .map_err(|e| TonicStatus::internal(e.to_string()))?;

        let put_result = PutResult {
            app_metadata: result_bytes.into(),
        };

        Ok(TonicResponse::new(
            Box::pin(stream::once(async move { Ok(put_result) })) as Self::DoPutStream,
        ))
    }

    async fn do_action(&self, request: TonicRequest<Action>) -> TonicResult<Self::DoActionStream> {
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let tenant_context =
            crate::storage::tenant::TenantContext::for_tenant_id(&auth_context.tenant_id);
        let action = request.into_inner();

        match action.r#type.as_str() {
            // Collection operations
            "create_collection" => {
                // Body: {"name": "...", "dimension": 768, "engine": "sst", "distance_metric": "cosine"}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| TonicStatus::invalid_argument("Missing 'name' field"))?;

                let dimension = params
                    .get("dimension")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| TonicStatus::invalid_argument("Missing 'dimension' field"))?
                    as u32;

                let engine = params
                    .get("engine")
                    .and_then(|v| v.as_str())
                    .unwrap_or("sst");

                let distance_metric = params
                    .get("distance_metric")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cosine");

                // Read canonical_embedding_precision from the action body so
                // Arrow Flight clients can opt their collections into
                // non-fp32 storage. Accept the same string-or-int format
                // the REST `apply_proto_enum_workarounds` does so SDKs see
                // consistent semantics across protocols.
                let canonical_embedding_precision: Option<i32> = match params
                    .get("canonical_embedding_precision")
                {
                    Some(serde_json::Value::String(s)) => {
                        use crate::proto::proximadb_v1::EmbeddingPrecision;
                        let key = s.to_ascii_lowercase();
                        let stripped = key.strip_prefix("embedding_precision_").unwrap_or(&key);
                        match stripped {
                            "unspecified" => Some(EmbeddingPrecision::Unspecified as i32),
                            "fp32" | "f32" | "float32" => Some(EmbeddingPrecision::Fp32 as i32),
                            "fp16" | "f16" | "float16" | "half" => {
                                Some(EmbeddingPrecision::Fp16 as i32)
                            }
                            "bf16" | "bfloat16" => Some(EmbeddingPrecision::Bf16 as i32),
                            "int8" | "i8" | "int8_scalar" => Some(EmbeddingPrecision::Int8 as i32),
                            "uint8" | "u8" | "uint8_scalar" => {
                                Some(EmbeddingPrecision::Uint8 as i32)
                            }
                            _ => None,
                        }
                    }
                    Some(serde_json::Value::Number(n)) => n.as_i64().map(|v| v as i32),
                    _ => None,
                };

                info!(
                    name = %name,
                    dimension = dimension,
                    engine = %engine,
                    precision = ?canonical_embedding_precision,
                    "Arrow Flight: create_collection"
                );

                // Build collection config with correct proto structure
                let storage_engine = match engine {
                    "helix" => crate::proto::proximadb_v1::StorageEngine::Helix,
                    "viper" => crate::proto::proximadb_v1::StorageEngine::Viper,
                    "swift" => crate::proto::proximadb_v1::StorageEngine::Swift,
                    "nova" => crate::proto::proximadb_v1::StorageEngine::Nova,
                    "raptor" => crate::proto::proximadb_v1::StorageEngine::Raptor,
                    "tst" => crate::proto::proximadb_v1::StorageEngine::Tst,
                    _ => crate::proto::proximadb_v1::StorageEngine::Sst,
                };

                let distance_metric_enum = match distance_metric {
                    "euclidean" | "l2" => crate::proto::proximadb_v1::DistanceMetric::Euclidean,
                    "dot" | "dot_product" => crate::proto::proximadb_v1::DistanceMetric::DotProduct,
                    _ => crate::proto::proximadb_v1::DistanceMetric::Cosine,
                };

                let config = crate::proto::proximadb_v1::CollectionConfig {
                    name: name.to_string(),
                    embedding_config: None,
                    dimension,
                    distance_metric: Some(distance_metric_enum as i32),
                    storage_engine: Some(storage_engine as i32),
                    tags: vec![],
                    description: None,
                    filterable_columns: vec![],
                    index_configs: vec![],
                    quantization: None,
                    storage_config: None,
                    primary_index: None,
                    auto_index_selection: None,
                    owner: None,
                    embedding_models: vec![],
                    record_schema: None,
                    enable_proxima_record: None,
                    text_columns: vec![],
                    text_storage_configs: vec![],
                    enable_dual_use_embeddings: None,
                    canonical_embedding_precision,
                    permitted_principals: vec![],
                    // Arrow Flight create does not expose the routing policy (REST
                    // is the typed surface); default to auto (None).
                    index_policy: None,
                    pax_vector_quant: None,
                };

                // Create collection via service
                let result = self
                    .collection_service
                    .create_collection_with_tenant_context(&config, Some(&tenant_context))
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to create collection: {}", e))
                    })?;

                // Extract collection ID from response
                let collection_id = result
                    .collection
                    .as_ref()
                    .map(|c| c.id.clone())
                    .unwrap_or_default();

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": result.success,
                    "collection_id": collection_id,
                    "name": name,
                    "storage_path": result.storage_path
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "delete_collection" => {
                // Body: {"collection_id": "..."} or {"name": "..."}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .or_else(|| params.get("name"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' or 'name' field")
                    })?;

                info!(collection_id = %collection_id, "Arrow Flight: delete_collection");

                // Delete collection via service
                self.collection_service
                    .delete_collection_with_tenant_context(collection_id, Some(&tenant_context))
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to delete collection: {}", e))
                    })?;

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "deleted": collection_id
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "get_collection" => {
                // Body: {"collection_id": "..."} or {"name": "..."}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .or_else(|| params.get("name"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' or 'name' field")
                    })?;

                debug!(collection_id = %collection_id, "Arrow Flight: get_collection");

                // Get collection via service
                let collection = self
                    .collection_service
                    .get_collection_with_tenant_context(collection_id, Some(&tenant_context))
                    .await
                    .map_err(|e| TonicStatus::internal(format!("Failed to get collection: {}", e)))?
                    .ok_or_else(|| {
                        TonicStatus::not_found(format!("Collection not found: {}", collection_id))
                    })?;

                // Extract name from config
                let name = collection
                    .config
                    .as_ref()
                    .map(|c| c.name.clone())
                    .unwrap_or_default();

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "collection": {
                        "id": collection.id,
                        "name": name,
                        "config": collection.config
                    }
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "list_collections" => {
                debug!("Arrow Flight: list_collections");

                // List all collections via service
                let collections = self
                    .collection_service
                    .list_collections_with_tenant_context(Some(&tenant_context))
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to list collections: {}", e))
                    })?;

                let collection_summaries: Vec<serde_json::Value> = collections
                    .iter()
                    .map(|c| {
                        let name = c
                            .config
                            .as_ref()
                            .map(|cfg| cfg.name.clone())
                            .unwrap_or_default();
                        serde_json::json!({
                            "id": c.id,
                            "name": name,
                            "dimension": c.config.as_ref().map_or(0, |cfg| cfg.dimension)
                        })
                    })
                    .collect();

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "count": collections.len(),
                    "collections": collection_summaries
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            // Vector operations
            "insert_vectors" => {
                // Body: {"collection_id": "...", "vectors": [...]}
                // Vectors format: [{"id": "...", "vector": [...], "metadata": {...}}, ...]
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' field")
                    })?;

                let vectors_json = params
                    .get("vectors")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| TonicStatus::invalid_argument("Missing 'vectors' array"))?;

                info!(
                    collection_id = %collection_id,
                    vector_count = vectors_json.len(),
                    "Arrow Flight: insert_vectors"
                );

                // Build canonical ProximaRecord envelopes at the protocol boundary.
                let now_ns = Utc::now().timestamp_nanos_opt().unwrap_or(0);
                let mut records = Vec::with_capacity(vectors_json.len());
                for v in vectors_json {
                    let oid = v
                        .get("id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();

                    let values: Vec<f32> = v
                        .get("vector")
                        .and_then(|x| x.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect()
                        })
                        .unwrap_or_default();
                    let dim = values.len() as u32;

                    let mut props = proximadb_records::ProximaTree::new();
                    if let Some(meta) = v.get("metadata").and_then(|x| x.as_object()) {
                        for (k, val) in meta {
                            let pv = match val {
                                serde_json::Value::String(s) => {
                                    proximadb_data_model::ProximaValue::String(s.clone())
                                }
                                serde_json::Value::Number(n) => {
                                    if let Some(i) = n.as_i64() {
                                        proximadb_data_model::ProximaValue::Int64(i)
                                    } else {
                                        proximadb_data_model::ProximaValue::Float64(
                                            n.as_f64().unwrap_or(0.0),
                                        )
                                    }
                                }
                                serde_json::Value::Bool(b) => {
                                    proximadb_data_model::ProximaValue::Boolean(*b)
                                }
                                _ => proximadb_data_model::ProximaValue::String(val.to_string()),
                            };
                            props.insert(k.clone(), proximadb_records::ProximaTreeNode::Value(pv));
                        }
                    }

                    records.push(ProximaRecord {
                        oid,
                        embeddings: vec![proximadb_records::EmbeddingCell {
                            model_id: "default".to_string(),
                            modality: "vector".to_string(),
                            dim,
                            values: proximadb_records::EmbeddingValues::Fp32(values),
                            ..Default::default()
                        }],
                        props,
                        created_at_ns: now_ns,
                        updated_at_ns: now_ns,
                        ..Default::default()
                    });
                }

                // Insert via the canonical rich-record port.
                let result = self
                    .record_port
                    .upsert_record_batch(
                        collection_id,
                        records,
                        Some(auth_context.tenant_id.as_str()),
                    )
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to insert vectors: {}", e))
                    })?;

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": result.success,
                    "inserted_count": result.metrics.successful_count,
                    "vector_ids": result.vector_ids,
                    "error_message": result.errors.first(),
                    "error_code": result.error_code
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "delete_vectors" => {
                // Body: {"collection_id": "...", "vector_ids": ["id1", "id2", ...]}
                // Note: Vector deletion is implemented via WAL tombstone markers
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' field")
                    })?;

                let vector_ids: Vec<String> = params
                    .get("vector_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                info!(
                    collection_id = %collection_id,
                    vector_count = vector_ids.len(),
                    "Arrow Flight: delete_vectors"
                );

                // Vector deletion is handled via soft-delete (tombstone markers)
                // For now, this returns the count of requested deletions
                // Actual deletion happens during compaction
                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "collection_id": collection_id,
                    "requested_deletions": vector_ids.len(),
                    "note": "Vectors marked for deletion. Run compact_collection to reclaim space."
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "get_vectors" => {
                // Body: {"collection_id": "...", "vector_ids": ["id1", "id2", ...], "include_vectors": true, "include_metadata": true}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' field")
                    })?;

                let vector_ids: Vec<String> = params
                    .get("vector_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                let include_vectors = params
                    .get("include_vectors")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                let include_metadata = params
                    .get("include_metadata")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                debug!(
                    collection_id = %collection_id,
                    vector_count = vector_ids.len(),
                    "Arrow Flight: get_vectors"
                );

                self.collection_service
                    .get_collection_with_tenant_context(collection_id, Some(&tenant_context))
                    .await
                    .map_err(|e| TonicStatus::internal(format!("Failed to get collection: {}", e)))?
                    .ok_or_else(|| {
                        TonicStatus::not_found(format!("Collection not found: {}", collection_id))
                    })?;

                // Get vectors via vector operations service
                let mut found_vectors = Vec::new();
                for vector_id in &vector_ids {
                    if let Ok(Some(record)) = self
                        .vector_operations_service
                        .vector(collection_id, vector_id, include_vectors, include_metadata)
                        .await
                    {
                        found_vectors.push(serde_json::json!({
                            "id": &record.oid,
                            "oid": &record.oid,
                            // INT-2.5b: as_fp32_cow returns a Cow whose temporary
                            // would drop before json! consumes the borrow. Owned
                            // Vec<f32> avoids the lifetime hazard.
                            "vector": if include_vectors {
                                record.embeddings.first().map(|embedding| embedding.as_fp32_cow().into_owned())
                            } else {
                                None
                            },
                            "metadata": if include_metadata {
                                Some(&record.props)
                            } else {
                                None
                            },
                            "tenant_id": &record.tenant_id,
                            "created_at_ns": record.created_at_ns,
                            "updated_at_ns": record.updated_at_ns
                        }));
                    }
                }

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "found_count": found_vectors.len(),
                    "requested_count": vector_ids.len(),
                    "vectors": found_vectors
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            // Storage operations
            "flush_collection" => {
                // Body: {"collection_id": "..."}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' field")
                    })?;

                info!(collection_id = %collection_id, "Arrow Flight: flush_collection");

                // Flush collection via vector operations service
                self.vector_operations_service
                    .force_flush_collection(collection_id)
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to flush collection: {}", e))
                    })?;

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "collection_id": collection_id,
                    "operation": "flush"
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "compact_collection" => {
                // Body: {"collection_id": "..."}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' field")
                    })?;

                info!(collection_id = %collection_id, "Arrow Flight: compact_collection");

                self.trigger_collection_compaction(collection_id)
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to compact collection: {}", e))
                    })?;

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "collection_id": collection_id,
                    "operation": "compact"
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "flush_and_compact" => {
                // Body: {"collection_id": "..."}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).map_err(|e| {
                        TonicStatus::invalid_argument(format!("Invalid JSON body: {}", e))
                    })?;

                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing 'collection_id' field")
                    })?;

                info!(collection_id = %collection_id, "Arrow Flight: flush_and_compact");

                // Flush first, then compact
                self.vector_operations_service
                    .force_flush_collection(collection_id)
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to flush collection: {}", e))
                    })?;

                self.trigger_collection_compaction(collection_id)
                    .await
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to compact collection: {}", e))
                    })?;

                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "success": true,
                    "collection_id": collection_id,
                    "operation": "flush_and_compact"
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "list_arrow_files" => {
                // List Arrow files for a collection
                // Body should be JSON: {"collection_id": "..."}
                let params: serde_json::Value =
                    serde_json::from_slice(&action.body).unwrap_or_default();
                let collection_id = params
                    .get("collection_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TonicStatus::invalid_argument("Missing collection_id in action body")
                    })?;

                // Get collection
                let collection = self
                    .collection_service
                    .get_collection_with_tenant_context(collection_id, Some(&tenant_context))
                    .await
                    .map_err(|e| TonicStatus::internal(format!("Failed to get collection: {}", e)))?
                    .ok_or_else(|| {
                        TonicStatus::not_found(format!("Collection not found: {}", collection_id))
                    })?;

                // List arrow files
                let files = self
                    .file_export_handler
                    .list_arrow_files(&collection, None, Some(1000))
                    .map_err(|e| {
                        TonicStatus::internal(format!("Failed to list arrow files: {}", e))
                    })?;

                // Return as JSON results
                let result_bytes = serde_json::to_vec(&files)
                    .map_err(|e| TonicStatus::internal(format!("Failed to serialize: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            "health_check" => {
                // Simple health check action
                let result_bytes = serde_json::to_vec(&serde_json::json!({
                    "status": "healthy",
                    "service": "proximadb-arrow-flight",
                    "version": env!("CARGO_PKG_VERSION")
                }))
                .map_err(|e| TonicStatus::internal(format!("Failed to serialize result: {}", e)))?;

                let result = arrow_flight::Result {
                    body: result_bytes.into(),
                };

                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            // R-7c.4b: stream the rank pipeline's per-doc match_features
            // as Arrow IPC for offline LTR. See module
            // `network::arrow_ipc::rank_features_export` for the wire
            // contract and column schema.
            "rank_features_export" => {
                let services = self.rank_services.clone().ok_or_else(|| {
                    TonicStatus::unimplemented(
                        "rank_features_export: server started without RankServices injection",
                    )
                })?;
                let ipc_bytes = super::rank_features_export::export_rank_features_to_arrow_ipc(
                    &services,
                    &action.body,
                )
                .await
                .map_err(|e| {
                    use proximadb_rank_core::RankError;
                    match e {
                        RankError::ProfileNotFound(name) => TonicStatus::not_found(format!(
                            "rank_features_export: rank profile not found: {name}"
                        )),
                        RankError::InvalidProfile(msg) => {
                            TonicStatus::invalid_argument(format!("rank_features_export: {msg}"))
                        }
                        other => TonicStatus::internal(format!("rank_features_export: {other}")),
                    }
                })?;
                let result = arrow_flight::Result {
                    body: ipc_bytes.into(),
                };
                Ok(TonicResponse::new(Box::pin(stream::once(async move {
                    Ok(result)
                }))))
            }

            _ => Err(TonicStatus::unimplemented(format!(
                "Unknown action: {}. Supported actions: create_collection, delete_collection, get_collection, list_collections, insert_vectors, delete_vectors, get_vectors, flush_collection, compact_collection, flush_and_compact, list_arrow_files, rank_features_export, health_check",
                action.r#type
            ))),
        }
    }

    async fn list_actions(
        &self,
        _request: TonicRequest<Empty>,
    ) -> TonicResult<Self::ListActionsStream> {
        let actions = vec![
            // Collection operations
            ActionType {
                r#type: "create_collection".to_string(),
                description: "Create a new vector collection. Body: {name, dimension, engine?, distance_metric?}".to_string(),
            },
            ActionType {
                r#type: "delete_collection".to_string(),
                description: "Delete a collection. Body: {collection_id} or {name}".to_string(),
            },
            ActionType {
                r#type: "get_collection".to_string(),
                description: "Get collection details. Body: {collection_id} or {name}".to_string(),
            },
            ActionType {
                r#type: "list_collections".to_string(),
                description: "List all collections. Body: {} (empty)".to_string(),
            },
            // Vector operations
            ActionType {
                r#type: "insert_vectors".to_string(),
                description: "Insert/upsert vectors. Body: {collection_id, vectors: [{id, vector, metadata?}]}".to_string(),
            },
            ActionType {
                r#type: "delete_vectors".to_string(),
                description: "Mark vectors for deletion (soft delete). Body: {collection_id, vector_ids: [id1, id2, ...]}".to_string(),
            },
            // Streaming batch operations
            ActionType {
                r#type: "bulk_insert".to_string(),
                description: "DoExchange batch insert. Descriptor path: [bulk_insert, collection_id]; stream Arrow RecordBatches with id/oid, vector?, and rich props".to_string(),
            },
            ActionType {
                r#type: "bulk_upsert".to_string(),
                description: "DoExchange batch upsert. Descriptor path: [bulk_upsert, collection_id]; stream Arrow RecordBatches with id/oid, vector?, and rich props".to_string(),
            },
            ActionType {
                r#type: "bulk_delete".to_string(),
                description: "DoExchange batch delete. Descriptor path: [bulk_delete, collection_id]; stream Arrow RecordBatches with id or oid".to_string(),
            },
            ActionType {
                r#type: "get_vectors".to_string(),
                description: "Get vectors by ID. Body: {collection_id, vector_ids, include_vectors?, include_metadata?}".to_string(),
            },
            // Storage operations
            ActionType {
                r#type: "flush_collection".to_string(),
                description: "Flush a collection's WAL to storage engine. Body: {collection_id}".to_string(),
            },
            ActionType {
                r#type: "compact_collection".to_string(),
                description: "Compact a collection's storage files. Body: {collection_id}".to_string(),
            },
            ActionType {
                r#type: "flush_and_compact".to_string(),
                description: "Flush and compact a collection. Body: {collection_id}".to_string(),
            },
            // File operations
            ActionType {
                r#type: "list_arrow_files".to_string(),
                description: "List available .arrow and .parquet files in a collection for export. Body: {collection_id}".to_string(),
            },
            // Ranking framework (R-7c.4b)
            ActionType {
                r#type: "rank_features_export".to_string(),
                description: "Stream the multi-phase ranking pipeline's per-doc match_features as Arrow IPC for offline LTR training. Body: JSON RankSearchRequest (collection, query_vector, query_text?, k, rank_profile?, rank_overrides?). Response: single Arrow IPC stream with schema [id Utf8, rank UInt32, score Float32, phase UInt8, <feature_N> Float64 nullable per profile.match_features].".to_string(),
            },
            // Health check
            ActionType {
                r#type: "health_check".to_string(),
                description: "Check service health. Body: {} (empty)".to_string(),
            },
        ];

        let stream = stream::iter(actions.into_iter().map(Ok));
        Ok(TonicResponse::new(Box::pin(stream)))
    }

    /// DoExchange implements bidirectional streaming for large data transfers
    ///
    /// ## Supported Exchange Types
    ///
    /// The exchange type is determined by the first FlightData message which should
    /// contain a FlightDescriptor with the exchange operation:
    ///
    /// - **bulk_insert/bulk_upsert**: Stream large batches of records for insertion/upsert
    ///   - Descriptor path: ["bulk_insert", collection_id] or ["bulk_upsert", collection_id]
    ///   - Input: Stream of Arrow RecordBatches with vector data
    ///   - Output: Stream of progress updates and final result
    ///
    /// - **bulk_delete**: Stream record ids for deletion
    ///   - Descriptor path: ["bulk_delete", collection_id]
    ///   - Input: Stream of Arrow RecordBatches with `id` or `oid`
    ///   - Output: Stream of progress updates and final result
    ///
    /// - **bulk_search**: Execute multiple search queries in parallel
    ///   - Descriptor path: ["bulk_search", collection_id]
    ///   - Input: Stream of Arrow RecordBatches with query vectors
    ///   - Output: Stream of search results as Arrow RecordBatches
    ///
    /// - **data_transfer**: Large data transfer with progress tracking
    ///   - Descriptor path: ["data_transfer", collection_id]
    ///   - Input: Stream of arbitrary Arrow data
    ///   - Output: Stream of acknowledgments and progress
    ///
    /// ## Buffer Management
    ///
    /// For large transfers, the implementation:
    /// - Buffers incoming data in configurable chunk sizes
    /// - Processes data in parallel using Rayon
    /// - Sends progress updates during long operations
    /// - Uses backpressure to avoid memory exhaustion
    async fn do_exchange(
        &self,
        request: TonicRequest<TonicStreaming<FlightData>>,
    ) -> TonicResult<Self::DoExchangeStream> {
        let auth_context = self
            .authenticated_flight_context(request.metadata(), request.peer_certs())
            .await?;
        let mut stream = request.into_inner();

        // First message should contain the descriptor with exchange type
        let first_msg = stream
            .message()
            .await
            .map_err(|e| TonicStatus::internal(format!("Failed to read first message: {}", e)))?
            .ok_or_else(|| TonicStatus::invalid_argument("Empty stream - expected descriptor"))?;

        // Parse descriptor to determine exchange type
        let descriptor = first_msg.flight_descriptor.clone().ok_or_else(|| {
            TonicStatus::invalid_argument("First message must contain FlightDescriptor")
        })?;

        let (exchange_type, collection_id, write_operation) =
            Self::parse_exchange_descriptor(&descriptor)
                .map_err(|e| TonicStatus::invalid_argument(format!("Invalid descriptor: {}", e)))?;
        let table_fqn = Self::table_fqn_from_descriptor(&descriptor).map_err(|e| {
            TonicStatus::invalid_argument(format!("Invalid table descriptor: {}", e))
        })?;
        let write_target = table_fqn.clone().unwrap_or_else(|| collection_id.clone());

        info!(
            exchange_type = %exchange_type,
            collection_id = %write_target,
            table_fqn = ?table_fqn,
            "Arrow Flight: do_exchange initiated"
        );

        match exchange_type.as_str() {
            "bulk_insert" | "bulk_upsert" => {
                let operation = write_operation.unwrap_or_default();
                self.handle_bulk_write_exchange(
                    write_target,
                    table_fqn,
                    operation,
                    Some(auth_context.tenant_id.clone()),
                    auth_context.capability.clone(),
                    first_msg,
                    stream,
                )
                .await
            }
            "bulk_delete" => {
                self.handle_bulk_write_exchange(
                    write_target,
                    table_fqn,
                    FlightWriteOperation::Delete,
                    Some(auth_context.tenant_id.clone()),
                    auth_context.capability.clone(),
                    first_msg,
                    stream,
                )
                .await
            }
            "graph_nodes" | "graph_edges" => {
                // Bulk columnar graph ingest. Upsert semantics: the columnar
                // batch is idempotent re-ingestible. Capability + tenant
                // namespacing mirror the record write path.
                Self::validate_flight_write_capability(
                    auth_context.capability.as_ref(),
                    &collection_id,
                    FlightWriteOperation::Upsert,
                    0,
                )?;
                let is_nodes = exchange_type == "graph_nodes";
                self.handle_graph_write_exchange(
                    collection_id,
                    is_nodes,
                    Some(auth_context.tenant_id.clone()),
                    first_msg,
                    stream,
                )
                .await
            }
            "bulk_search" => {
                Self::validate_flight_search_capability(
                    auth_context.capability.as_ref(),
                    &collection_id,
                )?;
                self.handle_bulk_search_exchange(
                    collection_id,
                    auth_context.owned_identity(),
                    first_msg,
                    stream,
                )
                .await
            }
            "data_transfer" => {
                if auth_context.capability.is_some() {
                    return Err(TonicStatus::permission_denied(
                        "Capability tokens cannot use data_transfer exchange",
                    ));
                }
                self.handle_data_transfer_exchange(collection_id, first_msg, stream)
                    .await
            }
            _ => Err(TonicStatus::unimplemented(format!(
                "Unknown exchange type: {}. Supported: bulk_insert, bulk_upsert, bulk_delete, bulk_search, data_transfer, graph_nodes, graph_edges",
                exchange_type
            ))),
        }
    }
}

impl ProximaFlightService {
    /// Handle bulk write exchange - stream large batches for upsert/insert/delete.
    async fn handle_bulk_write_exchange(
        &self,
        collection_id: String,
        table_fqn: Option<String>,
        operation: FlightWriteOperation,
        tenant_id: Option<String>,
        capability: Option<DataPlaneCapability>,
        first_msg: FlightData,
        stream: TonicStreaming<FlightData>,
    ) -> TonicResult<<Self as FlightService>::DoExchangeStream> {
        let mut total_records = 0u64;
        let mut total_failed = 0u64;
        let mut total_batches = 0u64;
        let mut all_success = true;
        let mut results = Vec::new();
        let mut insert_seen_ids = HashSet::new();
        Self::validate_flight_write_capability(capability.as_ref(), &collection_id, operation, 0)?;

        // Primary-pod write-router gate. The DoExchange bulk write path
        // (bulk_insert/upsert/delete) must gate symmetrically with do_put — a
        // displaced pod streaming a bulk batch would otherwise land it in a
        // memtable the new primary's reader never sees. Gate once for the whole
        // exchange, after capability validation, before any storage work.
        check_flight_primary_pod_gate(
            &self.primary_pod_gate,
            tenant_id.as_deref().unwrap_or(""),
            &collection_id,
        )?;

        // Resolve write lane once for the full exchange stream: DoExchange always
        // uses WAL durability because streaming batches require per-row ordering.
        let op_kind = crate::services::WriteOperationKind::from(operation);
        let exchange_intent = crate::services::WriteIntent::new(&collection_id, op_kind)
            .with_durability(crate::services::WriteDurabilityRequirement::WalRequired)
            .with_row_count_hint(0);
        let exchange_lane = crate::services::WriteLaneRouter::new().route(&exchange_intent);
        debug!(
            collection_id = %collection_id,
            operation = ?op_kind,
            write_lane = ?exchange_lane.lane,
            "Arrow Flight DoExchange write-lane decision"
        );
        exchange_lane
            .require_wal_lane("Arrow Flight DoExchange")
            .map_err(|e| TonicStatus::invalid_argument(e.to_string()))?;

        let mut batch_stream = Self::record_batch_stream(first_msg, stream);
        let mut total_input_rows = 0usize;
        while let Some(batch) = batch_stream.next().await {
            let batch = batch.map_err(|e| TonicStatus::internal(format!("Stream error: {}", e)))?;
            let batch_rows = batch.num_rows();
            total_input_rows += batch_rows;
            total_batches += 1;
            Self::validate_flight_write_capability(
                capability.as_ref(),
                &collection_id,
                operation,
                total_input_rows,
            )?;

            let result = match operation {
                FlightWriteOperation::Upsert | FlightWriteOperation::Insert => {
                    self.handle_record_write_batch(
                        &collection_id,
                        table_fqn.as_deref(),
                        operation,
                        WriteMode::WAL,
                        tenant_id.as_deref(),
                        batch,
                        (operation == FlightWriteOperation::Insert).then_some(&mut insert_seen_ids),
                    )
                    .await
                }
                FlightWriteOperation::Delete => {
                    self.handle_record_delete_batch(&collection_id, tenant_id.as_deref(), batch)
                        .await
                }
            }
            .map_err(|e| TonicStatus::internal(format!("{} failed: {}", operation.as_str(), e)))?;

            total_records += result.metrics.successful_count.max(0) as u64;
            total_failed += result
                .metrics
                .failed_count
                .max(result.errors.len() as i64)
                .max(0) as u64;
            all_success &= result.success;

            // Send progress update as FlightData
            let progress_metadata = Self::batch_progress_app_metadata(
                operation,
                total_batches,
                batch_rows,
                total_records,
                &result,
            )
            .map_err(|e| TonicStatus::internal(format!("Failed to encode progress: {}", e)))?;

            let progress_data = FlightData {
                flight_descriptor: None,
                data_header: Default::default(),
                app_metadata: progress_metadata.into(),
                data_body: Default::default(),
            };

            results.push(Ok(progress_data));
        }

        let final_metadata = Self::bulk_insert_complete_app_metadata(
            operation,
            total_batches,
            total_records,
            total_failed,
            all_success,
        )
        .map_err(|e| TonicStatus::internal(format!("Failed to encode final status: {}", e)))?;

        let final_data = FlightData {
            flight_descriptor: None,
            data_header: Default::default(),
            app_metadata: final_metadata.into(),
            data_body: Default::default(),
        };

        results.push(Ok(final_data));

        info!(
            operation = %operation.as_str(),
            total_batches = total_batches,
            total_records = total_records,
            total_failed = total_failed,
            "Arrow Flight: bulk write exchange completed"
        );

        let stream = stream::iter(results);
        Ok(TonicResponse::new(Box::pin(stream)))
    }

    /// Handle the batched columnar graph ingest exchange (`graph_nodes` /
    /// `graph_edges`). Each streamed Arrow `RecordBatch` is decoded via
    /// [`super::graph_codec`] into neutral nodes/edges and upserted through the
    /// live `GraphOperationsService`, so the rows are immediately
    /// queryable/traversable (not parked in the generic record store).
    async fn handle_graph_write_exchange(
        &self,
        graph_id: String,
        is_nodes: bool,
        tenant_id: Option<String>,
        first_msg: FlightData,
        stream: TonicStreaming<FlightData>,
    ) -> TonicResult<<Self as FlightService>::DoExchangeStream> {
        let graph = self.graph_service.as_ref().ok_or_else(|| {
            TonicStatus::unimplemented("graph Flight path requires the graph backing service")
        })?;
        // Structural isolation: resolve the tenant and route through the scoped handle,
        // which composes `{tenant}/{graph_id}` once. The graph_id stays tenant-clean.
        let tenant =
            Self::resolve_tenant_for_mode(tenant_id.as_deref(), &self.tenant_deployment_mode)?;
        let ops = graph.for_tenant(&tenant);

        let mut total_rows = 0u64;
        let mut total_batches = 0u64;
        let mut all_success = true;
        let mut results = Vec::new();

        let mut batch_stream = Self::record_batch_stream(first_msg, stream);
        while let Some(batch) = batch_stream.next().await {
            let batch = batch.map_err(|e| TonicStatus::internal(format!("Stream error: {}", e)))?;
            total_batches += 1;
            let row_count = batch.num_rows() as u64;

            let written = if is_nodes {
                let nodes = super::graph_codec::batch_to_nodes(&batch)
                    .map_err(|e| TonicStatus::invalid_argument(format!("decode nodes: {}", e)))?;
                ops.batch_create_nodes_with_strategy(&graph_id, nodes, "update")
                    .await
                    .map(|created| created.len() as u64)
            } else {
                let edges = super::graph_codec::batch_to_edges(&batch)
                    .map_err(|e| TonicStatus::invalid_argument(format!("decode edges: {}", e)))?;
                ops.batch_create_edges(&graph_id, edges)
                    .await
                    .map(|outcome| outcome.created.len() as u64)
            };

            let (written, ok) = match written {
                Ok(n) => (n, true),
                Err(e) => {
                    all_success = false;
                    tracing::error!(graph_id = %graph_id, error = %e, "graph Flight ingest batch failed");
                    (0, false)
                }
            };
            total_rows += written;

            let progress = serde_json::json!({
                "kind": if is_nodes { "graph_nodes" } else { "graph_edges" },
                "batch": total_batches,
                "rows_in": row_count,
                "rows_written": written,
                "success": ok,
            });
            results.push(Ok(FlightData {
                flight_descriptor: None,
                data_header: Default::default(),
                app_metadata: progress.to_string().into_bytes().into(),
                data_body: Default::default(),
            }));
        }

        let final_meta = serde_json::json!({
            "kind": if is_nodes { "graph_nodes" } else { "graph_edges" },
            "complete": true,
            "total_batches": total_batches,
            "total_rows_written": total_rows,
            "success": all_success,
        });
        results.push(Ok(FlightData {
            flight_descriptor: None,
            data_header: Default::default(),
            app_metadata: final_meta.to_string().into_bytes().into(),
            data_body: Default::default(),
        }));

        info!(
            graph_id = %graph_id,
            kind = if is_nodes { "graph_nodes" } else { "graph_edges" },
            total_batches = total_batches,
            total_rows_written = total_rows,
            "Arrow Flight: graph columnar ingest completed"
        );

        Ok(TonicResponse::new(Box::pin(stream::iter(results))))
    }

    /// Parse a DoGet `Ticket` as a graph export request. Returns `None` for
    /// non-graph tickets so they fall through to the vector-search path.
    fn parse_graph_ticket(ticket: &Ticket) -> Option<GraphTicket> {
        let parsed: GraphTicket = serde_json::from_slice(&ticket.ticket).ok()?;
        matches!(parsed.model.as_str(), "graph_nodes" | "graph_edges").then_some(parsed)
    }

    /// Stream a graph export as a paginated columnar Flight response (the DoGet
    /// half of the batched columnar graph path).
    ///
    /// Nodes are paged through `query_nodes` (limit/offset), so the server never
    /// holds the whole node set; the schema dimension is fixed from the first
    /// page so every page shares one Arrow schema. Edges are endpoint-scoped (the
    /// engine has no full edge scan) and chunked. The `FlightDataEncoder` emits
    /// one schema frame then a data frame per page; `ticket.limit` sets the page
    /// size (the client streams pages and may cancel early). Egress bytes are
    /// metered per frame.
    async fn graph_export_flight_stream(
        &self,
        ticket: GraphTicket,
        tenant_id: Option<String>,
        edge: crate::metrics::consumption_metrics::EdgePolicyContext,
    ) -> std::result::Result<<Self as FlightService>::DoGetStream, TonicStatus> {
        const DEFAULT_PAGE: usize = 1024;
        let graph = self.graph_service.clone().ok_or_else(|| {
            TonicStatus::unimplemented("graph Flight path requires the graph backing service")
        })?;
        // Read the SAME structural key the write path composes (`{tenant}/{graph_id}`).
        let tenant =
            Self::resolve_tenant_for_mode(tenant_id.as_deref(), &self.tenant_deployment_mode)?;
        let gid = crate::graph::scoped_graph_id(&tenant, &ticket.graph_id)
            .map_err(|e| TonicStatus::invalid_argument(format!("invalid tenant: {e}")))?;
        let page = ticket
            .limit
            .map(|l| l as usize)
            .filter(|&l| l > 0)
            .unwrap_or(DEFAULT_PAGE);

        // Egress-aware shaping (co-design D2): compress the columnar body for
        // chargeable (far) clients; the Arrow reader decompresses transparently.
        let ipc_options = if edge.shape_policy().compress {
            FlightCompression::Zstd.to_ipc_write_options()
        } else {
            FlightCompression::None.to_ipc_write_options()
        };

        let (schema, rb_stream): (
            Arc<Schema>,
            Pin<Box<dyn Stream<Item = GraphBatchResult> + Send>>,
        ) = if ticket.model == "graph_nodes" {
            let labels: Vec<String> = ticket.label.clone().into_iter().collect();
            // Peek the first page to fix the schema dimension for the whole stream.
            let first = Self::query_node_page(&graph, &gid, &labels, page, 0)
                .await
                .map_err(|e| TonicStatus::internal(format!("query nodes: {e}")))?;
            let dim = super::graph_codec::embedding_dim_of(&first)
                .map_err(|e| TonicStatus::invalid_argument(format!("embedding dim: {e}")))?;
            let schema = super::graph_codec::graph_node_schema(dim);

            let stream = stream::unfold(NodePage::First(first), move |state| {
                let graph = graph.clone();
                let gid = gid.clone();
                let labels = labels.clone();
                async move {
                    match state {
                        NodePage::First(nodes) => {
                            let next = if nodes.len() < page {
                                NodePage::Done
                            } else {
                                NodePage::More(page)
                            };
                            Some((
                                super::graph_codec::nodes_to_batch_with_dim(&nodes, dim)
                                    .map_err(graph_flight_err),
                                next,
                            ))
                        }
                        NodePage::More(offset) => {
                            match Self::query_node_page(&graph, &gid, &labels, page, offset).await {
                                Ok(nodes) if !nodes.is_empty() => {
                                    let next = if nodes.len() < page {
                                        NodePage::Done
                                    } else {
                                        NodePage::More(offset + page)
                                    };
                                    Some((
                                        super::graph_codec::nodes_to_batch_with_dim(&nodes, dim)
                                            .map_err(graph_flight_err),
                                        next,
                                    ))
                                }
                                Ok(_) => None,
                                Err(e) => Some((Err(graph_flight_err(e)), NodePage::Done)),
                            }
                        }
                        NodePage::Done => None,
                    }
                }
            });
            (schema, Box::pin(stream))
        } else {
            // Edges live in per-source adjacency. With an endpoint, use the
            // adjacency-scoped query; without one, do a full graph edge scan
            // (export/ETL — dump every edge), filtering by type if requested.
            let edges = if ticket.from_node_id.is_none() && ticket.to_node_id.is_none() {
                let mut all = graph
                    .all_edges(&gid)
                    .await
                    .map_err(|e| TonicStatus::internal(format!("scan edges: {e}")))?;
                if let Some(et) = &ticket.edge_type {
                    all.retain(|e| e.edge_type == *et);
                }
                all
            } else {
                let query = crate::graph::EdgeQuery {
                    graph_id: gid.clone(),
                    from_node_id: ticket.from_node_id.clone(),
                    to_node_id: ticket.to_node_id.clone(),
                    edge_types: ticket.edge_type.clone().into_iter().collect(),
                    filters: Vec::new(),
                    limit: None,
                    offset: None,
                    continuation_token: None,
                };
                graph
                    .query_edges(&gid, query)
                    .await
                    .map_err(|e| TonicStatus::internal(format!("query edges: {e}")))?
            };
            let edges: Vec<crate::graph::Edge> = edges.iter().map(|e| (**e).clone()).collect();
            let batches: Vec<GraphBatchResult> = edges
                .chunks(page)
                .map(|chunk| super::graph_codec::edges_to_batch(chunk).map_err(graph_flight_err))
                .collect();
            (
                super::graph_codec::graph_edge_schema(),
                Box::pin(stream::iter(batches)),
            )
        };

        let encoder = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .with_options(ipc_options)
            .build(rb_stream);

        // Meter the actual encoded bytes per frame, then surface encode errors
        // as a stream-level status.
        let metered = encoder.map(move |res| match res {
            Ok(fd) => {
                edge.record_result_egress(
                    tenant_id.as_deref(),
                    flight_data_wire_bytes(std::slice::from_ref(&fd)),
                );
                Ok(fd)
            }
            Err(e) => Err(TonicStatus::internal(format!("graph export encode: {e}"))),
        });

        Ok(Box::pin(metered))
    }

    /// Fetch one page of nodes (label-scoped, or full scan when `labels` is
    /// empty) as neutral nodes.
    async fn query_node_page(
        graph: &crate::graph::GraphOperationsService,
        graph_id: &str,
        labels: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<Vec<crate::graph::Node>> {
        let query = crate::graph::NodeQuery {
            graph_id: graph_id.to_string(),
            labels: labels.to_vec(),
            filters: Vec::new(),
            limit: Some(limit as u32),
            offset: Some(offset as u32),
            continuation_token: None,
        };
        let nodes = graph.query_nodes(graph_id, query).await?;
        Ok(nodes.iter().map(|n| (**n).clone()).collect())
    }

    /// Handle bulk search exchange - stream query vectors and return results
    async fn handle_bulk_search_exchange(
        &self,
        collection_id: String,
        identity: proximadb_tenant::ResolvedRequestIdentity,
        first_msg: FlightData,
        mut stream: TonicStreaming<FlightData>,
    ) -> TonicResult<<Self as FlightService>::DoExchangeStream> {
        let mut results: Vec<std::result::Result<FlightData, TonicStatus>> = Vec::new();
        let mut query_count = 0u64;

        // The bulk_search exchange carries query vectors as Arrow batches of
        // ProximaRecord envelopes. Optional control frames (JSON in
        // `app_metadata`) set top_k / filters / include_vector for the queries
        // that follow; absent a control frame the defaults are top_k=10 and no
        // filters — but now on the canonical v2 search path (TD-FLIGHT-1).
        let mut top_k: u32 = 10;
        let mut include_vector = false;
        let mut filters: Vec<FlightFilter> = Vec::new();
        let mut flight_messages = vec![first_msg];

        while let Some(data) = stream
            .message()
            .await
            .map_err(|e| TonicStatus::internal(format!("Stream error: {}", e)))?
        {
            if !data.app_metadata.is_empty()
                && let Ok(control) = serde_json::from_slice::<BulkSearchControl>(&data.app_metadata)
            {
                debug!(?control, "Received bulk_search control frame");
                if let Some(k) = control.top_k {
                    top_k = k;
                }
                include_vector = control.include_vector;
                if !control.filters.is_empty() {
                    filters = control.filters;
                }
                continue;
            }
            flight_messages.push(data);
        }

        let query_batches = ArrowProtoCodec::flight_data_stream_to_batches(&flight_messages)
            .map_err(|e| TonicStatus::internal(format!("Failed to parse query batches: {}", e)))?;

        for batch in query_batches {
            // Extract query vectors from batch using canonical ProximaRecord envelopes.
            let query_records = match ArrowProtoCodec::batches_to_proxima_records(vec![batch]) {
                Ok(v) => v,
                Err(e) => {
                    // Surface the per-batch decode error as a frame instead of
                    // silently dropping the whole batch.
                    let meta = serde_json::to_vec(&serde_json::json!({
                        "type": "error", "stage": "decode", "message": e.to_string()
                    }))
                    .unwrap_or_default();
                    results.push(Ok(control_flight_data(meta)));
                    warn!("Failed to extract query vectors: {e}");
                    continue;
                }
            };

            // Execute one canonical v2 search per query vector, preserving input
            // order. A failed query surfaces an error frame at its position
            // rather than vanishing (acceptance #2).
            for query_record in query_records {
                let query_index = query_count;
                query_count += 1;

                let query_vector = query_record
                    .embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default();

                // Build a canonical v2 ticket (reuses the DoGet filter lowering)
                // and route through the same RecordSearchPort authority.
                let ticket = FlightSearchTicket {
                    ticket_type: "vector_search".to_string(),
                    collection_id: collection_id.clone(),
                    query_vector,
                    top_k,
                    filters: filters.clone(),
                    include_vector,
                };

                let search_batches = match self
                    .handle_v2_search(ticket, proximadb_runtime::PortIdentity::from(&identity))
                    .await
                {
                    Ok(batches) => batches,
                    Err(e) => {
                        let meta = serde_json::to_vec(&serde_json::json!({
                            "type": "error",
                            "query_index": query_index,
                            "query_id": query_record.oid.to_string(),
                            "message": e.to_string()
                        }))
                        .unwrap_or_default();
                        results.push(Ok(control_flight_data(meta)));
                        warn!(query_index, query_id = %query_record.oid, "bulk_search query failed: {e}");
                        continue;
                    }
                };

                for result_batch in search_batches {
                    let flight_data_vec =
                        ArrowProtoCodec::batch_to_flight_data(&result_batch, &Default::default())
                            .map_err(|e| {
                            TonicStatus::internal(format!("Failed to encode result: {}", e))
                        })?;
                    for fd in flight_data_vec {
                        results.push(Ok(fd));
                    }
                }
            }
        }

        // Completion frame.
        let complete_meta = serde_json::to_vec(&serde_json::json!({
            "type": "complete",
            "query_count": query_count,
            "collection_id": collection_id
        }))
        .unwrap_or_default();
        results.push(Ok(control_flight_data(complete_meta)));

        info!(
            collection_id = %collection_id,
            query_count,
            "Arrow Flight: bulk_search exchange completed (canonical v2 path)"
        );

        Ok(TonicResponse::new(Box::pin(stream::iter(results))))
    }

    /// Handle generic data transfer exchange with progress tracking
    async fn handle_data_transfer_exchange(
        &self,
        collection_id: String,
        first_msg: FlightData,
        mut stream: TonicStreaming<FlightData>,
    ) -> TonicResult<<Self as FlightService>::DoExchangeStream> {
        let mut results = Vec::new();
        let mut total_bytes = 0u64;
        let mut chunk_count = 0u64;

        if !first_msg.data_body.is_empty() {
            chunk_count += 1;
            total_bytes += first_msg.data_body.len() as u64;
        }

        // Process data chunks
        while let Some(data) = stream
            .message()
            .await
            .map_err(|e| TonicStatus::internal(format!("Stream error: {}", e)))?
        {
            chunk_count += 1;
            total_bytes += data.data_body.len() as u64;

            // Send acknowledgment for each chunk
            let ack = serde_json::json!({
                "type": "ack",
                "chunk": chunk_count,
                "bytes_received": data.data_body.len(),
                "total_bytes": total_bytes
            });

            let ack_data = FlightData {
                flight_descriptor: None,
                data_header: Default::default(),
                app_metadata: serde_json::to_vec(&ack).unwrap_or_default().into(),
                data_body: Default::default(),
            };

            results.push(Ok(ack_data));
        }

        // Send final completion
        let complete = serde_json::json!({
            "type": "complete",
            "success": true,
            "total_chunks": chunk_count,
            "total_bytes": total_bytes,
            "collection_id": collection_id
        });

        let complete_data = FlightData {
            flight_descriptor: None,
            data_header: Default::default(),
            app_metadata: serde_json::to_vec(&complete).unwrap_or_default().into(),
            data_body: Default::default(),
        };

        results.push(Ok(complete_data));

        info!(
            collection_id = %collection_id,
            chunk_count = chunk_count,
            total_bytes = total_bytes,
            "Arrow Flight: data_transfer exchange completed"
        );

        Ok(TonicResponse::new(Box::pin(stream::iter(results))))
    }
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
