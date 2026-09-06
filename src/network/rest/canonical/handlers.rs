//! Aligned REST API handlers using protobuf-first approach
//!
//! These handlers demonstrate the proper pattern for REST APIs that:
//! 1. Accept protobuf types directly as JSON
//! 2. Return protobuf responses as JSON
//! 3. Use unified ApiError for consistent error handling

use axum::{
    extract::{Extension, Json, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json as JsonResponse, Response},
};
use proximadb_api::rest::deprecation::add_rest_v1_deprecation_headers;
use proximadb_graph_query::service::GraphExecutionService;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::errors::{ApiError, ApiResult};
use crate::network::middleware::tenant::TenantContext;
use crate::network::rest::canonical::analytics::{self, AnalyticsApiState};
use crate::network::rest::canonical::aql::{self, AqlApiState};
use crate::network::rest::canonical::nl::{self, NlApiState};
use crate::network::rest::health;
use crate::proto::proximadb_v1;
use crate::query::QueryFacadeAdapter;
use crate::query::aql::executor::AqlExecutor;
use crate::query::aql::sources::document::DocumentAqlSource;
use crate::query::aql::sources::graph::GraphAqlSource;
use crate::query::aql::sources::memory::MemoryAqlSource;
use crate::query::aql::sources::observability::ObservabilityAqlSource;
use crate::query::aql::sources::vector::VectorAqlSource;
use serde::{Deserialize, Serialize};

/// The concrete core services the REST `AppState` needs, extracted once at boot
/// (TD-104 S3-e/S3-f — ROOT decoupling). Every field here is also a
/// `SharedServices` field, so the REST server no longer reaches into the
/// (deleted) root `UnifiedHandlers` boot-adapter. Built via
/// [`RestCoreServices::from_shared_services`] at the boot call sites.
pub struct RestCoreServices {
    /// Shared xCatalog manager used by catalog-backed control-plane routes.
    pub catalog_manager: Arc<crate::catalog::CatalogManager>,
    /// Same model-registry service instance exposed by every native transport.
    pub model_registry_service:
        Arc<proximadb_catalog::model_registry_service::CatalogModelRegistryService>,
    /// Vector CRUD/search — same `Arc` as `SharedServices.vector_operations_service`.
    pub vector_operations_service: Arc<crate::services::VectorOperationsService>,
    /// Document storage — same `Arc` as `SharedServices.document_service`.
    pub document_service: Arc<crate::storage::document::DocumentService>,
    /// Collection lifecycle — same `Arc` as `SharedServices.collection_service`.
    pub collection_service: Arc<crate::services::CollectionService>,
    /// Graph ops (`GraphService` is an alias for `GraphOperationsService`) —
    /// same `Arc` as `SharedServices.graph_service`.
    pub graph_operations_service: Arc<crate::graph::GraphOperationsService>,
    /// Graph collection CRUD (create/list/get/delete graph) — the same
    /// `Arc<GraphCollectionService>` the former root handler held (shared with
    /// `GraphOperationsService`'s internal collection service).
    pub graph_collection_service: Arc<crate::services::GraphCollectionService>,
    /// Record write-path orchestration — the same `Arc<RecordOpsService>` the
    /// former root handler's `record_ops()` returned (it already impls
    /// `RecordOpsPort` independently).
    pub record_ops: Arc<crate::api_handlers::record_ops_service::RecordOpsService>,
    /// Observability — same `Arc` as `SharedServices.observability_service`.
    pub observability_service: Arc<crate::observability::ObservabilityService>,
    /// Event-log engine — same `Option<Arc>` as `SharedServices.event_log`.
    pub event_log: Option<Arc<crate::storage::engines::eventlog::EventLogEngine>>,
    /// Port-backed API handler (the runtime `UnifiedHandlers`) — same `Arc` as
    /// `SharedServices.api_handlers`.
    pub api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,

    /// Durable ABAC attribute-authority handle (TD-ABAC control-plane /
    /// `abac-policy`). Same `Option<Arc>` as `SharedServices.abac_authority`; flows
    /// into `AppState.abac_authority` so the admin endpoints share the enforcer's
    /// instance (hot-reload). `None` when ABAC is off.
    #[cfg(feature = "abac-policy")]
    pub abac_authority: Option<Arc<proximadb_abac::FileSystemAttributeAuthority>>,
    /// Durable ABAC policy-binding store handle. Same as
    /// `SharedServices.abac_binding_store`.
    #[cfg(feature = "abac-policy")]
    pub abac_binding_store: Option<Arc<proximadb_abac::FileSystemPolicyBindingStore>>,
    /// Durable ABAC predicate-object store handle. Same as
    /// `SharedServices.abac_predicate_store`.
    #[cfg(feature = "abac-policy")]
    pub abac_predicate_store: Option<Arc<proximadb_abac::FileSystemPredicateObjectStore>>,

    /// `SharedServices.abac_grant_store` (ADR-090 grants admin).
    #[cfg(feature = "abac-policy")]
    pub abac_grant_store: Option<Arc<proximadb_catalog::grants::FileSystemGrantStore>>,

    /// `SharedServices.abac_posture_store` (TD-SEC-2 per-tenant posture).
    #[cfg(feature = "abac-policy")]
    pub abac_posture_store:
        Option<Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>>,
}

impl RestCoreServices {
    /// Build the REST core-service bundle from `SharedServices`. Single source
    /// of truth for the ROOT→port mapping; the boot call sites
    /// (`RestServer::*`, `build_router_for_unified`) pass the result to
    /// [`AppState::new`].
    pub fn from_shared_services(
        services: &crate::network::shared_services::SharedServices,
    ) -> Self {
        Self {
            catalog_manager: services.catalog_manager.clone(),
            model_registry_service: services.model_registry_service.clone(),
            vector_operations_service: services.vector_operations_service.clone(),
            document_service: services.document_service.clone(),
            collection_service: services.collection_service.clone(),
            graph_operations_service: services.graph_service.clone(),
            graph_collection_service: services.graph_collection_service.clone(),
            record_ops: services.record_ops.clone(),
            observability_service: services.observability_service.clone(),
            event_log: services.event_log.clone(),
            api_handlers: services.api_handlers.clone(),
            // TD-ABAC control-plane: carry the durable handles through so
            // `AppState::new` can install them — the admin endpoints then share
            // the SAME instances the live enforcer reads (hot-reload).
            #[cfg(feature = "abac-policy")]
            abac_authority: services.abac_authority.clone(),
            #[cfg(feature = "abac-policy")]
            abac_binding_store: services.abac_binding_store.clone(),
            #[cfg(feature = "abac-policy")]
            abac_predicate_store: services.abac_predicate_store.clone(),
            #[cfg(feature = "abac-policy")]
            abac_grant_store: services.abac_grant_store.clone(),
            #[cfg(feature = "abac-policy")]
            abac_posture_store: services.abac_posture_store.clone(),
        }
    }
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// Extracted graph execution capability for query planning/execution helpers
    pub graph_execution_service: Arc<dyn GraphExecutionService>,
    /// Vector operations service, extracted at boot so the REST router-build
    /// (AQL/analytics/hybrid/memory wiring) no longer reaches into
    /// `request_handlers`'s concrete fields (TD-104 S5 — ROOT decoupling).
    /// Same `Arc` the root handler holds; mirrors `graph_execution_service`.
    pub vector_operations_service: Arc<crate::services::VectorOperationsService>,
    /// Document service, extracted at boot (TD-104 S5). Feeds the document AQL
    /// source; same `Arc` as the root handler.
    pub document_service: Arc<crate::storage::document::DocumentService>,
    /// Collection service, extracted at boot (TD-104 S5). Feeds the health
    /// endpoint's collection-count probe; same `Arc` as the root handler.
    pub collection_service: Arc<crate::services::CollectionService>,
    /// Graph operations service, extracted at boot (TD-104 S5). Feeds the v2
    /// graph fusion-search + impact-analysis handlers and the entity
    /// orchestrator; same `Arc` as the root handler. Replaces per-request
    /// `state.request_handlers.graph_operations_service` reads.
    pub graph_operations_service: Arc<crate::graph::GraphOperationsService>,
    /// Graph collection CRUD service (create/list/get/delete graph), extracted
    /// at boot (TD-104 S3-e). Feeds the v1 REST graph-collection handlers;
    /// same `Arc` as the former root handler.
    pub graph_collection_service: Arc<crate::services::GraphCollectionService>,
    /// Record write-path service (TD-104 S5). Owns record-batch insert/delete
    /// orchestration; same `Arc` the root handler's `record_ops()` returns.
    pub record_ops: Arc<crate::api_handlers::record_ops_service::RecordOpsService>,
    /// Cross-modal fusion service — the single retrieval port every search
    /// surface delegates to (`SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`).
    /// Constructed once at boot from the vector + graph services (mirrors the
    /// TD-104 S5 extraction pattern) instead of per-request in each handler.
    pub fusion_service: Arc<crate::services::fusion_service::FusionService>,
    /// Observability service, extracted at boot (TD-104 S5). Feeds the
    /// observability AQL source; same `Arc` as the root handler.
    pub observability_service: Arc<crate::observability::ObservabilityService>,
    /// Event-log engine for persistent audit trails (TD-050), extracted at
    /// boot (TD-104 S5). Feeds AQL audit + agent-memory consolidation sink;
    /// same `Option<Arc>` as the root handler.
    pub event_log: Option<Arc<crate::storage::engines::eventlog::EventLogEngine>>,
    /// Optional security coordinator for authentication/authorization
    pub security_coordinator: Option<Arc<crate::security::SecurityCoordinator>>,
    /// Data directory from config (e.g., server.data_dir from TOML)
    pub data_dir: std::path::PathBuf,
    /// Query facade adapter for unified query execution
    /// Optional for backward compatibility during feature flag transition
    pub query_adapter: Option<Arc<QueryFacadeAdapter>>,
    /// Per-collection full-text indices for hybrid BM25+vector search
    pub fulltext_indexes: Option<FullTextIndexMap>,
    /// Catalog manager for external catalog integration
    pub catalog_manager: Arc<crate::catalog::CatalogManager>,
    /// Tenant-scoped model-registry lifecycle authority shared with gRPC.
    pub model_registry_service:
        Arc<proximadb_catalog::model_registry_service::CatalogModelRegistryService>,
    /// Phase 6: per-collection pinning registry. Set from
    /// `SharedServices.pin_registry` so the REST handlers and the
    /// AxisTieringManager consumer share the same Arc.
    pub pin_registry: Arc<crate::storage::collection_pinning::CollectionPinRegistry>,
    /// Phase 7.2.4: per-collection cache-affinity registry. Set from
    /// `SharedServices.affinity_registry` so the REST operator
    /// endpoints and the `VectorOperationsService` data-plane
    /// recorder share the same `Arc`.
    pub affinity_registry: Arc<crate::cluster::cache_affinity::CacheAffinityRegistry>,
    /// Slice 3 of tenant-pod-affinity: per-(tenant, collection)
    /// primary-pod registry. Set from `SharedServices.primary_pod_registry`
    /// so the REST operator endpoints and the future gateway write
    /// router share the same `Arc`. The REST handlers in
    /// `crate::network::rest::canonical::primary_pod` gate every read/write
    /// behind an operator-permission check — see
    /// [`crate::network::rest::canonical::primary_pod::authorize_operator`].
    pub primary_pod_registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,

    /// Slice 4 of tenant-pod-affinity: this pod's identity, used by
    /// the gateway write-router gate to compare against bindings in
    /// `primary_pod_registry`. Resolved via
    /// [`crate::cluster::primary_pod_registry::resolve_self_pod_id`]:
    /// explicit config override → `PROXIMADB_POD_ID` env var →
    /// `"self"` fallback. Stored as a plain `String` (not `Arc`)
    /// because it's cheap to clone and immutable for the process
    /// lifetime.
    pub self_pod_id: String,
    /// PAX segment registry shared with the write path (gRPC v2, Arrow Flight).
    /// Enables Iceberg REST snapshot summaries to reflect real PAX segment stats.
    pub segment_registry: Arc<crate::catalog::SegmentRegistry>,
    /// LLM engine for semantic operations
    pub llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
    /// Port-based document service (from proximadb-api migration)
    pub doc_port: Option<Arc<dyn proximadb_runtime::DocumentPort>>,
    /// Port-based graph service (from proximadb-api migration)
    pub graph_port: Option<Arc<dyn proximadb_runtime::GraphPort>>,
    /// Port-based observability service (from proximadb-api migration)
    pub obs_port: Option<Arc<dyn proximadb_runtime::ObservabilityPort>>,
    /// Port-backed unified query service (Phase 9.9)
    pub unified_query_port: Option<Arc<dyn proximadb_runtime::UnifiedQueryPort>>,
    /// Port-backed API handler for collection/vector routes (Phase 9.10).
    ///
    /// The canonical v2 router (`create_v2_router`) delegates collection and
    /// record operations through this `ApiHandlersPort`
    /// (`CollectionPort`/`VectorOpsPort` trait objects).
    ///
    /// TD-104 S3-e: non-optional. Set at construction from
    /// [`RestCoreServices::api_handlers`] (the runtime port-based handler);
    /// `with_api_handlers` may still override it in tests. The legacy
    /// root-`UnifiedHandlers`-as-port default and the `create_router`
    /// `ports=None` fallback are gone.
    pub api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
    /// Optional queue client for async ingest. When `Some`, the v3
    /// `/documents?mode=async` handler routes through `producer.send`
    /// on the `embed-ingest` topic and the embedding drainer consumes
    /// it asynchronously. When `None`, async mode degrades to inline
    /// embedding (still returns 202 but no real queue path). Production
    /// deployments wire this from `apps/proximadb-server` startup.
    pub queue_client: Option<Arc<proximadb_queue::QueueClient>>,
    /// Optional ranking framework services (R-7c). When `Some`, the
    /// `/api/v2/rank/search` route routes through the multi-phase
    /// pipeline; when `None`, the route returns 501 Not Implemented
    /// (the DEFAULT server always wires this — 501 is reachable only in
    /// non-default/embedded constructions). See
    /// `src/network/rest/canonical/rank.rs` and
    /// `roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md`.
    pub rank_services: Option<Arc<crate::network::rest::canonical::rank::RankServices>>,

    /// Optional durable rank-profile catalog (R-7c.3 production wiring).
    /// NOTE: the `/api/v2/rank/profiles` REST routes referenced below are
    /// BUILT (rank_profile.rs dispatchers) but NOT MOUNTED anywhere —
    /// profile management is TOML-file/pgwire-only until they are
    /// registered (round-1 review finding, TD-SPECRAT-1 wave 8).
    pub rank_profile_store: Option<Arc<dyn crate::services::RankProfileStore>>,

    /// Optional recall-probe gate (TD-064 / LLD §5). When `Some`, the v2
    /// route-health endpoint reports per-scope `gate_open` state and flips
    /// `recall_probe.live_state_in_app_state: true`. Production wires this
    /// from `SharedServices.recall_probe_gate` via `with_recall_probe_gate`
    /// in `src/network/multi_server.rs`. Search-path consultation
    /// (`wired_to_query_path: true`) is a separate follow-up; this slot
    /// only proves the gate is reachable from request handlers.
    pub recall_probe_gate: Option<Arc<crate::catalog::RecallProbeGate>>,

    /// Phase 8 (F1) Continuous Discovery service. Wired from
    /// `SharedServices.discovery_service` via `with_discovery_service` in
    /// `src/network/multi_server.rs` so the v2 `discovery-jobs` endpoints reach
    /// the same registry the background executor consumes.
    pub discovery_service: Option<Arc<crate::services::discovery::DiscoveryService>>,

    /// Phase 8 (F5) External Collection service. Wired from
    /// `SharedServices.external_collection_service` via
    /// `with_external_collection_service` in `src/network/multi_server.rs` so the
    /// v2 `external-collections` endpoints reach the same registry the service
    /// owns. `None` when the service is not enabled (handlers return 501).
    pub external_collection_service:
        Option<Arc<crate::services::external_collection::ExternalCollectionService>>,

    /// Durable ABAC attribute-authority handle (TD-ABAC control-plane /
    /// `abac-policy`). The SAME `Arc<FileSystemAttributeAuthority>` instance the
    /// live enforcer reads, so an admin provision written through this handle is
    /// visible without a restart (hot-reload). Wired from
    /// `SharedServices.abac_authority` via `with_abac_authority` in
    /// `src/network/multi_server.rs`. `None` when ABAC is off; the
    /// attribute-binding admin endpoints return 503 in that case.
    #[cfg(feature = "abac-policy")]
    pub abac_authority: Option<Arc<proximadb_abac::FileSystemAttributeAuthority>>,

    /// Durable ABAC policy-binding store handle (TD-ABAC control-plane /
    /// `abac-policy`). Shared with the live enforcer; the admin policy-binding
    /// endpoints write (upsert/remove) through this handle. Wired from
    /// `SharedServices.abac_binding_store` via `with_abac_binding_store`.
    /// `None` when ABAC is off.
    #[cfg(feature = "abac-policy")]
    pub abac_binding_store: Option<Arc<proximadb_abac::FileSystemPolicyBindingStore>>,

    /// Durable ABAC predicate-object store handle (TD-ABAC control-plane /
    /// `abac-policy`). Shared with the live enforcer; the admin predicate-object
    /// endpoints register/revoke through this handle. Wired from
    /// `SharedServices.abac_predicate_store` via `with_abac_predicate_store`.
    /// `None` when ABAC is off.
    #[cfg(feature = "abac-policy")]
    pub abac_predicate_store: Option<Arc<proximadb_abac::FileSystemPredicateObjectStore>>,

    /// `SharedServices.abac_grant_store` (ADR-090 grants admin), via
    /// `with_abac_grant_store`.
    #[cfg(feature = "abac-policy")]
    pub abac_grant_store: Option<Arc<proximadb_catalog::grants::FileSystemGrantStore>>,

    /// `SharedServices.abac_posture_store` (TD-SEC-2 per-tenant posture).
    #[cfg(feature = "abac-policy")]
    pub abac_posture_store:
        Option<Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>>,
}

impl AppState {
    /// Create REST app state with the standard shared runtime components.
    pub fn new(
        core: RestCoreServices,
        graph_execution_service: Arc<dyn GraphExecutionService>,
        security_coordinator: Option<Arc<crate::security::SecurityCoordinator>>,
        data_dir: std::path::PathBuf,
        query_adapter: Option<Arc<QueryFacadeAdapter>>,
        llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
    ) -> Self {
        // TD-104 S3-e: the concrete services arrive pre-extracted in `core`
        // (built once from `SharedServices` at the boot call site) instead of
        // being pulled out of the deleted root `UnifiedHandlers` here.
        let RestCoreServices {
            catalog_manager,
            model_registry_service,
            vector_operations_service,
            document_service,
            collection_service,
            graph_operations_service,
            graph_collection_service,
            record_ops,
            observability_service,
            event_log,
            api_handlers,
            #[cfg(feature = "abac-policy")]
            abac_authority,
            #[cfg(feature = "abac-policy")]
            abac_binding_store,
            #[cfg(feature = "abac-policy")]
            abac_predicate_store,
            #[cfg(feature = "abac-policy")]
            abac_grant_store,
            #[cfg(feature = "abac-policy")]
            abac_posture_store,
        } = core;
        // Cross-modal fusion port — built once at boot from the vector + graph services and the
        // shared full-text index map (search-surface contract: one retrieval engine, shared port).
        // The empty default map here powers the document expander fail-closed; production replaces
        // it via `with_fulltext_indexes`, which rebuilds this service with the LIVE index map so the
        // document modality (TD-138) is powered for REST fusion.
        let fulltext_indexes: FullTextIndexMap =
            std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));
        let fusion_service =
            std::sync::Arc::new(crate::services::fusion_service::FusionService::new(
                vector_operations_service.clone(),
                graph_operations_service.clone(),
                fulltext_indexes.clone(),
            ));
        Self {
            api_handlers,
            graph_execution_service,
            vector_operations_service,
            document_service,
            collection_service,
            graph_operations_service,
            graph_collection_service,
            record_ops,
            fusion_service,
            observability_service,
            event_log,
            security_coordinator,
            data_dir,
            query_adapter,
            fulltext_indexes: Some(fulltext_indexes),
            catalog_manager,
            model_registry_service,
            // Default: standalone pin registry. Production wires
            // SharedServices.pin_registry via `with_pin_registry` so
            // REST handlers and the eventual AxisTieringManager
            // consumer share the same Arc.
            pin_registry: crate::storage::collection_pinning::new_shared(),
            // Default: standalone affinity registry. Production wires
            // SharedServices.affinity_registry via `with_affinity_registry`
            // so REST handlers and the search-path recorder share the
            // same Arc.
            affinity_registry: crate::cluster::cache_affinity::new_shared(),
            // Default: standalone primary-pod registry. Production
            // wires `SharedServices.primary_pod_registry` so REST
            // handlers and the future gateway write router share the
            // same `Arc`.
            primary_pod_registry: crate::cluster::primary_pod_registry::new_shared(),
            // Default pod identity: resolve from env var or fall
            // back to `"self"`. Production may override via
            // `with_self_pod_id` once the config field lands.
            self_pod_id: crate::cluster::primary_pod_registry::resolve_self_pod_id(None),
            segment_registry: Arc::new(crate::catalog::SegmentRegistry::new()),
            llm_engine,
            doc_port: None,
            graph_port: None,
            obs_port: None,
            unified_query_port: None,
            queue_client: None,
            rank_services: None,
            rank_profile_store: None,
            recall_probe_gate: None,
            discovery_service: None,
            external_collection_service: None,
            // TD-ABAC control-plane: carried through `RestCoreServices` from
            // `SharedServices.abac_*` (built once, shared with the live enforcer
            // so an admin provision is hot-visible). `None` when ABAC is off; the
            // `with_abac_*` builders remain for test overrides.
            #[cfg(feature = "abac-policy")]
            abac_authority,
            #[cfg(feature = "abac-policy")]
            abac_binding_store,
            #[cfg(feature = "abac-policy")]
            abac_predicate_store,
            #[cfg(feature = "abac-policy")]
            abac_grant_store,
            #[cfg(feature = "abac-policy")]
            abac_posture_store,
        }
    }

    /// Inject the process-wide pin registry (Phase 6 control surface).
    /// Wired from `SharedServices.pin_registry` in production so REST
    /// handlers and the eventual `AxisTieringManager` consumer share
    /// the same Arc.
    pub fn with_pin_registry(
        mut self,
        registry: Arc<crate::storage::collection_pinning::CollectionPinRegistry>,
    ) -> Self {
        self.pin_registry = registry;
        self
    }

    /// Inject the process-wide cache-affinity registry (Phase 7.2.4).
    /// Wired from `SharedServices.affinity_registry` so REST operator
    /// endpoints and the `VectorOperationsService` recorder share
    /// the same Arc.
    pub fn with_affinity_registry(
        mut self,
        registry: Arc<crate::cluster::cache_affinity::CacheAffinityRegistry>,
    ) -> Self {
        self.affinity_registry = registry;
        self
    }

    /// Inject the process-wide primary-pod registry (Slice 3 of
    /// tenant-pod-affinity). Wired from
    /// `SharedServices.primary_pod_registry` so REST operator
    /// endpoints and the future gateway write router share the same
    /// `Arc`.
    pub fn with_primary_pod_registry(
        mut self,
        registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
    ) -> Self {
        self.primary_pod_registry = registry;
        self
    }

    /// Inject the shared durable ABAC attribute-authority handle (TD-ABAC
    /// control-plane / `abac-policy`). Wired from `SharedServices.abac_authority`
    /// in `src/network/multi_server.rs` so the admin attribute-binding endpoints
    /// and the live enforcer share the SAME instance (hot-reload).
    #[cfg(feature = "abac-policy")]
    pub fn with_abac_authority(
        mut self,
        authority: Arc<proximadb_abac::FileSystemAttributeAuthority>,
    ) -> Self {
        self.abac_authority = Some(authority);
        self
    }

    /// Inject the shared durable ABAC policy-binding store handle (TD-ABAC
    /// control-plane / `abac-policy`). Wired from `SharedServices.abac_binding_store`.
    #[cfg(feature = "abac-policy")]
    pub fn with_abac_binding_store(
        mut self,
        store: Arc<proximadb_abac::FileSystemPolicyBindingStore>,
    ) -> Self {
        self.abac_binding_store = Some(store);
        self
    }

    /// Inject the shared durable ABAC predicate-object store handle (TD-ABAC
    /// control-plane / `abac-policy`). Wired from `SharedServices.abac_predicate_store`.
    #[cfg(feature = "abac-policy")]
    pub fn with_abac_predicate_store(
        mut self,
        store: Arc<proximadb_abac::FileSystemPredicateObjectStore>,
    ) -> Self {
        self.abac_predicate_store = Some(store);
        self
    }

    /// Durable ADR-090 grant store (grants admin). Wired from
    /// `SharedServices.abac_grant_store`.
    #[cfg(feature = "abac-policy")]
    pub fn with_abac_grant_store(
        mut self,
        store: Arc<proximadb_catalog::grants::FileSystemGrantStore>,
    ) -> Self {
        self.abac_grant_store = Some(store);
        self
    }

    /// Override the resolved self-pod identity (Slice 4 of
    /// tenant-pod-affinity). Production wires this when the config
    /// field or env var sets a known pod_id; tests inject deterministic
    /// values so the write-routing gate behaves predictably.
    pub fn with_self_pod_id(mut self, pod_id: impl Into<String>) -> Self {
        self.self_pod_id = pod_id.into();
        self
    }

    /// Inject the shared full-text index map (T3.2 Slice 1b).
    /// Production wires this from `SharedServices.fulltext_indexes` so
    /// REST `/api/v1/hybrid/search` and gRPC `hybrid_search` share the
    /// same in-process map — an indexed document is searchable on
    /// both protocols.
    pub fn with_fulltext_indexes(mut self, indexes: FullTextIndexMap) -> Self {
        // Rebuild the shared fusion service so its DocumentExpander (TD-138) shares this live
        // index map. This builder runs during boot, before any request is served, so the rebuild
        // is safe; the fusion service remains a single boot-constructed instance from the
        // request-serving perspective. (The map type is structurally `HybridFullTextIndexMap`.)
        self.fusion_service =
            std::sync::Arc::new(crate::services::fusion_service::FusionService::new(
                self.vector_operations_service.clone(),
                self.graph_operations_service.clone(),
                indexes.clone(),
            ));
        self.fulltext_indexes = Some(indexes);
        self
    }

    /// Inject the ranking framework services bundle. Production wires
    /// this from `apps/proximadb-server` startup after constructing the
    /// `ProfileRegistry`, registering built-in features into the
    /// `BlueprintFactory`, and selecting a `CandidateProvider` impl
    /// (production = adapter over the hybrid coordinator; tests +
    /// pre-R-7c.1 deployments = `MockRangeCandidateProvider`).
    pub fn with_rank_services(
        mut self,
        services: Arc<crate::network::rest::canonical::rank::RankServices>,
    ) -> Self {
        self.rank_services = Some(services);
        self
    }

    /// Inject the durable rank-profile catalog so the REST install / fetch /
    /// remove endpoints can persist DDL-style operations through the canonical
    /// WAL. Production wires this from `SharedServices.rank_profile_store`.
    pub fn with_rank_profile_store(
        mut self,
        store: Arc<dyn crate::services::RankProfileStore>,
    ) -> Self {
        self.rank_profile_store = Some(store);
        self
    }

    /// Inject a queue client for async ingest. Production wires this
    /// from `apps/proximadb-server` startup after opening the queue
    /// subsystem at the configured root path.
    pub fn with_queue_client(mut self, client: Arc<proximadb_queue::QueueClient>) -> Self {
        self.queue_client = Some(client);
        self
    }

    /// Inject a shared segment registry (same `Arc` as in `SharedServices`).
    pub fn with_segment_registry(mut self, registry: Arc<crate::catalog::SegmentRegistry>) -> Self {
        self.segment_registry = registry;
        self
    }

    /// Inject the shared xCatalog manager from the server composition root.
    pub fn with_catalog_manager(mut self, manager: Arc<crate::catalog::CatalogManager>) -> Self {
        if !Arc::ptr_eq(&self.catalog_manager, &manager) {
            self.model_registry_service = Arc::new(
                proximadb_catalog::model_registry_service::CatalogModelRegistryService::new(
                    manager.clone(),
                ),
            );
            self.catalog_manager = manager;
        }
        self
    }

    /// Inject the shared recall-probe gate from `SharedServices`. When set,
    /// `/api/v2/_diagnostics/collections/:id/route-health` resolves
    /// per-scope gate state and reports `recall_probe.gate_open` +
    /// `recall_probe.live_state_in_app_state: true`. Search-path
    /// consultation (`wired_to_query_path: true`) is separate.
    pub fn with_recall_probe_gate(mut self, gate: Arc<crate::catalog::RecallProbeGate>) -> Self {
        self.recall_probe_gate = Some(gate);
        self
    }

    /// Inject the Phase 8 Continuous Discovery service (F1) so the v2
    /// `discovery-jobs` endpoints can create/inspect jobs.
    pub fn with_discovery_service(
        mut self,
        discovery_service: Arc<crate::services::discovery::DiscoveryService>,
    ) -> Self {
        self.discovery_service = Some(discovery_service);
        self
    }

    /// Inject the Phase 8 (F5) External Collection service so the v2
    /// `external-collections` endpoints can register/build/search.
    pub fn with_external_collection_service(
        mut self,
        external_collection_service: Arc<
            crate::services::external_collection::ExternalCollectionService,
        >,
    ) -> Self {
        self.external_collection_service = Some(external_collection_service);
        self
    }

    /// Inject port-based service objects for API-crate-backed routes.
    pub fn with_ports(
        mut self,
        doc_port: Arc<dyn proximadb_runtime::DocumentPort>,
        graph_port: Arc<dyn proximadb_runtime::GraphPort>,
        obs_port: Arc<dyn proximadb_runtime::ObservabilityPort>,
    ) -> Self {
        self.doc_port = Some(doc_port);
        self.graph_port = Some(graph_port);
        self.obs_port = Some(obs_port);
        self
    }

    /// Inject unified query port (Phase 9.9).
    pub fn with_unified_query_port(
        mut self,
        port: Arc<dyn proximadb_runtime::UnifiedQueryPort>,
    ) -> Self {
        self.unified_query_port = Some(port);
        self
    }

    /// Override the port-backed API handler for collection/vector routes
    /// (Phase 9.10). Production wires the runtime port-based handler here,
    /// replacing the root-as-port default set in `AppState::new` (TD-104 2(b)).
    pub fn with_api_handlers(
        mut self,
        handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
    ) -> Self {
        self.api_handlers = handlers;
        self
    }

    /// Create health-check state from the same explicit REST capability view.
    pub fn health_state(&self) -> health::HealthState {
        health::HealthState::new(
            self.vector_operations_service.clone(),
            self.collection_service.clone(),
            self.graph_execution_service.clone(),
        )
    }
}

/// SQL query request structure
#[derive(Debug, Serialize, Deserialize)]
pub struct SqlQueryRequest {
    /// SQL query string
    pub query: String,
    /// Optional parameters for parameterized queries (proto-aligned)
    pub parameters: Option<Vec<proximadb_v1::SqlValue>>,
    /// Optional collection to use as default context
    pub collection: Option<String>,
    /// Optional timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Optional seeding strategy for hybrid (average | per_seed | none)
    pub seeding: Option<String>,
}

fn sql_params_to_proxima_values(
    parameters: Option<Vec<proximadb_v1::SqlValue>>,
) -> Option<Vec<proximadb_data_model::ProximaValue>> {
    parameters.map(|values| {
        values
            .iter()
            .map(proximadb_records::conversions::sql_value_to_proxima)
            .collect()
    })
}

/// ADR-012 graph branch merge endpoint — extractor shim.
///
/// All actual logic lives in [`merge_graph_branch_inner`] so integration
/// tests can drive it through their own minimal axum Router without
/// constructing a full `AppState`. See
/// `tests/graph_branch_merge_rest_integration_test.rs`.
pub async fn merge_graph_branch(
    State(state): State<AppState>,
    Path((collection, branch)): Path<(String, String)>,
    Json(request): Json<GraphBranchMergeRequest>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    merge_graph_branch_inner(&state.data_dir, &collection, &branch, request).await
}

/// Pure-logic core of the branch-merge handler. Decoupled from `AppState`
/// so it can be driven from integration tests (and any future REST
/// endpoint that needs the same logic) without service-stub scaffolding.
///
/// Reads the canonical WAL under `data_dir/pgwire/canonical-records.wal`,
/// filters by `collection`, runs `merge_branches`, and (if `!dry_run`)
/// writes the resolutions back through `write_back_merge` with
/// `origin = "branch_merge:<branch>:<request.target_branch>"`.
pub async fn merge_graph_branch_inner(
    data_dir: &std::path::Path,
    collection: &str,
    branch: &str,
    request: GraphBranchMergeRequest,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    if collection.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "collection path parameter must not be empty".to_string(),
        ));
    }
    if branch.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "branch path parameter must not be empty".to_string(),
        ));
    }
    if request.target_branch.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "target_branch must not be empty".to_string(),
        ));
    }

    let wal_path = graph_branch_merge_wal_path(data_dir);
    if !tokio::fs::try_exists(&wal_path).await.map_err(|err| {
        ApiError::Internal(format!(
            "checking canonical WAL {} failed: {}",
            wal_path.display(),
            err
        ))
    })? {
        return Err(ApiError::NotFound(format!(
            "canonical WAL not found at {}",
            wal_path.display()
        )));
    }

    let entries = crate::services::FramedTableWalAppender::read_entries_from_path(&wal_path)
        .await
        .map_err(|err| {
            ApiError::Internal(format!(
                "reading canonical WAL {} failed: {}",
                wal_path.display(),
                err
            ))
        })?;
    let collection_entries = filter_canonical_wal_for_collection(entries, collection);
    let report =
        crate::graph::merge::merge_branches(&collection_entries, branch, &request.target_branch)
            .ok_or_else(|| {
                ApiError::NotFound(format!(
                    "no mergeable branch entries found for collection '{}' source '{}' target '{}'",
                    collection, branch, request.target_branch
                ))
            })?;

    // If not dry-run, write the merged records to WAL
    let write_back_result = if !request.dry_run {
        match crate::graph::merge::write_back_merge(
            &collection_entries,
            &report,
            &wal_path,
            collection,
            branch,
            &request.target_branch,
            request.tenant_id.clone(),
        )
        .await
        {
            Ok(Some(result)) => Some(result),
            Ok(None) => None, // Nothing to write
            Err(err) => {
                return Err(ApiError::Internal(format!(
                    "merge write-back failed: {}",
                    err
                )));
            }
        }
    } else {
        None
    };

    Ok(JsonResponse(serde_json::json!({
        "collection": collection,
        "source_branch": branch,
        "target_branch": request.target_branch,
        "dry_run": request.dry_run,
        "merge_base_lsn": report.merge_base_lsn,
        "left_events": report.left_events,
        "right_events": report.right_events,
        "conflicts": report.conflicts,
        "resolutions": report.resolutions,
        "summary": {
            "left_event_count": report.left_events.len(),
            "right_event_count": report.right_events.len(),
            "conflict_count": report.conflicts.len(),
            "resolution_count": report.resolutions.len(),
            "wal_path": wal_path.display().to_string()
        },
        "write_back": match write_back_result {
            Some(result) => serde_json::json!({
                "first_lsn": result.first_lsn,
                "last_lsn": result.last_lsn,
                "entries_written": result.written_entries.len()
            }),
            None => serde_json::json!(null)
        }
    })))
}

fn graph_branch_merge_wal_path(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join("pgwire").join("canonical-records.wal")
}

fn filter_canonical_wal_for_collection(
    entries: Vec<proximadb_storage_common::CanonicalWalEntry>,
    collection: &str,
) -> Vec<proximadb_storage_common::CanonicalWalEntry> {
    entries
        .into_iter()
        .filter(|entry| match &entry.operation {
            proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id, ..
            } => collection_id == collection,
            proximadb_storage_common::CanonicalOperation::RecordDelete {
                collection_id, ..
            } => collection_id == collection,
            proximadb_storage_common::CanonicalOperation::Checkpoint(_)
            | proximadb_storage_common::CanonicalOperation::CdcBarrier { .. }
            | proximadb_storage_common::CanonicalOperation::CatalogMutation { .. } => false,
        })
        .collect()
}

// SQL query response structure
// For REST, we now return proximadb.v1 ExecuteQueryResponse directly, wrapped by ProtoApiResponse
/// Column information in SQL results
#[derive(Debug, Serialize, Deserialize)]
pub struct SqlColumnInfo {
    /// Column name
    pub name: String,
    /// Column data type
    pub data_type: String,
}

#[derive(Debug, Deserialize)]
pub struct CatalogRoutingQuery {
    pub table_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TableWriteExplainRequest {
    pub target_table: String,
    pub source_table: Option<String>,
    pub source_sql: Option<String>,
    pub write_mode: Option<String>,
    pub distribution: Option<String>,
    pub target_columns: Option<Vec<String>>,
    pub tenant_id: Option<String>,
    pub actor: Option<String>,
    pub idempotency_key: Option<String>,
    pub row_count_hint: Option<u64>,
    pub estimated_bytes: Option<u64>,
    pub requires_row_level_semantics: Option<bool>,
    pub batch_local_constraints_sufficient: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct GraphBranchMergeRequest {
    /// Target branch to merge into. Defaults to `main`.
    #[serde(default = "default_graph_branch_merge_target")]
    pub target_branch: String,
    /// Dry-run returns the ADR-012 merge report without writing a merge commit.
    #[serde(default = "default_graph_branch_merge_dry_run")]
    pub dry_run: bool,
    /// Optional tenant ID for multi-tenant deployments.
    pub tenant_id: Option<String>,
}

fn default_graph_branch_merge_target() -> String {
    "main".to_string()
}

fn default_graph_branch_merge_dry_run() -> bool {
    true
}

/// Execute SQL query handler
///
/// Supports vector similarity queries like:
/// ```sql
/// SELECT id, metadata, COSINE_DISTANCE(embedding, [0.1, 0.2, 0.3]) as score
/// FROM my_collection
/// WHERE metadata.category = 'electronics'
/// ORDER BY score ASC
/// LIMIT 10
/// ```
pub async fn execute_sql(
    State(state): State<AppState>,
    Json(request): Json<SqlQueryRequest>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    let start_time = std::time::Instant::now();
    let request_id = Uuid::new_v4().to_string();

    info!(
        "SQL query request {} with query: {}",
        request_id,
        request.query.chars().take(100).collect::<String>()
    );

    // Validate request
    if request.query.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "SQL query cannot be empty".to_string(),
        ));
    }

    // Prefer unified query port (decoupled from concrete QueryFacadeAdapter type)
    if let Some(ref qp) = state.unified_query_port {
        debug!("Using unified query port for SQL query");
        let query_with_hint = if let Some(ref seeding) = request.seeding {
            format!(
                "-- SEEDING: {}\n{}",
                seeding.to_ascii_uppercase(),
                request.query
            )
        } else {
            request.query.clone()
        };
        return match qp
            .execute_unified_query(
                query_with_hint,
                sql_params_to_proxima_values(request.parameters.clone()),
                request.collection.clone(),
                None,
            )
            .await
        {
            Ok(port_result) => {
                let execution_time_ms = start_time.elapsed().as_millis() as u64;
                let records = port_result
                    .get("records")
                    .cloned()
                    .unwrap_or(serde_json::Value::Array(vec![]));
                let total = port_result
                    .get("total_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let json_data = serde_json::json!({
                    "rows": records,
                    "execution_time_ms": execution_time_ms,
                    "rows_returned": total,
                    "row_count": total,
                    "request_id": request_id,
                });
                info!(
                    "SQL query {} (port) completed in {}ms",
                    request_id, execution_time_ms
                );
                Ok(JsonResponse(json_data))
            }
            Err(e) => {
                error!("SQL query {} (port) failed: {}", request_id, e);
                Err(ApiError::Internal(e.to_string()))
            }
        };
    }

    // Fallback: route through concrete query_adapter facade
    if let Some(ref adapter) = state.query_adapter {
        debug!("Using query_adapter fallback for SQL query");
        return match adapter.sql_query(&request.query).await {
            Ok(result) => {
                let execution_time_ms = start_time.elapsed().as_millis() as u64;
                let rows = match result.data {
                    crate::query::QueryResultData::Rows(rows) => rows,
                    crate::query::QueryResultData::Empty => vec![],
                    _ => vec![],
                };
                let json_data = serde_json::json!({
                    "rows": rows,
                    "execution_time_ms": execution_time_ms,
                    "rows_returned": rows.len(),
                    "row_count": rows.len(),
                    "request_id": request_id
                });
                info!(
                    "SQL query {} (adapter) completed in {}ms",
                    request_id, execution_time_ms
                );
                Ok(JsonResponse(json_data))
            }
            Err(e) => {
                error!("SQL query {} (adapter) failed: {}", request_id, e);
                Err(ApiError::Internal(e.to_string()))
            }
        };
    }

    // Legacy path: Execute through v1 path (typed params and rows)
    // Optional: read seeding strategy from HTTP header (X-Seeding-Strategy) or from request.parameters via a special key
    let _seeding_strategy = crate::query::execution::SeedingStrategy::Average; // default

    let query_with_hint = if let Some(seeding) = &request.seeding {
        let seed_upper = seeding.to_ascii_uppercase();
        format!("-- SEEDING: {}\n{}", seed_upper, request.query)
    } else {
        request.query.clone()
    };

    // Route through the ApiHandlersPort (TD-104 / seam S1) rather than the
    // concrete ROOT handler. In the default config `state.api_handlers` is the
    // ROOT handler exposed as a port (round-trips params back to the inherent
    // path, so behavior is unchanged); under the runtime override it is the
    // port-based handler over the reconciled QueryFacadeAdapter. Wire SqlValue
    // params are lowered to canonical ProximaValue at this protocol boundary.
    match state
        .api_handlers
        .execute_sql_v1(
            query_with_hint,
            sql_params_to_proxima_values(request.parameters.clone()),
            request.collection,
            proximadb_runtime::PortIdentity::anonymous(),
        )
        .await
    {
        Ok(v1_resp) => {
            let execution_time_ms = start_time.elapsed().as_millis() as u64;

            // Convert SQL response to JSON value for now
            // Deferred: Create proper JsonExecuteQueryResponse wrapper if needed
            let json_data = serde_json::json!({
                "rows": v1_resp.rows.iter().map(|row| {
                    // Convert fields to a JSON object instead of list of key/value pairs
                    let mut obj = serde_json::Map::new();
                    for field in &row.fields {
                        let value = field.value.as_ref().map_or(serde_json::Value::Null, sql_value_to_json);
                        obj.insert(field.key.clone(), value);
                    }
                    serde_json::Value::Object(obj)
                }).collect::<Vec<_>>(),
                "columns": v1_resp.columns,
                "column_types": v1_resp.column_types,
                "execution_time_ms": execution_time_ms,
                "rows_returned": v1_resp.rows_returned,
                "row_count": v1_resp.rows_returned,  // Add row_count alias for compatibility
                "rows_scanned": v1_resp.rows_scanned,
                "request_id": request_id
            });

            info!(
                "SQL query {} completed in {}ms",
                request_id, execution_time_ms
            );

            Ok(JsonResponse(json_data))
        }
        Err(e) => {
            error!("SQL query {} failed: {}", request_id, e);
            // A DML lock/fence conflict must surface as a retryable 409
            // lock_conflict, not a generic 500. The typed `DmlLockConflict` can
            // sit several `.context(...)` layers deep, so walk the chain — same
            // as the pgwire 55P03 and gRPC ABORTED mappings. (421 Misdirected =
            // wrong pod stays distinct from 409 = conflict on the right pod.)
            if let Some((resource, holder)) = crate::errors::extract_dml_lock_conflict(&e) {
                return Err(ApiError::LockConflict(match holder {
                    Some(h) => format!("{resource} held by {h}"),
                    None => resource,
                }));
            }
            Err(ApiError::Internal(e.to_string()))
        }
    }
}

/// Return table-level xCatalog routing metadata for REST clients.
pub async fn get_catalog_table_routing(
    State(state): State<AppState>,
    Query(query): Query<CatalogRoutingQuery>,
) -> ApiResult<Json<crate::services::CatalogIntrospectionResult>> {
    let sql = match query.table_name.as_deref().map(str::trim) {
        Some(table_name) if !table_name.is_empty() => format!(
            "SELECT * FROM information_schema.table_routing WHERE table_name = '{}'",
            table_name.replace('\'', "''")
        ),
        _ => "SELECT * FROM information_schema.table_routing".to_string(),
    };

    let result = crate::services::CatalogIntrospectionService::new(state.catalog_manager.clone())
        .execute_select(&sql)
        .await
        .map_err(|error| ApiError::Internal(error.to_string()))?
        .unwrap_or_else(crate::services::CatalogIntrospectionResult::empty);

    Ok(Json(result))
}

/// Explain table-write route selection without executing the write.
pub async fn explain_table_write_route(
    State(state): State<AppState>,
    Json(request): Json<TableWriteExplainRequest>,
) -> ApiResult<Json<crate::query::table_write_plan::TableWriteRouteExplanation>> {
    use crate::query::table_write_plan::{
        ConflictPolicy, CopyIntoPlan, DmlWritePlanRequest, DmlWritePlanner, WriteIntentOverrides,
        WriteMode,
    };

    let target = logical_table_ref_from_name(&request.target_table)?;
    let target_table_name = target.qualified_name();
    let (catalog, table_id) = state
        .catalog_manager
        .resolve_table(&target_table_name)
        .await
        .map_err(|error| ApiError::NotFound(error.to_string()))?;
    if !catalog
        .table_exists(&table_id)
        .await
        .map_err(|error| ApiError::Internal(error.to_string()))?
    {
        return Err(ApiError::NotFound(format!(
            "Table '{}' does not exist",
            target_table_name
        )));
    }
    let target_schema = catalog
        .get_table(&table_id)
        .await
        .map_err(|error| ApiError::Internal(error.to_string()))?;
    let target_stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
    let (source, source_schema, source_stats) =
        resolve_table_write_explain_source(&state, &request).await?;
    let write_mode = parse_table_write_mode(request.write_mode.as_deref())?;
    let distribution = parse_distribution_mode(request.distribution.as_deref())?;
    let conflict_policy = if matches!(write_mode, WriteMode::Upsert | WriteMode::Merge) {
        ConflictPolicy::Upsert
    } else {
        ConflictPolicy::Error
    };
    let plan = CopyIntoPlan {
        source,
        target,
        write_mode,
        conflict_policy,
        distribution,
    };
    let write_intent_overrides = WriteIntentOverrides {
        tenant_id: request.tenant_id.clone(),
        actor: request.actor.clone(),
        idempotency_key: request.idempotency_key.clone(),
        row_count_hint: request.row_count_hint,
        estimated_bytes: request.estimated_bytes,
        requires_row_level_semantics: request.requires_row_level_semantics,
        batch_local_constraints_sufficient: request.batch_local_constraints_sufficient,
    };
    let target_columns = request.target_columns.unwrap_or_default();
    let routed = DmlWritePlanner::default()
        .plan(DmlWritePlanRequest {
            target_schema: &target_schema,
            target_stats: Some(&target_stats),
            source_schema: source_schema.as_ref(),
            source_stats: source_stats.as_ref(),
            write_intent_overrides: Some(&write_intent_overrides),
            plan: &plan,
            target_columns: &target_columns,
        })
        .map_err(|error| ApiError::InvalidArgument(error.to_string()))?;

    Ok(Json(routed.route_explanation()))
}

async fn resolve_table_write_explain_source(
    state: &AppState,
    request: &TableWriteExplainRequest,
) -> ApiResult<(
    crate::query::table_write_plan::ReadSource,
    Option<proximadb_catalog::CatalogTableSchema>,
    Option<proximadb_catalog::CatalogTableStatistics>,
)> {
    use crate::query::table_write_plan::{ReadSource, SnapshotRef};

    let source_table = request
        .source_table
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let source_sql = request
        .source_sql
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    match (source_table, source_sql) {
        (Some(_), Some(_)) => Err(ApiError::InvalidArgument(
            "Provide either source_table or source_sql, not both".to_string(),
        )),
        (None, None) => Err(ApiError::InvalidArgument(
            "source_table or source_sql is required".to_string(),
        )),
        (Some(table_name), None) => {
            let table = logical_table_ref_from_name(table_name)?;
            let (catalog, table_id) = state
                .catalog_manager
                .resolve_table(&table.qualified_name())
                .await
                .map_err(|error| ApiError::NotFound(error.to_string()))?;
            if !catalog
                .table_exists(&table_id)
                .await
                .map_err(|error| ApiError::Internal(error.to_string()))?
            {
                return Err(ApiError::NotFound(format!(
                    "Source table '{}' does not exist",
                    table.qualified_name()
                )));
            }
            let schema = catalog
                .get_table(&table_id)
                .await
                .map_err(|error| ApiError::Internal(error.to_string()))?;
            let stats = catalog.get_statistics(&table_id).await.unwrap_or_default();
            Ok((
                ReadSource::CatalogTable {
                    table,
                    snapshot: SnapshotRef::Latest,
                },
                Some(schema),
                Some(stats),
            ))
        }
        (None, Some(sql)) => Ok((ReadSource::QuerySql(sql.to_string()), None, None)),
    }
}

fn logical_table_ref_from_name(
    name: &str,
) -> ApiResult<crate::query::table_write_plan::LogicalTableRef> {
    let parts: Vec<_> = name
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    let Some(table_name) = parts.last() else {
        return Err(ApiError::InvalidArgument(
            "table name cannot be empty".to_string(),
        ));
    };
    Ok(crate::query::table_write_plan::LogicalTableRef {
        namespace: parts[..parts.len().saturating_sub(1)]
            .iter()
            .map(|part| (*part).to_string())
            .collect(),
        name: (*table_name).to_string(),
    })
}

fn parse_table_write_mode(
    mode: Option<&str>,
) -> ApiResult<crate::query::table_write_plan::WriteMode> {
    use crate::query::table_write_plan::WriteMode;

    match mode
        .unwrap_or("append")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "append" => Ok(WriteMode::Append),
        "insert" | "insert_only" | "insert-only" => Ok(WriteMode::InsertOnly),
        "upsert" => Ok(WriteMode::Upsert),
        "overwrite" | "insert_overwrite" | "insert-overwrite" | "overwrite_table"
        | "overwrite-table" => Ok(WriteMode::OverwriteTable),
        "merge" => Ok(WriteMode::Merge),
        other => Err(ApiError::InvalidArgument(format!(
            "Unsupported table write mode '{}'",
            other
        ))),
    }
}

fn parse_distribution_mode(
    mode: Option<&str>,
) -> ApiResult<crate::query::table_write_plan::DistributionMode> {
    use crate::query::table_write_plan::DistributionMode;

    match mode.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(DistributionMode::Auto),
        "local" | "local_only" | "local-only" => Ok(DistributionMode::LocalOnly),
        "pseudo" | "pseudo_distributed" | "pseudo-distributed" => {
            Ok(DistributionMode::PseudoDistributed)
        }
        "distributed" => Ok(DistributionMode::Distributed),
        other => Err(ApiError::InvalidArgument(format!(
            "Unsupported distribution mode '{}'",
            other
        ))),
    }
}

/// Helper: convert proto SqlValue to serde_json::Value (temporary until full internal refactor)
fn sql_value_to_json(v: &proximadb_v1::SqlValue) -> serde_json::Value {
    proximadb_records::conversions::sql_value_to_json(v)
}

// =============================================================================
// Hybrid Search (BM25 + Vector with RRF Fusion)
// =============================================================================

/// Compatibility alias for vector search input
/// Maps internal VectorResult to simple wrapper used by handlers
#[allow(dead_code)]
struct VectorSearchInput {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    score: f32,
}

/// Legacy request body for hybrid search — DEAD CODE kept only for
/// deserialization-shape tests.
///
/// The hybrid_search handler that used this struct was removed (see
/// comment ~line 805 in this file): "Routes are now served by
/// create_hybrid_search_router() from proximadb-api backed by
/// Bm25IndexPortImpl and RestHybridPortImpl."
///
/// Naming note: this type used to be called `HybridSearchRequest` and
/// collided with `crate::proto::v1::HybridSearchRequest` (proto wire form
/// for vector+graph hybrid) AND with
/// `src/network/rest/canonical/hybrid.rs::ExperimentalHybridSearchRequest`.
/// Renamed to `LegacyHybridSearchRequest` to mark it as deprecated; the
/// struct can be deleted once the deserialization tests are moved to the
/// active proximadb-api hybrid handlers.
#[derive(Debug, Deserialize)]
pub struct LegacyHybridSearchRequest {
    /// Collection to search
    pub collection: String,
    /// Query vector for similarity search (optional if keyword-only)
    pub vector: Option<Vec<f32>>,
    /// Text query for BM25 keyword search (optional if vector-only)
    pub text_query: Option<String>,
    /// Number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Weight for vector results (0.0-1.0). BM25 weight = 1.0 - vector_weight.
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    /// RRF constant k (default 60)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,
    /// Minimum BM25 score threshold
    #[serde(default)]
    pub min_bm25_score: f64,
}

fn default_top_k() -> usize {
    10
}
fn default_vector_weight() -> f32 {
    0.5
}
fn default_rrf_k() -> u32 {
    60
}

/// Request body for indexing text documents for hybrid search
#[derive(Debug, Deserialize)]
pub struct HybridIndexRequest {
    /// Collection name
    pub collection: String,
    /// Documents to index: list of {id, text}
    pub documents: Vec<HybridDocument>,
}

/// A text document for hybrid search indexing
#[derive(Debug, Deserialize)]
pub struct HybridDocument {
    /// Document/vector ID
    pub id: String,
    /// Text content to index
    pub text: String,
}

/// Legacy response for hybrid search — DEAD CODE paired with
/// `LegacyHybridSearchRequest` (see ~line 695). The handler that produced
/// this response was removed (routes now served by proximadb-api's
/// create_hybrid_search_router backed by RestHybridPortImpl). The struct
/// survives only for serialization-shape tests in this file + tests/rest_api_v1_test.rs.
///
/// Naming note: renamed from `HybridSearchResponse` to
/// `LegacyHybridSearchResponse` to mark it as dead-code-eligible and to
/// distinguish from the proto wire form
/// `crate::proto::v1::HybridSearchResponse`.
#[derive(Debug, Serialize)]
pub struct LegacyHybridSearchResponse {
    /// Whether the search completed successfully
    pub success: bool,
    /// Fused search result hits
    pub results: Vec<HybridSearchHit>,
    /// Total number of results
    pub total: usize,
    /// Server-side processing time in microseconds
    pub processing_time_us: u64,
    /// Search mode used (e.g., "hybrid", "vector_only", "bm25_only")
    pub mode: String,
}

/// A single hybrid search result hit
#[derive(Debug, Serialize)]
pub struct HybridSearchHit {
    /// Vector/document identifier
    pub id: String,
    /// Fused score combining vector and BM25 signals
    pub combined_score: f64,
    /// Vector similarity score (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
    /// BM25 text relevance score (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f64>,
    /// Rank in vector-only results (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<usize>,
    /// Rank in BM25-only results (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_rank: Option<usize>,
    /// BM25 terms that matched the query
    pub matched_terms: Vec<String>,
}

/// Response for hybrid index operations
#[derive(Debug, Serialize)]
pub struct HybridIndexResponse {
    /// Whether the indexing operation succeeded
    pub success: bool,
    /// Collection that was indexed
    pub collection: String,
    /// Number of documents indexed in this operation
    pub documents_indexed: usize,
    /// Total number of documents in the full-text index
    pub total_documents: usize,
}

/// Shared state for per-collection full-text indices
pub type FullTextIndexMap = Arc<
    std::sync::RwLock<
        std::collections::HashMap<
            String,
            crate::storage::engines::core::formats::columnar::fulltext_index::FullTextIndex,
        >,
    >,
>;

// hybrid_index and hybrid_search handlers removed.
// Routes are now served by create_hybrid_search_router() from proximadb-api
// backed by Bm25IndexPortImpl and RestHybridPortImpl.

/// Create router with all REST endpoints
/// Operator / control-plane `/api/v2/...` routes.
///
/// These are intentionally separate from `create_v2_router` (the SDK data-plane:
/// collections / records / search / query, which feeds the published OpenAPI
/// spec via `ApiDoc`). They are NOT utoipa-annotated and are excluded from the
/// spec by convention — they are operator / experimental surfaces (ranking
/// pipeline, collection pinning/affinity, primary-pod placement, catalog
/// routing explain, progressive search, graph-branch merge).
///
/// Returned as `Router<AppState>` and merged into the base router inside
/// `create_router`, so they inherit the same default-tenant +
/// deprecation-header layers as every other route (no behaviour change vs. the
/// prior inline registration). Keeping them in one named builder is the
/// "single source of truth" for which `/api/v2` routes are non-SDK.
fn rank_json_rejection_response(status: StatusCode, message: String) -> Response {
    let error_type = match status {
        StatusCode::UNSUPPORTED_MEDIA_TYPE => "unsupported_media_type",
        StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_body",
        _ => "invalid_json",
    };
    (
        status,
        JsonResponse(serde_json::json!({
            "error": {
                "type": error_type,
                "message": message,
                "code": status.as_u16()
            }
        })),
    )
        .into_response()
}

fn operator_and_control_v2_routes() -> axum::Router<AppState> {
    use axum::routing::{get, post};

    let router = axum::Router::new()
        .route(
            "/api/v2/progressive/search/{collection_id}",
            post(crate::network::rest::progressive_search_handler::progressive_search_handler),
        )
        .route(
            "/api/v2/catalog/table-routing",
            get(get_catalog_table_routing),
        )
        .route(
            "/api/v2/catalog/table-write/explain",
            post(explain_table_write_route),
        )
        .route(
            "/api/v2/collections/{collection_id}/branches/{branch}/merge",
            post(merge_graph_branch),
        )
        // Multi-phase ranking pipeline (R-7c.1).
        // `rank_search_dispatch` runs the arena-bearing first phase
        // inside `tokio::task::spawn_blocking` so the outer future
        // stays `Send` — required by axum's tokio multi-threaded
        // runtime, blocked previously by `bumpalo::Bump` being `!Sync`.
        // The route returns:
        //   200 — successful rank pipeline result
        //   400/415/422 — canonical JSON errors for request extraction
        //   404 — named profile not found
        //   501 — RankServices not injected at startup
        //   500 — pipeline failure (model load, expression compile, …)
        .route(
            "/api/v2/rank/search",
            post(
                |State(state): State<AppState>,
                 payload: Result<
                    Json<crate::network::rest::canonical::rank::RankSearchRequest>,
                    axum::extract::rejection::JsonRejection,
                >| async move {
                    match payload {
                        Ok(Json(req)) => {
                            match crate::network::rest::canonical::rank::rank_search_dispatch(
                                state, req,
                            )
                            .await
                            {
                                Ok(response) => Json(response).into_response(),
                                Err(error) => error.into_response(),
                            }
                        }
                        Err(rejection) => {
                            rank_json_rejection_response(rejection.status(), rejection.body_text())
                        }
                    }
                },
            ),
        )
        // Phase 6: per-collection pinning control surface.
        // Operators PATCH a collection's pin state; the AxisTieringManager
        // honors the override on its next evaluation. See
        // src/storage/collection_pinning.rs for the control/data plane split.
        .route(
            "/api/v2/collections/{collection_id}/pin",
            axum::routing::patch(crate::network::rest::canonical::pinning::patch_pin)
                .get(crate::network::rest::canonical::pinning::get_pin),
        )
        .route(
            "/api/v2/collections/pinning",
            get(crate::network::rest::canonical::pinning::list_pins),
        )
        // Phase 7.2.4: per-collection cache-affinity inspect/invalidate.
        // Operators read or drop the affinity hint for routing
        // re-evaluation. See src/cluster/cache_affinity.rs.
        .route(
            "/api/v2/collections/{collection_id}/affinity",
            get(crate::network::rest::canonical::affinity::get_affinity)
                .delete(crate::network::rest::canonical::affinity::delete_affinity),
        )
        .route(
            "/api/v2/collections/affinity",
            get(crate::network::rest::canonical::affinity::list_affinity),
        )
        // Slice 3 of tenant-pod-affinity: per-(tenant, collection)
        // primary-pod operator API. Auth-gated inside each handler
        // to `SystemAdmin` or `ConfigureSystem` — these endpoints
        // expose cross-tenant placement and drive WAL write routing,
        // so they MUST NOT be reachable by regular tenants. The
        // auth_middleware_unified layer authenticates the request;
        // `authorize_operator` then checks for the operator-level
        // permission.
        .route(
            "/api/v2/primary-pod/{tenant_id}/{collection_id}",
            get(crate::network::rest::canonical::primary_pod::get_primary_pod)
                .put(crate::network::rest::canonical::primary_pod::put_primary_pod)
                .delete(crate::network::rest::canonical::primary_pod::delete_primary_pod),
        )
        .route(
            "/api/v2/primary-pod",
            get(crate::network::rest::canonical::primary_pod::list_primary_pods),
        );

    // TD-ABAC control-plane (PR-B writes + read endpoints): the ABAC
    // policy-provisioning admin API. Write handlers go through the shared
    // enforcer stores (hot-reload), resolve tenant string→u64 at write time, and
    // are auth-gated to the operator role; read handlers inspect the live policy.
    // `abac-policy`-only — default builds omit these routes entirely.
    #[cfg(feature = "abac-policy")]
    let router = router
        .route(
            "/api/v2/abac/policy-bindings/{tenant}/{object_id}",
            axum::routing::put(crate::network::rest::canonical::abac_admin::put_policy_binding)
                .delete(crate::network::rest::canonical::abac_admin::delete_policy_binding),
        )
        .route(
            "/api/v2/abac/policy-bindings/{tenant}",
            axum::routing::get(crate::network::rest::canonical::abac_admin::get_policy_bindings),
        )
        .route(
            "/api/v2/abac/tenant-posture/{tenant}",
            axum::routing::get(crate::network::rest::canonical::abac_admin::get_tenant_posture)
                .put(crate::network::rest::canonical::abac_admin::put_tenant_posture),
        )
        .route(
            "/api/v2/abac/grants",
            axum::routing::post(crate::network::rest::canonical::abac_admin::post_grant),
        )
        .route(
            "/api/v2/abac/grants/{owner_tenant}",
            axum::routing::get(crate::network::rest::canonical::abac_admin::list_grants),
        )
        .route(
            "/api/v2/abac/grants/{owner_tenant}/{grant_id}",
            axum::routing::delete(crate::network::rest::canonical::abac_admin::delete_grant),
        )
        .route(
            "/api/v2/abac/attribute-bindings",
            axum::routing::get(
                crate::network::rest::canonical::abac_admin::list_attribute_bindings,
            )
            .post(crate::network::rest::canonical::abac_admin::post_attribute_binding),
        )
        .route(
            "/api/v2/abac/predicate-objects",
            axum::routing::get(crate::network::rest::canonical::abac_admin::list_predicate_objects),
        )
        .route(
            "/api/v2/abac/predicate-objects/{object_id}",
            axum::routing::get(crate::network::rest::canonical::abac_admin::get_predicate_object)
                .put(crate::network::rest::canonical::abac_admin::put_predicate_object)
                .delete(crate::network::rest::canonical::abac_admin::delete_predicate_object),
        );

    router
}

pub fn create_router(state: AppState) -> axum::Router {
    use axum::routing::get;

    info!("🔵 REST API: Creating router with collection endpoints...");

    // Initialize SKS in-memory store (v1) using the same storage engine as vector operations
    let entities_router = {
        use crate::network::rest::canonical::entities::{self, EntityApiState};
        use crate::storage::entity_store::{
            CsrRelationsStore, InMemoryProvenanceRegistry, ProximaEntityStore,
        };

        let engine = state.vector_operations_service.unified_engine();
        let legacy_store = ProximaEntityStore::with_vector_service(
            engine,
            Arc::new(CsrRelationsStore::new()),
            Arc::new(InMemoryProvenanceRegistry::new()),
            state.vector_operations_service.clone(),
        );

        // Register legacy store globally for compatibility (entity API currently uses legacy store).
        let legacy_arc = Arc::new(legacy_store);
        ProximaEntityStore::register_global(legacy_arc.clone());

        // Use the same Arc - no need to clone the inner value
        let store = legacy_arc.clone();
        // Register store globally for hybrid executor access (embedding catalog)
        crate::storage::entity_store::ProximaEntityStore::register_global(store.clone());
        let entity_state = EntityApiState { store };
        entities::configure_routes().with_state(entity_state)
    };

    let mut router = axum::Router::new()
        // Operator / control-plane `/api/v2/...` routes — NOT part of the
        // published SDK surface (deliberately excluded from `ApiDoc` / the
        // OpenAPI spec by convention; see `openapi.rs`). Extracted into
        // `operator_and_control_v2_routes` for readability and merged here so
        // they share `create_router`'s default-tenant + deprecation-header
        // layers unchanged (zero behaviour change vs. the prior inline list).
        .merge(operator_and_control_v2_routes())
        // Health check endpoints
        .route("/health", get(comprehensive_health_check))
        .route("/health/live", get(liveness_check))
        .route("/health/ready", get(readiness_check))
        // SKS entity endpoints (storage-coupled path)
        .nest("/api", entities_router);

    // Graph routes — port-backed (always wired in production since Phase 9.12).
    //
    // Mounted only at the canonical `/api/v2` prefix (`/api/v2/graphs/...`).
    // The legacy `/api/v1/graph` mount (which produced the awkward doubled
    // `/api/v1/graph/graphs/...` paths) was removed in the API standardization
    // hard-rename; the in-router `*_legacy` redirects now point at `/api/v2`.
    if let Some(ref gp) = state.graph_port {
        use proximadb_api::rest::{GraphRestState, create_graph_router};
        let graph_state = GraphRestState {
            graph_port: gp.clone(),
        };
        router = router.nest("/api/v2", create_graph_router().with_state(graph_state));
        info!("✅ Graph API routing via port-based handler (proximadb-api) — /api/v2/graphs/*");
    }

    // Collection and vector CRUD are served exclusively by the canonical v2
    // router (`create_v2_router`, mounted at `/api/v2` in `server.rs`). The
    // legacy v1 surfaces (`/api/v1/collections`, `/api/v1/search`,
    // `/api/v1/vectors/*`) were removed in the API standardization hard-rename;
    // clients use `/api/v2/collections/*` (records/search). `state.api_handlers`
    // remains the canonical write/read port the v2 handlers delegate to.

    // Document routes — port-backed (always wired in production since Phase 9)
    if let Some(ref dp) = state.doc_port {
        use proximadb_api::rest::{DocumentRestState, create_document_router};
        router = router.nest(
            "/api/v2/document-collections",
            create_document_router().with_state(DocumentRestState {
                document_port: dp.clone(),
            }),
        );
        info!("✅ Document API routing via port-based handler (proximadb-api)");
    }

    // Observability routes — port-backed (always wired in production since Phase 9)
    if let Some(ref op) = state.obs_port {
        use proximadb_api::rest::{ObservabilityRestState, create_observability_router};
        router = router.nest(
            "/api/v2/observability",
            create_observability_router().with_state(ObservabilityRestState {
                observability_port: op.clone(),
            }),
        );
        info!("✅ Observability API routing via port-based handler (proximadb-api)");
    }

    // Unified multi-model query routes — port-backed (always wired in production since Phase 9.9)
    if let Some(ref uq_port) = state.unified_query_port {
        use proximadb_api::rest::UnifiedQueryRestState;
        let uq_state = UnifiedQueryRestState {
            unified_query_port: uq_port.clone(),
        };
        router = router.nest(
            "/api/v2/unified",
            proximadb_api::rest::create_multimodal_router().with_state(uq_state),
        );
        info!("✅ Unified Query API enabled at /api/v2/unified (port-backed)");
    }

    // Hybrid search — port-backed via RestHybridPortImpl (real BM25+vector fusion).
    // Shares the in-process HybridFullTextIndexMap with Bm25IndexPortImpl so indexed
    // documents are immediately searchable without a separate startup step.
    //
    // Built here but MERGED at the very end, after the blanket default-tenant
    // `Extension` layers below: those layers sit INSIDE the tenant middleware on
    // the request path and would overwrite the middleware's per-request
    // `proximadb_api::rest::TenantContext` with "default" — which is exactly how
    // the hybrid filter gate lost the caller's tenant (#949). Routes merged
    // after `.layer(...)` are not wrapped by it; the hybrid handler falls back
    // gracefully (`Option<Extension<..>>`) in mounts without the middleware.
    let hybrid_router = {
        use crate::network::hybrid_search::{Bm25IndexPortImpl, RestHybridPortImpl};
        use proximadb_api::rest::{HybridRestState, create_hybrid_search_router};

        let indexes = state
            .fulltext_indexes
            .clone()
            .unwrap_or_else(|| Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())));
        let vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort> =
            state.vector_operations_service.clone();
        let hybrid_port: Arc<dyn proximadb_runtime::HybridPort> =
            Arc::new(RestHybridPortImpl::new(vector_ops, indexes.clone()));
        let bm25_port: Arc<dyn proximadb_runtime::BM25IndexPort> =
            Arc::new(Bm25IndexPortImpl::new(indexes));
        let hybrid_state = HybridRestState {
            hybrid_port,
            bm25_port: Some(bm25_port),
        };
        info!("✅ Hybrid search at /api/v2/hybrid/* via RestHybridPortImpl (real BM25+vector)");
        create_hybrid_search_router().with_state(hybrid_state)
    };

    // SQL execution + explain are served by the canonical v2 query facade
    // (`/api/v2/query`, `/api/v2/query/explain` in `create_v2_router`). The
    // legacy `/api/v1/sql/execute` + `/api/v1/sql/explain` routes were removed
    // in the API standardization hard-rename.

    // Optional enterprise catalog endpoints
    #[cfg(feature = "enterprise-catalogs")]
    {
        let catalog_router = {
            use crate::network::rest::canonical::catalog::{self, CatalogApiState};

            let catalog_state = CatalogApiState::new(state.catalog_manager.clone());
            catalog::configure_routes().with_state(catalog_state)
        };
        // Nested under /api/v2 (the sole canonical prefix, SUPPORTED_SURFACE):
        // this surface previously lived at /api/v1/catalogs, where the v1_sunset
        // middleware 410'd it by default (TD-V1SUNSET-1 census finding 4) —
        // dead-on-arrival whenever the feature was on. Clean cut, no alias: the
        // surface was never reachable in a default deployment (double-prefix bug
        // until TD-CAT-8, then 410), and no SDK or client references the path.
        router = router.nest("/api/v2/catalogs", catalog_router);
        info!("✅ External Catalog API endpoints enabled at /api/v2/catalogs");
    }

    // Iceberg REST Catalog server — always on, no feature gate needed.
    // External engines (Spark, Trino, DuckDB, PyIceberg) connect via /iceberg/v1.
    {
        use crate::network::rest::canonical::iceberg_rest_catalog::{
            IcebergRestState, create_iceberg_rest_router,
        };
        let mut iceberg_state = IcebergRestState::with_defaults(state.catalog_manager.clone())
            .with_segment_registry(state.segment_registry.clone());
        // TD-119: persist/serve a real metadata.json from the SAME warehouse root the
        // materialize/write path uses (`file://{data_dir}/warehouse`), so the manifest log
        // drives snapshot history and `metadata-location` resolves. Best-effort: if the
        // store is unavailable, metadata is still served inline (synthesized, no persistence).
        let warehouse_root_url = format!("file://{}/warehouse", state.data_dir.display());
        match proximadb_iceberg_engine::IcebergObjectStoreBridge::from_url(&warehouse_root_url) {
            Ok(bridge) => {
                iceberg_state = iceberg_state.with_object_store_bridge(std::sync::Arc::new(bridge));
            }
            Err(e) => {
                info!(
                    "Iceberg REST: warehouse object store unavailable at {warehouse_root_url}: \
                     {e}; metadata served inline without persistence"
                );
            }
        }
        router = router.nest(
            "/iceberg/v1",
            create_iceberg_rest_router().with_state(iceberg_state),
        );
        info!(
            "✅ Iceberg REST Catalog server at /iceberg/v1 (Spark/Trino/DuckDB/PyIceberg compatible)"
        );
    }

    // Experimental hybrid API removed 2026-05-26: it returned mock-backed
    // results that misled customers into thinking the endpoint computed real
    // BM25+vector fusion. The production hybrid endpoint at `/api/v1/hybrid/search`
    // (port-backed via `RestHybridPortImpl`) is the supported path; gRPC parity
    // landed in commit 6a73ead7f. The module at `src/network/rest/canonical/hybrid.rs`
    // remains in-tree for reference but is no longer mounted.

    // Read-only collection analytics (Entanglement Index, etc.) — TD-043 sub-2
    let analytics_router = {
        let analytics_state = AnalyticsApiState::new(Some(state.vector_operations_service.clone()));
        analytics::create_router().with_state(analytics_state)
    };
    router = router.nest("/api/v2/analytics", analytics_router);
    info!("✅ Analytics API endpoints enabled at /api/v2/analytics");

    // Agentic Query Language (RUBICON / AQL) — TD-050
    let aql_router = {
        let mut executor = AqlExecutor::new();

        // Attach event log for persistent audit trails (TD-050 Phase 5)
        if let Some(log) = &state.event_log {
            executor = executor.with_event_log(log.clone());
        }

        // Register sources
        executor.register_source(
            "vector".to_string(),
            Arc::new(VectorAqlSource::new(
                state.vector_operations_service.clone(),
            )),
        );
        executor.register_source(
            "graph".to_string(),
            Arc::new(GraphAqlSource::new(state.graph_execution_service.clone())),
        );
        executor.register_source(
            "document".to_string(),
            Arc::new(DocumentAqlSource::new(state.document_service.clone())),
        );
        executor.register_source(
            "observability".to_string(),
            Arc::new(ObservabilityAqlSource::new(
                state.observability_service.clone(),
            )),
        );

        // Agent-memory read surface (TD-100, ADR-022): scoped + fused + audited.
        // Converges on the existing scoped vector search + BM25 + RRF fusion —
        // no new storage path. The lexical leg reuses the same in-process
        // HybridFullTextIndexMap as /api/v1/hybrid.
        {
            let mem_indexes = state.fulltext_indexes.clone().unwrap_or_else(|| {
                Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()))
            });
            let mem_vector_port: Arc<dyn proximadb_runtime::VectorOpsPort> =
                state.vector_operations_service.clone();
            let lexical = Arc::new(
                crate::network::rest::canonical::rank_backend::ProductionHybridBackend::new(
                    mem_vector_port,
                    mem_indexes,
                ),
            );
            executor.register_source(
                "memory".to_string(),
                Arc::new(
                    MemoryAqlSource::new(state.vector_operations_service.clone())
                        .with_lexical_backend(lexical),
                ),
            );
        }

        let aql_state = AqlApiState::new(executor);
        aql::create_router().with_state(aql_state)
    };
    router = router.nest("/api/v2/aql", aql_router);
    info!("✅ AQL (RUBICON) API endpoints enabled at /api/v2/aql");

    // Agent-memory write surface (TD-101) — POST /api/v2/memory/ingest.
    // Always mounted; the engine is built only when an LLM backend is present
    // (extraction + consolidation need it). Reuses VectorOperationsService +
    // the in-process EmbeddingService — no new storage path (ADR-022).
    {
        use crate::network::rest::canonical::memory::{self, MemoryApiState};
        use crate::services::agent_memory::{
            EmbeddingServiceEmbedder, EventLogConsolidationAuditSink, LlmConsolidationAgent,
            LlmExtractionAgent, MemoryWriteEngine, VectorMemoryStore,
        };
        let memory_engine = state.llm_engine.as_ref().map(|llm| {
            let extractor = Arc::new(LlmExtractionAgent::new(llm.clone()));
            let consolidator = Arc::new(LlmConsolidationAgent::new(llm.clone()));
            let store = Arc::new(VectorMemoryStore::new(
                state.vector_operations_service.clone(),
                Arc::new(EmbeddingServiceEmbedder),
            ));
            let mut engine = MemoryWriteEngine::new(extractor, consolidator, store);
            // Persist every consolidation decision to the shared audit log when
            // available (ADR-022 auditable memory). Same Arc<EventLogEngine> the
            // AQL audit trail uses; mirrors the AQL wiring above.
            if let Some(event_log) = &state.event_log {
                engine = engine.with_audit_sink(Arc::new(
                    EventLogConsolidationAuditSink::from_event_log(event_log.clone()),
                ));
            }
            Arc::new(engine)
        });
        let mounted = memory_engine.is_some();
        // The read surface (GET /consolidation/{session}) needs only the audit
        // log, independent of the LLM backend — pass the same shared Arc the
        // write sink persists to, so a deployment with no LLM can still serve
        // any audit trail it already recorded.
        let memory_state = MemoryApiState::new(memory_engine, state.event_log.clone());
        router = router.nest(
            "/api/v2/memory",
            memory::create_router().with_state(memory_state),
        );
        if mounted {
            info!("✅ Agent-memory write endpoint at /api/v2/memory/ingest (TD-101)");
        } else {
            info!(
                "✅ Agent-memory write endpoint mounted at /api/v2/memory/ingest but inactive (no LLM backend)"
            );
        }
    }

    // Natural Language Query Translation (AV-SQL) — TD-048
    if let Some(llm) = &state.llm_engine {
        let nl_state = NlApiState::new(llm.clone());
        let nl_router = nl::create_router().with_state(nl_state);
        router = router.nest("/api/v2/nl", nl_router);
        info!("✅ Natural Language (AV-SQL) API endpoints enabled at /api/v2/nl");
    } else {
        warn!("⚠️ LLM engine not available; Natural Language (AV-SQL) endpoints disabled");
    }

    // Convert to Router<()> by providing state, with default tenant context for all routes
    let default_tenant = TenantContext::new(
        "default",
        crate::network::middleware::tenant::TenantIdSource::Default,
    );
    let default_api_tenant = proximadb_api::rest::TenantContext {
        tenant_id: "default".to_string(),
    };
    let router = router
        .with_state(state)
        .layer(Extension(default_tenant))
        .layer(axum::Extension(default_api_tenant))
        .layer(axum::middleware::from_fn(add_rest_v1_deprecation_headers))
        // Merged AFTER the layers above so the per-request tenant from the
        // tenant middleware is not clobbered by the default-tenant Extension
        // (#949). The hybrid router carries its own v1-compat headers.
        .merge(hybrid_router);

    info!("✅ REST API: Router created with canonical v2 routes:");
    info!(
        "   POST   /api/v2/collections/:collection/records/batch (canonical ProximaRecord writes)"
    );
    info!("   POST   /api/v2/collections/:collection/search (canonical record/vector search)");
    info!(
        "   GET    /api/v2/collections/:collection/records/:record (canonical ProximaRecord fetch)"
    );
    // NOTE: the legacy /api/v1 vector/collection/search/hybrid REST surfaces were
    // removed in the API-standardization hard-rename; those paths now fall through
    // to a migration hint redirecting to /api/v2 (see `v1_replacement_for`). There
    // is NO live /api/v1 mount: the enterprise catalog surface moved to
    // /api/v2/catalogs (2026-08-29; it was 410'd at its old /api/v1/catalogs home
    // by the default-off sunset), and the Iceberg REST catalog serves at
    // /iceberg/v1. Do not re-advertise removed routes here — that misleads
    // callers into thinking v1 REST is still served.
    // (/api/v1/experimental/hybrid/search was removed 2026-05-26; it was
    // mock-backed. The real hybrid path is /api/v2/hybrid/search.)

    router
}

/// Comprehensive health check handler
///
/// Wraps the health module's health_check function with our AppState
pub async fn comprehensive_health_check(
    State(state): State<AppState>,
    query: Query<health::HealthParams>,
) -> ApiResult<Json<health::HealthResponse>> {
    let health_state = state.health_state();
    health::health_check(axum::extract::State(health_state), query).await
}

/// Liveness check handler
///
/// Simple liveness check for load balancers
pub async fn liveness_check(
    State(state): State<AppState>,
) -> ApiResult<Json<health::LivenessResponse>> {
    let health_state = state.health_state();
    health::liveness_check(axum::extract::State(health_state)).await
}

/// Readiness check handler
///
/// Returns 200 when ready, 503 when not ready
pub async fn readiness_check(
    State(state): State<AppState>,
) -> Result<Json<health::ReadinessResponse>, (StatusCode, Json<health::ReadinessResponse>)> {
    let health_state = state.health_state();
    health::readiness_check(axum::extract::State(health_state)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::rest::proto_json::ProtoApiResponse;
    use crate::proto::proximadb_v1::{CollectionOperation, CollectionRequest};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::collections::HashMap;
    use std::path::Path;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn branch_merge_test_entry(
        seq: u64,
        collection_id: &str,
        oid: &str,
        branch_id: Option<&str>,
    ) -> proximadb_storage_common::CanonicalWalEntry {
        let mut record = proximadb_records::ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        };
        record.branch_id = branch_id.map(String::from);
        proximadb_storage_common::CanonicalWalEntry::new(
            seq,
            proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id: collection_id.to_string(),
                record: Box::new(record),
                projections: vec![],
            },
            None,
        )
    }

    #[test]
    fn branch_merge_request_defaults_to_main_dry_run() {
        let request: GraphBranchMergeRequest = serde_json::from_value(serde_json::json!({}))
            .expect("empty request should use defaults");
        assert_eq!(request.target_branch, "main");
        assert!(request.dry_run);
    }

    #[test]
    fn branch_merge_collection_filter_keeps_only_matching_wal_entries() {
        let entries = vec![
            branch_merge_test_entry(1, "graph_a", "base", None),
            branch_merge_test_entry(2, "graph_a", "left", Some("feature")),
            branch_merge_test_entry(3, "graph_b", "right", Some("feature")),
        ];

        let filtered = filter_canonical_wal_for_collection(entries, "graph_a");
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|entry| match &entry.operation {
            proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id, ..
            } => collection_id == "graph_a",
            _ => false,
        }));
    }

    #[tokio::test]
    async fn v1_deprecation_middleware_marks_api_v1_only() {
        let router = axum::Router::new()
            .route("/api/v1/ping", axum::routing::get(|| async { "ok" }))
            .route("/api/v2/ping", axum::routing::get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(add_rest_v1_deprecation_headers));

        let v1_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(v1_response.status(), StatusCode::OK);
        assert_eq!(
            v1_response
                .headers()
                .get("deprecation")
                .and_then(|value| value.to_str().ok()),
            Some("true")
        );
        assert_eq!(
            v1_response
                .headers()
                .get("x-proximadb-api-status")
                .and_then(|value| value.to_str().ok()),
            Some("deprecated-compatibility")
        );

        let v2_response = router
            .oneshot(
                Request::builder()
                    .uri("/api/v2/ping")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(v2_response.status(), StatusCode::OK);
        assert!(v2_response.headers().get("deprecation").is_none());
    }

    fn file_url(path: &Path) -> String {
        format!("file://{}", path.to_string_lossy())
    }

    // ── Mock ports ────────────────────────────────────────────────────────────
    // Minimal stubs that let tests build a fully-ported AppState without real
    // services.  Each method returns an error; the tested routes either fail
    // before reaching the port (JSON parse, empty-field guard) or are self-
    // contained redirects that never call the port at all.

    use async_trait::async_trait;

    struct MockDocumentPort;
    #[async_trait]
    impl proximadb_runtime::DocumentPort for MockDocumentPort {
        async fn create_collection(
            &self,
            _: crate::proto::v1::CreateDocumentCollectionRequest,
        ) -> anyhow::Result<crate::proto::v1::CreateDocumentCollectionResponse> {
            anyhow::bail!("mock")
        }
        async fn list_collections(
            &self,
            _: crate::proto::v1::ListDocumentCollectionsRequest,
        ) -> anyhow::Result<crate::proto::v1::ListDocumentCollectionsResponse> {
            anyhow::bail!("mock")
        }
        async fn delete_collection(
            &self,
            _: crate::proto::v1::DeleteDocumentCollectionRequest,
        ) -> anyhow::Result<crate::proto::v1::DeleteDocumentCollectionResponse> {
            anyhow::bail!("mock")
        }
        async fn insert_document(
            &self,
            _: crate::proto::v1::InsertDocumentRequest,
        ) -> anyhow::Result<crate::proto::v1::InsertDocumentResponse> {
            anyhow::bail!("mock")
        }
        async fn get_document(
            &self,
            _: crate::proto::v1::GetDocumentRequest,
        ) -> anyhow::Result<crate::proto::v1::GetDocumentResponse> {
            anyhow::bail!("mock")
        }
        async fn update_document(
            &self,
            _: crate::proto::v1::UpdateDocumentRequest,
        ) -> anyhow::Result<crate::proto::v1::UpdateDocumentResponse> {
            anyhow::bail!("mock")
        }
        async fn delete_document(
            &self,
            _: crate::proto::v1::DeleteDocumentRequest,
        ) -> anyhow::Result<crate::proto::v1::DeleteDocumentResponse> {
            anyhow::bail!("mock")
        }
        async fn query_documents(
            &self,
            _: crate::proto::v1::QueryDocumentsRequest,
        ) -> anyhow::Result<crate::proto::v1::QueryDocumentsResponse> {
            anyhow::bail!("mock")
        }
        async fn aggregate_documents(
            &self,
            _: crate::proto::v1::AggregateDocumentsRequest,
        ) -> anyhow::Result<crate::proto::v1::AggregateDocumentsResponse> {
            anyhow::bail!("mock")
        }
    }

    struct MockGraphPort;
    #[async_trait]
    impl proximadb_runtime::GraphPort for MockGraphPort {
        async fn create_node(
            &self,
            _: crate::proto::v1::CreateNodeRequest,
        ) -> anyhow::Result<crate::proto::v1::Node> {
            anyhow::bail!("mock")
        }
        async fn get_node(
            &self,
            _: crate::proto::v1::GetNodeRequest,
        ) -> anyhow::Result<crate::proto::v1::Node> {
            anyhow::bail!("mock")
        }
        async fn update_node(
            &self,
            _: crate::proto::v1::UpdateNodeRequest,
        ) -> anyhow::Result<crate::proto::v1::Node> {
            anyhow::bail!("mock")
        }
        async fn delete_node(
            &self,
            _: crate::proto::v1::DeleteNodeRequest,
        ) -> anyhow::Result<crate::proto::v1::Node> {
            anyhow::bail!("mock")
        }
        async fn create_edge(
            &self,
            _: crate::proto::v1::CreateEdgeRequest,
        ) -> anyhow::Result<crate::proto::v1::Edge> {
            anyhow::bail!("mock")
        }
        async fn get_edge(
            &self,
            _: crate::proto::v1::GetEdgeRequest,
        ) -> anyhow::Result<crate::proto::v1::Edge> {
            anyhow::bail!("mock")
        }
        async fn update_edge(
            &self,
            _: crate::proto::v1::UpdateEdgeRequest,
        ) -> anyhow::Result<crate::proto::v1::Edge> {
            anyhow::bail!("mock")
        }
        async fn delete_edge(
            &self,
            _: crate::proto::v1::DeleteEdgeRequest,
        ) -> anyhow::Result<crate::proto::v1::Edge> {
            anyhow::bail!("mock")
        }
        async fn query_nodes(
            &self,
            _: crate::proto::v1::NodeQuery,
        ) -> anyhow::Result<crate::proto::v1::BatchResponse> {
            anyhow::bail!("mock")
        }
        async fn query_edges(
            &self,
            _: crate::proto::v1::EdgeQuery,
        ) -> anyhow::Result<crate::proto::v1::BatchResponse> {
            anyhow::bail!("mock")
        }
        async fn execute_query(
            &self,
            _: crate::proto::v1::GraphQueryRequest,
        ) -> anyhow::Result<crate::proto::v1::GraphQueryResponse> {
            anyhow::bail!("mock")
        }
        async fn execute_query_json(
            &self,
            _graph_id: &str,
            _query: &str,
        ) -> anyhow::Result<Vec<serde_json::Value>> {
            anyhow::bail!("mock")
        }
        async fn get_neighbors(
            &self,
            _: crate::proto::v1::GetNeighborsRequest,
        ) -> anyhow::Result<crate::proto::v1::BatchResponse> {
            anyhow::bail!("mock")
        }
        async fn traverse_graph(
            &self,
            _: crate::proto::v1::TraversalRequest,
        ) -> anyhow::Result<crate::proto::v1::TraversalResponse> {
            anyhow::bail!("mock")
        }
        async fn stream_traverse(
            &self,
            _: crate::proto::v1::TraversalRequest,
        ) -> anyhow::Result<Vec<crate::proto::v1::TraversalChunk>> {
            anyhow::bail!("mock")
        }
        async fn get_graph_stats(
            &self,
            _: crate::proto::v1::GetStatsRequest,
        ) -> anyhow::Result<crate::proto::v1::GraphStats> {
            anyhow::bail!("mock")
        }
        async fn shortest_path(
            &self,
            _: crate::proto::v1::ShortestPathRequest,
        ) -> anyhow::Result<crate::proto::v1::ShortestPathResponse> {
            anyhow::bail!("mock")
        }
        async fn get_connected_components(
            &self,
            _: crate::proto::v1::GetStatsRequest,
        ) -> anyhow::Result<crate::proto::v1::ConnectedComponentsResponse> {
            anyhow::bail!("mock")
        }
        async fn has_cycle(
            &self,
            _: crate::proto::v1::GetStatsRequest,
        ) -> anyhow::Result<crate::proto::v1::CycleCheckResponse> {
            anyhow::bail!("mock")
        }
        async fn add_unique_constraint(
            &self,
            _: crate::proto::v1::UniqueConstraintRequest,
        ) -> anyhow::Result<crate::proto::v1::UniqueConstraintResponse> {
            anyhow::bail!("mock")
        }
        async fn remove_unique_constraint(
            &self,
            _: crate::proto::v1::UniqueConstraintRequest,
        ) -> anyhow::Result<crate::proto::v1::UniqueConstraintResponse> {
            anyhow::bail!("mock")
        }
        async fn batch_create_nodes(
            &self,
            _: crate::proto::v1::BatchNodeRequest,
        ) -> anyhow::Result<crate::proto::v1::BatchResponse> {
            anyhow::bail!("mock")
        }
        async fn batch_create_edges(
            &self,
            _: crate::proto::v1::BatchEdgeRequest,
        ) -> anyhow::Result<crate::proto::v1::BatchResponse> {
            anyhow::bail!("mock")
        }
        async fn execute_hybrid_query(
            &self,
            _: crate::proto::v1::HybridSearchRequest,
        ) -> anyhow::Result<crate::proto::v1::HybridSearchResponse> {
            anyhow::bail!("mock")
        }
        async fn create_graph_collection(
            &self,
            _: crate::proto::v1::CreateGraphRequest,
        ) -> anyhow::Result<crate::proto::v1::GraphCollection> {
            anyhow::bail!("mock")
        }
        async fn get_graph_collection(
            &self,
            _: String,
        ) -> anyhow::Result<Option<crate::proto::v1::GraphCollection>> {
            anyhow::bail!("mock")
        }
        async fn delete_graph_collection(&self, _: String) -> anyhow::Result<bool> {
            anyhow::bail!("mock")
        }
        async fn list_graph_collections(
            &self,
        ) -> anyhow::Result<Vec<crate::proto::v1::GraphCollection>> {
            anyhow::bail!("mock")
        }
        async fn update_graph_schema(
            &self,
            _: String,
            _: crate::proto::v1::GraphSchema,
        ) -> anyhow::Result<crate::proto::v1::GraphCollection> {
            anyhow::bail!("mock")
        }
    }

    struct MockObservabilityPort;
    #[async_trait]
    impl proximadb_runtime::ObservabilityPort for MockObservabilityPort {
        async fn create_namespace(
            &self,
            _: crate::proto::v1::CreateObservabilityNamespaceRequest,
        ) -> anyhow::Result<crate::proto::v1::CreateObservabilityNamespaceResponse> {
            anyhow::bail!("mock")
        }
        async fn list_namespaces(
            &self,
            _: crate::proto::v1::ListNamespacesRequest,
        ) -> anyhow::Result<crate::proto::v1::ListNamespacesResponse> {
            anyhow::bail!("mock")
        }
        async fn delete_namespace(
            &self,
            _: crate::proto::v1::DeleteNamespaceRequest,
        ) -> anyhow::Result<crate::proto::v1::DeleteNamespaceResponse> {
            anyhow::bail!("mock")
        }
        async fn ingest_logs(
            &self,
            _: crate::proto::v1::IngestLogsRequest,
        ) -> anyhow::Result<crate::proto::v1::IngestLogsResponse> {
            anyhow::bail!("mock")
        }
        async fn query_logs(
            &self,
            _: crate::proto::v1::QueryLogsRequest,
        ) -> anyhow::Result<crate::proto::v1::QueryLogsResponse> {
            anyhow::bail!("mock")
        }
        async fn stream_logs(
            &self,
            _: crate::proto::v1::QueryLogsRequest,
        ) -> anyhow::Result<Vec<crate::proto::v1::LogEntry>> {
            anyhow::bail!("mock")
        }
        async fn ingest_metrics(
            &self,
            _: crate::proto::v1::IngestMetricsRequest,
        ) -> anyhow::Result<crate::proto::v1::IngestMetricsResponse> {
            anyhow::bail!("mock")
        }
        async fn query_metrics(
            &self,
            _: crate::proto::v1::QueryMetricsRequest,
        ) -> anyhow::Result<crate::proto::v1::QueryMetricsResponse> {
            anyhow::bail!("mock")
        }
        async fn aggregate_metrics(
            &self,
            _: crate::proto::v1::AggregateMetricsRequest,
        ) -> anyhow::Result<crate::proto::v1::AggregateMetricsResponse> {
            anyhow::bail!("mock")
        }
        async fn ingest_traces(
            &self,
            _: crate::proto::v1::IngestTracesRequest,
        ) -> anyhow::Result<crate::proto::v1::IngestTracesResponse> {
            anyhow::bail!("mock")
        }
        async fn query_traces(
            &self,
            _: crate::proto::v1::QueryTracesRequest,
        ) -> anyhow::Result<crate::proto::v1::QueryTracesResponse> {
            anyhow::bail!("mock")
        }
        async fn get_trace(
            &self,
            _: crate::proto::v1::GetTraceRequest,
        ) -> anyhow::Result<crate::proto::v1::GetTraceResponse> {
            anyhow::bail!("mock")
        }
        async fn upsert_alert_rule(
            &self,
            _: crate::proto::v1::UpsertAlertRuleRequest,
        ) -> anyhow::Result<crate::proto::v1::UpsertAlertRuleResponse> {
            anyhow::bail!("mock")
        }
        async fn delete_alert_rule(
            &self,
            _: crate::proto::v1::DeleteAlertRuleRequest,
        ) -> anyhow::Result<crate::proto::v1::DeleteAlertRuleResponse> {
            anyhow::bail!("mock")
        }
        async fn list_alerts(
            &self,
            _: crate::proto::v1::ListAlertsRequest,
        ) -> anyhow::Result<crate::proto::v1::ListAlertsResponse> {
            anyhow::bail!("mock")
        }
    }

    struct MockUnifiedQueryPort;
    #[async_trait]
    impl proximadb_runtime::UnifiedQueryPort for MockUnifiedQueryPort {
        async fn execute_unified_query(
            &self,
            _: String,
            _: Option<Vec<proximadb_data_model::ProximaValue>>,
            _: Option<String>,
            _: Option<u32>,
        ) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
        async fn execute_multi_model_query(
            &self,
            _: serde_json::Value,
        ) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
        async fn execute_federated_query(
            &self,
            _: String,
            _: Option<Vec<proximadb_data_model::ProximaValue>>,
        ) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
        async fn execute_distributed_query(
            &self,
            _: serde_json::Value,
        ) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
        async fn explain_unified_query(
            &self,
            _: String,
            _: Option<String>,
        ) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
        async fn prepare_statement(
            &self,
            _: Option<String>,
            _: String,
            _: bool,
            _: Option<u64>,
        ) -> anyhow::Result<String> {
            anyhow::bail!("mock")
        }
        async fn execute_prepared(
            &self,
            _: String,
            _: Option<Vec<proximadb_data_model::ProximaValue>>,
            _: Option<String>,
        ) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
        async fn delete_prepared(&self, _: String) -> anyhow::Result<()> {
            anyhow::bail!("mock")
        }
        async fn get_prepared_stats(&self, _: Vec<String>) -> anyhow::Result<serde_json::Value> {
            anyhow::bail!("mock")
        }
    }

    async fn build_test_app_state() -> (AppState, TempDir) {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let storage_path = temp_dir.path().join("storage");
        let metadata_path = temp_dir.path().join("metadata");
        let data_dir = temp_dir.path().join("server_data");
        std::fs::create_dir_all(&storage_path).expect("failed to create storage path");
        std::fs::create_dir_all(&metadata_path).expect("failed to create metadata path");
        std::fs::create_dir_all(&data_dir).expect("failed to create data dir");

        let mut config = crate::core::config::Config::default();
        config.server.data_dir = data_dir.clone();
        config.storage.metadata_url = file_url(&metadata_path);
        config.storage.storage_locations = vec![crate::core::config::StorageLocation {
            url: file_url(&storage_path),
            weight: 1,
            tags: vec!["test".to_string()],
            io_budget: None,
        }];

        let (shared_services, _) = crate::network::multi_server::SharedServices::new(
            None,
            &config.storage,
            None,
            Some(&config),
            crate::network::multi_server::ServiceProfile::Embedded,
        )
        .await
        .expect("failed to initialize shared services for test app state");

        let state = AppState::new(
            RestCoreServices::from_shared_services(&shared_services),
            shared_services.graph_execution_service,
            None,
            data_dir,
            None,
            None,
        )
        .with_ports(
            Arc::new(MockDocumentPort),
            Arc::new(MockGraphPort),
            Arc::new(MockObservabilityPort),
        )
        .with_unified_query_port(Arc::new(MockUnifiedQueryPort));
        (state, temp_dir)
    }

    #[test]
    fn test_error_conversion() {
        let err = ApiError::CollectionNotFound("test_collection".to_string());
        let response = ProtoApiResponse::<()>::error(err);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn rank_json_rejections_use_canonical_error_envelope() {
        for (status, expected_type) in [
            (StatusCode::BAD_REQUEST, "invalid_json"),
            (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type"),
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request_body"),
        ] {
            let response = rank_json_rejection_response(status, "invalid rank request".into());
            assert_eq!(response.status(), status);
            assert_eq!(
                response.headers().get(axum::http::header::CONTENT_TYPE),
                Some(&axum::http::HeaderValue::from_static("application/json"))
            );
            let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["type"], expected_type);
            assert_eq!(body["error"]["message"], "invalid rank request");
            assert_eq!(body["error"]["code"], status.as_u16());
        }
    }

    #[test]
    fn test_hybrid_search_request_deserialization() {
        let json = serde_json::json!({
            "collection": "test_col",
            "vector": [0.1, 0.2, 0.3],
            "text_query": "machine learning",
            "top_k": 5,
            "vector_weight": 0.7
        });
        let req: LegacyHybridSearchRequest =
            serde_json::from_value(json).expect("failed to deserialize HybridSearchRequest");
        assert_eq!(req.collection, "test_col");
        assert_eq!(req.vector.expect("vector should be present").len(), 3);
        assert_eq!(
            req.text_query.expect("text_query should be present"),
            "machine learning"
        );
        assert_eq!(req.top_k, 5);
        assert!((req.vector_weight - 0.7).abs() < 0.001);
        assert_eq!(req.rrf_k, 60); // default
    }

    #[test]
    fn test_hybrid_search_request_defaults() {
        let json = serde_json::json!({
            "collection": "test_col",
            "text_query": "hello"
        });
        let req: LegacyHybridSearchRequest =
            serde_json::from_value(json).expect("failed to deserialize HybridSearchRequest");
        assert_eq!(req.top_k, 10);
        assert!((req.vector_weight - 0.5).abs() < 0.001);
        assert_eq!(req.rrf_k, 60);
        assert!(req.vector.is_none());
    }

    #[test]
    fn test_hybrid_index_request_deserialization() {
        let json = serde_json::json!({
            "collection": "test_col",
            "documents": [
                {"id": "doc1", "text": "The quick brown fox"},
                {"id": "doc2", "text": "jumps over the lazy dog"}
            ]
        });
        let req: HybridIndexRequest =
            serde_json::from_value(json).expect("failed to deserialize HybridIndexRequest");
        assert_eq!(req.collection, "test_col");
        assert_eq!(req.documents.len(), 2);
        assert_eq!(req.documents[0].id, "doc1");
        assert_eq!(req.documents[1].text, "jumps over the lazy dog");
    }

    #[test]
    fn test_fulltext_index_map_operations() {
        use crate::storage::engines::core::formats::columnar::fulltext_index::{
            FullTextIndex, TokenizerConfig,
        };

        let map: FullTextIndexMap =
            Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));

        // Add an index
        {
            let mut indexes = map.write().expect("RwLock should not be poisoned");
            let mut index = FullTextIndex::new(TokenizerConfig::for_keyword_search());
            index
                .add_document("doc1", "machine learning neural networks")
                .expect("failed to add document to index");
            index
                .add_document("doc2", "deep learning transformers")
                .expect("failed to add document to index");
            index
                .add_document("doc3", "database systems query optimization")
                .expect("failed to add document to index");
            indexes.insert("test_col".to_string(), index);
        }

        // Search
        {
            let indexes = map.read().expect("RwLock should not be poisoned");
            let index = indexes
                .get("test_col")
                .expect("test_col index should exist");
            let results = index.search("learning", 10);
            assert_eq!(results.len(), 2);
            // doc1 and doc2 both contain "learning"
            let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
            assert!(ids.contains(&"doc1"));
            assert!(ids.contains(&"doc2"));
        }
    }

    #[test]
    fn test_hybrid_search_response_serialization() {
        let response = LegacyHybridSearchResponse {
            success: true,
            results: vec![
                HybridSearchHit {
                    id: "doc1".to_string(),
                    combined_score: 0.05,
                    vector_score: Some(0.95),
                    bm25_score: Some(3.2),
                    vector_rank: Some(1),
                    bm25_rank: Some(2),
                    matched_terms: vec!["learning".to_string()],
                },
                HybridSearchHit {
                    id: "doc2".to_string(),
                    combined_score: 0.03,
                    vector_score: None,
                    bm25_score: Some(5.1),
                    vector_rank: None,
                    bm25_rank: Some(1),
                    matched_terms: vec!["machine".to_string(), "learning".to_string()],
                },
            ],
            total: 2,
            processing_time_us: 1234,
            mode: "hybrid".to_string(),
        };

        let json =
            serde_json::to_string(&response).expect("failed to serialize HybridSearchResponse");
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"mode\":\"hybrid\""));
        // doc2 should NOT have vector_score/vector_rank (skip_serializing_if = None)
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("failed to deserialize JSON value");
        let doc2 = &parsed["results"][1];
        assert!(doc2.get("vector_score").is_none());
        assert!(doc2.get("vector_rank").is_none());
    }

    // Test ApiError variants
    #[test]
    fn test_api_error_variants() {
        use std::io;

        // Test CollectionNotFound
        let err = ApiError::CollectionNotFound("test_col".to_string());
        assert_eq!(err.to_string(), "Collection not found: test_col");

        // Test InvalidArgument
        let err = ApiError::InvalidArgument("bad argument".to_string());
        assert_eq!(err.to_string(), "Invalid argument: bad argument");

        // Test Internal
        let err = ApiError::Internal("internal error".to_string());
        assert_eq!(err.to_string(), "Internal error: internal error");

        // Test IO error message propagation
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let api_err = ApiError::Internal(io_err.to_string());
        assert!(api_err.to_string().contains("file not found"));
    }

    // Test ApiDisplay trait implementation
    #[test]
    fn test_api_display() {
        let err = ApiError::CollectionNotFound("my_collection".to_string());
        let display = format!("{}", err);
        assert!(display.contains("my_collection"));
    }

    // Test error message formatting
    #[test]
    fn test_error_message_formatting() {
        let errors = vec![
            ApiError::CollectionNotFound("test".to_string()),
            ApiError::InvalidArgument("invalid".to_string()),
            ApiError::Internal("server error".to_string()),
        ];

        for err in errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
            assert!(!msg.contains("ApiError(")); // Should be user-friendly
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_hybrid_search_canonical_production_route_returns_bad_request_not_not_found() {
        let (state, _temp_dir) = build_test_app_state().await;
        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/v2/hybrid/search")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"collection":"","text_query":"hybrid route test"}"#,
            ))
            .expect("failed to build request");

        let response = router
            .oneshot(request)
            .await
            .expect("router failed handling canonical hybrid route request");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    // `test_vector_search_canonical_production_route_*` removed: the legacy
    // `/api/v1/search` route was deleted in the API standardization hard-rename.
    // Canonical vector search is `/api/v2/collections/:id/search`, served by
    // `create_v2_router` (mounted in `server.rs`, not in `create_router`), so it
    // is exercised by the v2 records/search tests rather than this base-router guard.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_document_index_canonical_production_route_returns_bad_request_not_not_found() {
        let (state, _temp_dir) = build_test_app_state().await;
        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/v2/document-collections/ws1_docs/indexes")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"path":"content","index_type":"fulltext","unique":false}"#,
            ))
            .expect("failed to build request");

        let response = router
            .oneshot(request)
            .await
            .expect("router failed handling canonical document route request");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_graph_shortest_path_canonical_production_route_returns_unprocessable_entity_not_not_found()
     {
        let (state, _temp_dir) = build_test_app_state().await;
        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/v2/graphs/ws1_graph/shortest-path")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .expect("failed to build request");

        let response = router
            .oneshot(request)
            .await
            .expect("router failed handling canonical graph route request");
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_graph_legacy_nodes_endpoint_redirects_to_canonical_multi_graph_route() {
        let (state, _temp_dir) = build_test_app_state().await;
        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/v2/nodes")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .expect("failed to build request");

        let response = router
            .oneshot(request)
            .await
            .expect("router failed handling legacy graph nodes route request");
        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        let location = response
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok());
        assert_eq!(location, Some("/api/v2/graphs/default/nodes"));
        let deprecation = response
            .headers()
            .get("deprecation")
            .and_then(|v| v.to_str().ok());
        assert_eq!(deprecation, Some("true"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_graph_legacy_edges_endpoint_redirects_to_canonical_multi_graph_route() {
        let (state, _temp_dir) = build_test_app_state().await;
        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/api/v2/edges")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .expect("failed to build request");

        let response = router
            .oneshot(request)
            .await
            .expect("router failed handling legacy graph edges route request");
        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        let location = response
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok());
        assert_eq!(location, Some("/api/v2/graphs/default/edges"));
        let deprecation = response
            .headers()
            .get("deprecation")
            .and_then(|v| v.to_str().ok());
        assert_eq!(deprecation, Some("true"));
    }

    // ============================================================
    // sql_value_to_json tests (coverage improvement)
    // ============================================================

    #[test]
    fn test_sql_value_to_json_string() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::StringValue(
                "hello".to_string(),
            )),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::json!("hello"));
    }

    #[test]
    fn test_sql_value_to_json_number() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::NumberValue(42.5)),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::json!(42.5));
    }

    #[test]
    fn test_sql_value_to_json_bool() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::BoolValue(true)),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::json!(true));
    }

    #[test]
    fn test_sql_value_to_json_int64() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::Int64Value(9999)),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::json!(9999));
    }

    #[test]
    fn test_sql_value_to_json_bytes() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::BytesValue(vec![0, 1, 255])),
        };
        let json = sql_value_to_json(&val);
        // Round 15: delegated to the shared converter — bytes render base64.
        assert_eq!(json, serde_json::json!("AAH/"));
    }

    #[test]
    fn test_sql_value_to_json_null() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::NullValue(0)),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::Value::Null);
    }

    #[test]
    fn test_sql_value_to_json_none() {
        let val = proximadb_v1::SqlValue { value: None };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::Value::Null);
    }

    #[test]
    fn test_sql_value_to_json_array() {
        let arr = proximadb_v1::SqlArray {
            values: vec![
                proximadb_v1::SqlValue {
                    value: Some(proximadb_v1::sql_value::Value::Int64Value(1)),
                },
                proximadb_v1::SqlValue {
                    value: Some(proximadb_v1::sql_value::Value::Int64Value(2)),
                },
            ],
        };
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::ArrayValue(arr)),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json, serde_json::json!([1, 2]));
    }

    #[test]
    fn test_sql_value_to_json_object() {
        let mut fields = std::collections::HashMap::new();
        fields.insert(
            "name".to_string(),
            proximadb_v1::SqlValue {
                value: Some(proximadb_v1::sql_value::Value::StringValue(
                    "Alice".to_string(),
                )),
            },
        );
        fields.insert(
            "age".to_string(),
            proximadb_v1::SqlValue {
                value: Some(proximadb_v1::sql_value::Value::Int64Value(30)),
            },
        );
        let obj = proximadb_v1::SqlObject { fields };
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::ObjectValue(obj)),
        };
        let json = sql_value_to_json(&val);
        assert_eq!(json["name"], serde_json::json!("Alice"));
        assert_eq!(json["age"], serde_json::json!(30));
    }

    #[test]
    fn test_sql_value_to_json_nan_number() {
        let val = proximadb_v1::SqlValue {
            value: Some(proximadb_v1::sql_value::Value::NumberValue(f64::NAN)),
        };
        let json = sql_value_to_json(&val);
        // Round 15: the shared converter renders non-finite floats as null.
        assert_eq!(json, serde_json::json!(null));
    }

    // ============================================================
    // SqlQueryRequest/SqlColumnInfo tests
    // ============================================================

    #[test]
    fn test_sql_query_request_deserialization() {
        let json = serde_json::json!({
            "query": "SELECT * FROM my_collection LIMIT 10",
            "collection": "my_collection",
            "timeout_ms": 5000
        });
        let req: SqlQueryRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.query, "SELECT * FROM my_collection LIMIT 10");
        assert_eq!(req.collection, Some("my_collection".to_string()));
        assert_eq!(req.timeout_ms, Some(5000));
        assert!(req.parameters.is_none());
        assert!(req.seeding.is_none());
    }

    #[test]
    fn test_sql_column_info_serialization() {
        let col = SqlColumnInfo {
            name: "embedding".to_string(),
            data_type: "vector".to_string(),
        };
        let json = serde_json::to_string(&col).unwrap();
        assert!(json.contains("embedding"));
        assert!(json.contains("vector"));
    }

    /// Verify search response (VectorOperationResponse) serializes correctly to JSON
    #[test]
    fn test_search_response_serialization() {
        let response = proximadb_v1::VectorOperationResponse {
            success: true,
            operation: 1, // search
            metrics: Some(proximadb_v1::OperationMetrics {
                total_processed: 100,
                successful_count: 95,
                failed_count: 5,
                updated_count: 0,
                processing_time_us: 1500,
                wal_write_time_us: 0,
                index_update_time_us: 0,
            }),
            results: Some(proximadb_v1::SearchResult {
                results: vec![
                    {
                        let mut r = proximadb_v1::SearchVectorRecord::default();
                        r.id = "result_1".to_string();
                        r.score = 0.95;
                        r.vector = vec![0.1, 0.2, 0.3];
                        r.version = Some(1);
                        r.similarity = Some(0.95);
                        r
                    },
                    {
                        let mut r = proximadb_v1::SearchVectorRecord::default();
                        r.id = "result_2".to_string();
                        r.score = 0.87;
                        r.vector = vec![0.4, 0.5, 0.6];
                        r.version = Some(1);
                        r.similarity = Some(0.87);
                        r
                    },
                ],
                total_found: 2,
                collection_id: Some("test_collection".to_string()),
            }),
            vector_ids: vec!["result_1".to_string(), "result_2".to_string()],
            error_message: None,
            error_code: None,
        };

        let json_str = serde_json::to_string(&response)
            .expect("VectorOperationResponse should serialize to JSON");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("serialized JSON should parse back");

        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["operation"], 1);
        assert_eq!(parsed["results"]["total_found"], 2);
        assert_eq!(parsed["results"]["results"][0]["id"], "result_1");
        assert_eq!(parsed["results"]["results"][1]["id"], "result_2");
        assert!(parsed["results"]["results"][0]["score"].as_f64().unwrap() > 0.9);
        assert_eq!(parsed["vector_ids"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["metrics"]["total_processed"], 100);
        assert_eq!(parsed["metrics"]["processing_time_us"], 1500);
    }

    // ============================================================
    // SQL operations tests (3 tests)
    // ============================================================

    /// Verify SqlQueryRequest deserializes correctly with all fields
    #[test]
    fn test_sql_query_request_parsing() {
        // Full request with all optional fields
        let json = serde_json::json!({
            "query": "SELECT id, metadata FROM vectors WHERE category = 'electronics' ORDER BY score LIMIT 20",
            "collection": "products",
            "timeout_ms": 10000,
            "seeding": "per_seed",
            "parameters": [
                {"value": {"StringValue": "electronics"}},
                {"value": {"Int64Value": 20}}
            ]
        });
        let req: SqlQueryRequest =
            serde_json::from_value(json).expect("full SqlQueryRequest should deserialize");
        assert_eq!(
            req.query,
            "SELECT id, metadata FROM vectors WHERE category = 'electronics' ORDER BY score LIMIT 20"
        );
        assert_eq!(req.collection, Some("products".to_string()));
        assert_eq!(req.timeout_ms, Some(10000));
        assert_eq!(req.seeding, Some("per_seed".to_string()));
        assert!(req.parameters.is_some());
        assert_eq!(req.parameters.as_ref().unwrap().len(), 2);

        // Minimal request with only required fields
        let minimal_json = serde_json::json!({
            "query": "SELECT * FROM test"
        });
        let minimal_req: SqlQueryRequest = serde_json::from_value(minimal_json)
            .expect("minimal SqlQueryRequest should deserialize");
        assert_eq!(minimal_req.query, "SELECT * FROM test");
        assert!(minimal_req.collection.is_none());
        assert!(minimal_req.timeout_ms.is_none());
        assert!(minimal_req.seeding.is_none());
        assert!(minimal_req.parameters.is_none());
    }

    /// Verify SQL response format serializes correctly
    #[test]
    fn test_sql_response_serialization() {
        // Simulate the JSON response structure returned by execute_sql handler
        let response_json = serde_json::json!({
            "rows": [
                {"id": "vec1", "score": 0.95, "category": "electronics"},
                {"id": "vec2", "score": 0.87, "category": "books"}
            ],
            "columns": ["id", "score", "category"],
            "column_types": ["string", "float", "string"],
            "execution_time_ms": 42,
            "rows_returned": 2,
            "row_count": 2,
            "rows_scanned": 100,
            "request_id": "req-123"
        });

        // Verify all fields are present and have correct types
        assert_eq!(response_json["rows"].as_array().unwrap().len(), 2);
        assert_eq!(response_json["rows"][0]["id"], "vec1");
        assert_eq!(response_json["rows"][1]["category"], "books");
        assert_eq!(response_json["columns"].as_array().unwrap().len(), 3);
        assert_eq!(response_json["execution_time_ms"], 42);
        assert_eq!(response_json["rows_returned"], 2);
        assert_eq!(response_json["row_count"], 2);
        assert_eq!(response_json["rows_scanned"], 100);
        assert_eq!(response_json["request_id"], "req-123");

        // Verify the response round-trips through serialization
        let serialized = serde_json::to_string(&response_json)
            .expect("SQL response JSON should serialize to string");
        let deserialized: serde_json::Value = serde_json::from_str(&serialized)
            .expect("serialized SQL response should deserialize back");
        assert_eq!(response_json, deserialized);
    }

    /// Empty query string produces an error (validated in handler, not in parsing)
    #[test]
    fn test_invalid_sql_request() {
        // Empty query string deserializes fine at the serde level
        let empty_query = serde_json::json!({
            "query": ""
        });
        let req: SqlQueryRequest =
            serde_json::from_value(empty_query).expect("empty query should still deserialize");
        assert_eq!(req.query, "");
        // Handler-level validation: query.trim().is_empty() returns true
        assert!(
            req.query.trim().is_empty(),
            "empty query should be detected by handler validation"
        );

        // Whitespace-only query
        let whitespace_query = serde_json::json!({
            "query": "   \t\n  "
        });
        let req: SqlQueryRequest = serde_json::from_value(whitespace_query)
            .expect("whitespace query should still deserialize");
        assert!(
            req.query.trim().is_empty(),
            "whitespace-only query should be detected by handler validation"
        );

        // Missing required 'query' field entirely should fail deserialization
        let missing_query = serde_json::json!({
            "collection": "test_col",
            "timeout_ms": 5000
        });
        let result = serde_json::from_value::<SqlQueryRequest>(missing_query);
        assert!(
            result.is_err(),
            "missing 'query' field should fail deserialization"
        );
    }

    // ============================================================
    // Collection operations tests (4 tests)
    // ============================================================

    /// Verify CollectionRequest (create) deserializes from JSON
    #[test]
    fn test_create_collection_request_parsing() {
        let json = serde_json::json!({
            "operation": 1, // CollectionCreate
            "collection_id": "new_collection",
            "collection_config": {
                "name": "new_collection",
                "dimension": 128,
                "distance_metric": 0,
                "tags": ["test", "development"],
                "description": "A test collection for unit testing"
            }
        });
        let req: CollectionRequest =
            serde_json::from_value(json).expect("CollectionRequest (create) should deserialize");
        assert_eq!(req.operation, CollectionOperation::CollectionCreate as i32);
        assert_eq!(req.collection_id, Some("new_collection".to_string()));
        assert!(req.collection_config.is_some());
        let config = req.collection_config.unwrap();
        assert_eq!(config.name, "new_collection");
        assert_eq!(config.dimension, 128);
        assert_eq!(config.tags.len(), 2);
        assert_eq!(config.tags[0], "test");
        assert_eq!(
            config.description,
            Some("A test collection for unit testing".to_string())
        );

        // Minimal create request (only operation and name)
        let minimal_json = serde_json::json!({
            "operation": 1,
            "collection_config": {
                "name": "minimal_col",
                "dimension": 64
            }
        });
        let minimal_req: CollectionRequest = serde_json::from_value(minimal_json)
            .expect("minimal CollectionRequest should deserialize");
        assert_eq!(minimal_req.operation, 1);
        assert!(minimal_req.collection_config.is_some());
        assert_eq!(minimal_req.collection_config.unwrap().dimension, 64);
    }

    /// Verify CollectionResponse serializes correctly to JSON
    #[test]
    fn test_collection_response_serialization() {
        let response = proximadb_v1::CollectionResponse {
            success: true,
            collection: Some(proximadb_v1::Collection {
                id: "col_123".to_string(),
                config: Some(proximadb_v1::CollectionConfig {
                    name: "test_collection".to_string(),
                    dimension: 128,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            collections: vec![],
            error_message: None,
            error_code: None,
            operation: CollectionOperation::CollectionCreate as i32,
            affected_count: 1,
            total_count: 1,
            metadata: HashMap::new(),
            processing_time_us: 250,
        };

        let json_str =
            serde_json::to_string(&response).expect("CollectionResponse should serialize to JSON");
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .expect("serialized CollectionResponse should parse back");

        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["collection"]["id"], "col_123");
        assert_eq!(parsed["collection"]["config"]["name"], "test_collection");
        assert_eq!(parsed["collection"]["config"]["dimension"], 128);
        assert_eq!(
            parsed["operation"],
            CollectionOperation::CollectionCreate as i32
        );
        assert_eq!(parsed["affected_count"], 1);
        assert_eq!(parsed["processing_time_us"], 250);

        // Error response
        let error_response = proximadb_v1::CollectionResponse {
            success: false,
            collection: None,
            collections: vec![],
            error_message: Some("Collection already exists".to_string()),
            error_code: Some("ALREADY_EXISTS".to_string()),
            operation: CollectionOperation::CollectionCreate as i32,
            affected_count: 0,
            total_count: 0,
            metadata: HashMap::new(),
            processing_time_us: 10,
        };

        let err_json_str = serde_json::to_string(&error_response)
            .expect("error CollectionResponse should serialize");
        let err_parsed: serde_json::Value = serde_json::from_str(&err_json_str)
            .expect("serialized error response should parse back");
        assert_eq!(err_parsed["success"], false);
        assert_eq!(err_parsed["error_message"], "Collection already exists");
        assert_eq!(err_parsed["error_code"], "ALREADY_EXISTS");
    }

    /// Verify list collections response with multiple collections
    #[test]
    fn test_list_collections_response() {
        let response = proximadb_v1::CollectionResponse {
            success: true,
            collection: None,
            collections: vec![
                proximadb_v1::Collection {
                    id: "col_1".to_string(),
                    config: Some(proximadb_v1::CollectionConfig {
                        name: "vectors_prod".to_string(),
                        dimension: 128,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                proximadb_v1::Collection {
                    id: "col_2".to_string(),
                    config: Some(proximadb_v1::CollectionConfig {
                        name: "vectors_staging".to_string(),
                        dimension: 256,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                proximadb_v1::Collection {
                    id: "col_3".to_string(),
                    config: Some(proximadb_v1::CollectionConfig {
                        name: "embeddings_test".to_string(),
                        dimension: 512,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            error_message: None,
            error_code: None,
            operation: CollectionOperation::CollectionList as i32,
            affected_count: 0,
            total_count: 3,
            metadata: HashMap::new(),
            processing_time_us: 500,
        };

        let json_str =
            serde_json::to_string(&response).expect("list collections response should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("serialized list response should parse back");

        assert_eq!(parsed["success"], true);
        assert_eq!(
            parsed["operation"],
            CollectionOperation::CollectionList as i32
        );
        assert_eq!(parsed["total_count"], 3);
        let collections = parsed["collections"]
            .as_array()
            .expect("collections should be an array");
        assert_eq!(collections.len(), 3);
        assert_eq!(collections[0]["id"], "col_1");
        assert_eq!(collections[0]["config"]["name"], "vectors_prod");
        assert_eq!(collections[1]["id"], "col_2");
        assert_eq!(collections[1]["config"]["name"], "vectors_staging");
        assert_eq!(collections[2]["id"], "col_3");
        assert_eq!(collections[2]["config"]["name"], "embeddings_test");

        // Empty list response
        let empty_response = proximadb_v1::CollectionResponse {
            success: true,
            collection: None,
            collections: vec![],
            error_message: None,
            error_code: None,
            operation: CollectionOperation::CollectionList as i32,
            affected_count: 0,
            total_count: 0,
            metadata: HashMap::new(),
            processing_time_us: 50,
        };

        let empty_json =
            serde_json::to_string(&empty_response).expect("empty list response should serialize");
        let empty_parsed: serde_json::Value =
            serde_json::from_str(&empty_json).expect("empty list should parse back");
        assert_eq!(empty_parsed["collections"].as_array().unwrap().len(), 0);
        assert_eq!(empty_parsed["total_count"], 0);
    }

    /// Verify delete collection request constructs correctly
    #[test]
    fn test_delete_collection_request() {
        // Verify the CollectionRequest for delete operation can be constructed and serialized
        let delete_request = CollectionRequest {
            operation: CollectionOperation::CollectionDelete as i32,
            collection_id: Some("col_to_delete".to_string()),
            collection_config: None,
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        };

        assert_eq!(
            delete_request.operation,
            CollectionOperation::CollectionDelete as i32
        );
        assert_eq!(
            delete_request.collection_id,
            Some("col_to_delete".to_string())
        );
        assert!(delete_request.collection_config.is_none());

        // Verify it serializes to JSON
        let json_str = serde_json::to_string(&delete_request)
            .expect("delete CollectionRequest should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("serialized delete request should parse back");
        assert_eq!(
            parsed["operation"],
            CollectionOperation::CollectionDelete as i32
        );
        assert_eq!(parsed["collection_id"], "col_to_delete");

        // Verify deserialization round-trip
        let deserialized: CollectionRequest =
            serde_json::from_str(&json_str).expect("delete request should round-trip through JSON");
        assert_eq!(deserialized.operation, delete_request.operation);
        assert_eq!(deserialized.collection_id, delete_request.collection_id);

        // Verify CollectionOperation enum conversion
        let op = CollectionOperation::try_from(delete_request.operation);
        assert!(op.is_ok());
        assert_eq!(op.unwrap(), CollectionOperation::CollectionDelete);

        // Verify invalid operation value is rejected
        let invalid_op = CollectionOperation::try_from(999);
        assert!(
            invalid_op.is_err(),
            "invalid operation value 999 should fail enum conversion"
        );
    }

    // ============================================================
    // Hybrid search tests (3 tests)
    // ============================================================

    /// Verify HybridSearchRequest deserializes with all field combinations
    #[test]
    fn test_hybrid_search_request_parsing() {
        // Full hybrid request (vector + text)
        let full_json = serde_json::json!({
            "collection": "hybrid_col",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "text_query": "machine learning algorithms",
            "top_k": 20,
            "vector_weight": 0.6,
            "rrf_k": 100,
            "min_bm25_score": 0.5
        });
        let full_req: LegacyHybridSearchRequest =
            serde_json::from_value(full_json).expect("full HybridSearchRequest should deserialize");
        assert_eq!(full_req.collection, "hybrid_col");
        assert_eq!(full_req.vector.as_ref().unwrap().len(), 4);
        assert!((full_req.vector.as_ref().unwrap()[0] - 0.1).abs() < 1e-6);
        assert_eq!(
            full_req.text_query,
            Some("machine learning algorithms".to_string())
        );
        assert_eq!(full_req.top_k, 20);
        assert!((full_req.vector_weight - 0.6).abs() < 0.001);
        assert_eq!(full_req.rrf_k, 100);
        assert!((full_req.min_bm25_score - 0.5).abs() < 0.001);

        // Vector-only request (no text_query)
        let vector_only_json = serde_json::json!({
            "collection": "vec_only",
            "vector": [1.0, 2.0, 3.0]
        });
        let vec_req: LegacyHybridSearchRequest = serde_json::from_value(vector_only_json)
            .expect("vector-only HybridSearchRequest should deserialize");
        assert_eq!(vec_req.collection, "vec_only");
        assert!(vec_req.vector.is_some());
        assert!(vec_req.text_query.is_none());
        assert_eq!(vec_req.top_k, 10); // default
        assert!((vec_req.vector_weight - 0.5).abs() < 0.001); // default
        assert_eq!(vec_req.rrf_k, 60); // default

        // Text-only request (no vector)
        let text_only_json = serde_json::json!({
            "collection": "text_only",
            "text_query": "database systems"
        });
        let text_req: LegacyHybridSearchRequest = serde_json::from_value(text_only_json)
            .expect("text-only HybridSearchRequest should deserialize");
        assert_eq!(text_req.collection, "text_only");
        assert!(text_req.vector.is_none());
        assert_eq!(text_req.text_query, Some("database systems".to_string()));
    }

    /// Verify HybridSearchResponse serializes correctly including skip_serializing_if behavior
    #[test]
    fn test_hybrid_search_response_serialization_extended() {
        // Response with mixed hit types: some have vector scores, some have bm25 only
        let response = LegacyHybridSearchResponse {
            success: true,
            results: vec![
                HybridSearchHit {
                    id: "doc_a".to_string(),
                    combined_score: 0.042,
                    vector_score: Some(0.92),
                    bm25_score: Some(4.5),
                    vector_rank: Some(1),
                    bm25_rank: Some(3),
                    matched_terms: vec!["neural".to_string(), "network".to_string()],
                },
                HybridSearchHit {
                    id: "doc_b".to_string(),
                    combined_score: 0.035,
                    vector_score: None, // BM25-only hit
                    bm25_score: Some(6.2),
                    vector_rank: None,
                    bm25_rank: Some(1),
                    matched_terms: vec!["deep".to_string(), "learning".to_string()],
                },
                HybridSearchHit {
                    id: "doc_c".to_string(),
                    combined_score: 0.028,
                    vector_score: Some(0.85), // Vector-only hit
                    bm25_score: None,
                    vector_rank: Some(2),
                    bm25_rank: None,
                    matched_terms: vec![],
                },
            ],
            total: 3,
            processing_time_us: 2500,
            mode: "hybrid".to_string(),
        };

        let json_str =
            serde_json::to_string(&response).expect("HybridSearchResponse should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("serialized response should parse back");

        // Verify top-level fields
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["total"], 3);
        assert_eq!(parsed["processing_time_us"], 2500);
        assert_eq!(parsed["mode"], "hybrid");

        // doc_a: has all scores
        let doc_a = &parsed["results"][0];
        assert_eq!(doc_a["id"], "doc_a");
        assert!(doc_a.get("vector_score").is_some());
        assert!(doc_a.get("bm25_score").is_some());
        assert!(doc_a.get("vector_rank").is_some());
        assert!(doc_a.get("bm25_rank").is_some());
        assert_eq!(doc_a["matched_terms"].as_array().unwrap().len(), 2);

        // doc_b: BM25-only (vector_score and vector_rank should be absent due to skip_serializing_if)
        let doc_b = &parsed["results"][1];
        assert_eq!(doc_b["id"], "doc_b");
        assert!(
            doc_b.get("vector_score").is_none(),
            "BM25-only hit should omit vector_score"
        );
        assert!(
            doc_b.get("vector_rank").is_none(),
            "BM25-only hit should omit vector_rank"
        );
        assert!(doc_b.get("bm25_score").is_some());
        assert!(doc_b.get("bm25_rank").is_some());

        // doc_c: Vector-only (bm25_score and bm25_rank should be absent)
        let doc_c = &parsed["results"][2];
        assert_eq!(doc_c["id"], "doc_c");
        assert!(
            doc_c.get("bm25_score").is_none(),
            "vector-only hit should omit bm25_score"
        );
        assert!(
            doc_c.get("bm25_rank").is_none(),
            "vector-only hit should omit bm25_rank"
        );
        assert!(doc_c.get("vector_score").is_some());
        assert!(doc_c.get("vector_rank").is_some());
        assert_eq!(doc_c["matched_terms"].as_array().unwrap().len(), 0);
    }

    /// Verify HybridIndexRequest deserializes and validates correctly
    #[test]
    fn test_hybrid_index_request_parsing() {
        // Standard index request with multiple documents
        let json = serde_json::json!({
            "collection": "index_col",
            "documents": [
                {"id": "doc1", "text": "Introduction to machine learning"},
                {"id": "doc2", "text": "Advanced neural network architectures"},
                {"id": "doc3", "text": "Database query optimization techniques"},
                {"id": "doc4", "text": "Distributed systems and consensus algorithms"}
            ]
        });
        let req: HybridIndexRequest =
            serde_json::from_value(json).expect("HybridIndexRequest should deserialize");
        assert_eq!(req.collection, "index_col");
        assert_eq!(req.documents.len(), 4);
        assert_eq!(req.documents[0].id, "doc1");
        assert_eq!(req.documents[0].text, "Introduction to machine learning");
        assert_eq!(req.documents[3].id, "doc4");
        assert!(req.documents[3].text.contains("consensus"));

        // HybridIndexResponse serialization
        let response = HybridIndexResponse {
            success: true,
            collection: "index_col".to_string(),
            documents_indexed: 4,
            total_documents: 10,
        };
        let resp_json =
            serde_json::to_string(&response).expect("HybridIndexResponse should serialize");
        let resp_parsed: serde_json::Value =
            serde_json::from_str(&resp_json).expect("serialized index response should parse back");
        assert_eq!(resp_parsed["success"], true);
        assert_eq!(resp_parsed["collection"], "index_col");
        assert_eq!(resp_parsed["documents_indexed"], 4);
        assert_eq!(resp_parsed["total_documents"], 10);

        // Empty documents list should deserialize (validation happens in handler)
        let empty_docs = serde_json::json!({
            "collection": "empty_col",
            "documents": []
        });
        let empty_req: HybridIndexRequest = serde_json::from_value(empty_docs)
            .expect("empty documents HybridIndexRequest should deserialize");
        assert_eq!(empty_req.documents.len(), 0);
    }

    /// Spec↔router drift smoke gate.
    ///
    /// Every (method, path) in the published OpenAPI spec
    /// (`docs/openapi/proximadb-openapi.yaml`) MUST resolve to a registered
    /// route on the production public router. This catches the failure mode the
    /// SDK-from-spec contract exists to prevent: the generated client targets a
    /// path the server never serves → silent 404. The assertion keys on the
    /// canonical `not_found_fallback` body (`"No route for …"`), which cleanly
    /// separates a genuine routing-404 from a handler-level 404 (e.g. a missing
    /// collection) — only the former is drift.
    ///
    /// The case list mirrors the spec; `verify-openapi-spec` keeps the
    /// handler↔utoipa annotations aligned, this test keeps the
    /// router↔runtime aligned. (The 2026-07-19 qa-image 404 was a
    /// collection-resolution/tenant matter, NOT a route gap — but a real gap
    /// would fail here.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn all_openapi_spec_paths_resolve_to_a_route() {
        use axum::body::to_bytes;

        let (state, _temp_dir) = build_test_app_state().await;

        // Mirror the production public-router assembly (see
        // `build_router_for_unified` in server.rs): `create_router` (direct
        // `/api/v2` routes + feature sub-nests: graph / document / hybrid /
        // observability / analytics / …) with the canonical v2 router nested
        // at `/api/v2` (collections / records / search / query / …), the
        // production 404 fallback, and the default-tenant extension layers so
        // the v2 handlers' `Extension<TenantContext>` resolves.
        let default_tenant = crate::network::middleware::tenant::TenantContext::new(
            "default",
            crate::network::middleware::tenant::TenantIdSource::Default,
        );
        let default_api_tenant = proximadb_api::rest::TenantContext {
            tenant_id: "default".to_string(),
        };
        let v2_router = crate::network::rest::v2::create_v2_router().with_state(state.clone());
        let router = create_router(state)
            .nest("/api/v2", v2_router)
            .fallback(crate::network::rest::server::not_found_fallback)
            .layer(axum::Extension(default_tenant))
            .layer(axum::Extension(default_api_tenant));

        // (method, path, is_write). Path params are substituted with
        // placeholders — the route still matches by shape. Write methods carry
        // a deliberately truncated body so the `Json` extractor rejects before
        // any business logic runs: this test only needs the route to RESOLVE,
        // not the request to succeed. Mirrors docs/openapi/proximadb-openapi.yaml.
        const SPEC_PATHS: &[(axum::http::Method, &str, bool)] = &[
            (axum::http::Method::GET, "/api/v2/_meta/capabilities", false),
            (axum::http::Method::GET, "/api/v2/collections", false),
            (axum::http::Method::POST, "/api/v2/collections", true),
            (
                axum::http::Method::DELETE,
                "/api/v2/collections/tcol",
                false,
            ),
            (axum::http::Method::GET, "/api/v2/collections/tcol", false),
            (
                axum::http::Method::POST,
                "/api/v2/collections/tcol/documents",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/collections/tcol/entities",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/collections/tcol/entities/search",
                true,
            ),
            (
                axum::http::Method::DELETE,
                "/api/v2/collections/tcol/entities/tent",
                false,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/collections/tcol/entities/tent",
                false,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/collections/tcol/records/batch",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/collections/tcol/records/scan",
                true,
            ),
            (
                axum::http::Method::DELETE,
                "/api/v2/collections/tcol/records/trec",
                false,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/collections/tcol/records/trec",
                false,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/collections/tcol/schema",
                false,
            ),
            (
                axum::http::Method::PUT,
                "/api/v2/collections/tcol/schema",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/collections/tcol/search",
                true,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/document-collections",
                false,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/document-collections",
                true,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/document-collections/tcol/documents",
                false,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/document-collections/tcol/documents",
                true,
            ),
            (axum::http::Method::GET, "/api/v2/graphs", false),
            (axum::http::Method::POST, "/api/v2/graphs", true),
            (axum::http::Method::DELETE, "/api/v2/graphs/tgraph", false),
            (axum::http::Method::GET, "/api/v2/graphs/tgraph", false),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/edges",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/edges/batch",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/fusion-search",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/impact-analysis",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/nodes",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/nodes/batch",
                true,
            ),
            (
                axum::http::Method::DELETE,
                "/api/v2/graphs/tgraph/nodes/tnode",
                false,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/graphs/tgraph/nodes/tnode",
                false,
            ),
            (
                axum::http::Method::GET,
                "/api/v2/graphs/tgraph/stats",
                false,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/graphs/tgraph/traverse",
                true,
            ),
            (axum::http::Method::POST, "/api/v2/hybrid/index", true),
            (axum::http::Method::POST, "/api/v2/hybrid/search", true),
            (
                axum::http::Method::POST,
                "/api/v2/observability/namespaces/tns/logs",
                true,
            ),
            (
                axum::http::Method::POST,
                "/api/v2/observability/namespaces/tns/logs/search",
                true,
            ),
            (axum::http::Method::POST, "/api/v2/query", true),
            (axum::http::Method::POST, "/api/v2/query/explain", true),
            (axum::http::Method::GET, "/health", false),
            (axum::http::Method::GET, "/health/live", false),
            (axum::http::Method::GET, "/health/ready", false),
        ];

        for (method, path, is_write) in SPEC_PATHS {
            let request = Request::builder()
                .method(method.clone())
                .uri(*path)
                .header("content-type", "application/json")
                .body(if *is_write {
                    Body::from("{")
                } else {
                    Body::empty()
                })
                .expect("failed to build spec-path request");
            let response = router
                .clone()
                .oneshot(request)
                .await
                .expect("router failed handling spec-path request");
            let status = response.status();
            let body = to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("failed to read spec-path response body");
            let body_str = std::str::from_utf8(&body).unwrap_or("");
            assert!(
                !body_str.contains("No route for"),
                "spec {} {} did NOT resolve to a registered route (status {}): \
                 the SDK (generated from the spec) would 404. This is a \
                 spec↔router drift — add the route or drop the spec path.\n\
                 body: {}",
                method,
                path,
                status,
                body_str,
            );
        }
    }
}
