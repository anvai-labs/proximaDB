// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Shared service composition layer for ProximaDB protocol handlers.
//!
//! `SharedServices` owns and wires together all business-logic services
//! (storage, graph, document, observability, query) that are shared across
//! REST, gRPC, Arrow Flight, and PostgreSQL wire protocol handlers.
//! It is the composition root for the server-side service graph.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bridge implementing [`proximadb_storage_ports::CorpusVersionPort`] by delegating
/// to the catalog's `CorpusVersionRegistry`. Registered at bootstrap so the
/// compaction manager can bump corpus versions without depending on catalog.
/// (TD-DECOMP-66 edge 2: resolves storage-operations → catalog upward coupling.)
struct CorpusVersionBridge {
    registry: proximadb_catalog::CorpusVersionRegistry,
}

#[async_trait::async_trait]
impl proximadb_storage_ports::CorpusVersionPort for CorpusVersionBridge {
    async fn bump_collection_all_tenants(&self, collection: &str) -> usize {
        self.registry.bump_collection_all_tenants(collection).await
    }
}
use tracing::{debug, info, warn};

use crate::metrics::MetricsConfig;
use crate::monitoring::MetricsCollector;
use crate::observability::query::ObservabilityQueryEngine;
use crate::observability::storage::ObservabilityStorage;
use crate::query::facade::strategies::{DistributedQueryStrategy, DistributedStrategyConfig};
use crate::query::facade::{
    ColumnarStrategy, DocumentStrategy, FacadeConfig, GraphStrategy, ObservabilityStrategy,
    QueryFacadeAdapter, SqlStrategy, UnifiedQueryFacade, VectorSearchStrategy,
};
use crate::query::federated::FederatedQueryContext;
use crate::services::collection::manager::CollectionService;
use crate::services::{DmlService, VectorOperationsService};
use crate::storage::MultiModelStorageFacade;
use crate::storage::StorageEngine;
use crate::storage::document::DocumentService;
use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb_graph_query::service::{GraphExecutionService, GraphQueryService};
use proximadb_kernel::uuid::Uuid;

// ---- TD-MANIFEST-1: lease manifest retention knobs (parsed once per boot) ----
// Manifests are append-only and were never pruned, so a long-lived lease grew ~48k
// objects (562 MB) and made every `latest_version()` an O(n) full `list`. A
// low-priority prune of the stale tail caps n — itself the read-path fix. Default ON
// with conservative knobs; set PROXIMADB_LEASE_MANIFEST_RETENTION=0 to disable.

/// Whether the lease manifest prune loop runs (default ON).
fn lease_manifest_retention_enabled() -> bool {
    std::env::var("PROXIMADB_LEASE_MANIFEST_RETENTION")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(true)
}

/// Number of newest manifests to keep per lease log (the committer clamps this to ≥2).
fn lease_manifest_keep_k() -> usize {
    std::env::var("PROXIMADB_LEASE_MANIFEST_KEEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32)
}

/// Minimum age (seconds) a superseded manifest must reach before it is eligible to
/// prune — a grace window that protects the recent burst and release tombstones.
fn lease_manifest_min_age_secs() -> u64 {
    std::env::var("PROXIMADB_LEASE_MANIFEST_MIN_AGE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(900)
}

/// How often the prune loop sweeps each held lease log (seconds).
fn lease_manifest_prune_interval_secs() -> u64 {
    std::env::var("PROXIMADB_LEASE_MANIFEST_PRUNE_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300)
}

/// Join a subsystem subpath onto the configured durable storage base
/// (TD-OBJSTORE-1, #960).
///
/// `base` is `storage.metadata_url`: a bare local path (`/data`), a `file://`
/// URL, or an object-store URL (`s3://…`, `adls://…`, `abfs://…`, `gcs://…`).
/// The scheme is preserved verbatim; a bare path stays bare (every downstream
/// consumer normalizes with a `contains("://")` guard). Never strip a scheme
/// and re-prepend one around this join — stripping `file://` off metadata_url
/// and then formatting `file://{…}` back on is exactly what produced the
/// invalid `file://adls://…` URLs on object-store deployments.
pub(crate) fn join_storage_url(base: &str, sub: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    if sub.is_empty() {
        trimmed.to_string()
    } else {
        format!("{}/{}", trimmed, sub.trim_start_matches('/'))
    }
}

/// The local-filesystem view of a storage base, for subsystems that cannot yet
/// run over an object store (TD-OBJSTORE-1 deferred set — e.g. the TST
/// time-series engine). Returns `None` when the base carries a non-`file`
/// scheme, so callers can fail over loudly instead of handing an
/// `adls://…`-shaped string to `std::fs`.
fn local_storage_path(base: &str) -> Option<std::path::PathBuf> {
    if let Some(path) = base.strip_prefix("file://") {
        Some(std::path::PathBuf::from(path))
    } else if base.contains("://") {
        None
    } else {
        Some(std::path::PathBuf::from(base))
    }
}

/// Which consumer is constructing the shared service core.
///
/// The core (catalog, collection, vector/doc/graph compute, storage/WAL,
/// governance) is identical for every consumer. `ServiceProfile` only gates the
/// *network-dimension* machinery that has no consumer in-process: a fused
/// embedded library (`Embedded`) deletes Dimension 2 (network), so it must NOT
/// pay for the periodic Prometheus chargeback emitter or the metrics-persistence
/// / billing publisher that only feed a scrape/network surface. The networked
/// server (`Server`) keeps them. Co-design tenet 1 ("don't pay for a dimension
/// you deleted") + tenet 5 ("egress/KOU is inert in embedded").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceProfile {
    /// In-process / fused (PyO3, FFI, tests). No network-only background work.
    Embedded,
    /// Networked server. Constructs the observability/billing surfaces.
    Server,
}

impl ServiceProfile {
    /// True when constructing for the networked server.
    pub fn is_server(self) -> bool {
        matches!(self, ServiceProfile::Server)
    }
}

/// Shared services for thin protocol handlers
/// Responsibilities: business logic, metadata configuration, service coordination
#[derive(Clone)]
pub struct SharedServices {
    /// Filesystem factory for reading blocks
    pub filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    /// Shared xCatalog control plane for REST, gRPC, Arrow Flight, SQL, and query routing.
    pub catalog_manager: Arc<crate::catalog::CatalogManager>,
    /// Tenant-scoped embedding-model lifecycle authority. REST, gRPC, and
    /// compatibility adapters receive this exact `Arc`, including its atomic
    /// create coordination, instead of constructing transport-local registries.
    pub model_registry_service:
        Arc<proximadb_catalog::model_registry_service::CatalogModelRegistryService>,
    /// PAX segment registry — bridges write path with Iceberg REST snapshot stats.
    /// Shared with `AppState::segment_registry` via `Arc` clone in `build_router_for_unified`.
    pub segment_registry: Arc<crate::catalog::SegmentRegistry>,
    /// Collection lifecycle management service
    pub collection_service: Arc<CollectionService>,
    /// Vector CRUD and search operations service
    pub vector_operations_service: Arc<VectorOperationsService>,
    /// Concrete graph database operations service for native graph APIs and gRPC graph endpoints
    pub graph_service: Arc<crate::graph::GraphService>,
    /// Graph collection CRUD service (create/list/get/delete graph). The same
    /// shared `Arc<GraphCollectionService>` instance `GraphOperationsService`
    /// holds internally (TD-104 S3-f: hoisted out of the deleted root handler).
    pub graph_collection_service: Arc<crate::services::GraphCollectionService>,
    /// Extracted graph query/traversal capability for query-facing orchestration layers
    pub graph_query_service: Arc<dyn GraphQueryService>,
    /// Extracted graph execution capability for planners/executors and API state holders
    pub graph_execution_service: Arc<dyn GraphExecutionService>,
    /// Document storage and retrieval service
    pub document_service: Arc<DocumentService>,
    /// Observability service for logs, metrics, and traces
    pub observability_service: Arc<crate::observability::ObservabilityService>,
    /// Canonical record-batch write orchestration (TD-104 S3-f: built once
    /// here — formerly inside the deleted root `UnifiedHandlers::new` — so
    /// every former `request_handlers.record_ops()` site shares one `Arc`).
    pub record_ops: Arc<crate::api_handlers::record_ops_service::RecordOpsService>,
    /// ADR-022 auditable event log (append-only trail). The same
    /// `Arc<EventLogEngine>` handed to the unified handlers; exposed here so
    /// in-process surfaces (e.g. the reference MCP `event` tool) can append
    /// directly. `None` when the engine failed to initialize.
    pub event_log: Option<Arc<crate::storage::engines::eventlog::EventLogEngine>>,
    /// Optional metrics collector for Prometheus/monitoring integration
    pub metrics_collector: Option<Arc<MetricsCollector>>,
    /// Optional internal metrics updater for background metric publishing
    pub metrics_updater: Option<Arc<dyn crate::metrics::InternalMetricsUpdater + 'static>>,
    /// Port-backed API handler for collection/vector routes (runtime crate handler).
    ///
    /// Backed by `CollectionPort`, `VectorOpsPort`, and `QueryAdapterPort` trait objects
    /// so the REST/gRPC API surface is decoupled from root-crate concrete services.
    pub api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
    /// Unified query facade - single entry point for all query types
    /// Consolidates vector search, SQL, and graph query paths
    pub query_facade: Arc<UnifiedQueryFacade>,
    /// Shared SQL/query adapter over `query_facade`, carrying the `DmlService`
    /// so every consumer (runtime handler, pgwire, embedded) gets EXPLAIN
    /// `<DML>` routing on the port path. Built once in `new`; handed out by
    /// `query_adapter()` (TD-104 / seam S1: single SQL authority).
    pub query_adapter: Arc<QueryFacadeAdapter>,
    /// Optional cluster orchestration port (Phase 9.12 / Task #72).
    ///
    /// Production bootstrap currently passes `None` for single-node
    /// deployments. When `[distributed]` config is populated, a
    /// `ClusterManager` (which `impl proximadb_runtime::ClusterPort for`
    /// — see `src/cluster/mod.rs:305`) should be constructed and stored
    /// here. No consumer reads this field yet; the slot exists so that
    /// future health-endpoint / cluster-state-reporting code can pull
    /// cluster state via the port without re-plumbing.
    /// See `docs/_internal/status/PHASE9_REMAINING_2026_05_25.adoc`
    /// for the full Task #72 wiring plan.
    pub cluster_port: Option<Arc<dyn proximadb_runtime::ClusterPort>>,
    /// Port-typed view of `collection_service` for Phase 9.10 / Task #76
    /// consumer migration.
    ///
    /// Same underlying `CollectionService` instance as `collection_service`,
    /// just held behind the `CollectionPort` trait object so consumers can
    /// migrate off the concrete type incrementally. Once all consumers use
    /// `collection_port`, the concrete `collection_service` field can be
    /// dropped — that landing is what completes the Task #76 collection-service
    /// slice. Same parallel-field pattern as the existing `api_handlers`
    /// (which shadows `request_handlers`).
    pub collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
    /// Port-typed view of `vector_operations_service` for Phase 9.10 / Task #76
    /// consumer migration. Same pattern as `collection_port` above.
    pub vector_ops_port: Arc<dyn proximadb_runtime::VectorOpsPort>,
    /// Port-typed view of `document_service` for Phase 9.10 / Task #76
    /// consumer migration. Same pattern as `collection_port` above.
    ///
    /// Powered by `impl DocumentPort for DocumentService` directly on the
    /// bare service (ADR-015). The gRPC `DocumentServiceImpl` wrapper is
    /// no longer in the port chain.
    pub document_port: Arc<dyn proximadb_runtime::DocumentPort>,
    /// Port-typed view of the observability subsystem (Phase 9.10 / Task #76).
    ///
    /// **Suboptimal**: the port impl currently lives on the gRPC wrapper
    /// `ObservabilityServiceImpl`, not on the bare `ObservabilityService`
    /// (where ADR-015 says it should live). To unblock consumer migration
    /// now, this field constructs the wrapper and stores it as the port.
    /// The ADR-015 cleanup (move `impl ObservabilityPort` to the bare
    /// service, then update this field to coerce from `Arc<ObservabilityService>`
    /// directly) is a follow-up session — ~225 lines of port-impl bodies
    /// to lift from the 15 tonic methods.
    pub observability_port: Arc<dyn proximadb_runtime::ObservabilityPort>,
    /// Port-typed view of the graph subsystem (Phase 9.10 / Task #76).
    ///
    /// **Suboptimal**: same wrapper-as-port-host pattern as
    /// `observability_port`. ADR-015 cleanup is a follow-up session.
    pub graph_port: Arc<dyn proximadb_runtime::GraphPort>,

    /// Shared in-process full-text index map for hybrid retrieval
    /// (BM25 side of `/api/v1/hybrid/search`). REST and gRPC entry
    /// points read from this single map so an indexed document is
    /// immediately searchable on both protocols.
    ///
    /// Added 2026-05-26 (T3.2 Slice 1 of pre-release plan). Prior
    /// behavior had REST construct its own map locally in `AppState`
    /// and gRPC `HybridSearchServiceImpl` return mocks; this field
    /// gives both paths a shared backing.
    pub fulltext_indexes: crate::network::hybrid_search::HybridFullTextIndexMap,

    /// Process-wide cache of per-collection Vector Object Economy
    /// directories. Search paths fetch a [`CachedDirectoryHandle`] via
    /// `directory_cache.handle_for(collection_id)` and then `get_or_load`
    /// against a loader closure (typically wrapping
    /// [`load_directory_for`](crate::storage::engines::sst::object_economy_directory::load_directory_for)).
    /// First reader per collection pays the cost of loading the sidecar;
    /// subsequent readers reuse the cached `Arc<CachedDirectoryEntry>`.
    ///
    /// Writer/compactor will call
    /// [`VectorObjectEconomyDirectoryCache::invalidate`](crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache::invalidate)
    /// after `upsert_and_persist` lands a new directory version so the
    /// next reader picks up the change. That wiring is the next step
    /// after this slot exists — see the VECTOR_OBJECT_ECONOMY_ROUTE
    /// design doc.
    pub directory_cache: Arc<
        crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache,
    >,

    /// Process-wide per-collection pinning registry (Phase 6 control
    /// surface). Operators PATCH `/api/v1/collections/:id/pin` to
    /// record an explicit tier override; the `AxisTieringManager`
    /// consumer reads this registry during its evaluation loop and
    /// overrides its access-pattern policy when an operator pin is
    /// present. See `src/storage/collection_pinning.rs` for the
    /// control-plane / data-plane separation.
    pub pin_registry: Arc<crate::storage::collection_pinning::CollectionPinRegistry>,

    /// Process-wide cache-affinity registry (Phase 7.2). Tracks
    /// per-collection "which node most recently served queries" so
    /// that reads can be biased to whichever node owns the warm
    /// cache. Mirrors turbopuffer's published cache-affinity model
    /// ("subsequent queries route to the same query node for cache
    /// locality").
    ///
    /// The registry is process-wide and useful even in single-node
    /// deploys — it gives the operator API a place to inspect "which
    /// collections this node has been serving" and gives a future
    /// multi-node `RoutingService` an attach point via
    /// `with_affinity_registry`. The recording call lives in the
    /// data-plane search path so the registry reflects actual
    /// activity, not just routing decisions.
    pub affinity_registry: Arc<crate::cluster::cache_affinity::CacheAffinityRegistry>,

    /// Process-wide primary-pod registry (Slice 2 of
    /// `docs/12-design/TENANT_COLLECTION_POD_AFFINITY_2026_05_27.adoc`).
    /// Records the durable (tenant_id, collection_id) → primary pod
    /// binding that the gateway's write router consults on every
    /// write. Complementary to `affinity_registry`:
    ///
    /// | | `affinity_registry` (read) | `primary_pod_registry` (write) |
    /// |---|---|---|
    /// | Authority | in-memory hint, TTL-decayed | durable JSON sidecar today; xCatalog later |
    /// | Stickiness | soft preference | hard binding (writes MUST route here) |
    /// | Granularity | collection | tenant + collection |
    /// | Trigger | observed read traffic | explicit control-plane decision |
    ///
    /// Wired here in Slice 2; subsequent slices add the REST operator
    /// API (Slice 3), the gateway write-router consultation (Slice 4),
    /// and the xCatalog backing (Slice 5).
    pub primary_pod_registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,

    /// This pod's identity for primary-pod write-routing decisions
    /// (Slice 6.1). Resolved once at boot via
    /// [`crate::cluster::primary_pod_registry::resolve_self_pod_id`]
    /// so REST AppState and the gRPC v2 service both see the same
    /// value — if they diverged, the gate would flag legitimate
    /// writes as misrouted whenever one resolver path picked a
    /// different fallback.
    pub self_pod_id: String,

    /// A6 storage-write fence adapter over the durable `PartitionLeaseManager`
    /// (the SAME instance wired into RecordOpsService for lease-on-write). Handed
    /// to the storage engine's shutdown flush path by the bootstrap so a fenced-out
    /// pod cannot publish stale data. `None` when the lease store is unavailable
    /// (fail-open). Default-OFF until `PROXIMADB_WRITE_FENCING=1`.
    pub storage_write_fence: Option<Arc<dyn crate::storage::write_fence::StorageWriteFence>>,

    /// Partition lease manager for per-collection write authority (Phase 7c).
    ///
    /// This is the same manager used by RecordOps/DML and the storage-write
    /// fence, including its process-lifetime renewal loop. Protocol handlers
    /// consult it before DDL writes so no per-surface manager or lease timeline
    /// can diverge. `None` only when the configured lease store cannot open.
    /// `SystemCatalog` separately fences catalog snapshot publication; it does
    /// not fence MATERIALIZE/index side effects.
    pub partition_lease_manager:
        Option<Arc<crate::cluster::partition_lease::PartitionLeaseManager>>,

    /// Shared canonical WAL appender at `<data_dir>/pgwire/canonical-records.wal`.
    ///
    /// Opened once in `SharedServices::new` (when `opt_config` is provided so
    /// `cfg.server.data_dir` is known) and held as a single instance so both
    /// graph checkpoint emission (`GraphOperationsService::flush_wal`, TD-066)
    /// and pgwire direct record writes (`multi_server.rs` pgwire setup) share
    /// the same `next_sequence` counter and append lock. Without this
    /// sharing, two `FramedTableWalAppender::open` calls on the same file
    /// would each maintain independent next-sequence state, risking
    /// duplicate sequence numbers and silent recovery corruption.
    ///
    /// When `opt_config` is `None` (some test paths), this is `None` and
    /// both consumers fall back to their respective opt-out behavior
    /// (graph: tracing-only; pgwire: opens its own appender locally).
    pub canonical_wal_appender: Option<Arc<crate::services::FramedTableWalAppender>>,

    /// Experimental transactional ledger store (ADR-071 / TD-LEDGER-1) — the durable, tenant-scoped
    /// ledger port shared with the gRPC `ProximaLedgerService`. Node-level (one store per node;
    /// tenants are namespaced *inside* the keys, not by separate stores), with its WAL on local disk
    /// (ADR-069). Present only under the `experimental-ledger` feature.
    #[cfg(feature = "experimental-ledger")]
    pub ledger_store: Arc<proximadb_ledger::LedgerService<proximadb_ledger::DurableLedger>>,

    /// Durable ABAC attribute-authority handle (TD-ABAC control-plane /
    /// `abac-policy`). The SAME `Arc<FileSystemAttributeAuthority>` instance the
    /// live enforcer reads — so an admin provision written through this handle is
    /// visible without a restart (hot-reload). `None` unless `abac-policy` is on
    /// and `<data_dir>/abac/attribute-bindings.json` opens cleanly.
    #[cfg(feature = "abac-policy")]
    pub abac_authority: Option<Arc<proximadb_abac::FileSystemAttributeAuthority>>,

    /// Durable ABAC policy-binding store handle (TD-ABAC control-plane /
    /// `abac-policy`). Shared with the live enforcer; the admin policy-binding
    /// endpoints (TD-ABAC control-plane PR-B) write through this handle. `None`
    /// when ABAC is off.
    #[cfg(feature = "abac-policy")]
    pub abac_binding_store: Option<Arc<proximadb_abac::FileSystemPolicyBindingStore>>,

    /// Durable ADR-090 grant store handle (L1.1/L1.2). Shared with the live
    /// enforcer; the admin grant endpoints write through this handle so a
    /// grant/revoke is hot-visible without restart. `None` when ABAC is off.
    #[cfg(feature = "abac-policy")]
    pub abac_grant_store: Option<Arc<proximadb_catalog::grants::FileSystemGrantStore>>,

    /// TD-SEC-2 Slice C: per-tenant security posture store handle.
    #[cfg(feature = "abac-policy")]
    pub abac_posture_store:
        Option<Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>>,

    /// Durable ABAC predicate-object store handle (TD-ABAC control-plane /
    /// `abac-policy`). Shared with the live enforcer; the admin predicate-object
    /// endpoints register/revoke through this handle. `None` when ABAC is off.
    #[cfg(feature = "abac-policy")]
    pub abac_predicate_store: Option<Arc<proximadb_abac::FileSystemPredicateObjectStore>>,

    /// Process-shared ABAC enforcer wired into every data-plane surface,
    /// including pgwire's per-connection DML façade. Sharing this handle keeps
    /// policy epoch/cache state and the three live durable stores convergent.
    #[cfg(feature = "abac-policy")]
    pub abac_enforcer: Option<Arc<crate::security::rls::AbacEnforcer>>,

    /// The single canonical WAL-backed record store, built + WAL-recovered ONCE here and
    /// shared across ALL surfaces — the REST/gRPC `DmlService` and the pgwire direct-write
    /// path both route relational tables through this instance — so a write on any protocol
    /// is visible to reads + the CDC change-feed on every protocol. Previously the REST/gRPC
    /// `DmlService` used the vector-compatibility stub for relational tables while pgwire used
    /// the real WAL store, leaving the cross-surface APIs unconverged. `None` on test paths
    /// without `opt_config` (each consumer falls back to its own store).
    pub canonical_record_store:
        Option<Arc<crate::services::record_store::DirectWalTableRecordStore>>,

    /// F5 / TD-OLTP-WIRING-1: the process-shared fenced `ConditionalKeyStore`
    /// (ADR-072). `Some` only under the `oltp-integrity` feature with a durable
    /// `data_dir`. Held here as the single source of truth so pgwire
    /// (`multi_server.rs`) threads the SAME instance — one uniqueness index across
    /// every write surface (two stores would corrupt one WAL or fence
    /// inconsistently across gRPC vs pgwire).
    pub conditional_key_store: Option<Arc<dyn proximadb_storage_ports::ConditionalKeyStore>>,

    /// Process-wide recall-probe gate (TD-064 / LLD §5). The gate enables
    /// the quantized candidate route only after the recall-probe set passes
    /// the tenant's target for three consecutive builds; a single failure
    /// resets the streak. Held here so REST/gRPC handlers can read
    /// per-collection state via `is_open(ProbeScope)` without re-constructing
    /// the state machine — and so the planned Phase 5 stats refresher has a
    /// single registry to persist from. The gate was test-only prior to
    /// being slotted here; this is the first production wiring.
    pub recall_probe_gate: Arc<crate::catalog::RecallProbeGate>,

    /// Process-wide rank-pipeline singleton (R-7c.3 production wiring).
    ///
    /// REST, gRPC, and Arrow Flight all pull the same `Arc<RankServices>`
    /// from here via `AppState::with_rank_services` / equivalent, so SQL
    /// `RERANK(...)`, the REST `/api/v2/rank/search` route, and the
    /// `rank_features_export` Arrow Flight action share the same profile
    /// registry, candidate provider, scorer registry, and metric handles.
    /// Built around `ProductionHybridBackend` so retrieval lights up
    /// automatically as soon as ingestion populates per-collection BM25 +
    /// vector state.
    pub rank_services: Arc<crate::network::rest::canonical::rank::RankServices>,

    /// Durable rank-profile catalog backed by the canonical WAL spine.
    ///
    /// `RankServices` recovers profiles from this store at boot and
    /// `RankProfileStore::install` is the lowering target for `CREATE RANK
    /// PROFILE` DDL + the REST install endpoint. When `canonical_wal_appender`
    /// is `None` (some test paths), this store is backed by an in-memory
    /// appender that does not survive restart.
    pub rank_profile_store: Arc<dyn crate::services::RankProfileStore>,

    /// Durable SQL user-function catalog backed by the canonical WAL spine
    /// (UDF F5). `CREATE FUNCTION` DDL persists definitions here through the
    /// per-connection `DdlService`, and at boot every persisted entry is
    /// replayed + re-registered into the engine-neutral
    /// `proximadb_functions::builtins()` registry so user functions are live
    /// on both engines again after a restart. When `canonical_wal_appender`
    /// is `None` (some test paths) this store is backed by an in-memory
    /// appender that does not survive restart.
    pub function_store: Arc<dyn crate::services::FunctionStore>,

    /// Phase 8 (F1) snapshot-publish coordinator: pins canonical WAL snapshots
    /// and atomically republishes refined snapshots via the per-collection
    /// `discovery_active` projection freshness state machine. Shared by the
    /// discovery executor (data plane) and route-health disclosure (control
    /// plane). See `docs/12-design/PHASE8_CONTINUOUS_LOOP_HLD_LLD_2026_05_28.adoc`.
    pub snapshot_coordinator: Arc<crate::services::snapshot::SnapshotPublishCoordinator>,

    /// Phase 8 (F1) Continuous Discovery service: create/inspect discovery
    /// jobs. The background `DiscoveryJobExecutor` (spawned in `new`) consumes
    /// scheduled jobs from the same registry and republishes refined snapshots
    /// via `snapshot_coordinator`.
    pub discovery_service: Arc<crate::services::discovery::DiscoveryService>,

    /// Phase 8 (F5) External Collection service: register external lake tables
    /// un-copied and build/serve ProximaDB-owned indexes over them. Backs the
    /// v2 `external-collections` endpoints.
    pub external_collection_service:
        Arc<crate::services::external_collection::ExternalCollectionService>,

    /// Process-wide TurboQuant store registry (Phase H — Quantization
    /// Trait Convergence Plan). Owns one
    /// [`TurboQuantStore`](proximadb_vector::quantization::turboquant::TurboQuantStore)
    /// per collection, keyed on `collection_id`. Search paths that route
    /// through `lifecycle = ReadTime` (Phase C) pull the store from this
    /// registry; the
    /// [`TurboQuantAxisIndex`](crate::index::axis::indexes::TurboQuantAxisIndex)
    /// adapter (Phase D) constructs its inner `IdMapIndex` against the
    /// registry's store. xCatalog hydration (Phase I, follow-up) populates
    /// the registry at boot from `DerivedQuantizationLevel::TurboQuant`
    /// rows; collection-create writes back via
    /// `TurboQuantStoreRegistry::get_or_create`.
    ///
    /// `None` when `experimental-turboquant` is off OR the default
    /// constructor was used in a test path — consumers fall back to the
    /// full-precision scorer (correct but slower) when this slot is empty.
    /// Gated by the feature flag so default builds carry zero TurboQuant
    /// code in the SharedServices struct.
    #[cfg(feature = "experimental-turboquant")]
    pub turboquant_registry: Option<
        Arc<dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry>,
    >,
}

/// The three durable ABAC substrate stores, each behind a shared `Arc` handle so
/// the live `AbacEnforcer`(s) and the admin-provisioning writer observe the SAME
/// instance — a runtime provision is visible to the enforcer without a restart
/// (hot-reload, TD-ABAC control-plane). Opened once in
/// [`SharedServices::open_abac_stores`].
#[cfg(feature = "abac-policy")]
struct AbacDurableStores {
    authority: Arc<proximadb_abac::FileSystemAttributeAuthority>,
    bindings: Arc<proximadb_abac::FileSystemPolicyBindingStore>,
    predicate_objects: Arc<proximadb_abac::FileSystemPredicateObjectStore>,
    /// ADR-090 L1.2: the grant (entitlement) store, in the same durable dir.
    grants: Arc<proximadb_catalog::grants::FileSystemGrantStore>,
    /// TD-SEC-2 Slice C: per-tenant security posture, same durable dir.
    posture: Arc<proximadb_catalog::tenant_posture::FileSystemTenantPostureStore>,
}

impl SharedServices {
    /// Borrow the TurboQuant registry, if any. Convenience getter so call
    /// sites don't have to feature-gate the field access locally — this
    /// method is feature-gated to the same condition and returns `None`
    /// on default builds.
    ///
    /// Returns `None` when:
    ///   - The `experimental-turboquant` feature is off, OR
    ///   - The registry slot was never populated (test path using
    ///     `Default::default()`, or a future production path that opts
    ///     out of TurboQuant entirely).
    #[cfg(feature = "experimental-turboquant")]
    pub fn turboquant_registry(
        &self,
    ) -> Option<
        Arc<dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry>,
    > {
        self.turboquant_registry.clone()
    }
}

impl SharedServices {
    /// Open the ONE process-shared fenced `ConditionalKeyStore`
    /// (TD-OLTP-WIRING-1 / ADR-072). Returns `Some` only when the
    /// `oltp-integrity` feature is compiled AND a durable `data_dir` is
    /// configured — an in-memory store would lose uniqueness across restart. The
    /// WAL is replayed by `open`, so PK/FK fencing state survives a restart.
    /// `None` on default builds and embedded/ephemeral paths (unchanged behavior).
    fn open_shared_conditional_key_store(
        opt_config: Option<&crate::core::config::Config>,
    ) -> Option<Arc<dyn proximadb_storage_ports::ConditionalKeyStore>> {
        #[cfg(not(feature = "oltp-integrity"))]
        {
            let _ = opt_config;
            None
        }
        #[cfg(feature = "oltp-integrity")]
        {
            let data_dir = opt_config?.server.data_dir.clone();
            let path = std::path::Path::new(&data_dir).join("oltp-cks.wal");
            match proximadb_cks_local::LocalWalKeyStore::open(
                &path,
                proximadb_cks_local::SyncPolicy::PerOp,
            ) {
                Ok(store) => {
                    tracing::info!(
                        "oltp-integrity: fenced ConditionalKeyStore active at {}",
                        path.display()
                    );
                    Some(Arc::new(store) as Arc<dyn proximadb_storage_ports::ConditionalKeyStore>)
                }
                Err(e) => {
                    tracing::error!(
                        "oltp-integrity: failed to open ConditionalKeyStore at {}: {e}; fencing DISABLED",
                        path.display()
                    );
                    None
                }
            }
        }
    }

    /// Open the three durable ABAC substrate stores (`FileSystem*`) from
    /// `<data_dir>/abac/` once at boot, each behind a shared `Arc` handle. The
    /// live [`AbacEnforcer`](crate::security::rls::AbacEnforcer)(s) and the
    /// admin-provisioning writer share these SAME instances, so a runtime
    /// provision is visible to the enforcer without a restart (hot-reload,
    /// TD-ABAC control-plane).
    ///
    /// Returns `None` on default builds (the feature is OFF) and when there is no
    /// `data_dir` or a store cannot be opened — i.e. the status quo (no
    /// enforcement). It never synthesizes an allow: absent ⇒ no enforcer ⇒ no
    /// filtering, the same state as today.
    #[cfg(feature = "abac-policy")]
    fn open_abac_stores(
        opt_config: Option<&crate::core::config::Config>,
    ) -> Option<AbacDurableStores> {
        use proximadb_abac::{
            FileSystemAttributeAuthority, FileSystemPolicyBindingStore,
            FileSystemPredicateObjectStore,
        };

        let data_dir = opt_config?.server.data_dir.clone();
        let abac_dir = std::path::Path::new(&data_dir).join("abac");
        // A missing `data_dir` (embedded/ephemeral paths) ⇒ no durable substrate ⇒
        // no enforcer. Best-effort dir creation; a create failure ⇒ None too.
        if std::fs::create_dir_all(&abac_dir).is_err() {
            tracing::warn!(
                "abac-policy: could not create abac dir at {}",
                abac_dir.display()
            );
            return None;
        }

        let authority =
            match FileSystemAttributeAuthority::open(abac_dir.join("attribute-bindings.json")) {
                Ok(a) => Arc::new(a),
                Err(e) => {
                    tracing::error!(
                        "abac-policy: failed to open attribute authority at {}: {e}; \
                         enforcement DISABLED (no enforcer)",
                        abac_dir.display()
                    );
                    return None;
                }
            };
        let bindings =
            match FileSystemPolicyBindingStore::open(abac_dir.join("policy-bindings.json")) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!(
                        "abac-policy: failed to open policy binding store at {}: {e}; \
                     enforcement DISABLED (no enforcer)",
                        abac_dir.display()
                    );
                    return None;
                }
            };
        // TD-ABAC-4: durable predicate-object store — resolves the
        // `PolicyBinding.predicate_ref` ObjectIds to their `FilterExpression`s so
        // row-level enforcement (not just predicate-free table grants) works in
        // production. Empty ⇒ every predicate ref resolves fail-closed.
        let predicate_objects =
            match FileSystemPredicateObjectStore::open(abac_dir.join("predicate-objects.json")) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!(
                        "abac-policy: failed to open predicate object store at {}: {e}; \
                     enforcement DISABLED (no enforcer)",
                        abac_dir.display()
                    );
                    return None;
                }
            };

        tracing::info!(
            "abac-policy: durable ABAC stores active at {} (authority + policy binding store + predicate object store)",
            abac_dir.display()
        );
        let grants = match proximadb_catalog::grants::FileSystemGrantStore::open(&abac_dir) {
            Ok(store) => Arc::new(store),
            Err(error) => {
                tracing::warn!("abac-policy: could not open grant store: {error}");
                return None;
            }
        };
        let posture = match proximadb_catalog::tenant_posture::FileSystemTenantPostureStore::open(
            &abac_dir,
        ) {
            Ok(store) => Arc::new(store),
            Err(error) => {
                tracing::warn!("abac-policy: could not open tenant posture store: {error}");
                return None;
            }
        };

        Some(AbacDurableStores {
            authority,
            bindings,
            predicate_objects,
            grants,
            posture,
        })
    }

    /// Build a process-shared [`AbacEnforcer`] from already-opened shared stores,
    /// cloning the `Arc` handles so the enforcer reads the same instances an admin
    /// writer mutates (hot-reload). The policy-epoch source is in-memory today
    /// (follow-on: durable epochs).
    #[cfg(feature = "abac-policy")]
    fn build_enforcer_from_stores(
        stores: &AbacDurableStores,
    ) -> Arc<crate::security::rls::AbacEnforcer> {
        use proximadb_abac::InMemoryPolicyEpochs;
        let enforcer = crate::security::rls::AbacEnforcer::new(
            stores.authority.clone(),
            stores.predicate_objects.clone(),
            Arc::new(InMemoryPolicyEpochs::new()),
        )
        .with_binding_store(stores.bindings.clone())
        .with_grant_store(stores.grants.clone())
        .with_posture_store(stores.posture.clone());
        Arc::new(enforcer)
    }

    /// Open the durable stores and build the enforcer in one shot. Test-only
    /// convenience — production shares the opened stores across the vector/DML
    /// enforcers AND `AppState` (see [`open_abac_stores`] +
    /// [`build_enforcer_from_stores`]) so an admin provision is hot-visible to
    /// every reader. Gated to `test` so it carries no dead-code weight in
    /// `--lib` builds.
    #[cfg(all(feature = "abac-policy", test))]
    fn build_abac_enforcer(
        opt_config: Option<&crate::core::config::Config>,
    ) -> Option<Arc<crate::security::rls::AbacEnforcer>> {
        Self::open_abac_stores(opt_config).map(|s| Self::build_enforcer_from_stores(&s))
    }

    /// Create shared services with full business logic configuration
    /// SharedServices owns all business logic and configuration decisions
    /// Returns (SharedServices, CollectionService) - the collection service is needed by StorageEngine
    pub async fn new(
        metrics_collector: Option<Arc<MetricsCollector>>,
        storage_config: &crate::core::config::StorageConfig,
        orchestrator: Option<Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>>,
        // Optional full runtime config for hybrid/graph overrides
        opt_config: Option<&crate::core::config::Config>,
        // Which consumer is building the core; gates network-only background work
        // (see `ServiceProfile`). `Embedded` builds no Prometheus/billing surfaces.
        profile: ServiceProfile,
    ) -> Result<(Self, Arc<CollectionService>)> {
        info!("🔧 SharedServices: Initializing business logic hub for ALL protocols");
        debug!(
            "🔧 SharedServices::new - Starting with storage_config: {:?}",
            storage_config
        );

        // Initialize the process-global multitenant footer/index caches for the
        // PAX v2 ranged read path. Work-conserving elasticity: 256 MiB pooled,
        // an 8 MiB per-tenant floor (protected working set), and a 128 MiB (50%)
        // hard ceiling as a runaway guard — a solo tenant borrows idle capacity
        // up to 128 MiB; under pressure each is reclaimed toward the fair share
        // (total / active tenants). ObjectStoreVectorRecordStore auto-picks up.
        // Multitenant footer cache: 256 MiB pool, 8 MiB floor, 128 MiB hard
        // ceiling, work-conserving elasticity (0.9 watermark). An operator may
        // supply a tier policy via PROXIMADB_CACHE_TIERS_PATH (generic JSON, no
        // commercial data in OSS); the resolver maps tenant→tier through the
        // process-global registry the auth layer stamps (set_tenant_tier). With
        // no config it stays uniform elastic fair share.
        let cache_total_bytes: u64 = 256 * 1024 * 1024;
        let cache_budget = proximadb_cache::CacheBudget::new(cache_total_bytes, 128 * 1024 * 1024)
            .with_floor(8 * 1024 * 1024)
            .with_high_watermark(0.9);
        let limits_resolver = std::env::var("PROXIMADB_CACHE_TIERS_PATH")
            .ok()
            .and_then(|path| std::fs::read_to_string(&path).ok())
            .and_then(|json| proximadb_cache::TierPolicy::from_json(&json).ok())
            .map(|policy| {
                let policy = std::sync::Arc::new(policy);
                let default_tier = policy.default_tier.clone();
                // TD-TENANT-3 S3: probe the canonical tier id first, then the
                // operator's original spelling, so two spellings of one
                // entitlement cannot land in different cache buckets.
                let probe_policy = std::sync::Arc::clone(&policy);
                let tenant_to_tier: std::sync::Arc<dyn Fn(&str) -> String + Send + Sync> =
                    std::sync::Arc::new(move |t: &str| {
                        crate::services::record_store::tenant_tier_policy_key(t, |key| {
                            probe_policy.has_tier(key)
                        })
                        .unwrap_or_else(|| default_tier.clone())
                    });
                policy.resolver(cache_total_bytes, tenant_to_tier)
            });
        if limits_resolver.is_some() {
            info!("🎟️  SharedServices: cache tier policy loaded from PROXIMADB_CACHE_TIERS_PATH");
        }
        crate::services::record_store::init_segment_caches(cache_budget, limits_resolver);

        // Publish per-tenant cache stats for observability/chargeback (the
        // multitenant fairness + noisy-neighbor signal) on the consumption gauge.
        // Server-only: this 30s emitter feeds a Prometheus/network scrape surface
        // that no in-process embedded consumer reads (co-design tenets 1 & 5).
        if profile.is_server() {
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
                loop {
                    tick.tick().await;
                    let stats = crate::services::record_store::segment_cache_tenant_stats();
                    if !stats.is_empty() {
                        crate::metrics::consumption_metrics::record_cache_tenant_stats(
                            "footer", &stats,
                        );
                    }
                }
            });
        }

        let catalog_manager = Arc::new(crate::catalog::CatalogManager::new());
        let model_registry_service = Arc::new(
            proximadb_catalog::model_registry_service::CatalogModelRegistryService::new(
                catalog_manager.clone(),
            ),
        );
        // Inject the object-store filesystem resolver (root half of the
        // CatalogFilesystemResolver port-inversion). Lazily creates a
        // FilesystemFactory only if an s3://gs://az:// catalog URL is used;
        // local `file://` setups never touch it.
        catalog_manager
            .set_filesystem_resolver(Arc::new(crate::catalog::LazyFilesystemResolver::new()))
            .await;

        // SharedServices owns metadata configuration logic
        info!(
            "🔧 SharedServices: Metadata URL from config: {}",
            storage_config.metadata_url
        );
        if catalog_manager.list_catalogs().await.is_empty() {
            // System-catalog redesign: the default catalog is the read-heavy,
            // WAL-backed `SystemCatalog` (in-RAM authority + canonical WAL)
            // rather than the file-per-object `NativeCatalog`. It serves catalog
            // reads from RAM and persists each DDL as one fsync'd WAL append.
            // `PROXIMADB_DISABLE_SYSTEM_CATALOG` is a kill-switch back to
            // `NativeCatalog` for the duration of the cutover.
            let disable_system_catalog = std::env::var("PROXIMADB_DISABLE_SYSTEM_CATALOG").is_ok();
            let metadata_url = storage_config.metadata_url.clone();
            // The backend is DECIDED ONCE, as a value that carries its own
            // reason — see `catalog_backend_choice`. Scheme handling (ADR-036
            // aliases) and the Phase-5d object-store rule live there with unit
            // tests; this site executes the decision and, critically, LOGS every
            // arm. The fallback used to log nothing at all, so an operator's
            // only signal that their deployment had dropped off the WAL-backed
            // catalog was the absence of a line they had no reason to look for.
            use crate::services::catalog_backend_choice::{
                DefaultCatalogBackend, resolve_default_catalog_backend,
            };
            let backend = resolve_default_catalog_backend(
                &metadata_url,
                disable_system_catalog,
                opt_config.map(|c| c.server.data_dir.as_path()),
            );
            info!(
                "🗂️  SharedServices: default catalog backend = {} (metadata_url {})",
                backend.label(),
                metadata_url
            );

            match backend {
                DefaultCatalogBackend::SystemCatalogLocal { base } => {
                    // Phase 5 (two-tier operator/account): route the system
                    // catalog's own WAL + snapshot under the DrPathBuilder-validated
                    // operator control-plane prefix (`_operator/catalog/…`) so
                    // catalog I/O honours the structural-isolation mandate instead
                    // of a raw `{base}/system-catalog.wal`. Flag-gated + inert by
                    // default (mirrors the warehouse DrPath opt-in pattern): the local path
                    // is unchanged until a deployment opts in, keeping existing
                    // on-disk catalog state in place. The catalog is `Operator`-roled.
                    let wal_path = if std::env::var("PROXIMADB_CATALOG_DRPATH").is_ok() {
                        base.join(
                        crate::storage::trait_components::path_resolver::DrPathBuilder::system_catalog_wal_relpath(),
                    )
                    } else {
                        base.join("system-catalog.wal")
                    };
                    if let Some(parent) = wal_path.parent() {
                        tokio::fs::create_dir_all(parent).await.with_context(|| {
                            format!("creating system-catalog dir {}", parent.display())
                        })?;
                    }
                    let system_catalog =
                        crate::services::system_catalog::SystemCatalog::open("default", &wal_path)
                            .await
                            .with_context(|| {
                                format!("opening SystemCatalog WAL at {}", wal_path.display())
                            })?;
                    catalog_manager
                        .register(Arc::new(system_catalog))
                        .await
                        .context("Failed to register default SystemCatalog backend")?;
                    info!(
                        "✅ SharedServices: registered WAL-backed SystemCatalog (default) at {}",
                        wal_path.display()
                    );
                }
                DefaultCatalogBackend::SystemCatalogObjectStore { data_dir } => {
                    use crate::storage::trait_components::path_resolver::DrPathBuilder;
                    // Per-DDL WAL on the local working volume under the operator
                    // control-plane prefix; snapshot blob in the object store.
                    let wal_path = data_dir.join(DrPathBuilder::system_catalog_wal_relpath());
                    if let Some(parent) = wal_path.parent() {
                        tokio::fs::create_dir_all(parent).await.with_context(|| {
                            format!("creating system-catalog dir {}", parent.display())
                        })?;
                    }
                    let snapshot_store = Arc::new(
                    crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::from_url(
                        &metadata_url,
                        DrPathBuilder::system_catalog_manifests_subprefix(),
                    )
                    .with_context(|| {
                        format!("opening object-store catalog snapshot at {metadata_url}")
                    })?,
                );
                    let system_catalog = Arc::new(
                        crate::services::system_catalog::SystemCatalog::open_with_snapshot_store(
                            "default",
                            &wal_path,
                            snapshot_store,
                        )
                        .await
                        .with_context(|| {
                            format!(
                                "opening object-store SystemCatalog (WAL {})",
                                wal_path.display()
                            )
                        })?,
                    );
                    // Phase 6b: in a multi-pod deployment, tail the object-store
                    // snapshot so this pod's relcache stays coherent with DDL another
                    // pod commits (sinval-style lazy reload), and so a superseded
                    // owner steps down to read-only promptly. Inert by default
                    // (single-pod): gated behind `PROXIMADB_CATALOG_FOLLOWER_POLL_SECS`
                    // (> 0 to enable). The handle is detached — the loop is a
                    // cooperative tokio task that does no work when nothing changed.
                    if let Some(secs) = std::env::var("PROXIMADB_CATALOG_FOLLOWER_POLL_SECS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .filter(|n| *n > 0)
                    {
                        system_catalog
                            .clone()
                            .spawn_follower_poll(std::time::Duration::from_secs(secs));
                        info!(
                            "✅ SharedServices: catalog follower poll enabled (every {}s) — \
                         tailing object-store snapshot for cross-pod coherence",
                            secs
                        );
                    }
                    catalog_manager
                        .register(system_catalog)
                        .await
                        .context("Failed to register object-store SystemCatalog backend")?;
                    info!(
                        "✅ SharedServices: registered object-store-backed SystemCatalog (default); \
                     local WAL at {}, snapshot in {}",
                        wal_path.display(),
                        metadata_url
                    );
                }
                DefaultCatalogBackend::NativeFallback { reason } => {
                    // Loud, and it names what to change. `NativeCatalog` is
                    // file-per-object (a `read_dir` per `list_tables`) where the
                    // SystemCatalog serves from RAM, so this is a per-request
                    // round-trip class the operator did not choose — they must be
                    // able to see that they are on it.
                    warn!(
                        "⚠️  SharedServices: falling back to NativeCatalog for the default \
                     catalog — {}",
                        reason.explain()
                    );
                    catalog_manager
                        .create_native_catalog("default", &metadata_url)
                        .await
                        .context("Failed to initialize default xCatalog backend")?;
                }
            }
        }
        // TD-080 (2026-05-28 round 2): explicitly designate the "default"
        // catalog as the manager's default so `catalog_manager.default_catalog()`
        // returns Ok at boot. Without this, `ProximaDB::new` at `database.rs:159`
        // never wires `precision_resolver` into RequestHandlers, and fp16
        // collections emit canonical_bytes under `precision="fp32"` even
        // for direct REST/gRPC INSERT. The bug surfaced in test harnesses
        // (the metric test stayed #[ignore]'d) but was also latent in
        // production — every startup logged the degraded-boot warning at
        // `database.rs:188`. `set_default_catalog` is idempotent; the
        // explicit fallback to "default" matches the create call above.
        let catalogs_now = catalog_manager.list_catalogs().await;
        if let Some(first) = catalogs_now.first() {
            let target = if catalogs_now.iter().any(|n| n == "default") {
                "default".to_string()
            } else {
                first.clone()
            };
            if let Err(e) = catalog_manager.set_default_catalog(&target).await {
                tracing::warn!(
                    error = %e,
                    target = %target,
                    "SharedServices: failed to designate default catalog; \
                     precision_resolver and downstream catalog-driven paths \
                     will fall back to degraded behaviour"
                );
            }
        }

        // TD-ABAC-11 / ADR-087: make the policy-lookup key total before any
        // authenticated request can reach an armed enforcer. This is an
        // explicit bootstrap mutation, not an incidental side effect of the
        // first table create. Both production catalog backends persist it
        // (SystemCatalog WAL/snapshot; NativeCatalog account sidecar).
        let default_tenant_stable_id = catalog_manager
            .ensure_tenant_stable_id(proximadb_tenant::DEFAULT_TENANT)
            .await
            .context("Failed to mint the default tenant stable id at catalog bootstrap")?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "default catalog cannot mint a stable id for tenant '{}'",
                    proximadb_tenant::DEFAULT_TENANT
                )
            })?;
        info!(
            tenant = proximadb_tenant::DEFAULT_TENANT,
            tenant_stable_id = default_tenant_stable_id,
            "✅ SharedServices: default tenant stable id is durable"
        );

        // Phase P (Quantization Trait Convergence Plan): hoist the
        // TurboQuant store registry construction to BEFORE the
        // `collection_service` is built, so the SAME `Arc<dyn>` instance
        // can flow into:
        //   1. `CollectionService::with_turboquant_registry()` — the
        //      create-time wire (Phase P Site 1).
        //   2. The boot-time hydration loop below — the boot-time wire
        //      (Phase P Site 2).
        //   3. The `SharedServices` struct literal at the end of `new()`
        //      — the exposed-to-consumers slot (Phase H).
        // Sharing one `Arc` means create-time registrations land in the
        // same map the boot hydration populated.
        #[cfg(feature = "experimental-turboquant")]
        let turboquant_registry: Arc<
            dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry,
        > = Arc::new(
            crate::compute::quantization::turboquant_store_registry::InMemoryTurboQuantStoreRegistry::new(),
        );

        let collection_service = {
            let cs = CollectionService::new(storage_config.clone())
                .await?
                .with_catalog_manager(catalog_manager.clone());
            #[cfg(feature = "experimental-turboquant")]
            let cs = cs.with_turboquant_registry(turboquant_registry.clone());
            Arc::new(cs)
        };
        debug!("✅ SharedServices: CollectionService created successfully");

        // Collection service will be injected into StorageEngine by ProximaDB::new
        info!("✅ SharedServices: Collection service created for injection into StorageEngine");

        // ADR-031 allocator unification: raise the collection-id allocator floor
        // above every existing object_id — both `collection.id` (numeric) AND
        // `schema.object_id` — so a freshly minted collection id can never
        // collide with a legacy (pre-unification) schema.object_id (which would
        // corrupt the oid→name index). Best-effort: a failure logs + continues
        // (startup must not block on this).
        {
            let max_coll_id = collection_service
                .list_collections()
                .await
                .unwrap_or_default()
                .iter()
                .filter_map(|c| c.id.parse::<u64>().ok())
                .max();
            let max_schema_oid = match catalog_manager.default_catalog().await {
                // TD-CAT-7: `unwrap_or(None)` used to collapse three different
                // outcomes into one — no authority, an authority with nothing
                // minted, and an authority that FAILED — so a broken floor
                // recovery was indistinguishable from a federated catalog and
                // this collision guard silently did nothing. Name them apart.
                Ok(cat) if cat.identity_authority().is_none() => {
                    debug!(
                        "collection-id floor recovery: catalog '{}' is not an identity \
                         authority; no object_id floor to raise",
                        cat.name()
                    );
                    None
                }
                Ok(cat) => match cat.max_object_id().await {
                    Ok(max) => max,
                    Err(e) => {
                        warn!(
                            "collection-id floor recovery: catalog '{}' is an identity \
                             authority but failed to report its max object_id ({e}); \
                             a freshly minted collection id could collide with a legacy \
                             schema.object_id",
                            cat.name()
                        );
                        None
                    }
                },
                Err(e) => {
                    warn!("collection-id floor recovery: default catalog unavailable: {e}");
                    None
                }
            };
            if let Some(floor) = max_coll_id.max(max_schema_oid) {
                crate::services::collection::manager::recover_collection_id_floor(floor);
                debug!("collection-id allocator floor raised to {floor} (max existing object_id)");
            }
        }

        // Phase P Site 2 — boot-time hydration of the TurboQuant store
        // registry. After a restart, every existing collection whose
        // proto QuantizationConfig set `enable_turboquant=true` needs
        // its store re-registered so the first search reaches the
        // kernel instead of a silent full-precision fallback. Iterate
        // the existing collection list, project each collection's
        // proto into a `TurboQuantHydrationRow`, and drive the Phase O
        // `hydrate_registry_from_policy_rows` helper.
        //
        // Failures are logged and the loop continues — a single bad
        // collection MUST NOT block startup. The helper's per-row
        // "log + continue" contract (Phase O) handles the registry
        // side; we mirror it here for the catalog lookup side.
        #[cfg(feature = "experimental-turboquant")]
        {
            use crate::compute::quantization::turboquant_store_registry::{
                TurboQuantHydrationRow, hydrate_registry_from_policy_rows,
            };
            let collections = collection_service
                .list_collections()
                .await
                .unwrap_or_default();
            let mut rows: Vec<TurboQuantHydrationRow> = Vec::new();
            for c in &collections {
                // proto Collection.config carries the dimension; absence
                // is skipped (a collection with no config can't have
                // declared TurboQuant intent either).
                let Some(cfg) = c.config.as_ref() else {
                    continue;
                };
                match collection_service.native_quantization_config(&c.id).await {
                    Ok(Some(qcfg)) if qcfg.enable_turboquant.unwrap_or(false) => {
                        let seed = proximadb_quantization_types::derive_rotation_seed(&c.id);
                        rows.push(TurboQuantHydrationRow {
                            collection_id: c.id.clone(),
                            dim: cfg.dimension as usize,
                            bit_width: 4,
                            calibration_mode: "tq_plus".to_string(),
                            rotation_seed: seed,
                        });
                    }
                    Ok(_) => {} // Non-TurboQuant collection — silent skip.
                    Err(e) => tracing::warn!(
                        target: "proximadb::turboquant::hydrate",
                        collection_id = %c.id,
                        error = %e,
                        "Phase P boot hydration: native_quantization_config lookup failed; skipping",
                    ),
                }
            }
            if !rows.is_empty() {
                let attempted = rows.len();
                let hydrated =
                    hydrate_registry_from_policy_rows(turboquant_registry.as_ref(), &rows).await;
                tracing::info!(
                    target: "proximadb::turboquant::hydrate",
                    attempted,
                    hydrated,
                    "Phase P boot hydration: TurboQuant stores re-registered from catalog",
                );
            }
        }

        // 🚀 Create VectorOperationsService directly for 40-60% performance improvement
        // Use WAL config from TOML configuration
        debug!("🔧 SharedServices::new - Converting WAL config from TOML...");
        let mut wal_config = Self::convert_toml_to_wal_config(&storage_config.wal_config);

        // Override data_directories with storage_locations if available
        // This ensures embedded mode and config-specified storage locations are honored
        if !storage_config.storage_locations.is_empty() {
            wal_config.multi_disk.data_directories = storage_config
                .storage_locations
                .iter()
                .map(|loc| {
                    // Ensure proper file:// URL format
                    let url = if loc.url.starts_with("file://") {
                        loc.url.clone()
                    } else if loc.url.starts_with("/") {
                        format!("file://{}", loc.url)
                    } else {
                        loc.url.clone()
                    };
                    debug!(
                        "🔧 SharedServices: WAL directory URL from storage_locations: {}",
                        url
                    );
                    url
                })
                .collect();
            info!(
                "📂 SharedServices: WAL data directories set from storage_locations: {:?}",
                wal_config.multi_disk.data_directories
            );
        }
        debug!("✅ SharedServices::new - WAL config converted successfully from TOML");

        // Create filesystem factory for engines
        debug!("🔧 SharedServices::new - Creating filesystem factory for engines...");
        let filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(
                crate::storage::persistence::filesystem::FilesystemConfig::default(),
            )
            .await?,
        );
        debug!("✅ SharedServices::new - Filesystem factory for engines created successfully");

        // TD-DOC-PUSHDOWN-1: publish the storage filesystem factory as a process singleton so the
        // DataFusion `documents(collection)` UDTF can build a `PaxTableProvider` over a collection's
        // `.pax` segments for predicate pushdown. Idempotent (first wins); mirrors the
        // document/timeseries service singletons.
        crate::services::document_service::set_filesystem_factory(filesystem_factory.clone());

        // Create VIPER engine
        debug!("🔧 SharedServices::new - Creating VIPER engine...");
        let viper_config = crate::core::config::ViperConfig::default();
        debug!("🔧 SharedServices::new - VIPER config created, now creating engine...");
        let _viper_engine = Arc::new(
            crate::storage::engines::viper::ViperEngine::from_core_config(
                viper_config,
                filesystem_factory.clone(),
            )
            .await?,
        );
        debug!("✅ SharedServices::new - VIPER engine created successfully");

        // Vector Object Economy Phase 4 (1-B + 2-B): construct the
        // process-wide per-collection directory cache up front so both
        // the SST engine (producer side — emits directory updates after
        // atomic commit) and the vector operations service (consumer
        // side — loads cached directories during search) hold the same
        // `Arc`.
        let directory_cache = Arc::new(
            crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache::new(),
        );

        // Phase 6: per-collection pinning registry constructed up
        // front so REST handlers (control plane) and the SST
        // tier-migration integration (data plane) share the same
        // `Arc`. Operator PATCH calls land in the registry; the
        // tier-migration integration consults it during flush and
        // evaluate cycles, overriding policy when a pin is set.
        //
        // Slice 6.5: when `opt_config` is provided, the registry
        // auto-persists to `<data_dir>/pinning/registry.json` so
        // pins survive process restarts. Tests / embedded paths
        // without opt_config use the in-memory constructor.
        let pin_registry = match opt_config {
            Some(cfg) => {
                let registry_path = cfg.server.data_dir.join("pinning").join("registry.json");
                info!(
                    "📌 SharedServices: pin registry persistence enabled at {}",
                    registry_path.display()
                );
                crate::storage::collection_pinning::new_shared_at(registry_path)
            }
            None => crate::storage::collection_pinning::new_shared(),
        };

        // Phase 7.2: per-collection cache-affinity registry. In-memory
        // only (no persistence) — entries naturally re-populate from
        // the first query after a restart, so a stale persisted entry
        // would be more confusing than helpful. TTL defaults to 60s;
        // entries older than that are treated as cold.
        let affinity_registry = crate::cluster::cache_affinity::new_shared();
        info!("🧭 SharedServices: cache-affinity registry ready (TTL 60s)");

        // Slice 2 of tenant-pod-affinity: per-(tenant, collection)
        // primary-pod registry. Unlike cache_affinity above, this is
        // durable — writes MUST route to the bound pod for WAL
        // memtable consistency (see the 3-stage search at
        // `src/services/operations/vectors/legacy.rs:2827-2858`).
        // Persistence path is `<data_dir>/primary_pods/registry.json`
        // when opt_config is provided; tests / embedded paths fall
        // back to the in-memory constructor. Subsequent slices will
        // add the REST endpoint, the gateway router consultation,
        // and the xCatalog backing (deferred from this slice to keep
        // merge surface small while Phase 7 settles).
        let primary_pod_persistence_mode =
            crate::cluster::primary_pod_registry::resolve_persistence_mode();
        let primary_pod_registry = match opt_config {
            Some(cfg) => {
                let registry_path = cfg
                    .server
                    .data_dir
                    .join("primary_pods")
                    .join("registry.json");
                info!(
                    "📍 SharedServices: primary-pod registry persistence at {} (mode={})",
                    registry_path.display(),
                    primary_pod_persistence_mode.label()
                );
                crate::cluster::primary_pod_registry::new_shared_at_with_mode(
                    registry_path,
                    primary_pod_persistence_mode,
                )
            }
            None => crate::cluster::primary_pod_registry::new_shared(),
        };

        // Slice 5c: pull any catalog-side primary_pod bindings into
        // the registry. Existing entries (loaded from the JSON
        // sidecar in `new_shared_at`) take precedence per the
        // transition policy. Failures are non-fatal — they only mean
        // the registry has no catalog backfill, which is the same
        // state as a fresh install. A catalog mirror that lags will
        // also surface in the slice 5b.2 mirror-failure metric.
        match crate::cluster::primary_pod_registry::hydrate_from_catalog(
            &primary_pod_registry,
            &catalog_manager,
        )
        .await
        {
            Ok(report) => {
                info!(
                    "🔁 SharedServices: primary-pod catalog hydration: \
                     seen={} inserted={} skipped_existing={}",
                    report.seen, report.inserted, report.skipped_existing
                );
            }
            Err(err) => {
                warn!(
                    "⚠️ SharedServices: primary-pod catalog hydration failed: {} \
                     (registry continues with sidecar-only contents)",
                    err
                );
            }
        }

        // Slice 5d.1: push registry entries the catalog is missing
        // back into the catalog. Convergence step — over time the
        // catalog reaches feature-parity with the JSON sidecar so
        // slice 5d.2 can flip persistence priority safely. The
        // `migrated` counter trending to zero across boots is the
        // operator's "catalog is now authoritative" signal.
        match crate::cluster::primary_pod_registry::migrate_registry_to_catalog(
            &primary_pod_registry,
            &catalog_manager,
        )
        .await
        {
            Ok(report) => {
                info!(
                    "📤 SharedServices: primary-pod sidecar→catalog migration: \
                     seen={} migrated={} already_present={} skipped_table_missing={} failed={}",
                    report.seen,
                    report.migrated,
                    report.already_present,
                    report.skipped_table_missing,
                    report.failed
                );
            }
            Err(err) => {
                warn!(
                    "⚠️ SharedServices: primary-pod sidecar→catalog migration failed: {} \
                     (registry stays sidecar-primary)",
                    err
                );
            }
        }

        // Create WAL manager for two-stage search FIRST so the SST
        // engine can read its global manifest singleton when wiring the
        // Phase 5 freshness LSN source.
        debug!("🔧 SharedServices::new - Creating WAL manager for two-stage search...");
        let wal_manager = {
            use crate::storage::persistence::write_ahead_log::WriteAheadLogManager;

            // Create WAL manager directly. The batch-serialization strategy
            // stack was removed — the manager routes on `config.strategy_type`
            // plus the global write buffer, never a `WALBatchStrategy` object.
            Arc::new(WriteAheadLogManager::new(wal_config.clone()).await?)
        };
        debug!("✅ SharedServices::new - WAL manager created successfully");

        // Phase 5 (Slice 5.2): try to resolve the global manifest
        // singleton and wrap it as a `FreshnessLsnSource`. When the
        // singleton hasn't been initialised yet (some embedded/test
        // paths), the engine falls back to emitting `freshness_lsn = 0`
        // — strong-route readers will simply always re-scan the WAL
        // delta, which is correct but more expensive.
        let freshness_lsn_source: Option<
            Arc<dyn crate::storage::engines::sst::object_economy_directory::FreshnessLsnSource>,
        > = crate::storage::persistence::write_ahead_log::manifest::get_service().map(|svc| {
            Arc::new(
                crate::storage::persistence::write_ahead_log::manifest::WalCursorLsnSource::new(
                    svc,
                ),
            )
                as Arc<
                    dyn crate::storage::engines::sst::object_economy_directory::FreshnessLsnSource,
                >
        });

        // Create SST engine
        debug!("🔧 SharedServices::new - Creating SST engine...");
        let sst_engine = {
            let sst_config = storage_config.effective_sst_config();
            let distance_compute =
                Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());
            let mut engine = crate::storage::engines::sst::SstEngine::new_with_config(
                sst_config,
                filesystem_factory.clone(),
                distance_compute,
            )
            .await?
            .with_directory_cache(directory_cache.clone());
            if let Some(src) = freshness_lsn_source.clone() {
                engine = engine.with_freshness_lsn_source(src);
            }

            // Catalog-driven storage-URL resolution: attach the catalog/config
            // resolver so `get_collection_storage_url` resolves the real
            // per-collection location (retires the `/data/collections/{id}`
            // placeholder; the catalog is the single source of truth). Catalog
            // first → `ConfigFallbackResolver` default, wrapped in `CachedResolver`.
            // (WAL injection deferred: the WAL's fallback returns the base-location
            // PREFIX, while these resolvers return the ROOT `{base}/{id}`;
            // reconciling those contracts is a follow-up. SstEngine resolving
            // correctly already achieves write/read agreement.)
            {
                use crate::storage::trait_components::path_resolver::{
                    CachedResolver, CatalogResolver, CollectionPathResolver, CompositeResolver,
                    ConfigFallbackResolver,
                };
                let chain: Vec<Arc<dyn CollectionPathResolver>> = vec![
                    Arc::new(CatalogResolver::new()),
                    Arc::new(ConfigFallbackResolver::default()),
                ];
                engine = engine.with_path_resolver(Arc::new(CachedResolver::new(Arc::new(
                    CompositeResolver::new(chain),
                ))));
            }

            // Attach tier-migration integration when configured. Reads
            // the `[storage.sst_config.tiering]` block; defaults to
            // disabled. When `enabled = true`, the engine's
            // flush / search / compaction hooks start emitting access
            // events, flush-tier decisions, and migration evaluations
            // (see `src/storage/engines/sst/{search,flush}/coordinator.rs`).
            //
            // The integration's background evaluation loop is started
            // here so the policy engine can autonomously evaluate
            // pending migrations on its configured cadence.
            if let Some(tiering_cfg) = storage_config
                .sst_config
                .as_ref()
                .and_then(|sc| sc.tiering.clone())
            {
                if tiering_cfg.enabled {
                    use crate::storage::engines::sst::tiering_integration::SstTieringIntegration;
                    use crate::storage::tiering::TierMigrationExecutor;

                    // Build the migration executor first — it shares the
                    // filesystem factory with the engine so file://↔s3://
                    // moves use the same backend pool. Per-tier paths are
                    // pulled directly from the tiering config block.
                    //
                    // T2.2: Wire cache invalidation callback — migrations
                    // invalidate stale cache entries when data moves between
                    // tiers. Uses lazy global lookup since orchestrator
                    // is registered later; migrations run in background so
                    // the global is available when invoked.
                    let cache_invalidator =
                        std::sync::Arc::new(|collection: &str, item_id: &str| {
                            if let Some(orch) =
                                crate::storage::cache::orchestrator::CrossCacheOrchestrator::global(
                                )
                            {
                                let key = format!("{collection}/{item_id}");
                                // Fire-and-forget invalidation; errors logged by orchestrator
                                drop(tokio::spawn(async move {
                                    let _ = orch.orchestrate_cascade_invalidation(&key).await;
                                }));
                            }
                        });
                    let executor = Arc::new(
                        TierMigrationExecutor::from_tiering_config(
                            filesystem_factory.clone(),
                            &tiering_cfg,
                        )
                        .with_cache_invalidator(cache_invalidator),
                    );

                    match SstTieringIntegration::new(tiering_cfg) {
                        Ok(integration) => {
                            // Attach the executor BEFORE start() so the
                            // background eval loop, when it wakes, has
                            // somewhere to dispatch migration tasks.
                            // Attach the pin registry so flush-tier and
                            // evaluation honor operator pins (Phase 6
                            // data plane).
                            let mut integration = integration
                                .with_executor(executor)
                                .with_pin_registry(pin_registry.clone());
                            if let Err(e) = integration.start().await {
                                warn!(
                                    "⚠️ SharedServices: tier-migration integration failed to start ({}); continuing without tiering",
                                    e
                                );
                            } else {
                                info!(
                                    "🪜 SharedServices: SST tier-migration integration started — flush/search/compaction hooks active, executor dispatching tasks"
                                );
                                engine = engine.with_tiering_integration(Arc::new(integration));
                            }
                        }
                        Err(e) => warn!(
                            "⚠️ SharedServices: tier-migration integration could not be constructed ({}); continuing without tiering",
                            e
                        ),
                    }
                } else {
                    debug!(
                        "🪜 SharedServices: SST tier-migration configured but disabled (enabled=false); hooks remain no-ops"
                    );
                }
            }

            Arc::new(engine)
        };
        debug!("✅ SharedServices::new - SST engine created successfully");

        // Clone SST engine reference for DocumentService (used later for DocumentStrategy)
        let sst_engine_for_documents: Arc<dyn crate::storage::traits::UnifiedStorageFormat> =
            sst_engine.clone();

        // TD-075 / Phase 8 F2: the recall-probe gate is created here (rather than
        // inline in the return struct) so the same instance is shared by the
        // AxisManager (which consults it before the quantized route) and the
        // SharedServices field that AppState/route-health read.
        let recall_probe_gate = Arc::new(crate::catalog::RecallProbeGate::new());

        // The Cargo feature is compile-time capability only. This runtime
        // master switch must also be on before AXIS is registered with serving
        // or any AXIS background maintenance loop is started.
        #[cfg(feature = "axis")]
        let axis_runtime_enabled = storage_config.optimization.enable_axis_indexes;

        // Create AxisManager for index operations (AXIS-gated: the whole index
        // engine + its registrations vanish from a PAX-exact-scan build).
        #[cfg(feature = "axis")]
        let axis_manager = {
            debug!("🔧 SharedServices::new - Creating AxisManager for index operations...");
            let mut axis_manager_inner =
                crate::index::AxisManager::new(crate::index::AxisConfig::default()).await?;
            axis_manager_inner.set_recall_probe_gate(recall_probe_gate.clone());
            // ADR-023 R3 Slice 4: give AXIS the shared FilesystemFactory so index
            // persistence + cold-load can dispatch by scheme (s3/adls/gs/file).
            axis_manager_inner.set_filesystem_factory(filesystem_factory.clone());
            // Route index persistence through an object-store URI when configured
            // (PROXIMADB_INDEX_PERSIST_URL=s3://bucket/prefix | adls://… | gs://…) —
            // the cold-load path then reads only [header]+[COLD]+probed clusters via
            // byte-range GETs. Otherwise persist under the local data dir so a cold
            // collection warms from disk on first query (TD-087 Slice B; no-op
            // without a data dir).
            if let Ok(url) = std::env::var("PROXIMADB_INDEX_PERSIST_URL") {
                axis_manager_inner.set_index_persist_url(url);
            } else if let Some(cfg) = opt_config {
                axis_manager_inner.set_index_persist_dir(cfg.server.data_dir.join("axis_indexes"));
            }

            // CATALOG_OBJECT_MODEL #3 read-port: make catalog-resolved index locations
            // live for ALL collections — boot-present AND runtime-created — by injecting
            // a catalog resolver that AXIS pulls from on demand (and memoizes). For each
            // collection's VectorAnn projection, an explicit `projection.location` is
            // honored (relocated/tiered indexes); `PROXIMADB_INDEX_CATALOG_PATHS=1`
            // additionally opts the fleet into the DrPathBuilder `indexes/<projection>/`
            // layout. Default-off and additive: with no projection locations set the
            // resolver returns `None` and AXIS keeps the `index_persist_url`/`dir`
            // convention (mixed-safe). The resolver is catalog-free at the AXIS seam —
            // this adapter lives in the control layer (dependency inversion).
            {
                let migrate = std::env::var_os("PROXIMADB_INDEX_CATALOG_PATHS").is_some();
                axis_manager_inner.set_index_location_resolver(Arc::new(
                    crate::catalog::index_location_resolver::CatalogIndexLocationResolver::new(
                        catalog_manager.clone(),
                        migrate,
                    ),
                ));
            }

            let m = Arc::new(axis_manager_inner);
            debug!("✅ SharedServices::new - AxisManager created successfully");
            if axis_runtime_enabled {
                // Make AXIS available only when both its compile capability and
                // process-level policy are enabled. Collection-level intent is
                // checked at the ingest/search boundaries.
                crate::storage::entity_store::orion_backend::set_global_axis_manager(m.clone());
                crate::storage::engines::sst::core::set_sst_axis_manager(m.clone());
                crate::storage::common::axis_flush_hook::set_flush_axis_manager(m.clone());
                debug!("✅ SharedServices::new - AXIS runtime enabled and registered with serving");
            } else {
                info!(
                    "AXIS capability compiled but runtime-disabled by storage.optimization.enable_axis_indexes=false"
                );
            }
            m
        };

        // Create VectorOperationsService with optimized architecture and two-stage search
        debug!(
            "🔧 SharedServices::new - About to create VectorOperationsService with two-stage search..."
        );
        // Use the passed orchestrator if available, otherwise create a default one
        use crate::storage::cache::orchestrator::CrossCacheOrchestrator;
        let orchestrator = if let Some(orch) = orchestrator {
            orch
        } else {
            let mut default_orchestrator =
                CrossCacheOrchestrator::new((storage_config.cache_size_mb * 1024 * 1024) as usize);
            default_orchestrator.start_eviction_service(None);
            let orch = Arc::new(default_orchestrator);
            orch.clone().start_rebalancing_service();
            orch
        };
        // Always register globally — idempotent via OnceLock
        CrossCacheOrchestrator::register_global(orchestrator.clone());
        // Register the graph cache-hint bridge so graph engines (ORION) can hint
        // the cache through the leaf port without naming the concrete orchestrator.
        proximadb_storage_ports::register_graph_cache_hint(std::sync::Arc::new(
            crate::storage::cache::orchestrator::GraphCacheHintBridge,
        ));
        // TD-DECOMP-66: Register port-traits to resolve control-layer coupling edges.
        // orion-engine uses OrionMetricsPort instead of depending on metrics (control);
        // storage-operations uses CorpusVersionPort instead of depending on catalog (control).
        proximadb_storage_ports::register_orion_metrics(std::sync::Arc::new(
            proximadb_metrics::OrionMetricsAdapter,
        ));
        proximadb_storage_ports::register_corpus_version(std::sync::Arc::new(
            CorpusVersionBridge {
                registry: proximadb_catalog::CorpusVersionRegistry::global().clone(),
            },
        ));

        // =========================================================================
        // Initialize EventLog service and start AXIS consumer for async index building
        // This enables automatic AXIS index updates when data is flushed to storage
        // =========================================================================
        debug!("🔧 SharedServices::new - Initializing EventLog service for AXIS indexing...");

        // Use the global collection cache (shared across services)
        // Collections are registered in this cache when created via register_collection_in_cache()
        let collection_cache =
            crate::services::events::log::get_or_create_global_collection_cache();

        // Get base storage URL for EventLog persistence
        let base_storage_url = storage_config
            .storage_locations
            .first()
            .map(|loc| loc.url.clone());

        // Initialize the global EventLog service
        if let Err(e) = crate::services::events::log::initialize_event_log_service(
            collection_cache.clone(),
            filesystem_factory.clone(),
            base_storage_url.clone(),
        )
        .await
        {
            warn!(
                "⚠️ SharedServices: Failed to initialize EventLog service: {}. AXIS indexing will be disabled.",
                e
            );
        } else {
            info!("✅ SharedServices: EventLog service initialized successfully");

            // Start the AXIS EventLog consumer as a background task (AXIS-gated).
            // This polls the EventLog and builds AXIS indexes when flush events occur.
            #[cfg(feature = "axis")]
            if axis_runtime_enabled {
                let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                // TD-LIFECYCLE-1: registered (not leaked) so a clean shutdown
                // can stop the loop; see services::shutdown_registry.
                crate::services::shutdown_registry::register("axis-eventlog-consumer", shutdown_tx);
                if let Some(event_log_service) = crate::services::events::log::event_log_service() {
                    let _consumer_handle =
                        crate::index::axis::integration::eventlog_consumer::start_axis_consumer(
                            event_log_service.inner(),
                            axis_manager.clone(),
                            filesystem_factory.clone(),
                            collection_cache.clone(),
                            orchestrator.clone(),
                            shutdown_rx,
                        )
                        .await;

                    info!(
                        "✅ SharedServices: AXIS EventLog consumer started for explicitly indexed collections"
                    );
                } else {
                    warn!(
                        "⚠️ SharedServices: EventLog service unavailable after initialization; AXIS consumer not started."
                    );
                }
            }
        }

        // Open the durable ABAC stores ONCE here (outer scope) so the
        // vector-service enforcer, the DmlService enforcer, and the AppState
        // admin handles all share the SAME store instances — a runtime provision
        // is then hot-visible to every reader (TD-ABAC control-plane).
        #[cfg(feature = "abac-policy")]
        let abac_stores = Self::open_abac_stores(opt_config);
        #[cfg(feature = "abac-policy")]
        let abac_enforcer = abac_stores.as_ref().map(Self::build_enforcer_from_stores);

        // `directory_cache` constructed earlier (before SstEngine) so the
        // engine, the vector ops service, and the SharedServices public
        // field all share the same `Arc`.
        let vector_operations_service = {
            let svc = {
                #[cfg(feature = "axis")]
                {
                    VectorOperationsService::new(
                        sst_engine,
                        wal_manager,
                        axis_manager.clone(),
                        collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>,
                    )
                }
                #[cfg(not(feature = "axis"))]
                {
                    VectorOperationsService::new(
                        sst_engine,
                        wal_manager,
                        collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>,
                    )
                }
            }
            .with_orchestrator(Some(orchestrator.clone()))
            .with_directory_cache(directory_cache.clone())
            // Phase 7.2: thread the same affinity registry held by
            // the SharedServices field so search-path recordings and
            // operator inspection share state.
            .with_affinity_registry(affinity_registry.clone());
            #[cfg(feature = "axis")]
            let svc = svc.with_axis_runtime_enabled(axis_runtime_enabled);
            // FA-2 (abac-policy): wire the durable ABAC enforcer into the vector
            // service so `unified_search_native` enforces the subject's accessibility
            // predicate on every search (push-down for conjunctive policies, post-filter
            // otherwise). Default-OFF ⇒ default builds are byte-for-byte unchanged; `None`
            // ⇒ no enforcer ⇒ no per-record filtering (the status quo). Mirrors the
            // DmlService wiring (`build_abac_enforcer`, TD-ABAC-2).
            #[cfg(feature = "abac-policy")]
            let svc = match &abac_enforcer {
                Some(enforcer) => {
                    debug!(
                        "✅ SharedServices::new - durable ABAC enforcer wired into VectorOperationsService"
                    );
                    svc.with_abac_enforcer(enforcer.clone())
                }
                None => svc,
            };
            Arc::new(svc)
        };

        info!(
            "✅ SharedServices: VectorOperationsService created successfully - 40-60% performance boost enabled"
        );
        debug!("🔧 SharedServices::new - VectorOperationsService created successfully");

        info!(
            "🧠 SharedServices: Global Cross-Cache Orchestrator registered (budget={}MB)",
            storage_config.cache_size_mb
        );

        // Collection recovery will be handled by StorageEngine::start()
        // SharedServices no longer tries to recover before storage starts
        info!(
            "📋 SharedServices: Collection recovery will be handled by StorageEngine during startup"
        );

        // Placeholder for future assignment service recovery
        // Deferred: Add assignment service recovery after StorageEngine starts

        if false {
            // Disabled recovery code - will be moved to ProximaDB::new
            let recovered_collections = std::collections::HashMap::<
                String,
                crate::storage::metadata::VersionedCollectionMetadata,
            >::new();
            info!(
                "📦 SharedServices: Restoring {} collections to metadata backend",
                recovered_collections.len()
            );

            let collection_count = recovered_collections.len();
            for (collection_id, metadata) in recovered_collections {
                info!(
                    "📝 SharedServices: Restoring collection metadata for {}",
                    collection_id
                );

                // Convert storage metadata to proto collection format
                let collection_config = crate::proto::proximadb_v1::CollectionConfig {
                    name: metadata.name.clone(),
                    embedding_config: None,
                    dimension: metadata.dimension as u32,
                    distance_metric: Some(
                        crate::proto::proximadb_v1::DistanceMetric::Cosine as i32,
                    ), // Default
                    storage_engine: Some(crate::proto::proximadb_v1::StorageEngine::Sst as i32), // Default: SST
                    filterable_columns: vec![],
                    index_configs: vec![],
                    quantization: Some(crate::proto::proximadb_v1::QuantizationConfig {
                        enabled: Some(true), // Quantization enabled by default
                        strategy: Some(
                            crate::proto::proximadb_v1::quantization_config::Strategy::SmartDefaults
                                as i32,
                        ),
                        custom_levels: vec![],
                        enable_progressive_search: Some(true), // Progressive search enabled by default
                        binary_filter_selectivity: Some(0.3),
                        int8_ranking_selectivity: Some(0.1),
                        pq_ranking_selectivity: Some(0.05),
                        training_sample_size: Some(10000),
                        quality_threshold: Some(0.95),
                        enable_adaptive_training: Some(true),
                        optimize_for_storage: Some(false),
                        optimize_for_memory: Some(false),
                        enable_simd_acceleration: Some(true),
                        // NEW: Direct quantization type enables
                        enable_binary: Some(true),
                        enable_int8: Some(true),
                        enable_pq: Some(true),
                        // Product Quantization specific settings
                        pq_segments: Some(8),
                        pq_bits: Some(8),
                        pq_codebooks: Some(0),
                        // Thresholds for progressive search
                        binary_threshold: Some(0.3),
                        int8_threshold: Some(0.1),
                        pq_threshold: Some(0.05),
                        enable_turboquant: Some(false),
                    }),
                    storage_config: None, // VersionedCollectionMetadata doesn't have storage_assignment field
                    primary_index: Some(String::new()),
                    auto_index_selection: Some(false),
                    description: None,
                    tags: vec![],
                    owner: None,
                    embedding_models: vec![], // No embedding models for imported collections
                    // ProximaRecord schema configuration (NEW)
                    record_schema: None,
                    enable_proxima_record: None,
                    text_columns: vec![],
                    text_storage_configs: vec![],
                    enable_dual_use_embeddings: None,
                    canonical_embedding_precision: None,
                    permitted_principals: vec![],
                    // Coarse recovery reconstruction (quantization etc. hardcoded);
                    // routing policy defaults to auto (None).
                    index_policy: None,
                    pax_vector_quant: None,
                };

                let proto_collection = crate::proto::proximadb_v1::Collection {
                    id: format!("recovered-{}", Uuid::new_v4()),
                    config: Some(collection_config),
                    stats: Some(crate::proto::proximadb_v1::CollectionStats {
                        vector_count: metadata.vector_count as i64,
                        index_size_bytes: metadata.total_size_bytes as i64,
                        data_size_bytes: metadata.total_size_bytes as i64,
                    }),
                    created_at: metadata.timestamp as i64,
                    updated_at: metadata.timestamp as i64, // VersionedCollectionMetadata doesn't have updated_at field
                    storage_assignment: None, // VersionedCollectionMetadata doesn't have storage_assignment field
                };

                // Recovery now flows through xCatalog (catalog is the sole store);
                // this disabled branch is retained as a structural placeholder.
                let _ = &proto_collection;
                info!(
                    "✅ SharedServices: Successfully restored collection metadata for {}",
                    collection_id
                );
            }

            info!(
                "✅ SharedServices: Metadata recovery completed - {} collections restored",
                collection_count
            );
        } else {
            info!("📋 SharedServices: No collections found in WAL to restore");
        }

        // ==================================================================================
        // CRITICAL FIX FOR GRAPH API BUG - Ensure Single Shared GraphCollectionService
        // ==================================================================================
        //
        // ROOT CAUSE ANALYSIS:
        //
        // The previous implementation had TWO SEPARATE GraphCollectionService instances:
        // 1. One created by UnifiedHandlers::new() for REST/gRPC graph collection endpoints
        // 2. One created by GraphOperationsService::new() for node/edge operations
        //
        // This caused graph collections created via REST API to be INVISIBLE to graph
        // operations because they were stored in different instances.
        //
        // SOLUTION:
        //
        // Create a SINGLE GraphCollectionService instance here and pass it to BOTH:
        // - GraphOperationsService (via new_with_collection_service)
        // - UnifiedHandlers and query orchestration layers (via extracted graph contracts)
        //
        // This ensures ALL graph endpoints and operations share the same state.
        // ==================================================================================

        debug!(
            "🔧 SharedServices::new - Creating SHARED GraphCollectionService instance with auto-recovery..."
        );
        let graph_metadata_url =
            join_storage_url(&storage_config.metadata_url, "graph_collections.json");
        let graph_collection_service = match if graph_metadata_url.starts_with("file://") {
            crate::services::GraphCollectionService::new_with_recovery_at(std::path::PathBuf::from(
                graph_metadata_url.trim_start_matches("file://"),
            ))
            .await
        } else {
            crate::services::GraphCollectionService::new_with_recovery_at_url(
                graph_metadata_url.clone(),
                filesystem_factory.clone(),
            )
            .await
        } {
            Ok(svc) => Arc::new(svc),
            Err(e) => {
                warn!(
                    "Failed to create GraphCollectionService with recovery at {}: {}. Using non-persistent service.",
                    graph_metadata_url, e
                );
                Arc::new(crate::services::GraphCollectionService::new())
            }
        };
        debug!(
            "✅ SharedServices::new - Shared GraphCollectionService created (with auto-recovery)"
        );

        // T2.3 / TD-066 production wiring: open the canonical WAL appender
        // ONCE here so it can be shared between the graph checkpoint emission
        // path (`GraphOperationsService::flush_wal`) and the pgwire direct
        // record write path (constructed in `multi_server.rs`). Sharing the
        // same `FramedTableWalAppender` instance is required for correctness:
        // two independent `open()` calls on the same WAL file would each
        // initialize their own `next_sequence: AtomicU64`, leading to
        // duplicate sequence numbers in the persisted log and silent
        // recovery corruption.
        //
        // When `opt_config` is `None` (test paths that don't supply a full
        // Config), skip the appender entirely — graph falls back to its
        // tracing-only behavior and pgwire (if enabled) opens its own
        // appender locally as it did before.
        let canonical_wal_appender: Option<Arc<crate::services::FramedTableWalAppender>> =
            if let Some(cfg) = opt_config {
                let wal_path = cfg
                    .server
                    .data_dir
                    .join("pgwire")
                    .join("canonical-records.wal");
                match crate::services::FramedTableWalAppender::open(&wal_path).await {
                    Ok(appender) => {
                        info!(
                            "✅ SharedServices: canonical WAL appender opened at {} (shared by graph checkpoint emission + pgwire direct writes)",
                            wal_path.display()
                        );
                        Some(Arc::new(appender))
                    }
                    Err(e) => {
                        warn!(
                            "SharedServices: failed to open canonical WAL at {}: {}. Graph flush_wal will fall back to tracing-only and pgwire (if enabled) will open its own appender.",
                            wal_path.display(),
                            e
                        );
                        None
                    }
                }
            } else {
                debug!(
                    "SharedServices: opt_config is None; skipping canonical WAL appender setup (test path?)"
                );
                None
            };

        // Cross-surface unification: build the canonical WAL-backed record store ONCE and
        // WAL-recover it here, then share it across every surface (REST/gRPC DmlService below
        // + pgwire direct-write path). One store ⇒ one authoritative relational state and one
        // CDC change-feed, regardless of which protocol wrote the data.
        let canonical_record_store: Option<
            Arc<crate::services::record_store::DirectWalTableRecordStore>,
        > = if let Some(appender) = canonical_wal_appender.clone() {
            let store = Arc::new(
                crate::services::record_store::DirectWalTableRecordStore::new_partitioned(
                    appender.clone(),
                ),
            );
            match appender.read_entries().await {
                Ok(entries) => {
                    if let Err(e) = store.replay_wal_entries(entries).await {
                        warn!("SharedServices: canonical record-store WAL replay failed: {e}");
                    }
                }
                Err(e) => {
                    warn!("SharedServices: reading canonical WAL for shared store failed: {e}")
                }
            }
            info!(
                "✅ SharedServices: shared canonical record store built + recovered (unifies REST/gRPC + pgwire relational state)"
            );
            Some(store)
        } else {
            None
        };

        // Experimental transactional ledger store (ADR-071 / TD-LEDGER-1): a node-level durable
        // ledger shared with the gRPC `ProximaLedgerService`. Its WAL lives on LOCAL disk (ADR-069:
        // the per-write log belongs on a reattachable local volume, not object storage); when the
        // metadata store is object-backed (no `file://`), the ledger WAL falls back to a local
        // `data/ledger` directory. One store per node — tenants are namespaced inside the keys.
        #[cfg(feature = "experimental-ledger")]
        let ledger_store = {
            let ledger_url = join_storage_url(&storage_config.metadata_url, "ledger/ledger.wal");
            let ledger_path = match ledger_url.strip_prefix("file://") {
                Some(local) => std::path::PathBuf::from(local),
                None => std::path::PathBuf::from("data/ledger/ledger.wal"),
            };
            if let Some(parent) = ledger_path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating ledger WAL dir {}", parent.display()))?;
            }
            let durable = proximadb_ledger::DurableLedger::open(
                &ledger_path,
                proximadb_ledger::SyncPolicy::PerOp,
            )
            .with_context(|| format!("opening ledger WAL at {}", ledger_path.display()))?;
            info!(
                "✅ SharedServices: experimental ledger store opened at {}",
                ledger_path.display()
            );
            Arc::new(proximadb_ledger::LedgerService::new(durable))
        };

        // Timed TTL reclaim (ADR-071 / TD-LEDGER-1, invariant C2): sweep expired ledger leases on a
        // fixed cadence so a crashed reserver's held capacity is freed even when no request touches
        // the scope. Server profile only; the sweep is idempotent and O(held leases) — cheap.
        #[cfg(feature = "experimental-ledger")]
        if profile.is_server() {
            let sweeper = ledger_store.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(std::time::Duration::from_secs(60));
                loop {
                    tick.tick().await;
                    let now_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos().min(i64::MAX as u128) as i64)
                        .unwrap_or(0);
                    let reclaimed = sweeper.reclaim_expired(now_ns);
                    if reclaimed > 0 {
                        tracing::debug!("ledger sweeper reclaimed {reclaimed} expired lease(s)");
                    }
                }
            });
        }

        // Create GraphOperationsService for native graph database operations
        // IMPORTANT: Pass the shared GraphCollectionService instance
        debug!(
            "🔧 SharedServices::new - Creating GraphOperationsService with SHARED collection service..."
        );
        // ALWAYS use new_with_collection_service to ensure shared GraphCollectionService
        // Even if config is provided, we must share the collection service
        // (Config-specific settings can be applied later if needed)
        let mut graph_service_inst =
            crate::graph::GraphOperationsService::new_with_collection_service(
                graph_collection_service.clone(),
            );
        // T2.3 / TD-066: inject the shared appender so flush_wal persists
        // canonical checkpoints to disk. Also inject the WAL path so the
        // graph engine factory can plumb it into ORION's persistence
        // layer for the read-side recovery hook (TD-066 (c) Part 1).
        if let Some(appender) = canonical_wal_appender.as_ref() {
            graph_service_inst =
                graph_service_inst
                    .with_canonical_wal_appender(appender.clone()
                        as Arc<dyn crate::services::record_store::TableWalAppender>);
            graph_service_inst =
                graph_service_inst.with_canonical_wal_path(appender.path().to_path_buf());
        }
        // Wire the storage root so graph engines persist under the same base path as vectors
        let graph_storage_url = storage_config
            .storage_locations
            .first()
            .map(|loc| loc.url.clone())
            .unwrap_or_else(|| storage_config.metadata_url.clone());
        graph_service_inst.set_base_storage_url(graph_storage_url.clone());

        // TD-168 Phase 2: when the cold-payload tier is ON, back the graph's
        // canonical record store with a Cool-tiered object store so node/edge
        // payloads are durable off-RAM and the cold-fetch read path (#446) can
        // materialize them on a cache miss. Graph-only by construction, so every
        // object is Cool with no risk of mis-tiering hot relational data.
        // Default-OFF: with the gate unset nothing is constructed, the canonical
        // record store stays None, and the all-RAM path is unchanged.
        if crate::graph::service::cold_payloads_enabled() {
            let tier = proximadb_storage_filesystem_types::ObjectAccessTier::Cool;
            // #52: optionally back the cold tier with the BATCHED segment store
            // (`ColdGraphSegmentStore`) instead of the per-record store — far fewer
            // object PUTs. It BUFFERS in RAM, so it is crash-safe only with the
            // recovery re-population backstop (PR #520, gated separately and
            // default-OFF) plus the checkpoint/shutdown flush hook. If that backstop
            // is OFF we REFUSE to wire it and fall back to the durable-on-write
            // `ColdGraphRecordStore` — fail-safe, never an unprotected buffered store.
            //
            // Phase-1 precondition: recovery re-population rebuilds the cold store via
            // `engine.get_all_nodes()/get_all_edges()`, valid only while the ORION
            // engine is FULL-RESIDENT and snapshots full payloads. True payload-offload
            // cold-tiering (engine evicts payload) is a later phase that needs the
            // segment store to be a durable authority (segment-tail index rebuild), not
            // a projection.
            let mut use_segment_store = crate::graph::service::segment_store_enabled();
            if use_segment_store
                && !crate::storage::persistence::write_ahead_log::wal_operations::canonical_recovery_repopulate_enabled()
            {
                warn!(
                    "SharedServices: PROXIMADB_GRAPH_SEGMENT_STORE is set but PROXIMADB_GRAPH_CANONICAL_RECOVERY is OFF — the buffered segment store has no crash-recovery backstop; refusing to wire it and falling back to the durable-on-write ColdGraphRecordStore. Set PROXIMADB_GRAPH_CANONICAL_RECOVERY=1 to enable the segment store."
                );
                use_segment_store = false;
            }

            if use_segment_store {
                match crate::graph::ColdGraphSegmentStore::from_storage_root(
                    &graph_storage_url,
                    tier,
                )
                .await
                {
                    Ok(seg_store) => {
                        graph_service_inst =
                            graph_service_inst.with_canonical_record_store(Arc::new(seg_store));
                        info!(
                            "✅ SharedServices: graph cold-payload tier ON (SEGMENT store) — canonical node/edge payloads batched → Cool object storage ({graph_storage_url}); crash-recovery backstop enabled"
                        );
                    }
                    Err(e) => {
                        warn!(
                            "SharedServices: segment store requested (PROXIMADB_GRAPH_SEGMENT_STORE) but init failed for `{graph_storage_url}`: {e}; falling back to the all-RAM path"
                        );
                    }
                }
            } else {
                match crate::graph::ColdGraphRecordStore::from_storage_root(
                    &graph_storage_url,
                    tier,
                ) {
                    Ok(cold_store) => {
                        graph_service_inst =
                            graph_service_inst.with_canonical_record_store(Arc::new(cold_store));
                        info!(
                            "✅ SharedServices: graph cold-payload tier ON — canonical node/edge payloads → Cool object storage ({graph_storage_url})"
                        );
                    }
                    Err(e) => {
                        warn!(
                            "SharedServices: graph cold-payload tier requested (PROXIMADB_GRAPH_COLD_PAYLOADS) but cold store init failed for `{graph_storage_url}`: {e}; falling back to the all-RAM path"
                        );
                    }
                }
            }
        }

        // Create a simple file-backed metrics updater under data_root/metrics
        let filesystem_factory =
            Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);
        let metrics_config = MetricsConfig {
            enabled: true,
            collection_partitions: 16,
            storage_path: join_storage_url(&storage_config.metadata_url, "metrics"),
            flush_interval_seconds: 60,
            retention_days: 7,
            parallel_scan_threshold: 1000,
            sparsity_threshold: 0.5,
            quantization_size_threshold: 1024 * 1024, // 1MB
            max_memory_mb: 512,
            snapshot_interval_seconds: 300, // 5 minutes
        };
        // Server-only: the metrics persistence layer + billing/telemetry publisher
        // back a chargeback/scrape surface with no in-process consumer. The fused
        // embedded core skips them and leaves the graph engine's updater unset
        // (co-design tenets 1 & 5; the `metrics_updater` field is already optional).
        let metrics_updater: Option<Arc<dyn crate::metrics::InternalMetricsUpdater + 'static>> =
            if profile.is_server() {
                let metrics_store = Arc::new(
                    crate::metrics::store::MetricsPersistenceLayer::new(
                        filesystem_factory.clone(),
                        metrics_config,
                    )
                    .await?,
                );
                let updater: Arc<dyn crate::metrics::InternalMetricsUpdater + 'static> = Arc::new(
                    crate::metrics::updater::MetricsUpdateService::new(metrics_store.clone()),
                );
                graph_service_inst.set_metrics_updater(updater.clone());
                debug!("📈 GraphOperationsService metrics updater wired");
                Some(updater)
            } else {
                None
            };
        let graph_service = Arc::new(graph_service_inst);
        debug!(
            "✅ SharedServices::new - GraphOperationsService created with shared collection service"
        );

        // Create DocumentService (moved up for UnifiedHandlers)
        debug!("🔧 SharedServices::new - Creating DocumentService for document queries...");
        let document_base_path = join_storage_url(&storage_config.metadata_url, "");
        // TD-DOC-RETIRE-1 P2 rewires this to the canonical constructor
        // (with_canonical_record_store_and_wal); the deprecated call is intentional until then.
        #[allow(deprecated)]
        let document_service = match DocumentService::new_with_wal(
            sst_engine_for_documents,
            &document_base_path,
        )
        .await
        {
            Ok(service) => Arc::new(service),
            Err(e) => {
                warn!(
                    "Failed to create WAL-backed DocumentService: {}. Falling back to in-memory WAL-less service.",
                    e
                );
                Arc::new(DocumentService::new(
                    vector_operations_service.unified_engine(),
                ))
            }
        };

        // Create ObservabilityService (moved up for UnifiedHandlers)
        debug!(
            "🔧 SharedServices::new - Creating ObservabilityQueryEngine for observability queries..."
        );
        let observability_base_path = join_storage_url(&storage_config.metadata_url, "");
        let observability_storage = match ObservabilityStorage::new_with_wal(
            &observability_base_path,
        )
        .await
        {
            Ok(storage) => Arc::new(storage),
            Err(e) => {
                warn!(
                    "Failed to create WAL-backed ObservabilityStorage: {}. Falling back to non-WAL storage.",
                    e
                );
                Arc::new(ObservabilityStorage::new(&observability_base_path))
            }
        };
        let observability_service = Arc::new(
            crate::observability::ObservabilityService::new(observability_storage.clone()).await?,
        );
        let observability_query_engine =
            Arc::new(ObservabilityQueryEngine::new(observability_storage.clone()));

        // Create EventLogEngine for persistent audit trails (TD-050 Phase 5)
        debug!("🔧 SharedServices::new - Creating EventLogEngine for audit trails...");
        let event_log_base_path = join_storage_url(&storage_config.metadata_url, "auditlog");
        let event_log_config = crate::storage::engines::eventlog::EventLogConfig {
            base_dir: event_log_base_path,
            ..Default::default()
        };
        let event_log_filesystem = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                filesystem_factory.get_filesystem(&storage_config.metadata_url)?,
                "auditlog".to_string(),
                "eventlog".to_string(),
            ),
        );
        // `open` (not `new`) — recovers the sequence counter and index from what
        // is already persisted. `new` alone would restart the counter at 0 and
        // silently overwrite prior events (TD-EVENTLOG-1).
        let event_log = match crate::storage::engines::eventlog::EventLogEngine::open(
            event_log_config,
            event_log_filesystem,
        )
        .await
        {
            Ok(engine) => Some(Arc::new(engine)),
            Err(e) => {
                warn!("Failed to create EventLogEngine for audit trails: {}", e);
                None
            }
        };

        // TD-TS-1: initialise the process-global time-series service over the native
        // TST engine, rooted under the data dir. Non-fatal on failure (surface stays
        // unavailable rather than blocking bootstrap).
        {
            // The TST engine is local-filesystem-native (PathBuf WAL/segments);
            // it cannot yet run over an object store. On an object-store
            // metadata_url the time-series surface stays on local disk
            // (non-durable across VM loss — TD-OBJSTORE-1 deferred item)
            // instead of mangling the URL into a local path.
            let ts_base_path = match local_storage_path(&storage_config.metadata_url) {
                Some(local) => local.join("timeseries"),
                None => {
                    let fallback = std::env::temp_dir().join("proximadb").join("timeseries");
                    warn!(
                        "TimeSeriesService does not support object-store storage yet \
                         (metadata_url={}); falling back to LOCAL, NON-DURABLE {} \
                         (TD-OBJSTORE-1)",
                        storage_config.metadata_url,
                        fallback.display()
                    );
                    fallback
                }
            };
            if let Err(e) =
                crate::services::timeseries_service::init_timeseries_service(ts_base_path)
            {
                warn!("Failed to initialise TimeSeriesService: {}", e);
            }
        }

        // Derive extracted graph capability views once here so query/orchestration
        // layers depend on explicit contracts rather than the full concrete service.
        let graph_query_service = graph_service.clone();
        let graph_execution_service = graph_service.clone();

        // Create unified handlers with SHARED graph services
        // IMPORTANT: Pass the pre-created GraphCollectionService and graph execution service
        // to ensure ALL graph endpoints and operations share the same state
        // TD-104 S3-f: build the shared `RecordOpsService` once here (formerly
        // constructed inside the deleted root `UnifiedHandlers::new`). Every
        // former `request_handlers.record_ops()` site clones this same `Arc`.
        // The other services the root handler aggregated (document / graph /
        // observability / event_log / graph_collection) now live directly on
        // `SharedServices`.
        debug!("🔧 SharedServices::new - Creating shared RecordOpsService...");
        let record_ops = Arc::new(
            crate::api_handlers::record_ops_service::RecordOpsService::new(
                collection_service.clone(),
                vector_operations_service.clone(),
            ),
        );
        debug!("✅ SharedServices::new - Shared RecordOpsService created");
        // NOTE: the former `set_hybrid_runtime(cfg.hybrid)` on the root handler is
        // dropped here — `resolve_hybrid_static` (its only consumer) was
        // root-internal and is deleted with the root handler; the runtime handler
        // never read it, so the config was already dead wiring.

        // ADR-009 document convergence: wire the single shared DocumentService onto the same
        // tenant-scoped record/vector route REST v2 uses (via the shared RecordOpsService
        // built above). Default-OFF per-collection gate — this only makes the route
        // *available*; `doc_canonical_vector_enabled` decides per collection at call time. With
        // this, gRPC/DocumentService and REST v2 writes converge on one store (no store-split).
        document_service
            .set_record_route(record_ops.clone() as Arc<dyn proximadb_runtime::RecordRoutePort>);
        debug!(
            "✅ SharedServices::new - DocumentService wired to canonical record route (ADR-009, gate default-OFF)"
        );

        // ADR-055 P-DFSource: publish the document service as a process singleton so the DataFusion
        // `documents(collection)` table function (registered per SessionContext) can read it. Mirrors
        // the timeseries_service wiring; idempotent (first wins).
        crate::services::document_service::set_document_service(document_service.clone());

        // ==================================================================================
        // Create UnifiedQueryFacade - single entry point for all query types
        // This consolidates the 5 parallel query paths into a single unified interface
        // ==================================================================================
        debug!("🔧 SharedServices::new - Creating UnifiedQueryFacade with real strategies...");

        // Create VectorSearchStrategy wrapping VectorOperationsService.
        // Task #76 consumer migration: VectorSearchStrategy now takes
        // Arc<dyn CollectionPort> instead of Arc<CollectionService>.
        // Coerce the existing concrete service to the port trait object
        // (the same SharedServices collection_port field uses the same
        // coercion at the field-init site below).
        let vector_strategy: Arc<dyn crate::query::facade::QueryStrategy> =
            Arc::new(VectorSearchStrategy::new(
                vector_operations_service.clone(),
                collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>,
            ));

        // Create GraphStrategy wrapping the extracted graph query contract
        let graph_strategy: Arc<dyn crate::query::facade::QueryStrategy> =
            Arc::new(GraphStrategy::new(graph_query_service.clone()));

        // Create DocumentStrategy wrapping DocumentService for JSON document queries
        // DocumentService provides MongoDB-like document operations (CRUD, indexing, queries)
        let document_strategy: Arc<dyn crate::query::facade::QueryStrategy> =
            Arc::new(DocumentStrategy::new(document_service.clone()));
        debug!("✅ SharedServices::new - DocumentStrategy created for document queries");

        // Create ObservabilityStrategy wrapping ObservabilityQueryEngine for logs/metrics/traces
        // This enables unified query interface for observability data
        const QUERY_TELEMETRY_NAMESPACE: &str = "_proximadb_query";
        let telemetry_namespace_exists = observability_service
            .list_namespaces()
            .await
            .into_iter()
            .any(|namespace| namespace.name == QUERY_TELEMETRY_NAMESPACE);
        if !telemetry_namespace_exists {
            let telemetry_config = crate::proto::proximadb_v1::ObservabilityNamespaceConfig {
                name: QUERY_TELEMETRY_NAMESPACE.to_string(),
                retention: Some(crate::proto::proximadb_v1::RetentionConfig {
                    hot_retention_hours: 24,
                    warm_retention_days: 7,
                    cold_retention_days: 30,
                    archive_retention_days: 90,
                }),
                ingestion: None,
                alert_rules: Vec::new(),
                access: None,
            };
            if let Err(error) = observability_service
                .create_namespace(telemetry_config)
                .await
            {
                warn!(
                    "Failed to create internal query telemetry namespace '{}': {}",
                    QUERY_TELEMETRY_NAMESPACE, error
                );
            }
        }
        crate::query::utils::metrics::configure_query_telemetry(
            observability_service.clone(),
            QUERY_TELEMETRY_NAMESPACE,
        );

        let observability_strategy: Arc<dyn crate::query::facade::QueryStrategy> =
            Arc::new(ObservabilityStrategy::new(observability_query_engine));
        debug!(
            "✅ SharedServices::new - ObservabilityStrategy created for logs/metrics/traces queries"
        );

        // Create MultiModelStorageFacade for federated queries and wire the live stores
        debug!(
            "🔧 SharedServices::new - Creating MultiModelStorageFacade for federated queries..."
        );
        let vector_store = Arc::new({
            let vs = crate::storage::multimodel::VectorStore::with_engine(
                vector_operations_service.unified_engine(),
            );
            #[cfg(feature = "axis")]
            let vs = if axis_runtime_enabled {
                vs.with_index_manager(axis_manager.clone())
            } else {
                vs
            };
            vs
        });
        let graph_store = Arc::new(
            crate::storage::multimodel::GraphStore::new(Default::default())
                .with_service(graph_service.clone()),
        );
        let document_store = Arc::new(
            crate::storage::multimodel::DocumentStore::new(Default::default())
                .with_service(document_service.clone()),
        );
        let obs_base_path = join_storage_url(&storage_config.metadata_url, "");
        let observability_store = Arc::new(
            crate::storage::multimodel::ObservabilityStore::new(
                crate::storage::multimodel::stores::observability_store::ObservabilityStoreConfig {
                    base_path: obs_base_path,
                    ..Default::default()
                },
            )
            .with_service(observability_service.clone()),
        );
        let multimodal_storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store)
                .with_document_store(document_store)
                .with_observability_store(observability_store),
        );
        debug!("✅ SharedServices::new - MultiModelStorageFacade created and wired");

        // T3.2 Slice 1: shared full-text index map for hybrid retrieval.
        // Hoisted ahead of the FederatedQueryContext so the rank-pipeline
        // singleton can use the same Arc the SharedServices field holds.
        let fulltext_indexes: crate::network::hybrid_search::HybridFullTextIndexMap =
            Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));

        // R-7c.3 production wiring: construct the durable rank-profile store,
        // the production hybrid backend, the rank metrics handle, and the
        // singleton `RankServices` that REST / gRPC / Arrow Flight share.
        let (rank_services, rank_profile_store) = build_rank_services(
            vector_operations_service.clone() as Arc<dyn proximadb_runtime::VectorOpsPort>,
            fulltext_indexes.clone(),
            canonical_wal_appender.clone(),
            opt_config,
        )
        .await;
        info!(
            "✅ SharedServices: RankServices ready (profile_count={}, metrics=on)",
            rank_services.profile_registry.len()
        );

        // UDF F5 (5b-ii) boot recovery: build the durable function catalog over
        // the SAME canonical WAL spine the rank-profile store uses, replay every
        // persisted `CREATE FUNCTION`, and re-register each into the shared
        // `proximadb_functions::builtins()` registry so user functions survive a
        // restart on both engines. Threaded into every per-connection
        // `DdlService` (pgwire) so new `CREATE FUNCTION` statements persist here.
        let function_store = build_function_store(canonical_wal_appender.clone()).await;

        // Create FederatedQueryContext for SQL with multi-model extensions
        debug!("🔧 SharedServices::new - Creating FederatedQueryContext...");
        let federated_context = Arc::new(
            FederatedQueryContext::new(multimodal_storage)
                .with_collection_port(
                    collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>
                )
                .with_vector_operations(vector_operations_service.clone())
                .with_rank_services(rank_services.clone()),
        );
        debug!("✅ SharedServices::new - FederatedQueryContext created");

        // Create SqlStrategy wrapping FederatedQueryContext
        let sql_strategy: Arc<dyn crate::query::facade::QueryStrategy> =
            Arc::new(SqlStrategy::new(federated_context));

        // Create ColumnarStrategy for analytical queries (M2 Dual Columnar Execution)
        // This strategy handles SQL queries with aggregations, GROUP BY, DISTINCT
        // by routing them through Arrow/Parquet columnar providers
        let columnar_strategy: Arc<dyn crate::query::facade::QueryStrategy> =
            Arc::new(ColumnarStrategy::new());
        debug!("✅ SharedServices::new - ColumnarStrategy created for analytical queries");

        // Create DistributedQueryStrategy for cluster-aware federated execution.
        // This is only selected when the execution path is explicitly forced to "distributed".
        let distributed_strategy: Arc<dyn crate::query::facade::QueryStrategy> = Arc::new(
            DistributedQueryStrategy::new(
                "local-node".to_string(),
                DistributedStrategyConfig::default(),
            )
            .with_vector_ops(vector_operations_service.clone())
            .with_document_service(document_service.clone())
            .with_graph_service(graph_query_service.clone())
            .with_observability_service(observability_service.clone()),
        );
        debug!(
            "✅ SharedServices::new - DistributedQueryStrategy created for forced distributed execution"
        );

        // Build the unified facade with all strategies
        // Priority order: vector (100) > graph (75) > document (70) > observability (60) > columnar (50) > sql (25)
        // Distributed strategy is force-path only and will not be selected automatically.
        let strategies = vec![
            vector_strategy,
            graph_strategy,
            document_strategy,
            observability_strategy,
            columnar_strategy,
            distributed_strategy,
            sql_strategy,
        ];
        let query_facade = Arc::new(UnifiedQueryFacade::new(strategies, FacadeConfig::default()));

        info!(
            "✅ SharedServices: UnifiedQueryFacade created with 7 strategies (vector, graph, document, observability, columnar, distributed, sql)"
        );

        // Build the DmlService first so it can be shared by both the ROOT handler
        // (legacy EXPLAIN routing) and the QueryFacadeAdapter (port-path EXPLAIN
        // routing). EXPLAIN INSERT … SELECT queries arriving on the ExecuteSql RPC
        // are detected in execute_sql_v1 / the adapter and dispatched here instead
        // of the legacy SQL frontend.
        // Use the SHARED canonical store so REST/gRPC relational DML/reads/EXPLAIN and the
        // CDC change-feed operate on the SAME state pgwire writes to. Fall back to the
        // vector-compatibility store only when no canonical store exists (test paths).
        // A7: activate cross-pod DML locking. The durable lease lives on the
        // SAME object store as the catalog (`storage_config.metadata_url`), so
        // table-level DML contention is real across pods (only one pod's
        // `DmlLockService` can hold the CAS-backed lease). Fail-open — no locks
        // — if the store can't be opened (test/embedded paths); never block
        // bootstrap.
        //
        // Registry/pod-id unification (lease ↔ write-gate): the lease manager
        // MUST reconcile into the SAME `primary_pod_registry` the write gates
        // consult (built above, wired into AppState/gRPC/Flight gates below) and
        // use the SAME `self_pod_id` those gates compare against
        // (`resolve_self_pod_id(None)`). Previously the manager owned a throwaway
        // `PrimaryPodRegistry::new()` and a `pod-{pid}` id, so the renew loop's
        // `reconcile`→`assign` (step-down on lease loss) updated a registry
        // nobody read, and the assigned owner id never matched the gate's id —
        // a displaced pod kept seeing `Allow` and kept writing (split brain).
        let (dml_lock_service, lease_manager_for_writes) = {
            use crate::cluster::partition_lease::{
                DmlLockService, PartitionLeaseManager, PartitionLeaseStore,
            };
            match PartitionLeaseStore::from_url(&storage_config.metadata_url, "_operator/leases") {
                Ok(store) => {
                    let pod_id = crate::cluster::primary_pod_registry::resolve_self_pod_id(None);
                    let manager = Arc::new(PartitionLeaseManager::new(
                        Arc::new(store),
                        primary_pod_registry.clone(),
                        pod_id.clone(),
                        10_000,
                    ));
                    crate::cluster::partition_lease::install_global_partition_lease_manager(
                        manager.clone(),
                    );
                    // P1a: keep held leases warm. Without this the renew loop
                    // never runs in production (it was only spawned in tests),
                    // so held leases lapse after the 10s TTL and the
                    // leaseholder model the write-gate (421) + DML locks rely
                    // on silently degrades. Fire-and-forget for the process
                    // lifetime (matches SharedServices' existing background-task
                    // pattern); interval ≤ lease_ms/2 so a lease never lapses
                    // between renewals.
                    // `spawn_renew_loop` tokio::spawns the loop and returns its
                    // JoinHandle; drop it as a temporary so the task runs detached
                    // (fire-and-forget) without tripping clippy::let_underscore_future.
                    manager
                        .clone()
                        .spawn_renew_loop(std::time::Duration::from_millis(5_000));

                    // TD-MANIFEST-1: cap the lease manifest log. Manifests are
                    // append-only and were never pruned, so a long-lived lease grew
                    // ~48k objects (562 MB) and made every `latest_version()` an O(n)
                    // full `list` that pinned a CPU core (and on a cloud store is a
                    // paginated LIST that gets slower/costlier as n grows). A
                    // low-priority prune of the stale tail caps n — itself the
                    // read-path fix, since the tip scan stays cheap once bounded.
                    // Default ON with conservative knobs; set
                    // PROXIMADB_LEASE_MANIFEST_RETENTION=0 to disable.
                    if lease_manifest_retention_enabled() {
                        manager.clone().spawn_prune_loop(
                            std::time::Duration::from_secs(lease_manifest_prune_interval_secs()),
                            lease_manifest_keep_k(),
                            std::time::Duration::from_secs(lease_manifest_min_age_secs()),
                        );
                    }
                    let lock_service = Arc::new(DmlLockService::new(manager.clone(), pod_id));
                    let _ = lock_service.spawn_reconciliation_loop(5_000);
                    // Surface the manager so RecordOpsService can lease-on-write:
                    // acquire/confirm the collection lease before each vector-record
                    // write so the shared primary-pod registry the gates consult is
                    // durably backed (not empty-after-restart → wrong-pod write).
                    (Some(lock_service), Some(manager))
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        url = %storage_config.metadata_url,
                        "DML lock service disabled (lease store unavailable); \
                         DML writes run lock-free (fail-open)"
                    );
                    (None, None)
                }
            }
        };

        // A6: build the storage-write fence adapter over the SAME lease manager
        // before it is moved into RecordOpsService below, so the shutdown flush
        // path and the network write-gates share one ownership view. `None` ⇒ lease
        // store unavailable ⇒ fence fails open. Default-OFF until enforced.
        let storage_write_fence: Option<Arc<dyn crate::storage::write_fence::StorageWriteFence>> =
            lease_manager_for_writes.as_ref().map(|manager| {
                Arc::new(
                    crate::network::storage_write_fence::LeaseStorageWriteFence::new(
                        manager.clone(),
                    ),
                ) as Arc<dyn crate::storage::write_fence::StorageWriteFence>
            });

        // F5 / TD-OLTP-WIRING-1: open the ONE process-shared fenced
        // ConditionalKeyStore (durable at <data_dir>/oltp-cks.wal; WAL-replayed on
        // restart), threaded into this gRPC/REST `base_dml` and into pgwire via
        // `multi_server`. Feature-gated (`oltp-integrity`) AND requires a
        // `data_dir` — an in-memory store would silently lose uniqueness across a
        // restart, worse than the honest legacy probe. `None` otherwise ⇒ default
        // builds and embedded/ephemeral paths are byte-for-byte unchanged.
        let conditional_key_store: Option<Arc<dyn proximadb_storage_ports::ConditionalKeyStore>> =
            Self::open_shared_conditional_key_store(opt_config);

        let base_dml = match canonical_record_store.clone() {
            Some(store) => DmlService::with_direct_record_storage(
                catalog_manager.clone(),
                vector_operations_service.clone(),
                store,
            ),
            None => DmlService::new(catalog_manager.clone(), vector_operations_service.clone()),
        };
        let base_dml = match &conditional_key_store {
            Some(cks) => base_dml.with_conditional_key_store(cks.clone()),
            None => base_dml,
        };
        // TD-ABAC-2 (Phase 5b): construct the durable ABAC enforcer (authority +
        // policy binding store, both at <data_dir>/abac/) and DI it into the
        // DML read funnel. Fully behind `abac-policy` (default-OFF) ⇒ default
        // builds are byte-for-byte unchanged; `None` ⇒ no enforcement (status quo).
        #[cfg(feature = "abac-policy")]
        let base_dml = match &abac_enforcer {
            Some(enforcer) => {
                debug!("✅ SharedServices::new - durable ABAC enforcer wired into DmlService");
                base_dml.with_abac_enforcer(enforcer.clone())
            }
            None => base_dml,
        };
        let dml_service_for_grpc = Arc::new(match dml_lock_service {
            Some(lock_service) => base_dml.with_dml_lock_service(lock_service),
            None => base_dml,
        });

        // TD-135: one DdlService Arc shared by BOTH the adapter (gRPC port path →
        // adapter.execute_sql) and the RecordOpsService (REST write path), so DDL
        // writes over either surface address the same catalog state and execute
        // tenant-scoped.
        // Wire the write-lease authority into DDL so collection/table-scoped DDL
        // fast-fails misrouted writes (ADR-032): the primary-pod registry + pod id
        // for in-memory routing, and — when the lease system is on — the SAME
        // PartitionLeaseManager the DML write-gate uses (`lease_manager_for_writes`),
        // so DDL and DML share one ownership view.
        let mut ddl = crate::services::DdlService::new(catalog_manager.clone())
            .with_record_store(dml_service_for_grpc.record_store())
            .with_primary_pod_registry(primary_pod_registry.clone())
            .with_self_pod_id(crate::cluster::primary_pod_registry::resolve_self_pod_id(
                None,
            ));
        if let Some(manager) = &lease_manager_for_writes {
            ddl = ddl.with_partition_lease_manager(manager.clone());
        }
        let ddl_service = std::sync::Arc::new(ddl);
        // Wire QueryFacadeAdapter onto the runtime handler for unified SQL routing.
        // This enables SQL queries to flow through the facade when the
        // unified-facade-routing feature is enabled. The adapter carries the
        // DmlService (EXPLAIN `<DML>`) AND the DdlService (relational DDL) so the
        // port path (runtime handler → adapter.execute_sql) reproduces the former
        // ROOT SQL behavior (TD-104 / seam S1, single SQL authority).
        let query_adapter = Arc::new(
            QueryFacadeAdapter::new(query_facade.clone())
                .with_dml_service(dml_service_for_grpc.clone())
                .with_ddl_service(ddl_service.clone()),
        );
        // TD-104 S3-f: DML/DDL services and the lease manager are wired onto the
        // shared RecordOpsService (their real owner — the root handler only
        // forwarded these setters). The root's `set_query_adapter` is dropped: it
        // stored the adapter in a field read only inside the (deleted) root, and
        // the runtime handler already receives `query_adapter` at construction
        // below.
        record_ops.set_dml_service(dml_service_for_grpc);
        debug!("✅ SharedServices::new - DmlService wired to RecordOpsService for EXPLAIN routing");

        // Lease-on-write: give RecordOpsService the durable lease manager so the
        // vector-record write path acquires/confirms the collection lease and the
        // shared registry the network gates consult reflects ground truth after
        // restart/partition (Scenario-1 routing truth). Absent → fail-open.
        if let Some(lease_manager) = &lease_manager_for_writes {
            record_ops.set_lease_manager(lease_manager.clone());
            debug!(
                "✅ SharedServices::new - PartitionLeaseManager wired to RecordOpsService (lease-on-write)"
            );
        }

        // TD-135 note: the shared DdlService drives relational DDL through the
        // runtime handler's QueryFacadeAdapter (wired above via
        // `with_ddl_service`); the record write path itself does not need it.

        // Build a port-backed runtime handler for collection/vector REST routes.
        // Uses trait objects so API routes are decoupled from root-crate concrete services.
        let runtime_api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort> =
            Arc::new(proximadb_runtime::UnifiedHandlers::new(
                collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>,
                vector_operations_service.clone() as Arc<dyn proximadb_runtime::VectorOpsPort>,
                Some(query_adapter.clone() as Arc<dyn proximadb_runtime::QueryAdapterPort>),
            ));
        debug!("✅ SharedServices::new - Port-backed runtime API handlers created");

        // Phase 8 (F1) — Continuous Discovery loop. The snapshot-publish
        // coordinator and discovery service share the catalog manager; the
        // background executor is spawned for the process lifetime. The shutdown
        // sender is intentionally leaked (matching the always-on
        // start_axis_consumer maintenance pattern): dropping it would make the
        // executor's `shutdown.changed()` return Err and exit immediately.
        // Registry is durable below metadata_url when a full config is present;
        // embedded/test wiring without one remains in-memory.
        let snapshot_coordinator = Arc::new(
            crate::services::snapshot::SnapshotPublishCoordinator::new(catalog_manager.clone()),
        );
        // Phase 8 (F1): per-collection discovery-job registry. Durable when a
        // full config is present (jobs + states survive restart). Object-store
        // deployments route the sidecar through FileSystem below metadata_url;
        // embedded/test harnesses without a config remain in-memory.
        let discovery_registry = match opt_config {
            Some(_cfg) => {
                let url =
                    join_storage_url(&storage_config.metadata_url, "discovery_jobs/registry.json");
                info!(
                    "🔍 SharedServices: discovery-job registry persistence at {}",
                    url
                );
                if url.starts_with("file://") {
                    Arc::new(
                        crate::services::discovery::DiscoveryRegistry::load_or_create_at(
                            std::path::PathBuf::from(url.trim_start_matches("file://")),
                        ),
                    )
                } else {
                    Arc::new(
                        crate::services::discovery::DiscoveryRegistry::load_or_create_at_url(
                            url,
                            filesystem_factory.clone(),
                        )
                        .await
                        .context("opening object-store discovery registry")?,
                    )
                }
            }
            None => Arc::new(crate::services::discovery::DiscoveryRegistry::new()),
        };
        let discovery_service = Arc::new(crate::services::discovery::DiscoveryService::new(
            discovery_registry.clone(),
            snapshot_coordinator.clone(),
        ));
        // Phase 8 (F5): external-collection registry + service. Durable JSON
        // sidecar under the data dir (mirrors the discovery-job registry);
        // in-memory for embedded/test harnesses without a data dir.
        let external_collection_registry = match opt_config {
            Some(cfg) => {
                let path = cfg
                    .server
                    .data_dir
                    .join("external_collections")
                    .join("registry.json");
                info!(
                    "🔗 SharedServices: external-collection registry persistence at {}",
                    path.display()
                );
                Arc::new(
                    crate::services::external_collection::ExternalCollectionRegistry::load_or_create_at(
                        path,
                    ),
                )
            }
            None => {
                Arc::new(crate::services::external_collection::ExternalCollectionRegistry::new())
            }
        };
        let external_collection_service = Arc::new({
            #[cfg(feature = "axis")]
            {
                crate::services::external_collection::ExternalCollectionService::new(
                    external_collection_registry,
                    catalog_manager.clone(),
                    axis_manager.clone(),
                )
                .with_axis_runtime_enabled(axis_runtime_enabled)
            }
            #[cfg(not(feature = "axis"))]
            {
                crate::services::external_collection::ExternalCollectionService::new(
                    external_collection_registry,
                    catalog_manager.clone(),
                )
            }
        });
        {
            let executor = Arc::new(
                crate::services::discovery::DiscoveryJobExecutor::new(
                    discovery_registry.clone(),
                    snapshot_coordinator.clone(),
                )
                .with_vector_ops(vector_operations_service.clone()),
            );
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            crate::services::shutdown_registry::register("discovery-executor", shutdown_tx);
            // The discovery executor is a long-running background task; we
            // intentionally drop the JoinHandle so it runs for the process
            // lifetime. `spawn_discovery_executor` spawns its own task
            // internally, so what we drop here is a fire-and-forget future
            // already on the runtime, not a Future awaiting first poll.
            #[allow(clippy::let_underscore_future)]
            let _ = crate::services::discovery::spawn_discovery_executor(
                executor,
                shutdown_rx,
                crate::services::discovery::DEFAULT_POLL_INTERVAL,
            );
            info!("✅ SharedServices: DiscoveryJobExecutor spawned (Phase 8 CS/CD loop)");
        }
        // Phase-5 recall observer (AXIS-gated): the whole job probes AXIS-managed
        // IVF/HNSW recall, so it is not spawned in a PAX-exact-scan build.
        #[cfg(feature = "axis")]
        if axis_runtime_enabled {
            // Phase-5 recall observer (TD-075 / F2): periodically probes
            // quantized-vs-exact recall per collection and feeds the shared
            // RecallProbeGate, so the quantized IVF route opens once recall is
            // verified. Always-on (the watch sender is intentionally leaked,
            // matching the discovery executor).
            let observer = Arc::new(
                crate::services::recall_observer::RecallObserver::new(
                    axis_manager.clone(),
                    vector_operations_service.clone(),
                )
                // F1 trigger arm: a recall regression (gate open -> closed) is a
                // quality-driven discovery signal orthogonal to write-volume
                // drift — emit RecallDegraded so the index reclusters even when
                // recall degrades without new writes (coalesced, non-mutating).
                .with_discovery(discovery_service.clone()),
            );
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            crate::services::shutdown_registry::register("recall-observer", shutdown_tx);
            #[allow(clippy::let_underscore_future)]
            let _ = crate::services::recall_observer::spawn_recall_observer(
                observer,
                shutdown_rx,
                crate::services::recall_observer::DEFAULT_OBSERVE_INTERVAL,
            );
            info!(
                "✅ SharedServices: RecallObserver spawned (Phase 5 recall gate + F1 recall-degradation trigger)"
            );
        }
        #[cfg(feature = "axis")]
        if axis_runtime_enabled {
            // Trigger arm (T1.9): the write-volume drift watcher is the first
            // live producer — it counts each collection's own write batches since
            // its last completed recluster (per-collection, not a global-LSN
            // delta) and auto-enqueues a recluster once that count crosses the
            // threshold. Coalescing bounds it; recluster is non-mutating today,
            // so this is safe on by default.
            let watcher = Arc::new(
                crate::services::discovery::DriftWatcher::new(
                    discovery_service.clone(),
                    snapshot_coordinator.clone(),
                    crate::services::discovery::threshold_writes_from_env(),
                )
                // Sweep every collection (by name), not just those with prior
                // discovery history — makes the loop autonomous for brand-new
                // collections (no operator seed needed). Collections without
                // enough indexed vectors no-op in the recluster pass.
                .with_collection_source(
                    collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>
                ),
            );
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            crate::services::shutdown_registry::register("drift-watcher", shutdown_tx);
            #[allow(clippy::let_underscore_future)]
            let _ = crate::services::discovery::spawn_drift_watcher(
                watcher,
                shutdown_rx,
                crate::services::discovery::interval_from_env(),
            );
            info!(
                "✅ SharedServices: DriftWatcher spawned (Phase 8 F1 trigger arm — write-volume drift)"
            );
        }
        // Recall-drift sweeper (AXIS-gated): walks `recall_target:`-tagged
        // collections and emits axis_recall_drift_* metrics — no AXIS, no drift.
        #[cfg(feature = "axis")]
        if axis_runtime_enabled {
            // Recall-drift sweeper: every 5 min, walk every
            // collection with a `recall_target:` tag and emit a
            // Prometheus drift observation
            // (axis_recall_drift_status / _observations_total).
            // Read-only — never mutates AXIS state; the operator
            // drives hot-swaps via POST /recall-tune and rebuilds
            // via /recluster (forthcoming).
            let sweeper = Arc::new(
                crate::services::recall_drift_sweeper::RecallDriftSweeper::new(
                    collection_service.clone()
                        as Arc<dyn crate::services::recall_drift_sweeper::CollectionLister>,
                ),
            );
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            crate::services::shutdown_registry::register("drift-observer", shutdown_tx);
            #[allow(clippy::let_underscore_future)]
            let _ = crate::services::recall_drift_sweeper::spawn_recall_drift_sweeper(
                sweeper,
                shutdown_rx,
                crate::services::recall_drift_sweeper::DEFAULT_SWEEP_INTERVAL,
            );
            info!(
                "✅ SharedServices: RecallDriftSweeper spawned (axis_recall_drift_* metrics heartbeat)"
            );
        }

        // Phase 7c: resolve self_pod_id for partition lease manager initialization
        let self_pod_id_resolved = crate::cluster::primary_pod_registry::resolve_self_pod_id(None);

        // Expose the exact manager already wired to DML and its renewal loop.
        // The previous second constructor created an independent lease timeline
        // (different TTL, no renew loop) for protocol DDL.
        let partition_lease_manager = lease_manager_for_writes.clone();

        info!(
            "✅ SharedServices: Business logic hub ready for ALL protocols (gRPC, REST, WebSocket, etc.)"
        );

        Ok((
            Self {
                filesystem_factory,
                catalog_manager,
                model_registry_service,
                segment_registry: Arc::new(crate::catalog::SegmentRegistry::new()),
                collection_service: collection_service.clone(),
                vector_operations_service: vector_operations_service.clone(),
                graph_service: graph_service.clone(),
                graph_collection_service: graph_collection_service.clone(),
                graph_query_service,
                graph_execution_service,
                document_service: document_service.clone(),
                observability_service: observability_service.clone(),
                record_ops: record_ops.clone(),
                event_log,
                metrics_collector,
                metrics_updater: metrics_updater.clone(),
                query_facade,
                query_adapter: query_adapter.clone(),
                api_handlers: runtime_api_handlers.clone(),
                // Task #72: ClusterPort wiring slot. Defaults to None for
                // single-node bootstrap; populate via builder when [distributed]
                // config is present and a ClusterManager has been constructed.
                cluster_port: None,
                // Task #76 collection slice: port-typed view of the same
                // CollectionService instance held by `collection_service`.
                // Consumers should prefer `collection_port` going forward.
                collection_port: collection_service.clone()
                    as Arc<dyn proximadb_runtime::CollectionPort>,
                // Task #76 vector-ops slice: port-typed view of the same
                // VectorOperationsService instance held by `vector_operations_service`.
                vector_ops_port: vector_operations_service.clone()
                    as Arc<dyn proximadb_runtime::VectorOpsPort>,
                // Task #76 document slice (ADR-015 step 4): port-typed view
                // of the same DocumentService instance held by
                // `document_service`. Powered by the bare-service DocumentPort
                // impl in src/storage/document/service.rs (ADR-015 step 1).
                document_port: document_service.clone() as Arc<dyn proximadb_runtime::DocumentPort>,
                // Task #76 observability slice — wrapper-as-port-host pattern
                // (suboptimal vs ADR-015; cleanup is a follow-up session).
                observability_port: Arc::new(crate::network::grpc::ObservabilityServiceImpl::new(
                    observability_service.clone(),
                ))
                    as Arc<dyn proximadb_runtime::ObservabilityPort>,
                // Task #76 graph slice — same wrapper-as-port-host pattern.
                // Uses with_adapter() so search/explain methods have the query
                // adapter wired (matches the production wiring at
                // src/network/multi_server.rs:415).
                graph_port: Arc::new(crate::network::grpc::GraphServiceImpl::with_adapter(
                    graph_service.clone(),
                    runtime_api_handlers.clone(),
                    query_adapter.clone(),
                )) as Arc<dyn proximadb_runtime::GraphPort>,
                // T3.2 Slice 1: shared full-text index map for hybrid
                // retrieval. Same in-process map serves REST, gRPC, and the
                // R-7c rank-pipeline BM25 leg (`ProductionHybridBackend`).
                fulltext_indexes,
                // Vector Object Economy Phase 4 (2-B): process-wide
                // per-collection directory cache. The same Arc is also
                // attached to `vector_operations_service` via
                // `with_directory_cache` above so the search service can
                // touch the cache without re-resolving SharedServices.
                directory_cache,
                // Phase 6: per-collection pinning registry.
                // Constructed up front (line ~322) so REST handlers
                // (control plane) and the SST tier-migration
                // integration (data plane) hold the same `Arc`.
                pin_registry,
                // Phase 7.2: cache-affinity registry. Populated by
                // the unified search path; consumed by operator
                // inspection and future cluster-mode RoutingService
                // attach.
                affinity_registry,
                // Slice 2 of tenant-pod-affinity: primary-pod
                // registry. Constructed up front (alongside the
                // pin / affinity registries) so subsequent slices —
                // REST API, gateway write router, xCatalog backing
                // — all hold the same `Arc`.
                primary_pod_registry,
                // Slice 6.1: resolved once here so REST AppState and
                // the gRPC v2 service share the identical pod
                // identity. Pulled from `PROXIMADB_POD_ID` env var
                // with a `"self"` fallback for single-node setups.
                self_pod_id: self_pod_id_resolved,
                // Phase 7c: partition lease manager for per-collection
                // write authority (initialized above).
                partition_lease_manager,
                // T2.3 / TD-066 production wiring: the shared canonical
                // WAL appender opened earlier (Some when opt_config is
                // provided). Held here so multi_server.rs can clone it
                // for pgwire direct writes — guaranteeing both consumers
                // share the same next_sequence counter.
                canonical_wal_appender,
                canonical_record_store,
                conditional_key_store,
                #[cfg(feature = "experimental-ledger")]
                ledger_store,
                // TD-ABAC control-plane: the three durable ABAC store handles,
                // cloned from the single `abac_stores` trio opened above so the
                // admin-provisioning endpoints share the SAME instances the live
                // enforcer reads (hot-reload). `None` when ABAC is off.
                #[cfg(feature = "abac-policy")]
                abac_authority: abac_stores.as_ref().map(|s| s.authority.clone()),
                #[cfg(feature = "abac-policy")]
                abac_binding_store: abac_stores.as_ref().map(|s| s.bindings.clone()),
                #[cfg(feature = "abac-policy")]
                abac_grant_store: abac_stores.as_ref().map(|s| s.grants.clone()),
                #[cfg(feature = "abac-policy")]
                abac_posture_store: abac_stores.as_ref().map(|s| s.posture.clone()),
                #[cfg(feature = "abac-policy")]
                abac_predicate_store: abac_stores.as_ref().map(|s| s.predicate_objects.clone()),
                #[cfg(feature = "abac-policy")]
                abac_enforcer,
                // TD-064 / LLD §5: per-collection recall-probe gate. Empty
                // at startup; populated as the stats refresher / search path
                // observe probe outcomes. Route-health surfaces per-scope
                // state for operator visibility.
                recall_probe_gate,
                // R-7c.3 production wiring: shared rank-pipeline singleton +
                // durable rank-profile catalog. Both are built ahead of the
                // FederatedQueryContext so SQL RERANK shares the registry.
                rank_services,
                rank_profile_store,
                // UDF F5 (5b-ii): durable function catalog, replayed +
                // re-registered into builtins() above so user functions
                // survive restart.
                function_store,
                // Phase 8 (F1): snapshot-publish coordinator + Continuous
                // Discovery service. The background executor was spawned above.
                snapshot_coordinator,
                discovery_service,
                external_collection_service,
                // Phase H — TurboQuant store registry. Default to an
                // empty in-memory registry; collection-create / collection-
                // load (Phase I, follow-up) populates it from
                // `DerivedQuantizationLevel::TurboQuant` rows in the
                // catalog. The trait-object widening hides the concrete
                // `InMemoryTurboQuantStoreRegistry` so a future swap to a
                // distributed impl (Phase F4b) is a single-line change.
                //
                // Phase P: this slot reuses the SAME registry instance
                // that was hoisted ~1400 lines above and threaded into
                // `CollectionService::with_turboquant_registry()` and
                // the boot-time hydration loop. All three consumers
                // (create-time wire, boot-time hydration, downstream
                // search dispatch) share one map — see Phase P design
                // rationale §"Why hoist the registry construction".
                storage_write_fence,
                #[cfg(feature = "experimental-turboquant")]
                turboquant_registry: Some(turboquant_registry),
            },
            collection_service,
        ))
    }

    /// Optional metrics updater for wiring into services. Currently returns None
    /// unless a metrics updater is injected in the future.
    pub fn metrics_updater(
        &self,
    ) -> Option<Arc<dyn crate::metrics::InternalMetricsUpdater + 'static>> {
        self.metrics_updater.clone()
    }

    /// Set the cluster orchestration port. Phase 9.12 / Task #72.
    ///
    /// Use this when bootstrap detects `[distributed]` config and has
    /// constructed a `ClusterManager` (or any other `ClusterPort` impl —
    /// see `crates/platform/proximadb-runtime/src/cluster_port.rs`).
    /// Single-node deployments leave it as `None`.
    pub fn with_cluster_port(mut self, port: Arc<dyn proximadb_runtime::ClusterPort>) -> Self {
        self.cluster_port = Some(port);
        self
    }

    /// Get the unified query facade - single entry point for all query types
    ///
    /// The facade consolidates vector search, SQL, and graph queries into a unified
    /// interface with automatic strategy selection and routing.
    pub fn query_facade(&self) -> Arc<UnifiedQueryFacade> {
        self.query_facade.clone()
    }

    /// Create a QueryFacadeAdapter for protocol handlers
    ///
    /// The adapter provides protocol-agnostic methods that convert proto types
    /// to/from QueryRequest/QueryResult, enabling query routing.
    pub fn query_adapter(&self) -> Arc<QueryFacadeAdapter> {
        // Return the shared, DmlService-wired adapter built in `new` so the
        // port path (runtime handler / pgwire / embedded) reproduces ROOT's
        // EXPLAIN `<DML>` routing rather than a fresh DmlService-less adapter.
        self.query_adapter.clone()
    }

    /// Recover vectors from write buffer after StorageEngine has started
    /// This should be called from ProximaDB::new after storage.start()
    pub async fn recover_vectors_from_write_buffer(
        &self,
        storage: &Arc<RwLock<StorageEngine>>,
    ) -> Result<()> {
        info!("🔄 SharedServices: Starting vector recovery from write buffer");

        // Get collections that need vector recovery
        let storage_ref = storage.read().await;
        let recovered_collections = storage_ref.recovered_collections_metadata().await?;

        if recovered_collections.is_empty() {
            info!("📋 SharedServices: No collections found for vector recovery");
            return Ok(());
        }

        info!(
            "📦 SharedServices: Found {} collections for potential vector recovery",
            recovered_collections.len()
        );

        // Implement comprehensive vector recovery from WAL to VectorOperationsService
        let mut total_vectors_recovered = 0u64;

        for (collection_id, _collection) in &recovered_collections {
            // 1. Check if write buffer has unflushed data for this collection
            let unflushed_batches = match storage_ref
                .write_ahead_log_manager()
                .read_all_batches(collection_id, None)
                .await
            {
                Ok(batches) => batches,
                Err(e) => {
                    warn!(
                        "Failed to read unflushed batches for collection {}: {}",
                        collection_id, e
                    );
                    continue;
                }
            };

            if unflushed_batches.is_empty() {
                debug!(
                    "No unflushed vectors found for collection: {}",
                    collection_id
                );
                continue;
            }

            // 2. Load vectors from write buffer into VectorOperationsService memtable
            let mut collection_vectors_recovered = 0u64;

            for batch in unflushed_batches {
                let batch_size = batch.vector_records.len();

                // Insert each vector into the VectorOperationsService memtable
                for vector_record in batch.vector_records.iter() {
                    match self
                        .vector_operations_service
                        .insert_vectors_direct(collection_id, Arc::new(vec![vector_record.clone()]))
                        .await
                    {
                        Ok(_) => {
                            collection_vectors_recovered += 1;
                        }
                        Err(e) => {
                            warn!(
                                "Failed to recover vector {} for collection {}: {}",
                                &vector_record.oid, collection_id, e
                            );
                        }
                    }
                }

                debug!(
                    "Recovered batch {} with {} vectors for collection {}",
                    batch.batch_id.to_base62(),
                    batch_size,
                    collection_id
                );
            }

            total_vectors_recovered += collection_vectors_recovered;

            // 3. Mark recovery complete for this collection
            info!(
                "✅ Collection '{}': Recovered {} vectors from WAL to memtable",
                collection_id, collection_vectors_recovered
            );
        }

        info!(
            "✅ SharedServices: Vector recovery completed - {} vectors across {} collections",
            total_vectors_recovered,
            recovered_collections.len()
        );

        Ok(())
    }

    /// Convert TOML WALConfig to internal WALConfig
    fn convert_toml_to_wal_config(
        toml_config: &crate::core::config::WriteBufferUserConfig,
    ) -> crate::storage::persistence::write_ahead_log::config::WALConfig {
        // ADR-069 S1 guardrail: the WAL belongs on a local reattachable disk
        // (object-store WAL was the pre-pivot architecture and pays an I/O
        // round-trip per append). Warn — at `error` level when a durability-
        // sensitive `sync_mode` is also set, because an object store does not
        // honour `fsync` the way a local block device does, so a remote WAL +
        // PerBatch/Always makes a durability claim that does not hold. A bare
        // path (no `scheme://`) is local (resolved to `file://`).
        {
            let dir = toml_config.write_buffer_directory.trim();
            let is_remote = dir.contains("://") && !dir.starts_with("file:");
            if is_remote {
                let sync = toml_config.sync_mode.to_lowercase();
                if matches!(sync.as_str(), "perbatch" | "always") {
                    tracing::error!(
                        wal_dir = %dir,
                        sync_mode = %toml_config.sync_mode,
                        "ADR-069 S1: WAL write_buffer_directory is on a remote (object-store) \
                         scheme with a durability-sensitive sync_mode — object stores do not \
                         honour fsync like a local disk, so the durability claim is not trustworthy. \
                         Move the WAL to a local file:// path."
                    );
                } else {
                    tracing::warn!(
                        wal_dir = %dir,
                        "ADR-069 S1: WAL write_buffer_directory is on a remote (object-store) \
                         scheme; ADR-069 places the WAL on local disk. Remote WAL pays an I/O \
                         round-trip per append."
                    );
                }
            }
        }

        // TD-CONFIG-CONSOLIDATE-1 step 2: delegate the field-mapping to the SINGLE
        // canonical conversion (`WriteBufferUserConfig::to_engine_config`), then
        // apply the one server-only override — the optimized writer follows
        // `enable_wal` (the embedded path intentionally keeps it off). Removes the
        // dual-maintenance drift where every new WAL field had to be wired into BOTH
        // converters (the wal_local_dir near-miss during ADR-069 S1).
        let mut cfg = toml_config.to_engine_config();
        cfg.enable_optimized_writer = toml_config.enable_wal;
        cfg
    }
}

/// Build the process-wide `RankServices` singleton + the durable rank-profile
/// store that backs it. R-7c.3 production wiring.
///
/// When a canonical WAL appender is supplied the store is durable; otherwise
/// it falls back to an in-memory appender (sufficient for tests and the
/// embedded-mode boot path). Existing profiles in the canonical WAL are
/// replayed into the store and then compiled into the `ProfileRegistry` so
/// dashboards see them on a cold boot. Compile failures bump the
/// `proximadb_rank_profile_reload_total{outcome="error"}` counter but never
/// fail the boot — operators can repair the catalog entry without taking
/// the server down.
async fn build_rank_services(
    vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
    fulltext_indexes: crate::network::hybrid_search::HybridFullTextIndexMap,
    canonical_wal_appender: Option<Arc<crate::services::FramedTableWalAppender>>,
    opt_config: Option<&crate::core::config::Config>,
) -> (
    Arc<crate::network::rest::canonical::rank::RankServices>,
    Arc<dyn crate::services::RankProfileStore>,
) {
    use crate::services::record_store::TableWalAppender;
    use crate::services::{FramedTableWalAppender, MemoryTableWalAppender};

    // Load existing profiles from the canonical WAL (when present) so the
    // store starts populated even before the registry is built.
    let (store_appender, recovered_entries): (Arc<dyn TableWalAppender>, _) = if let Some(
        appender,
    ) =
        canonical_wal_appender
    {
        let path = appender.path().to_path_buf();
        let entries = match FramedTableWalAppender::read_entries_from_path(&path).await {
            Ok(entries) => entries,
            Err(err) => {
                warn!(
                    "SharedServices: failed to replay rank-profile WAL at {}: {} — starting with empty profile catalog",
                    path.display(),
                    err
                );
                Vec::new()
            }
        };
        (appender as Arc<dyn TableWalAppender>, entries)
    } else {
        (
            Arc::new(MemoryTableWalAppender::new()) as Arc<dyn TableWalAppender>,
            Vec::new(),
        )
    };

    build_rank_services_with_appender(
        vector_ops,
        fulltext_indexes,
        store_appender,
        &recovered_entries,
        opt_config,
    )
    .await
}

/// Construct the durable SQL user-function catalog (UDF F5) and re-register
/// every persisted definition into the shared `proximadb_functions::builtins()`
/// registry so user functions are live on both engines after a restart.
///
/// Mirrors [`build_rank_services`]: when a canonical WAL appender is supplied
/// the store is durable and its `__proxima_functions__` slice is replayed;
/// otherwise it falls back to an in-memory appender (tests / embedded boot).
/// Re-registration failures (e.g. a body that no longer lowers) are logged but
/// never fail the boot — operators can repair or drop the offending entry.
async fn build_function_store(
    canonical_wal_appender: Option<Arc<crate::services::FramedTableWalAppender>>,
) -> Arc<dyn crate::services::FunctionStore> {
    use crate::services::record_store::TableWalAppender;
    use crate::services::{
        CanonicalWalFunctionStore, FramedTableWalAppender, MemoryTableWalAppender,
    };

    let (store_appender, recovered_entries): (Arc<dyn TableWalAppender>, _) = if let Some(
        appender,
    ) =
        canonical_wal_appender
    {
        let path = appender.path().to_path_buf();
        let entries = match FramedTableWalAppender::read_entries_from_path(&path).await {
            Ok(entries) => entries,
            Err(err) => {
                warn!(
                    "SharedServices: failed to replay function-catalog WAL at {}: {} — starting with empty function catalog",
                    path.display(),
                    err
                );
                Vec::new()
            }
        };
        (appender as Arc<dyn TableWalAppender>, entries)
    } else {
        (
            Arc::new(MemoryTableWalAppender::new()) as Arc<dyn TableWalAppender>,
            Vec::new(),
        )
    };

    let store: Arc<dyn crate::services::FunctionStore> = Arc::new(
        CanonicalWalFunctionStore::from_wal_entries(store_appender, &recovered_entries),
    );

    let recovered = match store.list_all().await {
        Ok(functions) => functions,
        Err(err) => {
            warn!(
                "SharedServices: function-catalog recovery list_all failed: {} — starting with empty function registry",
                err
            );
            Vec::new()
        }
    };
    let mut recovered_count = 0usize;
    for function in &recovered {
        match crate::services::function_store::register_stored_function(function) {
            Ok(()) => {
                recovered_count += 1;
                debug!(
                    "SharedServices: recovered SQL function '{}' ({} params)",
                    function.name,
                    function.params.len()
                );
            }
            Err(err) => warn!(
                "SharedServices: failed to re-register SQL function '{}': {}",
                function.name, err
            ),
        }
    }
    info!(
        "✅ SharedServices: function catalog ready (recovered={})",
        recovered_count
    );

    store
}

/// Inner builder that takes a pre-resolved appender + recovered entries. Split
/// out so tests can drive it with an in-memory appender without a temp-dir
/// round-trip.
async fn build_rank_services_with_appender(
    vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
    fulltext_indexes: crate::network::hybrid_search::HybridFullTextIndexMap,
    store_appender: Arc<dyn crate::services::record_store::TableWalAppender>,
    recovered_entries: &[proximadb_storage_common::CanonicalWalEntry],
    opt_config: Option<&crate::core::config::Config>,
) -> (
    Arc<crate::network::rest::canonical::rank::RankServices>,
    Arc<dyn crate::services::RankProfileStore>,
) {
    use crate::core::search::hybrid::FusionStrategy;
    use crate::network::rest::canonical::rank::{HybridCoordinatorAdapter, RankServices};
    use crate::network::rest::canonical::rank_backend::ProductionHybridBackend;
    use crate::observability::rank_metrics::init_rank_pipeline_metrics;
    use crate::services::CanonicalWalRankProfileStore;

    let store: Arc<dyn crate::services::RankProfileStore> = Arc::new(
        CanonicalWalRankProfileStore::from_wal_entries(store_appender, recovered_entries),
    );

    // Build the production hybrid backend over the shared vector port + the
    // shared per-collection BM25 index map. Both surfaces are already
    // SharedServices fields.
    let backend = Arc::new(ProductionHybridBackend::new(vector_ops, fulltext_indexes));
    let adapter = Arc::new(HybridCoordinatorAdapter::new(
        FusionStrategy::ReciprocalRank { k: 60 },
        backend,
    ));

    // Register the spec §4.10 metric family against the process-wide
    // rank-metrics registry. Idempotent on hot-reload paths.
    let metrics = init_rank_pipeline_metrics();
    // TD-SELECTOR-1 gate 3: `[query.reranking]` from the server config reaches
    // the production cross-modal reranker. Before this, the handler hardcoded
    // `default_rerank_config()` and the config file's `rerank_top_k` was inert
    // (loaded, validated, read by nobody).
    let rerank_config = opt_config
        .and_then(|c| c.query.as_ref())
        .map(|q| q.reranking.clone())
        .unwrap_or_else(crate::network::rest::canonical::rank::default_rerank_config);
    let services = Arc::new(
        RankServices::new(adapter)
            .with_metrics(metrics)
            .with_rerank_config(rerank_config),
    );

    // Recover compiled profiles from the durable store. Validation /
    // compilation failures are logged + recorded as failed reloads — they do
    // not fail the boot.
    let recovered_profiles = match store.list_all().await {
        Ok(profiles) => profiles,
        Err(err) => {
            warn!(
                "SharedServices: rank-profile recovery list_all failed: {} — starting with empty registry",
                err
            );
            Vec::new()
        }
    };
    for profile in recovered_profiles {
        match recover_profile(&services, &profile) {
            Ok(()) => debug!(
                "SharedServices: recovered rank profile '{}' (version={})",
                profile.name, profile.version
            ),
            Err(err) => {
                warn!(
                    "SharedServices: failed to recover rank profile '{}' (version={}): {}",
                    profile.name, profile.version, err
                );
                services.record_profile_reload_error(&profile.name);
            }
        }
    }

    (services, store)
}

fn recover_profile(
    services: &crate::network::rest::canonical::rank::RankServices,
    profile: &crate::services::StoredRankProfile,
) -> Result<(), String> {
    use proximadb_rank_profile::{CompiledRankProfile, dsl::parse_single};

    let spec = parse_single(&profile.name, &profile.spec_toml).map_err(|e| e.to_string())?;
    let compiled = CompiledRankProfile::compile(spec, services.blueprint_factory.clone())
        .map_err(|e| e.to_string())?;
    services.install_profile(compiled);
    Ok(())
}

#[cfg(test)]
mod rank_services_wiring_tests {
    use super::*;
    use crate::services::record_store::TableWalAppender;
    use crate::services::{MemoryTableWalAppender, RankProfileStore};
    use async_trait::async_trait;
    use proximadb_runtime::VectorOpsPort;
    use serde_json::Value as JsonValue;
    use std::collections::HashMap;
    use std::sync::RwLock;

    // ── Minimal no-op vector port ────────────────────────────────────────────

    struct NoopVectorPort;

    #[async_trait]
    impl VectorOpsPort for NoopVectorPort {
        async fn search(
            &self,
            _request: crate::proto::proximadb_v1::VectorSearchRequest,
            _identity: proximadb_runtime::PortIdentity<'_>,
        ) -> anyhow::Result<crate::proto::proximadb_v1::VectorOperationResponse> {
            Ok(crate::proto::proximadb_v1::VectorOperationResponse {
                success: true,
                operation: 0,
                metrics: None,
                results: Some(crate::proto::proximadb_v1::SearchResult {
                    results: Vec::new(),
                    total_found: 0,
                    collection_id: None,
                }),
                vector_ids: Vec::new(),
                error_message: None,
                error_code: None,
            })
        }

        async fn batch_upsert(
            &self,
            _request: crate::proto::proximadb_v1::VectorBatchRequest,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<crate::proto::proximadb_v1::VectorOperationResponse> {
            unimplemented!()
        }

        async fn get_vector(
            &self,
            _collection_id: &str,
            _vector_id: &str,
            _include_vector: bool,
            _include_metadata: bool,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<crate::proto::proximadb_v1::VectorOperationResponse> {
            unimplemented!()
        }

        async fn flush_all(&self) -> anyhow::Result<()> {
            Ok(())
        }

        async fn metrics(&self) -> anyhow::Result<JsonValue> {
            Ok(JsonValue::Null)
        }
    }

    fn empty_indexes() -> crate::network::hybrid_search::HybridFullTextIndexMap {
        Arc::new(RwLock::new(HashMap::new()))
    }

    fn valid_profile_toml() -> String {
        // Simplest possible profile: a constant first_phase expression
        // (`"1.0"`) that the default `BlueprintFactory` compiles without any
        // extra feature registrations.
        r#"
[first_phase]
expression = "1.0"
heap_size = 50
"#
        .to_string()
    }

    fn invalid_profile_toml() -> String {
        // Refers to a feature that the default `BlueprintFactory` knows
        // nothing about — compilation should fail.
        r#"
[first_phase]
expression = "definitely_not_a_feature(\"missing\")"
heap_size = 50
"#
        .to_string()
    }

    #[tokio::test]
    async fn empty_catalog_produces_empty_registry() {
        let appender: Arc<dyn TableWalAppender> = Arc::new(MemoryTableWalAppender::new());
        let (services, store) = build_rank_services_with_appender(
            Arc::new(NoopVectorPort),
            empty_indexes(),
            appender,
            &[],
            None,
        )
        .await;

        assert_eq!(services.profile_registry.len(), 0);
        assert_eq!(
            store.list_all().await.unwrap().len(),
            0,
            "store should also start empty"
        );
    }

    #[tokio::test]
    async fn one_valid_profile_recovers_into_registry() {
        // Step 1: install a profile through a primed store so the appender
        // accumulates a real `RecordUpsert` entry. We keep a concrete
        // `Arc<MemoryTableWalAppender>` so the test can read entries back; the
        // builder receives the same Arc upcast to `dyn TableWalAppender`.
        let memory_appender = Arc::new(MemoryTableWalAppender::new());
        let primed_appender: Arc<dyn TableWalAppender> = memory_appender.clone();
        let primed = crate::services::CanonicalWalRankProfileStore::new(primed_appender);
        primed
            .install("good", valid_profile_toml(), None, None)
            .await
            .unwrap();

        let entries = memory_appender.entries().await;
        assert_eq!(entries.len(), 1);

        let builder_appender: Arc<dyn TableWalAppender> = memory_appender.clone();
        let (services, store) = build_rank_services_with_appender(
            Arc::new(NoopVectorPort),
            empty_indexes(),
            builder_appender,
            &entries,
            None,
        )
        .await;

        assert_eq!(services.profile_registry.len(), 1);
        assert!(services.profile_registry.get("good").is_some());
        assert_eq!(store.list_all().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn invalid_profile_does_not_panic_boot() {
        // A profile that parses but fails compilation should be logged and
        // skipped — boot must succeed and the registry stays empty for that
        // profile name.
        let memory_appender = Arc::new(MemoryTableWalAppender::new());
        let primed_appender: Arc<dyn TableWalAppender> = memory_appender.clone();
        let primed = crate::services::CanonicalWalRankProfileStore::new(primed_appender);
        primed
            .install("broken", invalid_profile_toml(), None, None)
            .await
            .unwrap();
        primed
            .install("good", valid_profile_toml(), None, None)
            .await
            .unwrap();

        let entries = memory_appender.entries().await;

        let builder_appender: Arc<dyn TableWalAppender> = memory_appender.clone();
        let (services, store) = build_rank_services_with_appender(
            Arc::new(NoopVectorPort),
            empty_indexes(),
            builder_appender,
            &entries,
            None,
        )
        .await;

        // Only the valid profile makes it into the live registry.
        assert!(services.profile_registry.get("good").is_some());
        assert!(
            services.profile_registry.get("broken").is_none(),
            "broken profile must not appear in the live registry"
        );
        // But both still exist in the durable store — operators repair, not
        // the boot path.
        assert_eq!(store.list_all().await.unwrap().len(), 2);
    }
}

#[cfg(test)]
mod function_store_wiring_tests {
    use super::*;
    use crate::services::canonical_wal::FramedTableWalAppender;
    use crate::services::{CanonicalWalFunctionStore, FunctionStore, StoredFunction};
    use proximadb_data_model::{ProximaType, ProximaValue};
    use tempfile::tempdir;

    /// UDF F5 (5b-ii): `build_function_store` is the boot-recovery entry point.
    /// It must replay the persisted `CREATE FUNCTION` catalog from the canonical
    /// WAL AND re-register every definition into the shared
    /// `proximadb_functions::builtins()` registry so user functions are live on
    /// both engines after a restart.
    #[tokio::test]
    async fn build_function_store_replays_and_reregisters_on_boot() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("functions.wal");

        // First boot: persist a CREATE FUNCTION definition into the canonical WAL.
        {
            let appender = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
            let store = CanonicalWalFunctionStore::new(appender);
            store
                .put(StoredFunction {
                    name: "boot_recovered_triple".to_string(),
                    params: vec![("x".to_string(), ProximaType::Int64)],
                    return_ty: ProximaType::Int64,
                    body: "x * 3".to_string(),
                    created_at_ms: 0,
                })
                .await
                .unwrap();
        }

        // Precondition: the fresh-named function is NOT yet in the process-wide
        // registry (it has never been registered in this test binary).
        assert!(
            proximadb_functions::builtins()
                .lookup_scalar("boot_recovered_triple")
                .is_none(),
            "precondition: function must not already be registered"
        );

        // "Restart": re-open the same WAL and run boot recovery.
        let reopened = Arc::new(FramedTableWalAppender::open(&wal_path).await.unwrap());
        let store = build_function_store(Some(reopened)).await;

        // The durable catalog lists the recovered function.
        let all = store.list_all().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "boot_recovered_triple");

        // And it is live on the shared registry — callable on both engines.
        let def = proximadb_functions::builtins()
            .lookup_scalar("boot_recovered_triple")
            .expect("function should be re-registered after boot recovery");
        let out = (def.kernel)(&[ProximaValue::Int64(7)]).expect("recovered fn eval");
        assert_eq!(out, ProximaValue::Int64(21));
    }

    /// A `None` appender (test / embedded boot path without a canonical WAL)
    /// yields an empty, in-memory-backed catalog instead of panicking.
    #[tokio::test]
    async fn build_function_store_without_wal_is_empty() {
        let store = build_function_store(None).await;
        assert_eq!(store.list_all().await.unwrap().len(), 0);
    }
}

#[cfg(test)]
mod join_storage_url_tests {
    use super::{join_storage_url, local_storage_path};

    /// TD-OBJSTORE-1 (#960): every (base, sub) pair must join scheme-preserving
    /// with no double scheme — the `file://adls://…` class must be impossible.
    #[test]
    fn joins_preserve_scheme_for_every_base_shape() {
        let cases = [
            ("/data", "metrics", "/data/metrics"),
            ("/data/", "metrics", "/data/metrics"),
            ("file:///data", "metrics", "file:///data/metrics"),
            ("s3://bucket/data", "auditlog", "s3://bucket/data/auditlog"),
            (
                "adls://container/data/",
                "timeseries",
                "adls://container/data/timeseries",
            ),
            (
                "abfs://container/data",
                "metrics",
                "abfs://container/data/metrics",
            ),
            (
                "gcs://bucket/data",
                "auditlog",
                "gcs://bucket/data/auditlog",
            ),
            ("azure://container/data", "", "azure://container/data"),
            ("s3://bucket/data", "", "s3://bucket/data"),
            ("/data", "", "/data"),
        ];
        for (base, sub, expected) in cases {
            let joined = join_storage_url(base, sub);
            assert_eq!(joined, expected, "join_storage_url({base:?}, {sub:?})");
            // No double scheme, ever.
            assert_eq!(
                joined.matches("://").count(),
                expected.matches("://").count(),
                "double scheme in {joined:?}"
            );
            assert!(
                !joined.starts_with("file://s3://") && !joined.starts_with("file://adls://"),
                "invalid double-scheme URL {joined:?}"
            );
        }
    }

    #[test]
    fn local_storage_path_rejects_object_store_bases() {
        assert_eq!(
            local_storage_path("/data").as_deref(),
            Some(std::path::Path::new("/data"))
        );
        assert_eq!(
            local_storage_path("file:///data").as_deref(),
            Some(std::path::Path::new("/data"))
        );
        assert!(local_storage_path("adls://container/data").is_none());
        assert!(local_storage_path("s3://bucket/data").is_none());
        assert!(local_storage_path("abfs://container/data").is_none());
    }
}

// ============================================================================
// ABAC composition-root integration test (TD-ABAC-2/3/4)
// ============================================================================
//
// The durable substrate (authority #1310 + bindings #1331 + predicate objects
// #1335) and the pgwire subject activation (#1333) are each proven in isolation:
// the abac-crate tests prove each `FileSystem*` store survives a restart, and
// #1324's `abac_relational_enforcement_tests` proves the enforcer + scan filter
// rows. The ONE integration point nothing covered was the composition root —
// `build_abac_enforcer` loading all THREE durable files into a working enforcer.
// These tests close that gap: seed the durable trio, call the real
// `build_abac_enforcer`, and assert it produces an enforcer that enforces.
#[cfg(all(test, feature = "abac-policy"))]
mod abac_composition_root_tests {
    use super::SharedServices;
    use crate::core::config::Config;
    use crate::core::search::{ComparisonOperator, FilterExpression};
    use crate::security::rls::AbacScanResult;
    use proximadb_abac::{
        AttributeBinding, FileSystemAttributeAuthority, FileSystemPolicyBindingStore,
        FileSystemPredicateObjectStore,
    };
    use proximadb_catalog::fc_metamodel::{
        AttrValue, Effect, PolicyBinding, Scope, SubjectId, Target,
    };
    use serde_json::json;

    fn unique_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "proximadb-abac-comproot-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Seed the durable trio, then exercise the REAL composition-root builder
    /// (`build_abac_enforcer`) — proving the three `FileSystem*` stores load into a
    /// working enforcer. This is the integration point no isolated unit test covers.
    #[test]
    fn build_abac_enforcer_loads_the_durable_trio_into_a_working_enforcer() {
        let dir = unique_dir();
        let abac_dir = dir.join("abac");
        std::fs::create_dir_all(&abac_dir).expect("abac dir");

        // 1. durable authority: alice → dept=eng in tenant 7.
        {
            let auth = FileSystemAttributeAuthority::open(abac_dir.join("attribute-bindings.json"))
                .expect("open authority");
            auth.upsert(
                AttributeBinding::new("alice", 7).with_attr("dept", AttrValue::Str("eng".into())),
            );
        }
        // 2. durable policy bindings: permit table 200 under predicate ref 42.
        {
            let bindings =
                FileSystemPolicyBindingStore::open(abac_dir.join("policy-bindings.json"))
                    .expect("open binding store");
            bindings.replace_tenant(
                7,
                vec![PolicyBinding {
                    object_id: 1,
                    tenant_stable_id: 7,
                    scope: Scope::Table(200),
                    effect: Effect::Permit,
                    predicate_ref: Some(42),
                    field_mask: None,
                }],
            );
        }
        // 3. durable predicate objects: ref 42 → dept == "eng".
        {
            let preds =
                FileSystemPredicateObjectStore::open(abac_dir.join("predicate-objects.json"))
                    .expect("open predicate store");
            preds.register(
                42,
                FilterExpression::Comparison {
                    field: "dept".to_string(),
                    operator: ComparisonOperator::Equals,
                    value: json!("eng"),
                },
            );
        }

        // Composition root: build the enforcer from the data_dir (reads all 3 files).
        let mut config = Config::default();
        config.server.data_dir = dir.clone();
        let enforcer = SharedServices::build_abac_enforcer(Some(&config))
            .expect("durable substrate present ⇒ enforcer built");

        let target = Target {
            namespace: 3,
            table: 200,
            column: None,
        };

        // alice (dept=eng) ⇒ Restricted to dept=eng rows — the full chain
        // (authority resolve → binding compose → predicate resolve) works through
        // the composition root.
        match enforcer.predicate_for(&SubjectId("alice".into()), 7, target) {
            AbacScanResult::Restricted(_) => {}
            _ => panic!("alice must be Restricted via the durable substrate"),
        }
        // An unbound subject ⇒ Denied (no attribute binding in the authority).
        assert!(
            matches!(
                enforcer.predicate_for(&SubjectId("mallory".into()), 7, target),
                AbacScanResult::Denied(_)
            ),
            "an unbound subject must be denied"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No `data_dir` (embedded/ephemeral paths) ⇒ no durable substrate ⇒ `None`.
    /// This is the documented fail-open-to-status-quo behavior (never a
    /// synthesized allow).
    #[test]
    fn build_abac_enforcer_returns_none_without_a_config() {
        assert!(
            SharedServices::build_abac_enforcer(None).is_none(),
            "no opt_config ⇒ no enforcer (status quo)"
        );
    }
}
