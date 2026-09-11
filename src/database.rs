//! ProximaDB Database Instance
//!
//! This module contains the main ProximaDB database instance implementation,
//! including initialization, lifecycle management, and core database operations.
//!
//! **TD-GOD-FILE**: This file (~870 lines) handles initialization, lifecycle,
//! server orchestration, and maintenance scheduling. It should be split into:
//! - `database/instance.rs` — ProximaDB struct + constructor
//! - `database/lifecycle.rs` — start/shutdown/health
//! - `database/maintenance.rs` — background tasks, RL checkpointing
//! See docs/10-quality/TECHNICAL_DEBT.adoc for tracking.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

// RL Planner checkpoint interval (5 minutes default)
const RL_CHECKPOINT_INTERVAL_SECS: u64 = 300;

use crate::{core, graph, network, proto, query, security, storage};
use proximadb_config::ServerTenantMode;

/// Main ProximaDB database instance
pub struct ProximaDB {
    /// Storage engine for vector data
    storage: Arc<RwLock<storage::StorageEngine>>,
    // consensus: consensus::ConsensusEngine,  // Disabled - requires raft dependency
    /// Multi-server for REST and gRPC endpoints
    multi_server: Option<network::MultiServer>,
    /// Configuration (stored for RL planner and other runtime access)
    _config: core::Config,
    /// Security coordinator for auth/RBAC
    #[allow(dead_code)]
    security: Option<Arc<security::SecurityCoordinator>>,
    /// Handle for RL planner checkpoint task (if enabled)
    rl_checkpoint_handle: Option<tokio::task::JoinHandle<()>>,
    /// Path where RL planner policy is persisted
    rl_policy_path: Option<String>,
    /// Queue subsystem for async ingest. Activated when the
    /// `PROXIMADB_QUEUE_ROOT` env var is set; otherwise `None` and
    /// async ingest degrades to inline embedding via the v3 handler
    /// fallback. Closed on `shutdown()`.
    queue_client: Option<Arc<proximadb_queue::QueueClient>>,
    /// Drainer task handle + its shutdown channel. Present when
    /// `queue_client` is present. `shutdown()` signals the drainer
    /// and awaits the join handle so in-flight work completes
    /// before the queue subsystem closes.
    drainer: Option<(
        tokio::task::JoinHandle<()>,
        tokio::sync::oneshot::Sender<()>,
    )>,
}

async fn initialize_configured_security(
    config: Option<security::SecurityConfig>,
    identity_dir: Option<std::path::PathBuf>,
) -> anyhow::Result<Option<Arc<security::SecurityCoordinator>>> {
    let Some(config) = config.filter(|config| config.enabled) else {
        return Ok(None);
    };

    let coordinator = security::initialize_security_with_identity(config, identity_dir)
        .await
        .map_err(|error| anyhow::anyhow!("enabled security initialization failed: {error:#}"))?;
    Ok(Some(Arc::new(coordinator)))
}

fn resolve_server_tenant_mode(
    server_tenant: &proximadb_config::ServerTenantConfig,
    security_config: Option<&security::SecurityConfig>,
) -> anyhow::Result<proximadb_tenant::TenantDeploymentMode> {
    let legacy_multi_tenant_required = security_config.is_some_and(|s| {
        s.authentication.require_authentication && s.mode != security::SecurityMode::Development
    });

    let mode = match server_tenant.mode {
        ServerTenantMode::Auto if legacy_multi_tenant_required => {
            proximadb_tenant::TenantDeploymentMode::MultiTenant
        }
        ServerTenantMode::Auto | ServerTenantMode::SingleTenant => {
            proximadb_tenant::TenantDeploymentMode::single_tenant(
                server_tenant.default_tenant.clone(),
            )
        }
        ServerTenantMode::MultiTenant => proximadb_tenant::TenantDeploymentMode::MultiTenant,
    };

    if let proximadb_tenant::TenantDeploymentMode::SingleTenant { default_tenant } = &mode {
        proximadb_tenant::validate_request_tenant(default_tenant).map_err(|err| {
            anyhow::anyhow!("invalid [server.tenant] default_tenant '{default_tenant}': {err}")
        })?;
    }

    Ok(mode)
}

impl ProximaDB {
    /// Create and initialize a new ProximaDB instance from the given configuration.
    pub async fn new(config: core::Config) -> anyhow::Result<Self> {
        tracing::info!("🚀 ProximaDB::new - STARTING database initialization");
        tracing::debug!("🔍 ProximaDB::new - Config: {:?}", config);

        // Fail closed before any storage init: memory-mapped I/O is invalid
        // against a cloud object store (S3/GCS/Azure) — the operator must set
        // `[storage.optimization] enable_mmap = false` for cloud deployments.
        config.storage.validate_cloud_mmap()?;

        // Wire corpus-revision durability before constructing any service.
        // SharedServices publishes the registry through the storage port, so
        // doing this later creates an observable in-memory-only window.
        if let Ok(path) = std::env::var("PROXIMADB_CORPUS_VERSION_PATH") {
            if !path.is_empty() {
                let store = std::sync::Arc::new(crate::catalog::FileSystemCorpusVersionStore::new(
                    path.clone(),
                ));
                let attached = crate::catalog::CorpusVersionRegistry::init_global_with_store(store);
                if attached {
                    let loaded = crate::catalog::CorpusVersionRegistry::global()
                        .hydrate_from_store()
                        .await;
                    tracing::info!(
                        path = %path,
                        rows = loaded,
                        "✅ corpus_version durability wired (FileSystemCorpusVersionStore)"
                    );
                } else {
                    tracing::warn!(
                        path = %path,
                        "corpus_version durable store was already attached; refusing replacement"
                    );
                }
            }
        } else {
            tracing::info!(
                "PROXIMADB_CORPUS_VERSION_PATH unset; corpus_version registry \
                 is in-memory only (restart resets all versions)"
            );
        }

        // Step 1: Create metrics collector first
        tracing::debug!("🔧 ProximaDB::new - Creating metrics collector...");
        let metrics_collector = Arc::new(crate::metrics::UnifiedMetricsCollector::new());
        // Register the embedding-precision metric family against its
        // dedicated registry. Idempotent (OnceLock-backed). The
        // `/metrics/prometheus` endpoint appends this registry's text
        // to the legacy exporter output, so the
        // `proximadb_embedding_precision_canonical_bytes` family
        // surfaces on every scrape. Without this call, the family
        // exists in code but is invisible to operators.
        crate::observability::precision_metrics::init_precision_metrics();
        tracing::debug!("✅ ProximaDB::new - Metrics collector created successfully");

        // Step 2: Initialize SharedServices with orchestrator
        tracing::info!("🌐 ProximaDB::new - Initializing SharedServices...");

        // Build global Cross-Cache Orchestrator from [cache] or fallback to storage.cache_size_mb
        use crate::storage::cache::orchestrator::CrossCacheOrchestrator;
        let cache_config = config.cache.clone().unwrap_or_default();
        let cache_budget_mb = cache_config.total_memory_mb;
        let mut orchestrator =
            CrossCacheOrchestrator::new((cache_budget_mb * 1024 * 1024) as usize);

        // Start cache eviction service if enabled
        if cache_config.eviction.enabled {
            // Convert our config to the cache eviction config format
            orchestrator.start_eviction_service(None); // Deferred: Wire up eviction config conversion
        }

        // Start cache warming service if enabled (disabled by default)
        if cache_config.enable_warming {
            info!("Cache warming enabled via configuration");
            // Convert our config to the cache warming config format
            orchestrator.start_warming_service(None); // Deferred: Wire up warming config conversion
        } else {
            info!("Cache warming disabled (default)");
        }

        let orchestrator = Arc::new(orchestrator);

        // Start periodic memory rebalancing service if enabled
        if cache_config.rebalancing.enabled {
            info!(
                "Starting cache memory rebalancing service (interval: {}s)",
                cache_config.rebalancing.interval_seconds
            );
            orchestrator.clone().start_rebalancing_service();
        }
        CrossCacheOrchestrator::register_global(orchestrator.clone());

        // Co-design T2.2: wire the io_trace flush to feed the trace-driven cache
        // sizing loop, so footer miss rate and GET fragmentation drive cache budget.
        // High footer miss rate → grow footer cache; fragmented GETs → coalesce tuning.
        crate::storage::cache::orchestrator::CrossCacheOrchestrator::install_trace_driven_sizing(
            orchestrator.clone(),
        );

        // T1.1: initialize global fusion metrics for cross-modal fusion observability.
        // Metrics are emitted to Prometheus for production monitoring and cost model
        // calibration (fusion_latency_seconds, sources_fused, sources_skipped).
        crate::metrics::fusion::init_global_fusion_metrics();

        // Co-design C4: wire the io_trace flush to feed the trace-driven route
        // cost model, so completed routed queries teach the ComputeScheduler the
        // measured cost of each (shape-class, backend). Observe-mode by default —
        // the live-override slice is landed but flag-gated
        // (PROXIMADB_ROUTE_COST_OVERRIDE, default OFF; enablement gated by
        // TD-ROUTE-1's capability-aware candidate fix).
        crate::query::route_cost_model::install_route_cost_observer();
        // Make the SECOND object-storage stack observable. `ProximaObjectStore`
        // (graph cold payloads/segments, the Iceberg bridge, RangedSegmentReader)
        // goes straight to upstream `object_store` and recorded nothing, so every
        // per-query I/O figure this system reported covered only the FileSystem
        // stack — i.e. vector and relational. Installed once here; absent, the
        // hook is inert and behaviour is unchanged.
        proximadb_object_store::install_io_recorder(std::sync::Arc::new(
            crate::storage::persistence::filesystem::IoTraceObjectStoreRecorder,
        ));
        // Warm the cost model from persisted cells so the measured routing
        // history survives restarts (behavior unchanged while override is off).
        crate::query::route_cost_model::load_persisted_cost_model(&config.server.data_dir);
        // ADR-030: wire the io_trace flush to the always-on per-tenant billing
        // meters (KRU read-compute from the snapshot's per-engine compute_ms).
        // Same snapshot as the route observer above — billed bytes/ms cannot
        // diverge from the cost model's. Always-on (billing is never gated).
        crate::metrics::consumption_metrics::install_billing_observer();
        // TD-161: arm the external OTLP metering push (ADR-027 dual-sink push half).
        // Inert unless `PROXIMADB_OTLP_ENDPOINT` is set (and no-op when the
        // `otlp-metering` feature is compiled out). Must run inside the runtime —
        // the grpc-tonic exporter's reader is driven on it.
        crate::observability::metering_otlp::init_from_env();
        // TD-TRACE-2 / ADR-066: durable io_trace ETL sink. A SEPARATE, default-OFF
        // observer (the billing observer above stays always-on/never-gated, ADR-027).
        // Installed only when `[observability.io_trace_sink]` resolves enabled (env
        // `PROXIMADB_IO_TRACE_SINK` or TOML); `resolve` returns None otherwise.
        if let Some(sink_cfg) = crate::core::config::IoTraceSinkConfig::try_resolve(
            config
                .observability
                .as_ref()
                .and_then(|o| o.io_trace_sink.as_ref()),
        )
        .map_err(anyhow::Error::msg)?
        {
            // S4b: run the background warehouse compactor when both an object-store
            // dispatch target and a compaction interval are configured (the compactor
            // reads the object-store segments). Capture before `sink_cfg` is moved.
            let warehouse_trigger = sink_cfg
                .object_store_uri
                .clone()
                .zip(sink_cfg.warehouse_compaction_interval_s);
            crate::observability::io_trace_sink::install(sink_cfg).map_err(anyhow::Error::msg)?;
            if let Some((uri, interval_s)) = warehouse_trigger {
                crate::observability::io_trace_warehouse::install(uri, interval_s)
                    .map_err(anyhow::Error::msg)?;
            }
        }
        // Co-design C5 (integration): register the tenant→tier Port to read the
        // header-fed tier registry (`X-Tenant-Tier` → `record_store::TENANT_TIERS`),
        // so the per-tenant tier the cache LimitsResolver already sees also drives
        // the cost-multiplier — one tier source, no new authority. Installed here
        // (depends on both catalog + services) to keep catalog free of a services dep.
        crate::catalog::tenant_tier::set_tenant_tier_resolver(Some(Box::new(
            crate::services::record_store::tenant_tier_resolved,
        )));
        // Co-design C5: install the config-driven tier cost-multiplier resolver
        // (Dimension 5, the final Cost(q) term) into the route cost model. Reads
        // per-tier multipliers from the tier-config overlay (control-plane policy
        // values) via the tenant→tier Port. Default-inert: neutral 1.0 until a
        // tenant→tier resolver is registered and non-neutral multipliers shipped.
        crate::catalog::tenant_tier::install_tier_cost_multiplier_resolver();
        // Co-design C3/KOU: derive this deployment's object-store read-egress
        // locality from the storage URL (provider) + control-plane-stamped
        // PROXIMADB_COMPUTE_REGION/PROXIMADB_STORE_REGION, and install it as the
        // process-wide locality the egress meter uses. Default Local (free, inert)
        // when unset; logs the derived locality. Turns B1's derivation live.
        {
            let storage_url = config
                .storage
                .storage_locations
                .first()
                .map(|loc| loc.url.as_str())
                .unwrap_or("");
            let placement = crate::metrics::consumption_metrics::placement_from_env(storage_url);
            crate::metrics::consumption_metrics::install_placement(&placement);
        }
        // Co-design KOU result-egress: load the client-locality scope map (CIDR→
        // locality, control-plane-authored via PROXIMADB_NETWORK_SCOPE_MAP).
        // Default-empty → clients classify as Unknown (safe). Consumed by the
        // gateway/result-egress path; the classifier (`classify_client`) is live.
        crate::metrics::consumption_metrics::install_network_scope_map(None);

        let (shared_services, collection_service) = network::multi_server::SharedServices::new(
            Some(metrics_collector.clone()),
            &config.storage,
            Some(orchestrator.clone()),
            Some(&config),
            network::multi_server::ServiceProfile::Server,
        )
        .await?;
        tracing::info!("✅ ProximaDB::new - SharedServices created with unified CollectionService");

        // ADR-030 KSU: per-tenant resident-storage metering on two decoupled
        // cadences. KSU is an *accrual* (`bytes·seconds`) over a slow-moving stock
        // (resident bytes only change on flush/compaction, minutes-to-hours apart),
        // so neither cadence needs to be fast — and the *durable* one must be slow,
        // since metering's own object-store writes are billable I/O (the dominant
        // cost term this co-design effort minimizes). Both are server-lifetime tokio
        // tasks (dropped on shutdown), first tick delayed one interval, best-effort.
        //
        //   1. **Prometheus level gauge** — in-memory `.set()` of the resident level
        //      that Prometheus integrates downstream. Default 5min (env
        //      `PROXIMADB_METERING_KSU_GAUGE_INTERVAL_SECS`); storage doesn't move faster, so
        //      a per-minute poll only multiplied the list-collections work.
        {
            let cs = collection_service.clone();
            let interval_secs = std::env::var("PROXIMADB_METERING_KSU_GAUGE_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|s| *s > 0)
                .unwrap_or(300);
            tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(std::time::Duration::from_secs(interval_secs));
                ticker.tick().await; // consume the immediate first tick
                loop {
                    ticker.tick().await;
                    if let Ok(collections) = cs.list_collections().await {
                        crate::metrics::consumption_metrics::record_storage_snapshot(&collections);
                    }
                }
            });
        }

        //   2. **Durable `_metering` sink (TD-161)** — coalesced **hourly** snapshot
        //      persisted to each tenant's object-store `_metering/` subtree
        //      (ADR-027 dual-sink; the differentiator AnvaiOps prices). Default 1h
        //      (env `PROXIMADB_METERING_FLUSH_INTERVAL_SECS`) matches AWS byte-hour
        //      billing granularity (~24 PUTs/tenant/day, not ~1440). Wholly
        //      best-effort: a missing storage location or filesystem-init failure
        //      disables only the durable sink — never startup or the gauge above.
        {
            let cs = collection_service.clone();
            let base_url = config.storage.storage_urls().into_iter().next();
            let interval_secs = std::env::var("PROXIMADB_METERING_FLUSH_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|s| *s > 0)
                .unwrap_or(3600);
            tokio::spawn(async move {
                let Some(base_url) = base_url else {
                    tracing::warn!(
                        "TD-161: no storage location configured; durable metering writer disabled"
                    );
                    return;
                };
                let factory =
                    match storage::persistence::filesystem::FilesystemFactory::create_default()
                        .await
                    {
                        Ok(f) => std::sync::Arc::new(f),
                        Err(e) => {
                            tracing::warn!(
                                "TD-161: filesystem init failed; durable metering disabled: {e}"
                            );
                            return;
                        }
                    };
                let writer =
                    crate::metrics::metering_writer::DurableMeteringWriter::new(factory, base_url);
                let mut ticker =
                    tokio::time::interval(std::time::Duration::from_secs(interval_secs));
                ticker.tick().await; // consume the immediate first tick
                loop {
                    ticker.tick().await;
                    if let Ok(collections) = cs.list_collections().await {
                        let usage = crate::metrics::consumption_metrics::aggregate_tenant_storage(
                            &collections,
                        );
                        // Account-less legacy `data/{tenant}/` render: the snapshot
                        // source (`Collection.config.owner`) carries no account id.
                        let now_secs = chrono::Utc::now().timestamp();
                        writer.write_storage_snapshot(&usage, None, now_secs).await;
                        // Same snapshot → the external OTLP push (inert unless
                        // configured): the durable record and pushed level cannot
                        // diverge by construction.
                        crate::observability::metering_otlp::record_storage_snapshot(&usage);
                    }
                }
            });
        }

        // Step 3: Initialize global WAL manifest
        tracing::info!("🌐 ProximaDB::new - Initializing global WAL manifest...");
        let mut wal_config = config.storage.wal_config.to_engine_config();
        // Set data directories from storage_locations
        wal_config.multi_disk.data_directories = config.storage.storage_urls();

        // Initialize global WAL manifest using WALConfig
        let _manifest_service = storage::persistence::write_ahead_log::manifest::init(&wal_config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to init WAL manifest: {}", e))?;
        tracing::info!("✅ ProximaDB::new - Global WAL manifest initialized");

        // Step 4: Set global catalog BEFORE creating StorageEngine.
        // The catalog is the sole authority for WAL/recovery collection
        // resolution, so WAL pool instances resolve collection paths through it.
        if let Some(catalog_manager) = collection_service.catalog_manager() {
            storage::persistence::write_ahead_log::set_global_catalog(catalog_manager).await;
            tracing::info!("✅ ProximaDB::new - Global catalog set for WAL/recovery");
        }

        // Step 5: Initialize the storage engine (SST/VIPER/etc)
        tracing::info!("🌐 ProximaDB::new - Creating StorageEngine...");
        let mut storage_engine =
            storage::StorageEngine::new_without_collection_service(config.storage.clone())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create storage engine: {}", e))?;

        // Wire CanonicalPrecisionResolver into Compaction. The catalog
        // already exists (constructed inside SharedServices::new above)
        // but the storage engine was built without knowing about it;
        // `Compaction::set_precision_resolver` is the post-construction
        // hook that closes the gap. When no default catalog is
        // registered (degraded boot), the resolver isn't wired and
        // compaction keeps producing fp32 records.
        match shared_services.catalog_manager.default_catalog().await {
            Ok(catalog) => {
                let resolver_cache =
                    Arc::new(proximadb_catalog::cache::CatalogCache::new(10_000, 60));
                let resolver = Arc::new(
                    proximadb_catalog::canonical_precision::CanonicalPrecisionResolver::new(
                        catalog,
                        resolver_cache,
                    ),
                );
                // TD-PRECISE-GLOBAL: arm the process-global resolver so EVERY
                // Compaction stamps precision_hint — including the per-collection
                // SstEngines the boot path can't individually wire. The former
                // `storage_engine.compaction_manager().set_precision_resolver(..)`
                // targeted the StorageEngine's idle instance, so fp16/bf16/int8
                // collections silently degraded to fp32 at compaction.
                crate::storage::engines::sst::set_global_precision_resolver(resolver.clone());
                tracing::info!(
                    "✅ Global compaction precision resolver armed — fp16/bf16/int8 \
                     collections preserve precision through compaction"
                );
                // Also wire the resolver into the REST/gRPC vector
                // batch path so direct inserts coerce to the
                // collection's canonical precision (matches what the
                // queue drainer does via BulkLoadDrainerSink).
                shared_services.record_ops.set_precision_resolver(resolver);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "compaction bootstrap: no default catalog; precision lookup \
                     disabled, compaction rewrites will produce fp32 records \
                     regardless of collection canonical precision"
                );
            }
        }

        // A6: inject the storage-write fence (over the SAME PartitionLeaseManager
        // the network write-gates use) into the storage engine's shutdown flush
        // path. Post-construction, mirroring set_precision_resolver: the engine is
        // built before SharedServices/the lease stack exist. Default-OFF; absent
        // lease store ⇒ None ⇒ fail-open. Enforced only with PROXIMADB_WRITE_FENCING=1.
        if let Some(fence) = shared_services.storage_write_fence.clone() {
            storage_engine.set_storage_write_fence(fence);
            tracing::info!(
                "✅ ProximaDB::new - A6 storage-write fence wired to StorageEngine \
                 (default-OFF; set PROXIMADB_WRITE_FENCING=1 to enforce)"
            );
        }

        let storage = Arc::new(RwLock::new(storage_engine));
        tracing::info!("✅ ProximaDB::new - StorageEngine initialized successfully");

        // Step 6: Create multi-server configuration from actual config values
        use std::net::SocketAddr;
        tracing::debug!("🔧 ProximaDB::new - Creating server addresses...");
        // Determine ports: prefer ApiConfig, fall back to ServerConfig
        let rest_port = config.api.rest_port;
        let grpc_port = config.api.grpc_port;

        let rest_addr: SocketAddr = format!("{}:{}", config.server.bind_address, rest_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid REST address: {}", e))?;
        let grpc_addr: SocketAddr = format!("{}:{}", config.server.bind_address, grpc_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid gRPC address: {}", e))?;
        tracing::debug!(
            "🔧 ProximaDB::new - REST address: {}, gRPC address: {}",
            rest_addr,
            grpc_addr
        );

        tracing::debug!("🔧 ProximaDB::new - Building multi-server configuration...");
        let mut builder = network::MultiServerBuilder::custom()
            .http(|h| h.bind_address(rest_addr))
            .grpc(|g| g.bind_address(grpc_addr))
            .with_api_config(config.api.clone())
            .with_data_dir(config.server.data_dir.clone())
            .with_admin_ui_enabled(config.server.admin_ui.enabled)
            .with_admin_ui_refresh(
                config.server.admin_ui.auto_refresh,
                config.server.admin_ui.refresh_interval_seconds,
            );

        // Add TLS configuration if enabled
        if config.api.enable_tls.unwrap_or(false) {
            tracing::debug!("🔧 ProximaDB::new - Adding TLS configuration...");
            if let Some(tls_config) = config.tls.as_ref()
                && let (Some(cert_file), Some(key_file)) =
                    (&tls_config.cert_file, &tls_config.key_file)
            {
                builder = builder.with_tls(cert_file.clone(), key_file.clone());
            }
        }

        tracing::debug!("🔧 ProximaDB::new - Building multi-server config...");
        let mut multi_config = builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create server config: {}", e))?;
        // Honor the optional `[api].pg_port` override so the pgwire
        // listener binds where the operator (or a test fixture) asked.
        // The builder hardcodes 5433; this is the seam test fixtures
        // use to pick a free port without colliding with a real
        // Postgres or a sibling test.
        if let Some(pg_port) = config.api.pg_port {
            multi_config.postgres_config.port = pg_port;
            multi_config.postgres_config.bind_address =
                format!("{}:{}", config.server.bind_address, pg_port)
                    .parse()
                    .map_err(|e| anyhow::anyhow!("Failed to parse postgres bind address: {}", e))?;
        }
        // Propagate `[api].arrow_flight_port` to the Arrow IPC listener. The
        // builder hardcodes 5680, so without this override test fixtures
        // can't pin the Flight listener to a free port. Production configs
        // already default arrow_flight_port to 5680, so this is a no-op for
        // operators who don't override.
        multi_config.arrow_ipc_config.port = config.api.arrow_flight_port;
        multi_config.arrow_ipc_config.bind_address = format!(
            "{}:{}",
            config.server.bind_address, config.api.arrow_flight_port
        )
        .parse()
        .map_err(|e| anyhow::anyhow!("Failed to parse arrow flight bind address: {}", e))?;

        // Portless ("embedded") transport: when `[api].transport = "uds"`, bind
        // the REST / gRPC / Arrow Flight surfaces to Unix-domain sockets under
        // `[api].socket_dir` instead of TCP ports, and disable the pgwire
        // listener (a standard PG TCP driver) so the process opens *no* TCP
        // listener at all. Mixed-read-safe: the default ("tcp") leaves every
        // existing deployment unchanged.
        if config.api.transport.eq_ignore_ascii_case("uds") {
            let socket_dir = config.api.socket_dir.clone().ok_or_else(|| {
                anyhow::anyhow!("[api].transport = \"uds\" requires [api].socket_dir to be set")
            })?;
            let socket_dir = std::path::PathBuf::from(socket_dir);
            std::fs::create_dir_all(&socket_dir).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create UDS socket dir {}: {}",
                    socket_dir.display(),
                    e
                )
            })?;
            tracing::info!(
                socket_dir = %socket_dir.display(),
                "🔌 Portless mode: binding REST/gRPC/Arrow Flight to Unix-domain sockets (no TCP ports); pgwire disabled"
            );
            multi_config.uds_socket_dir = Some(socket_dir);
            // pgwire has no UDS surface here; embedded clients use REST/gRPC/Flight.
            multi_config.postgres_config.enable_postgres = false;
        }
        tracing::debug!("✅ ProximaDB::new - Multi-server config created successfully");

        // Initialize security coordinator if configured
        let security_config = config.security.clone().filter(|sec_cfg| sec_cfg.enabled);
        let security = initialize_configured_security(
            config.security.clone(),
            Some(config.server.data_dir.join("identity")),
        )
        .await?;

        // Initialize LLM engine if configured
        let llm_engine = if let Some(llm_cfg) = config.llm.clone() {
            match crate::ai::llm_integration::LLMIntegrationEngine::new(llm_cfg).await {
                Ok(engine) => Some(Arc::new(engine)),
                Err(err) => {
                    tracing::warn!("LLM engine initialization failed: {:?}", err);
                    None
                }
            }
        } else {
            None
        };

        // Create MultiServer with SharedServices (network orchestrator)
        tracing::debug!("🔧 ProximaDB::new - Creating MultiServer...");
        let rest_auth_enabled = security_config
            .as_ref()
            .is_some_and(|sec_cfg| sec_cfg.authentication.enabled);
        // TD-AUTH-FC (D3): posture B (security enabled + authentication
        // disabled) now ENFORCES credentials on every network surface — the
        // coordinator is threaded to REST/gRPC/Flight/MCP/pgwire regardless
        // of the authentication flag, so unauthenticated calls are rejected
        // instead of served. Operators who want the old fail-open dev posture
        // should disable `[security]` entirely.
        if security.is_some() && !rest_auth_enabled {
            tracing::warn!(
                "security subsystem active with authentication.enabled = false: ALL network \
                 surfaces (REST/gRPC/Arrow Flight/MCP) now enforce credentials whenever the \
                 security coordinator exists (TD-AUTH-FC). To run unauthenticated, disable \
                 [security] entirely."
            );
        }
        let tenant_deployment_mode =
            resolve_server_tenant_mode(&config.server.tenant, security_config.as_ref())?;
        // Capture the drainer's record + vector services BEFORE shared_services
        // is moved into MultiServer below. The async-ingest drainer needs them
        // as the production DrainerInsertSink target via the BulkLoadDrainerSink
        // wrapper (TD-104 S3-e: passed directly, no root UnifiedHandlers handle).
        let record_ops_for_drainer = shared_services.record_ops.clone();
        let vector_ops_for_drainer = shared_services.vector_operations_service.clone();
        // Capture the catalog_manager Arc the same way — the drainer
        // needs it to construct CanonicalPrecisionResolver so per-payload
        // canonical_embedding_precision lookup populates
        // EmbeddedRecord.target_precision (fp16 ingest end-to-end).
        let catalog_manager_for_drainer = shared_services.catalog_manager.clone();

        // Open the queue subsystem BEFORE MultiServer so its Arc can
        // thread into AppState (so the v3 `/documents?mode=async` REST
        // handler routes through `producer.send`). The drainer is
        // spawned AFTER MultiServer because it needs the same handlers
        // the REST/gRPC stacks consume.
        //
        // Configuration is resolved through `QueueRuntimeConfig::resolve`
        // which folds env > TOML > defaults (see `core::config` for the
        // documented precedence pyramid). Passing `config.queue.as_ref()`
        // means the TOML `[queue]` section is the canonical source and
        // env vars exist for emergency per-pod overrides.
        let resolved_queue = core::config::QueueRuntimeConfig::resolve(config.queue.as_ref());
        let queue_client = if let Some(ref rq) = resolved_queue {
            Some(open_queue_client_from_resolved(rq).await?)
        } else {
            tracing::info!(
                "no [queue] section and PROXIMADB_QUEUE_ROOT unset; async ingest \
                 will degrade to inline embed"
            );
            None
        };

        // TD-TENANT-1: resolve the deployment-effective bare tenant-assertion
        // trust policy ONCE (env > [security.tenant] header_trust > deployment-
        // mode preset) and thread it into every network surface via MultiServer.
        let (tenant_header_trust, trust_warning) = proximadb_tenant::HeaderTrustPolicy::effective(
            match &tenant_deployment_mode {
                proximadb_tenant::TenantDeploymentMode::MultiTenant => {
                    proximadb_tenant::HeaderTrustPolicy::AuthenticatedOnly
                }
                proximadb_tenant::TenantDeploymentMode::SingleTenant { .. } => {
                    proximadb_tenant::HeaderTrustPolicy::Open
                }
            },
            config
                .security
                .as_ref()
                .and_then(|security| security.tenant.header_trust),
        );
        if let Some(warning) = trust_warning {
            tracing::warn!("{warning}");
        }
        tracing::info!(policy = %tenant_header_trust, "🔐 tenant header-trust policy (TD-TENANT-1)");

        // ADR-0053 W8: resolve the tier-claim gate identically (env
        // PROXIMADB_TIER_HEADER_TRUST > [security.tenant] tier_header_trust >
        // deployment-mode preset). The gates are separate because a
        // deployment may trust its gateway's tenant ids without (yet)
        // trusting its tier stamps.
        let (tier_header_trust, tier_trust_warning) =
            proximadb_tenant::HeaderTrustPolicy::effective_tier(
                match &tenant_deployment_mode {
                    proximadb_tenant::TenantDeploymentMode::MultiTenant => {
                        proximadb_tenant::HeaderTrustPolicy::AuthenticatedOnly
                    }
                    proximadb_tenant::TenantDeploymentMode::SingleTenant { .. } => {
                        proximadb_tenant::HeaderTrustPolicy::Open
                    }
                },
                config
                    .security
                    .as_ref()
                    .and_then(|security| security.tenant.tier_header_trust),
            );
        if let Some(warning) = tier_trust_warning {
            tracing::warn!("{warning}");
        }
        tracing::info!(policy = %tier_header_trust, "🔐 tier-claim trust policy (ADR-0053 W8)");

        let multi_server = network::MultiServer::new_with_queue_client(
            multi_config,
            shared_services,
            security.clone(),
            rest_auth_enabled,
            tenant_deployment_mode,
            llm_engine,
            queue_client.clone(),
        )
        .with_tenant_header_trust(tenant_header_trust)
        .with_tier_header_trust(tier_header_trust);
        tracing::debug!("✅ ProximaDB::new - MultiServer created");

        // Phase 2H wiring (drainer half): spawn the drainer only when
        // queue_client is Some — otherwise async ingest degrades to
        // inline embed in the REST handler.
        let drainer = if let Some(ref qc) = queue_client {
            // resolved_queue is always Some when queue_client is Some
            // (they're built from the same condition), so unwrapping
            // here is safe.
            #[allow(clippy::expect_used)]
            let rq = resolved_queue
                .as_ref()
                .expect("queue_client → resolved_queue");
            // BulkLoader's per-engine bulk-load fallback uses this
            // as the storage root when an override needs a base_path.
            // The canonical source is `config.storage.storage_locations[0].url`
            // — typically `adls://...`, `s3://...`, or `file:///tmp/proximadb`
            // depending on deployment; reads from the TOML loaded at startup.
            let default_storage_root = config
                .storage
                .storage_urls()
                .into_iter()
                .next()
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "config.storage.storage_locations is empty; \
                         BulkLoader falling back to data_dir for engine overrides"
                    );
                    format!("file://{}", config.server.data_dir.display())
                });
            Some(
                spawn_embedding_drainer_from_resolved(
                    qc.clone(),
                    record_ops_for_drainer,
                    vector_ops_for_drainer,
                    catalog_manager_for_drainer,
                    rq,
                    default_storage_root,
                )
                .await?,
            )
        } else {
            None
        };

        Ok(Self {
            storage,
            multi_server: Some(multi_server),
            _config: config,
            security,
            rl_checkpoint_handle: None,
            rl_policy_path: None,
            queue_client,
            drainer,
        })
    }

    /// Expose the queue client to startup code that wires `AppState`
    /// for the v3 async-ingest producer side. Returns `None` when the
    /// queue subsystem isn't enabled — the REST handler then takes the
    /// inline-embed fallback path.
    pub fn queue_client(&self) -> Option<Arc<proximadb_queue::QueueClient>> {
        self.queue_client.clone()
    }

    /// The shared canonical WAL-backed record store unified across ALL surfaces (REST/gRPC
    /// `DmlService` + pgwire direct writes both route relational tables through this single
    /// instance). `None` when direct record writes aren't configured. Exposed so embedded
    /// and integration consumers can observe the converged relational state — and the CDC
    /// change-feed — regardless of which protocol wrote the data.
    pub fn canonical_record_store(
        &self,
    ) -> Option<Arc<crate::services::record_store::DirectWalTableRecordStore>> {
        self.multi_server
            .as_ref()
            .and_then(|ms| ms.shared_services.canonical_record_store.clone())
    }

    /// Start all database services (network listeners, background tasks, WAL recovery).
    pub async fn start(&mut self) -> anyhow::Result<()> {
        tracing::info!("🚀 ProximaDB::start - Starting database services...");

        // TD-COMPACT-9: seed the coarse-probe (IVF) settings AUTHORITATIVELY from
        // the loaded server config. Production `SstEngine::new()` uses
        // `SstConfig::default()` (coarse_probe = None), so without this the
        // `[storage.sst_config.coarse_probe]` TOML surface (nprobe_multiplier,
        // enable_read_probe/write_train) was silently ignored. The overwritable
        // seed (a `Some` config always wins) makes the TOML effective regardless
        // of engine-construction order; env vars still override at call time.
        crate::storage::engines::sst::segment_format::init_coarse_probe_settings(
            self._config
                .storage
                .sst_config
                .as_ref()
                .and_then(|sst| sst.coarse_probe.as_ref()),
        );

        // TD-IOBUDGET-1: register per-location I/O budgets authoritatively from
        // the loaded config — same overwritable-seed shape as the coarse probe
        // above. Validation is fail-closed (invalid budgets abort startup) and
        // the INFO dump is the operator's audit artifact of the effective
        // ranged-GET geometry per storage location.
        self._config.storage.register_io_budgets()?;

        // Step 1: Start the storage engine. It restores the catalog, registers
        // collection storage engines, and replays WAL in that dependency order.
        tracing::info!(
            "📦 ProximaDB::start - Step 1: Starting storage engine and durable recovery..."
        );
        {
            let mut storage = self.storage.write().await;
            storage
                .start()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to start storage engine: {}", e))?;
        }
        tracing::info!("✅ ProximaDB::start - Storage engine started, catalog and WAL recovered");

        // Step 2: Recover graphs from snapshots + WAL
        tracing::info!(
            "🌳 ProximaDB::start - Step 2: Recovering graphs from persistent storage..."
        );
        if let Some(ref multi_server) = self.multi_server {
            match multi_server
                .shared_services
                .graph_service
                .recover_all_graphs()
                .await
            {
                Ok(()) => {
                    tracing::info!("✅ ProximaDB::start - Graphs recovered successfully");
                }
                Err(e) => {
                    tracing::warn!(
                        "⚠️  ProximaDB::start - Graph recovery failed (continuing anyway): {}",
                        e
                    );
                }
            }
        }

        // Step 3: Recover assignments from collection metadata
        tracing::info!(
            "🗺️ ProximaDB::start - Step 3: Recovering assignments from collection metadata..."
        );
        tracing::info!(
            "✅ ProximaDB::start - Assignment recovery completed (or skipped if no service)"
        );

        // Step 4: Recover vectors from write buffer (in-memory data)
        tracing::info!("🔄 ProximaDB::start - Step 4: Recovering vectors from write buffer...");
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .recover_vectors_from_write_buffer(&self.storage)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("Failed to recover vectors from write buffer: {}", e)
                })?;
        }

        // TD-CACHE-1 S2: background restart warming — replay per-tenant
        // warm-set manifests (demand-proven hot ranges from the previous
        // instance) + control-invariants prefill for the referenced segments.
        // Background task: boot latency unaffected; tenants without manifests
        // cost one negative read each. Gated by PROXIMADB_CACHE_PREFILL.
        {
            let base_url = self._config.storage.storage_urls().into_iter().next();
            if let Some(base_url) = base_url {
                tokio::spawn(async move {
                    let factory = match crate::storage::persistence::filesystem::FilesystemFactory::create_default().await {
                        Ok(f) => std::sync::Arc::new(f),
                        Err(e) => {
                            tracing::warn!("cache warming skipped (fs init): {e}");
                            return;
                        }
                    };
                    // Small grace so engine arming (which registers the cache
                    // handles) has completed before replay looks them up.
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    // Tenants = catalog-tagged tenant ids ∪ the single-tenant
                    // default — manifests are keyed by the REQUEST tenant, which
                    // is "default" for untagged/single-tenant deployments.
                    let mut tenant_set: std::collections::HashSet<String> =
                        crate::storage::persistence::write_ahead_log::list_collections_from_catalog()
                            .await
                            .iter()
                            .filter_map(proximadb_tenant::tenant_id_of)
                            .collect();
                    tenant_set.insert("default".to_string());
                    let tenants: Vec<String> = tenant_set.into_iter().collect();
                    crate::storage::engines::sst::warming::boot_warm(factory, base_url, tenants)
                        .await;
                });
            }
        }

        // TD-CACHE-1 S4: eviction-notice fast-drain — opt-in IMDS watch
        // (`PROXIMADB_EVICTION_WATCH=azure|aws|gcp`). On the Spot notice
        // (which precedes SIGTERM and may be followed by a hard kill),
        // persist the warm manifests and flush the memtable early.
        if let Ok(provider) = std::env::var("PROXIMADB_EVICTION_WATCH") {
            let provider = provider.trim().to_ascii_lowercase();
            if !provider.is_empty() && provider != "0" && provider != "off" {
                let base_url = self._config.storage.storage_urls().into_iter().next();
                if let Some(base_url) = base_url {
                    let storage = self.storage.clone();
                    tokio::spawn(async move {
                        let factory = match crate::storage::persistence::filesystem::FilesystemFactory::create_default().await {
                            Ok(f) => std::sync::Arc::new(f),
                            Err(e) => {
                                tracing::warn!("eviction watch skipped (fs init): {e}");
                                return;
                            }
                        };
                        let flush: std::sync::Arc<
                            dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync,
                        > = std::sync::Arc::new(move || {
                            let storage = storage.clone();
                            Box::pin(async move {
                                let engine = storage.read().await;
                                match engine.flush_memtable_to_storage().await {
                                    Ok(r) => tracing::info!(
                                        collections = r.collections_flushed,
                                        vectors = r.total_vectors_flushed,
                                        "fast-drain memtable flush complete"
                                    ),
                                    Err(e) => {
                                        tracing::warn!("fast-drain memtable flush failed: {e}")
                                    }
                                }
                            })
                        });
                        crate::storage::engines::sst::warming::eviction_watch(
                            provider, factory, base_url, flush,
                        )
                        .await;
                    });
                }
            }
        }

        // Step 5: Initialize RL Query Planner (if enabled)
        tracing::info!("🎯 ProximaDB::start - Step 5: Initializing RL Query Planner...");
        self.init_rl_planner().await?;

        // Step 6: Start multi-server (HTTP and gRPC on separate ports)
        tracing::info!(
            "🌐 ProximaDB::start - Step 6: Starting multi-server (gRPC:5679 + REST:5678)..."
        );
        if let Some(ref mut multi_server) = self.multi_server {
            multi_server
                .start()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to start multi-server: {}", e))?;
        }
        tracing::info!("✅ ProximaDB::start - Multi-server started successfully");

        // Periodic graph WAL checkpoint (gated on the same canonical-replay
        // scope as the checkpoint machinery itself). Without a cadence,
        // `flush_wal` runs only at shutdown, so the graph WAL grows for the
        // whole session and the shutdown snapshot is the first-ever (largest
        // possible) one. Cadence 60s with an edge-epoch dirty check; every
        // 10th tick runs unconditionally to cover node-only mutations, which
        // do not bump the edge epoch.
        if crate::storage::persistence::write_ahead_log::wal_operations::canonical_replay_scope_enabled()
            && let Some(ref multi_server) = self.multi_server
        {
            let graph_service = multi_server.shared_services.graph_service.clone();
            let (tx, mut rx) = tokio::sync::watch::channel(false);
            crate::services::shutdown_registry::register("graph-wal-checkpoint", tx);
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(tokio::time::Duration::from_secs(60));
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                let mut last_epochs: std::collections::HashMap<String, u64> =
                    std::collections::HashMap::new();
                let mut ticks: u64 = 0;
                loop {
                    tokio::select! {
                        _ = tick.tick() => {}
                        _ = rx.changed() => break,
                    }
                    ticks += 1;
                    let force = ticks.is_multiple_of(10);
                    let graphs = match graph_service.list_graphs().await {
                        Ok(g) => g,
                        Err(e) => {
                            tracing::debug!("graph checkpoint tick: list_graphs failed: {e}");
                            continue;
                        }
                    };
                    for graph_id in graphs {
                        let epoch = graph_service.edge_epoch(&graph_id).0;
                        let dirty = last_epochs.get(&graph_id) != Some(&epoch);
                        if !(dirty || force) {
                            continue;
                        }
                        match graph_service.flush_wal(&graph_id).await {
                            Ok(()) => {
                                last_epochs.insert(graph_id, epoch);
                            }
                            Err(e) => tracing::warn!(
                                "periodic graph WAL checkpoint failed for {graph_id}: {e:#}"
                            ),
                        }
                    }
                }
                tracing::debug!("graph WAL checkpoint loop exited");
            });
            tracing::info!("🕒 graph WAL checkpoint loop armed (60s, scope-gated)");
        }

        tracing::info!(
            "🎉 ProximaDB::start - Database startup complete with full persistence recovery!"
        );
        Ok(())
    }

    /// Shutdown the database instance gracefully.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        info!("Graceful shutdown requested");

        // TD-LIFECYCLE-1: signal every registered background loop FIRST so
        // cooperative exits overlap the rest of the shutdown sequence (no
        // in-flight discovery/observer pass outlives the runtime).
        let fired = crate::services::shutdown_registry::fire_all();
        if fired > 0 {
            tracing::info!(loops = fired, "background loops signaled to stop");
        }

        // Persist the learned route cost model so measured history survives the
        // restart (best-effort; off the hot path, on graceful shutdown only).
        crate::query::route_cost_model::persist_cost_model(&self._config.server.data_dir);

        // TD-TRACE-2: flush + stop the durable io_trace sink so buffered trace
        // segments are sealed to disk (no-op if the sink was not installed).
        crate::observability::io_trace_sink::shutdown().await;
        crate::observability::io_trace_warehouse::shutdown().await;

        // 1a. Stop the async-ingest drainer first so it stops consuming
        //     before we shut down the storage layer it inserts into.
        //     The queue subsystem itself is stopped after — its
        //     background tasks (uploader, reaper) need a few ticks to
        //     finish in-flight work.
        if let Some((handle, tx)) = self.drainer.take() {
            tracing::info!("Stopping embedding drainer...");
            let _ = tx.send(());
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
        if let Some(client) = self.queue_client.take() {
            tracing::info!("Stopping queue subsystem...");
            if let Err(e) = client.shutdown().await {
                tracing::warn!("queue shutdown error: {}", e);
            }
        }

        // 1. Stop RL planner checkpoint task and persist policy
        tracing::info!("Stopping RL Query Planner...");
        self.shutdown_rl_planner().await;

        // 2. Flush graph WAL for all graphs before shutdown
        tracing::info!("Flushing graph WAL for all graphs...");
        if let Some(ref multi_server) = self.multi_server {
            // 15s (was 5s): with PROXIMADB_GRAPH_CANONICAL_REPLAY_SCOPE on,
            // each graph's flush also writes a full snapshot and truncates
            // reclaimed WAL segments.
            match tokio::time::timeout(tokio::time::Duration::from_secs(15), async {
                let graphs = multi_server
                    .shared_services
                    .graph_service
                    .list_graphs()
                    .await?;
                let mut flushed = 0;
                for graph_id in graphs {
                    if let Err(e) = multi_server
                        .shared_services
                        .graph_service
                        .flush_wal(&graph_id)
                        .await
                    {
                        tracing::warn!("Failed to flush WAL for graph {}: {}", graph_id, e);
                    } else {
                        flushed += 1;
                    }
                }
                tracing::debug!("Flushed WAL for {} graphs", flushed);
                Ok::<_, anyhow::Error>(())
            })
            .await
            {
                Ok(Ok(())) => tracing::debug!("Graph WAL flush complete"),
                Ok(Err(e)) => tracing::warn!("Graph WAL flush error: {}", e),
                Err(_) => tracing::warn!("Graph WAL flush timeout - forcing continuation"),
            }
        }

        // TD-FLUSH-4: await in-flight inline size-flushes FIRST — a triggered
        // materialize takes tens of seconds (measured 43.9s at 884k entries)
        // and must complete + rename into place before teardown; killing it
        // strands a __flush/ staging file that startup cleanup deletes.
        let stragglers = crate::storage::persistence::write_ahead_log::drain_inline_flushes(
            std::time::Duration::from_secs(90),
        )
        .await;
        if stragglers > 0 {
            tracing::warn!(
                stragglers,
                "in-flight flush(es) did not finish before shutdown"
            );
        }

        // TD-CACHE-1 S2: persist per-tenant warm-set manifests so the next
        // boot (Spot replacement) replays the measured hot set instead of
        // paying the cold herd. Best-effort with a hard timeout — shutdown
        // never blocks on the object store.
        {
            let base_url = self._config.storage.storage_urls().into_iter().next();
            if let Some(base_url) = base_url {
                match crate::storage::persistence::filesystem::FilesystemFactory::create_default()
                    .await
                {
                    Ok(factory) => {
                        let factory = std::sync::Arc::new(factory);
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(10),
                            crate::storage::engines::sst::warming::emit_warm_manifests(
                                &factory, &base_url,
                            ),
                        )
                        .await
                        {
                            Ok(n) if n > 0 => {
                                tracing::info!("🔥 {n} warm manifest(s) persisted for next boot")
                            }
                            Ok(_) => {}
                            Err(_) => tracing::warn!("warm manifest emit timed out (skipped)"),
                        }
                    }
                    Err(e) => tracing::warn!("warm manifest emit skipped (fs init): {e}"),
                }
            }
        }

        // 3. Shutdown servers
        if let Some(mut multi_server) = self.multi_server.take() {
            multi_server
                .stop()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to stop multi-server: {}", e))?;
        }

        // 4. Stop storage engine with timeout
        // 60s (was 3s): the final flush-on-stop is a real materialize —
        // TD-FLUSH-4 measured 43.9s at 884k entries — and the same function
        // already budgets 90s for drain_inline_flushes. 3s guaranteed the
        // final flush (and with it WAL reclamation via free_wal) was
        // truncated on every non-trivial embedded run. Returns early when
        // the flush is small, which the embedded flush profile now ensures.
        match tokio::time::timeout(tokio::time::Duration::from_secs(60), async {
            let mut storage = self.storage.write().await;
            storage.stop().await
        })
        .await
        {
            Ok(Ok(())) => tracing::debug!("Storage engine stopped"),
            Ok(Err(e)) => tracing::warn!("Storage engine stop error: {}", e),
            Err(_) => tracing::warn!("Storage engine stop timeout"),
        }

        tracing::info!("Server shutdown complete");
        Ok(())
    }

    /// Check if the database instance is healthy.
    pub async fn is_healthy(&self) -> bool {
        if let Some(ref multi_server) = self.multi_server {
            multi_server.status().await.http_running
        } else {
            false
        }
    }

    /// Get current server status including versions and uptime.
    pub async fn server_status(&self) -> Option<network::multi_server::ServerStatus> {
        if let Some(ref multi_server) = self.multi_server {
            Some(multi_server.status().await)
        } else {
            None
        }
    }

    /// Get REST server address
    pub async fn rest_server_address(&self) -> Option<std::net::SocketAddr> {
        if let Some(status) = self.server_status().await {
            status.http_address
        } else {
            None
        }
    }

    /// Get gRPC server address
    pub async fn grpc_server_address(&self) -> Option<std::net::SocketAddr> {
        if let Some(status) = self.server_status().await {
            status.grpc_address
        } else {
            None
        }
    }

    /// Initialize the RL Query Planner from configuration
    async fn init_rl_planner(&mut self) -> anyhow::Result<()> {
        let rl_config = self
            ._config
            .query
            .as_ref()
            .map(|q| q.rl_planner.to_rl_planner_config())
            .unwrap_or_default();

        if !rl_config.enabled {
            tracing::info!("RL Query Planner is disabled in configuration");
            return Ok(());
        }

        let policy_path = rl_config.log_path.clone().unwrap_or_else(|| {
            format!("{}/rl_policy.json", self._config.server.data_dir.display())
        });
        self.rl_policy_path = Some(policy_path.clone());

        query::rl_planner::init_rl_planner(rl_config.clone());
        tracing::info!("✅ RL Query Planner initialized");

        if let Some(planner) = query::rl_planner::get_rl_planner() {
            if std::path::Path::new(&policy_path).exists() {
                planner.load_policy(&policy_path).await?;
                tracing::info!("✅ RL policy loaded from {}", policy_path);
            }

            let checkpoint_path = policy_path.clone();
            let checkpoint_handle = tokio::spawn(async move {
                let interval = std::time::Duration::from_secs(RL_CHECKPOINT_INTERVAL_SECS);
                let mut ticker = tokio::time::interval(interval);
                ticker.tick().await;
                // Dirty-flag: only rewrite the policy when learning actually
                // advanced since the last checkpoint. A long-running idle server
                // otherwise rewrites an identical multi-KB file every tick.
                let mut last_saved_updates: Option<u64> = None;

                loop {
                    ticker.tick().await;
                    if let Some(planner) = query::rl_planner::get_rl_planner() {
                        let updates = planner.total_updates().await;
                        if last_saved_updates == Some(updates) {
                            continue;
                        }
                        if let Err(e) = planner.save_policy(&checkpoint_path).await {
                            tracing::warn!("Failed to save RL policy checkpoint: {}", e);
                        } else {
                            last_saved_updates = Some(updates);
                        }
                    } else {
                        break;
                    }
                }
            });
            self.rl_checkpoint_handle = Some(checkpoint_handle);
        }

        Ok(())
    }

    /// Shutdown RL planner: stop checkpoint task and persist final policy
    async fn shutdown_rl_planner(&mut self) {
        if let Some(handle) = self.rl_checkpoint_handle.take() {
            handle.abort();
        }

        if let Some(ref policy_path) = self.rl_policy_path
            && let Some(planner) = query::rl_planner::get_rl_planner()
        {
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(5),
                planner.save_policy(policy_path),
            )
            .await
            {
                Ok(Ok(())) => tracing::info!("✅ RL policy persisted to {}", policy_path),
                Ok(Err(e)) => tracing::warn!("Failed to persist RL policy: {}", e),
                Err(_) => tracing::warn!("RL policy persist timeout"),
            }
        }
    }

    // =========================================================================
    // Graph Database API - High-level graph operations
    // =========================================================================

    /// Create a new graph collection
    pub async fn create_graph(
        &self,
        graph_id: &str,
        schema: Option<proto::proximadb_v1::GraphSchema>,
    ) -> anyhow::Result<()> {
        if let Some(ref multi_server) = self.multi_server {
            let request = proto::proximadb_v1::CreateGraphRequest {
                graph_id: graph_id.to_string(),
                name: Some(graph_id.to_string()),
                description: None,
                schema,
                storage_config: None,
                engine_config: None,
                access_control: None,
            };
            multi_server
                .shared_services
                .graph_service
                .create_graph_collection(request)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create graph collection: {}", e))?;
            Ok(())
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// List all graphs in the database
    pub async fn list_graphs(&self) -> anyhow::Result<Vec<String>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .list_graphs()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list graphs: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Create a node in a graph
    pub async fn create_node(
        &self,
        graph_id: &str,
        node: graph::Node,
    ) -> anyhow::Result<std::sync::Arc<graph::Node>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .create_node(graph_id, node)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create node: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Get a node by ID from a graph
    pub async fn get_node(
        &self,
        graph_id: &str,
        node_id: &str,
    ) -> anyhow::Result<Option<std::sync::Arc<graph::Node>>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .get_node(graph_id, &node_id.to_string())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get node: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Update a node in a graph
    pub async fn update_node(
        &self,
        graph_id: &str,
        node: graph::Node,
    ) -> anyhow::Result<std::sync::Arc<graph::Node>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .update_node(graph_id, node)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to update node: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Delete a node from a graph
    pub async fn delete_node(
        &self,
        graph_id: &str,
        node_id: &str,
    ) -> anyhow::Result<Option<std::sync::Arc<graph::Node>>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .delete_node(graph_id, &node_id.to_string())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to delete node: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Create an edge between two nodes in a graph
    pub async fn create_edge(
        &self,
        graph_id: &str,
        edge: graph::Edge,
    ) -> anyhow::Result<std::sync::Arc<graph::Edge>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .create_edge(graph_id, edge)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create edge: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Get an edge by ID from a graph
    pub async fn get_edge(
        &self,
        graph_id: &str,
        edge_id: &str,
    ) -> anyhow::Result<Option<std::sync::Arc<graph::Edge>>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .get_edge(graph_id, &edge_id.to_string())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get edge: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Delete an edge from a graph
    pub async fn delete_edge(
        &self,
        graph_id: &str,
        edge_id: &str,
    ) -> anyhow::Result<Option<std::sync::Arc<graph::Edge>>> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .delete_edge(graph_id, &edge_id.to_string())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to delete edge: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Get graph statistics
    pub async fn get_graph_stats(
        &self,
        graph_id: &str,
    ) -> anyhow::Result<crate::graph::GraphStats> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .get_stats(graph_id)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get graph stats: {}", e))
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Force-flush a collection's WAL/memtable to persisted engine storage
    /// (test/ops helper). Resolves the user-facing name to the canonical id
    /// first, then drains the WAL so subsequent reads must come from storage.
    pub async fn force_flush_collection(&self, collection: &str) -> anyhow::Result<()> {
        if let Some(ref multi_server) = self.multi_server {
            let vops = &multi_server.shared_services.vector_operations_service;
            let object_id = vops.resolve_collection_object_id(collection).await?;
            vops.force_flush_collection(&object_id.to_string())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to force-flush collection: {}", e))?;
            Ok(())
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }

    /// Flush the WAL for a specific graph to ensure durability
    pub async fn flush_graph_wal(&self, graph_id: &str) -> anyhow::Result<()> {
        if let Some(ref multi_server) = self.multi_server {
            multi_server
                .shared_services
                .graph_service
                .flush_wal(graph_id)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to flush graph WAL: {}", e))?;
            Ok(())
        } else {
            anyhow::bail!("Multi-server not initialized")
        }
    }
}

/// Initialize the async-ingest queue + drainer subsystem when the
/// `PROXIMADB_QUEUE_ROOT` env var is set. Returns `(None, None)` when
/// the var is absent — production deployments that haven't migrated
/// to async ingest just see no behavior change.
///
/// Config knobs (all env-driven for now):
///
/// - `PROXIMADB_QUEUE_ROOT` (required to enable): filesystem root URL,
///   typically `file:///var/lib/proximadb/queue` (PVC-backed in k8s)
///   or `adls://...` once the proximadb-filesystem extraction lands.
/// - `PROXIMADB_QUEUE_OBJECT_ARCHIVE` (optional): archive URL for
///   sealed segments — proves out the ECS-node-failure recovery story.
/// - `PROXIMADB_EMBED_DRAINER_PARTITIONS` (optional, default "0..16"):
///   partition range this replica drains. Single-replica deployments
///   leave it default; multi-replica needs disjoint per-pod ranges.
///
/// Open the queue client from an already-resolved config. The
/// `resolve()` step (see `core::config::QueueRuntimeConfig::resolve`)
/// has already folded env > TOML > defaults into a `ResolvedQueueConfig`,
/// so this function only does the I/O.
async fn open_queue_client_from_resolved(
    rq: &core::config::ResolvedQueueConfig,
) -> anyhow::Result<Arc<proximadb_queue::QueueClient>> {
    tracing::info!(
        queue_root = %rq.root,
        object_archive = ?rq.object_archive,
        sync_mode = %rq.sync_mode,
        "Opening queue subsystem for async ingest"
    );

    let mut topics = std::collections::HashMap::new();
    topics.insert(
        crate::services::EMBED_INGEST_TOPIC.to_string(),
        proximadb_queue::TopicConfig::default(),
    );
    let sync_mode = if rq.sync_mode == "lazy" {
        proximadb_queue::SyncMode::Lazy
    } else {
        proximadb_queue::SyncMode::Strict
    };
    let queue_cfg = proximadb_queue::QueueConfig {
        root: rq.root.clone(),
        object_archive: rq.object_archive.clone(),
        default_sync_mode: sync_mode,
        topics,
    };

    // Build a FilesystemFactory-backed QueueFs adapter so the queue's
    // `root` URL can be `file://`, `adls://`, `s3://`, etc. (any
    // scheme the factory knows). For pure-local file:// roots the
    // adapter behaves identically to the queue's built-in LocalFs;
    // for cloud schemes it routes through the factory's per-scheme
    // backend. This unblocks the "object_archive on adls:// / s3:// /
    // gcs://" use case for operators deploying with cross-scheme storage.
    let fs_config = crate::storage::persistence::filesystem::FilesystemConfig::default();
    let factory = std::sync::Arc::new(
        crate::storage::persistence::filesystem::FilesystemFactory::create(fs_config)
            .await
            .map_err(|e| anyhow::anyhow!("FilesystemFactory init: {}", e))?,
    );
    let fs_adapter = crate::services::queue_fs_adapter::FactoryQueueFs::new(factory, &rq.root)
        .map_err(|e| anyhow::anyhow!("queue filesystem capability check: {}", e))?;

    let queue = proximadb_queue::QueueClient::open_with_fs(queue_cfg, Some(fs_adapter))
        .await
        .map_err(|e| anyhow::anyhow!("queue open: {}", e))?;
    tracing::info!("✅ Queue subsystem opened (via FilesystemFactory adapter)");
    Ok(queue)
}

/// Spawn the embedding drainer using partitions from the resolved
/// queue config. The `drainer_partitions` field is the string form
/// (e.g. `"0..16"` or `"0,3..6,9"`); `parse_partition_list` turns it
/// into a `Vec<u32>`.
async fn spawn_embedding_drainer_from_resolved(
    queue: Arc<proximadb_queue::QueueClient>,
    record_ops: Arc<dyn proximadb_runtime::RecordOpsPort>,
    vector_operations_service: Arc<crate::services::VectorOperationsService>,
    catalog_manager: Arc<crate::catalog::CatalogManager>,
    rq: &core::config::ResolvedQueueConfig,
    default_storage_root: String,
) -> anyhow::Result<(
    tokio::task::JoinHandle<()>,
    tokio::sync::oneshot::Sender<()>,
)> {
    let bulk_loader = Arc::new(crate::services::bulk_load::BulkLoader::new(
        record_ops,
        vector_operations_service,
        default_storage_root,
    ));
    let sink: Arc<dyn crate::services::DrainerInsertSink> = Arc::new(
        crate::services::bulk_load::BulkLoadDrainerSink::new(bulk_loader),
    );
    let embed_service = proximadb_embedding::EmbeddingService::global();
    let drainer_cfg = crate::services::EmbeddingDrainerConfig::default();

    // Build the canonical-precision resolver from the same catalog +
    // cache the rest of the server uses. When no default catalog is
    // registered (degraded boot), the drainer falls back to None and
    // ships every record at fp32 — same behavior as before.
    let mut drainer =
        crate::services::EmbeddingDrainer::new(queue, embed_service, sink, drainer_cfg);
    match catalog_manager.default_catalog().await {
        Ok(catalog) => {
            // Separate CatalogCache instance for the resolver — same type as
            // the runtime catalog's cache, but its own Arc so TTL/capacity
            // and hit-rate signal stay independent. A
            // canonical_embedding_precision change can lag by up to the TTL
            // (60s) before the drainer picks it up; acceptable because
            // precision changes are rare and the mismatched-batch path
            // errors safely.
            let resolver_cache = Arc::new(proximadb_catalog::cache::CatalogCache::new(10_000, 60));
            let resolver = Arc::new(
                proximadb_catalog::canonical_precision::CanonicalPrecisionResolver::new(
                    catalog,
                    resolver_cache,
                ),
            );
            drainer = drainer.with_precision_resolver(resolver);
            tracing::info!(
                "✅ Embedding drainer wired with CanonicalPrecisionResolver — \
                 fp16 collections will produce fp16 records at ingest"
            );
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "drainer bootstrap: no default catalog; precision lookup disabled, \
                 records will ship as fp32 regardless of collection canonical precision"
            );
        }
    }

    let partitions: Vec<u32> = parse_partition_list(&rq.drainer_partitions).unwrap_or_else(|| {
        tracing::warn!(
            range = %rq.drainer_partitions,
            "failed to parse drainer partitions; falling back to 0..16"
        );
        (0..16u32).collect()
    });
    let pair = drainer.start(partitions);
    tracing::info!("✅ Embedding drainer started");
    Ok(pair)
}

/// Open the queue client when `PROXIMADB_QUEUE_ROOT` is set. Returns
/// `None` otherwise — deployments not opted into async ingest see no
/// behavior change.
///
/// **DEPRECATED**: prefer `QueueRuntimeConfig::resolve` +
/// `open_queue_client_from_resolved` for proper precedence handling.
/// This helper is retained until callers migrate.
#[allow(dead_code)]
async fn open_queue_client_if_configured()
-> anyhow::Result<Option<Arc<proximadb_queue::QueueClient>>> {
    let Some(root) = std::env::var("PROXIMADB_QUEUE_ROOT").ok() else {
        tracing::info!("PROXIMADB_QUEUE_ROOT not set; async ingest will degrade to inline embed");
        return Ok(None);
    };
    tracing::info!(queue_root = %root, "Opening queue subsystem for async ingest");

    let mut topics = std::collections::HashMap::new();
    topics.insert(
        crate::services::EMBED_INGEST_TOPIC.to_string(),
        proximadb_queue::TopicConfig::default(),
    );
    let mut queue_cfg = proximadb_queue::QueueConfig::from_env();
    queue_cfg.root = root;
    queue_cfg.topics = topics;

    let queue = proximadb_queue::QueueClient::open(queue_cfg)
        .await
        .map_err(|e| anyhow::anyhow!("queue open: {}", e))?;
    tracing::info!("✅ Queue subsystem opened");
    Ok(Some(queue))
}

/// Spawn the embedding drainer with the production
/// `BulkLoadDrainerSink`. The drainer subscribes to
/// `0..partition_count` by default. For multi-replica deployments,
/// `PROXIMADB_EMBED_DRAINER_PARTITIONS` carves out a per-pod range.
///
/// **DEPRECATED**: prefer `spawn_embedding_drainer_from_resolved`
/// which reads `default_storage_root` from the TOML config rather
/// than hard-defaulting.
#[allow(dead_code)]
async fn spawn_embedding_drainer(
    queue: Arc<proximadb_queue::QueueClient>,
    record_ops: Arc<dyn proximadb_runtime::RecordOpsPort>,
    vector_operations_service: Arc<crate::services::VectorOperationsService>,
) -> anyhow::Result<(
    tokio::task::JoinHandle<()>,
    tokio::sync::oneshot::Sender<()>,
)> {
    // Legacy fallback path; the resolved-from-config helper supplies
    // the canonical storage_locations URL instead of this hard default.
    let default_storage_root = "file:///tmp/proximadb".to_string();
    let bulk_loader = Arc::new(crate::services::bulk_load::BulkLoader::new(
        record_ops,
        vector_operations_service,
        default_storage_root,
    ));
    let sink: Arc<dyn crate::services::DrainerInsertSink> = Arc::new(
        crate::services::bulk_load::BulkLoadDrainerSink::new(bulk_loader),
    );
    let embed_service = proximadb_embedding::EmbeddingService::global();
    let drainer_cfg = crate::services::EmbeddingDrainerConfig::default();
    let drainer = crate::services::EmbeddingDrainer::new(queue, embed_service, sink, drainer_cfg);

    let partitions: Vec<u32> = std::env::var("PROXIMADB_EMBED_DRAINER_PARTITIONS")
        .ok()
        .and_then(|s| parse_partition_list(&s))
        .unwrap_or_else(|| (0..16u32).collect());
    let pair = drainer.start(partitions);
    tracing::info!("✅ Embedding drainer started");
    Ok(pair)
}

/// Parse "0,1,2" or "0..4" or "0,3..6,9" into a Vec<u32>.
fn parse_partition_list(s: &str) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    for chunk in s.split(',') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = chunk.split_once("..") {
            let lo: u32 = lo.parse().ok()?;
            let hi: u32 = hi.parse().ok()?;
            for p in lo..hi {
                out.push(p);
            }
        } else {
            out.push(chunk.parse().ok()?);
        }
    }
    Some(out)
}

#[cfg(test)]
mod async_ingest_wiring_tests {
    use super::parse_partition_list;

    #[test]
    fn parse_partition_list_handles_commas_and_ranges() {
        assert_eq!(parse_partition_list("0,1,2"), Some(vec![0, 1, 2]));
        assert_eq!(parse_partition_list("0..4"), Some(vec![0, 1, 2, 3]));
        assert_eq!(parse_partition_list("0,3..6,9"), Some(vec![0, 3, 4, 5, 9]));
        assert_eq!(parse_partition_list(""), Some(vec![]));
    }

    #[test]
    fn parse_partition_list_rejects_garbage() {
        assert_eq!(parse_partition_list("abc"), None);
        assert_eq!(parse_partition_list("0..abc"), None);
    }
}

#[cfg(test)]
mod security_initialization_tests {
    use super::initialize_configured_security;
    use crate::security::auth_service::{JwtConfig, SSOConfig};
    use crate::security::security_coordinator::{ComplianceConfig, TenantTrustConfig, TlsConfig};
    use crate::security::{
        AuthenticationConfig, AuthenticationMethod, EncryptionConfig, KeyStoreConfig, MtlsConfig,
        RBACConfig, SecurityConfig, SecurityMode,
    };
    use proximadb_security::AuditConfig;
    use std::collections::HashMap;

    fn security_config(enabled: bool) -> SecurityConfig {
        SecurityConfig {
            enabled,
            mode: SecurityMode::Production,
            authentication: AuthenticationConfig {
                enabled: true,
                methods: vec![AuthenticationMethod::ClientCertificate],
                require_authentication: true,
                default_session_timeout_minutes: 60,
                api_keys: HashMap::new(),
                jwt: JwtConfig {
                    enabled: false,
                    secret: String::new(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: String::new(),
                    audience: String::new(),
                    algorithm: "HS256".to_string(),
                },
                sso: SSOConfig {
                    enabled: false,
                    providers: Vec::new(),
                    token_cache_ttl_minutes: 5,
                },
                mtls: MtlsConfig {
                    enabled: true,
                    ca_cert_path: None,
                    require_client_cert: true,
                    cn_role_mapping: HashMap::new(),
                },
                audit_fail_closed: false,
                oidc: None,
            },
            rbac: RBACConfig::default(),
            audit: AuditConfig::default(),
            tls: TlsConfig {
                enabled: false,
                require_client_certificates: false,
                cert_file: None,
                key_file: None,
                ca_file: None,
            },
            compliance: ComplianceConfig {
                frameworks: Vec::new(),
                data_residency: None,
                encryption_at_rest: false,
                encryption_in_transit: false,
            },
            encryption: EncryptionConfig::default(),
            key_store: KeyStoreConfig::default(),
            tenant: TenantTrustConfig::default(),
        }
    }

    #[tokio::test]
    async fn enabled_security_initialization_failure_is_fatal() {
        let error = match initialize_configured_security(Some(security_config(true)), None).await {
            Ok(_) => panic!("enabled invalid security must fail startup"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("mTLS"));
    }

    #[tokio::test]
    async fn disabled_security_remains_an_explicit_oss_posture() {
        let security = initialize_configured_security(Some(security_config(false)), None)
            .await
            .expect("disabled security must not initialize its invalid providers");

        assert!(security.is_none());
    }
}
