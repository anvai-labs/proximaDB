// PostgreSQL Wire Protocol Implementation
//
// Provides PostgreSQL-compatible server for:
// - pgvector migration path
// - Standard PostgreSQL clients (psql, pgAdmin, etc.)
// - Existing application compatibility
//
// Protocol: PostgreSQL Protocol v3.0

/// Pure pgvector helpers: WHERE metadata-filter extraction (TD-100) and
/// extended-protocol parameter inference + result-column description (TD-102).
pub mod pgvector_params;
/// PostgreSQL Protocol v3.0 message parsing and encoding
pub mod protocol;
/// Bridge to the new relational pipeline (algebra → planner →
/// executor → engine). Opt-in via PROXIMADB_PGWIRE_RELATIONAL_PIPELINE.
pub mod relational_pipeline;
/// Server-side SCRAM-SHA-256 (RFC 7677) — pure SASL state machine backing the
/// pgwire `SCRAM-SHA-256` authentication mode (TD-PGWIRE-AUTH-1).
pub mod scram;
/// Session management for PostgreSQL client connections
pub mod session;
/// SQL-to-ProximaDB query translator (pgvector compatibility)
pub use proximadb_pgwire_translator::translator;
/// PostgreSQL type system mapping and conversions
pub use proximadb_pgwire_translator::types;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, error, info, warn};

use self::protocol::PostgresProtocol;
use self::session::SessionManager;
use crate::catalog::CatalogManager;
use crate::graph::GraphService;
use crate::observability::ObservabilityService;
use crate::services::VectorOperationsService;
use crate::storage::document::DocumentService;

/// Optional canonical write dependencies for PostgreSQL wire DML.
///
/// When present, catalog-routed relational tables can write through
/// `DmlService::with_direct_record_storage`; legacy vector-specialized tables
/// continue to use the compatibility route selected by xCatalog.
#[derive(Clone)]
pub struct DirectPgwireWriteServices {
    /// The single shared canonical record store (per-(tenant, collection)
    /// partitioned), built + WAL-recovered once at boot and shared across all
    /// pgwire connections so their in-memory partitions hold one authoritative
    /// state.
    canonical_store: Arc<crate::services::record_store::DirectWalTableRecordStore>,

    /// F5 / TD-OLTP-WIRING-1: the SAME process-shared fenced `ConditionalKeyStore`
    /// held by `SharedServices` (set via [`Self::with_conditional_key_store`]).
    /// Threaded into the pgwire `DmlService` so PK/FK uniqueness is fenced
    /// identically across pgwire and gRPC/REST. `None` ⇒ unfenced legacy probe.
    conditional_key_store: Option<Arc<dyn proximadb_storage_ports::ConditionalKeyStore>>,
}

impl DirectPgwireWriteServices {
    /// Build direct pgwire write dependencies from the shared canonical store.
    pub fn new(
        canonical_store: Arc<crate::services::record_store::DirectWalTableRecordStore>,
    ) -> Self {
        Self {
            canonical_store,
            conditional_key_store: None,
        }
    }

    /// Attach the process-shared fenced `ConditionalKeyStore` (the ONE instance
    /// from `SharedServices`), so pgwire writes fence PK/FK uniqueness on the same
    /// store as gRPC/REST.
    pub fn with_conditional_key_store(
        mut self,
        cks: Option<Arc<dyn proximadb_storage_ports::ConditionalKeyStore>>,
    ) -> Self {
        self.conditional_key_store = cks;
        self
    }
}

/// PostgreSQL-compatible server
pub struct PostgresServer {
    /// Listen address
    bind_address: SocketAddr,
    /// Session manager
    session_manager: Arc<SessionManager>,
    /// Collection port reference (Phase 9 / Task #76)
    collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
    /// Vector operations service for search
    vector_ops: Arc<VectorOperationsService>,
    /// Shared xCatalog manager for SQL DDL/DML and metadata introspection
    catalog_manager: Arc<CatalogManager>,
    /// Document service for JSON document collections
    document_service: Option<Arc<DocumentService>>,
    /// Graph service for graph collections
    graph_service: Option<Arc<GraphService>>,
    /// Observability service for logs/metrics/traces
    observability_service: Option<Arc<ObservabilityService>>,
    /// Optional canonical record/WAL writer for relational pgwire DML.
    direct_write_services: Option<DirectPgwireWriteServices>,
    /// Optional rank-pipeline singleton + durable catalog (R-7c.3
    /// production wiring). When `Some`, every per-connection DDL service
    /// is built with `with_rank_profile_store` + `with_rank_services` so
    /// SQL `CREATE RANK PROFILE` / `DROP RANK PROFILE` reach the same
    /// `RankServices` instance the REST / gRPC / Arrow Flight paths share.
    rank_pipeline: Option<PgwireRankPipeline>,
    /// Slice 6.3: when both are set, every per-connection
    /// `PostgresProtocol` is built with `with_primary_pod_gate` so
    /// pgwire INSERT/UPDATE/DELETE consults the same registry the REST
    /// and gRPC v2 paths consult. Held as a pair so `handle_connection`
    /// can decide once whether to wire the gate.
    primary_pod_registry: Option<Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>>,
    self_pod_id: Option<String>,
    partition_lease_manager: Option<Arc<crate::cluster::partition_lease::PartitionLeaseManager>>,
    /// Object-store root URL the warehouse materializer publishes Parquet snapshots
    /// under (the same URL the OLAP reader reopens the store from). When `Some`,
    /// every per-connection `DdlService` is wired with a `DmlTableMaterializer` so
    /// `ALTER TABLE … MATERIALIZE` works; when `None` the trigger returns a clean
    /// "requires a configured warehouse object store" error.
    warehouse_root_url: Option<String>,
    /// E0: shared per-IP rate limiter applied to the pgwire query path,
    /// consistent with REST. `None` (default) = no pgwire rate-limiting.
    rate_limiter: Option<Arc<crate::network::middleware::rate_limit::RateLimitState>>,
    /// TD-TENANT-1: the deployment's bare tenant-assertion trust policy,
    /// threaded into every per-connection `PostgresProtocol`. Default `Open`.
    tenant_header_trust: proximadb_tenant::HeaderTrustPolicy,
    /// ADR-0053 W8: tier-claim trust policy for the `proximadb_tier` startup
    /// parameter (the pgwire sibling of REST's X-Tenant-Tier gate).
    tier_header_trust: proximadb_tenant::HeaderTrustPolicy,
    /// Whether startup may omit the tenant/catalog and use a default.
    tenant_deployment_mode: proximadb_tenant::TenantDeploymentMode,
    /// TD-PGWIRE-AUTH-1: security coordinator for the SCRAM-SHA-256 mode.
    security_coordinator: Option<Arc<crate::security::SecurityCoordinator>>,
    /// TD-PGWIRE-AUTH-1: the resolved pgwire authentication posture.
    pgwire_auth: protocol::PgwireAuthMode,
    /// The exact composition-root enforcer used by REST/gRPC and mutated by the
    /// live ABAC admin API. Cloned into each pgwire DML façade at accept time.
    #[cfg(feature = "abac-policy")]
    abac_enforcer: Option<Arc<crate::security::rls::AbacEnforcer>>,
    /// Whether the server is running
    running: Arc<std::sync::atomic::AtomicBool>,
}

/// Bundle of the rank-pipeline + function-catalog handles pgwire threads into
/// every `DdlService` it constructs. Cloning is cheap (three `Arc`s).
#[derive(Clone)]
pub struct PgwireRankPipeline {
    pub services: Arc<crate::network::rest::canonical::rank::RankServices>,
    pub store: Arc<dyn crate::services::RankProfileStore>,
    /// Durable SQL user-function catalog (UDF F5) so `CREATE FUNCTION` over
    /// pgwire persists into the same store boot recovery replays.
    pub function_store: Arc<dyn crate::services::FunctionStore>,
}

impl PostgresServer {
    /// Create a new PostgreSQL server
    pub fn new(
        bind_address: SocketAddr,
        collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
        vector_ops: Arc<VectorOperationsService>,
        catalog_manager: Arc<CatalogManager>,
        document_service: Option<Arc<DocumentService>>,
        graph_service: Option<Arc<GraphService>>,
        observability_service: Option<Arc<ObservabilityService>>,
    ) -> Self {
        Self {
            bind_address,
            session_manager: Arc::new(SessionManager::new()),
            collection_port,
            vector_ops,
            catalog_manager,
            document_service,
            graph_service,
            observability_service,
            direct_write_services: None,
            rank_pipeline: None,
            primary_pod_registry: None,
            self_pod_id: None,
            partition_lease_manager: None,
            warehouse_root_url: None,
            rate_limiter: None,
            tenant_header_trust: proximadb_tenant::HeaderTrustPolicy::default(),
            tier_header_trust: proximadb_tenant::HeaderTrustPolicy::default(),
            tenant_deployment_mode: proximadb_tenant::TenantDeploymentMode::single_tenant_default(),
            security_coordinator: None,
            pgwire_auth: protocol::PgwireAuthMode::Trust,
            #[cfg(feature = "abac-policy")]
            abac_enforcer: None,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// TD-TENANT-1: set the deployment's bare tenant-assertion trust policy —
    /// the same `HeaderTrustPolicy` the REST/gRPC/Arrow Flight surfaces hold.
    pub fn with_tenant_header_trust(mut self, policy: proximadb_tenant::HeaderTrustPolicy) -> Self {
        self.tenant_header_trust = policy;
        self
    }

    /// TD-PGWIRE-AUTH-1: attach the security coordinator (verifier lookups +
    /// post-exchange identity resolution for the SCRAM-SHA-256 mode) and the
    /// resolved authentication posture. pgwire is the last network surface the
    /// coordinator is threaded into.
    pub fn with_security_coordinator(
        mut self,
        coordinator: Arc<crate::security::SecurityCoordinator>,
        auth_mode: protocol::PgwireAuthMode,
    ) -> Self {
        self.security_coordinator = Some(coordinator);
        self.pgwire_auth = auth_mode;
        self
    }

    /// ADR-0053 W8: set the tier-claim trust policy.
    pub fn with_tier_header_trust(mut self, policy: proximadb_tenant::HeaderTrustPolicy) -> Self {
        self.tier_header_trust = policy;
        self
    }

    pub fn with_tenant_deployment_mode(
        mut self,
        mode: proximadb_tenant::TenantDeploymentMode,
    ) -> Self {
        self.tenant_deployment_mode = mode;
        self
    }

    /// E0: attach a shared per-IP rate limiter so each per-connection
    /// `PostgresProtocol` rate-limits its query path consistently with REST,
    /// using the same converged `RateLimitState`.
    pub fn with_rate_limiter(
        mut self,
        limiter: Arc<crate::network::middleware::rate_limit::RateLimitState>,
    ) -> Self {
        self.rate_limiter = Some(limiter);
        self
    }

    /// Slice 6.3: attach the primary-pod write router so each
    /// connection's `PostgresProtocol` consults the gate on DML.
    /// Passing both as a pair makes "partially wired" unrepresentable
    /// — `handle_connection` only applies the gate when BOTH are set.
    pub fn with_primary_pod_gate(
        mut self,
        registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
        self_pod_id: String,
    ) -> Self {
        self.primary_pod_registry = Some(registry);
        self.self_pod_id = Some(self_pod_id);
        self
    }

    /// Attach the same durable lease manager used by the REST/DML write gates
    /// so per-connection pgwire DDL does not degrade to registry-only checks.
    pub fn with_partition_lease_manager(
        mut self,
        manager: Option<Arc<crate::cluster::partition_lease::PartitionLeaseManager>>,
    ) -> Self {
        self.partition_lease_manager = manager;
        self
    }

    /// Attach the process-wide rank-pipeline so each per-connection
    /// `DdlService` is built with the rank-profile catalog + live
    /// registry wired in. Production callers pass
    /// `SharedServices.rank_services` + `SharedServices.rank_profile_store`.
    pub fn with_rank_pipeline(
        mut self,
        services: Arc<crate::network::rest::canonical::rank::RankServices>,
        store: Arc<dyn crate::services::RankProfileStore>,
        function_store: Arc<dyn crate::services::FunctionStore>,
    ) -> Self {
        self.rank_pipeline = Some(PgwireRankPipeline {
            services,
            store,
            function_store,
        });
        self
    }

    /// Attach the warehouse object-store root URL so each per-connection
    /// `DdlService` is built with a `DmlTableMaterializer`, enabling
    /// `ALTER TABLE … MATERIALIZE` to publish Parquet snapshots there. Production
    /// callers pass the configured object-store/storage root; without it the
    /// trigger returns a clean "requires a configured warehouse object store" error.
    pub fn with_warehouse_materialization(mut self, warehouse_root_url: String) -> Self {
        self.warehouse_root_url = Some(warehouse_root_url);
        self
    }

    /// Enable direct canonical record/WAL writes for catalog-routed relational
    /// pgwire DML while preserving legacy vector compatibility routing.
    pub fn with_direct_record_writes(
        mut self,
        canonical_store: Arc<crate::services::record_store::DirectWalTableRecordStore>,
    ) -> Self {
        self.direct_write_services = Some(DirectPgwireWriteServices::new(canonical_store));
        self
    }

    /// Enable direct canonical record/WAL writes with prebuilt dependencies.
    pub fn with_direct_write_services(mut self, services: DirectPgwireWriteServices) -> Self {
        self.direct_write_services = Some(services);
        self
    }

    /// Wire the process-shared live ABAC enforcer into every accepted pgwire
    /// connection. This is independent of the direct-record-write option: reads
    /// must remain governed under either storage configuration.
    #[cfg(feature = "abac-policy")]
    pub fn with_abac_enforcer(
        mut self,
        enforcer: Option<Arc<crate::security::rls::AbacEnforcer>>,
    ) -> Self {
        self.abac_enforcer = enforcer;
        self
    }

    /// Start the PostgreSQL server
    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(self.bind_address).await?;
        info!("PostgreSQL server listening on {}", self.bind_address);

        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);

        while self.running.load(std::sync::atomic::Ordering::Relaxed) {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    info!("New PostgreSQL connection from {}", addr);
                    // Disable Nagle's algorithm on the pgwire socket (as libpq's
                    // server does): a SELECT response is written as several small
                    // segments (RowDescription, DataRow(s), CommandComplete), and
                    // with Nagle on, the second segment waits for the first to be
                    // ACKed while the client delays its ACK — a ~40 ms delayed-ACK
                    // stall PER QUERY, even on loopback. Measured as the dominant
                    // per-query wall floor (TD-OLAP-4 floor decomposition: setup/
                    // open/plan/compute/emit all ~0, ~46 ms unattributed). Best-
                    // effort: a failure here only forfeits the latency win.
                    if let Err(e) = stream.set_nodelay(true) {
                        warn!("pgwire: failed to set TCP_NODELAY on {}: {}", addr, e);
                    }
                    let session_manager = self.session_manager.clone();
                    let collection_port = self.collection_port.clone();
                    let vector_ops = self.vector_ops.clone();
                    let catalog_manager = self.catalog_manager.clone();
                    let document_service = self.document_service.clone();
                    let graph_service = self.graph_service.clone();
                    let observability_service = self.observability_service.clone();
                    let direct_write_services = self.direct_write_services.clone();
                    let rank_pipeline = self.rank_pipeline.clone();
                    // Slice 6.3 capture: same `Arc<PrimaryPodRegistry>`
                    // + pod-id pair the REST / gRPC v2 / Arrow Flight
                    // surfaces hold, so pgwire DML sees the identical
                    // routing decisions.
                    let primary_pod_registry = self.primary_pod_registry.clone();
                    let self_pod_id = self.self_pod_id.clone();
                    let partition_lease_manager = self.partition_lease_manager.clone();
                    let warehouse_root_url = self.warehouse_root_url.clone();
                    let rate_limiter = self.rate_limiter.clone();
                    let tenant_header_trust = self.tenant_header_trust;
                    let tier_header_trust = self.tier_header_trust;
                    let tenant_deployment_mode = self.tenant_deployment_mode.clone();
                    let security_coordinator = self.security_coordinator.clone();
                    let pgwire_auth = self.pgwire_auth;
                    #[cfg(feature = "abac-policy")]
                    let abac_enforcer = self.abac_enforcer.clone();

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(
                            stream,
                            addr,
                            session_manager,
                            collection_port,
                            vector_ops,
                            catalog_manager,
                            document_service,
                            graph_service,
                            observability_service,
                            direct_write_services,
                            rank_pipeline,
                            primary_pod_registry,
                            self_pod_id,
                            partition_lease_manager,
                            warehouse_root_url,
                            rate_limiter,
                            tenant_header_trust,
                            tier_header_trust,
                            tenant_deployment_mode,
                            security_coordinator,
                            pgwire_auth,
                            #[cfg(feature = "abac-policy")]
                            abac_enforcer,
                        )
                        .await
                        {
                            error!("Connection error from {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Stop the server
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
        info!("PostgreSQL server stopped");
    }

    /// Handle a client connection
    async fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        session_manager: Arc<SessionManager>,
        collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
        vector_ops: Arc<VectorOperationsService>,
        catalog_manager: Arc<CatalogManager>,
        document_service: Option<Arc<DocumentService>>,
        graph_service: Option<Arc<GraphService>>,
        observability_service: Option<Arc<ObservabilityService>>,
        direct_write_services: Option<DirectPgwireWriteServices>,
        rank_pipeline: Option<PgwireRankPipeline>,
        primary_pod_registry: Option<Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>>,
        self_pod_id: Option<String>,
        partition_lease_manager: Option<
            Arc<crate::cluster::partition_lease::PartitionLeaseManager>,
        >,
        warehouse_root_url: Option<String>,
        rate_limiter: Option<Arc<crate::network::middleware::rate_limit::RateLimitState>>,
        tenant_header_trust: proximadb_tenant::HeaderTrustPolicy,
        tier_header_trust: proximadb_tenant::HeaderTrustPolicy,
        tenant_deployment_mode: proximadb_tenant::TenantDeploymentMode,
        security_coordinator: Option<Arc<crate::security::SecurityCoordinator>>,
        pgwire_auth: protocol::PgwireAuthMode,
        #[cfg(feature = "abac-policy")] abac_enforcer: Option<
            Arc<crate::security::rls::AbacEnforcer>,
        >,
    ) -> Result<()> {
        // Create session
        let session = session_manager.create_session(addr).await?;
        let session_id = session.id.clone();

        // Create protocol handler
        let protocol = PostgresProtocol::new(
            stream,
            session,
            collection_port,
            vector_ops,
            document_service,
            graph_service,
            observability_service,
        )
        .with_session_manager(session_manager.clone());
        // TD-ABAC-10 (ADR-087): the same catalog-backed stable-id resolver
        // REST/gRPC/Arrow wire — stamps the session identity's ABAC policy key
        // ONCE at the startup handshake. catalog_manager already reaches
        // pgwire, so this is zero new plumbing.
        let protocol = protocol.with_stable_id_resolver(Some(Arc::new(
            crate::security::CatalogTenantStableIdResolver::new(catalog_manager.clone()),
        )));
        #[allow(unused_mut)] // feature build rebinds after attaching ABAC below
        let mut protocol = if let Some(direct_write_services) = direct_write_services {
            protocol.with_direct_catalog_manager(
                catalog_manager,
                direct_write_services.canonical_store,
                direct_write_services.conditional_key_store,
            )
        } else {
            protocol.with_catalog_manager(catalog_manager)
        };
        #[cfg(feature = "abac-policy")]
        let mut protocol = protocol.with_abac_enforcer(abac_enforcer);
        if let Some(pipeline) = rank_pipeline {
            protocol = protocol.with_rank_pipeline(
                pipeline.services,
                pipeline.store,
                pipeline.function_store,
            );
        }
        // Wire the warehouse materializer LAST so it augments the fully-assembled
        // (catalog + rank) DdlService rather than being clobbered by a later rebuild.
        if let Some(warehouse_root_url) = warehouse_root_url {
            protocol = protocol.with_materializer(warehouse_root_url);
        }
        // Slice 6.3: pair-wise wiring — only apply the gate when both
        // sides are present so a partial wiring fails closed (no
        // gate, legacy behavior) rather than silently misconfigured.
        if let (Some(registry), Some(pod_id)) = (primary_pod_registry, self_pod_id) {
            protocol = protocol.with_primary_pod_gate(registry.clone(), pod_id.clone());
            if let Some(manager) = partition_lease_manager {
                protocol = protocol.with_ddl_lease_manager(manager, registry, pod_id);
            }
        }
        // E0: apply the shared per-IP rate limiter (subject = this peer's IP) so
        // the pgwire query path is rate-limited consistently with REST.
        if let Some(limiter) = rate_limiter {
            protocol = protocol.with_rate_limiter(limiter, addr.ip());
        }
        // TD-TENANT-1: same bare-assertion trust policy as REST/gRPC/Flight,
        // enforced at the startup handshake and the tenant session vars.
        protocol = protocol.with_tenant_header_trust(tenant_header_trust);
        // ADR-0053 W8: tier-claim gate, same shape as the tenant gate above.
        protocol = protocol.with_tier_header_trust(tier_header_trust);
        protocol = protocol.with_tenant_deployment_mode(tenant_deployment_mode);
        // TD-PGWIRE-AUTH-1: coordinator + resolved posture for the SCRAM mode.
        if let Some(coordinator) = security_coordinator {
            protocol = protocol.with_security_coordinator(coordinator);
        }
        protocol = protocol.with_pgwire_auth_mode(pgwire_auth);

        // Run protocol loop
        match protocol.run().await {
            Ok(_) => {
                debug!("Session {} completed normally", session_id);
            }
            Err(e) => {
                warn!("Session {} error: {}", session_id, e);
            }
        }

        // Cleanup session
        session_manager.remove_session(&session_id).await;

        Ok(())
    }
}

/// PostgreSQL server configuration
#[derive(Debug, Clone)]
pub struct PostgresConfig {
    /// Bind address
    pub bind_address: SocketAddr,
    /// Maximum connections
    pub max_connections: usize,
    /// Idle timeout (seconds)
    pub idle_timeout_secs: u64,
    /// Statement cache size
    pub statement_cache_size: usize,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:5432".parse().unwrap_or_else(|_| {
                "127.0.0.1:5433"
                    .parse()
                    .unwrap_or_else(|_| std::net::SocketAddr::from(([127, 0, 0, 1], 5433)))
            }),
            max_connections: 100,
            idle_timeout_secs: 3600,
            statement_cache_size: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PostgresConfig::default();
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.idle_timeout_secs, 3600);
    }
}
