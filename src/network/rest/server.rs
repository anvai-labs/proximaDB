/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! REST server implementation using axum

use axum::{Router, extract::DefaultBodyLimit, middleware};
use proximadb_graph_query::service::GraphExecutionService;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::compression::CompressionLayer;
use tower_http::decompression::DecompressionLayer;
use tower_http::trace::TraceLayer;

use super::canonical::handlers::{AppState, RestCoreServices, create_router};
use crate::monitoring::MetricsCollector;
use crate::network::middleware::backpressure::{
    BackpressureConfig, create_concurrency_limit_layer,
};
use crate::network::middleware::cors::{CorsConfig, create_cors_layer};
use crate::network::middleware::request_id::request_id_middleware;
use crate::network::middleware::tenant::{
    TenantExtractor, TenantExtractorConfig, tenant_middleware,
};
use crate::network::middleware::timeout::{TimeoutConfig, create_timeout_layer};
use crate::network::tls::TlsConfig as NetworkTlsConfig;
use crate::security::SecurityCoordinator;

/// REST server for ProximaDB
pub struct RestServer {
    router: Router,
    bind_addr: SocketAddr,
    tls_config: Option<NetworkTlsConfig>,
    /// Portless ("embedded") mode: when set, the plaintext server binds this
    /// Unix-domain socket path instead of `bind_addr`'s TCP port. `bind_addr`
    /// is still carried for logging. TLS + UDS is not supported (embedded is
    /// always plaintext over the local socket).
    uds_path: Option<PathBuf>,
}

/// Port-based service objects for wiring proximadb-api REST handlers.
///
/// When provided to `RestServer`, the document/graph/observability routes are
/// served by the port-backed handlers from `proximadb-api` instead of the
/// legacy root-crate handlers.
pub struct RestServerPorts {
    /// Document/graph/observability route ports. Optional: TD-104 S1 lets the
    /// multi-port REST server carry `api_handlers` even when these are not built
    /// (e.g. gRPC disabled), so collection/vector dispatch always reaches the
    /// runtime port-based handler instead of falling back to the root handler.
    pub doc_port: Option<std::sync::Arc<dyn proximadb_runtime::DocumentPort>>,
    pub graph_port: Option<std::sync::Arc<dyn proximadb_runtime::GraphPort>>,
    pub obs_port: Option<std::sync::Arc<dyn proximadb_runtime::ObservabilityPort>>,
    /// Port-backed handler for collection/vector routes (Phase 9.10). Always set.
    pub api_handlers: std::sync::Arc<dyn proximadb_runtime::ApiHandlersPort>,
}

/// Authentication configuration for the REST server
#[derive(Debug, Clone, Default)]
pub struct RestAuthConfig {
    /// Whether unified authentication is enabled
    pub enabled: bool,
}

/// Security configuration for the REST server.
///
/// This struct bundles all security-related settings for the REST API.
/// Defaults are configured for production security.
#[derive(Debug, Clone)]
pub struct RestServerSecurityConfig {
    /// CORS configuration
    pub cors: CorsConfig,
    /// Request timeout configuration
    pub timeout: TimeoutConfig,
    /// Backpressure configuration for load shedding
    pub backpressure: BackpressureConfig,
    /// Unified authentication configuration
    pub auth: RestAuthConfig,
    /// Multi-tenant configuration
    pub tenant: TenantExtractorConfig,
    /// Whether to run in development mode (relaxed security)
    pub development_mode: bool,
    /// TLS configuration (None = plaintext, Some = TLS enabled)
    pub tls: Option<RestTlsConfig>,
}

/// TLS configuration for REST server
#[derive(Debug, Clone)]
pub struct RestTlsConfig {
    /// Path to certificate file (PEM format)
    pub cert_file: PathBuf,
    /// Path to private key file (PEM format)
    pub key_file: PathBuf,
    /// Path to CA certificate file for mTLS (optional)
    pub ca_file: Option<PathBuf>,
    /// Require client certificates (mTLS)
    pub require_client_certs: bool,
    /// Allowed CN patterns for mTLS (empty = allow all)
    pub allowed_cn_patterns: Vec<String>,
    /// CN to user ID mappings
    pub cn_to_user_mapping: std::collections::HashMap<String, String>,
    /// Default roles for mTLS-authenticated users
    pub default_roles: Vec<String>,
}

impl Default for RestTlsConfig {
    fn default() -> Self {
        Self {
            cert_file: PathBuf::new(),
            key_file: PathBuf::new(),
            ca_file: None,
            require_client_certs: false,
            allowed_cn_patterns: vec![],
            cn_to_user_mapping: std::collections::HashMap::new(),
            default_roles: vec!["reader".to_string()],
        }
    }
}

impl RestTlsConfig {
    /// Create new TLS config with certificate paths
    pub fn new(cert_file: PathBuf, key_file: PathBuf) -> Self {
        Self {
            cert_file,
            key_file,
            ..Default::default()
        }
    }

    /// Enable mTLS with CA certificate
    pub fn with_mtls(mut self, ca_file: PathBuf) -> Self {
        self.ca_file = Some(ca_file);
        self.require_client_certs = true;
        self
    }

    /// Set allowed CN patterns
    pub fn with_allowed_cn_patterns(mut self, patterns: Vec<String>) -> Self {
        self.allowed_cn_patterns = patterns;
        self
    }

    /// Set CN to user mappings
    pub fn with_cn_mappings(mut self, mappings: std::collections::HashMap<String, String>) -> Self {
        self.cn_to_user_mapping = mappings;
        self
    }

    /// Set default roles for mTLS users
    pub fn with_default_roles(mut self, roles: Vec<String>) -> Self {
        self.default_roles = roles;
        self
    }
}

impl Default for RestServerSecurityConfig {
    #[inline]
    fn default() -> Self {
        Self {
            cors: CorsConfig::production(),
            timeout: TimeoutConfig::default(),
            backpressure: BackpressureConfig::default(),
            auth: RestAuthConfig::default(),
            tenant: TenantExtractorConfig::default(), // Single-tenant mode by default
            development_mode: false,
            tls: None, // Plaintext by default
        }
    }
}

impl RestServerSecurityConfig {
    /// Create a new security configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a development configuration with relaxed security.
    ///
    /// **WARNING**: Only use for local development. Never in production!
    pub fn development() -> Self {
        Self {
            cors: CorsConfig::development(),
            timeout: TimeoutConfig::default(),
            backpressure: BackpressureConfig::disabled(),
            auth: RestAuthConfig { enabled: false },
            tenant: TenantExtractorConfig::single_tenant("default"), // Single-tenant for dev
            development_mode: true,
            tls: None,
        }
    }

    /// Create a production configuration with specified allowed origins.
    pub fn production_with_origins(origins: Vec<String>) -> Self {
        let mut cors = CorsConfig::production();
        cors.allowed_origins = origins;
        Self {
            cors,
            timeout: TimeoutConfig::default(),
            backpressure: BackpressureConfig::default(),
            auth: RestAuthConfig::default(),
            tenant: TenantExtractorConfig::default(),
            development_mode: false,
            tls: None,
        }
    }

    /// Create a production configuration with TLS enabled.
    pub fn production_with_tls(tls_config: RestTlsConfig) -> Self {
        Self {
            cors: CorsConfig::production(),
            timeout: TimeoutConfig::default(),
            backpressure: BackpressureConfig::default(),
            auth: RestAuthConfig::default(),
            tenant: TenantExtractorConfig::default(),
            development_mode: false,
            tls: Some(tls_config),
        }
    }

    /// Enable TLS on this configuration.
    pub fn with_tls(mut self, tls_config: RestTlsConfig) -> Self {
        self.tls = Some(tls_config);
        self
    }

    /// Create a multi-tenant configuration.
    pub fn multi_tenant() -> Self {
        Self {
            cors: CorsConfig::production(),
            timeout: TimeoutConfig::default(),
            backpressure: BackpressureConfig::default(),
            auth: RestAuthConfig { enabled: true }, // Auth required for multi-tenant
            tenant: TenantExtractorConfig::multi_tenant(),
            development_mode: false,
            tls: None,
        }
    }
}

impl RestServer {
    /// Create new REST server with default security configuration.
    ///
    /// Uses production-safe defaults:
    /// - CORS: No cross-origin requests allowed
    /// - Timeout: 30 second request timeout
    /// - Compression: Optional based on parameter
    pub fn new(
        bind_addr: SocketAddr,
        core: RestCoreServices,
        max_request_size_mb: Option<u64>,
        compression: bool,
        metrics_collector: Option<Arc<MetricsCollector>>,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
    ) -> Self {
        Self::with_security(
            bind_addr,
            core,
            max_request_size_mb,
            compression,
            metrics_collector,
            security_coordinator,
            RestServerSecurityConfig::default(),
            None,
        )
    }

    /// Create new REST server with development mode (relaxed security).
    ///
    /// **WARNING**: Only use for local development and testing!
    pub fn new_development(
        bind_addr: SocketAddr,
        core: RestCoreServices,
        max_request_size_mb: Option<u64>,
        compression: bool,
        metrics_collector: Option<Arc<MetricsCollector>>,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
    ) -> Self {
        tracing::warn!("🚨 Starting REST server in DEVELOPMENT mode - security is relaxed!");
        Self::with_security(
            bind_addr,
            core,
            max_request_size_mb,
            compression,
            metrics_collector,
            security_coordinator,
            RestServerSecurityConfig::development(),
            None,
        )
    }

    /// Create new REST server with custom security configuration.
    pub fn with_security(
        bind_addr: SocketAddr,
        core: RestCoreServices,
        max_request_size_mb: Option<u64>,
        compression: bool,
        metrics_collector: Option<Arc<MetricsCollector>>,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
        security_config: RestServerSecurityConfig,
        llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
    ) -> Self {
        let graph_execution_service: Arc<dyn GraphExecutionService> =
            core.graph_operations_service.clone();
        Self::with_security_and_config(
            bind_addr,
            core,
            graph_execution_service,
            max_request_size_mb,
            compression,
            metrics_collector,
            security_coordinator,
            security_config,
            std::path::PathBuf::from("/tmp/proximadb/data"), // Default fallback
            None, // No query adapter for legacy constructor
            llm_engine,
        )
    }

    /// Create new REST server with custom security configuration and data directory from config.
    pub fn with_security_and_config(
        bind_addr: SocketAddr,
        core: RestCoreServices,
        graph_execution_service: Arc<dyn GraphExecutionService>,
        max_request_size_mb: Option<u64>,
        compression: bool,
        metrics_collector: Option<Arc<MetricsCollector>>,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
        security_config: RestServerSecurityConfig,
        data_dir: std::path::PathBuf,
        query_adapter: Option<Arc<crate::query::facade::QueryFacadeAdapter>>,
        llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
    ) -> Self {
        Self::with_security_and_config_and_ports(
            bind_addr,
            core,
            graph_execution_service,
            max_request_size_mb,
            compression,
            metrics_collector,
            security_coordinator,
            security_config,
            data_dir,
            query_adapter,
            llm_engine,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Create new REST server with port-backed service objects wired in.
    ///
    /// When `ports` is `Some`, document/graph/observability routes are served
    /// by handlers from `proximadb-api` that delegate to the port trait objects.
    /// When `queue_client` is `Some`, the v3 `/documents?mode=async` handler
    /// routes through `producer.send`; otherwise it falls back to inline embed.
    pub fn with_security_and_config_and_ports(
        bind_addr: SocketAddr,
        core: RestCoreServices,
        graph_execution_service: Arc<dyn GraphExecutionService>,
        max_request_size_mb: Option<u64>,
        compression: bool,
        metrics_collector: Option<Arc<MetricsCollector>>,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
        security_config: RestServerSecurityConfig,
        data_dir: std::path::PathBuf,
        query_adapter: Option<Arc<crate::query::facade::QueryFacadeAdapter>>,
        llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
        ports: Option<RestServerPorts>,
        catalog_manager: Option<Arc<crate::catalog::CatalogManager>>,
        queue_client: Option<Arc<proximadb_queue::QueueClient>>,
        fulltext_indexes: Option<crate::network::rest::canonical::handlers::FullTextIndexMap>,
        discovery_service: Option<Arc<crate::services::discovery::DiscoveryService>>,
        external_collection_service: Option<
            Arc<crate::services::external_collection::ExternalCollectionService>,
        >,
    ) -> Self {
        // Tier 1.2 (pre-release foundational plan 2026-05-26):
        // Warn loudly if a production-mode server is being constructed with
        // authentication disabled.  `RestServerSecurityConfig::default()` and
        // `production_with_origins` / `production_with_tls` builders all
        // delegate to `RestAuthConfig::default()` which has `enabled: false`
        // for backward compatibility — explicit `multi_tenant()` setups turn
        // auth on.  Single-tenant production deployments that forget to wire
        // auth would otherwise silently accept unauthenticated requests; the
        // warning makes the foot-gun visible at server start.
        if !security_config.development_mode && !security_config.auth.enabled {
            tracing::warn!(
                target: "proximadb::security",
                bind_addr = %bind_addr,
                "REST server starting in production mode with auth.enabled=false. \
                 All requests will be accepted without authentication. If this is \
                 intentional (single-tenant trusted-network deployment), suppress \
                 by switching to development_mode or by explicitly enabling auth via \
                 RestServerSecurityConfig::multi_tenant() / auth.enabled=true. \
                 See PRE_RELEASE_FOUNDATIONS_2026_05_26.adoc T1.2 for context."
            );
        }

        let mut base_state = AppState::new(
            core,
            graph_execution_service,
            security_coordinator.clone(),
            data_dir,
            query_adapter.clone(),
            llm_engine,
        );
        // TD-TENANT-1 item 3: build the tenant-stable-id resolver from the
        // catalog handle before it's moved into base_state below, so the tenant
        // middleware can stamp tenant_stable_id for ABAC enforcement.
        let tenant_stable_id_resolver = catalog_manager.as_ref().map(|cm| {
            Arc::new(crate::security::CatalogTenantStableIdResolver::new(
                cm.clone(),
            )) as Arc<dyn proximadb_tenant::TenantStableIdResolver>
        });
        if let Some(manager) = catalog_manager {
            base_state = base_state.with_catalog_manager(manager);
        }
        if let Some(qc) = queue_client {
            base_state = base_state.with_queue_client(qc);
        }
        // T3.2 Slice 1b: share `fulltext_indexes` with `SharedServices` so REST
        // hybrid search and gRPC hybrid search read+write the same in-process
        // map. Without this, REST has its own empty map and gRPC's BM25
        // component returns 0 results.
        if let Some(indexes) = fulltext_indexes.clone() {
            base_state = base_state.with_fulltext_indexes(indexes);
        }
        // Phase 8 (F1): share the Continuous Discovery service so the v2
        // `discovery-jobs` endpoints reach the same registry the background
        // executor consumes (multi-port REST path).
        if let Some(discovery) = discovery_service {
            base_state = base_state.with_discovery_service(discovery);
        }
        // Phase 8 (F5): share the External Collection service so the v2
        // `external-collections` endpoints reach the same registry.
        if let Some(external) = external_collection_service {
            base_state = base_state.with_external_collection_service(external);
        }
        let state = if let Some(p) = ports {
            // Always wire the runtime port-based api_handlers (TD-104 S1); the
            // document/graph/observability route ports are mounted only when present.
            let mut s = base_state.with_api_handlers(p.api_handlers);
            if let (Some(doc_port), Some(graph_port), Some(obs_port)) =
                (p.doc_port, p.graph_port, p.obs_port)
            {
                s = s.with_ports(doc_port, graph_port, obs_port);
            }
            s
        } else {
            base_state
        };
        let state = if let Some(ref adapter) = query_adapter {
            let port = Arc::new(
                crate::query::UnifiedQueryPortImpl::new(adapter.clone())
                    .with_catalog_manager(state.catalog_manager.clone()),
            ) as Arc<dyn proximadb_runtime::UnifiedQueryPort>;
            state.with_unified_query_port(port)
        } else {
            state
        };

        // Calculate max request size in bytes (default to 64MB if not specified)
        let max_size_bytes = max_request_size_mb.unwrap_or(64) * 1024 * 1024;

        // Create metrics router if metrics collector is available
        let metrics_router = if let Some(collector) = metrics_collector {
            use crate::network::metrics_service::{MetricsService, MetricsServiceConfig};
            let metrics_config = MetricsServiceConfig::default();
            let metrics_service = MetricsService::new(metrics_config, collector);
            Some(metrics_service.create_router())
        } else {
            None
        };

        // Build service layers conditionally to avoid type mismatch
        let security_coordinator = state.security_coordinator.clone();
        let state_for_v2 = state.clone();
        let mut base_router = create_router(state.clone());

        // TD-V1SUNSET-1 RESOLVED (2026-08-29): the v1_sunset middleware is
        // REMOVED. /api/v1/* now falls to `not_found_fallback`, which answers
        // 404 + the exact /api/v2 replacement endpoint + docs link via
        // `v1_replacement_for` — the migration signal survives; only the
        // machine-readable Deprecation/Sunset headers are gone. Closure
        // evidence: every known client accounted for (SDKs + AnvaiOps — no
        // ProximaDB v1 calls, no REST_V1_COMPAT opt-ins), and /api/v1 has
        // been 410-by-default since #814 (2026-07-09).

        // TD-RATE-1: mount the REST request limiter, default OFF — the gate
        // being unset adds NO layer, so the default deployment is unchanged.
        // Keyed on the TD-TENANT-4 observed client address; `/health*` is
        // exempt (the middleware checks `limit_health_endpoints`, false by
        // default). Positioned beside the v1-sunset layer, i.e. inside the
        // auth/tenant layers: floods are counted after authentication, which
        // keeps per-IP accounting aligned with the authenticated principal.
        if let Some(limiter) =
            crate::network::middleware::rate_limit::rest_rate_limit_state_from_env()
        {
            tracing::info!(
                rpm_env = crate::network::middleware::rate_limit::REST_RATE_LIMIT_RPM_ENV,
                "REST request rate limiting enabled (TD-RATE-1)"
            );
            base_router = base_router.layer(middleware::from_fn_with_state(
                limiter,
                crate::network::middleware::rate_limit::rate_limit_middleware,
            ));
        }

        // Nest metrics router if available
        if let Some(metrics) = metrics_router {
            base_router = base_router.nest("/metrics", metrics);
            tracing::info!("✅ Metrics endpoints enabled at /metrics");
        }

        // NOTE: the read-only admin dashboard (/admin) is mounted ONLY on the
        // unified standalone path (`build_router_for_unified`), gated by
        // `[server.admin_ui] enabled`. This legacy multi-port path never serves it.

        // Add V2 API router with ProximaRecord support
        let v2_router = super::v2::create_v2_router().with_state(state_for_v2.clone());
        base_router = base_router.nest("/api/v2", v2_router);
        tracing::info!("✅ V2 API enabled at /api/v2 (ProximaRecord, typed schema)");

        // TD-MLOPS-1 slice 2: MLflow-compatible tracking wire, default OFF.
        if super::mlflow::enabled() {
            let mlflow_state = super::mlflow::MlflowState::new(
                std::sync::Arc::new(
                    crate::services::mlflow_run_store::SubstrateRunStoreFactory::new(
                        state.document_service.clone(),
                    ),
                ),
                state.model_registry_service.clone(),
                std::sync::Arc::new(crate::services::mlflow_run_store::LocalFsArtifacts::new(
                    state.data_dir.join("mlflow_artifacts"),
                )),
            );
            let mlflow_router = super::mlflow::mlflow_routes().with_state(mlflow_state.clone());
            base_router = base_router.nest("/api/2.0/mlflow", mlflow_router);
            let artifacts_router = super::mlflow::artifacts_router().with_state(mlflow_state);
            // merge (not nest): the artifacts family carries its full
            // absolute path and must combine with, not nest under, the base.
            base_router = base_router.merge(artifacts_router);
            tracing::info!(
                "✅ MLflow compatibility wire enabled at /api/2.0/mlflow (+ artifacts proxy)"
            );
        }

        // Unmatched routes (incl. the removed v1 surfaces) return the canonical
        // error envelope with a migration hint pointing at the v2 replacement.
        base_router = base_router.fallback(not_found_fallback);

        // Add WebSocket streaming routes
        let ws_state = super::websocket::WebSocketState::new();
        let ws_routes = super::websocket::websocket_routes(ws_state);
        base_router = base_router.nest("/ws", ws_routes);
        tracing::info!("✅ WebSocket streaming enabled at /ws");

        // Create CORS layer using secure configuration
        let cors_layer = create_cors_layer(&security_config.cors);

        // Create timeout layer (if enabled)
        let timeout_layer = create_timeout_layer(&security_config.timeout);

        // Create concurrency limit layer for backpressure (if enabled)
        let concurrency_layer = create_concurrency_limit_layer(&security_config.backpressure);

        // Authentication layer (unified): prefer SecurityCoordinator if available.
        // Auth ENABLED with NO coordinator is a security misconfiguration: rather
        // than serving every request unauthenticated (failing open), we fail
        // CLOSED — the deny layer applied after the router is built rejects the
        // data plane with 503. (Computed before the coordinator is moved below.)
        let auth_misconfigured = security_config.auth.enabled && security_coordinator.is_none();
        let auth_layer = if security_config.auth.enabled {
            if let Some(coordinator) = security_coordinator {
                Some(middleware::from_fn_with_state(
                    coordinator,
                    crate::network::auth::middleware::auth_middleware_unified,
                ))
            } else {
                tracing::error!(
                    "REST auth is enabled but no SecurityCoordinator is configured — \
                     failing CLOSED: data-plane (/api/*) requests are rejected with 503. \
                     Wire a SecurityCoordinator or set auth.enabled=false."
                );
                None
            }
        } else {
            None
        };

        // Tenant extraction layer for multi-tenant isolation.
        // Env overrides (PROXIMADB_TENANT_HEADER_TRUST) are applied here — at
        // server construction, never inside constructors — so tests stay hermetic.
        let mut tenant_extractor =
            TenantExtractor::with_config(security_config.tenant.clone().apply_env_overrides());
        if let Some(resolver) = &tenant_stable_id_resolver {
            tenant_extractor = tenant_extractor.with_stable_id_resolver(resolver.clone());
        }
        let tenant_layer = middleware::from_fn_with_state(tenant_extractor, tenant_middleware);

        // Log security configuration
        if security_config.development_mode {
            tracing::warn!("⚠️  REST server security: DEVELOPMENT MODE - CORS allows any origin");
        } else {
            tracing::info!(
                "🔒 REST server security: Production mode - CORS whitelist: {:?}",
                security_config.cors.allowed_origins
            );
        }

        // Log backpressure configuration
        if security_config.backpressure.enabled {
            tracing::info!(
                "🛡️  Backpressure enabled: max {} concurrent requests",
                security_config.backpressure.max_concurrent_requests
            );
        }

        // Apply concurrency limit layer first (if enabled) for early rejection
        let base_router = if let Some(concurrency) = concurrency_layer {
            base_router.layer(concurrency)
        } else {
            base_router
        };

        // Add request ID middleware for tracing and correlation
        let base_router = base_router.layer(middleware::from_fn(request_id_middleware));

        // Per-route REST metrics. `route_layer` runs post-routing so the
        // matched-path label is available (bounded cardinality); unmatched
        // requests fall through to the 404 fallback and are not metered here.
        let base_router = base_router.route_layer(middleware::from_fn(
            crate::network::middleware::metrics::metrics_middleware,
        ));

        let mut router = if compression {
            // Create compression layer with support for multiple algorithms
            // Priority order (fastest to best compression): deflate, gzip, zstd, brotli
            let compression_layer = CompressionLayer::new()
                .deflate(true) // Fastest, low CPU usage
                .gzip(true) // Good balance of speed and compression
                .zstd(true) // Best compression ratio with good speed
                .br(true); // Brotli - slower but excellent compression

            // Create decompression layer for handling compressed requests
            let decompression_layer = DecompressionLayer::new()
                .deflate(true)
                .gzip(true)
                .br(true)
                .zstd(true);

            // IMPORTANT: Layer ordering matters for security!
            // In Tower ServiceBuilder, layers wrap previous layers, so request flow is:
            // cors -> trace -> compression -> body_limit -> decompression -> handler
            //
            // We want body limit BEFORE decompression to prevent decompression bombs
            // (small compressed payload expanding to huge uncompressed data causing OOM)
            let service_builder = ServiceBuilder::new()
                .layer(decompression_layer) // Handle compressed requests (innermost)
                .layer(DefaultBodyLimit::max(max_size_bytes as usize)) // Body limit BEFORE decompression
                .layer(compression_layer) // Compress responses
                .layer(TraceLayer::new_for_http())
                .layer(cors_layer);

            // Add timeout layer if enabled
            if let Some(timeout) = timeout_layer {
                base_router.layer(service_builder.layer(timeout))
            } else {
                base_router.layer(service_builder)
            }
        } else {
            let service_builder = ServiceBuilder::new()
                .layer(DefaultBodyLimit::max(max_size_bytes as usize))
                .layer(TraceLayer::new_for_http())
                .layer(cors_layer);

            // Add timeout layer if enabled
            if let Some(timeout) = timeout_layer {
                base_router.layer(service_builder.layer(timeout))
            } else {
                base_router.layer(service_builder)
            }
        };

        // Apply tenant layer (runs after auth to access JWT claims)
        router = router.layer(tenant_layer);
        tracing::info!(
            "🏢 Tenant isolation: {}",
            if security_config.tenant.require_tenant {
                "MULTI-TENANT (tenant ID required)"
            } else {
                "SINGLE-TENANT (default tenant fallback)"
            }
        );

        // Apply auth layer last so request IDs/backpressure/cors are preserved
        if let Some(auth) = auth_layer {
            router = router.layer(auth);
        }
        // Fail closed on an auth misconfiguration (enabled but no coordinator):
        // deny the data plane with 503 rather than serving it unauthenticated.
        if auth_misconfigured {
            router = router.layer(middleware::from_fn(
                crate::network::middleware::auth_failclosed::auth_misconfigured_deny_data_plane,
            ));
        }

        // Build TLS config if specified
        let tls_config = security_config.tls.as_ref().map(|tls| {
            let mut config = NetworkTlsConfig::new(true)
                .with_cert_file(tls.cert_file.clone())
                .with_key_file(tls.key_file.clone());
            // Only add CA file if mTLS is enabled and CA file is provided
            if let Some(ca_file) = &tls.ca_file {
                config = config.with_ca_file(ca_file.clone());
            }
            config
        });

        if tls_config.is_some() {
            tracing::info!("TLS configured for REST server");
        }

        Self {
            router,
            bind_addr,
            tls_config,
            uds_path: None,
        }
    }

    /// Portless ("embedded") mode: serve the plaintext REST surface over the
    /// given Unix-domain socket path instead of a TCP port. `None` (default)
    /// keeps TCP. The `/health` readiness probe stays reachable over the UDS.
    pub fn with_uds_path(mut self, uds_path: Option<PathBuf>) -> Self {
        self.uds_path = uds_path;
        self
    }

    /// Build a REST router for unified mode without starting a server.
    ///
    /// `ports` injects port-backed handlers from `proximadb-api` for document,
    /// graph, and observability routes. When `None`, the legacy root-crate
    /// handlers are used.
    pub fn build_router_for_unified(
        core: RestCoreServices,
        graph_execution_service: Arc<dyn GraphExecutionService>,
        metrics_collector: Option<Arc<MetricsCollector>>,
        security_coordinator: Option<Arc<SecurityCoordinator>>,
        data_dir: std::path::PathBuf,
        query_adapter: Option<Arc<crate::query::facade::QueryFacadeAdapter>>,
        llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
        ports: Option<RestServerPorts>,
        segment_registry: Option<Arc<crate::catalog::SegmentRegistry>>,
        catalog_manager: Option<Arc<crate::catalog::CatalogManager>>,
        queue_client: Option<Arc<proximadb_queue::QueueClient>>,
        fulltext_indexes: Option<crate::network::rest::canonical::handlers::FullTextIndexMap>,
        recall_probe_gate: Option<Arc<crate::catalog::RecallProbeGate>>,
        rank_services: Option<Arc<crate::network::rest::canonical::rank::RankServices>>,
        rank_profile_store: Option<Arc<dyn crate::services::RankProfileStore>>,
        discovery_service: Option<Arc<crate::services::discovery::DiscoveryService>>,
        external_collection_service: Option<
            Arc<crate::services::external_collection::ExternalCollectionService>,
        >,
        tenant_config: TenantExtractorConfig,
        // Mount the read-only `/admin` dashboard (from `[server.admin_ui] enabled`).
        admin_ui_enabled: bool,
        // Dashboard auto-refresh (from `[server.admin_ui] auto_refresh`).
        admin_ui_auto_refresh: bool,
        // Dashboard auto-refresh interval seconds (from
        // `[server.admin_ui] refresh_interval_seconds`).
        admin_ui_refresh_interval_seconds: u32,
    ) -> Router {
        let mut base_state = AppState::new(
            core,
            graph_execution_service,
            security_coordinator.clone(),
            data_dir,
            query_adapter.clone(),
            llm_engine,
        );
        if let Some(reg) = segment_registry {
            base_state = base_state.with_segment_registry(reg);
        }
        // TD-TENANT-1 item 3: build the tenant-stable-id resolver from the
        // catalog handle before it's moved into base_state below.
        let tenant_stable_id_resolver = catalog_manager.as_ref().map(|cm| {
            Arc::new(crate::security::CatalogTenantStableIdResolver::new(
                cm.clone(),
            )) as Arc<dyn proximadb_tenant::TenantStableIdResolver>
        });
        if let Some(manager) = catalog_manager {
            base_state = base_state.with_catalog_manager(manager);
        }
        if let Some(qc) = queue_client {
            base_state = base_state.with_queue_client(qc);
        }
        // T3.2 Slice 1b: share `fulltext_indexes` with `SharedServices` so REST
        // hybrid search and gRPC hybrid search read+write the same in-process
        // map. Without this, REST has its own empty map and gRPC's BM25
        // component returns 0 results.
        if let Some(indexes) = fulltext_indexes.clone() {
            base_state = base_state.with_fulltext_indexes(indexes);
        }
        // TD-064 / LLD §5: share the recall-probe gate so v2 route-health
        // can resolve per-scope `gate_open` state without re-constructing.
        if let Some(gate) = recall_probe_gate {
            base_state = base_state.with_recall_probe_gate(gate);
        }
        // R-7c.3 production wiring: the REST `/api/v2/rank/search` route and
        // pgwire SQL `RERANK(...)` share the process-wide rank pipeline. Keep
        // the durable profile catalog in AppState for the profile dispatchers;
        // those management routes are not mounted yet.
        if let Some(services) = rank_services {
            base_state = base_state.with_rank_services(services);
        }
        if let Some(store) = rank_profile_store {
            base_state = base_state.with_rank_profile_store(store);
        }
        // Phase 8 (F1): share the Continuous Discovery service so the v2
        // `discovery-jobs` endpoints reach the same registry the background
        // executor consumes.
        if let Some(discovery) = discovery_service {
            base_state = base_state.with_discovery_service(discovery);
        }
        // Phase 8 (F5): share the External Collection service so the v2
        // `external-collections` endpoints reach the same registry.
        if let Some(external) = external_collection_service {
            base_state = base_state.with_external_collection_service(external);
        }
        let state = if let Some(p) = ports {
            // Always wire the runtime port-based api_handlers (TD-104 S1); the
            // document/graph/observability route ports are mounted only when present.
            let mut s = base_state.with_api_handlers(p.api_handlers);
            if let (Some(doc_port), Some(graph_port), Some(obs_port)) =
                (p.doc_port, p.graph_port, p.obs_port)
            {
                s = s.with_ports(doc_port, graph_port, obs_port);
            }
            s
        } else {
            base_state
        };
        let state = if let Some(ref adapter) = query_adapter {
            let port = Arc::new(
                crate::query::UnifiedQueryPortImpl::new(adapter.clone())
                    .with_catalog_manager(state.catalog_manager.clone()),
            ) as Arc<dyn proximadb_runtime::UnifiedQueryPort>;
            state.with_unified_query_port(port)
        } else {
            state
        };

        // Create metrics router if metrics collector is available
        let metrics_router = if let Some(collector) = metrics_collector {
            use crate::network::metrics_service::{MetricsService, MetricsServiceConfig};
            let metrics_config = MetricsServiceConfig::default();
            let metrics_service = MetricsService::new(metrics_config, collector);
            Some(metrics_service.create_router())
        } else {
            None
        };

        // Build base router with all endpoints
        let state_for_v2 = state.clone();
        let mut base_router = create_router(state.clone());

        // TD-V1SUNSET-1 RESOLVED (2026-08-29): the v1_sunset middleware is
        // REMOVED. /api/v1/* now falls to `not_found_fallback`, which answers
        // 404 + the exact /api/v2 replacement endpoint + docs link via
        // `v1_replacement_for` — the migration signal survives; only the
        // machine-readable Deprecation/Sunset headers are gone. Closure
        // evidence: every known client accounted for (SDKs + AnvaiOps — no
        // ProximaDB v1 calls, no REST_V1_COMPAT opt-ins), and /api/v1 has
        // been 410-by-default since #814 (2026-07-09).

        // TD-RATE-1: mount the REST request limiter, default OFF — the gate
        // being unset adds NO layer, so the default deployment is unchanged.
        // Keyed on the TD-TENANT-4 observed client address; `/health*` is
        // exempt (the middleware checks `limit_health_endpoints`, false by
        // default). Positioned beside the v1-sunset layer, i.e. inside the
        // auth/tenant layers: floods are counted after authentication, which
        // keeps per-IP accounting aligned with the authenticated principal.
        if let Some(limiter) =
            crate::network::middleware::rate_limit::rest_rate_limit_state_from_env()
        {
            tracing::info!(
                rpm_env = crate::network::middleware::rate_limit::REST_RATE_LIMIT_RPM_ENV,
                "REST request rate limiting enabled (TD-RATE-1)"
            );
            base_router = base_router.layer(middleware::from_fn_with_state(
                limiter,
                crate::network::middleware::rate_limit::rate_limit_middleware,
            ));
        }

        // Nest metrics router if available
        if let Some(metrics) = metrics_router {
            base_router = base_router.nest("/metrics", metrics);
            tracing::info!("✅ Metrics endpoints enabled at /metrics (unified mode)");
        }

        // Read-only admin dashboard at /admin (+ back-compat /dashboard alias),
        // gated by `[server.admin_ui] enabled` (off by default — empty router when
        // disabled). Mounted from the self-contained `proximadb-admin-ui` crate.
        // The auto-refresh fields are server-injected into the page so the toggle
        // reflects the operator's `[server.admin_ui]` default.
        base_router = base_router.merge(proximadb_admin_ui::admin_router_if(
            proximadb_admin_ui::AdminUiOptions {
                enabled: admin_ui_enabled,
                auto_refresh: admin_ui_auto_refresh,
                refresh_interval_seconds: admin_ui_refresh_interval_seconds,
            },
        ));
        if admin_ui_enabled {
            tracing::info!("🖥️  Read-only admin dashboard enabled at /admin (+ /dashboard)");
        }

        // Add V2 API router with ProximaRecord support
        let v2_router = super::v2::create_v2_router().with_state(state_for_v2.clone());
        base_router = base_router.nest("/api/v2", v2_router);
        tracing::info!("✅ V2 API enabled at /api/v2 (unified mode)");

        // TD-MLOPS-1 slice 2: MLflow-compatible tracking wire, default OFF.
        if super::mlflow::enabled() {
            let mlflow_state = super::mlflow::MlflowState::new(
                std::sync::Arc::new(
                    crate::services::mlflow_run_store::SubstrateRunStoreFactory::new(
                        state.document_service.clone(),
                    ),
                ),
                state.model_registry_service.clone(),
                std::sync::Arc::new(crate::services::mlflow_run_store::LocalFsArtifacts::new(
                    state.data_dir.join("mlflow_artifacts"),
                )),
            );
            let mlflow_router = super::mlflow::mlflow_routes().with_state(mlflow_state.clone());
            base_router = base_router.nest("/api/2.0/mlflow", mlflow_router);
            let artifacts_router = super::mlflow::artifacts_router().with_state(mlflow_state);
            // merge (not nest): the artifacts family carries its full
            // absolute path and must combine with, not nest under, the base.
            base_router = base_router.merge(artifacts_router);
            tracing::info!(
                "✅ MLflow compatibility wire enabled at /api/2.0/mlflow (+ artifacts proxy)"
            );
        }

        // Unmatched routes (incl. removed v1 surfaces) → canonical 404 + hint.
        base_router = base_router.fallback(not_found_fallback);

        // Add WebSocket streaming routes
        let ws_state = super::websocket::WebSocketState::new();
        let ws_routes = super::websocket::websocket_routes(ws_state);
        base_router = base_router.nest("/ws", ws_routes);
        tracing::info!("✅ WebSocket streaming enabled at /ws (unified mode)");

        // Tenant extraction layer for unified mode. v2/v3 handlers require an
        // `Extension<MiddlewareTenantContext>`; without this layer every
        // `/api/v2` request 500s with "missing request extension". The
        // multi-port path applies the same layer in `create_router`; the
        // unified path must apply it too (it is NOT wrapped by the caller).
        // The caller supplies the same deployment-derived policy used by the
        // multi-port path. Environment overrides are applied once here.
        let mut tenant_extractor =
            TenantExtractor::with_config(tenant_config.apply_env_overrides());
        if let Some(resolver) = &tenant_stable_id_resolver {
            tenant_extractor = tenant_extractor.with_stable_id_resolver(resolver.clone());
        }
        let tenant_layer = middleware::from_fn_with_state(tenant_extractor, tenant_middleware);

        // Auth layer — convergent with the multi-port `start_with_security`
        // path. When the caller supplies a `SecurityCoordinator`, attach
        // `auth_middleware_unified` so handlers that declare
        // `Option<Extension<UnifiedUserContext>>` see Some(ctx) for
        // authenticated requests. When `security_coordinator` is `None`
        // (auth disabled by config), no layer is attached and the
        // operator-gated handlers fall back to their `require_*_admin`
        // single-node bypass. multi_server.rs gates the coordinator it
        // passes by `rest_auth_enabled`, so this branch fires iff the
        // operator explicitly enabled REST auth — matching the
        // `start_with_security` gate at line 510.
        let auth_layer = security_coordinator.clone().map(|coordinator| {
            middleware::from_fn_with_state(
                coordinator,
                crate::network::auth::middleware::auth_middleware_unified,
            )
        });

        // Layer order:
        //   outermost: TraceLayer (sees every request, including unauth'd)
        //   tenant_layer (needs auth-injected ctx for tenant resolution)
        //   auth_layer  (injects UnifiedUserContext)
        // Axum applies layers in reverse order, so auth runs first on
        // the request path, then tenant, then handler.
        use tower_http::trace::TraceLayer;
        let mut router = base_router.layer(tenant_layer);
        if let Some(auth) = auth_layer {
            router = router.layer(auth);
            tracing::info!("🔒 Unified-port REST auth: ENABLED (auth_middleware_unified attached)");
        } else {
            tracing::info!(
                "🔓 Unified-port REST auth: DISABLED (no security coordinator supplied)"
            );
        }
        router.layer(TraceLayer::new_for_http())
    }

    /// Start the REST server
    pub async fn start(self) -> anyhow::Result<()> {
        if self.tls_config.is_some() {
            self.start_with_tls().await
        } else {
            self.start_plaintext().await
        }
    }

    /// Start the REST server with TLS
    async fn start_with_tls(self) -> anyhow::Result<()> {
        let tls_config = self
            .tls_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TLS config required for TLS server"))?;

        let (cert_path, key_path) = tls_config
            .get_certificate_paths()
            .ok_or_else(|| anyhow::anyhow!("Certificate paths not configured"))?;

        let require_client_certs = tls_config.require_client_certs;
        let ca_path = tls_config.get_ca_path();

        tracing::info!("Starting REST server with TLS on {}", self.bind_addr);
        tracing::info!("  Certificate: {:?}", cert_path);
        tracing::info!("  Private key: {:?}", key_path);
        tracing::info!(
            "  mTLS mode: {}",
            if require_client_certs {
                "ENABLED"
            } else {
                "DISABLED"
            }
        );
        if let Some(ref ca) = ca_path {
            tracing::info!("  CA Certificate: {:?}", ca);
        }

        Self::log_endpoints(&self.bind_addr, true);

        // Build rustls config - either mTLS or standard TLS
        if require_client_certs {
            if let Some(ca) = ca_path {
                self.start_with_mtls(cert_path, key_path, ca).await
            } else {
                Err(anyhow::anyhow!(
                    "Client certificates required but CA path not provided"
                ))
            }
        } else {
            // Standard TLS (no client certificates)
            let rustls_config =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to load TLS certificates: {}", e))?;

            // TD-TENANT-4: observe the real peer address (see start_plaintext).
            axum_server::bind_rustls(self.bind_addr, rustls_config)
                .serve(
                    self.router
                        .into_make_service_with_connect_info::<SocketAddr>(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("TLS server error: {}", e))?;

            Ok(())
        }
    }

    /// Start REST server with mTLS (mutual TLS) - requires client certificates
    async fn start_with_mtls(
        self,
        cert_path: PathBuf,
        key_path: PathBuf,
        ca_path: PathBuf,
    ) -> anyhow::Result<()> {
        use crate::network::tls::certificate_manager::utils::{
            load_certs_from_pem, load_private_key_from_pem,
        };
        use rustls::server::WebPkiClientVerifier;
        use rustls::{RootCertStore, ServerConfig};
        use std::sync::Arc;

        // rustls 0.23 requires a process-default CryptoProvider before builder(); idempotent.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

        tracing::info!("Configuring mTLS with client certificate verification");

        // Load server certificate and key
        let cert_pem = tokio::fs::read(&cert_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read server certificate: {}", e))?;
        let key_pem = tokio::fs::read(&key_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read server private key: {}", e))?;

        let certs = load_certs_from_pem(&cert_pem)
            .map_err(|e| anyhow::anyhow!("Failed to parse server certificate: {}", e))?;
        let key = load_private_key_from_pem(&key_pem)
            .map_err(|e| anyhow::anyhow!("Failed to parse server private key: {}", e))?;

        // Load CA certificate for client verification
        let ca_pem = tokio::fs::read(&ca_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read CA certificate: {}", e))?;
        let ca_certs = load_certs_from_pem(&ca_pem)
            .map_err(|e| anyhow::anyhow!("Failed to parse CA certificate: {}", e))?;

        // Build root cert store for client verification
        let mut root_store = RootCertStore::empty();
        for ca_cert in ca_certs {
            root_store.add(ca_cert).map_err(|e| {
                anyhow::anyhow!("Failed to add CA certificate to root store: {}", e)
            })?;
        }

        // Client certificate verifier — accept any cert chaining to our CA.
        let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build client cert verifier: {}", e))?;

        // Build mTLS server config (rustls 0.23 dropped with_safe_defaults()).
        let server_config = ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(certs, key)
            .map_err(|e| anyhow::anyhow!("Failed to build mTLS server config: {}", e))?;

        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config));

        tracing::info!("mTLS server configured - client certificates will be verified against CA");
        tracing::info!("Note: Client certificate info will be available in request extensions");

        // TD-TENANT-4: observe the real peer address (see start_plaintext).
        axum_server::bind_rustls(self.bind_addr, rustls_config)
            .serve(
                self.router
                    .into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("mTLS server error: {}", e))?;

        Ok(())
    }

    /// Start the REST server without TLS (plaintext)
    async fn start_plaintext(self) -> anyhow::Result<()> {
        // Portless ("embedded") mode: serve over a Unix-domain socket (unix-only).
        if let Some(uds_path) = self.uds_path.clone() {
            #[cfg(unix)]
            {
                tracing::info!(
                    "Starting REST server (plaintext) on unix:{}",
                    uds_path.display()
                );
                Self::log_endpoints(&self.bind_addr, false);
                // axum 0.8 serves any `Listener`; tokio's UnixListener qualifies.
                // TD-TENANT-4: deliberately NOT `_with_connect_info::<SocketAddr>`
                // — a Unix socket has no `SocketAddr`, so demanding one here
                // would not compile. A UDS peer is on the same host by
                // construction and classifies as local, which is correct rather
                // than a fallback.
                let listener = crate::network::uds::bind_unix_listener(&uds_path).map_err(|e| {
                    anyhow::anyhow!("REST UDS bind failed at {}: {}", uds_path.display(), e)
                })?;
                // Mark the surface instead, so a UDS request resolves as
                // `LocalSocket` rather than `Unobserved` — the latter must keep
                // meaning "ConnectInfo was not wired here".
                let router = self.router.layer(axum::Extension(
                    proximadb_network_middleware::client_addr::LocalSocketPeer,
                ));
                axum::serve(listener, router.into_make_service()).await?;
                return Ok(());
            }
            #[cfg(not(unix))]
            {
                anyhow::bail!(
                    "REST UDS (unix:{}) is not supported on this platform; use TCP",
                    uds_path.display()
                );
            }
        }

        tracing::info!("Starting REST server (plaintext) on {}", self.bind_addr);

        Self::log_endpoints(&self.bind_addr, false);

        // axum 0.8 / hyper 1.0: bind a tokio listener, then axum::serve.
        // TD-TENANT-4: `_with_connect_info` supplies the real peer address so
        // the edge classifies an OBSERVED client instead of a header claim.
        let listener = tokio::net::TcpListener::bind(&self.bind_addr).await?;
        axum::serve(
            listener,
            self.router
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await?;

        Ok(())
    }

    /// Log available endpoints
    fn log_endpoints(bind_addr: &SocketAddr, tls: bool) {
        let protocol = if tls { "https" } else { "http" };
        tracing::info!(
            "REST server using canonical v2 record routes plus v1 compatibility adapters"
        );
        tracing::info!("REST server listening on {}://{}", protocol, bind_addr);
        tracing::info!("Compression enabled: deflate, gzip, zstd, brotli (in priority order)");
        tracing::info!("Available endpoints:");
        tracing::info!("   GET    /health                           - Health check");
        tracing::info!("   GET    /dashboard                        - Web dashboard");
        tracing::info!("   GET    /metrics                          - Prometheus metrics");
        tracing::info!("   GET    /metrics/json                     - JSON metrics");
        tracing::info!("   GET    /metrics/health                   - Metrics health check");
        tracing::info!(
            "   POST   /api/v2/collections/:id/search     - Canonical record/vector search"
        );
        tracing::info!(
            "   POST   /api/v2/collections/:id/records/batch - Canonical ProximaRecord writes"
        );
        tracing::info!("Auxiliary endpoints:");
        tracing::info!("   POST   /api/v1/progressive/search/:id    - Progressive search (JSON)");
        // Legacy /api/v1 vector/collection/search REST surfaces are removed;
        // those paths return a /api/v2 migration hint (see `v1_replacement_for`).
        // Do not re-advertise them here.
        tracing::info!("   WS     /ws/insert                        - WebSocket vector streaming");
        tracing::info!(
            "   WS     /ws/subscribe                     - WebSocket live query subscription"
        );
        tracing::info!("   WS     /ws/status                        - WebSocket session status");
    }
}

/// 404 fallback handler for unmatched paths. Returns the canonical error envelope
/// and, for paths under the removed `/api/v1/*` surfaces, a migration hint
/// pointing at the v2 replacement.
///
/// `pub(crate)` so the router-level spec-path drift smoke test
/// (`canonical::handlers::all_openapi_spec_paths_resolve_to_a_route`) can mount
/// the SAME fallback the production routers use, and distinguish a genuine
/// routing-404 (`"No route for ..."`) from a handler-level 404 (e.g. collection
/// not found) — the distinction the drift gate hinges on.
pub(crate) async fn not_found_fallback(uri: axum::http::Uri) -> axum::response::Response {
    use axum::response::IntoResponse;
    let path = uri.path();

    let mut error = serde_json::json!({
        "type": "not_found",
        "code": 404u16,
    });
    if let Some(replacement) = v1_replacement_for(path) {
        error["message"] = serde_json::json!(format!(
            "{} was removed in the /api/v2 API standardization; use {}",
            path, replacement
        ));
        error["details"] = serde_json::json!({
            "removed_endpoint": path,
            "replacement_endpoint": replacement,
            "docs": "https://docs.proximadb.io/api/v2",
        });
    } else {
        error["message"] = serde_json::json!(format!("No route for {}", path));
    }
    if let Some(rid) = proximadb_api::rest::errors::current_request_id() {
        error["request_id"] = serde_json::json!(rid);
    }

    (
        axum::http::StatusCode::NOT_FOUND,
        axum::Json(serde_json::json!({ "error": error })),
    )
        .into_response()
}

/// Coarse old→new map for the v1 surfaces removed in the API standardization.
fn v1_replacement_for(path: &str) -> Option<&'static str> {
    let table: &[(&str, &str)] = &[
        ("/api/v1/collections", "/api/v2/collections"),
        ("/api/v1/search", "/api/v2/collections/{id}/search"),
        ("/api/v1/vectors", "/api/v2/collections/{id}/records"),
        ("/api/v1/sql", "/api/v2/query"),
        ("/api/v1/documents", "/api/v2/document-collections"),
        ("/api/v1/hybrid", "/api/v2/hybrid"),
        ("/api/v1/observability", "/api/v2/observability"),
        ("/api/v1/graph", "/api/v2/graphs"),
    ];
    for (prefix, replacement) in table {
        if path.starts_with(prefix) {
            return Some(replacement);
        }
    }
    if path.starts_with("/api/v1/") {
        return Some("/api/v2");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-V1SUNSET-1 successor contract (the middleware is REMOVED): a
    /// removed /api/v1 path must fall to `not_found_fallback` and answer 404
    /// **with the /api/v2 replacement named in the body**. This is what makes
    /// the removal non-silent — the migration signal survives the middleware.
    /// Negative-controllable: without `v1_replacement_for`, the body carries
    /// only "No route for ..." and this test fails.
    #[tokio::test]
    async fn removed_v1_paths_404_with_the_v2_replacement_named() {
        let resp =
            not_found_fallback("/api/v1/collections".parse::<axum::http::Uri>().unwrap()).await;
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body).to_string();
        assert!(
            text.contains("/api/v2"),
            "404 body must name the v2 replacement, got: {text}"
        );
    }

    use axum::body::Body;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use axum::middleware;
    use axum::routing::get;
    use tower::ServiceExt;

    use crate::network::auth::middleware::auth_middleware_unified;
    use crate::security::auth_service::{
        ApiKeyInfo, AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig,
    };
    use crate::security::rbac_service::RBACConfig;
    use crate::security::security_coordinator::{ComplianceConfig, TlsConfig};
    use crate::security::{AuditConfig, SecurityConfig, SecurityCoordinator, SecurityMode};

    fn build_api_key_security_config() -> SecurityConfig {
        let mut api_keys = std::collections::HashMap::new();
        api_keys.insert(
            "test-key".to_string(),
            ApiKeyInfo {
                user_id: "user1".to_string(),
                tenant_id: None,
                permissions: vec!["read".to_string()],
                roles: vec![],
                created_at: None,
                expires_at: None,
                rate_limit_per_minute: None,
                ip_restrictions: vec![],
            },
        );

        SecurityConfig {
            enabled: true,
            mode: SecurityMode::Development,
            authentication: AuthenticationConfig {
                enabled: true,
                methods: vec![AuthenticationMethod::ApiKey],
                require_authentication: true,
                default_session_timeout_minutes: 60,
                api_keys,
                jwt: JwtConfig {
                    enabled: false,
                    secret: "dev-secret".to_string(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: "test".to_string(),
                    audience: "test".to_string(),
                    algorithm: "HS256".to_string(),
                },
                sso: SSOConfig {
                    enabled: false,
                    providers: vec![],
                    token_cache_ttl_minutes: 5,
                },
                mtls: MtlsConfig::default(),
                oidc: None,
                audit_fail_closed: false,
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
                frameworks: vec![],
                data_residency: None,
                encryption_at_rest: false,
                encryption_in_transit: false,
            },
            encryption: crate::security::EncryptionConfig::default(),
            key_store: crate::security::KeyStoreConfig::default(),
            tenant: Default::default(),
        }
    }

    async fn build_test_router(auth_enabled: bool) -> Router {
        let base_router = Router::new().route("/api/v1/search", get(|| async { "ok" }));

        if auth_enabled {
            let coordinator = Arc::new(
                SecurityCoordinator::from_config(build_api_key_security_config())
                    .await
                    .unwrap(),
            );
            base_router.layer(middleware::from_fn_with_state(
                coordinator,
                auth_middleware_unified,
            ))
        } else {
            base_router
        }
    }

    #[tokio::test]
    async fn rest_auth_disabled_allows_requests_without_header() {
        let router = build_test_router(false).await;
        let request = Request::builder()
            .uri("/api/v1/search")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn rest_auth_enabled_requires_header() {
        let router = build_test_router(true).await;
        let request = Request::builder()
            .uri("/api/v1/search")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rest_auth_enabled_accepts_api_key() {
        let router = build_test_router(true).await;
        let request = Request::builder()
            .uri("/api/v1/search")
            .header("Authorization", "Api-Key test-key")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }
}
