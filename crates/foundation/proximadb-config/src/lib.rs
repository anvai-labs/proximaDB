//! Typed configuration contracts shared across ProximaDB workspace crates.
//!
//! Keep this crate limited to serializable configuration shapes. Runtime conversion and service
//! bootstrap stay in platform/root layers until those boundaries are independently extracted.

pub mod cdc_config;
pub mod cluster_config;
pub mod llm_config;

pub use cdc_config::{
    BatchConfig, CaptureConfig, CaptureOperation, CdcConfig, CdcSettings, ConnectionConfig,
    DeliveryConfig, DeliveryGuarantee, OffsetStorageConfig, OffsetStorageType, SinkConfig,
    SinkType, SnapshotConfig, SnapshotMode, SourceConfig, SourceType, SslMode, TransformConfig,
    TransformType,
};
pub use llm_config::{
    AWSBedrockConfig, AzureOpenAIConfig, EmbeddingProvider, FinishReason, GoogleVertexConfig,
    HuggingFaceConfig, LLMConfig, LLMError, LLMProvider, LLMRequest, LLMRequestContext,
    LLMResponse, OllamaConfig, ProviderHealthStatus, RAGConfig, RequestPriority,
    SemanticCacheConfig, TokenUsage, VLLMConfig,
};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Declare a memoized boolean env gate (CLAUDE.md Env-Gate-Registry #21).
///
/// Generates the canonical `OnceLock<bool>` getter the codebase hand-copied
/// ~12 times (victor post-program clone scan, cluster A5): read `$gate` once,
/// accept the registry's documented truthy set (`1|true|on|yes`,
/// case-insensitive); fall back to `$default` only when UNSET — a set
/// but unrecognized value evaluates false (matching every hand-written
/// body this replaces).
/// Note the widened truthy set: the hand-written getters accepted only
/// `1|true` — this macro aligns the code with the registry contract.
///
/// The gate name MUST be a string literal so `scripts/check_env_gates.py`
/// (which greps literal `PROXIMADB_*` names out of `.rs` sources) keeps
/// seeing it; never derive the name from the function identifier. Registry
/// rows stay hand-maintained in `docs/12-design/ENV_GATE_REGISTRY.adoc`.
///
/// ```ignore
/// proximadb_config::env_gate! {
///     /// Gate: native vectorized execution (ADR-054). Default OFF.
///     pub fn native_vectorized_enabled() { gate: "PROXIMADB_NATIVE_VECTORIZED", default: false }
/// }
/// ```
#[macro_export]
macro_rules! env_gate {
    ($(#[$meta:meta])* $vis:vis fn $name:ident() { gate: $gate:literal, default: $default:expr }) => {
        $(#[$meta])*
        $vis fn $name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| {
                std::env::var($gate)
                    .map(|v| {
                        matches!(
                            v.to_ascii_lowercase().as_str(),
                            "1" | "true" | "on" | "yes"
                        )
                    })
                    .unwrap_or($default)
            })
        }
    };
}

/// Declare a KILL-SWITCH env gate: ON by default (or `$default`), switched
/// OFF by `0 | false | off | no` (case-insensitive, surrounding whitespace
/// tolerated). The dual of [`env_gate!`] — opt-in gates default OFF and
/// enable; kill-switches default ON and disable.
///
/// DELIBERATELY NOT MEMOIZED, unlike `env_gate!`: kill-switches sit on
/// cold/error paths where the read cost is irrelevant, and tests flip them
/// mid-process (e.g. `ident_case_fold_e2e.rs` toggles the gate between
/// cases) — a `OnceLock` would freeze the first read forever. Prefer
/// `env_gate!` for hot-path opt-in gates.
///
/// ```ignore
/// proximadb_config::env_kill_switch! {
///     /// Kill-switch: identifier case folding (TD-OLAP-18). Default ON.
///     pub fn ident_case_fold_enabled() { gate: "PROXIMADB_IDENT_CASE_FOLD", default: true }
/// }
/// ```
#[macro_export]
macro_rules! env_kill_switch {
    ($(#[$meta:meta])* $vis:vis fn $name:ident() { gate: $gate:literal, default: $default:expr }) => {
        $(#[$meta])*
        $vis fn $name() -> bool {
            std::env::var($gate)
                .map(|v| {
                    !matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "0" | "false" | "off" | "no"
                    )
                })
                .unwrap_or($default)
        }
    };
}

/// Non-memoized boolean env-gate read (the plain-fn sibling of [`env_gate!`]).
///
/// Same truthy set as the macro (`1 | true | on | yes`, case-insensitive),
/// default false when unset — but read on EVERY call: for gates consulted on
/// cold paths, or whose value tests mutate mid-process, the macro's
/// `OnceLock` memoization would freeze the first read forever
/// (`cold_payloads_enabled` is the canonical example — six tests set and
/// unset it around individual cases).
pub fn env_flag_now(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
        .unwrap_or(false)
}

/// TLS transport security configuration shared by protocol listeners.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to the PEM-encoded TLS certificate file.
    pub cert_file: Option<String>,

    /// Path to the PEM-encoded TLS private key file.
    pub key_file: Option<String>,

    /// Whether TLS is enabled.
    pub enabled: bool,

    /// Network interface to bind the TLS listener to.
    pub bind_interface: Option<String>,
}

/// REST, gRPC, and Arrow Flight API endpoint configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// gRPC listening port (used in multi-port mode).
    pub grpc_port: u16,

    /// REST listening port (used in multi-port mode).
    pub rest_port: u16,

    /// Maximum request body size in megabytes.
    pub max_request_size_mb: u64,

    /// Request timeout in seconds.
    pub timeout_seconds: u64,

    /// Whether TLS is enabled for API endpoints.
    pub enable_tls: Option<bool>,

    /// Interval for background TTL sweeper in seconds.
    pub ttl_sweep_interval_seconds: u64,

    /// Enable REST API compression.
    pub rest_compression: bool,

    /// Enable gRPC compression.
    pub grpc_compression: bool,

    /// Compression algorithm: "gzip", "deflate", "br".
    pub compression_algorithm: String,

    /// Compression level 1-9 for gzip, 1-11 for brotli.
    pub compression_level: i32,

    /// Enable unified port mode (REST + gRPC + Arrow Flight on single port).
    #[serde(default)]
    pub unified_mode: bool,

    /// Unified port for all HTTP-based protocols.
    #[serde(default = "default_unified_port")]
    pub unified_port: u16,

    /// Arrow Flight port (used when unified_mode = false).
    #[serde(default = "default_arrow_flight_port")]
    pub arrow_flight_port: u16,

    /// Enable REST protocol in unified mode.
    #[serde(default = "default_true")]
    pub enable_rest: bool,

    /// Enable gRPC protocol in unified mode.
    #[serde(default = "default_true")]
    pub enable_grpc: bool,

    /// Enable Arrow Flight protocol in unified mode.
    #[serde(default = "default_true")]
    pub enable_arrow_flight: bool,

    /// HTTP/2 max concurrent streams (for gRPC and Arrow Flight).
    #[serde(default = "default_http2_max_concurrent_streams")]
    pub http2_max_concurrent_streams: u32,

    /// Maximum connections for unified server.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Optional override for the PostgreSQL wire-protocol port.
    /// When `None`, the server keeps the
    /// `PostgresServerConfig` default (5433). Lets test fixtures
    /// allocate a free port and run pgwire tests in parallel without
    /// colliding with a real Postgres on 5432 / a sibling test on 5433.
    /// Production deployments typically leave this `None` and set the
    /// port via the `[api]` TOML or the standard env-var precedence.
    #[serde(default)]
    pub pg_port: Option<u16>,

    /// Enable the local trust-auth pgwire listener. Disable in authenticated
    /// or multi-tenant deployments until pgwire supports credential binding.
    #[serde(default = "default_true")]
    pub enable_pgwire: bool,

    /// Optional override for the unified-mode internal multiplexer upstream
    /// port (the loopback gRPC + Arrow Flight listener the TCP multiplexer
    /// forwards HTTP/2 to). When `None` (the default), the port is resolved
    /// from the `PROXIMADB_INTERNAL_MUX_PORT` env var, else derived as
    /// `unified_port + 10001` (5678 → 15679, the historical constant), so
    /// co-hosted instances with distinct unified ports get distinct internal
    /// upstreams instead of hijacking each other's gRPC/Flight traffic
    /// (TD-NET-1).
    #[serde(default)]
    pub internal_mux_port: Option<u16>,

    /// Optional port for the reference MCP (Model Context Protocol) surface
    /// (ADR-037 Decision 5). When `None` (the default) the MCP transport is
    /// **off** — it is bound only when a port is configured here (or via the
    /// `[api]` TOML / env-var precedence). In-process: the MCP server projects
    /// the engine's existing surfaces (stats/describe/explain/search) by calling
    /// the services directly, so there is no separate process.
    #[serde(default)]
    pub mcp_port: Option<u16>,

    /// Network transport for the REST / gRPC / Arrow Flight surfaces.
    ///
    /// - `"tcp"` (default): bind TCP ports (`rest_port` / `grpc_port` /
    ///   `arrow_flight_port`). This is the production/server default and
    ///   keeps every existing deployment unchanged (mixed-read-safe: UDS
    ///   is strictly opt-in).
    /// - `"uds"`: bind Unix-domain sockets under [`socket_dir`] instead of
    ///   TCP ports — the **portless** embedded mode. The `*_port` fields are
    ///   ignored, and the pgwire listener (a standard PG TCP driver) is
    ///   disabled, so the process opens *no* TCP listener at all.
    #[serde(default = "default_transport")]
    pub transport: String,

    /// Directory that holds the Unix-domain sockets when `transport = "uds"`.
    ///
    /// The three surfaces bind to `<socket_dir>/proximadb-embedded.rest.sock`,
    /// `…grpc.sock`, and `…flight.sock`. Required (and created if missing)
    /// when `transport = "uds"`; ignored for TCP.
    #[serde(default)]
    pub socket_dir: Option<String>,
}

fn default_transport() -> String {
    "tcp".to_string()
}

fn default_unified_port() -> u16 {
    5678
}

fn default_arrow_flight_port() -> u16 {
    5680
}

fn default_true() -> bool {
    true
}

fn default_http2_max_concurrent_streams() -> u32 {
    1000
}

fn default_max_connections() -> usize {
    10000
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            grpc_port: 5679,
            rest_port: 5678,
            max_request_size_mb: 100,
            timeout_seconds: 60,
            enable_tls: Some(false),
            rest_compression: false,
            grpc_compression: false,
            compression_algorithm: "gzip".to_string(),
            compression_level: 6,
            ttl_sweep_interval_seconds: 900,
            unified_mode: false,
            unified_port: 5678,
            arrow_flight_port: 5680,
            enable_rest: true,
            enable_grpc: true,
            enable_arrow_flight: true,
            http2_max_concurrent_streams: 1000,
            max_connections: 10000,
            pg_port: None,
            enable_pgwire: true,
            internal_mux_port: None,
            mcp_port: None,
            transport: default_transport(),
            socket_dir: None,
        }
    }
}

/// Cluster bootstrap and peer discovery configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Raft node identifier (unique within the cluster).
    pub node_id: Option<u64>,

    /// Addresses of other cluster peers.
    pub cluster_peers: Vec<String>,

    /// Election timeout in milliseconds.
    pub election_timeout_ms: u64,

    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,

    /// Number of log entries before taking a snapshot.
    pub snapshot_threshold: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            node_id: None,
            cluster_peers: Vec::new(),
            election_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            snapshot_threshold: 10000,
        }
    }
}

/// Server identity and network bind configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerConfig {
    /// Unique node identifier within the cluster.
    pub node_id: String,

    /// IP address or hostname to bind the server to.
    pub bind_address: String,

    /// Primary listening port (REST/unified).
    pub port: u16,

    /// Optional gRPC port for convenience; if not set, ApiConfig.grpc_port is used.
    pub grpc_port: Option<u16>,

    /// Root directory for persistent data files.
    pub data_dir: PathBuf,

    /// Read-only embedded admin dashboard (`/admin`). Disabled by default;
    /// opt-in via `[server.admin_ui]`. `#[serde(default)]` so existing configs
    /// without the section keep parsing (and keep the dashboard off).
    #[serde(default)]
    pub admin_ui: AdminUiConfig,

    /// Request tenant resolution mode for protocol edges.
    ///
    /// Defaults to `auto` for mixed-read-safe compatibility: startup derives
    /// the historic behavior from security mode until operators explicitly set
    /// `single_tenant` or `multi_tenant`.
    #[serde(default)]
    pub tenant: ServerTenantConfig,
}

/// Configuration for the read-only embedded admin dashboard served at `/admin`.
///
/// **Disabled by default.** Intended for **standalone** local instances only —
/// keep it off for Kubernetes pods (each pod does not need it) and for
/// embedded / UDS mode (which serves over a local Unix-domain socket, not a
/// browser-reachable TCP port) unless an operator explicitly opts in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AdminUiConfig {
    /// Mount the read-only `/admin` (+ `/dashboard`) route. Default: `false`
    /// (`bool::default()`).
    #[serde(default)]
    pub enabled: bool,

    /// Enable client-side auto-refresh of the dashboard. Default: `false`
    /// (`bool::default()`). When `false`, the auto-refresh toggle in the UI is
    /// rendered **disabled** (greyed out) and the page only refreshes on an
    /// explicit click — an operator must opt in here to allow live polling.
    /// The setting is server-injected into the page; it does not persist per
    /// user session beyond the configured default.
    #[serde(default)]
    pub auto_refresh: bool,

    /// Auto-refresh poll interval in seconds. Default: `30`. Only takes effect
    /// when `auto_refresh = true`. Clamped to a minimum of `5` at serve time so
    /// a misconfigured tiny interval cannot DoS the diagnostic endpoints.
    #[serde(default = "default_admin_refresh_interval")]
    pub refresh_interval_seconds: u32,
}

/// Default admin-dashboard auto-refresh interval (seconds).
fn default_admin_refresh_interval() -> u32 {
    30
}

/// Request-tenant resolution mode encoded in TOML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ServerTenantMode {
    /// Preserve the pre-existing runtime heuristic while operators roll out an
    /// explicit mode. This is compatibility-only; production SaaS deployments
    /// should set `multi_tenant`.
    #[default]
    Auto,
    /// Missing tenant signal resolves to `default_tenant`.
    SingleTenant,
    /// Every request must carry an explicit tenant signal at the edge.
    MultiTenant,
}

/// Server-level tenant configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerTenantConfig {
    /// Deployment mode: `auto`, `single_tenant`, or `multi_tenant`.
    #[serde(default)]
    pub mode: ServerTenantMode,

    /// Default tenant used only in `single_tenant` mode, or when `auto`
    /// resolves to single-tenant.
    #[serde(default = "default_request_tenant")]
    pub default_tenant: String,
}

impl Default for ServerTenantConfig {
    fn default() -> Self {
        Self {
            mode: ServerTenantMode::Auto,
            default_tenant: default_request_tenant(),
        }
    }
}

fn default_request_tenant() -> String {
    "default".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            node_id: "node-1".to_string(),
            bind_address: "127.0.0.1".to_string(),
            port: 5678,
            grpc_port: None,
            data_dir: PathBuf::from("./data"),
            admin_ui: AdminUiConfig::default(),
            tenant: ServerTenantConfig::default(),
        }
    }
}

/// Semantic Knowledge Store feature and storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SksConfig {
    /// Enable SKS features.
    pub enabled: bool,

    /// Enable entity storage.
    pub enable_entities: bool,

    /// Enable graph relationships.
    pub enable_relations: bool,

    /// Enable provenance tracking.
    pub enable_provenance: bool,

    /// Enable temporal versioning.
    pub enable_temporal: bool,

    /// Enable SQL extensions (SIMILAR, FOLLOW, ASSEMBLE).
    pub enable_sql_extensions: bool,

    /// Maximum embedding versions per entity.
    pub max_embedding_versions: usize,

    /// Maximum graph traversal depth.
    pub max_traversal_depth: usize,

    /// Cache size for entity store in MB.
    pub entity_cache_mb: usize,

    /// Cache size for relations in MB.
    pub relations_cache_mb: usize,

    /// Default embedding model for text-to-vector conversion.
    pub default_embedding_model: String,

    /// Storage backend for SKS data ("memory", "sst", "viper").
    pub storage_backend: String,
}

impl Default for SksConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            enable_entities: true,
            enable_relations: true,
            enable_provenance: true,
            enable_temporal: false,
            enable_sql_extensions: true,
            max_embedding_versions: 10,
            max_traversal_depth: 5,
            entity_cache_mb: 256,
            relations_cache_mb: 128,
            default_embedding_model: "openai/text-embedding-3-large".to_string(),
            storage_backend: "sst".to_string(),
        }
    }
}

/// Graph runtime option contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRuntimeConfig {
    /// Enable bounded prefetch hints during traversals.
    pub enable_prefetch: bool,

    /// Per-node/iteration adjacency prefetch budget.
    pub prefetch_budget: usize,

    /// Select graph engine ("ORION").
    pub engine: String,

    /// Embedding storage mode: "none" (default), "cold", "memory".
    #[serde(default = "default_embedding_mode")]
    pub embedding_mode: String,

    /// Vector engine for cold tier embeddings.
    #[serde(default = "default_embedding_engine")]
    pub embedding_engine: String,

    /// Memory cache size in MB for embeddings.
    #[serde(default)]
    pub embedding_memory_cache_mb: Option<usize>,
}

impl Default for GraphRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_prefetch: true,
            prefetch_budget: 8,
            engine: default_graph_engine(),
            embedding_mode: default_embedding_mode(),
            embedding_engine: default_embedding_engine(),
            embedding_memory_cache_mb: None,
        }
    }
}

fn default_graph_engine() -> String {
    "ORION".to_string()
}

fn default_embedding_mode() -> String {
    "none".to_string()
}

fn default_embedding_engine() -> String {
    "sst".to_string()
}

/// Hybrid query runtime option contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRuntimeConfig {
    /// Default seeding strategy ("AVERAGE"|"PER_SEED"|"NONE").
    pub seeding_strategy: String,

    /// Fusion weights for [vector, graph].
    pub fusion_weights: Option<Vec<f64>>,

    /// Selectivity below this value uses filter-first hybrid execution.
    #[serde(default = "default_filter_first_max_selectivity")]
    pub filter_first_max_selectivity: f64,

    /// Selectivity above this value uses vector-first hybrid execution.
    #[serde(default = "default_vector_first_min_selectivity")]
    pub vector_first_min_selectivity: f64,
}

impl Default for HybridRuntimeConfig {
    fn default() -> Self {
        Self {
            seeding_strategy: "AVERAGE".to_string(),
            fusion_weights: Some(vec![0.6, 0.4]),
            filter_first_max_selectivity: default_filter_first_max_selectivity(),
            vector_first_min_selectivity: default_vector_first_min_selectivity(),
        }
    }
}

fn default_filter_first_max_selectivity() -> f64 {
    0.1
}

fn default_vector_first_min_selectivity() -> f64 {
    0.5
}

impl HybridRuntimeConfig {
    /// Validate hybrid runtime policy before startup.
    pub fn validate(&self) -> Result<(), String> {
        if !self.filter_first_max_selectivity.is_finite()
            || !(0.0..=1.0).contains(&self.filter_first_max_selectivity)
        {
            return Err(
                "hybrid.filter_first_max_selectivity must be finite and in [0.0, 1.0]".to_string(),
            );
        }
        if !self.vector_first_min_selectivity.is_finite()
            || !(0.0..=1.0).contains(&self.vector_first_min_selectivity)
        {
            return Err(
                "hybrid.vector_first_min_selectivity must be finite and in [0.0, 1.0]".to_string(),
            );
        }
        if self.filter_first_max_selectivity >= self.vector_first_min_selectivity {
            return Err(
                "hybrid.filter_first_max_selectivity must be less than hybrid.vector_first_min_selectivity"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Hardware acceleration configuration controlling SIMD and GPU features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Enable automatic hardware detection.
    pub enable_detection: bool,

    /// Enable GPU acceleration if detected.
    pub enable_gpu_acceleration: bool,

    /// Enable SIMD acceleration if detected.
    pub enable_simd: bool,

    /// Enable AVX-512 if available.
    pub enable_avx512: bool,

    /// Enable GPU for SQL parsing.
    pub enable_gpu_parsing: bool,

    /// Enable GPU for distance calculations.
    pub enable_gpu_similarity: bool,

    /// Minimum vector size to use GPU.
    pub gpu_min_vector_size: usize,

    /// Minimum batch size to use GPU.
    pub gpu_min_batch_size: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            enable_gpu_acceleration: true,
            enable_simd: true,
            enable_avx512: true,
            enable_gpu_parsing: true,
            enable_gpu_similarity: true,
            gpu_min_vector_size: 64,
            gpu_min_batch_size: 100,
        }
    }
}

/// Filesystem configuration for performance optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemOptimizationConfig {
    /// Enable write strategy caching.
    pub enable_write_strategy_cache: bool,

    /// Temp directory configuration.
    pub temp_strategy: TempStrategy,

    /// Atomic operations configuration.
    pub atomic_config: TransactionalOperationsConfig,
}

/// Temp strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TempStrategy {
    /// Same directory temp (recommended for local filesystem).
    SameDirectory,

    /// Configured temp directory.
    ConfiguredTemp {
        /// Path to the custom temporary directory.
        temp_dir: String,
    },

    /// System temp directory (fallback).
    SystemTemp,
}

/// Atomic operations configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionalOperationsConfig {
    /// Enable atomic writes for local filesystem.
    pub enable_local_atomic: bool,

    /// Enable write-temp-rename for object stores.
    pub enable_object_store_atomic: bool,

    /// Cleanup temp files on startup.
    pub cleanup_temp_on_startup: bool,
}

impl Default for FilesystemOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_write_strategy_cache: true,
            temp_strategy: TempStrategy::SameDirectory,
            atomic_config: TransactionalOperationsConfig::default(),
        }
    }
}

impl Default for TransactionalOperationsConfig {
    fn default() -> Self {
        Self {
            enable_local_atomic: true,
            enable_object_store_atomic: true,
            cleanup_temp_on_startup: true,
        }
    }
}

/// Observability and monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Whether Prometheus metrics collection is enabled.
    pub metrics_enabled: bool,

    /// Default tracing log level (e.g., "info", "debug", "trace").
    pub log_level: String,

    /// Dashboard refresh interval in seconds.
    #[serde(default = "default_dashboard_refresh_interval")]
    pub dashboard_refresh_interval_seconds: u64,
}

fn default_dashboard_refresh_interval() -> u64 {
    60
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            log_level: "info".to_string(),
            dashboard_refresh_interval_seconds: 60,
        }
    }
}

impl MonitoringConfig {
    /// Get dashboard refresh interval, ensuring it is at least 15 seconds.
    pub fn dashboard_refresh_interval(&self) -> u64 {
        self.dashboard_refresh_interval_seconds.max(15)
    }
}

/// Configuration for the metrics system
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable or disable the entire metrics system
    pub enabled: bool,

    /// Number of partitions for collection-based metrics storage
    pub collection_partitions: usize,

    /// Base path for metrics storage (e.g., "s3://bucket/metrics" or "file:///data/metrics")
    pub storage_path: String,

    /// Flush interval for metrics updates in seconds
    pub flush_interval_seconds: u64,

    /// Retention period in days (max: 30, default: 7)
    pub retention_days: u32,

    /// Threshold for parallel scan optimization (number of files)
    pub parallel_scan_threshold: usize,

    /// Sparsity threshold for compression decisions (% of zero/null values)
    pub sparsity_threshold: f32,

    /// Size threshold for quantization recommendations (bytes)
    pub quantization_size_threshold: u64,

    /// Snapshot interval for metrics aggregation in seconds
    pub snapshot_interval_seconds: u64,

    /// Maximum memory usage in MB for metrics cache
    pub max_memory_mb: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_partitions: 16,
            storage_path: "file:///data/proximadb/metrics".to_string(),
            flush_interval_seconds: 30, // 30 seconds
            retention_days: 7,
            parallel_scan_threshold: 10, // Suggest parallel scan if >10 files
            sparsity_threshold: 0.3,     // Consider sparse if >30% zeros
            quantization_size_threshold: 100 * 1024 * 1024, // 100MB
            snapshot_interval_seconds: 60, // 1 minute snapshots
            max_memory_mb: 512,          // 512MB max memory for metrics cache
        }
    }
}

impl MetricsConfig {
    /// Validate and adjust configuration to safe bounds
    pub fn validate(&mut self) -> anyhow::Result<()> {
        // Enforce minimum flush interval
        if self.flush_interval_seconds < 10 {
            tracing::warn!(
                "Flush interval {} too low, setting to minimum 10 seconds",
                self.flush_interval_seconds
            );
            self.flush_interval_seconds = 10;
        }

        // Enforce maximum retention
        if self.retention_days > 30 {
            tracing::warn!(
                "Retention period {} days too high, setting to maximum 30 days",
                self.retention_days
            );
            self.retention_days = 30;
        }

        // Enforce minimum partitions
        if self.collection_partitions < 1 {
            tracing::warn!(
                "Collection partitions {} too low, setting to minimum 1",
                self.collection_partitions
            );
            self.collection_partitions = 1;
        }

        // Enforce maximum partitions
        if self.collection_partitions > 256 {
            tracing::warn!(
                "Collection partitions {} too high, setting to maximum 256",
                self.collection_partitions
            );
            self.collection_partitions = 256;
        }

        Ok(())
    }
}

/// Device class backing a storage location (TD-IOBUDGET-1).
///
/// Selects the ranged-GET coalescing profile a location's I/O budget uses when
/// no explicit `min_bytes`/`target_bytes`/`max_bytes` override is given:
/// `ssd` → the local/NVMe profile (~1 MiB target), `hdd` → the remote-style
/// CLOUD profile (ADR-073: a seek-bound disk must issue few, large, coalesced
/// reads), `cloud` → "coalesce like a remote store for MY scheme" (a `file://`
/// location gets CLOUD; `s3://` keeps the S3 profile). A value outside these
/// three fails config LOAD (serde enum), so a typo never silently runs the
/// wrong profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiskClass {
    /// Local SSD / NVMe: no round-trip cost, small reads are fine.
    #[serde(rename = "ssd")]
    Ssd,
    /// Spinning disk (500-IOPS class): coalesce like a remote store (ADR-073).
    #[serde(rename = "hdd")]
    Hdd,
    /// Object storage (or a disk wanting remote-style coalescing):
    /// scheme-aware cloud default.
    #[serde(rename = "cloud")]
    Cloud,
}

/// Optional per-storage-location I/O budget (TD-IOBUDGET-1).
///
/// Sizes the ranged GETs the engine issues against THIS location — the reader
/// coalescing policies and the adaptive read planner's cap enumeration. (The
/// writer's cell-count derivation is NOT registry-reachable today: its call
/// site passes `None`, destination unknown at clustering time — TD-IVF-4.)
/// Resolution precedence: explicit
/// `min_bytes`/`target_bytes`/`max_bytes` override the `disk_class` profile,
/// which overrides the URL-scheme default. The root crate validates the
/// resolved budget (fail-closed at load) and registers it at boot.
///
/// `deny_unknown_fields` is deliberate: a misspelled key (`target_byets`)
/// must fail startup rather than be silently ignored (the repo-wide serde
/// policy otherwise drops unknown fields).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IoBudgetConfig {
    /// Device class — picks the base profile (`ssd` | `hdd` | `cloud`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_class: Option<DiskClass>,
    /// Smallest useful ranged GET in bytes (profile value when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_bytes: Option<u64>,
    /// Preferred ranged-GET size in bytes (profile value when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_bytes: Option<u64>,
    /// Largest single coalesced range in bytes (profile value when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<u64>,
}

/// Storage location configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageLocation {
    /// Storage URL (e.g., "file:///nvme1/proximadb", "s3://bucket/proximadb").
    pub url: String,

    /// Weight for weighted distribution.
    pub weight: u32,

    /// Tags for filtering (e.g., ["fast", "local"], ["cloud", "archive"]).
    pub tags: Vec<String>,

    /// Optional per-location I/O budget (TD-IOBUDGET-1). Unset → the URL
    /// scheme's default profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub io_budget: Option<IoBudgetConfig>,
}

impl Default for StorageLocation {
    fn default() -> Self {
        Self {
            url: "file://./data".to_string(),
            weight: 1,
            tags: vec!["local".to_string()],
            io_budget: None,
        }
    }
}

/// Assignment configuration for placing collection data across storage locations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentConfig {
    /// Assignment strategy: "hash", "round-robin", "weighted".
    pub strategy: String,

    /// Keep all collection data together (WAL, data, index on same location).
    pub affinity: bool,
}

impl Default for AssignmentConfig {
    fn default() -> Self {
        Self {
            strategy: "hash".to_string(),
            affinity: true,
        }
    }
}

/// Common compaction configuration shared across storage engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// L0 file count threshold for compaction.
    pub l0_file_threshold: usize,

    /// Query-visible file-count threshold for L1 and higher levels.
    ///
    /// This is intentionally lower than the L0 admission threshold: L0
    /// batches writes, while every steady-state higher-level segment adds a
    /// search cascade. Two is the smallest threshold that actually reduces
    /// fanout: a pair of immutable segments is merged once, while the
    /// single-file rewrite guard prevents pointless level promotion.
    #[serde(default = "default_higher_level_file_threshold")]
    pub higher_level_file_threshold: usize,

    /// L0 size threshold in MB for compaction.
    pub l0_size_threshold_mb: usize,

    /// Multiplier for higher-level file-count thresholds.
    ///
    /// Vector search pays a read-amplification cost for every query-visible
    /// segment, so the default is deliberately `1.0`: every level consolidates
    /// at the L0 threshold. A RocksDB-style `2.0` default optimizes write
    /// amplification but lets L1/L2 segment counts grow geometrically.
    pub level_multiplier: f64,

    /// Maximum number of levels.
    pub max_levels: u8,

    /// Compaction strategy: "count", "size", or "hybrid".
    pub strategy: String,

    /// Target output file size in MB for size-based compaction.
    pub target_file_size_mb: usize,

    /// Conservative peak-memory estimate per byte of input segment data.
    ///
    /// The initial `12.0` default includes headroom over the 9.85x incremental
    /// RSS measured by the 3.3M-vector PAX compaction benchmark. This is an
    /// admission estimate, not an allocation limit; operators can tighten it
    /// as the streaming writer reduces measured amplification.
    #[serde(default = "default_compaction_memory_amplification")]
    pub memory_amplification_factor: f64,

    /// Maximum share of process-visible capacity reserved for compactions.
    /// Process-visible capacity is cgroup-constrained in containers.
    #[serde(default = "default_compaction_memory_budget_fraction")]
    pub memory_budget_fraction: f64,

    /// Maximum share of currently available memory that compactions may
    /// reserve. This live-pressure guard is applied in addition to the stable
    /// capacity fraction.
    #[serde(default = "default_compaction_available_memory_fraction")]
    pub available_memory_fraction: f64,

    /// Optional absolute ceiling for all in-flight compaction reservations.
    /// Zero means automatic sizing from capacity and live availability.
    #[serde(default)]
    pub max_memory_mb: u64,

    /// Enable deterministic local-disk spill when an otherwise eligible
    /// compaction does not fit the in-memory projection. Default-off keeps the
    /// existing execution path behavior-neutral until spill has baked.
    #[serde(default)]
    pub spill_enabled: bool,

    /// Local scratch directory used only for recoverable compaction
    /// intermediates. This must be a filesystem path, not an object-store URL.
    #[serde(default)]
    pub spill_directory: Option<String>,

    /// Fixed RAM reservation for one spill compaction worker.
    #[serde(default = "default_compaction_spill_working_memory_mb")]
    pub spill_working_memory_mb: u64,

    /// Conservative scratch-space reservation per byte of input segment data.
    /// The 10x default includes 33% headroom over the 7.53x logical scratch
    /// peak measured by the release/Azurite 3.3M PAX spill benchmark. Scratch
    /// expands compressed input into external record and f32/cell-order runs
    /// while the disk-backed PAX writer builds its output components.
    #[serde(default = "default_compaction_spill_scratch_amplification")]
    pub spill_scratch_amplification_factor: f64,

    /// Maximum share of currently available scratch space that all spill
    /// compactions may reserve.
    #[serde(default = "default_compaction_spill_available_disk_fraction")]
    pub spill_available_disk_fraction: f64,

    /// Optional absolute ceiling for all in-flight scratch reservations.
    /// Zero means automatic sizing from live disk availability.
    #[serde(default)]
    pub spill_max_disk_mb: u64,

    /// Target live segment count for an operator FULL compaction
    /// (`COMPACT TABLE ... FULL` / `POST .../compact {"full":true}`). 1
    /// collapses the collection to a single segment, crossing levels the
    /// background policy never crosses.
    #[serde(default = "default_full_target_segments")]
    pub full_target_segments: usize,
    /// Bound on pairwise merge passes per FULL request (n segments need
    /// n-1 passes).
    #[serde(default = "default_full_max_passes")]
    pub full_max_passes: usize,
    /// Wall-clock budget for one FULL request, including inter-pass
    /// quiescence waits.
    #[serde(default = "default_full_timeout_ms")]
    pub full_timeout_ms: u64,
    /// TD-COMPACT-15: the target file size (MB) for the consolidate
    /// strategy — segments merge smallest-first until each output
    /// reaches this size, producing uniform I/O units.
    ///
    /// `Option` so that "the operator stated this" is distinguishable from
    /// "absent": the SST-section merge needs to know, and comparing against
    /// the default value cannot tell them apart (an operator explicitly
    /// setting 512 to override a top-level 4096 looked absent). Read it
    /// through [`CompactionConfig::consolidate_target_file_size_mb_or_default`].
    #[serde(default)]
    pub consolidate_target_file_size_mb: Option<u64>,
    /// TD-COMPACT-15: the write-amplification limit above which the
    /// consolidate strategy used to degrade to hybrid. Read by NOTHING while
    /// background consolidate is refused (TD-COMPACT-16); kept only so a
    /// config naming it still parses and gets the deprecation note at
    /// startup.
    #[serde(default)]
    pub consolidate_write_amp_limit: Option<u64>,
    /// TD-COMPACT-15: the server-level DEFAULT compaction strategy. Only
    /// `"hybrid"` (the code default) is accepted today: BACKGROUND
    /// consolidate is refused until TD-COMPACT-16, because its trigger
    /// scheduled an ordinary L0 morsel rather than the consolidate merge and
    /// could re-fire every flush without converging. Collapse a collection
    /// with `COMPACT TABLE ... FULL` instead.
    #[serde(default)]
    pub consolidate_strategy: Option<String>,
}

impl CompactionConfig {
    /// Validate strategy fields fail-closed: an unknown value must abort
    /// startup with an actionable message, never silently fall back (the
    /// dead-config failure mode TD-COMPACT-9 documented). The legacy
    /// `strategy` field drives the count/size level heuristic; it is NOT
    /// the consolidate/hybrid switch — reject the common mistake of
    /// setting `strategy = "consolidate"` there.
    pub fn validate(&self, section: &str) -> Result<(), String> {
        // Validate EXACTLY what the consumer honors. `compaction_utils`
        // matches `config.strategy.as_str()` raw and case-sensitively, so
        // trimming/lowercasing here let "Size" or " size" pass boot and then
        // fall through to the `_` arm as hybrid — the dead-config hazard
        // this validation exists to kill, one layer down.
        if !matches!(self.strategy.as_str(), "count" | "size" | "hybrid") {
            return Err(format!(
                "{section}.strategy must be exactly 'count', 'size' or 'hybrid' (lowercase, no surrounding whitespace — the level heuristic matches case-sensitively; got '{}'); to collapse a collection's segments use COMPACT TABLE ... FULL (background consolidate is refused, TD-COMPACT-16)",
                self.strategy
            ));
        }
        // Background consolidate is refused (TD-COMPACT-16). An explicit
        // REQUEST for it fails boot — silently not consolidating would leave
        // the operator believing it runs. The knobs that are merely dead now
        // ("hybrid" is already the default; the write-amp limit was read only
        // by the removed trigger) shipped as accepted settings in #2407, so
        // failing boot on them would turn an upgrade into an outage: they are
        // warned about instead (#2418 review round 13).
        // full_target_segments = 0 is the same knob-silently-rewritten
        // hazard the four Some(0) doors (REST, Flight, trait seam, engine)
        // reject: `.max(1)` would collapse every bare FULL to 1 while the
        // operator believes they stated 0 (#2418 review round 20).
        // Same knob-silently-rewritten class as `full_target_segments`
        // below: `compact_collection_full` applies `.max(1)` to this, so 0
        // boots and behaves as 1 — a FULL on a 10-segment collection does one
        // round and reports `stop_reason: "max_passes"`, `reached_target:
        // false`, while the operator believes they removed the bound
        // (#2418 review round 22).
        if self.full_max_passes == 0 {
            return Err(format!(
                "{section}.full_max_passes must be >= 1 (got 0); it is clamped to 1 at use,                  so 0 would silently mean 'one pass', not 'unbounded'"
            ));
        }
        if self.full_target_segments == 0 {
            return Err(format!(
                "{section}.full_target_segments must be >= 1 (got 0)"
            ));
        }
        if let Some(strategy) = &self.consolidate_strategy {
            let value = strategy.trim().to_ascii_lowercase();
            if value == "consolidate" {
                return Err(format!(
                    "{section}.consolidate_strategy = \"consolidate\" is not supported: background consolidate does not converge (TD-COMPACT-16). Collapse with COMPACT TABLE ... FULL instead"
                ));
            }
            if value != "hybrid" {
                return Err(format!(
                    "{section}.consolidate_strategy must be 'hybrid' (got '{strategy}')"
                ));
            }
            // "hybrid": accepted; reported by `deprecation_notes` once logging
            // is up (validate runs during config load, BEFORE the tracing
            // subscriber exists, so a warn! here was silently dropped —
            // #2418 review round 14).
        }
        // A stated 0 MiB target has no meaning (it was silently clamped to
        // 1 MiB by the FULL merge's grouping). Fail closed at boot.
        if self.consolidate_target_file_size_mb == Some(0) {
            return Err(format!(
                "{section}.consolidate_target_file_size_mb must be positive (got 0)"
            ));
        }
        Ok(())
    }
}

impl CompactionConfig {
    /// Deprecated-but-accepted settings to report once logging is up. Kept
    /// out of `validate`, which runs during config load before any tracing
    /// subscriber exists (a warning there never reached an operator). The
    /// caller (`ProximaDB::start`) logs each note at WARN.
    pub fn deprecation_notes(&self, section: &str) -> Vec<String> {
        let mut notes = Vec::new();
        if self.consolidate_strategy.is_some() {
            notes.push(format!(
                "{section}.consolidate_strategy is deprecated and has no effect (hybrid is the default; background consolidate is refused, TD-COMPACT-16); remove it"
            ));
        }
        if self.consolidate_write_amp_limit.is_some() {
            notes.push(format!(
                "{section}.consolidate_write_amp_limit is deprecated and has no effect (background consolidate is refused, TD-COMPACT-16); remove it"
            ));
        }
        notes
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            l0_file_threshold: 5,
            higher_level_file_threshold: default_higher_level_file_threshold(),
            l0_size_threshold_mb: 256,
            level_multiplier: 1.0,
            max_levels: 7,
            strategy: "hybrid".to_string(),
            target_file_size_mb: 128,
            memory_amplification_factor: default_compaction_memory_amplification(),
            memory_budget_fraction: default_compaction_memory_budget_fraction(),
            available_memory_fraction: default_compaction_available_memory_fraction(),
            max_memory_mb: 0,
            spill_enabled: false,
            spill_directory: None,
            spill_working_memory_mb: default_compaction_spill_working_memory_mb(),
            spill_scratch_amplification_factor: default_compaction_spill_scratch_amplification(),
            spill_available_disk_fraction: default_compaction_spill_available_disk_fraction(),
            spill_max_disk_mb: 0,
            full_target_segments: default_full_target_segments(),
            full_max_passes: default_full_max_passes(),
            full_timeout_ms: default_full_timeout_ms(),
            consolidate_target_file_size_mb: None,
            consolidate_write_amp_limit: None,
            consolidate_strategy: None,
        }
    }
}

fn default_full_target_segments() -> usize {
    1
}

fn default_full_max_passes() -> usize {
    16
}

fn default_full_timeout_ms() -> u64 {
    3_600_000
}

/// The consolidate target file size when the operator stated none (MiB).
pub const DEFAULT_CONSOLIDATE_TARGET_FILE_SIZE_MB: u64 = 512;

impl CompactionConfig {
    /// The consolidate target file size, defaulted. Read by the FULL merge's
    /// smallest-first grouping.
    pub fn consolidate_target_file_size_mb_or_default(&self) -> u64 {
        self.consolidate_target_file_size_mb
            .unwrap_or(DEFAULT_CONSOLIDATE_TARGET_FILE_SIZE_MB)
    }

    /// Merge an SST-section override over the top-level config, per FIELD —
    /// for EVERY field, not just the `consolidate_*` ones.
    ///
    /// Used by `effective_sst_config`. A wholesale pick is wrong because an
    /// operator who declares `[storage.sst_config.compaction_config]` for one
    /// knob (say `l0_file_threshold`) would silently lose every top-level
    /// value — including `full_target_segments`, which the FULL collapse and
    /// its follow-up floor both read (TD-COMPACT-15 F1, widened by the #2418
    /// review round 22: closing this for 3 of 23 fields left the other 20,
    /// among them the one this PR made load-bearing).
    ///
    /// Two presence rules, because the struct has two kinds of field:
    ///
    /// * The `consolidate_*` fields are `Option`, so `is_none()` is a TRUE
    ///   presence signal — genuinely unstated, not "happens to equal the
    ///   default".
    /// * Every other field is concrete with a serde default, so presence
    ///   CANNOT be observed after deserialization. There, a section value
    ///   equal to the code default is treated as unstated and inherits the
    ///   top-level value.
    ///
    /// The residual edge of that second rule, stated plainly rather than
    /// left to be discovered: an operator who writes a section value that
    /// EQUALS the code default while the top level says something else gets
    /// the top-level value, not their explicit restatement of the default.
    /// Both readings lose that intent today (the wholesale pick discarded
    /// the top-level value outright), and the common case — a section
    /// stating one knob — is now correct for all 23 fields. Observing
    /// presence properly needs an all-`Option` overlay type for the section;
    /// filed as TD-COMPACT-15 F10, not faked here.
    #[allow(clippy::float_cmp)] // comparing against the code default is exact by construction
    pub fn merge_section_over(top: &CompactionConfig, section: Option<&CompactionConfig>) -> Self {
        let Some(section) = section else {
            return top.clone();
        };
        let defaults = CompactionConfig::default();
        let mut merged = section.clone();

        if merged.consolidate_strategy.is_none() {
            merged.consolidate_strategy = top.consolidate_strategy.clone();
        }
        if merged.consolidate_target_file_size_mb.is_none() {
            merged.consolidate_target_file_size_mb = top.consolidate_target_file_size_mb;
        }
        if merged.consolidate_write_amp_limit.is_none() {
            merged.consolidate_write_amp_limit = top.consolidate_write_amp_limit;
        }

        macro_rules! inherit_if_default {
            ($($field:ident),+ $(,)?) => {
                $(
                    if merged.$field == defaults.$field {
                        merged.$field = top.$field.clone();
                    }
                )+
            };
        }
        // Every concrete field. `merge_section_over_covers_every_field` is a
        // destructuring guard that fails to COMPILE if a field is added
        // without being listed here.
        inherit_if_default!(
            l0_file_threshold,
            higher_level_file_threshold,
            l0_size_threshold_mb,
            level_multiplier,
            max_levels,
            strategy,
            target_file_size_mb,
            memory_amplification_factor,
            memory_budget_fraction,
            available_memory_fraction,
            max_memory_mb,
            spill_enabled,
            spill_directory,
            spill_working_memory_mb,
            spill_scratch_amplification_factor,
            spill_available_disk_fraction,
            spill_max_disk_mb,
            full_target_segments,
            full_max_passes,
            full_timeout_ms,
        );
        merged
    }
}

fn default_higher_level_file_threshold() -> usize {
    2
}

fn default_compaction_memory_amplification() -> f64 {
    12.0
}

fn default_compaction_memory_budget_fraction() -> f64 {
    0.25
}

fn default_compaction_available_memory_fraction() -> f64 {
    0.5
}

fn default_compaction_spill_working_memory_mb() -> u64 {
    512
}

fn default_compaction_spill_scratch_amplification() -> f64 {
    10.0
}

fn default_compaction_spill_available_disk_fraction() -> f64 {
    0.5
}

/// Performance optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable memory-mapped I/O for large files.
    #[serde(default = "default_enable_mmap")]
    pub enable_mmap: bool,

    /// Enable zone map pruning to skip irrelevant blocks.
    #[serde(default = "default_enable_zone_map_pruning")]
    pub enable_zone_map_pruning: bool,

    /// Enable AXIS indexes for approximate nearest neighbor search.
    ///
    /// Runtime master switch; the Cargo `axis` feature must also be compiled
    /// and the collection must explicitly declare `index_configs`. Default is
    /// `false` per ADR-070: the co-design PAX scan + survivor cache is the
    /// default ANN path. The lightweight manager value may still be
    /// constructed to keep the compile-time type graph stable, but it is not
    /// registered with serving and no AXIS maintenance loops start while this
    /// flag is false.
    #[serde(default = "default_enable_axis_indexes")]
    pub enable_axis_indexes: bool,

    /// Default index type for new collections: flat, hnsw, ivf, lsh.
    #[serde(default = "default_index_type")]
    pub default_index_type: String,

    /// Enable progressive quantization search (Binary -> INT8 -> FP32).
    #[serde(default = "default_enable_progressive_search")]
    pub enable_progressive_search: bool,

    /// Enable block-level bloom filters for metadata filtering.
    #[serde(default = "default_enable_bloom_filters")]
    pub enable_bloom_filters: bool,
}

fn default_enable_mmap() -> bool {
    true
}

fn default_enable_zone_map_pruning() -> bool {
    true
}

fn default_enable_axis_indexes() -> bool {
    // ADR-070: co-design (PAX scan + survivor cache) is the default ANN path.
    false
}

fn default_index_type() -> String {
    "hnsw".to_string()
}

fn default_enable_progressive_search() -> bool {
    true
}

fn default_enable_bloom_filters() -> bool {
    true
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_mmap: default_enable_mmap(),
            enable_zone_map_pruning: default_enable_zone_map_pruning(),
            enable_axis_indexes: default_enable_axis_indexes(),
            default_index_type: default_index_type(),
            enable_progressive_search: default_enable_progressive_search(),
            enable_bloom_filters: default_enable_bloom_filters(),
        }
    }
}

/// Configuration for search pruning, allowing for simple or advanced setup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PruneModeConfig {
    /// Simple pruning mode specified by a single strategy name.
    Simple(String),

    /// Advanced pruning mode with fine-grained control.
    Advanced(AdvancedPruneConfig),
}

/// Advanced configuration for search pruning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdvancedPruneConfig {
    /// Pruning algorithm type (e.g., "sqrt", "log").
    #[serde(default = "default_prune_type")]
    pub r#type: String,

    /// Minimum number of candidates to keep after pruning.
    pub min_keep: Option<usize>,

    /// Maximum number of candidates to keep after pruning.
    pub max_keep: Option<usize>,

    /// Pruning ratio controlling aggressiveness (0.0 to 1.0).
    pub ratio: Option<f32>,
}

fn default_prune_type() -> String {
    "sqrt".to_string()
}

impl Default for AdvancedPruneConfig {
    fn default() -> Self {
        Self {
            r#type: default_prune_type(),
            min_keep: None,
            max_keep: None,
            ratio: None,
        }
    }
}

// NOTE: `WalStorageConfig` + `WalDistributionStrategy` were RETIRED
// (TD-CONFIG-CONSOLIDATE-1, Core Directive #19). They were a stranded
// decomposition duplicate of the LIVE `WriteBufferUserConfig` (src/core/config.rs)
// — extracted into this foundation crate but never wired into the assembled
// `CoreStorageConfig`, and consumed only by a test-only `From<&WalStorageConfig>
// for WALConfig`. Keeping the dead twin caused drift (fields added to each in
// parallel). The canonical WAL config is `WriteBufferUserConfig`; do NOT
// re-add a foundation twin.

// ---------------------------------------------------------------------------
// Embedding-precision rollout (PR 3 of EMBEDDING_PRECISION_LLD_2026_05_22)
// ---------------------------------------------------------------------------

/// Feature-flag control for the embedding-precision rollout.
///
/// During Phase 2 deploy, operators flip `schema_v2_enabled` to `true` only
/// after every node in the cluster runs a binary that supports both v1 and v2
/// reads (PR 2 plumbing). The server `/version` endpoint reports
/// `precision_schema_v2_capable: true` so operators can verify before
/// flipping. PR 4 wires this flag into the WAL segment-header writer; PR 3
/// makes the flag the gate that rejects non-Fp32 records while the cluster
/// is still on the v1 wire shape.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingPrecisionConfig {
    /// When `true`, the WAL writer is allowed to emit schema-v2 records
    /// (precision-aware `EmbeddingValues`). When `false` (default), the
    /// writer stays on schema-v1 (Vec<f32> only) and the validation guard
    /// rejects records whose `EmbeddingCell.precision` is non-Fp32 with
    /// `unsupported_precision_schema_v1_only`.
    pub schema_v2_enabled: bool,
}

impl EmbeddingPrecisionConfig {
    /// Environment variable that overrides the configured flag at startup.
    pub const ENV_VAR: &'static str = "PROXIMADB_EMBED_PRECISION_SCHEMA_V2";

    /// Apply the env-var override on top of the config-file value.
    ///
    /// Parsing matches the convention of other ProximaDB boolean flags:
    /// `true|1|yes|on` (case-insensitive) → on; `false|0|no|off` → off.
    /// Any other value returns an error so deploys fail loudly instead of
    /// silently picking the default.
    pub fn with_env_override(mut self) -> anyhow::Result<Self> {
        match std::env::var(Self::ENV_VAR) {
            Ok(raw) => {
                self.schema_v2_enabled = parse_bool_flag(&raw, Self::ENV_VAR)?;
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("{} env var contains non-UTF-8 bytes", Self::ENV_VAR);
            }
        }
        Ok(self)
    }

    /// Process-wide cached config: env var is read at first call and the
    /// resolved value is memoized via `OnceLock`. Every subsequent caller
    /// gets the same `&'static` reference without re-parsing.
    ///
    /// Multiple call sites (the PR 3b ingest validator, the INT-2b WAL
    /// writer, future PAX writers in INT-3) need to consult this flag on
    /// the hot path. Centralizing the cache here means every caller sees
    /// the SAME value — operators can't end up with one subsystem on v2
    /// and another on v1 due to a race on env-var read.
    ///
    /// Parse failures (malformed env var) degrade to the safe default
    /// (`schema_v2_enabled = false`) + a `warn!` log so a typo doesn't
    /// take down the ingest path. The warning fires exactly once per
    /// process even if there are millions of calls.
    pub fn cached() -> &'static Self {
        static CACHED: std::sync::OnceLock<EmbeddingPrecisionConfig> = std::sync::OnceLock::new();
        CACHED.get_or_init(|| {
            EmbeddingPrecisionConfig::default()
                .with_env_override()
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        env = EmbeddingPrecisionConfig::ENV_VAR,
                        error = %e,
                        "failed to parse precision env var; defaulting to schema_v2_enabled=false"
                    );
                    EmbeddingPrecisionConfig::default()
                })
        })
    }
}

/// Parse a boolean feature-flag value from an env-var-style string.
fn parse_bool_flag(raw: &str, env_name: &str) -> anyhow::Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => anyhow::bail!("{env_name} must be true|false|1|0|yes|no|on|off, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn env_gate_truthy_set_and_default() {
        use std::sync::Mutex;
        // Serialize env mutations across tests in one process.
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: single-threaded test process section held under ENV_LOCK;
        // edition 2024 marks env mutation unsafe for multi-threaded UB.
        unsafe { std::env::set_var("ENVGATE_TEST_GATE", "on") };
        env_gate! {
            /// test gate
            fn envgate_test_on() { gate: "ENVGATE_TEST_GATE", default: false }
        }
        assert!(
            envgate_test_on(),
            "'on' must count as true per the registry contract"
        );

        unsafe { std::env::set_var("ENVGATE_TEST_GATE2", "bogus") };
        env_gate! {
            fn envgate_test_unrecognized() { gate: "ENVGATE_TEST_GATE2", default: true }
        }
        assert!(
            !envgate_test_unrecognized(),
            "an unrecognized value must evaluate false (matching every hand-written body), NOT the default"
        );

        unsafe { std::env::remove_var("ENVGATE_TEST_GATE3") };
        env_gate! {
            fn envgate_test_unset_default() { gate: "ENVGATE_TEST_GATE3", default: true }
        }
        assert!(
            envgate_test_unset_default(),
            "an UNSET gate must fall back to the declared default"
        );
    }
    use super::*;

    /// Exhaustiveness guard for `merge_section_over` (#2418 review round 22).
    ///
    /// This destructures `CompactionConfig` with no `..` rest pattern, so
    /// ADDING A FIELD BREAKS THIS TEST'S COMPILATION. That is the point: the
    /// new field must be added to `inherit_if_default!` (or given `Option`
    /// presence) in the same change. Without the guard, a field added later
    /// is silently discarded whenever an operator states any one knob in
    /// `[storage.sst_config.compaction_config]` — the exact class this
    /// review round reopened after it was closed for only 3 of 23 fields.
    #[test]
    fn merge_section_over_covers_every_field() {
        let CompactionConfig {
            // concrete fields — each must appear in `inherit_if_default!`
            l0_file_threshold: _,
            higher_level_file_threshold: _,
            l0_size_threshold_mb: _,
            level_multiplier: _,
            max_levels: _,
            strategy: _,
            target_file_size_mb: _,
            memory_amplification_factor: _,
            memory_budget_fraction: _,
            available_memory_fraction: _,
            max_memory_mb: _,
            spill_enabled: _,
            spill_directory: _,
            spill_working_memory_mb: _,
            spill_scratch_amplification_factor: _,
            spill_available_disk_fraction: _,
            spill_max_disk_mb: _,
            full_target_segments: _,
            full_max_passes: _,
            full_timeout_ms: _,
            // Option fields — carried by the `is_none()` arms
            consolidate_target_file_size_mb: _,
            consolidate_write_amp_limit: _,
            consolidate_strategy: _,
        } = CompactionConfig::default();
    }

    /// A section stating ONE knob must not discard the operator's other
    /// top-level values — above all `full_target_segments`, which this PR
    /// made load-bearing for both the FULL collapse and its follow-up floor.
    #[test]
    fn section_stating_one_knob_keeps_every_other_top_level_value() {
        let top = CompactionConfig {
            full_target_segments: 3,
            full_max_passes: 9,
            full_timeout_ms: 123_456,
            spill_enabled: !CompactionConfig::default().spill_enabled,
            max_memory_mb: 4242,
            strategy: "size".to_string(),
            consolidate_target_file_size_mb: Some(77),
            ..CompactionConfig::default()
        };
        // The operator declares the section for ONE unrelated knob.
        let section = CompactionConfig {
            l0_file_threshold: 8,
            ..CompactionConfig::default()
        };

        let merged = CompactionConfig::merge_section_over(&top, Some(&section));

        assert_eq!(merged.l0_file_threshold, 8, "the stated knob must win");
        // Everything else the operator set at the top level must survive.
        assert_eq!(
            merged.full_target_segments, 3,
            "full_target_segments was silently reset to the serde default before round 22"
        );
        assert_eq!(merged.full_max_passes, 9);
        assert_eq!(merged.full_timeout_ms, 123_456);
        assert_eq!(merged.spill_enabled, top.spill_enabled);
        assert_eq!(merged.max_memory_mb, 4242);
        assert_eq!(merged.strategy, "size");
        assert_eq!(merged.consolidate_target_file_size_mb, Some(77));
    }

    #[test]
    fn config_level_full_max_passes_zero_fails_boot() {
        // Round 23: `full_max_passes` was clamped to 1 at use, so 0 booted
        // happily and silently meant "one pass", not "unbounded" — the same
        // knob-silently-rewritten class as full_target_segments above.
        let config = CompactionConfig {
            full_max_passes: 0,
            ..Default::default()
        };
        let error = config
            .validate("storage.sst_config.compaction_config")
            .expect_err("full_max_passes = 0 must fail boot");
        assert!(
            error.contains("full_max_passes"),
            "error must name the knob: {error}"
        );
    }

    /// #2418 review round 20: a config-level `full_target_segments = 0` was
    /// silently clamped to 1 by `.max(1)` — the knob-silently-rewritten
    /// hazard the four per-request `Some(0)` doors (REST, Flight, the trait
    /// seam, the engine) all refuse. It must fail boot like they do.
    #[test]
    fn config_level_full_target_segments_zero_fails_boot() {
        let config = CompactionConfig {
            full_target_segments: 0,
            ..Default::default()
        };
        let error = config
            .validate("storage.compaction_config")
            .expect_err("a zero config target must not be silently clamped to 1");
        assert!(error.contains("full_target_segments"), "{error}");
        assert!(error.contains(">= 1"), "{error}");
        // The shipped default (and any positive value) still validates.
        assert!(
            CompactionConfig::default()
                .validate("storage.compaction_config")
                .is_ok()
        );
        assert!(
            CompactionConfig {
                full_target_segments: 3,
                ..Default::default()
            }
            .validate("storage.compaction_config")
            .is_ok()
        );
    }

    #[test]
    fn compaction_strategy_validation_fails_closed() {
        // Valid legacy values pass, including the default.
        // Only the exact forms the consumer matches — "Hybrid" must FAIL.
        for strategy in ["count", "size", "hybrid"] {
            let config = CompactionConfig {
                strategy: strategy.to_string(),
                ..Default::default()
            };
            assert!(
                config.validate("storage.compaction_config").is_ok(),
                "expected '{strategy}' valid"
            );
        }
        // The common dead-config mistake: 'consolidate' in the legacy field.
        let config = CompactionConfig {
            strategy: "consolidate".to_string(),
            ..Default::default()
        };
        let error = config.validate("storage.compaction_config").unwrap_err();
        // It must point at the path that WORKS: consolidate_strategy is
        // refused too (TD-COMPACT-16), so sending operators there would be a
        // second dead end.
        assert!(
            error.contains("COMPACT TABLE"),
            "error should point at the working path: {error}"
        );
        // Typos AND case/whitespace variants fail — the consumer would
        // silently treat each as the `_` (hybrid) arm.
        for bad in ["hybrd", "Hybrid", "SIZE", " size", "size "] {
            assert!(
                CompactionConfig {
                    strategy: bad.to_string(),
                    ..Default::default()
                }
                .validate("storage.compaction_config")
                .is_err(),
                "expected '{bad}' rejected"
            );
        }
    }

    #[test]
    fn consolidate_strategy_field_validated_and_default_none() {
        assert!(CompactionConfig::default().consolidate_strategy.is_none());
        let validate = |value: &str| {
            CompactionConfig {
                consolidate_strategy: Some(value.to_string()),
                ..Default::default()
            }
            .validate("storage.compaction_config")
        };
        // A dead knob shipped by #2407 ("hybrid" is already the default) is
        // accepted, so an upgrade still boots, and reported as deprecated.
        assert!(validate("hybrid").is_ok());
        assert_eq!(
            CompactionConfig {
                consolidate_strategy: Some("hybrid".to_string()),
                consolidate_write_amp_limit: Some(50),
                ..Default::default()
            }
            .deprecation_notes("storage.compaction_config")
            .len(),
            2
        );
        assert!(
            CompactionConfig::default()
                .deprecation_notes("s")
                .is_empty()
        );
        assert!(validate(" Hybrid ").is_ok());
        // An explicit request for the refused behaviour fails, pointing at
        // the operator path that converges.
        for value in ["consolidate", "Consolidate"] {
            let error = validate(value).expect_err("background consolidate must fail boot");
            assert!(error.contains("COMPACT TABLE"), "{error}");
        }
        // The write-amp limit is read only by that refused trigger; an
        // existing config naming it must still boot (deprecation warning).
        assert!(
            CompactionConfig {
                consolidate_write_amp_limit: Some(50),
                ..Default::default()
            }
            .validate("storage.compaction_config")
            .is_ok()
        );
        // Typos are still refused.
        assert!(validate("hybird").is_err());
        assert!(
            CompactionConfig {
                consolidate_strategy: Some("aggressive".to_string()),
                ..Default::default()
            }
            .validate("storage.compaction_config")
            .is_err()
        );
    }

    #[test]
    fn hardware_defaults_match_root_runtime_expectations() {
        let config = HardwareConfig::default();

        assert!(config.enable_detection);
        assert!(config.enable_gpu_acceleration);
        assert!(config.enable_simd);
        assert!(config.enable_avx512);
        assert!(config.enable_gpu_parsing);
        assert!(config.enable_gpu_similarity);
        assert_eq!(config.gpu_min_vector_size, 64);
        assert_eq!(config.gpu_min_batch_size, 100);
    }

    #[test]
    fn filesystem_optimization_defaults_match_root_runtime_expectations() {
        let config = FilesystemOptimizationConfig::default();

        assert!(config.enable_write_strategy_cache);
        assert!(matches!(config.temp_strategy, TempStrategy::SameDirectory));
        assert!(config.atomic_config.enable_local_atomic);
        assert!(config.atomic_config.enable_object_store_atomic);
        assert!(config.atomic_config.cleanup_temp_on_startup);
    }

    #[test]
    fn tls_defaults_match_root_runtime_expectations() {
        let config = TlsConfig::default();

        assert!(!config.enabled);
        assert!(config.cert_file.is_none());
        assert!(config.key_file.is_none());
        assert!(config.bind_interface.is_none());
    }

    #[test]
    fn api_defaults_match_root_runtime_expectations() {
        let config = ApiConfig::default();

        assert_eq!(config.grpc_port, 5679);
        assert_eq!(config.rest_port, 5678);
        assert_eq!(config.max_request_size_mb, 100);
        assert_eq!(config.timeout_seconds, 60);
        assert_eq!(config.enable_tls, Some(false));
        assert_eq!(config.ttl_sweep_interval_seconds, 900);
        assert!(!config.rest_compression);
        assert!(!config.grpc_compression);
        assert_eq!(config.compression_algorithm, "gzip");
        assert_eq!(config.compression_level, 6);
        assert!(!config.unified_mode);
        assert_eq!(config.unified_port, 5678);
        assert_eq!(config.arrow_flight_port, 5680);
        assert!(config.enable_rest);
        assert!(config.enable_grpc);
        assert!(config.enable_arrow_flight);
        assert_eq!(config.http2_max_concurrent_streams, 1000);
        assert_eq!(config.max_connections, 10000);
        assert_eq!(config.pg_port, None);
        assert_eq!(config.internal_mux_port, None);
    }

    /// TD-NET-1 S1: `[api] internal_mux_port` deserializes when present and
    /// defaults to `None` when absent (same optional-port pattern as
    /// `pg_port`).
    #[test]
    fn api_internal_mux_port_parses_and_defaults_none() {
        let mut value = serde_json::to_value(ApiConfig::default())
            .unwrap_or_else(|e| panic!("default api config must serialize: {e}"));
        let obj = value
            .as_object_mut()
            .unwrap_or_else(|| panic!("api config must serialize to a JSON object"));

        obj.insert("internal_mux_port".to_string(), serde_json::json!(25679));
        let parsed: ApiConfig = serde_json::from_value(value.clone())
            .unwrap_or_else(|e| panic!("api config with internal_mux_port must parse: {e}"));
        assert_eq!(parsed.internal_mux_port, Some(25679));

        let obj = value
            .as_object_mut()
            .unwrap_or_else(|| panic!("api config must serialize to a JSON object"));
        obj.remove("internal_mux_port");
        let absent: ApiConfig = serde_json::from_value(value)
            .unwrap_or_else(|e| panic!("api config without internal_mux_port must parse: {e}"));
        assert_eq!(absent.internal_mux_port, None);
    }

    #[test]
    fn consensus_defaults_match_root_runtime_expectations() {
        let config = ConsensusConfig::default();

        assert!(config.node_id.is_none());
        assert!(config.cluster_peers.is_empty());
        assert_eq!(config.election_timeout_ms, 5000);
        assert_eq!(config.heartbeat_interval_ms, 1000);
        assert_eq!(config.snapshot_threshold, 10000);
    }

    #[test]
    fn server_defaults_match_root_runtime_expectations() {
        let config = ServerConfig::default();

        assert_eq!(config.node_id, "node-1");
        assert_eq!(config.bind_address, "127.0.0.1");
        assert_eq!(config.port, 5678);
        assert_eq!(config.grpc_port, None);
        assert_eq!(config.data_dir, PathBuf::from("./data"));
    }

    #[test]
    fn sks_defaults_match_root_runtime_expectations() {
        let config = SksConfig::default();

        assert!(!config.enabled);
        assert!(config.enable_entities);
        assert!(config.enable_relations);
        assert!(config.enable_provenance);
        assert!(!config.enable_temporal);
        assert!(config.enable_sql_extensions);
        assert_eq!(config.max_embedding_versions, 10);
        assert_eq!(config.max_traversal_depth, 5);
        assert_eq!(config.entity_cache_mb, 256);
        assert_eq!(config.relations_cache_mb, 128);
        assert_eq!(
            config.default_embedding_model,
            "openai/text-embedding-3-large"
        );
        assert_eq!(config.storage_backend, "sst");
    }

    #[test]
    fn graph_runtime_defaults_match_root_runtime_expectations() {
        let config = GraphRuntimeConfig::default();

        assert!(config.enable_prefetch);
        assert_eq!(config.prefetch_budget, 8);
        assert_eq!(config.engine, "ORION");
        assert_eq!(config.embedding_mode, "none");
        assert_eq!(config.embedding_engine, "sst");
        assert_eq!(config.embedding_memory_cache_mb, None);
    }

    #[test]
    fn hybrid_runtime_defaults_match_root_runtime_expectations() {
        let config = HybridRuntimeConfig::default();

        assert_eq!(config.seeding_strategy, "AVERAGE");
        assert_eq!(config.fusion_weights, Some(vec![0.6, 0.4]));
        assert_eq!(config.filter_first_max_selectivity, 0.1);
        assert_eq!(config.vector_first_min_selectivity, 0.5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn monitoring_defaults_match_root_runtime_expectations() {
        let config = MonitoringConfig::default();

        assert!(config.metrics_enabled);
        assert_eq!(config.log_level, "info");
        assert_eq!(config.dashboard_refresh_interval_seconds, 60);
        assert_eq!(config.dashboard_refresh_interval(), 60);
    }

    #[test]
    fn monitoring_dashboard_refresh_interval_has_minimum() {
        let config = MonitoringConfig {
            dashboard_refresh_interval_seconds: 1,
            ..MonitoringConfig::default()
        };

        assert_eq!(config.dashboard_refresh_interval(), 15);
    }

    #[test]
    fn storage_location_defaults_match_root_runtime_expectations() {
        let config = StorageLocation::default();

        assert_eq!(config.url, "file://./data");
        assert_eq!(config.weight, 1);
        assert_eq!(config.tags, vec!["local"]);
    }

    #[test]
    fn assignment_defaults_match_root_runtime_expectations() {
        let config = AssignmentConfig::default();

        assert_eq!(config.strategy, "hash");
        assert!(config.affinity);
    }

    #[test]
    fn compaction_defaults_match_root_runtime_expectations() {
        let config = CompactionConfig::default();

        assert_eq!(config.l0_file_threshold, 5);
        assert_eq!(
            config.higher_level_file_threshold, 2,
            "two query-visible files must consolidate instead of stranding a pair"
        );
        assert_eq!(config.l0_size_threshold_mb, 256);
        assert_eq!(
            config.level_multiplier, 1.0,
            "vector-search levels must consolidate at the L0 threshold; \
             a RocksDB-style 2.0 multiplier strands query-visible segments"
        );
        assert_eq!(config.max_levels, 7);
        assert_eq!(config.strategy, "hybrid");
        assert_eq!(config.target_file_size_mb, 128);
    }

    #[test]
    fn optimization_defaults_match_root_runtime_expectations() {
        let config = OptimizationConfig::default();

        assert!(config.enable_mmap);
        assert!(config.enable_zone_map_pruning);
        // ADR-070: AXIS is opt-in; co-design PAX + survivor cache is the default.
        assert!(!config.enable_axis_indexes);
        assert_eq!(config.default_index_type, "hnsw");
        assert!(config.enable_progressive_search);
        assert!(config.enable_bloom_filters);
    }

    #[test]
    fn advanced_prune_defaults_match_root_runtime_expectations() {
        let config = AdvancedPruneConfig::default();

        assert_eq!(config.r#type, "sqrt");
        assert!(config.min_keep.is_none());
        assert!(config.max_keep.is_none());
        assert!(config.ratio.is_none());
    }

    // === PR 3: EmbeddingPrecisionConfig ===

    #[test]
    fn embedding_precision_default_is_off() {
        // PR 3: rolling deploy default is V1-only; operator must opt in
        // after every node is V2-capable.
        let cfg = EmbeddingPrecisionConfig::default();
        assert!(!cfg.schema_v2_enabled);
    }

    #[test]
    fn embedding_precision_env_var_name_matches_lld() {
        assert_eq!(
            EmbeddingPrecisionConfig::ENV_VAR,
            "PROXIMADB_EMBED_PRECISION_SCHEMA_V2"
        );
    }

    #[test]
    fn parse_bool_flag_accepts_canonical_true_forms() {
        for raw in [
            "true", "TRUE", "True", "1", "yes", "YES", "on", "ON", " true ",
        ] {
            assert!(
                parse_bool_flag(raw, "X").unwrap(),
                "expected {raw:?} to parse as true"
            );
        }
    }

    #[test]
    fn parse_bool_flag_accepts_canonical_false_forms() {
        for raw in ["false", "FALSE", "0", "no", "NO", "off", "OFF", " false "] {
            assert!(
                !parse_bool_flag(raw, "X").unwrap(),
                "expected {raw:?} to parse as false"
            );
        }
    }

    #[test]
    fn parse_bool_flag_rejects_garbage_to_avoid_silent_default() {
        for raw in ["maybe", "2", "y", "", "enabled"] {
            assert!(
                parse_bool_flag(raw, "X").is_err(),
                "expected {raw:?} to be rejected"
            );
        }
    }

    #[test]
    fn embedding_precision_serde_roundtrip_v2_off() {
        let cfg = EmbeddingPrecisionConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: EmbeddingPrecisionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn embedding_precision_serde_roundtrip_v2_on() {
        let cfg = EmbeddingPrecisionConfig {
            schema_v2_enabled: true,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: EmbeddingPrecisionConfig = serde_json::from_str(&json).unwrap();
        assert!(back.schema_v2_enabled);
    }

    #[test]
    fn embedding_precision_serde_defaults_missing_field_to_off() {
        // Back-compat with config files that pre-date PR 3.
        let cfg: EmbeddingPrecisionConfig = serde_json::from_str("{}").unwrap();
        assert!(!cfg.schema_v2_enabled);
    }
}
