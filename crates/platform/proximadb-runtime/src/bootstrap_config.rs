// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Server bootstrap configuration types for the ProximaDB multi-protocol server.
//!
//! Pure data types — no I/O, no service dependencies. Contains REST, gRPC,
//! Arrow Flight, PostgreSQL wire, cluster, and TLS subsystem configs.
//!
//! Phase 9.9 / Task #70 pre-work: lifted from `src/network/server_config.rs`
//! into the horizontal runtime crate so future bootstrap extraction (the
//! main Phase 9.9 work) doesn't have to drag a root-crate dependency along.
//! No behavioural change vs the previous location; downstream root-crate
//! callers continue to import via `crate::network::multi_server::*`
//! re-exports of `proximadb_runtime::bootstrap_config::*`.

use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Where a network surface should listen: a TCP socket or a Unix-domain socket.
///
/// Portless ("embedded") mode resolves every surface to [`BindTarget::Uds`] so
/// the process opens no TCP listener at all; the default server keeps binding
/// [`BindTarget::Tcp`]. This is the bind-target half of the `transport = "uds"`
/// opt-in (see `proximadb_config::ApiConfig::transport`).
#[derive(Debug, Clone)]
pub enum BindTarget {
    /// Bind a TCP socket at the given address (default / server mode).
    Tcp(SocketAddr),
    /// Bind a Unix-domain socket at the given filesystem path (embedded mode).
    Uds(PathBuf),
}

// Socket file names under `MultiServerConfig::uds_socket_dir`. The Python SDK
// must use identical names when deriving its UDS HTTP/gRPC/Flight clients.
/// Socket file name for the REST surface.
pub const UDS_REST_SOCKET_NAME: &str = "rest.sock";
/// Socket file name for the gRPC surface.
pub const UDS_GRPC_SOCKET_NAME: &str = "grpc.sock";
/// Socket file name for the Arrow Flight surface.
pub const UDS_FLIGHT_SOCKET_NAME: &str = "flight.sock";

/// Multi-server configuration supporting HTTP and gRPC with binary Avro payloads
///
/// ## Configuration Strategy:
///
/// The MultiServerConfig aggregates settings for both protocols,
/// allowing unified configuration while maintaining protocol-specific
/// optimizations.
///
/// ## Key Settings:
///
/// - **Ports**: Separate ports prevent protocol confusion
/// - **Compression**: Can differ between REST (JSON) and gRPC (Protobuf)
/// - **Message Limits**: gRPC typically needs larger limits for batch ops
/// - **TLS**: Shared or separate certificates supported
#[derive(Debug, Clone)]
pub struct MultiServerConfig {
    /// HTTP server configuration (REST/Dashboard/Metrics)
    /// Handles JSON payloads, web UI, and monitoring endpoints
    pub http_config: RestHttpServerConfig,

    /// gRPC server configuration with binary Avro payloads
    /// Optimized for high-throughput vector operations
    pub grpc_config: GrpcHttpServerConfig,

    /// Arrow IPC (Flight) server configuration
    /// Optimized for bulk ingestion and ETL pipelines
    pub arrow_ipc_config: ArrowIpcServerConfig,

    /// PostgreSQL wire protocol server configuration
    /// Enables pgvector compatibility and psql/pgAdmin connections
    pub postgres_config: PostgresServerConfig,

    /// Global TLS configuration - applies to all servers
    /// Can be overridden per-server if needed
    pub tls_config: TLSConfig,

    /// API configuration (request limits, timeouts, etc.)
    /// Shared limits and policies across all protocols
    pub api_config: Option<proximadb_config::ApiConfig>,

    /// Data directory from server config (server.data_dir from TOML)
    /// Used by REST handlers for document/observability storage paths
    pub data_dir: std::path::PathBuf,

    // ============================================================
    // Unified Port Architecture (Phase 14)
    // ============================================================
    /// Enable unified port mode (REST + gRPC + Arrow Flight on single port)
    /// When enabled, `unified_port` is used; individual ports are ignored.
    /// Default: false (legacy multi-port mode for backward compatibility)
    pub unified_mode: bool,

    /// Unified port for all HTTP-based protocols (REST, gRPC, Arrow Flight)
    /// Only used when `unified_mode = true`
    /// Default: 5678
    pub unified_port: u16,

    /// Bind address for unified port (e.g., "0.0.0.0")
    pub unified_bind_address: String,

    /// Portless ("embedded") mode: directory holding the Unix-domain sockets for
    /// the REST / gRPC / Arrow Flight surfaces. `None` (default) keeps TCP mode;
    /// `Some(dir)` makes [`Self::rest_bind_target`], [`Self::grpc_bind_target`],
    /// and [`Self::arrow_bind_target`] resolve to `<dir>/proximadb-embedded.*.sock`
    /// and the `*_port` fields are ignored. Set by the bootstrap when
    /// `[api].transport = "uds"`.
    pub uds_socket_dir: Option<PathBuf>,

    /// Mount the read-only embedded admin dashboard at `/admin` (+ `/dashboard`).
    /// Sourced from `[server.admin_ui] enabled` in the TOML config. Default:
    /// `false` — off for Kubernetes pods and embedded/UDS deployments unless an
    /// operator explicitly opts in for a standalone instance.
    pub admin_ui_enabled: bool,

    /// Sourced from `[server.admin_ui] auto_refresh` (default `false`). When
    /// `false`, the dashboard's auto-refresh toggle is rendered disabled.
    pub admin_ui_auto_refresh: bool,

    /// Sourced from `[server.admin_ui] refresh_interval_seconds` (default 30).
    pub admin_ui_refresh_interval_seconds: u32,

    /// Cluster mode configuration (consensus, replication, health services)
    /// Only used when `cluster` feature is enabled
    #[cfg(feature = "cluster")]
    pub cluster_config: Option<ClusterServerConfig>,
}

/// Cluster server configuration for distributed mode
#[cfg(feature = "cluster")]
#[derive(Debug, Clone)]
pub struct ClusterServerConfig {
    /// This node's unique identifier
    pub node_id: String,

    /// Enable cluster consensus service (Raft)
    pub enable_consensus: bool,

    /// Enable cluster replication service
    pub enable_replication: bool,

    /// Enable cluster health service
    pub enable_health: bool,
}

#[cfg(feature = "cluster")]
impl Default for ClusterServerConfig {
    fn default() -> Self {
        Self {
            node_id: format!("node-{}", uuid::Uuid::new_v4()),
            enable_consensus: true,
            enable_replication: true,
            enable_health: true,
        }
    }
}

/// PostgreSQL wire protocol server configuration
///
/// ## PostgreSQL Compatibility:
///
/// - **Wire Protocol**: PostgreSQL Protocol v3.0
/// - **pgvector Support**: <->, <=>, <#> distance operators
/// - **Clients**: psql, pgAdmin, application drivers
///
/// ## Use Cases:
///
/// - Migration path from pgvector
/// - Familiar SQL interface for vector operations
/// - Existing application compatibility
#[derive(Debug, Clone)]
pub struct PostgresServerConfig {
    /// PostgreSQL bind port (default: 5432 or 5433)
    pub port: u16,

    /// Bind address
    pub bind_address: SocketAddr,

    /// Enable PostgreSQL server
    pub enable_postgres: bool,

    /// Maximum connections
    pub max_connections: usize,

    /// Idle timeout (seconds)
    pub idle_timeout_secs: u64,

    /// Statement cache size per connection
    pub statement_cache_size: usize,

    /// Enable canonical record/WAL writes for catalog-routed relational DML over
    /// pgwire. Defaults on; retirement gates 1–5 are satisfied. Disable only
    /// when falling back to legacy VectorOps-only path for debugging.
    pub enable_direct_record_writes: bool,

    /// Optional framed canonical WAL path for pgwire direct record writes.
    /// Defaults to `<server.data_dir>/pgwire/canonical-records.wal`.
    pub direct_record_wal_path: Option<PathBuf>,
}

impl Default for PostgresServerConfig {
    fn default() -> Self {
        Self {
            port: 5433, // Use 5433 to avoid conflict with real PostgreSQL
            bind_address: SocketAddr::from(([127, 0, 0, 1], 5433)),
            enable_postgres: true,
            max_connections: 100,
            idle_timeout_secs: 3600,
            statement_cache_size: 100,
            enable_direct_record_writes: true,
            direct_record_wal_path: None,
        }
    }
}

impl PostgresServerConfig {
    /// Get active bind address
    pub fn active_bind_address(&self) -> SocketAddr {
        self.bind_address
    }
}

/// Global TLS configuration for all protocols
#[derive(Debug, Clone)]
pub struct TLSConfig {
    /// TLS certificate file path
    pub cert_file: Option<String>,

    /// TLS private key file path
    pub key_file: Option<String>,

    /// Private key password (for encrypted keys)
    pub key_password: Option<String>,

    /// Interface to bind (default: 0.0.0.0)
    pub bind_interface: String,

    /// Enable TLS (auto-detected from cert/key availability)
    pub enabled: bool,

    /// CA certificate file path for mTLS client verification
    pub ca_file: Option<String>,

    /// Require client certificates (mTLS mode)
    pub require_client_certs: bool,

    /// Auto-generate self-signed certificates for development
    pub auto_generate: bool,

    /// Certificate validity period in days (for auto-generated certs)
    pub validity_days: u32,

    /// Days before expiration to trigger renewal warnings
    pub renewal_threshold_days: u32,
}

impl Default for TLSConfig {
    fn default() -> Self {
        Self {
            cert_file: None,
            key_file: None,
            key_password: None,
            bind_interface: "0.0.0.0".to_string(),
            enabled: false,
            ca_file: None,
            require_client_certs: false,
            auto_generate: false,
            validity_days: 365,
            renewal_threshold_days: 30,
        }
    }
}

impl TLSConfig {
    /// Create a new TLS configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure for mTLS with CA certificate
    pub fn with_mtls(mut self, ca_file: &str) -> Self {
        self.ca_file = Some(ca_file.to_string());
        self.require_client_certs = true;
        self
    }

    /// Set certificate and key files
    pub fn with_certificates(mut self, cert_file: &str, key_file: &str) -> Self {
        self.cert_file = Some(cert_file.to_string());
        self.key_file = Some(key_file.to_string());
        self.enabled = true;
        self
    }

    /// Enable auto-generation of self-signed certificates
    pub fn with_auto_generate(mut self, validity_days: u32) -> Self {
        self.auto_generate = true;
        self.validity_days = validity_days;
        self
    }

    /// Check if mTLS is configured
    pub fn is_mtls_enabled(&self) -> bool {
        self.enabled && self.require_client_certs && self.ca_file.is_some()
    }

    /// Get CA certificate path if configured
    pub fn get_ca_path(&self) -> Option<&str> {
        self.ca_file.as_deref()
    }
}

/// HTTP server configuration for REST, Dashboard, and Metrics
///
/// ## REST Server Endpoints:
///
/// - `/v1/collections`: Collection CRUD operations
/// - `/v1/vectors`: Vector search and management
/// - `/v1/health`: Kubernetes health probes
/// - `/metrics`: Prometheus metrics
/// - `/dashboard`: Web UI (if enabled)
///
/// ## Compression Strategy:
///
/// HTTP compression disabled by default because:
/// - CPU overhead often exceeds network savings
/// - Most deployments use fast local/datacenter networks
/// - Can be enabled for WAN deployments
#[derive(Debug, Clone)]
pub struct RestHttpServerConfig {
    /// HTTP bind port (default: 5678)
    /// Standard port for ProximaDB REST API
    pub port: u16,

    /// Enable REST API endpoints
    /// Core CRUD and search operations
    pub enable_rest: bool,

    /// Enable monitoring dashboard
    /// Web UI for cluster monitoring
    pub enable_dashboard: bool,

    /// Enable metrics endpoint
    /// Prometheus-compatible metrics at /metrics
    pub enable_metrics: bool,

    /// Enable health check endpoint
    /// Kubernetes liveness/readiness probes
    pub enable_health: bool,

    /// Enable HTTP compression (default: false for better performance)
    /// Trade CPU for bandwidth - useful for WAN
    pub compression: bool,

    /// TLS certificate file path
    /// PEM-encoded X.509 certificate
    pub tls_cert_file: Option<String>,

    /// TLS private key file path
    /// PEM-encoded private key (RSA/ECDSA)
    pub tls_key_file: Option<String>,
}

impl RestHttpServerConfig {
    /// Check if TLS is enabled
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_cert_file.is_some() && self.tls_key_file.is_some()
    }

    /// Verify TLS certificates exist
    pub fn verify_tls_certificates(&self) -> bool {
        if let (Some(cert), Some(key)) = (&self.tls_cert_file, &self.tls_key_file) {
            std::path::Path::new(cert).exists() && std::path::Path::new(key).exists()
        } else {
            false
        }
    }

    /// Get active bind address
    pub fn active_bind_address(&self) -> SocketAddr {
        format!("0.0.0.0:{}", self.port)
            .parse()
            .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], self.port)))
    }
}

/// gRPC server configuration with binary Avro payload support
///
/// ## gRPC Advantages:
///
/// - **Binary Protocol**: 2-3x smaller than JSON
/// - **HTTP/2**: Multiplexing, server push, header compression
/// - **Streaming**: Bidirectional streams for bulk operations
/// - **Type Safety**: Strongly typed protobuf contracts
///
/// ## Message Size Considerations:
///
/// Default 64MB supports:
/// - 100K vectors of 128 dimensions
/// - 25K vectors of 512 dimensions
/// - 8K vectors of 1536 dimensions (OpenAI)
///
/// Increase for larger batches or use streaming.
#[derive(Debug, Clone)]
pub struct GrpcHttpServerConfig {
    /// gRPC bind port (default: 5679)
    /// Standard port for ProximaDB gRPC API
    pub port: u16,

    /// Bind address (computed from port and interface)
    /// Usually 0.0.0.0:5679 for all interfaces
    pub bind_address: SocketAddr,

    /// TLS bind address (optional)
    /// Same port, TLS-only listener
    pub tls_bind_address: Option<SocketAddr>,

    /// Enable gRPC endpoints
    /// Core service implementation
    pub enable_grpc: bool,

    /// Maximum message size in bytes
    /// Prevents OOM from malicious/accidental huge messages
    pub max_message_size: usize,

    /// Enable gRPC reflection
    /// Allows dynamic service discovery (grpcurl, etc)
    pub enable_reflection: bool,

    /// Enable gRPC compression for Avro payloads
    /// Further reduces already-compact protobuf
    pub compression: bool,

    /// TLS certificate file path
    /// Same format as REST server
    pub tls_cert_file: Option<String>,

    /// TLS private key file path
    /// Can share with REST or use separate
    pub tls_key_file: Option<String>,
}

impl GrpcHttpServerConfig {
    /// Check if TLS is enabled
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_cert_file.is_some() && self.tls_key_file.is_some()
    }

    /// Verify TLS certificates exist
    pub fn verify_tls_certificates(&self) -> bool {
        if let (Some(cert), Some(key)) = (&self.tls_cert_file, &self.tls_key_file) {
            std::path::Path::new(cert).exists() && std::path::Path::new(key).exists()
        } else {
            false
        }
    }

    /// Get active bind address
    pub fn active_bind_address(&self) -> SocketAddr {
        self.bind_address
    }
}

/// Arrow IPC (Flight) server configuration for high-throughput bulk ingestion
///
/// ## Arrow Flight Advantages:
///
/// - **High Throughput**: 100K-200K vectors/sec for bulk ingestion
/// - **Zero-Copy**: Arrow columnar format minimizes serialization overhead
/// - **Streaming**: Efficient for large dataset transfers
/// - **Standardized**: Apache Arrow IPC protocol
///
/// ## Use Cases:
///
/// - ETL pipelines and data migrations
/// - Bulk vector uploads from data lakes
/// - Direct writes bypassing WAL for maximum speed
/// - Explicit flush/compaction control
///
/// ## Message Size Considerations:
///
/// Default 512MB supports massive batch uploads:
/// - 1M vectors of 384 dimensions
/// - 500K vectors of 768 dimensions
/// - 200K vectors of 1536 dimensions
#[derive(Debug, Clone)]
pub struct ArrowIpcServerConfig {
    /// Arrow IPC bind port (default: 5680)
    /// Standard port for ProximaDB Arrow Flight API
    pub port: u16,

    /// Bind address (computed from port and interface)
    /// Usually 0.0.0.0:5680 for all interfaces
    pub bind_address: SocketAddr,

    /// Enable Arrow IPC endpoints
    /// Core Flight service implementation
    pub enable_arrow_ipc: bool,

    /// Maximum message size in bytes
    /// Default 512MB for large batch uploads
    pub max_message_size: usize,

    /// Enable Arrow IPC compression
    /// Note: Arrow has built-in compression, usually disabled at transport layer
    pub compression: bool,

    /// TLS certificate file path
    /// Same format as REST/gRPC servers
    pub tls_cert_file: Option<String>,

    /// TLS private key file path
    /// Can share with other servers or use separate
    pub tls_key_file: Option<String>,
}

impl ArrowIpcServerConfig {
    /// Check if TLS is enabled
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_cert_file.is_some() && self.tls_key_file.is_some()
    }

    /// Verify TLS certificates exist
    pub fn verify_tls_certificates(&self) -> bool {
        if let (Some(cert), Some(key)) = (&self.tls_cert_file, &self.tls_key_file) {
            std::path::Path::new(cert).exists() && std::path::Path::new(key).exists()
        } else {
            false
        }
    }

    /// Get active bind address
    pub fn active_bind_address(&self) -> SocketAddr {
        self.bind_address
    }
}

impl Default for MultiServerConfig {
    fn default() -> Self {
        Self {
            http_config: RestHttpServerConfig {
                port: 5678,
                enable_rest: true,
                enable_dashboard: true,
                enable_metrics: true,
                enable_health: true,
                compression: false, // Default to false for better debugging
                tls_cert_file: None,
                tls_key_file: None,
            },
            grpc_config: GrpcHttpServerConfig {
                port: 5679,
                bind_address: "0.0.0.0:5679"
                    .parse()
                    .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 5679))),
                tls_bind_address: None,
                enable_grpc: true,
                max_message_size: 64 * 1024 * 1024, // 64MB for bulk vector inserts with Avro
                enable_reflection: true,
                compression: true,
                tls_cert_file: None,
                tls_key_file: None,
            },
            arrow_ipc_config: ArrowIpcServerConfig {
                port: 5680,
                bind_address: "0.0.0.0:5680"
                    .parse()
                    .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 5680))),
                enable_arrow_ipc: true,
                max_message_size: 512 * 1024 * 1024, // 512MB for massive batch uploads
                compression: false,                  // Arrow has built-in compression
                tls_cert_file: None,
                tls_key_file: None,
            },
            postgres_config: PostgresServerConfig::default(),
            tls_config: TLSConfig::default(),
            api_config: None, // Will be set when creating from Config
            data_dir: std::path::PathBuf::from("/tmp/proximadb/data"), // Default fallback
            // Unified port defaults (Phase 14)
            unified_mode: false, // Legacy multi-port mode by default
            unified_port: 5678,  // Use REST port for unified
            unified_bind_address: "0.0.0.0".to_string(),
            uds_socket_dir: None,    // TCP mode by default; portless opt-in only
            admin_ui_enabled: false, // read-only /admin dashboard off by default; opt-in via TOML
            admin_ui_auto_refresh: false, // dashboard auto-refresh off by default; opt-in via TOML
            admin_ui_refresh_interval_seconds: 30, // default poll cadence (seconds)
            // Cluster mode defaults
            #[cfg(feature = "cluster")]
            cluster_config: None, // Cluster mode disabled by default
        }
    }
}

impl TLSConfig {
    /// Auto-detect TLS capability from certificate availability
    pub fn auto_detect_tls(&mut self) -> bool {
        if let (Some(cert_file), Some(key_file)) = (&self.cert_file, &self.key_file) {
            let cert_exists = std::path::Path::new(cert_file).exists();
            let key_exists = std::path::Path::new(key_file).exists();

            if cert_exists && key_exists {
                // Additional validation: try to parse the certificate
                if self.validate_certificates() {
                    self.enabled = true;
                    info!("🔒 TLS enabled - certificates validated");
                    return true;
                } else {
                    warn!("⚠️ TLS certificates found but invalid - using non-TLS");
                }
            } else {
                if !cert_exists {
                    debug!("📋 TLS certificate not found: {}", cert_file);
                }
                if !key_exists {
                    debug!("📋 TLS private key not found: {}", key_file);
                }
            }
        }

        self.enabled = false;
        info!("🌐 TLS disabled - using non-TLS mode");
        false
    }

    /// Validate certificate files can be read and parsed
    pub fn validate_certificates(&self) -> bool {
        if let (Some(cert_file), Some(key_file)) = (&self.cert_file, &self.key_file) {
            // Basic file existence and readability check
            match (std::fs::read(cert_file), std::fs::read(key_file)) {
                (Ok(cert_data), Ok(key_data)) => {
                    // Basic validation - check if files contain PEM markers
                    let cert_str = String::from_utf8_lossy(&cert_data);
                    let key_str = String::from_utf8_lossy(&key_data);

                    cert_str.contains("-----BEGIN CERTIFICATE-----")
                        && cert_str.contains("-----END CERTIFICATE-----")
                        && (key_str.contains("-----BEGIN PRIVATE KEY-----")
                            || key_str.contains("-----BEGIN RSA PRIVATE KEY-----")
                            || key_str.contains("-----BEGIN EC PRIVATE KEY-----"))
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Get bind address for given port
    pub fn bind_address(&self, port: u16) -> SocketAddr {
        self.bind_interface
            .parse()
            .map(|ip| SocketAddr::new(ip, port))
            .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], port)))
    }
}

/// Admission for surfaces without request-bound credentials and permissions.
/// TLS or a caller-supplied tenant assertion cannot substitute for those —
/// UNLESS the listener enforces its own per-connection authentication
/// (`listener_authenticated`, e.g. pgwire SCRAM, TD-PGWIRE-AUTH-1), in which
/// case the single-tenant/loopback/no-coordinator restriction doesn't apply:
/// the listener is no longer trust-only.
pub fn validate_trust_only_listener(
    surface: &str,
    address: SocketAddr,
    coordinator_present: bool,
    listener_authenticated: bool,
    mode: &proximadb_tenant::TenantDeploymentMode,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        listener_authenticated
            || (!coordinator_present
                && matches!(
                    mode,
                    proximadb_tenant::TenantDeploymentMode::SingleTenant { .. }
                )
                && address.ip().is_loopback()),
        "{surface} currently supports only unauthenticated single-tenant loopback access \
         unless the listener enforces its own per-connection authentication; \
         disable this listener in authenticated, multi-tenant, or non-loopback deployments"
    );
    Ok(())
}

impl MultiServerConfig {
    /// Validate the entire listener set before binding any of its sockets.
    /// `pgwire_authenticated` is true when pgwire enforces its own
    /// per-connection authentication (SCRAM, TD-PGWIRE-AUTH-1) — MCP has no
    /// such capability and stays trust-only regardless.
    pub fn validate_listener_security(
        &self,
        coordinator_present: bool,
        pgwire_authenticated: bool,
        mode: &proximadb_tenant::TenantDeploymentMode,
    ) -> anyhow::Result<()> {
        self.tls_config
            .bind_interface
            .parse::<std::net::IpAddr>()
            .map_err(|_| anyhow::anyhow!("invalid HTTP bind interface"))?;
        if self.is_unified_mode() {
            self.unified_bind_address
                .parse::<std::net::IpAddr>()
                .map_err(|_| anyhow::anyhow!("invalid unified bind interface"))?;
        }
        anyhow::ensure!(
            coordinator_present
                || !matches!(mode, proximadb_tenant::TenantDeploymentMode::MultiTenant),
            "multi-tenant network serving requires a security coordinator"
        );
        if self.postgres_config.enable_postgres {
            anyhow::ensure!(
                !self.is_uds_mode(),
                "pgwire must be disabled in portless mode"
            );
            validate_trust_only_listener(
                "pgwire",
                self.postgres_config.active_bind_address(),
                coordinator_present,
                pgwire_authenticated,
                mode,
            )?;
        }
        if let Some(port) = self.api_config.as_ref().and_then(|api| api.mcp_port) {
            anyhow::ensure!(
                !self.is_uds_mode(),
                "MCP TCP listener must be disabled in portless mode"
            );
            validate_trust_only_listener(
                "MCP",
                SocketAddr::new(self.http_bind_address().ip(), port),
                coordinator_present,
                false,
                mode,
            )?;
        }
        Ok(())
    }

    /// Get effective bind address for HTTP server
    pub fn http_bind_address(&self) -> SocketAddr {
        self.tls_config.bind_address(self.http_config.port)
    }

    /// Get effective bind address for gRPC server
    pub fn grpc_bind_address(&self) -> SocketAddr {
        self.tls_config.bind_address(self.grpc_config.port)
    }

    /// Get effective bind address for unified port mode
    pub fn unified_bind_address(&self) -> SocketAddr {
        self.unified_bind_address
            .parse()
            .map(|ip| SocketAddr::new(ip, self.unified_port))
            .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], self.unified_port)))
    }

    /// Resolve where the REST surface should listen. Returns a Unix-domain
    /// socket under [`Self::uds_socket_dir`] in portless mode, else the TCP
    /// HTTP bind address.
    pub fn rest_bind_target(&self) -> BindTarget {
        match &self.uds_socket_dir {
            Some(dir) => BindTarget::Uds(dir.join(UDS_REST_SOCKET_NAME)),
            None => BindTarget::Tcp(self.http_bind_address()),
        }
    }

    /// Resolve where the gRPC surface should listen (UDS in portless mode,
    /// else the TCP gRPC bind address).
    pub fn grpc_bind_target(&self) -> BindTarget {
        match &self.uds_socket_dir {
            Some(dir) => BindTarget::Uds(dir.join(UDS_GRPC_SOCKET_NAME)),
            None => BindTarget::Tcp(self.grpc_bind_address()),
        }
    }

    /// Resolve where the Arrow Flight surface should listen (UDS in portless
    /// mode, else the TCP Arrow IPC bind address).
    pub fn arrow_bind_target(&self) -> BindTarget {
        match &self.uds_socket_dir {
            Some(dir) => BindTarget::Uds(dir.join(UDS_FLIGHT_SOCKET_NAME)),
            None => BindTarget::Tcp(self.arrow_ipc_config.active_bind_address()),
        }
    }

    /// True when the server is in portless (Unix-domain socket) transport mode.
    pub fn is_uds_mode(&self) -> bool {
        self.uds_socket_dir.is_some()
    }

    /// Check if unified port mode is enabled
    pub fn is_unified_mode(&self) -> bool {
        self.unified_mode
    }

    /// Check if TLS is enabled globally
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_config.enabled
    }
}

/// Server status information returned by `MultiServer::status()`
#[derive(Debug, Clone)]
pub struct ServerStatus {
    /// Whether the HTTP/REST server is running
    pub http_running: bool,
    /// Whether the gRPC server is running
    pub grpc_running: bool,
    /// HTTP server bind address (if running)
    pub http_address: Option<SocketAddr>,
    /// gRPC server bind address (if running)
    pub grpc_address: Option<SocketAddr>,
    /// Whether TLS is enabled for connections
    pub tls_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_only_listener_rejects_shared_or_authenticated_deployment() {
        let local = SocketAddr::from(([127, 0, 0, 1], 5433));
        let single = proximadb_tenant::TenantDeploymentMode::single_tenant_default();
        for surface in ["pgwire", "MCP"] {
            assert!(validate_trust_only_listener(surface, local, false, false, &single).is_ok());
            assert!(validate_trust_only_listener(surface, local, true, false, &single).is_err());
            assert!(
                validate_trust_only_listener(
                    surface,
                    local,
                    false,
                    false,
                    &proximadb_tenant::TenantDeploymentMode::MultiTenant
                )
                .is_err()
            );
            for address in [
                "0.0.0.0:5433",
                "192.0.2.1:5433",
                "[::]:5433",
                "[::ffff:192.0.2.1]:5433",
            ] {
                assert!(
                    validate_trust_only_listener(
                        surface,
                        address.parse().unwrap(),
                        false,
                        false,
                        &single
                    )
                    .is_err()
                );
            }
            assert!(
                validate_trust_only_listener(
                    surface,
                    "[::1]:5433".parse().unwrap(),
                    false,
                    false,
                    &single
                )
                .is_ok()
            );
        }
    }

    /// TD-PGWIRE-AUTH-1: a listener that enforces its own per-connection
    /// authentication (SCRAM) is no longer trust-only — it's admitted with a
    /// coordinator present, in multi-tenant mode, and on a non-loopback
    /// address, none of which are individually sufficient without
    /// `listener_authenticated`.
    #[test]
    fn trust_only_listener_admits_an_authenticated_listener_anywhere() {
        assert!(
            validate_trust_only_listener(
                "pgwire",
                "0.0.0.0:5433".parse().unwrap(),
                true,
                true,
                &proximadb_tenant::TenantDeploymentMode::MultiTenant
            )
            .is_ok()
        );
        // Not authenticated: the exact same shared/multi-tenant/remote
        // deployment is still rejected.
        assert!(
            validate_trust_only_listener(
                "pgwire",
                "0.0.0.0:5433".parse().unwrap(),
                true,
                false,
                &proximadb_tenant::TenantDeploymentMode::MultiTenant
            )
            .is_err()
        );
    }

    #[test]
    fn listener_preflight_checks_optional_surfaces_before_startup() {
        let single = proximadb_tenant::TenantDeploymentMode::single_tenant_default();
        let mut config = MultiServerConfig::default();
        config.postgres_config.bind_address = SocketAddr::from(([127, 0, 0, 1], 5433));
        assert!(
            config
                .validate_listener_security(true, false, &single)
                .is_err()
        );
        config.postgres_config.enable_postgres = false;
        assert!(
            config
                .validate_listener_security(true, false, &single)
                .is_ok()
        );
        config.api_config = Some(proximadb_config::ApiConfig {
            mcp_port: Some(5700),
            ..Default::default()
        });
        assert!(
            config
                .validate_listener_security(true, false, &single)
                .is_err()
        );
        config.api_config = None;
        assert!(
            config
                .validate_listener_security(
                    false,
                    false,
                    &proximadb_tenant::TenantDeploymentMode::MultiTenant
                )
                .is_err()
        );
    }

    /// TD-PGWIRE-AUTH-1: an authenticated pgwire listener bypasses the
    /// trust-only restriction the previous test exercises above — the
    /// otherwise-identical shared-coordinator deployment now admits it.
    #[test]
    fn listener_preflight_admits_authenticated_pgwire_with_a_coordinator() {
        let single = proximadb_tenant::TenantDeploymentMode::single_tenant_default();
        let mut config = MultiServerConfig::default();
        config.postgres_config.bind_address = SocketAddr::from(([127, 0, 0, 1], 5433));
        assert!(
            config
                .validate_listener_security(true, true, &single)
                .is_ok()
        );
    }

    #[test]
    fn listener_preflight_preserves_local_mcp_but_not_portless_tcp() {
        let single = proximadb_tenant::TenantDeploymentMode::single_tenant_default();
        let mut config = MultiServerConfig::default();
        config.postgres_config.enable_postgres = false;
        config.tls_config.bind_interface = "127.0.0.1".into();
        config.api_config = Some(proximadb_config::ApiConfig {
            mcp_port: Some(5700),
            ..Default::default()
        });
        assert!(
            config
                .validate_listener_security(false, false, &single)
                .is_ok()
        );
        config.uds_socket_dir = Some(PathBuf::from("unused-test-sockets"));
        assert!(
            config
                .validate_listener_security(false, false, &single)
                .is_err()
        );
        config.api_config.as_mut().unwrap().mcp_port = None;
        assert!(
            config
                .validate_listener_security(false, false, &single)
                .is_ok()
        );
    }

    #[test]
    fn test_server_config_defaults() {
        let config = MultiServerConfig::default();

        // REST port
        assert_eq!(config.http_config.port, 5678);
        assert!(config.http_config.enable_rest);
        assert!(config.http_config.enable_dashboard);
        assert!(config.http_config.enable_metrics);
        assert!(config.http_config.enable_health);
        assert!(!config.http_config.compression);

        // gRPC port
        assert_eq!(config.grpc_config.port, 5679);
        assert!(config.grpc_config.enable_grpc);
        assert_eq!(config.grpc_config.max_message_size, 64 * 1024 * 1024); // 64MB
        assert!(config.grpc_config.enable_reflection);
        assert!(config.grpc_config.compression);

        // Arrow IPC port
        assert_eq!(config.arrow_ipc_config.port, 5680);
        assert!(config.arrow_ipc_config.enable_arrow_ipc);
        assert_eq!(config.arrow_ipc_config.max_message_size, 512 * 1024 * 1024); // 512MB
        assert!(!config.arrow_ipc_config.compression);

        // PostgreSQL port
        assert_eq!(config.postgres_config.port, 5433);
        assert!(config.postgres_config.enable_postgres);
        assert_eq!(config.postgres_config.max_connections, 100);
        assert!(config.postgres_config.enable_direct_record_writes);
        assert!(config.postgres_config.direct_record_wal_path.is_none());

        // TLS defaults
        assert!(!config.tls_config.enabled);
        assert!(config.tls_config.cert_file.is_none());
        assert!(config.tls_config.key_file.is_none());
    }

    #[test]
    fn test_server_config_unified_mode() {
        let config = MultiServerConfig::default();

        // Default: unified mode disabled (legacy multi-port)
        assert!(!config.unified_mode);
        assert!(!config.is_unified_mode());
        assert_eq!(config.unified_port, 5678);
        assert_eq!(config.unified_bind_address, "0.0.0.0");

        // Verify unified bind address computation
        let unified_addr = config.unified_bind_address();
        assert_eq!(unified_addr.port(), 5678);

        // Enable unified mode
        let mut unified_config = config;
        unified_config.unified_mode = true;
        unified_config.unified_port = 9999;
        assert!(unified_config.is_unified_mode());
        assert_eq!(unified_config.unified_bind_address().port(), 9999);
    }

    #[test]
    fn test_server_config_multi_port() {
        let mut config = MultiServerConfig {
            unified_mode: false,
            ..Default::default()
        };

        // Verify each protocol gets its own port
        let http_addr = config.http_bind_address();
        let grpc_addr = config.grpc_bind_address();

        assert_eq!(http_addr.port(), 5678);
        assert_eq!(grpc_addr.port(), 5679);

        // Arrow IPC and Postgres use their own bind addresses
        assert_eq!(config.arrow_ipc_config.active_bind_address().port(), 5680);
        assert_eq!(config.postgres_config.active_bind_address().port(), 5433);

        // Verify all ports are distinct
        let ports = [
            http_addr.port(),
            grpc_addr.port(),
            config.arrow_ipc_config.port,
            config.postgres_config.port,
        ];
        let unique: std::collections::HashSet<_> = ports.iter().collect();
        assert_eq!(
            unique.len(),
            ports.len(),
            "All protocol ports must be unique"
        );

        // Verify custom port assignment
        config.http_config.port = 8080;
        config.grpc_config.port = 8081;
        config.grpc_config.bind_address = "0.0.0.0:8081"
            .parse()
            .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 8081)));

        assert_eq!(config.http_bind_address().port(), 8080);
        assert_eq!(config.grpc_bind_address().port(), 8081);
    }

    #[test]
    fn test_protocol_detection() {
        // Test the unified mode protocol detection configuration
        // (actual TCP-level detection happens at runtime, but we verify the config wiring)
        let config = MultiServerConfig {
            unified_mode: true,
            ..Default::default()
        };

        // In unified mode, all protocols share one address
        let unified = config.unified_bind_address();
        assert_eq!(unified.port(), config.unified_port);

        // Verify TLS auto-detection with no certificates returns false
        let mut tls = TLSConfig::default();
        assert!(!tls.auto_detect_tls());
        assert!(!tls.enabled);

        // Verify TLS with certificates that don't exist returns false
        let mut tls_with_fake = TLSConfig {
            cert_file: Some("/nonexistent/cert.pem".to_string()),
            key_file: Some("/nonexistent/key.pem".to_string()),
            ..Default::default()
        };
        assert!(!tls_with_fake.auto_detect_tls());
        assert!(!tls_with_fake.enabled);

        // Verify mTLS detection
        let tls_no_mtls = TLSConfig::default();
        assert!(!tls_no_mtls.is_mtls_enabled());

        let tls_mtls = TLSConfig {
            enabled: true,
            require_client_certs: true,
            ca_file: Some("/path/to/ca.pem".to_string()),
            ..Default::default()
        };
        assert!(tls_mtls.is_mtls_enabled());
        assert_eq!(tls_mtls.get_ca_path(), Some("/path/to/ca.pem"));

        // Verify bind address construction
        let tls_for_bind = TLSConfig::default();
        let addr = tls_for_bind.bind_address(5678);
        assert_eq!(addr.port(), 5678);
        assert_eq!(addr.ip(), std::net::IpAddr::from([0, 0, 0, 0]));

        // Verify REST TLS detection
        let rest_config = RestHttpServerConfig {
            port: 5678,
            enable_rest: true,
            enable_dashboard: false,
            enable_metrics: false,
            enable_health: true,
            compression: false,
            tls_cert_file: None,
            tls_key_file: None,
        };
        assert!(!rest_config.is_tls_enabled());

        // Verify gRPC TLS detection
        let grpc_config = GrpcHttpServerConfig {
            port: 5679,
            bind_address: "0.0.0.0:5679"
                .parse()
                .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 5679))),
            tls_bind_address: None,
            enable_grpc: true,
            max_message_size: 64 * 1024 * 1024,
            enable_reflection: true,
            compression: true,
            tls_cert_file: None,
            tls_key_file: None,
        };
        assert!(!grpc_config.is_tls_enabled());

        // Verify Arrow IPC TLS detection
        let arrow_config = ArrowIpcServerConfig {
            port: 5680,
            bind_address: "0.0.0.0:5680"
                .parse()
                .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 5680))),
            enable_arrow_ipc: true,
            max_message_size: 512 * 1024 * 1024,
            compression: false,
            tls_cert_file: None,
            tls_key_file: None,
        };
        assert!(!arrow_config.is_tls_enabled());
    }
}
