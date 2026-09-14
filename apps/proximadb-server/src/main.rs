/*
 * Copyright 2024 Vijaykumar Singh
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

//! # ProximaDB Server - Production-Ready Vector Database Server
//!
//! This is the main server binary for ProximaDB, providing a complete vector database
//! server with REST and gRPC APIs, automatic hardware detection, and cloud-native storage.
//!
//! ## Usage
//!
//! ```bash
//! proximadb-server [OPTIONS]
//!
//! Options:
//!   -c, --config <PATH>      Configuration file [default: config/config.toml]
//!   -d, --data-dir <PATH>    Data directory override
//!   -p, --port <PORT>        REST API port override
//!   --node-id <ID>           Node identifier for clustering
//!   -l, --log-level <LEVEL>  Log level (error, warn, info, debug, trace)
//!   -h, --help               Print help information
//! ```

// Embedding precision rollout (EMBEDDING_PRECISION_LLD_2026_05_22 §Endianness):
// canonical embedding payloads, WAL segment headers, and PAX block headers all
// use little-endian byte order without runtime swap. Refuse to build on
// big-endian targets so the on-disk format is never written incorrectly.
#[cfg(target_endian = "big")]
compile_error!(
    "proximadb does not support big-endian targets; \
     target a little-endian arch (x86_64, aarch64, etc.)"
);

mod runtime_state;

use clap::Parser;
use proximadb::ProximaDB;
use proximadb::core::ConfigLoader;
use proximadb::core::hardware_capabilities::initialize_hardware_capabilities;
use std::path::PathBuf;
use tracing::{error, info, warn};

mod storage_ownership;

#[derive(Parser)]
#[command(name = "proximadb-server")]
#[command(about = "ProximaDB cloud-native vector database server")]
struct Args {
    #[arg(short, long, default_value = "config/config.toml")]
    config: PathBuf,

    #[arg(short, long)]
    data_dir: Option<PathBuf>,

    #[arg(short, long)]
    port: Option<u16>,

    #[arg(long)]
    node_id: Option<String>,

    #[arg(short, long, help = "Set log level (error, warn, info, debug, trace)")]
    log_level: Option<String>,
}

/// Ensure all required directories exist based on configuration
async fn ensure_required_directories(config: &proximadb::core::Config) -> anyhow::Result<()> {
    info!("🔧 Ensuring all required directories exist...");

    // Extract base data directory from storage configuration
    let base_data_dir = config.server.data_dir.to_string_lossy();
    info!("📂 Base data directory: {}", base_data_dir);

    // Create base data directory
    tokio::fs::create_dir_all(base_data_dir.as_ref())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create base data directory: {}", e))?;

    // Create storage location directories
    for location in &config.storage.storage_locations {
        if let Some(path) = location.url.strip_prefix("file://") {
            info!("📂 Creating storage location directory: {}", path);
            tokio::fs::create_dir_all(path).await.map_err(|e| {
                anyhow::anyhow!("Failed to create storage directory {}: {}", path, e)
            })?;
        }
    }

    // Create metadata directory with required subdirectories
    let metadata_url = &config.storage.metadata_url;
    if let Some(base_path) = metadata_url.strip_prefix("file://") {
        info!("📂 Creating metadata directories at: {}", base_path);

        // Create base metadata directory
        tokio::fs::create_dir_all(base_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create metadata directory: {}", e))?;

        // Create required subdirectories
        let subdirs = ["current", "archive", "__staging"];
        for subdir in &subdirs {
            let subdir_path = format!("{}/{}", base_path, subdir);
            info!("  📁 Creating subdirectory: {}", subdir_path);
            tokio::fs::create_dir_all(&subdir_path).await.map_err(|e| {
                anyhow::anyhow!("Failed to create subdirectory {}: {}", subdir_path, e)
            })?;
        }
    }

    info!("✅ All required directories created successfully");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // ProximaDB's pgwire OLAP path plans through DataFusion, whose SQL planner + optimizer recurse
    // deeply on nested-subquery / cross-modal plans — e.g. a derived table wrapping a
    // `vector_search` / `timeseries_range` / `graph_traverse` aggregate that is then JOINed to a
    // Parquet-backed base table. tokio's default 2 MiB worker stack overflows on that recursion and
    // aborts the WHOLE server (a single crafted query = DoS; TD-XMODAL-7). Databases routinely run
    // recursive query planners on a larger stack, so give the runtime workers 32 MiB — deep-but-
    // bounded planning then completes instead of crashing. (Debug builds have much larger frames,
    // so this also covers the amplified debug-mode overflow.)
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(32 * 1024 * 1024)
        .build()?;
    let result = runtime.block_on(run());
    // TD-LIFECYCLE-1: several always-on background loops leak their shutdown
    // senders by design (discovery executor, recall observer — see
    // shared_services.rs), so a plain Runtime drop can block forever on an
    // in-flight blocking pass and the process never exits after a clean
    // SIGTERM shutdown. All durable state is already persisted by
    // `ProximaDB::shutdown` before we get here; bound the teardown instead of
    // inheriting drop's wait-for-everything semantics.
    runtime.shutdown_timeout(std::time::Duration::from_secs(5));
    result
}

async fn run() -> anyhow::Result<()> {
    // Initialize tracing with rolling file appender
    use tracing_appender::rolling::{RollingFileAppender, Rotation};
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    // Create log directory if it doesn't exist
    if let Err(e) = std::fs::create_dir_all("./log") {
        eprintln!("Failed to create log directory: {}", e);
    }

    // Create rolling file appender (daily rotation)
    let file_appender = RollingFileAppender::new(Rotation::DAILY, "./log", "proximadb.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Create console appender for stdout - production friendly format
    let console_layer = tracing_subscriber::fmt::layer()
        .with_target(false) // Cleaner output without module paths
        .with_line_number(false) // No line numbers in production
        .with_file(false) // No file names in production
        .with_thread_ids(false) // No thread IDs for cleaner output
        .with_thread_names(false) // No thread names
        .compact() // Use compact format for better readability
        .with_writer(std::io::stdout);

    // Create file appender layer
    let file_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_line_number(true)
        .with_file(true)
        .with_ansi(false) // No ANSI colors in file
        .with_writer(non_blocking);

    // Parse arguments first to get config path
    let args = Args::parse();

    // Load configuration first to get log level from TOML
    let config_path_for_state = args.config.to_string_lossy().to_string();
    let mut config = ConfigLoader::load_with_defaults(args.config.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    // Initialize subscriber with both console and file output - fully configurable
    // Priority: Environment variable > CLI args > TOML config > Default (INFO)
    // Default to info level - only show info, warn, error (no debug/trace)
    let log_level = std::env::var("RUST_LOG")
        .or_else(|_| args.log_level.clone().ok_or(()))
        .unwrap_or_else(|_| {
            // If config has a log level, use it, otherwise default to info
            if config.monitoring.log_level.is_empty()
                || config.monitoring.log_level == "debug"
                || config.monitoring.log_level == "trace"
            {
                // Override debug/trace with info for production
                "info".to_string()
            } else {
                config.monitoring.log_level.clone()
            }
        });

    // Create environment filter with ProximaDB-specific defaults
    // This ensures we only see info and above for proximadb modules
    let env_filter = EnvFilter::try_new(&log_level)
        .or_else(|_| EnvFilter::try_new("proximadb=info"))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();

    info!(
        "🚀 ProximaDB Server v{} starting",
        env!("CARGO_PKG_VERSION")
    );
    info!(
        "📊 Log level: {} (use RUST_LOG env or --log-level to change)",
        log_level
    );

    // Override with CLI arguments
    if let Some(data_dir) = args.data_dir {
        // TD-PATH-1: `-d` must also rebase the storage/metadata locations that
        // are still at their BUILT-IN cwd-relative defaults — otherwise every
        // instance launched from one working directory shares `<cwd>/data` +
        // `<cwd>/metadata` (one catalog, one collection-id space) regardless
        // of `-d`, and instances cross-contaminate (observed 2026-07-25).
        // Explicitly configured URLs are never touched (mixed-safe).
        rebase_default_relative_paths(&mut config, &data_dir);
        config.server.data_dir = data_dir;
    }
    if let Some(port) = args.port {
        config.server.port = port;
    }
    if let Some(node_id) = args.node_id {
        config.server.node_id = node_id;
    }

    // Initialize hardware capabilities detection early with configuration
    info!("🔧 Initializing hardware detection...");
    let hardware_config = config.hardware.clone().unwrap_or_default();
    if let Err(e) = initialize_hardware_capabilities(hardware_config) {
        warn!("⚠️ Hardware capability detection failed: {}", e);
        info!("Continuing with CPU-only mode");
    }

    // PR 7a follow-up: probe per-precision matmul latency so the policy
    // resolver + embedding service can read the cached snapshot without
    // re-running the micro-bench. Idempotent — OnceLock guards re-init.
    // Runs BEFORE the embedding service so BgeModel can read the result
    // when picking its default loaded precision.
    let caps = proximadb_embedding::precision::hw_capability::init_precision_probe();
    info!(
        "🎯 Precision hw probe (dim={}): f32_f32={}ns, f16_f32={}ns, f16_f16={}ns, \
         bf16_supported={}, best_canonical={:?}",
        caps.probe_dim,
        caps.f32_f32_matmul_ns,
        caps.f16_f32_matmul_ns,
        caps.f16_f16_matmul_ns,
        caps.bf16_supported,
        caps.best_canonical_for_inference(),
    );

    // PR 7b follow-up: initialize the precision-metrics Prometheus
    // family and populate `proximadb_embedding_precision_hw_matmul_ns`
    // from the probe so the Grafana dashboard reflects the cached
    // measurement immediately. Other metrics (segments, conversions,
    // recall) populate from their respective hot-path callers as those
    // call sites land. Init is idempotent (OnceLock-backed).
    {
        use proximadb::observability::precision_metrics::{
            self, DTYPE_PAIR_F16_F16, DTYPE_PAIR_F16_F32, DTYPE_PAIR_F32_F32,
        };
        let pm = precision_metrics::init_precision_metrics();
        pm.set_hw_matmul_ns(DTYPE_PAIR_F32_F32, caps.f32_f32_matmul_ns as i64);
        pm.set_hw_matmul_ns(DTYPE_PAIR_F16_F32, caps.f16_f32_matmul_ns as i64);
        pm.set_hw_matmul_ns(DTYPE_PAIR_F16_F16, caps.f16_f16_matmul_ns as i64);
    }

    // Initialize the in-process embedding singleton. Loads BGE ONNX sessions
    // (or synthetic fallback when the `onnx` feature is off), spawns sync +
    // async tokio runtimes, and registers Prometheus instruments. Crashing
    // here is intentional — embedding is part of the data plane, not optional.
    info!("🧠 Initializing embedding service (in-process, Arc-shared)...");
    // Resolve the BGE variant from PROXIMADB_EMBED_VARIANT (small/large/m3,
    // default small). Maps to the corresponding EmbedRoute and ONNX file
    // resolved by Variant::onnx_path.
    let variant = proximadb_embedding::models::bge::resolve_variant(
        std::env::var("PROXIMADB_EMBED_VARIANT").ok().as_deref(),
    );
    let route = match variant {
        proximadb_embedding::models::bge::Variant::Small => {
            proximadb_embedding::config::EmbedRoute::BgeSmall
        }
        proximadb_embedding::models::bge::Variant::Large => {
            proximadb_embedding::config::EmbedRoute::BgeLarge
        }
        proximadb_embedding::models::bge::Variant::M3 => {
            proximadb_embedding::config::EmbedRoute::BgeM3
        }
    };
    info!(
        ?variant,
        ?route,
        "selected BGE variant from PROXIMADB_EMBED_VARIANT"
    );
    proximadb_embedding::EmbeddingService::initialize(
        proximadb_embedding::config::EmbeddingConfig { route },
        proximadb_embedding::scheduler::EmbedSchedulerConfig::from_env(),
    )
    .map_err(|e| anyhow::anyhow!("embedding service init: {}", e))?;
    info!("✅ Embedding service ready");

    info!("Starting ProximaDB server with config: {:?}", config);

    // Ensure all required directories exist
    ensure_required_directories(&config).await?;

    // Runtime state is observability, not fencing. Acquire kernel-backed locks
    // before publishing lifecycle state or initializing any engine.
    let _storage_ownership = storage_ownership::StorageOwnershipLease::acquire(&config)?;

    // Publish the lifecycle record BEFORE any slow work. Recovery can run for
    // minutes on a large data dir and binds no socket until it finishes, so a
    // supervising client that polls HTTP cannot distinguish "recovering" from
    // "dead" — it applied a fixed timeout and killed healthy servers (#1667).
    // The record carries pid + phase + an advancing heartbeat, and doubles as
    // the ownership marker that makes orphaned servers detectable
    // (anvai-labs/victor#911).
    // Symmetry with the client-side check: if a LIVE server already owns this
    // data dir, say so loudly. Two writers on one dir is how "cleared" state
    // silently resurrects. Not fatal — an operator may be doing this
    // deliberately, and refusing to boot would be a worse failure mode than
    // a warning they can see.
    if let Some(existing) = runtime_state::read_state(&config.server.data_dir) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if existing.is_live_owner(now_ms) {
            tracing::warn!(
                owner_pid = existing.pid,
                owner_phase = ?existing.phase,
                data_dir = %config.server.data_dir.display(),
                "another ProximaDB process appears to be serving this data dir;                  concurrent writers can corrupt or resurrect state"
            );
        } else {
            tracing::info!(
                stale_pid = existing.pid,
                "reclaiming data dir from a previous owner that did not shut down cleanly"
            );
        }
    }

    let runtime_state = runtime_state::RuntimeStateWriter::start(
        &config.server.data_dir,
        Some(config_path_for_state.clone()),
    );

    // Create and start the database
    runtime_state.set_phase(runtime_state::Phase::RecoveringStorage);
    let mut db = ProximaDB::new(config).await?;

    // Start the database engine
    runtime_state.set_phase(runtime_state::Phase::Binding);
    db.start().await?;

    let serving = runtime_state::Phase::Serving;
    runtime_state.set_phase(serving);
    info!(
        ready = serving.is_ready(),
        "ProximaDB server started successfully"
    );

    // Wait for a shutdown signal — SIGINT (ctrl-c) *or* SIGTERM. `docker stop`,
    // Kubernetes pod termination, and Spot-eviction drains all deliver SIGTERM;
    // waiting only on `ctrl_c()` left SIGTERM at its default disposition
    // (immediate process kill), so containerized stops bypassed `db.shutdown()`
    // entirely — no shutdown flush, no clean close, and any unflushed memtable
    // rode solely on WAL replay (issue #1125, finding A).
    // Terminal consumer failures cannot be retried in place without reacquiring
    // authority. Observe the existing task handle and leave replacement policy
    // to the process supervisor after normal awaited shutdown. Do not use
    // is_healthy(): its HTTP predicate excludes valid gRPC-only deployments.
    let drainer_stopped = async {
        if db.queue_client().is_none() {
            std::future::pending::<()>().await;
        }
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            tick.tick().await;
            if db.drainer_has_stopped() {
                break;
            }
        }
    };
    #[cfg(unix)]
    let critical_drainer_stopped = {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received SIGINT, stopping server...");
                false
            },
            _ = sigterm.recv() => {
                info!("Received SIGTERM, stopping server...");
                false
            },
            _ = drainer_stopped => true,
        }
    };
    #[cfg(not(unix))]
    let critical_drainer_stopped = {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal?;
                info!("Received shutdown signal, stopping server...");
                false
            },
            _ = drainer_stopped => true,
        }
    };
    if critical_drainer_stopped {
        error!("Critical embedding drainer stopped; shutting down server");
    }

    // Graceful shutdown
    runtime_state.set_phase(runtime_state::Phase::Stopping);
    if let Err(e) = db.shutdown().await {
        error!("Error during shutdown: {}", e);
    }

    // Clearing the record is the "no owner" signal: a supervisor that sees no
    // file knows the data dir is free, and one that sees a stale record knows
    // the previous owner died without cleaning up.
    runtime_state.finish();
    info!("ProximaDB server stopped");
    if critical_drainer_stopped {
        anyhow::bail!("critical embedding drainer terminated; server shutdown completed");
    }
    Ok(())
}

/// TD-PATH-1: when `-d <dir>` is given, rewrite any storage/metadata location
/// still at its built-in cwd-relative default so it lives under `<dir>`.
/// Explicit (non-default) configuration is left untouched.
fn rebase_default_relative_paths(
    config: &mut proximadb::core::config::Config,
    data_dir: &std::path::Path,
) {
    let base = data_dir.display();
    let rebased = |suffix: &str| format!("file://{base}/{suffix}");
    for loc in &mut config.storage.storage_locations {
        if loc.url == "file://./data" || loc.url == "./data" {
            loc.url = rebased("data");
            tracing::info!(url = %loc.url, "rebased default storage location under -d");
        }
    }
    if config.storage.metadata_url == "file://./metadata"
        || config.storage.metadata_url == "./metadata"
    {
        config.storage.metadata_url = rebased("metadata");
        tracing::info!(url = %config.storage.metadata_url, "rebased default metadata_url under -d");
    }
    let wal_dir = &mut config.storage.wal_config.write_buffer_directory;
    if wal_dir == "./data/write_buffer" || wal_dir == "file://./data/write_buffer" {
        *wal_dir = rebased("data/write_buffer");
        tracing::info!(url = %wal_dir, "rebased default WAL directory under -d");
    }
    if let Some(viper) = config.storage.viper_config.as_mut()
        && viper.data_directory == "./data/viper_data"
    {
        viper.data_directory = format!("{base}/data/viper_data");
        tracing::info!(dir = %viper.data_directory, "rebased default viper directory under -d");
    }
}

#[cfg(test)]
mod path_rebase_tests {
    use super::*;

    /// TD-PATH-1: `-d` rebases built-in relative defaults; explicit URLs stay.
    #[test]
    fn rebase_touches_only_builtin_defaults() {
        let mut config = proximadb::core::config::Config::default();
        // defaults are the cwd-relative built-ins
        rebase_default_relative_paths(&mut config, std::path::Path::new("/tmp/x"));
        assert!(
            config
                .storage
                .storage_locations
                .iter()
                .all(|l| l.url == "file:///tmp/x/data")
        );
        assert_eq!(config.storage.metadata_url, "file:///tmp/x/metadata");

        let mut explicit = proximadb::core::config::Config::default();
        explicit.storage.metadata_url = "file:///somewhere/else".into();
        rebase_default_relative_paths(&mut explicit, std::path::Path::new("/tmp/x"));
        assert_eq!(explicit.storage.metadata_url, "file:///somewhere/else");
    }
}
