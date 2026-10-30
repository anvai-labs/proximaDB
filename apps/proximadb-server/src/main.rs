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

mod cli;
mod cold_backup;
mod durable_locations;
mod runtime_state;
mod shutdown;

use clap::Parser;
use proximadb::ProximaDB;
use proximadb::core::ConfigLoader;
use proximadb::core::hardware_capabilities::initialize_hardware_capabilities;
use tracing::{error, info, warn};

mod storage_ownership;
mod writer_epoch;

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
    // rustls 0.23.46 (RUSTSEC-2026-0346..0349 fix, #2032) no longer auto-picks
    // the process-default CryptoProvider when the feature-unified build carries
    // BOTH `aws-lc-rs` (aws-smithy stack) and `ring` (axum-server/reqwest paths)
    // — it panics at first TLS construction instead, which the Docker health
    // check caught as a boot-time server death. Install aws-lc-rs explicitly
    // (rustls's own default preference, and what the Glue/aws path already
    // compiles) before anything can construct a TLS endpoint.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let cli = cli::Cli::parse();
    cli.validate()?;
    if let Some(command) = cli.command {
        // Offline maintenance: synchronous, no runtime, no embedding/hardware
        // setup — and it resolves storage roots exactly as serving would.
        return run_admin(&cli.common, command);
    }

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
    let result = runtime.block_on(run(cli.common, cli.serve));
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

async fn run(common: cli::CommonArgs, serve: cli::ServeArgs) -> anyhow::Result<()> {
    // Install the OS signal handlers BEFORE any server boots. A `Signal`
    // created here registers its handler immediately and buffers a
    // pending receipt, whereas `tokio::signal::ctrl_c()` only registers
    // when first polled — i.e. when the shutdown select arms, AFTER the
    // listeners are already serving. A SIGINT/SIGTERM landing in that
    // boot/serving window previously hit the default disposition and
    // killed the process with no shutdown flush (seen as the
    // cold_backup_restore "unclean exit during graceful stop: signal 2"
    // CI flake; the same window would drop an early `docker stop`).
    #[cfg(unix)]
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    // Initialize tracing with rolling file appender
    use tracing_appender::rolling::{RollingFileAppender, Rotation};
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    // The rolling log lives under the effective data directory when one is
    // set (TD-PKG-2: a Windows service gets no working directory, so a
    // cwd-relative ./log would write into System32). Without a data dir the
    // historical ./log is kept. Same precedence as apply_data_dir_override:
    // CLI -d, then PROXIMADB_DATA_DIR.
    let log_dir: std::path::PathBuf = common
        .data_dir
        .clone()
        .or_else(|| std::env::var_os("PROXIMADB_DATA_DIR").map(std::path::PathBuf::from))
        .map(|d| d.join("log"))
        .unwrap_or_else(|| std::path::PathBuf::from("./log"));
    if let Err(e) = std::fs::create_dir_all(&log_dir) {
        eprintln!(
            "Failed to create log directory {}: {}",
            log_dir.display(),
            e
        );
    }

    // Create rolling file appender (daily rotation)
    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "proximadb.log");
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

    // Load configuration first to get log level from TOML
    let config_path_for_state = common.config.to_string_lossy().to_string();
    let mut config = load_config(&common.config)?;

    // Initialize subscriber with both console and file output - fully configurable
    // Priority: Environment variable > CLI args > TOML config > Default (INFO)
    // Default to info level - only show info, warn, error (no debug/trace)
    let log_level = std::env::var("RUST_LOG")
        .or_else(|_| common.log_level.clone().ok_or(()))
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

    // Override with the effective data directory.
    // TD-PATH-1: `-d` must also rebase the storage/metadata locations that
    // are still at their BUILT-IN cwd-relative defaults — otherwise every
    // instance launched from one working directory shares `<cwd>/data` +
    // `<cwd>/metadata` (one catalog, one collection-id space) regardless
    // of `-d`, and instances cross-contaminate (observed 2026-07-25).
    // Explicitly configured URLs are never touched (mixed-safe).
    // TD-PKG-3: PROXIMADB_DATA_DIR gets the same treatment when -d is absent
    // (the container and the packages' unit set it).
    apply_data_dir_override(&mut config, common.data_dir.as_deref());
    if let Some(port) = serve.port {
        config.server.port = port;
    }
    if let Some(node_id) = serve.node_id {
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
    // A restore that did not complete leaves its marker in data_dir; booting on
    // a half-restored set of roots would silently serve partial data.
    let restore_marker = config
        .server
        .data_dir
        .join(cold_backup::RESTORE_IN_PROGRESS_FILE);
    if restore_marker.exists() {
        anyhow::bail!(
            "refusing to start: a cold restore into these storage roots did not complete ({}). \
             Re-run `proximadb-server restore --from <backup> --force` to finish it.",
            restore_marker.display()
        );
    }
    // Record where THIS process writes durable state outside the leased roots
    // (its env + config, absolute), so an offline backup run from another shell
    // can refuse instead of silently omitting it.
    durable_locations::record(&config)?;
    // ADR-095 D1/D2: with `[oltp.durable_volume]`, prove every durable root is
    // on the volume and win the writer epoch BEFORE any storage is opened, so a
    // replaced machine never writes alongside a still-running previous owner.
    // The watchdog it starts exits the process (code 75) if fenced.
    let _writer_epoch = writer_epoch::acquire(&config).await?;
    if let Some(guard) = &_writer_epoch {
        info!(
            epoch = guard.epoch,
            "durable volume owned under writer epoch"
        );
    }

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
        tokio::select! {
            _ = sigint.recv() => {
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
    shutdown::finish(runtime_state, async || {
        let result = db.shutdown().await;
        // A failed shutdown retains the queue only when the drainer is still
        // in flight. Database shutdown removes it before all later failures.
        // Keep this classifier aligned with that ordering; do not retry a
        // completed drainer's terminal error or match error-message strings.
        (result, db.queue_client().is_some())
    })
    .await?;
    info!("ProximaDB server stopped");
    if critical_drainer_stopped {
        anyhow::bail!("critical embedding drainer terminated; server shutdown completed");
    }
    Ok(())
}

/// Load the TOML configuration (shared by serving and offline subcommands).
fn load_config(path: &std::path::Path) -> anyhow::Result<proximadb::core::Config> {
    ConfigLoader::load_with_defaults(path.to_string_lossy().as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))
}

/// The configuration an offline subcommand operates on: the SAME load + `-d`
/// rebase serving applies, so backup/restore target exactly the roots the
/// server would own.
fn resolve_admin_config(common: &cli::CommonArgs) -> anyhow::Result<proximadb::core::Config> {
    let mut config = load_config(&common.config)?;
    apply_data_dir_override(&mut config, common.data_dir.as_deref());
    Ok(config)
}

fn run_admin(common: &cli::CommonArgs, command: cli::AdminCommand) -> anyhow::Result<()> {
    let filter = std::env::var("RUST_LOG")
        .ok()
        .or_else(|| common.log_level.clone())
        .unwrap_or_else(|| "warn".to_string());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&filter)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let config = resolve_admin_config(common)?;
    let (verb, summary) = match command {
        cli::AdminCommand::Backup { to } => ("backup", cold_backup::run_backup(&config, &to)?),
        cli::AdminCommand::Restore { from, force } => {
            ("restore", cold_backup::run_restore(&config, &from, force)?)
        }
    };
    for warning in &summary.warnings {
        eprintln!("WARNING: {warning}");
    }
    println!(
        "{verb} complete: {} root(s), {} file(s), {} bytes",
        summary.roots, summary.files, summary.bytes
    );
    Ok(())
}

/// Roots still pointing at the shipped development config's `/tmp/proximadb`
/// tree count as defaults too (TD-PKG-1/2/3): the packages, the container, the
/// Linux archive and the Homebrew formula all ship that config, and under
/// `PrivateTmp` / a removed container that tree is not durable. A deployment
/// that explicitly configured a root keeps it.
fn rebase_default_root(value: &str, base: &str, url_style: bool) -> Option<String> {
    for prefix in ["/tmp/proximadb/", "file:///tmp/proximadb/"] {
        if let Some(rest) = value.strip_prefix(prefix) {
            return Some(if url_style {
                format!("file://{base}/{rest}")
            } else {
                format!("{base}/{rest}")
            });
        }
    }
    None
}

/// TD-PATH-1 / TD-PKG-3: when a data directory is given (CLI `-d` or
/// `PROXIMADB_DATA_DIR`), rewrite every storage/metadata location still at a
/// built-in default — the cwd-relative built-ins or the shipped `/tmp/proximadb`
/// tree — so it lives under it. Explicit configuration is left untouched.
fn rebase_default_relative_paths(
    config: &mut proximadb::core::config::Config,
    data_dir: &std::path::Path,
) {
    let base = data_dir.display().to_string();
    let rebased = |suffix: &str| format!("file://{base}/{suffix}");
    for loc in &mut config.storage.storage_locations {
        let url = loc.url.clone();
        if url == "file://./data" || url == "./data" {
            loc.url = rebased("data");
            tracing::info!(url = %loc.url, "rebased default storage location under the data dir");
        } else if let Some(next) = rebase_default_root(&url, &base, true) {
            loc.url = next;
            tracing::info!(url = %loc.url, "rebased shipped-/tmp storage location under the data dir");
        }
    }
    let metadata_url = config.storage.metadata_url.clone();
    if metadata_url == "file://./metadata" || metadata_url == "./metadata" {
        config.storage.metadata_url = rebased("metadata");
        tracing::info!(url = %config.storage.metadata_url, "rebased default metadata_url under the data dir");
    } else if let Some(next) = rebase_default_root(&metadata_url, &base, true) {
        config.storage.metadata_url = next;
        tracing::info!(url = %config.storage.metadata_url, "rebased shipped-/tmp metadata_url under the data dir");
    }
    if let Some(manifest) = config.storage.wal_config.global_manifest_url.clone()
        && let Some(next) = rebase_default_root(&manifest, &base, true)
    {
        tracing::info!(manifest = %next, "rebased shipped-/tmp global_manifest_url under the data dir");
        config.storage.wal_config.global_manifest_url = Some(next);
    }
    let wal_dir_value = config.storage.wal_config.write_buffer_directory.clone();
    if wal_dir_value == "./data/write_buffer" || wal_dir_value == "file://./data/write_buffer" {
        config.storage.wal_config.write_buffer_directory = format!("{base}/data/write_buffer");
        tracing::info!(dir = %config.storage.wal_config.write_buffer_directory, "rebased default WAL directory under the data dir");
    } else if let Some(next) = rebase_default_root(&wal_dir_value, &base, false) {
        tracing::info!(dir = %next, "rebased shipped-/tmp WAL directory under the data dir");
        config.storage.wal_config.write_buffer_directory = next;
    }
    if let Some(viper) = config.storage.viper_config.as_mut() {
        let value = viper.data_directory.clone();
        if value == "./data/viper_data" {
            viper.data_directory = format!("{base}/data/viper_data");
            tracing::info!(dir = %viper.data_directory, "rebased default viper directory under the data dir");
        } else if let Some(next) = rebase_default_root(&value, &base, false) {
            tracing::info!(dir = %next, "rebased shipped-/tmp viper directory under the data dir");
            viper.data_directory = next;
        }
    }
    if let Some(sst) = config.storage.sst_config.as_mut() {
        let value = sst.data_directory.clone();
        if value == "./sst_data" {
            sst.data_directory = format!("{base}/sst_data");
            tracing::info!(dir = %sst.data_directory, "rebased default sst directory under the data dir");
        } else if let Some(next) = rebase_default_root(&value, &base, false) {
            tracing::info!(dir = %next, "rebased shipped-/tmp sst directory under the data dir");
            sst.data_directory = next;
        }
    }
    if let Some(security) = config.security.as_mut()
        && let proximadb::security::AuditStorageBackend::File { directory } =
            &mut security.audit.storage_backend
        && let Some(next) = rebase_default_root(directory, &base, false)
    {
        tracing::info!(dir = %next, "rebased shipped-/tmp audit directory under the data dir");
        *directory = next;
    }
}

/// Apply the effective data directory: CLI `-d` wins, then
/// `PROXIMADB_DATA_DIR` (TD-PKG-3: the container has always set it; before
/// this the loader never read it, so every root stayed at the shipped
/// `/tmp/proximadb` tree and the declared volume received no data).
fn apply_data_dir_override(
    config: &mut proximadb::core::config::Config,
    cli_dir: Option<&std::path::Path>,
) {
    let dir = cli_dir
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("PROXIMADB_DATA_DIR").map(std::path::PathBuf::from));
    if let Some(dir) = dir {
        rebase_default_relative_paths(config, &dir);
        config.server.data_dir = dir;
    }
}

#[cfg(test)]
mod path_rebase_tests {
    use super::*;

    /// A `SecurityConfig` whose audit backend is a file at `directory` (test
    /// helper: the full SecurityConfig has many required fields).
    fn security_config_with_file_audit(directory: &str) -> proximadb::security::SecurityConfig {
        proximadb::security::SecurityConfig {
            audit: proximadb::security::AuditConfig {
                storage_backend: proximadb::security::AuditStorageBackend::File {
                    directory: directory.to_string(),
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

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
    /// TD-PKG-3: the shipped development config roots everything under
    /// /tmp/proximadb; a data dir must move every one of those roots, or the
    /// packaged artifacts keep their data in a non-durable tree.
    #[test]
    fn rebase_moves_shipped_tmp_roots() {
        let mut config = proximadb::core::config::Config::default();
        config.storage.metadata_url = "file:///tmp/proximadb/metadata".into();
        config.storage.storage_locations = vec![
            proximadb::core::config::StorageLocation {
                url: "file:///tmp/proximadb/d1".into(),
                ..Default::default()
            },
            proximadb::core::config::StorageLocation {
                url: "file:///tmp/proximadb/d2".into(),
                ..Default::default()
            },
        ];
        config.storage.wal_config.global_manifest_url =
            Some("file:///tmp/proximadb/manifest".into());
        config.storage.wal_config.write_buffer_directory =
            "/tmp/proximadb/data/write_buffer".into();
        config.storage.sst_config = Some(proximadb::core::config::SstConfig {
            data_directory: "/tmp/proximadb/data/sst_data".into(),
            ..Default::default()
        });
        config.storage.viper_config = Some(proximadb::core::config::ViperConfig {
            data_directory: "/tmp/proximadb/data/data/viper_data".into(),
            ..Default::default()
        });
        config.security = Some(security_config_with_file_audit("/tmp/proximadb/audit"));

        rebase_default_relative_paths(&mut config, std::path::Path::new("/var/lib/proximadb"));

        assert_eq!(
            config.storage.metadata_url,
            "file:///var/lib/proximadb/metadata"
        );
        assert_eq!(
            config
                .storage
                .storage_locations
                .iter()
                .map(|l| l.url.as_str())
                .collect::<Vec<_>>(),
            vec![
                "file:///var/lib/proximadb/d1",
                "file:///var/lib/proximadb/d2"
            ]
        );
        assert_eq!(
            config.storage.wal_config.global_manifest_url.as_deref(),
            Some("file:///var/lib/proximadb/manifest")
        );
        assert_eq!(
            config.storage.wal_config.write_buffer_directory,
            "/var/lib/proximadb/data/write_buffer"
        );
        assert_eq!(
            config.storage.sst_config.as_ref().unwrap().data_directory,
            "/var/lib/proximadb/data/sst_data"
        );
        assert_eq!(
            config.storage.viper_config.as_ref().unwrap().data_directory,
            "/var/lib/proximadb/data/data/viper_data"
        );
        match &config.security.as_ref().unwrap().audit.storage_backend {
            proximadb::security::AuditStorageBackend::File { directory } => {
                assert_eq!(directory, "/var/lib/proximadb/audit");
            }
            other => panic!("audit backend changed shape: {other:?}"),
        }
    }

    /// An explicit root outside the shipped tree is never moved, even when
    /// other roots are.
    #[test]
    fn rebase_leaves_explicit_roots_alone() {
        let mut config = proximadb::core::config::Config::default();
        config.storage.metadata_url = "file:///mnt/nvme/meta".into();
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: "file:///tmp/proximadb/d1".into(),
            ..Default::default()
        }];
        rebase_default_relative_paths(&mut config, std::path::Path::new("/var/lib/proximadb"));
        assert_eq!(config.storage.metadata_url, "file:///mnt/nvme/meta");
        assert_eq!(
            config.storage.storage_locations[0].url,
            "file:///var/lib/proximadb/d1"
        );
    }
}
