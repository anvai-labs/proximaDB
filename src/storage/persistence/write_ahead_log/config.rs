// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! WAL Configuration with Smart Defaults for Performance

// NOTE: CompressionAlgorithm moved to unified_types.rs
// Write Buffer-specific configuration uses the unified type
pub use crate::core::CompressionAlgorithm;

use serde::{Deserialize, Serialize};

/// ADR-098 D3 (TD-SPINE-5): the WAL durability mode — exactly one of two
/// contracts governs what an acknowledged write survives. The mode is
/// **declared** (`[storage.wal_config].durability_mode`, env mirror
/// `PROXIMADB_WAL_DURABILITY_MODE`) or **inferred at boot** from one
/// deterministic input (see [`resolve_durability_mode`]); it is logged at INFO
/// and recorded in `durable_locations`. There is no silent legacy path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurabilityMode {
    /// The WAL frame is fsync'd on a local, reattachable volume (ADR-069 D1 /
    /// ADR-095 D1); the object store is the coalesced durable tier for flushed
    /// data.
    #[serde(rename = "local_volume")]
    LocalVolume,
    /// The WAL object is durable and LISTable on the object store (ADR-063 D3
    /// plus the WAL-object parts of D4: pre-PUT ordering token, put-if-absent
    /// WAL objects, LIST replay under `{collection}/wal/`). This is the
    /// intended serverless posture, not a degraded one.
    #[serde(rename = "object_store_serverless")]
    ObjectStoreServerless,
}

impl std::fmt::Display for DurabilityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DurabilityMode::LocalVolume => write!(f, "local_volume"),
            DurabilityMode::ObjectStoreServerless => write!(f, "object_store_serverless"),
        }
    }
}

impl DurabilityMode {
    /// Parse a config/env value (case-insensitive, surrounding whitespace
    /// tolerated). Unknown values are errors, not silent fallbacks — the
    /// TOML path (serde) already rejects them at config load.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local_volume" => Some(DurabilityMode::LocalVolume),
            "object_store_serverless" => Some(DurabilityMode::ObjectStoreServerless),
            _ => None,
        }
    }
}

/// Where a resolved [`DurabilityMode`] came from — evidence for the boot log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurabilityModeSource {
    /// Declared via TOML `[storage.wal_config].durability_mode` or its env
    /// mirror `PROXIMADB_WAL_DURABILITY_MODE` (env wins, repo convention).
    Declared,
    /// Inferred: a global `wal_local_dir` is set (`PROXIMADB_WAL_LOCAL_DIR`
    /// > TOML), and an explicit local WAL root IS the `local_volume` posture.
    WalLocalDir,
    /// Inferred from the scheme of the server's primary WAL location
    /// (`write_buffer_directory`, else `data_directories[0]`): an object URL
    /// ⇒ `object_store_serverless`, a local path ⇒ `local_volume`.
    PrimaryScheme,
}

impl DurabilityModeSource {
    /// Short label for logs.
    pub fn label(&self) -> &'static str {
        match self {
            DurabilityModeSource::Declared => "declared",
            DurabilityModeSource::WalLocalDir => "wal_local_dir set",
            DurabilityModeSource::PrimaryScheme => "primary WAL location scheme",
        }
    }
}

/// The boot resolution of [`DurabilityMode`] (ADR-098 D3 / TD-SPINE-5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDurabilityMode {
    /// The effective mode.
    pub mode: DurabilityMode,
    /// How it was decided.
    pub source: DurabilityModeSource,
    /// The input value that decided it (declared spelling, the set
    /// `wal_local_dir`, or the primary WAL location URL) — log evidence.
    pub evidence: String,
    /// TD-SPINE-5 default-root move: `Some("{data_dir}/wal")` when
    /// `local_volume` was resolved with NO explicit `wal_local_dir`, so the
    /// WAL root moves off the legacy co-located-with-data location. `None`
    /// when an explicit root already applies or the mode is
    /// `object_store_serverless` (per-collection WAL stays under the
    /// catalog-derived base location). Recovery still scans the legacy
    /// co-located root, so an upgrade never strands unflushed WAL objects.
    pub default_wal_root: Option<String>,
}

impl ResolvedDurabilityMode {
    /// The WAL root in force: an explicit `wal_local_dir` when set, else the
    /// TD-SPINE-5 default when one applies. `None` ⇒ per-collection WAL under
    /// the catalog-derived base location (legacy/serverless placement).
    pub fn effective_wal_root(&self, explicit_wal_local_dir: Option<&str>) -> Option<String> {
        let explicit = explicit_wal_local_dir
            .map(|dir| dir.trim().trim_end_matches('/').to_string())
            .filter(|dir| !dir.is_empty());
        explicit.or_else(|| self.default_wal_root.clone())
    }
}

/// True when `url` names a remote (object-store) scheme: it carries a
/// `scheme://` authority that is NOT `file:`. A bare path (no `://`) is local.
/// Mirrors the classification the former ADR-069 S1 boot guard used.
pub fn is_object_store_url(url: &str) -> bool {
    url.contains("://") && !url.starts_with("file:")
}

/// Resolve the WAL durability mode (ADR-098 D3 / TD-SPINE-5). Pure.
///
/// Ladder: a **declared** mode wins; else a set global `wal_local_dir` ⇒
/// `local_volume`; else the scheme of the **primary WAL location**
/// (`write_buffer_directory`, else `data_directories[0]` — the caller folds
/// that ladder into `primary_wal_location`): an object URL ⇒
/// `object_store_serverless`, a local path ⇒ `local_volume` with the WAL root
/// defaulting to `{data_dir}/wal`. `data_dir` is `None` only for conversions
/// that cannot know it (programmatic/embedded callers); then no default-root
/// move happens — the legacy co-located placement stays — and the mode is
/// still resolved and reportable.
pub fn resolve_durability_mode(
    declared: Option<DurabilityMode>,
    wal_local_dir: Option<&str>,
    primary_wal_location: &str,
    data_dir: Option<&std::path::Path>,
) -> ResolvedDurabilityMode {
    let explicit_root = wal_local_dir
        .map(|dir| dir.trim().trim_end_matches('/').to_string())
        .filter(|dir| !dir.is_empty());
    let local_default_root = |apply: bool| {
        apply
            .then(|| data_dir.map(|dir| format!("file://{}/wal", dir.display())))
            .flatten()
    };
    if let Some(mode) = declared {
        return ResolvedDurabilityMode {
            mode,
            source: DurabilityModeSource::Declared,
            evidence: mode.to_string(),
            default_wal_root: local_default_root(
                mode == DurabilityMode::LocalVolume && explicit_root.is_none(),
            ),
        };
    }
    if let Some(root) = explicit_root.clone() {
        return ResolvedDurabilityMode {
            mode: DurabilityMode::LocalVolume,
            source: DurabilityModeSource::WalLocalDir,
            evidence: root,
            default_wal_root: None,
        };
    }
    let primary = primary_wal_location.trim();
    let mode = if is_object_store_url(primary) {
        DurabilityMode::ObjectStoreServerless
    } else {
        DurabilityMode::LocalVolume
    };
    ResolvedDurabilityMode {
        mode,
        source: DurabilityModeSource::PrimaryScheme,
        evidence: primary.to_string(),
        default_wal_root: local_default_root(mode == DurabilityMode::LocalVolume),
    }
}

/// TD-SPINE-5 env mirror of `[storage.wal_config].durability_mode` (tier
/// config-mirror; unset ⇒ infer). Read once per process. An INVALID value is
/// logged at error and treated as unset (the ADR-098 D3 no-boot-break rule:
/// existing configurations must keep booting; the error names the fix).
pub(crate) fn durability_mode_env() -> Option<DurabilityMode> {
    static ENV: std::sync::OnceLock<Option<DurabilityMode>> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        let raw = std::env::var("PROXIMADB_WAL_DURABILITY_MODE").ok()?;
        match DurabilityMode::parse(&raw) {
            Some(mode) => Some(mode),
            None => {
                tracing::error!(
                    value = %raw,
                    "PROXIMADB_WAL_DURABILITY_MODE must be 'local_volume' or \
                     'object_store_serverless'; ignoring the invalid value and inferring the \
                     durability mode (ADR-098 D3 / TD-SPINE-5)"
                );
                None
            }
        }
    })
}

/// Backwards-compat alias for [`WalEncryptionConfig`].
pub type EncryptionConfig = WalEncryptionConfig;

/// Encryption configuration for WAL (TD-016)
#[derive(Debug, Clone)]
pub struct WalEncryptionConfig {
    /// Enable encryption for WAL segments
    pub enabled: bool,

    /// Master key environment variable name
    pub master_key_env_var: String,

    /// Key rotation interval in seconds (default: 30 days)
    pub key_rotation_interval_secs: u64,

    /// Chunk size for encryption (default: 4KB)
    pub chunk_size: usize,
}

impl Default for WalEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for backward compatibility
            master_key_env_var: "PROXIMADB_CRYPTO_MASTER_KEY".to_string(),
            key_rotation_interval_secs: 30 * 24 * 3600, // 30 days
            chunk_size: 4096,                           // 4KB
        }
    }
}

/// Backwards-compat alias for [`WalCompressionConfig`].
pub type CompressionConfig = WalCompressionConfig;

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalCompressionConfig {
    /// Algorithm to use
    pub algorithm: crate::core::CompressionAlgorithm,

    /// Enable compression for memory structures
    pub compress_memory: bool,

    /// Enable compression for disk storage
    pub compress_disk: bool,

    /// Minimum entry size to compress (bytes)
    pub min_compress_size: usize,
}

impl Default for WalCompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::default(),
            compress_memory: false, // Keep memory uncompressed for fast metadata filtering
            compress_disk: true,    // Compress disk for space efficiency with large vectors
            min_compress_size: 1024, // Compress larger entries (vectors) for better disk IOPS
        }
    }
}

/// Backwards-compat alias for [`WalPerformanceConfig`].
pub type PerformanceConfig = WalPerformanceConfig;

/// Performance configuration with smart defaults - size-based flush only
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalPerformanceConfig {
    /// Memory table flush threshold (bytes) - ONLY size-based trigger
    pub memory_flush_size_bytes: usize,

    /// Disk segment size (bytes) for each collection
    pub disk_segment_size: usize,

    /// Global WAL size threshold for forced flush (bytes)
    pub global_flush_threshold: usize,

    /// Write buffer size for disk operations
    pub write_buffer_size: usize,

    /// Number of concurrent flush operations
    pub concurrent_flushes: usize,

    /// MVCC cleanup interval (seconds)
    pub mvcc_cleanup_interval_secs: u64,

    /// TTL cleanup interval (seconds)
    pub ttl_cleanup_interval_secs: u64,

    /// Sync mode for disk writes
    pub sync_mode: SyncMode,

    /// Sync interval for periodic sync mode (seconds)
    pub sync_interval_seconds: u64,

    /// Shrink factor for global threshold management (percentage)
    /// When global threshold is exceeded, flush collections until memory usage drops to this percentage
    pub global_shrink_factor: f64,

    /// Cloud backup configuration
    pub cloud_backup: Option<CloudBackupConfig>,

    /// Enable optimized WAL writer for high-performance writes
    pub enable_optimized_write_buffer_writer: Option<bool>,

    /// Number of background writer threads for optimized WAL writer
    pub background_writer_threads: Option<usize>,

    /// Batch size for optimized WAL writer
    pub write_buffer_batch_size: Option<usize>,

    /// ADR-069 D2 — time-based memtable→SST flush interval (seconds); 0 = disabled.
    /// The RPO floor: bounds worst-case loss on volume loss to this interval.
    pub flush_interval_secs: u64,

    /// TD-FLUSH-3 S1 — minimum PREDICTED segment bytes (MB) before the size
    /// trigger may flush; 0 = floor disabled. Predicted = Σ_records
    /// (dim × 9/8 + 8) — the RaBitQ+SQ8 bytes the segment will occupy (the
    /// f32 tier, when tagged on, makes this an under-prediction → segments
    /// only get LARGER, which is the safe direction). Time (RPO) and capacity
    /// triggers override the floor.
    pub flush_floor_predicted_mb: u64,

    /// ADR-069 D6 — per-collection WAL size budget (bytes) for the capacity flush +
    /// backpressure watermarks; 0 = disabled.
    pub wal_max_bytes: usize,

    /// ADR-069 D3 — fraction of `wal_max_bytes` at which to force a flush + truncation.
    pub high_watermark_pct: f64,

    /// ADR-069 D3 — fraction of `wal_max_bytes` at which to apply write backpressure.
    pub critical_watermark_pct: f64,
}

impl Default for WalPerformanceConfig {
    fn default() -> Self {
        Self {
            // Optimized for write-triggered size-based flush only
            memory_flush_size_bytes: 2 * 1024 * 1024, // 2MB memory limit - reduced for faster recovery as per CLAUDE.md
            disk_segment_size: 512 * 1024 * 1024,     // 512MB segments optimized for large vectors
            global_flush_threshold: 4 * 1024 * 1024 * 1024, // 4GB global limit - recommended for global memory threshold
            write_buffer_size: 8 * 1024 * 1024, // 8MB write buffer for large vector throughput
            concurrent_flushes: num_cpus::get().min(4), // Max 4 concurrent flushes to avoid I/O contention
            mvcc_cleanup_interval_secs: 3600,           // Clean up old versions every hour
            ttl_cleanup_interval_secs: 300,             // Check TTL every 5 minutes
            sync_mode: SyncMode::PerBatch, // Balance safety and bulk insert performance
            sync_interval_seconds: 60,     // Default to 60 seconds for periodic sync
            global_shrink_factor: 0.4, // 40% shrink factor - recommended for global threshold management
            cloud_backup: None,        // Cloud backup disabled by default
            enable_optimized_write_buffer_writer: Some(false), // Disabled by default for gradual rollout
            background_writer_threads: None, // Will use 2 by default in optimized writer
            write_buffer_batch_size: None,   // Will use 100 by default in optimized writer
            // ADR-069 D2: time floor defaults to 300s (RPO safety net). The live server
            // also derives this from WriteBufferUserConfig (toml) via to_engine_config();
            // kept in sync here so programmatic WALConfig::default()/tests agree.
            // wal_max_bytes stays 0 → S4 write-shedding admission control stays OFF.
            flush_interval_secs: 300,
            flush_floor_predicted_mb: 128,
            wal_max_bytes: 0,
            high_watermark_pct: 0.80,
            critical_watermark_pct: 0.95,
        }
    }
}

/// Disk sync mode for durability vs performance trade-off
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncMode {
    /// Never sync (fastest, least durable)
    Never,
    /// Sync after each write (slowest, most durable)
    Always,
    /// Sync periodically (good balance)
    Periodic,
    /// Sync after each batch (good for batch workloads)
    PerBatch,
    /// Memory-only durability (no disk WAL, flush from memory to storage)
    MemoryOnly,
}

/// Durability level for WAL writes (more granular than SyncMode)
#[derive(Debug, Clone)]
pub enum DurabilityLevel {
    /// No sync - fastest, but risk of data loss (development only)
    NoSync,

    /// Sync metadata only (fdatasync) - good balance
    /// Data is written but file metadata (like timestamps) may not be
    SyncData,

    /// Full sync (fsync) - safest but slowest
    /// Both data and metadata are synced to disk
    SyncFull,

    /// Batch sync - sync every N writes or T seconds
    /// Provides configurable balance between durability and performance
    BatchSync {
        /// Number of writes before sync
        batch_size: usize,
        /// Time interval between syncs (seconds)
        interval_secs: u64,
    },
}

/// WAL strategy type selection
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum WriteBufferStrategyType {
    /// Modern Avro batch strategy with zero-copy optimization
    AvroBatch,
    /// Modern Bincode batch strategy with optimal Rust performance
    #[default]
    BincodeBatch,
    /// Modern Proto batch strategy for proto-first architecture
    ProtoBatch,
}

impl std::fmt::Display for WriteBufferStrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteBufferStrategyType::AvroBatch => write!(f, "AvroBatch"),
            WriteBufferStrategyType::BincodeBatch => write!(f, "BincodeBatch"),
            WriteBufferStrategyType::ProtoBatch => write!(f, "ProtoBatch"),
        }
    }
}

/// Multi-disk configuration for WAL distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDiskConfig {
    /// WAL directory URLs supporting multiple filesystem types
    /// Examples:
    /// - file:///path/to/wal1, file:///path/to/wal2 (local multi-disk)
    /// - s3://bucket1/wal, s3://bucket2/wal (S3 multi-bucket)
    /// - adls://account.dfs.core.windows.net/container1/wal (Azure)
    /// - gcs://bucket/wal (Google Cloud)
    pub data_directories: Vec<String>,

    /// Distribution strategy
    pub distribution_strategy: DiskDistributionStrategy,

    /// Enable collection affinity (collection stays on one disk)
    pub collection_affinity: bool,
}

/// Strategy for distributing collections across disks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiskDistributionStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Hash-based distribution (consistent)
    Hash,
    /// Load-balanced distribution (dynamic)
    LoadBalanced,
}

impl Default for MultiDiskConfig {
    fn default() -> Self {
        Self {
            data_directories: vec!["file://./data/wal".to_string()],
            distribution_strategy: DiskDistributionStrategy::LoadBalanced, // Optimal for bulk inserts
            collection_affinity: true, // Keep collection on one disk for sequential I/O
        }
    }
}

/// Memtable configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemTableConfig {
    /// Memtable strategy type
    pub memtable_type: MemTableType,

    /// Global memory limit across all collections (size-based only)
    pub global_memory_limit: usize,

    /// MVCC version retention count
    pub mvcc_versions_retained: usize,

    /// Enable concurrent operations
    pub enable_concurrency: bool,
}

/// Memtable strategy type selection
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum MemTableType {
    /// Skip List - High write throughput, ordered data (RocksDB/LevelDB default)
    SkipList,
    /// B+ Tree - Stable inserts/queries, general use, memory efficient
    BTree,
    /// ART - Concurrent Adaptive Radix Tree, high performance for range queries
    #[default]
    Art,
    /// Hash Map - Write-heavy, unordered ingestion, point lookups only
    HashMap,
}

impl Default for MemTableConfig {
    fn default() -> Self {
        Self {
            memtable_type: MemTableType::default(),
            global_memory_limit: 4 * 1024 * 1024 * 1024, // 4GB for write-triggered flush
            mvcc_versions_retained: 3,                   // Keep last 3 versions for MVCC
            enable_concurrency: true,                    // Enable concurrent operations
        }
    }
}

/// Comprehensive WAL configuration
#[derive(Debug, Clone)]
pub struct WALConfig {
    /// Strategy type to use
    pub strategy_type: WriteBufferStrategyType,

    /// Memtable configuration
    pub memtable: MemTableConfig,

    /// Multi-disk configuration
    pub multi_disk: MultiDiskConfig,

    /// Global manifest location (explicit configuration)
    /// If None, defaults to {first_data_directory}/wal
    /// Examples:
    /// - "file:///data/wal-metadata" (dedicated fast SSD)
    /// - "file:///shared/nfs/wal" (shared NFS for HA)
    /// - "s3://bucket/wal-global" (cloud-based for distributed deployments)
    pub global_manifest_url: Option<String>,

    /// ADR-069 S1: opt-in local WAL root (e.g. `file:///waldisk`). When set, the
    /// per-collection WAL is written under `{dir}/{collection_id}/wal/` on this
    /// local reattachable disk, DECOUPLED from the collection's object-store data
    /// assignment — WAL is high-frequency small appends (each an object-store
    /// PUT), so a local disk avoids that cost while data stays on object store.
    /// TOML `[storage.wal_config].wal_local_dir`; overridden by env
    /// `PROXIMADB_WAL_LOCAL_DIR`. None ⇒ legacy WAL-co-located-with-data.
    /// The location resolves through the FilesystemFactory scheme, so a future
    /// value could be `s3://…` (S3 Express) or another backend.
    pub wal_local_dir: Option<String>,

    /// ADR-098 D3 / TD-SPINE-5: the RESOLVED WAL durability mode, carried from
    /// the boot resolution (`WriteBufferUserConfig::resolve_durability_mode`)
    /// so engine-side surfaces (recovery, observability) can report it. `None`
    /// on programmatically built configs that never went through the boot
    /// resolution — then no mode claim is made (legacy behavior).
    pub durability_mode: Option<DurabilityMode>,

    /// Compression settings
    pub compression: WalCompressionConfig,

    /// Encryption settings (TD-016)
    pub encryption: WalEncryptionConfig,

    /// Performance tuning
    pub performance: WalPerformanceConfig,

    /// Enable MVCC versioning
    pub enable_mvcc: bool,

    /// Enable TTL support
    pub enable_ttl: bool,

    /// Enable background compaction
    pub enable_background_compaction: bool,

    /// Collection-specific overrides
    pub collection_overrides: std::collections::HashMap<String, CollectionWalConfig>,

    /// Enable optimized WAL writer (feature flag)
    pub enable_optimized_writer: bool,

    /// Optimized writer batch size
    pub optimized_writer_batch_size: Option<usize>,

    /// Optimized writer batch timeout in milliseconds
    pub optimized_writer_batch_timeout_ms: Option<u64>,

    /// Number of writer threads for optimized writer
    pub optimized_writer_threads: Option<usize>,

    /// Enable write combining for same collection
    pub optimized_writer_enable_combining: Option<bool>,
}

impl Default for WALConfig {
    fn default() -> Self {
        Self {
            strategy_type: WriteBufferStrategyType::default(), // Bincode for maximum vector ingestion performance
            memtable: MemTableConfig::default(), // ART for metadata filtering efficiency
            multi_disk: MultiDiskConfig::default(), // LoadBalanced for bulk insert optimization
            wal_local_dir: None,                 // ADR-069 S1: opt-in; None = legacy WAL-with-data
            durability_mode: None,               // TD-SPINE-5: set by the boot resolution
            compression: WalCompressionConfig::default(), // Snappy for balanced performance
            encryption: WalEncryptionConfig::default(), // Encryption disabled by default (TD-016)
            performance: WalPerformanceConfig::default(), // Optimized for large vectors and bulk processing
            enable_mvcc: true, // Enable for consistency and document versioning
            enable_ttl: true,  // Enable for data lifecycle management
            enable_background_compaction: true, // Enable for maintenance and space reclamation
            collection_overrides: std::collections::HashMap::new(),
            global_manifest_url: None, // Defaults to {first_data_directory}/wal
            enable_optimized_writer: false, // Disabled by default for gradual rollout
            optimized_writer_batch_size: None,
            optimized_writer_batch_timeout_ms: None,
            optimized_writer_threads: None,
            optimized_writer_enable_combining: None,
        }
    }
}

/// Collection-specific WAL configuration overrides
#[derive(Debug, Clone)]
pub struct CollectionWalConfig {
    /// Override memory flush size threshold for this collection (bytes)
    pub memory_flush_size_bytes: Option<usize>,

    /// Override disk segment size for this collection
    pub disk_segment_size: Option<usize>,

    /// Override compression settings
    pub compression: Option<WalCompressionConfig>,

    /// Override TTL settings
    pub default_ttl_days: Option<u32>,
}

impl WALConfig {
    /// Create configuration optimized for high-throughput writes
    pub fn high_throughput() -> Self {
        let mut config = Self {
            strategy_type: WriteBufferStrategyType::BincodeBatch, // Faster serialization
            ..Default::default()
        };
        config.memtable.memtable_type = MemTableType::HashMap; // Fastest writes for unordered data
        config.compression.algorithm = CompressionAlgorithm::Lz4; // Faster compression
        config.performance.memory_flush_size_bytes = 256 * 1024 * 1024; // 256MB
        config.performance.sync_mode = SyncMode::PerBatch; // Less frequent syncing
        config
    }

    /// Create configuration optimized for low-latency reads
    pub fn low_latency() -> Self {
        let mut config = Self::default();
        config.memtable.memtable_type = MemTableType::HashMap; // Fastest point lookups
        config.compression.compress_memory = false; // Faster memory access
        config.compression.compress_disk = false; // Faster disk reads
        config.performance.memory_flush_size_bytes = 32 * 1024 * 1024; // 32MB smaller memory footprint
        config.performance.sync_mode = SyncMode::Always; // Immediate consistency
        config
    }

    /// Create configuration optimized for storage efficiency
    pub fn storage_optimized() -> Self {
        let mut config = Self::default();
        config.memtable.memtable_type = MemTableType::BTree; // Most memory-efficient
        config.compression.algorithm = CompressionAlgorithm::Zstd; // Better compression
        config.compression.compress_memory = true; // Compress everything
        config.compression.min_compress_size = 64; // Compress smaller entries
        config.performance.disk_segment_size = 512 * 1024 * 1024; // Larger segments
        config
    }

    /// Create configuration optimized for range queries and analytics
    pub fn range_query_optimized() -> Self {
        let mut config = Self::default();
        config.memtable.memtable_type = MemTableType::BTree; // Excellent range scan performance
        config.strategy_type = WriteBufferStrategyType::AvroBatch; // Schema evolution for analytics
        config.compression.algorithm = CompressionAlgorithm::Snappy; // Balanced compression
        config.performance.memory_flush_size_bytes = 64 * 1024 * 1024; // 64MB moderate memory usage
        config
    }

    /// Create configuration optimized for high concurrency and string keys
    pub fn high_concurrency() -> Self {
        let mut config = Self::default();
        config.memtable.memtable_type = MemTableType::Art; // Excellent concurrency
        config.strategy_type = WriteBufferStrategyType::BincodeBatch; // Fast serialization
        config.compression.algorithm = CompressionAlgorithm::Lz4; // Fast compression
        config.memtable.enable_concurrency = true;
        config
    }

    /// Get effective configuration for a collection (with overrides)
    pub fn effective_config_for_collection(
        &self,
        collection_id: &str,
    ) -> CollectionEffectiveConfig {
        let overrides = self.collection_overrides.get(collection_id);

        CollectionEffectiveConfig {
            memory_flush_size_bytes: overrides
                .and_then(|o| o.memory_flush_size_bytes)
                .unwrap_or(self.performance.memory_flush_size_bytes),
            disk_segment_size: overrides
                .and_then(|o| o.disk_segment_size)
                .unwrap_or(self.performance.disk_segment_size),
            compression: overrides
                .and_then(|o| o.compression.clone())
                .unwrap_or_else(|| self.compression.clone()),
            default_ttl_days: overrides.and_then(|o| o.default_ttl_days),
        }
    }
}

/// Effective configuration for a specific collection
#[derive(Debug, Clone)]
pub struct CollectionEffectiveConfig {
    pub disk_segment_size: usize,
    pub compression: WalCompressionConfig,
    pub default_ttl_days: Option<u32>,
    /// Size-based flush threshold (bytes) - derived from memory_flush_size_bytes
    pub memory_flush_size_bytes: usize,
}

/// Cloud backup configuration for WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudBackupConfig {
    /// Enable cloud backup for WAL batches
    pub enabled: bool,

    /// Cloud storage URL for backup (e.g., s3://bucket/wal/, gcs://bucket/wal/)
    pub backup_url: String,

    /// Backup strategy

    /// Backup frequency configuration
    pub frequency: BackupFrequency,

    /// Automatic cleanup of old cloud backups
    pub cleanup_policy: Option<CloudCleanupPolicy>,

    /// Verify backup integrity
    pub verify_integrity: bool,

    /// Retry configuration for cloud operations
    pub retry_config: CloudRetryConfig,
}

impl Default for CloudBackupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backup_url: "file://./data/wal_backup".to_string(),
            // strategy removed -  CloudBackupStrategy::OnFlush,
            frequency: BackupFrequency::default(),
            cleanup_policy: Some(CloudCleanupPolicy::default()),
            verify_integrity: true,
            retry_config: CloudRetryConfig::default(),
        }
    }
}

/// Cloud backup strategy
#[derive(Debug, Clone, Default)]
pub enum CloudBackupStrategy {
    /// Real-time backup on every write
    RealTime,
    /// Periodic batch backup
    Periodic { interval_secs: u64 },
    /// Backup on flush events
    #[default]
    OnFlush,
    /// Backup on demand only
    OnDemand,
}

/// Backup frequency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupFrequency {
    /// Backup every N operations
    pub operations_threshold: Option<u64>,
    /// Backup every N seconds
    pub time_threshold_secs: Option<u64>,
    /// Backup when size exceeds threshold
    pub size_threshold_bytes: Option<usize>,
}

impl Default for BackupFrequency {
    fn default() -> Self {
        Self {
            operations_threshold: Some(1000),
            time_threshold_secs: Some(300),                // 5 minutes
            size_threshold_bytes: Some(100 * 1024 * 1024), // 100MB
        }
    }
}

/// Cloud cleanup policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudCleanupPolicy {
    /// Retain backups for N days
    pub retention_days: u32,
    /// Maximum number of backups to keep
    pub max_backups: Option<u32>,
    /// Cleanup frequency in hours
    pub cleanup_frequency_hours: u32,
}

impl Default for CloudCleanupPolicy {
    fn default() -> Self {
        Self {
            retention_days: 7,
            max_backups: Some(100),
            cleanup_frequency_hours: 24,
        }
    }
}

/// Cloud retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudRetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for CloudRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

// --- Tests inlined from tests/unit/config/test_flush_config.rs ---

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_default_flush_configuration() {
        let config = WALConfig::default();

        // Test default performance settings
        let perf = config.performance;
        assert!(
            perf.memory_flush_size_bytes > 0,
            "Should have positive memory flush size"
        );
        assert!(
            perf.disk_segment_size > 0,
            "Should have positive disk segment size"
        );

        // Test default memtable settings
        let memtable = config.memtable;
        assert!(
            memtable.global_memory_limit > 0,
            "Should have positive global memory limit"
        );
    }

    #[test]
    fn test_collection_specific_overrides() {
        // Create collection-specific configurations
        let mut collection_configs = HashMap::new();

        // Large collection needs higher threshold
        collection_configs.insert(
            "embeddings".to_string(),
            CollectionWalConfig {
                memory_flush_size_bytes: Some(50 * 1024 * 1024), // 50MB
                disk_segment_size: Some(1024 * 1024 * 1024),     // 1GB
                compression: None,
                default_ttl_days: Some(30),
            },
        );

        // Small collection can use lower threshold
        collection_configs.insert(
            "metadata".to_string(),
            CollectionWalConfig {
                memory_flush_size_bytes: Some(5 * 1024 * 1024), // 5MB
                disk_segment_size: Some(100 * 1024 * 1024),     // 100MB
                compression: None,
                default_ttl_days: Some(7),
            },
        );

        // Verify overrides
        let embeddings_config = collection_configs.get("embeddings").unwrap();
        assert_eq!(
            embeddings_config.memory_flush_size_bytes,
            Some(50 * 1024 * 1024)
        );
        assert_eq!(embeddings_config.default_ttl_days, Some(30));

        let metadata_config = collection_configs.get("metadata").unwrap();
        assert_eq!(
            metadata_config.memory_flush_size_bytes,
            Some(5 * 1024 * 1024)
        );
        assert_eq!(metadata_config.default_ttl_days, Some(7));
    }

    #[test]
    fn test_performance_config_limits() {
        let mut perf_config = WalPerformanceConfig::default();

        // Test setting custom limits
        perf_config.memory_flush_size_bytes = 1000 * 1024 * 1024; // 1000MB
        perf_config.disk_segment_size = 2048 * 1024 * 1024; // 2048MB

        assert_eq!(perf_config.memory_flush_size_bytes, 1000 * 1024 * 1024);
        assert_eq!(perf_config.disk_segment_size, 2048 * 1024 * 1024);
    }

    #[test]
    fn test_memtable_config() {
        let mut memtable_config = MemTableConfig::default();

        // Test setting memtable parameters
        memtable_config.global_memory_limit = 4096 * 1024 * 1024; // 4GB
        memtable_config.mvcc_versions_retained = 10;

        assert_eq!(memtable_config.global_memory_limit, 4096 * 1024 * 1024);
        assert_eq!(memtable_config.mvcc_versions_retained, 10);
    }

    #[test]
    fn test_effective_config_resolution() {
        // Test how collection-specific configs override defaults
        let default_config = CollectionWalConfig {
            memory_flush_size_bytes: Some(10 * 1024 * 1024), // 10MB default
            disk_segment_size: Some(256 * 1024 * 1024),      // 256MB default
            compression: None,
            default_ttl_days: None,
        };

        let override_config = CollectionWalConfig {
            memory_flush_size_bytes: Some(20 * 1024 * 1024), // Override to 20MB
            disk_segment_size: None,                         // Keep default
            compression: None,
            default_ttl_days: Some(14), // Add TTL
        };

        // Simulate resolving effective config
        let effective_memory = override_config
            .memory_flush_size_bytes
            .or(default_config.memory_flush_size_bytes)
            .unwrap();
        let effective_disk = override_config
            .disk_segment_size
            .or(default_config.disk_segment_size)
            .unwrap();
        let effective_ttl = override_config
            .default_ttl_days
            .or(default_config.default_ttl_days);

        assert_eq!(effective_memory, 20 * 1024 * 1024, "Should use override");
        assert_eq!(effective_disk, 256 * 1024 * 1024, "Should use default");
        assert_eq!(effective_ttl, Some(14), "Should use override");
    }

    #[tokio::test]
    async fn test_wal_strategy_type_variants() {
        let avro_strategy = WriteBufferStrategyType::AvroBatch;
        let bincode_strategy = WriteBufferStrategyType::BincodeBatch;

        assert_eq!(format!("{:?}", avro_strategy), "AvroBatch");
        assert_eq!(format!("{:?}", bincode_strategy), "BincodeBatch");

        let cloned_avro = avro_strategy;
        assert!(matches!(avro_strategy, WriteBufferStrategyType::AvroBatch));
        assert!(matches!(cloned_avro, WriteBufferStrategyType::AvroBatch));
    }

    #[tokio::test]
    async fn test_memtable_type_variants() {
        let btree_type = MemTableType::BTree;
        let hashmap_type = MemTableType::HashMap;
        let skiplist_type = MemTableType::SkipList;
        let art_type = MemTableType::Art;

        assert_eq!(format!("{:?}", btree_type), "BTree");
        assert_eq!(format!("{:?}", hashmap_type), "HashMap");
        assert_eq!(format!("{:?}", skiplist_type), "SkipList");
        assert_eq!(format!("{:?}", art_type), "Art");
    }

    #[tokio::test]
    async fn test_compression_config_default() {
        let config = WalCompressionConfig::default();

        // CompressionAlgorithm::default() returns None (no compression by default)
        assert!(matches!(config.algorithm, CompressionAlgorithm::None));
        assert!(!config.compress_memory);
        assert!(config.compress_disk);
        assert_eq!(config.min_compress_size, 1024);
    }

    #[tokio::test]
    async fn test_performance_config_default() {
        let config = WalPerformanceConfig::default();

        assert_eq!(config.memory_flush_size_bytes, 2 * 1024 * 1024); // Updated to 2MB for faster recovery
        assert_eq!(config.disk_segment_size, 512 * 1024 * 1024);
        assert_eq!(config.write_buffer_size, 8 * 1024 * 1024);
        assert!(matches!(
            config.sync_mode,
            crate::storage::persistence::write_ahead_log::config::SyncMode::PerBatch
        ));
    }

    #[tokio::test]
    async fn test_wal_config_default() {
        let config = WALConfig::default();

        assert!(matches!(
            config.strategy_type,
            WriteBufferStrategyType::BincodeBatch
        ));
        assert!(matches!(config.memtable.memtable_type, MemTableType::Art));
        assert!(matches!(
            config.multi_disk.distribution_strategy,
            DiskDistributionStrategy::LoadBalanced
        ));
        assert!(!&config.compression.compress_memory);
        assert!(&config.compression.compress_disk);
    }

    #[test]
    fn wal_performance_defaults_arm_time_trigger() {
        // ADR-069 D2: the 300s time floor is armed by default (RPO safety net) so the
        // auto-flush driver spawns and segments materialize while the server runs.
        // wal_max_bytes stays 0 → S4 capacity/backpressure trigger stays OFF.
        let p = WalPerformanceConfig::default();
        assert_eq!(p.flush_interval_secs, 300, "time floor armed by default");
        assert_eq!(p.wal_max_bytes, 0, "capacity budget off by default");
        assert_eq!(p.high_watermark_pct, 0.80);
        assert_eq!(p.critical_watermark_pct, 0.95);
    }

    #[tokio::test]
    async fn test_wal_config_custom() {
        let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let config = WALConfig {
            strategy_type: WriteBufferStrategyType::BincodeBatch,
            multi_disk: MultiDiskConfig {
                data_directories: vec![temp_dir.path().to_string_lossy().to_string()],
                distribution_strategy: DiskDistributionStrategy::Hash,
                collection_affinity: true,
            },
            global_manifest_url: None,
            wal_local_dir: None,
            durability_mode: None,
            memtable: MemTableConfig {
                memtable_type: MemTableType::SkipList,
                global_memory_limit: 256 * 1024 * 1024,
                mvcc_versions_retained: 5,
                enable_concurrency: true,
            },
            compression: WalCompressionConfig {
                algorithm: CompressionAlgorithm::Zstd,
                compress_memory: true,
                compress_disk: true,
                min_compress_size: 2048,
            },
            encryption: WalEncryptionConfig::default(),
            performance: WalPerformanceConfig {
                memory_flush_size_bytes: 128 * 1024 * 1024,
                disk_segment_size: 512 * 1024 * 1024,
                global_flush_threshold: 1024 * 1024 * 1024,
                write_buffer_size: 16384,
                concurrent_flushes: 2,
                mvcc_cleanup_interval_secs: 1800,
                cloud_backup: Default::default(),
                global_shrink_factor: 0.8,
                ttl_cleanup_interval_secs: 600,
                sync_mode: crate::storage::persistence::write_ahead_log::config::SyncMode::Always,
                sync_interval_seconds: 1,
                enable_optimized_write_buffer_writer: None,
                background_writer_threads: None,
                write_buffer_batch_size: None,
                flush_interval_secs: 0,
                flush_floor_predicted_mb: 0,
                wal_max_bytes: 0,
                high_watermark_pct: 0.80,
                critical_watermark_pct: 0.95,
            },
            enable_mvcc: true,
            enable_ttl: true,
            enable_background_compaction: true,
            collection_overrides: std::collections::HashMap::new(),
            enable_optimized_writer: false,
            optimized_writer_batch_size: None,
            optimized_writer_batch_timeout_ms: None,
            optimized_writer_threads: None,
            optimized_writer_enable_combining: None,
        };

        assert!(matches!(
            config.strategy_type,
            WriteBufferStrategyType::BincodeBatch
        ));
        assert!(matches!(
            config.memtable.memtable_type,
            MemTableType::SkipList
        ));
        assert!(matches!(
            config.multi_disk.distribution_strategy,
            DiskDistributionStrategy::Hash
        ));
        assert!(matches!(
            config.compression.algorithm,
            CompressionAlgorithm::Zstd
        ));
        assert!(&config.compression.compress_memory);
        assert_eq!(
            config.performance.memory_flush_size_bytes,
            128 * 1024 * 1024
        );
        assert!(matches!(
            config.performance.sync_mode,
            crate::storage::persistence::write_ahead_log::config::SyncMode::Always
        ));
    }

    #[tokio::test]
    async fn test_wal_config_debug_formatting() {
        let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");

        let config = WALConfig {
            strategy_type: WriteBufferStrategyType::AvroBatch,
            multi_disk: MultiDiskConfig {
                data_directories: vec![temp_dir.path().to_string_lossy().to_string()],
                distribution_strategy: DiskDistributionStrategy::LoadBalanced,
                collection_affinity: true,
            },
            global_manifest_url: None,
            wal_local_dir: None,
            durability_mode: None,
            memtable: MemTableConfig {
                memtable_type: MemTableType::Art,
                global_memory_limit: 512 * 1024 * 1024,
                mvcc_versions_retained: 3,
                enable_concurrency: true,
            },
            compression: WalCompressionConfig::default(),
            encryption: WalEncryptionConfig::default(),
            performance: WalPerformanceConfig::default(),
            enable_mvcc: true,
            enable_ttl: true,
            enable_background_compaction: true,
            collection_overrides: std::collections::HashMap::new(),
            enable_optimized_writer: false,
            optimized_writer_batch_size: None,
            optimized_writer_batch_timeout_ms: None,
            optimized_writer_threads: None,
            optimized_writer_enable_combining: None,
        };

        // Test debug formatting instead of serialization since WALConfig doesn't implement Serialize
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AvroBatch"));
        assert!(debug_str.contains("Art"));
        assert!(debug_str.contains("LoadBalanced"));
    }

    // Unit tests for flush and compaction threshold triggers (from threshold_triggers_test.rs)
    #[test]
    fn test_memory_flush_threshold_configuration() {
        // Test different memory flush threshold configurations
        let mut config = WalPerformanceConfig::default();

        // Default should be 2MB
        assert_eq!(config.memory_flush_size_bytes, 2 * 1024 * 1024);

        // Test setting custom threshold
        config.memory_flush_size_bytes = 1024 * 1024; // 1MB threshold
        assert_eq!(config.memory_flush_size_bytes, 1024 * 1024);

        // Test that global threshold is larger than memory threshold
        assert!(config.global_flush_threshold >= config.memory_flush_size_bytes);
    }

    #[test]
    fn test_global_flush_threshold_configuration() {
        let mut config = WalPerformanceConfig::default();

        // Default should be 4GB
        assert_eq!(config.global_flush_threshold, 4 * 1024 * 1024 * 1024);

        // Test setting custom global threshold
        config.global_flush_threshold = 2 * 1024 * 1024 * 1024; // 2GB threshold
        assert_eq!(config.global_flush_threshold, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_compaction_threshold_configuration() {
        // This tests SstConfig, not WalPerformanceConfig
        let mut config = crate::core::config::SstConfig::default();

        // Default should be 5 SSTables (as per SstConfig::default())
        assert_eq!(config.compaction_threshold, 5);

        // Test setting custom threshold
        config.compaction_threshold = 2; // Trigger compaction when level has 2+ SSTables
        assert_eq!(config.compaction_threshold, 2);
    }

    #[test]
    fn test_memory_threshold_logic() {
        let config = WalPerformanceConfig::default();
        let threshold = config.memory_flush_size_bytes;

        // Test different memory usage scenarios
        struct TestCase {
            name: &'static str,
            memory_usage: usize,
            should_trigger: bool,
        }

        let test_cases = vec![
            TestCase {
                name: "Below threshold",
                memory_usage: threshold / 2,
                should_trigger: false,
            },
            TestCase {
                name: "At threshold",
                memory_usage: threshold,
                should_trigger: true,
            },
            TestCase {
                name: "Above threshold",
                memory_usage: threshold + 1000,
                should_trigger: true,
            },
            TestCase {
                name: "Way above threshold",
                memory_usage: threshold * 5,
                should_trigger: true,
            },
        ];

        for test_case in test_cases {
            let should_trigger = test_case.memory_usage >= threshold;
            assert_eq!(
                should_trigger, test_case.should_trigger,
                "Test case '{}': Expected trigger mismatch. Usage: {}, Threshold: {}",
                test_case.name, test_case.memory_usage, threshold
            );
        }
    }

    #[test]
    fn test_compaction_threshold_logic() {
        let config = crate::core::config::SstConfig::default();
        let threshold = config.compaction_threshold;

        // Test different SSTable count scenarios
        struct TestCase {
            name: &'static str,
            sstable_count: u32,
            should_trigger: bool,
        }

        let test_cases = vec![
            TestCase {
                name: "Below threshold",
                sstable_count: threshold - 1,
                should_trigger: false,
            },
            TestCase {
                name: "At threshold",
                sstable_count: threshold,
                should_trigger: true,
            },
            TestCase {
                name: "Above threshold",
                sstable_count: threshold + 1,
                should_trigger: true,
            },
            TestCase {
                name: "Way above threshold",
                sstable_count: threshold * 2,
                should_trigger: true,
            },
        ];

        for test_case in test_cases {
            let should_trigger = test_case.sstable_count >= threshold;
            assert_eq!(
                should_trigger, test_case.should_trigger,
                "Test case '{}': Expected trigger mismatch. Count: {}, Threshold: {}",
                test_case.name, test_case.sstable_count, threshold
            );
        }
    }

    #[test]
    fn test_global_flush_threshold_logic() {
        let config = WalPerformanceConfig::default();
        let threshold = config.global_flush_threshold;

        // Test scenarios that would trigger global flush
        let test_cases = vec![
            (threshold / 2, false),   // Below threshold
            (threshold, true),        // At threshold
            (threshold + 1000, true), // Above threshold
        ];

        for (memory_usage, should_trigger) in test_cases {
            let triggers = memory_usage >= threshold;
            assert_eq!(
                triggers, should_trigger,
                "Global flush logic failed for usage: {}, threshold: {}",
                memory_usage, threshold
            );
        }
    }

    // Mock counter to track trigger invocations
    #[derive(Debug)]
    struct TriggerCounter {
        count: std::sync::atomic::AtomicUsize,
    }

    impl TriggerCounter {
        fn new() -> Self {
            Self {
                count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn increment(&self) {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        fn get_count(&self) -> usize {
            self.count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[test]
    fn test_multiple_flush_triggers() {
        use std::sync::Arc;

        let counter = Arc::new(TriggerCounter::new());
        let config = WalPerformanceConfig::default();

        // Simulate multiple memory threshold breaches
        let test_memory_usages = vec![
            config.memory_flush_size_bytes + 1000, // First breach
            config.memory_flush_size_bytes + 2000, // Second breach
            config.memory_flush_size_bytes + 3000, // Third breach
        ];

        for memory_usage in test_memory_usages {
            if memory_usage >= config.memory_flush_size_bytes {
                counter.increment(); // Simulate flush trigger
            }
        }

        // Should have triggered 3 times
        assert_eq!(counter.get_count(), 3, "Expected 3 flush triggers");
    }

    #[test]
    fn test_multiple_compaction_triggers() {
        use std::sync::Arc;

        let counter = Arc::new(TriggerCounter::new());
        let config = crate::core::config::SstConfig::default();

        // Simulate multiple SSTable count threshold breaches
        let test_sstable_counts = vec![
            config.compaction_threshold,     // First breach (at threshold)
            config.compaction_threshold + 1, // Second breach (above threshold)
            config.compaction_threshold + 2, // Third breach (further above)
        ];

        for sstable_count in test_sstable_counts {
            if sstable_count >= config.compaction_threshold {
                counter.increment(); // Simulate compaction trigger
            }
        }

        // Should have triggered 3 times
        assert_eq!(counter.get_count(), 3, "Expected 3 compaction triggers");
    }

    // Unit tests for WAL configuration types (from write_buffer_config_simple_test.rs)
    #[test]
    fn test_wal_strategy_type_defaults() {
        // Test default WAL strategy
        let default_strategy = WriteBufferStrategyType::default();
        assert_eq!(default_strategy, WriteBufferStrategyType::BincodeBatch);

        // Test all strategy types
        let strategies = vec![
            WriteBufferStrategyType::BincodeBatch,
            WriteBufferStrategyType::AvroBatch,
        ];

        for strategy in strategies {
            // Verify serialization works
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: WriteBufferStrategyType = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_memtable_type_defaults() {
        let default_type = MemTableType::default();
        assert_eq!(default_type, MemTableType::Art);

        // Test all memtable types
        let types = vec![
            MemTableType::SkipList,
            MemTableType::BTree,
            MemTableType::Art,
            MemTableType::HashMap,
        ];

        for memtable_type in types {
            let json = serde_json::to_string(&memtable_type).unwrap();
            let deserialized: MemTableType = serde_json::from_str(&json).unwrap();
            assert_eq!(memtable_type, deserialized);
        }
    }

    #[test]
    fn test_memtable_config_defaults() {
        let config = MemTableConfig::default();

        assert_eq!(config.memtable_type, MemTableType::Art);
        assert_eq!(config.global_memory_limit, 4 * 1024 * 1024 * 1024); // 4GB
        assert_eq!(config.mvcc_versions_retained, 3);
        assert!(config.enable_concurrency);

        // Test serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MemTableConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.memtable_type, deserialized.memtable_type);
        assert_eq!(config.global_memory_limit, deserialized.global_memory_limit);
    }

    #[test]
    fn test_disk_distribution_strategy() {
        let strategies = vec![
            DiskDistributionStrategy::RoundRobin,
            DiskDistributionStrategy::Hash,
            DiskDistributionStrategy::LoadBalanced,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: DiskDistributionStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_multi_disk_config_defaults() {
        let config = MultiDiskConfig::default();

        assert_eq!(config.data_directories.len(), 1);
        assert_eq!(config.data_directories[0], "file://./data/wal");
        assert_eq!(
            config.distribution_strategy,
            DiskDistributionStrategy::LoadBalanced
        );
        assert!(config.collection_affinity);

        // Test custom configuration
        let custom_config = MultiDiskConfig {
            data_directories: vec![
                "file:///disk1/wal".to_string(),
                "file:///disk2/wal".to_string(),
                "s3://bucket/wal".to_string(),
            ],
            distribution_strategy: DiskDistributionStrategy::Hash,
            collection_affinity: false,
        };

        let json = serde_json::to_string(&custom_config).unwrap();
        let deserialized: MultiDiskConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            custom_config.data_directories,
            deserialized.data_directories
        );
        assert_eq!(
            custom_config.distribution_strategy,
            deserialized.distribution_strategy
        );
        assert_eq!(
            custom_config.collection_affinity,
            deserialized.collection_affinity
        );
    }

    #[test]
    fn test_compression_config_defaults() {
        let config = WalCompressionConfig::default();

        assert_eq!(config.algorithm, CompressionAlgorithm::default());
        assert!(!config.compress_memory); // Memory should be uncompressed for fast access
        assert!(config.compress_disk); // Disk should be compressed for space
        assert_eq!(config.min_compress_size, 1024);

        // Test custom compression configuration
        let custom_config = WalCompressionConfig {
            algorithm: CompressionAlgorithm::Lz4,
            compress_memory: true,
            compress_disk: false,
            min_compress_size: 2048,
        };

        let json = serde_json::to_string(&custom_config).unwrap();
        let deserialized: WalCompressionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(custom_config.compress_memory, deserialized.compress_memory);
        assert_eq!(
            custom_config.min_compress_size,
            deserialized.min_compress_size
        );
    }

    #[test]
    fn test_sync_mode_variants() {
        let sync_modes = vec![
            SyncMode::Never,
            SyncMode::Always,
            SyncMode::Periodic,
            SyncMode::PerBatch,
            SyncMode::MemoryOnly,
        ];

        for mode in sync_modes {
            let json = serde_json::to_string(&mode).unwrap();
            let deserialized: SyncMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, deserialized);
        }
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = WalPerformanceConfig::default();

        // Verify size-based flush defaults
        assert_eq!(config.memory_flush_size_bytes, 2 * 1024 * 1024); // 2MB
        assert_eq!(config.disk_segment_size, 512 * 1024 * 1024); // 512MB
        assert_eq!(config.global_flush_threshold, 4 * 1024 * 1024 * 1024); // 4GB
        // Note: write_ahead_log_size field doesn't exist in WalPerformanceConfig
        // TODO: Determine correct field to assert or remove this test
        assert_eq!(config.mvcc_cleanup_interval_secs, 3600); // 1 hour
        assert_eq!(config.ttl_cleanup_interval_secs, 300); // 5 minutes
        assert_eq!(config.sync_mode, SyncMode::PerBatch);
        assert_eq!(config.global_shrink_factor, 0.4); // 40%
        assert!(config.cloud_backup.is_none());

        // Verify concurrent flushes is reasonable
        assert!(config.concurrent_flushes >= 1);
        assert!(config.concurrent_flushes <= 4);
    }

    #[test]
    fn test_performance_config_custom() {
        let custom_config = WalPerformanceConfig {
            memory_flush_size_bytes: 1024 * 1024,       // 1MB
            disk_segment_size: 64 * 1024 * 1024,        // 64MB
            global_flush_threshold: 1024 * 1024 * 1024, // 1GB
            write_buffer_size: 1024 * 1024,             // 1MB (replaces write_ahead_log_size)
            concurrent_flushes: 8,
            mvcc_cleanup_interval_secs: 7200, // 2 hours
            ttl_cleanup_interval_secs: 600,   // 10 minutes
            sync_mode: SyncMode::Always,
            sync_interval_seconds: 60,
            global_shrink_factor: 0.6,
            cloud_backup: None,
            enable_optimized_write_buffer_writer: None, // corrected field name
            background_writer_threads: None,
            write_buffer_batch_size: None, // corrected field name
            flush_interval_secs: 0,
            flush_floor_predicted_mb: 0,
            wal_max_bytes: 0,
            high_watermark_pct: 0.80,
            critical_watermark_pct: 0.95,
        };

        let json = serde_json::to_string(&custom_config).unwrap();
        let deserialized: WalPerformanceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            custom_config.memory_flush_size_bytes,
            deserialized.memory_flush_size_bytes
        );
        assert_eq!(
            custom_config.concurrent_flushes,
            deserialized.concurrent_flushes
        );
        assert_eq!(
            custom_config.global_shrink_factor,
            deserialized.global_shrink_factor
        );
        assert!(deserialized.cloud_backup.is_none());
    }

    // -------------------------------------------------------------------------
    // TD-SPINE-5 (ADR-098 D3): the durability-mode resolution ladder.
    // -------------------------------------------------------------------------

    fn data_dir() -> std::path::PathBuf {
        std::path::PathBuf::from("/srv/proximadb/data")
    }

    /// Per-mode ratchet (ADR-098 D3): a local primary WAL location infers
    /// `local_volume` AND applies the `{data_dir}/wal` default root; the
    /// resolved root's scheme is local.
    #[test]
    fn spine5_local_scheme_infers_local_volume_with_default_root() {
        let resolution =
            resolve_durability_mode(None, None, "./data/write_buffer", Some(&data_dir()));
        assert_eq!(resolution.mode, DurabilityMode::LocalVolume);
        assert_eq!(resolution.source, DurabilityModeSource::PrimaryScheme);
        assert_eq!(resolution.evidence, "./data/write_buffer");
        assert_eq!(
            resolution.default_wal_root.as_deref(),
            Some("file:///srv/proximadb/data/wal"),
            "local_volume defaults the WAL root to {{data_dir}}/wal"
        );
        assert!(
            !is_object_store_url(
                resolution
                    .effective_wal_root(None)
                    .as_deref()
                    .unwrap_or_default()
            ),
            "the resolved WAL root's scheme must match the mode (ratchet)"
        );
    }

    /// Per-mode ratchet (ADR-098 D3): an object-store primary WAL location
    /// infers `object_store_serverless` — the ADR-063 posture, not a bug —
    /// and applies NO local default root (per-collection WAL stays under the
    /// catalog-derived base).
    #[test]
    fn spine5_object_scheme_infers_serverless_without_local_root() {
        for url in ["s3://bucket/wal", "az://acct/container/wal", "gcs://b/wal"] {
            let resolution = resolve_durability_mode(None, None, url, Some(&data_dir()));
            assert_eq!(
                resolution.mode,
                DurabilityMode::ObjectStoreServerless,
                "{url}"
            );
            assert_eq!(resolution.source, DurabilityModeSource::PrimaryScheme);
            assert!(
                resolution.default_wal_root.is_none(),
                "{url}: serverless keeps the catalog-derived base, not a local root"
            );
            assert!(resolution.effective_wal_root(None).is_none());
        }
    }

    /// A `file://` primary location is LOCAL (not an object URL).
    #[test]
    fn spine5_file_scheme_is_local_volume() {
        let resolution =
            resolve_durability_mode(None, None, "file:///waldisk/buffer", Some(&data_dir()));
        assert_eq!(resolution.mode, DurabilityMode::LocalVolume);
        assert!(!is_object_store_url("file:///waldisk"));
    }

    /// A declared mode wins over inference, in BOTH directions.
    #[test]
    fn spine5_declared_mode_wins_over_inference() {
        let declared = resolve_durability_mode(
            Some(DurabilityMode::ObjectStoreServerless),
            None,
            "./data/write_buffer",
            Some(&data_dir()),
        );
        assert_eq!(declared.mode, DurabilityMode::ObjectStoreServerless);
        assert_eq!(declared.source, DurabilityModeSource::Declared);
        assert!(declared.default_wal_root.is_none());

        let declared_local = resolve_durability_mode(
            Some(DurabilityMode::LocalVolume),
            None,
            "s3://bucket/wal",
            Some(&data_dir()),
        );
        assert_eq!(declared_local.mode, DurabilityMode::LocalVolume);
        assert_eq!(
            declared_local.default_wal_root.as_deref(),
            Some("file:///srv/proximadb/data/wal"),
            "declared local_volume with no explicit root gets the same default"
        );
    }

    /// A set global `wal_local_dir` ⇒ `local_volume` (an explicit local root
    /// IS that posture) and suppresses the default-root move.
    #[test]
    fn spine5_wal_local_dir_infers_local_volume() {
        let resolution = resolve_durability_mode(
            None,
            Some("file:///waldisk"),
            "s3://bucket/wal",
            Some(&data_dir()),
        );
        assert_eq!(resolution.mode, DurabilityMode::LocalVolume);
        assert_eq!(resolution.source, DurabilityModeSource::WalLocalDir);
        assert_eq!(resolution.evidence, "file:///waldisk");
        assert!(resolution.default_wal_root.is_none());
        assert_eq!(
            resolution
                .effective_wal_root(Some("file:///waldisk"))
                .as_deref(),
            Some("file:///waldisk")
        );
    }

    /// `data_dir: None` (a conversion that cannot know it) still resolves the
    /// mode — only the default-root move is skipped, never the mode claim.
    #[test]
    fn spine5_unknown_data_dir_resolves_mode_without_default_root() {
        let resolution = resolve_durability_mode(None, None, "./data/write_buffer", None);
        assert_eq!(resolution.mode, DurabilityMode::LocalVolume);
        assert!(resolution.default_wal_root.is_none());
    }

    /// The env-value parser accepts exactly the two documented spellings
    /// (case-insensitively); anything else is None so the caller can log and
    /// infer rather than silently guess.
    #[test]
    fn spine5_mode_parsing() {
        assert_eq!(
            DurabilityMode::parse("local_volume"),
            Some(DurabilityMode::LocalVolume)
        );
        assert_eq!(
            DurabilityMode::parse("Object_Store_Serverless"),
            Some(DurabilityMode::ObjectStoreServerless)
        );
        assert_eq!(
            DurabilityMode::parse(" object_store_serverless "),
            Some(DurabilityMode::ObjectStoreServerless)
        );
        assert_eq!(DurabilityMode::parse("serverless"), None);
        assert_eq!(DurabilityMode::parse(""), None);
    }

    /// The TOML spelling round-trips through serde (the declared key fails
    /// config load on an unknown value — fail-fast, not a silent default).
    #[test]
    fn spine5_mode_serde_round_trip() {
        #[derive(serde::Deserialize)]
        struct Wrapper {
            durability_mode: DurabilityMode,
        }
        let local: Wrapper = toml::from_str("durability_mode = \"local_volume\"").unwrap();
        assert_eq!(local.durability_mode, DurabilityMode::LocalVolume);
        let serverless: Wrapper =
            toml::from_str("durability_mode = \"object_store_serverless\"").unwrap();
        assert_eq!(
            serverless.durability_mode,
            DurabilityMode::ObjectStoreServerless
        );
        assert!(toml::from_str::<Wrapper>("durability_mode = \"serverless\"").is_err());
    }
}
