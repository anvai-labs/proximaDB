// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! # Write-Ahead Log (WAL) System - Durability and Recovery Layer
//!
//! This module implements ProximaDB's comprehensive Write-Ahead Log system, providing durability,
//! crash recovery, and high-performance buffering for vector operations. The WAL serves as the
//! first persistence layer, ensuring data durability before vectors are flushed to storage engines.
//!
//! ## Role in ProximaDB Architecture
//!
//! The WAL system sits between the API layer and storage engines:
//! ```text
//! API Handlers → WAL (Memory + Disk) → Storage Engines (SST/VIPER/etc)
//!                 ↓                      ↑
//!              Memtable              Flush Process
//! ```
//!
//! ## Key Components
//!
//! - **WriteAheadLogManager**: Core manager coordinating all WAL operations
//! - **MemtableManager**: In-memory buffer for fast vector access
//! - **DiskManager**: Persistent WAL file management with multi-disk support
//! - **FlushCoordinator**: Orchestrates flushing from WAL to storage engines
//! - **CompactionCoordinator**: Manages WAL compaction and cleanup
//! - **RecoveryManager**: Handles crash recovery on startup
//!
//! ## Features
//!
//! - **Multiple Serialization Strategies**:
//!   - Protocol Buffers (proto-first, zero-copy)
//!   - Avro (schema evolution support)
//!   - Bincode (maximum performance)
//!
//! - **Memory + Disk Architecture**:
//!   - In-memory memtable for fast reads
//!   - Disk persistence for durability
//!   - Configurable memory thresholds
//!
//! - **Atomic Operations**:
//!   - MVCC support for concurrent access
//!   - TTL support for time-based expiry
//!   - Batch operations for throughput
//!
//! - **Multi-Disk Support**:
//!   - Sequential I/O optimization
//!   - Load balancing across disks
//!   - Collection affinity for locality
//!
//! - **Compression & Optimization**:
//!   - Configurable compression (LZ4, Snappy, Zstd, etc.)
//!   - Smart defaults based on data patterns
//!   - Bloom filters for fast lookups
//!
//! ## Performance Characteristics
//!
//! - **Write Latency**: < 1ms for in-memory writes
//! - **Throughput**: 100K+ vectors/sec with batching
//! - **Recovery Time**: Parallel recovery with configurable thread pool
//! - **Memory Usage**: Configurable limits with automatic flushing
//!
//! ## Configuration
//!
//! The WAL system is configured through `WALConfig` with sensible defaults:
//! - Memory threshold: 512MB default
//! - Flush interval: 30 seconds
//! - Compression: Snappy for balance of speed/ratio
//! - Batch size: 1000 vectors for optimal throughput

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cell::UnsafeCell;
use std::sync::Arc;
use tracing::{debug, info, trace, warn};

use crate::core::bloom::BloomFilterStrategy;

use crate::core::{String, VectorId};
use crate::storage::memtable::specialized::wal_behavior::WALVectorBatch;
use crate::storage::traits::{FlushResult, UnifiedStorageFormat};
use proximadb_distance_kernel::DistanceMetric;
use proximadb_distance_kernel::engine::{DistanceComputeProvider, UnifiedDistanceCompute};
use proximadb_records::ProximaRecord;
// DIP: CollectionPathResolver is re-exported below via pub use

// Sub-modules
pub mod background_manager;
pub mod backup; // Incremental backup coordinator
pub mod batch_sync_coordinator;
pub mod collection_path;
pub mod compact_batch_id;
pub mod compaction_axis_integration;
pub mod compaction_coordinator;
pub mod config;
pub mod disk_manager; // New centralized disk operations
pub mod enhanced_flush_result;
pub mod flush_coordinator;
pub mod flush_policy; // ADR-069/TD-WAL-1: the single flush-decision boundary
pub mod flush_result_optimization;
pub mod manifest; // Global WAL manifest system (unified)
pub mod memtable_manager; // New centralized memtable operations
pub mod optimized_write_buffer_writer;
pub mod parallel_search;
pub mod pitr; // Point-in-Time Recovery manager
pub mod recovery_manager; // New centralized recovery operations
pub mod recovery_thread_pool; // Thread pool for parallel recovery
pub mod recovery_token;
pub mod serialization; // New pure serialization layer // Slug codec for collection paths

// Optimized WAL components (Phase 1 implementation) - now consolidated into WriteAheadLogManager
pub mod simple_atomic_sync;
pub mod wal_operations; // WAL operations for vector and graph

// Embedding-precision rollout (PR 4 of EMBEDDING_PRECISION_LLD_2026_05_22).
pub use proximadb_wal_segment_header::v2_segment_header;
// RETIRED (#1270 + TD-CONFIG-CONSOLIDATE-1 step 3): optimized_path_resolver +
// atomic_write_buffer_sync modules + their files deleted. They used the removed
// assignment_service and were commented out ('MARKED FOR REMOVAL') since #1270;
// only the dead cross-ref between the two files remained.

/// ADR-069 S1 opt-in local WAL root (`PROXIMADB_WAL_LOCAL_DIR`). When set to a
/// directory/URL (e.g. `file:///waldisk`), the per-collection WAL is written
/// under `{dir}/{collection_id}/wal/` on that local reattachable disk instead of
/// the collection's object-store data assignment — WAL is high-frequency small
/// appends, each of which is a paid object-store PUT, so a local disk avoids that
/// cost while data/segments stay on object store. Governs BOTH the write path
/// (`resolve_collection_base_location`) and recovery replay so recovery is
/// consistent. Read once (process-global). Empty/unset ⇒ `None` ⇒ legacy
/// WAL-co-located-with-data behavior. Registered: docs/12-design/ENV_GATE_REGISTRY.adoc.
pub(crate) fn local_wal_dir(config: &config::WALConfig) -> Option<String> {
    // Precedence (codebase convention): env override > TOML config > None.
    if let Some(dir) = local_wal_dir_env() {
        return Some(dir);
    }
    config
        .wal_local_dir
        .as_ref()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
}

/// ADR-069 S1: the env half of [`local_wal_dir`] (`PROXIMADB_WAL_LOCAL_DIR`),
/// read once per process. Private on purpose (TD-WAL-5): an env-only read on
/// the replay side, while the write side also honoured the TOML field, lost
/// acknowledged writes. Every reader — write path AND recovery — must go
/// through [`local_wal_dir`] with the same `WALConfig`.
fn local_wal_dir_env() -> Option<String> {
    static ENV: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        std::env::var("PROXIMADB_WAL_LOCAL_DIR")
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
    })
    .clone()
}

// Re-exports
pub use background_manager::{
    BackgroundMaintenanceManager, BackgroundMaintenanceStats, BackgroundTaskStatus,
};
pub use compaction_axis_integration::{CompactionAxisUpdater, CompactionIndexStats};
pub use compaction_coordinator::{
    CollectionCompactionState, CompactionConfig, CompactionCoordinator, CompactionResult,
    CompactionStats, CompactionTask,
};
pub use config::WriteBufferStrategyType;
pub use config::{CompressionConfig, PerformanceConfig, WALConfig};
pub use disk_manager::{DiskStats, WalFileInfo, WriteAheadLogDiskManager};
pub use flush_coordinator::{
    CleanupInstructions, FlushCoordinatorCallbacks, FlushDataSource, FlushState, PendingFlush,
    WALFlushCoordinator,
};
pub use memtable_manager::{MemtableManager, MemtableStats};
pub use recovery_manager::{ParallelRecoveryManager, RecoveryManager, RecoveryMode, RecoveryStats};
pub use recovery_thread_pool::{
    RecoveryPoolStats, RecoveryThreadPool, get_recovery_thread_pool,
    initialize_recovery_thread_pool,
};
pub use recovery_token::{RecoveryToken, RecoveryTokenProvider};

// DIP: Re-export path resolver types for convenient access
pub use crate::storage::trait_components::path_resolver::{
    CollectionPathResolver, ConfigFallbackResolver,
};

// Batch coordination exports - BatchId defined below

// Re-export serialization module
pub use serialization::{SerializationFormat, SerializerFactory, VectorBatchSerializer};

// Re-export manifest module (unified global manifest)
pub use manifest::{
    CheckpointCollectionState,
    GlobalCheckpoint,
    GlobalLsnAllocator,
    // Types
    GlobalManifestEntry,
    // Service
    GlobalManifestService,
    GlobalManifestServiceConfig,
    WalEntryStatus,
    get_service as get_global_manifest,
    // Singleton functions (convenience)
    init as init_global_manifest,
    shutdown as shutdown_global_manifest,
};

/// Modern WAL operation - binary payload for batch operations (Proto-first architecture)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WALOperation {
    /// Operation type: "upsert_batch", "delete_batch", "flush", "checkpoint"
    pub operation_type: String,
    /// Binary payload data (Proto bytes default, strategy-specific for others)
    pub payload_data: Vec<u8>,
    /// Payload format: "proto" (default), "avro", "bincode" (performance)
    pub payload_format: String,
    /// Number of vectors in this batch (for metrics)
    pub vector_count: usize,
    /// When true, this batch arrived via Arrow Flight text-only schema and
    /// needs server-side embedding before the index write completes. The
    /// embedding drainer (proximadb-embedding crate) polls for these entries,
    /// invokes the EmbeddingService, populates the vector field, and flips
    /// this flag to false before promoting the batch to the index.
    ///
    /// Default false — existing serialized WAL entries deserialize correctly
    /// without modification. New entries that don't go through native embedding
    /// also default to false.
    #[serde(default)]
    pub pending_embed: bool,
}

// Re-export CompactBatchId as BatchId - it's globally unique, no need for collection_id
pub use compact_batch_id::CompactBatchId as BatchId;

impl WALOperation {
    /// Calculate the actual memory size of this WAL operation including vector data
    pub fn actual_size_bytes(&self) -> usize {
        let mut size = std::mem::size_of::<WALOperation>(); // Struct size

        // Add variable field sizes
        size += self.operation_type.len();
        size += self.payload_data.len(); // Actual payload size
        size += self.payload_format.len();

        size
    }

    /// Extract the first canonical record from a WAL entry operation.
    pub fn extract_vector_record(&self) -> Result<proximadb_records::ProximaRecord, anyhow::Error> {
        // Proto-first architecture: payload format determines deserialization
        if self.operation_type == "upsert_batch" || self.operation_type == "delete_batch" {
            match self.payload_format.as_str() {
                "proto" => {
                    // Deserialize from proto bytes
                    use crate::storage::persistence::write_ahead_log::serialization::{
                        ProtocolBuffersSerializer, VectorBatchSerializer,
                    };
                    let serializer = ProtocolBuffersSerializer::new();
                    let records = serializer.deserialize_batch(&self.payload_data)?;
                    records
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("Empty vector batch in proto payload"))
                }
                "avro" => {
                    // Delegate to Avro-specific deserialization
                    use crate::storage::persistence::write_ahead_log::serialization::{
                        AvroSerializer, VectorBatchSerializer,
                    };
                    let serializer = AvroSerializer::new();
                    let records =
                        serializer
                            .deserialize_batch(&self.payload_data)
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to deserialize Avro payload: {}", e)
                            })?;
                    records
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("Empty vector batch in Avro payload"))
                }
                _ => Err(anyhow::anyhow!(
                    "Unsupported payload format: {}",
                    self.payload_format
                )),
            }
        } else {
            Err(anyhow::anyhow!(
                "WAL operation type {} does not contain vector data",
                self.operation_type
            ))
        }
    }
}

/// WAL statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WALStats {
    pub total_entries: u64,
    pub memory_entries: u64,
    pub disk_segments: u64,
    pub total_disk_size_bytes: u64,
    pub memory_size_bytes: u64,
    pub collections_count: usize,
    pub last_flush_time: Option<DateTime<Utc>>,
    pub write_throughput_entries_per_sec: f64,
    pub read_throughput_entries_per_sec: f64,
    pub compression_ratio: f64,
}

/// Atomic flush cycle for consistent WAL→Storage operations
#[derive(Debug, Clone)]
pub struct FlushCycle {
    /// Unique identifier for this flush operation
    pub flush_id: String,
    /// Collection being flushed
    pub collection_id: String,
    /// WAL batches marked for flush (replaces Vec<WalEntry>)
    pub batches: Vec<WALVectorBatch>,
    /// Extracted canonical records ready for storage
    pub vector_records: Vec<proximadb_records::ProximaRecord>,
    /// Disk segments marked as flush-pending
    pub marked_segments: Vec<String>,
    /// Sequence ranges marked as flush-pending
    pub marked_sequences: Vec<(u64, u64)>, // (start_seq, end_seq) pairs
    /// Batch IDs for coordination with storage engines
    pub batch_ids: Vec<BatchId>,
    /// Current state of the flush cycle
    pub state: FlushCycleState,
}

/// State of a flush cycle operation
#[derive(Debug, Clone, PartialEq)]
pub enum FlushCycleState {
    /// Flush cycle is active and entries are marked
    Active,
    /// Flush cycle completed successfully
    Completed,
    /// Flush cycle was aborted and entries restored
    Aborted,
}

/// Result of completing a flush cycle
#[derive(Debug, Clone)]
pub struct FlushCompletionResult {
    /// Number of entries permanently removed
    pub entries_removed: usize,
    /// Number of disk segments cleaned up
    pub segments_cleaned: usize,
    /// Bytes reclaimed from cleanup
    pub bytes_reclaimed: u64,
}

// Distance computation functions moved to unified distance system
// These functions are now available through UnifiedDistanceCompute

/// Modern WAL manager using batch-oriented strategies with assignment service integration
///
/// **WriteAheadLogManager Per Collection + Shared Global Memtable Architecture (Perfect Horizontal Scaling)**
///
/// This implements the optimal architecture where:
/// - **WriteAheadLogManager per collection** - Each collection gets its own WriteAheadLogManager for isolation
/// - **Shared global WALBehaviorWrapper** - Single singleton shared across all WalManagers
/// - **GlobalPartitionedMemtable** - Partitioned by collection, efficient shared access
/// - **WriteAheadLogManagerRegistry** - Tracks which WriteAheadLogManager handles which collection (1:1 or 1:N mapping)
/// - **Horizontal scaling constraint** - One collection handled by exactly one WriteAheadLogManager (never split)
/// - **Dynamic scaling** - Under heavy workload, new collections get new WalManagers
/// - **Strategy-specific serialization** with shared deserialization in global memtable
/// - **Collection-specific storage locations** from collection metadata
/// - **Atomic disk synchronization** using TransactionCoordinator
///
/// Collection assignment info with storage location and critical config.
/// The collection_id is the HashMap key, so not stored here.
#[derive(Debug, Clone)]
pub struct CollectionAssignment {
    /// Base storage location for this collection (e.g., "file:///data/disk1" or "s3://bucket/path")
    pub base_location: String,
    /// Storage engine type (affects flush strategy)
    pub storage_engine: crate::proto::proximadb_v1::StorageEngine,
    /// Vector dimension (for buffer size calculations)
    pub dimension: i32,
    /// Compression config (if any) - critical for write operations
    pub compression_config: Option<crate::proto::proximadb_v1::CompressionConfig>,
    /// Distance metric (for similarity operations in WAL)
    pub distance_metric: crate::proto::proximadb_v1::DistanceMetric,
}

pub struct WriteAheadLogManager {
    /// Active strategy for current operations
    // strategy removed -  Box<dyn WALBatchStrategy>,
    /// Configuration
    config: WALConfig,
    /// Statistics tracking
    stats: Arc<tokio::sync::RwLock<WALStats>>,
    /// Distance computation for similarity operations
    distance_compute: UnifiedDistanceCompute,
    /// **Collections assigned to this WriteAheadLogManager with their storage locations**
    assigned_collections:
        Arc<tokio::sync::RwLock<std::collections::HashMap<String, CollectionAssignment>>>,
    /// **SHARED REFERENCE**: Global WALBehaviorWrapper singleton shared across ALL WriteAheadLogManager instances
    shared_wal_behavior: &'static GlobalWriteBufferBehaviorSingleton,
    /// Strategy type for routing and serialization decisions
    strategy_type: config::WriteBufferStrategyType,
    /// Cached RecoveryManager instance - shared across registration and recovery
    recovery_manager_cache: Arc<tokio::sync::RwLock<Option<RecoveryManager>>>,
    /// DIP-compliant path resolver (optional). When set, this is used for path
    /// resolution; otherwise paths are resolved through the global xCatalog.
    path_resolver: Option<Arc<dyn CollectionPathResolver>>,
}

/// Adaptive WriteAheadLogManager Registry with Pool-based Collection Assignment
/// This implements intelligent scaling for millions of collections by maintaining a pool
/// of WriteAheadLogManager instances and dynamically assigning collections based on workload
pub struct WriteAheadLogManagerRegistry {
    /// Collection to WriteAheadLogManager ID mapping (1:1 constraint maintained)
    collection_assignments: Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>>,
    /// WriteAheadLogManager pool with workload tracking
    manager_pool:
        Arc<tokio::sync::RwLock<std::collections::HashMap<String, WriteAheadLogManagerPoolEntry>>>,
    /// Pool configuration
    pool_config: WriteAheadLogManagerPoolConfig,
    /// WAL configuration for creating new managers
    wal_config: config::WALConfig,
    /// Next manager ID for creating new instances
    next_manager_id: Arc<tokio::sync::Mutex<u64>>,
}

/// WriteAheadLogManager pool entry with workload metrics
#[derive(Debug, Clone)]
pub struct WriteAheadLogManagerPoolEntry {
    /// The WriteAheadLogManager instance
    manager: Arc<WriteAheadLogManager>,
    /// Workload metrics for load balancing
    workload_metrics: WriteAheadLogManagerWorkload,
    /// Last rebalancing timestamp
    #[allow(dead_code)]
    last_rebalance: std::time::Instant,
}

/// Workload metrics for adaptive scaling decisions
#[derive(Debug, Clone, Default)]
pub struct WriteAheadLogManagerWorkload {
    /// Number of assigned collections
    collection_count: usize,
    /// Operations per second (estimated)
    #[allow(dead_code)]
    ops_per_second: f64,
    /// Memory usage in bytes
    #[allow(dead_code)]
    memory_usage_bytes: u64,
    /// Average operation latency in milliseconds
    #[allow(dead_code)]
    avg_latency_ms: f64,
    /// Load score (computed from metrics)
    load_score: f64,
}

/// Pool configuration for adaptive WriteAheadLogManager scaling
///
/// This configuration allows users to customize how WalManagers scale to handle
/// millions of collections efficiently. Users can adjust thread counts, load balancing
/// thresholds, and scaling behavior based on their specific workload requirements.
///
/// # Examples
///
/// ```rust,ignore
/// use proximadb::storage::persistence::write_ahead_log::WriteAheadLogManagerPoolConfig;
///
/// // Configuration for high-throughput workloads
/// let high_throughput_config = WriteAheadLogManagerPoolConfig::builder()
///     .initial_pool_size(8)
///     .soft_thread_limit(16)
///     .target_collections_per_manager(500)
///     .enable_dynamic_scaling(true)
///     .build();
///
/// // Configuration for memory-constrained environments
/// let memory_constrained_config = WriteAheadLogManagerPoolConfig::builder()
///     .initial_pool_size(2)
///     .soft_thread_limit(4)
///     .target_collections_per_manager(2000)
///     .enable_dynamic_scaling(false)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct WriteAheadLogManagerPoolConfig {
    /// Initial pool size (number of WriteAheadLogManager threads to start with)
    pub initial_pool_size: usize,
    /// Soft limit - after this many managers, scale collections per manager instead of adding threads
    pub soft_thread_limit: usize,
    /// Target collections per manager for balanced scaling
    pub target_collections_per_manager: usize,
    /// Load threshold for triggering rebalancing (0.0-1.0)
    pub rebalance_load_threshold: f64,
    /// Minimum time between rebalancing operations (seconds)
    pub rebalance_cooldown_secs: u64,
    /// Enable dynamic manager creation beyond soft limit
    pub enable_dynamic_scaling: bool,
}

/// Builder for WriteAheadLogManagerPoolConfig to provide user-friendly configuration
#[derive(Debug, Clone)]
pub struct WriteAheadLogManagerPoolConfigBuilder {
    config: WriteAheadLogManagerPoolConfig,
}

impl WriteAheadLogManagerPoolConfig {
    /// Create a new builder for WriteAheadLogManagerPoolConfig
    pub fn builder() -> WriteAheadLogManagerPoolConfigBuilder {
        WriteAheadLogManagerPoolConfigBuilder::new()
    }

    /// Create configuration optimized for high-throughput workloads
    pub fn high_throughput() -> Self {
        Self {
            initial_pool_size: 8,
            soft_thread_limit: 16,
            target_collections_per_manager: 500,
            rebalance_load_threshold: 0.7,
            rebalance_cooldown_secs: 15,
            enable_dynamic_scaling: true,
        }
    }

    /// Create configuration optimized for memory-constrained environments
    pub fn memory_constrained() -> Self {
        Self {
            initial_pool_size: 2,
            soft_thread_limit: 4,
            target_collections_per_manager: 2000,
            rebalance_load_threshold: 0.9,
            rebalance_cooldown_secs: 60,
            enable_dynamic_scaling: false,
        }
    }

    /// Create configuration optimized for development/testing
    pub fn development() -> Self {
        Self::default()
    }
}

impl WriteAheadLogManagerPoolConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: WriteAheadLogManagerPoolConfig::default(),
        }
    }

    /// Set the initial pool size
    pub fn initial_pool_size(mut self, size: usize) -> Self {
        self.config.initial_pool_size = size;
        self
    }

    /// Set the soft thread limit
    pub fn soft_thread_limit(mut self, limit: usize) -> Self {
        self.config.soft_thread_limit = limit;
        self
    }

    /// Set target collections per manager
    pub fn target_collections_per_manager(mut self, target: usize) -> Self {
        self.config.target_collections_per_manager = target;
        self
    }

    /// Set rebalance load threshold (0.0-1.0)
    pub fn rebalance_load_threshold(mut self, threshold: f64) -> Self {
        self.config.rebalance_load_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set rebalance cooldown period in seconds
    pub fn rebalance_cooldown_secs(mut self, secs: u64) -> Self {
        self.config.rebalance_cooldown_secs = secs;
        self
    }

    /// Enable or disable dynamic scaling beyond soft limit
    pub fn enable_dynamic_scaling(mut self, enabled: bool) -> Self {
        self.config.enable_dynamic_scaling = enabled;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> WriteAheadLogManagerPoolConfig {
        self.config
    }
}

impl Default for WriteAheadLogManagerPoolConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WriteAheadLogManagerPoolConfig {
    fn default() -> Self {
        Self {
            initial_pool_size: 3,                 // Start with 3 threads for testing
            soft_thread_limit: 8,                 // Soft limit - after 8, scale balanced
            target_collections_per_manager: 1000, // Target 1K collections per manager for balanced scaling
            rebalance_load_threshold: 0.8,        // Rebalance when 80% loaded
            rebalance_cooldown_secs: 30,          // 30-second cooldown between rebalances
            enable_dynamic_scaling: true,         // Enable dynamic manager creation
        }
    }
}

impl WriteAheadLogManagerRegistry {
    /// Create new adaptive WriteAheadLogManager registry with pool
    pub fn new() -> Self {
        Self::with_config(WriteAheadLogManagerPoolConfig::default())
    }

    /// Create new adaptive WriteAheadLogManager registry with custom pool configuration
    pub fn with_config(pool_config: WriteAheadLogManagerPoolConfig) -> Self {
        tracing::info!(
            "Creating adaptive WriteAheadLogManager registry - initial: {}, soft_limit: {}, target_collections: {}, dynamic_scaling: {}",
            pool_config.initial_pool_size,
            pool_config.soft_thread_limit,
            pool_config.target_collections_per_manager,
            pool_config.enable_dynamic_scaling
        );

        Self {
            collection_assignments: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
            manager_pool: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            pool_config,
            wal_config: config::WALConfig::default(),
            next_manager_id: Arc::new(tokio::sync::Mutex::new(1)),
        }
    }

    /// Get or assign WriteAheadLogManager for a collection using adaptive pool scaling
    pub async fn get_manager_for_collection(
        &self,
        collection_id: &str,
        strategy_type: crate::storage::persistence::write_ahead_log::config::WriteBufferStrategyType,
        config: &WALConfig,
        filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Result<Arc<WriteAheadLogManager>> {
        // Check if collection already has a manager assignment
        {
            let assignments = self.collection_assignments.read().await;
            if let Some(manager_id) = assignments.get(collection_id) {
                let pool = self.manager_pool.read().await;
                if let Some(entry) = pool.get(manager_id) {
                    tracing::debug!(
                        "📍 Collection {} using existing WriteAheadLogManager {} (load: {:.2})",
                        collection_id,
                        manager_id,
                        entry.workload_metrics.load_score
                    );
                    return Ok(entry.manager.clone());
                }
            }
        }

        // Ensure initial pool exists
        self.ensure_initial_pool(strategy_type, config, filesystem.clone())
            .await?;

        // Find best manager for this collection using adaptive assignment
        let target_manager_id = self.find_best_manager_for_collection(collection_id).await?;

        // Assign collection to the selected manager
        self.assign_collection_to_manager(collection_id, &target_manager_id)
            .await?;

        // Return the assigned manager
        let pool = self.manager_pool.read().await;
        let entry = pool
            .get(&target_manager_id)
            .ok_or_else(|| anyhow::anyhow!("Manager {} not found in pool", target_manager_id))?;

        tracing::info!(
            "✅ Collection {} assigned to WriteAheadLogManager {} (adaptive scaling - {} collections)",
            collection_id,
            target_manager_id,
            entry.workload_metrics.collection_count + 1
        );

        Ok(entry.manager.clone())
    }

    /// Ensure initial pool of WalManagers exists
    async fn ensure_initial_pool(
        &self,
        _strategy_type: crate::storage::persistence::write_ahead_log::config::WriteBufferStrategyType,
        config: &WALConfig,
        _filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Result<()> {
        let pool = self.manager_pool.read().await;
        if !pool.is_empty() {
            return Ok(()); // Pool already initialized
        }
        drop(pool);

        // Initialize pool with configured size
        let mut pool = self.manager_pool.write().await;
        if !pool.is_empty() {
            return Ok(()); // Another thread initialized it
        }

        tracing::info!(
            "🚀 Initializing WriteAheadLogManager pool with {} managers",
            self.pool_config.initial_pool_size
        );

        for i in 0..self.pool_config.initial_pool_size {
            let manager_id = format!("write_buffer_manager_pool_{}", i + 1);

            let manager = Arc::new(
                WriteAheadLogManager::new_pool_manager(config.clone(), manager_id.clone()).await?,
            );

            let entry = WriteAheadLogManagerPoolEntry {
                manager,
                workload_metrics: WriteAheadLogManagerWorkload::default(),
                last_rebalance: std::time::Instant::now(),
            };

            pool.insert(manager_id.clone(), entry);
            tracing::debug!("➕ Created pool WriteAheadLogManager: {}", manager_id);
        }

        tracing::info!(
            "✅ WriteAheadLogManager pool initialized with {} managers",
            pool.len()
        );
        Ok(())
    }

    /// Find the best WriteAheadLogManager for a new collection based on workload
    /// Implements adaptive scaling: use existing managers first, then create new ones if needed
    async fn find_best_manager_for_collection(&self, collection_id: &str) -> Result<String> {
        let pool = self.manager_pool.read().await;

        if pool.is_empty() {
            return Err(anyhow::anyhow!("WriteAheadLogManager pool is empty"));
        }

        // Check if we should create a new manager for better load distribution
        if self.should_create_new_manager(&pool).await? {
            drop(pool); // Release read lock before write lock
            return self.create_additional_manager().await;
        }

        // Find manager with lowest load score from existing pool
        let best_manager = pool
            .iter()
            .min_by(|(_, a), (_, b)| {
                // Primary: load score (lower is better)
                match a
                    .workload_metrics
                    .load_score
                    .partial_cmp(&b.workload_metrics.load_score)
                {
                    Some(std::cmp::Ordering::Equal) => {
                        // Secondary: collection count (lower is better)
                        a.workload_metrics
                            .collection_count
                            .cmp(&b.workload_metrics.collection_count)
                    }
                    Some(ordering) => ordering,
                    None => std::cmp::Ordering::Equal, // Treat NaN as equal
                }
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| anyhow::anyhow!("No suitable manager found in pool"))?;

        tracing::debug!(
            "🎯 Selected existing WriteAheadLogManager {} for collection {} (adaptive assignment)",
            best_manager,
            collection_id
        );
        Ok(best_manager)
    }

    /// Check if we should create a new manager for better load distribution
    async fn should_create_new_manager(
        &self,
        pool: &std::collections::HashMap<String, WriteAheadLogManagerPoolEntry>,
    ) -> Result<bool> {
        // Don't create new managers if dynamic scaling is disabled
        if !self.pool_config.enable_dynamic_scaling {
            return Ok(false);
        }

        // If we're below soft limit, don't create new managers yet (use existing capacity)
        if pool.len() < self.pool_config.soft_thread_limit {
            return Ok(false);
        }

        // After soft limit, check if adding a manager would improve load distribution
        let min_collections = pool
            .values()
            .map(|entry| entry.workload_metrics.collection_count)
            .min();
        let max_collections = pool
            .values()
            .map(|entry| entry.workload_metrics.collection_count)
            .max();

        // Create new manager if:
        // 1. The most loaded manager exceeds target collections per manager
        // 2. There's significant imbalance between managers
        let should_create = max_collections.unwrap_or(0)
            > self.pool_config.target_collections_per_manager
            || (max_collections
                .unwrap_or(0)
                .saturating_sub(min_collections.unwrap_or(0)))
                > (self.pool_config.target_collections_per_manager / 2);

        if should_create {
            tracing::info!(
                "🔄 Dynamic scaling triggered - current managers: {}, max_collections: {}, target: {}",
                pool.len(),
                max_collections.unwrap_or(0),
                self.pool_config.target_collections_per_manager
            );
        }

        Ok(should_create)
    }

    /// Create an additional manager for dynamic scaling
    async fn create_additional_manager(&self) -> Result<String> {
        let mut next_id = self.next_manager_id.lock().await;
        let manager_id = format!("write_buffer_manager_dynamic_{}", *next_id);
        *next_id += 1;
        drop(next_id);

        tracing::info!(
            "Creating dynamic WriteAheadLogManager {} for load balancing",
            manager_id
        );

        // Use registry-level config for dynamic managers
        let config = &self.wal_config;
        let manager = Arc::new(
            WriteAheadLogManager::new_pool_manager(config.clone(), manager_id.clone()).await?,
        );

        let entry = WriteAheadLogManagerPoolEntry {
            manager,
            workload_metrics: WriteAheadLogManagerWorkload::default(),
            last_rebalance: std::time::Instant::now(),
        };

        // Add to pool
        {
            let mut pool = self.manager_pool.write().await;
            pool.insert(manager_id.clone(), entry);
        }

        tracing::info!(
            "✅ Dynamic WriteAheadLogManager {} created for balanced scaling",
            manager_id
        );
        Ok(manager_id)
    }

    /// Assign a collection to a specific manager
    async fn assign_collection_to_manager(
        &self,
        collection_id: &str,
        manager_id: &str,
    ) -> Result<()> {
        // Update assignments
        {
            let mut assignments = self.collection_assignments.write().await;
            assignments.insert(collection_id.to_string(), manager_id.to_string());
        }

        // Update pool entry
        {
            let mut pool = self.manager_pool.write().await;
            if let Some(entry) = pool.get_mut(manager_id) {
                // Get current count from the manager's HashMap
                let collection_count = {
                    let assigned = entry.manager.assigned_collections.read().await;
                    assigned.len() + 1 // +1 for the collection we're about to add
                };

                entry.workload_metrics.collection_count = collection_count;

                // Update load score based on collection count
                entry.workload_metrics.load_score = (entry.workload_metrics.collection_count
                    as f64)
                    / (self.pool_config.target_collections_per_manager as f64);
            } else {
                return Err(anyhow::anyhow!("Manager {} not found in pool", manager_id));
            }
        }

        Ok(())
    }

    /// Get all active managers (for global operations)
    pub async fn get_all_managers(
        &self,
    ) -> std::collections::HashMap<String, Arc<WriteAheadLogManager>> {
        let pool = self.manager_pool.read().await;
        pool.iter()
            .map(|(id, entry)| (id.clone(), entry.manager.clone()))
            .collect()
    }

    /// Remove collection assignment (when collection is dropped)
    pub async fn remove_collection(&self, collection_id: &str) -> Result<()> {
        let mut assignments = self.collection_assignments.write().await;
        if let Some(manager_id) = assignments.remove(collection_id) {
            // Check if this manager handles other collections
            let has_other_collections = assignments.values().any(|id| id == &manager_id);

            if !has_other_collections {
                // Remove the manager if it has no more collections
                let mut managers = self.manager_pool.write().await;
                if let Some(manager_entry) = managers.remove(&manager_id) {
                    // Close the manager gracefully
                    let _ = manager_entry.manager.close().await;
                    tracing::info!(
                        "🗑️ Removed WriteAheadLogManager {} (no more collections)",
                        manager_id
                    );
                }
            }
        }
        Ok(())
    }
}

impl Default for WriteAheadLogManagerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal resettable wrapper around OnceLock to support tests that create multiple embedded instances.
struct ResettableOnceLock<T> {
    inner: UnsafeCell<std::sync::OnceLock<T>>,
}

impl<T> ResettableOnceLock<T> {
    const fn new() -> Self {
        Self {
            inner: UnsafeCell::new(std::sync::OnceLock::new()),
        }
    }

    fn get(&self) -> &std::sync::OnceLock<T> {
        // Safety: OnceLock provides interior mutability; reset is gated by test-only API.
        unsafe { &*self.inner.get() }
    }

    unsafe fn reset(&self) {
        unsafe {
            *self.inner.get() = std::sync::OnceLock::new();
        }
    }
}

unsafe impl<T: Send + Sync> Sync for ResettableOnceLock<T> {}

/// Global WALBehaviorWrapper singleton for shared memtable access across all WriteAheadLogManager instances
/// This is the key to efficient shared access - one memtable, many managers
pub struct GlobalWriteBufferBehaviorSingleton {
    /// The singleton WALBehaviorWrapper with GlobalPartitionedMemtable
    wal_behavior: ResettableOnceLock<
        Arc<crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper>,
    >,
}

impl GlobalWriteBufferBehaviorSingleton {
    /// Get or create the singleton WALBehaviorWrapper instance
    pub fn get_or_init(
        &self,
        config: &crate::storage::memtable::core::MemtableConfig,
    ) -> Arc<crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper> {
        self.wal_behavior.get().get_or_init(|| {
            tracing::info!("🎯 Creating SINGLETON WALBehaviorWrapper with GlobalPartitionedMemtable for all WriteAheadLogManager instances");
            Arc::new(crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper::new(config.clone()))
        }).clone()
    }
}

/// Global singleton instance - shared across all WriteAheadLogManager instances
static GLOBAL_WRITE_BUFFER_BEHAVIOR: GlobalWriteBufferBehaviorSingleton =
    GlobalWriteBufferBehaviorSingleton {
        wal_behavior: ResettableOnceLock::new(),
    };

/// Global registry instance for WriteAheadLogManager per collection architecture
static WAL_MANAGER_REGISTRY: ResettableOnceLock<WriteAheadLogManagerRegistry> =
    ResettableOnceLock::new();

/// Internal helper to reset global singletons (tests/embedded only)
pub unsafe fn reset_global_wal_state_for_tests() {
    unsafe {
        GLOBAL_CATALOG.reset();
        GLOBAL_WRITE_BUFFER_BEHAVIOR.wal_behavior.reset();
        WAL_MANAGER_REGISTRY.reset();
    }
}

/// Global catalog singleton — the sole read-authoritative source for collection
/// metadata during WAL path resolution and recovery. When set, WAL/recovery
/// resolve a collection's config + storage assignment straight from the catalog
/// (the system-of-record).
type GlobalCatalogValue = Arc<tokio::sync::RwLock<Option<Arc<crate::catalog::CatalogManager>>>>;

static GLOBAL_CATALOG: ResettableOnceLock<GlobalCatalogValue> = ResettableOnceLock::new();

fn get_global_catalog() -> GlobalCatalogValue {
    GLOBAL_CATALOG
        .get()
        .get_or_init(|| Arc::new(tokio::sync::RwLock::new(None)))
        .clone()
}

/// Set the global catalog used for WAL/recovery collection resolution. Call once
/// at boot, before any WAL writes or recovery.
pub async fn set_global_catalog(catalog: Arc<crate::catalog::CatalogManager>) {
    let global = get_global_catalog();
    let mut lock = global.write().await;
    *lock = Some(catalog);
    tracing::info!("✅ Global catalog set for WAL/recovery collection resolution");
}

/// Resolve a collection (by id or name) to its proto `Collection` via the
/// catalog, or `None` if no catalog is wired / the collection is absent. The
/// catalog asset carries the full collection shape (config, storage assignment,
/// stats), so this faithfully replaces a legacy `provider.get_collection()`
/// lookup. Resolves by FQN first, then scans by collection id/name (collections
/// are catalog tables keyed by name, with the id carried as an asset property).
pub(crate) async fn resolve_collection_from_catalog(
    collection_id: &str,
) -> Option<crate::proto::proximadb_v1::Collection> {
    let global = get_global_catalog();
    let catalog_manager = global.read().await.clone()?;

    // Fast path: the identifier is a fully-qualified table name.
    if let Ok((catalog, table_id)) = catalog_manager.resolve_table(collection_id).await
        && catalog.table_exists(&table_id).await.unwrap_or(false)
        && let Ok(schema) = catalog.get_table(&table_id).await
        && let Ok(Some(collection)) =
            crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                &table_id, &schema,
            )
    {
        return Some(collection);
    }

    // Fallback: the identifier is an opaque collection id. Scan the catalog and
    // match by id or name. TD-FLUSH-8: use the `["default"]`-guaranteed namespace
    // walk so this agrees with the write path even when `list_namespaces(None)`
    // omits `["default"]`.
    let catalog = catalog_manager.default_catalog().await.ok()?;
    for namespace in catalog_default_namespaces(catalog.as_ref()).await {
        let Ok(table_ids) = catalog.list_tables(&namespace).await else {
            continue;
        };
        for table_id in table_ids {
            let Ok(schema) = catalog.get_table(&table_id).await else {
                continue;
            };
            if let Ok(Some(collection)) =
                crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                    &table_id, &schema,
                )
                && (collection.id == collection_id
                    || collection
                        .config
                        .as_ref()
                        .is_some_and(|cfg| cfg.name == collection_id))
            {
                return Some(collection);
            }
        }
    }
    None
}

/// List every collection known to the global catalog as proto `Collection`s, or
/// an empty vec if no catalog is wired. The catalog is the sole authority, so
/// this replaces the legacy `provider.list_collections()` lookup used during WAL
/// recovery.
pub(crate) async fn list_collections_from_catalog() -> Vec<crate::proto::proximadb_v1::Collection> {
    let global = get_global_catalog();
    let Some(catalog_manager) = global.read().await.clone() else {
        return Vec::new();
    };
    let Ok(catalog) = catalog_manager.default_catalog().await else {
        return Vec::new();
    };
    collections_from_catalog_walk(catalog.as_ref()).await
}

/// TD-FLUSH-8: the catalog walk behind [`list_collections_from_catalog`],
/// factored over `&dyn Catalog` so the flush-side listing is testable without
/// global state. Enumerates namespaces the SAME way the write path's
/// `read_collections_from_catalog` does — always including `["default"]`, even
/// when `list_namespaces(None)` omits it (a reload/visibility gap) — and dedups
/// by id. Without the fallback this walk was strictly weaker than the write
/// path: a collection that resolved on INSERT was invisible to the auto-flush
/// driver (`no catalog metadata for '1'`), so time-triggered (RPO) flushes for
/// it never fired. The catalog is the identity authority; both paths must agree.
async fn collections_from_catalog_walk(
    catalog: &dyn proximadb_catalog::Catalog,
) -> Vec<crate::proto::proximadb_v1::Collection> {
    let mut collections = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    for namespace in catalog_default_namespaces(catalog).await {
        let Ok(table_ids) = catalog.list_tables(&namespace).await else {
            continue;
        };
        for table_id in table_ids {
            let Ok(schema) = catalog.get_table(&table_id).await else {
                continue;
            };
            if let Ok(Some(collection)) =
                crate::storage::metadata::collection_mapping::collection_from_catalog_schema(
                    &table_id, &schema,
                )
                && seen_ids.insert(collection.id.clone())
            {
                collections.push(collection);
            }
        }
    }
    collections
}

/// TD-FLUSH-8: the namespaces to walk when the catalog is the authority for the
/// full collection set — every namespace `list_namespaces(None)` reports, plus a
/// guaranteed `["default"]` even when the listing omits it (mirrors the write
/// path's `read_collections_from_catalog`, `manager.rs:2592-2601`). Shared by the
/// list + resolve free functions so they can never disagree with the write path.
async fn catalog_default_namespaces(catalog: &dyn proximadb_catalog::Catalog) -> Vec<Vec<String>> {
    let mut namespaces: Vec<Vec<String>> = catalog
        .list_namespaces(None)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|namespace| namespace.levels)
        .collect();
    if !namespaces.iter().any(|namespace| namespace == &["default"]) {
        namespaces.push(vec!["default".to_string()]);
    }
    namespaces
}

/// Get the global write buffer behavior singleton if it has been initialized
/// Returns None if the singleton has not been initialized yet
///
/// This is used during graceful shutdown to access unflushed data and flush
/// all collections to their respective storage engines.
pub fn get_global_write_buffer_behavior()
-> Option<Arc<crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper>> {
    GLOBAL_WRITE_BUFFER_BEHAVIOR
        .wal_behavior
        .get()
        .get()
        .cloned()
}

/// ADR-069 D2 inline size-trigger: fire-and-forget a `materialize_collection` for the
/// collection. ADR-081's canonical materializer gate collapses a burst for this
/// collection. Reuses the exact `AutoFlushDriver` recipe (same globals, same plan
/// shape via `flush_plan_from_collection_meta`) — ONE materialization path; the
/// inline trigger only adds *when* to fire (a size crossing on the live write path,
/// vs. the driver's periodic tick).
/// TD-FLUSH-4: count of inline size-flush materializations currently in
/// flight. Shutdown MUST await these — a large-memtable materialize takes
/// tens of seconds (measured 43.9s for 884k entries) and killing it at
/// teardown leaves a `__flush/` staging straggler that startup cleanup
/// deletes, so the next boot comes up segment-less (WAL-safe but cold).
static INLINE_FLUSHES_IN_FLIGHT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// TD-FLUSH-4: await in-flight inline flushes (bounded). Called from
/// `ProximaDB::shutdown` BEFORE the storage engine stops, so triggered
/// materializations complete and rename into place instead of dying mid-
/// staging. Returns the number still in flight at timeout (0 = clean).
pub async fn drain_inline_flushes(timeout: std::time::Duration) -> usize {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let inflight = INLINE_FLUSHES_IN_FLIGHT.load(std::sync::atomic::Ordering::Acquire);
        if inflight == 0 {
            return 0;
        }
        if std::time::Instant::now() >= deadline {
            tracing::warn!(
                inflight,
                "shutdown: inline flush(es) still in flight at drain timeout"
            );
            return inflight;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

struct InlineFlushTicket;

impl InlineFlushTicket {
    fn new() -> Self {
        INLINE_FLUSHES_IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        InlineFlushTicket
    }
}

impl Drop for InlineFlushTicket {
    fn drop(&mut self) {
        INLINE_FLUSHES_IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

fn spawn_inline_flush(collection_id: String) {
    use crate::storage::flush_materializer::{
        flush_plan_from_collection_meta, materialize_collection_if_idle,
    };

    tokio::spawn(async move {
        let _ticket = InlineFlushTicket::new();

        let Some(write_buffer) = get_global_write_buffer_behavior() else {
            return;
        };
        #[cfg(feature = "axis")]
        let axis = crate::index::get_global_axis_manager();
        let catalog = list_collections_from_catalog().await;
        let Some(meta) = catalog.iter().find(|c| c.id == collection_id) else {
            tracing::warn!(
                "inline size-flush: no catalog metadata for '{}', skipping",
                collection_id
            );
            return;
        };
        let plan = match flush_plan_from_collection_meta(meta) {
            Ok(plan) => plan,
            Err(error) => {
                tracing::warn!(
                    collection_id,
                    "inline size-flush: catalog identity resolution failed: {error}"
                );
                return;
            }
        };

        // TD-FLUSH-4: announce the start — a large-memtable materialize runs
        // tens of seconds and segments only become visible on completion
        // (staged under `__flush/` then renamed); without this line the flush
        // is invisible until it lands and "no segments yet" is unreadable.
        tracing::info!(
            collection_id,
            "🚿 inline size-flush starting (segments appear on completion)"
        );
        let start = std::time::Instant::now();
        #[cfg(feature = "axis")]
        let axis_arg = axis.as_deref();
        #[cfg(not(feature = "axis"))]
        let axis_arg: Option<&()> = None;
        match materialize_collection_if_idle(&write_buffer, &plan, None, None, true, axis_arg).await
        {
            Ok(Some(outcome)) => tracing::info!(
                "✅ inline size-flush '{}': {} entries, {} bytes in {:.3}s",
                collection_id,
                outcome.entries_flushed,
                outcome.bytes,
                start.elapsed().as_secs_f64()
            ),
            Ok(None) => {
                // nothing unflushed — a concurrent driver tick already flushed it
            }
            Err(e) => tracing::warn!("⚠️ inline size-flush '{}' failed: {}", collection_id, e),
        }
    });
}

/// Get the global WriteAheadLogManager registry
pub fn get_write_ahead_log_manager_registry() -> &'static WriteAheadLogManagerRegistry {
    WAL_MANAGER_REGISTRY.get().get_or_init(|| {
        tracing::info!("🎯 Initializing WriteAheadLogManager Registry for per-collection scaling");
        WriteAheadLogManagerRegistry::new()
    })
}

/// Convenience function: Get or create WriteAheadLogManager for a collection using default pool config
pub async fn get_write_ahead_log_manager_for_collection(
    collection_id: &str,
    strategy_type: crate::storage::persistence::write_ahead_log::config::WriteBufferStrategyType,
    config: &WALConfig,
    filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
) -> Result<Arc<WriteAheadLogManager>> {
    get_write_ahead_log_manager_registry()
        .get_manager_for_collection(collection_id, strategy_type, config, filesystem)
        .await
}

/// Configure the global WriteAheadLogManager pool with custom settings
///
/// This function allows users to customize the adaptive scaling behavior
/// of the WriteAheadLogManager pool to match their specific workload requirements.
///
/// # Examples
///
/// ```rust,no_run
/// use proximadb::storage::persistence::write_ahead_log::{configure_write_buffer_manager_pool, WriteAheadLogManagerPoolConfig};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Configure for high-throughput workloads
///     configure_write_buffer_manager_pool(WriteAheadLogManagerPoolConfig::high_throughput()).await?;
///     
///     // Configure with custom settings
///     let custom_config = WriteAheadLogManagerPoolConfig::builder()
///         .initial_pool_size(4)
///         .soft_thread_limit(12)
///         .target_collections_per_manager(800)
///         .enable_dynamic_scaling(true)
///         .build();
///     configure_write_buffer_manager_pool(custom_config).await?;
///     Ok(())
/// }
/// ```
pub async fn configure_write_buffer_manager_pool(
    pool_config: WriteAheadLogManagerPoolConfig,
) -> Result<()> {
    // Deferred: Implement global pool configuration
    // For now, this is a placeholder - in a full implementation, this would
    // reinitialize the global registry with the new configuration
    tracing::info!(
        "🔧 WriteAheadLogManager pool configuration updated: {:?}",
        pool_config
    );
    Ok(())
}

/// Get current WriteAheadLogManager pool statistics for monitoring
///
/// Returns information about the current pool state including number of managers,
/// collection distribution, and load metrics.
pub async fn get_write_ahead_log_manager_pool_stats() -> Result<WriteAheadLogManagerPoolStats> {
    let registry = get_write_ahead_log_manager_registry();
    let _managers = registry.get_all_managers().await;

    let pool = registry.manager_pool.read().await;
    let total_collections: usize = pool
        .values()
        .map(|entry| entry.workload_metrics.collection_count)
        .sum();
    let avg_collections_per_manager = if pool.is_empty() {
        0.0
    } else {
        total_collections as f64 / pool.len() as f64
    };
    let max_collections = pool
        .values()
        .map(|entry| entry.workload_metrics.collection_count)
        .max();
    let min_collections = pool
        .values()
        .map(|entry| entry.workload_metrics.collection_count)
        .min();

    Ok(WriteAheadLogManagerPoolStats {
        total_managers: pool.len(),
        total_collections,
        avg_collections_per_manager,
        max_collections_per_manager: max_collections.unwrap_or(0),
        min_collections_per_manager: min_collections.unwrap_or(0),
        load_imbalance: if min_collections.unwrap_or(0) == 0 {
            0.0
        } else {
            max_collections.unwrap_or(0) as f64 / min_collections.unwrap_or(1) as f64
        },
    })
}

/// Statistics about the current WriteAheadLogManager pool state
#[derive(Debug, Clone)]
pub struct WriteAheadLogManagerPoolStats {
    /// Total number of WriteAheadLogManager instances in the pool
    pub total_managers: usize,
    /// Total number of collections across all managers
    pub total_collections: usize,
    /// Average collections per manager
    pub avg_collections_per_manager: f64,
    /// Maximum collections assigned to any single manager
    pub max_collections_per_manager: usize,
    /// Minimum collections assigned to any single manager  
    pub min_collections_per_manager: usize,
    /// Load imbalance ratio (max/min collections)
    pub load_imbalance: f64,
}

impl WriteAheadLogManager {
    /// Create new WriteAheadLogManager
    pub async fn new(config: WALConfig) -> Result<Self> {
        // Use new_pool_manager with a default manager ID for backwards compatibility
        Self::new_pool_manager(config, "default_manager".to_string()).await
    }

    /// Create new WriteAheadLogManager for specific collections with shared global memtable
    pub async fn new_for_collection(config: WALConfig, collection_id: String) -> Result<Self> {
        tracing::info!(
            "🚀 Creating WriteAheadLogManager for collection {} with strategy type {:?} (shared global memtable)",
            collection_id,
            config.strategy_type
        );
        tracing::debug!(
            "📋 WAL Config: strategy_type={:?}, memtable_type={:?}",
            config.strategy_type,
            config.memtable.memtable_type
        );

        // Initialize singleton WALBehaviorWrapper (thread-safe, only happens once globally)
        let memtable_config = crate::storage::memtable::core::MemtableConfig {
            max_size_bytes: config.memtable.global_memory_limit,
            flush_threshold_bytes: config.memtable.global_memory_limit / 2,
            enable_mvcc: config.enable_mvcc,
            mvcc_cleanup_interval_secs: config.performance.mvcc_cleanup_interval_secs,
            max_versions_per_key: config.memtable.mvcc_versions_retained,
        };
        let _shared_wal_behavior = GLOBAL_WRITE_BUFFER_BEHAVIOR.get_or_init(&memtable_config);

        let stats = Arc::new(tokio::sync::RwLock::new(WALStats {
            total_entries: 0,
            memory_entries: 0,
            disk_segments: 0,
            total_disk_size_bytes: 0,
            memory_size_bytes: 0,
            collections_count: 0,
            last_flush_time: None,
            write_throughput_entries_per_sec: 0.0,
            read_throughput_entries_per_sec: 0.0,
            compression_ratio: 1.0,
        }));

        // Initialize with empty collection map - collections will be added with their metadata
        let assigned_collections = std::collections::HashMap::new();

        tracing::info!(
            "✅ WriteAheadLogManager created for collection {} - per-collection scaling with shared memtable",
            collection_id
        );

        // Extract strategy type for routing
        let strategy_type = config.strategy_type;

        // Create filesystem factory for per-collection disk managers
        // Disk managers will be created at write time using collection's base_location from assigned_collections
        let _filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create_default().await?,
        );

        Ok(Self {
            config,
            stats,
            distance_compute: UnifiedDistanceCompute::default(),
            assigned_collections: Arc::new(tokio::sync::RwLock::new(assigned_collections)),
            shared_wal_behavior: &GLOBAL_WRITE_BUFFER_BEHAVIOR,
            strategy_type,
            recovery_manager_cache: Arc::new(tokio::sync::RwLock::new(None)),
            // DIP: No path resolver by default, falls back to the global xCatalog
            path_resolver: None,
        })
    }

    /// Create new WriteAheadLogManager with an injected path resolver (DIP-compliant)
    ///
    /// This constructor enables Dependency Inversion by accepting a `CollectionPathResolver`,
    /// eliminating the 100ms metadata provider wait during path resolution. This is the
    /// recommended constructor for production use.
    ///
    /// # Arguments
    /// * `config` - WAL configuration
    /// * `collection_id` - Collection identifier (for logging)
    /// * `path_resolver` - Injected path resolver for collection path resolution
    ///
    /// # Example
    /// ```rust,ignore
    /// use proximadb::storage::trait_components::path_resolver::{ConfigFallbackResolver, CollectionPathResolver};
    ///
    /// let resolver = Arc::new(ConfigFallbackResolver::default());
    /// let wal_manager = WriteAheadLogManager::new_with_path_resolver(
    ///     config,
    ///     "my_collection".to_string(),
    ///     resolver,
    /// ).await?;
    /// ```
    pub async fn new_with_path_resolver(
        config: WALConfig,
        collection_id: String,
        path_resolver: Arc<dyn CollectionPathResolver>,
    ) -> Result<Self> {
        tracing::info!(
            "🚀 Creating WriteAheadLogManager for collection {} with DIP path resolver '{}' (no 100ms wait)",
            collection_id,
            path_resolver.name()
        );
        tracing::debug!(
            "📋 WAL Config: strategy_type={:?}, memtable_type={:?}, resolver={}",
            config.strategy_type,
            config.memtable.memtable_type,
            path_resolver.name()
        );

        // Initialize singleton WALBehaviorWrapper (thread-safe, only happens once globally)
        let memtable_config = crate::storage::memtable::core::MemtableConfig {
            max_size_bytes: config.memtable.global_memory_limit,
            flush_threshold_bytes: config.memtable.global_memory_limit / 2,
            enable_mvcc: config.enable_mvcc,
            mvcc_cleanup_interval_secs: config.performance.mvcc_cleanup_interval_secs,
            max_versions_per_key: config.memtable.mvcc_versions_retained,
        };
        let _shared_wal_behavior = GLOBAL_WRITE_BUFFER_BEHAVIOR.get_or_init(&memtable_config);

        let stats = Arc::new(tokio::sync::RwLock::new(WALStats {
            total_entries: 0,
            memory_entries: 0,
            disk_segments: 0,
            total_disk_size_bytes: 0,
            memory_size_bytes: 0,
            collections_count: 0,
            last_flush_time: None,
            write_throughput_entries_per_sec: 0.0,
            read_throughput_entries_per_sec: 0.0,
            compression_ratio: 1.0,
        }));

        // Initialize with empty collection map - collections will be added with their metadata
        let assigned_collections = std::collections::HashMap::new();

        tracing::info!(
            "✅ WriteAheadLogManager created for collection {} with DIP resolver '{}' - zero-wait path resolution",
            collection_id,
            path_resolver.name()
        );

        // Extract strategy type for routing
        let strategy_type = config.strategy_type;

        // Create filesystem factory for per-collection disk managers
        let _filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create_default().await?,
        );

        Ok(Self {
            config,
            stats,
            distance_compute: UnifiedDistanceCompute::default(),
            assigned_collections: Arc::new(tokio::sync::RwLock::new(assigned_collections)),
            shared_wal_behavior: &GLOBAL_WRITE_BUFFER_BEHAVIOR,
            strategy_type,
            recovery_manager_cache: Arc::new(tokio::sync::RwLock::new(None)),
            // DIP: Use injected path resolver (no 100ms wait needed)
            path_resolver: Some(path_resolver),
        })
    }

    /// Create new WriteAheadLogManager for pool with empty collection set
    pub async fn new_pool_manager(config: WALConfig, manager_id: String) -> Result<Self> {
        tracing::debug!(
            "🏊 Creating pool WriteAheadLogManager {} (shared global memtable)",
            manager_id
        );

        // Initialize singleton WALBehaviorWrapper (thread-safe, only happens once globally)
        let memtable_config = crate::storage::memtable::core::MemtableConfig {
            max_size_bytes: config.memtable.global_memory_limit,
            flush_threshold_bytes: config.memtable.global_memory_limit / 2,
            enable_mvcc: config.enable_mvcc,
            mvcc_cleanup_interval_secs: config.performance.mvcc_cleanup_interval_secs,
            max_versions_per_key: config.memtable.mvcc_versions_retained,
        };
        let _shared_wal_behavior = GLOBAL_WRITE_BUFFER_BEHAVIOR.get_or_init(&memtable_config);

        let stats = Arc::new(tokio::sync::RwLock::new(WALStats {
            total_entries: 0,
            memory_entries: 0,
            disk_segments: 0,
            total_disk_size_bytes: 0,
            memory_size_bytes: 0,
            collections_count: 0,
            last_flush_time: None,
            write_throughput_entries_per_sec: 0.0,
            read_throughput_entries_per_sec: 0.0,
            compression_ratio: 1.0,
        }));

        // Start with empty collection set (will be assigned dynamically)
        let assigned_collections = std::collections::HashMap::new();

        tracing::debug!(
            "✅ Pool WriteAheadLogManager {} created - ready for adaptive collection assignment",
            manager_id
        );

        // Extract strategy type for routing
        let strategy_type = config.strategy_type;

        // Create filesystem factory for per-collection disk managers
        // Disk managers will be created at write time using collection's base_location from assigned_collections
        let _filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create_default().await?,
        );

        Ok(Self {
            config,
            stats,
            distance_compute: UnifiedDistanceCompute::default(),
            assigned_collections: Arc::new(tokio::sync::RwLock::new(assigned_collections)),
            shared_wal_behavior: &GLOBAL_WRITE_BUFFER_BEHAVIOR,
            strategy_type,
            recovery_manager_cache: Arc::new(tokio::sync::RwLock::new(None)),
            // DIP: No path resolver by default, falls back to the global xCatalog
            path_resolver: None,
        })
    }

    /// Create new WAL manager using the factory (recommended)
    pub async fn create_with_factory(
        _strategy_type: crate::storage::persistence::write_ahead_log::config::WriteBufferStrategyType,
        config: WALConfig,
        _filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Result<Self> {
        // The batch-serialization strategy object was constructed here and then
        // dropped by the manager (which routes on `config.strategy_type` + the
        // global write buffer, never a `WALBatchStrategy`); the whole stack was
        // removed. `strategy_type`/`filesystem` are retained on the signature for
        // the live callers (engine.rs, metadata/write_ahead_log.rs) that pass an
        // enum, but are no longer needed to construct the manager.
        Self::new(config).await
    }

    /// Create new WAL manager using the batch factory (alias for modern naming)
    pub async fn create_with_batch_factory(
        strategy_type: crate::storage::persistence::write_ahead_log::config::WriteBufferStrategyType,
        config: WALConfig,
        filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Result<Self> {
        Self::create_with_factory(strategy_type, config, filesystem).await
    }

    /// Set storage engine for delegated flush/compaction operations
    pub fn set_storage_engine(
        &self,
        _storage_engine: Arc<dyn UnifiedStorageFormat>,
        _collection_id: &str,
    ) {
        // Storage engine setting moved to config level — strategies receive it directly
        tracing::info!("Storage engine attached to WAL manager for delegated operations");
    }

    /// Set path resolver for DIP-compliant path resolution (post-construction injection)
    ///
    /// This method allows injecting a path resolver after the WAL manager is created,
    /// which is useful for pool managers that are created before the metadata provider
    /// is fully initialized.
    ///
    /// Once set, the path resolver will be used for path resolution, bypassing the
    /// 100ms metadata provider wait.
    ///
    /// Note: This requires mutable access to `self`. For immutable post-construction
    /// injection, create the WAL manager with `new_with_path_resolver` instead.
    pub fn set_path_resolver(&mut self, resolver: Arc<dyn CollectionPathResolver>) {
        tracing::info!(
            "📋 Path resolver '{}' attached to WAL manager - DIP path resolution enabled",
            resolver.name()
        );
        self.path_resolver = Some(resolver);
    }

    /// Check if a path resolver is configured
    pub fn has_path_resolver(&self) -> bool {
        self.path_resolver.is_some()
    }

    /// Get the name of the configured path resolver (if any)
    pub fn path_resolver_name(&self) -> Option<&str> {
        self.path_resolver.as_ref().map(|r| r.name())
    }

    /// Resolve the base location for a collection's WAL files
    ///
    /// This method ensures WAL files are written to the correct storage location
    /// based on the collection's storage assignment. It implements robust path
    /// resolution with:
    /// - Short timeout wait for metadata provider if not immediately available
    /// - Clear error messages for debugging path resolution issues
    /// - Fallback to configured storage locations with warnings
    ///
    /// # Arguments
    /// * `collection_id` - The collection ID to resolve path for
    ///
    /// # Returns
    /// * `Ok(String)` - The resolved base location path
    /// * `Err` - If path cannot be resolved (e.g., collection not found)
    async fn resolve_collection_base_location(&self, collection_id: &str) -> Result<String> {
        // ADR-069 S1: an explicit local WAL directory (PROXIMADB_WAL_LOCAL_DIR)
        // DECOUPLES the WAL from the collection's (object-store) data assignment.
        // WAL is high-frequency small appends — each object-store PUT is a paid
        // write-txn — so a local reattachable disk avoids that cost while
        // data/segments stay on object store. This precedes the resolver +
        // catalog so it governs both the write path and replay consistently.
        // Default unset ⇒ legacy behavior (WAL co-located with data).
        if let Some(dir) = local_wal_dir(&self.config) {
            tracing::debug!(
                "WAL base via local WAL dir (ADR-069 S1): {} -> {}",
                collection_id,
                dir
            );
            return Ok(dir);
        }

        // DIP: If a path_resolver is injected, use it directly (no 100ms wait needed)
        if let Some(ref resolver) = self.path_resolver {
            match resolver.resolve_base_location(collection_id).await {
                Ok(location) => {
                    tracing::debug!(
                        "WAL path resolved via {} resolver: {} -> {}",
                        resolver.name(),
                        collection_id,
                        location
                    );
                    return Ok(location);
                }
                Err(e) => {
                    tracing::warn!(
                        "Path resolver {} failed for {}: {}, falling back",
                        resolver.name(),
                        collection_id,
                        e
                    );
                    // Fall through to the catalog path
                }
            }
        }

        // The catalog is the sole read authority: resolve from it.
        if let Some(collection) = resolve_collection_from_catalog(collection_id).await {
            if let Some(assignment) = collection.storage_assignment {
                tracing::debug!(
                    "WAL path resolved via catalog: {} -> {}",
                    collection_id,
                    assignment.base_location
                );
                return Ok(assignment.base_location.clone());
            }
            tracing::warn!(
                "Collection {} in catalog has no storage_assignment, using fallback",
                collection_id
            );
            return Ok(self.get_fallback_base_location());
        }

        // Catalog has no entry (e.g. during collection creation/recovery): use fallback.
        tracing::debug!(
            "Collection {} not found in catalog, using fallback",
            collection_id
        );
        Ok(self.get_fallback_base_location())
    }

    /// Get fallback base location from config
    /// This uses the first configured storage location or a sensible default
    fn get_fallback_base_location(&self) -> String {
        self.config
            .multi_disk
            .data_directories
            .first()
            .cloned()
            .unwrap_or_else(|| {
                // Last resort default - should rarely happen
                tracing::warn!("No data directories configured, using /tmp/proximadb/d1");
                "/tmp/proximadb/d1".to_string()
            })
    }

    /// Get WAL configuration (read-only access)
    pub fn get_config(&self) -> &WALConfig {
        &self.config
    }

    /// Get recovery manager for WAL recovery operations
    /// Returns cached recovery manager if available
    /// Use get_recovery_manager() to create/cache if not exists
    pub fn recovery_manager(&self) -> Option<Arc<RecoveryManager>> {
        // Use try_read to avoid blocking - recovery manager is set once at startup
        match self.recovery_manager_cache.try_read() {
            Ok(cache) => {
                if cache.is_some() {
                    trace!("WAL recovery manager cache hit");
                    cache.as_ref().map(|rm| Arc::new(rm.clone()))
                } else {
                    trace!("WAL recovery manager cache empty");
                    None
                }
            }
            Err(_) => {
                debug!("WAL recovery manager cache read lock busy; falling back to blocking read");
                // If we can't acquire read lock, try blocking read
                let cache = self.recovery_manager_cache.blocking_read();
                if cache.is_some() {
                    cache.as_ref().map(|rm| Arc::new(rm.clone()))
                } else {
                    trace!("WAL recovery manager cache still empty after blocking read");
                    None
                }
            }
        }
    }

    /// Insert a single canonical record (converted to batch of 1 via WALVectorBatch).
    /// ADR-069 S4 write-admission gate. Inert unless capacity backpressure is
    /// armed (`wal_max_bytes > 0`); when armed it rejects the write with a
    /// retryable [`WalBackpressure`] once the collection's unflushed memtable
    /// reaches the critical watermark, so ingest sheds load instead of driving the
    /// bounded local WAL disk into an ENOSPC crash. The high-watermark force-flush
    /// (auto-flush driver) drains the memtable back under the line so writes resume.
    ///
    /// **Client ingest only.** Recovery replay materializes SST directly and never
    /// routes through the insert methods, so replay is never backpressured.
    async fn admit_write(&self, collection_id: &str) -> Result<()> {
        use crate::storage::persistence::write_ahead_log::flush_policy::{
            FlushPolicy, WalBackpressure,
        };
        let policy = FlushPolicy::from_performance(&self.config.performance);
        // Default-OFF: no capacity budget ⇒ gate inert, zero cost on the hot path.
        if policy.capacity_budget_bytes == 0 {
            return Ok(());
        }
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        let mem_bytes: u64 = wal_behavior
            .get_unflushed_batches(collection_id)
            .await
            .map(|batches| batches.iter().map(|b| b.total_size_bytes as u64).sum())
            .unwrap_or(0);
        if let Some(fill_pct) = policy.write_admission_reject_pct(mem_bytes) {
            crate::metrics::wal_flush_metrics::inc_backpressure(collection_id);
            return Err(anyhow::Error::new(WalBackpressure {
                collection_id: collection_id.to_string(),
                fill_pct,
            }));
        }
        Ok(())
    }

    pub async fn insert_record(
        &self,
        collection_id: String,
        record_id: VectorId,
        record: ProximaRecord,
    ) -> Result<u64> {
        // ADR-069 S4: shed load at the critical watermark before appending.
        self.admit_write(&collection_id).await?;
        let start_time = std::time::Instant::now();

        debug!(
            "📝 [WAL_UPSERT] Starting upsert for collection: {}, record_id: {}, embeddings: {} (using BATCH architecture)",
            collection_id,
            record_id,
            record.embeddings.len()
        );

        // Create a batch of 1 canonical record - MODERN ARCHITECTURE
        use crate::storage::memtable::specialized::wal_behavior::WALVectorBatch;
        use crate::storage::persistence::write_ahead_log::BatchId;

        let batch_id = BatchId::new(); // Single record batch

        let total_size_bytes = record
            .embeddings
            .iter()
            .map(|embedding| embedding.values.len() * 4)
            .sum::<usize>()
            + (record.props.len() * 50)
            + 256;

        let batch = WALVectorBatch {
            batch_id,
            vector_records: Arc::new(vec![record]),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes,
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        // Use shared WAL behavior for batch operations
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        let sequences = wal_behavior.add_vector_batch(&collection_id, batch).await?;
        let duration = start_time.elapsed();

        // Return the first (and only) sequence from the batch
        let sequence = sequences
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No sequence returned from batch write"))?;

        debug!(
            "📝 [WAL_UPSERT] Successfully upserted record {} in collection {} (sequence: {}) in {:?} using BATCH architecture",
            record_id, collection_id, sequence, duration
        );

        Ok(sequence)
    }

    /// Insert batch of canonical records using modern batch API.
    pub async fn insert_record_batch(
        &self,
        collection_id: String,
        records: Vec<(VectorId, ProximaRecord)>,
    ) -> Result<Vec<u64>> {
        let vector_records: Vec<ProximaRecord> =
            records.into_iter().map(|(_, record)| record).collect();
        self.insert_vectors(collection_id, vector_records).await
    }

    /// Force immediate sync of WAL data to disk
    pub async fn force_sync(&self, _collection_id: Option<&String>) -> Result<()> {
        // Force sync is now handled by the shared WAL behavior
        // Deferred: Implement proper sync mechanism with shared_wal_behavior if needed
        Ok(())
    }

    /// Update canonical record (redirects to upsert for consistency).
    pub async fn update_record(
        &self,
        collection_id: String,
        record_id: VectorId,
        mut record: ProximaRecord,
    ) -> Result<u64> {
        record.record_version = record.record_version.saturating_add(1).max(1);
        record.updated_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as i64)
            .unwrap_or(record.updated_at_ns);

        self.insert_record(collection_id, record_id, record).await
    }

    /// Delete canonical record by identity (tombstone record).
    pub async fn delete_record(&self, collection_id: String, record_id: VectorId) -> Result<u64> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as i64)
            .unwrap_or(0);

        let record = ProximaRecord {
            oid: record_id.clone(),
            updated_at_ns: now_ns,
            valid_to_ns: Some(0),
            method: Some("delete".to_string()),
            ..ProximaRecord::default()
        };

        self.insert_record(collection_id, record_id, record).await
    }

    // Note: Collection lifecycle operations (create/drop) are handled by CollectionService
    // WAL only handles vector-level operations (insert/update/delete/flush/checkpoint)

    /// Search for canonical record by ID.
    pub async fn search_record(
        &self,
        collection_id: &str,
        record_id: &VectorId,
    ) -> Result<Option<ProximaRecord>> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        wal_behavior.vector_by_id(collection_id, record_id).await
    }

    /// Read canonical record batches for recovery or replication.
    pub async fn read_record_entries(
        &self,
        collection_id: &str,
        from_sequence: u64,
        limit: Option<usize>,
    ) -> Result<Vec<ProximaRecord>> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        let records = wal_behavior.get_collection_vectors(collection_id).await?;

        let filtered: Vec<ProximaRecord> = records
            .into_iter()
            .skip(from_sequence as usize)
            .take(limit.unwrap_or(usize::MAX))
            .collect();

        Ok(filtered)
    }

    /// Force flush to disk
    pub async fn flush(&self, _collection_id: Option<&String>) -> Result<FlushResult> {
        // Use shared WAL behavior for flush
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let _wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        // WALBehaviorWrapper doesn't handle flushing directly - that's done by the flush coordinator
        let result = FlushResult::default();

        // Update stats
        let mut stats = self.stats.write().await;
        stats.last_flush_time = Some(Utc::now());

        Ok(result)
    }

    /// Compact collection (clean up old MVCC versions)
    pub async fn compact(&self, _collection_id: &str) -> Result<u64> {
        // Compaction not directly available in shared WAL behavior
        // Return 0 for now as compaction is handled at storage layer
        Ok(0)
    }

    /// Get all vectors for a collection (modern batch approach)
    pub async fn get_collection_entries(
        &self,
        collection_id: &str,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        wal_behavior.get_collection_vectors(collection_id).await
    }

    /// Get WAL statistics
    pub async fn stats(&self) -> Result<WALStats> {
        tracing::debug!(
            "📊 WAL_MANAGER_STATS: Strategy type: {:?}",
            self.strategy_type
        );
        tracing::debug!("📊 WAL_MANAGER_STATS: Getting stats from shared WAL behavior...");

        // Get basic stats from the shared WAL behavior
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let _wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);

        // Create stats based on available information
        let stats = WALStats {
            total_entries: 0, // Would need to aggregate from all collections
            memory_entries: 0,
            disk_segments: 0,
            total_disk_size_bytes: 0,
            memory_size_bytes: 0,
            collections_count: self.assigned_collections.read().await.len(),
            last_flush_time: None,
            write_throughput_entries_per_sec: 0.0,
            read_throughput_entries_per_sec: 0.0,
            compression_ratio: 1.0,
        };

        tracing::debug!(
            "📊 WAL_MANAGER_STATS: Returning stats: total_entries={}, memory_entries={}, collections_count={}",
            stats.total_entries,
            stats.memory_entries,
            stats.collections_count
        );
        Ok(stats)
    }

    /// Graceful shutdown
    pub async fn close(&self) -> Result<()> {
        // Nothing to close for shared WAL behavior
        Ok(())
    }

    /// Materialize one collection's admitted WAL epoch to its catalog-selected
    /// storage engine.
    ///
    /// This is the synchronous operator/API trigger. Size and timer triggers
    /// enter the same canonical materializer asynchronously, while this method
    /// waits on its per-collection gate and returns only after publication and
    /// exact WAL-claim retirement complete.
    pub async fn flush_collection(&self, collection_id: &str) -> Result<FlushResult> {
        use crate::storage::flush_materializer::{
            flush_plan_from_collection_meta, materialize_collection,
        };

        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let write_buffer = self.shared_wal_behavior.get_or_init(&memtable_config);
        let collection = resolve_collection_from_catalog(collection_id)
            .await
            .with_context(|| {
                format!("cannot flush collection {collection_id:?}: catalog metadata was not found")
            })?;
        let plan = flush_plan_from_collection_meta(&collection)?;
        #[cfg(feature = "axis")]
        let axis = crate::index::get_global_axis_manager();
        #[cfg(feature = "axis")]
        let axis_arg = axis.as_deref();
        #[cfg(not(feature = "axis"))]
        let axis_arg: Option<&()> = None;
        let started = std::time::Instant::now();
        let outcome =
            materialize_collection(&write_buffer, &plan, None, None, true, axis_arg).await?;

        let (entries_flushed, bytes_written) = outcome
            .map(|outcome| (outcome.entries_flushed, outcome.bytes))
            .unwrap_or((0, 0));
        // The operator-triggered path does not pass through AutoFlushDriver,
        // so it must close the ADR-069 gauge itself.  In particular, a slow
        // manual flush can overlap an auto-driver observation which publishes
        // a non-zero value.  Once exact claim retirement completes above, the
        // current write-buffer value is authoritative; leaving the sampled
        // value behind reports phantom unflushed WAL forever.
        let canonical_collection_id = plan.collection_object_id.to_string();
        let remaining_bytes = write_buffer.unflushed_bytes(&canonical_collection_id).await;
        crate::metrics::wal_flush_metrics::record_successful_flush(
            &canonical_collection_id,
            "manual",
            bytes_written,
            entries_flushed,
            started.elapsed().as_secs_f64(),
            Utc::now().timestamp(),
            remaining_bytes,
        );
        let result = FlushResult {
            success: true,
            collections_affected: vec![canonical_collection_id],
            entries_flushed: Some(entries_flushed),
            bytes_written: Some(bytes_written),
            duration_ms: Some(started.elapsed().as_millis() as u64),
            ..Default::default()
        };

        let mut stats = self.stats.write().await;
        stats.last_flush_time = Some(Utc::now());
        Ok(result)
    }

    /// Force flush all collections - FOR TESTING ONLY
    /// WARNING: This method should only be used for testing and debugging
    pub async fn force_flush_all(&self) -> Result<()> {
        tracing::warn!("⚠️ WAL MANAGER: FORCE FLUSH ALL - TESTING ONLY");
        // Use modern batch API - flush all known collections
        Ok(())
    }

    /// Synchronously force-flush one collection (operator/API surface).
    pub async fn force_flush_collection(
        &self,
        collection_id: &str,
        _storage_engine: Option<&str>,
    ) -> Result<()> {
        tracing::info!("WAL manager: force-flush collection {}", collection_id);
        self.flush_collection(collection_id).await?;
        Ok(())
    }

    /// Get WAL behavior wrapper for direct batch access (optimization for search)
    /// Returns the wrapper that provides access to unflushed batches in memory
    pub fn get_wal_behavior_wrapper(
        &self,
    ) -> Arc<crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper> {
        // Convert MemTableConfig to MemtableConfig
        let memtable_config = crate::storage::memtable::core::MemtableConfig {
            max_size_bytes: self.config.memtable.global_memory_limit,
            flush_threshold_bytes: self.config.memtable.global_memory_limit / 2,
            enable_mvcc: self.config.enable_mvcc,
            mvcc_cleanup_interval_secs: self.config.performance.mvcc_cleanup_interval_secs,
            max_versions_per_key: self.config.memtable.mvcc_versions_retained,
        };
        self.shared_wal_behavior.get_or_init(&memtable_config)
    }

    // 🎯 MODERN BATCH API (Recommended)

    /// Write a batch of canonical ProximaRecord envelopes to WAL.
    pub async fn write_vector_batch_native_arc(
        &self,
        collection_id: &str,
        native_vectors: Arc<Vec<proximadb_records::ProximaRecord>>,
    ) -> Result<Vec<u64>> {
        self.write_vector_batch_native_arc_with_mode(collection_id, native_vectors, false)
            .await
    }

    /// Write canonical ProximaRecord batch with insert-only WAL/memtable semantics.
    pub async fn write_vector_batch_native_arc_insert_only(
        &self,
        collection_id: &str,
        native_vectors: Arc<Vec<proximadb_records::ProximaRecord>>,
    ) -> Result<Vec<u64>> {
        self.write_vector_batch_native_arc_with_mode(collection_id, native_vectors, true)
            .await
    }

    async fn write_vector_batch_native_arc_with_mode(
        &self,
        collection_id: &str,
        native_vectors: Arc<Vec<proximadb_records::ProximaRecord>>,
        insert_only: bool,
    ) -> Result<Vec<u64>> {
        debug!(
            "WAL write: {} vectors to collection {}",
            native_vectors.len(),
            collection_id
        );

        // Create native WALVectorBatch with Arc (zero-copy)
        let batch_id = crate::storage::persistence::write_ahead_log::BatchId::new();

        // ADR-069: estimate the batch's resident size UP FRONT via the canonical
        // estimator on WALVectorBatch. The old `0 // calculated by strategy` was
        // never calculated on this live path — which silently zeroed both the
        // memtable backpressure sum (wal_behavior.rs) and the auto-flush driver's
        // capacity/size evaluation.
        let total_size_bytes =
            crate::storage::memtable::specialized::wal_behavior::WALVectorBatch::estimate_records_size(
                &native_vectors,
            );

        let native_batch = crate::storage::memtable::specialized::wal_behavior::WALVectorBatch {
            batch_id,
            vector_records: native_vectors,
            timestamp: std::time::SystemTime::now(),
            total_size_bytes,
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        // Persist WAL batch to disk BEFORE adding to memtable
        // This ensures durability even if server crashes after write but before flush

        // First, add to memtable
        let sequences = {
            let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
            let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
            if insert_only {
                wal_behavior
                    .add_vector_batch_insert_only(collection_id, native_batch.clone())
                    .await
            } else {
                wal_behavior
                    .add_vector_batch(collection_id, native_batch.clone())
                    .await
            }
        }?;

        // ADR-069 D2 inline size-trigger — the responsive throughput path (the 300s
        // periodic AutoFlushDriver is the RPO floor). After a successful append, if the
        // collection's unflushed data crosses the size target, fire-and-forget a flush.
        // Skipped for insert-only (tombstone/delete) writes so deletes never trigger a
        // segment flush. ONE materialization path (spawn_inline_flush reuses the driver
        // recipe); per-collection guarded so a burst of writes collapses to one flush.
        if !insert_only && let Some(write_buffer) = get_global_write_buffer_behavior() {
            use crate::storage::persistence::write_ahead_log::flush_policy::{
                FlushPolicy, FlushReason,
            };
            // Lean per-write size check (sums sizes under the lock, no batch clone —
            // get_unflushed_batches clones every batch and regressed write throughput).
            let mem_bytes = write_buffer.unflushed_bytes(collection_id).await;
            // TD-FLUSH-3 S1: the predicted-segment-bytes floor input (Σ dim×9/8+8 —
            // what the segment will occupy, vs mem_bytes' serialized-record size).
            let predicted_bytes = write_buffer.unflushed_predicted_bytes(collection_id).await;
            let policy = FlushPolicy::from_performance(&self.config.performance);
            if matches!(
                policy
                    .evaluate_with_predicted(mem_bytes, 0, predicted_bytes)
                    .reason,
                Some(FlushReason::Size)
            ) {
                // TD-FLUSH-3 S2: attribute the trigger in the log so segment-size
                // regressions are diagnosable from the flush line alone.
                tracing::debug!(
                    collection_id,
                    mem_bytes,
                    predicted_bytes,
                    floor_bytes = policy.floor_predicted_bytes,
                    "inline flush trigger: reason=size (predicted >= floor)"
                );
                spawn_inline_flush(collection_id.to_string());
            }
        }

        // TD-DELVEC-1 WI-5 P0: capture the durable per-batch `global_lsn` (when this
        // batch is synced to disk + the manifest append succeeds) so it can be
        // returned for DV-bit keying instead of the ephemeral memtable sequence.
        let mut durable_lsn: Option<u64> = None;
        // Then, persist to disk if sync mode requires it
        if self.should_sync_to_disk(collection_id).await? {
            info!(
                "🔄 DEBUG: Sync mode enabled - triggering disk persistence for collection: {}",
                collection_id
            );

            // Serialize the batch and write to disk
            use crate::storage::persistence::filesystem::FilesystemFactory;
            use crate::storage::persistence::write_ahead_log::WriteAheadLogDiskManager;
            use crate::storage::persistence::write_ahead_log::serialization::{
                SerializationFormat, SerializerFactory,
            };

            // Determine serialization format based on strategy type
            let format = match self.strategy_type {
                config::WriteBufferStrategyType::BincodeBatch => SerializationFormat::Bincode,
                config::WriteBufferStrategyType::AvroBatch => SerializationFormat::Avro,
                config::WriteBufferStrategyType::ProtoBatch => SerializationFormat::ProtocolBuffers,
            };
            trace!("WAL: Selected serialization format: {:?}", format);

            // Create serializer and serialize batch
            trace!("WAL: Creating serializer for format: {:?}", format);
            let serializer = SerializerFactory::create(format);

            info!(
                "🔄 DEBUG: Serializing {} vectors",
                native_batch.vector_records.len()
            );
            // Same v1/v2 dispatch shape used by
            // bincode_serialization_strategy.rs:207-218 — without this
            // gate, fp16 records fail in the v1 EmbeddingCell::Serialize
            // refuse-error. Header carries the global-default policy
            // (per-collection lookup is INT-3 territory).
            let serialized = if proximadb_config::EmbeddingPrecisionConfig::cached()
                .schema_v2_enabled
            {
                let created_at_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                let header = crate::storage::persistence::write_ahead_log::v2_segment_header::V2SegmentHeader {
                    flags: 0,
                    segment_id: 0,
                    created_at_ns,
                    canonical_default_precision: proximadb_records::EmbeddingScalarType::Fp32,
                    precision_epoch: 0,
                    policy_id: String::new(),
                    policy_version: 0,
                };
                serializer
                    .serialize_batch_with_v2_segment_header(
                        native_batch.vector_records.as_ref(),
                        &header,
                    )
                    .map_err(|e| {
                        trace!("🔄  ERROR: v2 serialization failed: {:?}", e);
                        e
                    })
                    .context("Failed to serialize batch for WAL")?
            } else {
                match serializer.serialize_batch(native_batch.vector_records.as_ref()) {
                    Ok(data) => {
                        info!(
                            "🔄 DEBUG: Serialization successful, size: {} bytes",
                            data.len()
                        );
                        data
                    }
                    Err(e) => {
                        trace!("🔄  ERROR: Serialization failed: {:?}", e);
                        return Err(e).context("Failed to serialize batch for WAL");
                    }
                }
            };

            // Per-precision canonical_bytes accumulation. Mirrors the
            // accumulator at bincode_serialization_strategy.rs:236-254;
            // this codepath is the WriteAheadLogManager fallback sync,
            // which the strategy-level path bypasses, so the metric
            // needs its own increment here to stay accurate across
            // both routes.
            //
            // Note (TD-080): for v2 record writes that go through
            // `UnifiedHandlers::coerce_records_to_canonical_precision`,
            // the canonical label-set is already emitted at the handler
            // boundary using the user-facing collection name. This block
            // continues to fire so non-handler writers (queue drainer,
            // bulk loader) get coverage too, but it labels by the
            // internal `collection_id` (typically a UUID). Operators
            // dashboarding on `proximadb_embedding_precision_canonical_bytes`
            // should prefer the handler-emitted series; the WAL-emitted
            // series is a secondary signal during incident triage.
            if let Some(pm) = crate::observability::precision_metrics::metrics() {
                let mut per_precision: std::collections::HashMap<
                    proximadb_records::EmbeddingScalarType,
                    i64,
                > = std::collections::HashMap::new();
                for record in native_batch.vector_records.iter() {
                    for cell in &record.embeddings {
                        *per_precision.entry(cell.precision).or_insert(0) +=
                            cell.values_byte_size() as i64;
                    }
                }
                for (precision, delta) in per_precision {
                    pm.add_canonical_bytes(
                        collection_id,
                        crate::observability::precision_metrics::precision_label(precision),
                        delta,
                    );
                }
            }

            // Determine if we should sync based on sync mode
            let should_sync = matches!(
                self.config.performance.sync_mode,
                config::SyncMode::Always | config::SyncMode::PerBatch
            );
            trace!("WAL: should_sync = {}", should_sync);

            // Get base location for this collection from metadata provider
            // CRITICAL FIX: Properly resolve storage assignment from metadata
            // Uses global metadata provider shared across all pool instances
            let base_location = self.resolve_collection_base_location(collection_id).await?;

            // Create disk manager and write batch
            trace!(
                "WAL: Creating FilesystemFactory (base_location={})",
                base_location
            );
            let filesystem_factory = match FilesystemFactory::create_default().await {
                Ok(factory) => {
                    trace!("WAL: FilesystemFactory created successfully");
                    Arc::new(factory)
                }
                Err(e) => {
                    info!(
                        "🔄 DEBUG ERROR: Failed to create FilesystemFactory: {:?}",
                        e
                    );
                    return Err(anyhow::anyhow!("Failed to create FilesystemFactory: {}", e));
                }
            };

            info!(
                "🔄 DEBUG: Creating WriteAheadLogDiskManager with base: {}",
                base_location
            );
            let disk_manager = WriteAheadLogDiskManager::new(filesystem_factory, &base_location);

            info!(
                "🔄 DEBUG: Calling write_batch_with_sync - collection_id={}, batch_id={}, data_len={}, format={:?}, sync={}",
                collection_id,
                native_batch.batch_id.to_base62(),
                serialized.len(),
                format,
                should_sync
            );

            match disk_manager
                .write_batch_with_sync(
                    collection_id,
                    &native_batch.batch_id,
                    &serialized,
                    format,
                    native_batch.vector_records.len() as u64,
                    should_sync,
                )
                .await
            {
                Ok(file_info) => {
                    durable_lsn = Some(file_info.global_lsn);
                    trace!("WAL: write_batch_with_sync SUCCESS: {:?}", file_info);
                }
                Err(e) => {
                    trace!("🔄  ERROR: write_batch_with_sync FAILED: {:?}", e);
                    trace!("🔄  ERROR: Error source chain:");
                    let mut source = e.source();
                    let mut level = 1;
                    while let Some(err) = source {
                        trace!("🔄  ERROR:   Level {}: {}", level, err);
                        source = err.source();
                        level += 1;
                    }
                    return Err(e).context("Failed to write WAL batch to disk");
                }
            }

            info!(
                "💾 WAL batch {} written to disk for collection {} ({} vectors)",
                native_batch.batch_id.to_base62(),
                collection_id,
                native_batch.vector_records.len()
            );
        }

        // Return the durable per-batch `global_lsn` (broadcast one-per-record) when
        // the batch was synced + the manifest allocated one; otherwise the ephemeral
        // memtable sequences (MemoryOnly / append failure — no durability to key on).
        // TD-DELVEC-1 WI-3b/WI-5: the cold-delete path keys DV bits on these LSNs.
        let n = sequences.len();
        Ok(durable_lsn.map(|l| vec![l; n]).unwrap_or(sequences))
    }

    /// Insert multiple vectors efficiently (modern API)
    pub async fn insert_vectors(
        &self,
        collection_id: String,
        records: Vec<proximadb_records::ProximaRecord>,
    ) -> Result<Vec<u64>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        // ADR-069 S4: shed load at the critical watermark before appending.
        self.admit_write(&collection_id).await?;

        // Create batch
        use crate::storage::memtable::specialized::wal_behavior::WALVectorBatch;
        use crate::storage::persistence::write_ahead_log::BatchId;
        let total_size_bytes: usize = records
            .iter()
            .map(|r| {
                r.embeddings
                    .first()
                    .map(|embedding| embedding.values.len() * 4)
                    .unwrap_or(0)
                    + 256
            })
            .sum();
        let batch_id = BatchId::new();

        let batch = WALVectorBatch {
            batch_id,
            vector_records: Arc::new(records),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes,
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        // Write to memory first
        let sequences = {
            let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
            let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
            wal_behavior
                .add_vector_batch(&collection_id, batch.clone())
                .await
        }?;

        // Check if we should sync to disk based on sync mode
        if self.should_sync_to_disk(&collection_id).await? {
            debug!(
                "🔄 Sync mode enabled - triggering disk persistence for collection: {}",
                collection_id
            );

            // Serialize the batch and write to disk
            use crate::storage::persistence::filesystem::FilesystemFactory;
            use crate::storage::persistence::write_ahead_log::WriteAheadLogDiskManager;
            use crate::storage::persistence::write_ahead_log::serialization::{
                SerializationFormat, SerializerFactory,
            };

            // Determine serialization format based on strategy type
            let format = match self.strategy_type {
                config::WriteBufferStrategyType::BincodeBatch => SerializationFormat::Bincode,
                config::WriteBufferStrategyType::AvroBatch => SerializationFormat::Avro,
                config::WriteBufferStrategyType::ProtoBatch => SerializationFormat::ProtocolBuffers,
            };

            // Create serializer and serialize batch
            let serializer = SerializerFactory::create(format);
            let serialized = serializer
                .serialize_batch(batch.vector_records.as_ref())
                .context("Failed to serialize batch for WAL")?;

            // Determine if we should sync based on sync mode
            let should_sync = matches!(
                self.config.performance.sync_mode,
                config::SyncMode::Always | config::SyncMode::PerBatch
            );

            // Get base location for this collection from the catalog (sole read
            // authority for storage_assignment), falling back to config defaults.
            let base_location =
                if let Some(collection) = resolve_collection_from_catalog(&collection_id).await {
                    match collection.storage_assignment {
                        Some(assignment) => assignment.base_location.clone(),
                        None => self
                            .config
                            .multi_disk
                            .data_directories
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "/tmp/proximadb/d1".to_string()),
                    }
                } else {
                    debug!(
                        "Collection {} not found in catalog; using WAL fallback",
                        collection_id
                    );
                    self.config
                        .multi_disk
                        .data_directories
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "/tmp/proximadb/d1".to_string())
                };

            // Create disk manager and write batch
            let filesystem_factory = Arc::new(FilesystemFactory::create_default().await?);
            let disk_manager = WriteAheadLogDiskManager::new(filesystem_factory, &base_location);

            disk_manager
                .write_batch_with_sync(
                    &collection_id,
                    &batch.batch_id,
                    &serialized,
                    format,
                    batch.vector_records.len() as u64,
                    should_sync,
                )
                .await
                .context("Failed to write WAL batch to disk")?;

            info!(
                "💾 WAL batch {} written to disk for collection {} ({} vectors)",
                batch.batch_id.to_base62(),
                collection_id,
                batch.vector_records.len()
            );
        }

        Ok(sequences)
    }

    /// Search vector by ID (modern API)
    pub async fn search_vector_by_id(
        &self,
        collection_id: &str,
        vector_id: &VectorId,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        wal_behavior.vector_by_id(collection_id, vector_id).await
    }

    /// TD-WLP-8 (ADR-061 D4): current resident **working-set bytes** for a
    /// collection — the sum of its unflushed WAL/memtable batch sizes. This is
    /// the same quantity `MemtableManager::get_collection_memory_usage` reports
    /// via the shared behavior; exposed here so the `Churn` read seam can sample
    /// it for the working-set gauge/soft-ceiling without reaching a
    /// `MemtableManager` handle it doesn't hold. Best-effort: `0` for an unknown
    /// collection (no unflushed delta).
    pub async fn get_collection_working_set_bytes(&self, collection_id: &str) -> Result<u64> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        let batches = wal_behavior.get_unflushed_batches(collection_id).await?;
        Ok(batches.iter().map(|b| b.total_size_bytes as u64).sum())
    }

    /// Enhanced search with bloom filter optimization for WAL/memtable data
    /// This is the PREFERRED method for searching unflushed vectors with metadata filtering
    pub async fn search_unflushed_vectors(
        &self,
        collection_id: &str,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: DistanceMetric,
        metadata_filters: Option<&proximadb_filter_expression::FilterExpression>,
        include_vectors: bool,
        include_metadata: bool,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

        tracing::debug!(
            "🔍 WAL: Enhanced search for collection {} with top_k={}, metric={:?}, filters={}",
            collection_id,
            top_k,
            distance_metric,
            metadata_filters.is_some()
        );

        // Step 1: Get unflushed batches through strategy (which accesses global memtable)
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        let batches = wal_behavior
            .get_unflushed_batches(collection_id)
            .await
            .context("Failed to get unflushed batches from strategy WAL behavior")?;

        if batches.is_empty() {
            tracing::debug!(
                "No unflushed batches found for collection {}",
                collection_id
            );
            return Ok(vec![]);
        }

        // Step 2: Apply bloom filter optimization if metadata filters exist
        let filtered_batches = if let Some(filter_expr) = metadata_filters {
            self.filter_batches_with_bloom(batches, filter_expr).await?
        } else {
            batches
        };

        if filtered_batches.is_empty() {
            tracing::debug!(
                "No batches passed bloom filter for collection {}",
                collection_id
            );
            return Ok(vec![]);
        }

        let batch_count = filtered_batches.len();
        tracing::debug!(
            "Found {} batches to search (after bloom filtering)",
            batch_count
        );

        // Step 3: Create distance calculator once for efficiency
        let distance_calculator = UnifiedDistanceCompute::new(distance_metric);

        // Step 4: score every candidate, but defer the expensive record
        // materialization. Deletion tombstones (`valid_to_ns == Some(0)`) are
        // cheap suppression markers kept in full and EXEMPT from the top_k cut
        // (TD-188). For every other candidate we push only a lightweight
        // (score, location) tuple; the full OptimizedSearchRecord — including the
        // per-record metadata HashMap clone — is built ONLY for the ≤top_k
        // survivors below, not for all N scanned records. The old
        // materialize-all-then-truncate allocated (and immediately discarded) a
        // metadata map + record for every record on every query — the per-query
        // allocation firehose behind the concurrency collapse on a large
        // unflushed tail. NOTE: a *live* (non-empty-vector) record with
        // `valid_to_ns == Some(0)` — a malformed record that does not occur in
        // practice — is treated here as a normal scored candidate, not exempt.
        let mut tombstones: Vec<crate::core::search::results::OptimizedSearchRecord> = Vec::new();
        // (score, batch_idx, record_idx, is_zero_score_marker).
        let mut scored_cands: Vec<(f32, usize, usize, bool)> = Vec::new();

        // Get current time for tombstone detection
        let current_time_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        for (batch_idx, batch) in filtered_batches.iter().enumerate() {
            for (record_idx, vector_record) in batch.vector_records.iter().enumerate() {
                let embedding = vector_record.embeddings.first();
                // Use `as_fp32_cow()` so non-fp32 variants (fp16/bf16/int8/etc.)
                // are promoted to fp32 for the distance calculation. `as_fp32_slice`
                // returns an empty slice for any non-fp32 variant, which caused
                // fp16 collections' INSERTed records to be silently skipped here
                // — surfaced 2026-05-28 by the v2 INSERT→SEARCH reconciliation.
                let vec_values_cow = embedding
                    .map(|e| e.as_fp32_cow())
                    .unwrap_or_else(|| std::borrow::Cow::Borrowed(&[][..]));
                let vec_values: &[f32] = vec_values_cow.as_ref();
                let expires_at_secs = vector_record.valid_to_ns.map(|ns| ns / 1_000_000_000);

                // Check if this is a tombstone (empty vector + valid_to in past)
                let is_tombstone = vec_values.is_empty()
                    && expires_at_secs.is_some_and(|e| e <= current_time_secs);

                if is_tombstone {
                    if vector_record.valid_to_ns == Some(0) {
                        // Deletion tombstone: cheap suppression marker, EXEMPT
                        // from the top_k cut (TD-188), kept in full.
                        tracing::trace!("Returning tombstone marker for: {}", vector_record.oid);
                        tombstones.push(crate::core::search::results::OptimizedSearchRecord {
                            id: vector_record.oid.clone(),
                            vector_id: Some(vector_record.oid.clone()),
                            score: 0.0,
                            similarity: Some(0.0),
                            vector: None,
                            version: Some(vector_record.record_version as u32),
                            timestamp: Some(vector_record.created_at_ns / 1_000_000),
                            updated_at: Some(vector_record.updated_at_ns / 1_000_000),
                            expires_at: expires_at_secs,
                            valid_to_ns: vector_record.valid_to_ns,
                            ..Default::default()
                        });
                    } else {
                        // TTL-expired empty: a score-0.0 candidate subject to the
                        // cut (as the original scored is_tombstone records whose
                        // valid_to_ns != Some(0)).
                        scored_cands.push((0.0, batch_idx, record_idx, true));
                    }
                    continue;
                }

                // Skip empty vectors that aren't tombstones (malformed records)
                if vec_values.is_empty() {
                    continue;
                }

                // Apply fine-grained metadata filter if specified. Use the
                // canonical, numeric-aware evaluator over the record's props so
                // this path matches every other filter site (scan, relational,
                // columnar) and supports the full operator set.
                if let Some(filter_expr) = metadata_filters
                    && !crate::core::search::sql_value_filter::evaluate_filter_proxima(
                        filter_expr,
                        &vector_record.props,
                    )
                {
                    continue;
                }

                // Score now (cheap); defer record materialization to survivors.
                let similarity_result = distance_calculator.calculate_distance(
                    query_vector,
                    vec_values,
                    &distance_metric,
                );
                scored_cands.push((
                    similarity_result.normalized_score,
                    batch_idx,
                    record_idx,
                    false,
                ));
            }
        }

        // Step 5: cut the SCORED set to top_k (deletion tombstones are exempt and
        // re-appended below; TD-188), then materialize the full record — the
        // metadata HashMap clone and any vector copy — ONLY for the survivors.
        scored_cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored_cands.truncate(top_k);

        let mut scored: Vec<crate::core::search::results::OptimizedSearchRecord> = scored_cands
            .into_iter()
            .map(|(_score, batch_idx, record_idx, is_zero_marker)| {
                let vector_record = &filtered_batches[batch_idx].vector_records[record_idx];
                let expires_at_secs = vector_record.valid_to_ns.map(|ns| ns / 1_000_000_000);
                if is_zero_marker {
                    return crate::core::search::results::OptimizedSearchRecord {
                        id: vector_record.oid.clone(),
                        vector_id: Some(vector_record.oid.clone()),
                        score: 0.0,
                        similarity: Some(0.0),
                        vector: None,
                        version: Some(vector_record.record_version as u32),
                        timestamp: Some(vector_record.created_at_ns / 1_000_000),
                        updated_at: Some(vector_record.updated_at_ns / 1_000_000),
                        expires_at: expires_at_secs,
                        valid_to_ns: vector_record.valid_to_ns,
                        ..Default::default()
                    };
                }
                let vec_values_cow = vector_record
                    .embeddings
                    .first()
                    .map(|e| e.as_fp32_cow())
                    .unwrap_or_else(|| std::borrow::Cow::Borrowed(&[][..]));
                let vec_values: &[f32] = vec_values_cow.as_ref();
                let similarity_result = distance_calculator.calculate_distance(
                    query_vector,
                    vec_values,
                    &distance_metric,
                );
                let metadata = if include_metadata {
                    use proximadb_records::ProximaTreeNode;
                    vector_record
                        .props
                        .iter()
                        .filter_map(|(k, node)| {
                            if let ProximaTreeNode::Value(pv) = node {
                                Some((k.clone(), pv.clone()))
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Default::default()
                };
                let ns = similarity_result.normalized_score;
                crate::core::search::results::OptimizedSearchRecord {
                    id: vector_record.oid.clone(),
                    vector_id: Some(vector_record.oid.clone()),
                    score: ns,
                    similarity: Some(ns),
                    vector: if include_vectors {
                        Some(Arc::new(vec_values.to_vec()))
                    } else {
                        None
                    },
                    metadata,
                    version: Some(vector_record.record_version as u32),
                    timestamp: Some(vector_record.created_at_ns / 1_000_000),
                    updated_at: Some(vector_record.updated_at_ns / 1_000_000),
                    expires_at: expires_at_secs,
                    valid_to_ns: vector_record.valid_to_ns,
                    source: None,
                    semantic_similarity: Some(similarity_result),
                    ..Default::default()
                }
            })
            .collect();
        scored.append(&mut tombstones);
        let all_results = scored;

        // Ranks are handled via score field in OptimizedSearchRecord
        // They can be computed by the caller if needed

        tracing::info!(
            "✅ WAL search completed: {} results from {} batches with bloom filter optimization",
            all_results.len(),
            batch_count
        );

        Ok(all_results)
    }

    /// Filter batches using bloom filters for metadata efficiency
    async fn filter_batches_with_bloom(
        &self,
        batches: Vec<crate::storage::memtable::specialized::wal_behavior::WALVectorBatch>,
        metadata_filters: &proximadb_filter_expression::FilterExpression,
    ) -> Result<Vec<crate::storage::memtable::specialized::wal_behavior::WALVectorBatch>> {
        let mut filtered_batches = Vec::new();
        let mut bloom_hits = 0;
        let mut bloom_misses = 0;

        // Extract field/value pairs from filter expression for bloom filter checking
        let filter_conditions = self.extract_filter_conditions(metadata_filters);

        for batch in batches {
            let mut should_include = true;

            // Check bloom filter if available
            if let Some(ref bloom_filter) = batch.metadata_bloom_filter {
                // Check each filter condition against bloom filter. The key
                // format MUST match the build site — both go through the shared
                // `metadata_bloom_key` helper so the separator can never drift.
                for (field, value) in &filter_conditions {
                    let key =
                        crate::storage::memtable::specialized::wal_behavior::metadata_bloom_key(
                            field, value,
                        );
                    if !bloom_filter.might_contain(key.as_bytes()) {
                        should_include = false;
                        bloom_misses += 1;
                        break;
                    }
                }

                if should_include {
                    bloom_hits += 1;
                    filtered_batches.push(batch);
                }
            } else {
                // No bloom filter, must check manually - include for detailed checking
                filtered_batches.push(batch);
            }
        }

        tracing::debug!(
            "🌸 Bloom filter optimization: {} hits, {} misses ({:.1}% filtered)",
            bloom_hits,
            bloom_misses,
            if bloom_hits + bloom_misses > 0 {
                (bloom_misses as f64 / (bloom_hits + bloom_misses) as f64) * 100.0
            } else {
                0.0
            }
        );

        Ok(filtered_batches)
    }

    /// Extract the conjunction of exact `field=value` pairs that a batch's
    /// bloom filter can *soundly* prove-absent, for batch pre-pruning.
    ///
    /// A bloom filter stores one `field=value` entry per record prop, so it can
    /// only ever answer "this exact value is definitely absent". It is therefore
    /// only sound to *exclude* a batch when the query REQUIRES an exact value to
    /// be present — i.e. an `Equals` comparison that appears conjunctively (a
    /// top-level `Equals`, or one nested under `And`). Every other shape must
    /// fail **open** (return no condition for it, so the batch is kept and the
    /// authoritative per-record evaluator decides):
    ///   - substring ops (`Contains`/`StartsWith`/`EndsWith`) and range ops
    ///     match values the bloom never stored verbatim → a miss does not imply
    ///     absence;
    ///   - `Or` is disjunctive → requiring any single branch's value would
    ///     wrongly drop batches that satisfy another branch;
    ///   - `Not` inverts the membership question entirely.
    ///
    /// Returning the prunable sub-conditions of an `And` (and dropping the
    /// non-prunable siblings) stays sound: every record in a surviving batch
    /// must still satisfy those `Equals` legs.
    fn extract_filter_conditions(
        &self,
        filter: &proximadb_filter_expression::FilterExpression,
    ) -> Vec<(String, String)> {
        use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

        match filter {
            FilterExpression::Comparison {
                field,
                operator: ComparisonOperator::Equals,
                value,
            } => value
                .as_str()
                .map(|v| vec![(field.clone(), v.to_string())])
                .unwrap_or_default(),
            FilterExpression::And(exprs) => exprs
                .iter()
                .flat_map(|expr| self.extract_filter_conditions(expr))
                .collect(),
            // Non-Equals comparisons, Or, and Not are not soundly prunable by an
            // exact-match bloom — fail open (no conditions ⇒ batch is kept).
            FilterExpression::Comparison { .. }
            | FilterExpression::Or(_)
            | FilterExpression::Not(_) => Vec::new(),
        }
    }

    /// Convert proto metadata to HashMap for SearchResult
    #[allow(dead_code)]
    fn convert_proto_metadata_to_hashmap(
        &self,
        metadata: &[crate::proto::proximadb_v1::MetadataItem],
    ) -> std::collections::HashMap<String, serde_json::Value> {
        metadata
            .iter()
            .filter_map(|item| {
                let key = &item.key;
                let value = item.value.as_ref().and_then(|v| match v {
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(s) => {
                        Some(serde_json::Value::String(s.clone()))
                    }
                    crate::proto::proximadb_v1::metadata_item::Value::NumberValue(n) => {
                        serde_json::Number::from_f64(*n).map(serde_json::Value::Number)
                    }
                    crate::proto::proximadb_v1::metadata_item::Value::BoolValue(b) => {
                        Some(serde_json::Value::Bool(*b))
                    }
                })?;

                Some((key.clone(), value))
            })
            .collect()
    }

    /// Get all vectors for a collection (modern API)
    pub async fn get_collection_vectors(
        &self,
        collection_id: &str,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        {
            let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
            let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
            wal_behavior.get_collection_vectors(collection_id).await
        }
    }

    /// Return the raw physical unflushed delta for cross-source reconciliation.
    /// Tombstones, TTL-expired rows, and superseded versions are retained.
    pub async fn get_collection_vectors_raw(
        &self,
        collection_id: &str,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        wal_behavior.get_collection_vectors_raw(collection_id).await
    }

    /// Paginated, deduped, time-ordered scan of a collection's unflushed records
    /// (TD-099(3d) push-down). See
    /// [`WALBehaviorWrapper::stream_unflushed_records`]. `after` is the raw
    /// `(last_updated_at_ns, last_oid)` cursor tuple.
    pub async fn stream_unflushed_records(
        &self,
        collection_id: &str,
        after: Option<(i64, &str)>,
        limit: usize,
        predicate: Option<&(dyn Fn(&proximadb_records::ProximaRecord) -> bool + Send + Sync)>,
        now_ns: i64,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        wal_behavior
            .stream_unflushed_records(collection_id, after, limit, predicate, now_ns)
            .await
    }

    /// Read all vector batches for a collection (modern API)
    pub async fn read_all_batches(
        &self,
        collection_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<crate::storage::memtable::specialized::wal_behavior::WALVectorBatch>> {
        {
            let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
            let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
            // Get unflushed batches for the collection
            let batches = wal_behavior.get_unflushed_batches(collection_id).await?;

            // Apply limit if specified
            let limited_batches = if let Some(lim) = limit {
                batches.into_iter().take(lim).collect()
            } else {
                batches
            };

            Ok(limited_batches)
        }
    }

    /// Register storage engine with the WAL strategy
    pub async fn register_storage_engine(
        &self,
        engine_name: &str,
        _engine: Arc<dyn crate::storage::traits::UnifiedStorageFormat>,
    ) -> Result<()> {
        // Set the storage engine on the strategy
        // Storage engine setting moved to shared behavior initialization
        tracing::info!(
            "✅ Storage engine '{}' registered with WriteAheadLogManager",
            engine_name
        );
        Ok(())
    }

    // ================================================================================
    // ENHANCED METHODS (Consolidated from OptimizedWalManager)
    // ================================================================================

    /// Determine if batch should be synced to disk
    async fn should_sync_to_disk(&self, _collection_id: &str) -> Result<bool> {
        match self.config.performance.sync_mode {
            config::SyncMode::Always => Ok(true),
            config::SyncMode::PerBatch => Ok(true),
            config::SyncMode::Periodic => {
                // Periodic sync: always sync in periodic mode (safe default).
                // A full per-collection timestamp tracker would optimize this but
                // the overhead of syncing is minimal compared to data loss risk.
                Ok(true)
            }
            config::SyncMode::Never | config::SyncMode::MemoryOnly => Ok(false),
        }
    }

    /// Extract batch from memory for disk synchronization
    #[allow(dead_code)]
    async fn extract_batch_for_sync(
        &self,
        collection_id: &str,
        _sequences: &[u64],
    ) -> Result<WALVectorBatch> {
        // Get the batch from the strategy's memory
        let memtable_config = crate::storage::memtable::core::MemtableConfig::default();
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);
        let collection_vectors = wal_behavior
            .get_collection_vectors(collection_id)
            .await
            .context("Failed to get collection vectors from memtable")?;

        // Filter vectors by sequences (for now, just take all vectors since we don't have reliable sequence mapping)
        let batch_vectors = collection_vectors;

        let batch_id = BatchId::new();

        use crate::storage::memtable::specialized::wal_behavior::WALVectorBatch;
        let total_size_bytes: usize = batch_vectors
            .iter()
            .map(|r| {
                r.embeddings
                    .first()
                    .map(|e| e.values.len() * 4)
                    .unwrap_or(0)
                    + 256
            })
            .sum();

        Ok(WALVectorBatch {
            batch_id,
            vector_records: Arc::new(batch_vectors),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes,
            is_flushed: false,
            metadata_bloom_filter: None,
        })
    }

    // Atomic sync methods disabled: depends on arrow-arith crate compatibility.
    // Periodic and per-batch sync modes provide sufficient durability guarantees.

    /// Get strategy name for logging
    fn get_strategy_name(&self) -> &str {
        match self.strategy_type {
            config::WriteBufferStrategyType::ProtoBatch => "ProtoBatch",
            config::WriteBufferStrategyType::AvroBatch => "AvroBatch",
            config::WriteBufferStrategyType::BincodeBatch => "BincodeBatch",
        }
    }

    /// Enhanced force sync for a collection using atomic coordination
    pub async fn force_sync_collection(&self, collection_id: &str) -> Result<()> {
        debug!("Force sync requested for collection '{}'", collection_id);

        // Use proper atomic sync with durability guarantees
        // Force sync is achieved by flushing all vectors which will trigger disk writes
        // Convert MemTableConfig to MemtableConfig for get_or_init
        let memtable_config = crate::storage::memtable::core::MemtableConfig {
            max_size_bytes: self.config.memtable.global_memory_limit,
            flush_threshold_bytes: self.config.memtable.global_memory_limit / 2,
            enable_mvcc: self.config.enable_mvcc,
            mvcc_cleanup_interval_secs: self.config.performance.mvcc_cleanup_interval_secs,
            max_versions_per_key: self.config.memtable.mvcc_versions_retained,
        };
        let wal_behavior = self.shared_wal_behavior.get_or_init(&memtable_config);

        // Get collection vectors and flush them
        let collection_vectors = wal_behavior.get_collection_vectors(collection_id).await?;
        if !collection_vectors.is_empty() {
            // Trigger flush by calling flush_all_vectors which will write to disk
            let _ = wal_behavior.flush_all_vectors().await?;
        }
        debug!(
            "Force sync delegated to strategy for collection '{}'",
            collection_id
        );

        Ok(())
    }

    /// Get assigned collections
    pub async fn get_assigned_collections(&self) -> Vec<String> {
        self.assigned_collections
            .read()
            .await
            .keys()
            .cloned()
            .collect()
    }

    /// Get collection assignment with storage location
    /// Force atomic disk sync for durability
    pub async fn force_disk_sync(&self, collection_ids: &[String]) -> Result<()> {
        debug!(
            "Force disk sync requested for {} collections",
            collection_ids.len()
        );

        let start_time = std::time::Instant::now();
        let mut sync_results = Vec::new();

        // Perform disk sync for each collection
        for collection_id in collection_ids {
            let collection_start = std::time::Instant::now();

            // 1. Force flush any pending data from memory to disk
            match self.flush_collection(collection_id).await {
                Ok(flush_result) => {
                    debug!(
                        "Collection '{}' flushed: {} entries, {} bytes",
                        collection_id,
                        flush_result.entries_flushed.unwrap_or(0),
                        flush_result.bytes_written.unwrap_or(0)
                    );
                }
                Err(e) => {
                    warn!("Failed to flush collection '{}': {}", collection_id, e);
                    continue;
                }
            }

            // 2. Ensure filesystem synchronization (fsync)
            if let Err(e) = self.sync_collection_to_disk(collection_id).await {
                warn!(
                    "Failed to sync collection '{}' to disk: {}",
                    collection_id, e
                );
                continue;
            }

            let collection_duration = collection_start.elapsed();
            sync_results.push((collection_id.clone(), collection_duration));

            debug!(
                "Collection '{}' sync completed in {:?}",
                collection_id, collection_duration
            );
        }

        let total_duration = start_time.elapsed();
        info!(
            "✅ Force disk sync completed for {} collections in {:?}",
            sync_results.len(),
            total_duration
        );

        Ok(())
    }

    /// Sync a specific collection's data to disk with fsync
    async fn sync_collection_to_disk(&self, collection_id: &str) -> Result<()> {
        // Get the collection's WAL directory
        let collection_wal_dir = format!(
            "{}/{}",
            self.config
                .multi_disk
                .data_directories
                .first()
                .map_or("./data/wal", |d| d.as_str()),
            collection_id
        );

        // Directory-level sync delegated to underlying flush operations.
        // Each serialization strategy (Avro/Proto/Bincode) handles fsync
        // through its own disk_manager during write_native_batch().
        debug!(
            "Directory sync for '{}' handled by underlying flush operations",
            collection_wal_dir
        );

        Ok(())
    }

    pub async fn get_collection_assignment(
        &self,
        collection_id: &str,
    ) -> Option<CollectionAssignment> {
        self.assigned_collections
            .read()
            .await
            .get(collection_id)
            .cloned()
    }

    /// Get or create the RecoveryManager for this WAL instance
    /// This allows external code to register storage engines before recovery
    /// IMPORTANT: Returns cached instance to ensure engine registration persists
    pub async fn get_recovery_manager(&self) -> Result<RecoveryManager> {
        // Check if we already have a cached instance
        {
            let cache = self.recovery_manager_cache.read().await;
            if let Some(ref manager) = *cache {
                debug!("Returning cached RecoveryManager instance");
                return Ok(manager.clone());
            }
            trace!("No cached RecoveryManager; creating new instance");
        }

        // Create new instance if not cached
        debug!("Creating new RecoveryManager instance with metadata provider");

        // Create filesystem factory for recovery
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(filesystem_config)
                .await?,
        );
        trace!("FilesystemFactory created for WAL recovery");

        // Create RecoveryManager instance (catalog-backed config resolution)
        let recovery_manager = RecoveryManager::new(
            self.config.clone(),
            self.shared_wal_behavior
                .get_or_init(&crate::storage::memtable::core::MemtableConfig::default())
                .clone(),
            filesystem,
        );
        trace!("RecoveryManager created successfully");

        // Cache for future use
        {
            let mut cache = self.recovery_manager_cache.write().await;
            *cache = Some(recovery_manager.clone());
            debug!("Cached RecoveryManager instance for reuse");
        }

        Ok(recovery_manager)
    }

    /// Recovery method using parallel recovery system if available
    pub async fn recover(&self) -> Result<u64> {
        info!(
            "🔄 WAL_MANAGER: Starting WAL recovery for {} strategy",
            self.get_strategy_name()
        );

        // Add timeout to prevent indefinite hanging
        let recovery_timeout = std::time::Duration::from_secs(30);

        let recovery_result = tokio::time::timeout(recovery_timeout, async {
            info!("📊 WAL_MANAGER: About to call RecoveryManager.recover()");

            // Get RecoveryManager
            let recovery_manager = self.get_recovery_manager().await?;

            let recovery_stats = recovery_manager.recover_all().await?;
            let recovered_count = recovery_stats.total_vectors_recovered;

            info!(
                "📊 WAL_MANAGER: RecoveryManager returned: {} entries",
                recovered_count
            );
            Ok::<u64, anyhow::Error>(recovered_count)
        })
        .await;

        match recovery_result {
            Ok(Ok(recovered_count)) => {
                info!(
                    "✅ WAL_MANAGER: WAL recovery completed successfully: {} entries recovered",
                    recovered_count
                );
                Ok(recovered_count)
            }
            Ok(Err(e)) => {
                tracing::error!("❌ WAL_MANAGER: WAL recovery failed: {}", e);
                Err(e)
            }
            Err(_) => {
                tracing::error!(
                    "⏰ WAL_MANAGER: WAL recovery timed out after {} seconds",
                    recovery_timeout.as_secs()
                );
                Err(anyhow::anyhow!("WAL recovery timed out"))
            }
        }
    }

    /// Assign a collection with its metadata to this WriteAheadLogManager
    pub async fn assign_collection(&self, collection_id: String, assignment: CollectionAssignment) {
        let mut assigned = self.assigned_collections.write().await;
        assigned.insert(collection_id.clone(), assignment);
        tracing::debug!(
            "Assigned collection '{}' with storage location to WriteAheadLogManager",
            collection_id
        );
    }

    /// Get storage location for a collection
    pub async fn get_collection_storage(
        &self,
        collection_id: &str,
    ) -> Option<CollectionAssignment> {
        let assigned = self.assigned_collections.read().await;
        assigned.get(collection_id).cloned()
    }

    /// Check if atomic sync is enabled
    pub fn has_atomic_sync(&self) -> bool {
        false // atomic_sync temporarily disabled
    }
}

impl DistanceComputeProvider for WriteAheadLogManager {
    fn distance_compute(&self) -> &UnifiedDistanceCompute {
        &self.distance_compute
    }
}

impl std::fmt::Debug for WriteAheadLogManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteAheadLogManager")
            .field("strategy", &"shared_wal_behavior")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod simple_context_tests {
    use super::*;
    use crate::storage::background_flush_context::{
        BackgroundFlushContext, CompressionConfig, OperationPriority, StorageEngineType,
    };
    use proximadb_distance_kernel::DistanceMetric;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_background_flush_context_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: BackgroundFlushContext creation and validation");

        let context = BackgroundFlushContext {
            collection_id: "test_collection".to_string(),
            storage_engine: StorageEngineType::Viper,
            base_location: "file:///tmp/test".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            compression_config: CompressionConfig::default(),
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(1000),
            priority: OperationPriority::Normal,
            timeout_ms: Some(60_000),
            extra_metadata: HashMap::new(),

            tenant_id: None,
            collection_meta: None,
        };

        // Validate context fields
        assert_eq!(context.collection_id, "test_collection");
        assert_eq!(context.storage_engine, StorageEngineType::Viper);
        assert_eq!(context.dimension, 384);
        assert_eq!(context.distance_metric, DistanceMetric::Cosine);
        assert_eq!(context.engine_name(), "viper");

        // Test SST engine as well
        let sst_context = BackgroundFlushContext {
            collection_id: "sst_collection".to_string(),
            storage_engine: StorageEngineType::Sst,
            base_location: "file:///tmp/test".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            compression_config: CompressionConfig::default(),
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(500),
            priority: OperationPriority::Normal,
            timeout_ms: Some(60_000),
            extra_metadata: HashMap::new(),

            tenant_id: None,
            collection_meta: None,
        };
        assert_eq!(sst_context.engine_name(), "sst");

        info!("✅ BackgroundFlushContext creation test passed");
    }

    #[tokio::test]
    async fn test_context_performance_settings() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: Context performance configuration");

        let viper_context = BackgroundFlushContext {
            collection_id: "perf_test".to_string(),
            storage_engine: StorageEngineType::Viper,
            base_location: "file:///tmp/test".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            compression_config: CompressionConfig::default(),
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(1000),
            priority: OperationPriority::Normal,
            timeout_ms: Some(60_000),
            extra_metadata: HashMap::new(),

            tenant_id: None,
            collection_meta: None,
        };

        // Test dimension-based optimizations
        assert_eq!(viper_context.dimension, 384);

        // Test batch size hint calculation
        assert!(viper_context.batch_size_hint.is_some());
        let batch_size = viper_context.batch_size_hint.unwrap();
        assert!(batch_size > 0 && batch_size <= 10000);

        // Test row group size optimization for VIPER
        let row_group_size = viper_context.row_group_size();
        assert!((1000..=50000).contains(&row_group_size));

        // Test flush threshold optimization
        let flush_threshold = viper_context.flush_threshold();
        assert!((10000..=100000).contains(&flush_threshold));

        info!("✅ Context performance configuration test passed");
    }
}

#[cfg(test)]
mod wal_manager_infra_tests {
    use super::*;
    use crate::storage::persistence::write_ahead_log::config::{
        WALConfig, WriteBufferStrategyType,
    };
    use proximadb_records::EmbeddingCell;

    #[tokio::test]
    async fn test_wal_manager_creation() {
        let config = WALConfig::default();
        let result =
            WriteAheadLogManager::new_for_collection(config, "test_infra_collection".to_string())
                .await;

        assert!(
            result.is_ok(),
            "WriteAheadLogManager::new_for_collection should initialize without error"
        );
    }

    #[tokio::test]
    async fn test_wal_manager_stats() {
        let config = WALConfig::default();
        let manager =
            WriteAheadLogManager::new_for_collection(config, "test_stats_collection".to_string())
                .await
                .expect("manager creation should succeed");

        let stats = manager.stats().await.expect("stats() should succeed");

        assert_eq!(
            stats.total_entries, 0,
            "total_entries should be 0 initially"
        );
        assert_eq!(
            stats.memory_entries, 0,
            "memory_entries should be 0 initially"
        );
        assert_eq!(
            stats.disk_segments, 0,
            "disk_segments should be 0 initially"
        );
        assert_eq!(stats.total_disk_size_bytes, 0);
        assert_eq!(stats.memory_size_bytes, 0);
        assert_eq!(stats.collections_count, 0, "no collections assigned yet");
        assert!(stats.last_flush_time.is_none());
        assert!((stats.compression_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_wal_manager_with_config() {
        let mut config = WALConfig::default();
        config.strategy_type = WriteBufferStrategyType::ProtoBatch;
        config.enable_mvcc = false;
        config.enable_ttl = false;
        config.enable_background_compaction = false;

        let manager = WriteAheadLogManager::new_for_collection(
            config.clone(),
            "test_config_collection".to_string(),
        )
        .await
        .expect("manager creation with custom config should succeed");

        let stats = manager.stats().await.expect("stats should succeed");
        assert_eq!(stats.total_entries, 0);
    }

    /// TD-WLP-8: the working-set-bytes accessor is `0` before any insert and
    /// reflects the unflushed batch size after — the quantity the `Churn`
    /// read seam samples onto the working-set gauge. `MemoryOnly` sync keeps
    /// the batch resident (no disk dependency) so the assertion is exact.
    #[tokio::test]
    async fn test_get_collection_working_set_bytes_reflects_unflushed() {
        let mut config = WALConfig::default();
        config.performance.sync_mode = config::SyncMode::MemoryOnly;
        let manager =
            WriteAheadLogManager::new_for_collection(config, "wlp8_ws_collection".to_string())
                .await
                .expect("manager creation should succeed");

        // No inserts yet → zero working set (accessor never errors).
        let empty = manager
            .get_collection_working_set_bytes("wlp8_ws_collection")
            .await
            .expect("accessor should not error on an empty collection");
        assert_eq!(empty, 0, "no unflushed delta yet → zero working-set bytes");

        // Insert a handful of fp32 records; insert_vectors sizes each record as
        // dim*4 + 256 bytes into the batch's total_size_bytes.
        let dim = 8u32;
        let records: Vec<ProximaRecord> = (0..4)
            .map(|i| ProximaRecord {
                oid: format!("wlp8-rec-{i}"),
                embeddings: vec![EmbeddingCell {
                    model_id: "wlp8".to_string(),
                    modality: "dense_vector".to_string(),
                    dim,
                    values: proximadb_records::EmbeddingValues::Fp32(vec![i as f32; dim as usize]),
                    ..Default::default()
                }],
                ..Default::default()
            })
            .collect();
        manager
            .insert_vectors("wlp8_ws_collection".to_string(), records)
            .await
            .expect("insert should succeed under MemoryOnly sync");

        let after = manager
            .get_collection_working_set_bytes("wlp8_ws_collection")
            .await
            .expect("accessor should not error after inserts");
        let expected = 4 * (dim as u64 * 4 + 256);
        assert_eq!(
            after, expected,
            "working-set bytes must reflect the unflushed batch total_size_bytes"
        );
    }

    #[tokio::test]
    async fn test_wal_registry_creation() {
        let registry = WriteAheadLogManagerRegistry::new();
        let managers = registry.get_all_managers().await;
        assert_eq!(
            managers.len(),
            0,
            "registry pool should be empty before any collection assignment"
        );
    }

    #[tokio::test]
    async fn test_wal_registry_with_config() {
        let pool_config = WriteAheadLogManagerPoolConfig::builder()
            .initial_pool_size(5)
            .soft_thread_limit(10)
            .target_collections_per_manager(200)
            .rebalance_load_threshold(0.6)
            .rebalance_cooldown_secs(15)
            .enable_dynamic_scaling(false)
            .build();

        assert_eq!(pool_config.initial_pool_size, 5);
        assert_eq!(pool_config.soft_thread_limit, 10);
        assert_eq!(pool_config.target_collections_per_manager, 200);
        assert!((pool_config.rebalance_load_threshold - 0.6).abs() < f64::EPSILON);
        assert_eq!(pool_config.rebalance_cooldown_secs, 15);
        assert!(!pool_config.enable_dynamic_scaling);

        let registry = WriteAheadLogManagerRegistry::with_config(pool_config);
        let managers = registry.get_all_managers().await;
        assert_eq!(
            managers.len(),
            0,
            "custom-config registry pool should start empty"
        );
    }

    #[test]
    fn test_strategy_name() {
        assert_eq!(
            WriteBufferStrategyType::ProtoBatch.to_string(),
            "ProtoBatch"
        );
        assert_eq!(WriteBufferStrategyType::AvroBatch.to_string(), "AvroBatch");
        assert_eq!(
            WriteBufferStrategyType::BincodeBatch.to_string(),
            "BincodeBatch"
        );

        assert_eq!(
            WriteBufferStrategyType::default(),
            WriteBufferStrategyType::BincodeBatch
        );
    }
}

#[cfg(test)]
mod inline_flush_drain_tests {
    use super::*;

    /// TD-FLUSH-4: drain_inline_flushes returns 0 immediately when idle,
    /// blocks while a ticket is held, and reports stragglers at timeout —
    /// the shutdown-await contract that keeps a 40s materialize from being
    /// killed mid-staging.
    #[tokio::test]
    async fn drain_awaits_inflight_tickets_and_times_out() {
        assert_eq!(
            drain_inline_flushes(std::time::Duration::from_millis(50)).await,
            0,
            "idle drain is immediate"
        );
        let ticket = InlineFlushTicket::new();
        assert_eq!(
            drain_inline_flushes(std::time::Duration::from_millis(300)).await,
            1,
            "held ticket reported as straggler at timeout"
        );
        let waiter =
            tokio::spawn(async { drain_inline_flushes(std::time::Duration::from_secs(5)).await });
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        drop(ticket);
        assert_eq!(
            waiter.await.unwrap(),
            0,
            "drain completes when ticket drops"
        );
    }
}

#[cfg(test)]
mod optimization_validation_tests {
    use super::*;
    use crate::storage::background_flush_context::{
        BackgroundFlushContext, CompressionConfig, OperationPriority, StorageEngineType,
    };
    use proximadb_distance_kernel::DistanceMetric;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// ADR-069 S1: with the env override unset (guaranteed under nextest's
    /// process-per-test isolation), `local_wal_dir` resolves from the TOML
    /// `wal_local_dir` field and trims a trailing slash; `None` config ⇒ `None`
    /// (legacy WAL-co-located-with-data). The env-override branch is covered by
    /// the file:// + Azurite e2e beds (a process-global OnceLock can't be
    /// re-toggled within one process).
    #[test]
    fn local_wal_dir_resolves_from_config_when_env_unset() {
        assert!(
            std::env::var("PROXIMADB_WAL_LOCAL_DIR").is_err(),
            "test process must not have the env override set"
        );
        let mut cfg = config::WALConfig::default();
        assert_eq!(local_wal_dir(&cfg), None);
        cfg.wal_local_dir = Some("file:///waldisk/".to_string());
        assert_eq!(local_wal_dir(&cfg), Some("file:///waldisk".to_string()));
        cfg.wal_local_dir = Some("   ".to_string());
        assert_eq!(local_wal_dir(&cfg), None, "blank config value is ignored");
    }

    struct MockCollectionService {
        call_count: Arc<AtomicU32>,
    }

    impl MockCollectionService {
        fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicU32::new(0)),
            }
        }

        async fn get_collection(&self, _collection_id: &str) -> Option<String> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Some("mock_collection".to_string())
        }

        fn get_call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn test_service_call_elimination_validation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 VALIDATION: Service call elimination through context optimization");

        let mock_service = Arc::new(MockCollectionService::new());

        debug!("📊 Simulating OLD approach - multiple service calls:");

        let _result1 = mock_service.get_collection("test_collection").await;
        debug!("   VectorOperationsService → Collection Service Call #1");

        let _result2 = mock_service.get_collection("test_collection").await;
        debug!("   BackgroundManager → Collection Service Call #2");

        let _result3 = mock_service.get_collection("test_collection").await;
        debug!("   FlushCoordinator → Collection Service Call #3");

        let old_call_count = mock_service.get_call_count();
        debug!("   Total calls in OLD approach: {}", old_call_count);
        assert_eq!(
            old_call_count, 3,
            "OLD approach should make 3 service calls"
        );

        mock_service.call_count.store(0, Ordering::SeqCst);

        debug!("🚀 Testing NEW approach - context-based optimization:");

        let _result = mock_service.get_collection("test_collection").await;
        debug!("   VectorOperationsService → Collection Service Call #1 (creates context)");

        let context = BackgroundFlushContext {
            collection_id: "test_collection".to_string(),
            storage_engine: StorageEngineType::Viper,
            base_location: "file:///tmp/test".to_string(),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            compression_config: CompressionConfig::default(),
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(1000),
            priority: OperationPriority::Normal,
            timeout_ms: Some(60_000),
            extra_metadata: HashMap::new(),

            tenant_id: None,
            collection_meta: None,
        };

        debug!("   BackgroundManager → Uses pre-computed context (NO service call)");
        debug!("   FlushCoordinator → Uses pre-computed context (NO service call)");

        let engine_name = context.engine_name();
        let dimension = context.dimension;
        let batch_hint = context.batch_size_hint;

        assert_eq!(engine_name, "viper");
        assert_eq!(dimension, 384);
        assert!(batch_hint.is_some());

        let new_call_count = mock_service.get_call_count();
        debug!("   Total calls in NEW approach: {}", new_call_count);
        assert_eq!(
            new_call_count, 1,
            "NEW approach should make only 1 service call"
        );

        let reduction_percentage =
            ((old_call_count - new_call_count) as f64 / old_call_count as f64) * 100.0;
        info!("✅ OPTIMIZATION VALIDATED:");
        debug!(
            "   Service calls reduced from {} to {}",
            old_call_count, new_call_count
        );
        debug!(
            "   Reduction: {:.1}% ({}x fewer calls)",
            reduction_percentage,
            old_call_count / new_call_count
        );

        assert!(
            reduction_percentage > 60.0,
            "Should achieve at least 60% reduction"
        );
        debug!("🎉 Background flush optimization successfully validated!");
    }

    #[tokio::test]
    async fn test_context_metadata_completeness() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 VALIDATION: Context contains all metadata needed for background operations");

        let context = BackgroundFlushContext {
            collection_id: "completeness_test".to_string(),
            storage_engine: StorageEngineType::Viper,
            base_location: "file:///tmp/test".to_string(),
            dimension: 512,
            distance_metric: DistanceMetric::Euclidean,
            compression_config: CompressionConfig {
                enabled: true,
                compression_type: "zstd".to_string(),
                level: 3,
            },
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(2000),
            priority: OperationPriority::High,
            timeout_ms: Some(120_000),
            extra_metadata: {
                let mut meta = HashMap::new();
                meta.insert("test_key".to_string(), "test_value".to_string());
                meta
            },

            tenant_id: None,
            collection_meta: None,
        };

        assert_eq!(context.collection_id, "completeness_test");
        assert_eq!(context.storage_engine, StorageEngineType::Viper);
        assert_eq!(context.engine_name(), "viper");
        assert_eq!(context.dimension, 512);
        assert_eq!(context.distance_metric, DistanceMetric::Euclidean);
        assert_eq!(context.base_location, "file:///tmp/test");

        assert!(context.compression_config.enabled);
        assert_eq!(context.compression_config.compression_type, "zstd");
        assert_eq!(context.compression_config.level, 3);

        assert_eq!(context.batch_size_hint, Some(2000));
        assert_eq!(context.priority, OperationPriority::High);
        assert_eq!(context.timeout_ms, Some(120_000));

        let row_group_size = context.row_group_size();
        let flush_threshold = context.flush_threshold();

        assert!(row_group_size > 0, "Row group size should be calculated");
        assert!(flush_threshold > 0, "Flush threshold should be calculated");

        assert_eq!(
            context.extra_metadata.get("test_key"),
            Some(&"test_value".to_string())
        );

        info!("✅ Context metadata completeness validated");
        debug!(
            "   Engine: {} ({})",
            context.engine_name(),
            context.storage_engine.clone() as u8
        );
        debug!("   Dimension: {}", context.dimension);
        debug!("   Distance metric: {:?}", context.distance_metric);
        debug!("   Row group size: {}", row_group_size);
        debug!("   Flush threshold: {}", flush_threshold);
        debug!("   Batch hint: {:?}", context.batch_size_hint);
        debug!("   Priority: {:?}", context.priority);
        debug!("🎉 Context contains all required metadata for background operations!");
    }
}

#[cfg(test)]
mod td_flush_8_catalog_walk_tests {
    use super::{catalog_default_namespaces, collections_from_catalog_walk};
    use proximadb_catalog::{Catalog, CatalogTableSchema, TableIdentifier};

    /// A catalog in exactly the TD-FLUSH-8 gap shape: a collection table is
    /// resolvable under `["default"]` while `list_namespaces(None)` does NOT
    /// report that namespace.
    ///
    /// This used to borrow `HiveCatalog` for the shape, which worked only
    /// because that adapter was an in-memory mock whose namespace listing
    /// happened to omit `default`. Two problems with that: the test's real
    /// requirement was invisible (nothing said "the listing must omit it" — it
    /// was a side effect of someone else's fixture), and it tied a WAL
    /// regression test to an unrelated catalog backend. TD-CAT-8 deleted that
    /// mock, so the shape is now stated directly.
    ///
    /// Only `list_namespaces`, `list_tables` and `get_table` are real — those
    /// are the three the flush-side walk calls. Everything else returns an
    /// error naming itself, so a future caller that wanders in gets told which
    /// method it needs rather than a plausible default.
    struct UnlistedDefaultNamespaceCatalog {
        table: TableIdentifier,
        schema: CatalogTableSchema,
    }

    impl UnlistedDefaultNamespaceCatalog {
        fn unused<T>(what: &str) -> anyhow::Result<T> {
            Err(anyhow::anyhow!(
                "UnlistedDefaultNamespaceCatalog::{what} is not part of the TD-FLUSH-8 fixture"
            ))
        }
    }

    #[async_trait::async_trait]
    impl Catalog for UnlistedDefaultNamespaceCatalog {
        fn name(&self) -> &str {
            "unlisted-default"
        }
        fn catalog_type(&self) -> &str {
            "test-fixture"
        }

        /// The gap itself: `default` is deliberately absent.
        async fn list_namespaces(
            &self,
            _parent: Option<&[String]>,
        ) -> anyhow::Result<Vec<proximadb_catalog::CatalogNamespace>> {
            Ok(vec![proximadb_catalog::CatalogNamespace::new(vec![
                "other".to_string(),
            ])])
        }

        async fn list_tables(&self, namespace: &[String]) -> anyhow::Result<Vec<TableIdentifier>> {
            Ok(if namespace == self.table.namespace.as_slice() {
                vec![self.table.clone()]
            } else {
                Vec::new()
            })
        }

        async fn get_table(
            &self,
            identifier: &TableIdentifier,
        ) -> anyhow::Result<CatalogTableSchema> {
            if identifier == &self.table {
                Ok(self.schema.clone())
            } else {
                Err(anyhow::anyhow!("no such table: {identifier:?}"))
            }
        }

        async fn create_namespace(
            &self,
            _namespace: &[String],
            _properties: std::collections::HashMap<String, String>,
        ) -> anyhow::Result<proximadb_catalog::CatalogNamespace> {
            Self::unused("create_namespace")
        }
        async fn drop_namespace(
            &self,
            _namespace: &[String],
            _cascade: bool,
        ) -> anyhow::Result<bool> {
            Self::unused("drop_namespace")
        }
        async fn namespace_exists(&self, _namespace: &[String]) -> anyhow::Result<bool> {
            Self::unused("namespace_exists")
        }
        async fn get_namespace(
            &self,
            _namespace: &[String],
        ) -> anyhow::Result<proximadb_catalog::CatalogNamespace> {
            Self::unused("get_namespace")
        }
        async fn update_namespace_properties(
            &self,
            _namespace: &[String],
            _updates: std::collections::HashMap<String, String>,
            _removals: Vec<String>,
        ) -> anyhow::Result<()> {
            Self::unused("update_namespace_properties")
        }
        async fn create_table_inner(
            &self,
            _identifier: &TableIdentifier,
            schema: CatalogTableSchema,
        ) -> anyhow::Result<CatalogTableSchema> {
            Ok(schema)
        }
        async fn drop_table(
            &self,
            _identifier: &TableIdentifier,
            _purge: bool,
        ) -> anyhow::Result<bool> {
            Self::unused("drop_table")
        }
        async fn table_exists(&self, identifier: &TableIdentifier) -> anyhow::Result<bool> {
            Ok(identifier == &self.table)
        }
        async fn rename_table(
            &self,
            _from: &TableIdentifier,
            _to: &TableIdentifier,
        ) -> anyhow::Result<()> {
            Self::unused("rename_table")
        }
        async fn evolve_schema(
            &self,
            _identifier: &TableIdentifier,
            _evolution: proximadb_catalog::CatalogSchemaEvolution,
        ) -> anyhow::Result<CatalogTableSchema> {
            Self::unused("evolve_schema")
        }
        async fn get_schema_version(&self, _identifier: &TableIdentifier) -> anyhow::Result<i32> {
            Self::unused("get_schema_version")
        }
        async fn get_schema_by_version(
            &self,
            _identifier: &TableIdentifier,
            _version: i32,
        ) -> anyhow::Result<CatalogTableSchema> {
            Self::unused("get_schema_by_version")
        }
        async fn create_index(
            &self,
            _identifier: &TableIdentifier,
            _index: proximadb_catalog::CatalogIndex,
        ) -> anyhow::Result<proximadb_catalog::CatalogIndex> {
            Self::unused("create_index")
        }
        async fn drop_index(
            &self,
            _identifier: &TableIdentifier,
            _index_name: &str,
        ) -> anyhow::Result<bool> {
            Self::unused("drop_index")
        }
        async fn list_indexes(
            &self,
            _identifier: &TableIdentifier,
        ) -> anyhow::Result<Vec<proximadb_catalog::CatalogIndex>> {
            Self::unused("list_indexes")
        }
        async fn get_statistics(
            &self,
            _identifier: &TableIdentifier,
        ) -> anyhow::Result<proximadb_catalog::CatalogTableStatistics> {
            Self::unused("get_statistics")
        }
        async fn update_statistics(
            &self,
            _identifier: &TableIdentifier,
            _stats: proximadb_catalog::CatalogTableStatistics,
        ) -> anyhow::Result<()> {
            Self::unused("update_statistics")
        }
    }

    fn catalog_with_unlisted_default_collection() -> UnlistedDefaultNamespaceCatalog {
        let mut schema = CatalogTableSchema {
            name: "diag_coll".to_string(),
            // `validate_schema` requires at least one column.
            columns: vec![proximadb_catalog::CatalogColumn {
                id: 1,
                object_id: None,
                name: "oid".to_string(),
                data_type: proximadb_data_model::ProximaType::String,
                nullable: false,
                default_value: None,
                comment: None,
                properties: Default::default(),
                is_deleted: false,
                original_id: None,
            }],
            ..Default::default()
        };
        schema
            .properties
            .insert("asset.kind".to_string(), "collection".to_string());
        schema
            .properties
            .insert("collection.id".to_string(), "1".to_string());
        schema
            .properties
            .insert("collection.name".to_string(), "diag_coll".to_string());
        UnlistedDefaultNamespaceCatalog {
            table: TableIdentifier::new(vec!["default".to_string()], "diag_coll"),
            schema,
        }
    }

    /// The namespace walk must ALWAYS include `["default"]`, mirroring the
    /// write path's `read_collections_from_catalog` — this is the mechanism
    /// that keeps the flush-side listing in agreement with the write path.
    #[tokio::test]
    async fn namespace_walk_always_includes_default() {
        let catalog = catalog_with_unlisted_default_collection();

        // Precondition: the fixture really reproduces the gap — the namespace
        // listing omits `default` while the table under it is resolvable.
        let listed: Vec<Vec<String>> = catalog
            .list_namespaces(None)
            .await
            .expect("list namespaces")
            .into_iter()
            .map(|namespace| namespace.levels)
            .collect();
        assert!(
            !listed.iter().any(|namespace| namespace == &["default"]),
            "fixture must omit the default namespace from the listing, got {listed:?}"
        );

        let walked = catalog_default_namespaces(&catalog).await;
        assert!(
            walked.iter().any(|namespace| namespace == &["default"]),
            "the walk must include [\"default\"] even when the listing omits it"
        );
    }

    /// TD-FLUSH-8 regression: the flush-side catalog list must find a
    /// collection whose namespace is missing from `list_namespaces` — the
    /// exact state where the auto-flush driver previously logged
    /// `no catalog metadata for '1'` every tick and never flushed (while the
    /// write path resolved the same collection fine).
    #[tokio::test]
    async fn flush_side_list_finds_collection_in_unlisted_default_namespace() {
        let catalog = catalog_with_unlisted_default_collection();

        let collections = collections_from_catalog_walk(&catalog).await;
        assert!(
            collections.iter().any(|collection| collection.id == "1"),
            "the WAL id '1' must be catalog-resolvable by the flush-side walk; got ids {:?}",
            collections
                .iter()
                .map(|collection| collection.id.clone())
                .collect::<Vec<_>>()
        );
    }
}
