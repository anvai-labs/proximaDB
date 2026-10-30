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

//! SST Engine Core Module
//!
//! Contains the main SstEngine struct definition and initialization logic.
//! This module is responsible for:
//! - SstEngine struct definition with all components
//! - Engine construction and configuration
//! - Component initialization and coordination
//! - Core trait implementations

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::compute::quantization::{
    quantization_engine::{CodebookStore, InMemoryCodebookStore, UnifiedQuantizationEngine},
    storage_engine::StorageQuantizationEngine,
};
use crate::storage::engines::core::ops::{
    UniversalOptimizationStrategy, UniversalPerformanceOptimizer,
};
use crate::storage::engines::sst::{
    SstConfig, SstError, compaction::Compaction, decompression_cache, readers::UnifiedSstableReader,
};
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use crate::storage::persistence::filesystem::{FileSystem, FilesystemFactory};
use crate::storage::transaction_coordinator::TransactionCoordinator;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
use proximadb_records::ProximaRecord;

// AXIS manager for HNSW/IVF index integration
use proximadb_index_traits::IndexEngine;
use std::sync::OnceLock;

/// Global AXIS manager singleton for SST engine search optimization
static GLOBAL_SST_AXIS_MANAGER: OnceLock<Arc<dyn IndexEngine>> = OnceLock::new();

/// Register a global AXIS manager for SST engine to use in searches
pub fn set_sst_axis_manager(axis_manager: Arc<dyn IndexEngine>) {
    let _ = GLOBAL_SST_AXIS_MANAGER.set(axis_manager);
}

/// Get the global AXIS manager if registered
pub fn get_sst_axis_manager() -> Option<Arc<dyn IndexEngine>> {
    GLOBAL_SST_AXIS_MANAGER.get().cloned()
}

/// TD-CACHE-1: process-global handles to the warm-tier caches, registered at
/// engine arming so the boot warm task and the shutdown manifest emitter
/// (database.rs) can reach them without threading engine handles. Same
/// OnceLock pattern as the AXIS registry above; None until an engine arms.
static GLOBAL_WARM_TIER_CACHES: std::sync::OnceLock<(
    Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>,
    Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>,
)> = std::sync::OnceLock::new();

/// Register the armed warm-tier caches (first engine wins — the server builds
/// one SST engine; tests building more are no-ops here).
pub fn set_warm_tier_caches(
    invariants: Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>,
    survivor: Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>,
) {
    let _ = GLOBAL_WARM_TIER_CACHES.set((invariants, survivor));
}

/// The registered warm-tier caches, if any engine has armed them.
pub fn get_warm_tier_caches() -> Option<(
    Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>,
    Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>,
)> {
    GLOBAL_WARM_TIER_CACHES.get().cloned()
}

/// Reclaim all immutable PAX acceleration bytes below a successfully retired
/// collection directory. No cache is a valid no-op (env-unset behavior).
pub async fn purge_warm_tier_prefix(path_prefix: &str) -> usize {
    // TD-DELVEC-1 C3: invalidate the OID resolver cache entries under this prefix
    // (collection-drop) — BEFORE the warm_caches guard (the resolver cache is
    // independent of the warm-tier pair). Mirrors C2's compaction-retire placement.
    #[cfg(feature = "cold-deletion-vectors")]
    if let Some(cache) = get_global_oid_resolver_cache() {
        cache.invalidate_prefix(path_prefix);
    }
    let Some((invariants, survivor)) = get_warm_tier_caches() else {
        return 0;
    };
    let invariant_entries = invariants.invalidate_prefix_all(path_prefix).await;
    invariant_entries + survivor.purge_prefix(path_prefix).await
}

/// TD-DELVEC-1 WI-3c / C2: process-global OID resolver cache (sibling to
/// `GLOBAL_WARM_TIER_CACHES`). Hoisted to module scope so compaction's retire
/// path (`purge_retired_segment_cache_entries`) can invalidate entries for
/// compacted-away segments via `get_global_oid_resolver_cache()`. `None` inner =
/// cache disabled (`PROXIMADB_OID_RESOLVER_CACHE_MB == 0`).
#[cfg(feature = "cold-deletion-vectors")]
static GLOBAL_OID_RESOLVER_CACHE: std::sync::OnceLock<
    Option<Arc<crate::storage::engines::sst::oid_resolver_cache::OidResolverCache>>,
> = std::sync::OnceLock::new();

/// The process-global OID resolver cache, if an engine armed one (mb > 0).
/// Borrowed (not cloned) — `invalidate` only needs `&self`.
#[cfg(feature = "cold-deletion-vectors")]
pub fn get_global_oid_resolver_cache()
-> Option<&'static Arc<crate::storage::engines::sst::oid_resolver_cache::OidResolverCache>> {
    GLOBAL_OID_RESOLVER_CACHE.get().and_then(Option::as_ref)
}

/// Test-only setter (the production path uses the constructor's `get_or_init`).
/// Set-once; benign across tests (invalidate is miss-safe on unique paths).
#[cfg(all(test, feature = "cold-deletion-vectors"))]
pub fn set_global_oid_resolver_cache_for_tests(
    cache: Arc<crate::storage::engines::sst::oid_resolver_cache::OidResolverCache>,
) {
    let _ = GLOBAL_OID_RESOLVER_CACHE.set(Some(cache));
}

// Global PCA model cache for Z-Order spatial encoding
// Uses lazy_static for thread-safe initialization
lazy_static::lazy_static! {
    /// Global PCA model cache for SST engine Z-Order spatial encoding
    /// Key: collection_id, Value: PCA model
    static ref GLOBAL_PCA_MODEL_CACHE: std::sync::RwLock<std::collections::HashMap<String, super::pca_manager::EnhancedPCAModel>> =
        std::sync::RwLock::new(std::collections::HashMap::new());
}

/// Register a PCA model for a collection in the global cache
/// This is called during flush when a new PCA model is trained.
pub fn set_collection_pca_model(collection_id: &str, model: super::pca_manager::EnhancedPCAModel) {
    if let Ok(mut cache) = GLOBAL_PCA_MODEL_CACHE.write() {
        cache.insert(collection_id.to_string(), model);
        tracing::debug!("[SST] Cached PCA model for collection: {}", collection_id);
    }
}

/// Get the PCA model for a collection from the global cache
/// Returns None if no model is cached for the collection.
pub fn get_collection_pca_model(
    collection_id: &str,
) -> Option<super::pca_manager::EnhancedPCAModel> {
    if let Ok(cache) = GLOBAL_PCA_MODEL_CACHE.read() {
        cache.get(collection_id).cloned()
    } else {
        None
    }
}

/// SST Engine - Hybrid columnar (ProximaBlocks), write-optimized storage with three-stage filtering
///
/// # Architecture Overview
///
/// The SST (Sorted String Table) engine implements an LSM-tree based storage system
/// optimized for OLTP workloads with real-time query requirements. It uses a multi-stage
/// filtering pipeline to minimize I/O and maximize query performance.
///
/// ## Design Principles
///
/// - **Row-Oriented Storage**: Each vector record stored as complete row for fast point queries
/// - **Three-Stage Filtering**: Progressive elimination (bloom → quantized → full precision)
/// - **Write Optimization**: LSM-tree with MemTable → SSTable → Compaction flow
/// - **Singleton Pattern**: Single engine instance handles multiple collections efficiently
/// - **Lock-Free Reads**: Concurrent reads without blocking using Arc-based sharing
///
/// ## Data Flow
///
/// 1. **Write Path**: Records → WAL → MemTable → Flush → L0 SSTables → Background Compaction
/// 2. **Read Path**: Bloom Filter → Quantized Scan → Full Vector Retrieval → Distance Compute
/// 3. **Compaction**: Multi-level merge using transaction coordinator for atomicity
///
/// ## Performance Characteristics
///
/// - **Write Latency**: ~1-5ms (MemTable insertion + WAL)
/// - **Point Query**: ~2-10ms (bloom filter + 1-2 SSTable reads)
/// - **Range Query**: ~10-50ms (multiple SSTable scans with quantization)
/// - **Memory Overhead**: ~100MB base + (MemTable size × num_collections)
///
pub struct SstEngine {
    /// **Engine Configuration**
    ///
    /// Contains tuning parameters for:
    /// - MemTable size thresholds (default: 64MB)
    /// - Compaction strategy (size-tiered or leveled)
    /// - Bloom filter false positive rate (default: 1%)
    /// - Cache sizes and eviction policies
    ///
    /// Loaded from config/config.toml or defaults
    config: SstConfig,

    /// **Compaction Manager** (Optional)
    ///
    /// Background compaction orchestrator that:
    /// - Monitors SSTable count and triggers merges
    /// - Implements size-tiered compaction strategy
    /// - Manages compaction scheduling and resources
    /// - Handles tombstone cleanup and space reclamation
    ///
    /// None during initialization, Some after start_compaction() called
    compaction_manager: Option<Arc<Compaction>>,

    /// TD-COMPACT-8: per-collection `training_in_flight` guard. When a TD-COMPACT-5
    /// training compaction is running for a collection, its `storage_url` is inserted
    /// here. `should_trigger_compaction` arm 2 checks this set and skips re-arming
    /// while a training pass is in-flight — eliminating the redundant re-training loop
    /// (up to N-1 wasted cycles per ingest batch, each costing N GETs + PCA + PUT +
    /// DELETEs on object storage). The flag is set by the flush caller before
    /// `enqueue_due_compaction` and cleared by the background worker once the
    /// compaction completes (the flush caller also clears it on the no-worker
    /// paths: nothing-due `Ok(false)`, or enqueue `Err`).
    pub(crate) training_in_flight: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,

    /// **Filesystem Factory**
    ///
    /// Creates filesystem instances for different storage backends:
    /// - Local filesystem (file://)
    /// - S3 (s3://)
    /// - Azure Blob (azure://)
    /// - GCS (gs://)
    ///
    /// Shared across all filesystem operations for consistency
    filesystem: Arc<FilesystemFactory>,

    /// **Unified Caching Filesystem** (Optional)
    ///
    /// Wraps base filesystem with intelligent caching layer:
    /// - Transparent read-through disk cache
    /// - Prefetch engine for sequential access
    /// - Metadata caching for file stats
    /// - LRU eviction for memory management
    ///
    /// None until initialized, then Some for SSTable I/O
    unified_fs: Option<Arc<dyn FileSystem>>,

    /// **Transaction Coordinator**
    ///
    /// Ensures atomic flush and compaction operations:
    /// - ACID guarantees for SSTable writes
    /// - Two-phase commit for compaction
    /// - Crash recovery using WAL
    /// - Prevents torn writes during system failures
    ///
    /// Always present, wraps filesystem with transactional semantics
    #[allow(dead_code)]
    atomic_coordinator: Arc<TransactionCoordinator>,

    /// **SSTable Reader** (Shared)
    ///
    /// Unified reader for all SSTable formats:
    /// - Parses SSTable headers and index blocks
    /// - Performs bloom filter checks
    /// - Reads and decompresses data blocks
    /// - Handles both legacy and modern formats
    ///
    /// Shared across collections to reduce memory overhead
    sstable_reader: Arc<UnifiedSstableReader>,

    /// **Distance Computation Engine**
    ///
    /// Hardware-accelerated distance calculations:
    /// - Auto-detects SIMD capabilities (AVX2/AVX512/NEON)
    /// - Supports multiple metrics (L2, cosine, dot product)
    /// - Batch processing for throughput optimization
    /// - Fallback to scalar for unsupported architectures
    ///
    /// Shared singleton for all distance operations
    distance_compute: Arc<UnifiedDistanceCompute>,

    /// **Decompression Cache** (Shared)
    ///
    /// LRU cache for decompressed SSTable blocks:
    /// - Caches frequently accessed blocks (hot data)
    /// - Adaptive sizing based on access patterns
    /// - Reduces CPU overhead from repeated decompression
    /// - Memory bounded with configurable limits
    ///
    /// Shared across all collections for better hit rates
    decompression_cache: Arc<decompression_cache::DecompressionCache>,

    /// **Storage Quantization Engine** (Collection-Aware)
    ///
    /// Persistent quantization with collection-specific codebooks:
    /// - Stores PQ codebooks in filesystem
    /// - Trains once, reuses across queries
    /// - Supports binary, INT8, PQ4/8/16/32
    /// - Automatically selects best quantization level
    ///
    /// Used for consistent quantization across flush/search
    storage_quantization_engine: Arc<StorageQuantizationEngine>,

    /// **Fallback Quantization Engine** (Stateless)
    ///
    /// In-memory quantization for ad-hoc queries:
    /// - No persistent codebooks needed
    /// - Faster for one-off quantization
    /// - Uses k-means++ clustering
    /// - Falls back when codebook unavailable
    ///
    /// Used when storage engine doesn't have trained codebooks
    fallback_quantization_engine: Arc<UnifiedQuantizationEngine>,

    /// **Universal Performance Optimizer**
    ///
    /// Dynamic query optimization system:
    /// - Analyzes query patterns and data distribution
    /// - Selects optimal execution strategy
    /// - Adaptive index selection (bloom, quantized, full)
    /// - Cost-based decision making
    ///
    /// Updates strategy based on runtime metrics
    universal_optimizer: UniversalPerformanceOptimizer,

    /// **Cross-Cache Orchestrator** (Optional)
    ///
    /// Coordinates metadata and filter caches:
    /// - Invalidates dependent caches on updates
    /// - Propagates filter pushdowns to storage
    /// - Tracks cache dependencies and relationships
    /// - Manages cache memory budgets
    ///
    /// None if caching disabled, Some in production
    orchestrator: Option<Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>>,

    /// **AXIS Manager** (Optional)
    ///
    /// Integration with AXIS indexing service:
    /// - Provides HNSW index for fast approximate search
    /// - Enables IVF-based partition pruning
    /// - Supports hybrid vector + metadata search
    /// - Coordinated with query optimizer
    ///
    /// Initialized from global singleton or set explicitly
    axis_manager: Option<Arc<dyn IndexEngine>>,

    /// **PCA Model Cache** (Per-Collection)
    ///
    /// Cached PCA models for Z-Order spatial encoding:
    /// - Key: collection_id
    /// - Value: Trained PCA model for that collection
    /// - Trained during flush, reused during search
    /// - Eliminates per-query PCA training overhead (40ms+ saved)
    ///
    /// Models are persisted to `{collection_dir}/__model/pca_model.bin`
    pca_model_cache: Arc<
        tokio::sync::RwLock<
            std::collections::HashMap<String, super::pca_manager::EnhancedPCAModel>,
        >,
    >,

    /// **Vector Object Economy Directory Cache** (Optional, Phase 4)
    ///
    /// Process-wide per-collection cache populated by the read-side and
    /// invalidated by the writer/compactor. When `Some`, the SST engine's
    /// flush implementation emits a directory entry after each
    /// successful atomic commit and calls `cache.invalidate(collection_id)`
    /// so the next reader sees the new file.
    ///
    /// When `None` (default), directory emission is skipped entirely —
    /// no path derivation, no fallback inference. Pre-existing call
    /// sites are unaffected until they opt in via
    /// [`Self::with_directory_cache`].
    directory_cache: Option<
        Arc<
            crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache,
        >,
    >,
    /// PR2: per-segment invariants cache (header+region+footer). Hot queries
    /// skip the 3 read_range calls → 3 GETs → 0.
    pub(crate) segment_invariants_cache:
        Option<Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>>,

    /// TD-DELVEC-1 WI-3c: per-segment OID→position resolver cache. Feature-gated
    /// `cold-deletion-vectors`. Empty on boot; lazy-filled by the resolve path
    /// (`read_resolver` on cache miss). Compaction invalidates on retire.
    #[cfg(feature = "cold-deletion-vectors")]
    pub(crate) oid_resolver_cache:
        Option<Arc<crate::storage::engines::sst::oid_resolver_cache::OidResolverCache>>,

    /// TD-DELVEC-1 WI-3a-remaining-A: durable, CAS'd per-segment deletion-vector
    /// store. Feature-gated `cold-deletion-vectors`. Per-engine (authoritative
    /// owned state, not a shared cache), keyed by segment path; lazy-filled on
    /// first touch + reload. Wired end-to-end (WI-3b delete, WI-4 merge-on-read,
    /// WI-5 recovery reconcile, WI-6 compaction).
    #[cfg(feature = "cold-deletion-vectors")]
    pub(crate) deletion_vector_store:
        Option<Arc<crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore>>,

    /// ADR-065 Q3: ranged RAM cache for survivor (Region B SQ8) + OID (Region D)
    /// byte ranges. Hot repeat queries skip the per-range `fs.read_range` GETs.
    /// Configured from the shared warm-tier policy at construction and
    /// replaceable through [`Self::with_warm_tier_caches`] for controlled
    /// benchmarks and explicit dependency injection.
    pub(crate) survivor_cache:
        Option<Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>>,

    /// **Tiering Integration** (Optional)
    ///
    /// Drives access-pattern → tier-migration decisions for SST segments.
    /// When `Some`, the engine:
    /// - Calls `record_access()` from the search path so the policy engine
    ///   sees hot vs cold collections
    /// - Calls `determine_flush_tier()` so newly-flushed segments land in
    ///   the right tier from the start
    /// - Calls `evaluate_collection()` from the compaction path so merged
    ///   segments can be re-tiered based on the post-compaction set
    ///
    /// When `None` (default), all these calls become no-ops — preserving
    /// the legacy "everything is hot" behavior until an operator opts in
    /// via [`Self::with_tiering_integration`] and config enables it.
    tiering_integration: Option<
        Arc<crate::storage::engines::sst::tiering_integration::SstTieringIntegration>,
    >,

    /// **Vector Object Economy Freshness Source** (Optional, Phase 5)
    ///
    /// Provides the "current committed LSN" used as the directory's
    /// `freshness_watermark_lsn` at emission time. When `None`, the
    /// emit path uses `0` — a conservative value that forces strong-route
    /// readers to always re-scan the WAL delta. Wired by
    /// `SharedServices::new` with a `WalCursorLsnSource` that delegates
    /// to the global manifest service.
    freshness_lsn_source: Option<
        Arc<dyn crate::storage::engines::sst::object_economy_directory::FreshnessLsnSource>,
    >,

    /// DIP storage-URL resolver (optional). When set, `get_collection_storage_url`
    /// resolves the collection's real `base_location` through it (catalog → config
    /// fallback chain); the `/data/collections/{id}` placeholder is retired.
    /// Mirrors the WAL's `path_resolver` + this struct's `freshness_lsn_source`.
    /// Wired by `SharedServices::new`.
    pub(crate) path_resolver:
        Option<Arc<dyn crate::storage::trait_components::path_resolver::CollectionPathResolver>>,

    /// ADR-081 regression hook: rejected preflights must not reach the
    /// vector-proportional sort boundary.
    #[cfg(test)]
    pub(crate) flush_sort_invocations: std::sync::atomic::AtomicU64,
}

impl SstEngine {
    /// Create a new SST engine instance (stateless)
    ///
    /// Collection info comes from FlushParameters and StorageQueryContext at runtime.
    /// The engine is designed as a singleton that can handle multiple collections.
    pub async fn new() -> Result<Self> {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await?);
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        Self::new_with_config(config, filesystem, distance_compute).await
    }

    /// Create SST engine with specific configuration
    ///
    /// This method allows for custom configuration of the SST engine components.
    /// Used internally and for testing with specific configurations.
    pub async fn new_with_config(
        config: SstConfig,
        filesystem: Arc<FilesystemFactory>,
        distance_compute: Arc<UnifiedDistanceCompute>,
    ) -> Result<Self> {
        info!("🌲 Creating SST storage engine (collection-agnostic singleton)");

        // Create atomic coordinator for safe operations
        let atomic_coordinator = Arc::new(
            TransactionCoordinator::new(filesystem.clone(), None)
                .await
                .map_err(|e| {
                    SstError::Internal(format!("Failed to create atomic coordinator: {}", e))
                })?,
        );

        // Create UnifiedCachingFilesystem for transparent cloud storage support
        let base_fs = filesystem
            .get_filesystem("file://")
            .map_err(|e| SstError::Internal(format!("Failed to get base filesystem: {}", e)))?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            String::new(), // Empty collection_id for singleton
            "sst".to_string(),
        ));

        // Create SSTable reader - using empty collection_id as SST is now singleton
        let sstable_reader = Arc::new(UnifiedSstableReader::new(
            filesystem.clone(),
            unified_fs.clone(),
            String::new(), // Empty collection_id for singleton
        ));

        // Initialize quantization engines
        let (storage_quantization_engine, fallback_quantization_engine) =
            Self::initialize_quantization_engines(distance_compute.clone()).await?;

        // Initialize decompression cache with default size (64MB)
        let decompression_cache = Arc::new(decompression_cache::DecompressionCache::new(
            64 * 1024 * 1024, // Default to 64MB cache
        ));

        // Register cache providers with orchestrator
        let orchestrator = Self::register_cache_providers(decompression_cache.clone()).await?;

        // Initialize compaction manager (always enabled) — TD-COMPACT-6 D0: start
        // background workers on THIS instance (the one the flush path uses via
        // self.compaction_manager()). Without this, schedule_compaction enqueues to
        // a workerless queue — the 2 workers at engine.rs:212 run on a DIFFERENT
        // Compaction instance (the StorageEngine's), leaving the SstEngine's
        // task_queue with zero consumers. Wire the worker count to the existing
        // SstConfig.background_thread_count (was hardcoded 2 at engine.rs:212;
        // the config field existed but was unused at the spawn site).
        //
        // TD-COMPACT-6 D1: the training_in_flight guard is created HERE and
        // shared (same Arc) between this SstEngine and its Compaction, so the
        // background workers can clear a collection's entry once the async
        // compaction completes (re-arming the flush path's training trigger).
        // It MUST be linked before start_workers, which clones the handle into
        // each worker task.
        let training_in_flight: Arc<std::sync::Mutex<std::collections::HashSet<String>>> =
            Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        // TD-COMPACT-6 D1: construct via with_atomic_coordinator so the
        // background workers carry the SAME atomic coordinator the flush path
        // uses — preserving the ADR-046 LSN-coherent atomic-swap publish.
        let mut compaction =
            Compaction::with_atomic_coordinator(config.clone(), Some(atomic_coordinator.clone()))
                .await
                .map_err(|e| {
                    SstError::Internal(format!("Failed to create compaction manager: {}", e))
                })?;
        compaction.set_training_in_flight(training_in_flight.clone());
        // Wrap in Arc BEFORE start_workers: start_workers takes `self: &Arc<Self>`
        // so each worker can clone the Arc and reuse THIS persistent instance
        // (warmed reader/compactor/factory) instead of building a cold
        // `temp_manager` per task.
        let compaction: Arc<Compaction> = Arc::new(compaction);
        compaction
            .start_workers(config.background_thread_count as usize)
            .await
            .map_err(|e| {
                SstError::Internal(format!(
                    "Failed to start {} compaction workers: {}",
                    config.background_thread_count, e
                ))
            })?;
        let compaction_manager = Some(compaction);

        // Initialize universal performance optimization
        let universal_optimizer =
            UniversalPerformanceOptimizer::with_strategy(UniversalOptimizationStrategy::Balanced)
                .await
                .map_err(|e| {
                    SstError::Internal(format!(
                        "Failed to create universal performance optimizer: {}",
                        e
                    ))
                })?;

        // Get AXIS manager from global singleton if available
        let axis_manager = get_sst_axis_manager();
        if axis_manager.is_some() {
            info!("🔗 SST Engine: AXIS manager integration enabled (HNSW/IVF indexes available)");
        }

        // TD-CACHE-4: ARM the warm-tier caches at construction. The builder
        // injection boundary below had ZERO production callers, so both caches
        // stayed `None` on the live serving path and every novel query re-paid
        // the full GET chain (the
        // ADR-065 #32-35 warm tier was dead code; found when the TD-METRICS-1
        // hit/miss counters scraped 0 across a 920-query 1M run).
        //
        // Budgets are config-first (`SstConfig.segment_invariants_cache_mb` =
        // 256 MB / `survivor_cache_mb` = 1024 MB — the 2026-07-24 sizing
        // decision that also updated ADR-065's survivor default-OFF), env
        // override wins, 0 disables either.
        // PROCESS-GLOBAL singletons: the engine is constructed once per
        // collection/location, but the warm tier is one budget pool per
        // process — N instances would multiply the configured budget by N and
        // leave the boot-warm/shutdown-emit hooks (which see one registered
        // pair) pointed at a cache no query ever touches.

        // TD-RDSTRAT-8 / COGS: initialize the coarse-probe (IVF) settings ONCE
        // from the TOML config (`SstConfig.coarse_probe` →
        // `[storage.sst_config.coarse_probe]`). Env overrides apply at the call
        // sites in segment_format.rs (probe/nprobe) + block_cluster.rs (write-train).
        crate::storage::engines::sst::segment_format::init_coarse_probe_settings(
            config.coarse_probe.as_ref(),
        );

        static SHARED_LOCAL_DISK_STORE: std::sync::OnceLock<
            Option<Arc<proximadb_cache::PersistentByteStore>>,
        > = std::sync::OnceLock::new();
        let local_disk_store = SHARED_LOCAL_DISK_STORE
            .get_or_init(|| {
                let path = std::env::var("PROXIMADB_CACHE_LOCAL_DISK_PATH")
                    .ok()
                    .filter(|path| !path.trim().is_empty())
                    .or_else(|| config.cache_local_disk_path.clone());
                let max_gb = std::env::var("PROXIMADB_CACHE_LOCAL_DISK_MAX_GB")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(config.cache_local_disk_max_gb);
                path.and_then(|path| {
                    if path.contains("://") {
                        tracing::warn!(
                            "persistent PAX cache disabled: local-disk path must be a filesystem path"
                        );
                        return None;
                    }
                    if max_gb == 0 {
                        tracing::warn!(
                            "persistent PAX cache disabled: local-disk budget is zero GiB"
                        );
                        return None;
                    }
                    let root = std::path::PathBuf::from(path).join("pax-warm-tier-v1");
                    let max_bytes = max_gb.saturating_mul(1024 * 1024 * 1024);
                    match proximadb_cache::PersistentByteStore::open(root, max_bytes) {
                        Ok(store) => {
                            // TD-TIER-1 SC4-1: register this singleton as the
                            // process residency store — the reservation
                            // registry's budget bridge then arms its protection
                            // budget (a late registration picks up the current
                            // cell, so construction order does not matter).
                            // First registration wins: a second engine
                            // construction in-process keeps the original.
                            if proximadb_cache::register_process_store(Arc::new(store.clone())) {
                                // TD-TIER-1 SC4-4: grants restored at startup
                                // registered their groups before this store
                                // existed — backfill what they cover now.
                                crate::storage::engines::sst::segment_io::spawn_warm_prefixes(
                                    filesystem.clone(),
                                    proximadb_cache::process_protection_prefixes(),
                                );
                            } else {
                                tracing::debug!(
                                    "TD-TIER-1 SC4-1: a persistent store was already \
                                     registered for this process; keeping the original"
                                );
                            }
                            Some(Arc::new(store))
                        }
                        Err(error) => {
                            tracing::warn!(
                                "persistent PAX cache disabled: unable to open local-disk tier: {error}"
                            );
                            None
                        }
                    }
                })
            })
            .clone();

        static SHARED_INVARIANTS_CACHE: std::sync::OnceLock<
            Option<Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>>,
        > = std::sync::OnceLock::new();
        let segment_invariants_cache = SHARED_INVARIANTS_CACHE
            .get_or_init(|| {
                let mb = std::env::var("PROXIMADB_SEGMENT_INVARIANTS_CACHE_MB")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(config.segment_invariants_cache_mb);
                (mb > 0).then(|| {
                    Arc::new(if let Some(store) = &local_disk_store {
                        crate::storage::engines::sst::segment_format::SegmentInvariantsCache::with_l2(
                            (mb as usize) * 1024 * 1024,
                            store.clone(),
                        )
                    } else {
                        crate::storage::engines::sst::segment_format::SegmentInvariantsCache::new(
                            (mb as usize) * 1024 * 1024,
                        )
                    })
                })
            })
            .clone();
        static SHARED_SURVIVOR_CACHE: std::sync::OnceLock<
            Option<Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>>,
        > = std::sync::OnceLock::new();
        let survivor_cache = SHARED_SURVIVOR_CACHE.get_or_init(|| {
            let mb = std::env::var("PROXIMADB_SURVIVOR_CACHE_BUDGET_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(config.survivor_cache_mb);
            (mb > 0).then(|| {
                // TD-CACHE-3 S1: per-tenant tier floors/weights via the same
                // PROXIMADB_CACHE_TIERS_PATH policy the footer/index caches use
                // (tenant→tier resolved live through the auth-stamped registry).
                let budget_bytes = mb * 1024 * 1024;
                let resolver = std::env::var("PROXIMADB_CACHE_TIERS_PATH")
                    .ok()
                    .and_then(|path| std::fs::read_to_string(&path).ok())
                    .and_then(|json| proximadb_cache::TierPolicy::from_json(&json).ok())
                    .map(|policy| {
                        let policy = std::sync::Arc::new(policy);
                        let default_tier = policy.default_tier.clone();
                        // TD-TENANT-3 S3: canonical-then-raw probe, matching
                        // the shared-services resolver exactly.
                        let probe_policy = std::sync::Arc::clone(&policy);
                        let tenant_to_tier: std::sync::Arc<
                            dyn Fn(&str) -> String + Send + Sync,
                        > = std::sync::Arc::new(move |t: &str| {
                            crate::services::record_store::tenant_tier_policy_key(t, |key| {
                                probe_policy.has_tier(key)
                            })
                            .unwrap_or_else(|| default_tier.clone())
                        });
                        policy.resolver(budget_bytes, tenant_to_tier)
                    });
                Arc::new(
                    crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache::with_resolver_and_l2(
                        budget_bytes,
                        resolver,
                        local_disk_store.clone(),
                    ),
                )
            })
        }).clone();
        if segment_invariants_cache.is_some() || survivor_cache.is_some() {
            info!(
                "🧠 SST warm-tier caches armed: invariants={} survivor={}",
                segment_invariants_cache.is_some(),
                survivor_cache.is_some()
            );
            // TD-CACHE-1: publish the handles for the boot warm task + the
            // shutdown manifest emitter.
            if let (Some(inv), Some(surv)) = (&segment_invariants_cache, &survivor_cache) {
                set_warm_tier_caches(inv.clone(), surv.clone());
            }
        }

        // TD-DELVEC-1 WI-3c: process-global OID resolver cache (sibling to the
        // invariants cache). Feature-gated; 0 disables (lazy-load on each delete).
        #[cfg(feature = "cold-deletion-vectors")]
        let oid_resolver_cache = GLOBAL_OID_RESOLVER_CACHE
            .get_or_init(|| {
                let mb = std::env::var("PROXIMADB_OID_RESOLVER_CACHE_MB")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(config.oid_resolver_cache_mb);
                (mb > 0).then(|| {
                    Arc::new(
                        crate::storage::engines::sst::oid_resolver_cache::OidResolverCache::new(
                            (mb as usize) * 1024 * 1024,
                        ),
                    )
                })
            })
            .clone();

        // TD-DELVEC-1 WI-3a-remaining-A: per-engine deletion-vector store. Not a
        // process-global singleton (unlike the resolver cache): the DV store is
        // authoritative owned state for this engine's segments, so per-engine
        // ownership is the natural model + avoids the OnceLock test hazard.
        #[cfg(feature = "cold-deletion-vectors")]
        let deletion_vector_store = Some(std::sync::Arc::new(
            crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore::new(
                filesystem.clone(),
            ),
        ));

        Ok(Self {
            config,
            compaction_manager,
            // TD-COMPACT-6 D1: the same Arc linked into the Compaction above —
            // the flush path sets/reads it, the workers clear it on completion.
            training_in_flight,
            filesystem,
            unified_fs: None, // Created per collection
            atomic_coordinator,
            sstable_reader,
            distance_compute,
            decompression_cache,
            storage_quantization_engine,
            fallback_quantization_engine,
            universal_optimizer,
            orchestrator,
            axis_manager,
            pca_model_cache: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            directory_cache: None,
            segment_invariants_cache,
            survivor_cache,
            #[cfg(feature = "cold-deletion-vectors")]
            oid_resolver_cache,
            #[cfg(feature = "cold-deletion-vectors")]
            deletion_vector_store,
            tiering_integration: None,
            freshness_lsn_source: None,
            path_resolver: None,
            #[cfg(test)]
            flush_sort_invocations: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Attach a tiering integration so flush / search / compaction hooks
    /// drive tier migration. When unset (default), all tiering hooks are
    /// no-ops — preserving the legacy single-tier behavior.
    ///
    /// Returns `self` for builder-style chaining. The integration is
    /// expected to be wrapped in `Arc` so the engine can share it across
    /// flush / search / compaction call sites without cloning state.
    pub fn with_tiering_integration(
        mut self,
        integration: Arc<crate::storage::engines::sst::tiering_integration::SstTieringIntegration>,
    ) -> Self {
        self.tiering_integration = Some(integration);
        self
    }

    /// Borrow the attached tiering integration, if any. Used by flush /
    /// search / compaction paths to dispatch hooks without panicking when
    /// tiering isn't configured.
    pub fn tiering_integration(
        &self,
    ) -> Option<&Arc<crate::storage::engines::sst::tiering_integration::SstTieringIntegration>>
    {
        self.tiering_integration.as_ref()
    }

    /// Attach the Vector Object Economy per-collection directory cache.
    /// Called by `SharedServices::new` so the engine's flush path can
    /// emit directory updates and invalidate the read-side cache after
    /// each successful atomic commit. When unset (default), directory
    /// emission is skipped.
    pub fn with_directory_cache(
        mut self,
        cache: Arc<
            crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache,
        >,
    ) -> Self {
        self.directory_cache = Some(cache);
        self
    }

    /// Replace both query-side warm tiers as one policy boundary. Passing
    /// `None` explicitly disables a tier, which keeps paired benchmarks from
    /// inheriting process-global cache state accidentally.
    pub fn with_warm_tier_caches(
        mut self,
        segment_invariants: Option<
            Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>,
        >,
        survivor: Option<
            Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>,
        >,
    ) -> Self {
        self.segment_invariants_cache = segment_invariants;
        self.survivor_cache = survivor;
        self
    }

    /// True when [`Self::with_directory_cache`] has supplied a cache.
    /// Regression-tested so the default (no emission) cannot silently
    /// flip to "always emit."
    pub fn directory_cache_configured(&self) -> bool {
        self.directory_cache.is_some()
    }

    /// Borrow the configured cache, if any. Used by the flush
    /// implementation to construct hooks at the call site.
    pub(crate) fn directory_cache_ref(
        &self,
    ) -> Option<
        &Arc<
            crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache,
        >,
    >{
        self.directory_cache.as_ref()
    }

    /// Attach the [`FreshnessLsnSource`] used to populate the directory's
    /// `freshness_watermark_lsn` at emission time. When unset (default),
    /// the engine emits `0` — making strong-route readers always re-scan
    /// the WAL delta.
    pub fn with_freshness_lsn_source(
        mut self,
        source: Arc<dyn crate::storage::engines::sst::object_economy_directory::FreshnessLsnSource>,
    ) -> Self {
        self.freshness_lsn_source = Some(source);
        self
    }

    /// Attach the catalog/config storage-URL resolver. When set,
    /// `get_collection_storage_url` resolves the real `base_location` through it
    /// (retiring the `/data/collections/{id}` placeholder). When unset, the
    /// method default-constructs a `ConfigFallbackResolver`.
    pub fn with_path_resolver(
        mut self,
        resolver: Arc<dyn crate::storage::trait_components::path_resolver::CollectionPathResolver>,
    ) -> Self {
        self.path_resolver = Some(resolver);
        self
    }

    /// True when [`Self::with_freshness_lsn_source`] has supplied a
    /// source. Used by tests/operators to confirm wiring.
    pub fn freshness_lsn_source_configured(&self) -> bool {
        self.freshness_lsn_source.is_some()
    }

    /// Borrow the configured freshness source, if any. Used by the
    /// flush implementation to resolve the watermark before constructing
    /// hooks.
    pub(crate) fn freshness_lsn_source_ref(
        &self,
    ) -> Option<&Arc<dyn crate::storage::engines::sst::object_economy_directory::FreshnessLsnSource>>
    {
        self.freshness_lsn_source.as_ref()
    }

    /// Initialize quantization engines (storage-aware and fallback)
    async fn initialize_quantization_engines(
        distance_compute: Arc<UnifiedDistanceCompute>,
    ) -> Result<(
        Arc<StorageQuantizationEngine>,
        Arc<UnifiedQuantizationEngine>,
    )> {
        // Create storage-aware quantization engine for persistent collection-based PQ
        let codebook_store: Arc<dyn CodebookStore> = Arc::new(InMemoryCodebookStore::new());
        let unified_quantization = Arc::new(UnifiedQuantizationEngine::new(
            distance_compute.clone(),
            codebook_store.clone(),
        ));

        let storage_config =
            crate::compute::quantization::storage_engine::StorageQuantizationConfig::default();
        let storage_quantization_engine = Arc::new(StorageQuantizationEngine::new(
            unified_quantization.clone(),
            distance_compute.clone(),
            storage_config,
        ));

        // Create fallback stateless quantization engine for ad-hoc queries
        let fallback_codebook_store: Arc<dyn CodebookStore> =
            Arc::new(InMemoryCodebookStore::new());
        let fallback_quantization_engine = Arc::new(UnifiedQuantizationEngine::new(
            distance_compute.clone(),
            fallback_codebook_store,
        ));

        Ok((storage_quantization_engine, fallback_quantization_engine))
    }

    /// Register cache providers with the orchestrator
    async fn register_cache_providers(
        decompression_cache: Arc<decompression_cache::DecompressionCache>,
    ) -> Result<Option<Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>>> {
        if let Some(ref orch) =
            crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
        {
            use crate::storage::cache::orchestrator::{CacheStatsProvider, CacheType};

            // Register decompression cache provider (VectorData)
            let provider: Arc<dyn CacheStatsProvider + Send + Sync> =
                Arc::new(crate::storage::engines::sst::decompression_cache::DecompressionCacheStatsProvider::new(
                    decompression_cache.clone(),
                ));
            orch.register_cache_provider(CacheType::VectorData, provider);

            // Register lightweight providers for FilterBitmap and Metadata
            struct SstStaticProvider;
            impl CacheStatsProvider for SstStaticProvider {
                fn snapshot(&self) -> crate::storage::cache::orchestrator::UsageStats {
                    crate::storage::cache::orchestrator::UsageStats {
                        hit_rate: 0.0,
                        avg_entry_size: 4096,
                        access_frequency: 0.0,
                        last_rebalance: std::time::SystemTime::now(),
                    }
                }
            }

            let provider2: Arc<dyn CacheStatsProvider + Send + Sync> = Arc::new(SstStaticProvider);
            orch.register_cache_provider(CacheType::FilterBitmap, provider2.clone());
            orch.register_cache_provider(CacheType::Metadata, provider2);

            Ok(Some(orch.clone()))
        } else {
            Ok(None)
        }
    }

    // Getter methods for accessing engine components

    /// Get engine configuration
    pub fn config(&self) -> &SstConfig {
        &self.config
    }

    /// Get compaction manager
    pub fn compaction_manager(&self) -> Option<&Arc<Compaction>> {
        self.compaction_manager.as_ref()
    }

    /// Get filesystem factory
    pub fn filesystem(&self) -> &Arc<FilesystemFactory> {
        &self.filesystem
    }

    /// Get the filesystem factory as the extracted-port view (TD-DECOMP-82):
    /// `FilesystemFactory` impls `FilesystemPort` root-side, so callers bound
    /// for the sst-engine crate coerce through here.
    pub fn filesystem_port(&self) -> std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> {
        self.filesystem.clone()
    }

    /// Get unified filesystem (if initialized)
    pub fn unified_fs(&self) -> Option<&Arc<dyn FileSystem>> {
        self.unified_fs.as_ref()
    }

    /// Get atomic coordinator
    pub fn atomic_coordinator(&self) -> &Arc<TransactionCoordinator> {
        &self.atomic_coordinator
    }

    /// Get SSTable reader
    pub fn sstable_reader(&self) -> &Arc<UnifiedSstableReader> {
        &self.sstable_reader
    }

    /// Get distance computation engine
    pub fn distance_compute(&self) -> &Arc<UnifiedDistanceCompute> {
        &self.distance_compute
    }

    /// Get decompression cache
    pub fn decompression_cache(&self) -> &Arc<decompression_cache::DecompressionCache> {
        &self.decompression_cache
    }

    /// Get storage quantization engine
    pub fn storage_quantization_engine(&self) -> &Arc<StorageQuantizationEngine> {
        &self.storage_quantization_engine
    }

    /// Get fallback quantization engine
    pub fn fallback_quantization_engine(&self) -> &Arc<UnifiedQuantizationEngine> {
        &self.fallback_quantization_engine
    }

    /// Get universal optimizer
    pub fn universal_optimizer(&self) -> &UniversalPerformanceOptimizer {
        &self.universal_optimizer
    }

    /// Get cache orchestrator
    pub fn orchestrator(
        &self,
    ) -> Option<&Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>> {
        self.orchestrator.as_ref()
    }

    /// Get the AXIS manager for HNSW/IVF index operations
    ///
    /// Returns the AXIS manager if available, enabling:
    /// - HNSW-based approximate nearest neighbor search
    /// - IVF partition pruning
    /// - Hybrid vector + metadata queries
    ///
    /// Note: This method checks both the instance field and the global singleton
    /// to handle cases where the SST engine is created before AXIS manager initialization.
    pub fn axis_manager(&self) -> Option<Arc<dyn IndexEngine>> {
        // First check instance field, then fall back to global singleton
        self.axis_manager.clone().or_else(get_sst_axis_manager)
    }

    // =========================================================================
    // PCA Model Caching Methods (for Z-Order spatial encoding)
    // =========================================================================

    /// Get the PCA model cache for read access during search
    ///
    /// This is used by the search coordinator to project queries
    /// to PCA space for Z-Order pruning.
    pub fn pca_model_cache(
        &self,
    ) -> &Arc<
        tokio::sync::RwLock<
            std::collections::HashMap<String, super::pca_manager::EnhancedPCAModel>,
        >,
    > {
        &self.pca_model_cache
    }

    /// Get cached PCA model for a collection (if available)
    ///
    /// Returns the in-memory cached model, or loads from disk if not in cache.
    pub async fn get_pca_model(
        &self,
        collection_id: &str,
        collection_data_dir: &str,
    ) -> Option<super::pca_manager::EnhancedPCAModel> {
        // First check in-memory cache
        {
            let cache = self.pca_model_cache.read().await;
            if let Some(model) = cache.get(collection_id) {
                return Some(model.clone());
            }
        }

        // Try to load from disk
        if let Ok(Some(model)) = self.load_pca_model(collection_data_dir).await {
            // Cache it for future use
            {
                let mut cache = self.pca_model_cache.write().await;
                cache.insert(collection_id.to_string(), model.clone());
            }
            return Some(model);
        }

        None
    }

    /// Construct the PCA model file path for a collection
    /// Path: {collection_data_dir}/__model/pca_model.bin
    fn get_pca_model_path(&self, collection_data_dir: &str) -> String {
        format!("{}/__model/pca_model.bin", collection_data_dir)
    }

    /// Load PCA model from filesystem for a collection
    pub async fn load_pca_model(
        &self,
        collection_data_dir: &str,
    ) -> Result<Option<super::pca_manager::EnhancedPCAModel>> {
        let model_path = self.get_pca_model_path(collection_data_dir);

        let filesystem = self
            .filesystem
            .get_filesystem(collection_data_dir)
            .map_err(|e| SstError::Internal(format!("Failed to get filesystem: {}", e)))?;

        match filesystem.exists(&model_path).await {
            Ok(true) => {
                let data = filesystem
                    .read(&model_path)
                    .await
                    .map_err(|e| SstError::Internal(format!("Failed to read PCA model: {}", e)))?;

                let model: super::pca_manager::EnhancedPCAModel = bincode::deserialize(&data)
                    .map_err(|e| {
                        SstError::Internal(format!("Failed to deserialize PCA model: {}", e))
                    })?;

                info!(
                    "[SST] Loaded persisted PCA model for collection (version: {}, {} components)",
                    model.version, model.n_components
                );
                Ok(Some(model))
            }
            Ok(false) => {
                tracing::debug!("[SST] No persisted PCA model found at {}", model_path);
                Ok(None)
            }
            Err(e) => {
                tracing::debug!("[SST] Error checking PCA model at {}: {}", model_path, e);
                Ok(None)
            }
        }
    }

    /// Save PCA model to filesystem for a collection
    pub async fn save_pca_model(
        &self,
        collection_data_dir: &str,
        model: &super::pca_manager::EnhancedPCAModel,
    ) -> Result<()> {
        let model_path = self.get_pca_model_path(collection_data_dir);

        let filesystem = self
            .filesystem
            .get_filesystem(collection_data_dir)
            .map_err(|e| SstError::Internal(format!("Failed to get filesystem: {}", e)))?;

        // Ensure __model directory exists
        let model_dir = format!("{}/__model", collection_data_dir);
        filesystem.create_dir_all(&model_dir).await.map_err(|e| {
            SstError::Internal(format!("Failed to create __model directory: {}", e))
        })?;

        // Serialize model with bincode
        let data = bincode::serialize(model)
            .map_err(|e| SstError::Internal(format!("Failed to serialize PCA model: {}", e)))?;

        filesystem
            .write(&model_path, &data, None)
            .await
            .map_err(|e| SstError::Internal(format!("Failed to write PCA model: {}", e)))?;

        info!(
            "[SST] Persisted PCA model for collection at {} ({} components)",
            model_path, model.n_components
        );
        Ok(())
    }

    /// Train PCA model from vectors and cache it
    ///
    /// This should be called during flush when we have new vectors.
    /// The model is trained using adaptive dimensions based on vector dimensionality.
    pub async fn train_and_cache_pca_model(
        &self,
        collection_id: &str,
        collection_data_dir: &str,
        records: &[ProximaRecord],
    ) -> Result<()> {
        use super::pca_manager::AdaptivePcaConfig;

        if records.is_empty() {
            return Ok(());
        }

        let vectors: Vec<Vec<f32>> = records
            .iter()
            .filter_map(|record| {
                record
                    .embeddings
                    .first()
                    .filter(|embedding| !embedding.values.is_empty())
                    .map(|embedding| embedding.values.to_fp32_owned())
            })
            .collect();

        if vectors.is_empty() {
            return Ok(());
        }

        let vector_dim = vectors[0].len();
        if vector_dim == 0 {
            return Ok(());
        }

        // Use adaptive configuration for optimal PCA dimensions
        let pca_config = AdaptivePcaConfig::for_vector_dim(vector_dim);
        let n_components = pca_config.n_components;

        // Need at least n_components samples for training
        if vectors.len() < n_components {
            tracing::debug!(
                "[SST] Not enough vectors ({}) for PCA training (need at least {})",
                vectors.len(),
                n_components
            );
            return Ok(());
        }

        info!(
            "[SST] Training PCA model: {} vectors → {} components (from {}-dim)",
            vectors.len(),
            n_components,
            vector_dim
        );

        // Train PCA model
        let model =
            super::pca_manager::EnhancedPCAModel::train_from_vectors(&vectors, n_components)
                .map_err(|e| SstError::Internal(format!("Failed to train PCA model: {}", e)))?;

        // Save to disk
        self.save_pca_model(collection_data_dir, &model).await?;

        // Cache in memory
        {
            let mut cache = self.pca_model_cache.write().await;
            cache.insert(collection_id.to_string(), model);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sst_engine_creation() {
        let engine = SstEngine::new().await.expect("Failed to create SST engine");

        // Verify core components are initialized
        assert!(engine.compaction_manager().is_some());
        // Note: orchestrator is None in standalone test context (requires global CrossCacheOrchestrator)
        // It's only available when running within full ProximaDB runtime
        // assert!(engine.orchestrator().is_some()); // Removed - not applicable in isolated tests

        // Verify configuration is set
        assert_eq!(
            engine.config().block_size_kb,
            SstConfig::default().block_size_kb
        );
    }

    #[tokio::test]
    async fn test_sst_engine_with_custom_config() {
        let mut config = SstConfig::default();
        config.block_size_kb = 128; // Custom block size

        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        let engine = SstEngine::new_with_config(config.clone(), filesystem, distance_compute)
            .await
            .expect("Failed to create SST engine with custom config");

        assert_eq!(engine.config().block_size_kb, 128);
    }
}
