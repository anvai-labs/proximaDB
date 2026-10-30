// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! # AXIS Index Manager - Adaptive eXperimental Index System
//!
//! This module implements the central coordination layer for ProximaDB's adaptive
//! indexing system. AXIS dynamically selects and migrates between different index
//! types based on workload patterns, data characteristics, and query requirements.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │            AXIS Manager                      │
//! ├─────────────────────────────────────────────┤
//! │  Monitoring  →  Analysis  →  Adaptation     │
//! │      ↓            ↓            ↓            │
//! │  Metrics    Workload      Migration         │
//! │  Collection  Patterns      Engine           │
//! ├─────────────────────────────────────────────┤
//! │        Index Selection Strategy             │
//! │    ┌──────┬──────┬──────┬──────┐          │
//! │    │ HNSW │ IVF  │ LSH  │ Flat │          │
//! │    └──────┴──────┴──────┴──────┘          │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Key Features
//!
//! ### 1. **Adaptive Index Selection**
//! - Monitors query patterns and data distribution
//! - Automatically selects optimal index type
//! - Seamless migration between index types
//!
//! ### 2. **Performance Monitoring**
//! - Real-time tracking of index performance
//! - Query latency and throughput metrics
//! - Resource utilization monitoring
//!
//! ### 3. **Index Migration Engine**
//! - Zero-downtime index migrations
//! - Incremental index building
//! - Rollback capabilities
//!
//! ### 4. **Clustering Engine**
//! - Automatic data clustering for IVF indexes
//! - Centroid optimization
//! - Balanced cluster distribution
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use proximadb::index::axis::AxisManager;
//! use proximadb::index::axis::AxisConfig;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create AXIS manager with adaptive configuration
//! let config = AxisConfig {
//!     enable_adaptive_indexing: true,
//!     migration_threshold: 0.8,
//!     monitoring_interval_ms: 5000,
//!     ..Default::default()
//! };
//!
//! let axis_manager = AxisManager::new(config).await?;
//!
//! // AXIS automatically adapts to workload patterns
//! axis_manager.start_monitoring().await?;
//!
//! // Manual strategy override if needed
//! axis_manager.set_collection_strategy(
//!     "high_traffic_collection",
//!     IndexSelectionStrategy::Hnsw
//! ).await?;
//! # Ok(())
//! # }
//! ```

use crate::core::search::bounded_queue::TopKHeap;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Lazily resolves per-record filter metadata under the collection read
/// guard, memoizing each id on first touch. Replaces the previous
/// whole-collection `metadata_by_id` prebuild in `query_hnsw_with_predicate`:
/// the HNSW predicate traversal probes ~ef·M nodes, so building
/// `record_filter_metadata` (a fresh map + ≥3 String allocs) for every record
/// in the collection was O(N) work per filtered query disproportionate to
/// what the traversal touches. Verdicts are identical — same maps, same
/// filter — only WHEN the maps are built changes.
struct LazyFilterMetadata<'a> {
    /// Read guard over `AxisManager::collection_vectors`; `None` when no
    /// metadata expression was compiled (resolve then always returns `None`,
    /// exactly like the previous empty-map lookup).
    records:
        Option<tokio::sync::RwLockReadGuard<'a, HashMap<String, HashMap<String, ProximaRecord>>>>,
    /// Collection whose inner id→record map is resolved.
    collection_id: String,
    manager: &'a AxisManager,
    memo: std::sync::Mutex<MetadataMemo>,
}

#[derive(Default)]
struct MetadataMemo {
    resolved: HashMap<String, Arc<HashMap<String, Value>>>,
    absent: HashSet<String>,
}

impl<'a> LazyFilterMetadata<'a> {
    async fn new(manager: &'a AxisManager, collection_id: &str, needed: bool) -> Self {
        let records = if needed {
            Some(manager.collection_vectors.read().await)
        } else {
            None
        };
        Self {
            records,
            collection_id: collection_id.to_string(),
            manager,
            memo: std::sync::Mutex::new(MetadataMemo::default()),
        }
    }

    /// Resolve the filter-metadata map for `id`. Returns `None` when the id
    /// is not resident in the collection (or when no metadata expression was
    /// compiled) — the same `Option` the previous eager map produced.
    fn resolve(&self, id: &str) -> Option<Arc<HashMap<String, Value>>> {
        let records = self.records.as_ref()?;
        let collection = records.get(&self.collection_id)?;
        let mut memo = self
            .memo
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(hit) = memo.resolved.get(id) {
            return Some(Arc::clone(hit));
        }
        if memo.absent.contains(id) {
            return None;
        }
        match collection.get(id) {
            Some(record) => {
                let metadata = Arc::new(self.manager.record_filter_metadata(record));
                memo.resolved.insert(id.to_string(), Arc::clone(&metadata));
                Some(metadata)
            }
            None => {
                memo.absent.insert(id.to_string());
                None
            }
        }
    }
}
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error};

use crate::compute::distance_computation::engine::UnifiedDistanceCompute;
use crate::core::{String, VectorId};
use crate::index::axis::management::{
    migration_engine::{IndexMigrationEngine, MigrationDecision},
    monitor::PerformanceMonitor,
};
use crate::index::axis::{
    clustering::AxisClusteringEngine,
    clustering::ClusteringConfig,
    hmgi::{
        DetectionResult, HmgiPartitionKey, HmgiRegistry, HmgiRouter, ModalityDetector,
        ModalityExtractor, VectorRecordSample,
    },
    management::adaptive_engine::AdaptiveIndexEngine,
    types::{AxisConfig, Data, IndexAlgorithm, IndexSelectionStrategy},
};
use crate::index::{DenseVectorIndex, GlobalIdIndex, JoinEngine, MetadataIndex, SparseVectorIndex};
use proximadb_records::{ProximaRecord, ProximaTreeNode};
// Temporarily disabled due to arrow-arith compilation conflicts - DEFERRED: Re-enable when resolved
// use crate::storage::engines::viper::QuantizationMethod;

/// Central manager for AXIS with adaptive capabilities
///
/// The `AxisManager` coordinates all indexing operations in ProximaDB, providing
/// a unified interface for vector indexing with automatic adaptation based on
/// workload characteristics.
///
/// # Components
///
/// - **Core Indexes**: Global ID, metadata, dense/sparse vector indexes
/// - **Adaptive Engine**: Monitors and adapts to workload patterns
/// - **Migration Engine**: Handles index type transitions
/// - **Performance Monitor**: Tracks metrics and performance
/// - **Clustering Engine**: Manages IVF clustering operations
///
/// # Thread Safety
///
/// All operations are thread-safe through internal synchronization using
/// `Arc<RwLock>` for shared state and `DashMap` for concurrent collections.
///
/// ## Index Selection Strategy:
///
/// ```text
/// Dataset Size    Dimensions    QPS       → Recommended Index
/// ------------------------------------------------------------
/// < 10K          Any           Any       → Flat (exact)
/// 10K-100K       < 100         < 1000    → HNSW
/// 10K-100K       > 100         < 1000    → IVF + PQ
/// 100K-1M        < 200         < 5000    → HNSW + PQ
/// 100K-1M        > 200         Any       → IVF + PQ
/// > 1M           Any           < 1000    → IVF + PQ
/// > 1M           Any           > 1000    → LSH
/// ```
///
/// ## Migration Triggers:
///
/// AXIS automatically triggers migration when:
/// - Query latency degrades by >20%
/// - Memory usage exceeds threshold
/// - Dataset size crosses boundaries (10K, 100K, 1M)
/// - Query pattern changes significantly
///
/// ## EventLog Integration:
///
/// AXIS receives index update events from the EventLog:
/// ```text
/// Storage Flush → EventLog → AXIS Consumer
///                              ↓
///                        Index Update
///                              ↓
///                        Background Build
/// ```
///
/// Resolves a collection's catalog-addressed index location on demand
/// (CATALOG_OBJECT_MODEL read-port, open-Q #3). Catalog-free **by design**: AXIS
/// depends only on this trait, never on `proximadb-catalog`, so the dependency is
/// inverted — an adapter in the wiring/control layer implements it over the
/// catalog. Returning `None` means "no catalog-resolved location", and AXIS falls
/// back to the `index_persist_url` / `index_persist_dir` convention. The result is
/// memoized by [`AxisManager`], so this is consulted at most once per collection.
#[async_trait::async_trait]
pub trait IndexLocationResolver: Send + Sync {
    /// Resolve `collection_id`'s ANN index location, or `None` for the convention.
    async fn resolve_index_location(&self, collection_id: &str) -> Option<String>;
}

pub struct AxisManager {
    /// Core index components for different data types

    /// Global ID index for fast ID-based lookups
    /// Maps vector IDs to storage locations across all collections
    global_id_index: Arc<GlobalIdIndex>,

    /// Metadata index for filtered search
    /// Supports range queries, equality, and complex predicates
    metadata_index: Arc<MetadataIndex>,

    /// Dense vector index for similarity search
    /// Supports HNSW, IVF, LSH, Annoy, PQ, Flat algorithms
    dense_vector_index: Arc<DenseVectorIndex>,

    /// Sparse vector index for keyword/document search
    /// Optimized for high-dimensional sparse vectors
    sparse_vector_index: Arc<SparseVectorIndex>,

    /// Join engine for hybrid queries
    /// Combines results from multiple indexes
    #[allow(dead_code)]
    join_engine: Arc<JoinEngine>,

    /// Adaptive intelligence components for workload optimization

    /// Monitors workload patterns and triggers adaptations
    /// Analyzes query distribution, data growth, and access patterns
    adaptive_engine: Arc<AdaptiveIndexEngine>,

    /// Handles zero-downtime index migrations
    /// Builds new index in background, validates, then switches atomically
    migration_engine: Arc<IndexMigrationEngine>,

    /// Tracks performance metrics and anomalies
    /// Monitors latency, throughput, accuracy, and resource usage
    #[allow(dead_code)]
    performance_monitor: Arc<PerformanceMonitor>,

    /// Manages clustering for IVF indexes
    /// Performs k-means clustering and centroid optimization
    #[allow(dead_code)]
    clustering_engine: Arc<AxisClusteringEngine>,

    /// Collection-specific configurations
    /// Maps collection_id → selected index strategy
    /// Can be manually overridden or automatically determined
    collection_strategies: Arc<RwLock<HashMap<String, IndexSelectionStrategy>>>,

    /// Active migrations
    /// Tracks ongoing index migrations for monitoring and rollback
    active_migrations: Arc<RwLock<HashMap<String, MigrationStatus>>>,

    /// Configuration and metrics
    /// Global AXIS configuration (thresholds, intervals, etc.)
    #[allow(dead_code)]
    config: AxisConfig,

    /// Aggregated metrics across all managed indexes
    metrics: Arc<RwLock<AxisMetrics>>,

    /// Collection service for IndexConfig retrieval
    /// Provides access to collection metadata and index configurations
    /// Set via set_collection_service() after initialization
    collection_service: Option<Arc<crate::services::collection::manager::CollectionService>>,

    /// TD-075 / Phase 8 F2 recall-probe gate. When set, the IVF query path
    /// consults it before selecting the quantized route; a closed gate routes
    /// to exact search. Set via set_recall_probe_gate().
    recall_probe_gate: Option<Arc<crate::catalog::RecallProbeGate>>,

    /// TD-087 Slice B: root directory for persisted IVF indexes. When set,
    /// trained indexes are written to `<dir>/<collection_id>/ivf.bin` after each
    /// rebuild and lazily reloaded on the first query for a cold collection.
    /// `None` ⇒ persistence disabled (embedded/test harnesses without a data dir).
    index_persist_dir: Option<std::path::PathBuf>,

    /// ADR-023 R3 Slice 4: optional object-store URI for index persistence
    /// (`s3://bucket/prefix`, `adls://`, `gs://`, `file://path`). When set with
    /// `filesystem_factory`, persistence AND cold-load route through the
    /// canonical `FilesystemFactory` by scheme — converging on the SAME
    /// `FileSystem`-trait path the ranged cold-load already reads through. Takes
    /// precedence over `index_persist_dir`.
    index_persist_url: Option<String>,

    /// ADR-023 R3 Slice 4: the shared `FilesystemFactory` (injected at boot) that
    /// `index_persist_url` dispatches through by scheme. `None` ⇒ URI mode off
    /// (falls back to the local `index_persist_dir`).
    filesystem_factory: Option<Arc<crate::storage::persistence::filesystem::FilesystemFactory>>,

    /// CATALOG_OBJECT_MODEL P1: per-collection catalog-resolved index location
    /// (a `CatalogProjection.location`, or the `DrPathBuilder`-derived
    /// `indexes/<projection>/` default). When set for a collection it is
    /// authoritative and takes precedence over the `index_persist_url` /
    /// `index_persist_dir` conventions, so an index can be relocated/tiered from
    /// the catalog. Empty ⇒ the legacy convention is used unchanged (mixed-safe,
    /// default-off).
    ///
    /// This is a **memo cache**: populated eagerly by the boot adapter and lazily
    /// by [`index_location_resolver`](Self::index_location_resolver) on first use,
    /// so it is interior-mutable (`Arc<RwLock>`) to allow runtime fills after the
    /// manager is shared.
    index_location_overrides: Arc<RwLock<std::collections::HashMap<String, String>>>,

    /// CATALOG_OBJECT_MODEL read-port: resolves a collection's index location from
    /// the catalog on demand (the pull seam). When present, a memo miss consults
    /// it once and caches the result — this is what makes catalog-resolved index
    /// locations cover *runtime-created* collections, not just those present at
    /// boot. `None` keeps the eager-only (boot-pushed) behavior.
    index_location_resolver: Option<Arc<dyn IndexLocationResolver>>,

    /// Phase 8 F4a (TD-094): collections whose in-memory IVF index was evicted by
    /// `suspend_collection` to free memory. The persisted `ivf.bin` remains, so
    /// the next query (or `resume_collection`) warm-loads it; the marker is
    /// cleared on that warm-load. Surfaced in route-health.
    suspended_collections: Arc<RwLock<std::collections::HashSet<String>>>,

    /// ADR-023 R3 (c): when `true`, a cold-loaded (ranged) IVF index serves
    /// PURELY on-probe — the eager background warm-apply is skipped, so unprobed
    /// clusters are never downloaded (the object-store mode: pay only for what a
    /// query touches). Default `false` keeps the eager fill → `FullTwoStage`,
    /// which is right for cheap local I/O.
    cold_lazy_warm: bool,

    /// Shared collection cache from VectorOperationsService (read-only access)
    /// This avoids duplicating collection metadata in memory
    /// Collections are cached by VectorOperationsService and shared here
    /// for fast access during index operations
    shared_collection_cache:
        Option<Arc<dashmap::DashMap<String, Arc<crate::proto::proximadb_v1::Collection>>>>,

    /// Real HNSW indexes per collection for vector similarity search
    /// Maps collection_id → AxisHnswIndex instance
    /// These are the actual in-memory HNSW indexes that store and query vectors
    /// NOTE: HNSW has poor recall with incremental indexing - prefer IVF for production
    hnsw_indexes:
        Arc<RwLock<HashMap<String, Arc<crate::index::axis::indexes::hnsw_index::AxisHnswIndex>>>>,

    /// Real IVF indexes per collection for vector similarity search (DEFAULT)
    /// Maps collection_id → UnifiedIvfIndex instance
    /// IVF is better suited for incremental workloads as new vectors are simply
    /// assigned to their nearest cluster without degrading graph quality
    ivf_indexes: Arc<
        RwLock<
            HashMap<
                String,
                Arc<
                    tokio::sync::RwLock<
                        crate::index::axis::indexes::dual_store_ivf::UnifiedIvfIndex,
                    >,
                >,
            >,
        >,
    >,

    /// Pending vectors buffer for IVF training
    /// IVF requires k-means training before vectors can be added
    /// Vectors are buffered here until we have enough for training (min_train_size)
    ivf_pending_vectors: Arc<RwLock<HashMap<String, Vec<(String, Vec<f32>)>>>>,

    /// Per-collection served-index generation. Incremented whenever the served
    /// index Arc is atomically rebuilt+swapped (Phase 8 F1 recluster apply-step).
    /// Lets operators / tests observe that a swap happened.
    index_generations: Arc<RwLock<HashMap<String, u64>>>,

    /// Exact vector records tracked per collection for correctness-first filtered search.
    /// This is the source of truth for metadata-aware fallback execution and MVCC timestamps.
    collection_vectors: Arc<RwLock<HashMap<String, HashMap<String, ProximaRecord>>>>,

    /// HMGI - Hierarchical Multi-modality Graph Indexing components
    /// Registry for managing per-modality HNSW partitions
    hmgi_registry: Option<Arc<HmgiRegistry>>,

    /// Router for directing queries to relevant modality partitions
    hmgi_router: Option<Arc<HmgiRouter>>,

    /// Modality extractor for determining vector modality from metadata
    hmgi_extractor: Option<Arc<ModalityExtractor>>,

    /// Modality detector for auto-enabling HMGI on multi-modality collections
    hmgi_detector: Option<Arc<ModalityDetector>>,

    /// Collections with HMGI enabled
    hmgi_enabled_collections: Arc<RwLock<std::collections::HashSet<String>>>,

    /// Collection OID lookup for HMGI partition key generation
    /// Maps collection_id → oid for partition key creation
    hmgi_collection_oids: Arc<RwLock<HashMap<String, u64>>>,
}

/// Status of ongoing migrations
///
/// ## Migration Lifecycle:
///
/// 1. **Triggered**: Migration decision made
/// 2. **Building**: New index being constructed (0-90%)
/// 3. **Validating**: Comparing accuracy (90-95%)
/// 4. **Switching**: Atomic switchover (95-99%)
/// 5. **Cleanup**: Old index removal (100%)
///
/// Progress tracking enables:
/// - User visibility into long-running migrations
/// - Cancellation/rollback capabilities
/// - Resource planning (CPU/memory allocation)
#[derive(Debug, Clone)]
pub struct MigrationStatus {
    /// Unique identifier for tracking
    pub migration_id: proximadb_kernel::uuid::Uuid,

    /// Source index strategy
    pub from_strategy: IndexSelectionStrategy,

    /// Target index strategy
    pub to_strategy: IndexSelectionStrategy,

    /// When migration started
    pub start_time: DateTime<Utc>,

    /// Current progress (0.0 to 100.0)
    pub progress_percentage: f64,

    /// Estimated completion based on current rate
    pub estimated_completion: Option<DateTime<Utc>>,
}

/// AXIS metrics
///
/// ## Key Metrics:
///
/// - **Migration Success Rate**: successful/total migrations
/// - **Average Migration Time**: Indicates system adaptation speed
/// - **Rebuild Frequency**: High rebuilds may indicate instability
/// - **Vector Growth Rate**: Helps predict future resource needs
///
/// These metrics feed into:
/// - Adaptive decision making
/// - Capacity planning
/// - Performance monitoring dashboards
#[derive(Debug, Clone, Default)]
pub struct AxisMetrics {
    /// Total migration attempts
    pub total_migrations: u64,

    /// Successfully completed migrations
    pub successful_migrations: u64,

    /// Failed migrations (rolled back)
    pub failed_migrations: u64,

    /// Average time to complete migration
    pub average_migration_time_ms: u64,

    /// Number of collections under management
    pub total_collections_managed: u64,

    /// Total vectors across all indexes
    pub total_vectors_indexed: u64,

    /// Full index rebuilds (usually after corruption)
    pub total_rebuilds: u64,
}

impl AxisManager {
    /// Create a new AXIS index manager
    pub async fn new(config: AxisConfig) -> Result<Self> {
        // Initialize core index components
        let global_id_index = Arc::new(GlobalIdIndex::new().await?);
        let metadata_index = Arc::new(MetadataIndex::new().await?);
        let dense_vector_index = Arc::new(DenseVectorIndex::new().await?);
        let sparse_vector_index = Arc::new(SparseVectorIndex::new().await?);
        let join_engine = Arc::new(JoinEngine::new().await?);

        // Initialize adaptive components
        let adaptive_engine = Arc::new(AdaptiveIndexEngine::new(config.clone()).await?);
        let migration_engine = Arc::new(IndexMigrationEngine::new(config.clone()).await?);
        let performance_monitor = Arc::new(PerformanceMonitor::new(config.clone()).await?);

        // Initialize clustering engine with default config
        let clustering_config = ClusteringConfig::default();
        let clustering_engine = Arc::new(AxisClusteringEngine::new(clustering_config));

        // Initialize HMGI components with the default modality field. Collections are still
        // opted in only after explicit enablement or auto-detection of multiple modalities.
        let hmgi_registry = Arc::new(HmgiRegistry::new());
        let hmgi_extractor = Arc::new(ModalityExtractor::new());
        let hmgi_detector = Arc::new(ModalityDetector::default_config());
        let hmgi_router = Arc::new(HmgiRouter::new(
            hmgi_registry.clone(),
            hmgi_extractor.clone(),
        ));

        Ok(Self {
            global_id_index,
            metadata_index,
            dense_vector_index,
            sparse_vector_index,
            join_engine,
            adaptive_engine,
            migration_engine,
            performance_monitor,
            clustering_engine,
            collection_strategies: Arc::new(RwLock::new(HashMap::new())),
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(RwLock::new(AxisMetrics::default())),
            collection_service: None, // Will be set later via set_collection_service
            recall_probe_gate: None,  // Set later via set_recall_probe_gate (TD-075)
            index_persist_dir: None,  // Set later via set_index_persist_dir (TD-087 Slice B)
            index_persist_url: None,  // Set later via set_index_persist_url (ADR-023 R3 Slice 4)
            index_location_overrides: Arc::new(RwLock::new(std::collections::HashMap::new())), // catalog-resolved memo (P1)
            index_location_resolver: None, // read-port injected at boot (open-Q #3)
            filesystem_factory: None, // Set later via set_filesystem_factory (ADR-023 R3 Slice 4)
            suspended_collections: Arc::new(RwLock::new(std::collections::HashSet::new())), // F4a
            cold_lazy_warm: false,    // ADR-023 R3 (c): eager warm by default
            shared_collection_cache: None, // Will be set via set_shared_collection_cache
            hnsw_indexes: Arc::new(RwLock::new(HashMap::new())), // Real HNSW indexes per collection
            ivf_indexes: Arc::new(RwLock::new(HashMap::new())), // Real IVF indexes per collection (DEFAULT)
            ivf_pending_vectors: Arc::new(RwLock::new(HashMap::new())), // Buffer for IVF training
            index_generations: Arc::new(RwLock::new(HashMap::new())), // Served-index swap generations
            collection_vectors: Arc::new(RwLock::new(HashMap::new())),
            // HMGI components initialized up front; collections are enabled on demand.
            hmgi_registry: Some(hmgi_registry),
            hmgi_router: Some(hmgi_router),
            hmgi_extractor: Some(hmgi_extractor),
            hmgi_detector: Some(hmgi_detector),
            hmgi_enabled_collections: Arc::new(RwLock::new(std::collections::HashSet::new())),
            hmgi_collection_oids: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Set the collection service for IndexConfig retrieval
    pub fn set_collection_service(
        &mut self,
        collection_service: Arc<crate::services::collection::manager::CollectionService>,
    ) {
        self.collection_service = Some(collection_service);
        tracing::info!("🔗 AXIS: Collection service set for IndexConfig retrieval");
    }

    /// Set the recall-probe gate (TD-075 / Phase 8 F2) so the IVF query path can
    /// consult it before selecting the quantized route.
    pub fn set_recall_probe_gate(&mut self, gate: Arc<crate::catalog::RecallProbeGate>) {
        self.recall_probe_gate = Some(gate);
        tracing::info!("🔗 AXIS: RecallProbeGate set for quantized-route gating (TD-075)");
    }

    /// Set the root directory for persisted IVF indexes (TD-087 Slice B). Once
    /// set, trained indexes are written after each rebuild and lazily reloaded
    /// on the first query for a collection with no in-memory index.
    pub fn set_index_persist_dir(&mut self, dir: std::path::PathBuf) {
        self.index_persist_dir = Some(dir);
        tracing::info!("🔗 AXIS: IVF index persistence enabled (TD-087 Slice B)");
    }

    /// ADR-023 R3 Slice 4: route IVF index persistence + cold-load through an
    /// object-store URI (`s3://bucket/prefix`, `adls://`, `gs://`, `file://path`)
    /// via the canonical `FilesystemFactory`. Takes precedence over the local
    /// persist dir; requires `set_filesystem_factory` to dispatch by scheme.
    pub fn set_index_persist_url(&mut self, url: String) {
        tracing::info!(
            "🔗 AXIS: IVF index persistence routed via object-store URI '{}'",
            url
        );
        self.index_persist_url = Some(url);
    }

    /// CATALOG_OBJECT_MODEL P1: register the catalog-resolved index location for a
    /// collection's ANN projection. When set, it takes precedence over the
    /// `index_persist_url`/`index_persist_dir` conventions in
    /// [`index_fs_and_path`](Self::index_fs_and_path), so the index path is
    /// authoritative-from-catalog (relocatable/tierable per projection). The boot
    /// adapter computes this via `DrCollectionPath::resolve_index_location`
    /// (projection `location`, else the derived `indexes/<projection>/` default).
    ///
    /// `&self` + async because the location memo is interior-mutable (filled at
    /// boot AND lazily at runtime via the read-port), so a shared `AxisManager`
    /// can still register a location after it has been `Arc`-wrapped.
    pub async fn set_index_location(&self, collection_id: impl Into<String>, location: String) {
        let collection_id = collection_id.into();
        tracing::info!(
            "🔗 AXIS: catalog-resolved index location for '{}' → '{}'",
            collection_id,
            location
        );
        self.index_location_overrides
            .write()
            .await
            .insert(collection_id, location);
    }

    /// CATALOG_OBJECT_MODEL read-port (open-Q #3): inject the resolver that pulls a
    /// collection's index location from the catalog on demand. A memo miss in
    /// [`index_fs_and_path`](Self::index_fs_and_path) consults it once and caches
    /// the result, so catalog-resolved locations cover runtime-created collections
    /// — not just those the boot adapter enumerated. Catalog-free trait, so AXIS
    /// stays independent of `proximadb-catalog` (the dependency is inverted).
    pub fn set_index_location_resolver(&mut self, resolver: Arc<dyn IndexLocationResolver>) {
        self.index_location_resolver = Some(resolver);
    }

    /// ADR-023 R3 Slice 4: inject the shared `FilesystemFactory` that
    /// `index_persist_url` dispatches through by scheme.
    pub fn set_filesystem_factory(
        &mut self,
        factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) {
        self.filesystem_factory = Some(factory);
    }

    /// Whether IVF index persistence is enabled by either mechanism (object-store
    /// URI or local dir). Suspend/resume require it (no in-memory eviction without
    /// a resumable on-disk copy).
    fn persistence_enabled(&self) -> bool {
        (self.index_persist_url.is_some() && self.filesystem_factory.is_some())
            || self.index_persist_dir.is_some()
    }

    /// ADR-023 R3 (c): enable PURE-LAZY cold warm — skip the eager background
    /// fill so a ranged cold index downloads only the clusters queries probe (the
    /// object-store mode: pay only for what's touched). Off by default (eager
    /// fill → `FullTwoStage`, right for cheap local I/O). The index stays
    /// `ColdBinaryOnly` and serves via on-probe rerank.
    pub fn set_cold_lazy_warm(&mut self, lazy: bool) {
        self.cold_lazy_warm = lazy;
    }

    /// ADR-023 R3 (c) observability: the cold→warm serving status of a loaded IVF
    /// index, as `(serving_state, warm_clusters_fetched, warm_clusters_total)`.
    /// `ColdBinaryOnly` with `fetched < total` means the collection is serving
    /// reduced-recall while the fp32 tier streams/loads on demand; `FullTwoStage`
    /// (or `fetched == total`) means full recall. `None` if no IVF index is
    /// loaded; `total == 0` for whole-file / non-binary loads (no per-cluster
    /// tracking). Surfaced in route-health so operators see the cold-start window.
    pub async fn cold_serving_status(
        &self,
        collection_id: &str,
    ) -> Option<(crate::index::axis::IvfServingState, usize, usize)> {
        let idx = self.ivf_indexes.read().await.get(collection_id).cloned()?;
        let g = idx.read().await;
        Some((
            g.serving_state(),
            g.fetched_cluster_count(),
            g.total_warm_clusters(),
        ))
    }

    /// ADR-023 R3 Slice 4 (convergence): resolve `(filesystem, path)` for a
    /// collection's persisted IVF index, or `None` when persistence is disabled.
    /// Object-store URI mode (`index_persist_url` + factory) takes precedence over
    /// the legacy local dir; BOTH feed the same `FileSystem`-trait persist +
    /// cold-load path, so the read and write sides never diverge.
    async fn index_fs_and_path(
        &self,
        collection_id: &str,
    ) -> Option<(
        Arc<dyn crate::storage::persistence::filesystem::FileSystem>,
        String,
    )> {
        // CATALOG_OBJECT_MODEL P1 + read-port: a catalog-resolved index location is
        // authoritative — it already encodes the per-projection path (and any
        // relocation/tiering), so we append the index file and dispatch by scheme
        // via the factory, ahead of the legacy url/dir conventions. The location
        // is taken from the memo (boot-pushed) or, on a miss, pulled once from the
        // read-port and cached — this is what covers runtime-created collections.
        let catalog_location = {
            let memo = self
                .index_location_overrides
                .read()
                .await
                .get(collection_id)
                .cloned();
            match memo {
                Some(loc) => Some(loc),
                None => match &self.index_location_resolver {
                    Some(resolver) => match resolver.resolve_index_location(collection_id).await {
                        Some(loc) => {
                            self.index_location_overrides
                                .write()
                                .await
                                .insert(collection_id.to_string(), loc.clone());
                            Some(loc)
                        }
                        None => None,
                    },
                    None => None,
                },
            }
        };
        if let (Some(loc), Some(factory)) = (catalog_location, &self.filesystem_factory) {
            let path = format!("{}/ivf.bin", loc.trim_end_matches('/'));
            return match factory.get_filesystem(&path) {
                Ok(fs) => Some((fs, path)),
                Err(e) => {
                    tracing::warn!(
                        "AXIS: no filesystem for catalog index location '{}' ({}); persistence disabled",
                        path,
                        e
                    );
                    None
                }
            };
        }
        if let (Some(url), Some(factory)) = (&self.index_persist_url, &self.filesystem_factory) {
            let path = format!("{}/{}/ivf.bin", url.trim_end_matches('/'), collection_id);
            return match factory.get_filesystem(&path) {
                Ok(fs) => Some((fs, path)),
                Err(e) => {
                    tracing::warn!(
                        "AXIS: no filesystem for index URI '{}' ({}); persistence disabled",
                        path,
                        e
                    );
                    None
                }
            };
        }
        if let Some(dir) = &self.index_persist_dir {
            let path = dir
                .join(collection_id)
                .join("ivf.bin")
                .to_string_lossy()
                .to_string();
            return match Self::local_index_filesystem().await {
                Ok(fs) => Some((fs, path)),
                Err(e) => {
                    tracing::warn!("AXIS: local filesystem unavailable ({})", e);
                    None
                }
            };
        }
        None
    }

    /// Best-effort persist of a trained IVF index to disk. Logged, never fatal —
    /// a persistence failure must not fail the build/rebuild.
    async fn persist_ivf_index(
        &self,
        collection_id: &str,
        index: &Arc<
            tokio::sync::RwLock<crate::index::axis::indexes::dual_store_ivf::UnifiedIvfIndex>,
        >,
    ) {
        let Some((fs, path)) = self.index_fs_and_path(collection_id).await else {
            return;
        };
        let guard = index.read().await;
        match crate::index::axis::storage::serialization::IndexSerializer::persist_ivf_index(
            &guard,
            collection_id,
            &path,
            &fs,
        )
        .await
        {
            Ok(()) => tracing::info!(
                "💾 AXIS: persisted IVF index for '{}' → {}",
                collection_id,
                path
            ),
            Err(e) => tracing::warn!(
                "AXIS: failed to persist IVF index for '{}' ({}); index remains in-memory only",
                collection_id,
                e
            ),
        }
    }

    /// Lazily warm a collection's IVF index from disk on the first query (TD-087
    /// Slice B). No-op when the index is already served, persistence is disabled,
    /// or no persisted file exists. On a successful load the index is installed
    /// in `ivf_indexes` and an IVF routing strategy is registered.
    async fn ensure_ivf_index_loaded(&self, collection_id: &str) {
        if self.has_ivf_index(collection_id).await {
            return;
        }
        // ADR-023 R3 Slice 4: resolve the (filesystem, path) ONCE and use it for
        // both the existence check and the cold/whole-file load — so a local dir
        // and an `s3://`/`adls://`/`gs://` URI drive the identical load path.
        let Some((fs, path_str)) = self.index_fs_and_path(collection_id).await else {
            return;
        };
        if !fs.exists(&path_str).await.unwrap_or(false) {
            return;
        }
        use crate::index::axis::storage::serialization::{IndexSerializer, RangedColdLoad};

        // ADR-023 R3 (b): binary indexes load cold-first via byte-RANGE reads —
        // only [header]+[COLD] is read before serving Stage-1, then each cluster's
        // fp32 is fetched per-cluster on demand (never the whole WARM tier; the
        // win on object storage). v1 / non-binary indexes fall back to a
        // whole-file load (their membership lives in the full body).
        match IndexSerializer::try_cold_load_ranged(&fs, &path_str).await {
            Ok(Some(RangedColdLoad {
                index,
                metadata: _,
                warm_base,
                directory,
            })) => {
                let (dimension, n) = (index.dimension(), index.len());
                // ADR-023 R3 (c): wire the on-probe warm source so queries during
                // the cold-load window rerank exactly — fetching ONLY the probed
                // (survivor) clusters' fp32 — instead of serving Stage-1-only
                // Hamming. The eager background warm-apply below still runs to
                // FullTwoStage; `add_fp32` is idempotent (keyed by id), so any
                // overlap between an on-probe fetch and the background fill is
                // harmless, and `fetched_clusters` dedups repeat on-probe reads.
                let mut index = index;
                if !directory.is_empty() {
                    let dir_map = directory.iter().map(|e| (e.cluster_id, *e)).collect();
                    index.set_warm_source(
                        crate::index::axis::indexes::dual_store_ivf::RangedWarmSource {
                            fs: fs.clone(),
                            path: path_str.clone(),
                            warm_base,
                            directory: dir_map,
                        },
                    );
                }
                let Some(index_arc) = self
                    .install_loaded_ivf(collection_id, index, dimension, n)
                    .await
                else {
                    return; // another task loaded it concurrently
                };
                // Background ranged warm-apply: range-fetch each cluster's fp32,
                // install under the index READ lock (released between clusters so
                // serving overlaps the fill), then the instant FullTwoStage flip.
                // ADR-023 R3 (c): skipped in pure-lazy mode — the index stays
                // ColdBinaryOnly and serves via on-probe fetch, so unprobed
                // clusters are never downloaded (object-store mode).
                if !directory.is_empty() && !self.cold_lazy_warm {
                    let coll = collection_id.to_string();
                    let fs2 = fs.clone();
                    let path2 = path_str.clone();
                    tokio::spawn(async move {
                        let n_clusters = directory.len();
                        for ext in &directory {
                            let vecs = match IndexSerializer::fetch_warm_cluster_ranged(
                                &fs2, &path2, warm_base, ext,
                            )
                            .await
                            {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::warn!(
                                        "AXIS: warm cluster fetch failed for '{}': {}",
                                        coll,
                                        e
                                    );
                                    return;
                                }
                            };
                            let g = index_arc.read().await;
                            if let Err(e) = g.restore_warm_cluster(&vecs) {
                                tracing::warn!(
                                    "AXIS: warm cluster install failed for '{}': {}",
                                    coll,
                                    e
                                );
                                return;
                            }
                        }
                        index_arc.write().await.mark_full_two_stage();
                        tracing::info!(
                            "🔥 AXIS: warm tier (ranged) applied for '{}' ({} clusters) → FullTwoStage",
                            coll,
                            n_clusters,
                        );
                    });
                }
                tracing::info!(
                    "🔥 AXIS: cold-first (ranged) IVF index for '{}' from {} ({} vectors)",
                    collection_id,
                    path_str,
                    n,
                );
            }
            Ok(None) => {
                // v1 / non-binary: whole-file load (membership lives in the body).
                self.load_ivf_whole_file(collection_id, &fs, &path_str)
                    .await;
            }
            Err(e) => {
                tracing::warn!(
                    "AXIS: ranged cold-load for '{}' failed ({}); trying whole-file load",
                    collection_id,
                    e,
                );
                self.load_ivf_whole_file(collection_id, &fs, &path_str)
                    .await;
            }
        }
    }

    /// Local `FileSystem` for IVF index I/O — the default backend and the
    /// fallback when no object-store URI is configured (ADR-023 R3 Slice 4;
    /// object-store schemes are resolved via `FilesystemFactory` in
    /// `index_fs_and_path`).
    async fn local_index_filesystem()
    -> std::result::Result<Arc<dyn crate::storage::persistence::filesystem::FileSystem>, String>
    {
        use crate::storage::persistence::filesystem::LocalFileSystem;
        use crate::storage::persistence::filesystem::local::LocalConfig;
        let local = LocalFileSystem::new(LocalConfig {
            root_dir: None,
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false,
        })
        .await
        .map_err(|e| e.to_string())?;
        Ok(Arc::new(local))
    }

    /// Install a loaded IVF index into `ivf_indexes` (with a concurrent-load
    /// double-check), register its routing strategy, and clear any suspension
    /// marker (F4a). Returns the shared handle, or `None` if another task won the
    /// race.
    async fn install_loaded_ivf(
        &self,
        collection_id: &str,
        index: crate::index::axis::indexes::dual_store_ivf::UnifiedIvfIndex,
        dimension: usize,
        n: usize,
    ) -> Option<
        Arc<tokio::sync::RwLock<crate::index::axis::indexes::dual_store_ivf::UnifiedIvfIndex>>,
    > {
        let index_arc = Arc::new(tokio::sync::RwLock::new(index));
        {
            let mut indexes = self.ivf_indexes.write().await;
            if indexes.contains_key(collection_id) {
                return None;
            }
            indexes.insert(collection_id.to_string(), index_arc.clone());
        }
        self.register_loaded_ivf_strategy(collection_id, dimension, n)
            .await;
        self.suspended_collections
            .write()
            .await
            .remove(collection_id);
        Some(index_arc)
    }

    /// Whole-file IVF load (the pre-R3(b) path): used for v1 / non-binary indexes
    /// (whose posting-list membership lives in the full body) and as a fallback
    /// when the ranged header read fails. Cold-first when the file has a binary
    /// tier, else FullEager; any deferred WARM tier is applied per cluster in the
    /// background.
    async fn load_ivf_whole_file(
        &self,
        collection_id: &str,
        fs: &Arc<dyn crate::storage::persistence::filesystem::FileSystem>,
        path: &str,
    ) {
        use crate::index::axis::storage::serialization::{ColdLoadResult, IndexSerializer};
        let bytes = match fs.read(path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "AXIS: read IVF index for '{}' failed ({})",
                    collection_id,
                    e
                );
                return;
            }
        };
        match IndexSerializer::load_ivf_cold_path(&bytes).await {
            Ok(ColdLoadResult {
                index,
                metadata,
                warm,
            }) => {
                let (dimension, n) = (index.dimension(), index.len());
                let cold_first = warm.is_some();
                let Some(index_arc) = self
                    .install_loaded_ivf(collection_id, index, dimension, n)
                    .await
                else {
                    return;
                };
                if let Some(warm_bytes) = warm {
                    let coll = collection_id.to_string();
                    // v3 indexes carry a WARM byte-directory; decode the deferred
                    // bytes per-cluster via the directory, else the v2 whole tier.
                    let warm_dir = metadata.warm_directory();
                    tokio::spawn(async move {
                        let decoded = match warm_dir {
                            Some(dir) => {
                                IndexSerializer::decode_warm_clusters_dir(&warm_bytes, &dir)
                            }
                            None => IndexSerializer::decode_warm_clusters(&warm_bytes),
                        };
                        let clusters = match decoded {
                            Ok(c) => c,
                            Err(e) => {
                                tracing::warn!(
                                    "AXIS: decode warm tier failed for '{}': {}",
                                    coll,
                                    e
                                );
                                return;
                            }
                        };
                        for (_cluster_id, vectors) in &clusters {
                            let g = index_arc.read().await;
                            if let Err(e) = g.restore_warm_cluster(vectors) {
                                tracing::warn!(
                                    "AXIS: warm cluster install failed for '{}': {}",
                                    coll,
                                    e
                                );
                                return;
                            }
                        }
                        index_arc.write().await.mark_full_two_stage();
                        tracing::info!(
                            "🔥 AXIS: warm tier applied for '{}' ({} clusters) → FullTwoStage",
                            coll,
                            clusters.len()
                        );
                    });
                }
                tracing::info!(
                    "🔥 AXIS: {}-loaded IVF index for '{}' from {} ({} vectors)",
                    if cold_first { "cold-first" } else { "warm" },
                    collection_id,
                    path,
                    n
                );
            }
            Err(e) => tracing::warn!(
                "AXIS: failed to load persisted IVF index for '{}' ({}); falling back to rebuild",
                collection_id,
                e
            ),
        }
    }

    /// Register an IVF routing strategy for a warm-loaded index so `query()`
    /// routes it through the IVF path (mirrors the build-time strategy shape).
    async fn register_loaded_ivf_strategy(&self, collection_id: &str, dimension: usize, n: usize) {
        use crate::index::axis::types::{
            Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification,
        };
        let nlist = ((n as f32).sqrt() as usize * 2).clamp(16, 256) as u32;
        let nprobe = (nlist / 2).max(1);
        let strategy = IndexSelectionStrategy {
            indexes: vec![IndexSpecification::new(
                Data::DenseVector { dimension },
                IndexAlgorithm::IVF {
                    nlist,
                    nprobe,
                    quantizer: None,
                },
            )],
            routing_rules: vec![],
        };
        let _ = self
            .update_collection_strategy(collection_id, strategy)
            .await;
    }

    /// Phase 8 F4a (TD-094): suspend a collection — evict its in-memory IVF index
    /// to free memory while keeping the persisted `ivf.bin`, the routing strategy,
    /// and the metadata indexes so the catalog stays queryable. The next query
    /// (or `resume_collection`) warm-loads it from disk. Errors when persistence
    /// is disabled (the index could not be resumed) or there is no in-memory IVF
    /// index to evict.
    pub async fn suspend_collection(&self, collection_id: &str) -> Result<()> {
        if !self.persistence_enabled() {
            anyhow::bail!(
                "cannot suspend '{collection_id}': index persistence is not enabled \
                 (no in-memory eviction without a resumable on-disk copy)"
            );
        }
        if !self.has_ivf_index(collection_id).await {
            anyhow::bail!(
                "cannot suspend '{collection_id}': no in-memory IVF index to evict \
                 (not an IVF collection, or already suspended)"
            );
        }

        // Persist the current in-memory index so the on-disk copy is up to date,
        // then evict it. The evicted Arc is the sole long-lived holder, so its
        // memory (centroids / posting lists / vectors) is freed on drop.
        let served = {
            let indexes = self.ivf_indexes.read().await;
            indexes.get(collection_id).cloned()
        };
        if let Some(index) = served {
            self.persist_ivf_index(collection_id, &index).await;
        }
        self.ivf_indexes.write().await.remove(collection_id);
        self.suspended_collections
            .write()
            .await
            .insert(collection_id.to_string());

        tracing::info!(
            "❄️ AXIS: suspended collection '{}' (IVF index evicted, metadata retained)",
            collection_id
        );
        Ok(())
    }

    /// Phase 8 F4a: eagerly resume a suspended collection by warm-loading its IVF
    /// index from disk now (rather than waiting for the next query). Returns
    /// whether an index is served afterward. Lazy resume still happens on `query`.
    pub async fn resume_collection(&self, collection_id: &str) -> Result<bool> {
        self.ensure_ivf_index_loaded(collection_id).await;
        Ok(self.has_ivf_index(collection_id).await)
    }

    /// Phase 8 F4a: whether `collection_id` is currently suspended (its IVF index
    /// was evicted and not yet warm-loaded back).
    pub async fn is_suspended(&self, collection_id: &str) -> bool {
        self.suspended_collections
            .read()
            .await
            .contains(collection_id)
    }

    /// Whether a persisted IVF index exists for `collection_id` (the resumability
    /// signal surfaced in route-health) — checked through the resolved backend, so
    /// it works for object-store URIs as well as the local dir.
    pub async fn has_persisted_ivf_index(&self, collection_id: &str) -> bool {
        match self.index_fs_and_path(collection_id).await {
            Some((fs, path)) => fs.exists(&path).await.unwrap_or(false),
            None => false,
        }
    }

    /// Collection ids that have a trained IVF index with quantized storage —
    /// the candidates the Phase-5 recall observer probes.
    pub async fn quantized_ivf_collections(&self) -> Vec<String> {
        let indexes = self.ivf_indexes.read().await;
        let mut out = Vec::new();
        for (collection_id, index_lock) in indexes.iter() {
            let index = index_lock.read().await;
            if index.is_trained() && index.has_quantized_storage() {
                out.push(collection_id.clone());
            }
        }
        out
    }

    /// Phase-5 recall observer: probe quantized-vs-exact recall over `queries`
    /// and feed the outcome into the recall-probe gate. Returns the resulting
    /// `ProbeState`, or `None` if there's nothing to probe (no gate, no quantized
    /// IVF index for the collection, untrained index, or empty queries). The
    /// gate opens after `passes_required` consecutive passes (default 3), at
    /// which point `query_ivf` starts selecting the quantized route.
    pub async fn probe_and_observe(
        &self,
        collection_id: &str,
        queries: &[Vec<f32>],
        k: usize,
        recall_floor: f32,
    ) -> Option<crate::catalog::ProbeState> {
        let gate = self.recall_probe_gate.as_ref()?;
        if queries.is_empty() || k == 0 {
            return None;
        }
        let indexes = self.ivf_indexes.read().await;
        let index = indexes.get(collection_id)?.read().await;
        if !index.is_trained() || !index.has_quantized_storage() {
            return None;
        }

        let mut recalls: Vec<f32> = Vec::with_capacity(queries.len());
        for q in queries {
            let exact = index.search(q, k, None).await.ok()?;
            let quant = index
                .search_with_quantized_acceleration(q, k, None)
                .await
                .ok()?;
            let exact_ids: Vec<String> = exact.into_iter().map(|(id, _)| id).collect();
            let quant_ids: Vec<String> = quant.into_iter().map(|(id, _)| id).collect();
            recalls.push(recall_at_k(&exact_ids, &quant_ids, k));
        }
        if recalls.is_empty() {
            return None;
        }
        let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
        let outcome = recall_outcome(mean_recall, recall_floor);
        let scope = crate::catalog::ProbeScope::new(collection_id, collection_id);
        let state = gate.observe(&scope, outcome).await;
        tracing::info!(
            target: "axis_diag",
            site = "recall_observer.probe",
            collection_id = collection_id,
            mean_recall = mean_recall,
            recall_floor = recall_floor,
            outcome = ?outcome,
            gate_open = state.gate_open,
            consecutive_passes = state.consecutive_passes,
            "recall probe observed"
        );
        Some(state)
    }

    /// Set shared collection cache from VectorOperationsService
    pub fn set_shared_collection_cache(
        &mut self,
        cache: Arc<dashmap::DashMap<String, Arc<crate::proto::proximadb_v1::Collection>>>,
    ) {
        self.shared_collection_cache = Some(cache);
        tracing::info!("🔗 AXIS: Shared collection cache set for read-only access");
    }

    /// Get collection's IndexConfig from collection service for index build decisions
    pub async fn get_native_index_config(
        &self,
        collection_id: &str,
    ) -> Result<crate::index::config::RuntimeIndexConfig> {
        if let Some(collection_service) = &self.collection_service {
            match collection_service.native_index_config(collection_id).await {
                Ok(Some(config)) => {
                    tracing::debug!(
                        "📋 AXIS: Retrieved IndexConfig for collection: {}",
                        collection_id
                    );
                    Ok(config)
                }
                Ok(None) => {
                    tracing::warn!(
                        "⚠️ AXIS: Collection not found for IndexConfig: {}",
                        collection_id
                    );
                    // Return default IndexConfig as fallback
                    Ok(crate::index::config::RuntimeIndexConfig::default())
                }
                Err(e) => {
                    tracing::error!(
                        "❌ AXIS: Failed to retrieve IndexConfig for collection {}: {}",
                        collection_id,
                        e
                    );
                    // Return default IndexConfig as fallback
                    Ok(crate::index::config::RuntimeIndexConfig::default())
                }
            }
        } else {
            tracing::warn!("⚠️ AXIS: Collection service not available, using default IndexConfig");
            // Default implementation: return default IndexConfig
            Ok(crate::index::config::RuntimeIndexConfig::default())
        }
    }

    /// Insert a canonical ProximaRecord into the AXIS index with adaptive indexing.
    pub async fn insert<R>(&self, collection_id: &str, vector: &R) -> Result<()>
    where
        R: Clone + Into<ProximaRecord>,
    {
        let vector: ProximaRecord = vector.clone().into();

        // Ensure we have a search_strategy for this collection
        self.ensure_collection_strategy(collection_id).await?;

        // Check if vector is expired (MVCC support)
        if let Some(valid_to_ns) = vector.valid_to_ns {
            let now_ns = Utc::now().timestamp_nanos_opt().unwrap_or(0);
            if valid_to_ns <= now_ns {
                return Ok(());
            }
        }

        // Quantization is handled internally by the indexes; pass through as-is
        let processed_vector = vector.clone();

        if !vector.oid.is_empty() {
            let mut collection_vectors = self.collection_vectors.write().await;
            collection_vectors
                .entry(collection_id.to_string())
                .or_default()
                .insert(vector.oid.clone(), vector.clone());
        }

        // Insert into appropriate indexes based on current search_strategy
        let search_strategy = self.get_collection_strategy(collection_id).await?;
        let vec_values = processed_vector
            .embeddings
            .first()
            .map(|e| e.as_fp32_slice())
            .unwrap_or(&[]);
        let has_dense_vector_index = search_strategy
            .indexes
            .iter()
            .any(|index_spec| matches!(index_spec.data_type, Data::DenseVector { .. }));

        // HMGI is only used when explicitly opted-in via
        // `enable_hmgi(...)` (operator action) or
        // `maybe_auto_enable_hmgi(...)` (background detection task).
        // The insert path itself no longer triggers enablement — see
        // `ensure_hmgi_collection_enabled` for the rationale.
        // Collections without HMGI fall through to
        // `insert_into_hnsw`, which already honors the collection's
        // configured distance metric.
        if has_dense_vector_index && !processed_vector.oid.is_empty() && !vec_values.is_empty() {
            self.ensure_hmgi_collection_enabled(collection_id).await?;
        }

        // Insert into global ID index if ID is present
        if !processed_vector.oid.is_empty() {
            self.global_id_index
                .insert(
                    processed_vector.oid.clone(),
                    collection_id,
                    &processed_vector,
                )
                .await?;
        }

        // Insert into other indexes based on search_strategy
        for index_spec in &search_strategy.indexes {
            match index_spec.data_type {
                Data::Metadata => {
                    self.metadata_index.insert(&processed_vector).await?;
                }
                Data::DenseVector { .. } => {
                    self.insert_dense_vector_index(
                        collection_id,
                        &processed_vector,
                        &index_spec.algorithm,
                    )
                    .await?;
                }
                Data::SparseVector { .. } => {
                    self.sparse_vector_index.insert(&processed_vector).await?;
                }
                _ => {} // Handle other data types
            }
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_vectors_indexed += 1;

        // Check if we should evaluate search_strategy change
        self.maybe_evaluate_strategy(collection_id).await?;

        Ok(())
    }

    /// Insert a canonical ProximaRecord into the AXIS index.
    pub async fn insert_record(&self, collection_id: &str, record: &ProximaRecord) -> Result<()> {
        self.insert(collection_id, record).await
    }

    /// Delete a vector (soft delete with expires_at)
    pub async fn delete(&self, collection_id: &str, vector_id: VectorId) -> Result<()> {
        // For MVCC, we don't actually delete - we set expires_at to now
        // This is handled by the storage layer creating a tombstone

        // Skip if vector_id is empty
        if vector_id.is_empty() {
            return Ok(());
        }

        // Ensure we have a search_strategy for this collection
        self.ensure_collection_strategy(collection_id).await?;

        // Remove from indexes
        let search_strategy = self.get_collection_strategy(collection_id).await?;

        {
            let mut collection_vectors = self.collection_vectors.write().await;
            if let Some(vectors) = collection_vectors.get_mut(collection_id) {
                vectors.remove(&vector_id);
            }
            let should_remove_collection = collection_vectors
                .get(collection_id)
                .map(|vectors| vectors.is_empty())
                .unwrap_or(false);
            if should_remove_collection {
                collection_vectors.remove(collection_id);
            }
        }

        self.global_id_index.remove(&vector_id).await?;

        for index_spec in &search_strategy.indexes {
            match index_spec.data_type {
                Data::Metadata => {
                    self.metadata_index.remove(&vector_id).await?;
                }
                Data::DenseVector { .. } => {
                    self.remove_dense_vector_index(collection_id, &vector_id)
                        .await?;
                }
                Data::SparseVector { .. } => {
                    self.sparse_vector_index.remove(&vector_id).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Query vectors using adaptive indexes
    pub async fn query(&self, query: AxisHybridQuery) -> Result<AxisManagerQueryResult> {
        let start = std::time::Instant::now();

        // Execute query using current search_strategy
        let collection_id = &query.collection_id;

        // TD-087 Slice B: lazily warm a cold collection's IVF index from disk
        // before routing (no-op when already served or persistence is disabled).
        self.ensure_ivf_index_loaded(collection_id).await;

        // Ensure we have a search_strategy for this collection
        self.ensure_collection_strategy(collection_id).await?;

        let search_strategy = self.get_collection_strategy(collection_id).await?;

        let hmgi_query_safe = self.is_hmgi_routable_query(&query);

        // Query using the appropriate index based on strategy algorithm
        // HNSW: O(log N) search - best for search latency
        // IVF: O(√N) search - acceptable if insert-optimized
        let has_filters = !query.metadata_filters.is_empty() || !query.id_filters.is_empty();
        // TD-064: track shortfall across paths (only inline path produces it
        // today; pre-filter exact never undercounts and post-filter is not
        // wired here yet).
        let mut predicate_shortfall: Option<
            crate::observability::search_plan_trace::PredicateShortfall,
        > = None;

        // TD-064 / ADR-011: when the caller supplied both a filtering policy
        // and a selectivity estimate, the catalog policy drives the mode
        // selection. Otherwise we fall back to the legacy
        // `query.ann_filtering_mode` hint.
        //
        // Backward compat: policy-driven routing is only taken when the
        // caller opts in via a non-None policy. Legacy callers that set
        // only `ann_filtering_mode` keep the historical behavior where
        // PostFilter / PreFilter / unspecified all fall through to the
        // exact-filtered scan; only `Inline` was historically wired into
        // the HNSW predicate path. This avoids regressing tests and
        // production paths that depended on that mapping.
        let policy_driven_mode: Option<AnnFilteringMode> = match (
            query.ann_filtering_policy.as_ref(),
            query.estimated_selectivity,
        ) {
            (Some(policy), Some(selectivity)) => {
                Some(ann_mode_from_catalog(policy.routing_mode(selectivity)))
            }
            _ => None,
        };
        let effective_mode: AnnFilteringMode =
            policy_driven_mode.unwrap_or(query.ann_filtering_mode);
        let mut selected_filtering_mode: Option<AnnFilteringMode> = None;

        // Determine which inner search path will run. Promote to
        // info so the bench's tracing layer captures it — the same
        // log was previously trace and silently dropped, masking
        // the fact that the HNSW cell wasn't actually exercising
        // query_hnsw at all (it was hitting an entirely different
        // path, presumably IVF or the empty-result fallback).
        // TD-075: which IVF route ran — Some(true)=quantized accelerator used,
        // Some(false)=quantized storage present but gate forced exact, None=n/a.
        let mut quantized_route: Option<bool> = None;
        let results = if self.is_hmgi_enabled(collection_id).await && hmgi_query_safe {
            tracing::info!(
                target: "axis_diag",
                site = "axis_manager.query",
                route = "search_hmgi",
                collection_id = collection_id,
                "query routed through HMGI"
            );
            self.search_hmgi(collection_id, &query, query.top_k).await?
        } else if has_filters && effective_mode == AnnFilteringMode::Inline {
            // ADR-011 Inline: thread predicate into HNSW walk (ACORN semantics).
            // query_hnsw_with_predicate evaluates ID filters and record-backed
            // metadata predicates during traversal, then reapplies a residual
            // guard before returning top-k.
            selected_filtering_mode = Some(AnnFilteringMode::Inline);
            let (results, shortfall) = self
                .query_hnsw_with_predicate(collection_id, &query, AnnFilteringMode::Inline)
                .await?;
            predicate_shortfall = shortfall;
            results
        } else if has_filters && policy_driven_mode == Some(AnnFilteringMode::PostFilter) {
            // ADR-011 PostFilter (policy-driven only): ANN first then
            // filter, with policy-driven oversample. We reuse the inline
            // path's traversal but pass `PostFilter` so the oversample
            // factor uses `AnnFilteringPolicy::effective_top_k_for_post_filter`
            // and the shortfall is tagged with the correct mode label.
            // Legacy callers that set `ann_filtering_mode = PostFilter`
            // without a policy fall through to the exact path below — this
            // preserves historical behavior.
            selected_filtering_mode = Some(AnnFilteringMode::PostFilter);
            let (results, shortfall) = self
                .query_hnsw_with_predicate(collection_id, &query, AnnFilteringMode::PostFilter)
                .await?;
            predicate_shortfall = shortfall;
            results
        } else if has_filters {
            // ADR-011 PreFilter: evaluate scalar predicates first, then exact
            // vector scoring over the candidate set.
            selected_filtering_mode = Some(AnnFilteringMode::PreFilter);
            self.execute_exact_filtered_query(collection_id, &query)
                .await?
        } else {
            // Find the dense vector index spec to determine algorithm
            let use_ivf = search_strategy.indexes.iter().any(|spec| {
                matches!(spec.data_type, Data::DenseVector { .. })
                    && matches!(
                        spec.algorithm,
                        IndexAlgorithm::IVF { .. } | IndexAlgorithm::PQ { .. }
                    )
            });

            if use_ivf {
                tracing::info!(
                    target: "axis_diag",
                    site = "axis_manager.query",
                    route = "query_ivf",
                    collection_id = collection_id,
                    "query routed through IVF"
                );
                // TD-075: consult the recall-probe gate before the IVF route may
                // select quantized acceleration. Collection-scoped probe (no
                // tenant in AxisHybridQuery yet — tenant threading is follow-up).
                let gate_open = match &self.recall_probe_gate {
                    Some(gate) => {
                        gate.is_open(&crate::catalog::ProbeScope::new(
                            collection_id,
                            collection_id,
                        ))
                        .await
                    }
                    None => false,
                };
                let (ivf_results, route) = self.query_ivf(collection_id, &query, gate_open).await?;
                quantized_route = route;
                ivf_results
            } else {
                tracing::info!(
                    target: "axis_diag",
                    site = "axis_manager.query",
                    route = "query_hnsw",
                    collection_id = collection_id,
                    "query routed through legacy HNSW"
                );
                self.query_hnsw(collection_id, &query).await?
            }
        };

        // Filter out expired results (MVCC)
        let active_results: Vec<_> = results
            .into_iter()
            .filter(|result| {
                if query.include_expired {
                    return true;
                }
                // Check if result is not expired
                if let Some(expires_at) = result.expires_at.as_ref() {
                    Utc::now() < *expires_at
                } else {
                    true // No expiration
                }
            })
            .collect();

        Ok(AxisManagerQueryResult {
            results: active_results,
            strategy_used: search_strategy,
            execution_time_ms: start.elapsed().as_millis() as u64,
            predicate_shortfall,
            selected_filtering_mode,
            quantized_route,
        })
    }

    /// Insert dense vectors through the canonical HMGI path.
    ///
    /// Legacy collection-scoped HNSW/IVF remains as a fallback for collections
    /// that have not been initialized for HMGI, but normal AXIS dense indexing is
    /// per-modality so insert and query semantics match.
    async fn insert_dense_vector_index(
        &self,
        collection_id: &str,
        vector: &ProximaRecord,
        algorithm: &IndexAlgorithm,
    ) -> Result<()> {
        if self.is_hmgi_enabled(collection_id).await {
            // Only log the first insert per collection to avoid log
            // spam at 10K+ scale. The collection_vectors map is the
            // canonical "already seen this collection" signal.
            let first = self
                .collection_vectors
                .read()
                .await
                .get(collection_id)
                .is_none_or(|v| v.is_empty());
            if first {
                tracing::info!(
                    target: "axis_diag",
                    site = "insert_dense_vector_index",
                    branch = "hmgi",
                    collection_id = collection_id,
                    "FIRST INSERT — routing through HMGI"
                );
            }
            self.insert_hmgi(collection_id, vector.clone(), Some(algorithm))
                .await?;
            return Ok(());
        }

        let first = self
            .collection_vectors
            .read()
            .await
            .get(collection_id)
            .is_none_or(|v| v.is_empty());

        match algorithm {
            IndexAlgorithm::HNSW { .. } => {
                if first {
                    tracing::info!(
                        target: "axis_diag",
                        site = "insert_dense_vector_index",
                        branch = "hnsw",
                        collection_id = collection_id,
                        algorithm = ?algorithm,
                        "FIRST INSERT — routing through legacy HNSW"
                    );
                }
                self.insert_into_hnsw(collection_id, vector, Some(algorithm))
                    .await
            }
            IndexAlgorithm::IVF { .. } | IndexAlgorithm::PQ { .. } => {
                if first {
                    tracing::info!(
                        target: "axis_diag",
                        site = "insert_dense_vector_index",
                        branch = "ivf_pq",
                        collection_id = collection_id,
                        algorithm = ?algorithm,
                        "FIRST INSERT — routing through IVF/PQ"
                    );
                }
                self.insert_into_ivf(collection_id, vector, Some(algorithm))
                    .await
            }
            other => {
                if first {
                    tracing::info!(
                        target: "axis_diag",
                        site = "insert_dense_vector_index",
                        branch = "fallback_hnsw",
                        collection_id = collection_id,
                        algorithm = ?other,
                        "FIRST INSERT — algorithm not in match, falling back to HNSW"
                    );
                }
                // Non-HNSW algorithm asked to use HNSW — fall back
                // with no spec to extract, so HNSW config uses its
                // own defaults.
                self.insert_into_hnsw(collection_id, vector, None).await
            }
        }
    }

    /// Remove dense vectors through the same layout used for dense inserts.
    async fn remove_dense_vector_index(
        &self,
        collection_id: &str,
        vector_id: &VectorId,
    ) -> Result<()> {
        if self.is_hmgi_enabled(collection_id).await {
            self.remove_hmgi_vector(collection_id, vector_id).await?;
            return Ok(());
        }

        use crate::index::axis::index_factory::AxisVectorIndex;

        if let Some(index) = self.hnsw_indexes.read().await.get(collection_id).cloned() {
            index.remove(vector_id).await?;
        }

        if let Some(index) = self.ivf_indexes.read().await.get(collection_id).cloned() {
            index.read().await.remove(vector_id).await?;
        }

        {
            let mut pending = self.ivf_pending_vectors.write().await;
            if let Some(vectors) = pending.get_mut(collection_id) {
                vectors.retain(|(id, _)| id != vector_id);
            }
        }

        self.dense_vector_index.remove(vector_id).await
    }

    /// Insert a vector into the real HNSW index for a collection
    async fn insert_into_hnsw(
        &self,
        collection_id: &str,
        vector: &ProximaRecord,
        algorithm: Option<&IndexAlgorithm>,
    ) -> Result<()> {
        use crate::compute::distance_computation::DistanceMetric;

        use crate::index::axis::index_factory::AxisVectorIndex;
        use crate::index::axis::indexes::hnsw_index::{AxisHnswConfig, AxisHnswIndex};

        let vec_values = vector
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();
        // Get or create HNSW index for this collection
        let dimension = vec_values.len();
        if dimension == 0 || vector.oid.is_empty() {
            return Ok(()); // Skip empty vectors or missing IDs
        }

        // Check if index exists, if not create it
        {
            let indexes = self.hnsw_indexes.read().await;
            if !indexes.contains_key(collection_id) {
                drop(indexes);

                // Get collection's distance metric from its config
                // This ensures HNSW uses the same metric as the collection
                let resolved_metric = self.get_collection_distance_metric(collection_id).await;
                let distance_metric = resolved_metric.unwrap_or(DistanceMetric::DotProduct); // Default to DotProduct for compatibility with FAISS/benchmarks

                // **End-to-end ef_search wiring (2026-05-29)**: previously
                // this site built `AxisHnswConfig { distance_metric,
                // ..Default::default() }`, ignoring the strategy spec's
                // m / ef_construction / ef_search entirely. Customers
                // (and the bench) had to use the (un-prefixed) BENCH_HNSW_EF
                // env override to influence search behaviour because
                // the `IndexAlgorithm::HNSW { m, ef_construction,
                // ef_search, max_elements }` fields they put in their
                // catalog spec were silently dropped. Now those fields
                // flow through into the partition config so DotProduct
                // workloads (which need ef ~ 5×sqrt(N) for >0.90
                // recall) can configure it without env vars.
                let (config_m, config_ef_construction, config_ef_search) = match algorithm {
                    Some(IndexAlgorithm::HNSW {
                        m,
                        ef_construction,
                        ef_search,
                        ..
                    }) => (*m, *ef_construction, *ef_search),
                    _ => {
                        let defaults = AxisHnswConfig::default();
                        (
                            defaults.m as u32,
                            defaults.ef_construction as u32,
                            defaults.ef as u32,
                        )
                    }
                };

                let config = AxisHnswConfig {
                    distance_metric,
                    m: config_m as usize,
                    ef_construction: config_ef_construction as usize,
                    ef: config_ef_search as usize,
                    ..Default::default()
                };

                tracing::info!(
                    target: "axis_diag",
                    site = "insert_into_hnsw",
                    collection_id = collection_id,
                    resolved_metric = ?resolved_metric,
                    using_metric = ?distance_metric,
                    m = config.m,
                    ef_construction = config.ef_construction,
                    ef_search = config.ef,
                    spec_source = if algorithm.is_some() { "strategy" } else { "default" },
                    "legacy HNSW path resolved config"
                );
                tracing::info!(
                    "🔗 AXIS: Creating HNSW index for collection {} with metric {:?}",
                    collection_id,
                    distance_metric
                );

                let index = AxisHnswIndex::new_with_collection(
                    Some(collection_id.to_string()),
                    config,
                    dimension,
                )?;
                let mut indexes = self.hnsw_indexes.write().await;
                indexes.insert(collection_id.to_string(), Arc::new(index));
                tracing::debug!(
                    "🔗 AXIS: Created new HNSW index for collection {} (dimension={}, metric={:?})",
                    collection_id,
                    dimension,
                    distance_metric
                );
            }
        }

        // Insert into the index using the AxisVectorIndex trait.
        //
        // TD-064: extract filterable metadata and use `add_with_metadata` so
        // the index caches its own policy-bearing fields (tenant_id, RLS tags,
        // created_at_ns, expires_at_ns). Typed-attr extraction is gated on
        // a collection-level `FilterableFieldsConfig` once that lands; for
        // now the default config still extracts the core fields.
        let indexes = self.hnsw_indexes.read().await;
        if let Some(index) = indexes.get(collection_id) {
            let filterable_metadata =
                crate::index::axis::filterable_metadata::extract_filterable_metadata(
                    vector,
                    &crate::index::axis::filterable_metadata::FilterableFieldsConfig::default(),
                );
            index
                .add_with_metadata(vector.oid.clone(), vec_values, &filterable_metadata)
                .await?;
        }

        Ok(())
    }

    /// Query vectors from the real HNSW index.
    ///
    /// **Score-units note**: `AxisVectorIndex::search` returns
    /// `(id, raw_distance)` where the f32 is a metric-native value
    /// with "lower = closer" semantics (HNSW's `metric_aware_distance`
    /// negates DotProduct internally so heap ordering is consistent).
    /// `ScoredResult.similarity` is contractually higher = better in
    /// [0, 1], so we convert through `SimilarityResult` here. Without
    /// this conversion, sorting descending by `similarity` returns
    /// the FARTHEST records first (was the recall=0 bug in HMGI
    /// before commit b3985b59c). Mirroring the fix on this legacy
    /// path.
    async fn query_hnsw(
        &self,
        collection_id: &str,
        query: &AxisHybridQuery,
    ) -> Result<Vec<ScoredResult>> {
        use crate::compute::distance_computation::engine::{DistanceMetricExt, SimilarityResult};
        use crate::index::axis::index_factory::AxisVectorIndex;

        let indexes = self.hnsw_indexes.read().await;
        if let Some(index) = indexes.get(collection_id) {
            // Extract query vector
            if let Some(VectorQuery::Dense { vector, .. }) = &query.vector_query {
                // Map the caller's search effort onto the HNSW `ef` budget so
                // `approximate`/`approximate:N` actually trade recall for
                // latency on the warm path. `None` ⇒ index size-aware default.
                let ef_override = query
                    .search_effort
                    .and_then(|effort| effort.hnsw_ef_override(query.top_k));
                let results = index
                    .search_with_effort(vector, query.top_k, ef_override, None)
                    .await?;
                let metric = index.distance_metric();
                return Ok(results
                    .into_iter()
                    .map(|(id, raw_distance)| {
                        let raw_for_similarity = if metric.is_similarity() {
                            -raw_distance
                        } else {
                            raw_distance
                        };
                        let similarity =
                            SimilarityResult::new(raw_for_similarity, metric).normalized_score;
                        let expires_at = self.lookup_record_expiration(collection_id, &id);
                        ScoredResult {
                            vector_id: id,
                            similarity,
                            expires_at,
                        }
                    })
                    .collect());
            }
        }

        // Return empty if no index or no dense vector query
        Ok(Vec::new())
    }

    /// ADR-011 Inline mode: HNSW traversal with predicate closure (ACORN semantics).
    ///
    /// Builds a record-aware predicate for inline HNSW traversal.
    ///
    /// ID filters are evaluated directly. Metadata filters are evaluated against
    /// the current in-memory ProximaRecord projection while the record/PAX lookup
    /// bridge is still being wired. Missing metadata fails closed, and the same
    /// predicate is applied again as a residual guard before returning results.
    ///
    /// TD-064: Returns `(results, Option<PredicateShortfall>)` so the caller
    /// can surface recall-shortfall disclosure on `SearchPlanTrace`. The
    /// shortfall is populated only when post-filter trimming drops the
    /// survivor count below `query.top_k`.
    ///
    /// `effective_mode` is the catalog-policy-derived mode the caller
    /// selected (Inline or PostFilter). It controls oversample sizing
    /// — PostFilter uses
    /// `AnnFilteringPolicy::effective_top_k_for_post_filter(top_k)` when
    /// a policy is present on the query; Inline uses a 2× default. The
    /// mode is also tagged onto the shortfall record so EXPLAIN can
    /// distinguish which path produced the shortfall.
    async fn query_hnsw_with_predicate(
        &self,
        collection_id: &str,
        query: &AxisHybridQuery,
        effective_mode: AnnFilteringMode,
    ) -> Result<(
        Vec<ScoredResult>,
        Option<crate::observability::search_plan_trace::PredicateShortfall>,
    )> {
        let index = {
            let indexes = self.hnsw_indexes.read().await;
            let Some(index) = indexes.get(collection_id).cloned() else {
                return Ok((Vec::new(), None));
            };
            index
        };
        let Some(VectorQuery::Dense { vector, .. }) = &query.vector_query else {
            return Ok((Vec::new(), None));
        };

        let metadata_expression = if query.metadata_filters.is_empty() {
            None
        } else {
            Some(self.metadata_filters_to_expression(&query.metadata_filters))
        };

        // Lazy per-candidate metadata resolution: the traversal probes only
        // ~ef·M nodes, so materializing `record_filter_metadata` for EVERY
        // record in the collection per query was O(N) map builds + String
        // clones disproportionate to use. The resolver borrows the collection
        // under its read guard and memoizes each id on first touch — verdicts
        // are identical (same maps, same filter); only WHEN they are built
        // changes. Worst case (a sparse predicate draining the whole graph)
        // degenerates to the previous O(N) cost, never worse.
        let lazy_metadata =
            LazyFilterMetadata::new(self, collection_id, metadata_expression.is_some()).await;

        // F7 / TD-HYBRID-1 D2: one compiled predicate (id filters + metadata
        // expression) shared by the traversal closure below and the residual
        // re-filter further down, replacing three hand-rolled copies.
        let compiled_filter = crate::core::search::CompiledGlobalFilter::new(
            query.id_filters.clone(),
            metadata_expression.clone(),
        );
        let predicate_filter = compiled_filter.clone();
        let predicate = {
            let resolver = &lazy_metadata;
            move |id: &str| -> bool {
                predicate_filter.matches(id, resolver.resolve(id).as_deref())
            }
        };

        // TD-064 / ADR-011 §4.3: oversample to absorb post-filter recall loss.
        // PostFilter uses the catalog policy's effective_top_k_for_post_filter
        // (default 2×) when a policy is present; Inline keeps the 2× default
        // regardless. Either way we apply max(top_k) so we never request
        // FEWER candidates than the caller asked for.
        //
        // NB (1.3 investigation, 2026-06-14): oversample *escalation* on shortfall
        // was evaluated and rejected — `search_layer_predicate` pushes ALL
        // neighbours onto an unbounded frontier and only early-terminates once it
        // has `ef` PASSING results, so when matches are sparse it already drains
        // the full reachable graph regardless of `oversample_k`. A larger pool
        // therefore cannot recover more matches; the real lever for filtered
        // recall is graph connectivity (ACORN `gamma`), not oversampling.
        let oversample_k = match (effective_mode, query.ann_filtering_policy.as_ref()) {
            (AnnFilteringMode::PostFilter, Some(policy)) => {
                policy.effective_top_k_for_post_filter(query.top_k)
            }
            _ => query.top_k.saturating_mul(2),
        }
        .max(query.top_k);

        // Map the caller's effort onto the predicate traversal `ef` (recall vs
        // latency). `oversample_k` above is the orthogonal post-filter pool and
        // is left to the filtering policy.
        let ef_override = query
            .search_effort
            .and_then(|effort| effort.hnsw_ef_override(query.top_k));
        let raw = index
            .search_with_predicate_fn(vector, oversample_k, predicate, ef_override)
            .await?;

        // See query_hnsw for the score-units rationale — raw values
        // out of HNSW are lower-better metric-native distances; we
        // convert through SimilarityResult so ScoredResult.similarity
        // honors the higher=better contract.
        use crate::compute::distance_computation::engine::{DistanceMetricExt, SimilarityResult};
        let metric = index.distance_metric();
        // Every raw result passed `predicate`, so each id below is already
        // memoized in the resolver — the residual re-filter costs only map
        // hits. Dropping the resolver (and its collection read guard) here
        // keeps `lookup_record_expiration`'s try_read seeing the same lock
        // state it did before this change.
        // Every raw result passed `predicate`, so each id below is already
        // memoized in the resolver — the residual re-filter costs only map
        // hits.
        let filtered: Vec<(String, f32)> = {
            let resolver = &lazy_metadata;
            raw.into_iter()
                .filter(|(id, _)| compiled_filter.matches(id, resolver.resolve(id).as_deref()))
                .take(query.top_k)
                .collect()
        };
        // Release the collection read guard BEFORE the expiration lookups:
        // `lookup_record_expiration` try_reads the same lock, and holding the
        // guard here would flip those lookups to the fallback path.
        drop(lazy_metadata);

        let results: Vec<ScoredResult> = filtered
            .into_iter()
            .map(|(id, raw_distance)| {
                let raw_for_similarity = if metric.is_similarity() {
                    -raw_distance
                } else {
                    raw_distance
                };
                let similarity = SimilarityResult::new(raw_for_similarity, metric).normalized_score;
                let expires_at = self.lookup_record_expiration(collection_id, &id);
                ScoredResult {
                    vector_id: id,
                    similarity,
                    expires_at,
                }
            })
            .collect();

        // TD-064: record shortfall when post-filter trimmed below top_k.
        // Four observability channels (the trace field is the canonical
        // gateway-facing one; the rest are for operators / cross-layer
        // wiring):
        //   - Prometheus counter (operator dashboards)
        //   - structured tracing event (SIEM/log pipelines)
        //   - task-local diagnostics bus (REST/gRPC handler reads at
        //     trace-build time without intermediate layers having to
        //     declare a predicate_shortfall field)
        //   - PredicateShortfall on the returned tuple (direct callers /
        //     unit tests / future end-to-end plumbs)
        let mode_label: &'static str = match effective_mode {
            AnnFilteringMode::Inline => "inline",
            AnnFilteringMode::PostFilter => "post_filter",
            AnnFilteringMode::PreFilter => "pre_filter",
        };
        let shortfall = if results.len() < query.top_k {
            crate::metrics::td064_metrics::record_shortfall(
                collection_id,
                mode_label,
                query.top_k as u32,
                results.len() as u32,
            );
            tracing::warn!(
                target: "axis.predicate_shortfall",
                collection_id = %collection_id,
                ann_filtering_mode = mode_label,
                requested_k = query.top_k,
                returned_k = results.len(),
                oversample_pool = oversample_k,
                "TD-064: predicate-aware ANN returned fewer matches than requested top_k"
            );
            let sf = crate::observability::search_plan_trace::PredicateShortfall {
                requested_k: query.top_k as u32,
                returned_k: results.len() as u32,
                oversample_pool: oversample_k as u32,
                ann_filtering_mode: mode_label.to_string(),
            };
            crate::observability::predicate_diagnostics::record_shortfall(sf.clone());
            Some(sf)
        } else {
            None
        };

        Ok((results, shortfall))
    }

    /// Insert a vector into the IVF index for a collection (DEFAULT for incremental workloads)
    /// Insert vector into IVF index (DEFAULT)
    ///
    /// IVF requires k-means training before vectors can be added:
    /// 1. Buffer vectors until we have min_train_size (100 vectors)
    /// 2. Train index with buffered vectors to build centroids
    /// 3. Add all buffered vectors to trained index
    /// 4. Future inserts go directly to trained index
    async fn insert_into_ivf(
        &self,
        collection_id: &str,
        vector: &ProximaRecord,
        algorithm: Option<&IndexAlgorithm>,
    ) -> Result<()> {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::indexes::dual_store_ivf::{UnifiedIvfConfig, UnifiedIvfIndex};

        let vec_values = vector
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();
        let dimension = vec_values.len();
        if dimension == 0 || vector.oid.is_empty() {
            return Ok(()); // Skip empty vectors or missing IDs
        }

        const MIN_TRAIN_SIZE: usize = 100; // Minimum vectors needed for k-means training

        // Check if index exists and is trained
        let index_exists_and_trained = {
            let indexes = self.ivf_indexes.read().await;
            if let Some(index) = indexes.get(collection_id) {
                let idx = index.read().await;
                idx.is_trained()
            } else {
                false
            }
        };

        if index_exists_and_trained {
            // Index is trained, add vector directly.
            //
            // TD-064: insert via the AxisVectorIndex trait so the index
            // also caches filterable metadata for predicate-aware search.
            use crate::index::axis::index_factory::AxisVectorIndex;
            let indexes = self.ivf_indexes.read().await;
            if let Some(index) = indexes.get(collection_id) {
                let idx = index.read().await;
                let filterable_metadata =
                    crate::index::axis::filterable_metadata::extract_filterable_metadata(
                        vector,
                        &crate::index::axis::filterable_metadata::FilterableFieldsConfig::default(),
                    );
                idx.add_with_metadata(vector.oid.clone(), vec_values.clone(), &filterable_metadata)
                    .await?;
            }
            return Ok(());
        }

        // Buffer the vector for training
        {
            let mut pending = self.ivf_pending_vectors.write().await;
            let buffer = pending.entry(collection_id.to_string()).or_default();
            buffer.push((vector.oid.clone(), vec_values.clone()));

            // Check if we have enough vectors to train
            if buffer.len() >= MIN_TRAIN_SIZE {
                tracing::info!(
                    "🎯 AXIS: IVF training triggered for collection {} with {} vectors",
                    collection_id,
                    buffer.len()
                );

                // Take ownership of buffered vectors
                let training_vectors: Vec<(String, Vec<f32>)> = std::mem::take(buffer);
                drop(pending); // Release the lock

                // **End-to-end IVF strategy-spec wiring (2026-05-30)**:
                // when the caller passed an `IndexAlgorithm::IVF`
                // spec (via update_collection_strategy or the
                // catalog), honor its nlist / nprobe instead of the
                // hardcoded sqrt-based formula. Without this, the
                // strategy spec is silently dropped — analogous to
                // the HNSW.ef_search bug fixed in 476dc951a. The
                // sqrt-based formula is kept as a fallback for the
                // adaptive-engine path that doesn't supply a spec.
                let (n_clusters, n_probe) = match algorithm {
                    Some(IndexAlgorithm::IVF { nlist, nprobe, .. }) => {
                        (*nlist as usize, *nprobe as usize)
                    }
                    _ => {
                        // Original incremental-training fallback:
                        // n_clusters = clamp(sqrt(N) * 2, 16, 256)
                        let n_clusters = {
                            let sqrt_based = (training_vectors.len() as f32).sqrt() as usize * 2;
                            const MIN_CLUSTERS: usize = 16;
                            const MAX_CLUSTERS: usize = 256;
                            sqrt_based.clamp(MIN_CLUSTERS, MAX_CLUSTERS)
                        };
                        // n_probe = max(n_clusters/2, sqrt(n_clusters)*3)
                        let n_probe = {
                            let half_clusters = n_clusters / 2;
                            let sqrt_based = ((n_clusters as f32).sqrt() * 3.0) as usize;
                            std::cmp::max(half_clusters, sqrt_based).min(n_clusters)
                        };
                        (n_clusters, n_probe)
                    }
                };

                tracing::info!(
                    target: "axis_diag",
                    site = "insert_into_ivf",
                    collection_id = collection_id,
                    n_clusters = n_clusters,
                    n_probe = n_probe,
                    training_size = training_vectors.len(),
                    spec_source = if matches!(algorithm, Some(IndexAlgorithm::IVF { .. })) {
                        "strategy"
                    } else {
                        "incremental_default"
                    },
                    "IVF config resolved"
                );

                // **Metric correctness (2026-05-29)**: previously
                // hardcoded `DistanceMetric::Cosine` for every
                // collection. Now resolves from
                // `get_collection_distance_metric` so Euclidean /
                // DotProduct collections actually train and rank by
                // their configured metric. Without this fix, my
                // earlier change to `UnifiedIvfIndex::search`'s
                // inner loop (which now uses
                // `self.config.distance_metric`) silently used
                // Cosine for ALL metrics — Euclidean collections
                // saw recall drop from 1.000 to 0.55 because IVF
                // ranked by cosine while the exact path ranked by
                // euclidean.
                let resolved_metric = self
                    .get_collection_distance_metric(collection_id)
                    .await
                    .unwrap_or(DistanceMetric::Cosine);
                let config = UnifiedIvfConfig {
                    n_clusters,
                    n_probe,
                    dimension,
                    distance_metric: resolved_metric,
                    min_train_size: MIN_TRAIN_SIZE,
                    ..Default::default()
                };

                let mut index = UnifiedIvfIndex::new(collection_id.to_string(), config)?;

                // Train with just the vector data (not IDs)
                let vector_data: Vec<Vec<f32>> =
                    training_vectors.iter().map(|(_, v)| v.clone()).collect();
                index.train(vector_data).await?;

                tracing::info!(
                    "✅ AXIS: IVF index trained for collection {} with {} clusters",
                    collection_id,
                    n_clusters
                );

                // Add all buffered vectors to the trained index.
                //
                // TD-064: cold-start cache hydration. collection_vectors holds
                // the canonical ProximaRecord; extract filterable metadata per
                // id so the first batch of vectors isn't excluded by
                // fail-closed predicate evaluation. Records that aren't found
                // in collection_vectors fall back to add() (metadata gap is
                // contained to those rows).
                use crate::index::axis::index_factory::AxisVectorIndex;
                let collection_vectors_snapshot = self.collection_vectors.read().await;
                let fields_config =
                    crate::index::axis::filterable_metadata::FilterableFieldsConfig::default();
                for (id, vec) in &training_vectors {
                    let metadata = collection_vectors_snapshot
                        .get(collection_id)
                        .and_then(|records| records.get(id))
                        .map(|record| {
                            crate::index::axis::filterable_metadata::extract_filterable_metadata(
                                record,
                                &fields_config,
                            )
                        });
                    match metadata {
                        Some(meta) => {
                            index
                                .add_with_metadata(id.clone(), vec.clone(), &meta)
                                .await?;
                        }
                        None => {
                            index.add_vector(id.clone(), vec.clone(), None).await?;
                        }
                    }
                }
                drop(collection_vectors_snapshot);

                tracing::info!(
                    "✅ AXIS: Added {} vectors to IVF index for collection {}",
                    training_vectors.len(),
                    collection_id
                );

                // Store the trained index
                let served = Arc::new(tokio::sync::RwLock::new(index));
                {
                    let mut indexes = self.ivf_indexes.write().await;
                    indexes.insert(collection_id.to_string(), served.clone());
                }
                // TD-087 Slice B: persist the freshly trained index (best-effort).
                self.persist_ivf_index(collection_id, &served).await;
            }
        }

        Ok(())
    }

    /// Query vectors from the IVF index (DEFAULT)
    async fn query_ivf(
        &self,
        collection_id: &str,
        query: &AxisHybridQuery,
        gate_open: bool,
    ) -> Result<(Vec<ScoredResult>, Option<bool>)> {
        let indexes = self.ivf_indexes.read().await;
        if let Some(index_lock) = indexes.get(collection_id) {
            let index = index_lock.read().await;

            // Check if index is trained
            if !index.is_trained() {
                tracing::debug!(
                    "🔍 AXIS: IVF index for collection {} not yet trained, returning empty results",
                    collection_id
                );
                return Ok((Vec::new(), None));
            }

            if let Some(VectorQuery::Dense { vector, .. }) = &query.vector_query {
                let start = std::time::Instant::now();
                // TD-075: route to the quantized accelerator only when the gate
                // is open AND the index has quantized storage; otherwise exact.
                let has_quantized = index.has_quantized_storage();
                // ADR-023 T-E: a cold-loaded index (`ColdBinaryOnly`) has NO fp32
                // store yet, so exact search would return nothing — force the
                // binary (Stage-1) route regardless of the recall gate. The gate
                // governs the warm (`FullTwoStage`) path normally.
                let cold_binary =
                    index.serving_state() == crate::index::axis::IvfServingState::ColdBinaryOnly;
                let use_quantized = cold_binary || decide_quantized_route(gate_open, has_quantized);
                let route = if has_quantized {
                    Some(use_quantized)
                } else {
                    None
                };
                if has_quantized && !cold_binary && !use_quantized {
                    // TD-075 / F2: quantized storage exists but the recall-probe
                    // gate forced exact — surface this degraded route to EXPLAIN
                    // via the per-request diagnostics bus (no-op outside a scope).
                    crate::observability::predicate_diagnostics::record_quantized_downgrade();
                }
                // Map the caller's accuracy-vs-latency effort onto IVF `nprobe`
                // (cells probed). `None` ⇒ the index's configured default;
                // Exact ⇒ all `nlist` cells; Approximate ⇒ fewer cells, faster.
                let nprobe = query
                    .search_effort
                    .map(|effort| effort.ivf_nprobe(index.nlist()));
                let results = if use_quantized {
                    index
                        .search_with_quantized_acceleration(vector, query.top_k, nprobe)
                        .await?
                } else {
                    index.search(vector, query.top_k, nprobe).await?
                };
                let search_time = start.elapsed();

                tracing::info!(
                    "🔍 AXIS: IVF search completed for collection {} - {} results in {:?} (top_k={}, quantized={})",
                    collection_id,
                    results.len(),
                    search_time,
                    query.top_k,
                    use_quantized
                );

                // Score-units conversion — see query_hnsw for rationale.
                // IVF returns lower-better metric-native distances;
                // ScoredResult.similarity must be higher-better [0,1].
                use crate::compute::distance_computation::engine::{
                    DistanceMetricExt, SimilarityResult,
                };
                let metric = index.distance_metric();
                let scored: Vec<ScoredResult> = results
                    .into_iter()
                    .map(|(id, raw_distance)| {
                        let raw_for_similarity = if metric.is_similarity() {
                            -raw_distance
                        } else {
                            raw_distance
                        };
                        let similarity =
                            SimilarityResult::new(raw_for_similarity, metric).normalized_score;
                        let expires_at = self.lookup_record_expiration(collection_id, &id);
                        ScoredResult {
                            vector_id: id,
                            similarity,
                            expires_at,
                        }
                    })
                    .collect();
                return Ok((scored, route));
            }
        } else {
            tracing::debug!(
                "🔍 AXIS: No IVF index found for collection {}, falling back to storage engine search",
                collection_id
            );
        }

        // Return empty if no index or no dense vector query
        Ok((Vec::new(), None))
    }

    /// Analyze collection and trigger migration if beneficial
    pub async fn analyze_and_optimize(&self, collection_id: &str) -> Result<()> {
        // Check if migration is already in progress
        let migrations = self.active_migrations.read().await;
        if migrations.contains_key(collection_id) {
            return Ok(()); // Migration already in progress
        }
        drop(migrations);

        // Analyze collection characteristics
        let characteristics = self
            .adaptive_engine
            .analyze_collection(collection_id)
            .await?;

        // Determine if migration is beneficial
        let decision = self
            .adaptive_engine
            .should_migrate(collection_id, &characteristics)
            .await?;

        match decision {
            MigrationDecision::Migrate {
                from,
                to,
                estimated_improvement,
                ..
            } => {
                debug!(
                    "AXIS: Initiating migration for collection {} from {} to {} indexes (estimated improvement: {:.2}%)",
                    collection_id,
                    from.indexes.len(),
                    to.indexes.len(),
                    estimated_improvement * 100.0
                );

                // Start migration
                self.start_migration(collection_id, from, to).await?;
            }
            MigrationDecision::Stay { reason } => {
                debug!(
                    "AXIS: Collection {} staying with current // search_strategy removed -  {}",
                    collection_id, reason
                );
            }
        }

        Ok(())
    }

    /// Start migration to new indexing search_strategy
    async fn start_migration(
        &self,
        collection_id: &str,
        from: IndexSelectionStrategy,
        to: IndexSelectionStrategy,
    ) -> Result<()> {
        let migration_id = proximadb_kernel::uuid::Uuid::new_v4();

        // Record migration start
        let mut migrations = self.active_migrations.write().await;
        migrations.insert(
            collection_id.to_string(),
            MigrationStatus {
                migration_id,
                from_strategy: from.clone(),
                to_strategy: to.clone(),
                start_time: Utc::now(),
                progress_percentage: 0.0,
                estimated_completion: None,
            },
        );
        drop(migrations);

        // Execute migration in background
        let migration_engine = self.migration_engine.clone();
        let collection_id = collection_id.to_string();
        let active_migrations = self.active_migrations.clone();
        let collection_strategies = self.collection_strategies.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let result = migration_engine
                .execute_migration(&collection_id, from, to)
                .await;

            // Update status
            let mut migrations = active_migrations.write().await;
            migrations.remove(&collection_id);

            match result {
                Ok(migration_result) => {
                    // Update search_strategy
                    let mut strategies = collection_strategies.write().await;
                    strategies.insert(collection_id.to_string(), migration_result.new_strategy);

                    // Update metrics
                    let mut metrics = metrics.write().await;
                    metrics.total_migrations += 1;
                    metrics.successful_migrations += 1;
                    metrics.average_migration_time_ms = (metrics.average_migration_time_ms
                        * (metrics.total_migrations - 1)
                        + migration_result.duration_ms)
                        / metrics.total_migrations;

                    debug!(
                        "AXIS: Migration completed for collection {} in {}ms",
                        collection_id, migration_result.duration_ms
                    );
                }
                Err(e) => {
                    let mut metrics = metrics.write().await;
                    metrics.total_migrations += 1;
                    metrics.failed_migrations += 1;

                    error!(
                        "AXIS: Migration failed for collection {}: {}",
                        collection_id, e
                    );
                }
            }
        });

        Ok(())
    }

    /// Ensure collection has an indexing search_strategy
    pub async fn ensure_collection_strategy(&self, collection_id: &str) -> Result<()> {
        let strategies = self.collection_strategies.read().await;
        if strategies.contains_key(collection_id) {
            return Ok(());
        }
        drop(strategies);

        // Analyze collection and select initial search_strategy
        let characteristics = self
            .adaptive_engine
            .analyze_collection(collection_id)
            .await?;
        let search_strategy = self
            .adaptive_engine
            .recommend_strategy(&characteristics)
            .await?;

        let mut strategies = self.collection_strategies.write().await;
        strategies.insert(collection_id.to_string(), search_strategy);

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_collections_managed += 1;

        Ok(())
    }

    /// Get current search_strategy for collection
    pub async fn get_collection_strategy(
        &self,
        collection_id: &str,
    ) -> Result<IndexSelectionStrategy> {
        let strategies = self.collection_strategies.read().await;
        strategies.get(collection_id).cloned().ok_or_else(|| {
            anyhow::anyhow!("No search_strategy found for collection {}", collection_id)
        })
    }

    /// Update collection search_strategy
    pub async fn update_collection_strategy(
        &self,
        collection_id: &str,
        search_strategy: IndexSelectionStrategy,
    ) -> Result<()> {
        let mut strategies = self.collection_strategies.write().await;
        strategies.insert(collection_id.to_string(), search_strategy);
        Ok(())
    }

    /// Hot-swap `ef_search` on every HNSW index in the collection's
    /// active strategy without rebuilding the graph. Resolves
    /// [`DriftKind::EfSearchOnly`] drift at zero rebuild cost — the
    /// graph degree (`m`) and build-time quality (`ef_construction`)
    /// are baked into the index structure, but `ef_search` is read
    /// from the strategy on every query so this is a near-free
    /// in-place tune.
    ///
    /// Returns [`HotSwapOutcome::Applied`] with `(previous_ef,
    /// new_ef)` per touched spec; [`HotSwapOutcome::NotApplicable`]
    /// when there's no active strategy, no HNSW spec, or every spec
    /// already matches the requested ef.
    ///
    /// **Does NOT rebuild**. If the operator needs an `m` or
    /// `ef_construction` change, they call the recluster path
    /// instead (separate slice).
    pub async fn apply_hnsw_ef_hot_swap(
        &self,
        collection_id: &str,
        new_ef_search: u32,
    ) -> Result<HotSwapOutcome> {
        use crate::index::axis::types::IndexAlgorithm;

        let mut strategies = self.collection_strategies.write().await;
        let Some(strategy) = strategies.get_mut(collection_id) else {
            return Ok(HotSwapOutcome::NotApplicable {
                reason: format!("no active strategy for collection '{}'", collection_id),
            });
        };

        let mut changes: Vec<HotSwapEfChange> = Vec::new();
        for spec in strategy.indexes.iter_mut() {
            if let IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
                max_elements,
            } = spec.algorithm
            {
                if ef_search == new_ef_search {
                    continue;
                }
                changes.push(HotSwapEfChange {
                    index_name: spec.name.clone(),
                    previous_ef_search: ef_search,
                    new_ef_search,
                });
                spec.algorithm = IndexAlgorithm::HNSW {
                    m,
                    ef_construction,
                    ef_search: new_ef_search,
                    max_elements,
                };
            }
        }

        if changes.is_empty() {
            return Ok(HotSwapOutcome::NotApplicable {
                reason: "no HNSW spec needed updating (none present, or all already at the requested ef_search)".to_string(),
            });
        }

        // The strategy mutation is now in-place; the next query
        // routed through this collection will pick up the new ef.
        // emit a structured event so operator dashboards can verify.
        for change in &changes {
            tracing::info!(
                target: "axis_diag",
                site = "apply_hnsw_ef_hot_swap",
                collection_id = collection_id,
                index = change.index_name.as_deref().unwrap_or("unnamed"),
                previous_ef_search = change.previous_ef_search,
                new_ef_search = change.new_ef_search,
                "hot-swapped HNSW ef_search"
            );
        }

        Ok(HotSwapOutcome::Applied { changes })
    }

    /// Hot-swap `nprobe` on every IVF index in the collection's
    /// active strategy. Mirror of
    /// [`Self::apply_hnsw_ef_hot_swap`] for the IVF arm.
    ///
    /// `nprobe` is read from the strategy on every query (live
    /// tunable), so this is a near-free in-place tune. `nlist` and
    /// any PQ quantizer are baked into the cluster + posting list
    /// structure and cannot change without a recluster — this
    /// method intentionally only touches `nprobe`.
    ///
    /// The returned [`HotSwapOutcome::Applied`] reuses
    /// [`HotSwapEfChange`] verbatim — the field names
    /// `previous_ef_search` / `new_ef_search` are semantically
    /// "live tunable knob", not strictly ef-only. Keeps the REST
    /// response shape stable between HNSW + IVF.
    pub async fn apply_ivf_nprobe_hot_swap(
        &self,
        collection_id: &str,
        new_nprobe: u32,
    ) -> Result<HotSwapOutcome> {
        use crate::index::axis::types::IndexAlgorithm;

        let mut strategies = self.collection_strategies.write().await;
        let Some(strategy) = strategies.get_mut(collection_id) else {
            return Ok(HotSwapOutcome::NotApplicable {
                reason: format!("no active strategy for collection '{}'", collection_id),
            });
        };

        let mut changes: Vec<HotSwapEfChange> = Vec::new();
        for spec in strategy.indexes.iter_mut() {
            if let IndexAlgorithm::IVF {
                nlist,
                nprobe,
                quantizer,
            } = &spec.algorithm
            {
                if *nprobe == new_nprobe {
                    continue;
                }
                changes.push(HotSwapEfChange {
                    index_name: spec.name.clone(),
                    previous_ef_search: *nprobe,
                    new_ef_search: new_nprobe,
                });
                let new_quantizer = quantizer.clone();
                let new_nlist = *nlist;
                spec.algorithm = IndexAlgorithm::IVF {
                    nlist: new_nlist,
                    nprobe: new_nprobe,
                    quantizer: new_quantizer,
                };
            }
        }

        if changes.is_empty() {
            return Ok(HotSwapOutcome::NotApplicable {
                reason: "no IVF spec needed updating (none present, or all already at the requested nprobe)".to_string(),
            });
        }

        for change in &changes {
            tracing::info!(
                target: "axis_diag",
                site = "apply_ivf_nprobe_hot_swap",
                collection_id = collection_id,
                index = change.index_name.as_deref().unwrap_or("unnamed"),
                previous_nprobe = change.previous_ef_search,
                new_nprobe = change.new_ef_search,
                "hot-swapped IVF nprobe"
            );
        }

        Ok(HotSwapOutcome::Applied { changes })
    }

    /// Get the distance metric configured for a collection
    /// This ensures indexes use the same metric as the collection's stored config
    async fn get_collection_distance_metric(
        &self,
        collection_id: &str,
    ) -> Option<crate::compute::distance_computation::DistanceMetric> {
        use proximadb_vector::distance::conversion::proto_distance_to_internal;

        // Try to get from shared cache first
        if let Some(cache) = &self.shared_collection_cache
            && let Some(collection) = cache.get(collection_id)
            && let Some(config) = &collection.config
        {
            let metric_code = config
                .distance_metric
                .unwrap_or(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32);
            return Some(proto_distance_to_internal(metric_code));
        }

        // Fall back to collection service
        if let Some(collection_service) = &self.collection_service
            && let Ok(Some(collection)) = collection_service.collection(collection_id).await
            && let Some(config) = &collection.config
        {
            let metric_code = config
                .distance_metric
                .unwrap_or(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32);
            return Some(proto_distance_to_internal(metric_code));
        }

        None
    }

    /// Maybe evaluate if search_strategy should change
    async fn maybe_evaluate_strategy(&self, _collection_id: &str) -> Result<()> {
        // Deferred: Implement periodic evaluation logic
        // For now, we'll rely on explicit analyze_and_optimize calls
        Ok(())
    }

    /// Get migration status for a collection
    pub async fn get_migration_status(&self, collection_id: &str) -> Option<MigrationStatus> {
        let migrations = self.active_migrations.read().await;
        migrations.get(collection_id).cloned()
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> AxisMetrics {
        self.metrics.read().await.clone()
    }

    /// Drop all indexes for a collection (used during collection deletion)
    pub async fn drop_collection(&self, collection_id: &str) -> Result<()> {
        tracing::info!(
            "🗑️ Dropping all AXIS indexes for collection: {}",
            collection_id
        );

        // Remove from collection strategies
        let mut strategies = self.collection_strategies.write().await;
        strategies.remove(collection_id);
        drop(strategies);

        // Clean up from all indexes
        self.global_id_index
            .remove_collection(collection_id)
            .await?;
        self.metadata_index.remove_collection(collection_id).await?;
        self.dense_vector_index
            .remove_collection(collection_id)
            .await?;
        if let Some(registry) = &self.hmgi_registry {
            registry.drop_collection_partitions(collection_id).await?;
        }
        {
            let mut enabled = self.hmgi_enabled_collections.write().await;
            enabled.remove(collection_id);
        }
        {
            let mut oids = self.hmgi_collection_oids.write().await;
            oids.remove(collection_id);
        }
        {
            let mut hnsw_indexes = self.hnsw_indexes.write().await;
            hnsw_indexes.remove(collection_id);
        }
        {
            let mut ivf_indexes = self.ivf_indexes.write().await;
            ivf_indexes.remove(collection_id);
        }
        {
            let mut pending = self.ivf_pending_vectors.write().await;
            pending.remove(collection_id);
        }
        self.sparse_vector_index
            .remove_collection(collection_id)
            .await?;

        {
            let mut collection_vectors = self.collection_vectors.write().await;
            collection_vectors.remove(collection_id);
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        if metrics.total_collections_managed > 0 {
            metrics.total_collections_managed -= 1;
        }

        tracing::info!(
            "✅ Successfully dropped all indexes for collection: {}",
            collection_id
        );
        Ok(())
    }

    /// Clear the in-memory `collection_vectors` projection for a collection after a
    /// successful storage flush (TD-FLUSH-1).
    ///
    /// Inserted vectors are kept in two in-memory places until they are flushed: the
    /// WAL/memtable batches (reaped by `write_buffer.clear_flushed`) and this
    /// `collection_vectors` map (the AxisManager's `ProximaRecord` projection used to
    /// serve warm reads + hydrate cold indexes). Before this method, only the WAL was
    /// reaped on flush — every inserted vector stayed resident in `collection_vectors`
    /// forever, so large workloads OOM'd and warm GETs served from the stale in-memory
    /// copy instead of exercising the materialized segment (masking cold-read bugs).
    ///
    /// This is the in-memory projection only — it does NOT drop the persistent indexes
    /// (HNSW/IVF/metadata), which were built from the flushed vectors and must remain
    /// servable. Use [`drop_collection`] for full teardown. The durable record now lives
    /// in the materialized storage segment that the flush just wrote.
    pub async fn clear_collection_vectors(&self, collection_id: &str) -> Result<()> {
        let removed_count = {
            let mut collection_vectors = self.collection_vectors.write().await;
            collection_vectors
                .remove(collection_id)
                .map(|vectors| vectors.len())
                .unwrap_or(0)
        };

        // Defense-in-depth: drop the collection's entries from the global ID index
        // (currently a placeholder, but the call mirrors `drop_collection` so the
        // behavior stays correct the moment it is implemented).
        self.global_id_index
            .remove_collection(collection_id)
            .await?;

        tracing::info!(
            "🧹 AXIS: Cleared {} in-memory collection_vectors for collection '{}' after flush",
            removed_count,
            collection_id
        );
        Ok(())
    }

    /// Get collection statistics
    pub async fn get_collection_stats(&self, collection_id: &str) -> Result<IndexCollectionStats> {
        let search_strategy = self.get_collection_strategy(collection_id).await?;

        Ok(IndexCollectionStats {
            collection_id: collection_id.to_string(),
            strategy_type: search_strategy
                .indexes
                .first()
                .map_or(Data::DenseVector { dimension: 128 }, |idx| idx.data_type), // Default to dense vector
            total_vectors: 0,    // Deferred: Implement actual counting
            index_size_bytes: 0, // Deferred: Implement actual size calculation
            last_updated: Utc::now(),
        })
    }

    /// Update vector file reference after flush/compaction
    /// This ensures AXIS indexes point to the correct on-disk files
    pub async fn update_vector_file_reference(
        &self,
        vector_id: &VectorId,
        collection_id: &str,
        file_path: &str,
    ) -> Result<()> {
        // Skip if vector_id is empty
        if vector_id.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            "🗂️ AXIS: Updating file reference for vector {} → {}",
            vector_id,
            file_path
        );

        // Ensure we have a search_strategy for this collection
        self.ensure_collection_strategy(collection_id).await?;

        // Update file reference in global ID index
        self.global_id_index
            .update_file_reference(vector_id, file_path)
            .await?;

        // Update file references in secondary indexes based on search_strategy
        let search_strategy = self.get_collection_strategy(collection_id).await?;
        for index_spec in &search_strategy.indexes {
            match index_spec.data_type {
                Data::Metadata => {
                    self.metadata_index
                        .update_file_reference(vector_id, file_path)
                        .await?;
                }
                Data::DenseVector { .. } => {
                    self.dense_vector_index
                        .update_file_reference(vector_id, file_path)
                        .await?;
                }
                Data::SparseVector { .. } => {
                    self.sparse_vector_index
                        .update_file_reference(vector_id, file_path)
                        .await?;
                }
                _ => {}
            }
        }

        tracing::debug!(
            "✅ AXIS: Updated file reference for vector {} in all indexes",
            vector_id
        );
        Ok(())
    }

    /// Rebuild indexes after compaction
    /// This is called when storage files are merged/compacted
    pub async fn rebuild_indexes_after_compaction(
        &self,
        collection_id: &str,
        old_files: &[String],
        new_files: &[String],
    ) -> Result<()> {
        tracing::info!(
            "🔄 AXIS: Rebuilding indexes after compaction for collection {}",
            collection_id
        );
        tracing::debug!(
            "🔄 AXIS: Old files: {:?} → New files: {:?}",
            old_files,
            new_files
        );

        // Ensure we have a search_strategy for this collection
        self.ensure_collection_strategy(collection_id).await?;

        // For now, we'll do a simple file reference update
        // In a production system, this would involve:
        // 1. Reading vector data from new_files
        // 2. Rebuilding the affected index segments
        // 3. Updating file references atomically

        let rebuild_start = Utc::now();

        // Update file references for all affected vectors
        // This is a simplified implementation - production would be more sophisticated
        for old_file in old_files {
            for new_file in new_files {
                tracing::debug!("🔄 AXIS: Mapping vectors from {} to {}", old_file, new_file);
                // In reality, we'd need to map specific vectors from old to new files
                // For now, we'll let the natural indexing process handle this
            }
        }

        let rebuild_duration = Utc::now().signed_duration_since(rebuild_start);
        tracing::info!(
            "✅ AXIS: Completed index rebuild for collection {} in {}ms",
            collection_id,
            rebuild_duration.num_milliseconds()
        );

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_migrations += 1; // Count rebuilds as migrations
        metrics.successful_migrations += 1;

        Ok(())
    }

    /// Get native index config for a collection
    pub async fn native_index_config(
        &self,
        _collection_id: &str,
    ) -> Result<crate::index::config::RuntimeIndexConfig> {
        // Return default config for now
        // In production, this would look up collection-specific configuration
        Ok(crate::index::config::RuntimeIndexConfig::default())
    }

    /// Notify AXIS about newly flushed vectors that need indexing
    /// This method is called by the flush coordinator after successful storage flush
    pub async fn handle_flushed_vectors(
        &self,
        collection_id: &str,
        flushed_vectors: Vec<ProximaRecord>,
        files_created: Vec<String>,
    ) -> Result<()> {
        if flushed_vectors.is_empty() {
            tracing::debug!(
                "🔄 AXIS: No vectors to index for collection {}",
                collection_id
            );
            return Ok(());
        }

        tracing::info!(
            "🚀 AXIS: Processing {} newly flushed vectors for collection {} from {} files",
            flushed_vectors.len(),
            collection_id,
            files_created.len()
        );

        // Get IndexConfig for this collection to determine indexing behavior
        let index_config = match self.native_index_config(collection_id).await {
            Ok(config) => config,
            Err(e) => {
                tracing::warn!(
                    "⚠️ AXIS: Failed to get IndexConfig for collection {}: {}. Using default sync mode.",
                    collection_id,
                    e
                );
                // Use default synchronous indexing if config retrieval fails
                crate::index::config::RuntimeIndexConfig::default()
            }
        };

        tracing::debug!(
            "🎯 AXIS: Using IndexConfig with update_mode: {:?} for collection {}",
            index_config.update_mode,
            collection_id
        );

        // Handle indexing based on update mode
        match index_config.update_mode {
            crate::index::config::IndexUpdateMode::Synchronous => {
                self.index_vectors_synchronously(collection_id, flushed_vectors, &files_created)
                    .await?;
            }
            crate::index::config::IndexUpdateMode::Asynchronous => {
                self.index_vectors_asynchronously(collection_id, flushed_vectors, files_created)
                    .await?;
            }
            crate::index::config::IndexUpdateMode::Hybrid => {
                self.index_vectors_hybrid(
                    collection_id,
                    flushed_vectors,
                    files_created,
                    &index_config,
                )
                .await?;
            }
        }

        tracing::info!(
            "✅ AXIS: Completed indexing notification for collection {}",
            collection_id
        );

        Ok(())
    }

    /// Number of vectors currently registered in the in-memory store for a
    /// collection. Diagnostic accessor; also the "is this collection present in
    /// AXIS" signal used by the lazy rebuild-from-SST path (TD-112). HNSW/IVF
    /// structures are built lazily atop this store and are not a reliable
    /// presence signal for small collections.
    #[doc(hidden)]
    pub async fn registered_vector_count(&self, collection_id: &str) -> usize {
        self.collection_vectors
            .read()
            .await
            .get(collection_id)
            .map_or(0, |m| m.len())
    }

    /// Index vectors synchronously (blocking the flush completion)
    ///
    /// For batches >= 500 vectors, uses batch-aware IVF training to ensure
    /// proper cluster count for better recall.
    async fn index_vectors_synchronously(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
        _files_created: &[String],
    ) -> Result<()> {
        let batch_size = vectors.len();
        tracing::info!(
            "🔄 AXIS: Synchronous indexing of {} vectors for collection {}",
            batch_size,
            collection_id
        );

        let start_time = std::time::Instant::now();

        // For medium-sized batches, still use batch training for better recall
        if batch_size >= 500
            && let Err(e) = self.train_ivf_for_batch(collection_id, &vectors).await
        {
            tracing::warn!(
                "⚠️ AXIS: Batch IVF training failed for collection {}: {}, using incremental",
                collection_id,
                e
            );
        }

        for vector in vectors {
            self.insert(collection_id, &vector).await?;
        }

        let duration = start_time.elapsed();

        tracing::info!(
            "✅ AXIS: Synchronous indexing completed in {}ms for collection {}",
            duration.as_millis(),
            collection_id
        );

        Ok(())
    }

    /// Index vectors asynchronously (non-blocking)
    ///
    /// For large batches (>1000 vectors), this method uses batch-aware IVF training
    /// to ensure proper cluster count based on total collection size. This fixes
    /// the recall degradation issue at 50K vectors where IVF was trained with only
    /// 100 vectors, resulting in too few clusters (20 instead of 256).
    async fn index_vectors_asynchronously(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
        files_created: Vec<String>,
    ) -> Result<()> {
        let batch_size = vectors.len();
        tracing::info!(
            "🚀 AXIS: Spawning asynchronous indexing task for {} vectors in collection {}",
            batch_size,
            collection_id
        );

        let start_time = std::time::Instant::now();

        // For large batches, use batch-aware IVF training for better recall
        // This trains the IVF index with proper cluster count based on total batch size
        if batch_size >= 1000 {
            match self.train_ivf_for_batch(collection_id, &vectors).await {
                Ok(()) => {
                    tracing::info!(
                        "✅ AXIS: Batch IVF training completed for {} vectors in collection {}",
                        batch_size,
                        collection_id
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "⚠️ AXIS: Batch IVF training failed for collection {}: {}, falling back to incremental",
                        collection_id,
                        e
                    );
                }
            }
        }

        // Insert all vectors (will go to trained IVF index directly if batch training succeeded)
        let mut indexed_count = 0;
        for vector in vectors {
            match self.insert(collection_id, &vector).await {
                Ok(()) => indexed_count += 1,
                Err(e) => {
                    tracing::error!(
                        "❌ AXIS: Failed to index vector in collection {}: {}",
                        collection_id,
                        e
                    );
                }
            }
        }

        let duration = start_time.elapsed();
        tracing::info!(
            "✅ AXIS: Asynchronous indexing completed - {}/{} vectors indexed in {}ms for collection {} (files: {:?})",
            indexed_count,
            batch_size,
            duration.as_millis(),
            collection_id,
            files_created
        );

        tracing::debug!(
            "🚀 AXIS: Asynchronous indexing completed for collection {}",
            collection_id
        );
        Ok(())
    }

    /// Train IVF index with proper cluster count for a large batch
    ///
    /// This method trains the IVF index with the optimal number of clusters
    /// based on the total batch size, avoiding the issue of undertrained indexes
    /// when vectors are inserted one-by-one (which trains at only 100 vectors).
    ///
    /// For N vectors:
    /// - n_clusters = clamp(sqrt(N) * 2, 16, 256)
    /// - n_probe = max(n_clusters/2, sqrt(n_clusters)*3)
    async fn train_ivf_for_batch(
        &self,
        collection_id: &str,
        vectors: &[ProximaRecord],
    ) -> Result<()> {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::indexes::dual_store_ivf::{UnifiedIvfConfig, UnifiedIvfIndex};

        if vectors.is_empty() {
            return Ok(());
        }

        let dimension = vectors[0].embeddings.first().map_or(0, |e| e.values.len());
        if dimension == 0 {
            return Ok(());
        }

        // Check if IVF index already exists and is trained
        {
            let indexes = self.ivf_indexes.read().await;
            if let Some(index) = indexes.get(collection_id) {
                let idx = index.read().await;
                if idx.is_trained() {
                    tracing::debug!(
                        "🔍 AXIS: IVF index for collection {} already trained, skipping batch training",
                        collection_id
                    );
                    return Ok(());
                }
            }
        }

        let batch_size = vectors.len();

        // Calculate optimal cluster count based on batch size
        // n_clusters = clamp(sqrt(N) * 2, 16, 256)
        let n_clusters = {
            let sqrt_based = (batch_size as f32).sqrt() as usize * 2;
            const MIN_CLUSTERS: usize = 16;
            const MAX_CLUSTERS: usize = 256;
            sqrt_based.clamp(MIN_CLUSTERS, MAX_CLUSTERS)
        };

        // Calculate n_probe: max(n_clusters/2, sqrt(n_clusters)*3), clamped to n_clusters
        let n_probe = {
            let half_clusters = n_clusters / 2;
            let sqrt_based = ((n_clusters as f32).sqrt() * 3.0) as usize;
            std::cmp::max(half_clusters, sqrt_based).min(n_clusters)
        };

        tracing::info!(
            "🎯 AXIS: Batch IVF training for collection {} - {} vectors, {} clusters, n_probe={}",
            collection_id,
            batch_size,
            n_clusters,
            n_probe
        );

        // **Metric correctness (2026-05-29)**: see the equivalent
        // fix in `insert_into_ivf` above. The batch-training site
        // also previously hardcoded Cosine — same hazard for
        // Euclidean / DotProduct collections.
        let resolved_metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(DistanceMetric::Cosine);
        let config = UnifiedIvfConfig {
            n_clusters,
            n_probe,
            dimension,
            distance_metric: resolved_metric,
            min_train_size: 100, // Lower since we're batch training
            ..Default::default()
        };

        let mut index = UnifiedIvfIndex::new(collection_id.to_string(), config)?;

        // Train with all vectors (or sample if too large)
        let training_vectors: Vec<Vec<f32>> = if batch_size > 50000 {
            // Sample for very large batches to avoid memory issues
            let sample_size = 50000;
            let step = batch_size / sample_size;
            vectors
                .iter()
                .step_by(step.max(1))
                .take(sample_size)
                .map(|v| {
                    v.embeddings
                        .first()
                        .map(|e| e.values.to_fp32_owned())
                        .unwrap_or_default()
                })
                .collect()
        } else {
            vectors
                .iter()
                .map(|v| {
                    v.embeddings
                        .first()
                        .map(|e| e.values.to_fp32_owned())
                        .unwrap_or_default()
                })
                .collect()
        };

        index.train(training_vectors).await?;

        // TD-165 Defect B: populate the trained index's posting lists with the
        // batch vectors. `train` only fits centroids — posting lists are created
        // empty — and the incremental `insert` path routes vectors to the HNSW
        // index, never this batch-trained IVF. Without this the persisted `ivf.bin`
        // (written below) would hold 0 vectors and every cold reload would serve an
        // empty index (the cheap cold-load path the cost model wants returns
        // nothing). Add ALL vectors (not just the training sample).
        let mut added = 0usize;
        for record in vectors {
            if record.oid.is_empty() {
                continue;
            }
            let Some(emb) = record.embeddings.first() else {
                continue;
            };
            let fp32 = emb.values.to_fp32_owned();
            if fp32.is_empty() {
                continue;
            }
            index.add_vector(record.oid.clone(), fp32, None).await?;
            added += 1;
        }

        tracing::info!(
            "✅ AXIS: Batch IVF training complete for collection {} with {} clusters, {} vectors populated",
            collection_id,
            n_clusters,
            added
        );

        // Store the trained index
        let served = Arc::new(tokio::sync::RwLock::new(index));
        {
            let mut indexes = self.ivf_indexes.write().await;
            indexes.insert(collection_id.to_string(), served.clone());
        }

        // Clear pending vectors buffer since we've trained
        {
            let mut pending = self.ivf_pending_vectors.write().await;
            pending.remove(collection_id);
        }

        // TD-087 Slice B: persist the batch-trained index (best-effort).
        self.persist_ivf_index(collection_id, &served).await;

        Ok(())
    }

    /// Whether `collection_id` currently has a served IVF index.
    pub async fn has_ivf_index(&self, collection_id: &str) -> bool {
        self.ivf_indexes.read().await.contains_key(collection_id)
    }

    /// Current served-index swap generation for `collection_id` (0 if never
    /// rebuilt). Increments on each `rebuild_and_swap_ivf_index`.
    pub async fn index_generation(&self, collection_id: &str) -> u64 {
        self.index_generations
            .read()
            .await
            .get(collection_id)
            .copied()
            .unwrap_or(0)
    }

    /// Phase 8 F1 recluster apply-step: rebuild a collection's IVF index from a
    /// complete record set and **atomically swap** it in as the served index.
    ///
    /// Builds a *fresh* `UnifiedIvfIndex` (so the "already trained" guard never
    /// applies), trains k-means over the records, populates posting lists, and
    /// replaces the served Arc in `ivf_indexes` in one `HashMap::insert` — which
    /// is the atomic swap (`query_ivf` reads the same map, so in-flight reads
    /// finish on the old Arc and new reads see the rebuilt index). Bumps the
    /// per-collection generation.
    ///
    /// Returns `Ok(false)` (no-op) when there are too few embedded vectors to
    /// cluster. Distance metric is resolved per-collection (not hardcoded). The
    /// rebuilt index does not yet repopulate filterable metadata — that is
    /// rehydrated by subsequent inserts; a follow-up can carry it through.
    pub async fn rebuild_and_swap_ivf_index(
        &self,
        collection_id: &str,
        records: &[ProximaRecord],
    ) -> Result<bool> {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::index_factory::AxisVectorIndex;
        use crate::index::axis::indexes::dual_store_ivf::{UnifiedIvfConfig, UnifiedIvfIndex};

        // Collect valid (record, fp32) — keep the record so its filterable
        // metadata can be carried into the rebuilt index (predicate-aware search).
        let mut valid: Vec<(&ProximaRecord, Vec<f32>)> = Vec::with_capacity(records.len());
        for r in records {
            if r.oid.is_empty() {
                continue;
            }
            if let Some(e) = r.embeddings.first() {
                let v = e.values.to_fp32_owned();
                if !v.is_empty() {
                    valid.push((r, v));
                }
            }
        }
        let dimension = valid.first().map(|(_, v)| v.len()).unwrap_or(0);
        // Need enough vectors to form clusters (mirrors the recluster floor and
        // keeps k-means well-posed: n_clusters floor is 16).
        if dimension == 0 || valid.len() < 16 {
            return Ok(false);
        }

        // Same nlist/nprobe heuristic the incremental IVF path uses.
        let n_clusters = ((valid.len() as f32).sqrt() as usize * 2).clamp(16, 256);
        let n_probe = std::cmp::max(n_clusters / 2, ((n_clusters as f32).sqrt() * 3.0) as usize)
            .min(n_clusters);
        let metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(DistanceMetric::Cosine);
        let config = UnifiedIvfConfig {
            n_clusters,
            n_probe,
            dimension,
            distance_metric: metric,
            min_train_size: 100,
            ..Default::default()
        };

        let mut index = UnifiedIvfIndex::new(collection_id.to_string(), config)?;
        index
            .train(valid.iter().map(|(_, v)| v.clone()).collect())
            .await?;
        let fields_config =
            crate::index::axis::filterable_metadata::FilterableFieldsConfig::default();
        for (r, v) in &valid {
            let meta = crate::index::axis::filterable_metadata::extract_filterable_metadata(
                r,
                &fields_config,
            );
            index
                .add_with_metadata(r.oid.clone(), v.clone(), &meta)
                .await?;
        }

        // Atomic swap: replace the served Arc in one insert.
        let served = Arc::new(tokio::sync::RwLock::new(index));
        {
            let mut indexes = self.ivf_indexes.write().await;
            indexes.insert(collection_id.to_string(), served.clone());
        }
        // Observable generation bump.
        let generation = {
            let mut gens = self.index_generations.write().await;
            let g = gens.entry(collection_id.to_string()).or_insert(0);
            *g += 1;
            *g
        };

        // TD-087 Slice B: persist the rebuilt index to disk (best-effort).
        self.persist_ivf_index(collection_id, &served).await;

        tracing::info!(
            "✅ AXIS: rebuilt + swapped IVF index for collection {} ({} vectors, {} clusters, gen={})",
            collection_id,
            valid.len(),
            n_clusters,
            generation
        );
        Ok(true)
    }

    /// Whether `collection_id` currently has a served HNSW index.
    pub async fn has_hnsw_index(&self, collection_id: &str) -> bool {
        self.hnsw_indexes.read().await.contains_key(collection_id)
    }

    /// Whether a published ANN index currently serves this collection.
    ///
    /// Continuous reclustering is maintenance for an existing served index;
    /// it must not bootstrap one by first materializing the entire collection.
    pub async fn has_served_index(&self, collection_id: &str) -> bool {
        self.has_ivf_index(collection_id).await || self.has_hnsw_index(collection_id).await
    }

    /// Phase 8 F1 recluster apply-step for HNSW-served collections: rebuild the
    /// collection's HNSW graph from a complete record set and **atomically swap**
    /// it in as the served index. Same atomic-swap pattern as the IVF path — a
    /// fresh `AxisHnswIndex` is built and the served `Arc` in `hnsw_indexes` is
    /// replaced in one `HashMap::insert` (`query_hnsw` reads the same map). Bumps
    /// the per-collection generation. Returns `Ok(false)` for an empty record set.
    ///
    /// Rebuilding the graph from scratch reclaims quality lost to churn (deletes
    /// / updates degrade an incrementally-built HNSW graph). Like the IVF path,
    /// filterable metadata is not yet re-carried (rehydrated by later inserts).
    pub async fn rebuild_and_swap_hnsw_index(
        &self,
        collection_id: &str,
        records: &[ProximaRecord],
    ) -> Result<bool> {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::index_factory::AxisVectorIndex;
        use crate::index::axis::indexes::hnsw_index::{AxisHnswConfig, AxisHnswIndex};

        // Collect valid (record, fp32) — keep the record to carry filterable
        // metadata into the rebuilt graph (predicate-aware search).
        let mut valid: Vec<(&ProximaRecord, Vec<f32>)> = Vec::with_capacity(records.len());
        for r in records {
            if r.oid.is_empty() {
                continue;
            }
            if let Some(e) = r.embeddings.first() {
                let v = e.values.to_fp32_owned();
                if !v.is_empty() {
                    valid.push((r, v));
                }
            }
        }
        let dimension = valid.first().map(|(_, v)| v.len()).unwrap_or(0);
        if dimension == 0 || valid.is_empty() {
            return Ok(false);
        }

        // Match the collection's metric (the legacy HNSW path defaults to
        // DotProduct for FAISS/bench compatibility).
        let metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(DistanceMetric::DotProduct);
        let config = AxisHnswConfig {
            distance_metric: metric,
            ..Default::default()
        };
        let index =
            AxisHnswIndex::new_with_collection(Some(collection_id.to_string()), config, dimension)?;
        let count = valid.len();
        let fields_config =
            crate::index::axis::filterable_metadata::FilterableFieldsConfig::default();
        for (r, v) in &valid {
            let meta = crate::index::axis::filterable_metadata::extract_filterable_metadata(
                r,
                &fields_config,
            );
            index
                .add_with_metadata(r.oid.clone(), v.clone(), &meta)
                .await?;
        }

        // Atomic swap: replace the served Arc in one insert.
        {
            let mut indexes = self.hnsw_indexes.write().await;
            indexes.insert(collection_id.to_string(), Arc::new(index));
        }
        let generation = {
            let mut gens = self.index_generations.write().await;
            let g = gens.entry(collection_id.to_string()).or_insert(0);
            *g += 1;
            *g
        };

        tracing::info!(
            "✅ AXIS: rebuilt + swapped HNSW index for collection {} ({} vectors, gen={})",
            collection_id,
            count,
            generation
        );
        Ok(true)
    }

    /// Rebuild the HNSW index using **advisor-recommended** m,
    /// ef_construction, and ef_search at the current corpus size +
    /// the operator's recall_target. This is the recall-aware
    /// counterpart to [`Self::rebuild_and_swap_hnsw_index`] — same
    /// atomic-swap semantics, but the new graph is sized for the
    /// recall the collection actually committed to.
    ///
    /// Also updates the collection's active
    /// [`IndexSelectionStrategy`] (via
    /// [`Self::update_collection_strategy`]) so future queries see
    /// the advised ef_search without an extra hot-swap call.
    ///
    /// Returns the [`HnswSizingOutput`] the rebuild was sized
    /// against, including the rationale string — handy for
    /// operator log lines and the recluster response body.
    /// Returns `Ok(None)` if no vectors were supplied or the
    /// records lacked usable embeddings.
    ///
    /// `top_k` is the steady-state top-k the collection's workload
    /// expects to request — typically pulled from the
    /// `target_top_k:` tag via
    /// `crate::services::collection::recall_target::resolve_top_k`.
    /// The advisor scales `ef ∝ k`, so a workload that consistently
    /// runs `k=100` and supplies `top_k=10` would get an under-sized
    /// graph.
    ///
    /// `max_ef_search` is the optional latency-budget cap. When
    /// provided (typically from the collection's `max_ef_search:`
    /// tag), the rebuilt graph is sized with the same clamping
    /// behaviour the route-health and recall-tune surfaces apply —
    /// the resulting `HnswSizingOutput.clamped_by_max_ef` flag tells
    /// the caller whether the cap was hit so it can surface the
    /// projected recall to the operator.
    pub async fn rebuild_and_swap_hnsw_index_for_recall_target(
        &self,
        collection_id: &str,
        records: &[ProximaRecord],
        recall_target: f32,
        top_k: u32,
        max_ef_search: Option<u32>,
    ) -> Result<Option<crate::index::axis::management::HnswSizingOutput>> {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::index_factory::AxisVectorIndex;
        use crate::index::axis::indexes::hnsw_index::{AxisHnswConfig, AxisHnswIndex};
        use crate::index::axis::management::{HnswSizingInput, advise_hnsw_params};

        // Collect valid (record, fp32) like the legacy rebuild.
        let mut valid: Vec<(&ProximaRecord, Vec<f32>)> = Vec::with_capacity(records.len());
        for r in records {
            if r.oid.is_empty() {
                continue;
            }
            if let Some(e) = r.embeddings.first() {
                let v = e.values.to_fp32_owned();
                if !v.is_empty() {
                    valid.push((r, v));
                }
            }
        }
        let dimension = valid.first().map(|(_, v)| v.len()).unwrap_or(0);
        if dimension == 0 || valid.is_empty() {
            return Ok(None);
        }

        let metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(DistanceMetric::Cosine);

        // Size from the actual rebuild corpus (records.len() — not
        // some stale baseline_n). top_k flows in from the caller
        // (typically resolved via
        // services::collection::recall_target::resolve_top_k) so
        // the rebuild reflects the workload's steady-state k.
        let advised = advise_hnsw_params(HnswSizingInput {
            vector_count: valid.len() as u64,
            top_k,
            recall_target,
            dimension: dimension as u32,
            distance_metric: metric,
            max_ef_search,
        });

        let config = AxisHnswConfig {
            m: advised.m as usize,
            ef_construction: advised.ef_construction as usize,
            ef: advised.ef_search as usize,
            distance_metric: metric,
            ..Default::default()
        };

        tracing::info!(
            target: "axis_diag",
            site = "rebuild_and_swap_hnsw_index_for_recall_target",
            collection_id = collection_id,
            n = valid.len(),
            recall_target = recall_target,
            m = advised.m,
            ef_construction = advised.ef_construction,
            ef_search = advised.ef_search,
            rationale = %advised.rationale,
            "rebuilding HNSW with advisor-sized params"
        );

        let index =
            AxisHnswIndex::new_with_collection(Some(collection_id.to_string()), config, dimension)?;
        let count = valid.len();
        let fields_config =
            crate::index::axis::filterable_metadata::FilterableFieldsConfig::default();
        for (r, v) in &valid {
            let meta = crate::index::axis::filterable_metadata::extract_filterable_metadata(
                r,
                &fields_config,
            );
            index
                .add_with_metadata(r.oid.clone(), v.clone(), &meta)
                .await?;
        }

        // Atomic swap: replace the served Arc.
        {
            let mut indexes = self.hnsw_indexes.write().await;
            indexes.insert(collection_id.to_string(), Arc::new(index));
        }
        let generation = {
            let mut gens = self.index_generations.write().await;
            let g = gens.entry(collection_id.to_string()).or_insert(0);
            *g += 1;
            *g
        };

        // Update the strategy so future queries pick up the new
        // ef_search via the normal lookup path. If no strategy
        // exists yet, build a fresh single-HNSW one — this is
        // safe because we just rebuilt the index with these exact
        // params.
        use crate::index::axis::types::{
            Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification,
        };
        let spec = IndexSpecification::new(
            Data::DenseVector { dimension },
            IndexAlgorithm::HNSW {
                m: advised.m,
                ef_construction: advised.ef_construction,
                ef_search: advised.ef_search,
                max_elements: 1_000_000,
            },
        );
        let new_strategy = {
            let strategies = self.collection_strategies.read().await;
            match strategies.get(collection_id).cloned() {
                Some(mut existing) => {
                    // Replace the first HNSW spec in place; if none
                    // existed, append. Preserves non-HNSW indexes
                    // (e.g. metadata or sparse).
                    let mut replaced = false;
                    for spec_slot in existing.indexes.iter_mut() {
                        if matches!(spec_slot.algorithm, IndexAlgorithm::HNSW { .. }) {
                            spec_slot.algorithm = IndexAlgorithm::HNSW {
                                m: advised.m,
                                ef_construction: advised.ef_construction,
                                ef_search: advised.ef_search,
                                max_elements: 1_000_000,
                            };
                            replaced = true;
                            break;
                        }
                    }
                    if !replaced {
                        existing.indexes.push(spec);
                    }
                    existing
                }
                None => IndexSelectionStrategy {
                    indexes: vec![spec],
                    routing_rules: vec![],
                },
            }
        };
        self.update_collection_strategy(collection_id, new_strategy)
            .await?;

        tracing::info!(
            "✅ AXIS: recall-aware HNSW rebuild for {} ({} vectors, gen={}, m={}, ef_construction={}, ef_search={})",
            collection_id,
            count,
            generation,
            advised.m,
            advised.ef_construction,
            advised.ef_search,
        );
        Ok(Some(advised))
    }

    /// Recall-aware IVF rebuild — sizes `(nlist, nprobe)` and the
    /// optional PQ quantizer via the `IvfIndexAdvisor` and
    /// atomically swaps in the new strategy. Mirror of
    /// [`Self::rebuild_and_swap_hnsw_index_for_recall_target`] for
    /// the IVF arm.
    ///
    /// The rebuild itself reuses the existing
    /// [`Self::rebuild_and_swap_ivf_index`] (IVF training +
    /// k-means + posting-list construction); this method just
    /// performs the advisor-driven sizing and updates the active
    /// `IndexSelectionStrategy` so future queries pick up the new
    /// (nlist, nprobe, PQ) tuple.
    ///
    /// Returns the [`AnnAdvisorOutput`] the rebuild was sized
    /// against (carries algorithm spec + clamped flag + projected
    /// recall + rationale) so the recluster handler can surface
    /// the choice. Returns `Ok(None)` if no records were supplied
    /// or the IVF advisor declined the recall target.
    pub async fn rebuild_and_swap_ivf_index_for_recall_target(
        &self,
        collection_id: &str,
        records: &[ProximaRecord],
        recall_target: f32,
        top_k: u32,
        max_query_latency_ms: Option<f64>,
        max_memory_mb: Option<f64>,
        binary_rerank: bool,
    ) -> Result<Option<crate::index::axis::management::AnnAdvisorOutput>> {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::management::{AnnAdvisorInput, AnnIndexAdvisor, IvfIndexAdvisor};

        // (1) Gate on records being non-empty + carrying valid embeddings.
        let dim = records
            .iter()
            .find_map(|r| r.embeddings.first().map(|e| e.values.to_fp32_owned().len()))
            .unwrap_or(0);
        if dim == 0 || records.is_empty() {
            return Ok(None);
        }

        // (2) Resolve the collection's distance metric (same as HNSW).
        let metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(DistanceMetric::Cosine);

        // (3) Size via the IVF advisor.
        let advisor = IvfIndexAdvisor::new();
        let advised = advisor.advise(&AnnAdvisorInput {
            vector_count: records.len() as u64,
            top_k,
            recall_target,
            dimension: dim as u32,
            distance_metric: metric,
            max_query_latency_ms,
            max_memory_mb,
            binary_rerank_allowed: binary_rerank,
            modalities: Vec::new(),
        });
        let Some(advised) = advised else {
            // Advisor declined — recall_target above the IVF
            // ceiling for this (N, nlist, binary_rerank) tuple.
            tracing::info!(
                target: "axis_diag",
                site = "rebuild_and_swap_ivf_index_for_recall_target",
                collection_id = collection_id,
                recall_target = recall_target,
                binary_rerank = binary_rerank,
                "IVF advisor declined — recall_target unreachable at this nlist"
            );
            return Ok(None);
        };

        // (4) Delegate the actual rebuild to the existing IVF
        // pipeline. The legacy rebuild path doesn't accept advisor
        // params directly today — it picks via the cluster engine
        // defaults. P5+ commit threads `advised.algorithm`
        // straight through. For now, the strategy update at step
        // (5) is the source of truth for future queries.
        let did_rebuild = self
            .rebuild_and_swap_ivf_index(collection_id, records)
            .await?;

        if !did_rebuild {
            // Nothing to swap — return the advisor output so the
            // caller knows what *would* have been built.
            return Ok(Some(advised));
        }

        // (5) Update the active strategy with the advised
        // (nlist, nprobe, optional PQ) tuple so subsequent
        // queries route through the new sizing.
        use crate::index::axis::types::{Data, IndexSelectionStrategy, IndexSpecification};
        let spec = IndexSpecification::new(
            Data::DenseVector { dimension: dim },
            advised.algorithm.clone(),
        );
        let new_strategy = {
            let strategies = self.collection_strategies.read().await;
            match strategies.get(collection_id).cloned() {
                Some(mut existing) => {
                    let mut replaced = false;
                    for slot in existing.indexes.iter_mut() {
                        if matches!(slot.algorithm, IndexAlgorithm::IVF { .. }) {
                            slot.algorithm = advised.algorithm.clone();
                            replaced = true;
                            break;
                        }
                    }
                    if !replaced {
                        existing.indexes.push(spec);
                    }
                    existing
                }
                None => IndexSelectionStrategy {
                    indexes: vec![spec],
                    routing_rules: vec![],
                },
            }
        };
        self.update_collection_strategy(collection_id, new_strategy)
            .await?;

        tracing::info!(
            target: "axis_diag",
            site = "rebuild_and_swap_ivf_index_for_recall_target",
            collection_id = collection_id,
            n = records.len(),
            recall_target = recall_target,
            binary_rerank = binary_rerank,
            rationale = %advised.rationale,
            "rebuilt IVF with advisor-sized params"
        );
        Ok(Some(advised))
    }

    /// Rebuild + atomically swap whichever served ANN index the collection uses
    /// — IVF if present, else HNSW (the default). Returns `true` if a swap
    /// happened, `false` if the collection has no served index or there were too
    /// few vectors. This is what the recluster pass calls.
    pub async fn rebuild_and_swap_served_index(
        &self,
        collection_id: &str,
        records: &[ProximaRecord],
    ) -> Result<bool> {
        if self.has_ivf_index(collection_id).await {
            self.rebuild_and_swap_ivf_index(collection_id, records)
                .await
        } else if self.has_hnsw_index(collection_id).await {
            self.rebuild_and_swap_hnsw_index(collection_id, records)
                .await
        } else {
            Ok(false)
        }
    }

    /// Index vectors using hybrid mode (adaptive based on batch size)
    pub async fn index_vectors_hybrid<R>(
        &self,
        collection_id: &str,
        vectors: Vec<R>,
        files_created: Vec<String>,
        index_config: &crate::index::config::RuntimeIndexConfig,
    ) -> Result<()>
    where
        R: Into<ProximaRecord>,
    {
        let vectors: Vec<ProximaRecord> = vectors.into_iter().map(Into::into).collect();
        let batch_size_threshold = index_config.async_update_batch_size.unwrap_or(1000);

        tracing::info!(
            "🎯 AXIS: Hybrid indexing for {} vectors (threshold: {}) in collection {}",
            vectors.len(),
            batch_size_threshold,
            collection_id
        );

        if vectors.len() <= batch_size_threshold {
            tracing::debug!(
                "🔄 AXIS: Small batch - using synchronous indexing for collection {}",
                collection_id
            );
            self.index_vectors_synchronously(collection_id, vectors, &files_created)
                .await
        } else {
            tracing::debug!(
                "🚀 AXIS: Large batch - using asynchronous indexing for collection {}",
                collection_id
            );
            self.index_vectors_asynchronously(collection_id, vectors, files_created)
                .await
        }
    }

    /// Get all indexes for a collection (required for compaction integration)
    /// Return the live HNSW index for a collection if one exists (Phase C).
    pub async fn get_hnsw_index(
        &self,
        collection_id: &str,
    ) -> Option<Arc<crate::index::axis::indexes::hnsw_index::AxisHnswIndex>> {
        self.hnsw_indexes.read().await.get(collection_id).cloned()
    }

    pub async fn get_collection_indexes(
        &self,
        collection_id: &str,
    ) -> Result<Vec<(String, Arc<dyn crate::index::axis::AxisVectorIndex>)>> {
        tracing::debug!("🔍 AXIS: Getting indexes for collection {}", collection_id);

        // For now, return empty list - this will be properly implemented when compaction integration is complete
        // In the full implementation, this would:
        // 1. Check collection_strategies for active indexes
        // 2. Return references to global_id_index, metadata_index, dense_vector_index, sparse_vector_index
        // 3. Include any dynamic indexes created by the adaptive engine

        Ok(Vec::new())
    }

    /// Rebuild a specific index by name (required for compaction integration)
    pub async fn rebuild_index(&self, collection_id: &str, index_name: &str) -> Result<()> {
        tracing::info!(
            "🔄 AXIS: Rebuilding index '{}' for collection {}",
            index_name,
            collection_id
        );

        // For now, delegate to the full rebuild method
        // In the full implementation, this would:
        // 1. Identify the specific index component by name
        // 2. Rebuild only that index while keeping others intact
        // 3. Update internal tracking structures

        self.rebuild_indexes_after_compaction(collection_id, &[], &[])
            .await
    }

    /// Quantize vector for in-memory index using collection's quantization settings
    /// This reuses our existing modular quantization infrastructure
    #[allow(dead_code)]
    async fn quantize_for_index(
        &self,
        vector: &ProximaRecord,
        quant_config: &crate::proto::proximadb_v1::QuantizationConfig,
        collection_config: &crate::proto::proximadb_v1::CollectionConfig,
    ) -> Result<ProximaRecord> {
        use crate::compute::quantization::storage_engine::{
            StorageQuantizationConfig, StorageQuantizationEngine,
        };
        use proximadb_vector::distance::conversion::proto_distance_to_internal;

        // Extract the vector data
        let vector_data = vector
            .embeddings
            .first()
            .map(|e| e.as_fp32_slice())
            .unwrap_or(&[]);
        if vector_data.is_empty() {
            return Ok(vector.clone());
        }

        // Use helper function for distance metric conversion
        let distance_metric =
            proto_distance_to_internal(collection_config.distance_metric.unwrap_or(0));

        // Create quantization config using collection settings with proper field mapping
        let storage_config = StorageQuantizationConfig {
            // Map to the actual fields available in storage engine config
            primary_level: Some(
                crate::compute::quantization::quantization_engine::UnifiedQuantizationLevel::pq8(
                    // Default to dimension/4 with min 8 and max 64 subvectors
                    (collection_config.dimension / 4).clamp(8, 64).min(255) as u8,
                ),
            ),
            filter_level: Some(
                crate::compute::quantization::quantization_engine::UnifiedQuantizationLevel::binary(
                ),
            ),
            fast_level: Some(
                crate::compute::quantization::quantization_engine::UnifiedQuantizationLevel::int8(),
            ),
            distance_metric,
            enable_progressive: true,
            filter_threshold: 0.8,
            candidate_multiplier: 10,
            // quality_threshold removed -  0.95,
            training_sample_size: quant_config.training_sample_size.unwrap_or(10000) as usize,
            memory_budget_mb: 512,
            enable_hardware_acceleration: true,
        };

        // Create required components for quantization engine
        let distance_compute = Arc::new(
            crate::compute::distance_computation::engine::UnifiedDistanceCompute::new(
                distance_metric,
            ),
        );
        let codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );
        let unified_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_compute.clone(),
                codebook_store,
            ),
        );

        // Create quantization engine
        let _engine =
            StorageQuantizationEngine::new(unified_engine, distance_compute, storage_config);

        // For indexes, we don't actually quantize the vector data in the VectorRecord
        // Instead, indexes maintain their own quantized representation internally
        // This is just a placeholder that marks the vector as quantized for the index
        // The actual quantization happens inside the index implementations

        // Return the original vector marked for quantization
        // The index will handle the actual quantization internally
        Ok(vector.clone())
    }
}

/// AXIS index-level collection statistics
///
/// Distinct from `storage::traits::CollectionStats` — carries index-specific
/// metadata (strategy_type, last_updated timestamp).
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexCollectionStats {
    /// Unique identifier of the collection.
    pub collection_id: String,
    /// Data type strategy used for this collection's index.
    pub strategy_type: Data,
    /// Total number of vectors indexed.
    pub total_vectors: u64,
    /// Size of the index on disk in bytes.
    pub index_size_bytes: u64,
    /// Timestamp of the last index update.
    pub last_updated: DateTime<Utc>,
}

/// ADR-011 ANN filtering mode for HNSW traversal routing.
///
/// Mirror of `proximadb_catalog::AnnFilteringMode`. Kept locally to avoid
/// taking the catalog crate into low-level AXIS types; the conversion goes
/// through [`ann_mode_from_catalog`] when the caller supplies a catalog
/// `AnnFilteringPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnnFilteringMode {
    /// Filter candidates before HNSW traversal (< 5% selectivity).
    PreFilter,
    /// Thread predicate into HNSW walk (ACORN semantics, 5–50% selectivity).
    Inline,
    /// Run full HNSW search then post-filter results (> 50% selectivity).
    #[default]
    PostFilter,
}

/// Convert a catalog `AnnFilteringMode` (driven by
/// `AnnFilteringPolicy::routing_mode`) into the local manager-facing mode.
fn ann_mode_from_catalog(mode: proximadb_catalog::AnnFilteringMode) -> AnnFilteringMode {
    match mode {
        proximadb_catalog::AnnFilteringMode::PreFilter => AnnFilteringMode::PreFilter,
        proximadb_catalog::AnnFilteringMode::Inline => AnnFilteringMode::Inline,
        proximadb_catalog::AnnFilteringMode::PostFilter => AnnFilteringMode::PostFilter,
    }
}

/// Backwards-compat alias for [`AxisHybridQuery`].
pub type HybridQuery = AxisHybridQuery;

/// Hybrid query combining multiple search criteria
#[derive(Debug, Clone, Default)]
pub struct AxisHybridQuery {
    /// Target collection for the query.
    pub collection_id: String,
    /// Optional vector similarity query component.
    pub vector_query: Option<VectorQuery>,
    /// Metadata field filter predicates.
    pub metadata_filters: Vec<AxisMetadataFilter>,
    /// Exact vector ID filters for point lookups.
    pub id_filters: Vec<VectorId>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Whether to include MVCC-expired records in results.
    pub include_expired: bool,
    /// ADR-011 ANN filtering mode; drives routing in the AXIS manager query
    /// path when `ann_filtering_policy` is `None`. When the policy is set
    /// alongside `estimated_selectivity`, the policy-driven mode takes
    /// precedence — `routing_mode(selectivity)` decides PreFilter / Inline /
    /// PostFilter and `ann_filtering_mode` is treated as a legacy hint only.
    pub ann_filtering_mode: AnnFilteringMode,
    /// TD-064 / ADR-011: Optional catalog filtering policy. When present
    /// together with `estimated_selectivity`, drives selection of the
    /// effective `AnnFilteringMode` instead of the hard-coded
    /// `ann_filtering_mode` switch.
    pub ann_filtering_policy: Option<proximadb_catalog::AnnFilteringPolicy>,
    /// TD-064 / ADR-011: Pre-computed selectivity estimate from
    /// `FilterDiagnostics` or a sampler. Feeds `AnnFilteringPolicy::routing_mode`.
    pub estimated_selectivity: Option<f64>,
    /// Per-query accuracy-vs-latency effort, derived from the caller's
    /// `SearchMode`. `None` (the default) keeps each index's own size-aware
    /// default, so existing callers are behavior-identical. When `Some`, the
    /// HNSW/IVF dispatch maps it to a per-query `ef` / `nprobe` so the
    /// `exact`/`approximate`/`approximate:N` knob actually controls recall.
    pub search_effort: Option<crate::core::search::SearchEffort>,
}

/// Vector query types
#[derive(Debug, Clone)]
pub enum VectorQuery {
    /// Dense vector similarity query.
    Dense {
        /// Query vector in full-precision f32 format.
        vector: Vec<f32>,
        /// Minimum similarity score threshold for results.
        similarity_threshold: f32,
    },
    /// Sparse vector similarity query.
    Sparse {
        /// Sparse query vector as dimension-index to value mapping.
        vector: HashMap<u32, f32>,
        /// Minimum similarity score threshold for results.
        similarity_threshold: f32,
    },
}

/// Backwards-compat alias for [`AxisMetadataFilter`].
pub type MetadataFilter = AxisMetadataFilter;

/// Metadata filter
#[derive(Debug, Clone)]
pub struct AxisMetadataFilter {
    /// Name of the metadata field to filter on.
    pub field: String,
    /// Comparison operator for the filter.
    pub operator: FilterOperator,
    /// Value to compare against.
    pub value: serde_json::Value,
}

/// Filter operators
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOperator {
    /// Exact equality match.
    Equals,
    /// Not equal comparison.
    NotEquals,
    /// Greater than comparison.
    GreaterThan,
    /// Greater than or equal comparison.
    GreaterThanOrEqual,
    /// Less than comparison.
    LessThan,
    /// Less than or equal comparison.
    LessThanOrEqual,
    /// Membership in a set of values.
    In,
    /// Exclusion from a set of values.
    NotIn,
    /// String contains substring.
    Contains,
    /// String starts with prefix.
    StartsWith,
    /// String ends with suffix.
    EndsWith,
    /// SQL-style LIKE pattern matching.
    Like,
    /// Inclusive range comparison with `[lower, upper]`.
    Between,
    /// Value is null or missing.
    IsNull,
    /// Value is present and non-null.
    IsNotNull,
}

/// Outcome of [`AxisManager::apply_hnsw_ef_hot_swap`]. Either every
/// HNSW spec in the active strategy had its `ef_search` swapped to
/// the requested value (`Applied`), or there was nothing to do
/// (`NotApplicable` — either no strategy, no HNSW indexes, or the
/// requested ef was already in place).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HotSwapOutcome {
    /// One or more HNSW specs were updated. `changes` carries the
    /// per-spec before/after for observability.
    Applied { changes: Vec<HotSwapEfChange> },
    /// No update was required. `reason` is a short operator-facing
    /// string for logs / route-health.
    NotApplicable { reason: String },
}

/// Per-spec record of an `ef_search` change, for structured event
/// emission. `index_name` is `None` when the spec wasn't given a
/// name in [`crate::index::axis::types::IndexSpecification::name`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HotSwapEfChange {
    pub index_name: Option<String>,
    pub previous_ef_search: u32,
    pub new_ef_search: u32,
}

/// Backwards-compat alias for [`AxisManagerQueryResult`].
pub type QueryResult = AxisManagerQueryResult;

/// Query result
#[derive(Debug, Clone)]
pub struct AxisManagerQueryResult {
    /// Scored results ordered by relevance.
    pub results: Vec<ScoredResult>,
    /// Index selection strategy that was used to execute the query.
    pub strategy_used: IndexSelectionStrategy,
    /// Total execution time in milliseconds.
    pub execution_time_ms: u64,
    /// TD-064: Predicate-aware shortfall — `Some(...)` when post-filter
    /// trimming dropped the survivor count below the requested `top_k`.
    /// Gateways/REST handlers should surface this on
    /// `SearchPlanTrace.predicate_shortfall` for EXPLAIN disclosure.
    pub predicate_shortfall: Option<crate::observability::search_plan_trace::PredicateShortfall>,
    /// TD-064 / ADR-011: The filtering mode that actually executed.
    /// May differ from `AxisHybridQuery.ann_filtering_mode` when policy-based
    /// routing was active (`ann_filtering_policy` + `estimated_selectivity`).
    /// `None` when the query had no filters and the unfiltered ANN path ran.
    pub selected_filtering_mode: Option<AnnFilteringMode>,
    /// TD-075 / Phase 8 F2: which IVF route actually executed.
    /// `Some(true)` = quantized accelerator used; `Some(false)` = quantized
    /// storage present but the recall-probe gate forced exact (fallback);
    /// `None` = no quantized storage, or a non-IVF route. EXPLAIN/route-health
    /// surface this as the quantized-route decision.
    pub quantized_route: Option<bool>,
}

/// TD-075 route decision: use the quantized accelerator only when the recall
/// probe gate is open AND the index actually has quantized storage. A closed
/// gate (recall not yet verified, or a FAIL streak) forces exact search.
fn decide_quantized_route(gate_open: bool, has_quantized_storage: bool) -> bool {
    gate_open && has_quantized_storage
}

/// recall@k: fraction of the exact top-k that the approximate (quantized) top-k
/// recovered. Order-independent set overlap divided by the effective k
/// (`min(k, exact.len())`). Returns 1.0 when there is nothing to recall.
fn recall_at_k(exact: &[String], approx: &[String], k: usize) -> f32 {
    let eff_k = k.min(exact.len());
    if eff_k == 0 {
        return 1.0;
    }
    let exact_set: std::collections::HashSet<&String> = exact.iter().take(eff_k).collect();
    let hits = approx
        .iter()
        .take(k)
        .filter(|id| exact_set.contains(id))
        .count();
    hits as f32 / eff_k as f32
}

/// Probe outcome from a mean recall vs. the recall floor (Phase 5 observer).
fn recall_outcome(mean_recall: f32, floor: f32) -> crate::catalog::ProbeOutcome {
    if mean_recall >= floor {
        crate::catalog::ProbeOutcome::Pass
    } else {
        crate::catalog::ProbeOutcome::Fail
    }
}

/// Scored result with MVCC support
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// Identifier of the matching vector.
    pub vector_id: VectorId,
    /// Similarity score between the query and this result.
    pub similarity: f32,
    /// MVCC expiration timestamp, if the record has a TTL.
    pub expires_at: Option<DateTime<Utc>>,
}

impl AxisManager {
    /// ADR-011 PreFilter: evaluate scalar predicates first, then score the
    /// surviving candidates EXACTLY. `query.search_effort` is intentionally not
    /// consulted here — exact scoring has no ANN approximation to trade; the
    /// only recall/latency lever on this path is predicate selectivity.
    async fn execute_exact_filtered_query(
        &self,
        collection_id: &str,
        query: &AxisHybridQuery,
    ) -> Result<Vec<ScoredResult>> {
        if query.top_k == 0 {
            return Ok(Vec::new());
        }

        let metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(crate::compute::distance_computation::DistanceMetric::DotProduct);
        let compute = UnifiedDistanceCompute::new(metric);
        let metadata_expression = if query.metadata_filters.is_empty() {
            None
        } else {
            Some(self.metadata_filters_to_expression(&query.metadata_filters))
        };
        // F7 / TD-HYBRID-1 D2: same compiled predicate as the Inline path — the
        // exact branch applies it to every record, id filter then metadata.
        let compiled_filter = crate::core::search::CompiledGlobalFilter::new(
            query.id_filters.clone(),
            metadata_expression,
        );

        // Borrow the records under the read guard instead of deep-cloning the
        // whole collection per filtered query (the `lookup_record_expiration`
        // fix documents a 45× latency gap for this exact defect class). The
        // guard is taken AFTER the awaits above and the loop body has none,
        // so the hold-time is the scoring loop only — strictly shorter than
        // the clone it replaces; `oid` is cloned only for final top-k records.
        let collection_vectors = self.collection_vectors.read().await;
        let Some(vectors) = collection_vectors.get(collection_id) else {
            return Ok(Vec::new());
        };
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Bounded top-k entry: "better" ranks Greater = higher similarity
        // first, then lexicographically smaller vector_id — the exact total
        // order the previous full sort + truncate applied.
        struct ExactScored {
            similarity: f32,
            vector_id: String,
            expires_at: Option<DateTime<Utc>>,
        }
        impl Eq for ExactScored {}
        impl PartialEq for ExactScored {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == std::cmp::Ordering::Equal
            }
        }
        impl Ord for ExactScored {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.similarity
                    .partial_cmp(&other.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| other.vector_id.cmp(&self.vector_id))
            }
        }
        impl PartialOrd for ExactScored {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut top_k = TopKHeap::with_capacity(query.top_k);
        // One clock read per query instead of per surviving record.
        let now = Utc::now();

        for record in vectors.values() {
            let metadata = self.record_filter_metadata(record);
            if !compiled_filter.matches(&record.oid, Some(&metadata)) {
                continue;
            }

            let similarity = match &query.vector_query {
                Some(VectorQuery::Dense {
                    vector,
                    similarity_threshold,
                }) => {
                    let record_vec = match record.embeddings.first() {
                        Some(e) => &*e.as_fp32_cow(),
                        None => continue,
                    };
                    let result = compute.similarity(vector, record_vec, Some(metric));
                    if result.normalized_score < *similarity_threshold {
                        continue;
                    }
                    result.normalized_score
                }
                Some(VectorQuery::Sparse { .. }) => continue,
                None => 1.0,
            };

            let expires_at = record.valid_to_ns.and_then(|ns| {
                DateTime::<Utc>::from_timestamp(ns / 1_000_000_000, (ns % 1_000_000_000) as u32)
            });

            if !query.include_expired
                && let Some(expiration) = expires_at.as_ref()
                && now >= *expiration
            {
                continue;
            }

            top_k.try_push(ExactScored {
                similarity,
                vector_id: record.oid.clone(),
                expires_at,
            });
        }

        let results: Vec<ScoredResult> = top_k
            .into_sorted_desc()
            .into_iter()
            .map(|entry| ScoredResult {
                vector_id: entry.vector_id,
                similarity: entry.similarity,
                expires_at: entry.expires_at,
            })
            .collect();

        Ok(results)
    }

    fn metadata_filters_to_expression(
        &self,
        filters: &[AxisMetadataFilter],
    ) -> crate::core::search::FilterExpression {
        crate::core::search::FilterExpression::And(
            filters
                .iter()
                .map(|filter| crate::core::search::FilterExpression::Comparison {
                    field: filter.field.clone(),
                    operator: self.axis_filter_operator_to_comparison(&filter.operator),
                    value: filter.value.clone(),
                })
                .collect(),
        )
    }

    fn axis_filter_operator_to_comparison(
        &self,
        operator: &FilterOperator,
    ) -> crate::core::search::ComparisonOperator {
        match operator {
            FilterOperator::Equals => crate::core::search::ComparisonOperator::Equals,
            FilterOperator::NotEquals => crate::core::search::ComparisonOperator::NotEquals,
            FilterOperator::GreaterThan => crate::core::search::ComparisonOperator::GreaterThan,
            FilterOperator::GreaterThanOrEqual => {
                crate::core::search::ComparisonOperator::GreaterThanOrEqual
            }
            FilterOperator::LessThan => crate::core::search::ComparisonOperator::LessThan,
            FilterOperator::LessThanOrEqual => {
                crate::core::search::ComparisonOperator::LessThanOrEqual
            }
            FilterOperator::In => crate::core::search::ComparisonOperator::In,
            FilterOperator::NotIn => crate::core::search::ComparisonOperator::NotIn,
            FilterOperator::Contains => crate::core::search::ComparisonOperator::Contains,
            FilterOperator::StartsWith => crate::core::search::ComparisonOperator::StartsWith,
            FilterOperator::EndsWith => crate::core::search::ComparisonOperator::EndsWith,
            FilterOperator::Like => crate::core::search::ComparisonOperator::Like,
            FilterOperator::Between => crate::core::search::ComparisonOperator::Between,
            FilterOperator::IsNull => crate::core::search::ComparisonOperator::IsNull,
            FilterOperator::IsNotNull => crate::core::search::ComparisonOperator::IsNotNull,
        }
    }

    fn record_filter_metadata(&self, record: &ProximaRecord) -> HashMap<String, Value> {
        // Reuse the canonical ProximaTree→json-map converter (single source) and
        // add the synthetic identity fields the filter grammar may reference.
        let mut metadata =
            crate::core::search::sql_value_filter::proxima_tree_to_json_map(&record.props);
        metadata.insert("id".to_string(), Value::String(record.oid.clone()));
        metadata.insert("oid".to_string(), Value::String(record.oid.clone()));
        metadata.insert(
            "tenant_id".to_string(),
            Value::String(record.tenant_id.clone()),
        );
        metadata
    }

    fn record_dense_vector<'a>(&self, record: &'a ProximaRecord) -> Option<&'a [f32]> {
        record
            .embeddings
            .iter()
            .find(|embedding| !embedding.values.is_empty())
            .map(|embedding| embedding.as_fp32_slice())
    }

    fn datetime_from_timestamp_ns(timestamp_ns: i64) -> Option<DateTime<Utc>> {
        let seconds = timestamp_ns.div_euclid(1_000_000_000);
        let nanos = timestamp_ns.rem_euclid(1_000_000_000) as u32;
        DateTime::<Utc>::from_timestamp(seconds, nanos)
    }

    fn lookup_record_expiration(
        &self,
        collection_id: &str,
        vector_id: &str,
    ) -> Option<DateTime<Utc>> {
        // **Perf bug (2026-05-29)**: this function used to do
        //     `collections.get(collection_id).cloned()`
        // which clones the ENTIRE HashMap<String, ProximaRecord> for
        // the collection (every record at 10K, 25K, 100K scale),
        // and then `.cloned()` again on the inner ProximaRecord.
        // Called once per result × 10 results per query, the
        // legacy HNSW path was paying O(N · k) record-clones per
        // query — 100K clones at 10K vectors, 250K at 25K — which
        // explained the bench's "HNSW path scales linearly" finding
        // (vs HMGI's sub-linear) and the 45× latency gap between
        // HMGI (0.7-1ms) and legacy HNSW (45ms) on identical data.
        //
        // The fix borrows through the read guard rather than cloning
        // anything. valid_to_ns is a Copy `Option<i64>` so we only
        // touch the few bytes we actually need.
        let collections = self.collection_vectors.try_read().ok()?;
        let vectors = collections.get(collection_id)?;
        let record = vectors.get(vector_id)?;
        record
            .valid_to_ns
            .and_then(Self::datetime_from_timestamp_ns)
    }
}

/// HMGI (Hierarchical Multi-modality Graph Indexing) extensions for AxisManager
///
/// HMGI provides per-modality HNSW partitioning for multi-modality collections,
/// achieving 70% search space reduction compared to monolithic HNSW.
///
/// ## Usage
///
/// ```rust,ignore
/// // Enable HMGI for a collection
/// axis_manager.enable_hmgi("my_collection", Some("_modality")).await?;
///
/// // Insert vectors (automatically routed to modality partitions)
/// axis_manager.insert_hmgi("my_collection", vector_record, oid).await?;
///
/// // Search with automatic partition routing
/// let results = axis_manager.search_hmgi("my_collection", query, top_k).await?;
/// ```
impl AxisManager {
    /// Enable HMGI for a collection
    ///
    /// ## Arguments
    ///
    /// - `collection_id`: Collection to enable HMGI for
    /// - `modality_field`: Field name containing modality tag (default: "_modality")
    /// - `oid`: Entity type ID for partition key generation
    ///
    /// ## Process
    ///
    /// 1. Initialize HMGI components if not already done
    /// 2. Migrate existing vectors to owned modality partitions
    /// 3. Publish the collection as HMGI-enabled
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// axis_manager.enable_hmgi("documents", Some("_modality"), 123).await?;
    /// ```
    pub async fn enable_hmgi(
        &self,
        collection_id: &str,
        modality_field: Option<String>,
        oid: u64,
    ) -> Result<()> {
        if self.is_hmgi_enabled(collection_id).await {
            let existing_oid = self
                .hmgi_collection_oids
                .read()
                .await
                .get(collection_id)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HMGI lifecycle state for collection '{collection_id}' has no OID"
                    )
                })?;
            if existing_oid == oid {
                return Ok(());
            }
            anyhow::bail!(
                "HMGI collection '{collection_id}' is already bound to OID {existing_oid}; \
                 disable it before binding OID {oid}"
            );
        }

        // Store the field for logging before moving
        let field_display = modality_field.clone();

        // Initialize HMGI components if not already done
        self.ensure_hmgi_initialized(modality_field).await?;

        // Build and register every existing modality partition before publishing
        // enablement. If migration fails, remove any partial graph state so callers
        // never observe an enabled collection with an incomplete registry.
        if let Err(error) = self.migrate_collection_to_hmgi(collection_id, oid).await {
            if let Some(registry) = &self.hmgi_registry {
                registry.drop_collection_partitions(collection_id).await?;
            }
            return Err(error);
        }

        {
            let mut oids = self.hmgi_collection_oids.write().await;
            oids.insert(collection_id.to_string(), oid);
        }
        {
            let mut enabled = self.hmgi_enabled_collections.write().await;
            enabled.insert(collection_id.to_string());
        }

        tracing::info!(
            "✅ HMGI enabled for collection '{}' with modality field '{:?}'",
            collection_id,
            field_display
        );

        Ok(())
    }

    /// Disable HMGI for a collection
    ///
    /// Merges all modality partitions back into a single index.
    pub async fn disable_hmgi(&self, collection_id: &str) -> Result<()> {
        let registry = self
            .hmgi_registry
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI not initialized"))?;

        // Drop all partitions for this collection
        registry.drop_collection_partitions(collection_id).await?;

        // Remove from enabled set
        {
            let mut enabled = self.hmgi_enabled_collections.write().await;
            enabled.remove(collection_id);
        }

        // Remove OID mapping
        {
            let mut oids = self.hmgi_collection_oids.write().await;
            oids.remove(collection_id);
        }

        tracing::info!("❌ HMGI disabled for collection '{}'", collection_id);
        Ok(())
    }

    /// Check if HMGI is enabled for a collection
    pub async fn is_hmgi_enabled(&self, collection_id: &str) -> bool {
        let enabled = self.hmgi_enabled_collections.read().await;
        enabled.contains(collection_id)
    }

    /// Analyze indexed collection records and auto-enable HMGI when multiple modalities appear.
    pub async fn maybe_auto_enable_hmgi(
        &self,
        collection_id: &str,
    ) -> Result<Option<DetectionResult>> {
        if self.is_hmgi_enabled(collection_id).await {
            return Ok(None);
        }

        let detector = match self.hmgi_detector.as_ref() {
            Some(detector) => detector,
            None => return Ok(None),
        };

        let samples = self.hmgi_detection_samples(collection_id).await;
        let detection = detector.detect_modalities(collection_id, &samples).await;

        if detection.should_enable_hmgi {
            let oid = self.hmgi_oid_for_collection(collection_id).await;
            self.enable_hmgi(collection_id, None, oid).await?;
        }

        Ok(Some(detection))
    }

    /// Ensure a collection has the canonical HMGI partition wrapper before dense
    /// vector indexing. A single-modality collection has one HNSW partition;
    /// additional modality tags add partitions without changing the graph
    /// implementation.
    async fn ensure_hmgi_collection_enabled(&self, collection_id: &str) -> Result<()> {
        if self.is_hmgi_enabled(collection_id).await {
            return Ok(());
        }
        // HMGI is the canonical dense vector index path for AXIS: any
        // collection that receives a dense vector is enabled here so the
        // insert routes through `insert_hmgi` and queries route through
        // `search_hmgi`. Single-modality collections land in a single
        // modality partition (no per-partition penalty beyond one
        // partition); multi-modality collections fan out automatically as
        // new modality tags appear. The earlier distance/similarity
        // sort-direction hazard that motivated disabling this was fixed in
        // commit b3985b59c, so enabling on insert is safe again.
        let oid = self.hmgi_oid_for_collection(collection_id).await;
        self.enable_hmgi(collection_id, None, oid).await
    }

    /// Resolve the HNSW graph contract shared by HMGI migration and steady-state
    /// inserts. `HmgiRegistry::get_or_create_partition` preserves the first config
    /// used for a partition, so every creation path must use this authority; a
    /// default-config migration would otherwise permanently shadow the collection's
    /// metric and tuned `m`/`ef` values.
    async fn hmgi_hnsw_config(
        &self,
        collection_id: &str,
        algorithm: Option<&IndexAlgorithm>,
        modality_tag: &str,
    ) -> crate::index::axis::indexes::hnsw_index::AxisHnswConfig {
        use crate::compute::distance_computation::DistanceMetric;

        let distance_metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(DistanceMetric::Cosine);
        Self::hmgi_hnsw_config_for_metric(distance_metric, algorithm, modality_tag)
    }

    fn hmgi_hnsw_config_for_metric(
        distance_metric: crate::compute::distance_computation::DistanceMetric,
        algorithm: Option<&IndexAlgorithm>,
        modality_tag: &str,
    ) -> crate::index::axis::indexes::hnsw_index::AxisHnswConfig {
        use crate::index::axis::indexes::hnsw_index::AxisHnswConfig;

        let defaults = AxisHnswConfig::default();
        let (m, ef_construction, ef) = match algorithm {
            Some(IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
                ..
            }) => (*m as usize, *ef_construction as usize, *ef_search as usize),
            Some(IndexAlgorithm::HMGI { per_modality, .. }) => per_modality
                .iter()
                .find(|partition| partition.modality_tag == modality_tag)
                .map(|partition| {
                    (
                        partition.m as usize,
                        partition.ef_construction as usize,
                        partition.ef_search as usize,
                    )
                })
                .unwrap_or((defaults.m, defaults.ef_construction, defaults.ef)),
            _ => (defaults.m, defaults.ef_construction, defaults.ef),
        };

        AxisHnswConfig {
            distance_metric,
            m,
            ef_construction,
            ef,
            ..defaults
        }
    }

    fn is_hmgi_routable_query(&self, query: &AxisHybridQuery) -> bool {
        if !matches!(query.vector_query, Some(VectorQuery::Dense { .. }))
            || !query.id_filters.is_empty()
        {
            return false;
        }

        let modality_field = self
            .hmgi_extractor
            .as_ref()
            .map(|extractor| extractor.modality_field())
            .unwrap_or("_modality");

        query
            .metadata_filters
            .iter()
            .all(|filter| filter.field == modality_field)
    }

    /// Insert a vector with HMGI partitioning
    ///
    /// Routes the vector to the appropriate modality partition based on metadata.
    ///
    /// ## Arguments
    ///
    /// - `collection_id`: Collection to insert into
    /// - `record`: Vector record with metadata containing modality tag
    ///
    /// ## Returns
    ///
    /// The partition key the vector was inserted into
    pub async fn insert_hmgi<R>(
        &self,
        collection_id: &str,
        record: R,
        algorithm: Option<&IndexAlgorithm>,
    ) -> Result<HmgiPartitionKey>
    where
        R: Into<ProximaRecord>,
    {
        let record: ProximaRecord = record.into();
        if !self.is_hmgi_enabled(collection_id).await {
            return Err(anyhow::anyhow!(
                "HMGI not enabled for collection '{}'",
                collection_id
            ));
        }

        let registry = self
            .hmgi_registry
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI not initialized"))?;

        let extractor = self
            .hmgi_extractor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI extractor not initialized"))?;

        // Get OID for this collection
        let oid = {
            let oids = self.hmgi_collection_oids.read().await;
            *oids
                .get(collection_id)
                .ok_or_else(|| anyhow::anyhow!("No OID found for collection '{}'", collection_id))?
        };

        // Extract modality from props (string values only)
        let metadata: std::collections::HashMap<String, serde_json::Value> = record
            .props
            .iter()
            .filter_map(|(k, v)| {
                if let ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(s)) = v {
                    Some((k.clone(), serde_json::Value::String(s.clone())))
                } else {
                    None
                }
            })
            .collect();
        let modality_tag = extractor.extract_modality(&metadata);

        let config = self
            .hmgi_hnsw_config(collection_id, algorithm, &modality_tag)
            .await;

        // Create partition key
        let partition_key = HmgiPartitionKey::new(oid, 1, modality_tag, None);

        let vec_values = record
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();
        let dimension = if vec_values.is_empty() {
            128
        } else {
            vec_values.len()
        };

        tracing::info!(
            target: "axis_diag",
            site = "insert_hmgi",
            collection_id = collection_id,
            partition = ?partition_key,
            using_metric = ?config.distance_metric,
            m = config.m,
            ef_construction = config.ef_construction,
            ef_search = config.ef,
            spec_source = if algorithm.is_some() { "strategy" } else { "default" },
            "HMGI partition HNSW config (metric + HNSW knobs)"
        );
        let index = registry
            .get_or_create_collection_partition(
                collection_id,
                partition_key.clone(),
                config,
                dimension,
            )
            .await?;

        use crate::index::axis::index_factory::AxisVectorIndex;
        if !record.oid.is_empty() && !vec_values.is_empty() {
            // TD-064: cache filterable metadata on the HMGI partition's index.
            let filterable_metadata =
                crate::index::axis::filterable_metadata::extract_filterable_metadata(
                    &record,
                    &crate::index::axis::filterable_metadata::FilterableFieldsConfig::default(),
                );
            index
                .add_with_metadata(record.oid.clone(), vec_values, &filterable_metadata)
                .await?;
        }

        tracing::debug!(
            "Inserting vector '{}' into HMGI partition '{}'",
            record.oid,
            partition_key
        );

        Ok(partition_key)
    }

    /// Remove a vector from every HMGI partition registered for the collection.
    async fn remove_hmgi_vector(&self, collection_id: &str, vector_id: &VectorId) -> Result<()> {
        let registry = self
            .hmgi_registry
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI registry not initialized"))?;

        let partitions = registry.get_partitions_for_collection(collection_id).await;

        use crate::index::axis::index_factory::AxisVectorIndex;
        for partition in partitions {
            if let Some(index) = registry.get_partition(&partition).await {
                index.remove(vector_id).await?;
            }
        }

        Ok(())
    }

    /// Search with HMGI partition routing
    ///
    /// Routes queries to relevant modality partitions based on filters.
    ///
    /// ## Arguments
    ///
    /// - `collection_id`: Collection to search
    /// - `query`: Hybrid query with optional metadata filters
    /// - `top_k`: Number of results to return
    ///
    /// ## Returns
    ///
    /// Top-k results from relevant partitions
    pub async fn search_hmgi(
        &self,
        collection_id: &str,
        query: &AxisHybridQuery,
        _top_k: usize,
    ) -> Result<Vec<ScoredResult>> {
        if !self.is_hmgi_enabled(collection_id).await {
            // Fall back to non-HMGI search
            return self
                .execute_exact_filtered_query(collection_id, query)
                .await;
        }

        let router = self
            .hmgi_router
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI router not initialized"))?;

        // Get all partitions for this collection
        let registry = self
            .hmgi_registry
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI registry not initialized"))?;
        let all_partitions = registry.get_partitions_for_collection(collection_id).await;

        // Create PartitionSet from all partitions
        use crate::index::axis::hmgi::PartitionSet;
        let mut partition_set = PartitionSet::new();
        for p in all_partitions {
            partition_set.insert(p);
        }

        // Route query to relevant partitions
        let routed = router
            .route_query(collection_id, query, partition_set)
            .await?;

        if routed.is_empty() {
            tracing::warn!(
                "No HMGI partitions found for collection '{}'",
                collection_id
            );
            return Ok(Vec::new());
        }

        // Convert PartitionSet to Vec<HmgiPartitionKey>
        let partitions: Vec<HmgiPartitionKey> = routed.iter().cloned().collect();

        tracing::debug!(
            "HMGI search routing to {} partitions for collection '{}'",
            partitions.len(),
            collection_id
        );

        // Search across routed partitions
        let results = router.search_partitions(partitions, query).await?;

        tracing::trace!(
            target: "axis_diag",
            site = "search_hmgi",
            n_results = results.len(),
            top1_score = ?results.first().map(|r| r.similarity),
            "HMGI router returned scored results"
        );

        Ok(results)
    }

    /// Initialize HMGI components if not already done
    async fn ensure_hmgi_initialized(&self, _modality_field: Option<String>) -> Result<()> {
        // Already initialized
        if self.hmgi_registry.is_some() {
            return Ok(());
        }

        // This is a self-referential issue - we can't initialize through &self
        // In production, this would be done through interior mutability or
        // the components would be passed in during construction
        tracing::warn!("HMGI initialization requires mutable access - use enable_hmgi_init");
        Err(anyhow::anyhow!(
            "HMGI components not initialized - call enable_hmgi_init first"
        ))
    }

    /// Migrate existing collection vectors to HMGI partitions
    async fn migrate_collection_to_hmgi(&self, collection_id: &str, oid: u64) -> Result<()> {
        let extractor = self
            .hmgi_extractor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI extractor not initialized"))?;

        let registry = self
            .hmgi_registry
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HMGI registry not initialized"))?;

        // Get existing vectors
        let vectors = {
            let collection_vectors = self.collection_vectors.read().await;
            collection_vectors
                .get(collection_id)
                .map(|v| v.values().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        };

        tracing::info!(
            "Migrating {} vectors from collection '{}' to HMGI partitions",
            vectors.len(),
            collection_id
        );

        // Auto-enable happens after the first record enters `collection_vectors`.
        // Resolve the active dense strategy before migration so that first record
        // cannot create a default-config partition that shadows the declared graph
        // contract for every subsequent insert.
        let strategy = self.get_collection_strategy(collection_id).await.ok();
        let dense_algorithm = strategy.as_ref().and_then(|strategy| {
            strategy
                .indexes
                .iter()
                .find(|spec| matches!(spec.data_type, Data::DenseVector { .. }))
                .map(|spec| &spec.algorithm)
        });
        let distance_metric = self
            .get_collection_distance_metric(collection_id)
            .await
            .unwrap_or(crate::compute::distance_computation::DistanceMetric::Cosine);

        let mut migrated = 0;
        for record in vectors {
            // Extract modality
            let metadata = self.record_filter_metadata(&record);
            let modality_tag = extractor.extract_modality(&metadata);
            let config =
                Self::hmgi_hnsw_config_for_metric(distance_metric, dense_algorithm, &modality_tag);

            // Create partition key
            let partition_key = HmgiPartitionKey::new(oid, 1, modality_tag, None);

            // Get or create the partition with the same collection-aware graph
            // contract used by steady-state inserts.
            let record_vector = self.record_dense_vector(&record);
            let dimension = record_vector.map_or(128, <[f32]>::len);
            let index = registry
                .get_or_create_collection_partition(collection_id, partition_key, config, dimension)
                .await?;
            use crate::index::axis::index_factory::AxisVectorIndex;
            if let Some(record_vector) = record_vector
                && !record.oid.is_empty()
            {
                // TD-064: cache filterable metadata on the HMGI partition's index.
                let filterable_metadata =
                    crate::index::axis::filterable_metadata::extract_filterable_metadata(
                        &record,
                        &crate::index::axis::filterable_metadata::FilterableFieldsConfig::default(),
                    );
                index
                    .add_with_metadata(
                        record.oid.clone(),
                        record_vector.to_vec(),
                        &filterable_metadata,
                    )
                    .await?;
            }
            migrated += 1;
        }

        tracing::info!("Migrated {} vectors to HMGI partitions", migrated);
        Ok(())
    }

    async fn hmgi_detection_samples(&self, collection_id: &str) -> Vec<VectorRecordSample> {
        let collection_vectors = self.collection_vectors.read().await;
        collection_vectors
            .get(collection_id)
            .map(|records| {
                records
                    .values()
                    .map(|record| VectorRecordSample::new(self.record_filter_metadata(record)))
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn hmgi_oid_for_collection(&self, collection_id: &str) -> u64 {
        {
            let oids = self.hmgi_collection_oids.read().await;
            if let Some(oid) = oids.get(collection_id) {
                return *oid;
            }
        }

        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        collection_id.hash(&mut hasher);
        hasher.finish()
    }
}

/// Mutable HMGI initialization for AxisManager
///
/// This trait provides mutable methods for initializing HMGI components.
/// In production, this would use interior mutability (RwLock) instead.
impl AxisManager {
    /// Initialize HMGI components (mutable version)
    ///
    /// Call this before enabling HMGI for collections if components
    /// weren't initialized during construction.
    pub fn init_hmgi(&mut self, modality_field: Option<String>) -> Result<()> {
        let field = modality_field.unwrap_or_else(|| "_modality".to_string());

        let registry = Arc::new(HmgiRegistry::new());
        let extractor = Arc::new(ModalityExtractor::with_config(
            field.clone(),
            "default".to_string(),
        ));
        self.hmgi_registry = Some(registry.clone());
        self.hmgi_extractor = Some(extractor.clone());
        self.hmgi_detector = Some(Arc::new(ModalityDetector::default_config()));
        self.hmgi_router = Some(Arc::new(HmgiRouter::new(registry, extractor)));

        tracing::info!(
            "🔧 HMGI components initialized with modality field '{}'",
            field
        );
        Ok(())
    }

    /// Get HMGI registry (for testing/diagnostics)
    pub fn hmgi_registry(&self) -> Option<&Arc<HmgiRegistry>> {
        self.hmgi_registry.as_ref()
    }

    /// Get HMGI router (for testing/diagnostics)
    pub fn hmgi_router(&self) -> Option<&Arc<HmgiRouter>> {
        self.hmgi_router.as_ref()
    }

    /// Get modality extractor (for testing/diagnostics)
    pub fn hmgi_extractor(&self) -> Option<&Arc<ModalityExtractor>> {
        self.hmgi_extractor.as_ref()
    }

    /// Get modality detector (for testing/diagnostics)
    pub fn hmgi_detector(&self) -> Option<&Arc<ModalityDetector>> {
        self.hmgi_detector.as_ref()
    }
}

#[cfg(test)]
mod adr011_routing_tests {
    //! TD-064 / ADR-011: policy-driven routing-mode selection. These tests
    //! exercise the small pure-fn surface (catalog policy → local mode)
    //! without spinning up a full AxisManager — that's the right
    //! granularity for the routing contract.
    use super::*;
    use proximadb_catalog::AnnFilteringPolicy;

    #[test]
    fn catalog_pre_filter_maps_to_local_pre_filter() {
        let catalog = proximadb_catalog::AnnFilteringMode::PreFilter;
        assert_eq!(ann_mode_from_catalog(catalog), AnnFilteringMode::PreFilter);
    }

    #[test]
    fn catalog_inline_maps_to_local_inline() {
        let catalog = proximadb_catalog::AnnFilteringMode::Inline;
        assert_eq!(ann_mode_from_catalog(catalog), AnnFilteringMode::Inline);
    }

    #[test]
    fn catalog_post_filter_maps_to_local_post_filter() {
        let catalog = proximadb_catalog::AnnFilteringMode::PostFilter;
        assert_eq!(ann_mode_from_catalog(catalog), AnnFilteringMode::PostFilter);
    }

    /// Defaults: <5% selectivity → PreFilter, 5–50% → Inline, >50% → PostFilter.
    #[test]
    fn default_policy_routes_by_selectivity_bands() {
        let policy = AnnFilteringPolicy::default();
        assert_eq!(
            ann_mode_from_catalog(policy.routing_mode(0.01)),
            AnnFilteringMode::PreFilter,
            "1% selectivity must hit PreFilter band"
        );
        assert_eq!(
            ann_mode_from_catalog(policy.routing_mode(0.20)),
            AnnFilteringMode::Inline,
            "20% selectivity must hit Inline band"
        );
        assert_eq!(
            ann_mode_from_catalog(policy.routing_mode(0.80)),
            AnnFilteringMode::PostFilter,
            "80% selectivity must hit PostFilter band"
        );
    }

    /// `force_mode` overrides selectivity — used by integration tests
    /// and emergency operator overrides.
    #[test]
    fn force_mode_overrides_selectivity() {
        let mut policy = AnnFilteringPolicy::default();
        policy.force_mode = Some(proximadb_catalog::AnnFilteringMode::Inline);
        // Selectivity that would normally route to PreFilter.
        assert_eq!(
            ann_mode_from_catalog(policy.routing_mode(0.001)),
            AnnFilteringMode::Inline,
            "force_mode must win over the selectivity band"
        );
    }

    /// PostFilter oversample uses `effective_top_k_for_post_filter` from
    /// the policy — default 2× of top_k. This is what
    /// `query_hnsw_with_predicate` consults when the caller routes
    /// PostFilter through the inline traversal.
    #[test]
    fn post_filter_oversample_uses_policy_factor() {
        let mut policy = AnnFilteringPolicy::default();
        policy.post_filter_oversample_factor = 3.0;
        let top_k = 10;
        assert_eq!(
            policy.effective_top_k_for_post_filter(top_k),
            30,
            "policy oversample factor must drive PostFilter request size"
        );
    }

    /// The `AxisHybridQuery` shape carries policy + selectivity so the
    /// caller can request policy-driven routing without inventing a side
    /// channel. Defaults preserve legacy behavior (no policy + no
    /// selectivity → caller's `ann_filtering_mode` is honored).
    #[test]
    fn default_query_has_no_policy_and_no_selectivity() {
        let q = AxisHybridQuery::default();
        assert!(q.ann_filtering_policy.is_none());
        assert!(q.estimated_selectivity.is_none());
    }

    #[test]
    fn quantized_route_requires_open_gate_and_quantized_storage() {
        use super::decide_quantized_route;
        // gate open + quantized storage → use the quantized accelerator
        assert!(decide_quantized_route(true, true));
        // gate closed (recall unverified / FAIL streak) → exact fallback
        assert!(!decide_quantized_route(false, true));
        // no quantized storage → exact regardless of gate
        assert!(!decide_quantized_route(true, false));
        assert!(!decide_quantized_route(false, false));
    }

    #[test]
    fn recall_at_k_measures_set_overlap() {
        use super::recall_at_k;
        let id = |s: &str| s.to_string();
        let exact = vec![id("a"), id("b"), id("c"), id("d")];
        // perfect recall (same set, any order)
        assert_eq!(
            recall_at_k(&exact, &[id("d"), id("c"), id("b"), id("a")], 4),
            1.0
        );
        // half the exact top-k recovered
        assert_eq!(
            recall_at_k(&exact, &[id("a"), id("b"), id("x"), id("y")], 4),
            0.5
        );
        // disjoint → 0
        assert_eq!(recall_at_k(&exact, &[id("x"), id("y")], 4), 0.0);
        // empty exact (nothing to recall) → 1.0
        assert_eq!(recall_at_k(&[], &[id("a")], 4), 1.0);
        // k clipped to exact len
        assert_eq!(recall_at_k(&[id("a")], &[id("a")], 10), 1.0);
    }

    #[test]
    fn recall_outcome_thresholds_on_floor() {
        use super::recall_outcome;
        use crate::catalog::ProbeOutcome;
        assert_eq!(recall_outcome(0.96, 0.95), ProbeOutcome::Pass);
        assert_eq!(recall_outcome(0.95, 0.95), ProbeOutcome::Pass); // >= floor
        assert_eq!(recall_outcome(0.94, 0.95), ProbeOutcome::Fail);
    }
}

#[cfg(test)]
mod recluster_apply_tests {
    //! Phase 8 F1 apply-step: `rebuild_and_swap_ivf_index` rebuilds a
    //! collection's IVF index and atomically swaps it as the served index.
    use super::*;
    use crate::index::axis::index_factory::AxisVectorIndex;
    use crate::index::axis::types::AxisConfig;
    use proximadb_records::{EmbeddingCell, EmbeddingValues};

    fn rec(id: &str, v: Vec<f32>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            // A core filterable field so the rebuild has metadata to carry —
            // lets `supports_predicate_search` distinguish add_with_metadata
            // (used now) from a plain add (the gap this closes).
            tenant_id: "t1".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "t",
                "dense_vector",
                v.len() as u32,
                v,
            )],
            ..ProximaRecord::default()
        }
    }

    fn batch(prefix: &str, n: usize, dim: usize, shift: usize) -> Vec<ProximaRecord> {
        (0..n)
            .map(|i| {
                let mut x = vec![0.0f32; dim];
                x[i % dim] = 1.0;
                x[(i / dim + shift) % dim] += 0.5;
                rec(&format!("{prefix}{i}"), x)
            })
            .collect()
    }

    /// `rec` plus a single filterable `grp` property for predicate-aware tests.
    fn rec_grp(id: &str, v: Vec<f32>, grp: &str) -> ProximaRecord {
        use proximadb_data_model::ProximaValue;
        use proximadb_records::ProximaTreeNode;
        let mut r = rec(id, v);
        r.props.insert(
            "grp".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(grp.to_string())),
        );
        r
    }

    /// TD-112 characterization lock-in.
    ///
    /// `handle_flushed_vectors` is the AXIS entry point that the flush/compaction
    /// path is *meant* to call to register newly-flushed vectors into the ANN
    /// index. This test proves the primitive works in isolation: when invoked, it
    /// indexes the batch into the manager (here observed via `collection_vectors`,
    /// which `insert` populates).
    ///
    /// The TD-112 live-path guard lives in SST's `do_flush` ->
    /// `flush_implementation`; this test keeps the lower-level indexing
    /// primitive locked down independently. If this test ever goes red, the
    /// primitive itself regressed rather than the SST call site.
    #[tokio::test]
    async fn td112_handle_flushed_vectors_indexes_flushed_batch() {
        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();
        let collection = "td112_flush";
        // Small batch: with the default `Synchronous` update mode, indexing
        // completes before `handle_flushed_vectors` returns (no async race).
        let flushed = batch("flushed", 4, 8, 0);
        let expected_oids: Vec<String> = flushed.iter().map(|r| r.oid.clone()).collect();

        manager
            .handle_flushed_vectors(collection, flushed, vec!["segment-000001.sst".to_string()])
            .await
            .expect("handle_flushed_vectors should index a small flushed batch");

        let stored = manager.collection_vectors.read().await;
        let collection_store = stored
            .get(collection)
            .expect("flushed vectors must be registered under the collection");
        for oid in &expected_oids {
            assert!(
                collection_store.contains_key(oid),
                "flushed vector {oid} must be indexed into AXIS by handle_flushed_vectors"
            );
        }
        assert_eq!(collection_store.len(), expected_oids.len());
    }

    /// TD-064 / 1.3 characterization: a selective metadata filter whose matches
    /// rank BEYOND the base 2× oversample window is still fully recovered by the
    /// predicate-aware HNSW walk — WITHOUT any oversample escalation. This is the
    /// evidence that escalation is unnecessary here: `search_layer_predicate`
    /// pushes every neighbour onto an unbounded frontier and only early-terminates
    /// once it holds `ef` PASSING results, so when matches are sparse it drains
    /// the whole reachable graph regardless of `oversample_k`. Guards against a
    /// future regression that would re-introduce premature termination.
    #[tokio::test]
    async fn predicate_aware_search_recovers_sparse_matches_without_oversampling() {
        use crate::compute::distance_computation::DistanceMetric;
        use crate::index::axis::indexes::hnsw_index::{AxisHnswConfig, AxisHnswIndex};
        use std::collections::HashMap;

        let dim = 8;
        let top_k = 5usize;
        let n = 40usize;
        // Matching records sit at ranks 12..36 — all well beyond the base
        // oversample window (2×top_k = 10) — but reachable along a connected
        // manifold (a line), so this exercises window/ef breadth, NOT graph
        // disconnection.
        let target_ranks = [12usize, 18, 24, 30, 36];

        let q = vec![0.0f32; dim]; // query at the origin (rank 0 is nearest)

        // (id, vector, grp): points spread along axis 0 at increasing distance,
        // so consecutive ranks are neighbours → a connected HNSW graph.
        let specs: Vec<(String, Vec<f32>, &str)> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[0] = 0.5 * i as f32;
                let grp = if target_ranks.contains(&i) {
                    "target"
                } else {
                    "other"
                };
                (format!("v{i:02}"), v, grp)
            })
            .collect();

        // Tiny search `ef`: `ef = config.ef.max(k)` so the base pool is 10.
        let config = AxisHnswConfig {
            ef: 4,
            distance_metric: DistanceMetric::Euclidean,
            ..AxisHnswConfig::default()
        };
        let index = AxisHnswIndex::new(config, dim).unwrap();
        for (id, v, _) in &specs {
            index.add(id.clone(), v.clone()).await.unwrap();
        }

        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();
        manager
            .hnsw_indexes
            .write()
            .await
            .insert("col".to_string(), std::sync::Arc::new(index));
        let by_id: HashMap<String, ProximaRecord> = specs
            .iter()
            .map(|(id, v, grp)| (id.clone(), rec_grp(id, v.clone(), grp)))
            .collect();
        manager
            .collection_vectors
            .write()
            .await
            .insert("col".to_string(), by_id);

        let query = AxisHybridQuery {
            collection_id: "col".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: q.clone(),
                similarity_threshold: 0.0,
            }),
            metadata_filters: vec![AxisMetadataFilter {
                field: "grp".to_string(),
                operator: FilterOperator::Equals,
                value: serde_json::json!("target"),
            }],
            id_filters: vec![],
            top_k,
            include_expired: false,
            ann_filtering_mode: AnnFilteringMode::Inline,
            ann_filtering_policy: None,
            estimated_selectivity: None,
            search_effort: None,
        };

        let (results, _shortfall) = manager
            .query_hnsw_with_predicate("col", &query, AnnFilteringMode::Inline)
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            top_k,
            "predicate-aware search must recover all {top_k} reachable matches \
             (ranks {target_ranks:?}) despite the base 2× window, got {}",
            results.len()
        );
        assert!(
            results.iter().all(|r| r.vector_id.starts_with('v')
                && target_ranks.contains(&r.vector_id[1..].parse::<usize>().unwrap_or(usize::MAX))),
            "only matching records may be returned: {:?}",
            results.iter().map(|r| &r.vector_id).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn cold_binary_index_serves_despite_closed_recall_gate() {
        // ADR-023 T-E: a cold-loaded (`ColdBinaryOnly`) index has no fp32 store.
        // The recall gate is CLOSED by default on a fresh process; without forcing
        // the binary route, `query_ivf` would route to exact search → empty fp32 →
        // empty results. The binary (Stage-1) route must be forced while cold.
        use crate::index::axis::indexes::dual_store_ivf::{UnifiedIvfConfig, UnifiedIvfIndex};
        let _ = proximadb_hardware::hardware_capabilities();
        let dim = 8;
        let cfg = UnifiedIvfConfig {
            use_binary: true,
            dimension: dim,
            n_clusters: 2,
            n_probe: 2,
            min_train_size: 4,
            ..Default::default()
        };
        let mut full = UnifiedIvfIndex::new("colb".to_string(), cfg).unwrap();
        let data: Vec<(String, Vec<f32>)> = (0..16)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| if (i + d) % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                (format!("v{i}"), v)
            })
            .collect();
        full.train(data.iter().map(|(_, v)| v.clone()).collect())
            .await
            .unwrap();
        for (id, v) in &data {
            full.add_vector(id.clone(), v.clone(), None).await.unwrap();
        }
        let cold = full.export_cold_tier().await.unwrap();
        let mut cold_idx =
            UnifiedIvfIndex::new("colb".to_string(), cold.config.to_config()).unwrap();
        cold_idx.restore_cold_only(cold).await.unwrap();
        assert_eq!(
            cold_idx.serving_state(),
            crate::index::axis::IvfServingState::ColdBinaryOnly
        );

        let m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.ivf_indexes.write().await.insert(
            "colb".to_string(),
            Arc::new(tokio::sync::RwLock::new(cold_idx)),
        );

        let query = AxisHybridQuery {
            collection_id: "colb".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: data[0].1.clone(),
                similarity_threshold: 0.0,
            }),
            top_k: 3,
            ..AxisHybridQuery::default()
        };
        // gate_open = false (the fresh-process default).
        let (results, _route) = m.query_ivf("colb", &query, false).await.unwrap();
        assert!(
            !results.is_empty(),
            "ColdBinaryOnly index must serve Stage-1 despite a closed recall gate"
        );
    }

    #[tokio::test]
    async fn served_index_presence_is_false_until_an_index_is_published() {
        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();

        assert!(!manager.has_served_index("pax-only").await);

        let records = batch("served", 40, 8, 1);
        assert!(
            manager
                .rebuild_and_swap_ivf_index("indexed", &records)
                .await
                .unwrap()
        );
        assert!(manager.has_served_index("indexed").await);
    }

    #[tokio::test]
    async fn rebuild_and_swap_serves_the_new_index() {
        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();
        let dim = 8;

        // First rebuild: 40 vectors -> built, swapped, generation 1.
        let v1 = batch("a", 40, dim, 1);
        assert!(
            manager
                .rebuild_and_swap_ivf_index("col", &v1)
                .await
                .unwrap()
        );
        assert_eq!(manager.index_generation("col").await, 1);
        assert!(manager.has_ivf_index("col").await);

        // The rebuilt index is populated + queryable.
        let mut qv = vec![0.0f32; dim];
        qv[0] = 1.0;
        let q = AxisHybridQuery {
            collection_id: "col".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: qv,
                similarity_threshold: 0.0,
            }),
            top_k: 5,
            ..AxisHybridQuery::default()
        };
        let (r1, _) = manager.query_ivf("col", &q, false).await.unwrap();
        assert!(!r1.is_empty(), "rebuilt index should be queryable");

        // Second rebuild: 25 DIFFERENT vectors -> generation 2, and the served
        // index's vector_count is now 25 (not 40) — proving the atomic swap
        // replaced the served index that `query_ivf` reads.
        let v2 = batch("b", 25, dim, 2);
        assert!(
            manager
                .rebuild_and_swap_ivf_index("col", &v2)
                .await
                .unwrap()
        );
        assert_eq!(manager.index_generation("col").await, 2);
        {
            let indexes = manager.ivf_indexes.read().await;
            let idx = indexes.get("col").unwrap().read().await;
            assert_eq!(
                idx.stats().vector_count,
                25,
                "served index must be the V2 rebuild (atomic swap)"
            );
            // Predicate-aware search gap closed: the rebuild carried filterable
            // metadata (add_with_metadata), so the index supports predicate search.
            assert!(
                idx.supports_predicate_search(),
                "rebuilt IVF index must carry filterable metadata"
            );
        }

        // Too few vectors -> no-op: no swap, generation unchanged.
        let tiny = batch("c", 5, dim, 1);
        assert!(
            !manager
                .rebuild_and_swap_ivf_index("col", &tiny)
                .await
                .unwrap()
        );
        assert_eq!(
            manager.index_generation("col").await,
            2,
            "no-op rebuild must not bump the generation"
        );
    }

    #[tokio::test]
    async fn hnsw_rebuild_and_swap_serves_the_new_index() {
        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();
        let dim = 8;

        // First rebuild: 40 vectors -> built, swapped, generation 1.
        let v1 = batch("a", 40, dim, 1);
        assert!(
            manager
                .rebuild_and_swap_hnsw_index("hcol", &v1)
                .await
                .unwrap()
        );
        assert_eq!(manager.index_generation("hcol").await, 1);
        assert!(manager.has_hnsw_index("hcol").await);

        // The rebuilt graph is populated + queryable.
        let mut qv = vec![0.0f32; dim];
        qv[0] = 1.0;
        let q = AxisHybridQuery {
            collection_id: "hcol".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: qv,
                similarity_threshold: 0.0,
            }),
            top_k: 5,
            ..AxisHybridQuery::default()
        };
        let r1 = manager.query_hnsw("hcol", &q).await.unwrap();
        assert!(!r1.is_empty(), "rebuilt HNSW index should be queryable");

        // Second rebuild via the unified orchestrator (routes to HNSW since the
        // collection has no IVF index): 25 vectors -> generation 2, and the
        // served graph's size is now 25 (not 40) — proving the atomic swap.
        let v2 = batch("b", 25, dim, 2);
        assert!(
            manager
                .rebuild_and_swap_served_index("hcol", &v2)
                .await
                .unwrap()
        );
        assert_eq!(manager.index_generation("hcol").await, 2);
        {
            let indexes = manager.hnsw_indexes.read().await;
            let idx = indexes.get("hcol").unwrap();
            assert_eq!(idx.size(), 25, "served HNSW index must be the V2 rebuild");
            // Predicate-aware search gap closed: rebuilt graph carries metadata.
            assert!(
                idx.supports_predicate_search(),
                "rebuilt HNSW index must carry filterable metadata"
            );
        }

        // Orchestrator is a no-op for a collection with no served index.
        assert!(
            !manager
                .rebuild_and_swap_served_index("never_seen", &v2)
                .await
                .unwrap()
        );
    }

    // ─── TD-087 Slice B: persist-after-train + load-on-demand ───────────────

    #[tokio::test]
    async fn ivf_index_persists_on_rebuild_and_warm_loads_on_query() {
        let dir = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let v = batch("p", 40, dim, 1);

        // Manager #1: persistence enabled → rebuild writes the index to disk.
        let mut m1 = AxisManager::new(AxisConfig::default()).await.unwrap();
        m1.set_index_persist_dir(dir.path().to_path_buf());
        assert!(m1.rebuild_and_swap_ivf_index("col", &v).await.unwrap());
        let path = dir.path().join("col").join("ivf.bin");
        assert!(path.exists(), "rebuild must persist the IVF index to disk");

        let mut qv = vec![0.0f32; dim];
        qv[3] = 1.0;
        let mk_query = || AxisHybridQuery {
            collection_id: "col".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: qv.clone(),
                similarity_threshold: 0.0,
            }),
            top_k: 3,
            ..AxisHybridQuery::default()
        };
        let want = m1.query(mk_query()).await.unwrap();
        let want_top: Vec<String> = want.results.iter().map(|r| r.vector_id.clone()).collect();
        assert!(!want_top.is_empty());

        // Manager #2: same dir, starts COLD (no in-memory index) → first query
        // warm-loads the index from disk and serves identical top-k.
        let mut m2 = AxisManager::new(AxisConfig::default()).await.unwrap();
        m2.set_index_persist_dir(dir.path().to_path_buf());
        assert!(!m2.has_ivf_index("col").await, "fresh manager starts cold");
        let got = m2.query(mk_query()).await.unwrap();
        assert!(
            m2.has_ivf_index("col").await,
            "query must warm-load the IVF index from disk"
        );
        let got_top: Vec<String> = got.results.iter().map(|r| r.vector_id.clone()).collect();
        assert_eq!(
            got_top, want_top,
            "warm-loaded index must serve identical top-k"
        );
    }

    #[tokio::test]
    async fn ivf_batch_path_persists_populated_index_td165_defect_b() {
        // TD-165 Defect B regression: the batch-index path
        // (`index_vectors_synchronously` ≥ 500 vectors) trains + persists the IVF
        // index BEFORE the insert loop populates its posting lists. Without the
        // post-insert re-persist, the on-disk `ivf.bin` would hold 0 vectors and a
        // cold reload would serve an empty index. Drive the real batch path, then
        // cold-load the persisted index and assert it actually contains the vectors.
        let dir = tempfile::TempDir::new().unwrap();
        let dim = 8;
        // 600 ≥ 500 (batch-train threshold) and ≤ 1000 (sync, fully awaited →
        // deterministic). batch() yields valid dense ProximaRecords.
        let recs = batch("v", 600, dim, 1);

        let mut m1 = AxisManager::new(AxisConfig::default()).await.unwrap();
        m1.set_index_persist_dir(dir.path().to_path_buf());
        m1.index_vectors_hybrid(
            "colb",
            recs,
            Vec::new(),
            &crate::index::config::RuntimeIndexConfig::default(),
        )
        .await
        .unwrap();

        let path = dir.path().join("colb").join("ivf.bin");
        assert!(
            path.exists(),
            "batch indexing must persist the IVF index to disk"
        );

        let mut qv = vec![0.0f32; dim];
        qv[3] = 1.0;
        let mk_query = || AxisHybridQuery {
            collection_id: "colb".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: qv.clone(),
                similarity_threshold: 0.0,
            }),
            top_k: 5,
            ..AxisHybridQuery::default()
        };
        let want: Vec<String> = m1
            .query(mk_query())
            .await
            .unwrap()
            .results
            .iter()
            .map(|r| r.vector_id.clone())
            .collect();
        assert!(
            !want.is_empty(),
            "warm batch-trained index must return results"
        );

        // Cold manager: the first query warm-loads the persisted ivf.bin and serves
        // from it. Pre-fix, ivf.bin was persisted with 0 vectors (posting lists were
        // never populated — `train` only fits centroids and `insert` feeds HNSW), so
        // the cold IVF query returned EMPTY. Populating the posting lists before the
        // persist is what makes the cold-loaded index serve real vectors. (We assert
        // non-empty rather than top-k equality because the warm path serves from
        // HNSW while the cold path serves from the IVF.)
        let mut m2 = AxisManager::new(AxisConfig::default()).await.unwrap();
        m2.set_index_persist_dir(dir.path().to_path_buf());
        assert!(!m2.has_ivf_index("colb").await, "fresh manager starts cold");
        let got: Vec<String> = m2
            .query(mk_query())
            .await
            .unwrap()
            .results
            .iter()
            .map(|r| r.vector_id.clone())
            .collect();
        assert!(
            !got.is_empty(),
            "TD-165 Defect B: cold-loaded IVF must serve real vectors (an empty persisted index returns nothing)"
        );
    }

    #[tokio::test]
    async fn ranged_cold_load_eager_and_pure_lazy_through_the_manager() {
        // ADR-023 R3 (b)/(c): exercise the RANGED manager load path end-to-end
        // with a persisted BINARY (v3) index — the default rebuild is non-binary,
        // so the other manager tests only hit the whole-file fallback. Verifies
        // both modes: EAGER (background ranged warm-apply → FullTwoStage) and
        // PURE-LAZY (eager fill skipped → stays ColdBinaryOnly, serves on-probe).
        use crate::index::axis::indexes::dual_store_ivf::{UnifiedIvfConfig, UnifiedIvfIndex};
        use crate::index::axis::storage::serialization::IndexSerializer;
        let _ = proximadb_hardware::hardware_capabilities();

        let dir = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let cfg = UnifiedIvfConfig {
            use_binary: true,
            dimension: dim,
            n_clusters: 2,
            n_probe: 2, // probe all → on-probe rerank is exact
            min_train_size: 4,
            ..Default::default()
        };
        let data: Vec<(String, Vec<f32>)> = (0..24)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| if (i + d) % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                (format!("v{i}"), v)
            })
            .collect();
        let mut full = UnifiedIvfIndex::new("colz".to_string(), cfg).unwrap();
        full.train(data.iter().map(|(_, v)| v.clone()).collect())
            .await
            .unwrap();
        for (id, v) in &data {
            full.add_vector(id.clone(), v.clone(), None).await.unwrap();
        }
        let q = data[0].1.clone();
        let want: Vec<String> = full
            .search(&q, 3, Some(2))
            .await
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        // Persist as a binary v3 index at the manager's expected path, via the
        // canonical FileSystem trait (ADR-023 R3 Slice 4).
        let path = dir.path().join("colz").join("ivf.bin");
        let local_fs: Arc<dyn crate::storage::persistence::filesystem::FileSystem> = Arc::new(
            crate::storage::persistence::filesystem::LocalFileSystem::new(
                crate::storage::persistence::filesystem::local::LocalConfig {
                    root_dir: None,
                    follow_symlinks: true,
                    default_permissions: None,
                    sync_enabled: false,
                },
            )
            .await
            .unwrap(),
        );
        IndexSerializer::persist_ivf_index(&full, "colz", &path.to_string_lossy(), &local_fs)
            .await
            .unwrap();

        // EAGER (default): ranged cold-load + background ranged warm-apply →
        // FullTwoStage within a short window.
        let mut m_eager = AxisManager::new(AxisConfig::default()).await.unwrap();
        m_eager.set_index_persist_dir(dir.path().to_path_buf());
        m_eager.ensure_ivf_index_loaded("colz").await;
        let idx_e = m_eager
            .ivf_indexes
            .read()
            .await
            .get("colz")
            .cloned()
            .expect("ranged warm-load installed the index");
        let mut flipped = false;
        for _ in 0..100 {
            if idx_e.read().await.serving_state()
                == crate::index::axis::IvfServingState::FullTwoStage
            {
                flipped = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            flipped,
            "eager ranged background warm-apply reaches FullTwoStage"
        );

        // PURE-LAZY: eager fill skipped → stays ColdBinaryOnly, serves on-probe.
        let mut m_lazy = AxisManager::new(AxisConfig::default()).await.unwrap();
        m_lazy.set_index_persist_dir(dir.path().to_path_buf());
        m_lazy.set_cold_lazy_warm(true);
        m_lazy.ensure_ivf_index_loaded("colz").await;
        let idx_l = m_lazy
            .ivf_indexes
            .read()
            .await
            .get("colz")
            .cloned()
            .expect("ranged warm-load installed the index");
        assert_eq!(
            idx_l.read().await.serving_state(),
            crate::index::axis::IvfServingState::ColdBinaryOnly,
            "pure-lazy keeps the index cold (no eager FullTwoStage fill)"
        );
        let got: Vec<String> = idx_l
            .read()
            .await
            .search_with_binary_acceleration(&q, 3, Some(2))
            .await
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(
            got, want,
            "pure-lazy on-probe rerank matches the warm top-k"
        );
        assert!(
            idx_l.read().await.fetched_cluster_count() <= 2,
            "on-probe fetched only the probed clusters"
        );

        // ADR-023 R3 (c) observability: cold_serving_status surfaces the warm
        // progress. Lazy stays ColdBinaryOnly with total=2 clusters in the
        // directory; the eager manager (flipped) reports FullTwoStage.
        let (lstate, lfetched, ltotal) = m_lazy.cold_serving_status("colz").await.unwrap();
        assert_eq!(lstate, crate::index::axis::IvfServingState::ColdBinaryOnly);
        assert_eq!(ltotal, 2, "byte-directory has both clusters");
        assert!(lfetched <= ltotal);
        let (estate, _, etotal) = m_eager.cold_serving_status("colz").await.unwrap();
        assert_eq!(estate, crate::index::axis::IvfServingState::FullTwoStage);
        assert_eq!(etotal, 2);
        assert!(
            m_eager.cold_serving_status("absent").await.is_none(),
            "no status for an unloaded collection"
        );
    }

    #[tokio::test]
    async fn persistence_disabled_is_a_noop() {
        // No persist dir set → rebuild succeeds and writes nothing (no panic/err).
        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();
        let v = batch("np", 40, 8, 1);
        assert!(manager.rebuild_and_swap_ivf_index("col", &v).await.unwrap());
        assert!(manager.index_fs_and_path("col").await.is_none());
    }

    /// ADR-023 R3 Slice 4 (convergence): index persistence + cold-load routed
    /// through an object-store URI via the canonical `FilesystemFactory`. `file://`
    /// here drives the IDENTICAL code path as `s3://`/`adls://`/`gs://` (scheme
    /// dispatch in `index_fs_and_path` → the same `FileSystem`-trait persist/load),
    /// so this proves the write side no longer diverges from the read side.
    #[tokio::test]
    async fn index_persist_via_object_store_uri_roundtrips_through_factory() {
        use crate::storage::persistence::filesystem::FilesystemFactory;
        let dir = tempfile::TempDir::new().unwrap();
        let url = format!("file://{}", dir.path().display());
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());

        // Persist through the factory (file:// backend) in URI mode.
        let mut m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.set_filesystem_factory(factory.clone());
        m.set_index_persist_url(url.clone());
        // URI mode takes precedence + reports persistence enabled.
        assert!(m.persistence_enabled());
        let (_, resolved) = m.index_fs_and_path("col").await.unwrap();
        assert!(
            resolved.starts_with("file://") && resolved.ends_with("/col/ivf.bin"),
            "URI-mode path: {resolved}"
        );
        let v = batch("uri", 60, 8, 1);
        assert!(m.rebuild_and_swap_ivf_index("col", &v).await.unwrap());
        assert!(
            m.has_persisted_ivf_index("col").await,
            "index persisted via object-store URI (factory file:// backend)"
        );

        // A fresh manager with the same URI + factory cold-loads it — the read
        // side resolves through the SAME factory/scheme path.
        let mut m2 = AxisManager::new(AxisConfig::default()).await.unwrap();
        m2.set_filesystem_factory(factory);
        m2.set_index_persist_url(url);
        assert!(!m2.has_ivf_index("col").await);
        m2.ensure_ivf_index_loaded("col").await;
        assert!(
            m2.has_ivf_index("col").await,
            "index cold-loaded via object-store URI through the factory"
        );
    }

    /// CATALOG_OBJECT_MODEL P1: a catalog-resolved index location takes precedence
    /// over the `index_persist_url` convention, and round-trips through the same
    /// factory/scheme path. Proves the index is addressable from the catalog
    /// (relocatable/tierable per projection), not just the base-url convention.
    #[tokio::test]
    async fn catalog_index_location_overrides_persist_url() {
        use crate::storage::persistence::filesystem::FilesystemFactory;
        let base = tempfile::TempDir::new().unwrap();
        let catalog_dir = tempfile::TempDir::new().unwrap();
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());

        let mut m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.set_filesystem_factory(factory.clone());
        // Convention base would resolve under {base}/col/ivf.bin …
        m.set_index_persist_url(format!("file://{}", base.path().display()));
        // … but the catalog-resolved per-projection location wins.
        let catalog_loc = format!("file://{}/indexes/vector_ann", catalog_dir.path().display());
        m.set_index_location("col", catalog_loc.clone()).await;

        let (_, resolved) = m.index_fs_and_path("col").await.unwrap();
        assert_eq!(
            resolved,
            format!("{}/ivf.bin", catalog_loc),
            "catalog index location must take precedence over index_persist_url"
        );
        assert!(
            !resolved.contains(&base.path().display().to_string()),
            "must NOT fall back to the convention base when a catalog location is set"
        );

        // And it actually persists + cold-loads through that catalog location.
        let v = batch("cat", 60, 8, 1);
        assert!(m.rebuild_and_swap_ivf_index("col", &v).await.unwrap());
        assert!(m.has_persisted_ivf_index("col").await);
    }

    /// CATALOG_OBJECT_MODEL #3 read-port: with no boot push and no convention URL,
    /// AXIS pulls the index location from the injected resolver on first use and
    /// memoizes it — the resolver is consulted at most once per collection. This is
    /// what makes catalog-resolved locations cover runtime-created collections.
    #[tokio::test]
    async fn read_port_resolves_index_location_lazily_and_memoizes() {
        use crate::storage::persistence::filesystem::FilesystemFactory;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingResolver {
            location: String,
            calls: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl IndexLocationResolver for CountingResolver {
            async fn resolve_index_location(&self, _collection_id: &str) -> Option<String> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Some(self.location.clone())
            }
        }

        let catalog_dir = tempfile::TempDir::new().unwrap();
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());
        let loc = format!("file://{}/indexes/vector_ann", catalog_dir.path().display());
        let calls = Arc::new(AtomicUsize::new(0));

        let mut m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.set_filesystem_factory(factory);
        // No explicit push and no convention url — only the read-port resolver.
        m.set_index_location_resolver(Arc::new(CountingResolver {
            location: loc.clone(),
            calls: calls.clone(),
        }));

        // First resolution pulls from the resolver …
        let (_, p1) = m.index_fs_and_path("col").await.unwrap();
        assert_eq!(p1, format!("{}/ivf.bin", loc));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // … and the result is memoized (no second pull).
        let (_, p2) = m.index_fs_and_path("col").await.unwrap();
        assert_eq!(p2, p1);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "resolver consulted at most once per collection (memoized)"
        );
    }

    /// Read-port returning `None` (no catalog location) falls back to the legacy
    /// `index_persist_url` convention — mixed-safe, default behavior preserved.
    #[tokio::test]
    async fn read_port_none_falls_back_to_convention() {
        use crate::storage::persistence::filesystem::FilesystemFactory;

        struct NoneResolver;
        #[async_trait::async_trait]
        impl IndexLocationResolver for NoneResolver {
            async fn resolve_index_location(&self, _c: &str) -> Option<String> {
                None
            }
        }

        let base = tempfile::TempDir::new().unwrap();
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());
        let mut m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.set_filesystem_factory(factory);
        m.set_index_persist_url(format!("file://{}", base.path().display()));
        m.set_index_location_resolver(Arc::new(NoneResolver));

        let (_, resolved) = m.index_fs_and_path("col").await.unwrap();
        assert!(
            resolved.ends_with("/col/ivf.bin")
                && resolved.contains(&base.path().display().to_string()),
            "None resolver must fall back to the convention: {resolved}"
        );
    }

    // ─── Phase 8 F4a: suspend / resume ──────────────────────────────────────

    fn suspend_query(dim: usize) -> AxisHybridQuery {
        let mut qv = vec![0.0f32; dim];
        qv[3] = 1.0;
        AxisHybridQuery {
            collection_id: "col".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: qv,
                similarity_threshold: 0.0,
            }),
            top_k: 3,
            ..AxisHybridQuery::default()
        }
    }

    #[tokio::test]
    async fn suspend_evicts_then_query_lazily_resumes() {
        let dir = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let mut m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.set_index_persist_dir(dir.path().to_path_buf());
        assert!(
            m.rebuild_and_swap_ivf_index("col", &batch("s", 40, dim, 1))
                .await
                .unwrap()
        );
        let top = |r: &AxisManagerQueryResult| {
            r.results
                .iter()
                .map(|x| x.vector_id.clone())
                .collect::<Vec<_>>()
        };
        let before = top(&m.query(suspend_query(dim)).await.unwrap());

        // Suspend: in-memory index evicted; file + strategy retained; marked.
        m.suspend_collection("col").await.unwrap();
        assert!(!m.has_ivf_index("col").await, "index evicted from memory");
        assert!(m.is_suspended("col").await);
        assert!(
            dir.path().join("col").join("ivf.bin").exists(),
            "persisted copy kept"
        );
        assert!(
            m.get_collection_strategy("col").await.is_ok(),
            "routing strategy retained across suspend"
        );

        // A query lazily warm-resumes from disk; identical top-k; marker cleared.
        let after = top(&m.query(suspend_query(dim)).await.unwrap());
        assert!(m.has_ivf_index("col").await, "query warm-loaded the index");
        assert!(
            !m.is_suspended("col").await,
            "warm-load cleared the suspend marker"
        );
        assert_eq!(after, before, "resumed index serves identical top-k");
    }

    #[tokio::test]
    async fn suspend_requires_persistence_and_an_in_memory_index() {
        // No persist dir → suspend errors (could not resume).
        let m = AxisManager::new(AxisConfig::default()).await.unwrap();
        assert!(
            m.rebuild_and_swap_ivf_index("col", &batch("s2", 40, 8, 1))
                .await
                .unwrap()
        );
        assert!(m.suspend_collection("col").await.is_err());

        // Persist dir set, but no in-memory index for the id → error.
        let dir = tempfile::TempDir::new().unwrap();
        let mut m2 = AxisManager::new(AxisConfig::default()).await.unwrap();
        m2.set_index_persist_dir(dir.path().to_path_buf());
        assert!(m2.suspend_collection("never").await.is_err());
    }

    #[tokio::test]
    async fn resume_collection_eagerly_warms_from_disk() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut m = AxisManager::new(AxisConfig::default()).await.unwrap();
        m.set_index_persist_dir(dir.path().to_path_buf());
        assert!(
            m.rebuild_and_swap_ivf_index("col", &batch("r", 40, 8, 1))
                .await
                .unwrap()
        );
        m.suspend_collection("col").await.unwrap();
        assert!(!m.has_ivf_index("col").await);
        assert!(
            m.resume_collection("col").await.unwrap(),
            "eager resume serves the index"
        );
        assert!(m.has_ivf_index("col").await);
        assert!(!m.is_suspended("col").await);
    }
}

#[cfg(test)]
mod hot_swap_ef_tests {
    //! Tests for AxisManager::apply_hnsw_ef_hot_swap — the
    //! zero-rebuild in-place ef_search tune that resolves
    //! DriftKind::EfSearchOnly drift on the route-health surface.
    use super::*;
    use crate::index::axis::types::{
        Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification,
    };

    async fn make_manager() -> AxisManager {
        AxisManager::new(AxisConfig::default())
            .await
            .expect("AxisManager::new")
    }

    fn hnsw_strategy(ef_search: u32, name: Option<&str>) -> IndexSelectionStrategy {
        let mut spec = IndexSpecification::new(
            Data::DenseVector { dimension: 128 },
            IndexAlgorithm::HNSW {
                m: 16,
                ef_construction: 200,
                ef_search,
                max_elements: 1_000_000,
            },
        );
        spec.name = name.map(String::from);
        IndexSelectionStrategy {
            indexes: vec![spec],
            routing_rules: vec![],
        }
    }

    #[tokio::test]
    async fn hot_swap_updates_ef_search_on_hnsw_spec() {
        let manager = make_manager().await;
        manager
            .update_collection_strategy("c1", hnsw_strategy(100, Some("primary")))
            .await
            .unwrap();

        let outcome = manager.apply_hnsw_ef_hot_swap("c1", 400).await.unwrap();
        match outcome {
            HotSwapOutcome::Applied { changes } => {
                assert_eq!(changes.len(), 1);
                assert_eq!(changes[0].previous_ef_search, 100);
                assert_eq!(changes[0].new_ef_search, 400);
                assert_eq!(changes[0].index_name.as_deref(), Some("primary"));
            }
            HotSwapOutcome::NotApplicable { reason } => {
                panic!("expected Applied, got NotApplicable: {}", reason)
            }
        }

        // The strategy must reflect the new ef on subsequent reads.
        let updated = manager.get_collection_strategy("c1").await.unwrap();
        match &updated.indexes[0].algorithm {
            IndexAlgorithm::HNSW { ef_search, .. } => {
                assert_eq!(*ef_search, 400, "ef_search must be live after hot-swap")
            }
            other => panic!("expected HNSW algorithm, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn hot_swap_preserves_m_and_ef_construction() {
        let manager = make_manager().await;
        manager
            .update_collection_strategy("c1", hnsw_strategy(100, None))
            .await
            .unwrap();
        let _ = manager.apply_hnsw_ef_hot_swap("c1", 400).await.unwrap();

        let updated = manager.get_collection_strategy("c1").await.unwrap();
        match &updated.indexes[0].algorithm {
            IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
                ..
            } => {
                assert_eq!(*m, 16, "m must be untouched by hot-swap");
                assert_eq!(*ef_construction, 200, "ef_construction must be untouched");
                assert_eq!(*ef_search, 400);
            }
            _ => panic!("expected HNSW"),
        }
    }

    #[tokio::test]
    async fn hot_swap_with_no_strategy_returns_not_applicable() {
        let manager = make_manager().await;
        let outcome = manager
            .apply_hnsw_ef_hot_swap("never_created", 400)
            .await
            .unwrap();
        assert!(matches!(outcome, HotSwapOutcome::NotApplicable { .. }));
    }

    #[tokio::test]
    async fn hot_swap_with_matching_ef_is_noop() {
        let manager = make_manager().await;
        manager
            .update_collection_strategy("c1", hnsw_strategy(400, None))
            .await
            .unwrap();
        let outcome = manager.apply_hnsw_ef_hot_swap("c1", 400).await.unwrap();
        match outcome {
            HotSwapOutcome::NotApplicable { reason } => {
                assert!(
                    reason.contains("no HNSW spec needed updating"),
                    "reason should explain why: {}",
                    reason
                );
            }
            HotSwapOutcome::Applied { changes } => {
                panic!(
                    "expected NotApplicable for matching ef, got {} changes",
                    changes.len()
                )
            }
        }
    }

    #[tokio::test]
    async fn rebuild_for_recall_target_sizes_from_advisor() {
        use proximadb_records::{EmbeddingCell, EmbeddingValues};

        let manager = make_manager().await;

        // Build N=200 records (well above the dimension=8 minimum)
        // with simple unit vectors so the rebuild has real data.
        let dim = 8usize;
        let mut records: Vec<ProximaRecord> = Vec::with_capacity(200);
        for i in 0..200 {
            let mut v = vec![0.0_f32; dim];
            v[i % dim] = 1.0;
            let r = ProximaRecord {
                oid: format!("v{i}"),
                embeddings: vec![EmbeddingCell::new_fp32(
                    "test",
                    "dense_vector",
                    dim as u32,
                    v,
                )],
                ..Default::default()
            };
            records.push(r);
        }

        let advised = manager
            .rebuild_and_swap_hnsw_index_for_recall_target("c_rebuild", &records, 0.95, 10, None)
            .await
            .unwrap()
            .expect("rebuild should return advisor output");

        // recall_target=0.95 falls in the m=32 tier per the advisor.
        assert_eq!(advised.m, 32);
        assert_eq!(advised.ef_construction, 256);
        assert!(advised.ef_search >= 16, "ef_search must clear the floor");

        // The strategy must reflect the new sizing so subsequent
        // queries see it via the standard lookup path.
        let strategy = manager.get_collection_strategy("c_rebuild").await.unwrap();
        let hnsw_count = strategy
            .indexes
            .iter()
            .filter(|s| matches!(s.algorithm, IndexAlgorithm::HNSW { .. }))
            .count();
        assert_eq!(hnsw_count, 1, "strategy must carry one HNSW spec");
        match &strategy.indexes[0].algorithm {
            IndexAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
                ..
            } => {
                assert_eq!(*m, advised.m);
                assert_eq!(*ef_construction, advised.ef_construction);
                assert_eq!(*ef_search, advised.ef_search);
            }
            other => panic!("expected HNSW, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn rebuild_for_recall_target_with_no_records_returns_none() {
        let manager = make_manager().await;
        let advised = manager
            .rebuild_and_swap_hnsw_index_for_recall_target("c_empty", &[], 0.95, 10, None)
            .await
            .unwrap();
        assert!(advised.is_none(), "no records → no rebuild");
    }

    fn ivf_strategy(nprobe: u32, name: Option<&str>) -> IndexSelectionStrategy {
        let mut spec = IndexSpecification::new(
            Data::DenseVector { dimension: 128 },
            IndexAlgorithm::IVF {
                nlist: 256,
                nprobe,
                quantizer: None,
            },
        );
        spec.name = name.map(String::from);
        IndexSelectionStrategy {
            indexes: vec![spec],
            routing_rules: vec![],
        }
    }

    #[tokio::test]
    async fn apply_ivf_nprobe_hot_swap_updates_nprobe() {
        let manager = make_manager().await;
        manager
            .update_collection_strategy("c_ivf", ivf_strategy(10, Some("ivf_primary")))
            .await
            .unwrap();

        let outcome = manager
            .apply_ivf_nprobe_hot_swap("c_ivf", 40)
            .await
            .unwrap();
        match outcome {
            HotSwapOutcome::Applied { changes } => {
                assert_eq!(changes.len(), 1);
                assert_eq!(changes[0].previous_ef_search, 10);
                assert_eq!(changes[0].new_ef_search, 40);
            }
            HotSwapOutcome::NotApplicable { reason } => {
                panic!("expected Applied, got NotApplicable: {}", reason);
            }
        }
        let updated = manager.get_collection_strategy("c_ivf").await.unwrap();
        match &updated.indexes[0].algorithm {
            IndexAlgorithm::IVF { nprobe, .. } => {
                assert_eq!(*nprobe, 40, "nprobe must be live after hot-swap");
            }
            other => panic!("expected IVF, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn apply_ivf_nprobe_hot_swap_preserves_nlist() {
        let manager = make_manager().await;
        manager
            .update_collection_strategy("c_ivf", ivf_strategy(10, None))
            .await
            .unwrap();
        let _ = manager
            .apply_ivf_nprobe_hot_swap("c_ivf", 40)
            .await
            .unwrap();

        let updated = manager.get_collection_strategy("c_ivf").await.unwrap();
        match &updated.indexes[0].algorithm {
            IndexAlgorithm::IVF {
                nlist,
                nprobe,
                quantizer,
            } => {
                assert_eq!(*nlist, 256, "nlist must be untouched by hot-swap");
                assert_eq!(*nprobe, 40);
                assert!(quantizer.is_none(), "quantizer must stay None");
            }
            _ => panic!("expected IVF"),
        }
    }

    #[tokio::test]
    async fn apply_ivf_nprobe_hot_swap_no_strategy_returns_not_applicable() {
        let manager = make_manager().await;
        let outcome = manager
            .apply_ivf_nprobe_hot_swap("never_created", 40)
            .await
            .unwrap();
        assert!(matches!(outcome, HotSwapOutcome::NotApplicable { .. }));
    }

    #[tokio::test]
    async fn apply_ivf_nprobe_hot_swap_matching_nprobe_is_noop() {
        let manager = make_manager().await;
        manager
            .update_collection_strategy("c_ivf", ivf_strategy(40, None))
            .await
            .unwrap();
        let outcome = manager
            .apply_ivf_nprobe_hot_swap("c_ivf", 40)
            .await
            .unwrap();
        assert!(
            matches!(outcome, HotSwapOutcome::NotApplicable { .. }),
            "matching nprobe must be NotApplicable, got {:?}",
            outcome
        );
    }

    #[tokio::test]
    async fn apply_ivf_nprobe_hot_swap_skips_non_ivf_specs() {
        let manager = make_manager().await;
        // HNSW-only strategy — no IVF spec to swap.
        manager
            .update_collection_strategy("c_hnsw_only", hnsw_strategy(100, None))
            .await
            .unwrap();
        let outcome = manager
            .apply_ivf_nprobe_hot_swap("c_hnsw_only", 40)
            .await
            .unwrap();
        assert!(matches!(outcome, HotSwapOutcome::NotApplicable { .. }));
    }

    #[tokio::test]
    async fn rebuild_and_swap_ivf_with_no_records_returns_none() {
        let manager = make_manager().await;
        let advised = manager
            .rebuild_and_swap_ivf_index_for_recall_target(
                "c_empty_ivf",
                &[],
                0.70,
                10,
                None,
                None,
                false,
            )
            .await
            .unwrap();
        assert!(advised.is_none(), "no records → no IVF rebuild");
    }

    #[tokio::test]
    async fn hot_swap_skips_non_hnsw_specs() {
        let manager = make_manager().await;
        // IVF-only strategy — no HNSW spec to swap.
        let ivf_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 128 },
            IndexAlgorithm::IVF {
                nlist: 100,
                nprobe: 10,
                quantizer: None,
            },
        );
        manager
            .update_collection_strategy(
                "c1",
                IndexSelectionStrategy {
                    indexes: vec![ivf_spec],
                    routing_rules: vec![],
                },
            )
            .await
            .unwrap();

        let outcome = manager.apply_hnsw_ef_hot_swap("c1", 400).await.unwrap();
        assert!(matches!(outcome, HotSwapOutcome::NotApplicable { .. }));
    }
}

#[cfg(test)]
mod clear_collection_vectors_tests {
    //! Tests for `AxisManager::clear_collection_vectors` — the in-memory reaping
    //! step that must run after a successful storage flush (TD-FLUSH-1). Before the
    //! fix, `db.flush()` cleared the WAL memtable batches but left the AxisManager's
    //! `collection_vectors` projection fully resident → OOM for large workloads and
    //! warm GETs serving from the unreaped copy (cold-read bugs masked).
    use super::*;
    use chrono::Utc;
    use proximadb_records::{EmbeddingCell, LabelSet, ProximaRecord, ProximaTree};

    async fn make_manager() -> AxisManager {
        AxisManager::new(AxisConfig::default())
            .await
            .expect("AxisManager::new")
    }

    fn make_record(oid: &str) -> ProximaRecord {
        let now_ns = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        ProximaRecord {
            oid: oid.to_string(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 1,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: None,
            memory_type: None,
            props: ProximaTree::new(),
            refs: Vec::new(),
            edge: None,
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn clear_collection_vectors_empties_the_in_memory_projection() {
        // The core TD-FLUSH-1 regression: after flush, the in-memory projection must
        // be gone. Before the fix this map was never cleared → unbounded memory +
        // warm reads served from the stale copy.
        let manager = make_manager().await;

        // Insert two records into collection "c1" + one into "c2".
        manager
            .insert_record("c1", &make_record("v1"))
            .await
            .unwrap();
        manager
            .insert_record("c1", &make_record("v2"))
            .await
            .unwrap();
        manager
            .insert_record("c2", &make_record("v3"))
            .await
            .unwrap();

        assert_eq!(manager.registered_vector_count("c1").await, 2);
        assert_eq!(manager.registered_vector_count("c2").await, 1);

        // Simulate the flush callback reaping c1's in-memory projection.
        manager.clear_collection_vectors("c1").await.unwrap();

        assert_eq!(
            manager.registered_vector_count("c1").await,
            0,
            "c1's collection_vectors must be reaped after flush"
        );
        // c2 is untouched (clear is per-collection, not global).
        assert_eq!(
            manager.registered_vector_count("c2").await,
            1,
            "clear_collection_vectors must not touch other collections"
        );
    }

    #[tokio::test]
    async fn clear_collection_vectors_is_idempotent_on_unknown_collection() {
        // Flush iterates collections-with-unflushed-data; a collection that had no
        // WAL batches (or was already reaped) must not error when cleared again.
        let manager = make_manager().await;
        // No inserts — collection "ghost" has never been seen.
        manager.clear_collection_vectors("ghost").await.unwrap();
        assert_eq!(manager.registered_vector_count("ghost").await, 0);
    }

    #[tokio::test]
    async fn clear_collection_vectors_allows_repopulation_after_flush() {
        // After a flush reaps the projection, subsequent inserts (a new write batch
        // landed between flush and reap, or a fresh post-flush insert) must repopulate
        // the map cleanly — the projection is a cache, not the durable store.
        let manager = make_manager().await;
        manager
            .insert_record("c1", &make_record("v1"))
            .await
            .unwrap();
        assert_eq!(manager.registered_vector_count("c1").await, 1);

        manager.clear_collection_vectors("c1").await.unwrap();
        assert_eq!(manager.registered_vector_count("c1").await, 0);

        // A fresh insert after the reap must land back in the projection.
        manager
            .insert_record("c1", &make_record("v2"))
            .await
            .unwrap();
        assert_eq!(
            manager.registered_vector_count("c1").await,
            1,
            "post-flush insert must repopulate the projection"
        );
    }
}

#[cfg(test)]
mod hmgi_lifecycle_tests {
    use super::*;
    use proximadb_records::{EmbeddingCell, EmbeddingValues};

    #[tokio::test]
    async fn failed_enable_does_not_publish_or_leave_owned_partitions() {
        let manager = AxisManager::new(AxisConfig::default()).await.unwrap();
        let collection_id = "invalid_hmgi";
        let invalid_vector = vec![0.0; 100_001];
        let record = ProximaRecord {
            oid: "oversized_embedding".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                invalid_vector.len() as u32,
                invalid_vector,
            )],
            ..Default::default()
        };
        manager
            .collection_vectors
            .write()
            .await
            .entry(collection_id.to_string())
            .or_default()
            .insert(record.oid.clone(), record);

        let error = manager
            .enable_hmgi(collection_id, None, 42)
            .await
            .err()
            .expect("oversized partition migration must fail");

        assert!(error.to_string().contains("dimension"));
        assert!(!manager.is_hmgi_enabled(collection_id).await);
        assert!(
            !manager
                .hmgi_collection_oids
                .read()
                .await
                .contains_key(collection_id)
        );
        let registry = manager.hmgi_registry().unwrap();
        assert!(
            registry
                .get_partitions_for_collection(collection_id)
                .await
                .is_empty()
        );
        assert!(registry.is_empty().await);
    }
}
