//! Dependency-inversion port traits for ProximaDB storage (Slice D).
//!
//! Storage is high-level policy (engines, compaction, flush) that *drives*
//! higher-level collaborators (collection metadata, the ANN index). To let
//! `src/storage` depend *down* instead of *up* into `crate::services` /
//! `crate::index`, those collaborators are expressed here as narrow traits.
//! The concrete implementors (e.g. `CollectionService`) `impl` these traits in
//! their own modules, and the composition root injects `Arc<dyn …Port>` into
//! storage at startup. No upward edge anywhere.
//!
//! This crate is trait-only: no behavior, no concrete types beyond the stable
//! foundation proto types the contracts unavoidably name.

use anyhow::Result;
use proximadb_distance_kernel::DistanceMetric;
use proximadb_graph_model::{GraphOperation, GraphWalEntry, MarkerKind};
use proximadb_proto::proximadb_v1::Collection;
use proximadb_storage_common::StorageEngineType;
use proximadb_storage_filesystem_types::{DirEntry, FileOptions, FileSystem, FsResult};
use std::sync::Arc;

pub mod capabilities;
pub use capabilities::*;
/// The fenced atomic OLTP uniqueness primitive (ADR-072 D3/D11/D14).
pub mod conditional_key_store;
pub use conditional_key_store::{
    AtomicScope, ConditionalKeyStore, Generation, Identity, Oid, PutOutcome,
};
pub mod scan_strategy;
pub mod writer_epoch;
pub use scan_strategy::{ScanCostEstimate, ScanIterator, ScanStatistics, ScanStrategy};
pub use writer_epoch::{EpochGrant, EpochStore, WriterClaim};
pub mod path_resolver;
pub use path_resolver::{
    CollectionPathResolver, StorageAssignment, collection_data_path_typed,
    typed_identity_from_storage_assignment, typed_path_identity, typed_path_identity_when,
    typed_paths_enabled,
};

/// Read access to collection metadata that storage needs at flush/compaction
/// time (fetch the proto `Collection` for a name or UUID).
///
/// Inverts `crate::services::collection::CollectionService`: storage holds an
/// `Arc<dyn CollectionMetadataPort>` and never references the service crate.
/// The measured storage-driven surface is exactly one method — every call site
/// (viper engine/flush, the flush coordinator, the background-flush context)
/// only fetches the collection; richer service behavior stays in the service.
#[async_trait::async_trait]
pub trait CollectionMetadataPort: Send + Sync {
    /// Fetch the full proto collection (with all metadata) by name or UUID.
    /// Returns `Ok(None)` when the collection does not exist.
    async fn collection(&self, identifier: &str) -> Result<Option<Collection>>;
}

/// Event-log operations that storage needs when flush/compaction publishes
/// index-maintenance work.
///
/// Inverts the storage dependency on AXIS event-log concrete types. Storage
/// emits storage-native facts (collection, files, engine, quantization flags);
/// higher layers translate them into AXIS `IndexEvent`s.
#[async_trait::async_trait]
pub trait StorageEventLogPort: Send + Sync {
    /// Record a flush event and wait until the event log acknowledges it.
    async fn notify_flush(
        &self,
        collection_id: &str,
        flushed_files: Vec<String>,
        vector_count: usize,
        storage_engine: StorageEngineType,
        has_quantized: bool,
        has_fp32: bool,
    ) -> Result<()>;

    /// Record a completed compaction event.
    async fn notify_compaction(
        &self,
        collection_id: &str,
        output_files: Vec<String>,
        vector_count: usize,
        storage_engine: StorageEngineType,
    ) -> Result<()>;

    /// Return whether a file can be compacted. Implementations should fail
    /// open when the event log is unavailable.
    async fn can_compact(&self, collection_id: &str, file_path: &str) -> Result<bool>;
}

/// Level-specific vector quantization + Hamming distance that storage's
/// search/compaction paths need.
///
/// Inverts `crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine`
/// (modality-tier `proximadb-quantization-kernel`, re-exported up through
/// `compute::quantization`): storage holds an `Arc<dyn QuantizationEnginePort>`
/// and never names the concrete engine, so the storage→modality up-edge dissolves.
///
/// The measured storage-driven surface is the level-specific `quantize_to_*`
/// encoders plus Hamming distance — every signature is primitive in/out
/// (`&[f32]` → `Vec<u8>`/`Vec<u16>`/tuples/`u32`), so NO modality type crosses
/// the seam (no facade DTOs needed). The generic level-based `quantize(level)`
/// is intentionally NOT here: it would drag the modality `UnifiedQuantizationLevel`
/// type across the boundary. Callers use the level-specific method instead.
///
/// The concrete `UnifiedQuantizationEngine` `impl`s this in its own crate — a
/// downward dep (modality→storage-ports is layering-allowed; the forbidden
/// direction is storage→modality). The composition root injects the impl.
pub trait QuantizationEnginePort: Send + Sync {
    /// 1-bit sign/threshold quantization → packed bits.
    fn quantize_to_binary(&self, vector: &[f32]) -> Result<Vec<u8>>;
    /// 8-bit signed scalar quantization.
    fn quantize_to_int8(&self, vector: &[f32]) -> Result<Vec<u8>>;
    /// 4-bit scalar quantization → (codes, min, max, len).
    fn quantize_to_u4(&self, vector: &[f32]) -> Result<(Vec<u8>, f32, f32, usize)>;
    /// 6-bit scalar quantization → (codes, min, max, len).
    fn quantize_to_u6(&self, vector: &[f32]) -> Result<(Vec<u8>, f32, f32, usize)>;
    /// 8-bit unsigned scalar quantization → (codes, min, max).
    fn quantize_to_u8(&self, vector: &[f32]) -> Result<(Vec<u8>, f32, f32)>;
    /// 16-bit unsigned scalar quantization → (codes, min, max).
    fn quantize_to_u16(&self, vector: &[f32]) -> Result<(Vec<u16>, f32, f32)>;
    /// Product quantization → packed codes.
    fn quantize_to_pq(
        &self,
        vector: &[f32],
        num_subvectors: usize,
        bits_per_code: u32,
    ) -> Result<Vec<u8>>;
    /// Hamming distance between two bit-packed binary codes.
    fn calculate_hamming_distance(&self, a: &[u8], b: &[u8]) -> u32;
}

/// Graph collection operations that observability's telemetry-graph-linker needs.
///
/// Inverts `GraphOperationsService` (root `src/graph/service.rs`): observability
/// holds `Arc<dyn GraphOperationsPort>` instead of `Arc<GraphService>`. The
/// measured surface is 2 methods, both foundation-typed (proto `CreateGraphRequest`
/// + `GraphCollection`, `Vec<String>`). The concrete `GraphOperationsService`
/// `impl`s this in the root (it defines the type → can impl any trait for it).
#[async_trait::async_trait]
pub trait GraphOperationsPort: Send + Sync {
    /// List all graph collection IDs.
    async fn list_graphs(&self) -> Result<Vec<String>>;

    /// Create a new graph collection. Returns the created collection.
    async fn create_graph_collection(
        &self,
        request: proximadb_proto::proximadb_v1::CreateGraphRequest,
    ) -> Result<std::sync::Arc<proximadb_proto::proximadb_v1::GraphCollection>>;
}

use proximadb_quantization_model::{
    QuantizedVector, StorageQuantizedData, UnifiedQuantizationLevel,
};

/// Collection-aware batch quantization + dequantization that storage's
/// write/compaction/search paths need.
///
/// Inverts `StorageQuantizationEngine` (modality-tier kernel): storage holds an
/// `Arc<dyn StorageQuantizationEnginePort>` and never names the concrete engine.
/// The return types (`Vec<StorageQuantizedData>`, `Vec<f32>`) and params
/// (`UnifiedQuantizationLevel`, `&QuantizedVector`) are all foundation types
/// (moved by #842) → **facade-FREE** (no DTO/adaptor layer).
///
/// The concrete `StorageQuantizationEngine` `impl`s this in its own crate (a
/// downward dep — modality→storage-ports is layering-allowed). The composition
/// root injects the impl. Training (`train(&mut self)`) is NOT on the port
/// (it's `&mut self`, not object-safe) — training happens before injection.
#[async_trait::async_trait]
pub trait StorageQuantizationEnginePort: Send + Sync {
    /// Quantize a batch of vectors using the engine's configured levels.
    async fn quantize_batch(
        &self,
        vectors: &[Vec<f32>],
        ids: Option<&[String]>,
    ) -> Result<Vec<StorageQuantizedData>>;

    /// Quantize a batch with a specific level override.
    async fn quantize_batch_with_level(
        &self,
        vectors: &[Vec<f32>],
        level: UnifiedQuantizationLevel,
    ) -> Result<Vec<StorageQuantizedData>>;

    /// Dequantize a vector back to approximate float values.
    async fn dequantize(&self, quantized: &QuantizedVector) -> Result<Vec<f32>>;
}

/// Filesystem access port — inverts the storage→root `FilesystemFactory` dependency.
///
/// Engine leaves hold `Arc<dyn FilesystemPort>` instead of the root-local
/// `FilesystemFactory` concrete type, so engine modules (`viper/pipeline`,
/// `raptor/*`, …) can move to crates. The surface is the routing + staging methods
/// engines actually use — measured across `EngineFilesystemAccess`'s default
/// methods, `trait_components::writer`, and the engine tests. The concrete
/// `FilesystemFactory` impls this in the root crate (a downward edge —
/// root→storage-ports is layering-allowed); the composition root injects it.
///
/// This unblocks the engine-leaf extraction (see
/// `docs/12-design/ROOT_CRATE_DECOMPOSITION_ENGINES_EXTRACTION_2026_07_12.adoc`):
/// leaves swap their `Arc<FilesystemFactory>` fields for `Arc<dyn FilesystemPort>`.
#[async_trait::async_trait]
pub trait FilesystemPort: Send + Sync {
    /// Resolve the `FileSystem` for a URL's scheme (cached; routed by scheme).
    fn get_filesystem(&self, url: &str) -> FsResult<std::sync::Arc<dyn FileSystem>>;
    /// Recursively create the directory at `url`.
    async fn create_dir_all(&self, url: &str) -> FsResult<()>;
    /// Write `data` to `url`.
    async fn write(&self, url: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()>;
    /// Atomically move `from_url` → `to_url`.
    async fn move_atomic(&self, from_url: &str, to_url: &str) -> FsResult<()>;
    /// Delete the file/dir at `url`.
    async fn delete(&self, url: &str) -> FsResult<()>;
    /// Read the full contents of the file at `url` (scheme-routed).
    async fn read(&self, url: &str) -> FsResult<Vec<u8>>;
    /// List the directory entries at `url` (scheme-routed).
    async fn list(&self, url: &str) -> FsResult<Vec<DirEntry>>;
}

/// Cache-tier usage snapshot reported by engines/services to the
/// `CrossCacheOrchestrator` — hoisted from the root cache orchestrator
/// (TD-DECOMP-82) so extracted engine leaves can implement the provider
/// without a root dependency.
#[derive(Clone, Debug)]
pub struct UsageStats {
    /// Hit rate over the reporting window (0.0–1.0).
    pub hit_rate: f64,
    /// Average entry size in bytes.
    pub avg_entry_size: usize,
    /// Mean access frequency (accesses/second over the window).
    pub access_frequency: f64,
    /// When the tier was last rebalanced.
    pub last_rebalance: std::time::SystemTime,
}

/// Provider trait for engines/services to report usage snapshots — the root
/// orchestrator collects these per `CacheType` for dynamic rebalancing.
pub trait CacheStatsProvider {
    /// Report a usage snapshot for the provider's cache tier.
    fn snapshot(&self) -> UsageStats;
}

/// Cache-kind for access-pattern tracking — the engine-facing subset of the root
/// `CacheType` (the 5 variants engines actually track, measured across
/// `src/storage/engines/`: VectorData, Metadata, DistanceTable, FilterBitmap,
/// IndexStructure). Foundation-neutral so engine leaves can name it without
/// depending on the root-local `CacheType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheKind {
    VectorData,
    Metadata,
    DistanceTable,
    FilterBitmap,
    IndexStructure,
}

/// Cache access-pattern tracking port — inverts the storage→root
/// `CrossCacheOrchestrator` dependency for access tracking.
///
/// Engine leaves hold `Arc<dyn CacheAccessPatternPort>` instead of the root-local
/// `CrossCacheOrchestrator` (a heavily-coupled global singleton), so engine modules
/// can move to crates. The surface is exactly the one method engines use —
/// `pattern_tracker().track_access_async(key, cache_type)` — measured across 12
/// engine files. The concrete `CrossCacheOrchestrator` impls this in the root crate
/// (downward edge); the composition root injects it. Non-blocking (the underlying
/// tracker queues events for background processing).
pub trait CacheAccessPatternPort: Send + Sync {
    /// Record a cache access for pattern learning / predictive prefetching.
    fn track_access(&self, key: String, cache_kind: CacheKind);
}

/// AXIS clustering port — inverts the raptor→root `AxisClusteringEngine`
/// dependency for the clustering surface RAPTOR's writer needs (k-means
/// clustering, centroid-distance matrix, component-boosted assignment). The
/// surface is exactly `ReusableClusteringEngine`'s 3 methods (sync,
/// primitive-typed + `DistanceMetric`). Engine leaves hold `Arc<dyn
/// AxisClusteringPort>` instead of the root-local `AxisClusteringEngine`; the
/// concrete engine impls this in the root (downward edge). Enables the raptor
/// `writer` extraction.
pub trait AxisClusteringPort: Send + Sync {
    /// k-means clustering (k-means++ init) -> (centroids, cluster assignments).
    fn cluster_vectors_simple(
        &self,
        vectors: &[Vec<f32>],
        k: usize,
        distance_metric: DistanceMetric,
        max_iterations: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>)>;

    /// Centroid-to-centroid distance matrix (k*k).
    fn calculate_centroid_distance_matrix(
        &self,
        centroids: &[Vec<f32>],
        distance_metric: DistanceMetric,
    ) -> Result<Vec<Vec<f32>>>;

    /// Assign vectors to clusters with component boosting -> (cluster_id, boosted_distance).
    fn assign_vectors_with_component_boosting(
        &self,
        vectors: &[Vec<f32>],
        centroids: &[Vec<f32>],
        centroid_distances: &[Vec<f32>],
        distance_metric: DistanceMetric,
        boosting_weights: &[f32],
    ) -> Result<Vec<(usize, f32)>>;
}

/// Dependency-inversion port for a graph engine's WAL sink (ORION extraction).
///
/// ORION (and future graph engines) append graph operations + canonical-sync
/// markers through this port rather than naming the concrete unified WAL
/// writer/operation types, so the engine can be extracted to a crate without a
/// cyclic dependency on the root crate's storage layer. The composition root
/// injects a concrete impl (e.g. the unified WAL writer, which wraps
/// [`GraphOperation`] / [`MarkerKind`] into the unified operation enum).
#[async_trait::async_trait]
pub trait GraphWalPort: Send + Sync {
    /// Append a graph operation; returns the assigned sequence number (LSN).
    async fn append_graph_op(&mut self, op: GraphOperation) -> Result<u64>;

    /// Append a non-data canonical-sync marker; returns the assigned sequence
    /// number (LSN).
    async fn append_graph_marker(&mut self, marker: MarkerKind) -> Result<u64>;

    /// Flush any buffered WAL frames to durable storage.
    async fn flush(&mut self) -> Result<()>;

    /// Reclaim WAL segments fully covered by a durable snapshot whose canonical
    /// checkpoint is at `checkpoint_lsn` (every segment whose frames all precede
    /// the matching `CanonicalEmission(checkpoint_lsn)` marker). Returns the
    /// number of segments reclaimed.
    async fn truncate_through_canonical_marker(&mut self, checkpoint_lsn: u64) -> Result<u64>;
}

/// Dependency-inversion port for a graph engine's WAL *reader* (ORION
/// extraction) — the read-side counterpart to [`GraphWalPort`].
///
/// A graph engine replays its WAL through this port rather than naming the
/// concrete unified WAL reader/entry/operation types. The port yields only the
/// graph-relevant frames ([`GraphWalEntry`]); non-graph unified operations are
/// filtered out by the reader. As with [`GraphWalPort`], the composition root
/// injects a concrete impl (e.g. the unified WAL reader), so the engine can be
/// extracted to a crate without a cyclic dependency on the root storage layer.
#[async_trait::async_trait]
pub trait GraphWalReaderPort: Send + Sync {
    /// Read every graph-relevant frame from the WAL, in append order. Returns an
    /// empty vector when the WAL is absent or empty (e.g. before the first write).
    async fn read_all_graph(&self) -> Result<Vec<GraphWalEntry>>;
}

/// Dependency-injection factory for a graph engine's WAL writer + reader (ORION
/// extraction).
///
/// This is the seam that lets the engine obtain its [`GraphWalPort`] (writer)
/// and [`GraphWalReaderPort`] (reader) WITHOUT naming the concrete unified WAL
/// types: the engine calls `make_writer` / `make_reader` at construction, and
/// the composition root injects a concrete factory (the unified WAL factory),
/// which is the only place the concrete types are named. Without this, the
/// engine would construct the writer/reader itself and could not be extracted
/// to a crate. The factory is injected once at the composition root and threaded
/// down through the engine constructors — the single object that crosses the
/// engine↔root boundary.
#[async_trait::async_trait]
pub trait GraphWalFactory: Send + Sync {
    /// Build a graph WAL writer backed by `wal_path`, wrapped in the async mutex
    /// the writer is shared under (`append` & friends take `&mut self`).
    async fn make_writer(
        &self,
        wal_path: &str,
    ) -> Result<Arc<tokio::sync::Mutex<dyn GraphWalPort>>>;

    /// Build a graph WAL reader backed by `wal_path`. Opens no files until a
    /// read is issued; tolerant of an absent/empty WAL.
    async fn make_reader(&self, wal_path: &str) -> Result<Arc<dyn GraphWalReaderPort>>;

    /// Build the filesystem port the engine uses for snapshot I/O (read/list).
    /// The engine never names the concrete filesystem factory; this is the
    /// single composition-root seam that does (mirrors make_writer/make_reader).
    async fn make_filesystem(&self) -> Result<Arc<dyn FilesystemPort>>;

    /// Build the canonical-WAL reader the engine uses for TD-066 checkpoint-LSN
    /// correlation on recovery. The engine never names the concrete appender.
    async fn make_canonical_wal_reader(&self) -> Result<Arc<dyn CanonicalWalReaderPort>>;
}

/// Graph-engine cache-hint surface (ORION extraction). A graph engine (ORION)
/// emits best-effort access-tracking + prefetch hints to the root-owned cache
/// orchestrator through this port, registered once at startup, so the engine
/// never names the concrete `CrossCacheOrchestrator` / `CacheType`. Mirrors the
/// distance-kernel GPU-factory global-registration pattern (#162). The
/// [`CacheKind`] enum (the storage-engine subset) intentionally omits graph
/// kinds, so this enum carries the graph variants the engine tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphCacheKind {
    GraphNode,
    GraphEdge,
    GraphAdjacency,
}

/// Global graph cache-hint port (write-once, set at server startup by the
/// composition root; read by graph engines).
#[async_trait::async_trait]
pub trait GraphCacheHintPort: Send + Sync {
    /// Record an access for the access-pattern tracker (best-effort, fire-and-forget).
    fn track_access(&self, key: String, kind: GraphCacheKind);
    /// Request a predictive prefetch (best-effort; queue-size-guarded internally).
    async fn request_prefetch(&self, key: &str, kind: GraphCacheKind);
}

static GLOBAL_GRAPH_CACHE_HINT: std::sync::OnceLock<Arc<dyn GraphCacheHintPort>> =
    std::sync::OnceLock::new();

/// Register the global graph cache-hint port. Called once by the composition
/// root at startup (alongside `CrossCacheOrchestrator::register_global`).
pub fn register_graph_cache_hint(hint: Arc<dyn GraphCacheHintPort>) {
    let _ = GLOBAL_GRAPH_CACHE_HINT.set(hint);
}

/// Access the registered graph cache-hint port, if any. Graph engines call this
/// and treat `None` as a no-op (e.g. before startup registration or in tests).
pub fn graph_cache_hint() -> Option<&'static Arc<dyn GraphCacheHintPort>> {
    GLOBAL_GRAPH_CACHE_HINT.get()
}

/// Dependency-injection port for reading the shared canonical WAL (ORION
/// extraction). The engine reads canonical-WAL entries through this port — to
/// correlate the latest checkpoint LSN during recovery (TD-066 Part 2) — rather
/// than naming the concrete framed-table appender. Entries are the leaf
/// [`CanonicalWalEntry`] type.
#[async_trait::async_trait]
pub trait CanonicalWalReaderPort: Send + Sync {
    /// Read every entry from the canonical WAL at `canonical_wal_path`.
    async fn read_entries(
        &self,
        canonical_wal_path: &std::path::Path,
    ) -> Result<Vec<proximadb_storage_common::wal_entry::CanonicalWalEntry>>;
}

// ===========================================================================
// TD-DECOMP-66: Port-traits to resolve grandfathered control-layer coupling
// ===========================================================================

/// Inverts the orion-engine → metrics (modality→control) edge.
///
/// The concrete impl (bumping Prometheus counters) is registered at bootstrap;
/// `None` (pre-registration, tests) is a no-op — recovery behavior is unchanged.
pub trait OrionMetricsPort: Send + Sync {
    /// Record an ORION recovery checkpoint observation (TD-066 Part 2 Option E).
    /// Best-effort, infallible — observability must never fail the caller.
    fn record_recovery_checkpoint_observation(
        &self,
        graph_id: &str,
        checkpoint_lsn: Option<u64>,
        checkpoint_timestamp_ms: Option<u64>,
    );
}

static GLOBAL_ORION_METRICS: std::sync::OnceLock<Arc<dyn OrionMetricsPort>> =
    std::sync::OnceLock::new();

/// Register the global ORION metrics port. Called once by the composition root.
pub fn register_orion_metrics(m: Arc<dyn OrionMetricsPort>) {
    let _ = GLOBAL_ORION_METRICS.set(m);
}

/// Access the registered ORION metrics port, if any. ORION engine calls this
/// and treats `None` as a no-op (tests, pre-registration).
pub fn orion_metrics() -> Option<&'static Arc<dyn OrionMetricsPort>> {
    GLOBAL_ORION_METRICS.get()
}

/// Inverts the storage-operations → catalog (storage→control) edge.
///
/// The concrete impl delegates to `CorpusVersionRegistry::global()`;
/// `None` (pre-registration, tests) is a no-op — compaction is unaffected.
#[async_trait::async_trait]
pub trait CorpusVersionPort: Send + Sync {
    /// Bump the corpus version for every tenant that has cached plans against
    /// this collection. Returns the number of entries bumped.
    async fn bump_collection_all_tenants(&self, collection: &str) -> usize;
}

static GLOBAL_CORPUS_VERSION: std::sync::OnceLock<Arc<dyn CorpusVersionPort>> =
    std::sync::OnceLock::new();

/// Register the global corpus-version port. Called once by the composition root.
pub fn register_corpus_version(p: Arc<dyn CorpusVersionPort>) {
    let _ = GLOBAL_CORPUS_VERSION.set(p);
}

/// Access the registered corpus-version port, if any. The compaction manager
/// calls this and treats `None` as a no-op (tests, pre-registration).
pub fn corpus_version() -> Option<&'static Arc<dyn CorpusVersionPort>> {
    GLOBAL_CORPUS_VERSION.get()
}
