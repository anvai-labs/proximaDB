//! Engine capability descriptors — pure-metadata subsystem (hoisted from the
//! root crate's `src/storage/trait_components/capabilities.rs`,
//! `src/storage/scan_strategy.rs`, and `src/storage/traits/types.rs`).
//!
//! This is a cohesive cluster of ~10 types that describe *what an engine can
//! do* without naming any concrete engine (no `SstEngine` / `ViperEngine` /
//! … references — verified via grep). It hoists as a unit because every member
//! depends only on the others, on proto (`CompressionAlgorithm`,
//! `StorageEngine`), on `serde`, and on `std`. Clearing it from the root lets
//! the engine-port-traits module move to its own crate; the last root-dep of
//! that module was the CapabilityFactory type.
//!
//! The old import paths are preserved via `pub use` re-export shims so every
//! existing caller resolves unchanged (see the three source files listed
//! above).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use proximadb_proto::proximadb_v1::{CompressionAlgorithm, StorageEngine};

// ---------------------------------------------------------------------------
// Storage engine strategy enum (moved from `src/storage/traits/types.rs`)
// ---------------------------------------------------------------------------

/// Storage engine strategy enumeration for polymorphic engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StorageEngineStrategy {
    /// SST engine - hybrid columnar with ProximaBlocks
    #[default]
    Sst,

    /// VIPER engine - columnar Parquet with advanced quantization
    Viper,

    /// HELIX engine - spatial-locality vector storage (Hilbert-curve clustering, zone maps)
    Helix,

    /// NOVA engine - next-gen columnar with integrated quantization
    Nova,

    /// SWIFT engine - hierarchical superblock architecture (`experimental-engines` gate; incomplete)
    Swift,

    /// RAPTOR engine - experimental parallel tiered storage (`experimental-engines` gate)
    Raptor,

    /// Hybrid - reserved, not implemented (the factory falls back to SST)
    Hybrid,

    /// TST engine - time-series optimized storage
    TimeSeries,

    /// CEDAR engine - document-oriented compatibility projection (Columnar Extensible Document Archive)
    Cedar,

    /// CHRONO engine - removed; retained for wire compatibility (the factory returns an error)
    Chrono,
}

/// Backwards-compat **format** alias for [`StorageEngineStrategy`] (engines →
/// formats convergence). Variants are reached through the alias, e.g.
/// `StorageFormatStrategy::Sst`. New code may use this name;
/// `StorageEngineStrategy` remains during the migration window (see
/// `docs/12-design/NAMING_CONVENTIONS.adoc`).
pub type StorageFormatStrategy = StorageEngineStrategy;

// ---------------------------------------------------------------------------
// Scan capabilities (moved from `src/storage/scan_strategy.rs`)
// ---------------------------------------------------------------------------

/// Scan capabilities derived from actual engine implementations
#[derive(Debug, Clone, Default)]
pub struct ScanCapabilities {
    // From VIPER/NOVA (columnar engines)
    pub supports_predicate_pushdown: bool,
    pub supports_column_projection: bool,
    pub supports_row_group_pruning: bool,
    pub supports_parallel_column_evaluation: bool,

    // From SST (hybrid columnar engine)
    pub supports_bloom_filters: bool,
    pub supports_block_cache: bool,
    pub supports_range_scans: bool,
    pub supports_index_scans: bool,

    // From NOVA (progressive search)
    pub supports_progressive_quantization: bool,
    pub supports_zone_maps: bool,
    pub supports_streaming: bool,

    // From RAPTOR (tiered storage)
    pub supports_tier_aware_scanning: bool,
    pub supports_consolidated_reading: bool,
}

// ---------------------------------------------------------------------------
// Engine capability config structs (moved from
// `src/storage/trait_components/capabilities.rs`)
// ---------------------------------------------------------------------------

/// Flush thresholds configuration
#[derive(Debug, Clone)]
pub struct FlushThresholds {
    /// Memory threshold in bytes (flush when exceeded)
    pub memory_threshold_bytes: usize,
    /// Entry count threshold (flush when exceeded)
    pub entry_count_threshold: usize,
    /// Time threshold in seconds (flush if idle for this long)
    pub time_threshold_secs: u64,
    /// Whether to use global buffer behavior
    pub use_global_buffer: bool,
}

impl Default for FlushThresholds {
    fn default() -> Self {
        Self {
            memory_threshold_bytes: 16 * 1024 * 1024, // 16 MB
            entry_count_threshold: 10_000,
            time_threshold_secs: 60,
            use_global_buffer: true,
        }
    }
}

/// Compaction heuristics configuration
#[derive(Debug, Clone)]
pub struct CompactionHeuristics {
    /// Minimum number of SST files to trigger compaction
    pub min_files_to_compact: usize,
    /// Maximum file size ratio for level-based compaction
    pub max_level_ratio: f64,
    /// Target file size after compaction
    pub target_file_size_bytes: usize,
    /// Whether to use size-tiered compaction
    pub size_tiered: bool,
    /// Whether to use leveled compaction
    pub leveled: bool,
}

impl Default for CompactionHeuristics {
    fn default() -> Self {
        Self {
            min_files_to_compact: 4,
            max_level_ratio: 10.0,
            target_file_size_bytes: 64 * 1024 * 1024, // 64 MB
            size_tiered: true,
            leveled: false,
        }
    }
}

/// Engine capabilities trait (OCP-compliant interface)
///
/// Each storage engine implementation provides its capabilities through this trait.
/// This eliminates match statements that hardcode engine-specific behavior.
pub trait EngineCapabilities: Send + Sync {
    /// Engine name for logging/debugging
    fn engine_name(&self) -> &'static str;

    /// Get scan capabilities for this engine
    fn scan_capabilities(&self) -> ScanCapabilities;

    /// Whether this engine supports collection-level operations
    fn supports_collection_level_operations(&self) -> bool;

    /// Whether this engine supports atomic operations
    fn supports_atomic_operations(&self) -> bool;

    /// Get flush thresholds for this engine
    fn flush_thresholds(&self) -> FlushThresholds {
        FlushThresholds::default()
    }

    /// Get compaction heuristics for this engine
    fn compaction_heuristics(&self) -> CompactionHeuristics {
        CompactionHeuristics::default()
    }

    /// Get supported compression algorithms
    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        // Default: common algorithms
        let mut supported = HashSet::new();
        supported.insert(CompressionAlgorithm::CompressionNone);
        supported.insert(CompressionAlgorithm::CompressionZstd);
        supported.insert(CompressionAlgorithm::CompressionLz4);
        supported.insert(CompressionAlgorithm::CompressionSnappy);
        supported
    }

    /// Check if a specific compression algorithm is supported
    fn is_compression_supported(&self, algorithm: CompressionAlgorithm) -> bool {
        self.supported_compression().contains(&algorithm)
    }

    /// Whether this engine supports progressive quantization search
    fn supports_progressive_quantization(&self) -> bool {
        self.scan_capabilities().supports_progressive_quantization
    }

    /// Whether this engine supports zone maps for pruning
    fn supports_zone_maps(&self) -> bool {
        self.scan_capabilities().supports_zone_maps
    }

    /// Whether this engine supports bloom filters
    fn supports_bloom_filters(&self) -> bool {
        self.scan_capabilities().supports_bloom_filters
    }
}

// ============================================================================
// Per-Engine Capability Implementations
// ============================================================================

/// SST engine capabilities
pub struct SstCapabilities;

impl EngineCapabilities for SstCapabilities {
    fn engine_name(&self) -> &'static str {
        "SST"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_bloom_filters: true,
            supports_block_cache: true,
            supports_range_scans: true,
            supports_index_scans: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        false // SST operates on entire tree
    }

    fn supports_atomic_operations(&self) -> bool {
        false
    }

    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        let mut supported = HashSet::new();
        supported.insert(CompressionAlgorithm::CompressionNone);
        supported.insert(CompressionAlgorithm::CompressionZstd);
        supported.insert(CompressionAlgorithm::CompressionLz4);
        supported.insert(CompressionAlgorithm::CompressionSnappy);
        supported.insert(CompressionAlgorithm::CompressionGzip);
        supported.insert(CompressionAlgorithm::CompressionBrotli);
        supported.insert(CompressionAlgorithm::CompressionBzip2);
        supported.insert(CompressionAlgorithm::CompressionDeflate);
        supported.insert(CompressionAlgorithm::CompressionXz);
        supported.insert(CompressionAlgorithm::CompressionZlib);
        supported.insert(CompressionAlgorithm::CompressionLz4hc);
        supported.insert(CompressionAlgorithm::CompressionLzma);
        supported
    }
}

/// HELIX engine capabilities
pub struct HelixCapabilities;

impl EngineCapabilities for HelixCapabilities {
    fn engine_name(&self) -> &'static str {
        "HELIX"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_predicate_pushdown: true,
            supports_bloom_filters: true,
            supports_block_cache: true,
            supports_range_scans: true,
            supports_index_scans: true,
            supports_progressive_quantization: true,
            supports_zone_maps: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        true // HELIX supports collection-level ops
    }

    fn supports_atomic_operations(&self) -> bool {
        true // HELIX provides atomic guarantees
    }

    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        // Same as SST (SST-based engine)
        SstCapabilities.supported_compression()
    }
}

/// VIPER engine capabilities
pub struct ViperCapabilities;

impl EngineCapabilities for ViperCapabilities {
    fn engine_name(&self) -> &'static str {
        "VIPER"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_predicate_pushdown: true,
            supports_column_projection: true,
            supports_row_group_pruning: true,
            supports_parallel_column_evaluation: true,
            supports_zone_maps: true,
            supports_streaming: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        true // VIPER supports collection-level ops
    }

    fn supports_atomic_operations(&self) -> bool {
        true
    }

    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        // Parquet-compatible algorithms
        let mut supported = HashSet::new();
        supported.insert(CompressionAlgorithm::CompressionNone);
        supported.insert(CompressionAlgorithm::CompressionZstd);
        supported.insert(CompressionAlgorithm::CompressionLz4);
        supported.insert(CompressionAlgorithm::CompressionSnappy);
        supported.insert(CompressionAlgorithm::CompressionGzip);
        supported.insert(CompressionAlgorithm::CompressionBrotli);
        supported
    }
}

/// SWIFT engine capabilities
pub struct SwiftCapabilities;

impl EngineCapabilities for SwiftCapabilities {
    fn engine_name(&self) -> &'static str {
        "SWIFT"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_bloom_filters: true,
            supports_block_cache: true,
            supports_range_scans: true,
            supports_index_scans: true,
            supports_tier_aware_scanning: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        true // SWIFT supports collection-level ops
    }

    fn supports_atomic_operations(&self) -> bool {
        true // SWIFT provides atomic guarantees
    }

    fn flush_thresholds(&self) -> FlushThresholds {
        FlushThresholds {
            memory_threshold_bytes: 8 * 1024 * 1024, // 8 MB (smaller for low-latency)
            entry_count_threshold: 5_000,
            time_threshold_secs: 30,
            use_global_buffer: false, // SWIFT manages its own buffers
        }
    }
}

/// NOVA engine capabilities
pub struct NovaCapabilities;

impl EngineCapabilities for NovaCapabilities {
    fn engine_name(&self) -> &'static str {
        "NOVA"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_predicate_pushdown: true,
            supports_column_projection: true,
            supports_row_group_pruning: true,
            supports_parallel_column_evaluation: true,
            supports_progressive_quantization: true,
            supports_zone_maps: true,
            supports_streaming: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        true
    }

    fn supports_atomic_operations(&self) -> bool {
        true
    }

    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        // Same as VIPER (columnar-based)
        ViperCapabilities.supported_compression()
    }
}

/// TST engine capabilities
pub struct TstCapabilities;

impl EngineCapabilities for TstCapabilities {
    fn engine_name(&self) -> &'static str {
        "TST"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_predicate_pushdown: true,
            supports_column_projection: true,
            supports_row_group_pruning: true,
            supports_parallel_column_evaluation: true,
            supports_block_cache: true,
            supports_range_scans: true,
            supports_zone_maps: true,
            supports_streaming: true,
            supports_tier_aware_scanning: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        true
    }

    fn supports_atomic_operations(&self) -> bool {
        true
    }

    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        ViperCapabilities.supported_compression()
    }
}

/// RAPTOR engine capabilities
pub struct RaptorCapabilities;

impl EngineCapabilities for RaptorCapabilities {
    fn engine_name(&self) -> &'static str {
        "RAPTOR"
    }

    fn scan_capabilities(&self) -> ScanCapabilities {
        ScanCapabilities {
            supports_predicate_pushdown: true,
            supports_column_projection: true,
            supports_row_group_pruning: true,
            supports_bloom_filters: true,
            supports_streaming: true,
            supports_tier_aware_scanning: true,
            supports_consolidated_reading: true,
            ..Default::default()
        }
    }

    fn supports_collection_level_operations(&self) -> bool {
        true // RAPTOR supports collection-level ops
    }

    fn supports_atomic_operations(&self) -> bool {
        false // RAPTOR uses eventual consistency
    }

    fn compaction_heuristics(&self) -> CompactionHeuristics {
        CompactionHeuristics {
            min_files_to_compact: 2, // RAPTOR uses consolidated reading
            max_level_ratio: 5.0,
            target_file_size_bytes: 128 * 1024 * 1024, // 128 MB (larger files)
            size_tiered: true,
            leveled: true, // RAPTOR uses hybrid approach
        }
    }

    fn supported_compression(&self) -> HashSet<CompressionAlgorithm> {
        // Same as VIPER (columnar-based)
        ViperCapabilities.supported_compression()
    }
}

/// Engine bundle: pairs engine instance with its capabilities
///
/// This is the return type from the factory, ensuring capabilities
/// are always available alongside the engine.
pub struct EngineBundle<E> {
    /// The engine instance
    pub engine: E,
    /// The engine's capabilities
    pub capabilities: Box<dyn EngineCapabilities>,
}

impl<E> EngineBundle<E> {
    /// Create a new engine bundle
    pub fn new(engine: E, capabilities: Box<dyn EngineCapabilities>) -> Self {
        Self {
            engine,
            capabilities,
        }
    }
}

/// Factory for creating capability instances based on engine strategy
pub struct CapabilityFactory;

impl CapabilityFactory {
    /// Create capabilities for a given engine strategy
    pub fn create(strategy: StorageFormatStrategy) -> Box<dyn EngineCapabilities> {
        match strategy {
            StorageFormatStrategy::Sst => Box::new(SstCapabilities),
            StorageFormatStrategy::Helix => Box::new(HelixCapabilities),
            StorageFormatStrategy::Viper => Box::new(ViperCapabilities),
            StorageFormatStrategy::Swift => Box::new(SwiftCapabilities),
            StorageFormatStrategy::Nova => Box::new(NovaCapabilities),
            StorageFormatStrategy::TimeSeries => Box::new(TstCapabilities),
            StorageFormatStrategy::Raptor => Box::new(RaptorCapabilities),
            // Default to SST capabilities for unknown strategies
            _ => Box::new(SstCapabilities),
        }
    }

    /// Create capabilities from proto StorageEngine enum (for static utility bridge)
    ///
    /// This method bridges the static `EngineCapabilities` utility with the trait-based
    /// capability system, enabling OCP compliance while maintaining the static API.
    pub fn from_proto_engine(engine: StorageEngine) -> Box<dyn EngineCapabilities> {
        match engine {
            StorageEngine::Sst => Box::new(SstCapabilities),
            StorageEngine::Helix => Box::new(HelixCapabilities),
            StorageEngine::Viper => Box::new(ViperCapabilities),
            StorageEngine::Swift => Box::new(SwiftCapabilities),
            StorageEngine::Nova => Box::new(NovaCapabilities),
            StorageEngine::Tst => Box::new(TstCapabilities),
            StorageEngine::Raptor => Box::new(RaptorCapabilities),
            // Default to SST capabilities for unknown engines
            _ => Box::new(SstCapabilities),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_format_strategy_alias_is_interchangeable() {
        // The format alias names the same type — default + variants match.
        assert_eq!(StorageFormatStrategy::default(), StorageEngineStrategy::Sst);
        let s: StorageFormatStrategy = StorageEngineStrategy::Viper;
        assert_eq!(s, StorageEngineStrategy::Viper);
    }

    #[test]
    fn storage_engine_strategy_default_is_sst() {
        assert_eq!(StorageEngineStrategy::default(), StorageEngineStrategy::Sst);
    }

    #[test]
    fn test_sst_capabilities() {
        let caps = SstCapabilities;
        assert_eq!(caps.engine_name(), "SST");
        assert!(caps.supports_bloom_filters());
        assert!(!caps.supports_zone_maps());
        assert!(!caps.supports_collection_level_operations());
    }

    #[test]
    fn test_viper_capabilities() {
        let caps = ViperCapabilities;
        assert_eq!(caps.engine_name(), "VIPER");
        assert!(caps.scan_capabilities().supports_predicate_pushdown);
        assert!(caps.scan_capabilities().supports_column_projection);
        assert!(caps.supports_collection_level_operations());
    }

    #[test]
    fn test_helix_capabilities() {
        let caps = HelixCapabilities;
        assert_eq!(caps.engine_name(), "HELIX");
        assert!(caps.supports_progressive_quantization());
        assert!(caps.supports_zone_maps());
    }

    #[test]
    fn test_nova_capabilities() {
        let caps = NovaCapabilities;
        assert_eq!(caps.engine_name(), "NOVA");
        assert!(caps.supports_progressive_quantization());
        assert!(caps.supports_atomic_operations());
    }

    #[test]
    fn test_raptor_capabilities() {
        let caps = RaptorCapabilities;
        assert_eq!(caps.engine_name(), "RAPTOR");
        assert!(caps.scan_capabilities().supports_consolidated_reading);
        assert!(caps.scan_capabilities().supports_tier_aware_scanning);

        let heuristics = caps.compaction_heuristics();
        assert_eq!(heuristics.min_files_to_compact, 2);
    }

    #[test]
    fn test_swift_capabilities() {
        let caps = SwiftCapabilities;
        assert_eq!(caps.engine_name(), "SWIFT");

        let thresholds = caps.flush_thresholds();
        assert_eq!(thresholds.memory_threshold_bytes, 8 * 1024 * 1024);
        assert!(!thresholds.use_global_buffer);
    }

    #[test]
    fn test_compression_support() {
        let sst = SstCapabilities;
        assert!(sst.is_compression_supported(CompressionAlgorithm::CompressionZstd));
        assert!(sst.is_compression_supported(CompressionAlgorithm::CompressionLz4));

        let viper = ViperCapabilities;
        // VIPER supports fewer algorithms (Parquet-compatible)
        assert!(viper.is_compression_supported(CompressionAlgorithm::CompressionZstd));
    }

    #[test]
    fn test_capability_factory() {
        let sst_caps = CapabilityFactory::create(StorageFormatStrategy::Sst);
        assert_eq!(sst_caps.engine_name(), "SST");

        let helix_caps = CapabilityFactory::create(StorageFormatStrategy::Helix);
        assert_eq!(helix_caps.engine_name(), "HELIX");
    }

    #[test]
    fn test_capability_factory_from_proto_engine() {
        let caps = CapabilityFactory::from_proto_engine(StorageEngine::Viper);
        assert_eq!(caps.engine_name(), "VIPER");
    }

    #[test]
    fn test_engine_bundle() {
        struct MockEngine;

        let bundle = EngineBundle::new(MockEngine, Box::new(SstCapabilities));
        assert_eq!(bundle.capabilities.engine_name(), "SST");
    }
}
