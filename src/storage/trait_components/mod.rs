//! # Storage Engine Trait Components
//!
//! This module provides a decomposed trait hierarchy for storage engines,
//! following the Interface Segregation Principle (ISP) from SOLID.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! UnifiedStorageFormat (composite - requires all sub-traits)
//! ├── StorageIdentity       (engine_name, version, strategy, capabilities)
//! ├── StorageReader         (vector_by_id, search_vectors_unified)
//! ├── StorageWriter         (flush, staging operations)
//! ├── StorageCompactor      (compact, compaction heuristics)
//! ├── StorageMetrics        (metrics, health_check)
//! ├── StorageScan           (create_scan, scan_capabilities)
//! └── StorageLifecycle      (optimize, statistics)
//! ```
//!
//! ## Benefits
//!
//! 1. **Interface Segregation**: Clients depend only on traits they use
//! 2. **Single Responsibility**: Each trait has one clear purpose
//! 3. **Easier Testing**: Mock only the traits you need
//! 4. **Incremental Implementation**: New engines can implement traits gradually
//!
//! ## Usage
//!
//! Import the specific traits you need:
//! ```rust,ignore
//! use crate::storage::trait_components::{StorageIdentity, StorageReader};
//! ```
//!
//! Or use the re-exports from the main traits module:
//! ```rust,ignore
//! use crate::storage::traits::{StorageIdentity, StorageReader};
//! ```

pub mod capabilities;
mod compactor;
pub mod extractor;
mod identity;
mod lifecycle;
mod metrics;
pub mod path_resolver;
mod reader;
mod scan;
mod writer;

// Re-export all sub-traits
pub use capabilities::{
    CapabilityFactory, CompactionHeuristics, EngineBundle, EngineCapabilities, FlushThresholds,
    HelixCapabilities, NovaCapabilities, RaptorCapabilities, SstCapabilities, SwiftCapabilities,
};
pub use compactor::StorageCompactor;
pub use extractor::{
    ExtractedVector, ExtractionCapabilities, ExtractionCost, ExtractionError, ExtractionFactory,
    ExtractionMode, ExtractionRequest, ExtractionResult, ExtractionScope, ExtractionStats,
    QuantizedVector, VectorExtractor,
};
pub use identity::StorageIdentity;
pub use lifecycle::StorageLifecycle;
pub use metrics::StorageMetrics;
pub use path_resolver::{
    CachedResolver, CollectionPathResolver, CompositeResolver, ConfigFallbackResolver,
    StorageAssignment,
};
pub use reader::StorageReader;
pub use scan::StorageScan;
pub use writer::StorageWriter;

#[cfg(test)]
mod tests {
    #[test]
    fn test_trait_components_compile() {
        // If this compiles, the trait components are correctly defined
        assert!(true);
    }
}
