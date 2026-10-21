//! Core infrastructure for storage engines
//!
//! This module contains all shared infrastructure used by storage engine implementations.
//! It provides a clean separation between infrastructure (how things work) and
//! implementations (what engines do).

pub mod adaptive_strategy_optimizer; // Intelligent strategy optimization and tuning
pub use proximadb_engine_core::coalesce_strategy; // extracted TD-DECOMP-79 (cost-driven, tier-aware read-coalescing chooser, TD-RDSTRAT-1)
pub mod columnar_format_serializer;
pub mod constants; // Centralized constants for all storage engines
pub mod filter_evaluator; // Unified filter evaluation for all engines
pub mod formats; // Storage formats: row-based, columnar
pub mod io; // I/O operations: zero-copy, filesystem abstractions
pub mod matrix_trinity_serializer; // Matrix Trinity (RAPTOR) metadata serializer
pub mod metadata_serializer; // Shared metadata serializer helpers (DRY across engines)
pub mod ops; // Common operations: compression, encoding, optimization
pub mod parquet_format_serializer; // Parquet format (VIPER) metadata serializer
pub mod pca; // PCA model management for spatial clustering
pub mod proximablocks_compact_serializer; // ProximaBlocks compact (SWIFT) metadata serializer
pub mod proximablocks_format_serializer; // ProximaBlocks format (HELIX) metadata serializer
pub mod read_strategy; // Unified read access strategy for all engines
pub mod search; // Search infrastructure: progressive search, filtering
pub mod sst_format_serializer; // SST (Sorted String Table) format metadata serializer // Columnar format (NOVA) metadata serializer

// Re-export commonly used types for convenience
pub use filter_evaluator::{
    UnifiedFilterEvaluator, create_filter_fn, create_json_filter_fn, evaluate_filter,
    evaluate_filter_strings,
};
pub use formats::{
    columnar::{ParquetIOLayer, ParquetQueryEngine},
    proximablocks::{ProximaBlockMetadata, ProximaDataBlock, RowBasedUtilities},
};
pub use io::zero_copy::ZeroCopyIOSystem;
pub use pca::{EnhancedPCAModel, PCAConfig, PCAManagerConfig, PCAModelManager};
pub use search::{ProgressiveSearchEngine, SearchContext};
// Common operations exports available from ops module directly
