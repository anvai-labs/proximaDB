//! Parquet Reading Implementations
//!
//! ## Unified Architecture (Consolidated with Columnar Module)
//! - UnifiedParquetReader: Now re-exported from columnar module to avoid duplication
//! - All VIPER-specific reader functionality has been merged into columnar::UnifiedParquetReader
//! - Automatic optimization based on query characteristics and storage type
//! - Support for all query patterns: direct reading, metadata filtering, quantized search

// DEPRECATED: unified_parquet_reader.rs - functionality moved to columnar module
// pub mod unified_parquet_reader; // Deferred: Remove this file after migration complete

// Supporting modules
pub mod parquet_reconstructor;

// Note: readers_tests module flattened to tests/ in Phase 2
// Tests are now in src/storage/engines/viper/tests/

// Re-export from columnar module to maintain API compatibility
pub use crate::storage::engines::core::formats::columnar::{
    CollectionContext,
    FilterValue,
    QuantizationMethod,
    // All VIPER types now consolidated in columnar module
    ReaderConfig,
    ReadingStrategy,
    RowGroupAccessPattern,
    // Note: MetadataFilter is already in columnar module
    SchemaMapping,
    SearchType,
    SeekRange,
    Stage2Strategy,
    UnifiedParquetReader,
    VectorPosition,
};

// Supporting types
pub use parquet_reconstructor::{ParquetReconstructor, ReconstructedParquetData};
