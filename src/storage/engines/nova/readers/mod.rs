//! NOVA Parquet Reading Implementations
//!
//! NOVA uses standard Parquet format with embedded zone maps in metadata.
//! This module re-exports the UnifiedParquetReader from the columnar module
//! following the same pattern as VIPER.
//!
//! ## Architecture
//! - UnifiedParquetReader: Re-exported from columnar module
//! - Zone maps embedded in Parquet metadata for efficient pruning
//! - Progressive search with hierarchical zone map traversal
//! - Support for multi-resolution quantization at the column level

// Re-export from columnar module to maintain API compatibility
pub use crate::storage::engines::core::formats::columnar::{
    CollectionContext, FilterValue, MetadataFilter, ReaderConfig, ReadingStrategy,
    RowGroupAccessPattern, SchemaMapping, SearchType, SeekRange, Stage2Strategy,
    UnifiedParquetReader, VectorPosition,
};

// NOVA-specific re-exports
pub use super::zone_maps::PruningStrategy;
