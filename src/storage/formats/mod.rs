//! # Storage Formats Module
//!
//! This module provides the format abstraction layer for Hadoop-style storage-compute separation.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                     FORMAT ABSTRACTION LAYER                                 │
//! │  ┌───────────────────────────────────────┐  ┌─────────────────────────────┐ │
//! │  │         INTERNAL FORMATS              │  │    OPEN TABLE FORMATS       │ │
//! │  │  SST, Helix, Viper, Nova, Swift,     │  │  Delta Lake, Iceberg, Hudi, │ │
//! │  │  Raptor, Orion                       │  │  LanceDB, DuckDB, Parquet   │ │
//! │  └───────────────────────────────────────┘  └─────────────────────────────┘ │
//! │                          │                              │                    │
//! │                          └──────────────┬───────────────┘                    │
//! │                                         ▼                                    │
//! │                              ┌─────────────────────┐                         │
//! │                              │   Format Registry   │                         │
//! │                              │  Detection + Lookup │                         │
//! │                              └─────────────────────┘                         │
//! │                                         │                                    │
//! │                                         ▼                                    │
//! │                              ┌─────────────────────┐                         │
//! │                              │  Arrow RecordBatch  │                         │
//! │                              │   (Data Exchange)   │                         │
//! │                              └─────────────────────┘                         │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design Principles
//!
//! 1. **Storage-Compute Separation**: Storage formats provide ONLY serialization/deserialization.
//!    Compute operations (filtering, aggregation, joins) are handled by the compute layer.
//!
//! 2. **Arrow as Data Exchange**: All data transfer uses Arrow RecordBatch for zero-copy,
//!    cross-system compatibility.
//!
//! 3. **Extensibility**: New formats can be registered at runtime via FormatRegistry.
//!
//! 4. **Trait-Based Abstraction**: InternalFormat and OpenTableFormat traits provide clean
//!    interfaces for different format categories.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::storage::formats::{FormatRegistry, FormatType, ReadContext};
//!
//! // Get global registry
//! let registry = global_registry();
//!
//! // Auto-detect format from path
//! let format_type = registry.detect_format("/data/my_table").await?;
//!
//! // Get format implementation
//! if let Some(format) = registry.get_internal(FormatType::Sst) {
//!     let batches = format.read_batches(&ReadContext::default()).await?;
//! }
//! ```

// Core traits and types
pub mod traits;

// Format registry for discovery and lookup
pub mod registry;

// Arrow conversion utilities
pub mod arrow_conversion;

// Open table format implementations (Phase 2)
pub mod open;

// Internal format adapters (bridge to existing engines)
pub mod adapters;

// File split abstraction for parallel reading
pub mod splits;

// Re-exports for convenience
pub use traits::{
    ColumnStats,
    CompactionContext,
    CompactionResult,
    ComparisonOp,
    CompressionCodec,
    // Default implementations
    DefaultFormatDetector,
    FileEntry,
    FileStats,
    FilterExpression,
    FormatDetector,
    FormatStatistics,
    // Types
    FormatType,
    InternalFormat,
    MergeAction,
    OpenTableFormat,
    OptimizeContext,
    OptimizeResult,
    ReadContext,
    // Streams
    RecordBatchStream,
    Snapshot,
    // Core traits
    StorageFormat,
    VectorBatch,
    VectorBatchStream,
    VectorReadContext,
    VectorWriteContext,
    WriteContext,
    WriteMode,
    WriteResult,
};

pub use registry::{FormatRegistry, global_registry};

pub use arrow_conversion::{
    document_schema,
    filter_to_string,
    graph_edge_schema,
    graph_node_schema,
    json_to_sql_value,
    record_batch_to_vector_batch,
    record_batch_to_vector_records,
    sql_value_to_arrow_type,
    // Conversions
    vector_batch_to_record_batch,
    vector_records_to_record_batch,
    // Schema utilities
    vector_schema,
    vector_schema_flat,
};

pub use splits::{
    CacheStatus,
    ColumnBounds,
    // Core split types
    FileSplit,
    // Pruning types
    ScalarPredicate,
    ScalarValue,
    SpatialBounds,
    SplitCost,
    // Split generation
    SplitGenerator,
    // Locality and scheduling
    SplitLocality,
    SplitPlanner,
    SplitStatistics,
    SplitType,
    StorageTier,
};

// Re-export open table format implementations
pub use open::{
    DeltaLakeConfig,
    // Delta Lake
    DeltaLakeFormat,
    IcebergConfig,
    // Iceberg
    IcebergFormat,
    // Common types
    StorageOptions,
    TableMetadata,
};

// Re-export internal format adapters
pub use adapters::{
    HelixFormatAdapter,
    // Generic adapter
    InternalFormatAdapter,
    NovaFormatAdapter,
    // Engine-specific type aliases
    SstFormatAdapter,
    SwiftFormatAdapter,
    ViperFormatAdapter,
    create_helix_adapter,
    create_nova_adapter,
    // Factory functions
    create_sst_adapter,
    create_swift_adapter,
    create_viper_adapter,
};
// RAPTOR adapters are `experimental-engines`-gated (RAPTOR needs AXIS clustering).
#[cfg(feature = "experimental-engines")]
pub use adapters::{RaptorFormatAdapter, create_raptor_adapter};

// ============================================================================
// Module-Level Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_type_classification() {
        // Internal formats
        assert!(FormatRegistry::is_internal_format(FormatType::Sst));
        assert!(FormatRegistry::is_internal_format(FormatType::Helix));
        assert!(FormatRegistry::is_internal_format(FormatType::Viper));
        assert!(FormatRegistry::is_internal_format(FormatType::Nova));
        assert!(FormatRegistry::is_internal_format(FormatType::Swift));
        assert!(FormatRegistry::is_internal_format(FormatType::Raptor));
        assert!(FormatRegistry::is_internal_format(FormatType::Orion));

        // Open formats
        assert!(FormatRegistry::is_open_format(FormatType::DeltaLake));
        assert!(FormatRegistry::is_open_format(FormatType::Iceberg));
        assert!(FormatRegistry::is_open_format(FormatType::Hudi));
        assert!(FormatRegistry::is_open_format(FormatType::LanceDb));
        assert!(FormatRegistry::is_open_format(FormatType::DuckDb));
        assert!(FormatRegistry::is_open_format(FormatType::Parquet));
        assert!(FormatRegistry::is_open_format(FormatType::Avro));
    }

    #[test]
    fn test_registry_creation() {
        let registry = FormatRegistry::new();
        assert!(registry.list_all_formats().is_empty());

        let registry = FormatRegistry::with_defaults();
        // Registry created but no implementations registered yet
        assert!(registry.list_all_formats().is_empty());
    }
}
