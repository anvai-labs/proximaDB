//! # Schema Module - Arrow-Native Schema System
//!
//! This module provides ProximaDB's unified schema system designed for compute engine
//! compatibility with Spark, Trino, DataFusion, DuckDB, and PostgreSQL.
//!
//! ## Key Components
//!
//! - **ProximaSchema**: Arrow-based schema replacing VectorRecord
//! - **SchemaEvolution**: ADD/DROP/RENAME column operations
//! - **SchemaRegistry**: Schema versioning and management
//! - **TypeMapping**: ProximaDB ↔ Arrow ↔ Spark ↔ Trino conversions
//! - **ProximaRecordBridge**: Conversion between ProximaRecord and Arrow RecordBatch
//! - **ProximaHeaderCache**: Smart I/O layer for rowgroup/block pruning
//!
//! ## Design Philosophy
//!
//! Arrow-native approach (like LanceDB) for zero-copy data exchange with compute engines.
//! Full migration from VectorRecord-era DTOs to ProximaRecord and ProximaSchema.
//!
//! ## ProximaRecord Arrow Bridge
//!
//! The ProximaRecordBridge provides conversion between canonical ProximaRecord
//! envelopes and Arrow RecordBatches, enabling:
//! - Zero-copy conversion where possible
//! - Schema inference from ProximaRecord properties
//! - Avro-style schema serialization for schema registries
//! - Support for both JSON and structured metadata modes
//!
//! ## Smart I/O with ProximaHeaderCache
//!
//! ProximaHeaderCache caches file metadata for sub-millisecond pruning decisions BEFORE
//! issuing any S3/cloud I/O. This enables 80-95% I/O reduction for selective queries.

pub mod bloom_consolidator;
pub mod centroid_tree;
pub mod evolution;
pub mod header_cache;
pub mod header_loaders;
pub mod proxima_record_bridge;
pub mod proxima_schema;
pub mod pruning_strategies;
pub mod registry;
pub mod type_mapping;

// Re-exports
pub use evolution::{
    DefaultSchemaEvolution, EvolutionValidation, MigrationCost, MigrationPlan, MigrationStep,
    SchemaEvolution, SchemaEvolutionOp, TypeCompatibility,
};
pub use proxima_schema::{
    AutoGenerateType, AvroStyleField, AvroStyleSchema, AvroStyleType, DefaultValue, ProximaColumn,
    ProximaSchema, TimeUnit,
};
pub use registry::{
    InMemorySchemaRegistry, PersistentSchemaRegistry, SchemaRegistry, SchemaVersionInfo,
};
pub use type_mapping::TypeMapper;

// ProximaRecord bridge exports
pub use proxima_record_bridge::{
    DefaultProximaRecordBridge,
    MetadataMode,
    // Core trait and implementation
    ProximaRecordBridge,
    // Schema inference
    infer_schema_from_proxima_records,
};

// Header cache exports
pub use header_cache::{
    CachedHeader,
    CachingHeaderLoader,
    ColumnBounds,
    ColumnValue,
    // Encoding and stats
    EncodingInfo,
    HeaderCacheStats,
    // Header loading
    HeaderLoader,
    IoSavingsEstimate,
    // Core types
    ProximaHeaderCache,
    RowGroupMeta,
    ScalarPredicate,
    // Spatial pruning
    SpatialRange,
    // Global cache
    global_header_cache,
    init_global_header_cache,
};

// Header loader implementations (bridges to existing readers)
pub use header_loaders::{
    // Registry
    HeaderLoaderRegistry,
    // Parquet-based loaders (NOVA, RAPTOR)
    ParquetHeaderLoader,
    // ProximaBlocks-based loaders (SST, HELIX, SWIFT)
    ProximaBlocksHeaderLoader,
};

// Pruning strategies (SOLID principle: Interface Segregation)
pub use pruning_strategies::{
    BloomCheckResult,
    BloomChecker,
    // Composite pruner
    CompositePruner,
    NullBloomChecker,
    NullScalarPruner,
    // Null implementations (for testing and fallback)
    NullVectorPruner,
    // Result types
    PruningResult,
    PruningStats,
    ScalarPruner,
    SpatialPruner,
    SpatialRangeType,
    // Core traits
    VectorPruner,
};

// CentroidTree for O(log n) vector pruning
pub use centroid_tree::{CentroidNode, CentroidTree, CentroidTreeConfig, SharedCentroidTree};

// Bloom filter consolidation
pub use bloom_consolidator::{
    BloomConsolidator, ConsolidatedBloom, IncrementalBloomBuilder, SharedConsolidatedBloom,
};

// Enhanced header cache with CentroidTree integration
pub use header_cache::EnhancedCachedHeader;
