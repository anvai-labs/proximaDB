//! # Storage Module - Persistence and Data Management Layer
//!
//! This module provides ProximaDB's sophisticated storage subsystem with multiple
//! storage engines, write-ahead logging, metadata management, and advanced caching.
//! It implements a layered architecture for durability, performance, and flexibility.
//!
//! ## Role in ProximaDB Architecture
//!
//! The storage layer manages all data persistence:
//! ```text
//! Services Layer
//!       ↓
//! ┌─────────────────────────────────────────────┐
//! │            Storage Subsystem                 │
//! ├─────────────────────────────────────────────┤
//! │  WAL → MemTable → Storage Engine → Disk     │
//! │   ↓       ↓           ↓             ↓       │
//! │ EventLog Cache   Compaction    Filesystem   │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Core Components
//!
//! ### 1. **Storage Engines** (`engines/`)
//! Six specialized engines for different workloads:
//! - **SST**: Hybrid columnar (ProximaBlocks) for OLTP, real-time queries
//! - **VIPER**: Columnar Parquet (deprecated, ADR-093; existing collections only)
//! - **NOVA**: Hybrid quantized columnar
//! - **SWIFT**: High-speed hierarchical blocks
//! - **PRISM**: Tree-based with Proxima
//! - **RAPTOR**: Matrix-optimized with adaptive PXK
//!
//! ### 2. **Write-Ahead Log** (`persistence/write_ahead_log/`)
//! Durability and recovery system:
//! - Append-only log for all operations
//! - Configurable sync modes (immediate, batch)
//! - Recovery from crashes
//! - Compaction coordination
//!
//! ### 3. **MemTable** (`memtable/`)
//! In-memory write buffer:
//! - Lock-free concurrent writes
//! - Automatic flushing to storage
//! - Multiple implementations (SkipList, BTree, ART)
//! - WAL integration for durability
//!
//! ### 4. **Metadata Store** (`metadata/`)
//! Collection and system metadata:
//! - Atomic metadata operations
//! - Cloud backend support (S3, Azure, GCS)
//! - Schema management
//! - Index configurations
//!
//! ### 5. **Cache System** (`cache/`)
//! Multi-level caching infrastructure:
//! - Query result cache
//! - Metadata cache
//! - Block cache for storage engines
//! - Adaptive eviction policies
//!
//! ### 6. **Transaction Coordinator** (`transaction_coordinator.rs`)
//! Atomic operations across components:
//! - Two-phase commit protocol
//! - Lock-free coordination with DashMap
//! - Rollback on failures
//! - Staging area management
//!
//! ## Storage Flow
//!
//! ### Write Path
//! ```text
//! Insert Request
//!       ↓
//! Write to WAL (durability)
//!       ↓
//! Insert to MemTable (fast access)
//!       ↓
//! Return to client
//!       ↓
//! Background: Flush to Storage Engine
//!       ↓
//! Background: Compaction
//! ```
//!
//! ### Read Path
//! ```text
//! Query Request
//!       ↓
//! Check Cache
//!       ↓
//! Search MemTable (recent data)
//!       ↓
//! Search Storage Engine (persistent data)
//!       ↓
//! Merge Results
//!       ↓
//! Update Cache
//! ```
//!
//! ## Key Features
//!
//! ### Strategy Pattern
//! All engines implement `UnifiedStorageFormat` trait:
//! ```rust,ignore
//! trait UnifiedStorageFormat {
//!     async fn insert(&self, records: Vec<VectorRecord>) -> Result<InsertResult>;
//!     async fn search(&self, query: SearchQuery) -> Result<SearchResult>;
//!     async fn flush(&self, params: FlushParameters) -> Result<FlushResult>;
//!     async fn compact(&self, params: CompactionParameters) -> Result<CompactionResult>;
//! }
//! ```
//!
//! ### Lock-Free Concurrency
//! Using DashMap for concurrent access:
//! - No global locks
//! - Sharded internal structure
//! - Wait-free reads
//! - Low contention writes
//!
//! ### Cloud-Native Storage
//! Filesystem abstraction supports:
//! - Local filesystem
//! - Amazon S3
//! - Azure Blob Storage
//! - Google Cloud Storage
//! - HDFS
//!
//! ## Configuration
//!
//! ```toml
//! [storage]
//! # The engine is chosen per collection at create time (default: "sst").
//! data_directory = "/data/proximadb"
//!
//! # WAL configuration
//! [storage.wal]
//! enabled = true
//! sync_mode = "batch"  # or "immediate"
//! batch_size = 100
//! flush_interval_ms = 1000
//!
//! # MemTable configuration
//! [storage.memtable]
//! type = "skiplist"  # or "btree", "art"
//! max_size_mb = 256
//! flush_threshold = 0.8
//!
//! # Compaction settings
//! [storage.compaction]
//! strategy = "leveled"  # or "tiered", "unified"
//! max_background_jobs = 2
//! target_file_size_mb = 128
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Write Latency**: < 1ms (memtable)
//! - **Flush Throughput**: 500MB/sec
//! - **Compaction Speed**: 200MB/sec
//! - **Recovery Time**: < 10s for 1GB WAL
//! - **Cache Hit Rate**: 80-95%
//!
//! ## Module Organization
//!
//! - **`builder.rs`**: Storage system builder pattern
//! - **`traits.rs`**: Core storage traits
//! - **`types.rs`**: Common type definitions  
//! - **`validation.rs`**: Configuration validation
//! - **`engine.rs`**: Main storage engine implementation
//! - **`optimization.rs`**: Storage optimization utilities
//! - **`strategy.rs`**: Collection lifecycle strategies
//! - **`common/`**: Shared utilities and helpers
//!
//! ## Error Handling
//!
//! Unified error types for storage operations:
//! - `StorageError::DiskIO` - I/O failures
//! - `StorageError::WAL` - Write-ahead log errors
//! - `StorageError::Corruption` - Data corruption detected
//! - `StorageError::OutOfSpace` - Disk space exhausted
//! - `StorageError::Configuration` - Invalid configuration

pub mod error;
pub mod scan_strategy;
pub mod trait_components;
pub mod traits;
pub mod types;

// Common reusable components
pub mod common;

// Background operation context (optimization)
pub mod background_flush_context;

// A6 storage-write fence seam (default-OFF; `PROXIMADB_WRITE_FENCING=1`)
pub mod write_fence;

// Shared per-collection flush materialization (server shutdown + embedded)
pub mod flush_materializer;

// ADR-069/TD-WAL-1: live auto-flush driver (policy-gated materialization).
pub mod auto_flush_driver;

// Engine capabilities and supportability checks
pub mod engine_capabilities;

// Core storage engines (organized)
pub mod engines;

// Data persistence layer (organized)
pub mod persistence;

// Encryption at rest for data security
// Extracted to the `proximadb-storage-encryption` crate; re-exported here for source
// compatibility (root-crate decomposition).
pub use proximadb_storage_encryption as encryption;

// EncryptedFilesystem decorator tests stayed in root because they drive the real
// root-resident LocalFileSystem backend (see the module's header for rationale).
#[cfg(test)]
mod encryption_filesystem_tests;

// Unified atomic operations
pub mod transaction_coordinator;

// Core modules
pub mod engine;
// Vector Object Economy Phase 6: per-collection pinning control surface
pub mod collection_pinning;
// Unified memtable system
pub mod memtable;
pub mod metadata;
// Quantization now handled by unified compute module
// Storage optimization utilities — extracted to proximadb-storage-optimization (decomp slice B)
pub use proximadb_storage_optimization as optimization;
// Strategy module for collection lifecycle configuration — extracted to proximadb-storage-strategy (decomp slice B)
pub use proximadb_storage_strategy as strategy;
// Specialized cache system with shared infrastructure
pub mod cache;

// Multi-tenant architecture modules — extracted to proximadb-storage-tenant (decomp slice B)
pub use proximadb_storage_tenant as tenant;

// Auto-tiering policy engine for data lifecycle management
pub mod tiering;

// Multi-model transaction coordinator for ACID transactions across stores
// Extracted to the `proximadb-storage-transaction` crate; re-exported here for
// source compatibility (root-crate decomposition).
pub use proximadb_storage_transaction as transaction;

// Semantic Knowledge Store (SKS) modules
pub mod entity_store;
pub mod relations;

// Key-value storage interface — extracted to the `proximadb-storage-kv` crate
// (root-crate decomposition Slice A); re-exported here for source compatibility
// (`crate::storage::kv::{StorageKV, FsKV}`).
pub use proximadb_storage_kv as kv;

// Unified operations coordination (flush, compaction, re-quantization)
// Extracted to the `proximadb-storage-operations` crate; re-exported here for source
// compatibility (root-crate decomposition).
pub use proximadb_storage_operations as operations;

// Document storage for MongoDB-like JSON document capabilities
pub mod document;

// Multi-model storage facade for unified access to all specialized stores
pub mod multimodel;

// Storage format abstraction layer for Hadoop-style storage-compute separation
pub mod formats;

// Arrow-native schema system for compute engine compatibility
pub mod schema;

/// Storage↔compute glue relocated from `src/compute/quantization` (quantization
/// kernel split, step Q2): code that needs the storage/core layers and so cannot
/// live in a foundation kernel crate.
pub mod compute_bridge;

// Lock-free implementations have been integrated into the main implementations
// TransactionCoordinator now uses DashMap for active_operations
// StorageEngine now uses DashMap for lsm_trees and mmap_readers

// Main exports from organized structure
pub use types::StorageEngineType;

// Strategy pattern exports
pub use traits::{
    CompactionParameters, CompactionResult, EngineHealth, EngineStatistics, FlushParameters,
    FlushResult as TraitFlushResult, StorageFormatStrategy, UnifiedStorageFormat,
};

// New decomposed trait hierarchy (ISP-compliant)
pub use trait_components::{
    StorageCompactor, StorageIdentity, StorageLifecycle, StorageMetrics, StorageReader,
    StorageScan, StorageWriter,
};

// Engine exports
pub use engines::sst::SstEngine;
// Arrow integration re-enabled - compilation conflicts resolved
// pub use engines::viper::ViperEngine;

// Persistence exports
pub use persistence::{DiskManager, FilesystemConfig, FilesystemFactory};

// Atomic operations exports
pub use transaction_coordinator::{
    StagingConfig, TransactionCoordinator, TransactionStageType, TransactionalOperationMetadata,
    TransactionalOperationStatus, ViperTransactionalOperations, WalTransactionalOperations,
};

// Storage engine exports
pub use engine::StorageEngine;
// Write Buffer system exports
use crate::core::StorageError;
pub use metadata::{MetadataStore, SystemMetadata};
pub use persistence::write_ahead_log::{BatchId, WALConfig, WALOperation, WriteAheadLogManager};

// Multi-model transaction exports
pub use transaction::{
    ConflictResolution, DocumentOperation, GraphOperation, IsolationLevel,
    MultiModelTransactionManager, ObservabilityOperation, OperationType, TransactionConfig,
    TransactionContext, TransactionOperation, VectorOperation,
};

// ResultProcessor has naming conflicts, import explicitly when needed

// Multi-model storage exports
pub use multimodel::{
    AggregationFunction,
    CardinalityConfig,
    CardinalityLimiter,
    CheckResult,
    CommitResult,
    DocumentStore,
    GraphStore,
    // Isolation management
    IsolationLevel as MultiModelIsolationLevel,
    IsolationManager,
    LabelStats,
    LimitAction,
    ModelType,
    MultiModelStorageEngine,
    MultiModelStorageFacade,
    ObservabilityStore,
    Partition,
    PartitionConfig,
    PartitionGranularity,
    PartitionRange,
    PrepareResult,
    QueryCharacteristics,
    RDBMSStore,
    ReadSnapshot,
    ReplicationConfig,
    // HTAP replication and routing
    ReplicationCoordinator,
    ReplicationStats,
    RollupConfig,
    RollupInterval,
    RollupManager,
    RollupView,
    RoutingDecision,
    StoreCapabilities,
    // Observability time series management
    TimePartitioner,
    Transaction as MultiModelTransaction,
    TransactionConfig as MultiModelTxConfig,
    // Transaction coordination (aliased to avoid conflict with transaction_coordinator.rs)
    TransactionCoordinator as MultiModelTransactionCoordinator,
    TransactionState,
    TransactionStats,
    TwoPhaseCommitProtocol,
    VectorStore,
    WorkloadRouter,
    WorkloadType,
    WriteSet,
};

pub type Result<T> = std::result::Result<T, StorageError>;

// Format abstraction layer exports
pub use formats::{
    CacheStatus,
    ColumnBounds as FormatColumnBounds,
    ColumnStats,
    CompactionContext as FormatCompactionContext,
    CompactionResult as FormatCompactionResult,
    ComparisonOp,
    CompressionCodec as FormatCompressionCodec,
    DefaultFormatDetector,
    FileEntry,
    // File splits for parallel reading (Arrow-Native FileFormat API)
    FileSplit,
    FileStats,
    FilterExpression,
    FormatDetector,
    // Registry
    FormatRegistry,
    FormatStatistics,
    // Types
    FormatType,
    InternalFormat,
    MergeAction,
    OpenTableFormat,
    OptimizeContext,
    OptimizeResult,
    ReadContext,
    RecordBatchStream,
    ScalarPredicate as FormatScalarPredicate,
    ScalarValue,
    Snapshot,
    SpatialBounds as FormatSpatialBounds,
    SplitCost,
    SplitGenerator,
    SplitLocality,
    SplitPlanner,
    SplitStatistics,
    SplitType,
    // Core traits
    StorageFormat,
    StorageTier,
    VectorBatch,
    VectorBatchStream,
    VectorReadContext,
    VectorWriteContext,
    WriteContext,
    WriteMode,
    WriteResult as FormatWriteResult,
    global_registry,
};

// Schema system exports (Arrow-native for compute engine compatibility)
pub use schema::{
    AutoGenerateType,
    AvroStyleField,
    AvroStyleSchema,
    AvroStyleType,
    // Bloom filter consolidation (WS1)
    BloomConsolidator,
    CachedHeader,
    CachingHeaderLoader,
    CentroidNode,
    // CentroidTree for O(log n) vector pruning (WS1)
    CentroidTree,
    CentroidTreeConfig,
    ColumnBounds,
    ColumnValue,
    ConsolidatedBloom,
    DefaultProximaRecordBridge,
    DefaultSchemaEvolution,
    DefaultValue,
    EncodingInfo,
    // Enhanced header cache with CentroidTree integration
    EnhancedCachedHeader,
    EvolutionValidation,
    HeaderCacheStats,
    HeaderLoader,
    HeaderLoaderRegistry,
    InMemorySchemaRegistry,
    IncrementalBloomBuilder,
    IoSavingsEstimate,
    MetadataMode,
    MigrationCost,
    MigrationPlan,
    MigrationStep,
    // Header loaders (bridges to existing readers)
    ParquetHeaderLoader,
    PersistentSchemaRegistry,
    ProximaBlocksHeaderLoader,
    ProximaColumn,
    // Header cache for smart I/O
    ProximaHeaderCache,
    // ProximaRecord bridge (WS5: ProximaRecord to Arrow RecordBatch conversion)
    ProximaRecordBridge,
    // Core schema types
    ProximaSchema,
    RowGroupMeta,
    ScalarPredicate,
    // Schema evolution
    SchemaEvolution,
    SchemaEvolutionOp,
    // Schema registry
    SchemaRegistry,
    SchemaVersionInfo,
    SharedCentroidTree,
    SharedConsolidatedBloom,
    SpatialRange,
    TimeUnit,
    TypeCompatibility,
    // Type mapping
    TypeMapper,
    global_header_cache,
    infer_schema_from_proxima_records,
    init_global_header_cache,
};

// Tests module
#[cfg(test)]
mod tests;
