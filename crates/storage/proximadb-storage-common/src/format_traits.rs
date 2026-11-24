//! # Format Abstraction Traits
//!
//! Core traits for storage format abstraction following Hadoop-style storage-compute separation.
//! Storage engines provide ONLY serialization/deserialization - compute is handled separately.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! StorageFormat (base)
//!       │
//!       ├── InternalFormat (SST, Helix, etc.)
//!       │   - Full control over format
//!       │   - Managed compaction
//!       │   - ProximaDB-native statistics
//!       │
//!       └── OpenTableFormat (Delta, Iceberg, Hudi, etc.)
//!           - Transaction log-based
//!           - ACID semantics
//!           - External ecosystem compatibility
//! ```

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use std::pin::Pin;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use serde::{Deserialize, Serialize};

// ============================================================================
// Core Types
// ============================================================================

/// Stream of Arrow RecordBatches for efficient data transfer
pub type RecordBatchStream = Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>;

/// Stream of vector batches (optimized for vector data)
pub type VectorBatchStream = Pin<Box<dyn Stream<Item = Result<VectorBatch>> + Send>>;

/// Vector batch with optional metadata
#[derive(Debug, Clone)]
pub struct VectorBatch {
    /// Vector IDs
    pub ids: Vec<String>,
    /// Raw vector data (flattened)
    pub vectors: Vec<f32>,
    /// Dimension of each vector
    pub dimension: usize,
    /// Optional metadata per vector
    pub metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
}

/// Context for read operations
#[derive(Debug, Clone)]
pub struct ReadContext {
    /// Path to read from
    pub path: String,
    /// Optional column projection
    pub projection: Option<Vec<String>>,
    /// Optional filter expression (for pushdown)
    pub filter: Option<FilterExpression>,
    /// Batch size for reading
    pub batch_size: usize,
    /// Enable parallel reading
    pub parallel: bool,
    /// Maximum parallelism
    pub max_parallelism: usize,
}

impl Default for ReadContext {
    fn default() -> Self {
        Self {
            path: String::new(),
            projection: None,
            filter: None,
            batch_size: 10000,
            parallel: true,
            max_parallelism: 4,
        }
    }
}

/// Context for vector-specific reads
#[derive(Debug, Clone)]
pub struct VectorReadContext {
    /// Base read context
    pub base: ReadContext,
    /// Return vectors (may be false for metadata-only queries)
    pub include_vectors: bool,
    /// Vector IDs to read (if None, read all)
    pub vector_ids: Option<Vec<String>>,
}

/// Context for write operations
#[derive(Debug, Clone)]
pub struct WriteContext {
    /// Path to write to
    pub path: String,
    /// Compression codec
    pub compression: CompressionCodec,
    /// Target file size (for splitting)
    pub target_file_size_bytes: u64,
    /// Write mode
    pub mode: WriteMode,
    /// Partitioning columns
    pub partition_by: Vec<String>,
}

impl Default for WriteContext {
    fn default() -> Self {
        Self {
            path: String::new(),
            compression: CompressionCodec::Lz4,
            target_file_size_bytes: 128 * 1024 * 1024, // 128MB
            mode: WriteMode::Append,
            partition_by: Vec::new(),
        }
    }
}

/// Context for vector-specific writes
#[derive(Debug, Clone)]
pub struct VectorWriteContext {
    /// Base write context
    pub base: WriteContext,
    /// Enable quantization
    pub quantize: bool,
    /// Quantization bits (if enabled)
    pub quantization_bits: Option<u8>,
}

/// Context for compaction operations
#[derive(Debug, Clone)]
pub struct CompactionContext {
    /// Input files to compact
    pub input_files: Vec<String>,
    /// Output directory
    pub output_dir: String,
    /// Target file size
    pub target_file_size_bytes: u64,
    /// Compression codec
    pub compression: CompressionCodec,
    /// Z-order columns for clustering
    pub clustering_columns: Vec<String>,
    /// Merge small files threshold
    pub small_file_threshold_bytes: u64,
}

/// Context for open table format optimization
#[derive(Debug, Clone)]
pub struct OptimizeContext {
    /// Enable Z-ordering
    pub z_order: bool,
    /// Z-order columns
    pub z_order_columns: Vec<String>,
    /// Target file size
    pub target_file_size_bytes: u64,
    /// Vacuum old files
    pub vacuum: bool,
    /// Retention period for vacuum
    pub retention_hours: u64,
}

/// Write result containing written file info
#[derive(Debug, Clone)]
pub struct WriteResult {
    /// Files written
    pub files_written: Vec<FileEntry>,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total records written
    pub records_written: u64,
    /// Write duration
    pub duration_ms: u64,
}

/// Compaction result
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Input files processed
    pub input_files: usize,
    /// Output files created
    pub output_files: usize,
    /// Bytes read
    pub bytes_read: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// Records processed
    pub records_processed: u64,
    /// Duration
    pub duration_ms: u64,
}

/// Optimization result for open table formats
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// Files optimized
    pub files_optimized: usize,
    /// Files vacuumed (deleted)
    pub files_vacuumed: usize,
    /// Space reclaimed
    pub space_reclaimed_bytes: u64,
    /// Duration
    pub duration_ms: u64,
}

/// File entry in a table/collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    /// File path (relative or absolute)
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Number of records
    pub record_count: u64,
    /// Optional partition values
    pub partition_values: Option<HashMap<String, String>>,
    /// File statistics
    pub stats: Option<FileStats>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Statistics for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStats {
    /// Column-level statistics
    pub column_stats: HashMap<String, ColumnStats>,
    /// Row count
    pub row_count: u64,
    /// Total size
    pub total_size_bytes: u64,
}

/// Column-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Minimum value (JSON encoded)
    pub min: Option<serde_json::Value>,
    /// Maximum value (JSON encoded)
    pub max: Option<serde_json::Value>,
    /// Null count
    pub null_count: u64,
    /// Distinct count (approximate)
    pub distinct_count: Option<u64>,
}

/// Format-level statistics for query planning
#[derive(Debug, Clone)]
pub struct FormatStatistics {
    /// Total row count
    pub row_count: u64,
    /// Total size in bytes
    pub size_bytes: u64,
    /// File count
    pub file_count: usize,
    /// Column statistics
    pub column_stats: HashMap<String, ColumnStats>,
    /// Schema
    pub schema: ArrowSchema,
}

/// Snapshot for open table formats (Delta/Iceberg style)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot/version ID
    pub version: i64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Files in this snapshot
    pub files: Vec<FileEntry>,
    /// Schema at this snapshot
    pub schema_string: String,
    /// Properties
    pub properties: HashMap<String, String>,
}

// Re-export foundation compression algorithm type for backward compatibility
// TODO: Migrate all uses to FoundationCompressionAlgorithm (Phase 2.2)
pub use proximadb_compression_types::CompressionAlgorithm as CompressionCodec;

/// Write mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Append to existing data
    Append,
    /// Overwrite existing data
    Overwrite,
    /// Error if data exists
    ErrorIfExists,
}

/// Filter expression for pushdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    /// Column comparison
    Comparison {
        column: String,
        op: ComparisonOp,
        value: serde_json::Value,
    },
    /// Logical AND
    And(Vec<FilterExpression>),
    /// Logical OR
    Or(Vec<FilterExpression>),
    /// Logical NOT
    Not(Box<FilterExpression>),
    /// Column is null
    IsNull { column: String },
    /// Column is not null
    IsNotNull { column: String },
    /// Column in list
    In {
        column: String,
        values: Vec<serde_json::Value>,
    },
}

/// Comparison operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Like,
}

/// Format type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormatType {
    // Internal formats
    Sst,
    Helix,
    Nova,
    Swift,
    Raptor,
    Orion,
    // Open table formats
    DeltaLake,
    Iceberg,
    Hudi,
    LanceDb,
    DuckDb,
    Parquet,
    Avro,
}

// ============================================================================
// Core Traits
// ============================================================================

/// Base trait for all storage formats (internal + open)
///
/// This trait defines the common interface that ALL storage formats must implement,
/// regardless of whether they are internal ProximaDB formats or external open formats.
#[async_trait]
pub trait StorageFormat: Send + Sync + Debug {
    /// Get the format name (e.g., "sst", "delta", "iceberg")
    fn format_name(&self) -> &str;

    /// Get the format version
    fn format_version(&self) -> &str;

    /// Get supported Arrow data types
    fn supported_data_types(&self) -> Vec<ArrowDataType>;

    /// Infer schema from a path
    async fn infer_schema(&self, path: &str) -> Result<ArrowSchema>;

    /// Validate a schema is compatible with this format
    fn validate_schema(&self, schema: &ArrowSchema) -> Result<()>;

    /// Get format type
    fn format_type(&self) -> FormatType;

    /// Check if format supports a specific feature
    fn supports_feature(&self, feature: &str) -> bool;
}

/// Internal formats with full control (SST, Helix, etc.)
///
/// Internal formats are fully managed by ProximaDB and provide:
/// - Direct control over file layout
/// - Custom compaction strategies
/// - ProximaDB-native statistics and bloom filters
/// - Optimized vector storage
#[async_trait]
pub trait InternalFormat: StorageFormat {
    // ========================================================================
    // Read Path - Returns Arrow RecordBatch
    // ========================================================================

    /// Read batches from storage as Arrow RecordBatches
    async fn read_batches(&self, ctx: &ReadContext) -> Result<RecordBatchStream>;

    /// Read vector data specifically
    async fn read_vectors(&self, ctx: &VectorReadContext) -> Result<VectorBatchStream>;

    /// Read a single vector by ID
    async fn read_vector_by_id(&self, path: &str, vector_id: &str) -> Result<Option<VectorBatch>>;

    // ========================================================================
    // Write Path - Accepts Arrow RecordBatch
    // ========================================================================

    /// Write a batch to storage
    async fn write_batch(&self, batch: &RecordBatch, ctx: &WriteContext) -> Result<WriteResult>;

    /// Write vector data specifically
    async fn write_vectors(
        &self,
        vectors: &VectorBatch,
        ctx: &VectorWriteContext,
    ) -> Result<WriteResult>;

    // ========================================================================
    // Compaction (internal formats manage their own compaction)
    // ========================================================================

    /// Perform compaction
    async fn compact(&self, ctx: &CompactionContext) -> Result<CompactionResult>;

    /// Check if compaction should be triggered
    fn should_compact(&self, stats: &FormatStatistics) -> bool;

    // ========================================================================
    // Statistics for Query Planning
    // ========================================================================

    /// Get statistics for query planning
    async fn get_statistics(&self, path: &str) -> Result<FormatStatistics>;

    /// Get bloom filter for a column (if available)
    async fn get_bloom_filter(&self, path: &str, column: &str) -> Result<Option<Vec<u8>>>;

    /// List all files in a path
    async fn list_files(&self, path: &str) -> Result<Vec<FileEntry>>;
}

/// Open table formats (Delta, Iceberg, Hudi) with transaction log
///
/// Open table formats provide:
/// - ACID transaction semantics
/// - Time travel (version/snapshot queries)
/// - Schema evolution
/// - External ecosystem compatibility
#[async_trait]
pub trait OpenTableFormat: StorageFormat {
    // ========================================================================
    // Transaction Log Operations
    // ========================================================================

    /// Get the current snapshot (latest version)
    async fn get_current_snapshot(&self, table_path: &str) -> Result<Snapshot>;

    /// Get snapshot at a specific version
    async fn get_snapshot_at(&self, table_path: &str, version: i64) -> Result<Snapshot>;

    /// List files in a snapshot
    async fn list_files(&self, snapshot: &Snapshot) -> Result<Vec<FileEntry>>;

    /// Get all available versions
    async fn list_versions(&self, table_path: &str) -> Result<Vec<i64>>;

    // ========================================================================
    // Read via Snapshot
    // ========================================================================

    /// Read from a specific snapshot
    async fn read_snapshot(
        &self,
        snapshot: &Snapshot,
        ctx: &ReadContext,
    ) -> Result<RecordBatchStream>;

    /// Read vectors from a snapshot (if format supports vectors)
    #[allow(unused_variables)]
    async fn read_snapshot_vectors(
        &self,
        snapshot: &Snapshot,
        ctx: &VectorReadContext,
    ) -> Result<Option<VectorBatchStream>> {
        // Default implementation returns None (format doesn't support vectors)
        Ok(None)
    }

    // ========================================================================
    // Write with ACID (creates new snapshot)
    // ========================================================================

    /// Write atomically, creating a new snapshot
    async fn write_atomic(
        &self,
        table_path: &str,
        batches: Vec<RecordBatch>,
        ctx: &WriteContext,
    ) -> Result<Snapshot>;

    /// Merge into existing data (upsert/update/delete)
    async fn merge_into(
        &self,
        table_path: &str,
        source: RecordBatchStream,
        merge_condition: &str,
        matched_action: MergeAction,
        not_matched_action: MergeAction,
    ) -> Result<Snapshot>;

    // ========================================================================
    // Time Travel
    // ========================================================================

    /// Get snapshot at a specific timestamp
    async fn time_travel(&self, table_path: &str, timestamp: DateTime<Utc>) -> Result<Snapshot>;

    /// Restore to a previous version
    async fn restore(&self, table_path: &str, version: i64) -> Result<Snapshot>;

    // ========================================================================
    // Optimization (coordinated by external scheduler)
    // ========================================================================

    /// Optimize table (Z-ordering, file compaction, etc.)
    async fn optimize(&self, table_path: &str, ctx: &OptimizeContext) -> Result<OptimizeResult>;

    /// Vacuum old files
    async fn vacuum(&self, table_path: &str, retention_hours: u64) -> Result<u64>;

    // ========================================================================
    // Schema Evolution
    // ========================================================================

    /// Get schema at a specific version
    async fn get_schema_at(&self, table_path: &str, version: i64) -> Result<ArrowSchema>;

    /// Evolve schema (add columns, etc.)
    async fn evolve_schema(&self, table_path: &str, new_schema: &ArrowSchema) -> Result<Snapshot>;
}

/// Merge action for MERGE INTO operations
#[derive(Debug, Clone)]
pub enum MergeAction {
    /// Update matched rows
    Update {
        assignments: HashMap<String, String>,
    },
    /// Delete matched rows
    Delete,
    /// Insert new rows
    Insert,
    /// Do nothing
    DoNothing,
}

// ============================================================================
// Format Detection
// ============================================================================

/// Trait for format detection
#[async_trait]
pub trait FormatDetector: Send + Sync {
    /// Detect format from path
    async fn detect(&self, path: &str) -> Result<Option<FormatType>>;

    /// Priority (higher = checked first)
    fn priority(&self) -> i32;
}

/// Default format detector based on file extensions and metadata
pub struct DefaultFormatDetector;

#[async_trait]
impl FormatDetector for DefaultFormatDetector {
    async fn detect(&self, path: &str) -> Result<Option<FormatType>> {
        let path = Path::new(path);

        // Check for open table format markers
        if path.join("_delta_log").exists() {
            return Ok(Some(FormatType::DeltaLake));
        }
        if path.join("metadata").exists() {
            // Could be Iceberg - check for metadata files
            if path.join("metadata/version-hint.text").exists() {
                return Ok(Some(FormatType::Iceberg));
            }
        }
        if path.join(".hoodie").exists() {
            return Ok(Some(FormatType::Hudi));
        }
        if path.join(".lance").exists() {
            return Ok(Some(FormatType::LanceDb));
        }

        // Check file extensions
        let ext = path.extension().and_then(|e| e.to_str());
        match ext {
            Some("parquet") => Ok(Some(FormatType::Parquet)),
            Some("avro") => Ok(Some(FormatType::Avro)),
            Some("arrow") | Some("ipc") => Ok(Some(FormatType::Sst)), // Arrow IPC -> SST
            Some("duckdb") | Some("db") => Ok(Some(FormatType::DuckDb)),
            _ => Ok(None),
        }
    }

    fn priority(&self) -> i32 {
        0 // Default priority
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_type_serialization() {
        let format = FormatType::DeltaLake;
        let json = serde_json::to_string(&format).unwrap();
        assert_eq!(json, "\"DeltaLake\"");

        let parsed: FormatType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, FormatType::DeltaLake);
    }

    #[test]
    fn test_compression_codec_default() {
        let codec = CompressionCodec::default();
        assert_eq!(codec, CompressionCodec::None);
    }

    #[test]
    fn test_read_context_default() {
        let ctx = ReadContext::default();
        assert_eq!(ctx.batch_size, 10000);
        assert!(ctx.parallel);
    }

    #[test]
    fn test_filter_expression() {
        let filter = FilterExpression::And(vec![
            FilterExpression::Comparison {
                column: "price".to_string(),
                op: ComparisonOp::Gt,
                value: serde_json::json!(100),
            },
            FilterExpression::IsNotNull {
                column: "name".to_string(),
            },
        ]);

        let json = serde_json::to_string(&filter).unwrap();
        let parsed: FilterExpression = serde_json::from_str(&json).unwrap();

        match parsed {
            FilterExpression::And(filters) => assert_eq!(filters.len(), 2),
            _ => panic!("Expected And filter"),
        }
    }
}
