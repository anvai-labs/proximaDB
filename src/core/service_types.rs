//! Service Types Module - Core Types for Vector Operations Service
//!
//! This module defines all the essential types for vector operations, including
//! collection operations, search requests/responses, service metrics,
//! and metrics. These types form the core API for the vector operations service.

use crate::core::search::OptimizedSearchRecord;
use crate::security::EncryptionConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// SearchResult is now only used from proto layer - not re-exported in core::search

/// Domain search hit (engine-agnostic)
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Vector record identifier
    pub id: String,
    /// Similarity score (lower is closer for L2, higher is closer for cosine)
    pub score: f32,
    /// The matched vector data
    pub vector: Vec<f32>,
    /// Metadata key-value pairs attached to the record
    pub metadata: HashMap<String, serde_json::Value>,
    /// MVCC version number, if versioning is enabled
    pub version: Option<i64>,
}

/// Domain search result set
#[derive(Debug, Clone)]
pub struct DomainSearchResult {
    /// Ordered list of search hits
    pub results: Vec<SearchHit>,
    /// Total number of matching records (may exceed results length)
    pub total_found: i64,
    /// Identifier of the collection that was searched
    pub collection_id: Option<String>,
}

/// Use proto-generated enums as single source of truth
// Note: Using string representations instead of proto enums for JSON serialization
// Proto enums don't derive Serialize/Deserialize by default
/// Distance metric name (e.g., "cosine", "l2", "dot_product")
pub type DistanceMetric = String;
/// Indexing algorithm name (e.g., "hnsw", "ivf", "flat")
pub type IndexingAlgorithm = String;
/// Storage engine name (e.g., "sst", "nova", "helix")
pub type StorageEngine = String;

// Re-export foundation compression algorithm type for backward compatibility
// TODO: Migrate all uses to FoundationCompressionAlgorithm (Phase 2.2)
pub use proximadb_compression_types::CompressionAlgorithm;

/// Compaction strategies for storage optimization
#[derive(Debug, Clone, Default)]
pub enum CompactionStrategy {
    /// Size-tiered compaction (default - good for write-heavy workloads)
    #[default]
    SizeTiered,
    /// Leveled compaction (better read performance)
    Leveled,
    /// Time-window compaction (optimal for time-series data)
    TimeWindow,
    /// No automatic compaction
    None,
}

/// Backwards-compat alias for [`ServiceCompactionConfig`].
pub type CompactionConfig = ServiceCompactionConfig;

/// Compaction configuration for storage engines
#[derive(Debug, Clone)]
pub struct ServiceCompactionConfig {
    /// Maximum SSTable file size in megabytes before splitting
    pub max_sstable_size_mb: u64,
    /// Maximum total size per level in megabytes
    pub max_level_size_mb: u64,
    /// Number of background compaction threads
    pub compaction_threads: u32,
    /// Whether background compaction is enabled
    pub enable_background_compaction: bool,
    /// Interval in seconds between compaction checks
    pub compaction_interval_seconds: u64,
}

impl Default for ServiceCompactionConfig {
    fn default() -> Self {
        Self {
            // strategy removed -  CompactionStrategy::SizeTiered,
            max_sstable_size_mb: 64,
            max_level_size_mb: 512,
            compaction_threads: 2,
            enable_background_compaction: true,
            compaction_interval_seconds: 300,
        }
    }
}

/// Security configuration for a collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionSecurityConfig {
    /// Enable Row-Level Security for this collection
    pub rls_enabled: bool,
    /// Names of RLS policies applied to this collection
    /// Policies are registered separately with the RLS service
    #[serde(default)]
    pub rls_policy_names: Vec<String>,
    /// Enable field-level encryption
    pub field_encryption_enabled: bool,
    /// Field encryption configuration
    #[serde(default)]
    pub encryption_config: EncryptionConfig,
    /// Enable audit logging for this collection
    pub audit_enabled: bool,
    /// Audit logging level
    #[serde(default)]
    pub audit_level: AuditLevel,
}

/// Audit logging level for collection operations
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AuditLevel {
    /// No audit logging
    #[default]
    None,
    /// Log read operations only
    Reads,
    /// Log write operations only
    Writes,
    /// Log all operations
    All,
}

/// Backwards-compat alias for [`ServiceCollectionConfig`] used by pre-disambiguation imports.
pub type CollectionConfig = ServiceCollectionConfig;

/// Collection configuration for CREATE and UPDATE operations
#[derive(Debug, Clone)]
pub struct ServiceCollectionConfig {
    /// Collection name
    pub name: String,
    /// Vector dimension (required)
    pub dimension: i32,
    /// Distance metric for similarity calculations
    pub distance_metric: DistanceMetric,
    /// Storage engine for persistence
    pub storage_engine: StorageEngine,
    /// Indexing algorithm for search optimization
    pub indexing_algorithm: IndexingAlgorithm,
    /// Metadata fields that can be filtered on
    pub filterable_metadata_fields: Vec<String>,
    /// Indexing configuration parameters
    pub indexing_config: HashMap<String, String>,
    /// New filterable columns API (replaces filterable_metadata_fields)
    pub filterable_columns: Vec<String>,
    /// Security configuration for the collection
    #[allow(dead_code)]
    pub security: Option<CollectionSecurityConfig>,
}

/// Vector operation response metrics
#[derive(Debug, Clone, Default)]
pub struct VectorOperationMetrics {
    /// Total number of vectors processed
    pub total_processed: i64,
    /// Number of successful operations
    pub successful_count: i64,
    /// Number of failed operations
    pub failed_count: i64,
    /// Number of updated vectors (for UPSERT)
    pub updated_count: i64,
    /// Total processing time in microseconds
    pub processing_time_us: i64,
    /// WAL write time in microseconds
    pub wal_write_time_us: i64,
    /// Index update time in microseconds
    pub index_update_time_us: i64,
}

/// Vector insert request for zero-copy operations
#[derive(Debug, Clone)]
pub struct VectorInsertRequest {
    /// Target collection identifier
    pub collection_id: String,
    /// Vector records to insert (supports single or batch)
    pub vectors: Vec<proximadb_records::ProximaRecord>,
    /// Update if vector ID already exists
    pub upsert_mode: bool,
    /// Optional batch identifier for tracking
    pub batch_id: Option<String>,
}

/// Vector operation response for INSERT operations
#[derive(Debug, Clone)]
pub struct VectorInsertResponse {
    /// Operation success status
    pub success: bool,
    /// Performance metrics
    pub metrics: VectorOperationMetrics,
    /// Generated or affected vector IDs
    pub vector_ids: Vec<String>,
    /// Error message if operation failed

    /// Error code if operation failed
    pub error_code: Option<String>,
}

impl VectorInsertRequest {
    /// Create a single vector insert request
    pub fn single_insert(
        collection_id: String,
        vector_record: proximadb_records::ProximaRecord,
    ) -> Self {
        Self {
            collection_id,
            vectors: vec![vector_record],
            upsert_mode: false,
            batch_id: None,
        }
    }

    /// Create a batch vector insert request
    pub fn batch_insert(
        collection_id: String,
        vectors: Vec<proximadb_records::ProximaRecord>,
    ) -> Self {
        Self {
            collection_id,
            vectors,
            upsert_mode: false,
            batch_id: None,
        }
    }

    /// Create an upsert request
    pub fn upsert(collection_id: String, vectors: Vec<proximadb_records::ProximaRecord>) -> Self {
        Self {
            collection_id,
            vectors,
            upsert_mode: true,
            batch_id: None,
        }
    }
}

impl VectorInsertResponse {
    /// Create a successful vector insert response
    pub fn success(metrics: VectorOperationMetrics, vector_ids: Vec<String>) -> Self {
        Self {
            success: true,
            metrics,
            vector_ids,
            // error_message removed -  None,
            error_code: None,
        }
    }

    /// Create a failed vector insert response
    pub fn error(error_code: Option<String>) -> Self {
        Self {
            success: false,
            metrics: VectorOperationMetrics::default(),
            vector_ids: Vec::new(),
            error_code,
        }
    }
}

/// Request for batch vector similarity search
#[derive(Debug, Clone)]
pub struct BatchSearchRequest {
    /// Target collection to search
    pub collection_id: String,
    /// Query vector for similarity matching
    pub query_vector: Vector,
    /// Number of nearest neighbors to return per query
    pub k: usize,
    /// Optional metadata filter predicates
    pub filter: Option<HashMap<String, serde_json::Value>>,
}

/// Search metadata for performance tracking
#[derive(Debug, Clone)]
pub struct SearchMetadata {
    /// Algorithm that was used (e.g., "hnsw", "flat")
    pub algorithm_used: String,
    /// Unique query identifier for correlation
    pub query_id: Option<String>,
    /// Estimated query complexity score
    pub query_complexity: f64,
    /// Total number of results found
    pub total_results: i64,
    /// Search execution time in milliseconds
    pub search_time_ms: f64,
    /// Suggested optimization hint for the caller
    pub performance_hint: Option<String>,
    /// Index-level performance statistics
    pub index_stats: Option<ServiceIndexStats>,
}

/// Index performance statistics
///
/// Per-query counters emitted by the search runtime. Anchors the planner v2
/// training inputs (see `SearchPlanTrace` for the per-query envelope that
/// wraps this) and any operator-side metering wrapper that consumes scan
/// statistics.
///
/// All new fields default to zero so unimplemented features can stub-emit
/// without changing the JSON shape consumed by downstream telemetry
/// (planner trainers, gateway scan-stats wrappers, etc.).
/// Backwards-compat alias for [`ServiceIndexStats`].
// Pre-extracted to foundation proximadb-data-model (Slice D).
pub use proximadb_data_model::{IndexStats, ServiceIndexStats};

/// Search debug information
#[derive(Debug, Clone)]
pub struct SearchDebugInfo {
    /// Ordered list of search execution steps
    pub search_steps: Vec<String>,
    /// IVF clusters or HNSW layers that were searched
    pub clusters_searched: Vec<String>,
    /// Whether metadata filter pushdown was enabled
    pub filter_pushdown_enabled: bool,
    /// Parquet columns that were scanned
    pub parquet_columns_scanned: Vec<String>,
    /// Per-phase timing breakdown in milliseconds
    pub timing_breakdown: std::collections::HashMap<String, f64>,
    /// Peak memory usage during search in megabytes
    pub memory_usage_mb: Option<f64>,
    /// Estimated total cost from the query planner
    pub estimated_total_cost: Option<f64>,
    /// Actual measured cost after execution
    pub actual_cost: Option<f64>,
    /// Per-component cost breakdown
    pub cost_breakdown: Option<std::collections::HashMap<String, f64>>,
}

/// Response from a vector similarity search
#[derive(Debug, Clone)]
pub struct VectorSearchResponse {
    /// Whether the search completed successfully
    pub success: bool,
    /// Ordered search result records
    pub results: Vec<OptimizedSearchRecord>,
    /// Number of results returned
    pub total_count: i64,
    /// Total matching records (may exceed returned count)
    pub total_found: i64,
    /// Processing time in microseconds
    pub processing_time_us: i64,
    /// Search algorithm that was used
    pub algorithm_used: String,
    /// Detailed search performance metadata
    pub search_metadata: SearchMetadata,
    /// Optional debug information for query analysis
    pub debug_info: Option<SearchDebugInfo>,
}

/// Collection operation types
#[derive(Debug, Clone)]
pub enum CollectionOperation {
    /// Create a new collection
    Create,
    /// Update an existing collection's configuration
    Update,
    /// Retrieve a collection by ID
    Get,
    /// List all collections
    List,
    /// Delete a collection
    Delete,
    /// Migrate a collection to a different engine
    Migrate,
}

/// Unified collection request - handles all collection operations
#[derive(Debug, Clone)]
pub struct CollectionRequest {
    /// Type of collection operation to perform
    pub operation: CollectionOperation,
    /// Collection identifier (required for all ops except CREATE and LIST)
    pub collection_id: Option<String>,
    /// Collection configuration (for CREATE and UPDATE operations)
    pub collection_config: Option<ServiceCollectionConfig>,
    /// Query parameters (limit, offset, filters, etc.)
    pub query_params: Option<HashMap<String, String>>,
    /// Operation options (force, include_stats, etc.)
    pub options: Option<HashMap<String, bool>>,
    /// Migration configuration for MIGRATE operations
    pub migration_config: Option<HashMap<String, String>>,
}

/// Unified collection response - handles all collection operation responses
#[derive(Debug, Clone)]
pub struct CollectionResponse {
    /// Operation success status
    pub success: bool,
    /// Type of operation that was performed
    pub operation: CollectionOperation,
    /// Number of affected items
    pub affected_count: i64,
    /// Total count for pagination
    pub total_count: Option<i64>,
    /// Operation metadata
    pub metadata: HashMap<String, String>,
    /// Error message if operation failed
    pub error_message: Option<String>,
    /// Error code if operation failed
    pub error_code: Option<String>,
    /// Processing time in microseconds
    pub processing_time_us: i64,
}
// Implementation blocks for new types
impl CollectionRequest {
    /// Create a new collection creation request
    pub fn create_collection(config: ServiceCollectionConfig) -> Self {
        Self {
            operation: CollectionOperation::Create,
            collection_id: None,
            collection_config: Some(config),
            query_params: None,
            options: None,
            migration_config: None,
        }
    }

    /// Create a collection retrieval request
    pub fn collection(collection_id: String) -> Self {
        Self {
            operation: CollectionOperation::Get,
            collection_id: Some(collection_id),
            collection_config: None,
            query_params: None,
            options: None,
            migration_config: None,
        }
    }

    /// Create a collection list request
    pub fn list_collections() -> Self {
        Self {
            operation: CollectionOperation::List,
            collection_id: None,
            collection_config: None,
            query_params: None,
            options: None,
            migration_config: None,
        }
    }

    /// Create a collection deletion request
    pub fn delete_collection(collection_id: String) -> Self {
        Self {
            operation: CollectionOperation::Delete,
            collection_id: Some(collection_id),
            collection_config: None,
            query_params: None,
            options: None,
            migration_config: None,
        }
    }
}

impl CollectionResponse {
    /// Create a successful collection response
    pub fn success(operation: CollectionOperation, processing_time_us: i64) -> Self {
        Self {
            success: true,
            operation,
            affected_count: 0,
            total_count: None,
            metadata: HashMap::new(),
            error_message: None,
            error_code: None,
            processing_time_us,
        }
    }

    /// Create a failed collection response
    pub fn error(
        operation: CollectionOperation,
        error_message: String,
        error_code: Option<String>,
        processing_time_us: i64,
    ) -> Self {
        Self {
            success: false,
            operation,
            affected_count: 0,
            total_count: None,
            metadata: HashMap::new(),
            error_message: Some(error_message),
            error_code,
            processing_time_us,
        }
    }
}

/// Unique vector record identifier
pub type VectorId = String;
/// Re-exported String type for compatibility
pub type String = std::string::String;
/// Cluster node identifier
pub type NodeId = String;
/// Dense floating-point vector
pub type Vector = Vec<f32>;

/// Metadata filter for server-side filtering operations
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Field-based filter with specific condition
    Field {
        /// Metadata field name to filter on
        field: String,
        /// Condition to evaluate against the field value
        condition: FieldCondition,
    },
    /// Logical AND of multiple filters
    And(Vec<MetadataFilter>),
    /// Logical OR of multiple filters
    Or(Vec<MetadataFilter>),
    /// Logical NOT of a filter
    Not(Box<MetadataFilter>),
}

/// Conditions for field-based filtering
#[derive(Debug, Clone)]
pub enum FieldCondition {
    /// Equal to value
    Equals(serde_json::Value),
    /// Not equal to value
    NotEquals(serde_json::Value),
    /// Greater than value
    GreaterThan(serde_json::Value),
    /// Less than value
    LessThan(serde_json::Value),
    /// Greater than or equal to value
    GreaterThanOrEqual(serde_json::Value),
    /// Less than or equal to value
    LessThanOrEqual(serde_json::Value),
    /// Value in list
    In(Vec<serde_json::Value>),
    /// Value not in list
    NotIn(Vec<serde_json::Value>),
    /// String contains substring
    Contains(String),
    /// String starts with prefix
    StartsWith(String),
    /// String ends with suffix
    EndsWith(String),
    /// Value is null
    IsNull,
    /// Value is not null
    IsNotNull,
    /// Range query
    Range {
        /// Inclusive lower bound
        min: serde_json::Value,
        /// Inclusive upper bound
        max: serde_json::Value,
    },
}

/// Vector operations for batch processing
//
// The `Insert` and `Search` variants are intentionally large (a full
// `ProximaRecord` / `SearchRequest`). Boxing them would force every
// caller — including tests and the storage/transaction layer — to
// allocate even on the hot insert path. Operations are typically held
// in `Vec<VectorOperation>` for batch execution; the per-variant size
// is amortised across the batch and matched against the heap-resident
// payloads they describe.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum VectorOperation {
    /// Insert a new vector
    Insert {
        /// Vector record to insert
        record: proximadb_records::ProximaRecord,
        /// Whether to update indexes immediately or defer
        index_immediately: bool,
    },
    /// Update an existing vector
    Update {
        /// ID of the vector to update
        vector_id: String,
        /// New vector data (None to keep existing)
        new_vector: Option<Vec<f32>>,
        /// New metadata (None to keep existing)
        new_metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Delete a vector
    Delete {
        /// ID of the vector to delete
        vector_id: String,
        /// Whether to soft-delete (mark as deleted) or hard-delete
        soft_delete: bool,
    },
    /// Search for similar vectors
    Search(SearchRequest),
    /// Get a vector by ID
    Get {
        /// ID of the vector to retrieve
        vector_id: String,
        /// Whether to include the full vector data
        include_vector: bool,
    },
    /// Batch operation
    Batch {
        /// List of operations to execute as a batch
        operations: Vec<VectorOperation>,
        /// Whether all operations must succeed or fail together
        transactional: bool,
    },
}

/// SearchRequest - External API representation of a vector search request.
///
/// This structure represents what clients send when requesting a search operation.
/// It's designed for API compatibility and ease of use from client SDKs.
///
/// # Purpose
/// - API contract for search requests
/// - Client SDK interface
/// - REST/gRPC request representation
///
/// # Usage
/// Used by API handlers to receive search requests from clients.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// Target collection to search
    pub collection_id: String,
    /// Query vector for similarity matching
    pub query_vector: Vec<f32>,
    /// Number of nearest neighbors to return
    pub k: usize,
    /// Optional metadata filters to apply
    pub filters: Option<Vec<MetadataFilter>>,
    /// Algorithm-specific hints (e.g., "ef_search" for HNSW)
    pub algorithm_hints: HashMap<String, String>,
    /// Minimum similarity threshold for returned results
    pub threshold: Option<f32>,
    /// Maximum search time in milliseconds
    pub timeout_ms: Option<u64>,
    /// Whether to include debug information in the response
    pub include_debug_info: bool,
    /// Whether to include vectors in the response
    pub include_vectors: bool,
}

/// Search strategy configuration
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exact search (brute force)
    Exact,
    /// Approximate search with configurable accuracy
    Approximate {
        /// Target recall accuracy (0.0 to 1.0)
        accuracy: f32,
    },
    /// Adaptive search based on query characteristics
    Adaptive {
        /// Estimated query complexity (higher = more complex)
        query_complexity_score: f32,
        /// Maximum allowed search time in milliseconds
        time_budget_ms: u64,
        /// Recall vs latency preference (0.0 = speed, 1.0 = accuracy)
        accuracy_preference: f32,
    },
}

/// Operation result enum
#[derive(Debug, Clone)]
pub enum OperationResult {
    /// Vector was inserted
    Inserted {
        /// ID of the newly inserted vector
        vector_id: String,
    },
    /// Vector was updated
    Updated {
        /// ID of the updated vector
        vector_id: String,
        /// Number of fields changed
        changes: i64,
    },
    /// Vector was deleted
    Deleted {
        /// ID of the deleted vector
        vector_id: String,
    },
    /// Search results
    SearchResults(Vec<OptimizedSearchRecord>),
    /// Vector data retrieved
    VectorData {
        /// ID of the retrieved vector
        vector_id: String,
        /// Vector data (if requested)
        vector: Option<Vec<f32>>,
        /// Associated metadata
        metadata: serde_json::Value,
    },
    /// Batch operation results
    BatchResults(Vec<OperationResult>),
    /// Operation error
    Error {
        /// Name of the failed operation
        operation: String,
        /// Error description
        error: String,
        /// Whether the operation can be retried
        recoverable: bool,
    },
}

// Vector insert response, operation metrics, and search response are already defined above

// Collection request and config are already defined above

// Search metadata, index stats, and debug info are already defined above

/// Metrics response structure for binary Avro serialization
#[derive(Debug, Clone)]
pub struct MetricsResponse {
    /// Service-level metrics
    pub service_metrics: ServiceMetrics,
    /// Write Buffer-specific metrics
    pub wal_metrics: WriteBufferMetrics,
    /// Timestamp when metrics were collected (microseconds)
    pub timestamp: i64,
}

/// Service-level performance metrics
#[derive(Debug, Clone)]
pub struct ServiceMetrics {
    /// Total operations performed
    pub total_operations: i64,
    /// Number of successful operations
    pub successful_operations: i64,
    /// Number of failed operations
    pub failed_operations: i64,
    /// Average processing time in microseconds
    pub avg_processing_time_us: f64,
    /// Last operation timestamp (microseconds)
    pub last_operation_time: Option<i64>,
}

/// Write Buffer-specific metrics
#[derive(Debug, Clone)]
pub struct WriteBufferMetrics {
    /// Total entries in WAL
    pub total_entries: i64,
    /// Entries currently in memory
    pub memory_entries: i64,
    /// Number of disk segments
    pub disk_segments: i64,
    /// Total disk size in bytes
    pub total_disk_size_bytes: i64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

/// Generic operation result for any database operation
#[derive(Debug, Clone)]
pub struct OperationResponse {
    /// Operation success status
    pub success: bool,
    /// Error message if operation failed

    /// Error code if operation failed
    pub error_code: Option<String>,
    /// Number of items affected by the operation
    pub affected_count: i64,
    /// Processing time in microseconds
    pub processing_time_us: i64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl OperationResponse {
    /// Create a successful operation response
    pub fn success(affected_count: i64, processing_time_us: i64) -> Self {
        Self {
            success: true,
            // error_message removed -  None,
            error_code: None,
            affected_count,
            processing_time_us,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed operation response
    pub fn error(
        // error_message removed -  String,
        error_code: Option<String>,
        processing_time_us: i64,
    ) -> Self {
        Self {
            success: false,
            // error_message removed -  Some(error_message),
            error_code,
            affected_count: 0,
            processing_time_us,
            metadata: HashMap::new(),
        }
    }
}

/// Backward-compatible alias for OptimizedSearchRecord
pub type UnifiedSearchResult = OptimizedSearchRecord;
/// Backward-compatible alias for OptimizedSearchRecord
pub type VectorSearchResult = OptimizedSearchRecord;
