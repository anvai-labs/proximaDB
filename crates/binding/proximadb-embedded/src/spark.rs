//! # Apache Spark DataSource V2 Connector
//!
//! Provides a JNI bridge for Apache Spark integration via the DataSource V2 API.
//! This enables Spark jobs to read from and write to ProximaDB collections natively.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Spark Driver                                  │
//! │  ┌─────────────────────┐    ┌─────────────────────┐                    │
//! │  │ ProximaDBScan       │    │ ProximaDBWrite      │                    │
//! │  │ (Scala/Java)        │    │ (Scala/Java)        │                    │
//! │  └─────────────────────┘    └─────────────────────┘                    │
//! │            │                          │                                 │
//! └────────────┼──────────────────────────┼─────────────────────────────────┘
//!              │ JNI                       │ JNI
//!              ▼                           ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Rust JNI Bridge                                  │
//! │  ┌─────────────────────┐    ┌─────────────────────┐                    │
//! │  │ SparkReadBridge     │    │ SparkWriteBridge    │                    │
//! │  └─────────────────────┘    └─────────────────────┘                    │
//! └─────────────────────────────────────────────────────────────────────────┘
//!              │                           │
//!              ▼                           ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    ProximaDB Storage Engines                            │
//! │         (SST, HELIX, SWIFT, NOVA, VIPER, RAPTOR)                       │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage in Spark
//!
//! ```scala
//! // Scala DataFrame API
//! val df = spark.read
//!   .format("proximadb")
//!   .option("host", "localhost:5678")
//!   .option("collection", "embeddings")
//!   .load()
//!
//! // Write back to ProximaDB
//! df.write
//!   .format("proximadb")
//!   .option("collection", "processed_embeddings")
//!   .mode("append")
//!   .save()
//!
//! // SQL with Vector Search
//! spark.sql("""
//!   SELECT * FROM proximadb.embeddings
//!   WHERE vector_search(embedding, array(0.1, 0.2, ...), 10) > 0.8
//! """)
//! ```
//!
//! ## Filter push-down (Rust-side foundation)
//!
//! Spark's pushed-down predicates arrive as a JSON array of [`SparkFilter`]
//! (the `filters_json` argument to [`spark_plan_input_partitions`]).
//! [`lower_spark_filters_to_expression`] lowers them into the canonical
//! [`FilterExpression`] tree; the planner stamps the result onto each
//! [`SparkInputPartition`], and the partition reader pushes it into the WAL
//! record scan, where it is evaluated against each record's **property tree**
//! (`record.props` — where relational/document table columns live) BEFORE the
//! page limit, so executors receive only matching rows.
//!
//! Contract: a filter is pushed only if it can be represented and evaluated
//! EXACTLY. Filters referencing nested struct-path columns, `AlwaysTrue/False`,
//! or an `And/Or/Not` with any unrepresentable child are **dropped** during
//! lowering (never partially applied). The (future) JVM DataSource V2 connector
//! must report dropped filters to Spark for re-application — until that JVM
//! layer exists, this path has no production driver (it is exercised by the
//! native unit/e2e tests). Example `filters_json`:
//! `[{"filter_type":"EqualTo","column":"category","value":"science","children":[]}]`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema as ArrowSchema;
use serde::{Deserialize, Serialize};

use proximadb::core::search::{ComparisonOperator, FilterExpression};
use proximadb::storage::formats::FileSplit;
use proximadb::storage::schema::ProximaSchema;

/// Configuration for Spark connector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparkConnectorConfig {
    /// ProximaDB server host
    pub host: String,
    /// ProximaDB server port
    pub port: u16,
    /// Authentication token (optional)
    pub auth_token: Option<String>,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Read timeout in milliseconds
    pub read_timeout_ms: u64,
    /// Maximum number of concurrent readers
    pub max_concurrent_readers: usize,
    /// Batch size for reading
    pub batch_size: usize,
    /// Enable filter pushdown
    pub enable_filter_pushdown: bool,
    /// Enable projection pushdown
    pub enable_projection_pushdown: bool,
    /// Enable aggregate pushdown
    pub enable_aggregate_pushdown: bool,
}

impl Default for SparkConnectorConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5678,
            auth_token: None,
            connection_timeout_ms: 30000,
            read_timeout_ms: 60000,
            max_concurrent_readers: 8,
            batch_size: 8192,
            enable_filter_pushdown: true,
            enable_projection_pushdown: true,
            enable_aggregate_pushdown: true,
        }
    }
}

/// Spark DataSource V2 Table representation
#[derive(Debug, Clone)]
pub struct SparkTable {
    /// Table name (collection name in ProximaDB)
    pub name: String,
    /// Arrow schema
    pub schema: Arc<ArrowSchema>,
    /// ProximaDB schema with additional metadata
    pub proxima_schema: Arc<ProximaSchema>,
    /// Table properties
    pub properties: HashMap<String, String>,
    /// Partition columns (if any)
    pub partition_columns: Vec<String>,
}

/// Spark InputPartition - represents a unit of work for a Spark task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparkInputPartition {
    /// Partition ID
    pub partition_id: i32,
    /// Underlying file splits
    pub splits: Vec<FileSplit>,
    /// Preferred host locations for locality
    pub preferred_locations: Vec<String>,
    /// Estimated rows in this partition
    pub estimated_rows: Option<i64>,
    /// Estimated bytes in this partition
    pub estimated_bytes: Option<i64>,
    /// Pushed-down filter (lowered from Spark's `filters_json`) to apply during
    /// the scan. Stamped by `spark_plan_input_partitions`; travels with the
    /// serialized partition to the executor's reader.
    #[serde(default)]
    pub filter: Option<FilterExpression>,
}

impl SparkInputPartition {
    /// Create a new input partition from file splits
    pub fn from_splits(partition_id: i32, splits: Vec<FileSplit>) -> Self {
        let estimated_rows = splits
            .iter()
            .filter_map(|s| s.statistics.row_count)
            .sum::<u64>();
        let estimated_bytes = splits
            .iter()
            .filter_map(|s| s.statistics.byte_size)
            .sum::<u64>();

        // Collect preferred hosts from all splits
        let mut preferred_locations: Vec<String> = splits
            .iter()
            .flat_map(|s| s.locality.preferred_hosts.clone())
            .collect();
        preferred_locations.dedup();

        Self {
            partition_id,
            splits,
            preferred_locations,
            estimated_rows: if estimated_rows > 0 {
                Some(estimated_rows as i64)
            } else {
                None
            },
            estimated_bytes: if estimated_bytes > 0 {
                Some(estimated_bytes as i64)
            } else {
                None
            },
            filter: None,
        }
    }
}

/// Spark ScanBuilder - builds scans with pushdown support
#[derive(Debug, Clone)]
pub struct SparkScanBuilder {
    /// Table to scan
    pub table: SparkTable,
    /// Column projection (None = all columns)
    pub projection: Option<Vec<String>>,
    /// Filter expression (serialized)
    pub filters: Vec<SparkFilter>,
    /// Limit (if any)
    pub limit: Option<i64>,
}

/// Spark filter expression (simplified for JNI transfer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparkFilter {
    /// Filter type
    pub filter_type: SparkFilterType,
    /// Column name (for column-based filters)
    pub column: Option<String>,
    /// Literal value (JSON encoded)
    pub value: Option<serde_json::Value>,
    /// Child filters (for AND/OR/NOT)
    pub children: Vec<SparkFilter>,
}

/// Spark filter types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SparkFilterType {
    /// Equality comparison (=)
    EqualTo,
    /// Inequality comparison (!=)
    NotEqualTo,
    /// Greater-than comparison (>)
    GreaterThan,
    /// Greater-than-or-equal comparison (>=)
    GreaterThanOrEqual,
    /// Less-than comparison (<)
    LessThan,
    /// Less-than-or-equal comparison (<=)
    LessThanOrEqual,
    /// String prefix match
    StringStartsWith,
    /// String suffix match
    StringEndsWith,
    /// String substring match
    StringContains,
    /// Check for null value
    IsNull,
    /// Check for non-null value
    IsNotNull,
    /// Logical AND of child filters
    And,
    /// Logical OR of child filters
    Or,
    /// Logical NOT of a child filter
    Not,
    /// Membership test (IN list)
    In,
    /// Always-true constant predicate
    AlwaysTrue,
    /// Always-false constant predicate
    AlwaysFalse,
}

/// Lower Spark's pushed-down filters (a JSON array of [`SparkFilter`]) into the
/// canonical [`FilterExpression`] boolean tree, or `None` when nothing can be
/// pushed.
///
/// Each TOP-LEVEL filter is lowered all-or-nothing: a filter that references an
/// unrepresentable shape — a nested struct-path column, `AlwaysTrue`/`AlwaysFalse`,
/// or an `And`/`Or`/`Not` whose subtree has any unrepresentable node — is
/// DROPPED, never partially applied. So what we push is always evaluated
/// exactly, and a (future) JVM DataSource V2 connector re-applies the dropped
/// filters on its side (standard best-effort push-down). The resulting
/// expression is evaluated against the record property tree (`record.props`).
pub fn lower_spark_filters_to_expression(
    filters_json: &str,
) -> Result<Option<FilterExpression>, SparkError> {
    let trimmed = filters_json.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let filters: Vec<SparkFilter> = serde_json::from_str(trimmed)
        .map_err(|e| SparkError::invalid_argument(format!("invalid filters_json: {e}")))?;
    let mut lowered: Vec<FilterExpression> =
        filters.iter().filter_map(try_lower_spark_filter).collect();
    Ok(match lowered.len() {
        0 => None,
        1 => lowered.pop(),
        // Spark's pushed filter array is an implicit conjunction.
        _ => Some(FilterExpression::And(lowered)),
    })
}

/// Recursively lower one [`SparkFilter`]; `None` if it (or any node in its
/// subtree) cannot be represented exactly.
fn try_lower_spark_filter(filter: &SparkFilter) -> Option<FilterExpression> {
    match filter.filter_type {
        SparkFilterType::EqualTo
        | SparkFilterType::NotEqualTo
        | SparkFilterType::GreaterThan
        | SparkFilterType::GreaterThanOrEqual
        | SparkFilterType::LessThan
        | SparkFilterType::LessThanOrEqual
        | SparkFilterType::StringStartsWith
        | SparkFilterType::StringEndsWith
        | SparkFilterType::StringContains
        | SparkFilterType::In => Some(FilterExpression::Comparison {
            field: simple_column(filter)?,
            operator: spark_op_to_comparison_op(filter.filter_type)?,
            value: filter.value.clone()?,
        }),
        SparkFilterType::IsNull | SparkFilterType::IsNotNull => {
            Some(FilterExpression::Comparison {
                field: simple_column(filter)?,
                operator: spark_op_to_comparison_op(filter.filter_type)?,
                value: serde_json::Value::Null,
            })
        }
        SparkFilterType::And => Some(FilterExpression::And(lower_children(filter)?)),
        SparkFilterType::Or => Some(FilterExpression::Or(lower_children(filter)?)),
        SparkFilterType::Not => Some(FilterExpression::Not(Box::new(try_lower_spark_filter(
            filter.children.first()?,
        )?))),
        // No literal-bool node in FilterExpression; drop and let Spark re-apply.
        SparkFilterType::AlwaysTrue | SparkFilterType::AlwaysFalse => None,
    }
}

/// A simple top-level column name — rejects empty and nested struct paths
/// (`a.b`), which can't be addressed as a single property-tree leaf here.
fn simple_column(filter: &SparkFilter) -> Option<String> {
    let column = filter.column.as_ref()?;
    if column.is_empty() || column.contains('.') {
        return None;
    }
    Some(column.clone())
}

/// Lower every child all-or-nothing (any unrepresentable child fails the whole
/// node). Empty child lists are rejected.
fn lower_children(filter: &SparkFilter) -> Option<Vec<FilterExpression>> {
    if filter.children.is_empty() {
        return None;
    }
    filter.children.iter().map(try_lower_spark_filter).collect()
}

fn spark_op_to_comparison_op(filter_type: SparkFilterType) -> Option<ComparisonOperator> {
    Some(match filter_type {
        SparkFilterType::EqualTo => ComparisonOperator::Equals,
        SparkFilterType::NotEqualTo => ComparisonOperator::NotEquals,
        SparkFilterType::GreaterThan => ComparisonOperator::GreaterThan,
        SparkFilterType::GreaterThanOrEqual => ComparisonOperator::GreaterThanOrEqual,
        SparkFilterType::LessThan => ComparisonOperator::LessThan,
        SparkFilterType::LessThanOrEqual => ComparisonOperator::LessThanOrEqual,
        SparkFilterType::In => ComparisonOperator::In,
        SparkFilterType::StringStartsWith => ComparisonOperator::StartsWith,
        SparkFilterType::StringEndsWith => ComparisonOperator::EndsWith,
        SparkFilterType::StringContains => ComparisonOperator::Contains,
        SparkFilterType::IsNull => ComparisonOperator::IsNull,
        SparkFilterType::IsNotNull => ComparisonOperator::IsNotNull,
        SparkFilterType::And
        | SparkFilterType::Or
        | SparkFilterType::Not
        | SparkFilterType::AlwaysTrue
        | SparkFilterType::AlwaysFalse => return None,
    })
}

impl SparkScanBuilder {
    /// Create a new scan builder for a table
    pub fn new(table: SparkTable) -> Self {
        Self {
            table,
            projection: None,
            filters: Vec::new(),
            limit: None,
        }
    }

    /// Add column projection
    pub fn with_projection(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }

    /// Add filter
    pub fn with_filter(mut self, filter: SparkFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add limit
    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = Some(limit);
        self
    }

    // `SparkScanBuilder::plan_partitions` removed (TD-097(3) B3): the
    // canonical partition planner now lives on
    // `EmbeddedProximaDB::plan_partitions` (see `src/embedded/mod.rs`),
    // which the JNI surface delegates to. Keeping a parallel placeholder
    // here would violate the Convergence Gate.
}

/// Spark PartitionReader — reads data from a single partition by
/// driving `EmbeddedProximaDB::scan_records` with cursor pagination
/// (TD-097(3) B3). The reader holds NO direct reference to the
/// embedded DB; the JNI wrapper and unit tests pass `&EmbeddedProximaDB`
/// per call (`spark_read_next_batch`). This keeps the struct
/// trivially `Send + Sync` and lets tests construct it without an
/// Arc/OnceLock dance.
#[derive(Debug)]
pub struct SparkPartitionReader {
    collection: String,
    cursor: Option<String>,
    finished: bool,
    batch_size: usize,
    records_read: u64,
    /// Pushed-down filter carried from the partition; applied per scan page.
    filter: Option<FilterExpression>,
}

impl SparkPartitionReader {
    /// Build a reader from a `SparkInputPartition`. The reader extracts
    /// the collection name from `partition.splits[0].file_path` — the
    /// single-partition fallback emitted by
    /// `EmbeddedProximaDB::plan_partitions` uses the `collection://<name>`
    /// scheme defined by [`FileSplit::whole_collection`].
    pub fn new(partition: SparkInputPartition, batch_size: usize) -> Result<Self, SparkError> {
        let collection = partition
            .splits
            .first()
            .and_then(|s| s.file_path.strip_prefix("collection://"))
            .ok_or_else(|| {
                SparkError::invalid_argument(
                    "partition must contain at least one split with a collection://… file_path",
                )
            })?
            .to_string();
        Ok(Self {
            collection,
            cursor: None,
            finished: false,
            batch_size,
            records_read: 0,
            filter: partition.filter,
        })
    }

    /// Collection name extracted from the partition's first split.
    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// Has the reader drained the partition?
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Cumulative records returned across all `spark_read_next_batch`
    /// calls.
    pub fn records_read(&self) -> u64 {
        self.records_read
    }

    /// Progress estimate (0.0 → 1.0). The single-partition fallback
    /// has no total rowcount up front, so progress is binary today
    /// (0.0 mid-scan / 1.0 finished).
    pub fn progress(&self) -> f64 {
        if self.finished { 1.0 } else { 0.0 }
    }
}

/// Spark WriteBuilder - builds write operations
#[derive(Debug, Clone)]
pub struct SparkWriteBuilder {
    /// Target table name
    pub table_name: String,
    /// Schema for data being written
    pub schema: Arc<ArrowSchema>,
    /// Write mode
    pub mode: SparkWriteMode,
    /// Partition columns
    pub partition_by: Vec<String>,
    /// Additional options
    pub options: HashMap<String, String>,
}

/// Spark write modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparkWriteMode {
    /// Append to existing data
    Append,
    /// Overwrite existing data
    Overwrite,
    /// Fail if data exists
    ErrorIfExists,
    /// Ignore if data exists
    Ignore,
}

impl SparkWriteBuilder {
    /// Create a new write builder
    pub fn new(table_name: String, schema: Arc<ArrowSchema>) -> Self {
        Self {
            table_name,
            schema,
            mode: SparkWriteMode::Append,
            partition_by: Vec::new(),
            options: HashMap::new(),
        }
    }

    /// Set write mode
    pub fn with_mode(mut self, mode: SparkWriteMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set partition columns
    pub fn with_partition_by(mut self, columns: Vec<String>) -> Self {
        self.partition_by = columns;
        self
    }

    /// Add option
    pub fn with_option(mut self, key: String, value: String) -> Self {
        self.options.insert(key, value);
        self
    }
}

/// Spark DataWriter — writes data from a single Spark task by
/// converting Arrow IPC batches to `ProximaRecord`s and inserting
/// via `EmbeddedProximaDB::insert_proxima_records`. Holds NO direct
/// reference to the embedded DB; the JNI wrapper and unit tests pass
/// `&EmbeddedProximaDB` per call (`spark_write_batch`).
#[derive(Debug)]
pub struct SparkDataWriter {
    collection: String,
    schema: Arc<ArrowSchema>,
    partition_id: i32,
    records_written: u64,
    bytes_written: u64,
}

impl SparkDataWriter {
    /// Build a writer for the named collection. `schema` is the Spark
    /// task's batch schema (Arrow); it's stored so future calls can
    /// validate the inbound IPC bytes match.
    pub fn new(collection: String, schema: Arc<ArrowSchema>, partition_id: i32) -> Self {
        Self {
            collection,
            schema,
            partition_id,
            records_written: 0,
            bytes_written: 0,
        }
    }

    /// Collection name the writer targets.
    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// Spark task partition ID (unique per concurrent writer).
    pub fn partition_id(&self) -> i32 {
        self.partition_id
    }

    /// Arrow schema captured when Spark created this writer.
    pub fn schema(&self) -> &ArrowSchema {
        self.schema.as_ref()
    }

    /// Cumulative records ingested across all `spark_write_batch` calls.
    pub fn records_written(&self) -> u64 {
        self.records_written
    }

    /// Cumulative inbound IPC bytes seen by the writer.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Build the commit message Spark expects back from each task.
    pub fn commit_message(&self) -> SparkWriteCommitMessage {
        SparkWriteCommitMessage {
            partition_id: self.partition_id,
            records_written: self.records_written as i64,
            bytes_written: self.bytes_written as i64,
            files_created: Vec::new(),
        }
    }
}

/// Write commit message returned by each task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparkWriteCommitMessage {
    /// Partition ID that completed
    pub partition_id: i32,
    /// Records written by this task
    pub records_written: i64,
    /// Bytes written by this task
    pub bytes_written: i64,
    /// Files created by this task
    pub files_created: Vec<String>,
}

/// Write error
#[derive(Debug)]
pub struct SparkWriteError {
    /// Human-readable error description
    pub message: String,
    /// Numeric error code
    pub code: i32,
}

impl std::fmt::Display for SparkWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SparkWriteError({}): {}", self.code, self.message)
    }
}

impl std::error::Error for SparkWriteError {}

// ============================================================================
// Spark embedded surface (TD-097(3) B3)
// ============================================================================
//
// Nine pure-Rust functions implementing the JNI-facing operations.
// Each takes `&EmbeddedProximaDB` so they're unit-testable in
// `cargo test` against a tmpdir fixture — NO JVM required.
//
// The JNI cdylib at `crates/binding/proximadb-spark-jni/src/lib.rs`
// fetches the embedded singleton (added in B4) and delegates to these
// `spark_*` functions. The legacy `jni_*` scaffolds (kept temporarily
// for ABI compat with the existing cdylib wrappers in B3) call into
// the new `spark_*` impls via a dummy "no embedded" path that returns
// the same placeholder values as before — they will be rewritten in
// B4 once the OnceLock singleton is in place.
//
// All four connectors now funnel into the canonical authority:
//   DuckDB/Hadoop → REST handlers → UnifiedHandlers → VectorOperationsService
//   Trino → Arrow Flight → UnifiedHandlers → VectorOperationsService
//   Spark JNI → EmbeddedProximaDB → SharedServices.request_handlers
//                                  → VectorOperationsService

/// Errors produced by the `spark_*` embedded surface. JNI wrappers map
/// these to typed Java exceptions; unit tests `assert!` on the variants.
#[derive(Debug, thiserror::Error)]
pub enum SparkError {
    /// Inbound JSON / partition shape was malformed or missed a
    /// required field. Maps to IllegalArgumentException in Java.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Write was rejected by `EmbeddedProximaDB::check_write_access`
    /// (SharedRead mode, follower role, etc.). Maps to
    /// SparkException(WRITE_REJECTED) in Java. **Never silently
    /// swallowed** — surfaces as a hard error per the silent-Ok-
    /// failure anti-pattern.
    #[error("write rejected: {0}")]
    WriteRejected(String),
    /// Arrow encode / decode failure on the wire format.
    #[error("arrow IPC error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// Wrapper for embedded-DB or other backend errors.
    #[error("embedded error: {0}")]
    Embedded(String),
}

impl SparkError {
    fn invalid_argument(msg: impl Into<String>) -> Self {
        SparkError::InvalidArgument(msg.into())
    }
    fn embedded<E: std::fmt::Display>(e: E) -> Self {
        SparkError::Embedded(e.to_string())
    }
}

/// Get the named collection's schema as a JSON string. Spark's
/// DataSource V2 `Table::schema()` consumes this. Returns a small
/// `{"error":"..."}` JSON envelope on failure so the Java side can
/// surface the message verbatim without a separate error channel.
pub fn spark_get_table_schema(emb: &crate::EmbeddedProximaDB, table: &str) -> String {
    match emb.get_collection_schema(table) {
        Ok(Some(json)) => json.to_string(),
        Ok(None) => format!(r#"{{"error":"collection '{table}' not found"}}"#),
        Err(e) => format!(r#"{{"error":"{e}"}}"#),
    }
}

/// Plan input partitions for parallel reads. Returns a JSON array of
/// [`SparkInputPartition`]; Java deserializes and hands one partition
/// per executor. Today this delegates to
/// [`EmbeddedProximaDB::plan_partitions`] (single-partition fallback —
/// real shard-aware planning is a follow-up TD).
pub fn spark_plan_input_partitions(
    emb: &crate::EmbeddedProximaDB,
    table: &str,
    filters_json: &str,
    num_partitions: i32,
) -> String {
    // Lower Spark's pushed filters once at plan time and stamp the (exactly
    // representable) expression onto every emitted partition, so each executor's
    // reader applies it during the scan. Malformed JSON => no push-down.
    let filter = lower_spark_filters_to_expression(filters_json)
        .ok()
        .flatten();
    match emb.plan_partitions(table, num_partitions.max(1) as u32) {
        Ok(mut partitions) => {
            for partition in &mut partitions {
                partition.filter = filter.clone();
            }
            serde_json::to_string(&partitions).unwrap_or_else(|_| "[]".to_string())
        }
        Err(_) => "[]".to_string(),
    }
}

/// Construct a [`SparkPartitionReader`] from a partition JSON blob
/// (typically the JSON Spark stored in `InputPartition`). The reader
/// itself holds no DB reference; subsequent `spark_read_next_batch`
/// calls take `&EmbeddedProximaDB` as a separate parameter so the
/// reader stays trivially `Send + Sync` (Spark serializes / scatters
/// reader handles across executors).
pub fn spark_create_partition_reader(
    partition_json: &str,
) -> Result<SparkPartitionReader, SparkError> {
    let partition: SparkInputPartition = serde_json::from_str(partition_json)
        .map_err(|e| SparkError::invalid_argument(format!("decode partition JSON: {e}")))?;
    SparkPartitionReader::new(partition, 1024)
}

/// Drive one page of `EmbeddedProximaDB::scan_records` and serialize
/// the result as Arrow IPC. Returns an empty `Vec<u8>` when the
/// partition is drained (mirrors Spark's `PartitionReader.next()`
/// returning `false`). Updates the reader's cursor + `finished` flag
/// so the next call resumes where this one left off.
pub fn spark_read_next_batch(
    emb: &crate::EmbeddedProximaDB,
    reader: &mut SparkPartitionReader,
) -> Result<Vec<u8>, SparkError> {
    if reader.finished {
        return Ok(Vec::new());
    }
    let (records, next_cursor) = emb
        .scan_records_filtered(
            reader.collection.clone().as_str(),
            reader.cursor.clone(),
            reader.batch_size,
            reader.filter.as_ref(),
        )
        .map_err(SparkError::embedded)?;

    if records.is_empty() {
        reader.finished = true;
        return Ok(Vec::new());
    }

    reader.records_read = reader.records_read.saturating_add(records.len() as u64);
    let next_is_some = next_cursor.is_some();
    reader.cursor = next_cursor;
    if !next_is_some {
        reader.finished = true;
    }

    let batch = proxima_records_to_record_batch(&records)?;
    record_batch_to_arrow_ipc(&batch).map_err(SparkError::from)
}

/// Drop the reader. Matches Spark's `PartitionReader.close()` ABI;
/// the JNI wrapper takes the boxed reader back via `Box::from_raw`
/// (in B4) which drops it as it leaves scope. Provided as a named fn
/// so the test surface mirrors the JNI shape.
pub fn spark_close_partition_reader(_reader: SparkPartitionReader) {
    // intentional: drop in the box destructor
}

/// Construct a [`SparkDataWriter`] for the named collection.
/// `schema_json` is the Spark batch schema (informational here; the
/// actual Arrow `RecordBatch` schema is taken from the inbound IPC
/// bytes on each `spark_write_batch` call). We perform a minimal
/// well-formedness check (parse as JSON) but do not yet validate
/// Spark↔Arrow type compatibility — that's a follow-up TD once the
/// Spark schema-spec is locked.
pub fn spark_create_data_writer(
    table: &str,
    schema_json: &str,
    partition_id: i32,
) -> Result<SparkDataWriter, SparkError> {
    // Minimal validity check: must be valid JSON. Don't try to deserialize
    // into an arrow::datatypes::Schema — arrow's Schema doesn't impl
    // serde::Deserialize and Spark's schema-JSON shape doesn't match
    // arrow's anyway.
    let _: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| SparkError::invalid_argument(format!("schema JSON: {e}")))?;
    // Empty arrow schema as placeholder; the real per-batch schema is
    // re-derived from the inbound IPC payload at write time.
    Ok(SparkDataWriter::new(
        table.to_string(),
        Arc::new(ArrowSchema::empty()),
        partition_id,
    ))
}

/// Decode the inbound Arrow IPC bytes into a `RecordBatch`, convert
/// each row to a `ProximaRecord`, and insert via
/// `EmbeddedProximaDB::insert_proxima_records`. Honors the access
/// mode (SharedRead / follower) by returning
/// `SparkError::WriteRejected` — NEVER a silent `Ok(())`.
pub fn spark_write_batch(
    emb: &crate::EmbeddedProximaDB,
    writer: &mut SparkDataWriter,
    arrow_bytes: &[u8],
) -> Result<(), SparkError> {
    if !emb.can_write() {
        return Err(SparkError::WriteRejected(format!(
            "embedded DB access mode = {:?}",
            emb.access_mode()
        )));
    }
    let batch = arrow_ipc_to_record_batch(arrow_bytes)?;
    let records = record_batch_to_proxima_records(&batch)?;
    let count = records.len();
    emb.insert_proxima_records(&writer.collection, records)
        .map_err(SparkError::embedded)?;
    writer.records_written = writer.records_written.saturating_add(count as u64);
    writer.bytes_written = writer
        .bytes_written
        .saturating_add(arrow_bytes.len() as u64);
    Ok(())
}

/// Flush the embedded DB so written data is durable, then return the
/// commit message as JSON (Spark consumes this as the task's
/// `WriterCommitMessage`).
pub fn spark_commit_writer(
    emb: &crate::EmbeddedProximaDB,
    writer: SparkDataWriter,
) -> Result<String, SparkError> {
    emb.flush().map_err(SparkError::embedded)?;
    let msg = writer.commit_message();
    serde_json::to_string(&msg)
        .map_err(|e| SparkError::embedded(format!("encode commit message: {e}")))
}

/// Drop the writer without flushing. Matches Spark's
/// `DataWriter.abort()` semantics for failed tasks.
pub fn spark_abort_writer(_writer: SparkDataWriter) {
    // intentional: drop in the box destructor
}

// ----------------------------------------------------------------------------
// Legacy `jni_*` ABI shims (TD-097(3) B3 transitional)
//
// The cdylib at `crates/binding/proximadb-spark-jni/src/lib.rs` imports
// these names. In B4 they will be rewritten to fetch the
// `OnceLock<Arc<EmbeddedProximaDB>>` singleton and delegate to the
// matching `spark_*` impl above. For B3 they preserve the old
// placeholder behavior so the cdylib + Java smoke harness keep
// compiling — the unit-test gate proves the `spark_*` implementations
// are correct in isolation.
// ----------------------------------------------------------------------------

#[doc(hidden)]
pub fn jni_get_table_schema(_table_name: &str) -> String {
    r#"{"type":"struct","fields":[]}"#.to_string()
}

#[doc(hidden)]
pub fn jni_plan_input_partitions(
    _table_name: &str,
    _filters_json: &str,
    _num_partitions: i32,
) -> String {
    "[]".to_string()
}

#[doc(hidden)]
pub fn jni_create_partition_reader(_partition_json: &str) -> i64 {
    0
}

#[doc(hidden)]
pub fn jni_read_next_batch(_reader_handle: i64) -> Vec<u8> {
    Vec::new()
}

#[doc(hidden)]
pub fn jni_close_partition_reader(_reader_handle: i64) {}

#[doc(hidden)]
pub fn jni_create_data_writer(_table_name: &str, _schema_json: &str, _partition_id: i32) -> i64 {
    0
}

#[doc(hidden)]
pub fn jni_write_batch(_writer_handle: i64, _arrow_batch: &[u8]) {}

#[doc(hidden)]
pub fn jni_commit_writer(_writer_handle: i64) -> String {
    r#"{"partition_id":0,"records_written":0,"bytes_written":0,"files_created":[]}"#.to_string()
}

#[doc(hidden)]
pub fn jni_abort_writer(_writer_handle: i64) {}

// ----------------------------------------------------------------------------
// Internal helpers: ProximaRecord <-> RecordBatch
// ----------------------------------------------------------------------------

/// Convert a slice of `ProximaRecord`s into a 2-column `RecordBatch`
/// (`id: Utf8`, `vector: List<Float32>`). This is the minimal schema
/// Spark's DataSource V2 needs to round-trip a scan; per-record `props`
/// flattening is a follow-up.
fn proxima_records_to_record_batch(
    records: &[proximadb_records::ProximaRecord],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    use arrow::array::{ListArray, StringArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field};

    let ids: Vec<&str> = records.iter().map(|r| r.oid.as_str()).collect();
    let id_arr = StringArray::from(ids);

    // Build a List<Float32> of variable length per record. Empty
    // vector = record had no embeddings.
    let mut flat: Vec<f32> = Vec::new();
    let mut offsets: Vec<i32> = vec![0];
    for r in records {
        let v: Vec<f32> = r
            .embeddings
            .first()
            .map(|c| c.values.to_fp32_owned())
            .unwrap_or_default();
        flat.extend_from_slice(&v);
        offsets.push(flat.len() as i32);
    }
    let values_arr = Arc::new(arrow::array::Float32Array::from(flat));
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_arr = ListArray::new(field, OffsetBuffer::new(offsets.into()), values_arr, None);

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(vector_arr)])
}

/// Inverse of [`proxima_records_to_record_batch`]: extract `id` +
/// `vector` columns and emit one `ProximaRecord` per row.
fn record_batch_to_proxima_records(
    batch: &RecordBatch,
) -> Result<Vec<proximadb_records::ProximaRecord>, SparkError> {
    use arrow::array::{Float32Array, ListArray, StringArray};
    use proximadb_records::{EmbeddingCell, EmbeddingScalarType, EmbeddingValues, ProximaRecord};

    let id_col = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SparkError::invalid_argument("batch missing Utf8 `id` column"))?;
    let vec_col = batch
        .column_by_name("vector")
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| {
            SparkError::invalid_argument("batch missing List<Float32> `vector` column")
        })?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let oid = id_col.value(row).to_string();
        let raw = vec_col.value(row);
        let vals = raw.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
            SparkError::invalid_argument("vector column inner type must be Float32")
        })?;
        let v: Vec<f32> = vals.values().to_vec();
        let dim = v.len() as u32;
        out.push(ProximaRecord {
            oid: oid.clone(),
            local_id: Some(oid),
            embeddings: vec![EmbeddingCell {
                model_id: "spark".to_string(),
                modality: "dense_vector".to_string(),
                dim,
                values: EmbeddingValues::Fp32(v),
                precision: EmbeddingScalarType::Fp32,
                ..Default::default()
            }],
            ..ProximaRecord::default()
        });
    }
    Ok(out)
}

// ============================================================================
// Arrow IPC helpers (TD-097 B2 — Spark DataSource V2 wire format)
//
// Spark DataSource V2 expects a single multi-column Arrow `RecordBatch`
// per `readNextBatch` / `writeBatch` call, encoded as plain Arrow IPC
// **stream** format (schema message + record-batch message). This is
// distinct from Trino's per-column block format
// (`record_batch_to_trino_page` in `src/connectors/trino.rs`) — DO NOT
// reuse the Trino helpers here.
// ============================================================================

/// Encode a multi-column `RecordBatch` as Arrow IPC stream bytes
/// (schema + batch). Used by `jni_read_next_batch` to ship a page of
/// records back to Java for Spark to consume.
pub fn record_batch_to_arrow_ipc(batch: &RecordBatch) -> Result<Vec<u8>, arrow::error::ArrowError> {
    use arrow::ipc::writer::StreamWriter;
    let schema = batch.schema();
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Decode Arrow IPC stream bytes into a single multi-column
/// `RecordBatch`. Used by `jni_write_batch` to consume a page Spark
/// hands across the JNI boundary. Returns the first batch in the
/// stream (Spark sends exactly one batch per call).
pub fn arrow_ipc_to_record_batch(bytes: &[u8]) -> Result<RecordBatch, arrow::error::ArrowError> {
    use arrow::error::ArrowError;
    use arrow::ipc::reader::StreamReader;
    let mut reader = StreamReader::try_new(bytes, None)?;

    reader
        .next()
        .ok_or_else(|| ArrowError::IpcError("arrow IPC stream contained no batches".to_string()))?
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spark_connector_config_default() {
        let config = SparkConnectorConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5678);
        assert!(config.enable_filter_pushdown);
    }

    #[test]
    fn test_spark_input_partition() {
        let partition = SparkInputPartition {
            partition_id: 0,
            splits: Vec::new(),
            preferred_locations: vec!["host1".to_string()],
            estimated_rows: Some(1000),
            estimated_bytes: Some(1024 * 1024),
            filter: None,
        };

        assert_eq!(partition.partition_id, 0);
        assert_eq!(partition.estimated_rows, Some(1000));
    }

    #[test]
    fn test_spark_filter() {
        let filter = SparkFilter {
            filter_type: SparkFilterType::EqualTo,
            column: Some("category".to_string()),
            value: Some(serde_json::json!("science")),
            children: Vec::new(),
        };

        assert_eq!(filter.filter_type, SparkFilterType::EqualTo);
        assert_eq!(filter.column, Some("category".to_string()));
    }

    #[test]
    fn lower_filters_empty_returns_none() {
        assert!(lower_spark_filters_to_expression("").unwrap().is_none());
        assert!(lower_spark_filters_to_expression("[]").unwrap().is_none());
    }

    #[test]
    fn lower_filters_malformed_returns_err() {
        let err = lower_spark_filters_to_expression("not json").unwrap_err();
        assert!(matches!(err, SparkError::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn lower_filters_equal_to() {
        let json =
            r#"[{"filter_type":"EqualTo","column":"category","value":"science","children":[]}]"#;
        let expr = lower_spark_filters_to_expression(json).unwrap().unwrap();
        assert_eq!(
            expr,
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::json!("science"),
            }
        );
    }

    #[test]
    fn lower_filters_comparison_and_string_operators_map() {
        for (ft, op) in [
            ("GreaterThan", ComparisonOperator::GreaterThan),
            ("LessThanOrEqual", ComparisonOperator::LessThanOrEqual),
            ("StringContains", ComparisonOperator::Contains),
            ("StringStartsWith", ComparisonOperator::StartsWith),
            ("NotEqualTo", ComparisonOperator::NotEquals),
        ] {
            let json =
                format!(r#"[{{"filter_type":"{ft}","column":"c","value":"v","children":[]}}]"#);
            let expr = lower_spark_filters_to_expression(&json).unwrap().unwrap();
            match expr {
                FilterExpression::Comparison { operator, .. } => assert_eq!(operator, op, "{ft}"),
                other => panic!("expected comparison for {ft}, got {other:?}"),
            }
        }
    }

    #[test]
    fn lower_filters_in_keeps_array_value() {
        let json = r#"[{"filter_type":"In","column":"c","value":["a","b"],"children":[]}]"#;
        let expr = lower_spark_filters_to_expression(json).unwrap().unwrap();
        assert_eq!(
            expr,
            FilterExpression::Comparison {
                field: "c".to_string(),
                operator: ComparisonOperator::In,
                value: serde_json::json!(["a", "b"]),
            }
        );
    }

    #[test]
    fn lower_filters_is_null_uses_null_value() {
        let json = r#"[{"filter_type":"IsNull","column":"c","value":null,"children":[]}]"#;
        let expr = lower_spark_filters_to_expression(json).unwrap().unwrap();
        assert!(matches!(
            expr,
            FilterExpression::Comparison {
                operator: ComparisonOperator::IsNull,
                ..
            }
        ));
    }

    #[test]
    fn lower_filters_and_or_not_nest() {
        let json = r#"[{"filter_type":"And","column":null,"value":null,"children":[
            {"filter_type":"EqualTo","column":"a","value":1,"children":[]},
            {"filter_type":"Or","column":null,"value":null,"children":[
                {"filter_type":"EqualTo","column":"b","value":2,"children":[]},
                {"filter_type":"Not","column":null,"value":null,"children":[
                    {"filter_type":"EqualTo","column":"c","value":3,"children":[]}
                ]}
            ]}
        ]}]"#;
        let expr = lower_spark_filters_to_expression(json).unwrap().unwrap();
        match expr {
            FilterExpression::And(children) => {
                assert_eq!(children.len(), 2);
                assert!(matches!(children[1], FilterExpression::Or(_)));
            }
            other => panic!("expected And, got {other:?}"),
        }
    }

    #[test]
    fn lower_filters_always_true_false_dropped() {
        let t = r#"[{"filter_type":"AlwaysTrue","column":null,"value":null,"children":[]}]"#;
        assert!(lower_spark_filters_to_expression(t).unwrap().is_none());
        let f = r#"[{"filter_type":"AlwaysFalse","column":null,"value":null,"children":[]}]"#;
        assert!(lower_spark_filters_to_expression(f).unwrap().is_none());
    }

    #[test]
    fn lower_filters_struct_path_column_dropped() {
        let json = r#"[{"filter_type":"EqualTo","column":"user.email","value":"x","children":[]}]"#;
        assert!(lower_spark_filters_to_expression(json).unwrap().is_none());
    }

    #[test]
    fn lower_filters_partial_top_level_drops_only_unsupported() {
        // One supported + one unsupported (struct path): keep only the supported.
        let json = r#"[
            {"filter_type":"EqualTo","column":"category","value":"a","children":[]},
            {"filter_type":"EqualTo","column":"user.email","value":"x","children":[]}
        ]"#;
        let expr = lower_spark_filters_to_expression(json).unwrap().unwrap();
        assert_eq!(
            expr,
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::json!("a"),
            }
        );
    }

    #[test]
    fn lower_filters_multiple_top_level_anded() {
        let json = r#"[
            {"filter_type":"EqualTo","column":"a","value":1,"children":[]},
            {"filter_type":"GreaterThan","column":"b","value":2,"children":[]}
        ]"#;
        let expr = lower_spark_filters_to_expression(json).unwrap().unwrap();
        match expr {
            FilterExpression::And(children) => assert_eq!(children.len(), 2),
            other => panic!("expected And, got {other:?}"),
        }
    }

    #[test]
    fn lower_filters_and_with_unrepresentable_child_drops_whole_node() {
        // An And containing a struct-path leaf can't be partially pushed → drop it all.
        let json = r#"[{"filter_type":"And","column":null,"value":null,"children":[
            {"filter_type":"EqualTo","column":"a","value":1,"children":[]},
            {"filter_type":"EqualTo","column":"x.y","value":2,"children":[]}
        ]}]"#;
        assert!(lower_spark_filters_to_expression(json).unwrap().is_none());
    }

    #[test]
    fn test_spark_write_mode() {
        let builder =
            SparkWriteBuilder::new("test_table".to_string(), Arc::new(ArrowSchema::empty()))
                .with_mode(SparkWriteMode::Overwrite);

        assert_eq!(builder.mode, SparkWriteMode::Overwrite);
    }

    #[test]
    fn test_record_batch_arrow_ipc_round_trip() {
        use arrow::array::{Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field};

        // 2-column batch (Int64, Utf8) with 3 rows including a null.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let original = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();

        let ipc = record_batch_to_arrow_ipc(&original).expect("encode");
        assert!(!ipc.is_empty(), "encoded IPC must be non-empty");

        let decoded = arrow_ipc_to_record_batch(&ipc).expect("decode");
        assert_eq!(decoded.num_columns(), 2);
        assert_eq!(decoded.num_rows(), 3);
        assert_eq!(decoded.schema().field(0).name(), "id");
        assert_eq!(decoded.schema().field(1).name(), "name");
        // Verify the null in column 1 round-trips.
        let name_col = decoded
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Utf8");
        assert!(name_col.is_null(1), "null at row 1 must survive round trip");
        assert_eq!(name_col.value(0), "a");
        assert_eq!(name_col.value(2), "c");
    }

    #[test]
    fn test_arrow_ipc_to_record_batch_empty_stream_errors() {
        // Stream with no batches → ArrowError (not silent None).
        let bytes: Vec<u8> = Vec::new();
        assert!(arrow_ipc_to_record_batch(&bytes).is_err());
    }

    // ========================================================================
    // TD-097 (3) B3 — embedded `spark_*` impls (no JVM)
    // ========================================================================

    use crate::{EmbeddedConfig, EmbeddedProximaDB};
    use proximadb_records::{EmbeddingCell, EmbeddingScalarType, EmbeddingValues, ProximaRecord};

    fn build_spark_test_db() -> (EmbeddedProximaDB, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let mut config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
        config.enable_wal = true;
        let db = EmbeddedProximaDB::new(config).expect("embedded db");
        (db, temp_dir)
    }

    fn make_spark_record(oid: &str) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            local_id: Some(oid.to_string()),
            embeddings: vec![EmbeddingCell {
                model_id: "spark-test".to_string(),
                modality: "dense_vector".to_string(),
                dim: 4,
                values: EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0, 4.0]),
                precision: EmbeddingScalarType::Fp32,
                ..Default::default()
            }],
            ..ProximaRecord::default()
        }
    }

    #[test]
    fn test_spark_get_table_schema_returns_collection_shape() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_sch_col", 4, None)
            .expect("create");
        let json = spark_get_table_schema(&db, "spark_sch_col");
        assert!(
            json.contains("\"name\""),
            "schema must include name: {json}"
        );
        assert!(
            json.contains("spark_sch_col"),
            "schema must include collection name: {json}"
        );
    }

    #[test]
    fn test_spark_get_table_schema_missing_collection_returns_error_envelope() {
        let (db, _td) = build_spark_test_db();
        let json = spark_get_table_schema(&db, "no_such_col");
        assert!(
            json.contains("\"error\""),
            "must wrap error in JSON: {json}"
        );
        assert!(
            json.contains("no_such_col"),
            "error must surface the bad collection name: {json}"
        );
    }

    #[test]
    fn test_spark_plan_input_partitions_returns_single_partition_json() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_part_col", 4, None)
            .expect("create");
        let json = spark_plan_input_partitions(&db, "spark_part_col", "{}", 8);
        let parsed: Vec<SparkInputPartition> = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.len(), 1, "single-partition fallback: {parsed:?}");
        assert_eq!(parsed[0].partition_id, 0);
        assert_eq!(parsed[0].splits.len(), 1);
        assert_eq!(parsed[0].splits[0].file_path, "collection://spark_part_col");
    }

    #[test]
    fn test_spark_create_partition_reader_parses_collection_uri() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_rdr_col", 4, None)
            .expect("create");
        let json = spark_plan_input_partitions(&db, "spark_rdr_col", "{}", 1);
        let partitions: Vec<SparkInputPartition> = serde_json::from_str(&json).unwrap();
        let part_json = serde_json::to_string(&partitions[0]).unwrap();
        let reader = spark_create_partition_reader(&part_json).expect("create reader");
        assert_eq!(reader.collection(), "spark_rdr_col");
        assert!(!reader.is_finished());
        assert_eq!(reader.records_read(), 0);
    }

    #[test]
    fn test_spark_create_partition_reader_rejects_malformed_split() {
        let bad = SparkInputPartition::from_splits(
            0,
            vec![proximadb::storage::formats::FileSplit {
                split_id: "bad".into(),
                file_path: "file:///not/a/collection".into(),
                offset: 0,
                length: 0,
                split_type: proximadb::storage::formats::SplitType::ByteRange {
                    estimated_records: 0,
                },
                statistics: Default::default(),
                locality: Default::default(),
                object_size: None,
            }],
        );
        let json = serde_json::to_string(&bad).unwrap();
        let err = spark_create_partition_reader(&json).expect_err("must error");
        assert!(
            matches!(err, SparkError::InvalidArgument(_)),
            "expected InvalidArgument: {err}"
        );
    }

    #[test]
    fn test_spark_read_next_batch_returns_empty_when_collection_empty() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("empty_spark_col", 4, None)
            .expect("create");
        let json = spark_plan_input_partitions(&db, "empty_spark_col", "{}", 1);
        let partitions: Vec<SparkInputPartition> = serde_json::from_str(&json).unwrap();
        let mut reader =
            spark_create_partition_reader(&serde_json::to_string(&partitions[0]).unwrap())
                .expect("reader");
        let bytes = spark_read_next_batch(&db, &mut reader).expect("read");
        assert!(bytes.is_empty(), "empty collection ⇒ empty IPC");
        assert!(reader.is_finished());
    }

    #[test]
    fn test_spark_read_next_batch_round_trips_inserted_records() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_rt_col", 4, None)
            .expect("create");
        let recs = (0..3)
            .map(|i| make_spark_record(&format!("spark_rec_{i}")))
            .collect();
        db.insert_proxima_records("spark_rt_col", recs)
            .expect("insert");

        let json = spark_plan_input_partitions(&db, "spark_rt_col", "{}", 1);
        let partitions: Vec<SparkInputPartition> = serde_json::from_str(&json).unwrap();
        let mut reader =
            spark_create_partition_reader(&serde_json::to_string(&partitions[0]).unwrap())
                .expect("reader");

        let bytes = spark_read_next_batch(&db, &mut reader).expect("read");
        assert!(!bytes.is_empty(), "non-empty collection ⇒ non-empty IPC");
        let batch = arrow_ipc_to_record_batch(&bytes).expect("decode");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).name(), "id");
        assert_eq!(batch.schema().field(1).name(), "vector");
        assert_eq!(reader.records_read(), 3);
        assert!(reader.is_finished(), "single page ⇒ finished");
    }

    fn make_spark_record_with_category(oid: &str, category: &str) -> ProximaRecord {
        let mut record = make_spark_record(oid);
        record.props.insert(
            "category".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_records::ProximaValue::String(
                category.to_string(),
            )),
        );
        record
    }

    /// Drain a reader fully and return the total Arrow rows produced.
    fn drain_reader(db: &EmbeddedProximaDB, reader: &mut SparkPartitionReader) -> usize {
        let mut total = 0;
        loop {
            let bytes = spark_read_next_batch(db, reader).expect("read batch");
            if bytes.is_empty() {
                break;
            }
            total += arrow_ipc_to_record_batch(&bytes)
                .expect("decode")
                .num_rows();
        }
        total
    }

    fn plan_and_read(
        db: &EmbeddedProximaDB,
        collection: &str,
        filters_json: &str,
    ) -> (Option<FilterExpression>, usize) {
        let json = spark_plan_input_partitions(db, collection, filters_json, 1);
        let partitions: Vec<SparkInputPartition> = serde_json::from_str(&json).unwrap();
        let filter = partitions[0].filter.clone();
        let mut reader =
            spark_create_partition_reader(&serde_json::to_string(&partitions[0]).unwrap())
                .expect("reader");
        (filter, drain_reader(db, &mut reader))
    }

    #[test]
    fn spark_filter_pushdown_predicate_eq() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_eq_col", 4, None)
            .expect("create");
        let recs = (0..10)
            .map(|i| {
                make_spark_record_with_category(
                    &format!("rec_{i}"),
                    if i % 2 == 0 { "a" } else { "b" },
                )
            })
            .collect();
        db.insert_proxima_records("spark_eq_col", recs)
            .expect("insert");

        let filters =
            r#"[{"filter_type":"EqualTo","column":"category","value":"a","children":[]}]"#;
        let (filter, total) = plan_and_read(&db, "spark_eq_col", filters);
        assert!(
            filter.is_some(),
            "filter must be stamped onto the partition"
        );
        assert_eq!(
            total, 5,
            "only the 5 category='a' records pass the pushed filter"
        );
    }

    #[test]
    fn spark_filter_pushdown_in_set() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_in_col", 4, None)
            .expect("create");
        let recs = (0..12)
            .map(|i| {
                let cat = ["a", "b", "c"][i % 3];
                make_spark_record_with_category(&format!("rec_{i}"), cat)
            })
            .collect();
        db.insert_proxima_records("spark_in_col", recs)
            .expect("insert");

        let filters =
            r#"[{"filter_type":"In","column":"category","value":["a","b"],"children":[]}]"#;
        let (_filter, total) = plan_and_read(&db, "spark_in_col", filters);
        assert_eq!(
            total, 8,
            "category IN ('a','b') matches 8 of 12 (4 'c' excluded)"
        );
    }

    #[test]
    fn spark_filter_pushdown_no_match_returns_empty() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_nomatch_col", 4, None)
            .expect("create");
        let recs = (0..6)
            .map(|i| make_spark_record_with_category(&format!("rec_{i}"), "a"))
            .collect();
        db.insert_proxima_records("spark_nomatch_col", recs)
            .expect("insert");

        let filters =
            r#"[{"filter_type":"EqualTo","column":"category","value":"zzz","children":[]}]"#;
        let (_filter, total) = plan_and_read(&db, "spark_nomatch_col", filters);
        assert_eq!(total, 0, "no record matches ⇒ scan returns nothing");
    }

    #[test]
    fn spark_filter_pushdown_malformed_filter_drains_all() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_allf_col", 4, None)
            .expect("create");
        let recs = (0..4)
            .map(|i| make_spark_record_with_category(&format!("rec_{i}"), "a"))
            .collect();
        db.insert_proxima_records("spark_allf_col", recs)
            .expect("insert");

        // Malformed / non-array filter JSON ⇒ no push-down ⇒ all rows.
        let (filter, total) = plan_and_read(&db, "spark_allf_col", "{}");
        assert!(filter.is_none(), "malformed filter ⇒ nothing stamped");
        assert_eq!(total, 4);
    }

    #[test]
    fn test_spark_create_data_writer_parses_schema_json() {
        let schema_json =
            r#"{"fields":[{"name":"id","data_type":"Utf8","nullable":false}],"metadata":{}}"#;
        let writer = spark_create_data_writer("write_col", schema_json, 7).expect("create writer");
        assert_eq!(writer.collection(), "write_col");
        assert_eq!(writer.partition_id(), 7);
        assert_eq!(writer.records_written(), 0);
    }

    #[test]
    fn test_spark_create_data_writer_rejects_malformed_schema() {
        let err = spark_create_data_writer("col", "{garbage", 0).expect_err("must error");
        assert!(matches!(err, SparkError::InvalidArgument(_)), "got: {err}");
    }

    #[test]
    fn test_spark_write_batch_inserts_records_into_collection() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_wr_col", 4, None)
            .expect("create");
        let schema_json =
            r#"{"fields":[{"name":"id","data_type":"Utf8","nullable":false}],"metadata":{}}"#;
        let mut writer = spark_create_data_writer("spark_wr_col", schema_json, 0).expect("writer");
        let recs = vec![make_spark_record("w_001"), make_spark_record("w_002")];
        let batch = proxima_records_to_record_batch(&recs).expect("encode rb");
        let bytes = record_batch_to_arrow_ipc(&batch).expect("encode ipc");
        spark_write_batch(&db, &mut writer, &bytes).expect("write");
        assert_eq!(writer.records_written(), 2);
        assert!(writer.bytes_written() > 0);
    }

    #[test]
    fn test_spark_commit_writer_returns_commit_message_json() {
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_cm_col", 4, None)
            .expect("create");
        let schema_json =
            r#"{"fields":[{"name":"id","data_type":"Utf8","nullable":false}],"metadata":{}}"#;
        let mut writer = spark_create_data_writer("spark_cm_col", schema_json, 3).expect("writer");
        let recs = vec![make_spark_record("c1")];
        let batch = proxima_records_to_record_batch(&recs).unwrap();
        spark_write_batch(
            &db,
            &mut writer,
            &record_batch_to_arrow_ipc(&batch).unwrap(),
        )
        .expect("write");
        let json = spark_commit_writer(&db, writer).expect("commit");
        let parsed: SparkWriteCommitMessage = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.partition_id, 3);
        assert_eq!(parsed.records_written, 1);
        assert!(parsed.bytes_written > 0);
    }

    /// Critical end-to-end test (per Plan-agent recommendation):
    /// 50 records written via JNI writer flow, then drained via JNI
    /// reader flow; assert total rowcount + distinct oids + that the
    /// cursor advances (no stuck-cursor bug returning duplicate pages).
    #[test]
    fn test_spark_read_after_write_round_trip() {
        use std::collections::HashSet;
        let (db, _td) = build_spark_test_db();
        db.create_collection("spark_e2e_col", 4, None)
            .expect("create");

        let schema_json =
            r#"{"fields":[{"name":"id","data_type":"Utf8","nullable":false}],"metadata":{}}"#;
        let mut writer = spark_create_data_writer("spark_e2e_col", schema_json, 0).expect("writer");
        let recs: Vec<ProximaRecord> = (0..50)
            .map(|i| make_spark_record(&format!("e2e_{i:03}")))
            .collect();
        let batch = proxima_records_to_record_batch(&recs).unwrap();
        let bytes = record_batch_to_arrow_ipc(&batch).unwrap();
        spark_write_batch(&db, &mut writer, &bytes).expect("write");
        spark_commit_writer(&db, writer).expect("commit");

        let json = spark_plan_input_partitions(&db, "spark_e2e_col", "{}", 1);
        let partitions: Vec<SparkInputPartition> = serde_json::from_str(&json).unwrap();
        let mut reader =
            spark_create_partition_reader(&serde_json::to_string(&partitions[0]).unwrap())
                .expect("reader");

        let mut all_ids: HashSet<String> = HashSet::new();
        let mut page_count = 0;
        loop {
            let page_bytes = spark_read_next_batch(&db, &mut reader).expect("read");
            if page_bytes.is_empty() {
                break;
            }
            page_count += 1;
            let batch = arrow_ipc_to_record_batch(&page_bytes).expect("decode");
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<arrow::array::StringArray>())
                .expect("id col");
            for row in 0..batch.num_rows() {
                let id = id_col.value(row).to_string();
                assert!(
                    all_ids.insert(id.clone()),
                    "duplicate id detected (stuck-cursor bug): {id}"
                );
            }
            assert!(page_count < 100, "page count exceeded sanity bound");
        }
        assert_eq!(all_ids.len(), 50, "expected exactly 50 distinct rows");
        assert!(reader.is_finished());
        spark_close_partition_reader(reader);
    }
}
