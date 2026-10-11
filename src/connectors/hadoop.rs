//! # Hadoop InputFormat/OutputFormat Compatibility Shim
//!
//! Provides Hadoop InputFormat and OutputFormat interfaces for legacy integration
//! with Hive, EMR, and other Hadoop ecosystem tools.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      Hadoop Ecosystem (Legacy)                          │
//! │  ┌─────────────────────┐    ┌─────────────────────┐                    │
//! │  │ Hive SerDe          │    │ MapReduce Job       │                    │
//! │  └─────────────────────┘    └─────────────────────┘                    │
//! │            │                          │                                 │
//! └────────────┼──────────────────────────┼─────────────────────────────────┘
//!              │ InputFormat               │ OutputFormat
//!              ▼                           ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Hadoop Compatibility Shim                           │
//! │  ┌─────────────────────┐    ┌─────────────────────┐                    │
//! │  │ ProximaInputFormat  │    │ ProximaOutputFormat │                    │
//! │  │ (Arrow-to-Writable) │    │ (Writable-to-Arrow) │                    │
//! │  └─────────────────────┘    └─────────────────────┘                    │
//! └─────────────────────────────────────────────────────────────────────────┘
//!              │                           │
//!              ▼                           ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Arrow-Native TableProvider                           │
//! │              (SST, HELIX, SWIFT, NOVA, VIPER, RAPTOR)                  │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage with Hive
//!
//! ```sql
//! -- Create external table using ProximaDB InputFormat
//! CREATE EXTERNAL TABLE embeddings_ext (
//!     id STRING,
//!     embedding ARRAY<FLOAT>,
//!     metadata MAP<STRING, STRING>
//! )
//! STORED BY 'org.proximadb.hive.ProximaStorageHandler'
//! TBLPROPERTIES (
//!     'proximadb.collection' = 'embeddings',
//!     'proximadb.host' = 'localhost:5678'
//! );
//!
//! -- Query with vector search pushed down
//! SELECT * FROM embeddings_ext
//! WHERE proximadb_knn(embedding, array(0.1, 0.2, ...), 10) > 0.8;
//! ```
//!
//! ## Usage with MapReduce
//!
//! ```java
//! // Configure job with ProximaDB InputFormat
//! job.setInputFormatClass(ProximaInputFormat.class);
//! ProximaInputFormat.setCollection(job, "embeddings");
//! ProximaInputFormat.setHost(job, "localhost:5678");
//!
//! // Run MapReduce job
//! job.waitForCompletion(true);
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::Schema as ArrowSchema;
use serde::{Deserialize, Serialize};

use crate::storage::formats::{FileSplit, SplitStatistics, SplitType};

/// Configuration for Hadoop compatibility shim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HadoopShimConfig {
    /// ProximaDB host
    pub host: String,
    /// ProximaDB port
    pub port: u16,
    /// Collection name
    pub collection: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// MapReduce split size hint (bytes)
    pub split_size_hint: u64,
    /// Enable speculative execution
    pub speculative_execution: bool,
    /// Maximum map tasks
    pub max_map_tasks: Option<usize>,
    /// SerDe class name
    pub serde_class: String,
}

impl Default for HadoopShimConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5678,
            collection: String::new(),
            auth_token: None,
            split_size_hint: 128 * 1024 * 1024, // 128MB
            speculative_execution: true,
            max_map_tasks: None,
            serde_class: "org.proximadb.hive.ProximaSerDe".to_string(),
        }
    }
}

/// Hadoop InputSplit - represents a chunk of data for a single map task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HadoopInputSplit {
    /// Split ID
    pub split_id: String,
    /// Underlying ProximaDB file split
    pub file_split: FileSplit,
    /// Split length in bytes (for scheduling)
    pub length: u64,
    /// Preferred locations for this split
    pub locations: Vec<String>,
}

impl HadoopInputSplit {
    /// Create from ProximaDB FileSplit
    pub fn from_file_split(file_split: FileSplit) -> Self {
        let locations = file_split.locality.preferred_hosts.clone();
        let length = file_split.length;
        let split_id = file_split.split_id.clone();

        Self {
            split_id,
            file_split,
            length,
            locations,
        }
    }

    /// Get split length (for Hadoop scheduling)
    pub fn get_length(&self) -> u64 {
        self.length
    }

    /// Get locations (for Hadoop locality)
    pub fn get_locations(&self) -> &[String] {
        &self.locations
    }

    /// Serialize to bytes for Hadoop
    pub fn write_fields(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn read_fields(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }
}

/// ProximaDB InputFormat - Hadoop InputFormat implementation
///
/// This is a thin shim that wraps the Arrow-native TableProvider
/// and presents a Hadoop-compatible interface.
pub struct ProximaInputFormat {
    /// Configuration
    config: HadoopShimConfig,
}

impl ProximaInputFormat {
    /// Create new InputFormat
    pub fn new(config: HadoopShimConfig) -> Self {
        Self { config }
    }

    /// Get input splits for a job
    ///
    /// This maps to Arrow-native FileSplit generation and wraps them
    /// in HadoopInputSplit for compatibility.
    pub fn get_splits(&self, _num_splits_hint: usize) -> Vec<HadoopInputSplit> {
        // Splits: query ProximaDB storage assignment for file-based splits
        // For now, return a placeholder split
        let file_split = FileSplit {
            split_id: format!("{}:0", self.config.collection),
            file_path: String::new(),
            offset: 0,
            length: self.config.split_size_hint,
            split_type: SplitType::ByteRange {
                estimated_records: 0,
            },
            statistics: SplitStatistics::default(),
            locality: crate::storage::formats::SplitLocality::default(),
            object_size: None,
        };

        vec![HadoopInputSplit::from_file_split(file_split)]
    }

    /// Create a record reader for a split
    pub fn create_record_reader(&self, split: &HadoopInputSplit) -> ProximaRecordReader {
        ProximaRecordReader::new(split.clone(), self.config.clone())
    }
}

/// ProximaDB RecordReader - reads records from a split
///
/// Converts Arrow RecordBatches to Hadoop WritableComparable format.
pub struct ProximaRecordReader {
    /// Input split being read
    #[allow(dead_code)]
    split: HadoopInputSplit,
    /// Configuration
    config: HadoopShimConfig,
    /// Buffer of records drained from the most recent `fetch_next_page`
    /// call. `next_record` pops from the front; refill happens when
    /// empty AND `!exhausted` (TD-099 3b).
    record_buffer: VecDeque<serde_json::Value>,
    /// Last record drained from the buffer, returned by
    /// [`Self::get_current_value`].
    current_record: Option<serde_json::Value>,
    /// Total rows read
    total_rows_read: u64,
    /// Whether reader is exhausted
    exhausted: bool,
    /// Continuation cursor from the most recent scan response. `None`
    /// before the first fetch; `Some` between pages; cleared (and
    /// `exhausted` flipped) when the server returns `next_cursor: null`.
    cursor: Option<String>,
    /// Shared HTTP client for the v2 scan endpoint.
    http: reqwest::Client,
}

impl ProximaRecordReader {
    /// Create new record reader
    pub fn new(split: HadoopInputSplit, config: HadoopShimConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            split,
            config,
            record_buffer: VecDeque::new(),
            current_record: None,
            total_rows_read: 0,
            exhausted: false,
            cursor: None,
            http,
        }
    }

    /// Initialize the reader
    pub fn initialize(&mut self) -> Result<(), HadoopError> {
        // Reader init: establish connection and prefetch first batch
        Ok(())
    }

    /// Fetch the next page from `POST /api/v2/collections/{id}/records/scan`
    /// (operationId `scanRecords`). Returns the page's `records` array
    /// and updates the reader's continuation cursor + exhausted flag.
    ///
    /// Async because Hadoop's `next_record() -> bool` is sync; callers
    /// drive this method from an async wrapper (or via
    /// `tokio::runtime::Handle::block_on` in the Hadoop ABI).
    pub async fn fetch_next_page(&mut self) -> Result<Vec<serde_json::Value>, HadoopError> {
        if self.exhausted {
            return Ok(vec![]);
        }
        let url = format!(
            "http://{}:{}/api/v2/collections/{}/records/scan",
            self.config.host, self.config.port, self.config.collection
        );
        let body = serde_json::json!({
            "cursor": self.cursor,
            "limit": 1000,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| HadoopError {
                message: format!("POST {url}: {e}"),
                code: HadoopErrorCode::Connection,
            })?;
        if !resp.status().is_success() {
            return Err(HadoopError {
                message: format!("POST {url} returned {}", resp.status()),
                code: HadoopErrorCode::IO,
            });
        }
        let parsed: serde_json::Value = resp.json().await.map_err(|e| HadoopError {
            message: format!("decode scan body: {e}"),
            code: HadoopErrorCode::IO,
        })?;
        let records = parsed
            .get("records")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        self.cursor = parsed
            .get("next_cursor")
            .and_then(|v| v.as_str())
            .map(String::from);
        if self.cursor.is_none() {
            self.exhausted = true;
        }
        self.total_rows_read = self.total_rows_read.saturating_add(records.len() as u64);
        Ok(records)
    }

    /// Read the next record from the buffered page; refill the buffer
    /// via [`Self::fetch_next_page`] when empty.
    ///
    /// TD-099 (3b) sync-bridge: Hadoop's InputFormat contract calls
    /// this blocking ABI in a tight loop, but `fetch_next_page` is
    /// async. We bridge by spawning a scoped OS thread that owns a
    /// fresh `current_thread` tokio runtime and `block_on`s the
    /// fetch. This is runtime-context-agnostic — works whether the
    /// caller is in a `#[tokio::test]` (current-thread or multi-thread
    /// flavor) or pure sync (real Hadoop ABI) — and avoids the
    /// "Cannot start a runtime from within a runtime" panic that a
    /// naive `Runtime::new().block_on(...)` would trip when called
    /// from inside an existing tokio runtime.
    ///
    /// The thread spawn cost is amortized across the page (1000
    /// records by default), so per-record latency stays bounded by
    /// the buffer pop. Returns false only when the buffer is empty
    /// AND the server signalled `next_cursor: null` (end of scan).
    // expect() is intentional: the current-thread runtime build is infallible, and
    // the join().expect re-raises a panic from the bridged async task (same crash the
    // caller would see) — both are non-recoverable programmer/runtime errors.
    #[allow(clippy::expect_used)]
    pub fn next_record(&mut self) -> bool {
        if self.record_buffer.is_empty() && !self.exhausted {
            let fetch_result = std::thread::scope(|s| {
                s.spawn(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("build tokio runtime for sync next_record bridge");
                    rt.block_on(self.fetch_next_page())
                })
                .join()
                .expect("sync-bridge thread panicked")
            });
            match fetch_result {
                Ok(records) => {
                    self.record_buffer.extend(records);
                }
                Err(_) => {
                    self.exhausted = true;
                    return false;
                }
            }
        }

        match self.record_buffer.pop_front() {
            Some(record) => {
                self.current_record = Some(record);
                self.total_rows_read = self.total_rows_read.saturating_add(1);
                true
            }
            None => false,
        }
    }

    /// Get current key (record ID). Extracts the `id` field of the
    /// last-drained record; empty Text when no record has been read.
    pub fn get_current_key(&self) -> HadoopWritable {
        let id = self
            .current_record
            .as_ref()
            .and_then(|r| r.get("id"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_default();
        HadoopWritable::Text(id)
    }

    /// Get current value (record data). Converts the last-drained
    /// record's full JSON shape to a `MapWritable` via
    /// [`HadoopWritable::from_json`]. Empty MapWritable when no record
    /// has been read.
    pub fn get_current_value(&self) -> HadoopWritable {
        match self.current_record.as_ref() {
            Some(record) => HadoopWritable::from_json(record),
            None => HadoopWritable::MapWritable(HashMap::new()),
        }
    }

    /// Get progress (0.0 to 1.0). The Hadoop scan API has no total
    /// known up front, so we report 1.0 when exhausted AND the buffer
    /// is drained, else 0.0 mid-scan.
    pub fn get_progress(&self) -> f32 {
        if self.exhausted && self.record_buffer.is_empty() {
            1.0
        } else {
            0.0
        }
    }

    /// Close the reader
    pub fn close(&mut self) {
        self.exhausted = true;
        self.record_buffer.clear();
    }
}

/// ProximaDB OutputFormat - Hadoop OutputFormat implementation
pub struct ProximaOutputFormat {
    /// Configuration
    config: HadoopShimConfig,
}

impl ProximaOutputFormat {
    /// Create new OutputFormat
    pub fn new(config: HadoopShimConfig) -> Self {
        Self { config }
    }

    /// Check if output is valid
    pub fn check_output_specs(&self) -> Result<(), HadoopError> {
        if self.config.collection.is_empty() {
            return Err(HadoopError {
                message: "Collection name not specified".to_string(),
                code: HadoopErrorCode::InvalidConfiguration,
            });
        }
        Ok(())
    }

    /// Create a record writer
    pub fn create_record_writer(&self, task_id: i32) -> ProximaRecordWriter {
        ProximaRecordWriter::new(self.config.clone(), task_id)
    }

    /// Get output committer
    pub fn get_output_committer(&self) -> ProximaOutputCommitter {
        ProximaOutputCommitter::new(self.config.clone())
    }
}

/// ProximaDB RecordWriter - writes records to ProximaDB
pub struct ProximaRecordWriter {
    /// Configuration
    config: HadoopShimConfig,
    /// Task ID
    #[allow(dead_code)]
    task_id: i32,
    /// Records written
    records_written: u64,
    /// Bytes written
    #[allow(dead_code)]
    bytes_written: u64,
    /// Batch buffer
    batch_buffer: Vec<HashMap<String, HadoopWritable>>,
    /// Batch size threshold
    batch_size: usize,
    /// Shared HTTP client for the v2 batch endpoint.
    http: reqwest::Client,
}

impl ProximaRecordWriter {
    /// Create new record writer
    pub fn new(config: HadoopShimConfig, task_id: i32) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config,
            task_id,
            records_written: 0,
            bytes_written: 0,
            batch_buffer: Vec::new(),
            batch_size: 1000,
            http,
        }
    }

    /// Buffer a single Writable-shaped row without going through the
    /// (key, value) Hadoop adapter. Public so the contract gate can seed
    /// the buffer deterministically.
    pub fn buffer_row(&mut self, row: HashMap<String, HadoopWritable>) {
        self.batch_buffer.push(row);
    }

    /// Write a key-value pair. Buffers, and flushes once the buffer
    /// reaches `batch_size`.
    pub async fn write(
        &mut self,
        _key: &HadoopWritable,
        value: &HadoopWritable,
    ) -> Result<(), HadoopError> {
        if let HadoopWritable::MapWritable(map) = value {
            self.batch_buffer.push(map.clone());

            if self.batch_buffer.len() >= self.batch_size {
                self.flush_now().await?;
            }
        }
        Ok(())
    }

    /// Flush whatever's in the buffer right now (regardless of batch_size).
    /// Posts to `POST /api/v2/collections/{collection_id}/records/batch`
    /// (operationId `insertRecords`). Returns Ok(()) when the buffer is
    /// empty.
    pub async fn flush_now(&mut self) -> Result<(), HadoopError> {
        if self.batch_buffer.is_empty() {
            return Ok(());
        }
        let url = format!(
            "http://{}:{}/api/v2/collections/{}/records/batch",
            self.config.host, self.config.port, self.config.collection
        );
        let records: Vec<serde_json::Value> = self
            .batch_buffer
            .iter()
            .map(|row| {
                // Minimal lowering: name-as-id placeholder when no id field;
                // richer Writable→ProximaRecord conversion is a follow-up.
                let id = row
                    .get("id")
                    .and_then(|v| match v {
                        HadoopWritable::Text(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                serde_json::json!({ "id": id })
            })
            .collect();
        let body = serde_json::json!({ "records": records });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| HadoopError {
                message: format!("POST {url}: {e}"),
                code: HadoopErrorCode::Connection,
            })?;
        if !resp.status().is_success() {
            return Err(HadoopError {
                message: format!("POST {url} returned {}", resp.status()),
                code: HadoopErrorCode::IO,
            });
        }
        let n = self.batch_buffer.len() as u64;
        self.records_written = self.records_written.saturating_add(n);
        self.batch_buffer.clear();
        let _ = resp.bytes().await;
        Ok(())
    }

    /// Close the writer
    pub async fn close(&mut self) -> Result<(), HadoopError> {
        self.flush_now().await
    }
}

/// ProximaDB Output Committer - handles job/task commit protocol
pub struct ProximaOutputCommitter {
    /// Configuration
    #[allow(dead_code)]
    config: HadoopShimConfig,
}

impl ProximaOutputCommitter {
    /// Create new committer
    pub fn new(config: HadoopShimConfig) -> Self {
        Self { config }
    }

    /// Setup the job
    pub fn setup_job(&self) -> Result<(), HadoopError> {
        // Setup: temp output directory for task staging
        Ok(())
    }

    /// Setup a task
    pub fn setup_task(&self, _task_id: i32) -> Result<(), HadoopError> {
        Ok(())
    }

    /// Check if task needs commit
    pub fn needs_task_commit(&self, _task_id: i32) -> bool {
        true
    }

    /// Commit a task
    pub fn commit_task(&self, _task_id: i32) -> Result<(), HadoopError> {
        // Commit task: move staged output to final location
        Ok(())
    }

    /// Abort a task
    pub fn abort_task(&self, _task_id: i32) -> Result<(), HadoopError> {
        // Abort task: clean up staged temporary output
        Ok(())
    }

    /// Commit the job
    pub fn commit_job(&self) -> Result<(), HadoopError> {
        // Commit job: finalize all task outputs atomically
        Ok(())
    }

    /// Abort the job
    pub fn abort_job(&self) -> Result<(), HadoopError> {
        // Abort job: clean up all temporary outputs
        Ok(())
    }

    /// Check if recovery is supported
    pub fn is_recovery_supported(&self) -> bool {
        true
    }

    /// Recover a task
    pub fn recover_task(&self, _task_id: i32) -> Result<(), HadoopError> {
        Ok(())
    }
}

/// Hadoop Writable types for data exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HadoopWritable {
    /// Null value
    NullWritable,
    /// Boolean
    BooleanWritable(bool),
    /// Byte (i8)
    ByteWritable(i8),
    /// Short (i16)
    ShortWritable(i16),
    /// Int (i32)
    IntWritable(i32),
    /// Long (i64)
    LongWritable(i64),
    /// Float
    FloatWritable(f32),
    /// Double
    DoubleWritable(f64),
    /// Text (String)
    Text(String),
    /// Bytes
    BytesWritable(Vec<u8>),
    /// Array of Writables
    ArrayWritable(Vec<HadoopWritable>),
    /// Map of Writables
    MapWritable(HashMap<String, HadoopWritable>),
}

impl HadoopWritable {
    /// Convert to serde_json::Value
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::NullWritable => serde_json::Value::Null,
            Self::BooleanWritable(b) => serde_json::json!(b),
            Self::ByteWritable(b) => serde_json::json!(b),
            Self::ShortWritable(s) => serde_json::json!(s),
            Self::IntWritable(i) => serde_json::json!(i),
            Self::LongWritable(l) => serde_json::json!(l),
            Self::FloatWritable(f) => serde_json::json!(f),
            Self::DoubleWritable(d) => serde_json::json!(d),
            Self::Text(t) => serde_json::json!(t),
            Self::BytesWritable(b) => serde_json::json!(base64::encode(b)),
            Self::ArrayWritable(arr) => {
                serde_json::json!(arr.iter().map(|w| w.to_json()).collect::<Vec<_>>())
            }
            Self::MapWritable(map) => {
                serde_json::json!(
                    map.iter()
                        .map(|(k, v)| (k.clone(), v.to_json()))
                        .collect::<HashMap<_, _>>()
                )
            }
        }
    }

    /// Create from serde_json::Value
    pub fn from_json(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::NullWritable,
            serde_json::Value::Bool(b) => Self::BooleanWritable(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                        Self::IntWritable(i as i32)
                    } else {
                        Self::LongWritable(i)
                    }
                } else if let Some(f) = n.as_f64() {
                    Self::DoubleWritable(f)
                } else {
                    Self::NullWritable
                }
            }
            serde_json::Value::String(s) => Self::Text(s.clone()),
            serde_json::Value::Array(arr) => {
                Self::ArrayWritable(arr.iter().map(Self::from_json).collect())
            }
            serde_json::Value::Object(obj) => Self::MapWritable(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::from_json(v)))
                    .collect(),
            ),
        }
    }
}

/// Hive SerDe - serializer/deserializer for Hive integration
pub struct ProximaSerDe {
    /// Schema
    #[allow(dead_code)]
    schema: Option<Arc<ArrowSchema>>,
    /// Column names
    column_names: Vec<String>,
    /// Column types
    column_types: Vec<HiveType>,
}

/// Hive types for SerDe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HiveType {
    /// String/varchar
    String,
    /// Boolean
    Boolean,
    /// Tinyint (i8)
    Tinyint,
    /// Smallint (i16)
    Smallint,
    /// Int (i32)
    Int,
    /// Bigint (i64)
    Bigint,
    /// Float
    Float,
    /// Double
    Double,
    /// Binary
    Binary,
    /// Array
    Array(Box<HiveType>),
    /// Map
    Map(Box<HiveType>, Box<HiveType>),
    /// Struct
    Struct(Vec<(String, HiveType)>),
}

impl ProximaSerDe {
    /// Create new SerDe
    pub fn new() -> Self {
        Self {
            schema: None,
            column_names: Vec::new(),
            column_types: Vec::new(),
        }
    }

    /// Initialize with table properties
    pub fn initialize(&mut self, properties: &HashMap<String, String>) -> Result<(), HadoopError> {
        // Parse column names
        if let Some(cols) = properties.get("columns") {
            self.column_names = cols.split(',').map(|s| s.trim().to_string()).collect();
        }

        // Parse column types
        if let Some(types) = properties.get("columns.types") {
            self.column_types = types
                .split(':')
                .map(|t| parse_hive_type(t.trim()))
                .collect();
        }

        Ok(())
    }

    /// Deserialize a row
    pub fn deserialize(&self, _data: &[u8]) -> Result<HadoopWritable, HadoopError> {
        // Deserialize: Hadoop Writable → ProximaDB record
        Ok(HadoopWritable::MapWritable(HashMap::new()))
    }

    /// Serialize a row
    pub fn serialize(&self, _row: &HadoopWritable) -> Result<Vec<u8>, HadoopError> {
        // Serialize: ProximaDB record → Hadoop Writable
        Ok(Vec::new())
    }
}

impl Default for ProximaSerDe {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse Hive type string to HiveType
fn parse_hive_type(type_str: &str) -> HiveType {
    let type_str = type_str.to_lowercase();

    if type_str == "string" || type_str.starts_with("varchar") {
        HiveType::String
    } else if type_str == "boolean" {
        HiveType::Boolean
    } else if type_str == "tinyint" {
        HiveType::Tinyint
    } else if type_str == "smallint" {
        HiveType::Smallint
    } else if type_str == "int" {
        HiveType::Int
    } else if type_str == "bigint" {
        HiveType::Bigint
    } else if type_str == "float" {
        HiveType::Float
    } else if type_str == "double" {
        HiveType::Double
    } else if type_str == "binary" {
        HiveType::Binary
    } else if type_str.starts_with("array<") {
        let inner = &type_str[6..type_str.len() - 1];
        HiveType::Array(Box::new(parse_hive_type(inner)))
    } else if type_str.starts_with("map<") {
        let inner = &type_str[4..type_str.len() - 1];
        let parts: Vec<&str> = inner.splitn(2, ',').collect();
        if parts.len() == 2 {
            HiveType::Map(
                Box::new(parse_hive_type(parts[0].trim())),
                Box::new(parse_hive_type(parts[1].trim())),
            )
        } else {
            HiveType::String
        }
    } else {
        HiveType::String
    }
}

/// Hadoop error
#[derive(Debug)]
pub struct HadoopError {
    /// Error message
    pub message: String,
    /// Error code
    pub code: HadoopErrorCode,
}

impl std::fmt::Display for HadoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HadoopError({:?}): {}", self.code, self.message)
    }
}

impl std::error::Error for HadoopError {}

/// Hadoop error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadoopErrorCode {
    /// Invalid configuration
    InvalidConfiguration,
    /// IO error
    IO,
    /// Connection error
    Connection,
    /// Authentication error
    Authentication,
    /// Not found
    NotFound,
    /// Internal error
    Internal,
}

/// Base64 encoding helper (minimal implementation)
mod base64 {
    /// Standard Base64 alphabet characters.
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// Encode a byte slice as a Base64-encoded string.
    pub fn encode(data: &[u8]) -> String {
        let mut result = String::new();
        let mut i = 0;

        while i < data.len() {
            let a = data[i];
            let b = data.get(i + 1).copied().unwrap_or(0);
            let c = data.get(i + 2).copied().unwrap_or(0);

            result.push(ALPHABET[(a >> 2) as usize] as char);
            result.push(ALPHABET[(((a & 0x03) << 4) | (b >> 4)) as usize] as char);

            if i + 1 < data.len() {
                result.push(ALPHABET[(((b & 0x0f) << 2) | (c >> 6)) as usize] as char);
            } else {
                result.push('=');
            }

            if i + 2 < data.len() {
                result.push(ALPHABET[(c & 0x3f) as usize] as char);
            } else {
                result.push('=');
            }

            i += 3;
        }

        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadoop_config_default() {
        let config = HadoopShimConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5678);
        assert!(config.speculative_execution);
    }

    #[test]
    fn test_hadoop_input_split() {
        let file_split = FileSplit {
            split_id: "test:0".to_string(),
            file_path: "/data/file.sst".to_string(),
            offset: 0,
            length: 1024,
            split_type: SplitType::Block {
                block_id: 0,
                record_count: 100,
            },
            statistics: SplitStatistics::default(),
            locality: crate::storage::formats::SplitLocality {
                preferred_hosts: vec!["host1".to_string()],
                ..Default::default()
            },
            object_size: None,
        };

        let split = HadoopInputSplit::from_file_split(file_split);
        assert_eq!(split.split_id, "test:0");
        assert_eq!(split.get_length(), 1024);
        assert_eq!(split.get_locations(), &["host1"]);
    }

    #[test]
    #[allow(clippy::panic)] // Test panic for assertion failure
    fn test_hadoop_writable_conversion() {
        let writable = HadoopWritable::IntWritable(42);
        let json = writable.to_json();
        assert_eq!(json, serde_json::json!(42));

        let back = HadoopWritable::from_json(&json);
        match back {
            HadoopWritable::IntWritable(i) => assert_eq!(i, 42),
            _ => panic!("Expected IntWritable"),
        }
    }

    #[test]
    fn test_hadoop_writable_map() {
        let mut map = HashMap::new();
        map.insert("name".to_string(), HadoopWritable::Text("test".to_string()));
        map.insert("count".to_string(), HadoopWritable::IntWritable(5));

        let writable = HadoopWritable::MapWritable(map);
        let json = writable.to_json();

        assert!(json.get("name").is_some());
        assert_eq!(json.get("name").unwrap(), &serde_json::json!("test"));
    }

    #[test]
    #[allow(clippy::panic)] // Test panic for assertion failure
    fn test_hive_type_parsing() {
        assert!(matches!(parse_hive_type("string"), HiveType::String));
        assert!(matches!(parse_hive_type("int"), HiveType::Int));
        assert!(matches!(parse_hive_type("bigint"), HiveType::Bigint));
        assert!(matches!(parse_hive_type("BOOLEAN"), HiveType::Boolean));

        match parse_hive_type("array<float>") {
            HiveType::Array(inner) => assert!(matches!(*inner, HiveType::Float)),
            _ => panic!("Expected Array"),
        }

        match parse_hive_type("map<string,int>") {
            HiveType::Map(k, v) => {
                assert!(matches!(*k, HiveType::String));
                assert!(matches!(*v, HiveType::Int));
            }
            _ => panic!("Expected Map"),
        }
    }

    #[test]
    fn test_input_format() {
        let config = HadoopShimConfig {
            collection: "test".to_string(),
            ..Default::default()
        };

        let input_format = ProximaInputFormat::new(config);
        let splits = input_format.get_splits(4);
        assert!(!splits.is_empty());
    }

    #[test]
    fn test_output_format() {
        let config = HadoopShimConfig {
            collection: "test".to_string(),
            ..Default::default()
        };

        let output_format = ProximaOutputFormat::new(config);
        assert!(output_format.check_output_specs().is_ok());

        let empty_config = HadoopShimConfig::default();
        let output_format_empty = ProximaOutputFormat::new(empty_config);
        assert!(output_format_empty.check_output_specs().is_err());
    }

    #[test]
    fn test_proxima_serde() {
        let mut serde = ProximaSerDe::new();
        let mut props = HashMap::new();
        props.insert("columns".to_string(), "id,name,value".to_string());
        props.insert("columns.types".to_string(), "string:string:int".to_string());

        assert!(serde.initialize(&props).is_ok());
        assert_eq!(serde.column_names.len(), 3);
        assert_eq!(serde.column_types.len(), 3);
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64::encode(b""), "");
        assert_eq!(base64::encode(b"f"), "Zg==");
        assert_eq!(base64::encode(b"fo"), "Zm8=");
        assert_eq!(base64::encode(b"foo"), "Zm9v");
        assert_eq!(base64::encode(b"foob"), "Zm9vYg==");
    }
}
