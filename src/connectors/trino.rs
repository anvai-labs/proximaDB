//! # Trino SPI Connector
//!
//! Provides a Trino Service Provider Interface (SPI) connector for ProximaDB.
//! This enables Trino queries to read from and write to ProximaDB collections.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Trino Coordinator                             │
//! │  ┌─────────────────────┐    ┌─────────────────────┐                    │
//! │  │ ProximaDBMetadata   │    │ ProximaDBSplitMgr   │                    │
//! │  │ (Java Plugin)       │    │ (Java Plugin)       │                    │
//! │  └─────────────────────┘    └─────────────────────┘                    │
//! │            │                          │                                 │
//! └────────────┼──────────────────────────┼─────────────────────────────────┘
//!              │ Arrow Flight             │ Arrow Flight
//!              ▼                          ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    ProximaDB Arrow Flight Server                        │
//! │  ┌─────────────────────┐    ┌─────────────────────┐                    │
//! │  │ TrinoMetadataHandler│    │ TrinoDataHandler    │                    │
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
//! ## Usage in Trino
//!
//! ```sql
//! -- Create catalog
//! CREATE CATALOG proximadb USING proximadb
//! WITH (
//!     "connection-url" = "grpc://localhost:5680",
//!     "auth-token" = "secret"
//! );
//!
//! -- Query vectors
//! SELECT * FROM proximadb.default.embeddings
//! WHERE category = 'science'
//! LIMIT 100;
//!
//! -- Vector search (custom function)
//! SELECT id, vector_similarity(embedding, ARRAY[0.1, 0.2, ...]) as score
//! FROM proximadb.default.embeddings
//! ORDER BY score DESC
//! LIMIT 10;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::Schema as ArrowSchema;
use serde::{Deserialize, Serialize};

use crate::storage::formats::{FileSplit, SplitStatistics};
use crate::storage::schema::ProximaSchema;

/// Configuration for Trino connector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoConnectorConfig {
    /// ProximaDB Arrow Flight endpoint
    pub flight_endpoint: String,
    /// Authentication token (optional)
    pub auth_token: Option<String>,
    /// Default schema (namespace)
    pub default_schema: String,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
    /// Maximum splits per node
    pub max_splits_per_node: usize,
    /// Enable dynamic filtering
    pub enable_dynamic_filtering: bool,
    /// Enable predicate pushdown
    pub enable_predicate_pushdown: bool,
    /// Enable projection pushdown
    pub enable_projection_pushdown: bool,
    /// Enable TopN pushdown
    pub enable_topn_pushdown: bool,
    /// Enable aggregation pushdown
    pub enable_aggregation_pushdown: bool,
}

impl Default for TrinoConnectorConfig {
    fn default() -> Self {
        Self {
            flight_endpoint: "grpc://localhost:5680".to_string(),
            auth_token: None,
            default_schema: "default".to_string(),
            connection_timeout_ms: 30000,
            query_timeout_ms: 300000, // 5 minutes
            max_splits_per_node: 16,
            enable_dynamic_filtering: true,
            enable_predicate_pushdown: true,
            enable_projection_pushdown: true,
            enable_topn_pushdown: true,
            enable_aggregation_pushdown: true,
        }
    }
}

/// Trino schema representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoSchema {
    /// Schema name
    pub name: String,
    /// Tables in this schema
    pub tables: Vec<String>,
}

/// Trino table representation
#[derive(Debug, Clone)]
pub struct TrinoTable {
    /// Schema name
    pub schema_name: String,
    /// Table name
    pub table_name: String,
    /// Arrow schema
    pub schema: Arc<ArrowSchema>,
    /// ProximaDB schema with metadata
    pub proxima_schema: Arc<ProximaSchema>,
    /// Table properties
    pub properties: HashMap<String, String>,
    /// Comment/description
    pub comment: Option<String>,
}

/// Trino column metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoColumnMetadata {
    /// Column name
    pub name: String,
    /// Trino type (varchar, bigint, array(real), etc.)
    pub trino_type: String,
    /// Arrow type name for reference
    pub arrow_type: String,
    /// Whether column is nullable
    pub nullable: bool,
    /// Whether column is hidden
    pub hidden: bool,
    /// Column comment
    pub comment: Option<String>,
}

/// Trino split - unit of parallel work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoSplit {
    /// Split identifier
    pub split_id: String,
    /// Catalog name
    pub catalog: String,
    /// Schema name
    pub schema: String,
    /// Table name
    pub table: String,
    /// Underlying file split
    pub file_split: FileSplit,
    /// Addresses of nodes where split is local
    pub addresses: Vec<TrinoHostAddress>,
    /// Whether split is remotely accessible
    pub remotely_accessible: bool,
}

impl TrinoSplit {
    /// Create a new Trino split from a file split
    pub fn from_file_split(
        catalog: String,
        schema: String,
        table: String,
        file_split: FileSplit,
    ) -> Self {
        let addresses: Vec<TrinoHostAddress> = file_split
            .locality
            .preferred_hosts
            .iter()
            .map(|host| TrinoHostAddress {
                host: host.clone(),
                port: 5680, // Arrow Flight port
            })
            .collect();

        Self {
            split_id: file_split.split_id.clone(),
            catalog,
            schema,
            table,
            file_split,
            addresses,
            remotely_accessible: true,
        }
    }
}

/// Trino host address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoHostAddress {
    /// Hostname
    pub host: String,
    /// Port
    pub port: u16,
}

/// Trino TupleDomain - constraint domain for predicate pushdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoTupleDomain {
    /// Column constraints (column_name -> domain)
    pub domains: HashMap<String, TrinoDomain>,
    /// Whether all rows pass (no constraints)
    pub is_all: bool,
    /// Whether no rows pass (impossible constraints)
    pub is_none: bool,
}

impl TrinoTupleDomain {
    /// Create domain that matches all rows
    pub fn all() -> Self {
        Self {
            domains: HashMap::new(),
            is_all: true,
            is_none: false,
        }
    }

    /// Create domain that matches no rows
    pub fn none() -> Self {
        Self {
            domains: HashMap::new(),
            is_all: false,
            is_none: true,
        }
    }

    /// Create domain with constraints
    pub fn with_domains(domains: HashMap<String, TrinoDomain>) -> Self {
        Self {
            domains,
            is_all: false,
            is_none: false,
        }
    }
}

/// Trino domain - constraint on a single column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoDomain {
    /// Column name
    pub column: String,
    /// Value ranges
    pub ranges: Vec<TrinoRange>,
    /// Null allowed
    pub null_allowed: bool,
}

/// Trino range - value range for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoRange {
    /// Low bound (JSON encoded)
    pub low: Option<serde_json::Value>,
    /// Low bound inclusive
    pub low_inclusive: bool,
    /// High bound (JSON encoded)
    pub high: Option<serde_json::Value>,
    /// High bound inclusive
    pub high_inclusive: bool,
}

impl TrinoRange {
    /// Create an equality range
    pub fn equal(value: serde_json::Value) -> Self {
        Self {
            low: Some(value.clone()),
            low_inclusive: true,
            high: Some(value),
            high_inclusive: true,
        }
    }

    /// Create a greater-than range
    pub fn greater_than(value: serde_json::Value, inclusive: bool) -> Self {
        Self {
            low: Some(value),
            low_inclusive: inclusive,
            high: None,
            high_inclusive: false,
        }
    }

    /// Create a less-than range
    pub fn less_than(value: serde_json::Value, inclusive: bool) -> Self {
        Self {
            low: None,
            low_inclusive: false,
            high: Some(value),
            high_inclusive: inclusive,
        }
    }

    /// Create a between range
    pub fn between(
        low: serde_json::Value,
        low_inclusive: bool,
        high: serde_json::Value,
        high_inclusive: bool,
    ) -> Self {
        Self {
            low: Some(low),
            low_inclusive,
            high: Some(high),
            high_inclusive,
        }
    }
}

/// Trino connector session
#[derive(Debug, Clone)]
pub struct TrinoConnectorSession {
    /// Query ID
    pub query_id: String,
    /// User
    pub user: String,
    /// Source
    pub source: Option<String>,
    /// Time zone
    pub time_zone: String,
    /// Session properties
    pub properties: HashMap<String, String>,
}

/// Trino table layout - describes how to read a table
#[derive(Debug, Clone)]
pub struct TrinoTableLayout {
    /// Table handle
    pub table: TrinoTable,
    /// Constraint from query
    pub constraint: TrinoTupleDomain,
    /// Columns to project
    pub projection: Option<Vec<String>>,
    /// Optional limit
    pub limit: Option<i64>,
}

/// Trino split manager - generates splits for parallel execution
pub struct TrinoSplitManager {
    /// Configuration
    #[allow(dead_code)]
    config: TrinoConnectorConfig,
}

impl TrinoSplitManager {
    /// Create a new split manager
    pub fn new(config: TrinoConnectorConfig) -> Self {
        Self { config }
    }

    /// Generate splits for a table layout
    pub fn get_splits(&self, layout: &TrinoTableLayout) -> Vec<TrinoSplit> {
        // Splits: file-location-based via Arrow Flight GetFlightInfo
        // For now, return a placeholder split
        let file_split = FileSplit {
            split_id: format!("{}.{}.0", layout.table.schema_name, layout.table.table_name),
            file_path: String::new(),
            offset: 0,
            length: 0,
            split_type: crate::storage::formats::SplitType::ByteRange {
                estimated_records: 0,
            },
            statistics: SplitStatistics::default(),
            locality: crate::storage::formats::SplitLocality::default(),
            object_size: None,
        };

        vec![TrinoSplit::from_file_split(
            "proximadb".to_string(),
            layout.table.schema_name.clone(),
            layout.table.table_name.clone(),
            file_split,
        )]
    }
}

/// Trino page source — streams data pages from a split via Arrow
/// Flight `DoGet`. Fifth live Flight method (TD-098, 2026-06-01); first
/// of the streaming RPCs.
///
/// Dial pattern matches the prior four live methods
/// (`Channel::from_shared(...).connect_lazy()` +
/// `FlightServiceClient::new(channel)`). The split is JSON-encoded
/// into the `Ticket.ticket` bytes — same wire format
/// `flight_get_splits` produced on the receive side. The response
/// stream is wrapped in `arrow_flight::decode::FlightRecordBatchStream`
/// so each `get_next_page` pulls the next `RecordBatch`, then converts
/// it into a `TrinoPage` (one `TrinoBlock` per column, each block
/// holding the column slice as a stream-IPC-encoded single-column
/// batch).
pub struct TrinoPageSource {
    #[allow(dead_code)]
    split: TrinoSplit,
    #[allow(dead_code)]
    schema: Arc<ArrowSchema>,
    stream: Option<arrow_flight::decode::FlightRecordBatchStream>,
    finished: bool,
    bytes_read: usize,
    rows_read: usize,
}

impl TrinoPageSource {
    /// Dial the Flight endpoint and open a `DoGet` stream for the
    /// supplied split. Connect / RPC failures mark the source as
    /// `finished` immediately so `get_next_page` returns `None` on
    /// the first call without further I/O — callers see "no rows"
    /// instead of a panic on transient blips.
    pub async fn new(endpoint: &str, split: TrinoSplit, schema: Arc<ArrowSchema>) -> Self {
        use arrow_flight::Ticket;
        use arrow_flight::decode::FlightRecordBatchStream;
        use arrow_flight::error::FlightError;
        use arrow_flight::flight_service_client::FlightServiceClient;
        use futures::TryStreamExt;

        let dial = endpoint
            .strip_prefix("grpc://")
            .map(|rest| format!("http://{rest}"))
            .unwrap_or_else(|| endpoint.to_string());

        let channel = match tonic::transport::Channel::from_shared(dial)
            .ok()
            .map(|e| e.connect_lazy())
        {
            Some(ch) => ch,
            None => {
                return Self {
                    split,
                    schema,
                    stream: None,
                    finished: true,
                    bytes_read: 0,
                    rows_read: 0,
                };
            }
        };

        let mut client = FlightServiceClient::new(channel);
        let ticket = Ticket {
            ticket: serde_json::to_vec(&split).unwrap_or_default().into(),
        };
        let stream = match client.do_get(tonic::Request::new(ticket)).await {
            Ok(resp) => {
                let mapped = resp
                    .into_inner()
                    .map_err(|s| FlightError::Tonic(Box::new(s)));
                Some(FlightRecordBatchStream::new_from_flight_data(mapped))
            }
            Err(_) => None,
        };
        let finished = stream.is_none();

        Self {
            split,
            schema,
            stream,
            finished,
            bytes_read: 0,
            rows_read: 0,
        }
    }

    /// Pull the next `RecordBatch` off the wire and surface it as a
    /// `TrinoPage`. Stream exhaustion, decode failure, or per-column
    /// IPC encoding failure flips the source to `finished` and
    /// returns `None` — callers stop iterating.
    pub async fn get_next_page(&mut self) -> Option<TrinoPage> {
        use futures::StreamExt;

        if self.finished {
            return None;
        }
        let Some(stream) = self.stream.as_mut() else {
            self.finished = true;
            return None;
        };
        match stream.next().await {
            Some(Ok(batch)) => match record_batch_to_trino_page(&batch) {
                Ok(page) => {
                    self.rows_read += batch.num_rows();
                    self.bytes_read += page.size_in_bytes;
                    Some(page)
                }
                Err(_) => {
                    self.finished = true;
                    None
                }
            },
            Some(Err(_)) | None => {
                self.finished = true;
                None
            }
        }
    }

    /// Check if source is finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get completion percentage (0.0 to 1.0)
    pub fn get_completion_percentage(&self) -> f64 {
        if self.finished { 1.0 } else { 0.0 }
    }

    /// Get bytes read
    pub fn get_bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Get rows read
    pub fn get_rows_read(&self) -> usize {
        self.rows_read
    }

    /// Close the page source
    pub fn close(&mut self) {
        self.finished = true;
        self.stream = None;
    }
}

/// Convert an Arrow `RecordBatch` to a Trino-shaped `TrinoPage`. Each
/// column becomes one `TrinoBlock` whose `data` is the stream-IPC
/// bytes of a single-column `RecordBatch` (so the consumer can decode
/// each block independently via `arrow::ipc::reader::StreamReader`).
///
/// Inverse of [`trino_page_to_record_batch`]; both round-trip through
/// the wire today (page-source decode + page-sink encode).
pub fn record_batch_to_trino_page(
    batch: &arrow::array::RecordBatch,
) -> Result<TrinoPage, arrow::error::ArrowError> {
    use arrow::array::RecordBatch;
    use arrow::ipc::writer::StreamWriter;

    let schema = batch.schema();
    let mut blocks = Vec::with_capacity(batch.num_columns());
    for (idx, col) in batch.columns().iter().enumerate() {
        let field = schema.field(idx).clone();
        let single_schema = Arc::new(ArrowSchema::new(vec![field.clone()]));
        let single_batch = RecordBatch::try_new(single_schema.clone(), vec![col.clone()])?;
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &single_schema)?;
            writer.write(&single_batch)?;
            writer.finish()?;
        }
        blocks.push(TrinoBlock {
            column: field.name().clone(),
            data_type: field.data_type().to_string(),
            data: buf,
            nulls: None,
            position_count: col.len(),
        });
    }
    Ok(TrinoPage {
        size_in_bytes: blocks.iter().map(|b| b.data.len()).sum(),
        position_count: batch.num_rows(),
        blocks,
    })
}

/// Trino page - columnar data format
#[derive(Debug)]
pub struct TrinoPage {
    /// Column blocks
    pub blocks: Vec<TrinoBlock>,
    /// Row count
    pub position_count: usize,
    /// Size in bytes
    pub size_in_bytes: usize,
}

/// Trino block - single column data
#[derive(Debug, Clone)]
pub struct TrinoBlock {
    /// Column name
    pub column: String,
    /// Data type
    pub data_type: String,
    /// Raw data (Arrow IPC encoded)
    pub data: Vec<u8>,
    /// Null flags
    pub nulls: Option<Vec<bool>>,
    /// Position count
    pub position_count: usize,
}

/// Trino page sink — writes data pages to ProximaDB via Arrow Flight
/// `DoPut`. Closes TD-098: this is the 7th and final live Flight
/// method; the second streaming RPC (after `TrinoPageSource`).
///
/// Bidirectional streaming: the client opens a `do_put` channel,
/// sends one `FlightData` per inbound page (preceded by a one-shot
/// schema message on the first append), and the server emits a
/// `PutResult` ack per inbound message. The sink does not block on
/// per-page acks today — `finish` drains all remaining `PutResult`s
/// before returning the summary, so the server has had a chance to
/// surface any per-batch errors.
pub struct TrinoPageSink {
    #[allow(dead_code)]
    table: TrinoTable,
    rows_written: usize,
    bytes_written: usize,
    committed: bool,
    /// Sender into the bidirectional do_put request stream. Dropping
    /// it (in `finish` / `abort`) closes the client side of the
    /// stream, which lets the server drain its loop.
    request_tx: Option<tokio::sync::mpsc::Sender<arrow_flight::FlightData>>,
    /// Server's PutResult stream. Drained on `finish` so caller sees
    /// any server-reported errors before the summary lands.
    response_rx: Option<tonic::Streaming<arrow_flight::PutResult>>,
    /// Schema message is sent lazily on the first `append_page` call
    /// (so a sink that's opened-then-immediately-finished writes no
    /// data over the wire).
    schema_sent: bool,
}

impl TrinoPageSink {
    /// Dial the Flight endpoint and open a bidirectional `do_put`
    /// stream for the supplied table. Returns `TrinoError` on
    /// connect / RPC failure so the planner can fast-fail before
    /// any pages are produced.
    pub async fn new(endpoint: &str, table: TrinoTable) -> Result<Self, TrinoError> {
        use arrow_flight::flight_service_client::FlightServiceClient;

        let dial = endpoint
            .strip_prefix("grpc://")
            .map(|rest| format!("http://{rest}"))
            .unwrap_or_else(|| endpoint.to_string());

        let channel = tonic::transport::Channel::from_shared(dial)
            .map_err(|e| TrinoError {
                error_code: TrinoErrorCode::InvalidArguments,
                message: format!("invalid flight endpoint: {e}"),
            })?
            .connect_lazy();

        let mut client = FlightServiceClient::new(channel);

        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<arrow_flight::FlightData>(16);
        let request_stream = tokio_stream::wrappers::ReceiverStream::new(request_rx);

        let response = client
            .do_put(tonic::Request::new(request_stream))
            .await
            .map_err(|s| TrinoError {
                error_code: TrinoErrorCode::ConnectionError,
                message: format!("do_put open failed: {s}"),
            })?;
        let response_rx = response.into_inner();

        Ok(Self {
            table,
            rows_written: 0,
            bytes_written: 0,
            committed: false,
            request_tx: Some(request_tx),
            response_rx: Some(response_rx),
            schema_sent: false,
        })
    }

    /// Decode the page back into a `RecordBatch`, send a one-shot
    /// schema `FlightData` if this is the first page, then send the
    /// batch `FlightData`. Counts rows + bytes for the eventual
    /// summary.
    pub async fn append_page(&mut self, page: TrinoPage) -> Result<(), TrinoError> {
        use arrow_flight::SchemaAsIpc;

        let tx = self.request_tx.as_ref().ok_or_else(|| TrinoError {
            error_code: TrinoErrorCode::GenericInternalError,
            message: "page sink already finished".to_string(),
        })?;

        let batch = trino_page_to_record_batch(&page).map_err(|e| TrinoError {
            error_code: TrinoErrorCode::InvalidArguments,
            message: format!("page decode failed: {e}"),
        })?;
        let batch_schema = batch.schema();
        let options = arrow::ipc::writer::IpcWriteOptions::default();

        if !self.schema_sent {
            let schema_msg: arrow_flight::FlightData =
                SchemaAsIpc::new(batch_schema.as_ref(), &options).into();
            tx.send(schema_msg).await.map_err(|e| TrinoError {
                error_code: TrinoErrorCode::ConnectionError,
                message: format!("schema send failed: {e}"),
            })?;
            self.schema_sent = true;
        }

        let messages =
            arrow_flight::utils::batches_to_flight_data(batch_schema.as_ref(), vec![batch.clone()])
                .map_err(|e| TrinoError {
                    error_code: TrinoErrorCode::GenericInternalError,
                    message: format!("batch encode failed: {e}"),
                })?;
        // batches_to_flight_data prepends a schema message we've
        // already sent above; skip it.
        for msg in messages.into_iter().skip(1) {
            self.bytes_written += msg.data_body.len();
            tx.send(msg).await.map_err(|e| TrinoError {
                error_code: TrinoErrorCode::ConnectionError,
                message: format!("batch send failed: {e}"),
            })?;
        }
        self.rows_written += batch.num_rows();
        Ok(())
    }

    /// Close the request stream, drain the server's PutResult acks
    /// (so any per-batch error surfaces here, not silently), and
    /// return the write summary.
    pub async fn finish(&mut self) -> Result<TrinoWriteSummary, TrinoError> {
        use futures::StreamExt;

        // Drop the sender to close the client side of the stream.
        self.request_tx = None;
        if let Some(mut rx) = self.response_rx.take() {
            while let Some(item) = rx.next().await {
                if let Err(s) = item {
                    return Err(TrinoError {
                        error_code: TrinoErrorCode::GenericInternalError,
                        message: format!("server reported error: {s}"),
                    });
                }
            }
        }
        self.committed = true;
        Ok(TrinoWriteSummary {
            rows_written: self.rows_written as i64,
            bytes_written: self.bytes_written as i64,
        })
    }

    /// Abort writing — drop the request stream without draining the
    /// response stream. The server's `do_put` loop sees its inbound
    /// channel close and exits.
    pub fn abort(&mut self) {
        self.request_tx = None;
        self.response_rx = None;
        self.committed = false;
    }
}

/// Decode a `TrinoPage` back to a `RecordBatch`. Each block holds a
/// single-column stream-IPC `RecordBatch`; this reverses that
/// encoding into a multi-column batch in block order.
///
/// Inverse of [`record_batch_to_trino_page`].
pub fn trino_page_to_record_batch(
    page: &TrinoPage,
) -> Result<arrow::array::RecordBatch, arrow::error::ArrowError> {
    use arrow::array::RecordBatch;
    use arrow::datatypes::Field;
    use arrow::error::ArrowError;
    use arrow::ipc::reader::StreamReader;

    let mut fields: Vec<Field> = Vec::with_capacity(page.blocks.len());
    let mut arrays = Vec::with_capacity(page.blocks.len());
    for (idx, block) in page.blocks.iter().enumerate() {
        let mut reader = StreamReader::try_new(&block.data[..], None)?;
        let batch = reader.next().ok_or_else(|| {
            ArrowError::IpcError(format!("block {idx} produced no record batch"))
        })??;
        if batch.num_columns() != 1 {
            return Err(ArrowError::SchemaError(format!(
                "block {idx} must encode exactly one column, got {}",
                batch.num_columns()
            )));
        }
        fields.push(batch.schema().field(0).clone());
        arrays.push(batch.column(0).clone());
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    RecordBatch::try_new(schema, arrays)
}

/// Trino write summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinoWriteSummary {
    /// Rows written
    pub rows_written: i64,
    /// Bytes written
    pub bytes_written: i64,
}

/// Trino error
#[derive(Debug)]
pub struct TrinoError {
    /// Error code
    pub error_code: TrinoErrorCode,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for TrinoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TrinoError({:?}): {}", self.error_code, self.message)
    }
}

impl std::error::Error for TrinoError {}

/// Trino standard error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrinoErrorCode {
    /// Generic internal error
    GenericInternalError,
    /// Table not found
    TableNotFound,
    /// Schema not found
    SchemaNotFound,
    /// Column not found
    ColumnNotFound,
    /// Permission denied
    PermissionDenied,
    /// Query exceeded limits
    ExceededLimits,
    /// Connection error
    ConnectionError,
    /// Timeout
    Timeout,
    /// Invalid arguments
    InvalidArguments,
    /// Not supported
    NotSupported,
}

// ============================================================================
// Arrow Flight Integration Points
// ============================================================================
//
// TD-098 RESOLVED 2026-06-02 — all 7 Flight methods live (real
// `FlightServiceClient` + RPC):
//   - `flight_list_schemas`             (list_flights    → bucketed TrinoSchema)
//   - `flight_list_tables`              (list_flights    → schema-filtered Vec<String>)
//   - `flight_get_table_schema`         (get_schema      → SchemaResult → ArrowSchema)
//   - `flight_get_splits`               (get_flight_info → endpoint tickets → Vec<TrinoSplit>)
//   - `TrinoPageSource::get_next_page`  (do_get          → FlightRecordBatchStream → TrinoPage)
//   - `TrinoPageSink::append_page`      (do_put          → FlightData stream → server PutResult ack)
//   - `TrinoPageSink::finish`           (drains PutResult acks → TrinoWriteSummary)
//
// See per-fn docstrings + the contract gate at
// `tests/connectors_flight_contract.rs::trino_flight_pilot` for the
// in-process tonic round-trip proof. Both have async signatures —
// older sync callers need a runtime handle.
//
// Dial template: `grpc://` → `http://` normalize →
// `Channel::from_shared(...).connect_lazy()` →
// `FlightServiceClient::new(channel)` (tonic ≥0.12 removed the
// convenience `connect()` shorthand). See
// `docs/10-quality/TECHNICAL_DEBT.adoc` TD-098 for the migration
// history.

/// Live Arrow Flight `ListFlights` query against the configured ProximaDB
/// Flight endpoint. Returns one [`TrinoSchema`] per unique first-path
/// segment in the returned `FlightInfo.flight_descriptor.path` — matches
/// the server's routing convention at
/// `src/network/arrow_ipc/service.rs:313-342` (`["relational", table_fqn]`
/// + `["vectors", collection_id]`).
///
/// First live Flight method (TD-098 pilot, 2026-06-01). The other 6
/// `flight_*` helpers below stay scaffolded; they follow the same
/// `FlightServiceClient::connect → list_flights / get_schema / do_get`
/// pattern and are mechanical follow-up.
///
/// Returns an empty Vec on connect / list failures so callers can degrade
/// to "no schemas visible" instead of panicking on a transient blip.
pub async fn flight_list_schemas(endpoint: &str, _catalog: &str) -> Vec<TrinoSchema> {
    use arrow_flight::Criteria;
    use arrow_flight::flight_service_client::FlightServiceClient;
    use futures::StreamExt;
    use std::collections::BTreeMap;

    // Normalize `grpc://host:port` → `http://host:port` so tonic's HTTP/2
    // transport accepts it. tonic's `Channel::from_shared` rejects the
    // `grpc://` scheme even though the wire protocol is gRPC-over-HTTP/2.
    let dial = endpoint
        .strip_prefix("grpc://")
        .map(|rest| format!("http://{rest}"))
        .unwrap_or_else(|| endpoint.to_string());

    // tonic ≥0.12 removed the convenience `connect` constructor on
    // codegen client types; build the Channel explicitly and hand
    // it to `FlightServiceClient::new`. Behavior unchanged from
    // the previous `connect()` form.
    let channel = match tonic::transport::Channel::from_shared(dial)
        .ok()
        .map(|e| e.connect_lazy())
    {
        Some(ch) => ch,
        None => return Vec::new(),
    };
    let mut client = FlightServiceClient::new(channel);

    let mut stream = match client
        .list_flights(tonic::Request::new(Criteria {
            expression: Default::default(),
        }))
        .await
    {
        Ok(resp) => resp.into_inner(),
        Err(_) => return Vec::new(),
    };

    // Bucket each FlightInfo by its first descriptor-path segment. Tables
    // accumulate under the same schema name; preserves insertion order
    // via BTreeMap.
    let mut by_schema: BTreeMap<String, Vec<String>> = BTreeMap::new();
    while let Some(item) = stream.next().await {
        let Ok(flight_info) = item else { continue };
        let Some(desc) = flight_info.flight_descriptor else {
            continue;
        };
        let mut path_iter = desc.path.into_iter();
        let Some(schema_name) = path_iter.next() else {
            continue;
        };
        let table_name = path_iter.next().unwrap_or_default();
        let entry = by_schema.entry(schema_name).or_default();
        if !table_name.is_empty() && !entry.contains(&table_name) {
            entry.push(table_name);
        }
    }

    by_schema
        .into_iter()
        .map(|(name, tables)| TrinoSchema { name, tables })
        .collect()
}

/// Live Arrow Flight `ListFlights` query that returns the table names
/// under a given schema. Same wire as [`flight_list_schemas`] but
/// filters by `path[0] == schema` and returns the per-schema table
/// list as `Vec<String>` (deduplicated, insertion-ordered).
///
/// Empty Vec on connect / list failures (no panic).
///
/// TD-098 progress: 2/7 live (was 1/7 after the `flight_list_schemas`
/// pilot).
pub async fn flight_list_tables(endpoint: &str, _catalog: &str, schema: &str) -> Vec<String> {
    use arrow_flight::Criteria;
    use arrow_flight::flight_service_client::FlightServiceClient;
    use futures::StreamExt;

    let dial = endpoint
        .strip_prefix("grpc://")
        .map(|rest| format!("http://{rest}"))
        .unwrap_or_else(|| endpoint.to_string());

    // tonic ≥0.12 removed the convenience `FlightServiceClient::connect`;
    // build the Channel via `Channel::from_shared(...).connect_lazy()`
    // and hand it to `FlightServiceClient::new`. Same pattern as
    // `flight_list_schemas` above.
    let channel = match tonic::transport::Channel::from_shared(dial)
        .ok()
        .map(|e| e.connect_lazy())
    {
        Some(ch) => ch,
        None => return Vec::new(),
    };
    let mut client = FlightServiceClient::new(channel);

    let mut stream = match client
        .list_flights(tonic::Request::new(Criteria {
            expression: Default::default(),
        }))
        .await
    {
        Ok(resp) => resp.into_inner(),
        Err(_) => return Vec::new(),
    };

    let mut tables: Vec<String> = Vec::new();
    while let Some(item) = stream.next().await {
        let Ok(flight_info) = item else { continue };
        let Some(desc) = flight_info.flight_descriptor else {
            continue;
        };
        let mut path_iter = desc.path.into_iter();
        let Some(schema_name) = path_iter.next() else {
            continue;
        };
        if schema_name != schema {
            continue;
        }
        let Some(table_name) = path_iter.next() else {
            continue;
        };
        if !tables.iter().any(|t| t == &table_name) {
            tables.push(table_name);
        }
    }
    tables
}

/// Live Arrow Flight `GetSchema` query for a `catalog.schema.table`
/// triple. Third live Flight method (TD-098, 2026-06-01).
///
/// Dials the ProximaDB Flight endpoint, calls `get_schema` with a
/// `FlightDescriptor::path([schema, table])`, and decodes the IPC
/// schema bytes back into an `ArrowSchema` via
/// `arrow_flight::SchemaResult::try_into`. Same dial pattern as
/// `flight_list_schemas` / `flight_list_tables` —
/// `Channel::from_shared(...).connect_lazy()` +
/// `FlightServiceClient::new(channel)` (tonic ≥0.12 removed the
/// convenience `connect()` shorthand).
///
/// `catalog` is currently informational — Flight descriptors are
/// `[schema, table]` two-segment paths; the catalog is implicit in the
/// Flight endpoint binding. Connect / RPC / decode failures degrade
/// to `None` so callers see "schema unknown" instead of panicking on
/// transient blips.
pub async fn flight_get_table_schema(
    endpoint: &str,
    _catalog: &str,
    schema: &str,
    table: &str,
) -> Option<Arc<ArrowSchema>> {
    use arrow_flight::FlightDescriptor;
    use arrow_flight::SchemaResult;
    use arrow_flight::flight_service_client::FlightServiceClient;

    let dial = endpoint
        .strip_prefix("grpc://")
        .map(|rest| format!("http://{rest}"))
        .unwrap_or_else(|| endpoint.to_string());

    let channel = match tonic::transport::Channel::from_shared(dial)
        .ok()
        .map(|e| e.connect_lazy())
    {
        Some(ch) => ch,
        None => return None,
    };
    let mut client = FlightServiceClient::new(channel);

    let descriptor = FlightDescriptor::new_path(vec![schema.to_string(), table.to_string()]);
    let result: SchemaResult = match client.get_schema(tonic::Request::new(descriptor)).await {
        Ok(resp) => resp.into_inner(),
        Err(_) => return None,
    };

    match ArrowSchema::try_from(&result) {
        Ok(s) => Some(Arc::new(s)),
        Err(_) => None,
    }
}

/// Live Arrow Flight `GetFlightInfo` query that materializes the
/// `TrinoSplit` set for a `catalog.schema.table` triple. Fourth live
/// Flight method (TD-098, 2026-06-01).
///
/// Dials the ProximaDB Flight endpoint, calls `get_flight_info` with a
/// `FlightDescriptor` carrying `path=[schema, table]` and `cmd =
/// JSON-encoded constraint` (the constraint is informational over the
/// wire today; the ProximaDB Flight server is expected to apply it
/// when generating endpoints), then decodes each
/// `FlightInfo.endpoint[i].ticket.ticket` bytes back into a
/// `TrinoSplit` via serde-json. Same dial pattern as the prior three
/// live methods — `Channel::from_shared(...).connect_lazy()` +
/// `FlightServiceClient::new(channel)`.
///
/// `catalog` is informational (Flight descriptors are two-segment
/// `[schema, table]` paths; the catalog is implicit in the endpoint
/// binding). Connect / RPC / decode failures degrade to an empty Vec.
pub async fn flight_get_splits(
    endpoint: &str,
    _catalog: &str,
    schema: &str,
    table: &str,
    constraint: &TrinoTupleDomain,
) -> Vec<TrinoSplit> {
    use arrow_flight::FlightDescriptor;
    use arrow_flight::flight_service_client::FlightServiceClient;

    let dial = endpoint
        .strip_prefix("grpc://")
        .map(|rest| format!("http://{rest}"))
        .unwrap_or_else(|| endpoint.to_string());

    let channel = match tonic::transport::Channel::from_shared(dial)
        .ok()
        .map(|e| e.connect_lazy())
    {
        Some(ch) => ch,
        None => return Vec::new(),
    };
    let mut client = FlightServiceClient::new(channel);

    // FlightDescriptor type is PATH (path is set); cmd carries the
    // JSON-encoded predicate so the server can do pushdown without a
    // separate RPC. Tonic serialization is lossless for both fields.
    let cmd_bytes = serde_json::to_vec(constraint).unwrap_or_default();
    let mut descriptor = FlightDescriptor::new_path(vec![schema.to_string(), table.to_string()]);
    descriptor.cmd = cmd_bytes.into();

    let info = match client
        .get_flight_info(tonic::Request::new(descriptor))
        .await
    {
        Ok(resp) => resp.into_inner(),
        Err(_) => return Vec::new(),
    };

    let mut splits = Vec::with_capacity(info.endpoint.len());
    for ep in info.endpoint {
        let Some(ticket) = ep.ticket else { continue };
        match serde_json::from_slice::<TrinoSplit>(&ticket.ticket) {
            Ok(split) => splits.push(split),
            Err(_) => continue,
        }
    }
    splits
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trino_config_default() {
        let config = TrinoConnectorConfig::default();
        assert_eq!(config.flight_endpoint, "grpc://localhost:5680");
        assert!(config.enable_predicate_pushdown);
    }

    #[test]
    fn test_trino_tuple_domain() {
        let domain = TrinoTupleDomain::all();
        assert!(domain.is_all);
        assert!(!domain.is_none);

        let domain = TrinoTupleDomain::none();
        assert!(!domain.is_all);
        assert!(domain.is_none);
    }

    #[test]
    fn test_trino_range() {
        let eq = TrinoRange::equal(serde_json::json!(42));
        assert!(eq.low_inclusive);
        assert!(eq.high_inclusive);

        let gt = TrinoRange::greater_than(serde_json::json!(10), false);
        assert!(!gt.low_inclusive);
        assert!(gt.high.is_none());
    }

    #[test]
    fn test_trino_split_from_file_split() {
        let file_split = FileSplit {
            split_id: "test:0".to_string(),
            file_path: "/data/file.sst".to_string(),
            offset: 0,
            length: 1024,
            split_type: crate::storage::formats::SplitType::Block {
                block_id: 0,
                record_count: 100,
            },
            statistics: SplitStatistics::default(),
            locality: crate::storage::formats::SplitLocality::default(),
            object_size: None,
        };

        let trino_split = TrinoSplit::from_file_split(
            "proximadb".to_string(),
            "default".to_string(),
            "vectors".to_string(),
            file_split,
        );

        assert_eq!(trino_split.split_id, "test:0");
        assert_eq!(trino_split.catalog, "proximadb");
        assert!(trino_split.remotely_accessible);
    }

    #[tokio::test]
    async fn test_trino_page_source() {
        let file_split = FileSplit {
            split_id: "test:0".to_string(),
            file_path: String::new(),
            offset: 0,
            length: 0,
            split_type: crate::storage::formats::SplitType::ByteRange {
                estimated_records: 0,
            },
            statistics: SplitStatistics::default(),
            locality: crate::storage::formats::SplitLocality::default(),
            object_size: None,
        };

        let split = TrinoSplit::from_file_split(
            "proximadb".to_string(),
            "default".to_string(),
            "test".to_string(),
            file_split,
        );

        // Unreachable endpoint → constructor marks source as
        // `finished` immediately (connect_lazy + first do_get fails
        // when the test runs in an env with no Flight server on
        // this port). Test pins that contract: no panic, no
        // pages returned, source ends up finished.
        let mut source =
            TrinoPageSource::new("grpc://127.0.0.1:1", split, Arc::new(ArrowSchema::empty())).await;
        // First call should return None and mark finished
        assert!(source.get_next_page().await.is_none());
        assert!(source.is_finished());
    }
}
