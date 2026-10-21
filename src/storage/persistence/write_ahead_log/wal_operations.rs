/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Unified WAL Operations
//!
//! Extends the WAL system to handle both vector and graph operations,
//! enabling atomic hybrid transactions across both data types.

use crate::proto::proximadb_v1::{DocumentUpdate, LogEntry, MetricSample};
use crate::storage::document::DocumentRecord;
use crate::storage::memtable::implementations::graph_memtable::GraphOperation;
use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb_graph_model::{GraphWalEntry, GraphWalRecord};
use proximadb_records::ProximaRecord;
use proximadb_storage_ports::{
    CanonicalWalReaderPort, FilesystemPort, GraphWalFactory, GraphWalPort, GraphWalReaderPort,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Unified WAL operation supporting vector, graph, document, and observability operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnifiedWALOperation {
    /// Vector operation (existing)
    VectorOp(VectorOperation),

    /// Graph operation
    GraphOp(GraphOperation),

    /// Document operation
    DocumentOp(DocumentOperation),

    /// Observability operation (logs, metrics, traces)
    ObservabilityOp(ObservabilityOperation),

    /// Hybrid operation combining vector and graph operations
    HybridOp {
        vector_ops: Vec<VectorOperation>,
        graph_ops: Vec<GraphOperation>,
        /// Ensures atomicity across both operation types
        transaction_id: String,
    },

    /// Batch operation for performance
    BatchOp {
        operations: Vec<UnifiedWALOperation>,
        /// Optional batch identifier
        batch_id: Option<String>,
    },

    /// Time-series operation
    TimeSeriesOp(TimeSeriesOperation),

    /// Checkpoint operation for recovery
    Checkpoint {
        sequence_number: u64,
        timestamp_ms: u64,
        /// Collections/graphs/document-collections/observability namespaces included in checkpoint
        collections: Vec<String>,
        graphs: Vec<String>,
        document_collections: Vec<String>,
        observability_namespaces: Vec<String>,
    },

    /// TD-066 (c) Part 2: a non-data *marker* frame correlating the engine WAL
    /// with the canonical WAL's checkpoint LSN. Written once by `flush_wal`
    /// right after the canonical `Checkpoint` is emitted. Replay uses the
    /// latest marker whose LSN ≤ the recovered canonical checkpoint LSN as the
    /// truncation point (frames at/before it are already covered by the durable
    /// canonical snapshot). Appended LAST so existing variants keep their
    /// bincode discriminants — old segments decode unchanged (mixed-read-safe);
    /// emission is feature-gated default-OFF, so old binaries never read these
    /// frames until the fleet is upgraded.
    GraphMarker(MarkerKind),
}

/// Non-data marker kinds carried in [`UnifiedWALOperation::GraphMarker`].
///
/// Definition moved to the `proximadb-graph-model` foundation leaf (named by
/// the `GraphWalPort` contract + the ORION graph engine); re-exported here so
/// existing `crate::storage::persistence::write_ahead_log::...::MarkerKind`
/// references resolve unchanged.
pub use proximadb_graph_model::MarkerKind;

/// TD-066 (c) Part 2 feature gate (default OFF per the storage-format-migration
/// mandate). When enabled, `flush_wal` emits `CanonicalEmission` marker frames
/// and `replay_wal` truncates engine-WAL replay at the recovered canonical
/// checkpoint LSN. When disabled (default), recovery behavior is unchanged
/// (full replay) and no marker frames are written — old binaries/fleets are
/// unaffected. Flip fleet-wide after baking.
pub fn canonical_replay_scope_enabled() -> bool {
    std::env::var("PROXIMADB_GRAPH_CANONICAL_REPLAY_SCOPE").as_deref() == Ok("1")
}

/// Graph WAL segment size in bytes (`PROXIMADB_GRAPH_WAL_SEGMENT_MB`, default
/// 64, clamped to [1, 64]). Truncation-through-checkpoint deletes only whole
/// segments strictly below the marker's segment, so a WAL smaller than one
/// segment can never be reclaimed — embedded profiles set this to 8 so a
/// modest graph WAL spans several segments and checkpointing reclaims the
/// bulk. Server default is unchanged (64 = the previous hardcoded value).
pub fn graph_wal_segment_size_bytes() -> usize {
    const DEFAULT_MB: usize = 64;
    let mb = std::env::var("PROXIMADB_GRAPH_WAL_SEGMENT_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MB)
        .clamp(1, DEFAULT_MB);
    mb * 1024 * 1024
}

/// TD-066 (c) Part 2 — canonical-store recovery re-population (default OFF).
/// Graph recovery replays the engine WAL into the in-memory engine only; the
/// canonical/cold record store (a write-time projection, and a *buffered* one for
/// `ColdGraphSegmentStore`) is NOT rebuilt — an unflushed buffer is lost on crash.
/// When enabled, recovery re-drives every recovered node/edge through the
/// canonical store (idempotent upsert) so the cold store is rebuilt from the
/// authoritative recovered state — closing the data-loss path that blocks wiring
/// `ColdGraphSegmentStore` to production. Aligns graph durability to ADR-020
/// (canonical store rebuildable from recovery). Flip fleet-wide after baking.
pub fn canonical_recovery_repopulate_enabled() -> bool {
    std::env::var("PROXIMADB_GRAPH_CANONICAL_RECOVERY").as_deref() == Ok("1")
}

/// Parse the segment index out of a `wal_{:08}.log` segment filename. Accepts
/// either a bare filename or a full path (takes the last path component).
/// Returns `None` for anything that isn't a WAL segment file.
///
/// TD-066 (d): segment discovery (writer resume, reader replay, and LSN-bounded
/// truncation) keys off this so the three paths agree on which files are
/// segments and what their indices are — the basis for surviving a
/// non-contiguous (post-truncation) segment layout.
fn parse_wal_segment_index(name: &str) -> Option<u32> {
    let file = name.rsplit('/').next()?;
    file.strip_prefix("wal_")?
        .strip_suffix(".log")?
        .parse::<u32>()
        .ok()
}

/// Time-series operations for WAL
//
// `InsertRecord` carries a full ProximaRecord; the other variants are
// small. Boxing would force every write-path caller to allocate even
// when buffered in batch Vecs (where the per-variant size is
// amortised). WAL ops are serialised on the durable path — heap
// layout is dominated by the record payload, not the enum tag.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSeriesOperation {
    /// Insert a time-series record into a partition
    InsertRecord {
        collection_id: String,
        /// Timestamp in milliseconds since Unix epoch
        timestamp_ms: i64,
        record: ProximaRecord,
    },
    /// Insert an OHLC bar
    InsertOHLC {
        collection_id: String,
        symbol: String,
        /// Timestamp in milliseconds since Unix epoch
        timestamp_ms: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: i64,
    },
    /// Create a new time partition
    CreatePartition {
        collection_id: String,
        /// Partition start timestamp in milliseconds since Unix epoch
        partition_key_ms: i64,
    },
    /// Drop a time partition
    DropPartition {
        collection_id: String,
        /// Partition start timestamp in milliseconds since Unix epoch
        partition_key_ms: i64,
    },
}

/// Document operations for WAL
//
// `InsertDocument` carries a full DocumentRecord; smaller variants
// describe drops/edits/projections. Boxing every variant would force
// per-call heap allocation on the hot write path. WAL ops are framed
// for durable serialisation — the document payload itself dominates
// the byte budget.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentOperation {
    /// Insert or update a document
    InsertDocument {
        collection_id: String,
        document: DocumentRecord,
    },
    /// Insert or update a document as canonical durable record state.
    ///
    /// This is the Phase 2 document rebase WAL shape from
    /// `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`.
    /// The document facade may still rebuild legacy `DocumentRecord`
    /// projections during recovery, but durable intent is expressed as a
    /// `ProximaRecord`.
    UpsertCanonicalDocumentRecord {
        collection_id: String,
        record: ProximaRecord,
    },
    /// Update a document with patch operations
    UpdateDocument {
        collection_id: String,
        document_id: String,
        updates: Vec<DocumentUpdate>,
        new_version: u64,
    },
    /// Delete a document
    DeleteDocument {
        collection_id: String,
        document_id: String,
    },
    /// Delete a document by canonical record identity.
    DeleteCanonicalDocumentRecord {
        collection_id: String,
        document_id: String,
        record_oid: String,
    },
    /// Batch insert documents
    BatchDocuments {
        collection_id: String,
        documents: Vec<DocumentRecord>,
    },
    /// Create document collection
    CreateCollection {
        collection_id: String,
        config_json: String, // Serialized DocumentCollectionConfig
    },
    /// Delete document collection
    DeleteCollection { collection_id: String },
}

/// Observability operations for WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObservabilityOperation {
    /// Write a log entry
    WriteLog { namespace: String, log: LogEntry },
    /// Write a batch of logs
    WriteLogs {
        namespace: String,
        logs: Vec<LogEntry>,
    },
    /// Write a metric sample
    WriteMetric {
        namespace: String,
        metric: MetricSample,
    },
    /// Write a batch of metrics
    WriteMetrics {
        namespace: String,
        metrics: Vec<MetricSample>,
    },
    /// Write a trace span (serialized as JSON for flexibility)
    WriteSpan {
        namespace: String,
        span_json: String,
    },
    /// Create observability namespace
    CreateNamespace {
        namespace: String,
        config_json: String,
    },
    /// Delete observability namespace
    DeleteNamespace { namespace: String },
}

/// Vector operations (existing functionality)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorOperation {
    AddVector {
        collection_id: String,
        record: ProximaRecord,
    },
    UpdateVector {
        collection_id: String,
        record: ProximaRecord,
    },
    DeleteVector {
        collection_id: String,
        vector_id: String,
    },
    BatchVectors {
        collection_id: String,
        records: Vec<ProximaRecord>,
    },
}

/// WAL entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedWALEntry {
    /// Unique sequence number
    pub sequence_number: u64,

    /// Operation to apply
    pub operation: UnifiedWALOperation,

    /// Timestamp of the operation (milliseconds since Unix epoch)
    pub timestamp_ms: u64,

    /// CRC32 checksum for integrity
    pub checksum: u32,

    /// Optional metadata
    pub metadata: Option<WALEntryMetadata>,
}

/// Metadata for WAL entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WALEntryMetadata {
    /// Client identifier
    pub client_id: Option<String>,

    /// Request ID for tracing
    pub request_id: Option<String>,

    /// Whether this operation requires fsync
    pub requires_fsync: bool,

    /// Priority for recovery ordering
    pub priority: i32,
}

impl UnifiedWALEntry {
    /// Create a new WAL entry
    pub fn new(sequence_number: u64, operation: UnifiedWALOperation) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        let checksum = Self::calculate_checksum(&operation);

        Self {
            sequence_number,
            operation,
            timestamp_ms,
            checksum,
            metadata: None,
        }
    }

    /// Calculate a stable operation checksum.
    fn calculate_checksum(operation: &UnifiedWALOperation) -> u32 {
        let serialized = Self::canonical_operation_bytes(operation)
            .or_else(|| bincode::serialize(operation).ok())
            .unwrap_or_default();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        serialized.hash(&mut hasher);
        hasher.finish() as u32
    }

    fn canonical_operation_bytes(operation: &UnifiedWALOperation) -> Option<Vec<u8>> {
        let value = serde_json::to_value(operation).ok()?;
        let canonical = Self::canonical_json_value(value);
        serde_json::to_vec(&canonical).ok()
    }

    fn canonical_json_value(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Array(items) => serde_json::Value::Array(
                items.into_iter().map(Self::canonical_json_value).collect(),
            ),
            serde_json::Value::Object(map) => {
                let mut entries = map.into_iter().collect::<Vec<_>>();
                entries.sort_by(|left, right| left.0.cmp(&right.0));

                let mut canonical = serde_json::Map::new();
                for (key, value) in entries {
                    canonical.insert(key, Self::canonical_json_value(value));
                }
                serde_json::Value::Object(canonical)
            }
            scalar => scalar,
        }
    }

    /// Verify checksum integrity
    pub fn verify_checksum(&self) -> bool {
        let calculated = Self::calculate_checksum(&self.operation);
        calculated == self.checksum
    }

    /// Check if this is a graph operation
    pub fn is_graph_operation(&self) -> bool {
        matches!(
            &self.operation,
            UnifiedWALOperation::GraphOp(_) | UnifiedWALOperation::HybridOp { .. }
        )
    }

    /// Check if this is a vector operation
    pub fn is_vector_operation(&self) -> bool {
        matches!(
            &self.operation,
            UnifiedWALOperation::VectorOp(_) | UnifiedWALOperation::HybridOp { .. }
        )
    }

    /// Check if this is a document operation
    pub fn is_document_operation(&self) -> bool {
        matches!(&self.operation, UnifiedWALOperation::DocumentOp(_))
    }

    /// Check if this is an observability operation
    pub fn is_observability_operation(&self) -> bool {
        matches!(&self.operation, UnifiedWALOperation::ObservabilityOp(_))
    }

    /// Check if this is a time-series operation
    pub fn is_timeseries_operation(&self) -> bool {
        matches!(&self.operation, UnifiedWALOperation::TimeSeriesOp(_))
    }
}

/// WAL writer extension for unified operations
pub struct UnifiedWALWriter {
    /// Base path for WAL files
    base_path: String,

    /// Current sequence number
    sequence_number: std::sync::atomic::AtomicU64,

    /// Filesystem factory for file operations
    filesystem: Arc<FilesystemFactory>,

    /// Current segment path
    current_segment_path: Option<String>,

    /// Current segment data buffer
    current_segment_data: Vec<u8>,

    /// Maximum size per WAL segment
    max_segment_size: usize,

    /// Segment counter
    segment_counter: u32,
}

impl UnifiedWALWriter {
    /// Create a new unified WAL writer
    pub async fn new(base_path: String) -> anyhow::Result<Self> {
        // Create filesystem factory with default config
        let filesystem = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create filesystem: {}", e))?,
        );

        // Ensure base directory exists. If caller passed a URL, don't prefix again.
        let base_url = if base_path.contains("://") {
            base_path.clone()
        } else {
            format!("file://{}", base_path)
        };
        let fs = filesystem.get_filesystem(&base_url)?;
        fs.create_dir_all(&base_url).await?;

        // Discover existing WAL files to resume from max sequence number and
        // the highest existing segment index.
        let mut next_seq: u64 = 0;
        let mut segment_count: u64 = 0;
        // TD-066 (d): track the HIGHEST existing segment index, not just the
        // count. After an LSN-bounded prefix truncation the surviving segments
        // are non-contiguous (e.g. [3, 4]); resuming from `count` (= 2) would
        // make the next rotation reopen and clobber a kept segment. `max + 1`
        // is collision-free for both the contiguous and post-truncation layouts.
        let mut max_segment_index: Option<u32> = None;
        if let Ok(files) = fs.list(&base_url).await {
            for file_info in &files {
                if let Some(name) = file_info.name.split('/').next_back()
                    && name.starts_with("wal_")
                {
                    segment_count += 1;
                    // Resume the segment index from the `wal_{:08}.log` layout
                    // actually written by `open_new_segment`.
                    if let Some(idx) = parse_wal_segment_index(name) {
                        max_segment_index = Some(max_segment_index.map_or(idx, |cur| cur.max(idx)));
                    }
                    // Legacy `wal_YYYYMMDD_HHMMSS_{min_seq}_{max_seq}_{uuid}`
                    // layout: extract max sequence (field 3, 0-indexed).
                    let parts: Vec<&str> = name.split('_').collect();
                    if parts.len() >= 4
                        && let Ok(seq) = parts[3].parse::<u64>()
                    {
                        next_seq = next_seq.max(seq.saturating_add(1));
                    }
                }
            }
        }

        // The current `wal_{segment}.log` filename carries no sequence range.
        // Recover the allocator from frame contents; otherwise every reopen
        // restarts at zero and duplicates sequence numbers within one WAL.
        if max_segment_index.is_some() {
            let reader = UnifiedWALReader::new(base_url.clone()).await?;
            if let Some(last_seq) = reader
                .read_all()
                .await?
                .into_iter()
                .map(|entry| entry.sequence_number)
                .max()
            {
                next_seq = next_seq.max(last_seq.saturating_add(1));
            }
        }

        // Next segment to open: one past the highest existing index (0 when fresh).
        let segment_counter = max_segment_index.map_or(0, |idx| idx + 1);

        if segment_count > 0 {
            tracing::info!(
                "WAL recovery: found {} segments (next index {}), resuming from sequence {}",
                segment_count,
                segment_counter,
                next_seq
            );
        } else {
            tracing::debug!("WAL writer initialized fresh for path: {}", base_path);
        }

        Ok(Self {
            // Store the SCHEME-QUALIFIED url, never the bare path: every
            // segment path is joined from this, and no site below may
            // re-prepend `file://` — on an object-store base that yields
            // invalid `file://s3://…` URLs (TD-OBJSTORE-1, #960).
            base_path: base_url,
            sequence_number: std::sync::atomic::AtomicU64::new(next_seq),
            filesystem,
            current_segment_path: None,
            current_segment_data: Vec::new(),
            max_segment_size: graph_wal_segment_size_bytes(),
            segment_counter,
        })
    }

    /// Append an operation to the WAL
    pub async fn append(&mut self, operation: UnifiedWALOperation) -> anyhow::Result<u64> {
        use std::sync::atomic::Ordering;

        // fetch_add returns the old value before incrementing
        // So first call returns 0, then increments to 1
        let seq = self.sequence_number.fetch_add(1, Ordering::SeqCst);
        let entry = UnifiedWALEntry::new(seq, operation);

        // Serialize the entry
        let serialized = bincode::serialize(&entry)?;
        let size = serialized.len();

        tracing::debug!(
            "WAL append: seq={}, size={} bytes, checksum={}",
            seq,
            size,
            entry.checksum
        );

        // Check if we need to rotate the segment
        if self.current_segment_data.len() + size + 4 > self.max_segment_size {
            self.rotate_segment().await?;
        }

        // If no current segment, create one
        if self.current_segment_path.is_none() {
            self.open_new_segment().await?;
        }

        // Append to buffer
        // Write size header (4 bytes)
        self.current_segment_data
            .extend_from_slice(&(size as u32).to_le_bytes());
        // Write serialized entry
        self.current_segment_data.extend_from_slice(&serialized);

        // If requires immediate fsync, flush to disk
        if entry.metadata.as_ref().is_some_and(|m| m.requires_fsync) {
            self.flush_current_segment().await?;
        }

        Ok(seq)
    }

    /// Flush any buffered WAL entries to disk
    /// This should be called before shutdown to ensure durability
    pub async fn flush(&mut self) -> anyhow::Result<()> {
        self.flush_current_segment().await
    }

    /// Rotate to a new WAL segment
    async fn rotate_segment(&mut self) -> anyhow::Result<()> {
        // Flush current segment if exists
        if self.current_segment_path.is_some() {
            self.flush_current_segment().await?;
        }

        self.segment_counter += 1;
        self.current_segment_data.clear();
        self.open_new_segment().await?;

        Ok(())
    }

    /// Flush current segment to disk
    async fn flush_current_segment(&mut self) -> anyhow::Result<()> {
        if let Some(ref path) = self.current_segment_path
            && !self.current_segment_data.is_empty()
        {
            // `path` is already scheme-qualified (joined from the normalized
            // base_path by `open_new_segment`).
            let url = path.clone();
            let fs = self.filesystem.get_filesystem(&url)?;

            if self.base_path.starts_with("file://") {
                // Local: WAL segments are append-only. Avoid
                // read-modify-write here: cached reads can lag behind recent
                // writes, and rewriting the segment risks dropping entries
                // that were already durable.
                fs.append(&url, &self.current_segment_data).await?;
                fs.sync_file(&url).await?;

                // Clear buffer after successful write
                self.current_segment_data.clear();
            } else {
                // Object store (s3://, adls://, …): block/immutable blobs
                // reject append, so the buffer holds the WHOLE segment
                // (bounded by max_segment_size and cleared on rotation) and
                // each flush overwrites the segment object with the full
                // contents. The byte layout is identical to the appended
                // local segment, so recovery reads both the same way
                // (TD-OBJSTORE-1, #960).
                let options = crate::storage::persistence::filesystem::FileOptions {
                    create_dirs: true,
                    overwrite: true,
                    ..Default::default()
                };
                fs.write(&url, &self.current_segment_data, Some(options))
                    .await?;
                // Buffer intentionally NOT cleared: it is the durable
                // segment image until rotation.
            }
        }
        Ok(())
    }

    /// Open a new WAL segment file
    async fn open_new_segment(&mut self) -> anyhow::Result<()> {
        let filename = format!("{}/wal_{:08}.log", self.base_path, self.segment_counter);
        self.current_segment_path = Some(filename.clone());
        self.current_segment_data.clear();

        // Ensure the file exists. `filename` is already scheme-qualified
        // (base_path is normalized in `new`).
        let url = filename;
        let fs = self.filesystem.get_filesystem(&url)?;
        if !fs.exists(&url).await? {
            // Create empty file
            fs.write(&url, &[], None).await?;
        }

        Ok(())
    }

    /// Sync all pending writes
    pub async fn sync(&mut self) -> anyhow::Result<()> {
        // Flush any pending data to disk
        self.flush_current_segment().await?;
        Ok(())
    }

    /// TD-066 (d): reclaim WAL segments fully covered by a durable canonical
    /// snapshot. Deletes every whole segment whose frames all precede the
    /// `CanonicalEmission(lsn)` marker; KEEPS the marker's own segment and all
    /// later segments (they carry the marker that `replay_wal` searches for plus
    /// every post-checkpoint frame). Returns the number of segments reclaimed.
    ///
    /// Crash-safe by construction:
    ///  * Callers MUST invoke this only AFTER the snapshot at `lsn` is durable
    ///    (see `GraphOperationsService::flush_wal`), so every deleted frame is
    ///    already covered by a durable snapshot — a crash after this point still
    ///    recovers `snapshot + surviving post-marker frames`.
    ///  * Segments are deleted LOWEST-index first, so a crash mid-truncation
    ///    always leaves a contiguous suffix `[k..=N]` with the marker segment
    ///    intact — recovery still finds the marker and replays only post-marker
    ///    frames. Each segment is one file and `delete` is atomic, so there is
    ///    no torn-segment state.
    ///
    /// No-op (returns 0) when the marker for `lsn` isn't found, so a stale or
    /// not-yet-written checkpoint can never delete live frames.
    pub async fn truncate_through_canonical_marker(&mut self, lsn: u64) -> anyhow::Result<u64> {
        // Ensure the marker frame itself (appended just before the snapshot) is
        // durable on disk before we scan for it or delete anything.
        self.flush().await?;

        let base_url = if self.base_path.contains("://") {
            self.base_path.clone()
        } else {
            format!("file://{}", self.base_path)
        };
        let fs = self.filesystem.get_filesystem(&base_url)?;

        // Enumerate surviving segments in ascending index order.
        let mut indices: Vec<u32> = Vec::new();
        if let Ok(files) = fs.list(&base_url).await {
            for file_info in &files {
                if let Some(idx) = parse_wal_segment_index(&file_info.name) {
                    indices.push(idx);
                }
            }
        }
        indices.sort_unstable();

        // Find the segment that contains `CanonicalEmission(lsn)`. There is at
        // most one such marker (one per checkpoint), so any scan order is
        // correct; ascending keeps it simple and the lower segments are read
        // exactly once before they are deleted.
        let reader = UnifiedWALReader::new(self.base_path.clone()).await?;
        let mut marker_segment: Option<u32> = None;
        for &seg in &indices {
            let entries = reader.read_segment(seg).await?;
            let has_marker = entries.iter().any(|entry| {
                matches!(
                    &entry.operation,
                    UnifiedWALOperation::GraphMarker(MarkerKind::CanonicalEmission(marker_lsn))
                        if *marker_lsn == lsn
                )
            });
            if has_marker {
                marker_segment = Some(seg);
                break;
            }
        }

        let Some(marker_segment) = marker_segment else {
            tracing::debug!(
                lsn,
                "TD-066 (d): no CanonicalEmission marker found; WAL truncation skipped"
            );
            return Ok(0);
        };

        // Delete every segment strictly below the marker's segment, lowest
        // first. The marker's segment and everything above it are kept.
        let mut reclaimed = 0u64;
        for seg in indices.into_iter().filter(|&s| s < marker_segment) {
            let url = format!("{}/wal_{:08}.log", base_url, seg);
            fs.delete(&url).await?;
            reclaimed += 1;
            tracing::debug!(segment = seg, lsn, "TD-066 (d): reclaimed WAL segment");
        }

        // TD-WAL-1 S6: observe segments reclaimed (operator aggregate; per-tenant
        // durable attribution lives in the io_trace warehouse, ADR-066).
        crate::metrics::wal_flush_metrics::inc_truncation_segments_reclaimed(reclaimed);
        Ok(reclaimed)
    }

    /// Lower the per-segment size cap so tests can force segment rotation
    /// without writing 64 MB of data.
    #[cfg(test)]
    pub fn set_max_segment_size_for_test(&mut self, bytes: usize) {
        self.max_segment_size = bytes;
    }
}

/// The unified WAL writer is the composition-root-provided sink for the ORION
/// graph engine: it implements [`GraphWalPort`] by wrapping graph operations /
/// markers into the unified operation enum, so ORION can append through the
/// port without naming the concrete writer or the unified operation type.
#[async_trait::async_trait]
impl GraphWalPort for UnifiedWALWriter {
    async fn append_graph_op(&mut self, op: GraphOperation) -> anyhow::Result<u64> {
        self.append(UnifiedWALOperation::GraphOp(op)).await
    }

    async fn append_graph_marker(&mut self, marker: MarkerKind) -> anyhow::Result<u64> {
        self.append(UnifiedWALOperation::GraphMarker(marker)).await
    }

    async fn flush(&mut self) -> anyhow::Result<()> {
        UnifiedWALWriter::flush(self).await
    }

    async fn truncate_through_canonical_marker(
        &mut self,
        checkpoint_lsn: u64,
    ) -> anyhow::Result<u64> {
        UnifiedWALWriter::truncate_through_canonical_marker(self, checkpoint_lsn).await
    }
}

/// The unified WAL reader is the composition-root-provided recovery source for
/// the ORION graph engine: it implements [`GraphWalReaderPort`] by projecting
/// graph operations / markers out of the unified entry stream, so ORION can
/// replay through the port without naming the concrete reader or the unified
/// operation/entry types.
#[async_trait::async_trait]
impl GraphWalReaderPort for UnifiedWALReader {
    async fn read_all_graph(&self) -> anyhow::Result<Vec<GraphWalEntry>> {
        let entries = self.read_all().await?;
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            let record = match entry.operation {
                UnifiedWALOperation::GraphOp(op) => GraphWalRecord::Op(Box::new(op)),
                UnifiedWALOperation::GraphMarker(marker) => GraphWalRecord::Marker(marker),
                // Non-graph unified ops (Document/Observability/Hybrid/…) are
                // not part of a graph engine's stream — filtered out.
                _ => continue,
            };
            out.push(GraphWalEntry {
                sequence_number: entry.sequence_number,
                record,
            });
        }
        Ok(out)
    }
}

/// Composition-root factory for a graph engine's WAL writer + reader: the single
/// place the concrete `UnifiedWALWriter` / `UnifiedWALReader` are named, so the
/// ORION engine can obtain its [`GraphWalPort`] / [`GraphWalReaderPort`] through
/// the [`GraphWalFactory`] port without naming these types itself.
#[derive(Default)]
pub struct UnifiedWalFactory;

#[async_trait::async_trait]
impl GraphWalFactory for UnifiedWalFactory {
    async fn make_writer(
        &self,
        wal_path: &str,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<dyn GraphWalPort>>> {
        let writer = UnifiedWALWriter::new(wal_path.to_string()).await?;
        let writer: Arc<tokio::sync::Mutex<dyn GraphWalPort>> =
            Arc::new(tokio::sync::Mutex::new(writer));
        Ok(writer)
    }

    async fn make_reader(&self, wal_path: &str) -> anyhow::Result<Arc<dyn GraphWalReaderPort>> {
        let reader = UnifiedWALReader::new(wal_path.to_string()).await?;
        let reader: Arc<dyn GraphWalReaderPort> = Arc::new(reader);
        Ok(reader)
    }

    async fn make_filesystem(&self) -> anyhow::Result<Arc<dyn FilesystemPort>> {
        let fs = FilesystemFactory::create_default()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create default filesystem: {e}"))?;
        let fs: Arc<dyn FilesystemPort> = Arc::new(fs);
        Ok(fs)
    }

    async fn make_canonical_wal_reader(&self) -> anyhow::Result<Arc<dyn CanonicalWalReaderPort>> {
        Ok(Arc::new(
            crate::services::canonical_wal::CanonicalWalReaderBridge,
        ))
    }
}

/// Convenience constructor for the default (unified) graph WAL factory, erased
/// to the [`GraphWalFactory`] port. Composition-root callers (and tests) inject
/// this into the ORION engine constructors; it is the only symbol outside this
/// module that needs to name a concrete graph WAL type.
pub fn unified_wal_factory() -> Arc<dyn GraphWalFactory> {
    Arc::new(UnifiedWalFactory)
}

/// WAL reader for recovery
pub struct UnifiedWALReader {
    base_path: String,
    filesystem: Arc<FilesystemFactory>,
}

impl UnifiedWALReader {
    /// Create a new WAL reader
    pub async fn new(base_path: String) -> anyhow::Result<Self> {
        // Create filesystem factory with default config
        let filesystem = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create filesystem: {}", e))?,
        );

        // Normalize to a scheme-qualified url once, mirroring the writer:
        // object-store bases (s3://, adls://, …) pass through untouched
        // (TD-OBJSTORE-1, #960).
        let base_url = if base_path.contains("://") {
            base_path
        } else {
            format!("file://{}", base_path)
        };

        Ok(Self {
            base_path: base_url,
            filesystem,
        })
    }

    /// Read all WAL entries from a segment
    pub async fn read_segment(&self, segment_number: u32) -> anyhow::Result<Vec<UnifiedWALEntry>> {
        // base_path is scheme-qualified (normalized in `new`).
        let url = format!("{}/wal_{:08}.log", self.base_path, segment_number);
        let fs = self.filesystem.get_filesystem(&url)?;

        if !fs.exists(&url).await? {
            return Ok(Vec::new());
        }

        let data = fs.read(&url).await?;
        tracing::debug!(
            "Reading WAL segment {}: {} total bytes",
            segment_number,
            data.len()
        );
        let mut entries = Vec::new();
        let mut cursor = 0;

        while cursor < data.len() {
            // Read size header
            if cursor + 4 > data.len() {
                tracing::debug!(
                    "End of WAL segment at cursor {}, {} bytes remaining",
                    cursor,
                    data.len() - cursor
                );
                break;
            }

            let size = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]) as usize;

            cursor += 4;

            // Sanity check on size
            if size == 0 || size > 10 * 1024 * 1024 {
                // 10MB max per entry
                tracing::warn!(
                    "Invalid WAL entry size {} at cursor {}, skipping rest of segment",
                    size,
                    cursor - 4
                );
                break;
            }

            // Read entry
            if cursor + size > data.len() {
                tracing::warn!(
                    "Truncated WAL entry at cursor {}: need {} bytes, have {}",
                    cursor,
                    size,
                    data.len() - cursor
                );
                break;
            }

            let entry_data = &data[cursor..cursor + size];
            match bincode::deserialize::<UnifiedWALEntry>(entry_data) {
                Ok(entry) => {
                    if entry.verify_checksum() {
                        entries.push(entry);
                    } else {
                        tracing::warn!("Checksum mismatch for entry {}", entries.len());
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to deserialize WAL entry at cursor {}, size {}: {:?}",
                        cursor,
                        size,
                        e
                    );
                    // Log first few bytes for debugging
                    let preview = &entry_data[..entry_data.len().min(32)];
                    tracing::debug!(
                        "Entry data preview (first {} bytes): {:?}",
                        preview.len(),
                        preview
                    );
                    // Continue trying to read more entries
                }
            }

            cursor += size;
        }

        Ok(entries)
    }

    /// Read all WAL entries for recovery
    pub async fn read_all(&self) -> anyhow::Result<Vec<UnifiedWALEntry>> {
        let mut all_entries = Vec::new();

        // TD-066 (d): enumerate segments by listing the directory and reading
        // them in ascending index order, tolerating a missing LOW prefix. The
        // old `0, 1, 2… break-on-empty` scan assumed segments are contiguous
        // from 0; after an LSN-bounded prefix truncation the surviving segments
        // start at S > 0, and that scan would stop at segment 0's gap and
        // silently drop every surviving (post-checkpoint) frame.
        let base_url = if self.base_path.contains("://") {
            self.base_path.clone()
        } else {
            format!("file://{}", self.base_path)
        };
        let fs = self.filesystem.get_filesystem(&base_url)?;
        let mut indices: Vec<u32> = Vec::new();
        if let Ok(files) = fs.list(&base_url).await {
            for file_info in &files {
                if let Some(idx) = parse_wal_segment_index(&file_info.name) {
                    indices.push(idx);
                }
            }
        }
        indices.sort_unstable();

        tracing::debug!(
            "WAL reader reading {} segments from path: {}",
            indices.len(),
            self.base_path
        );

        for segment in indices {
            let entries = self.read_segment(segment).await?;
            tracing::debug!("Read segment {}: {} entries", segment, entries.len());
            all_entries.extend(entries);
        }

        tracing::debug!("WAL reader total entries read: {}", all_entries.len());
        Ok(all_entries)
    }
}

#[cfg(test)]
mod tests {

    /// WAL-retention slice: `PROXIMADB_GRAPH_WAL_SEGMENT_MB` parsing + clamp.
    /// Runs under nextest process isolation, so the env mutation is contained.
    #[test]
    fn graph_wal_segment_size_env_knob_clamps() {
        // SAFETY: process-local env mutation; nextest isolates test processes.
        unsafe { std::env::remove_var("PROXIMADB_GRAPH_WAL_SEGMENT_MB") };
        assert_eq!(super::graph_wal_segment_size_bytes(), 64 * 1024 * 1024);
        unsafe { std::env::set_var("PROXIMADB_GRAPH_WAL_SEGMENT_MB", "8") };
        assert_eq!(super::graph_wal_segment_size_bytes(), 8 * 1024 * 1024);
        unsafe { std::env::set_var("PROXIMADB_GRAPH_WAL_SEGMENT_MB", "9999") };
        assert_eq!(
            super::graph_wal_segment_size_bytes(),
            64 * 1024 * 1024,
            "values above the previous hardcoded 64MB clamp down"
        );
        unsafe { std::env::set_var("PROXIMADB_GRAPH_WAL_SEGMENT_MB", "0") };
        assert_eq!(
            super::graph_wal_segment_size_bytes(),
            1024 * 1024,
            "zero clamps up to the 1MB minimum"
        );
        unsafe { std::env::set_var("PROXIMADB_GRAPH_WAL_SEGMENT_MB", "garbage") };
        assert_eq!(
            super::graph_wal_segment_size_bytes(),
            64 * 1024 * 1024,
            "unparsable values fall back to the default"
        );
        unsafe { std::env::remove_var("PROXIMADB_GRAPH_WAL_SEGMENT_MB") };
    }

    use super::*;
    use crate::graph::Node;
    use proximadb_records::EmbeddingCell;

    #[tokio::test]
    async fn test_unified_wal_operations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        // Create writer
        let mut writer = UnifiedWALWriter::new(path.clone()).await.unwrap();

        // Test graph operation
        let graph_op = UnifiedWALOperation::GraphOp(GraphOperation::CreateNode {
            graph_id: "test_graph".to_string(),
            node: Node {
                id: "node1".to_string(),
                labels: vec!["TestNode".to_string()],
                properties: Default::default(),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        });

        let seq1 = writer.append(graph_op.clone()).await.unwrap();
        assert_eq!(seq1, 0);

        // Test vector operation
        let vector_op = UnifiedWALOperation::VectorOp(VectorOperation::AddVector {
            collection_id: "test_collection".to_string(),
            record: test_record("vec1", vec![0.1, 0.2, 0.3]),
        });

        let seq2 = writer.append(vector_op).await.unwrap();
        assert_eq!(seq2, 1);

        // Sync to disk
        writer.sync().await.unwrap();

        // Read back
        let reader = UnifiedWALReader::new(path).await.unwrap();
        let entries = reader.read_all().await.unwrap();

        assert_eq!(entries.len(), 2);
        assert!(entries[0].is_graph_operation());
        assert!(entries[1].is_vector_operation());
    }

    #[test]
    fn test_hybrid_operation() {
        let hybrid_op = UnifiedWALOperation::HybridOp {
            vector_ops: vec![VectorOperation::AddVector {
                collection_id: "coll1".to_string(),
                record: test_record("vec1", vec![0.1, 0.2]),
            }],
            graph_ops: vec![GraphOperation::CreateNode {
                graph_id: "graph1".to_string(),
                node: Node {
                    id: "node1".to_string(),
                    labels: vec!["Label".to_string()],
                    properties: Default::default(),
                    embedding: None,
                    created_at_ms: 0,
                    updated_at_ms: 0,
                },
            }],
            transaction_id: "tx123".to_string(),
        };

        let entry = UnifiedWALEntry::new(0, hybrid_op);
        assert!(entry.is_graph_operation());
        assert!(entry.is_vector_operation());
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_document_checksum_survives_unordered_props_roundtrip() {
        let mut record = test_record("document/coll/doc1", vec![]);
        record.props.insert(
            "zeta".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "last".to_string(),
            )),
        );
        record.props.insert(
            "alpha".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "first".to_string(),
            )),
        );

        let entry = UnifiedWALEntry::new(
            7,
            UnifiedWALOperation::DocumentOp(DocumentOperation::UpsertCanonicalDocumentRecord {
                collection_id: "coll".to_string(),
                record,
            }),
        );
        let encoded = bincode::serialize(&entry).expect("serialize wal entry");
        let decoded: UnifiedWALEntry =
            bincode::deserialize(&encoded).expect("deserialize wal entry");

        assert!(decoded.verify_checksum());
    }

    // ----- TD-066 (d): WAL truncation + non-contiguous-segment recovery -----

    fn node_op(graph: &str, id: &str) -> UnifiedWALOperation {
        UnifiedWALOperation::GraphOp(GraphOperation::CreateNode {
            graph_id: graph.to_string(),
            node: Node {
                id: id.to_string(),
                labels: vec!["N".to_string()],
                properties: Default::default(),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        })
    }

    fn seqs(entries: &[UnifiedWALEntry]) -> Vec<u64> {
        entries.iter().map(|e| e.sequence_number).collect()
    }

    /// Actual on-disk segment indices, ascending. Under a tiny test segment cap
    /// the first append can exceed the cap and rotate before segment 0 is ever
    /// opened, so segments are NOT guaranteed dense-from-zero — tests must read
    /// the real layout rather than assume it.
    fn list_segment_files(path: &str) -> Vec<u32> {
        let mut v: Vec<u32> = std::fs::read_dir(path)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                name.strip_prefix("wal_")?
                    .strip_suffix(".log")?
                    .parse::<u32>()
                    .ok()
            })
            .collect();
        v.sort_unstable();
        v
    }

    /// Write enough rotating segments to place a `CanonicalEmission` marker in a
    /// known non-zero segment, then truncate. The segments below the marker must
    /// be reclaimed, and recovery must return exactly the post-marker frames
    /// (crash-after-truncate). This is the core size-bounding + no-loss proof.
    #[tokio::test]
    async fn truncate_reclaims_prefix_and_recovery_reads_post_marker() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        let mut writer = UnifiedWALWriter::new(path.clone()).await.unwrap();
        // Tiny cap → roughly one entry per segment.
        writer.set_max_segment_size_for_test(64);

        for id in ["a", "b", "c"] {
            writer.append(node_op("g", id)).await.unwrap();
        }
        writer
            .append(UnifiedWALOperation::GraphMarker(
                MarkerKind::CanonicalEmission(100),
            ))
            .await
            .unwrap();
        for id in ["d", "e"] {
            writer.append(node_op("g", id)).await.unwrap();
        }
        writer.flush().await.unwrap();

        // Locate the marker's segment and compute the expected survivors
        // (everything from the marker's segment onward).
        let reader = UnifiedWALReader::new(path.clone()).await.unwrap();
        let mut marker_segment = None;
        let mut seg = 0u32;
        let mut expected_after: Vec<UnifiedWALEntry> = Vec::new();
        // Read a generous range; missing segments just yield empty.
        for s in 0..32u32 {
            let entries = reader.read_segment(s).await.unwrap();
            if entries.iter().any(|e| {
                matches!(
                    &e.operation,
                    UnifiedWALOperation::GraphMarker(MarkerKind::CanonicalEmission(100))
                )
            }) {
                marker_segment = Some(s);
            }
            if marker_segment.is_some() {
                expected_after.extend(entries);
            }
            seg = s;
        }
        let _ = seg;
        let marker_segment = marker_segment.expect("marker must land in some segment");
        // Count the segments that actually exist below the marker's segment
        // (the layout is not dense-from-zero under the tiny test cap).
        let expected_reclaim = list_segment_files(&path)
            .into_iter()
            .filter(|&s| s < marker_segment)
            .count() as u64;
        assert!(
            expected_reclaim > 0,
            "test needs segments below the marker to exercise prefix deletion"
        );

        let reclaimed = writer.truncate_through_canonical_marker(100).await.unwrap();
        assert_eq!(
            reclaimed, expected_reclaim,
            "every existing segment below the marker's segment should be reclaimed"
        );

        // crash-after-truncate: recovery returns exactly the post-marker frames.
        let after = UnifiedWALReader::new(path.clone())
            .await
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(
            seqs(&after),
            seqs(&expected_after),
            "recovery must read the kept marker segment + all later frames, nothing below"
        );
        // The marker and the post-marker nodes survive; the pre-marker nodes do not.
        assert!(after.iter().any(|e| matches!(
            &e.operation,
            UnifiedWALOperation::GraphMarker(MarkerKind::CanonicalEmission(100))
        )));
    }

    /// A missing LOW segment prefix (as produced by truncation, or simulated by
    /// deleting `wal_00000000.log` mid-truncate) must not stop recovery — it
    /// reads the surviving suffix. Guards the `read_all` precursor fix.
    #[tokio::test]
    async fn read_all_tolerates_missing_low_segment_prefix() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        let mut writer = UnifiedWALWriter::new(path.clone()).await.unwrap();
        writer.set_max_segment_size_for_test(64);
        for id in ["a", "b", "c", "d"] {
            writer.append(node_op("g", id)).await.unwrap();
        }
        writer.flush().await.unwrap();

        // Simulate a crash mid-truncation: the LOWEST existing segment is gone.
        let segments = list_segment_files(&path);
        assert!(segments.len() >= 2, "need multiple segments for this test");
        let lowest = segments[0];
        std::fs::remove_file(format!("{}/wal_{:08}.log", path, lowest)).unwrap();

        let reader = UnifiedWALReader::new(path.clone()).await.unwrap();
        let expected: Vec<UnifiedWALEntry> = {
            let mut acc = Vec::new();
            for &s in &segments[1..] {
                acc.extend(reader.read_segment(s).await.unwrap());
            }
            acc
        };
        let all = reader.read_all().await.unwrap();
        assert_eq!(
            seqs(&all),
            seqs(&expected),
            "read_all must skip the missing prefix and return the surviving suffix"
        );
        assert!(!all.is_empty(), "surviving segments must still be read");
    }

    /// After a prefix delete, reopening the writer must resume at `max_index + 1`
    /// and never reopen/clobber a kept higher segment. Guards the `new`
    /// segment-numbering precursor fix.
    #[tokio::test]
    async fn writer_resumes_past_highest_segment_after_prefix_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        let mut writer = UnifiedWALWriter::new(path.clone()).await.unwrap();
        writer.set_max_segment_size_for_test(64);
        for id in ["a", "b", "c"] {
            writer.append(node_op("g", id)).await.unwrap();
        }
        writer.flush().await.unwrap();
        drop(writer);

        // Capture the highest surviving segment's content, then delete segment 0.
        let reader = UnifiedWALReader::new(path.clone()).await.unwrap();
        let mut highest = 0u32;
        for s in 0..32u32 {
            if !reader.read_segment(s).await.unwrap().is_empty() {
                highest = s;
            }
        }
        let segments = list_segment_files(&path);
        assert!(segments.len() >= 2, "need multiple segments for this test");
        let highest_before = reader.read_segment(highest).await.unwrap();
        // Delete the lowest existing segment, leaving a non-contiguous layout.
        std::fs::remove_file(format!("{}/wal_{:08}.log", path, segments[0])).unwrap();

        // Reopen: file COUNT is now below the kept top index; only `max + 1`
        // avoids clobbering the kept top segment.
        let mut writer2 = UnifiedWALWriter::new(path.clone()).await.unwrap();
        writer2.append(node_op("g", "z")).await.unwrap();
        writer2.flush().await.unwrap();

        // The kept top segment is untouched...
        let highest_after = reader.read_segment(highest).await.unwrap();
        assert_eq!(
            seqs(&highest_before),
            seqs(&highest_after),
            "the kept top segment must not be reopened/clobbered"
        );
        // ...and the new frame landed in a brand-new higher segment.
        let new_seg = reader.read_segment(highest + 1).await.unwrap();
        assert_eq!(
            new_seg.len(),
            1,
            "new append must open a fresh segment past the max"
        );
    }

    #[tokio::test]
    async fn writer_reopen_resumes_sequence_after_current_format_segments() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        let mut writer = UnifiedWALWriter::new(path.clone()).await.unwrap();
        assert_eq!(writer.append(node_op("g", "a")).await.unwrap(), 0);
        assert_eq!(writer.append(node_op("g", "b")).await.unwrap(), 1);
        writer.flush().await.unwrap();
        drop(writer);

        let mut reopened = UnifiedWALWriter::new(path.clone()).await.unwrap();
        assert_eq!(
            reopened.append(node_op("g", "c")).await.unwrap(),
            2,
            "the current wal_NNNNNNNN.log format must restore the next sequence"
        );
        reopened.flush().await.unwrap();

        let entries = UnifiedWALReader::new(path)
            .await
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(seqs(&entries), vec![0, 1, 2]);
    }

    /// A stale/missing checkpoint marker must never delete live frames: truncate
    /// is a strict no-op when the marker for `lsn` is absent.
    #[tokio::test]
    async fn truncate_is_noop_when_marker_absent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap().to_string();

        let mut writer = UnifiedWALWriter::new(path.clone()).await.unwrap();
        writer.set_max_segment_size_for_test(64);
        for id in ["a", "b", "c"] {
            writer.append(node_op("g", id)).await.unwrap();
        }
        writer.flush().await.unwrap();

        let before = UnifiedWALReader::new(path.clone())
            .await
            .unwrap()
            .read_all()
            .await
            .unwrap();

        // No CanonicalEmission(999) marker was ever written.
        let reclaimed = writer.truncate_through_canonical_marker(999).await.unwrap();
        assert_eq!(reclaimed, 0, "absent marker must reclaim nothing");

        let after = UnifiedWALReader::new(path.clone())
            .await
            .unwrap()
            .read_all()
            .await
            .unwrap();
        assert_eq!(seqs(&before), seqs(&after), "no frames may be lost");
    }

    fn test_record(id: &str, vector: Vec<f32>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                vector.len() as u32,
                vector,
            )],
            created_at_ns: 0,
            ..Default::default()
        }
    }
}
