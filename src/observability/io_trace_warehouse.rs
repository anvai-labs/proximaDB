// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-TRACE-2 S4 — Iceberg-managed Parquet warehouse for the io_trace stream.
//!
//! An async compactor reads the durable JSONL+zstd trace segments the sink lands
//! under `_operator/io_trace/` (each line a [`TraceEnvelope`], ADR-066 D1), and
//! projects them into a **star schema** of Iceberg-managed Parquet tables under
//! `_operator/warehouse/`:
//!
//! ```text
//!  trace_header       one row per record — the homogeneous header (cost spine)
//!  trace_relational   one row per relational record — plan geometry
//!  trace_exec         one row per operator (relational `exec[]` flattened)
//!  trace_vector       one row per vector-ANN record — prune / striped-read
//!  trace_vector_access one row per physical vector operator access
//!  trace_embedding    one row per embedding record — KEU token counts
//! ```
//!
//! **Idempotent + crash-safe.** Each source segment maps to a deterministically
//! named per-table Parquet file (`{table}/data/{segment_stem}.parquet`), and
//! [`publish_iceberg_snapshot`](IcebergObjectStoreBridge::publish_iceberg_snapshot)
//! enumerates *all* files under the table's data prefix into a full snapshot. So a
//! crash between writing Parquet and advancing the watermark simply re-writes the
//! same files (overwrite) and re-publishes the same file set — no duplicate rows.
//! Within a segment, records are de-duplicated by the ingestion identity
//! `{writer_uuid, sequence}`. A source-retirement watermark (`_watermark.json`)
//! records processed segments so the happy path skips already-compacted work, and
//! is written only *after* the snapshot commit.
//!
//! Observe-only: billing is untouched (this reads durable segments off the hot
//! path). Unparseable / unknown-modality records are skipped, not fatal.

use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use proximadb_iceberg_engine::IcebergObjectStoreBridge;
use proximadb_iceberg_engine::iceberg::{FormatVersion, IcebergField};
use proximadb_iceberg_engine::manifest::CommitOutcome;
use proximadb_kernel::error::StorageError;
use proximadb_kernel::time::now_unix_ms;
use proximadb_object_store::ProximaObjectStore;
use proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::observability::trace_envelope::{TraceEnvelope, TracePayload};
use crate::storage::trait_components::path_resolver::DrPathBuilder;

const T_HEADER: &str = "trace_header";
const T_RELATIONAL: &str = "trace_relational";
const T_EXEC: &str = "trace_exec";
const T_VECTOR: &str = "trace_vector";
const T_VECTOR_ACCESS: &str = "trace_vector_access";
const T_EMBEDDING: &str = "trace_embedding";
const T_COMPUTE: &str = "trace_compute";

/// Outcome of one [`IoTraceWarehouse::compact`] run.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct CompactionSummary {
    /// Source segments compacted this run (0 ⇒ nothing new).
    pub segments_processed: usize,
    /// Total records read across those segments.
    pub records_ingested: usize,
    /// Records dropped as duplicates of an already-seen `{writer_uuid, sequence}`.
    pub duplicates_dropped: usize,
    /// Rows written per star-schema table this run.
    pub header_rows: usize,
    pub relational_rows: usize,
    pub exec_rows: usize,
    pub vector_rows: usize,
    pub vector_access_rows: usize,
    pub embedding_rows: usize,
    /// Rows in `trace_compute` — one per `(record, engine)` from `compute_ms`.
    pub compute_rows: usize,
}

/// Persisted source-retirement watermark (`_operator/warehouse/_watermark.json`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct Watermark {
    /// Source segment keys already compacted (skip on the happy path).
    #[serde(default)]
    processed_segments: BTreeSet<String>,
}

/// The io_trace → Iceberg Parquet warehouse compactor.
pub struct IoTraceWarehouse {
    store: ProximaObjectStore,
    bridge: IcebergObjectStoreBridge,
    source_prefix: String,
    warehouse_prefix: String,
}

impl IoTraceWarehouse {
    /// Build over an existing store, with explicit source + warehouse prefixes.
    pub fn new(
        store: ProximaObjectStore,
        source_prefix: impl Into<String>,
        warehouse_prefix: impl Into<String>,
    ) -> Self {
        let bridge = IcebergObjectStoreBridge::new(store.clone());
        Self {
            store,
            bridge,
            source_prefix: source_prefix.into(),
            warehouse_prefix: warehouse_prefix.into(),
        }
    }

    /// Build from an object-store URL, using the canonical operator prefixes
    /// (`_operator/io_trace/` source, `_operator/warehouse/` warehouse) via
    /// `DrPathBuilder` — keys are never raw.
    pub fn from_url(url: &str) -> Result<Self, StorageError> {
        let store = ProximaObjectStore::from_url(url)?;
        let source_prefix = DrPathBuilder::operator_subprefix("io_trace")
            .unwrap_or_else(|_| "_operator/io_trace/".to_string());
        let warehouse_prefix = DrPathBuilder::operator_subprefix("warehouse")
            .unwrap_or_else(|_| "_operator/warehouse/".to_string());
        Ok(Self::new(store, source_prefix, warehouse_prefix))
    }

    /// Compact every not-yet-processed source segment into the warehouse. `now_ms`
    /// stamps the Iceberg snapshot commit time (passed in for determinism/testing).
    pub async fn compact(&self, now_ms: i64) -> Result<CompactionSummary, StorageError> {
        let mut watermark = self.read_watermark().await?;
        let segments = self.list_new_segments(&watermark).await?;
        let mut summary = CompactionSummary::default();
        if segments.is_empty() {
            return Ok(summary);
        }

        let mut touched: BTreeSet<&'static str> = BTreeSet::new();
        for segment in &segments {
            let records = self.read_segment(segment).await?;
            let stem = segment_stem(segment);

            // De-duplicate within the segment by ingestion identity.
            let mut seen: HashSet<(String, u64)> = HashSet::new();
            let mut deduped: Vec<TraceEnvelope> = Vec::with_capacity(records.len());
            for env in records {
                summary.records_ingested += 1;
                if seen.insert((env.writer_uuid.clone(), env.sequence)) {
                    deduped.push(env);
                } else {
                    summary.duplicates_dropped += 1;
                }
            }

            summary.header_rows += self
                .write_segment_parquet(T_HEADER, &stem, build_header_batch(&deduped)?, &mut touched)
                .await?;
            summary.relational_rows += self
                .write_segment_parquet(
                    T_RELATIONAL,
                    &stem,
                    build_relational_batch(&deduped)?,
                    &mut touched,
                )
                .await?;
            summary.exec_rows += self
                .write_segment_parquet(T_EXEC, &stem, build_exec_batch(&deduped)?, &mut touched)
                .await?;
            summary.vector_rows += self
                .write_segment_parquet(T_VECTOR, &stem, build_vector_batch(&deduped)?, &mut touched)
                .await?;
            summary.vector_access_rows += self
                .write_segment_parquet(
                    T_VECTOR_ACCESS,
                    &stem,
                    build_vector_access_batch(&deduped)?,
                    &mut touched,
                )
                .await?;
            summary.embedding_rows += self
                .write_segment_parquet(
                    T_EMBEDDING,
                    &stem,
                    build_embedding_batch(&deduped)?,
                    &mut touched,
                )
                .await?;
            summary.compute_rows += self
                .write_segment_parquet(
                    T_COMPUTE,
                    &stem,
                    build_compute_batch(&deduped)?,
                    &mut touched,
                )
                .await?;
            summary.segments_processed += 1;
        }

        // Publish one Iceberg snapshot per touched table (full-snapshot over its
        // data prefix), then advance the watermark once the commits are durable. The
        // caller-supplied `now_ms` is the snapshot id — unique per run, so a
        // watermark reset (crash) never reuses a snapshot id.
        for table in &touched {
            self.publish_table(table, now_ms, now_ms).await?;
        }
        for segment in segments {
            watermark.processed_segments.insert(segment);
        }
        self.write_watermark(&watermark).await?;
        Ok(summary)
    }

    /// Write one table's per-segment Parquet file (skipping empty batches), marking
    /// the table touched so it gets republished.
    async fn write_segment_parquet(
        &self,
        table: &'static str,
        stem: &str,
        batch: Option<RecordBatch>,
        touched: &mut BTreeSet<&'static str>,
    ) -> Result<usize, StorageError> {
        let Some(batch) = batch else { return Ok(0) };
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(0);
        }
        let schema = batch.schema();
        let bytes = record_batches_to_parquet_bytes(&[batch], schema, Some(warehouse_props()))?;
        let key = format!("{}{}/data/{}.parquet", self.warehouse_prefix, table, stem);
        self.store
            .put(&ObjectPath::from(key), Bytes::from(bytes))
            .await?;
        touched.insert(table);
        Ok(rows)
    }

    /// Publish a full Iceberg snapshot over `{table}/data` (enumerates all Parquet
    /// files there). A conflicting concurrent commit is logged and left for the next
    /// run — the single-writer compactor is idempotent.
    async fn publish_table(
        &self,
        table: &str,
        snapshot_id: i64,
        now_ms: i64,
    ) -> Result<(), StorageError> {
        let Some(schema) = schema_for(table) else {
            return Ok(());
        };
        let data_prefix = ObjectPath::from(format!("{}{}/data", self.warehouse_prefix, table));
        let metadata_prefix = format!("{}{}/metadata", self.warehouse_prefix, table);
        let location = format!("{}{}", self.warehouse_prefix, table);
        let fields = iceberg_fields(&schema);
        let table_uuid = stable_table_uuid(table);
        match self
            .bridge
            .publish_iceberg_snapshot(
                &data_prefix,
                &metadata_prefix,
                &table_uuid,
                &location,
                &fields,
                snapshot_id,
                now_ms,
                FormatVersion::V3,
            )
            .await?
        {
            CommitOutcome::Committed(_) => Ok(()),
            CommitOutcome::Conflict { latest } => {
                tracing::warn!(
                    "io_trace warehouse: {table} snapshot conflict (latest={latest:?}); \
                     retried next run"
                );
                Ok(())
            }
        }
    }

    async fn read_segment(&self, key: &str) -> Result<Vec<TraceEnvelope>, StorageError> {
        let compressed = self.store.get(&ObjectPath::from(key.to_string())).await?;
        let decoded = zstd::decode_all(&compressed[..]).map_err(|e| {
            StorageError::Serialization(format!("io_trace warehouse: zstd decode {key}: {e}"))
        })?;
        let text = String::from_utf8(decoded).map_err(|e| {
            StorageError::Serialization(format!("io_trace warehouse: utf8 {key}: {e}"))
        })?;
        let mut out = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<TraceEnvelope>(line) {
                Ok(env) => out.push(env),
                // Schema-evolution tolerance: a record this build can't parse is
                // skipped (unknown modalities already map to TracePayload::Other).
                Err(e) => {
                    tracing::warn!("io_trace warehouse: skip malformed record in {key}: {e}")
                }
            }
        }
        Ok(out)
    }

    async fn list_new_segments(&self, watermark: &Watermark) -> Result<Vec<String>, StorageError> {
        let prefix = ObjectPath::from(self.source_prefix.trim_end_matches('/').to_string());
        let base = self.store.base().as_ref().to_string();
        let mut segments: Vec<String> = self
            .store
            .list(Some(&prefix))
            .await?
            .into_iter()
            .map(|meta| strip_base(&base, meta.location.as_ref()))
            .filter(|key| {
                key.ends_with(".jsonl.zst") && !watermark.processed_segments.contains(key)
            })
            .collect();
        segments.sort();
        Ok(segments)
    }

    async fn read_watermark(&self) -> Result<Watermark, StorageError> {
        let key = ObjectPath::from(format!("{}_watermark.json", self.warehouse_prefix));
        match self.store.get(&key).await {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
                StorageError::Serialization(format!("io_trace warehouse: watermark parse: {e}"))
            }),
            // Absent (or unreadable) ⇒ a fresh warehouse. Reprocessing is idempotent.
            Err(_) => Ok(Watermark::default()),
        }
    }

    async fn write_watermark(&self, watermark: &Watermark) -> Result<(), StorageError> {
        let key = ObjectPath::from(format!("{}_watermark.json", self.warehouse_prefix));
        let bytes = serde_json::to_vec(watermark).map_err(|e| {
            StorageError::Serialization(format!("io_trace warehouse: watermark encode: {e}"))
        })?;
        self.store.put(&key, Bytes::from(bytes)).await
    }
}

// ---------------------------------------------------------------------------
// S4b — background compaction trigger.
// ---------------------------------------------------------------------------

/// Wall-clock milliseconds since the Unix epoch — the per-run Iceberg snapshot id.
/// Wall-clock is fine in production; only workflow scripts forbid it.
struct TriggerHandle {
    shutdown: tokio::sync::watch::Sender<bool>,
    join: tokio::task::JoinHandle<()>,
}

static TRIGGER: OnceLock<Mutex<Option<TriggerHandle>>> = OnceLock::new();
static TRIGGER_ACTIVE: AtomicBool = AtomicBool::new(false);

fn trigger_slot() -> &'static Mutex<Option<TriggerHandle>> {
    TRIGGER.get_or_init(|| Mutex::new(None))
}

/// Install the background warehouse compactor: build the compactor from
/// `object_store_uri` (the store the sink dispatches segments to) and spawn a task
/// that runs [`IoTraceWarehouse::compact`] every `interval_s` seconds — and once
/// more on shutdown. Rejects a duplicate install without disturbing the live worker.
/// Default-OFF: a caller installs it only when S4b config opts in. Must run inside a
/// tokio runtime (it is — `ProximaDB::new`).
pub fn install(object_store_uri: String, interval_s: u64) -> Result<(), String> {
    if TRIGGER_ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("io_trace warehouse compactor is already installed".to_string());
    }
    let warehouse = match IoTraceWarehouse::from_url(&object_store_uri) {
        Ok(w) => w,
        Err(e) => {
            TRIGGER_ACTIVE.store(false, Ordering::Release);
            return Err(format!(
                "io_trace warehouse compactor: object_store_uri {object_store_uri} invalid: {e}"
            ));
        }
    };
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let join = tokio::spawn(run_compaction_loop(
        warehouse,
        interval_s.max(1),
        shutdown_rx,
    ));
    *trigger_slot().lock().unwrap_or_else(|p| p.into_inner()) = Some(TriggerHandle {
        shutdown: shutdown_tx,
        join,
    });
    tracing::info!("io_trace warehouse compactor installed (interval {interval_s}s)");
    Ok(())
}

/// Signal the compactor to run a final pass and stop, awaiting it with a short
/// timeout. Called from `ProximaDB::shutdown()`. No-op if not installed.
pub async fn shutdown() {
    let handle = trigger_slot()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take();
    if let Some(handle) = handle {
        let _ = handle.shutdown.send(true);
        match tokio::time::timeout(Duration::from_secs(30), handle.join).await {
            Ok(Ok(())) => tracing::info!("io_trace warehouse compactor stopped"),
            Ok(Err(e)) => tracing::warn!("io_trace warehouse compactor join error: {e}"),
            Err(_) => tracing::warn!("io_trace warehouse compactor shutdown timed out (30s)"),
        }
        TRIGGER_ACTIVE.store(false, Ordering::Release);
    }
}

async fn run_compaction_loop(
    warehouse: IoTraceWarehouse,
    interval_s: u64,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_s));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Consume the immediate first tick; a fresh install shouldn't compact instantly.
    interval.tick().await;
    loop {
        tokio::select! {
            _ = interval.tick() => run_compaction_pass(&warehouse).await,
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    // Final pass so a graceful shutdown lands the latest segments.
                    run_compaction_pass(&warehouse).await;
                    return;
                }
            }
        }
    }
}

/// One compaction pass; best-effort — a failure is logged, never fatal.
async fn run_compaction_pass(warehouse: &IoTraceWarehouse) {
    match warehouse.compact(now_unix_ms()).await {
        Ok(summary) if summary.segments_processed > 0 => tracing::info!(
            "io_trace warehouse: compacted {} segment(s), {} header row(s)",
            summary.segments_processed,
            summary.header_rows
        ),
        Ok(_) => {}
        Err(e) => tracing::warn!("io_trace warehouse: compaction pass failed: {e}"),
    }
}

/// Snappy-compressed Parquet, warehouse-grade.
fn warehouse_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build()
}

/// A stable, deterministic table UUID (so re-publishing keeps the same table
/// identity). Derived from a sha256 of the table name — avoids the `uuid` `v5`
/// feature while staying content-stable.
fn stable_table_uuid(table: &str) -> String {
    let digest = Sha256::digest(table.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes).to_string()
}

/// The per-segment Parquet stem (unique — carries the writer/sequence/digest).
fn segment_stem(key: &str) -> String {
    let file = key.rsplit('/').next().unwrap_or(key);
    file.strip_suffix(".jsonl.zst").unwrap_or(file).to_string()
}

/// Recover the caller-relative key from a listed `ObjectMeta` location.
/// `ProximaObjectStore::list` returns base-prefixed locations, but `get`/`put`
/// re-prepend the base — so a listed location must have the base stripped before it
/// is fed back to `get` (else the base is doubled).
fn strip_base(base: &str, location: &str) -> String {
    if base.is_empty() {
        return location.to_string();
    }
    location
        .strip_prefix(base)
        .map(|s| s.trim_start_matches('/').to_string())
        .unwrap_or_else(|| location.to_string())
}

fn iceberg_type_name(dt: &DataType) -> &'static str {
    match dt {
        DataType::Int64 => "long",
        DataType::Boolean => "boolean",
        _ => "string",
    }
}

fn iceberg_fields(schema: &Schema) -> Vec<IcebergField> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| IcebergField {
            id: (i + 1) as i32,
            name: f.name().clone(),
            type_name: iceberg_type_name(f.data_type()).to_string(),
            required: !f.is_nullable(),
        })
        .collect()
}

fn schema_for(table: &str) -> Option<SchemaRef> {
    match table {
        T_HEADER => Some(header_schema()),
        T_RELATIONAL => Some(relational_schema()),
        T_EXEC => Some(exec_schema()),
        T_VECTOR => Some(vector_schema()),
        T_VECTOR_ACCESS => Some(vector_access_schema()),
        T_EMBEDDING => Some(embedding_schema()),
        T_COMPUTE => Some(compute_schema()),
        _ => None,
    }
}

fn req(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, false)
}
fn opt(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, true)
}

fn header_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        req("schema_version", DataType::Int64),
        opt("query_id", DataType::Utf8),
        opt("tenant", DataType::Utf8),
        req("event_time_unix_ms", DataType::Int64),
        opt("route_shape", DataType::Utf8),
        opt("route_backend", DataType::Utf8),
        opt("storage_profile", DataType::Utf8),
        req("modality", DataType::Utf8),
        req("get_ops", DataType::Int64),
        req("put_ops", DataType::Int64),
        req("list_ops", DataType::Int64),
        req("delete_ops", DataType::Int64),
        req("bytes_read", DataType::Int64),
        req("bytes_written", DataType::Int64),
        req("range_gets", DataType::Int64),
        req("footer_hits", DataType::Int64),
        req("footer_misses", DataType::Int64),
        req("setup_ms", DataType::Int64),
        req("open_ms", DataType::Int64),
        req("plan_ms", DataType::Int64),
        req("emit_ms", DataType::Int64),
        req("session_ms", DataType::Int64),
        req("table_open_hits", DataType::Int64),
        req("table_open_misses", DataType::Int64),
        req("egress_bytes", DataType::Int64),
        req("compute_ms_total", DataType::Int64),
        // Persistent-L2 cache probes (ADR-085 / TD-IOTRACE-4). Appended and
        // NULLABLE: parquet segments written before these columns existed lack
        // them, and readers must resolve the absence to NULL — never fail.
        opt("l2_hits", DataType::Int64),
        opt("l2_misses", DataType::Int64),
        // In-process DRAM cache outcomes. Also appended/nullable for mixed
        // reads with warehouse segments written before this evidence existed.
        opt("survivor_l1_hits", DataType::Int64),
        opt("survivor_l1_misses", DataType::Int64),
    ]))
}

fn relational_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        opt("query_id", DataType::Utf8),
        req("plan_depth", DataType::Int64),
        req("plan_nodes", DataType::Int64),
        req("plan_leaves", DataType::Int64),
        req("plan_fanout", DataType::Int64),
        req("plan_blocking", DataType::Int64),
        req("stack_hwm_bytes", DataType::Int64),
        req("splits_total", DataType::Int64),
        req("splits_pruned", DataType::Int64),
        req("runtime_filter_arrived", DataType::Int64),
        req("runtime_filter_timed_out", DataType::Int64),
        req("runtime_filter_wait_ms", DataType::Int64),
    ]))
}

fn exec_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        opt("query_id", DataType::Utf8),
        req("op_index", DataType::Int64),
        req("op", DataType::Utf8),
        req("rows_in", DataType::Int64),
        req("rows_out", DataType::Int64),
        req("ms_self", DataType::Int64),
        opt("bytes", DataType::Int64),
        req("spill", DataType::Boolean),
    ]))
}

fn vector_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        opt("query_id", DataType::Utf8),
        req("centroid_total_blocks", DataType::Int64),
        req("centroid_pruned_blocks", DataType::Int64),
        req("logical_striped_bytes", DataType::Int64),
        req("logical_striped_gets", DataType::Int64),
        req("ivf_cells_total", DataType::Int64),
        req("ivf_cells_probed", DataType::Int64),
        req("ivf_probed_rows", DataType::Int64),
        req("ivf_region_a_bytes", DataType::Int64),
        req("ivf_region_b_bytes", DataType::Int64),
        req("ivf_fetch_rounds", DataType::Int64),
        req("ivf_whole_region_fallback", DataType::Int64),
    ]))
}

fn vector_access_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        opt("query_id", DataType::Utf8),
        req("access_index", DataType::Int64),
        req("engine", DataType::Utf8),
        req("dimensions", DataType::Int64),
        req("top_k", DataType::Int64),
        req("has_filter", DataType::Boolean),
        req("requested_mode", DataType::Utf8),
        req("actual_path", DataType::Utf8),
        // Appended/nullable: vector-access satellites written before this
        // dimension existed do not contain the column.
        opt("storage_scope", DataType::Utf8),
        // Appended/nullable for mixed-read safety with pre-policy satellites.
        opt("cache_policy", DataType::Utf8),
    ]))
}

fn embedding_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        opt("query_id", DataType::Utf8),
        req("embedding_calls", DataType::Int64),
        req("embedding_input_tokens", DataType::Int64),
        req("embedding_output_tokens", DataType::Int64),
    ]))
}

/// One row per `(record, engine)` — the header `compute_ms{engine→ms}` map flattened
/// so cost analytics can ask "which engine wins per query shape".
fn compute_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        req("writer_uuid", DataType::Utf8),
        req("sequence", DataType::Int64),
        opt("query_id", DataType::Utf8),
        req("engine", DataType::Utf8),
        req("ms", DataType::Int64),
    ]))
}

fn str_arr(v: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(v))
}
fn opt_str_arr(v: Vec<Option<String>>) -> ArrayRef {
    Arc::new(StringArray::from_iter(v))
}
fn i64_arr(v: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(v))
}

fn build_header_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    if envs.is_empty() {
        return Ok(None);
    }
    let n = envs.len();
    let mut writer_uuid = Vec::with_capacity(n);
    let mut sequence = Vec::with_capacity(n);
    let mut schema_version = Vec::with_capacity(n);
    let mut query_id = Vec::with_capacity(n);
    let mut tenant = Vec::with_capacity(n);
    let mut event_time = Vec::with_capacity(n);
    let mut route_shape = Vec::with_capacity(n);
    let mut route_backend = Vec::with_capacity(n);
    let mut storage_profile = Vec::with_capacity(n);
    let mut modality = Vec::with_capacity(n);
    let mut get_ops = Vec::with_capacity(n);
    let mut put_ops = Vec::with_capacity(n);
    let mut list_ops = Vec::with_capacity(n);
    let mut delete_ops = Vec::with_capacity(n);
    let mut bytes_read = Vec::with_capacity(n);
    let mut bytes_written = Vec::with_capacity(n);
    let mut range_gets = Vec::with_capacity(n);
    let mut footer_hits = Vec::with_capacity(n);
    let mut footer_misses = Vec::with_capacity(n);
    let mut setup_ms = Vec::with_capacity(n);
    let mut open_ms = Vec::with_capacity(n);
    let mut plan_ms = Vec::with_capacity(n);
    let mut emit_ms = Vec::with_capacity(n);
    let mut session_ms = Vec::with_capacity(n);
    let mut table_open_hits = Vec::with_capacity(n);
    let mut table_open_misses = Vec::with_capacity(n);
    let mut egress_bytes = Vec::with_capacity(n);
    let mut compute_ms_total = Vec::with_capacity(n);
    let mut l2_hits = Vec::with_capacity(n);
    let mut l2_misses = Vec::with_capacity(n);
    let mut survivor_l1_hits = Vec::with_capacity(n);
    let mut survivor_l1_misses = Vec::with_capacity(n);
    for e in envs {
        let h = &e.header;
        writer_uuid.push(e.writer_uuid.clone());
        sequence.push(e.sequence as i64);
        schema_version.push(e.schema_version as i64);
        query_id.push(h.query_id.clone());
        tenant.push(h.tenant.clone());
        event_time.push(h.event_time_unix_ms as i64);
        let (rs, rb) = match &h.route {
            Some((s, b)) => (Some(s.clone()), Some(b.clone())),
            None => (None, None),
        };
        route_shape.push(rs);
        route_backend.push(rb);
        storage_profile.push(h.storage_profile.clone());
        modality.push(e.payload.modality().to_string());
        get_ops.push(h.get_ops as i64);
        put_ops.push(h.put_ops as i64);
        list_ops.push(h.list_ops as i64);
        delete_ops.push(h.delete_ops as i64);
        bytes_read.push(h.bytes_read as i64);
        bytes_written.push(h.bytes_written as i64);
        range_gets.push(h.range_gets as i64);
        footer_hits.push(h.footer_hits as i64);
        footer_misses.push(h.footer_misses as i64);
        setup_ms.push(h.setup_ms as i64);
        open_ms.push(h.open_ms as i64);
        plan_ms.push(h.plan_ms as i64);
        emit_ms.push(h.emit_ms as i64);
        session_ms.push(h.session_ms as i64);
        table_open_hits.push(h.table_open_hits as i64);
        table_open_misses.push(h.table_open_misses as i64);
        egress_bytes.push(h.egress_bytes as i64);
        compute_ms_total.push(h.compute_ms.values().sum::<u64>() as i64);
        l2_hits.push(h.l2_hits as i64);
        l2_misses.push(h.l2_misses as i64);
        survivor_l1_hits.push(h.survivor_l1_hits as i64);
        survivor_l1_misses.push(h.survivor_l1_misses as i64);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        i64_arr(schema_version),
        opt_str_arr(query_id),
        opt_str_arr(tenant),
        i64_arr(event_time),
        opt_str_arr(route_shape),
        opt_str_arr(route_backend),
        opt_str_arr(storage_profile),
        str_arr(modality),
        i64_arr(get_ops),
        i64_arr(put_ops),
        i64_arr(list_ops),
        i64_arr(delete_ops),
        i64_arr(bytes_read),
        i64_arr(bytes_written),
        i64_arr(range_gets),
        i64_arr(footer_hits),
        i64_arr(footer_misses),
        i64_arr(setup_ms),
        i64_arr(open_ms),
        i64_arr(plan_ms),
        i64_arr(emit_ms),
        i64_arr(session_ms),
        i64_arr(table_open_hits),
        i64_arr(table_open_misses),
        i64_arr(egress_bytes),
        i64_arr(compute_ms_total),
        i64_arr(l2_hits),
        i64_arr(l2_misses),
        i64_arr(survivor_l1_hits),
        i64_arr(survivor_l1_misses),
    ];
    RecordBatch::try_new(header_schema(), cols)
        .map(Some)
        .map_err(|e| StorageError::Serialization(format!("io_trace warehouse: header batch: {e}")))
}

fn build_relational_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    let mut writer_uuid = Vec::new();
    let mut sequence = Vec::new();
    let mut query_id = Vec::new();
    let mut plan_depth = Vec::new();
    let mut plan_nodes = Vec::new();
    let mut plan_leaves = Vec::new();
    let mut plan_fanout = Vec::new();
    let mut plan_blocking = Vec::new();
    let mut stack_hwm_bytes = Vec::new();
    let mut splits_total = Vec::new();
    let mut splits_pruned = Vec::new();
    let mut rf_arrived = Vec::new();
    let mut rf_timed_out = Vec::new();
    let mut rf_wait_ms = Vec::new();
    for e in envs {
        if let TracePayload::Relational(p) = &e.payload {
            writer_uuid.push(e.writer_uuid.clone());
            sequence.push(e.sequence as i64);
            query_id.push(e.header.query_id.clone());
            plan_depth.push(p.plan_depth as i64);
            plan_nodes.push(p.plan_nodes as i64);
            plan_leaves.push(p.plan_leaves as i64);
            plan_fanout.push(p.plan_fanout as i64);
            plan_blocking.push(p.plan_blocking as i64);
            stack_hwm_bytes.push(p.stack_hwm_bytes as i64);
            splits_total.push(p.splits_total as i64);
            splits_pruned.push(p.splits_pruned as i64);
            rf_arrived.push(p.runtime_filter_arrived as i64);
            rf_timed_out.push(p.runtime_filter_timed_out as i64);
            rf_wait_ms.push(p.runtime_filter_wait_ms as i64);
        }
    }
    if writer_uuid.is_empty() {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        opt_str_arr(query_id),
        i64_arr(plan_depth),
        i64_arr(plan_nodes),
        i64_arr(plan_leaves),
        i64_arr(plan_fanout),
        i64_arr(plan_blocking),
        i64_arr(stack_hwm_bytes),
        i64_arr(splits_total),
        i64_arr(splits_pruned),
        i64_arr(rf_arrived),
        i64_arr(rf_timed_out),
        i64_arr(rf_wait_ms),
    ];
    RecordBatch::try_new(relational_schema(), cols)
        .map(Some)
        .map_err(|e| {
            StorageError::Serialization(format!("io_trace warehouse: relational batch: {e}"))
        })
}

fn build_exec_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    let mut writer_uuid = Vec::new();
    let mut sequence = Vec::new();
    let mut query_id = Vec::new();
    let mut op_index = Vec::new();
    let mut op = Vec::new();
    let mut rows_in = Vec::new();
    let mut rows_out = Vec::new();
    let mut ms_self = Vec::new();
    let mut bytes = Vec::new();
    let mut spill = Vec::new();
    for e in envs {
        if let TracePayload::Relational(p) = &e.payload {
            for (i, o) in p.exec.iter().enumerate() {
                writer_uuid.push(e.writer_uuid.clone());
                sequence.push(e.sequence as i64);
                query_id.push(e.header.query_id.clone());
                op_index.push(i as i64);
                op.push(o.op.clone());
                rows_in.push(o.rows_in as i64);
                rows_out.push(o.rows_out as i64);
                ms_self.push(o.ms_self as i64);
                bytes.push(o.bytes.map(|b| b as i64));
                spill.push(o.spill);
            }
        }
    }
    if writer_uuid.is_empty() {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        opt_str_arr(query_id),
        i64_arr(op_index),
        str_arr(op),
        i64_arr(rows_in),
        i64_arr(rows_out),
        i64_arr(ms_self),
        Arc::new(Int64Array::from(bytes)),
        Arc::new(BooleanArray::from(spill)),
    ];
    RecordBatch::try_new(exec_schema(), cols)
        .map(Some)
        .map_err(|e| StorageError::Serialization(format!("io_trace warehouse: exec batch: {e}")))
}

fn build_vector_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    let mut writer_uuid = Vec::new();
    let mut sequence = Vec::new();
    let mut query_id = Vec::new();
    let mut centroid_total = Vec::new();
    let mut centroid_pruned = Vec::new();
    let mut striped_bytes = Vec::new();
    let mut striped_gets = Vec::new();
    let mut ivf_cells_total = Vec::new();
    let mut ivf_cells_probed = Vec::new();
    let mut ivf_probed_rows = Vec::new();
    let mut ivf_region_a_bytes = Vec::new();
    let mut ivf_region_b_bytes = Vec::new();
    let mut ivf_fetch_rounds = Vec::new();
    let mut ivf_whole_region_fallback = Vec::new();
    for e in envs {
        if let TracePayload::VectorAnn(p) = &e.payload {
            writer_uuid.push(e.writer_uuid.clone());
            sequence.push(e.sequence as i64);
            query_id.push(e.header.query_id.clone());
            centroid_total.push(p.centroid_total_blocks as i64);
            centroid_pruned.push(p.centroid_pruned_blocks as i64);
            striped_bytes.push(p.logical_striped_bytes as i64);
            striped_gets.push(p.logical_striped_gets as i64);
            ivf_cells_total.push(p.ivf_cells_total as i64);
            ivf_cells_probed.push(p.ivf_cells_probed as i64);
            ivf_probed_rows.push(p.ivf_probed_rows as i64);
            ivf_region_a_bytes.push(p.ivf_region_a_bytes as i64);
            ivf_region_b_bytes.push(p.ivf_region_b_bytes as i64);
            ivf_fetch_rounds.push(p.ivf_fetch_rounds as i64);
            ivf_whole_region_fallback.push(p.ivf_whole_region_fallback as i64);
        }
    }
    if writer_uuid.is_empty() {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        opt_str_arr(query_id),
        i64_arr(centroid_total),
        i64_arr(centroid_pruned),
        i64_arr(striped_bytes),
        i64_arr(striped_gets),
        i64_arr(ivf_cells_total),
        i64_arr(ivf_cells_probed),
        i64_arr(ivf_probed_rows),
        i64_arr(ivf_region_a_bytes),
        i64_arr(ivf_region_b_bytes),
        i64_arr(ivf_fetch_rounds),
        i64_arr(ivf_whole_region_fallback),
    ];
    RecordBatch::try_new(vector_schema(), cols)
        .map(Some)
        .map_err(|e| StorageError::Serialization(format!("io_trace warehouse: vector batch: {e}")))
}

fn build_vector_access_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    let mut writer_uuid = Vec::new();
    let mut sequence = Vec::new();
    let mut query_id = Vec::new();
    let mut access_index = Vec::new();
    let mut engine = Vec::new();
    let mut dimensions = Vec::new();
    let mut top_k = Vec::new();
    let mut has_filter = Vec::new();
    let mut requested_mode = Vec::new();
    let mut actual_path = Vec::new();
    let mut storage_scope = Vec::new();
    let mut cache_policy = Vec::new();
    for envelope in envs {
        if let TracePayload::VectorAnn(payload) = &envelope.payload {
            for (index, access) in payload.access_paths.iter().enumerate() {
                writer_uuid.push(envelope.writer_uuid.clone());
                sequence.push(envelope.sequence as i64);
                query_id.push(envelope.header.query_id.clone());
                access_index.push(index as i64);
                engine.push(access.engine.clone());
                dimensions.push(access.dimensions as i64);
                top_k.push(access.top_k as i64);
                has_filter.push(access.has_filter);
                requested_mode.push(access.requested_mode.as_str().to_string());
                actual_path.push(access.actual_path.as_str().to_string());
                storage_scope.push(access.storage_scope.as_str().to_string());
                cache_policy.push(access.cache_policy.as_str().to_string());
            }
        }
    }
    if writer_uuid.is_empty() {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        opt_str_arr(query_id),
        i64_arr(access_index),
        str_arr(engine),
        i64_arr(dimensions),
        i64_arr(top_k),
        Arc::new(BooleanArray::from(has_filter)),
        str_arr(requested_mode),
        str_arr(actual_path),
        str_arr(storage_scope),
        str_arr(cache_policy),
    ];
    RecordBatch::try_new(vector_access_schema(), cols)
        .map(Some)
        .map_err(|e| {
            StorageError::Serialization(format!("io_trace warehouse: vector access batch: {e}"))
        })
}

fn build_embedding_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    let mut writer_uuid = Vec::new();
    let mut sequence = Vec::new();
    let mut query_id = Vec::new();
    let mut calls = Vec::new();
    let mut input_tokens = Vec::new();
    let mut output_tokens = Vec::new();
    for e in envs {
        if let TracePayload::Embedding(p) = &e.payload {
            writer_uuid.push(e.writer_uuid.clone());
            sequence.push(e.sequence as i64);
            query_id.push(e.header.query_id.clone());
            calls.push(p.embedding_calls as i64);
            input_tokens.push(p.embedding_input_tokens as i64);
            output_tokens.push(p.embedding_output_tokens as i64);
        }
    }
    if writer_uuid.is_empty() {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        opt_str_arr(query_id),
        i64_arr(calls),
        i64_arr(input_tokens),
        i64_arr(output_tokens),
    ];
    RecordBatch::try_new(embedding_schema(), cols)
        .map(Some)
        .map_err(|e| {
            StorageError::Serialization(format!("io_trace warehouse: embedding batch: {e}"))
        })
}

fn build_compute_batch(envs: &[TraceEnvelope]) -> Result<Option<RecordBatch>, StorageError> {
    let mut writer_uuid = Vec::new();
    let mut sequence = Vec::new();
    let mut query_id = Vec::new();
    let mut engine = Vec::new();
    let mut ms = Vec::new();
    for e in envs {
        // `compute_ms` is a BTreeMap, so engines emit in a stable order per record.
        for (eng, val) in &e.header.compute_ms {
            writer_uuid.push(e.writer_uuid.clone());
            sequence.push(e.sequence as i64);
            query_id.push(e.header.query_id.clone());
            engine.push(eng.clone());
            ms.push(*val as i64);
        }
    }
    if writer_uuid.is_empty() {
        return Ok(None);
    }
    let cols: Vec<ArrayRef> = vec![
        str_arr(writer_uuid),
        i64_arr(sequence),
        opt_str_arr(query_id),
        str_arr(engine),
        i64_arr(ms),
    ];
    RecordBatch::try_new(compute_schema(), cols)
        .map(Some)
        .map_err(|e| StorageError::Serialization(format!("io_trace warehouse: compute batch: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::trace_envelope::{
        EmbeddingPayload, RelationalPayload, TraceHeader, VectorAnnPayload,
    };
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use proximadb_observability_engine::io_trace::{
        ExecOpTrace, VectorAccessPath, VectorAccessTrace, VectorSearchIntent,
    };

    fn store_url(dir: &std::path::Path) -> (ProximaObjectStore, String) {
        let url = format!("file://{}", dir.display());
        (ProximaObjectStore::from_url(&url).unwrap(), url)
    }

    fn header(writer: &str, seq: u64) -> TraceHeader {
        TraceHeader {
            query_id: Some(format!("q-{writer}-{seq}")),
            tenant: Some("acme".to_string()),
            event_time_unix_ms: 1_700_000_000_000 + seq,
            bytes_read: 1024 * (seq + 1),
            range_gets: seq + 1,
            ..Default::default()
        }
    }

    fn env(writer: &str, seq: u64, payload: TracePayload) -> TraceEnvelope {
        TraceEnvelope {
            schema_version: 2,
            writer_uuid: writer.to_string(),
            sequence: seq,
            header: header(writer, seq),
            payload,
        }
    }

    /// TD-IOTRACE-4: the header batch carries the persistent-L2 probe columns
    /// (appended, nullable — pre-existing parquet segments lack them and read
    /// as NULL).
    #[test]
    fn header_batch_carries_l2_probe_columns() {
        let mut e = env("w", 1, TracePayload::Generic);
        e.header.survivor_l1_hits = 11;
        e.header.survivor_l1_misses = 4;
        e.header.l2_hits = 9;
        e.header.l2_misses = 3;
        let batch = build_header_batch(&[e]).unwrap().unwrap();
        let schema = batch.schema();
        let (i1h, i1m) = (
            schema.index_of("survivor_l1_hits").unwrap(),
            schema.index_of("survivor_l1_misses").unwrap(),
        );
        let (ih, im) = (
            schema.index_of("l2_hits").unwrap(),
            schema.index_of("l2_misses").unwrap(),
        );
        assert!(schema.field(i1h).is_nullable() && schema.field(i1m).is_nullable());
        assert!(schema.field(ih).is_nullable() && schema.field(im).is_nullable());
        let l1_hits = batch
            .column(i1h)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let l1_misses = batch
            .column(i1m)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let hits = batch
            .column(ih)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let misses = batch
            .column(im)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!((l1_hits.value(0), l1_misses.value(0)), (11, 4));
        assert_eq!((hits.value(0), misses.value(0)), (9, 3));
    }

    #[test]
    fn vector_access_batch_carries_route_time_dimensions() {
        use proximadb_observability_engine::io_trace::{VectorCachePolicy, VectorStorageScope};

        let e = env(
            "w",
            1,
            TracePayload::VectorAnn(VectorAnnPayload {
                access_paths: vec![VectorAccessTrace {
                    engine: "sst".to_string(),
                    dimensions: 384,
                    top_k: 10,
                    has_filter: false,
                    requested_mode: VectorSearchIntent::Exact,
                    actual_path: VectorAccessPath::Exact,
                    storage_scope: VectorStorageScope::Remote,
                    cache_policy: VectorCachePolicy::Full,
                }],
                ..Default::default()
            }),
        );
        let batch = build_vector_access_batch(&[e]).unwrap().unwrap();
        let index = batch.schema().index_of("storage_scope").unwrap();
        assert!(batch.schema().field(index).is_nullable());
        let values = batch
            .column(index)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.value(0), "remote");
        let index = batch.schema().index_of("cache_policy").unwrap();
        assert!(batch.schema().field(index).is_nullable());
        let values = batch
            .column(index)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.value(0), "full");
    }

    fn relational(writer: &str, seq: u64, n_exec: usize) -> TraceEnvelope {
        let exec = (0..n_exec)
            .map(|i| ExecOpTrace {
                op: format!("Op{i}"),
                rows_in: 100,
                rows_out: 40,
                ms_self: 5,
                bytes: Some(2048),
                spill: false,
            })
            .collect();
        env(
            writer,
            seq,
            TracePayload::Relational(RelationalPayload {
                plan_nodes: 7,
                plan_depth: 4,
                exec,
                ..Default::default()
            }),
        )
    }

    /// Write a JSONL+zstd segment of `envelopes` under the source prefix, named so
    /// its stem is unique.
    async fn write_segment(store: &ProximaObjectStore, name: &str, envelopes: &[TraceEnvelope]) {
        let mut jsonl = String::new();
        for e in envelopes {
            jsonl.push_str(&serde_json::to_string(e).unwrap());
            jsonl.push('\n');
        }
        let compressed = zstd::encode_all(jsonl.as_bytes(), 3).unwrap();
        let key = format!("_operator/io_trace/{name}.jsonl.zst");
        store
            .put(&ObjectPath::from(key), Bytes::from(compressed))
            .await
            .unwrap();
    }

    async fn read_table(store: &ProximaObjectStore, table: &str) -> Vec<RecordBatch> {
        let prefix = ObjectPath::from(format!("_operator/warehouse/{table}/data"));
        let base = store.base().as_ref().to_string();
        let metas = store.list(Some(&prefix)).await.unwrap();
        let mut out = Vec::new();
        for meta in metas {
            let key = strip_base(&base, meta.location.as_ref());
            if !key.ends_with(".parquet") {
                continue;
            }
            let bytes = store.get(&ObjectPath::from(key)).await.unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
                .unwrap()
                .build()
                .unwrap();
            for b in reader {
                out.push(b.unwrap());
            }
        }
        out
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    fn warehouse(url: &str) -> IoTraceWarehouse {
        IoTraceWarehouse::new(
            ProximaObjectStore::from_url(url).unwrap(),
            "_operator/io_trace/",
            "_operator/warehouse/",
        )
    }

    #[tokio::test]
    async fn compacts_star_schema_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        write_segment(
            &store,
            "trace-seg-a",
            &[
                relational("w1", 0, 2),
                env(
                    "w1",
                    1,
                    TracePayload::VectorAnn(VectorAnnPayload {
                        access_paths: vec![
                            VectorAccessTrace {
                                engine: "sst".to_string(),
                                dimensions: 384,
                                top_k: 10,
                                has_filter: false,
                                requested_mode: VectorSearchIntent::Exact,
                                actual_path: VectorAccessPath::Exact,
                                storage_scope: proximadb_observability_engine::io_trace::VectorStorageScope::Local,
                                cache_policy: proximadb_observability_engine::io_trace::VectorCachePolicy::Disabled,
                            },
                            VectorAccessTrace {
                                engine: "sst".to_string(),
                                dimensions: 768,
                                top_k: 25,
                                has_filter: true,
                                requested_mode: VectorSearchIntent::Approximate,
                                actual_path: VectorAccessPath::Ann,
                                storage_scope: proximadb_observability_engine::io_trace::VectorStorageScope::Remote,
                                cache_policy: proximadb_observability_engine::io_trace::VectorCachePolicy::Full,
                            },
                        ],
                        centroid_total_blocks: 64,
                        centroid_pruned_blocks: 50,
                        ..Default::default()
                    }),
                ),
                env(
                    "w1",
                    2,
                    TracePayload::Embedding(EmbeddingPayload {
                        embedding_calls: 3,
                        embedding_input_tokens: 500,
                        embedding_output_tokens: 200,
                    }),
                ),
                env("w1", 3, TracePayload::Generic),
            ],
        )
        .await;

        let summary = warehouse(&url).compact(1_700_000_000_000).await.unwrap();
        assert_eq!(summary.segments_processed, 1);
        assert_eq!(summary.records_ingested, 4);
        assert_eq!(summary.header_rows, 4, "one header row per record");
        assert_eq!(summary.relational_rows, 1);
        assert_eq!(summary.exec_rows, 2, "exec[] flattened to one row per op");
        assert_eq!(summary.vector_rows, 1);
        assert_eq!(summary.vector_access_rows, 2);
        assert_eq!(summary.embedding_rows, 1);

        // Read the tables back from Parquet.
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 4);
        assert_eq!(total_rows(&read_table(&store, T_EXEC).await), 2);
        assert_eq!(total_rows(&read_table(&store, T_VECTOR).await), 1);
        assert_eq!(
            total_rows(&read_table(&store, T_VECTOR_ACCESS).await),
            2,
            "one normalized row per vector operator"
        );
        assert_eq!(total_rows(&read_table(&store, T_EMBEDDING).await), 1);
        assert_eq!(total_rows(&read_table(&store, T_RELATIONAL).await), 1);

        // An Iceberg metadata log exists for the header table.
        let md = store
            .list(Some(&ObjectPath::from(
                "_operator/warehouse/trace_header/metadata".to_string(),
            )))
            .await
            .unwrap();
        assert!(
            !md.is_empty(),
            "iceberg metadata committed for trace_header"
        );
    }

    #[tokio::test]
    async fn dedups_within_segment_by_writer_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        // Same {writer_uuid, sequence} twice → one header row.
        write_segment(
            &store,
            "trace-dup",
            &[
                relational("w1", 7, 0),
                relational("w1", 7, 0),
                relational("w1", 8, 0),
            ],
        )
        .await;
        let summary = warehouse(&url).compact(1).await.unwrap();
        assert_eq!(summary.records_ingested, 3);
        assert_eq!(summary.duplicates_dropped, 1);
        assert_eq!(summary.header_rows, 2);
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 2);
    }

    #[tokio::test]
    async fn watermark_skips_processed_segments() {
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        write_segment(&store, "trace-1", &[relational("w1", 0, 1)]).await;
        let first = warehouse(&url).compact(1).await.unwrap();
        assert_eq!(first.segments_processed, 1);

        // Second run with no new segments → nothing processed, table unchanged.
        let second = warehouse(&url).compact(2).await.unwrap();
        assert_eq!(second.segments_processed, 0);
        assert_eq!(second.header_rows, 0);
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 1);

        // A new segment is picked up incrementally.
        write_segment(&store, "trace-2", &[relational("w1", 1, 1)]).await;
        let third = warehouse(&url).compact(3).await.unwrap();
        assert_eq!(third.segments_processed, 1);
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 2);
    }

    /// Crash/retry idempotency: reprocessing a segment (watermark not advanced)
    /// overwrites the same deterministic Parquet file and republishes the same file
    /// set — the table stays at exactly one row per record, no duplicates.
    #[tokio::test]
    async fn reprocessing_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        write_segment(
            &store,
            "trace-x",
            &[relational("w1", 0, 1), relational("w1", 1, 1)],
        )
        .await;

        warehouse(&url).compact(1).await.unwrap();
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 2);

        // Simulate a crash-before-watermark: wipe the watermark, recompact.
        store
            .delete(&ObjectPath::from(
                "_operator/warehouse/_watermark.json".to_string(),
            ))
            .await
            .ok();
        let retry = warehouse(&url).compact(2).await.unwrap();
        assert_eq!(retry.segments_processed, 1, "segment reprocessed");
        assert_eq!(
            total_rows(&read_table(&store, T_HEADER).await),
            2,
            "no duplicate rows after reprocess"
        );
    }

    /// Schema-evolution tolerance: an unknown-modality record (a newer writer) is
    /// still compacted — it lands one header row and no satellite row, without
    /// failing the run.
    #[tokio::test]
    async fn unknown_modality_record_is_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        // Hand-write a segment with a future modality tag + a malformed line.
        let future = r#"{"schema_version":9,"writer_uuid":"w9","sequence":0,"header":{"event_time_unix_ms":1,"bytes_read":5},"payload":{"modality":"rank_fusion","top_k":8}}"#;
        let malformed = "{not json";
        let jsonl = format!("{future}\n{malformed}\n");
        let compressed = zstd::encode_all(jsonl.as_bytes(), 3).unwrap();
        store
            .put(
                &ObjectPath::from("_operator/io_trace/trace-future.jsonl.zst".to_string()),
                Bytes::from(compressed),
            )
            .await
            .unwrap();

        let summary = warehouse(&url).compact(1).await.unwrap();
        assert_eq!(summary.records_ingested, 1, "malformed line skipped");
        assert_eq!(
            summary.header_rows, 1,
            "unknown modality still lands a header row"
        );
        assert_eq!(summary.relational_rows, 0);
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 1);
    }

    /// The per-engine `compute_ms` map flattens into `trace_compute` — one row per
    /// (record, engine) — so cost analytics can compare engines per query shape.
    #[tokio::test]
    async fn compacts_per_engine_compute_satellite() {
        use arrow_array::Array;
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        let mut e = relational("w1", 0, 0);
        e.header.compute_ms = std::collections::BTreeMap::from([
            ("datafusion".to_string(), 12),
            ("native".to_string(), 3),
        ]);
        write_segment(&store, "trace-compute", &[e]).await;

        let summary = warehouse(&url).compact(1).await.unwrap();
        assert_eq!(summary.compute_rows, 2, "one trace_compute row per engine");

        let batches = read_table(&store, T_COMPUTE).await;
        assert_eq!(total_rows(&batches), 2);
        let b = &batches[0];
        let engines = b
            .column(b.schema().index_of("engine").unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let names: Vec<&str> = (0..engines.len()).map(|i| engines.value(i)).collect();
        assert!(
            names.contains(&"datafusion") && names.contains(&"native"),
            "engines flattened: {names:?}"
        );
    }

    static TRIGGER_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

    /// S4b: install the background compactor over a `file://` store; a duplicate
    /// install is rejected, and a graceful shutdown runs a final compaction pass that
    /// lands the pending segment in the warehouse.
    #[tokio::test]
    async fn install_shutdown_runs_final_compaction() {
        let _g = TRIGGER_TEST_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let dir = tempfile::tempdir().unwrap();
        let (store, url) = store_url(dir.path());
        write_segment(&store, "trace-trig", &[relational("w1", 0, 1)]).await;
        // A long interval means only the shutdown final-pass compacts (deterministic).
        install(url.clone(), 3600).unwrap();
        assert!(
            install(url.clone(), 3600).is_err(),
            "duplicate install rejected while the first worker is live"
        );
        shutdown().await;
        assert_eq!(total_rows(&read_table(&store, T_HEADER).await), 1);
    }
}
