// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Crash-aware `io_trace` delivery sink (ADR-066 / TD-TRACE-2 S1–S2b).
//!
//! The query path only serializes and enqueues into bounded memory. A background
//! worker compresses each segment, durably seals an immutable `.pending` file,
//! conditionally creates the object, verifies collisions, and only then retires
//! the pending file. Startup retries every pending file.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use chrono::Utc;
use object_store::path::Path as ObjectPath;
use proximadb_kernel::error::StorageError;
use proximadb_object_store::ProximaObjectStore;
use proximadb_storage_filesystem_types::ObjectAccessTier;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::core::config::ResolvedIoTraceSinkConfig;
use crate::metrics::io_trace_sink_metrics as delivery_metrics;
use crate::observability::io_trace;
use crate::observability::trace_envelope::TraceEnvelope;
use crate::storage::trait_components::path_resolver::DrPathBuilder;

#[derive(Debug, PartialEq, Eq)]
struct SpoolLine {
    bytes: Vec<u8>,
    sequence: u64,
}

impl SpoolLine {
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

/// Bounded, non-blocking ingress. Overflow always drops oldest and accounts the
/// exact records and uncompressed bytes lost.
struct Spool {
    deque: VecDeque<SpoolLine>,
    bytes: usize,
    cap: usize,
}

impl Spool {
    fn new(cap: usize) -> Self {
        Self {
            deque: VecDeque::new(),
            bytes: 0,
            cap: cap.max(1),
        }
    }

    fn push(&mut self, line: SpoolLine) {
        self.bytes = self.bytes.saturating_add(line.len());
        self.deque.push_back(line);
        while self.bytes > self.cap {
            let Some(old) = self.deque.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(old.len());
            delivery_metrics::record_drop(1, old.len() as u64);
        }
    }

    /// Drain at most one target-sized segment, leaving later records queued. A
    /// single large record is returned whole rather than split.
    fn drain_segment(&mut self, target_bytes: usize) -> Vec<SpoolLine> {
        let mut out = Vec::new();
        let mut bytes = 0usize;
        while bytes < target_bytes.max(1) {
            let Some(line) = self.deque.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(line.len());
            bytes = bytes.saturating_add(line.len());
            out.push(line);
        }
        out
    }

    fn prepend(&mut self, lines: Vec<SpoolLine>) {
        for line in lines.into_iter().rev() {
            self.bytes = self.bytes.saturating_add(line.len());
            self.deque.push_front(line);
        }
        while self.bytes > self.cap {
            let Some(oldest) = self.deque.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(oldest.len());
            delivery_metrics::record_drop(1, oldest.len() as u64);
        }
    }

    fn drop_all(&mut self) {
        let records = self.deque.len() as u64;
        let bytes = self.bytes as u64;
        self.deque.clear();
        self.bytes = 0;
        if records > 0 {
            delivery_metrics::record_drop(records, bytes);
        }
    }
}

struct SinkHandle {
    shutdown: tokio::sync::watch::Sender<bool>,
    join: tokio::task::JoinHandle<()>,
}

static SINK: OnceLock<Mutex<Option<SinkHandle>>> = OnceLock::new();
static SINK_ACTIVE: AtomicBool = AtomicBool::new(false);

fn sink_slot() -> &'static Mutex<Option<SinkHandle>> {
    SINK.get_or_init(|| Mutex::new(None))
}

type SharedSpool = Arc<Mutex<Spool>>;

/// Install exactly one sink worker. Duplicate installation is rejected without
/// replacing the observer or aborting the live worker.
pub fn install(cfg: ResolvedIoTraceSinkConfig) -> Result<(), String> {
    let mut slot = sink_slot().lock().unwrap_or_else(|p| p.into_inner());
    if SINK_ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        delivery_metrics::DUPLICATE_INSTALLS_TOTAL.inc();
        return Err("io_trace sink is already installed".to_string());
    }

    if let Err(e) = std::fs::create_dir_all(&cfg.local_dir) {
        SINK_ACTIVE.store(false, Ordering::Release);
        return Err(format!(
            "io_trace sink cannot create local_dir {}: {e}",
            cfg.local_dir
        ));
    }

    let object_store = match cfg.object_store_uri.as_deref() {
        Some(uri) => match ProximaObjectStore::from_url(uri) {
            Ok(store) => Some(store),
            Err(e) => {
                SINK_ACTIVE.store(false, Ordering::Release);
                return Err(format!(
                    "io_trace sink object_store_uri {uri} is invalid: {e}"
                ));
            }
        },
        None => None,
    };
    let writer_uuid = Uuid::new_v4().to_string();
    let next_sequence = Arc::new(AtomicU64::new(0));
    let spool: SharedSpool = Arc::new(Mutex::new(Spool::new(cfg.spool_max_bytes as usize)));

    let spool_obs = Arc::clone(&spool);
    let sequence_obs = Arc::clone(&next_sequence);
    let writer_obs = writer_uuid.clone();
    io_trace::set_trace_observer(Some(Box::new(move |snap, tenant| {
        let sequence = sequence_obs.fetch_add(1, Ordering::Relaxed);
        let event_time_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
            .unwrap_or(0);
        // S3 (ADR-066 D1): serialize the header + modality-tagged envelope, NOT a
        // flat snapshot. Billing is untouched — it reads the in-memory snapshot
        // directly, so meters stay byte-identical; the envelope is only this sink's
        // durable serialization view.
        let envelope =
            TraceEnvelope::from_snapshot(snap, tenant, &writer_obs, sequence, event_time_unix_ms);
        match serde_json::to_vec(&envelope) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                spool_obs
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(SpoolLine { bytes, sequence });
            }
            Err(e) => {
                delivery_metrics::record_drop(1, 0);
                tracing::warn!("io_trace sink: record serialization failed: {e}");
            }
        }
    })));

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let worker = Worker::new(cfg, spool, object_store, writer_uuid);
    let join = tokio::spawn(worker.run(shutdown_rx));
    *slot = Some(SinkHandle {
        shutdown: shutdown_tx,
        join,
    });
    tracing::info!("io_trace sink installed (best-effort ingress, crash-aware delivery)");
    Ok(())
}

/// Stop accepting records and request a final delivery cycle. Timing out is
/// observable and detaches (rather than aborts) the worker so sealed work can
/// still finish.
pub async fn shutdown() {
    let handle = sink_slot().lock().unwrap_or_else(|p| p.into_inner()).take();
    if let Some(handle) = handle {
        io_trace::set_trace_observer(None);
        let _ = handle.shutdown.send(true);
        match tokio::time::timeout(Duration::from_secs(5), handle.join).await {
            Ok(Ok(())) => {
                SINK_ACTIVE.store(false, Ordering::Release);
                tracing::info!("io_trace sink flushed and stopped");
            }
            Ok(Err(e)) => {
                SINK_ACTIVE.store(false, Ordering::Release);
                tracing::warn!("io_trace sink worker join error: {e}");
            }
            Err(_) => {
                delivery_metrics::SHUTDOWN_TIMEOUTS_TOTAL.inc();
                tracing::warn!("io_trace sink flush timed out (5s); worker left running");
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PendingSegment {
    path: PathBuf,
    object_filename: String,
    date: String,
    digest: String,
}

impl PendingSegment {
    fn parse(path: PathBuf) -> Result<Self, String> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "pending filename is not UTF-8".to_string())?;
        let object_filename = filename
            .strip_suffix(".pending")
            .ok_or_else(|| format!("not a pending trace file: {filename}"))?
            .to_string();
        let stem = object_filename
            .strip_suffix(".jsonl.zst")
            .ok_or_else(|| format!("invalid pending trace suffix: {filename}"))?;
        let mut parts = stem.rsplitn(4, '-');
        let digest = parts.next().unwrap_or_default().to_string();
        let last = parts.next().unwrap_or_default();
        let first = parts.next().unwrap_or_default();
        let prefix = parts.next().unwrap_or_default();
        if digest.len() != 64
            || !digest.bytes().all(|b| b.is_ascii_hexdigit())
            || first.parse::<u64>().is_err()
            || last.parse::<u64>().is_err()
            || !prefix.starts_with("trace-")
        {
            return Err(format!("invalid pending trace identity: {filename}"));
        }
        let identity = &prefix["trace-".len()..];
        let date = identity
            .get(..10)
            .filter(|d| d.as_bytes().get(4) == Some(&b'-') && d.as_bytes().get(7) == Some(&b'-'))
            .ok_or_else(|| format!("invalid pending trace date: {filename}"))?
            .to_string();
        Ok(Self {
            path,
            object_filename,
            date,
            digest,
        })
    }
}

struct Worker {
    cfg: ResolvedIoTraceSinkConfig,
    spool: SharedSpool,
    writer_uuid: String,
    object_store: Option<ProximaObjectStore>,
    tier: ObjectAccessTier,
    trace_prefix: String,
    pending_bytes: u64,
}

impl Worker {
    fn new(
        cfg: ResolvedIoTraceSinkConfig,
        spool: SharedSpool,
        object_store: Option<ProximaObjectStore>,
        writer_uuid: String,
    ) -> Self {
        let tier = cfg.access_tier;
        let trace_prefix = DrPathBuilder::operator_subprefix("io_trace")
            .unwrap_or_else(|_| "_operator/io_trace/".to_string());
        Self {
            cfg,
            spool,
            writer_uuid,
            object_store,
            tier,
            trace_prefix,
            pending_bytes: 0,
        }
    }

    async fn run(mut self, mut shutdown_rx: tokio::sync::watch::Receiver<bool>) {
        self.delivery_cycle().await;
        let mut interval = tokio::time::interval(Duration::from_secs(self.cfg.flush_interval_s));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Consume the immediate first tick; startup delivery above already did the work.
        interval.tick().await;
        loop {
            tokio::select! {
                _ = interval.tick() => self.delivery_cycle().await,
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        self.delivery_cycle().await;
                        if self.object_store.is_some()
                            && self.pending_bytes >= self.cfg.pending_max_bytes
                        {
                            self.spool.lock().unwrap_or_else(|p| p.into_inner()).drop_all();
                        }
                        return;
                    }
                }
            }
        }
    }

    async fn delivery_cycle(&mut self) {
        if self.object_store.is_some() {
            self.retry_pending().await;
        } else {
            self.refresh_pending_metrics().await;
        }

        loop {
            if self.object_store.is_some() && self.pending_bytes >= self.cfg.pending_max_bytes {
                break;
            }
            let lines = self
                .spool
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .drain_segment(self.cfg.segment_bytes as usize);
            if lines.is_empty() {
                break;
            }
            if !self.seal(lines).await {
                break;
            }
        }
    }

    /// Returns false when the exact compressed segment would cross the durable
    /// pending cap. In that case the original lines are returned to ingress.
    async fn seal(&mut self, lines: Vec<SpoolLine>) -> bool {
        let records = lines.len() as u64;
        let first_sequence = lines.first().map(|l| l.sequence).unwrap_or(0);
        let last_sequence = lines.last().map(|l| l.sequence).unwrap_or(first_sequence);
        let uncompressed_bytes = lines.iter().map(SpoolLine::len).sum::<usize>() as u64;
        let mut raw = Vec::with_capacity(uncompressed_bytes as usize);
        for line in &lines {
            raw.extend_from_slice(&line.bytes);
        }
        let compressed =
            match tokio::task::spawn_blocking(move || zstd::encode_all(&raw[..], 3)).await {
                Ok(Ok(data)) => data,
                Ok(Err(e)) => {
                    delivery_metrics::SEAL_FAILURES_TOTAL.inc();
                    delivery_metrics::record_drop(records, uncompressed_bytes);
                    tracing::warn!("io_trace sink: compression failed: {e}");
                    return true;
                }
                Err(e) => {
                    delivery_metrics::SEAL_FAILURES_TOTAL.inc();
                    delivery_metrics::record_drop(records, uncompressed_bytes);
                    tracing::warn!("io_trace sink: compression task failed: {e}");
                    return true;
                }
            };
        let digest = digest_hex(&compressed);
        let date = Utc::now().format("%Y-%m-%d").to_string();
        let filename = format!(
            "trace-{date}-{}-{first_sequence:020}-{last_sequence:020}-{digest}.jsonl.zst",
            self.writer_uuid
        );

        if self.object_store.is_some() {
            if self.pending_bytes.saturating_add(compressed.len() as u64)
                > self.cfg.pending_max_bytes
            {
                self.spool
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .prepend(lines);
                return false;
            }
            let pending = match self.persist_pending(&filename, compressed).await {
                Ok(pending) => pending,
                Err(e) => {
                    delivery_metrics::SEAL_FAILURES_TOTAL.inc();
                    delivery_metrics::record_drop(records, uncompressed_bytes);
                    tracing::warn!("io_trace sink: durable pending seal failed: {e}");
                    return true;
                }
            };
            self.refresh_pending_metrics().await;
            self.upload_pending(pending, false).await;
        } else if let Err(e) = self.write_local_final(&filename, compressed).await {
            delivery_metrics::SEAL_FAILURES_TOTAL.inc();
            delivery_metrics::record_drop(records, uncompressed_bytes);
            tracing::warn!("io_trace sink: immutable local seal failed: {e}");
        }
        true
    }

    async fn persist_pending(
        &self,
        filename: &str,
        data: Vec<u8>,
    ) -> Result<PendingSegment, String> {
        let path = Path::new(&self.cfg.local_dir).join(format!("{filename}.pending"));
        let path_for_write = path.clone();
        tokio::task::spawn_blocking(move || write_immutable(&path_for_write, &data))
            .await
            .map_err(|e| e.to_string())??;
        PendingSegment::parse(path)
    }

    async fn write_local_final(&self, filename: &str, data: Vec<u8>) -> Result<(), String> {
        let path = Path::new(&self.cfg.local_dir).join(filename);
        tokio::task::spawn_blocking(move || write_immutable(&path, &data))
            .await
            .map_err(|e| e.to_string())?
    }

    async fn retry_pending(&mut self) {
        let mut pending = self.inventory_pending().await;
        pending.sort_by(|a, b| a.object_filename.cmp(&b.object_filename));
        for segment in pending {
            self.upload_pending(segment, true).await;
        }
        self.refresh_pending_metrics().await;
    }

    async fn upload_pending(&mut self, pending: PendingSegment, retry: bool) {
        let Some(store) = self.object_store.clone() else {
            return;
        };
        if retry {
            delivery_metrics::UPLOAD_RETRIES_TOTAL.inc();
        }
        let path = pending.path.clone();
        let data = match tokio::task::spawn_blocking(move || std::fs::read(path)).await {
            Ok(Ok(data)) => data,
            Ok(Err(e)) => {
                delivery_metrics::UPLOAD_FAILURES_TOTAL.inc();
                tracing::warn!("io_trace sink: pending read failed: {e}");
                return;
            }
            Err(e) => {
                delivery_metrics::UPLOAD_FAILURES_TOTAL.inc();
                tracing::warn!("io_trace sink: pending read task failed: {e}");
                return;
            }
        };
        if digest_hex(&data) != pending.digest {
            delivery_metrics::UPLOAD_FAILURES_TOTAL.inc();
            tracing::error!(
                "io_trace sink: pending digest mismatch; refusing upload: {}",
                pending.path.display()
            );
            return;
        }

        let key = format!(
            "{}{}/{}",
            self.trace_prefix, pending.date, pending.object_filename
        );
        let object_path = ObjectPath::from(key.clone());
        let verified = match store
            .put_if_absent_with_tier(&object_path, Bytes::from(data), self.tier)
            .await
        {
            Ok(()) => true,
            Err(StorageError::AlreadyExists(_)) => match store.get(&object_path).await {
                Ok(existing) if digest_hex(&existing) == pending.digest => true,
                Ok(_) => {
                    tracing::error!("io_trace sink: object collision has different content: {key}");
                    false
                }
                Err(e) => {
                    tracing::warn!("io_trace sink: cannot verify existing object {key}: {e}");
                    false
                }
            },
            Err(e) => {
                tracing::warn!("io_trace sink: conditional object PUT {key} failed: {e}");
                false
            }
        };
        if !verified {
            delivery_metrics::UPLOAD_FAILURES_TOTAL.inc();
            return;
        }

        delivery_metrics::record_delivery_success();
        let retire = pending.path.clone();
        match tokio::task::spawn_blocking(move || std::fs::remove_file(retire)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(
                "io_trace sink: verified object but pending retirement failed (safe to retry): {e}"
            ),
            Err(e) => tracing::warn!("io_trace sink: pending retirement task failed: {e}"),
        }
        self.refresh_pending_metrics().await;
    }

    async fn inventory_pending(&self) -> Vec<PendingSegment> {
        let dir = self.cfg.local_dir.clone();
        match tokio::task::spawn_blocking(move || inventory_pending_sync(Path::new(&dir))).await {
            Ok(items) => items,
            Err(e) => {
                tracing::warn!("io_trace sink: pending inventory task failed: {e}");
                Vec::new()
            }
        }
    }

    async fn refresh_pending_metrics(&mut self) {
        let dir = self.cfg.local_dir.clone();
        let (files, bytes) = tokio::task::spawn_blocking(move || pending_usage(Path::new(&dir)))
            .await
            .unwrap_or((0, 0));
        self.pending_bytes = bytes;
        delivery_metrics::set_pending(files, bytes);
    }
}

fn digest_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// Commit immutable bytes with create-only semantics. A fully synced hidden temp
/// file is atomically linked into the visible name; an existing identical file is
/// an idempotent success, while different bytes fail closed.
fn write_immutable(path: &Path, data: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "immutable filename is not UTF-8".to_string())?;
    let temp = parent.join(format!(".{file_name}.{}.tmp", Uuid::new_v4()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|e| e.to_string())?;
    if let Err(e) = file.write_all(data).and_then(|_| file.sync_all()) {
        let _ = std::fs::remove_file(&temp);
        return Err(e.to_string());
    }
    drop(file);
    match std::fs::hard_link(&temp, path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&temp);
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(path).map_err(|read| read.to_string());
            let _ = std::fs::remove_file(&temp);
            match existing {
                Ok(existing) if existing == data => Ok(()),
                Ok(_) => Err(format!(
                    "immutable path already exists with different content: {}",
                    path.display()
                )),
                Err(read) => Err(read),
            }
        }
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            Err(e.to_string())
        }
    }
}

fn inventory_pending_sync(dir: &Path) -> Vec<PendingSegment> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().ends_with(".jsonl.zst.pending"))
        .filter_map(|path| match PendingSegment::parse(path) {
            Ok(segment) => Some(segment),
            Err(e) => {
                delivery_metrics::UPLOAD_FAILURES_TOTAL.inc();
                tracing::error!("io_trace sink: invalid pending file left in place: {e}");
                None
            }
        })
        .collect()
}

fn pending_usage(dir: &Path) -> (u64, u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (0, 0);
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().ends_with(".pending"))
        .fold((0u64, 0u64), |(files, bytes), path| {
            let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            (files.saturating_add(1), bytes.saturating_add(len))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    static INSTALL_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

    fn cfg(dir: &str, seg: u64, spool: u64) -> ResolvedIoTraceSinkConfig {
        ResolvedIoTraceSinkConfig {
            local_dir: dir.to_string(),
            segment_bytes: seg,
            flush_interval_s: 1,
            spool_max_bytes: spool,
            pending_max_bytes: 1 << 20,
            compression: "zstd".to_string(),
            format: "jsonl".to_string(),
            object_store_uri: None,
            access_tier: ObjectAccessTier::Cold,
            warehouse_compaction_interval_s: None,
        }
    }

    fn line(sequence: u64, bytes: &[u8]) -> SpoolLine {
        SpoolLine {
            bytes: bytes.to_vec(),
            sequence,
        }
    }

    fn tempdir(label: &str) -> tempfile::TempDir {
        tempfile::Builder::new().prefix(label).tempdir().unwrap()
    }

    fn trace_files_recursive(root: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut dirs = vec![root.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.to_string_lossy().ends_with(".jsonl.zst") {
                    out.push(path);
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn spool_drops_oldest_on_overflow() {
        let before = delivery_metrics::RECORDS_DROPPED_TOTAL.get();
        let mut spool = Spool::new(10);
        spool.push(line(0, b"aaaa"));
        spool.push(line(1, b"bbbb"));
        spool.push(line(2, b"cccc"));
        assert_eq!(spool.bytes, 8);
        assert_eq!(delivery_metrics::RECORDS_DROPPED_TOTAL.get(), before + 1);
        assert_eq!(
            spool.drain_segment(100),
            vec![line(1, b"bbbb"), line(2, b"cccc")]
        );
    }

    #[tokio::test]
    async fn local_segment_round_trips_with_digest_identity() {
        let dir = tempdir("iotrace-local-");
        let spool = Arc::new(Mutex::new(Spool::new(1 << 20)));
        {
            let mut guard = spool.lock().unwrap();
            guard.push(line(7, b"{\"query_id\":\"q1\"}\n"));
            guard.push(line(8, b"{\"query_id\":\"q2\"}\n"));
        }
        let mut worker = Worker::new(
            cfg(&dir.path().to_string_lossy(), 1 << 20, 1 << 20),
            spool,
            None,
            Uuid::new_v4().to_string(),
        );
        worker.delivery_cycle().await;
        let files = trace_files_recursive(dir.path());
        assert_eq!(files.len(), 1);
        let text =
            String::from_utf8(zstd::decode_all(&std::fs::read(&files[0]).unwrap()[..]).unwrap())
                .unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(files[0].to_string_lossy().contains("00000000000000000007"));
        assert!(files[0].to_string_lossy().contains("00000000000000000008"));
    }

    #[tokio::test]
    async fn startup_retry_delivers_once_and_retires_pending() {
        let base = tempdir("iotrace-retry-");
        let local = base.path().join("local");
        let objects = base.path().join("objects");
        std::fs::create_dir_all(&local).unwrap();
        std::fs::create_dir_all(&objects).unwrap();
        let store = ProximaObjectStore::from_url(&format!("file://{}", objects.display())).unwrap();
        let spool = Arc::new(Mutex::new(Spool::new(1024)));
        let mut worker = Worker::new(
            cfg(&local.to_string_lossy(), 1024, 1024),
            spool,
            Some(store),
            Uuid::new_v4().to_string(),
        );
        let compressed = zstd::encode_all(&b"{\"query_id\":\"q\"}\n"[..], 3).unwrap();
        let digest = digest_hex(&compressed);
        let name = format!(
            "trace-2026-07-18-{}-{:020}-{:020}-{digest}.jsonl.zst",
            worker.writer_uuid, 0, 0
        );
        let pending = worker.persist_pending(&name, compressed).await.unwrap();
        assert!(pending.path.exists());

        worker.retry_pending().await;
        assert!(!pending.path.exists());
        assert_eq!(trace_files_recursive(&objects).len(), 1);
        worker.retry_pending().await;
        assert_eq!(trace_files_recursive(&objects).len(), 1);
    }

    #[tokio::test]
    async fn already_exists_with_different_content_fails_closed() {
        let base = tempdir("iotrace-collision-");
        let local = base.path().join("local");
        let objects = base.path().join("objects");
        std::fs::create_dir_all(&local).unwrap();
        std::fs::create_dir_all(&objects).unwrap();
        let store = ProximaObjectStore::from_url(&format!("file://{}", objects.display())).unwrap();
        let mut worker = Worker::new(
            cfg(&local.to_string_lossy(), 1024, 1024),
            Arc::new(Mutex::new(Spool::new(1024))),
            Some(store.clone()),
            Uuid::new_v4().to_string(),
        );
        let compressed = zstd::encode_all(&b"correct\n"[..], 3).unwrap();
        let digest = digest_hex(&compressed);
        let name = format!(
            "trace-2026-07-18-{}-{:020}-{:020}-{digest}.jsonl.zst",
            worker.writer_uuid, 4, 4
        );
        let pending = worker.persist_pending(&name, compressed).await.unwrap();
        let key = ObjectPath::from(format!("_operator/io_trace/2026-07-18/{name}"));
        store
            .put_if_absent(&key, Bytes::from_static(b"different"))
            .await
            .unwrap();

        worker.upload_pending(pending.clone(), true).await;
        assert!(
            pending.path.exists(),
            "unverified pending must be preserved"
        );
        assert_eq!(&store.get(&key).await.unwrap()[..], b"different");
    }

    #[tokio::test]
    async fn pending_cap_preserves_sealed_file_and_stops_ingress_drain() {
        let base = tempdir("iotrace-cap-");
        let local = base.path().join("local");
        let objects = base.path().join("objects");
        std::fs::create_dir_all(&local).unwrap();
        std::fs::create_dir_all(&objects).unwrap();
        let store = ProximaObjectStore::from_url(&format!("file://{}", objects.display())).unwrap();
        let spool = Arc::new(Mutex::new(Spool::new(1024)));
        spool.lock().unwrap().push(line(9, b"queued\n"));
        let mut config = cfg(&local.to_string_lossy(), 1024, 1024);
        config.pending_max_bytes = 1;
        let mut worker = Worker::new(
            config,
            Arc::clone(&spool),
            Some(store.clone()),
            Uuid::new_v4().to_string(),
        );
        let compressed = zstd::encode_all(&b"sealed\n"[..], 3).unwrap();
        let digest = digest_hex(&compressed);
        let name = format!(
            "trace-2026-07-18-{}-{:020}-{:020}-{digest}.jsonl.zst",
            worker.writer_uuid, 1, 1
        );
        let pending = worker.persist_pending(&name, compressed).await.unwrap();
        let key = ObjectPath::from(format!("_operator/io_trace/2026-07-18/{name}"));
        store
            .put_if_absent(&key, Bytes::from_static(b"collision"))
            .await
            .unwrap();

        worker.delivery_cycle().await;
        assert!(
            pending.path.exists(),
            "sealed pending data is never evicted"
        );
        assert_eq!(spool.lock().unwrap().bytes, b"queued\n".len());
    }

    #[tokio::test]
    async fn install_shutdown_emits_record_identity() {
        let _guard = INSTALL_TEST_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let dir = tempdir("iotrace-e2e-");
        install(cfg(&dir.path().to_string_lossy(), 1 << 20, 1 << 20)).unwrap();
        io_trace::instrument(Some("acme".to_string()), "test", async {
            io_trace::record_bytes_read(128);
        })
        .await;
        shutdown().await;
        let files = trace_files_recursive(dir.path());
        assert_eq!(files.len(), 1);
        let text =
            String::from_utf8(zstd::decode_all(&std::fs::read(&files[0]).unwrap()[..]).unwrap())
                .unwrap();
        // S3 envelope shape: ingestion identity at top level, homogeneous header +
        // modality-tagged payload (ADR-066 D1).
        assert!(text.contains("\"schema_version\":2"), "envelope v2: {text}");
        assert!(text.contains("\"writer_uuid\""), "writer identity: {text}");
        assert!(text.contains("\"sequence\":0"));
        assert!(text.contains("\"header\":{"), "header object: {text}");
        assert!(
            text.contains("\"tenant\":\"acme\""),
            "tenant in header: {text}"
        );
        // Billing-class input re-exposed in the header, byte-identical.
        assert!(
            text.contains("\"bytes_read\":128"),
            "billing input in header: {text}"
        );
        // A bytes-only snapshot has no distinguishing modality → generic payload.
        assert!(
            text.contains("\"payload\":{\"modality\":\"generic\"}"),
            "generic modality payload: {text}"
        );
    }

    #[tokio::test]
    async fn duplicate_install_is_rejected_without_aborting_live_worker() {
        let _guard = INSTALL_TEST_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let first = tempdir("iotrace-first-");
        let second = tempdir("iotrace-second-");
        install(cfg(&first.path().to_string_lossy(), 1024, 1024)).unwrap();
        assert!(install(cfg(&second.path().to_string_lossy(), 1024, 1024)).is_err());
        io_trace::instrument(None, "test", async { io_trace::record_bytes_read(1) }).await;
        shutdown().await;
        assert_eq!(trace_files_recursive(first.path()).len(), 1);
        assert!(trace_files_recursive(second.path()).is_empty());
    }

    /// ADR-066 §6 #3 — the trace sink's *query-path* cost is bounded and CPU-only.
    ///
    /// The sink's observer closure (the only thing that runs synchronously on the
    /// query path) does: `TraceEnvelope::from_snapshot` + `serde_json::to_vec` + a
    /// bounded spool `push`. fsync / object-store upload / zstd compression all
    /// happen in the *background* `Worker` (off the query path). This micro-bench
    /// measures the closure's components over N representative envelopes and prints
    /// p50/p95/p99 + bytes/query for the evidence ledger — the sink's incremental
    /// per-query cost. Run:
    ///   `cargo nextest run --lib observer_closure_hot_path_is_bounded -- --ignored --nocapture`
    #[test]
    #[ignore = "ADR-066 hot-path measurement — run with --ignored --nocapture"]
    fn observer_closure_hot_path_is_bounded() {
        // Representative vector-ANN snapshot, built via the public API to avoid
        // struct-literal drift: 12 GETs, ~1.2 MB read, 8 range-gets, a centroid
        // prune, two compute engines — the shape that lands in the warehouse.
        let trace = io_trace::IoTrace::new();
        for _ in 0..12 {
            trace.record_op(io_trace::IoOp::Get);
        }
        trace.record_bytes_read(1_200_000);
        trace.record_range_gets(8);
        trace.record_centroid_prune(64, 40);
        trace.record_compute_ms("volcano", 3);
        trace.record_compute_ms("datafusion", 17);
        let snap = trace.snapshot();

        // The sink's spool (large cap ⇒ no drops during the bench).
        let spool: Arc<Mutex<Spool>> = Arc::new(Mutex::new(Spool::new(64 * 1024 * 1024)));

        const N: usize = 20_000;
        let mut samples_ns: Vec<u64> = Vec::with_capacity(N);
        let mut last_envelope_bytes = 0usize;
        for i in 0..N as u64 {
            let t0 = std::time::Instant::now();
            let envelope = TraceEnvelope::from_snapshot(
                &snap,
                Some("tenant-acme"),
                "writer-bench",
                i,
                1_700_000_000_000,
            );
            let mut bytes = serde_json::to_vec(&envelope).expect("serialize envelope");
            bytes.push(b'\n');
            last_envelope_bytes = bytes.len();
            spool
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(SpoolLine { bytes, sequence: i });
            samples_ns.push(t0.elapsed().as_nanos() as u64);
        }
        samples_ns.sort_unstable();
        let p50 = samples_ns[N / 2];
        let p95 = samples_ns[(N * 95) / 100];
        let p99 = samples_ns[(N * 99) / 100];
        eprintln!(
            "ADR-066 hot-path: N={N} observer-closure p50={p50}ns p95={p95}ns p99={p99}ns \
             envelope_bytes={last_envelope_bytes} (CPU-only: from_snapshot + serde_json + \
             spool_push; no fsync/network/compress on the query path — those are the \
             background worker)"
        );
        // Sanity guard (NOT a perf SLA): the sink must not add pathological per-query
        // overhead. 100µs is well above any representative envelope's serialize+push
        // and leaves headroom for a loaded CI runner; the evidence doc records the
        // actual p50/p95 measured on a quiet machine.
        assert!(
            p95 < 100_000,
            "sink observer p95 {p95}ns unexpectedly high — investigate"
        );
        assert!(last_envelope_bytes > 0);
    }
}
