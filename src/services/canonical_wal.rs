//! File-backed canonical WAL appenders for service-layer table writes.
//!
//! This module provides a production-oriented `TableWalAppender`
//! implementation for the canonical DML path. It writes shared
//! `CanonicalWalEntry` records instead of vector-specific operation envelopes,
//! matching the Layer 0 spine described by ADR-010.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use crc32fast::Hasher as Crc32Hasher;
use lazy_static::lazy_static;
use prometheus::{IntCounter, register_int_counter};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::services::record_store::TableWalAppender;
use crate::services::writer_fence::WriterFence;

const FRAME_MAGIC: &[u8; 8] = b"PXWAL001";
const FRAME_VERSION: u16 = 1;
/// ADR-095 D3: one frame carrying a whole transaction (`Vec<CanonicalWalEntry>`
/// payload, one CRC). Readers expand it into ordinary entries, so replay, CDC
/// and every `CanonicalOperation` consumer are unchanged. Binaries that predate
/// it reject the version rather than apply part of a transaction.
const FRAME_VERSION_GROUP: u16 = 2;
const CODEC_MSGPACK_NAMED: u8 = 1;
const FRAME_HEADER_LEN: usize = 8 + 2 + 1 + 1 + 8 + 4 + 4;

/// Test-only fault injection for the atomic commit path (TD-PGOLTP-1 crash
/// matrix). Read per append so a test server can arm it through its env.
const COMMIT_FAULT_ENV: &str = "PROXIMADB_TEST_TXN_COMMIT_FAULT";

fn torn_tail_counter() -> IntCounter {
    register_int_counter!(
        "proximadb_canonical_wal_torn_tail_truncations_total",
        "Canonical WAL opens that truncated an unacknowledged torn tail (bytes quarantined beside the WAL)"
    )
    .unwrap_or_else(|_| {
        IntCounter::new("proximadb_canonical_wal_torn_tail_truncations_total_fallback", "fallback")
            .unwrap_or_else(|_| unreachable!("valid counter metric descriptor"))
    })
}

fn segment_rotation_counter() -> IntCounter {
    register_int_counter!(
        "proximadb_canonical_wal_segment_rotations_total",
        "Canonical WAL active files sealed into segments (P2.1 segment rotation)"
    )
    .unwrap_or_else(|_| {
        IntCounter::new(
            "proximadb_canonical_wal_segment_rotations_total_fallback",
            "fallback",
        )
        .unwrap_or_else(|_| unreachable!("valid counter metric descriptor"))
    })
}

lazy_static! {
    static ref TORN_TAIL_TRUNCATIONS: IntCounter = torn_tail_counter();
    static ref SEGMENT_ROTATIONS: IntCounter = segment_rotation_counter();
}

/// Env gate for P2.1 segment rotation of the shared canonical WAL: rotate the
/// active file once it reaches this many bytes. Unset, `0` or unparsable = off.
pub const SEGMENT_BYTES_ENV: &str = "PROXIMADB_CANONICAL_WAL_SEGMENT_BYTES";

/// The configured rotation size for the shared canonical WAL (`0` = off).
pub fn canonical_wal_segment_bytes() -> u64 {
    std::env::var(SEGMENT_BYTES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// A reader racing a rotation re-reads (or waits 1 ms while one is mid-rename);
/// rotations are size-bounded (tens of MiB apart), so more than this many
/// attempts in one read means something is wrong.
const MAX_ROTATION_RETRIES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppendFault {
    FailBefore,
    AbortAfter,
    Torn,
    /// Write part of the frame, then fail as ENOSPC would.
    FailMidWrite,
    /// Write everything, then fail `sync_data`.
    FailSync,
}

fn armed_commit_fault() -> Option<AppendFault> {
    match std::env::var(COMMIT_FAULT_ENV).ok()?.as_str() {
        "fail_before_append" => Some(AppendFault::FailBefore),
        "abort_after_append" => Some(AppendFault::AbortAfter),
        "torn_append" => Some(AppendFault::Torn),
        "fail_mid_write" => Some(AppendFault::FailMidWrite),
        "fail_sync" => Some(AppendFault::FailSync),
        _ => None,
    }
}

/// Where the bytes dropped by a torn-tail truncation are kept, for forensics.
pub(crate) fn torn_tail_path(path: &Path, offset: u64) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    // Offset + time: a second torn tail at the same offset never overwrites the
    // first one's evidence.
    name.push(format!(".torn-{offset}-{nanos}"));
    path.with_file_name(name)
}

/// An append failed after bytes may have reached the WAL and they could not be
/// proven rolled back (or `sync_data` itself failed). The write may or may not
/// be durable; the appender is poisoned and refuses further appends until a
/// restart re-reads the file. Callers must report "outcome unknown" (08007),
/// never "nothing was committed".
#[derive(Debug)]
pub struct WalAppendIndeterminate(pub String);

impl std::fmt::Display for WalAppendIndeterminate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "canonical WAL append outcome is indeterminate: {}",
            self.0
        )
    }
}

impl std::error::Error for WalAppendIndeterminate {}

/// An append was refused because an EARLIER append left the appender poisoned
/// (see [`WalAppendIndeterminate`]). Nothing from THIS append was written, so
/// callers report "nothing committed", not "outcome unknown".
#[derive(Debug)]
pub struct WalAppenderPoisoned;

impl std::fmt::Display for WalAppenderPoisoned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "canonical WAL appender is poisoned by an earlier indeterminate write; \
             nothing was written; restart the server to recover",
        )
    }
}

impl std::error::Error for WalAppenderPoisoned {}

/// Framed binary canonical WAL appender.
///
/// Frame layout:
/// `[magic:8][version:u16][codec:u8][flags:u8][sequence:u64][payload_len:u32][payload_crc32:u32][payload]`.
///
/// Payloads currently use MessagePack named-field encoding for compact,
/// language-readable, schema-tolerant storage. The frame header keeps codec and
/// version explicit so later protobuf/Avro encodings can be introduced without
/// changing the `TableWalAppender` contract. Version 1 carries one entry;
/// version 2 carries an all-or-nothing group (ADR-095).
pub struct FramedTableWalAppender {
    path: PathBuf,
    next_sequence: AtomicU64,
    append_lock: Mutex<()>,
    /// ADR-095 M5: per-appender, so only the shared pgwire canonical WAL opts in.
    /// When set, a multi-operation `append_operations` is one group frame.
    atomic_groups: AtomicBool,
    /// Checked before every append; tripped by the writer-epoch watchdog.
    fence: Arc<WriterFence>,
    /// Set when an append failed in a way whose durability cannot be proven
    /// (see [`WalAppendIndeterminate`]); every later append is refused.
    poisoned: AtomicBool,
    /// TD-OLAP-17: per-collection high-water sequence_number, bumped atomically
    /// under `append_lock` on every append, AND seeded from the recovered entries
    /// on `open` (so read-only-after-restart tables fast-path immediately). Lets
    /// `read_changes_since` skip the O(WAL) scan when a collection hasn't advanced
    /// past a reader's `since_lsn` (the DF table-OPEN floor). Sync lock
    /// (parking_lot) so the sync trait method `collection_max_sequence_for` can
    /// read it without `.await`. Never regresses (worst case = scan).
    collection_max: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// P2.1 (ADR-097 S2): rotate the active file into a sealed segment once it
    /// reaches this many bytes. `0` = never rotate (the default; only the shared
    /// pgwire canonical WAL opts in via [`Self::enable_segment_rotation`]).
    segment_bytes: AtomicU64,
    /// Sealed segments (`{path}.{first_seq:020}.seg`), oldest first, with the
    /// LSN ranges a range read uses to skip them.
    sealed: parking_lot::RwLock<Vec<SegmentSummary>>,
    /// Running summary of the active file at `path`.
    active: parking_lot::Mutex<SegmentSummary>,
    /// Rotation generation, seqlock style: odd while a rotation is moving the
    /// active file into a segment, even otherwise. A reader that started on an
    /// odd value, or saw the value change mid-read, may have missed entries
    /// that moved from the active file into a new segment, and re-reads.
    rotations: AtomicU64,
}

/// LSN coverage of one WAL segment: its overall sequence range and, per
/// collection, the range of that collection's entries. A range read
/// ([`FramedTableWalAppender::read_entries_since`]) opens only segments whose
/// summary can contain a matching entry.
#[derive(Debug, Clone, Default)]
struct SegmentSummary {
    path: PathBuf,
    first_seq: Option<u64>,
    last_seq: u64,
    /// collection id → highest sequence of that collection in this segment.
    collection_max: std::collections::HashMap<String, u64>,
    /// Whether any entry carries no collection (catalog/checkpoint ops), which
    /// a collection-scoped read never needs.
    has_uncollected: bool,
}

impl SegmentSummary {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            ..Default::default()
        }
    }

    fn observe(&mut self, entry: &CanonicalWalEntry) {
        let seq = entry.sequence_number;
        self.first_seq = Some(self.first_seq.map_or(seq, |f| f.min(seq)));
        self.last_seq = self.last_seq.max(seq);
        match collection_id_of(&entry.operation) {
            Some(c) => {
                let cur = self.collection_max.entry(c.to_string()).or_insert(0);
                *cur = (*cur).max(seq);
            }
            None => self.has_uncollected = true,
        }
    }

    /// Whether this segment can hold an entry for `collection_id` with
    /// sequence `> since`.
    fn may_contain(&self, collection_id: &str, since: u64) -> bool {
        self.last_seq > since
            && self
                .collection_max
                .get(collection_id)
                .is_some_and(|max| *max > since)
    }
}

/// Path of the sealed segment whose first entry has sequence `first_seq`.
fn sealed_segment_path(active: &Path, first_seq: u64) -> PathBuf {
    let mut name = active
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(format!(".{first_seq:020}.seg"));
    active.with_file_name(name)
}

/// Sealed segments next to `active`, ordered by first sequence. Files are
/// matched exactly (`{name}.{20 digits}.seg`), so torn-tail quarantine files and
/// compaction temps are never mistaken for segments.
async fn discover_sealed_segments(active: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let (Some(dir), Some(name)) = (active.parent(), active.file_name()) else {
        return Ok(Vec::new());
    };
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let prefix = format!("{}.", name.to_string_lossy());
    let mut out = Vec::new();
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(err) => {
            return Err(err)
                .with_context(|| format!("listing canonical WAL segments in {}", dir.display()));
        }
    };
    while let Some(entry) = entries.next_entry().await? {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        let Some(rest) = file_name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".seg") else {
            continue;
        };
        if digits.len() == 20
            && digits.bytes().all(|b| b.is_ascii_digit())
            && let Ok(first_seq) = digits.parse::<u64>()
        {
            out.push((first_seq, entry.path()));
        }
    }
    out.sort_by_key(|(first_seq, _)| *first_seq);
    Ok(out)
}

/// Read every complete entry of a sealed segment. A sealed segment was synced
/// before it was sealed, so anything short of a clean parse is corruption and
/// fails closed (unlike the active file, whose torn tail is repaired on open).
async fn read_sealed_segment(path: &Path) -> Result<Vec<CanonicalWalEntry>> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading canonical WAL segment {}", path.display()))?;
    let scan = scan_wal_bytes(&bytes, path)?;
    if scan.valid_len != bytes.len() {
        bail!(
            "sealed canonical WAL segment {} is truncated or corrupt at byte {} of {}",
            path.display(),
            scan.valid_len,
            bytes.len()
        );
    }
    Ok(scan.entries)
}

/// Extract the `collection_id` a canonical operation targets, if any (the
/// shared WAL tags every entry; non-record ops carry none).
fn collection_id_of(op: &CanonicalOperation) -> Option<&str> {
    match op {
        CanonicalOperation::RecordUpsert { collection_id, .. }
        | CanonicalOperation::RecordDelete { collection_id, .. } => Some(collection_id.as_str()),
        _ => None,
    }
}

impl FramedTableWalAppender {
    /// Open or create a framed binary canonical WAL file.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("creating canonical WAL directory {}", parent.display())
            })?;
        }

        // Sealed segments are discovered whether or not this appender rotates:
        // a WAL rotated by an earlier run must read back in full.
        let mut sealed = Vec::new();
        let mut sealed_last = 0u64;
        let mut sealed_collection_max: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        for (_, segment_path) in discover_sealed_segments(&path).await? {
            let mut summary = SegmentSummary::new(segment_path.clone());
            for entry in read_sealed_segment(&segment_path).await? {
                summary.observe(&entry);
            }
            sealed_last = sealed_last.max(summary.last_seq);
            for (c, max) in &summary.collection_max {
                let cur = sealed_collection_max.entry(c.clone()).or_insert(0);
                *cur = (*cur).max(*max);
            }
            sealed.push(summary);
        }

        let mut recovery = recover_wal_state(&path).await?;
        recovery.last_sequence = recovery.last_sequence.max(sealed_last);
        for (c, max) in sealed_collection_max {
            let cur = recovery.collection_max.entry(c).or_insert(0);
            *cur = (*cur).max(max);
        }
        if recovery.valid_len < recovery.file_len {
            // An unacknowledged torn tail (appends ack only after `sync_data`).
            // Keep the dropped bytes beside the WAL before truncating, so a
            // truncation is never silent.
            // Best effort: a torn tail is most likely right when the disk is
            // full, and failing to keep forensics must not block recovery.
            let quarantine = torn_tail_path(&path, recovery.valid_len);
            if let Err(err) = write_synced(&quarantine, &recovery.tail).await {
                tracing::warn!(
                    quarantine = %quarantine.display(),
                    error = %err,
                    "canonical WAL: could not keep the torn tail for forensics; truncating anyway"
                );
            }
            TORN_TAIL_TRUNCATIONS.inc();
            tracing::warn!(
                wal = %path.display(),
                valid_len = recovery.valid_len,
                dropped_bytes = recovery.file_len - recovery.valid_len,
                quarantine = %quarantine.display(),
                "canonical WAL: truncating unacknowledged torn tail"
            );
            truncate_wal(&path, recovery.valid_len).await?;
        }

        let mut active = SegmentSummary::new(path.clone());
        for entry in &recovery.active_entries {
            active.observe(entry);
        }

        Ok(Self {
            path,
            next_sequence: AtomicU64::new(recovery.last_sequence),
            append_lock: Mutex::new(()),
            atomic_groups: AtomicBool::new(false),
            fence: WriterFence::global(),
            poisoned: AtomicBool::new(false),
            collection_max: parking_lot::Mutex::new(recovery.collection_max),
            segment_bytes: AtomicU64::new(0),
            sealed: parking_lot::RwLock::new(sealed),
            active: parking_lot::Mutex::new(active),
            rotations: AtomicU64::new(0),
        })
    }

    /// P2.1 (ADR-097 S2): seal the active file into a segment once it reaches
    /// `bytes`, so later range reads and truncation work segment by segment
    /// instead of over one ever-growing file. `0` disables rotation.
    pub fn enable_segment_rotation(&self, bytes: u64) {
        self.segment_bytes.store(bytes, Ordering::SeqCst);
    }

    /// Seal the active file: rename it to `{path}.{first_seq:020}.seg`. Called
    /// under the append lock after a synced append, so the sealed segment holds
    /// only acknowledged, complete frames. A failed rename leaves everything in
    /// the active file (nothing is lost; rotation is retried on the next append).
    async fn rotate_active(&self) {
        let summary = {
            let active = self.active.lock();
            if active.first_seq.is_none() {
                return;
            }
            active.clone()
        };
        let Some(first_seq) = summary.first_seq else {
            return;
        };
        let sealed_path = sealed_segment_path(&self.path, first_seq);
        // Odd generation: readers that overlap the rename re-read.
        self.rotations.fetch_add(1, Ordering::SeqCst);
        if let Err(err) = tokio::fs::rename(&self.path, &sealed_path).await {
            self.rotations.fetch_add(1, Ordering::SeqCst);
            tracing::warn!(
                wal = %self.path.display(),
                segment = %sealed_path.display(),
                error = %err,
                "canonical WAL: segment rotation failed; continuing on the active file"
            );
            return;
        }
        if let Some(dir) = self.path.parent()
            && let Ok(handle) = std::fs::File::open(dir)
        {
            let _ = handle.sync_all();
        }
        let mut sealed_summary = summary;
        sealed_summary.path = sealed_path;
        self.sealed.write().push(sealed_summary);
        *self.active.lock() = SegmentSummary::new(self.path.clone());
        // Even again: the segment list and active summary agree with the disk.
        self.rotations.fetch_add(1, Ordering::SeqCst);
        SEGMENT_ROTATIONS.inc();
    }

    /// Entries with `sequence_number > since` from every segment that can hold
    /// an entry for `collection_id`, oldest first. Segments whose summary rules
    /// the collection out are not read, so a change-feed read costs the
    /// segments it needs, not the whole log. Entries of other collections in a
    /// read segment are returned too; callers filter by collection.
    pub async fn read_entries_since(
        &self,
        collection_id: &str,
        since: u64,
    ) -> Result<Vec<CanonicalWalEntry>> {
        for _ in 0..MAX_ROTATION_RETRIES {
            let generation = self.rotations.load(Ordering::SeqCst);
            if generation % 2 == 1 {
                // A rotation is mid-rename; it finishes in one rename + fsync.
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                continue;
            }
            let out = self.read_entries_since_once(collection_id, since).await?;
            if self.rotations.load(Ordering::SeqCst) == generation {
                return Ok(out);
            }
        }
        bail!("canonical WAL kept rotating during a range read; retry later")
    }

    async fn read_entries_since_once(
        &self,
        collection_id: &str,
        since: u64,
    ) -> Result<Vec<CanonicalWalEntry>> {
        let sealed: Vec<PathBuf> = self
            .sealed
            .read()
            .iter()
            .filter(|s| s.may_contain(collection_id, since))
            .map(|s| s.path.clone())
            .collect();
        let read_active = self.active.lock().may_contain(collection_id, since);
        let mut out = Vec::new();
        for segment in sealed {
            out.extend(
                read_sealed_segment(&segment)
                    .await?
                    .into_iter()
                    .filter(|e| e.sequence_number > since),
            );
        }
        if read_active {
            out.extend(
                read_active_file(&self.path)
                    .await?
                    .into_iter()
                    .filter(|e| e.sequence_number > since),
            );
        }
        Ok(out)
    }

    /// Make every multi-operation `append_operations` a single all-or-nothing
    /// group frame (ADR-095; `[oltp] atomic_commit`). Set only on the shared
    /// pgwire canonical WAL.
    pub fn enable_atomic_groups(&self) {
        self.atomic_groups.store(true, Ordering::SeqCst);
    }

    /// Replace the writer fence (tests inject their own; production uses the
    /// process-wide one).
    pub fn with_writer_fence(mut self, fence: Arc<WriterFence>) -> Self {
        self.fence = fence;
        self
    }

    /// Assign sequences, encode, write, `sync_data`, then publish the new
    /// sequence floor and per-collection high-water marks. `group` writes one
    /// version-2 frame; otherwise one version-1 frame per entry. Fault hooks run
    /// only on the atomic path.
    async fn append_entries(
        &self,
        operations: Vec<(CanonicalOperation, Option<String>)>,
        group: bool,
        fault_hooks: bool,
    ) -> Result<Vec<CanonicalWalEntry>> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self.append_lock.lock().await;
        // Checked under the append lock so no write races past a tripped fence.
        self.fence.ensure_owner()?;
        if self.poisoned.load(Ordering::SeqCst) {
            bail!(WalAppenderPoisoned);
        }
        // Test seams only exist in debug builds: a release server can never be
        // aborted through an environment variable.
        let fault = if fault_hooks && cfg!(debug_assertions) {
            armed_commit_fault()
        } else {
            None
        };
        if fault == Some(AppendFault::FailBefore) {
            bail!("injected fault: {COMMIT_FAULT_ENV}=fail_before_append");
        }

        let base_sequence = self.next_sequence.load(Ordering::SeqCst);
        let entries: Vec<CanonicalWalEntry> = operations
            .into_iter()
            .enumerate()
            .map(|(offset, (operation, tenant_id))| {
                CanonicalWalEntry::new(base_sequence + offset as u64 + 1, operation, tenant_id)
            })
            .collect();

        let mut frames = Vec::new();
        if group {
            encode_group_frame(&entries, &mut frames)
                .context("encoding canonical WAL group frame")?;
        } else {
            for entry in &entries {
                encode_frame(entry, &mut frames).context("encoding canonical WAL frame")?;
            }
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .with_context(|| format!("opening canonical WAL {}", self.path.display()))?;

        let write_len = if matches!(fault, Some(AppendFault::Torn | AppendFault::FailMidWrite)) {
            frames.len() / 2
        } else {
            frames.len()
        };
        let pre_len = file
            .metadata()
            .await
            .with_context(|| format!("sizing canonical WAL {}", self.path.display()))?
            .len();
        let written = async {
            file.write_all(&frames[..write_len])
                .await
                .with_context(|| format!("writing canonical WAL {}", self.path.display()))?;
            file.flush()
                .await
                .with_context(|| format!("flushing canonical WAL {}", self.path.display()))?;
            if fault == Some(AppendFault::FailMidWrite) {
                bail!("injected fault: write failed after a partial frame (ENOSPC)");
            }
            Ok::<(), anyhow::Error>(())
        }
        .await;
        // (error, failed_at_sync): which stage failed decides whether a
        // rollback can make the outcome determinate.
        let failure = match written {
            Err(err) => Some((err, false)),
            Ok(()) if fault == Some(AppendFault::FailSync) => {
                Some((anyhow!("injected fault: sync_data failed"), true))
            }
            Ok(()) => match file.sync_data().await {
                Ok(()) => None,
                Err(err) => Some((
                    anyhow::Error::new(err)
                        .context(format!("syncing canonical WAL {}", self.path.display())),
                    true,
                )),
            },
        };
        if let Some((err, failed_at_sync)) = failure {
            // Bytes may have reached the file. Roll the file back to its
            // pre-append length (still under the append lock) so a partial frame
            // can never sit in front of later, acknowledged frames. A failed
            // `sync_data` is treated as indeterminate even after a rollback:
            // after an fsync error the kernel's page state is not trustworthy.
            let rolled_back = async {
                file.set_len(pre_len).await?;
                file.sync_data().await
            }
            .await;
            if rolled_back.is_err() || failed_at_sync {
                self.poisoned.store(true, Ordering::SeqCst);
                tracing::error!(
                    wal = %self.path.display(),
                    error = %format!("{err:#}"),
                    rollback = ?rolled_back.err(),
                    "canonical WAL append failed indeterminately; appender poisoned"
                );
                bail!(WalAppendIndeterminate(format!("{err:#}")));
            }
            return Err(
                err.context("canonical WAL append failed and was rolled back; nothing was written")
            );
        }
        if matches!(fault, Some(AppendFault::Torn | AppendFault::AbortAfter)) {
            tracing::error!(fault = ?fault, "{COMMIT_FAULT_ENV}: aborting the process (test crash point)");
            std::process::abort();
        }

        self.next_sequence
            .store(base_sequence + entries.len() as u64, Ordering::SeqCst);

        // TD-OLAP-17: bump the per-collection high-water so `read_changes_since`
        // can skip the O(WAL) scan for collections that haven't advanced past a
        // reader's `since_lsn`. Still under `append_lock` (held above) and after
        // `sync_data`, so a reader that observes this high-water is guaranteed the
        // entry is durable.
        {
            let mut maxes = self.collection_max.lock();
            for entry in &entries {
                if let Some(c) = collection_id_of(&entry.operation) {
                    let cur = maxes.entry(c.to_string()).or_insert(0);
                    if entry.sequence_number > *cur {
                        *cur = entry.sequence_number;
                    }
                }
            }
        }
        {
            let mut active = self.active.lock();
            for entry in &entries {
                active.observe(entry);
            }
        }

        // P2.1: seal the active file once it is large enough. Still under the
        // append lock, after `sync_data`, so the next append opens a fresh file.
        let segment_bytes = self.segment_bytes.load(Ordering::SeqCst);
        if segment_bytes > 0 && pre_len + frames.len() as u64 >= segment_bytes {
            drop(file);
            self.rotate_active().await;
        }

        Ok(entries)
    }

    /// Path backing this WAL appender.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Raise the next-assigned sequence floor so future appends never reuse a
    /// sequence at or below `floor` (monotonic — a lower floor is ignored).
    ///
    /// `next_sequence` is otherwise derived from the on-disk WAL's highest frame.
    /// That is correct only while the WAL still carries the full history; once a
    /// snapshot watermark advances *past* the on-disk frames — the WAL was
    /// compacted to empty (Phase 3), or this pod adopted a *peer's* snapshot
    /// whose watermark lives in a different local sequence space (Phase 6b) — the
    /// file-derived floor sits below the snapshot's `applied_seq`. A fresh append
    /// would then land at a sequence the in-RAM authority has already applied and
    /// be idempotently dropped on fold. Seeding the floor from the snapshot
    /// watermark at open closes that gap.
    pub fn advance_sequence_floor(&self, floor: u64) {
        self.next_sequence.fetch_max(floor, Ordering::SeqCst);
    }

    /// Read all complete entries currently present in the WAL: every sealed
    /// segment, oldest first, then the active file.
    pub async fn read_entries(&self) -> Result<Vec<CanonicalWalEntry>> {
        read_entries_from_path(&self.path).await
    }

    /// Read all complete entries from a framed WAL path.
    pub async fn read_entries_from_path(path: impl AsRef<Path>) -> Result<Vec<CanonicalWalEntry>> {
        read_entries_from_path(path.as_ref()).await
    }
}

/// Root bridge from [`proximadb_storage_ports::CanonicalWalReaderPort`] to
/// [`FramedTableWalAppender::read_entries_from_path`], so graph engines can read
/// the canonical WAL for TD-066 checkpoint-LSN correlation without naming the
/// concrete appender.
pub(crate) struct CanonicalWalReaderBridge;

#[async_trait]
impl proximadb_storage_ports::CanonicalWalReaderPort for CanonicalWalReaderBridge {
    async fn read_entries(&self, canonical_wal_path: &Path) -> Result<Vec<CanonicalWalEntry>> {
        FramedTableWalAppender::read_entries_from_path(canonical_wal_path).await
    }
}

#[async_trait]
impl TableWalAppender for FramedTableWalAppender {
    async fn read_all_entries(&self) -> Result<Vec<CanonicalWalEntry>> {
        self.read_entries().await
    }

    async fn read_entries_since(
        &self,
        collection_id: &str,
        since: u64,
    ) -> Result<Vec<CanonicalWalEntry>> {
        FramedTableWalAppender::read_entries_since(self, collection_id, since).await
    }

    async fn append_operations(
        &self,
        operations: Vec<CanonicalOperation>,
        tenant_id: Option<String>,
    ) -> Result<Vec<CanonicalWalEntry>> {
        let group = operations.len() > 1 && self.atomic_groups.load(Ordering::SeqCst);
        let operations = operations
            .into_iter()
            .map(|operation| (operation, tenant_id.clone()))
            .collect();
        self.append_entries(operations, group, false).await
    }

    async fn append_operations_atomic(
        &self,
        operations: Vec<(CanonicalOperation, Option<String>)>,
    ) -> Result<Vec<CanonicalWalEntry>> {
        // A single operation is already atomic as a version-1 frame; using it
        // keeps one-row commits readable by binaries that predate groups.
        let group = operations.len() > 1;
        self.append_entries(operations, group, true).await
    }

    fn collection_max_sequence_for(&self, collection_id: &str) -> Option<u64> {
        self.collection_max.lock().get(collection_id).copied()
    }

    fn tracks_collection_high_water(&self) -> bool {
        true
    }
}

/// In-memory `TableWalAppender` used when no on-disk canonical WAL is
/// configured. Operations are stored in a `Mutex<Vec<CanonicalWalEntry>>`
/// keyed by a single monotonic counter so consumers that need to replay or
/// inspect appended entries can do so.
///
/// Intended for test paths and the `opt_config = None` boot path where the
/// shared canonical WAL is not opened. Not durable — entries are lost when
/// the process exits.
pub struct MemoryTableWalAppender {
    next_sequence: AtomicU64,
    entries: Mutex<Vec<CanonicalWalEntry>>,
    /// TD-OLAP-17 per-collection high-water (see `FramedTableWalAppender`).
    collection_max: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
}

impl MemoryTableWalAppender {
    /// Build an empty in-memory appender.
    pub fn new() -> Self {
        Self {
            next_sequence: AtomicU64::new(0),
            entries: Mutex::new(Vec::new()),
            collection_max: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Snapshot the currently appended entries. Cheap clone; intended for
    /// tests + the rank-profile store's in-memory recovery path.
    pub async fn entries(&self) -> Vec<CanonicalWalEntry> {
        self.entries.lock().await.clone()
    }
}

impl Default for MemoryTableWalAppender {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TableWalAppender for MemoryTableWalAppender {
    async fn append_operations(
        &self,
        operations: Vec<CanonicalOperation>,
        tenant_id: Option<String>,
    ) -> Result<Vec<CanonicalWalEntry>> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        let base = self.next_sequence.load(Ordering::SeqCst);
        let entries: Vec<CanonicalWalEntry> = operations
            .into_iter()
            .enumerate()
            .map(|(offset, op)| {
                CanonicalWalEntry::new(base + offset as u64 + 1, op, tenant_id.clone())
            })
            .collect();
        self.next_sequence
            .store(base + entries.len() as u64, Ordering::SeqCst);
        self.entries.lock().await.extend(entries.clone());
        // TD-OLAP-17: bump the per-collection high-water (see FramedTableWalAppender).
        {
            let mut maxes = self.collection_max.lock();
            for entry in &entries {
                if let Some(c) = collection_id_of(&entry.operation) {
                    let cur = maxes.entry(c.to_string()).or_insert(0);
                    if entry.sequence_number > *cur {
                        *cur = entry.sequence_number;
                    }
                }
            }
        }
        Ok(entries)
    }

    /// All-or-nothing by construction (one push under the entries lock); the
    /// lock is taken BEFORE allocating sequences so concurrent appends cannot
    /// interleave inside a group.
    async fn append_operations_atomic(
        &self,
        operations: Vec<(CanonicalOperation, Option<String>)>,
    ) -> Result<Vec<CanonicalWalEntry>> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        let mut stored = self.entries.lock().await;
        let base = self.next_sequence.load(Ordering::SeqCst);
        let entries: Vec<CanonicalWalEntry> = operations
            .into_iter()
            .enumerate()
            .map(|(offset, (op, tenant))| {
                CanonicalWalEntry::new(base + offset as u64 + 1, op, tenant)
            })
            .collect();
        self.next_sequence
            .store(base + entries.len() as u64, Ordering::SeqCst);
        stored.extend(entries.clone());
        drop(stored);
        {
            let mut maxes = self.collection_max.lock();
            for entry in &entries {
                if let Some(c) = collection_id_of(&entry.operation) {
                    let cur = maxes.entry(c.to_string()).or_insert(0);
                    if entry.sequence_number > *cur {
                        *cur = entry.sequence_number;
                    }
                }
            }
        }
        Ok(entries)
    }

    /// TD-OLAP-17: surface the in-memory entries through the trait so
    /// `read_changes_since` (the CDC feed + OLAP delta merge) works on the
    /// `opt_config = None` boot path that uses this appender. Without this
    /// override the trait default returned empty, so the change-feed was silently
    /// empty for in-memory WALs.
    async fn read_all_entries(&self) -> Result<Vec<CanonicalWalEntry>> {
        Ok(self.entries().await)
    }

    fn collection_max_sequence_for(&self, collection_id: &str) -> Option<u64> {
        self.collection_max.lock().get(collection_id).copied()
    }

    fn tracks_collection_high_water(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct WalRecoveryState {
    last_sequence: u64,
    /// TD-OLAP-17 recovery rebuild: per-collection high-water seeded from the
    /// recovered entries, so read-only-after-restart tables fast-path immediately.
    collection_max: std::collections::HashMap<String, u64>,
    valid_len: u64,
    file_len: u64,
    /// Bytes after `valid_len` (the torn tail), quarantined before truncation.
    tail: Vec<u8>,
    /// Entries recovered from the active file (seeds its segment summary).
    active_entries: Vec<CanonicalWalEntry>,
}

async fn recover_wal_state(path: &Path) -> Result<WalRecoveryState> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(WalRecoveryState {
                last_sequence: 0,
                collection_max: std::collections::HashMap::new(),
                valid_len: 0,
                file_len: 0,
                tail: Vec::new(),
                active_entries: Vec::new(),
            });
        }
        Err(err) => {
            return Err(err).with_context(|| format!("opening canonical WAL {}", path.display()));
        }
    };

    let scan = scan_wal_bytes(&bytes, path)?;
    let last_sequence = scan
        .entries
        .iter()
        .map(|entry| entry.sequence_number)
        .max()
        .unwrap_or(0);
    // TD-OLAP-17 recovery rebuild: seed the per-collection high-water from the
    // recovered entries so a read-mostly table fast-paths immediately after a
    // restart, without waiting for the next write to repopulate it. No extra I/O
    // — `scan_wal_bytes` already parsed every entry for `last_sequence`.
    let mut collection_max: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for entry in &scan.entries {
        if let Some(c) = collection_id_of(&entry.operation) {
            let cur = collection_max.entry(c.to_string()).or_insert(0);
            if entry.sequence_number > *cur {
                *cur = entry.sequence_number;
            }
        }
    }

    Ok(WalRecoveryState {
        last_sequence,
        collection_max,
        valid_len: scan.valid_len as u64,
        file_len: bytes.len() as u64,
        tail: bytes[scan.valid_len..].to_vec(),
        active_entries: scan.entries,
    })
}

/// Every complete entry of the WAL at `path`: sealed segments oldest first, then
/// the active file (P2.1). A WAL that never rotated is just its active file.
async fn read_entries_from_path(path: &Path) -> Result<Vec<CanonicalWalEntry>> {
    // A rotation between listing the segments and reading the active file would
    // move entries out of the file we are about to read; the segment list then
    // differs on the second look, and we re-read.
    for _ in 0..MAX_ROTATION_RETRIES {
        let segments = discover_sealed_segments(path).await?;
        let mut out = Vec::new();
        for (_, segment) in &segments {
            out.extend(read_sealed_segment(segment).await?);
        }
        out.extend(read_active_file(path).await?);
        if discover_sealed_segments(path).await? == segments {
            return Ok(out);
        }
    }
    bail!(
        "canonical WAL {} kept rotating during a read; retry later",
        path.display()
    )
}

/// Complete entries of the active WAL file (a missing file reads as empty).
async fn read_active_file(path: &Path) -> Result<Vec<CanonicalWalEntry>> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(err).with_context(|| format!("opening canonical WAL {}", path.display()));
        }
    };
    Ok(scan_wal_bytes(&bytes, path)?.entries)
}

/// Write `bytes` to a new file and fsync it.
async fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = tokio::fs::File::create(path).await?;
    file.write_all(bytes).await?;
    file.sync_all().await?;
    Ok(())
}

async fn truncate_wal(path: &Path, valid_len: u64) -> Result<()> {
    let file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .await
        .with_context(|| format!("opening canonical WAL {} for truncation", path.display()))?;
    file.set_len(valid_len)
        .await
        .with_context(|| format!("truncating canonical WAL {}", path.display()))
}

#[derive(Debug)]
struct WalScan {
    entries: Vec<CanonicalWalEntry>,
    valid_len: usize,
}

fn scan_wal_bytes(bytes: &[u8], path: &Path) -> Result<WalScan> {
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset < bytes.len() {
        let decoded = match decode_frame(&bytes[offset..]) {
            Ok(decoded) => decoded,
            // A zero-extended tail (the file grew but the data blocks never
            // landed) is a torn, unacknowledged write — not corruption. Every
            // other decode failure stays a hard error: a non-zero bad frame may
            // be a committed frame from a newer binary or real corruption, and
            // must never be silently dropped (ADR-095 review B2).
            Err(_) if bytes[offset..].iter().all(|byte| *byte == 0) => break,
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "decoding canonical WAL frame at byte {} in {}",
                        offset,
                        path.display()
                    )
                });
            }
        };
        match decoded {
            Some((frame_entries, consumed)) => {
                entries.extend(frame_entries);
                offset += consumed;
            }
            None => break,
        }
    }

    Ok(WalScan {
        entries,
        valid_len: offset,
    })
}

fn encode_frame(entry: &CanonicalWalEntry, out: &mut Vec<u8>) -> Result<()> {
    let payload = rmp_serde::to_vec_named(entry).context("serializing canonical WAL payload")?;
    let payload_len: u32 = payload
        .len()
        .try_into()
        .context("canonical WAL payload exceeds u32 frame size")?;
    let payload_crc = checksum(&payload);

    out.extend_from_slice(FRAME_MAGIC);
    out.extend_from_slice(&FRAME_VERSION.to_le_bytes());
    out.push(CODEC_MSGPACK_NAMED);
    out.push(0);
    out.extend_from_slice(&entry.sequence_number.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&payload_crc.to_le_bytes());
    out.extend_from_slice(&payload);

    Ok(())
}

/// Encode `entries` as ONE version-2 frame: one header, one CRC over the whole
/// group. A crash anywhere inside it leaves an incomplete frame, which recovery
/// drops whole — never a prefix of the group.
fn encode_group_frame(entries: &[CanonicalWalEntry], out: &mut Vec<u8>) -> Result<()> {
    let first = entries
        .first()
        .ok_or_else(|| anyhow!("canonical WAL group frame needs at least one entry"))?;
    let payload =
        rmp_serde::to_vec_named(entries).context("serializing canonical WAL group payload")?;
    let payload_len: u32 = payload
        .len()
        .try_into()
        .context("canonical WAL group payload exceeds u32 frame size")?;
    let payload_crc = checksum(&payload);

    out.extend_from_slice(FRAME_MAGIC);
    out.extend_from_slice(&FRAME_VERSION_GROUP.to_le_bytes());
    out.push(CODEC_MSGPACK_NAMED);
    out.push(0);
    out.extend_from_slice(&first.sequence_number.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&payload_crc.to_le_bytes());
    out.extend_from_slice(&payload);

    Ok(())
}

/// Atomically rewrite the canonical WAL at `path` so it contains exactly
/// `entries`, **preserving their original `sequence_number`s**. Writes a temp
/// file, fsyncs it, atomically renames it over `path`, then best-effort fsyncs
/// the parent directory so the rename is durable.
///
/// Used by the system catalog to compact its WAL after a durable snapshot: the
/// snapshot covers every mutation up to its watermark LSN, so the WAL is
/// rewritten to keep only the entries after it. Because the sequence numbers are
/// preserved verbatim (not reassigned), the live appender's monotonic counter
/// and the snapshot watermark stay consistent across the rewrite.
pub async fn rewrite_canonical_wal(path: &Path, entries: &[CanonicalWalEntry]) -> Result<()> {
    // A rotated WAL keeps history in sealed segments this rewrite would not
    // touch, so a rewrite could resurrect or duplicate entries. Rotation and
    // rewrite-compaction are separate retention mechanisms; refuse the mix.
    if !discover_sealed_segments(path).await?.is_empty() {
        bail!(
            "refusing to rewrite canonical WAL {}: it has sealed segments (segment \
             rotation and rewrite-compaction must not be combined)",
            path.display()
        );
    }
    let mut frames = Vec::new();
    for entry in entries {
        encode_frame(entry, &mut frames)?;
    }
    let tmp = path.with_extension("wal-compact-tmp");
    {
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .await
            .with_context(|| format!("opening compaction temp {}", tmp.display()))?;
        file.write_all(&frames)
            .await
            .with_context(|| format!("writing compaction temp {}", tmp.display()))?;
        file.flush().await?;
        file.sync_data()
            .await
            .with_context(|| format!("fsync compaction temp {}", tmp.display()))?;
    }
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("atomically replacing canonical WAL {}", path.display()))?;
    if let Some(dir) = path.parent()
        && let Ok(handle) = std::fs::File::open(dir)
    {
        let _ = handle.sync_all();
    }
    Ok(())
}

/// Decode one frame. Version 1 yields one entry; version 2 yields its whole
/// group. `Ok(None)` = an incomplete trailing frame (torn write).
fn decode_frame(bytes: &[u8]) -> Result<Option<(Vec<CanonicalWalEntry>, usize)>> {
    if bytes.len() < FRAME_HEADER_LEN {
        return Ok(None);
    }

    let mut offset = 0;

    let magic = &bytes[offset..offset + FRAME_MAGIC.len()];
    offset += FRAME_MAGIC.len();
    if magic != FRAME_MAGIC {
        bail!("invalid canonical WAL frame magic");
    }

    let version = read_u16(bytes, &mut offset)?;
    if version != FRAME_VERSION && version != FRAME_VERSION_GROUP {
        bail!("unsupported canonical WAL frame version {version}");
    }

    let codec = bytes[offset];
    offset += 1;
    if codec != CODEC_MSGPACK_NAMED {
        bail!("unsupported canonical WAL codec {codec}");
    }

    let _flags = bytes[offset];
    offset += 1;

    let header_sequence = read_u64(bytes, &mut offset)?;
    let payload_len = read_u32(bytes, &mut offset)? as usize;
    let expected_crc = read_u32(bytes, &mut offset)?;

    let frame_len = FRAME_HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| anyhow!("canonical WAL frame length overflow"))?;
    if bytes.len() < frame_len {
        return Ok(None);
    }

    let payload = &bytes[offset..frame_len];
    let actual_crc = checksum(payload);
    if actual_crc != expected_crc {
        bail!("canonical WAL payload checksum mismatch");
    }

    if version == FRAME_VERSION_GROUP {
        let entries: Vec<CanonicalWalEntry> =
            rmp_serde::from_slice(payload).context("deserializing canonical WAL group payload")?;
        if entries.is_empty() {
            bail!("canonical WAL group frame at sequence {header_sequence} is empty");
        }
        for (index, entry) in entries.iter().enumerate() {
            let expected = header_sequence + index as u64;
            if entry.sequence_number != expected {
                bail!(
                    "canonical WAL group frame sequences are not contiguous: expected {}, found {}",
                    expected,
                    entry.sequence_number
                );
            }
        }
        return Ok(Some((entries, frame_len)));
    }

    let entry: CanonicalWalEntry =
        rmp_serde::from_slice(payload).context("deserializing canonical WAL payload")?;
    if entry.sequence_number != header_sequence {
        bail!(
            "canonical WAL header sequence {} does not match payload sequence {}",
            header_sequence,
            entry.sequence_number
        );
    }

    Ok(Some((vec![entry], frame_len)))
}

fn checksum(payload: &[u8]) -> u32 {
    let mut hasher = Crc32Hasher::new();
    hasher.update(payload);
    hasher.finalize()
}

// Each `read_uN` helper takes an `&[u8]` and an offset, slices N bytes,
// and converts to a fixed-size array via `try_into`. The `.get(...)?`
// returns `None` when the slice would overrun, so by the time we hit
// `try_into`, the source slice is exactly N bytes long and the conversion
// is infallible. The `expect` is a documented "slice length matches array
// size" invariant — keep it inline so future readers see why panic is
// unreachable.
#[allow(clippy::expect_used)]
fn read_u16(bytes: &[u8], offset: &mut usize) -> Result<u16> {
    let end = *offset + 2;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("canonical WAL frame missing u16"))?;
    *offset = end;
    Ok(u16::from_le_bytes(
        value.try_into().expect("u16 slice length"),
    ))
}

#[allow(clippy::expect_used)]
fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32> {
    let end = *offset + 4;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("canonical WAL frame missing u32"))?;
    *offset = end;
    Ok(u32::from_le_bytes(
        value.try_into().expect("u32 slice length"),
    ))
}

#[allow(clippy::expect_used)]
fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let end = *offset + 8;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("canonical WAL frame missing u64"))?;
    *offset = end;
    Ok(u64::from_le_bytes(
        value.try_into().expect("u64 slice length"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaRecord;
    use proximadb_storage_common::ProjectionDirective;
    use std::sync::Arc;

    fn upsert_operation(collection_id: &str, oid: &str) -> CanonicalOperation {
        CanonicalOperation::RecordUpsert {
            collection_id: collection_id.to_string(),
            record: Box::new(ProximaRecord {
                oid: oid.to_string(),
                ..ProximaRecord::default()
            }),
            projections: vec![ProjectionDirective::ColumnarVariation {
                collection_id: collection_id.to_string(),
                fields: vec!["id".to_string()],
            }],
        }
    }

    #[tokio::test]
    async fn framed_table_wal_appender_persists_and_recovers_sequence() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("tables").join("catalog.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        let first_entries = appender
            .append_operations(
                vec![
                    upsert_operation("orders", "order-1"),
                    upsert_operation("orders", "order-2"),
                ],
                Some("tenant-a".to_string()),
            )
            .await?;

        assert_eq!(
            first_entries
                .iter()
                .map(|entry| entry.sequence_number)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let next_entries = reopened
            .append_operations(vec![upsert_operation("orders", "order-3")], None)
            .await?;
        assert_eq!(next_entries[0].sequence_number, 3);

        let entries = reopened.read_entries().await?;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(entries[2].sequence_number, 3);

        let entries_from_path = FramedTableWalAppender::read_entries_from_path(&wal_path).await?;
        assert_eq!(entries_from_path.len(), 3);

        let bytes = tokio::fs::read(&wal_path).await?;
        assert!(String::from_utf8(bytes.clone()).is_err());

        let mut offset = 0;
        let mut decoded_entries = Vec::new();
        while let Some((entries, consumed)) = decode_frame(&bytes[offset..])? {
            decoded_entries.extend(entries);
            offset += consumed;
        }
        assert_eq!(decoded_entries.len(), 3);

        Ok(())
    }

    /// Frame versions of every frame in a WAL file, in order.
    fn frame_versions(bytes: &[u8]) -> Result<Vec<u16>> {
        let mut offset = 0;
        let mut versions = Vec::new();
        while let Some((_, consumed)) = decode_frame(&bytes[offset..])? {
            let mut cursor = offset + FRAME_MAGIC.len();
            versions.push(read_u16(bytes, &mut cursor)?);
            offset += consumed;
        }
        Ok(versions)
    }

    fn tenant_op(
        collection_id: &str,
        oid: &str,
        tenant: &str,
    ) -> (CanonicalOperation, Option<String>) {
        (
            upsert_operation(collection_id, oid),
            Some(tenant.to_string()),
        )
    }

    // ADR-095 D3 / TD-PGOLTP-1: a transaction is ONE frame. Every entry keeps its
    // own tenant and sequence, and the file holds a single version-2 frame.
    #[tokio::test]
    async fn atomic_group_round_trips_as_one_frame() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("group.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        let entries = appender
            .append_operations_atomic(vec![
                tenant_op("accounts", "a1", "t1"),
                tenant_op("accounts", "a2", "t1"),
                tenant_op("history", "h1", "t2"),
            ])
            .await?;
        assert_eq!(
            entries
                .iter()
                .map(|e| e.sequence_number)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        let bytes = tokio::fs::read(&wal_path).await?;
        assert_eq!(frame_versions(&bytes)?, vec![FRAME_VERSION_GROUP]);

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let recovered = reopened.read_entries().await?;
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[2].tenant_id.as_deref(), Some("t2"));
        assert_eq!(reopened.collection_max_sequence_for("accounts"), Some(2));
        assert_eq!(reopened.collection_max_sequence_for("history"), Some(3));
        Ok(())
    }

    // Mixed-read safety (mandate #8): version-1 and version-2 frames coexist and
    // replay in sequence order.
    #[tokio::test]
    async fn mixed_single_and_group_frames_replay_in_order() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("mixed.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("t", "r1")], None)
            .await?;
        appender
            .append_operations_atomic(vec![tenant_op("t", "r2", "x"), tenant_op("t", "r3", "x")])
            .await?;
        appender
            .append_operations(vec![upsert_operation("t", "r4")], None)
            .await?;

        let bytes = tokio::fs::read(&wal_path).await?;
        assert_eq!(
            frame_versions(&bytes)?,
            vec![FRAME_VERSION, FRAME_VERSION_GROUP, FRAME_VERSION]
        );
        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        assert_eq!(
            reopened
                .read_entries()
                .await?
                .iter()
                .map(|e| e.sequence_number)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        Ok(())
    }

    // A crash mid-write of a group must lose the WHOLE group, never a prefix.
    #[tokio::test]
    async fn torn_group_frame_drops_the_whole_group() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("torn.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("t", "committed")], None)
            .await?;
        let committed_len = tokio::fs::metadata(&wal_path).await?.len();

        let group: Vec<CanonicalWalEntry> = (0..3)
            .map(|i| CanonicalWalEntry::new(2 + i, upsert_operation("t", &format!("g{i}")), None))
            .collect();
        let mut frame = Vec::new();
        encode_group_frame(&group, &mut frame)?;
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .await?;
        file.write_all(&frame[..frame.len() / 2]).await?;
        file.flush().await?;
        drop(file);

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let recovered = reopened.read_entries().await?;
        assert_eq!(recovered.len(), 1, "no member of a torn group may survive");
        assert_eq!(tokio::fs::metadata(&wal_path).await?.len(), committed_len);
        let next = reopened
            .append_operations(vec![upsert_operation("t", "after")], None)
            .await?;
        assert_eq!(next[0].sequence_number, 2);
        Ok(())
    }

    // A zero-extended tail (file size grew, data blocks never landed) is a torn
    // write, not corruption: boot must succeed, and the dropped bytes are kept.
    #[tokio::test]
    async fn zero_extended_tail_is_quarantined_and_truncated() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("zeros.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("t", "r1")], None)
            .await?;
        let valid_len = tokio::fs::metadata(&wal_path).await?.len();
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .await?;
        file.write_all(&[0u8; 4096]).await?;
        file.flush().await?;
        drop(file);

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        assert_eq!(reopened.read_entries().await?.len(), 1);
        assert_eq!(tokio::fs::metadata(&wal_path).await?.len(), valid_len);
        let prefix = format!("zeros.wal.torn-{valid_len}-");
        let mut quarantined = Vec::new();
        for entry in std::fs::read_dir(dir.path())? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                quarantined.push(std::fs::read(entry.path())?);
            }
        }
        assert_eq!(quarantined.len(), 1);
        assert_eq!(quarantined[0].len(), 4096);
        Ok(())
    }

    // Corruption that is NOT a torn tail stays a hard error: a bad frame followed
    // by more data must never be silently dropped.
    #[tokio::test]
    async fn corrupt_frame_before_valid_data_is_still_an_error() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("midfile.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("t", "r1")], None)
            .await?;
        let first_len = tokio::fs::metadata(&wal_path).await?.len() as usize;
        appender
            .append_operations(vec![upsert_operation("t", "r2")], None)
            .await?;

        let mut bytes = tokio::fs::read(&wal_path).await?;
        bytes[first_len - 1] ^= 0x01;
        tokio::fs::write(&wal_path, bytes).await?;

        assert!(FramedTableWalAppender::open(&wal_path).await.is_err());
        Ok(())
    }

    #[test]
    fn group_frame_with_non_contiguous_sequences_is_rejected() -> Result<()> {
        let group = vec![
            CanonicalWalEntry::new(5, upsert_operation("t", "a"), None),
            CanonicalWalEntry::new(7, upsert_operation("t", "b"), None),
        ];
        let mut frame = Vec::new();
        encode_group_frame(&group, &mut frame)?;
        assert!(decode_frame(&frame).is_err());
        Ok(())
    }

    // ADR-095 D2 pre-ack self-fence: once a newer writer epoch has been observed,
    // this process must not make another write durable.
    #[tokio::test]
    async fn fenced_appender_refuses_every_append() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("fenced.wal");
        let fence = Arc::new(WriterFence::default());

        let appender = FramedTableWalAppender::open(&wal_path)
            .await?
            .with_writer_fence(fence.clone());
        appender
            .append_operations(vec![upsert_operation("t", "before")], None)
            .await?;
        fence.mark_lost();

        let plain = appender
            .append_operations(vec![upsert_operation("t", "after")], None)
            .await;
        let atomic = appender
            .append_operations_atomic(vec![tenant_op("t", "after2", "x")])
            .await;
        for result in [plain.map(|_| ()), atomic.map(|_| ())] {
            let err = result.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(err.contains("writer fenced"), "got: {err}");
        }
        assert_eq!(appender.read_entries().await?.len(), 1);
        Ok(())
    }

    // M5: the group flag is per appender. When it is on, a plain multi-op append
    // (one multi-row statement) is also a single frame; when off, nothing changes.
    #[tokio::test]
    async fn multi_op_append_uses_one_group_frame_only_when_enabled() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let off_path = dir.path().join("off.wal");
        let on_path = dir.path().join("on.wal");
        let ops = || vec![upsert_operation("t", "a"), upsert_operation("t", "b")];

        FramedTableWalAppender::open(&off_path)
            .await?
            .append_operations(ops(), None)
            .await?;
        let on = FramedTableWalAppender::open(&on_path).await?;
        on.enable_atomic_groups();
        on.append_operations(ops(), Some("t1".to_string())).await?;

        assert_eq!(
            frame_versions(&tokio::fs::read(&off_path).await?)?,
            vec![FRAME_VERSION, FRAME_VERSION]
        );
        assert_eq!(
            frame_versions(&tokio::fs::read(&on_path).await?)?,
            vec![FRAME_VERSION_GROUP]
        );
        assert_eq!(on.read_entries().await?[1].tenant_id.as_deref(), Some("t1"));
        Ok(())
    }

    // Review finding 1: a write that fails part-way must not leave a partial
    // frame in front of later, acknowledged frames. ENOSPC mid-write → the
    // file is rolled back and the next append (and a reopen) is clean.
    #[tokio::test]
    async fn failed_mid_write_is_rolled_back_and_later_appends_survive() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("enospc.wal");
        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("t", "before")], None)
            .await?;
        let before_len = tokio::fs::metadata(&wal_path).await?.len();

        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var(COMMIT_FAULT_ENV, "fail_mid_write") };
        let failed = appender
            .append_operations_atomic(vec![tenant_op("t", "x", "a"), tenant_op("t", "y", "a")])
            .await;
        unsafe { std::env::remove_var(COMMIT_FAULT_ENV) };
        let err = failed
            .err()
            .ok_or_else(|| anyhow!("the injected write must fail"))?;
        assert!(
            err.downcast_ref::<WalAppendIndeterminate>().is_none(),
            "rolled back = determinate"
        );
        assert_eq!(tokio::fs::metadata(&wal_path).await?.len(), before_len);

        let next = appender
            .append_operations(vec![upsert_operation("t", "after")], None)
            .await?;
        assert_eq!(
            next[0].sequence_number, 2,
            "sequence not consumed by the failed write"
        );
        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        assert_eq!(reopened.read_entries().await?.len(), 2);
        Ok(())
    }

    // fsyncgate: after a failed sync_data the outcome is unknowable → typed
    // indeterminate error, and the appender refuses everything afterwards.
    #[tokio::test]
    async fn failed_sync_poisons_the_appender() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("fsync.wal");
        let appender = FramedTableWalAppender::open(&wal_path).await?;

        unsafe { std::env::set_var(COMMIT_FAULT_ENV, "fail_sync") };
        let failed = appender
            .append_operations_atomic(vec![tenant_op("t", "x", "a"), tenant_op("t", "y", "a")])
            .await;
        unsafe { std::env::remove_var(COMMIT_FAULT_ENV) };
        let err = failed
            .err()
            .ok_or_else(|| anyhow!("the injected sync must fail"))?;
        assert!(
            err.downcast_ref::<WalAppendIndeterminate>().is_some(),
            "{err:#}"
        );

        let later = appender
            .append_operations(vec![upsert_operation("t", "later")], None)
            .await
            .err()
            .ok_or_else(|| anyhow!("a poisoned appender must refuse"))?;
        // Refused before writing: "nothing written", not "outcome unknown".
        assert!(later.downcast_ref::<WalAppenderPoisoned>().is_some());
        assert!(later.downcast_ref::<WalAppendIndeterminate>().is_none());
        Ok(())
    }

    #[tokio::test]
    async fn memory_appender_atomic_assigns_contiguous_sequences() -> Result<()> {
        let appender = MemoryTableWalAppender::new();
        appender
            .append_operations(vec![upsert_operation("t", "a")], None)
            .await?;
        let entries = appender
            .append_operations_atomic(vec![tenant_op("t", "b", "x"), tenant_op("u", "c", "y")])
            .await?;
        assert_eq!(
            entries
                .iter()
                .map(|e| e.sequence_number)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(appender.entries().await.len(), 3);
        assert_eq!(appender.collection_max_sequence_for("u"), Some(3));
        Ok(())
    }

    #[tokio::test]
    async fn framed_table_wal_appender_truncates_trailing_partial_frame() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("partial.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("orders", "order-1")], None)
            .await?;

        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .await?;
        file.write_all(&FRAME_MAGIC[..4]).await?;
        file.flush().await?;

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let next_entries = reopened
            .append_operations(vec![upsert_operation("orders", "order-2")], None)
            .await?;
        assert_eq!(next_entries[0].sequence_number, 2);

        let final_reopen = FramedTableWalAppender::open(&wal_path).await?;
        let entries = final_reopen.read_entries().await?;
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.sequence_number)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        Ok(())
    }

    #[tokio::test]
    async fn framed_table_wal_appender_rejects_corrupt_complete_frame() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("corrupt.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_operation("orders", "order-1")], None)
            .await?;

        let mut bytes = tokio::fs::read(&wal_path).await?;
        let last_byte = bytes
            .last_mut()
            .expect("test WAL should contain one complete frame");
        *last_byte ^= 0x01;
        tokio::fs::write(&wal_path, bytes).await?;

        let err = match FramedTableWalAppender::open(&wal_path).await {
            Ok(_) => bail!("corrupt complete frame must not be silently truncated"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("decoding canonical WAL frame")
                || err
                    .chain()
                    .any(|cause| cause.to_string().contains("checksum mismatch"))
        );

        Ok(())
    }

    /// TD-OLAP-17 recovery rebuild: reopening a WAL (process restart) must
    /// re-seed the per-collection high-water from the recovered entries, so a
    /// read-mostly table fast-paths immediately instead of scanning every query
    /// until the next write repopulates it.
    #[tokio::test]
    async fn framed_table_wal_rebuilds_collection_high_water_on_open() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("highwater.wal");

        // Append for two collections, then drop the appender (simulate process exit).
        {
            let appender = FramedTableWalAppender::open(&wal_path).await?;
            appender
                .append_operations(vec![upsert_operation("orders", "o1")], None)
                .await?;
            appender
                .append_operations(vec![upsert_operation("orders", "o2")], None)
                .await?;
            appender
                .append_operations(vec![upsert_operation("lineitems", "l1")], None)
                .await?;
        }

        // Reopen = restart. The high-water is rebuilt from the recovered WAL:
        // orders → seq 2, lineitems → seq 3 (the global counter assigned 1,2,3).
        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        assert_eq!(reopened.collection_max_sequence_for("orders"), Some(2));
        assert_eq!(reopened.collection_max_sequence_for("lineitems"), Some(3));
        assert_eq!(reopened.collection_max_sequence_for("never-written"), None);

        Ok(())
    }
}
