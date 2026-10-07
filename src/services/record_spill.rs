//! Durable, bounded relational record storage — TD-USUB-1 slice 2a.
//!
//! [`MemtableRecordStorage`](crate::services::MemtableRecordStorage) is a `DashMap`
//! with no flush, evict, spill or compact path, so a long-lived relational table
//! grows the heap for the life of the process (ADR-094 defect **a**). Slice 1 made
//! that growth visible and boundable; this slice makes it *recoverable* by giving
//! the store somewhere to put rows.
//!
//! # What this slice does and does not claim
//!
//! It bounds resident memory **within a process run** — which is the operational
//! failure mode, because the process is OOM-killed long before it is restarted.
//!
//! It does **not** bound the heap *across* restarts, and saying otherwise would be
//! overclaiming: the canonical WAL is still authoritative and replay repopulates
//! the memtable in full. Making the bound survive a restart requires WAL
//! truncation, which is slice 2b and is gated on ADR-094's truncation rule
//! (truncate only below the minimum LSN durably committed in a catalog-referenced
//! segment).
//!
//! Keeping truncation out is what makes this slice **unable to lose data**:
//! recovery is byte-for-byte what it is today, and a segment is a read-side
//! representation of rows the WAL already made durable. If every segment were
//! deleted, the store would still be correct after replay.
//!
//! # Why it is a read-merge, not a spill
//!
//! `DirectWalTableRecordStore::ensure_unique_index_built` builds UNIQUE/PK
//! enforcement by **scanning the partition store**. A store that evicted rows
//! without serving them back would therefore silently stop catching duplicate
//! keys — a correctness regression disguised as a memory optimisation. So reads
//! here are a merge over `memtable ∪ segments`, with:
//!
//! * the **memtable winning** over any segment copy (it is strictly newer), and
//! * a **newer segment winning** over an older one, and
//! * **tombstones suppressing** a segment copy entirely.
//!
//! # Why tombstones exist at all
//!
//! The memtable hard-deletes (`records.remove(oid)`). A segment cannot: it is
//! immutable. So once a row has been flushed, deleting it means recording a
//! tombstone that suppresses it on read — and that tombstone has to survive the
//! Parquet round trip, which is exactly what TD-USUB-11's reserved
//! `__proxima_valid_to_ns` system column exists for. Without it a deleted row
//! would read back live.
//!
//! # Composition
//!
//! Segments are read back through [`read_segment_records`], the canonical
//! mixed-format router — which can decode Parquet only because TD-USUB-5 added
//! the `PAR1` arm and the `RecordReader` seam. Encoding uses the canonical
//! system-column encoder from TD-USUB-11. No decoder or encoder is hand-rolled
//! here.
//!
//! # Default OFF
//!
//! With no flush threshold configured this never writes a segment and behaves
//! exactly like the plain memtable, so enabling nothing changes nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use proximadb_records::{
    ProximaRecord, RecordKey, RecordScan, RecordScanOptions, RecordScanPredicate, RecordStore,
    RecordStoreResult,
};
use proximadb_storage_common::proxima_arrow::{
    infer_proxima_schema, proxima_records_to_record_batch_with_system_columns,
};
use proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes;
use proximadb_storage_filesystem_types::FileSystem;

use crate::storage::engines::sst::segment_format::read_segment_records;

/// Flush threshold: resident records after which the memtable is written to a
/// segment. Unset ⇒ **never flush** (the shipped default — behaviourally
/// identical to the plain memtable).
pub const SPILL_MAX_RESIDENT_ENV: &str = "PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT";

/// Read the configured flush threshold. `None` ⇒ never flush.
///
/// A zero or unparsable value reads as unset rather than "flush on every write" —
/// a typo must not turn every insert into an object write.
///
/// Read at every partition construction rather than cached in a `OnceLock`: the
/// gate must be observable per store, and a process-lifetime cache is what made
/// the TD-USUB-6 residual-tail benchmark measure the same mode twice.
pub(crate) fn configured_flush_threshold() -> Option<usize> {
    std::env::var(SPILL_MAX_RESIDENT_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// A relational record store that keeps recent rows resident and flushes older
/// ones to durable Parquet segments, serving reads as a merge over both.
#[derive(Debug)]
pub struct SpillRecordStorage {
    /// Resident rows. Always strictly newer than any segment copy.
    memtable: DashMap<String, ProximaRecord>,
    /// Oids deleted after having been flushed. Suppresses the segment copy on
    /// read; cleared for an oid that is re-inserted.
    tombstones: DashSet<String>,
    /// `oid → durable WAL LSN` for rows still RESIDENT, i.e. not yet in a
    /// segment. The minimum over this map is the lowest LSN the WAL must retain
    /// on this partition's account (TD-USUB-1, slice 2b prerequisite).
    ///
    /// Kept beside the memtable rather than on `ProximaRecord` so the central
    /// record type is untouched and the LSN never has to survive the Parquet
    /// round trip — once a row is in a segment it is durable there, and its LSN
    /// stops constraining truncation, which is exactly when this entry is
    /// dropped.
    ///
    /// Populated only through [`RecordStore::upsert_record_at_lsn`]. A write
    /// arriving via plain `upsert_record` contributes **no** entry — which is
    /// why `min_unflushed_lsn` refuses to answer when any resident row is
    /// untracked, rather than returning a minimum that silently excludes it.
    resident_lsns: DashMap<String, u64>,
    /// Flushed segment paths, oldest first. Later entries shadow earlier ones.
    segments: parking_lot::RwLock<Vec<String>>,
    /// Serializes `flush` against any other structural mutation of the
    /// segment set — today `delete_record` and `purge_durable_objects`.
    ///
    /// **Why a lock held across object I/O (TD-USUB-13).** `flush` publishes in
    /// three steps that were not atomic: snapshot the memtable, write the
    /// object, then push the path to `segments`. `delete_record` decides whether
    /// to record a tombstone from `!segments.is_empty()`. A delete landing
    /// between the snapshot and the push therefore saw an empty segment list on
    /// a partition's first flush, recorded **no tombstone**, and reported
    /// success — while flush went on to write that row into the segment from its
    /// pre-delete snapshot. The row came back live, with no WAL truncation
    /// involved (mandate #16a).
    ///
    /// `tokio::sync::Mutex`, not `parking_lot`: the critical section spans an
    /// `await` on the object write, and a blocking mutex held across an await
    /// stalls the whole runtime thread.
    ///
    /// **Not the cheapest possible fix, chosen deliberately.** TD-USUB-13 lists
    /// a finer-grained alternative (publish the in-flight oid set under a short
    /// lock, release before the I/O, and have `delete_record` consult it). That
    /// avoids holding anything across the write, but it admits a tombstone for a
    /// row whose flush then *fails* — and a spurious tombstone now pins WAL
    /// truncation forever, because `min_unflushed_lsn` fails closed on any
    /// tombstone. Serializing is obviously correct and introduces no such state;
    /// the spill path has no production caller yet, so the contention it costs is
    /// currently zero and should be *measured* before trading correctness
    /// obviousness for it (mandate #6).
    flush_guard: tokio::sync::Mutex<()>,
    /// Resident-row count after which a write triggers a flush. `None` ⇒ never.
    flush_threshold: Option<usize>,
    filesystem: Arc<dyn FileSystem>,
    /// Directory under which this partition's segments are written. Callers build
    /// it with `DrPathBuilder`; this store never constructs a raw path itself.
    base_path: String,
    /// Monotonic segment counter, **resumed from durable state** by
    /// [`Self::ensure_discovered`] before the first flush.
    ///
    /// Resumption is what keeps names fresh across instances that share a
    /// `base_path` **sequentially** — a process restart, or a
    /// `drop_partition_state` eviction whose next access rebuilds the partition
    /// after the old store is gone. A store that started this counter at 0
    /// regardless would re-issue a live segment's name.
    ///
    /// It does NOT make names fresh across instances that are **concurrently
    /// live**, and that case is reachable: `drop_partition_state` only removes
    /// the map entry, so a caller still holding the old `Arc` keeps writing
    /// while the next access builds a second store over the same prefix.
    /// Discovery runs once per store, so both can resume to the same number and
    /// both issue it. There the guarantee is `write_if_absent`, which fails the
    /// loser's flush rather than letting it destroy a segment — a lost write
    /// surfaced as an error, not silent corruption (mandate #1). The removed
    /// per-instance UUID nonce was collision-free in that case; this is a
    /// deliberate trade of concurrent-writer tolerance for a segment set whose
    /// ORDER is recoverable from a listing, which the nonce made impossible and
    /// on which all three read paths depend. Single-writer-per-partition is the
    /// design (one store per `(tenant, collection)` per process, one process per
    /// WAL), so the traded-away case is already outside it.
    ///
    /// This replaces the per-instance UUID nonce that previously bought the same
    /// freshness: a nonce makes names unique but leaves them **unordered**,
    /// because a lexicographic sort of `spill-{uuid}-{seq}` sorts on the random
    /// nonce. All three read paths depend on segment order (newest wins), so a
    /// nonce-named set cannot be recovered from a listing — only its membership
    /// can. A resumed, zero-padded counter delivers freshness *and* order, which
    /// is why `src/graph/cold_segment_store.rs` — the precedent TD-USUB-1
    /// names — uses one.
    next_segment: AtomicU64,
    /// Guards [`Self::ensure_discovered`] so the listing runs at most once per
    /// store, and so concurrent first-touches cannot both populate `segments`.
    discovered: tokio::sync::OnceCell<()>,
}

impl SpillRecordStorage {
    /// Create a store rooted at `base_path`, with the flush threshold taken from
    /// the environment (unset ⇒ never flush).
    pub fn new(filesystem: Arc<dyn FileSystem>, base_path: impl Into<String>) -> Self {
        Self::with_flush_threshold(filesystem, base_path, configured_flush_threshold())
    }

    /// Create a store with an explicit flush threshold — used by tests and by
    /// callers that configure the bound directly rather than through the
    /// environment.
    pub fn with_flush_threshold(
        filesystem: Arc<dyn FileSystem>,
        base_path: impl Into<String>,
        flush_threshold: Option<usize>,
    ) -> Self {
        Self {
            memtable: DashMap::new(),
            tombstones: DashSet::new(),
            resident_lsns: DashMap::new(),
            segments: parking_lot::RwLock::new(Vec::new()),
            flush_guard: tokio::sync::Mutex::new(()),
            flush_threshold,
            filesystem,
            base_path: base_path.into(),
            next_segment: AtomicU64::new(0),
            discovered: tokio::sync::OnceCell::new(),
        }
    }

    /// Resident (unflushed) row count. This is the number the heap bound applies
    /// to — NOT the logical row count of the table.
    pub fn resident_len(&self) -> usize {
        self.memtable.len()
    }

    /// Number of durable segments this store knows about.
    ///
    /// Sync, so it cannot itself discover: before any path that calls
    /// `ensure_discovered` has run, this reports only what THIS instance
    /// flushed, not what the prefix holds. Callers that need the durable count
    /// should touch a read path first.
    pub fn segment_count(&self) -> usize {
        self.segments.read().len()
    }

    /// The configured flush threshold, if any.
    pub fn flush_threshold(&self) -> Option<usize> {
        self.flush_threshold
    }

    /// Candidate segment objects for this partition, as paths under this
    /// store's own prefix.
    ///
    /// These are CANDIDATES, not a verified set: a listing can propose an object
    /// that is not ours (see below), and the guarantee is about where a returned
    /// path can point, not about whether something lives there.
    ///
    /// The one place that interprets a listing, shared by discovery and purge so
    /// the rule lives once (mandate #12).
    ///
    /// # Why a listing cannot simply be trusted
    ///
    /// Only the local backend — and HDFS, whose `LISTSTATUS` is also
    /// directory-scoped, though the factory never constructs it — lists a
    /// *directory* (`read_dir`, non-recursive). A directory-scoped backend is
    /// the safe case; the hazard below is the prefix-scoped ones.
    /// The object stores list by key prefix and **recursively**, and GCS passes
    /// the prefix through raw with no trailing delimiter
    /// (`ListObjectsRequest { prefix }`). So a partition at `…/order` would also
    /// see `…/orders/…` — a different table's objects, whose basenames parse as
    /// perfectly good segment names.
    ///
    /// # The two properties that make this safe
    ///
    /// **1. Correctness comes from canonical reconstruction, not from the
    /// listing.** A returned path is always under `{base_path}/`, and for a
    /// parseable name it is exactly `{base_path}/{spill_segment_name(seq)}`,
    /// rebuilt from the sequence rather than from the listing. (The legacy arm
    /// is the one exception: a name with no sequence has only its basename to
    /// identify it, so that is re-rooted under our prefix instead. `entry.name`
    /// is a bare basename on every backend, so it cannot contain a `/` and
    /// cannot escape the prefix — the *where* guarantee holds for both arms,
    /// which is what the rest of this argument needs.) So
    /// a candidate that is not in fact ours cannot name another table's object:
    /// it names a path under OUR prefix, which either holds our segment or holds
    /// nothing. Reading it then fails loudly with not-found; purge deletes only
    /// paths under our own prefix and tolerates absence. The failure mode is an
    /// error, never a wrong answer (mandate #1), and that is what round 1's
    /// `entry.url`-deleting purge got wrong: it deleted the *listing's* path.
    ///
    /// **2. Availability comes from scoping the listing.** `list` is called with
    /// a trailing delimiter, which is what keeps GCS's raw prefix from matching a
    /// sibling table. Without it, a sibling's segment at a sequence we lack
    /// becomes a canonical path that does not exist, and every read of this
    /// partition fails — correct, but useless. The delimiter is a no-op
    /// elsewhere, though for a reason worth stating precisely: `ObjPath::from`
    /// splits on `/` and drops empty segments (`Path::from("a/b/") == "a/b"`),
    /// so our delimiter is DISCARDED before it reaches S3/Azure — those two are
    /// scoped because `object_store` re-appends a delimiter itself when listing
    /// a prefix. That is upstream behaviour this repo does not pin, so if it
    /// ever changed, S3/Azure would acquire the sibling-prefix availability
    /// problem and no test here would catch it (the local-backend double cannot
    /// model it). GCS keeps our delimiter verbatim, which is the backend this
    /// call is for. `read_dir` ignores it.
    ///
    /// Note what is deliberately NOT done: membership is not decided by
    /// comparing the listing's location against `base_path`. Those two strings
    /// come from different places — `base_path` is whatever the caller passed,
    /// `DirEntry::url` is reconstructed by the backend — and they disagree in
    /// real configurations (a bare-relative `base_path`, the shape
    /// `DrPathBuilder` emits, against a `root_dir`-anchored local filesystem;
    /// S3/Azure percent-encoding characters `DrPathBuilder::validate_id` permits
    /// in an identifier). A mismatch would have skipped **our own** segments,
    /// which is silent loss.
    ///
    /// Nor is membership probed with `exists`, which is unreliable in BOTH
    /// directions. False negative: every backend collapses a transport failure
    /// into `Ok(false)` (`aws_s3`/`azure_blob`/`gcs_store` are
    /// `Ok(head(..).is_ok())`, and `std::path::Path::exists` is false on
    /// `EACCES`/`EIO` too), so one throttled HEAD would read as "not ours" — and
    /// because discovery is a `OnceCell`, that answer would stick for the
    /// store's lifetime. False positive: `UnifiedCachingFilesystem::exists`
    /// returns `Ok(true)` on any metadata-cache hit, and `put_negative` inserts
    /// an ordinary entry, so a path confirmed ABSENT reads back as PRESENT for
    /// the next 60s. An oracle that errs both ways cannot decide membership.
    async fn list_partition_objects(&self) -> Result<Vec<PartitionObject>> {
        let prefix = self.base_path.trim_end_matches('/').to_string();
        // Trailing delimiter: see property 2 above.
        let entries = match self.filesystem.list(&format!("{prefix}/")).await {
            Ok(entries) => entries,
            // A partition that never flushed has no prefix at all: the common
            // case, not an error. BOTH `is_absent` arms are needed — the local
            // backend reports a missing directory as `Io(ErrorKind::NotFound)`
            // from `read_dir`, not the typed `NotFound` variant.
            Err(err) if is_absent(&err) => return Ok(Vec::new()),
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "spill: list segment prefix '{prefix}' failed: {err}"
                ));
            }
        };

        let mut objects: Vec<PartitionObject> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in entries {
            if entry.metadata.is_directory {
                continue;
            }
            // Basename only. A recursive listing reports whole keys, and `name`
            // is already `key.rsplit('/').next()` on the object stores.
            let name = entry.name.rsplit('/').next().unwrap_or(&entry.name);
            if !is_spill_segment(name) {
                // A wrapper filesystem that rewrites names on write but not on
                // `list` would otherwise make every segment invisible here.
                // Silent loss plus a permanently failing flush — refuse instead.
                if is_mangled_spill_segment(name) {
                    return Err(anyhow::anyhow!(
                        "spill: segment '{name}' under '{prefix}' carries an extra extension, so \
                         a filesystem wrapper is rewriting object names without un-mangling \
                         `list` (see TD-ENCFS-1); refusing to serve this partition rather than \
                         report it empty"
                    ));
                }
                continue;
            }
            let seq = parse_spill_seq(name);
            let path = match seq {
                Some(seq) => format!("{prefix}/{}", spill_segment_name(seq)),
                // A legacy name carries no sequence to rebuild from, so its
                // basename is the only locator. Still re-rooted under OUR
                // prefix, so the guarantee above holds; only purge consumes it.
                None => format!("{prefix}/{name}"),
            };
            // A paginated listing can repeat a key, and two candidates can map
            // to one canonical path.
            //
            // This is a COST guard, not a correctness one, and mutation testing
            // is what established the difference: removing it changes no
            // observable behaviour, because `ensure_discovered` independently
            // refuses a path already in `segments` and purge tolerates a
            // second delete of the same object as absent. What it buys is
            // avoiding the duplicate work before those backstops see it.
            if !seen.insert(path.clone()) {
                continue;
            }
            objects.push(PartitionObject { path, seq });
        }
        Ok(objects)
    }

    /// Populate `segments` from durable state and resume `next_segment` past it,
    /// so a store built over a prefix an earlier instance wrote serves that
    /// instance's rows instead of ignoring them — and never reuses a name.
    ///
    /// Runs at most once per store, lazily. It must precede the FIRST path whose
    /// answer depends on the segment list, not merely the first read:
    /// `delete_record` decides whether to record a tombstone from
    /// `!segments.is_empty()`, so a delete that ran before discovery would skip
    /// the tombstone and a later discovery would then expose the flushed copy —
    /// a resurrection. Hence the call at the head of every such path.
    ///
    /// **Order comes from the names, not from a durable list.** Each name
    /// carries its sequence, and discovery sorts on the parsed value. That is
    /// deliberate: a segment written but not yet recorded anywhere is still
    /// found, and still ordered correctly. A design that took order from a
    /// catalog-side list instead would make the window between "object landed"
    /// and "list updated" a silent-data-loss window once the WAL is truncated
    /// (slice 2b).
    ///
    /// Scope: this recovers segments under a prefix the caller already knows.
    /// It does NOT let recovery *find* that prefix — WAL replay carries no
    /// namespace, so the prefix itself still has to come from the catalog. See
    /// `DirectTableRecordWriter::new_spilling` and TD-USUB-1.
    ///
    /// **Never call this while holding `flush_guard`**: it acquires the guard
    /// itself, so a caller that already held it would wait on the `OnceCell`
    /// while the initializer waits on the guard — a permanent hang. Every caller
    /// is correct today (`flush` and `delete_record` discover *before* taking
    /// it; `purge_durable_objects` never discovers), and after the first success
    /// the cell short-circuits without touching the guard, so the window is the
    /// first call only. `tokio::sync::Mutex` exposes no ownership query, so
    /// neither a `debug_assert` nor a type-state can enforce this — hence prose.
    ///
    /// Why the guard is taken at all: `purge_durable_objects` CLEARS `segments`
    /// under it, so a discovery whose listing ran before a concurrent purge
    /// would otherwise publish paths the purge had already deleted. It is NOT
    /// needed against `flush` — `flush` is the only other writer of `segments`
    /// and it discovers first, so the list is provably empty inside the
    /// initializer. An earlier version of this comment gave the flush race as
    /// the reason, which was wrong.
    async fn ensure_discovered(&self) -> Result<()> {
        self.discovered
            .get_or_try_init(|| async {
                // Exclusive with `purge_durable_objects`, which CLEARS
                // `segments` under this same guard. Without it a discovery whose
                // listing ran before a concurrent purge could publish paths the
                // purge has already deleted, after the purge emptied the list.
                //
                // NOT needed against `flush`: `flush` is the only writer of
                // `segments` and it discovers first, so `segments` is provably
                // empty whenever this closure runs. An earlier version of this
                // comment claimed the flush race as the reason, which was wrong
                // — the guard is load-bearing, for purge.
                let _exclusive = self.flush_guard.lock().await;
                let objects = self.list_partition_objects().await?;

                let mut found: Vec<(u64, String)> = Vec::with_capacity(objects.len());
                for object in objects {
                    // Segment-shaped but unparseable: FAIL, do not skip
                    // (mandate #1). Skipping would drop a durable segment out of
                    // the merge silently — rows held only in that segment would
                    // read as absent, and a DELETE of such a row would record no
                    // tombstone. Without a sequence this store cannot establish
                    // the order its read paths require, so it refuses to serve
                    // rather than serve a subset. `purge_durable_objects` still
                    // reclaims the object, so a DROP is the way out.
                    let Some(seq) = object.seq else {
                        return Err(anyhow::anyhow!(
                            "spill: segment '{}' has an unrecognized name; refusing to serve \
                             this partition from a segment set whose order cannot be \
                             established",
                            object.path
                        ));
                    };
                    found.push((seq, object.path));
                }

                found.sort_by_key(|(seq, _)| *seq);
                // `checked_add`, not `seq + 1`: the sequence comes from a
                // FILENAME, so `spill-ffffffffffffffff.parquet` — a corrupt or
                // hostile object under the prefix — would otherwise overflow.
                // That panics in debug (mandate #4) and WRAPS in release, and a
                // wrap resumes at 0, straight onto live segment names (mandate
                // #1). Exhaustion is an explicit refusal instead.
                let resume = match found.last() {
                    Some((seq, _)) => match seq.checked_add(1) {
                        Some(next) => next,
                        None => {
                            return Err(anyhow::anyhow!(
                                "spill: segment sequence space exhausted under '{}' (highest \
                                 segment is {seq}); refusing to serve rather than reuse a live \
                                 segment name",
                                self.base_path.trim_end_matches('/')
                            ));
                        }
                    },
                    None => 0,
                };

                {
                    let mut segments = self.segments.write();
                    for (_, path) in found {
                        // Defense-in-depth, not a guard against an observed
                        // case: `flush` is the only other writer of `segments`
                        // and it discovers first, so this list is provably empty
                        // here. Kept because a duplicate entry would silently
                        // double the reads a segment costs, at the price of one
                        // short linear scan run once per store.
                        if !segments.contains(&path) {
                            segments.push(path);
                        }
                    }
                }
                // `fetch_max`, not `store`: defense-in-depth, so a counter that
                // somehow already advanced past `resume` is never rewound onto a
                // name it already used.
                self.next_segment.fetch_max(resume, Ordering::SeqCst);
                Ok(())
            })
            .await
            .map(|_| ())
    }

    /// Write every resident row to a fresh durable segment and clear the
    /// memtable. A no-op when nothing is resident.
    ///
    /// Tombstones are deliberately **retained** across a flush: they suppress
    /// copies in *older* segments, which this flush does not rewrite.
    pub async fn flush(&self) -> Result<Option<String>> {
        // BEFORE the guard: `ensure_discovered` takes it. Also before the
        // `fetch_add` below, which is what makes this flush's name fresh with
        // respect to an earlier instance's segments.
        self.ensure_discovered().await?;
        // Held for the WHOLE publish — snapshot, write, push, evict — so no
        // delete can observe the pre-push segment list for a row this flush is
        // about to make durable (TD-USUB-13). See `flush_guard`.
        let _publish = self.flush_guard.lock().await;
        let records: Vec<ProximaRecord> = self
            .memtable
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        if records.is_empty() {
            return Ok(None);
        }

        // Parquet is self-describing, so the segment carries its own schema and
        // read-back needs no catalog lookup. `with_system_columns` is what keeps
        // identity and `valid_to_ns` intact (TD-USUB-11).
        let schema = infer_proxima_schema(&records);
        let batch = proxima_records_to_record_batch_with_system_columns(&records, &schema)
            .context("spill: encode resident records to Arrow")?;
        // The file schema MUST be the batch's own schema, not `schema`: the batch
        // carries the reserved system columns on top of the user columns, and
        // writing it under the bare user schema silently drops them — which is
        // exactly the identity/tombstone loss this store depends on not happening.
        let file_schema = batch.schema();
        let bytes = record_batches_to_parquet_bytes(&[batch], file_schema, None)
            .context("spill: encode Arrow batch to parquet bytes")?;

        // `next_segment` was resumed past every durable segment by
        // `ensure_discovered` at the top of this method, so this number is fresh
        // with respect to objects an earlier instance wrote at the same path —
        // not merely fresh within this process. See the `next_segment` field.
        let seq = self.next_segment.fetch_add(1, Ordering::SeqCst);
        let path = format!(
            "{}/{}",
            self.base_path.trim_end_matches('/'),
            spill_segment_name(seq)
        );

        // `create_dirs`: a local-filesystem backend creates the parent only when
        // asked (`local.rs` gates it on this flag), and a partition's base path
        // is fresh by construction — nothing has created
        // `…/{tenant}/{collection}/` before the first flush. Object-store
        // backends ignore it: keys are flat (ADR-036).
        //
        // Not caught until this store was wired behind the partition factory:
        // its own tests hand it a `tempdir()` that already exists, so they
        // exercised the flush MECHANISM while never exercising the path a real
        // caller supplies.
        //
        // `write_if_absent`, NOT `write` with `overwrite: false`. A segment is
        // immutable and the name above is fresh by construction — so a collision
        // is a bug, and it must fail loudly rather than destroy a durable segment
        // (mandate #1: fail closed, never silently wrong).
        //
        // `FileOptions::overwrite` would NOT deliver that: only the local backend
        // consults it. `aws_s3`/`azure_blob`/`gcs_store` all ignore it in `write`
        // and issue an unconditional PUT, so on exactly the deployments that
        // matter the "fail loudly" guarantee would be absent while the code
        // claimed it. `write_if_absent` is this codebase's designated commit
        // primitive — a real conditional create (`PutMode::Create` on S3,
        // `create_new(true)` locally), whose trait doc requires implementations
        // "must not emulate it with a racy exists-then-write", and whose default
        // impl errors rather than silently downgrading.
        //
        // This CANNOT wedge a retry, so do not "fix" it back to `write`: `seq` is
        // taken by `fetch_add` ABOVE, so a failed flush consumes its number and
        // the next attempt uses a new one. A name is never reused. The cost is
        // that a write which failed *after* the object landed leaves an orphan —
        // inherent to ADR-062 fresh-name discipline, and strictly better than
        // overwriting a live segment.
        let options = crate::storage::persistence::filesystem::FileOptions {
            create_dirs: true,
            overwrite: false,
            ..Default::default()
        };
        self.filesystem
            .write_if_absent(&path, &bytes, Some(options))
            .await
            .map_err(|e| anyhow::anyhow!("spill: write segment '{path}' failed: {e}"))?;

        // Only drop the resident rows AFTER the segment is durable. A crash
        // before this point simply leaves them resident, and the WAL replays
        // them regardless — the store cannot lose a row either way.
        self.segments.write().push(path.clone());
        for record in &records {
            self.memtable.remove(&record.oid);
            // The row is durable in the segment now, so its WAL entry no longer
            // constrains truncation. Dropping the association here is what makes
            // `min_unflushed_lsn` advance.
            self.resident_lsns.remove(&record.oid);
        }
        Ok(Some(path))
    }

    /// Decode one segment back to records through the canonical mixed-format
    /// router (TD-USUB-5), which detects `PAR1` and dispatches to the Parquet
    /// `RecordReader`.
    async fn read_segment(&self, path: &str) -> Result<Vec<ProximaRecord>> {
        let bytes = self
            .filesystem
            .read(path)
            .await
            .map_err(|e| anyhow::anyhow!("spill: read segment '{path}' failed: {e}"))?;
        read_segment_records(&bytes, &[], &[], None)
            .with_context(|| format!("spill: decode segment '{path}'"))
    }

    /// Every live row, newest-wins: segments oldest→newest, then the memtable on
    /// top, with tombstoned oids removed.
    ///
    /// Known cost, stated rather than hidden: this reads every segment on every
    /// scan. Bounding that is compaction plus a per-segment oid index — slice 2c.
    /// It is acceptable here only because the path is default-OFF.
    async fn merged_records(&self) -> Result<Vec<ProximaRecord>> {
        self.ensure_discovered().await?;
        let paths = self.segments.read().clone();
        let mut merged: std::collections::HashMap<String, ProximaRecord> =
            std::collections::HashMap::new();

        for path in &paths {
            for record in self.read_segment(path).await? {
                merged.insert(record.oid.clone(), record);
            }
        }
        for entry in self.memtable.iter() {
            merged.insert(entry.key().clone(), entry.value().clone());
        }
        for oid in self.tombstones.iter() {
            merged.remove(oid.key());
        }
        // Mandate #16a: apply the canonical dead-record predicate at this read
        // boundary. `ProximaRecord::is_dead` covers both the `Some(0)` tombstone
        // and a TTL-expired `valid_to_ns`.
        //
        // A no-op on today's relational path, and deliberately added anyway. The
        // store's own suppression is an IN-MEMORY set, so it protects only rows
        // this process deleted; it says nothing about a record that arrives from
        // a segment already carrying `valid_to_ns`. The predicate is idempotent
        // and the mandate explicitly encourages the overlap, so the cost of
        // having it is one comparison per row and the cost of lacking it is a
        // resurrected row (TD-USUB-11's failure mode) the first time anything
        // writes a dead record here.
        let now_ns = now_ns();
        Ok(merged
            .into_values()
            .filter(|r| !r.is_dead(now_ns))
            .collect())
    }
}

#[async_trait]
impl RecordStore for SpillRecordStorage {
    async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord> {
        // Re-inserting a previously deleted oid must clear its tombstone, or the
        // new row would stay invisible behind the suppression of the old one.
        self.tombstones.remove(&record.oid);
        self.memtable.insert(record.oid.clone(), record.clone());

        if let Some(threshold) = self.flush_threshold
            && self.memtable.len() >= threshold
        {
            self.flush().await?;
        }
        Ok(record)
    }

    async fn upsert_record_at_lsn(
        &self,
        record: ProximaRecord,
        lsn: u64,
    ) -> RecordStoreResult<ProximaRecord> {
        // Record the association BEFORE the write, so a flush triggered by this
        // very insert already sees it and can evict it. Doing it afterwards
        // would leave a flushed oid tracked as resident, pinning the WAL at an
        // LSN that is in fact already durable in a segment.
        self.resident_lsns.insert(record.oid.clone(), lsn);
        self.upsert_record(record).await
    }

    /// Minimum LSN among resident (unflushed) rows.
    ///
    /// Deliberately does NOT call `ensure_discovered`, and must not: it reads
    /// only `resident_lsns` and `tombstones`, both in memory, and discovery adds
    /// to neither — it adds *flushed* segments, whose rows are by definition no
    /// longer resident and no longer constrain truncation. Adding the call would
    /// be harmless but misleading about what this answer depends on.
    ///
    /// **Fails closed when any resident row is untracked.** A row written through
    /// plain `upsert_record` has no LSN association, so a minimum computed over
    /// the tracked set alone would silently exclude it — and a caller truncating
    /// below that minimum would discard the only durable copy of that row. When
    /// the counts disagree this therefore returns an error rather than a number:
    /// the consumer's correct response is to not truncate.
    ///
    /// `Ok(None)` means genuinely nothing resident, i.e. no constraint.
    async fn min_unflushed_lsn(&self) -> RecordStoreResult<Option<u64>> {
        // An in-memory tombstone is NEVER durable, so the WAL entry that
        // recreates it can never be safely discarded — and a tombstone is not
        // cleared by `flush()` (it must keep suppressing the segment copy until
        // compaction removes that segment; only a re-insert or a purge clears
        // it). Reporting a minimum here would be actively wrong:
        //
        //   row inserted @10 -> flushed (association dropped)
        //   row deleted  @20 -> tombstone in memory only, nothing resident
        //   -> a minimum over resident rows says "no constraint"
        //   -> truncation discards @20
        //   -> restart: replay never sees the delete, the segment still holds the
        //      row live, and the DELETED ROW RESURRECTS (mandate #16a).
        //
        // TD-USUB-1 already names persisting tombstones as a hard precondition
        // on 2b. Until that lands, any tombstone makes the question unanswerable.
        if !self.tombstones.is_empty() {
            return Err(anyhow::anyhow!(
                "spill: cannot compute a safe truncation point — {} in-memory tombstone(s) \
                 exist, and a tombstone is not durable in any segment, so the WAL entries that \
                 reconstruct them must be retained. Persisting tombstones into the segment is a \
                 precondition on slice 2b (TD-USUB-1). Do not truncate.",
                self.tombstones.len()
            ));
        }
        let resident = self.memtable.len();
        if resident == 0 {
            return Ok(None);
        }
        let tracked = self.resident_lsns.len();
        if tracked < resident {
            return Err(anyhow::anyhow!(
                "spill: cannot compute a safe truncation point — {} resident row(s) but only {} \
                 carry a WAL LSN. Rows written through `upsert_record` instead of \
                 `upsert_record_at_lsn` are untracked, and a minimum over the tracked subset \
                 would exclude them. Do not truncate.",
                resident,
                tracked
            ));
        }
        Ok(self.resident_lsns.iter().map(|e| *e.value()).min())
    }

    async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>> {
        self.ensure_discovered().await?;
        if self.tombstones.contains(&key.oid) {
            return Ok(None);
        }
        let now_ns = now_ns();
        if let Some(entry) = self.memtable.get(&key.oid) {
            // The MEMTABLE is a read boundary too (mandate #16a). #1952 applied
            // the predicate to the segment arms and this commit added the third
            // one in `delete_record`; both left the resident copies unfiltered,
            // so a row written already-dead and never flushed read back live.
            // The memtable is strictly newer than any segment, so a dead
            // resident copy means the row is dead — return None rather than
            // falling through to a stale segment copy.
            let record = entry.value().clone();
            return Ok((!record.is_dead(now_ns)).then_some(record));
        }
        // Newest segment first: a later flush shadows an earlier copy.
        // `now_ns` is taken once above and reused, so the memtable and segment
        // arms judge liveness against the same instant.
        let paths = self.segments.read().clone();
        for path in paths.iter().rev() {
            if let Some(found) = self
                .read_segment(path)
                .await?
                .into_iter()
                .find(|r| r.oid == key.oid)
            {
                // Mandate #16a here too. Newest segment first, so the first hit
                // is authoritative: a dead hit means the row is GONE, not that an
                // older segment should be consulted for a live copy.
                if found.is_dead(now_ns) {
                    return Ok(None);
                }
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
        // One instant for both liveness decisions below (the resident copy and
        // the segment probe), so they cannot disagree about the same row — the
        // same property `get_record` states for its two arms.
        let now_ns = now_ns();
        // BEFORE the guarded block below, for two reasons: `ensure_discovered`
        // takes the same guard, and the `has_segments` decision it guards is
        // only correct once discovery has run.
        self.ensure_discovered().await?;
        // Exclusive with `flush` (TD-USUB-13): `has_segments` below and the
        // tombstone insert must not straddle a flush's `segments.push`, or a row
        // that flush is mid-way through making durable gets no tombstone and
        // comes back live. Scoped so the guard is released before the segment
        // READS at the end of this method — those are idempotent and need no
        // exclusion.
        let (was_resident, resident_was_dead, has_segments) = {
            let _exclusive = self.flush_guard.lock().await;

            // Already suppressed: nothing live to remove, so report false even
            // if a stale segment copy still exists on disk.
            if self.tombstones.contains(&key.oid) {
                self.memtable.remove(&key.oid);
                return Ok(false);
            }

            // Remove unconditionally, and distinguish THREE states rather than
            // two: absent, resident-and-live, resident-and-dead.
            //
            // Collapsing the last two into one `false` was a real defect: a dead
            // resident copy then fell through to the segment probe below, where a
            // stale LIVE copy of the same oid reported that this DELETE removed a
            // row — while `get_record` and `merged_records`, on identical state,
            // both report the row absent. Mandate #16a names affected-row counts.
            let removed = self.memtable.remove(&key.oid);
            let resident_was_dead = removed
                .as_ref()
                .is_some_and(|(_, record)| record.is_dead(now_ns));
            let was_resident = removed.is_some() && !resident_was_dead;
            let has_segments = !self.segments.read().is_empty();

            // A row already written to an immutable segment cannot be removed,
            // so it is suppressed instead. Inside the guard with the
            // `has_segments` read: deciding and acting on that decision as one
            // step is what makes the reasoning local, rather than depending on
            // the separate argument that flush's snapshot could no longer
            // contain this oid.
            if has_segments {
                self.tombstones.insert(key.oid.clone());
            }
            // A deleted row is no longer resident, so its INSERT's LSN stops
            // constraining truncation. The DELETE's own LSN is a separate
            // matter: if the row had been flushed, the tombstone above is the
            // only in-memory evidence of the deletion and `min_unflushed_lsn`
            // fails closed on it — dropping this association does NOT release
            // that obligation.
            self.resident_lsns.remove(&key.oid);

            (was_resident, resident_was_dead, has_segments)
        };

        // Report truthfully whether a live row went away. The segment read is
        // paid ONLY when the answer is not already known from the memtable.
        if was_resident {
            return Ok(true);
        }
        // The memtable is strictly newer than any segment, so a dead resident
        // copy means the row is dead: do not consult the segments, exactly as
        // `get_record` does not. The tombstone inserted above is what keeps the
        // older live segment copy suppressed, so returning here cannot resurrect
        // it.
        if resident_was_dead {
            return Ok(false);
        }
        if !has_segments {
            return Ok(false);
        }
        let paths = self.segments.read().clone();
        for path in paths.iter().rev() {
            // Newest segment first, and the FIRST hit is authoritative — the
            // same rule `get_record` states and `merged_records` implements. A
            // dead hit means the row is gone; it does NOT mean an older segment
            // should be consulted for a live copy.
            //
            // Getting this wrong is subtle, and an earlier version of this
            // commit did: it filtered dead rows *within* each segment and then
            // CONTINUED to an older one, so a row dead in the newest segment and
            // live in an older one reported that this DELETE removed it — while
            // `get_record` and `merged_records`, on identical state, both
            // reported it absent. Filtering per-segment answers "does this
            // segment hold a live copy", which is the wrong question; the right
            // one is "what does the newest segment holding this oid say".
            if let Some(found) = self
                .read_segment(path)
                .await?
                .into_iter()
                .find(|r| r.oid == key.oid)
            {
                return Ok(!found.is_dead(now_ns));
            }
        }
        Ok(false)
    }

    /// Delete every spill segment under this partition's prefix.
    ///
    /// Deletes by **listing the prefix**, not by walking the in-memory
    /// `segments` list, and that difference is the point: `base_path` is derived
    /// deterministically from `(tenant, collection)`, so the prefix also holds
    /// segments written by *earlier* store instances at the same path — a prior
    /// process, or an instance discarded by `drop_partition_state`. Those are
    /// invisible to this instance's list and would survive a DROP forever.
    ///
    /// A missing prefix is success, not failure: a partition that never flushed
    /// has nothing to delete, and a re-drop must not error (the contract requires
    /// idempotence). Every other I/O error propagates — reporting success for a
    /// failed delete would leave data the tenant cannot reach or remove.
    async fn purge_durable_objects(&self) -> RecordStoreResult<()> {
        // Exclusive with `flush` for the same reason as `delete_record`
        // (TD-USUB-13): purge lists the prefix and deletes what it finds, so a
        // concurrent flush could publish a segment *after* the listing and leave
        // an object the purge was supposed to remove — orphaned, with the
        // partition about to be dropped and nothing left to reference it.
        let _exclusive = self.flush_guard.lock().await;

        // Shared with discovery, which is what makes this safe on the object
        // stores: a listing is by key prefix and recursive there, and GCS passes
        // the prefix through raw, so dropping table `order` would otherwise
        // delete table `orders`' segments — round 1's defect was deleting the
        // LISTING's path. `list_partition_objects` returns paths rebuilt
        // canonically under our own prefix, so a candidate that is not ours
        // names a path that holds nothing and the delete below tolerates its
        // absence. See that method.
        let objects = self.list_partition_objects().await?;

        for object in &objects {
            match self.filesystem.delete(&object.path).await {
                Ok(()) => {}
                // Already gone: either a retry of a partially-completed purge,
                // or a candidate the listing proposed that was never ours (a
                // sibling prefix bled in by a raw-prefix LIST — the canonical
                // path under OUR prefix simply holds nothing). Deleting nothing
                // is the correct outcome in both cases, and the contract
                // requires idempotence, so this must not fail the DROP.
                Err(err) if is_absent(&err) => {}
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "spill: delete segment '{}' during purge failed: {err}",
                        object.path
                    ));
                }
            }
        }

        // Only after the objects are gone: a crash midway leaves the remaining
        // objects still listed under the prefix, so a retry finds and removes
        // them. An empty prefix takes this path too (the contract requires
        // idempotence), and BOTH pieces of state are cleared on every path so an
        // orphan reaper calling purge standalone cannot leave a store whose
        // segments are gone while its tombstones survive.
        self.segments.write().clear();
        self.tombstones.clear();
        Ok(())
    }
}

/// Wall clock in nanoseconds, for the canonical dead-record predicate.
fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// `true` when a `list` error means "this path does not exist" rather than a real
/// failure.
///
/// How each backend reports a missing prefix, checked rather than assumed:
///
/// * **local** — `Io(ErrorKind::NotFound)`, straight out of `read_dir`. This is
///   the case that matters: without it, an ordinary DROP of a table that never
///   spilled would fail.
/// * **S3 / Azure / GCS** — no error at all. They stream a listing and yield
///   `Ok(vec![])` for a prefix with no keys, so this predicate is never consulted
///   (flat keyspace: there is no directory to be missing).
/// * **HDFS** — `Network("HDFS list error: 404 …")`. Deliberately **not** matched:
///   classifying it would mean string-matching a status code out of a message,
///   and for a *deletion* path the conservative failure is the right one. Treating
///   an unrecognised list error as absence would silently skip deleting objects
///   that are really there. The cost is that DROP of a never-spilled table on HDFS
///   reports a purge failure; HDFS is not in `SUPPORTED_SURFACE.adoc` and the
///   filesystem factory never constructs it, so nothing reaches this today.
///
/// Erring toward "real failure" is the safe direction here: a false *absence*
/// loses data silently, a false *failure* is loud and recorded.
fn is_absent(err: &proximadb_storage_filesystem_types::FilesystemError) -> bool {
    use proximadb_storage_filesystem_types::FilesystemError;
    match err {
        FilesystemError::NotFound(_) => true,
        FilesystemError::Io(io) => io.kind() == std::io::ErrorKind::NotFound,
        _ => false,
    }
}

/// One candidate segment object for a partition, as a path under that
/// partition's prefix.
///
/// The PATH is always directly under the prefix; the listing entry that proposed
/// it need not have been — a recursive listing can report an object nested
/// deeper, and its basename is indistinguishable from one of ours. That is safe
/// by construction rather than by filtering: see `list_partition_objects`. `seq` is `None` for a spill-shaped name carrying no parseable
/// sequence (e.g. the superseded nonce scheme).
struct PartitionObject {
    /// Canonical path, built the way `flush` builds it — never a listing-derived
    /// string.
    path: String,
    seq: Option<u64>,
}

/// Filename prefix and extension of a spill segment.
const SPILL_PREFIX: &str = "spill-";
const SPILL_SUFFIX: &str = ".parquet";

/// Width of the hex sequence in a segment name: fixed, so a lexicographic sort
/// of segment names equals their numeric order (`spill-10` after `spill-9`).
///
/// Note what this does and does not buy TODAY. Discovery sorts numerically on
/// the parsed sequence, so current ordering does not depend on the padding; what
/// the fixed width is load-bearing for is `parse_spill_seq`, which REJECTS any
/// other width, so an unpadded name becomes a hard error rather than a
/// mis-ordered segment. The padding is kept because it makes that rejection
/// meaningful and keeps the names correct for any future consumer that orders a
/// listing directly, which is cheap insurance — not because anything sorts
/// lexicographically now.
const SPILL_SEQ_HEX_WIDTH: usize = 16;

/// Name of the segment carrying sequence `seq`.
fn spill_segment_name(seq: u64) -> String {
    format!(
        "{SPILL_PREFIX}{seq:0width$x}{SPILL_SUFFIX}",
        width = SPILL_SEQ_HEX_WIDTH
    )
}

/// Sequence encoded in a segment's name, or `None` if the name is not one this
/// store could have written.
///
/// Strict where [`is_spill_segment`] is permissive, and the asymmetry is
/// deliberate — see that function. A caller that needs ORDER (discovery) must
/// treat `None` on a segment-shaped name as an error rather than skipping it;
/// a caller that needs only RECLAMATION (purge) is right to be permissive.
fn parse_spill_seq(name: &str) -> Option<u64> {
    let name = name.rsplit('/').next()?;
    let hex = name
        .strip_prefix(SPILL_PREFIX)?
        .strip_suffix(SPILL_SUFFIX)?;
    // Reject any other width: accepting a short name would admit a sequence
    // whose lexicographic order disagrees with its numeric order, which is the
    // one property the name exists to carry.
    if hex.len() != SPILL_SEQ_HEX_WIDTH {
        return None;
    }
    // Require the ALLOWED class, do not enumerate disallowed ones. An earlier
    // version rejected uppercase and stopped there, which did NOT deliver the
    // injectivity it claimed: `from_str_radix` also accepts a leading `+`, so
    // `spill-+00000000000000f.parquet` parsed to 15 — sixteen bytes, no
    // uppercase, both checks passed — exactly as
    // `spill-000000000000000f.parquet` does. Two objects claiming one sequence,
    // and a set holding both has no defined order. Worse for the ordering
    // property, `+` is 0x2B and sorts BEFORE `0` (0x30), so such a name sorts
    // ahead of sequence 0 while carrying 15.
    //
    // `spill_segment_name` emits exactly `[0-9a-f]`, so that is the whole
    // alphabet a name this store wrote can use; anything else is not ours.
    if !hex
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// `true` for a name that is a spill segment with one extra extension appended —
/// the shape a non-transparent wrapper filesystem produces.
///
/// `EncryptedFilesystem` appends its `encrypted_extension` (default `.enc`) to
/// every `read`/`write`/`write_if_absent`/`delete`/`exists` path, but its `list`
/// WAS a bare passthrough that did not un-mangle what it returned (fixed in
/// `fc6fefe32`; see the note below). So a segment
/// written as `spill-{seq}.parquet` is listed as `spill-{seq}.parquet.enc`.
///
/// Without this detector that name is simply not spill-shaped, and the
/// consequences are silent: discovery finds nothing, so every flushed row reads
/// as absent and `delete_record` records no tombstone; `resume` restarts at 0,
/// so the next flush collides with the underlying object and `write_if_absent`
/// fails permanently; and purge reclaims nothing while reporting success. The
/// wrapper is the layer at fault (TD-ENCFS-1), so this store refuses to serve
/// rather than pretend the partition is empty.
///
/// **That wrapper was fixed in `fc6fefe32`, so this detector is now redundant
/// defence-in-depth rather than a live mitigation.** It is kept deliberately: it
/// would still trip on a future wrapper that renames objects on write without
/// un-mangling `list`, which is a silent-emptiness failure this store cannot
/// otherwise distinguish from a genuinely empty partition.
///
/// Scope, stated so it is not mistaken for general: this detects ONE extra
/// dot-separated extension, which is the only reachable shape today
/// (`maybe_wrap_with_encryption` uses `EncryptedFilesystem::new`, i.e. the
/// hardcoded `.enc`). A two-dot extension (`.enc.v2`) or a dotless one (`enc`)
/// would evade it and restore the silent empty-partition failure.
/// `with_extension` has no callers, so neither is reachable — but this is
/// brittle to a one-line configuration change, which is a further reason the
/// fix belongs in the wrapper (TD-ENCFS-1), not here.
fn is_mangled_spill_segment(name: &str) -> bool {
    if !name.starts_with(SPILL_PREFIX) {
        return false;
    }
    match name.rfind('.') {
        Some(dot) => is_spill_segment(&name[..dot]),
        None => false,
    }
}

/// `true` for an object this store wrote — `spill-{seq}.parquet`.
///
/// Matched by prefix and extension rather than by parsing the name, and
/// deliberately narrow: the prefix belongs to this partition, but refusing to
/// delete anything that does not look like our own output means a path
/// misconfiguration cannot turn a DROP into a delete of someone else's objects.
///
/// Deliberately PERMISSIVE relative to [`parse_spill_seq`]: purge must reclaim
/// every object this store could have written, including one left by an older
/// naming scheme, or a DROP leaks it forever.
fn is_spill_segment(name: &str) -> bool {
    name.starts_with(SPILL_PREFIX) && name.ends_with(SPILL_SUFFIX)
}

#[async_trait]
impl RecordScan for SpillRecordStorage {
    async fn scan_records(&self, limit: usize) -> RecordStoreResult<Vec<ProximaRecord>> {
        Ok(self
            .merged_records()
            .await?
            .into_iter()
            .take(limit)
            .collect())
    }

    async fn scan_records_with_options(
        &self,
        options: RecordScanOptions,
    ) -> RecordStoreResult<Vec<ProximaRecord>> {
        let limit = options.limit.unwrap_or(usize::MAX);
        Ok(self
            .merged_records()
            .await?
            .into_iter()
            .filter(|record| options.matches_record(record))
            .take(limit)
            .collect())
    }

    async fn scan_records_filtered(
        &self,
        options: RecordScanOptions,
        predicate: Option<&RecordScanPredicate<'_>>,
    ) -> RecordStoreResult<Vec<ProximaRecord>> {
        let limit = options.limit.unwrap_or(usize::MAX);
        Ok(self
            .merged_records()
            .await?
            .into_iter()
            .filter(|record| options.matches_record(record) && predicate.is_none_or(|p| p(record)))
            .take(limit)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaTreeNode;

    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    /// Back the store with the REAL `LocalFileSystem` over a unique tempdir
    /// rather than a mock: the segment round trip is the thing under test, so it
    /// should go through a genuine write/read rather than a map that cannot fail
    /// the way a filesystem can.
    ///
    /// The tempdir is leaked deliberately (mandate #17c) — the store holds an
    /// `Arc<dyn FileSystem>` that outlives this helper's frame.
    async fn store(threshold: Option<usize>) -> SpillRecordStorage {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let fs = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        SpillRecordStorage::with_flush_threshold(Arc::new(fs), base, threshold)
    }

    /// Count segment objects directly on disk, independent of what the store
    /// believes it has — the only way to tell deletion from forgetting.
    fn segment_files_on_disk(base: &str) -> usize {
        std::fs::read_dir(base)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| is_spill_segment(&e.file_name().to_string_lossy()))
                    .count()
            })
            .unwrap_or(0)
    }

    /// DROP must DELETE the objects, not merely forget them.
    ///
    /// Asserted on the filesystem rather than through `segment_count()`: clearing
    /// the in-memory list is exactly the bug this guards against, so a test that
    /// trusted the store's own bookkeeping would pass while every object survived.
    #[tokio::test]
    async fn purge_deletes_the_objects_not_just_the_bookkeeping() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let fs = Arc::new(
            LocalFileSystem::new(LocalConfig::default())
                .await
                .expect("local filesystem"),
        );
        let s = SpillRecordStorage::with_flush_threshold(fs, base.clone(), Some(2));

        for i in 0..6 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }
        let before = segment_files_on_disk(&base);
        assert!(before > 0, "precondition: the store must have flushed");

        s.purge_durable_objects().await?;

        assert_eq!(
            segment_files_on_disk(&base),
            0,
            "purge must delete every segment object (had {before})"
        );
        assert_eq!(
            s.segment_count(),
            0,
            "purge must also clear the tracked list"
        );
        Ok(())
    }

    /// Purge must reclaim segments written by an EARLIER store instance at the
    /// same prefix — the orphans a restart or a `drop_partition_state` eviction
    /// leaves behind, which the current instance never knew about.
    ///
    /// This is why purge lists the prefix instead of walking `self.segments`.
    #[tokio::test]
    async fn purge_reclaims_orphans_from_a_previous_instance() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let fs = Arc::new(
            LocalFileSystem::new(LocalConfig::default())
                .await
                .expect("local filesystem"),
        );

        // Instance 1 flushes, then goes away without purging (restart / eviction).
        {
            let first = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), Some(2));
            for i in 0..4 {
                first
                    .upsert_record(record(&format!("a{i}"), "open"))
                    .await?;
            }
            assert!(first.segment_count() > 0, "precondition: first flushed");
        }
        let orphans = segment_files_on_disk(&base);
        assert!(orphans > 0, "precondition: orphaned objects exist");

        // Instance 2 at the same prefix has an EMPTY segment list.
        let second = SpillRecordStorage::with_flush_threshold(fs, base.clone(), Some(2));
        assert_eq!(
            second.segment_count(),
            0,
            "precondition: the new instance knows nothing of the orphans"
        );

        second.purge_durable_objects().await?;

        assert_eq!(
            segment_files_on_disk(&base),
            0,
            "purge must delete orphans it never tracked (had {orphans})"
        );
        Ok(())
    }

    /// The minimum is over RESIDENT rows only, and it ADVANCES as rows flush.
    ///
    /// This is the quantity ADR-094's truncation rule needs. Asserted as a
    /// minimum over an unordered set rather than a prefix, because that is what
    /// a `DashMap` keyed by oid can honestly report.
    #[tokio::test]
    async fn min_unflushed_lsn_tracks_resident_rows_and_advances_on_flush() -> Result<()> {
        let s = store(Some(3)).await;
        assert_eq!(s.min_unflushed_lsn().await?, None, "nothing resident");

        // LSNs deliberately out of insertion order: the answer must be the
        // minimum, not the first or last seen.
        s.upsert_record_at_lsn(record("o1", "open"), 70).await?;
        s.upsert_record_at_lsn(record("o2", "open"), 50).await?;
        assert_eq!(s.min_unflushed_lsn().await?, Some(50));

        // Third insert crosses the threshold -> flush -> all three become durable
        // in a segment, so none of their LSNs constrains the WAL any more.
        s.upsert_record_at_lsn(record("o3", "open"), 90).await?;
        assert!(s.segment_count() > 0, "precondition: flushed");
        assert_eq!(
            s.min_unflushed_lsn().await?,
            None,
            "flushed rows must stop pinning the WAL"
        );

        // A later write re-establishes a constraint at its own LSN.
        s.upsert_record_at_lsn(record("o4", "open"), 120).await?;
        assert_eq!(s.min_unflushed_lsn().await?, Some(120));
        Ok(())
    }

    /// The minimum is NOT the high-water mark — the distinction the TD's warning
    /// is about.
    ///
    /// A high-water scheme would stamp the segment with the largest LSN it had
    /// seen and let the WAL truncate below it, discarding the entry for a row
    /// that is still only resident. Here row `keep` sits at a LOWER LSN than an
    /// already-flushed row, so a high-water answer (90) and the correct answer
    /// (40) differ — and truncating at 90 would lose `keep`.
    #[tokio::test]
    async fn min_unflushed_lsn_is_not_the_high_water_mark() -> Result<()> {
        let s = store(Some(2)).await;
        s.upsert_record_at_lsn(record("a", "open"), 80).await?;
        s.upsert_record_at_lsn(record("b", "open"), 90).await?;
        assert!(s.segment_count() > 0, "precondition: a and b flushed");

        // Resident, and OLDER than everything already durable.
        s.upsert_record_at_lsn(record("keep", "open"), 40).await?;

        assert_eq!(
            s.min_unflushed_lsn().await?,
            Some(40),
            "the answer must be the resident minimum, not the 90 a high-water \
             scheme would report"
        );
        Ok(())
    }

    /// An untracked resident row makes the question unanswerable, and that must
    /// FAIL rather than return a minimum that excludes it (mandate #1).
    ///
    /// A row inserted through plain `upsert_record` carries no LSN. Reporting
    /// `Some(60)` here would invite a caller to truncate past the untracked
    /// row's WAL entry — the only durable copy of it.
    #[tokio::test]
    async fn min_unflushed_lsn_fails_closed_when_a_resident_row_is_untracked() -> Result<()> {
        let s = store(None).await;
        s.upsert_record_at_lsn(record("tracked", "open"), 60)
            .await?;
        s.upsert_record(record("untracked", "open")).await?;

        let err = s
            .min_unflushed_lsn()
            .await
            .expect_err("an untracked resident row must not yield a minimum");
        let msg = err.to_string();
        assert!(
            msg.contains("Do not truncate"),
            "the error must tell the caller what to do, got: {msg}"
        );
        Ok(())
    }

    /// An in-memory tombstone must make the truncation point UNANSWERABLE.
    ///
    /// This is the resurrection path (mandate #16a), and the reason a minimum
    /// over resident rows alone is not enough: the row is durable in a segment,
    /// so nothing is resident and a naive answer is "no constraint" — but the
    /// only durable evidence of the DELETE is its WAL entry. Truncating past it
    /// brings the row back on restart.
    #[tokio::test]
    async fn min_unflushed_lsn_fails_closed_while_a_tombstone_is_memory_only() -> Result<()> {
        let s = store(Some(2)).await;
        s.upsert_record_at_lsn(record("x", "open"), 10).await?;
        s.upsert_record_at_lsn(record("y", "open"), 11).await?;
        assert!(s.segment_count() > 0, "precondition: flushed to a segment");
        assert_eq!(
            s.min_unflushed_lsn().await?,
            None,
            "precondition: nothing resident once flushed"
        );

        // Delete a FLUSHED row: suppressed by an in-memory tombstone only.
        s.delete_record(&RecordKey::new("x".to_string())).await?;

        let err = s
            .min_unflushed_lsn()
            .await
            .expect_err("a memory-only tombstone must not yield a minimum");
        let msg = err.to_string();
        assert!(
            msg.contains("tombstone") && msg.contains("Do not truncate"),
            "the error must name the cause and the action, got: {msg}"
        );
        Ok(())
    }

    /// Deleting a resident row releases its LSN — it is no longer durable-pending.
    #[tokio::test]
    async fn delete_releases_the_lsn_constraint() -> Result<()> {
        let s = store(None).await;
        s.upsert_record_at_lsn(record("gone", "open"), 30).await?;
        s.upsert_record_at_lsn(record("stays", "open"), 55).await?;
        assert_eq!(s.min_unflushed_lsn().await?, Some(30));

        s.delete_record(&RecordKey::new("gone".to_string())).await?;
        assert_eq!(
            s.min_unflushed_lsn().await?,
            Some(55),
            "a deleted row must stop pinning the WAL"
        );
        Ok(())
    }

    /// Purge is idempotent: a partition that never flushed, and a re-drop, are
    /// both success. The contract says so, and `drop_table_records` propagates
    /// errors — so a spurious failure here would make DROP TABLE fail outright.
    #[tokio::test]
    async fn purge_is_idempotent_and_tolerates_a_missing_prefix() -> Result<()> {
        let s = store(Some(2)).await;
        // Never flushed: the prefix may not even exist.
        s.purge_durable_objects().await?;
        s.purge_durable_objects().await?;

        for i in 0..4 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }
        s.purge_durable_objects().await?;
        s.purge_durable_objects().await?;
        Ok(())
    }

    /// Mandate #16a: a record that arrives from a segment already carrying a
    /// tombstone `valid_to_ns` must NOT read back as live.
    ///
    /// The store's own suppression is an in-memory set, so it only covers rows
    /// this process deleted. This drives the other way in: a record that is
    /// already dead when it is written, so nothing is in `tombstones` and only
    /// the canonical predicate can catch it. That is the shape a durable
    /// tombstone will take once segments are discoverable (TD-USUB-1), and the
    /// shape a TTL'd row would take today on any path that sets `valid_to_ns`.
    ///
    /// Asserted through BOTH read surfaces, because they filter independently.
    #[tokio::test]
    async fn a_dead_record_from_a_segment_does_not_read_back_live() -> Result<()> {
        let s = store(Some(2)).await;

        // `live` is an ordinary row; `gone` is already a tombstone when written.
        s.upsert_record(record("live", "open")).await?;
        let mut dead = record("gone", "open");
        dead.valid_to_ns = Some(0);
        s.upsert_record(dead).await?;

        // Threshold 2 ⇒ both were flushed into a segment, and the in-memory
        // tombstone set is empty because nothing was deleted through this store.
        assert!(s.segment_count() > 0, "precondition: flushed to a segment");
        assert_eq!(
            s.resident_len(),
            0,
            "precondition: nothing resident, so reads come from the segment"
        );

        assert!(
            s.get_record(&RecordKey::new("gone".to_string()))
                .await?
                .is_none(),
            "a dead record from a segment must not be returned by get_record"
        );
        let scanned = s.scan_records(usize::MAX).await?;
        assert!(
            !scanned.iter().any(|r| r.oid == "gone"),
            "a dead record from a segment must not appear in a scan: {:?}",
            scanned.iter().map(|r| &r.oid).collect::<Vec<_>>()
        );
        assert!(
            scanned.iter().any(|r| r.oid == "live"),
            "the live row must still be readable"
        );
        Ok(())
    }

    /// TD-USUB-13: a delete that lands mid-flush must not resurrect the row.
    ///
    /// Sequence, made deterministic by `BarrierFs`:
    ///
    ///   1. two rows cross the flush threshold, so `flush()` runs and parks
    ///      inside the object write — AFTER snapshotting the memtable, BEFORE
    ///      pushing to `segments`;
    ///   2. a delete for one of those rows runs in that window;
    ///   3. the write is released and the flush completes, publishing a segment
    ///      built from the PRE-delete snapshot.
    ///
    /// Without serialization the delete saw an empty segment list, recorded no
    /// tombstone, and reported success — and the row came back live from the
    /// segment flush then published. No WAL truncation involved (mandate #16a).
    ///
    /// Asserts the OBSERVABLE property — the row is gone from both `get_record`
    /// and a scan — not the presence of a tombstone, so a different fix (an
    /// in-flight oid set, say) still satisfies it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_during_a_flush_does_not_resurrect_the_row() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);

        let reached = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let fs = Arc::new(BarrierFs {
            inner: LocalFileSystem::new(LocalConfig::default())
                .await
                .expect("local filesystem"),
            reached: reached.clone(),
            release: release.clone(),
            armed: std::sync::atomic::AtomicBool::new(true),
        });

        // Threshold 2: the second insert triggers the flush that parks.
        let store = Arc::new(SpillRecordStorage::with_flush_threshold(fs, base, Some(2)));

        let writer = {
            let store = store.clone();
            tokio::spawn(async move {
                store.upsert_record(record("keep", "open")).await?;
                // This one crosses the threshold and parks inside the write.
                store.upsert_record(record("victim", "open")).await?;
                Ok::<(), anyhow::Error>(())
            })
        };

        // Wait until the flush is genuinely parked mid-write.
        reached.notified().await;

        // The delete MUST be a separate task. Awaiting it here would deadlock
        // once the fix is in: `delete_record` blocks on the same guard the
        // parked `flush` holds, so this task would never reach the release
        // below. That deadlock is the fix working — "a delete during a flush"
        // is exactly what serialization makes impossible — so the test has to
        // let the delete queue rather than wait for it.
        let deleter = {
            let store = store.clone();
            tokio::spawn(async move {
                store
                    .delete_record(&RecordKey::new("victim".to_string()))
                    .await
            })
        };

        // Give an UNSERIALIZED delete the chance to slip into the window. With
        // the fix it is parked on the guard and this elapses; without the fix it
        // completes here, inside the flush, which is the defect being pinned.
        // Either way the assertions below are the same, so this bounds nothing
        // but how long the window stays open.
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;

        release.notify_one();
        writer.await.expect("writer task").expect("writer result");
        let deleted = deleter.await.expect("deleter task")?;
        assert!(deleted, "the delete must report that a live row went away");

        // The row must be gone by every read path.
        assert!(
            store
                .get_record(&RecordKey::new("victim".to_string()))
                .await?
                .is_none(),
            "deleted row resurrected via get_record from the segment flush published"
        );
        let scanned = store.scan_records(usize::MAX).await?;
        assert!(
            !scanned.iter().any(|r| r.oid == "victim"),
            "deleted row resurrected in a scan: {:?}",
            scanned.iter().map(|r| &r.oid).collect::<Vec<_>>()
        );
        assert!(
            scanned.iter().any(|r| r.oid == "keep"),
            "the surviving row must still be readable"
        );
        Ok(())
    }

    /// A filesystem that pauses inside `write_if_absent` until released.
    ///
    /// This is the only way to open TD-USUB-13's window deterministically: the
    /// race needs a delete to land between `flush`'s memtable snapshot and its
    /// `segments.push`, and the gap between them is exactly one object write.
    /// Timing-based attempts (a "slow enough" real write) would be flaky, and a
    /// test-only hook in the production path would be worse.
    ///
    /// Deliberately local and lock-free of process-global state: no env arming
    /// and no `OnceLock`, because a process-lifetime gate is what made the
    /// TD-USUB-6 benchmark measure the same mode twice. The barrier is passed in
    /// by `Arc`, so each test owns its own.
    #[derive(Debug)]
    struct BarrierFs {
        inner: LocalFileSystem,
        /// Signalled by the FS when a write has arrived and is parked.
        reached: Arc<tokio::sync::Notify>,
        /// Awaited by the FS; signalled by the test to let the write proceed.
        release: Arc<tokio::sync::Notify>,
        /// Only the FIRST write parks; later ones pass through, so `flush`'s
        /// eviction and any follow-up work cannot deadlock.
        armed: std::sync::atomic::AtomicBool,
    }

    #[async_trait]
    impl FileSystem for BarrierFs {
        async fn write_if_absent(
            &self,
            path: &str,
            data: &[u8],
            options: Option<proximadb_storage_filesystem_types::FileOptions>,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            if self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
                self.reached.notify_one();
                self.release.notified().await;
            }
            self.inner.write_if_absent(path, data, options).await
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
            self.inner.read(path).await
        }
        async fn write(
            &self,
            path: &str,
            data: &[u8],
            options: Option<proximadb_storage_filesystem_types::FileOptions>,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.write(path, data, options).await
        }
        async fn append(
            &self,
            path: &str,
            data: &[u8],
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.append(path, data).await
        }
        async fn delete(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.delete(path).await
        }
        async fn exists(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<bool> {
            self.inner.exists(path).await
        }
        async fn metadata(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<
            proximadb_storage_filesystem_types::FsFileMetadata,
        > {
            self.inner.metadata(path).await
        }
        async fn list(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<
            Vec<proximadb_storage_filesystem_types::DirEntry>,
        > {
            self.inner.list(path).await
        }
        async fn create_dir(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.create_dir(path).await
        }
        async fn create_dir_all(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.create_dir_all(path).await
        }
        async fn copy(
            &self,
            from: &str,
            to: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.copy(from, to).await
        }
        async fn move_file(
            &self,
            from: &str,
            to: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.move_file(from, to).await
        }
        fn filesystem_type(&self) -> &'static str {
            self.inner.filesystem_type()
        }
        async fn sync(&self) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.sync().await
        }
        async fn open_file(
            &self,
            path: &str,
            create: bool,
        ) -> proximadb_storage_filesystem_types::FsResult<
            Box<dyn proximadb_storage_filesystem_types::FilesystemFile>,
        > {
            self.inner.open_file(path, create).await
        }
    }

    /// A filesystem that reproduces the OBJECT-STORE listing contract on top of
    /// the local backend, so the hazard it creates is testable without GCS.
    ///
    /// Local `list` is `read_dir`: directory-scoped and non-recursive, so it can
    /// never return a foreign object. The object stores list by KEY PREFIX and
    /// recursively, and GCS passes the prefix through raw with no trailing
    /// delimiter — so a partition at `…/order` also sees `…/orders/…`. This
    /// double injects exactly those entries, shaped the way the real backends
    /// shape them: `name` is a bare basename (`key.rsplit('/').next()`) and
    /// `url` is the full location.
    #[derive(Debug)]
    struct ListInjectFs {
        inner: LocalFileSystem,
        /// Entries returned ONLY when the listed path lacks a trailing
        /// delimiter — modelling GCS's raw prefix, where `…/order` also matches
        /// `…/orders/…`. Scoping the listing is what makes these disappear, so a
        /// test that relies on them is testing that we scope.
        inject_unscoped: Vec<proximadb_storage_filesystem_types::DirEntry>,
        /// Entries returned regardless — modelling a RECURSIVE listing, which
        /// returns keys nested below the prefix even when correctly scoped.
        inject_always: Vec<proximadb_storage_filesystem_types::DirEntry>,
    }

    impl ListInjectFs {
        fn entry(url: &str) -> proximadb_storage_filesystem_types::DirEntry {
            proximadb_storage_filesystem_types::DirEntry {
                name: url.rsplit('/').next().unwrap_or(url).to_string(),
                url: url.to_string(),
                metadata: proximadb_storage_filesystem_types::FsFileMetadata::default(),
            }
        }
    }

    #[async_trait]
    impl FileSystem for ListInjectFs {
        async fn list(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<
            Vec<proximadb_storage_filesystem_types::DirEntry>,
        > {
            let mut entries = self.inner.list(path).await?;
            entries.extend(self.inject_always.iter().cloned());
            if !path.ends_with('/') {
                entries.extend(self.inject_unscoped.iter().cloned());
            }
            Ok(entries)
        }
        async fn write_if_absent(
            &self,
            path: &str,
            data: &[u8],
            options: Option<proximadb_storage_filesystem_types::FileOptions>,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.write_if_absent(path, data, options).await
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
            self.inner.read(path).await
        }
        async fn write(
            &self,
            path: &str,
            data: &[u8],
            options: Option<proximadb_storage_filesystem_types::FileOptions>,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.write(path, data, options).await
        }
        async fn append(
            &self,
            path: &str,
            data: &[u8],
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.append(path, data).await
        }
        async fn delete(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.delete(path).await
        }
        async fn exists(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<bool> {
            self.inner.exists(path).await
        }
        async fn metadata(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<
            proximadb_storage_filesystem_types::FsFileMetadata,
        > {
            self.inner.metadata(path).await
        }
        async fn create_dir(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.create_dir(path).await
        }
        async fn create_dir_all(
            &self,
            path: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.create_dir_all(path).await
        }
        async fn copy(
            &self,
            from: &str,
            to: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.copy(from, to).await
        }
        async fn move_file(
            &self,
            from: &str,
            to: &str,
        ) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.move_file(from, to).await
        }
        fn filesystem_type(&self) -> &'static str {
            self.inner.filesystem_type()
        }
        async fn sync(&self) -> proximadb_storage_filesystem_types::FsResult<()> {
            self.inner.sync().await
        }
        async fn open_file(
            &self,
            path: &str,
            create: bool,
        ) -> proximadb_storage_filesystem_types::FsResult<
            Box<dyn proximadb_storage_filesystem_types::FilesystemFile>,
        > {
            self.inner.open_file(path, create).await
        }
    }

    fn record(oid: &str, status: &str) -> ProximaRecord {
        let mut r = ProximaRecord {
            oid: oid.to_string(),
            tenant_id: "tenant-a".to_string(),
            ..Default::default()
        };
        r.props.insert(
            "status".to_string(),
            ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                status.to_string(),
            )),
        );
        r
    }

    /// Default (no threshold) never writes a segment — enabling nothing changes
    /// nothing, so the shipped behaviour is the plain memtable's.
    #[tokio::test]
    async fn default_never_flushes() -> Result<()> {
        let s = store(None).await;
        for i in 0..50 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }
        assert_eq!(s.segment_count(), 0);
        assert_eq!(s.resident_len(), 50);
        assert_eq!(s.scan_records(usize::MAX).await?.len(), 50);
        Ok(())
    }

    /// TWO stores at the SAME base path must not destroy each other's segments.
    ///
    /// This is the shape the partition factory now creates: `base_path` is a pure
    /// function of `(tenant, collection)`, so a restart — or a
    /// `drop_partition_state` eviction followed by the next access — builds a
    /// second store over the first one's objects with its counter back at 0.
    /// A store that did not resume its counter would re-issue sequence 0, and
    /// the first store's durable rows would be gone — silently, if the write
    /// were an unconditional overwrite. Two things now prevent that: the resumed
    /// counter makes the name fresh, and `write_if_absent` fails loudly if that
    /// reasoning is ever wrong. (This doc previously described the superseded
    /// `spill-{uuid}-{seq:010}` naming and a `write` with `options: None`;
    /// neither exists any more.)
    ///
    /// Asserts the observable property (both stores' segments survive and stay
    /// readable), not the file-naming scheme, so a different uniqueness strategy
    /// still passes.
    #[tokio::test]
    async fn two_stores_sharing_a_base_path_do_not_overwrite_segments() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let shared = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let fs = Arc::new(
            LocalFileSystem::new(LocalConfig::default())
                .await
                .expect("local filesystem"),
        );

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), shared.clone(), Some(2));
        for i in 0..4 {
            first
                .upsert_record(record(&format!("a{i}"), "open"))
                .await?;
        }
        assert!(first.segment_count() > 0, "first store must have flushed");
        let first_rows = first.scan_records(usize::MAX).await?.len();

        // A SECOND store over the same path, counter back at 0.
        let second = SpillRecordStorage::with_flush_threshold(fs, shared, Some(2));
        for i in 0..4 {
            second
                .upsert_record(record(&format!("b{i}"), "open"))
                .await?;
        }
        assert!(second.segment_count() > 0, "second store must have flushed");

        // The first store's rows must still be readable: its segments were not
        // clobbered by the second store's writes.
        assert_eq!(
            first.scan_records(usize::MAX).await?.len(),
            first_rows,
            "the second store overwrote the first store's segments"
        );
        Ok(())
    }

    /// A partition's base path does NOT exist before its first flush, and a
    /// local-filesystem backend will not create it on write.
    ///
    /// Every other test here uses `store()`, which hands over a `tempdir()` the
    /// harness already created — so they exercise the flush mechanism while
    /// never exercising the path a real caller supplies. This one points the
    /// store at a nested path that does not exist, which is what the partition
    /// factory produces (`…/{tenant}/{collection}`), and is the shape that
    /// failed when slice 2a was first wired behind it.
    #[tokio::test]
    async fn flush_creates_a_base_path_that_does_not_exist_yet() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = format!(
            "{}/tenant-a/orders",
            dir.path().to_string_lossy().trim_end_matches('/')
        );
        std::mem::forget(dir);
        assert!(
            !std::path::Path::new(&nested).exists(),
            "precondition: the partition path must NOT exist yet"
        );

        let fs = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        let s = SpillRecordStorage::with_flush_threshold(Arc::new(fs), nested, Some(2));

        for i in 0..6 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }

        assert!(s.segment_count() > 0, "the store must have flushed");
        assert_eq!(
            s.scan_records(usize::MAX).await?.len(),
            6,
            "every row must survive a flush into a freshly created directory"
        );
        Ok(())
    }

    /// Flushing bounds RESIDENT memory while every row stays readable — the
    /// actual point of the slice.
    #[tokio::test]
    async fn flush_bounds_residency_without_losing_rows() -> Result<()> {
        let s = store(Some(10)).await;
        for i in 0..35 {
            s.upsert_record(record(&format!("o{i:02}"), "open")).await?;
        }
        assert!(
            s.segment_count() >= 3,
            "expected flushes, got {}",
            s.segment_count()
        );
        assert!(
            s.resident_len() < 10,
            "resident set must stay under the threshold, got {}",
            s.resident_len()
        );

        let all = s.scan_records(usize::MAX).await?;
        assert_eq!(
            all.len(),
            35,
            "every row must still be visible after flushing"
        );
        for i in 0..35 {
            let oid = format!("o{i:02}");
            assert!(
                s.get_record(&RecordKey::new(oid.clone())).await?.is_some(),
                "flushed row {oid} must still be fetchable"
            );
        }
        Ok(())
    }

    /// **The resurrection test.** Deleting a row that has already been flushed
    /// must keep it deleted — the segment is immutable, so only a tombstone can
    /// suppress it. This is the failure TD-USUB-11's `valid_to_ns` round trip and
    /// this tombstone set exist to prevent.
    #[tokio::test]
    async fn delete_after_flush_does_not_resurrect() -> Result<()> {
        let s = store(Some(4)).await;
        for i in 0..8 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }
        assert!(
            s.segment_count() >= 1,
            "row must actually have been flushed"
        );

        let key = RecordKey::new("o1".to_string());
        assert!(
            s.get_record(&key).await?.is_some(),
            "precondition: o1 is live"
        );
        assert!(
            s.delete_record(&key).await?,
            "deleting a flushed row reports true"
        );

        assert!(
            s.get_record(&key).await?.is_none(),
            "a deleted flushed row must NOT come back from the segment"
        );
        let all = s.scan_records(usize::MAX).await?;
        assert!(
            !all.iter().any(|r| r.oid == "o1"),
            "scan must not resurrect the deleted row"
        );
        assert_eq!(all.len(), 7);

        // Deleting again reports false — there is no longer a live row.
        assert!(!s.delete_record(&key).await?);
        Ok(())
    }

    /// Re-inserting a deleted oid must clear the tombstone, or the new row would
    /// stay invisible behind the suppression of the old one.
    #[tokio::test]
    async fn reinsert_after_delete_clears_the_tombstone() -> Result<()> {
        let s = store(Some(4)).await;
        for i in 0..8 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }
        let key = RecordKey::new("o1".to_string());
        assert!(s.delete_record(&key).await?);
        assert!(s.get_record(&key).await?.is_none());

        s.upsert_record(record("o1", "reopened")).await?;
        let back = s
            .get_record(&key)
            .await?
            .expect("re-inserted row must be visible");
        assert!(matches!(
            back.props.get("status"),
            Some(ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(s))) if s == "reopened"
        ));
        Ok(())
    }

    /// The memtable is strictly newer than any segment, so an update after a
    /// flush must shadow the stale segment copy — never merge behind it.
    #[tokio::test]
    async fn update_after_flush_shadows_the_segment_copy() -> Result<()> {
        let s = store(Some(4)).await;
        for i in 0..4 {
            s.upsert_record(record(&format!("o{i}"), "open")).await?;
        }
        assert_eq!(s.segment_count(), 1);
        assert_eq!(s.resident_len(), 0);

        s.upsert_record(record("o0", "closed")).await?;
        let got = s
            .get_record(&RecordKey::new("o0".to_string()))
            .await?
            .unwrap();
        assert!(
            matches!(
                got.props.get("status"),
                Some(ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(v))) if v == "closed"
            ),
            "the newer resident value must win over the flushed one"
        );

        // And exactly once in a scan — no duplicate from the segment.
        let all = s.scan_records(usize::MAX).await?;
        assert_eq!(all.iter().filter(|r| r.oid == "o0").count(), 1);
        assert_eq!(all.len(), 4);
        Ok(())
    }

    /// **The UNIQUE/PK trap.** `ensure_unique_index_built` builds duplicate-key
    /// enforcement by SCANNING the partition store. If a flushed row vanished
    /// from `scan_records`, duplicate detection would silently stop working — so
    /// the scan must return flushed rows, and this asserts it at the exact shape
    /// the index builder relies on.
    #[tokio::test]
    async fn scan_returns_flushed_rows_so_unique_enforcement_still_sees_them() -> Result<()> {
        let s = store(Some(5)).await;
        for i in 0..20 {
            s.upsert_record(record(&format!("o{i:02}"), "open")).await?;
        }
        assert!(s.segment_count() >= 3);
        assert!(s.resident_len() < 5, "most rows are no longer resident");

        // The index builder's shape: one unbounded scan must yield every oid.
        let scanned = s.scan_records(usize::MAX).await?;
        let mut oids: Vec<&str> = scanned.iter().map(|r| r.oid.as_str()).collect();
        oids.sort_unstable();
        let expected: Vec<String> = (0..20).map(|i| format!("o{i:02}")).collect();
        assert_eq!(
            oids,
            expected.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "a scan that misses flushed rows would silently break UNIQUE/PK enforcement"
        );
        Ok(())
    }

    /// Identity and tenant survive the flush round trip — the fidelity TD-USUB-11
    /// added the system columns for, exercised end-to-end through a real segment.
    #[tokio::test]
    async fn flushed_rows_keep_identity_and_tenant() -> Result<()> {
        let s = store(Some(2)).await;
        s.upsert_record(record("order-7", "open")).await?;
        s.upsert_record(record("order-8", "open")).await?;
        assert_eq!(s.segment_count(), 1);

        let got = s
            .get_record(&RecordKey::new("order-7".to_string()))
            .await?
            .expect("flushed row must be fetchable");
        assert_eq!(
            got.oid, "order-7",
            "oid must survive the segment round trip"
        );
        assert_eq!(got.tenant_id, "tenant-a", "tenant must survive it too");
        Ok(())
    }

    /// An invalid threshold reads as unset, never as "flush on every write".
    #[test]
    fn invalid_threshold_reads_as_never_flush() {
        for bad in ["0", "-1", "abc", ""] {
            // SAFETY: single-threaded unit test mutating process env, cleared below.
            unsafe { std::env::set_var(SPILL_MAX_RESIDENT_ENV, bad) };
            assert_eq!(
                configured_flush_threshold(),
                None,
                "invalid threshold {bad:?}"
            );
        }
        unsafe { std::env::set_var(SPILL_MAX_RESIDENT_ENV, "256") };
        assert_eq!(configured_flush_threshold(), Some(256));
        unsafe { std::env::remove_var(SPILL_MAX_RESIDENT_ENV) };
        assert_eq!(configured_flush_threshold(), None);
    }
    /// Pins the NAME FORMAT: a lexicographic sort of segment names equals their
    /// numeric order.
    ///
    /// Note what this does not claim. Discovery sorts numerically on the parsed
    /// sequence, so today's ordering does not depend on the padding — an earlier
    /// version of this comment said it did, which was false of the shipped code.
    /// The fixed width is load-bearing for `parse_spill_seq`, which rejects any
    /// other width, and this test keeps the names correct for any future
    /// consumer that orders a listing directly. It fails if
    /// `spill_segment_name` stops padding.
    #[test]
    fn segment_names_sort_in_sequence_order() {
        let seqs: Vec<u64> = (0..40).chain([255, 256, 4095, 4096, u64::MAX]).collect();
        let names: Vec<String> = seqs.iter().map(|q| spill_segment_name(*q)).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(
            sorted, names,
            "lexicographic order of segment names must equal sequence order"
        );
    }

    /// `parse_spill_seq` is the strict counterpart of the permissive
    /// `is_spill_segment`, and the asymmetry is the point: purge must reclaim the
    /// legacy nonce-shaped name, while discovery must refuse to assign it an
    /// order it does not carry.
    #[test]
    fn segment_names_round_trip_and_reject_non_canonical() {
        for seq in [0u64, 1, 9, 10, 16, 255, 4096, u64::MAX] {
            let name = spill_segment_name(seq);
            assert_eq!(parse_spill_seq(&name), Some(seq), "{name} must round-trip");
            assert!(
                is_spill_segment(&name),
                "{name} must match the purge filter"
            );
            assert_eq!(
                parse_spill_seq(&format!("tenant/coll/{name}")),
                Some(seq),
                "a full path must parse by its final component"
            );
        }

        for bad in [
            "spill-9.parquet",                 // unpadded
            "spill-000000000000000.parquet",   // 15 digits
            "spill-00000000000000000.parquet", // 17 digits
            "spill-.parquet",                  // no sequence
            "spill-zzzzzzzzzzzzzzzz.parquet",  // not hex
            "other-0000000000000001.parquet",  // not our prefix
            "spill-0000000000000001.txt",      // not our extension
            "spill-00000000000000FF.parquet",  // uppercase: not injective
            "spill-00000000000000Ff.parquet",  // mixed case
            // `from_str_radix` accepts a leading sign, so a name outside the
            // `[0-9a-f]` alphabet could claim a sequence another name already
            // owns. `+` also sorts BEFORE `0`, so it breaks the ordering
            // property too. An earlier guard rejected uppercase and stopped
            // there, which did not deliver the injectivity it claimed.
            "spill-+00000000000000f.parquet", // leading plus: parses to 15
            "spill-+000000000000000.parquet", // leading plus: parses to 0
            "spill--00000000000000f.parquet", // leading minus
            "spill- 00000000000000f.parquet", // leading space
            "spill-0x0000000000000f.parquet", // radix prefix
            "spill-0000000000000_0f.parquet", // underscore separator
        ] {
            assert_eq!(parse_spill_seq(bad), None, "{bad} must not parse");
        }

        // The property those rejections exist for: no two accepted names may
        // claim one sequence. Checked directly, because an earlier guard
        // satisfied its own tests while failing this.
        for seq in [0u64, 15, 255, 4096, u64::MAX] {
            let canonical = spill_segment_name(seq);
            for candidate in [
                format!("spill-+{:015x}.parquet", seq),
                format!("spill-{:015X}.parquet", seq),
                format!("spill-{:016X}.parquet", seq),
            ] {
                if candidate == canonical {
                    continue;
                }
                assert_ne!(
                    parse_spill_seq(&candidate),
                    Some(seq),
                    "{candidate} must not claim the sequence {canonical} owns"
                );
            }
        }

        // The legacy nonce shape: still reclaimable by purge, never orderable.
        let legacy = "spill-deadbeefcafe-0000000001.parquet";
        assert!(
            is_spill_segment(legacy),
            "purge must still reclaim a legacy-named segment or DROP leaks it"
        );
        assert_eq!(
            parse_spill_seq(legacy),
            None,
            "a nonce-named segment carries no recoverable order"
        );
    }

    /// Build a filesystem over a fresh leaked tempdir, returning both.
    async fn fs_and_base() -> (Arc<dyn FileSystem>, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let fs = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        (Arc::new(fs) as Arc<dyn FileSystem>, base)
    }

    fn status_is(record: &ProximaRecord, expected: &str) -> bool {
        matches!(
            record.props.get("status"),
            Some(ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(v)))
                if v == expected
        )
    }

    /// A store built over a prefix an earlier instance wrote serves that
    /// instance's rows, and resolves a key to the value in the NEWEST segment.
    ///
    /// Asserting the newest value — not merely the row count — is what makes
    /// this a test of recovered ORDER rather than recovered membership. Reverse
    /// the sort and the row count still passes while `k` reads `old`.
    #[tokio::test]
    async fn discovery_recovers_segment_set_and_order() -> Result<()> {
        let (fs, base) = fs_and_base().await;

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        first.upsert_record(record("k", "old")).await?;
        first.upsert_record(record("p", "keep")).await?;
        first.flush().await?;
        first.upsert_record(record("k", "new")).await?;
        first.flush().await?;
        assert_eq!(first.segment_count(), 2, "two segments are needed to order");
        drop(first);

        // A second instance with nothing in memory: everything it serves came
        // from discovery.
        let second = SpillRecordStorage::with_flush_threshold(fs, base, None);
        let got = second
            .get_record(&RecordKey::new("k".to_string()))
            .await?
            .expect("a flushed row must be visible to a new instance");
        assert!(
            status_is(&got, "new"),
            "get_record must resolve to the newest segment's value"
        );

        let all = second.scan_records(usize::MAX).await?;
        assert_eq!(all.len(), 2, "the merge must dedup `k` across segments");
        let merged_k = all
            .iter()
            .find(|r| r.oid == "k")
            .expect("`k` must survive the merge");
        assert!(
            status_is(merged_k, "new"),
            "the merge must take the newest segment's value for `k`"
        );
        Ok(())
    }

    /// Discovery resumes `next_segment` past every durable segment, so a second
    /// instance's first flush cannot reuse a live name.
    ///
    /// This is the guarantee the removed per-instance UUID nonce used to buy.
    /// Break the resume and this fails either way: `write_if_absent` rejects the
    /// duplicate name (loud), or — if that backstop were also removed — the
    /// first instance's rows vanish from the third instance's merge (silent).
    #[tokio::test]
    async fn discovery_resumes_the_sequence_across_instances() -> Result<()> {
        let (fs, base) = fs_and_base().await;

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        first.upsert_record(record("a0", "open")).await?;
        first.upsert_record(record("a1", "open")).await?;
        first.flush().await?;
        drop(first);

        let second = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        second.upsert_record(record("b0", "open")).await?;
        second.flush().await?;
        assert_eq!(
            second.segment_count(),
            2,
            "the second instance must have discovered the first's segment \
             alongside its own"
        );
        drop(second);

        let third = SpillRecordStorage::with_flush_threshold(fs, base.clone(), None);
        let all = third.scan_records(usize::MAX).await?;
        let mut oids: Vec<String> = all.into_iter().map(|r| r.oid).collect();
        oids.sort();
        assert_eq!(
            oids,
            vec!["a0".to_string(), "a1".to_string(), "b0".to_string()],
            "no instance's segment may be clobbered by a later one"
        );
        assert_eq!(
            segment_files_on_disk(&base),
            2,
            "two distinct objects must exist on disk"
        );
        Ok(())
    }

    /// A segment-shaped object whose name carries no sequence must make the
    /// partition FAIL rather than serve the rest of the set.
    ///
    /// Skipping it would be silently wrong twice over: rows held only in that
    /// segment read as absent, and `delete_record` would see a shorter segment
    /// list and could skip a tombstone it needs. Order cannot be established
    /// over a set with an unplaceable member, so the store refuses to serve.
    #[tokio::test]
    async fn discovery_fails_closed_on_an_unrecognized_segment_name() -> Result<()> {
        let (fs, base) = fs_and_base().await;

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        first.upsert_record(record("k", "old")).await?;
        first.flush().await?;
        drop(first);

        // An object under our prefix that `is_spill_segment` claims but
        // `parse_spill_seq` cannot place — e.g. one written by the superseded
        // nonce scheme.
        std::fs::write(
            format!(
                "{}/spill-deadbeefcafe-0000000001.parquet",
                base.trim_end_matches('/')
            ),
            b"not a parquet file",
        )
        .expect("write unparseable segment");

        let second = SpillRecordStorage::with_flush_threshold(fs, base, None);
        let err = second
            .scan_records(usize::MAX)
            .await
            .expect_err("an unplaceable segment must fail the read, not be skipped");
        let msg = err.to_string();
        assert!(
            msg.contains("unrecognized name"),
            "the error must name the cause; got: {msg}"
        );
        Ok(())
    }

    /// The resurrection guard for the `delete_record` call site.
    ///
    /// `delete_record` decides whether to record a tombstone from
    /// `!segments.is_empty()`. A delete that is a new instance's FIRST operation
    /// would, without discovery ahead of it, see an empty segment list, skip the
    /// tombstone, and then a later read would discover the flushed copy and
    /// serve the deleted row. Remove `ensure_discovered` from `delete_record`
    /// and this test fails.
    #[tokio::test]
    async fn a_delete_as_the_first_touch_still_suppresses_a_flushed_row() -> Result<()> {
        let (fs, base) = fs_and_base().await;

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        first.upsert_record(record("k", "old")).await?;
        first.upsert_record(record("p", "keep")).await?;
        first.flush().await?;
        drop(first);

        let second = SpillRecordStorage::with_flush_threshold(fs, base, None);
        // FIRST operation on this instance is the delete — nothing has read, so
        // nothing else could have populated the segment list.
        //
        // The return value is checked LAST, deliberately. Asserting it here
        // would short-circuit the run on a weaker symptom ("reported it removed
        // nothing") and the test would never reach the claim it is named for.
        let reported_removal = second
            .delete_record(&RecordKey::new("k".to_string()))
            .await?;

        assert!(
            second
                .get_record(&RecordKey::new("k".to_string()))
                .await?
                .is_none(),
            "the deleted row must not come back from the segment"
        );
        let survivors: Vec<String> = second
            .scan_records(usize::MAX)
            .await?
            .into_iter()
            .map(|r| r.oid)
            .collect();
        assert_eq!(
            survivors,
            vec!["p".to_string()],
            "only the untouched row may survive the merge"
        );
        assert!(
            reported_removal,
            "deleting a flushed row must also report that it removed something"
        );
        Ok(())
    }
    /// A sequence read from a FILENAME must not overflow when resumed.
    ///
    /// `spill-ffffffffffffffff.parquet` is a well-formed name carrying
    /// `u64::MAX`, so a naive `seq + 1` panics in debug and WRAPS in release —
    /// and a wrap resumes at 0, straight onto live segment names. Exhaustion is
    /// an explicit refusal instead (mandates #1 and #4).
    #[tokio::test]
    async fn discovery_refuses_an_exhausted_sequence_rather_than_overflowing() -> Result<()> {
        let (fs, base) = fs_and_base().await;
        std::fs::write(
            format!(
                "{}/{}",
                base.trim_end_matches('/'),
                spill_segment_name(u64::MAX)
            ),
            b"not a parquet file",
        )
        .expect("write max-sequence segment");

        let store = SpillRecordStorage::with_flush_threshold(fs, base, None);
        let err = store
            .scan_records(usize::MAX)
            .await
            .expect_err("an exhausted sequence must refuse, not overflow or wrap");
        let msg = err.to_string();
        assert!(
            msg.contains("sequence space exhausted"),
            "the error must name the cause; got: {msg}"
        );
        Ok(())
    }

    /// Build `{tmp}/order` (ours) and `{tmp}/orders` (a sibling table), with one
    /// real segment object in each, plus a filesystem whose `list` injects the
    /// sibling the way a recursive/raw-prefix object-store LIST would.
    async fn bleeding_pair() -> (Arc<ListInjectFs>, String, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let ours = format!("{root}/order");
        let sibling = format!("{root}/orders");
        std::fs::create_dir_all(&ours).expect("mkdir ours");
        std::fs::create_dir_all(&sibling).expect("mkdir sibling");
        // The sibling's object has a PERFECTLY VALID segment name — that is the
        // whole point: its basename is indistinguishable from one of ours.
        //
        // Sequence 5, NOT 0: our own first flush takes 0, and an implementation
        // that decided membership from the basename would then produce the same
        // canonical path for both and dedup the sibling away, so these tests
        // would pass while the defect was present. A distinct sequence makes the
        // adoption observable.
        let sibling_object = format!("{sibling}/{}", spill_segment_name(5));
        std::fs::write(&sibling_object, b"sibling table's segment").expect("write sibling");
        let inner = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        let fs = Arc::new(ListInjectFs {
            inner,
            inject_unscoped: vec![ListInjectFs::entry(&format!("file://{sibling_object}"))],
            inject_always: Vec::new(),
        });
        (fs, ours, sibling_object)
    }

    /// A sibling prefix's objects must be IGNORED, not adopted.
    ///
    /// The object stores list by key prefix and recursively, and GCS passes the
    /// prefix through raw, so a partition at `…/order` also sees `…/orders/…`.
    /// Deciding membership from the basename would re-root another table's
    /// object under our prefix: every read of `order` would then fail on a key
    /// that does not exist, and `delete_record` would see a spurious
    /// `has_segments` and tombstone a never-flushed row — which pins
    /// `min_unflushed_lsn` fail-closed forever.
    #[tokio::test]
    async fn discovery_ignores_objects_from_a_sibling_prefix() -> Result<()> {
        let (fs, ours, _sibling) = bleeding_pair().await;

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), ours.clone(), None);
        first.upsert_record(record("k", "ours")).await?;
        first.flush().await?;
        drop(first);

        let second = SpillRecordStorage::with_flush_threshold(fs, ours, None);
        let all = second.scan_records(usize::MAX).await?;
        assert_eq!(
            all.len(),
            1,
            "only this partition's own segment may be discovered"
        );
        assert_eq!(all[0].oid, "k");
        assert_eq!(
            second.segment_count(),
            1,
            "the sibling table's object must not enter the segment list"
        );
        Ok(())
    }

    /// Purge must reclaim a LEGACY-named segment — the flag day's escape hatch.
    ///
    /// The rename to `spill-{seq:016x}` ships with no migration: a partition
    /// holding `spill-{uuid}-{seq}` objects refuses every read, because the
    /// order of such a set cannot be established. That is tolerable ONLY because
    /// a DROP reclaims them, which is why `is_spill_segment` stays permissive
    /// (prefix+suffix) where `parse_spill_seq` is strict. Tighten
    /// `is_spill_segment` to agree with `parse_spill_seq` and those objects leak
    /// forever — so this test is what makes that asymmetry load-bearing rather
    /// than incidental.
    ///
    /// Claimed in `ENV_GATE_REGISTRY` and in TD-USUB-1; untested until now.
    #[tokio::test]
    async fn purge_reclaims_a_legacy_named_segment() -> Result<()> {
        let (fs, base) = fs_and_base().await;
        let legacy = format!(
            "{}/spill-deadbeefcafe-0000000001.parquet",
            base.trim_end_matches('/')
        );
        std::fs::write(&legacy, b"a segment written under the previous naming")
            .expect("write legacy segment");

        let store = SpillRecordStorage::with_flush_threshold(fs, base.clone(), None);
        // Captured, not asserted yet: reads must refuse while the object is
        // there (the flag day), but asserting it HERE would short-circuit the
        // run on that symptom and the test would never reach the escape hatch
        // it exists to pin. Checked last.
        let read_refused = store.scan_records(usize::MAX).await.is_err();

        store.purge_durable_objects().await?;

        assert!(
            !std::path::Path::new(&legacy).exists(),
            "purge must reclaim the legacy-named segment, or a DROP leaks it forever"
        );
        assert_eq!(segment_files_on_disk(&base), 0);
        assert!(
            read_refused,
            "a legacy-named segment must also make reads refuse"
        );
        Ok(())
    }

    /// A RESIDENT row that is already dead must not read back live, and
    /// deleting it must report no removal.
    ///
    /// The memtable is a read boundary like the segments are (mandate #16a).
    /// #1952 filtered the two segment arms and this commit added the third in
    /// `delete_record`'s probe — all three only ever see FLUSHED copies. The
    /// resident copies were left unfiltered, and
    /// `a_dead_record_from_a_segment_does_not_read_back_live` cannot catch it
    /// because it flushes first. With the threshold unset nothing flushes, so
    /// this exercises the memtable alone.
    #[tokio::test]
    async fn a_dead_resident_record_is_neither_read_nor_counted_as_deleted() -> Result<()> {
        let s = store(None).await;
        let mut dead = record("gone", "open");
        dead.valid_to_ns = Some(0);
        s.upsert_record(dead).await?;
        s.upsert_record(record("live", "open")).await?;
        assert_eq!(s.segment_count(), 0, "precondition: nothing flushed");
        assert_eq!(s.resident_len(), 2, "precondition: both rows resident");

        assert!(
            s.get_record(&RecordKey::new("gone".to_string()))
                .await?
                .is_none(),
            "a dead resident row must not read back live"
        );
        assert!(
            s.get_record(&RecordKey::new("live".to_string()))
                .await?
                .is_some(),
            "and a live resident row must still be served"
        );
        assert_eq!(
            s.scan_records(usize::MAX).await?.len(),
            1,
            "the scan already filtered it; get_record must agree"
        );
        assert!(
            !s.delete_record(&RecordKey::new("gone".to_string())).await?,
            "deleting an already-dead resident row must report no removal"
        );
        Ok(())
    }

    /// A DEAD resident copy over a LIVE segment copy: every surface must agree
    /// the row is gone, including `delete_record`'s return value.
    ///
    /// This is the state the resident filter was added for and did not cover.
    /// `was_resident` collapsed "absent" and "resident but dead" into one
    /// `false`, so the dead copy fell through to the segment probe, found the
    /// stale LIVE copy, and reported that the DELETE removed a row — while
    /// `get_record` and `merged_records`, on the same state, both reported it
    /// absent. Neither earlier test could reach it:
    /// `a_dead_resident_record_is_neither_read_nor_counted_as_deleted` never
    /// flushes (so it exits on `!has_segments`), and
    /// `deleting_an_already_dead_segment_row_reports_no_removal` puts the dead
    /// copy in the SEGMENT with nothing resident.
    #[tokio::test]
    async fn a_dead_resident_copy_shadows_a_live_segment_copy_on_every_surface() -> Result<()> {
        let s = store(Some(2)).await;
        // Flush a LIVE `k` into a segment.
        s.upsert_record(record("k", "live-in-segment")).await?;
        s.upsert_record(record("filler", "open")).await?;
        assert!(
            s.segment_count() > 0,
            "precondition: `k` is flushed and live"
        );
        assert_eq!(s.resident_len(), 0, "precondition: nothing resident");

        // Now a DEAD resident copy of the same oid.
        let mut dead = record("k", "dead-resident");
        dead.valid_to_ns = Some(0);
        s.upsert_record(dead).await?;

        let key = RecordKey::new("k".to_string());
        assert!(
            s.get_record(&key).await?.is_none(),
            "get_record must not serve the stale live segment copy"
        );
        assert!(
            !s.scan_records(usize::MAX)
                .await?
                .iter()
                .any(|r| r.oid == "k"),
            "the scan must agree the row is gone"
        );
        assert!(
            !s.delete_record(&key).await?,
            "delete_record must agree too — not re-count the row from the segment"
        );

        // AFTER the delete, too. Without this the test cannot distinguish a
        // correct early return from one that skipped the tombstone: moving the
        // `tombstones.insert` below `resident_was_dead`'s return would resurrect
        // the live segment copy here, and every other assertion would still
        // pass.
        assert!(
            s.get_record(&key).await?.is_none(),
            "the live segment copy must stay suppressed after the delete"
        );
        assert!(
            !s.scan_records(usize::MAX)
                .await?
                .iter()
                .any(|r| r.oid == "k"),
            "and must not reappear in a scan"
        );
        Ok(())
    }

    /// A row DEAD in the newest segment and LIVE in an older one is gone, on
    /// every surface including `delete_record`'s count.
    ///
    /// Newest-segment-first is only authoritative if the FIRST hit decides. An
    /// earlier version of this commit filtered dead rows within each segment and
    /// then continued to an older one, so this state made `delete_record` report
    /// a removal while `get_record` and `merged_records` reported the row
    /// absent — the same cross-surface disagreement, one layer out from the
    /// resident case above.
    #[tokio::test]
    async fn a_row_dead_in_the_newest_segment_is_gone_despite_a_live_older_copy() -> Result<()> {
        let s = store(None).await;
        s.upsert_record(record("x", "live")).await?;
        s.upsert_record(record("keep", "live")).await?;
        s.flush().await?; // segment 0: x LIVE
        let mut dead = record("x", "expired");
        dead.valid_to_ns = Some(0);
        s.upsert_record(dead).await?;
        s.flush().await?; // segment 1 (newest): x DEAD
        assert_eq!(s.segment_count(), 2, "precondition: two segments hold `x`");
        assert_eq!(s.resident_len(), 0, "precondition: nothing resident");

        let key = RecordKey::new("x".to_string());
        assert!(
            s.get_record(&key).await?.is_none(),
            "the newest segment's dead copy is authoritative"
        );
        assert!(
            !s.scan_records(usize::MAX)
                .await?
                .iter()
                .any(|r| r.oid == "x"),
            "the merge must agree"
        );
        assert!(
            !s.delete_record(&key).await?,
            "delete_record must not consult an OLDER segment for a live copy"
        );
        Ok(())
    }

    /// Two listing entries proposing ONE canonical path must collapse to one
    /// segment, not two reads of the same object.
    ///
    /// A paginated listing can repeat a key, and a foreign object whose basename
    /// matches one of ours maps to the path our own segment already occupies.
    /// The other sibling/nested tests deliberately pick sequences 5 and 7 to
    /// AVOID collapsing, so none of them reaches this state.
    ///
    /// Pins the OBSERVABLE property, and deliberately does not claim to isolate
    /// one mechanism: TWO independent layers deliver it — the `seen` set in
    /// `list_partition_objects` and `ensure_discovered`'s refusal of a path
    /// already in `segments` — so removing either alone leaves this passing.
    /// Mutation testing is what established that, and the `seen` comment now
    /// says it is a cost guard rather than a correctness one.
    #[tokio::test]
    async fn two_entries_proposing_one_canonical_path_collapse_to_one_segment() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        // Flush with a PLAIN filesystem first. The double injects on every
        // listing, so injecting before the flush would have the first store
        // adopt a sequence-0 path that does not exist yet and flush to 1
        // instead — the duplicate must name a REAL object to model a repeated
        // listing page.
        let plain = Arc::new(
            LocalFileSystem::new(LocalConfig::default())
                .await
                .expect("local filesystem"),
        ) as Arc<dyn FileSystem>;
        let first = SpillRecordStorage::with_flush_threshold(plain, base.clone(), None);
        first.upsert_record(record("k", "ours")).await?;
        let written = first
            .flush()
            .await?
            .expect("the flush must publish a segment");
        drop(first);
        assert!(
            written.ends_with(&spill_segment_name(0)),
            "precondition: the flush took sequence 0, got {written}"
        );

        let inner = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        let fs = Arc::new(ListInjectFs {
            inner,
            inject_unscoped: Vec::new(),
            // The same real object, reported a second time.
            inject_always: vec![ListInjectFs::entry(&format!("file://{written}"))],
        });

        let second = SpillRecordStorage::with_flush_threshold(fs, base, None);
        let all = second.scan_records(usize::MAX).await?;
        assert_eq!(all.len(), 1, "the row must appear once, not twice");
        assert_eq!(
            second.segment_count(),
            1,
            "one object must yield one segment entry, however often it is listed"
        );
        Ok(())
    }

    /// A wrapper filesystem that renames objects on write but not on `list`
    /// must make the partition REFUSE, not look empty.
    ///
    /// `EncryptedFilesystem` appends its `encrypted_extension` (default `.enc`)
    /// to every read/write/delete/exists path while its `list` passes names
    /// through unchanged, so a segment written as `spill-{seq}.parquet` is
    /// listed as `spill-{seq}.parquet.enc` (TD-ENCFS-1). Without the detector
    /// that name is simply not spill-shaped and the consequences are all
    /// silent: discovery finds nothing, so flushed rows read as absent and
    /// `delete_record` records no tombstone; `resume` restarts at 0 so the next
    /// flush collides with the underlying object forever; purge reclaims
    /// nothing while reporting success.
    #[tokio::test]
    async fn a_wrapper_mangled_segment_name_refuses_rather_than_reading_empty() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        // Exactly what the encryption wrapper's `list` would report.
        let mangled = format!(
            "{}/{}.enc",
            base.trim_end_matches('/'),
            spill_segment_name(3)
        );
        std::fs::write(&mangled, b"an encrypted segment").expect("write mangled segment");
        let inner = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        let fs = Arc::new(ListInjectFs {
            inner,
            inject_unscoped: Vec::new(),
            inject_always: Vec::new(),
        });

        let store = SpillRecordStorage::with_flush_threshold(fs, base, None);
        let err = store.scan_records(usize::MAX).await.expect_err(
            "a mangled segment name must refuse the read, not report an empty partition",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("rewriting object names"),
            "the error must name the cause; got: {msg}"
        );
        Ok(())
    }

    /// Canonical reconstruction alone must protect a foreign object, with no
    /// help from listing scope.
    ///
    /// The sibling tests cannot show this: scoping and canonical reconstruction
    /// are independent protections and either one suffices, so neither mutation
    /// alone makes them fail. Here the foreign entry is injected into even a
    /// correctly SCOPED listing — the conservative assumption that a backend may
    /// report a key we did not expect — so the only thing standing between purge
    /// and another table's object is that the path it deletes is rebuilt under
    /// OUR prefix.
    ///
    /// Mutate the path source to the listing's own url and this fails.
    #[tokio::test]
    async fn purge_never_deletes_a_path_outside_its_own_prefix() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let ours = format!("{root}/order");
        let foreign_dir = format!("{root}/orders");
        std::fs::create_dir_all(&ours).expect("mkdir ours");
        std::fs::create_dir_all(&foreign_dir).expect("mkdir foreign");
        // Sequence 5, distinct from our own flush at 0, so it cannot be deduped.
        let foreign = format!("{foreign_dir}/{}", spill_segment_name(5));
        std::fs::write(&foreign, b"another table's segment").expect("write foreign");
        let inner = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        let fs = Arc::new(ListInjectFs {
            inner,
            inject_unscoped: Vec::new(),
            // Returned even for a scoped listing.
            inject_always: vec![ListInjectFs::entry(&format!("file://{foreign}"))],
        });

        let store = SpillRecordStorage::with_flush_threshold(fs, ours.clone(), None);
        store.upsert_record(record("k", "ours")).await?;
        store.flush().await?;
        assert_eq!(segment_files_on_disk(&ours), 1);

        store.purge_durable_objects().await?;

        assert!(
            std::path::Path::new(&foreign).exists(),
            "purge deleted a path OUTSIDE its own prefix: {foreign}"
        );
        assert_eq!(
            segment_files_on_disk(&ours),
            0,
            "and must still delete its own segment"
        );
        Ok(())
    }

    /// Purge must not delete a sibling prefix's objects.
    ///
    /// Same listing hazard as above, with the worse consequence: on GCS,
    /// dropping table `order` would DELETE table `orders`' segments. Asserts
    /// the sibling's bytes survive, not merely that purge reported success.
    #[tokio::test]
    async fn purge_only_deletes_objects_under_its_own_prefix() -> Result<()> {
        let (fs, ours, sibling_object) = bleeding_pair().await;

        let store = SpillRecordStorage::with_flush_threshold(fs, ours.clone(), None);
        store.upsert_record(record("k", "ours")).await?;
        store.flush().await?;
        assert_eq!(segment_files_on_disk(&ours), 1);

        store.purge_durable_objects().await?;

        assert_eq!(
            segment_files_on_disk(&ours),
            0,
            "purge must delete this partition's own segments"
        );
        assert!(
            std::path::Path::new(&sibling_object).exists(),
            "purge deleted ANOTHER table's segment: {sibling_object}"
        );
        Ok(())
    }

    /// Deleting a row whose only segment copy is already DEAD must report that
    /// it removed nothing.
    ///
    /// `delete_record`'s segment probe is the THIRD segment-read boundary, and
    /// #1952 applied the canonical dead-record predicate at the other two
    /// (`merged_records`, `get_record`) while missing this one. Without the
    /// filter the probe counts a tombstoned/expired row as present and DELETE
    /// reports a wrong affected-row count — a count/stats surface mandate #16a
    /// names explicitly.
    #[tokio::test]
    async fn deleting_an_already_dead_segment_row_reports_no_removal() -> Result<()> {
        let s = store(Some(2)).await;
        let mut dead = record("gone", "open");
        dead.valid_to_ns = Some(0);
        s.upsert_record(dead).await?;
        s.upsert_record(record("other", "open")).await?;
        assert!(s.segment_count() > 0, "precondition: flushed to a segment");
        assert_eq!(s.resident_len(), 0, "precondition: nothing resident");

        assert!(
            !s.delete_record(&RecordKey::new("gone".to_string())).await?,
            "an already-dead segment row must not be reported as removed"
        );
        Ok(())
    }

    /// A spill-named object nested BELOW the prefix fails the read LOUDLY.
    ///
    /// A recursive listing returns it even when correctly scoped, and its
    /// basename is indistinguishable from one of ours — so it is proposed as a
    /// candidate. What keeps that safe is canonical reconstruction: the path
    /// built from its sequence points under OUR prefix, where nothing lives, so
    /// the read fails with not-found instead of serving a foreign object's
    /// bytes. Wrong-and-loud, never wrong-and-silent (mandate #1).
    ///
    /// Nothing writes nested spill names today; this pins the failure DIRECTION
    /// if anything ever does.
    #[tokio::test]
    async fn a_nested_spill_named_object_fails_the_read_rather_than_being_served() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let nested_dir = format!("{base}/sub");
        std::fs::create_dir_all(&nested_dir).expect("mkdir nested");
        // Sequence 7, distinct from our own flush, so it cannot be deduped away.
        let nested = format!("{nested_dir}/{}", spill_segment_name(7));
        std::fs::write(&nested, b"not ours").expect("write nested");
        let inner = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        let fs = Arc::new(ListInjectFs {
            inner,
            inject_unscoped: Vec::new(),
            // Returned even for a scoped listing: that is what "recursive" means.
            inject_always: vec![ListInjectFs::entry(&format!("file://{nested}"))],
        });

        let store = SpillRecordStorage::with_flush_threshold(fs, base, None);
        store.upsert_record(record("k", "ours")).await?;
        store.flush().await?;

        let err = store
            .scan_records(usize::MAX)
            .await
            .expect_err("an adopted foreign candidate must fail the read, not be served");
        let msg = err.to_string();
        assert!(
            msg.contains("read segment") || msg.contains("decode segment"),
            "the failure must be a read of OUR canonical path, not a decode of \
             the foreign object; got: {msg}"
        );
        assert!(
            std::path::Path::new(&nested).exists(),
            "and the foreign object is never touched"
        );
        Ok(())
    }

    /// OUR OWN segments must be found when the listing spells their location
    /// differently from the `base_path` we were given.
    ///
    /// This is the regression test for the dangerous direction. `base_path` is a
    /// caller string; `DirEntry::url` is reconstructed by the backend, and they
    /// disagree in real configurations — notably a bare-relative `base_path`
    /// (the shape `DrPathBuilder` emits) against a `root_dir`-anchored local
    /// filesystem, which reports an absolute `file://…` url because `local.rs`
    /// takes its relative branch only for a path literally starting `./`.
    ///
    /// An implementation that decided membership by comparing those two strings
    /// would skip every one of our segments and return Ok: reads would report
    /// flushed rows as absent, `delete_record` would record no tombstone, and
    /// purge would delete nothing and report success. Silent loss, with no
    /// fail-closed arm firing — which is why membership is not decided by
    /// comparing those two strings at all. A candidate's path is rebuilt
    /// canonically instead, so a spelling difference cannot hide our own
    /// segments.
    #[tokio::test]
    async fn discovery_finds_our_segments_when_the_listing_spells_them_differently() -> Result<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        std::mem::forget(dir);
        let inner = LocalFileSystem::new(LocalConfig {
            root_dir: Some(root.clone()),
            ..LocalConfig::default()
        })
        .await
        .expect("local filesystem with a root_dir");
        let fs = Arc::new(inner) as Arc<dyn FileSystem>;

        // Bare-relative, no leading `./` — the shape `DrPathBuilder` emits.
        let base = "data/tenant-a/ns1/orders".to_string();

        let first = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        first.upsert_record(record("k", "old")).await?;
        first.flush().await?;
        first.upsert_record(record("k", "new")).await?;
        first.flush().await?;
        drop(first);

        let second = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        assert_eq!(
            second.segment_count(),
            0,
            "nothing discovered before the first read"
        );
        let got = second
            .get_record(&RecordKey::new("k".to_string()))
            .await?
            .expect("our own flushed row must be discovered despite the url spelling");
        assert!(
            status_is(&got, "new"),
            "and must resolve to the newest segment"
        );
        assert_eq!(second.segment_count(), 2, "both our segments must be found");

        // Purge must find them too — the same membership rule decides both.
        second.purge_durable_objects().await?;
        let on_disk = root.join(&base);
        assert_eq!(
            segment_files_on_disk(&on_disk.to_string_lossy()),
            0,
            "purge must delete our segments, not silently report success"
        );
        Ok(())
    }

    /// Discovery changes what an ORPHANED segment costs, and this pins the
    /// premise that makes it so.
    ///
    /// The partition prefix is a pure function of `(tenant, collection)`, and
    /// with `PROXIMADB_WAL_OBJECT_ID_KEY` default-OFF the collection key is the
    /// bare table NAME — so a table dropped and recreated under the same name
    /// reuses its predecessor's prefix. DROP purges the prefix first
    /// (`purge_durable_objects`), so this is reachable only when that purge
    /// failed or crashed part-way: TD-USUB-1's open deletion-obligation item.
    ///
    /// Before discovery that window cost wasted storage. With discovery, a
    /// surviving object is found and SERVED — a dropped table's rows appearing
    /// in a recreated one. The mechanism is not wrong; its precondition is.
    /// Hence the obligation marker + drain is a correctness prerequisite for
    /// enabling the spill gate, not hygiene to be done later.
    ///
    /// Asserts the premise (a second store over the same prefix serves what the
    /// first left) rather than asserting resurrection as desired behaviour.
    #[tokio::test]
    async fn an_unpurged_prefix_is_served_to_the_next_store_at_that_path() -> Result<()> {
        let (fs, base) = fs_and_base().await;

        let dropped = SpillRecordStorage::with_flush_threshold(fs.clone(), base.clone(), None);
        dropped.upsert_record(record("leftover", "old")).await?;
        dropped.flush().await?;
        // NOTE: no `purge_durable_objects` — standing in for a purge that failed
        // or a crash before it ran. The live DROP path does call it.
        drop(dropped);
        assert_eq!(
            segment_files_on_disk(&base),
            1,
            "the orphaned object must still be on disk for this to mean anything"
        );

        let recreated = SpillRecordStorage::with_flush_threshold(fs, base, None);
        let served = recreated.scan_records(usize::MAX).await?;
        assert_eq!(
            served.len(),
            1,
            "discovery serves whatever the prefix holds — which is why the \
             prefix must be empty before it is reused (TD-USUB-1 open item: \
             deletion-obligation marker + drain)"
        );
        Ok(())
    }
}
