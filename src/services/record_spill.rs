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
    /// Monotonic segment counter — unique **within one store instance**.
    next_segment: AtomicU64,
    /// Per-instance nonce, unique **across** instances sharing a `base_path`.
    ///
    /// The counter alone is not enough for ADR-062's fresh-name discipline once
    /// `base_path` is derived deterministically from `(tenant, collection)`, as
    /// the partition factory now derives it. A second store at the same path —
    /// after a process restart, or after `drop_partition_state` evicts the
    /// partition and the next access rebuilds it — restarts its counter at 0 and
    /// would re-issue `spill-0000000000`. `FileSystem::write` with `options:
    /// None` takes the overwrite branch, so that clobbers the earlier object
    /// **silently**.
    ///
    /// The nonce makes names fresh by construction across instances; `flush`
    /// additionally writes with `overwrite: false` so that if this reasoning is
    /// ever wrong the write FAILS rather than destroying a segment (mandate #1).
    instance: String,
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
            instance: uuid::Uuid::new_v4().simple().to_string(),
        }
    }

    /// Resident (unflushed) row count. This is the number the heap bound applies
    /// to — NOT the logical row count of the table.
    pub fn resident_len(&self) -> usize {
        self.memtable.len()
    }

    /// Number of durable segments written so far.
    pub fn segment_count(&self) -> usize {
        self.segments.read().len()
    }

    /// The configured flush threshold, if any.
    pub fn flush_threshold(&self) -> Option<usize> {
        self.flush_threshold
    }

    /// Write every resident row to a fresh durable segment and clear the
    /// memtable. A no-op when nothing is resident.
    ///
    /// Tombstones are deliberately **retained** across a flush: they suppress
    /// copies in *older* segments, which this flush does not rewrite.
    pub async fn flush(&self) -> Result<Option<String>> {
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

        // `instance` before `seq`: the counter is unique only within one store,
        // and `base_path` is now derived deterministically from the partition
        // identity, so two stores at the same path would otherwise both emit
        // `spill-0000000000`. See the `instance` field.
        let seq = self.next_segment.fetch_add(1, Ordering::SeqCst);
        let path = format!(
            "{}/spill-{}-{seq:010}.parquet",
            self.base_path.trim_end_matches('/'),
            self.instance
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
        Ok(merged.into_values().collect())
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
        if self.tombstones.contains(&key.oid) {
            return Ok(None);
        }
        if let Some(record) = self.memtable.get(&key.oid) {
            return Ok(Some(record.value().clone()));
        }
        // Newest segment first: a later flush shadows an earlier copy.
        let paths = self.segments.read().clone();
        for path in paths.iter().rev() {
            if let Some(found) = self
                .read_segment(path)
                .await?
                .into_iter()
                .find(|r| r.oid == key.oid)
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
        // Exclusive with `flush` (TD-USUB-13): `has_segments` below and the
        // tombstone insert must not straddle a flush's `segments.push`, or a row
        // that flush is mid-way through making durable gets no tombstone and
        // comes back live. Scoped so the guard is released before the segment
        // READS at the end of this method — those are idempotent and need no
        // exclusion.
        let (was_resident, has_segments) = {
            let _exclusive = self.flush_guard.lock().await;

            // Already suppressed: nothing live to remove, so report false even
            // if a stale segment copy still exists on disk.
            if self.tombstones.contains(&key.oid) {
                self.memtable.remove(&key.oid);
                return Ok(false);
            }

            let was_resident = self.memtable.remove(&key.oid).is_some();
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

            (was_resident, has_segments)
        };

        // Report truthfully whether a live row went away. The segment read is
        // paid ONLY when the answer is not already known from the memtable.
        if was_resident {
            return Ok(true);
        }
        if !has_segments {
            return Ok(false);
        }
        let paths = self.segments.read().clone();
        for path in paths.iter().rev() {
            if self
                .read_segment(path)
                .await?
                .iter()
                .any(|r| r.oid == key.oid)
            {
                return Ok(true);
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
        let prefix = self.base_path.trim_end_matches('/').to_string();
        let entries = match self.filesystem.list(&prefix).await {
            Ok(entries) => entries,
            // Nothing was ever written under this prefix — a partition that never
            // flushed has nothing to delete, and the contract requires idempotence.
            //
            // BOTH arms are needed. The local backend surfaces a missing directory
            // as `Io(ErrorKind::NotFound)` from `read_dir`, not as the typed
            // `NotFound` variant — matching only the latter would make DROP TABLE
            // fail outright for any table that never spilled.
            Err(err) if is_absent(&err) => {
                // Clear BOTH, exactly as the success path does. Harmless today
                // because `drop_table_records` releases the whole store right
                // after — but an orphan reaper calling purge standalone would
                // otherwise leave a store whose segments are gone while its
                // tombstones survive, and the two arms must not diverge.
                self.segments.write().clear();
                self.tombstones.clear();
                return Ok(());
            }
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "spill: list segment prefix '{prefix}' for purge failed: {err}"
                ));
            }
        };

        for entry in entries {
            if entry.metadata.is_directory || !is_spill_segment(&entry.name) {
                continue;
            }
            self.filesystem.delete(&entry.url).await.map_err(|err| {
                anyhow::anyhow!(
                    "spill: delete segment '{}' during purge failed: {err}",
                    entry.url
                )
            })?;
        }

        // Only after the objects are gone: a crash midway leaves the remaining
        // objects still listed under the prefix, so a retry finds and removes
        // them.
        self.segments.write().clear();
        self.tombstones.clear();
        Ok(())
    }
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

/// `true` for an object this store wrote — `spill-{instance}-{seq}.parquet`.
///
/// Matched by prefix and extension rather than by parsing the name, and
/// deliberately narrow: the prefix belongs to this partition, but refusing to
/// delete anything that does not look like our own output means a path
/// misconfiguration cannot turn a DROP into a delete of someone else's objects.
fn is_spill_segment(name: &str) -> bool {
    name.starts_with("spill-") && name.ends_with(".parquet")
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

    /// Purge is idempotent: a partition that never flushed, and a re-drop, are
    /// both success. The contract says so, and `drop_table_records` now propagates
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
    /// Without the per-instance nonce both would emit `spill-0000000000`, and
    /// `write` with `options: None` overwrites silently: the first store's
    /// durable rows would be gone with no error anywhere.
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
}
