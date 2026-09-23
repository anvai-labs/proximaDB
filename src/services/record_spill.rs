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
fn configured_flush_threshold() -> Option<usize> {
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
    /// Flushed segment paths, oldest first. Later entries shadow earlier ones.
    segments: parking_lot::RwLock<Vec<String>>,
    /// Resident-row count after which a write triggers a flush. `None` ⇒ never.
    flush_threshold: Option<usize>,
    filesystem: Arc<dyn FileSystem>,
    /// Directory under which this partition's segments are written. Callers build
    /// it with `DrPathBuilder`; this store never constructs a raw path itself.
    base_path: String,
    /// Monotonic segment counter, so segment names are fresh-by-construction and
    /// a write never overwrites a live object (ADR-062 fresh-name discipline).
    next_segment: AtomicU64,
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
            segments: parking_lot::RwLock::new(Vec::new()),
            flush_threshold,
            filesystem,
            base_path: base_path.into(),
            next_segment: AtomicU64::new(0),
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

        let seq = self.next_segment.fetch_add(1, Ordering::SeqCst);
        let path = format!(
            "{}/spill-{seq:010}.parquet",
            self.base_path.trim_end_matches('/')
        );

        self.filesystem
            .write(&path, &bytes, None)
            .await
            .map_err(|e| anyhow::anyhow!("spill: write segment '{path}' failed: {e}"))?;

        // Only drop the resident rows AFTER the segment is durable. A crash
        // before this point simply leaves them resident, and the WAL replays
        // them regardless — the store cannot lose a row either way.
        self.segments.write().push(path.clone());
        for record in &records {
            self.memtable.remove(&record.oid);
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
        // Already suppressed: nothing live to remove, so report false even if a
        // stale segment copy still exists on disk.
        if self.tombstones.contains(&key.oid) {
            self.memtable.remove(&key.oid);
            return Ok(false);
        }

        let was_resident = self.memtable.remove(&key.oid).is_some();
        let has_segments = !self.segments.read().is_empty();

        // A row already written to an immutable segment cannot be removed, so it
        // is suppressed instead. The tombstone is recorded whenever segments
        // exist — a redundant tombstone is harmless, a missing one resurrects
        // the row.
        if has_segments {
            self.tombstones.insert(key.oid.clone());
        }

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
