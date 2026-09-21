//! Canonical in-memory current-state store for direct record writes.
//!
//! This store is a neutral `ProximaRecord` memtable used by early pgwire
//! relational DML wiring while PAX/LSM current-state storage is still being
//! connected. Durability remains in the canonical WAL; this structure is
//! rebuildable from Layer 0 entries and must not grow independent persistence.
//!
//! # TD-USUB-1 slice 1 — make the growth visible and boundable
//!
//! Every pgwire relational table lands here, and this map had **no flush, evict,
//! spill or compact path** — so a long-lived table grew the heap without bound,
//! and (because the canonical WAL is never truncated) the WAL alongside it.
//! ADR-094 records this as defect (a).
//!
//! The durable fix is a real record store plus WAL rotation (TD-USUB-1 slice 2).
//! That work needs to know how fast this actually grows, and mandate #6 says
//! measure before you build. So this slice does exactly two things, neither of
//! which can regress anything:
//!
//! * **Observe** — export the live record count as a gauge, so growth is visible
//!   in the same Prometheus surface as everything else.
//! * **Bound, fail-closed** — an optional cap that rejects the write which would
//!   exceed it, instead of letting the process OOM. Default is **unbounded**, so
//!   with no configuration the behaviour is byte-identical to before.
//!
//! Deliberately NOT done here: this slice adds no persistence, honouring the
//! module contract above. It does not change recovery semantics, so it cannot
//! lose data.
//!
//! **Known limitation, stated plainly:** the cap counts *records*, not bytes. Row
//! width varies, so a record count is a proxy for memory, not a guarantee. Byte
//! accounting needs a size estimator `ProximaRecord` does not have today; it is
//! slice 2's job, alongside the durable store that makes eviction possible.

use anyhow::{Result, bail};
use async_trait::async_trait;
use dashmap::DashMap;
use proximadb_records::{
    ProximaRecord, RecordKey, RecordRecoverySummary, RecordScan, RecordScanOptions,
    RecordScanPredicate, RecordStore, RecordStoreResult,
};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};

/// Cap on live records per relational memtable partition. Unset ⇒ **unbounded**
/// (the shipped default; byte-identical to the pre-TD-USUB-1 behaviour).
const MEMTABLE_MAX_RECORDS_ENV: &str = "PROXIMADB_RELATIONAL_MEMTABLE_MAX_RECORDS";

/// Live records held across all relational memtable partitions.
///
/// Registered lazily and tolerantly. Observability must never take down the write
/// path, so every failure mode degrades instead of panicking: a construction error
/// yields `None` (no gauge, writes unaffected), and a duplicate registration —
/// several servers in one test process — leaves a working but un-exported gauge.
/// No `.expect()` here: this is `src/` production code (mandate #4).
static MEMTABLE_RECORDS: std::sync::LazyLock<Option<prometheus::IntGauge>> =
    std::sync::LazyLock::new(|| {
        let gauge = prometheus::IntGauge::new(
            "proximadb_relational_memtable_records",
            "Live records held in relational memtables (TD-USUB-1: unbounded until slice 2)",
        )
        .ok()?;
        // Ignore AlreadyReg — the gauge still works, it is simply not re-exported.
        let _ = prometheus::register(Box::new(gauge.clone()));
        Some(gauge)
    });

/// Read the configured per-partition record cap. `None` ⇒ unbounded.
///
/// A zero or unparsable value is treated as unset rather than as "reject every
/// write" — a typo in an env var must not silently make the database read-only.
fn configured_max_records() -> Option<usize> {
    std::env::var(MEMTABLE_MAX_RECORDS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Rebuildable current-state memtable keyed by canonical record OID.
#[derive(Debug, Default)]
pub struct MemtableRecordStorage {
    records: DashMap<String, ProximaRecord>,
    /// Max live records; `None` ⇒ unbounded (default).
    max_records: Option<usize>,
}

impl MemtableRecordStorage {
    /// Create an empty current-state memtable, bounded per
    /// `PROXIMADB_RELATIONAL_MEMTABLE_MAX_RECORDS` (unset ⇒ unbounded).
    pub fn new() -> Self {
        Self {
            records: DashMap::new(),
            max_records: configured_max_records(),
        }
    }

    /// Create an empty memtable with an explicit cap — used by tests and by callers
    /// that configure the bound directly rather than through the environment.
    pub fn with_max_records(max_records: Option<usize>) -> Self {
        Self {
            records: DashMap::new(),
            max_records,
        }
    }

    /// The configured cap, if any.
    pub fn max_records(&self) -> Option<usize> {
        self.max_records
    }

    /// Publish the current record count to the process gauge, if one exists.
    fn publish_len(&self) {
        if let Some(gauge) = MEMTABLE_RECORDS.as_ref() {
            gauge.set(self.records.len() as i64);
        }
    }

    /// Rebuild current state from canonical WAL entries.
    pub async fn replay_wal_entries<I>(&self, entries: I) -> Result<RecordRecoverySummary>
    where
        I: IntoIterator<Item = CanonicalWalEntry>,
    {
        let mut summary = RecordRecoverySummary::default();

        for entry in entries {
            match entry.operation {
                CanonicalOperation::RecordUpsert { record, .. } => {
                    self.upsert_record(*record).await?;
                    summary.upserts_replayed += 1;
                }
                CanonicalOperation::RecordDelete { oid, .. } => {
                    self.delete_record(&RecordKey::new(oid)).await?;
                    summary.deletes_replayed += 1;
                }
                // A partition drop is scoped by the DirectWal replay (which owns
                // the partition map and discards the whole memtable); a single
                // memtable has no cross-collection scope to clear.
                CanonicalOperation::RecordPartitionDrop { .. } => {}
                // Checkpoints, CDC barriers, and system-catalog mutations carry
                // no record state for the memtable to replay.
                CanonicalOperation::Checkpoint(_)
                | CanonicalOperation::CdcBarrier { .. }
                | CanonicalOperation::CatalogMutation { .. } => {}
            }
        }

        Ok(summary)
    }

    /// Number of records currently visible in the memtable.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the memtable has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[async_trait]
impl RecordStore for MemtableRecordStorage {
    async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord> {
        // TD-USUB-1: fail closed at the cap rather than growing without bound.
        // Only a write that ADDS an oid can exceed it — updating a record already
        // resident does not grow the set, so it is always admitted (rejecting it
        // would make a full table permanently un-correctable, which is worse).
        if let Some(max) = self.max_records
            && self.records.len() >= max
            && !self.records.contains_key(&record.oid)
        {
            bail!(
                "relational memtable is full: {max} live records \
                 ({MEMTABLE_MAX_RECORDS_ENV}). This store has no spill path yet \
                 (TD-USUB-1); raise the cap, or reduce resident rows."
            );
        }
        self.records.insert(record.oid.clone(), record.clone());
        self.publish_len();
        Ok(record)
    }

    async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>> {
        Ok(self
            .records
            .get(&key.oid)
            .map(|record| record.value().clone()))
    }

    async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
        let removed = self.records.remove(&key.oid).is_some();
        if removed {
            self.publish_len();
        }
        Ok(removed)
    }
}

#[async_trait]
impl RecordScan for MemtableRecordStorage {
    async fn scan_records(&self, limit: usize) -> RecordStoreResult<Vec<ProximaRecord>> {
        Ok(self
            .records
            .iter()
            .take(limit)
            .map(|record| record.value().clone())
            .collect())
    }

    async fn scan_records_with_options(
        &self,
        options: RecordScanOptions,
    ) -> RecordStoreResult<Vec<ProximaRecord>> {
        let limit = options.limit.unwrap_or(usize::MAX);
        Ok(self
            .records
            .iter()
            .filter_map(|record| {
                let record = record.value();
                options.matches_record(record).then(|| record.clone())
            })
            .take(limit)
            .collect())
    }

    /// Push-down override: evaluate `options` + `predicate` during a single
    /// DashMap pass and stop at `options.limit`, cloning only matching records.
    /// Avoids the materialize-whole-table-then-filter cost the default impl pays.
    async fn scan_records_filtered(
        &self,
        options: RecordScanOptions,
        predicate: Option<&RecordScanPredicate<'_>>,
    ) -> RecordStoreResult<Vec<ProximaRecord>> {
        let limit = options.limit.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        for entry in self.records.iter() {
            if out.len() >= limit {
                break;
            }
            let record = entry.value();
            if options.matches_record(record) && predicate.is_none_or(|p| p(record)) {
                out.push(record.clone());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaTreeNode;
    use proximadb_storage_common::ProjectionDirective;

    fn record(oid: &str, tenant_id: &str) -> ProximaRecord {
        let mut record = ProximaRecord {
            oid: oid.to_string(),
            tenant_id: tenant_id.to_string(),
            ..ProximaRecord::default()
        };
        record.props.insert(
            "status".to_string(),
            ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "open".to_string(),
            )),
        );
        record
    }

    fn upsert_entry(seq: u64, record: ProximaRecord) -> CanonicalWalEntry {
        CanonicalWalEntry::new(
            seq,
            CanonicalOperation::RecordUpsert {
                collection_id: "orders".to_string(),
                record: Box::new(record),
                projections: vec![ProjectionDirective::ColumnarVariation {
                    collection_id: "orders".to_string(),
                    fields: vec!["status".to_string()],
                }],
            },
            None,
        )
    }

    fn delete_entry(seq: u64, oid: &str) -> CanonicalWalEntry {
        CanonicalWalEntry::new(
            seq,
            CanonicalOperation::RecordDelete {
                collection_id: "orders".to_string(),
                oid: oid.to_string(),
                projections: Vec::new(),
            },
            None,
        )
    }

    #[tokio::test]
    async fn memtable_record_storage_replays_canonical_wal_entries() -> Result<()> {
        let storage = MemtableRecordStorage::new();
        let summary = storage
            .replay_wal_entries(vec![
                upsert_entry(1, record("order-1", "tenant-a")),
                upsert_entry(2, record("order-2", "tenant-a")),
                delete_entry(3, "order-1"),
            ])
            .await?;

        assert_eq!(summary.upserts_replayed, 2);
        assert_eq!(summary.deletes_replayed, 1);
        assert_eq!(storage.len(), 1);
        assert!(
            storage
                .get_record(&RecordKey::new("order-1"))
                .await?
                .is_none()
        );
        assert!(
            storage
                .get_record(&RecordKey::new("order-2"))
                .await?
                .is_some()
        );

        let scanned = storage
            .scan_records_with_options(
                RecordScanOptions::unbounded()
                    .with_tenant_id("tenant-a")
                    .with_string_property("status", "open"),
            )
            .await?;
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].oid, "order-2");

        Ok(())
    }

    #[tokio::test]
    async fn scan_records_filtered_early_stops_at_limit() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let storage = MemtableRecordStorage::new();
        for i in 0..100u32 {
            storage
                .upsert_record(record(&format!("o{i:03}"), "tenant-a"))
                .await?;
        }
        // Predicate matches everything; limit is 5. The push-down must evaluate
        // the predicate only until 5 matches are collected — NOT all 100 rows.
        let calls = AtomicUsize::new(0);
        let pred = |_r: &ProximaRecord| {
            calls.fetch_add(1, Ordering::Relaxed);
            true
        };
        let got = storage
            .scan_records_filtered(RecordScanOptions::limit(5), Some(&pred))
            .await?;
        assert_eq!(got.len(), 5, "capped at limit");
        assert_eq!(
            calls.load(Ordering::Relaxed),
            5,
            "predicate evaluated only until the limit — not across the whole 100-row table"
        );
        Ok(())
    }

    #[tokio::test]
    async fn scan_records_filtered_returns_only_matching_rows() -> Result<()> {
        use proximadb_data_model::ProximaValue;
        let storage = MemtableRecordStorage::new();
        storage.upsert_record(record("keep", "tenant-a")).await?; // status=open
        let mut closed = record("drop", "tenant-a");
        closed.props.insert(
            "status".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("closed".to_string())),
        );
        storage.upsert_record(closed).await?;

        let pred = |r: &ProximaRecord| {
            matches!(
                r.props.get("status"),
                Some(ProximaTreeNode::Value(ProximaValue::String(s))) if s == "open"
            )
        };
        let got = storage
            .scan_records_filtered(RecordScanOptions::unbounded(), Some(&pred))
            .await?;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].oid, "keep");
        Ok(())
    }

    // --- TD-USUB-1 slice 1: the bound ---------------------------------------

    /// The shipped default must stay unbounded, so enabling nothing changes nothing.
    #[tokio::test]
    async fn default_memtable_is_unbounded() -> Result<()> {
        let storage = MemtableRecordStorage::with_max_records(None);
        for i in 0..64 {
            storage
                .upsert_record(record(&format!("oid-{i}"), "t"))
                .await?;
        }
        assert_eq!(storage.len(), 64);
        assert_eq!(storage.max_records(), None);
        Ok(())
    }

    /// At the cap a NEW oid is rejected — fail closed with an actionable message,
    /// rather than growing until the process is OOM-killed.
    #[tokio::test]
    async fn insert_past_cap_fails_closed() -> Result<()> {
        let storage = MemtableRecordStorage::with_max_records(Some(2));
        storage.upsert_record(record("a", "t")).await?;
        storage.upsert_record(record("b", "t")).await?;

        let err = storage
            .upsert_record(record("c", "t"))
            .await
            .expect_err("a third distinct oid must be rejected at cap 2");
        let msg = err.to_string();
        assert!(
            msg.contains("memtable is full") && msg.contains(MEMTABLE_MAX_RECORDS_ENV),
            "error must name the limit and the knob that raises it, got: {msg}"
        );

        // The rejected write must not have been applied.
        assert_eq!(storage.len(), 2);
        assert!(
            storage
                .get_record(&RecordKey::new("c".to_string()))
                .await?
                .is_none()
        );
        Ok(())
    }

    /// Updating a record already resident does not grow the set, so it must still be
    /// admitted at the cap — otherwise a full table becomes permanently
    /// un-correctable (you could not even delete rows by updating them first).
    #[tokio::test]
    async fn update_at_cap_is_admitted_and_delete_frees_room() -> Result<()> {
        let storage = MemtableRecordStorage::with_max_records(Some(2));
        storage.upsert_record(record("a", "t")).await?;
        storage.upsert_record(record("b", "t")).await?;

        // In-place update of an existing oid: allowed at cap.
        storage
            .upsert_record(record("a", "t2"))
            .await
            .expect("updating a resident oid must be admitted at cap");
        assert_eq!(storage.len(), 2);

        // Deleting frees a slot, so the previously-rejected insert now succeeds —
        // proving the bound is a live property, not a one-way latch.
        assert!(
            storage
                .delete_record(&RecordKey::new("b".to_string()))
                .await?
        );
        storage
            .upsert_record(record("c", "t"))
            .await
            .expect("a freed slot must be reusable");
        assert_eq!(storage.len(), 2);
        Ok(())
    }

    /// A typo'd or zero cap must read as "unset", never as "reject every write" —
    /// a bad env value must not silently make the database read-only.
    #[test]
    fn invalid_cap_reads_as_unbounded() {
        // SAFETY: single-threaded unit test mutating process env, restored below.
        for bad in ["0", "-1", "abc", ""] {
            unsafe { std::env::set_var(MEMTABLE_MAX_RECORDS_ENV, bad) };
            assert_eq!(
                configured_max_records(),
                None,
                "invalid cap {bad:?} must read as unbounded, not as a zero cap"
            );
        }
        unsafe { std::env::set_var(MEMTABLE_MAX_RECORDS_ENV, "500") };
        assert_eq!(configured_max_records(), Some(500));
        unsafe { std::env::remove_var(MEMTABLE_MAX_RECORDS_ENV) };
        assert_eq!(configured_max_records(), None);
    }
}
