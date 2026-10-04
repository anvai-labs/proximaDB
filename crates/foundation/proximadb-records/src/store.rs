//! Canonical `ProximaRecord` store contracts.
//!
//! These traits describe the durable record spine used by modality facades.
//! Document, graph, vector, observability, SKS/entity, and event services can
//! adapt to these contracts without owning separate record envelopes or
//! modality-specific WAL/recovery semantics.

use async_trait::async_trait;

use crate::{ProximaRecord, ProximaTreeNode};
use proximadb_data_model::ProximaValue;

/// Result type for canonical record-store operations.
pub type RecordStoreResult<T> = anyhow::Result<T>;

/// Key used to address one canonical record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordKey {
    /// Canonical record object id.
    pub oid: String,
}

impl RecordKey {
    pub fn new(oid: impl Into<String>) -> Self {
        Self { oid: oid.into() }
    }
}

impl From<&ProximaRecord> for RecordKey {
    fn from(record: &ProximaRecord) -> Self {
        Self::new(record.oid.clone())
    }
}

/// Batch write result for canonical record operations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordWriteResult {
    /// Number of records accepted by the durable store.
    pub records_written: usize,
    /// Canonical ids written by the operation.
    pub record_oids: Vec<String>,
}

/// Canonical record-store operation recovered from a durable log.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordRecoveryOperation {
    /// Replay an insert/update as the authoritative record state.
    Upsert(Box<ProximaRecord>),
    /// Replay a delete by canonical object id.
    Delete(RecordKey),
}

/// Summary of canonical record-store recovery replay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordRecoverySummary {
    /// Number of recovered upserts applied to the record store.
    pub upserts_replayed: usize,
    /// Number of recovered deletes applied to the record store.
    pub deletes_replayed: usize,
}

/// Borrowed row predicate for scan push-down.
///
/// Explicitly higher-ranked over the record reference (`for<'r>`) so it composes
/// with `#[async_trait]` scan methods — without the explicit HRTB, async_trait
/// hoists the elided lifetime to the method's lifetime and a short-lived
/// `&ProximaRecord` can no longer be passed. `Send + Sync` keeps the async
/// future `Send`.
pub type RecordScanPredicate<'a> = dyn for<'r> Fn(&'r ProximaRecord) -> bool + Send + Sync + 'a;

/// Predicate/options accepted by canonical record scan implementations.
///
/// This remains modality-neutral: document/graph/vector facades express their
/// scan needs as canonical labels, properties, and RLS fields instead of adding
/// separate durable query contracts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordScanOptions {
    /// Maximum number of records to return. `None` means unbounded.
    pub limit: Option<usize>,
    /// Required label on the canonical record, if any.
    pub required_label: Option<String>,
    /// Required owning tenant, if any.
    pub tenant_id: Option<String>,
    /// Required canonical property values.
    pub properties: Vec<(String, ProximaValue)>,
}

impl RecordScanOptions {
    pub fn limit(limit: usize) -> Self {
        Self {
            limit: Some(limit),
            ..Self::default()
        }
    }

    pub fn unbounded() -> Self {
        Self::default()
    }

    pub fn with_required_label(mut self, label: impl Into<String>) -> Self {
        self.required_label = Some(label.into());
        self
    }

    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn with_property(mut self, key: impl Into<String>, value: ProximaValue) -> Self {
        self.properties.push((key.into(), value));
        self
    }

    pub fn with_string_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.properties
            .push((key.into(), ProximaValue::String(value.into())));
        self
    }

    pub fn matches_record(&self, record: &ProximaRecord) -> bool {
        // Dead-record filter (defense-in-depth, applies to EVERY scan using
        // RecordScanOptions): a tombstone (valid_to_ns == Some(0)) or
        // TTL-expired (valid_to_ns in the past) record must never surface.
        // Uses the canonical ProximaRecord::is_visible_at on valid_to_ns
        // (ns) — never the unit-muddled expires_at.
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        if !record.is_visible_at(now_ns) {
            return false;
        }

        if let Some(label) = &self.required_label
            && !record.labels.contains(label)
        {
            return false;
        }

        if let Some(tenant_id) = &self.tenant_id
            && &record.tenant_id != tenant_id
        {
            return false;
        }

        self.properties.iter().all(|(key, expected)| {
            matches!(
                record.props.get(key),
                Some(ProximaTreeNode::Value(actual)) if actual == expected
            )
        })
    }
}

/// Narrow canonical store contract over `ProximaRecord`.
///
/// This is intentionally modality-neutral. It does not describe document JSON
/// paths, graph adjacency, vector ANN, or observability rollups; those are
/// facades/projections layered above this trait.
#[async_trait]
pub trait RecordStore: Send + Sync {
    async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord>;

    /// Upsert a record whose durable WAL sequence number (LSN) is known.
    ///
    /// The default **ignores the LSN** and delegates to [`Self::upsert_record`],
    /// so no existing implementor is affected. Only a store that must reason
    /// about WAL retention needs to override it.
    ///
    /// # Why this exists, and why it is not a watermark
    ///
    /// ADR-094's WAL truncation rule permits truncation only below the minimum
    /// LSN durably committed in a referenced segment. Computing that requires
    /// knowing which LSN each *resident* record came from — not a single
    /// high-water mark. A high-water taken at flush time is **unsafe**: the WAL
    /// append covers a whole batch and completes before the per-record inserts
    /// run, so a flush triggered partway through a multi-row statement produces
    /// a segment that excludes later rows whose LSNs are already committed.
    /// Truncating below such a mark would discard the only durable copy. See
    /// TD-USUB-1.
    ///
    /// Passing an LSN therefore records an association, not a bound: the store
    /// keeps `oid → lsn` for rows it still holds, and the safe truncation point
    /// is the minimum over that set.
    ///
    /// The caller must only pass an LSN whose WAL entry is already **durable**.
    /// `TableWalAppender::append_operations` returns its entries after
    /// `sync_data()`, so sequence numbers taken from its result satisfy that.
    async fn upsert_record_at_lsn(
        &self,
        record: ProximaRecord,
        lsn: u64,
    ) -> RecordStoreResult<ProximaRecord> {
        let _ = lsn;
        self.upsert_record(record).await
    }

    /// The lowest WAL LSN this store still needs retained — i.e. the minimum LSN
    /// among records it holds that are **not** yet in a durable segment.
    ///
    /// `None` means "nothing to retain on my account", which is the default for
    /// every store that does not track LSNs. A caller computing a truncation
    /// point must treat `None` as *no constraint from this store* and must still
    /// respect every other store's answer.
    ///
    /// Deliberately not called `flushed_up_to`: this is a minimum over an
    /// unordered resident set, not a contiguous prefix.
    async fn min_unflushed_lsn(&self) -> RecordStoreResult<Option<u64>> {
        Ok(None)
    }

    async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>>;

    /// Batch point-lookup: fetch many records by key in one logical operation,
    /// returning one slot per input key, in order (`None` = absent).
    ///
    /// The default loops [`Self::get_record`] (correct, no I/O saving). Stores
    /// backed by object storage SHOULD override to issue the independent gets
    /// **concurrently**, so K point-lookups cost ~one round-trip of latency
    /// instead of K serial RTTs — the depth-collapse for result/frontier
    /// materialization (ADR-034 P1: latency = depth × RTT). Note this collapses
    /// *latency/depth*, not the op count (distinct objects ⇒ still K GETs).
    async fn get_records(
        &self,
        keys: &[RecordKey],
    ) -> RecordStoreResult<Vec<Option<ProximaRecord>>> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            out.push(self.get_record(key).await?);
        }
        Ok(out)
    }

    async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool>;

    async fn upsert_records(
        &self,
        records: Vec<ProximaRecord>,
    ) -> RecordStoreResult<RecordWriteResult> {
        let mut record_oids = Vec::with_capacity(records.len());

        for record in records {
            let written = self.upsert_record(record).await?;
            record_oids.push(written.oid);
        }

        Ok(RecordWriteResult {
            records_written: record_oids.len(),
            record_oids,
        })
    }

    /// Flush any in-memory write buffer to durable storage. The default is a no-op:
    /// durable-on-write stores (one object PUT per record) have nothing to flush.
    /// A **buffered** store (e.g. a segment store that batches records into one
    /// object) MUST override this so callers can force its buffer durable at a
    /// checkpoint / graceful-shutdown boundary — otherwise an unflushed buffer is
    /// lost on crash. Called from the graph checkpoint/shutdown path (`flush_wal`).
    async fn flush(&self) -> RecordStoreResult<()> {
        Ok(())
    }

    /// Permanently delete the durable objects this store owns, for a DROP of the
    /// partition it backs. The default is a no-op: an in-memory store owns no
    /// objects, so releasing its handle already is complete deletion.
    ///
    /// A store that writes objects MUST override this, because releasing the
    /// in-memory handle does **not** delete them. Without it a `DROP TABLE`
    /// leaves every written object in the bucket with nothing referencing it —
    /// billable storage the tenant can neither see nor reach, under a statement
    /// whose whole contract is deletion.
    ///
    /// Must be **idempotent** (re-dropping, or dropping a partition that never
    /// wrote anything, is not an error) and must **propagate real I/O failures**
    /// rather than reporting success — silently failing here leaves undeletable
    /// data.
    async fn purge_durable_objects(&self) -> RecordStoreResult<()> {
        Ok(())
    }
}

/// Optional scan contract for stores that can expose canonical record ranges.
///
/// Keep this separate from `RecordStore` so point-write stores can implement
/// the minimal durable contract before scan/query planning is extracted.
#[async_trait]
pub trait RecordScan: Send + Sync {
    async fn scan_records(&self, limit: usize) -> RecordStoreResult<Vec<ProximaRecord>>;

    async fn scan_records_with_options(
        &self,
        options: RecordScanOptions,
    ) -> RecordStoreResult<Vec<ProximaRecord>> {
        let mut records = self
            .scan_records(options.limit.unwrap_or(usize::MAX))
            .await?;
        records.retain(|record| options.matches_record(record));
        Ok(records)
    }

    /// Scan with an arbitrary row `predicate` pushed in alongside `options`.
    ///
    /// Returns up to `options.limit` records that match BOTH `options`
    /// (label/tenant/property) AND `predicate`. The store is responsible for
    /// applying `predicate`, so callers must not re-filter afterward — this lets
    /// hot implementations evaluate the predicate during iteration and stop at
    /// the limit instead of materializing the whole table first.
    ///
    /// The predicate is a borrowed trait object (not stored in the
    /// `Clone`/`PartialEq` `RecordScanOptions`, and not a catalog/SQL IR — this
    /// crate stays modality- and query-language-neutral). `Send + Sync` so the
    /// `#[async_trait]` future stays `Send`.
    ///
    /// Default: correct but not early-stopping — scans all `options`-matching
    /// records, applies `predicate`, then caps at the limit. Hot stores override.
    async fn scan_records_filtered(
        &self,
        options: RecordScanOptions,
        predicate: Option<&RecordScanPredicate<'_>>,
    ) -> RecordStoreResult<Vec<ProximaRecord>> {
        let limit = options.limit.unwrap_or(usize::MAX);
        let mut opts = options;
        opts.limit = None;
        let mut all = self.scan_records_with_options(opts).await?;
        let mut kept = 0usize;
        all.retain(|record| {
            if kept >= limit {
                return false;
            }
            let keep = predicate.is_none_or(|p| p(record));
            if keep {
                kept += 1;
            }
            keep
        });
        Ok(all)
    }
}

/// Composite canonical storage contract for services that need both point
/// operations and scans.
///
/// Keep document/graph/vector semantics out of this trait. Facades filter and
/// project records after scanning, while the durable contract remains a shared
/// `ProximaRecord` spine.
pub trait RecordStorage: RecordStore + RecordScan {}

impl<T> RecordStorage for T where T: RecordStore + RecordScan + ?Sized {}

/// Replay recovered canonical record operations into a `RecordStore`.
///
/// This is the shared record-store recovery hook used by modality facades while
/// their API-specific WAL shapes migrate toward the canonical WAL envelope.
pub async fn replay_record_recovery_operations<S, I>(
    store: &S,
    operations: I,
) -> RecordStoreResult<RecordRecoverySummary>
where
    S: RecordStore + ?Sized,
    I: IntoIterator<Item = RecordRecoveryOperation>,
{
    let mut summary = RecordRecoverySummary::default();

    for operation in operations {
        match operation {
            RecordRecoveryOperation::Upsert(record) => {
                store.upsert_record(*record).await?;
                summary.upserts_replayed += 1;
            }
            RecordRecoveryOperation::Delete(key) => {
                store.delete_record(&key).await?;
                summary.deletes_replayed += 1;
            }
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::RwLock;

    #[test]
    fn matches_record_excludes_tombstone_and_expired() {
        // A live record passes the unbounded scan filter.
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        let live = ProximaRecord {
            oid: "alive".into(),
            ..ProximaRecord::default()
        };
        assert!(
            RecordScanOptions::unbounded().matches_record(&live),
            "live record matches"
        );

        // A delete tombstone (valid_to_ns == 0) is excluded.
        let tombstone = ProximaRecord::tombstone("dead", now_ns);
        assert!(
            !RecordScanOptions::unbounded().matches_record(&tombstone),
            "tombstone excluded from scan"
        );

        // A TTL-expired record (valid_to_ns in the past) is excluded.
        let expired = ProximaRecord {
            oid: "stale".into(),
            valid_to_ns: Some(now_ns - 1_000_000_000),
            ..ProximaRecord::default()
        };
        assert!(
            !RecordScanOptions::unbounded().matches_record(&expired),
            "TTL-expired record excluded from scan"
        );
    }

    #[derive(Default)]
    struct MemoryRecordStore {
        records: RwLock<HashMap<String, ProximaRecord>>,
    }

    #[async_trait]
    impl RecordStore for MemoryRecordStore {
        async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord> {
            self.records
                .write()
                .expect("memory record store write lock")
                .insert(record.oid.clone(), record.clone());
            Ok(record)
        }

        async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>> {
            Ok(self
                .records
                .read()
                .expect("memory record store read lock")
                .get(&key.oid)
                .cloned())
        }

        async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
            Ok(self
                .records
                .write()
                .expect("memory record store write lock")
                .remove(&key.oid)
                .is_some())
        }
    }

    #[async_trait]
    impl RecordScan for MemoryRecordStore {
        async fn scan_records(&self, limit: usize) -> RecordStoreResult<Vec<ProximaRecord>> {
            Ok(self
                .records
                .read()
                .expect("memory record store read lock")
                .values()
                .take(limit)
                .cloned()
                .collect())
        }

        async fn scan_records_with_options(
            &self,
            options: RecordScanOptions,
        ) -> RecordStoreResult<Vec<ProximaRecord>> {
            let mut records = Vec::new();

            for record in self
                .records
                .read()
                .expect("memory record store read lock")
                .values()
            {
                if options.matches_record(record) {
                    records.push(record.clone());
                    if records.len() >= options.limit.unwrap_or(usize::MAX) {
                        break;
                    }
                }
            }

            Ok(records)
        }
    }

    #[tokio::test]
    async fn record_store_contract_supports_point_lifecycle() {
        let store = MemoryRecordStore::default();
        let record = ProximaRecord {
            oid: "doc-1".to_string(),
            ..ProximaRecord::default()
        };

        let key = RecordKey::from(&record);
        let written = store.upsert_record(record).await.expect("upsert");
        assert_eq!(written.oid, "doc-1");

        let fetched = store.get_record(&key).await.expect("get");
        assert_eq!(fetched.map(|record| record.oid), Some("doc-1".to_string()));

        assert!(store.delete_record(&key).await.expect("delete"));
        assert!(store.get_record(&key).await.expect("get deleted").is_none());
    }

    #[tokio::test]
    async fn record_store_default_batch_upsert_reports_written_ids() {
        let store = MemoryRecordStore::default();
        let records = vec![
            ProximaRecord {
                oid: "r1".to_string(),
                ..ProximaRecord::default()
            },
            ProximaRecord {
                oid: "r2".to_string(),
                ..ProximaRecord::default()
            },
        ];

        let result = store.upsert_records(records).await.expect("batch upsert");
        assert_eq!(result.records_written, 2);
        assert_eq!(result.record_oids, vec!["r1", "r2"]);
    }

    #[tokio::test]
    async fn record_storage_composes_point_and_scan_contracts() {
        let store = MemoryRecordStore::default();
        let storage: &dyn RecordStorage = &store;

        storage
            .upsert_record(ProximaRecord {
                oid: "r1".to_string(),
                ..ProximaRecord::default()
            })
            .await
            .expect("upsert");

        let records = storage.scan_records(10).await.expect("scan");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].oid, "r1");
    }

    #[tokio::test]
    async fn scan_records_filtered_applies_predicate_and_caps_at_limit() {
        let store = MemoryRecordStore::default();
        for i in 0..6u64 {
            store
                .upsert_record(ProximaRecord {
                    oid: format!("r{i}"),
                    record_version: i,
                    ..ProximaRecord::default()
                })
                .await
                .expect("upsert");
        }

        // Predicate selects even record_version (r0,r2,r4 = 3 matches); limit 2.
        let pred = |r: &ProximaRecord| r.record_version.is_multiple_of(2);
        let got = store
            .scan_records_filtered(RecordScanOptions::limit(2), Some(&pred))
            .await
            .expect("filtered scan");
        assert_eq!(got.len(), 2, "capped at limit");
        assert!(
            got.iter().all(|r| r.record_version % 2 == 0),
            "only predicate matches returned"
        );

        // No predicate behaves like a plain limited scan.
        let none = store
            .scan_records_filtered(RecordScanOptions::limit(10), None)
            .await
            .expect("no-predicate scan");
        assert_eq!(none.len(), 6);
    }

    #[tokio::test]
    async fn scan_options_filter_by_label_tenant_and_proxima_property() {
        let store = MemoryRecordStore::default();
        let mut matching = ProximaRecord {
            oid: "doc-1".to_string(),
            tenant_id: "tenant-a".to_string(),
            ..ProximaRecord::default()
        };
        matching.labels.insert("document");
        matching.props.insert(
            "_document_collection".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("docs".to_string())),
        );

        let mut other_collection = matching.clone();
        other_collection.oid = "doc-2".to_string();
        other_collection.props.insert(
            "_document_collection".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("other".to_string())),
        );

        let mut other_tenant = matching.clone();
        other_tenant.oid = "doc-3".to_string();
        other_tenant.tenant_id = "tenant-b".to_string();

        store.upsert_record(matching).await.expect("matching");
        store
            .upsert_record(other_collection)
            .await
            .expect("other collection");
        store
            .upsert_record(other_tenant)
            .await
            .expect("other tenant");

        let records = store
            .scan_records_with_options(
                RecordScanOptions::unbounded()
                    .with_required_label("document")
                    .with_tenant_id("tenant-a")
                    .with_string_property("_document_collection", "docs"),
            )
            .await
            .expect("filtered scan");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].oid, "doc-1");
    }

    #[tokio::test]
    async fn recovery_replay_applies_upserts_and_deletes_in_order() {
        let store = MemoryRecordStore::default();
        let summary = replay_record_recovery_operations(
            &store,
            vec![
                RecordRecoveryOperation::Upsert(Box::new(ProximaRecord {
                    oid: "r1".to_string(),
                    ..ProximaRecord::default()
                })),
                RecordRecoveryOperation::Upsert(Box::new(ProximaRecord {
                    oid: "r2".to_string(),
                    ..ProximaRecord::default()
                })),
                RecordRecoveryOperation::Delete(RecordKey::new("r1")),
            ],
        )
        .await
        .expect("recovery replay");

        assert_eq!(summary.upserts_replayed, 2);
        assert_eq!(summary.deletes_replayed, 1);
        assert!(
            store
                .get_record(&RecordKey::new("r1"))
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .get_record(&RecordKey::new("r2"))
                .await
                .unwrap()
                .is_some()
        );
    }
}
