//! In-memory fixtures for tests of canonical record-store consumers.
//!
//! Enable the non-default `test-support` feature in a dev-dependency. This
//! fixture implements only the point [`RecordStore`] contract: scans have
//! separate semantics and belong to the tests that need them.

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;

use crate::{ProximaRecord, RecordKey, RecordStore, RecordStoreResult};

/// Isolated map-backed store with replacement by canonical object id.
///
/// Writes and reads return owned copies of the supplied record, including dead
/// records. Deletion physically removes the entry and reports whether it existed.
/// This fixture provides neither persistence nor WAL sequence tracking.
#[derive(Default)]
pub struct MemoryRecordStore {
    // Foundation's scan-contract tests retain their own scan implementation.
    // Downstream consumers use the point port rather than accessing the map.
    pub(crate) records: RwLock<HashMap<String, ProximaRecord>>,
}

impl MemoryRecordStore {
    /// Remove all records, for tests that reset state before recovery replay.
    pub fn clear(&self) -> RecordStoreResult<()> {
        self.records
            .write()
            .map_err(|_| anyhow::anyhow!("memory record store write lock poisoned"))?
            .clear();
        Ok(())
    }

    /// Whether the fixture currently contains no records.
    pub fn is_empty(&self) -> RecordStoreResult<bool> {
        Ok(self
            .records
            .read()
            .map_err(|_| anyhow::anyhow!("memory record store read lock poisoned"))?
            .is_empty())
    }
}

#[async_trait]
impl RecordStore for MemoryRecordStore {
    async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord> {
        self.records
            .write()
            .map_err(|_| anyhow::anyhow!("memory record store write lock poisoned"))?
            .insert(record.oid.clone(), record.clone());
        Ok(record)
    }

    async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>> {
        Ok(self
            .records
            .read()
            .map_err(|_| anyhow::anyhow!("memory record store read lock poisoned"))?
            .get(&key.oid)
            .cloned())
    }

    async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
        Ok(self
            .records
            .write()
            .map_err(|_| anyhow::anyhow!("memory record store write lock poisoned"))?
            .remove(&key.oid)
            .is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProximaTreeNode, ProximaValue};
    use std::sync::Arc;

    #[tokio::test]
    async fn point_port_preserves_payload_replacement_and_isolation() -> RecordStoreResult<()> {
        let store: &dyn RecordStore = &MemoryRecordStore::default();
        let separate: &dyn RecordStore = &MemoryRecordStore::default();
        let key = RecordKey::new("same-id");
        assert!(store.get_record(&key).await?.is_none());
        assert!(!store.delete_record(&key).await?);

        let mut original = ProximaRecord {
            oid: key.oid.clone(),
            tenant_id: "tenant-a".into(),
            record_version: 4,
            ..ProximaRecord::default()
        };
        original.labels.insert("document");
        original.props.insert(
            "name".into(),
            ProximaTreeNode::Value(ProximaValue::String("Ada".into())),
        );
        let mut returned = store.upsert_record(original.clone()).await?;
        assert_eq!(returned, original);
        returned.props.clear();
        assert_eq!(store.get_record(&key).await?, Some(original.clone()));
        assert!(separate.get_record(&key).await?.is_none());

        // A replacement removes the old payload rather than merging it.
        let replacement = ProximaRecord {
            oid: key.oid.clone(),
            tenant_id: "tenant-b".into(),
            record_version: 5,
            ..ProximaRecord::default()
        };
        assert_eq!(store.upsert_record(replacement.clone()).await?, replacement);
        let mut fetched = store.get_record(&key).await?.expect("replacement exists");
        assert_eq!(fetched, replacement);
        fetched.tenant_id.clear();
        assert_eq!(store.get_record(&key).await?, Some(replacement));
        assert!(store.delete_record(&key).await?);
        assert!(!store.delete_record(&key).await?);
        assert!(store.get_record(&key).await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn point_port_returns_dead_records_and_ordered_batch_slots() -> RecordStoreResult<()> {
        let store: &dyn RecordStore = &MemoryRecordStore::default();
        let tombstone = ProximaRecord::tombstone("dead", 1);
        let expired = ProximaRecord {
            oid: "expired".into(),
            valid_to_ns: Some(1),
            ..ProximaRecord::default()
        };
        store
            .upsert_records(vec![tombstone.clone(), expired.clone()])
            .await?;
        assert_eq!(
            store
                .get_records(&[
                    RecordKey::new("expired"),
                    RecordKey::new("missing"),
                    RecordKey::new("dead"),
                    RecordKey::new("expired"),
                ])
                .await?,
            vec![
                Some(expired.clone()),
                None,
                Some(tombstone.clone()),
                Some(expired)
            ]
        );
        assert_eq!(
            store
                .get_record_including_dead(&RecordKey::new("dead"))
                .await?,
            Some(tombstone)
        );
        Ok(())
    }

    #[tokio::test]
    async fn point_port_reports_poisoned_locks_as_errors() {
        let store = Arc::new(MemoryRecordStore::default());
        let poison = Arc::clone(&store);
        assert!(
            std::thread::spawn(move || {
                let _guard = poison.records.write().expect("initial lock");
                panic!("poison fixture lock for error-path test");
            })
            .join()
            .is_err()
        );

        let port: &dyn RecordStore = store.as_ref();
        assert_eq!(
            port.get_record(&RecordKey::new("id"))
                .await
                .expect_err("poisoned read")
                .to_string(),
            "memory record store read lock poisoned"
        );
        assert_eq!(
            port.upsert_record(ProximaRecord::default())
                .await
                .expect_err("poisoned upsert")
                .to_string(),
            "memory record store write lock poisoned"
        );
        assert_eq!(
            port.delete_record(&RecordKey::new("id"))
                .await
                .expect_err("poisoned delete")
                .to_string(),
            "memory record store write lock poisoned"
        );
        assert!(store.clear().is_err());
        assert!(store.is_empty().is_err());
    }

    #[tokio::test]
    async fn reset_clears_all_records_before_replay() -> RecordStoreResult<()> {
        let store = MemoryRecordStore::default();
        let port: &dyn RecordStore = &store;
        assert!(store.is_empty()?);
        port.upsert_records(vec![
            ProximaRecord {
                oid: "first".into(),
                ..ProximaRecord::default()
            },
            ProximaRecord {
                oid: "second".into(),
                ..ProximaRecord::default()
            },
        ])
        .await?;
        assert!(!store.is_empty()?);
        store.clear()?;
        store.clear()?;
        assert!(store.is_empty()?);
        assert_eq!(
            port.get_records(&[RecordKey::new("first"), RecordKey::new("second")])
                .await?,
            vec![None, None]
        );
        port.upsert_record(ProximaRecord {
            oid: "first".into(),
            ..ProximaRecord::default()
        })
        .await?;
        assert!(!store.is_empty()?);
        assert!(port.get_record(&RecordKey::new("first")).await?.is_some());
        Ok(())
    }
}
