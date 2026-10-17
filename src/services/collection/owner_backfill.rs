// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Backfill collection owner: record an owner on collections created before
//! ownership was recorded.
//!
//! An unattributed collection belongs to the default tenant
//! (`proximadb_tenant::ownership`), which is correct for single-tenant
//! deployments. Collections a non-default tenant created before ownership was
//! recorded are assigned that tenant here, inferred from the tenant every
//! record is stamped with at write.
//!
//! Safety properties:
//! * **Never re-assigns.** Only unattributed collections are considered, and
//!   the write itself re-checks under the catalog's read-modify-write lock
//!   (`assign_owner_if_unattributed`), so a collection that gained an owner
//!   meanwhile is left alone.
//! * **Idempotent / safe to interrupt.** Each collection is one independent
//!   catalog write; a re-run skips everything already assigned.
//! * **Bounded inspection.** At most `sample_cap` records per collection, read
//!   from unflushed records first; flushed storage is read only when the
//!   collection's recorded size is within [`STORAGE_SAMPLE_MAX_RECORDS`].
//!   Larger (or unsized) collections are left to the default tenant.
//! * **Decided once.** A collection the backfill cannot attribute (records
//!   from several tenants, or too large to sample) is marked with an
//!   [`OWNER_BACKFILL_TAG`] tag, logged once, and skipped by later runs.
//! * **Fail closed.** Every undecided or failed case leaves the collection
//!   owned by the default tenant only.
//! * **Non-fatal.** Any failure is logged; the run continues.

use std::collections::BTreeSet;

use anyhow::Result;
use async_trait::async_trait;

/// Records inspected per collection.
pub const DEFAULT_SAMPLE_CAP: usize = 1_000;

/// Largest recorded collection size whose flushed storage the backfill reads
/// to sample record tenants (storage engines expose whole-collection reads).
pub const STORAGE_SAMPLE_MAX_RECORDS: i64 = 10_000;

/// Tag prefix recording that the backfill could not attribute a collection;
/// later runs skip it. Removing the tag has it re-examined.
pub const OWNER_BACKFILL_TAG: &str = "owner_backfill:";

/// What the sampled record tenants say about a collection's owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnerInference {
    /// Exactly one tenant (or none: no records / only unlabelled ones ⇒ the
    /// default tenant).
    Assign(String),
    /// Records from several tenants: leave the collection unattributed.
    Ambiguous(BTreeSet<String>),
}

/// Infer the owner from the set of tenants seen on sampled records.
pub fn infer_owner(sampled_tenants: &BTreeSet<String>) -> OwnerInference {
    let tenants: BTreeSet<String> = sampled_tenants
        .iter()
        .map(|tenant| tenant.trim())
        .filter(|tenant| !tenant.is_empty())
        .map(str::to_string)
        .collect();
    if tenants.len() > 1 {
        return OwnerInference::Ambiguous(tenants);
    }
    OwnerInference::Assign(
        tenants
            .into_iter()
            .next()
            .unwrap_or_else(|| proximadb_tenant::DEFAULT_TENANT.to_string()),
    )
}

/// What sampling a collection's records produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordSample {
    /// The distinct tenants seen on up to `cap` records.
    Tenants(BTreeSet<String>),
    /// The collection is too large (or of unknown size) to sample.
    TooLarge,
}

/// Source of the tenants recorded on a collection's records.
#[async_trait]
pub trait RecordTenantSampler: Send + Sync {
    /// Distinct `tenant_id`s of up to `cap` records of `collection_id`.
    async fn sample_record_tenants(&self, collection_id: &str, cap: usize) -> Result<RecordSample>;
}

/// The catalog side of the backfill.
#[async_trait]
pub trait CollectionOwnerStore: Send + Sync {
    /// Ids of collections that record no owner and are not yet marked.
    async fn unattributed_collections(&self) -> Result<Vec<String>>;
    /// Record `tenant` as owner unless the collection already records one.
    async fn assign_owner_if_unattributed(&self, collection_id: &str, tenant: &str)
    -> Result<bool>;
    /// Record that the backfill could not attribute the collection (it stays
    /// owned by the default tenant) so later runs skip it.
    async fn mark_unattributable(&self, collection_id: &str, reason: &str) -> Result<()>;
}

/// Outcome counts of one backfill run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BackfillReport {
    pub assigned: usize,
    pub ambiguous: usize,
    pub too_large: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Run the backfill once over every unattributed collection.
pub async fn backfill_collection_owners(
    store: &dyn CollectionOwnerStore,
    sampler: &dyn RecordTenantSampler,
    sample_cap: usize,
) -> BackfillReport {
    let mut report = BackfillReport::default();
    let collections = match store.unattributed_collections().await {
        Ok(collections) => collections,
        Err(error) => {
            tracing::error!(%error, "collection owner backfill: listing collections failed");
            report.failed += 1;
            return report;
        }
    };
    for collection_id in collections {
        let sampled = match sampler
            .sample_record_tenants(&collection_id, sample_cap.max(1))
            .await
        {
            Ok(RecordSample::Tenants(sampled)) => sampled,
            Ok(RecordSample::TooLarge) => {
                tracing::error!(
                    collection = %collection_id,
                    "collection owner backfill: collection too large to sample; it stays \
                     owned by the default tenant until its owner is recorded"
                );
                mark(store, &collection_id, "unsampled").await;
                report.too_large += 1;
                continue;
            }
            Err(error) => {
                tracing::warn!(
                    collection = %collection_id,
                    %error,
                    "collection owner backfill: sampling records failed; the collection stays \
                     owned by the default tenant"
                );
                report.failed += 1;
                continue;
            }
        };
        match infer_owner(&sampled) {
            OwnerInference::Assign(tenant) => {
                match store
                    .assign_owner_if_unattributed(&collection_id, &tenant)
                    .await
                {
                    Ok(true) => {
                        tracing::info!(
                            collection = %collection_id,
                            owner = %tenant,
                            "collection owner backfill: owner recorded"
                        );
                        report.assigned += 1;
                    }
                    Ok(false) => report.skipped += 1,
                    Err(error) => {
                        tracing::warn!(
                            collection = %collection_id,
                            %error,
                            "collection owner backfill: recording the owner failed; the \
                             collection stays owned by the default tenant"
                        );
                        report.failed += 1;
                    }
                }
            }
            OwnerInference::Ambiguous(tenants) => {
                tracing::error!(
                    collection = %collection_id,
                    tenants = ?tenants,
                    "collection owner backfill: records belong to several tenants; the \
                     collection stays owned by the default tenant until its owner is recorded"
                );
                mark(store, &collection_id, "ambiguous").await;
                report.ambiguous += 1;
            }
        }
    }
    report
}

async fn mark(store: &dyn CollectionOwnerStore, collection_id: &str, reason: &str) {
    if let Err(error) = store.mark_unattributable(collection_id, reason).await {
        tracing::warn!(
            collection = %collection_id,
            %error,
            "collection owner backfill: recording the outcome failed"
        );
    }
}

#[async_trait]
impl CollectionOwnerStore for crate::services::collection::manager::CollectionService {
    async fn unattributed_collections(&self) -> Result<Vec<String>> {
        Ok(self
            .list_collections()
            .await?
            .into_iter()
            .filter(|collection| {
                proximadb_tenant::collection_owner(collection)
                    == proximadb_tenant::CollectionOwner::Unattributed
                    && !collection.config.as_ref().is_some_and(|config| {
                        config
                            .tags
                            .iter()
                            .any(|tag| tag.starts_with(OWNER_BACKFILL_TAG))
                    })
            })
            .map(|collection| collection.id)
            .collect())
    }

    async fn assign_owner_if_unattributed(
        &self,
        collection_id: &str,
        tenant: &str,
    ) -> Result<bool> {
        crate::services::collection::manager::CollectionService::assign_owner_if_unattributed(
            self,
            collection_id,
            tenant,
        )
        .await
    }

    async fn mark_unattributable(&self, collection_id: &str, reason: &str) -> Result<()> {
        crate::services::collection::manager::CollectionService::mark_owner_unattributable(
            self,
            collection_id,
            &format!("{OWNER_BACKFILL_TAG}{reason}"),
        )
        .await
    }
}

#[async_trait]
impl RecordTenantSampler for crate::services::operations::vectors::VectorOperationsService {
    async fn sample_record_tenants(&self, collection_id: &str, cap: usize) -> Result<RecordSample> {
        // Unflushed records first: in memory, and capped here.
        let unflushed = self
            .sample_unflushed_records_internal(collection_id, cap)
            .await?;
        if !unflushed.is_empty() {
            return Ok(RecordSample::Tenants(
                unflushed
                    .into_iter()
                    .map(|record| record.tenant_id)
                    .collect(),
            ));
        }
        // Flushed storage is read whole by the engines, so only within a
        // recorded size bound; unknown or larger ⇒ not sampled.
        let within_bound = self
            .recorded_collection_size_internal(collection_id)
            .await?
            .is_some_and(|count| count <= STORAGE_SAMPLE_MAX_RECORDS);
        if !within_bound {
            return Ok(RecordSample::TooLarge);
        }
        let records = self.list_all_records_internal(collection_id).await?;
        Ok(RecordSample::Tenants(
            records
                .into_iter()
                .take(cap)
                .map(|record| record.tenant_id)
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn inference_branches() {
        assert_eq!(
            infer_owner(&set(&["acme"])),
            OwnerInference::Assign("acme".into())
        );
        assert_eq!(
            infer_owner(&set(&["default"])),
            OwnerInference::Assign("default".into())
        );
        assert_eq!(
            infer_owner(&set(&[])),
            OwnerInference::Assign("default".into())
        );
        assert_eq!(
            infer_owner(&set(&["", "  "])),
            OwnerInference::Assign("default".into())
        );
        assert_eq!(
            infer_owner(&set(&["acme", " acme "])),
            OwnerInference::Assign("acme".into())
        );
        assert!(matches!(
            infer_owner(&set(&["acme", "globex"])),
            OwnerInference::Ambiguous(t) if t.len() == 2
        ));
        assert!(matches!(
            infer_owner(&set(&["acme", "default"])),
            OwnerInference::Ambiguous(_)
        ));
    }

    /// In-memory catalog: collection id → (recorded owner, marked reason).
    struct FakeStore {
        entries: Mutex<HashMap<String, (Option<String>, Option<String>)>>,
        writes: Mutex<usize>,
    }

    impl FakeStore {
        fn new(entries: &[(&str, Option<&str>)]) -> Self {
            Self {
                entries: Mutex::new(
                    entries
                        .iter()
                        .map(|(id, owner)| (id.to_string(), (owner.map(str::to_string), None)))
                        .collect(),
                ),
                writes: Mutex::new(0),
            }
        }
        fn owner(&self, id: &str) -> Option<String> {
            self.entries
                .lock()
                .ok()
                .and_then(|entries| entries.get(id).and_then(|(owner, _)| owner.clone()))
        }
        fn marked(&self, id: &str) -> Option<String> {
            self.entries
                .lock()
                .ok()
                .and_then(|entries| entries.get(id).and_then(|(_, mark)| mark.clone()))
        }
        fn writes(&self) -> usize {
            self.writes.lock().map(|w| *w).unwrap_or(0)
        }
    }

    #[async_trait]
    impl CollectionOwnerStore for FakeStore {
        async fn unattributed_collections(&self) -> Result<Vec<String>> {
            let entries = self
                .entries
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?;
            let mut ids: Vec<String> = entries
                .iter()
                .filter(|(_, (owner, mark))| owner.is_none() && mark.is_none())
                .map(|(id, _)| id.clone())
                .collect();
            ids.sort();
            Ok(ids)
        }
        async fn assign_owner_if_unattributed(&self, id: &str, tenant: &str) -> Result<bool> {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?;
            match entries.get_mut(id) {
                Some((slot @ None, _)) => {
                    *slot = Some(tenant.to_string());
                    *self
                        .writes
                        .lock()
                        .map_err(|_| anyhow::anyhow!("poisoned"))? += 1;
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
        async fn mark_unattributable(&self, id: &str, reason: &str) -> Result<()> {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?;
            if let Some((None, mark)) = entries.get_mut(id) {
                *mark = Some(reason.to_string());
                *self
                    .writes
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned"))? += 1;
            }
            Ok(())
        }
    }

    struct FakeSampler(HashMap<String, Result<RecordSample, String>>);

    #[async_trait]
    impl RecordTenantSampler for FakeSampler {
        async fn sample_record_tenants(&self, id: &str, _cap: usize) -> Result<RecordSample> {
            match self.0.get(id) {
                Some(Ok(sample)) => Ok(sample.clone()),
                Some(Err(e)) => Err(anyhow::anyhow!(e.clone())),
                None => Ok(RecordSample::Tenants(BTreeSet::new())),
            }
        }
    }

    #[tokio::test]
    async fn backfill_assigns_infers_and_fails_closed() {
        let store = FakeStore::new(&[
            ("single", None),
            ("defaults", None),
            ("empty", None),
            ("mixed", None),
            ("huge", None),
            ("owned", Some("globex")),
            ("broken", None),
        ]);
        let tenants = |items: &[&str]| Ok(RecordSample::Tenants(set(items)));
        let sampler = FakeSampler(HashMap::from([
            ("single".to_string(), tenants(&["acme"])),
            ("defaults".to_string(), tenants(&["default"])),
            ("empty".to_string(), tenants(&[])),
            ("mixed".to_string(), tenants(&["acme", "globex"])),
            ("huge".to_string(), Ok(RecordSample::TooLarge)),
            // Sampling the already-owned collection would say "acme": it must
            // never be consulted, let alone re-assigned.
            ("owned".to_string(), tenants(&["acme"])),
            ("broken".to_string(), Err("io".to_string())),
        ]));
        let report = backfill_collection_owners(&store, &sampler, DEFAULT_SAMPLE_CAP).await;
        assert_eq!(store.owner("single").as_deref(), Some("acme"));
        assert_eq!(store.owner("defaults").as_deref(), Some("default"));
        assert_eq!(store.owner("empty").as_deref(), Some("default"));
        assert_eq!(store.owner("mixed"), None, "ambiguous stays unattributed");
        assert_eq!(store.marked("mixed").as_deref(), Some("ambiguous"));
        assert_eq!(store.owner("huge"), None, "unsampled stays unattributed");
        assert_eq!(store.marked("huge").as_deref(), Some("unsampled"));
        assert_eq!(
            store.owner("owned").as_deref(),
            Some("globex"),
            "never re-assigned"
        );
        assert_eq!(store.owner("broken"), None, "failure stays unattributed");
        assert_eq!(
            store.marked("broken"),
            None,
            "a failure is retried next run"
        );
        assert_eq!(
            report,
            BackfillReport {
                assigned: 3,
                ambiguous: 1,
                too_large: 1,
                skipped: 0,
                failed: 1
            }
        );

        // Re-run: idempotent — nothing assigned twice, decided collections
        // are not rescanned; only the failed one is retried.
        let writes_before = store.writes();
        let again = backfill_collection_owners(&store, &sampler, DEFAULT_SAMPLE_CAP).await;
        assert_eq!(
            again,
            BackfillReport {
                failed: 1,
                ..BackfillReport::default()
            }
        );
        assert_eq!(store.writes(), writes_before);
    }

    /// A collection that gains an owner between listing and the write is left
    /// alone (the write re-checks).
    #[tokio::test]
    async fn assignment_rechecks_ownership_at_write_time() {
        struct RacingStore(FakeStore);
        #[async_trait]
        impl CollectionOwnerStore for RacingStore {
            async fn unattributed_collections(&self) -> Result<Vec<String>> {
                Ok(vec!["raced".to_string()])
            }
            async fn assign_owner_if_unattributed(&self, id: &str, t: &str) -> Result<bool> {
                self.0.assign_owner_if_unattributed(id, t).await
            }
            async fn mark_unattributable(&self, id: &str, reason: &str) -> Result<()> {
                self.0.mark_unattributable(id, reason).await
            }
        }
        let store = RacingStore(FakeStore::new(&[("raced", Some("globex"))]));
        let sampler = FakeSampler(HashMap::from([(
            "raced".to_string(),
            Ok(RecordSample::Tenants(set(&["acme"]))),
        )]));
        let report = backfill_collection_owners(&store, &sampler, 10).await;
        assert_eq!(report.skipped, 1);
        assert_eq!(store.0.owner("raced").as_deref(), Some("globex"));
    }
}
