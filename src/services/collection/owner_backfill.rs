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
//!   the write itself re-checks (`assign_owner_if_unattributed`), so a
//!   collection that gained an owner meanwhile is left alone.
//! * **Idempotent / safe to interrupt.** Each collection is one independent
//!   catalog write; a re-run skips everything already assigned.
//! * **Bounded inspection.** At most `sample_cap` records per collection.
//! * **Fail closed on ambiguity.** Records from more than one tenant ⇒ the
//!   collection stays unattributed (default tenant only) and one error is
//!   logged for the operator.
//! * **Non-fatal.** Any failure is logged; that collection stays unattributed.

use std::collections::BTreeSet;

use anyhow::Result;
use async_trait::async_trait;

/// Records inspected per collection.
pub const DEFAULT_SAMPLE_CAP: usize = 1_000;

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
    match tenants.len() {
        0 => OwnerInference::Assign(proximadb_tenant::DEFAULT_TENANT.to_string()),
        1 => tenants
            .into_iter()
            .next()
            .map(OwnerInference::Assign)
            .unwrap_or_else(|| {
                OwnerInference::Assign(proximadb_tenant::DEFAULT_TENANT.to_string())
            }),
        _ => OwnerInference::Ambiguous(tenants),
    }
}

/// Source of the tenants recorded on a collection's records.
#[async_trait]
pub trait RecordTenantSampler: Send + Sync {
    /// Distinct `tenant_id`s of up to `cap` records of `collection_id`.
    async fn sample_record_tenants(
        &self,
        collection_id: &str,
        cap: usize,
    ) -> Result<BTreeSet<String>>;
}

/// The catalog side of the backfill.
#[async_trait]
pub trait CollectionOwnerStore: Send + Sync {
    /// Ids of collections that record no owner.
    async fn unattributed_collections(&self) -> Result<Vec<String>>;
    /// Record `tenant` as owner unless the collection already records one.
    async fn assign_owner_if_unattributed(&self, collection_id: &str, tenant: &str)
    -> Result<bool>;
}

/// Outcome counts of one backfill run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BackfillReport {
    pub assigned: usize,
    pub ambiguous: usize,
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
            Ok(sampled) => sampled,
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
                     collection stays owned by the default tenant until an operator records \
                     its owner"
                );
                report.ambiguous += 1;
            }
        }
    }
    report
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
}

#[async_trait]
impl RecordTenantSampler for crate::services::operations::vectors::VectorOperationsService {
    async fn sample_record_tenants(
        &self,
        collection_id: &str,
        cap: usize,
    ) -> Result<BTreeSet<String>> {
        // Internal (unscoped) read of WAL + flushed storage.
        let records = self
            .list_all_records_with_tenant_context(collection_id, None)
            .await?;
        Ok(records
            .into_iter()
            .take(cap)
            .map(|record| record.tenant_id)
            .collect())
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

    /// In-memory catalog: collection id → recorded owner (None = unattributed).
    struct FakeStore {
        owners: Mutex<HashMap<String, Option<String>>>,
        writes: Mutex<usize>,
    }

    impl FakeStore {
        fn new(entries: &[(&str, Option<&str>)]) -> Self {
            Self {
                owners: Mutex::new(
                    entries
                        .iter()
                        .map(|(id, owner)| (id.to_string(), owner.map(str::to_string)))
                        .collect(),
                ),
                writes: Mutex::new(0),
            }
        }
        fn owner(&self, id: &str) -> Option<String> {
            self.owners
                .lock()
                .ok()
                .and_then(|owners| owners.get(id).cloned().flatten())
        }
    }

    #[async_trait]
    impl CollectionOwnerStore for FakeStore {
        async fn unattributed_collections(&self) -> Result<Vec<String>> {
            let owners = self
                .owners
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?;
            let mut ids: Vec<String> = owners
                .iter()
                .filter(|(_, owner)| owner.is_none())
                .map(|(id, _)| id.clone())
                .collect();
            ids.sort();
            Ok(ids)
        }
        async fn assign_owner_if_unattributed(&self, id: &str, tenant: &str) -> Result<bool> {
            let mut owners = self
                .owners
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?;
            match owners.get_mut(id) {
                Some(slot @ None) => {
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
    }

    struct FakeSampler(HashMap<String, Result<BTreeSet<String>, String>>);

    #[async_trait]
    impl RecordTenantSampler for FakeSampler {
        async fn sample_record_tenants(&self, id: &str, _cap: usize) -> Result<BTreeSet<String>> {
            match self.0.get(id) {
                Some(Ok(tenants)) => Ok(tenants.clone()),
                Some(Err(e)) => Err(anyhow::anyhow!(e.clone())),
                None => Ok(BTreeSet::new()),
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
            ("owned", Some("globex")),
            ("broken", None),
        ]);
        let sampler = FakeSampler(HashMap::from([
            ("single".to_string(), Ok(set(&["acme"]))),
            ("defaults".to_string(), Ok(set(&["default"]))),
            ("empty".to_string(), Ok(set(&[]))),
            ("mixed".to_string(), Ok(set(&["acme", "globex"]))),
            // Sampling the already-owned collection would say "acme": it must
            // never be consulted, let alone re-assigned.
            ("owned".to_string(), Ok(set(&["acme"]))),
            ("broken".to_string(), Err("io".to_string())),
        ]));
        let report = backfill_collection_owners(&store, &sampler, DEFAULT_SAMPLE_CAP).await;
        assert_eq!(store.owner("single").as_deref(), Some("acme"));
        assert_eq!(store.owner("defaults").as_deref(), Some("default"));
        assert_eq!(store.owner("empty").as_deref(), Some("default"));
        assert_eq!(store.owner("mixed"), None, "ambiguous stays unattributed");
        assert_eq!(
            store.owner("owned").as_deref(),
            Some("globex"),
            "never re-assigned"
        );
        assert_eq!(store.owner("broken"), None, "failure stays unattributed");
        assert_eq!(
            report,
            BackfillReport {
                assigned: 3,
                ambiguous: 1,
                skipped: 0,
                failed: 1
            }
        );

        // Re-run: idempotent — nothing assigned twice, ambiguous still left.
        let writes_before = store.writes.lock().map(|w| *w).unwrap_or(0);
        let again = backfill_collection_owners(&store, &sampler, DEFAULT_SAMPLE_CAP).await;
        assert_eq!(again.assigned, 0);
        assert_eq!(again.ambiguous, 1);
        assert_eq!(store.writes.lock().map(|w| *w).unwrap_or(0), writes_before);
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
        }
        let store = RacingStore(FakeStore::new(&[("raced", Some("globex"))]));
        let sampler = FakeSampler(HashMap::from([("raced".to_string(), Ok(set(&["acme"])))]));
        let report = backfill_collection_owners(&store, &sampler, 10).await;
        assert_eq!(report.skipped, 1);
        assert_eq!(store.0.owner("raced").as_deref(), Some("globex"));
    }
}
