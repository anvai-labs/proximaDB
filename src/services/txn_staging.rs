// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Transaction staging over any [`TableRecordStore`] (ADR-095 D3/D9, TD-PGOLTP-1).
//!
//! At COMMIT the pgwire session re-executes its buffered statements against a
//! [`StagingTableRecordStore`] instead of the real store. Writes are captured, not
//! made durable; reads see the captured rows layered over the base store, so a
//! later statement in the transaction observes an earlier one exactly as it
//! would after a real write (INSERT then UPDATE of the same row, unique checks
//! against rows inserted earlier in the transaction, …). When every statement has
//! succeeded, [`StagingTableRecordStore::into_batches`] hands the captured writes
//! to [`TableRecordStore::commit_atomic`], which makes them durable as ONE unit.
//! If any statement fails, the staging store is dropped and nothing was written.
//!
//! This is a decorator over the port, not a change inside a concrete store: the
//! DML service, its constraint checks and its UPDATE/DELETE resolution run
//! unmodified, and the base store never sees an uncommitted row.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use parking_lot::Mutex;
use proximadb_catalog::CatalogTableSchema;
use proximadb_records::{ProximaRecord, RecordScanPredicate};

#[cfg(feature = "abac-policy")]
use proximadb_abac::{ReadContext, SystemReadReason};

use crate::services::record_store::{
    ChangeRow, StagedTableWrite, TableRecordGetRequest, TableRecordGetResponse,
    TableRecordMutation, TableRecordMutationKind, TableRecordScanRequest, TableRecordScanResponse,
    TableRecordStore, TableRecordWriteResult, UniqueCandidateSet, UniqueConflict,
    memtable_partition_keys, proxima_record_to_get_response, record_unique_tuple,
};
use crate::storage::tenant::context::TenantContext;

/// The record store (or the route a transaction spans) cannot commit atomically.
/// Nothing was made durable. pgwire maps it to SQLSTATE `0A000`.
#[derive(Debug)]
pub struct AtomicCommitUnsupported(pub String);

impl std::fmt::Display for AtomicCommitUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AtomicCommitUnsupported {}

/// COMMIT found that state changed under the transaction (a writer that does
/// not take table op-locks inserted or removed a key the transaction validated).
/// Nothing was made durable. pgwire maps it to SQLSTATE `40001`.
#[derive(Debug)]
pub struct TxnSerializationConflict(pub String);

impl std::fmt::Display for TxnSerializationConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not serialize access due to concurrent update: {}",
            self.0
        )
    }
}

impl std::error::Error for TxnSerializationConflict {}

/// The commit record IS durable, but applying it to the in-memory state failed.
/// The transaction will be visible after restart replay; the client must not
/// assume either outcome. pgwire maps it to SQLSTATE `08007`.
#[derive(Debug)]
pub struct CommitOutcomeUnknown(pub String);

impl std::fmt::Display for CommitOutcomeUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "transaction commit record is durable but applying it failed; outcome visible after restart: {}",
            self.0
        )
    }
}

impl std::error::Error for CommitOutcomeUnknown {}

/// `(tenant, table)` — the scope a staged row lives in. Every trait call carries
/// the resolved schema, so the canonical table name is a stable key.
type TableKey = (String, String);

/// Per-table staged rows: `Some` = current staged image, `None` = staged delete.
type TableOverlay = BTreeMap<String, Option<ProximaRecord>>;

#[derive(Default)]
struct StagedState {
    writes: Vec<StagedTableWrite>,
    overlay: HashMap<TableKey, TableOverlay>,
}

/// Captures a transaction's writes and serves reads as base ∪ staged.
pub struct StagingTableRecordStore {
    base: Arc<dyn TableRecordStore>,
    state: Mutex<StagedState>,
}

fn table_key(
    table_schema: &CatalogTableSchema,
    tenant_context: Option<&TenantContext>,
) -> TableKey {
    // The same partition id the durable store keys by (stable object id under
    // ADR-031 keying, else the name), so two same-named tables in different
    // namespaces never share a staging overlay.
    (
        tenant_context
            .map(|tenant| tenant.tenant_id.clone())
            .unwrap_or_default(),
        memtable_partition_keys(table_schema).0,
    )
}

fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn project(mut record: ProximaRecord, include_vector: bool, include_props: bool) -> ProximaRecord {
    if !include_vector {
        record.embeddings.clear();
    }
    if !include_props {
        record.props.clear();
    }
    record
}

impl StagingTableRecordStore {
    pub fn new(base: Arc<dyn TableRecordStore>) -> Self {
        Self {
            base,
            state: Mutex::new(StagedState::default()),
        }
    }

    /// The store the staged writes will be committed to.
    pub fn base(&self) -> Arc<dyn TableRecordStore> {
        self.base.clone()
    }

    /// The captured writes, in execution order, for `commit_atomic`.
    pub fn into_batches(self) -> Vec<StagedTableWrite> {
        self.state.into_inner().writes
    }

    /// Staged state for one key: `None` = untouched by the transaction,
    /// `Some(None)` = staged delete, `Some(Some(r))` = staged image.
    fn staged(&self, key: &TableKey, oid: &str) -> Option<Option<ProximaRecord>> {
        self.state
            .lock()
            .overlay
            .get(key)
            .and_then(|table| table.get(oid))
            .cloned()
    }

    fn staged_table(&self, key: &TableKey) -> TableOverlay {
        self.state
            .lock()
            .overlay
            .get(key)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether `oid` is visible in the merged view (staged first, then base).
    async fn exists(
        &self,
        table_schema: &CatalogTableSchema,
        key: &TableKey,
        oid: &str,
        tenant_context: Option<&TenantContext>,
    ) -> Result<bool> {
        if let Some(staged) = self.staged(key, oid) {
            return Ok(staged.is_some_and(|record| !record.is_dead(now_ns())));
        }
        Ok(self
            .base
            .get_by_key(
                table_schema,
                TableRecordGetRequest {
                    table_id: table_schema.name.clone(),
                    key: oid.to_string(),
                    include_vector: false,
                    include_props: false,
                },
                tenant_context,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(
                    SystemReadReason::ForeignKeyResolution,
                    "txn_staging::exists",
                ),
            )
            .await?
            .is_some())
    }
}

#[async_trait]
impl TableRecordStore for StagingTableRecordStore {
    async fn write_mutations(
        &self,
        table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        let key = table_key(table_schema, tenant_context);
        // Validate the whole call first (same contract as the durable store: a
        // failing call stages nothing), against the merged view so earlier
        // statements of this transaction count.
        let mut local: HashMap<String, bool> = HashMap::new();
        for mutation in &mutations {
            let oid = mutation.record.oid.clone();
            let present = match local.get(&oid) {
                Some(present) => *present,
                None => {
                    self.exists(table_schema, &key, &oid, tenant_context)
                        .await?
                }
            };
            match mutation.kind {
                TableRecordMutationKind::Insert if present => {
                    return Ok(TableRecordWriteResult::failure(
                        format!(
                            "Record '{}' already exists in table '{}'",
                            oid, table_schema.name
                        ),
                        "INSERT_CONFLICT",
                    ));
                }
                TableRecordMutationKind::Update if !present => {
                    return Ok(TableRecordWriteResult::failure(
                        format!(
                            "Record '{}' does not exist in table '{}'",
                            oid, table_schema.name
                        ),
                        "UPDATE_NOT_FOUND",
                    ));
                }
                TableRecordMutationKind::Insert
                | TableRecordMutationKind::Update
                | TableRecordMutationKind::Upsert => {
                    local.insert(oid, true);
                }
                TableRecordMutationKind::Delete => {
                    local.insert(oid, false);
                }
                TableRecordMutationKind::OverwriteSnapshot
                | TableRecordMutationKind::ReplacePartitions
                | TableRecordMutationKind::Merge => {
                    return Err(anyhow!(
                        "mutation kind {:?} for table '{}' is not supported inside an atomic transaction",
                        mutation.kind,
                        table_schema.name
                    ));
                }
            }
        }

        let record_ids: Vec<String> = mutations.iter().map(|m| m.record.oid.clone()).collect();
        let mut state = self.state.lock();
        let table = state.overlay.entry(key).or_default();
        for mutation in &mutations {
            let image = match mutation.kind {
                TableRecordMutationKind::Delete => None,
                _ => Some(mutation.record.clone()),
            };
            table.insert(mutation.record.oid.clone(), image);
        }
        state.writes.push(StagedTableWrite {
            table_schema: table_schema.clone(),
            mutations,
            tenant_context: tenant_context.cloned(),
        });
        Ok(TableRecordWriteResult::success(record_ids))
    }

    async fn drop_table_records(
        &self,
        table_schema: &CatalogTableSchema,
        _tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        Err(anyhow!(
            "DROP of table '{}' is not allowed inside a transaction",
            table_schema.name
        ))
    }

    async fn commit_atomic(
        &self,
        _batches: Vec<StagedTableWrite>,
    ) -> Result<Vec<TableRecordWriteResult>> {
        Err(anyhow!("nested atomic commit inside a staged transaction"))
    }

    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        let key = table_key(table_schema, tenant_context);
        match self.staged(&key, &request.key) {
            Some(Some(record)) if !record.is_dead(now_ns()) => {
                Ok(Some(proxima_record_to_get_response(
                    record,
                    table_schema,
                    request.include_vector,
                    request.include_props,
                )))
            }
            Some(_) => Ok(None),
            None => {
                self.base
                    .get_by_key(
                        table_schema,
                        request,
                        tenant_context,
                        #[cfg(feature = "abac-policy")]
                        read_context,
                    )
                    .await
            }
        }
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let key = table_key(table_schema, tenant_context);
        let staged = self.staged_table(&key);
        if staged.is_empty() {
            return self
                .base
                .scan_records(
                    table_schema,
                    request,
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    read_context,
                )
                .await;
        }
        // The limit applies to the MERGED view: strip it from the base scan, or
        // rows the transaction deleted could crowd out live ones.
        let limit = request.limit;
        let (include_vector, include_props) = (request.include_vector, request.include_props);
        let mut base_request = request;
        base_request.limit = None;
        let mut merged = self
            .base
            .scan_records(
                table_schema,
                base_request,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await?;
        merged.retain(|record| !staged.contains_key(&record.oid));
        let now = now_ns();
        merged.extend(
            staged
                .into_values()
                .flatten()
                .filter(|record| !record.is_dead(now))
                .map(|record| project(record, include_vector, include_props)),
        );
        if let Some(limit) = limit {
            merged.truncate(limit);
        }
        Ok(merged)
    }

    /// Push the predicate down to the base (index/pruning paths stay in play),
    /// then overlay the staged rows that satisfy it — instead of the trait
    /// default, which materialises the whole merged table while the commit holds
    /// the tables' op-locks.
    async fn scan_records_filtered(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        predicate: Option<&RecordScanPredicate<'_>>,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let staged = self.staged_table(&table_key(table_schema, tenant_context));
        if staged.is_empty() {
            return self
                .base
                .scan_records_filtered(
                    table_schema,
                    request,
                    predicate,
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    read_context,
                )
                .await;
        }
        let limit = request.limit;
        let (include_vector, include_props) = (request.include_vector, request.include_props);
        let mut base_request = request;
        base_request.limit = None;
        let mut merged = self
            .base
            .scan_records_filtered(
                table_schema,
                base_request,
                predicate,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await?;
        merged.retain(|record| !staged.contains_key(&record.oid));
        let now = now_ns();
        merged.extend(
            staged
                .into_values()
                .flatten()
                .filter(|record| !record.is_dead(now))
                .filter(|record| predicate.is_none_or(|matches| matches(record)))
                .map(|record| project(record, include_vector, include_props)),
        );
        if let Some(limit) = limit {
            merged.truncate(limit);
        }
        Ok(merged)
    }

    async fn check_unique_conflict(
        &self,
        table_schema: &CatalogTableSchema,
        table_id: &str,
        primary_key: Option<&str>,
        sets: &[UniqueCandidateSet],
        exclude_oids: &HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<UniqueConflict>> {
        let key = table_key(table_schema, tenant_context);
        let staged = self.staged_table(&key);
        let now = now_ns();
        // Rows the transaction wrote: they hold the CURRENT value for their key.
        for record in staged.values().flatten() {
            if exclude_oids.contains(&record.oid) || record.is_dead(now) {
                continue;
            }
            for set in sets {
                if let Some(tuple) = record_unique_tuple(record, &set.columns, primary_key)
                    && set.candidates.contains(&tuple)
                {
                    return Ok(Some(UniqueConflict {
                        columns: set.columns.clone(),
                        tuple,
                    }));
                }
            }
        }
        // Base rows the transaction touched are superseded by their staged image
        // (or deleted), so the base must not report them.
        let mut base_exclude = exclude_oids.clone();
        base_exclude.extend(staged.keys().cloned());
        self.base
            .check_unique_conflict(
                table_schema,
                table_id,
                primary_key,
                sets,
                &base_exclude,
                tenant_context,
            )
            .await
    }

    async fn lookup_secondary(
        &self,
        table_schema: &CatalogTableSchema,
        column: &str,
        values: &HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<Vec<String>>> {
        // A base index cannot see staged rows: decline, so the caller falls back
        // to a scan, which does.
        if !self
            .staged_table(&table_key(table_schema, tenant_context))
            .is_empty()
        {
            return Ok(None);
        }
        self.base
            .lookup_secondary(table_schema, column, values, tenant_context)
            .await
    }

    async fn read_changes_since(
        &self,
        collection_id: &str,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        self.base.read_changes_since(collection_id, since_lsn).await
    }

    async fn read_changes_since_scoped(
        &self,
        collection_id: &str,
        tenant: Option<&str>,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        self.base
            .read_changes_since_scoped(collection_id, tenant, since_lsn)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::canonical_wal::MemoryTableWalAppender;
    use crate::services::record_store::{DirectWalTableRecordStore, TableWalAppender};
    use proximadb_data_model::ProximaValue;
    use proximadb_records::ProximaTreeNode;

    fn schema() -> CatalogTableSchema {
        CatalogTableSchema::new("accounts")
    }

    fn row(oid: &str, balance: i64) -> ProximaRecord {
        let mut record = ProximaRecord {
            oid: oid.to_string(),
            variation_id: Some("accounts".to_string()),
            ..Default::default()
        };
        record.props.insert(
            "balance".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(balance)),
        );
        record
    }

    fn balance_of(record: &ProximaRecord) -> Option<i64> {
        match record.props.get("balance") {
            Some(ProximaTreeNode::Value(ProximaValue::Int64(v))) => Some(*v),
            _ => None,
        }
    }

    fn mutation(kind: TableRecordMutationKind, record: ProximaRecord) -> Vec<TableRecordMutation> {
        vec![TableRecordMutation::new(kind, record)]
    }

    fn get(oid: &str) -> TableRecordGetRequest {
        TableRecordGetRequest {
            table_id: "accounts".to_string(),
            key: oid.to_string(),
            include_vector: false,
            include_props: true,
        }
    }

    fn scan() -> TableRecordScanRequest {
        TableRecordScanRequest {
            table_id: "accounts".to_string(),
            include_props: true,
            ..Default::default()
        }
    }

    #[cfg(feature = "abac-policy")]
    fn ctx() -> ReadContext {
        ReadContext::system(SystemReadReason::Statistics, "txn_staging::tests")
    }

    async fn scan_oids(store: &dyn TableRecordStore) -> Vec<String> {
        let mut oids: Vec<String> = store
            .scan_records(
                &schema(),
                scan(),
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|record| record.oid)
            .collect();
        oids.sort();
        oids
    }

    async fn base_with(
        rows: &[(&str, i64)],
    ) -> (Arc<DirectWalTableRecordStore>, Arc<MemoryTableWalAppender>) {
        let wal = Arc::new(MemoryTableWalAppender::new());
        let store = Arc::new(DirectWalTableRecordStore::new_partitioned(wal.clone()));
        for (oid, balance) in rows {
            let result = store
                .write_mutations(
                    &schema(),
                    mutation(TableRecordMutationKind::Insert, row(oid, *balance)),
                    None,
                )
                .await;
            assert!(result.is_ok_and(|r| r.success));
        }
        (store, wal)
    }

    #[tokio::test]
    async fn staged_insert_is_visible_to_the_transaction_but_not_the_base() -> Result<()> {
        let (base, wal) = base_with(&[("a", 10)]).await;
        let staging = StagingTableRecordStore::new(base.clone());

        let result = staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("b", 5)),
                None,
            )
            .await?;
        assert!(result.success);

        assert_eq!(scan_oids(&staging).await, vec!["a", "b"]);
        assert_eq!(scan_oids(base.as_ref()).await, vec!["a"]);
        assert_eq!(
            wal.entries().await.len(),
            1,
            "staging must not write the WAL"
        );
        Ok(())
    }

    #[tokio::test]
    async fn insert_then_update_in_one_transaction_reads_the_update() -> Result<()> {
        let (base, _) = base_with(&[]).await;
        let staging = StagingTableRecordStore::new(base);
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        let updated = staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Update, row("a", 2)),
                None,
            )
            .await?;
        assert!(
            updated.success,
            "UPDATE must find a row INSERTed earlier in the txn"
        );

        let got = staging
            .get_by_key(
                &schema(),
                get("a"),
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await?;
        assert!(got.is_some());
        let rows = staging
            .scan_records(
                &schema(),
                scan(),
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await?;
        assert_eq!(
            rows.iter().map(balance_of).collect::<Vec<_>>(),
            vec![Some(2)]
        );
        Ok(())
    }

    #[tokio::test]
    async fn staged_delete_hides_a_base_row_and_duplicate_insert_conflicts() -> Result<()> {
        let (base, _) = base_with(&[("a", 10), ("b", 20)]).await;
        let staging = StagingTableRecordStore::new(base);

        let dup = staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        assert_eq!(dup.error_code.as_deref(), Some("INSERT_CONFLICT"));

        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Delete, row("a", 10)),
                None,
            )
            .await?;
        assert_eq!(scan_oids(&staging).await, vec!["b"]);
        let gone = staging
            .get_by_key(
                &schema(),
                get("a"),
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await?;
        assert!(gone.is_none());

        // After the staged delete, re-inserting the key is legal.
        let reinsert = staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 3)),
                None,
            )
            .await?;
        assert!(reinsert.success);
        Ok(())
    }

    #[tokio::test]
    async fn scan_limit_applies_after_the_merge() -> Result<()> {
        let (base, _) = base_with(&[("a", 1), ("b", 2)]).await;
        let staging = StagingTableRecordStore::new(base);
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Delete, row("a", 1)),
                None,
            )
            .await?;
        let mut request = scan();
        request.limit = Some(1);
        let rows = staging
            .scan_records(
                &schema(),
                request,
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await?;
        assert_eq!(
            rows.iter().map(|r| r.oid.as_str()).collect::<Vec<_>>(),
            vec!["b"]
        );
        Ok(())
    }

    /// `accounts` with a declared `UNIQUE (balance)` — the durable store's unique
    /// index is built from the schema's declared sets, exactly as the catalog
    /// would declare them.
    fn unique_schema() -> CatalogTableSchema {
        let mut schema = schema();
        schema.relational_capabilities.constraints.push(
            proximadb_catalog::ColumnConstraint::Unique {
                columns: vec!["balance".to_string()],
            },
        );
        schema
    }

    #[tokio::test]
    async fn unique_conflict_sees_staged_rows_and_ignores_superseded_base_rows() -> Result<()> {
        let wal = Arc::new(MemoryTableWalAppender::new());
        let base = Arc::new(DirectWalTableRecordStore::new_partitioned(wal));
        let schema = unique_schema();
        base.write_mutations(
            &schema,
            mutation(TableRecordMutationKind::Insert, row("a", 10)),
            None,
        )
        .await?;
        let staging = StagingTableRecordStore::new(base);
        let set = |value: &str| UniqueCandidateSet {
            columns: vec!["balance".to_string()],
            candidates: [vec![value.to_string()]].into_iter().collect(),
        };
        let none = HashSet::new();

        // Base row a=10 conflicts before the txn touches it.
        assert!(
            staging
                .check_unique_conflict(&schema, "accounts", None, &[set("10")], &none, None)
                .await?
                .is_some()
        );
        // Update a → 11: the base image (10) is superseded, the staged one (11) counts.
        staging
            .write_mutations(
                &schema,
                mutation(TableRecordMutationKind::Update, row("a", 11)),
                None,
            )
            .await?;
        assert!(
            staging
                .check_unique_conflict(&schema, "accounts", None, &[set("10")], &none, None)
                .await?
                .is_none(),
            "the superseded base value must no longer conflict"
        );
        assert!(
            staging
                .check_unique_conflict(&schema, "accounts", None, &[set("11")], &none, None)
                .await?
                .is_some(),
            "the staged value must conflict"
        );
        Ok(())
    }

    // Review F7: the filtered override pushes the predicate to the base and
    // overlays staged rows — a superseded base match must not leak through.
    #[tokio::test]
    async fn filtered_scan_overlays_staged_rows_and_hides_superseded_base_rows() -> Result<()> {
        let (base, _) = base_with(&[("a", 10), ("b", 50)]).await;
        let staging = StagingTableRecordStore::new(base);
        // a: 10 → 60 (now matches), b stays 50 (matches), c staged at 70 (matches).
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Update, row("a", 60)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("c", 70)),
                None,
            )
            .await?;
        let over_40 = |record: &ProximaRecord| balance_of(record).is_some_and(|b| b > 40);
        let predicate: Option<&RecordScanPredicate<'_>> = Some(&over_40);
        let mut rows: Vec<(String, Option<i64>)> = staging
            .scan_records_filtered(
                &schema(),
                scan(),
                predicate,
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await?
            .into_iter()
            .map(|r| (r.oid.clone(), balance_of(&r)))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![
                ("a".to_string(), Some(60)),
                ("b".to_string(), Some(50)),
                ("c".to_string(), Some(70))
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn into_batches_preserves_execution_order() -> Result<()> {
        let (base, _) = base_with(&[]).await;
        let staging = StagingTableRecordStore::new(base);
        let other = CatalogTableSchema::new("history");
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &other,
                mutation(TableRecordMutationKind::Insert, row("h1", 0)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Update, row("a", 2)),
                None,
            )
            .await?;
        let tables: Vec<String> = staging
            .into_batches()
            .into_iter()
            .map(|batch| batch.table_schema.name)
            .collect();
        assert_eq!(tables, vec!["accounts", "history", "accounts"]);
        Ok(())
    }

    /// Counts appends so a test can prove a commit is ONE durable write.
    #[derive(Default)]
    struct CountingAppender {
        inner: MemoryTableWalAppender,
        plain: std::sync::atomic::AtomicUsize,
        atomic: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl TableWalAppender for CountingAppender {
        async fn append_operations(
            &self,
            operations: Vec<proximadb_storage_common::CanonicalOperation>,
            tenant_id: Option<String>,
        ) -> Result<Vec<proximadb_storage_common::CanonicalWalEntry>> {
            self.plain.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.append_operations(operations, tenant_id).await
        }

        async fn append_operations_atomic(
            &self,
            operations: Vec<(proximadb_storage_common::CanonicalOperation, Option<String>)>,
        ) -> Result<Vec<proximadb_storage_common::CanonicalWalEntry>> {
            self.atomic
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.append_operations_atomic(operations).await
        }
    }

    #[tokio::test]
    async fn commit_atomic_is_one_append_applied_in_order() -> Result<()> {
        let wal = Arc::new(CountingAppender::default());
        let base = Arc::new(DirectWalTableRecordStore::new_partitioned(wal.clone()));
        let staging = StagingTableRecordStore::new(base.clone());
        let history = CatalogTableSchema::new("history");
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &history,
                mutation(TableRecordMutationKind::Insert, row("h1", 0)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Update, row("a", 2)),
                None,
            )
            .await?;

        let results = base.commit_atomic(staging.into_batches()).await?;
        assert_eq!(results.len(), 3);
        assert_eq!(wal.atomic.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(wal.plain.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(wal.inner.entries().await.len(), 3);

        let rows = base
            .scan_records(
                &schema(),
                scan(),
                None,
                #[cfg(feature = "abac-policy")]
                &ctx(),
            )
            .await?;
        assert_eq!(
            rows.iter().map(balance_of).collect::<Vec<_>>(),
            vec![Some(2)]
        );
        Ok(())
    }

    // M1: a writer without op-locks inserts the key between staging and commit.
    // The commit must abort with nothing durable, not silently overwrite.
    #[tokio::test]
    async fn commit_atomic_aborts_on_a_concurrent_insert() -> Result<()> {
        let wal = Arc::new(CountingAppender::default());
        let base = Arc::new(DirectWalTableRecordStore::new_partitioned(wal.clone()));
        let staging = StagingTableRecordStore::new(base.clone());
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("b", 1)),
                None,
            )
            .await?;

        base.write_mutations(
            &schema(),
            mutation(TableRecordMutationKind::Insert, row("b", 99)),
            None,
        )
        .await?;
        let before = wal.inner.entries().await.len();

        let err = base
            .commit_atomic(staging.into_batches())
            .await
            .err()
            .ok_or_else(|| anyhow!("commit must fail"))?;
        assert!(
            err.downcast_ref::<TxnSerializationConflict>().is_some(),
            "{err:#}"
        );
        assert_eq!(wal.inner.entries().await.len(), before, "nothing durable");
        assert_eq!(wal.atomic.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(scan_oids(base.as_ref()).await, vec!["b"]);
        Ok(())
    }

    #[tokio::test]
    async fn router_refuses_a_commit_spanning_two_stores() -> Result<()> {
        use crate::services::record_store::CatalogRoutingTableRecordStore;
        let relational: Arc<dyn TableRecordStore> = Arc::new(
            DirectWalTableRecordStore::new_partitioned(Arc::new(MemoryTableWalAppender::new())),
        );
        let vector: Arc<dyn TableRecordStore> = Arc::new(
            DirectWalTableRecordStore::new_partitioned(Arc::new(MemoryTableWalAppender::new())),
        );
        let router =
            CatalogRoutingTableRecordStore::new(relational.clone(), vector.clone(), vector);
        // Same route for every batch → delegated (succeeds).
        let staging = StagingTableRecordStore::new(relational.clone());
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        assert!(router.commit_atomic(staging.into_batches()).await.is_ok());

        // A relational batch plus a vector-routed batch → refused, nothing written.
        let mut vector_schema = CatalogTableSchema::new("embeddings");
        vector_schema.storage_specialization =
            proximadb_catalog::CatalogStorageSpecialization::VectorAnn;
        let staging = StagingTableRecordStore::new(relational.clone());
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("b", 1)),
                None,
            )
            .await?;
        staging
            .write_mutations(
                &vector_schema,
                mutation(TableRecordMutationKind::Insert, row("v", 1)),
                None,
            )
            .await?;
        let err = router.commit_atomic(staging.into_batches()).await.err();
        assert!(
            err.is_some_and(|e| e.to_string().contains("spans storage routes")),
            "a commit across two stores must be refused"
        );
        assert_eq!(scan_oids(relational.as_ref()).await, vec!["a"]);
        Ok(())
    }

    #[tokio::test]
    async fn wal_untouched_until_commit() -> Result<()> {
        let (base, wal) = base_with(&[]).await;
        let staging = StagingTableRecordStore::new(base);
        staging
            .write_mutations(
                &schema(),
                mutation(TableRecordMutationKind::Insert, row("a", 1)),
                None,
            )
            .await?;
        assert!(wal.read_all_entries().await?.is_empty());
        Ok(())
    }
}
