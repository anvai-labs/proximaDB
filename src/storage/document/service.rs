// Document service - CRUD operations for JSON documents
//
// Provides MongoDB-like document operations:
// - Insert/upsert documents with automatic indexing
// - Get by ID with optional projection
// - Update with patch operations (set, unset, inc, push, pull, etc.)
// - Delete by ID or filter
// - Query with filters, sorting, and pagination
// - Aggregate with pipeline operations
//
// Durability: All write operations are WAL-logged before in-memory update

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::metrics::collectors::DocumentMetricsCollector;
use crate::proto::proximadb_v1::{
    DocumentCollectionConfig, DocumentFilter, DocumentUpdate, SqlObject, SqlValue, UpdateOperation,
};
use crate::storage::persistence::write_ahead_log::wal_operations::{
    DocumentOperation, UnifiedWALOperation, UnifiedWALWriter,
};
use crate::storage::traits::UnifiedStorageFormat;
#[cfg(feature = "canonical-document-store")]
use proximadb_document::{DOCUMENT_COLLECTION_PROP, DOCUMENT_RECORD_LABEL, DocumentRecordKey};
#[cfg(feature = "canonical-document-store")]
use proximadb_records::{
    RecordKey, RecordRecoveryOperation, RecordScanOptions, RecordStorage,
    replay_record_recovery_operations,
};

use super::DocumentStorageEngine;
use super::aggregation_extensions::LookupFetcher;
use super::canonical_adapter::legacy_document_to_proxima_record;
use super::canonical_adapter::proxima_record_to_legacy_document;
use super::canonical_adapter::{proxima_tree_to_sql_object, sql_value_to_tree_node};
use super::indexes::IndexManager;
use super::query::QueryExecutor;
use super::{
    DocumentCollection, DocumentIngestResult, DocumentQueryParams, DocumentQueryResult,
    DocumentRecord, FlushToStorageResult,
};
use proximadb_data_model::ProximaValue;
use proximadb_records::conversions::sql_value_to_proxima;
use proximadb_records::{ProximaTree, ProximaTreeNode, tree_get};

/// Compose the structural per-tenant document collection key `{tenant}/{collection}` — the
/// document counterpart of graph's `scoped_graph_id`.
///
/// The DEFAULT tenant stays UNSCOPED (bare `collection`), matching the bare collections the
/// pgwire / REST v1 create paths register; named tenants scope. Validates the tenant as a path
/// segment (fail-closed) before it becomes part of a storage key. BOTH the storage-key parameter
/// AND the record's `collection_id` must be composed from this — they key the same canonical OID
/// (`document/{collection_id}/{doc_id}`), so a scoped write and its recovery/read agree
/// (TD-DOC-TENANT-1); scoping only one side mis-keys the live read immediately.
///
/// The collection name is validated too (no `/`), so a client-supplied name can never
/// address another tenant's `{tenant}/{collection}` key.
pub fn scoped_document_collection(tenant: &str, collection: &str) -> anyhow::Result<String> {
    proximadb_tenant::scoped_resource_key(tenant, collection)
        .map_err(|e| anyhow::anyhow!("invalid document collection key: {e}"))
}

/// Inverse of [`scoped_document_collection`]: recover `(tenant, clean_collection)` from a
/// scoped storage key.
///
/// The ADR-009 canonical-vector route needs the CLEAN collection name (the shared
/// record/vector catalog registers bare names, resolved per-tenant via `TenantContext`)
/// plus the tenant as a separate value — NOT the folded `{tenant}/{collection}` key.
/// Well-defined because a tenant is a validated path segment (no `/`) and collection
/// names match `^[a-z][a-z0-9_]*$` (no `/`): the first `/` is unambiguously the fold
/// separator, and a bare key (DEFAULT tenant) has none ⇒ `(None, key)`.
pub fn unscope_document_collection(scoped: &str) -> (Option<&str>, &str) {
    match scoped.split_once('/') {
        Some((tenant, collection)) => (Some(tenant), collection),
        None => (None, scoped),
    }
}

/// Document service for CRUD operations
pub struct DocumentService {
    /// Storage engine for persistence (vector-centric, used for legacy flush path)
    storage_engine: Arc<dyn UnifiedStorageFormat>,
    /// Document-native storage engine (CEDAR) for direct document operations.
    /// When present, CRUD operations delegate to this engine instead of
    /// the in-memory HashMap cache. (Phase 2: wire CRUD through this)
    #[allow(dead_code)]
    document_engine: Option<Arc<dyn DocumentStorageEngine>>,
    /// Index manager for path and full-text indexes
    index_manager: Arc<IndexManager>,
    /// Query executor for filter evaluation
    query_executor: Arc<QueryExecutor>,
    /// Collection metadata cache
    collections: Arc<RwLock<HashMap<String, DocumentCollection>>>,
    /// In-memory document store (hot cache, backed by WAL)
    /// Used when document_engine is None (legacy mode)
    documents: Arc<DashMap<String, HashMap<String, DocumentRecord>>>,
    /// Canonical durable record store for the Phase 2 document rebase.
    /// When the `canonical-document-store` feature is enabled and this is
    /// configured, document writes/read-throughs use `ProximaRecord` as the
    /// source of durable truth. The in-memory map and indexes remain hot
    /// compatibility/projection surfaces during migration.
    #[cfg(feature = "canonical-document-store")]
    canonical_record_store: Option<Arc<dyn RecordStorage>>,
    /// WAL writer for durability
    wal_writer: Arc<Mutex<Option<UnifiedWALWriter>>>,
    /// Base path for WAL files
    wal_path: String,
    /// Optional metrics collector for observability
    metrics_collector: Option<Arc<DocumentMetricsCollector>>,
    /// ADR-009 convergence route onto the shared record/vector store. When wired
    /// AND the runtime gate is ON for a collection (default-OFF), document
    /// writes/reads flow through the same tenant-scoped record surface REST v2 uses
    /// — so a document is visible cross-surface, metered, and stored once. When the
    /// gate is OFF the legacy `document_wal`/`documents` path is used unchanged,
    /// and pre-cutover docs remain reachable via the legacy read-fallback
    /// (mixed-read-safe). See `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`.
    ///
    /// A `OnceLock` because `RecordOpsService` is constructed *after* this service (it takes
    /// the shared `document_service` handle), so the route is injected once, post-construction,
    /// through the already-`Arc`-shared facade — never mutated again.
    record_route: std::sync::OnceLock<Arc<dyn proximadb_runtime::RecordRoutePort>>,
}

/// Bound on the canonical-vector read scan for document point/query reads. The point-get
/// is currently satisfied by a bounded scan (the shared record surface exposes scan, not a
/// labelled point-get); a follow-up can push an `oid`/`local_id` predicate into the scan for
/// O(1) point reads. Documents are a beta surface behind a default-OFF gate, so the bound is
/// generous but finite — a silent truncation past it would drop reads, so it is logged.
const CANONICAL_DOC_SCAN_LIMIT: usize = 100_000;

/// Runtime gate for the canonical-vector document route (ADR-009), **DEFAULT-ON**
/// (post-bake cutover, TD-DOC-CONV-2). The bake proved recall / single-store /
/// restart-recovery, and the write path now meters per-tenant; the default is flipped ON
/// and shipped default-reversible via a global kill-switch.
///
/// `PROXIMADB_DOC_CANONICAL_VECTOR`:
/// * unset / empty / `1` / `true` / `on` / `all` / `*` ⇒ **ON** (default)
/// * `0` / `false` / `off` / `none` ⇒ **OFF** for every collection — the global
///   **kill-switch** (force back to the legacy path, e.g. to roll the cutover back)
/// * a comma-separated list ⇒ ON **only** for the named collections (a partial-rollback
///   scoping: unlisted collections are OFF)
///
/// NB this is only HALF the routing decision: the canonical route additionally requires
/// the collection to actually exist as a canonical/vector collection (see
/// [`DocumentService::canonical_route`]), so a pure-document (non-vector) collection stays
/// on the legacy path even with the gate ON — the flip is mixed-read/write-safe, no
/// flag-day. Legacy pre-cutover docs remain reachable via the read-fallback.
pub fn doc_canonical_vector_enabled(collection: &str) -> bool {
    match std::env::var("PROXIMADB_DOC_CANONICAL_VECTOR") {
        // Unset ⇒ ON (default-ON post-cutover).
        Err(_) => true,
        Ok(raw) => {
            let raw = raw.trim();
            match raw.to_ascii_lowercase().as_str() {
                // Empty ⇒ treat as unset ⇒ ON.
                "" | "1" | "true" | "on" | "all" | "*" => true,
                // Global kill-switch ⇒ OFF for every collection (default-reversible).
                "0" | "false" | "off" | "none" => false,
                // Explicit allowlist ⇒ ON only for the listed collections; an explicit
                // list is a deliberate scoping, so it overrides default-ON for the rest.
                _ => raw.split(',').map(str::trim).any(|c| c == collection),
            }
        }
    }
}

impl DocumentService {
    /// Create a new document service (legacy mode, uses in-memory HashMap)
    pub fn new(storage_engine: Arc<dyn UnifiedStorageFormat>) -> Self {
        Self {
            storage_engine,
            document_engine: None,
            index_manager: Arc::new(IndexManager::new()),
            query_executor: Arc::new(QueryExecutor::new()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(DashMap::new()),
            #[cfg(feature = "canonical-document-store")]
            canonical_record_store: None,
            wal_writer: Arc::new(Mutex::new(None)),
            wal_path: String::new(),
            metrics_collector: None,
            record_route: std::sync::OnceLock::new(),
        }
    }

    /// Create a document service backed by a DocumentStorageEngine (CEDAR).
    ///
    /// When a document engine is provided, all CRUD operations delegate to it
    /// instead of the in-memory HashMap cache.
    pub fn with_document_engine(
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        document_engine: Arc<dyn DocumentStorageEngine>,
    ) -> Self {
        Self {
            storage_engine,
            document_engine: Some(document_engine),
            index_manager: Arc::new(IndexManager::new()),
            query_executor: Arc::new(QueryExecutor::new()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(DashMap::new()),
            #[cfg(feature = "canonical-document-store")]
            canonical_record_store: None,
            wal_writer: Arc::new(Mutex::new(None)),
            wal_path: String::new(),
            metrics_collector: None,
            record_route: std::sync::OnceLock::new(),
        }
    }

    /// Create a new document service with metrics collection
    pub fn new_with_metrics(
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        metrics_collector: Arc<DocumentMetricsCollector>,
    ) -> Self {
        Self {
            storage_engine,
            document_engine: None,
            index_manager: Arc::new(IndexManager::new()),
            query_executor: Arc::new(QueryExecutor::new()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(DashMap::new()),
            #[cfg(feature = "canonical-document-store")]
            canonical_record_store: None,
            wal_writer: Arc::new(Mutex::new(None)),
            wal_path: String::new(),
            metrics_collector: Some(metrics_collector),
            record_route: std::sync::OnceLock::new(),
        }
    }

    /// Create a document service backed by canonical `ProximaRecord` storage.
    ///
    /// This is the Phase 2 migration path from
    /// `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`.
    /// The service API still accepts/returns legacy document protocol shapes,
    /// but durable state is written/read as `ProximaRecord`.
    #[cfg(feature = "canonical-document-store")]
    pub fn with_canonical_record_store(
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        record_store: Arc<dyn RecordStorage>,
    ) -> Self {
        Self {
            storage_engine,
            document_engine: None,
            index_manager: Arc::new(IndexManager::new()),
            query_executor: Arc::new(QueryExecutor::new()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(DashMap::new()),
            canonical_record_store: Some(record_store),
            wal_writer: Arc::new(Mutex::new(None)),
            wal_path: String::new(),
            metrics_collector: None,
            record_route: std::sync::OnceLock::new(),
        }
    }

    /// Create a canonical-record-backed document service with WAL recovery.
    ///
    /// Recovered canonical document WAL entries are replayed through the shared
    /// `proximadb_records::replay_record_recovery_operations` hook so the
    /// canonical record store, not the document facade map, owns durable state.
    #[cfg(feature = "canonical-document-store")]
    pub async fn with_canonical_record_store_and_wal(
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        record_store: Arc<dyn RecordStorage>,
        wal_base_path: &str,
    ) -> Result<Self> {
        let wal_path = format!("{}/document_wal", wal_base_path);
        let wal_writer = UnifiedWALWriter::new(wal_path.clone())
            .await
            .context("Failed to create document WAL writer")?;

        let mut service = Self {
            storage_engine,
            document_engine: None,
            index_manager: Arc::new(IndexManager::new()),
            query_executor: Arc::new(QueryExecutor::new()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(DashMap::new()),
            canonical_record_store: Some(record_store),
            wal_writer: Arc::new(Mutex::new(Some(wal_writer))),
            wal_path,
            metrics_collector: None,
            record_route: std::sync::OnceLock::new(),
        };

        service.recover_from_wal().await?;

        Ok(service)
    }

    /// Create a new document service with WAL support.
    #[deprecated(
        since = "0.2.3",
        note = "legacy document_wal path is fallback-only after the ADR-055 default-ON cutover; use with_canonical_record_store_and_wal. Retirement: TD-DOC-RETIRE-1."
    )]
    pub async fn new_with_wal(
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        wal_base_path: &str,
    ) -> Result<Self> {
        // TD-DOC-RETIRE-1 P2 rewires this to the canonical constructor; the deprecated
        // delegate is intentional until then (the runtime warn fires in the leaf below).
        #[allow(deprecated)]
        Self::new_with_wal_and_metrics(storage_engine, wal_base_path, None).await
    }

    /// Create a new document service with WAL support and optional metrics.
    #[deprecated(
        since = "0.2.3",
        note = "legacy document_wal path is fallback-only after the ADR-055 default-ON cutover; use with_canonical_record_store_and_wal. Retirement: TD-DOC-RETIRE-1."
    )]
    pub async fn new_with_wal_and_metrics(
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        wal_base_path: &str,
        metrics_collector: Option<Arc<DocumentMetricsCollector>>,
    ) -> Result<Self> {
        tracing::warn!(
            "DocumentService::new_with_wal[_and_metrics] is deprecated: the legacy document_wal \
             path is fallback-only after the ADR-055 default-ON cutover. Migrate to \
             with_canonical_record_store_and_wal (retirement tracked by TD-DOC-RETIRE-1)."
        );
        let wal_path = format!("{}/document_wal", wal_base_path);
        let wal_writer = UnifiedWALWriter::new(wal_path.clone())
            .await
            .context("Failed to create document WAL writer")?;

        let mut service = Self {
            storage_engine,
            document_engine: None,
            index_manager: Arc::new(IndexManager::new()),
            query_executor: Arc::new(QueryExecutor::new()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(DashMap::new()),
            #[cfg(feature = "canonical-document-store")]
            canonical_record_store: None,
            wal_writer: Arc::new(Mutex::new(Some(wal_writer))),
            wal_path,
            metrics_collector,
            record_route: std::sync::OnceLock::new(),
        };

        // Recover from WAL on startup
        service.recover_from_wal().await?;

        Ok(service)
    }

    /// Wire the ADR-009 convergence route onto the shared record/vector store, once.
    ///
    /// Called post-construction (the `RecordOpsService` that backs the route is built after
    /// this service and takes its shared handle), through the already-`Arc`-shared facade.
    /// Idempotent-safe: a second call is ignored. The legacy `document_wal`/`documents` path
    /// is retained for the default-OFF gate and as the mixed-read-safe fallback for
    /// pre-cutover documents.
    pub fn set_record_route(&self, record_route: Arc<dyn proximadb_runtime::RecordRoutePort>) {
        let _ = self.record_route.set(record_route);
    }

    /// The canonical record route port, if registered (ungated) — the DataFusion
    /// `documents()` UDTF uses it to resolve PAX scan inputs for predicate pushdown
    /// (TD-DOC-PUSHDOWN-1). Unlike [`Self::canonical_route`], this is not gated on the
    /// per-collection canonical-vector flag: pushdown-eligibility is decided by whether
    /// `pax_scan_inputs` resolves, and the caller falls back to the in-memory scan.
    pub fn record_route_port(&self) -> Option<Arc<dyn proximadb_runtime::RecordRoutePort>> {
        self.record_route.get().cloned()
    }

    /// Select the canonical-vector route for `collection`, else `None` ⇒ legacy path. The single
    /// decision point for the store-split cutover — every write/read branches on this. Three
    /// conditions, all required (mixed-safe under the DEFAULT-ON gate, TD-DOC-CONV-2):
    ///
    /// 1. a canonical route is wired,
    /// 2. the runtime gate is ON for `collection` (default-ON; kill-switch / allowlist honored),
    /// 3. `collection` actually EXISTS as a canonical/vector collection.
    ///
    /// (3) is what makes default-ON safe: a pure-document (non-vector) collection — one never
    /// created via REST v2 / DDL — is not resolvable canonically, so it stays on the legacy path
    /// instead of hard-failing the canonical write (`insert_records` errors on an unresolvable
    /// collection). Legacy pre-cutover docs stay reachable via the read-fallback.
    async fn canonical_route(
        &self,
        collection: &str,
    ) -> Option<&Arc<dyn proximadb_runtime::RecordRoutePort>> {
        let route = self.record_route.get()?;
        if !doc_canonical_vector_enabled(collection) {
            return None;
        }
        let (tenant, clean_collection) = unscope_document_collection(collection);
        if route.collection_exists(clean_collection, tenant).await {
            Some(route)
        } else {
            None
        }
    }

    /// Build the durable canonical envelope for a document on the ADR-009 vector route.
    ///
    /// Stamps the CLEAN collection into the label prop (the catalog resolves it per-tenant),
    /// a raw-id OID (reconstruction reads the id from `local_id`/props, not the OID), and the
    /// tenant onto the record as structural-isolation defense-in-depth for the scan filter.
    fn canonical_document_record(
        record: &DocumentRecord,
        clean_collection: &str,
        tenant: Option<&str>,
    ) -> proximadb_records::ProximaRecord {
        let mut for_store = record.clone();
        for_store.collection_id = clean_collection.to_string();
        let mut proxima = legacy_document_to_proxima_record(&for_store);
        proxima.oid = record.id.clone();
        proxima.tenant_id = tenant
            .unwrap_or(proximadb_tenant::DEFAULT_TENANT)
            .to_string();
        proxima
    }

    /// Record insert metrics if collector is configured
    async fn record_insert_metrics(&self, start: std::time::Instant, is_error: bool) {
        if let Some(ref collector) = self.metrics_collector {
            let latency_us = start.elapsed().as_micros() as f64;
            collector.record_insert(latency_us, is_error).await;
        }
    }

    /// Record update metrics if collector is configured
    async fn record_update_metrics(&self, start: std::time::Instant, is_error: bool) {
        if let Some(ref collector) = self.metrics_collector {
            let latency_us = start.elapsed().as_micros() as f64;
            collector.record_update(latency_us, is_error).await;
        }
    }

    /// Record delete metrics if collector is configured
    async fn record_delete_metrics(&self, start: std::time::Instant, is_error: bool) {
        if let Some(ref collector) = self.metrics_collector {
            let latency_us = start.elapsed().as_micros() as f64;
            collector.record_delete(latency_us, is_error).await;
        }
    }

    /// Record query metrics if collector is configured
    async fn record_query_metrics(&self, start: std::time::Instant, is_error: bool) {
        if let Some(ref collector) = self.metrics_collector {
            let latency_us = start.elapsed().as_micros() as f64;
            collector.record_query(latency_us, is_error).await;
        }
    }

    /// Recover state from WAL on startup
    async fn recover_from_wal(&mut self) -> Result<()> {
        use crate::storage::persistence::write_ahead_log::wal_operations::UnifiedWALReader;

        info!("Recovering document service from WAL at: {}", self.wal_path);
        let reader = UnifiedWALReader::new(self.wal_path.clone()).await?;
        let entries = reader.read_all().await?;

        let mut recovered_docs = 0;
        let mut recovered_collections = 0;
        #[cfg(feature = "canonical-document-store")]
        let mut canonical_recovery_ops = Vec::new();

        for entry in entries {
            if entry.is_document_operation()
                && let UnifiedWALOperation::DocumentOp(op) = entry.operation
            {
                match op {
                    DocumentOperation::InsertDocument {
                        collection_id,
                        document,
                    } => {
                        // Replay insert
                        let documents = &*self.documents;
                        let mut collection_docs = documents.entry(collection_id).or_default();
                        collection_docs.insert(document.id.clone(), document);
                        recovered_docs += 1;
                    }
                    DocumentOperation::UpsertCanonicalDocumentRecord {
                        collection_id,
                        record,
                    } => {
                        #[cfg(feature = "canonical-document-store")]
                        canonical_recovery_ops
                            .push(RecordRecoveryOperation::Upsert(Box::new(record.clone())));

                        if let Some(document) = proxima_record_to_legacy_document(&record) {
                            let documents = &*self.documents;
                            let mut collection_docs = documents.entry(collection_id).or_default();
                            collection_docs.insert(document.id.clone(), document);
                            recovered_docs += 1;
                        } else {
                            warn!(
                                "Skipping canonical WAL record '{}' because it is not a document",
                                record.oid
                            );
                        }
                    }
                    DocumentOperation::UpdateDocument {
                        collection_id,
                        document_id,
                        new_version,
                        ..
                    } => {
                        // Update version (simplified recovery - full update replay would need stored doc)
                        let documents = &*self.documents;
                        if let Some(mut collection_docs) = documents.get_mut(&collection_id)
                            && let Some(doc) = collection_docs.get_mut(&document_id)
                        {
                            doc.version = new_version;
                        }
                    }
                    DocumentOperation::DeleteDocument {
                        collection_id,
                        document_id,
                    } => {
                        // Replay delete
                        let documents = &*self.documents;
                        if let Some(mut collection_docs) = documents.get_mut(&collection_id) {
                            collection_docs.remove(&document_id);
                        }
                    }
                    DocumentOperation::DeleteCanonicalDocumentRecord {
                        collection_id,
                        document_id,
                        record_oid,
                    } => {
                        #[cfg(feature = "canonical-document-store")]
                        canonical_recovery_ops
                            .push(RecordRecoveryOperation::Delete(RecordKey::new(record_oid)));
                        #[cfg(not(feature = "canonical-document-store"))]
                        let _ = record_oid;

                        let documents = &*self.documents;
                        if let Some(mut collection_docs) = documents.get_mut(&collection_id) {
                            collection_docs.remove(&document_id);
                        }
                    }
                    DocumentOperation::BatchDocuments {
                        collection_id,
                        documents: docs,
                    } => {
                        // Replay batch insert
                        let doc_store = &*self.documents;
                        let mut collection_docs = doc_store.entry(collection_id).or_default();
                        for doc in docs {
                            collection_docs.insert(doc.id.clone(), doc);
                            recovered_docs += 1;
                        }
                    }
                    DocumentOperation::CreateCollection {
                        collection_id,
                        config_json,
                    } => {
                        // Replay collection creation
                        if let Ok(config) =
                            serde_json::from_str::<DocumentCollectionConfig>(&config_json)
                        {
                            let collection = DocumentCollection::new(collection_id.clone(), config);
                            let mut collections = self.collections.write().await;
                            collections.insert(collection_id, collection);
                            recovered_collections += 1;
                        }
                    }
                    DocumentOperation::DeleteCollection { collection_id } => {
                        // Replay collection deletion
                        let mut collections = self.collections.write().await;
                        collections.remove(&collection_id);
                        let documents = &*self.documents;
                        documents.remove(&collection_id);
                    }
                }
            }
        }

        #[cfg(feature = "canonical-document-store")]
        if let Some(record_store) = &self.canonical_record_store {
            let summary =
                replay_record_recovery_operations(record_store.as_ref(), canonical_recovery_ops)
                    .await
                    .context("Failed to replay canonical document WAL into record store")?;

            if summary.upserts_replayed > 0 || summary.deletes_replayed > 0 {
                info!(
                    "Canonical document WAL recovery complete: {} upserts, {} deletes replayed into record store",
                    summary.upserts_replayed, summary.deletes_replayed
                );
            }
        } else if !canonical_recovery_ops.is_empty() {
            warn!(
                "Recovered {} canonical document WAL operations without a canonical record store",
                canonical_recovery_ops.len()
            );
        }

        info!(
            "WAL recovery complete: {} documents, {} collections recovered",
            recovered_docs, recovered_collections
        );
        Ok(())
    }

    /// Write operation to WAL (if enabled)
    async fn write_to_wal(&self, operation: DocumentOperation) -> Result<()> {
        let mut wal_guard = self.wal_writer.lock().await;
        if let Some(ref mut writer) = *wal_guard {
            let wal_op = UnifiedWALOperation::DocumentOp(operation);
            writer.append(wal_op).await?;
        }
        Ok(())
    }

    /// Write a document upsert to WAL using canonical record intent when the
    /// canonical store is active, otherwise use the legacy document entry.
    async fn write_document_upsert_to_wal(
        &self,
        collection: &str,
        record: &DocumentRecord,
    ) -> Result<()> {
        #[cfg(feature = "canonical-document-store")]
        if self.canonical_record_store.is_some() {
            return self
                .write_to_wal(DocumentOperation::UpsertCanonicalDocumentRecord {
                    collection_id: collection.to_string(),
                    record: legacy_document_to_proxima_record(record),
                })
                .await;
        }

        self.write_to_wal(DocumentOperation::InsertDocument {
            collection_id: collection.to_string(),
            document: record.clone(),
        })
        .await
    }

    /// Write a document delete to WAL using canonical record identity when the
    /// canonical store is active, otherwise use the legacy document entry.
    async fn write_document_delete_to_wal(&self, collection: &str, id: &str) -> Result<()> {
        #[cfg(feature = "canonical-document-store")]
        if self.canonical_record_store.is_some() {
            let record_oid = DocumentRecordKey::new(collection, id).canonical_oid();
            return self
                .write_to_wal(DocumentOperation::DeleteCanonicalDocumentRecord {
                    collection_id: collection.to_string(),
                    document_id: id.to_string(),
                    record_oid,
                })
                .await;
        }

        self.write_to_wal(DocumentOperation::DeleteDocument {
            collection_id: collection.to_string(),
            document_id: id.to_string(),
        })
        .await
    }

    /// Apply insert projection maintenance using canonical record-shaped input
    /// when the canonical store is active.
    async fn index_document_projection(
        &self,
        collection: &str,
        record: &DocumentRecord,
    ) -> Result<()> {
        #[cfg(feature = "canonical-document-store")]
        if self.canonical_record_store.is_some() {
            self.index_manager
                .index_record_projection(&legacy_document_to_proxima_record(record))
                .await?;
            return Ok(());
        }

        self.index_manager.index_document(collection, record).await
    }

    /// Refresh projection maintenance using canonical record-shaped input when
    /// the canonical store is active.
    async fn reindex_document_projection(
        &self,
        collection: &str,
        record: &DocumentRecord,
    ) -> Result<()> {
        #[cfg(feature = "canonical-document-store")]
        if self.canonical_record_store.is_some() {
            self.index_manager
                .reindex_record_projection(&legacy_document_to_proxima_record(record))
                .await?;
            return Ok(());
        }

        self.index_manager
            .reindex_document(collection, record)
            .await
    }

    /// Remove projection entries for a document facade id.
    async fn remove_document_projection(&self, collection: &str, id: &str) -> Result<()> {
        #[cfg(feature = "canonical-document-store")]
        if self.canonical_record_store.is_some() {
            let record_oid = DocumentRecordKey::new(collection, id).canonical_oid();
            self.index_manager
                .remove_record_projection(collection, &record_oid)
                .await?;
            return Ok(());
        }

        self.index_manager.remove_document(collection, id).await
    }

    /// Flush WAL to disk
    pub async fn flush_wal(&self) -> Result<()> {
        let mut wal_guard = self.wal_writer.lock().await;
        if let Some(ref mut writer) = *wal_guard {
            writer.flush().await?;
        }
        Ok(())
    }

    /// Flush documents from a collection to persistent storage (SST engine)
    ///
    /// This method converts in-memory documents to ProximaRecords and flushes them
    /// to the underlying storage engine for cold tier persistence.
    ///
    /// Documents are stored with:
    /// - id: original document ID prefixed with collection (e.g., "mycollection::doc1")
    /// - vector: [0.0] placeholder (documents don't have inherent vectors)
    /// - metadata: "_document" contains serialized JSON, "_collection" for routing
    pub async fn flush_to_storage(&self, collection: &str) -> Result<FlushToStorageResult> {
        use crate::storage::traits::FlushParameters;

        info!(
            "Flushing documents from collection '{}' to storage engine",
            collection
        );
        let start = std::time::Instant::now();

        // Get documents for this collection
        let docs_to_flush: Vec<DocumentRecord> = {
            let documents = &*self.documents;
            match documents.get(collection) {
                Some(collection_docs) => collection_docs.values().cloned().collect(),
                None => {
                    warn!(
                        "No documents found in collection '{}' for flush",
                        collection
                    );
                    return Ok(FlushToStorageResult {
                        documents_flushed: 0,
                        bytes_written: 0,
                        duration_ms: start.elapsed().as_millis() as u64,
                        success: true,
                    });
                }
            }
        };

        if docs_to_flush.is_empty() {
            return Ok(FlushToStorageResult {
                documents_flushed: 0,
                bytes_written: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                success: true,
            });
        }

        let vector_records: Vec<proximadb_records::ProximaRecord> = docs_to_flush
            .iter()
            .filter_map(|doc| self.document_to_proxima_record(doc, collection))
            .collect();

        let record_count = vector_records.len();
        let estimated_size: usize = vector_records
            .iter()
            .map(|r| r.oid.len() + 4 + r.props.len() * 50)
            .sum();

        // Build flush parameters
        let params = FlushParameters {
            collection_id: Some(format!("_documents_{}", collection)),
            force: true,
            synchronous: true,
            vector_records,
            trigger_compaction: false,
            estimated_size,
            ..Default::default()
        };

        // Flush to storage engine
        let result = self.storage_engine.flush(params).await?;

        let duration_ms = start.elapsed().as_millis() as u64;
        info!(
            "Flushed {} documents from '{}' to storage in {}ms",
            record_count, collection, duration_ms
        );

        Ok(FlushToStorageResult {
            documents_flushed: record_count,
            bytes_written: result.bytes_written.unwrap_or(0) as usize,
            duration_ms,
            success: result.success,
        })
    }

    /// Convert a DocumentRecord to ProximaRecord for storage engine persistence.
    fn document_to_proxima_record(
        &self,
        doc: &DocumentRecord,
        collection: &str,
    ) -> Option<proximadb_records::ProximaRecord> {
        let doc_json = match serde_json::to_string(&proxima_tree_to_sql_object(&doc.props)) {
            Ok(json) => json,
            Err(e) => {
                warn!("Failed to serialize document {}: {}", doc.id, e);
                return None;
            }
        };

        let mut props = proximadb_records::ProximaTree::new();
        props.insert(
            "_type".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "document".to_string(),
            )),
        );
        props.insert(
            "_collection".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                collection.to_string(),
            )),
        );
        props.insert(
            "_document".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                doc_json,
            )),
        );
        props.insert(
            "_version".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::Int64(
                doc.version as i64,
            )),
        );

        Some(proximadb_records::ProximaRecord {
            oid: format!("{}::{}", collection, doc.id),
            props,
            created_at_ns: doc.updated_at_ns,
            updated_at_ns: doc.updated_at_ns,
            record_version: doc.version,
            ..Default::default()
        })
    }

    /// Read documents from storage engine (cold tier)
    ///
    /// This is used for documents that have been flushed to storage
    /// but evicted from the in-memory hot cache.
    ///
    /// The implementation follows SOLID principles:
    /// - Uses DocumentMetadataFilterBuilder to construct the search filter (SRP)
    /// - Uses ColdTierRetriever trait for storage access (DIP)
    /// - Can be extended for different storage backends (OCP)
    ///
    /// Documents are stored as ProximaRecords with properties:
    /// - `_type`: "document"
    /// - `_collection`: collection name
    /// - `_document`: serialized JSON content
    /// - `_version`: document version
    #[allow(dead_code)]
    pub async fn read_from_storage(
        &self,
        collection: &str,
        ids: &[&str],
    ) -> Result<Vec<DocumentRecord>> {
        use crate::storage::document::storage::cold_tier::{
            ColdTierRetriever, StorageEngineColdTierRetriever,
        };

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Cold tier: Reading {} documents from collection '{}'",
            ids.len(),
            collection
        );

        // Create the cold tier retriever using the storage engine
        let retriever = StorageEngineColdTierRetriever::new(self.storage_engine.clone());

        // Retrieve documents from cold storage
        match retriever.retrieve_documents(collection, ids).await {
            Ok(documents) => {
                debug!(
                    "Cold tier: Successfully retrieved {} of {} requested documents",
                    documents.len(),
                    ids.len()
                );
                Ok(documents)
            }
            Err(e) => {
                warn!(
                    "Cold tier: Failed to retrieve documents from collection '{}': {}",
                    collection, e
                );
                // Return empty result on error to allow graceful degradation
                // The caller can fall back to other retrieval mechanisms
                Ok(Vec::new())
            }
        }
    }

    /// Flush all collections to storage
    pub async fn flush_all_to_storage(&self) -> Result<FlushToStorageResult> {
        info!("Flushing all document collections to storage");
        let start = std::time::Instant::now();

        let collection_names: Vec<String> = {
            let collections = self.collections.read().await;
            collections.keys().cloned().collect()
        };

        let mut total_docs = 0;
        let mut total_bytes = 0;

        for collection_name in collection_names {
            let result = self.flush_to_storage(&collection_name).await?;
            total_docs += result.documents_flushed;
            total_bytes += result.bytes_written;
        }

        Ok(FlushToStorageResult {
            documents_flushed: total_docs,
            bytes_written: total_bytes,
            duration_ms: start.elapsed().as_millis() as u64,
            success: true,
        })
    }

    /// Create a new document collection
    pub async fn create_collection(
        &self,
        name: &str,
        config: DocumentCollectionConfig,
    ) -> Result<String> {
        info!("Creating document collection: {}", name);

        // Check if collection already exists
        {
            let collections = self.collections.read().await;
            if collections.contains_key(name) {
                return Err(anyhow!("Collection '{}' already exists", name));
            }
        }

        // Write to WAL first (durability before in-memory update)
        let config_json =
            serde_json::to_string(&config).context("Failed to serialize collection config")?;
        self.write_to_wal(DocumentOperation::CreateCollection {
            collection_id: name.to_string(),
            config_json,
        })
        .await?;

        // Create collection metadata
        let collection = DocumentCollection::new(name.to_string(), config.clone());

        // Initialize indexes
        for index_def in &config.indexes {
            self.index_manager
                .create_index(name, index_def)
                .await
                .context("Failed to create index")?;
        }

        // Store collection metadata
        {
            let mut collections = self.collections.write().await;
            collections.insert(name.to_string(), collection);
        }

        // P-Provision (ADR-055): also provision the canonical (record/vector) collection so this
        // document collection CONVERGES on the shared store instead of the legacy path — even for
        // pure-document (vectorless) use. Uses the CLEAN name (the catalog registers bare names,
        // resolved per-tenant); dimension 0 = vectorless (documents that carry vectors are inserted
        // through the metered /documents path). Best-effort + mixed-safe: on failure we warn and
        // keep the legacy path (the canonical route's capability check just stays false).
        // `ensure_or_create_collection` funnels through here too, so first-write auto-create is
        // covered without a second hook.
        if let Some(route) = self.record_route.get() {
            let (tenant, clean_collection) = unscope_document_collection(name);
            // P-Shred follow-up (ADR-055): seed props-auto-promotion from the collection's declared
            // indexes so those hot fields shred into typed columns at flush. Only simple top-level
            // SCALAR keys shred usefully (a nested `$.user.email` top-level is a Map, not a scalar);
            // skip nested/array paths rather than promoting a column that would never populate.
            let promote_keys: Vec<String> = config
                .indexes
                .iter()
                .filter_map(|idx| {
                    let key = idx.path.trim_start_matches('$').trim_start_matches('.');
                    if key.is_empty() || key.contains('.') || key.contains('[') {
                        None
                    } else {
                        Some(key.to_string())
                    }
                })
                .collect();
            if let Err(e) = route
                .ensure_collection(clean_collection, 0, tenant, &promote_keys)
                .await
            {
                warn!(
                    "P-Provision: canonical collection provision for '{}' failed (staying on legacy path): {}",
                    name, e
                );
            }
        }

        info!("Created document collection: {}", name);
        Ok(name.to_string())
    }

    /// Get collection metadata
    pub async fn get_collection(&self, name: &str) -> Result<Option<DocumentCollection>> {
        let collections = self.collections.read().await;
        Ok(collections.get(name).cloned())
    }

    /// Get the collection metadata for `name`, lazily provisioning it with defaults if absent —
    /// the get-or-create counterpart of [`get_collection`].
    ///
    /// This lets a tenant-scoped first write auto-register its collection under the SCOPED key
    /// (`{tenant}/{collection}`): the v2 document gRPC path scopes reads/writes, but there is no
    /// scoped create path (pgwire/REST-v1 create bare), so without this the scoped collection has
    /// no metadata and the existence gate rejects the insert (TD-DOC-TENANT-1). Provisioning goes
    /// through [`create_collection`] so the `CreateCollection` WAL op is logged and the collection
    /// metadata is rebuilt on restart (recovery replays collections only from that op). Idempotent
    /// and race-safe: a lost create race is resolved by re-fetch.
    pub async fn ensure_or_create_collection(&self, name: &str) -> Result<DocumentCollection> {
        if let Some(existing) = self.get_collection(name).await? {
            return Ok(existing);
        }
        let config = DocumentCollectionConfig {
            name: name.to_string(),
            ..Default::default()
        };
        // On success OR a lost create race, re-fetch the now-registered metadata.
        let _ = self.create_collection(name, config).await;
        self.get_collection(name)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' missing after provision", name))
    }

    /// List all collections
    pub async fn list_collections(&self) -> Result<Vec<DocumentCollection>> {
        let collections = self.collections.read().await;
        Ok(collections.values().cloned().collect())
    }

    /// Delete a collection
    pub async fn delete_collection(&self, name: &str) -> Result<bool> {
        info!("Deleting document collection: {}", name);

        // Check if collection exists first
        let exists = {
            let collections = self.collections.read().await;
            collections.contains_key(name)
        };

        if !exists {
            return Ok(false);
        }

        // Write to WAL first (durability before in-memory update)
        self.write_to_wal(DocumentOperation::DeleteCollection {
            collection_id: name.to_string(),
        })
        .await?;

        // Remove from cache
        {
            let mut collections = self.collections.write().await;
            collections.remove(name);
        }

        // Drop indexes
        self.index_manager.drop_collection_indexes(name).await?;

        // Remove documents
        {
            let documents = &*self.documents;
            documents.remove(name);
        }

        Ok(true)
    }

    /// Insert a single document
    pub async fn insert_document(
        &self,
        collection: &str,
        id: Option<&str>,
        document: SqlObject,
    ) -> Result<DocumentRecord> {
        // The legacy v1 `SqlObject` is converted into the neutral, ProximaTree-
        // native `DocumentRecord` here at the wire edge; all durability flows
        // through the `DocumentRecord`-native path below.
        let doc_id = id.map_or_else(|| uuid::Uuid::new_v4().to_string(), |s| s.to_string());
        let record = DocumentRecord::new(doc_id, document, collection.to_string());
        self.insert_document_record(collection, record).await
    }

    /// Insert a pre-built [`DocumentRecord`] — the ProximaTree-native (v2) write
    /// path that never touches the legacy v1 `SqlObject`. Same durability
    /// (WAL → canonical record store → indexes → hot cache → stats) as
    /// [`insert_document`], which now converts `SqlObject` → `DocumentRecord` at
    /// the wire edge and delegates here. The record's `id` is used verbatim
    /// (callers generate/normalize it).
    pub async fn insert_document_record(
        &self,
        collection: &str,
        record: DocumentRecord,
    ) -> Result<DocumentRecord> {
        let start = std::time::Instant::now();
        let doc_id = record.id.clone();
        debug!(
            "Inserting document {} into collection: {}",
            doc_id, collection
        );

        // Get — or lazily provision — the collection (get-or-create). A tenant-scoped first write
        // auto-registers its collection under the scoped key, so named-tenant document create→use
        // works without an out-of-band scoped create (TD-DOC-TENANT-1). The default tenant's
        // collections are bare and already exist, so they are found, not recreated.
        let _collection_meta = match self.ensure_or_create_collection(collection).await {
            Ok(meta) => meta,
            Err(e) => {
                self.record_insert_metrics(start, true).await;
                return Err(e);
            }
        };

        // ADR-009 canonical-vector route (gate ON, per-collection opt-in): persist to the
        // shared record/vector store — the exact path REST v2 uses — so the document is
        // visible cross-surface, metered, and stored once. The legacy `document_wal`/DashMap
        // is skipped on this path (the vector WAL owns recovery); pre-cutover docs stay
        // reachable via the read-fallback in `get_document`/`query_documents`.
        if let Some(route) = self.canonical_route(collection).await {
            let (tenant, clean_collection) = unscope_document_collection(collection);
            let proxima = Self::canonical_document_record(&record, clean_collection, tenant);
            match route
                .insert_records(clean_collection, vec![proxima], tenant)
                .await
            {
                Ok(_) => {
                    debug!(
                        "Inserted document {} into {} via canonical-vector route",
                        doc_id, clean_collection
                    );
                    self.record_insert_metrics(start, false).await;
                    return Ok(record);
                }
                Err(e) => {
                    self.record_insert_metrics(start, true).await;
                    return Err(e.context("canonical-vector document insert failed"));
                }
            }
        }

        // Write to WAL first (durability before in-memory update)
        if let Err(e) = self.write_document_upsert_to_wal(collection, &record).await {
            self.record_insert_metrics(start, true).await;
            return Err(e);
        }

        // Phase 2 migration path: when configured, canonical records are the
        // durable truth and legacy maps/indexes are compatibility projections.
        #[cfg(feature = "canonical-document-store")]
        let record = if let Some(record_store) = &self.canonical_record_store {
            let stored = record_store
                .upsert_record(legacy_document_to_proxima_record(&record))
                .await
                .context("Failed to upsert canonical document record")?;

            proxima_record_to_legacy_document(&stored).ok_or_else(|| {
                anyhow!(
                    "Canonical document record '{}' could not be rebuilt",
                    stored.oid
                )
            })?
        } else {
            record
        };

        // Update indexes
        if let Err(e) = self.index_document_projection(collection, &record).await {
            self.record_insert_metrics(start, true).await;
            return Err(e);
        }

        // Store document in memory (backed by WAL for durability)
        {
            let documents = &*self.documents;
            let mut collection_docs = documents.entry(collection.to_string()).or_default();
            collection_docs.insert(doc_id.clone(), record.clone());
        }

        // Update collection stats
        {
            let mut collections = self.collections.write().await;
            if let Some(coll) = collections.get_mut(collection) {
                coll.document_count += 1;
                coll.updated_at_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            }
        }

        debug!("Inserted document {} into {}", doc_id, collection);
        self.record_insert_metrics(start, false).await;
        Ok(record)
    }

    /// Insert multiple documents in batch
    pub async fn insert_documents(
        &self,
        collection: &str,
        documents: Vec<(Option<String>, SqlObject)>,
    ) -> Result<DocumentIngestResult> {
        let start = std::time::Instant::now();
        let mut result = DocumentIngestResult::default();

        for (id, doc) in documents {
            match self.insert_document(collection, id.as_deref(), doc).await {
                Ok(_) => result.ingested += 1,
                Err(e) => {
                    result.failed += 1;
                    result.errors.push(e.to_string());
                }
            }
        }

        result.processing_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Get a document by ID
    pub async fn get_document(
        &self,
        collection: &str,
        id: &str,
        projection: Option<Vec<String>>,
    ) -> Result<Option<DocumentRecord>> {
        debug!("Getting document {} from {}", id, collection);

        // ADR-009 canonical-vector route: the shared vector store is authoritative. Point-get it
        // by id (O(log n) bloom + B+ tree — TD-DOC-CONV-1, no full-collection scan), rebuild the
        // doc, return on hit. On miss, fall back to the legacy in-memory map for pre-cutover docs
        // (mixed-read-safe). Note: no `get_collection` existence gate here — a canonical
        // collection may live only in the record/vector catalog (e.g. created + written via
        // REST v2), never in this facade's own map.
        if let Some(route) = self.canonical_route(collection).await {
            let (tenant, clean_collection) = unscope_document_collection(collection);
            if let Some(mut doc) = route
                .get_record(clean_collection, id, tenant)
                .await
                .context("canonical-vector document get failed")?
                .as_ref()
                .and_then(proxima_record_to_legacy_document)
            {
                if let Some(fields) = projection.as_ref().filter(|f| !f.is_empty()) {
                    doc.props = self.apply_projection(&doc.props, fields);
                }
                return Ok(Some(doc));
            }
            return Ok(self.legacy_dashmap_get(collection, id, projection));
        }

        // Verify collection exists
        self.get_collection(collection)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection))?;

        // Phase 2 migration path: canonical record store is authoritative when
        // supplied. The legacy in-memory map remains the fallback path.
        #[cfg(feature = "canonical-document-store")]
        if let Some(record_store) = &self.canonical_record_store {
            let key = DocumentRecordKey::new(collection, id);
            let record = record_store
                .get_record(&RecordKey::new(key.canonical_oid()))
                .await
                .context("Failed to get canonical document record")?;

            if let Some(record) = record {
                let mut result = proxima_record_to_legacy_document(&record).ok_or_else(|| {
                    anyhow!(
                        "Canonical document record '{}' could not be rebuilt",
                        record.oid
                    )
                })?;

                if let Some(fields) = projection
                    && !fields.is_empty()
                {
                    result.props = self.apply_projection(&result.props, &fields);
                }

                return Ok(Some(result));
            }

            return Ok(None);
        }

        // Retrieve from in-memory store
        let documents = &*self.documents;
        if let Some(collection_docs) = documents.get(collection)
            && let Some(record) = collection_docs.get(id)
        {
            let mut result = record.clone();

            // Apply projection if specified
            if let Some(fields) = projection
                && !fields.is_empty()
            {
                result.props = self.apply_projection(&result.props, &fields);
            }

            return Ok(Some(result));
        }

        Ok(None)
    }

    /// Legacy in-memory (DashMap) point lookup — the mixed-read-safe fallback for
    /// pre-cutover documents still keyed under the scoped `collection`. Used only from
    /// the canonical-vector read path after a store miss; the gate-OFF path keeps its
    /// own inline lookup unchanged.
    fn legacy_dashmap_get(
        &self,
        collection: &str,
        id: &str,
        projection: Option<Vec<String>>,
    ) -> Option<DocumentRecord> {
        let documents = &*self.documents;
        let collection_docs = documents.get(collection)?;
        let mut result = collection_docs.get(id)?.clone();
        if let Some(fields) = projection.as_ref().filter(|f| !f.is_empty()) {
            result.props = self.apply_projection(&result.props, fields);
        }
        Some(result)
    }

    /// Apply field projection over the canonical props tree (TD-106 Slice 7e).
    ///
    /// Nested scalar paths (`a.b`) project under their last segment; top-level
    /// object/array fields are copied whole.
    fn apply_projection(&self, props: &ProximaTree, fields: &[String]) -> ProximaTree {
        let mut projected = ProximaTree::new();

        for field in fields {
            let key = field.split('.').next_back().unwrap_or(field);
            if let Some(value) = tree_get(props, field.trim_start_matches("$.")) {
                projected.insert(key.to_string(), ProximaTreeNode::Value(value.clone()));
            } else if let Some(node) = props.get(field) {
                // Top-level object/array (or a field whose path traverses an object).
                projected.insert(field.clone(), node.clone());
            }
        }

        projected
    }

    /// Update a document by ID
    pub async fn update_document(
        &self,
        collection: &str,
        id: &str,
        updates: Vec<DocumentUpdate>,
        expected_version: Option<u64>,
    ) -> Result<DocumentRecord> {
        let start = std::time::Instant::now();
        debug!("Updating document {} in {}", id, collection);

        // Get existing document
        let mut record = match self
            .get_document(collection, id, None)
            .await?
            .ok_or_else(|| anyhow!("Document '{}' not found", id))
        {
            Ok(r) => r,
            Err(e) => {
                self.record_update_metrics(start, true).await;
                return Err(e);
            }
        };

        // Check version for optimistic locking
        if let Some(expected) = expected_version
            && record.version != expected
        {
            self.record_update_metrics(start, true).await;
            return Err(anyhow!(
                "Version mismatch: expected {}, got {}",
                expected,
                record.version
            ));
        }

        // Apply updates onto the canonical props tree (TD-106 Slice 7d).
        for update in &updates {
            if let Err(e) = self.apply_update(&mut record.props, update) {
                self.record_update_metrics(start, true).await;
                return Err(e);
            }
        }

        // Increment version
        let new_version = record.version + 1;
        record.version = new_version;
        record.updated_at_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

        // ADR-009 canonical-vector route: write the updated document back to the shared
        // vector store (upsert on the raw-id OID) so the update is visible cross-surface.
        if let Some(route) = self.canonical_route(collection).await {
            let (tenant, clean_collection) = unscope_document_collection(collection);
            let proxima = Self::canonical_document_record(&record, clean_collection, tenant);
            match route
                .insert_records(clean_collection, vec![proxima], tenant)
                .await
            {
                Ok(_) => {
                    debug!(
                        "Updated document {} in {} via canonical-vector route",
                        id, clean_collection
                    );
                    self.record_update_metrics(start, false).await;
                    return Ok(record);
                }
                Err(e) => {
                    self.record_update_metrics(start, true).await;
                    return Err(e.context("canonical-vector document update failed"));
                }
            }
        }

        // Write to WAL first (durability before in-memory update)
        // Store full updated document for proper recovery replay
        if let Err(e) = self.write_document_upsert_to_wal(collection, &record).await {
            self.record_update_metrics(start, true).await;
            return Err(e);
        }

        // Phase 2 migration path: when configured, persist the updated
        // document as canonical durable state before refreshing projections.
        #[cfg(feature = "canonical-document-store")]
        if let Some(record_store) = &self.canonical_record_store {
            let stored = record_store
                .upsert_record(legacy_document_to_proxima_record(&record))
                .await
                .context("Failed to upsert updated canonical document record")?;

            record = proxima_record_to_legacy_document(&stored).ok_or_else(|| {
                anyhow!(
                    "Canonical document record '{}' could not be rebuilt",
                    stored.oid
                )
            })?;
        }

        // Update indexes
        if let Err(e) = self.reindex_document_projection(collection, &record).await {
            self.record_update_metrics(start, true).await;
            return Err(e);
        }

        // Persist updated document to in-memory store
        {
            let documents = &*self.documents;
            if let Some(mut collection_docs) = documents.get_mut(collection) {
                collection_docs.insert(id.to_string(), record.clone());
            }
        }

        debug!("Updated document {} in {}", id, collection);
        self.record_update_metrics(start, false).await;
        Ok(record)
    }

    /// Apply a single update operation to the canonical props tree.
    ///
    /// TD-106 Slice 7d: navigation/mutation is `ProximaTree`-native; the proto
    /// `update.value` (SqlValue wire operand) is lifted to canonical at this
    /// boundary (`$set` keeps object structure via `sql_value_to_tree_node`;
    /// array/scalar ops lift via `sql_value_to_proxima`).
    fn apply_update(&self, props: &mut ProximaTree, update: &DocumentUpdate) -> Result<()> {
        use crate::proto::proximadb_v1::sql_value::Value as SqlVal;
        let path = &update.path;
        let value = update.value.as_ref();

        match UpdateOperation::try_from(update.operation).unwrap_or(UpdateOperation::Unspecified) {
            UpdateOperation::Set => {
                if let Some(v) = value {
                    self.set_path_value(props, path, sql_value_to_tree_node(v))?;
                }
            }
            UpdateOperation::Unset => {
                self.unset_path(props, path)?;
            }
            UpdateOperation::Inc => {
                if let Some(v) = value {
                    let inc = match &v.value {
                        Some(SqlVal::Int64Value(i)) => *i as f64,
                        Some(SqlVal::NumberValue(f)) => *f,
                        _ => return Err(anyhow!("Increment value must be numeric")),
                    };
                    self.increment_path(props, path, inc)?;
                }
            }
            UpdateOperation::Push => {
                if let Some(v) = value {
                    self.push_to_array(props, path, sql_value_to_proxima(v))?;
                }
            }
            UpdateOperation::Pull => {
                if let Some(v) = value {
                    self.pull_from_array(props, path, &sql_value_to_proxima(v))?;
                }
            }
            UpdateOperation::AddToSet => {
                if let Some(v) = value {
                    self.add_to_set(props, path, sql_value_to_proxima(v))?;
                }
            }
            UpdateOperation::Rename => {
                if let Some(v) = value {
                    // Value should be the new field name.
                    let new_name = match &v.value {
                        Some(SqlVal::StringValue(s)) => s.clone(),
                        _ => return Err(anyhow!("New name must be a string")),
                    };
                    self.rename_path(props, path, &new_name)?;
                }
            }
            UpdateOperation::Unspecified => {
                return Err(anyhow!("Unspecified update operation"));
            }
        }

        Ok(())
    }

    // Path manipulation helpers (canonical ProximaTree, TD-106 Slice 7d)

    /// Split `$.a.b.c` into segments; errors on an empty path.
    fn path_segments(path: &str) -> Result<Vec<&str>> {
        let parts: Vec<&str> = path.trim_start_matches("$.").split('.').collect();
        if parts.is_empty() || (parts.len() == 1 && parts[0].is_empty()) {
            return Err(anyhow!("Empty path"));
        }
        Ok(parts)
    }

    /// Set a node at a dotted path, creating intermediate objects as needed.
    fn set_path_value(
        &self,
        doc: &mut ProximaTree,
        path: &str,
        node: ProximaTreeNode,
    ) -> Result<()> {
        let parts = Self::path_segments(path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            let entry = current
                .entry(part.to_string())
                .or_insert_with(|| ProximaTreeNode::Object(ProximaTree::new()));
            match entry {
                ProximaTreeNode::Object(sub) => current = sub,
                _ => return Err(anyhow!("Path {} is not an object", part)),
            }
        }
        current.insert(last.to_string(), node);
        Ok(())
    }

    /// Remove a field at a dotted path.
    fn unset_path(&self, doc: &mut ProximaTree, path: &str) -> Result<()> {
        let parts = Self::path_segments(path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            match current.get_mut(*part) {
                Some(ProximaTreeNode::Object(sub)) => current = sub,
                Some(_) => return Err(anyhow!("Path {} is not an object", part)),
                None => return Ok(()), // path doesn't exist, nothing to unset
            }
        }
        current.remove(*last);
        Ok(())
    }

    /// Increment a numeric leaf at a dotted path (creating it if absent).
    fn increment_path(&self, doc: &mut ProximaTree, path: &str, inc: f64) -> Result<()> {
        let parts = Self::path_segments(path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            match current.get_mut(*part) {
                Some(ProximaTreeNode::Object(sub)) => current = sub,
                Some(_) => return Err(anyhow!("Path {} is not an object", part)),
                None => return Err(anyhow!("Path {} does not exist", part)),
            }
        }
        match current.get_mut(*last) {
            Some(ProximaTreeNode::Value(ProximaValue::Int64(v))) => *v += inc as i64,
            Some(ProximaTreeNode::Value(ProximaValue::Float64(v))) => *v += inc,
            Some(_) => return Err(anyhow!("Field at path {} is not numeric", path)),
            None => {
                current.insert(
                    last.to_string(),
                    ProximaTreeNode::Value(ProximaValue::Float64(inc)),
                );
            }
        }
        Ok(())
    }

    /// Push a value onto an array leaf at a dotted path (creating it if absent).
    fn push_to_array(&self, doc: &mut ProximaTree, path: &str, value: ProximaValue) -> Result<()> {
        let parts = Self::path_segments(path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            match current.get_mut(*part) {
                Some(ProximaTreeNode::Object(sub)) => current = sub,
                Some(_) => return Err(anyhow!("Path {} is not an object", part)),
                None => return Err(anyhow!("Path {} does not exist", part)),
            }
        }
        match current.get_mut(*last) {
            Some(ProximaTreeNode::Value(ProximaValue::Array(arr))) => arr.push(value),
            Some(_) => return Err(anyhow!("Field at path {} is not an array", path)),
            None => {
                current.insert(
                    last.to_string(),
                    ProximaTreeNode::Value(ProximaValue::Array(vec![value])),
                );
            }
        }
        Ok(())
    }

    /// Pull (remove) matching values from an array leaf at a dotted path.
    fn pull_from_array(
        &self,
        doc: &mut ProximaTree,
        path: &str,
        value: &ProximaValue,
    ) -> Result<()> {
        let parts = Self::path_segments(path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            match current.get_mut(*part) {
                Some(ProximaTreeNode::Object(sub)) => current = sub,
                Some(_) => return Err(anyhow!("Path {} is not an object", part)),
                None => return Ok(()), // path doesn't exist
            }
        }
        match current.get_mut(*last) {
            Some(ProximaTreeNode::Value(ProximaValue::Array(arr))) => arr.retain(|v| v != value),
            Some(_) => return Err(anyhow!("Field at path {} is not an array", path)),
            None => {} // field doesn't exist
        }
        Ok(())
    }

    /// Add a value to a set (array with unique values) at a dotted path.
    fn add_to_set(&self, doc: &mut ProximaTree, path: &str, value: ProximaValue) -> Result<()> {
        let parts = Self::path_segments(path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            match current.get_mut(*part) {
                Some(ProximaTreeNode::Object(sub)) => current = sub,
                Some(_) => return Err(anyhow!("Path {} is not an object", part)),
                None => return Err(anyhow!("Path {} does not exist", part)),
            }
        }
        match current.get_mut(*last) {
            Some(ProximaTreeNode::Value(ProximaValue::Array(arr))) => {
                if !arr.contains(&value) {
                    arr.push(value);
                }
            }
            Some(_) => return Err(anyhow!("Field at path {} is not an array", path)),
            None => {
                current.insert(
                    last.to_string(),
                    ProximaTreeNode::Value(ProximaValue::Array(vec![value])),
                );
            }
        }
        Ok(())
    }

    /// Rename a field at a dotted path. `new_name` is the new (leaf) field name.
    fn rename_path(&self, doc: &mut ProximaTree, old_path: &str, new_name: &str) -> Result<()> {
        let parts = Self::path_segments(old_path)?;
        let (last, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("path must have at least one segment"))?;

        let mut current = doc;
        for part in parents {
            match current.get_mut(*part) {
                Some(ProximaTreeNode::Object(sub)) => current = sub,
                Some(_) => return Err(anyhow!("Path {} is not an object", part)),
                None => return Ok(()), // path doesn't exist
            }
        }
        if let Some(node) = current.remove(*last) {
            current.insert(new_name.to_string(), node);
        }
        Ok(())
    }

    /// Delete a document by ID
    pub async fn delete_document(&self, collection: &str, id: &str) -> Result<bool> {
        let start = std::time::Instant::now();
        debug!("Deleting document {} from {}", id, collection);

        // ADR-009 canonical-vector route: tombstone in the shared vector store (visible
        // cross-surface), and also drop any pre-cutover legacy copy under the scoped key so
        // it cannot resurface (mixed-read-safe). Returns true if either store held the doc.
        if let Some(route) = self.canonical_route(collection).await {
            let (tenant, clean_collection) = unscope_document_collection(collection);
            let deleted = route
                .delete_records(clean_collection, vec![id.to_string()], tenant)
                .await
                .context("canonical-vector document delete failed")?;
            let legacy_present = self
                .documents
                .get(collection)
                .is_some_and(|docs| docs.contains_key(id));
            if legacy_present {
                if let Err(e) = self.write_document_delete_to_wal(collection, id).await {
                    self.record_delete_metrics(start, true).await;
                    return Err(e);
                }
                let _ = self.remove_document_projection(collection, id).await;
                if let Some(mut docs) = self.documents.get_mut(collection) {
                    docs.remove(id);
                }
            }
            self.record_delete_metrics(start, false).await;
            return Ok(deleted > 0 || legacy_present);
        }

        // Verify collection exists
        if let Err(e) = self
            .get_collection(collection)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection))
        {
            self.record_delete_metrics(start, true).await;
            return Err(e);
        }

        // Check if document exists before WAL write. In canonical mode the
        // RecordStore is authoritative; otherwise use the legacy hot map.
        #[cfg(feature = "canonical-document-store")]
        let canonical_key = RecordKey::new(DocumentRecordKey::new(collection, id).canonical_oid());

        let exists = {
            #[cfg(feature = "canonical-document-store")]
            if let Some(record_store) = &self.canonical_record_store {
                record_store
                    .get_record(&canonical_key)
                    .await
                    .context("Failed to check canonical document record existence")?
                    .is_some()
            } else {
                let documents = &*self.documents;
                documents
                    .get(collection)
                    .is_some_and(|docs| docs.contains_key(id))
            }

            #[cfg(not(feature = "canonical-document-store"))]
            {
                let documents = &*self.documents;
                documents
                    .get(collection)
                    .is_some_and(|docs| docs.contains_key(id))
            }
        };

        if !exists {
            self.record_delete_metrics(start, false).await;
            return Ok(false);
        }

        // Write to WAL first (durability before in-memory update)
        if let Err(e) = self.write_document_delete_to_wal(collection, id).await {
            self.record_delete_metrics(start, true).await;
            return Err(e);
        }

        #[cfg(feature = "canonical-document-store")]
        if let Some(record_store) = &self.canonical_record_store {
            record_store
                .delete_record(&canonical_key)
                .await
                .context("Failed to delete canonical document record")?;
        }

        // Remove from indexes
        if let Err(e) = self.remove_document_projection(collection, id).await {
            self.record_delete_metrics(start, true).await;
            return Err(e);
        }

        // Remove from in-memory store
        {
            let documents = &*self.documents;
            if let Some(mut collection_docs) = documents.get_mut(collection) {
                collection_docs.remove(id);
            }
        }

        // Update collection stats
        {
            let mut collections = self.collections.write().await;
            if let Some(coll) = collections.get_mut(collection) {
                if coll.document_count > 0 {
                    coll.document_count -= 1;
                }
                coll.updated_at_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            }
        }

        debug!("Deleted document {} from {}", id, collection);
        self.record_delete_metrics(start, false).await;
        Ok(true)
    }

    /// Query documents with filter, sort, and pagination
    pub async fn query_documents(
        &self,
        collection: &str,
        params: DocumentQueryParams,
    ) -> Result<DocumentQueryResult> {
        let start = std::time::Instant::now();
        debug!("Querying documents in {}", collection);

        // ADR-009 canonical-vector route: source documents from the shared vector store
        // (cross-surface visibility), then run the same in-memory query executor. A queried
        // collection is either opted-in (canonical) or not; pre-cutover point reads stay
        // served by `get_document`'s legacy fallback.
        if let Some(route) = self.canonical_route(collection).await {
            let (tenant, clean_collection) = unscope_document_collection(collection);
            let records = route
                .scan_records(clean_collection, CANONICAL_DOC_SCAN_LIMIT, tenant)
                .await
                .context("canonical-vector document scan failed")?;
            if records.len() == CANONICAL_DOC_SCAN_LIMIT {
                warn!(
                    "canonical-vector query scan hit the {} cap for collection {}; results may be incomplete",
                    CANONICAL_DOC_SCAN_LIMIT, clean_collection
                );
            }
            let documents: Vec<DocumentRecord> = records
                .iter()
                .filter_map(proxima_record_to_legacy_document)
                .collect();
            let (documents, total_count) = match self
                .query_executor
                .execute(collection, &documents, &params, &self.index_manager)
                .await
            {
                Ok(result) => result,
                Err(e) => {
                    self.record_query_metrics(start, true).await;
                    return Err(e);
                }
            };
            let query_time_ms = start.elapsed().as_millis() as u64;
            self.record_query_metrics(start, false).await;
            return Ok(DocumentQueryResult {
                documents,
                total_count: if params.include_count {
                    Some(total_count)
                } else {
                    None
                },
                query_time_ms,
            });
        }

        // Verify collection exists
        let _collection_meta = match self
            .get_collection(collection)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection))
        {
            Ok(meta) => meta,
            Err(e) => {
                self.record_query_metrics(start, true).await;
                return Err(e);
            }
        };

        // Execute query
        #[cfg(feature = "canonical-document-store")]
        let documents: Vec<DocumentRecord> =
            if let Some(record_store) = &self.canonical_record_store {
                record_store
                    .scan_records_with_options(
                        RecordScanOptions::unbounded()
                            .with_required_label(DOCUMENT_RECORD_LABEL)
                            .with_property(
                                DOCUMENT_COLLECTION_PROP,
                                proximadb_data_model::ProximaValue::String(collection.to_string()),
                            ),
                    )
                    .await
                    .context("Failed to scan canonical document records")?
                    .iter()
                    .filter_map(proxima_record_to_legacy_document)
                    .collect()
            } else {
                let docs = &*self.documents;
                match docs.get(collection) {
                    Some(collection_docs) => collection_docs.values().cloned().collect(),
                    None => Vec::new(),
                }
            };

        #[cfg(not(feature = "canonical-document-store"))]
        let documents: Vec<DocumentRecord> = {
            let docs = &*self.documents;
            match docs.get(collection) {
                Some(collection_docs) => collection_docs.values().cloned().collect(),
                None => Vec::new(),
            }
        };

        let (documents, total_count) = match self
            .query_executor
            .execute(collection, &documents, &params, &self.index_manager)
            .await
        {
            Ok(result) => result,
            Err(e) => {
                self.record_query_metrics(start, true).await;
                return Err(e);
            }
        };

        let query_time_ms = start.elapsed().as_millis() as u64;
        self.record_query_metrics(start, false).await;

        Ok(DocumentQueryResult {
            documents,
            total_count: if params.include_count {
                Some(total_count)
            } else {
                None
            },
            query_time_ms,
        })
    }

    /// Aggregate documents with a pipeline
    ///
    /// Executes a MongoDB-style aggregation pipeline on documents in a collection.
    /// Supports stages: match, group, project, sort, limit, skip, unwind.
    ///
    /// # Arguments
    /// * `collection` - The collection name
    /// * `filter` - Optional filter to apply before the pipeline
    /// * `pipeline` - The aggregation pipeline stages
    ///
    /// # Returns
    /// A vector of aggregated result documents
    pub async fn aggregate_documents(
        &self,
        collection: &str,
        filter: Option<crate::proto::proximadb_v1::DocumentFilter>,
        pipeline: Vec<crate::proto::proximadb_v1::AggregationStage>,
    ) -> Result<crate::storage::document::AggregateResult> {
        use crate::storage::document::aggregation::AggregationExecutor;

        let start = std::time::Instant::now();
        debug!("Aggregating documents in {}", collection);

        // Get all documents from the collection — the shared vector store when routed
        // (ADR-009 cross-surface visibility), else the legacy in-memory map.
        let documents: Vec<DocumentRecord> = if let Some(route) =
            self.canonical_route(collection).await
        {
            let (tenant, clean_collection) = unscope_document_collection(collection);
            let records = route
                .scan_records(clean_collection, CANONICAL_DOC_SCAN_LIMIT, tenant)
                .await
                .context("canonical-vector document scan failed")?;
            if records.len() == CANONICAL_DOC_SCAN_LIMIT {
                warn!(
                    "canonical-vector aggregate scan hit the {} cap for collection {}; results may be incomplete",
                    CANONICAL_DOC_SCAN_LIMIT, clean_collection
                );
            }
            records
                .iter()
                .filter_map(proxima_record_to_legacy_document)
                .collect()
        } else {
            // Verify collection exists (legacy path only)
            self.get_collection(collection)
                .await?
                .ok_or_else(|| anyhow!("Collection '{}' not found", collection))?;
            let docs = &*self.documents;
            match docs.get(collection) {
                Some(collection_docs) => collection_docs.values().cloned().collect(),
                None => Vec::new(),
            }
        };

        // Execute the aggregation pipeline
        let executor = AggregationExecutor::new();
        let results = executor.execute(documents, filter.as_ref(), &pipeline)?;

        let query_time_ms = start.elapsed().as_millis() as u64;
        debug!(
            "Aggregation complete: {} results in {}ms",
            results.len(),
            query_time_ms
        );

        Ok(crate::storage::document::AggregateResult {
            results,
            query_time_ms,
        })
    }

    /// Aggregate documents with $lookup support
    ///
    /// Extended aggregation that supports cross-collection joins via the $lookup stage.
    /// For each lookup stage in the pipeline, this method fetches matching documents
    /// from the foreign collection and merges them into the local documents.
    ///
    /// # Arguments
    /// * `collection` - Source collection name
    /// * `filter` - Optional filter to apply before aggregation
    /// * `pipeline` - Aggregation pipeline stages (may include $lookup)
    ///
    /// # Returns
    /// Aggregated results with lookup fields populated
    ///
    /// # Note
    /// For lookup stages, this method requires Arc<DocumentService>. Use the async variant
    /// that takes Arc<DocumentService> for full lookup support.
    pub async fn aggregate_documents_with_lookup(
        &self,
        collection: &str,
        filter: Option<DocumentFilter>,
        pipeline: Vec<crate::proto::proximadb_v1::AggregationStage>,
    ) -> Result<crate::storage::document::AggregateResult> {
        use crate::storage::document::aggregation::AggregationExecutor;

        let start = std::time::Instant::now();
        debug!(
            "Aggregating documents in {} with lookup support",
            collection
        );

        // Verify source collection exists
        self.get_collection(collection)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection))?;

        // Get all documents from the source collection
        let documents: Vec<DocumentRecord> = {
            let docs = &*self.documents;
            match docs.get(collection) {
                Some(collection_docs) => collection_docs.values().cloned().collect(),
                None => Vec::new(),
            }
        };

        // Check if pipeline contains lookup stages
        let has_lookup = pipeline.iter().any(|stage| {
            matches!(
                &stage.stage,
                Some(crate::proto::proximadb_v1::aggregation_stage::Stage::Lookup(_))
            )
        });

        if has_lookup {
            // For lookup stages, caller should use aggregate_documents_with_lookup_arc
            return Err(anyhow!(
                "Pipeline contains $lookup stages - use aggregate_documents_with_lookup_arc which takes Arc<DocumentService>"
            ));
        }

        // Execute the aggregation pipeline (without lookup support)
        let executor = AggregationExecutor::new();
        let results = executor.execute(documents, filter.as_ref(), &pipeline)?;

        let query_time_ms = start.elapsed().as_millis() as u64;
        debug!(
            "Aggregation complete: {} results in {}ms",
            results.len(),
            query_time_ms
        );

        Ok(crate::storage::document::AggregateResult {
            results,
            query_time_ms,
        })
    }

    /// Aggregate documents with $lookup support (Arc variant)
    ///
    /// This is the preferred method for aggregation pipelines that include $lookup stages.
    /// It takes Arc<DocumentService> which allows the lookup fetcher to query foreign collections.
    ///
    /// # Arguments
    /// * `collection` - Source collection name
    /// * `filter` - Optional filter to apply before aggregation
    /// * `pipeline` - Aggregation pipeline stages (may include $lookup)
    ///
    /// # Returns
    /// Aggregated results with lookup fields populated
    pub async fn aggregate_documents_with_lookup_arc(
        this: Arc<DocumentService>,
        collection: &str,
        filter: Option<DocumentFilter>,
        pipeline: Vec<crate::proto::proximadb_v1::AggregationStage>,
    ) -> Result<crate::storage::document::AggregateResult> {
        use crate::storage::document::aggregation::AggregationExecutor;

        let start = std::time::Instant::now();
        debug!(
            "Aggregating documents in {} with lookup support (Arc variant)",
            collection
        );

        // Verify source collection exists
        this.get_collection(collection)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection))?;

        // Get all documents from the source collection
        let documents: Vec<DocumentRecord> = {
            let docs = &*this.documents;
            match docs.get(collection) {
                Some(collection_docs) => collection_docs.values().cloned().collect(),
                None => Vec::new(),
            }
        };

        // Create a lookup fetcher that can query foreign collections
        let fetcher = DocumentServiceLookupFetcher {
            service: this.clone(),
        };

        // Execute aggregation pipeline with lookup support. The working set is the
        // canonical `ProximaTree` (TD-106 Slice 5); Slice 6 reads the record's
        // canonical `props` tree directly at the input edge.
        let executor = AggregationExecutor::new();
        let mut working_set: Vec<ProximaTree> = if let Some(f) = &filter {
            documents
                .iter()
                .filter(|doc| executor.matches_filter(doc, f))
                .map(|doc| doc.props.clone())
                .collect()
        } else {
            documents.into_iter().map(|doc| doc.props).collect()
        };

        // Process each stage, handling lookups specially
        for (stage_idx, stage) in pipeline.iter().enumerate() {
            use crate::proto::proximadb_v1::aggregation_stage::Stage;

            match &stage.stage {
                Some(Stage::Lookup(lookup_stage)) => {
                    working_set = executor.process_lookup(&working_set, lookup_stage, &fetcher)?;
                    debug!(
                        "After lookup stage {}: {} documents",
                        stage_idx,
                        working_set.len()
                    );
                }
                Some(_) => {
                    // Use the standard process_stage for non-lookup stages
                    working_set = executor.process_stage(&working_set, stage, stage_idx)?;
                    debug!("After stage {}: {} documents", stage_idx, working_set.len());
                }
                None => {
                    return Err(anyhow!("Empty stage at index {}", stage_idx));
                }
            }
        }

        let query_time_ms = start.elapsed().as_millis() as u64;
        debug!(
            "Aggregation with lookup complete: {} results in {}ms",
            working_set.len(),
            query_time_ms
        );

        // Output edge: lower the canonical working set back to the proto row shape
        // for the public `AggregateResult` contract.
        let results: Vec<SqlObject> = working_set.iter().map(proxima_tree_to_sql_object).collect();

        Ok(crate::storage::document::AggregateResult {
            results,
            query_time_ms,
        })
    }

    /// Calculate full-text search scores for documents matching a query
    ///
    /// This provides basic TF-IDF-like scoring for text matching.
    /// For production use, integrate with Tantivy index.
    ///
    /// # Arguments
    /// * `collection` - The collection name
    /// * `query_terms` - Terms to search for
    /// * `text_paths` - Document paths to search in (e.g., ["title", "body"])
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// Documents sorted by relevance score (highest first)
    pub async fn fulltext_search(
        &self,
        collection: &str,
        query_terms: Vec<String>,
        text_paths: Vec<String>,
        limit: usize,
    ) -> Result<Vec<(DocumentRecord, f32)>> {
        use crate::storage::document::aggregation::AggregationExecutor;

        debug!(
            "Full-text search in {} for terms: {:?}",
            collection, query_terms
        );

        // Verify collection exists
        self.get_collection(collection)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection))?;

        // Score BORROWED documents (the previous path cloned the entire
        // collection per query — the scorer also cloned every props tree and
        // record — and then truncated to `limit`). Survivors are cloned once.
        // The DashMap shard guard must outlive the borrowed documents, so
        // scoring, sorting, truncation, AND survivor materialization all
        // happen inside its scope.
        let top: Vec<(DocumentRecord, f32)> = {
            let docs = &*self.documents;
            match docs.get(collection) {
                Some(collection_docs) => {
                    let documents: Vec<&DocumentRecord> = collection_docs.values().collect();
                    // Calculate scores
                    let executor = AggregationExecutor::new();
                    let mut scored =
                        executor.calculate_fulltext_scores(&documents, &query_terms, &text_paths);

                    // Sort by score descending
                    scored
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                    // Apply limit
                    scored.truncate(limit);

                    scored
                        .into_iter()
                        .map(|(score, doc)| ((*doc).clone(), score))
                        .collect()
                }
                None => Vec::new(),
            }
        };

        Ok(top)
    }
}

// =============================================================================
// TRAIT IMPLEMENTATION: DocumentStorageOperations
// =============================================================================
// This implements the SOLID-compliant trait interface for document operations,
// bridging the existing DocumentService to the new multi-model storage traits.

use crate::storage::traits::{
    DocumentCollectionInfo as TraitDocCollectionInfo, DocumentRecord as TraitDocRecord,
    DocumentStorageOperations,
};
use async_trait::async_trait;
use proximadb_document_query::{
    DocumentQueryResult as ContractDocumentQueryResult, DocumentQueryService,
    DocumentSearchRequest, DocumentSearchResult, SortDirection,
};
use proximadb_kernel::error::ProximaDBError;

/// Convert internal DocumentRecord to trait DocumentRecord
fn to_trait_doc_record(doc: &DocumentRecord) -> TraitDocRecord {
    TraitDocRecord {
        id: doc.id.clone(),
        document: proxima_tree_to_sql_object(&doc.props),
        version: doc.version,
        // Use updated_at_ns for both since internal type doesn't have created_at_ns
        created_at_ns: doc.updated_at_ns,
        updated_at_ns: doc.updated_at_ns,
    }
}

#[async_trait]
impl DocumentStorageOperations for DocumentService {
    async fn insert_document(
        &self,
        collection: &str,
        id: &str,
        document: SqlObject,
        indexed_paths: Vec<String>,
    ) -> Result<TraitDocRecord> {
        // Use existing insert_document method (with impl block method name)
        let doc_record = DocumentService::insert_document(
            self,
            collection,
            Some(id), // Already &str, no conversion needed
            document,
        )
        .await?;

        // Note: index_document indexes all configured paths for the document
        // The indexed_paths parameter specifies which paths to index from config
        // The actual indexing is done during insert_document call above
        // We log if additional path-specific indexing is requested but not yet supported
        if !indexed_paths.is_empty() {
            debug!(
                "Additional indexed_paths specified: {:?} (already indexed via collection config)",
                indexed_paths
            );
        }

        Ok(to_trait_doc_record(&doc_record))
    }

    async fn get_document(&self, collection: &str, id: &str) -> Result<Option<TraitDocRecord>> {
        // Use existing get_document method with projection=None
        match DocumentService::get_document(self, collection, id, None).await? {
            Some(doc) => Ok(Some(to_trait_doc_record(&doc))),
            None => Ok(None),
        }
    }

    async fn query_documents(
        &self,
        collection: &str,
        filter: Option<DocumentFilter>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<TraitDocRecord>> {
        let params = DocumentQueryParams {
            filter,
            projection: Vec::new(),
            sort: Vec::new(),
            limit: limit as u32,
            offset: offset as u32,
            include_count: false,
        };

        let result = DocumentService::query_documents(self, collection, params).await?;

        Ok(result.documents.iter().map(to_trait_doc_record).collect())
    }

    async fn update_document(
        &self,
        collection: &str,
        id: &str,
        updates: Vec<DocumentUpdate>,
    ) -> Result<TraitDocRecord> {
        let doc = DocumentService::update_document(
            self, collection, id, updates, None, // expected_version
        )
        .await?;

        Ok(to_trait_doc_record(&doc))
    }

    async fn delete_document(&self, collection: &str, id: &str) -> Result<bool> {
        DocumentService::delete_document(self, collection, id).await
    }

    async fn create_document_collection(&self, config: DocumentCollectionConfig) -> Result<String> {
        // Internal create_collection takes (name, config) - clone name first to avoid borrow
        let name = config.name.clone();
        self.create_collection(&name, config).await
    }

    async fn list_document_collections(&self) -> Result<Vec<TraitDocCollectionInfo>> {
        let collections = self.list_collections().await?;

        Ok(collections
            .into_iter()
            .map(|coll| TraitDocCollectionInfo {
                name: coll.name,
                document_count: coll.document_count,
                storage_size_bytes: coll.storage_size_bytes,
                indexes: coll.indexes,
            })
            .collect())
    }
}

#[async_trait]
impl DocumentQueryService for DocumentService {
    async fn document_search(
        &self,
        request: DocumentSearchRequest,
    ) -> ContractDocumentQueryResult<DocumentSearchResult> {
        if request.filter.is_some() {
            return Err(ProximaDBError::Query(
                proximadb_kernel::error::QueryError::InvalidFilter(
                    "string document filters are not yet parsed by DocumentQueryService adapter"
                        .to_string(),
                ),
            ));
        }

        let sort = request
            .sort
            .map(|sort| crate::proto::proximadb_v1::SortField {
                path: sort.field,
                order: match sort.direction {
                    SortDirection::Ascending => crate::proto::proximadb_v1::SortOrder::Asc as i32,
                    SortDirection::Descending => crate::proto::proximadb_v1::SortOrder::Desc as i32,
                },
            })
            .into_iter()
            .collect();

        let params = DocumentQueryParams {
            filter: None,
            projection: request.projection.unwrap_or_default(),
            sort,
            limit: request.limit as u32,
            offset: request.offset as u32,
            include_count: true,
        };

        let query_result = DocumentService::query_documents(self, &request.collection_id, params)
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))?;

        let total_count = query_result
            .total_count
            .unwrap_or(query_result.documents.len() as u64) as usize;

        Ok(DocumentSearchResult {
            results: query_result
                .documents
                .iter()
                .map(|document| document.to_proto_result(None))
                .collect(),
            total_count,
            execution_time_ms: query_result.query_time_ms,
        })
    }

    async fn get_document(
        &self,
        collection_id: String,
        document_id: String,
    ) -> ContractDocumentQueryResult<Option<proximadb_document_query::DocumentRecord>> {
        DocumentService::get_document(self, &collection_id, &document_id, None)
            .await
            .map(|document| document.map(|document| document.to_proto_result(None)))
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }
}

// =============================================================================
// LOOKUP FETCHER FOR AGGREGATION PIPELINE
// =============================================================================

/// Implementation of LookupFetcher that queries the DocumentService
pub struct DocumentServiceLookupFetcher {
    pub service: Arc<DocumentService>,
}

impl LookupFetcher for DocumentServiceLookupFetcher {
    /// Fetch documents from a collection where a field matches a value
    fn fetch_matching(
        &self,
        collection: &str,
        field_path: &str,
        match_value: &SqlValue,
    ) -> Result<Vec<SqlObject>> {
        // Use blocking API to query documents from another collection
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|e| anyhow!("Failed to get runtime handle: {}", e))?;

        rt.block_on(async {
            // Create a filter for the field match
            let filter = DocumentFilter {
                conditions: vec![create_field_eq_filter(field_path, match_value.clone())],
                ..Default::default()
            };

            // Query documents with the filter
            let params = DocumentQueryParams {
                filter: Some(filter),
                limit: 1000, // Reasonable limit for lookup results
                ..Default::default()
            };

            let result = self.service.query_documents(collection, params).await?;

            Ok(result
                .documents
                .into_iter()
                .map(|doc| proxima_tree_to_sql_object(&doc.props))
                .collect())
        })
    }
}

/// Helper to create an equality filter for a field
fn create_field_eq_filter(
    field_path: &str,
    value: SqlValue,
) -> crate::proto::proximadb_v1::DocFilterCondition {
    use crate::proto::proximadb_v1::DocFilterOperator;

    crate::proto::proximadb_v1::DocFilterCondition {
        path: field_path.to_string(),
        operator: DocFilterOperator::Eq as i32,
        value: Some(value),
        values: Vec::new(), // Empty for equality operator
    }
}

// =============================================================================
// DOCUMENTPORT IMPL — ADR-015 (port impl lives on bare concrete service)
// =============================================================================
//
// Bare-service implementation of `proximadb_runtime::DocumentPort`. Each
// method takes the proto request shape directly, performs the same proto-
// conversion + bare-service-call logic that was previously in
// `impl DocumentService for DocumentServiceImpl` (the gRPC tonic wrapper),
// and returns the proto response wrapped in `anyhow::Result`.
//
// This impl coexists with the existing `impl DocumentPort for DocumentServiceImpl`
// at `src/network/grpc/document_service.rs:284` during the migration window.
// Subsequent commits will remove the wrapper's port impl + shrink the tonic
// `impl DocumentService for DocumentServiceImpl` methods to one-liners that
// delegate to this port impl. See ADR-015 for the full rationale.

#[async_trait::async_trait]
impl proximadb_runtime::DocumentPort for DocumentService {
    async fn create_collection(
        &self,
        request: crate::proto::proximadb_v1::CreateDocumentCollectionRequest,
    ) -> Result<crate::proto::proximadb_v1::CreateDocumentCollectionResponse> {
        let config = request.config.ok_or_else(|| anyhow!("Missing config"))?;
        let name = config.name.clone();
        let id = self
            .create_collection(&name, config)
            .await
            .map_err(|e| anyhow!("Failed to create collection: {}", e))?;
        Ok(
            crate::proto::proximadb_v1::CreateDocumentCollectionResponse {
                collection_id: id,
                success: true,
            },
        )
    }

    async fn list_collections(
        &self,
        _request: crate::proto::proximadb_v1::ListDocumentCollectionsRequest,
    ) -> Result<crate::proto::proximadb_v1::ListDocumentCollectionsResponse> {
        let collections = self
            .list_collections()
            .await
            .map_err(|e| anyhow!("Failed to list collections: {}", e))?;
        let infos: Vec<crate::proto::proximadb_v1::DocumentCollectionInfo> = collections
            .iter()
            .map(|c| crate::proto::proximadb_v1::DocumentCollectionInfo {
                name: c.name.clone(),
                document_count: c.document_count,
                storage_size_bytes: c.storage_size_bytes,
                indexes: c.indexes.clone(),
            })
            .collect();
        Ok(crate::proto::proximadb_v1::ListDocumentCollectionsResponse { collections: infos })
    }

    async fn delete_collection(
        &self,
        request: crate::proto::proximadb_v1::DeleteDocumentCollectionRequest,
    ) -> Result<crate::proto::proximadb_v1::DeleteDocumentCollectionResponse> {
        self.delete_collection(&request.collection)
            .await
            .map_err(|e| anyhow!("Failed to delete collection: {}", e))?;
        Ok(crate::proto::proximadb_v1::DeleteDocumentCollectionResponse { success: true })
    }

    async fn insert_document(
        &self,
        request: crate::proto::proximadb_v1::InsertDocumentRequest,
    ) -> Result<crate::proto::proximadb_v1::InsertDocumentResponse> {
        let document = request
            .document
            .ok_or_else(|| anyhow!("Missing document"))?;
        let id = request.id.as_deref();
        let record = self
            .insert_document(&request.collection, id, document)
            .await
            .map_err(|e| anyhow!("Failed to insert document: {}", e))?;
        Ok(crate::proto::proximadb_v1::InsertDocumentResponse {
            id: record.id,
            version: record.version,
        })
    }

    async fn get_document(
        &self,
        request: crate::proto::proximadb_v1::GetDocumentRequest,
    ) -> Result<crate::proto::proximadb_v1::GetDocumentResponse> {
        let projection = if request.projection.is_empty() {
            None
        } else {
            Some(request.projection)
        };
        match self
            .get_document(&request.collection, &request.id, projection)
            .await
            .map_err(|e| anyhow!("Failed to get document: {}", e))?
        {
            Some(record) => Ok(crate::proto::proximadb_v1::GetDocumentResponse {
                document: Some(proxima_tree_to_sql_object(&record.props)),
                version: record.version,
                found: true,
            }),
            None => Ok(crate::proto::proximadb_v1::GetDocumentResponse {
                document: None,
                version: 0,
                found: false,
            }),
        }
    }

    async fn update_document(
        &self,
        request: crate::proto::proximadb_v1::UpdateDocumentRequest,
    ) -> Result<crate::proto::proximadb_v1::UpdateDocumentResponse> {
        let record = self
            .update_document(
                &request.collection,
                &request.id,
                request.updates,
                request.expected_version,
            )
            .await
            .map_err(|e| anyhow!("Failed to update document: {}", e))?;
        Ok(crate::proto::proximadb_v1::UpdateDocumentResponse {
            new_version: record.version,
            success: true,
        })
    }

    async fn delete_document(
        &self,
        request: crate::proto::proximadb_v1::DeleteDocumentRequest,
    ) -> Result<crate::proto::proximadb_v1::DeleteDocumentResponse> {
        let deleted = self
            .delete_document(&request.collection, &request.id)
            .await
            .map_err(|e| anyhow!("Failed to delete document: {}", e))?;
        Ok(crate::proto::proximadb_v1::DeleteDocumentResponse { deleted })
    }

    async fn query_documents(
        &self,
        request: crate::proto::proximadb_v1::QueryDocumentsRequest,
    ) -> Result<crate::proto::proximadb_v1::QueryDocumentsResponse> {
        let params = DocumentQueryParams {
            filter: request.filter,
            projection: request.projection,
            sort: request.sort,
            limit: request.limit,
            offset: request.offset,
            include_count: request.include_count,
        };
        let result = self
            .query_documents(&request.collection, params)
            .await
            .map_err(|e| anyhow!("Failed to query documents: {}", e))?;
        let documents: Vec<crate::proto::proximadb_v1::DocumentResult> = result
            .documents
            .into_iter()
            .map(|d| crate::proto::proximadb_v1::DocumentResult {
                id: d.id,
                document: Some(proxima_tree_to_sql_object(&d.props)),
                version: d.version,
                score: None,
            })
            .collect();
        Ok(crate::proto::proximadb_v1::QueryDocumentsResponse {
            documents,
            total_count: result.total_count,
            query_time_ms: result.query_time_ms,
        })
    }

    async fn aggregate_documents(
        &self,
        request: crate::proto::proximadb_v1::AggregateDocumentsRequest,
    ) -> Result<crate::proto::proximadb_v1::AggregateDocumentsResponse> {
        let result = self
            .aggregate_documents(&request.collection, request.filter, request.pipeline)
            .await
            .map_err(|e| anyhow!("Failed to aggregate documents: {}", e))?;
        Ok(crate::proto::proximadb_v1::AggregateDocumentsResponse {
            results: result.results,
            query_time_ms: result.query_time_ms,
        })
    }
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
