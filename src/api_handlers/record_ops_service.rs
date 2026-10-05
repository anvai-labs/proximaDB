/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Canonical record-batch orchestration service (TD-104 / CONVERGENCE_AUDIT S1, slice S3-c)
//!
//! [`RecordOpsService`] owns the record-batch insert/upsert/delete orchestration that
//! the Arrow Flight ingest path (`do_put`) and the REST/gRPC v2 record-batch path drive.
//! It is the single canonical implementation of [`proximadb_runtime::RecordOpsPort`].
//!
//! ## Convergence
//!
//! This is the record write path's single home. TD-104 S3-f deleted the legacy
//! root `UnifiedHandlers` that previously wrapped this service — its record
//! logic had already been delegated here, so the convergence is complete with no
//! duplicated logic. The Arrow Flight ingest path (`do_put`) and the REST/gRPC
//! v2 record-batch path depend on the canonical
//! [`proximadb_runtime::RecordOpsPort`] backed directly by this service.
//!
//! ## Authority / parity
//!
//! No durable authority lives here — this is a façade over the same vector/record
//! services. Write-path behaviour is preserved EXACTLY versus the prior ROOT inline
//! implementation:
//! - tenant-context load (`CollectionService::load_tenant_context`),
//! - collection-id resolution incl. the TTL cache (`resolve_collection_id_internal`),
//! - the WAL-lane gate ([`enforce_wal_lane_for_record_batch`] → `WAL_LANE_REJECTED`),
//! - schema validation (`SCHEMA_VALIDATION_FAILED`),
//! - canonical-precision coercion + the per-precision `canonical_bytes` metric
//!   (TD-080/TD-082),
//! - DML row-count stat bumps (+n insert / -n delete),
//! - `NOT_FOUND` / `RECORD_INSERT_FAILED` error shapes.

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;

use crate::core::search::FilterExpression;
use crate::services::DmlService;
use crate::services::collection::manager::CollectionService;
use crate::services::operations::BatchOperationResult;
use crate::services::operations::vectors::{
    RichRecordBatchRequest, RichRecordDeleteBatchRequest, RichRecordGetRequest,
    RichRecordGetResponse, RichSearchRequest, RichSearchResponse, VectorOperationsService,
};
use crate::services::record_store::ChangeRow;
use crate::services::scan_cursor::ScanCursor;
use crate::services::{
    WriteDurabilityRequirement, WriteIntent, WriteLaneRouter, WriteOperationKind,
};
use proximadb_records::ProximaRecord;

// ── Relocated from the deleted root `request_handlers.rs` (TD-104 S3-f) ──────
// `CollectionIdCache` and `enforce_wal_lane_for_record_batch` were the only
// items in that module this service still depended on; both are record-write-
// path helpers, so they now live here next to their sole consumer.

const COLLECTION_ID_CACHE_TTL_SECS: u64 = 300;
const COLLECTION_ID_CACHE_MAX_SIZE: usize = 1000;

/// Cache entry for collection ID resolution
#[derive(Clone)]
struct CollectionIdCacheEntry {
    collection_id: String,
    cached_at: Instant,
}

/// Thread-safe TTL-based cache for collection ID resolution
///
/// Reduces latency from ~5ms/request (metadata backend lookup) to ~0.1ms (cache hit).
/// Uses a simple HashMap with RwLock for concurrent access.
pub struct CollectionIdCache {
    cache: std::sync::RwLock<HashMap<String, CollectionIdCacheEntry>>,
    ttl: Duration,
    max_size: usize,
}

impl CollectionIdCache {
    /// Create a new cache with default TTL and max size
    pub fn new() -> Self {
        Self {
            cache: std::sync::RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(COLLECTION_ID_CACHE_TTL_SECS),
            max_size: COLLECTION_ID_CACHE_MAX_SIZE,
        }
    }

    /// Create a new cache with custom TTL
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: std::sync::RwLock::new(HashMap::new()),
            ttl,
            max_size: COLLECTION_ID_CACHE_MAX_SIZE,
        }
    }

    /// Get a cached collection ID if it exists and is not expired
    pub fn get(&self, identifier: &str) -> Option<String> {
        let cache = self.cache.read().ok()?;
        if let Some(entry) = cache.get(identifier)
            && entry.cached_at.elapsed() < self.ttl
        {
            debug!(
                "Collection ID cache hit: '{}' -> '{}'",
                identifier, entry.collection_id
            );
            return Some(entry.collection_id.clone());
        }
        None
    }

    /// Insert a collection ID into the cache
    pub fn insert(&self, identifier: String, collection_id: String) {
        if let Ok(mut cache) = self.cache.write() {
            // Evict expired entries if cache is too large
            if cache.len() >= self.max_size {
                self.evict_expired(&mut cache);
            }

            // If still too large after eviction, remove oldest entries
            if cache.len() >= self.max_size {
                // Simple eviction: clear half the cache
                let keys_to_remove: Vec<_> = cache
                    .iter()
                    .take(cache.len() / 2)
                    .map(|(k, _)| k.clone())
                    .collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }

            cache.insert(
                identifier,
                CollectionIdCacheEntry {
                    collection_id,
                    cached_at: Instant::now(),
                },
            );
        }
    }

    /// Invalidate a specific cache entry (call on collection delete/update)
    pub fn invalidate(&self, identifier: &str) {
        if let Ok(mut cache) = self.cache.write() {
            cache.remove(identifier);
            // Also remove any entries that might have the collection_id as the identifier
            // (since resolve_collection_id accepts both name and id)
            let keys_to_remove: Vec<_> = cache
                .iter()
                .filter(|(_, entry)| entry.collection_id == identifier)
                .map(|(k, _)| k.clone())
                .collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }

    /// Evict expired entries from the cache
    fn evict_expired(&self, cache: &mut HashMap<String, CollectionIdCacheEntry>) {
        let keys_to_remove: Vec<_> = cache
            .iter()
            .filter(|(_, entry)| entry.cached_at.elapsed() >= self.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for key in keys_to_remove {
            cache.remove(&key);
        }
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
}

impl Default for CollectionIdCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Enforce the WAL-lane gate for a record-batch write: reject fast when the
/// write-intent router denies a WAL-required write (returns `WAL_LANE_REJECTED`
/// upstream). Relocated verbatim from the deleted root `request_handlers.rs`.
pub(crate) fn enforce_wal_lane_for_record_batch(
    collection_id: &str,
    operation_kind: WriteOperationKind,
    row_count: u64,
    context: &str,
) -> Result<(), String> {
    let intent = WriteIntent::new(collection_id, operation_kind)
        .with_durability(WriteDurabilityRequirement::WalRequired)
        .with_row_count_hint(row_count);
    let decision = WriteLaneRouter::new().route(&intent);
    decision
        .require_wal_lane(context)
        .map_err(|e| e.to_string())
}

/// Canonical record-batch orchestration service.
///
/// Holds exactly the dependencies the record write path needs. It is the single
/// owner of the collection-id cache, the DML-service handle, and the
/// canonical-precision resolver for the record path; ROOT forwards its
/// post-construction setters here so there is one logical owner of this state.
pub struct RecordOpsService {
    collection_service: Arc<CollectionService>,
    vector_operations_service: Arc<VectorOperationsService>,
    /// Cache for collection ID resolution (one owner — ROOT delegates here).
    collection_id_cache: CollectionIdCache,
    /// DML service for schema validation + row-count stats. Settable
    /// post-construction (thread-safe), mirroring ROOT's prior behaviour.
    dml_service: std::sync::RwLock<Option<Arc<DmlService>>>,
    /// Set-once canonical-precision resolver — wired at server bootstrap so the
    /// record path coerces embeddings to each collection's canonical precision
    /// before WAL append (TD-080/TD-082).
    precision_resolver: std::sync::OnceLock<
        Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
    >,
    /// Durable partition-lease manager for lease-on-write (Scenario-1 routing
    /// truth). When present, the record write path acquires/confirms this pod's
    /// collection lease before writing, so the shared primary-pod registry that
    /// the network gates consult is durably backed (not empty-after-restart).
    /// Settable post-construction (thread-safe); absent in embedded/test builds.
    lease_manager:
        std::sync::RwLock<Option<Arc<crate::cluster::partition_lease::PartitionLeaseManager>>>,
}

impl RecordOpsService {
    fn revision_tenant(tenant_id: Option<&str>) -> &str {
        tenant_id
            .filter(|tenant| !tenant.is_empty())
            .unwrap_or(CollectionService::DEFAULT_VERSION_TENANT)
    }

    pub async fn content_revision_snapshot(
        &self,
        tenant_id: Option<&str>,
        collection_id: &str,
    ) -> (u64, String) {
        let snapshot = crate::catalog::CorpusVersionRegistry::global()
            .content_snapshot(
                Self::revision_tenant(tenant_id),
                collection_id,
                crate::catalog::process_content_revision_incarnation(),
            )
            .await;
        (snapshot.revision, snapshot.token)
    }

    /// Build the service from the same Arcs ROOT holds.
    pub fn new(
        collection_service: Arc<CollectionService>,
        vector_operations_service: Arc<VectorOperationsService>,
    ) -> Self {
        Self {
            collection_service,
            vector_operations_service,
            collection_id_cache: CollectionIdCache::new(),
            dml_service: std::sync::RwLock::new(None),
            precision_resolver: std::sync::OnceLock::new(),
            lease_manager: std::sync::RwLock::new(None),
        }
    }

    /// Wire a `DmlService` for schema validation + row-count stats.
    /// Callable post-initialization; thread-safe.
    pub fn set_dml_service(&self, svc: Arc<DmlService>) {
        if let Ok(mut guard) = self.dml_service.write() {
            *guard = Some(svc);
        }
    }

    pub(crate) fn get_dml_service(&self) -> Option<Arc<DmlService>> {
        self.dml_service.read().ok().and_then(|guard| guard.clone())
    }

    /// Wire the durable partition-lease manager for lease-on-write.
    /// Callable post-initialization; thread-safe.
    pub fn set_lease_manager(
        &self,
        manager: Arc<crate::cluster::partition_lease::PartitionLeaseManager>,
    ) {
        if let Ok(mut guard) = self.lease_manager.write() {
            *guard = Some(manager);
        }
    }

    /// Lease-on-write: make the shared primary-pod registry truthful for
    /// `(tenant, collection)` by acquiring/confirming this pod's durable
    /// collection lease before the write proceeds. The durable key is always the
    /// resolved canonical collection UUID; aliases/names must never fork the
    /// generation fence used by WAL recovery. Idempotent + cheap after the first
    /// write (registry fast-path; the renew loop keeps the lease warm). A missing manager still
    /// indicates embedded/single-node operation; once a manager is wired, conflicts
    /// and acquire errors reject the write instead of proceeding fail-open.
    async fn ensure_collection_lease(
        &self,
        tenant_id: &str,
        collection_name: &str,
        canonical_collection_id: &str,
    ) -> Result<()> {
        let Some(manager) = self.lease_manager.read().ok().and_then(|g| g.clone()) else {
            return Ok(());
        };
        let now_ms = chrono::Utc::now().timestamp_millis();
        let routing_owned = manager
            .ensure_owned(tenant_id, canonical_collection_id, now_ms)
            .await?;
        if !routing_owned {
            tracing::warn!(
                target = "proximadb.primary_pod.lease_on_write",
                tenant_id = %tenant_id,
                collection_id = %collection_name,
                "lease-on-write: this pod is not the primary for the collection; rejecting write"
            );
            return Err(anyhow!(
                "this pod is not the primary for collection '{}'",
                collection_name
            ));
        }
        match manager
            .begin_writer_incarnation(tenant_id, canonical_collection_id, now_ms)
            .await
        {
            Ok(true) => Ok(()),
            Ok(false) => {
                tracing::warn!(
                    target = "proximadb.primary_pod.lease_on_write",
                    tenant_id = %tenant_id,
                    collection_id = %canonical_collection_id,
                    "writer incarnation is fenced; rejecting WAL write"
                );
                Err(anyhow!(
                    "this pod is not the primary for collection '{}'",
                    canonical_collection_id
                ))
            }
            Err(e) => {
                tracing::warn!(
                    target = "proximadb.primary_pod.lease_on_write",
                    error = %e,
                    tenant_id = %tenant_id,
                    collection_id = %collection_name,
                    "lease-on-write acquire failed; rejecting write"
                );
                Err(anyhow!("lease-on-write acquire failed: {}", e))
            }
        }
    }

    /// Post-construction setter for the canonical-precision resolver.
    /// Called once at server bootstrap (`ProximaDB::new` in `src/database.rs`).
    pub fn set_precision_resolver(
        &self,
        resolver: Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
    ) {
        let _ = self.precision_resolver.set(resolver);
    }

    // ---- collection-id resolution (cache is owned here) ----

    /// Resolve a collection name or UUID to its canonical UUID with caching.
    pub async fn resolve_collection_id_cached(&self, identifier: &str) -> Result<Option<String>> {
        if let Some(cached_id) = self.collection_id_cache.get(identifier) {
            return Ok(Some(cached_id));
        }

        let result = self
            .collection_service
            .resolve_collection_id(identifier)
            .await?;

        if let Some(ref id) = result {
            self.collection_id_cache
                .insert(identifier.to_string(), id.clone());
            debug!(
                "Collection ID cache miss: '{}' -> '{}' (cached)",
                identifier, id
            );
        }

        Ok(result)
    }

    /// Invalidate a single collection ID cache entry.
    pub fn invalidate_collection_cache(&self, identifier: &str) {
        self.collection_id_cache.invalidate(identifier);
        debug!("Collection ID cache invalidated: '{}'", identifier);
    }

    /// Clear the entire collection ID cache.
    pub fn clear_collection_cache(&self) {
        self.collection_id_cache.clear();
        debug!("Collection ID cache cleared");
    }

    pub(crate) async fn resolve_collection_id_internal(
        &self,
        collection_identifier: &str,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<Option<String>> {
        if let Some(tenant_ctx) = tenant_context {
            Ok(self
                .collection_service
                .get_collection_with_tenant_context(collection_identifier, Some(tenant_ctx))
                .await?
                .map(|collection| collection.id))
        } else {
            self.resolve_collection_id_cached(collection_identifier)
                .await
        }
    }

    /// Resolve BOTH the canonical collection id AND its display name from a
    /// client identifier (which may be either). The name is what the catalog
    /// table key (`default.{config.name}`), the storage URL, and the row-count
    /// stats resolve against — using the raw `request.collection_id` here (which
    /// a client may send as the numeric id, e.g. `"1"`) misses the table and
    /// leaves `record_count` stuck at 0 (the name-vs-id defect). Tenant path
    /// fetches the real `config.name`; the cached/non-tenant path falls back to
    /// the identifier (best-effort — the tenant path is the production one).
    pub(crate) async fn resolve_collection_id_and_name(
        &self,
        collection_identifier: &str,
        tenant_context: Option<&crate::storage::tenant::TenantContext>,
    ) -> Result<Option<(String, String)>> {
        if let Some(tenant_ctx) = tenant_context {
            Ok(self
                .collection_service
                .get_collection_with_tenant_context(collection_identifier, Some(tenant_ctx))
                .await?
                .map(|collection| {
                    let name = collection
                        .config
                        .as_ref()
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| collection_identifier.to_string());
                    (collection.id, name)
                }))
        } else {
            Ok(self
                .resolve_collection_id_cached(collection_identifier)
                .await?
                .map(|id| (id, collection_identifier.to_string())))
        }
    }

    /// Resolve name or object id to one canonical identity for protocol
    /// cursors and cache validators.
    pub async fn resolve_collection_identity_for_tenant(
        &self,
        collection_identifier: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<(String, String)>> {
        let tenant_context = self.collection_service.load_tenant_context(tenant_id)?;
        self.resolve_collection_id_and_name(collection_identifier, tenant_context.as_ref())
            .await
    }

    // ---- precision coercion (TD-080/TD-082) ----

    /// Mirror of `CollectionManager::collection_table_identifier` so the resolver
    /// hits the same catalog key the manager wrote at create time.
    fn collection_to_table_identifier(
        target_collection: &str,
    ) -> proximadb_catalog::TableIdentifier {
        let parsed = proximadb_catalog::TableIdentifier::parse(target_collection);
        if parsed.namespace.is_empty() {
            proximadb_catalog::TableIdentifier::new(vec!["default".to_string()], parsed.name)
        } else {
            parsed
        }
    }

    /// Coerce every embedding cell on every record to the collection's canonical
    /// precision (resolved via xCatalog) **in place**, then emit the per-precision
    /// `canonical_bytes` metric. Single source of truth for the v2 precision-coercion
    /// contract across the insert-only (Arrow Flight) and upsert (REST/gRPC v2) paths
    /// (TD-082 closure).
    pub(crate) async fn coerce_records_to_canonical_precision(
        &self,
        records: &mut [proximadb_records::ProximaRecord],
        collection_id: &str,
    ) {
        if let Some(resolver) = self.precision_resolver.get() {
            let table_id = Self::collection_to_table_identifier(collection_id);
            match resolver.resolve(&table_id).await {
                Ok(target_precision)
                    if target_precision != proximadb_records::EmbeddingScalarType::Fp32 =>
                {
                    for record in records.iter_mut() {
                        for cell in &mut record.embeddings {
                            cell.coerce_to_precision(target_precision);
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(
                        collection = %collection_id,
                        error = %e,
                        "v2 record batch: precision resolver lookup failed; \
                         records will land at their input precision"
                    );
                }
            }
        }
        if let Some(pm) = crate::observability::precision_metrics::metrics() {
            let mut per_precision: std::collections::HashMap<
                proximadb_records::EmbeddingScalarType,
                i64,
            > = std::collections::HashMap::new();
            for record in records.iter() {
                for cell in &record.embeddings {
                    *per_precision.entry(cell.precision).or_insert(0) +=
                        cell.values_byte_size() as i64;
                }
            }
            for (precision, delta) in per_precision {
                pm.add_canonical_bytes(
                    collection_id,
                    crate::observability::precision_metrics::precision_label(precision),
                    delta,
                );
            }
        }
    }

    // ---- record-batch orchestration ----

    /// Canonical rich-record delete handler used by v2 REST/gRPC/internal callers.
    pub async fn handle_record_delete_batch_for_tenant(
        &self,
        request: RichRecordDeleteBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        let tenant_context = self.collection_service.load_tenant_context(tenant_id)?;
        let (collection_id, collection_name) = match self
            .resolve_collection_id_and_name(&request.collection_id, tenant_context.as_ref())
            .await?
        {
            Some((id, name)) => (id, name),
            None => {
                return Err(anyhow!("Collection '{}' not found", request.collection_id));
            }
        };
        if let Err(e) = enforce_wal_lane_for_record_batch(
            &collection_name,
            WriteOperationKind::Delete,
            request.record_ids.len() as u64,
            "REST/gRPC handle_record_delete_batch_for_tenant",
        ) {
            return Ok(BatchOperationResult::failure(
                e,
                "WAL_LANE_REJECTED".to_string(),
            ));
        }
        if let Err(e) = self
            .ensure_collection_lease(tenant_id.unwrap_or(""), &collection_name, &collection_id)
            .await
        {
            return Ok(BatchOperationResult::failure(
                e.to_string(),
                "LEASE_NOT_HELD".to_string(),
            ));
        }
        let result = self
            .vector_operations_service
            .delete_records_with_tenant_context(
                &collection_id,
                request.record_ids,
                tenant_context.as_ref(),
            )
            .await?;
        if result.success {
            let n = result.vector_ids.len() as i64;
            if n > 0
                && let Some(dml_svc) = self.get_dml_service()
            {
                dml_svc.bump_row_count_stats(&collection_name, -n).await;
            }
        }
        Ok(result)
    }

    /// Canonical rich-record batch (upsert) handler used by v2 REST/gRPC/internal callers.
    pub async fn handle_record_batch_for_tenant(
        &self,
        request: RichRecordBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        let tenant_context = self.collection_service.load_tenant_context(tenant_id)?;
        let (collection_id, collection_name) = match self
            .resolve_collection_id_and_name(&request.collection_id, tenant_context.as_ref())
            .await?
        {
            Some((id, name)) => (id, name),
            None => {
                return Ok(BatchOperationResult::failure(
                    format!("Collection '{}' not found", request.collection_id),
                    "NOT_FOUND".to_string(),
                ));
            }
        };
        if let Some(dml_svc) = self.get_dml_service()
            && let Err(e) = dml_svc
                .validate_record_batch_against_schema(&collection_name, &request.records)
                .await
        {
            return Ok(BatchOperationResult::failure(
                format!("Schema validation failed: {}", e),
                "SCHEMA_VALIDATION_FAILED".to_string(),
            ));
        }
        if let Err(e) = enforce_wal_lane_for_record_batch(
            &collection_name,
            WriteOperationKind::Insert,
            request.records.len() as u64,
            "REST/gRPC handle_record_batch_for_tenant",
        ) {
            return Ok(BatchOperationResult::failure(
                e,
                "WAL_LANE_REJECTED".to_string(),
            ));
        }
        if let Err(e) = self
            .ensure_collection_lease(tenant_id.unwrap_or(""), &collection_name, &collection_id)
            .await
        {
            return Ok(BatchOperationResult::failure(
                e.to_string(),
                "LEASE_NOT_HELD".to_string(),
            ));
        }

        let mut records = request.records;
        self.coerce_records_to_canonical_precision(&mut records, &collection_name)
            .await;

        match self
            .vector_operations_service
            .insert_records_with_tenant_context(&collection_id, records, tenant_context.as_ref())
            .await
        {
            Ok(result) => {
                if result.success {
                    let n = result.vector_ids.len() as i64;
                    if let Some(dml_svc) = self.get_dml_service() {
                        dml_svc.bump_row_count_stats(&collection_name, n).await;
                    }
                }
                Ok(result)
            }
            Err(error) => {
                tracing::error!("Failed to process rich record batch: {:?}", error);
                // ADR-069 S4: preserve the `WalBackpressure` discriminant so the
                // boundary surfaces 429 / RESOURCE_EXHAUSTED (retryable) instead
                // of a generic RECORD_INSERT_FAILED (non-retryable).
                let error_code = crate::storage::persistence::write_ahead_log::flush_policy::write_batch_error_code(&error, "RECORD_INSERT_FAILED");
                Ok(BatchOperationResult::failure(
                    format!("Record insert failed: {}", error),
                    error_code,
                ))
            }
        }
    }

    /// Canonical insert-only rich-record handler used when callers require existing
    /// records to be rejected instead of upserted (Arrow Flight `do_put`).
    pub async fn handle_record_insert_batch_for_tenant(
        &self,
        request: RichRecordBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        let tenant_context = self.collection_service.load_tenant_context(tenant_id)?;
        let (collection_id, collection_name) = match self
            .resolve_collection_id_and_name(&request.collection_id, tenant_context.as_ref())
            .await?
        {
            Some((id, name)) => (id, name),
            None => {
                return Ok(BatchOperationResult::failure(
                    format!("Collection '{}' not found", request.collection_id),
                    "NOT_FOUND".to_string(),
                ));
            }
        };
        if let Some(dml_svc) = self.get_dml_service()
            && let Err(e) = dml_svc
                .validate_record_batch_against_schema(&collection_name, &request.records)
                .await
        {
            return Ok(BatchOperationResult::failure(
                format!("Schema validation failed: {}", e),
                "SCHEMA_VALIDATION_FAILED".to_string(),
            ));
        }
        if let Err(e) = enforce_wal_lane_for_record_batch(
            &collection_name,
            WriteOperationKind::Insert,
            request.records.len() as u64,
            "REST/gRPC handle_record_insert_batch_for_tenant",
        ) {
            return Ok(BatchOperationResult::failure(
                e,
                "WAL_LANE_REJECTED".to_string(),
            ));
        }
        if let Err(e) = self
            .ensure_collection_lease(tenant_id.unwrap_or(""), &collection_name, &collection_id)
            .await
        {
            return Ok(BatchOperationResult::failure(
                e.to_string(),
                "LEASE_NOT_HELD".to_string(),
            ));
        }
        let mut records = request.records;
        self.coerce_records_to_canonical_precision(&mut records, &collection_name)
            .await;
        match self
            .vector_operations_service
            .insert_records_only_with_tenant_context(
                &collection_id,
                records,
                tenant_context.as_ref(),
            )
            .await
        {
            Ok(result) => {
                if result.success {
                    let n = result.vector_ids.len() as i64;
                    if let Some(dml_svc) = self.get_dml_service() {
                        dml_svc.bump_row_count_stats(&collection_name, n).await;
                    }
                }
                Ok(result)
            }
            Err(error) => {
                tracing::error!(
                    "Failed to process insert-only rich record batch: {:?}",
                    error
                );
                // ADR-069 S4: preserve the `WalBackpressure` discriminant (→ 429).
                let error_code = crate::storage::persistence::write_ahead_log::flush_policy::write_batch_error_code(&error, "RECORD_INSERT_FAILED");
                Ok(BatchOperationResult::failure(
                    format!("Record insert failed: {}", error),
                    error_code,
                ))
            }
        }
    }

    // ---- record READ path (TD-104 REST phase 2) -------------------------------
    // Historically moved verbatim from the legacy root `UnifiedHandlers` so the
    // REST layer reaches these via `state.record_ops`. With the root handler now
    // deleted (TD-104 S3-f), this service IS the record path. Behaviour-identical:
    // the `self.<svc>` references resolve to the same Arcs
    // (collection_service / vector_operations_service / dml_service), and
    // `resolve_collection_id_internal` is this service's own (single logical owner
    // of the collection-id cache).

    /// Canonical rich-record search handler used by v2 REST/gRPC/internal callers.
    ///
    /// Threads the request subject + tenant stable id into the vector service so
    /// ABAC enforces at the shared `unified_search_v1_inner` seam (fail-closed on
    /// deny). Callers without a subject (gRPC/Flight/internal today) pass `None` ⇒
    /// `System` passthrough (today's behavior). Enforcement is `abac-policy`-gated
    /// inside the vector service; default builds are a pass-through.
    pub async fn handle_record_search_for_tenant(
        &self,
        request: RichSearchRequest,
        identity: proximadb_runtime::PortIdentity<'_>,
    ) -> Result<RichSearchResponse> {
        let tenant_context = self
            .collection_service
            .load_tenant_context(identity.tenant_id)?;
        let request = RichSearchRequest {
            collection_id: match self
                .resolve_collection_id_internal(&request.collection_id, tenant_context.as_ref())
                .await?
            {
                Some(id) => id,
                None => {
                    return Err(anyhow!("Collection '{}' not found", request.collection_id));
                }
            },
            ..request
        };

        self.vector_operations_service
            .search_records_with_tenant_context(request, tenant_context.as_ref(), identity)
            .await
    }

    /// Canonical rich-record get handler used by v2 REST/gRPC/internal callers.
    ///
    /// Threads `identity.subject` + `identity.tenant_stable_id` so a
    /// provisioned policy admit-checks the fetched record (fail-closed on
    /// deny); a subject-less identity (gRPC/internal callers) is a
    /// pass-through. Enforcement is `abac-policy`-gated inside the vector
    /// service (TD-ABAC-7: one method, identity decides — no `_abac` twin).
    pub async fn handle_record_get_for_tenant(
        &self,
        request: RichRecordGetRequest,
        identity: proximadb_runtime::PortIdentity<'_>,
    ) -> Result<RichRecordGetResponse> {
        let tenant_context = self
            .collection_service
            .load_tenant_context(identity.tenant_id)?;
        let request = RichRecordGetRequest {
            collection_id: match self
                .resolve_collection_id_internal(&request.collection_id, tenant_context.as_ref())
                .await?
            {
                Some(id) => id,
                None => {
                    return Err(anyhow!("Collection '{}' not found", request.collection_id));
                }
            },
            ..request
        };

        self.vector_operations_service
            .get_record_with_tenant_context(request, tenant_context.as_ref(), identity)
            .await
    }

    /// Point-get a single FULL record by id, tenant-scoped (TD-DOC-CONV-1). Resolves the tenant
    /// and collection, then returns the whole `ProximaRecord` (labels + props) via the O(log n)
    /// bloom-filter and B+ tree point lookup — the document facade needs the `document` label that
    /// the search-shaped get drops. `None` when the collection or record is absent.
    pub async fn handle_record_get_full_for_tenant(
        &self,
        collection_id: &str,
        record_id: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        let tenant_context = self.collection_service.load_tenant_context(tenant_id)?;
        let resolved = match self
            .resolve_collection_id_internal(collection_id, tenant_context.as_ref())
            .await?
        {
            Some(id) => id,
            None => return Ok(None),
        };
        self.vector_operations_service
            .get_full_record_with_tenant_context(&resolved, record_id, tenant_context.as_ref())
            .await
    }

    /// Paginated rich-record scan (TD-099(3d) push-down): resolves tenant +
    /// collection id, then streams a single page from the deduped, time-ordered
    /// scan index and returns `(page, next_cursor)`.
    #[allow(clippy::too_many_arguments)]
    pub async fn handle_record_scan_paginated_for_tenant(
        &self,
        collection_id: &str,
        cursor: Option<&ScanCursor>,
        limit: usize,
        include_vector: bool,
        include_props: bool,
        tenant_id: Option<&str>,
        filter: Option<&FilterExpression>,
        now_ns: i64,
    ) -> Result<(Vec<ProximaRecord>, Option<ScanCursor>)> {
        let tenant_context = self.collection_service.load_tenant_context(tenant_id)?;
        let resolved_id = match self
            .resolve_collection_id_internal(collection_id, tenant_context.as_ref())
            .await?
        {
            Some(id) => id,
            None => {
                return Err(anyhow!("Collection '{}' not found", collection_id));
            }
        };

        self.vector_operations_service
            .scan_records_paginated(
                &resolved_id,
                // Cursor identity: mint against the same caller-facing id the
                // REST handler decodes against, not the resolved internal id.
                collection_id,
                cursor,
                limit,
                include_vector,
                include_props,
                tenant_context.as_ref(),
                filter,
                now_ns,
            )
            .await
    }

    /// Change feed: rows changed since `since_lsn` (delegates to the DML service).
    pub async fn table_changes(&self, table: &str, since_lsn: u64) -> Result<Vec<ChangeRow>> {
        match self.get_dml_service() {
            Some(dml) => dml.changes_since(table, since_lsn).await,
            None => Ok(Vec::new()),
        }
    }
}

// TD-104 S3-c: the canonical `RecordOpsPort` implementation now lives on the
// dedicated `RecordOpsService` so the Arrow Flight write path can hold an
// `Arc<dyn RecordOpsPort>` backed directly by this service — no longer routed
// through the ROOT `UnifiedHandlers`. ROOT's own `RecordOpsPort` impl delegates
// here, so behaviour is identical regardless of which surface is wired.
#[async_trait::async_trait]
impl proximadb_runtime::RecordOpsPort for RecordOpsService {
    async fn insert_record_batch(
        &self,
        collection_id: &str,
        records: Vec<proximadb_records::ProximaRecord>,
        tenant_id: Option<&str>,
    ) -> Result<proximadb_runtime::BatchOperationResult> {
        self.handle_record_insert_batch_for_tenant(
            RichRecordBatchRequest {
                collection_id: collection_id.to_string(),
                records,
            },
            tenant_id,
        )
        .await
    }

    async fn upsert_record_batch(
        &self,
        collection_id: &str,
        records: Vec<proximadb_records::ProximaRecord>,
        tenant_id: Option<&str>,
    ) -> Result<proximadb_runtime::BatchOperationResult> {
        self.handle_record_batch_for_tenant(
            RichRecordBatchRequest {
                collection_id: collection_id.to_string(),
                records,
            },
            tenant_id,
        )
        .await
    }

    async fn delete_record_batch(
        &self,
        collection_id: &str,
        record_ids: Vec<String>,
        tenant_id: Option<&str>,
    ) -> Result<proximadb_runtime::BatchOperationResult> {
        self.handle_record_delete_batch_for_tenant(
            RichRecordDeleteBatchRequest {
                collection_id: collection_id.to_string(),
                record_ids,
            },
            tenant_id,
        )
        .await
    }

    // ── ADR-094: the REST v2 handler surface (records.rs consumers). The
    // inherent methods win method resolution over the trait methods, so these
    // are pure delegations, not recursion. ──

    async fn handle_record_batch_for_tenant(
        &self,
        request: RichRecordBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        RecordOpsService::handle_record_batch_for_tenant(self, request, tenant_id).await
    }

    async fn handle_record_delete_batch_for_tenant(
        &self,
        request: RichRecordDeleteBatchRequest,
        tenant_id: Option<&str>,
    ) -> Result<BatchOperationResult> {
        RecordOpsService::handle_record_delete_batch_for_tenant(self, request, tenant_id).await
    }

    async fn handle_record_get_for_tenant(
        &self,
        request: RichRecordGetRequest,
        identity: proximadb_runtime::PortIdentity<'_>,
    ) -> Result<RichRecordGetResponse> {
        RecordOpsService::handle_record_get_for_tenant(self, request, identity).await
    }

    async fn handle_record_scan_paginated_for_tenant(
        &self,
        collection_id: &str,
        cursor: Option<&ScanCursor>,
        limit: usize,
        include_vector: bool,
        include_props: bool,
        tenant_id: Option<&str>,
        filter: Option<&FilterExpression>,
        now_ns: i64,
    ) -> Result<(Vec<proximadb_records::ProximaRecord>, Option<ScanCursor>)> {
        RecordOpsService::handle_record_scan_paginated_for_tenant(
            self,
            collection_id,
            cursor,
            limit,
            include_vector,
            include_props,
            tenant_id,
            filter,
            now_ns,
        )
        .await
    }

    async fn handle_record_search_for_tenant(
        &self,
        request: RichSearchRequest,
        identity: proximadb_runtime::PortIdentity<'_>,
    ) -> Result<RichSearchResponse> {
        RecordOpsService::handle_record_search_for_tenant(self, request, identity).await
    }
}

// TD-FLIGHT-1: canonical v2 search read port. The Arrow Flight search surfaces
// (do_get + do_exchange bulk_search) consume this instead of the deprecated v1
// `ApiHandlersPort::handle_vector_search_v1_for_tenant`, so they inherit the
// same typed-filter, WAL delta-merge, MVCC/tombstone, Strong-freshness, and
// tenant-collection-access behavior REST v2 / gRPC v2 get. Pure delegation to
// the single canonical search authority.
#[async_trait::async_trait]
impl proximadb_runtime::RecordSearchPort for RecordOpsService {
    async fn search_record(
        &self,
        request: RichSearchRequest,
        identity: proximadb_runtime::PortIdentity<'_>,
    ) -> Result<RichSearchResponse> {
        self.handle_record_search_for_tenant(request, identity)
            .await
    }
}

// ADR-009 convergence: the document facade's canonical branch routes here so a
// document written via gRPC/DocumentService lands in the same tenant-scoped record
// store REST v2 already uses — closing the store-split. Writes upsert (a re-inserted
// document id updates in place); reads use the paginated scan, which returns full
// `ProximaRecord`s (labels + props) needed to rebuild the document facade.
#[async_trait::async_trait]
impl proximadb_runtime::RecordRoutePort for RecordOpsService {
    async fn insert_records(
        &self,
        collection_id: &str,
        records: Vec<proximadb_records::ProximaRecord>,
        tenant: Option<&str>,
    ) -> Result<usize> {
        let result = self
            .handle_record_batch_for_tenant(
                RichRecordBatchRequest {
                    collection_id: collection_id.to_string(),
                    records,
                },
                tenant,
            )
            .await?;
        if result.success {
            Ok(result.vector_ids.len())
        } else {
            Err(anyhow!(
                "record route insert failed: {}",
                result.errors.join("; ")
            ))
        }
    }

    async fn get_record(
        &self,
        collection_id: &str,
        record_id: &str,
        tenant: Option<&str>,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        self.handle_record_get_full_for_tenant(collection_id, record_id, tenant)
            .await
    }

    async fn scan_records(
        &self,
        collection_id: &str,
        limit: usize,
        tenant: Option<&str>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let (records, _next_cursor) = self
            .handle_record_scan_paginated_for_tenant(
                collection_id,
                None,
                limit,
                false,
                true,
                tenant,
                None,
                now_ns,
            )
            .await?;
        Ok(records)
    }

    async fn collection_exists(&self, collection_id: &str, tenant: Option<&str>) -> bool {
        // Resolve through the SAME catalog path the write path uses (so the answer matches
        // whether an insert would resolve), tenant-scoped. Any error ⇒ false (fail toward
        // the safe legacy path) so the default-ON document gate never hard-fails a write to
        // a non-canonical collection.
        let tenant_context = match self.collection_service.load_tenant_context(tenant) {
            Ok(tc) => tc,
            Err(_) => return false,
        };
        matches!(
            self.resolve_collection_id_internal(collection_id, tenant_context.as_ref())
                .await,
            Ok(Some(_))
        )
    }

    async fn ensure_collection(
        &self,
        collection_id: &str,
        dimension: u32,
        tenant: Option<&str>,
        promote_keys: &[String],
    ) -> Result<()> {
        // Idempotent create-if-not-exists so a document collection becomes a resolvable canonical
        // collection (P-Provision, ADR-055). Delegates to the CollectionService helper (which owns
        // the v1 `CollectionConfig` construction) — this keeps `record_ops_service` v1-proto-free
        // (TD-123 ratchet). `dimension == 0` ⇒ a vectorless (pure-document) collection.
        // `promote_keys` seed props-auto-promotion so declared hot fields shred (P-Shred follow-up).
        let tenant_context = self.collection_service.load_tenant_context(tenant)?;
        self.collection_service
            .get_or_create_by_name(
                collection_id,
                dimension,
                tenant_context.as_ref(),
                promote_keys,
            )
            .await
    }

    async fn pax_scan_inputs(
        &self,
        collection_id: &str,
        tenant: Option<&str>,
    ) -> Option<proximadb_runtime::PaxScanInputs> {
        use proximadb_runtime::{PaxColumnDesc, PaxScanInputs};
        // Resolve the collection's catalog schema through the SAME catalog the write
        // path uses, tenant-scoped. Any miss ⇒ None (the caller falls back to the
        // in-memory document scan — mixed-read-safe).
        let catalog_manager = self.collection_service.catalog_manager()?;
        let (catalog, table_id) = catalog_manager
            .resolve_table_scoped(collection_id, tenant)
            .await
            .ok()?;
        let schema = catalog.get_table(&table_id).await.ok()?;
        let tenant_context = self.collection_service.load_tenant_context(tenant).ok()?;
        let base_path = crate::services::record_store::object_store_write_base_path(
            &schema,
            tenant_context.as_ref(),
        );
        // Shredded promoted columns: prop key (SQL-facing) → the `props__<key>` catalog
        // column's id + type. Mirrors the write-path shred spec (record_store.rs) so the
        // reader keys the SAME column ids that were written.
        let columns = schema
            .props_auto_promotion
            .promoted_keys
            .iter()
            .filter_map(|(prop_key, col_name)| {
                schema
                    .columns
                    .iter()
                    .find(|c| &c.name == col_name)
                    .map(|c| PaxColumnDesc {
                        sql_name: prop_key.clone(),
                        col_id: c.id,
                        data_type: c.data_type.clone(),
                    })
            })
            .collect();
        Some(PaxScanInputs { base_path, columns })
    }

    async fn unflushed_records(
        &self,
        collection_id: &str,
        tenant: Option<&str>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let tenant_context = self.collection_service.load_tenant_context(tenant)?;
        // Unknown collection ⇒ empty (mixed-safe: the caller falls back / merges nothing).
        let Some(resolved_id) = self
            .resolve_collection_id_internal(collection_id, tenant_context.as_ref())
            .await?
        else {
            return Ok(Vec::new());
        };
        self.vector_operations_service
            .list_unflushed_raw_with_tenant_context(&resolved_id, tenant_context.as_ref())
            .await
    }

    async fn delete_records(
        &self,
        collection_id: &str,
        record_ids: Vec<String>,
        tenant: Option<&str>,
    ) -> Result<usize> {
        let result = self
            .handle_record_delete_batch_for_tenant(
                RichRecordDeleteBatchRequest {
                    collection_id: collection_id.to_string(),
                    record_ids,
                },
                tenant,
            )
            .await?;
        if result.success {
            Ok(result.vector_ids.len())
        } else {
            Err(anyhow!(
                "record route delete failed: {}",
                result.errors.join("; ")
            ))
        }
    }
}
