//! WAL write-primitive collaborator extracted from `VectorOperationsService`
//! (Phase 2.1 god-object decomposition, slice 5).
//!
//! Owns the lowest layer of the insert write stack: routing a batch to the
//! bulk lane vs the standard WAL path, and persisting records through the WAL
//! manager (with optional insert-only conflict semantics + ingestion-time
//! pseudo-query enrichment). `VectorOperationsService` keeps its public surface
//! (`bulk_write`) and the private `insert_vectors_via_wal*` helpers (internal
//! callers in the delete + batch paths) as thin delegators over a cheap
//! on-demand `VectorWriteCoordinator`.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use tracing::{debug, error, info, warn};

use proximadb_records::ProximaRecord;

use crate::services::operations::{BatchOperationResult, BulkWriteRouter, OperationMetrics};
use crate::storage::persistence::write_ahead_log::WriteAheadLogManager;

use super::resolver::CollectionResolver;
use super::validation::{PseudoQueryGenerator, apply_pseudo_query_metadata};

/// Stamp each record with the request tenant, rejecting any record that already
/// carries a *different* tenant_id (cross-tenant write attempt). Pure helper.
pub(crate) fn ensure_tenant_on_records(
    records: &mut [ProximaRecord],
    tenant_id: &str,
) -> Result<()> {
    for record in records.iter_mut() {
        if !record.tenant_id.is_empty() && record.tenant_id != tenant_id {
            return Err(anyhow::anyhow!(
                "Record '{}' has tenant_id '{}' but request is scoped to tenant '{}'",
                record.oid,
                record.tenant_id,
                tenant_id
            ));
        }
        record.tenant_id = tenant_id.to_string();
    }
    Ok(())
}

/// Build delete-tombstone records (valid_to_ns = 0, origin = "delete") for the
/// given ids at the supplied timestamp. Pure helper.
pub(crate) fn tombstone_records_for_ids(record_ids: &[String], now_ns: i64) -> Vec<ProximaRecord> {
    record_ids
        .iter()
        .map(|id| ProximaRecord {
            oid: id.clone(),
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_to_ns: Some(0),
            origin: Some("delete".to_string()),
            ..Default::default()
        })
        .collect()
}

/// Detect a record id appearing more than once within a single insert request
/// (an in-batch duplicate is an immediate insert conflict). Pure helper.
pub(crate) fn duplicate_insert_conflict_result(
    collection_id: &str,
    records: &[ProximaRecord],
) -> Option<BatchOperationResult> {
    let mut seen_ids = HashSet::new();
    for record in records {
        if !seen_ids.insert(record.oid.as_str()) {
            return Some(BatchOperationResult::failure(
                format!(
                    "Record '{}' appears more than once in insert request for collection '{}'",
                    record.oid, collection_id
                ),
                "INSERT_CONFLICT".to_string(),
            ));
        }
    }

    None
}

/// Build the conflict result for an insert-only request whose id already exists
/// in the collection. Pure helper.
pub(crate) fn insert_existing_record_conflict_result(
    collection_id: &str,
    record_id: &str,
) -> BatchOperationResult {
    BatchOperationResult::failure(
        format!(
            "Record '{}' already exists in collection '{}'",
            record_id, collection_id
        ),
        "INSERT_CONFLICT".to_string(),
    )
}

/// Build the per-tenant+collection lock key guarding insert-only
/// check-and-append operations. Pure helper.
pub(crate) fn insert_only_lock_key(collection_id: &str, tenant_id: Option<&str>) -> String {
    match tenant_id {
        Some(tenant_id) => format!("{tenant_id}:{collection_id}"),
        None => collection_id.to_string(),
    }
}

/// Owns the handles needed to persist a record batch durably: the WAL manager,
/// the bulk-vs-standard routing policy, and the ingestion-time pseudo-query
/// enricher. Holds only `Arc`/cheap-`Clone` handles; constructed on demand by
/// `VectorOperationsService::write_coordinator`.
pub(crate) struct VectorWriteCoordinator {
    wal_manager: Arc<WriteAheadLogManager>,
    bulk_write_router: BulkWriteRouter,
    pseudo_query_generator: Arc<dyn PseudoQueryGenerator>,
    resolver: CollectionResolver,
}

impl VectorWriteCoordinator {
    pub(crate) fn new(
        wal_manager: Arc<WriteAheadLogManager>,
        bulk_write_router: BulkWriteRouter,
        pseudo_query_generator: Arc<dyn PseudoQueryGenerator>,
        resolver: CollectionResolver,
    ) -> Self {
        Self {
            wal_manager,
            bulk_write_router,
            pseudo_query_generator,
            resolver,
        }
    }

    /// Validate a record batch against the target collection's schema
    /// (dimension, embedding finiteness, …). Resolves the collection via the
    /// owned resolver, then runs the pure validators; a collection without a
    /// config is treated as unconstrained (`Ok`).
    pub(crate) async fn validate_records_for_insert(
        &self,
        collection_id: &str,
        records: &[ProximaRecord],
    ) -> Result<()> {
        let collection = self.resolver.get_or_load_collection(collection_id).await?;
        match &collection.config {
            Some(config) => {
                super::input_validation::validate_records_for_insert(collection_id, config, records)
            }
            None => Ok(()),
        }
    }

    /// Insert a batch of canonical records with smart routing: enrich with
    /// pseudo-query metadata, validate against the collection schema, then
    /// route to the bulk lane or the standard WAL path per the router decision.
    pub(crate) async fn insert_batch_internal(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<BatchOperationResult> {
        let mut vectors = vectors;
        apply_pseudo_query_metadata(&mut vectors, &*self.pseudo_query_generator);

        self.validate_records_for_insert(collection_id, &vectors)
            .await
            .map_err(crate::services::record_store::rejected_before_append)?;

        let decision = self.bulk_write_router.route_records(&vectors);

        debug!(
            "📦 insert_batch: collection={}, vectors={}, estimated_size={} bytes, path={}",
            collection_id,
            decision.vector_count,
            decision.estimated_size_bytes,
            if decision.use_bulk_lane {
                "BULK_WAL"
            } else {
                "WAL"
            }
        );

        if decision.use_bulk_lane {
            // Large batch: use bulk write (optimized for throughput)
            info!(
                "🚀 Routing to bulk_write: {} (vectors: {}, size: {} bytes)",
                decision.reason, decision.vector_count, decision.estimated_size_bytes
            );
            self.bulk_write(collection_id, vectors).await
        } else {
            // Small batch: use standard WAL path (optimized for durability)
            debug!(
                "📝 Routing to WAL path: {} (vectors: {}, size: {} bytes)",
                decision.reason, decision.vector_count, decision.estimated_size_bytes
            );
            self.insert_vectors_via_wal(collection_id, vectors).await
        }
    }

    pub(crate) async fn bulk_write(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<BatchOperationResult> {
        let start_time = std::time::Instant::now();
        let vector_count = vectors.len();
        let vector_ids: Vec<String> = vectors.iter().map(|v| v.oid.clone()).collect();
        let decision = self.bulk_write_router.route_records(&vectors);

        info!(
            "📦 Bulk write: collection={}, vectors={}, estimated_size={} bytes, decision={}",
            collection_id,
            vector_count,
            decision.estimated_size_bytes,
            if decision.use_bulk_lane {
                "BULK_WAL"
            } else {
                "WAL"
            }
        );

        // If below thresholds, fall back to standard WAL path
        if !decision.use_bulk_lane {
            debug!(
                "📝 Batch below bulk threshold ({}), using standard WAL path",
                decision.reason
            );
            return self.insert_vectors_via_wal(collection_id, vectors).await;
        }

        // Large-batch path. It remains WAL-backed until direct segment commit
        // has an accepted durability proof.
        info!(
            "🚀 Using WAL-backed bulk path for batch: {} vectors (reason: {})",
            vector_count, decision.reason
        );

        // Write vectors via WAL for durability. A WAL-skipping engine path
        // remains deferred because it needs atomic segment+manifest commit,
        // replay or repair semantics, and idempotency.
        // `vectors` is not used after this point — move it into the Arc rather
        // than cloning. The non-bulk helper below already follows this pattern.
        let vectors_arc = Arc::new(vectors);

        match self
            .wal_manager
            .write_vector_batch_native_arc(collection_id, vectors_arc)
            .await
        {
            Ok(_) => {
                let duration = start_time.elapsed();
                let vectors_per_sec = if duration.as_secs_f64() > 0.0 {
                    (vector_count as f64 / duration.as_secs_f64()) as u64
                } else {
                    vector_count as u64
                };

                info!(
                    "✅ WAL-backed bulk write completed: {} vectors in {:?} ({} vectors/sec)",
                    vector_count, duration, vectors_per_sec
                );

                Ok(BatchOperationResult::success(
                    vector_ids,
                    OperationMetrics {
                        total_processed: vector_count as i64,
                        successful_count: vector_count as i64,
                        failed_count: 0,
                        updated_count: 0,
                        processing_time_us: duration.as_micros() as i64,
                        wal_write_time_us: duration.as_micros() as i64,
                        index_update_time_us: 0,
                    },
                ))
            }
            Err(e) => {
                error!("❌ Bulk write failed: {}", e);
                Err(e)
            }
        }
    }

    /// Internal helper: insert records via standard WAL path
    pub(crate) async fn insert_vectors_via_wal(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<BatchOperationResult> {
        self.insert_vectors_via_wal_with_mode(collection_id, vectors, false)
            .await
    }

    pub(crate) async fn insert_vectors_via_wal_insert_only(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<BatchOperationResult> {
        self.insert_vectors_via_wal_with_mode(collection_id, vectors, true)
            .await
    }

    async fn insert_vectors_via_wal_with_mode(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
        insert_only: bool,
    ) -> Result<BatchOperationResult> {
        // Back-compat wrapper: discard the per-record LSNs. TD-DELVEC-1 WI-3b
        // surfaces them via `insert_vectors_via_wal_returning_lsns` for the
        // cold-delete DV-bit path; callers that don't need the LSNs use this.
        let (result, _lsns) = self
            .insert_vectors_via_wal_with_mode_returning_lsns(collection_id, vectors, insert_only)
            .await?;
        Ok(result)
    }

    /// Insert records via the standard WAL path, returning the per-record WAL/MVCC
    /// LSNs alongside the batch result. TD-DELVEC-1 WI-3b / WI-5 P0: these are the
    /// durable per-batch manifest `global_lsn` (broadcast one-per-record, in input
    /// order) when the batch is synced — the cold-delete path keys deletion-vector
    /// bits on them so the generation shares the read side's `snapshot_lsn` number
    /// space (correct MVCC). Falls back to ephemeral memtable sequences in
    /// MemoryOnly mode (no durability to key on).
    #[cfg(feature = "cold-deletion-vectors")]
    pub(crate) async fn insert_vectors_via_wal_returning_lsns(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<(BatchOperationResult, Vec<u64>)> {
        self.insert_vectors_via_wal_with_mode_returning_lsns(collection_id, vectors, false)
            .await
    }

    async fn insert_vectors_via_wal_with_mode_returning_lsns(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
        insert_only: bool,
    ) -> Result<(BatchOperationResult, Vec<u64>)> {
        let mut vectors = vectors;
        apply_pseudo_query_metadata(&mut vectors, &*self.pseudo_query_generator);

        let start_time = std::time::Instant::now();
        let vector_count = vectors.len();
        let vector_ids: Vec<String> = vectors.iter().map(|v| v.oid.clone()).collect();

        // Write vectors via WAL manager
        let vectors_arc = Arc::new(vectors);

        let wal_result = if insert_only {
            self.wal_manager
                .write_vector_batch_native_arc_insert_only(collection_id, vectors_arc)
                .await
        } else {
            self.wal_manager
                .write_vector_batch_native_arc(collection_id, vectors_arc)
                .await
        };

        match wal_result {
            Ok(lsns) => {
                let duration = start_time.elapsed();
                let _vectors_per_sec = if duration.as_secs_f64() > 0.0 {
                    (vector_count as f64 / duration.as_secs_f64()) as u64
                } else {
                    vector_count as u64
                };

                debug!(
                    "📝 WAL write completed: {} vectors in {:?}",
                    vector_count, duration
                );

                Ok((
                    BatchOperationResult::success(
                        vector_ids,
                        OperationMetrics {
                            total_processed: vector_count as i64,
                            successful_count: vector_count as i64,
                            failed_count: 0,
                            updated_count: 0,
                            processing_time_us: duration.as_micros() as i64,
                            wal_write_time_us: duration.as_micros() as i64,
                            index_update_time_us: 0,
                        },
                    ),
                    lsns,
                ))
            }
            Err(e) => {
                if insert_only && e.to_string().contains("INSERT_CONFLICT") {
                    return Ok((
                        BatchOperationResult::failure(
                            format!("Record insert failed: {}", e),
                            "INSERT_CONFLICT".to_string(),
                        ),
                        vec![],
                    ));
                }
                warn!("WAL batch insert failed: {}", e);
                // ADR-069 S4: preserve the `WalBackpressure` discriminant
                // (→ "WAL_BACKPRESSURE") so the REST/gRPC boundary surfaces a
                // retryable 429 / RESOURCE_EXHAUSTED instead of a non-retryable
                // 400/500. Other errors keep the generic `WAL_WRITE_ERROR` code.
                let error_code = crate::storage::persistence::write_ahead_log::flush_policy::write_batch_error_code(&e, "WAL_WRITE_ERROR");
                Ok((
                    BatchOperationResult::failure(
                        format!("Batch insert failed: {}", e),
                        error_code,
                    ),
                    vec![],
                ))
            }
        }
    }
}
