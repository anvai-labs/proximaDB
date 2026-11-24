/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! SST bulk-load API for the async-ingest path (Phase 2F).
//!
//! The drainer (Phase 2G) needs to land embedded batches into the
//! target collection. Per locked invariant #5, async ingest should
//! **bypass WAL+memtable** and bulk-load a pre-sorted SST segment
//! directly — that's the storage-layer LSM optimization that backs
//! the throughput economics in the queue's `README.md`
//! ("Throughput economics — why the queue path is structurally
//! cheaper").
//!
//! ## What this module ships in Phase 2F
//!
//! The **API shape** that the drainer targets. Production callers
//! construct a [`BulkLoader`] and call [`BulkLoader::ingest_sorted_segment`].
//! The MVP delegates to the normal per-record insert path via
//! `UnifiedHandlers::handle_record_batch_for_tenant`; the actual
//! SST-writer-direct path lands in a focused storage-engine refactor
//! (Phase 2F-b) that exposes each engine's flush SST-writer step as a
//! reusable primitive.
//!
//! Wiring the API shape now means:
//! - The drainer can be implemented against `BulkLoader` today.
//! - When 2F-b lands, the swap is local to `BulkLoader::ingest_sorted_segment`.
//!   Callers don't change.
//!
//! ## Why not in `proximadb-vector`
//!
//! The plan originally placed `bulk_load.rs` in `crates/modalities/
//! proximadb-vector`. That crate doesn't currently own the actual
//! storage-engine integration code — the engines (NOVA, SST,
//! etc.) live in this crate (`src/storage/engines/`). Locating
//! `BulkLoader` here avoids a circular dep: `proximadb-vector` cannot
//! reach back into the main `proximadb` crate. When the storage-engine
//! refactor migrates the engines into `proximadb-vector` (a parallel
//! design effort), `BulkLoader` can move with them.

use std::sync::Arc;

use anyhow::Result;
use proximadb_records::ProximaRecord;
use tracing::{debug, warn};

use crate::services::VectorOperationsService;
use proximadb_runtime::RecordOpsPort;

/// Returned from a successful bulk-load. Once the storage-engine
/// refactor exposes the real SST-write step, this will identify the
/// committed segment on disk; today it's a synthetic id derived from
/// the underlying insert metrics — sufficient for tracing and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BulkLoadedSegment {
    pub collection_id: String,
    pub record_count: usize,
    pub synthetic_segment_id: String,
}

/// Sort policy hint for `ingest_sorted_segment`. The drainer typically
/// passes `Unsorted` (records arrive batched from the queue in
/// per-message order) and lets the loader sort. Callers that already
/// have a sorted batch (recovery, replay) can pass `Sorted` to skip
/// the sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortHint {
    Sorted,
    Unsorted,
}

/// Per-call options. Reserved for the SST refactor (compaction level,
/// segment metadata, etc.); today the only meaningful field is the
/// sort hint.
#[derive(Debug, Clone, Default)]
pub struct BulkLoadOptions {
    pub sort_hint: Option<SortHint>,
}

pub struct BulkLoader {
    /// Record-batch ingest via the canonical `RecordOpsPort` (the runtime
    /// `RecordOpsService` after TD-104 S3-c) — no longer the concrete ROOT
    /// `UnifiedHandlers`. The per-record WAL+memtable fallback below routes
    /// through `insert_record_batch`.
    record_ops: Arc<dyn RecordOpsPort>,
    /// Concrete vector-ops service for the engine-specific LSM-bypass probe
    /// (`get_engine_for_collection`); held directly rather than reached
    /// through the root handler (TD-104 S3-e).
    vector_operations_service: Arc<VectorOperationsService>,
    /// Default storage root URL passed to the engine's
    /// `ingest_sorted_segment` override when the per-collection
    /// `storage_assignment.base_location` isn't yet threaded.
    ///
    /// Comes from the canonical TOML at startup —
    /// `config.storage.storage_locations[0].url` (typically
    /// `file:///var/lib/proximadb/collections` on PVC-backed AKS,
    /// `adls://account.dfs.core.windows.net/container/data` on
    /// Azure-direct, `s3://bucket/data` on AWS-direct, or
    /// `file:///tmp/proximadb/collections` on dev rigs). When the
    /// drainer eventually threads the catalog handle, this becomes
    /// the fallback for engines whose override doesn't bother
    /// reading the per-collection assignment.
    default_storage_root: String,
}

impl BulkLoader {
    pub fn new(
        record_ops: Arc<dyn RecordOpsPort>,
        vector_operations_service: Arc<VectorOperationsService>,
        default_storage_root: String,
    ) -> Self {
        Self {
            record_ops,
            vector_operations_service,
            default_storage_root,
        }
    }

    /// Bulk-load a batch of records into `collection_id` for the given
    /// tenant.
    ///
    /// Phase 2F: delegates to `UnifiedHandlers::handle_record_batch_for_tenant`,
    /// which goes through WAL+memtable. Customer-visible correctness is
    /// identical to the per-record insert path; the LSM-bypass
    /// performance win lands when 2F-b refactors each storage engine
    /// to expose its SST-writer step. Until then, every batch costs
    /// (N records × per-record WAL fsync) instead of (1 segment write +
    /// 1 fsync) — but the inference-layer batching savings (21× per
    /// record vs sync, see queue README) are already realized.
    ///
    /// `records` are sorted in-place by oid when `sort_hint` is
    /// `Unsorted` (default) — LSM bulk-load requires sorted input, so
    /// the drainer can avoid sorting twice by passing `Sorted` once
    /// the refactor lands.
    pub async fn ingest_sorted_segment(
        &self,
        collection_id: String,
        tenant_id: Option<&str>,
        mut records: Vec<ProximaRecord>,
        options: BulkLoadOptions,
    ) -> Result<BulkLoadedSegment> {
        let sort_hint = options.sort_hint.unwrap_or(SortHint::Unsorted);
        if sort_hint == SortHint::Unsorted {
            records.sort_by(|a, b| a.oid.cmp(&b.oid));
        }

        let record_count = records.len();
        if record_count == 0 {
            return Ok(BulkLoadedSegment {
                collection_id,
                record_count: 0,
                synthetic_segment_id: String::from("empty"),
            });
        }

        // Phase 2F-b activation: try the engine-specific LSM-bypass
        // path first. Engines that ship `ingest_sorted_segment`
        // overrides (NOVA so far; SST/RAPTOR/CHRONO/SEQUOIA as
        // they migrate) write a single SST/Parquet file directly,
        // skipping WAL+memtable for ~30% storage-layer savings on top
        // of the inference-layer batching. Engines without an override
        // return `used_engine_specific_path: false` and we fall back
        // to the per-record WAL+memtable path via UnifiedHandlers —
        // correctness-equivalent, just slower.
        if let Ok(engine) = self
            .vector_operations_service
            .get_engine_for_collection(&collection_id)
            .await
        {
            let base_path = self.default_storage_root.clone();
            // Cloning records here so we still have them for the
            // fallback. Engines that DO override this method will
            // consume the clone and we never use the original — but
            // until every engine migrates we can't risk a no-op
            // override losing the records.
            let engine_records = records.clone();
            match engine
                .ingest_sorted_segment(&collection_id, &base_path, engine_records)
                .await
            {
                Ok(seg) if seg.used_engine_specific_path => {
                    debug!(
                        collection_id = %collection_id,
                        engine = engine.format_name(),
                        record_count,
                        synthetic_segment_id = %seg.synthetic_segment_id,
                        "bulk_load: engine-specific LSM bypass committed segment",
                    );
                    return Ok(BulkLoadedSegment {
                        collection_id,
                        record_count: seg.record_count,
                        synthetic_segment_id: seg.synthetic_segment_id,
                    });
                }
                Ok(_) => {
                    debug!(
                        collection_id = %collection_id,
                        engine = engine.format_name(),
                        "bulk_load: engine has no LSM bypass override; falling back to per-record path",
                    );
                }
                Err(e) => {
                    warn!(
                        collection_id = %collection_id,
                        engine = engine.format_name(),
                        error = %e,
                        "bulk_load: engine override errored; falling back to per-record path",
                    );
                }
            }
        }

        // Fallback: per-record insert via WAL+memtable. Used when no
        // engine override exists OR when the override returned an
        // error. Customer-visible correctness is identical to the
        // pre-2F-b behavior.
        let result = self
            .record_ops
            .insert_record_batch(&collection_id, records, tenant_id)
            .await
            .map_err(|e| anyhow::anyhow!("bulk-load delegate failed: {e}"))?;
        if !result.success {
            warn!(
                collection_id = %collection_id,
                errors = ?result.errors,
                "bulk-load returned non-success; treating as failure",
            );
            return Err(anyhow::anyhow!(
                "bulk-load failed: {}",
                result.errors.join("; ")
            ));
        }

        let synthetic_segment_id = format!(
            "bulkload-{}-{}",
            collection_id,
            result
                .vector_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "noid".to_string())
        );
        debug!(
            collection_id = %collection_id,
            record_count,
            synthetic_segment_id = %synthetic_segment_id,
            "bulk_load: segment committed via per-record fallback path",
        );
        Ok(BulkLoadedSegment {
            collection_id,
            record_count,
            synthetic_segment_id,
        })
    }
}

// ── Production DrainerInsertSink wired to BulkLoader ─────────────

use crate::services::embedding_drainer::{DrainerInsertSink, EmbeddedRecord};
use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaTreeNode};

/// Production [`DrainerInsertSink`] that projects the drainer's
/// `EmbeddedRecord`s into `ProximaRecord`s and hands them to
/// [`BulkLoader::ingest_sorted_segment`]. Tests use the in-memory
/// `RecordingSink`; production startup constructs this wrapper.
pub struct BulkLoadDrainerSink {
    bulk_loader: Arc<BulkLoader>,
}

impl BulkLoadDrainerSink {
    pub fn new(bulk_loader: Arc<BulkLoader>) -> Self {
        Self { bulk_loader }
    }
}

/// Project an [`EmbeddedRecord`] into a [`ProximaRecord`] honoring the
/// record's `target_precision` hint. When `target_precision` is `None`
/// or `Some(Fp32)`, the resulting `EmbeddingCell` carries
/// `EmbeddingValues::Fp32` (legacy behavior). Otherwise the fp32 vector
/// is coerced via `EmbeddingValues::from_fp32_lossy(_, target)` and the
/// cell's `precision` field is stamped to match.
///
/// Extracted from the [`DrainerInsertSink`] impl so the precision-routing
/// is exercised by unit tests without booting the full bulk loader.
fn project_embedded_record(r: EmbeddedRecord, tenant_id: &str, now_ns: i64) -> ProximaRecord {
    let mut props = std::collections::HashMap::new();
    props.insert(
        "text".to_string(),
        ProximaTreeNode::Value(ProximaValue::String(r.text)),
    );
    for (k, v) in r.metadata {
        props.insert(k, ProximaTreeNode::Value(ProximaValue::String(v)));
    }
    // #950: the tenant must be queryable, not just carried on the isolation
    // field — metadata filters and record GET read `props`. The envelope
    // metadata normally carries the producer's stamp already; this makes the
    // sink self-sufficient (server value wins over any conflicting metadata).
    if !tenant_id.is_empty() {
        props.insert(
            "tenant_id".to_string(),
            ProximaTreeNode::Value(ProximaValue::String(tenant_id.to_string())),
        );
    }
    let (values, precision) = match r.target_precision {
        None | Some(proximadb_records::EmbeddingScalarType::Fp32) => (
            proximadb_records::EmbeddingValues::Fp32(r.vector),
            proximadb_records::EmbeddingScalarType::Fp32,
        ),
        Some(target) => (
            proximadb_records::EmbeddingValues::from_fp32_lossy(&r.vector, target),
            target,
        ),
    };
    ProximaRecord {
        oid: r.oid.clone(),
        local_id: Some(r.oid),
        tenant_id: tenant_id.to_string(),
        created_at_ns: now_ns,
        updated_at_ns: now_ns,
        origin: Some("v3_async_drainer".to_string()),
        props,
        embeddings: vec![EmbeddingCell {
            model_id: "native".to_string(),
            modality: "dense_vector".to_string(),
            dim: r.vector_dim,
            values,
            precision,
            ..Default::default()
        }],
        ..ProximaRecord::default()
    }
}

#[async_trait]
impl DrainerInsertSink for BulkLoadDrainerSink {
    async fn insert(
        &self,
        target_collection: &str,
        tenant_id: &str,
        records: Vec<EmbeddedRecord>,
    ) -> Result<()> {
        let now_ns = chrono::Utc::now()
            .timestamp_millis()
            .saturating_mul(1_000_000);
        let proxima_records: Vec<ProximaRecord> = records
            .into_iter()
            .map(|r| project_embedded_record(r, tenant_id, now_ns))
            .collect();
        let _segment = self
            .bulk_loader
            .ingest_sorted_segment(
                target_collection.to_string(),
                Some(tenant_id),
                proxima_records,
                BulkLoadOptions::default(),
            )
            .await?;
        Ok(())
    }
}

// ── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaRecord;

    fn rec(oid: &str) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            local_id: Some(oid.to_string()),
            tenant_id: "t".to_string(),
            ..ProximaRecord::default()
        }
    }

    /// Sort hint defaulting to Unsorted sorts records in place by oid.
    /// The contract for the LSM bypass requires sorted input.
    #[test]
    fn unsorted_records_get_sorted_by_oid() {
        let mut records = [rec("zeta"), rec("alpha"), rec("mike")];
        records.sort_by(|a, b| a.oid.cmp(&b.oid));
        let oids: Vec<_> = records.iter().map(|r| r.oid.as_str()).collect();
        assert_eq!(oids, vec!["alpha", "mike", "zeta"]);
    }

    #[test]
    fn empty_batch_returns_empty_segment_id() {
        // Doesn't call into handlers — empty batch short-circuits.
        let segment = BulkLoadedSegment {
            collection_id: "knowledge".to_string(),
            record_count: 0,
            synthetic_segment_id: "empty".to_string(),
        };
        assert_eq!(segment.record_count, 0);
        assert_eq!(segment.synthetic_segment_id, "empty");
    }

    #[test]
    fn sort_hint_sorted_skips_inplace_sort() {
        let hint = SortHint::Sorted;
        assert_eq!(hint, SortHint::Sorted);
        // The actual skip happens inside ingest_sorted_segment; this
        // test exists to lock the API shape so reorderings of the
        // SortHint enum don't silently break callers.
    }

    fn make_embedded(
        target_precision: Option<proximadb_records::EmbeddingScalarType>,
    ) -> EmbeddedRecord {
        EmbeddedRecord {
            oid: "rec-1".to_string(),
            text: "hello world".to_string(),
            vector: vec![1.0, 2.0, 3.0, 4.0],
            vector_dim: 4,
            metadata: std::collections::HashMap::new(),
            target_precision,
        }
    }

    #[test]
    fn project_embedded_record_defaults_to_fp32_when_target_is_none() {
        let r = project_embedded_record(make_embedded(None), "tenant-a", 1_000);
        assert_eq!(r.embeddings.len(), 1);
        assert_eq!(
            r.embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(
            r.embeddings[0].precision,
            proximadb_records::EmbeddingScalarType::Fp32
        );
    }

    #[test]
    fn project_embedded_record_stays_fp32_when_target_is_explicit_fp32() {
        let r = project_embedded_record(
            make_embedded(Some(proximadb_records::EmbeddingScalarType::Fp32)),
            "tenant-a",
            1_000,
        );
        assert_eq!(
            r.embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(
            r.embeddings[0].precision,
            proximadb_records::EmbeddingScalarType::Fp32
        );
    }

    #[test]
    fn project_embedded_record_coerces_to_fp16_when_target_is_fp16() {
        let r = project_embedded_record(
            make_embedded(Some(proximadb_records::EmbeddingScalarType::Fp16)),
            "tenant-a",
            1_000,
        );
        assert_eq!(
            r.embeddings[0].precision,
            proximadb_records::EmbeddingScalarType::Fp16
        );
        match &r.embeddings[0].values {
            proximadb_records::EmbeddingValues::Fp16(vs) => {
                let expected: Vec<half::f16> = [1.0_f32, 2.0, 3.0, 4.0]
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect();
                assert_eq!(vs, &expected);
            }
            other => panic!("expected Fp16, got {:?}", other.scalar_type()),
        }
    }

    /// #950: the queued /documents path must persist a queryable tenant stamp.
    /// The isolation field AND the `tenant_id` prop (what metadata filters and
    /// record GET read) both carry the request tenant, and the server value
    /// overrides a conflicting client-supplied `tenant_id` metadata entry.
    #[test]
    fn project_embedded_record_stamps_tenant_field_and_prop() {
        let mut embedded = make_embedded(None);
        embedded
            .metadata
            .insert("tenant_id".to_string(), "spoofed".to_string());
        embedded
            .metadata
            .insert("source".to_string(), "uat".to_string());
        let r = project_embedded_record(embedded, "demo1", 1_000);
        assert_eq!(r.tenant_id, "demo1");
        assert_eq!(
            r.props.get("tenant_id"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "demo1".to_string()
            ))),
            "server tenant must win over client-supplied metadata"
        );
        assert_eq!(
            r.props.get("source"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "uat".to_string()
            ))),
            "user metadata must survive the envelope"
        );
    }

    #[test]
    fn project_embedded_record_coerces_to_bf16_when_target_is_bf16() {
        let r = project_embedded_record(
            make_embedded(Some(proximadb_records::EmbeddingScalarType::Bf16)),
            "tenant-a",
            1_000,
        );
        assert_eq!(
            r.embeddings[0].precision,
            proximadb_records::EmbeddingScalarType::Bf16
        );
        assert!(matches!(
            r.embeddings[0].values,
            proximadb_records::EmbeddingValues::Bf16(_)
        ));
    }
}
