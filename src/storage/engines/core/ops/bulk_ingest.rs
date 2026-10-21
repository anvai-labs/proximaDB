//! Shared bulk-ingest plumbing for `ingest_sorted_segment` (redundancy
//! pass 3). The four engine copies (SST/NOVA/VIPER/RAPTOR) were
//! line-identical modulo the engine label, the fallback segment-id
//! prefix, and which flush seam they call — that seam stays at each call
//! site; everything else lives here.

use crate::proto::proximadb_v1::{Collection, StorageAssignment};
use crate::storage::traits::{FlushParameters, FlushResult, SegmentIngestResult};

/// The empty-ingest shortcut every engine starts with.
pub fn empty_segment_result(collection_id: &str) -> SegmentIngestResult {
    SegmentIngestResult {
        collection_id: collection_id.to_string(),
        record_count: 0,
        synthetic_segment_id: "empty".to_string(),
        used_engine_specific_path: true,
    }
}

/// Build the synthetic Collection + FlushParameters for a bulk ingest.
///
/// `config = None` lets dimension fall back to inspecting
/// `records[0].embeddings[0].dim` inside the flush path; `base_location`
/// is the only assignment field the flush reads for URL resolution.
/// `engine_label` names the engine in the numeric-catalog-id error (bulk
/// ingest requires a numeric collection id).
#[allow(clippy::too_many_arguments)]
pub fn build_bulk_flush_parameters(
    collection_id: &str,
    base_path: &str,
    records: Vec<proximadb_records::ProximaRecord>,
    engine_label: &str,
) -> anyhow::Result<FlushParameters> {
    let collection_config = Some(Collection {
        id: collection_id.to_string(),
        config: None,
        stats: None,
        created_at: 0,
        updated_at: 0,
        storage_assignment: Some(StorageAssignment {
            primary_path: base_path.to_string(),
            backup_paths: vec![],
            engine: 0,
            engine_config: std::collections::HashMap::new(),
            base_location: base_path.to_string(),
            assigned_at: 0,
            ..Default::default()
        }),
    });

    Ok(FlushParameters {
        collection_id: Some(
            collection_id
                .parse::<u64>()
                .map_err(|error| {
                    anyhow::anyhow!(
                        "{engine_label} bulk ingest requires a numeric catalog object id, got {collection_id:?}: {error}"
                    )
                })?
                .to_string(),
        ),
        force: true,
        synchronous: true,
        hints: std::collections::HashMap::new(),
        timeout_ms: None,
        vector_records: records,
        trigger_compaction: false,
        batch_ids: vec![],
        collection_config,
        estimated_size: 0,
    })
}

/// Map a flush result to the SegmentIngestResult contract: first flushed
/// file path as the synthetic id, else the `{prefix}-bulkload-{id}-{count}`
/// fallback.
pub fn segment_result_from_flush(
    collection_id: &str,
    count: usize,
    flush_result: &FlushResult,
    fallback_prefix: &str,
) -> SegmentIngestResult {
    let synthetic_segment_id = flush_result
        .file_paths
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{fallback_prefix}-bulkload-{collection_id}-{count}"));

    SegmentIngestResult {
        collection_id: collection_id.to_string(),
        record_count: count,
        synthetic_segment_id,
        used_engine_specific_path: true,
    }
}
