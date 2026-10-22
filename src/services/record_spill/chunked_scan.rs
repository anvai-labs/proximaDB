//! Bounded-memory, oid-ordered scan of a [`SpillRecordStorage`] partition (P2.4).
//!
//! `merged_records` reads every segment whole into one map. This scan instead
//! k-way merges the segments — each one oid-sorted (`flush` sorts and declares
//! `sorting_columns`) — under a snapshot of the memtable, reading each segment one
//! row-group-sized batch at a time through `FileSystemParquetReader` ranged
//! reads. Resident memory is one decoded batch per segment plus the memtable snapshot (bounded by the flush
//! threshold), not the table.
//!
//! The view is `SpillRecordStorage::view` — segment list, resident rows and
//! tombstones taken together under the segments lock — the same flush-consistent
//! view `merged_records` reads, so a flush landing mid-scan neither loses rows nor
//! resurrects deletes. Segments are immutable once published. Every row is also
//! checked against `is_dead`, as `merged_records` does.

use std::sync::Arc;

use futures::StreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::file::metadata::ParquetMetaData;
use proximadb_records::chunked::{
    RecordChunkStream, chunks_of_records, effective_chunk_rows, filter_chunks, flatten_setup,
    merge_newest_wins,
};
use proximadb_records::{RecordScanOptions, RecordScanView};
use proximadb_storage_common::proxima_arrow::{SYS_COL_OID, record_batch_to_proxima_records};
use proximadb_storage_filesystem_types::FileSystem;

use crate::storage::persistence::filesystem::parquet_async_reader::FileSystemParquetReader;

use super::{SpillRecordStorage, now_ns};

/// Whether every row group of `metadata` declares itself sorted ascending by
/// the oid column. Segments written before the sorted layout do not, and are
/// served sorted in memory instead (they predate the bound, not the contract).
fn sorted_by_oid(metadata: &ParquetMetaData) -> bool {
    let Some(oid_column) = metadata
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|c| c.path().string() == SYS_COL_OID)
    else {
        return false;
    };
    metadata.row_groups().iter().all(|rg| {
        rg.sorting_columns().is_some_and(|cols| {
            cols.first()
                .is_some_and(|c| c.column_idx as usize == oid_column && !c.descending)
        })
    })
}

/// One segment as an ordered source: ranged row-group batches when the segment
/// declares the oid sort, else a whole-segment read sorted in memory.
async fn segment_source(
    fs: Arc<dyn FileSystem>,
    path: String,
    chunk_rows: usize,
) -> anyhow::Result<RecordChunkStream<'static>> {
    let reader = FileSystemParquetReader::new(fs.clone(), path.clone());
    let builder = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .map_err(|e| anyhow::anyhow!("spill: open segment '{path}' failed: {e}"))?;
    if !sorted_by_oid(builder.metadata()) {
        let bytes = fs
            .read(&path)
            .await
            .map_err(|e| anyhow::anyhow!("spill: read segment '{path}' failed: {e}"))?;
        let records = crate::storage::engines::sst::segment_format::read_segment_records(
            &bytes,
            &[],
            &[],
            None,
        )
        .map_err(|e| anyhow::anyhow!("spill: decode segment '{path}': {e}"))?;
        return Ok(chunks_of_records(records, chunk_rows));
    }
    let batches = builder
        .with_batch_size(chunk_rows)
        .build()
        .map_err(|e| anyhow::anyhow!("spill: stream segment '{path}' failed: {e}"))?;
    Ok(batches
        .map(move |batch| {
            batch
                .map(|b| record_batch_to_proxima_records(&b))
                .map_err(|e| anyhow::anyhow!("spill: decode segment '{path}': {e}"))
        })
        .boxed())
}

impl SpillRecordStorage {
    /// The ordered chunk scan behind `RecordScan::scan_record_chunks`; see the
    /// module docs.
    pub(super) fn chunked_scan(
        self: Arc<Self>,
        options: RecordScanOptions,
        chunk_rows: usize,
    ) -> RecordChunkStream<'static> {
        // Parquet reads a batch size of zero as end-of-file.
        let chunk_rows = effective_chunk_rows(chunk_rows);
        flatten_setup(async move {
            let RecordScanView {
                resident,
                tombstones,
                segments,
            } = self.view().await?;
            let tombstoned: std::collections::HashSet<String> = tombstones.into_iter().collect();
            // Oldest → newest, the memtable on top: newest-wins is the read rule.
            let mut sources = Vec::with_capacity(segments.len() + 1);
            for path in segments {
                sources.push(segment_source(self.filesystem.clone(), path, chunk_rows).await?);
            }
            sources.push(chunks_of_records(resident, chunk_rows));
            let merged = merge_newest_wins(sources, chunk_rows);
            let now = now_ns();
            anyhow::Ok(filter_chunks(merged, move |record| {
                !tombstoned.contains(&record.oid)
                    && !record.is_dead(now)
                    && options.matches_record(record)
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use futures::TryStreamExt;
    use proximadb_records::{
        ProximaRecord, ProximaTreeNode, ProximaValue, RecordKey, RecordScan, RecordStore,
    };

    async fn store(threshold: Option<usize>) -> Arc<SpillRecordStorage> {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir);
        let fs = LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem");
        Arc::new(SpillRecordStorage::with_flush_threshold(
            Arc::new(fs),
            base,
            threshold,
        ))
    }

    fn record(oid: &str, version: i64) -> ProximaRecord {
        let mut r = ProximaRecord {
            oid: oid.to_string(),
            ..Default::default()
        };
        r.props.insert(
            "v".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(version)),
        );
        r
    }

    async fn chunked(s: &Arc<SpillRecordStorage>, chunk_rows: usize) -> Vec<Vec<ProximaRecord>> {
        s.clone()
            .scan_record_chunks(RecordScanOptions::unbounded(), chunk_rows)
            .try_collect()
            .await
            .expect("chunked scan")
    }

    fn by_oid(mut records: Vec<ProximaRecord>) -> Vec<(String, Option<ProximaTreeNode>)> {
        records.sort_by(|a, b| a.oid.cmp(&b.oid));
        records
            .into_iter()
            .map(|r| (r.oid.clone(), r.props.get("v").cloned()))
            .collect()
    }

    /// Across several segments, updates shadowing flushed copies, deletes of
    /// flushed rows (persisted tombstones and in-memory ones) and resident rows,
    /// the chunked scan returns exactly what the materializing scan does — in
    /// strictly ascending oid order and in chunks no larger than asked.
    #[tokio::test]
    async fn chunked_scan_equals_the_merged_scan() -> anyhow::Result<()> {
        let s = store(Some(7)).await;
        for i in 0..40 {
            s.upsert_record(record(&format!("o{i:03}"), 1)).await?;
        }
        for i in (0..40).step_by(3) {
            s.upsert_record(record(&format!("o{i:03}"), 2)).await?;
        }
        for i in (0..40).step_by(5) {
            s.delete_record(&RecordKey::new(format!("o{i:03}"))).await?;
        }
        s.upsert_record(record("o005", 3)).await?; // re-insert after delete
        assert!(s.segment_count() >= 3, "precondition: several segments");
        assert!(s.resident_len() > 0, "precondition: resident rows too");

        let chunks = chunked(&s, 4).await;
        assert!(chunks.iter().all(|c| !c.is_empty() && c.len() <= 4));
        let flat: Vec<ProximaRecord> = chunks.into_iter().flatten().collect();
        let oids: Vec<&str> = flat.iter().map(|r| r.oid.as_str()).collect();
        assert!(
            oids.windows(2).all(|w| w[0] < w[1]),
            "strictly ascending oids: {oids:?}"
        );
        assert_eq!(by_oid(flat), by_oid(s.scan_records(usize::MAX).await?));
        // A chunk size of zero is one, not an empty segment read.
        let ones: Vec<ProximaRecord> = chunked(&s, 0).await.into_iter().flatten().collect();
        assert_eq!(by_oid(ones), by_oid(s.scan_records(usize::MAX).await?));
        Ok(())
    }

    /// A segment written before segments were oid-sorted declares no sort: it is
    /// served sorted in memory instead of breaking the merge.
    #[tokio::test]
    async fn an_unsorted_legacy_segment_is_served_in_order() -> anyhow::Result<()> {
        let s = store(None).await;
        for oid in ["m", "c", "x"] {
            s.upsert_record(record(oid, 1)).await?;
        }
        s.flush().await?;
        // Rewrite the segment the legacy way: unsorted, no declared sort.
        let path = s.segments.read()[0].clone();
        let records = vec![record("x", 1), record("c", 1), record("m", 1)];
        let schema = infer_test_schema(&records);
        let batch =
            proximadb_storage_common::proxima_arrow::proxima_records_to_record_batch_with_system_columns(
                &records, &schema,
            )?;
        let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
            &[batch.clone()],
            batch.schema(),
            None,
        )?;
        s.filesystem.write(&path, &bytes, None).await?;
        s.upsert_record(record("d", 2)).await?;

        let flat: Vec<String> = chunked(&s, 2)
            .await
            .into_iter()
            .flatten()
            .map(|r| r.oid)
            .collect();
        assert_eq!(flat, vec!["c", "d", "m", "x"]);
        Ok(())
    }

    fn infer_test_schema(
        records: &[ProximaRecord],
    ) -> proximadb_storage_common::proxima_schema::ProximaSchema {
        proximadb_storage_common::proxima_arrow::infer_proxima_schema(records)
    }
}
