//! Compaction operations module for NOVA engine
//! Handles file merging, optimization, and hierarchical statistics management

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::formats::columnar::{
    HybridParquetWriter, HybridWriterConfig, ParquetWriterConfig, UnifiedParquetReader,
};
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::traits::{CompactionParameters, CompactionResult};
use proximadb_records::ProximaRecord;

/// Handles all compaction operations for NOVA engine
pub struct NovaCompactionOperations {
    filesystem: Arc<FilesystemFactory>,
}

impl NovaCompactionOperations {
    /// Create new compaction operations handler
    pub fn new(filesystem: Arc<FilesystemFactory>) -> Self {
        Self { filesystem }
    }

    /// Perform compaction operation
    pub async fn compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
        let start = std::time::Instant::now();

        let collection_id = params.collection_id.as_deref().unwrap_or("default");
        info!(
            "🔄 NOVA: Starting compaction for collection {}",
            collection_id
        );

        // Get base_location from collection config
        let base_location = params
            .collection_config
            .as_ref()
            .and_then(|c| c.storage_assignment.as_ref())
            .map_or("/data/collections", |s| s.base_location.as_str());

        // Use standard path: {base_location}/{collection_id}/data
        let data_path = proximadb_storage_common::storage_path::StoragePath::collection_data_path(
            base_location,
            collection_id,
        );

        debug!(
            "🔄 NOVA compaction: base_location={}, data_path={}",
            base_location, data_path
        );

        // List parquet files in the data directory
        let fs = self.filesystem.get_filesystem(&data_path)?;
        let entries = match fs.list(&data_path).await {
            Ok(e) => e,
            Err(err) => {
                debug!(
                    "🔄 NOVA compaction: Directory {} does not exist or is empty: {}",
                    data_path, err
                );
                return Ok(CompactionResult {
                    success: true,
                    collections_affected: vec![collection_id.to_string()],
                    entries_processed: Some(0),
                    entries_removed: Some(0),
                    bytes_read: Some(0),
                    bytes_written: Some(0),
                    input_files: Some(0),
                    output_files: Some(0),
                    duration_ms: Some(start.elapsed().as_millis() as u64),
                    completed_at: chrono::Utc::now(),
                    engine_metrics: HashMap::new(),
                });
            }
        };

        let input_files: Vec<String> = entries
            .into_iter()
            .filter(|e| !e.metadata.is_directory && e.name.ends_with(".parquet"))
            .map(|e| format!("{}/{}", data_path, e.name))
            .collect();

        debug!(
            "🔄 NOVA compaction: Found {} parquet files",
            input_files.len()
        );

        if input_files.len() < 2 {
            debug!("🔄 NOVA compaction: Not enough files to compact (need at least 2)");
            return Ok(CompactionResult {
                success: true,
                collections_affected: vec![collection_id.to_string()],
                entries_processed: Some(0),
                entries_removed: Some(0),
                bytes_read: Some(0),
                bytes_written: Some(0),
                input_files: Some(input_files.len() as u64),
                output_files: Some(input_files.len() as u64),
                duration_ms: Some(start.elapsed().as_millis() as u64),
                completed_at: chrono::Utc::now(),
                engine_metrics: HashMap::new(),
            });
        }

        // Read all input files
        let mut all_records = Vec::new();
        let mut bytes_before = 0u64;

        info!(
            "🔄 NOVA compaction: Reading {} input files",
            input_files.len()
        );
        for file_path in &input_files {
            let fs_for_meta = self.filesystem.get_filesystem(file_path)?;
            let file_metadata = fs_for_meta.metadata(file_path).await?;
            bytes_before += file_metadata.size;

            // Read records using UnifiedParquetReader
            let records = self.read_parquet_file(file_path, collection_id).await?;
            debug!(
                "🔄 NOVA compaction: Read {} records from {}",
                records.len(),
                file_path
            );
            all_records.extend(records);
        }

        let original_count = all_records.len();
        info!(
            "🔄 NOVA compaction: Read total {} records from {} files",
            original_count,
            input_files.len()
        );

        // Keep exactly the NEWEST copy of each id, under the same order the
        // read path uses (`proximadb_records::VersionKey`; TD-ENGINE-MVCC-1).
        // This used to sort by id and keep the FIRST occurrence, which is
        // file-listing order, not newest: compaction could keep v1 and discard
        // v2 or its tombstone, permanently resurrecting the old copy.
        // Tombstones are kept (a newest-copy tombstone must keep shadowing
        // copies in files outside this compaction set); TTL rows are kept with
        // their expiry and are hidden by the read path.
        let unique_records: Vec<VectorRecord> = keep_newest_per_id(all_records)
            .iter()
            .map(VectorRecord::from)
            .collect();

        let entries_removed = (original_count - unique_records.len()) as u64;
        info!(
            "🔄 NOVA compaction: After deduplication: {} records ({} removed)",
            unique_records.len(),
            entries_removed
        );

        // Write compacted file to same data directory
        let output_path = self.generate_compacted_file_path(&data_path, collection_id);
        debug!(
            "🔄 NOVA compaction: Writing compacted file to {}",
            output_path
        );

        let bytes_after = self
            .write_compacted_file(
                params,
                &output_path,
                unique_records.clone(),
                128 * 1024 * 1024, // Default 128MB target size
            )
            .await?;

        info!(
            "🔄 NOVA compaction: Written {} bytes to {}",
            bytes_after, output_path
        );

        // Clean up input files after successful compaction
        for file_path in &input_files {
            debug!("🔄 NOVA compaction: Deleting input file {}", file_path);
            let fs_for_delete = self.filesystem.get_filesystem(file_path)?;
            if let Err(e) = fs_for_delete.delete(file_path).await {
                warn!("Failed to delete input file {}: {}", file_path, e);
            }
        }

        info!(
            "🔄 NOVA compaction: Complete - {} files → 1 file, {} → {} bytes, {} duplicates removed",
            input_files.len(),
            bytes_before,
            bytes_after,
            entries_removed
        );

        Ok(CompactionResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_processed: Some(original_count as u64),
            entries_removed: Some(entries_removed),
            bytes_read: Some(bytes_before),
            bytes_written: Some(bytes_after),
            input_files: Some(input_files.len() as u64),
            output_files: Some(1),
            duration_ms: Some(start.elapsed().as_millis() as u64),
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::new(),
        })
    }

    /// Read parquet file using UnifiedParquetReader
    async fn read_parquet_file(
        &self,
        file_path: &str,
        collection_id: &str,
    ) -> Result<Vec<ProximaRecord>> {
        // Get filesystem
        let fs = self.filesystem.get_filesystem(file_path)?;

        // Create unified caching filesystem
        let unified_fs = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                fs,
                collection_id.to_string(),
                "nova".to_string(),
            ),
        );

        // Create UnifiedParquetReader - dimension will be read from file
        let reader = UnifiedParquetReader::new(
            vec![file_path.to_string()],
            128, // Default dimension, will be overridden from file metadata
            self.filesystem.clone(),
            unified_fs,
            collection_id.to_string(),
            "nova".to_string(),
        )?;

        // `0` is the START OFFSET, not a batch size. This passed 100000, which
        // skipped the first 100,000 rows of any larger file; compaction then
        // wrote the output without them and deleted the inputs (data loss).
        reader.read_all_records(0, None).await
    }

    /// Write compacted file using HybridParquetWriter
    async fn write_compacted_file(
        &self,
        params: &CompactionParameters,
        output_path: &str,
        records: Vec<VectorRecord>,
        target_size: u64,
    ) -> Result<u64> {
        // Get dimension from collection config FIRST, then fall back to actual vectors
        let dimension = params.collection_config
            .as_ref()
            .and_then(|collection| collection.config.as_ref())
            .map(|config| config.dimension as usize)
            .or_else(|| records.first().map(|r| r.vector.len()))
            .ok_or_else(|| anyhow::anyhow!(
                "Cannot determine vector dimension: no collection config and no records provided"
            ))?;

        let writer_config = ParquetWriterConfig {
            row_group_size: 50000,
            page_size: 1024 * 1024,
            write_batch_size: 10000,
            compression: parquet::basic::Compression::ZSTD(Default::default()),
            compression_level: None,
            enable_dictionary: true,
            enable_bloom_filters: true,
            bloom_filter_fpp: 0.01,
            bloom_filter_ndv: 100000,
            enable_statistics: true,
            enable_page_index: true,
            sort_columns: vec![],
            id_less_storage: false,
            filterable_metadata_columns: None,
            quantization: Default::default(),
            max_records_per_file: None,
            target_file_size_bytes: Some(target_size as usize),
            enable_async_io: true,
        };

        let hybrid_config = HybridWriterConfig {
            base_config: writer_config,
            initial_mode: crate::storage::engines::core::formats::columnar::hybrid_writer::WriterMode::Streaming,
            enable_auto_switch: true,
            mode_switch_threshold: 1000,
            pattern_window_size: 100,
            streaming_threshold: 1000.0,
            batch_threshold: 10000,
            max_buffer_size: 100 * 1024 * 1024,
            buffer_time_limit: std::time::Duration::from_secs(30),
            enable_concurrent_writes: false,
            max_concurrent_writers: 1,
            optimize_row_group_size: true,
            min_row_group_size: 1000,
            max_row_group_size: 100000,
        };

        // Convert to ProximaRecord for write_with_cache
        let proxima_records: Vec<proximadb_records::ProximaRecord> = records
            .iter()
            .map(proximadb_records::ProximaRecord::from)
            .collect();

        // Use HybridParquetWriter::write_with_cache for compaction
        let (stats, _) = HybridParquetWriter::write_with_cache(
            &proxima_records,
            dimension,
            hybrid_config,
            output_path,
            &self.filesystem,
            None, // filterable_columns
            None, // metadata_collector
        )
        .await?;

        Ok(stats.file_size)
    }

    /// Generate compacted file path in the data directory
    fn generate_compacted_file_path(&self, data_path: &str, collection_id: &str) -> String {
        let timestamp = chrono::Utc::now().timestamp_millis();
        let uuid = uuid::Uuid::new_v4();
        format!(
            "{}/nova_{}_compacted_{}_{}.parquet",
            data_path, collection_id, timestamp, uuid
        )
    }
}

/// Keep exactly the newest copy of each id under the shared order
/// (`proximadb_records::VersionKey`), sorted by id for locality. Tombstones are
/// kept: a newest-copy tombstone must keep shadowing copies in files outside the
/// compaction set.
fn keep_newest_per_id(records: Vec<ProximaRecord>) -> Vec<ProximaRecord> {
    let mut out =
        proximadb_records::keep_newest_by(records, |r| r.oid.as_str(), ProximaRecord::version_key);
    out.sort_by(|a, b| a.oid.cmp(&b.oid));
    out
}

#[cfg(test)]
mod keep_newest_tests {
    use super::*;

    fn rec(oid: &str, updated_ms: i64, version: u64, valid_to_ns: Option<i64>) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            created_at_ns: updated_ms * 1_000_000,
            updated_at_ns: updated_ms * 1_000_000,
            record_version: version,
            valid_to_ns,
            ..Default::default()
        }
    }

    fn pick(input: Vec<ProximaRecord>) -> ProximaRecord {
        let out = keep_newest_per_id(input);
        assert_eq!(out.len(), 1, "one copy per id");
        out.into_iter().next().unwrap_or_default()
    }

    #[test]
    fn newest_wins_in_either_input_order() {
        // The old dedup kept the FIRST occurrence: order-dependent. Pin both orders.
        let old = rec("a", 1_000, 1, None);
        let new = rec("a", 2_000, 2, Some(9_000_000_000_000));
        assert_eq!(pick(vec![old.clone(), new.clone()]).record_version, 2);
        assert_eq!(pick(vec![new, old]).record_version, 2);
    }

    #[test]
    fn same_instant_tombstone_beats_live_copy_and_is_kept() {
        let live = rec("a", 5_000, 3, None);
        let tomb = rec("a", 5_000, 0, Some(0));
        assert_eq!(pick(vec![live.clone(), tomb.clone()]).valid_to_ns, Some(0));
        assert_eq!(pick(vec![tomb, live]).valid_to_ns, Some(0));
    }

    #[test]
    fn distinct_ids_all_survive_sorted() {
        let out = keep_newest_per_id(vec![
            rec("c", 1, 1, None),
            rec("a", 1, 1, None),
            rec("b", 1, 1, None),
        ]);
        let ids: Vec<_> = out.iter().map(|r| r.oid.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }
}
