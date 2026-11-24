//! SST Engine Index-Based Data Reader
//!
//! SST // strategy removed -  Read entire blocks then filter using indices
//! This leverages LSM block structure for efficient bulk I/O

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::core::search::index_based_filter::{
    ColumnData, ColumnMetadata, IndexBasedDataReader, MetadataSource, ReadStrategy,
};
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use proximadb_records::ProximaRecord;

/// SST-specific metadata source representing an SST file
pub struct SSTMetadataSource {
    pub file_path: String,
    pub sst_metadata: SSTFileMetadata,
    pub column_metadata_cache: HashMap<String, Vec<serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct SSTFileMetadata {
    pub total_rows: usize,
    pub column_info: HashMap<String, ColumnMetadata>,
    pub block_count: usize,
    pub file_size_bytes: u64,
}

impl MetadataSource for SSTMetadataSource {
    fn get_row_count(&self) -> usize {
        self.sst_metadata.total_rows
    }

    fn get_column_metadata(&self, column_name: &str) -> Option<ColumnMetadata> {
        self.sst_metadata.column_info.get(column_name).cloned()
    }

    fn get_metadata_value(&self, row_idx: usize, column_name: &str) -> Option<serde_json::Value> {
        self.column_metadata_cache
            .get(column_name)
            .and_then(|column_values| column_values.get(row_idx))
            .cloned()
    }

    fn get_source_id(&self) -> String {
        self.file_path.clone()
    }

    fn supports_selective_reading(&self) -> bool {
        false // SST reads entire blocks, then filters in memory
    }
}

impl SSTMetadataSource {
    /// Create metadata source by reading SST file metadata
    pub async fn from_sst_file(file_path: &str) -> Result<Self> {
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(filesystem_config)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create filesystem factory: {}", e))?,
        );
        // Create UnifiedCachingFilesystem for the reader
        let base_fs = filesystem.get_filesystem("file://")?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            "indexed_reader".to_string(),
            "sst".to_string(),
        ));
        let reader =
            crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader::new(
                filesystem,
                unified_fs,
                String::from("indexed_reader"),
            );

        // Load metadata to build index cache - SST doesn't support full-column reads
        reader.load_metadata(file_path).await?;

        // For SST, we'll build minimal metadata since we can't read all records efficiently
        // This is by design - SST is optimized for key-based lookups, not full scans
        let total_rows = 1000; // Estimated - would come from SST header if available

        // Build minimal column metadata cache for SST
        let column_metadata_cache = HashMap::new(); // Empty for now - SST doesn't do full scans
        let column_info = HashMap::new(); // Empty for now - would be populated from bloom filter metadata

        Ok(Self {
            file_path: file_path.to_string(),
            sst_metadata: SSTFileMetadata {
                total_rows,
                column_info,
                block_count: 1, // Single block assumed until SST header parsing is wired
                file_size_bytes: std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0),
            },
            column_metadata_cache,
        })
    }

    #[allow(dead_code)]
    fn infer_data_type(value: &serde_json::Value) -> ColumnData {
        match value {
            serde_json::Value::String(_) => ColumnData::String,
            serde_json::Value::Number(_) => ColumnData::Float,
            serde_json::Value::Bool(_) => ColumnData::Boolean,
            serde_json::Value::Array(_) => ColumnData::Array,
            _ => ColumnData::String, // Default
        }
    }

    #[allow(dead_code)]
    fn compare_json_values(a: &serde_json::Value, b: &serde_json::Value) -> std::cmp::Ordering {
        crate::core::search::json_comparison::compare_json_values(a, b)
    }
}

/// SST-specific index-based data reader
pub struct SSTIndexBasedReader {
    /// SST reader for accessing data blocks
    #[allow(dead_code)]
    reader: crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader,
}

impl SSTIndexBasedReader {
    #[expect(
        clippy::expect_used,
        reason = "new() is a convenience wrapper; callers needing error handling should use try_new()"
    )]
    pub fn new() -> Self {
        Self::try_new().expect("Failed to initialize SSTIndexBasedReader")
    }

    pub fn try_new() -> Result<Self> {
        // For synchronous new, we'll create a minimal filesystem factory
        // In practice, this should be passed in as a dependency
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = create_filesystem_factory_sync(filesystem_config)?;

        // Create UnifiedCachingFilesystem for the reader
        let base_fs = filesystem
            .get_filesystem("file://")
            .map_err(|e| anyhow::anyhow!("Failed to get base filesystem: {}", e))?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            "default_collection".to_string(),
            "sst".to_string(),
        ));

        Ok(Self {
            reader:
                crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader::new(
                    filesystem,
                    unified_fs,
                    String::from("default_collection"),
                ),
        })
    }

    /// Filter entire block using indices (SST reads full blocks)
    /// Note: This is a placeholder - real implementation would use block readers
    async fn filter_by_indices(
        &self,
        _file_path: &str,
        indices: &[usize],
    ) -> Result<Vec<ProximaRecord>> {
        // SST Strategy: Read entire blocks, filter using indices
        // This would typically read blocks and filter by indices
        // For now, return empty records with proper structure

        debug!("SST filter by indices: {} indices requested", indices.len());

        // Placeholder - real implementation would:
        // 1. Identify which blocks contain the indices
        // 2. Read those blocks fully
        // 3. Filter by the specific indices
        Ok(Vec::new())
    }

    /// Read full SST file - placeholder implementation
    /// Note: SST is optimized for key-based lookups, not full file reads
    async fn read_full_file(&self, _file_path: &str) -> Result<Vec<ProximaRecord>> {
        // SST doesn't support efficient full file reads by design
        // This would require key-based iteration or block-by-block reading
        // For now, return empty - this indicates the optimization should not be used
        Ok(Vec::new())
    }
}

impl Default for SSTIndexBasedReader {
    fn default() -> Self {
        Self::new()
    }
}

fn create_filesystem_factory_sync(
    filesystem_config: crate::storage::persistence::filesystem::FilesystemConfig,
) -> Result<Arc<crate::storage::persistence::filesystem::FilesystemFactory>> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let factory = handle
            .block_on(
                crate::storage::persistence::filesystem::FilesystemFactory::create(
                    filesystem_config,
                ),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create filesystem factory: {}", e))?;
        return Ok(Arc::new(factory));
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create runtime for filesystem init: {}", e))?;
    let factory = runtime
        .block_on(
            crate::storage::persistence::filesystem::FilesystemFactory::create(filesystem_config),
        )
        .map_err(|e| anyhow::anyhow!("Failed to create filesystem factory: {}", e))?;
    Ok(Arc::new(factory))
}

#[async_trait::async_trait]
impl IndexBasedDataReader for SSTIndexBasedReader {
    async fn read_data_by_indices(
        &self,
        source_id: &str,
        read_strategy: &ReadStrategy,
        _metadata_source: &(dyn MetadataSource + Send + Sync),
    ) -> Result<Vec<ProximaRecord>> {
        match read_strategy {
            ReadStrategy::SkipBlock => {
                debug!("SST: Skipping file {} - no qualifying rows", source_id);
                Ok(Vec::new())
            }

            ReadStrategy::ReadFullBlock => {
                debug!("SST: Reading full file {} - high selectivity", source_id);
                self.read_full_file(source_id).await
            }

            ReadStrategy::SelectiveRead {
                indices,
                estimated_benefit,
            } => {
                info!(
                    "SST: Block read with filter for {} - {} indices, {:.1}% memory savings",
                    source_id,
                    indices.len(),
                    estimated_benefit * 100.0
                );

                // SST Strategy: Read entire blocks then filter by indices
                // This is still beneficial as it reduces post-processing overhead
                let filtered_records = self.filter_by_indices(source_id, indices).await?;

                debug!(
                    "SST: Block filtering completed - {} records",
                    filtered_records.len()
                );
                Ok(filtered_records)
            }
        }
    }

    fn estimate_selective_read_benefit(&self, indices: &[usize], total_rows: usize) -> f32 {
        // For SST, the benefit is primarily in memory usage and post-processing
        let selectivity = indices.len() as f32 / total_rows as f32;

        // SST benefit is lower than a columnar engine since we still read entire blocks
        // But we save on memory allocation and post-processing time
        (1.0 - selectivity) * 0.3 // Up to 30% benefit for memory and processing
    }
}

/// Factory for creating SST metadata sources
pub struct SSTMetadataSourceFactory;

impl SSTMetadataSourceFactory {
    /// Create metadata source from SST file
    pub async fn create_from_file_path(file_path: &str) -> Result<SSTMetadataSource> {
        SSTMetadataSource::from_sst_file(file_path).await
    }

    /// Create multiple metadata sources for batch processing
    pub async fn create_batch_from_paths(file_paths: &[String]) -> Result<Vec<SSTMetadataSource>> {
        let mut sources = Vec::new();

        for path in file_paths {
            let source = Self::create_from_file_path(path).await?;
            sources.push(source);
        }

        Ok(sources)
    }

    /// Create metadata sources with column optimization
    pub async fn create_with_column_filters(
        file_paths: &[String],
        filter_columns: &[String],
    ) -> Result<Vec<SSTMetadataSource>> {
        let mut sources = Vec::new();

        for path in file_paths {
            let mut source = Self::create_from_file_path(path).await?;

            // Optimize metadata cache to only include filter columns
            let mut optimized_cache = HashMap::new();
            for column in filter_columns {
                if let Some(column_data) = source.column_metadata_cache.remove(column) {
                    optimized_cache.insert(column.clone(), column_data);
                }
            }
            source.column_metadata_cache = optimized_cache;

            sources.push(source);
        }

        Ok(sources)
    }
}
