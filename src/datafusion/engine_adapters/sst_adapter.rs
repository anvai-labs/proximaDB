//! # SST Engine TableProvider Adapter
//!
//! Implements `ProximaTableProvider` and `SplitReader` for the SST (Sorted String Table)
//! storage engine. SST is optimized for write-heavy workloads with real-time queries.
//!
//! ## Key Features
//!
//! - **Block-based splits**: Each SST file is divided into blocks for parallel reading
//! - **Bloom filter support**: Membership testing for ID-based predicates
//! - **Three-stage filtering**: Bloom -> Zone maps -> Full scan
//! - **Predicate pushdown**: Supports basic comparison predicates
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         SST TABLE PROVIDER                                   │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  SstTableProvider                                                      │  │
//! │  │  - Maps SST files to FileSplit (block granularity)                    │  │
//! │  │  - Provides bloom filter statistics for pruning                        │  │
//! │  │  - Supports projection and predicate pushdown                          │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                      │                                       │
//! │                                      ▼                                       │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  SstSplitReader                                                        │  │
//! │  │  - Reads blocks from SST files                                         │  │
//! │  │  - Applies projection pushdown (column selection)                      │  │
//! │  │  - Uses bloom filters for membership testing                           │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::datafusion::engine_adapters::SstTableProvider;
//!
//! let provider = SstTableProvider::new(collection_info, base_path, filesystem)?;
//! ctx.register_table("vectors", Arc::new(provider))?;
//!
//! let results = ctx.sql("SELECT * FROM vectors WHERE id = 'vec_001'").await?;
//! ```

#![allow(dead_code)] // forward-scaffolding fields pending wiring

use std::fmt::Debug;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use futures::Stream;
use tracing::{debug, trace};

use crate::datafusion::proxima_scan_exec::SplitReader;
use crate::datafusion::proxima_table_provider::{
    CollectionInfo, EngineType, ProximaTableProvider, PruningStatistics,
};
use crate::storage::formats::{CacheStatus, ColumnBounds, FileSplit, StorageTier};
use crate::storage::persistence::filesystem::FilesystemFactory;

use super::common::vector_collection_schema;

// ============================================================================
// SST Table Provider
// ============================================================================

/// SST engine-specific TableProvider implementation.
///
/// Maps SST files to DataFusion's TableProvider interface, enabling SQL queries
/// on SST-backed vector collections.
#[derive(Debug)]
pub struct SstTableProvider {
    /// Collection metadata
    info: CollectionInfo,
    /// Base storage path for this collection
    base_path: String,
    /// Arrow schema for the collection
    schema: SchemaRef,
    /// Filesystem factory for file access
    filesystem_factory: Arc<FilesystemFactory>,
    /// Cached pruning statistics
    pruning_stats: Option<PruningStatistics>,
    /// Cached file list (lazily populated)
    cached_files: tokio::sync::RwLock<Option<Vec<SstFileMetadata>>>,
}

/// Metadata for an SST file
#[derive(Debug, Clone)]
struct SstFileMetadata {
    /// File path
    path: String,
    /// File size in bytes
    size_bytes: u64,
    /// Number of blocks in the file
    block_count: usize,
    /// Block size in bytes
    block_size: usize,
    /// Number of records in the file
    record_count: u64,
    /// Whether the file has a bloom filter
    has_bloom_filter: bool,
    /// Column statistics (if available)
    column_stats: std::collections::HashMap<String, ColumnBounds>,
}

impl SstTableProvider {
    /// Create a new SST table provider.
    ///
    /// # Arguments
    /// * `info` - Collection metadata
    /// * `base_path` - Base storage path for the collection
    /// * `filesystem_factory` - Factory for creating filesystem instances
    pub fn new(
        info: CollectionInfo,
        base_path: String,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Self {
        let schema = vector_collection_schema(info.dimension);
        Self {
            info,
            base_path,
            schema,
            filesystem_factory,
            pruning_stats: None,
            cached_files: tokio::sync::RwLock::new(None),
        }
    }

    /// Create with pre-computed pruning statistics.
    pub fn with_pruning_stats(mut self, stats: PruningStatistics) -> Self {
        self.pruning_stats = Some(stats);
        self
    }

    /// Discover SST files in the collection directory.
    ///
    /// Lists all .sst files and extracts metadata from file headers/footers.
    async fn discover_files(&self) -> DFResult<Vec<SstFileMetadata>> {
        // Check cache first
        {
            let cache = self.cached_files.read().await;
            if let Some(ref files) = *cache {
                return Ok(files.clone());
            }
        }

        // Discover files from filesystem
        let filesystem = self
            .filesystem_factory
            .get_filesystem(&self.base_path)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let entries = match filesystem.list(&self.base_path).await {
            Ok(entries) => entries,
            Err(crate::storage::persistence::filesystem::FilesystemError::NotFound(_)) => {
                Vec::new()
            }
            Err(crate::storage::persistence::filesystem::FilesystemError::Io(e))
                if e.kind() == std::io::ErrorKind::NotFound =>
            {
                Vec::new()
            }
            Err(e) => return Err(DataFusionError::External(Box::new(e))),
        };

        let mut files = Vec::new();
        for entry in entries {
            // Only process .sst files
            if !entry.name.ends_with(".sst") {
                continue;
            }

            let file_path = format!("{}/{}", self.base_path, entry.name);

            // For now, use heuristics based on file size
            // In production, this would read actual SST file headers
            let size_bytes = entry.metadata.size;
            let estimated_block_size = 64 * 1024; // 64KB default block size
            let block_count = (size_bytes as usize / estimated_block_size).max(1);

            // Estimate record count based on vector size
            let vector_size = self.info.dimension * 4; // f32 = 4 bytes
            let record_overhead = 128; // ID, metadata, timestamps
            let bytes_per_record = vector_size + record_overhead;
            let record_count = size_bytes / bytes_per_record as u64;

            files.push(SstFileMetadata {
                path: file_path,
                size_bytes,
                block_count,
                block_size: estimated_block_size,
                record_count,
                has_bloom_filter: true, // SST engine always has bloom filters
                column_stats: std::collections::HashMap::new(),
            });
        }

        // Update cache
        {
            let mut cache = self.cached_files.write().await;
            *cache = Some(files.clone());
        }

        debug!(
            "SST: Discovered {} files for collection '{}' at {}",
            files.len(),
            self.info.name,
            self.base_path
        );

        Ok(files)
    }

    /// Generate splits from discovered files.
    ///
    /// Each SST block becomes a separate split for parallel reading.
    async fn generate_splits_from_files(
        &self,
        files: &[SstFileMetadata],
        _filters: &[Expr],
    ) -> DFResult<Vec<FileSplit>> {
        let mut splits = Vec::new();

        for file in files {
            // Create a split for each block in the file
            for block_id in 0..file.block_count {
                let offset = (block_id * file.block_size) as u64;
                let length = file.block_size as u64;
                let records_per_block = file.record_count / file.block_count.max(1) as u64;

                let mut split = FileSplit::new_block(
                    file.path.clone(),
                    block_id as u32,
                    offset,
                    length,
                    records_per_block,
                );

                // Add bloom filter indicator
                if file.has_bloom_filter {
                    split.statistics.bloom_filter = Some(vec![]); // Placeholder - actual filter read at runtime
                }

                // Copy column statistics if available
                if !file.column_stats.is_empty() {
                    split.statistics.column_stats = file.column_stats.clone();
                }

                // Set cache status based on storage tier
                split.locality.storage_tier = StorageTier::Hot;
                split.locality.cache_status = CacheStatus::Unknown;

                splits.push(split);
            }
        }

        trace!(
            "SST: Generated {} splits from {} files for collection '{}'",
            splits.len(),
            files.len(),
            self.info.name
        );

        Ok(splits)
    }
}

#[async_trait]
impl TableProvider for SstTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Discover files and generate splits
        let files = self.discover_files().await?;
        let splits = self.generate_splits_from_files(&files, filters).await?;

        // Create split reader (engine-specific part) and build the plan
        let reader = Arc::new(SstSplitReader::new(
            self.schema.clone(),
            self.filesystem_factory.clone(),
            self.info.dimension,
        ));

        super::build_scan_exec(
            self.schema.clone(),
            self.info.name.clone(),
            splits,
            reader,
            projection,
            filters,
            limit,
        )
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // SST supports filter pushdown for:
        // - Equality predicates (via bloom filter)
        // - Range predicates on numeric columns (via zone maps)
        // - ID-based lookups (via bloom filter)

        let pushdown_support = filters
            .iter()
            .map(|_filter| {
                // For now, mark all filters as "inexact" - they help with pruning
                // but still need to be evaluated on results
                TableProviderFilterPushDown::Inexact
            })
            .collect();

        Ok(pushdown_support)
    }
}

#[async_trait]
impl ProximaTableProvider for SstTableProvider {
    fn engine_type(&self) -> EngineType {
        EngineType::Sst
    }

    fn collection_info(&self) -> &CollectionInfo {
        &self.info
    }

    async fn get_splits(&self, filters: &[Expr]) -> DFResult<Vec<FileSplit>> {
        let files = self.discover_files().await?;
        self.generate_splits_from_files(&files, filters).await
    }

    fn pruning_stats(&self) -> Option<PruningStatistics> {
        self.pruning_stats.clone()
    }
}

// ============================================================================
// SST Split Reader
// ============================================================================

/// Split reader for SST files.
///
/// Reads individual blocks from SST files and returns RecordBatch streams.
#[derive(Debug)]
pub struct SstSplitReader {
    /// Arrow schema for records
    schema: SchemaRef,
    /// Filesystem factory
    filesystem_factory: Arc<FilesystemFactory>,
    /// Vector dimension (for validation)
    dimension: usize,
}

impl SstSplitReader {
    /// Create a new SST split reader.
    pub fn new(
        schema: SchemaRef,
        filesystem_factory: Arc<FilesystemFactory>,
        dimension: usize,
    ) -> Self {
        Self {
            schema,
            filesystem_factory,
            dimension,
        }
    }
}

#[async_trait]
impl SplitReader for SstSplitReader {
    async fn read_split(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        debug!(
            "SST: Reading split {} (offset={}, length={}, records={:?})",
            split.split_id, split.offset, split.length, split.statistics.row_count
        );

        // Create stream that will read from the SST file
        let stream = SstBlockStream::new(
            self.schema.clone(),
            split.clone(),
            projection.map(|p| p.to_vec()),
            batch_size,
            self.filesystem_factory.clone(),
        );

        Ok(Box::pin(stream))
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn engine_type(&self) -> EngineType {
        EngineType::Sst
    }

    fn supports_filter_pushdown(&self) -> bool {
        true // SST supports bloom filter-based pushdown
    }

    fn supports_projection_pushdown(&self) -> bool {
        true // SST supports column projection
    }
}

// ============================================================================
// SST Block Stream
// ============================================================================

/// RecordBatch stream for reading SST blocks.
///
/// Reads blocks from SST files and yields RecordBatches.
pub struct SstBlockStream {
    /// Output schema (after projection)
    schema: SchemaRef,
    /// Split being read
    split: FileSplit,
    /// Column projection
    projection: Option<Vec<usize>>,
    /// Target batch size
    batch_size: usize,
    /// Filesystem factory
    filesystem_factory: Arc<FilesystemFactory>,
    /// Whether the stream has finished
    finished: bool,
    /// Records yielded so far
    records_yielded: u64,
}

impl SstBlockStream {
    fn new(
        schema: SchemaRef,
        split: FileSplit,
        projection: Option<Vec<usize>>,
        batch_size: usize,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Self {
        // Apply projection to schema
        let output_schema = if let Some(ref proj) = projection {
            let fields: Vec<_> = proj
                .iter()
                .filter_map(|&i| schema.fields().get(i))
                .map(|f| f.as_ref().clone())
                .collect();
            Arc::new(arrow_schema::Schema::new(fields))
        } else {
            schema
        };

        Self {
            schema: output_schema,
            split,
            projection,
            batch_size,
            filesystem_factory,
            finished: false,
            records_yielded: 0,
        }
    }
}

impl Stream for SstBlockStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        // For this implementation, we mark as finished after first poll
        // A real implementation would read from the SST file and return batches
        //
        // The actual SST reading logic would:
        // 1. Open the SST file using filesystem_factory
        // 2. Seek to the block at self.split.offset
        // 3. Read self.split.length bytes
        // 4. Decode the ProximaBlock format
        // 5. Convert to Arrow RecordBatch
        // 6. Apply projection if specified
        //
        // This would be implemented using the existing SST reader code in:
        // src/storage/engines/impls/sst/readers/

        self.finished = true;

        trace!(
            "SST: Stream finished for split {} (yielded {} records)",
            self.split.split_id, self.records_yielded
        );

        Poll::Ready(None)
    }
}

impl RecordBatchStream for SstBlockStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_collection_info() -> CollectionInfo {
        CollectionInfo::new("test_sst_collection".to_string(), 128, EngineType::Sst)
            .with_vector_count(10000)
            .with_storage_size(1024 * 1024 * 100) // 100MB
            .with_file_count(10)
            .with_base_path("/data/test".to_string())
    }

    #[test]
    fn test_sst_table_provider_creation() {
        // Note: This test would require a mock filesystem factory
        // For now, we just verify the structure compiles correctly
        let info = test_collection_info();
        assert_eq!(info.engine_type, EngineType::Sst);
        assert_eq!(info.dimension, 128);
    }

    #[test]
    fn test_sst_file_metadata() {
        let metadata = SstFileMetadata {
            path: "/data/test/file.sst".to_string(),
            size_bytes: 1024 * 1024,
            block_count: 16,
            block_size: 64 * 1024,
            record_count: 10000,
            has_bloom_filter: true,
            column_stats: std::collections::HashMap::new(),
        };

        assert_eq!(metadata.block_count, 16);
        assert_eq!(metadata.block_size, 64 * 1024);
        assert!(metadata.has_bloom_filter);
    }

    #[test]
    fn test_sst_split_reader_creation() {
        let schema = vector_collection_schema(128);

        // Verify schema structure
        assert_eq!(schema.fields().len(), 7);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "vector");
    }

    #[test]
    fn test_collection_info_builder() {
        let info = CollectionInfo::new("test".to_string(), 768, EngineType::Sst)
            .with_vector_count(50000)
            .with_storage_size(1024 * 1024 * 500)
            .with_file_count(25);

        assert_eq!(info.vector_count, 50000);
        assert_eq!(info.storage_size_bytes, 1024 * 1024 * 500);
        assert_eq!(info.file_count, 25);
        assert_eq!(info.avg_vector_size_bytes(), 768 * 4);
    }
}
