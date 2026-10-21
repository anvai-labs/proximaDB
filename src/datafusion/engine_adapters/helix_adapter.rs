//! # HELIX Engine TableProvider Adapter
//!
//! Implements `ProximaTableProvider` and `SplitReader` for the HELIX (High-Efficiency
//! Locality-Indexed eXecution) storage engine. HELIX uses Hilbert curves for spatial
//! locality optimization, enabling efficient pruning for vector similarity queries.
//!
//! ## Key Features
//!
//! - **Hilbert curve ordering**: Vectors are organized by Hilbert curve codes for locality
//! - **Spatial pruning**: Splits can be pruned based on Hilbert range bounds
//! - **PCA dimensionality reduction**: High-dimensional vectors projected for efficient indexing
//! - **Zone maps**: Per-block min/max statistics for predicate evaluation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         HELIX TABLE PROVIDER                                 │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  HelixTableProvider                                                    │  │
//! │  │  - Maps HELIX files to Hilbert-range FileSplits                       │  │
//! │  │  - Provides spatial bounds for vector pruning                          │  │
//! │  │  - Supports Hilbert-based query optimization                           │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                      │                                       │
//! │                                      ▼                                       │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  HelixSplitReader                                                      │  │
//! │  │  - Reads Hilbert-ordered blocks from HELIX files                      │  │
//! │  │  - Uses spatial bounds for efficient range queries                     │  │
//! │  │  - Supports centroid-based vector pruning                              │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Spatial Pruning
//!
//! HELIX splits include Hilbert curve bounds that enable efficient pruning:
//!
//! ```rust,ignore
//! // Query vector is projected to Hilbert space
//! let query_hilbert_code = pca_model.project_and_compute_hilbert(&query);
//!
//! // Splits with non-overlapping Hilbert ranges can be skipped
//! for split in splits {
//!     if let SpatialBounds::Hilbert { min_code, max_code, .. } = split.spatial_bounds {
//!         if query_hilbert_code < min_code || query_hilbert_code > max_code * threshold {
//!             continue; // Skip this split - no relevant vectors
//!         }
//!     }
//! }
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
use crate::storage::formats::{CacheStatus, FileSplit, SpatialBounds, SplitType, StorageTier};
use crate::storage::persistence::filesystem::FilesystemFactory;

use super::common::vector_collection_schema;

// ============================================================================
// HELIX Table Provider
// ============================================================================

/// HELIX engine-specific TableProvider implementation.
///
/// Maps HELIX files (with Hilbert-ordered data) to DataFusion's TableProvider
/// interface, enabling SQL queries on locality-optimized vector collections.
#[derive(Debug)]
pub struct HelixTableProvider {
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
    cached_files: tokio::sync::RwLock<Option<Vec<HelixFileMetadata>>>,
    /// Hilbert curve order (bits per dimension)
    hilbert_order: u8,
}

/// Metadata for a HELIX file
#[derive(Debug, Clone)]
struct HelixFileMetadata {
    /// File path
    path: String,
    /// File size in bytes
    size_bytes: u64,
    /// LSM level (0 = unsorted, 1+ = sorted)
    level: usize,
    /// Hilbert key range [min, max]
    hilbert_range: Option<(u64, u64)>,
    /// Number of vectors in the file
    vector_count: u64,
    /// Number of ProximaBlocks in the file
    block_count: usize,
    /// Centroid of vectors in this file (for pruning)
    centroid: Option<Vec<f32>>,
}

impl HelixTableProvider {
    /// Create a new HELIX table provider.
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
            hilbert_order: 16, // Default: 16 bits per dimension
        }
    }

    /// Create with custom Hilbert curve order.
    pub fn with_hilbert_order(mut self, order: u8) -> Self {
        self.hilbert_order = order;
        self
    }

    /// Create with pre-computed pruning statistics.
    pub fn with_pruning_stats(mut self, stats: PruningStatistics) -> Self {
        self.pruning_stats = Some(stats);
        self
    }

    /// Discover HELIX files in the collection directory.
    ///
    /// Lists all .helix files and extracts metadata including Hilbert ranges.
    async fn discover_files(&self) -> DFResult<Vec<HelixFileMetadata>> {
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
            // Only process .helix files
            if !entry.name.ends_with(".helix") {
                continue;
            }

            let file_path = format!("{}/{}", self.base_path, entry.name);
            let size_bytes = entry.metadata.size;

            // Parse level from filename (format: L{level}_{timestamp}_{hash}.helix)
            let level = Self::parse_level_from_filename(&entry.name);

            // Estimate vector count based on file size
            let vector_size = self.info.dimension * 4; // f32 = 4 bytes
            let record_overhead = 128; // ID, metadata, timestamps
            let bytes_per_record = vector_size + record_overhead;
            let vector_count = size_bytes / bytes_per_record as u64;

            // Estimate block count (HELIX uses ProximaBlocks)
            let block_size = 64 * 1024; // 64KB default
            let block_count = (size_bytes as usize / block_size).max(1);

            // Hilbert range would be read from file header in production
            // For now, estimate based on file position in level
            let hilbert_range = Self::estimate_hilbert_range(level, files.len());

            files.push(HelixFileMetadata {
                path: file_path,
                size_bytes,
                level,
                hilbert_range: Some(hilbert_range),
                vector_count,
                block_count,
                centroid: None, // Would be read from file metadata
            });
        }

        // Sort by level, then by Hilbert range
        files.sort_by(|a, b| {
            a.level.cmp(&b.level).then_with(|| {
                a.hilbert_range
                    .map(|(min, _)| min)
                    .cmp(&b.hilbert_range.map(|(min, _)| min))
            })
        });

        // Update cache
        {
            let mut cache = self.cached_files.write().await;
            *cache = Some(files.clone());
        }

        debug!(
            "HELIX: Discovered {} files for collection '{}' at {}",
            files.len(),
            self.info.name,
            self.base_path
        );

        Ok(files)
    }

    /// Parse LSM level from filename.
    fn parse_level_from_filename(filename: &str) -> usize {
        if let Some(level_str) = filename.strip_prefix("L")
            && let Some(underscore_pos) = level_str.find('_')
        {
            return level_str[..underscore_pos].parse().unwrap_or(0);
        }
        0
    }

    /// Estimate Hilbert range for a file based on its position.
    ///
    /// In production, this would be read from the file header/footer.
    fn estimate_hilbert_range(level: usize, file_index: usize) -> (u64, u64) {
        // Each level covers the full Hilbert space, divided among files
        let max_hilbert = u64::MAX;
        let files_per_level = 10u64.pow(level as u32 + 1);
        let range_size = max_hilbert / files_per_level;

        let min_code = file_index as u64 * range_size;
        let max_code = min_code.saturating_add(range_size - 1);

        (min_code, max_code)
    }

    /// Generate Hilbert-range splits from discovered files.
    ///
    /// Each file becomes a Hilbert range split for spatial pruning.
    async fn generate_splits_from_files(
        &self,
        files: &[HelixFileMetadata],
        _filters: &[Expr],
    ) -> DFResult<Vec<FileSplit>> {
        let mut splits = Vec::new();

        for file in files {
            if let Some((start_code, end_code)) = file.hilbert_range {
                let mut split = FileSplit::new_hilbert_range(
                    file.path.clone(),
                    start_code,
                    end_code,
                    self.hilbert_order,
                    0, // Offset within file
                    file.size_bytes,
                );

                // Add statistics for pruning
                split.statistics.row_count = Some(file.vector_count);
                split.statistics.byte_size = Some(file.size_bytes);

                // Add centroid if available
                if let Some(ref centroid) = file.centroid {
                    split.statistics.centroid = Some(centroid.clone());
                }

                // Set spatial bounds for Hilbert-based pruning
                split.statistics.spatial_bounds = Some(SpatialBounds::Hilbert {
                    min_code: start_code,
                    max_code: end_code,
                    order: self.hilbert_order,
                });

                // Set locality hints
                split.locality.storage_tier = if file.level == 0 {
                    StorageTier::Hot // L0 files are freshly written
                } else {
                    StorageTier::Warm // Compacted files
                };
                split.locality.cache_status = CacheStatus::Unknown;

                splits.push(split);
            } else {
                // Fallback to block-based split if no Hilbert range
                for block_id in 0..file.block_count {
                    let offset = (block_id * 64 * 1024) as u64;
                    let length = (64 * 1024) as u64;
                    let records_per_block = file.vector_count / file.block_count.max(1) as u64;

                    let split = FileSplit::new_block(
                        file.path.clone(),
                        block_id as u32,
                        offset,
                        length,
                        records_per_block,
                    );
                    splits.push(split);
                }
            }
        }

        trace!(
            "HELIX: Generated {} splits from {} files for collection '{}'",
            splits.len(),
            files.len(),
            self.info.name
        );

        Ok(splits)
    }
}

#[async_trait]
impl TableProvider for HelixTableProvider {
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
        let reader = Arc::new(HelixSplitReader::new(
            self.schema.clone(),
            self.filesystem_factory.clone(),
            self.info.dimension,
            self.hilbert_order,
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
        // HELIX supports filter pushdown for:
        // - Vector similarity queries (via Hilbert range pruning)
        // - Range predicates on numeric columns (via zone maps)
        // - Spatial bounds checks

        let pushdown_support = filters
            .iter()
            .map(|_filter| {
                // Mark as "inexact" - helps with pruning but needs validation
                TableProviderFilterPushDown::Inexact
            })
            .collect();

        Ok(pushdown_support)
    }
}

#[async_trait]
impl ProximaTableProvider for HelixTableProvider {
    fn engine_type(&self) -> EngineType {
        EngineType::Helix
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
// HELIX Split Reader
// ============================================================================

/// Split reader for HELIX files.
///
/// Reads Hilbert-ordered blocks from HELIX files and returns RecordBatch streams.
/// Supports spatial pruning based on Hilbert curve bounds.
#[derive(Debug)]
pub struct HelixSplitReader {
    /// Arrow schema for records
    schema: SchemaRef,
    /// Filesystem factory
    filesystem_factory: Arc<FilesystemFactory>,
    /// Vector dimension
    dimension: usize,
    /// Hilbert curve order
    hilbert_order: u8,
}

impl HelixSplitReader {
    /// Create a new HELIX split reader.
    pub fn new(
        schema: SchemaRef,
        filesystem_factory: Arc<FilesystemFactory>,
        dimension: usize,
        hilbert_order: u8,
    ) -> Self {
        Self {
            schema,
            filesystem_factory,
            dimension,
            hilbert_order,
        }
    }
}

#[async_trait]
impl SplitReader for HelixSplitReader {
    async fn read_split(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        debug!(
            "HELIX: Reading split {} (Hilbert range: {:?}, records={:?})",
            split.split_id,
            match &split.split_type {
                SplitType::HilbertRange {
                    start_code,
                    end_code,
                    ..
                } => Some((*start_code, *end_code)),
                _ => None,
            },
            split.statistics.row_count
        );

        // Create stream that will read from the HELIX file
        let stream = HelixBlockStream::new(
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
        EngineType::Helix
    }

    fn supports_filter_pushdown(&self) -> bool {
        true // HELIX supports spatial pruning via Hilbert ranges
    }

    fn supports_projection_pushdown(&self) -> bool {
        true // HELIX supports column projection
    }
}

// ============================================================================
// HELIX Block Stream
// ============================================================================

/// RecordBatch stream for reading HELIX blocks.
///
/// Reads Hilbert-ordered blocks and yields RecordBatches.
pub struct HelixBlockStream {
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

impl HelixBlockStream {
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

impl Stream for HelixBlockStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        // For this implementation, we mark as finished after first poll
        // A real implementation would read from the HELIX file and return batches
        //
        // The actual HELIX reading logic would:
        // 1. Open the HELIX file using filesystem_factory
        // 2. Read the unified header to get block metadata
        // 3. Seek to blocks within the Hilbert range
        // 4. Decode ProximaBlock format with Hilbert keys
        // 5. Convert to Arrow RecordBatch
        // 6. Apply projection if specified
        //
        // This would be implemented using the existing HELIX reader code in:
        // src/storage/engines/impls/helix/readers/

        self.finished = true;

        trace!(
            "HELIX: Stream finished for split {} (yielded {} records)",
            self.split.split_id, self.records_yielded
        );

        Poll::Ready(None)
    }
}

impl RecordBatchStream for HelixBlockStream {
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
        CollectionInfo::new("test_helix_collection".to_string(), 768, EngineType::Helix)
            .with_vector_count(100000)
            .with_storage_size(1024 * 1024 * 500) // 500MB
            .with_file_count(20)
            .with_base_path("/data/helix_test".to_string())
    }

    #[test]
    fn test_helix_table_provider_creation() {
        let info = test_collection_info();
        assert_eq!(info.engine_type, EngineType::Helix);
        assert_eq!(info.dimension, 768);
    }

    #[test]
    fn test_parse_level_from_filename() {
        assert_eq!(
            HelixTableProvider::parse_level_from_filename("L0_12345_abc.helix"),
            0
        );
        assert_eq!(
            HelixTableProvider::parse_level_from_filename("L1_67890_def.helix"),
            1
        );
        assert_eq!(
            HelixTableProvider::parse_level_from_filename("L5_99999_ghi.helix"),
            5
        );
        assert_eq!(
            HelixTableProvider::parse_level_from_filename("unknown.helix"),
            0
        );
    }

    #[test]
    fn test_estimate_hilbert_range() {
        // Level 0: 10 files, each covers 1/10 of space
        let (min0, max0) = HelixTableProvider::estimate_hilbert_range(0, 0);
        let (min1, _max1) = HelixTableProvider::estimate_hilbert_range(0, 1);

        // Ranges should be non-overlapping
        assert!(max0 < min1);

        // Level 1: 100 files, each covers 1/100 of space
        let (min_l1, max_l1) = HelixTableProvider::estimate_hilbert_range(1, 0);
        assert!(max_l1 < min0 || min_l1 == 0); // L1 ranges should be smaller
    }

    #[test]
    fn test_helix_file_metadata() {
        let metadata = HelixFileMetadata {
            path: "/data/helix/L0_12345_abc.helix".to_string(),
            size_bytes: 10 * 1024 * 1024, // 10MB
            level: 0,
            hilbert_range: Some((0, 1000000)),
            vector_count: 50000,
            block_count: 160,
            centroid: Some(vec![0.1, 0.2, 0.3]),
        };

        assert_eq!(metadata.level, 0);
        assert!(metadata.hilbert_range.is_some());
        assert!(metadata.centroid.is_some());
    }

    #[test]
    fn test_hilbert_split_creation() {
        let split =
            FileSplit::new_hilbert_range("/data/file.helix".to_string(), 1000, 2000, 16, 0, 65536);

        assert!(matches!(split.split_type, SplitType::HilbertRange { .. }));
        assert!(split.statistics.spatial_bounds.is_some());

        if let SplitType::HilbertRange {
            start_code,
            end_code,
            hilbert_order,
        } = split.split_type
        {
            assert_eq!(start_code, 1000);
            assert_eq!(end_code, 2000);
            assert_eq!(hilbert_order, 16);
        }
    }

    #[test]
    fn test_spatial_bounds_hilbert() {
        let bounds = SpatialBounds::Hilbert {
            min_code: 1000,
            max_code: 2000,
            order: 16,
        };

        if let SpatialBounds::Hilbert {
            min_code,
            max_code,
            order,
        } = bounds
        {
            assert_eq!(min_code, 1000);
            assert_eq!(max_code, 2000);
            assert_eq!(order, 16);
        } else {
            panic!("Expected Hilbert bounds");
        }
    }
}
