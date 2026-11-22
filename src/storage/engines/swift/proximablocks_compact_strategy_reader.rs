//! Unified SWIFT Reader with strategy-aware caching
//!
//! This reader implements the unified ReadAccessStrategy pattern for SWIFT engine,
//! choosing between cached and direct reads based on the access pattern.

use anyhow::Result;
use std::sync::Arc;

use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::read_strategy::{ReadAccessStrategy, StrategyAwareReader};
use crate::storage::persistence::filesystem::FileSystem;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;

use super::MetadataFilter;
use super::proximablocks_compact_reader::SwiftReaderConfig;

/// Unified SWIFT reader that implements strategy-aware reading
///
/// This reader automatically chooses between:
/// - DirectSWIFTReader: For full scans (compaction, batch operations)
/// - CachedSWIFTReader: For selective queries (point lookups, searches)
pub struct UnifiedSWIFTReader {
    /// Filesystem factory for direct reads
    filesystem_factory: Arc<FilesystemFactory>,

    /// Cached filesystem for selective reads
    cached_filesystem: Option<Arc<UnifiedCachingFilesystem>>,

    /// Current read strategy
    strategy: ReadAccessStrategy,

    /// Collection ID
    collection_id: String,

    /// SWIFT-specific configuration
    config: SwiftReaderConfig,
}

impl UnifiedSWIFTReader {
    /// Create a new unified SWIFT reader
    pub fn new(
        filesystem_factory: Arc<FilesystemFactory>,
        collection_id: String,
        strategy: ReadAccessStrategy,
        config: SwiftReaderConfig,
    ) -> Result<Self> {
        // Create cached filesystem if needed by strategy
        let cached_filesystem = if strategy.should_use_cache() {
            let base_fs = filesystem_factory.get_filesystem("file://")?;
            Some(Arc::new(UnifiedCachingFilesystem::new(
                base_fs,
                collection_id.clone(),
                "swift".to_string(),
            )))
        } else {
            None
        };

        Ok(Self {
            filesystem_factory,
            cached_filesystem,
            strategy,
            collection_id,
            config,
        })
    }

    /// Create a reader optimized for compaction (direct reads)
    pub fn for_compaction(
        filesystem_factory: Arc<FilesystemFactory>,
        collection_id: String,
    ) -> Result<Self> {
        let config = SwiftReaderConfig {
            enable_prefetch: true, // Sequential reads benefit from prefetch
            max_concurrent_reads: 4,
            coalesce_threshold_bytes: 1024 * 1024, // 1MB
            cache_metadata: false,                 // No caching for compaction
            streaming_threshold_mb: 10,
        };

        Self::new(
            filesystem_factory,
            collection_id,
            ReadAccessStrategy::DirectStream,
            config,
        )
    }

    /// Create a reader optimized for search (cached reads)
    pub fn for_search(
        filesystem_factory: Arc<FilesystemFactory>,
        collection_id: String,
    ) -> Result<Self> {
        let config = SwiftReaderConfig {
            enable_prefetch: false, // Random access doesn't benefit from prefetch
            max_concurrent_reads: 8,
            coalesce_threshold_bytes: 64 * 1024, // 64KB
            cache_metadata: true,                // Cache metadata for searches
            streaming_threshold_mb: 5,
        };

        Self::new(
            filesystem_factory,
            collection_id,
            ReadAccessStrategy::CachedSearch {
                prefetch_metadata: true,
            },
            config,
        )
    }

    /// Create a reader for filtered queries (cached reads)
    pub fn for_filtered_query(
        filesystem_factory: Arc<FilesystemFactory>,
        collection_id: String,
        filter: Option<proximadb_filter_expression::FilterExpression>,
    ) -> Result<Self> {
        let config = SwiftReaderConfig {
            enable_prefetch: false,
            max_concurrent_reads: 8,
            coalesce_threshold_bytes: 64 * 1024,
            cache_metadata: true,
            streaming_threshold_mb: 5,
        };

        Self::new(
            filesystem_factory,
            collection_id,
            ReadAccessStrategy::CachedSelective { filter },
            config,
        )
    }

    /// Convert to legacy SwiftReadStrategy for compatibility
    #[allow(dead_code)]
    fn to_swift_strategy(&self) -> super::proximablocks_compact_reader::SwiftReadStrategy {
        match &self.strategy {
            ReadAccessStrategy::DirectStream => {
                super::proximablocks_compact_reader::SwiftReadStrategy::StreamAll
            }
            ReadAccessStrategy::CachedSelective { filter: _ } => {
                super::proximablocks_compact_reader::SwiftReadStrategy::HierarchicalPrune {
                    metadata_filter: None, // Deferred: Convert FilterExpression to MetadataFilter
                    id_filter: None,
                }
            }
            ReadAccessStrategy::CachedSearch { .. } => {
                super::proximablocks_compact_reader::SwiftReadStrategy::HierarchicalPrune {
                    metadata_filter: None,
                    id_filter: None,
                }
            }
            _ => super::proximablocks_compact_reader::SwiftReadStrategy::StreamAll,
        }
    }

    /// Read vectors using the current strategy with optional collection config
    pub async fn read_with_strategy(
        &self,
        file_path: &str,
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Vec<VectorRecord>> {
        match &self.strategy {
            ReadAccessStrategy::DirectStream => {
                self.read_direct_stream(file_path, collection).await
            }
            _ => self.read_with_cache(file_path, collection).await,
        }
    }

    /// Direct streaming read (bypasses cache) - for compaction and full scans
    /// Similar to SST's CompactionFullRead strategy
    async fn read_direct_stream(
        &self,
        file_path: &str,
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Vec<VectorRecord>> {
        // Use filesystem factory directly for streaming reads to avoid cache pollution
        let fs = self.filesystem_factory.get_filesystem("file://")?;
        let data = fs.read(file_path).await?;

        // Deserialize the SWIFT file with collection config
        let swift_file = super::SwiftFile::deserialize(&data, collection)?;

        // Stream all records without filtering (full scan for compaction)
        let mut records = Vec::new();
        for superblock in &swift_file.superblocks {
            for block in &superblock.blocks {
                records.extend(block.records.iter().map(VectorRecord::from));
            }
        }

        Ok(records)
    }

    /// Cached read with predicate pushdown - for queries
    /// Similar to SST's SelectiveWithCache strategy
    async fn read_with_cache(
        &self,
        file_path: &str,
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Vec<VectorRecord>> {
        let cached_fs = self
            .cached_filesystem
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cached filesystem not initialized"))?;

        // Use UnifiedCachingFilesystem for optimal I/O with caching (implements FileSystem trait)
        use crate::storage::persistence::filesystem::FileSystem;
        let data = cached_fs
            .read(file_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;

        // Deserialize the SWIFT file with collection config
        let swift_file = super::SwiftFile::deserialize(&data, collection)?;

        // Apply predicate pushdown based on strategy
        let mut records = Vec::new();

        match &self.strategy {
            ReadAccessStrategy::CachedSelective { filter } => {
                // Predicate pushdown: only read blocks that match filter
                for superblock in &swift_file.superblocks {
                    // ✅ Check Proxima auto-generated bloom filters for quick negative lookups
                    // Aggregate bloom filters from all blocks in superblock
                    let mut should_skip = false;
                    for block in &superblock.blocks {
                        if let Some(ref bloom) = block.bloom_filter {
                            // Early skip if bloom filter says no match possible
                            if !self.check_bloom_filter(bloom, filter) {
                                should_skip = true;
                                break;
                            }
                        }
                    }
                    if should_skip {
                        continue;
                    }

                    // Process blocks that might contain matching records
                    for block in &superblock.blocks {
                        // Apply filter at block level for efficiency
                        let legacy_records: Vec<VectorRecord> =
                            block.records.iter().map(VectorRecord::from).collect();
                        let filtered = self.apply_filter_to_block(&legacy_records, filter)?;
                        records.extend(filtered);
                    }
                }
            }
            _ => {
                // No filter - but still use hierarchical pruning for search
                for superblock in &swift_file.superblocks {
                    for block in &superblock.blocks {
                        records.extend(block.records.iter().map(VectorRecord::from));
                    }
                }
            }
        }

        Ok(records)
    }

    /// Check bloom filter for potential matches
    fn check_bloom_filter(
        &self,
        bloom: &crate::core::bloom::SstableBloomFilter,
        filter: &Option<proximadb_filter_expression::FilterExpression>,
    ) -> bool {
        // Extract field name from filter for bloom filter check
        if let Some(proximadb_filter_expression::FilterExpression::Comparison {
            field,
            value,
            ..
        }) = filter
        {
            // Build a MetadataItem from the filter value for bloom lookup
            let item = crate::proto::proximadb_v1::MetadataItem {
                key: field.clone(),
                value: match value {
                    serde_json::Value::String(s) => Some(
                        crate::proto::proximadb_v1::metadata_item::Value::StringValue(s.clone()),
                    ),
                    serde_json::Value::Number(n) => n
                        .as_f64()
                        .map(crate::proto::proximadb_v1::metadata_item::Value::NumberValue),
                    _ => None,
                },
            };
            // might_match_metadata returns Result<bool>; on error, conservatively return true
            bloom.might_match_metadata(field, &item).unwrap_or(true)
        } else {
            true // No simple comparison filter, must check block
        }
    }

    /// Apply filter to block records (predicate pushdown)
    fn apply_filter_to_block(
        &self,
        records: &[VectorRecord],
        filter: &Option<proximadb_filter_expression::FilterExpression>,
    ) -> Result<Vec<VectorRecord>> {
        let filter_expr = match filter {
            Some(expr) => expr,
            None => return Ok(records.to_vec()),
        };

        let filtered = records
            .iter()
            .filter(|record| {
                // VectorRecord.metadata is HashMap<String, SqlValue> — use directly
                crate::core::search::sql_value_filter::evaluate_filter(
                    filter_expr,
                    &record.metadata,
                )
            })
            .cloned()
            .collect();

        Ok(filtered)
    }
}

impl StrategyAwareReader for UnifiedSWIFTReader {
    fn strategy(&self) -> &ReadAccessStrategy {
        &self.strategy
    }

    fn set_strategy(&mut self, strategy: ReadAccessStrategy) {
        // Update strategy and potentially recreate cached filesystem if needed
        let strategy_changed = self.strategy.should_use_cache() != strategy.should_use_cache();
        self.strategy = strategy;

        if strategy_changed {
            // Update config based on new strategy
            match &self.strategy {
                ReadAccessStrategy::DirectStream => {
                    self.config.cache_metadata = false;
                    self.config.streaming_threshold_mb = 0; // Always stream
                    self.config.enable_prefetch = true;
                }
                _ => {
                    self.config.cache_metadata = true;
                    self.config.streaming_threshold_mb = 10; // Stream for files > 10MB
                    self.config.enable_prefetch = false;
                }
            }

            // Recreate cached filesystem if strategy changed to cached
            if self.strategy.should_use_cache()
                && self.cached_filesystem.is_none()
                && let Ok(base_fs) = self.filesystem_factory.get_filesystem("file://")
            {
                self.cached_filesystem = Some(Arc::new(UnifiedCachingFilesystem::new(
                    base_fs,
                    self.collection_id.clone(),
                    "swift".to_string(),
                )));
            }
        }
    }
}

/// Direct SWIFT Reader (bypasses cache completely)
///
/// Use this for:
/// - Compaction operations
/// - Full superblock scans
/// - Batch migrations
/// - Sequential reads of entire files
pub struct DirectSWIFTReader {
    filesystem_factory: Arc<FilesystemFactory>,
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    config: SwiftReaderConfig,
}

impl DirectSWIFTReader {
    pub fn new(filesystem_factory: Arc<FilesystemFactory>, collection_id: String) -> Self {
        let config = SwiftReaderConfig {
            enable_prefetch: true,
            max_concurrent_reads: 4,
            coalesce_threshold_bytes: 1024 * 1024,
            cache_metadata: false,
            streaming_threshold_mb: 10,
        };

        Self {
            filesystem_factory,
            collection_id,
            config,
        }
    }

    /// Stream superblocks directly without caching
    pub async fn stream_superblocks(&self, file_path: &str) -> Result<Vec<VectorRecord>> {
        let fs = self.filesystem_factory.get_filesystem("file://")?;
        let _ = fs.read(file_path).await?;

        // Deferred: Implement SWIFT superblock streaming
        Ok(vec![])
    }
}

/// Cached SWIFT Reader (uses UnifiedCachingFilesystem)
///
/// Use this for:
/// - Point queries by ID
/// - Hierarchical pruning queries
/// - Search operations with metadata filters
/// - Any operation that benefits from caching superblock metadata
pub struct CachedSWIFTReader {
    cached_filesystem: Arc<UnifiedCachingFilesystem>,
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    config: SwiftReaderConfig,
}

impl CachedSWIFTReader {
    pub fn new(filesystem_factory: Arc<FilesystemFactory>, collection_id: String) -> Result<Self> {
        let base_fs = filesystem_factory.get_filesystem("file://")?;
        let cached_filesystem = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            collection_id.clone(),
            "swift".to_string(),
        ));

        let config = SwiftReaderConfig {
            enable_prefetch: false,
            max_concurrent_reads: 8,
            coalesce_threshold_bytes: 64 * 1024,
            cache_metadata: true,
            streaming_threshold_mb: 5,
        };

        Ok(Self {
            cached_filesystem,
            collection_id,
            config,
        })
    }

    /// Read with hierarchical pruning and caching
    pub async fn read_with_pruning(
        &self,
        file_path: &str,
        _metadata_filter: Option<MetadataFilter>,
    ) -> Result<Vec<VectorRecord>> {
        let _ = self.cached_filesystem.read(file_path).await?;

        // Deferred: Implement hierarchical pruning with metadata filter
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::FilesystemConfig;

    #[tokio::test]
    async fn test_swift_strategy_selection() {
        let factory = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .unwrap(),
        );

        // Compaction should use DirectStream
        let compaction_reader =
            UnifiedSWIFTReader::for_compaction(factory.clone(), "test_collection".to_string())
                .unwrap();
        assert_eq!(
            compaction_reader.strategy(),
            &ReadAccessStrategy::DirectStream
        );
        assert!(!compaction_reader.is_using_cache());

        // Search should use CachedSearch
        let search_reader =
            UnifiedSWIFTReader::for_search(factory.clone(), "test_collection".to_string()).unwrap();
        assert!(matches!(
            search_reader.strategy(),
            ReadAccessStrategy::CachedSearch { .. }
        ));
        assert!(search_reader.is_using_cache());
    }

    #[tokio::test]
    async fn test_config_updates_with_strategy() {
        let factory = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .unwrap(),
        );
        let mut reader = UnifiedSWIFTReader::for_search(factory, "test".to_string()).unwrap();

        // Initially configured for search (cached)
        assert!(reader.config.cache_metadata);
        assert_eq!(reader.config.streaming_threshold_mb, 5);

        // Change to direct stream
        reader.set_strategy(ReadAccessStrategy::DirectStream);
        assert!(!reader.config.cache_metadata);
        assert_eq!(reader.config.streaming_threshold_mb, 0); // Always stream
    }
}
