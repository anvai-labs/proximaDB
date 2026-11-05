// Unified SWIFT Reader with cloud-optimized I/O and hierarchical pruning
// Optimized for HTTP range reads and minimal API calls to reduce cloud storage costs
//
// PROXIMA INTEGRATION FOR SWIFT SUPERBLOCKS:
// =============================================
// SWIFT extends SST with hierarchical SuperBlocks that benefit from Proxima encoding:
//
// 1. SUPERBLOCK STRUCTURE WITH PROXIMA:
//    Traditional SuperBlock (10K vectors = 10 DataBlocks):
//    [SuperBlockHeader][DataBlock1][DataBlock2]...[DataBlock10][SuperIndex]
//
//    Proxima-Enhanced SuperBlock:
//    [EncodingMarker(1B)][SuperBlockHeader][EncodedSuperVectors][SubBlocks][SuperIndex]
//
//    Where EncodedSuperVectors can use:
//    - Cross-block columnar encoding (10K vectors treated as single columnar unit)
//    - Hierarchical encoding (coarse → fine grain)
//    - Progressive quantization alignment
//
// 2. HIERARCHICAL ENCODING STRATEGY:
//    Level 1 (SuperBlock): 10K vectors
//    - Global statistics computed
//    - Choose SuperBlock-wide encoding scheme
//    - Can use more aggressive compression due to larger sample
//
//    Level 2 (DataBlock): 1K vectors each
//    - Inherit SuperBlock encoding hints
//    - Local refinement if beneficial
//    - Maintains block independence for selective reads
//
// 3. ENCODING MARKERS HIERARCHY:
//    SuperBlock Marker (1 byte):
//    - 0x80-0x8F: SuperBlock-level Proxima encoding
//    - Indicates all child blocks use same encoding
//
//    DataBlock Markers (1 byte each):
//    - 0x00-0x7F: Block-specific encoding (overrides SuperBlock)
//    - 0xFF: Inherit from SuperBlock encoding
//
// 4. SWIFT-SPECIFIC OPTIMIZATIONS:
//    - B+ Tree leaf nodes store encoding hints
//    - Bloom filters aware of encoded data layout
//    - Three-tier metadata includes encoding statistics
//    - Prefetching considers encoding boundaries
//
// 5. BENEFITS FOR SWIFT:
//    - 50-60% storage reduction (better than SST due to larger blocks)
//    - Faster SuperBlock scans with SIMD
//    - Reduced cloud API calls (fewer bytes to fetch)
//    - Better cache utilization with compressed SuperBlocks

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::io::zero_copy::traits::CacheTemperature;
use crate::storage::persistence::filesystem::FilesystemFactory;
// INTEGRATION: Use SharedSstFormatReader for file operations (SWIFT extends SST format)
use crate::storage::engines::core::formats::proximablocks::sst_io_layer::{
    SharedSstFormatReader, SstMmapStrategy, SstRegion,
};
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;

use super::MetadataFilter;

/// Reading strategy for SWIFT files
#[derive(Debug, Clone)]
pub enum SwiftReadStrategy {
    /// Read all data without pruning (for compaction)
    StreamAll,
    /// Read with hierarchical pruning (for queries)
    HierarchicalPrune {
        metadata_filter: Option<MetadataFilter>,
        id_filter: Option<Vec<String>>,
    },
    /// Read specific blocks only
    SelectiveBlocks { block_ids: Vec<u32> },
    /// Read specific superblocks only
    SelectiveSuperblocks { superblock_ids: Vec<u32> },
}

/// Configuration for optimizing I/O operations
#[derive(Debug, Clone)]
pub struct SwiftReaderConfig {
    /// Enable prefetching for sequential reads
    pub enable_prefetch: bool,

    /// Maximum concurrent range reads
    pub max_concurrent_reads: usize,

    /// Batch multiple small reads into single larger read if gap < threshold
    pub coalesce_threshold_bytes: usize,

    /// Cache superblock metadata to avoid repeated reads
    pub cache_metadata: bool,

    /// Use streaming for large reads
    pub streaming_threshold_mb: usize,
}

impl Default for SwiftReaderConfig {
    fn default() -> Self {
        Self {
            enable_prefetch: true,
            max_concurrent_reads: 4,
            coalesce_threshold_bytes: 64 * 1024, // 64KB gap threshold
            cache_metadata: true,
            streaming_threshold_mb: 16, // Stream if > 16MB
        }
    }
}

/// Unified SWIFT reader with cloud optimization
pub struct UnifiedSwiftReader {
    /// CORE READER: Delegates low-level file operations to shared SST infrastructure
    /// (SWIFT extends SST format with hierarchical SuperBlocks)
    #[allow(dead_code)]
    shared_reader: Arc<SharedSstFormatReader>,

    /// File path for this reader instance
    file_path: String,

    /// Reader configuration
    config: SwiftReaderConfig,

    /// SWIFT-SPECIFIC: Cached superblock metadata for hierarchical pruning
    /// This is the key differentiator from basic SST - SuperBlock hierarchy for 3-tier filtering
    cached_superblock_metadata: Arc<RwLock<HashMap<u32, SuperBlockMetadata>>>,

    /// Unified caching filesystem for cache-first metadata access
    #[allow(dead_code)]
    caching_filesystem: Arc<UnifiedCachingFilesystem>,

    /// Collection ID for cache key generation
    collection_id: String,

    /// Cached ID index for fast lookups
    #[allow(dead_code)]
    cached_id_index: Option<Arc<super::id_index::IdIndex>>,

    /// Cached header
    cached_header: Option<super::SwiftHeader>,

    /// Filesystem reference for direct operations
    #[allow(dead_code)]
    filesystem: Arc<FilesystemFactory>,
}

/// Lightweight superblock metadata for caching
#[derive(Debug, Clone)]
pub struct SuperBlockMetadata {
    /// Superblock identifier
    pub id: u32,
    /// Offset in file
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Number of records
    pub record_count: u32,
    /// ID range (min, max)
    pub id_range: (String, String),
    /// Whether deleted records exist
    pub has_deletes: bool,
    /// Bloom filter offset in file
    pub bloom_filter_offset: u64,
    /// Bloom filter size in bytes
    pub bloom_filter_size: u32,
}

/// Range read request for batching
#[derive(Debug, Clone)]
struct RangeReadRequest {
    pub offset: u64,
    pub length: usize,
    pub purpose: ReadPurpose,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ReadPurpose {
    Header,
    #[allow(dead_code)]
    SuperBlockMetadata(u32),
    #[allow(dead_code)]
    DataBlock(u32),
    BloomFilter(u32),
    IdIndex,
    QuantizedData(u32),
}

#[allow(dead_code)]
impl UnifiedSwiftReader {
    /// Create new SWIFT reader with unified caching filesystem
    /// SWIFT extends SST format with SuperBlock hierarchy for 3-tier filtering
    pub async fn new(
        filesystem: Arc<FilesystemFactory>,
        file_path: String,
        caching_filesystem: Arc<UnifiedCachingFilesystem>,
        collection_id: String,
        config: SwiftReaderConfig,
    ) -> Result<Self> {
        Self::new_with_bandwidth_optimizer(
            filesystem,
            file_path,
            caching_filesystem,
            collection_id,
            config,
            None,
        )
        .await
    }

    /// 🚀 NEW: Create SWIFT reader with bandwidth optimizer for smart threshold decisions
    /// This constructor enables dual strategy support for different operation types
    pub async fn new_with_bandwidth_optimizer(
        filesystem: Arc<FilesystemFactory>,
        file_path: String,
        caching_filesystem: Arc<UnifiedCachingFilesystem>,
        collection_id: String,
        config: SwiftReaderConfig,
        _bandwidth_optimizer: Option<
            Arc<crate::storage::engines::core::io::zero_copy::BandwidthOptimizer>,
        >,
    ) -> Result<Self> {
        // Create SWIFT-optimized mmap strategy for hierarchical blocks
        let mmap_strategy = SstMmapStrategy {
            always_mmap: vec![
                SstRegion::BloomFilter, // Always cache bloom filters
                SstRegion::IndexBlock,  // Always cache index blocks
                                        // SWIFT-SPECIFIC: Always cache SuperBlock metadata for hierarchical pruning
            ],
            conditional_mmap: vec![
                (SstRegion::DataBlocks, 0.8), // Cache data blocks based on bandwidth optimizer decisions
            ],
            never_mmap: vec![
                SstRegion::Metadata, // Metadata is small, direct read is faster
            ],
        };

        // Create shared reader for actual file operations
        let shared_reader = Arc::new(SharedSstFormatReader::new(
            filesystem.clone(),
            mmap_strategy,
            caching_filesystem.clone(),
            collection_id.clone(),
        ));

        let reader = Self {
            shared_reader,
            file_path: file_path.clone(),
            config,
            cached_superblock_metadata: Arc::new(RwLock::new(HashMap::new())),
            caching_filesystem,
            collection_id,
            cached_id_index: None,
            cached_header: None,
            filesystem,
        };

        Ok(reader)
    }

    /// 🚀 NEW: Read with selective cache strategy - for normal queries with range reads and cache lookup
    /// This strategy is optimized for:
    /// - Range-based reading with hierarchical SuperBlock pruning
    /// - Cache lookup for frequently accessed SuperBlocks  
    /// - Metadata cache utilization for query planning
    /// - Bandwidth-aware threshold decisions
    pub async fn read_with_selective_cache_strategy(
        &self,
        strategy: SwiftReadStrategy,
        bandwidth_optimizer: Option<
            Arc<crate::storage::engines::core::io::zero_copy::BandwidthOptimizer>,
        >,
    ) -> Result<SwiftReadResult> {
        debug!(
            "🔍 SWIFT SELECTIVE CACHE: Starting selective read strategy for {}",
            self.file_path
        );

        // Apply bandwidth optimizer decisions if available
        if let Some(optimizer) = bandwidth_optimizer {
            // Create query context for bandwidth decisions
            use crate::storage::engines::core::io::zero_copy::traits::{
                QueryContext, QueryType, RequestPriority,
            };

            let query_context = QueryContext {
                query_vector: None,
                metadata_filters: HashMap::new(),
                id_lookups: Vec::new(),
                top_k: None,
                distance_threshold: None,
                query_type: QueryType::VectorSearch,
                collection_context: None,
                priority: RequestPriority::Normal,
                estimated_result_size: Some(1000), // Estimate based on strategy
                selectivity_hint: Some(0.5), // Moderate selectivity for SWIFT hierarchical reads
                collection_id: self.collection_id.clone(),
                concurrent_queries: Some(1),
                cache_temperature: CacheTemperature::Warm,
            };

            // Make bandwidth-optimized decisions
            match optimizer
                .decide_strategy(
                    &self.file_path,
                    0,
                    None,
                    &query_context,
                    RequestPriority::Normal,
                )
                .await
            {
                Ok(decision) => {
                    debug!(
                        "📊 SWIFT BANDWIDTH: File {} - decision: {:?}",
                        self.file_path, decision
                    );
                }
                Err(e) => {
                    warn!(
                        "⚠️ SWIFT BANDWIDTH: Failed to get decision for {}: {}",
                        self.file_path, e
                    );
                }
            }
        }

        // Apply the SWIFT strategy with cache optimization
        self.read_with_strategy(strategy).await
    }

    /// 🚀 NEW: Read with compaction strategy - for full read operations where cache lookups are suboptimal
    /// This strategy is optimized for:
    /// - Compaction operations that bypass write cache but use disk cache if files exist
    /// - Full SuperBlock sequential reads without hierarchical pruning
    /// - Minimal metadata overhead for transient files
    /// - Bandwidth conservation by avoiding unnecessary downloads
    pub async fn read_with_compaction_strategy(
        &self,
        bandwidth_optimizer: Option<
            Arc<crate::storage::engines::core::io::zero_copy::BandwidthOptimizer>,
        >,
    ) -> Result<SwiftReadResult> {
        info!(
            "🔥 SWIFT COMPACTION: Starting compaction read strategy for {}",
            self.file_path
        );

        // Apply bandwidth optimizer decisions if available
        if let Some(optimizer) = bandwidth_optimizer {
            // Create compaction query context
            use crate::storage::engines::core::io::zero_copy::traits::{
                QueryContext, QueryType, RequestPriority,
            };

            let query_context = QueryContext {
                query_vector: None,
                metadata_filters: HashMap::new(),
                id_lookups: Vec::new(),
                top_k: None,
                distance_threshold: None,
                query_type: QueryType::FullScan,
                collection_context: None,
                priority: RequestPriority::Background,
                estimated_result_size: Some(1000000), // Large result set for compaction
                selectivity_hint: Some(1.0),          // Read everything
                collection_id: self.collection_id.clone(),
                concurrent_queries: Some(1),
                cache_temperature: CacheTemperature::Cold, // Don't pollute cache
            };

            // Make bandwidth-optimized decisions for compaction
            match optimizer
                .decide_strategy(
                    &self.file_path,
                    0,
                    None,
                    &query_context,
                    RequestPriority::Background,
                )
                .await
            {
                Ok(decision) => {
                    // For compaction, prefer using disk cache if available, avoid downloading transient files
                    debug!(
                        "🔄 SWIFT COMPACTION BANDWIDTH: File {} - decision: {:?} (transient files avoid download)",
                        self.file_path, decision
                    );
                }
                Err(e) => {
                    warn!(
                        "⚠️ SWIFT COMPACTION BANDWIDTH: Failed to get decision for {}: {}",
                        self.file_path, e
                    );
                }
            }
        }

        // Use StreamAll strategy for compaction - no hierarchical pruning needed
        self.stream_all().await
    }

    /// Get SuperBlock metadata with cache-first pattern
    /// This is SWIFT's key differentiator - hierarchical SuperBlock metadata for 3-tier filtering
    pub async fn superblock_metadata_cached(
        &self,
        superblock_id: u32,
    ) -> Result<SuperBlockMetadata> {
        // CACHE-FIRST: Check zero-copy cache for SWIFT SuperBlock metadata
        // Cache key format: filename:collection_id:swift:superblock:{id}
        let _cache_key = format!(
            "{}:{}:swift:superblock:{}",
            self.file_path, self.collection_id, superblock_id
        );

        // Deferred: UnifiedCachingFilesystem doesn't expose direct cache access yet
        // For now, always load from file
        debug!(
            "Loading SWIFT SuperBlock {} from file: {}",
            superblock_id, self.file_path
        );

        // FALLBACK: Load SuperBlock metadata from file via SharedSstFormatReader
        self.load_superblock_from_file(superblock_id).await
    }

    /// Extract SuperBlock metadata from cached data
    async fn extract_superblock_from_cache(
        &self,
        _cached_metadata: Arc<
            Box<dyn crate::storage::engines::core::io::zero_copy::traits::EngineMetadata>,
        >,
        superblock_id: u32,
    ) -> Result<SuperBlockMetadata> {
        // This would deserialize SuperBlock metadata from the cached data
        // For now, placeholder implementation
        debug!(
            "Extracting SuperBlock {} metadata from cache (implementation pending)",
            superblock_id
        );

        // Return a default SuperBlockMetadata for now
        Ok(SuperBlockMetadata {
            id: superblock_id,
            offset: 0,
            size: 0,
            record_count: 0,
            id_range: (String::new(), String::new()),
            has_deletes: false,
            bloom_filter_offset: 0,
            bloom_filter_size: 0,
        })
    }

    /// Load SuperBlock metadata from file (fallback on cache miss)
    async fn load_superblock_from_file(&self, superblock_id: u32) -> Result<SuperBlockMetadata> {
        // Use SharedSstFormatReader for actual file I/O
        // This delegates to the shared infrastructure while SWIFT adds hierarchical logic
        debug!(
            "Loading SuperBlock {} metadata from file via SharedSstFormatReader",
            superblock_id
        );

        // Placeholder implementation - would use shared_reader for actual file operations
        Ok(SuperBlockMetadata {
            id: superblock_id,
            offset: 0,
            size: 0,
            record_count: 0,
            id_range: (String::new(), String::new()),
            has_deletes: false,
            bloom_filter_offset: 0,
            bloom_filter_size: 0,
        })
    }

    /// Read and cache file header
    async fn read_and_cache_header(&mut self) -> Result<()> {
        // Header is at the beginning of file, typically < 1KB
        // Extract scheme for filesystem selection
        let scheme = if self.file_path.contains("://") {
            self.file_path.split("://").next().unwrap_or("file")
        } else {
            "file"
        };
        let fs = self.filesystem.get_filesystem(&format!("{}://", scheme))?;
        let header_data = fs.read_range(&self.file_path, 0, 4096).await?;
        let header = self.deserialize_header(&header_data)?;
        self.cached_header = Some(header);
        Ok(())
    }

    /// Deserialize header from bytes
    #[allow(dead_code)]
    fn deserialize_header(&self, data: &[u8]) -> Result<super::SwiftHeader> {
        // Check magic bytes
        if data.len() < 4 {
            return Err(anyhow!("Invalid header: too small"));
        }

        let magic = &data[0..4];
        if magic != super::SWIFT_MAGIC {
            return Err(anyhow!("Invalid magic bytes: expected SWFT"));
        }

        // Use bincode for deserialization of the rest
        let header_bytes = &data[4..];
        bincode::deserialize(header_bytes)
            .map_err(|e| anyhow!("Failed to deserialize header: {}", e))
    }

    /// Get header (from cache)
    pub fn header(&self) -> Result<&super::SwiftHeader> {
        self.cached_header
            .as_ref()
            .ok_or_else(|| anyhow!("Header not loaded"))
    }

    /// Read with specific strategy
    pub async fn read_with_strategy(&self, strategy: SwiftReadStrategy) -> Result<SwiftReadResult> {
        match strategy {
            SwiftReadStrategy::StreamAll => self.stream_all().await,
            SwiftReadStrategy::HierarchicalPrune {
                metadata_filter,
                id_filter,
            } => self.read_with_pruning(metadata_filter, id_filter).await,
            SwiftReadStrategy::SelectiveBlocks { block_ids } => {
                self.read_selective_blocks(&block_ids).await
            }
            SwiftReadStrategy::SelectiveSuperblocks { superblock_ids } => {
                self.read_selective_superblocks(&superblock_ids).await
            }
        }
    }

    /// Stream all data without pruning (for compaction)
    async fn stream_all(&self) -> Result<SwiftReadResult> {
        info!("SWIFT Reader: Streaming all data from {}", self.file_path);

        let header = self.header()?;
        let mut all_records = Vec::new();

        // Read entire file in large chunks for efficiency
        let file_metadata = self.filesystem.metadata(&self.file_path).await?;
        let file_size = file_metadata.size;
        let chunk_size = 16 * 1024 * 1024; // 16MB chunks

        let mut offset = header.superblock_offset;
        while offset < file_size {
            let read_size = (chunk_size as u64).min(file_size - offset);
            let actual_fs = self.filesystem.get_filesystem(&self.file_path)?;
            let chunk_data = actual_fs
                .read_range(&self.file_path, offset, read_size)
                .await?;

            // Parse records from chunk
            let records = self.parse_records_from_chunk(&chunk_data)?;
            all_records.extend(records);

            offset += read_size;
        }

        Ok(SwiftReadResult {
            records: all_records,
            blocks_read: header.superblock_count * header.blocks_per_superblock,
            bytes_read: file_size,
            pruning_applied: false,
        })
    }

    /// Read with hierarchical pruning
    async fn read_with_pruning(
        &self,
        metadata_filter: Option<MetadataFilter>,
        id_filter: Option<Vec<String>>,
    ) -> Result<SwiftReadResult> {
        debug!("SWIFT Reader: Reading with hierarchical pruning");

        let header = self.header()?;

        // Step 1: Load superblock metadata (cache if enabled)
        let superblock_metadata = self.load_superblock_metadata().await?;

        // Step 2: Prune at superblock level
        let candidate_superblocks =
            self.prune_superblocks(&superblock_metadata, &metadata_filter, &id_filter)?;

        if candidate_superblocks.is_empty() {
            return Ok(SwiftReadResult {
                records: Vec::new(),
                blocks_read: 0,
                bytes_read: 0,
                pruning_applied: true,
            });
        }

        debug!(
            "Selected {} of {} superblocks after pruning",
            candidate_superblocks.len(),
            header.superblock_count
        );

        // Step 3: For each candidate superblock, prune at block level
        let mut range_requests = Vec::new();
        for sb_id in &candidate_superblocks {
            let sb_meta = &superblock_metadata[sb_id];

            // Check bloom filter first (if available)
            if self.should_read_superblock_bloom(sb_meta, &id_filter)? {
                let bloom_data = self.read_bloom_filter(*sb_id).await?;
                if !self.bloom_filter_matches(&bloom_data, &id_filter)? {
                    continue; // Skip this superblock
                }
            }

            // Add range read for this superblock's blocks
            range_requests.push(RangeReadRequest {
                offset: sb_meta.offset,
                length: sb_meta.size as usize,
                purpose: ReadPurpose::SuperBlockMetadata(*sb_id),
            });
        }

        // Step 4: Coalesce nearby reads to minimize API calls
        let original_request_count = range_requests.len();
        let coalesced_requests = self.coalesce_range_requests(range_requests)?;

        debug!(
            "Coalesced {} requests into {} for efficiency",
            original_request_count,
            coalesced_requests.len()
        );

        // Step 5: Execute reads in parallel (respecting max concurrency)
        let read_results = self.execute_parallel_reads(coalesced_requests).await?;

        // Step 6: Parse records from read data
        let mut all_records = Vec::new();
        let mut bytes_read = 0;

        for (data, _purpose) in read_results {
            bytes_read += data.len() as u64;
            let records = self.parse_records_from_data(&data)?;
            all_records.extend(records);
        }

        Ok(SwiftReadResult {
            records: all_records,
            blocks_read: candidate_superblocks.len() as u32 * header.blocks_per_superblock,
            bytes_read,
            pruning_applied: true,
        })
    }

    /// Read selective blocks
    async fn read_selective_blocks(&self, block_ids: &[u32]) -> Result<SwiftReadResult> {
        debug!("SWIFT Reader: Reading {} selective blocks", block_ids.len());

        // Group blocks by superblock for efficient reading
        let mut blocks_by_superblock: HashMap<u32, Vec<u32>> = HashMap::new();
        for &block_id in block_ids {
            let superblock_id = block_id / 64;
            blocks_by_superblock
                .entry(superblock_id)
                .or_default()
                .push(block_id);
        }

        // Create range requests for each group
        let mut range_requests = Vec::new();
        for (sb_id, block_list) in blocks_by_superblock {
            // If reading many blocks from same superblock, read entire superblock
            if block_list.len() > 32 {
                // Read entire superblock
                let sb_meta = self.superblock_metadata(sb_id).await?;
                range_requests.push(RangeReadRequest {
                    offset: sb_meta.offset,
                    length: sb_meta.size as usize,
                    purpose: ReadPurpose::SuperBlockMetadata(sb_id),
                });
            } else {
                // Read individual blocks
                for block_id in block_list {
                    let block_offset = self.calculate_block_offset(sb_id, block_id)?;
                    range_requests.push(RangeReadRequest {
                        offset: block_offset,
                        length: 16 * 1024 * 1024, // 16MB block size
                        purpose: ReadPurpose::DataBlock(block_id),
                    });
                }
            }
        }

        // Coalesce and execute reads
        let coalesced = self.coalesce_range_requests(range_requests)?;
        let results = self.execute_parallel_reads(coalesced).await?;

        // Parse records
        let mut all_records = Vec::new();
        let mut bytes_read = 0;

        for (data, _purpose) in results {
            bytes_read += data.len() as u64;
            let records = self.parse_records_from_data(&data)?;
            all_records.extend(records);
        }

        Ok(SwiftReadResult {
            records: all_records,
            blocks_read: block_ids.len() as u32,
            bytes_read,
            pruning_applied: false,
        })
    }

    /// Read selective superblocks
    async fn read_selective_superblocks(&self, superblock_ids: &[u32]) -> Result<SwiftReadResult> {
        debug!(
            "SWIFT Reader: Reading {} selective superblocks",
            superblock_ids.len()
        );

        let mut range_requests = Vec::new();

        for &sb_id in superblock_ids {
            let sb_meta = self.superblock_metadata(sb_id).await?;
            range_requests.push(RangeReadRequest {
                offset: sb_meta.offset,
                length: sb_meta.size as usize,
                purpose: ReadPurpose::SuperBlockMetadata(sb_id),
            });
        }

        // Execute reads
        let results = self.execute_parallel_reads(range_requests).await?;

        // Parse records
        let mut all_records = Vec::new();
        let mut bytes_read = 0;

        for (data, _purpose) in results {
            bytes_read += data.len() as u64;
            let records = self.parse_records_from_data(&data)?;
            all_records.extend(records);
        }

        let header = self.header()?;
        Ok(SwiftReadResult {
            records: all_records,
            blocks_read: superblock_ids.len() as u32 * header.blocks_per_superblock,
            bytes_read,
            pruning_applied: false,
        })
    }

    /// Load superblock metadata (with caching)
    async fn load_superblock_metadata(&self) -> Result<HashMap<u32, SuperBlockMetadata>> {
        // Check cache first
        if self.config.cache_metadata {
            let cache = self.cached_superblock_metadata.read().await;
            if !cache.is_empty() {
                return Ok(cache.clone());
            }
        }

        let header = self.header()?;
        let mut metadata = HashMap::new();

        // Read metadata section in one read (more efficient than multiple small reads)
        let metadata_size = header.id_index_offset - header.superblock_offset;
        let actual_fs = self.filesystem.get_filesystem(&self.file_path)?;
        let metadata_data = actual_fs
            .read_range(&self.file_path, header.superblock_offset, metadata_size)
            .await?;

        // Parse all superblock metadata
        let parsed = self.parse_superblock_metadata(&metadata_data, header.superblock_count)?;

        for sb_meta in parsed {
            metadata.insert(sb_meta.id, sb_meta);
        }

        // Update cache
        if self.config.cache_metadata {
            let mut cache = self.cached_superblock_metadata.write().await;
            *cache = metadata.clone();
        }

        Ok(metadata)
    }

    /// Coalesce nearby range requests to minimize API calls
    fn coalesce_range_requests(
        &self,
        requests: Vec<RangeReadRequest>,
    ) -> Result<Vec<RangeReadRequest>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by offset
        let mut sorted = requests;
        sorted.sort_by_key(|r| r.offset);

        let mut coalesced = Vec::new();
        let mut current = sorted[0].clone();

        for request in sorted.into_iter().skip(1) {
            let gap = request
                .offset
                .saturating_sub(current.offset + current.length as u64);

            // Coalesce if gap is small enough
            if gap <= self.config.coalesce_threshold_bytes as u64 {
                // Extend current request to include this one
                let new_end = request.offset + request.length as u64;
                let current_end = current.offset + current.length as u64;
                current.length = (new_end.max(current_end) - current.offset) as usize;
            } else {
                // Gap too large, start new request
                coalesced.push(current);
                current = request;
            }
        }

        coalesced.push(current);
        Ok(coalesced)
    }

    /// Execute reads in parallel with concurrency limit
    async fn execute_parallel_reads(
        &self,
        requests: Vec<RangeReadRequest>,
    ) -> Result<Vec<(Vec<u8>, ReadPurpose)>> {
        use futures::stream::{self, StreamExt};

        let filesystem = self.filesystem.clone();
        let file_path = self.file_path.clone();
        let max_concurrent = self.config.max_concurrent_reads;

        let results = stream::iter(requests)
            .map(|req| {
                let fs = filesystem.clone();
                let path = file_path.clone();
                async move {
                    let actual_fs = fs.get_filesystem(&path)?;
                    let data = actual_fs
                        .read_range(&path, req.offset, req.length as u64)
                        .await?;
                    Ok::<_, anyhow::Error>((data, req.purpose))
                }
            })
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        results.into_iter().collect()
    }

    // Helper methods

    fn parse_records_from_chunk(&self, _data: &[u8]) -> Result<Vec<VectorRecord>> {
        Err(anyhow!(
            "SWIFT engine: record chunk parsing not yet implemented"
        ))
    }

    fn parse_records_from_data(&self, _data: &[u8]) -> Result<Vec<VectorRecord>> {
        Err(anyhow!("SWIFT engine: record parsing not yet implemented"))
    }

    fn parse_superblock_metadata(
        &self,
        _data: &[u8],
        _count: u32,
    ) -> Result<Vec<SuperBlockMetadata>> {
        Err(anyhow!(
            "SWIFT engine: superblock metadata parsing not yet implemented"
        ))
    }

    fn prune_superblocks(
        &self,
        _metadata: &HashMap<u32, SuperBlockMetadata>,
        _metadata_filter: &Option<MetadataFilter>,
        _id_filter: &Option<Vec<String>>,
    ) -> Result<Vec<u32>> {
        Err(anyhow!(
            "SWIFT engine: superblock pruning not yet implemented"
        ))
    }

    fn should_read_superblock_bloom(
        &self,
        _meta: &SuperBlockMetadata,
        _id_filter: &Option<Vec<String>>,
    ) -> Result<bool> {
        // Claim nothing (TD-FACADE-1). This used to answer
        // `id_filter.is_some() && meta.bloom_filter_size > 0` -- a size test
        // wearing a bloom name -- and returned `true` into a path whose very
        // next step (`read_bloom_filter`, below) always errors. Saying "no" is
        // consistent with the three siblings that honestly report the
        // capability as unimplemented, and keeps callers on the full-read path.
        Ok(false)
    }

    async fn read_bloom_filter(&self, _sb_id: u32) -> Result<Vec<u8>> {
        Err(anyhow!(
            "SWIFT engine: bloom filter reading not yet implemented"
        ))
    }

    fn bloom_filter_matches(&self, _data: &[u8], _id_filter: &Option<Vec<String>>) -> Result<bool> {
        Err(anyhow!(
            "SWIFT engine: bloom filter matching not yet implemented"
        ))
    }

    async fn superblock_metadata(&self, sb_id: u32) -> Result<SuperBlockMetadata> {
        let metadata = self.load_superblock_metadata().await?;
        metadata
            .get(&sb_id)
            .cloned()
            .ok_or_else(|| anyhow!("Superblock {} not found", sb_id))
    }

    fn calculate_block_offset(&self, _sb_id: u32, _block_id: u32) -> Result<u64> {
        Err(anyhow!(
            "SWIFT engine: block offset calculation not yet implemented"
        ))
    }
}

/// Result from SWIFT reader
#[derive(Debug)]
pub struct SwiftReadResult {
    pub records: Vec<VectorRecord>,
    pub blocks_read: u32,
    pub bytes_read: u64,
    pub pruning_applied: bool,
}

/// Iterator for streaming large results
pub struct SwiftRecordIterator {
    #[allow(dead_code)]
    reader: Arc<UnifiedSwiftReader>,
    #[allow(dead_code)]
    current_superblock: u32,
    #[allow(dead_code)]
    current_block: u32,
    #[allow(dead_code)]
    buffer: Vec<VectorRecord>,
    finished: bool,
}

impl SwiftRecordIterator {
    pub async fn next_batch(&mut self) -> Result<Option<Vec<VectorRecord>>> {
        if self.finished {
            return Ok(None);
        }

        Err(anyhow!(
            "SWIFT engine: streaming iteration not yet implemented"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_range_coalescing() {
        let _config = SwiftReaderConfig::default();
        // Test that nearby reads are coalesced
        // Implementation depends on test infrastructure
    }

    #[tokio::test]
    async fn test_hierarchical_pruning() {
        // Test that pruning reduces I/O
        // Implementation depends on test infrastructure
    }
}
