// SST Metadata Serializer for Zero-Copy Cache
// Hierarchical structure: Global Info → DataBlock Info
// Enables filtering at global level without reading datablocks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::warn;

use proximadb_bloom::SstableBloomFilter;
use proximadb_kernel::error::ProximaDBError;
use proximadb_storage_common::zero_copy_traits::{
    DataRange, EngineMetadata, MetadataSerializer, QueryContext,
};

use super::utilities::stable_bytes_hash;

/// SST Global metadata (fixed size, bytemuck-compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SstGlobalHeader {
    /// Total file size
    pub file_size: u64,

    /// Number of data blocks
    pub num_blocks: u32,

    /// Global bloom filter offset in metadata
    pub bloom_filter_offset: u32,

    /// Global bloom filter size
    pub bloom_filter_size: u32,

    /// Index block offset in metadata
    pub index_offset: u32,

    /// Index block size
    pub index_size: u32,

    /// Total number of records
    pub total_records: u64,

    /// Minimum timestamp (for TTL filtering)
    pub min_timestamp: u64,

    /// Maximum timestamp (for TTL filtering)
    pub max_timestamp: u64,

    /// File compression ratio (0-100)
    pub compression_ratio: u8,

    /// Reserved for future use
    pub reserved: [u8; 7],
}

/// SST DataBlock metadata (fixed size, array-based)
#[repr(C)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SstBlockHeader {
    /// Block offset in file
    pub offset: u64,

    /// Block size (compressed)
    pub compressed_size: u32,

    /// Block size (uncompressed)
    pub uncompressed_size: u32,

    /// Number of records in block
    pub record_count: u32,

    /// Block-level bloom filter offset (relative to start of variable data)
    pub bloom_offset: u32,

    /// Block-level bloom filter size
    pub bloom_size: u32,

    /// Minimum key hash for range queries
    pub min_key_hash: u64,

    /// Maximum key hash for range queries
    pub max_key_hash: u64,

    /// Block priority (0=cold, 255=hot)
    pub priority: u8,

    /// Reserved
    pub reserved: [u8; 7],
}

/// Complete SST metadata structure
#[derive(Debug, Serialize, Deserialize)]
pub struct SstMetadata {
    /// Fixed-size global header
    pub global: SstGlobalHeader,

    /// Fixed-size block headers (array)
    pub blocks: Vec<SstBlockHeader>,

    /// Variable-size data (bloom filters, index data)
    pub variable_data: Vec<u8>,

    /// Parsed global bloom filter (lazy-loaded)
    #[serde(skip)]
    #[allow(dead_code)]
    global_bloom: parking_lot::RwLock<Option<Arc<SstableBloomFilter>>>,

    /// Parsed block bloom filters (lazy-loaded)
    #[serde(skip)]
    #[allow(dead_code)]
    block_blooms: parking_lot::RwLock<HashMap<u32, Arc<Vec<u8>>>>,
}

/// SST metadata serializer implementation
pub struct SstMetadataSerializer {
    /// Filesystem for reading files
    filesystem: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>,
}

impl SstMetadataSerializer {
    pub fn new(filesystem: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort>) -> Self {
        Self { filesystem }
    }

    /// Parse SST file and extract metadata
    async fn extract_metadata(&self, file_path: &str) -> Result<SstMetadata, ProximaDBError> {
        // Read SST file structure
        let fs = self.filesystem.get_filesystem(file_path)?;

        // Read footer to get file structure
        let metadata = fs.metadata(file_path).await?;
        let file_size = metadata.size;
        let footer_size = 1024; // SST footer is typically small
        let footer_data = fs
            .read_range(
                file_path,
                file_size.saturating_sub(footer_size),
                footer_size,
            )
            .await?;

        // Parse footer to get block locations
        let block_locations = self.parse_footer(&footer_data)?;

        // Read global bloom filter
        const BLOOM_OFFSET: u64 = 0;
        const BLOOM_SIZE: u64 = 4096;
        let global_bloom_data = fs.read_range(file_path, BLOOM_OFFSET, BLOOM_SIZE).await?;

        // Read index block
        const INDEX_OFFSET: u64 = BLOOM_SIZE;
        const INDEX_SIZE: u64 = 60 * 1024; // 60KB
        let index_data = fs.read_range(file_path, INDEX_OFFSET, INDEX_SIZE).await?;

        // Extract statistics from blocks
        let mut blocks = Vec::new();
        let mut variable_data = Vec::new();

        // Add global bloom filter to variable data
        let bloom_offset = variable_data.len();
        variable_data.extend_from_slice(&global_bloom_data);

        // Add index data to variable data
        let index_offset = variable_data.len();
        variable_data.extend_from_slice(&index_data);

        // Process each block
        for (i, &(offset, size)) in block_locations.iter().enumerate() {
            // Read block header to get statistics
            let block_header_data = fs.read_range(file_path, offset, 256).await?; // Read first 256 bytes
            let (record_count, min_hash, max_hash) =
                self.parse_block_statistics(&block_header_data)?;

            // Create block header
            let block_header = SstBlockHeader {
                offset,
                compressed_size: size as u32,
                uncompressed_size: size as u32, // Will be updated if we have compression info
                record_count,
                bloom_offset: 0, // Block-level blooms not implemented yet
                bloom_size: 0,
                min_key_hash: min_hash,
                max_key_hash: max_hash,
                priority: if i < 3 { 255 } else { 128 }, // First few blocks are hot
                reserved: [0; 7],
            };

            blocks.push(block_header);
        }

        // Create global header
        let global = SstGlobalHeader {
            file_size,
            num_blocks: blocks.len() as u32,
            bloom_filter_offset: bloom_offset as u32,
            bloom_filter_size: global_bloom_data.len() as u32,
            index_offset: index_offset as u32,
            index_size: index_data.len() as u32,
            total_records: blocks.iter().map(|b| b.record_count as u64).sum(),
            min_timestamp: 0, // Would extract from file metadata
            max_timestamp: u64::MAX,
            compression_ratio: 50, // Estimate
            reserved: [0; 7],
        };

        Ok(SstMetadata {
            global,
            blocks,
            variable_data,
            global_bloom: parking_lot::RwLock::new(None),
            block_blooms: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    fn parse_footer(&self, _footer_data: &[u8]) -> Result<Vec<(u64, u64)>, ProximaDBError> {
        // Parse SST footer to extract block locations
        // This is a simplified implementation
        let mut locations = Vec::new();

        // For demo, assume 16MB file with 4 blocks of 4MB each
        let _total_size = 16 * 1024 * 1024;
        let block_size = 4 * 1024 * 1024;
        let header_size = 64 * 1024; // Skip bloom + index

        for i in 0..4 {
            let offset = header_size + (i * block_size);
            locations.push((offset, block_size));
        }

        Ok(locations)
    }

    fn parse_block_statistics(
        &self,
        _block_data: &[u8],
    ) -> Result<(u32, u64, u64), ProximaDBError> {
        // Extract block statistics from header
        // This would parse the actual block format

        // For demo, return reasonable values
        Ok((1000, 0x1000_0000_0000_0000, 0x8000_0000_0000_0000))
    }
}

impl MetadataSerializer for SstMetadataSerializer {
    fn engine_id(&self) -> &'static str {
        "SST"
    }

    fn serialize_metadata(
        &self,
        file_path: &str,
        _collection_id: &str,
    ) -> Result<Vec<u8>, ProximaDBError> {
        // This would be async in practice, but trait doesn't support async yet
        // Use tokio::task::block_in_place for now
        let metadata = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.extract_metadata(file_path))
        })?;

        let mut buffer = Vec::new();

        // Serialize global header manually
        buffer.extend_from_slice(&metadata.global.file_size.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.num_blocks.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.bloom_filter_offset.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.bloom_filter_size.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.index_offset.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.index_size.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.total_records.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.min_timestamp.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.max_timestamp.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.compression_ratio.to_le_bytes());
        buffer.extend_from_slice(&metadata.global.reserved);

        // Serialize block count
        buffer.extend_from_slice(&(metadata.blocks.len() as u32).to_le_bytes());

        // Serialize block headers manually
        for block in &metadata.blocks {
            buffer.extend_from_slice(&block.offset.to_le_bytes());
            buffer.extend_from_slice(&block.compressed_size.to_le_bytes());
            buffer.extend_from_slice(&block.uncompressed_size.to_le_bytes());
            buffer.extend_from_slice(&block.record_count.to_le_bytes());
            buffer.extend_from_slice(&block.bloom_offset.to_le_bytes());
            buffer.extend_from_slice(&block.bloom_size.to_le_bytes());
            buffer.extend_from_slice(&block.min_key_hash.to_le_bytes());
            buffer.extend_from_slice(&block.max_key_hash.to_le_bytes());
            buffer.extend_from_slice(&block.priority.to_le_bytes());
            buffer.extend_from_slice(&block.reserved);
        }

        // Serialize variable data size
        buffer.extend_from_slice(&(metadata.variable_data.len()).to_le_bytes());

        // Serialize variable data (compressed with bincode if enabled)
        buffer.extend_from_slice(&metadata.variable_data);

        Ok(buffer)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<Box<dyn EngineMetadata>, ProximaDBError> {
        let mut offset = 0;

        // Deserialize global header manually
        if data.len() < offset + 64 {
            // Size of header
            return Err(ProximaDBError::InvalidInput(
                "Invalid SST metadata size".into(),
            ));
        }

        let global = SstGlobalHeader {
            file_size: u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                ProximaDBError::InvalidInput("Invalid byte slice size for file_size".into())
            })?),
            num_blocks: u32::from_le_bytes(data[offset + 8..offset + 12].try_into().map_err(
                |_| ProximaDBError::InvalidInput("Invalid byte slice size for num_blocks".into()),
            )?),
            bloom_filter_offset: u32::from_le_bytes(
                data[offset + 12..offset + 16].try_into().map_err(|_| {
                    ProximaDBError::InvalidInput(
                        "Invalid byte slice size for bloom_filter_offset".into(),
                    )
                })?,
            ),
            bloom_filter_size: u32::from_le_bytes(
                data[offset + 16..offset + 20].try_into().map_err(|_| {
                    ProximaDBError::InvalidInput(
                        "Invalid byte slice size for bloom_filter_size".into(),
                    )
                })?,
            ),
            index_offset: u32::from_le_bytes(data[offset + 20..offset + 24].try_into().map_err(
                |_| ProximaDBError::InvalidInput("Invalid byte slice size for index_offset".into()),
            )?),
            index_size: u32::from_le_bytes(data[offset + 24..offset + 28].try_into().map_err(
                |_| ProximaDBError::InvalidInput("Invalid byte slice size for index_size".into()),
            )?),
            total_records: u64::from_le_bytes(data[offset + 28..offset + 36].try_into().map_err(
                |_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for total_records".into())
                },
            )?),
            min_timestamp: u64::from_le_bytes(data[offset + 36..offset + 44].try_into().map_err(
                |_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for min_timestamp".into())
                },
            )?),
            max_timestamp: u64::from_le_bytes(data[offset + 44..offset + 52].try_into().map_err(
                |_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for max_timestamp".into())
                },
            )?),
            compression_ratio: data[offset + 52],
            reserved: data[offset + 53..offset + 60].try_into().map_err(|_| {
                ProximaDBError::InvalidInput("Invalid byte slice size for reserved".into())
            })?,
        };
        offset += 60;

        // Deserialize block count
        if data.len() < offset + 4 {
            return Err(ProximaDBError::InvalidInput("Invalid SST metadata".into()));
        }
        let block_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Deserialize block headers manually
        let block_header_size = 56; // Size of SstBlockHeader
        let total_block_size = block_count as usize * block_header_size;

        if data.len() < offset + total_block_size {
            return Err(ProximaDBError::InvalidInput(
                "Invalid SST block headers".into(),
            ));
        }

        let mut blocks = Vec::new();
        for _ in 0..block_count {
            let block = SstBlockHeader {
                offset: u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for offset".into())
                })?),
                compressed_size: u32::from_le_bytes(
                    data[offset + 8..offset + 12].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for compressed_size".into(),
                        )
                    })?,
                ),
                uncompressed_size: u32::from_le_bytes(
                    data[offset + 12..offset + 16].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for uncompressed_size".into(),
                        )
                    })?,
                ),
                record_count: u32::from_le_bytes(
                    data[offset + 16..offset + 20].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for record_count".into(),
                        )
                    })?,
                ),
                bloom_offset: u32::from_le_bytes(
                    data[offset + 20..offset + 24].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for bloom_offset".into(),
                        )
                    })?,
                ),
                bloom_size: u32::from_le_bytes(data[offset + 24..offset + 28].try_into().map_err(
                    |_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for bloom_size".into(),
                        )
                    },
                )?),
                min_key_hash: u64::from_le_bytes(
                    data[offset + 28..offset + 36].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for min_key_hash".into(),
                        )
                    })?,
                ),
                max_key_hash: u64::from_le_bytes(
                    data[offset + 36..offset + 44].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for max_key_hash".into(),
                        )
                    })?,
                ),
                priority: data[offset + 44],
                reserved: data[offset + 45..offset + 52].try_into().map_err(|_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for reserved".into())
                })?,
            };
            blocks.push(block);
            offset += block_header_size;
        }

        // Deserialize variable data size
        if data.len() < offset + 4 {
            return Err(ProximaDBError::InvalidInput(
                "Invalid SST variable data size".into(),
            ));
        }
        let var_data_size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Deserialize variable data
        if data.len() < offset + var_data_size as usize {
            return Err(ProximaDBError::InvalidInput(
                "Invalid SST variable data".into(),
            ));
        }
        let variable_data = data[offset..offset + var_data_size as usize].to_vec();

        let metadata = SstMetadata {
            global,
            blocks,
            variable_data,
            global_bloom: parking_lot::RwLock::new(None),
            block_blooms: parking_lot::RwLock::new(HashMap::new()),
        };

        Ok(Box::new(metadata))
    }

    fn can_skip_file(&self, metadata: &dyn EngineMetadata, query_context: &QueryContext) -> bool {
        let Some(sst_metadata) = metadata.as_any().downcast_ref::<SstMetadata>() else {
            warn!("Invalid metadata type for SST serializer in can_skip_file; cannot skip");
            return false;
        };

        // Check TTL first (fastest filter)
        if let Some(ttl_threshold) = query_context.metadata_filters.get("ttl_threshold")
            && let Ok(threshold) = ttl_threshold.parse::<u64>()
            && sst_metadata.global.max_timestamp < threshold
        {
            return true; // Entire file is expired
        }

        // Check global bloom filter for ID lookups
        if !query_context.id_lookups.is_empty() {
            // Load and check global bloom filter
            let global_bloom_data =
                &sst_metadata.variable_data[sst_metadata.global.bloom_filter_offset as usize
                    ..(sst_metadata.global.bloom_filter_offset
                        + sst_metadata.global.bloom_filter_size) as usize];

            // Simple bloom check (would use proper bloom filter implementation)
            let mut any_might_exist = false;
            for id in &query_context.id_lookups {
                if self.check_bloom_simple(global_bloom_data, id.as_bytes()) {
                    any_might_exist = true;
                    break;
                }
            }

            if !any_might_exist {
                return true; // None of the IDs exist in this file
            }
        }

        // Check vector similarity threshold
        if let Some(threshold) = query_context.distance_threshold {
            let estimated_selectivity = metadata.estimated_selectivity(query_context);
            if estimated_selectivity < 0.001 && threshold > 0.8 {
                return true; // Very low selectivity with high threshold
            }
        }

        false // Can't skip, need to read
    }

    fn get_required_ranges(
        &self,
        metadata: &dyn EngineMetadata,
        query_context: &QueryContext,
    ) -> Option<Vec<DataRange>> {
        let Some(sst_metadata) = metadata.as_any().downcast_ref::<SstMetadata>() else {
            warn!(
                "Invalid metadata type for SST serializer in get_required_ranges; falling back to full read"
            );
            return None;
        };

        // If we can read everything, return None (signals read entire file)
        if query_context.id_lookups.is_empty() && query_context.query_vector.is_some() {
            return None; // Similarity search needs everything
        }

        // For ID lookups, determine which blocks to read
        if !query_context.id_lookups.is_empty() {
            let mut required_ranges = Vec::new();

            // Always include header (bloom + index)
            required_ranges.push(DataRange {
                offset: 0,
                length: 64 * 1024, // Header size
                priority: 255,     // Critical
            });

            // Check each block against ID lookups
            for block in sst_metadata.blocks.iter() {
                let mut need_block = false;

                for id in &query_context.id_lookups {
                    let id_hash = self.hash_string(id);
                    if id_hash >= block.min_key_hash && id_hash <= block.max_key_hash {
                        need_block = true;
                        break;
                    }
                }

                if need_block {
                    required_ranges.push(DataRange {
                        offset: block.offset,
                        length: block.compressed_size as u64,
                        priority: block.priority,
                    });
                }
            }

            // Sort by priority (highest first)
            required_ranges.sort_by_key(|r| std::cmp::Reverse(r.priority));

            return Some(required_ranges);
        }

        None
    }

    // Helper methods
    fn check_bloom_simple(&self, bloom_data: &[u8], key: &[u8]) -> bool {
        // Simplified bloom filter check
        // In practice, would use proper bloom filter implementation
        let hash = self.hash_bytes(key);
        let index = (hash % bloom_data.len() as u64) as usize;
        bloom_data[index] != 0
    }

    fn hash_string(&self, s: &str) -> u64 {
        self.hash_bytes(s.as_bytes())
    }

    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        stable_bytes_hash(bytes)
    }
}

impl EngineMetadata for SstMetadata {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EngineMetadata> {
        Box::new(self.clone())
    }

    fn file_size(&self) -> u64 {
        self.global.file_size
    }

    fn estimated_selectivity(&self, query_context: &QueryContext) -> f32 {
        // Estimate selectivity based on query type
        if !query_context.id_lookups.is_empty() {
            // ID lookups are very selective
            let ids_per_file = self.global.total_records as f32;
            let requested_ids = query_context.id_lookups.len() as f32;
            (requested_ids / ids_per_file).min(1.0)
        } else if query_context.query_vector.is_some() {
            // Vector similarity depends on top_k
            if let Some(top_k) = query_context.top_k {
                let total_records = self.global.total_records as f32;
                (top_k as f32 / total_records).min(1.0)
            } else {
                0.1 // Default 10% selectivity for similarity
            }
        } else {
            1.0 // Scan everything
        }
    }

    fn memory_footprint(&self) -> usize {
        std::mem::size_of::<SstGlobalHeader>()
            + self.blocks.len() * std::mem::size_of::<SstBlockHeader>()
            + self.variable_data.len()
    }
}

// Implement Clone for SstMetadata
impl Clone for SstMetadata {
    fn clone(&self) -> Self {
        Self {
            global: self.global,
            blocks: self.blocks.clone(),
            variable_data: self.variable_data.clone(),
            global_bloom: parking_lot::RwLock::new(None), // Reset lazy-loaded data
            block_blooms: parking_lot::RwLock::new(HashMap::new()),
        }
    }
}
