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

        // NO global-bloom read (TD-BLOOM-1). This used to be
        // `fs.read_range(file_path, 0, 4096)` labelled "global bloom filter",
        // but offset 0 is the file's magic + header -- the real filter lives at
        // `8 + header.header_size` and no writer emits one here. Reading it
        // bought nothing except a 4 KiB range read per metadata load and a
        // plausible-looking buffer that made the `can_skip_file` probe's
        // "definitely absent" answers look legitimate. The header's
        // bloom_filter_{offset,size} are reported as 0 below = honestly "no
        // filter present"; 0-size is already representable in the serialized
        // metadata format, so this changes values, not layout.

        // Read index block. The 4 KiB start offset is the pre-existing file
        // layout assumption and is deliberately unchanged here.
        const INDEX_OFFSET: u64 = 4096;
        const INDEX_SIZE: u64 = 60 * 1024; // 60KB
        let index_data = fs.read_range(file_path, INDEX_OFFSET, INDEX_SIZE).await?;

        // Extract statistics from blocks
        let mut blocks = Vec::new();
        let mut variable_data = Vec::new();

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
            // 0/0 = no global filter present (TD-BLOOM-1); see the read above.
            bloom_filter_offset: 0,
            bloom_filter_size: 0,
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

    /// Encode an [`SstMetadata`] into the wire envelope read back by
    /// [`MetadataSerializer::deserialize_metadata`]:
    /// `[global: 60B][u32 block_count][blocks: 52B each][u32 var_len][variable_data]`.
    ///
    /// Wire sizes are the *packed* field sizes — `repr(C)` padding is never
    /// serialized, so `size_of::<SstBlockHeader>()` (56) must NOT be used as
    /// the block stride anywhere in this envelope. Count/length prefixes are
    /// checked-converted to u32 (a truncating `as` cast would wrap ≥ 2^32
    /// and silently drop the tail on decode).
    fn encode_metadata(metadata: &SstMetadata) -> Result<Vec<u8>, ProximaDBError> {
        let mut buffer = Vec::new();

        // Global header (60 bytes on the wire)
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

        // Block count (checked: > u32::MAX blocks cannot be encoded)
        let block_count = u32::try_from(metadata.blocks.len()).map_err(|_| {
            ProximaDBError::InvalidInput("SST metadata block count exceeds u32".into())
        })?;
        buffer.extend_from_slice(&block_count.to_le_bytes());

        // Block headers (52 bytes each on the wire)
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

        // Variable-data length prefix: u32, matching what the reader consumes.
        // (A usize prefix would shift every round-tripped buffer by 4 stray
        // bytes — this exact bug shipped and is pinned by the round-trip test.
        // Checked: > 4 GiB of variable data cannot be length-prefixed as u32.)
        let var_data_len = u32::try_from(metadata.variable_data.len()).map_err(|_| {
            ProximaDBError::InvalidInput("SST metadata variable data exceeds u32".into())
        })?;
        buffer.extend_from_slice(&var_data_len.to_le_bytes());

        // Variable data (compressed with bincode if enabled)
        buffer.extend_from_slice(&metadata.variable_data);

        Ok(buffer)
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

        Self::encode_metadata(&metadata)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<Box<dyn EngineMetadata>, ProximaDBError> {
        let mut offset = 0;

        // Deserialize global header manually (60 bytes on the wire; do not
        // use size_of::<SstGlobalHeader>() — repr(C) padding is not serialized)
        if data.len() < offset + 60 {
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
        // 52 bytes per block header on the wire (packed field sizes; the
        // repr(C) size is 56 and striding by it corrupts every block after
        // the first — pinned by the round-trip test below)
        let block_header_size = 52;
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

        // NO BLOOM PRUNING FOR ID LOOKUPS -- deliberately (TD-BLOOM-1).
        //
        // This branch used to probe `variable_data[bloom_filter_offset..]` with a
        // one-byte-per-key "simple bloom check" and treat a zero byte as
        // "definitely absent", returning `true` = skip the whole file. Three
        // things were wrong with it, and the combination is silent data loss on
        // read rather than a slow path:
        //
        //   1. Those bytes are not a filter. `extract_metadata` reads
        //      `BLOOM_OFFSET = 0, BLOOM_SIZE = 4096` -- the file's magic +
        //      header region -- and labels it the global bloom. The real SST
        //      reader puts the filter at `8 + header.header_size`
        //      (`sst/readers/sst_query_engine.rs`). No writer emits filter bytes
        //      at offset 0, so the probe sampled header padding, where `0x00` is
        //      the likely value => "absent" => skip a file that holds the id.
        //   2. The hash was `DefaultHasher`, which is explicitly not stable
        //      across Rust releases, so even a self-consistent writer/reader
        //      pair would diverge on a toolchain bump.
        //   3. Both the slice above and `hash % bloom_data.len()` could panic
        //      (out-of-range offsets; empty filter), violating the no-panic
        //      policy. The sibling trait default in
        //      `proximadb-storage-common::zero_copy_traits` guards the empty
        //      case; this override had silently dropped that guard.
        //
        // An absent capability must answer conservatively: fall through and read
        // the file. Wiring real pruning means populating `global_bloom` (the
        // field is declared below and already imports
        // `proximadb_bloom::SstableBloomFilter`) from bytes a writer actually
        // emits, which is an on-disk format change under CLAUDE.md #8 -- not a
        // one-line re-enable here. DO NOT reintroduce a probe that can answer
        // "absent"; see TD-BLOOM-1 and TD-FACADE-1.

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

#[cfg(test)]
mod tests {
    use super::*;

    /// A metadata record shaped exactly like the pre-TD-BLOOM-1 reader built
    /// it: a header claiming a 4 KiB "global bloom" at offset 0, over bytes that
    /// are all zero (header padding). `max_timestamp` is caller-chosen so the
    /// TTL branch can be exercised independently.
    fn fabricated_bloom_metadata(max_timestamp: u64) -> SstMetadata {
        const FAKE_BLOOM_LEN: u32 = 4096;
        SstMetadata {
            global: SstGlobalHeader {
                file_size: 1 << 20,
                num_blocks: 0,
                bloom_filter_offset: 0,
                bloom_filter_size: FAKE_BLOOM_LEN,
                index_offset: FAKE_BLOOM_LEN,
                index_size: 0,
                total_records: 100,
                min_timestamp: 0,
                max_timestamp,
                compression_ratio: 0,
                reserved: [0; 7],
            },
            blocks: Vec::new(),
            variable_data: vec![0u8; FAKE_BLOOM_LEN as usize],
            global_bloom: parking_lot::RwLock::new(None),
            block_blooms: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// TD-BLOOM-1 pin. This is the exact scenario the removed probe got wrong:
    /// every byte it sampled was 0x00, so it answered "definitely absent" for
    /// every id and `can_skip_file` skipped a file that may hold them. With no
    /// real filter behind the bytes, an ID lookup must never be pruned.
    #[test]
    fn id_lookup_is_never_pruned_without_a_real_filter() {
        let serializer = SstMetadataSerializer::new(crate::testutil::noop_filesystem_port());
        let metadata = fabricated_bloom_metadata(u64::MAX);
        let ctx = QueryContext::for_id_lookup(vec![
            "id-present-in-file".to_string(),
            "another-id".to_string(),
        ]);

        assert!(
            !serializer.can_skip_file(&metadata, &ctx),
            "can_skip_file pruned an ID lookup on bytes that are not a bloom filter \
             (TD-BLOOM-1): this is the false-negative / silent-data-loss direction"
        );
    }

    /// Positive control for the pin above: `can_skip_file` must still be able to
    /// answer `true` when a real criterion holds, or the pin would pass merely
    /// because the function had been reduced to `false`. The TTL branch is that
    /// criterion: every record is older than the threshold.
    #[test]
    fn ttl_expiry_still_prunes_the_whole_file() {
        let serializer = SstMetadataSerializer::new(crate::testutil::noop_filesystem_port());
        let metadata = fabricated_bloom_metadata(1_000);
        let mut ctx = QueryContext::for_id_lookup(vec!["any".to_string()]);
        ctx.metadata_filters
            .insert("ttl_threshold".to_string(), "2000".to_string());

        assert!(
            serializer.can_skip_file(&metadata, &ctx),
            "TTL pruning must still fire when max_timestamp < ttl_threshold"
        );
    }

    /// Encode→decode round-trip — the inverse-pair test this serializer never
    /// had. It pins the two wire bugs that shipped silently:
    /// (1) the reader strided block headers by `size_of::<SstBlockHeader>()`
    ///     (56) while the writer emits 52 packed bytes, corrupting every
    ///     block after the first;
    /// (2) the writer emitted a `usize` (8-byte) variable-data length prefix
    ///     while the reader consumes `u32`, shifting the payload by 4 stray
    ///     bytes.
    /// Three blocks with distinct fields + 1000 bytes of patterned variable
    /// data exercise both paths (fails on either bug).
    #[test]
    fn encode_decode_round_trip_is_lossless() {
        let serializer = SstMetadataSerializer::new(crate::testutil::noop_filesystem_port());

        let blocks: Vec<SstBlockHeader> = (0..3u64)
            .map(|i| SstBlockHeader {
                offset: 4096 + i * (4 * 1024 * 1024),
                compressed_size: 1000 + i as u32,
                uncompressed_size: 2000 + i as u32,
                record_count: 300 + i as u32,
                bloom_offset: 0,
                bloom_size: 0,
                min_key_hash: 0x1000_0000_0000_0000 + i,
                max_key_hash: 0x8000_0000_0000_0000 + i,
                priority: 255 - i as u8,
                reserved: [i as u8; 7],
            })
            .collect();

        let original = SstMetadata {
            global: SstGlobalHeader {
                file_size: 16 * 1024 * 1024,
                num_blocks: blocks.len() as u32,
                // Distinct values: three 0-valued u32 offset fields would be
                // swap-blind (a writer/reader offset swap among them would
                // still round-trip).
                bloom_filter_offset: 111,
                bloom_filter_size: 222,
                index_offset: 333,
                index_size: 60 * 1024,
                total_records: 903,
                min_timestamp: 17,
                max_timestamp: 1_700_000_000_000,
                compression_ratio: 42,
                reserved: [9; 7],
            },
            blocks,
            variable_data: (0..1000u32).map(|i| (i % 251) as u8).collect(),
            global_bloom: parking_lot::RwLock::new(None),
            block_blooms: parking_lot::RwLock::new(HashMap::new()),
        };

        let encoded = SstMetadataSerializer::encode_metadata(&original)
            .expect("encode must succeed for a well-formed fixture");
        let decoded = serializer
            .deserialize_metadata(&encoded)
            .expect("round-trip decode must succeed");
        let decoded = decoded
            .as_any()
            .downcast_ref::<SstMetadata>()
            .expect("decoded metadata must be SstMetadata");

        assert_eq!(decoded.global.file_size, original.global.file_size);
        assert_eq!(decoded.global.num_blocks, 3);
        assert_eq!(
            decoded.global.bloom_filter_offset,
            original.global.bloom_filter_offset
        );
        assert_eq!(
            decoded.global.bloom_filter_size,
            original.global.bloom_filter_size
        );
        assert_eq!(decoded.global.index_offset, original.global.index_offset);
        assert_eq!(decoded.global.index_size, original.global.index_size);
        assert_eq!(decoded.global.total_records, original.global.total_records);
        assert_eq!(decoded.global.min_timestamp, original.global.min_timestamp);
        assert_eq!(decoded.global.max_timestamp, original.global.max_timestamp);
        assert_eq!(decoded.global.compression_ratio, 42);
        assert_eq!(decoded.global.reserved, [9; 7]);

        assert_eq!(decoded.blocks.len(), 3, "block count must survive");
        for (i, (got, want)) in decoded
            .blocks
            .iter()
            .zip(original.blocks.iter())
            .enumerate()
        {
            assert_eq!(got.offset, want.offset, "block {i} offset (stride bug)");
            assert_eq!(got.compressed_size, want.compressed_size, "block {i}");
            assert_eq!(got.uncompressed_size, want.uncompressed_size, "block {i}");
            assert_eq!(got.record_count, want.record_count, "block {i}");
            assert_eq!(got.bloom_offset, want.bloom_offset, "block {i}");
            assert_eq!(got.bloom_size, want.bloom_size, "block {i}");
            assert_eq!(
                got.min_key_hash, want.min_key_hash,
                "block {i} (stride bug)"
            );
            assert_eq!(
                got.max_key_hash, want.max_key_hash,
                "block {i} (stride bug)"
            );
            assert_eq!(got.priority, want.priority, "block {i}");
            assert_eq!(got.reserved, want.reserved, "block {i}");
        }

        assert_eq!(
            decoded.variable_data, original.variable_data,
            "variable data must survive (length-prefix width bug)"
        );
    }
}
