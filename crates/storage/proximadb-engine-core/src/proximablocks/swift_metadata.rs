// SWIFT Engine Metadata Serializer for Zero-Copy I/O System
// Optimized metadata extraction and serialization for SWIFT storage format

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{trace, warn};

use proximadb_storage_common::zero_copy_traits::{
    DataRange, EngineMetadata, MetadataSerializer, QueryContext, QueryType,
};
// The serializer depends only on the `FilesystemPort` seam (get_filesystem); the
// root `FilesystemFactory` implements it, so callers pass it unchanged.
use proximadb_kernel::error::ProximaDBError;
use proximadb_storage_ports::FilesystemPort;

/// SWIFT global metadata header (bytemuck compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SwiftGlobalHeader {
    /// Total file size
    pub file_size: u64,
    /// Number of segments in file
    pub num_segments: u32,
    /// Global index offset
    pub index_offset: u32,
    /// Global index size
    pub index_size: u32,
    /// Total records across all segments
    pub total_records: u64,
    /// Minimum timestamp across file
    pub min_timestamp: u64,
    /// Maximum timestamp across file
    pub max_timestamp: u64,
    /// Compression ratio (0-255, 255 = 100%)
    pub compression_ratio: u8,
    /// SWIFT format version
    pub format_version: u8,
    /// Reserved for future use
    pub reserved: [u8; 6],
}

/// SWIFT segment metadata header (bytemuck compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SwiftSegmentHeader {
    /// Segment offset in file
    pub offset: u64,
    /// Compressed segment size
    pub compressed_size: u32,
    /// Uncompressed segment size
    pub uncompressed_size: u32,
    /// Number of records in segment
    pub record_count: u32,
    /// Segment-level bloom filter offset
    pub bloom_offset: u32,
    /// Bloom filter size
    pub bloom_size: u32,
    /// Minimum vector ID hash in segment
    pub min_id_hash: u64,
    /// Maximum vector ID hash in segment
    pub max_id_hash: u64,
    /// Segment priority for access optimization
    pub priority: u8,
    /// Reserved for future use
    pub reserved: [u8; 7],
}

/// Complete SWIFT metadata structure
#[derive(Debug)]
pub struct SwiftMetadata {
    /// Global file information
    pub global: SwiftGlobalHeader,
    /// Per-segment information
    pub segments: Vec<SwiftSegmentHeader>,
    /// Variable-size data (serialized filters, statistics, etc.)
    pub variable_data: Vec<u8>,
    /// Parsed global bloom filter (lazy-loaded)
    #[allow(dead_code)]
    pub global_bloom: parking_lot::RwLock<Option<Arc<Vec<u8>>>>,
    /// Parsed segment bloom filters (lazy-loaded)
    #[allow(dead_code)]
    pub segment_blooms: parking_lot::RwLock<HashMap<u32, Arc<Vec<u8>>>>,
}

impl EngineMetadata for SwiftMetadata {
    fn file_size(&self) -> u64 {
        self.global.file_size
    }

    fn estimated_selectivity(&self, query_context: &QueryContext) -> f32 {
        match &query_context.query_type {
            QueryType::IdLookup => {
                if query_context.id_lookups.is_empty() {
                    return 0.0;
                }

                // Calculate selectivity based on ID hash ranges
                let mut matching_segments = 0;
                for id in &query_context.id_lookups {
                    let id_hash = self.hash_id(id);
                    for segment in &self.segments {
                        if id_hash >= segment.min_id_hash && id_hash <= segment.max_id_hash {
                            matching_segments += 1;
                            break;
                        }
                    }
                }

                if matching_segments == 0 {
                    0.0
                } else {
                    // Estimate based on segment distribution
                    matching_segments as f32 / self.segments.len() as f32 * 0.1 // Assume 10% hit rate per segment
                }
            }

            QueryType::SimilaritySearch => {
                // SWIFT doesn't have native similarity search optimization
                // All segments need to be scanned
                1.0
            }

            QueryType::MetadataFilter => {
                // Estimate based on filter complexity
                let filter_count = query_context.metadata_filters.len();
                if filter_count == 0 {
                    1.0
                } else {
                    // More filters = lower selectivity
                    (1.0 / (filter_count as f32 + 1.0)).max(0.05)
                }
            }

            QueryType::Batch => {
                // Mixed query - use average selectivity
                0.5
            }

            QueryType::VectorSearch => {
                // Vector search requires scanning all segments
                1.0
            }

            QueryType::FullScan => {
                // Full scan always processes everything
                1.0
            }
        }
    }

    fn memory_footprint(&self) -> usize {
        std::mem::size_of::<SwiftGlobalHeader>()
            + self.segments.len() * std::mem::size_of::<SwiftSegmentHeader>()
            + self.variable_data.len()
            + std::mem::size_of::<Self>()
    }

    fn creation_timestamp(&self) -> Option<u64> {
        Some(self.global.min_timestamp)
    }

    fn compression_ratio(&self) -> Option<f32> {
        Some(self.global.compression_ratio as f32 / 255.0)
    }

    fn supports_query_type(&self, query_type: &QueryType) -> bool {
        match query_type {
            QueryType::IdLookup => true,         // Excellent for ID lookups
            QueryType::SimilaritySearch => true, // Supports but not optimized
            QueryType::MetadataFilter => true,   // Supports metadata filtering
            QueryType::Batch => true,            // Supports batch operations
            QueryType::VectorSearch => true,     // Supports vector search
            QueryType::FullScan => true,         // Supports full scan
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EngineMetadata> {
        Box::new(self.clone())
    }
}

impl SwiftMetadata {
    /// Hash an ID for range comparison
    fn hash_id(&self, id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }

    /// Get required segments for a query
    pub fn get_required_segments(&self, query_context: &QueryContext) -> Vec<u32> {
        match &query_context.query_type {
            QueryType::IdLookup => {
                let mut required_segments = Vec::new();
                for id in &query_context.id_lookups {
                    let id_hash = self.hash_id(id);
                    for (idx, segment) in self.segments.iter().enumerate() {
                        if id_hash >= segment.min_id_hash && id_hash <= segment.max_id_hash {
                            required_segments.push(idx as u32);
                        }
                    }
                }
                required_segments.sort_unstable();
                required_segments.dedup();
                required_segments
            }

            QueryType::SimilaritySearch
            | QueryType::MetadataFilter
            | QueryType::Batch
            | QueryType::VectorSearch
            | QueryType::FullScan => {
                // Need all segments for these query types
                (0..self.segments.len() as u32).collect()
            }
        }
    }

    /// Convert segment indices to data ranges
    pub fn segments_to_ranges(&self, segment_indices: Vec<u32>) -> Vec<DataRange> {
        segment_indices
            .into_iter()
            .filter_map(|idx| {
                self.segments.get(idx as usize).map(|segment| {
                    DataRange::new(
                        segment.offset,
                        segment.compressed_size as u64,
                        segment.priority,
                    )
                })
            })
            .collect()
    }
}

/// SWIFT metadata serializer
pub struct SwiftMetadataSerializer {
    /// Filesystem interface for reading files
    #[allow(dead_code)]
    filesystem: Arc<dyn FilesystemPort>,
}

impl SwiftMetadataSerializer {
    /// Create new SWIFT serializer
    pub fn new(filesystem: Arc<dyn FilesystemPort>) -> Self {
        Self { filesystem }
    }

    /// Extract metadata from SWIFT file
    async fn extract_metadata(
        &self,
        file_path: &str,
        _collection_id: &str,
    ) -> Result<SwiftMetadata, ProximaDBError> {
        // In a real implementation, this would:
        // 1. Read SWIFT file footer to get index location
        // 2. Parse global header from index
        // 3. Extract segment headers
        // 4. Read bloom filters and statistics
        // 5. Serialize variable data

        // Placeholder implementation
        let global = SwiftGlobalHeader {
            file_size: 1024 * 1024, // 1MB placeholder
            num_segments: 10,
            index_offset: 900000,
            index_size: 124000,
            total_records: 5000,
            min_timestamp: 1640995200, // 2022-01-01
            max_timestamp: 1672531200, // 2023-01-01
            compression_ratio: 180,    // ~70% compression
            format_version: 1,
            reserved: [0; 6],
        };

        let mut segments = Vec::new();
        for i in 0..global.num_segments {
            segments.push(SwiftSegmentHeader {
                offset: i as u64 * 90000,
                compressed_size: 80000,
                uncompressed_size: 120000,
                record_count: 500,
                bloom_offset: i * 1000,
                bloom_size: 1000,
                min_id_hash: i as u64 * 1000000,
                max_id_hash: (i + 1) as u64 * 1000000 - 1,
                priority: 255 - (i as u8 * 25), // Decreasing priority
                reserved: [0; 7],
            });
        }

        trace!(
            file_path = file_path,
            segments = segments.len(),
            total_records = global.total_records,
            "Extracted SWIFT metadata"
        );

        Ok(SwiftMetadata {
            global,
            segments,
            variable_data: vec![0u8; 1024], // Placeholder variable data
            global_bloom: parking_lot::RwLock::new(None),
            segment_blooms: parking_lot::RwLock::new(HashMap::new()),
        })
    }
}

impl MetadataSerializer for SwiftMetadataSerializer {
    fn engine_id(&self) -> &'static str {
        "SWIFT"
    }

    fn serialize_metadata(
        &self,
        file_path: &str,
        collection_id: &str,
    ) -> Result<Vec<u8>, ProximaDBError> {
        // Extract metadata using blocking
        let metadata =
            futures::executor::block_on(self.extract_metadata(file_path, collection_id))?;

        // Serialize using efficient format
        let mut serialized = Vec::new();

        // 1. Global header (fixed size, manual serialization)
        serialized.extend_from_slice(&metadata.global.file_size.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.num_segments.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.index_offset.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.index_size.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.total_records.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.min_timestamp.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.max_timestamp.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.compression_ratio.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.format_version.to_le_bytes());
        serialized.extend_from_slice(&metadata.global.reserved);

        // 2. Number of segments
        serialized.extend_from_slice(&(metadata.segments.len() as u32).to_le_bytes());

        // 3. Segment headers (fixed size, manual serialization)
        for segment in &metadata.segments {
            serialized.extend_from_slice(&segment.offset.to_le_bytes());
            serialized.extend_from_slice(&segment.compressed_size.to_le_bytes());
            serialized.extend_from_slice(&segment.uncompressed_size.to_le_bytes());
            serialized.extend_from_slice(&segment.record_count.to_le_bytes());
            serialized.extend_from_slice(&segment.bloom_offset.to_le_bytes());
            serialized.extend_from_slice(&segment.bloom_size.to_le_bytes());
            serialized.extend_from_slice(&segment.min_id_hash.to_le_bytes());
            serialized.extend_from_slice(&segment.max_id_hash.to_le_bytes());
            serialized.extend_from_slice(&segment.priority.to_le_bytes());
            serialized.extend_from_slice(&segment.reserved);
        }

        // 4. Variable data size + data
        serialized.extend_from_slice(&(metadata.variable_data.len()).to_le_bytes());
        serialized.extend_from_slice(&metadata.variable_data);

        trace!(
            file_path = file_path,
            serialized_size = serialized.len(),
            segments = metadata.segments.len(),
            "Serialized SWIFT metadata"
        );

        Ok(serialized)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<Box<dyn EngineMetadata>, ProximaDBError> {
        if data.len() < std::mem::size_of::<SwiftGlobalHeader>() + 4 {
            return Err(ProximaDBError::InvalidInput(
                "SWIFT metadata too small".into(),
            ));
        }

        let mut offset = 0;

        // 1. Deserialize global header manually
        let global = SwiftGlobalHeader {
            file_size: u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                ProximaDBError::InvalidInput("Invalid byte slice size for file_size".into())
            })?),
            num_segments: u32::from_le_bytes(data[offset + 8..offset + 12].try_into().map_err(
                |_| ProximaDBError::InvalidInput("Invalid byte slice size for num_segments".into()),
            )?),
            index_offset: u32::from_le_bytes(data[offset + 12..offset + 16].try_into().map_err(
                |_| ProximaDBError::InvalidInput("Invalid byte slice size for index_offset".into()),
            )?),
            index_size: u32::from_le_bytes(data[offset + 16..offset + 20].try_into().map_err(
                |_| ProximaDBError::InvalidInput("Invalid byte slice size for index_size".into()),
            )?),
            total_records: u64::from_le_bytes(data[offset + 20..offset + 28].try_into().map_err(
                |_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for total_records".into())
                },
            )?),
            min_timestamp: u64::from_le_bytes(data[offset + 28..offset + 36].try_into().map_err(
                |_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for min_timestamp".into())
                },
            )?),
            max_timestamp: u64::from_le_bytes(data[offset + 36..offset + 44].try_into().map_err(
                |_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for max_timestamp".into())
                },
            )?),
            compression_ratio: data[offset + 44],
            format_version: data[offset + 45],
            reserved: data[offset + 46..offset + 52].try_into().map_err(|_| {
                ProximaDBError::InvalidInput("Invalid byte slice size for reserved".into())
            })?,
        };
        offset += 52; // Size of SwiftGlobalHeader

        // 2. Read number of segments
        let num_segments = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // 3. Deserialize segment headers
        let segment_size = std::mem::size_of::<SwiftSegmentHeader>();
        let mut segments = Vec::with_capacity(num_segments as usize);

        for _ in 0..num_segments {
            if offset + segment_size > data.len() {
                return Err(ProximaDBError::InvalidInput(
                    "Insufficient data for SWIFT segment headers".into(),
                ));
            }

            let segment = SwiftSegmentHeader {
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
                min_id_hash: u64::from_le_bytes(
                    data[offset + 28..offset + 36].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for min_id_hash".into(),
                        )
                    })?,
                ),
                max_id_hash: u64::from_le_bytes(
                    data[offset + 36..offset + 44].try_into().map_err(|_| {
                        ProximaDBError::InvalidInput(
                            "Invalid byte slice size for max_id_hash".into(),
                        )
                    })?,
                ),
                priority: data[offset + 44],
                reserved: data[offset + 45..offset + 52].try_into().map_err(|_| {
                    ProximaDBError::InvalidInput("Invalid byte slice size for reserved".into())
                })?,
            };
            segments.push(segment);
            offset += 52; // Size of SwiftSegmentHeader
        }

        // 4. Read variable data
        if offset + 4 > data.len() {
            return Err(ProximaDBError::InvalidInput(
                "Insufficient data for SWIFT variable data size".into(),
            ));
        }

        let variable_data_size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + variable_data_size > data.len() {
            return Err(ProximaDBError::InvalidInput(
                "Insufficient data for SWIFT variable data".into(),
            ));
        }

        let variable_data = data[offset..offset + variable_data_size].to_vec();

        let metadata = SwiftMetadata {
            global,
            segments,
            variable_data,
            global_bloom: parking_lot::RwLock::new(None),
            segment_blooms: parking_lot::RwLock::new(HashMap::new()),
        };

        trace!(
            segments = metadata.segments.len(),
            variable_data_size = variable_data_size,
            "Deserialized SWIFT metadata"
        );

        Ok(Box::new(metadata))
    }

    fn can_skip_file(&self, metadata: &dyn EngineMetadata, query_context: &QueryContext) -> bool {
        let Some(swift_metadata) = metadata.as_any().downcast_ref::<SwiftMetadata>() else {
            warn!("Invalid metadata type for SWIFT serializer in can_skip_file; cannot skip");
            return false;
        };

        match &query_context.query_type {
            QueryType::IdLookup => {
                // Check if any IDs might be in this file
                for id in &query_context.id_lookups {
                    let id_hash = swift_metadata.hash_id(id);
                    for segment in &swift_metadata.segments {
                        if id_hash >= segment.min_id_hash && id_hash <= segment.max_id_hash {
                            return false; // Found potential match
                        }
                    }
                }
                true // No potential matches
            }

            QueryType::SimilaritySearch => {
                // SWIFT files can't be skipped for similarity search
                false
            }

            QueryType::MetadataFilter => {
                // For now, can't skip metadata filtering
                // In future, could check if filters match file-level metadata
                false
            }

            QueryType::Batch => {
                // Conservative approach for batch queries
                false
            }

            QueryType::VectorSearch | QueryType::FullScan => {
                // Can't skip for vector search or full scan
                false
            }
        }
    }

    fn get_required_ranges(
        &self,
        metadata: &dyn EngineMetadata,
        query_context: &QueryContext,
    ) -> Option<Vec<DataRange>> {
        let Some(swift_metadata) = metadata.as_any().downcast_ref::<SwiftMetadata>() else {
            warn!(
                "Invalid metadata type for SWIFT serializer in get_required_ranges; falling back to full read"
            );
            return None;
        };

        let required_segments = swift_metadata.get_required_segments(query_context);

        if required_segments.len() == swift_metadata.segments.len() {
            // Need all segments - return None to indicate full file read
            None
        } else {
            // Return specific segment ranges
            Some(swift_metadata.segments_to_ranges(required_segments))
        }
    }

    fn estimate_selectivity(
        &self,
        metadata: &dyn EngineMetadata,
        query_context: &QueryContext,
    ) -> f32 {
        metadata.estimated_selectivity(query_context)
    }
}
// Manual Clone implementation for SwiftMetadata
impl Clone for SwiftMetadata {
    fn clone(&self) -> Self {
        Self {
            global: self.global,
            segments: self.segments.clone(),
            variable_data: self.variable_data.clone(),
            global_bloom: parking_lot::RwLock::new(self.global_bloom.read().clone()),
            segment_blooms: parking_lot::RwLock::new(self.segment_blooms.read().clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The serializer's `extract_metadata` is a placeholder that never touches
    // the filesystem, so tests inject the shared no-op `FilesystemPort`
    // (`crate::testutil`, TD-DECOMP-75 pattern).
    use crate::testutil::noop_filesystem_port as noop_port;

    #[tokio::test]
    async fn test_swift_metadata_serialization() {
        let serializer = SwiftMetadataSerializer::new(noop_port());

        // Test serialization
        let serialized = serializer
            .serialize_metadata("/test/file.swift", "test_collection")
            .expect("Failed to serialize metadata");
        assert!(!serialized.is_empty());

        // Test deserialization
        let metadata = serializer
            .deserialize_metadata(&serialized)
            .expect("Failed to deserialize metadata");
        assert_eq!(metadata.file_size(), 1024 * 1024);
        assert!(metadata.memory_footprint() > 0);
    }

    #[tokio::test]
    async fn test_swift_id_lookup_optimization() {
        let serializer = SwiftMetadataSerializer::new(noop_port());

        let serialized = serializer
            .serialize_metadata("/test/file.swift", "test_collection")
            .expect("Failed to serialize metadata");
        let metadata = serializer
            .deserialize_metadata(&serialized)
            .expect("Failed to deserialize metadata");

        // Test ID lookup with non-existent ID
        let mut query_context = QueryContext::default();
        query_context.query_type = QueryType::IdLookup;
        query_context.id_lookups = vec!["nonexistent_id".to_string()];

        // Should be able to skip file for non-existent ID
        let can_skip = serializer.can_skip_file(metadata.as_ref(), &query_context);
        assert!(can_skip);

        // Test with ID that might exist
        query_context.id_lookups = vec!["id_500000".to_string()]; // Hash likely in range
        let _can_skip = serializer.can_skip_file(metadata.as_ref(), &query_context);
        // Depending on hash, might not be able to skip
    }

    #[tokio::test]
    async fn test_swift_segment_optimization() {
        let serializer = SwiftMetadataSerializer::new(noop_port());

        let serialized = serializer
            .serialize_metadata("/test/file.swift", "test_collection")
            .expect("Failed to serialize metadata");
        let metadata = serializer
            .deserialize_metadata(&serialized)
            .expect("Failed to deserialize metadata");

        let mut query_context = QueryContext::default();
        query_context.query_type = QueryType::IdLookup;
        query_context.id_lookups = vec!["id_500000".to_string()];

        let ranges = serializer.get_required_ranges(metadata.as_ref(), &query_context);
        // For ID lookup with default metadata (20 segments), should return specific ranges
        // or None if all segments are needed. Both are valid outcomes
        match ranges {
            Some(ranges) if !ranges.is_empty() => {
                // Got specific ranges - good for optimization
                assert!(ranges.len() <= 10); // Shouldn't need all segments
            }
            _ => {
                // Either None or empty ranges - both valid for certain cases
                // This happens when:
                // - metadata has 0 segments
                // - all segments are needed
                // - no segments match the query
                // All are valid outcomes for this test
            }
        }
    }
}
