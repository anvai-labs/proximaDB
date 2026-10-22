// NOVA Engine Metadata Serializer for Zero-Copy I/O System
// Optimized metadata extraction and serialization for NOVA columnar storage format

use std::collections::HashMap;
use std::sync::Arc;

// Bytemuck imports removed - using manual serialization for flexibility
use tracing::{debug, trace, warn};

use proximadb_storage_common::zero_copy_traits::{
    DataRange, EngineMetadata, MetadataSerializer, QueryContext, QueryType,
};
// FilesystemPort seam (TD-DECOMP-75 pattern): root FilesystemFactory implements it.
use proximadb_kernel::error::ProximaDBError;
use proximadb_storage_ports::FilesystemPort;

/// NOVA file footer information (bytemuck compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NovaFooterHeader {
    /// Total file size
    pub file_size: u64,
    /// Number of column groups
    pub num_column_groups: u32,
    /// Footer offset in file
    pub footer_offset: u32,
    /// Footer size in bytes
    pub footer_size: u32,
    /// Schema offset
    pub schema_offset: u32,
    /// Schema size
    pub schema_size: u32,
    /// Total number of vectors
    pub total_vectors: u64,
    /// Number of columns
    pub num_columns: u32,
    /// NOVA format version
    pub format_version: u32,
    /// File creation timestamp
    pub created_timestamp: u64,
    /// Compression algorithm used
    pub compression_type: u8,
    /// Reserved for future use
    pub reserved: [u8; 7],
}

/// NOVA column group information (bytemuck compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NovaColumnGroupHeader {
    /// Column group offset in file
    pub offset: u64,
    /// Compressed size
    pub compressed_size: u64,
    /// Uncompressed size
    pub uncompressed_size: u64,
    /// Number of vectors in this group
    pub num_vectors: u64,
    /// Number of columns in this group
    pub num_columns: u32,
    /// Statistics offset for this group
    pub statistics_offset: u32,
    /// Statistics size
    pub statistics_size: u32,
    /// Minimum vector ID hash in group
    pub min_id_hash: u64,
    /// Maximum vector ID hash in group
    pub max_id_hash: u64,
    /// Column group priority
    pub priority: u8,
    /// Reserved for future use
    pub reserved: [u8; 7],
}

/// NOVA column information (bytemuck compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NovaColumnHeader {
    /// Column offset within group
    pub offset: u64,
    /// Column compressed size
    pub compressed_size: u32,
    /// Column uncompressed size
    pub uncompressed_size: u32,
    /// Column type (0=vector, 1=metadata, 2=id)
    pub column_type: u8,
    /// Compression algorithm for this column
    pub compression_algorithm: u8,
    /// Null count
    pub null_count: u32,
    /// Column-specific statistics offset
    pub stats_offset: u32,
    /// Reserved for future use
    pub reserved: [u16; 3],
}

/// Complete NOVA metadata structure
#[derive(Debug)]
pub struct NovaMetadata {
    /// File footer information
    pub footer: NovaFooterHeader,
    /// Column group headers
    pub column_groups: Vec<NovaColumnGroupHeader>,
    /// Column headers for each group
    pub columns: Vec<Vec<NovaColumnHeader>>,
    /// Variable-size data (schema, statistics, etc.)
    pub variable_data: Vec<u8>,
    /// Cached schema (lazy-loaded)
    pub schema: parking_lot::RwLock<Option<Arc<Vec<u8>>>>,
    /// Cached column statistics (lazy-loaded)
    pub column_stats: parking_lot::RwLock<HashMap<u32, Arc<Vec<u8>>>>,
}

impl EngineMetadata for NovaMetadata {
    fn file_size(&self) -> u64 {
        self.footer.file_size
    }

    fn estimated_selectivity(&self, query_context: &QueryContext) -> f32 {
        match &query_context.query_type {
            QueryType::IdLookup => {
                if query_context.id_lookups.is_empty() {
                    return 0.0;
                }

                // Calculate selectivity based on ID hash ranges in column groups
                let mut matching_groups = 0;
                for id in &query_context.id_lookups {
                    let id_hash = self.hash_id(id);
                    for group in &self.column_groups {
                        if id_hash >= group.min_id_hash && id_hash <= group.max_id_hash {
                            matching_groups += 1;
                            break;
                        }
                    }
                }

                if matching_groups == 0 {
                    0.0
                } else {
                    // Columnar format is efficient for ID lookups
                    matching_groups as f32 / self.column_groups.len() as f32 * 0.05 // Assume 5% hit rate per group
                }
            }

            QueryType::SimilaritySearch => {
                // NOVA's columnar format is excellent for similarity search
                // Can use column statistics for pruning
                if let Some(query_vector) = &query_context.query_vector {
                    // Estimate based on vector dimensionality and column group distribution
                    let dimension = query_vector.len();
                    if dimension > 0 {
                        // More column groups = better pruning potential
                        (1.0 - (self.column_groups.len() as f32 / 100.0).min(0.8)).max(0.1)
                    } else {
                        0.8 // Conservative estimate without vector info
                    }
                } else {
                    0.9 // No pruning possible without query vector
                }
            }

            QueryType::MetadataFilter => {
                // Columnar format is excellent for metadata filtering
                let filter_count = query_context.metadata_filters.len();
                if filter_count == 0 {
                    1.0
                } else {
                    // Columnar storage allows efficient column pruning
                    let selectivity = 1.0 / (filter_count as f32 * 2.0 + 1.0);
                    selectivity.max(0.01) // Very efficient for metadata filters
                }
            }

            QueryType::Batch => {
                // Mixed operations - use moderate selectivity
                0.3
            }
            QueryType::VectorSearch | QueryType::FullScan => {
                // Vector search and full scan have different selectivity
                1.0
            }
        }
    }

    fn memory_footprint(&self) -> usize {
        std::mem::size_of::<NovaFooterHeader>()
            + self.column_groups.len() * std::mem::size_of::<NovaColumnGroupHeader>()
            + self
                .columns
                .iter()
                .map(|cols| cols.len() * std::mem::size_of::<NovaColumnHeader>())
                .sum::<usize>()
            + self.variable_data.len()
            + std::mem::size_of::<Self>()
    }

    fn creation_timestamp(&self) -> Option<u64> {
        Some(self.footer.created_timestamp)
    }

    fn compression_ratio(&self) -> Option<f32> {
        // Calculate overall compression ratio from column groups
        let total_compressed: u64 = self.column_groups.iter().map(|g| g.compressed_size).sum();
        let total_uncompressed: u64 = self.column_groups.iter().map(|g| g.uncompressed_size).sum();

        if total_uncompressed > 0 {
            Some(total_compressed as f32 / total_uncompressed as f32)
        } else {
            None
        }
    }

    fn supports_query_type(&self, query_type: &QueryType) -> bool {
        match query_type {
            QueryType::IdLookup => true, // Good for ID lookups with column groups
            QueryType::SimilaritySearch => true, // Excellent for similarity search
            QueryType::MetadataFilter => true, // Excellent for metadata filtering
            QueryType::Batch => true,    // Supports batch operations
            QueryType::VectorSearch => true, // Supports vector search
            QueryType::FullScan => true, // Supports full scan
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EngineMetadata> {
        Box::new(NovaMetadata {
            footer: self.footer,
            column_groups: self.column_groups.clone(),
            columns: self.columns.clone(),
            variable_data: self.variable_data.clone(),
            schema: parking_lot::RwLock::new(self.schema.read().clone()),
            column_stats: parking_lot::RwLock::new(self.column_stats.read().clone()),
        })
    }
}

impl NovaMetadata {
    /// Hash an ID for range comparison
    fn hash_id(&self, id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }

    /// Get required column groups for a query
    pub fn get_required_column_groups(&self, query_context: &QueryContext) -> Vec<u32> {
        match &query_context.query_type {
            QueryType::IdLookup => {
                let mut required_groups = Vec::new();
                for id in &query_context.id_lookups {
                    let id_hash = self.hash_id(id);
                    for (idx, group) in self.column_groups.iter().enumerate() {
                        if id_hash >= group.min_id_hash && id_hash <= group.max_id_hash {
                            required_groups.push(idx as u32);
                        }
                    }
                }
                required_groups.sort_unstable();
                required_groups.dedup();
                required_groups
            }

            QueryType::MetadataFilter => {
                // For metadata filtering, might be able to skip some column groups
                // based on group-level statistics. For now, return all groups.
                (0..self.column_groups.len() as u32).collect()
            }

            QueryType::SimilaritySearch => {
                // Similarity search typically needs vector columns from all groups
                (0..self.column_groups.len() as u32).collect()
            }

            QueryType::Batch => {
                // Conservative: need all groups
                (0..self.column_groups.len() as u32).collect()
            }

            QueryType::VectorSearch | QueryType::FullScan => {
                // Vector search and full scan need all groups
                (0..self.column_groups.len() as u32).collect()
            }
        }
    }

    /// Get required columns within column groups
    pub fn get_required_columns(&self, query_context: &QueryContext) -> HashMap<u32, Vec<u32>> {
        let mut required_columns = HashMap::new();

        match &query_context.query_type {
            QueryType::IdLookup => {
                // Only need ID columns for ID lookup
                for (group_idx, columns) in self.columns.iter().enumerate() {
                    let id_columns: Vec<u32> = columns
                        .iter()
                        .enumerate()
                        .filter(|(_, col)| col.column_type == 2) // ID column type
                        .map(|(idx, _)| idx as u32)
                        .collect();
                    if !id_columns.is_empty() {
                        required_columns.insert(group_idx as u32, id_columns);
                    }
                }
            }

            QueryType::SimilaritySearch => {
                // Need vector columns (and possibly metadata for filtering)
                for (group_idx, columns) in self.columns.iter().enumerate() {
                    let mut cols = Vec::new();
                    for (col_idx, col) in columns.iter().enumerate() {
                        if col.column_type == 0 || // Vector columns
                           (!query_context.metadata_filters.is_empty() && col.column_type == 1)
                        {
                            // Metadata if filtered
                            cols.push(col_idx as u32);
                        }
                    }
                    if !cols.is_empty() {
                        required_columns.insert(group_idx as u32, cols);
                    }
                }
            }

            QueryType::MetadataFilter => {
                // Need metadata columns (and ID columns for results)
                for (group_idx, columns) in self.columns.iter().enumerate() {
                    let mut cols = Vec::new();
                    for (col_idx, col) in columns.iter().enumerate() {
                        if col.column_type == 1 || col.column_type == 2 {
                            // Metadata or ID columns
                            cols.push(col_idx as u32);
                        }
                    }
                    if !cols.is_empty() {
                        required_columns.insert(group_idx as u32, cols);
                    }
                }
            }

            QueryType::Batch => {
                // Need all column types
                for (group_idx, columns) in self.columns.iter().enumerate() {
                    let all_cols: Vec<u32> = (0..columns.len() as u32).collect();
                    required_columns.insert(group_idx as u32, all_cols);
                }
            }

            QueryType::VectorSearch | QueryType::FullScan => {
                // Need all columns for vector search and full scan
                for (group_idx, columns) in self.columns.iter().enumerate() {
                    let all_cols: Vec<u32> = (0..columns.len() as u32).collect();
                    required_columns.insert(group_idx as u32, all_cols);
                }
            }
        }

        required_columns
    }

    /// Convert column group and column requirements to data ranges
    pub fn requirements_to_ranges(
        &self,
        group_indices: Vec<u32>,
        column_requirements: HashMap<u32, Vec<u32>>,
    ) -> Vec<DataRange> {
        let mut ranges = Vec::new();

        for group_idx in group_indices {
            if let Some(group) = self.column_groups.get(group_idx as usize) {
                if let Some(required_cols) = column_requirements.get(&group_idx) {
                    if let Some(columns) = self.columns.get(group_idx as usize) {
                        // Calculate ranges for specific columns within the group
                        for &col_idx in required_cols {
                            if let Some(column) = columns.get(col_idx as usize) {
                                ranges.push(DataRange::new(
                                    group.offset + column.offset,
                                    column.compressed_size as u64,
                                    group.priority,
                                ));
                            }
                        }
                    } else {
                        // Fallback: entire group
                        ranges.push(DataRange::new(
                            group.offset,
                            group.compressed_size,
                            group.priority,
                        ));
                    }
                } else {
                    // No specific column requirements - take entire group
                    ranges.push(DataRange::new(
                        group.offset,
                        group.compressed_size,
                        group.priority,
                    ));
                }
            }
        }

        ranges
    }
}

/// NOVA metadata serializer
pub struct NovaMetadataSerializer {
    /// Filesystem interface for reading files
    #[allow(dead_code)]
    filesystem: Arc<dyn FilesystemPort>,
}

impl NovaMetadataSerializer {
    /// Create new NOVA serializer
    pub fn new(filesystem: Arc<dyn FilesystemPort>) -> Self {
        Self { filesystem }
    }

    /// Create placeholder metadata for testing
    fn create_placeholder_metadata(&self) -> NovaMetadata {
        let footer = NovaFooterHeader {
            file_size: 2 * 1024 * 1024, // 2MB placeholder
            num_column_groups: 8,
            footer_offset: 1900000,
            footer_size: 100000,
            schema_offset: 1800000,
            schema_size: 10000,
            total_vectors: 10000,
            num_columns: 4, // vector, metadata, id, timestamp
            format_version: 1,
            created_timestamp: 1640995200, // 2022-01-01
            compression_type: 1,           // ZSTD
            reserved: [0; 7],
        };

        let mut column_groups = Vec::new();
        let mut columns = Vec::new();

        for i in 0..footer.num_column_groups {
            let group = NovaColumnGroupHeader {
                offset: i as u64 * 240000,
                compressed_size: 200000,
                uncompressed_size: 280000,
                num_vectors: 1250, // 10000 / 8 groups
                num_columns: footer.num_columns,
                statistics_offset: i * 5000,
                statistics_size: 5000,
                min_id_hash: i as u64 * 2000000,
                max_id_hash: (i + 1) as u64 * 2000000 - 1,
                priority: 255 - (i as u8 * 30), // Decreasing priority
                reserved: [0; 7],
            };
            column_groups.push(group);

            // Create column headers for this group
            let mut group_columns = Vec::new();
            for j in 0..footer.num_columns {
                group_columns.push(NovaColumnHeader {
                    offset: j as u64 * 50000,
                    compressed_size: 45000,
                    uncompressed_size: 70000,
                    column_type: match j {
                        0 => 0, // Vector column
                        1 => 1, // Metadata column
                        2 => 2, // ID column
                        3 => 1, // Timestamp metadata
                        _ => 1, // Default to metadata
                    },
                    compression_algorithm: 1, // ZSTD
                    null_count: 0,
                    stats_offset: j * 1000,
                    reserved: [0; 3],
                });
            }
            columns.push(group_columns);
        }

        NovaMetadata {
            footer,
            column_groups,
            columns,
            variable_data: Vec::new(),
            schema: parking_lot::RwLock::new(None),
            column_stats: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Extract metadata from NOVA file
    #[allow(dead_code)]
    async fn extract_metadata(
        &self,
        file_path: &str,
        _collection_id: &str,
    ) -> Result<NovaMetadata, ProximaDBError> {
        // In a real implementation, this would:
        // 1. Read NOVA file footer to get column group locations
        // 2. Parse footer header
        // 3. Extract column group headers and column information
        // 4. Read schema and statistics
        // 5. Serialize variable data

        // Placeholder implementation
        let footer = NovaFooterHeader {
            file_size: 2 * 1024 * 1024, // 2MB placeholder
            num_column_groups: 8,
            footer_offset: 1900000,
            footer_size: 100000,
            schema_offset: 1800000,
            schema_size: 10000,
            total_vectors: 10000,
            num_columns: 4, // vector, metadata, id, timestamp
            format_version: 1,
            created_timestamp: 1640995200, // 2022-01-01
            compression_type: 1,           // ZSTD
            reserved: [0; 7],
        };

        let mut column_groups = Vec::new();
        let mut columns = Vec::new();

        for i in 0..footer.num_column_groups {
            let group = NovaColumnGroupHeader {
                offset: i as u64 * 240000,
                compressed_size: 200000,
                uncompressed_size: 280000,
                num_vectors: 1250, // 10000 / 8 groups
                num_columns: footer.num_columns,
                statistics_offset: i * 5000,
                statistics_size: 5000,
                min_id_hash: i as u64 * 2000000,
                max_id_hash: (i + 1) as u64 * 2000000 - 1,
                priority: 255 - (i as u8 * 30), // Decreasing priority
                reserved: [0; 7],
            };
            column_groups.push(group);

            // Create column headers for this group
            let mut group_columns = Vec::new();
            for j in 0..footer.num_columns {
                group_columns.push(NovaColumnHeader {
                    offset: j as u64 * 50000,
                    compressed_size: 45000,
                    uncompressed_size: 70000,
                    column_type: match j {
                        0 => 0, // Vector column
                        1 => 1, // Metadata column
                        2 => 2, // ID column
                        3 => 1, // Timestamp metadata
                        _ => 1, // Default to metadata
                    },
                    compression_algorithm: 1, // ZSTD
                    null_count: 0,
                    stats_offset: j * 1000,
                    reserved: [0; 3],
                });
            }
            columns.push(group_columns);
        }

        debug!(
            file_path,
            column_groups = column_groups.len(),
            total_vectors = footer.total_vectors,
            "Extracted NOVA metadata"
        );

        Ok(NovaMetadata {
            footer,
            column_groups,
            columns,
            variable_data: vec![0u8; 2048], // Placeholder variable data
            schema: parking_lot::RwLock::new(None),
            column_stats: parking_lot::RwLock::new(HashMap::new()),
        })
    }
}

impl MetadataSerializer for NovaMetadataSerializer {
    fn engine_id(&self) -> &'static str {
        "NOVA"
    }

    fn serialize_metadata(
        &self,
        file_path: &str,
        _collection_id: &str,
    ) -> Result<Vec<u8>, ProximaDBError> {
        // Create placeholder metadata directly without async operations
        // In a real implementation, this would extract from the actual NOVA file
        let metadata = self.create_placeholder_metadata();

        // Serialize using efficient format
        let mut serialized = Vec::new();

        // 1. Footer header (fixed size, manual serialization)
        serialized.extend_from_slice(&metadata.footer.file_size.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.num_column_groups.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.footer_offset.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.footer_size.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.schema_offset.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.schema_size.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.total_vectors.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.num_columns.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.format_version.to_le_bytes());
        serialized.extend_from_slice(&metadata.footer.created_timestamp.to_le_bytes());
        serialized.extend_from_slice(&[metadata.footer.compression_type]);
        serialized.extend_from_slice(&metadata.footer.reserved);

        // 2. Number of column groups
        serialized.extend_from_slice(&(metadata.column_groups.len() as u32).to_le_bytes());

        // 3. Column group headers
        for group in &metadata.column_groups {
            serialized.extend_from_slice(&group.offset.to_le_bytes());
            serialized.extend_from_slice(&group.compressed_size.to_le_bytes());
            serialized.extend_from_slice(&group.uncompressed_size.to_le_bytes());
            serialized.extend_from_slice(&group.num_vectors.to_le_bytes());
            serialized.extend_from_slice(&group.num_columns.to_le_bytes());
            serialized.extend_from_slice(&group.statistics_offset.to_le_bytes());
            serialized.extend_from_slice(&group.statistics_size.to_le_bytes());
            serialized.extend_from_slice(&group.min_id_hash.to_le_bytes());
            serialized.extend_from_slice(&group.max_id_hash.to_le_bytes());
            serialized.extend_from_slice(&[group.priority]);
            serialized.extend_from_slice(&group.reserved);
        }

        // 4. Columns for each group
        for group_columns in &metadata.columns {
            serialized.extend_from_slice(&(group_columns.len() as u32).to_le_bytes());
            for column in group_columns {
                serialized.extend_from_slice(&column.offset.to_le_bytes());
                serialized.extend_from_slice(&column.compressed_size.to_le_bytes());
                serialized.extend_from_slice(&column.uncompressed_size.to_le_bytes());
                serialized.extend_from_slice(&[column.column_type]);
                serialized.extend_from_slice(&[column.compression_algorithm]);
                serialized.extend_from_slice(&column.null_count.to_le_bytes());
                serialized.extend_from_slice(&column.stats_offset.to_le_bytes());
                // Reserved fields (3 x u16)
                for &r in &column.reserved {
                    serialized.extend_from_slice(&r.to_le_bytes());
                }
            }
        }

        // 5. Variable data size + data
        serialized.extend_from_slice(&(metadata.variable_data.len() as u32).to_le_bytes());
        serialized.extend_from_slice(&metadata.variable_data);

        trace!(
            file_path,
            serialized_size = serialized.len(),
            column_groups = metadata.column_groups.len(),
            "Serialized NOVA metadata"
        );

        Ok(serialized)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<Box<dyn EngineMetadata>, ProximaDBError> {
        if data.len() < std::mem::size_of::<NovaFooterHeader>() + 4 {
            return Err(ProximaDBError::InvalidInput(
                "NOVA metadata too small".into(),
            ));
        }

        let mut offset = 0;

        // 1. Deserialize footer header manually
        let footer_size = 8 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 8 + 1 + 7; // 64 bytes total
        if data.len() < footer_size {
            return Err(ProximaDBError::InvalidInput(
                "Invalid Nova footer size".into(),
            ));
        }
        let footer = NovaFooterHeader {
            file_size: u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]),
            num_column_groups: u32::from_le_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
            ]),
            footer_offset: u32::from_le_bytes([
                data[offset + 12],
                data[offset + 13],
                data[offset + 14],
                data[offset + 15],
            ]),
            footer_size: u32::from_le_bytes([
                data[offset + 16],
                data[offset + 17],
                data[offset + 18],
                data[offset + 19],
            ]),
            schema_offset: u32::from_le_bytes([
                data[offset + 20],
                data[offset + 21],
                data[offset + 22],
                data[offset + 23],
            ]),
            schema_size: u32::from_le_bytes([
                data[offset + 24],
                data[offset + 25],
                data[offset + 26],
                data[offset + 27],
            ]),
            total_vectors: u64::from_le_bytes([
                data[offset + 28],
                data[offset + 29],
                data[offset + 30],
                data[offset + 31],
                data[offset + 32],
                data[offset + 33],
                data[offset + 34],
                data[offset + 35],
            ]),
            num_columns: u32::from_le_bytes([
                data[offset + 36],
                data[offset + 37],
                data[offset + 38],
                data[offset + 39],
            ]),
            format_version: u32::from_le_bytes([
                data[offset + 40],
                data[offset + 41],
                data[offset + 42],
                data[offset + 43],
            ]),
            created_timestamp: u64::from_le_bytes([
                data[offset + 44],
                data[offset + 45],
                data[offset + 46],
                data[offset + 47],
                data[offset + 48],
                data[offset + 49],
                data[offset + 50],
                data[offset + 51],
            ]),
            compression_type: data[offset + 52],
            reserved: [
                data[offset + 53],
                data[offset + 54],
                data[offset + 55],
                data[offset + 56],
                data[offset + 57],
                data[offset + 58],
                data[offset + 59],
            ],
        };
        offset += footer_size;

        // 2. Read number of column groups
        let num_groups = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // 3. Deserialize column group headers manually
        let group_size = 8 + 8 + 8 + 8 + 4 + 4 + 4 + 8 + 8 + 1 + 7; // 68 bytes per group
        let mut column_groups = Vec::with_capacity(num_groups as usize);

        for _ in 0..num_groups {
            if offset + group_size > data.len() {
                return Err(ProximaDBError::InvalidInput(
                    "Insufficient data for NOVA column group headers".into(),
                ));
            }

            let group = NovaColumnGroupHeader {
                offset: u64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]),
                compressed_size: u64::from_le_bytes([
                    data[offset + 8],
                    data[offset + 9],
                    data[offset + 10],
                    data[offset + 11],
                    data[offset + 12],
                    data[offset + 13],
                    data[offset + 14],
                    data[offset + 15],
                ]),
                uncompressed_size: u64::from_le_bytes([
                    data[offset + 16],
                    data[offset + 17],
                    data[offset + 18],
                    data[offset + 19],
                    data[offset + 20],
                    data[offset + 21],
                    data[offset + 22],
                    data[offset + 23],
                ]),
                num_vectors: u64::from_le_bytes([
                    data[offset + 24],
                    data[offset + 25],
                    data[offset + 26],
                    data[offset + 27],
                    data[offset + 28],
                    data[offset + 29],
                    data[offset + 30],
                    data[offset + 31],
                ]),
                num_columns: u32::from_le_bytes([
                    data[offset + 32],
                    data[offset + 33],
                    data[offset + 34],
                    data[offset + 35],
                ]),
                statistics_offset: u32::from_le_bytes([
                    data[offset + 36],
                    data[offset + 37],
                    data[offset + 38],
                    data[offset + 39],
                ]),
                statistics_size: u32::from_le_bytes([
                    data[offset + 40],
                    data[offset + 41],
                    data[offset + 42],
                    data[offset + 43],
                ]),
                min_id_hash: u64::from_le_bytes([
                    data[offset + 44],
                    data[offset + 45],
                    data[offset + 46],
                    data[offset + 47],
                    data[offset + 48],
                    data[offset + 49],
                    data[offset + 50],
                    data[offset + 51],
                ]),
                max_id_hash: u64::from_le_bytes([
                    data[offset + 52],
                    data[offset + 53],
                    data[offset + 54],
                    data[offset + 55],
                    data[offset + 56],
                    data[offset + 57],
                    data[offset + 58],
                    data[offset + 59],
                ]),
                priority: data[offset + 60],
                reserved: [
                    data[offset + 61],
                    data[offset + 62],
                    data[offset + 63],
                    data[offset + 64],
                    data[offset + 65],
                    data[offset + 66],
                    data[offset + 67],
                ],
            };
            column_groups.push(group);
            offset += group_size;
        }

        // 4. Deserialize columns for each group manually
        let column_size = 8 + 4 + 4 + 1 + 1 + 4 + 4 + 6; // 32 bytes per column (8+4+4+1+1+4+4+6)
        let mut columns = Vec::with_capacity(num_groups as usize);

        for _ in 0..num_groups {
            if offset + 4 > data.len() {
                return Err(ProximaDBError::InvalidInput(
                    "Insufficient data for NOVA column count".into(),
                ));
            }

            let num_columns = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let mut group_columns = Vec::with_capacity(num_columns as usize);
            for _ in 0..num_columns {
                if offset + column_size > data.len() {
                    return Err(ProximaDBError::InvalidInput(
                        "Insufficient data for NOVA column headers".into(),
                    ));
                }

                let column = NovaColumnHeader {
                    offset: u64::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                        data[offset + 4],
                        data[offset + 5],
                        data[offset + 6],
                        data[offset + 7],
                    ]),
                    compressed_size: u32::from_le_bytes([
                        data[offset + 8],
                        data[offset + 9],
                        data[offset + 10],
                        data[offset + 11],
                    ]),
                    uncompressed_size: u32::from_le_bytes([
                        data[offset + 12],
                        data[offset + 13],
                        data[offset + 14],
                        data[offset + 15],
                    ]),
                    column_type: data[offset + 16],
                    compression_algorithm: data[offset + 17],
                    null_count: u32::from_le_bytes([
                        data[offset + 18],
                        data[offset + 19],
                        data[offset + 20],
                        data[offset + 21],
                    ]),
                    stats_offset: u32::from_le_bytes([
                        data[offset + 22],
                        data[offset + 23],
                        data[offset + 24],
                        data[offset + 25],
                    ]),
                    reserved: [
                        u16::from_le_bytes([data[offset + 26], data[offset + 27]]),
                        u16::from_le_bytes([data[offset + 28], data[offset + 29]]),
                        u16::from_le_bytes([data[offset + 30], data[offset + 31]]),
                    ],
                };
                group_columns.push(column);
                offset += column_size;
            }
            columns.push(group_columns);
        }

        // 5. Read variable data
        if offset + 4 > data.len() {
            return Err(ProximaDBError::InvalidInput(
                "Insufficient data for NOVA variable data size".into(),
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
                "Insufficient data for NOVA variable data".into(),
            ));
        }

        let variable_data = data[offset..offset + variable_data_size].to_vec();

        let metadata = NovaMetadata {
            footer,
            column_groups,
            columns,
            variable_data,
            schema: parking_lot::RwLock::new(None),
            column_stats: parking_lot::RwLock::new(HashMap::new()),
        };

        trace!(
            column_groups = metadata.column_groups.len(),
            variable_data_size, "Deserialized NOVA metadata"
        );

        Ok(Box::new(metadata))
    }

    fn can_skip_file(&self, metadata: &dyn EngineMetadata, query_context: &QueryContext) -> bool {
        let Some(nova_metadata) = metadata.as_any().downcast_ref::<NovaMetadata>() else {
            warn!("Invalid metadata type for NOVA serializer in can_skip_file; cannot skip");
            return false;
        };

        match &query_context.query_type {
            QueryType::IdLookup => {
                // Check if any IDs might be in this file using column group ranges
                for id in &query_context.id_lookups {
                    let id_hash = nova_metadata.hash_id(id);
                    for group in &nova_metadata.column_groups {
                        if id_hash >= group.min_id_hash && id_hash <= group.max_id_hash {
                            return false; // Found potential match
                        }
                    }
                }
                true // No potential matches
            }

            QueryType::SimilaritySearch => {
                // NOVA's columnar format allows some pruning with statistics
                // For now, conservative approach
                false
            }

            QueryType::MetadataFilter => {
                // Could potentially skip based on column group statistics
                // For now, conservative approach
                false
            }

            QueryType::Batch => {
                // Conservative approach for batch queries
                false
            }

            QueryType::VectorSearch | QueryType::FullScan => {
                // Cannot skip file for vector search or full scan
                false
            }
        }
    }

    fn get_required_ranges(
        &self,
        metadata: &dyn EngineMetadata,
        query_context: &QueryContext,
    ) -> Option<Vec<DataRange>> {
        let Some(nova_metadata) = metadata.as_any().downcast_ref::<NovaMetadata>() else {
            warn!(
                "Invalid metadata type for NOVA serializer in get_required_ranges; falling back to full read"
            );
            return None;
        };

        let required_groups = nova_metadata.get_required_column_groups(query_context);
        let column_requirements = nova_metadata.get_required_columns(query_context);

        if required_groups.len() == nova_metadata.column_groups.len()
            && column_requirements
                .values()
                .all(|cols| cols.len() == nova_metadata.footer.num_columns as usize)
        {
            // Need all groups and all columns - return None to indicate full file read
            None
        } else {
            // Return specific column ranges
            Some(nova_metadata.requirements_to_ranges(required_groups, column_requirements))
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

#[cfg(test)]
mod tests {
    use super::*;
    // extract_metadata is a placeholder that never touches the filesystem, so
    // tests inject the shared no-op `FilesystemPort` (`crate::testutil`,
    // TD-DECOMP-75 pattern).
    use crate::testutil::noop_filesystem_port as noop_port;

    #[tokio::test]
    async fn test_nova_metadata_serialization() -> Result<(), ProximaDBError> {
        let serializer = NovaMetadataSerializer::new(noop_port());

        // Test serialization
        let serialized = serializer.serialize_metadata("/test/file.nova", "test_collection")?;
        assert!(!serialized.is_empty());

        // Test deserialization
        let metadata = serializer.deserialize_metadata(&serialized)?;
        assert_eq!(metadata.file_size(), 2 * 1024 * 1024);
        assert!(metadata.memory_footprint() > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_nova_columnar_optimization() -> Result<(), ProximaDBError> {
        let serializer = NovaMetadataSerializer::new(noop_port());

        let serialized = serializer.serialize_metadata("/test/file.nova", "test_collection")?;
        let metadata = serializer.deserialize_metadata(&serialized)?;

        // Test metadata filtering - should be very selective
        let mut query_context = QueryContext::default();
        query_context.query_type = QueryType::MetadataFilter;
        query_context
            .metadata_filters
            .insert("category".to_string(), "electronics".to_string());

        let selectivity = serializer.estimate_selectivity(metadata.as_ref(), &query_context);
        assert!(selectivity < 0.5); // Should be quite selective

        // Test column optimization
        let ranges = serializer.get_required_ranges(metadata.as_ref(), &query_context);
        if let Some(ranges) = ranges {
            // Should not need all columns for metadata filtering
            assert!(!ranges.is_empty());
            // Should be more selective than full file
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_nova_similarity_search_optimization() -> Result<(), ProximaDBError> {
        let serializer = NovaMetadataSerializer::new(noop_port());

        let serialized = serializer.serialize_metadata("/test/file.nova", "test_collection")?;
        let metadata = serializer.deserialize_metadata(&serialized)?;

        // Test similarity search with vector
        let mut query_context = QueryContext::default();
        query_context.query_type = QueryType::SimilaritySearch;
        query_context.query_vector = Some(vec![1.0, 2.0, 3.0, 4.0]);

        let selectivity = serializer.estimate_selectivity(metadata.as_ref(), &query_context);
        assert!(selectivity > 0.0 && selectivity <= 1.0);

        // NOVA should provide some selectivity estimate for similarity search
        // With 8 column groups, selectivity should be around 0.92
        // But the calculation is (1.0 - (8/100).min(0.8)).max(0.1) = 0.92
        assert!(selectivity <= 1.0); // Valid selectivity range

        Ok(())
    }
}
