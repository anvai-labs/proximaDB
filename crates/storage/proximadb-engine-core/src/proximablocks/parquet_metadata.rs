// Parquet Metadata Serializer for Zero-Copy Cache
// Hierarchical structure: Footer → Row Group Info → Column Info
// Enables filtering at row group level without reading column data

use std::collections::HashMap;
use std::sync::Arc;

use proximadb_storage_common::zero_copy_traits::{
    DataRange, EngineMetadata, MetadataSerializer, QueryContext,
};
// FilesystemPort seam (TD-DECOMP-75 pattern): root FilesystemFactory implements it.
use proximadb_kernel::error::ProximaDBError;
use proximadb_storage_ports::FilesystemPort;

use super::utilities::stable_id_hash;

/// Parquet Footer metadata (fixed size, bytemuck-compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParquetFooterHeader {
    /// Total file size
    pub file_size: u64,

    /// Number of row groups
    pub num_row_groups: u32,

    /// Footer offset in metadata
    pub footer_offset: u32,

    /// Footer size
    pub footer_size: u32,

    /// Schema offset in metadata
    pub schema_offset: u32,

    /// Schema size
    pub schema_size: u32,

    /// Total number of rows
    pub total_rows: u64,

    /// Number of columns
    pub num_columns: u32,

    /// File version
    pub parquet_version: u32,

    /// Created timestamp
    pub created_timestamp: u64,

    /// File compression used
    pub compression_type: u8,

    /// Reserved for future use
    pub reserved: [u8; 7],
}

/// Parquet Row Group metadata (fixed size, array-based)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParquetRowGroupHeader {
    /// Row group offset in file
    pub offset: u64,

    /// Row group total byte size (compressed)
    pub compressed_size: u64,

    /// Row group total byte size (uncompressed)
    pub uncompressed_size: u64,

    /// Number of rows in this row group
    pub num_rows: u64,

    /// Number of columns in this row group
    pub num_columns: u32,

    /// Statistics offset (relative to start of variable data)
    pub statistics_offset: u32,

    /// Statistics size
    pub statistics_size: u32,

    /// Minimum key value hash (for range filtering)
    pub min_key_hash: u64,

    /// Maximum key value hash (for range filtering)
    pub max_key_hash: u64,

    /// Row group priority (0=cold, 255=hot)
    pub priority: u8,

    /// Reserved
    pub reserved: [u8; 7],
}

/// Parquet Column Chunk metadata (fixed size)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParquetColumnHeader {
    /// Column chunk offset in file
    pub offset: u64,

    /// Column chunk size (compressed)
    pub compressed_size: u32,

    /// Column chunk size (uncompressed)
    pub uncompressed_size: u32,

    /// Data type encoding (INT32, BYTE_ARRAY, etc.)
    pub data_type: u32,

    /// Encoding type (PLAIN, RLE, etc.)
    pub encoding: u32,

    /// Compression type for this column
    pub compression: u32,

    /// Number of values in column
    pub num_values: u64,

    /// Number of null values
    pub null_count: u64,

    /// Distinct count (if available)
    pub distinct_count: u64,

    /// Min value hash (for range queries)
    pub min_value_hash: u64,

    /// Max value hash (for range queries)
    pub max_value_hash: u64,

    /// Reserved
    pub reserved: [u32; 2],
}

/// Complete Parquet metadata structure
pub struct ParquetMetadata {
    /// Fixed-size footer header
    pub footer: ParquetFooterHeader,

    /// Fixed-size row group headers (array)
    pub row_groups: Vec<ParquetRowGroupHeader>,

    /// Column headers for each row group
    pub columns: Vec<Vec<ParquetColumnHeader>>,

    /// Variable-size data (schema, statistics, column indexes)
    pub variable_data: Vec<u8>,

    /// Parsed schema (lazy-loaded)
    #[allow(dead_code)]
    schema: parking_lot::RwLock<Option<Arc<Vec<u8>>>>,

    /// Parsed column statistics (lazy-loaded by row group)
    #[allow(dead_code)]
    column_stats: parking_lot::RwLock<HashMap<u32, Arc<Vec<u8>>>>,
}

/// Parquet metadata serializer implementation
pub struct ParquetMetadataSerializer {
    /// Filesystem for reading files
    filesystem: Arc<dyn FilesystemPort>,
}

impl ParquetMetadataSerializer {
    pub fn new(filesystem: Arc<dyn FilesystemPort>) -> Self {
        Self { filesystem }
    }

    /// Parse Parquet file and extract metadata
    async fn extract_metadata(&self, file_path: &str) -> Result<ParquetMetadata, ProximaDBError> {
        let fs = self.filesystem.get_filesystem(file_path)?;

        // Read file size
        let metadata = fs.metadata(file_path).await?;
        let file_size = metadata.size;

        // Read footer from the end of file (Parquet format stores footer at end)
        let footer_size_bytes = 4; // Last 4 bytes contain footer size
        let footer_size_data = fs
            .read_range(file_path, file_size - footer_size_bytes, footer_size_bytes)
            .await?;

        let footer_size = u32::from_le_bytes([
            footer_size_data[0],
            footer_size_data[1],
            footer_size_data[2],
            footer_size_data[3],
        ]) as u64;

        // Read the actual footer
        let footer_start = file_size - footer_size - footer_size_bytes;
        let footer_data = fs.read_range(file_path, footer_start, footer_size).await?;

        // Parse footer to get row group and schema information
        let (row_group_info, schema_info) = self.parse_parquet_footer(&footer_data)?;

        // Build row group headers
        let mut row_groups = Vec::new();
        let mut columns = Vec::new();
        let mut variable_data = Vec::new();

        // Add schema to variable data
        let schema_offset = variable_data.len();
        variable_data.extend_from_slice(&schema_info);

        // Process each row group
        for (rg_idx, rg_info) in row_group_info.iter().enumerate() {
            // Add statistics to variable data
            let stats_offset = variable_data.len();
            let stats_data = self.extract_row_group_statistics(rg_info)?;
            variable_data.extend_from_slice(&stats_data);

            // Create row group header
            let rg_header = ParquetRowGroupHeader {
                offset: rg_info.offset,
                compressed_size: rg_info.compressed_size,
                uncompressed_size: rg_info.uncompressed_size,
                num_rows: rg_info.num_rows,
                num_columns: rg_info.columns.len() as u32,
                statistics_offset: stats_offset as u32,
                statistics_size: stats_data.len() as u32,
                min_key_hash: rg_info.min_key_hash,
                max_key_hash: rg_info.max_key_hash,
                priority: if rg_idx < 3 { 255 } else { 128 }, // First few row groups are hot
                reserved: [0; 7],
            };

            row_groups.push(rg_header);

            // Process columns in this row group
            let mut rg_columns = Vec::new();
            for col_info in &rg_info.columns {
                let col_header = ParquetColumnHeader {
                    offset: col_info.offset,
                    compressed_size: col_info.compressed_size,
                    uncompressed_size: col_info.uncompressed_size,
                    data_type: col_info.data_type,
                    encoding: col_info.encoding,
                    compression: col_info.compression,
                    num_values: col_info.num_values,
                    null_count: col_info.null_count,
                    distinct_count: col_info.distinct_count,
                    min_value_hash: col_info.min_value_hash,
                    max_value_hash: col_info.max_value_hash,
                    reserved: [0; 2],
                };
                rg_columns.push(col_header);
            }
            columns.push(rg_columns);
        }

        // Create footer header
        let footer = ParquetFooterHeader {
            file_size,
            num_row_groups: row_groups.len() as u32,
            footer_offset: 0, // Will be updated when serialized
            footer_size: footer_data.len() as u32,
            schema_offset: schema_offset as u32,
            schema_size: schema_info.len() as u32,
            total_rows: row_groups.iter().map(|rg| rg.num_rows).sum(),
            num_columns: if !row_groups.is_empty() {
                row_groups[0].num_columns
            } else {
                0
            },
            parquet_version: 2, // Parquet version 2.x
            created_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0),
            compression_type: 1, // SNAPPY as default
            reserved: [0; 7],
        };

        Ok(ParquetMetadata {
            footer,
            row_groups,
            columns,
            variable_data,
            schema: parking_lot::RwLock::new(None),
            column_stats: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    /// Parse Parquet footer to extract row group and schema information
    fn parse_parquet_footer(
        &self,
        _footer_data: &[u8],
    ) -> Result<(Vec<RowGroupInfo>, Vec<u8>), ProximaDBError> {
        // Deferred: Implement actual Parquet footer parsing
        // This requires reading the actual Parquet file metadata structure
        // Should use the parquet crate's FileMetaData parsing

        // For now, return error indicating this needs implementation
        Err(ProximaDBError::Internal(
            "Parquet footer parsing needs real implementation using parquet crate".to_string(),
        ))
    }

    /// Extract row group statistics
    fn extract_row_group_statistics(
        &self,
        _rg_info: &RowGroupInfo,
    ) -> Result<Vec<u8>, ProximaDBError> {
        // Extract column statistics for row group
        // This would parse actual Parquet statistics in real implementation

        // For demo, create placeholder statistics
        Ok(b"row_group_statistics_placeholder".to_vec())
    }
}

/// Temporary row group info for parsing
#[derive(Debug)]
struct RowGroupInfo {
    offset: u64,
    compressed_size: u64,
    uncompressed_size: u64,
    num_rows: u64,
    columns: Vec<ColumnInfo>,
    min_key_hash: u64,
    max_key_hash: u64,
}

/// Temporary column info for parsing
#[derive(Debug)]
struct ColumnInfo {
    offset: u64,
    compressed_size: u32,
    uncompressed_size: u32,
    data_type: u32,
    encoding: u32,
    compression: u32,
    num_values: u64,
    null_count: u64,
    distinct_count: u64,
    min_value_hash: u64,
    max_value_hash: u64,
}

impl MetadataSerializer for ParquetMetadataSerializer {
    fn engine_id(&self) -> &'static str {
        "PARQUET"
    }

    fn serialize_metadata(
        &self,
        file_path: &str,
        _collection_id: &str,
    ) -> Result<Vec<u8>, ProximaDBError> {
        // This would be async in practice, but trait doesn't support async yet
        let metadata = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.extract_metadata(file_path))
        })?;

        let mut buffer = Vec::new();

        // Serialize footer header (fixed size)
        buffer.extend_from_slice(&metadata.footer.file_size.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.num_row_groups.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.footer_offset.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.footer_size.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.schema_offset.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.schema_size.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.total_rows.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.num_columns.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.parquet_version.to_le_bytes());
        buffer.extend_from_slice(&metadata.footer.created_timestamp.to_le_bytes());
        buffer.extend_from_slice(&[metadata.footer.compression_type]);
        buffer.extend_from_slice(&metadata.footer.reserved);

        // Serialize row group count
        buffer.extend_from_slice(&(metadata.row_groups.len() as u32).to_le_bytes());

        // Serialize row group headers (fixed size array)
        for rg in &metadata.row_groups {
            buffer.extend_from_slice(&rg.offset.to_le_bytes());
            buffer.extend_from_slice(&rg.compressed_size.to_le_bytes());
            buffer.extend_from_slice(&rg.uncompressed_size.to_le_bytes());
            buffer.extend_from_slice(&rg.num_rows.to_le_bytes());
            buffer.extend_from_slice(&rg.num_columns.to_le_bytes());
            buffer.extend_from_slice(&rg.statistics_offset.to_le_bytes());
            buffer.extend_from_slice(&rg.statistics_size.to_le_bytes());
            buffer.extend_from_slice(&rg.min_key_hash.to_le_bytes());
            buffer.extend_from_slice(&rg.max_key_hash.to_le_bytes());
            buffer.extend_from_slice(&[rg.priority]);
            buffer.extend_from_slice(&rg.reserved);
        }

        // Serialize column headers for each row group
        for rg_columns in &metadata.columns {
            buffer.extend_from_slice(&(rg_columns.len() as u32).to_le_bytes());
            for col in rg_columns {
                buffer.extend_from_slice(&col.offset.to_le_bytes());
                buffer.extend_from_slice(&col.compressed_size.to_le_bytes());
                buffer.extend_from_slice(&col.uncompressed_size.to_le_bytes());
                buffer.extend_from_slice(&col.num_values.to_le_bytes());
                buffer.extend_from_slice(&col.null_count.to_le_bytes());
                buffer.extend_from_slice(&col.distinct_count.to_le_bytes());
                buffer.extend_from_slice(&col.min_value_hash.to_le_bytes());
                buffer.extend_from_slice(&col.max_value_hash.to_le_bytes());
                buffer.extend_from_slice(&col.data_type.to_le_bytes());
                buffer.extend_from_slice(&col.encoding.to_le_bytes());
                buffer.extend_from_slice(&col.compression.to_le_bytes());
                for &r in &col.reserved {
                    buffer.extend_from_slice(&r.to_le_bytes());
                }
            }
        }

        // Serialize variable data size
        buffer.extend_from_slice(&(metadata.variable_data.len()).to_le_bytes());

        // Serialize variable data (compressed with bincode if enabled)
        buffer.extend_from_slice(&metadata.variable_data);

        Ok(buffer)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<Box<dyn EngineMetadata>, ProximaDBError> {
        let mut offset = 0;

        // Deserialize footer header manually
        let footer_size = 8 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + 8 + 1 + 7; // 64 bytes total
        if data.len() < offset + footer_size {
            return Err(ProximaDBError::InvalidInput(
                "Invalid Parquet metadata size".into(),
            ));
        }

        let footer = ParquetFooterHeader {
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
            num_row_groups: u32::from_le_bytes([
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
            total_rows: u64::from_le_bytes([
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
            parquet_version: u32::from_le_bytes([
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

        // Deserialize row group count
        if data.len() < offset + 4 {
            return Err(ProximaDBError::InvalidInput(
                "Invalid Parquet metadata".into(),
            ));
        }
        let rg_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Deserialize row group headers manually
        let rg_header_size = 8 + 8 + 8 + 8 + 4 + 4 + 4 + 8 + 8 + 1 + 7; // 72 bytes per header
        let total_rg_size = rg_count as usize * rg_header_size;

        if data.len() < offset + total_rg_size {
            return Err(ProximaDBError::InvalidInput(
                "Invalid Parquet row group headers".into(),
            ));
        }

        let mut row_groups = Vec::new();
        for _ in 0..rg_count {
            let rg = ParquetRowGroupHeader {
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
                num_rows: u64::from_le_bytes([
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
                min_key_hash: u64::from_le_bytes([
                    data[offset + 44],
                    data[offset + 45],
                    data[offset + 46],
                    data[offset + 47],
                    data[offset + 48],
                    data[offset + 49],
                    data[offset + 50],
                    data[offset + 51],
                ]),
                max_key_hash: u64::from_le_bytes([
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
            row_groups.push(rg);
            offset += rg_header_size;
        }

        // Deserialize column headers for each row group
        let mut columns = Vec::new();
        for _ in 0..rg_count {
            if data.len() < offset + 4 {
                return Err(ProximaDBError::InvalidInput("Invalid column count".into()));
            }
            let col_count = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let col_header_size = 8 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 8 + 8 + 8 + 8; // 76 bytes per column header
            let total_col_size = col_count as usize * col_header_size;

            if data.len() < offset + total_col_size {
                return Err(ProximaDBError::InvalidInput(
                    "Invalid Parquet column headers".into(),
                ));
            }

            let mut rg_columns = Vec::new();
            for _ in 0..col_count {
                let col = ParquetColumnHeader {
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
                    data_type: u32::from_le_bytes([
                        data[offset + 16],
                        data[offset + 17],
                        data[offset + 18],
                        data[offset + 19],
                    ]),
                    encoding: u32::from_le_bytes([
                        data[offset + 20],
                        data[offset + 21],
                        data[offset + 22],
                        data[offset + 23],
                    ]),
                    compression: u32::from_le_bytes([
                        data[offset + 24],
                        data[offset + 25],
                        data[offset + 26],
                        data[offset + 27],
                    ]),
                    num_values: u64::from_le_bytes([
                        data[offset + 28],
                        data[offset + 29],
                        data[offset + 30],
                        data[offset + 31],
                        data[offset + 32],
                        data[offset + 33],
                        data[offset + 34],
                        data[offset + 35],
                    ]),
                    null_count: u64::from_le_bytes([
                        data[offset + 36],
                        data[offset + 37],
                        data[offset + 38],
                        data[offset + 39],
                        data[offset + 40],
                        data[offset + 41],
                        data[offset + 42],
                        data[offset + 43],
                    ]),
                    distinct_count: u64::from_le_bytes([
                        data[offset + 44],
                        data[offset + 45],
                        data[offset + 46],
                        data[offset + 47],
                        data[offset + 48],
                        data[offset + 49],
                        data[offset + 50],
                        data[offset + 51],
                    ]),
                    min_value_hash: u64::from_le_bytes([
                        data[offset + 52],
                        data[offset + 53],
                        data[offset + 54],
                        data[offset + 55],
                        data[offset + 56],
                        data[offset + 57],
                        data[offset + 58],
                        data[offset + 59],
                    ]),
                    max_value_hash: u64::from_le_bytes([
                        data[offset + 60],
                        data[offset + 61],
                        data[offset + 62],
                        data[offset + 63],
                        data[offset + 64],
                        data[offset + 65],
                        data[offset + 66],
                        data[offset + 67],
                    ]),
                    reserved: [
                        u32::from_le_bytes([
                            data[offset + 68],
                            data[offset + 69],
                            data[offset + 70],
                            data[offset + 71],
                        ]),
                        u32::from_le_bytes([
                            data[offset + 72],
                            data[offset + 73],
                            data[offset + 74],
                            data[offset + 75],
                        ]),
                    ],
                };
                rg_columns.push(col);
                offset += col_header_size;
            }
            columns.push(rg_columns);
        }

        // Deserialize variable data size
        if data.len() < offset + 4 {
            return Err(ProximaDBError::InvalidInput(
                "Invalid Parquet variable data size".into(),
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
                "Invalid Parquet variable data".into(),
            ));
        }
        let variable_data = data[offset..offset + var_data_size as usize].to_vec();

        let metadata = ParquetMetadata {
            footer,
            row_groups,
            columns,
            variable_data,
            schema: parking_lot::RwLock::new(None),
            column_stats: parking_lot::RwLock::new(HashMap::new()),
        };

        Ok(Box::new(metadata))
    }

    fn can_skip_file(&self, metadata: &dyn EngineMetadata, query_context: &QueryContext) -> bool {
        let Some(parquet_metadata) = metadata.as_any().downcast_ref::<ParquetMetadata>() else {
            return false;
        };

        // Check if we can skip entire file based on row group statistics
        if !query_context.id_lookups.is_empty() {
            // Check if any row group might contain the requested IDs
            let mut any_might_exist = false;

            for id in &query_context.id_lookups {
                let id_hash = self.hash_string(id);

                for rg in &parquet_metadata.row_groups {
                    if id_hash >= rg.min_key_hash && id_hash <= rg.max_key_hash {
                        any_might_exist = true;
                        break;
                    }
                }

                if any_might_exist {
                    break;
                }
            }

            if !any_might_exist {
                return true; // None of the IDs exist in any row group
            }
        }

        // Check metadata filters
        if let Some(ttl_threshold) = query_context.metadata_filters.get("ttl_threshold")
            && let Ok(threshold) = ttl_threshold.parse::<u64>()
            && parquet_metadata.footer.created_timestamp < threshold
        {
            return true; // Entire file is expired
        }

        // Check vector similarity threshold with estimated selectivity
        if let Some(threshold) = query_context.distance_threshold {
            let estimated_selectivity = metadata.estimated_selectivity(query_context);
            if estimated_selectivity < 0.001 && threshold > 0.9 {
                return true; // Very low selectivity with high threshold
            }
        }

        false // Can't skip, need to read some row groups
    }

    fn get_required_ranges(
        &self,
        metadata: &dyn EngineMetadata,
        query_context: &QueryContext,
    ) -> Option<Vec<DataRange>> {
        let parquet_metadata = metadata.as_any().downcast_ref::<ParquetMetadata>()?;

        // For vector similarity search, typically need entire file
        if query_context.query_vector.is_some() && query_context.id_lookups.is_empty() {
            return None; // Read entire file for similarity search
        }

        // For ID lookups, determine which row groups to read
        if !query_context.id_lookups.is_empty() {
            let mut required_ranges = Vec::new();

            // Always include footer for metadata
            required_ranges.push(DataRange {
                offset: parquet_metadata.footer.file_size
                    - parquet_metadata.footer.footer_size as u64
                    - 4,
                length: parquet_metadata.footer.footer_size as u64 + 4,
                priority: 255, // Critical
            });

            // Check which row groups contain the requested IDs
            for (rg_idx, rg) in parquet_metadata.row_groups.iter().enumerate() {
                let mut need_row_group = false;

                for id in &query_context.id_lookups {
                    let id_hash = self.hash_string(id);
                    if id_hash >= rg.min_key_hash && id_hash <= rg.max_key_hash {
                        need_row_group = true;
                        break;
                    }
                }

                if need_row_group {
                    // For columnar format, we might only need specific columns
                    if let Some(columns) = query_context.metadata_filters.get("required_columns") {
                        // Parse required columns and add only those column chunks
                        let col_names: Vec<&str> = columns.split(',').collect();

                        if let Some(rg_columns) = parquet_metadata.columns.get(rg_idx) {
                            for (col_idx, col) in rg_columns.iter().enumerate() {
                                if col_idx < col_names.len() {
                                    required_ranges.push(DataRange {
                                        offset: col.offset,
                                        length: col.compressed_size as u64,
                                        priority: rg.priority,
                                    });
                                }
                            }
                        }
                    } else {
                        // Read entire row group
                        required_ranges.push(DataRange {
                            offset: rg.offset,
                            length: rg.compressed_size,
                            priority: rg.priority,
                        });
                    }
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
        stable_id_hash(s)
    }
}

impl EngineMetadata for ParquetMetadata {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EngineMetadata> {
        Box::new(self.clone())
    }

    fn file_size(&self) -> u64 {
        self.footer.file_size
    }

    fn estimated_selectivity(&self, query_context: &QueryContext) -> f32 {
        // Estimate selectivity based on query type
        if !query_context.id_lookups.is_empty() {
            // ID lookups are very selective
            let total_rows = self.footer.total_rows as f32;
            let requested_ids = query_context.id_lookups.len() as f32;
            (requested_ids / total_rows).min(1.0)
        } else if query_context.query_vector.is_some() {
            // Vector similarity depends on top_k
            if let Some(top_k) = query_context.top_k {
                let total_rows = self.footer.total_rows as f32;
                (top_k as f32 / total_rows).min(1.0)
            } else {
                0.1 // Default 10% selectivity for similarity
            }
        } else {
            1.0 // Scan everything
        }
    }

    fn memory_footprint(&self) -> usize {
        std::mem::size_of::<ParquetFooterHeader>()
            + self.row_groups.len() * std::mem::size_of::<ParquetRowGroupHeader>()
            + self
                .columns
                .iter()
                .map(|rg_cols| rg_cols.len() * std::mem::size_of::<ParquetColumnHeader>())
                .sum::<usize>()
            + self.variable_data.len()
    }
}

// Enable downcasting for the metadata
impl ParquetMetadata {
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Implement Clone for ParquetMetadata
impl Clone for ParquetMetadata {
    fn clone(&self) -> Self {
        Self {
            footer: self.footer,
            row_groups: self.row_groups.clone(),
            columns: self.columns.clone(),
            variable_data: self.variable_data.clone(),
            schema: parking_lot::RwLock::new(None), // Reset lazy-loaded data
            column_stats: parking_lot::RwLock::new(HashMap::new()),
        }
    }
}
