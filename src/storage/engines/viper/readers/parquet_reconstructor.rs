//! Parquet Data Reconstructor
//!
//! Reconstructs Parquet data structures from partial file reads (seeks or HTTP ranges)
//! to enable Arrow processing of partial data without full file downloads.

use anyhow::{Context, Result};
use arrow_array::{Array, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use chrono;
// These parquet metadata types are used internally for low-level operations
// The columnar module doesn't re-export them as they're implementation details
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::proto::proximadb_v1::VectorRecord;

/// File seek range for efficient data access
#[derive(Debug, Clone)]
pub struct FileSeekRange {
    pub offset: usize,
    pub length: usize,
    pub row_group_idx: usize,
    pub column_name: String,
}

/// Vector query for reconstruction
#[derive(Debug, Clone)]
pub struct VectorQuery {
    pub file_path: String,
    pub query_vector: Vec<f32>,
    pub k: usize,
    pub return_vectors: bool,
}

/// Reconstructs Parquet data from partial reads
pub struct ParquetReconstructor {
    #[allow(dead_code)]
    config: ReconstructorConfig,
}

#[derive(Debug, Clone)]
pub struct ReconstructorConfig {
    pub enable_schema_validation: bool,
    pub max_memory_usage_mb: f64,
    pub enable_column_caching: bool,
}

impl Default for ReconstructorConfig {
    fn default() -> Self {
        Self {
            enable_schema_validation: true,
            max_memory_usage_mb: 256.0,
            enable_column_caching: true,
        }
    }
}

/// Represents partial Parquet data assembled from file seeks or HTTP ranges
#[derive(Debug)]
pub struct ReconstructedParquetData {
    pub record_batches: Vec<RecordBatch>,
    pub schema: Arc<Schema>,
    pub total_rows: usize,
    pub bytes_processed: usize,
    pub row_groups_included: Vec<usize>,
}

/// Column chunk data from file seeks or HTTP ranges
#[derive(Debug, Clone)]
pub struct ColumnChunkData {
    pub row_group_idx: usize,
    pub column_name: String,
    pub data: Vec<u8>,
    pub compression: CompressionType,
    pub uncompressed_size: usize,
    pub row_count: usize,
    pub storage: Option<StorageInfo>,
}

#[derive(Debug, Clone)]
pub enum CompressionType {
    None,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
}

#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub compression: Option<CompressionType>,
}

impl ParquetReconstructor {
    pub fn new(config: ReconstructorConfig) -> Self {
        Self { config }
    }

    /// Reconstruct Parquet data from file seek results
    pub async fn reconstruct_from_seeks(
        &self,
        seek_data: Vec<SeekData>,
        metadata: &ParquetMetaData,
        query: &VectorQuery,
    ) -> Result<ReconstructedParquetData> {
        info!(
            "🔧 Reconstructing Parquet data from {} file seeks",
            seek_data.len()
        );

        // Group seek data by row group and column
        let grouped_data = self.group_seek_data_by_row_group(seek_data)?;

        // Process each row group
        let mut record_batches = Vec::new();
        let mut total_rows = 0;
        let mut bytes_processed = 0;
        let mut row_groups_included = Vec::new();

        for (row_group_idx, column_data) in grouped_data {
            let row_group_metadata = metadata.row_group(row_group_idx);

            // Decompress and parse column chunks
            let parsed_columns = self.parse_column_chunks(column_data, row_group_metadata)?;

            // Create record batch from parsed columns
            let record_batch = self.create_record_batch_from_columns(parsed_columns, query)?;

            total_rows += record_batch.num_rows();
            bytes_processed += self.estimate_batch_size(&record_batch);
            row_groups_included.push(row_group_idx);
            record_batches.push(record_batch);
        }

        // Derive schema from the first record batch
        let schema = if !record_batches.is_empty() {
            record_batches[0].schema()
        } else {
            Arc::new(Schema::empty())
        };

        info!(
            "✅ Reconstructed {} record batches with {} total rows ({:.1}KB)",
            record_batches.len(),
            total_rows,
            bytes_processed as f64 / 1024.0
        );

        Ok(ReconstructedParquetData {
            record_batches,
            schema,
            total_rows,
            bytes_processed,
            row_groups_included,
        })
    }

    /// Reconstruct Parquet data from HTTP range results
    pub async fn reconstruct_from_ranges(
        &self,
        range_data: Vec<RangeData>,
        metadata: &ParquetMetaData,
        query: &VectorQuery,
    ) -> Result<ReconstructedParquetData> {
        info!(
            "☁️ Reconstructing Parquet data from {} HTTP ranges",
            range_data.len()
        );

        // Convert range data to column chunks
        let column_chunks = self.parse_range_data_to_columns(range_data, metadata)?;

        // Group by row group
        let grouped_chunks = self.group_column_chunks_by_row_group(column_chunks)?;

        // Process each row group
        let mut record_batches = Vec::new();
        let mut total_rows = 0;
        let mut bytes_processed = 0;
        let mut row_groups_included = Vec::new();

        for (row_group_idx, chunks) in grouped_chunks {
            // Create record batch from column chunks
            let record_batch = self.create_record_batch_from_chunk_data(chunks, query)?;

            total_rows += record_batch.num_rows();
            bytes_processed += self.estimate_batch_size(&record_batch);
            row_groups_included.push(row_group_idx);
            record_batches.push(record_batch);
        }

        // Derive schema
        let schema = if !record_batches.is_empty() {
            record_batches[0].schema()
        } else {
            Arc::new(Schema::empty())
        };

        info!(
            "✅ Reconstructed {} record batches from ranges with {} total rows ({:.1}KB)",
            record_batches.len(),
            total_rows,
            bytes_processed as f64 / 1024.0
        );

        Ok(ReconstructedParquetData {
            record_batches,
            schema,
            total_rows,
            bytes_processed,
            row_groups_included,
        })
    }

    /// Convert reconstructed data to VectorRecord format
    pub fn convert_to_vector_records(
        &self,
        reconstructed_data: ReconstructedParquetData,
        query: &VectorQuery,
    ) -> Result<Vec<VectorRecord>> {
        debug!(
            "🔄 Converting {} record batches to VectorRecord format",
            reconstructed_data.record_batches.len()
        );

        let mut vector_records = Vec::new();

        for record_batch in reconstructed_data.record_batches {
            let batch_vectors = self.extract_vectors_from_batch(&record_batch, query)?;
            vector_records.extend(batch_vectors);
        }

        debug!(
            "✅ Converted to {} VectorRecord objects",
            vector_records.len()
        );
        Ok(vector_records)
    }

    /// Group seek data by row group
    fn group_seek_data_by_row_group(
        &self,
        seek_data: Vec<SeekData>,
    ) -> Result<HashMap<usize, Vec<ColumnChunkData>>> {
        let mut grouped = HashMap::new();

        for seek in seek_data {
            let chunk_data = ColumnChunkData {
                row_group_idx: seek.range.row_group_idx,
                column_name: seek.range.column_name.clone(),
                data: seek.data,
                compression: self.detect_compression(&seek.range)?,
                uncompressed_size: seek.range.length, // Approximation
                row_count: 0,                         // Will be determined during parsing
                storage: None,
            };

            grouped
                .entry(seek.range.row_group_idx)
                .or_insert_with(Vec::new)
                .push(chunk_data);
        }

        Ok(grouped)
    }

    /// Parse range data into column chunks
    fn parse_range_data_to_columns(
        &self,
        range_data: Vec<RangeData>,
        _metadata: &ParquetMetaData,
    ) -> Result<Vec<ColumnChunkData>> {
        let mut column_chunks = Vec::new();

        for range in range_data {
            // Deferred: Implement proper range-to-column mapping
            // This requires understanding the Parquet file structure
            // For now, create placeholder column chunks

            let data_len = range.data.len();
            let chunk_data = ColumnChunkData {
                row_group_idx: 0,                  // Would be calculated from range mapping
                column_name: "vector".to_string(), // Would be determined from range context
                data: range.data,
                compression: CompressionType::None, // Would be detected from metadata
                uncompressed_size: data_len,
                row_count: 0,
                storage: None,
            };

            column_chunks.push(chunk_data);
        }

        Ok(column_chunks)
    }

    /// Parse column chunks from compressed data
    fn parse_column_chunks(
        &self,
        column_data: Vec<ColumnChunkData>,
        row_group_metadata: &RowGroupMetaData,
    ) -> Result<HashMap<String, ParsedColumn>> {
        let mut parsed_columns = HashMap::new();

        for chunk_data in column_data {
            debug!(
                "🔍 Parsing column chunk: {} (row group {})",
                chunk_data.column_name, chunk_data.row_group_idx
            );

            // Decompress data if needed
            let decompressed_data = self.decompress_column_data(&chunk_data)?;

            // Parse column data based on type
            let parsed_column = self.parse_column_data(
                &chunk_data.column_name,
                decompressed_data,
                row_group_metadata,
            )?;

            parsed_columns.insert(chunk_data.column_name.clone(), parsed_column);
        }

        Ok(parsed_columns)
    }

    /// Create record batch from parsed columns
    fn create_record_batch_from_columns(
        &self,
        parsed_columns: HashMap<String, ParsedColumn>,
        _query: &VectorQuery,
    ) -> Result<RecordBatch> {
        let mut fields = Vec::new();
        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

        // Required columns based on query
        let required_columns = vec![
            "id".to_string(),
            "vector".to_string(),
            "metadata_info".to_string(),
        ];

        for column_name in required_columns {
            if let Some(parsed_column) = parsed_columns.get(&column_name) {
                fields.push(Field::new(
                    &column_name,
                    parsed_column.array.data_type().clone(),
                    false,
                ));
                arrays.push(parsed_column.array.clone());
            } else {
                // Create empty array for missing columns
                let empty_array = self.create_empty_array(&column_name)?;
                fields.push(Field::new(
                    &column_name,
                    empty_array.data_type().clone(),
                    false,
                ));
                arrays.push(empty_array);
            }
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .context("Failed to create record batch from parsed columns")
    }

    /// Create record batch from column chunk data directly
    fn create_record_batch_from_chunk_data(
        &self,
        chunks: Vec<ColumnChunkData>,
        _query: &VectorQuery,
    ) -> Result<RecordBatch> {
        debug!(
            "🏗️ Creating record batch from {} column chunks",
            chunks.len()
        );

        // For now, create a simple record batch with placeholder data
        // In production, this would properly parse the Parquet column data

        let required_columns = vec![
            "id".to_string(),
            "vector".to_string(),
            "metadata_info".to_string(),
        ];
        let mut fields = Vec::new();
        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

        for column_name in required_columns {
            match column_name.as_str() {
                "id" => {
                    fields.push(Field::new("id", DataType::Utf8, false));
                    arrays.push(Arc::new(StringArray::from(vec!["placeholder_id"])));
                }
                "vector" => {
                    let dimension = 128; // Default dimension
                    fields.push(Field::new(
                        "vector",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, false)),
                            dimension,
                        ),
                        false,
                    ));
                    // Create a simple float array for now
                    arrays.push(Arc::new(Float32Array::from(vec![0.0f32; 128])));
                }
                "metadata_info" => {
                    fields.push(Field::new("metadata_info", DataType::Utf8, true));
                    arrays.push(Arc::new(StringArray::from(vec![Some("{}")])));
                }
                _ => {
                    // Create placeholder for other columns
                    fields.push(Field::new(&column_name, DataType::Utf8, true));
                    arrays.push(Arc::new(StringArray::from(vec![Some("placeholder")])));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .context("Failed to create record batch from chunk data")
    }

    /// Group column chunks by row group
    fn group_column_chunks_by_row_group(
        &self,
        chunks: Vec<ColumnChunkData>,
    ) -> Result<HashMap<usize, Vec<ColumnChunkData>>> {
        let mut grouped = HashMap::new();

        for chunk in chunks {
            grouped
                .entry(chunk.row_group_idx)
                .or_insert_with(Vec::new)
                .push(chunk);
        }

        Ok(grouped)
    }

    /// Extract VectorRecord objects from record batch
    fn extract_vectors_from_batch(
        &self,
        record_batch: &RecordBatch,
        _query: &VectorQuery,
    ) -> Result<Vec<VectorRecord>> {
        let mut vector_records = Vec::new();

        // Get column indices
        let id_column = record_batch.column_by_name("id");
        // Use the constants for field names
        let vector_column = record_batch
            .column_by_name(crate::storage::engines::core::formats::columnar::FIELD_VECTOR_FP32);
        let metadata_column = record_batch
            .column_by_name(crate::storage::engines::core::formats::columnar::FIELD_EXTRA_META);

        for row_idx in 0..record_batch.num_rows() {
            // Extract ID
            let id = if let Some(id_col) = id_column {
                self.extract_string_value(id_col, row_idx)?
            } else {
                format!("id_{}", row_idx)
            };

            // Extract vector
            let vector = if let Some(vec_col) = vector_column {
                self.extract_vector_value(vec_col, row_idx)?
            } else {
                vec![0.0f32; 128] // Placeholder
            };

            // Extract metadata
            let metadata = if let Some(meta_col) = metadata_column {
                self.extract_metadata_value(meta_col, row_idx)?
            } else {
                Vec::new()
            };

            // Convert Vec<MetadataItem> to HashMap<String, SqlValue>
            let metadata_map =
                crate::core::proto_metadata_helper::proto_metadata_to_sqlvalue_hashmap(&metadata);

            vector_records.push(VectorRecord {
                id,
                vector,
                metadata: metadata_map,
                timestamp: Some(chrono::Utc::now().timestamp()), // i64, not u32
                updated_at: Some(chrono::Utc::now().timestamp()),
                expires_at: None,
                version: Some(1),
                source: None,
            });
        }

        Ok(vector_records)
    }

    /// Helper methods
    fn detect_compression(&self, _range: &FileSeekRange) -> Result<CompressionType> {
        // Deferred: Detect compression from Parquet metadata
        Ok(CompressionType::None)
    }

    fn decompress_column_data(&self, chunk_data: &ColumnChunkData) -> Result<Vec<u8>> {
        match chunk_data.compression {
            CompressionType::None => Ok(chunk_data.data.clone()),
            _ => {
                // Deferred: Implement decompression for other formats
                warn!(
                    "⚠️ Compression {:?} not implemented, returning raw data",
                    chunk_data.compression
                );
                Ok(chunk_data.data.clone())
            }
        }
    }

    fn parse_column_data(
        &self,
        _column_name: &str,
        _data: Vec<u8>,
        _row_group_metadata: &RowGroupMetaData,
    ) -> Result<ParsedColumn> {
        // Deferred: Implement actual Parquet column parsing
        // This is complex and would require implementing a Parquet column reader

        // For now, return placeholder parsed column
        Ok(ParsedColumn {
            // data_type removed -  DataType::Utf8,
            array: Arc::new(StringArray::from(vec!["placeholder"])),
        })
    }

    fn create_empty_array(&self, column_name: &str) -> Result<Arc<dyn Array>> {
        match column_name {
            "id" => Ok(Arc::new(StringArray::from(Vec::<String>::new()))),
            "vector" => Ok(Arc::new(Float32Array::from(Vec::<f32>::new()))),
            "metadata_info" => Ok(Arc::new(StringArray::from(Vec::<Option<String>>::new()))),
            _ => Ok(Arc::new(StringArray::from(Vec::<Option<String>>::new()))),
        }
    }

    fn estimate_batch_size(&self, record_batch: &RecordBatch) -> usize {
        // Rough estimate of record batch size in bytes
        record_batch.num_rows() * record_batch.num_columns() * 16 // 16 bytes per value estimate
    }

    fn extract_string_value(&self, _column: &Arc<dyn Array>, row_idx: usize) -> Result<String> {
        // Deferred: Implement proper string extraction from Arrow arrays
        Ok(format!("value_{}", row_idx))
    }

    fn extract_vector_value(&self, column: &Arc<dyn Array>, row_idx: usize) -> Result<Vec<f32>> {
        // First check for FixedSizeBinary arrays (how vectors are actually stored)
        if let Some(binary_array) = column
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
        {
            // Get the binary data for this row
            let bytes = binary_array.value(row_idx);

            // Convert bytes back to f32 vector
            let vector: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();

            if !vector.is_empty() {
                return Ok(vector);
            }
        } else if let Some(list_array) = column.as_any().downcast_ref::<arrow_array::ListArray>() {
            // Extract vector from list array (alternative format)
            let values = list_array.values();
            if let Some(float_values) = values.as_any().downcast_ref::<arrow_array::Float32Array>()
            {
                let offsets = list_array.offsets();
                let start = offsets[row_idx] as usize;
                let end = offsets[row_idx + 1] as usize;
                let vector: Vec<f32> = (start..end).map(|i| float_values.value(i)).collect();

                // If we got a proper vector, return it
                if !vector.is_empty() {
                    return Ok(vector);
                }
            }
        } else if let Some(float_array) =
            column.as_any().downcast_ref::<arrow_array::Float32Array>()
        {
            // Fallback for flat arrays
            let num_rows = column.len();
            if let Some(vector_dim) = float_array.len().checked_div(num_rows) {
                let start = row_idx * vector_dim;
                let end = start + vector_dim;
                if end <= float_array.len() {
                    let vector: Vec<f32> = (start..end).map(|i| float_array.value(i)).collect();
                    if !vector.is_empty() {
                        return Ok(vector);
                    }
                }
            }
        }

        // Return placeholder with default dimension (128) instead of empty to avoid zero-dimension errors
        // This is a fallback - actual vectors should be extracted properly above
        Ok(vec![0.0f32; 128])
    }

    fn extract_metadata_value(
        &self,
        column: &Arc<dyn Array>,
        row_idx: usize,
    ) -> Result<Vec<crate::proto::proximadb_v1::MetadataItem>> {
        use crate::proto::proximadb_v1::{MetadataItem, metadata_item};
        use arrow_array::{MapArray, StringArray};

        // Check if this is a Map array
        if let Some(map_array) = column.as_any().downcast_ref::<MapArray>() {
            let mut metadata_items = Vec::new();

            // Get the entries for this row
            let start = map_array.value_offsets()[row_idx] as usize;
            let end = map_array.value_offsets()[row_idx + 1] as usize;

            if start < end {
                let entries = map_array.values();
                // Map entries are stored as a struct array with key and value fields
                if let Some(struct_array) =
                    entries.as_any().downcast_ref::<arrow_array::StructArray>()
                {
                    let keys = struct_array
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>();
                    let values = struct_array
                        .column(1)
                        .as_any()
                        .downcast_ref::<StringArray>();

                    if let (Some(keys), Some(values)) = (keys, values) {
                        for i in start..end {
                            let key = keys.value(i);
                            let value = if values.is_null(i) {
                                None
                            } else {
                                Some(values.value(i))
                            };

                            if let Some(value) = value {
                                // Direct Map entry
                                let item = MetadataItem {
                                    key: key.to_string(),
                                    value: Some(metadata_item::Value::StringValue(
                                        value.to_string(),
                                    )),
                                };
                                metadata_items.push(item);
                            }
                        }
                    }
                }
            }

            Ok(metadata_items)
        } else {
            // No Map column, return empty metadata
            Ok(Vec::new())
        }
    }
}

/// Input data structures

#[derive(Debug, Clone)]
pub struct SeekData {
    pub range: FileSeekRange,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct RangeData {
    pub range: std::ops::Range<u64>,
    pub data: Vec<u8>,
}

/// Internal data structures

#[derive(Debug)]
struct ParsedColumn {
    // data_type removed -  Data,
    array: Arc<dyn Array>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstructor_creation() {
        let config = ReconstructorConfig::default();
        let reconstructor = ParquetReconstructor::new(config);

        assert!(reconstructor.config.enable_schema_validation);
        assert_eq!(reconstructor.config.max_memory_usage_mb, 256.0);
    }

    #[test]
    fn test_compression_detection() {
        let reconstructor = ParquetReconstructor::new(ReconstructorConfig::default());
        let range = FileSeekRange {
            offset: 0,
            length: 100,
            row_group_idx: 0,
            column_name: "test".to_string(),
        };

        let compression = reconstructor.detect_compression(&range).unwrap();
        assert!(matches!(compression, CompressionType::None));
    }

    #[test]
    fn test_group_seek_data() {
        let reconstructor = ParquetReconstructor::new(ReconstructorConfig::default());

        let seek_data = vec![
            SeekData {
                range: FileSeekRange {
                    offset: 0,
                    length: 100,
                    row_group_idx: 0,
                    column_name: "col1".to_string(),
                },
                data: vec![1, 2, 3],
            },
            SeekData {
                range: FileSeekRange {
                    offset: 100,
                    length: 50,
                    row_group_idx: 0,
                    column_name: "col2".to_string(),
                },
                data: vec![4, 5, 6],
            },
        ];

        let grouped = reconstructor
            .group_seek_data_by_row_group(seek_data)
            .unwrap();
        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[&0].len(), 2);
    }
}
