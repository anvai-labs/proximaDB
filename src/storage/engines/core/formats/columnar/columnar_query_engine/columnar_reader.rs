//! Parquet Reader Core Implementation
//!
//! This module provides the core reader functionality for Parquet files,
//! including file access, record batch reading, and conversion to ProximaRecords.

use crate::storage::persistence::filesystem::FileSystem;
use anyhow::{Context, Result};
use arrow::record_batch::RecordBatch;
use arrow_array::Array;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::sync::Arc;
use tracing::{debug, info};

use crate::storage::engines::core::formats::columnar::constants::{
    FIELD_EXPIRES_AT, FIELD_ID, FIELD_IS_DELETED, FIELD_TIMESTAMP, FIELD_VECTOR_FP32, FIELD_VERSION,
};
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};

use super::columnar_query_reader::UnifiedParquetReader;
use super::{QueryConfig, QueryStatistics};

/// Core Parquet reader implementation
pub struct ParquetReader {
    #[allow(dead_code)]
    config: QueryConfig,
    stats: QueryStatistics,
}

impl ParquetReader {
    /// Create new reader with configuration
    pub fn new(config: QueryConfig) -> Self {
        Self {
            config,
            stats: QueryStatistics::default(),
        }
    }

    /// Read Parquet file and return all records
    pub fn read_all(
        &mut self,
        file_path: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<ProximaRecord>>> + Send + '_>>
    {
        let file_path = file_path.to_string();
        Box::pin(async move {
            info!("Reading all records from {}", file_path);

            let file = File::open(&file_path).context("Failed to open Parquet file")?;

            let reader = SerializedFileReader::new(file)?;
            let metadata = reader.metadata();

            debug!(
                "File has {} row groups with {} total rows",
                metadata.num_row_groups(),
                metadata.file_metadata().num_rows()
            );

            // Use UnifiedParquetReader for actual reading
            // Create UnifiedCachingFilesystem for optimal performance
            let filesystem_factory = Arc::new(
                crate::storage::persistence::filesystem::FilesystemFactory::create_default()
                    .await?,
            );
            let base_fs = filesystem_factory.get_filesystem("file://")?;
            let cached_filesystem = Arc::new(
                crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                    base_fs,
                    "default_collection".to_string(),
                    "columnar".to_string(),
                ),
            );
            let unified_reader = UnifiedParquetReader::new(
                vec![file_path.clone()],
                1024,
                filesystem_factory,
                cached_filesystem,
                "default_collection".to_string(),
                "columnar".to_string(),
            )?;

            // Read all records
            let records = unified_reader.read_all_records(0, None).await?;

            // Update statistics
            self.stats.files_read += 1;
            self.stats.records_read += records.len();
            self.stats.row_groups_read += metadata.num_row_groups();

            Ok(records)
        })
    }

    /// Read Parquet file using filesystem API
    pub async fn read_all_with_filesystem(
        &mut self,
        file_path: &str,
        filesystem: Arc<dyn FileSystem>,
    ) -> Result<Vec<ProximaRecord>> {
        info!(
            "Reading all records from {} using filesystem API",
            file_path
        );

        // For now, we'll read the entire file into memory and create a byte slice
        // This is not optimal for large files, but works with the current Parquet reader API
        // Note: In the future, implement streaming readers that work with async I/O
        let file_data = filesystem
            .read(file_path)
            .await
            .context("Failed to read Parquet file from filesystem")?;

        // Use bytes::Bytes which implements ChunkReader
        let bytes = bytes::Bytes::from(file_data);
        let reader = SerializedFileReader::new(bytes.clone())?;
        let metadata = reader.metadata();

        debug!(
            "File has {} row groups with {} total rows",
            metadata.num_row_groups(),
            metadata.file_metadata().num_rows()
        );

        // Read data using Arrow ParquetRecordBatchReader
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let arrow_reader = builder.build()?;

        let mut all_records = Vec::new();
        for batch_result in arrow_reader {
            let batch = batch_result?;
            let records = self.batch_to_records(batch)?;
            all_records.extend(records);
        }

        // Update statistics
        self.stats.files_read += 1;
        self.stats.records_read += all_records.len();
        self.stats.row_groups_read += metadata.num_row_groups();

        Ok(all_records)
    }

    /// Read specific row groups
    pub async fn read_row_groups(
        &mut self,
        file_path: &str,
        row_groups: &[usize],
    ) -> Result<Vec<ProximaRecord>> {
        debug!("Reading row groups {:?} from {}", row_groups, file_path);

        let mut all_records = Vec::new();

        for &row_group in row_groups {
            let file = File::open(file_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.with_row_groups(vec![row_group]).build()?;

            for batch_result in reader {
                let batch = batch_result?;
                let records = self.batch_to_records(batch)?;
                all_records.extend(records);
            }
        }

        self.stats.row_groups_read += row_groups.len();
        self.stats.records_read += all_records.len();

        Ok(all_records)
    }

    /// Read with column projection
    pub async fn read_projected(
        &mut self,
        file_path: &str,
        columns: &[String],
    ) -> Result<RecordBatch> {
        debug!("Reading projected columns {:?} from {}", columns, file_path);

        let file = File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema();

        // Build projection mask
        let column_indices: Vec<usize> = columns
            .iter()
            .filter_map(|name| schema.index_of(name).ok())
            .collect();

        let projection = ProjectionMask::roots(builder.parquet_schema(), column_indices.clone());

        let reader = builder.with_projection(projection).build()?;

        // Read first batch for now (could be extended)
        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No data in file"))??;

        self.stats.columns_projected += columns.len();

        Ok(batch)
    }

    /// Convert Arrow RecordBatch to ProximaRecords.
    ///
    /// `pub(crate)` so the NOVA ranged row-group reader (TD-040) decodes
    /// object_store-streamed batches identically to the full-file path —
    /// keeping ranged-read results byte-for-byte equal to a full read.
    pub(crate) fn batch_to_records(&self, batch: RecordBatch) -> Result<Vec<ProximaRecord>> {
        let num_rows = batch.num_rows();
        let mut records = Vec::with_capacity(num_rows);

        debug!("🔍 DEBUG batch_to_records: Processing {} rows", num_rows);
        debug!("🔍 DEBUG: Batch has {} columns", batch.num_columns());

        // Debug: Print all column names
        let schema = batch.schema();
        for (idx, field) in schema.fields().iter().enumerate() {
            debug!(
                "🔍 DEBUG: Column[{}]: name='{}', type={:?}",
                idx,
                field.name(),
                field.data_type()
            );
        }

        // Extract ID column
        let id_array = batch
            .column_by_name(FIELD_ID)
            .ok_or_else(|| anyhow::anyhow!("ID column not found"))?
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| anyhow::anyhow!("ID column is not string type"))?;

        // Extract timestamp column
        let timestamp_array = batch
            .column_by_name(FIELD_TIMESTAMP)
            .ok_or_else(|| anyhow::anyhow!("Timestamp column not found"))?
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("Timestamp column is not i64 type"))?;

        // Extract vector column - handle both List and FixedSizeList for compatibility
        let vector_column = batch
            .column_by_name(FIELD_VECTOR_FP32)
            .ok_or_else(|| anyhow::anyhow!("Vector column not found"))?;

        // Try FixedSizeListArray first (preferred), then fallback to ListArray
        let vector_values = if let Some(fixed_list) = vector_column
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>(
        ) {
            // Handle FixedSizeList (newer format)
            debug!("🔍 DEBUG: Vector column is FixedSizeList");

            // Extract vectors from fixed size list
            let values = fixed_list.values();
            let float_array = values
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Vector values are not float32"))?;

            let list_size = fixed_list.value_length() as usize;
            let mut vector_values = Vec::with_capacity(num_rows);

            for row in 0..num_rows {
                let start = row * list_size;
                let end = start + list_size;
                let vector: Vec<f32> = (start..end).map(|i| float_array.value(i)).collect();
                vector_values.push(vector);
            }
            vector_values
        } else if let Some(list_array) = vector_column
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
        {
            // Handle ListArray (older format)
            debug!("🔍 DEBUG: Vector column is ListArray (legacy format)");

            let mut vector_values = Vec::with_capacity(num_rows);
            let values = list_array.values();
            let float_array = values
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Vector values are not float32"))?;

            for row in 0..num_rows {
                let start = list_array.value_offsets()[row] as usize;
                let end = list_array.value_offsets()[row + 1] as usize;
                let vector: Vec<f32> = (start..end).map(|i| float_array.value(i)).collect();
                vector_values.push(vector);
            }
            vector_values
        } else {
            return Err(anyhow::anyhow!(
                "Vector column is neither FixedSizeList nor List type"
            ));
        };

        // Extract metadata columns - look for any columns that aren't standard columns
        let standard_columns = [
            FIELD_ID,
            FIELD_TIMESTAMP,
            FIELD_VECTOR_FP32,
            FIELD_VERSION,
            FIELD_EXPIRES_AT,
            FIELD_IS_DELETED,
            "row_group_offset",
            "row_index",
        ];

        for row in 0..num_rows {
            let mut metadata: std::collections::HashMap<String, ProximaTreeNode> =
                std::collections::HashMap::new();

            // Check each column to see if it's a metadata column
            for field in schema.fields() {
                let column_name = field.name();

                // Skip standard columns
                if standard_columns.contains(&column_name.as_str()) {
                    continue;
                }

                debug!(
                    "🔍 DEBUG: Processing potential metadata column: {}",
                    column_name
                );

                // Try to extract metadata value from this column
                if let Some(column) = batch.column_by_name(column_name) {
                    // Check if this is a Map column (for metadata stored as key-value pairs)
                    if column_name == "extra_meta" {
                        // Handle Map type for metadata
                        if let Some(map_array) =
                            column.as_any().downcast_ref::<arrow::array::MapArray>()
                        {
                            debug!(
                                "🔍 DEBUG: Processing Map for row {}, is_null={}",
                                row,
                                map_array.is_null(row)
                            );
                            if !map_array.is_null(row) {
                                let offsets = map_array.offsets();
                                let start = offsets[row] as usize;
                                let end = offsets[row + 1] as usize;
                                debug!(
                                    "🔍 DEBUG: Map offsets for row {}: start={}, end={}, entries={}",
                                    row,
                                    start,
                                    end,
                                    end - start
                                );

                                // Get the struct array that contains key-value pairs
                                // MapArray.values() returns the flattened entries, not individual maps
                                // We need to use the struct array directly
                                let entries = map_array.entries();
                                debug!("🔍 DEBUG: Map entries type: {:?}", entries.data_type());
                                if let Some(struct_array) =
                                    entries.as_any().downcast_ref::<arrow::array::StructArray>()
                                {
                                    debug!(
                                        "🔍 DEBUG: Found StructArray with {} entries, {} columns",
                                        struct_array.len(),
                                        struct_array.num_columns()
                                    );
                                    // Get key and value arrays
                                    if let (Some(key_array), Some(value_array)) = (
                                        struct_array.column_by_name("key"),
                                        struct_array.column_by_name("value"),
                                    ) {
                                        debug!("🔍 DEBUG: Found key and value arrays");
                                        if let (Some(keys), Some(values)) = (
                                            key_array
                                                .as_any()
                                                .downcast_ref::<arrow::array::StringArray>(),
                                            value_array
                                                .as_any()
                                                .downcast_ref::<arrow::array::StringArray>(),
                                        ) {
                                            // Extract all key-value pairs for this row
                                            debug!(
                                                "🔍 DEBUG: Extracting {} entries for row {}",
                                                end - start,
                                                row
                                            );
                                            for i in start..end {
                                                if !keys.is_null(i) && !values.is_null(i) {
                                                    let key = keys.value(i);
                                                    let value = values.value(i);
                                                    debug!(
                                                        "🔍 DEBUG: Found map metadata {}={} for row {}",
                                                        key, value, row
                                                    );
                                                    metadata.insert(
                                                        key.to_string(),
                                                        ProximaTreeNode::Value(
                                                            ProximaValue::String(value.to_string()),
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        continue; // Skip the regular column type checks for map columns
                    }

                    // Try different types of columns for non-map metadata
                    if let Some(string_array) =
                        column.as_any().downcast_ref::<arrow::array::StringArray>()
                    {
                        // Skip null values - use is_null() method which checks the null bitmap
                        if !string_array.is_null(row) {
                            let value = string_array.value(row);
                            debug!(
                                "🔍 DEBUG: Found string metadata {}={} for row {}",
                                column_name, value, row
                            );
                            metadata.insert(
                                column_name.to_string(),
                                ProximaTreeNode::Value(ProximaValue::String(value.to_string())),
                            );
                        }
                    } else if let Some(int_array) =
                        column.as_any().downcast_ref::<arrow::array::Int64Array>()
                    {
                        if !int_array.is_null(row) {
                            let value = int_array.value(row);
                            debug!(
                                "🔍 DEBUG: Found int metadata {}={} for row {}",
                                column_name, value, row
                            );
                            metadata.insert(
                                column_name.to_string(),
                                ProximaTreeNode::Value(ProximaValue::Int64(value)),
                            );
                        }
                    } else if let Some(float_array) =
                        column.as_any().downcast_ref::<arrow::array::Float64Array>()
                    {
                        if !float_array.is_null(row) {
                            let value = float_array.value(row);
                            debug!(
                                "🔍 DEBUG: Found float metadata {}={} for row {}",
                                column_name, value, row
                            );
                            metadata.insert(
                                column_name.to_string(),
                                ProximaTreeNode::Value(ProximaValue::Float64(value)),
                            );
                        }
                    } else if let Some(bool_array) =
                        column.as_any().downcast_ref::<arrow::array::BooleanArray>()
                        && !bool_array.is_null(row)
                    {
                        let value = bool_array.value(row);
                        debug!(
                            "🔍 DEBUG: Found bool metadata {}={} for row {}",
                            column_name, value, row
                        );
                        metadata.insert(
                            column_name.to_string(),
                            ProximaTreeNode::Value(ProximaValue::Boolean(value)),
                        );
                    }
                }
            }

            let values = vector_values[row].clone();
            let dim = values.len() as u32;
            let mut record = ProximaRecord {
                oid: id_array.value(row).to_string(),
                created_at_ns: timestamp_array.value(row) * 1_000_000,
                props: metadata,
                ..Default::default()
            };
            if !values.is_empty() {
                record
                    .embeddings
                    .push(EmbeddingCell::new_fp32("default", "vector", dim, values));
            }

            if row < 3 || record.oid.contains("_A_") || record.oid.contains("_B_") && row < 25 {
                debug!(
                    "🔍 DEBUG: Created record {}: metadata keys={:?}, values={:?}",
                    record.oid,
                    record.props.keys().collect::<Vec<_>>(),
                    record
                        .props
                        .iter()
                        .map(|(k, v)| {
                            let val_str = match v {
                                ProximaTreeNode::Value(ProximaValue::String(s)) => s.clone(),
                                ProximaTreeNode::Value(ProximaValue::Float64(f)) => f.to_string(),
                                ProximaTreeNode::Value(ProximaValue::Boolean(b)) => b.to_string(),
                                ProximaTreeNode::Value(ProximaValue::Int64(i)) => i.to_string(),
                                _ => "?".to_string(),
                            };
                            format!("{k}={val_str}")
                        })
                        .collect::<Vec<_>>()
                );
            }

            records.push(record);
        }

        debug!(
            "🔍 DEBUG batch_to_records: Extracted {} records",
            records.len()
        );
        Ok(records)
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> QueryStatistics {
        self.stats.clone()
    }
}

/// Builder for ParquetReader
pub struct ReaderBuilder {
    config: QueryConfig,
}

impl ReaderBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: QueryConfig {
                enable_pushdown: true,
                enable_projection: true,
                enable_statistics: true,
                cache_strategy: super::CacheStrategy::LRU,
                limit: None,
                enable_parallel: true,
                parallel_workers: 4,
            },
        }
    }

    /// Set query configuration
    pub fn with_config(mut self, config: QueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable/disable predicate pushdown
    pub fn with_pushdown(mut self, enable: bool) -> Self {
        self.config.enable_pushdown = enable;
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.config.limit = Some(limit);
        self
    }

    /// Build the reader
    pub fn build(self) -> ParquetReader {
        ParquetReader::new(self.config)
    }
}

impl Default for ReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reader_builder() {
        let reader = ReaderBuilder::new()
            .with_pushdown(false)
            .with_limit(100)
            .build();

        assert!(!reader.config.enable_pushdown);
        assert_eq!(reader.config.limit, Some(100));
    }
}
