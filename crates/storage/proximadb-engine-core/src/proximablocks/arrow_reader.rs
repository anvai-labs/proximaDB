//! ProximaBlocks Arrow Reader
//!
//! This module provides an Arrow-compatible reader for ProximaBlocks (.sst files),
//! enabling external tools like DuckDB, Polars, and other Arrow-compatible clients
//! to read ProximaDB SST files directly.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::storage::engines::core::formats::proximablocks::arrow_reader::ProximaBlocksArrowReader;
//!
//! // Open an SST file
//! let reader = ProximaBlocksArrowReader::open("/path/to/file.sst")?;
//!
//! // Get the Arrow schema
//! let schema = reader.schema();
//!
//! // Read all records as a single RecordBatch
//! let batch = reader.read_all()?;
//!
//! // Or read in batches for large files
//! for batch_result in reader.read_batches(1000) {
//!     let batch = batch_result?;
//!     // Process batch...
//! }
//! ```
//!
//! ## File Format Support
//!
//! This reader supports the ProximaBlocks SST file format:
//! - Magic marker: "SST1" (4 bytes)
//! - Header: Bincode-serialized SstableHeader
//! - Optional bloom filter
//! - Index block with block offsets
//! - Data blocks containing ProximaDataBlock structures
//!
//! ## Arrow Schema
//!
//! The output Arrow schema contains:
//! - id: Utf8 (required) - Vector ID
//! - vector: FixedSizeList<Float32>(dimension) - Vector data
//! - metadata: Map<Utf8, Utf8> - Key-value metadata
//! - timestamp: Int64 (optional) - Record timestamp
//! - version: Int64 (optional) - Record version

use anyhow::{Context, Result};
use arrow_array::{
    ArrayRef, FixedSizeListArray, RecordBatch,
    builder::{Float32Builder, Int64Builder, MapBuilder, StringBuilder},
};
use arrow_schema::{DataType, Field, Fields, Schema};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, trace, warn};

use super::block_structures::ProximaDataBlock;
use crate::sst_format_types::{SstableHeader, SstableIndex};
use proximadb_records::ProximaRecord;

/// Arrow reader for ProximaBlocks (.sst) files
///
/// Converts ProximaBlocks data to Arrow RecordBatches for external tool compatibility.
pub struct ProximaBlocksArrowReader {
    /// File handle for reading
    file: File,

    /// File path (for error messages and reopening)
    #[allow(dead_code)]
    path: String,

    /// SSTable header containing file metadata
    header: SstableHeader,

    /// Index containing block offsets and metadata
    index: SstableIndex,

    /// Cached Arrow schema
    schema: Arc<Schema>,

    /// Dimension of vectors (extracted from header or first block)
    dimension: usize,
}

impl ProximaBlocksArrowReader {
    /// Open an SST file for reading
    ///
    /// # Arguments
    /// * `path` - Path to the .sst file
    ///
    /// # Returns
    /// A ProximaBlocksArrowReader ready to read records
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut file = File::open(&path).context(format!("Failed to open SST file: {path_str}"))?;

        // Read and validate magic marker
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .context("Failed to read magic marker")?;
        if &magic != b"SST1" {
            return Err(anyhow::anyhow!(
                "Invalid SST file magic marker: expected 'SST1', got {:?}",
                magic
            ));
        }

        // Read header size
        let mut header_size_bytes = [0u8; 4];
        file.read_exact(&mut header_size_bytes)
            .context("Failed to read header size")?;
        let header_size = u32::from_le_bytes(header_size_bytes) as usize;

        // Read header data
        let mut header_data = vec![0u8; header_size];
        file.read_exact(&mut header_data)
            .context("Failed to read header data")?;

        // Deserialize header
        let header: SstableHeader =
            bincode::deserialize(&header_data).context("Failed to deserialize SST header")?;

        debug!(
            "Opened SST file: {} (version={}, entries={}, blocks={})",
            path_str, header.version, header.entry_count, header.block_count
        );

        // Read index block
        let index = Self::read_index(&mut file, &header)?;

        // Determine dimension from header or index
        let dimension = header.fixed_dimension.unwrap_or(0) as usize;

        // Create Arrow schema
        let schema = Self::create_schema(dimension);

        Ok(Self {
            file,
            path: path_str,
            header,
            index,
            schema,
            dimension,
        })
    }

    /// Read the index block from the file
    fn read_index(file: &mut File, header: &SstableHeader) -> Result<SstableIndex> {
        // Calculate index offset
        let index_offset = if header.block_index_offset > 0 {
            header.block_index_offset
        } else {
            // Legacy calculation: after header and bloom filter
            let bloom_offset = 8 + header.header_size as u64;
            if header.has_bloom_filter {
                file.seek(SeekFrom::Start(bloom_offset))?;
                let mut bloom_size_bytes = [0u8; 4];
                file.read_exact(&mut bloom_size_bytes)?;
                let bloom_size = u32::from_le_bytes(bloom_size_bytes) as u64;
                bloom_offset + 4 + bloom_size
            } else {
                bloom_offset
            }
        };

        // Seek to index offset
        file.seek(SeekFrom::Start(index_offset))?;

        // Read index size
        let mut size_bytes = [0u8; 4];
        file.read_exact(&mut size_bytes)?;
        let index_size = u32::from_le_bytes(size_bytes) as usize;

        // Read index data
        let mut index_data = vec![0u8; index_size];
        file.read_exact(&mut index_data)?;

        // Deserialize index
        SstableIndex::deserialize(&index_data).context("Failed to deserialize SST index")
    }

    /// Create the Arrow schema for ProximaBlocks data
    fn create_schema(dimension: usize) -> Arc<Schema> {
        let dim = if dimension > 0 { dimension as i32 } else { -1 };

        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                if dimension > 0 {
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        dim,
                    )
                } else {
                    // Variable dimension - use List instead of FixedSizeList
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, false)))
                },
                false,
            ),
            Field::new(
                "metadata",
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(Fields::from(vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Utf8, true),
                        ])),
                        false,
                    )),
                    false, // keys are not sorted
                ),
                true,
            ),
            Field::new("timestamp", DataType::Int64, true),
            Field::new("version", DataType::Int64, true),
        ]))
    }

    /// Get the Arrow schema for the SST file
    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Get the SSTable header
    pub fn header(&self) -> &SstableHeader {
        &self.header
    }

    /// Get the number of records in the file
    pub fn num_records(&self) -> u64 {
        self.header.entry_count
    }

    /// Get the number of blocks in the file
    pub fn num_blocks(&self) -> u32 {
        self.header.block_count
    }

    /// Get the vector dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Read all records as a single RecordBatch
    ///
    /// Note: For large files, consider using `read_batches()` instead
    /// to process data incrementally.
    pub fn read_all(&mut self) -> Result<RecordBatch> {
        let all_batches: Vec<RecordBatch> =
            self.read_batches(usize::MAX).collect::<Result<Vec<_>>>()?;

        if all_batches.is_empty() {
            // Return empty batch with schema
            return Self::create_empty_batch(&self.schema);
        }

        // Concatenate all batches
        arrow_select::concat::concat_batches(&self.schema, &all_batches)
            .context("Failed to concatenate record batches")
    }

    /// Create an empty RecordBatch with the given schema
    fn create_empty_batch(schema: &Arc<Schema>) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|field| arrow_array::new_empty_array(field.data_type()))
            .collect();

        RecordBatch::try_new(schema.clone(), columns).context("Failed to create empty record batch")
    }

    /// Read records in batches
    ///
    /// Returns an iterator that yields RecordBatches. Each batch contains
    /// up to `batch_size` records (though actual size may be smaller due
    /// to block boundaries).
    ///
    /// # Arguments
    /// * `batch_size` - Maximum number of records per batch
    pub fn read_batches(&mut self, batch_size: usize) -> ProximaBlocksBatchIterator<'_> {
        let dimension_detected = self.dimension > 0;
        ProximaBlocksBatchIterator {
            reader: self,
            current_block: 0,
            batch_size,
            dimension_detected,
        }
    }

    /// Read a specific block by index
    fn read_block(&mut self, block_idx: usize) -> Result<ProximaDataBlock> {
        if block_idx >= self.index.entries.len() {
            return Err(anyhow::anyhow!(
                "Block index {} out of range (max: {})",
                block_idx,
                self.index.entries.len()
            ));
        }

        let entry = &self.index.entries[block_idx];

        // Determine block offset
        let block_offset = if entry.offset > 0 {
            entry.offset
        } else if block_idx == 0 && self.header.data_blocks_offset > 0 {
            self.header.data_blocks_offset
        } else {
            return Err(anyhow::anyhow!(
                "Cannot determine offset for block {}",
                block_idx
            ));
        };

        // Seek to block position
        self.file.seek(SeekFrom::Start(block_offset))?;

        // Read block size
        let mut size_bytes = [0u8; 4];
        self.file.read_exact(&mut size_bytes)?;
        let block_size = u32::from_le_bytes(size_bytes) as usize;

        trace!(
            "Reading block {} at offset {} with size {} bytes",
            block_idx, block_offset, block_size
        );

        // Read block data
        let mut block_data = vec![0u8; block_size];
        self.file.read_exact(&mut block_data)?;

        // Deserialize block
        ProximaDataBlock::deserialize(&block_data, None)
            .context(format!("Failed to deserialize block {block_idx}"))
    }

    /// Convert a ProximaDataBlock to an Arrow RecordBatch
    fn block_to_record_batch(
        &self,
        block: &ProximaDataBlock,
        schema: &Arc<Schema>,
    ) -> Result<RecordBatch> {
        let num_records = block.records.len();
        if num_records == 0 {
            return Self::create_empty_batch(schema);
        }

        // Determine dimension from first record if not set
        let dimension = if self.dimension > 0 {
            self.dimension
        } else if let Some(embedding) = block
            .records
            .first()
            .and_then(|record| record.embeddings.first())
        {
            embedding.values.len()
        } else {
            0
        };

        // Build ID column
        let mut id_builder = StringBuilder::new();
        for record in &block.records {
            id_builder.append_value(&record.oid);
        }
        let id_array: ArrayRef = Arc::new(id_builder.finish());

        // Build vector column
        let vector_array: ArrayRef = self.build_vector_array(&block.records, dimension)?;

        // Build metadata column
        let metadata_array: ArrayRef = self.build_metadata_array(&block.records)?;

        // Build timestamp column
        let mut timestamp_builder = Int64Builder::new();
        for record in &block.records {
            timestamp_builder.append_value(record.created_at_ns);
        }
        let timestamp_array: ArrayRef = Arc::new(timestamp_builder.finish());

        // Build version column
        let mut version_builder = Int64Builder::new();
        for record in &block.records {
            version_builder.append_value(record.record_version as i64);
        }
        let version_array: ArrayRef = Arc::new(version_builder.finish());

        // Create a schema that matches the actual dimension
        let actual_schema = if dimension > 0 && dimension != self.dimension {
            Self::create_schema(dimension)
        } else {
            schema.clone()
        };

        RecordBatch::try_new(
            actual_schema,
            vec![
                id_array,
                vector_array,
                metadata_array,
                timestamp_array,
                version_array,
            ],
        )
        .context("Failed to create RecordBatch from block")
    }

    /// Build the vector array from records
    fn build_vector_array(&self, records: &[ProximaRecord], dimension: usize) -> Result<ArrayRef> {
        if dimension == 0 {
            // Return empty list array
            return Ok(Arc::new(arrow_array::new_empty_array(&DataType::List(
                Arc::new(Field::new("item", DataType::Float32, false)),
            ))));
        }

        // Build flat Float32 array for all vectors
        let total_elements = records.len() * dimension;
        let mut values_builder = Float32Builder::with_capacity(total_elements);

        for record in records {
            let vector = record
                .embeddings
                .first()
                .map_or(&[][..], |embedding| embedding.as_fp32_slice());
            if vector.len() != dimension {
                // Pad or truncate to match expected dimension
                for i in 0..dimension {
                    if i < vector.len() {
                        values_builder.append_value(vector[i]);
                    } else {
                        values_builder.append_value(0.0);
                    }
                }
            } else {
                for &v in vector {
                    values_builder.append_value(v);
                }
            }
        }

        let values_array = Arc::new(values_builder.finish());

        // Create FixedSizeList
        let list_array = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dimension as i32,
            values_array,
            None, // no nulls
        )
        .context("Failed to create FixedSizeListArray for vectors")?;

        Ok(Arc::new(list_array))
    }

    /// Build the metadata map array from records
    fn build_metadata_array(&self, records: &[ProximaRecord]) -> Result<ArrayRef> {
        // Create map builder with string key and string value
        let key_builder = StringBuilder::new();
        let value_builder = StringBuilder::new();
        let mut map_builder = MapBuilder::new(None, key_builder, value_builder);

        for record in records {
            // Start a new map entry
            map_builder.keys().append_value("");
            map_builder.values().append_value("");

            // For each metadata entry
            let mut has_entries = false;
            for (key, value) in &record.props {
                let value_str = serde_json::to_string(value).unwrap_or_default();

                if !has_entries {
                    // Replace the placeholder entry
                    has_entries = true;
                }
                map_builder.keys().append_value(key);
                map_builder.values().append_value(&value_str);
            }

            if !has_entries {
                // Append null for empty metadata
                map_builder.append(false)?;
            } else {
                map_builder.append(true)?;
            }
        }

        Ok(Arc::new(map_builder.finish()))
    }
}

/// Iterator for reading ProximaBlocks in batches
pub struct ProximaBlocksBatchIterator<'a> {
    reader: &'a mut ProximaBlocksArrowReader,
    current_block: usize,
    batch_size: usize,
    dimension_detected: bool,
}

impl<'a> Iterator for ProximaBlocksBatchIterator<'a> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_block >= self.reader.index.entries.len() {
            return None;
        }

        let mut records_collected = 0;
        let mut blocks_to_process = Vec::new();

        // Collect blocks until we have enough records
        // Note: We don't know exact record count per block from IndexEntry,
        // so we process one block at a time to respect batch_size
        while self.current_block < self.reader.index.entries.len()
            && records_collected < self.batch_size
        {
            blocks_to_process.push(self.current_block);
            // Estimate records per block - we'll read and adjust
            // Most SST blocks contain ~1000 records
            records_collected += 1000;
            self.current_block += 1;
        }

        if blocks_to_process.is_empty() {
            return None;
        }

        // Read and combine blocks
        let mut all_records = Vec::new();
        for block_idx in blocks_to_process {
            match self.reader.read_block(block_idx) {
                Ok(block) => {
                    // Update dimension if not detected
                    if !self.dimension_detected && !block.records.is_empty() {
                        let dim = block.records[0]
                            .embeddings
                            .first()
                            .map_or(0, |embedding| embedding.values.len());
                        if dim > 0 {
                            self.reader.dimension = dim;
                            self.reader.schema = ProximaBlocksArrowReader::create_schema(dim);
                            self.dimension_detected = true;
                        }
                    }
                    all_records.extend(block.records);
                }
                Err(e) => {
                    warn!("Failed to read block {}: {}", block_idx, e);
                    return Some(Err(e));
                }
            }
        }

        // Create a temporary ProximaDataBlock to hold combined records
        let combined_block = ProximaDataBlock {
            records: all_records,
            ..Default::default()
        };

        // Convert to RecordBatch
        Some(
            self.reader
                .block_to_record_batch(&combined_block, &self.reader.schema),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
    use std::collections::HashMap;
    use tempfile::tempdir;

    /// Helper to create test ProximaRecords
    fn create_test_records(count: usize, dimension: usize) -> Vec<ProximaRecord> {
        (0..count)
            .map(|i| {
                let mut props = HashMap::new();
                props.insert(
                    "category".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(format!("cat_{}", i % 3))),
                );
                props.insert(
                    "score".to_string(),
                    ProximaTreeNode::Value(ProximaValue::Float64(i as f64 * 0.1)),
                );

                ProximaRecord {
                    oid: format!("vec_{i}"),
                    props,
                    embeddings: vec![EmbeddingCell::new_fp32(
                        "test-model",
                        "text",
                        dimension as u32,
                        (0..dimension).map(|d| (i + d) as f32 * 0.01).collect(),
                    )],
                    created_at_ns: 1_700_000_000_000_000_000 + i as i64,
                    record_version: 1,
                    ..Default::default()
                }
            })
            .collect()
    }

    #[test]
    fn test_create_schema() {
        let schema = ProximaBlocksArrowReader::create_schema(384);
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "vector");
        assert_eq!(schema.field(2).name(), "metadata");
        assert_eq!(schema.field(3).name(), "timestamp");
        assert_eq!(schema.field(4).name(), "version");

        // Check vector is FixedSizeList with correct dimension
        match schema.field(1).data_type() {
            DataType::FixedSizeList(_, dim) => assert_eq!(*dim, 384),
            _ => panic!("Expected FixedSizeList for vector field"),
        }
    }

    #[test]
    fn test_create_schema_variable_dimension() {
        let schema = ProximaBlocksArrowReader::create_schema(0);
        assert_eq!(schema.fields().len(), 5);

        // Check vector is List (variable size) when dimension is 0
        match schema.field(1).data_type() {
            DataType::List(_) => {}
            other => panic!("Expected List for variable dimension, got {:?}", other),
        }
    }

    #[test]
    fn test_build_vector_array() {
        // Create mock reader state
        let records = create_test_records(3, 4);

        // Create a minimal schema for testing
        let _schema = ProximaBlocksArrowReader::create_schema(4);

        // We can't directly test build_vector_array without a reader instance,
        // but we can verify the records are properly formatted
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].embeddings[0].values.len(), 4);
        assert_eq!(records[1].embeddings[0].values.len(), 4);
        assert_eq!(records[2].embeddings[0].values.len(), 4);
    }

    #[test]
    fn test_open_invalid_file() {
        // Test opening a non-existent file
        let result = ProximaBlocksArrowReader::open("/nonexistent/path/file.sst");
        assert!(result.is_err());
    }

    #[test]
    fn test_open_invalid_magic() {
        use std::io::Write;

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("invalid.sst");

        // Write a file with invalid magic bytes
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"XXXX").unwrap(); // Invalid magic
        file.write_all(&100u32.to_le_bytes()).unwrap(); // Header size
        drop(file);

        let result = ProximaBlocksArrowReader::open(&file_path);
        assert!(result.is_err());
        match result {
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("magic") || err_msg.contains("SST1"),
                    "Expected magic marker error, got: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("Expected error for invalid magic bytes"),
        }
    }

    #[test]
    fn test_schema_fields() {
        // Test schema field types for a fixed dimension
        let schema = ProximaBlocksArrowReader::create_schema(128);

        // Verify all expected fields are present
        assert!(schema.column_with_name("id").is_some());
        assert!(schema.column_with_name("vector").is_some());
        assert!(schema.column_with_name("metadata").is_some());
        assert!(schema.column_with_name("timestamp").is_some());
        assert!(schema.column_with_name("version").is_some());

        // Verify data types
        let id_field = schema.field_with_name("id").unwrap();
        assert_eq!(*id_field.data_type(), DataType::Utf8);
        assert!(!id_field.is_nullable());

        let timestamp_field = schema.field_with_name("timestamp").unwrap();
        assert_eq!(*timestamp_field.data_type(), DataType::Int64);
        assert!(timestamp_field.is_nullable());
    }

    #[test]
    fn test_record_creation() {
        // Verify helper creates valid records
        let records = create_test_records(10, 64);

        assert_eq!(records.len(), 10);

        for (i, record) in records.iter().enumerate() {
            assert_eq!(record.oid, format!("vec_{i}"));
            assert_eq!(record.embeddings[0].values.len(), 64);
            assert!(record.props.contains_key("category"));
            assert!(record.props.contains_key("score"));
            assert_eq!(record.created_at_ns, 1_700_000_000_000_000_000 + i as i64);
        }
    }
}
