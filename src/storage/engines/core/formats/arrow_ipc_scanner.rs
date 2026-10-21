//! Arrow IPC Scanner - High-throughput full scan implementation for all engines
//!
//! This module provides Arrow IPC-based scanning for maximum throughput during:
//! - Compaction operations
//! - Data export/backup
//! - Maintenance tasks
//! - Full table scans
//!
//! ## Key Benefits
//! - Zero-copy serialization
//! - 3-5x faster than Parquet for sequential reads
//! - Memory-mapped I/O support
//! - Streaming support for large datasets
//! - Direct RecordBatch iteration

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, BinaryArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::reader::{FileReader as IpcFileReader, StreamReader as IpcStreamReader};
use arrow_ipc::writer::{
    FileWriter as IpcFileWriter, IpcWriteOptions, StreamWriter as IpcStreamWriter,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, trace};

use super::VectorSerializer;
use crate::storage::scan_strategy::{ScanIterator, ScanStatistics};
use proximadb_records::{EmbeddingCell, ProximaRecord};

/// Configuration for Arrow IPC scanning
#[derive(Debug, Clone)]
pub struct IpcScanConfig {
    /// Use streaming format vs file format
    pub use_streaming: bool,
    /// Enable memory mapping for file format
    pub enable_mmap: bool,
    /// Batch size for iteration
    pub batch_size: usize,
    /// Enable zero-copy operations
    pub enable_zero_copy: bool,
    /// Cache directory for IPC files
    pub cache_dir: PathBuf,
    /// Auto-convert Parquet to IPC for repeated scans
    pub auto_convert: bool,
    /// Delete IPC cache after use
    pub cleanup_cache: bool,
}

impl Default for IpcScanConfig {
    fn default() -> Self {
        Self {
            use_streaming: false, // File format allows random access
            enable_mmap: true,    // Memory mapping for speed
            batch_size: 10000,
            enable_zero_copy: true,
            cache_dir: PathBuf::from("/tmp/proximadb/ipc_cache"),
            auto_convert: true,
            cleanup_cache: false, // Keep cache for repeated scans
        }
    }
}

/// Arrow IPC scanner for full scan operations
pub struct ArrowIpcScanner {
    config: IpcScanConfig,
    stats: ScanStatistics,
}

impl ArrowIpcScanner {
    pub fn new(config: IpcScanConfig) -> Result<Self> {
        // Ensure cache directory exists
        std::fs::create_dir_all(&config.cache_dir)?;

        Ok(Self {
            config,
            stats: ScanStatistics::default(),
        })
    }

    /// Create scanner from source file (Parquet or IPC)
    pub async fn from_file(
        source_path: &str,
        config: IpcScanConfig,
    ) -> Result<Box<dyn ScanIterator>> {
        let scanner = Self::new(config)?;

        // Check if source is already IPC
        if source_path.ends_with(".arrow") || source_path.ends_with(".ipc") {
            scanner.scan_ipc_file(source_path).await
        } else {
            // Convert from Parquet if needed
            scanner.scan_with_conversion(source_path).await
        }
    }

    /// Scan IPC file directly
    async fn scan_ipc_file(&self, ipc_path: &str) -> Result<Box<dyn ScanIterator>> {
        info!("Starting Arrow IPC scan of {}", ipc_path);

        if self.config.enable_mmap {
            // Memory-mapped access for maximum speed
            self.create_mmap_iterator(ipc_path)
        } else if self.config.use_streaming {
            // Streaming format for lower memory usage
            self.create_stream_iterator(ipc_path)
        } else {
            // File format with random access
            self.create_file_iterator(ipc_path)
        }
    }

    /// Scan with automatic conversion from Parquet
    async fn scan_with_conversion(&self, parquet_path: &str) -> Result<Box<dyn ScanIterator>> {
        let ipc_path = self.get_ipc_cache_path(parquet_path);

        // Check if cached IPC exists and is newer than source
        if !self.needs_conversion(parquet_path, &ipc_path)? {
            debug!("Using cached IPC file: {}", ipc_path);
            return self.scan_ipc_file(&ipc_path).await;
        }

        // Convert Parquet to IPC
        info!(
            "Converting {} to Arrow IPC for faster scanning",
            parquet_path
        );
        self.convert_parquet_to_ipc(parquet_path, &ipc_path).await?;

        self.scan_ipc_file(&ipc_path).await
    }

    /// Convert Parquet to Arrow IPC format
    async fn convert_parquet_to_ipc(&self, parquet_path: &str, ipc_path: &str) -> Result<()> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema();
        let mut reader = builder.with_batch_size(self.config.batch_size).build()?;

        // Create IPC writer
        let ipc_file = File::create(ipc_path)?;
        let options = IpcWriteOptions::default();

        let mut writer = if self.config.use_streaming {
            IpcStreamWriter::try_new_with_options(ipc_file, &schema, options)?
        } else {
            IpcFileWriter::try_new_with_options(ipc_file, &schema, options)?
        };

        // Stream batches from Parquet to IPC
        let mut batch_count = 0;
        for batch_result in reader {
            let batch = batch_result?;
            writer.write(&batch)?;
            batch_count += 1;

            if batch_count % 100 == 0 {
                debug!("Converted {} batches to IPC", batch_count);
            }
        }

        writer.finish()?;
        info!(
            "IPC conversion complete: {} batches written to {}",
            batch_count, ipc_path
        );

        Ok(())
    }

    /// Create memory-mapped iterator
    fn create_mmap_iterator(&self, ipc_path: &str) -> Result<Box<dyn ScanIterator>> {
        let file = File::open(ipc_path)?;

        // Use memory mapping for zero-copy access
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        // Create reader from memory-mapped data
        let reader = IpcFileReader::try_new(std::io::Cursor::new(mmap), None)?;

        Ok(Box::new(MmapIpcIterator {
            reader,
            current_batch: 0,
            total_batches: reader.num_batches(),
            stats: ScanStatistics::default(),
            config: self.config.clone(),
        }))
    }

    /// Create streaming iterator
    fn create_stream_iterator(&self, ipc_path: &str) -> Result<Box<dyn ScanIterator>> {
        let file = File::open(ipc_path)?;
        let reader = IpcStreamReader::try_new(file, None)?;

        Ok(Box::new(StreamIpcIterator {
            reader,
            stats: ScanStatistics::default(),
            config: self.config.clone(),
            buffer: Vec::new(),
        }))
    }

    /// Create file format iterator
    fn create_file_iterator(&self, ipc_path: &str) -> Result<Box<dyn ScanIterator>> {
        let file = File::open(ipc_path)?;
        let reader = IpcFileReader::try_new(file, None)?;
        let total_batches = reader.num_batches();

        Ok(Box::new(FileIpcIterator {
            reader,
            current_batch: 0,
            total_batches,
            stats: ScanStatistics::default(),
            config: self.config.clone(),
        }))
    }

    /// Get IPC cache path for a source file
    fn get_ipc_cache_path(&self, source_path: &str) -> String {
        let file_name = Path::new(source_path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();

        self.config
            .cache_dir
            .join(format!("{}.arrow", file_name))
            .to_string_lossy()
            .to_string()
    }

    /// Check if conversion is needed
    fn needs_conversion(&self, source_path: &str, ipc_path: &str) -> Result<bool> {
        if !self.config.auto_convert {
            return Ok(true);
        }

        let ipc_meta = match std::fs::metadata(ipc_path) {
            Ok(meta) => meta,
            Err(_) => return Ok(true), // IPC doesn't exist
        };

        let source_meta = std::fs::metadata(source_path)?;

        // Convert if source is newer than cache
        Ok(source_meta.modified()? > ipc_meta.modified()?)
    }
}

/// Memory-mapped IPC iterator
struct MmapIpcIterator {
    reader: IpcFileReader<std::io::Cursor<memmap2::Mmap>>,
    current_batch: usize,
    total_batches: usize,
    stats: ScanStatistics,
    config: IpcScanConfig,
}

#[async_trait]
impl ScanIterator for MmapIpcIterator {
    async fn next_batch(&mut self) -> Result<Option<Vec<ProximaRecord>>> {
        if self.current_batch >= self.total_batches {
            return Ok(None);
        }

        let batch = self.reader.get_batch(self.current_batch)?;
        self.current_batch += 1;

        // Update statistics
        self.stats.records_scanned += batch.num_rows();
        self.stats.bytes_read += batch.get_array_memory_size();

        // Convert RecordBatch to VectorRecords
        let records = batch_to_vector_records(&batch)?;
        self.stats.records_matched += records.len();

        Ok(Some(records))
    }

    fn statistics(&self) -> ScanStatistics {
        self.stats.clone()
    }

    fn cancel(&mut self) {
        self.current_batch = self.total_batches; // Skip to end
    }
}

/// Streaming IPC iterator
struct StreamIpcIterator {
    reader: IpcStreamReader<File>,
    stats: ScanStatistics,
    config: IpcScanConfig,
    buffer: Vec<ProximaRecord>,
}

#[async_trait]
impl ScanIterator for StreamIpcIterator {
    async fn next_batch(&mut self) -> Result<Option<Vec<ProximaRecord>>> {
        // Try to fill buffer up to batch_size
        self.buffer.clear();

        while self.buffer.len() < self.config.batch_size {
            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.stats.records_scanned += batch.num_rows();
                    self.stats.bytes_read += batch.get_array_memory_size();

                    let records = batch_to_vector_records(&batch)?;
                    self.stats.records_matched += records.len();
                    self.buffer.extend(records);
                }
                Some(Err(e)) => return Err(e.into()),
                None => break, // End of stream
            }
        }

        if self.buffer.is_empty() {
            Ok(None)
        } else {
            Ok(Some(std::mem::take(&mut self.buffer)))
        }
    }

    fn statistics(&self) -> ScanStatistics {
        self.stats.clone()
    }

    fn cancel(&mut self) {
        // No clean way to cancel streaming reader
    }
}

/// File format IPC iterator
struct FileIpcIterator {
    reader: IpcFileReader<File>,
    current_batch: usize,
    total_batches: usize,
    stats: ScanStatistics,
    config: IpcScanConfig,
}

#[async_trait]
impl ScanIterator for FileIpcIterator {
    async fn next_batch(&mut self) -> Result<Option<Vec<ProximaRecord>>> {
        if self.current_batch >= self.total_batches {
            return Ok(None);
        }

        let batch = self.reader.get_batch(self.current_batch)?;
        self.current_batch += 1;

        // Update statistics
        self.stats.records_scanned += batch.num_rows();
        self.stats.bytes_read += batch.get_array_memory_size();

        // Convert RecordBatch to VectorRecords
        let records = batch_to_vector_records(&batch)?;
        self.stats.records_matched += records.len();

        Ok(Some(records))
    }

    fn statistics(&self) -> ScanStatistics {
        self.stats.clone()
    }

    fn cancel(&mut self) {
        self.current_batch = self.total_batches;
    }
}

/// Convert Arrow RecordBatch to VectorRecords
fn batch_to_vector_records(batch: &RecordBatch) -> Result<Vec<ProximaRecord>> {
    let mut records = Vec::with_capacity(batch.num_rows());

    // Extract columns
    let id_array = batch
        .column_by_name("id")
        .context("Missing 'id' column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("'id' column is not StringArray")?;

    let vector_array = batch
        .column_by_name("vector")
        .context("Missing 'vector' column")?;

    // Handle both binary and float array formats
    let vectors: Vec<Vec<f32>> =
        if let Some(binary_array) = vector_array.as_any().downcast_ref::<BinaryArray>() {
            // Binary format (serialized vectors)
            (0..binary_array.len())
                .map(|i| {
                    let bytes = binary_array.value(i);
                    deserialize_vector(bytes)
                })
                .collect::<Result<Vec<_>>>()?
        } else if let Some(float_array) = vector_array.as_any().downcast_ref::<Float32Array>() {
            // Direct float array (for fixed dimensions)
            let dim = float_array.len() / batch.num_rows();
            (0..batch.num_rows())
                .map(|i| float_array.values()[i * dim..(i + 1) * dim].to_vec())
                .collect()
        } else {
            return Err(anyhow::anyhow!("Unsupported vector column type"));
        };

    // Build ProximaRecord from Arrow columns
    for i in 0..batch.num_rows() {
        let values = vectors[i].clone();
        let dim = values.len() as u32;
        let mut record = ProximaRecord {
            oid: id_array.value(i).to_string(),
            ..ProximaRecord::default()
        };
        if !values.is_empty() {
            record.embeddings.push(EmbeddingCell::new_typed("default", "vector", dim, values));
        }
        records.push(record);
    }

    Ok(records)
}

/// Deserialize vector from binary format
/// NOTE: Now using shared VectorSerializer from core/formats/vector_serialization.rs
fn deserialize_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    VectorSerializer::deserialize_raw(bytes)
}

/// Create an IPC scanner for any storage engine
pub async fn create_ipc_scanner(
    source_path: &str,
    config: Option<IpcScanConfig>,
) -> Result<Box<dyn ScanIterator>> {
    let config = config.unwrap_or_default();
    ArrowIpcScanner::from_file(source_path, config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipc_scanner_creation() {
        let config = IpcScanConfig::default();
        let scanner = ArrowIpcScanner::new(config);
        assert!(scanner.is_ok());
    }

    #[test]
    fn test_vector_deserialization() {
        let vector = vec![1.0_f32, 2.0, 3.0];
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        let deserialized = deserialize_vector(&bytes).unwrap();
        assert_eq!(vector, deserialized);
    }

    #[test]
    fn test_batch_to_vector_records_returns_canonical_records() {
        let ids = StringArray::from(vec!["vec_1", "vec_2"]);
        let v1: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let v2: Vec<u8> = [3.0_f32, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let vectors = BinaryArray::from_vec(vec![v1.as_slice(), v2.as_slice()]);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("vector", DataType::Binary, false),
            ])),
            vec![Arc::new(ids), Arc::new(vectors)],
        )
        .expect("record batch");

        let records = batch_to_vector_records(&batch).expect("canonical records");

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].oid, "vec_1");
        assert_eq!(records[0].embeddings[0].modality, "vector");
        assert_eq!(records[0].embeddings[0].dim, 2);
        assert_eq!(records[0].embeddings[0].values, vec![1.0, 2.0]);
        assert_eq!(records[1].oid, "vec_2");
        assert_eq!(records[1].embeddings[0].values, vec![3.0, 4.0]);
    }
}
