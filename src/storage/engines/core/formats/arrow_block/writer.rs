//! Arrow Block Writer
//!
//! Writes ProximaRecords to Arrow IPC format with B+ tree indexing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_ipc::writer::{FileWriter, IpcWriteOptions};
use arrow_schema::Schema as ArrowSchema;
use proximadb_records::ProximaRecord;
use tracing::{debug, info};

use crate::storage::schema::proxima_record_bridge::DefaultProximaRecordBridge;
use crate::storage::schema::proxima_schema::ProximaSchema;

use super::config::{ArrowBlockConfig, ArrowBlockMetadata};
use super::index::{ArrowBlockIndex, ArrowIndexEntry};
use super::{ARROW_BLOCK_MAGIC, ARROW_BLOCK_VERSION, ArrowBlockError, ArrowBlockResult};

/// Writer for Arrow block files
pub struct ArrowBlockWriter {
    /// Output file writer
    inner: BufWriter<File>,

    /// Arrow IPC writer
    arrow_writer: Option<FileWriter<BufWriter<File>>>,

    /// Configuration
    config: ArrowBlockConfig,

    /// ProximaRecord to Arrow bridge
    bridge: DefaultProximaRecordBridge,

    /// Block index
    index: ArrowBlockIndex,

    /// Current block number
    current_block: u32,

    /// Accumulated records for current block
    pending_records: Vec<ProximaRecord>,

    /// File offset tracker
    current_offset: u64,

    /// Metadata accumulator
    metadata: ArrowBlockMetadata,

    /// Global min/max ID
    ///
    /// Tracks the minimum and maximum vector IDs across all blocks for range queries.
    global_min_id: Option<String>,
    global_max_id: Option<String>,

    /// Global timestamp range
    ///
    /// Tracks the minimum and maximum timestamps across all records for time-based queries.
    global_min_timestamp: Option<i64>,
    global_max_timestamp: Option<i64>,

    /// File path for sidecar index creation
    path: String,
}

impl ArrowBlockWriter {
    /// Create new writer
    pub fn new<P: AsRef<Path>>(path: P, config: ArrowBlockConfig) -> ArrowBlockResult<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);

        let schema = ProximaSchema::vector_record_schema(config.dimension);
        let bridge = DefaultProximaRecordBridge::new(schema);

        Ok(Self {
            inner: writer,
            arrow_writer: None,
            config: config.clone(),
            bridge,
            index: ArrowBlockIndex::new(),
            current_block: 0,
            pending_records: Vec::new(),
            current_offset: 0,
            metadata: ArrowBlockMetadata::from_config(&config),
            global_min_id: None,
            global_max_id: None,
            global_min_timestamp: None,
            global_max_timestamp: None,
            path: path_str,
        })
    }

    /// Add a single record
    pub fn add_record(&mut self, record: ProximaRecord) -> ArrowBlockResult<()> {
        // Update global ranges
        self.update_global_ranges(&record);

        self.pending_records.push(record);

        // Flush block if full
        if self.pending_records.len() >= self.config.records_per_block as usize {
            self.flush_block()?;
        }

        Ok(())
    }

    /// Add multiple records
    pub fn add_records(&mut self, records: &[ProximaRecord]) -> ArrowBlockResult<()> {
        for record in records {
            self.add_record(record.clone())?;
        }
        Ok(())
    }

    /// Write a complete block of records
    pub fn write_block(&mut self, records: &[ProximaRecord]) -> ArrowBlockResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Update global ranges
        for record in records {
            self.update_global_ranges(record);
        }

        // Convert to Arrow RecordBatch
        let batch = self.bridge.proxima_records_to_batch(records)?;

        // Initialize Arrow writer on first write
        if self.arrow_writer.is_none() {
            self.initialize_arrow_writer(batch.schema())?;
        }

        // Record block offset before writing
        let block_offset = self.current_offset;

        // Write batch
        if let Some(ref mut writer) = self.arrow_writer {
            writer.write(&batch)?;
        }

        // Estimate block size (will be accurate after finalize)
        let estimated_size = self.estimate_batch_size(&batch);
        self.current_offset += estimated_size;

        // Create index entry
        let (min_id, max_id) = self.get_id_range(records)?;
        let (min_ts, max_ts) = self.get_timestamp_range(records)?;

        let entry = ArrowIndexEntry::new(
            self.current_block,
            min_id,
            max_id,
            block_offset,
            estimated_size,
            records.len() as u32,
        )
        .with_timestamps(min_ts, max_ts);

        self.index.add_entry(entry);
        self.current_block += 1;
        self.metadata.total_records += records.len() as u64;

        debug!(
            "Wrote block {} with {} records at offset {}",
            self.current_block - 1,
            records.len(),
            block_offset
        );

        Ok(())
    }

    /// Flush pending records to a block
    fn flush_block(&mut self) -> ArrowBlockResult<()> {
        if self.pending_records.is_empty() {
            return Ok(());
        }

        let records = std::mem::take(&mut self.pending_records);
        self.write_block(&records)
    }

    /// Initialize Arrow IPC writer
    fn initialize_arrow_writer(&mut self, schema: Arc<ArrowSchema>) -> ArrowBlockResult<()> {
        // Use default options (no compression) since Arrow IPC compression
        // requires feature flags that may not be enabled
        let options = IpcWriteOptions::default();

        // Take ownership of inner writer temporarily
        // Note: We create a temp file as placeholder - it will be replaced immediately
        let temp_file = File::create("/dev/null").map_err(|e| {
            ArrowBlockError::Io(std::io::Error::other(format!(
                "Failed to create temp file: {e}"
            )))
        })?;
        let inner = std::mem::replace(&mut self.inner, BufWriter::new(temp_file));

        let writer = FileWriter::try_new_with_options(inner, &schema, options)
            .map_err(ArrowBlockError::Arrow)?;

        self.arrow_writer = Some(writer);

        Ok(())
    }

    /// Finalize the file with index written to sidecar file
    ///
    /// Creates two files:
    /// - Main file: Pure Arrow IPC (compatible with standard readers)
    /// - Sidecar file (.idx): B+ tree index for fast lookups
    pub fn finalize(mut self) -> ArrowBlockResult<ArrowBlockMetadata> {
        // Flush any pending records
        self.flush_block()?;

        // Finish Arrow writer - this writes the proper Arrow IPC footer
        let inner = if let Some(writer) = self.arrow_writer.take() {
            writer.into_inner().map_err(ArrowBlockError::Arrow)?
        } else {
            self.inner
        };

        // Flush the main Arrow file
        drop(inner);

        // Build B+ tree index
        self.index.build_bplus_tree(64);

        // Update metadata
        self.metadata.num_blocks = self.current_block;
        self.metadata.id_range = match (&self.global_min_id, &self.global_max_id) {
            (Some(min), Some(max)) => Some((min.clone(), max.clone())),
            _ => None,
        };
        self.metadata.timestamp_range = match (self.global_min_timestamp, self.global_max_timestamp)
        {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        };

        // Write sidecar index file
        let index_path = format!("{}.idx", self.path);
        let mut index_file = BufWriter::new(File::create(&index_path)?);

        // Write magic and version
        index_file.write_all(ARROW_BLOCK_MAGIC)?;
        index_file.write_all(&ARROW_BLOCK_VERSION.to_le_bytes())?;

        // Write metadata
        let meta_bytes = self.metadata.to_bytes();
        index_file.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
        index_file.write_all(&meta_bytes)?;

        // Write index
        let index_bytes = self.index.to_bytes();
        index_file.write_all(&(index_bytes.len() as u32).to_le_bytes())?;
        index_file.write_all(&index_bytes)?;

        index_file.flush()?;

        info!(
            "Finalized Arrow block file: {} blocks, {} records (index: {})",
            self.current_block, self.metadata.total_records, index_path
        );

        Ok(self.metadata)
    }

    /// Get the path for the sidecar index file
    ///
    /// Constructs the index file path by appending `.idx` to the Arrow file path.
    pub fn index_path(arrow_path: &str) -> String {
        format!("{arrow_path}.idx")
    }

    /// Update global ID and timestamp ranges
    fn update_global_ranges(&mut self, record: &ProximaRecord) {
        // Update ID range
        match &self.global_min_id {
            None => self.global_min_id = Some(record.oid.clone()),
            Some(min_id) if record.oid < *min_id => self.global_min_id = Some(record.oid.clone()),
            _ => {}
        }
        match &self.global_max_id {
            None => self.global_max_id = Some(record.oid.clone()),
            Some(max_id) if record.oid > *max_id => self.global_max_id = Some(record.oid.clone()),
            _ => {}
        }

        // Update timestamp range
        let ts = record.created_at_ns / 1_000_000;
        if ts > 0 {
            match self.global_min_timestamp {
                None => self.global_min_timestamp = Some(ts),
                Some(min_ts) if ts < min_ts => self.global_min_timestamp = Some(ts),
                _ => {}
            }
            match self.global_max_timestamp {
                None => self.global_max_timestamp = Some(ts),
                Some(max_ts) if ts > max_ts => self.global_max_timestamp = Some(ts),
                _ => {}
            }
        }
    }

    /// Get ID range from records
    fn get_id_range(&self, records: &[ProximaRecord]) -> ArrowBlockResult<(String, String)> {
        let mut ids: Vec<_> = records.iter().map(|r| r.oid.as_str()).collect();
        ids.sort();
        let min = ids.first().copied().ok_or_else(|| {
            ArrowBlockError::ConversionError("No IDs found in records".to_string())
        })?;
        let max = ids.last().copied().ok_or_else(|| {
            ArrowBlockError::ConversionError("No IDs found in records".to_string())
        })?;
        Ok((min.to_string(), max.to_string()))
    }

    /// Get timestamp range from records
    fn get_timestamp_range(&self, records: &[ProximaRecord]) -> ArrowBlockResult<(i64, i64)> {
        let timestamps: Vec<i64> = records
            .iter()
            .map(|r| r.created_at_ns / 1_000_000)
            .filter(|ts| *ts > 0)
            .collect();
        let min = timestamps.iter().min().copied().ok_or_else(|| {
            ArrowBlockError::ConversionError("No timestamps found in records".to_string())
        })?;
        let max = timestamps.iter().max().copied().ok_or_else(|| {
            ArrowBlockError::ConversionError("No timestamps found in records".to_string())
        })?;
        Ok((min, max))
    }

    /// Estimate batch size in bytes
    fn estimate_batch_size(&self, batch: &RecordBatch) -> u64 {
        let mut size = 0u64;

        for column in batch.columns() {
            // Arrow arrays have a get_buffer_memory_size method
            size += column.get_buffer_memory_size() as u64;
        }

        // Add overhead for IPC framing
        size += 256;

        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::EmbeddingCell;
    use tempfile::tempdir;

    fn create_test_record(id: &str, dim: usize) -> ProximaRecord {
        let timestamp_ns = chrono::Utc::now()
            .timestamp_millis()
            .saturating_mul(1_000_000);
        ProximaRecord {
            oid: id.to_string(),
            created_at_ns: timestamp_ns,
            updated_at_ns: timestamp_ns,
            record_version: 1,
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                dim as u32,
                (0..dim).map(|i| i as f32 * 0.1).collect(),
            )],
            ..ProximaRecord::default()
        }
    }

    #[test]
    fn test_write_single_block() {
        let dir = tempdir().expect("Failed to create tempdir");
        let path = dir.path().join("test.arrow");

        let config = ArrowBlockConfig::new(128);
        let mut writer =
            ArrowBlockWriter::new(&path, config).expect("Failed to create ArrowBlockWriter");

        let records: Vec<_> = (0..100)
            .map(|i| create_test_record(&format!("vec_{:05}", i), 128))
            .collect();

        writer.write_block(&records).expect("Failed to write block");
        let metadata = writer.finalize().expect("Failed to finalize writer");

        assert_eq!(metadata.num_blocks, 1);
        assert_eq!(metadata.total_records, 100);
        assert!(path.exists());
    }

    #[test]
    fn test_write_multiple_blocks() {
        let dir = tempdir().expect("Failed to create tempdir");
        let path = dir.path().join("test_multi.arrow");

        let config = ArrowBlockConfig {
            dimension: 64,
            records_per_block: 50,
            ..Default::default()
        };
        let mut writer =
            ArrowBlockWriter::new(&path, config).expect("Failed to create ArrowBlockWriter");

        // Add 200 records (should create 4 blocks)
        for i in 0..200 {
            writer
                .add_record(create_test_record(&format!("vec_{:05}", i), 64))
                .expect("Failed to add record");
        }

        let metadata = writer.finalize().expect("Failed to finalize writer");

        assert_eq!(metadata.num_blocks, 4);
        assert_eq!(metadata.total_records, 200);
    }

    /// INT-3-followup-c: fp16 records survive an end-to-end SST-equivalent
    /// flush cycle (write to disk via ArrowBlockWriter, read back via
    /// ArrowBlockReader) bit-exact. This validates the engine flush path
    /// "just works" with the typed bridge from INT-3-followup-a — no
    /// engine-layer coercion needed when the records already match the
    /// collection's canonical precision.
    #[test]
    fn fp16_records_survive_write_read_cycle_bit_exact() {
        use crate::storage::engines::core::formats::arrow_block::ArrowBlockReader;

        let dir = tempdir().expect("Failed to create tempdir");
        let path = dir.path().join("fp16_e2e.arrow");

        let dimension = 16;
        let config = ArrowBlockConfig::new(dimension as u32);
        let mut writer =
            ArrowBlockWriter::new(&path, config).expect("Failed to create ArrowBlockWriter");

        // Spread input across fp16 dynamic range to catch any silent
        // downconversion through the file format layer.
        let sources: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..dimension)
                    .map(|j| ((i as f32) * 1.5 - 8.0) + (j as f32) * 0.125)
                    .collect()
            })
            .collect();
        let records: Vec<ProximaRecord> = sources
            .iter()
            .enumerate()
            .map(|(i, src)| {
                let f16s: Vec<half::f16> = src.iter().map(|&x| half::f16::from_f32(x)).collect();
                ProximaRecord {
                    oid: format!("fp16_{:05}", i),
                    created_at_ns: chrono::Utc::now()
                        .timestamp_millis()
                        .saturating_mul(1_000_000),
                    record_version: 1,
                    embeddings: vec![EmbeddingCell {
                        model_id: "test".to_string(),
                        modality: "dense_vector".to_string(),
                        dim: dimension as u32,
                        values: proximadb_records::EmbeddingValues::Fp16(f16s),
                        precision: proximadb_records::EmbeddingScalarType::Fp16,
                        ..Default::default()
                    }],
                    ..ProximaRecord::default()
                }
            })
            .collect();

        writer
            .write_block(&records)
            .expect("write_block must accept fp16 records");
        let metadata = writer.finalize().expect("Failed to finalize writer");
        assert_eq!(metadata.total_records, 50);

        // Read back and verify the column dtype stayed fp16 + values are bit-exact.
        let reader = ArrowBlockReader::open(&path).expect("open reader");
        let read_records = reader.read_all().expect("read_all");
        assert_eq!(read_records.len(), 50);

        for (orig, got) in records.iter().zip(read_records.iter()) {
            let orig_f16 = match &orig.embeddings[0].values {
                proximadb_records::EmbeddingValues::Fp16(v) => v.clone(),
                other => panic!("orig should be Fp16, got {:?}", other.scalar_type()),
            };
            let got_f16 = match &got.embeddings[0].values {
                proximadb_records::EmbeddingValues::Fp16(v) => v.clone(),
                other => panic!(
                    "recovered must be Fp16 (file-format layer must not downconvert), got {:?}",
                    other.scalar_type()
                ),
            };
            assert_eq!(
                orig_f16, got_f16,
                "fp16 bit-exact round-trip through SST file"
            );
            assert_eq!(
                got.embeddings[0].precision,
                proximadb_records::EmbeddingScalarType::Fp16,
                "EmbeddingCell.precision must be stamped from the recovered column dtype"
            );
        }
    }

    /// INT-4-full: the demonstrable fp16 storage payoff. Writes the same
    /// content as both fp32 and fp16 to two real Arrow files, then asserts
    /// the fp16 file's vector-column bytes are ~half the fp32 file's.
    ///
    /// Why a tolerance band (45-55%) instead of exact 50%: the file
    /// includes the schema header, B+ tree index, metadata footer, and
    /// other fixed per-file overhead that doesn't shrink with the vector
    /// column. With a large enough vector payload (records × dim × bytes)
    /// the fixed overhead is amortized below the noise floor and the ratio
    /// approaches 0.5; we pick a deliberately wide band that still rejects
    /// any silent downconversion (which would make the ratio ~1.0).
    ///
    /// This is the file-format-level proof. The metric-endpoint flavor
    /// (`/metrics/prometheus` `proximadb_embedding_precision_canonical_bytes`)
    /// needs a running server and is the next layer up.
    #[test]
    fn fp16_file_is_approximately_half_the_fp32_size() {
        use crate::storage::engines::core::formats::arrow_block::ArrowBlockReader;

        let dir = tempdir().expect("Failed to create tempdir");
        let dimension: usize = 256;
        let num_records: usize = 500;

        // Generate the same source content twice — once as fp32, once as fp16.
        // Same input bytes means the only delta in output bytes is the column
        // encoding, which is the signal we're measuring.
        let make_src = |i: usize| -> Vec<f32> {
            (0..dimension)
                .map(|j| ((i as f32) * 0.001) + ((j as f32) * 0.0001))
                .collect()
        };

        // ---- fp32 file ----
        let fp32_path = dir.path().join("baseline_fp32.arrow");
        let mut writer_fp32 =
            ArrowBlockWriter::new(&fp32_path, ArrowBlockConfig::new(dimension as u32))
                .expect("fp32 writer");
        let fp32_records: Vec<ProximaRecord> = (0..num_records)
            .map(|i| {
                let src = make_src(i);
                ProximaRecord {
                    oid: format!("fp32_{:06}", i),
                    record_version: 1,
                    embeddings: vec![EmbeddingCell::new_fp32(
                        "test",
                        "dense_vector",
                        dimension as u32,
                        src,
                    )],
                    ..ProximaRecord::default()
                }
            })
            .collect();
        writer_fp32
            .write_block(&fp32_records)
            .expect("fp32 write_block");
        writer_fp32.finalize().expect("fp32 finalize");

        // ---- fp16 file ----
        let fp16_path = dir.path().join("native_fp16.arrow");
        let mut writer_fp16 =
            ArrowBlockWriter::new(&fp16_path, ArrowBlockConfig::new(dimension as u32))
                .expect("fp16 writer");
        let fp16_records: Vec<ProximaRecord> = (0..num_records)
            .map(|i| {
                let src = make_src(i);
                let f16s: Vec<half::f16> = src.iter().map(|&x| half::f16::from_f32(x)).collect();
                ProximaRecord {
                    oid: format!("fp16_{:06}", i),
                    record_version: 1,
                    embeddings: vec![EmbeddingCell {
                        model_id: "test".to_string(),
                        modality: "dense_vector".to_string(),
                        dim: dimension as u32,
                        values: proximadb_records::EmbeddingValues::Fp16(f16s),
                        precision: proximadb_records::EmbeddingScalarType::Fp16,
                        ..Default::default()
                    }],
                    ..ProximaRecord::default()
                }
            })
            .collect();
        writer_fp16
            .write_block(&fp16_records)
            .expect("fp16 write_block");
        writer_fp16.finalize().expect("fp16 finalize");

        // ---- measure + assert ----
        let fp32_size = std::fs::metadata(&fp32_path).expect("fp32 metadata").len();
        let fp16_size = std::fs::metadata(&fp16_path).expect("fp16 metadata").len();
        let ratio = fp16_size as f64 / fp32_size as f64;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "fp16 file should be 45-55% of the fp32 file (vector column \
             is the dominant payload at {dimension} dims × {num_records} records). \
             Got fp32={fp32_size}B, fp16={fp16_size}B, ratio={ratio:.4}. \
             A ratio near 1.0 indicates the fp16 column was silently \
             downconverted to fp32 somewhere in the bridge / Arrow IPC stack."
        );

        // Sanity check the contents are recoverable + still fp16 / fp32 typed —
        // size-only assertion could pass for a malformed file; we want both.
        let reader_fp16 = ArrowBlockReader::open(&fp16_path).expect("open fp16 reader");
        let read_fp16 = reader_fp16.read_all().expect("read fp16");
        assert_eq!(read_fp16.len(), num_records);
        assert!(matches!(
            read_fp16[0].embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp16(_)
        ));

        let reader_fp32 = ArrowBlockReader::open(&fp32_path).expect("open fp32 reader");
        let read_fp32 = reader_fp32.read_all().expect("read fp32");
        assert_eq!(read_fp32.len(), num_records);
        assert!(matches!(
            read_fp32[0].embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(_)
        ));
    }

    #[test]
    fn test_id_range_tracking() {
        let dir = tempdir().expect("Failed to create tempdir");
        let path = dir.path().join("test_range.arrow");

        let config = ArrowBlockConfig::new(32);
        let mut writer =
            ArrowBlockWriter::new(&path, config).expect("Failed to create ArrowBlockWriter");

        let records = vec![
            create_test_record("aaa", 32),
            create_test_record("mmm", 32),
            create_test_record("zzz", 32),
        ];

        writer.write_block(&records).expect("Failed to write block");
        let metadata = writer.finalize().expect("Failed to finalize writer");

        assert_eq!(
            metadata.id_range,
            Some(("aaa".to_string(), "zzz".to_string()))
        );
    }
}
