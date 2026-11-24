// Unified Parquet Compaction for Columnar Engines (NOVA, etc.)
//
// Works directly with Arrow RecordBatches for efficient columnar compaction
// without unnecessary conversions to/from ProximaRecord

use anyhow::{Context, Result};
use arrow_array::{Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_ord::sort::{SortOptions, sort_to_indices};
use arrow_schema::Schema;
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::metadata_collector::MetadataCollector;
use super::{FIELD_EXPIRES_AT, FIELD_ID, FIELD_TIMESTAMP, FIELD_VERSION};
use crate::storage::common::compaction_orchestrator::FilenameCodec;
use crate::storage::persistence::filesystem::FilesystemFactory;

/// Version continuity enforcement mode for MVCC
#[derive(Debug, Clone, Copy, Default)]
pub enum VersionContinuityMode {
    /// Strict: Versions must be exactly contiguous (0, 1, 2, ...)
    #[default]
    Strict,
    /// Relaxed: Allow gaps but reject huge jumps (e.g., jump from 1 to 9999)
    Relaxed { max_jump: i64 },
    /// Disabled: No version continuity checking (legacy mode)
    Disabled,
}

/// Result of unified compaction operation
#[derive(Debug)]
pub struct ColumnarCompactionResult {
    pub input_files: Vec<String>,
    pub output_files: Vec<String>,
    pub entries_processed: u64,
    pub entries_removed: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

impl From<ColumnarCompactionResult> for crate::storage::traits::CompactionResult {
    fn from(val: ColumnarCompactionResult) -> Self {
        crate::storage::traits::CompactionResult {
            success: true,
            collections_affected: vec![], // Will be filled by the caller
            entries_processed: Some(val.entries_processed),
            entries_removed: Some(val.entries_removed),
            bytes_read: Some(val.bytes_read),
            bytes_written: Some(val.bytes_written),
            input_files: None,     // Will be filled by the caller
            output_files: Some(1), // Always produces one output file
            duration_ms: None,     // Will be filled by the caller
            completed_at: chrono::Utc::now(),
            engine_metrics: std::collections::HashMap::new(),
        }
    }
}

/// Unified compaction service for columnar engines
pub struct UnifiedColumnarCompaction {
    filesystem_factory: Arc<FilesystemFactory>,
    version_continuity: VersionContinuityMode,
}

impl UnifiedColumnarCompaction {
    pub fn new(filesystem_factory: Arc<FilesystemFactory>) -> Self {
        Self {
            filesystem_factory,
            version_continuity: VersionContinuityMode::default(),
        }
    }

    pub fn with_version_mode(mut self, mode: VersionContinuityMode) -> Self {
        self.version_continuity = mode;
        self
    }

    /// Main compaction entry point for any columnar engine
    pub async fn compact_parquet_files(
        &self,
        collection_id: &str,
        input_files: Vec<String>,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
        engine_name: &str, // "VIPER", "NOVA", etc. for logging
        mut metadata_collector: Option<Box<dyn MetadataCollector>>, // For NOVA hierarchical metadata
    ) -> Result<ColumnarCompactionResult> {
        info!(
            "[{}] Starting columnar compaction for collection {}",
            engine_name, collection_id
        );

        // Discover input files if not provided
        let input_files = if input_files.is_empty() {
            self.discover_files(collection_id, collection_config, engine_name)
                .await?
        } else {
            input_files
        };

        if input_files.is_empty() {
            info!(
                "[{}] No files to compact for collection {}",
                engine_name, collection_id
            );
            return Ok(ColumnarCompactionResult {
                input_files: vec![],
                output_files: vec![],
                entries_processed: 0,
                entries_removed: 0,
                bytes_read: 0,
                bytes_written: 0,
            });
        }

        // Read all records as Arrow batches
        let (all_batches, schema, total_bytes_read, total_records) =
            self.read_all_records(&input_files).await?;

        // Deduplicate with MVCC directly on Arrow batches
        let (deduped_batch, expired_count) =
            self.deduplicate_arrow_batches(all_batches, schema.clone())?;
        let entries_removed = total_records - deduped_batch.num_rows() as u64 + expired_count;

        // Generate output filename
        let output_path = self
            .generate_output_filename(collection_id, engine_name, collection_config)
            .await?;

        // Write compacted batch with sorting and bloom filters
        let bytes_written = self
            .write_compacted_arrow_batch(
                &output_path,
                &deduped_batch,
                schema,
                collection_config,
                &mut metadata_collector,
            )
            .await?;

        // Atomic replacement
        self.atomic_file_replacement(&input_files, &output_path)
            .await?;

        info!(
            "[{}] Compaction complete: {} -> {} records ({} removed)",
            engine_name,
            total_records,
            deduped_batch.num_rows(),
            entries_removed
        );

        Ok(ColumnarCompactionResult {
            input_files,
            output_files: vec![output_path],
            entries_processed: total_records,
            entries_removed,
            bytes_read: total_bytes_read,
            bytes_written,
        })
    }

    /// Discover files to compact based on engine-specific strategy
    async fn discover_files(
        &self,
        collection_id: &str,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
        engine_name: &str,
    ) -> Result<Vec<String>> {
        let base_path = collection_config
            .and_then(|c| c.storage_assignment.as_ref())
            .map(|sa| format!("{}/{}/data", sa.base_location, collection_id))
            .ok_or_else(|| anyhow::anyhow!("No storage assignment for collection"))?;

        let fs = self.filesystem_factory.get_filesystem(&base_path)?;
        let entries = fs.list(&base_path).await?;

        let mut files: Vec<String> = entries
            .iter()
            .filter(|e| e.name.ends_with(".parquet") && !e.metadata.is_directory)
            .map(|e| format!("{}/{}", base_path, e.name))
            .collect();

        files.sort();

        // Simple strategy: compact if we have more than 4 files
        if files.len() <= 4 {
            debug!(
                "[{}] Only {} files, no compaction needed",
                engine_name,
                files.len()
            );
            return Ok(vec![]);
        }

        // Keep newest files for reads, compact older ones
        files.truncate(files.len() - 2);
        Ok(files)
    }

    /// Read all records from input files using Arrow
    async fn read_all_records(
        &self,
        input_files: &[String],
    ) -> Result<(Vec<RecordBatch>, Arc<Schema>, u64, u64)> {
        let fs = self.filesystem_factory.get_filesystem("file:///")?;
        let mut all_batches = Vec::new();
        let mut total_bytes = 0u64;
        let mut total_records = 0u64;
        let mut schema: Option<Arc<Schema>> = None;

        for file_path in input_files {
            debug!("Reading file: {}", file_path);
            let file_data = fs.read(file_path).await?;
            total_bytes += file_data.len() as u64;

            let file_bytes = bytes::Bytes::from(file_data);
            let reader = ParquetRecordBatchReaderBuilder::try_new(file_bytes.clone())?.build()?;

            if schema.is_none() {
                // Get schema from reader's metadata
                let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(file_bytes.clone())?;
                schema = Some(parquet_reader.schema().clone());
            }

            for batch_result in reader {
                let batch = batch_result?;
                total_records += batch.num_rows() as u64;
                all_batches.push(batch);
            }
        }

        Ok((
            all_batches,
            schema.ok_or_else(|| anyhow::anyhow!("No schema found"))?,
            total_bytes,
            total_records,
        ))
    }

    /// Apply quantization to a batch of records during compaction
    /// This recalculates quantization for merged data distribution
    async fn apply_quantization_to_batch(
        &self,
        batch: &RecordBatch,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<RecordBatch> {
        use crate::compute::quantization::storage_engine::{
            StorageQuantizationConfig, StorageQuantizationEngine,
        };
        use crate::storage::engines::core::formats::columnar::constants;
        use arrow::array::ArrayRef;
        use arrow_array::{FixedSizeListArray, Float32Array, builder::BinaryBuilder};
        use proximadb_distance_kernel::DistanceMetric;
        use proximadb_runtime_common::pool::VectorMemoryPool;
        use std::sync::Arc;

        // Extract vector column from batch
        let vector_column = batch
            .column_by_name(constants::FIELD_VECTOR_FP32)
            .or_else(|| batch.column_by_name("vector"))
            .ok_or_else(|| anyhow::anyhow!("No vector column found"))?;

        let vectors: Vec<Vec<f32>> =
            if let Some(fixed_list) = vector_column.as_any().downcast_ref::<FixedSizeListArray>() {
                let values = fixed_list
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow::anyhow!("Invalid vector values"))?;

                let dim = fixed_list.value_length() as usize;
                let mut all_vectors = Vec::with_capacity(batch.num_rows());

                for i in 0..batch.num_rows() {
                    let start = i * dim;
                    let end = start + dim;
                    let vector: Vec<f32> = (start..end).map(|j| values.value(j)).collect();
                    all_vectors.push(vector);
                }
                all_vectors
            } else {
                return Err(anyhow::anyhow!("Vector column is not FixedSizeList"));
            };

        // Get distance metric from config
        let distance_metric = collection_config
            .and_then(|c| c.config.as_ref())
            .and_then(|c| DistanceMetric::try_from(c.distance_metric.unwrap_or(0)).ok())
            .unwrap_or(DistanceMetric::Cosine);

        // Create storage quantization config based on collection settings
        let mut config = StorageQuantizationConfig {
            distance_metric,
            enable_hardware_acceleration: true,
            ..Default::default()
        };

        // Determine which quantization levels to generate
        let quantization_config = collection_config
            .and_then(|c| c.config.as_ref())
            .and_then(|c| c.quantization.as_ref());

        // Configure levels based on collection config
        if let Some(q_config) = quantization_config {
            // Parse quantization config to determine levels
            // Default: Binary + INT8 for fast pre-filtering
            config.filter_level =
                Some(proximadb_quantization_model::UnifiedQuantizationLevel::Binary);
            config.fast_level =
                Some(proximadb_quantization_model::UnifiedQuantizationLevel::int8());

            // PQ only if explicitly enabled due to training cost
            if q_config.enable_pq.unwrap_or(false) {
                config.primary_level =
                    Some(proximadb_quantization_model::UnifiedQuantizationLevel::pq8(
                        q_config.pq_segments.unwrap_or(1).max(1) as u8,
                    ));
            }
        }

        // Create quantization engine with memory pool and SIMD optimization
        let mut engine = StorageQuantizationEngine::new_with_config(config);

        // Create unified memory pool for batch processing
        let _memory_pool = Arc::new(VectorMemoryPool::new());

        // Use batch configuration for optimal performance
        let batch_config = crate::storage::strategy::BatchConfig {
            batch_size: 1024, // Optimal for SIMD and cache locality
            batch_timeout_ms: 100,
        };

        info!(
            "🔬 Training quantization on {} vectors for compaction",
            vectors.len()
        );

        // Train on a sample if dataset is large (PQ training doesn't need all vectors)
        let training_vectors = if vectors.len() > 10000 {
            // Sample 10K vectors for training to limit memory usage
            let step = vectors.len() / 10000;
            vectors.iter().step_by(step).cloned().collect::<Vec<_>>()
        } else {
            vectors.clone()
        };

        engine.train(&training_vectors).await?;

        // Quantize in batches for memory efficiency and SIMD performance
        info!(
            "⚡ Quantizing {} vectors in batches of {} with SIMD optimization",
            vectors.len(),
            batch_config.batch_size
        );

        let mut quantized_results = Vec::with_capacity(vectors.len());

        // Process in batches to leverage SIMD and reduce memory pressure
        for chunk in vectors.chunks(batch_config.batch_size) {
            // Memory pool provides aligned memory for SIMD operations internally

            // The StorageQuantizationEngine internally uses:
            // - Unified memory pools for zero-copy operations
            // - AVX2/AVX512 for x86_64
            // - NEON for ARM
            // - Aligned memory buffers for optimal SIMD performance
            // - UnifiedDistanceBuffer for vectorized distance computations

            let batch_results = engine.quantize_batch(chunk, None).await?;
            quantized_results.extend(batch_results);

            // Allow async runtime to handle other tasks
            tokio::task::yield_now().await;
        }

        info!(
            "✅ Quantized {} vectors with SIMD acceleration",
            quantized_results.len()
        );

        // Build new columns for quantized vectors
        let mut columns: Vec<(String, ArrayRef)> = Vec::new();

        // Copy all original columns
        for field in batch.schema().fields() {
            let column_name = field.name();
            if let Some(column) = batch.column_by_name(column_name) {
                columns.push((column_name.to_string(), column.clone()));
            }
        }

        // Add binary quantized column if present
        if quantized_results.iter().any(|r| r.filter.is_some()) {
            let mut binary_builder = BinaryBuilder::new();
            for result in &quantized_results {
                if let Some(binary) = &result.filter {
                    binary_builder.append_value(&binary.data);
                } else {
                    binary_builder.append_null();
                }
            }
            columns.push((
                constants::FIELD_Q_BINARY.to_string(),
                Arc::new(binary_builder.finish()),
            ));
            info!(
                "✅ Added {} column with recalculated binary quantization",
                constants::FIELD_Q_BINARY
            );
        }

        // Add INT8 quantized column if present
        if quantized_results.iter().any(|r| r.fast.is_some()) {
            let mut int8_builder = BinaryBuilder::new();
            for result in &quantized_results {
                if let Some(int8) = &result.fast {
                    int8_builder.append_value(&int8.data);
                } else {
                    int8_builder.append_null();
                }
            }
            columns.push((
                constants::FIELD_Q_INT8.to_string(),
                Arc::new(int8_builder.finish()),
            ));
            info!(
                "✅ Added {} column with recalculated INT8 quantization",
                constants::FIELD_Q_INT8
            );
        }

        // Add PQ quantized column if present
        if quantized_results.iter().any(|r| r.primary.is_some()) {
            let mut pq_builder = BinaryBuilder::new();
            for result in &quantized_results {
                if let Some(pq) = &result.primary {
                    pq_builder.append_value(&pq.data);
                } else {
                    pq_builder.append_null();
                }
            }
            columns.push((
                constants::FIELD_Q_PQ8.to_string(),
                Arc::new(pq_builder.finish()),
            ));
            info!(
                "✅ Added {} column with retrained PQ codebooks",
                constants::FIELD_Q_PQ8
            );
        }

        // Create new schema with quantized columns
        let mut fields = batch.schema().fields().to_vec();

        // Add new fields if they don't exist
        if !fields.iter().any(|f| f.name() == constants::FIELD_Q_BINARY)
            && columns
                .iter()
                .any(|(name, _)| name == constants::FIELD_Q_BINARY)
        {
            fields.push(Arc::new(arrow::datatypes::Field::new(
                constants::FIELD_Q_BINARY,
                arrow::datatypes::DataType::Binary,
                true,
            )));
        }

        if !fields.iter().any(|f| f.name() == constants::FIELD_Q_INT8)
            && columns
                .iter()
                .any(|(name, _)| name == constants::FIELD_Q_INT8)
        {
            fields.push(Arc::new(arrow::datatypes::Field::new(
                constants::FIELD_Q_INT8,
                arrow::datatypes::DataType::Binary,
                true,
            )));
        }

        if !fields.iter().any(|f| f.name() == constants::FIELD_Q_PQ8)
            && columns
                .iter()
                .any(|(name, _)| name == constants::FIELD_Q_PQ8)
        {
            fields.push(Arc::new(arrow::datatypes::Field::new(
                constants::FIELD_Q_PQ8,
                arrow::datatypes::DataType::Binary,
                true,
            )));
        }

        let new_schema = Arc::new(arrow::datatypes::Schema::new(fields));

        // Build the final batch with quantized columns
        let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, array)| array).collect();
        RecordBatch::try_new(new_schema, arrays)
            .context("Failed to create batch with quantized vectors")
    }

    /// Deduplicate Arrow batches using MVCC with version continuity checking
    fn deduplicate_arrow_batches(
        &self,
        batches: Vec<RecordBatch>,
        schema: Arc<Schema>,
    ) -> Result<(RecordBatch, u64)> {
        if batches.is_empty() {
            return Err(anyhow::anyhow!("No batches to compact"));
        }

        let mut latest_records: HashMap<String, (usize, usize, i64, i64)> = HashMap::new();
        let mut expired_count = 0u64;
        let mut version_violations = 0u64;
        let current_time = chrono::Utc::now().timestamp();

        // Process all batches to find latest versions
        for (batch_idx, batch) in batches.iter().enumerate() {
            let id_array = batch
                .column_by_name(FIELD_ID)
                .context("Missing ID column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("ID column has wrong type")?;

            let version_array = batch
                .column_by_name(FIELD_VERSION)
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

            let timestamp_array = batch
                .column_by_name(FIELD_TIMESTAMP)
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

            let expires_at_array = batch
                .column_by_name(FIELD_EXPIRES_AT)
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

            for row_idx in 0..batch.num_rows() {
                // Skip expired records
                if let Some(expires_at) = expires_at_array
                    && expires_at.is_valid(row_idx)
                {
                    let expiry = expires_at.value(row_idx);
                    if expiry > 0 && expiry < current_time {
                        expired_count += 1;
                        continue;
                    }
                }

                // Skip null IDs
                if id_array.is_null(row_idx) {
                    continue;
                }

                let id = id_array.value(row_idx);
                let version = version_array.map_or(0, |v| {
                    if v.is_valid(row_idx) {
                        v.value(row_idx)
                    } else {
                        0
                    }
                });
                let timestamp = timestamp_array.map_or(0, |t| {
                    if t.is_valid(row_idx) {
                        t.value(row_idx)
                    } else {
                        0
                    }
                });

                // Check version continuity
                if !self.check_version_continuity(id, version, &latest_records) {
                    version_violations += 1;
                    warn!(
                        "Version continuity violation for ID '{}': version {}",
                        id, version
                    );
                    continue;
                }

                // MVCC resolution: higher version or timestamp wins
                if let Some(&(_, _, existing_ver, existing_ts)) = latest_records.get(id) {
                    if version > existing_ver
                        || (version == existing_ver && timestamp > existing_ts)
                    {
                        latest_records
                            .insert(id.to_string(), (batch_idx, row_idx, version, timestamp));
                    }
                } else {
                    latest_records.insert(id.to_string(), (batch_idx, row_idx, version, timestamp));
                }
            }
        }

        if version_violations > 0 {
            warn!(
                "Version continuity: {} violations detected",
                version_violations
            );
        }

        // Build result batch by selecting deduplicated rows
        let result_batch = self.select_deduplicated_rows(batches, latest_records, schema)?;

        Ok((result_batch, expired_count))
    }

    /// Check version continuity based on configured mode
    fn check_version_continuity(
        &self,
        id: &str,
        version: i64,
        latest: &HashMap<String, (usize, usize, i64, i64)>,
    ) -> bool {
        match self.version_continuity {
            VersionContinuityMode::Strict => match latest.get(id) {
                Some(&(_, _, existing_ver, _)) => {
                    version == existing_ver || version == existing_ver + 1
                }
                None => version == 0 || version == 1,
            },
            VersionContinuityMode::Relaxed { max_jump } => match latest.get(id) {
                Some(&(_, _, existing_ver, _)) => version <= existing_ver + max_jump,
                None => version <= max_jump,
            },
            VersionContinuityMode::Disabled => true,
        }
    }

    /// Select specific rows from batches to build deduplicated result
    fn select_deduplicated_rows(
        &self,
        batches: Vec<RecordBatch>,
        selected: HashMap<String, (usize, usize, i64, i64)>,
        schema: Arc<Schema>,
    ) -> Result<RecordBatch> {
        // First, concatenate all batches into a single batch
        let combined_batch = concat_batches(&schema, &batches)?;

        // Build indices array for selected rows
        let mut indices_to_keep = Vec::new();

        // Sort selected records by ID for better compression
        let mut sorted_selected: Vec<_> = selected.into_iter().collect();
        sorted_selected.sort_by_key(|(id, _)| id.clone());

        for (_, (batch_idx, row_idx, _, _)) in sorted_selected {
            // Calculate absolute index in combined batch
            let mut abs_idx = row_idx as u32;
            for batch in batches.iter().take(batch_idx) {
                abs_idx += batch.num_rows() as u32;
            }
            indices_to_keep.push(abs_idx);
        }

        // Use take to select specific rows
        let indices_array = UInt32Array::from(indices_to_keep);
        let mut selected_columns = Vec::new();

        for i in 0..combined_batch.num_columns() {
            let column = combined_batch.column(i);
            let selected = take(column.as_ref(), &indices_array, None)?;
            selected_columns.push(selected);
        }

        RecordBatch::try_new(schema, selected_columns)
            .context("Failed to create deduplicated batch")
    }

    /// Write compacted Arrow batch directly using ArrowWriter with metadata collection
    async fn write_compacted_arrow_batch(
        &self,
        output_path: &str,
        batch: &RecordBatch,
        schema: Arc<Schema>,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
        metadata_collector: &mut Option<Box<dyn MetadataCollector>>,
    ) -> Result<u64> {
        // Sort by filterable columns if configured
        let sorted_batch = self.sort_batch_by_filterable(batch, collection_config)?;

        // CRITICAL: Quantization-aware compaction requirement
        // When quantization is enabled, compaction CANNOT simply merge row groups!
        // It MUST:
        // 1. Load ALL FP32 vectors into memory (not just metadata)
        // 2. Recalculate quantization on the merged dataset:
        //    - INT8: Scan all vectors to find new min/max for scaling
        //    - PQ: Retrain codebooks using k-means on merged vectors
        //    - Binary: Optionally recalculate thresholds
        // 3. Create new quantized columns with updated values
        //
        // This makes compaction significantly more expensive:
        // - Memory: O(n * dimension * 4 bytes) for all vectors
        // - Compute: O(n * k * iterations) for PQ training
        // - I/O: Must read entire vector columns, not just statistics
        //
        // Implementation approach:
        let quantization_enabled = collection_config
            .and_then(|c| c.config.as_ref())
            .and_then(|c| c.quantization.as_ref())
            .is_some();

        let final_batch = if quantization_enabled {
            // Implement quantization-aware compaction using StorageQuantizationEngine
            match self
                .apply_quantization_to_batch(&sorted_batch, collection_config)
                .await
            {
                Ok(quantized_batch) => {
                    info!("✅ Applied quantization during compaction");
                    quantized_batch
                }
                Err(e) => {
                    warn!(
                        "⚠️ Quantization during compaction failed: {}, using original vectors",
                        e
                    );
                    sorted_batch
                }
            }
        } else {
            sorted_batch
        };

        // Build writer properties with bloom filters and compression
        let mut props_builder = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_write_batch_size(10000)
            .set_data_page_size_limit(1024 * 1024) // 1MB pages
            .set_bloom_filter_enabled(true);

        // Enable bloom filters for ID and filterable columns
        if let Some(collection) = collection_config
            && let Some(config) = &collection.config
        {
            let filterable_cols = &config.filterable_columns;
            // ID column always gets bloom filter
            props_builder = props_builder.set_column_bloom_filter_enabled(
                parquet::schema::types::ColumnPath::from(FIELD_ID),
                true,
            );

            // Add bloom filters for each filterable column
            for col in filterable_cols {
                props_builder = props_builder.set_column_bloom_filter_enabled(
                    parquet::schema::types::ColumnPath::from(col.name.as_str()),
                    true,
                );
            }
        }

        let writer_properties = props_builder.build();

        // Strip file:// prefix if present for local file operations
        let local_path = if let Some(stripped) = output_path.strip_prefix("file://") {
            stripped
        } else {
            output_path
        };

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(local_path).parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }

        // Create and write with ArrowWriter
        let file =
            File::create(local_path).with_context(|| format!("Failed to create: {local_path}"))?;

        let mut writer = ArrowWriter::try_new(file, schema, Some(writer_properties))?;

        // Process batch with metadata collector if provided (for NOVA)
        if let Some(collector) = metadata_collector.as_mut() {
            // Process batch through metadata collector
            collector.on_batch_write(&final_batch, 0, 0)?;
        }

        writer.write(&final_batch)?;
        let metadata = writer.close()?;

        // Write metadata sidecar file if collector was used
        if let Some(collector) = metadata_collector {
            let sidecar_path = format!("{}.meta", local_path);
            // Finalize and write sidecar file
            collector.finalize(1)?;
            let metadata_bytes = collector.serialize_metadata()?;
            if !metadata_bytes.is_empty() {
                // Use output_path (with file://) for filesystem operations
                let sidecar_file = format!("{}.{}", output_path, collector.sidecar_extension());
                let fs = self.filesystem_factory.get_filesystem("file:///")?;
                fs.write(&sidecar_file, &metadata_bytes, None).await?;
            }
            // Sidecar file written successfully
            debug!("Wrote NOVA metadata sidecar to {}", sidecar_path);
        }

        let bytes_written = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.total_byte_size())
            .sum::<i64>() as u64;

        Ok(bytes_written)
    }

    /// Sort batch by first filterable column for better compression
    fn sort_batch_by_filterable(
        &self,
        batch: &RecordBatch,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<RecordBatch> {
        // Try to find a filterable column to sort by
        if let Some(collection) = collection_config
            && let Some(config) = &collection.config
        {
            let filterable_cols = &config.filterable_columns;
            if let Some(first_col) = filterable_cols.first() {
                // Check if this column exists in the batch
                if let Some(column) = batch.column_by_name(&first_col.name) {
                    // Sort by this column
                    let sort_options = SortOptions {
                        descending: false,
                        nulls_first: false,
                    };
                    let indices = sort_to_indices(column, Some(sort_options), None)?;

                    // Take all columns using the sorted indices
                    let mut sorted_columns = Vec::new();
                    for i in 0..batch.num_columns() {
                        let sorted = take(batch.column(i).as_ref(), &indices, None)?;
                        sorted_columns.push(sorted);
                    }

                    debug!("Sorted batch by filterable column: {}", first_col.name);
                    return RecordBatch::try_new(batch.schema(), sorted_columns)
                        .context("Failed to create sorted batch");
                }
            }
        }

        // No filterable column found or configured, return as-is
        Ok(batch.clone())
    }

    /// Generate output filename using unified FilenameCodec
    async fn generate_output_filename(
        &self,
        collection_id: &str,
        _engine_name: &str,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<String> {
        let codec = FilenameCodec::new();
        let filename = codec.generate(1, "parquet"); // L1 for compacted files

        // Get the base location from the collection config
        let base_path = collection_config
            .and_then(|c| c.storage_assignment.as_ref())
            .map(|sa| format!("{}/{}/data", sa.base_location, collection_id))
            .ok_or_else(|| anyhow::anyhow!("No storage assignment for collection"))?;

        Ok(format!("{base_path}/{filename}"))
    }

    /// Atomically replace old files with new
    async fn atomic_file_replacement(
        &self,
        input_files: &[String],
        output_path: &str,
    ) -> Result<()> {
        let fs = self.filesystem_factory.get_filesystem("file:///")?;

        // Verify output exists
        if !fs.exists(output_path).await? {
            return Err(anyhow::anyhow!("Output file doesn't exist after write"));
        }

        // Delete input files
        for input_file in input_files {
            if let Err(e) = fs.delete(input_file).await {
                warn!("Failed to delete input file {}: {}", input_file, e);
            }
        }

        Ok(())
    }
}
