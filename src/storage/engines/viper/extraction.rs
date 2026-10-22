//! VIPER Vector Extractor
//!
//! Implements the VectorExtractor trait for VIPER engine.
//! VIPER uses columnar Parquet format optimized for analytics workloads.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

use crate::core::types::StorageEngineType;
use crate::storage::engines::core::formats::columnar::columnar_query_engine::columnar_query_reader::UnifiedParquetReader;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use crate::storage::trait_components::extractor::{
    ExtractedVector, ExtractionCapabilities, ExtractionCost, ExtractionError, ExtractionMode,
    ExtractionRequest, ExtractionResult, VectorExtractor, filter_existing_files,
    finalize_extraction, id_filter_allows, mode_wants_fp32,
};

/// VIPER Vector Extractor
///
/// Extracts vectors from VIPER Parquet files for AXIS indexing.
/// VIPER is optimized for analytics with columnar storage.
pub struct ViperExtractor {
    /// Unified caching filesystem for file access
    filesystem: Arc<UnifiedCachingFilesystem>,
}

impl ViperExtractor {
    /// Create a new VIPER extractor
    pub fn new(filesystem: Arc<UnifiedCachingFilesystem>) -> Self {
        Self { filesystem }
    }
}

#[async_trait]
impl VectorExtractor for ViperExtractor {
    async fn extract_vectors(
        &self,
        request: ExtractionRequest,
    ) -> Result<ExtractionResult, ExtractionError> {
        let start = Instant::now();

        if request.file_paths.is_empty() {
            return Ok(ExtractionResult::empty());
        }

        // Filter out non-existent files to handle race conditions gracefully
        // (e.g., temp directories cleaned up before AXIS consumer processes events)
        let existing_files = filter_existing_files("VIPER", &request.file_paths);

        if existing_files.is_empty() {
            debug!("[VIPER Extractor] No existing files to process, returning empty result");
            return Ok(ExtractionResult::empty());
        }

        // Create filesystem factory for this extraction operation
        let filesystem_factory = Arc::new(
            FilesystemFactory::create_default()
                .await
                .map_err(|e| ExtractionError::IoError(e.to_string()))?,
        );

        // ID filter as a set, built once per extraction (was: a linear
        // `ids.contains` per record — TD-DSEFF-1 parity with sst/nova/swift).
        let id_filter = crate::storage::trait_components::extractor::build_id_filter(
            request.vector_ids.as_ref(),
        );

        let mut all_vectors = Vec::new();
        let mut total_bytes_read = 0usize;
        let mut files_processed = 0usize;

        for file_path in &existing_files {
            // Create a reader for this file
            let reader = match UnifiedParquetReader::new(
                vec![file_path.clone()],
                768, // Default dimension, will be read from file
                filesystem_factory.clone(),
                self.filesystem.clone(),
                "extraction".to_string(),
                "viper".to_string(),
            ) {
                Ok(r) => r,
                Err(e) => {
                    debug!("Failed to create reader for {}: {}", file_path, e);
                    continue;
                }
            };

            // Read all records from the file
            let records = reader
                .read_all_records(0, None)
                .await
                .map_err(|e| ExtractionError::EngineError(format!("VIPER read error: {}", e)))?;

            debug!(
                "[VIPER Extractor] Read {} records from {}",
                records.len(),
                file_path
            );

            // Track bytes (approximate)
            for record in &records {
                total_bytes_read += record
                    .embeddings
                    .first()
                    .map_or(0, |embedding| embedding.values.len() * 4);
            }
            files_processed += 1;

            // Convert canonical ProximaRecord to ExtractedVector
            for record in records {
                let record_id = record.oid.clone();
                // Handle vector filtering based on mode
                let fp32_vector = if mode_wants_fp32(request.mode) {
                    record
                        .embeddings
                        .first()
                        .map(|embedding| embedding.values.to_fp32_owned())
                        .filter(|values| !values.is_empty())
                } else {
                    None
                };

                // VIPER doesn't store quantized vectors inline in VectorRecord
                // Quantization is handled at the columnar storage layer
                let quantized = None;

                // Handle metadata - convert ProximaRecord props to JSON Value
                let metadata = if !record.props.is_empty() {
                    let json_map: serde_json::Map<String, serde_json::Value> = record
                        .props
                        .into_iter()
                        .filter_map(|(k, v)| serde_json::to_value(v).ok().map(|jv| (k, jv)))
                        .collect();
                    if json_map.is_empty() {
                        None
                    } else {
                        Some(serde_json::Value::Object(json_map))
                    }
                } else {
                    None
                };

                let should_include = id_filter_allows(&id_filter, &record_id);

                if should_include && fp32_vector.is_some() {
                    all_vectors.push(ExtractedVector {
                        id: record_id,
                        fp32_vector,
                        quantized,
                        metadata,
                    });
                }
            }
        }

        let result = finalize_extraction(
            all_vectors,
            request.offset,
            request.limit,
            total_bytes_read,
            files_processed,
            start,
        );

        debug!(
            "[VIPER Extractor] Extracted {} vectors from {} files in {}ms",
            result.vectors.len(),
            files_processed,
            result.stats.duration_ms
        );

        Ok(result)
    }

    fn extraction_capabilities(&self) -> ExtractionCapabilities {
        ExtractionCapabilities {
            supports_streaming: true,
            supports_incremental: false, // Parquet-based, no LSN
            available_modes: vec![
                ExtractionMode::Fp32Only,
                ExtractionMode::QuantizedOnly,
                ExtractionMode::Both,
                ExtractionMode::Auto,
            ],
            supports_metadata: true,
            optimal_batch_size: 100_000, // Parquet row groups are large
        }
    }

    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost {
        // VIPER uses columnar Parquet - efficient for full scans
        let files = request.file_paths.len();
        let vectors_per_file = 100_000;
        let bytes_per_vector = 768 * 4;

        ExtractionCost {
            estimated_vectors: files * vectors_per_file,
            estimated_bytes: files * vectors_per_file * bytes_per_vector,
            estimated_duration_ms: (files * 200) as u64, // ~200ms per large file
            io_cost: 0.7,                                // Columnar is efficient for full scans
        }
    }

    fn engine_type(&self) -> StorageEngineType {
        StorageEngineType::VIPER
    }
}

// Unit tests are minimal since actual extraction is tested in integration tests.
// The extractor trait tests in extractor.rs cover the core functionality.
