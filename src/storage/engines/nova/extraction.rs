//! NOVA Vector Extractor
//!
//! Implements the VectorExtractor trait for NOVA engine.
//! NOVA uses progressive columnar format with zone maps and quantization.

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

/// NOVA Vector Extractor
///
/// Extracts vectors from NOVA Parquet files.
/// NOVA uses progressive columnar format with zone maps for efficient pruning.
pub struct NovaExtractor {
    /// Unified caching filesystem for file access
    filesystem: Arc<UnifiedCachingFilesystem>,
}

impl NovaExtractor {
    /// Create a new NOVA extractor
    pub fn new(filesystem: Arc<UnifiedCachingFilesystem>) -> Self {
        Self { filesystem }
    }
}

#[async_trait]
impl VectorExtractor for NovaExtractor {
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
        let existing_files = filter_existing_files("NOVA", &request.file_paths);

        if existing_files.is_empty() {
            debug!("[NOVA Extractor] No existing files to process, returning empty result");
            return Ok(ExtractionResult::empty());
        }

        // Create filesystem factory for this extraction operation
        let filesystem_factory = Arc::new(
            FilesystemFactory::create_default()
                .await
                .map_err(|e| ExtractionError::IoError(e.to_string()))?,
        );

        // ID filter as a set, built once per extraction call (was: a linear
        // scan of the requested-id Vec for every record).
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
                "nova".to_string(),
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
                .map_err(|e| ExtractionError::EngineError(format!("NOVA read error: {}", e)))?;

            debug!(
                "[NOVA Extractor] Read {} records from {}",
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
                // Apply the ID filter BEFORE any materialization (was: after
                // the fp32 copy and metadata conversion).
                let should_include = id_filter_allows(&id_filter, &record_id);
                if !should_include {
                    continue;
                }
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

                // NOVA doesn't store quantized vectors inline in VectorRecord
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

                if fp32_vector.is_some() {
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
            "[NOVA Extractor] Extracted {} vectors from {} files in {}ms",
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
            optimal_batch_size: 50_000, // NOVA uses zone maps for batching
        }
    }

    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost {
        // NOVA uses progressive columnar with zone maps
        let files = request.file_paths.len();
        let vectors_per_file = 50_000;
        let bytes_per_vector = 768 * 4;

        ExtractionCost {
            estimated_vectors: files * vectors_per_file,
            estimated_bytes: files * vectors_per_file * bytes_per_vector,
            estimated_duration_ms: (files * 150) as u64, // ~150ms per file
            io_cost: 0.75,                               // Good with zone maps
        }
    }

    fn engine_type(&self) -> StorageEngineType {
        StorageEngineType::NOVA
    }
}

// Unit tests are minimal since actual extraction is tested in integration tests.
// The extractor trait tests in extractor.rs cover the core functionality.
