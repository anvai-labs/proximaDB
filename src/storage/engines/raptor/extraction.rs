//! RAPTOR Vector Extractor
//!
//! Implements the VectorExtractor trait for RAPTOR engine.
//! RAPTOR uses adaptive row-groups with Matrix Trinity (K×K, P×K, P²) navigation.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

use crate::core::types::StorageEngineType;
use crate::storage::cache::orchestrator::CrossCacheOrchestrator;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use crate::storage::trait_components::extractor::{
    ExtractedVector, ExtractionCapabilities, ExtractionCost, ExtractionError, ExtractionMode,
    ExtractionRequest, ExtractionResult, ExtractionStats, VectorExtractor,
};

use super::RaptorConfig;
use super::consolidated_reader::{RaptorReader, ScanStrategy};

/// RAPTOR Vector Extractor
///
/// Extracts vectors from RAPTOR files using the consolidated reader.
/// RAPTOR is an adaptive engine with multi-tier row-group storage.
pub struct RaptorExtractor {
    /// Unified caching filesystem for file access
    filesystem: Arc<UnifiedCachingFilesystem>,
}

impl RaptorExtractor {
    /// Create a new RAPTOR extractor
    pub fn new(filesystem: Arc<UnifiedCachingFilesystem>) -> Self {
        Self { filesystem }
    }
}

#[async_trait]
impl VectorExtractor for RaptorExtractor {
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
        let mut existing_files = Vec::with_capacity(request.file_paths.len());
        let mut missing_count = 0;
        for path in &request.file_paths {
            // Strip file:// prefix if present
            let local_path = if path.starts_with("file://") {
                path.strip_prefix("file://").unwrap_or(path)
            } else {
                path.as_str()
            };
            if std::path::Path::new(local_path).exists() {
                existing_files.push(path.clone());
            } else {
                missing_count += 1;
                debug!("[RAPTOR Extractor] File not found (skipping): {}", path);
            }
        }

        if missing_count > 0 {
            tracing::warn!(
                "[RAPTOR Extractor] {} of {} files not found (temp dir cleanup race?)",
                missing_count,
                request.file_paths.len()
            );
        }

        if existing_files.is_empty() {
            debug!("[RAPTOR Extractor] No existing files to process, returning empty result");
            return Ok(ExtractionResult::empty());
        }

        // ID filter as a set, built once per extraction (was: a linear
        // `ids.contains` per record — TD-DSEFF-1 parity with sst/nova/swift).
        let id_filter = crate::storage::trait_components::extractor::build_id_filter(
            request.vector_ids.as_ref(),
        );

        let mut all_vectors = Vec::new();
        let mut total_bytes_read = 0usize;
        let mut files_processed = 0usize;

        // Create shared dependencies for RaptorReader
        let cache = Arc::new(CrossCacheOrchestrator::new(1000));
        let config = RaptorConfig::default();

        for file_path in &existing_files {
            debug!("[RAPTOR Extractor] Processing file: {}", file_path);

            // Create a RaptorReader for this file
            let mut reader = RaptorReader::new(
                file_path.clone(),        // base_path = file path
                "extraction".to_string(), // collection_id for cache keys
                config.clone(),
                cache.clone(),
                self.filesystem.clone(),
            );

            // Use full scan strategy for extraction
            let records = reader
                .scan_vectors_with_strategy(file_path, ScanStrategy::FullScan)
                .await
                .map_err(|e| ExtractionError::EngineError(format!("RAPTOR scan error: {}", e)))?;

            debug!(
                "[RAPTOR Extractor] Read {} records from {}",
                records.len(),
                file_path
            );

            // Track bytes (approximate)
            for record in &records {
                total_bytes_read += record.vector.len() * 4;
            }
            files_processed += 1;

            // Convert VectorRecord to ExtractedVector
            for record in records {
                // Handle vector filtering based on mode
                let fp32_vector = match request.mode {
                    ExtractionMode::Fp32Only | ExtractionMode::Both | ExtractionMode::Auto => {
                        if !record.vector.is_empty() {
                            Some(record.vector)
                        } else {
                            None
                        }
                    }
                    ExtractionMode::QuantizedOnly => None,
                };

                // RAPTOR doesn't store quantized vectors inline in VectorRecord
                // Quantization is handled at the storage layer via ProximaCodec
                let quantized = None;

                // Handle metadata - convert HashMap to JSON Value
                let metadata = if !record.metadata.is_empty() {
                    Some(serde_json::to_value(&record.metadata).unwrap_or(serde_json::Value::Null))
                } else {
                    None
                };

                let should_include = match &id_filter {
                    Some(ids) => ids.contains(&record.id),
                    None => true,
                };

                if should_include && fp32_vector.is_some() {
                    all_vectors.push(ExtractedVector {
                        id: record.id,
                        fp32_vector,
                        quantized,
                        metadata,
                    });
                }
            }
        }

        // Apply limit and offset if specified
        let total_before_pagination = all_vectors.len();
        if let Some(offset) = request.offset {
            if offset < all_vectors.len() {
                all_vectors = all_vectors.into_iter().skip(offset).collect();
            } else {
                all_vectors.clear();
            }
        }
        if let Some(limit) = request.limit {
            all_vectors.truncate(limit);
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        debug!(
            "[RAPTOR Extractor] Extracted {} vectors from {} files in {}ms",
            all_vectors.len(),
            files_processed,
            duration_ms
        );

        Ok(ExtractionResult {
            vectors: all_vectors,
            stats: ExtractionStats {
                vectors_extracted: total_before_pagination,
                bytes_read: total_bytes_read,
                files_processed,
                duration_ms,
                vectors_skipped: 0,
            },
            continuation_token: None,
        })
    }

    fn extraction_capabilities(&self) -> ExtractionCapabilities {
        ExtractionCapabilities {
            supports_streaming: true,
            supports_incremental: false, // No LSN-based incremental
            available_modes: vec![
                ExtractionMode::Fp32Only,
                ExtractionMode::QuantizedOnly,
                ExtractionMode::Both,
                ExtractionMode::Auto,
            ],
            supports_metadata: true,
            optimal_batch_size: 30_000, // RAPTOR uses adaptive row-groups
        }
    }

    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost {
        // RAPTOR is highly optimized with Matrix Trinity navigation
        let files = request.file_paths.len();
        let vectors_per_file = 30_000;
        let bytes_per_vector = 768 * 4;

        ExtractionCost {
            estimated_vectors: files * vectors_per_file,
            estimated_bytes: files * vectors_per_file * bytes_per_vector,
            estimated_duration_ms: (files * 80) as u64, // ~80ms per file
            io_cost: 0.6,                               // RAPTOR is very efficient
        }
    }

    fn engine_type(&self) -> StorageEngineType {
        StorageEngineType::RAPTOR
    }
}

// Unit tests are minimal since actual extraction is tested in integration tests.
// The extractor trait tests in extractor.rs cover the core functionality.
