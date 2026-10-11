//! SWIFT Vector Extractor
//!
//! Implements the VectorExtractor trait for SWIFT engine.
//! Uses UnifiedSwiftReader with StreamAll strategy for efficient extraction.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

use crate::core::types::StorageEngineType;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use crate::storage::trait_components::extractor::{
    ExtractedVector, ExtractionCapabilities, ExtractionCost, ExtractionError, ExtractionMode,
    ExtractionRequest, ExtractionResult, ExtractionStats, VectorExtractor,
};

use super::proximablocks_compact_reader::{
    SwiftReadStrategy, SwiftReaderConfig, UnifiedSwiftReader,
};

/// SWIFT Vector Extractor
///
/// Extracts vectors from SWIFT files using UnifiedSwiftReader with StreamAll strategy.
/// SWIFT's hierarchical superblock structure provides efficient batch extraction.
pub struct SwiftExtractor {
    /// Unified caching filesystem for file access
    filesystem: Arc<UnifiedCachingFilesystem>,
}

impl SwiftExtractor {
    /// Create a new SWIFT extractor
    pub fn new(filesystem: Arc<UnifiedCachingFilesystem>) -> Self {
        Self { filesystem }
    }
}

#[async_trait]
impl VectorExtractor for SwiftExtractor {
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
                debug!("[SWIFT Extractor] File not found (skipping): {}", path);
            }
        }

        if missing_count > 0 {
            tracing::warn!(
                "[SWIFT Extractor] {} of {} files not found (temp dir cleanup race?)",
                missing_count,
                request.file_paths.len()
            );
        }

        if existing_files.is_empty() {
            debug!("[SWIFT Extractor] No existing files to process, returning empty result");
            return Ok(ExtractionResult::empty());
        }

        // Create filesystem factory for this extraction operation
        let filesystem_factory = Arc::new(
            FilesystemFactory::create_default()
                .await
                .map_err(|e| ExtractionError::IoError(e.to_string()))?,
        );

        let mut all_vectors = Vec::new();
        let mut total_bytes_read = 0u64;
        let mut files_processed = 0usize;

        for file_path in &existing_files {
            let config = SwiftReaderConfig {
                enable_prefetch: true,
                max_concurrent_reads: 4,
                coalesce_threshold_bytes: 64 * 1024, // 64KB
                cache_metadata: false,               // Disable for extraction
                streaming_threshold_mb: 10,          // Stream if file > 10MB
            };

            let reader = UnifiedSwiftReader::new(
                filesystem_factory.clone(),
                file_path.to_string(),
                self.filesystem.clone(),
                "extraction".to_string(),
                config,
            )
            .await
            .map_err(|e| ExtractionError::EngineError(format!("SWIFT reader error: {}", e)))?;

            // Use StreamAll strategy for full extraction
            let read_result = reader
                .read_with_strategy(SwiftReadStrategy::StreamAll)
                .await
                .map_err(|e| ExtractionError::EngineError(format!("SWIFT read error: {}", e)))?;

            total_bytes_read += read_result.bytes_read;
            files_processed += 1;

            debug!(
                "[SWIFT Extractor] Read {} records from {}",
                read_result.records.len(),
                file_path
            );

            // Convert legacy VectorRecord reader output to ExtractedVector
            // ID filter as a set, built once per extraction call (was: a
            // linear scan of the requested-id Vec for every record).
            let id_filter: Option<std::collections::HashSet<&String>> =
                request.vector_ids.as_ref().map(|ids| ids.iter().collect());
            for record in read_result.records {
                let record_id = record.id.clone();
                // Apply the ID filter BEFORE any materialization (was: after
                // the fp32 copy and metadata conversion).
                let should_include = match &id_filter {
                    Some(ids) => ids.contains(&record_id),
                    None => true,
                };
                if !should_include {
                    continue;
                }
                // Handle vector filtering based on mode
                let fp32_vector = match request.mode {
                    ExtractionMode::Fp32Only | ExtractionMode::Both | ExtractionMode::Auto => {
                        if record.vector.is_empty() {
                            None
                        } else {
                            Some(record.vector.clone())
                        }
                    }
                    ExtractionMode::QuantizedOnly => None,
                };

                // SWIFT doesn't store quantized vectors inline in VectorRecord
                // Quantization is handled at the storage layer via superblock structure
                let quantized = None;

                // Handle metadata - convert HashMap<String, SqlValue> to JSON Value
                let metadata = if !record.metadata.is_empty() {
                    // Round 16: delegate to the shared converter — the
                    // private helper rendered bytes as hex and arrays/objects
                    // as placeholder strings, drifting per-surface. Unset
                    // oneofs lower to JSON null (same semantics, one home).
                    let json_map: serde_json::Map<String, serde_json::Value> = record
                        .metadata
                        .into_iter()
                        .map(|(k, v)| (k, proximadb_records::conversions::sql_value_to_json(&v)))
                        .collect();
                    // (Non-empty by the outer guard; the converter always
                    // yields a value per key — the old filter_map could drop.)
                    Some(serde_json::Value::Object(json_map))
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
            "[SWIFT Extractor] Extracted {} vectors from {} files in {}ms",
            all_vectors.len(),
            files_processed,
            duration_ms
        );

        Ok(ExtractionResult {
            vectors: all_vectors,
            stats: ExtractionStats {
                vectors_extracted: total_before_pagination,
                bytes_read: total_bytes_read as usize,
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
            supports_incremental: false, // SWIFT doesn't have LSN-based incremental
            available_modes: vec![
                ExtractionMode::Fp32Only,
                ExtractionMode::QuantizedOnly,
                ExtractionMode::Both,
                ExtractionMode::Auto,
            ],
            supports_metadata: true,
            optimal_batch_size: 50_000, // SWIFT handles larger batches due to superblock structure
        }
    }

    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost {
        // SWIFT is optimized for batch reads due to superblock structure
        let files = request.file_paths.len();
        let vectors_per_file = 50_000; // Larger due to superblock structure
        let bytes_per_vector = 768 * 4; // Typical dimension * f32 size

        ExtractionCost {
            estimated_vectors: files * vectors_per_file,
            estimated_bytes: files * vectors_per_file * bytes_per_vector,
            estimated_duration_ms: (files * 100) as u64, // ~100ms per file (larger files)
            io_cost: 0.8, // SWIFT is more efficient than SST for batch reads
        }
    }

    fn engine_type(&self) -> StorageEngineType {
        StorageEngineType::SWIFT
    }
}

// Unit tests are minimal since actual extraction is tested in integration tests.
// The extractor trait tests in extractor.rs cover the core functionality.
