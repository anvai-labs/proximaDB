//! HELIX Vector Extractor
//!
//! Implements the VectorExtractor trait for HELIX engine.
//! Reads vectors from HELIX SSTable files which use ProximaBlocks format
//! with Hilbert curve spatial clustering.

use async_trait::async_trait;
use std::io::Read;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

use crate::core::types::StorageEngineType;
use crate::storage::engines::core::formats::proximablocks::ProximaDataBlock;
use crate::storage::persistence::filesystem::FileSystem;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use crate::storage::trait_components::extractor::{
    ExtractedVector, ExtractionCapabilities, ExtractionCost, ExtractionError, ExtractionMode,
    ExtractionRequest, ExtractionResult, ExtractionStats, VectorExtractor,
};

/// HELIX Vector Extractor
///
/// Extracts vectors from HELIX SSTable files.
/// HELIX uses Hilbert curve spatial clustering for locality-optimized storage.
pub struct HelixExtractor {
    /// Unified caching filesystem for file access
    filesystem: Arc<UnifiedCachingFilesystem>,
}

impl HelixExtractor {
    /// Create a new HELIX extractor
    pub fn new(filesystem: Arc<UnifiedCachingFilesystem>) -> Self {
        Self { filesystem }
    }

    /// Read all records from a HELIX SSTable file
    async fn read_sstable(
        &self,
        path: &str,
    ) -> Result<Vec<crate::proto::proximadb_v1::VectorRecord>, ExtractionError> {
        // Read file data
        let file_data = self.filesystem.read(path).await.map_err(|e| {
            ExtractionError::IoError(format!("Failed to read HELIX file {}: {}", path, e))
        })?;

        let mut cursor = std::io::Cursor::new(file_data);

        // Skip magic (4 bytes) and version (4 bytes)
        cursor.set_position(8);

        // Read number of blocks
        let mut num_blocks_bytes = [0u8; 4];
        cursor.read_exact(&mut num_blocks_bytes).map_err(|e| {
            ExtractionError::ParseError(format!("Failed to read block count: {}", e))
        })?;
        let num_blocks = u32::from_le_bytes(num_blocks_bytes);

        let mut records = Vec::new();

        // Read blocks
        for block_idx in 0..num_blocks {
            // Read block size
            let mut size_bytes = [0u8; 4];
            if cursor.read_exact(&mut size_bytes).is_err() {
                debug!("Reached end of file at block {}", block_idx);
                break;
            }
            let block_size = u32::from_le_bytes(size_bytes) as usize;

            if block_size == 0 {
                continue;
            }

            // Read block data
            let mut block_data = vec![0u8; block_size];
            if cursor.read_exact(&mut block_data).is_err() {
                debug!("Incomplete block {} at end of file", block_idx);
                break;
            }

            // Deserialize block
            match ProximaDataBlock::deserialize(&block_data, None) {
                Ok(block) => {
                    records.extend(
                        block
                            .records
                            .iter()
                            .map(crate::proto::proximadb_v1::VectorRecord::from),
                    );
                }
                Err(e) => {
                    debug!("Failed to deserialize block {}: {}", block_idx, e);
                    // Continue with other blocks
                }
            }
        }

        Ok(records)
    }
}

#[async_trait]
impl VectorExtractor for HelixExtractor {
    async fn extract_vectors(
        &self,
        request: ExtractionRequest,
    ) -> Result<ExtractionResult, ExtractionError> {
        let start = Instant::now();

        if request.file_paths.is_empty() {
            return Ok(ExtractionResult::empty());
        }

        // Filter out non-existent files to handle race conditions gracefully
        // (e.g., temp directories cleaned up before AXIS consumer processes events).
        // Use the configured filesystem: std::path::Path::exists would classify every
        // object-store URL as missing and silently skip extraction.
        let mut existing_files = Vec::with_capacity(request.file_paths.len());
        let mut missing_count = 0;
        for path in &request.file_paths {
            match self.filesystem.exists(path).await {
                Ok(true) => existing_files.push(path.clone()),
                Ok(false) => {
                    missing_count += 1;
                    debug!("[HELIX Extractor] File not found (skipping): {}", path);
                }
                Err(error) => {
                    return Err(ExtractionError::IoError(format!(
                        "Failed to check HELIX file {}: {}",
                        path, error
                    )));
                }
            }
        }

        if missing_count > 0 {
            tracing::warn!(
                "[HELIX Extractor] {} of {} files not found (temp dir cleanup race?)",
                missing_count,
                request.file_paths.len()
            );
        }

        if existing_files.is_empty() {
            debug!("[HELIX Extractor] No existing files to process, returning empty result");
            return Ok(ExtractionResult::empty());
        }

        let mut all_vectors = Vec::new();
        let mut total_bytes_read = 0usize;
        let mut files_processed = 0usize;

        for file_path in &existing_files {
            let records = self.read_sstable(file_path).await?;

            debug!(
                "[HELIX Extractor] Read {} records from {}",
                records.len(),
                file_path
            );

            // Track bytes (approximate)
            for record in &records {
                total_bytes_read += record.vector.len() * 4;
            }
            files_processed += 1;

            // Convert legacy VectorRecord reader output to ExtractedVector
            for record in records {
                let record_id = record.id.clone();
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

                // HELIX doesn't store quantized vectors inline in VectorRecord
                // Quantization is handled at the storage layer via ProximaDataBlock's QuantizedSection
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

                // Apply ID filter if selective extraction
                let should_include = match &request.vector_ids {
                    Some(ids) => ids.contains(&record_id),
                    None => true,
                };

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
            "[HELIX Extractor] Extracted {} vectors from {} files in {}ms",
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
            supports_incremental: false, // HELIX doesn't have LSN-based incremental
            available_modes: vec![
                ExtractionMode::Fp32Only,
                ExtractionMode::QuantizedOnly,
                ExtractionMode::Both,
                ExtractionMode::Auto,
            ],
            supports_metadata: true,
            optimal_batch_size: 20_000, // HELIX uses spatial clustering
        }
    }

    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost {
        // HELIX has moderate I/O due to spatial clustering
        let files = request.file_paths.len();
        let vectors_per_file = 20_000;
        let bytes_per_vector = 768 * 4;

        ExtractionCost {
            estimated_vectors: files * vectors_per_file,
            estimated_bytes: files * vectors_per_file * bytes_per_vector,
            estimated_duration_ms: (files * 60) as u64, // ~60ms per file
            io_cost: 0.9, // Slightly better than SST due to spatial locality
        }
    }

    fn engine_type(&self) -> StorageEngineType {
        StorageEngineType::HELIX
    }
}

// Unit tests are minimal since actual extraction is tested in integration tests.
// The extractor trait tests in extractor.rs cover the core functionality.
