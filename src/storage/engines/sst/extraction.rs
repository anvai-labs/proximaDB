//! SST Vector Extractor
//!
//! Implements the VectorExtractor trait for SST engine.
//! Wraps existing UnifiedSstableReader functionality.

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

use super::readers::sst_query_engine::UnifiedSstableReader;

/// SST Vector Extractor
///
/// Extracts vectors from SST files using the UnifiedSstableReader.
/// This is the reference implementation for other engines.
pub struct SstExtractor {
    /// Unified caching filesystem for file access
    filesystem: Arc<UnifiedCachingFilesystem>,
}

impl SstExtractor {
    /// Create a new SST extractor
    pub fn new(filesystem: Arc<UnifiedCachingFilesystem>) -> Self {
        Self { filesystem }
    }
}

#[async_trait]
impl VectorExtractor for SstExtractor {
    async fn extract_vectors(
        &self,
        request: ExtractionRequest,
    ) -> Result<ExtractionResult, ExtractionError> {
        let start = Instant::now();

        if request.file_paths.is_empty() {
            return Ok(ExtractionResult::empty());
        }

        // Convert file:// URLs to local paths if needed
        let file_paths: Vec<String> = request
            .file_paths
            .iter()
            .map(|p| {
                if p.starts_with("file://") {
                    // Strip file:// prefix to get local path
                    p.strip_prefix("file://").unwrap_or(p).to_string()
                } else {
                    p.clone()
                }
            })
            .collect();

        debug!(
            "[SST Extractor] Processing {} files: {:?}",
            file_paths.len(),
            file_paths
        );

        // Filter out non-existent files to handle race conditions gracefully
        // (e.g., temp directories cleaned up before AXIS consumer processes events)
        let mut existing_files = Vec::with_capacity(file_paths.len());
        let mut missing_count = 0;
        for path in &file_paths {
            if std::path::Path::new(path).exists() {
                existing_files.push(path.clone());
            } else {
                missing_count += 1;
                debug!("[SST Extractor] File not found (skipping): {}", path);
            }
        }

        if missing_count > 0 {
            tracing::warn!(
                "[SST Extractor] {} of {} files not found (temp dir cleanup race?)",
                missing_count,
                file_paths.len()
            );
        }

        if existing_files.is_empty() {
            debug!("[SST Extractor] No existing files to process, returning empty result");
            return Ok(ExtractionResult::empty());
        }

        let file_paths = existing_files;

        // Create filesystem factory for this extraction operation
        let filesystem_factory = Arc::new(
            FilesystemFactory::create_default()
                .await
                .map_err(|e| ExtractionError::IoError(e.to_string()))?,
        );

        // Create the SST reader
        let reader = UnifiedSstableReader::new(
            filesystem_factory,
            self.filesystem.clone(),
            "extractor".to_string(),
        );

        // Use the existing read_all_records_for_compaction method
        let records = reader
            .read_all_records_for_compaction(&file_paths)
            .await
            .map_err(|e| ExtractionError::EngineError(format!("SST read error: {}", e)))?;

        debug!(
            "[SST Extractor] Read {} records from {} files",
            records.len(),
            request.file_paths.len()
        );

        // Convert canonical records to extracted vectors without touching the v1 proto envelope.
        let mut vectors = Vec::with_capacity(records.len());
        let mut bytes_read = 0usize;

        // ID filter as a set, built once per extraction (was: a linear scan
        // of the requested-id Vec for EVERY record in every block).
        let id_filter: Option<std::collections::HashSet<&String>> =
            request.vector_ids.as_ref().map(|ids| ids.iter().collect());

        for record in records {
            let record_id = record
                .local_id
                .clone()
                .unwrap_or_else(|| record.oid.clone());
            // Apply the ID filter BEFORE any materialization (was: after the
            // props→JSON conversion and the fp32 copy, so excluded records
            // paid the full conversion cost). `bytes_read` below still
            // accounts every scanned vector's fp32 bytes exactly as before.
            let should_include = match &id_filter {
                Some(ids) => ids.contains(&record_id),
                None => true,
            };
            // Handle vector filtering based on mode
            let fp32_vector = match request.mode {
                ExtractionMode::Fp32Only | ExtractionMode::Both | ExtractionMode::Auto => {
                    let vector = record
                        .embeddings
                        .first()
                        .map(|embedding| embedding.as_fp32_slice())
                        .unwrap_or(&[]);
                    if vector.is_empty() {
                        None
                    } else {
                        bytes_read += vector.len() * 4;
                        if should_include {
                            Some(vector.to_vec())
                        } else {
                            None
                        }
                    }
                }
                ExtractionMode::QuantizedOnly => None,
            };
            if !should_include {
                continue;
            }

            // SST doesn't store quantized vectors inline in VectorRecord
            // Quantization is handled at the storage layer
            let quantized = None;

            let metadata = if !record.props.is_empty() {
                let json_map: serde_json::Map<String, serde_json::Value> = record
                    .props
                    .iter()
                    .map(|(k, node)| {
                        // REHYDRATION format, not an API rendering: the
                        // AXIS consumer re-parses this JSON back into
                        // ProximaValue (Value::String stays String, arrays
                        // stay arrays), so the FILTER spelling keeps
                        // extraction-recovered records filtering IDENTICALLY
                        // to live ones (a Binary renders as the int-array on
                        // both sides; the canonical base64 string would
                        // rehydrate as String and silently diverge). Display
                        // surfaces read through the canonical converter, not
                        // this extractor. KNOWN family divergence (tracked
                        // in TD-PROTO-2): helix/swift still render the
                        // canonical spelling and raptor its raw prost
                        // envelope — all feed the same AXIS rehydration, so
                        // recovered-record filtering diverges per engine
                        // until the family converges here.
                        let value = match node {
                            proximadb_records::ProximaTreeNode::Value(value) => {
                                crate::core::search::sql_value_filter::proxima_value_to_json(value)
                            }
                            proximadb_records::ProximaTreeNode::Object(subtree) => {
                                let object =
                                    crate::core::search::sql_value_filter::proxima_tree_to_json_map(
                                        subtree,
                                    );
                                serde_json::Value::Object(object.into_iter().collect())
                            }
                        };
                        (k.clone(), value)
                    })
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
                vectors.push(ExtractedVector {
                    id: record_id,
                    fp32_vector,
                    quantized,
                    metadata,
                });
            }
        }

        // Apply limit and offset if specified
        let total_before_pagination = vectors.len();
        if let Some(offset) = request.offset {
            if offset < vectors.len() {
                vectors = vectors.into_iter().skip(offset).collect();
            } else {
                vectors.clear();
            }
        }
        if let Some(limit) = request.limit {
            vectors.truncate(limit);
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        debug!(
            "[SST Extractor] Extracted {} vectors in {}ms",
            vectors.len(),
            duration_ms
        );

        Ok(ExtractionResult {
            vectors,
            stats: ExtractionStats {
                vectors_extracted: total_before_pagination,
                bytes_read,
                files_processed: request.file_paths.len(),
                duration_ms,
                vectors_skipped: 0,
            },
            continuation_token: None,
        })
    }

    fn extraction_capabilities(&self) -> ExtractionCapabilities {
        ExtractionCapabilities {
            supports_streaming: true,
            supports_incremental: true,
            available_modes: vec![
                ExtractionMode::Fp32Only,
                ExtractionMode::QuantizedOnly,
                ExtractionMode::Both,
                ExtractionMode::Auto,
            ],
            supports_metadata: true,
            optimal_batch_size: 10_000,
        }
    }

    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost {
        // Estimate based on file count and typical SST file characteristics
        let files = request.file_paths.len();
        let vectors_per_file = 10_000; // Typical SST file size
        let bytes_per_vector = 768 * 4; // Typical dimension * f32 size

        ExtractionCost {
            estimated_vectors: files * vectors_per_file,
            estimated_bytes: files * vectors_per_file * bytes_per_vector,
            estimated_duration_ms: (files * 50) as u64, // ~50ms per file
            io_cost: 1.0,                               // SST is the baseline
        }
    }

    fn engine_type(&self) -> StorageEngineType {
        StorageEngineType::SST
    }
}

// Unit tests are minimal since actual extraction is tested in integration tests.
// The extractor trait tests in extractor.rs cover the core functionality.
