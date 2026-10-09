//! # Vector Extractor Trait
//!
//! Unified protocol for extracting vectors from storage files.
//! This enables AXIS index building, compaction, analytics, and future use cases.
//!
//! ## Design Goals
//!
//! 1. **Engine-Agnostic**: Same interface for SST, HELIX, VIPER, NOVA, SWIFT, RAPTOR
//! 2. **Flexible Extraction**: Support full, incremental, and selective extraction
//! 3. **Format Awareness**: Handle FP32, INT8, Binary, and PQ vectors
//! 4. **Cost Estimation**: Enable query planning with extraction cost estimates
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::storage::trait_components::extractor::{ExtractionFactory, ExtractionRequest};
//!
//! let extractor = ExtractionFactory::create(StorageEngineType::SST, filesystem);
//! let result = extractor.extract_vectors(request).await?;
//! ```

use async_trait::async_trait;
use std::sync::Arc;
use thiserror::Error;

use crate::core::types::StorageEngineType;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;

/// Unified interface for extracting vectors from storage files.
///
/// Each storage engine implements this trait to provide vector extraction
/// capabilities for AXIS index building and other use cases.
#[async_trait]
pub trait VectorExtractor: Send + Sync {
    /// Extract vectors from specified files.
    ///
    /// # Arguments
    /// * `request` - Parameters specifying what to extract
    ///
    /// # Returns
    /// * `Ok(ExtractionResult)` - Extracted vectors with metadata
    /// * `Err(ExtractionError)` - If extraction fails
    async fn extract_vectors(
        &self,
        request: ExtractionRequest,
    ) -> Result<ExtractionResult, ExtractionError>;

    /// Get capabilities of this extractor.
    ///
    /// Used for query planning and feature discovery.
    fn extraction_capabilities(&self) -> ExtractionCapabilities;

    /// Estimate cost/time for extraction.
    ///
    /// Used for query planning to choose optimal extraction strategy.
    fn estimate_extraction_cost(&self, request: &ExtractionRequest) -> ExtractionCost;

    /// Get the engine type this extractor is for.
    fn engine_type(&self) -> StorageEngineType;
}

/// Request parameters for vector extraction.
#[derive(Debug, Clone)]
pub struct ExtractionRequest {
    /// Files to extract from (SST files, Parquet files, etc.)
    pub file_paths: Vec<String>,

    /// Extraction scope
    pub scope: ExtractionScope,

    /// Desired vector format
    pub mode: ExtractionMode,

    /// Optional: specific vector IDs to extract
    pub vector_ids: Option<Vec<String>>,

    /// Optional: time range filter (for incremental extraction)
    pub time_range: Option<(u64, u64)>,

    /// Maximum vectors to extract (for pagination)
    pub limit: Option<usize>,

    /// Offset for pagination
    pub offset: Option<usize>,
}

impl ExtractionRequest {
    /// Create a full extraction request for all vectors in files.
    pub fn full(file_paths: Vec<String>) -> Self {
        Self {
            file_paths,
            scope: ExtractionScope::Full,
            mode: ExtractionMode::Fp32Only,
            vector_ids: None,
            time_range: None,
            limit: None,
            offset: None,
        }
    }

    /// Create an incremental extraction request since a given LSN.
    pub fn incremental(file_paths: Vec<String>, last_lsn: u64) -> Self {
        Self {
            file_paths,
            scope: ExtractionScope::Incremental { last_lsn },
            mode: ExtractionMode::Fp32Only,
            vector_ids: None,
            time_range: None,
            limit: None,
            offset: None,
        }
    }

    /// Create a selective extraction request for specific vector IDs.
    pub fn selective(file_paths: Vec<String>, vector_ids: Vec<String>) -> Self {
        Self {
            file_paths,
            scope: ExtractionScope::Selective,
            mode: ExtractionMode::Fp32Only,
            vector_ids: Some(vector_ids),
            time_range: None,
            limit: None,
            offset: None,
        }
    }

    /// Set the extraction mode.
    pub fn with_mode(mut self, mode: ExtractionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a limit on extracted vectors.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set an offset for pagination.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// What scope of vectors to extract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionScope {
    /// Full extraction - all vectors in files (for index rebuild)
    Full,
    /// Incremental - only new vectors since last extraction
    Incremental {
        /// Last processed LSN
        last_lsn: u64,
    },
    /// Selective - specific vector IDs only
    Selective,
}

/// Vector format preference for extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionMode {
    /// Only FP32 vectors
    Fp32Only,
    /// Only quantized (INT8, Binary, PQ)
    QuantizedOnly,
    /// Both FP32 and quantized
    Both,
    /// Let extractor decide based on availability
    Auto,
}

/// Result of vector extraction.
#[derive(Debug)]
pub struct ExtractionResult {
    /// Extracted vectors with IDs
    pub vectors: Vec<ExtractedVector>,

    /// Statistics about the extraction
    pub stats: ExtractionStats,

    /// Continuation token for pagination (if more results available)
    pub continuation_token: Option<String>,
}

impl ExtractionResult {
    /// Create an empty extraction result.
    pub fn empty() -> Self {
        Self {
            vectors: Vec::new(),
            stats: ExtractionStats::default(),
            continuation_token: None,
        }
    }

    /// Get the number of extracted vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if no vectors were extracted.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Convert to (id, vector) tuples for AXIS index building.
    pub fn into_id_vector_pairs(self) -> Vec<(String, Vec<f32>)> {
        self.vectors
            .into_iter()
            .filter_map(|v| v.fp32_vector.map(|vec| (v.id, vec)))
            .collect()
    }
}

/// A single extracted vector with all available representations.
#[derive(Debug, Clone)]
pub struct ExtractedVector {
    /// Unique vector ID
    pub id: String,
    /// FP32 vector (if available/requested)
    pub fp32_vector: Option<Vec<f32>>,
    /// Quantized representation (if available/requested)
    pub quantized: Option<QuantizedVector>,
    /// Vector metadata (if available/requested)
    pub metadata: Option<serde_json::Value>,
}

impl ExtractedVector {
    /// Create a new extracted vector with just FP32 data.
    pub fn new_fp32(id: String, vector: Vec<f32>) -> Self {
        Self {
            id,
            fp32_vector: Some(vector),
            quantized: None,
            metadata: None,
        }
    }

    /// Create a new extracted vector with FP32 and metadata.
    pub fn new_with_metadata(id: String, vector: Vec<f32>, metadata: serde_json::Value) -> Self {
        Self {
            id,
            fp32_vector: Some(vector),
            quantized: None,
            metadata: Some(metadata),
        }
    }

    /// Add quantized representation.
    pub fn with_quantized(mut self, quantized: QuantizedVector) -> Self {
        self.quantized = Some(quantized);
        self
    }
}

/// Quantized vector representations.
#[derive(Debug, Clone)]
pub enum QuantizedVector {
    /// Binary quantized (1 bit per dimension)
    Binary(Vec<u8>),
    /// INT8 scalar quantized
    Int8(Vec<i8>),
    /// Product quantized
    Pq {
        /// PQ codes
        codes: Vec<u8>,
        /// Bits per subvector
        subvector_bits: u8,
    },
}

/// Statistics about an extraction operation.
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Total vectors extracted
    pub vectors_extracted: usize,
    /// Total bytes read
    pub bytes_read: usize,
    /// Files processed
    pub files_processed: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Vectors skipped (filtered out)
    pub vectors_skipped: usize,
}

/// Capabilities of a vector extractor.
#[derive(Debug, Clone)]
pub struct ExtractionCapabilities {
    /// Supports streaming extraction (memory-efficient for large files)
    pub supports_streaming: bool,
    /// Supports incremental extraction (only new vectors)
    pub supports_incremental: bool,
    /// Available extraction modes
    pub available_modes: Vec<ExtractionMode>,
    /// Can extract metadata alongside vectors
    pub supports_metadata: bool,
    /// Optimal batch size for this extractor
    pub optimal_batch_size: usize,
}

impl Default for ExtractionCapabilities {
    fn default() -> Self {
        Self {
            supports_streaming: false,
            supports_incremental: false,
            available_modes: vec![ExtractionMode::Fp32Only],
            supports_metadata: false,
            optimal_batch_size: 10_000,
        }
    }
}

/// Cost estimate for extraction planning.
#[derive(Debug, Clone, Default)]
pub struct ExtractionCost {
    /// Estimated number of vectors
    pub estimated_vectors: usize,
    /// Estimated bytes to read
    pub estimated_bytes: usize,
    /// Estimated duration in milliseconds
    pub estimated_duration_ms: u64,
    /// Relative I/O cost (1.0 = baseline)
    pub io_cost: f64,
}

/// Errors that can occur during vector extraction.
#[derive(Debug, Error)]
pub enum ExtractionError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Unsupported format for extraction: {0}")]
    UnsupportedFormat(String),

    #[error("IO error during extraction: {0}")]
    IoError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Engine-specific error: {0}")]
    EngineError(String),

    #[error("Extraction cancelled")]
    Cancelled,
}

impl From<std::io::Error> for ExtractionError {
    fn from(err: std::io::Error) -> Self {
        ExtractionError::IoError(err.to_string())
    }
}

impl From<anyhow::Error> for ExtractionError {
    fn from(err: anyhow::Error) -> Self {
        ExtractionError::EngineError(err.to_string())
    }
}

/// Factory for creating extractors based on engine type.
///
/// This provides a unified way to get the appropriate extractor
/// for any storage engine type.
pub struct ExtractionFactory;

impl ExtractionFactory {
    /// Create an extractor for the given engine type.
    ///
    /// # Arguments
    /// * `engine_type` - The storage engine type
    /// * `filesystem` - Unified filesystem for file access
    ///
    /// # Returns
    /// An Arc-wrapped extractor implementing VectorExtractor
    pub fn create(
        engine_type: StorageEngineType,
        filesystem: Arc<UnifiedCachingFilesystem>,
    ) -> Arc<dyn VectorExtractor> {
        match engine_type {
            StorageEngineType::SST => Arc::new(
                crate::storage::engines::sst::extraction::SstExtractor::new(filesystem),
            ),
            #[cfg(feature = "experimental-engines")]
            StorageEngineType::SWIFT => Arc::new(
                crate::storage::engines::swift::extraction::SwiftExtractor::new(filesystem),
            ),
            // SWIFT requires `experimental-engines`; unreachable without it.
            #[cfg(not(feature = "experimental-engines"))]
            StorageEngineType::SWIFT => Arc::new(
                crate::storage::engines::sst::extraction::SstExtractor::new(filesystem),
            ),
            StorageEngineType::HELIX => Arc::new(
                crate::storage::engines::helix::extraction::HelixExtractor::new(filesystem),
            ),
            StorageEngineType::VIPER => Arc::new(
                crate::storage::engines::viper::extraction::ViperExtractor::new(filesystem),
            ),
            StorageEngineType::NOVA => {
                Arc::new(crate::storage::engines::nova::extraction::NovaExtractor::new(filesystem))
            }
            #[cfg(feature = "experimental-engines")]
            StorageEngineType::RAPTOR => Arc::new(
                crate::storage::engines::raptor::extraction::RaptorExtractor::new(filesystem),
            ),
            // RAPTOR requires `experimental-engines`; unreachable on a build without
            // it, so fall back to the SST extractor to keep the match exhaustive.
            #[cfg(not(feature = "experimental-engines"))]
            StorageEngineType::RAPTOR => Arc::new(
                crate::storage::engines::sst::extraction::SstExtractor::new(filesystem),
            ),
            StorageEngineType::TST => Arc::new(
                crate::storage::engines::tst::extraction::TstExtractor::new(filesystem),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_request_full() {
        let request = ExtractionRequest::full(vec!["file1.sst".to_string()]);
        assert_eq!(request.scope, ExtractionScope::Full);
        assert_eq!(request.mode, ExtractionMode::Fp32Only);
        assert_eq!(request.file_paths.len(), 1);
    }

    #[test]
    fn test_extraction_request_incremental() {
        let request = ExtractionRequest::incremental(vec!["file1.sst".to_string()], 12345);
        assert_eq!(
            request.scope,
            ExtractionScope::Incremental { last_lsn: 12345 }
        );
    }

    #[test]
    fn test_extraction_result_into_pairs() {
        let result = ExtractionResult {
            vectors: vec![
                ExtractedVector::new_fp32("v1".to_string(), vec![1.0, 2.0]),
                ExtractedVector::new_fp32("v2".to_string(), vec![3.0, 4.0]),
            ],
            stats: ExtractionStats::default(),
            continuation_token: None,
        };

        let pairs = result.into_id_vector_pairs();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "v1");
        assert_eq!(pairs[0].1, vec![1.0, 2.0]);
    }

    #[test]
    fn test_extracted_vector_with_metadata() {
        let metadata = serde_json::json!({"key": "value"});
        let vector =
            ExtractedVector::new_with_metadata("v1".to_string(), vec![1.0, 2.0], metadata.clone());

        assert_eq!(vector.id, "v1");
        assert!(vector.fp32_vector.is_some());
        assert!(vector.metadata.is_some());
        assert_eq!(vector.metadata.unwrap(), metadata);
    }
}
