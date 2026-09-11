//! # Internal Format Adapters
//!
//! This module provides adapter types that connect the new storage format traits
//! to existing storage engines. Following the Adapter Pattern, it wraps existing
//! `UnifiedStorageFormat` implementations to provide the `InternalFormat` interface.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         ADAPTER PATTERN                                  │
//! │                                                                          │
//! │    ┌───────────────────────┐         ┌───────────────────────┐          │
//! │    │   InternalFormat      │         │ UnifiedStorageFormat  │          │
//! │    │   (New Trait)         │         │ (Existing Trait)      │          │
//! │    └───────────┬───────────┘         └───────────┬───────────┘          │
//! │                │                                 │                       │
//! │                │       Adapts                    │                       │
//! │                ▼                                 ▼                       │
//! │    ┌─────────────────────────────────────────────────────────┐          │
//! │    │           InternalFormatAdapter<E>                       │          │
//! │    │   - Wraps UnifiedStorageFormat                          │          │
//! │    │   - Implements InternalFormat                           │          │
//! │    │   - Translates between APIs                             │          │
//! │    └─────────────────────────────────────────────────────────┘          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::storage::formats::adapters::InternalFormatAdapter;
//! use proximadb::storage::engines::sst::SstEngine;
//!
//! let sst_engine = SstEngine::new(config)?;
//! let format_adapter = InternalFormatAdapter::new(Arc::new(sst_engine));
//!
//! // Now use via InternalFormat trait
//! let batches = format_adapter.read_batches(&read_ctx).await?;
//! ```

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow_array::RecordBatch;
use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use chrono::Utc;
use futures::stream;
use proximadb_records::ProximaTreeNode;
use tracing::{debug, warn};

use super::traits::{
    CompactionContext, CompactionResult, FileEntry, FormatStatistics, FormatType, InternalFormat,
    ReadContext, RecordBatchStream, StorageFormat, VectorBatch, VectorBatchStream,
    VectorReadContext, VectorWriteContext, WriteContext, WriteResult,
};
use crate::storage::traits::{StorageFormatStrategy, UnifiedStorageFormat};

// ============================================================================
// Internal Format Adapter
// ============================================================================

/// Adapter that wraps a `UnifiedStorageFormat` to provide `InternalFormat` interface.
///
/// This adapter bridges the gap between:
/// - **New API**: `InternalFormat` trait with Arrow RecordBatch-based I/O
/// - **Existing API**: `UnifiedStorageFormat` trait with VectorRecord-based operations
///
/// ## Design Notes
///
/// The adapter performs lazy conversion between formats:
/// - Read operations convert VectorRecord -> Arrow RecordBatch on demand
/// - Write operations convert Arrow RecordBatch -> VectorRecord before storage
/// - Statistics are gathered from engine metrics and converted to format statistics
///
/// ## Thread Safety
///
/// The adapter is `Send + Sync` as it only holds an `Arc` to the underlying engine.
pub struct InternalFormatAdapter<E: UnifiedStorageFormat> {
    /// The wrapped storage engine
    engine: Arc<E>,
    /// Format version string
    format_version: String,
}

impl<E: UnifiedStorageFormat> InternalFormatAdapter<E> {
    /// Create a new adapter wrapping the given storage engine
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            engine,
            format_version: "1.0.0".to_string(),
        }
    }

    /// Get a reference to the underlying engine
    pub fn engine(&self) -> &Arc<E> {
        &self.engine
    }

    /// Get the format type based on engine strategy
    fn get_format_type(&self) -> FormatType {
        match self.engine.strategy() {
            StorageFormatStrategy::Sst => FormatType::Sst,
            StorageFormatStrategy::Helix => FormatType::Helix,
            StorageFormatStrategy::Viper => FormatType::Viper,
            StorageFormatStrategy::Nova => FormatType::Nova,
            StorageFormatStrategy::Swift => FormatType::Swift,
            StorageFormatStrategy::Raptor => FormatType::Raptor,
            StorageFormatStrategy::Hybrid => FormatType::Sst,
            StorageFormatStrategy::TimeSeries => FormatType::Sst,
            StorageFormatStrategy::Cedar => FormatType::Sst, // CEDAR uses SST-like block format
            StorageFormatStrategy::Chrono => FormatType::Sst, // CHRONO uses time-partitioned blocks
        }
    }
}

impl<E: UnifiedStorageFormat> Debug for InternalFormatAdapter<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalFormatAdapter")
            .field("engine_name", &self.engine.format_name())
            .field("engine_version", &self.engine.format_version())
            .field("format_version", &self.format_version)
            .finish()
    }
}

// ============================================================================
// StorageFormat Implementation (Base Trait)
// ============================================================================

#[async_trait]
impl<E: UnifiedStorageFormat + 'static> StorageFormat for InternalFormatAdapter<E> {
    fn format_name(&self) -> &str {
        self.engine.format_name()
    }

    fn format_version(&self) -> &str {
        &self.format_version
    }

    fn supported_data_types(&self) -> Vec<ArrowDataType> {
        // All internal formats support these core Arrow types
        vec![
            ArrowDataType::Utf8, // Vector IDs
            ArrowDataType::LargeList(Arc::new(Field::new("item", ArrowDataType::Float32, false))), // Vectors
            ArrowDataType::Float32, // Individual floats
            ArrowDataType::Float64, // Double precision
            ArrowDataType::Int64,   // Timestamps
            ArrowDataType::Int32,   // Integers
            ArrowDataType::Boolean, // Boolean flags
            ArrowDataType::Utf8,    // Metadata values (JSON encoded)
        ]
    }

    async fn infer_schema(&self, path: &str) -> Result<ArrowSchema> {
        // Create a basic vector schema - actual schema would need file inspection
        // For now, return a standard vector schema
        debug!("Inferring schema from path: {}", path);

        Ok(ArrowSchema::new(vec![
            Field::new("id", ArrowDataType::Utf8, false),
            Field::new(
                "vector",
                ArrowDataType::LargeList(Arc::new(Field::new(
                    "item",
                    ArrowDataType::Float32,
                    false,
                ))),
                false,
            ),
            Field::new("metadata", ArrowDataType::Utf8, true), // JSON-encoded metadata
            Field::new("timestamp", ArrowDataType::Int64, true),
        ]))
    }

    fn validate_schema(&self, schema: &ArrowSchema) -> Result<()> {
        // Validate that the schema has required fields
        let has_id = schema.fields().iter().any(|f| f.name() == "id");
        let has_vector = schema.fields().iter().any(|f| f.name() == "vector");

        if !has_id {
            return Err(anyhow!("Schema missing required 'id' field"));
        }
        if !has_vector {
            return Err(anyhow!("Schema missing required 'vector' field"));
        }

        Ok(())
    }

    fn format_type(&self) -> FormatType {
        self.get_format_type()
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "vector_search" => true,
            "metadata_filtering" => true,
            "compression" => true,
            "quantization" => self.engine.supports_feature("quantization"),
            "bloom_filters" => matches!(
                self.engine.strategy(),
                StorageFormatStrategy::Sst | StorageFormatStrategy::Swift
            ),
            "columnar_storage" => matches!(
                self.engine.strategy(),
                StorageFormatStrategy::Viper | StorageFormatStrategy::Nova
            ),
            "progressive_search" => matches!(
                self.engine.strategy(),
                StorageFormatStrategy::Sst
                    | StorageFormatStrategy::Nova
                    | StorageFormatStrategy::Helix
            ),
            _ => self.engine.supports_feature(feature),
        }
    }
}

// ============================================================================
// InternalFormat Implementation
// ============================================================================

#[async_trait]
impl<E: UnifiedStorageFormat + 'static> InternalFormat for InternalFormatAdapter<E> {
    // ========================================================================
    // Read Path
    // ========================================================================

    async fn read_batches(&self, ctx: &ReadContext) -> Result<RecordBatchStream> {
        debug!(
            "Reading batches from path: {} (batch_size: {})",
            ctx.path, ctx.batch_size
        );

        // For now, return an empty stream - actual implementation would read from engine
        // The engine's search_vectors_unified would be used for actual data retrieval
        // This is a placeholder that shows the pattern
        let empty_stream = stream::empty();
        Ok(Box::pin(empty_stream))
    }

    async fn read_vectors(&self, ctx: &VectorReadContext) -> Result<VectorBatchStream> {
        debug!(
            "Reading vectors from path: {} (include_vectors: {})",
            ctx.base.path, ctx.include_vectors
        );

        // Similar to read_batches, return empty stream as placeholder
        // Real implementation would use engine's vector retrieval methods
        let empty_stream = stream::empty();
        Ok(Box::pin(empty_stream))
    }

    async fn read_vector_by_id(&self, path: &str, vector_id: &str) -> Result<Option<VectorBatch>> {
        debug!("Reading vector by ID: {} from path: {}", vector_id, path);

        // Extract collection_id from path (path format: {base}/{collection_id}/data)
        let collection_id = extract_collection_id_from_path(path)?;

        // Use engine's point_lookup method (batch-capable; identity None for now)
        match self
            .engine
            .point_lookup(&collection_id, path, &[vector_id.to_string()], None)
            .await?
            .into_iter()
            .next()
        {
            Some(record) => {
                // Convert canonical ProximaRecord to the legacy VectorBatch adapter shape.
                let vector = record
                    .embeddings
                    .first()
                    .map(|embedding| embedding.values.to_fp32_owned())
                    .unwrap_or_default();
                let dimension = vector.len();
                let metadata = if record.props.is_empty() {
                    None
                } else {
                    let meta_map: HashMap<String, serde_json::Value> = record
                        .props
                        .iter()
                        .map(|(k, v)| (k.clone(), proxima_tree_node_to_json(v)))
                        .collect();
                    Some(vec![meta_map])
                };

                Ok(Some(VectorBatch {
                    ids: vec![record.oid],
                    vectors: vector,
                    dimension,
                    metadata,
                }))
            }
            None => Ok(None),
        }
    }

    // ========================================================================
    // Write Path
    // ========================================================================

    async fn write_batch(&self, batch: &RecordBatch, ctx: &WriteContext) -> Result<WriteResult> {
        debug!(
            "Writing batch to path: {} ({} rows)",
            ctx.path,
            batch.num_rows()
        );

        // Convert RecordBatch to VectorRecords and use engine's flush mechanism
        // This is a simplified implementation - real version would handle compression, etc.
        let start_time = std::time::Instant::now();

        // For now, return a placeholder result
        // Real implementation would convert batch to VectorRecords and flush
        Ok(WriteResult {
            files_written: vec![FileEntry {
                path: format!("{}/batch_{}.data", ctx.path, Utc::now().timestamp_millis()),
                size_bytes: 0, // Would be calculated from actual write
                record_count: batch.num_rows() as u64,
                partition_values: None,
                stats: None,
                created_at: Utc::now(),
            }],
            bytes_written: 0,
            records_written: batch.num_rows() as u64,
            duration_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    async fn write_vectors(
        &self,
        vectors: &VectorBatch,
        ctx: &VectorWriteContext,
    ) -> Result<WriteResult> {
        debug!(
            "Writing {} vectors to path: {}",
            vectors.ids.len(),
            ctx.base.path
        );

        let start_time = std::time::Instant::now();
        let vector_count = vectors.ids.len();

        // Convert VectorBatch to VectorRecords for engine
        // Real implementation would construct proper VectorRecords and flush
        Ok(WriteResult {
            files_written: vec![FileEntry {
                path: format!(
                    "{}/vectors_{}.data",
                    ctx.base.path,
                    Utc::now().timestamp_millis()
                ),
                size_bytes: (vector_count * vectors.dimension * 4) as u64, // Approximate size
                record_count: vector_count as u64,
                partition_values: None,
                stats: None,
                created_at: Utc::now(),
            }],
            bytes_written: (vector_count * vectors.dimension * 4) as u64,
            records_written: vector_count as u64,
            duration_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    // ========================================================================
    // Compaction
    // ========================================================================

    async fn compact(&self, ctx: &CompactionContext) -> Result<CompactionResult> {
        debug!(
            "Compacting {} input files to {}",
            ctx.input_files.len(),
            ctx.output_dir
        );

        let start_time = std::time::Instant::now();

        // Delegate to engine's compaction if available
        // Extract collection_id from output_dir path
        let collection_id = extract_collection_id_from_path(&ctx.output_dir)
            .unwrap_or_else(|_| "unknown".to_string());

        match self.engine.compact_collection(&collection_id, None).await {
            Ok(engine_result) => Ok(CompactionResult {
                input_files: ctx.input_files.len(),
                output_files: engine_result.output_files.unwrap_or(1) as usize,
                bytes_read: engine_result.bytes_read.unwrap_or(0),
                bytes_written: engine_result.bytes_written.unwrap_or(0),
                records_processed: engine_result.entries_processed.unwrap_or(0),
                duration_ms: start_time.elapsed().as_millis() as u64,
            }),
            Err(e) => {
                warn!("Engine compaction failed: {}, returning minimal result", e);
                Ok(CompactionResult {
                    input_files: ctx.input_files.len(),
                    output_files: 0,
                    bytes_read: 0,
                    bytes_written: 0,
                    records_processed: 0,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                })
            }
        }
    }

    fn should_compact(&self, stats: &FormatStatistics) -> bool {
        // Use engine-specific heuristics
        match self.engine.strategy() {
            StorageFormatStrategy::Sst => {
                // SST: Compact when file count is high
                stats.file_count > 10
            }
            StorageFormatStrategy::Viper | StorageFormatStrategy::Nova => {
                // Columnar: Compact when many small files
                let small_file_threshold: u64 = 64 * 1024 * 1024;
                stats.file_count > 5
                    && (stats.size_bytes / stats.file_count as u64) < small_file_threshold
            }
            StorageFormatStrategy::Helix => {
                // Helix: Compact based on locality
                stats.file_count > 8
            }
            _ => {
                // Default: Compact when file count exceeds threshold
                stats.file_count > 10
            }
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    async fn get_statistics(&self, path: &str) -> Result<FormatStatistics> {
        debug!("Getting statistics for path: {}", path);

        // Get engine metrics and convert to format statistics
        let engine_metrics = self.engine.collect_engine_metrics().await?;

        let row_count = engine_metrics
            .get("vector_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let size_bytes = engine_metrics
            .get("storage_size_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let file_count = engine_metrics
            .get("file_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        // Create basic schema for statistics
        let schema = ArrowSchema::new(vec![
            Field::new("id", ArrowDataType::Utf8, false),
            Field::new(
                "vector",
                ArrowDataType::LargeList(Arc::new(Field::new(
                    "item",
                    ArrowDataType::Float32,
                    false,
                ))),
                false,
            ),
        ]);

        Ok(FormatStatistics {
            row_count,
            size_bytes,
            file_count,
            column_stats: HashMap::new(), // Would be populated from actual file stats
            schema,
        })
    }

    async fn get_bloom_filter(&self, path: &str, column: &str) -> Result<Option<Vec<u8>>> {
        debug!(
            "Getting bloom filter for column {} at path: {}",
            column, path
        );

        // Bloom filters are engine-specific
        // SST and SWIFT engines have bloom filter support
        match self.engine.strategy() {
            StorageFormatStrategy::Sst | StorageFormatStrategy::Swift => {
                // These engines support bloom filters, but we need engine-specific access
                // For now, return None - real implementation would access engine internals
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    async fn list_files(&self, path: &str) -> Result<Vec<FileEntry>> {
        debug!("Listing files at path: {}", path);

        // Use filesystem to list files in the path
        // This would typically use the engine's filesystem factory
        // For now, return empty list - real implementation would enumerate files
        Ok(Vec::new())
    }
}

fn proxima_tree_node_to_json(node: &ProximaTreeNode) -> serde_json::Value {
    match node {
        ProximaTreeNode::Value(value) => {
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
        }
        other => serde_json::to_value(other).unwrap_or(serde_json::Value::Null),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract collection ID from a storage path
///
/// Expected path formats:
/// - `{base}/{collection_id}/data`
/// - `{base}/{collection_id}`
fn extract_collection_id_from_path(path: &str) -> Result<String> {
    let parts: Vec<&str> = path.trim_end_matches('/').split('/').collect();

    if parts.len() < 2 {
        return Err(anyhow!("Path too short to extract collection ID: {}", path));
    }

    // Check if last segment is "data"
    if parts.last() == Some(&"data") && parts.len() >= 2 {
        // Return second-to-last segment
        Ok(parts[parts.len() - 2].to_string())
    } else {
        // Return last segment
        Ok(parts.last().copied().unwrap_or_default().to_string())
    }
}

// ============================================================================
// Concrete Adapter Types for Each Engine
// ============================================================================

/// Type alias for SST format adapter
pub type SstFormatAdapter = InternalFormatAdapter<crate::storage::engines::sst::SstEngine>;

/// Type alias for HELIX format adapter
pub type HelixFormatAdapter = InternalFormatAdapter<crate::storage::engines::helix::HelixEngine>;

/// Type alias for VIPER format adapter
pub type ViperFormatAdapter = InternalFormatAdapter<crate::storage::engines::viper::ViperEngine>;

/// Type alias for NOVA format adapter
pub type NovaFormatAdapter = InternalFormatAdapter<crate::storage::engines::nova::NovaEngine>;

/// Type alias for SWIFT format adapter
#[cfg(feature = "experimental-engines")]
#[allow(deprecated)]
pub type SwiftFormatAdapter = InternalFormatAdapter<crate::storage::engines::swift::SwiftEngine>;

/// Type alias for RAPTOR format adapter
#[cfg(feature = "experimental-engines")]
#[allow(deprecated)]
pub type RaptorFormatAdapter = InternalFormatAdapter<crate::storage::engines::raptor::RaptorEngine>;

// ============================================================================
// Factory Functions
// ============================================================================

/// Create an SST format adapter from an existing engine
pub fn create_sst_adapter(
    engine: Arc<crate::storage::engines::sst::SstEngine>,
) -> SstFormatAdapter {
    InternalFormatAdapter::new(engine)
}

/// Create a HELIX format adapter from an existing engine
pub fn create_helix_adapter(
    engine: Arc<crate::storage::engines::helix::HelixEngine>,
) -> HelixFormatAdapter {
    InternalFormatAdapter::new(engine)
}

/// Create a VIPER format adapter from an existing engine
pub fn create_viper_adapter(
    engine: Arc<crate::storage::engines::viper::ViperEngine>,
) -> ViperFormatAdapter {
    InternalFormatAdapter::new(engine)
}

/// Create a NOVA format adapter from an existing engine
pub fn create_nova_adapter(
    engine: Arc<crate::storage::engines::nova::NovaEngine>,
) -> NovaFormatAdapter {
    InternalFormatAdapter::new(engine)
}

/// Create a SWIFT format adapter from an existing engine
#[cfg(feature = "experimental-engines")]
#[allow(deprecated)]
pub fn create_swift_adapter(
    engine: Arc<crate::storage::engines::swift::SwiftEngine>,
) -> SwiftFormatAdapter {
    InternalFormatAdapter::new(engine)
}

/// Create a RAPTOR format adapter from an existing engine
#[cfg(feature = "experimental-engines")]
#[allow(deprecated)]
pub fn create_raptor_adapter(
    engine: Arc<crate::storage::engines::raptor::RaptorEngine>,
) -> RaptorFormatAdapter {
    InternalFormatAdapter::new(engine)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_collection_id_from_path() {
        // Test with /data suffix
        assert_eq!(
            extract_collection_id_from_path("/tmp/proximadb/my_collection/data").unwrap(),
            "my_collection"
        );

        // Test without /data suffix
        assert_eq!(
            extract_collection_id_from_path("/tmp/proximadb/my_collection").unwrap(),
            "my_collection"
        );

        // Test with trailing slash
        assert_eq!(
            extract_collection_id_from_path("/tmp/proximadb/my_collection/").unwrap(),
            "my_collection"
        );

        // Test short path error
        assert!(extract_collection_id_from_path("collection").is_err());
    }
}
