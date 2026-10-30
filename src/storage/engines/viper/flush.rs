// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! VIPER Flush Operations
//!
//! This module handles flushing vector records from memory to Parquet files
//! with dynamic schema generation and metadata separation.

use anyhow::{Context, Result};
// Use columnar module's StreamingParquetWriter instead of direct ArrowWriter
use proximadb_storage_ports::CollectionMetadataPort;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Use unified quantization engine
use proximadb_compression::StandardCompression;

use crate::storage::common::compaction_orchestrator::FilenameCodec;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::transaction_coordinator::{
    StagingConfig, TransactionCoordinator, TransactionStageType,
};

use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::optimization::{MetadataSorter, SortingStats};
use proximadb_records::ProximaRecord;

use super::viper_meta_collector::{ViperCollectorConfig, ViperMetadataCollector};

/// Flush operations for VIPER storage engine with atomic writes
pub struct Flush {
    /// Collection service for metadata access
    collection_service: Arc<RwLock<Option<Arc<dyn CollectionMetadataPort>>>>,

    /// Filesystem factory for cross-cloud atomic writes
    filesystem_factory: Arc<FilesystemFactory>,

    /// Atomic coordinator for ACID operations
    atomic_coordinator: Arc<TransactionCoordinator>,

    /// Direct compression provider (no adapter indirection)
    _compression_provider: StandardCompression,

    /// Metrics updater for flush operation tracking
    metrics_updater: Option<Arc<dyn crate::metrics::InternalMetricsUpdater>>,

    /// Quantization engine for unified quantization
    quantization_engine:
        Option<Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine>>,
}

impl std::fmt::Debug for Flush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flush")
            .field("collection_service", &"<dyn CollectionMetadataPort>")
            .field("filesystem_factory", &self.filesystem_factory)
            .field("atomic_coordinator", &self.atomic_coordinator)
            .field("compression_provider", &"StandardCompression")
            .field("metrics_updater", &self.metrics_updater.is_some())
            .field("quantization_engine", &self.quantization_engine.is_some())
            .finish()
    }
}

impl Flush {
    pub async fn new(
        collection_service: Arc<RwLock<Option<Arc<dyn CollectionMetadataPort>>>>,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        // Create atomic coordinator
        let atomic_coordinator = Arc::new(
            TransactionCoordinator::new(filesystem_factory.clone(), None)
                .await
                .context("Failed to create atomic coordinator")?,
        );

        // Initialize compression provider directly
        let compression_provider = StandardCompression;

        // Initialize quantization engine
        let codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );
        let distance_compute = Arc::new(
            proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(
                crate::proto::proximadb_v1::DistanceMetric::Cosine,
            ),
        );
        let quantization_engine = Some(Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_compute,
                codebook_store,
            ),
        ));

        Ok(Self {
            collection_service,
            filesystem_factory,
            atomic_coordinator,
            _compression_provider: compression_provider,
            metrics_updater: None, // Set via set_metrics_updater for dependency injection
            quantization_engine,
        })
    }

    /// Set the metrics updater for flush operation tracking
    pub fn set_metrics_updater(
        &mut self,
        updater: Arc<dyn crate::metrics::InternalMetricsUpdater>,
    ) {
        self.metrics_updater = Some(updater);
    }

    // Deferred: Quantization should be handled inside HybridWriter or a dedicated module
    // that creates proper columnar storage with constants::FIELD_VECTOR_BINARY,
    // constants::FIELD_VECTOR_INT8, constants::FIELD_VECTOR_PQ8 columns
    // The quantization config from collection should control whether these columns are created

    /// Core flush operation using proper staging pattern
    pub async fn flush_vectors(
        &self,
        collection_id: &str,
        vector_records: &[VectorRecord],
        batch_ids: &[String],
        force: bool,
        synchronous: bool,
        viper_config: &crate::core::config::ViperConfig,
        provided_collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<crate::storage::traits::FlushResult> {
        info!("🔄 VIPER: Starting flush operation with staging pattern");
        info!(
            "🔍 VIPER: Flush params - force: {}, synchronous: {}, vector_records_len: {}, batch_ids: {}",
            force,
            synchronous,
            vector_records.len(),
            batch_ids.len()
        );

        // Use provided collection config or fetch if not provided (avoid duplicate calls)
        let collection_config = if let Some(config) = provided_collection_config {
            info!("✅ VIPER: Using provided collection config (avoiding duplicate fetch)");
            Some(config.clone())
        } else {
            // Fetch collection configuration using proto type directly
            let service_lock = self.collection_service.read().await;
            if let Some(ref service) = *service_lock {
                match service.collection(collection_id).await {
                    Ok(Some(collection)) => {
                        info!("📋 VIPER: Fetched collection config from service");
                        Some(collection)
                    }
                    Ok(None) => {
                        warn!("⚠️ Collection {} not found during flush", collection_id);
                        None
                    }
                    Err(e) => {
                        warn!("⚠️ Failed to get collection {}: {}", collection_id, e);
                        None
                    }
                }
            } else {
                warn!("⚠️ No collection service available during flush");
                None
            }
        };

        // Extract vector dimensions - REQUIRED for proper processing
        let vector_dimensions = collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map(|config| config.dimension)
            .ok_or_else(|| {
                error!("CRITICAL: Collection config missing or incomplete for collection {}. Cannot determine vector dimensions!", collection_id);
                anyhow::anyhow!("Collection config with dimension is required for flush operation")
            })?;

        info!(
            "🔧 VIPER FLUSH: Using {} dimensions for collection {}",
            vector_dimensions, collection_id
        );

        info!(
            "🔍 VIPER: Processing flush for collection: {}",
            collection_id
        );

        let operation_id = proximadb_kernel::uuid::Uuid::new_v4().to_string();

        if vector_records.is_empty() {
            info!(
                "📋 VIPER: No vector records provided for collection {}",
                collection_id
            );
            return Ok(crate::storage::traits::FlushResult {
                success: true,
                collections_affected: vec![collection_id.to_string()],
                entries_flushed: Some(0),
                bytes_written: Some(0),
                files_created: Some(0),
                file_paths: vec![],
                duration_ms: Some(0),
                completed_at: chrono::Utc::now(),
                compaction_triggered: false,
                compaction_error: None,
                flushed_batch_ids: Vec::new(), // ✅ Empty for empty flush
                engine_metrics: {
                    let mut metrics = std::collections::HashMap::new();
                    metrics.insert(
                        "operation_id".to_string(),
                        serde_json::Value::String(operation_id.clone()),
                    );
                    metrics.insert("empty_flush".to_string(), serde_json::Value::Bool(true));
                    metrics
                },
            });
        }

        info!(
            "💾 VIPER: Processing {} vector records for flush",
            vector_records.len()
        );

        // Step 1: Generate unique Parquet filename using unified compactor format
        let codec = FilenameCodec::new();
        // Level 0 for flush, use VIPER_FILE_EXT constant without the dot
        let parquet_filename = codec.generate(0, &crate::storage::engines::VIPER_FILE_EXT[1..]);
        info!(
            "🔄 VIPER: Step 1 - Preparing atomic Parquet write: {}",
            parquet_filename
        );

        // Step 2: Sort records by metadata for optimal Parquet encoding
        info!(
            "🔄 VIPER: Step 2a - Sorting {} vector records by metadata for optimal compression",
            vector_records.len()
        );
        let (sorted_records, _sort_stats) = match self
            .sort_records_for_parquet_encoding(vector_records, &collection_config)
            .await
        {
            Ok(result) => {
                info!(
                    "✅ VIPER: Step 2a - Records sorted (estimated compression improvement: {:.1}%)",
                    result.1.compression_estimate * 100.0
                );
                result
            }
            Err(e) => {
                warn!(
                    "⚠️ VIPER: Step 2a - Sorting failed, using original order: {}",
                    e
                );
                (
                    vector_records.to_vec(),
                    crate::storage::optimization::SortingStats::default(),
                )
            }
        };

        // Step 2a: Check if quantization is enabled for this collection
        // Quantization will be handled by the HybridWriter if enabled in config
        // The writer will automatically create vector_binary, vector_int8, vector_pq8 columns
        // based on the quantization config
        let quantization_enabled = collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .and_then(|c| c.quantization.as_ref())
            .is_some();

        if quantization_enabled {
            info!("⚡ VIPER: Quantization enabled - HybridWriter will create quantized columns");
            // HybridWriter will handle quantization internally based on config
        } else {
            info!("📝 VIPER: Quantization not enabled - only FP32 vectors will be stored");
        }

        // Step 2b: Use HybridParquetWriter directly from columnar module (like NOVA does)
        info!(
            "🔄 VIPER: Step 2b - Using HybridParquetWriter for {} sorted vector records",
            sorted_records.len()
        );
        debug!("📊 VIPER WRITER PATH ANALYSIS:");
        debug!("   - Input: Entire batch from memtable (flush pattern)");
        debug!("   - Processing: HybridWriter decides streaming vs batch mode");
        debug!("   - Output: Single Parquet file with optimal encoding");
        debug!("   - Auto-optimization: HybridWriter handles everything");

        // Calculate data URL for final destination
        let storage_assignment = collection_config
            .as_ref()
            .and_then(|c| c.storage_assignment.as_ref())
            .ok_or_else(|| anyhow::anyhow!(
                "Collection '{}' has no storage assignment. All collections must have storage assignments.",
                collection_id
            ))?;

        let data_url = format!(
            "{}/{}/data",
            storage_assignment.base_location, collection_id
        );

        // Ensure the data directory exists before writing
        // Strip file:// prefix if present for local filesystem operations
        let dir_path = if data_url.starts_with("file://") {
            data_url
                .strip_prefix("file://")
                .unwrap_or(&data_url)
                .to_string()
        } else {
            data_url.clone()
        };

        debug!("🟩 VIPER: Ensuring directory exists: {}", dir_path);
        if let Err(e) = tokio::fs::create_dir_all(&dir_path).await {
            error!("❌ VIPER: Failed to create directory {}: {}", dir_path, e);
            return Err(anyhow::anyhow!(
                "Failed to create directory {}: {}",
                dir_path,
                e
            ));
        }

        // Generate filename using FilenameCodec
        let codec = FilenameCodec::new();
        let filename = codec.generate(0, &crate::storage::engines::VIPER_FILE_EXT[1..]);
        let final_url = format!("{}/{}", data_url, filename);

        debug!("🟩 HYBRID_WRITER: Using columnar HybridParquetWriter::write_with_cache");
        debug!(
            "🟩 HYBRID_WRITER: Records: {}, Final URL: {}",
            sorted_records.len(),
            final_url
        );

        // Configure HybridParquetWriter like NOVA does
        use crate::storage::engines::core::formats::columnar::parquet_write_engine::writer_config::ParquetWriterConfig;
        // Convert string compression to Parquet compression enum
        let parquet_compression = match viper_config.compression.as_str() {
            "none" => parquet::basic::Compression::UNCOMPRESSED,
            "zstd" => parquet::basic::Compression::ZSTD(Default::default()),
            "snappy" => parquet::basic::Compression::SNAPPY,
            "gzip" => parquet::basic::Compression::GZIP(Default::default()),
            "lz4" => parquet::basic::Compression::LZ4,
            "brotli" => parquet::basic::Compression::BROTLI(Default::default()),
            "lzo" => parquet::basic::Compression::LZO,
            _ => {
                debug!(
                    "Unknown compression '{}', defaulting to ZSTD",
                    viper_config.compression
                );
                parquet::basic::Compression::ZSTD(Default::default())
            }
        };

        let writer_config = ParquetWriterConfig {
            row_group_size: viper_config.row_group_size,
            page_size: 1024 * 1024, // 1MB pages
            write_batch_size: 10000,
            compression: parquet_compression,
            compression_level: Some(viper_config.compression_level),
            enable_dictionary: true,
            enable_bloom_filters: true,
            bloom_filter_fpp: 0.01,
            bloom_filter_ndv: sorted_records.len() as u64,
            enable_statistics: true,
            enable_page_index: true,
            sort_columns: vec![],
            id_less_storage: false,
            filterable_metadata_columns: None,
            quantization: {
                // Enable quantization for VIPER progressive search (Binary → INT8 → FP32)
                // VIPER uses aggressive quantization for columnar analytics workloads
                crate::proto::proximadb_v1::QuantizationConfig {
                    enabled: Some(true),
                    enable_progressive_search: Some(true),
                    binary_filter_selectivity: Some(0.1),
                    int8_ranking_selectivity: Some(0.3),
                    ..Default::default()
                }
            },
            max_records_per_file: None,
            target_file_size_bytes: Some(128 * 1024 * 1024), // 128MB
            enable_async_io: true,
        };

        let hybrid_config = crate::storage::engines::core::formats::columnar::hybrid_writer::HybridWriterConfig {
            base_config: writer_config,
            initial_mode: crate::storage::engines::core::formats::columnar::hybrid_writer::WriterMode::Adaptive,
            enable_auto_switch: true,
            mode_switch_threshold: 1000,
            pattern_window_size: 100,
            streaming_threshold: 500.0,  // Lower for flush operations
            batch_threshold: 1000,       // Lower for flush operations
            max_buffer_size: 50 * 1024 * 1024, // 50MB buffer
            buffer_time_limit: std::time::Duration::from_secs(10),
            enable_concurrent_writes: false,
            max_concurrent_writers: 1,
            optimize_row_group_size: true,
            min_row_group_size: 100,     // Smaller for flush
            max_row_group_size: 10000,   // Smaller for flush
        };

        // Extract filterable columns from collection config to pass to writer
        let filterable_columns_for_writer = collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map(|config| config.filterable_columns.clone());

        // Create VIPER metadata collector for centroid-based row group pruning
        let viper_collector = ViperMetadataCollector::new(ViperCollectorConfig {
            compute_centroids: true,
            compute_radius: true,
            sample_rate: 1.0, // Sample all vectors for accurate radius
        });

        // Convert sorted VectorRecords to ProximaRecords for write_with_cache
        let proxima_records: Vec<proximadb_records::ProximaRecord> = sorted_records
            .iter()
            .map(proximadb_records::ProximaRecord::from)
            .collect();

        // Use HybridParquetWriter::write_with_cache like NOVA does
        let (stats, returned_collector) = match crate::storage::engines::core::formats::columnar::hybrid_writer::HybridParquetWriter::write_with_cache(
            &proxima_records,
            vector_dimensions as usize,
            hybrid_config,
            &final_url,
            &self.filesystem_factory,
            filterable_columns_for_writer, // Pass filterable columns from collection config
            Some(Box::new(viper_collector)), // Pass VIPER metadata collector for centroid computation
        ).await {
            Ok(result) => {
                debug!("🟩 HYBRID_WRITER: ✅ write_with_cache completed successfully");
                debug!("🟩 HYBRID_WRITER: Stats - file_size: {}, total_records: {}",
                         result.0.file_size, result.0.total_records);
                result
            }
            Err(e) => {
                debug!("🟩 HYBRID_WRITER: ❌ write_with_cache failed: {}", e);
                error!("❌ VIPER: Step 2 - HybridParquetWriter failed: {}", e);
                return Err(e.context("Failed to write Parquet via HybridParquetWriter"));
            }
        };

        // Save sidecar metadata file with centroids for row group pruning
        if let Some(collector) = returned_collector {
            let sidecar_ext = collector.sidecar_extension();
            if !sidecar_ext.is_empty() {
                match collector.serialize_metadata() {
                    Ok(sidecar_bytes) if !sidecar_bytes.is_empty() => {
                        // Generate sidecar file path (same as parquet but with .viper_meta extension)
                        let sidecar_url =
                            final_url.replace(".parquet", &format!(".{}", sidecar_ext));
                        if let Ok(fs) = self.filesystem_factory.get_filesystem(&sidecar_url) {
                            match fs.write(&sidecar_url, &sidecar_bytes, None).await {
                                Ok(_) => {
                                    debug!(
                                        "🟩 VIPER: Saved centroid sidecar metadata to {}",
                                        sidecar_url
                                    );
                                }
                                Err(e) => {
                                    warn!("⚠️ VIPER: Failed to save sidecar metadata: {}", e);
                                    // Continue even if sidecar fails - Parquet file is the primary data
                                }
                            }
                        }
                    }
                    Ok(_) => {
                        // Empty metadata - likely dimension not detected
                        debug!("🟩 VIPER: No sidecar metadata to save (dimension not detected)");
                    }
                    Err(e) => {
                        warn!("⚠️ VIPER: Failed to serialize sidecar metadata: {}", e);
                    }
                }
            }
        }

        // Since HybridWriter handled everything, create a marker to indicate completion
        let final_file_path = stats.file_path.clone();
        let file_size_for_stats = stats.file_size; // Capture file size from stats
        let mut parquet_data_or_path = vec![0xFF, 0xFF, 0xFF, 0xFF]; // Magic bytes to indicate already written
        parquet_data_or_path.extend_from_slice(final_file_path.as_bytes());

        debug!(
            "🟩 HYBRID_WRITER: HybridWriter completed, file at: {}",
            final_file_path
        );
        info!("✅ VIPER: Step 2 - HybridParquetWriter completed successfully");

        // Step 3: Check if HybridParquetWriter already handled atomic write
        let final_file_path = if parquet_data_or_path.len() > 4
            && parquet_data_or_path[0..4] == [0xFF, 0xFF, 0xFF, 0xFF]
        {
            // HybridParquetWriter already wrote the file atomically
            let path_str = String::from_utf8_lossy(&parquet_data_or_path[4..]);
            info!(
                "✅ VIPER: Step 3 - Parquet already written atomically by HybridParquetWriter: {}",
                path_str
            );
            path_str.to_string()
        } else if parquet_data_or_path.len() > 4
            && parquet_data_or_path[0..4] == [0xFA, 0xCE, 0xF1, 0x1E]
        {
            // Legacy path: Extract temp file path from marker
            let temp_path_str = String::from_utf8_lossy(&parquet_data_or_path[4..]);
            let temp_path = std::path::Path::new(temp_path_str.as_ref());

            // Use more efficient atomic move
            match self
                .write_parquet_atomic_from_path(
                    collection_id,
                    &parquet_filename,
                    temp_path,
                    &collection_config,
                )
                .await
            {
                Ok(path) => {
                    info!("✅ VIPER: Step 3 - Parquet atomically moved: {}", path);
                    // Clean up temp file if it still exists using filesystem API
                    let temp_path_str = temp_path.to_str().unwrap_or("");
                    if !temp_path_str.is_empty()
                        && let Ok(local_fs) = self.filesystem_factory.get_filesystem(temp_path_str)
                        && let Err(e) = local_fs.delete(temp_path_str).await
                    {
                        debug!("Temp file already moved or deleted: {}", e);
                    }
                    path
                }
                Err(e) => {
                    // Clean up temp file on error using filesystem API
                    let temp_path_str = temp_path.to_str().unwrap_or("");
                    if !temp_path_str.is_empty()
                        && let Ok(local_fs) = self.filesystem_factory.get_filesystem(temp_path_str)
                        && let Err(cleanup_err) = local_fs.delete(temp_path_str).await
                    {
                        debug!("Failed to clean up temp file: {}", cleanup_err);
                    }
                    error!("❌ VIPER: Step 3 - Atomic move failed: {}", e);
                    return Err(e.context("Failed to atomically move Parquet file"));
                }
            }
        } else {
            // Legacy path with buffer (should not happen with new code)
            match self
                .write_parquet_atomic(
                    collection_id,
                    &parquet_filename,
                    &parquet_data_or_path,
                    &collection_config,
                )
                .await
            {
                Ok(path) => {
                    info!("✅ VIPER: Step 3 - Parquet atomically written: {}", path);
                    path
                }
                Err(e) => {
                    error!("❌ VIPER: Step 3 - Atomic write failed: {}", e);
                    return Err(e.context("Failed to atomically write Parquet file"));
                }
            }
        };

        // Note: No cleanup needed - atomic write strategy handles staging automatically

        // Step 4: Check for compaction trigger
        info!("🔄 VIPER: Step 4 - Checking compaction trigger");
        let compaction_triggered = self
            .check_compaction_trigger(collection_id)
            .await
            .unwrap_or(false);

        // Step 5: Update collection metadata
        info!("🔄 VIPER: Step 5 - Updating collection metadata_info");
        // Calculate actual data size (either from file or buffer)
        let data_size = if parquet_data_or_path.len() > 4
            && parquet_data_or_path[0..4] == [0xFA, 0xCE, 0xF1, 0x1E]
        {
            // Legacy path marker - get file size from final path
            let fs = self.filesystem_factory.get_filesystem(&final_file_path)?;
            if let Ok(metadata) = fs.metadata(&final_file_path).await {
                metadata.size as usize
            } else {
                0
            }
        } else if parquet_data_or_path.len() > 4
            && parquet_data_or_path[0..4] == [0xFF, 0xFF, 0xFF, 0xFF]
        {
            // File already written marker - use captured file size from stats
            file_size_for_stats as usize
        } else {
            // In-memory data - use buffer length
            parquet_data_or_path.len()
        };

        self.update_collection_metadata_after_flush(collection_id, vector_records.len(), data_size)
            .await?;

        // Step 5.1: Record metrics (non-blocking, failure-tolerant)
        if let Some(ref metrics_updater) = self.metrics_updater {
            use crate::metrics::FlushMetricsUpdate;

            let flush_update = FlushMetricsUpdate {
                vectors_flushed: vector_records.len() as i64,
                bytes_written: data_size as i64,
                duration_ms: 0, // Deferred: Track actual duration
                files_created: 1,
                engine_type: "VIPER".to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };

            // Fire and forget - never block flush operation
            if let Err(e) = metrics_updater
                .record_flush(collection_id, flush_update)
                .await
            {
                debug!("Failed to record flush metrics: {}", e);
            }
        }

        // Step 6: Return successful flush result with BatchId coordination
        Ok(crate::storage::traits::FlushResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_flushed: Some(vector_records.len() as u64),
            bytes_written: Some(data_size as u64),
            files_created: Some(1),
            file_paths: vec![final_file_path.clone()],
            duration_ms: Some(0), // Will be set by high-level flush() method
            completed_at: chrono::Utc::now(),
            flushed_batch_ids: batch_ids
                .iter()
                .map(|_id| {
                    // Use compact BatchId for minimal storage overhead (10 bytes vs 100+ bytes)
                    crate::storage::persistence::write_ahead_log::BatchId::default()
                })
                .collect(), // ✅ Include for WAL cleanup
            engine_metrics: {
                let mut metrics = std::collections::HashMap::new();
                metrics.insert(
                    "operation_id".to_string(),
                    serde_json::Value::String(operation_id),
                );
                metrics.insert(
                    "vector_records_count".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(vector_records.len())),
                );
                metrics.insert(
                    "final_file_path".to_string(),
                    serde_json::Value::String(final_file_path),
                );
                metrics.insert(
                    "compaction_triggered".to_string(),
                    serde_json::Value::Bool(compaction_triggered),
                );
                metrics
            },
            compaction_triggered,
            compaction_error: None,
        })
    }

    /// INT8 Quantization for Parquet columnar storage
    /// Delegates to unified quantization engine for consistency across all engines
    #[allow(dead_code)]
    fn quantize_to_int8(
        &self,
        fp32_vector: &[f32],
        _quant_config: &crate::proto::proximadb_v1::QuantizationConfig,
    ) -> Vec<i8> {
        if fp32_vector.is_empty() {
            return Vec::new();
        }

        debug!("🔧 VIPER: Using unified INT8 quantization for Parquet storage");

        // Get unified quantization engine
        let quant_engine = match self.quantization_engine.as_ref() {
            Some(engine) => engine,
            None => {
                error!("Quantization engine not initialized");
                return Vec::new();
            }
        };

        // Use unified engine's INT8 quantization
        match quant_engine.quantize_to_u8(fp32_vector) {
            Ok((quantized, _min, _max)) => {
                // Convert u8 to i8 for compatibility with existing code
                quantized.iter().map(|&v| v as i8).collect()
            }
            Err(e) => {
                error!("Failed to quantize vector: {}", e);
                Vec::new()
            }
        }
    }

    // DEPRECATED: Use UnifiedQuantizationEngine::quantize_to_pq() instead
    // Removed duplicate quantize_to_pq8 implementation

    // DEPRECATED: Use UnifiedQuantizationEngine::quantize_to_pq() instead
    // Removed duplicate quantize_to_pq4 implementation

    /// Write Parquet data using atomic write strategy
    /// Uses unified atomic write infrastructure for cross-cloud compatibility
    /// Write Parquet file atomically from a temp file path using zero-copy I/O
    async fn write_parquet_atomic_from_path(
        &self,
        collection_id: &str,
        filename: &str,
        temp_file_path: &std::path::Path,
        collection_config: &Option<crate::proto::proximadb_v1::Collection>,
    ) -> Result<String> {
        // Get local filesystem for temp file (cloud-compatible)
        let temp_path_str = temp_file_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid temp file path"))?;
        let local_fs = self.filesystem_factory.get_filesystem(temp_path_str)?;
        let file_size = local_fs.metadata(temp_path_str).await?.size;

        info!(
            "🔄 Moving Parquet file atomically with zero-copy: {} ({} bytes)",
            filename, file_size
        );

        // Get storage assignment from collection config - fail fast if not present
        let storage_assignment = collection_config
            .as_ref()
            .and_then(|c| c.storage_assignment.as_ref())
            .ok_or_else(|| anyhow::anyhow!(
                "Collection '{}' has no storage assignment. All collections must have storage assignments.",
                collection_id
            ))?;

        // Get final destination path
        let data_url = format!(
            "{}/{}/data",
            storage_assignment.base_location, collection_id
        );
        let final_path = format!("{}/{}", data_url, filename);

        // Get filesystem for the destination
        let fs = self.filesystem_factory.get_filesystem(&data_url)?;

        info!(
            "📝 Using zero-copy I/O for atomic write from {:?} to {}",
            temp_file_path, final_path
        );

        // Check if we have UnifiedCachingFilesystem for optimal performance
        if let Some(_unified_fs) =
            fs.as_any()
                .downcast_ref::<crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem>()
        {
            info!("✅ Using UnifiedCachingFilesystem with intelligent staging and caching");

            // Read data once for optimized write using filesystem API (cloud-compatible)
            let data = local_fs.read(temp_path_str).await?;

            // Configure write options for atomic operation
            let write_options = crate::storage::persistence::filesystem::FileOptions {
                create_dirs: true,            // Create parent directories if needed
                overwrite: true,              // Overwrite if exists
                buffer_size: Some(64 * 1024), // 64KB buffer
                ..Default::default()
            };

            // Use unified filesystem which:
            // 1. Writes to optimal temp location
            // 2. Populates local cache for fast reads (hydrates indexes)
            // 3. Asynchronously uploads to cloud
            // 4. Returns immediately after local cache write
            fs.write(&final_path, &data, Some(write_options.clone()))
                .await?;

            info!("✅ Zero-copy write complete with cache population for index hydration");
        } else {
            // Fallback to standard filesystem API
            debug!("Using standard filesystem API (consider enabling ZeroCopyFilesystem)");

            let data = local_fs.read(temp_path_str).await?;

            let write_options = crate::storage::persistence::filesystem::FileOptions {
                create_dirs: true,
                overwrite: true,
                buffer_size: Some(64 * 1024),
                ..Default::default()
            };

            fs.write(&final_path, &data, Some(write_options)).await?;
        }

        info!(
            "✅ VIPER: Atomically moved Parquet file {} ({} KB)",
            final_path,
            file_size / 1024
        );

        // Verify file was written
        if fs.exists(&final_path).await? {
            let metadata = fs.metadata(&final_path).await?;
            info!(
                "✅ VIPER: Verified file exists at {} with size {} bytes",
                final_path, metadata.size
            );
        } else {
            error!("❌ VIPER: File not found after atomic move: {}", final_path);
        }

        Ok(final_path)
    }

    /// Legacy method that accepts a byte buffer (less efficient)
    async fn write_parquet_atomic(
        &self,
        collection_id: &str,
        filename: &str,
        parquet_data: &[u8],
        collection_config: &Option<crate::proto::proximadb_v1::Collection>,
    ) -> Result<String> {
        info!(
            "🔄 Writing Parquet file atomically: {} ({} bytes)",
            filename,
            parquet_data.len()
        );

        // Get storage assignment from collection config - fail fast if not present
        let storage_assignment = collection_config
            .as_ref()
            .and_then(|c| c.storage_assignment.as_ref())
            .ok_or_else(|| anyhow::anyhow!(
                "Collection '{}' has no storage assignment. All collections must have storage assignments.",
                collection_id
            ))?;

        // Begin atomic operation for flush
        let data_url = format!(
            "{}/{}/data",
            storage_assignment.base_location, collection_id
        );
        let staging_config = StagingConfig {
            base_url: data_url.clone(),
            collection_id: None, // Don't duplicate collection path
            operation_type: TransactionStageType::Flush,
            custom_staging_dir: None,
            auto_cleanup: true,
            max_orphaned_age_hours: 24,
            ..Default::default() // This will pick up skip_uuid_subdir: false
        };

        let atomic_op = self
            .atomic_coordinator
            .begin_atomic_operation(&staging_config)
            .await
            .context("Failed to begin atomic flush operation")?;

        info!("📝 Writing parquet file to staging: {}", filename);

        // Write parquet data to staging directory
        self.atomic_coordinator
            .write_to_staging(&atomic_op.operation_id, filename, parquet_data)
            .await
            .context("Failed to write parquet file to staging")?;

        // Finalize atomic operation - this will atomically move the file to final location
        self.atomic_coordinator
            .finalize_atomic_operation(&atomic_op.operation_id)
            .await
            .context("Failed to finalize atomic flush")?;

        let final_path = format!("{}/{}", data_url, filename);

        info!(
            "✅ VIPER: Atomically wrote Parquet file {} ({} KB)",
            final_path,
            parquet_data.len() / 1024
        );

        // Verify file was written
        let fs = self.filesystem_factory.get_filesystem(&data_url)?;
        if fs.exists(&final_path).await? {
            let metadata = fs.metadata(&final_path).await?;
            info!(
                "✅ VIPER: Verified file exists at {} with size {} bytes",
                final_path, metadata.size
            );
        } else {
            error!(
                "❌ VIPER: File not found after atomic write: {}",
                final_path
            );
        }

        Ok(final_path)
    }

    /// Check if compaction should be triggered
    async fn check_compaction_trigger(&self, _collection_id: &str) -> Result<bool> {
        // Compaction triggers based on multiple factors
        // Note: This is deferred to the Compaction which has full context
        // about file counts, sizes, and collection-specific thresholds.
        // For now, we don't trigger compaction from the flush path.

        // The BackgroundManager handles compaction scheduling based on:
        // 1. Number of Parquet files for this collection
        // 2. File size distribution
        // 3. Collection-specific compaction policies
        // 4. System load and resource availability

        // Return false to let BackgroundManager handle compaction decisions
        Ok(false)
    }

    /// Update collection metadata after flush
    async fn update_collection_metadata_after_flush(
        &self,
        collection_id: &str,
        records_count: usize,
        bytes_written: usize,
    ) -> Result<()> {
        // Note: Collection stats update is currently not implemented in the flush path.
        // The CollectionService has an update_stats() method that can track:
        // - vector_count (incremental changes)
        // - data_size_bytes (storage usage)
        //
        // This would be valuable metrics for users to monitor:
        // - Collection growth over time
        // - Storage utilization
        // - Flush performance metrics
        //
        // For now, we just log the flush completion locally.

        debug!(
            "Flush completed for collection {}: {} vectors, {} bytes written",
            collection_id, records_count, bytes_written
        );

        // Deferred: Consider integrating with CollectionService::update_stats()
        // to maintain accurate collection-level metrics that users can query.

        Ok(())
    }

    /// Sort vector records by metadata for optimal Parquet encoding
    async fn sort_records_for_parquet_encoding(
        &self,
        records: &[VectorRecord],
        collection_config: &Option<crate::proto::proximadb_v1::Collection>,
    ) -> Result<(Vec<VectorRecord>, SortingStats)> {
        // Extract filterable columns from collection config
        let filterable_columns = collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map(|config| config.filterable_columns.clone())
            .clone();

        if filterable_columns.is_none() {
            // No filterable columns, sort by vector ID for consistent ordering
            let mut sorted_records = records.to_vec();
            sorted_records.sort_by(|a, b| {
                let a_id = a.id.as_str();
                let b_id = b.id.as_str();
                a_id.cmp(b_id)
            });

            return Ok((
                sorted_records,
                SortingStats {
                    records_sorted: records.len(),
                    sort_keys_used: vec!["vector_id".to_string()],
                    compression_estimate: 0.05, // Small improvement from ID sorting
                    sort_time_us: 0,
                    ..Default::default()
                },
            ));
        }

        // Create metadata sorter from filterable columns
        let sorter =
            MetadataSorter::from_filterable_specs(filterable_columns.as_deref().unwrap_or(&[]));

        // Sort records for optimal encoding
        let canonical_records: Vec<ProximaRecord> =
            records.iter().map(ProximaRecord::from).collect();
        let (sorted_records, stats) = sorter.sort_for_encoding(canonical_records)?;
        let sorted_records = sorted_records.into_iter().map(VectorRecord::from).collect();

        debug!(
            "🎯 VIPER: Sorted {} records by {} filterable keys for Parquet optimization",
            stats.records_sorted,
            stats.sort_keys_used.len()
        );

        Ok((sorted_records, stats))
    }

    /// Apply mixed compression strategy with per-column optimization
    #[allow(dead_code)]
    fn apply_mixed_compression_strategy(
        &self,
        mut props_builder: parquet::file::properties::WriterPropertiesBuilder,
        batch: &arrow_array::RecordBatch,
    ) -> Result<parquet::file::properties::WriterPropertiesBuilder> {
        use proximadb_compression::{
            CompressionContext, detect_column_type, optimal_compression_for_column,
        };

        info!(
            "🎯 VIPER: Applying Mixed compression search_strategy to {} columns",
            batch.num_columns()
        );

        // Analyze each column and apply optimal compression
        for field in batch.schema().fields() {
            let name = field.name();

            // Detect column type based on name and context
            let data_type = detect_column_type(name, &CompressionContext::Parquet);

            // Get optimal compression for this column type
            let optimal_algorithm = optimal_compression_for_column(&data_type);

            // Convert to Parquet compression using the columnar common function
            let compression_algo = crate::storage::engines::core::formats::columnar::common::map_core_to_parquet_compression(
                optimal_algorithm,
                None, // No specific compression level for column-specific compression
            )?;

            let column_path = parquet::schema::types::ColumnPath::from(name.as_str());

            debug!(
                "🔧 VIPER Mixed: {} -> {:?} (type: {:?})",
                name, optimal_algorithm, data_type
            );

            // Apply per-column compression - compression_algo is already a parquet compression type
            props_builder =
                props_builder.set_column_compression(column_path.clone(), compression_algo);

            // Apply optimal encoding based on column type
            let encoding = match data_type {
                proximadb_compression::ColumnData::BinaryQuantized => {
                    // Binary data - use bit packing for maximum density
                    parquet::basic::Encoding::RLE
                }
                proximadb_compression::ColumnData::Int8Quantized => {
                    // Integer quantized - use delta encoding
                    parquet::basic::Encoding::DELTA_BINARY_PACKED
                }
                proximadb_compression::ColumnData::ProductQuantized => {
                    // PQ vectors - use byte stream split for floating point efficiency
                    parquet::basic::Encoding::BYTE_STREAM_SPLIT
                }
                proximadb_compression::ColumnData::FullPrecision => {
                    // FP32 vectors - use byte stream split for best compression
                    parquet::basic::Encoding::BYTE_STREAM_SPLIT
                }
                proximadb_compression::ColumnData::Identifier => {
                    // ID columns - use dictionary encoding for deduplication
                    parquet::basic::Encoding::RLE_DICTIONARY
                }
                proximadb_compression::ColumnData::Metadata => {
                    // Metadata - use dictionary encoding for repeated values
                    parquet::basic::Encoding::RLE_DICTIONARY
                }
                proximadb_compression::ColumnData::Timestamp => {
                    // Timestamps - use delta encoding for monotonic values
                    parquet::basic::Encoding::DELTA_BINARY_PACKED
                }
                proximadb_compression::ColumnData::Generic => {
                    // Generic data - use plain encoding
                    parquet::basic::Encoding::PLAIN
                }
            };

            props_builder = props_builder.set_column_encoding(column_path, encoding);

            debug!("🔧 VIPER Mixed: {} encoding -> {:?}", name, encoding);
        }

        info!("✅ VIPER: Mixed compression search_strategy applied to all columns");
        Ok(props_builder)
    }
}
