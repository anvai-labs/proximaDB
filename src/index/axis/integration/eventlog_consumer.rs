/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! AXIS EventLog consumer that processes flush and compaction events
//! This runs as a background task and builds/updates indexes asynchronously

use anyhow::{Context, Result};
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use crate::index::axis::AxisManager;
use crate::index::axis::eventlog::{
    EventLogService, EventType, ExtractionMode, IndexEvent, StorageEngineType,
};
use crate::proto::proximadb_v1::Collection;
use crate::storage::persistence::filesystem::FilesystemFactory;

/// AXIS EventLog consumer configuration
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    /// How often to poll for new events (milliseconds)
    pub poll_interval_ms: u64,

    /// Maximum events to process in a single batch
    pub batch_size: usize,

    /// Whether to enable concurrent processing
    pub concurrent_processing: bool,

    /// Maximum concurrent index operations
    pub max_concurrent_ops: usize,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 100, // Poll every 100ms for low latency
            batch_size: 10,        // Process up to 10 events at once
            concurrent_processing: true,
            max_concurrent_ops: 4, // Max 4 concurrent index operations
        }
    }
}

/// Consumer metrics
// Type alias for compatibility
pub type EventLogConsumer = AxisEventLogConsumer;

/// Statistics for the event log consumer.
#[derive(Debug, Clone, Default)]
pub struct ConsumerStats {
    /// Total number of events successfully processed.
    pub events_processed: u64,
    /// Total number of events that failed processing.
    pub events_failed: u64,
    /// Timestamp of the most recently processed event.
    pub last_processed_timestamp: Option<std::time::SystemTime>,
}

#[derive(Default)]
struct ConsumerMetrics {
    events_skipped: AtomicU64,
}

/// AXIS EventLog consumer
pub struct AxisEventLogConsumer {
    /// Configuration
    config: ConsumerConfig,

    /// EventLog service to consume from
    event_log: Arc<dyn EventLogService>,

    /// AXIS manager for index operations
    axis_manager: Arc<AxisManager>,

    /// Filesystem factory for reading data files
    #[allow(dead_code)]
    filesystem_factory: Arc<FilesystemFactory>,

    /// Collection cache
    #[allow(dead_code)]
    collection_cache: Arc<DashMap<String, Arc<Collection>>>,

    /// Unified cache orchestrator (shared across system)
    #[allow(dead_code)]
    cache_orchestrator: Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>,

    /// Metrics
    metrics: Arc<ConsumerMetrics>,

    /// Shutdown signal
    shutdown: tokio::sync::watch::Receiver<bool>,
}

impl AxisEventLogConsumer {
    /// Create new consumer
    pub fn new(
        config: ConsumerConfig,
        event_log: Arc<dyn EventLogService>,
        axis_manager: Arc<AxisManager>,
        filesystem_factory: Arc<FilesystemFactory>,
        collection_cache: Arc<DashMap<String, Arc<Collection>>>,
        cache_orchestrator: Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> Self {
        Self {
            config,
            event_log,
            axis_manager,
            filesystem_factory,
            collection_cache,
            cache_orchestrator,
            metrics: Arc::new(ConsumerMetrics::default()),
            shutdown,
        }
    }

    /// Start consuming events
    pub async fn run(self) {
        info!("Starting AXIS EventLog consumer");

        loop {
            // Check for shutdown
            if *self.shutdown.borrow() {
                info!("AXIS EventLog consumer shutting down");
                break;
            }

            // Process next batch of events
            match self.process_batch().await {
                Ok(processed) => {
                    if processed > 0 {
                        debug!("Processed {} events from EventLog", processed);
                    }
                }
                Err(e) => {
                    warn!("Error processing EventLog batch: {}", e);
                }
            }

            // Sleep before next poll
            sleep(Duration::from_millis(self.config.poll_interval_ms)).await;
        }

        info!("AXIS EventLog consumer stopped");
    }

    /// Process a batch of events
    async fn process_batch(&self) -> Result<usize> {
        // Get next batch of events
        let events = self
            .event_log
            .get_next_batch(self.config.batch_size)
            .await
            .context("Failed to get event batch")?;

        if events.is_empty() {
            return Ok(0);
        }

        let count = events.len();
        debug!("Processing {} events from EventLog", count);

        // Process events concurrently if enabled
        if self.config.concurrent_processing {
            self.process_concurrent(events).await?;
        } else {
            self.process_sequential(events).await?;
        }

        Ok(count)
    }

    /// Process events sequentially
    async fn process_sequential(&self, events: Vec<IndexEvent>) -> Result<()> {
        for event in events {
            self.process_event(event).await?;
        }
        Ok(())
    }

    /// Process events concurrently
    async fn process_concurrent(&self, events: Vec<IndexEvent>) -> Result<()> {
        use futures::stream::{self, StreamExt};

        let results = stream::iter(events)
            .map(|event| self.process_event(event))
            .buffer_unordered(self.config.max_concurrent_ops)
            .collect::<Vec<_>>()
            .await;

        // Check for errors
        for result in results {
            result?;
        }

        Ok(())
    }

    /// Process a single event
    async fn process_event(&self, event: IndexEvent) -> Result<()> {
        let start_time = std::time::Instant::now();
        let event_id = event.event_id.clone();
        let event_type = match event.operation {
            EventType::Flush => "flush",
            EventType::Compaction => "compaction_info",
            EventType::Delete => "delete",
        };

        debug!(
            "[AXIS Consumer] Starting processing of {} event {}:\n  Collection: {}\n  Files: {:?}\n  Vector Count: {}\n  Has Quantized: {}\n  Has FP32: {}\n  Storage Engine: {:?}\n  Created At: {:?}",
            event_type,
            event_id,
            event.collection_id,
            event.file_paths,
            event.vector_count,
            event.has_quantized,
            event.has_fp32,
            event.storage_engine,
            event.timestamp
        );

        // Get collection configuration
        debug!(
            "[AXIS Consumer] Looking up collection {} in cache_info",
            event.collection_id
        );
        let collection = self
            .collection_cache
            .get(&event.collection_id)
            .ok_or_else(|| {
                error!(
                    "[AXIS Consumer] Collection {} not found in cache for event {}",
                    event.collection_id, event_id
                );
                anyhow::anyhow!("Collection {} not found", event.collection_id)
            })?
            .clone();

        debug!(
            "[AXIS Consumer] Collection {} found: dimension={:?}, engine={:?}, quantization_enabled={:?}",
            event.collection_id,
            collection.config.as_ref().map(|c| c.dimension),
            collection.config.as_ref().map(|c| c.storage_engine),
            collection
                .config
                .as_ref()
                .and_then(|c| c.quantization.as_ref())
                .map(|q| q.enabled)
        );

        let result = match event.operation {
            EventType::Flush => {
                debug!(
                    "[AXIS Consumer] Routing to flush event processor for event {}",
                    event_id
                );
                self.process_flush_event(event, collection).await
            }
            EventType::Compaction => {
                debug!(
                    "[AXIS Consumer] Routing to compaction event processor for event {}",
                    event_id
                );
                self.process_compaction_event(event, collection).await
            }
            EventType::Delete => {
                debug!(
                    "[AXIS Consumer] Routing to delete event processor for event {}",
                    event_id
                );
                self.process_delete_event(event, collection).await
            }
        };

        let duration = start_time.elapsed();

        match &result {
            Ok(()) => {
                debug!(
                    "[AXIS Consumer] Successfully processed {} event {} in {:.2}ms",
                    event_type,
                    event_id,
                    duration.as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                error!(
                    "[AXIS Consumer] Failed to process {} event {} after {:.2}ms: {}",
                    event_type,
                    event_id,
                    duration.as_secs_f64() * 1000.0,
                    e
                );
            }
        }

        result
    }

    async fn acknowledge_skipped_event(&self, event: &IndexEvent) -> Result<()> {
        self.metrics.events_skipped.fetch_add(1, Ordering::Relaxed);
        self.event_log
            .acknowledge_event(event.event_id.clone())
            .await
    }

    /// Process flush event - build or update indexes
    async fn process_flush_event(
        &self,
        event: IndexEvent,
        collection: Arc<Collection>,
    ) -> Result<()> {
        let event_id = event.event_id.clone();
        let start_time = std::time::Instant::now();

        // Check if collection has ANY indexes configured
        let has_indexes = collection
            .config
            .as_ref()
            .is_some_and(|c| !c.index_configs.is_empty());

        if !has_indexes {
            // No indexes configured - mark event as completed without processing
            info!(
                "[AXIS Consumer] No indexes configured for collection {}, marking flush event {} as completed",
                event.collection_id, event_id
            );

            // Exact/brute-force is the default when no ANN index is configured, but
            // we still need to acknowledge the EventLog entry so compaction can proceed.
            self.acknowledge_skipped_event(&event).await?;
            return Ok(());
        }

        // Determine extraction mode based on index types
        let extraction_mode = self.determine_extraction_mode(&collection);
        let extraction_mode_str = match extraction_mode {
            ExtractionMode::Fp32Only => "FP32Only",
            ExtractionMode::QuantizedOnly => "QuantizedOnly",
            ExtractionMode::Both => "Both",
            ExtractionMode::Auto => "Auto",
        };

        debug!(
            "[AXIS Consumer] Flush event {} extraction mode determined: {} (quantization_enabled={:?}, has_fp32={}, has_quantized={})",
            event_id,
            extraction_mode_str,
            collection
                .config
                .as_ref()
                .and_then(|c| c.quantization.as_ref())
                .map(|q| q.enabled),
            event.has_fp32,
            event.has_quantized
        );

        info!(
            "[AXIS Consumer] Processing flush event {} for collection {} with {} vectors in {} mode",
            event_id, event.collection_id, event.vector_count, extraction_mode_str
        );

        // Read vectors from data files based on extraction mode
        let read_start = std::time::Instant::now();
        debug!(
            "[AXIS Consumer] Reading vectors from {} files for event {}: {:?}",
            event.file_paths.len(),
            event_id,
            event.file_paths
        );

        let vectors = self
            .read_vectors_from_files(
                &event.file_paths,
                extraction_mode,
                event.storage_engine,
                &event.collection_id,
            )
            .await?;

        let read_duration = read_start.elapsed();
        debug!(
            "[AXIS Consumer] Read {} vectors from files in {:.2}ms for event {}",
            vectors.len(),
            read_duration.as_secs_f64() * 1000.0,
            event_id
        );

        // Update AXIS indexes
        let update_start = std::time::Instant::now();
        debug!(
            "[AXIS Consumer] Updating AXIS indexes for collection {} with {} vectors from event {}",
            event.collection_id,
            vectors.len(),
            event_id
        );

        // Get index configuration for the collection
        let index_config = self
            .axis_manager
            .native_index_config(&event.collection_id)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    "[AXIS Consumer] Failed to get index config for {}: {}, using defaults",
                    event.collection_id, e
                );
                crate::index::config::RuntimeIndexConfig::default()
            });

        // Update AXIS indexes using hybrid indexing (adapts based on batch size)
        self.axis_manager
            .index_vectors_hybrid(
                &event.collection_id,
                vectors.clone(),
                event.file_paths.clone(),
                &index_config,
            )
            .await
            .map_err(|e| {
                error!(
                    "[AXIS Consumer] Failed to update AXIS indexes for event {}: {}",
                    event_id, e
                );
                e
            })?;

        let update_duration = update_start.elapsed();
        debug!(
            "[AXIS Consumer] AXIS index update completed in {:.2}ms for event {}",
            update_duration.as_secs_f64() * 1000.0,
            event_id
        );

        // Acknowledge processing completion
        let ack_start = std::time::Instant::now();
        debug!(
            "[AXIS Consumer] Acknowledging event {} with EventLog",
            event_id
        );

        self.event_log
            .acknowledge_event(event.event_id.clone())
            .await?;

        let ack_duration = ack_start.elapsed();
        let total_duration = start_time.elapsed();

        info!(
            "[AXIS Consumer] Successfully processed flush event {} for collection {} \n  Total: {:.2}ms (read: {:.2}ms, update: {:.2}ms, ack: {:.2}ms)",
            event_id,
            event.collection_id,
            total_duration.as_secs_f64() * 1000.0,
            read_duration.as_secs_f64() * 1000.0,
            update_duration.as_secs_f64() * 1000.0,
            ack_duration.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Process compaction event - update file references
    async fn process_compaction_event(
        &self,
        event: IndexEvent,
        collection: Arc<Collection>,
    ) -> Result<()> {
        let event_id = event.event_id.clone();
        let start_time = std::time::Instant::now();

        // Check if collection has ANY indexes configured
        let has_indexes = collection
            .config
            .as_ref()
            .is_some_and(|c| !c.index_configs.is_empty());

        if !has_indexes {
            // No indexes configured - mark event as completed without processing
            info!(
                "[AXIS Consumer] No indexes configured for collection {}, marking compaction event {} as completed",
                event.collection_id, event_id
            );

            // The compaction is done on storage side, we just have nothing to update.
            self.acknowledge_skipped_event(&event).await?;
            return Ok(());
        }

        debug!(
            "[AXIS Consumer] Processing compaction event {} for collection {}:\n  Output Files: {:?}\n  Vector Count: {}\n  Storage Engine: {:?}",
            event_id,
            event.collection_id,
            event.file_paths,
            event.vector_count,
            event.storage_engine
        );

        info!(
            "[AXIS Consumer] Processing compaction event {} for collection {} with {} output files",
            event_id,
            event.collection_id,
            event.file_paths.len()
        );

        // For compaction, we need to update the indexes to reflect the new file structure
        // The vectors are the same but in different files now
        let update_start = std::time::Instant::now();
        debug!(
            "[AXIS Consumer] Rebuilding AXIS indexes after compaction for {} output files in event {}",
            event.file_paths.len(),
            event_id
        );

        // Rebuild indexes with the new compacted files
        // Empty deleted_files since compaction already removed the old files
        self.axis_manager
            .rebuild_indexes_after_compaction(
                &event.collection_id,
                &event.file_paths, // New compacted files
                &[],               // Old files already deleted by compaction
            )
            .await
            .map_err(|e| {
                error!(
                    "[AXIS Consumer] Failed to rebuild indexes after compaction for event {}: {}",
                    event_id, e
                );
                e
            })?;

        let update_duration = update_start.elapsed();
        debug!(
            "[AXIS Consumer] File reference update completed in {:.2}ms for event {}",
            update_duration.as_secs_f64() * 1000.0,
            event_id
        );

        // Acknowledge processing
        let ack_start = std::time::Instant::now();
        debug!(
            "[AXIS Consumer] Acknowledging compaction event {} with EventLog",
            event_id
        );

        self.event_log
            .acknowledge_event(event.event_id.clone())
            .await?;

        let ack_duration = ack_start.elapsed();
        let total_duration = start_time.elapsed();

        info!(
            "[AXIS Consumer] Successfully processed compaction event {} for collection {}\n  Total: {:.2}ms (update: {:.2}ms, ack: {:.2}ms)",
            event_id,
            event.collection_id,
            total_duration.as_secs_f64() * 1000.0,
            update_duration.as_secs_f64() * 1000.0,
            ack_duration.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Process delete event - remove vectors from indexes
    async fn process_delete_event(
        &self,
        event: IndexEvent,
        _collection: Arc<Collection>,
    ) -> Result<()> {
        let event_id = event.event_id.clone();

        debug!(
            "[AXIS Consumer] Processing delete event {} for collection {}",
            event_id, event.collection_id
        );

        // For now, just acknowledge the event
        // Deferred: Implement actual deletion logic when needed
        warn!(
            "[AXIS Consumer] Delete event processing not yet implemented for event {}",
            event_id
        );

        // Acknowledge processing
        self.event_log
            .acknowledge_event(event.event_id.clone())
            .await?;

        Ok(())
    }

    /// Determine extraction mode based on collection's index configuration
    fn determine_extraction_mode(&self, collection: &Collection) -> ExtractionMode {
        // Check if collection has quantization enabled
        let has_quantization = collection
            .config
            .as_ref()
            .and_then(|c| c.quantization.as_ref())
            .map(|q| q.enabled);

        // Get index algorithm if configured
        let index_algorithm = collection
            .config
            .as_ref()
            .and_then(|c| c.index_configs.first())
            .map_or_else(
                || "None".to_string(),
                |i| format!("Algorithm({})", i.algorithm),
            );

        debug!(
            "[AXIS Consumer] Determining extraction mode for collection {}:\n  Quantization Enabled: {:?}\n  Index Algorithm: {}",
            collection.id, has_quantization, index_algorithm
        );

        // Check index types to determine what data we need
        // HNSW typically needs FP32 for accuracy
        // IVF can work with quantized data
        // PQ indexes prefer quantized data

        if has_quantization != Some(Some(true)) {
            // No quantization, must use FP32
            debug!("[AXIS Consumer] No quantization enabled, using FP32Only mode");
            ExtractionMode::Fp32Only
        } else {
            // For now, extract both and let AXIS decide
            // In future, we can be smarter based on index type
            debug!("[AXIS Consumer] Quantization enabled, using Both mode for flexibility");
            ExtractionMode::Both
        }
    }

    /// Read vectors from data files using the unified VectorExtractor protocol.
    ///
    /// This method uses the ExtractionFactory to create engine-specific extractors,
    /// providing a consistent interface across all storage engines (SST, SWIFT, HELIX,
    /// VIPER, NOVA, RAPTOR).
    async fn read_vectors_from_files(
        &self,
        files: &[String],
        extraction_mode: ExtractionMode,
        storage_engine: StorageEngineType,
        collection_id: &str,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
        use crate::storage::trait_components::extractor::{
            ExtractionFactory, ExtractionMode as TraitMode, ExtractionRequest,
        };

        let start_time = std::time::Instant::now();

        debug!(
            "[AXIS Consumer] Starting vector extraction via unified VectorExtractor:\n  Files: {:?}\n  Count: {}\n  Storage Engine: {:?}\n  Extraction Mode: {:?}",
            files,
            files.len(),
            storage_engine,
            extraction_mode
        );

        if files.is_empty() {
            debug!("[AXIS Consumer] No files to extract, returning empty result");
            return Ok(Vec::new());
        }

        // Create filesystem factory with proper configuration
        let filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(Default::default())
                .await
                .context("Failed to create filesystem factory")?,
        );

        // Create UnifiedCachingFilesystem for the collection
        let base_fs = filesystem_factory.get_filesystem("file://")?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            collection_id.to_string(),
            "axis".to_string(),
        ));

        info!(
            "[AXIS Consumer] Using unified VectorExtractor for {:?} engine",
            storage_engine
        );

        // Create extractor via factory pattern
        let extractor = ExtractionFactory::create(storage_engine, unified_fs);

        // Convert extraction mode from eventlog type to extractor trait type
        let mode = match extraction_mode {
            ExtractionMode::Fp32Only => TraitMode::Fp32Only,
            ExtractionMode::QuantizedOnly => TraitMode::QuantizedOnly,
            ExtractionMode::Both => TraitMode::Both,
            ExtractionMode::Auto => TraitMode::Auto,
        };

        // Build extraction request for full scan
        let request = ExtractionRequest::full(files.to_vec()).with_mode(mode);

        debug!(
            "[AXIS Consumer] Sending extraction request: {} files, mode={:?}",
            files.len(),
            mode
        );

        // Extract vectors using unified interface
        let engine_name = extractor.engine_type();
        let result = extractor.extract_vectors(request).await.map_err(|e| {
            error!(
                "[AXIS Consumer] VectorExtractor failed for {:?} engine: {}",
                engine_name, e
            );
            anyhow::anyhow!("Vector extraction failed: {}", e)
        })?;

        // Log extraction stats
        debug!(
            "[AXIS Consumer] Extraction stats: {} vectors, {} bytes, {} files in {}ms",
            result.stats.vectors_extracted,
            result.stats.bytes_read,
            result.stats.files_processed,
            result.stats.duration_ms
        );

        // Convert ExtractedVector to ProximaRecord
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let all_vectors: Vec<proximadb_records::ProximaRecord> = result
            .vectors
            .into_iter()
            .filter_map(|v| {
                v.fp32_vector.map(|fp32_vec| {
                    let dim = fp32_vec.len() as u32;
                    let props: proximadb_records::ProximaTree = v
                        .metadata
                        .and_then(|m| {
                            if let serde_json::Value::Object(map) = m {
                                Some(
                                    map.into_iter()
                                        .map(|(k, val)| {
                                            use proximadb_data_model::ProximaValue;
                                            let pv = match val {
                                                serde_json::Value::String(s) => {
                                                    ProximaValue::String(s)
                                                }
                                                serde_json::Value::Number(n) => {
                                                    ProximaValue::Float64(n.as_f64().unwrap_or(0.0))
                                                }
                                                serde_json::Value::Bool(b) => {
                                                    ProximaValue::Boolean(b)
                                                }
                                                // Round 19: the extraction
                                                // contract changed (round 16:
                                                // real arrays/objects instead
                                                // of "[array]"/"[object]"
                                                // placeholders) — lower
                                                // structured values via the
                                                // canonical JSON→ProximaValue
                                                // mapping instead of Null.
                                                other => {
                                                    proximadb_records::conversions::json_to_proxima(
                                                        &other,
                                                    )
                                                }
                                            };
                                            (k, proximadb_records::ProximaTreeNode::Value(pv))
                                        })
                                        .collect(),
                                )
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    proximadb_records::ProximaRecord {
                        oid: v.id,
                        created_at_ns: now_ns,
                        updated_at_ns: now_ns,
                        props,
                        embeddings: vec![proximadb_records::EmbeddingCell {
                            model_id: "default".to_string(),
                            modality: "vector".to_string(),
                            values: proximadb_records::EmbeddingValues::Fp32(fp32_vec),
                            dim,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }
                })
            })
            .collect();

        let duration = start_time.elapsed();

        if all_vectors.is_empty() {
            warn!(
                "[AXIS Consumer] Vector extraction returned empty after {:.2}ms",
                duration.as_secs_f64() * 1000.0
            );
        } else {
            info!(
                "[AXIS Consumer] Vector extraction completed in {:.2}ms: extracted {} vectors from {} files using {:?} extractor",
                duration.as_secs_f64() * 1000.0,
                all_vectors.len(),
                files.len(),
                engine_name
            );
        }

        Ok(all_vectors)
    }
}

/// Start the AXIS EventLog consumer
pub async fn start_axis_consumer(
    event_log: Arc<dyn EventLogService>,
    axis_manager: Arc<AxisManager>,
    filesystem_factory: Arc<FilesystemFactory>,
    collection_cache: Arc<DashMap<String, Arc<Collection>>>,
    cache_orchestrator: Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    let config = ConsumerConfig::default();

    let consumer = AxisEventLogConsumer::new(
        config,
        event_log,
        axis_manager,
        filesystem_factory,
        collection_cache,
        cache_orchestrator,
        shutdown,
    );

    tokio::spawn(async move {
        consumer.run().await;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::axis::AxisConfig;
    use crate::index::axis::eventlog::{EventLogConfig, EventLogServiceAdapter, IndexEventBuilder};
    use crate::proto::proximadb_v1::{CollectionConfig, CollectionStats};
    use crate::storage::cache::orchestrator::CrossCacheOrchestrator;
    use crate::storage::persistence::filesystem::FilesystemConfig;
    use tempfile::TempDir;

    async fn create_no_index_consumer(
        collection_id: &str,
    ) -> (
        AxisEventLogConsumer,
        Arc<dyn EventLogService>,
        Arc<Collection>,
        TempDir,
    ) {
        let temp_dir = TempDir::new().unwrap();
        let base_url = format!("file://{}", temp_dir.path().display());

        let mut filesystem_config = FilesystemConfig::default();
        filesystem_config.default_fs = Some(base_url.clone());
        let filesystem_factory = Arc::new(
            FilesystemFactory::create(filesystem_config)
                .await
                .expect("filesystem factory"),
        );

        let collection = Arc::new(Collection {
            id: collection_id.to_string(),
            config: Some(CollectionConfig {
                name: collection_id.to_string(),
                dimension: 384,
                index_configs: vec![],
                ..Default::default()
            }),
            stats: Some(CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            ..Default::default()
        });

        let collection_cache = Arc::new(DashMap::new());
        collection_cache.insert(collection_id.to_string(), collection.clone());

        let event_log: Arc<dyn EventLogService> = EventLogServiceAdapter::embedded(
            EventLogConfig {
                base_storage_url: base_url,
                max_events_in_memory: 100,
                cleanup_interval_secs: 60,
                enable_recovery: true,
            },
            filesystem_factory.clone(),
            collection_cache.clone(),
        )
        .await
        .expect("event log");

        let axis_manager = Arc::new(AxisManager::new(AxisConfig::default()).await.unwrap());
        let orchestrator = Arc::new(CrossCacheOrchestrator::new(1024 * 1024));
        let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        (
            AxisEventLogConsumer::new(
                ConsumerConfig::default(),
                event_log.clone(),
                axis_manager,
                filesystem_factory,
                collection_cache,
                orchestrator,
                shutdown_rx,
            ),
            event_log,
            collection,
            temp_dir,
        )
    }

    #[tokio::test]
    async fn test_flush_without_indexes_acknowledges_event_immediately() {
        let collection_id = "exact_default_collection";
        let file_path = "file1.sstable".to_string();
        let (consumer, event_log, collection, _temp_dir) =
            create_no_index_consumer(collection_id).await;

        let event = IndexEventBuilder::flush_event(
            collection_id.to_string(),
            vec![file_path.clone()],
            1_000,
            StorageEngineType::SST,
            false,
            true,
        );

        event_log.add_event(event.clone()).await.unwrap();
        let initial_status = event_log
            .get_file_status(&file_path)
            .await
            .unwrap()
            .expect("file status");
        assert!(!initial_status.ready_for_compaction);

        consumer
            .process_flush_event(event, collection)
            .await
            .expect("flush event should be acknowledged");

        let final_status = event_log
            .get_file_status(&file_path)
            .await
            .unwrap()
            .expect("file status after acknowledgment");
        assert!(final_status.pending_indexes.is_empty());
        assert!(final_status.ready_for_compaction);
    }

    #[tokio::test]
    async fn test_compaction_without_indexes_acknowledges_event_immediately() {
        let collection_id = "exact_default_collection";
        let file_path = "file1.sstable".to_string();
        let (consumer, event_log, collection, _temp_dir) =
            create_no_index_consumer(collection_id).await;

        let event = crate::index::axis::eventlog::IndexEventBuilder::compaction_event(
            collection_id.to_string(),
            vec![file_path.clone()],
            1_000,
            StorageEngineType::SST,
        );

        event_log.add_event(event.clone()).await.unwrap();
        let initial_status = event_log
            .get_file_status(&file_path)
            .await
            .unwrap()
            .expect("file status");
        assert!(!initial_status.ready_for_compaction);

        consumer
            .process_compaction_event(event, collection)
            .await
            .expect("compaction event should be acknowledged");

        let final_status = event_log
            .get_file_status(&file_path)
            .await
            .unwrap()
            .expect("file status after acknowledgment");
        assert!(final_status.pending_indexes.is_empty());
        assert!(final_status.ready_for_compaction);
    }
}
