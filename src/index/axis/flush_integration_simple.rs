/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Simple integration between storage flush operations and AXIS metadata queue
//! Fire-and-forget pattern with zero latency impact on writes

use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

use crate::index::axis::eventlog::{EventLogManager as MetadataQueueService, StorageEngineType};
use crate::storage::engines::{CompactionParameters, FlushParameters};

/// Simple flush-to-AXIS notifier
/// Just sends metadata events - never blocks storage operations
pub struct SimpleFlushNotifier {
    /// Shared metadata queue service
    queue_service: Arc<MetadataQueueService>,
}

impl SimpleFlushNotifier {
    /// Create new notifier with shared queue service
    pub fn new(queue_service: Arc<MetadataQueueService>) -> Self {
        Self { queue_service }
    }

    /// Notify AXIS about flush - NEVER BLOCKS
    pub fn notify_flush(
        &self,
        params: &FlushParameters,
        flushed_files: Vec<String>,
        vector_count: usize,
    ) {
        let collection_id = match params.collection_id.as_ref() {
            Some(id) => id.clone(),
            None => return, // No collection ID, skip
        };

        // Determine what representations are available
        let (has_quantized, has_fp32) = Self::detect_representations(params);

        // Determine storage engine type
        let storage_engine = Self::detect_storage_engine(params);

        // Fire and forget - returns immediately
        let service = self.queue_service.clone();
        let collection_id_clone = collection_id.clone();
        tokio::spawn(async move {
            let _ = service
                .add_flush_event(
                    &collection_id_clone,
                    flushed_files,
                    vector_count,
                    storage_engine,
                    has_quantized,
                    has_fp32,
                )
                .await;
        });

        debug!("Notified AXIS about flush for collection {}", collection_id);
    }

    /// Notify AXIS about compaction - NEVER BLOCKS
    pub fn notify_compaction(
        &self,
        params: &CompactionParameters,
        output_files: Vec<String>,
        vector_count: usize,
    ) {
        let collection_id = match params.collection_id.as_ref() {
            Some(id) => id.clone(),
            None => return,
        };

        let storage_engine = Self::detect_storage_engine_from_compaction(params);

        // Fire and forget
        let service = self.queue_service.clone();
        let collection_id_clone = collection_id.clone();
        tokio::spawn(async move {
            let _ = service
                .add_compaction_event(
                    &collection_id_clone,
                    output_files,
                    vector_count,
                    storage_engine,
                )
                .await;
        });

        debug!(
            "Notified AXIS about compaction for collection {}",
            collection_id
        );
    }

    /// Check if file can be compacted (async but fast lookup)
    pub async fn can_compact(&self, collection_id: &str, file_path: &str) -> bool {
        self.queue_service
            .can_compact(collection_id, file_path)
            .await
    }

    /// Cleanup after compaction
    pub async fn cleanup_compacted(
        &self,
        collection_id: &str,
        deleted_files: Vec<String>,
    ) -> Result<()> {
        self.queue_service
            .cleanup_compacted_files(collection_id, deleted_files)
            .await
    }

    // Helper methods

    fn detect_representations(_params: &FlushParameters) -> (bool, bool) {
        // Check collection config for quantization
        // Check collection config for quantization - field might not exist
        let has_quantization = false;

        // FP32 is always available unless explicitly quantized-only
        (has_quantization, true)
    }

    fn detect_storage_engine(_params: &FlushParameters) -> StorageEngineType {
        // VIPER was removed (ADR-093); every remaining engine that reports
        // through this notifier maps to the SST-baseline event.
        StorageEngineType::SST
    }

    fn detect_storage_engine_from_compaction(_params: &CompactionParameters) -> StorageEngineType {
        StorageEngineType::SST
    }
}

/// Integration point for storage engines
impl SimpleFlushNotifier {
    /// Called by SST engine after flush
    pub fn sst_flush_complete(
        &self,
        collection_id: &str,
        flushed_files: Vec<String>,
        vector_count: usize,
        has_quantized: bool,
    ) {
        let service = self.queue_service.clone();
        let collection_id = collection_id.to_string();

        tokio::spawn(async move {
            let _ = service
                .add_flush_event(
                    &collection_id,
                    flushed_files,
                    vector_count,
                    StorageEngineType::SST,
                    has_quantized,
                    true, // SST always has FP32
                )
                .await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // MetadataQueueServiceConfig moved or removed
    // use crate::index::axis::metadata::MetadataQueueServiceConfig;
    use crate::proto::proximadb_v1::Collection;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use dashmap::DashMap;
    use tempfile::TempDir;

    async fn create_test_notifier() -> (SimpleFlushNotifier, Arc<MetadataQueueService>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let base_url = format!("file://{}", temp_dir.path().display());

        let mut fs_config = FilesystemConfig::default();
        fs_config.default_fs = Some(base_url.clone());

        let filesystem_factory = Arc::new(FilesystemFactory::create(fs_config).await.unwrap());

        let collection_cache = Arc::new(DashMap::new());
        let mut collection = Collection::default();
        collection.id = "test_collection".to_string();
        collection_cache.insert("test_collection".to_string(), Arc::new(collection));

        // Create EventLogManager (aliased as MetadataQueueService)
        let event_config = crate::index::axis::eventlog::EventLogConfig::default();
        let queue_service =
            MetadataQueueService::new(event_config, filesystem_factory, collection_cache)
                .await
                .unwrap();

        let notifier = SimpleFlushNotifier::new(queue_service.clone());

        (notifier, queue_service, temp_dir)
    }

    #[tokio::test]
    async fn test_notify_flush_never_blocks() {
        let (notifier, queue_service, _dir) = create_test_notifier().await;

        let mut params = FlushParameters::default();
        params.collection_id = Some("test_collection".to_string());

        // This should return immediately
        let start = std::time::Instant::now();
        for i in 0..1000 {
            notifier.notify_flush(&params, vec![format!("file_{}.sstable", i)], 100);
        }
        let elapsed = start.elapsed();

        // Should be near instant (< 10ms for 1000 notifications)
        assert!(
            elapsed.as_millis() < 10,
            "Notify took {:?}, should be instant",
            elapsed
        );

        // Give async tasks time to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify events were queued
        let stats = queue_service.stats().await;
        assert!(stats.total_pending_events > 0);
    }

    #[tokio::test]
    async fn test_compaction_check() {
        let (notifier, _service, _dir) = create_test_notifier().await;

        // Can compact unknown files by default
        assert!(
            notifier
                .can_compact("test_collection", "unknown.sstable")
                .await
        );

        // After adding flush event, file shouldn't be compactable
        let mut params = FlushParameters::default();
        params.collection_id = Some("test_collection".to_string());

        notifier.notify_flush(&params, vec!["file1.sstable".to_string()], 100);

        // Give async task time to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Now file shouldn't be compactable until indexes process it
        assert!(
            !notifier
                .can_compact("test_collection", "file1.sstable")
                .await
        );
    }
}
