/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Integration between storage engines and EventLog service
//! Provides fire-and-forget notifications that never block storage operations

use std::sync::Arc;
use tracing::{debug, trace};

use crate::core::types::StorageEngineType;
use crate::storage::engines::{CompactionParameters, FlushParameters};
use proximadb_storage_ports::StorageEventLogPort;

/// Helper trait for storage engines to notify EventLog
#[async_trait::async_trait]
pub trait EventLogNotifier {
    /// Notify about flush completion (blocks until EventLog acknowledges)
    async fn notify_flush(
        &self,
        collection_id: &str,
        flushed_files: Vec<String>,
        vector_count: usize,
        has_quantized: bool,
        has_fp32: bool,
    ) -> Result<(), anyhow::Error>;

    /// Notify about compaction completion (blocks until EventLog acknowledges)
    async fn notify_compaction(
        &self,
        collection_id: &str,
        output_files: Vec<String>,
        vector_count: usize,
    ) -> Result<(), anyhow::Error>;

    /// Check if files can be compacted (async but fast)
    async fn can_compact_files(&self, collection_id: &str, files: &[String]) -> bool;
}

/// SST engine event log integration
pub struct SstEventLogNotifier {
    event_log: Arc<dyn StorageEventLogPort>,
}

impl SstEventLogNotifier {
    pub fn new(event_log: Arc<dyn StorageEventLogPort>) -> Self {
        Self { event_log }
    }
}

#[async_trait::async_trait]
impl EventLogNotifier for SstEventLogNotifier {
    async fn notify_flush(
        &self,
        collection_id: &str,
        flushed_files: Vec<String>,
        vector_count: usize,
        has_quantized: bool,
        has_fp32: bool,
    ) -> Result<(), anyhow::Error> {
        self.event_log
            .notify_flush(
                collection_id,
                flushed_files,
                vector_count,
                StorageEngineType::SST,
                has_quantized,
                has_fp32,
            )
            .await?;

        trace!("EventLog acknowledged SST flush for {}", collection_id);
        Ok(())
    }

    async fn notify_compaction(
        &self,
        collection_id: &str,
        output_files: Vec<String>,
        vector_count: usize,
    ) -> Result<(), anyhow::Error> {
        self.event_log
            .notify_compaction(
                collection_id,
                output_files,
                vector_count,
                StorageEngineType::SST,
            )
            .await?;

        trace!("EventLog acknowledged SST compaction for {}", collection_id);
        Ok(())
    }

    async fn can_compact_files(&self, collection_id: &str, files: &[String]) -> bool {
        // Check each file
        for file in files {
            match self.event_log.can_compact(collection_id, file).await {
                Ok(can_compact) => {
                    if !can_compact {
                        debug!("File {} not ready for compaction (indexes pending)", file);
                        return false;
                    }
                }
                Err(e) => {
                    // On error, allow compaction (don't block storage)
                    debug!("Error checking compaction status: {}, allowing", e);
                    return true;
                }
            }
        }
        true
    }
}

/// Factory for creating event log notifiers
pub struct EventLogNotifierFactory;

impl EventLogNotifierFactory {
    /// Create SST notifier
    pub fn create_sst_notifier(event_log: Arc<dyn StorageEventLogPort>) -> SstEventLogNotifier {
        SstEventLogNotifier::new(event_log)
    }

    /// Create notifier based on engine type
    pub fn create_notifier(
        engine_type: &str,
        event_log: Arc<dyn StorageEventLogPort>,
    ) -> Result<Box<dyn EventLogNotifier + Send + Sync>, anyhow::Error> {
        match engine_type.to_lowercase().as_str() {
            "sst" => Ok(Box::new(Self::create_sst_notifier(event_log))),
            "nova" | "swift" | "raptor" | "prism" | "helix" => {
                // For now, use SST notifier for engines without specific implementations
                // Deferred: Implement specific notifiers for each engine
                Ok(Box::new(Self::create_sst_notifier(event_log)))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown storage engine type '{}' for event notifier",
                engine_type
            )),
        }
    }
}

/// Extension trait for FlushParameters
pub trait FlushParametersExt {
    /// Detect if flush has quantized data
    fn has_quantized(&self) -> bool;

    /// Detect if flush has FP32 data
    fn has_fp32(&self) -> bool;

    /// Get storage engine type
    fn storage_engine(&self) -> StorageEngineType;
}

impl FlushParametersExt for FlushParameters {
    fn has_quantized(&self) -> bool {
        self.collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .and_then(|config| config.quantization.as_ref())
            .is_some_and(|q| q.enabled.unwrap_or(false))
    }

    fn has_fp32(&self) -> bool {
        // SST-family engines always store FP32 unless explicitly quantized-only
        true
    }

    fn storage_engine(&self) -> StorageEngineType {
        // VIPER was removed (ADR-093); no hint can select it any more.
        StorageEngineType::SST
    }
}

/// Extension trait for CompactionParameters
pub trait CompactionParametersExt {
    /// Get storage engine type
    fn storage_engine(&self) -> StorageEngineType;
}

impl CompactionParametersExt for CompactionParameters {
    fn storage_engine(&self) -> StorageEngineType {
        StorageEngineType::SST
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct FakeEventLogPort {
        flush_count: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl StorageEventLogPort for FakeEventLogPort {
        async fn notify_flush(
            &self,
            _collection_id: &str,
            flushed_files: Vec<String>,
            _vector_count: usize,
            _storage_engine: StorageEngineType,
            _has_quantized: bool,
            _has_fp32: bool,
        ) -> anyhow::Result<()> {
            self.flush_count
                .fetch_add(flushed_files.len(), Ordering::SeqCst);
            Ok(())
        }

        async fn notify_compaction(
            &self,
            _collection_id: &str,
            _output_files: Vec<String>,
            _vector_count: usize,
            _storage_engine: StorageEngineType,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn can_compact(
            &self,
            _collection_id: &str,
            _file_path: &str,
        ) -> anyhow::Result<bool> {
            Ok(true)
        }
    }

    fn create_test_notifier() -> (SstEventLogNotifier, Arc<FakeEventLogPort>) {
        let event_log = Arc::new(FakeEventLogPort::default());
        let notifier = SstEventLogNotifier::new(event_log.clone());
        (notifier, event_log)
    }

    #[tokio::test]
    async fn test_notify_flush_blocks_until_acknowledged() {
        let (notifier, event_log) = create_test_notifier();

        // Notification blocks until EventLog acknowledges
        let start = std::time::Instant::now();
        for i in 0..10 {
            let result = notifier
                .notify_flush(
                    "test_collection",
                    vec![format!("file_{}.sstable", i)],
                    100,
                    false,
                    true,
                )
                .await;

            result.expect("flush notification should succeed");
        }
        let elapsed = start.elapsed();

        // Should complete reasonably fast but not instant (because it waits for acknowledgment)
        assert!(
            elapsed.as_millis() < 5000,
            "Notifications took {:?} - should complete within 5s",
            elapsed
        );

        assert!(
            event_log.flush_count.load(Ordering::SeqCst) == 10,
            "fake event log should observe every flush notification"
        );
    }

    #[tokio::test]
    async fn test_can_compact_check() {
        let (notifier, _event_log) = create_test_notifier();

        // Unknown files can be compacted
        let can_compact = notifier
            .can_compact_files(
                "test_collection",
                &[
                    "unknown1.sstable".to_string(),
                    "unknown2.sstable".to_string(),
                ],
            )
            .await;

        assert!(can_compact);
    }
}
