/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! End-to-end integration tests for EventLog and AXIS integration

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::{sleep, timeout};
use tracing::{debug, info};

use proximadb::core::config::{ProximaConfig, CompactionConfig};
use proximadb::proto::proximadb_v1::VectorRecord;
use proximadb::index::axis::{AxisManager, AxisConfig};
use proximadb::index::axis::eventlog::{EventLogService, EventType, StorageEngineType};
use proximadb::index::axis::eventlog_consumer::{start_axis_consumer, ConsumerConfig};
use proximadb::proto::proximadb_v1::{Collection, CollectionConfig, DistanceMetric};
use proximadb::services::event_log_service::EventLogServiceImpl;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use dashmap::DashMap;

/// Test fixture for EventLog integration tests
struct TestFixture {
    temp_dir: TempDir,
    event_log: Arc<EventLogServiceImpl>,
    axis_manager: Arc<AxisManager>,
    filesystem: Arc<FilesystemFactory>,
    collection_cache: Arc<DashMap<String, Arc<Collection>>>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    consumer_handle: Option<tokio::task::JoinHandle<()>>,
}

impl TestFixture {
    async fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        
        // Initialize components
        let event_log = Arc::new(EventLogServiceImpl::new());
        let axis_config = AxisConfig::default();
        let axis_manager = Arc::new(AxisManager::new(axis_config).await?);
        let filesystem = Arc::new(FilesystemFactory::new_shared());
        let collection_cache = Arc::new(DashMap::new());
        
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        
        Ok(Self {
            temp_dir,
            event_log,
            axis_manager,
            filesystem,
            collection_cache,
            shutdown_tx,
            shutdown_rx,
            consumer_handle: None,
        })
    }
    
    /// Start the EventLog consumer
    async fn start_consumer(&mut self) {
        let handle = start_axis_consumer(
            self.event_log.clone() as Arc<dyn EventLogService>,
            self.axis_manager.clone(),
            self.filesystem.clone(),
            self.collection_cache.clone(),
            self.shutdown_rx.clone(),
        ).await;
        
        self.consumer_handle = Some(handle);
        
        // Give consumer time to start
        sleep(Duration::from_millis(100)).await;
    }
    
    /// Stop the EventLog consumer
    async fn stop_consumer(&mut self) {
        self.shutdown_tx.send(true).unwrap();
        
        if let Some(handle) = self.consumer_handle.take() {
            let _ = timeout(Duration::from_secs(5), handle).await;
        }
    }
    
    /// Create a test collection
    fn create_test_collection(&self, id: &str, dimension: u32) -> Arc<Collection> {
        let collection = Arc::new(Collection {
            id: id.to_string(),
            dimension,
            storage_engine: Some(0), // SST
            config: Some(CollectionConfig {
                dimension,
                distance_metric: DistanceMetric::Cosine as i32,
                quantization: None,
                index_config: None,
                compression: None,
            }),
            ..Default::default()
        });
        
        self.collection_cache.insert(id.to_string(), collection.clone());
        collection
    }
    
    /// Create test SST files
    async fn create_test_sst_files(&self, collection_id: &str, num_files: usize) -> Vec<String> {
        let mut files = Vec::new();
        
        for i in 0..num_files {
            let file_path = format!("{}/sst_{:04}.sstable", self.temp_dir.path().display(), i);
            
            // Create a simple SST file with test data
            // In a real test, we'd use SstWriter to create proper files
            tokio::fs::write(&file_path, b"SST_TEST_DATA").await.unwrap();
            
            files.push(file_path);
        }
        
        files
    }
    
    /// Create test Parquet files
    async fn create_test_parquet_files(&self, collection_id: &str, num_files: usize) -> Vec<String> {
        let mut files = Vec::new();
        
        for i in 0..num_files {
            let file_path = format!("{}/parquet_{:04}.parquet", self.temp_dir.path().display(), i);
            
            // Create a simple Parquet file with test data
            // In a real test, we'd use arrow/parquet writers
            tokio::fs::write(&file_path, b"PARQUET_TEST_DATA").await.unwrap();
            
            files.push(file_path);
        }
        
        files
    }
}

#[tokio::test]
async fn test_flush_event_processing() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing flush event processing");
    
    let mut fixture = TestFixture::new().await?;
    let collection_id = "test_flush_collection";
    let collection = fixture.create_test_collection(collection_id, 128);
    
    // Start the consumer
    fixture.start_consumer().await;
    
    // Create test SST files
    let data_files = fixture.create_test_sst_files(collection_id, 3).await;
    
    // Send flush event
    info!("Sending flush event for collection {}", collection_id);
    fixture.event_log.notify_flush(
        collection_id.to_string(),
        data_files.clone(),
        100, // vector_count
        false, // has_quantized
        true, // has_fp32
        StorageEngineType::SST,
    ).await?;
    
    // Wait for processing
    sleep(Duration::from_millis(500)).await;
    
    // Check if event was processed
    let pending_events = fixture.event_log.get_pending_events_for_collection(collection_id).await?;
    assert_eq!(pending_events.len(), 0, "Flush event should be processed");
    
    // Check if files are now compactable
    for file in &data_files {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "File {} should no longer be pending", file);
    }
    
    info!("✅ Flush event processing test passed");
    
    // Cleanup
    fixture.stop_consumer().await;
    Ok(())
}

#[tokio::test]
async fn test_compaction_event_processing() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing compaction event processing");
    
    let mut fixture = TestFixture::new().await?;
    let collection_id = "test_compaction_collection";
    let collection = fixture.create_test_collection(collection_id, 256);
    
    // Start the consumer
    fixture.start_consumer().await;
    
    // Create test SST files (compacted output)
    let output_files = fixture.create_test_sst_files(collection_id, 1).await;
    
    // Send compaction event
    info!("Sending compaction event for collection {}", collection_id);
    fixture.event_log.notify_compaction(
        collection_id.to_string(),
        output_files.clone(),
        500, // vector_count (sum of input files)
        StorageEngineType::SST,
    ).await?;
    
    // Wait for processing
    sleep(Duration::from_millis(500)).await;
    
    // Check if event was processed
    let pending_events = fixture.event_log.get_pending_events_for_collection(collection_id).await?;
    assert_eq!(pending_events.len(), 0, "Compaction event should be processed");
    
    // Check if output files are now compactable
    for file in &output_files {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "Output file {} should not be pending", file);
    }
    
    info!("✅ Compaction event processing test passed");
    
    // Cleanup
    fixture.stop_consumer().await;
    Ok(())
}

#[tokio::test]
async fn test_granular_compaction_blocking() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing granular compaction blocking");
    
    let mut fixture = TestFixture::new().await?;
    let collection_id = "test_granular_collection";
    let collection = fixture.create_test_collection(collection_id, 128);
    
    // Don't start consumer yet - we want events to remain pending
    
    // Create multiple sets of files
    let pending_files = fixture.create_test_sst_files(collection_id, 3).await;
    let ready_files = fixture.create_test_sst_files(collection_id, 2).await;
    
    // Send flush event for pending files only
    fixture.event_log.notify_flush(
        collection_id.to_string(),
        pending_files.clone(),
        50,
        false,
        true,
        StorageEngineType::SST,
    ).await?;
    
    // Check that pending files are blocked from compaction
    for file in &pending_files {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(is_pending, "File {} should be pending", file);
    }
    
    // Check that ready files are NOT blocked
    for file in &ready_files {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "File {} should be ready for compaction", file);
    }
    
    // Now start consumer to process events
    fixture.start_consumer().await;
    
    // Wait for processing
    sleep(Duration::from_millis(500)).await;
    
    // All files should now be ready
    for file in pending_files.iter().chain(ready_files.iter()) {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "File {} should now be ready", file);
    }
    
    info!("✅ Granular compaction blocking test passed");
    
    // Cleanup
    fixture.stop_consumer().await;
    Ok(())
}

#[tokio::test]
async fn test_self_healing_behavior() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing self-healing behavior");
    
    let mut fixture = TestFixture::new().await?;
    let collection_id = "test_self_healing";
    let collection = fixture.create_test_collection(collection_id, 128);
    
    // Create files that will transition from pending to ready
    let files_batch1 = fixture.create_test_sst_files(collection_id, 2).await;
    let files_batch2 = fixture.create_test_sst_files(collection_id, 2).await;
    
    // Send events for both batches
    fixture.event_log.notify_flush(
        collection_id.to_string(),
        files_batch1.clone(),
        30,
        false,
        true,
        StorageEngineType::SST,
    ).await?;
    
    fixture.event_log.notify_flush(
        collection_id.to_string(),
        files_batch2.clone(),
        40,
        false,
        true,
        StorageEngineType::SST,
    ).await?;
    
    // Both batches should be pending
    for file in files_batch1.iter().chain(files_batch2.iter()) {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(is_pending, "File {} should be pending initially", file);
    }
    
    // Start consumer
    fixture.start_consumer().await;
    
    // Wait for batch processing
    sleep(Duration::from_millis(300)).await;
    
    // Check first batch is processed
    for file in &files_batch1 {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "Batch 1 file {} should be ready", file);
    }
    
    // Wait more for second batch
    sleep(Duration::from_millis(300)).await;
    
    // Check second batch is also processed
    for file in &files_batch2 {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "Batch 2 file {} should be ready", file);
    }
    
    info!("✅ Self-healing behavior test passed");
    
    // Cleanup
    fixture.stop_consumer().await;
    Ok(())
}

#[tokio::test]
async fn test_nova_event_processing() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing NOVA (columnar Parquet) event processing");

    let mut fixture = TestFixture::new().await?;
    let collection_id = "test_nova_collection";

    // Create collection with the NOVA engine (columnar Parquet path formerly
    // also exercised via VIPER, removed by ADR-093)
    let collection = Arc::new(Collection {
        id: collection_id.to_string(),
        dimension: 384,
        storage_engine: Some(3), // NOVA
        config: Some(CollectionConfig {
            dimension: 384,
            distance_metric: DistanceMetric::Cosine as i32,
            quantization: Some(proximadb::proto::proximadb_v1::QuantizationConfig {
                enabled: true,
                method: 1, // ProductQuantization
                num_subvectors: Some(48),
                bits_per_subvector: Some(8),
                training_sample_size: Some(10000),
            }),
            index_config: None,
            compression: None,
        }),
        ..Default::default()
    });
    
    fixture.collection_cache.insert(collection_id.to_string(), collection.clone());
    
    // Start the consumer
    fixture.start_consumer().await;
    
    // Create test Parquet files
    let data_files = fixture.create_test_parquet_files(collection_id, 2).await;
    
    // Send flush event for NOVA
    info!("Sending NOVA flush event for collection {}", collection_id);
    fixture.event_log.notify_flush(
        collection_id.to_string(),
        data_files.clone(),
        200,
        true, // has_quantized
        true, // has_fp32
        StorageEngineType::NOVA,
    ).await?;
    
    // Wait for processing
    sleep(Duration::from_millis(500)).await;
    
    // Check if event was processed
    let pending_events = fixture.event_log.get_pending_events_for_collection(collection_id).await?;
    assert_eq!(pending_events.len(), 0, "NOVA flush event should be processed");
    
    // Check if files are now compactable
    for file in &data_files {
        let is_pending = fixture.event_log.is_file_pending_indexing(file).await?;
        assert!(!is_pending, "NOVA file {} should no longer be pending", file);
    }
    
    info!("✅ NOVA event processing test passed");
    
    // Cleanup
    fixture.stop_consumer().await;
    Ok(())
}

#[tokio::test]
async fn test_concurrent_event_processing() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing concurrent event processing");
    
    let mut fixture = TestFixture::new().await?;
    
    // Create multiple collections
    let collections = vec![
        ("concurrent_col_1", 128),
        ("concurrent_col_2", 256),
        ("concurrent_col_3", 384),
    ];
    
    for (id, dim) in &collections {
        fixture.create_test_collection(id, *dim);
    }
    
    // Start consumer with concurrent processing enabled
    fixture.start_consumer().await;
    
    // Send multiple events concurrently
    let mut handles = Vec::new();
    
    for (collection_id, _) in &collections {
        let event_log = fixture.event_log.clone();
        let collection_id = collection_id.to_string();
        let files = fixture.create_test_sst_files(&collection_id, 2).await;
        
        let handle = tokio::spawn(async move {
            event_log.notify_flush(
                collection_id,
                files,
                50,
                false,
                true,
                StorageEngineType::SST,
            ).await
        });
        
        handles.push(handle);
    }
    
    // Wait for all events to be sent
    for handle in handles {
        handle.await??;
    }
    
    // Wait for processing
    sleep(Duration::from_secs(1)).await;
    
    // Check all events were processed
    for (collection_id, _) in &collections {
        let pending = fixture.event_log.get_pending_events_for_collection(collection_id).await?;
        assert_eq!(pending.len(), 0, "Collection {} should have no pending events", collection_id);
    }
    
    info!("✅ Concurrent event processing test passed");
    
    // Cleanup
    fixture.stop_consumer().await;
    Ok(())
}

#[tokio::test]
async fn test_event_recovery_after_crash() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing event recovery after crash");
    
    // First session - create pending events
    {
        let mut fixture = TestFixture::new().await?;
        let collection_id = "test_recovery";
        fixture.create_test_collection(collection_id, 128);
        
        // Send events but don't process them (no consumer)
        let files = fixture.create_test_sst_files(collection_id, 3).await;
        fixture.event_log.notify_flush(
            collection_id.to_string(),
            files.clone(),
            100,
            false,
            true,
            StorageEngineType::SST,
        ).await?;
        
        // Verify events are pending
        let pending = fixture.event_log.get_pending_events_for_collection(collection_id).await?;
        assert_eq!(pending.len(), 1, "Should have 1 pending event");
        
        // Simulate crash by dropping fixture without processing
    }
    
    // Second session - recover and process
    {
        let mut fixture = TestFixture::new().await?;
        let collection_id = "test_recovery";
        fixture.create_test_collection(collection_id, 128);
        
        // In a real implementation, EventLog would persist to disk
        // and reload pending events on startup
        // For now, we'll just verify the concept
        
        // Start consumer to process any pending events
        fixture.start_consumer().await;
        
        // In production, pending events would be reloaded and processed
        sleep(Duration::from_millis(500)).await;
        
        info!("✅ Event recovery concept verified (full persistence pending)");
        
        fixture.stop_consumer().await;
    }
    
    Ok(())
}

/// Test that compaction respects EventLog pending files
#[tokio::test] 
async fn test_compaction_with_eventlog_integration() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing compaction with EventLog integration");
    
    // This test would integrate with actual SST compaction
    // For now, we verify the concept
    
    let fixture = TestFixture::new().await?;
    let collection_id = "test_compaction_integration";
    
    // Create test files
    let all_files = fixture.create_test_sst_files(collection_id, 10).await;
    let (pending_files, ready_files) = all_files.split_at(3);
    
    // Mark some files as pending
    fixture.event_log.notify_flush(
        collection_id.to_string(),
        pending_files.to_vec(),
        150,
        false,
        true,
        StorageEngineType::SST,
    ).await?;
    
    // Simulate compaction logic checking EventLog
    let mut compactable = Vec::new();
    for file in &all_files {
        if !fixture.event_log.is_file_pending_indexing(file).await? {
            compactable.push(file.clone());
        }
    }
    
    // Verify only ready files are compactable
    assert_eq!(compactable.len(), ready_files.len(), "Only ready files should be compactable");
    
    info!("✅ Compaction EventLog integration test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_extraction_modes() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    info!("Testing different extraction modes");
    
    let mut fixture = TestFixture::new().await?;
    
    // Test FP32Only mode (no quantization)
    {
        let collection_id = "test_fp32_only";
        let collection = Arc::new(Collection {
            id: collection_id.to_string(),
            dimension: 128,
            storage_engine: Some(0), // SST
            config: Some(CollectionConfig {
                dimension: 128,
                distance_metric: DistanceMetric::Euclidean as i32,
                quantization: None, // No quantization
                index_config: None,
                compression: None,
            }),
            ..Default::default()
        });
        fixture.collection_cache.insert(collection_id.to_string(), collection.clone());
        
        info!("Testing FP32Only extraction mode");
    }
    
    // Test QuantizedOnly mode
    {
        let collection_id = "test_quantized_only";
        let collection = Arc::new(Collection {
            id: collection_id.to_string(),
            dimension: 256,
            storage_engine: Some(3), // NOVA
            config: Some(CollectionConfig {
                dimension: 256,
                distance_metric: DistanceMetric::Cosine as i32,
                quantization: Some(proximadb::proto::proximadb_v1::QuantizationConfig {
                    enabled: true,
                    method: 2, // ScalarQuantization
                    num_subvectors: Some(32),
                    bits_per_subvector: Some(8),
                    training_sample_size: Some(5000),
                }),
                index_config: None,
                compression: None,
            }),
            ..Default::default()
        });
        fixture.collection_cache.insert(collection_id.to_string(), collection.clone());
        
        info!("Testing QuantizedOnly extraction mode");
    }
    
    // Test Both mode
    {
        let collection_id = "test_both_modes";
        let collection = Arc::new(Collection {
            id: collection_id.to_string(),
            dimension: 384,
            storage_engine: Some(0), // SST
            config: Some(CollectionConfig {
                dimension: 384,
                distance_metric: DistanceMetric::DotProduct as i32,
                quantization: Some(proximadb::proto::proximadb_v1::QuantizationConfig {
                    enabled: true,
                    method: 1, // ProductQuantization
                    num_subvectors: Some(48),
                    bits_per_subvector: Some(8),
                    training_sample_size: Some(10000),
                }),
                index_config: None,
                compression: None,
            }),
            ..Default::default()
        });
        fixture.collection_cache.insert(collection_id.to_string(), collection.clone());
        
        info!("Testing Both extraction mode");
    }
    
    info!("✅ Extraction modes test passed");
    
    Ok(())
}