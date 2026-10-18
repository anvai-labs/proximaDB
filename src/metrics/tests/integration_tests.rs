//! Integration tests for metrics collection across VectorOperationsService, FlushCoordinator, and BackgroundManager

#[cfg(test)]
mod tests {
    use super::super::super::{MetricsConfig, store::MetricsPersistenceLayer};
    use crate::compute::distance_computation::DistanceMetric;
    use crate::metrics::InternalMetricsUpdater;
    use crate::metrics::updater::{
        MetricsUpdateService, OperationMetricsUpdate, SearchMetricsUpdate,
    };
    use crate::storage::background_flush_context::{
        BackgroundFlushContext, CompressionConfig, OperationPriority, StorageEngineType,
    };
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use crate::storage::persistence::write_ahead_log::{
        WALConfig, background_manager::BackgroundMaintenanceManager,
        flush_coordinator::WALFlushCoordinator,
    };
    use crate::storage::traits::{
        CompactionParameters, CompactionResult, FlushParameters, FlushResult, UnifiedStorageFormat,
    };
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::{Mutex, RwLock};
    use tokio::time::{Duration, sleep};
    use tracing::{debug, info};

    /// Mock storage engine for integration testing
    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    struct MockStorageEngineWithMetrics {
        engine_name: String,
        operation_calls: Arc<Mutex<Vec<String>>>,
    }

    impl MockStorageEngineWithMetrics {
        fn new(name: &str) -> Self {
            Self {
                engine_name: name.to_string(),
                operation_calls: Arc::new(Mutex::new(Vec::new())),
            }
        }

        async fn get_operation_calls(&self) -> Vec<String> {
            self.operation_calls.lock().await.clone()
        }
    }

    #[async_trait::async_trait]
    impl UnifiedStorageFormat for MockStorageEngineWithMetrics {
        fn engine_name(&self) -> &'static str {
            "mock_engine"
        }

        async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
            let collection_id = params
                .collection_id
                .as_ref()
                .unwrap_or(&"unknown".to_string())
                .clone();

            self.operation_calls
                .lock()
                .await
                .push(format!("flush:{}", collection_id));

            debug!(
                "🧪 MockEngine: Flush called for collection {} with {} vectors",
                collection_id,
                params.vector_records.len()
            );

            // Simulate measurable flush work. The trait `flush()` wrapper
            // (trait_components/writer.rs) is authoritative for timing — it
            // overwrites `duration_ms` with the real elapsed wall clock of
            // `do_flush`. Without spending real time here, an instant mock
            // intermittently yields `duration_ms == 0`, flaking the metrics
            // assertion `last_flush_duration_ms > 0`. The `Some(100)` below is
            // discarded by that wrapper; the sleep is what makes the recorded
            // duration reliably non-zero.
            sleep(Duration::from_millis(2)).await;

            Ok(FlushResult {
                success: true,
                collections_affected: vec![collection_id],
                entries_flushed: Some(params.vector_records.len() as u64),
                bytes_written: Some((params.vector_records.len() * 1024) as u64),
                files_created: Some(1),
                file_paths: vec![],
                duration_ms: Some(100),
                completed_at: chrono::Utc::now(),
                engine_metrics: HashMap::new(),
                compaction_triggered: false,
                compaction_error: None,
                flushed_batch_ids: params.batch_ids.clone(),
            })
        }

        async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
            let collection_id = params
                .collection_id
                .as_ref()
                .unwrap_or(&"unknown".to_string())
                .clone();

            self.operation_calls
                .lock()
                .await
                .push(format!("compact:{}", collection_id));

            debug!(
                "🧪 MockEngine: Compaction called for collection {}",
                collection_id
            );

            Ok(CompactionResult {
                success: true,
                collections_affected: vec![collection_id],
                entries_processed: Some(1000),
                entries_removed: Some(100),
                bytes_read: Some(10 * 1024 * 1024),
                bytes_written: Some(6 * 1024 * 1024),
                input_files: Some(5),
                output_files: Some(2),
                duration_ms: Some(200),
                completed_at: chrono::Utc::now(),
                engine_metrics: HashMap::new(),
            })
        }

        // Implement other required methods with minimal functionality
        fn engine_version(&self) -> &'static str {
            "integration-test-1.0"
        }

        fn strategy(&self) -> crate::storage::traits::StorageFormatStrategy {
            crate::storage::traits::StorageFormatStrategy::Viper
        }

        async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
            Ok(HashMap::new())
        }

        async fn vector_by_id(
            &self,
            _collection_id: &str,
            _base_path: &str,
            _vector_id: &str,
        ) -> Result<Option<proximadb_records::ProximaRecord>> {
            Ok(None)
        }

        async fn search_vectors_unified(
            &self,
            _ctx: &crate::storage::traits::StorageQueryContext,
        ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
            self.operation_calls
                .lock()
                .await
                .push("search_vectors_unified".to_string());
            Ok(Vec::new())
        }
    }

    async fn create_test_metrics_components()
    -> Result<(Arc<MetricsUpdateService>, Arc<MetricsPersistenceLayer>)> {
        create_test_metrics_components_with_cleanup(false).await
    }

    async fn create_test_metrics_components_with_cleanup(
        _cleanup: bool,
    ) -> Result<(Arc<MetricsUpdateService>, Arc<MetricsPersistenceLayer>)> {
        // Unique tempdir per call so tests don't collide on a shared path — the
        // file-backed MetricsPersistenceLayer accumulates state across runs,
        // making total_flushes assertions flaky. (Tests that need a SHARED path
        // across "restarts" use `create_test_metrics_components_at` directly.)
        let temp_dir = tempfile::TempDir::new()?;
        let storage_path = format!("file://{}", temp_dir.path().display());
        std::mem::forget(temp_dir);
        create_test_metrics_components_at(storage_path).await
    }

    /// Create test metrics components at an explicit storage path — for tests
    /// that need a SHARED path across "restarts" (e.g. persistence-across-restarts).
    async fn create_test_metrics_components_at(
        storage_path: String,
    ) -> Result<(Arc<MetricsUpdateService>, Arc<MetricsPersistenceLayer>)> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let config = MetricsConfig {
            enabled: true,
            collection_partitions: 4,
            storage_path,
            flush_interval_seconds: 30,
            retention_days: 7,
            parallel_scan_threshold: 10,
            sparsity_threshold: 0.3,
            quantization_size_threshold: 1_000_000,
            snapshot_interval_seconds: 60,
            max_memory_mb: 512,
        };

        let filesystem_config = Default::default();
        let filesystem_factory = Arc::new(FilesystemFactory::create(filesystem_config).await?);
        let store = Arc::new(MetricsPersistenceLayer::new(filesystem_factory, config).await?);
        let updater = Arc::new(MetricsUpdateService::new(store.clone()));

        Ok((updater, store))
    }

    fn create_test_context(
        collection_id: &str,
        engine_type: StorageEngineType,
    ) -> BackgroundFlushContext {
        BackgroundFlushContext {
            collection_id: collection_id.to_string(),
            storage_engine: engine_type,
            base_location: format!("/tmp/integration_test/{}", collection_id),
            dimension: 384,
            distance_metric: DistanceMetric::Cosine,
            compression_config: CompressionConfig::default(),
            filterable_columns: Vec::new(),
            quantization: None,
            batch_size_hint: Some(1000),
            priority: OperationPriority::Normal,
            timeout_ms: Some(60_000),
            extra_metadata: HashMap::new(),
            tenant_id: None,
            collection_meta: None,
        }
    }

    fn create_test_vectors(count: usize) -> Vec<proximadb_records::ProximaRecord> {
        (0..count)
            .map(|i| proximadb_records::ProximaRecord {
                oid: format!("integration_vector_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell {
                    model_id: "default".to_string(),
                    modality: "dense_vector".to_string(),
                    dim: 384,
                    values: proximadb_records::EmbeddingValues::Fp32(vec![0.1; 384]),
                    ..Default::default()
                }],
                record_version: 1,
                ..proximadb_records::ProximaRecord::default()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_directvectorservice_metrics_integration() {
        // Initialize hardware capabilities for testing
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: VectorOperationsService metrics integration (simulated)");

        let (metrics_updater, metrics_store) = create_test_metrics_components().await.unwrap();

        // Since VectorOperationsService constructor requires engines which are not available in this test,
        // we simulate what VectorOperationsService would do by directly recording metrics
        let collection_id = "directvectorservice_integration_test";

        // Simulate insert operation with metrics
        let insert_start = std::time::Instant::now();

        // Create test vectors
        let test_vectors = create_test_vectors(100);
        debug!("📊 Simulating insert of {} vectors", test_vectors.len());

        // Manually record insert metrics (simulating what VectorOperationsService would do)
        let insert_duration = insert_start.elapsed().as_micros() as f64;
        let operation_update = OperationMetricsUpdate {
            operation_type: "insert".to_string(),
            latency_us: insert_duration,
            success: true,
            bytes_processed: test_vectors.len() * 384 * 4, // Assuming 4 bytes per float
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        metrics_updater
            .record_operation(collection_id, operation_update)
            .await
            .unwrap();

        // Simulate search operation with metrics
        let search_update = SearchMetricsUpdate {
            query_latency_us: 1200.0,
            results_count: 10,
            vectors_scanned: 10000,
            cache_hit: true,
            index_used: "hnsw_direct_test".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        metrics_updater
            .record_search(collection_id, search_update)
            .await
            .unwrap();

        // Allow metrics processing time
        sleep(Duration::from_millis(200)).await;

        // Verify metrics were recorded
        let stored_metrics = metrics_store
            .collection_metrics(collection_id)
            .await
            .unwrap();
        assert!(
            stored_metrics.is_some(),
            "VectorOperationsService metrics should be stored"
        );

        let collection_metrics = stored_metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, collection_id);
        assert!(
            collection_metrics.total_inserts > 0,
            "Insert operations should be recorded"
        );
        assert!(
            collection_metrics.total_searches > 0,
            "Search operations should be recorded"
        );
        assert!(
            collection_metrics.avg_insert_latency_us > 0.0,
            "Insert latency should be recorded"
        );
        assert!(
            collection_metrics.avg_search_latency_us > 0.0,
            "Search latency should be recorded"
        );

        debug!(
            "📊 VectorOperationsService metrics: {} inserts, {} searches",
            collection_metrics.total_inserts, collection_metrics.total_searches
        );

        info!("✅ VectorOperationsService metrics integration test passed");
    }

    #[tokio::test]
    async fn test_flushcoordinator_metrics_integration() {
        // Initialize hardware capabilities for testing
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: FlushCoordinator metrics integration");

        let (metrics_updater, metrics_store) = create_test_metrics_components().await.unwrap();
        let mut flush_coordinator = WALFlushCoordinator::new();

        // Register metrics updater with FlushCoordinator
        flush_coordinator.set_metrics_updater(metrics_updater.clone());

        // Create and register mock storage engine
        let mock_engine = Arc::new(MockStorageEngineWithMetrics::new("viper"));
        flush_coordinator
            .register_storage_engine("viper", mock_engine.clone())
            .await;

        let collection_id = "flushcoordinator_integration_test";
        let test_vectors = create_test_vectors(50);
        let context = create_test_context(collection_id, StorageEngineType::Viper);

        // Execute coordinated flush with metrics
        let flush_data = crate::storage::persistence::write_ahead_log::flush_coordinator::FlushDataSource::VectorRecords(
            test_vectors
        );

        let flush_result = flush_coordinator
            .execute_coordinated_flush(collection_id, flush_data, None, Some(&context))
            .await;

        assert!(
            flush_result.is_ok(),
            "Flush with metrics should succeed: {:?}",
            flush_result
        );

        // Allow metrics processing time
        sleep(Duration::from_millis(300)).await;

        // Verify flush metrics were recorded
        let stored_metrics = metrics_store
            .collection_metrics(collection_id)
            .await
            .unwrap();
        assert!(
            stored_metrics.is_some(),
            "FlushCoordinator metrics should be stored"
        );

        let collection_metrics = stored_metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, collection_id);
        assert!(
            collection_metrics.total_flushes > 0,
            "Flush operations should be recorded"
        );
        assert!(
            collection_metrics.last_flush_duration_ms > 0,
            "Flush duration should be recorded"
        );
        assert!(
            collection_metrics.last_flush_timestamp > 0,
            "Flush timestamp should be recorded"
        );

        // Verify storage engine was called
        let engine_calls = mock_engine.get_operation_calls().await;
        assert!(
            !engine_calls.is_empty(),
            "Storage engine should have been called"
        );
        assert!(engine_calls.iter().any(|call| call.starts_with("flush:")));

        debug!(
            "📊 FlushCoordinator metrics: {} flushes, last duration: {}ms",
            collection_metrics.total_flushes, collection_metrics.last_flush_duration_ms
        );

        info!("✅ FlushCoordinator metrics integration test passed");
    }

    #[tokio::test]
    async fn test_backgroundmanager_metrics_integration() {
        // Initialize hardware capabilities for testing
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: BackgroundManager metrics integration");

        let (metrics_updater, metrics_store) = create_test_metrics_components().await.unwrap();

        let config = Arc::new(WALConfig::default());
        let bg_manager = BackgroundMaintenanceManager::new(config);

        // Register metrics updater with BackgroundManager
        // TODO: Add set_metrics_updater to BackgroundMaintenanceManager
        // bg_manager.set_metrics_updater(metrics_updater.clone());

        // Create and register mock storage engine
        let mock_engine = Arc::new(MockStorageEngineWithMetrics::new("viper"));
        bg_manager
            .register_storage_engine("viper", mock_engine.clone())
            .await
            .unwrap();

        let collection_id = "backgroundmanager_integration_test";
        let context = create_test_context(collection_id, StorageEngineType::Viper);

        // Create storage engines registry for compaction
        let storage_engines = Arc::new(RwLock::new({
            let mut engines = HashMap::new();
            engines.insert(
                "viper".to_string(),
                mock_engine.clone() as Arc<dyn UnifiedStorageFormat>,
            );
            engines
        }));

        // Execute compaction with metrics
        let metrics_updater_dyn: Arc<dyn InternalMetricsUpdater> = metrics_updater.clone();
        let compaction_result = BackgroundMaintenanceManager::execute_compaction_with_context(
            &storage_engines,
            &context,
            Some(&metrics_updater_dyn),
        )
        .await;

        assert!(
            compaction_result.is_ok(),
            "Compaction with metrics should succeed: {:?}",
            compaction_result
        );

        // Allow metrics processing time
        sleep(Duration::from_millis(300)).await;

        // Verify compaction metrics were recorded
        let stored_metrics = metrics_store
            .collection_metrics(collection_id)
            .await
            .unwrap();
        assert!(
            stored_metrics.is_some(),
            "BackgroundManager metrics should be stored"
        );

        let collection_metrics = stored_metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, collection_id);
        assert!(
            collection_metrics.total_compactions > 0,
            "Compaction operations should be recorded"
        );
        assert!(
            collection_metrics.last_compaction_duration_ms > 0,
            "Compaction duration should be recorded"
        );
        assert!(
            collection_metrics.last_compaction_timestamp > 0,
            "Compaction timestamp should be recorded"
        );

        // Verify storage engine was called
        let engine_calls = mock_engine.get_operation_calls().await;
        assert!(
            !engine_calls.is_empty(),
            "Storage engine should have been called"
        );
        assert!(engine_calls.iter().any(|call| call.starts_with("compact:")));

        debug!(
            "📊 BackgroundManager metrics: {} compactions, last duration: {}ms",
            collection_metrics.total_compactions, collection_metrics.last_compaction_duration_ms
        );

        info!("✅ BackgroundManager metrics integration test passed");
    }

    #[tokio::test]
    async fn test_end_to_end_metrics_collection() {
        // Initialize hardware capabilities for testing
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: End-to-end metrics collection across all components");

        let (metrics_updater, metrics_store) = create_test_metrics_components().await.unwrap();

        // Set up all components with metrics
        let mut flush_coordinator = WALFlushCoordinator::new();
        flush_coordinator.set_metrics_updater(metrics_updater.clone());

        let config = Arc::new(WALConfig::default());
        let bg_manager = BackgroundMaintenanceManager::new(config);
        // Note: bg_manager doesn't have set_metrics_updater method, which is fine for this test

        // Create mock storage engine
        let mock_engine = Arc::new(MockStorageEngineWithMetrics::new("viper"));
        flush_coordinator
            .register_storage_engine("viper", mock_engine.clone())
            .await;
        bg_manager
            .register_storage_engine("viper", mock_engine.clone())
            .await
            .unwrap();

        let collection_id = "end_to_end_integration_test";
        let context = create_test_context(collection_id, StorageEngineType::Viper);

        // Step 1: Simulate VectorOperationsService operations
        let operation_update = OperationMetricsUpdate {
            operation_type: "insert".to_string(),
            latency_us: 300.0,
            success: true,
            bytes_processed: 50 * 384 * 4,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        metrics_updater
            .record_operation(collection_id, operation_update)
            .await
            .unwrap();

        let search_update = SearchMetricsUpdate {
            query_latency_us: 1800.0,
            results_count: 20,
            vectors_scanned: 15000,
            cache_hit: false,
            index_used: "hnsw_e2e".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        metrics_updater
            .record_search(collection_id, search_update)
            .await
            .unwrap();

        // Step 2: Execute FlushCoordinator operation
        let test_vectors = create_test_vectors(25);
        let flush_data = crate::storage::persistence::write_ahead_log::flush_coordinator::FlushDataSource::VectorRecords(
            test_vectors
        );

        let flush_result = flush_coordinator
            .execute_coordinated_flush(collection_id, flush_data, None, Some(&context))
            .await;
        assert!(flush_result.is_ok());

        // Step 3: Execute BackgroundManager compaction
        let storage_engines = Arc::new(RwLock::new({
            let mut engines = HashMap::new();
            engines.insert(
                "viper".to_string(),
                mock_engine.clone() as Arc<dyn UnifiedStorageFormat>,
            );
            engines
        }));

        let metrics_updater_dyn: Arc<dyn InternalMetricsUpdater> = metrics_updater.clone();
        let compaction_result = BackgroundMaintenanceManager::execute_compaction_with_context(
            &storage_engines,
            &context,
            Some(&metrics_updater_dyn),
        )
        .await;
        assert!(compaction_result.is_ok());

        // Allow all metrics to be processed
        sleep(Duration::from_millis(500)).await;

        // Step 4: Verify comprehensive metrics collection
        let stored_metrics = metrics_store
            .collection_metrics(collection_id)
            .await
            .unwrap();
        assert!(
            stored_metrics.is_some(),
            "End-to-end metrics should be stored"
        );

        let collection_metrics = stored_metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, collection_id);

        // Verify all operation types were recorded
        assert!(
            collection_metrics.total_inserts > 0,
            "VectorOperationsService inserts should be recorded"
        );
        assert!(
            collection_metrics.total_searches > 0,
            "VectorOperationsService searches should be recorded"
        );
        assert!(
            collection_metrics.total_flushes > 0,
            "FlushCoordinator flushes should be recorded"
        );
        assert!(
            collection_metrics.total_compactions > 0,
            "BackgroundManager compactions should be recorded"
        );

        // Verify latency metrics
        assert!(
            collection_metrics.avg_insert_latency_us > 0.0,
            "Insert latency should be recorded"
        );
        assert!(
            collection_metrics.avg_search_latency_us > 0.0,
            "Search latency should be recorded"
        );

        // Verify timestamps
        assert!(
            collection_metrics.last_flush_timestamp > 0,
            "Flush timestamp should be recorded"
        );
        assert!(
            collection_metrics.last_compaction_timestamp > 0,
            "Compaction timestamp should be recorded"
        );
        assert!(
            collection_metrics.updated_at > 0,
            "Updated timestamp should be current"
        );

        // Verify storage engine operations occurred
        let engine_calls = mock_engine.get_operation_calls().await;
        assert!(
            engine_calls.iter().any(|call| call.starts_with("flush:")),
            "Flush operation should have occurred"
        );
        assert!(
            engine_calls.iter().any(|call| call.starts_with("compact:")),
            "Compaction operation should have occurred"
        );

        debug!("📊 End-to-end metrics summary:");
        debug!(
            "   📈 Inserts: {}, Searches: {}",
            collection_metrics.total_inserts, collection_metrics.total_searches
        );
        debug!(
            "   💾 Flushes: {}, Compactions: {}",
            collection_metrics.total_flushes, collection_metrics.total_compactions
        );
        debug!(
            "   ⏱️  Avg Insert: {:.1}µs, Avg Search: {:.1}µs",
            collection_metrics.avg_insert_latency_us, collection_metrics.avg_search_latency_us
        );
        debug!("   🔧 Storage operations: {:?}", engine_calls);

        info!("✅ End-to-end metrics collection test passed");
    }

    #[tokio::test]
    async fn test_metrics_collection_with_multiple_collections() {
        // Initialize hardware capabilities for testing
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: Metrics collection with multiple collections");

        let (metrics_updater, metrics_store) = create_test_metrics_components_with_cleanup(true)
            .await
            .unwrap();

        let collections = vec![
            "multi_collection_test_001",
            "multi_collection_test_002",
            "multi_collection_test_003",
        ];

        // Set up FlushCoordinator with metrics
        let mut flush_coordinator = WALFlushCoordinator::new();
        flush_coordinator.set_metrics_updater(metrics_updater.clone());

        let mock_engine = Arc::new(MockStorageEngineWithMetrics::new("viper"));
        flush_coordinator
            .register_storage_engine("viper", mock_engine.clone())
            .await;

        // Process operations for each collection
        for (i, collection_id) in collections.iter().enumerate() {
            let context = create_test_context(collection_id, StorageEngineType::Viper);

            // Record different metrics for each collection
            let operation_update = OperationMetricsUpdate {
                operation_type: "insert".to_string(),
                latency_us: 200.0 + (i as f64 * 100.0),
                success: true,
                bytes_processed: (i + 1) * 1000,
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            metrics_updater
                .record_operation(collection_id, operation_update)
                .await
                .unwrap();

            // Execute flush for each collection
            let test_vectors = create_test_vectors(10 + (i * 5));
            let flush_data = crate::storage::persistence::write_ahead_log::flush_coordinator::FlushDataSource::VectorRecords(
                test_vectors
            );

            flush_coordinator
                .execute_coordinated_flush(collection_id, flush_data, None, Some(&context))
                .await
                .unwrap();
        }

        // Allow metrics processing
        sleep(Duration::from_millis(400)).await;

        // Verify metrics for each collection
        for collection_id in &collections {
            let stored_metrics = metrics_store
                .collection_metrics(collection_id)
                .await
                .unwrap();
            assert!(
                stored_metrics.is_some(),
                "Metrics should exist for collection {}",
                collection_id
            );

            let collection_metrics = stored_metrics.unwrap();
            assert_eq!(collection_metrics.collection_id, *collection_id);
            assert!(collection_metrics.total_inserts > 0);
            assert!(collection_metrics.total_flushes > 0);

            debug!(
                "📊 Collection '{}': {} inserts, {} flushes",
                collection_id, collection_metrics.total_inserts, collection_metrics.total_flushes
            );
        }

        // Verify collection list includes all test collections
        let collection_list = metrics_store.list_collections().await.unwrap();
        for collection_id in &collections {
            assert!(
                collection_list.contains(&collection_id.to_string()),
                "Collection list should include {}",
                collection_id
            );
        }

        info!("✅ Multiple collections metrics test passed");
    }

    #[tokio::test]
    async fn test_metrics_persistence_across_restarts() {
        // Initialize hardware capabilities for testing
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        debug!("🧪 TEST: Metrics persistence across component restarts");

        let collection_id = "persistence_test_collection";

        // Shared storage path across both phases so Phase 2 reads what Phase 1
        // persisted (simulates a restart on the same disk). Leaked (forget) so it
        // outlives both component instances.
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage_path = format!("file://{}", temp_dir.path().display());
        std::mem::forget(temp_dir);

        // Phase 1: Create initial metrics store and record data
        {
            let (metrics_updater, _) = create_test_metrics_components_at(storage_path.clone())
                .await
                .unwrap();

            let operation_update = OperationMetricsUpdate {
                operation_type: "insert".to_string(),
                latency_us: 400.0,
                success: true,
                bytes_processed: 2000,
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            metrics_updater
                .record_operation(collection_id, operation_update)
                .await
                .unwrap();

            // Allow processing and persistence
            sleep(Duration::from_millis(200)).await;
        } // Components go out of scope

        // Phase 2: Create new metrics store (simulating restart) and verify persistence
        {
            let (_, metrics_store) = create_test_metrics_components_at(storage_path.clone())
                .await
                .unwrap();

            let stored_metrics = metrics_store
                .collection_metrics(collection_id)
                .await
                .unwrap();
            assert!(
                stored_metrics.is_some(),
                "Metrics should persist across restarts"
            );

            let collection_metrics = stored_metrics.unwrap();
            assert_eq!(collection_metrics.collection_id, collection_id);
            assert!(
                collection_metrics.total_inserts > 0,
                "Persisted insert count should be > 0"
            );

            debug!(
                "📊 Persisted metrics: {} inserts, updated at {}",
                collection_metrics.total_inserts, collection_metrics.updated_at
            );
        }

        info!("✅ Metrics persistence test passed");
    }
}
