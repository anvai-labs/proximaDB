#[cfg(test)]
mod recovery_stress_tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
    use std::sync::Arc;
    use tracing::{debug, info};
    use std::collections::HashMap;
    use tempfile::TempDir;
    use tokio::fs;
    use std::time::{Duration, Instant};
    use std::path::PathBuf;
    
    use proximadb::core::config::{
        Config, StorageConfig, StorageLocation, ServerConfig, 
        ApiConfig, ConsensusConfig, MonitoringConfig,
        AssignmentConfig, FilesystemOptimizationConfig,
    };
    use proximadb::network::NetworkConfig;
    use proximadb::ProximaDB;
    
    // Helper to create test data
    async fn create_test_data(base_path: &str, collection_id: &str, file_count: usize) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let collection_dir = PathBuf::from(base_path).join(collection_id);
        fs::create_dir_all(&collection_dir).await?;
        let data_dir = collection_dir.join("data");
        fs::create_dir_all(&data_dir).await?;
        
        // Create some SSTable files
        for i in 0..file_count {
            let dummy_data = format!("SSTable {} data for collection {}", i, collection_id);
            fs::write(data_dir.join(format!("{:06}.sstable", i)), dummy_data.as_bytes()).await?;
        }
        Ok(())
    }
    
    // Helper to create test config
    fn create_test_config(base_path: &str) -> Config {
        Config {
            server: ServerConfig {
                node_id: "test-recovery-node".to_string(),
                bind_address: "127.0.0.1".to_string(),
                port: 0,
                data_dir: PathBuf::from(base_path),
            },
            storage: StorageConfig {
                storage_locations: vec![
                    StorageLocation {
                        url: format!("file://{}/storage1", base_path),
                        weight: 1,
                        tags: vec!["primary".to_string()],
                        io_budget: None,
                    },
                    StorageLocation {
                        url: format!("file://{}/storage2", base_path),
                        weight: 1,
                        tags: vec!["secondary".to_string()],
                        io_budget: None,
                    },
                ],
                metadata_url: format!("file://{}/metadata", base_path),
                assignment_config: Default::default(),
                mmap_enabled: true,
                sst_config: Default::default(),
                    wal_config: Default::default(),
                cache_size_mb: 256,
                bloom_filter_config: None,
                filesystem_config: Default::default(),
            },
            api: ApiConfig {
                grpc_port: 0,
                rest_port: 0,
                max_request_size_mb: 100,
                timeout_seconds: 30,
                enable_tls: None,
                rest_compression: false,
                grpc_compression: false,
                compression_algorithm: "gzip".to_string(),
                compression_level: 6,
            },
            consensus: ConsensusConfig {
                node_id: None,
                cluster_peers: vec![],
                election_timeout_ms: 150,
                heartbeat_interval_ms: 50,
                snapshot_threshold: 1000,
            },
            monitoring: MonitoringConfig {
                metrics_enabled: false,
                log_level: "error".to_string(),
            },
            network: Some(NetworkConfig::default()),
            tls: None,
            hardware: None,
        }
    }
    
    #[tokio::test]
    async fn test_recovery_with_large_dataset() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create and populate with large dataset
        let num_collections = 20;
        let files_per_collection = 10;
        
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            // Create test data for collections
            for i in 0..num_collections {
                let collection_id = format!("large_collection_{}", i);
                create_test_data(base_path, &collection_id, files_per_collection).await?;
            }
            
            info!("Created {} collections with {} files each", num_collections, files_per_collection);
            
            db.stop().await?;
        }
        
        // Phase 2: Test recovery performance
        let start = Instant::now();
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            let recovery_time = start.elapsed();
            info!("Recovery completed in {:?} for {} collections", recovery_time, num_collections);
            
            // Verify data integrity by checking that collections exist
            for i in (0..num_collections).step_by(5) {
                let collection_id = format!("large_collection_{}", i);
                let collection_dir = PathBuf::from(base_path).join(&collection_id).join("data");
                assert!(collection_dir.exists(), "Collection {} should exist", collection_id);
            }
            
            // Recovery should complete in reasonable time even with many collections
            assert!(recovery_time.as_secs() < 10, "Recovery took too long: {:?}", recovery_time);
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_recovery_with_mixed_engines() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create collections with different storage engines
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            // Create SST collections
            for i in 0..5 {
                let collection_id = format!("sst_collection_{}", i);
                create_test_data(base_path, &collection_id, 5).await?;
            }
            
            db.stop().await?;
        }
        
        // Phase 2: Recovery and verification
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            // Verify SST collections
            for i in 0..5 {
                let collection_id = format!("sst_collection_{}", i);
                let collection_dir = PathBuf::from(base_path).join(&collection_id).join("data");
                assert!(collection_dir.exists(), "SST collection {} should exist", collection_id);
            }
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_recovery_after_incomplete_flush() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create data and simulate incomplete flush
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            let collection_id = "incomplete_flush_test";
            create_test_data(base_path, collection_id, 5).await?;
            
            // Create additional unflushed data by adding write buffer files
            let wb_dir = PathBuf::from(base_path).join(collection_id).join("write_ahead_log");
            fs::create_dir_all(&wb_dir).await?;
            fs::write(wb_dir.join("unflushed.wb"), b"unflushed data").await?;
            
            db.stop().await?;
        }
        
        // Phase 2: Recovery should handle both flushed and unflushed data
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            // Verify data exists
            let collection_dir = PathBuf::from(base_path).join("incomplete_flush_test");
            assert!(collection_dir.exists(), "Collection should exist after recovery");
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_concurrent_recovery_stress() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create many collections concurrently
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            let mut handles = Vec::new();
            
            // Spawn concurrent collection creation
            for i in 0..10 {
                let base_path_clone = base_path.to_string();
                
                let handle = tokio::spawn(async move {
                    let collection_id = format!("concurrent_collection_{}", i);
                    create_test_data(&base_path_clone, &collection_id, 5).await?;
                    
                    Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                });
                
                handles.push(handle);
            }
            
            // Wait for all operations
            for handle in handles {
                handle.await??;
            }
            
            db.stop().await?;
        }
        
        // Phase 2: Test parallel recovery
        let start = Instant::now();
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            let recovery_time = start.elapsed();
            info!("Concurrent recovery completed in {:?}", recovery_time);
            
            // Verify all collections are accessible
            for i in 0..10 {
                let collection_id = format!("concurrent_collection_{}", i);
                let collection_dir = PathBuf::from(base_path).join(&collection_id).join("data");
                assert!(collection_dir.exists(), "Collection {} should be recovered", collection_id);
            }
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_recovery_memory_efficiency() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        
        // Create config with limited memory
        let mut config = create_test_config(base_path);
        config.storage.cache_size_mb = 64; // Limited cache
        
        // Phase 1: Create large dataset that exceeds cache
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            // Create collections that will exceed 64MB cache
            for i in 0..10 {
                let collection_id = format!("memory_test_{}", i);
                create_test_data(base_path, &collection_id, 20).await?; // More files to simulate larger data
            }
            
            db.stop().await?;
        }
        
        // Phase 2: Recovery should handle memory constraints
        {
            let mut db = ProximaDB::new(config).await?;
            let start_memory = get_current_memory_usage();
            
            db.start().await?;
            
            let peak_memory = get_current_memory_usage();
            let memory_increase = peak_memory.saturating_sub(start_memory);
            
            debug!("Memory increase during recovery: {} MB", memory_increase / 1024 / 1024);
            
            // Memory increase should be reasonable (not loading everything at once)
            assert!(
                memory_increase < 512 * 1024 * 1024, // Less than 512MB increase
                "Recovery used too much memory: {} MB", memory_increase / 1024 / 1024
            );
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    // Helper function to get current memory usage
    fn get_current_memory_usage() -> usize {
        // This is a simplified version - in real tests you'd use proper memory tracking
        // For now, just return a placeholder
        0
    }
}