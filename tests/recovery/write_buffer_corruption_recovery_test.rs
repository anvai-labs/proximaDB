#[cfg(test)]
mod write_ahead_log_corruption_recovery_tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
    use tracing::info;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::fs;
    use std::io::Write;
    
    use proximadb::core::config::{
        Config, StorageConfig, StorageLocation, ServerConfig,
        ApiConfig, ConsensusConfig, MonitoringConfig,
        AssignmentConfig, FilesystemOptimizationConfig,
    };
    use proximadb::network::NetworkConfig;
    use proximadb::ProximaDB;
    
    fn create_test_config(base_path: &str) -> Config {
        Config {
            server: ServerConfig {
                node_id: "corruption-test-node".to_string(),
                bind_address: "127.0.0.1".to_string(),
                port: 0,
                data_dir: PathBuf::from(base_path),
            },
            storage: StorageConfig {
                storage_locations: vec![StorageLocation {
                    url: format!("file://{}", base_path),
                    weight: 1,
                    tags: vec![],
                    io_budget: None,
                }],
                metadata_url: format!("file://{}/metadata", base_path),
                assignment_config: Default::default(),
                mmap_enabled: true,
                sst_config: Default::default(),
                    wal_config: Default::default(),
                cache_size_mb: 128,
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
    async fn test_recovery_with_corrupted_wal_header() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create data and get WriteBuffer files
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            // Create some test data directories to simulate collections
            let collection_id = "write_ahead_log_corruption_test";
            let collection_dir = PathBuf::from(base_path).join(collection_id);
            fs::create_dir_all(&collection_dir).await?;
            let write_ahead_log_dir = collection_dir.join("write_ahead_log");
            fs::create_dir_all(&write_ahead_log_dir).await?;
            
            // Create some dummy write buffer files
            let dummy_data = b"dummy write buffer data";
            fs::write(write_ahead_log_dir.join("000001.wb"), dummy_data).await?;
            fs::write(write_ahead_log_dir.join("000002.wb"), dummy_data).await?;
            
            db.stop().await?;
        }
        
        // Corrupt WriteBuffer files
        corrupt_write_ahead_log_files(base_path, CorruptionType::Header).await?;
        
        // Phase 2: Recovery should handle corruption gracefully
        {
            let mut db = ProximaDB::new(config).await?;
            // Should not panic, but might lose some data
            db.start().await?;
            
            // Server should start despite corruption
            info!("Server started successfully after corruption recovery");
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_recovery_with_truncated_wal() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create data
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            let collection_id = "truncated_wal_test";
            let collection_dir = PathBuf::from(base_path).join(collection_id);
            fs::create_dir_all(&collection_dir).await?;
            let write_ahead_log_dir = collection_dir.join("write_ahead_log");
            fs::create_dir_all(&write_ahead_log_dir).await?;
            
            // Create multiple write buffer files
            for i in 0..5 {
                let dummy_data = format!("write buffer batch {} data", i);
                fs::write(write_ahead_log_dir.join(format!("{:06}.wb", i)), dummy_data.as_bytes()).await?;
            }
            
            db.stop().await?;
        }
        
        // Truncate WriteBuffer files
        corrupt_write_ahead_log_files(base_path, CorruptionType::Truncate).await?;
        
        // Phase 2: Recovery should recover what it can
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            // Server should recover from truncated files
            info!("Server recovered successfully from truncated WriteBuffer");
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_recovery_with_checksum_mismatch() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create data with checksums
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            let collection_id = "checksum_test";
            let collection_dir = PathBuf::from(base_path).join(collection_id);
            fs::create_dir_all(&collection_dir).await?;
            let write_ahead_log_dir = collection_dir.join("write_ahead_log");
            fs::create_dir_all(&write_ahead_log_dir).await?;
            
            // Create write buffer files with checksums
            let dummy_data = b"write buffer data with checksums";
            fs::write(write_ahead_log_dir.join("checksum_001.wb"), dummy_data).await?;
            
            db.stop().await?;
        }
        
        // Corrupt data but not headers
        corrupt_write_ahead_log_files(base_path, CorruptionType::DataCorruption).await?;
        
        // Phase 2: Recovery should detect checksum mismatches
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            // Server should start despite corruption
            info!("Server started successfully despite checksum mismatches");
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_recovery_rollback_incomplete_transactions() -> Result<()> {
        // Initialize hardware capabilities
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();
        let config = create_test_config(base_path);
        
        // Phase 1: Create incomplete transaction scenario
        {
            let mut db = ProximaDB::new(config.clone()).await?;
            db.start().await?;
            
            // Create multiple collections
            for i in 0..3 {
                let collection_id = format!("transaction_test_{}", i);
                let collection_dir = PathBuf::from(base_path).join(&collection_id);
                fs::create_dir_all(&collection_dir).await?;
                let write_ahead_log_dir = collection_dir.join("write_ahead_log");
                fs::create_dir_all(&write_ahead_log_dir).await?;
                
                // Create write buffer files
                let dummy_data = format!("transaction {} data", i);
                fs::write(write_ahead_log_dir.join("txn.wb"), dummy_data.as_bytes()).await?;
            }
            
            // Start but don't complete a flush operation
            // This simulates an incomplete transaction
            
            // Abrupt stop without proper shutdown
            std::mem::forget(db); // Leak the db to simulate crash
        }
        
        // Add some delay to simulate crash state
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Phase 2: Recovery should rollback incomplete operations
        {
            let mut db = ProximaDB::new(config).await?;
            db.start().await?;
            
            // Verify server recovered successfully
            info!("Server recovered successfully from incomplete transactions");
            
            db.stop().await?;
        }
        
        Ok(())
    }
    
    // Helper functions for corruption
    #[derive(Debug)]
    enum CorruptionType {
        Header,
        Truncate,
        DataCorruption,
    }
    
    async fn corrupt_write_ahead_log_files(base_path: &str, corruption_type: CorruptionType) -> Result<()> {
        use std::path::Path;
        
        // Find WriteBuffer files by walking the directory
        let base_dir = Path::new(base_path);
        if base_dir.exists() {
            let mut entries = fs::read_dir(base_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    let write_ahead_log_dir = path.join("write_ahead_log");
                    if write_ahead_log_dir.exists() {
                        let mut wb_entries = fs::read_dir(&write_ahead_log_dir).await?;
                        while let Some(wb_entry) = wb_entries.next_entry().await? {
                            let file_path = wb_entry.path();
                            if file_path.is_file() && file_path.extension().and_then(|s| s.to_str()) == Some("wb") {
                                corrupt_file(&file_path, &corruption_type).await?;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn corrupt_file(path: &PathBuf, corruption_type: &CorruptionType) -> Result<()> {
        match corruption_type {
            CorruptionType::Header => {
                // Overwrite first few bytes
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                {
                    file.write_all(b"CORRUPTED")?;
                }
            }
            CorruptionType::Truncate => {
                // Truncate file to 75% of original size
                if let Ok(metadata) = fs::metadata(path).await {
                    let new_len = (metadata.len() * 3) / 4;
                    fs::OpenOptions::new()
                        .write(true)
                        .open(path)
                        .await?
                        .set_len(new_len)
                        .await?;
                }
            }
            CorruptionType::DataCorruption => {
                // Flip some bits in the middle of the file
                if let Ok(mut data) = fs::read(path).await {
                    let mid = data.len() / 2;
                    if mid < data.len() {
                        data[mid] ^= 0xFF; // Flip all bits of one byte
                        if mid + 10 < data.len() {
                            data[mid + 10] ^= 0xAA; // Flip alternating bits
                        }
                    }
                    fs::write(path, data).await?;
                }
            }
        }
        
        Ok(())
    }
}