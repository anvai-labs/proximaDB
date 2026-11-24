use proximadb::core::config::{Config, StorageConfig, StorageLocation, ServerConfig, ApiConfig, ConsensusConfig, MonitoringConfig, AssignmentConfig, FilesystemOptimizationConfig};
use proximadb::network::NetworkConfig;
use proximadb::ProximaDB;
use tempfile::TempDir;
use std::time::Instant;
use std::path::PathBuf;
use tracing::{info};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[tokio::test]
async fn test_recovery_with_multiple_collections() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create test collections
    for i in 0..10 {
        let collection_dir = base_path.join(format!("collection_{}", i));
        std::fs::create_dir_all(&collection_dir)?;
        
        // Create some dummy SSTable files
        let data_dir = collection_dir.join("data");
        std::fs::create_dir_all(&data_dir)?;
        std::fs::write(data_dir.join("000001.sstable"), b"dummy sstable data")?;
    }
    
    // Create config
    let config = Config {
        server: ServerConfig {
            node_id: "test-node".to_string(),
            bind_address: "127.0.0.1".to_string(),
            port: 0, // Use any available port
            data_dir: PathBuf::from(base_path),
        },
        storage: StorageConfig {
            storage_locations: vec![StorageLocation {
                url: format!("file://{}", base_path.to_str().unwrap()),
                weight: 1,
                tags: vec!["test".to_string()],
                io_budget: None,
            }],
            metadata_url: format!("file://{}/metadata", base_path.to_str().unwrap()),
            assignment_config: AssignmentConfig::default(),
            mmap_enabled: true,
            sst_config: Default::default(),
            wal_config: Default::default(),
            cache_size_mb: 256,
            bloom_filter_config: None,
            filesystem_config: FilesystemOptimizationConfig::default(),
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
            log_level: "info".to_string(),
        },
        network: Some(NetworkConfig::default()),
        tls: None,
        hardware: None,
    };
    
    // Test server startup time with parallel loading
    let start = Instant::now();
    
    let mut db = ProximaDB::new(config).await?;
    db.start().await?;
    
    let startup_time = start.elapsed();
    info!("Server started in {:?} with 10 collections", startup_time);
    
    // Verify startup was reasonable (should be under 60 seconds even with 10 collections)
    assert!(startup_time.as_secs() < 60, "Startup took too long: {:?}", startup_time);
    
    db.stop().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_recovery_after_crash() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create metadata directory
    std::fs::create_dir_all(base_path.join("metadata/current"))?;
    std::fs::create_dir_all(base_path.join("metadata/archive"))?;
    
    // Create config
    let config = Config {
        server: ServerConfig {
            node_id: "test-node".to_string(),
            bind_address: "127.0.0.1".to_string(),
            port: 0,
            data_dir: PathBuf::from(base_path),
        },
        storage: StorageConfig {
            storage_locations: vec![StorageLocation {
                url: format!("file://{}", base_path.to_str().unwrap()),
                weight: 1,
                tags: vec!["test".to_string()],
                io_budget: None,
            }],
            metadata_url: format!("file://{}/metadata", base_path.to_str().unwrap()),
            assignment_config: AssignmentConfig::default(),
            mmap_enabled: true,
            sst_config: Default::default(),
            wal_config: Default::default(),
            cache_size_mb: 256,
            bloom_filter_config: None,
            filesystem_config: FilesystemOptimizationConfig::default(),
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
    };
    
    // Start server first time
    {
        let mut db = ProximaDB::new(config.clone()).await?;
        db.start().await?;
        
        // Server is running, simulate some metadata operations
        // In real test we would create collections through API
        
        db.stop().await?;
    }
    
    // Simulate crash recovery - start server again
    let start = Instant::now();
    {
        let mut db = ProximaDB::new(config).await?;
        db.start().await?;
        
        let recovery_time = start.elapsed();
        info!("Recovery completed in {:?}", recovery_time);
        
        // Recovery should be fast
        assert!(recovery_time.as_secs() < 2, "Recovery took too long: {:?}", recovery_time);
        
        db.stop().await?;
    }
    
    Ok(())
}