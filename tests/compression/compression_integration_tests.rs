// ORPHANED TEST (not in any compile target — see TD-ORPHAN-1): references the
// VIPER engine removed by ADR-093 stage 3 / TD-VIPER-1 S3. Unreachable code;
// cleanup of the file itself belongs to the orphan-file program, not the
// engine-removal PR.
// Compression Integration Tests
// Tests the complete flow from SDK to storage and retrieval

// TODO: This test file needs to be updated to use the new API structure
// Commenting out for now as it uses outdated imports and APIs

/*
use proximadb::proto::proximadb_v1::{
    CollectionRequest, InsertRequest, SearchRequest, SearchQuery,
    VectorRecord, OptimizationHints, CompressionConfig as ProtoCompressionConfig,
    CompressionAlgorithm as ProtoCompressionAlgorithm,
};
use proximadb::services::direct_vector_service::DirectVectorService;
use proximadb::storage::engines::sst::{CompressionAlgorithmSst};
use proximadb::storage::engines::viper::ViperEngine;
use proximadb::core::models::{Collection, CollectionConfig, DistanceMetric};
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn test_sst_compression_end_to_end() {
    // Initialize hardware capabilities for testing
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    // Create temporary directory for storage
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_str().unwrap();
    
    // Create DirectVectorService with required parameters
    let write_buffer_config = proximadb::storage::persistence::write_buffer::WriteBufferConfig::default();
    let viper_engine = Arc::new(ViperEngine::new().await.unwrap());
    let sst_storage = Arc::new(proximadb::storage::engines::sst::SstEngine::new().await.unwrap());
    let service = DirectVectorService::new(write_buffer_config, viper_engine, sst_storage).await.unwrap();
    
    // Create collection with SST compression via SDK (unified structure)
    let compression_config = ProtoCompressionConfig {
        algorithm: ProtoCompressionAlgorithm::CompressionZstd as i32,
        level: Some(6),
        adaptive: true,
        min_ratio: Some(0.5),
        enable_quantization: false,
        quantization_type: None,
        normalization_method: None,
        block_size_mb: Some(32),
        dynamic_block_sizing: false,
    };
    
    let collection_config = CollectionConfig {
        id: "test_compressed".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: Some(compression_config.clone()),
    };
    
    // Create collection
    service.create_collection(collection_config.clone()).await.unwrap();
    
    // Insert vectors with compression
    let mut vectors = vec![];
    for i in 0..1000 {
        vectors.push(VectorRecord {
            id: Some(format!("vec_{}", i)),
            vector: vec![i as f32 / 1000.0; 128],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        });
    }
    
    service.insert_vectors("test_compressed", vectors.clone()).await.unwrap();
    
    // Force flush to create compressed SST files
    service.flush_collection("test_compressed").await.unwrap();
    
    // Search with compression-aware hints
    let query = SearchQuery {
        collection_id: "test_compressed".to_string(),
        vector: vec![0.5; 128],
        k: 10,
        filter: None,
        include_metadata: true,
        include_vectors: false,
        radius: None,
        optimization_hints: Some(OptimizationHints {
            prefer_compressed_search: true,
            decompression_budget_ms: Some(100),
            use_decompression_cache: true,
            compression_aware_routing: true,
            enable_two_stage: false,
            coarse_quantization_k: None,
            rerank_multiplier: None,
            allow_approximate: false,
            target_recall: None,
            cache_key: Some("test_query".to_string()),
            prefetch_count: Some(5),
            parallel_decompression: true,
            adaptive_k: false,
            use_hnsw_ef_runtime: None,
            skip_reranking: false,
            batch_mode: false,
            streaming_mode: false,
            hardware_hints: None,
            predicate_pushdown: false,
        }),
    };
    
    let results = service.search_with_planner(&query).await.unwrap();
    
    // Verify results
    assert_eq!(results.len(), 10);
    assert!(results[0].similarity > 0.9);
    
    // Verify compression was applied
    let stats = service.get_compression_stats("test_compressed").await.unwrap();
    assert!(stats.compressed_files > 0);
    assert!(stats.compression_ratio < 1.0);
    
    // Test cache hit on second query
    let results2 = service.search_with_planner(&query).await.unwrap();
    assert_eq!(results2.len(), 10);
    
    let cache_stats = service.get_cache_stats().await.unwrap();
    assert!(cache_stats.hits > 0);
}

#[tokio::test]
async fn test_viper_dual_column_compression() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_str().unwrap();
    
    let write_buffer_config = proximadb::storage::persistence::write_buffer::WriteBufferConfig::default();
    let viper_engine = Arc::new(ViperEngine::new().await.unwrap());
    let sst_storage = Arc::new(proximadb::storage::engines::sst::SstEngine::new().await.unwrap());
    let service = DirectVectorService::new(write_buffer_config, viper_engine, sst_storage).await.unwrap();
    
    // Create collection with VIPER dual columns
    let compression_config = ProtoCompressionConfig {
        algorithm: ProtoCompressionAlgorithm::CompressionZstd as i32,
        level: Some(9),
        adaptive: false,
        min_ratio: None,
        enable_quantization: true,
        quantization_type: Some("int8".to_string()),
        normalization_method: Some("mean".to_string()),
        block_size_mb: None,
        dynamic_block_sizing: false,
    };
    
    let collection_config = CollectionConfig {
        id: "test_viper_dual".to_string(),
        dimension: 256,
        distance_metric: DistanceMetric::Euclidean,
        index_type: "ivf".to_string(),
        storage_engine: "viper".to_string(),
        compression_config: Some(compression_config),
    };
    
    service.create_collection(collection_config).await.unwrap();
    
    // Insert vectors
    let mut vectors = vec![];
    for i in 0..500 {
        vectors.push(VectorRecord {
            id: Some(format!("viper_{}", i)),
            vector: vec![i as f32 / 500.0; 256],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        });
    }
    
    service.insert_vectors("test_viper_dual", vectors).await.unwrap();
    service.flush_collection("test_viper_dual").await.unwrap();
    
    // Search with two-stage enabled
    let query = SearchQuery {
        collection_id: "test_viper_dual".to_string(),
        vector: vec![0.3; 256],
        k: 5,
        filter: None,
        include_metadata: false,
        include_vectors: false,
        radius: None,
        optimization_hints: Some(OptimizationHints {
            prefer_compressed_search: false,
            decompression_budget_ms: None,
            use_decompression_cache: false,
            compression_aware_routing: false,
            enable_two_stage: true,
            coarse_quantization_k: Some(50),
            rerank_multiplier: Some(2.0),
            allow_approximate: false,
            target_recall: Some(0.95),
            cache_key: None,
            prefetch_count: None,
            parallel_decompression: false,
            adaptive_k: false,
            use_hnsw_ef_runtime: None,
            skip_reranking: false,
            batch_mode: false,
            streaming_mode: false,
            hardware_hints: None,
            predicate_pushdown: false,
        }),
    };
    
    let results = service.search_with_planner(&query).await.unwrap();
    
    assert_eq!(results.len(), 5);
    assert!(results[0].similarity > 0.0);
}

#[tokio::test]
async fn test_mixed_compression_query_planning() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_str().unwrap();
    
    let write_buffer_config = proximadb::storage::persistence::write_buffer::WriteBufferConfig::default();
    let viper_engine = Arc::new(ViperEngine::new().await.unwrap());
    let sst_storage = Arc::new(proximadb::storage::engines::sst::SstEngine::new().await.unwrap());
    let service = DirectVectorService::new(write_buffer_config, viper_engine, sst_storage).await.unwrap();
    
    // Create multiple collections with different compression settings
    
    // Collection 1: SST with ZSTD
    let config1 = CollectionConfig {
        id: "sst_zstd".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: Some(ProtoCompressionConfig {
            algorithm: ProtoCompressionAlgorithm::CompressionZstd as i32,
            level: Some(3),
            adaptive: false,
            min_ratio: None,
            enable_quantization: false,
            quantization_type: None,
            normalization_method: None,
            block_size_mb: None,
            dynamic_block_sizing: false,
        }),
    };
    
    // Collection 2: SST with LZ4
    let config2 = CollectionConfig {
        id: "sst_lz4".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: Some(ProtoCompressionConfig {
            algorithm: ProtoCompressionAlgorithm::CompressionLz4 as i32,
            level: Some(1),
            adaptive: false,
            min_ratio: None,
            enable_quantization: false,
            quantization_type: None,
            normalization_method: None,
            block_size_mb: None,
            dynamic_block_sizing: false,
        }),
    };
    
    // Collection 3: VIPER with dual columns
    let config3 = CollectionConfig {
        id: "viper_dual".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        index_type: "hnsw".to_string(),
        storage_engine: "viper".to_string(),
        compression_config: Some(ProtoCompressionConfig {
            algorithm: ProtoCompressionAlgorithm::CompressionZstd as i32,
            level: Some(3),
            adaptive: false,
            min_ratio: None,
            enable_quantization: true,
            quantization_type: Some("int8".to_string()),
            normalization_method: Some("mean".to_string()),
            block_size_mb: None,
            dynamic_block_sizing: false,
        }),
    };
    
    // Collection 4: No compression
    let config4 = CollectionConfig {
        id: "no_compression".to_string(),
        dimension: 64,
        distance_metric: DistanceMetric::Cosine,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: None,
    };
    
    // Create all collections
    service.create_collection(config1).await.unwrap();
    service.create_collection(config2).await.unwrap();
    service.create_collection(config3).await.unwrap();
    service.create_collection(config4).await.unwrap();
    
    // Insert test data into each
    for collection_id in &["sst_zstd", "sst_lz4", "viper_dual", "no_compression"] {
        let mut vectors = vec![];
        for i in 0..100 {
            vectors.push(VectorRecord {
                id: Some(format!("{}_{}", collection_id, i)),
                vector: vec![i as f32 / 100.0; 64],
                metadata: vec![],
                timestamp: 0,
                updated_at: None,
                expires_at: None,
                version: None,
                distance: None,
                rank: None,
                score: None,
            });
        }
        service.insert_vectors(collection_id, vectors).await.unwrap();
        service.flush_collection(collection_id).await.unwrap();
    }
    
    // Test SQL query across mixed compression
    let sql = r#"
        SELECT id, VECTOR_SIMILARITY(vector, [0.5; 64], "cosine") as similarity
        FROM sst_zstd
        ORDER BY similarity DESC
        LIMIT 5
    "#;
    
    let results = service.execute_sql_with_planner(sql).await.unwrap();
    assert_eq!(results.len(), 5);
    
    // Verify planner statistics
    let planner_stats = service.get_planner_stats().await.unwrap();
    assert!(planner_stats.queries_planned > 0);
    assert!(planner_stats.compression_aware_routes > 0);
}

#[tokio::test]
async fn test_cache_invalidation_on_updates() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_str().unwrap();
    
    let write_buffer_config = proximadb::storage::persistence::write_buffer::WriteBufferConfig::default();
    let viper_engine = Arc::new(ViperEngine::new().await.unwrap());
    let sst_storage = Arc::new(proximadb::storage::engines::sst::SstEngine::new().await.unwrap());
    let service = DirectVectorService::new(write_buffer_config, viper_engine, sst_storage).await.unwrap();
    
    // Create compressed collection
    let config = CollectionConfig {
        id: "cache_test".to_string(),
        dimension: 32,
        distance_metric: DistanceMetric::Cosine,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: Some(ProtoCompressionConfig {
            algorithm: ProtoCompressionAlgorithm::CompressionSnappy as i32,
            level: Some(1),
            adaptive: false,
            min_ratio: None,
            enable_quantization: false,
            quantization_type: None,
            normalization_method: None,
            block_size_mb: None,
            dynamic_block_sizing: false,
        }),
    };
    
    service.create_collection(config).await.unwrap();
    
    // Insert initial data
    let mut vectors = vec![];
    for i in 0..50 {
        vectors.push(VectorRecord {
            id: Some(format!("cache_{}", i)),
            vector: vec![i as f32 / 50.0; 32],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        });
    }
    
    service.insert_vectors("cache_test", vectors.clone()).await.unwrap();
    service.flush_collection("cache_test").await.unwrap();
    
    // First search to populate cache
    let query = SearchQuery {
        collection_id: "cache_test".to_string(),
        vector: vec![0.5; 32],
        k: 5,
        filter: None,
        include_metadata: false,
        include_vectors: false,
        radius: None,
        optimization_hints: Some(OptimizationHints {
            use_decompression_cache: true,
            cache_key: Some("cache_test_query".to_string()),
            ..Default::default()
        }),
    };
    
    let results1 = service.search_with_planner(&query).await.unwrap();
    assert_eq!(results1.len(), 5);
    
    // Check cache was populated
    let cache_stats1 = service.get_cache_stats().await.unwrap();
    let initial_size = cache_stats1.current_size_bytes;
    assert!(initial_size > 0);
    
    // Update collection (should invalidate cache)
    let new_vectors = vec![
        VectorRecord {
            id: Some("cache_new".to_string()),
            vector: vec![0.5; 32],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        }
    ];
    
    service.insert_vectors("cache_test", new_vectors).await.unwrap();
    service.flush_collection("cache_test").await.unwrap();
    
    // Cache should be invalidated
    let cache_stats2 = service.get_cache_stats().await.unwrap();
    assert!(cache_stats2.current_size_bytes < initial_size);
    
    // Search again to verify new data is found
    let results2 = service.search_with_planner(&query).await.unwrap();
    assert_eq!(results2.len(), 5);
    
    // New vector should be in results
    let found = results2.iter().any(|r| r.id == "cache_new");
    assert!(found);
}

#[tokio::test]
async fn test_adaptive_compression_threshold() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_str().unwrap();
    
    let write_buffer_config = proximadb::storage::persistence::write_buffer::WriteBufferConfig::default();
    let viper_engine = Arc::new(ViperEngine::new().await.unwrap());
    let sst_storage = Arc::new(proximadb::storage::engines::sst::SstEngine::new().await.unwrap());
    let service = DirectVectorService::new(write_buffer_config, viper_engine, sst_storage).await.unwrap();
    
    // Create collection with adaptive compression
    let config = CollectionConfig {
        id: "adaptive_test".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: Some(ProtoCompressionConfig {
            algorithm: ProtoCompressionAlgorithm::CompressionZstd as i32,
            level: Some(6),
            adaptive: true,
            min_ratio: Some(0.5),
            enable_quantization: false,
            quantization_type: None,
            normalization_method: None,
            block_size_mb: Some(10), // Only compress blocks > 10KB
            dynamic_block_sizing: false,
        }),
    };
    
    service.create_collection(config).await.unwrap();
    
    // Insert small batch (should not be compressed)
    let small_batch = vec![
        VectorRecord {
            id: Some("small_1".to_string()),
            vector: vec![0.1; 128],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        }
    ];
    
    service.insert_vectors("adaptive_test", small_batch).await.unwrap();
    service.flush_collection("adaptive_test").await.unwrap();
    
    // Insert large batch (should be compressed)
    let mut large_batch = vec![];
    for i in 0..1000 {
        large_batch.push(VectorRecord {
            id: Some(format!("large_{}", i)),
            vector: vec![i as f32 / 1000.0; 128],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        });
    }
    
    service.insert_vectors("adaptive_test", large_batch).await.unwrap();
    service.flush_collection("adaptive_test").await.unwrap();
    
    // Verify mixed compression status
    let stats = service.get_compression_stats("adaptive_test").await.unwrap();
    assert!(stats.compressed_files > 0);
    assert!(stats.uncompressed_files > 0);
}

#[tokio::test]
async fn test_parallel_decompression() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_str().unwrap();
    
    let write_buffer_config = proximadb::storage::persistence::write_buffer::WriteBufferConfig::default();
    let viper_engine = Arc::new(ViperEngine::new().await.unwrap());
    let sst_storage = Arc::new(proximadb::storage::engines::sst::SstEngine::new().await.unwrap());
    let service = DirectVectorService::new(write_buffer_config, viper_engine, sst_storage).await.unwrap();
    
    // Create highly compressed collection
    let config = CollectionConfig {
        id: "parallel_test".to_string(),
        dimension: 512,
        distance_metric: DistanceMetric::DotProduct,
        index_type: "flat".to_string(),
        storage_engine: "sst".to_string(),
        compression_config: Some(ProtoCompressionConfig {
            algorithm: ProtoCompressionAlgorithm::CompressionZstd as i32,
            level: Some(9), // High compression
            adaptive: false,
            min_ratio: None,
            enable_quantization: false,
            quantization_type: None,
            normalization_method: None,
            block_size_mb: Some(8), // Small blocks for more parallelism
            dynamic_block_sizing: false,
        }),
    };
    
    service.create_collection(config).await.unwrap();
    
    // Insert large dataset
    let mut vectors = vec![];
    for i in 0..5000 {
        vectors.push(VectorRecord {
            id: Some(format!("parallel_{}", i)),
            vector: vec![i as f32 / 5000.0; 512],
            metadata: vec![],
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            version: None,
            distance: None,
            rank: None,
            score: None,
        });
    }
    
    service.insert_vectors("parallel_test", vectors).await.unwrap();
    service.flush_collection("parallel_test").await.unwrap();
    
    // Search with parallel decompression
    let query = SearchQuery {
        collection_id: "parallel_test".to_string(),
        vector: vec![0.5; 512],
        k: 100,
        filter: None,
        include_metadata: false,
        include_vectors: false,
        radius: None,
        optimization_hints: Some(OptimizationHints {
            parallel_decompression: true,
            decompression_budget_ms: Some(200),
            prefetch_count: Some(10),
            ..Default::default()
        }),
    };
    
    let start = std::time::Instant::now();
    let results = service.search_with_planner(&query).await.unwrap();
    let elapsed = start.elapsed();
    
    assert_eq!(results.len(), 100);
    
    // Parallel decompression should be faster
    assert!(elapsed.as_millis() < 500);
}*/
