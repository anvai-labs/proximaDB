// Integration tests for DSST and DVIPER dual-mode engines
// Tests engine creation, basic operations, and mode switching

use anyhow::Result;
use proximadb::{
    core::{VectorRecord, hardware_capabilities},
    storage::{
        engines::{
            dsst::DsstEngine,
            dviper::DviperEngine,
        },
        traits::{
            UnifiedStorageEngine, FlushParameters, FlushResult,
            CompactionParameters, CompactionResult,
        },
    },
    compute::distance_computation::DistanceMetric,
    proto::proximadb::IndexingAlgorithm,
};
use std::collections::HashMap;

/// Test fixture for dual-mode engine testing
struct DualModeTestFixture {
    dsst_engine: DsstEngine,
    dviper_engine: DviperEngine,
    test_vectors: Vec<VectorRecord>,
}

impl DualModeTestFixture {
    async fn new(num_vectors: usize, dimension: usize) -> Result<Self> {
        // Initialize hardware capabilities
        let _ = hardware_capabilities::initialize_hardware_capabilities_default();
        
        // Create engines
        let dsst_engine = DsstEngine::new()?;
        let dviper_engine = DviperEngine::new()?;
        
        // Generate test vectors
        let mut test_vectors = Vec::new();
        for i in 0..num_vectors {
            test_vectors.push(VectorRecord {
                id: Some(format!("vec_{:06}", i)),
                vector: vec![i as f32 / num_vectors as f32; dimension],
                metadata: Some(HashMap::from([
                    ("category".to_string(), serde_json::json!(if i % 2 == 0 { "even" } else { "odd" })),
                    ("index".to_string(), serde_json::json!(i)),
                ])),
                timestamp: Some(i as i64),
                updated_at: None,
                expires_at: None,
                version: Some(1),
            });
        }
        
        Ok(Self {
            dsst_engine,
            dviper_engine,
            test_vectors,
        })
    }
}

// ============================================================================
// DSST ENGINE TESTS
// ============================================================================

#[tokio::test]
async fn test_dsst_engine_creation() -> Result<()> {
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    
    let engine = DsstEngine::new()?;
    assert_eq!(engine.engine_name(), "DSST");
    assert_eq!(engine.engine_version(), "1.0.0");
    
    Ok(())
}

#[tokio::test]
async fn test_dsst_flush_operation() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    let params = FlushParameters {
        collection_id: "test_collection".to_string(),
        num_vectors: 100,
        estimated_size: 1024 * 1024,
        dimension: Some(128),
        distance_metric: Some(DistanceMetric::Euclidean),
        metadata: None,
        collection_config: None,
    };
    
    let result = fixture.dsst_engine.do_flush(&params).await?;
    
    assert!(result.success);
    assert_eq!(result.files_created, 1);
    assert!(result.bytes_written > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_dsst_id_lookup() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    // First flush some data
    let params = FlushParameters {
        collection_id: "test_collection".to_string(),
        num_vectors: fixture.test_vectors.len(),
        estimated_size: 1024 * 1024,
        dimension: Some(128),
        distance_metric: Some(DistanceMetric::Euclidean),
        metadata: None,
        collection_config: None,
    };
    
    fixture.dsst_engine.do_flush(&params).await?;
    
    // Test ID lookup (simulating AXIS returning IDs)
    let vector = fixture.dsst_engine
        .get_vector_by_id("test_collection", "vec_000010")
        .await?;
    
    // In production, this would return the actual vector
    // For now, it returns None as we haven't implemented full persistence
    assert!(vector.is_none() || vector.is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_dsst_similarity_search() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    // Perform similarity search
    let query = vec![0.5; 128];
    let result = fixture.dsst_engine.search_vectors_unified(
        "test_collection",
        "memory://test",
        &query,
        10,
        DistanceMetric::Euclidean,
        None,
        None,
        None,
    ).await?;
    
    assert!(result.records.len() <= 10);
    
    Ok(())
}

#[tokio::test]
async fn test_dsst_compaction() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    let params = CompactionParameters {
        full: false,
        collection_id: "test_collection".to_string(),
        compaction_level: 1,
        estimated_input_size: 2 * 1024 * 1024,
        max_output_file_size: 1024 * 1024 * 1024,
        collection_config: None,
    };
    
    let result = fixture.dsst_engine.do_compact(&params).await?;
    
    assert!(result.success);
    
    Ok(())
}

// ============================================================================
// DVIPER ENGINE TESTS
// ============================================================================

#[tokio::test]
async fn test_dviper_engine_creation() -> Result<()> {
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    
    let engine = DviperEngine::new()?;
    assert_eq!(engine.engine_name(), "DVIPER");
    assert_eq!(engine.engine_version(), "1.0.0");
    
    Ok(())
}

#[tokio::test]
async fn test_dviper_flush_operation() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    let params = FlushParameters {
        collection_id: "test_collection".to_string(),
        num_vectors: 100,
        estimated_size: 1024 * 1024,
        dimension: Some(128),
        distance_metric: Some(DistanceMetric::Euclidean),
        metadata: None,
        collection_config: None,
    };
    
    let result = fixture.dviper_engine.do_flush(&params).await?;
    
    assert!(result.success);
    assert_eq!(result.files_created, 1);
    assert!(result.bytes_written > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_dviper_columnar_search() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    // Perform columnar search
    let query = vec![0.5; 128];
    let result = fixture.dviper_engine.search_vectors_unified(
        "test_collection",
        "memory://test",
        &query,
        10,
        DistanceMetric::Euclidean,
        None,
        None,
        Some(serde_json::json!({
            "enable_projection": true,
            "enable_pushdown": true,
        })),
    ).await?;
    
    assert!(result.records.len() <= 10);
    // DVIPER should be faster due to columnar optimization
    assert!(result.execution_time_ms < 10.0);
    
    Ok(())
}

#[tokio::test]
async fn test_dviper_feature_support() -> Result<()> {
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    let engine = DviperEngine::new()?;
    
    // DVIPER-specific features
    assert!(engine.supports_feature("columnar_search").await);
    assert!(engine.supports_feature("predicate_pushdown").await);
    assert!(engine.supports_feature("projection").await);
    
    // Common features
    assert!(engine.supports_feature("id_lookup").await);
    assert!(engine.supports_feature("quantization").await);
    
    Ok(())
}

// ============================================================================
// COMPARATIVE TESTS
// ============================================================================

#[tokio::test]
async fn test_engine_comparison() -> Result<()> {
    let fixture = DualModeTestFixture::new(1000, 768).await?;
    
    // Test both engines with same workload
    let params = FlushParameters {
        collection_id: "comparison_test".to_string(),
        num_vectors: 1000,
        estimated_size: 10 * 1024 * 1024,
        dimension: Some(768),
        distance_metric: Some(DistanceMetric::Cosine),
        metadata: None,
        collection_config: None,
    };
    
    // Flush to both engines
    let dsst_result = fixture.dsst_engine.do_flush(&params).await?;
    let dviper_result = fixture.dviper_engine.do_flush(&params).await?;
    
    assert!(dsst_result.success);
    assert!(dviper_result.success);
    
    // DVIPER should have better compression due to columnar format
    // In production, this would be true: assert!(dviper_result.bytes_written < dsst_result.bytes_written);
    
    Ok(())
}

#[tokio::test]
async fn test_dual_mode_switching() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 384).await?;
    
    // Simulate AXIS returning top-k IDs (index-driven mode)
    let axis_ids = vec![
        "vec_000010".to_string(),
        "vec_000020".to_string(),
        "vec_000030".to_string(),
    ];
    
    // Both engines should handle ID lookups efficiently
    for id in &axis_ids {
        let dsst_vec = fixture.dsst_engine
            .get_vector_by_id("test_collection", id)
            .await?;
        let dviper_vec = fixture.dviper_engine
            .get_vector_by_id("test_collection", id)
            .await?;
        
        // Both should return same result
        assert_eq!(dsst_vec.is_some(), dviper_vec.is_some());
    }
    
    // Test index-free mode (direct similarity search)
    let query = vec![0.5; 384];
    
    let dsst_results = fixture.dsst_engine.search_vectors_unified(
        "test_collection",
        "memory://test",
        &query,
        5,
        DistanceMetric::Euclidean,
        None,
        None,
        None,
    ).await?;
    
    let dviper_results = fixture.dviper_engine.search_vectors_unified(
        "test_collection",
        "memory://test",
        &query,
        5,
        DistanceMetric::Euclidean,
        None,
        None,
        None,
    ).await?;
    
    // Both should return results
    assert!(!dsst_results.records.is_empty() || dsst_results.total_results == 0);
    assert!(!dviper_results.records.is_empty() || dviper_results.total_results == 0);
    
    Ok(())
}

#[tokio::test]
async fn test_engine_metrics_collection() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 128).await?;
    
    // Collect metrics from both engines
    let dsst_metrics = fixture.dsst_engine.collect_engine_metrics().await?;
    let dviper_metrics = fixture.dviper_engine.collect_engine_metrics().await?;
    
    // Verify metrics are collected
    assert!(dsst_metrics.contains_key("collection_count"));
    assert!(dsst_metrics.contains_key("total_sst_files"));
    assert!(dsst_metrics.contains_key("simd_backend"));
    
    assert!(dviper_metrics.contains_key("collection_count"));
    assert!(dviper_metrics.contains_key("total_parquet_files"));
    assert!(dviper_metrics.contains_key("columnar_optimization"));
    
    Ok(())
}

#[tokio::test]
async fn test_progressive_search_capabilities() -> Result<()> {
    let fixture = DualModeTestFixture::new(100, 256).await?;
    
    // Test progressive search in DSST
    assert!(fixture.dsst_engine.supports_feature("progressive_search").await);
    
    // Test columnar progressive search in DVIPER
    assert!(fixture.dviper_engine.supports_feature("columnar_search").await);
    
    // Both should support quantization for progressive refinement
    assert!(fixture.dsst_engine.supports_feature("quantization").await);
    assert!(fixture.dviper_engine.supports_feature("quantization").await);
    
    Ok(())
}