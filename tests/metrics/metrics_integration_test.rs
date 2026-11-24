//! Engine metrics integration test
//! Demonstrates engine integration with the unified metrics framework (ORPHANED: references removed DSST/DVIPER engines; see TD-ORPHAN-1)

use anyhow::Result;
use proximadb::{
    core::hardware_capabilities,
    proto::proximadb::{VectorRecord, MetadataItem, metadata_item},
    storage::{
        engines::{StorageEngineFactory},
        traits::{UnifiedStorageEngine, FlushParameters},
    },
    compute::distance_computation::DistanceMetric,
    metrics::{EngineMetricsCollector, EngineComparison, MetricsConfig, UnifiedMetricsCollector},
    proto::proximadb::StorageEngine as ProtoStorageEngine,
};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::time::sleep;

#[tokio::test]
async fn test_unified_metrics_integration() -> Result<()> {
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    
    // Create unified metrics framework
    let metrics_config = MetricsConfig::default();
    let mut unified_collector = UnifiedMetricsCollector::new();
    
    // Create engine metrics collector and register it
    let engine_collector = Arc::new(EngineMetricsCollector::new());
    unified_collector.register(engine_collector.clone());
    
    // Create engines with metrics integration
    let dsst_engine = StorageEngineFactory::create_with_metrics(
        ProtoStorageEngine::Dsst,
        engine_collector.clone(),
    )?;
    
    let dviper_engine = StorageEngineFactory::create_with_metrics(
        ProtoStorageEngine::Dviper,
        engine_collector.clone(),
    )?;
    
    println!("✅ Created engines with metrics integration");
    
    // Generate test data
    let test_vectors = generate_test_vectors(100, 128);
    let flush_params = FlushParameters {
        collection_id: "metrics_test".to_string(),
        num_vectors: test_vectors.len(),
        estimated_size: test_vectors.len() * 128 * 4,
        dimension: Some(128),
        distance_metric: Some(DistanceMetric::Cosine),
        metadata: None,
        collection_config: None,
    };
    
    // Perform operations on both engines
    println!("🔄 Performing operations to generate metrics...");
    
    // DSST operations
    dsst_engine.do_flush(&flush_params).await?;
    for i in 0..10 {
        let query = vec![0.5; 128];
        let _result = dsst_engine.search_vectors_unified(
            "metrics_test",
            "memory://test",
            &query,
            5,
            DistanceMetric::Cosine,
            None,
            None,
            None,
        ).await?;
        
        if i % 3 == 0 {
            let _vector = dsst_engine.get_vector_by_id("metrics_test", &format!("vec_{:04}", i)).await?;
        }
    }
    
    // DVIPER operations
    dviper_engine.do_flush(&flush_params).await?;
    for i in 0..10 {
        let query = vec![0.3; 128];
        let _result = dviper_engine.search_vectors_unified(
            "metrics_test",
            "memory://test",
            &query,
            5,
            DistanceMetric::Cosine,
            None,
            None,
            Some(serde_json::json!({
                "enable_projection": true,
                "enable_pushdown": true,
            })),
        ).await?;
        
        if i % 2 == 0 {
            let _vector = dviper_engine.get_vector_by_id("metrics_test", &format!("vec_{:04}", i)).await?;
        }
    }
    
    // Wait for metrics to be collected
    sleep(Duration::from_millis(100)).await;
    
    // Collect unified metrics
    println!("📊 Collecting unified metrics...");
    let all_samples = unified_collector.collect_all().await?;
    
    // Verify engine metrics are collected
    let engine_samples = all_samples.iter()
        .find(|s| s.collector == "engine")
        .expect("Engine metrics should be collected");
    
    println!("📈 Engine metrics collected: {} values", engine_samples.values.len());
    
    // Print some key metrics
    for (key, value) in &engine_samples.values {
        if key.contains("operations_total") || key.contains("error_rate") || key.contains("avg_latency") {
            println!("  {}: {:.2}", key, value);
        }
    }
    
    // Test engine comparison
    println!("🏆 Comparing engine performance...");
    let comparison = engine_collector.compare_engines().await;
    
    println!("Engine comparison results:");
    if let Some(winner) = &comparison.winner {
        println!("  🥇 Winner: {}", winner);
    }
    
    for (engine, stats) in &comparison.engine_stats {
        println!("  📊 {}: {} ops, {:.1}% errors, {:.2}ms avg latency",
            engine,
            stats.total_operations,
            stats.error_rate * 100.0,
            stats.max_avg_latency
        );
    }
    
    println!("\n💡 Recommendations:");
    for rec in &comparison.recommendations {
        println!("  - {}", rec);
    }
    
    // Test metrics summary (dashboard compatibility)
    let summary = unified_collector.get_metrics_summary().await;
    println!("\n🎯 System summary:");
    println!("  System health: {:.1}%", summary.system_health * 100.0);
    println!("  CPU usage: {:.1}%", summary.cpu_usage);
    println!("  Memory usage: {:.1}%", summary.memory_usage_percent);
    println!("  Cache hit rate: {:.1}%", summary.cache_hit_rate * 100.0);
    println!("  Queries/sec: {:.1}", summary.queries_per_second);
    println!("  P99 latency: {:.1}ms", summary.query_latency_p99);
    println!("  Active alerts: {}", summary.active_alerts_count);
    
    // Verify both engines have recorded operations
    let dsst_stats = engine_collector.get_engine_statistics("DSST").await;
    let dviper_stats = engine_collector.get_engine_statistics("DVIPER").await;
    
    assert!(dsst_stats.total_operations > 0, "DSST should have recorded operations");
    assert!(dviper_stats.total_operations > 0, "DVIPER should have recorded operations");
    assert!(dsst_stats.total_bytes_processed > 0, "DSST should have processed bytes");
    assert!(dviper_stats.total_bytes_processed > 0, "DVIPER should have processed bytes");
    
    println!("\n✅ Unified metrics integration test completed successfully!");
    println!("   DSST: {} operations, {} bytes", dsst_stats.total_operations, dsst_stats.total_bytes_processed);
    println!("   DVIPER: {} operations, {} bytes", dviper_stats.total_operations, dviper_stats.total_bytes_processed);
    
    Ok(())
}

#[tokio::test]
async fn test_metrics_collection_performance() -> Result<()> {
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    
    // Test that metrics collection doesn't significantly impact performance
    let engine_collector = Arc::new(EngineMetricsCollector::new());
    
    // Create engine with metrics
    let engine = StorageEngineFactory::create_with_metrics(
        ProtoStorageEngine::Dsst,
        engine_collector.clone(),
    )?;
    
    let test_vectors = generate_test_vectors(50, 64);
    let flush_params = FlushParameters {
        collection_id: "performance_test".to_string(),
        num_vectors: test_vectors.len(),
        estimated_size: test_vectors.len() * 64 * 4,
        dimension: Some(64),
        distance_metric: Some(DistanceMetric::Euclidean),
        metadata: None,
        collection_config: None,
    };
    
    // Perform operations and measure time
    let start = std::time::Instant::now();
    
    engine.do_flush(&flush_params).await?;
    
    for i in 0..20 {
        let query = vec![0.1; 64];
        let _result = engine.search_vectors_unified(
            "performance_test",
            "memory://test",
            &query,
            3,
            DistanceMetric::Euclidean,
            None,
            None,
            None,
        ).await?;
    }
    
    let elapsed = start.elapsed();
    println!("⏱️  Operations with metrics took: {:?}", elapsed);
    
    // Verify metrics were collected
    let stats = engine_collector.get_engine_statistics("DSST").await;
    assert!(stats.total_operations >= 20, "Should have recorded at least 20 operations");
    
    // Ensure overhead is reasonable (this is a basic sanity check)
    assert!(elapsed.as_millis() < 5000, "Operations with metrics should complete within 5 seconds");
    
    println!("✅ Metrics collection performance test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_error_tracking() -> Result<()> {
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();
    
    let engine_collector = Arc::new(EngineMetricsCollector::new());
    
    // Manually record some errors to test error tracking
    engine_collector.record_operation("DSST", "search", 50.0, false, 1024).await;
    engine_collector.record_operation("DSST", "search", 75.0, true, 0).await; // Error
    engine_collector.record_operation("DSST", "search", 45.0, false, 2048).await;
    engine_collector.record_operation("DSST", "get_by_id", 10.0, true, 0).await; // Error
    
    let stats = engine_collector.get_engine_statistics("DSST").await;
    
    assert_eq!(stats.total_operations, 4);
    assert_eq!(stats.total_errors, 2);
    assert_eq!(stats.error_rate, 0.5); // 50% error rate
    
    let comparison = engine_collector.compare_engines().await;
    
    // Should have recommendations about high error rate
    let has_error_recommendation = comparison.recommendations.iter()
        .any(|r| r.contains("error rate") || r.contains("stability"));
    
    assert!(has_error_recommendation, "Should recommend investigating high error rate");
    
    println!("✅ Error tracking test passed");
    
    Ok(())
}

// Helper function to generate test vectors
fn generate_test_vectors(count: usize, dimension: usize) -> Vec<VectorRecord> {
    (0..count)
        .map(|i| VectorRecord {
            id: format!("vec_{:04}", i),
            vector: (0..dimension)
                .map(|d| ((i + d) as f32).sin())
                .collect(),
            metadata: vec![
                MetadataItem {
                    key: "test".to_string(),
                    value: Some(metadata_item::Value::BoolValue(true)),
                },
                MetadataItem {
                    key: "index".to_string(),
                    value: Some(metadata_item::Value::NumberValue(i as f64)),
                },
            ],
            timestamp: i as u32,
            updated_at: None,
            expires_at: None,
            version: Some(1),
            quantized_vector: None,
            source: None,
        })
        .collect()
}