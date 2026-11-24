//! Universal Distance Adapter Demo
//!
//! This example demonstrates how to use the Universal Distance Adapter
//! with all storage engines (NOVA, SWIFT, SST, RAPTOR, HELIX) and shows
//! the PQ and INT8 optimized distance computations with progressive refinement.

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::hardware_capabilities;
use proximadb::storage::engines::universal::{
    CandidateVector, DistanceComputationRequest, EngineType, ProgressiveRefinementConfig,
    StorageFormat, UniversalAdapterConfig, UniversalDistanceAdapter,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for demo
    tracing_subscriber::fmt::init();

    println!("🚀 Universal Distance Adapter Demo");
    println!("===================================");

    // Initialize hardware capabilities
    let _ = hardware_capabilities::initialize_hardware_capabilities_default();

    // Create universal adapter with custom configuration
    let config = create_demo_config();
    let adapter = UniversalDistanceAdapter::with_config(config).await?;

    println!("✅ Universal Distance Adapter initialized successfully");

    // Demo 1: Basic distance computation with different engines
    println!("\n🎯 Demo 1: Basic Distance Computation Across Engines");
    demo_basic_distance_computation(&adapter).await?;

    // Demo 2: Progressive refinement pipeline
    println!("\n🔄 Demo 2: Progressive Refinement Pipeline");
    demo_progressive_refinement(&adapter).await?;

    // Demo 3: Quantized distance computations
    println!("\n⚡ Demo 3: Quantized Distance Computations");
    demo_quantized_computations(&adapter).await?;

    // Demo 4: Storage format optimization
    println!("\n📊 Demo 4: Storage Format Optimization");
    demo_storage_format_optimization(&adapter).await?;

    // Demo 5: Performance comparison
    println!("\n🏆 Demo 5: Performance Comparison");
    demo_performance_comparison(&adapter).await?;

    // Demo 6: Hardware acceleration
    println!("\n💨 Demo 6: Hardware Acceleration");
    demo_hardware_acceleration(&adapter).await?;

    println!("\n🎉 Universal Distance Adapter Demo completed successfully!");

    Ok(())
}

/// Create demo configuration for the universal adapter
fn create_demo_config() -> UniversalAdapterConfig {
    let mut config = UniversalAdapterConfig::default();

    // Enable all features for demo
    config.enable_progressive_refinement = true;
    config.enable_hardware_acceleration = true;
    config.enable_distance_caching = true;
    config.max_cache_size_mb = 128;
    config.simd_threshold = 32;

    println!("📋 Configuration:");
    println!(
        "   - Progressive Refinement: {}",
        config.enable_progressive_refinement
    );
    println!(
        "   - Hardware Acceleration: {}",
        config.enable_hardware_acceleration
    );
    println!("   - Distance Caching: {}", config.enable_distance_caching);
    println!("   - Cache Size: {} MB", config.max_cache_size_mb);

    config
}

/// Demo 1: Basic distance computation with different engines
async fn demo_basic_distance_computation(adapter: &UniversalDistanceAdapter) -> Result<()> {
    let query_vector = create_random_vector(128);
    let candidates = create_demo_candidates(100, 128);

    let engines = vec![EngineType::NOVA, EngineType::SWIFT, EngineType::SST];

    for engine_type in engines {
        let start_time = Instant::now();

        let request = DistanceComputationRequest {
            query_vector: query_vector.clone(),
            candidates: candidates.clone(),
            distance_metric: DistanceMetric::Euclidean,
            storage_format: StorageFormat::FP32,
            refinement_config: None,
            max_results: 10,
            enable_acceleration: true,
            quality_threshold: Some(0.8),
            collection_id: proximadb::utils::Uuid::new_v4(),
            engine_type,
        };

        let result = adapter.compute_progressive_distance(request).await?;
        let elapsed = start_time.elapsed();

        println!(
            "   Engine {:?}: {} results in {:?}",
            engine_type,
            result.results.len(),
            elapsed
        );

        if !result.results.is_empty() {
            println!(
                "     Best result: {:.4} (quality: {:.2})",
                result.results[0].rank_value, result.quality_metrics.average_confidence
            );
        }
    }

    Ok(())
}

/// Demo 2: Progressive refinement pipeline
async fn demo_progressive_refinement(adapter: &UniversalDistanceAdapter) -> Result<()> {
    let query_vector = create_random_vector(256);
    let candidates = create_demo_candidates(1000, 256);

    // Configure progressive refinement
    let mut refinement_config = ProgressiveRefinementConfig::default();
    refinement_config.enable_stage_skipping = true;
    refinement_config.min_improvement_threshold = 0.1;

    let request = DistanceComputationRequest {
        query_vector,
        candidates,
        distance_metric: DistanceMetric::Cosine,
        storage_format: StorageFormat::QuantizedPQ {
            segments: 8,
            bits: 8,
        },
        refinement_config: Some(refinement_config),
        max_results: 20,
        enable_acceleration: true,
        quality_threshold: Some(0.85),
        collection_id: proximadb::utils::Uuid::new_v4(),
        engine_type: EngineType::NOVA,
    };

    let result = adapter.compute_progressive_distance(request).await?;

    println!("   Refinement stages used: {:?}", result.refinement_stages);
    println!("   Final stage: {:?}", result.final_stage);
    println!(
        "   Quality improvement: {:.2}",
        result.quality_metrics.quality_improvement
    );
    println!(
        "   Total distance calculations: {}",
        result.performance_metrics.distance_calculations
    );

    // Show stage-wise performance
    for (stage, time) in &result.performance_metrics.stage_times_us {
        println!("     Stage {:?}: {}μs", stage, time);
    }

    Ok(())
}

/// Demo 3: Quantized distance computations
async fn demo_quantized_computations(adapter: &UniversalDistanceAdapter) -> Result<()> {
    let query_vector = create_random_vector(128);
    let candidates = create_demo_candidates(500, 128);

    let quantization_formats = vec![
        (
            "INT8",
            StorageFormat::QuantizedINT8 {
                scale: 1.0,
                zero_point: 0,
            },
        ),
        (
            "PQ-8x8",
            StorageFormat::QuantizedPQ {
                segments: 8,
                bits: 8,
            },
        ),
        (
            "PQ-16x4",
            StorageFormat::QuantizedPQ {
                segments: 16,
                bits: 4,
            },
        ),
        ("Binary", StorageFormat::Binary),
    ];

    for (name, format) in quantization_formats {
        let start_time = Instant::now();

        let request = DistanceComputationRequest {
            query_vector: query_vector.clone(),
            candidates: candidates.clone(),
            distance_metric: DistanceMetric::Euclidean,
            storage_format: format.clone(),
            refinement_config: None,
            max_results: 10,
            enable_acceleration: true,
            quality_threshold: None,
            collection_id: proximadb::utils::Uuid::new_v4(),
            engine_type: EngineType::NOVA,
        };

        match adapter.compute_progressive_distance(request).await {
            Ok(result) => {
                let elapsed = start_time.elapsed();
                println!(
                    "   {} format: {} results in {:?}",
                    name,
                    result.results.len(),
                    elapsed
                );

                let data_size = format.data_size_per_vector(128);
                println!("     Data size per vector: {} bytes", data_size);
                println!("     Cache hits: {}", result.cache_hits);
            }
            Err(e) => {
                println!("   {} format: Failed - {}", name, e);
            }
        }
    }

    Ok(())
}

/// Demo 4: Storage format optimization
async fn demo_storage_format_optimization(adapter: &UniversalDistanceAdapter) -> Result<()> {
    let test_scenarios = vec![
        ("Small dataset, high recall", 128, 1_000, 0.95),
        ("Medium dataset, medium recall", 256, 100_000, 0.85),
        ("Large dataset, low recall", 512, 10_000_000, 0.75),
        ("High-dim, analytics", 1024, 1_000_000, 0.80),
    ];

    for (scenario, dimension, dataset_size, target_recall) in test_scenarios {
        println!("   Scenario: {}", scenario);
        println!(
            "     Dimension: {}, Dataset size: {}, Target recall: {:.2}",
            dimension, dataset_size, target_recall
        );

        for engine_type in &[EngineType::NOVA] {
            let optimal_format = adapter
                .get_optimal_format(engine_type, dimension, dataset_size, target_recall)
                .await?;

            println!(
                "     {:?} optimal format: {:?}",
                engine_type, optimal_format
            );
        }
        println!();
    }

    Ok(())
}

/// Demo 5: Performance comparison
async fn demo_performance_comparison(adapter: &UniversalDistanceAdapter) -> Result<()> {
    let query_vector = create_random_vector(128);
    let small_candidates = create_demo_candidates(100, 128);
    let large_candidates = create_demo_candidates(10000, 128);

    let test_cases = vec![
        ("Small dataset (100 vectors)", small_candidates),
        ("Large dataset (10K vectors)", large_candidates),
    ];

    for (case_name, candidates) in test_cases {
        println!("   {}", case_name);

        // Test with and without acceleration
        for enable_acceleration in &[false, true] {
            let start_time = Instant::now();

            let request = DistanceComputationRequest {
                query_vector: query_vector.clone(),
                candidates: candidates.clone(),
                distance_metric: DistanceMetric::Euclidean,
                storage_format: StorageFormat::FP32,
                refinement_config: None,
                max_results: 10,
                enable_acceleration: *enable_acceleration,
                quality_threshold: None,
                collection_id: proximadb::utils::Uuid::new_v4(),
                engine_type: EngineType::SWIFT,
            };

            let result = adapter.compute_progressive_distance(request).await?;
            let elapsed = start_time.elapsed();

            let acceleration_status = if *enable_acceleration {
                "enabled"
            } else {
                "disabled"
            };
            println!(
                "     Acceleration {}: {:?} ({} results)",
                acceleration_status,
                elapsed,
                result.results.len()
            );
        }
        println!();
    }

    Ok(())
}

/// Demo 6: Hardware acceleration
async fn demo_hardware_acceleration(adapter: &UniversalDistanceAdapter) -> Result<()> {
    let stats = adapter.get_statistics().await?;

    println!("   Hardware acceleration statistics:");
    println!("     Total computations: {}", stats.total_computations);
    println!(
        "     Hardware acceleration usage: {:.1}%",
        stats.hardware_acceleration_usage * 100.0
    );
    println!("     Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
    println!(
        "     Average computation time: {}μs",
        stats.average_computation_time_us
    );
    println!("     Supported engines: {:?}", stats.supported_engines);

    // Show supported formats for each engine
    for engine_type in &stats.supported_engines {
        let formats = adapter.get_supported_formats(engine_type).await?;
        println!(
            "     {:?} supported formats: {} types",
            engine_type,
            formats.len()
        );
    }

    Ok(())
}

/// Create a random vector for testing
fn create_random_vector(dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|i| (i as f32 * 0.1 * (i % 7) as f32).sin())
        .collect()
}

/// Create demo candidate vectors
fn create_demo_candidates(count: usize, dimension: usize) -> Vec<CandidateVector> {
    let mut candidates = Vec::with_capacity(count);

    for i in 0..count {
        let vector: Vec<f32> = (0..dimension)
            .map(|j| (i + j) as f32 * 0.01 + (j as f32 * 0.1).cos())
            .collect();

        // Convert to bytes (FP32 format)
        let data: Vec<u8> = vector
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect();

        candidates.push(CandidateVector {
            id: proximadb::utils::Uuid::new_v4(),
            data,
            original_vector: Some(vector),
            metadata: Some({
                let mut meta = HashMap::new();
                meta.insert("index".to_string(), i.to_string());
                meta.insert("category".to_string(), format!("cat_{}", i % 5));
                meta
            }),
            quality_score: Some(0.7 + (i as f32 * 0.0003) % 0.3),
        });
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_demo_functionality() {
        let _ = hardware_capabilities::initialize_hardware_capabilities_default();

        let config = create_demo_config();
        let adapter = UniversalDistanceAdapter::with_config(config).await.unwrap();

        // Test basic functionality
        demo_basic_distance_computation(&adapter).await.unwrap();
        demo_quantized_computations(&adapter).await.unwrap();
    }

    #[test]
    fn test_demo_data_creation() {
        let vector = create_random_vector(64);
        assert_eq!(vector.len(), 64);

        let candidates = create_demo_candidates(10, 32);
        assert_eq!(candidates.len(), 10);
        assert_eq!(candidates[0].data.len(), 32 * 4); // 32 dimensions * 4 bytes per float
    }
}
