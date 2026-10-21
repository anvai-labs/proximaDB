// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;

use anyhow::Result;
use proximadb::compute::distance_computation::engine::{DistanceMetric, UnifiedDistanceCompute};
use proximadb::compute::quantization::{
    BinaryQuantization, InMemoryCodebookStore, ProductQuantization as PqConfig, QuantizationLevel,
    StorageQuantizationConfig, StorageQuantizationEngine, UnifiedQuantizationEngine,
    UnifiedQuantizationLevel,
};
use proximadb::core::SstConfig;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::local::LocalConfig;
use proximadb::storage::persistence::filesystem::{FileOptions, FilesystemPerformanceConfig};
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine};
use proximadb_records::ProximaRecord;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

#[tokio::test]
async fn test_quantization_statistics_comprehensive() -> Result<()> {
    println!("\n{}", "=".repeat(80));
    println!("🎯 COMPREHENSIVE QUANTIZATION STATISTICS TEST");
    println!("{}", "=".repeat(80));

    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // Test parameters
    let run_full_matrix = std::env::var("PROXIMADB_QUANT_STATS_FULL_MATRIX")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let default_vectors = if run_full_matrix { 1000 } else { 100 };
    let num_vectors = std::env::var("PROXIMADB_QUANT_STATS_NUM_VECTORS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_vectors);
    let dimension = 768; // BERT-like dimensions
    let sparsity_levels = if run_full_matrix {
        vec![0, 30, 50, 70, 90]
    } else {
        vec![0, 90]
    };

    // Generate test vectors with different sparsity levels
    let mut all_test_data = Vec::new();
    for sparsity in &sparsity_levels {
        let vectors = generate_sparse_vectors(num_vectors, dimension, *sparsity);
        all_test_data.push((*sparsity, vectors));
    }

    // Run tests for different quantization configurations
    let quantization_configs = if run_full_matrix {
        vec![
            ("No Quantization", None, None),
            (
                "Binary Quantization",
                Some(UnifiedQuantizationLevel {
                    level_type: Some(QuantizationLevel::Binary(BinaryQuantization {
                        threshold: None,
                        sign_based: false,
                    })),
                }),
                None,
            ),
            (
                "INT8 Quantization",
                None,
                Some(UnifiedQuantizationLevel::int8()),
            ),
            (
                "PQ8 Quantization",
                Some(UnifiedQuantizationLevel {
                    level_type: Some(QuantizationLevel::Pq(PqConfig {
                        num_subvectors: 8,
                        bits_per_code: 8,
                        codebook_id: None,
                        adaptive_subvectors: false,
                    })),
                }),
                None,
            ),
            (
                "PQ4 Quantization",
                Some(UnifiedQuantizationLevel {
                    level_type: Some(QuantizationLevel::Pq(PqConfig {
                        num_subvectors: 16,
                        bits_per_code: 4,
                        codebook_id: None,
                        adaptive_subvectors: false,
                    })),
                }),
                None,
            ),
        ]
    } else {
        vec![
            ("No Quantization", None, None),
            (
                "PQ8 Quantization",
                Some(UnifiedQuantizationLevel {
                    level_type: Some(QuantizationLevel::Pq(PqConfig {
                        num_subvectors: 8,
                        bits_per_code: 8,
                        codebook_id: None,
                        adaptive_subvectors: false,
                    })),
                }),
                None,
            ),
        ]
    };

    println!("\n📊 TEST CONFIGURATION:");
    println!("  • Vectors: {}", num_vectors);
    println!("  • Dimension: {}", dimension);
    println!("  • Sparsity levels: {:?}%", sparsity_levels);
    println!("  • Quantization types: {}", quantization_configs.len());
    println!(
        "  • Matrix mode: {}",
        if run_full_matrix { "full" } else { "quick" }
    );

    // Store results for summary
    let mut all_results = Vec::new();

    for (config_name, primary_level, fast_level) in quantization_configs {
        println!("\n{}", "-".repeat(80));
        println!("🔧 Testing: {}", config_name);
        println!("{}", "-".repeat(80));

        for (sparsity, vectors) in &all_test_data {
            let result = run_quantization_test(
                config_name,
                *sparsity,
                &vectors,
                dimension,
                primary_level.clone(),
                fast_level.clone(),
            )
            .await?;

            all_results.push(result);
        }
    }

    // Print comprehensive summary
    print_summary_statistics(&all_results);

    Ok(())
}

async fn run_quantization_test(
    config_name: &str,
    sparsity: usize,
    vectors: &[ProximaRecord],
    dimension: usize,
    primary_level: Option<UnifiedQuantizationLevel>,
    fast_level: Option<UnifiedQuantizationLevel>,
) -> Result<TestResult> {
    let start_time = Instant::now();

    // Create temporary directory
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();

    // Setup SST engine with test configuration
    let sst_config = SstConfig::test_config_256kb(); // Use 256KB blocks for clustering
    let fs_config = FilesystemConfig {
        default_fs: Some(format!("file://{}", base_path)),
        local: Some(LocalConfig::default()),
        global_options: FileOptions::default(),
        auth_config: None,
        performance_config: FilesystemPerformanceConfig::default(),
        scheme_mapping: HashMap::new(),
    };
    let filesystem = Arc::new(FilesystemFactory::create(fs_config).await?);

    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let mut sst_storage = SstEngine::new().await?;

    // Calculate baseline (uncompressed) size
    let baseline_size = vectors.len() * dimension * 4; // FP32 = 4 bytes per float

    // Setup quantization if enabled
    let (quantized_size, quantization_time) = if primary_level.is_some() || fast_level.is_some() {
        let quant_start = Instant::now();

        let config = StorageQuantizationConfig::default();

        let distance_compute = Arc::new(UnifiedDistanceCompute::default());
        let codebook_store = Arc::new(InMemoryCodebookStore::new());
        let unified_engine = Arc::new(UnifiedQuantizationEngine::new(
            distance_compute.clone(),
            codebook_store,
        ));
        let mut quantization_engine =
            StorageQuantizationEngine::new(unified_engine, distance_compute, config);

        // Train quantization model
        let vector_data: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                v.embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default()
            })
            .collect();
        quantization_engine.train(&vector_data).await?;

        // Quantize vectors
        let quantized_data = quantization_engine
            .quantize_batch(
                &vector_data,
                Some(&vectors.iter().map(|v| v.oid.clone()).collect::<Vec<_>>()),
            )
            .await?;

        // Calculate quantized size
        let mut total_quantized_size = 0usize;
        for qdata in &quantized_data {
            if let Some(ref primary) = qdata.primary {
                total_quantized_size += primary.data.len();
            }
            if let Some(ref fast) = qdata.fast {
                total_quantized_size += fast.data.len();
            }
            if let Some(ref filter) = qdata.filter {
                total_quantized_size += filter.data.len();
            }
        }

        (total_quantized_size, quant_start.elapsed())
    } else {
        (baseline_size, std::time::Duration::ZERO)
    };

    // Flush vectors to SST
    let flush_start = Instant::now();
    let test_env = common::integration_test_helpers::UnifiedTestEnvironment::new().await?;
    let collection = test_env.create_test_collection();
    let flush_params = FlushParameters {
        collection_id: Some("test_collection".to_string()),
        force: false,
        synchronous: true,
        hints: std::collections::HashMap::new(),
        timeout_ms: None,
        vector_records: vectors.to_vec(),
        trigger_compaction: false,
        batch_ids: vec![],
        collection_config: Some(collection.clone()),
        estimated_size: 0,
    };

    // Vectors are already in flush_params, no need for insert_batch
    let flush_result = sst_storage.do_flush(&flush_params).await?;
    let flush_time = flush_start.elapsed();

    // Perform search test
    let query = vectors[0]
        .embeddings
        .first()
        .map(|e| e.values.to_fp32_owned())
        .unwrap_or_default();
    let search_params = std::sync::Arc::new(proximadb::core::search::SearchParams {
        vector: Some(query),
        top_k: Some(10),
        distance_metric: Some(DistanceMetric::Cosine),
        ..Default::default()
    });

    let collection = std::sync::Arc::new(collection);
    let query_context = proximadb::storage::traits::StorageQueryContext {
        search_params,
        collection,
        metadata: proximadb::storage::traits::StorageQueryMetadata::default(),
        user_context: None,
        tenant_context: None,
    };

    let search_start = Instant::now();
    let search_results = sst_storage.search_vectors_unified(&query_context).await?;
    let search_time = search_start.elapsed();

    // Calculate compression ratio
    let compression_ratio = if quantized_size > 0 {
        baseline_size as f64 / quantized_size as f64
    } else {
        1.0
    };

    let space_savings = if quantized_size < baseline_size {
        ((baseline_size - quantized_size) as f64 / baseline_size as f64) * 100.0
    } else {
        0.0
    };

    // Print test results
    println!("\n  📈 Sparsity {}%:", sparsity);
    println!(
        "    • Baseline size: {:.2} MB",
        baseline_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    • Quantized size: {:.2} MB",
        quantized_size as f64 / (1024.0 * 1024.0)
    );
    println!("    • Compression ratio: {:.2}x", compression_ratio);
    println!("    • Space savings: {:.1}%", space_savings);
    println!(
        "    • Quantization time: {:.2}ms",
        quantization_time.as_millis()
    );
    println!("    • Flush time: {:.2}ms", flush_time.as_millis());
    println!(
        "    • Search time: {:.2}ms",
        search_time.as_micros() as f64 / 1000.0
    );
    println!("    • Search results: {} found", search_results.len());
    println!("    • Entries flushed: {:?}", flush_result.entries_flushed);

    Ok(TestResult {
        config_name: config_name.to_string(),
        sparsity,
        baseline_size,
        quantized_size,
        compression_ratio,
        space_savings,
        quantization_time,
        flush_time,
        search_time,
        search_accuracy: search_results.len() as f64 / 10.0, // Assuming we want 10 results
    })
}

fn generate_sparse_vectors(
    count: usize,
    dim: usize,
    sparsity_percent: usize,
) -> Vec<ProximaRecord> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut vectors = Vec::with_capacity(count);

    for i in 0..count {
        let mut values = vec![0.0f32; dim];
        let non_zero_count = dim * (100 - sparsity_percent) / 100;

        // Randomly set non-zero values
        for _j in 0..non_zero_count {
            let idx = rng.gen_range(0..dim);
            values[idx] = rng.gen_range(-1.0..1.0);
        }

        // Normalize vector
        let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut values {
                *val /= norm;
            }
        }

        vectors.push(ProximaRecord {
            oid: format!("vec_{}", i),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default", "vector", dim as u32, values,
            )],
            record_version: 1,
            created_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            updated_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            ..Default::default()
        });
    }

    vectors
}

fn print_summary_statistics(results: &[TestResult]) {
    println!("\n{}", "=".repeat(80));
    println!("📊 QUANTIZATION SUMMARY STATISTICS");
    println!("{}", "=".repeat(80));

    // Group results by configuration
    let mut by_config: std::collections::HashMap<String, Vec<&TestResult>> =
        std::collections::HashMap::new();
    for result in results {
        by_config
            .entry(result.config_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    // Print comparison table
    println!("\n📈 COMPRESSION RATIO COMPARISON (Higher is Better):");
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Configuration", "0% Sparse", "30% Sparse", "50% Sparse", "70% Sparse", "90% Sparse"
    );
    println!("{}", "-".repeat(85));

    for (config, results) in by_config.iter() {
        print!("{:<25}", config);
        for sparsity in &[0, 30, 50, 70, 90] {
            if let Some(result) = results.iter().find(|r| r.sparsity == *sparsity) {
                print!(" {:>9.2}x", result.compression_ratio);
            } else {
                print!(" {:>10}", "-");
            }
        }
        println!();
    }

    println!("\n💾 SPACE SAVINGS COMPARISON (% Reduction):");
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Configuration", "0% Sparse", "30% Sparse", "50% Sparse", "70% Sparse", "90% Sparse"
    );
    println!("{}", "-".repeat(85));

    for (config, results) in by_config.iter() {
        print!("{:<25}", config);
        for sparsity in &[0, 30, 50, 70, 90] {
            if let Some(result) = results.iter().find(|r| r.sparsity == *sparsity) {
                print!(" {:>9.1}%", result.space_savings);
            } else {
                print!(" {:>10}", "-");
            }
        }
        println!();
    }

    println!("\n⚡ SEARCH LATENCY COMPARISON (ms):");
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Configuration", "0% Sparse", "30% Sparse", "50% Sparse", "70% Sparse", "90% Sparse"
    );
    println!("{}", "-".repeat(85));

    for (config, results) in by_config.iter() {
        print!("{:<25}", config);
        for sparsity in &[0, 30, 50, 70, 90] {
            if let Some(result) = results.iter().find(|r| r.sparsity == *sparsity) {
                print!(" {:>9.2}ms", result.search_time.as_micros() as f64 / 1000.0);
            } else {
                print!(" {:>10}", "-");
            }
        }
        println!();
    }

    // Calculate and print averages
    println!("\n📊 AVERAGE METRICS ACROSS ALL SPARSITY LEVELS:");
    println!("{}", "-".repeat(85));

    for (config, results) in by_config.iter() {
        let avg_compression: f64 =
            results.iter().map(|r| r.compression_ratio).sum::<f64>() / results.len() as f64;
        let avg_savings: f64 =
            results.iter().map(|r| r.space_savings).sum::<f64>() / results.len() as f64;
        let avg_search: f64 = results
            .iter()
            .map(|r| r.search_time.as_micros() as f64)
            .sum::<f64>()
            / results.len() as f64
            / 1000.0;

        println!(
            "{:<25} Compression: {:.2}x | Savings: {:.1}% | Search: {:.2}ms",
            config, avg_compression, avg_savings, avg_search
        );
    }

    println!("\n✅ QUANTIZATION TEST COMPLETE");
    println!("{}", "=".repeat(80));
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TestResult {
    config_name: String,
    sparsity: usize,
    baseline_size: usize,
    quantized_size: usize,
    compression_ratio: f64,
    space_savings: f64,
    quantization_time: std::time::Duration,
    flush_time: std::time::Duration,
    search_time: std::time::Duration,
    search_accuracy: f64,
}
