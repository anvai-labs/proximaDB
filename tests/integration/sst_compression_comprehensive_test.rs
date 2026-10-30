//! Comprehensive SST compression test for different data patterns and algorithms
//!
//! Tests compression effectiveness on:
//! - Dense data (random floats - poor compression)
//! - Sparse data (mostly zeros - excellent compression)
//! - Multiple compression algorithms and levels

// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;

use common::integration_test_helpers::{UnifiedTestEnvironment, operations};
use proximadb::compute::distance_computation::UnifiedDistanceCompute;
use proximadb::proto::proximadb_v1::StorageEngine;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::traits::UnifiedStorageEngine;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Create dense random vectors (hard to compress)
fn create_dense_vectors(
    _env: &UnifiedTestEnvironment,
    count: usize,
    dimension: usize,
) -> Vec<proximadb_records::ProximaRecord> {
    (0..count)
        .map(|i| {
            let values: Vec<f32> = (0..dimension)
                .map(|j| {
                    ((i as f32 * 0.1 + j as f32 * 0.01).sin()
                        * (i as f32 * 0.05 + j as f32 * 0.02).cos()
                        * (i as f32 * 0.03 + j as f32 * 0.04).tan())
                    .abs()
                })
                .collect();
            proximadb_records::ProximaRecord {
                oid: format!("dense_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    dimension as u32,
                    values,
                )],
                record_version: 1,
                created_at_ns: (1000 + i) as i64 * 1_000_000_000,
                updated_at_ns: (1000 + i) as i64 * 1_000_000_000,
                ..Default::default()
            }
        })
        .collect()
}

/// Create sparse vectors (mostly zeros - easy to compress)
fn create_sparse_vectors(
    _env: &UnifiedTestEnvironment,
    count: usize,
    dimension: usize,
) -> Vec<proximadb_records::ProximaRecord> {
    (0..count)
        .map(|i| {
            let mut values = vec![0.0f32; dimension];
            let non_zero_count = dimension / 100;
            for j in 0..non_zero_count {
                let idx = (i * 7 + j * 13) % dimension;
                values[idx] = (i as f32 + j as f32) * 0.1;
            }
            proximadb_records::ProximaRecord {
                oid: format!("sparse_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    dimension as u32,
                    values,
                )],
                record_version: 1,
                created_at_ns: (1000 + i) as i64 * 1_000_000_000,
                updated_at_ns: (1000 + i) as i64 * 1_000_000_000,
                ..Default::default()
            }
        })
        .collect()
}

/// Test compression on a specific data type with a specific algorithm
async fn test_compression_for_data(
    data_type: &str,
    vectors: Vec<proximadb_records::ProximaRecord>,
    algorithm: &str,
    level: i32,
) -> anyhow::Result<(u64, u64, f64)> {
    // Create separate environments
    let env_uncompressed = UnifiedTestEnvironment::new().await?;
    let env_compressed = UnifiedTestEnvironment::new().await?;

    let distance_compute = Arc::new(UnifiedDistanceCompute::new(
        proximadb::compute::distance_computation::DistanceMetric::Cosine,
    ));

    // Test UNCOMPRESSED first with 256KB blocks to see vector splitting with quantization
    let mut config_uncompressed = env_uncompressed.sst_config.clone();
    config_uncompressed.compression = "none".to_string();
    config_uncompressed.compression_level = 0;
    config_uncompressed.block_size_kb = 256; // Use 256KB blocks for better quantization clustering

    let uncompressed_engine = SstEngine::new().await?;

    let vectors_uncompressed = vectors.clone();
    let flush_params_uncompressed =
        operations::build_flush_params(&env_uncompressed, vectors_uncompressed, StorageEngine::Sst)
            .await?;

    let uncompressed_result = uncompressed_engine
        .do_flush(&flush_params_uncompressed)
        .await?;
    assert!(
        uncompressed_result.success,
        "Uncompressed flush should succeed"
    );

    // Log block information for uncompressed
    info!("📦 Uncompressed flush with 256KB blocks:");
    info!("  • Files created: {:?}", uncompressed_result.files_created);
    info!(
        "  • Entries flushed: {:?}",
        uncompressed_result.entries_flushed
    );
    info!(
        "  • Vectors per block: ~{}",
        256 * 1024
            / (vectors[0]
                .embeddings
                .first()
                .map(|e| e.dim as usize)
                .unwrap_or(1)
                * 4)
    );

    let uncompressed_size =
        get_sst_files_size(env_uncompressed.get_sst_data_directory().to_str().unwrap()).await;

    // Test COMPRESSED with 256KB blocks for better quantization clustering
    let mut config_compressed = env_compressed.sst_config.clone();
    config_compressed.compression = algorithm.to_string();
    config_compressed.compression_level = level;
    config_compressed.block_size_kb = 256; // Use 256KB blocks to see vector grouping with quantization

    let compressed_engine = SstEngine::new().await?;

    // Save dimensions before vectors is moved
    let vector_count = vectors.len();
    let vector_dim = vectors
        .first()
        .and_then(|v| v.embeddings.first())
        .map(|e| e.dim as usize)
        .unwrap_or(0);

    // Build flush params with compression config in the collection
    let mut flush_params_compressed =
        operations::build_flush_params(&env_compressed, vectors, StorageEngine::Sst).await?;

    // Set compression in the collection storage config
    if let Some(ref mut collection_config) = flush_params_compressed.collection_config {
        if let Some(ref mut config) = collection_config.config {
            use proximadb::proto::proximadb_v1::{
                CompressionAlgorithm, StorageConfig as ProtoStorageConfig,
            };

            let compression_algo = match algorithm {
                "none" => CompressionAlgorithm::CompressionNone as i32,
                "zstd" => CompressionAlgorithm::CompressionZstd as i32,
                "lz4" => CompressionAlgorithm::CompressionLz4 as i32,
                "snappy" => CompressionAlgorithm::CompressionSnappy as i32,
                "gzip" => CompressionAlgorithm::CompressionGzip as i32,
                _ => CompressionAlgorithm::CompressionZstd as i32,
            };

            config.storage_config = Some(ProtoStorageConfig {
                compression: Some(compression_algo),
                ..Default::default()
            });
        }
    }

    let compressed_result = compressed_engine.do_flush(&flush_params_compressed).await?;
    assert!(compressed_result.success, "Compressed flush should succeed");

    // Log block information for compressed
    info!(
        "📦 Compressed flush with 256KB blocks ({} level {}):",
        algorithm, level
    );
    info!("  • Files created: {:?}", compressed_result.files_created);
    info!(
        "  • Entries flushed: {:?}",
        compressed_result.entries_flushed
    );
    info!(
        "  • Expected blocks: ~{}",
        vector_count * vector_dim * 4 / (256 * 1024)
    );

    let compressed_size =
        get_sst_files_size(env_compressed.get_sst_data_directory().to_str().unwrap()).await;

    let ratio = if uncompressed_size > 0 {
        compressed_size as f64 / uncompressed_size as f64
    } else {
        1.0
    };

    debug!(
        "{} data with {} level {}: {} -> {} bytes (ratio: {:.3})",
        data_type, algorithm, level, uncompressed_size, compressed_size, ratio
    );

    Ok((compressed_size, uncompressed_size, ratio))
}

/// Test dense data compression (should have poor compression)
#[tokio::test]
async fn test_compression_dense_data() -> anyhow::Result<()> {
    info!("Testing compression on DENSE random data...");

    let base_env = UnifiedTestEnvironment::new().await?;
    let dense_vectors = create_dense_vectors(&base_env, 500, 1024);

    // Test with ZSTD level 3
    let (compressed, uncompressed, ratio) =
        test_compression_for_data("DENSE", dense_vectors, "zstd", 3).await?;

    info!("DENSE DATA Results:");
    info!("  Uncompressed: {} bytes", uncompressed);
    info!("  Compressed (zstd-3): {} bytes", compressed);
    info!(
        "  Compression ratio: {:.3} ({}% reduction)",
        ratio,
        ((1.0 - ratio) * 100.0) as i32
    );

    // Check if compression is working
    if compressed >= uncompressed {
        println!(
            "⚠️  WARNING: Compressed size ({}) is >= uncompressed ({}). \
            Compression appears to not be working! This may be due to dense random data.",
            compressed, uncompressed
        );
    }

    // Dense random data doesn't compress well
    // Expect minimal compression (maybe 5-20% at best)
    assert!(
        compressed > uncompressed * 80 / 100,
        "Dense data shouldn't compress more than 20%. Got {} vs {}",
        compressed,
        uncompressed
    );

    Ok(())
}

/// Test sparse data compression (should have excellent compression)
#[tokio::test]
async fn test_compression_sparse_data() -> anyhow::Result<()> {
    info!("Testing compression on SPARSE data (99% zeros)...");

    let base_env = UnifiedTestEnvironment::new().await?;
    let sparse_vectors = create_sparse_vectors(&base_env, 500, 1024);

    // Test with ZSTD level 3
    let (compressed, uncompressed, ratio) =
        test_compression_for_data("SPARSE", sparse_vectors, "zstd", 3).await?;

    info!("SPARSE DATA Results:");
    info!("  Uncompressed: {} bytes", uncompressed);
    info!("  Compressed (zstd-3): {} bytes", compressed);
    info!(
        "  Compression ratio: {:.3} ({}% reduction)",
        ratio,
        ((1.0 - ratio) * 100.0) as i32
    );

    // Check if compression is working
    if compressed >= uncompressed {
        println!(
            "⚠️  WARNING: Compressed size ({}) is >= uncompressed ({}). \
            Compression is not working as expected for sparse data!",
            compressed, uncompressed
        );
    }

    // Sparse data should compress very well (at least 50% reduction)
    assert!(
        compressed < uncompressed * 50 / 100,
        "Sparse data should compress by at least 50%. Got {} vs {}",
        compressed,
        uncompressed
    );

    Ok(())
}

/// Test multiple compression algorithms and levels
#[tokio::test]
async fn test_compression_algorithms_and_levels() -> anyhow::Result<()> {
    info!("Testing multiple compression algorithms and levels...");

    let base_env = UnifiedTestEnvironment::new().await?;

    // Test both data types
    let test_cases = vec![
        ("DENSE", create_dense_vectors(&base_env, 200, 512)),
        ("SPARSE", create_sparse_vectors(&base_env, 200, 512)),
    ];

    // Test various algorithms and levels
    let algorithms = vec![
        ("none", 0),
        ("zstd", 1),
        ("zstd", 3),
        ("zstd", 6),
        ("lz4", 0),
        ("snappy", 0),
        ("gzip", 1),
        ("gzip", 6),
    ];

    let mut results: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    for (data_type, vectors) in &test_cases {
        let mut data_results = Vec::new();

        for (algo, level) in &algorithms {
            match test_compression_for_data(data_type, vectors.clone(), algo, *level).await {
                Ok((_, _, ratio)) => {
                    let label = if *level > 0 {
                        format!("{}-{}", algo, level)
                    } else {
                        algo.to_string()
                    };
                    data_results.push((label, ratio));
                }
                Err(e) => {
                    debug!("Failed to test {} with {}: {}", data_type, algo, e);
                }
            }
        }

        results.insert(data_type.to_string(), data_results);
    }

    // Print comparison table
    info!("\n📊 COMPRESSION EFFECTIVENESS BY DATA TYPE:");
    info!("┌─────────────┬──────────────┬──────────────┐");
    info!("│ Algorithm   │ Dense Ratio  │ Sparse Ratio │");
    info!("├─────────────┼──────────────┼──────────────┤");

    for (algo, _) in &algorithms {
        let algo_label =
            if let Some((_, level)) = algorithms.iter().find(|(a, l)| a == algo && *l > 0) {
                if *level > 0 {
                    format!("{}-{}", algo, level)
                } else {
                    algo.to_string()
                }
            } else {
                algo.to_string()
            };

        let dense_ratio = results
            .get("enable_two_stage_search")
            .and_then(|v| v.iter().find(|(a, _)| a.starts_with(algo)).map(|(_, r)| r))
            .unwrap_or(&1.0);

        let sparse_ratio = results
            .get("enable_two_stage_search")
            .and_then(|v| v.iter().find(|(a, _)| a.starts_with(algo)).map(|(_, r)| r))
            .unwrap_or(&1.0);

        info!(
            "│ {:11} │ {:>12.3} │ {:>12.3} │",
            algo_label, dense_ratio, sparse_ratio
        );
    }
    info!("└─────────────┴──────────────┴──────────────┘");
    info!("Note: Lower ratio is better (1.0 = no compression)");

    // Verify key expectations
    if let Some(sparse_results) = results.get("sparse") {
        if let Some((_, zstd_ratio)) = sparse_results.iter().find(|(a, _)| a == "zstd-3") {
            assert!(
                *zstd_ratio < 0.5,
                "ZSTD should achieve >50% compression on sparse data. Got ratio: {}",
                zstd_ratio
            );
        }
    }

    if let Some(dense_results) = results.get("sparse") {
        if let Some((_, none_ratio)) = dense_results.iter().find(|(a, _)| a == "none") {
            assert!(
                *none_ratio >= 0.99 && *none_ratio <= 1.01,
                "No compression should have ratio ~1.0. Got: {}",
                none_ratio
            );
        }
    }

    Ok(())
}

/// Helper to get total size of SST files in a directory
async fn get_sst_files_size(path: &str) -> u64 {
    use std::fs;
    use std::path::Path;

    fn sst_size(path: &Path) -> u64 {
        let mut size = 0;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    size += sst_size(&path);
                } else if path.extension().and_then(|s| s.to_str()) == Some("sst") {
                    size += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        size
    }

    sst_size(Path::new(path))
}
