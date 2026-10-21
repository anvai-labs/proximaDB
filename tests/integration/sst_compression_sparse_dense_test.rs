//! Tests for SST compression effectiveness on different data patterns
//!
//! This test demonstrates that compression effectiveness varies based on data patterns:
//! - Sparse data (mostly zeros) compresses very well
//! - Dense random data doesn't compress well

// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;

use common::integration_test_helpers::{UnifiedTestEnvironment, operations};
use proximadb::compute::distance_computation::UnifiedDistanceCompute;
use proximadb::proto::proximadb_v1::StorageEngine;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb::storage::traits::UnifiedStorageEngine;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Test compression on sparse data (should compress very well)
#[tokio::test]
async fn test_compression_sparse_data() -> anyhow::Result<()> {
    let base_env = UnifiedTestEnvironment::new().await?;

    // Create sparse vectors (mostly zeros with few non-zero values)
    let vectors: Vec<proximadb_records::ProximaRecord> = (0..500)
        .map(|i| {
            let mut values = vec![0.0f32; 1024];
            for j in 0..10 {
                let idx = (i * 7 + j * 13) % 1024;
                values[idx] = (i as f32) * 0.1;
            }
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "type".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String("sparse".to_string()),
                ),
            );
            proximadb_records::ProximaRecord {
                oid: format!("sparse_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default", "vector", 1024, values,
                )],
                props,
                record_version: 1,
                created_at_ns: (1000 + i) as i64 * 1_000_000_000,
                updated_at_ns: (1000 + i) as i64 * 1_000_000_000,
                ..Default::default()
            }
        })
        .collect();

    info!("Testing compression on SPARSE data (99% zeros)...");

    let distance_compute = Arc::new(UnifiedDistanceCompute::new(
        proximadb::compute::distance_computation::DistanceMetric::Cosine,
    ));

    // Test with compression enabled
    let mut config_compressed = base_env.sst_config.clone();
    config_compressed.compression = "zstd".to_string();
    config_compressed.compression_level = 3;

    let compressed_engine = SstEngine::new().await?;

    // Flush compressed sparse data with compression config
    let mut flush_params =
        operations::build_flush_params(&base_env, vectors.clone(), StorageEngine::Sst).await?;

    // Set compression in the collection storage config
    if let Some(ref mut collection_config) = flush_params.collection_config {
        if let Some(ref mut config) = collection_config.config {
            use proximadb::proto::proximadb_v1::{
                CompressionAlgorithm, StorageConfig as ProtoStorageConfig,
            };

            config.storage_config = Some(ProtoStorageConfig {
                compression: Some(CompressionAlgorithm::CompressionZstd as i32),
                ..Default::default()
            });
        }
    }

    let compressed_result = compressed_engine.do_flush(&flush_params).await?;
    assert!(compressed_result.success);

    let compressed_size = get_directory_size(base_env.get_sst_data_directory().as_path()).await;

    // Clean and test uncompressed using filesystem API
    let fs = base_env.filesystem.get_filesystem("file:///")?;
    let storage_url = format!(
        "file://{}",
        base_env.get_sst_data_directory().to_str().unwrap()
    );
    let _ = fs.delete(&storage_url).await; // Ignore errors if directory doesn't exist
    tokio::fs::create_dir_all(base_env.get_sst_data_directory()).await?;

    let mut config_uncompressed = base_env.sst_config.clone();
    config_uncompressed.compression = "none".to_string();

    let uncompressed_engine = SstEngine::new().await?;

    // Create fresh flush params for uncompressed (don't reuse the modified compressed ones)
    let flush_params_uncompressed =
        operations::build_flush_params(&base_env, vectors, StorageEngine::Sst).await?;
    let uncompressed_result = uncompressed_engine
        .do_flush(&flush_params_uncompressed)
        .await?;
    assert!(uncompressed_result.success);

    let uncompressed_size = get_directory_size(base_env.get_sst_data_directory().as_path()).await;

    info!("SPARSE DATA compression results:");
    info!("  Compressed: {} bytes", compressed_size);
    info!("  Uncompressed: {} bytes", uncompressed_size);

    let ratio = if uncompressed_size > 0 {
        compressed_size as f64 / uncompressed_size as f64
    } else {
        1.0
    };

    info!(
        "  Compression ratio: {:.3} ({}% reduction)",
        ratio,
        ((1.0 - ratio) * 100.0) as i32
    );

    // Verify compression actually happened
    assert!(
        compressed_size < uncompressed_size,
        "Compressed size ({}) should be less than uncompressed size ({})",
        compressed_size,
        uncompressed_size
    );

    // Sparse data should compress well (at least 30% reduction expected)
    assert!(
        compressed_size < uncompressed_size * 70 / 100,
        "Sparse data should compress by at least 30%. Got {} vs {} (ratio: {:.3})",
        compressed_size,
        uncompressed_size,
        ratio
    );

    Ok(())
}

/// Test compression on dense random data (shouldn't compress well)
#[tokio::test]
async fn test_compression_dense_data() -> anyhow::Result<()> {
    let base_env = UnifiedTestEnvironment::new().await?;

    // Create dense random vectors (hard to compress)
    let vectors: Vec<proximadb_records::ProximaRecord> = (0..500)
        .map(|i| {
            let values: Vec<f32> = (0..1024)
                .map(|j| {
                    ((i as f32 * 0.1 + j as f32 * 0.01).sin()
                        * (i as f32 * 0.05 + j as f32 * 0.02).cos())
                })
                .collect();
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "type".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String("dense".to_string()),
                ),
            );
            proximadb_records::ProximaRecord {
                oid: format!("dense_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default", "vector", 1024, values,
                )],
                props,
                record_version: 1,
                created_at_ns: (1000 + i) as i64 * 1_000_000_000,
                updated_at_ns: (1000 + i) as i64 * 1_000_000_000,
                ..Default::default()
            }
        })
        .collect();

    info!("Testing compression on DENSE data (random floats)...");

    let distance_compute = Arc::new(UnifiedDistanceCompute::new(
        proximadb::compute::distance_computation::DistanceMetric::Cosine,
    ));

    // Test with compression enabled
    let mut config_compressed = base_env.sst_config.clone();
    config_compressed.compression = "zstd".to_string();
    config_compressed.compression_level = 3;

    let compressed_engine = SstEngine::new().await?;

    // Flush compressed dense data with compression config
    let mut flush_params =
        operations::build_flush_params(&base_env, vectors.clone(), StorageEngine::Sst).await?;

    // Set compression in the collection storage config
    if let Some(ref mut collection_config) = flush_params.collection_config {
        if let Some(ref mut config) = collection_config.config {
            use proximadb::proto::proximadb_v1::{
                CompressionAlgorithm, StorageConfig as ProtoStorageConfig,
            };

            config.storage_config = Some(ProtoStorageConfig {
                compression: Some(CompressionAlgorithm::CompressionZstd as i32),
                ..Default::default()
            });
        }
    }

    let compressed_result = compressed_engine.do_flush(&flush_params).await?;
    assert!(compressed_result.success);

    let compressed_size = get_directory_size(base_env.get_sst_data_directory().as_path()).await;

    // Clean and test uncompressed using filesystem API
    let fs = base_env.filesystem.get_filesystem("file:///")?;
    let storage_url = format!(
        "file://{}",
        base_env.get_sst_data_directory().to_str().unwrap()
    );
    let _ = fs.delete(&storage_url).await; // Ignore errors if directory doesn't exist
    tokio::fs::create_dir_all(base_env.get_sst_data_directory()).await?;

    let mut config_uncompressed = base_env.sst_config.clone();
    config_uncompressed.compression = "none".to_string();

    let uncompressed_engine = SstEngine::new().await?;

    // Create fresh flush params for uncompressed (don't reuse the modified compressed ones)
    let flush_params_uncompressed =
        operations::build_flush_params(&base_env, vectors, StorageEngine::Sst).await?;
    let uncompressed_result = uncompressed_engine
        .do_flush(&flush_params_uncompressed)
        .await?;
    assert!(uncompressed_result.success);

    let uncompressed_size = get_directory_size(base_env.get_sst_data_directory().as_path()).await;

    info!("DENSE DATA compression results:");
    info!("  Compressed: {} bytes", compressed_size);
    info!("  Uncompressed: {} bytes", uncompressed_size);
    if uncompressed_size > 0 {
        let ratio = compressed_size as f64 / uncompressed_size as f64;
        info!(
            "  Compression ratio: {:.3} ({}% reduction)",
            ratio,
            ((1.0 - ratio) * 100.0) as i32
        );
    }

    // Dense random data doesn't compress well - just ensure it doesn't get worse
    // Allow up to 110% (compression overhead) for small datasets
    assert!(
        compressed_size <= uncompressed_size * 110 / 100,
        "Compression shouldn't increase dense data size by >10%. Got {} vs {}",
        compressed_size,
        uncompressed_size
    );

    debug!("Dense data compression test passed - minimal compression as expected");

    Ok(())
}

/// Helper to get total size of SST files in a directory
async fn get_directory_size(path: &std::path::Path) -> u64 {
    use std::fs;

    fn dir_size(path: &std::path::Path) -> u64 {
        let mut size = 0;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    size += dir_size(&path);
                } else if path.extension().and_then(|s| s.to_str()) == Some("sst") {
                    size += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        size
    }

    dir_size(path)
}
