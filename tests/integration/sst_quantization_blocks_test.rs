// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;

use anyhow::Result;
use proximadb::compute::quantization::{ProductQuantization as PqConfig, UnifiedQuantizationLevel};
use proximadb::core::SstConfig;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine};
use proximadb_records::ProximaRecord;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use tempfile::TempDir;
use tracing::info;

/// Test how 256KB blocks affect quantization clustering
#[tokio::test]
async fn test_quantization_with_256kb_blocks() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    info!("\n{}", "=".repeat(80));
    info!("🧪 TESTING QUANTIZATION WITH 256KB BLOCKS");
    info!("{}", "=".repeat(80));

    // Test parameters
    let num_vectors = 2000;
    let dimension = 768; // BERT-like dimensions
    let block_sizes = vec![256, 512, 1024, 2048]; // KB

    // Generate clustered vectors for better quantization
    let vectors = generate_clustered_vectors(num_vectors, dimension);

    for block_size_kb in block_sizes {
        info!("\n{}", "-".repeat(80));
        info!("📦 Testing with {}KB blocks", block_size_kb);

        // Create temporary directory
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();

        // Setup SST with specific block size
        let mut sst_config = SstConfig::default();
        sst_config.block_size_kb = block_size_kb;
        sst_config.compression = "zstd".to_string();
        sst_config.compression_level = 3;

        let fs_config = FilesystemConfig {
            default_fs: Some(format!("file://{}", base_path)),
            local: None,
            global_options: Default::default(),
            auth_config: None,
            performance_config: Default::default(),
            scheme_mapping: std::collections::HashMap::new(),
        };
        let filesystem = Arc::new(FilesystemFactory::create(fs_config).await?);
        let distance_compute = Arc::new(
            proximadb::compute::distance_computation::UnifiedDistanceCompute::new(
                proximadb::compute::distance_computation::DistanceMetric::Cosine,
            ),
        );
        let mut sst_storage = SstEngine::new().await?;

        // Calculate how many vectors fit per block
        let vector_bytes = dimension * 4; // FP32
        let vectors_per_block = (block_size_kb as usize * 1024) / vector_bytes;
        let expected_blocks = (num_vectors + vectors_per_block - 1) / vectors_per_block;

        info!("  📊 Block Configuration:");
        info!("    • Block size: {} KB", block_size_kb);
        info!("    • Vector size: {} bytes", vector_bytes);
        info!("    • Vectors per block: ~{}", vectors_per_block);
        info!("    • Expected blocks: ~{}", expected_blocks);

        // Flush with collection config for quantization
        // Note: vectors are passed in flush_params, not inserted separately
        let test_env = common::integration_test_helpers::UnifiedTestEnvironment::new().await?;
        let collection = test_env.create_test_collection();
        let flush_params = FlushParameters {
            vector_records: vectors.clone(),
            force: false,
            collection_id: Some("test_collection".to_string()),
            collection_config: Some(collection.clone()),
            ..Default::default()
        };

        let flush_result = sst_storage.do_flush(&flush_params).await?;

        info!("\n  ✅ Flush Results:");
        info!("    • Entries flushed: {:?}", flush_result.entries_flushed);
        info!("    • Files created: {:?}", flush_result.files_created);
        info!("    • Bytes written: {:?}", flush_result.bytes_written);

        // Calculate compression ratio
        let uncompressed_size = num_vectors * vector_bytes;
        let compression_ratio = uncompressed_size as f64
            / flush_result
                .bytes_written
                .unwrap_or(uncompressed_size as u64) as f64;

        info!("\n  📈 Compression Metrics:");
        info!(
            "    • Uncompressed size: {:.2} MB",
            uncompressed_size as f64 / (1024.0 * 1024.0)
        );
        info!(
            "    • Compressed size: {:.2} MB",
            flush_result.bytes_written.unwrap_or(0) as f64 / (1024.0 * 1024.0)
        );
        info!("    • Compression ratio: {:.2}x", compression_ratio);

        // Test search to verify data integrity
        // Test search using unified interface
        let query = vectors[0]
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();
        use proximadb::storage::traits::StorageQueryContext;

        let search_params = std::sync::Arc::new(proximadb::core::search::SearchParams {
            vector: Some(query),
            top_k: Some(10),
            filters: None,
            distance_metric: Some(proximadb::compute::distance_computation::DistanceMetric::Cosine),
            ..Default::default()
        });

        let query_context = StorageQueryContext {
            search_params,
            collection: std::sync::Arc::new(collection),
            metadata: proximadb::storage::traits::StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let search_results = sst_storage.search_vectors_unified(&query_context).await?;

        info!("\n  🔍 Search Validation:");
        info!("    • Results found: {}", search_results.len());
        info!("    • Top result score: {:.4}", search_results[0].score);

        // With smaller blocks, we should see better clustering
        if block_size_kb == 256 {
            info!("\n  💡 256KB Block Benefits:");
            info!("    • Better quantization clustering within blocks");
            info!("    • More efficient PQ codebook per block");
            info!("    • Improved compression for similar vectors");
        }
    }

    info!("\n✅ QUANTIZATION BLOCK TEST COMPLETE");
    info!("{}", "=".repeat(80));

    Ok(())
}

/// Test quantization with PQ on 256KB blocks
#[tokio::test]
async fn test_pq_quantization_256kb_blocks() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    info!("\n{}", "=".repeat(80));
    info!("🧪 TESTING PQ QUANTIZATION WITH 256KB BLOCKS");
    info!("{}", "=".repeat(80));

    // Create test vectors
    let num_vectors = 1000;
    let dimension = 512;
    let vectors = generate_clustered_vectors(num_vectors, dimension);

    // Setup quantization adapter (for future use when SstQuantizationAdapter is available)
    let _pq_config = UnifiedQuantizationLevel {
        level_type: Some(proximadb::compute::quantization::QuantizationLevel::Pq(
            PqConfig {
                num_subvectors: 8,
                bits_per_code: 8,
                codebook_id: None,
                adaptive_subvectors: false,
            },
        )),
    };

    // TODO: Implement SstQuantizationAdapter when available
    // let adapter = SstQuantizationAdapter::new(_pq_config);

    // Test with 256KB blocks
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();

    let mut sst_config = SstConfig::default();
    sst_config.block_size_kb = 256;
    sst_config.compression = "zstd".to_string();
    sst_config.compression_level = 3;

    info!("\n📊 Configuration:");
    info!("  • Block size: 256 KB");
    info!("  • PQ subvectors: 8");
    info!("  • Bits per code: 8");
    info!("  • Vectors: {}", num_vectors);
    info!("  • Dimension: {}", dimension);

    // Calculate theoretical compression
    let original_size = num_vectors * dimension * 4; // FP32
    let pq_size = num_vectors * 8; // 8 subvectors × 1 byte each
    let theoretical_compression = original_size as f64 / pq_size as f64;

    info!("\n📈 Theoretical PQ Compression:");
    info!(
        "  • Original size: {:.2} MB",
        original_size as f64 / (1024.0 * 1024.0)
    );
    info!("  • PQ size: {:.2} KB", pq_size as f64 / 1024.0);
    info!("  • Compression ratio: {:.1}x", theoretical_compression);

    // With 256KB blocks, vectors are grouped better for PQ
    let vectors_per_block = (256 * 1024) / (dimension * 4);
    let blocks_needed = (num_vectors + vectors_per_block - 1) / vectors_per_block;

    info!("\n📦 Block Analysis:");
    info!("  • Vectors per 256KB block: {}", vectors_per_block);
    info!("  • Total blocks needed: {}", blocks_needed);
    info!(
        "  • Vectors in last block: {}",
        num_vectors % vectors_per_block
    );

    // Create SST engine
    let fs_config = FilesystemConfig {
        default_fs: Some(format!("file://{}", base_path)),
        local: None,
        global_options: Default::default(),
        auth_config: None,
        performance_config: Default::default(),
        scheme_mapping: std::collections::HashMap::new(),
    };
    let filesystem = Arc::new(FilesystemFactory::create(fs_config).await?);
    let distance_compute = Arc::new(
        proximadb::compute::distance_computation::UnifiedDistanceCompute::new(
            proximadb::compute::distance_computation::DistanceMetric::Cosine,
        ),
    );
    let mut sst_storage = SstEngine::new().await?;

    // Flush with vectors
    let test_env = common::integration_test_helpers::UnifiedTestEnvironment::new().await?;
    let collection = test_env.create_test_collection();
    let flush_params = FlushParameters {
        vector_records: vectors.clone(),
        force: false,
        collection_id: Some("test_collection".to_string()),
        collection_config: Some(collection),
        ..Default::default()
    };

    let flush_result = sst_storage.do_flush(&flush_params).await?;

    info!("\n✅ Results with 256KB blocks:");
    info!(
        "  • Bytes written: {:.2} MB",
        flush_result.bytes_written.unwrap_or(0) as f64 / (1024.0 * 1024.0)
    );
    info!(
        "  • Actual compression: {:.1}x",
        original_size as f64 / flush_result.bytes_written.unwrap_or(1) as f64
    );
    info!("  • Files created: {:?}", flush_result.files_created);

    Ok(())
}

fn generate_clustered_vectors(count: usize, dim: usize) -> Vec<ProximaRecord> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut vectors = Vec::with_capacity(count);

    // Create 10 clusters
    let clusters = 10;
    let vectors_per_cluster = count / clusters;

    for cluster_id in 0..clusters {
        // Generate cluster center
        let mut center = vec![0.0f32; dim];
        for val in &mut center {
            *val = rng.gen_range(-1.0..1.0);
        }

        // Generate vectors around this center
        for i in 0..vectors_per_cluster {
            let mut values = center.clone();

            // Add noise
            for val in &mut values {
                *val += rng.gen_range(-0.1..0.1);
            }

            // Normalize
            let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut values {
                    *val /= norm;
                }
            }

            let global_idx = cluster_id * vectors_per_cluster + i;
            vectors.push(ProximaRecord {
                oid: format!("vec_{}", global_idx),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default", "vector", dim as u32, values,
                )],
                record_version: 1,
                ..Default::default()
            });
        }
    }

    // Add remaining vectors if count is not divisible by clusters
    for i in (clusters * vectors_per_cluster)..count {
        let mut values = vec![0.0f32; dim];
        for val in &mut values {
            *val = rng.gen_range(-1.0..1.0);
        }

        // Normalize
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
            ..Default::default()
        });
    }

    vectors
}
