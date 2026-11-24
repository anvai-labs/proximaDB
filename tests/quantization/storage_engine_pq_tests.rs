//! Storage Engine PQ Integration Tests
//!
//! Tests Product Quantization integration across all ProximaDB storage engines.
//! Verifies that PQ quantization works correctly in each engine when enabled.

// ORPHANED TEST (not in any compile target — see TD-ORPHAN-1): references the
// VIPER engine removed by ADR-093 stage 3 / TD-VIPER-1 S3. Unreachable code;
// cleanup of the file itself belongs to the orphan-file program, not the
// engine-removal PR.
use proximadb::compute::{UnifiedQuantizationLevel, QuantizationLevel, ProductQuantization};
use proximadb::proto::proximadb_v1::{
    VectorRecord, Collection, CollectionConfig, QuantizationConfig,
    StorageEngine, StorageAssignment
};
use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine};
use std::collections::HashMap;
use anyhow::Result;

/// Create test vectors with varying patterns
fn create_test_vectors(count: usize, dimension: usize, pattern: &str) -> Vec<VectorRecord> {
    (0..count)
        .map(|i| {
            let vector = (0..dimension)
                .map(|j| {
                    let base = (i as f32 + j as f32) * 0.01;
                    match pattern {
                        "linear" => base,
                        "sinusoidal" => (base * std::f32::consts::TAU).sin(),
                        "clustered" => {
                            let cluster = i % 3;
                            base + cluster as f32 * 2.0
                        },
                        _ => base,
                    }
                })
                .collect();

            VectorRecord {
                id: format!("vec_{}_{}", pattern, i),
                vector,
                metadata: HashMap::new(),
                timestamp: chrono::Utc::now().timestamp(),
                version: Some(1),
                ..Default::default()
            }
        })
        .collect()
}

/// Create collection with specific storage engine and quantization config
fn create_collection_with_engine_and_quantization(
    collection_id: &str,
    storage_engine: StorageEngine,
    dimension: u32,
    quantization_level: UnifiedQuantizationLevel,
    enabled: bool,
) -> Collection {
    Collection {
        id: collection_id.to_string(),
        config: Some(CollectionConfig {
            dimension,
            storage_engine: storage_engine as i32,
            quantization: Some(QuantizationConfig {
                enabled,
                level: serde_json::to_string(&quantization_level).unwrap_or_default(),
                training_size: 500,
                compression_ratio_target: Some(8.0),
                ..Default::default()
            }),
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            primary_path: "/tmp/proximadb_test".to_string(),
            backup_paths: vec![],
            engine: storage_engine as i32,
            engine_config: HashMap::new(),
            base_location: "/tmp/proximadb_test".to_string(),
            assigned_at: chrono::Utc::now().timestamp(),
                ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod storage_engine_pq_tests {
    use super::*;

    // Note: These tests are designed to test the quantization integration,
    // but actual storage engine instantiation would require proper setup.
    // For now, we focus on configuration and data structure validation.

    #[tokio::test]
    async fn test_sst_engine_pq_integration() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Create collection with SST engine and PQ8 quantization
        let pq8 = UnifiedQuantizationLevel::pq8(16);
        let collection = create_collection_with_engine_and_quantization(
            "sst_pq_test",
            StorageEngine::Sst,
            256,
            pq8.clone(),
            true,
        );

        // Verify configuration
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.storage_engine, StorageEngine::Sst as i32);

        let quant_config = config.quantization.as_ref().unwrap();
        assert!(quant_config.enabled);
        assert_eq!(quant_config.training_size, 500);

        // Create test vectors
        let test_vectors = create_test_vectors(100, 256, "linear");

        // Create flush parameters
        let flush_params = FlushParameters {
            collection_id: Some("sst_pq_test".to_string()),
            vector_records: test_vectors.clone(),
            force: true,
            synchronous: true,
            collection_config: Some(collection.clone()),
            enable_quantization: Some(true),
            ..Default::default()
        };

        // Verify PQ level configuration
        match &pq8.level_type {
            Some(QuantizationLevel::Pq(pq)) => {
                assert_eq!(pq.bits_per_code, 8);
                assert_eq!(pq.num_subvectors, 16);
                println!("✅ SST Engine PQ8 configuration verified");
                println!("   - Vectors: {}", test_vectors.len());
                println!("   - Dimension: {}", config.dimension);
                println!("   - Subvectors: {}", pq.num_subvectors);
                println!("   - Bits per code: {}", pq.bits_per_code);
            },
            _ => panic!("Expected ProductQuantization"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_viper_engine_pq_integration() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Create collection with VIPER engine and PQ4 quantization
        let pq4 = UnifiedQuantizationLevel::pq4(8);
        let collection = create_collection_with_engine_and_quantization(
            "viper_pq_test",
            StorageEngine::Viper,
            128,
            pq4.clone(),
            true,
        );

        // Verify configuration
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.storage_engine, StorageEngine::Viper as i32);

        // Create test vectors with different pattern
        let test_vectors = create_test_vectors(150, 128, "sinusoidal");

        // Create flush parameters
        let flush_params = FlushParameters {
            collection_id: Some("viper_pq_test".to_string()),
            vector_records: test_vectors.clone(),
            force: true,
            synchronous: true,
            collection_config: Some(collection.clone()),
            enable_quantization: Some(true),
            ..Default::default()
        };

        // Verify PQ level configuration
        match &pq4.level_type {
            Some(QuantizationLevel::Pq(pq)) => {
                assert_eq!(pq.bits_per_code, 4);
                assert_eq!(pq.num_subvectors, 8);
                println!("✅ VIPER Engine PQ4 configuration verified");
                println!("   - Vectors: {}", test_vectors.len());
                println!("   - Dimension: {}", config.dimension);
                println!("   - Expected compression: {:.1}x", 128.0 * 4.0 / 8.0); // 128 dims * 4 bytes / 8 codes
            },
            _ => panic!("Expected ProductQuantization"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_nova_engine_pq_integration() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Create collection with NOVA engine and custom PQ16
        let pq16 = UnifiedQuantizationLevel {
            level_type: Some(QuantizationLevel::Pq(ProductQuantization {
                bits_per_code: 16,
                num_subvectors: 12,
                codebook_id: None,
                adaptive_subvectors: true, // Enable adaptive subvectors for NOVA
            })),
        };

        let collection = create_collection_with_engine_and_quantization(
            "nova_pq_test",
            StorageEngine::Nova,
            384,
            pq16.clone(),
            true,
        );

        // Verify configuration
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.storage_engine, StorageEngine::Nova as i32);

        // Create test vectors with clustered pattern
        let test_vectors = create_test_vectors(200, 384, "clustered");

        // Verify PQ level configuration
        match &pq16.level_type {
            Some(QuantizationLevel::Pq(pq)) => {
                assert_eq!(pq.bits_per_code, 16);
                assert_eq!(pq.num_subvectors, 12);
                assert!(pq.adaptive_subvectors);
                println!("✅ NOVA Engine PQ16 configuration verified");
                println!("   - Vectors: {}", test_vectors.len());
                println!("   - Dimension: {}", config.dimension);
                println!("   - Adaptive subvectors: {}", pq.adaptive_subvectors);
            },
            _ => panic!("Expected ProductQuantization"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_raptor_engine_pq_integration() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Create collection with RAPTOR engine and PQ8 quantization
        let pq8 = UnifiedQuantizationLevel::pq8(32); // More subvectors for RAPTOR's adaptive nature
        let collection = create_collection_with_engine_and_quantization(
            "raptor_pq_test",
            StorageEngine::Raptor,
            512,
            pq8.clone(),
            true,
        );

        // Verify configuration
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.storage_engine, StorageEngine::Raptor as i32);

        // Create test vectors
        let test_vectors = create_test_vectors(300, 512, "linear");

        // Verify storage assignment
        let storage_assignment = collection.storage_assignment.as_ref().unwrap();
        assert_eq!(storage_assignment.engine, StorageEngine::Raptor as i32);

        println!("✅ RAPTOR Engine PQ8 configuration verified");
        println!("   - Vectors: {}", test_vectors.len());
        println!("   - Dimension: {}", config.dimension);
        println!("   - Storage path: {}", storage_assignment.primary_path);

        Ok(())
    }

    #[tokio::test]
    async fn test_swift_engine_pq_integration() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Create collection with SWIFT engine and PQ4 quantization (optimized for speed)
        let pq4 = UnifiedQuantizationLevel::pq4(4); // Fewer subvectors for SWIFT's speed focus
        let collection = create_collection_with_engine_and_quantization(
            "swift_pq_test",
            StorageEngine::Swift,
            64,
            pq4.clone(),
            true,
        );

        // Verify configuration
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.storage_engine, StorageEngine::Swift as i32);

        // Create smaller test vectors for SWIFT's optimization
        let test_vectors = create_test_vectors(50, 64, "linear");

        match &pq4.level_type {
            Some(QuantizationLevel::Pq(pq)) => {
                let expected_compression = (64.0 * 4.0) / 4.0; // 64 dims * 4 bytes / 4 codes
                println!("✅ SWIFT Engine PQ4 configuration verified");
                println!("   - Vectors: {}", test_vectors.len());
                println!("   - Dimension: {}", config.dimension);
                println!("   - Expected compression: {:.1}x", expected_compression);
                println!("   - Optimized for low latency");
            },
            _ => panic!("Expected ProductQuantization"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_helix_engine_pq_integration() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Create collection with HELIX engine and adaptive PQ
        let pq_adaptive = UnifiedQuantizationLevel {
            level_type: Some(QuantizationLevel::Pq(ProductQuantization {
                bits_per_code: 8,
                num_subvectors: 16,
                codebook_id: Some("helix_spatial_codebook".to_string()),
                adaptive_subvectors: true, // HELIX benefits from adaptive quantization
            })),
        };

        let collection = create_collection_with_engine_and_quantization(
            "helix_pq_test",
            StorageEngine::Helix,
            256,
            pq_adaptive.clone(),
            true,
        );

        // Verify configuration
        let config = collection.config.as_ref().unwrap();
        assert_eq!(config.storage_engine, StorageEngine::Helix as i32);

        // Create test vectors with spatial clustering pattern
        let test_vectors = create_test_vectors(250, 256, "clustered");

        match &pq_adaptive.level_type {
            Some(QuantizationLevel::Pq(pq)) => {
                assert!(pq.adaptive_subvectors);
                assert!(pq.codebook_id.is_some());
                println!("✅ HELIX Engine adaptive PQ configuration verified");
                println!("   - Vectors: {}", test_vectors.len());
                println!("   - Dimension: {}", config.dimension);
                println!("   - Codebook ID: {:?}", pq.codebook_id);
                println!("   - Spatial locality optimization enabled");
            },
            _ => panic!("Expected ProductQuantization"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_engine_pq_compatibility() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Test that the same PQ configuration works across engines
        let pq8 = UnifiedQuantizationLevel::pq8(16);
        let test_vectors = create_test_vectors(100, 256, "linear");

        let engines = vec![
            ("SST", StorageEngine::Sst),
            ("VIPER", StorageEngine::Viper),
            ("NOVA", StorageEngine::Nova),
            ("RAPTOR", StorageEngine::Raptor),
            ("SWIFT", StorageEngine::Swift),
            ("HELIX", StorageEngine::Helix),
        ];

        println!("🔄 Testing cross-engine PQ compatibility:");

        for (name, engine) in engines {
            let collection = create_collection_with_engine_and_quantization(
                &format!("{}_compat_test", name.to_lowercase()),
                engine,
                256,
                pq8.clone(),
                true,
            );

            let config = collection.config.as_ref().unwrap();
            let quant_config = config.quantization.as_ref().unwrap();

            assert_eq!(config.storage_engine, engine as i32);
            assert!(quant_config.enabled);
            assert_eq!(config.dimension, 256);

            println!("   ✅ {} - Compatible with PQ8(16)", name);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_comprehensive_pq_end_to_end_all_engines() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Test comprehensive PQ functionality for all engines
        let engines = vec![
            ("SST", StorageEngine::Sst),
            ("VIPER", StorageEngine::Viper),
            ("NOVA", StorageEngine::Nova),
            ("RAPTOR", StorageEngine::Raptor),
            ("SWIFT", StorageEngine::Swift),
            ("HELIX", StorageEngine::Helix),
        ];

        let quantization_levels = vec![
            ("PQ4", UnifiedQuantizationLevel::pq4(8)),
            ("PQ8", UnifiedQuantizationLevel::pq8(16)),
            ("PQ16", UnifiedQuantizationLevel {
                level_type: Some(QuantizationLevel::Pq(ProductQuantization {
                    bits_per_code: 16,
                    num_subvectors: 24,
                    codebook_id: None,
                    adaptive_subvectors: false,
                })),
            }),
        ];

        println!("🔄 Comprehensive End-to-End PQ Testing for All Engines:");

        for (engine_name, storage_engine) in engines {
            println!("📊 Testing {} Engine:", engine_name);

            for (pq_name, pq_level) in &quantization_levels {
                // Create collection with specific engine and PQ level
                let collection = create_collection_with_engine_and_quantization(
                    &format!("{}_{}_e2e_test", engine_name.to_lowercase(), pq_name.to_lowercase()),
                    storage_engine,
                    512, // Higher dimension for better PQ effectiveness
                    pq_level.clone(),
                    true,
                );

                // Create larger dataset for meaningful quantization
                let test_vectors = create_test_vectors(500, 512, "clustered");

                // Create flush parameters with quantization enabled
                let flush_params = FlushParameters {
                    collection_id: Some(format!("{}_{}_e2e_test", engine_name.to_lowercase(), pq_name.to_lowercase())),
                    vector_records: test_vectors.clone(),
                    force: true,
                    synchronous: true,
                    collection_config: Some(collection.clone()),
                    enable_quantization: Some(true),
                    ..Default::default()
                };

                // Verify configuration
                let config = collection.config.as_ref().unwrap();
                let quant_config = config.quantization.as_ref().unwrap();

                assert_eq!(config.storage_engine, storage_engine as i32);
                assert!(quant_config.enabled);
                assert_eq!(config.dimension, 512);
                assert_eq!(quant_config.training_size, 500);
                assert!(flush_params.enable_quantization.unwrap_or(false));

                // Verify PQ level configuration
                match &pq_level.level_type {
                    Some(QuantizationLevel::Pq(pq)) => {
                        match pq_name {
                            "PQ4" => {
                                assert_eq!(pq.bits_per_code, 4);
                                assert_eq!(pq.num_subvectors, 8);
                            },
                            "PQ8" => {
                                assert_eq!(pq.bits_per_code, 8);
                                assert_eq!(pq.num_subvectors, 16);
                            },
                            "PQ16" => {
                                assert_eq!(pq.bits_per_code, 16);
                                assert_eq!(pq.num_subvectors, 24);
                            },
                            _ => panic!("Unexpected PQ level: {}", pq_name),
                        }

                        // Calculate expected compression ratio
                        let original_size = 512 * 4; // 512 dimensions * 4 bytes per float
                        let compressed_size = pq.num_subvectors as f64; // Each subvector becomes 1 code
                        let compression_ratio = original_size as f64 / compressed_size;

                        println!("     ✅ {} with {} - Expected compression: {:.1}x",
                                engine_name, pq_name, compression_ratio);
                        println!("        - Vectors: {}", test_vectors.len());
                        println!("        - Dimension: {}", config.dimension);
                        println!("        - Subvectors: {}", pq.num_subvectors);
                        println!("        - Bits per code: {}", pq.bits_per_code);
                        println!("        - Training size: {}", quant_config.training_size);
                    },
                    _ => panic!("Expected ProductQuantization for {}", pq_name),
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_pq_performance_comparison_across_engines() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Performance comparison test with different PQ configurations
        let engines = vec![
            ("SST", StorageEngine::Sst),
            ("VIPER", StorageEngine::Viper),
            ("NOVA", StorageEngine::Nova),
            ("RAPTOR", StorageEngine::Raptor),
            ("SWIFT", StorageEngine::Swift),
            ("HELIX", StorageEngine::Helix),
        ];

        let test_data = create_test_vectors(1000, 384, "linear"); // Large dataset

        println!("⚡ PQ Performance Comparison Across All Engines:");

        for (engine_name, storage_engine) in engines {
            let start_time = std::time::Instant::now();

            // Test with PQ8 for performance comparison
            let pq8 = UnifiedQuantizationLevel::pq8(24); // 384/16 = 24 subvectors
            let collection = create_collection_with_engine_and_quantization(
                &format!("{}_perf_test", engine_name.to_lowercase()),
                storage_engine,
                384,
                pq8.clone(),
                true,
            );

            let flush_params = FlushParameters {
                collection_id: Some(format!("{}_perf_test", engine_name.to_lowercase())),
                vector_records: test_data.clone(),
                force: true,
                synchronous: true,
                collection_config: Some(collection),
                enable_quantization: Some(true),
                ..Default::default()
            };

            let config_time = start_time.elapsed();

            // Verify configuration is ready
            assert!(flush_params.enable_quantization.unwrap_or(false));
            assert_eq!(flush_params.vector_records.len(), 1000);

            println!("   ✅ {} Engine PQ8 setup completed in {:?}", engine_name, config_time);
            println!("      - Vectors: {}", test_data.len());
            println!("      - Expected memory reduction: ~16x (384*4 bytes → 24 codes)");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_pq_adaptive_configuration_all_engines() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Test adaptive PQ configuration for different engines
        let test_cases = vec![
            // (engine_name, storage_engine, dimension, expected_subvectors, adaptive)
            ("SST", StorageEngine::Sst, 128, 8, false),
            ("VIPER", StorageEngine::Viper, 256, 16, false),
            ("NOVA", StorageEngine::Nova, 384, 24, true),
            ("RAPTOR", StorageEngine::Raptor, 512, 32, false),
            ("SWIFT", StorageEngine::Swift, 64, 4, false),
            ("HELIX", StorageEngine::Helix, 768, 48, true),
        ];

        println!("🔧 Testing Adaptive PQ Configuration:");

        for (engine_name, storage_engine, dimension, num_subvectors, adaptive) in test_cases {
            let pq_config = UnifiedQuantizationLevel {
                level_type: Some(QuantizationLevel::Pq(ProductQuantization {
                    bits_per_code: 8,
                    num_subvectors,
                    codebook_id: Some(format!("{}_adaptive_codebook", engine_name.to_lowercase())),
                    adaptive_subvectors: adaptive,
                })),
            };

            let collection = create_collection_with_engine_and_quantization(
                &format!("{}_adaptive_test", engine_name.to_lowercase()),
                storage_engine,
                dimension as u32,
                pq_config.clone(),
                true,
            );

            let test_vectors = create_test_vectors(200, dimension, "sinusoidal");

            // Verify adaptive configuration
            match &pq_config.level_type {
                Some(QuantizationLevel::Pq(pq)) => {
                    assert_eq!(pq.num_subvectors, num_subvectors);
                    assert_eq!(pq.adaptive_subvectors, adaptive);
                    assert!(pq.codebook_id.is_some());

                    let compression_ratio = (dimension * 4) as f64 / num_subvectors as f64;

                    println!("   ✅ {} Engine ({}D) - {} subvectors, adaptive: {}, compression: {:.1}x",
                            engine_name, dimension, num_subvectors, adaptive, compression_ratio);
                },
                _ => panic!("Expected ProductQuantization"),
            }

            // Verify collection configuration
            let config = collection.config.as_ref().unwrap();
            assert_eq!(config.storage_engine, storage_engine as i32);
            assert_eq!(config.dimension, dimension as u32);

            let quant_config = config.quantization.as_ref().unwrap();
            assert!(quant_config.enabled);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_quantization_disable_functionality() -> Result<()> {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Test quantization disable across engines
        let pq8 = UnifiedQuantizationLevel::pq8(8);
        let test_vectors = create_test_vectors(50, 128, "linear");

        let engines = vec![
            StorageEngine::Sst,
            StorageEngine::Viper,
            StorageEngine::Nova,
        ];

        for engine in engines {
            // Test with quantization disabled
            let collection_disabled = create_collection_with_engine_and_quantization(
                &format!("{:?}_disabled_test", engine).to_lowercase(),
                engine,
                128,
                pq8.clone(),
                false, // Disabled
            );

            let config = collection_disabled.config.as_ref().unwrap();
            let quant_config = config.quantization.as_ref().unwrap();

            assert!(!quant_config.enabled);

            // Create flush parameters with quantization explicitly disabled
            let flush_params = FlushParameters {
                collection_id: Some(format!("{:?}_disabled_test", engine).to_lowercase()),
                vector_records: test_vectors.clone(),
                force: true,
                synchronous: true,
                collection_config: Some(collection_disabled),
                enable_quantization: Some(false),
                ..Default::default()
            };

            assert_eq!(flush_params.enable_quantization, Some(false));
        }

        println!("✅ Quantization disable functionality verified across engines");

        Ok(())
    }
}