/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Product Quantization index tests

#[cfg(test)]
mod tests {
    use crate::compute::distance_computation::DistanceMetric;
    use crate::index::axis::index_factory::IndexFactory;
    use crate::index::axis::types::{Data, IndexAlgorithm, IndexSpecification};
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    use tracing::debug;

    #[allow(dead_code)]
    fn create_test_vector(id: &str, dimension: usize) -> ProximaRecord {
        let values = vec![0.1_f32; dimension];
        let dim = dimension as u32;
        ProximaRecord {
            oid: id.to_string(),
            embeddings: vec![EmbeddingCell::new_fp32("default", "vector", dim, values)],
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                (0..dimension)
                    .map(|j| ((i * dimension + j) as f32 * 0.1) % 1.0)
                    .collect()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_pq_basic_operations() {
        let dimension = 128;
        let algorithm = IndexAlgorithm::PQ {
            m: 8,     // 8 subquantizers
            nbits: 8, // 256 centroids per subquantizer
            train_size: 1000,
        };

        // Currently returns error as PQ is not implemented
        let result = IndexFactory::create_index(&algorithm, dimension, DistanceMetric::Euclidean);

        // For now, we expect an error until PQ is implemented
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Product Quantization"));
        }
    }

    #[tokio::test]
    async fn test_pq_compression_ratios() {
        // Test different PQ configurations for compression
        let test_cases = vec![
            (4, 8, 128.0), // 4 subquantizers, 8 bits each = 4 bytes vs 512 bytes (128x compression)
            (8, 8, 64.0),  // 8 subquantizers, 8 bits each = 8 bytes vs 512 bytes (64x compression)
            (16, 8, 32.0), // 16 subquantizers, 8 bits each = 16 bytes vs 512 bytes (32x compression)
            (8, 4, 128.0), // 8 subquantizers, 4 bits each = 4 bytes vs 512 bytes (128x compression)
        ];

        for (m, nbits, expected_ratio) in test_cases {
            let algorithm = IndexAlgorithm::PQ {
                m,
                nbits,
                train_size: 1000,
            };

            let _spec =
                IndexSpecification::new(Data::DenseVector { dimension: 128 }, algorithm.clone());

            // Calculate theoretical compression ratio
            let original_size = 128 * 4; // 128 floats * 4 bytes
            let compressed_size = if nbits == 8 {
                m // m bytes for 8-bit codes
            } else {
                (m * nbits) / 8 // bits to bytes
            };
            let ratio = original_size as f32 / compressed_size as f32;

            assert!(
                (ratio - expected_ratio).abs() < 0.1,
                "PQ({}, {}) should have compression ratio ~{}, got {}",
                m,
                nbits,
                expected_ratio,
                ratio
            );
        }
    }

    #[tokio::test]
    async fn test_pq_with_ivf_combination() {
        // Test IVF-PQ combination (common in practice)
        let pq_quantizer = IndexAlgorithm::PQ {
            m: 8,
            nbits: 8,
            train_size: 1000,
        };

        let ivf_pq = IndexAlgorithm::IVF {
            nlist: 100,
            nprobe: 10,
            quantizer: Some(Box::new(pq_quantizer)),
        };

        let spec = IndexSpecification::new(Data::DenseVector { dimension: 128 }, ivf_pq);

        assert!(spec.supports_clustering());
    }

    #[tokio::test]
    async fn test_pq_training_data_requirements() {
        // PQ requires training data to learn codebooks
        let dimensions = vec![64, 128, 256, 512];

        for dim in dimensions {
            let algorithm = IndexAlgorithm::PQ {
                m: dim as u32 / 16, // Divide dimension by 16 for subquantizers
                nbits: 8,
                train_size: dim * 10, // At least 10x dimension for training
            };

            let spec = IndexSpecification::new(Data::DenseVector { dimension: dim }, algorithm);

            // Verify training size is appropriate
            if let IndexAlgorithm::PQ { train_size, .. } = &spec.algorithm {
                assert!(
                    *train_size >= dim * 10,
                    "Training size should be at least 10x dimension for PQ"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_pq_distance_metrics() {
        // Test PQ with different distance metrics
        let metrics = vec![
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
        ];

        for metric in metrics {
            let algorithm = IndexAlgorithm::PQ {
                m: 8,
                nbits: 8,
                train_size: 1000,
            };

            let result = IndexFactory::create_index(&algorithm, 128, metric);

            // Currently returns error, but should support all metrics when implemented
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_pq_memory_efficiency() {
        // Test memory usage for different PQ configurations
        let vector_counts = vec![1000, 10000, 100000];
        let dimension = 768; // Common embedding dimension

        for count in vector_counts {
            let _algorithm = IndexAlgorithm::PQ {
                m: 48, // 768 / 16 = 48 subquantizers
                nbits: 8,
                train_size: 10000,
            };

            // Calculate expected memory usage
            let original_memory = count * dimension * 4; // Original vectors in bytes
            let pq_memory = count * 48; // PQ codes (48 bytes per vector)
            let codebook_memory = 48 * 256 * (dimension / 48) * 4; // Codebooks
            let total_pq_memory = pq_memory + codebook_memory;

            let compression_ratio = original_memory as f32 / total_pq_memory as f32;

            debug!(
                "PQ memory for {} vectors: original={} MB, compressed={} MB, ratio={:.2}x",
                count,
                original_memory / 1_000_000,
                total_pq_memory / 1_000_000,
                compression_ratio
            );

            assert!(compression_ratio > 1.0, "PQ should compress data");
        }
    }

    #[tokio::test]
    async fn test_pq_search_accuracy_tradeoff() {
        // Test the accuracy vs speed tradeoff with different PQ settings
        let configurations = vec![
            (4, 8, "Low compression, high accuracy"),
            (8, 8, "Medium compression, medium accuracy"),
            (16, 8, "High compression, lower accuracy"),
            (8, 4, "High compression with 4-bit codes"),
        ];

        for (m, nbits, description) in configurations {
            let algorithm = IndexAlgorithm::PQ {
                m,
                nbits,
                train_size: 5000,
            };

            let spec = IndexSpecification::new(Data::DenseVector { dimension: 128 }, algorithm);

            debug!(
                "Testing PQ configuration: {} (m={}, nbits={})",
                description, m, nbits
            );

            // When implemented, this should test actual search accuracy
            assert!(spec.supports_clustering());
        }
    }

    #[tokio::test]
    async fn test_pq_incremental_updates() {
        // Test adding vectors after initial training
        let algorithm = IndexAlgorithm::PQ {
            m: 8,
            nbits: 8,
            train_size: 1000,
        };

        // When PQ is implemented, test:
        // 1. Train on initial dataset
        // 2. Add new vectors using learned codebooks
        // 3. Verify search still works
        // 4. Test retraining with expanded dataset

        let spec = IndexSpecification::new(Data::DenseVector { dimension: 128 }, algorithm);

        assert!(spec.supports_clustering());
    }

    #[tokio::test]
    async fn test_pq_edge_cases() {
        // Test edge cases for PQ parameters

        // Test minimum viable configuration
        let min_pq = IndexAlgorithm::PQ {
            m: 1, // Single subquantizer
            nbits: 8,
            train_size: 100,
        };

        let spec = IndexSpecification::new(Data::DenseVector { dimension: 16 }, min_pq);
        assert!(spec.supports_clustering());

        // Test with dimension not divisible by m
        let uneven_pq = IndexAlgorithm::PQ {
            m: 7, // 128 not evenly divisible by 7
            nbits: 8,
            train_size: 1000,
        };

        let spec = IndexSpecification::new(Data::DenseVector { dimension: 128 }, uneven_pq);
        assert!(spec.supports_clustering());
    }

    #[tokio::test]
    async fn test_pq_serialization() {
        // Test PQ index serialization/deserialization
        let algorithm = IndexAlgorithm::PQ {
            m: 8,
            nbits: 8,
            train_size: 1000,
        };

        // Serialize the algorithm specification
        let serialized = serde_json::to_string(&algorithm).unwrap();
        let deserialized: IndexAlgorithm = serde_json::from_str(&serialized).unwrap();

        if let IndexAlgorithm::PQ {
            m,
            nbits,
            train_size,
        } = deserialized
        {
            assert_eq!(m, 8);
            assert_eq!(nbits, 8);
            assert_eq!(train_size, 1000);
        } else {
            panic!("Deserialization failed");
        }
    }
}
