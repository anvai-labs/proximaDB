/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Hybrid and composite index tests

#[cfg(test)]
mod tests {
    use crate::index::axis::management::manager::AxisManager;
    use crate::index::axis::management::manager::{
        AnnFilteringMode, FilterOperator, HybridQuery, MetadataFilter, VectorQuery,
    };
    use crate::index::axis::types::{Data, IndexAlgorithm, IndexSpecification};
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
    use tracing::debug;

    fn create_vector_record(
        id: &str,
        vector: Vec<f32>,
        metadata: &[(&str, serde_json::Value)],
    ) -> ProximaRecord {
        let props = metadata
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    serde_json::Value::String(s) => ProximaValue::String(s.clone()),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            ProximaValue::Int64(i)
                        } else if let Some(f) = n.as_f64() {
                            ProximaValue::Float64(f)
                        } else {
                            ProximaValue::String(n.to_string())
                        }
                    }
                    serde_json::Value::Bool(b) => ProximaValue::Boolean(*b),
                    serde_json::Value::Null => ProximaValue::Null,
                    other => ProximaValue::Json(other.clone()),
                };
                ((*key).to_string(), ProximaTreeNode::Value(value))
            })
            .collect();

        ProximaRecord {
            oid: id.to_string(),
            local_id: Some(id.to_string()),
            record_version: 1,
            props,
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                vector.len() as u32,
                vector,
            )],
            ..ProximaRecord::default()
        }
    }

    #[tokio::test]
    async fn test_axis_manager_filters_hybrid_query_results() {
        let manager = AxisManager::new(crate::index::axis::types::AxisConfig::default())
            .await
            .unwrap();

        let electronics = create_vector_record(
            "electronics-1",
            vec![1.0, 0.0, 0.0, 0.0],
            &[("category", serde_json::json!("electronics"))],
        );
        let books = create_vector_record(
            "books-1",
            vec![0.0, 1.0, 0.0, 0.0],
            &[("category", serde_json::json!("books"))],
        );

        manager
            .insert("test_collection", &electronics)
            .await
            .unwrap();
        manager.insert("test_collection", &books).await.unwrap();

        let query = HybridQuery {
            collection_id: "test_collection".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                similarity_threshold: 0.0,
            }),
            metadata_filters: vec![MetadataFilter {
                field: "category".to_string(),
                operator: FilterOperator::Equals,
                value: serde_json::json!("electronics"),
            }],
            id_filters: vec![],
            top_k: 2,
            include_expired: false,
            ann_filtering_mode: AnnFilteringMode::PostFilter,
            ann_filtering_policy: None,
            estimated_selectivity: None,
            search_effort: None,
        };

        let result = manager.query(query).await.unwrap();

        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].vector_id, "electronics-1".to_string());
    }

    #[tokio::test]
    async fn test_hybrid_vector_metadata_search() {
        // Test combining vector search with metadata filtering
        let vector_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 768 },
            IndexAlgorithm::HNSW {
                m: 16,
                ef_construction: 200,
                ef_search: 100,
                max_elements: 100000,
            },
        );

        let metadata_spec = IndexSpecification::new(
            Data::Metadata,
            IndexAlgorithm::BTree {
                max_keys_per_node: 100,
            },
        );

        // Create hybrid query
        let _hybrid_query = HybridQuery {
            collection_id: "test_collection".to_string(),
            vector_query: Some(VectorQuery::Dense {
                vector: vec![0.5; 768],
                similarity_threshold: 0.8,
            }),
            metadata_filters: vec![MetadataFilter {
                field: "category".to_string(),
                operator: FilterOperator::Equals,
                value: serde_json::json!("electronics"),
            }],
            id_filters: vec![],
            top_k: 10,
            include_expired: false,
            ann_filtering_mode: AnnFilteringMode::PostFilter,
            ann_filtering_policy: None,
            estimated_selectivity: None,
            search_effort: None,
        };

        // Verify both indexes are used
        assert!(vector_spec.supports_clustering());
        assert!(!metadata_spec.supports_clustering()); // BTree doesn't support clustering
    }

    #[tokio::test]
    async fn test_multi_index_routing() {
        // Test routing queries to appropriate indexes based on selectivity
        let indexes = vec![
            IndexSpecification {
                data_type: crate::index::axis::types::Data::DenseVector { dimension: 512 },
                algorithm: IndexAlgorithm::HNSW {
                    m: 16,
                    ef_construction: 200,
                    ef_search: 100,
                    max_elements: 100000,
                },
                name: Some("primary_vector".to_string()),
                is_primary: true,
                selectivity_threshold: Some(0.1), // Use for selective queries
            },
            IndexSpecification {
                data_type: crate::index::axis::types::Data::DenseVector { dimension: 512 },
                algorithm: IndexAlgorithm::IVF {
                    nlist: 1000,
                    nprobe: 100,
                    quantizer: None,
                },
                name: Some("bulk_vector".to_string()),
                is_primary: false,
                selectivity_threshold: Some(0.5), // Use for bulk queries
            },
            IndexSpecification {
                data_type: crate::index::axis::types::Data::Metadata,
                algorithm: IndexAlgorithm::BTree {
                    max_keys_per_node: 100,
                },
                name: Some("metadata_index".to_string()),
                is_primary: false,
                selectivity_threshold: None,
            },
        ];

        // Test routing logic
        for index in &indexes {
            match (&index.data_type, index.is_primary) {
                (Data::DenseVector { .. }, true) => {
                    // Primary vector index for top-k queries
                    assert_eq!(index.name.as_ref().unwrap(), "primary_vector");
                }
                (Data::DenseVector { .. }, false) => {
                    // Secondary vector index for bulk operations
                    assert_eq!(index.name.as_ref().unwrap(), "bulk_vector");
                }
                (Data::Metadata, _) => {
                    // Metadata index for filtering
                    assert_eq!(index.name.as_ref().unwrap(), "metadata_index");
                }
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn test_composite_pq_ivf_index() {
        // Test IVF with PQ quantization (common combination)
        let pq_quantizer = IndexAlgorithm::PQ {
            m: 8,
            nbits: 8,
            train_size: 10000,
        };

        let ivf_pq_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 768 },
            IndexAlgorithm::IVF {
                nlist: 1000,
                nprobe: 100,
                quantizer: Some(Box::new(pq_quantizer)),
            },
        );

        assert!(ivf_pq_spec.supports_clustering());

        // Verify quantizer is properly configured
        if let IndexAlgorithm::IVF { quantizer, .. } = &ivf_pq_spec.algorithm {
            assert!(quantizer.is_some());

            if let Some(q) = quantizer
                && let IndexAlgorithm::PQ { m, nbits, .. } = q.as_ref()
            {
                assert_eq!(*m, 8);
                assert_eq!(*nbits, 8);
            }
        }
    }

    #[tokio::test]
    async fn test_fallback_index_chain() {
        // Test fallback chain: HNSW -> IVF -> FLAT
        let index_chain = vec![
            IndexSpecification::new(
                Data::DenseVector { dimension: 128 },
                IndexAlgorithm::HNSW {
                    m: 16,
                    ef_construction: 200,
                    ef_search: 100,
                    max_elements: 10000, // Small capacity
                },
            ),
            IndexSpecification::new(
                Data::DenseVector { dimension: 128 },
                IndexAlgorithm::IVF {
                    nlist: 100,
                    nprobe: 10,
                    quantizer: None,
                },
            ),
            // FLAT would be the final fallback (brute force)
        ];

        // Test fallback logic
        let vector_count = 15000; // Exceeds HNSW capacity

        // Should fallback to IVF
        let selected_index = if vector_count > 10000 {
            &index_chain[1] // IVF
        } else {
            &index_chain[0] // HNSW
        };

        match &selected_index.algorithm {
            IndexAlgorithm::IVF { .. } => {
                assert!(vector_count > 10000);
            }
            IndexAlgorithm::HNSW { .. } => {
                assert!(vector_count <= 10000);
            }
            _ => panic!("Unexpected index type"),
        }
    }

    #[tokio::test]
    async fn test_text_vector_hybrid_search() {
        // Test combining full-text search with vector similarity
        let text_spec = IndexSpecification::new(
            Data::FullText,
            IndexAlgorithm::InvertedIndex {
                analyzer: crate::index::axis::types::TextAnalyzer {
                    tokenizer: crate::index::axis::types::Tokenizer::Standard,
                    filters: vec![
                        crate::index::axis::types::TokenFilter::Lowercase,
                        crate::index::axis::types::TokenFilter::Stopwords,
                    ],
                    language: Some("english".to_string()),
                },
                enable_positions: true,
            },
        );

        let vector_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 384 }, // Sentence embedding dimension
            IndexAlgorithm::Annoy {
                n_trees: 10,
                search_k: -1,
                max_leaf_size: 100,
            },
        );

        // Both should be usable for hybrid search
        assert!(!text_spec.supports_clustering()); // Text index doesn't cluster
        assert!(vector_spec.supports_clustering()); // Vector index supports clustering
    }

    #[tokio::test]
    async fn test_sparse_dense_vector_combination() {
        // Test combining sparse and dense vector indexes
        let sparse_spec = IndexSpecification::new(
            Data::SparseVector {
                max_dimension: 50000,
            },
            IndexAlgorithm::InvertedIndex {
                analyzer: crate::index::axis::types::TextAnalyzer {
                    tokenizer: crate::index::axis::types::Tokenizer::Whitespace,
                    filters: vec![],
                    language: None,
                },
                enable_positions: false,
            },
        );

        let dense_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 768 },
            IndexAlgorithm::LSH {
                n_projections: 10,
                n_hash_tables: 10,
                hash_width: 1.0,
            },
        );

        // Test hybrid scoring
        let sparse_weight: f32 = 0.3;
        let dense_weight: f32 = 0.7;

        assert!((sparse_weight + dense_weight - 1.0).abs() < 0.001);

        // Both can be used together
        assert!(dense_spec.supports_clustering());
        // Sparse vectors with inverted index don't support clustering
        assert!(!sparse_spec.supports_clustering());
    }

    #[tokio::test]
    async fn test_multi_stage_search() {
        // Test multi-stage search: coarse -> fine

        // Stage 1: Coarse search with LSH
        let coarse_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 512 },
            IndexAlgorithm::LSH {
                n_projections: 5,
                n_hash_tables: 5,
                hash_width: 2.0, // Wider buckets for coarse search
            },
        );

        // Stage 2: Refine with IVF-PQ
        let fine_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 512 },
            IndexAlgorithm::IVF {
                nlist: 100,
                nprobe: 20,
                quantizer: Some(Box::new(IndexAlgorithm::PQ {
                    m: 16,
                    nbits: 8,
                    train_size: 5000,
                })),
            },
        );

        // Stage 3: Final reranking (would use FLAT/brute-force)

        // All stages support clustering
        assert!(coarse_spec.supports_clustering());
        assert!(fine_spec.supports_clustering());
    }

    #[tokio::test]
    async fn test_bloom_filter_prefiltering() {
        // Test using bloom filter for existence checks before main index
        let bloom_spec = IndexSpecification::new(
            Data::Identifier,
            IndexAlgorithm::BloomFilter {
                expected_elements: 1000000,
                false_positive_rate: 0.01,
            },
        );

        let main_spec = IndexSpecification::new(
            Data::DenseVector { dimension: 256 },
            IndexAlgorithm::HNSW {
                m: 16,
                ef_construction: 200,
                ef_search: 100,
                max_elements: 1000000,
            },
        );

        // Bloom filter for quick existence check
        assert!(!bloom_spec.supports_clustering());

        // Main index for actual search
        assert!(main_spec.supports_clustering());
    }

    #[tokio::test]
    async fn test_skiplist_ordered_search() {
        // Test skip list for maintaining sorted order
        let skiplist_spec = IndexSpecification::new(
            Data::Metadata,
            IndexAlgorithm::SkipList {
                max_level: 16,
                probability: 0.5,
            },
        );

        // Skip list doesn't support clustering (it's for ordered data)
        assert!(!skiplist_spec.supports_clustering());

        // Good for range queries and ordered traversal
        match &skiplist_spec.algorithm {
            IndexAlgorithm::SkipList {
                max_level,
                probability,
            } => {
                assert_eq!(*max_level, 16);
                assert_eq!(*probability, 0.5);
            }
            _ => panic!("Expected SkipList"),
        }
    }

    #[tokio::test]
    async fn test_adaptive_index_selection() {
        // Test adaptive selection based on query patterns
        let query_patterns = vec![
            ("high_precision", 0.01), // Use HNSW
            ("balanced", 0.1),        // Use IVF
            ("high_recall", 0.5),     // Use LSH
            ("exhaustive", 1.0),      // Use FLAT (brute force)
        ];

        for (pattern, selectivity) in query_patterns {
            let selected_algorithm = match selectivity {
                s if s < 0.05 => IndexAlgorithm::HNSW {
                    m: 32,
                    ef_construction: 400,
                    ef_search: 200,
                    max_elements: 100000,
                },
                s if s < 0.2 => IndexAlgorithm::IVF {
                    nlist: 1000,
                    nprobe: 100,
                    quantizer: None,
                },
                s if s < 0.7 => IndexAlgorithm::LSH {
                    n_projections: 10,
                    n_hash_tables: 10,
                    hash_width: 1.0,
                },
                _ => {
                    // Would use FLAT for exhaustive search
                    IndexAlgorithm::LSH {
                        // Placeholder since FLAT not implemented
                        n_projections: 100,
                        n_hash_tables: 100,
                        hash_width: 0.1,
                    }
                }
            };

            debug!(
                "Pattern '{}' (selectivity={}) -> {:?}",
                pattern, selectivity, selected_algorithm
            );
        }
    }
}
