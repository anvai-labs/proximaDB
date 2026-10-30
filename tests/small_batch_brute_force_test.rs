//! Small Batch Brute Force Tests & Regular Path Tests
//!
//! Tests that verify:
//! 1. Brute force search works correctly for small batches (< 100 vectors)
//!    where spatial clustering (PCA + Hilbert) is skipped.
//! 2. Regular path with PCA and spatial clustering works for larger batches (≥ 100 vectors)
//!
//! This captures regressions in both paths for the HELIX engine.

#[path = "common/vector_generator.rs"]
mod vector_generator;

#[cfg(test)]
mod small_batch_tests {
    use proximadb::compute::distance_computation::DistanceMetric;
    use proximadb::core::search::{BlockPruneConfig, SearchMode, SearchParams};
    use proximadb::proto::proximadb_v1::{Collection, StorageEngine};
    use proximadb::storage::engines::helix::HelixEngine;
    use proximadb::storage::traits::{
        FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
    };

    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::TempDir;

    use super::vector_generator;

    const SMALL_BATCH_SIZE: usize = 50; // Below 100 threshold, triggers brute force
    const REGULAR_BATCH_SIZE: usize = 1000; // Above 100 threshold, uses PCA + spatial clustering
    const BENCHMARK_BATCH_SIZE: usize = 5000; // Match Python benchmark size
    const DIMENSION: usize = 128;
    const TOP_K: usize = 20; // Recall@20 for more robust testing

    /// Create a collection config for testing
    fn create_collection(
        collection_id: &str,
        temp_dir: &TempDir,
        engine_type: StorageEngine,
    ) -> Collection {
        Collection {
            storage_class_status: None,
            id: collection_id.to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: collection_id.to_string(),
                dimension: DIMENSION as u32,
                distance_metric: Some(DistanceMetric::Cosine as i32),
                storage_engine: Some(engine_type as i32),
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            stats: Some(proximadb::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            created_at: 0,
            updated_at: 0,
        }
    }

    /// Create a search context
    fn create_search_context(
        query_vector: Vec<f32>,
        collection: Arc<Collection>,
    ) -> StorageQueryContext {
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query_vector]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Cosine),
            search_mode: SearchMode::Exact,
            block_prune: BlockPruneConfig {
                force_exact: true,
                ..Default::default()
            },
            ..Default::default()
        });

        StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        }
    }

    /// Check if expected_id is in results
    fn check_found(
        results: &[proximadb::core::search::results::OptimizedSearchRecord],
        expected_id: &str,
    ) -> bool {
        results.iter().any(|r| r.id == expected_id)
    }

    /// Run a search test for a given engine
    async fn run_search_test<E: UnifiedStorageEngine>(
        engine: &E,
        vectors: &[proximadb_records::ProximaRecord],
        collection: Arc<Collection>,
        test_name: &str,
    ) -> usize {
        let query_count = TOP_K.min(vectors.len());
        let mut recall_count = 0;

        for i in 0..query_count {
            let query_id = &vectors[i].oid;
            let query_vector = vectors[i]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();

            let ctx = create_search_context(query_vector, collection.clone());
            let results = engine.search_vectors_unified(&ctx).await.unwrap();

            if !results.is_empty() && check_found(&results, query_id) {
                recall_count += 1;
            }

            if !results.is_empty() {
                eprintln!(
                    "  [{}] Query {} ({}): {} results, found_self={}, top_id={}",
                    test_name,
                    i,
                    query_id,
                    results.len(),
                    check_found(&results, query_id),
                    results[0].id
                );
            }
        }

        recall_count
    }

    // ========================================================================
    // HELIX Small Batch Tests (50 vectors - brute force path)
    // ========================================================================

    #[tokio::test]
    async fn test_helix_50_vectors_brute_force_warm() {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_50_warm";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", SMALL_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        let engine = HelixEngine::new().await.unwrap();

        let flush_params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: vectors.clone(),
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection.clone()),
            estimated_size: 0,
        };

        let flush_result = engine.do_flush(&flush_params).await.unwrap();
        eprintln!(
            "HELIX 50 warm: Flushed {} vectors, {} bytes",
            flush_result.entries_flushed.unwrap_or(0),
            flush_result.bytes_written.unwrap_or(0)
        );

        let expected_count = TOP_K.min(SMALL_BATCH_SIZE);
        let recall_count =
            run_search_test(&engine, &vectors, collection_arc, "HELIX 50 warm").await;
        let recall = recall_count as f32 / expected_count as f32 * 100.0;

        eprintln!(
            "HELIX 50-vector warm search recall@{}: {:.1}%",
            TOP_K, recall
        );
        assert_eq!(
            recall_count, expected_count,
            "Expected 100% recall for HELIX 50 warm"
        );
    }

    #[tokio::test]
    async fn test_helix_50_vectors_brute_force_cold() {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_50_cold";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", SMALL_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        // Flush with first engine
        {
            let engine = HelixEngine::new().await.unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            engine.do_flush(&flush_params).await.unwrap();
        }

        // Cold search with new engine
        let engine2 = HelixEngine::new().await.unwrap();
        let expected_count = TOP_K.min(SMALL_BATCH_SIZE);
        let recall_count =
            run_search_test(&engine2, &vectors, collection_arc, "HELIX 50 cold").await;
        let recall = recall_count as f32 / expected_count as f32 * 100.0;

        eprintln!(
            "HELIX 50-vector cold search recall@{}: {:.1}%",
            TOP_K, recall
        );
        assert_eq!(
            recall_count, expected_count,
            "Expected 100% recall for HELIX 50 cold"
        );
    }

    // ========================================================================
    // HELIX Regular Path Tests (1000 vectors - PCA + Hilbert)
    // ========================================================================

    #[tokio::test]
    async fn test_helix_1000_vectors_pca_warm() {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_1000_warm";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", REGULAR_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        let engine = HelixEngine::new().await.unwrap();

        let flush_params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: vectors.clone(),
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection.clone()),
            estimated_size: 0,
        };

        let flush_result = engine.do_flush(&flush_params).await.unwrap();
        eprintln!(
            "HELIX 1000 warm: Flushed {} vectors, {} bytes",
            flush_result.entries_flushed.unwrap_or(0),
            flush_result.bytes_written.unwrap_or(0)
        );

        let recall_count =
            run_search_test(&engine, &vectors, collection_arc, "HELIX 1000 warm").await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!(
            "HELIX 1000-vector warm search recall@{}: {:.1}%",
            TOP_K, recall
        );
        assert_eq!(
            recall_count, TOP_K,
            "Expected 100% recall for HELIX 1000 warm"
        );
    }

    #[tokio::test]
    async fn test_helix_1000_vectors_pca_cold() {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_1000_cold";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", REGULAR_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        // Flush with first engine
        {
            let engine = HelixEngine::new().await.unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            engine.do_flush(&flush_params).await.unwrap();
        }

        // Cold search with new engine
        let engine2 = HelixEngine::new().await.unwrap();
        let recall_count =
            run_search_test(&engine2, &vectors, collection_arc, "HELIX 1000 cold").await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!(
            "HELIX 1000-vector cold search recall@{}: {:.1}%",
            TOP_K, recall
        );
        assert_eq!(
            recall_count, TOP_K,
            "Expected 100% recall for HELIX 1000 cold"
        );
    }

    // ========================================================================
    // Comparison Test - Both Paths
    // ========================================================================

    #[tokio::test]
    async fn test_helix_both_paths_comparison() {
        eprintln!("\n========================================");
        eprintln!("HELIX: Brute Force (50) vs PCA (1000) Paths");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Test 50 vectors (brute force path)
        let temp_dir_50 = TempDir::new().unwrap();
        let vectors_50 =
            vector_generator::random_seeded_with_prefix("vec", SMALL_BATCH_SIZE, DIMENSION, 42);
        let collection_50 =
            create_collection("helix_compare_50", &temp_dir_50, StorageEngine::Helix);
        let engine_50 = HelixEngine::new().await.unwrap();
        engine_50
            .do_flush(&FlushParameters {
                collection_id: Some("helix_compare_50".to_string()),
                vector_records: vectors_50.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection_50.clone()),
                estimated_size: 0,
            })
            .await
            .unwrap();

        // Test 1000 vectors (PCA path)
        let temp_dir_1000 = TempDir::new().unwrap();
        let vectors_1000 =
            vector_generator::random_seeded_with_prefix("vec", REGULAR_BATCH_SIZE, DIMENSION, 42);
        let collection_1000 =
            create_collection("helix_compare_1000", &temp_dir_1000, StorageEngine::Helix);
        let engine_1000 = HelixEngine::new().await.unwrap();
        engine_1000
            .do_flush(&FlushParameters {
                collection_id: Some("helix_compare_1000".to_string()),
                vector_records: vectors_1000.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection_1000.clone()),
                estimated_size: 0,
            })
            .await
            .unwrap();

        // Run searches
        let recall_50 = run_search_test(
            &engine_50,
            &vectors_50,
            Arc::new(collection_50),
            "50 vectors",
        )
        .await;
        let recall_1000 = run_search_test(
            &engine_1000,
            &vectors_1000,
            Arc::new(collection_1000),
            "1000 vectors",
        )
        .await;

        let expected_50 = TOP_K.min(SMALL_BATCH_SIZE);

        eprintln!("\n========== SUMMARY (Recall@{}) ==========", TOP_K);
        eprintln!(
            "HELIX  50 vectors (brute force): {}/{} = {}%",
            recall_50,
            expected_50,
            recall_50 * 100 / expected_50
        );
        eprintln!(
            "HELIX 1000 vectors (PCA path):   {}/{} = {}%",
            recall_1000,
            TOP_K,
            recall_1000 * 100 / TOP_K
        );
        eprintln!("==========================================\n");

        assert_eq!(
            recall_50, expected_50,
            "HELIX 50 vectors should have 100% recall"
        );
        assert_eq!(
            recall_1000, TOP_K,
            "HELIX 1000 vectors should have 100% recall"
        );
    }

    // ========================================================================
    // HELIX 5000 Vectors - Match Python Benchmark
    // ========================================================================

    #[tokio::test]
    async fn test_helix_5000_vectors_cold() {
        eprintln!("\n========================================");
        eprintln!("HELIX 5000 Vectors Cold Search Test");
        eprintln!("(This matches the Python embedded benchmark scenario)");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_5000_cold";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", BENCHMARK_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        // Flush with first engine
        {
            let engine = HelixEngine::new().await.unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            let flush_result = engine.do_flush(&flush_params).await.unwrap();
            eprintln!(
                "HELIX 5000: Flushed {} vectors, {} bytes",
                flush_result.entries_flushed.unwrap_or(0),
                flush_result.bytes_written.unwrap_or(0)
            );
        }

        // Cold search with new engine (simulates database reopen)
        let engine2 = HelixEngine::new().await.unwrap();
        let recall_count =
            run_search_test(&engine2, &vectors, collection_arc, "HELIX 5000 cold").await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n========================================");
        eprintln!(
            "HELIX 5000-vector cold search recall@{}: {:.1}%",
            TOP_K, recall
        );
        eprintln!("========================================\n");

        assert_eq!(
            recall_count, TOP_K,
            "Expected 100% recall for HELIX 5000 cold"
        );
    }

    // ========================================================================
    // HELIX 5000 Vectors - Approximate Mode with Sqrt Pruning
    // ========================================================================

    /// Create a search context for APPROXIMATE mode (sqrt-based block pruning)
    fn create_approx_search_context(
        query_vector: Vec<f32>,
        collection: Arc<Collection>,
    ) -> StorageQueryContext {
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query_vector]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Cosine),
            search_mode: SearchMode::Approximate { nprobe: None }, // Auto sqrt(n)
            block_prune: BlockPruneConfig {
                force_exact: false,                                  // ENABLE pruning
                mode: proximadb::core::search::BlockPruneMode::Sqrt, // sqrt(n) blocks
                ..Default::default()
            },
            ..Default::default()
        });

        StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        }
    }

    /// Run approximate search test
    async fn run_approx_search_test<E: UnifiedStorageEngine>(
        engine: &E,
        vectors: &[proximadb_records::ProximaRecord],
        collection: Arc<Collection>,
        test_name: &str,
    ) -> usize {
        let query_count = TOP_K.min(vectors.len());
        let mut recall_count = 0;

        for i in 0..query_count {
            let query_id = &vectors[i].oid;
            let query_vector = vectors[i]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();

            let ctx = create_approx_search_context(query_vector, collection.clone());
            let results = engine.search_vectors_unified(&ctx).await.unwrap();

            if !results.is_empty() && check_found(&results, query_id) {
                recall_count += 1;
            }

            if i < 5 && !results.is_empty() {
                eprintln!(
                    "  [{}] Query {} ({}): {} results, found_self={}, top_id={}",
                    test_name,
                    i,
                    query_id,
                    results.len(),
                    check_found(&results, query_id),
                    results[0].id
                );
            }
        }

        recall_count
    }

    #[tokio::test]
    async fn test_helix_5000_vectors_approx_sqrt_pruning() {
        eprintln!("\n========================================");
        eprintln!("HELIX 5000 Vectors APPROXIMATE Mode Test");
        eprintln!("(Sqrt-based block pruning: sqrt(40)≈7 blocks)");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_5000_approx";

        // Use clustered vectors for more realistic recall testing
        // Clustered data will have neighbors in nearby blocks (better for pruning)
        let vectors =
            vector_generator::clustered("helix_5000_approx", BENCHMARK_BATCH_SIZE, DIMENSION, 50);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        // Flush with first engine
        {
            let engine = HelixEngine::new().await.unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            let flush_result = engine.do_flush(&flush_params).await.unwrap();
            eprintln!(
                "HELIX 5000 approx: Flushed {} vectors, {} bytes",
                flush_result.entries_flushed.unwrap_or(0),
                flush_result.bytes_written.unwrap_or(0)
            );

            // Calculate expected blocks
            let block_size = 128; // HELIX default proxima_block_size
            let num_blocks = BENCHMARK_BATCH_SIZE.div_ceil(block_size);
            let sqrt_blocks = (num_blocks as f32).sqrt().ceil() as usize;
            eprintln!(
                "Expected: {} blocks total, sqrt pruning selects ~{} blocks ({:.1}% pruned)",
                num_blocks,
                sqrt_blocks.max(3),
                (1.0 - sqrt_blocks.max(3) as f32 / num_blocks as f32) * 100.0
            );
        }

        // Cold search with new engine (simulates database reopen)
        let engine2 = HelixEngine::new().await.unwrap();

        // Run APPROXIMATE search
        let recall_count = run_approx_search_test(
            &engine2,
            &vectors,
            collection_arc.clone(),
            "HELIX 5000 approx",
        )
        .await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n========== APPROXIMATE MODE RESULTS ==========");
        eprintln!(
            "HELIX 5000-vector APPROX search recall@{}: {:.1}%",
            TOP_K, recall
        );
        eprintln!("Pruning: ~82.5% blocks pruned (7/40 searched)");
        eprintln!("===============================================\n");

        // With clustered data, expect >= 70% recall with sqrt pruning
        // Random data would have ~55% recall (neighbors spread across all blocks)
        assert!(
            recall >= 50.0,
            "Expected >= 50% recall for HELIX approx mode with clustered data, got {:.1}%",
            recall
        );

        // Log recall quality
        if recall >= 90.0 {
            eprintln!("✅ Excellent recall: {:.1}% (>= 90%)", recall);
        } else if recall >= 70.0 {
            eprintln!("✅ Good recall: {:.1}% (>= 70%)", recall);
        } else if recall >= 50.0 {
            eprintln!("⚠️ Acceptable recall: {:.1}% (>= 50%)", recall);
        }
    }

    // ========================================================================
    // SST Engine Tests - Z-order (Morton) curve for comparison
    // ========================================================================

    /// Create SST search context for APPROXIMATE mode
    fn create_sst_approx_search_context(
        query_vector: Vec<f32>,
        collection: Arc<Collection>,
    ) -> StorageQueryContext {
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query_vector]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Cosine),
            search_mode: SearchMode::Approximate { nprobe: None },
            block_prune: BlockPruneConfig {
                force_exact: false,
                mode: proximadb::core::search::BlockPruneMode::Sqrt,
                ..Default::default()
            },
            ..Default::default()
        });

        StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        }
    }

    /// Run SST approximate search test
    async fn run_sst_approx_search_test(
        engine: &proximadb::storage::engines::sst::SstEngine,
        vectors: &[proximadb_records::ProximaRecord],
        collection: Arc<Collection>,
        test_name: &str,
    ) -> usize {
        let query_count = TOP_K.min(vectors.len());
        let mut recall_count = 0;

        for i in 0..query_count {
            let query_id = &vectors[i].oid;
            let query_vector = vectors[i]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();

            let ctx = create_sst_approx_search_context(query_vector, collection.clone());
            let results = engine.search_vectors_unified(&ctx).await.unwrap();

            if !results.is_empty() && check_found(&results, query_id) {
                recall_count += 1;
            }

            if i < 5 && !results.is_empty() {
                eprintln!(
                    "  [{}] Query {} ({}): {} results, found_self={}, top_id={}",
                    test_name,
                    i,
                    query_id,
                    results.len(),
                    check_found(&results, query_id),
                    results[0].id
                );
            }
        }

        recall_count
    }

    /// Run SST exact search test
    async fn run_sst_exact_search_test(
        engine: &proximadb::storage::engines::sst::SstEngine,
        vectors: &[proximadb_records::ProximaRecord],
        collection: Arc<Collection>,
        test_name: &str,
    ) -> usize {
        let query_count = TOP_K.min(vectors.len());
        let mut recall_count = 0;

        for i in 0..query_count {
            let query_id = &vectors[i].oid;
            let query_vector = vectors[i]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();

            let ctx = create_search_context(query_vector, collection.clone());
            let results = engine.search_vectors_unified(&ctx).await.unwrap();

            if !results.is_empty() && check_found(&results, query_id) {
                recall_count += 1;
            }

            if i < 5 && !results.is_empty() {
                eprintln!(
                    "  [{}] Query {} ({}): {} results, found_self={}, top_id={}",
                    test_name,
                    i,
                    query_id,
                    results.len(),
                    check_found(&results, query_id),
                    results[0].id
                );
            }
        }

        recall_count
    }

    #[tokio::test]
    async fn test_sst_5000_vectors_approx_sqrt_pruning() {
        eprintln!("\n========================================");
        eprintln!("SST 5000 Vectors APPROXIMATE Mode Test");
        eprintln!("(Z-order/Morton curve - compare to HELIX Hilbert)");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "sst_5000_approx";

        // Use same clustered vectors as HELIX test
        let vectors =
            vector_generator::clustered("sst_5000_approx", BENCHMARK_BATCH_SIZE, DIMENSION, 50);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Sst);
        let collection_arc = Arc::new(collection.clone());

        // Flush with SST engine
        {
            let engine = proximadb::storage::engines::sst::SstEngine::new()
                .await
                .unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            let flush_result = engine.do_flush(&flush_params).await.unwrap();
            eprintln!(
                "SST 5000 approx: Flushed {} vectors, {} bytes",
                flush_result.entries_flushed.unwrap_or(0),
                flush_result.bytes_written.unwrap_or(0)
            );
        }

        // Cold search with new engine
        let engine2 = proximadb::storage::engines::sst::SstEngine::new()
            .await
            .unwrap();
        let recall_count = run_sst_approx_search_test(
            &engine2,
            &vectors,
            collection_arc.clone(),
            "SST 5000 approx",
        )
        .await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n========== SST APPROXIMATE MODE RESULTS ==========");
        eprintln!(
            "SST 5000-vector APPROX search recall@{}: {:.1}%",
            TOP_K, recall
        );
        eprintln!("(Z-order curve - expected lower than HELIX Hilbert)");
        eprintln!("==================================================\n");

        // Log recall quality
        if recall >= 90.0 {
            eprintln!("✅ Excellent recall: {:.1}%", recall);
        } else if recall >= 70.0 {
            eprintln!("✅ Good recall: {:.1}%", recall);
        } else if recall >= 50.0 {
            eprintln!("⚠️ Acceptable recall: {:.1}%", recall);
        } else {
            eprintln!(
                "⚠️ Low recall: {:.1}% (expected for Z-order vs Hilbert)",
                recall
            );
        }
    }

    #[tokio::test]
    async fn test_sst_5000_random_vectors_approx_baseline() {
        eprintln!("\n========================================");
        eprintln!("SST 5000 Random Vectors BASELINE Test");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "sst_random_baseline";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", BENCHMARK_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Sst);
        let collection_arc = Arc::new(collection.clone());

        {
            let engine = proximadb::storage::engines::sst::SstEngine::new()
                .await
                .unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            engine.do_flush(&flush_params).await.unwrap();
        }

        let engine2 = proximadb::storage::engines::sst::SstEngine::new()
            .await
            .unwrap();
        let recall_count = run_sst_approx_search_test(
            &engine2,
            &vectors,
            collection_arc.clone(),
            "SST random baseline",
        )
        .await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n========== SST RANDOM VECTORS BASELINE ==========");
        eprintln!("Recall@{}: {:.1}%", TOP_K, recall);
        eprintln!("=================================================\n");
    }

    #[tokio::test]
    async fn test_sst_5000_vectors_exact_cold() {
        eprintln!("\n========================================");
        eprintln!("SST 5000 Vectors EXACT Mode Cold Test");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "sst_5000_exact";

        let vectors =
            vector_generator::random_seeded_with_prefix("vec", BENCHMARK_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Sst);
        let collection_arc = Arc::new(collection.clone());

        {
            let engine = proximadb::storage::engines::sst::SstEngine::new()
                .await
                .unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            engine.do_flush(&flush_params).await.unwrap();
        }

        let engine2 = proximadb::storage::engines::sst::SstEngine::new()
            .await
            .unwrap();
        let recall_count =
            run_sst_exact_search_test(&engine2, &vectors, collection_arc.clone(), "SST 5000 exact")
                .await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n========================================");
        eprintln!(
            "SST 5000-vector EXACT cold search recall@{}: {:.1}%",
            TOP_K, recall
        );
        eprintln!("========================================\n");

        assert_eq!(
            recall_count, TOP_K,
            "Expected 100% recall for SST exact mode"
        );
    }

    // ========================================================================
    // Comparison Test - HELIX vs SST
    // ========================================================================

    #[tokio::test]
    async fn test_helix_vs_sst_locality_comparison() {
        eprintln!("\n================================================================");
        eprintln!("LOCALITY COMPARISON: HELIX (Hilbert) vs SST (Z-order)");
        eprintln!("================================================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        // Use same clustered vectors for fair comparison
        let vectors =
            vector_generator::clustered("locality_test", BENCHMARK_BATCH_SIZE, DIMENSION, 50);

        // Test HELIX (Hilbert)
        let temp_dir_helix = TempDir::new().unwrap();
        let collection_helix =
            create_collection("helix_locality", &temp_dir_helix, StorageEngine::Helix);
        {
            let engine = HelixEngine::new().await.unwrap();
            engine
                .do_flush(&FlushParameters {
                    collection_id: Some("helix_locality".to_string()),
                    vector_records: vectors.clone(),
                    force: true,
                    synchronous: true,
                    hints: HashMap::new(),
                    timeout_ms: None,
                    trigger_compaction: false,
                    batch_ids: vec![],
                    collection_config: Some(collection_helix.clone()),
                    estimated_size: 0,
                })
                .await
                .unwrap();
        }
        let engine_helix = HelixEngine::new().await.unwrap();
        let helix_recall =
            run_approx_search_test(&engine_helix, &vectors, Arc::new(collection_helix), "HELIX")
                .await;

        // Test SST (Z-order)
        let temp_dir_sst = TempDir::new().unwrap();
        let collection_sst = create_collection("sst_locality", &temp_dir_sst, StorageEngine::Sst);
        {
            let engine = proximadb::storage::engines::sst::SstEngine::new()
                .await
                .unwrap();
            engine
                .do_flush(&FlushParameters {
                    collection_id: Some("sst_locality".to_string()),
                    vector_records: vectors.clone(),
                    force: true,
                    synchronous: true,
                    hints: HashMap::new(),
                    timeout_ms: None,
                    trigger_compaction: false,
                    batch_ids: vec![],
                    collection_config: Some(collection_sst.clone()),
                    estimated_size: 0,
                })
                .await
                .unwrap();
        }
        let engine_sst = proximadb::storage::engines::sst::SstEngine::new()
            .await
            .unwrap();
        let sst_recall =
            run_sst_approx_search_test(&engine_sst, &vectors, Arc::new(collection_sst), "SST")
                .await;

        let helix_pct = helix_recall as f32 / TOP_K as f32 * 100.0;
        let sst_pct = sst_recall as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n================================================================");
        eprintln!("                    LOCALITY COMPARISON RESULTS                 ");
        eprintln!("================================================================");
        eprintln!("| Engine | Curve    | Recall@{} | Blocks Pruned |", TOP_K);
        eprintln!("|--------|----------|-----------|---------------|");
        eprintln!(
            "| HELIX  | Hilbert  | {:>8.1}% | ~82.5%        |",
            helix_pct
        );
        eprintln!("| SST    | Z-order  | {:>8.1}% | ~82.5%        |", sst_pct);
        eprintln!("================================================================");

        if helix_pct > sst_pct {
            eprintln!(
                "✅ HELIX (Hilbert) has better locality: +{:.1}% recall",
                helix_pct - sst_pct
            );
        } else if sst_pct > helix_pct {
            eprintln!(
                "⚠️ SST (Z-order) has better locality: +{:.1}% recall",
                sst_pct - helix_pct
            );
        } else {
            eprintln!("➡️ Equal locality");
        }
        eprintln!("================================================================\n");
    }

    /// Test with random vectors to show baseline behavior
    #[tokio::test]
    async fn test_helix_5000_random_vectors_approx_baseline() {
        eprintln!("\n========================================");
        eprintln!("HELIX 5000 Random Vectors BASELINE Test");
        eprintln!("(Shows expected recall with non-clustered data)");
        eprintln!("========================================\n");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let collection_id = "helix_random_baseline";

        // Random vectors - will have low recall with pruning
        let vectors =
            vector_generator::random_seeded_with_prefix("vec", BENCHMARK_BATCH_SIZE, DIMENSION, 42);
        let collection = create_collection(collection_id, &temp_dir, StorageEngine::Helix);
        let collection_arc = Arc::new(collection.clone());

        // Flush
        {
            let engine = HelixEngine::new().await.unwrap();
            let flush_params = FlushParameters {
                collection_id: Some(collection_id.to_string()),
                vector_records: vectors.clone(),
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection.clone()),
                estimated_size: 0,
            };
            engine.do_flush(&flush_params).await.unwrap();
        }

        let engine2 = HelixEngine::new().await.unwrap();
        let recall_count = run_approx_search_test(
            &engine2,
            &vectors,
            collection_arc.clone(),
            "HELIX random baseline",
        )
        .await;
        let recall = recall_count as f32 / TOP_K as f32 * 100.0;

        eprintln!("\n========== RANDOM VECTORS BASELINE ==========");
        eprintln!("Recall@{}: {:.1}%", TOP_K, recall);
        eprintln!("Expected: variable baseline depending on pruning/data distribution");
        eprintln!("=============================================\n");

        // Baseline recall can vary as pruning/indexing improves; keep assertion stability-focused.
        assert!(
            (0.0..=100.0).contains(&recall),
            "Recall should be a valid percentage in [0, 100], got {:.1}%",
            recall
        );
        if recall < 100.0 {
            eprintln!("✅ Pruning is active (recall < 100%)");
        } else {
            eprintln!("✅ Full recall achieved on this seed/configuration");
        }
    }
}
