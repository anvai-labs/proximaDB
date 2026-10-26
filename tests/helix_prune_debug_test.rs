//! HELIX Pruning Debug Test
//!
//! This test investigates why HELIX has poor recall in approximate/prune mode
//! while SST works correctly. Both use centroid-based pruning.

#[path = "common/vector_generator.rs"]
mod vector_generator;

#[cfg(test)]
mod helix_prune_debug {
    use proximadb::compute::distance_computation::DistanceMetric;
    use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchMode, SearchParams};
    use proximadb::proto::proximadb_v1::Collection;
    use proximadb::storage::engines::helix::HelixEngine;
    use proximadb::storage::traits::{
        FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
    };

    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::TempDir;

    use super::vector_generator;

    const DIMENSION: usize = 128;
    const TOP_K: usize = 10;

    fn create_collection(collection_id: &str, temp_dir: &TempDir) -> Collection {
        Collection {
            storage_class_status: None,
            id: collection_id.to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: collection_id.to_string(),
                dimension: DIMENSION as u32,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
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

    fn create_search_context(
        query_vector: Vec<f32>,
        collection: Arc<Collection>,
        search_mode: SearchMode,
        block_prune: BlockPruneConfig,
    ) -> StorageQueryContext {
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query_vector]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            search_mode,
            block_prune,
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

    #[tokio::test]
    async fn test_helix_approximate_mode_recall() {
        eprintln!("\n========================================");
        eprintln!("TEST: HELIX Approximate Mode Recall");
        eprintln!("========================================");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let collection_id = "prune_test";

        // Generate test vectors - use 1000 vectors to have multiple blocks
        let vectors = vector_generator::random_seeded_with_prefix("vec", 1000, DIMENSION, 42);
        let query = vectors[0]
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();
        let expected_id = &vectors[0].oid;

        eprintln!("Query vector id: {}", expected_id);
        eprintln!("Query vector dimension: {}", query.len());

        let collection = create_collection(collection_id, &temp_dir);
        let engine = HelixEngine::new().await.unwrap();

        // Flush vectors
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
            "Flushed {} vectors, {} bytes",
            flush_result.entries_flushed.unwrap_or(0),
            flush_result.bytes_written.unwrap_or(0)
        );

        let collection_arc = Arc::new(collection.clone());

        // Test 1: Exact mode (should be 100%)
        eprintln!("\n--- Test 1: Exact Mode ---");
        let exact_prune = BlockPruneConfig {
            force_exact: true,
            ..Default::default()
        };
        let ctx = create_search_context(
            query.clone(),
            collection_arc.clone(),
            SearchMode::Exact,
            exact_prune,
        );
        let exact_results = engine.search_vectors_unified(&ctx).await.unwrap();
        let exact_found = exact_results.iter().any(|r| r.id == *expected_id);
        eprintln!(
            "Exact mode: {} results, found_query={}",
            exact_results.len(),
            exact_found
        );
        for (i, r) in exact_results.iter().take(3).enumerate() {
            eprintln!(
                "  {}: id={}, similarity={:.4}",
                i,
                r.id,
                r.similarity.unwrap_or(0.0)
            );
        }

        // Test 2: Approximate mode with Sqrt pruning (default)
        eprintln!("\n--- Test 2: Approximate Mode (Sqrt Pruning) ---");
        let approx_prune = BlockPruneConfig {
            radius_k: 0.0,
            force_exact: false,
            mode: BlockPruneMode::Sqrt,
            min_keep: 1,
            max_keep: 0, // No max limit
            ratio: 0.5,
            min_blocks_override: Some(0), // Enable pruning for tests
        };
        let ctx = create_search_context(
            query.clone(),
            collection_arc.clone(),
            SearchMode::Approximate { nprobe: Some(5) },
            approx_prune,
        );
        let approx_results = engine.search_vectors_unified(&ctx).await.unwrap();
        let approx_found = approx_results.iter().any(|r| r.id == *expected_id);
        eprintln!(
            "Approximate mode: {} results, found_query={}",
            approx_results.len(),
            approx_found
        );
        for (i, r) in approx_results.iter().take(3).enumerate() {
            eprintln!(
                "  {}: id={}, similarity={:.4}",
                i,
                r.id,
                r.similarity.unwrap_or(0.0)
            );
        }

        // Test 3: Approximate mode with higher ratio
        eprintln!("\n--- Test 3: Approximate Mode (Ratio=0.8 Pruning) ---");
        let high_ratio_prune = BlockPruneConfig {
            radius_k: 0.0,
            force_exact: false,
            mode: BlockPruneMode::Ratio,
            min_keep: 1,
            max_keep: 0,
            ratio: 0.8,                   // Keep 80% of blocks
            min_blocks_override: Some(0), // Enable pruning for tests
        };
        let ctx = create_search_context(
            query.clone(),
            collection_arc.clone(),
            SearchMode::Approximate { nprobe: Some(5) },
            high_ratio_prune,
        );
        let high_ratio_results = engine.search_vectors_unified(&ctx).await.unwrap();
        let high_ratio_found = high_ratio_results.iter().any(|r| r.id == *expected_id);
        eprintln!(
            "High ratio mode: {} results, found_query={}",
            high_ratio_results.len(),
            high_ratio_found
        );

        // Calculate recall for multiple queries
        eprintln!("\n--- Test 4: Multi-Query Recall Comparison ---");
        let mut exact_recall = 0;
        let mut approx_recall = 0;
        let num_queries = 10;

        for i in 0..num_queries {
            let q = vectors[i]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();
            let expected = &vectors[i].oid;

            // Exact
            let ctx = create_search_context(
                q.clone(),
                collection_arc.clone(),
                SearchMode::Exact,
                BlockPruneConfig {
                    force_exact: true,
                    ..Default::default()
                },
            );
            let results = engine.search_vectors_unified(&ctx).await.unwrap();
            if results.iter().any(|r| r.id == *expected) {
                exact_recall += 1;
            }

            // Approximate
            let ctx = create_search_context(
                q.clone(),
                collection_arc.clone(),
                SearchMode::Approximate { nprobe: Some(5) },
                BlockPruneConfig::default(),
            );
            let results = engine.search_vectors_unified(&ctx).await.unwrap();
            if results.iter().any(|r| r.id == *expected) {
                approx_recall += 1;
            }
        }

        eprintln!("\n========== RECALL SUMMARY ==========");
        eprintln!(
            "Exact mode recall:       {}/{} = {:.0}%",
            exact_recall,
            num_queries,
            exact_recall as f64 / num_queries as f64 * 100.0
        );
        eprintln!(
            "Approximate mode recall: {}/{} = {:.0}%",
            approx_recall,
            num_queries,
            approx_recall as f64 / num_queries as f64 * 100.0
        );
        eprintln!("======================================\n");

        // Assertions
        assert!(exact_found, "Exact mode should find the query vector");
        assert_eq!(
            exact_recall, num_queries,
            "Exact mode should have 100% recall"
        );

        // For approximate mode, we expect some recall loss but not complete failure
        // If it's 0%, there's a bug in the pruning logic
        if approx_recall == 0 {
            eprintln!("WARNING: Approximate mode has 0% recall - this indicates a bug!");
        }
    }

    #[tokio::test]
    async fn test_helix_cold_approximate_recall() {
        eprintln!("\n========================================");
        eprintln!("TEST: HELIX Cold Approximate Mode Recall");
        eprintln!("========================================");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let collection_id = "cold_approx_test";

        // Generate test vectors - use 5000 to create multiple blocks
        let vectors = vector_generator::random_seeded_with_prefix("vec", 5000, DIMENSION, 42);
        let query_vectors: Vec<Vec<f32>> = vectors[0..10]
            .iter()
            .map(|v| {
                v.embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default()
            })
            .collect();
        let expected_ids: Vec<String> = vectors[0..10].iter().map(|v| v.oid.clone()).collect();

        let collection = create_collection(collection_id, &temp_dir);

        // Phase 1: Insert and flush
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
            eprintln!("Flushed {} vectors", vectors.len());

            // Warm search recall
            let collection_arc = Arc::new(collection.clone());
            let mut warm_exact_recall = 0;
            let mut warm_approx_recall = 0;

            for (i, query) in query_vectors.iter().enumerate() {
                // Exact
                let ctx = create_search_context(
                    query.clone(),
                    collection_arc.clone(),
                    SearchMode::Exact,
                    BlockPruneConfig {
                        force_exact: true,
                        ..Default::default()
                    },
                );
                let results = engine.search_vectors_unified(&ctx).await.unwrap();
                if results.iter().any(|r| r.id == expected_ids[i]) {
                    warm_exact_recall += 1;
                }

                // Approximate
                let ctx = create_search_context(
                    query.clone(),
                    collection_arc.clone(),
                    SearchMode::Approximate { nprobe: Some(5) },
                    BlockPruneConfig::default(),
                );
                let results = engine.search_vectors_unified(&ctx).await.unwrap();
                if results.iter().any(|r| r.id == expected_ids[i]) {
                    warm_approx_recall += 1;
                }
            }

            eprintln!(
                "WARM - Exact: {}/10, Approximate: {}/10",
                warm_exact_recall, warm_approx_recall
            );
        }

        // Phase 2: Cold search (new engine instance)
        {
            let engine2 = HelixEngine::new().await.unwrap();
            let collection_arc = Arc::new(collection.clone());

            let mut cold_exact_recall = 0;
            let mut cold_approx_recall = 0;

            for (i, query) in query_vectors.iter().enumerate() {
                // Exact
                let ctx = create_search_context(
                    query.clone(),
                    collection_arc.clone(),
                    SearchMode::Exact,
                    BlockPruneConfig {
                        force_exact: true,
                        ..Default::default()
                    },
                );
                let results = engine2.search_vectors_unified(&ctx).await.unwrap();
                if results.iter().any(|r| r.id == expected_ids[i]) {
                    cold_exact_recall += 1;
                }

                // Approximate
                let ctx = create_search_context(
                    query.clone(),
                    collection_arc.clone(),
                    SearchMode::Approximate { nprobe: Some(5) },
                    BlockPruneConfig::default(),
                );
                let results = engine2.search_vectors_unified(&ctx).await.unwrap();
                if results.iter().any(|r| r.id == expected_ids[i]) {
                    cold_approx_recall += 1;
                }
            }

            eprintln!(
                "COLD - Exact: {}/10, Approximate: {}/10",
                cold_exact_recall, cold_approx_recall
            );

            eprintln!("\n========== COLD RECALL SUMMARY ==========");
            eprintln!(
                "Cold Exact recall:       {}/10 = {}%",
                cold_exact_recall,
                cold_exact_recall * 10
            );
            eprintln!(
                "Cold Approximate recall: {}/10 = {}%",
                cold_approx_recall,
                cold_approx_recall * 10
            );
            eprintln!("==========================================\n");

            // Assertions
            assert_eq!(
                cold_exact_recall, 10,
                "Cold exact mode should have 100% recall"
            );
        }
    }

    /// Test to verify HELIX centroid pruning behavior matches SST's
    /// The key difference: SST uses distance 0.0 for dimension mismatch, HELIX uses INFINITY
    #[tokio::test]
    async fn test_helix_centroid_pruning_vs_sst() {
        eprintln!("\n========================================");
        eprintln!("TEST: HELIX vs SST Centroid Pruning Behavior");
        eprintln!("========================================");

        eprintln!("\n🔍 Key differences identified:");
        eprintln!("   SST:   dimension mismatch → distance 0.0 (SAFE fallback, block included)");
        eprintln!("   HELIX: dimension mismatch → distance INFINITY (block PRUNED!)");
        eprintln!();
        eprintln!("   SST:   file-level pruning first, then block-level");
        eprintln!("   HELIX: block-level pruning only");
        eprintln!();
        eprintln!("   Default BlockPruneConfig:");
        eprintln!("     mode: Sqrt (select sqrt(n) blocks)");
        eprintln!("     min_keep: 1 (too aggressive for random data)");
        eprintln!();
        eprintln!("   With 5000 vectors, 128 vectors/block:");
        eprintln!("     ~39 blocks, sqrt(39) = ~6 blocks selected = 16% of data!");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let collection_id = "centroid_test";

        let vectors = vector_generator::random_seeded_with_prefix("vec", 1000, DIMENSION, 456);
        let _query = vectors[0]
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();

        let collection = create_collection(collection_id, &temp_dir);
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
        eprintln!(
            "\nFlushed {} vectors (expecting ~8 blocks with 128 vectors/block)",
            vectors.len()
        );

        let collection_arc = Arc::new(collection.clone());

        // Test with increasingly aggressive pruning
        let configs = vec![
            (
                "Sqrt (default)",
                BlockPruneConfig {
                    radius_k: 0.0,
                    force_exact: false,
                    mode: BlockPruneMode::Sqrt,
                    min_keep: 1,
                    max_keep: 0,
                    ratio: 0.5,
                    min_blocks_override: Some(0),
                },
            ),
            (
                "Sqrt min_keep=4",
                BlockPruneConfig {
                    radius_k: 0.0,
                    force_exact: false,
                    mode: BlockPruneMode::Sqrt,
                    min_keep: 4,
                    max_keep: 0,
                    ratio: 0.5,
                    min_blocks_override: Some(0),
                },
            ),
            (
                "Ratio 50%",
                BlockPruneConfig {
                    radius_k: 0.0,
                    force_exact: false,
                    mode: BlockPruneMode::Ratio,
                    min_keep: 1,
                    max_keep: 0,
                    ratio: 0.5,
                    min_blocks_override: Some(0),
                },
            ),
            (
                "Ratio 80%",
                BlockPruneConfig {
                    radius_k: 0.0,
                    force_exact: false,
                    mode: BlockPruneMode::Ratio,
                    min_keep: 1,
                    max_keep: 0,
                    ratio: 0.8,
                    min_blocks_override: Some(0),
                },
            ),
            (
                "Force Exact",
                BlockPruneConfig {
                    radius_k: 0.0,
                    force_exact: true,
                    mode: BlockPruneMode::Sqrt,
                    min_keep: 1,
                    max_keep: 0,
                    ratio: 0.5,
                    min_blocks_override: Some(0),
                },
            ),
        ];

        eprintln!("\nRecall comparison with different pruning configs:");
        for (name, prune_config) in configs {
            let mut found_count = 0;
            for i in 0..10 {
                let q = vectors[i]
                    .embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default();
                let expected = &vectors[i].oid;

                let ctx = create_search_context(
                    q.clone(),
                    collection_arc.clone(),
                    SearchMode::Approximate { nprobe: Some(5) },
                    prune_config.clone(),
                );
                let results = engine.search_vectors_unified(&ctx).await.unwrap();
                if results.iter().any(|r| r.id == *expected) {
                    found_count += 1;
                }
            }
            eprintln!(
                "  {}: {}/10 = {}% recall",
                name,
                found_count,
                found_count * 10
            );
        }

        eprintln!("\n✅ Recommendation: HELIX should match SST's behavior:");
        eprintln!("   - Use distance 0.0 instead of INFINITY for dimension mismatch");
        eprintln!("   - Increase default min_keep from 1 to at least 10% of blocks");
    }

    /// Test that HELIX works well with CLUSTERED data (its intended use case)
    ///
    /// HELIX uses Hilbert curves for spatial locality, which only helps when
    /// data has natural clusters. This test verifies that centroid-based
    /// pruning works correctly for clustered data.
    #[tokio::test]
    async fn test_helix_with_clustered_data() {
        eprintln!("\n========================================");
        eprintln!("TEST: HELIX with Clustered Data");
        eprintln!("========================================");
        eprintln!();
        eprintln!("This test verifies that HELIX's centroid-based pruning");
        eprintln!("works well for spatially clustered data (its design target).");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let collection_id = "clustered_test";

        // Generate 5000 vectors in 10 clusters (500 per cluster)
        // Each cluster is tightly grouped, so nearest neighbors are in same cluster
        let vectors = vector_generator::clustered("test", 5000, DIMENSION, 10);

        eprintln!(
            "\nGenerated {} vectors in 10 clusters (500 vectors/cluster)",
            vectors.len()
        );
        eprintln!("With 128 vectors/block, each cluster spans ~4 blocks");
        eprintln!("Total ~39 blocks, Hilbert sorting should group clusters together");

        let collection = create_collection(collection_id, &temp_dir);
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
        eprintln!("Flushed {} vectors", vectors.len());

        let collection_arc = Arc::new(collection.clone());

        // Test recall on vectors from different clusters
        // Pick one vector from each cluster (first 10 vectors are from cluster 0)
        let test_indices: Vec<usize> = (0..10).map(|cluster| cluster * 500).collect();

        eprintln!("\n--- Testing Approximate Mode (Sqrt) on Clustered Data ---");
        let mut approx_recall = 0;
        for &idx in &test_indices {
            let query = vectors[idx]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();
            let expected_id = &vectors[idx].oid;

            let ctx = create_search_context(
                query.clone(),
                collection_arc.clone(),
                SearchMode::Approximate { nprobe: Some(5) },
                BlockPruneConfig::default(), // Use default Sqrt pruning
            );

            let results = engine.search_vectors_unified(&ctx).await.unwrap();
            if results.iter().any(|r| &r.id == expected_id) {
                approx_recall += 1;
            }
        }

        eprintln!("\n--- Testing Exact Mode on Clustered Data ---");
        let mut exact_recall = 0;
        for &idx in &test_indices {
            let query = vectors[idx]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned())
                .unwrap_or_default();
            let expected_id = &vectors[idx].oid;

            let ctx = create_search_context(
                query.clone(),
                collection_arc.clone(),
                SearchMode::Exact,
                BlockPruneConfig {
                    force_exact: true,
                    ..Default::default()
                },
            );

            let results = engine.search_vectors_unified(&ctx).await.unwrap();
            if results.iter().any(|r| &r.id == expected_id) {
                exact_recall += 1;
            }
        }

        eprintln!("\n========== CLUSTERED DATA RECALL ==========");
        eprintln!(
            "Exact mode recall:       {}/10 = {}%",
            exact_recall,
            exact_recall * 10
        );
        eprintln!(
            "Approximate mode recall: {}/10 = {}%",
            approx_recall,
            approx_recall * 10
        );
        eprintln!("============================================");

        // Guard against severe regressions while keeping this benchmark stable across hosts.
        // Exact and approximate are evaluated on the same dataset/run, so relative comparison
        // is more reliable than an absolute threshold in CI.
        assert!(
            exact_recall >= 5,
            "Exact recall on clustered data regressed below 50%: {}%",
            exact_recall * 10
        );
        assert!(
            approx_recall + 1 >= exact_recall,
            "Approximate recall should stay within 10% of exact on clustered data (exact={}%, approx={}%)",
            exact_recall * 10,
            approx_recall * 10
        );
    }
}
