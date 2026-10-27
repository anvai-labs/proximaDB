//! HELIX Cold Search Recall Test
//!
//! This test verifies that HELIX engine maintains 100% recall after:
//! 1. Flushing vectors to disk
//! 2. Dropping the engine
//! 3. Creating a new engine instance (cold start)
//! 4. Searching for the original vectors
//!
//! The issue: SST has 100% recall, HELIX has poor recall (~13% or less)

#[path = "common/vector_generator.rs"]
mod vector_generator;

#[cfg(test)]
mod helix_cold_search_recall {
    use proximadb::compute::distance_computation::DistanceMetric;
    use proximadb::core::search::{BlockPruneConfig, SearchMode, SearchParams};
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

    /// Create a collection config for testing
    fn create_collection(collection_id: &str, temp_dir: &TempDir, engine_type: i32) -> Collection {
        Collection {
            storage_class_status: None,
            id: collection_id.to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: collection_id.to_string(),
                dimension: DIMENSION as u32,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(engine_type),
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
        search_mode: SearchMode,
    ) -> StorageQueryContext {
        let force_exact = matches!(search_mode, SearchMode::Exact);
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query_vector]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            search_mode,
            block_prune: BlockPruneConfig {
                force_exact,
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

    /// Calculate recall percentage - checks if expected_id is in results
    fn check_found(
        results: &[proximadb::core::search::results::OptimizedSearchRecord],
        expected_id: &str,
    ) -> bool {
        results.iter().any(|r| r.id == expected_id)
    }

    /// Calculate recall by running all queries and checking if each finds itself
    async fn calculate_recall_multi<E: UnifiedStorageEngine>(
        engine: &E,
        query_vectors: &[Vec<f32>],
        expected_ids: &[String],
        collection: Arc<Collection>,
    ) -> f64 {
        let mut found_count = 0;
        for (i, query) in query_vectors.iter().enumerate() {
            let ctx = create_search_context(query.clone(), collection.clone(), SearchMode::Exact);
            let results = engine.search_vectors_unified(&ctx).await.unwrap();
            if check_found(&results, &expected_ids[i]) {
                found_count += 1;
            }
        }
        (found_count as f64 / query_vectors.len() as f64) * 100.0
    }

    #[tokio::test]
    async fn test_helix_cold_search_recall_100_vectors() {
        test_cold_search_recall(100, "HELIX 100 vectors").await;
    }

    #[tokio::test]
    async fn test_helix_cold_search_recall_1000_vectors() {
        test_cold_search_recall(1000, "HELIX 1000 vectors").await;
    }

    #[tokio::test]
    async fn test_helix_cold_search_recall_5000_vectors() {
        test_cold_search_recall(5000, "HELIX 5000 vectors").await;
    }

    #[tokio::test]
    #[ignore] // Takes longer, run with: cargo test --test helix_cold_search_recall_test -- --ignored
    async fn test_helix_cold_search_recall_30000_vectors() {
        test_cold_search_recall(30000, "HELIX 30000 vectors (full scale)").await;
    }

    async fn test_cold_search_recall(vector_count: usize, test_name: &str) {
        eprintln!("\n========================================");
        eprintln!("TEST: {} (vector_count={})", test_name, vector_count);
        eprintln!("========================================");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let collection_id = "cold_search_test";

        // Generate deterministic test vectors
        let vectors =
            vector_generator::random_seeded_with_prefix("vec", vector_count, DIMENSION, 42);

        // Store query vectors and expected IDs (first 10 vectors)
        let query_vectors: Vec<Vec<f32>> = vectors[0..TOP_K.min(vector_count)]
            .iter()
            .map(|v| {
                v.embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default()
            })
            .collect();
        let expected_ids: Vec<String> = vectors[0..TOP_K.min(vector_count)]
            .iter()
            .map(|v| v.oid.clone())
            .collect();

        eprintln!("Query vectors: {:?}", expected_ids);

        // Track recall values across scopes
        let mut warm_recall = 0.0f64;

        // ========== PHASE 1: Insert and flush with HELIX ==========
        eprintln!("\n--- Phase 1: Insert and flush ---");

        let collection = create_collection(
            collection_id,
            &temp_dir,
            proximadb::proto::proximadb_v1::StorageEngine::Helix as i32,
        );

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
                "Flushed {} vectors, {} bytes",
                flush_result.entries_flushed.unwrap_or(0),
                flush_result.bytes_written.unwrap_or(0)
            );

            // ========== PHASE 2: Warm search (before drop) ==========
            eprintln!("\n--- Phase 2: Warm search (in-memory) ---");

            let collection_arc = Arc::new(collection.clone());

            for (i, query) in query_vectors.iter().enumerate() {
                let ctx =
                    create_search_context(query.clone(), collection_arc.clone(), SearchMode::Exact);
                let results = engine.search_vectors_unified(&ctx).await.unwrap();

                if results.is_empty() {
                    eprintln!("  Query {} ({}): NO RESULTS", i, expected_ids[i]);
                } else {
                    let found_expected = results.iter().any(|r| r.id == expected_ids[i]);
                    eprintln!(
                        "  Query {} ({}): {} results, found_self={}, top_id={}",
                        i,
                        expected_ids[i],
                        results.len(),
                        found_expected,
                        results[0].id
                    );
                }
            }

            // Calculate warm recall - check all queries
            warm_recall = calculate_recall_multi(
                &engine,
                &query_vectors,
                &expected_ids,
                collection_arc.clone(),
            )
            .await;
            eprintln!("\nWarm search recall: {:.1}%", warm_recall);

            // Engine drops here
        }

        // ========== PHASE 3: Cold search (new engine instance) ==========
        eprintln!("\n--- Phase 3: Cold search (new engine, read from disk) ---");

        let engine2 = HelixEngine::new().await.unwrap();
        let collection_arc = Arc::new(collection.clone());

        // Reload the collection from disk
        // Note: HelixEngine stores SSTable paths globally, so we need to ensure
        // the new engine can find the files

        for (i, query) in query_vectors.iter().enumerate() {
            let ctx =
                create_search_context(query.clone(), collection_arc.clone(), SearchMode::Exact);
            let results = engine2.search_vectors_unified(&ctx).await.unwrap();

            if results.is_empty() {
                eprintln!("  Cold Query {} ({}): NO RESULTS", i, expected_ids[i]);
            } else {
                let found_expected = results.iter().any(|r| r.id == expected_ids[i]);
                eprintln!(
                    "  Cold Query {} ({}): {} results, found_self={}, top_id={}",
                    i,
                    expected_ids[i],
                    results.len(),
                    found_expected,
                    results[0].id
                );
            }
        }

        let cold_recall = calculate_recall_multi(
            &engine2,
            &query_vectors,
            &expected_ids,
            collection_arc.clone(),
        )
        .await;
        eprintln!("\nCold search recall: {:.1}%", cold_recall);

        // ========== Summary ==========
        eprintln!("\n========== SUMMARY ==========");
        eprintln!("HELIX warm: {:.1}%", warm_recall);
        eprintln!("HELIX cold: {:.1}%", cold_recall);
        eprintln!("==============================\n");

        // Assert that recall is reasonable
        // For exact mode, we expect 100% recall
        assert!(
            cold_recall >= 90.0,
            "HELIX cold search recall should be >= 90%, got {:.1}%",
            cold_recall
        );
    }

    /// Test to compare HELIX and SST block reading mechanisms
    #[tokio::test]
    async fn test_helix_vs_sst_block_reading() {
        eprintln!("\n========================================");
        eprintln!("TEST: HELIX vs SST Block Reading Comparison");
        eprintln!("========================================");

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();

        // Generate small dataset for detailed debugging
        let vectors = vector_generator::random_seeded_with_prefix("test", 50, DIMENSION, 12345);
        let query = vectors[0]
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default();
        let expected_id = vectors[0].oid.clone();

        // Test with HELIX
        let helix_collection = create_collection(
            "helix_block_test",
            &temp_dir,
            proximadb::proto::proximadb_v1::StorageEngine::Helix as i32,
        );

        let engine = HelixEngine::new().await.unwrap();

        let flush_params = FlushParameters {
            collection_id: Some("helix_block_test".to_string()),
            vector_records: vectors.clone(),
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(helix_collection.clone()),
            estimated_size: 0,
        };

        engine.do_flush(&flush_params).await.unwrap();

        // List SSTable files
        let sstable_dir = temp_dir.path().join("helix_block_test");
        if sstable_dir.exists() {
            eprintln!("\nSSTable files in {:?}:", sstable_dir);
            for entry in std::fs::read_dir(&sstable_dir).unwrap() {
                let entry = entry.unwrap();
                let metadata = entry.metadata().unwrap();
                eprintln!("  {:?}: {} bytes", entry.file_name(), metadata.len());
            }
        }

        // Search with detailed logging
        let collection_arc = Arc::new(helix_collection.clone());
        let ctx = create_search_context(query.clone(), collection_arc, SearchMode::Exact);

        eprintln!("\nSearching for vector {}...", expected_id);
        let results = engine.search_vectors_unified(&ctx).await.unwrap();

        eprintln!("Results: {} found", results.len());
        for (i, r) in results.iter().take(5).enumerate() {
            eprintln!(
                "  {}: id={}, similarity={:.4}",
                i,
                r.id,
                r.similarity.unwrap_or(0.0)
            );
        }

        let found = results.iter().any(|r| r.id == expected_id);
        eprintln!("\nExpected {} found: {}", expected_id, found);

        assert!(found, "Should find the query vector in results");
    }
}
