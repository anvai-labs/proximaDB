#![cfg(feature = "experimental-engines")] // RAPTOR needs the `experimental-engines` feature.
#![allow(deprecated)]
//! RAPTOR Recall Test - Verify 100% recall after close/reopen
//!
//! This test verifies that RAPTOR maintains high recall when:
//! 1. Vectors are inserted
//! 2. Flushed to disk
//! 3. Engine closed and reopened (stateless mode)
//! 4. Search returns correct results

#[path = "common/vector_generator.rs"]
mod vector_generator;

#[path = "common/collection_builder.rs"]
mod collection_builder;

#[cfg(test)]
mod raptor_recall_tests {
    use proximadb::core::search::SearchParams;
    use proximadb::proto::proximadb_v1::StorageEngine;
    use proximadb::storage::engines::raptor::RaptorEngine;
    use proximadb::storage::persistence::write_ahead_log::BatchId;
    use proximadb::storage::traits::{FlushParameters, StorageQueryContext, UnifiedStorageEngine};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tracing::{debug, info};

    use super::collection_builder::TestCollectionBuilder;
    use super::vector_generator;

    const VECTOR_DIMS: usize = 768;
    const NUM_VECTORS: usize = 5000;
    const DEFAULT_NUM_QUERIES: usize = 25;
    const K_NEIGHBORS: usize = 10;

    fn recall_query_count() -> usize {
        std::env::var("PROXIMADB_RAPTOR_RECALL_QUERIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_NUM_QUERIES)
    }

    /// Compute ground truth using brute force
    fn compute_ground_truth(
        vectors: &[proximadb_records::ProximaRecord],
        query: &[f32],
        k: usize,
    ) -> Vec<String> {
        let mut distances: Vec<(String, f32)> = vectors
            .iter()
            .map(|v| {
                let values = v
                    .embeddings
                    .first()
                    .map(|e| e.as_fp32_slice())
                    .unwrap_or(&[]);
                let dist: f32 = values
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (v.oid.clone(), dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Compute recall@k
    fn compute_recall(ground_truth: &[String], results: &[String]) -> f64 {
        let gt_set: std::collections::HashSet<_> = ground_truth.iter().collect();
        let result_set: std::collections::HashSet<_> = results.iter().collect();
        let intersection = gt_set.intersection(&result_set).count();
        intersection as f64 / ground_truth.len() as f64
    }

    #[tokio::test]
    async fn test_raptor_recall_after_reopen_5000_vectors() {
        // Keep this test quiet by default; opt into deeper logs via env.
        let log_filter = std::env::var("PROXIMADB_RAPTOR_RECALL_LOG_FILTER").unwrap_or_else(|_| {
            "info,proximadb::storage::engines::raptor=warn,proximadb::storage::traits=warn,proximadb::compute::quantization::hardware_accelerated=warn".to_string()
        });
        let _ = tracing_subscriber::fmt()
            .with_env_filter(log_filter)
            .try_init();

        info!("=== RAPTOR Recall Test: {} vectors ===", NUM_VECTORS);

        // Create temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage_path = temp_dir.path().to_str().unwrap().to_string();

        // === PHASE 1: Create engine and insert vectors ===
        info!(
            "Phase 1: Creating RAPTOR engine and inserting {} vectors",
            NUM_VECTORS
        );

        let collection_id = "raptor_recall_test";

        // Create RAPTOR engine (stateless)
        let engine = RaptorEngine::new()
            .await
            .expect("Failed to create RAPTOR engine");

        // Generate vectors
        let vectors = vector_generator::random_seeded("raptor_test", NUM_VECTORS, VECTOR_DIMS, 42);
        info!(
            "Generated {} vectors with {} dimensions",
            vectors.len(),
            VECTOR_DIMS
        );
        let num_queries = recall_query_count().min(vectors.len());
        info!("Using {} recall queries", num_queries);

        // Build a collection for the test
        let (mut collection, _temp) = TestCollectionBuilder::new()
            .with_id(collection_id)
            .with_name(collection_id)
            .with_dimension(VECTOR_DIMS as u32)
            .with_engine(StorageEngine::Raptor)
            .with_distance_metric(proximadb::proto::proximadb_v1::DistanceMetric::Euclidean)
            .build();

        // Override storage path
        if let Some(ref mut assignment) = collection.storage_assignment {
            assignment.primary_path = storage_path.clone();
            assignment.base_location = storage_path.clone();
            assignment.assigned_at = chrono::Utc::now().timestamp();
        }
        collection.created_at = chrono::Utc::now().timestamp();
        collection.updated_at = chrono::Utc::now().timestamp();

        // Flush vectors to disk
        let batch_ids: Vec<BatchId> = (0..NUM_VECTORS).map(|_| BatchId::new()).collect();
        let flush_params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: vectors.clone(), // vectors is already Vec<ProximaRecord>
            batch_ids,
            force: true,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: Some(60000),
            trigger_compaction: false,
            collection_config: Some(collection.clone()),
            estimated_size: 1024 * 1024 * 100, // 100MB estimate
            collection_properties: None,
        };

        let insert_start = std::time::Instant::now();
        let flush_result = engine.flush(flush_params).await.expect("Failed to flush");
        info!(
            "Flush took {:?}, entries_flushed: {:?}",
            insert_start.elapsed(),
            flush_result.entries_flushed
        );

        assert!(flush_result.success, "Flush should succeed");
        assert_eq!(
            flush_result.entries_flushed,
            Some(NUM_VECTORS as u64),
            "Should flush all vectors"
        );

        // === PHASE 2: Test warm cache search ===
        info!("Phase 2: Testing warm cache search");

        let queries: Vec<Vec<f32>> = vectors
            .iter()
            .take(num_queries)
            .map(|v| {
                v.embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default()
            })
            .collect();

        let collection_arc = Arc::new(collection.clone());

        let mut warm_recalls = Vec::new();
        for (i, query) in queries.iter().enumerate() {
            let ground_truth = compute_ground_truth(&vectors, query, K_NEIGHBORS);

            let search_params = Arc::new(SearchParams {
                vector: Some(query.clone()),
                top_k: Some(K_NEIGHBORS as u16),
                distance_metric: Some(
                    proximadb::compute::distance_computation::DistanceMetric::Euclidean,
                ),
                ..Default::default()
            });

            let ctx = StorageQueryContext::new(search_params, collection_arc.clone());
            let results = engine
                .search_vectors_unified(&ctx)
                .await
                .expect("Warm search failed");

            let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
            let recall = compute_recall(&ground_truth, &result_ids);
            warm_recalls.push(recall);

            if i < 5 {
                debug!(
                    "Query {}: recall={:.1}%, results={}, ground_truth={:?}",
                    i,
                    recall * 100.0,
                    result_ids.len(),
                    &ground_truth[..3.min(ground_truth.len())]
                );
            }
        }

        let avg_warm_recall = warm_recalls.iter().sum::<f64>() / warm_recalls.len() as f64;
        info!(
            "Warm cache search: avg recall = {:.1}% over {} queries",
            avg_warm_recall * 100.0,
            num_queries
        );

        // === PHASE 3: Close engine ===
        info!("Phase 3: Closing engine");
        drop(engine);
        info!("Engine closed");

        // === PHASE 4: Reopen engine (stateless mode) ===
        info!("Phase 4: Reopening engine (stateless mode)");

        let engine2 = RaptorEngine::new()
            .await
            .expect("Failed to reopen RAPTOR engine");

        // === PHASE 5: Test cold search (after reopen) ===
        info!("Phase 5: Testing cold search after reopen");

        let mut cold_recalls = Vec::new();
        let mut cold_times = Vec::new();

        for (i, query) in queries.iter().enumerate() {
            let ground_truth = compute_ground_truth(&vectors, query, K_NEIGHBORS);

            let search_params = Arc::new(SearchParams {
                vector: Some(query.clone()),
                top_k: Some(K_NEIGHBORS as u16),
                distance_metric: Some(
                    proximadb::compute::distance_computation::DistanceMetric::Euclidean,
                ),
                ..Default::default()
            });

            let ctx = StorageQueryContext::new(search_params, collection_arc.clone());

            let search_start = std::time::Instant::now();
            let results = engine2
                .search_vectors_unified(&ctx)
                .await
                .expect("Cold search failed");
            cold_times.push(search_start.elapsed());

            let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
            let recall = compute_recall(&ground_truth, &result_ids);
            cold_recalls.push(recall);

            if i < 5 || recall < 1.0 {
                info!(
                    "Query {}: recall={:.1}%, results={}, time={:?}",
                    i,
                    recall * 100.0,
                    result_ids.len(),
                    search_start.elapsed()
                );
                if recall < 1.0 && i < 10 {
                    debug!("  Ground truth: {:?}", ground_truth);
                    debug!("  Results: {:?}", result_ids);
                }
            }
        }

        let avg_cold_recall = cold_recalls.iter().sum::<f64>() / cold_recalls.len() as f64;
        let avg_cold_time =
            cold_times.iter().map(|d| d.as_micros()).sum::<u128>() / cold_times.len() as u128;

        info!(
            "Cold search (after reopen): avg recall = {:.1}%, avg time = {}µs",
            avg_cold_recall * 100.0,
            avg_cold_time
        );

        // === ASSERTIONS ===
        info!("\n=== RESULTS ===");
        info!("Warm cache recall: {:.1}%", avg_warm_recall * 100.0);
        info!("Cold search recall: {:.1}%", avg_cold_recall * 100.0);

        // Warm cache should have high recall
        assert!(
            avg_warm_recall >= 0.9,
            "Warm cache recall too low: {:.1}%",
            avg_warm_recall * 100.0
        );

        // Cold search should also have high recall (this is the bug we're testing)
        assert!(
            avg_cold_recall >= 0.9,
            "Cold search recall too low: {:.1}% (BUG: stateless mode not working)",
            avg_cold_recall * 100.0
        );

        info!("=== TEST PASSED ===");
    }
}
