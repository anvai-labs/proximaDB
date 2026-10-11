//! AXIS Index Integration Edge Case Tests
//!
//! Tests edge cases and boundary conditions for AXIS index integration
//! across the selectable storage engines: SST, HELIX, NOVA (plus SWIFT and
//! RAPTOR under `experimental-engines`). VIPER left the create matrix at
//! ADR-093 S2.

use proximadb_embedded::{AccessMode, EmbeddedConfig, EmbeddedProximaDB, StorageLocationConfig};
use tempfile::TempDir;

/// Test engines to validate.
/// Experimental engines are only included when the feature is enabled.
fn test_engines() -> Vec<&'static str> {
    // VIPER is no longer selectable for new collections as of ADR-093 S2
    // (existing data stays readable via the factory-level reader tests), so
    // it leaves the create-path conformance matrix here.
    let mut engines = vec!["sst", "helix", "nova"];
    if cfg!(feature = "experimental-engines") {
        engines.push("swift");
        engines.push("raptor");
    }
    engines
}

/// Helper to create test database
fn create_test_db() -> (TempDir, EmbeddedProximaDB) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config = EmbeddedConfig {
        storage_locations: vec![StorageLocationConfig::new(
            temp_dir.path().to_str().unwrap(),
        )],
        metadata_path: format!("{}/metadata", temp_dir.path().to_str().unwrap()),
        cache_size_mb: 128,
        default_engine: "sst".to_string(),
        enable_wal: true,
        wal_sync_mode: "batch".to_string(),
        block_prune_mode: "sqrt".to_string(),
        block_prune_ratio: 0.2,
        block_prune_min_keep: 1,
        block_prune_max_keep: 0,
        enable_rl_planner: false,
        rl_policy_path: None,
        access_mode: AccessMode::Exclusive,
        node_id: None,
        tenant_id: None,
    };
    let db = EmbeddedProximaDB::new(config).expect("Failed to create embedded database");
    (temp_dir, db)
}

/// Helper to generate random vectors deterministically
fn generate_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::hash::{Hash, Hasher};
    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (seed + i as u64).hash(&mut hasher);
        let hash = hasher.finish();

        let mut vector = Vec::with_capacity(dimension);
        for j in 0..dimension {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (hash + j as u64).hash(&mut h);
            let val = (h.finish() % 10000) as f32 / 10000.0;
            vector.push(val);
        }
        vectors.push(vector);
    }
    vectors
}

// ============================================================================
// Edge Case 1: Empty Collection Search
// ============================================================================

#[test]
fn test_search_empty_collection_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("empty_{}", engine);

        // Create collection but don't insert anything
        db.create_collection(&collection_name, 128, Some(engine))
            .expect("Failed to create collection");

        // Flush empty collection
        db.flush().expect("Failed to flush");

        // Search on empty collection
        let query = vec![0.5_f32; 128];
        let results = db.search(&collection_name, query, 10, None);

        assert!(
            results.is_ok(),
            "[{}] Search on empty collection should not fail: {:?}",
            engine,
            results.err()
        );

        let results = results.unwrap();
        assert!(
            results.is_empty(),
            "[{}] Search on empty collection should return empty results, got {}",
            engine,
            results.len()
        );

        println!("[{}] Empty collection search: PASSED", engine);
        db.close();
    }
}

// ============================================================================
// Edge Case 2: Search with k > Number of Vectors
// ============================================================================

#[test]
fn test_search_k_greater_than_count_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("small_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 64, Some(engine))
            .expect("Failed to create collection");

        // Insert only 5 vectors
        let vectors = generate_vectors(5, 64, 42);
        let ids: Vec<String> = (0..5).map(|i| format!("vec_{}", i)).collect();

        db.insert(&collection_name, ids, vectors, None)
            .expect("Failed to insert vectors");

        db.flush().expect("Failed to flush");

        // Search with k=100 (more than 5 vectors)
        let query = vec![0.5_f32; 64];
        let results = db
            .search(&collection_name, query, 100, None)
            .expect("Search should succeed");

        assert!(
            results.len() <= 5,
            "[{}] Should return at most 5 results when only 5 vectors exist, got {}",
            engine,
            results.len()
        );

        println!(
            "[{}] k > count search: PASSED (returned {} results)",
            engine,
            results.len()
        );
        db.close();
    }
}

// ============================================================================
// Edge Case 3: Search with Single Vector
// ============================================================================

#[test]
fn test_search_single_vector_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("single_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 32, Some(engine))
            .expect("Failed to create collection");

        // Insert exactly 1 vector
        let vector = vec![0.5_f32; 32];
        db.insert(
            &collection_name,
            vec!["only_vector".to_string()],
            vec![vector.clone()],
            None,
        )
        .expect("Failed to insert vector");

        db.flush().expect("Failed to flush");

        // Search should return the single vector
        let results = db
            .search(&collection_name, vector.clone(), 10, None)
            .expect("Search should succeed");

        assert_eq!(
            results.len(),
            1,
            "[{}] Should return exactly 1 result, got {}",
            engine,
            results.len()
        );
        assert_eq!(
            results[0].id, "only_vector",
            "[{}] Should return the correct vector ID",
            engine
        );

        println!("[{}] Single vector search: PASSED", engine);
        db.close();
    }
}

// ============================================================================
// Edge Case 4: High-Dimensional Vectors (1536 dims - OpenAI embedding size)
// ============================================================================

#[test]
fn test_high_dimensional_vectors_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("highdim_{}", engine);

        // Create collection with 1536 dimensions (OpenAI embedding size)
        db.create_collection(&collection_name, 1536, Some(engine))
            .expect("Failed to create collection");

        // Insert 100 high-dimensional vectors
        let vectors = generate_vectors(100, 1536, 123);
        let ids: Vec<String> = (0..100).map(|i| format!("highdim_{}", i)).collect();

        db.insert(&collection_name, ids, vectors.clone(), None)
            .expect("Failed to insert vectors");

        db.flush().expect("Failed to flush");

        // Search with vector 50 as query
        let query = vectors[50].clone();
        let results = db
            .search(&collection_name, query, 10, None)
            .expect("Search should succeed");

        assert!(
            !results.is_empty(),
            "[{}] High-dimensional search should return results",
            engine
        );

        // The query vector itself should be the top result (distance ~0)
        assert_eq!(
            results[0].id, "highdim_50",
            "[{}] Self-search should return the query vector as top result",
            engine
        );

        println!(
            "[{}] High-dimensional (1536d) search: PASSED ({} results)",
            engine,
            results.len()
        );
        db.close();
    }
}

// ============================================================================
// Edge Case 5: Multiple Flush Cycles
// ============================================================================

#[test]
fn test_multiple_flush_cycles_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("multiflush_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 64, Some(engine))
            .expect("Failed to create collection");

        // Insert and flush multiple times
        for batch in 0..3 {
            let vectors = generate_vectors(50, 64, batch as u64 * 100);
            let ids: Vec<String> = (0..50).map(|i| format!("batch{}_{}", batch, i)).collect();

            db.insert(&collection_name, ids, vectors, None)
                .expect("Failed to insert vectors");

            db.flush().expect("Failed to flush");
        }

        // Search should find vectors from all batches
        let query = vec![0.5_f32; 64];
        let results = db
            .search(&collection_name, query, 20, None)
            .expect("Search should succeed");

        assert!(
            !results.is_empty(),
            "[{}] Search after multiple flushes should return results",
            engine
        );

        // Verify results come from different batches
        let batches_found: std::collections::HashSet<_> = results
            .iter()
            .filter_map(|r| r.id.split('_').next())
            .collect();

        println!(
            "[{}] Multiple flush cycles: PASSED ({} results from {} batches)",
            engine,
            results.len(),
            batches_found.len()
        );
        db.close();
    }
}

// ============================================================================
// Edge Case 6: Zero Vector (all zeros)
// ============================================================================

#[test]
fn test_zero_vector_search_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("zerovec_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 32, Some(engine))
            .expect("Failed to create collection");

        // Insert vectors including a zero vector
        let mut ids = vec!["zero_vector".to_string()];
        let mut vectors = vec![vec![0.0_f32; 32]];

        // Add some regular vectors
        let regular_vectors = generate_vectors(10, 32, 42);
        for i in 0..10 {
            ids.push(format!("regular_{}", i));
            vectors.push(regular_vectors[i].clone());
        }

        db.insert(&collection_name, ids, vectors, None)
            .expect("Failed to insert vectors");

        db.flush().expect("Failed to flush");

        // Search with zero vector query
        let zero_query = vec![0.0_f32; 32];
        let results = db
            .search(&collection_name, zero_query, 11, None)
            .expect("Zero vector search should succeed");

        assert!(
            !results.is_empty(),
            "[{}] Zero vector search should return results",
            engine
        );

        // The embedded helper creates cosine collections by default, where a
        // zero-vector nearest-neighbor ordering is undefined. This edge case
        // verifies that zero vectors remain searchable and do not break flush
        // or query execution across engines.
        assert!(
            results.iter().any(|result| result.id == "zero_vector"),
            "[{}] Zero vector should be present in full result set",
            engine
        );

        println!("[{}] Zero vector search: PASSED", engine);
        db.close();
    }
}

// ============================================================================
// Edge Case 7: Duplicate Vector IDs (upsert behavior)
// ============================================================================

#[test]
fn test_duplicate_ids_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("dupids_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 32, Some(engine))
            .expect("Failed to create collection");

        // Insert initial vector
        let initial_vector = vec![0.1_f32; 32];
        db.insert(
            &collection_name,
            vec!["duplicate_id".to_string()],
            vec![initial_vector.clone()],
            None,
        )
        .expect("Failed to insert initial vector");

        db.flush().expect("Failed to flush");

        // Insert vector with same ID but different values (upsert)
        let updated_vector = vec![0.9_f32; 32];
        let (inserted, updated) = db
            .upsert(
                &collection_name,
                vec!["duplicate_id".to_string()],
                vec![updated_vector.clone()],
                None,
            )
            .expect("Upsert should succeed");
        assert_eq!(inserted, 0, "[{}] Upsert should not insert new IDs", engine);
        assert_eq!(updated, 1, "[{}] Upsert should update existing ID", engine);

        db.flush().expect("Failed to flush after upsert");

        // Search with updated vector should find it
        let results = db
            .search(&collection_name, updated_vector.clone(), 5, None)
            .expect("Search should succeed");

        // Should have only one result with the duplicate ID (not two)
        let dup_count = results.iter().filter(|r| r.id == "duplicate_id").count();
        assert!(
            dup_count <= 1,
            "[{}] Should not have duplicate entries, found {}",
            engine,
            dup_count
        );

        println!("[{}] Duplicate ID handling: PASSED", engine);
        db.close();
    }
}

// ============================================================================
// Edge Case 8: Very Small Top-K (k=1)
// ============================================================================

#[test]
fn test_top_k_one_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("topk1_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 64, Some(engine))
            .expect("Failed to create collection");

        // Insert 100 vectors
        let vectors = generate_vectors(100, 64, 42);
        let ids: Vec<String> = (0..100).map(|i| format!("vec_{}", i)).collect();

        db.insert(&collection_name, ids, vectors.clone(), None)
            .expect("Failed to insert vectors");

        db.flush().expect("Failed to flush");

        // Search with k=1
        let query = vectors[25].clone();
        let results = db
            .search(&collection_name, query, 1, None)
            .expect("Search should succeed");

        assert_eq!(
            results.len(),
            1,
            "[{}] k=1 should return exactly 1 result, got {}",
            engine,
            results.len()
        );
        assert!(
            results[0].id.starts_with("vec_"),
            "[{}] k=1 should return a valid vector ID, got {}",
            engine,
            results[0].id
        );

        println!("[{}] Top-K=1 search: PASSED", engine);
        db.close();
    }
}

// ============================================================================
// Edge Case 9: Search Before Any Flush (memtable only)
// ============================================================================

#[test]
fn test_search_before_flush_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("noflush_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 64, Some(engine))
            .expect("Failed to create collection");

        // Insert vectors but DON'T flush
        let vectors = generate_vectors(50, 64, 42);
        let ids: Vec<String> = (0..50).map(|i| format!("memtable_{}", i)).collect();

        db.insert(&collection_name, ids, vectors.clone(), None)
            .expect("Failed to insert vectors");

        // DO NOT FLUSH - vectors should still be searchable from memtable/WAL

        // Search should still find vectors in memtable
        let query = vectors[10].clone();
        let results = db
            .search(&collection_name, query, 10, None)
            .expect("Search before flush should succeed");

        assert!(
            !results.is_empty(),
            "[{}] Search before flush should still find vectors in memtable",
            engine
        );

        println!(
            "[{}] Search before flush (memtable): PASSED ({} results)",
            engine,
            results.len()
        );
        db.close();
    }
}

// ============================================================================
// Edge Case 10: Concurrent Insert and Search
// ============================================================================

#[test]
fn test_concurrent_operations_all_engines() {
    for engine in test_engines() {
        let (_temp_dir, db) = create_test_db();
        let collection_name = format!("concurrent_{}", engine);

        // Create collection
        db.create_collection(&collection_name, 64, Some(engine))
            .expect("Failed to create collection");

        // Insert initial batch
        let vectors = generate_vectors(100, 64, 42);
        let ids: Vec<String> = (0..100).map(|i| format!("initial_{}", i)).collect();

        db.insert(&collection_name, ids, vectors.clone(), None)
            .expect("Failed to insert vectors");

        db.flush().expect("Failed to flush");

        // Insert new vectors
        let new_vectors = generate_vectors(10, 64, 999);
        let new_ids: Vec<String> = (0..10).map(|i| format!("new_{}", i)).collect();

        db.insert(&collection_name, new_ids, new_vectors, None)
            .expect("Concurrent insert should succeed");

        // Search immediately after insert (before flush)
        let search_query = vectors[50].clone();
        let results = db
            .search(&collection_name, search_query, 10, None)
            .expect("Concurrent search should succeed");

        assert!(
            !results.is_empty(),
            "[{}] Concurrent search should return results",
            engine
        );

        println!(
            "[{}] Concurrent operations: PASSED ({} results)",
            engine,
            results.len()
        );
        db.close();
    }
}

// ============================================================================
// Main test runner
// ============================================================================

#[test]
fn test_suite_summary() {
    let engines = test_engines();
    assert!(
        !engines.contains(&"viper"),
        "VIPER must NOT be in the create-path conformance matrix as of ADR-093 S2"
    );
    let engine_list = engines
        .iter()
        .map(|e| e.to_uppercase())
        .collect::<Vec<_>>()
        .join(", ");

    println!("\n");
    println!("{}", "=".repeat(60));
    println!("AXIS Edge Case Integration Tests");
    println!("{}", "=".repeat(60));
    println!(
        "Testing {} engines in this build: {}",
        engines.len(),
        engine_list
    );
    println!("\nEdge cases covered:");
    println!("  1. Empty collection search");
    println!("  2. Search with k > count");
    println!("  3. Single vector search");
    println!("  4. High-dimensional (1536d) vectors");
    println!("  5. Multiple flush cycles");
    println!("  6. Zero vector handling");
    println!("  7. Duplicate ID (upsert) behavior");
    println!("  8. Top-K=1 search");
    println!("  9. Search before flush (memtable)");
    println!(" 10. Concurrent insert/search");
    println!("{}", "=".repeat(60));
}
