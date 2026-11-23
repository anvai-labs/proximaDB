use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};

fn deterministic_unit_vector(index: usize, dimension: usize) -> Vec<f32> {
    fn next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    let mut state = 0xC0FF_EE12_3456_7890_u64 ^ (index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let values: Vec<f64> = (0..dimension)
        .map(|_| {
            let bits = next_u64(&mut state);
            (bits as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect();
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    values
        .into_iter()
        .map(|value| (value / norm) as f32)
        .collect()
}

fn catalog_collection(
    db: &EmbeddedProximaDB,
    name: &str,
) -> proximadb::proto::proximadb_v1::Collection {
    db.runtime()
        .block_on(
            db.shared_services()
                .collection_service
                .get_collection_with_tenant_context(name, None),
        )
        .expect("catalog lookup")
        .expect("catalog collection")
}

fn collection_data_path(collection: &proximadb::proto::proximadb_v1::Collection) -> String {
    use proximadb::storage::trait_components::path_resolver::{
        collection_data_path_typed, typed_identity_from_storage_assignment, typed_path_identity,
    };

    let assignment = collection
        .storage_assignment
        .as_ref()
        .expect("storage assignment");
    collection_data_path_typed(
        &assignment.base_location,
        &collection.id,
        typed_path_identity(typed_identity_from_storage_assignment(Some(assignment))),
    )
}

fn persisted_segments(db: &EmbeddedProximaDB, path: &str) -> Vec<String> {
    db.runtime().block_on(async {
        let filesystem = db
            .shared_services()
            .filesystem_factory
            .get_filesystem(path)
            .expect("filesystem for collection data path");
        filesystem
            .list(path)
            .await
            .expect("list collection data path")
            .into_iter()
            .filter(|entry| {
                !entry.metadata.is_directory
                    && (entry.name.ends_with(".pax") || entry.name.ends_with(".sst"))
            })
            .map(|entry| entry.url)
            .collect()
    })
}

// Verifies that a close/flush cycle writes data to the catalog-addressed storage path
// and that reopening the embedded database can search the flushed data without
// needing WAL replay. Also exercises idempotent flush (second flush is a no-op).
#[test]
fn embedded_flush_persists_and_recovers() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    // Create DB, insert, and flush
    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");

    db.create_collection("test_collection", 4, Some("sst"))
        .expect("create collection");
    let created_collection = catalog_collection(&db, "test_collection");
    let created_data_path = collection_data_path(&created_collection);

    let ids = vec!["v0".to_string(), "v1".to_string()];
    let vectors = vec![vec![0.1_f32, 0.2, 0.3, 0.4], vec![0.1_f32, 0.2, 0.3, 0.5]];

    db.insert("test_collection", ids.clone(), vectors.clone(), None)
        .expect("insert");

    // First flush should write SST + update manifest; second flush should be a no-op
    db.flush().expect("flush");
    db.flush().expect("idempotent flush");
    let segments = persisted_segments(&db, &created_data_path);
    assert!(
        !segments.is_empty(),
        "flush must publish a segment at the catalog-addressed data path {created_data_path}"
    );
    drop(db); // close database

    // Reopen and search; results should be served from flushed storage, not WAL
    let mut reopen_config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    reopen_config.enable_wal = true;
    let reopened = EmbeddedProximaDB::new(reopen_config).expect("reopen db");
    let recovered_collection = catalog_collection(&reopened, "test_collection");
    assert_eq!(
        recovered_collection.id, created_collection.id,
        "catalog object identity must remain stable across reopen"
    );
    assert_eq!(
        collection_data_path(&recovered_collection),
        created_data_path,
        "catalog-addressed data path must remain stable across reopen"
    );

    let results = reopened
        .search("test_collection", vectors[0].clone(), 2, None)
        .expect("search");

    assert!(
        results.iter().any(|r| r.id == "v0"),
        "Expected to recover vector v0 after reopen; got ids: {:?}",
        results.iter().map(|r| r.id.clone()).collect::<Vec<_>>()
    );

    // Additional flush to ensure no pending WAL remains after search
    reopened.flush().expect("final flush");
}

/// Moderate-scale recovery test (kept bounded to avoid excessive runtime).
#[test]
fn embedded_flush_persists_many_vectors() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    eprintln!("📁 Test data path: {:?}", data_path);

    // Create DB with 100 vectors
    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");

    db.create_collection("test_collection", 128, Some("sst"))
        .expect("create collection");

    // Insert a bounded set that still exercises SST flushing/recovery paths.
    let num_vectors = 2_000;
    let dimension = 128;
    let mut ids = Vec::with_capacity(num_vectors);
    let mut vectors = Vec::with_capacity(num_vectors);

    for i in 0..num_vectors {
        ids.push(format!("vec_{}", i));
        vectors.push(deterministic_unit_vector(i, dimension));
    }

    db.insert("test_collection", ids.clone(), vectors.clone(), None)
        .expect("insert");

    eprintln!("✅ Inserted {} vectors", num_vectors);

    // Flush
    db.flush().expect("flush");
    eprintln!("✅ Flushed to disk");

    // List files in data directory to confirm SST files were written
    fn print_dir_recursive(path: &std::path::Path, indent: usize) {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                let prefix = "  ".repeat(indent);
                if path.is_dir() {
                    eprintln!(
                        "{}📁 {}/",
                        prefix,
                        path.file_name().unwrap().to_string_lossy()
                    );
                    print_dir_recursive(&path, indent + 1);
                } else {
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    eprintln!(
                        "{}📄 {} ({} bytes)",
                        prefix,
                        path.file_name().unwrap().to_string_lossy(),
                        size
                    );
                }
            }
        }
    }
    eprintln!("📂 Files in data directory:");
    print_dir_recursive(&data_path, 1);

    drop(db);
    eprintln!("✅ Closed database");

    // Reopen
    let mut reopen_config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    reopen_config.enable_wal = true;
    let reopened = EmbeddedProximaDB::new(reopen_config).expect("reopen db");

    eprintln!("✅ Reopened database");

    // Search for first vector - should find it
    let results = reopened
        .search("test_collection", vectors[0].clone(), 10, None)
        .expect("search");

    eprintln!("🔍 Search results: {} found", results.len());
    for (i, result) in results.iter().enumerate() {
        eprintln!("  [{}] id={}, score={}", i, result.id, result.score);
    }

    // vec_0 should be the top result with score 1.0 (exact match)
    assert!(
        !results.is_empty(),
        "Should find at least 1 result after reopen"
    );
    assert_eq!(
        results[0].id, "vec_0",
        "Query vector should be the top result"
    );
    assert!(
        (results[0].score - 1.0).abs() < 0.001,
        "Score should be ~1.0 for exact match"
    );

    eprintln!("✅ Test passed!");
}

/// Test that verifies SST block serialization/deserialization roundtrip
/// This ensures the writer format is compatible with the reader format.
#[test]
fn sst_block_serialization_roundtrip() {
    use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    // Create collection and insert vectors of varying sizes to test block boundaries
    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");

    db.create_collection("roundtrip_test", 64, Some("sst"))
        .expect("create collection");

    // Insert vectors in batches to create multiple blocks without overloading CI.
    // SST uses ~500 vectors per block, so 2K vectors yields multiple blocks.
    let num_batches = 4;
    let batch_size = 500;
    let total_to_insert = num_batches * batch_size;

    for batch_idx in 0..num_batches {
        let mut ids = Vec::with_capacity(batch_size);
        let mut vectors = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let global_id = batch_idx * batch_size + i;
            ids.push(format!("vec_{}", global_id));
            vectors.push(deterministic_unit_vector(global_id, 64));
        }

        db.insert("roundtrip_test", ids, vectors, None)
            .unwrap_or_else(|_| panic!("insert batch {}", batch_idx));
    }

    eprintln!(
        "✅ Inserted {} vectors in {} batches",
        total_to_insert, num_batches
    );

    // Flush to disk
    db.flush().expect("flush");
    eprintln!("✅ Flushed to disk");

    // Verify pre-reopen search works (search for first vector, should be in top-10)
    let query_vec = deterministic_unit_vector(0, 64);
    let pre_results = db
        .search("roundtrip_test", query_vec.clone(), 10, None)
        .expect("pre-reopen search");
    assert!(!pre_results.is_empty(), "Should find vectors before reopen");
    assert_eq!(
        pre_results[0].id, "vec_0",
        "vec_0 should be top result before reopen"
    );
    eprintln!(
        "✅ Pre-reopen: vec_0 is top result with score {}",
        pre_results[0].score
    );

    drop(db);
    eprintln!("✅ Closed database");

    // Reopen and verify all vectors are accessible
    let mut reopen_config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    reopen_config.enable_wal = true;
    let reopened = EmbeddedProximaDB::new(reopen_config).expect("reopen db");

    let post_results = reopened
        .search("roundtrip_test", query_vec, 10, None)
        .expect("post-reopen search");

    eprintln!(
        "📊 Post-reopen: {} results, top result: id={}, score={}",
        post_results.len(),
        post_results
            .first()
            .map(|r| r.id.as_str())
            .unwrap_or("none"),
        post_results.first().map(|r| r.score).unwrap_or(0.0)
    );

    // Verify vec_0 is still the top result after reopen (proves all blocks were read)
    assert!(!post_results.is_empty(), "Should find vectors after reopen");
    assert_eq!(
        post_results[0].id, "vec_0",
        "SERIALIZATION BUG: vec_0 not found as top result after reopen (got {})",
        post_results[0].id
    );
    assert!(
        (post_results[0].score - 1.0).abs() < 0.001,
        "vec_0 should have score ~1.0 (exact match), got {}",
        post_results[0].score
    );

    eprintln!("✅ Serialization roundtrip test passed!");
}

/// Test that verifies large k values work correctly (k=1000)
/// This ensures no hidden limits on search results.
#[test]
fn test_large_k_search_returns_correct_count() {
    use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    // Create collection and insert enough vectors to test k=1000 behavior
    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");

    db.create_collection("test_large_k", 64, Some("sst"))
        .expect("create collection");

    let num_vectors = 2_000;
    let mut ids = Vec::with_capacity(num_vectors);
    let mut vectors = Vec::with_capacity(num_vectors);

    for i in 0..num_vectors {
        ids.push(format!("vec_{}", i));
        vectors.push(deterministic_unit_vector(i, 64));
    }

    db.insert("test_large_k", ids.clone(), vectors.clone(), None)
        .expect("insert");

    eprintln!(
        "Inserted {} vectors (ids from {} to {})",
        ids.len(),
        ids.first().unwrap(),
        ids.last().unwrap()
    );

    // Check collection info to verify vector count
    let collection_info = db.get_collection("test_large_k").expect("get collection");
    if let Some(info) = collection_info {
        eprintln!("Collection info: {} vectors", info.vector_count);
    }

    // Search BEFORE flush/close to check if the limit is in WAL vs SST path
    // Test with different k values to trace where the limit is applied
    eprintln!("\nPre-flush k scan:");
    for k in [10, 50, 100, 200, 500, 1000] {
        let results = db
            .search("test_large_k", vectors[0].clone(), k, None)
            .unwrap_or_else(|_| panic!("search pre-flush k={}", k));
        eprintln!("  k={}: got {} results", k, results.len());
    }
    let results_pre_flush = db
        .search("test_large_k", vectors[0].clone(), 1000, None)
        .expect("search pre-flush");

    db.flush().expect("flush");

    // Search AFTER flush but before close
    let results_post_flush = db
        .search("test_large_k", vectors[0].clone(), 1000, None)
        .expect("search post-flush");
    eprintln!(
        "Post-flush k=1000: got {} results",
        results_post_flush.len()
    );

    drop(db);

    // Reopen and search with k=1000
    let mut reopen_config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    reopen_config.enable_wal = true;
    let reopened = EmbeddedProximaDB::new(reopen_config).expect("reopen db");

    // Test with k=1000 - should return exactly 1000 results
    let results_1000 = reopened
        .search("test_large_k", vectors[0].clone(), 1000, None)
        .expect("search k=1000");

    eprintln!("Post-reopen k=1000: got {} results", results_1000.len());

    // Check all three phases
    eprintln!("\nSummary:");
    eprintln!(
        "  Pre-flush (WAL only):     {} results",
        results_pre_flush.len()
    );
    eprintln!(
        "  Post-flush (WAL+SST):     {} results",
        results_post_flush.len()
    );
    eprintln!("  Post-reopen (SST only):   {} results", results_1000.len());

    assert_eq!(
        results_1000.len(),
        1000,
        "k=1000 should return exactly 1000 results when 2000 vectors exist"
    );
    assert_eq!(
        results_1000[0].id, "vec_0",
        "Query vector should be top result"
    );

    eprintln!("✅ Large k test passed - no hidden limits!");
}

/// Cold-read recall gate for TD-163 / TD-165.
///
/// `sst_block_serialization_roundtrip` above asserts recall only on `vec_0` — a
/// trivial, well-separated corner case whose top-1 stays correct even when the cold
/// SST read path misranks every other query. That weak assertion is exactly why a
/// real cold-read ranking regression slipped through and TD-165 was marked Resolved
/// prematurely (the runtime RCA showed post-restart top-1 jumping from `vec_0` to
/// `vec_8` for a mid-range query, while `vec_0` alone still looked correct).
///
/// This test closes that gap. It uses well-separated, deterministic, seeded-random
/// **unit** vectors so every inserted vector is its own unambiguous exact nearest
/// neighbour (cosine ≈ 1.0 for self, ≈ 0 for the rest — no near-tie ambiguity), then
/// measures recall@k on the **hot** path (memtable, pre-reopen) and the **cold**
/// path (post-reopen, vector WAL freed by the shared `materialize_collection` helper
/// → recall served from the SST segment). The gate is that cold recall must not
/// regress hot recall — the precise failure mode of a broken cold read, and the one
/// `free_wal=true` (TD-163) would expose if TD-165's fix were incomplete.
#[test]
fn cold_read_recall_survives_flush_and_reopen() {
    use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};

    // Deterministic splitmix64 PRNG (no `rand` dev-dep) → reproducible unit vectors.
    fn next_u64(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit_vec(seed: &mut u64, dim: usize) -> Vec<f32> {
        let v: Vec<f64> = (0..dim)
            .map(|_| {
                let bits = next_u64(seed);
                (bits as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v.iter().map(|x| (x / norm) as f32).collect()
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum::<f64>() as f32
    }
    // recall@k + exact-self-match-top-1 check over a spread of query vectors.
    fn recall_at_k(
        db: &EmbeddedProximaDB,
        vectors: &[Vec<f32>],
        query_idxs: &[usize],
        truth: &[Vec<usize>],
        top_k: usize,
    ) -> f32 {
        let mut sum = 0f32;
        for (qi, &q) in query_idxs.iter().enumerate() {
            let res = db
                .search("cold_recall", vectors[q].clone(), top_k, None)
                .expect("search");
            // Well-separated ⇒ the exact self-match must be top-1 (cosine 1.0).
            assert_eq!(
                res.first().map(|r| r.id.as_str()).unwrap_or(""),
                format!("v{q}"),
                "query v{q}: exact self-match is not top-1 (got {:?})",
                res.first().map(|r| r.id.clone())
            );
            let returned: std::collections::HashSet<String> =
                res.iter().map(|r| r.id.clone()).collect();
            let hits = truth[qi]
                .iter()
                .filter(|i| returned.contains(&format!("v{i}")))
                .count();
            sum += hits as f32 / top_k as f32;
        }
        sum / query_idxs.len() as f32
    }

    let dim = 64usize;
    let n = 1000usize;
    let top_k = 10usize;

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    let mut seed: u64 = 0xC0FFEE_1234_5678;
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| unit_vec(&mut seed, dim)).collect();

    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");
    db.create_collection("cold_recall", dim as u32, Some("sst"))
        .expect("create collection");

    // Insert in batches (~500/block → multiple SST blocks, exercises block boundaries).
    let batch = 500usize;
    let mut global = 0usize;
    while global < n {
        let end = (global + batch).min(n);
        let ids: Vec<String> = (global..end).map(|i| format!("v{i}")).collect();
        let vecs: Vec<Vec<f32>> = vectors[global..end].to_vec();
        db.insert("cold_recall", ids, vecs, None)
            .expect("insert batch");
        global = end;
    }
    db.flush()
        .expect("flush — materialize_collection frees the vector WAL (free_wal=true)");

    // Brute-force ground-truth top-k by cosine for a spread of query vectors.
    let query_idxs: Vec<usize> = vec![0, 200, 500, 799, 999];
    let truth: Vec<Vec<usize>> = query_idxs
        .iter()
        .map(|&q| {
            let mut ranked: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, cosine(&vectors[q], &vectors[i])))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.iter().take(top_k).map(|(i, _)| *i).collect()
        })
        .collect();

    let hot_recall = recall_at_k(&db, &vectors, &query_idxs, &truth, top_k);
    eprintln!("📊 hot (memtable) recall@{top_k} = {hot_recall:.3}");
    drop(db);

    // Reopen: the vector WAL was freed on flush, so recall is now served from the
    // cold SST read path — the path TD-165 fixed and TD-163 (free_wal=true) relies on.
    let mut reopen = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    reopen.enable_wal = true;
    let reopened = EmbeddedProximaDB::new(reopen).expect("reopen db");
    let cold_recall = recall_at_k(&reopened, &vectors, &query_idxs, &truth, top_k);
    eprintln!("📊 cold (SST, post-reopen) recall@{top_k} = {cold_recall:.3}");

    // The cold path must not regress the hot path — this is the gate that catches a
    // broken cold read (hot correct, cold wrong), the TD-165 failure mode.
    assert!(
        cold_recall + 0.05 >= hot_recall,
        "cold recall ({cold_recall:.3}) regressed hot recall ({hot_recall:.3}) — \
         the SST cold-read path is misranking"
    );
    // Absolute floor: well-separated unit vectors ⇒ recall must be high on both paths.
    let floor = 0.80f32;
    assert!(
        cold_recall >= floor,
        "cold recall@{top_k} = {cold_recall:.3} below floor {floor}"
    );

    eprintln!(
        "✅ Cold-read recall survives flush+reopen (hot={hot_recall:.3}, cold={cold_recall:.3})"
    );
}

/// Graph data must survive a close/reopen cycle in the in-process engine.
///
/// The sibling test above covers VECTOR durability and has always passed,
/// which is precisely why this gap survived: graph writes go through a
/// different WAL, and nothing on the in-process path ever flushed it. Writes
/// land in `UnifiedWALWriter::current_segment_data`, an in-memory buffer that
/// reaches disk only on an explicit flush or a 64 MB segment rotation;
/// `UnifiedWALEntry::new` never sets `requires_fsync`, so no graph write
/// triggers the former, and a realistic graph never reaches the latter.
///
/// Measured before the fix on a real code graph: ingest reported 177,046
/// edges, `graph_stats` agreed, `close()` returned cleanly, the WAL file was
/// 0 bytes on disk, and a fresh process saw `edges=0` — total silent loss
/// with every durability API reporting success (#1524). The server never had
/// this bug because `ProximaDB::stop` flushes graph WALs, but that loop is
/// gated on the NETWORK server object, which port-free embedded never builds.
///
/// Deliberately asserts through a REOPEN rather than in-memory stats: stats
/// answered 177,046 while the disk held nothing, so only a reopen can tell
/// durability from bookkeeping.
#[test]
fn embedded_graph_survives_close_and_reopen() {
    use proximadb_embedded::{EmbeddedGraphEdge, EmbeddedGraphNode};

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("graph-data");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    let graph_id = "durability_graph";
    let node_ids: Vec<String> = (0..50).map(|i| format!("n{i}")).collect();

    {
        let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
        config.enable_wal = true;
        let db = EmbeddedProximaDB::new(config).expect("create db");

        db.create_graph(graph_id, None).expect("create graph");
        db.create_nodes(
            graph_id,
            node_ids
                .iter()
                .map(|id| EmbeddedGraphNode::new(id.clone()).with_label("Symbol"))
                .collect(),
        )
        .expect("create nodes");
        db.create_edges(
            graph_id,
            node_ids
                .windows(2)
                .enumerate()
                .map(|(i, pair)| {
                    EmbeddedGraphEdge::new(pair[0].clone(), pair[1].clone(), "CALLS")
                        .with_id(format!("e{i}"))
                })
                .collect(),
        )
        .expect("create edges");

        let stats = db.graph_stats(graph_id).expect("stats before close");
        assert_eq!(
            stats.total_edges,
            (node_ids.len() - 1) as u64,
            "sanity: the engine should report the edges it just accepted"
        );

        db.close();
    }

    // Fresh instance over the same directory — the only honest durability test.
    let mut reopen_config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    reopen_config.enable_wal = true;
    let reopened = EmbeddedProximaDB::new(reopen_config).expect("reopen db");

    let stats = reopened.graph_stats(graph_id).expect("stats after reopen");
    assert_eq!(
        stats.total_edges,
        (node_ids.len() - 1) as u64,
        "graph edges must survive close/reopen; 0 here is the #1524 data loss"
    );

    let outgoing = reopened
        .get_outgoing_edges(graph_id, &node_ids[0], None)
        .expect("outgoing after reopen");
    assert_eq!(
        outgoing.len(),
        1,
        "topology must be queryable after reopen, not merely counted"
    );
    assert_eq!(outgoing[0].to_node_id, node_ids[1]);
}
