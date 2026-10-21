//! End-to-end HTTP test for F5 Slice 3 (BM25 over external text + hybrid
//! retrieval, TD-090). Boots a full ProximaDB server and drives the v2
//! `external-collections` endpoints over REST:
//!
//!   write external Parquet (id, title, vector)
//!     → POST /api/v2/external-collections        (register, text_column=title)
//!     → POST /api/v2/external-collections/:id/build
//!     → POST /api/v2/external-collections/:id/search { vector, text, k }
//!
//! Asserts the hybrid (vector + BM25) search returns fused hits over HTTP: a
//! lexical-only match (whose vector is NOT the nearest) is fused in alongside
//! the vector match, and hits carry federated props from the source. This
//! exercises the full wiring (SharedServices → AppState → REST router →
//! ExternalCollectionService::hybrid_search → HybridFusionEngine), not just the
//! service API.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, StringBuilder};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use serde_json::json;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

/// Shared harness boot with the one config this suite needs beyond the defaults.
async fn start() -> anyhow::Result<TestPgServer> {
    TestPgServer::start_with(|config| {
        // Required for `POST /external-collections/:id/build`: external vector
        // indexes are refused unless AXIS indexes are enabled, and the default
        // is false (proximadb-config asserts that default). Without this the
        // build step 500s and this test never reached its own assertions --
        // pre-existing, found while adding the path-confinement gate opt-in.
        config.storage.optimization.enable_axis_indexes = true;
    })
    .await
}

/// Write `n` rows: `id="doc-i"`, `title="title-i"` (unique lexical token per
/// row), one-hot `vector` at position `i` (requires `dim >= n`).
fn write_parquet(path: &std::path::Path, n: usize, dim: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));
    let mut id_b = StringBuilder::new();
    let mut title_b = StringBuilder::new();
    let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for i in 0..n {
        id_b.append_value(format!("doc-{i}"));
        title_b.append_value(format!("title-{i}"));
        let mut v = vec![0.0f32; dim];
        v[i] = 1.0 + (i as f32) * 0.01;
        for x in &v {
            vec_b.values().append_value(*x);
        }
        vec_b.append(true);
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_b.finish()),
            Arc::new(title_b.finish()),
            Arc::new(vec_b.finish()),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_collection_bm25_hybrid_e2e_over_http() {
    let server = start().await.expect("server start");
    let (http, base) = server.rest_client().expect("rest client");

    let dim = 48usize;
    let parquet = std::env::temp_dir().join(format!(
        "{}.parquet",
        server.unique_name("proximadb_bm25_e2e")
    ));
    write_parquet(&parquet, 40, dim);
    allowlist_parent_of(&parquet);

    // 1) Register the external collection with a BM25 text column.
    let reg = http
        .post(format!("{base}/api/v2/external-collections"))
        .json(&json!({
            "name": server.unique_name("ext_bm25"),
            "location": parquet.to_str().unwrap(),
            "id_column": "id",
            "vector_column": "vector",
            "dimension": dim,
            "text_column": "title",
        }))
        .send()
        .await
        .expect("register send");
    assert!(
        reg.status().is_success(),
        "register: {} {}",
        reg.status(),
        reg.text().await.unwrap_or_default()
    );
    let reg_body: serde_json::Value = reg.json().await.expect("register json");
    let ext_id = reg_body["collection"]["id"]
        .as_str()
        .expect("collection id")
        .to_string();

    // 2) Build the in-place IVF + BM25 indexes.
    let build = http
        .post(format!("{base}/api/v2/external-collections/{ext_id}/build"))
        .send()
        .await
        .expect("build send");
    assert!(
        build.status().is_success(),
        "build: {} {}",
        build.status(),
        build.text().await.unwrap_or_default()
    );
    let build_body: serde_json::Value = build.json().await.expect("build json");
    assert_eq!(build_body["indexed_record_count"].as_u64(), Some(40));

    // 3) Hybrid search: vector points at doc-3; text query is doc-7's title.
    let mut qvec = vec![0.0f32; dim];
    qvec[3] = 1.0;
    let search = http
        .post(format!(
            "{base}/api/v2/external-collections/{ext_id}/search"
        ))
        .json(&json!({ "vector": qvec, "text": "title-7", "k": 5 }))
        .send()
        .await
        .expect("search send");
    assert!(
        search.status().is_success(),
        "search: {} {}",
        search.status(),
        search.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = search.json().await.expect("search json");
    let hits = body["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty(), "hybrid search returned hits");

    let ids: Vec<&str> = hits.iter().filter_map(|h| h["id"].as_str()).collect();
    assert!(
        ids.contains(&"doc-7"),
        "BM25 lexical match fused in over HTTP: {ids:?}"
    );
    assert!(
        ids.contains(&"doc-3"),
        "vector match retained over HTTP: {ids:?}"
    );

    // Fused hits carry federated props (the title) from the un-copied source.
    let doc7 = hits.iter().find(|h| h["id"] == "doc-7").expect("doc-7 hit");
    assert!(
        doc7["props"].get("title").is_some(),
        "hit must carry federated props: {doc7}"
    );

    let _ = std::fs::remove_file(&parquet);
}

/// Allowlist the parquet's own directory for this test process.
///
/// `POST /api/v2/external-collections` is now gated by the same fail-closed
/// `PROXIMADB_EXTERNAL_TABLE_ROOTS` allowlist as `CREATE EXTERNAL TABLE`
/// (TD-SECSCAN-1: the location previously went straight to `fs::metadata` /
/// `fs::read_dir`). Unset means every location is rejected, which is the
/// shipped default, so a test that reads a real file must opt in.
///
/// The root is written as a BARE path, not `file://...`, because these tests
/// pass a bare path as the location and the allowlist match is textual.
fn allowlist_parent_of(path: &std::path::Path) {
    let dir = path
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "/".to_string());
    // Safe here: integration tests get their own process, and this runs before
    // the service under test reads the gate.
    unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &dir) };
}
