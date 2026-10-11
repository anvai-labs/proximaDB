// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-137: pins the REAL `POST /api/v2/graphs/{graph_id}/fusion-search` REST contract.
//!
//! The fusion core (`FusionService`) is covered by unit tests and by the embedded
//! parity gate (`tests/embedded_code_graph_parity_e2e.rs`), which argues
//! "embedded result IS the server result by construction" — but the REST handler
//! itself (`fusion_search_v2`: request validation, grain/route-policy/weight
//! mapping, DTO serialization, the router + tenant-middleware path) was never
//! driven over HTTP by any test. This e2e closes that gap: it seeds a live
//! server over REST (vector collection + graph with canonical co-indexed oids)
//! and asserts the fused output's correctness properties, including the ones
//! that discriminate a silently-dark graph leg:
//!
//! - canonical-oid hits carry `source_count >= 2` (vector AND graph contributed);
//! - an `orphan` graph node with NO vector record still appears (expansion ran);
//! - a vector-only record stays `source_count == 1` (no false consensus);
//! - graph labels ride on fused hits (#485); vector-only hits have none.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

struct RestServer {
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl RestServer {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let pg_port = free_port();
        let tmp = TempDir::new()?;
        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp.path().display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp.path().display());
        // TD-CONV-1: anchor the system catalog in this server's temp dir, not the
        // shared CWD-relative `file://./metadata` default (cross-run DDL replay).
        config.storage.metadata_url = format!("file://{}", tmp.path().join("metadata").display());
        let mut db = ProximaDB::new(config).await?;
        db.start().await?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => anyhow::bail!("REST not ready"),
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        sleep(Duration::from_millis(200)).await;
        Ok(Self {
            rest_port,
            db: Some(db),
            _tmp: tmp,
        })
    }

    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.rest_port)
    }
}

impl Drop for RestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

#[test]
fn fusion_search_rest_e2e() {
    std::thread::Builder::new()
        .name("fusion-search-rest-e2e".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rt")
                .block_on(body())
        })
        .expect("spawn")
        .join()
        .expect("panic");
}

/// main --CALLS--> parse --CALLS--> validate; main --CALLS--> io;
/// parse --IMPORTS--> util; main --CALLS--> orphan (graph-only, no vector record).
/// `solo` is a vector record with no graph node.
struct Fixture {
    graph_id: String,
    collection: String,
}

async fn post_json(
    http: &reqwest::Client,
    url: String,
    body: Value,
) -> (reqwest::StatusCode, String) {
    let resp = http.post(url).json(&body).send().await.expect("POST");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    (status, text)
}

/// Strictly decreasing positive cosine to the query `e0`: main=1.0, parse≈0.894,
/// validate≈0.707, io≈0.447, util≈0.316, solo=0.0 (strictly last). Deterministic
/// seed ordering (top-5 = the five canonical nodes) requires NO ties.
fn embedding_for(nid: &str) -> Vec<f32> {
    match nid {
        "main" => vec![1.0, 0.0, 0.0, 0.0],
        "parse" => vec![1.0, 0.5, 0.0, 0.0],
        "validate" => vec![1.0, 1.0, 0.0, 0.0],
        "io" => vec![1.0, 2.0, 0.0, 0.0],
        "util" => vec![1.0, 3.0, 0.0, 0.0],
        "solo" => vec![0.0, 1.0, 0.0, 0.0],
        _ => unreachable!("fixture node {nid}"),
    }
}

async fn seed(server: &RestServer, http: &reqwest::Client) -> Fixture {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let graph_id = format!("g_fusion_{nanos}");
    let collection = format!("fusion_vecs_{nanos}");

    // Vector collection (fp32 cosine, matching release_smoke_v2's isolated shape).
    let (status, text) = post_json(
        http,
        format!("{}/api/v2/collections", server.base()),
        json!({
            "name": collection,
            "dimension": 4,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false,
        }),
    )
    .await;
    assert!(status.is_success(), "collection create: {status} {text}");

    // Records whose ids are the canonical co-indexed oids (`graph/{gid}/node/{nid}`)
    // plus one vector-only record. Similarities are strictly ordered (see
    // `embedding_for`): query e0 → `main` is the exact nearest, `solo` is last.
    let node_ids = ["main", "parse", "validate", "io", "util"];
    let mut records: Vec<Value> = node_ids
        .iter()
        .map(|nid| {
            json!({
                "id": format!("graph/{graph_id}/node/{nid}"),
                "vector": embedding_for(nid),
            })
        })
        .collect();
    records.push(json!({ "id": "solo", "vector": embedding_for("solo") }));
    let (status, text) = post_json(
        http,
        format!(
            "{}/api/v2/collections/{}/records/batch",
            server.base(),
            collection
        ),
        json!({ "records": records }),
    )
    .await;
    assert!(status.is_success(), "record batch: {status} {text}");

    // Graph with matching node ids (bare — the fusion seed strips the canonical
    // prefix), an `orphan` node that has NO vector record, and the edges.
    let (status, text) = post_json(
        http,
        format!("{}/api/v2/graphs", server.base()),
        json!({ "graph_id": graph_id, "name": graph_id }),
    )
    .await;
    assert!(status.is_success(), "graph create: {status} {text}");

    // `solo` intentionally has NO graph node — it is vector-only by construction.
    let nodes: Vec<Value> = node_ids
        .iter()
        .copied()
        .chain(std::iter::once("orphan"))
        .map(|nid| json!({ "id": nid, "labels": ["Function"] }))
        .collect();
    let (status, text) = post_json(
        http,
        format!("{}/api/v2/graphs/{graph_id}/nodes/batch", server.base()),
        json!({ "nodes": nodes }),
    )
    .await;
    assert!(status.is_success(), "node batch: {status} {text}");

    let edges = [
        ("e1", "main", "parse", "CALLS"),
        ("e2", "parse", "validate", "CALLS"),
        ("e3", "main", "io", "CALLS"),
        ("e4", "parse", "util", "IMPORTS"),
        ("e5", "main", "orphan", "CALLS"),
    ];
    let (status, text) = post_json(
        http,
        format!("{}/api/v2/graphs/{graph_id}/edges/batch", server.base()),
        json!({
            "edges": edges
                .iter()
                .map(|(id, from, to, ty)| json!({
                    "id": id, "from_node_id": from, "to_node_id": to, "edge_type": ty,
                }))
                .collect::<Vec<_>>()
        }),
    )
    .await;
    assert!(status.is_success(), "edge batch: {status} {text}");

    // Settle WAL → search visibility (release_smoke_v2 precedent).
    sleep(Duration::from_millis(750)).await;

    Fixture {
        graph_id,
        collection,
    }
}

fn hit_by_oid<'a>(results: &'a [Value], oid: &str) -> Option<&'a Value> {
    results.iter().find(|r| r["oid"].as_str() == Some(oid))
}

async fn fusion_request(
    server: &RestServer,
    http: &reqwest::Client,
    fx: &Fixture,
    extra: Value,
) -> (reqwest::StatusCode, String) {
    let mut body = json!({
        "vector_collection": fx.collection,
        "query_vector": [1.0, 0.0, 0.0, 0.0],
        "limit": 10,
        "max_depth": 2,
        "max_seeds": 5,
    });
    body.as_object_mut()
        .expect("object")
        .extend(extra.as_object().cloned().unwrap_or_default());
    post_json(
        http,
        format!(
            "{}/api/v2/graphs/{}/fusion-search",
            server.base(),
            fx.graph_id
        ),
        body,
    )
    .await
}

async fn body() {
    let server = RestServer::start().await.expect("server");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .expect("http");
    let fx = seed(&server, &http).await;

    // --- 1. Default (PIT-calibrated) policy: fusion correctness properties. ---
    let (status, text) = fusion_request(&server, &http, &fx, json!({})).await;
    assert!(status.is_success(), "fusion-search: {status} {text}");
    let resp: Value = serde_json::from_str(&text).expect("fusion JSON");
    let results = resp["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "fusion returned no results");

    let main_oid = format!("graph/{}/node/main", fx.graph_id);
    let parse_oid = format!("graph/{}/node/parse", fx.graph_id);
    let orphan_oid = format!("graph/{}/node/orphan", fx.graph_id);

    // Graph leg contributed: every canonical node hit is in BOTH sources.
    for (nid, oid) in [("main", main_oid.as_str()), ("parse", parse_oid.as_str())] {
        let hit = hit_by_oid(results, oid)
            .unwrap_or_else(|| panic!("fused results must contain {nid} ({oid}); got: {text}"));
        let sc = hit["source_count"].as_u64().unwrap_or_default();
        assert!(
            sc >= 2,
            "{nid} must fuse vector+graph (source_count >= 2), got {sc}"
        );
    }

    // Expansion ran: the orphan node has NO vector record — it can only be in
    // the union via the graph leg (a dark graph leg silently drops it).
    let orphan = hit_by_oid(results, &orphan_oid)
        .unwrap_or_else(|| panic!("orphan (graph-only) node missing from results: {text}"));
    assert_eq!(
        orphan["source_count"].as_u64().unwrap_or_default(),
        1,
        "orphan is graph-only"
    );
    assert_eq!(
        orphan["labels"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or_default(),
        1,
        "orphan carries its graph label (#485): {orphan}"
    );

    // No false consensus for the vector-only record.
    let solo = hit_by_oid(results, "solo")
        .unwrap_or_else(|| panic!("vector-only record missing from results: {text}"));
    assert_eq!(
        solo["source_count"].as_u64().unwrap_or_default(),
        1,
        "solo must stay source_count == 1"
    );
    assert!(
        solo["labels"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(true),
        "solo has no graph node — labels must be empty: {solo}"
    );

    // The ANN seed's nearest neighbor leads the ranking.
    assert_eq!(
        results[0]["oid"].as_str(),
        Some(main_oid.as_str()),
        "main (exact query match + consensus) must rank first: {text}"
    );

    // Stats are coherent.
    let stats = &resp["stats"];
    assert!(
        stats["sources_fused"].as_u64().unwrap_or_default() >= 2,
        "vector + graph must both fuse: {stats}"
    );
    assert_eq!(
        stats["items_out"].as_u64().unwrap_or_default(),
        results.len() as u64,
        "items_out must match results length"
    );

    // --- 2. `limit` is honored. ---
    let (status, text) = fusion_request(&server, &http, &fx, json!({ "limit": 3 })).await;
    assert!(status.is_success(), "fusion limit variant: {status} {text}");
    let resp: Value = serde_json::from_str(&text).expect("fusion JSON");
    assert!(
        resp["results"].as_array().expect("results").len() <= 3,
        "limit=3 exceeded: {text}"
    );

    // --- 3. RRF fallback policy also works end-to-end. ---
    let (status, text) = fusion_request(&server, &http, &fx, json!({ "rrf": true })).await;
    assert!(status.is_success(), "fusion rrf variant: {status} {text}");
    let resp: Value = serde_json::from_str(&text).expect("fusion JSON");
    assert!(
        !resp["results"].as_array().expect("results").is_empty(),
        "rrf fusion returned no results: {text}"
    );

    // --- 4. Edge-grain fusion (D8) is reachable over REST. ---
    let (status, text) = fusion_request(&server, &http, &fx, json!({ "grain": "both" })).await;
    assert!(
        status.is_success(),
        "fusion grain=both variant: {status} {text}"
    );
    let resp: Value = serde_json::from_str(&text).expect("fusion JSON");
    assert!(
        !resp["results"].as_array().expect("results").is_empty(),
        "grain=both fusion returned no results: {text}"
    );

    // --- 5. Request validation fails closed (handler contract). ---
    let (status, _) = fusion_request(&server, &http, &fx, json!({ "query_vector": [] })).await;
    assert_eq!(
        status,
        reqwest::StatusCode::BAD_REQUEST,
        "empty query_vector must 400"
    );
    let (status, _) =
        fusion_request(&server, &http, &fx, json!({ "vector_collection": "   " })).await;
    assert_eq!(
        status,
        reqwest::StatusCode::BAD_REQUEST,
        "blank vector_collection must 400"
    );
}
