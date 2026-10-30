//! v0.2 release-smoke battery (REST v2 record CRUD + route-health round trip).
//!
//! Required by `MVP_RELEASE_READINESS_RECONCILIATION_2026_05_28.adoc` P0 #3:
//! "Add one non-ignored REST/gRPC v2 record smoke that creates a collection,
//! inserts records, searches, reads route-health, and deletes/tombstones a record."
//!
//! Must be runnable via `cargo test --test release_smoke_v2 -- --test-threads=1`
//! with **no pre-existing server**. The test owns its lifetime via `ProximaDB::new`
//! / `ProximaDB::shutdown`.
//!
//! Failures here block the release cut by virtue of being wired into
//! `make release-check` → `release-smoke`.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct ReleaseSmokeServer {
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl ReleaseSmokeServer {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let flight_port = free_port();
        let pg_port = free_port();
        let tmp_data = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.arrow_flight_port = flight_port;
        config.api.pg_port = Some(pg_port);
        config.api.unified_mode = false;
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp_data.path().display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            if let Ok(resp) = http.get(&health_url).send().await
                && resp.status().is_success()
            {
                break;
            }
            if std::time::Instant::now() > deadline {
                anyhow::bail!("REST didn't become ready in 20s");
            }
            sleep(Duration::from_millis(100)).await;
        }

        Ok(Self {
            rest_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.rest_port)
    }
}

impl Drop for ReleaseSmokeServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

/// CREATE → INSERT → SEARCH → route-health → DELETE on the v2 record path.
///
/// Asserts the v0.2 contract:
/// 1. v2 record insert returns success.
/// 2. v2 search finds the inserted record (covers TD-072 / v2 insert→search
///    wiring fix; if this regresses the release is blocked).
/// 3. Route-health exposes the documented diagnostic blocks.
/// 4. Delete tombstones the record so a subsequent search returns 0 matches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_v2_record_release_smoke_round_trip() {
    let server = ReleaseSmokeServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();

    let coll = format!(
        "release_smoke_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dim: usize = 8;
    let n: usize = 5;

    // 1. CREATE — fp32 SST cosine collection (no fp16, so this test isolates
    //    the v2 INSERT→SEARCH path from any precision-coercion confounds).
    let create_resp = http
        .post(format!("{}/api/v2/collections", server.base_url()))
        .json(&json!({
            "name": coll,
            "dimension": dim,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false,
        }))
        .send()
        .await
        .expect("v2 create");
    assert!(
        create_resp.status().is_success(),
        "v2 create: {} {}",
        create_resp.status(),
        create_resp.text().await.unwrap_or_default()
    );

    // 2. INSERT — n records with deterministic vectors.
    let records: Vec<Value> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|j| (i as f32) * 0.1 + (j as f32) * 0.01)
                .collect();
            json!({ "id": format!("rec-{i}"), "vector": v })
        })
        .collect();
    let insert_resp = http
        .post(format!(
            "{}/api/v2/collections/{}/records/batch",
            server.base_url(),
            coll
        ))
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("v2 insert");
    let insert_status = insert_resp.status();
    let insert_body = insert_resp.text().await.unwrap_or_default();
    assert!(
        insert_status.is_success(),
        "v2 insert: {insert_status} {insert_body}"
    );

    // Settling for WAL → search delta merge visibility.
    sleep(Duration::from_millis(750)).await;

    // 3. SEARCH — query the exact vector for rec-0; expect cosine ≈ 1.0.
    let q: Vec<f32> = (0..dim).map(|j| (j as f32) * 0.01).collect();
    let search_resp = http
        .post(format!(
            "{}/api/v2/collections/{}/search",
            server.base_url(),
            coll
        ))
        .json(&json!({ "vector": q, "top_k": 10 }))
        .send()
        .await
        .expect("v2 search");
    let search_status = search_resp.status();
    let search_body = search_resp.text().await.unwrap_or_default();
    assert!(
        search_status.is_success(),
        "v2 search: {search_status} {search_body}"
    );
    let search_json: Value = serde_json::from_str(&search_body).expect("search JSON");
    let total = search_json
        .get("total_matches")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    assert!(
        total >= 1,
        "v2 INSERT must feed v2 SEARCH index (P0 #3 from MVP reconciliation). \
         Got total_matches={total}, body={search_body}"
    );

    // 4. Route-health — must surface the documented diagnostic blocks.
    let rh_resp = http
        .get(format!(
            "{}/api/v2/_diagnostics/collections/{}/route-health",
            server.base_url(),
            coll
        ))
        .send()
        .await
        .expect("route-health GET");
    assert!(rh_resp.status().is_success(), "route-health status");
    let rh_body = rh_resp.text().await.unwrap_or_default();
    let rh_json: Value = serde_json::from_str(&rh_body).expect("route-health JSON");
    for block in [
        "writes",
        "freshness",
        "filtered_ann",
        "object_economy",
        "recall_probe",
        "discovery",
        "suspension",
        "cold_serving",
    ] {
        assert!(
            rh_json.get(block).is_some(),
            "route-health missing required v0.2 block '{block}'. Body: {rh_body}"
        );
    }
    // v0.2 contract: WriteContractHealth disallows conditional/filter/patch.
    let wc = &rh_json["writes"];
    assert_eq!(
        wc.get("conditional_write").and_then(Value::as_bool),
        Some(false),
        "v0.2: writes.conditional_write must be false. Body: {rh_body}"
    );
    assert_eq!(
        wc.get("filter_write").and_then(Value::as_bool),
        Some(false),
        "v0.2: writes.filter_write must be false. Body: {rh_body}"
    );
    assert_eq!(
        wc.get("patch").and_then(Value::as_bool),
        Some(false),
        "v0.2: writes.patch must be false. Body: {rh_body}"
    );

    // 5. DELETE — tombstone rec-0; subsequent search must drop it.
    let delete_resp = http
        .delete(format!(
            "{}/api/v2/collections/{}/records/rec-0",
            server.base_url(),
            coll
        ))
        .send()
        .await
        .expect("v2 delete");
    assert!(
        delete_resp.status().is_success(),
        "v2 delete: {} {}",
        delete_resp.status(),
        delete_resp.text().await.unwrap_or_default()
    );

    sleep(Duration::from_millis(500)).await;
    let search2 = http
        .post(format!(
            "{}/api/v2/collections/{}/search",
            server.base_url(),
            coll
        ))
        .json(&json!({ "vector": q, "top_k": 10 }))
        .send()
        .await
        .expect("v2 search post-delete");
    let search2_body = search2.text().await.unwrap_or_default();
    let search2_json: Value = serde_json::from_str(&search2_body).expect("search2 JSON");
    let ids: Vec<String> = search2_json
        .get("matches")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|s| s.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !ids.iter().any(|id| id == "rec-0"),
        "rec-0 should be tombstoned after DELETE. Got ids={ids:?}, body={search2_body}"
    );
}

/// v0.2 release-readiness audit round 2: POST to a non-existent collection
/// must return HTTP 404, not HTTP 200 with `BatchOperationResult::failure`
/// in the body. The first reconciliation missed this because every test in
/// the suite used a freshly created collection; the bug was discoverable
/// only by adversarial calls.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_v2_insert_to_missing_collection_returns_404() {
    let server = ReleaseSmokeServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .unwrap();

    let missing = "definitely_not_a_real_collection_4d4f0a";
    let resp = http
        .post(format!(
            "{}/api/v2/collections/{}/records/batch",
            server.base_url(),
            missing
        ))
        .json(&json!({ "records": [{ "id": "x", "vector": [0.0, 0.0] }] }))
        .send()
        .await
        .expect("insert send");
    assert_eq!(
        resp.status().as_u16(),
        404,
        "POST to missing collection must return HTTP 404 (got {}): {}",
        resp.status(),
        resp.text().await.unwrap_or_default()
    );
}

/// v0.2 release-readiness audit round 2: the search endpoint must cap
/// `top_k` to prevent a malformed/malicious client from requesting an
/// unbounded result buffer allocation. Default cap is 10_000; the test
/// requests one above the cap and asserts HTTP 400.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_v2_search_top_k_above_cap_returns_400() {
    let server = ReleaseSmokeServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .unwrap();

    let coll = format!(
        "release_top_k_cap_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let create_resp = http
        .post(format!("{}/api/v2/collections", server.base_url()))
        .json(&json!({
            "name": coll,
            "dimension": 4,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
        }))
        .send()
        .await
        .expect("v2 create");
    assert!(create_resp.status().is_success(), "v2 create");

    let resp = http
        .post(format!(
            "{}/api/v2/collections/{}/search",
            server.base_url(),
            coll
        ))
        .json(&json!({ "vector": [0.0, 0.0, 0.0, 0.0], "top_k": 100_001 }))
        .send()
        .await
        .expect("search send");
    assert_eq!(
        resp.status().as_u16(),
        400,
        "top_k above cap must return HTTP 400 (got {}): {}",
        resp.status(),
        resp.text().await.unwrap_or_default()
    );
}

/// TD-ENGSEL-1: a REST v2 create that omits `engine` (no index, quantization,
/// recall target or policy) lands on SST, the supported default — not HELIX
/// (Beta), which the auto-router used to pick for this exact shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_v2_create_without_engine_defaults_to_sst() {
    let server = ReleaseSmokeServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .unwrap();

    let coll = format!(
        "release_default_engine_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let create_resp = http
        .post(format!("{}/api/v2/collections", server.base_url()))
        .json(&json!({
            "name": coll,
            "dimension": 4,
            "distance_metric": "cosine",
        }))
        .send()
        .await
        .expect("v2 create");
    assert!(
        create_resp.status().is_success(),
        "v2 create: {}",
        create_resp.text().await.unwrap_or_default()
    );

    let got: Value = http
        .get(format!("{}/api/v2/collections/{}", server.base_url(), coll))
        .send()
        .await
        .expect("v2 get")
        .json()
        .await
        .expect("v2 get json");
    assert_eq!(
        got["engine"], "sst",
        "a create without `engine` must default to SST: {got}"
    );
}
