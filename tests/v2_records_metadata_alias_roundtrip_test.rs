//! TD-REC-1 e2e guard: a REST v2 insert written with the documents-API
//! `metadata` spelling must land in `props` and be filterable — through the
//! live server, not just the DTO boundary.
//!
//! The unit test added with the alias (PR #2337) pins deserialization of
//! `ProximaRecordInput`. This file pins the END-TO-END property the TD
//! demands: the v0.5.0 Homebrew smoke inserted records with `metadata`,
//! got `3/3, failed_count: 0`, and then a filtered search over that field
//! matched NOTHING — the write side silently dropped the field. With the
//! alias, the same request must (a) succeed, (b) persist the field where
//! filters look, and (c) be returned by both the scan filter and the typed
//! search filter. The both-keys-present case must fail loud (serde
//! duplicate-field), never silently prefer one spelling.
//!
//! Path A (`/records/scan`) is the load-bearing read: it applies the filter
//! against the visible record set directly. Path B (typed `/search`) proves
//! the same field is visible to the search entry point the server actually
//! serves.

use std::collections::BTreeSet;
use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct TestServer {
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl TestServer {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let tmp_data = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = free_port();
        config.api.arrow_flight_port = free_port();
        config.api.pg_port = Some(free_port());
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
            .timeout(Duration::from_secs(5))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!("REST didn't become ready in 15s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
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

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

fn ids(values: &[serde_json::Value]) -> BTreeSet<String> {
    values
        .iter()
        .filter_map(|v| {
            v.get("id")
                .or_else(|| v.get("record_id"))
                .and_then(|id| id.as_str())
                .map(str::to_string)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metadata_spelled_insert_is_filterable_on_all_read_paths() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let collection = format!(
        "malias_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    // dim-1 collection with an indexed, filterable account_id column.
    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": collection,
            "dimension": 1,
            "enable_proxima_record": true,
            "schema": {
                "columns": [
                    { "name": "account_id", "data_type": "text", "indexed": true, "filterable": true }
                ],
                "enforcement": "hybrid",
                "allow_additional_fields": true
            }
        }))
        .send()
        .await
        .expect("create");
    assert!(
        create.status().is_success(),
        "create: {} {}",
        create.status(),
        create.text().await.unwrap_or_default()
    );

    // THE GUARD: every record is written with the documents-API `metadata`
    // spelling — exactly the request shape the v0.5.0 Homebrew smoke sent.
    // Before the alias this insert reported success while dropping the
    // field, and every filter below matched nothing.
    let insert = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({
            "records": [
                { "id": "a1", "vector": [0.0], "metadata": { "account_id": "acctA" } },
                { "id": "a2", "vector": [0.0], "metadata": { "account_id": "acctA" } },
                { "id": "b1", "vector": [0.0], "metadata": { "account_id": "acctB" } }
            ],
            "upsert": true
        }))
        .send()
        .await
        .expect("insert");
    assert!(
        insert.status().is_success(),
        "insert with the metadata spelling must be accepted: {} {}",
        insert.status(),
        insert.text().await.unwrap_or_default()
    );

    sleep(Duration::from_millis(500)).await;

    // The field must have PERSISTED (read-back by id), not just been accepted.
    let read_back = http
        .get(format!("{base}/api/v2/collections/{collection}/records/a1"))
        .send()
        .await
        .expect("get a1");
    assert!(
        read_back.status().is_success(),
        "get a1: {}",
        read_back.status()
    );
    let a1_body = read_back.text().await.expect("a1 body");
    assert!(
        a1_body.contains("acctA"),
        "the metadata-spelled field must be persisted and readable back; got: {a1_body}"
    );

    let expected: BTreeSet<String> = ["a1".to_string(), "a2".to_string()].into_iter().collect();

    // ── Path A: /records/scan with an equality-map filter ───────────────────
    let scan = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/scan"
        ))
        .json(&json!({ "limit": 10, "filter": { "account_id": "acctA" } }))
        .send()
        .await
        .expect("scan");
    assert!(scan.status().is_success(), "scan status {}", scan.status());
    let scan_body: serde_json::Value = scan.json().await.expect("scan json");
    let scan_records = scan_body
        .get("records")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        ids(&scan_records),
        expected,
        "scan must return exactly the acctA records written via the metadata spelling, got {scan_body}"
    );

    // ── Path B: typed /search with a typed-filter list ──────────────────────
    let search = http
        .post(format!("{base}/api/v2/collections/{collection}/search"))
        .json(&json!({
            "vector": [0.0],
            "top_k": 10,
            "filters": [{ "field": "account_id", "op": "eq", "value": "acctA" }]
        }))
        .send()
        .await
        .expect("search");
    assert!(
        search.status().is_success(),
        "search status {}",
        search.status()
    );
    let search_body: serde_json::Value = search.json().await.expect("search json");
    let search_records = search_body
        .get("results")
        .or_else(|| search_body.get("records"))
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        ids(&search_records),
        expected,
        "typed search must return exactly the acctA records written via the metadata spelling, got {search_body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metadata_and_props_together_fail_loud() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let collection = format!(
        "mboth_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": collection,
            "dimension": 1,
            "enable_proxima_record": true,
            "schema": {
                "columns": [
                    { "name": "account_id", "data_type": "text", "indexed": true, "filterable": true }
                ],
                "enforcement": "hybrid",
                "allow_additional_fields": true
            }
        }))
        .send()
        .await
        .expect("create");
    assert!(
        create.status().is_success(),
        "create: {} {}",
        create.status(),
        create.text().await.unwrap_or_default()
    );

    // Both spellings present in ONE record is an ambiguous write: it must be
    // REJECTED (serde duplicate-field) — never silently resolved by
    // preferring one key, which would make the accepted spelling depend on
    // map iteration order.
    let insert = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({
            "records": [
                {
                    "id": "x1",
                    "vector": [0.0],
                    "props": { "account_id": "acctA" },
                    "metadata": { "account_id": "acctA" }
                }
            ],
            "upsert": true
        }))
        .send()
        .await
        .expect("insert");
    assert!(
        !insert.status().is_success(),
        "props + metadata in one record must fail loud, got {} {}",
        insert.status(),
        insert.text().await.unwrap_or_default()
    );
}
