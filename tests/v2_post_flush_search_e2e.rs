//! Characterization: does a v2-inserted record stay retrievable **after a
//! real WAL→storage flush**?
//!
//! Prior handoffs asserted a "v2 INSERT→search-index flush gap": records served
//! from the WAL memtable while unflushed, but dropped from `search`/`scan` once
//! flushed (because flush did not register them with the searchable index). That
//! claim was never pinned by a test — the cited `fp16_v2_insert_search_e2e`
//! waits only ~1s and never forces a flush, so it only ever exercised the
//! WAL-resident path.
//!
//! This test forces the flush explicitly (`ProximaDB::force_flush_collection`,
//! which drains the WAL so subsequent reads must come from persisted storage)
//! and re-runs the SAME search + filtered scan. It is the regression gate for
//! the post-flush read path on the going-forward v2 surface. If the flush gap is
//! real, the post-flush assertions fail and this file is the minimal repro; if
//! it passes, the post-flush path is locked in.
//!
//! Two collections are covered because the v2 write path forks on
//! `enable_proxima_record`:
//!   - `false` → canonical vector-record path (the fp16 round-trip path),
//!   - `true`  → ProximaRecord path (the metadata-filter path; what the hybrid
//!     filter enforcement reads via `list_all_records_with_tenant_context`).

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

    /// Drain the collection's WAL/memtable into persisted storage so the next
    /// read cannot be served from memory.
    async fn force_flush(&self, collection: &str) -> anyhow::Result<()> {
        self.db
            .as_ref()
            .expect("db present")
            .force_flush_collection(collection)
            .await
    }

    fn persisted_pax_files(&self) -> Vec<String> {
        fn visit(path: &std::path::Path, files: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(path) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit(&path, files);
                } else if path.extension().is_some_and(|ext| ext == "pax") {
                    files.push(path.display().to_string());
                }
            }
        }

        let mut files = Vec::new();
        visit(self._tmp_data.path(), &mut files);
        files.sort();
        files
    }

    fn persisted_files(&self) -> Vec<String> {
        fn visit(path: &std::path::Path, files: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(path) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit(&path, files);
                } else {
                    files.push(path.display().to_string());
                }
            }
        }

        let mut files = Vec::new();
        visit(self._tmp_data.path(), &mut files);
        files.sort();
        files
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

async fn unflushed_batch_count(collection_id: &str) -> usize {
    let Some(write_buffer) =
        proximadb::storage::persistence::write_ahead_log::get_global_write_buffer_behavior()
    else {
        return 0;
    };
    write_buffer.unflushed_batch_count(collection_id).await
}

/// enable_proxima_record=false (canonical vector path): exact-match search must
/// return the same top-1 hit before AND after a forced flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_vector_search_survives_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_flush_vec_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dim: usize = 8;
    let n: usize = 5;

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": dim,
            "engine": "sst",
            "distance_metric": "cosine",
            "enable_proxima_record": false,
        }))
        .send()
        .await
        .expect("create");
    let create_status = create.status();
    let create_body = create.text().await.expect("create response body");
    assert!(
        create_status.is_success(),
        "create: {} {}",
        create_status,
        create_body
    );
    let create_json: serde_json::Value =
        serde_json::from_str(&create_body).expect("create response json");
    let collection_id = create_json
        .get("collection_id")
        .and_then(serde_json::Value::as_str)
        .expect("create response collection_id");

    // One-hot directions so cosine cleanly separates them; rec-2 = e_2.
    let records: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            json!({ "id": format!("rec-{i}"), "vector": v })
        })
        .collect();
    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("insert");
    assert!(
        insert.status().is_success(),
        "insert: {} {}",
        insert.status(),
        insert.text().await.unwrap_or_default()
    );
    sleep(Duration::from_millis(500)).await;

    let mut query = vec![0.0f32; dim];
    query[2 % dim] = 1.0;
    let search = |q: Vec<f32>| {
        let http = http.clone();
        let url = format!("{base}/api/v2/collections/{name}/search");
        async move {
            let resp = http
                .post(url)
                .json(&json!({ "vector": q, "top_k": 1 }))
                .send()
                .await
                .expect("search");
            assert!(
                resp.status().is_success(),
                "search status {}",
                resp.status()
            );
            let body: serde_json::Value = resp.json().await.expect("search json");
            let hits = body
                .get("results")
                .or_else(|| body.get("hits"))
                .or_else(|| body.get("records"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            (hits, body)
        }
    };

    // Pre-flush sanity: WAL-resident search returns rec-2 (proven path).
    let (pre_hits, pre_body) = search(query.clone()).await;
    let pre_top = pre_hits
        .first()
        .and_then(|h| h.get("id").or_else(|| h.get("record_id")))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(
        pre_top, "rec-2",
        "pre-flush top-1 should be rec-2: {pre_body}"
    );
    assert!(
        unflushed_batch_count(collection_id).await > 0,
        "the regression setup must have a live WAL epoch before force-flush"
    );

    // Force the WAL→storage flush, then re-run the identical search.
    server.force_flush(&name).await.expect("force flush");
    assert_eq!(
        unflushed_batch_count(collection_id).await,
        0,
        "force-flush must return only after retiring the published WAL epoch"
    );

    let (post_hits, post_body) = search(query).await;
    assert!(
        !post_hits.is_empty(),
        "POST-FLUSH search returned no hits — the flush gap is real: {post_body}"
    );
    let post_top = post_hits
        .first()
        .and_then(|h| h.get("id").or_else(|| h.get("record_id")))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(
        post_top, "rec-2",
        "POST-FLUSH top-1 should still be rec-2: {post_body}"
    );
}

/// enable_proxima_record=true (ProximaRecord path): a metadata-filtered scan —
/// the exact read path the hybrid BM25 filter enforcement reuses — must return
/// the same matching set before AND after a forced flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_filtered_scan_survives_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_flush_scan_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": 1,
            "engine": "sst",
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
    let create_status = create.status();
    let create_body = create.text().await.expect("create response body");
    assert!(
        create_status.is_success(),
        "create: {} {}",
        create_status,
        create_body
    );
    let create_json: serde_json::Value =
        serde_json::from_str(&create_body).expect("create response json");
    let collection_id = create_json
        .get("collection_id")
        .and_then(serde_json::Value::as_str)
        .expect("create response collection_id");

    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({
            "records": [
                { "id": "a1", "vector": [0.0], "props": { "account_id": "acctA" } },
                { "id": "a2", "vector": [0.0], "props": { "account_id": "acctA" } },
                { "id": "b1", "vector": [0.0], "props": { "account_id": "acctB" } }
            ],
            "upsert": true
        }))
        .send()
        .await
        .expect("insert");
    assert!(
        insert.status().is_success(),
        "insert: {} {}",
        insert.status(),
        insert.text().await.unwrap_or_default()
    );
    sleep(Duration::from_millis(500)).await;

    let expected: BTreeSet<String> = ["a1".to_string(), "a2".to_string()].into_iter().collect();
    let scan = |filter: Option<serde_json::Value>| {
        let http = http.clone();
        let url = format!("{base}/api/v2/collections/{name}/records/scan");
        async move {
            let resp = http
                .post(url)
                .json(&json!({ "limit": 10, "filter": filter }))
                .send()
                .await
                .expect("scan");
            assert!(resp.status().is_success(), "scan status {}", resp.status());
            let body: serde_json::Value = resp.json().await.expect("scan json");
            let recs = body
                .get("records")
                .and_then(|r| r.as_array())
                .cloned()
                .unwrap_or_default();
            (ids(&recs), body)
        }
    };

    // Pre-flush: WAL-resident filtered scan returns exactly acctA.
    let account_filter = Some(json!({ "account_id": "acctA" }));
    let (pre_ids, pre_body) = scan(account_filter.clone()).await;
    assert_eq!(pre_ids, expected, "pre-flush filtered scan: {pre_body}");
    assert!(
        unflushed_batch_count(collection_id).await > 0,
        "the metadata regression setup must have a live WAL epoch before force-flush"
    );

    // Force the WAL→storage flush, then re-run the identical filtered scan.
    server.force_flush(&name).await.expect("force flush");
    assert_eq!(
        unflushed_batch_count(collection_id).await,
        0,
        "force-flush must retire the metadata path's published WAL epoch"
    );
    let persisted_pax_files = server.persisted_pax_files();
    assert!(
        !persisted_pax_files.is_empty(),
        "force-flush returned without publishing a PAX segment; physical files: {:?}",
        server.persisted_files()
    );

    let expected_all: BTreeSet<String> =
        ["a1", "a2", "b1"].into_iter().map(str::to_string).collect();
    let (post_all_ids, post_all_body) = scan(None).await;
    assert_eq!(
        post_all_ids, expected_all,
        "POST-FLUSH unfiltered scan must discover every persisted record: \
         body={post_all_body}, physical_pax={persisted_pax_files:?}"
    );

    let (post_ids, post_body) = scan(account_filter).await;
    assert_eq!(
        post_ids, expected,
        "POST-FLUSH filtered scan must still return exactly acctA records \
         (the path hybrid filter enforcement reuses): {post_body}"
    );
}

/// TD-PGWIRE-VECDEL-1 (storage-inclusive scan): a record that was FLUSHED and
/// then deleted must not come back from `records/scan`. The scan merges the
/// unflushed WAL delta with flushed storage; it read the deduplicating memtable
/// reader, which drops a tombstoned winner, so the unflushed delete never met
/// the flushed live copy and the merge resurrected it (invariant #16d).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_scan_hides_a_record_deleted_after_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_flush_del_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": 1,
            "engine": "sst",
            "enable_proxima_record": true
        }))
        .send()
        .await
        .expect("create");
    let create_status = create.status();
    let create_body = create.text().await.expect("create response body");
    assert!(
        create_status.is_success(),
        "create: {create_status} {create_body}"
    );

    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({
            "records": [
                { "id": "keep", "vector": [0.0] },
                { "id": "gone", "vector": [0.0] }
            ],
            "upsert": true
        }))
        .send()
        .await
        .expect("insert");
    assert!(insert.status().is_success(), "insert: {}", insert.status());
    sleep(Duration::from_millis(500)).await;

    // Both live copies are now flushed; the delete below stays unflushed.
    server.force_flush(&name).await.expect("force flush");
    assert!(
        !server.persisted_pax_files().is_empty(),
        "setup: force-flush must publish a PAX segment; physical files: {:?}",
        server.persisted_files()
    );

    let delete = http
        .delete(format!("{base}/api/v2/collections/{name}/records/gone"))
        .send()
        .await
        .expect("delete");
    assert!(delete.status().is_success(), "delete: {}", delete.status());

    let resp = http
        .post(format!("{base}/api/v2/collections/{name}/records/scan"))
        .json(&json!({ "limit": 10 }))
        .send()
        .await
        .expect("scan");
    assert!(resp.status().is_success(), "scan status {}", resp.status());
    let body: serde_json::Value = resp.json().await.expect("scan json");
    let recs = body
        .get("records")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        ids(&recs),
        ["keep".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "a record deleted after flush must not be resurrected by the scan: {body}"
    );

    let get = http
        .get(format!("{base}/api/v2/collections/{name}/records/gone"))
        .send()
        .await
        .expect("get");
    assert_eq!(
        get.status(),
        reqwest::StatusCode::NOT_FOUND,
        "the by-id read must agree with the scan"
    );

    // Vector search over the flushed segment must not return it either.
    let resp = http
        .post(format!("{base}/api/v2/collections/{name}/search"))
        .json(&json!({ "vector": [0.0], "top_k": 10 }))
        .send()
        .await
        .expect("search");
    assert!(
        resp.status().is_success(),
        "search status {}",
        resp.status()
    );
    let body: serde_json::Value = resp.json().await.expect("search json");
    let hits = body
        .get("results")
        .or_else(|| body.get("hits"))
        .or_else(|| body.get("records"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        ids(&hits),
        ["keep".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "vector search must not return a record deleted after flush: {body}"
    );
}

/// The by-id read must not let a STALE unflushed tombstone hide a newer flushed
/// copy. Flush retires vector batches but deliberately retains tombstone-only
/// batches (tombstone persistence is still open, TD-USUB-1), so after
/// delete → re-insert → flush the memtable holds only the old tombstone while
/// storage holds the live re-insert. A by-id read that trusted the memtable's
/// dead winner without consulting storage returned 404 for a live record.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_get_returns_a_record_reinserted_after_delete_and_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_reinsert_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": 1,
            "engine": "sst",
            "enable_proxima_record": true
        }))
        .send()
        .await
        .expect("create");
    assert!(create.status().is_success(), "create: {}", create.status());

    let upsert = |v: f32| {
        let http = http.clone();
        let url = format!("{base}/api/v2/collections/{name}/records/batch");
        async move {
            let resp = http
                .post(url)
                .json(&json!({ "records": [ { "id": "x", "vector": [v] } ], "upsert": true }))
                .send()
                .await
                .expect("upsert");
            assert!(resp.status().is_success(), "upsert: {}", resp.status());
        }
    };

    upsert(0.0).await;
    sleep(Duration::from_millis(200)).await;
    let delete = http
        .delete(format!("{base}/api/v2/collections/{name}/records/x"))
        .send()
        .await
        .expect("delete");
    assert!(delete.status().is_success(), "delete: {}", delete.status());
    sleep(Duration::from_millis(200)).await;
    upsert(1.0).await;
    sleep(Duration::from_millis(500)).await;

    // Flush: the re-insert goes to storage; the tombstone-only batch stays.
    server.force_flush(&name).await.expect("force flush");

    let get = http
        .get(format!("{base}/api/v2/collections/{name}/records/x"))
        .send()
        .await
        .expect("get");
    let status = get.status();
    let body = get.text().await.unwrap_or_default();
    assert_eq!(
        status,
        reqwest::StatusCode::OK,
        "a record re-inserted after its delete must be readable after flush: {body}"
    );
    let got: serde_json::Value = serde_json::from_str(&body).expect("get json");
    let vector = got
        .get("vector")
        .or_else(|| got.get("record").and_then(|r| r.get("vector")))
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        vector,
        json!([1.0]),
        "the by-id read must return the RE-INSERTED version, not the deleted original: {body}"
    );

    let resp = http
        .post(format!("{base}/api/v2/collections/{name}/records/scan"))
        .json(&json!({ "limit": 10 }))
        .send()
        .await
        .expect("scan");
    let body: serde_json::Value = resp.json().await.expect("scan json");
    let recs = body
        .get("records")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        ids(&recs),
        ["x".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "the scan must agree with the by-id read: {body}"
    );
}

/// Two versions of one id flushed together: the by-id read must return the
/// newer one. (Probe for the flushed point-lookup winner rule.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_get_returns_the_newer_of_two_flushed_versions() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_twover_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({ "name": name, "dimension": 1, "engine": "sst", "enable_proxima_record": true }))
        .send()
        .await
        .expect("create");
    assert!(create.status().is_success(), "create: {}", create.status());
    for v in [0.0_f32, 1.0] {
        let resp = http
            .post(format!("{base}/api/v2/collections/{name}/records/batch"))
            .json(&json!({ "records": [ { "id": "x", "vector": [v] } ], "upsert": true }))
            .send()
            .await
            .expect("upsert");
        assert!(resp.status().is_success(), "upsert: {}", resp.status());
        sleep(Duration::from_millis(200)).await;
    }
    let before: serde_json::Value = http
        .get(format!("{base}/api/v2/collections/{name}/records/x"))
        .send()
        .await
        .expect("get")
        .json()
        .await
        .expect("json");
    server.force_flush(&name).await.expect("force flush");
    let after: serde_json::Value = http
        .get(format!("{base}/api/v2/collections/{name}/records/x"))
        .send()
        .await
        .expect("get")
        .json()
        .await
        .expect("json");
    assert_eq!(
        before.get("vector"),
        Some(&json!([1.0])),
        "pre-flush: {before}"
    );
    assert_eq!(
        after.get("vector"),
        Some(&json!([1.0])),
        "post-flush must keep the newer version: {after}"
    );
    let scan: serde_json::Value = http
        .post(format!("{base}/api/v2/collections/{name}/records/scan"))
        .json(&json!({ "limit": 10, "include_vector": true }))
        .send()
        .await
        .expect("scan")
        .json()
        .await
        .expect("scan json");
    let recs = scan
        .get("records")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(recs.len(), 1, "one current version per id: {scan}");
    assert_eq!(
        recs[0].get("vector"),
        Some(&json!([1.0])),
        "the scan must also serve the newer flushed version: {scan}"
    );
}

/// TD-READFID-1: a point read on an `fp32` collection must return the written
/// f32 vector BYTE-EXACTLY after the record is flushed to a PAX segment.
///
/// Before the fix the flush wrote only the lossy RaBitQ/SQ8 columns (the
/// exact-f32 tier was opt-in via a `pax_f32_tier` tag nobody sets), so
/// `GET /records/{id}` served an SQ8 reconstruction once the memtable drained.
/// Values are deliberately irregular so no quantization grid can hit them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_fp32_point_read_is_byte_exact_after_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_fp32_fid_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dim: usize = 8;
    let n: usize = 6;

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": dim,
            "engine": "sst",
            "distance_metric": "euclidean",
            "canonical_embedding_precision": "fp32",
        }))
        .send()
        .await
        .expect("create");
    let create_status = create.status();
    let create_body = create.text().await.expect("create response body");
    assert!(
        create_status.is_success(),
        "create: {create_status} {create_body}"
    );
    let create_json: serde_json::Value =
        serde_json::from_str(&create_body).expect("create response json");
    let collection_id = create_json
        .get("collection_id")
        .and_then(serde_json::Value::as_str)
        .expect("create response collection_id")
        .to_string();

    let written: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.013_717).sin() * 0.731 + j as f32 * 0.01)
                .collect()
        })
        .collect();
    let records: Vec<serde_json::Value> = written
        .iter()
        .enumerate()
        .map(|(i, v)| json!({ "id": format!("rec-{i}"), "vector": v }))
        .collect();
    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("insert");
    assert!(
        insert.status().is_success(),
        "insert: {} {}",
        insert.status(),
        insert.text().await.unwrap_or_default()
    );

    let read = |id: String| {
        let http = http.clone();
        let url = format!("{base}/api/v2/collections/{name}/records/{id}");
        async move {
            let resp = http.get(url).send().await.expect("get");
            let status = resp.status();
            let body: serde_json::Value = resp.json().await.expect("get json");
            assert!(status.is_success(), "get {id}: {status} {body}");
            body.get("vector")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("get {id}: no vector in {body}"))
                .iter()
                .map(|x| x.as_f64().expect("numeric component") as f32)
                .collect::<Vec<f32>>()
        }
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    // Control: memtable-served reads are exact.
    for (i, want) in written.iter().enumerate() {
        let got = read(format!("rec-{i}")).await;
        assert_eq!(
            bits(&got),
            bits(want),
            "pre-flush rec-{i}: {got:?} vs {want:?}"
        );
    }

    server.force_flush(&name).await.expect("force flush");
    assert_eq!(
        unflushed_batch_count(&collection_id).await,
        0,
        "force-flush must drain the WAL so reads come from the segment"
    );
    assert!(
        !server.persisted_pax_files().is_empty(),
        "flush must have produced a PAX segment: {:?}",
        server.persisted_files()
    );

    for (i, want) in written.iter().enumerate() {
        let got = read(format!("rec-{i}")).await;
        assert_eq!(
            bits(&got),
            bits(want),
            "POST-FLUSH rec-{i} is not the written f32 (TD-READFID-1): got {got:?} want {want:?}"
        );
    }
}

/// TD-READFID-2: the exact-f32 tier must SURVIVE compaction on the production
/// path. Compaction inputs reach the tier predicates as `file://{abs}` URLs;
/// before the fix the predicates `std::fs::read` those URLs, fail, and conclude
/// "no input has the tier", so the merged segment was written lossy and an
/// `fp32` collection re-lossified at its first compaction.
///
/// Drives the real trigger: five forced flushes reach the default L0 threshold
/// (`L0_COMPACTION_THRESHOLD = 5`), the flush enqueues compaction, and the
/// background worker merges the five tier'd segments into one. No test-only
/// compaction entry point is used.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v2_fp32_point_read_is_byte_exact_after_compaction() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = format!(
        "v2_fp32_cmp_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dim: usize = 8;
    let flushes: usize = 5;
    let per_flush: usize = 3;

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": dim,
            "engine": "sst",
            "distance_metric": "euclidean",
            "canonical_embedding_precision": "fp32",
        }))
        .send()
        .await
        .expect("create");
    let create_status = create.status();
    let create_body = create.text().await.expect("create response body");
    assert!(
        create_status.is_success(),
        "create: {create_status} {create_body}"
    );

    let written: Vec<Vec<f32>> = (0..flushes * per_flush)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.017_291).cos() * 0.613 + j as f32 * 0.011)
                .collect()
        })
        .collect();

    for flush in 0..flushes {
        let records: Vec<serde_json::Value> = (flush * per_flush..(flush + 1) * per_flush)
            .map(|i| json!({ "id": format!("rec-{i}"), "vector": written[i] }))
            .collect();
        let insert = http
            .post(format!("{base}/api/v2/collections/{name}/records/batch"))
            .json(&json!({ "records": records }))
            .send()
            .await
            .expect("insert");
        assert!(
            insert.status().is_success(),
            "insert: {} {}",
            insert.status(),
            insert.text().await.unwrap_or_default()
        );
        server.force_flush(&name).await.expect("force flush");
    }

    // Every flushed segment carries the exact tier (TD-READFID-1). Then wait
    // for the background compaction the fifth flush enqueued to retire them.
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let merged = loop {
        let pax = server.persisted_pax_files();
        if pax.len() == 1 {
            break pax[0].clone();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "compaction never merged the {flushes} flushed segments: {pax:?}"
        );
        sleep(Duration::from_millis(250)).await;
    };

    let merged_bytes = std::fs::read(&merged).expect("read merged segment");
    let merged_is_exact =
        proximadb::storage::engines::sst::segment_format::pax_segment_has_exact_vector_authority(
            &merged_bytes,
        );

    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let mut mismatches = Vec::new();
    for (i, want) in written.iter().enumerate() {
        let id = format!("rec-{i}");
        let resp = http
            .get(format!("{base}/api/v2/collections/{name}/records/{id}"))
            .send()
            .await
            .expect("get");
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.expect("get json");
        assert!(status.is_success(), "get {id}: {status} {body}");
        let got: Vec<f32> = body
            .get("vector")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("get {id}: no vector in {body}"))
            .iter()
            .map(|x| x.as_f64().expect("numeric component") as f32)
            .collect();
        if bits(&got) != bits(want) {
            mismatches.push(format!("{id}: got {got:?} want {want:?}"));
        }
    }
    assert!(
        merged_is_exact && mismatches.is_empty(),
        "POST-COMPACTION fp32 fidelity lost (TD-READFID-2): merged segment {merged} \
         has_exact_authority={merged_is_exact}; {} of {} records differ: {mismatches:#?}",
        mismatches.len(),
        written.len()
    );
}

// ---------------------------------------------------------------------------
// Filtered vector search: read-after-write correctness (mandate 16a/16d).
//
// The filtered path merges the WAL leg with flushed PAX storage. The WAL leg
// pre-prunes batches with a per-batch metadata bloom filter on conjunctive
// `Equals` conditions, and evaluates the predicate per record. Tombstones
// and superseding versions carry props that never match the queried value,
// so a filtered query must still see them — otherwise the stale flushed copy
// wins the merge. The reinsert test pins the opposite hazard: a stale
// retained tombstone must not hide a newer flushed copy.
// ---------------------------------------------------------------------------

async fn create_filterable_collection(http: &reqwest::Client, base: &str, prefix: &str) -> String {
    let name = format!(
        "{prefix}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name,
            "dimension": 2,
            "engine": "sst",
            "enable_proxima_record": true,
            "schema": {
                "columns": [
                    { "name": "cat", "data_type": "integer", "indexed": true, "filterable": true }
                ],
                "enforcement": "hybrid",
                "allow_additional_fields": true
            }
        }))
        .send()
        .await
        .expect("create");
    let status = create.status();
    let body = create.text().await.expect("create body");
    assert!(status.is_success(), "create: {status} {body}");
    name
}

async fn upsert(http: &reqwest::Client, base: &str, name: &str, records: serde_json::Value) {
    let resp = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({ "records": records, "upsert": true }))
        .send()
        .await
        .expect("upsert");
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    assert!(status.is_success(), "upsert: {status} {body}");
}

async fn filtered_search_ids(
    http: &reqwest::Client,
    base: &str,
    name: &str,
    cat: i64,
) -> (BTreeSet<String>, serde_json::Value) {
    let resp = http
        .post(format!("{base}/api/v2/collections/{name}/search"))
        .json(&json!({
            "vector": [0.0, 0.0],
            "top_k": 10,
            "filters": [{ "field": "cat", "op": "eq", "value": cat }]
        }))
        .send()
        .await
        .expect("filtered search");
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.expect("search json");
    assert!(status.is_success(), "filtered search: {status} {body}");
    let hits = body
        .get("results")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    (ids(&hits), body)
}

/// R1: a record deleted after flush must not reappear in a FILTERED search.
/// The tombstone batch carries no props, so a metadata bloom built from props
/// can never contain `cat=3`; skipping that batch hides the tombstone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filtered_search_hides_a_record_deleted_after_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = create_filterable_collection(&http, &base, "flt_del").await;

    upsert(
        &http,
        &base,
        &name,
        json!([
            { "id": "keep", "vector": [0.0, 0.0], "props": { "cat": 3 } },
            { "id": "gone", "vector": [0.0, 0.1], "props": { "cat": 3 } }
        ]),
    )
    .await;
    sleep(Duration::from_millis(500)).await;
    server.force_flush(&name).await.expect("force flush");
    assert!(
        !server.persisted_pax_files().is_empty(),
        "setup: flush must publish a PAX segment; files: {:?}",
        server.persisted_files()
    );

    let delete = http
        .delete(format!("{base}/api/v2/collections/{name}/records/gone"))
        .send()
        .await
        .expect("delete");
    assert!(delete.status().is_success(), "delete: {}", delete.status());

    let (got, body) = filtered_search_ids(&http, &base, &name, 3).await;
    assert_eq!(
        got,
        ["keep".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "a record deleted after flush must not be resurrected by a FILTERED search: {body}"
    );
}

/// R2: an update that moves a flushed record OUT of the predicate must hide
/// the stale flushed copy from a filtered search. The newer WAL version does
/// not match `cat=3`, so a per-record filter skips it, and the older flushed
/// copy (which still matches) must not win the merge.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filtered_search_hides_a_flushed_record_updated_out_of_the_predicate() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = create_filterable_collection(&http, &base, "flt_upd").await;

    upsert(
        &http,
        &base,
        &name,
        json!([
            { "id": "stay", "vector": [0.0, 0.0], "props": { "cat": 3 } },
            { "id": "moved", "vector": [0.0, 0.1], "props": { "cat": 3 } }
        ]),
    )
    .await;
    sleep(Duration::from_millis(500)).await;
    server.force_flush(&name).await.expect("force flush");

    // Unflushed newer version of `moved` with cat=5.
    upsert(
        &http,
        &base,
        &name,
        json!([{ "id": "moved", "vector": [0.0, 0.1], "props": { "cat": 5 } }]),
    )
    .await;
    sleep(Duration::from_millis(300)).await;

    let (got3, body3) = filtered_search_ids(&http, &base, &name, 3).await;
    assert_eq!(
        got3,
        ["stay".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "the stale flushed copy of an updated record must not match the old predicate: {body3}"
    );
    let (got5, body5) = filtered_search_ids(&http, &base, &name, 5).await;
    assert_eq!(
        got5,
        ["moved".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "the new version must match the new predicate: {body5}"
    );
}

/// Guard for any R1/R2 fix: a STALE retained tombstone must not hide a newer
/// flushed re-insert from a filtered search (flush retains tombstone-only
/// batches — TD-USUB-1 — so after delete → re-insert → flush the memtable
/// still holds the old tombstone).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filtered_search_returns_a_record_reinserted_after_delete_and_flush() {
    let server = TestServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.base_url();
    let name = create_filterable_collection(&http, &base, "flt_reins").await;

    upsert(
        &http,
        &base,
        &name,
        json!([{ "id": "phoenix", "vector": [0.0, 0.0], "props": { "cat": 3 } }]),
    )
    .await;
    sleep(Duration::from_millis(300)).await;
    server.force_flush(&name).await.expect("flush 1");

    let delete = http
        .delete(format!("{base}/api/v2/collections/{name}/records/phoenix"))
        .send()
        .await
        .expect("delete");
    assert!(delete.status().is_success(), "delete: {}", delete.status());
    sleep(Duration::from_millis(300)).await;

    upsert(
        &http,
        &base,
        &name,
        json!([{ "id": "phoenix", "vector": [0.0, 0.0], "props": { "cat": 3 } }]),
    )
    .await;
    sleep(Duration::from_millis(300)).await;
    server.force_flush(&name).await.expect("flush 2");

    let (got, body) = filtered_search_ids(&http, &base, &name, 3).await;
    assert_eq!(
        got,
        ["phoenix".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "a record re-inserted after delete and flush must be visible to a filtered search: {body}"
    );
}
