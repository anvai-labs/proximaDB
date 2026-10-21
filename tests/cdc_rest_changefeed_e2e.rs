//! CDC change-feed REST surface: write over pgwire, read the changes over REST.
//!
//! Proves the unified change-feed surface — `GET /api/v2/collections/:id/changes?since_lsn=N`
//! reflects writes made on a DIFFERENT protocol (pgwire), because every surface now shares one
//! canonical record store. Also exercises the `since_lsn` cursor.
//!
//! The pgwire database names the tenant (`proximadb` here), and the feed is
//! tenant-scoped, so the REST reads act as that same tenant.
//!
//!   cargo test --test cdc_rest_changefeed_e2e -- --nocapture

use std::time::Duration;

use serde_json::Value;
use tokio::time::sleep;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_writes_are_visible_on_the_rest_change_feed() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("connect");

    // Write over pgwire.
    let _ = client.simple_query("DROP TABLE IF EXISTS crfeed").await;
    client
        .simple_query("CREATE TABLE crfeed (id INT PRIMARY KEY, bal INT)")
        .await
        .expect("create");
    client
        .simple_query("INSERT INTO crfeed (id, bal) VALUES (1, 10), (2, 20)")
        .await
        .expect("insert");
    client
        .simple_query("UPDATE crfeed SET bal = 15 WHERE id = 1")
        .await
        .expect("update");
    client
        .simple_query("DELETE FROM crfeed WHERE id = 2")
        .await
        .expect("delete");
    sleep(Duration::from_millis(200)).await;

    // Read the change-feed over REST — a DIFFERENT protocol than the writes.
    let (http, base) = server.rest_client().expect("rest client");
    let resp = http
        .get(format!(
            "{base}/api/v2/collections/crfeed/changes?since_lsn=0"
        ))
        .header("X-Tenant-ID", "proximadb")
        .send()
        .await
        .expect("changes GET");
    assert!(
        resp.status().is_success(),
        "changes endpoint status: {}",
        resp.status()
    );
    let body: Value = resp.json().await.expect("changes json");
    let changes = body["changes"].as_array().expect("changes array");

    // insert(1), insert(2), update(1)→upsert, delete(2) → 4 change rows, lsn-ordered.
    assert_eq!(
        changes.len(),
        4,
        "pgwire INSERT×2 + UPDATE + DELETE must surface on the REST feed; body={body}"
    );
    let ops: Vec<&str> = changes.iter().filter_map(|c| c["op"].as_str()).collect();
    assert_eq!(
        ops,
        vec!["upsert", "upsert", "upsert", "delete"],
        "op sequence"
    );
    assert_eq!(
        changes.last().unwrap()["key"],
        "2",
        "delete carries the key"
    );
    let lsns: Vec<u64> = changes.iter().filter_map(|c| c["lsn"].as_u64()).collect();
    assert!(
        lsns.windows(2).all(|w| w[0] < w[1]),
        "lsn-ordered: {lsns:?}"
    );

    // Cursor: since_lsn = first change's lsn → only the 3 later changes.
    let resp2 = http
        .get(format!(
            "{base}/api/v2/collections/crfeed/changes?since_lsn={}",
            lsns[0]
        ))
        .header("X-Tenant-ID", "proximadb")
        .send()
        .await
        .expect("changes GET 2");
    let body2: Value = resp2.json().await.expect("changes json 2");
    assert_eq!(
        body2["changes"].as_array().unwrap().len(),
        3,
        "since_lsn cursor advances past the first change"
    );
    eprintln!("✓ CDC REST change-feed reflects pgwire writes (unified surface) + cursor works");
}
