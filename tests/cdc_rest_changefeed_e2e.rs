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

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use serde_json::Value;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

struct Server {
    pg_port: u16,
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Server {
    async fn start() -> anyhow::Result<Self> {
        let pg_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
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
        let mut db = ProximaDB::new(config).await?;
        db.start().await?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
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
            pg_port,
            rest_port,
            db: Some(db),
            _tmp: tmp,
        })
    }
    fn conn_str(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }
    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.rest_port)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_writes_are_visible_on_the_rest_change_feed() {
    let server = Server::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

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
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .unwrap();
    let resp = http
        .get(format!(
            "{}/api/v2/collections/crfeed/changes?since_lsn=0",
            server.base()
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
            "{}/api/v2/collections/crfeed/changes?since_lsn={}",
            server.base(),
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
