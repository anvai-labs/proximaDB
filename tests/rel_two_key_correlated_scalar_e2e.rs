// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-5 slice 3 (TPC-H Q20 root cause) — a correlated scalar aggregate
//! with TWO correlation keys (`WHERE lpk = pk AND lsk = sk`) must decorrelate and
//! execute on the native route. Previously declined at the single-key MVP guard
//! (`correlated scalar subquery`), which — masked upstream by the outer `IN`'s
//! fallback — was the true blocker behind Q20's "IN in expression position"
//! decline. Proves GROUP BY on both keys + a LEFT JOIN ON both keys, row-exact.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}
struct PgServer {
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}
impl PgServer {
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
        // Anchor the system catalog to this test's temp dir: the CWD-relative
        // default shares one catalog across concurrently running nextest
        // processes, racing same-named DDL (TD-CONV-1).
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
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
}
impl Drop for PgServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}
async fn rows(client: &tokio_postgres::Client, sql: &str) -> Vec<Vec<String>> {
    let msgs = client.simple_query(sql).await.unwrap_or_else(|e| {
        panic!(
            "{sql}\n  {}",
            e.as_db_error()
                .map(|d| format!("[{}] {}", d.code().code(), d.message()))
                .unwrap_or_else(|| e.to_string())
        )
    });
    msgs.iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).unwrap_or("").to_string())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect()
}

#[test]
fn native_two_key_correlated_scalar_aggregate() {
    std::thread::Builder::new()
        .name("rel-2key-corr-scalar-e2e-16m".into())
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

async fn body() {
    let server = PgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    for ddl in [
        "DROP TABLE IF EXISTS line",
        "DROP TABLE IF EXISTS ps",
        "CREATE TABLE ps (psid INT PRIMARY KEY, pk INT, sk INT, availqty INT)",
        "CREATE TABLE line (lid INT PRIMARY KEY, lpk INT, lsk INT, qty INT)",
        "INSERT INTO ps VALUES (1,1,1,100),(2,1,2,10),(3,2,1,50)",
        // per (pk,sk): 0.5*sum(qty) threshold; keep ps rows with availqty > threshold
        //   (1,1): qty 40+40 = 80 → 0.5*80 = 40 ; availqty 100 > 40  ✓
        //   (1,2): qty 30      = 30 → 0.5*30 = 15 ; availqty 10  > 15 ✗
        //   (2,1): qty 10      = 10 → 0.5*10 = 5  ; availqty 50  > 5  ✓
        "INSERT INTO line VALUES (10,1,1,40),(11,1,1,40),(12,1,2,30),(13,2,1,10)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q20-shape inner: a correlated scalar aggregate correlated on TWO keys
    // (lpk = pk AND lsk = sk). The decorrelation groups line by (lpk, lsk) and
    // LEFT JOINs ps on both keys.
    let mut got = rows(
        &client,
        "SELECT pk, sk FROM ps \
         WHERE availqty > (SELECT 0.5 * sum(qty) FROM line WHERE lpk = pk AND lsk = sk) \
         ORDER BY pk, sk",
    )
    .await;
    let pairs: Vec<(i64, i64)> = got
        .drain(..)
        .map(|r| (r[0].parse().expect("pk"), r[1].parse().expect("sk")))
        .collect();
    assert_eq!(
        pairs,
        vec![(1, 1), (2, 1)],
        "only (pk,sk) whose availqty exceeds half its per-(pk,sk) qty total qualify"
    );

    eprintln!(
        "✓ TD-REL-LOWER-5 two-key correlated scalar aggregate (Q20 root cause) on native route"
    );
}
