// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-5 slice 1 (TPC-H Q2) — a correlated scalar aggregate whose body
//! is MULTI-TABLE (`min(cost) FROM ps, s WHERE ps_sk = sk AND region = 'EU' AND
//! ps_pk = p.pk`) must decorrelate and execute on the native route. Previously
//! declined (`correlated scalar subquery`): only a single-table subquery body
//! lowered. Proves the inner join conditions + inner-local filters fold into the
//! inner Filter while the one correlation equi becomes the group/join key,
//! row-exact.

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
fn native_multi_table_correlated_scalar_aggregate() {
    std::thread::Builder::new()
        .name("rel-corr-scalar-mt-e2e-16m".into())
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
        "DROP TABLE IF EXISTS ps",
        "DROP TABLE IF EXISTS s",
        "DROP TABLE IF EXISTS p",
        "CREATE TABLE p (pk INT PRIMARY KEY, budget INT)",
        "CREATE TABLE s (sk INT PRIMARY KEY, region VARCHAR)",
        "CREATE TABLE ps (psid INT PRIMARY KEY, ps_pk INT, ps_sk INT, cost INT)",
        // p.budget is the target min-supplycost each part is compared against.
        "INSERT INTO p VALUES (1,10),(2,5),(3,99)",
        // suppliers: sk=1 in EU, sk=2 in US (US is filtered OUT of the subquery).
        "INSERT INTO s VALUES (1,'EU'),(2,'US')",
        // partsupp (ps_pk, ps_sk, cost):
        //   pk=1: (sk1,cost10 EU),(sk2,cost20 US)  → min over EU = 10
        //   pk=2: (sk1,cost5  EU),(sk2,cost3  US)  → min over EU = 5  (US 3 excluded)
        //   pk=3: (sk2,cost7  US)                  → no EU row → min = NULL
        "INSERT INTO ps VALUES (100,1,1,10),(101,1,2,20),(102,2,1,5),(103,2,2,3),(104,3,2,7)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q2 shape: keep parts whose budget equals the MIN cost across the joined
    // (ps ⋈ s, region='EU') for that part — correlated on ps_pk = p.pk. The
    // subquery body is MULTI-TABLE (ps, s); the inner join (ps_sk = sk) and
    // inner-local filter (region = 'EU') fold into the inner Filter, the one
    // correlation equi becomes the group/join key.
    //   pk=1: 10 == min{10} ✓   pk=2: 5 == min{5} ✓   pk=3: 99 == NULL ✗
    let mut got = rows(
        &client,
        "SELECT pk FROM p WHERE budget = \
         (SELECT min(cost) FROM ps, s WHERE ps_sk = sk AND region = 'EU' AND ps_pk = pk) \
         ORDER BY pk",
    )
    .await;
    let pks: Vec<i64> = got
        .drain(..)
        .map(|r| r[0].parse::<i64>().expect("i64"))
        .collect();
    assert_eq!(
        pks,
        vec![1, 2],
        "only parts whose budget equals the min EU supplycost qualify (pk=3's group is empty → NULL)"
    );

    eprintln!(
        "✓ TD-REL-LOWER-5 multi-table correlated scalar aggregate (Q2 shape) on native route"
    );
}
