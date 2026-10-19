// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-3 (ORDER-BY-alias case, q3/q52) — a GROUP BY key projected with
//! an alias and used as an ORDER BY key must execute on the native route, not
//! error `rebind_columns: column '<alias>' not in narrowed schema`.
//!
//! Root cause (fixed): the frontend put the projection ALIAS on the inner
//! `ColumnRef` of a grouped-key output instead of the group-key's source name,
//! so the planner's name-based rebind could not resolve it.

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
async fn rows(client: &tokio_postgres::Client, sql: &str) -> Vec<String> {
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
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            _ => None,
        })
        .collect()
}

#[test]
fn native_orderby_alias_of_group_key() {
    std::thread::Builder::new()
        .name("rel-lower3-e2e-16m".into())
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
        "DROP TABLE IF EXISTS sales",
        "CREATE TABLE sales (id INT PRIMARY KEY, brand VARCHAR, amt INT)",
        "INSERT INTO sales VALUES (1,'A',10),(2,'A',20),(3,'B',5),(4,'C',30),(5,'C',30)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // GROUP-BY key aliased AND used as an ORDER BY key (q3/q52 shape). Ordered
    // by the alias, so the result order is deterministic.
    let got = rows(
        &client,
        "SELECT brand AS b, sum(amt) AS total FROM sales GROUP BY brand ORDER BY total DESC, b",
    )
    .await;
    // total DESC: C=60, A=30, B=5.
    assert_eq!(
        got,
        vec!["C|60".to_string(), "A|30".to_string(), "B|5".to_string()],
        "ORDER BY on a group-key alias returned wrong/erroring result"
    );

    // The pure alias-as-only-sort-key form.
    let got2 = rows(
        &client,
        "SELECT brand AS b, count(*) AS c FROM sales GROUP BY brand ORDER BY b",
    )
    .await;
    assert_eq!(
        got2,
        vec!["A|2".to_string(), "B|1".to_string(), "C|2".to_string()]
    );
    eprintln!("✓ TD-REL-LOWER-3 ORDER-BY-alias of a group key executes correctly");
}
