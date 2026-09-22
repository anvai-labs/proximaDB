// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-USUB-4: publication must be **additive**, and re-publication idempotent.
//!
//! `materialize_table_to_parquet` published with
//! `set_storage_layouts(&table_id, vec![layout])`, which replaced the table's whole
//! layout list. A table could therefore hold exactly one physical representation,
//! which is what made cost-based routing across representations *inexpressible* —
//! the router cannot choose among alternatives the catalog has no way to record
//! (ADR-094 Decision 2).
//!
//! The merge semantics (preserve other layouts, replace same-named in place, stable
//! order) are asserted directly as a unit test on `upsert_storage_layout`. What this
//! e2e adds is the end-to-end property a user can actually hit: **materializing more
//! than once must stay correct** — no duplicated layout, no lost routing, no wrong
//! rows — through the real catalog, DDL and query paths.

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
        // TD-CONV-1: anchor the system catalog in this server's temp dir, not the
        // shared CWD-relative `file://./metadata` default (cross-run DDL replay).
        config.storage.metadata_url = format!("file://{}", tmp.path().join("metadata").display());
        let mut db = ProximaDB::new(config).await?;
        db.start().await?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
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

fn explain_err(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

async fn scalar(client: &tokio_postgres::Client, sql: &str) -> String {
    let msgs = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}`: {}", explain_err(&e)));
    for m in msgs {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            return r.get(0).unwrap_or("NULL").trim().to_string();
        }
    }
    panic!("`{sql}` returned no rows");
}

#[test]
fn repeat_materialize_is_idempotent_and_preserves_routing() {
    std::thread::Builder::new()
        .name("usub4-additive-layouts".into())
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

    client
        .simple_query("CREATE TABLE usub4_t (id BIGINT, val BIGINT)")
        .await
        .expect("create");
    client
        .simple_query("INSERT INTO usub4_t (id, val) VALUES (1, 5), (2, 20), (3, 12), (4, 40)")
        .await
        .expect("insert");

    // Pre-materialize baseline, so a later difference is attributable to publication.
    assert_eq!(scalar(&client, "SELECT COUNT(*) FROM usub4_t").await, "4");

    // First publication — records the `parquet-snapshot` layout.
    client
        .simple_query("ALTER TABLE usub4_t MATERIALIZE")
        .await
        .expect("first materialize");
    assert_eq!(
        scalar(&client, "SELECT COUNT(*) FROM usub4_t").await,
        "4",
        "the published table must read correctly"
    );
    assert_eq!(
        scalar(&client, "SELECT SUM(val) FROM usub4_t").await,
        "77",
        "values must survive publication (5+20+12+40)"
    );

    // Second publication of the SAME layout name. Under the old clobbering write this
    // replaced the list wholesale; under the additive upsert it must replace the
    // same-named layout **in place** — no duplicate entry, no lost routing.
    client
        .simple_query("ALTER TABLE usub4_t MATERIALIZE")
        .await
        .expect("second materialize must succeed");
    assert_eq!(
        scalar(&client, "SELECT COUNT(*) FROM usub4_t").await,
        "4",
        "re-publication must not duplicate or drop rows"
    );
    assert_eq!(
        scalar(&client, "SELECT SUM(val) FROM usub4_t").await,
        "77",
        "re-publication must not corrupt values"
    );

    // Staleness, asserted as it actually behaves rather than as one might assume.
    //
    // A write landing AFTER publication is NOT visible over the published base by
    // default: the ADR-025 read-merge that would reconcile the snapshot against the
    // post-`snapshot_lsn` delta is a per-table opt-in, so a plain materialized table
    // serves the snapshot as-of publication.
    //
    // This is pre-existing behaviour, NOT introduced here — with no prior layouts the
    // additive upsert produces the same single-element list the old `vec![layout]`
    // write did, so this path is unchanged by TD-USUB-4. It is pinned because ADR-094
    // names publication staleness as a live design constraint (it is one of the three
    // reasons "land in PAX, publish Parquet" was rejected as the architecture), and a
    // silent change here should trip a test rather than surprise someone.
    client
        .simple_query("INSERT INTO usub4_t (id, val) VALUES (5, 100)")
        .await
        .expect("post-publication insert");
    assert_eq!(
        scalar(&client, "SELECT COUNT(*) FROM usub4_t").await,
        "4",
        "documented staleness: a published table serves its snapshot until re-published \
         (the delta merge is a per-table opt-in). If this becomes 5, the merge default \
         changed — update this pin deliberately."
    );

    // Re-publishing folds the delta into a fresh snapshot — and, with the additive
    // upsert, does so by replacing the same-named layout in place rather than by
    // discarding the list.
    client
        .simple_query("ALTER TABLE usub4_t MATERIALIZE")
        .await
        .expect("third materialize");
    assert_eq!(
        scalar(&client, "SELECT COUNT(*) FROM usub4_t").await,
        "5",
        "re-publication must capture the post-snapshot write"
    );
    assert_eq!(
        scalar(&client, "SELECT SUM(val) FROM usub4_t").await,
        "177",
        "re-publication must capture post-snapshot values (77 + 100)"
    );
}
