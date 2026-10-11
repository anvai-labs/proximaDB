// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-USUB-2: SQL capability must not depend on physical storage format.
//!
//! Before this, `allow_engine_sql_fallback` (the `ctx.sql` escape hatch for every
//! shape the shared relational frontend declines) was reachable only on the
//! DataFusion route, which required `parquet_backed` — i.e. a manual
//! `ALTER TABLE … MATERIALIZE`. So CTEs, window functions, joins and subqueries
//! worked on materialized tables and failed on native ones, and the error text told
//! the user to go run DDL.
//!
//! This test pins the parity: the **same SQL** must behave the same on a native
//! (never-materialized) table as on a materialized one.
//!
//! The feature is default-OFF (mandate #8), so the test enables
//! `PROXIMADB_NATIVE_TABLE_PROVIDER` for the enabled half and asserts the
//! default-off half still declines — proving the gate actually gates.

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

/// The shapes that were Parquet-only before TD-USUB-2. Each is declined by the
/// shared relational frontend and therefore reachable only through `ctx.sql`.
const CAPABILITY_SHAPES: &[(&str, &str)] = &[
    (
        "cte",
        "with hot as (select id, val from usub2_t where val > 10) select id, val from hot order by id",
    ),
    (
        "window_lag",
        "select id, val, val - lag(val) over (order by id) as delta from usub2_t order by id",
    ),
    (
        "window_rank",
        "select id, rank() over (order by val desc) as r from usub2_t order by r",
    ),
    (
        "scalar_subquery",
        "select id, val from usub2_t where val > (select avg(val) from usub2_t) order by id",
    ),
];

fn explain_err(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

#[test]
fn native_table_capability_parity() {
    std::thread::Builder::new()
        .name("usub2-capability-parity".into())
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
    // --- Phase 1: gate OFF (the shipped default) --------------------------------
    // SAFETY: this integration test is its own process and single-threaded here; the
    // gate is read per-query via `std::env::var`, so toggling it between phases is
    // exactly how an operator would enable the feature.
    unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER") };

    let server = PgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    client
        .simple_query("CREATE TABLE usub2_t (id BIGINT, val BIGINT)")
        .await
        .expect("create");
    client
        .simple_query("INSERT INTO usub2_t (id, val) VALUES (1, 5), (2, 20), (3, 12), (4, 40)")
        .await
        .expect("insert");

    // Baseline sanity: a shape the native path DOES serve must work with the gate off,
    // so a later failure is attributable to the gate and not to a broken fixture.
    let base = client
        .simple_query("SELECT id FROM usub2_t")
        .await
        .expect("plain select must work on a native table regardless of the gate");
    let base_rows = base
        .iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count();
    assert_eq!(base_rows, 4, "fixture should hold 4 rows");

    // With the gate OFF, at least one capability shape must still be Parquet-only —
    // otherwise this test proves nothing about the gate. We do NOT assert a specific
    // SQLSTATE: the proximate decline belongs to a shared guard this change does not own.
    let mut declined_with_gate_off = 0usize;
    for (_label, sql) in CAPABILITY_SHAPES {
        if client.simple_query(sql).await.is_err() {
            declined_with_gate_off += 1;
        }
    }
    assert!(
        declined_with_gate_off > 0,
        "at least one capability shape must be Parquet-only with the gate OFF — \
         otherwise this test proves nothing about the gate"
    );

    // KNOWN-BAD, pinned deliberately (ADR-040 pattern).
    //
    // A bare window function over a NATIVE table does not merely decline with the gate
    // off — it returns a **silently wrong answer**: `val - lag(val) OVER (...)` comes
    // back as `val` for every row, because `lag(...)` contributes nothing.
    //
    // This is the TD-187 defect, which was fixed only for *materialized* tables. The
    // TD-187 engagement fix makes the query engage the relational pipeline, but with no
    // DataFusion route available for a native table `try_run_select` returns `None` and
    // the query falls through to the legacy single-table path, whose projection
    // extractor has no expression parser and re-reads the bare `val` column.
    //
    // So TD-USUB-2 closes a **correctness** bug on this path, not just a capability gap
    // (mandate #1: fail closed, never silently wrong). When the legacy path is fixed or
    // the gate becomes default-ON, this assertion trips and should be deleted.
    let known_bad = client
        .simple_query(
            "select id, val - lag(val) over (order by id) as delta from usub2_t order by id",
        )
        .await
        .expect("legacy path currently serves this (wrongly) rather than declining");
    let bad_deltas: Vec<String> = known_bad
        .iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                Some(r.get(1).unwrap_or("NULL").trim().to_string())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        bad_deltas,
        vec!["5", "20", "12", "40"],
        "KNOWN-BAD (TD-187 residual on the native path): with the gate OFF, delta must \
         still equal val — if this changed, the legacy path was fixed and this pin \
         should be removed"
    );

    drop(client);
    drop(server);

    // --- Phase 2: gate ON -------------------------------------------------------
    // SAFETY: see phase 1.
    unsafe { std::env::set_var("PROXIMADB_NATIVE_TABLE_PROVIDER", "1") };

    let server = PgServer::start().await.expect("server 2");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect 2");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    client
        .simple_query("CREATE TABLE usub2_t (id BIGINT, val BIGINT)")
        .await
        .expect("create 2");
    client
        .simple_query("INSERT INTO usub2_t (id, val) VALUES (1, 5), (2, 20), (3, 12), (4, 40)")
        .await
        .expect("insert 2");

    // NOTE: no `ALTER TABLE … MATERIALIZE` anywhere in this test. That is the point.
    for (label, sql) in CAPABILITY_SHAPES {
        let rows = client.simple_query(sql).await.unwrap_or_else(|e| {
            panic!(
                "`{label}` must serve on a NATIVE table: {}",
                explain_err(&e)
            )
        });
        let n = rows
            .iter()
            .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
            .count();
        assert!(n > 0, "`{label}` returned no rows on the native table");
    }

    // Value-correctness, not just "it executed" (ADR-040): LAG must produce the real
    // previous-row delta and NULL for the first row — the same bug class TD-187 fixed
    // on the materialized path. Rows are ordered by id: 5, 20, 12, 40.
    let msgs = client
        .simple_query(
            "select id, val - lag(val) over (order by id) as delta from usub2_t order by id",
        )
        .await
        .expect("lag delta");
    let deltas: Vec<Option<String>> = msgs
        .iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                Some(r.get(1).map(|s| s.trim().to_string()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(deltas.len(), 4, "expected 4 delta rows, got {deltas:?}");
    assert!(
        deltas[0].is_none() || deltas[0].as_deref() == Some(""),
        "first row has no predecessor → delta must be NULL, got {:?}",
        deltas[0]
    );
    let numeric: Vec<i64> = deltas[1..]
        .iter()
        .map(|d| {
            d.as_deref()
                .unwrap_or_default()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric delta {d:?}")) as i64
        })
        .collect();
    assert_eq!(
        numeric,
        vec![15, -8, 28],
        "LAG deltas must be the real previous-row differences (20-5, 12-20, 40-12)"
    );

    // SAFETY: restore the default so the process leaves no gate set.
    unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER") };
}
