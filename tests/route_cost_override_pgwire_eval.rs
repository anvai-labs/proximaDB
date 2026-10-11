//! TD-ROUTE-1 — pgwire correctness eval for the trace-driven route cost override.
//!
//! The `RouteCostModel` can, when `PROXIMADB_ROUTE_COST_OVERRIDE` is on, flip an
//! `olap/parquet` query between the DataFusion and Native (Volcano) backends
//! (the two freshness-safe candidates). Enabling that override is only safe if a
//! flipped route returns the **same results** as the static (override-OFF)
//! baseline. This eval is the gate for that claim (mandate #13): it materializes
//! a small dataset (parquet-backed ⇒ `olap/parquet`), runs a set of queries with
//! the override OFF and captures results, warms the model so the freshness-safe
//! challenger is the cheaper arm, enables the override, re-runs, and asserts
//! **row parity**. A divergence fails the test — which correctly keeps the live
//! override gated OFF until it is fixed.
//!
//! Complements `route_cost_offline_eval.rs` (which proves the flip *logic* on a
//! synthetic isolated model); this proves *result correctness* end-to-end over
//! the real pgwire + router + engine path.
//!
//!   RUST_LOG=proximadb=debug cargo test --test route_cost_override_pgwire_eval -- --nocapture

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::observability::io_trace::IoTraceSnapshot;
use proximadb::query::route_cost_model::GLOBAL_ROUTE_COST_MODEL;
use proximadb::query::table_write_plan::ComputeBackend;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

/// Minimal in-process pgwire server (mirrors `tpch_pgwire_e2e::PgServer`).
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

/// Run a query and return its result rows as sorted `col|col|…` strings (sorted
/// so the comparison is order-insensitive where a query has no ORDER BY). Returns
/// `Err` with the server-side cause if the query fails, so a route flip to a
/// backend that cannot execute the query is observed rather than panicking.
async fn query_rows(client: &tokio_postgres::Client, sql: &str) -> Result<Vec<String>, String> {
    let msgs = client.simple_query(sql).await.map_err(|e| {
        e.as_db_error()
            .map(|d| format!("[{}] {}", d.code().code(), d.message()))
            .unwrap_or_else(|| e.to_string())
    })?;
    let mut rows = Vec::new();
    for m in msgs {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            let cols: Vec<String> = (0..r.len())
                .map(|i| r.get(i).unwrap_or("").to_string())
                .collect();
            rows.push(cols.join("|"));
        }
    }
    rows.sort();
    Ok(rows)
}

/// TD-ROUTE-2 harness pin: the pgwire parse→lower→plan→execute path overflows
/// the default test-thread stack in the dev profile (pre-existing on develop —
/// verified 2026-07-13; passes at 16 MiB). Until the recursion depth is
/// bisected and fixed at the source, run the eval body on a dedicated 16 MiB
/// thread with a current-thread runtime (same semantics as `#[tokio::test]`)
/// so the TD-ROUTE-1 enable gate runs at stock settings. Remove this wrapper
/// when TD-ROUTE-2's root fix lands.
#[test]
fn override_on_preserves_materialized_query_results() {
    std::thread::Builder::new()
        .name("route-override-eval-16m".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(override_eval_body())
        })
        .expect("spawn eval thread")
        .join()
        .expect("eval thread panicked");
}

async fn override_eval_body() {
    let server = PgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // 1. Small schema + deterministic data.
    for ddl in [
        "DROP TABLE IF EXISTS lineitem",
        "DROP TABLE IF EXISTS orders",
        "CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice DOUBLE PRECISION, o_orderstatus VARCHAR)",
        "CREATE TABLE lineitem (l_orderkey INT, l_quantity DOUBLE PRECISION, l_extendedprice DOUBLE PRECISION)",
    ] {
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    for ins in [
        "INSERT INTO orders VALUES (1,10,100.0,'O'),(2,20,200.0,'F'),(3,10,50.0,'O'),(4,30,300.0,'P'),(5,20,150.0,'F')",
        "INSERT INTO lineitem VALUES (1,5.0,50.0),(1,2.0,20.0),(2,3.0,60.0),(3,1.0,10.0),(4,4.0,80.0),(5,2.0,30.0)",
    ] {
        client
            .simple_query(ins)
            .await
            .unwrap_or_else(|e| panic!("insert: {e}"));
    }

    // 2. Materialize to Parquet so SELECTs route to the DataFusion OLAP engine
    //    (shape-class `olap/parquet`, the only override-eligible class).
    for t in ["orders", "lineitem"] {
        if let Err(e) = client
            .simple_query(&format!("ALTER TABLE {t} MATERIALIZE"))
            .await
        {
            eprintln!("  · MATERIALIZE {t}: {e}");
        }
    }

    // Scan/aggregate queries — Native (Volcano) can evaluate these, so a route
    // flip must preserve their results.
    let safe_queries = [
        "SELECT count(*) FROM orders",
        "SELECT o_orderstatus, count(*) FROM orders GROUP BY o_orderstatus ORDER BY o_orderstatus",
        "SELECT sum(l_extendedprice) FROM lineitem",
    ];
    // A join — the eval's boundary case. Native is a freshness-safe candidate for
    // `olap/parquet` but cannot execute joins today, so flipping this one is
    // unsafe until the candidate set excludes Volcano for join-bearing plans.
    let join_query = "SELECT o.o_orderstatus, sum(l.l_extendedprice) FROM orders o, lineitem l WHERE o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus";

    // 3. Baseline (override OFF): warm the model, then capture results.
    GLOBAL_ROUTE_COST_MODEL.set_override_enabled(false);
    for _ in 0..8 {
        for q in safe_queries {
            let _ = query_rows(&client, q).await;
        }
        let _ = query_rows(&client, join_query).await;
    }
    let mut baseline = Vec::new();
    for q in safe_queries {
        baseline.push(
            query_rows(&client, q)
                .await
                .expect("baseline scan/aggregate (DataFusion) must succeed"),
        );
    }
    let join_baseline = query_rows(&client, join_query)
        .await
        .expect("baseline join (DataFusion) must succeed");

    // 4. Warm the freshness-safe challenger (Native) to be the cheaper arm on
    //    every learned `olap/parquet` class, so the override flips those routes
    //    DataFusion → Native.
    let cheap = IoTraceSnapshot {
        range_gets: 1,
        bytes_read: 64,
        ..Default::default()
    };
    let mut olap_classes = 0;
    for (class, _backend) in GLOBAL_ROUTE_COST_MODEL.learned_cell_keys() {
        if class.starts_with("olap/parquet") {
            for _ in 0..40 {
                GLOBAL_ROUTE_COST_MODEL.observe(&class, &ComputeBackend::Native, &cheap);
            }
            olap_classes += 1;
        }
    }
    eprintln!("✓ warmed the Native challenger on {olap_classes} olap/parquet class(es)");
    assert!(
        olap_classes > 0,
        "no olap/parquet classes were learned — materialize/routing changed?"
    );

    // 5. Enable the override (recomputes the frozen consult table).
    GLOBAL_ROUTE_COST_MODEL.set_override_enabled(true);
    assert!(
        GLOBAL_ROUTE_COST_MODEL.override_active(),
        "override did not activate"
    );

    // 6a. Scan/aggregate results MUST match the OFF baseline. A flip that fails
    //     or changes a scan/aggregate answer is a correctness regression.
    for (i, q) in safe_queries.iter().enumerate() {
        let got = query_rows(&client, q).await.unwrap_or_else(|e| {
            panic!("override flipped a scan/aggregate query to a backend that FAILED (unsafe):\n  {q}\n  {e}")
        });
        assert_eq!(
            got, baseline[i],
            "route cost override changed scan/aggregate results (unsafe):\n  {q}\n  baseline={:?} override={:?}",
            baseline[i], got
        );
    }

    // 6b. TD-ROUTE-1 capability gate (the former documented boundary): the
    //     `QueryShape::join_bearing` eligibility bit keeps join-bearing plans
    //     off Native under the override, so the join MUST now execute and
    //     match the baseline — a strict parity assert. This is the enable-gate
    //     for the global override. (Result parity alone also holds via the
    //     #853 DataFusion floor, so 6c below asserts the DECISION level too.)
    let got = query_rows(&client, join_query).await.expect(
        "join under the override must not error — the TD-ROUTE-1 capability gate keeps \
         join-bearing plans off the Native arm (no multi-table native-over-parquet scan)",
    );
    assert_eq!(
        got, join_baseline,
        "join under the override diverged from the DataFusion baseline:\n  baseline={:?} override={:?}",
        join_baseline, got
    );

    // 6c. Decision-level gate assert (review follow-up of #923): with the
    //     WARMED global model and the override ON, a join-bearing olap/parquet
    //     shape must keep the static DataFusion route — eligibility, not cost,
    //     decides. Without this, result parity alone would pass even with the
    //     gate reverted (the DataFusion floor serves the query), hiding a
    //     regression that still poisons Native's cost cells via the io_trace.
    let decision = proximadb::query::compute_scheduler::ComputeScheduler::new()
        .route_select_advised(
            proximadb::query::compute_scheduler::QueryShape {
                engages_relational: true,
                parquet_backed: true,
                join_bearing: true,
                ..Default::default()
            },
            Some(&GLOBAL_ROUTE_COST_MODEL),
            proximadb::query::compute_scheduler::RouteFlags::default(),
        );
    assert_eq!(
        decision.backend,
        ComputeBackend::DataFusionLocal,
        "join-bearing olap/parquet shape flipped off DataFusion under the override: {}",
        decision.reason
    );
    assert!(
        !decision.reason.contains("OVERRIDE") && !decision.reason.contains("EXPLORE"),
        "join-bearing shape must not carry an override/explore flip, got: {}",
        decision.reason
    );
}
