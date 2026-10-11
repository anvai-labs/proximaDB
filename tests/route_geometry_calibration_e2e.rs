// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-EXEC-2 §3 calibration-accuracy RATCHET — the "predicted-vs-actual error
//! becomes a gated ratchet … a regression in prediction fails CI, exactly like
//! the conformance ratchets" gate, made executable.
//!
//! The routing key carries an AST-estimated geometry band (`geom=<depth>x<blocking>`,
//! stamped by `query_geometry_class` in the pgwire pipeline because routing
//! precedes planning); at fold time the observe→ingest seam compares that
//! estimate against the MEASURED plan geometry the io_trace carries and emits
//! `proximadb_route_geometry_estimate_total{estimated,measured}`. This test
//! drives a fixed, geometry-diverse SQL battery through the REAL pgwire path on
//! BOTH route seams — Volcano/native (pre-MATERIALIZE, `plan_instrumented`) and
//! DataFusion (post-MATERIALIZE, the #931 logical-plan walk) — then reads the
//! counter and ratchets two invariants:
//!
//! 1. **The calibration loop is LIVE on both seams.** Each phase must add
//!    samples. This falsifies "measurement went dark" regressions — the exact
//!    failure #931 fixed (the DataFusion route emitted no measured geometry, so
//!    the estimator was unaudited on the dominant OLAP path).
//! 2. **Divergence rate ≤ the ratchet.** Divergence = fraction of samples where
//!    the estimated band differs from the measured band. The estimator-v2 refit
//!    (#942) measured 8.2% across the full TPC-H/TPC-DS ledger sweep; this
//!    battery is chosen to calibrate cleanly today, so the bound is tight. The
//!    threshold may only ever move DOWN (ratchet); an estimator or measurement
//!    change that pushes divergence above it fails CI and needs a band refit
//!    (TD-EXEC-2 §2 evidence pattern), not a threshold bump.
//!
//!   RUST_LOG=proximadb=debug cargo test --test route_geometry_calibration_e2e -- --nocapture

use std::net::TcpListener;
use std::time::Duration;

use prometheus::core::Collector;
use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

/// The calibration-accuracy ratchet: the divergence rate over the battery must
/// stay at or below this. Move it DOWN as the estimator improves; never up.
/// (Full-ledger context: 8.2% after the #942 v2 refit; the named remainder —
/// q61/q65/q67 group-by wrappers, Q4/Q17 one band shy — is not in this battery,
/// which pins the estimator's known-good surface against regression.)
const MAX_DIVERGENCE_RATE: f64 = 0.10;

/// Every battery query must land at least this many calibration samples per
/// phase in total — guards against the loop silently going dark (a query that
/// stops stamping `geom=` or a seam that stops measuring emits nothing, which
/// without this floor would read as 0% divergence).
const MIN_SAMPLES_PER_PHASE: f64 = 8.0;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

/// Minimal in-process pgwire server (mirrors `route_cost_override_pgwire_eval`).
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

/// Snapshot the `route_geometry_estimate_total` counter as
/// `(estimated, measured) → count` pairs from the in-process registry.
fn geometry_samples() -> Vec<(String, String, f64)> {
    proximadb::metrics::route_metrics::ROUTE_GEOMETRY_ESTIMATE_TOTAL
        .collect()
        .into_iter()
        .flat_map(|mf| mf.get_metric().to_owned())
        .map(|m| {
            let mut estimated = String::new();
            let mut measured = String::new();
            for lp in m.get_label() {
                match lp.name() {
                    "estimated" => estimated = lp.value().to_string(),
                    "measured" => measured = lp.value().to_string(),
                    _ => {}
                }
            }
            (estimated, measured, m.get_counter().value())
        })
        .collect()
}

/// The pgwire client surfaces failures as a generic `db error`; the real cause
/// is the server-side `DbError` code/message.
fn db_error_detail(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

fn total_samples(samples: &[(String, String, f64)]) -> f64 {
    samples.iter().map(|(_, _, c)| c).sum()
}

fn divergent_samples(samples: &[(String, String, f64)]) -> f64 {
    samples
        .iter()
        .filter(|(e, m, _)| e != m)
        .map(|(_, _, c)| c)
        .sum::<f64>()
        .abs() // an empty sum is -0.0 — normalize for display
}

/// Geometry-diverse single-table battery: scalar aggregates (shallow /
/// low-blocking), grouped aggregates, ORDER BY/LIMIT wrappers, HAVING. Every
/// query engages the relational path so the route stamps a `geom=` band; each
/// executes on BOTH the native (Volcano) and DataFusion routes.
const CORE_BATTERY: &[&str] = &[
    "SELECT count(*) FROM orders",
    "SELECT sum(l_extendedprice) FROM lineitem",
    "SELECT count(*) FROM orders WHERE o_totalprice > 100",
    "SELECT o_orderstatus, count(*) FROM orders GROUP BY o_orderstatus",
    "SELECT o_custkey, sum(o_totalprice) FROM orders GROUP BY o_custkey ORDER BY 2 DESC LIMIT 3",
    "SELECT o_custkey, count(*) AS c FROM orders GROUP BY o_custkey HAVING c > 1",
];

/// Join / predicate-subquery shapes (deeper geometry, the v2 hoisted-branch
/// terms). Run on BOTH seams since the TD-REL-LOWER-1 fix (qualified ORDER BY
/// keys over aliased joins no longer decline to the legacy single-table path).
const JOIN_BATTERY: &[&str] = &[
    "SELECT o.o_orderstatus, sum(l.l_extendedprice) FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus",
    "SELECT count(*) FROM orders WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem WHERE l_quantity > 2.0)",
    "SELECT avg(t.s) FROM (SELECT o_custkey, sum(o_totalprice) AS s FROM orders GROUP BY o_custkey) t",
];

/// TD-ROUTE-2 harness pin (same as `route_cost_override_pgwire_eval`): the
/// pgwire parse→lower→plan→execute path overflows the default test-thread
/// stack in the dev profile; run on a dedicated 16 MiB thread until the
/// TD-ROUTE-2 root fix lands.
#[test]
fn geometry_estimator_divergence_stays_within_ratchet() {
    std::thread::Builder::new()
        .name("geom-calibration-16m".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(calibration_body())
        })
        .expect("spawn eval thread")
        .join()
        .expect("eval thread panicked");
}

async fn calibration_body() {
    let server = PgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    for ddl in [
        "DROP TABLE IF EXISTS lineitem",
        "DROP TABLE IF EXISTS orders",
        "CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice DOUBLE PRECISION, o_orderstatus VARCHAR)",
        "CREATE TABLE lineitem (l_orderkey INT, l_quantity DOUBLE PRECISION, l_extendedprice DOUBLE PRECISION)",
        "INSERT INTO orders VALUES (1,10,100.0,'O'),(2,20,200.0,'F'),(3,10,50.0,'O'),(4,30,300.0,'P'),(5,20,150.0,'F')",
        "INSERT INTO lineitem VALUES (1,5.0,50.0),(1,2.0,20.0),(2,3.0,60.0),(3,1.0,10.0),(4,4.0,80.0),(5,2.0,30.0)",
    ] {
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("{ddl}: {}", db_error_detail(&e)));
    }

    // Run a battery on one route seam; every query must EXECUTE (an erroring
    // query emits no calibration sample, silently shrinking coverage).
    let run_battery = |phase: &'static str, battery: &'static [&'static str]| {
        let client = &client;
        async move {
            for q in battery {
                for _ in 0..2 {
                    client.simple_query(q).await.unwrap_or_else(|e| {
                        panic!(
                            "battery query failed on the {phase} seam:\n  {q}\n  {}",
                            db_error_detail(&e)
                        )
                    });
                }
            }
        }
    };

    // Phase A — native (Volcano) route: pre-MATERIALIZE, measured geometry from
    // the native planner's `plan_instrumented`.
    let before_native = total_samples(&geometry_samples());
    run_battery("native", CORE_BATTERY).await;
    run_battery("native", JOIN_BATTERY).await;
    // The fold runs on the io_trace flush at query completion — already done by
    // the time the pgwire response returns, but yield once to be safe.
    sleep(Duration::from_millis(200)).await;
    let after_native = total_samples(&geometry_samples());
    let native_delta = after_native - before_native;
    eprintln!("native-seam calibration samples: {native_delta}");

    // Phase B — DataFusion route: MATERIALIZE, then the same battery; measured
    // geometry from the DataFusion adapter's logical-plan walk (#931).
    for t in ["orders", "lineitem"] {
        client
            .simple_query(&format!("ALTER TABLE {t} MATERIALIZE"))
            .await
            .unwrap_or_else(|e| panic!("MATERIALIZE {t}: {e}"));
    }
    let before_df = total_samples(&geometry_samples());
    run_battery("datafusion", CORE_BATTERY).await;
    run_battery("datafusion", JOIN_BATTERY).await;
    sleep(Duration::from_millis(200)).await;
    let samples = geometry_samples();
    let df_delta = total_samples(&samples) - before_df;
    eprintln!("datafusion-seam calibration samples: {df_delta}");

    // Full estimated×measured matrix for diagnosis on failure.
    eprintln!("\n=== route_geometry_estimate_total (estimated → measured) ===");
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| (a.0.as_str(), a.1.as_str()).cmp(&(b.0.as_str(), b.1.as_str())));
    for (e, m, c) in &sorted {
        let mark = if e == m { " " } else { "≠" };
        eprintln!("  {mark} {e:>6} → {m:<6} {c:>5}");
    }
    let total = total_samples(&samples);
    let divergent = divergent_samples(&samples);
    let rate = if total > 0.0 { divergent / total } else { 0.0 };
    eprintln!("total={total:.0} divergent={divergent:.0} rate={rate:.3}\n");

    // Ratchet 1: the loop is LIVE on both route seams.
    assert!(
        native_delta >= MIN_SAMPLES_PER_PHASE,
        "native-seam calibration loop went dark: {native_delta} samples (need ≥ {MIN_SAMPLES_PER_PHASE}) — \
         did the pgwire path stop stamping geom= or the native planner stop measuring plan geometry?"
    );
    assert!(
        df_delta >= MIN_SAMPLES_PER_PHASE,
        "DataFusion-seam calibration loop went dark: {df_delta} samples (need ≥ {MIN_SAMPLES_PER_PHASE}) — \
         the #931 logical-plan geometry walk stopped emitting (the pre-#931 dark spot)?"
    );

    // Ratchet 2: estimator accuracy. A failure here means an estimator or
    // measurement change regressed calibration — refit the band from measured
    // evidence (TD-EXEC-2 §2 pattern); do not raise the threshold.
    assert!(
        rate <= MAX_DIVERGENCE_RATE,
        "geometry-estimator divergence {rate:.3} exceeds the calibration ratchet {MAX_DIVERGENCE_RATE} \
         ({divergent} of {total} samples off-band; see the matrix above)"
    );
}
