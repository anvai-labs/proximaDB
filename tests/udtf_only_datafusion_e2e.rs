// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-097 P3.2: a query over table functions alone — no catalog table — runs
//! on DataFusion over pgwire.
//!
//! The shared relational frontend cannot lower a table function, and with no
//! catalog table neither DataFusion arm admitted the query, so an aggregate,
//! window or CTE over `timeseries_range(...)` fell through to the legacy
//! single-table path. With `PROXIMADB_NATIVE_TABLE_PROVIDER` on, an ungoverned
//! session now reaches the DataFusion floor, where the table functions are
//! registered.
//!
//!   cargo test --features datafusion-integration --test udtf_only_datafusion_e2e
#![cfg(feature = "datafusion-integration")]

use std::time::Duration;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

async fn rows(client: &tokio_postgres::Client, sql: &str) -> Vec<Vec<String>> {
    let msgs = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {}", pg_server::db_err(&e)));
    msgs.into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).unwrap_or("NULL").to_string())
                    .collect(),
            ),
            _ => None,
        })
        .collect()
}

/// Turns the native-provider gate on, and back off when dropped — even on a
/// failed assertion.
struct GateOn;

impl GateOn {
    fn set() -> Self {
        // SAFETY: set between awaited queries — no request is in flight, so no
        // server thread reads the variable concurrently; tests run under
        // nextest process isolation.
        unsafe { std::env::set_var("PROXIMADB_NATIVE_TABLE_PROVIDER", "1") };
        GateOn
    }
}

impl Drop for GateOn {
    fn drop(&mut self) {
        // SAFETY: see `set`.
        unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER") };
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn table_function_only_queries_run_on_datafusion() {
    let server = TestPgServer::start().await.expect("server start");
    // Unique per run: the tenant is the pgwire database and the REST header.
    let tenant = format!("udtf{}", server.pg_port);
    let coll = format!("meter_{}", server.pg_port);

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("http client");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let create = http
        .post(format!("{base}/api/v2/timeseries/collections"))
        .header("X-Tenant-ID", &tenant)
        .json(&serde_json::json!({ "name": coll, "value_columns": [{ "name": "amount" }] }))
        .send()
        .await
        .expect("create req");
    assert!(create.status().is_success(), "create: {}", create.status());
    let ingest = http
        .post(format!(
            "{base}/api/v2/timeseries/collections/{coll}/ingest"
        ))
        .header("X-Tenant-ID", &tenant)
        .json(&serde_json::json!({
            "points": [
                { "timestamp": 1_000, "values": { "amount": 100.0 } },
                { "timestamp": 2_000, "values": { "amount": 9_000.0 } },
                { "timestamp": 3_000, "values": { "amount": 400.0 } },
            ]
        }))
        .send()
        .await
        .expect("ingest req");
    assert!(ingest.status().is_success(), "ingest: {}", ingest.status());

    let (client, conn) = tokio_postgres::connect(
        &format!(
            "host=127.0.0.1 port={} user=postgres dbname={tenant} sslmode=disable",
            server.pg_port
        ),
        tokio_postgres::NoTls,
    )
    .await
    .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let range = format!("timeseries_range('{coll}', 0, 9999999)");
    let window = format!(
        "SELECT t.\"timestamp\", sum(t.value) over (order by t.\"timestamp\") as running \
         FROM {range} t order by t.\"timestamp\""
    );

    // Gate off: no DataFusion arm admits a table-function-only query, and the
    // legacy path cannot answer a window over it.
    let gate_off = client.simple_query(&window).await;
    assert!(
        gate_off.is_err(),
        "gate off: a window over a table function alone is not DataFusion's: {gate_off:?}"
    );

    let _gate = GateOn::set();

    assert_eq!(
        rows(&client, &window).await,
        [["1000", "100"], ["2000", "9100"], ["3000", "9500"]],
        "a window over the table function, on DataFusion"
    );
    assert_eq!(
        rows(
            &client,
            &format!(
                "SELECT count(*) AS n, max(t.value) AS top FROM {range} t WHERE t.value > 200"
            ),
        )
        .await,
        [["2", "9000"]],
        "an aggregate over the table function"
    );
    assert_eq!(
        rows(
            &client,
            &format!(
                "WITH hi AS (SELECT t.value FROM {range} t WHERE t.value > 200) \
                 SELECT count(*) AS n FROM hi"
            ),
        )
        .await,
        [["2"]],
        "a CTE over the table function"
    );
    assert_eq!(
        rows(
            &client,
            &format!("SELECT t.value FROM {range} t WHERE t.value < 1000 ORDER BY t.value"),
        )
        .await,
        [["100"], ["400"]],
        "a plain select over the table function"
    );

    drop(client);
    server.shutdown().await;
}
