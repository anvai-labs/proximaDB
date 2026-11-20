// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! External parquet tables on S3-compatible storage over pgwire — TD-OLAP-4
//! follow-up to `external_table_pgwire_e2e.rs` (which covers `file://`).
//!
//! Proves that an `external_location` may point at an S3-compatible,
//! path-style endpoint (SeaweedFS fixture): `s3://` resolution forwards the
//! process environment (`AWS_*` endpoint/credentials/path-style), the
//! bare-directory listing path scans `…/data/*.parquet` with no manifest, and
//! SELECT over the registered tables routes to the DataFusion OLAP engine.
//!
//! Env-gated so CI stays deterministic: the test SKIPS (passes, printing a
//! notice) unless `PROXIMADB_S3_E2E=1` AND the SeaweedFS fixture on
//! localhost:9000 is up. Run it with:
//!   RUST_MIN_STACK=16777216 PROXIMADB_S3_E2E=1 cargo nextest run \
//!     --features aws,datafusion-integration --test external_table_s3_pgwire_e2e
//! (RUST_MIN_STACK is the TD-STACK-1 pgwire-e2e floor; fixture creds are
//! throwaway fakes.)
#![cfg(feature = "datafusion-integration")]

use tokio_postgres::NoTls;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

/// Fixture object locations (Lakekeeper-discovered, baked here: the fixture's
/// table UUIDs are stable for this SeaweedFS instance's smoke generation).
const LOCATIONS: &[(&str, &str)] = &[
    (
        "store_returns",
        "s3://warehouse/01a1f6c8-b15f-7799-9d52-73f57c309323/data",
    ),
    (
        "date_dim",
        "s3://warehouse/01a1f6c8-af13-74a9-bc5a-96fb7846dd60/data",
    ),
    (
        "store",
        "s3://warehouse/01a1f6c8-b123-764a-82cb-0e2b9b5541a2/data",
    ),
    (
        "customer",
        "s3://warehouse/01a1f6c8-aba8-72fc-8ebc-d6531a708138/data",
    ),
];

/// DDL matching the parquet schema for the columns q1 references (TPC-DS
/// standard names; the catalog type must mirror the parquet type: int64 →
/// BIGINT, decimal128(7,2) → DECIMAL(7,2)).
const DDL: &[(&str, &str)] = &[
    (
        "store_returns",
        "CREATE TABLE store_returns \
         (sr_returned_date_sk BIGINT, sr_item_sk BIGINT, sr_customer_sk BIGINT, \
          sr_store_sk BIGINT, sr_return_amt DECIMAL(7,2)) \
         WITH (format='parquet', external_location='{loc}', authority='external')",
    ),
    (
        "date_dim",
        "CREATE TABLE date_dim (d_date_sk BIGINT, d_year INT) \
         WITH (format='parquet', external_location='{loc}', authority='external')",
    ),
    (
        "store",
        "CREATE TABLE store (s_store_sk BIGINT, s_state VARCHAR) \
         WITH (format='parquet', external_location='{loc}', authority='external')",
    ),
    (
        "customer",
        "CREATE TABLE customer (c_customer_sk BIGINT, c_customer_id VARCHAR) \
         WITH (format='parquet', external_location='{loc}', authority='external')",
    ),
];

fn fixture_available() -> bool {
    if std::env::var("PROXIMADB_S3_E2E").as_deref() != Ok("1") {
        println!("SKIP: set PROXIMADB_S3_E2E=1 (requires the SeaweedFS fixture on :9000)");
        return false;
    }
    true
}

fn set_s3_env() {
    // SAFETY: test-process env setup before the in-process server reads it
    // (edition-2024 unsafe env mutation, same pattern as the file:// e2e).
    // Throwaway fixture fakes (RULES-style), path-style, plain HTTP.
    unsafe {
        std::env::set_var("AWS_ACCESS_KEY_ID", "lakehouse-test");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "lakehouse-test-password");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ENDPOINT", "http://localhost:9000");
        std::env::set_var("AWS_ALLOW_HTTP", "true");
        std::env::set_var("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "false");
        std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");
        std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", "s3://warehouse");
    }
}

async fn connect(server: &TestPgServer) -> tokio_postgres::Client {
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// Register the four fixture tables (DROP-first: the native catalog persists
/// across runs, mirroring the tpcds harness).
async fn register_tables(client: &tokio_postgres::Client) {
    for (name, _) in LOCATIONS {
        client
            .simple_query(&format!("DROP TABLE IF EXISTS {name}"))
            .await
            .ok();
    }
    for ((name, loc), (ddl_name, ddl)) in LOCATIONS.iter().zip(DDL.iter()) {
        // LOCATIONS and DDL are same-order by construction; keep them in sync.
        assert_eq!(name, ddl_name, "DDL/table order");
        let sql = ddl.replace("{loc}", loc);
        client
            .simple_query(&sql)
            .await
            .unwrap_or_else(|e| panic!("CREATE TABLE {name} failed: {e}"));
    }
}

async fn scalar(client: &tokio_postgres::Client, sql: &str) -> i64 {
    let rows = client.simple_query(sql).await.expect("query");
    for m in rows {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            return r.get(0).unwrap().parse().expect("int column");
        }
    }
    panic!("no row returned for: {sql}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_external_tables_register_and_aggregate() {
    if !fixture_available() {
        return;
    }
    set_s3_env();
    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;
    register_tables(&client).await;

    // Sanity aggregates through the DataFusion route over the S3 listing
    // reader (row counts must match the fixture's duckdb-computed goldens).
    let store_rows = scalar(&client, "SELECT count(*) FROM store").await;
    assert_eq!(store_rows, 102, "store row count via s3 external table");

    let dim_1998 = scalar(
        &client,
        "SELECT count(*) FROM date_dim WHERE d_year = 1998 GROUP BY d_year",
    )
    .await;
    assert_eq!(dim_1998, 365, "date_dim rows for d_year=1998");

    // The q1 CTE: a star join (fact ⋈ dim) + GROUP BY over the S3 tables —
    // proves join routing engages with external-authoritative locations.
    let cte_rows = scalar(
        &client,
        "SELECT count(*) FROM (SELECT sr_customer_sk, sr_store_sk, \
         sum(sr_return_amt) AS ctr_total_return \
         FROM store_returns, date_dim \
         WHERE sr_returned_date_sk = d_date_sk AND d_year = 2001 \
         GROUP BY sr_customer_sk, sr_store_sk) ctr",
    )
    .await;
    assert_eq!(
        cte_rows, 534_343,
        "q1-CTE grouped customer/store count (duckdb golden)"
    );

    // SAFETY: see set_s3_env.
    unsafe {
        std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS");
    }
}

/// The full official q1 (correlated scalar-subquery form) — recorded as a
/// separate probe so a planner limitation shows as THIS test failing, not the
/// registration/aggregation basics above.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_external_tables_full_q1() {
    if !fixture_available() {
        return;
    }
    set_s3_env();
    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;
    register_tables(&client).await;

    // Official q1 shape (correlated scalar subquery over the CTE), asserting
    // both the LIMIT-ed row count and the ordered content head (duckdb golden:
    // 100 rows, first ids AAAAAAAAAAAAEAAA / AAAAAAAAAAABBAAA / AAAAAAAAAAABHAAA;
    // 48,304 rows before LIMIT).
    let sql = "WITH customer_total_return AS ( \
        SELECT sr_customer_sk AS ctr_customer_sk, sr_store_sk AS ctr_store_sk, \
               sum(sr_return_amt) AS ctr_total_return \
        FROM store_returns, date_dim \
        WHERE sr_returned_date_sk = d_date_sk AND d_year = 2001 \
        GROUP BY sr_customer_sk, sr_store_sk) \
        SELECT c_customer_id FROM customer_total_return ctr1, store, customer \
        WHERE ctr1.ctr_total_return > (SELECT avg(ctr_total_return) * 1.2 \
                                       FROM customer_total_return ctr2 \
                                       WHERE ctr1.ctr_store_sk = ctr2.ctr_store_sk) \
        AND s_store_sk = ctr1.ctr_store_sk AND s_state = 'TN' \
        AND ctr1.ctr_customer_sk = c_customer_sk \
        ORDER BY c_customer_id LIMIT 100";
    let rows = client.simple_query(sql).await.expect("official q1");
    let ids: Vec<String> = rows
        .iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r.get(0).unwrap().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        ids.len(),
        100,
        "official q1 must return exactly its LIMIT 100 rows"
    );
    assert_eq!(
        ids[..3],
        [
            "AAAAAAAAAAAAEAAA".to_string(),
            "AAAAAAAAAAABBAAA".to_string(),
            "AAAAAAAAAAABHAAA".to_string(),
        ],
        "official q1 ordered head must match the duckdb golden"
    );

    // SAFETY: see set_s3_env.
    unsafe {
        std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS");
    }
}
