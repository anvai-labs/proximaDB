// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! External parquet table over pgwire — TD-OLAP-4 (external-table hardening).
//!
//! Proves the end-to-end path for registering an existing Parquet object as a
//! read-only table WITHOUT ingestion:
//!   CREATE TABLE t (...) WITH (format='parquet', external_location='…', authority='external')
//! is persisted in the native catalog as an external-authoritative Parquet
//! layout, and `SELECT` over it routes to the DataFusion OLAP engine and reads
//! the object directly.
//!
//! Also pins the Path-Isolation guardrail: an `external_location` outside the
//! operator allowlist (`PROXIMADB_EXTERNAL_TABLE_ROOTS`) is rejected fail-closed.
//!
//! Gated on `datafusion-integration` (the external read routes to DataFusion).
//!   cargo test --features datafusion-integration --test external_table_pgwire_e2e -- --nocapture
#![cfg(feature = "datafusion-integration")]

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;
use tokio_postgres::NoTls;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
/// Write a tiny two-column Parquet object and return its `file://` URI.
fn write_parquet(dir: &std::path::Path) -> String {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .expect("batch");
    let path = dir.join("ext.parquet");
    let file = std::fs::File::create(&path).expect("create parquet");
    let mut w = ArrowWriter::try_new(file, schema, None).expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
    format!("file://{}", path.display())
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn external_parquet_table_registers_and_reads_via_datafusion() {
    let tmp = TempDir::new().expect("tmp");
    // Allowlist this run's temp dir (which holds the external object).
    let root = format!("file://{}", tmp.path().display());
    unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &root) };

    let ext_dir = tmp.path().join("external");
    std::fs::create_dir_all(&ext_dir).expect("mkdir");
    let location = write_parquet(&ext_dir);

    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;

    // The native catalog persists outside this test's tempdir, so clear any
    // entry a prior run left behind (mirrors the tpc harness's DROP-first).
    client
        .simple_query("DROP TABLE IF EXISTS ext_hits")
        .await
        .ok();

    // Register the existing Parquet object as a read-only external table.
    client
        .simple_query(&format!(
            "CREATE TABLE ext_hits (id INT, name VARCHAR) \
             WITH (format='parquet', external_location='{location}', authority='external')"
        ))
        .await
        .expect("create external table");

    // SELECT routes to the DataFusion OLAP engine and reads the object directly.
    // NOTE: the query must "engage" the relational pipeline (join / GROUP BY /
    // aggregate) — pgwire keeps a *simple* single-table SELECT on the legacy path
    // that reads the (empty) backing collection, so a bare `SELECT * FROM ext`
    // returns no rows today. Parquet-backed / external tables always routing to
    // DataFusion is a separate follow-up; ClickBench's queries are aggregates and
    // engage naturally. GROUP BY here both engages and preserves the row values.
    let rows = client
        .simple_query("SELECT id, name FROM ext_hits GROUP BY id, name ORDER BY id")
        .await
        .expect("select");
    let data: Vec<_> = rows
        .iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some((
                r.get("id").unwrap().to_string(),
                r.get("name").unwrap().to_string(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        data,
        vec![
            ("1".to_string(), "a".to_string()),
            ("2".to_string(), "b".to_string()),
            ("3".to_string(), "c".to_string()),
        ],
        "external parquet rows must be read back via the DataFusion route"
    );

    unsafe { std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS") };
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn external_location_outside_allowlist_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    // Allowlist ONLY this run's dir; the table below points outside it.
    unsafe {
        std::env::set_var(
            "PROXIMADB_EXTERNAL_TABLE_ROOTS",
            format!("file://{}", tmp.path().display()),
        )
    };
    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;
    client.simple_query("DROP TABLE IF EXISTS bad").await.ok();

    let err = client
        .simple_query(
            "CREATE TABLE bad (id INT) \
             WITH (format='parquet', external_location='file:///etc/', authority='external')",
        )
        .await
        .expect_err("CREATE must be rejected when the location is outside the allowlist");
    // tokio-postgres' Display is the terse "db error"; the real message is on
    // the DbError cause.
    let msg = err
        .as_db_error()
        .map(|e| e.message().to_lowercase())
        .unwrap_or_else(|| err.to_string().to_lowercase());
    assert!(
        msg.contains("allowlist") || msg.contains("external"),
        "rejection should cite the external-location allowlist, got: {msg}"
    );

    unsafe { std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS") };
}
