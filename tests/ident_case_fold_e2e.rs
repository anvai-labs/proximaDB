// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Identifier case-folding over pgwire — TD-OLAP-18 (ANSI/PostgreSQL conformance).
//!
//! ANSI SQL / PostgreSQL fold **unquoted** identifiers, so a column declared as
//! `ColMine` must resolve from `colmine`, `COLMINE`, or `ColMine`; **quoted**
//! identifiers stay case-exact. ProximaDB historically resolved unquoted
//! identifiers case-sensitively, which walled off 37/43 ClickBench queries
//! (`Schema error: No field named <lowercase>` — the ClickBench DDL/Parquet is
//! CamelCase and DataFusion's planner folds the unquoted query references).
//!
//! Two surfaces are pinned here:
//! 1. External CamelCase Parquet (the exact ClickBench geometry, tiny fixture):
//!    ClickBench-shaped aggregates referencing `AdvEngineID`/`UserID`/… must
//!    resolve regardless of the case the query spells them in.
//! 2. Native (non-parquet) round-trip: `CREATE TABLE CaseTbl (ColMine INT)`;
//!    `SELECT colmine FROM casetbl` works (folded) and
//!    `SELECT "ColMine" FROM "CaseTbl"` works (case-exact quoted).
//!
//! Kill switch: `PROXIMADB_IDENT_CASE_FOLD=0` restores the legacy case-exact
//! resolution (mixed-read safety; the fold only turns hard errors into
//! successes, so it ships default-ON).
//!
//! Gated on `datafusion-integration` (the external read routes to DataFusion).
//!   cargo nextest run --test ident_case_fold_e2e
#![cfg(feature = "datafusion-integration")]

use arrow_array::{Int16Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::net::TcpListener;
use std::sync::Arc;
use tempfile::TempDir;
use tokio_postgres::{NoTls, SimpleQueryMessage};

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
async fn connect(server: &TestPgServer) -> tokio_postgres::Client {
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// First data row of a simple_query result, as strings.
fn first_row(msgs: &[SimpleQueryMessage]) -> Option<Vec<String>> {
    msgs.iter().find_map(|m| match m {
        SimpleQueryMessage::Row(r) => Some(
            (0..r.len())
                .map(|i| r.get(i).unwrap_or("NULL").to_string())
                .collect(),
        ),
        _ => None,
    })
}

/// Run a query and return the first row, panicking with the server-side
/// `DbError` message (not tokio-postgres' terse "db error") on failure.
async fn query_row(client: &tokio_postgres::Client, sql: &str) -> Vec<String> {
    match client.simple_query(sql).await {
        Ok(msgs) => first_row(&msgs).unwrap_or_else(|| panic!("no rows for `{sql}`")),
        Err(e) => {
            let msg = e
                .as_db_error()
                .map(|d| d.message().to_string())
                .unwrap_or_else(|| e.to_string());
            panic!("`{sql}` failed: {msg}");
        }
    }
}

/// Write a tiny CamelCase-schema Parquet fixture — the exact ClickBench
/// geometry (the official `hits.parquet` field names are CamelCase).
fn write_camelcase_parquet(dir: &std::path::Path) -> String {
    let schema = Arc::new(Schema::new(vec![
        Field::new("AdvEngineID", DataType::Int16, false),
        Field::new("UserID", DataType::Int64, false),
        Field::new("SearchPhrase", DataType::Utf8, false),
        Field::new("ResolutionWidth", DataType::Int16, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int16Array::from(vec![0, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![100, 100, 200, 300])),
            Arc::new(StringArray::from(vec!["", "phone", "phone case", ""])),
            Arc::new(Int16Array::from(vec![1024, 1366, 1366, 1920])),
        ],
    )
    .expect("batch");
    let path = dir.join("hits_cf.parquet");
    let file = std::fs::File::create(&path).expect("create parquet");
    let mut w = ArrowWriter::try_new(file, schema, None).expect("writer");
    w.write(&batch).expect("write");
    w.close().expect("close");
    format!("file://{}", path.display())
}

/// ClickBench geometry: CamelCase Parquet + CamelCase DDL; queries must
/// resolve whether they spell the columns CamelCase (as ClickBench does — the
/// planner folds them) or lowercase (ANSI folding).
///
/// Runs on a manual runtime with 8 MiB worker stacks (repo e2e precedent,
/// `abac_relational_transport_e2e.rs`): the full server + DataFusion lowering
/// chain overflows the 2 MiB libtest thread in debug builds.
#[test]
fn camelcase_external_parquet_resolves_any_case() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(camelcase_external_parquet_resolves_any_case_inner());
}

async fn camelcase_external_parquet_resolves_any_case_inner() {
    let ext_tmp = TempDir::new().expect("ext tmp");
    let root = format!("file://{}", ext_tmp.path().display());
    unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &root) };

    let ext_dir = ext_tmp.path().join("external");
    std::fs::create_dir_all(&ext_dir).expect("mkdir");
    let location = write_camelcase_parquet(&ext_dir);

    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;
    client
        .simple_query("DROP TABLE IF EXISTS hits_cf")
        .await
        .ok();
    client
        .simple_query(&format!(
            "CREATE TABLE hits_cf (AdvEngineID SMALLINT, UserID BIGINT, \
             SearchPhrase VARCHAR, ResolutionWidth SMALLINT) \
             WITH (format='parquet', external_location='{location}', authority='external')"
        ))
        .await
        .expect("register CamelCase external table");

    // q02 shape — CamelCase spelled exactly as ClickBench writes it.
    let row = query_row(
        &client,
        "SELECT COUNT(*) FROM hits_cf WHERE AdvEngineID <> 0",
    )
    .await;
    assert_eq!(row, vec!["3"], "q02 shape (CamelCase refs)");

    // q03 shape — multiple CamelCase aggregates.
    let row = query_row(
        &client,
        "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits_cf",
    )
    .await;
    assert_eq!(row[0], "7", "q03 SUM(AdvEngineID)");
    assert_eq!(row[1], "4", "q03 COUNT(*)");

    // q05 shape — COUNT(DISTINCT UserID).
    let row = query_row(&client, "SELECT COUNT(DISTINCT UserID) FROM hits_cf").await;
    assert_eq!(row, vec!["3"], "q05 shape (COUNT DISTINCT)");

    // q08 shape — GROUP BY + ORDER BY on a CamelCase column.
    let row = query_row(
        &client,
        "SELECT AdvEngineID, COUNT(*) FROM hits_cf WHERE AdvEngineID <> 0 \
         GROUP BY AdvEngineID ORDER BY COUNT(*) DESC",
    )
    .await;
    assert_eq!(row, vec!["2", "2"], "q08 shape (GROUP BY CamelCase)");

    // LIKE predicate over a CamelCase string column (q20+ shapes).
    let row = query_row(
        &client,
        "SELECT COUNT(*) FROM hits_cf WHERE SearchPhrase LIKE '%phone%'",
    )
    .await;
    assert_eq!(row, vec!["2"], "LIKE over CamelCase column");

    // SUM(DISTINCT …) is NOT lowered by the shared frontend → exercises the
    // DataFusion SQL-fallback path (`ctx.sql`), which folds unquoted idents.
    let row = query_row(&client, "SELECT SUM(DISTINCT AdvEngineID) FROM hits_cf").await;
    assert_eq!(row, vec!["5"], "ctx.sql fallback path (SUM DISTINCT)");

    // ANSI folding: the same column referenced in LOWERCASE must also resolve.
    let row = query_row(&client, "SELECT COUNT(DISTINCT userid) FROM hits_cf").await;
    assert_eq!(row, vec!["3"], "lowercase ref against CamelCase column");
    let row = query_row(
        &client,
        "SELECT advengineid, COUNT(*) FROM hits_cf GROUP BY advengineid ORDER BY advengineid",
    )
    .await;
    assert_eq!(row, vec!["0", "1"], "lowercase GROUP BY ref");

    // Quoted, case-exact (matches the Parquet/DDL declared case) still works.
    let row = query_row(
        &client,
        "SELECT COUNT(*) FROM hits_cf WHERE \"AdvEngineID\" <> 0",
    )
    .await;
    assert_eq!(row, vec!["3"], "quoted case-exact ref");

    unsafe { std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS") };
}

/// ANSI round-trip on a native (non-parquet) table:
/// `CREATE TABLE CaseTbl (ColMine INT)` must serve `SELECT colmine FROM
/// casetbl` (unquoted folds) AND `SELECT "ColMine" FROM "CaseTbl"`
/// (quoted stays case-exact against the declared case).
#[test]
fn native_table_round_trip_ansi_case_folding() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(native_table_round_trip_ansi_case_folding_inner());
}

async fn native_table_round_trip_ansi_case_folding_inner() {
    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;
    client
        .simple_query("DROP TABLE IF EXISTS CaseTbl")
        .await
        .ok();
    client
        .simple_query("DROP TABLE IF EXISTS casetbl")
        .await
        .ok();

    client
        .simple_query("CREATE TABLE CaseTbl (ColMine INT PRIMARY KEY, OtherCol VARCHAR)")
        .await
        .expect("CREATE TABLE CaseTbl");
    client
        .simple_query("INSERT INTO CaseTbl (ColMine, OtherCol) VALUES (1, 'a')")
        .await
        .expect("INSERT declared case");
    // ANSI: unquoted DML identifiers fold too.
    client
        .simple_query("INSERT INTO casetbl (colmine, othercol) VALUES (2, 'b')")
        .await
        .expect("INSERT folded case");

    // Unquoted references fold — declared case, lowercase, and UPPERCASE all
    // resolve to the same column.
    for sql in [
        "SELECT ColMine FROM CaseTbl ORDER BY ColMine",
        "SELECT colmine FROM casetbl ORDER BY colmine",
        "SELECT COLMINE FROM CASETBL ORDER BY COLMINE",
    ] {
        let row = query_row(&client, sql).await;
        assert_eq!(row, vec!["1"], "unquoted fold failed for `{sql}`");
    }

    // Quoted identifiers stay case-exact against the DECLARED case.
    let row = query_row(
        &client,
        "SELECT \"ColMine\" FROM \"CaseTbl\" ORDER BY \"ColMine\"",
    )
    .await;
    assert_eq!(row, vec!["1"], "quoted case-exact round-trip");

    // Aggregate shape (engages the relational pipeline, not the legacy
    // single-table path) with folded refs.
    let row = query_row(
        &client,
        "SELECT othercol, COUNT(*) FROM casetbl GROUP BY othercol ORDER BY othercol",
    )
    .await;
    assert_eq!(row, vec!["a", "1"], "relational-pipeline folded refs");

    // WHERE predicate with folded column name.
    let row = query_row(&client, "SELECT othercol FROM casetbl WHERE colmine = 2").await;
    assert_eq!(row, vec!["b"], "folded WHERE predicate");
}

/// Kill switch: `PROXIMADB_IDENT_CASE_FOLD=0` restores legacy case-exact
/// resolution (a lowercase ref against a CamelCase column fails again).
/// Runs in its own process under nextest, so the env var cannot leak.
#[test]
fn kill_switch_restores_case_exact_resolution() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(kill_switch_restores_case_exact_resolution_inner());
}

async fn kill_switch_restores_case_exact_resolution_inner() {
    unsafe { std::env::set_var("PROXIMADB_IDENT_CASE_FOLD", "0") };
    let server = TestPgServer::start().await.expect("server");
    let client = connect(&server).await;
    client
        .simple_query("DROP TABLE IF EXISTS KillTbl")
        .await
        .ok();
    client
        .simple_query("CREATE TABLE KillTbl (MyCol INT PRIMARY KEY)")
        .await
        .expect("CREATE TABLE KillTbl");
    client
        .simple_query("INSERT INTO KillTbl (MyCol) VALUES (1)")
        .await
        .expect("INSERT");

    // Declared case still works…
    let row = query_row(&client, "SELECT MyCol FROM KillTbl").await;
    assert_eq!(row, vec!["1"]);
    // …but the folded TABLE reference must NOT resolve with the switch off.
    // (Column-level ci on the DML SELECT path predates the gate — the
    // projection/predicate resolvers were already case-insensitive — so the
    // kill switch's observable edge is table resolution. The aggregate/GROUP-BY
    // route is also out of scope: its table key is normalize_table_key, which
    // lowercases by design, gate-independently.)
    let err = client.simple_query("SELECT MyCol FROM killtbl").await;
    assert!(
        err.is_err(),
        "folded table ref must fail with PROXIMADB_IDENT_CASE_FOLD=0"
    );
    unsafe { std::env::remove_var("PROXIMADB_IDENT_CASE_FOLD") };
}
