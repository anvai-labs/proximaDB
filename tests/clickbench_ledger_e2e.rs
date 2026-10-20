// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ClickBench single-node OLAP ledger over pgwire (TD-OLAP-4, ClickBench track).
//!
//! Registers the pre-built ClickBench `hits` Parquet as a read-only EXTERNAL
//! table (no ingestion) and runs the canonical queries on the DataFusion route,
//! recording per-query wall + rows into a JSON ledger. When a DuckDB binary is
//! available (`DUCKDB_BIN`) it runs the same queries over the same Parquet as a
//! single-node baseline. Advisory — asserts harness integrity, never perf.
//!
//! Requires a ClickBench `hits` Parquet (VARCHAR string cols); point
//! `CLICKBENCH_PARQUET` at it. Its directory is auto-allowlisted for the
//! external-table load. Column case is no longer a constraint (TD-OLAP-18 fix):
//! unquoted references resolve case-insensitively against declared CamelCase
//! columns (declared-case always wins; unique fold on miss), so the stock
//! CamelCase `hits` Parquet works — the lowercase pre-folded fixture remains
//! supported and exercises the exact-match path. See the v1 evidence doc. Run:
//!   CLICKBENCH_PARQUET=/path/cb_hits.parquet DUCKDB_BIN=/path/duckdb \
//!     cargo test --features datafusion-integration --test clickbench_ledger_e2e \
//!     clickbench_ledger -- --ignored --nocapture
#![cfg(feature = "datafusion-integration")]

use std::sync::Mutex;
use std::time::{Duration, Instant};

use proximadb::observability::io_trace::{self, IoTraceSnapshot};
use tokio::time::sleep;
use tokio_postgres::{NoTls, SimpleQueryMessage};

/// Per-query I/O trace capture. The server fires the billing observer at each
/// statement's `io_trace` scope close, so we clear before a query and drain
/// after — the same pattern as `tpc_perf_ledger_e2e.rs`. This turns the ledger
/// from a wall-clock anecdote into a bytes-scanned ledger (the co-design signal).
static CAPTURE: Mutex<Vec<IoTraceSnapshot>> = Mutex::new(Vec::new());

const CLICKBENCH_DDL_COLS: &str = "WatchID BIGINT, JavaEnable SMALLINT, Title VARCHAR, GoodEvent SMALLINT, EventTime BIGINT, EventDate INT, CounterID INT, ClientIP INT, RegionID INT, UserID BIGINT, CounterClass SMALLINT, OS SMALLINT, UserAgent SMALLINT, URL VARCHAR, Referer VARCHAR, IsRefresh SMALLINT, RefererCategoryID SMALLINT, RefererRegionID INT, URLCategoryID SMALLINT, URLRegionID INT, ResolutionWidth SMALLINT, ResolutionHeight SMALLINT, ResolutionDepth SMALLINT, FlashMajor SMALLINT, FlashMinor SMALLINT, FlashMinor2 VARCHAR, NetMajor SMALLINT, NetMinor SMALLINT, UserAgentMajor SMALLINT, UserAgentMinor VARCHAR, CookieEnable SMALLINT, JavascriptEnable SMALLINT, IsMobile SMALLINT, MobilePhone SMALLINT, MobilePhoneModel VARCHAR, Params VARCHAR, IPNetworkID INT, TraficSourceID SMALLINT, SearchEngineID SMALLINT, SearchPhrase VARCHAR, AdvEngineID SMALLINT, IsArtifical SMALLINT, WindowClientWidth SMALLINT, WindowClientHeight SMALLINT, ClientTimeZone SMALLINT, ClientEventTime BIGINT, SilverlightVersion1 SMALLINT, SilverlightVersion2 SMALLINT, SilverlightVersion3 INT, SilverlightVersion4 SMALLINT, PageCharset VARCHAR, CodeVersion INT, IsLink SMALLINT, IsDownload SMALLINT, IsNotBounce SMALLINT, FUniqID BIGINT, OriginalURL VARCHAR, HID INT, IsOldCounter SMALLINT, IsEvent SMALLINT, IsParameter SMALLINT, DontCountHits SMALLINT, WithHash SMALLINT, HitColor VARCHAR, LocalEventTime BIGINT, Age SMALLINT, Sex SMALLINT, Income SMALLINT, Interests SMALLINT, Robotness SMALLINT, RemoteIP INT, WindowName INT, OpenerName INT, HistoryLength SMALLINT, BrowserLanguage VARCHAR, BrowserCountry VARCHAR, SocialNetwork VARCHAR, SocialAction VARCHAR, HTTPError SMALLINT, SendTiming INT, DNSTiming INT, ConnectTiming INT, ResponseStartTiming INT, ResponseEndTiming INT, FetchTiming INT, SocialSourceNetworkID SMALLINT, SocialSourcePage VARCHAR, ParamPrice BIGINT, ParamOrderID VARCHAR, ParamCurrency VARCHAR, ParamCurrencyID SMALLINT, OpenstatServiceName VARCHAR, OpenstatCampaignID VARCHAR, OpenstatAdID VARCHAR, OpenstatSourceID VARCHAR, UTMSource VARCHAR, UTMMedium VARCHAR, UTMCampaign VARCHAR, UTMContent VARCHAR, UTMTerm VARCHAR, FromTag VARCHAR, HasGCLID SMALLINT, RefererHash BIGINT, URLHash BIGINT, CLID INT";

fn clickbench_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("q01", "SELECT COUNT(*) FROM hits"),
        ("q02", "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0"),
        (
            "q03",
            "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits",
        ),
        ("q04", "SELECT AVG(UserID) FROM hits"),
        ("q05", "SELECT COUNT(DISTINCT UserID) FROM hits"),
        ("q06", "SELECT COUNT(DISTINCT SearchPhrase) FROM hits"),
        ("q07", "SELECT MIN(EventDate), MAX(EventDate) FROM hits"),
        (
            "q08",
            "SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC",
        ),
        (
            "q09",
            "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10",
        ),
        (
            "q10",
            "SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10",
        ),
        (
            "q11",
            "SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10",
        ),
        (
            "q12",
            "SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10",
        ),
        (
            "q13",
            "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10",
        ),
        (
            "q14",
            "SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10",
        ),
        (
            "q15",
            "SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10",
        ),
        (
            "q16",
            "SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10",
        ),
        (
            "q17",
            "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10",
        ),
        (
            "q18",
            "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10",
        ),
        (
            "q19",
            "SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10",
        ),
        (
            "q20",
            "SELECT UserID FROM hits WHERE UserID = 435090932899640449",
        ),
        ("q21", "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'"),
        (
            "q22",
            "SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10",
        ),
        (
            "q23",
            "SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10",
        ),
        (
            "q24",
            "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10",
        ),
        (
            "q25",
            "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10",
        ),
        (
            "q26",
            "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10",
        ),
        (
            "q27",
            "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10",
        ),
        (
            "q28",
            "SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25",
        ),
        (
            "q29",
            "SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25",
        ),
        (
            "q30",
            "SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 2), SUM(ResolutionWidth + 3), SUM(ResolutionWidth + 4), SUM(ResolutionWidth + 5), SUM(ResolutionWidth + 6), SUM(ResolutionWidth + 7), SUM(ResolutionWidth + 8), SUM(ResolutionWidth + 9), SUM(ResolutionWidth + 10), SUM(ResolutionWidth + 11), SUM(ResolutionWidth + 12), SUM(ResolutionWidth + 13), SUM(ResolutionWidth + 14), SUM(ResolutionWidth + 15), SUM(ResolutionWidth + 16), SUM(ResolutionWidth + 17), SUM(ResolutionWidth + 18), SUM(ResolutionWidth + 19), SUM(ResolutionWidth + 20), SUM(ResolutionWidth + 21), SUM(ResolutionWidth + 22), SUM(ResolutionWidth + 23), SUM(ResolutionWidth + 24), SUM(ResolutionWidth + 25), SUM(ResolutionWidth + 26), SUM(ResolutionWidth + 27), SUM(ResolutionWidth + 28), SUM(ResolutionWidth + 29), SUM(ResolutionWidth + 30), SUM(ResolutionWidth + 31), SUM(ResolutionWidth + 32), SUM(ResolutionWidth + 33), SUM(ResolutionWidth + 34), SUM(ResolutionWidth + 35), SUM(ResolutionWidth + 36), SUM(ResolutionWidth + 37), SUM(ResolutionWidth + 38), SUM(ResolutionWidth + 39), SUM(ResolutionWidth + 40), SUM(ResolutionWidth + 41), SUM(ResolutionWidth + 42), SUM(ResolutionWidth + 43), SUM(ResolutionWidth + 44), SUM(ResolutionWidth + 45), SUM(ResolutionWidth + 46), SUM(ResolutionWidth + 47), SUM(ResolutionWidth + 48), SUM(ResolutionWidth + 49), SUM(ResolutionWidth + 50), SUM(ResolutionWidth + 51), SUM(ResolutionWidth + 52), SUM(ResolutionWidth + 53), SUM(ResolutionWidth + 54), SUM(ResolutionWidth + 55), SUM(ResolutionWidth + 56), SUM(ResolutionWidth + 57), SUM(ResolutionWidth + 58), SUM(ResolutionWidth + 59), SUM(ResolutionWidth + 60), SUM(ResolutionWidth + 61), SUM(ResolutionWidth + 62), SUM(ResolutionWidth + 63), SUM(ResolutionWidth + 64), SUM(ResolutionWidth + 65), SUM(ResolutionWidth + 66), SUM(ResolutionWidth + 67), SUM(ResolutionWidth + 68), SUM(ResolutionWidth + 69), SUM(ResolutionWidth + 70), SUM(ResolutionWidth + 71), SUM(ResolutionWidth + 72), SUM(ResolutionWidth + 73), SUM(ResolutionWidth + 74), SUM(ResolutionWidth + 75), SUM(ResolutionWidth + 76), SUM(ResolutionWidth + 77), SUM(ResolutionWidth + 78), SUM(ResolutionWidth + 79), SUM(ResolutionWidth + 80), SUM(ResolutionWidth + 81), SUM(ResolutionWidth + 82), SUM(ResolutionWidth + 83), SUM(ResolutionWidth + 84), SUM(ResolutionWidth + 85), SUM(ResolutionWidth + 86), SUM(ResolutionWidth + 87), SUM(ResolutionWidth + 88), SUM(ResolutionWidth + 89) FROM hits",
        ),
        (
            "q31",
            "SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10",
        ),
        (
            "q32",
            "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10",
        ),
        (
            "q33",
            "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10",
        ),
        (
            "q34",
            "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10",
        ),
        (
            "q35",
            "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10",
        ),
        (
            "q36",
            "SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10",
        ),
        (
            "q37",
            "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10",
        ),
        (
            "q38",
            "SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10",
        ),
        (
            "q39",
            "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000",
        ),
        (
            "q40",
            "SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000",
        ),
        (
            "q41",
            "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100",
        ),
        (
            "q42",
            "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000",
        ),
        (
            "q43",
            "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000",
        ),
    ]
}

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

#[derive(serde::Serialize)]
struct QueryRecord {
    query: String,
    engine: String,
    ok: bool,
    rows: usize,
    wall_ms: u128,
    /// Object-store bytes read for this query (ProximaDB route only; the
    /// co-design cost term). `None` for the DuckDB baseline (out-of-process).
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_read: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    range_gets: Option<u64>,
    /// pgwire relational-pipeline setup wall ms — pre-execution xCatalog schema
    /// resolution + route classification (TD-OLAP-4).
    #[serde(skip_serializing_if = "Option::is_none")]
    setup_ms: Option<u64>,
    /// pgwire result-emit wall ms — row encode + socket write (TD-OLAP-4).
    #[serde(skip_serializing_if = "Option::is_none")]
    emit_ms: Option<u64>,
    /// SessionContext build wall ms — per-query context+UDF setup (TD-OLAP-4).
    #[serde(skip_serializing_if = "Option::is_none")]
    session_ms: Option<u64>,
    /// Execution (compute) wall ms attributed by the engine (TD-OLAP-4).
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_ms: Option<u64>,
    /// Engine/route this query was served on — `(shape_class, backend_label)`, the
    /// engine dimension of the geometry-dependent dispatch (TD-OLAP-4 Slice 0).
    /// For external-parquet ClickBench this is `DataFusionLocal` (native cannot
    /// serve external parquet yet), making the current engine coverage explicit.
    #[serde(skip_serializing_if = "Option::is_none")]
    route: Option<String>,
    /// Full per-engine compute breakdown (`{engine: ms}`) — distinguishes
    /// `datafusion` / `native-vectorized` / `volcano` for the cost-model tensor.
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_by_engine: Option<std::collections::BTreeMap<String, u64>>,
    /// Table-OPEN floor (discovery + footer) wall ms — TD-OLAP-4. Drops to ~0 on a
    /// warm table-open cache hit; the direct signal for the cache lever.
    #[serde(skip_serializing_if = "Option::is_none")]
    open_ms: Option<u64>,
    /// Lowering + planning wall ms — the other half of the per-query floor.
    #[serde(skip_serializing_if = "Option::is_none")]
    plan_ms: Option<u64>,
    /// Table-OPEN cache hits recorded for this query (1 per registered table).
    #[serde(skip_serializing_if = "Option::is_none")]
    table_open_hits: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Drain the per-query I/O trace snapshot the server emitted at scope close.
/// Polls briefly because the billing observer fires asynchronously server-side.
async fn drain_capture() -> Option<IoTraceSnapshot> {
    for _ in 0..60 {
        if let Some(s) = CAPTURE.lock().expect("capture lock").pop() {
            return Some(s);
        }
        sleep(Duration::from_millis(5)).await;
    }
    None
}

/// Set a ledger-required env default unless the caller explicitly set it
/// (explicit env always wins). Ledger runs must exercise every routable
/// engine permutation by default — the router is only as grounded as the
/// paths the evidence actually exercised.
fn default_env(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        unsafe { std::env::set_var(key, value) };
    }
}

/// Ledger-run env defaults: native-over-parquet PRIMARY route + vectorized
/// modes ON (so Native samples land in the same olap/parquet classes as
/// DataFusion's and the cost cells warm per engine), per-query wall budget ON
/// (a pathological query becomes an ERR record, not a wedged sweep). The cost
/// override (PROXIMADB_ROUTE_COST_OVERRIDE) is deliberately NOT defaulted —
/// ledgers observe routing, they never flip it (ADR-058 D4).
fn apply_ledger_env_defaults(timeout_key: &str) {
    default_env("PROXIMADB_NATIVE_ROUTE", "1");
    default_env("PROXIMADB_NATIVE_VECTORIZED", "1");
    default_env(timeout_key, "300000");
}

/// The final ledger path (`CLICKBENCH_LEDGER_OUT`, defaulted) — shared by the
/// end-of-run write and the incremental partial sidecar.
fn ledger_out_path() -> String {
    std::env::var("CLICKBENCH_LEDGER_OUT")
        .unwrap_or_else(|_| "target/clickbench-ledger/ledger.json".to_string())
}

/// Record one query result: live progress line, crash-safe append to
/// `<out>.partial.jsonl`, then collect. The harness previously went silent
/// between load and the final write, holding every record in memory — a
/// wedged/killed run could neither name the culprit query nor keep any
/// completed measurement. Now a run that never completes still leaves the
/// full trail up to the query it died on.
fn push_cb(records: &mut Vec<QueryRecord>, rec: QueryRecord) {
    match &rec.error {
        None => eprintln!(
            "[clickbench/{}] {} {} ms rows={}",
            rec.engine, rec.query, rec.wall_ms, rec.rows
        ),
        Some(e) => eprintln!(
            "[clickbench/{}] {} {} ms ERR: {e}",
            rec.engine, rec.query, rec.wall_ms
        ),
    }
    append_partial_record(&ledger_out_path(), &rec);
    records.push(rec);
}

/// Append one record to `<out>.partial.jsonl`. Best-effort by design: the
/// sidecar is diagnostics, so an append failure must never fail the harness.
fn append_partial_record<T: serde::Serialize>(out_path: &str, rec: &T) {
    use std::io::Write as _;
    let path = format!("{out_path}.partial.jsonl");
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let (Ok(line), Ok(mut f)) = (
        serde_json::to_string(rec),
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path),
    ) {
        let _ = writeln!(f, "{line}");
    }
}

/// Rewrite `FROM hits` to a DuckDB `read_parquet(...)` over the same object.
/// Only used by the CLI fallback (when the `duckdb` feature is OFF); the
/// in-process engine registers a `hits` view, so no rewrite is needed.
#[cfg(not(feature = "duckdb"))]
fn duckdb_sql(sql: &str, parquet: &str) -> String {
    sql.replace("FROM hits", &format!("FROM read_parquet('{parquet}')"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ClickBench single-node ledger (TD-OLAP-4) — advisory; needs CLICKBENCH_PARQUET"]
async fn clickbench_ledger() {
    // Dev-only: honor RUST_LOG so the native-shadow probe's decline reasons are
    // visible during a diagnostic capture (no-op if RUST_LOG is unset).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let parquet = match std::env::var("CLICKBENCH_PARQUET") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("CLICKBENCH_PARQUET unset — skipping ClickBench ledger");
            return;
        }
    };
    let parquet_uri = if parquet.starts_with("file://") {
        parquet.clone()
    } else {
        format!("file://{parquet}")
    };
    // Allowlist the parquet's directory for the external-table load.
    let dir = std::path::Path::new(&parquet)
        .parent()
        .map(|p| format!("file://{}", p.display()))
        .unwrap_or_else(|| "file:///".to_string());
    unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &dir) };

    apply_ledger_env_defaults("CLICKBENCH_QUERY_TIMEOUT_MS");
    // Fresh partial sidecar per run — stale lines from a prior run must not
    // mix into this run's crash-safe trail.
    let _ = std::fs::remove_file(format!("{}.partial.jsonl", ledger_out_path()));
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;
    client.simple_query("DROP TABLE IF EXISTS hits").await.ok();
    client
        .simple_query(&format!(
            "CREATE TABLE hits ({CLICKBENCH_DDL_COLS}) \
             WITH (format='parquet', external_location='{parquet_uri}', authority='external')"
        ))
        .await
        .expect("register ClickBench external table");

    let queries = clickbench_queries();
    let mut records: Vec<QueryRecord> = Vec::new();

    // Capture the per-query I/O trace the server emits at each statement's scope close.
    io_trace::set_billing_observer(Some(Box::new(|snap: &IoTraceSnapshot, _tenant| {
        CAPTURE.lock().expect("capture lock").push(snap.clone());
    })));

    // ProximaDB (DataFusion route).
    // Optional per-query wall budget (`CLICKBENCH_QUERY_TIMEOUT_MS`, unset/0 =
    // off): a pathological query becomes an ERR record + a server-side cancel
    // instead of wedging the sweep. The cancel keeps the connection usable —
    // dropping the future alone would queue the next query behind this one.
    let budget_ms: u64 = std::env::var("CLICKBENCH_QUERY_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for (id, sql) in &queries {
        CAPTURE.lock().expect("capture lock").clear();
        let t0 = Instant::now();
        let res = if budget_ms == 0 {
            client.simple_query(sql).await
        } else {
            match tokio::time::timeout(Duration::from_millis(budget_ms), client.simple_query(sql))
                .await
            {
                Ok(r) => r,
                Err(_elapsed) => {
                    let _ = client
                        .cancel_token()
                        .cancel_query(tokio_postgres::NoTls)
                        .await;
                    push_cb(
                        &mut records,
                        QueryRecord {
                            query: (*id).into(),
                            engine: "proximadb".into(),
                            ok: false,
                            rows: 0,
                            wall_ms: t0.elapsed().as_millis(),
                            bytes_read: None,
                            range_gets: None,
                            setup_ms: None,
                            emit_ms: None,
                            session_ms: None,
                            compute_ms: None,
                            open_ms: None,
                            plan_ms: None,
                            table_open_hits: None,
                            route: None,
                            compute_by_engine: None,
                            error: Some(format!(
                                "harness timeout after {budget_ms} ms (query cancelled)"
                            )),
                        },
                    );
                    continue;
                }
            }
        };
        let wall_ms = t0.elapsed().as_millis();
        let snap = drain_capture().await;
        let (
            bytes_read,
            range_gets,
            setup_ms,
            emit_ms,
            session_ms,
            compute_ms,
            open_ms,
            plan_ms,
            table_open_hits,
            route,
            compute_by_engine,
        ) = snap
            .map(|s| {
                (
                    Some(s.bytes_read),
                    Some(s.range_gets),
                    Some(s.setup_ms),
                    Some(s.emit_ms),
                    Some(s.session_ms),
                    Some(s.total_compute_ms()),
                    Some(s.open_ms),
                    Some(s.plan_ms),
                    Some(s.table_open_hits),
                    s.route.as_ref().map(|(_, backend)| backend.clone()),
                    Some(s.compute_ms.clone()),
                )
            })
            .unwrap_or((
                None, None, None, None, None, None, None, None, None, None, None,
            ));
        match res {
            Ok(msgs) => {
                let rows = msgs
                    .iter()
                    .filter(|m| matches!(m, SimpleQueryMessage::Row(_)))
                    .count();
                push_cb(
                    &mut records,
                    QueryRecord {
                        query: (*id).into(),
                        engine: "proximadb".into(),
                        ok: true,
                        rows,
                        wall_ms,
                        bytes_read,
                        range_gets,
                        setup_ms,
                        emit_ms,
                        session_ms,
                        compute_ms,
                        open_ms,
                        plan_ms,
                        table_open_hits,
                        route,
                        compute_by_engine,
                        error: None,
                    },
                );
            }
            Err(e) => push_cb(
                &mut records,
                QueryRecord {
                    query: (*id).into(),
                    engine: "proximadb".into(),
                    ok: false,
                    rows: 0,
                    wall_ms,
                    bytes_read,
                    range_gets,
                    setup_ms,
                    emit_ms,
                    session_ms,
                    compute_ms,
                    open_ms,
                    plan_ms,
                    table_open_hits,
                    route,
                    compute_by_engine,
                    error: Some(
                        e.as_db_error()
                            .map(|d| d.message().to_string())
                            .unwrap_or_else(|| e.to_string()),
                    ),
                },
            ),
        }
    }
    io_trace::set_billing_observer(None);

    // DuckDB baseline (ADR-059): in-process, same Parquet, io-traced — the
    // engine-behavior-only discriminant (compute_ms; DF-vs-DuckDB on the same
    // parquet/SQL/process). Re-arm the billing observer (cleared above after the
    // ProximaDB route) so the instrument scope's snapshot lands in CAPTURE.
    #[cfg(feature = "duckdb")]
    {
        use proximadb::query::execution::engine::{
            QueryExecutionContext, execute_sql_with_backend,
        };
        use proximadb::query::table_write_plan::ComputeBackend;
        io_trace::set_billing_observer(Some(Box::new(|snap: &IoTraceSnapshot, _tenant| {
            CAPTURE.lock().expect("capture lock").push(snap.clone());
        })));
        for (id, sql) in &queries {
            CAPTURE.lock().expect("capture lock").clear();
            // DuckDbLocalEngine registers `CREATE VIEW hits AS SELECT * FROM
            // read_parquet('{parquet_uri}')`, so the original `FROM hits` SQL
            // works — no duckdb_sql rewrite needed.
            let ctx = QueryExecutionContext {
                parquet_tables: vec![("hits".to_string(), parquet_uri.clone())],
                ..Default::default()
            };
            let t0 = Instant::now();
            // instrument sets the IO_TRACE scope; the engine's record_compute_ms
            // lands in it; instrument emits the snapshot → observer → CAPTURE.
            let sql_owned = sql.to_string();
            let res = io_trace::instrument(None, "duckdb".to_string(), async move {
                execute_sql_with_backend(ComputeBackend::DuckDbCompat, &sql_owned, ctx).await
            })
            .await;
            let wall_ms = t0.elapsed().as_millis();
            let snap = drain_capture().await;
            let compute_ms = snap.as_ref().map(|s| s.total_compute_ms());
            let compute_by_engine = snap.and_then(|s| {
                if s.compute_ms.is_empty() {
                    None
                } else {
                    Some(s.compute_ms)
                }
            });
            let (ok, rows, error) = match res {
                Ok(r) => (true, r.rows.len(), None),
                Err(e) => (false, 0, Some(e.to_string())),
            };
            push_cb(
                &mut records,
                QueryRecord {
                    query: (*id).into(),
                    engine: "duckdb".into(),
                    ok,
                    rows,
                    wall_ms,
                    bytes_read: None,
                    range_gets: None,
                    setup_ms: None,
                    emit_ms: None,
                    session_ms: None,
                    compute_ms,
                    open_ms: None,
                    plan_ms: None,
                    table_open_hits: None,
                    route: Some("DuckDbCompat".into()),
                    compute_by_engine,
                    error,
                },
            );
        }
        io_trace::set_billing_observer(None);
    }
    // Fallback: DuckDB CLI subprocess (only when the `duckdb` feature is OFF —
    // out-of-process, no io-trace; the in-process path above is the ledger route).
    #[cfg(not(feature = "duckdb"))]
    if let Ok(duckdb) = std::env::var("DUCKDB_BIN") {
        for (id, sql) in &queries {
            let dsql = duckdb_sql(sql, &parquet);
            let t0 = Instant::now();
            let out = std::process::Command::new(&duckdb)
                .arg("-c")
                .arg(&dsql)
                .output();
            let wall_ms = t0.elapsed().as_millis();
            let ok = out.as_ref().map(|o| o.status.success()).unwrap_or(false);
            push_cb(
                &mut records,
                QueryRecord {
                    query: (*id).into(),
                    engine: "duckdb".into(),
                    ok,
                    rows: 0,
                    wall_ms,
                    bytes_read: None,
                    range_gets: None,
                    setup_ms: None,
                    emit_ms: None,
                    session_ms: None,
                    compute_ms: None,
                    open_ms: None,
                    plan_ms: None,
                    table_open_hits: None,
                    route: None,
                    compute_by_engine: None,
                    error: (!ok).then(|| {
                        out.as_ref()
                            .map(|o| String::from_utf8_lossy(&o.stderr).trim().to_string())
                            .unwrap_or_else(|_| "duckdb spawn failed".to_string())
                    }),
                },
            );
        }
    }

    // Console summary + ledger.
    for engine in ["proximadb", "duckdb"] {
        let rs: Vec<&QueryRecord> = records.iter().filter(|r| r.engine == engine).collect();
        if rs.is_empty() {
            continue;
        }
        let ok = rs.iter().filter(|r| r.ok).count();
        let mut walls: Vec<u128> = rs.iter().filter(|r| r.ok).map(|r| r.wall_ms).collect();
        walls.sort_unstable();
        let median = walls.get(walls.len() / 2).copied().unwrap_or(0);
        let total: u128 = walls.iter().sum();
        let bytes: u64 = rs.iter().filter_map(|r| r.bytes_read).sum();
        eprintln!(
            "[clickbench/{engine}] ok {ok}/{} · median {median} ms · total {total} ms · total bytes_read {bytes}",
            rs.len()
        );
    }

    let out_path = ledger_out_path();
    if let Some(d) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(d).expect("ledger dir");
    }
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&records).expect("serialize"),
    )
    .expect("write ledger");
    eprintln!("ledger written: {out_path}");

    unsafe { std::env::remove_var("PROXIMADB_EXTERNAL_TABLE_ROOTS") };

    // Integrity only: every query has a ProximaDB record.
    let pdb = records.iter().filter(|r| r.engine == "proximadb").count();
    assert_eq!(pdb, queries.len(), "one ProximaDB record per query");
}
