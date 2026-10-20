//! Multi-modal dual-engine "cost of a query" trace harness (co-design).
//!
//! Runs representative SQL-over-pgwire workloads for each modality (graph,
//! document, timeseries) against a scale-parameterized dataset, executing every
//! query on BOTH physical engines — the native/Volcano engine (pre-MATERIALIZE)
//! and DataFusion (post-MATERIALIZE) — and capturing, per query, the perf
//! (`wall_ms`) and the `observability::io_trace` billing snapshot (object-store
//! ops, bytes read/written, footer-cache hits, egress, per-engine `compute_ms`).
//!
//! Output: a console table (run with `--nocapture`) + a persistent JSON trace
//! artifact (`COST_TRACE_OUT`, default `target/cost_trace.json`) — the durable
//! co-design artifact the ephemeral conformance suites never produced.
//!
//! `#[ignore]` (not CI-gated; it's a perf/trace tool). Run on demand:
//!
//!   COST_TRACE_SCALE=4 cargo test --test cost_trace_pgwire_multimodal_e2e -- --ignored --nocapture
//!
//! NOTE (the headline co-design finding this surfaces): the native route emits
//! the full I/O trace; DataFusion currently records `compute_ms` only (its parquet
//! readers don't call `io_trace::record_*`), so DataFusion rows show compute with
//! ZERO I/O. Closing that gap is a documented follow-up (instrument
//! `src/datafusion/engine_adapters/*parquet_reader*`).

use proximadb::observability::io_trace::{self, IoTraceSnapshot};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tokio_postgres::{Client, SimpleQueryMessage};

/// One billing snapshot per query, pushed by the collecting observer at io_trace
/// scope close (`protocol.rs` wraps each statement in its own scope).
static CAPTURE: Mutex<Vec<IoTraceSnapshot>> = Mutex::new(Vec::new());

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
fn explain_err(e: &tokio_postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        format!("[{}] {}", db.code().code(), db.message())
    } else {
        e.to_string()
    }
}

// ---------------------------------------------------------------------------
// Scale-parameterized generators (deterministic — no RNG, reproducible).
// Batched multi-row INSERTs (chunked) to keep seeding fast.
// ---------------------------------------------------------------------------

fn chunked_inserts(table: &str, cols: &str, values: Vec<String>, chunk: usize) -> Vec<String> {
    values
        .chunks(chunk)
        .map(|c| format!("INSERT INTO {table} ({cols}) VALUES {}", c.join(", ")))
        .collect()
}

/// Graph: `n` persons, each "knows" `degree` neighbors (stored both directions).
fn generate_graph(n: usize, degree: usize) -> Vec<String> {
    let persons: Vec<String> = (1..=n)
        .map(|i| format!("({i}, 'P{i}', 'L{i}', DATE '2010-01-01')"))
        .collect();
    let mut edges = Vec::new();
    for a in 1..=n {
        for d in 1..=degree {
            let b = ((a + d - 1) % n) + 1; // deterministic neighbor, wraps
            if a != b {
                edges.push(format!("({a}, {b}, DATE '2011-01-01')"));
                edges.push(format!("({b}, {a}, DATE '2011-01-01')"));
            }
        }
    }
    let mut out = chunked_inserts(
        "person",
        "p_id, p_firstname, p_lastname, p_creationdate",
        persons,
        500,
    );
    out.extend(chunked_inserts(
        "knows",
        "k_person1, k_person2, k_creationdate",
        edges,
        500,
    ));
    out
}

/// Document: `n` JSON docs with nested author + tags (LDBC-like product/article).
fn generate_documents(n: usize) -> Vec<String> {
    let countries = ["CA", "US", "GB", "DE"];
    let names = ["Ann", "Bob", "Cy", "Di"];
    let vals: Vec<String> = (1..=n)
        .map(|i| {
            let kind = if i % 3 == 0 { "article" } else { "product" };
            let country = countries[i % countries.len()];
            let name = names[i % names.len()];
            let price = 5 + (i % 50);
            let in_stock = i % 2 == 0;
            let doc = format!(
                "{{\"title\":\"t{i}\",\"price\":{price},\"in_stock\":{in_stock},\
                 \"tags\":[\"x\",\"y\"],\"author\":{{\"name\":\"{name}\",\"country\":\"{country}\"}}}}"
            );
            format!("({i}, '{kind}', '{doc}')")
        })
        .collect();
    chunked_inserts("docs", "doc_id, kind, doc", vals, 300)
}

/// Timeseries: `hosts` hosts × `points` timestamps of CPU metrics (TSBS-like).
fn generate_timeseries(hosts: usize, points: usize) -> Vec<String> {
    let regions = ["us-east", "us-west", "eu-central"];
    let mut vals = Vec::new();
    for h in 0..hosts {
        let region = regions[h % regions.len()];
        for p in 0..points {
            // ts increments by 10 minutes; usage varies deterministically.
            let minute = p * 10;
            let hh = (minute / 60) % 24;
            let mm = minute % 60;
            let ts = format!("2016-01-01 {hh:02}:{mm:02}:00");
            let usage_user = (((h * 7 + p * 13) % 100) as f64) + 0.5;
            let usage_system = ((h * 3 + p * 5) % 40) as f64;
            let usage_idle = 100.0 - usage_user.min(100.0);
            vals.push(format!(
                "(TIMESTAMP '{ts}', 'host_{h}', '{region}', '{region}-1a', \
                 {usage_user}, {usage_system}, {usage_idle})"
            ));
        }
    }
    chunked_inserts(
        "cpu",
        "ts, hostname, region, datacenter, usage_user, usage_system, usage_idle",
        vals,
        400,
    )
}

// ---------------------------------------------------------------------------
// Query sets (reused verbatim from the per-modality conformance suites).
// ---------------------------------------------------------------------------

fn graph_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "g01_node_lookup",
            "select p_id, p_firstname, p_lastname from person where p_id = 1",
        ),
        (
            "g02_one_hop",
            "select k.k_person2 as friend from knows k where k.k_person1 = 1 order by friend",
        ),
        (
            "g03_one_hop_profile",
            "select p.p_id, p.p_firstname from knows k join person p on k.k_person2 = p.p_id where k.k_person1 = 1 order by p.p_id",
        ),
        (
            "g04_two_hop_fof",
            "select distinct k2.k_person2 as fof from knows k1 join knows k2 on k1.k_person2 = k2.k_person1 where k1.k_person1 = 1 and k2.k_person2 <> 1 order by fof",
        ),
        (
            "g05_degree",
            "select k_person1 as person, count(*) as degree from knows group by k_person1 order by degree desc, person limit 10",
        ),
        (
            "g06_mutual",
            "select k_person2 as f from knows where k_person1 = 1 intersect select k_person2 as f from knows where k_person1 = 4",
        ),
        (
            "g07_triangle",
            "select count(*) as triangles from knows e1 join knows e2 on e1.k_person2 = e2.k_person1 join knows e3 on e2.k_person2 = e3.k_person1 and e3.k_person2 = e1.k_person1 where e1.k_person1 < e1.k_person2 and e1.k_person2 < e2.k_person2",
        ),
        (
            "g08_top_degree",
            "select k_person1, count(*) as degree from knows group by k_person1 order by degree desc, k_person1 limit 3",
        ),
        (
            "g09_edge_filter_join",
            "select count(*) as n from knows k join person a on k.k_person1 = a.p_id join person b on k.k_person2 = b.p_id where k.k_creationdate >= DATE '2011-01-01' and k.k_person1 < k.k_person2",
        ),
        (
            "g10_recursive_reach",
            "with recursive reach(id) as (select 1 union select k.k_person2 from knows k join reach r on k.k_person1 = r.id) select count(distinct id) as reachable from reach",
        ),
    ]
}

fn doc_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("d01_count_all", "select count(*) as n from docs"),
        (
            "d02_filter_col",
            "select count(*) as n from docs where kind = 'product'",
        ),
        (
            "d03_count_by_kind",
            "select kind, count(*) as n from docs group by kind order by kind",
        ),
        (
            "d04_arrow_text",
            "select doc_id, doc->>'title' as title from docs order by doc_id limit 10",
        ),
        (
            "d06_filter_json_scalar",
            "select count(*) as n from docs where (doc->>'price')::int > 30",
        ),
        (
            "d08_nested_path",
            "select doc_id, doc->'author'->>'name' as author from docs order by doc_id limit 10",
        ),
        (
            "d09_group_by_json",
            "select doc->'author'->>'country' as country, count(*) as n from docs group by doc->'author'->>'country' order by country",
        ),
        (
            "d10_json_extract_fn",
            "select doc_id, json_extract_path_text(doc, 'title') as title from docs order by doc_id limit 10",
        ),
        (
            "d11_bool_field",
            "select count(*) as n from docs where (doc->>'in_stock')::boolean = true",
        ),
    ]
}

fn ts_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "t01_single_groupby",
            "select date_trunc('hour', ts) as hour, max(usage_user) as max_user from cpu where hostname = 'host_0' group by date_trunc('hour', ts) order by hour",
        ),
        (
            "t02_double_groupby",
            "select date_trunc('hour', ts) as hour, hostname, avg(usage_user) as au from cpu group by date_trunc('hour', ts), hostname order by hour, hostname limit 20",
        ),
        (
            "t03_max_all",
            "select hostname, max(usage_user) as mu from cpu group by hostname order by hostname limit 20",
        ),
        (
            "t04_high_cpu",
            "select count(*) as n from cpu where usage_user > 90.0",
        ),
        (
            "t05_high_cpu_by_region",
            "select region, count(*) as n from cpu where usage_user > 90.0 group by region order by region",
        ),
        (
            "t06_groupby_limit",
            "select date_trunc('hour', ts) as hour, max(usage_user) as max_user from cpu group by date_trunc('hour', ts) order by max_user desc limit 5",
        ),
        (
            "t07_last_point",
            "select hostname, ts, usage_user from (select hostname, ts, usage_user, row_number() over (partition by hostname order by ts desc) as rn from cpu) t where rn = 1 order by hostname limit 20",
        ),
        (
            "t08_extract_hour",
            "select extract(hour from ts) as hr, count(*) as n from cpu group by extract(hour from ts) order by hr",
        ),
        (
            "t09_region_having",
            "select region, avg(usage_user) as au from cpu group by region having avg(usage_user) > 40.0 order by region",
        ),
    ]
}

// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct TraceRecord {
    modality: String,
    query: String,
    engine: String,
    ok: bool,
    error: Option<String>,
    rows: usize,
    wall_ms: u128,
    #[serde(flatten)]
    snapshot: IoTraceSnapshot,
}

/// Run one query: clear the capture, time the client round-trip, drain the
/// per-query billing snapshot (the observer fires server-side at scope close,
/// so poll briefly for it). Returns `Err((wall_ms, msg))` if the engine could
/// not execute the query (a capability gap is data, not a crash).
async fn measure(
    client: &Client,
    sql: &str,
) -> Result<(usize, u128, IoTraceSnapshot), (u128, String)> {
    CAPTURE.lock().expect("lock").clear();
    let t0 = Instant::now();
    let res = client.simple_query(sql).await;
    let wall_ms = t0.elapsed().as_millis();
    match res {
        Ok(msgs) => {
            let rows = msgs
                .iter()
                .filter(|m| matches!(m, SimpleQueryMessage::Row(_)))
                .count();
            let mut snap = IoTraceSnapshot::default();
            for _ in 0..60 {
                if let Some(s) = CAPTURE.lock().expect("lock").pop() {
                    snap = s;
                    break;
                }
                sleep(Duration::from_millis(5)).await;
            }
            Ok((rows, wall_ms, snap))
        }
        Err(e) => Err((wall_ms, explain_err(&e))),
    }
}

fn push_record(
    out: &mut Vec<TraceRecord>,
    modality: &str,
    query: &str,
    engine: &str,
    result: Result<(usize, u128, IoTraceSnapshot), (u128, String)>,
) {
    let rec = match result {
        Ok((rows, wall_ms, snapshot)) => TraceRecord {
            modality: modality.into(),
            query: query.into(),
            engine: engine.into(),
            ok: true,
            error: None,
            rows,
            wall_ms,
            snapshot,
        },
        Err((wall_ms, err)) => TraceRecord {
            modality: modality.into(),
            query: query.into(),
            engine: engine.into(),
            ok: false,
            error: Some(err),
            rows: 0,
            wall_ms,
            snapshot: IoTraceSnapshot::default(),
        },
    };
    out.push(rec);
}

async fn seed(client: &Client, schema: &[(&str, &str)], inserts: Vec<String>) {
    for (name, ddl) in schema {
        let _ = client
            .simple_query(&format!("DROP TABLE IF EXISTS {name}"))
            .await;
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("CREATE {name}: {}", explain_err(&e)));
    }
    for sql in &inserts {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("INSERT: {}", explain_err(&e)));
    }
}

async fn run_modality(
    client: &Client,
    modality: &str,
    schema: &[(&str, &str)],
    inserts: Vec<String>,
    queries: Vec<(&'static str, &'static str)>,
    out: &mut Vec<TraceRecord>,
) {
    let n_inserts = inserts.len();
    seed(client, schema, inserts).await;
    eprintln!(
        "[{modality}] seeded ({n_inserts} insert batches across {} tables)",
        schema.len()
    );

    // Native/Volcano route (pre-MATERIALIZE).
    for (id, sql) in &queries {
        let r = measure(client, sql).await;
        push_record(out, modality, id, "native", r);
    }

    // Flip to parquet-backed → DataFusion route.
    for (name, _) in schema {
        if let Err(e) = client
            .simple_query(&format!("ALTER TABLE {name} MATERIALIZE"))
            .await
        {
            eprintln!("[{modality}] · MATERIALIZE {name}: {}", explain_err(&e));
        }
    }

    // DataFusion route (post-MATERIALIZE).
    for (id, sql) in &queries {
        let r = measure(client, sql).await;
        push_record(out, modality, id, "datafusion", r);
    }
}

fn compute_ms_total(s: &IoTraceSnapshot) -> u64 {
    s.compute_ms.values().sum()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "perf/trace harness — run on demand with --ignored --nocapture"]
async fn cost_trace_multimodal() {
    let scale: usize = std::env::var("COST_TRACE_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    eprintln!("=== cost-trace harness (COST_TRACE_SCALE={scale}) ===");

    let server = TestPgServer::start().await.expect("server start");
    // Override the process billing observer (installed by db.start()) with a
    // collector that captures each query's snapshot for this harness.
    io_trace::set_billing_observer(Some(Box::new(|snap: &IoTraceSnapshot, _tenant| {
        CAPTURE.lock().expect("capture lock").push(snap.clone());
    })));
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let graph_schema: &[(&str, &str)] = &[
        (
            "person",
            "CREATE TABLE person (p_id INT PRIMARY KEY, p_firstname VARCHAR, p_lastname VARCHAR, p_creationdate DATE)",
        ),
        (
            "knows",
            "CREATE TABLE knows (k_person1 INT, k_person2 INT, k_creationdate DATE)",
        ),
    ];
    let doc_schema: &[(&str, &str)] = &[(
        "docs",
        "CREATE TABLE docs (doc_id INT PRIMARY KEY, kind VARCHAR, doc JSON)",
    )];
    let ts_schema: &[(&str, &str)] = &[(
        "cpu",
        "CREATE TABLE cpu (ts TIMESTAMP, hostname VARCHAR, region VARCHAR, datacenter VARCHAR, usage_user DOUBLE PRECISION, usage_system DOUBLE PRECISION, usage_idle DOUBLE PRECISION)",
    )];

    let mut records: Vec<TraceRecord> = Vec::new();

    run_modality(
        &client,
        "graph",
        graph_schema,
        generate_graph(1000 * scale, 8),
        graph_queries(),
        &mut records,
    )
    .await;
    run_modality(
        &client,
        "document",
        doc_schema,
        generate_documents(1500 * scale),
        doc_queries(),
        &mut records,
    )
    .await;
    run_modality(
        &client,
        "timeseries",
        ts_schema,
        generate_timeseries(40, 50 * scale),
        ts_queries(),
        &mut records,
    )
    .await;

    // --- console table ---
    eprintln!(
        "\n{:<10} {:<22} {:<11} {:>4} {:>6} {:>8} {:>7} {:>8} {:>9} {:>10} {:>7}",
        "modality",
        "query",
        "engine",
        "ok",
        "rows",
        "wall_ms",
        "cmp_ms",
        "get_ops",
        "range_get",
        "bytes_read",
        "footer%"
    );
    eprintln!("{}", "-".repeat(122));
    for r in &records {
        if !r.ok {
            eprintln!(
                "{:<10} {:<22} {:<11} {:>4} {:>6} {:>8}  (unsupported: {})",
                r.modality,
                r.query,
                r.engine,
                "ERR",
                "-",
                r.wall_ms,
                r.error.as_deref().unwrap_or("?"),
            );
            continue;
        }
        let footer = r
            .snapshot
            .footer_hit_ratio()
            .map(|f| format!("{:.0}", f * 100.0))
            .unwrap_or_else(|| "-".into());
        eprintln!(
            "{:<10} {:<22} {:<11} {:>4} {:>6} {:>8} {:>7} {:>8} {:>9} {:>10} {:>7}",
            r.modality,
            r.query,
            r.engine,
            "ok",
            r.rows,
            r.wall_ms,
            compute_ms_total(&r.snapshot),
            r.snapshot.total_ops(),
            r.snapshot.range_gets,
            r.snapshot.bytes_read,
            footer,
        );
    }

    // --- per-engine roll-up (the co-design asymmetry, as data) ---
    for engine in ["native", "datafusion"] {
        let rs: Vec<&TraceRecord> = records.iter().filter(|r| r.engine == engine).collect();
        let ran = rs.iter().filter(|r| r.ok).count();
        let wall: u128 = rs.iter().filter(|r| r.ok).map(|r| r.wall_ms).sum();
        let cmp: u64 = rs.iter().map(|r| compute_ms_total(&r.snapshot)).sum();
        let ops: u64 = rs.iter().map(|r| r.snapshot.total_ops()).sum();
        let bytes: u64 = rs.iter().map(|r| r.snapshot.bytes_read).sum();
        eprintln!(
            "\n[{engine}] {ran}/{} queries ran — Σwall={wall}ms Σcompute={cmp}ms Σobject_ops={ops} Σbytes_read={bytes}",
            rs.len()
        );
    }
    eprintln!(
        "NOTE: DataFusion object_ops/bytes_read are 0 by design today (its parquet readers \
         don't feed io_trace) — the metering gap this harness surfaces. compute_ms is metered for both."
    );

    // --- persistent JSON trace artifact ---
    let out_path =
        std::env::var("COST_TRACE_OUT").unwrap_or_else(|_| "target/cost_trace.json".into());
    let json = serde_json::to_string_pretty(&records).expect("serialize trace");
    if let Err(e) = std::fs::write(&out_path, &json) {
        eprintln!("could not write trace artifact to {out_path}: {e}");
    } else {
        eprintln!(
            "\n✓ co-design trace written: {out_path} ({} records)",
            records.len()
        );
    }

    // --- ratchet: every query must have executed on both engines ---
    assert_eq!(
        records.len(),
        2 * (graph_queries().len() + doc_queries().len() + ts_queries().len()),
        "every query should run on both engines"
    );
}
