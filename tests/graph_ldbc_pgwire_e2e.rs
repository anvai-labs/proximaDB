//! Graph (LDBC SNB-style) over pgwire — conformance harness (first cut).
//!
//! Models the LDBC Social Network Benchmark property graph as relational tables
//! (`person` nodes, `knows` friendship edges) queried over the PostgreSQL wire
//! protocol — graph traversals expressed as SQL joins and recursive CTEs, the open
//! standard for graph-over-SQL. Same harness model as the other modality suites:
//! CREATE → INSERT → MATERIALIZE → query one by one, routed by ProximaDB. Each
//! query that executes cleanly counts toward `GRAPH_RATCHET`.
//!
//! Edges are stored BOTH directions (knows is symmetric in LDBC) so 1-hop is a
//! direct lookup. The seeded graph: triangle 1-2-3, then 3-4, 4-5, 5-6.
//!
//!   RUST_LOG=proximadb=debug cargo test --test graph_ldbc_pgwire_e2e -- --nocapture

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

const GRAPH_RATCHET: usize = 10;

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

fn explain_err(e: &tokio_postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        format!("[{}] {}", db.code().code(), db.message())
    } else {
        e.to_string()
    }
}

/// Run a single-column query and parse each row's first cell as `i64`, returned
/// **sorted** for order-independent set comparison (ADR-040 `canon()`-style — row
/// order across engines/routes is a separate concern from result correctness; see
/// TD-185 for the materialized-read ORDER BY discrepancy this surfaced).
async fn col_ints(client: &tokio_postgres::Client, sql: &str) -> Vec<i64> {
    let mut v: Vec<i64> = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", explain_err(&e)))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                r.get(0).and_then(|s| s.parse::<i64>().ok())
            }
            _ => None,
        })
        .collect();
    v.sort_unstable();
    v
}

/// Run a two-column query and parse each row into an `(i64, i64)` pair, returned
/// **sorted** for order-independent set comparison (see [`col_ints`]).
async fn col_int_pairs(client: &tokio_postgres::Client, sql: &str) -> Vec<(i64, i64)> {
    let mut v: Vec<(i64, i64)> = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", explain_err(&e)))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some((
                r.get(0).unwrap_or_default().parse().unwrap_or(i64::MIN),
                r.get(1).unwrap_or_default().parse().unwrap_or(i64::MIN),
            )),
            _ => None,
        })
        .collect();
    v.sort_unstable();
    v
}

const SCHEMA: &[(&str, &str)] = &[
    (
        "person",
        "CREATE TABLE person (p_id INT PRIMARY KEY, p_firstname VARCHAR, p_lastname VARCHAR, p_creationdate DATE)",
    ),
    (
        "knows",
        "CREATE TABLE knows (k_person1 INT, k_person2 INT, k_creationdate DATE)",
    ),
];

fn data() -> Vec<String> {
    let persons = "INSERT INTO person (p_id, p_firstname, p_lastname, p_creationdate) VALUES \
        (1, 'Alice', 'A', DATE '2010-01-01'), (2, 'Bob', 'B', DATE '2010-02-01'), \
        (3, 'Carol', 'C', DATE '2010-03-01'), (4, 'Dan', 'D', DATE '2010-04-01'), \
        (5, 'Eve', 'E', DATE '2010-05-01'), (6, 'Frank', 'F', DATE '2010-06-01')"
        .to_string();
    // Undirected edges: triangle 1-2-3, then 3-4, 4-5, 5-6. Stored both ways.
    let undirected = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6)];
    let mut vals = Vec::new();
    for (a, b) in undirected {
        vals.push(format!("({a}, {b}, DATE '2011-01-01')"));
        vals.push(format!("({b}, {a}, DATE '2011-01-01')"));
    }
    let knows = format!(
        "INSERT INTO knows (k_person1, k_person2, k_creationdate) VALUES {}",
        vals.join(", ")
    );
    vec![persons, knows]
}

/// LDBC SNB-style traversal queries.
fn graph_queries() -> Vec<(&'static str, String)> {
    vec![
        // IS1-style: node profile lookup by id.
        ("g01_node_lookup", "select p_id, p_firstname, p_lastname from person where p_id = 1".to_string()),
        // 1-hop: direct friends of person 1 (expect {2,3}).
        ("g02_one_hop", "select k.k_person2 as friend from knows k where k.k_person1 = 1 order by friend".to_string()),
        // 1-hop with the friend's profile (join edge → node).
        ("g03_one_hop_profile", "select p.p_id, p.p_firstname from knows k join person p on k.k_person2 = p.p_id where k.k_person1 = 1 order by p.p_id".to_string()),
        // 2-hop friends-of-friends of person 1, excluding self and direct friends.
        ("g04_two_hop_fof", "select distinct k2.k_person2 as fof from knows k1 join knows k2 on k1.k_person2 = k2.k_person1 where k1.k_person1 = 1 and k2.k_person2 <> 1 and k2.k_person2 not in (select k_person2 from knows where k_person1 = 1) order by fof".to_string()),
        // Degree: friend count per person.
        ("g05_degree", "select k_person1 as person, count(*) as degree from knows group by k_person1 order by degree desc, person".to_string()),
        // Mutual friends of person 1 and person 4 (INTERSECT; expect {3}).
        ("g06_mutual", "select k_person2 as f from knows where k_person1 = 1 intersect select k_person2 as f from knows where k_person1 = 4".to_string()),
        // Triangles: 3-cycles a<b<c all mutually connected (expect 1-2-3).
        ("g07_triangle", "select e1.k_person1 as a, e1.k_person2 as b, e2.k_person2 as c from knows e1 join knows e2 on e1.k_person2 = e2.k_person1 join knows e3 on e2.k_person2 = e3.k_person1 and e3.k_person2 = e1.k_person1 where e1.k_person1 < e1.k_person2 and e1.k_person2 < e2.k_person2 order by a, b, c".to_string()),
        // Top-N most-connected persons (degree + orderby + limit).
        ("g08_top_degree", "select p.p_firstname, count(*) as degree from knows k join person p on k.k_person1 = p.p_id group by p.p_firstname order by degree desc, p.p_firstname limit 3".to_string()),
        // Friendships created after a date, joined to both endpoints' names.
        ("g09_edge_filter_join", "select a.p_firstname as f1, b.p_firstname as f2 from knows k join person a on k.k_person1 = a.p_id join person b on k.k_person2 = b.p_id where k.k_creationdate >= DATE '2011-01-01' and k.k_person1 < k.k_person2 order by f1, f2".to_string()),
        // Variable-hop reachability from person 1 via a recursive CTE (expect all 6).
        ("g10_recursive_reach", "with recursive reach(id) as (select 1 union select k.k_person2 from knows k join reach r on k.k_person1 = r.id) select count(distinct id) as reachable from reach".to_string()),
    ]
}

// 8 MB stack: this LDBC/pgwire e2e evaluates a `WITH RECURSIVE` transitive-closure
// CTE over the `knows` graph, whose bounded-but-deep recursion overflows the tokio
// worker default 2 MB stack intermittently under CI load. Run on a larger-stack
// runtime, matching the tpcds/tpch/duckdb pgwire e2e tests.
#[test]
fn graph_ldbc_pgwire_conformance() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(graph_ldbc_pgwire_conformance_inner());
}

async fn graph_ldbc_pgwire_conformance_inner() {
    let server = PgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    for (name, ddl) in SCHEMA {
        let _ = client
            .simple_query(&format!("DROP TABLE IF EXISTS {name}"))
            .await;
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("CREATE {name}: {}", explain_err(&e)));
    }
    eprintln!("✓ schema: {} tables created", SCHEMA.len());

    for sql in data() {
        client
            .simple_query(&sql)
            .await
            .unwrap_or_else(|e| panic!("INSERT failed: {}\n  sql: {sql}", explain_err(&e)));
    }
    eprintln!("✓ data: graph loaded");

    let mut materialized = 0;
    for (name, _) in SCHEMA {
        match client
            .simple_query(&format!("ALTER TABLE {name} MATERIALIZE"))
            .await
        {
            Ok(_) => materialized += 1,
            Err(e) => eprintln!("  · MATERIALIZE {name}: {}", explain_err(&e)),
        }
    }
    eprintln!("✓ materialize: {materialized}/{} tables", SCHEMA.len());

    let queries = graph_queries();
    let mut passed = Vec::new();
    let mut failed = Vec::new();
    for (id, sql) in &queries {
        match client.simple_query(sql).await {
            Ok(_) => {
                eprintln!("  ✓ {id}");
                passed.push(*id);
            }
            Err(e) => {
                eprintln!("  ✗ {id}: {}", explain_err(&e));
                failed.push((*id, explain_err(&e)));
            }
        }
    }

    eprintln!(
        "\n=== Graph LDBC pgwire conformance: {}/{} passed (ratchet {}) ===",
        passed.len(),
        queries.len(),
        GRAPH_RATCHET
    );
    if !failed.is_empty() {
        eprintln!("failing:");
        for (id, err) in &failed {
            eprintln!("  {id}: {err}");
        }
    }

    assert!(
        passed.len() >= GRAPH_RATCHET,
        "Graph LDBC conformance regressed: {} passed < ratchet {}",
        passed.len(),
        GRAPH_RATCHET
    );

    // ADR-040 P1 (TD-182): VALUE-correctness spot checks. The ratchet above only
    // proves these LDBC queries EXECUTE; assert the actual graph results so a
    // wrong-but-clean answer cannot pass. Comparison is order-independent (results
    // are sorted — ADR-040 `canon()` style; the materialized-read ORDER BY quirk
    // this surfaced is tracked as TD-185). Expected values are hand-derived from the
    // seed graph (triangle 1-2-3 + chain 3-4-5-6, edges stored both directions) —
    // adjacency 1:{2,3} 2:{1,3} 3:{1,2,4} 4:{3,5} 5:{4,6} 6:{5}.
    assert_eq!(
        col_ints(
            &client,
            "select k.k_person2 as friend from knows k where k.k_person1 = 1 order by friend"
        )
        .await,
        vec![2, 3],
        "g02: direct (1-hop) friends of person 1"
    );
    assert_eq!(
        col_ints(
            &client,
            "select distinct k2.k_person2 as fof from knows k1 join knows k2 on k1.k_person2 = k2.k_person1 where k1.k_person1 = 1 and k2.k_person2 <> 1 and k2.k_person2 not in (select k_person2 from knows where k_person1 = 1) order by fof"
        )
        .await,
        vec![4],
        "g04: two-hop friends-of-friends of 1 (exclude self + direct friends)"
    );
    assert_eq!(
        col_int_pairs(
            &client,
            "select k_person1 as person, count(*) as degree from knows group by k_person1 order by degree desc, person"
        )
        .await,
        vec![(1, 2), (2, 2), (3, 3), (4, 2), (5, 2), (6, 1)],
        "g05: degree per person (3 is the cut vertex, degree 3)"
    );
    assert_eq!(
        col_ints(
            &client,
            "select k_person2 as f from knows where k_person1 = 1 intersect select k_person2 as f from knows where k_person1 = 4"
        )
        .await,
        vec![3],
        "g06: mutual friends of 1 and 4 (INTERSECT) is person 3"
    );
    // g10 (TD-185a fixed): variable-hop reachability via a recursive CTE returns
    // ALL nodes reachable from person 1 — the whole triangle + chain. The WITH
    // header now engages the relational/OLAP route, the CTE name resolves as
    // engine-local (not a catalog table), and the DataFusion floor plans the
    // recursive CTE natively. Before the fix this silently returned ZERO rows
    // (a clean pgwire success the execution ratchet counted as a pass).
    assert_eq!(
        col_ints(
            &client,
            "with recursive reach(id) as (select 1 union select k.k_person2 from knows k join reach r on k.k_person1 = r.id) select count(distinct id) as reachable from reach"
        )
        .await,
        vec![6],
        "g10 (TD-185a): recursive-CTE reachability from person 1 covers all 6 nodes"
    );
}

/// ADR-025 / ADR-040 P2 (TD-182): graph DELETION-VECTOR accuracy — the gap the audit
/// flagged HIGH (no coverage today). A graph edge table with a single-column PK (so
/// the ADR-025 cold-path read-merge is eligible) is materialized, then edges are
/// DELETEd / INSERTed after the snapshot; a JOIN + GROUP BY out-degree traversal
/// (which routes to the DataFusion OLAP engine, where the merge lives) must reflect
/// the *merged* edges, not the stale snapshot. This is the graph analog of
/// `pgwire_olap_delta_merge_e2e`: without it, a deletion-vector bug at the graph layer
/// passes every existing graph test (the conformance suite never mutates after
/// MATERIALIZE). The materialized-but-unmutated `gperson` dimension exercises the
/// empty-delta fast path in the same query.
// 8 MB stack (same rationale as `graph_ldbc_pgwire_conformance`).
#[test]
fn graph_deletion_vector_reflects_edge_mutations() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(graph_deletion_vector_reflects_edge_mutations_inner());
}

async fn graph_deletion_vector_reflects_edge_mutations_inner() {
    let server = PgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // Idempotent setup (the catalog can survive local re-runs). Distinct names so
    // they never collide with the conformance test's person/knows.
    client.simple_query("DROP TABLE IF EXISTS edge").await.ok();
    client
        .simple_query("DROP TABLE IF EXISTS gperson")
        .await
        .ok();
    client
        .simple_query("CREATE TABLE gperson (p_id INT PRIMARY KEY, name VARCHAR)")
        .await
        .unwrap_or_else(|e| panic!("create gperson: {}", explain_err(&e)));
    client
        .simple_query("CREATE TABLE edge (e_id INT PRIMARY KEY, src INT, dst INT)")
        .await
        .unwrap_or_else(|e| panic!("create edge: {}", explain_err(&e)));
    for sql in [
        "INSERT INTO gperson (p_id, name) VALUES (1, 'A')",
        "INSERT INTO gperson (p_id, name) VALUES (2, 'B')",
        "INSERT INTO gperson (p_id, name) VALUES (3, 'C')",
        "INSERT INTO gperson (p_id, name) VALUES (4, 'D')",
        "INSERT INTO edge (e_id, src, dst) VALUES (1, 1, 2)",
        "INSERT INTO edge (e_id, src, dst) VALUES (2, 1, 3)",
        "INSERT INTO edge (e_id, src, dst) VALUES (3, 2, 4)",
    ] {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("seed `{sql}`: {}", explain_err(&e)));
    }
    for t in ["gperson", "edge"] {
        client
            .simple_query(&format!("ALTER TABLE {t} MATERIALIZE"))
            .await
            .unwrap_or_else(|e| panic!("materialize {t}: {}", explain_err(&e)));
    }

    // Post-snapshot edge mutations: delete edge 1 (drops 1→2), insert edge 4 (2→3).
    client
        .simple_query("DELETE FROM edge WHERE e_id = 1")
        .await
        .unwrap_or_else(|e| panic!("delete edge: {}", explain_err(&e)));
    client
        .simple_query("INSERT INTO edge (e_id, src, dst) VALUES (4, 2, 3)")
        .await
        .unwrap_or_else(|e| panic!("insert edge: {}", explain_err(&e)));

    // Out-degree per source = JOIN(edge, gperson) + GROUP BY → DataFusion OLAP route
    // (where the read-merge lives). col_int_pairs sorts → order-independent.
    let q = "select e.src as person, count(e.dst) as out_degree from edge e \
             join gperson p on e.src = p.p_id group by e.src order by e.src";

    // Kill-switch OFF → stale snapshot: src 1 has {e1,e2}=2, src 2 has {e3}=1.
    // SAFETY: env toggled only between awaited queries; the graph CI job runs with
    // --test-threads=1, so no in-flight request reads the env concurrently.
    unsafe { std::env::set_var("PROXIMADB_OLAP_DELTA_MERGE", "0") };
    assert_eq!(
        col_int_pairs(&client, q).await,
        vec![(1, 2), (2, 1)],
        "kill-switch: stale out-degrees (src 1 = 2 edges, src 2 = 1 edge)"
    );

    // Default-ON → deletion-vector merge: edge 1 deleted (src 1 now {e2}=1), edge 4
    // added (src 2 now {e3,e4}=2).
    unsafe { std::env::remove_var("PROXIMADB_OLAP_DELTA_MERGE") };
    assert_eq!(
        col_int_pairs(&client, q).await,
        vec![(1, 1), (2, 2)],
        "DV merge: out-degree reflects post-snapshot edge delete(1) + insert(4)"
    );
}
