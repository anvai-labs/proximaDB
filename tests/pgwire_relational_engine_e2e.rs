//! pgwire relational pipeline (PATH B) over REAL data — joins & aggregates.
//!
//! The legacy single-table SELECT path can't do JOIN / GROUP BY / aggregates.
//! This test proves the gated, additive wiring: queries that engage those
//! features route through the algebra engine (frontend → planner → executor)
//! reading REAL `RecordStorage` rows inserted over pgwire, while simple
//! single-table SELECTs stay on the legacy path. Default config (the new
//! pipeline is on by default; it falls through to legacy for queries the gate
//! doesn't engage and for pg-specific syntax).

use std::collections::BTreeSet;
use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;
use tokio_postgres::SimpleQueryMessage;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct PgwireTestServer {
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl PgwireTestServer {
    async fn start() -> anyhow::Result<Self> {
        let pg_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
        let tmp_data = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp_data.path().display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http_client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!(
                            "REST server didn't become ready on port {} within 20s",
                            rest_port
                        );
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            pg_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn pg_connection_string(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }
}

impl Drop for PgwireTestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

/// First value of column `col` across the rows, as text.
fn scalar(messages: &[SimpleQueryMessage], col: &str) -> Option<String> {
    messages.iter().find_map(|msg| match msg {
        SimpleQueryMessage::Row(row) => row.get(col).map(|s| s.to_string()),
        _ => None,
    })
}

/// Collect column `col` across all rows into a sorted set.
fn col_set(messages: &[SimpleQueryMessage], col: &str) -> BTreeSet<String> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get(col).map(|s| s.to_string()),
            _ => None,
        })
        .collect()
}

/// Collect `(a, b)` column pairs across all rows into a sorted set.
fn pair_set(messages: &[SimpleQueryMessage], a: &str, b: &str) -> BTreeSet<(String, String)> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => {
                Some((row.get(a)?.to_string(), row.get(b)?.to_string()))
            }
            _ => None,
        })
        .collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_relational_engine_serves_joins_and_aggregates_over_real_data() {
    let server = PgwireTestServer::start().await.expect("server start");

    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
            .await
            .expect("tokio-postgres connect");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let inv = format!("inv_{suffix}");
    let dept = format!("dept_{suffix}");
    let emp = format!("emp_{suffix}");

    // --- inv: single-table aggregate / GROUP BY + simple-OR regression ---
    client
        .simple_query(&format!(
            "CREATE TABLE {inv} (id VARCHAR PRIMARY KEY, status VARCHAR, qty BIGINT)"
        ))
        .await
        .expect("CREATE inv");
    for (id, status, qty) in [
        ("i1", "active", 5),
        ("i2", "active", 15),
        ("i3", "idle", 25),
        ("i4", "idle", 35),
    ] {
        client
            .simple_query(&format!(
                "INSERT INTO {inv} (id, status, qty) VALUES ('{id}', '{status}', {qty})"
            ))
            .await
            .expect("INSERT inv");
    }

    // --- dept / emp: 2-table INNER JOIN ---
    client
        .simple_query(&format!(
            "CREATE TABLE {dept} (id BIGINT PRIMARY KEY, dname VARCHAR)"
        ))
        .await
        .expect("CREATE dept");
    client
        .simple_query(&format!(
            "CREATE TABLE {emp} (id BIGINT PRIMARY KEY, dept_id BIGINT, ename VARCHAR)"
        ))
        .await
        .expect("CREATE emp");
    // dept 3 (hr) has no employee → unmatched right for RIGHT/FULL joins.
    for (id, dname) in [(1, "eng"), (2, "sales"), (3, "hr")] {
        client
            .simple_query(&format!(
                "INSERT INTO {dept} (id, dname) VALUES ({id}, '{dname}')"
            ))
            .await
            .expect("INSERT dept");
    }
    // emp 13 (dan) has dept_id 99 (no such dept) → unmatched left for FULL joins.
    for (id, dept_id, ename) in [
        (10, 1, "ann"),
        (11, 1, "bob"),
        (12, 2, "cas"),
        (13, 99, "dan"),
    ] {
        client
            .simple_query(&format!(
                "INSERT INTO {emp} (id, dept_id, ename) VALUES ({id}, {dept_id}, '{ename}')"
            ))
            .await
            .expect("INSERT emp");
    }

    sleep(Duration::from_millis(500)).await;

    // (1) COUNT(*) over real rows → 4.
    let rows = client
        .simple_query(&format!("SELECT COUNT(*) AS n FROM {inv}"))
        .await
        .expect("SELECT COUNT(*)");
    assert_eq!(
        scalar(&rows, "n").as_deref(),
        Some("4"),
        "COUNT(*) over real data"
    );

    // (2) GROUP BY over real rows → active:2, idle:2.
    let rows = client
        .simple_query(&format!(
            "SELECT status, COUNT(*) AS n FROM {inv} GROUP BY status"
        ))
        .await
        .expect("SELECT GROUP BY");
    assert_eq!(
        pair_set(&rows, "status", "n"),
        [
            ("active".to_string(), "2".to_string()),
            ("idle".to_string(), "2".to_string())
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "GROUP BY status counts"
    );

    // (2b) Filtered GROUP BY — the WHERE is pushed into the inv scan (the
    // reader is the sole predicate applier), so only qty >= 15 rows are
    // materialized: active:1 (i2), idle:2 (i3,i4).
    let rows = client
        .simple_query(&format!(
            "SELECT status, COUNT(*) AS n FROM {inv} WHERE qty >= 15 GROUP BY status"
        ))
        .await
        .expect("SELECT filtered GROUP BY");
    assert_eq!(
        pair_set(&rows, "status", "n"),
        [
            ("active".to_string(), "1".to_string()),
            ("idle".to_string(), "2".to_string())
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "predicate pushed into the scan yields correct filtered groups"
    );

    // (2c) PK-equality WHERE under an aggregate → the planner pushes id='i2'
    // into the inv scan and rewrites it to ScanAccess::PkLookup (point lookup),
    // exercising the relational reader's lookup_pk over real data. i2 is active,
    // so the single matched row groups as {(active, 1)}.
    let rows = client
        .simple_query(&format!(
            "SELECT status, COUNT(*) AS n FROM {inv} WHERE id = 'i2' GROUP BY status"
        ))
        .await
        .expect("SELECT PK-lookup aggregate");
    assert_eq!(
        pair_set(&rows, "status", "n"),
        [("active".to_string(), "1".to_string())]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "PK point lookup feeds the aggregate with the correct single row"
    );

    // (2c-batch) TD-128: discrete multi-key `PK IN (...)` under an aggregate →
    // the planner rewrites the IN-list into ScanAccess::PkLookupBatch (one probe
    // per key, predicate dropped) over real data, exercising the relational
    // reader's lookup_pk_batch. i1+i2 are both active → {(active, 2)}.
    let rows = client
        .simple_query(&format!(
            "SELECT status, COUNT(*) AS n FROM {inv} WHERE id IN ('i1', 'i2') GROUP BY status"
        ))
        .await
        .expect("SELECT PK-IN-list aggregate");
    assert_eq!(
        pair_set(&rows, "status", "n"),
        [("active".to_string(), "2".to_string())]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "PK IN-list batch lookup feeds the aggregate with both matched rows"
    );

    // (2d) Stage 2 — uncorrelated scalar subquery in a WHERE comparison. The
    // subquery `(SELECT avg(qty) FROM inv)` over [5,15,25,35] = 20 is hoisted to
    // a LEFT JOIN ON TRUE (AssertMaxOneRow-guarded) and the predicate becomes
    // `qty > col(hoisted)`, so only i3(25) and i4(35) qualify.
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {inv} WHERE qty > (SELECT avg(qty) FROM {inv})"
        ))
        .await
        .expect("SELECT WHERE scalar subquery");
    assert_eq!(
        col_set(&rows, "id"),
        ["i3".to_string(), "i4".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "WHERE qty > (SELECT avg(qty)) keeps only above-average rows"
    );

    // (2e) Part B gate tweak — a scalar subquery in the PROJECTION (no WHERE
    // subquery / join / aggregate / group-by) must engage PATH B so the Stage 1
    // hoist serves it. `(SELECT max(qty) FROM inv)` = 35 for every row. Before the
    // gate tweak this routed to the legacy single-table path, which can't serve a
    // scalar subquery.
    let rows = client
        .simple_query(&format!(
            "SELECT id, (SELECT max(qty) FROM {inv}) AS m FROM {inv}"
        ))
        .await
        .expect("SELECT projection scalar subquery");
    assert_eq!(
        col_set(&rows, "m"),
        ["35".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "projection scalar subquery yields the global max (35) on every row"
    );
    assert_eq!(
        col_set(&rows, "id"),
        ["i1", "i2", "i3", "i4"]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>(),
        "projection scalar subquery preserves all base rows"
    );

    // (3) INNER JOIN over real rows from two tables → 3 joined rows.
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} JOIN {dept} ON {emp}.dept_id = {dept}.id"
        ))
        .await
        .expect("SELECT JOIN");
    assert_eq!(
        pair_set(&rows, "ename", "dname"),
        [
            ("ann".to_string(), "eng".to_string()),
            ("bob".to_string(), "eng".to_string()),
            ("cas".to_string(), "sales".to_string()),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "INNER JOIN emp×dept"
    );

    // (3b) Join-predicate pushdown: WHERE on the dimension (dept) is pushed into
    // dept's scan (PK → point lookup), then joined → only eng employees.
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} JOIN {dept} ON {emp}.dept_id = {dept}.id WHERE {dept}.id = 1"
        ))
        .await
        .expect("SELECT join + dimension PK filter");
    assert_eq!(
        pair_set(&rows, "ename", "dname"),
        [
            ("ann".to_string(), "eng".to_string()),
            ("bob".to_string(), "eng".to_string()),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "dimension PK filter pushed through the join"
    );

    // (3c) LEFT-join right-side-filter correctness: the WHERE on the null-supplying
    // (right) side must NOT be pushed into dept's scan (planner can_push_to_right
    // is false for LEFT) — it stays above the join, so the post-join filter removes
    // null-extended rows per standard SQL. Only emp rows whose matched dept has
    // dname='sales' (cas) survive.
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} LEFT JOIN {dept} ON {emp}.dept_id = {dept}.id WHERE {dept}.dname = 'sales'"
        ))
        .await
        .expect("SELECT left join + right-side filter");
    assert_eq!(
        pair_set(&rows, "ename", "dname"),
        [("cas".to_string(), "sales".to_string())]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "LEFT-join right-side filter stays above the join (correct semantics)"
    );

    // (3d) ON-predicate pushdown: an INNER join with a right-side ON filter
    // (dept.dname='eng') pushes the filter into dept's scan (residual ON = the
    // equi, which upgrades to Hash). Correctness-equivalence: only emp rows
    // matched to the eng dept survive → {ann-eng, bob-eng}.
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} JOIN {dept} ON {emp}.dept_id = {dept}.id AND {dept}.dname = 'eng'"
        ))
        .await
        .expect("SELECT join + ON-side filter");
    assert_eq!(
        pair_set(&rows, "ename", "dname"),
        [
            ("ann".to_string(), "eng".to_string()),
            ("bob".to_string(), "eng".to_string()),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        "ON-side filter pushed into the dimension scan, result unchanged"
    );

    // (3e) RIGHT JOIN preserves the right (dept) side: dept 'hr' has no employee,
    // so it's emitted with a NULL ename. Distinct dnames include hr (pre-fix RIGHT
    // behaved like INNER and dropped it).
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} RIGHT JOIN {dept} ON {emp}.dept_id = {dept}.id"
        ))
        .await
        .expect("SELECT right join");
    assert_eq!(
        col_set(&rows, "dname"),
        set(&["eng", "sales", "hr"]),
        "RIGHT join preserves unmatched dept rows (hr)"
    );

    // (3f) FULL JOIN preserves both sides: emp 'dan' (dept_id 99) has no dept and
    // dept 'hr' has no employee. dan appears in the ename set (NULL dname) and hr
    // appears in the dname set (NULL ename).
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} FULL JOIN {dept} ON {emp}.dept_id = {dept}.id"
        ))
        .await
        .expect("SELECT full join");
    assert!(
        col_set(&rows, "ename").contains("dan"),
        "FULL join emits unmatched left (dan)"
    );
    assert!(
        col_set(&rows, "dname").contains("hr"),
        "FULL join emits unmatched right (hr)"
    );

    // (3f-corr) Stage 3 — correlated EXISTS. `dept` rows that have at least one
    // matching `emp` (dept_id = dept.id). The correlation `emp.dept_id = dept.id`
    // is lifted to the Semi-join ON: depts 1(eng) and 2(sales) have employees;
    // 3(hr) has none.
    let rows = client
        .simple_query(&format!(
            "SELECT dname FROM {dept} WHERE EXISTS \
             (SELECT 1 FROM {emp} WHERE {emp}.dept_id = {dept}.id)"
        ))
        .await
        .expect("SELECT correlated EXISTS");
    assert_eq!(
        col_set(&rows, "dname"),
        ["eng".to_string(), "sales".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "correlated EXISTS keeps only depts with a matching employee"
    );

    // (3f-corr2) Correlated NOT EXISTS → the complement: only 'hr' has no employee.
    let rows = client
        .simple_query(&format!(
            "SELECT dname FROM {dept} WHERE NOT EXISTS \
             (SELECT 1 FROM {emp} WHERE {emp}.dept_id = {dept}.id)"
        ))
        .await
        .expect("SELECT correlated NOT EXISTS");
    assert_eq!(
        col_set(&rows, "dname"),
        ["hr".to_string()].into_iter().collect::<BTreeSet<_>>(),
        "correlated NOT EXISTS keeps only the employee-less dept"
    );

    // (3f-corr3) Stage 3 — correlated SCALAR aggregate (self-correlated). For each
    // emp, the subquery `(SELECT max(e2.id) FROM emp e2 WHERE e2.dept_id = emp.dept_id)`
    // decorrelates to `emp LEFT JOIN (emp GROUP BY dept_id → max(id)) ON dept_id`,
    // and the WHERE keeps only the highest-id employee in each dept: bob(11,dept1),
    // cas(12,dept2), dan(13,dept99). ann(10) loses dept1 to bob.
    let rows = client
        .simple_query(&format!(
            "SELECT ename FROM {emp} WHERE id = \
             (SELECT max(e2.id) FROM {emp} e2 WHERE e2.dept_id = {emp}.dept_id)"
        ))
        .await
        .expect("SELECT correlated scalar aggregate");
    assert_eq!(
        col_set(&rows, "ename"),
        ["bob".to_string(), "cas".to_string(), "dan".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "correlated scalar max(id) per dept keeps the top-id employee of each dept"
    );

    // (3g) RIGHT JOIN ON-predicate pushdown to the null-supplying (left) side:
    // the left-only ON conjunct `emp.ename = 'ann'` is pushed into emp's scan
    // (emp is null-supplying under RIGHT), residual ON = the equi (→ Hash). All
    // dept rows stay preserved (dname set unchanged), but only ann matches the
    // ON filter, so the only non-NULL ename is 'ann' — bob (also eng) and cas
    // (sales) are excluded by the pushed-down filter (without pushdown the ename
    // set would be {ann, bob, cas}).
    let rows = client
        .simple_query(&format!(
            "SELECT ename, dname FROM {emp} RIGHT JOIN {dept} \
             ON {emp}.dept_id = {dept}.id AND {emp}.ename = 'ann'"
        ))
        .await
        .expect("SELECT right join with left-side ON predicate");
    assert_eq!(
        col_set(&rows, "dname"),
        set(&["eng", "sales", "hr"]),
        "RIGHT join still preserves all dept rows under ON pushdown"
    );
    assert_eq!(
        col_set(&rows, "ename"),
        set(&["ann"]),
        "left-side ON conjunct pushed: only ann matches, bob/cas excluded"
    );

    // (3h) EXPLAIN SELECT discloses the planned PHYSICAL plan (ADR-004 / P0 EXPLAIN
    // gate), not just the route. An engaging JOIN with a PK filter on the dimension
    // must surface compute_route=Native(Volcano) AND a physical_plan array whose
    // dept scan is a PkLookup (WHERE dept.id=1 → PK rewrite) under a Hash join.
    let rows = client
        .simple_query(&format!(
            "EXPLAIN SELECT ename, dname FROM {emp} JOIN {dept} \
             ON {emp}.dept_id = {dept}.id WHERE {dept}.id = 1"
        ))
        .await
        .expect("EXPLAIN join with PK filter");
    let plan = query_plan_cell(&rows);
    assert!(
        plan.contains("\"compute_route\"") && plan.contains("Native(Volcano)"),
        "EXPLAIN discloses the native route: {plan}"
    );
    assert!(
        plan.contains("physical_plan"),
        "EXPLAIN surfaces the physical plan for an engaging query: {plan}"
    );
    assert!(
        plan.contains("access=PkLookup"),
        "dept WHERE id=1 disclosed as a PkLookup scan: {plan}"
    );
    assert!(
        plan.contains("Join kind=INNER") && plan.contains("strategy=Hash"),
        "join strategy disclosed as Hash: {plan}"
    );
    // Plain EXPLAIN does NOT execute → no ANALYZE metrics (whole-query or per-op).
    assert!(
        !plan.contains("execution_rows")
            && !plan.contains("execution_elapsed_us")
            && !plan.contains("actual rows=")
            && !plan.contains("self=")
            && !plan.contains("\"exec\""),
        "plain EXPLAIN omits ANALYZE execution metrics (incl. structured exec[]): {plan}"
    );

    // (3h-trace) ADR-064 / TD-TRACE-1: EXPLAIN over a query the native frontend
    // DECLINES surfaces the REAL structured decline reason ({node, why}) — not
    // silence, and not the legacy text-guessed mask ("comma-separated table
    // list"). An IN-subquery under an OR is not a liftable conjunct, so native
    // lowering declines with node="subquery"; the query still engages PATH B
    // (the WHERE carries a subquery) so the decline is captured and disclosed.
    let rows = client
        .simple_query(&format!(
            "EXPLAIN SELECT ename FROM {emp} WHERE dept_id > 0 OR dept_id IN (SELECT id FROM {dept})"
        ))
        .await
        .expect("EXPLAIN declining subquery");
    let plan = query_plan_cell(&rows);
    assert!(
        plan.contains("declines") && plan.contains("subquery"),
        "EXPLAIN discloses the REAL structured lowering decline (node=subquery): {plan}"
    );
    assert!(
        !plan.contains("comma-separated"),
        "the decline is the real reason, NOT the legacy text-guessed mask: {plan}"
    );

    // (3h-batch) TD-128: EXPLAIN discloses a `PK IN (...)` rewrite as a
    // PkLookupBatch scan access path (the EXPLAIN-access-path evidence, not a
    // microbenchmark). Under an aggregate so the query engages PATH B.
    let rows = client
        .simple_query(&format!(
            "EXPLAIN SELECT status, COUNT(*) FROM {inv} WHERE id IN ('i1', 'i2') GROUP BY status"
        ))
        .await
        .expect("EXPLAIN PK IN-list aggregate");
    let plan = query_plan_cell(&rows);
    assert!(
        plan.contains("access=PkLookupBatch"),
        "PK IN-list disclosed as a PkLookupBatch scan: {plan}"
    );

    // (3h-secondary-fallback) TD-127: an equality on a NON-indexed, non-PK column
    // must NOT rewrite to SecondaryLookup (no secondary index is declared via DDL
    // yet — that path stays dormant). It stays a FullScan with the predicate
    // pushed, and still returns correct rows. (active = i1,i2 → count 2.)
    let rows = client
        .simple_query(&format!(
            "EXPLAIN SELECT status, COUNT(*) FROM {inv} WHERE status = 'active' GROUP BY status"
        ))
        .await
        .expect("EXPLAIN non-indexed equality aggregate");
    let plan = query_plan_cell(&rows);
    assert!(
        plan.contains("access=FullScan") && !plan.contains("access=SecondaryLookup"),
        "non-indexed equality stays a FullScan (secondary path dormant): {plan}"
    );
    let rows = client
        .simple_query(&format!(
            "SELECT status, COUNT(*) AS n FROM {inv} WHERE status = 'active' GROUP BY status"
        ))
        .await
        .expect("SELECT non-indexed equality aggregate");
    assert_eq!(
        pair_set(&rows, "status", "n"),
        [("active".to_string(), "2".to_string())]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        "non-indexed equality returns correct rows on the scan+filter path"
    );

    // (3h-analyze) EXPLAIN ANALYZE additionally EXECUTES the (read-only) plan and
    // reports measured rows + elapsed alongside the structural plan. emp×dept inner
    // join over the equi key matches ann/bob (eng) + cas (sales) = 3 rows.
    let rows = client
        .simple_query(&format!(
            "EXPLAIN ANALYZE SELECT ename, dname FROM {emp} JOIN {dept} \
             ON {emp}.dept_id = {dept}.id"
        ))
        .await
        .expect("EXPLAIN ANALYZE join");
    let plan = query_plan_cell(&rows);
    assert!(
        plan.contains("physical_plan") && plan.contains("\"execution_rows\": 3"),
        "EXPLAIN ANALYZE reports the actual row count (3): {plan}"
    );
    assert!(
        plan.contains("execution_elapsed_us"),
        "EXPLAIN ANALYZE reports measured elapsed time: {plan}"
    );
    // Per-operator actuals annotate the plan lines: actual rows + inclusive time +
    // self (exclusive) time. The inner join emits 3 rows.
    assert!(
        plan.contains("actual rows=3"),
        "EXPLAIN ANALYZE annotates per-operator actual rows (join=3): {plan}"
    );
    assert!(
        plan.contains("self="),
        "EXPLAIN ANALYZE annotates per-operator self (exclusive) time: {plan}"
    );
    // (3h-analyze-exec) TD-TRACE-1 Slice 2: the STRUCTURED per-operator exec[]
    // vector — `{op, rows_in, rows_out, ms_self, spill}` — alongside the text
    // annotations. The inner join's rows_out is 3 (== execution_rows), proving the
    // structured metrics are populated and consistent.
    assert!(
        plan.contains("\"exec\""),
        "EXPLAIN ANALYZE surfaces the structured exec[] vector: {plan}"
    );
    assert!(
        plan.contains("\"rows_out\": 3"),
        "exec[] carries the join's actual rows_out (3): {plan}"
    );

    // (3i) EXPLAIN of a simple (non-engaging) SELECT stays route-only: no physical
    // plan is attached (the query runs on the legacy path, not PATH B).
    let rows = client
        .simple_query(&format!("EXPLAIN SELECT id FROM {inv} WHERE id = 'i1'"))
        .await
        .expect("EXPLAIN simple select");
    let plan = query_plan_cell(&rows);
    assert!(
        plan.contains("\"compute_route\"") && !plan.contains("physical_plan"),
        "simple SELECT EXPLAIN is route-only (no physical_plan): {plan}"
    );

    // (3j) Semi join via uncorrelated IN-subquery: emp whose dept_id is in the set
    // of 'eng' dept ids (eng = id 1) → ann, bob (dept_id 1). The subquery lowers to
    // a Semi join (membership filter) — output is the emp (left) schema only.
    let rows = client
        .simple_query(&format!(
            "SELECT ename FROM {emp} WHERE dept_id IN (SELECT id FROM {dept} WHERE dname = 'eng')"
        ))
        .await
        .expect("SELECT IN-subquery (semi join)");
    assert_eq!(
        col_set(&rows, "ename"),
        set(&["ann", "bob"]),
        "IN-subquery → Semi join membership filter"
    );

    // (3k) Anti join via NOT EXISTS over an empty subquery: the subquery (dname that
    // doesn't exist) is empty, so the Anti join keeps every emp row.
    let rows = client
        .simple_query(&format!(
            "SELECT ename FROM {emp} WHERE NOT EXISTS (SELECT id FROM {dept} WHERE dname = 'nope')"
        ))
        .await
        .expect("SELECT NOT EXISTS (anti join)");
    assert_eq!(
        col_set(&rows, "ename"),
        set(&["ann", "bob", "cas", "dan"]),
        "NOT EXISTS over empty subquery → Anti join keeps all rows"
    );

    // (3l) INTERSECT: dept ids that are used as some emp's dept_id. dept = {1,2,3};
    // emp dept_ids = {1,1,2,99} → intersection {1,2} (hr=3 has no employee).
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {dept} INTERSECT SELECT dept_id FROM {emp}"
        ))
        .await
        .expect("SELECT INTERSECT");
    assert_eq!(
        col_set(&rows, "id"),
        set(&["1", "2"]),
        "INTERSECT → dept ids used by some employee"
    );

    // (3m) EXCEPT: dept ids NOT used by any employee → {3} (hr).
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {dept} EXCEPT SELECT dept_id FROM {emp}"
        ))
        .await
        .expect("SELECT EXCEPT");
    assert_eq!(
        col_set(&rows, "id"),
        set(&["3"]),
        "EXCEPT → dept ids with no employee"
    );

    // (3n) CASE in a projection (over an engaging JOIN): searched CASE classifies
    // each matched employee's department. ann/bob (dept_id 1) → 'eng-dept'; cas → 'other'
    // (dan dropped by the inner join).
    let rows = client
        .simple_query(&format!(
            "SELECT ename, CASE WHEN dept_id = 1 THEN 'eng-dept' ELSE 'other' END AS d \
             FROM {emp} JOIN {dept} ON {emp}.dept_id = {dept}.id"
        ))
        .await
        .expect("SELECT CASE over join");
    assert_eq!(
        pair_set(&rows, "ename", "d"),
        [("ann", "eng-dept"), ("bob", "eng-dept"), ("cas", "other")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect(),
        "CASE projection classifies joined rows"
    );

    // (3o) COALESCE over a LEFT JOIN: dan (dept_id 99) has no dept → dname is NULL →
    // COALESCE substitutes 'none'. Proves end-to-end NULL-aware evaluation.
    let rows = client
        .simple_query(&format!(
            "SELECT ename, COALESCE(dname, 'none') AS d \
             FROM {emp} LEFT JOIN {dept} ON {emp}.dept_id = {dept}.id"
        ))
        .await
        .expect("SELECT COALESCE over left join");
    assert_eq!(
        pair_set(&rows, "ename", "d"),
        [
            ("ann", "eng"),
            ("bob", "eng"),
            ("cas", "sales"),
            ("dan", "none"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect(),
        "COALESCE substitutes a default for the NULL (unmatched) side"
    );

    // (3p) NOT IN: dept ids NOT used by any employee → {3} (hr). Lowers to a
    // null-aware anti join (NULL-correct; here the subquery has no NULLs). Previously
    // NOT IN was declined and fell through to the legacy path.
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {dept} WHERE id NOT IN (SELECT dept_id FROM {emp})"
        ))
        .await
        .expect("SELECT NOT IN (null-aware anti join)");
    assert_eq!(
        col_set(&rows, "id"),
        set(&["3"]),
        "NOT IN → dept ids with no employee"
    );

    // (3q) Multi-column IN: a row constructor `(a, b) IN (SELECT x, y …)` lowers to a
    // Semi join with an AND of column equalities. Only the employee whose (id, dept_id)
    // matches ann's (10, 1) qualifies → {ann}.
    let rows = client
        .simple_query(&format!(
            "SELECT ename FROM {emp} WHERE (id, dept_id) IN \
             (SELECT id, dept_id FROM {emp} WHERE ename = 'ann')"
        ))
        .await
        .expect("SELECT multi-column IN");
    assert_eq!(
        col_set(&rows, "ename"),
        set(&["ann"]),
        "multi-column IN matches the full (id, dept_id) tuple"
    );

    // (4) Regression: a simple single-table OR SELECT still returns correct rows
    // (the gate keeps it on the hardened legacy path, not PATH B).
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {inv} WHERE status = 'active' OR qty >= 35"
        ))
        .await
        .expect("SELECT simple OR");
    assert_eq!(
        col_set(&rows, "id"),
        set(&["i1", "i2", "i4"]),
        "simple OR stays correct on the legacy path"
    );
}

/// Extract the single `QUERY PLAN` cell from an `EXPLAIN` simple-query response.
fn query_plan_cell(messages: &[SimpleQueryMessage]) -> String {
    messages
        .iter()
        .find_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get("QUERY PLAN").map(|s| s.to_string()),
            _ => None,
        })
        .expect("EXPLAIN returns a QUERY PLAN row")
}
