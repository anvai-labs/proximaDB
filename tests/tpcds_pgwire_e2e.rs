//! TPC-DS over pgwire — ANSI SQL conformance harness (first-cut subset).
//!
//! Same model as the TPC-H harness: a coherent star-schema subset
//! (date_dim / item / store / customer / customer_address / store_sales) is
//! created, seeded, and materialized to Parquet, then a representative slice of
//! TPC-DS queries runs ONE BY ONE over the PostgreSQL wire protocol — routed by
//! ProximaDB to the DataFusion OLAP engine, never bypassing pgwire.
//!
//! The query slice is chosen to exercise SQL features TPC-H does NOT: window
//! functions, ROLLUP / GROUPING SETS, INTERSECT / EXCEPT, and CTEs. Each query
//! that executes cleanly counts toward `TPCDS_RATCHET`.
//!
//!   RUST_LOG=proximadb=debug cargo test --test tpcds_pgwire_e2e -- --nocapture

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

/// TPC-DS subset queries expected to execute cleanly over pgwire (the ratchet).
const TPCDS_RATCHET: usize = 18;
/// ADR-040 accuracy ratchet (TD-182 P1): TPC-DS queries with full-row anchors
/// over the seed (the shapes not already anchored in the conformance test's
/// ROLLUP/CUBE/rank/RANGE-GROUPS/INTERSECT section). Only goes up.
const TPCDS_ACCURATE_RATCHET: usize = 10;

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

/// Normalize a result cell for value comparison: numeric cells are rounded to 2
/// decimals (collapsing f64 decimal-accumulation error on `DOUBLE PRECISION` price
/// sums, e.g. 69.4799… → "69.48") and rendered minimally (whole numbers without a
/// decimal); SQL NULL → "NULL"; other text passes through.
fn norm_cell(s: &str) -> String {
    match s.parse::<f64>() {
        Ok(f) => {
            let r = (f * 100.0).round() / 100.0;
            if r.fract() == 0.0 {
                format!("{}", r as i64)
            } else {
                format!("{r}")
            }
        }
        Err(_) => s.to_string(),
    }
}

/// Run a query and return its rows as normalized text cells, **sorted** for
/// order-independent set comparison (ADR-040 `canon()` style; row order across
/// routes is a separate concern — see TD-185).
async fn canon_rows(client: &tokio_postgres::Client, sql: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", explain_err(&e)))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| norm_cell(r.get(i).unwrap_or("NULL")))
                    .collect(),
            ),
            _ => None,
        })
        .collect();
    rows.sort();
    rows
}

/// Core TPC-DS star-schema subset (standard column names, subset of columns).
const SCHEMA: &[(&str, &str)] = &[
    (
        "date_dim",
        "CREATE TABLE date_dim (d_date_sk INT PRIMARY KEY, d_date DATE, d_year INT, d_moy INT, d_qoy INT, d_dom INT, d_dow INT)",
    ),
    (
        "item",
        "CREATE TABLE item (i_item_sk INT PRIMARY KEY, i_item_id VARCHAR, i_brand_id INT, i_brand VARCHAR, i_class_id INT, i_class VARCHAR, i_category_id INT, i_category VARCHAR, i_manufact_id INT, i_current_price DOUBLE PRECISION)",
    ),
    (
        "store",
        "CREATE TABLE store (s_store_sk INT PRIMARY KEY, s_store_id VARCHAR, s_store_name VARCHAR, s_state VARCHAR)",
    ),
    (
        "customer",
        "CREATE TABLE customer (c_customer_sk INT PRIMARY KEY, c_customer_id VARCHAR, c_first_name VARCHAR, c_last_name VARCHAR, c_current_addr_sk INT, c_birth_country VARCHAR)",
    ),
    (
        "customer_address",
        "CREATE TABLE customer_address (ca_address_sk INT PRIMARY KEY, ca_state VARCHAR, ca_city VARCHAR, ca_zip VARCHAR, ca_country VARCHAR, ca_gmt_offset DOUBLE PRECISION)",
    ),
    (
        "store_sales",
        "CREATE TABLE store_sales (ss_sold_date_sk INT, ss_item_sk INT, ss_store_sk INT, ss_customer_sk INT, ss_addr_sk INT, ss_ticket_number INT, ss_quantity INT, ss_sales_price DOUBLE PRECISION, ss_ext_sales_price DOUBLE PRECISION, ss_ext_discount_amt DOUBLE PRECISION, ss_net_profit DOUBLE PRECISION)",
    ),
];

/// Tiny coherent dataset (keys join across the star).
const DATA: &[&str] = &[
    "INSERT INTO date_dim (d_date_sk, d_date, d_year, d_moy, d_qoy, d_dom, d_dow) VALUES (24001, DATE '2000-11-01', 2000, 11, 4, 1, 3), (24002, DATE '2000-11-15', 2000, 11, 4, 15, 3), (24003, DATE '2000-12-10', 2000, 12, 4, 10, 0), (23001, DATE '1999-11-05', 1999, 11, 4, 5, 5)",
    "INSERT INTO item (i_item_sk, i_item_id, i_brand_id, i_brand, i_class_id, i_class, i_category_id, i_category, i_manufact_id, i_current_price) VALUES (1, 'ITEM001', 11, 'brandA', 1, 'classX', 1, 'Electronics', 100, 19.99), (2, 'ITEM002', 12, 'brandB', 1, 'classX', 1, 'Electronics', 101, 29.50), (3, 'ITEM003', 21, 'brandC', 2, 'classY', 2, 'Books', 200, 9.99), (4, 'ITEM004', 22, 'brandD', 2, 'classY', 2, 'Books', 201, 14.00)",
    "INSERT INTO store (s_store_sk, s_store_id, s_store_name, s_state) VALUES (1, 'STORE01', 'Downtown', 'CA'), (2, 'STORE02', 'Uptown', 'NY')",
    "INSERT INTO customer_address (ca_address_sk, ca_state, ca_city, ca_zip, ca_country, ca_gmt_offset) VALUES (1, 'CA', 'San Jose', '95101', 'United States', -8.0), (2, 'NY', 'Albany', '12201', 'United States', -5.0)",
    "INSERT INTO customer (c_customer_sk, c_customer_id, c_first_name, c_last_name, c_current_addr_sk, c_birth_country) VALUES (1, 'CUST001', 'Ann', 'Lee', 1, 'CANADA'), (2, 'CUST002', 'Bob', 'Ng', 2, 'MEXICO'), (3, 'CUST003', 'Cy', 'Ortiz', 1, 'CANADA')",
    "INSERT INTO store_sales (ss_sold_date_sk, ss_item_sk, ss_store_sk, ss_customer_sk, ss_addr_sk, ss_ticket_number, ss_quantity, ss_sales_price, ss_ext_sales_price, ss_ext_discount_amt, ss_net_profit) VALUES (24001, 1, 1, 1, 1, 1001, 2, 19.99, 39.98, 0.0, 12.0), (24001, 2, 1, 2, 2, 1002, 1, 29.50, 29.50, 2.0, 8.0), (24002, 3, 2, 1, 1, 1003, 5, 9.99, 49.95, 5.0, 20.0), (24003, 4, 2, 3, 1, 1004, 3, 14.00, 42.00, 0.0, 10.0), (23001, 1, 1, 1, 1, 1005, 1, 19.99, 19.99, 0.0, 6.0)",
];

/// TPC-DS subset: real queries (q3/q42/q52/q55/q98 adapted to the subset
/// columns) plus targeted probes for features TPC-H lacks.
fn tpcds_queries() -> Vec<(&'static str, String)> {
    vec![
        // q42 — year/category revenue.
        ("q42", "select dt.d_year, item.i_category_id, item.i_category, sum(ss_ext_sales_price) as revenue from date_dim dt, store_sales, item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and item.i_manufact_id = 100 and dt.d_moy = 11 and dt.d_year = 2000 group by dt.d_year, item.i_category_id, item.i_category order by revenue desc, dt.d_year".to_string()),
        // q52 — brand revenue by year/month.
        ("q52", "select dt.d_year, item.i_brand_id as brand_id, item.i_brand as brand, sum(ss_ext_sales_price) as ext_price from date_dim dt, store_sales, item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and dt.d_moy = 11 and dt.d_year = 2000 group by dt.d_year, item.i_brand, item.i_brand_id order by dt.d_year, ext_price desc, brand_id".to_string()),
        // q55 — single-brand revenue.
        ("q55", "select i_brand_id as brand_id, i_brand as brand, sum(ss_ext_sales_price) as ext_price from date_dim, store_sales, item where store_sales.ss_sold_date_sk = date_dim.d_date_sk and store_sales.ss_item_sk = item.i_item_sk and i_manufact_id = 100 and d_moy = 11 and d_year = 2000 group by i_brand, i_brand_id order by ext_price desc, i_brand_id".to_string()),
        // q3 — manufacturer revenue by year/brand.
        ("q3", "select dt.d_year, item.i_brand_id as brand_id, item.i_brand as brand, sum(ss_ext_sales_price) as sum_agg from date_dim dt, store_sales, item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and item.i_manufact_id = 100 and dt.d_moy = 11 group by dt.d_year, item.i_brand, item.i_brand_id order by dt.d_year, sum_agg desc, brand_id".to_string()),
        // q98 — WINDOW: ratio of each item's revenue to its class total.
        ("q98", "select i_item_id, i_category, i_class, i_current_price, sum(ss_ext_sales_price) as itemrevenue, sum(ss_ext_sales_price)*100/sum(sum(ss_ext_sales_price)) over (partition by i_class) as revenueratio from store_sales, item, date_dim where ss_item_sk = i_item_sk and i_category in ('Electronics', 'Books') and ss_sold_date_sk = d_date_sk and d_year = 2000 group by i_item_id, i_category, i_class, i_current_price order by i_category, i_class, i_item_id, revenueratio".to_string()),
        // WINDOW: rank items by revenue within category.
        ("win_rank", "select i_category, i_brand, sum(ss_ext_sales_price) as rev, rank() over (partition by i_category order by sum(ss_ext_sales_price) desc) as rnk from store_sales, item where ss_item_sk = i_item_sk group by i_category, i_brand order by i_category, rnk".to_string()),
        // WINDOW: running sum (frame).
        ("win_running", "select d_date, sum(ss_ext_sales_price) as daily, sum(sum(ss_ext_sales_price)) over (order by d_date rows between unbounded preceding and current row) as running from store_sales, date_dim where ss_sold_date_sk = d_date_sk group by d_date order by d_date".to_string()),
        // WINDOW: RANGE value-offset frame (must NOT execute as ROWS).
        ("win_range", "select ss_quantity, sum(ss_net_profit) over (order by ss_quantity range between 1 preceding and current row) as s from store_sales order by ss_quantity".to_string()),
        // WINDOW: GROUPS peer-group frame (must NOT execute as ROWS).
        ("win_groups", "select ss_quantity, sum(ss_net_profit) over (order by ss_quantity groups between 1 preceding and current row) as s from store_sales order by ss_quantity".to_string()),
        // ROLLUP.
        ("rollup", "select i_category, i_class, sum(ss_net_profit) as profit from store_sales, item where ss_item_sk = i_item_sk group by rollup(i_category, i_class) order by i_category, i_class".to_string()),
        // GROUPING SETS.
        ("grouping_sets", "select i_category, i_brand, sum(ss_quantity) as qty from store_sales, item where ss_item_sk = i_item_sk group by grouping sets ((i_category), (i_brand), ()) order by i_category, i_brand".to_string()),
        // CUBE.
        ("cube", "select d_year, i_category, sum(ss_ext_sales_price) as rev from store_sales, item, date_dim where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk group by cube(d_year, i_category) order by d_year, i_category".to_string()),
        // INTERSECT — customers who bought Electronics AND Books.
        ("intersect", "select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Electronics' intersect select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Books'".to_string()),
        // EXCEPT — customers who bought Electronics but NOT Books.
        ("except", "select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Electronics' except select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Books'".to_string()),
        // CTE (WITH).
        ("cte", "with cat_rev as (select i_category, sum(ss_ext_sales_price) as rev from store_sales, item where ss_item_sk = i_item_sk group by i_category) select i_category, rev from cat_rev where rev > 10 order by rev desc".to_string()),
        // COUNT(DISTINCT) + HAVING.
        ("count_distinct", "select i_category, count(distinct ss_customer_sk) as buyers from store_sales, item where ss_item_sk = i_item_sk group by i_category having count(distinct ss_customer_sk) >= 1 order by buyers desc, i_category".to_string()),
        // Correlated scalar subquery.
        ("correlated", "select i_item_sk, i_current_price from item where i_current_price > (select avg(i_current_price) from item i2 where i2.i_category = item.i_category) order by i_item_sk".to_string()),
        // CASE + aggregate.
        ("case_agg", "select i_category, sum(case when ss_net_profit > 10 then 1 else 0 end) as hi, sum(case when ss_net_profit <= 10 then 1 else 0 end) as lo from store_sales, item where ss_item_sk = i_item_sk group by i_category order by i_category".to_string()),
    ]
}

/// Same 8MB stack boost as TPC-H (see tpch_pgwire_e2e.rs for rationale).
#[test]
fn tpcds_pgwire_conformance() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(tpcds_pgwire_conformance_inner());
}

async fn tpcds_pgwire_conformance_inner() {
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

    for sql in DATA {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("INSERT failed: {}\n  sql: {sql}", explain_err(&e)));
    }
    eprintln!("✓ data: {} insert batches loaded", DATA.len());

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

    let queries = tpcds_queries();
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
        "\n=== TPC-DS pgwire conformance: {}/{} passed (ratchet {}) ===",
        passed.len(),
        queries.len(),
        TPCDS_RATCHET
    );
    if !failed.is_empty() {
        eprintln!("failing:");
        for (id, err) in &failed {
            eprintln!("  {id}: {err}");
        }
    }

    assert!(
        passed.len() >= TPCDS_RATCHET,
        "TPC-DS conformance regressed: {} passed < ratchet {}",
        passed.len(),
        TPCDS_RATCHET
    );

    // Value-correctness spot check on the INTERSECT: customers who bought both an
    // Electronics item (1,2) and a Books item (3,4). From the seeded sales only
    // customer 1 bought from both categories → exactly one row, ss_customer_sk=1.
    // Proves set-op semantics compute correctly, not just that they parse.
    let intersect = &tpcds_queries()
        .into_iter()
        .find(|(id, _)| *id == "intersect")
        .expect("intersect query present")
        .1;
    let rows: Vec<_> = client
        .simple_query(intersect)
        .await
        .expect("intersect re-run")
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 1, "INTERSECT should yield exactly 1 customer");
    assert_eq!(
        rows[0].get(0).unwrap_or(""),
        "1",
        "INTERSECT customer should be ss_customer_sk=1"
    );
    eprintln!("✓ INTERSECT value-correctness: exactly customer 1");

    // ADR-040 (TD-182): VALUE-correctness anchors for the feature-distinct shapes
    // the relational-engine test does NOT cover — ROLLUP, CUBE, and a rank() window.
    // Expected values hand-derived from the seed (store_sales ⋈ item ⋈ date_dim).
    // canon_rows sorts + 2-dp-normalizes, so comparison is order-independent and
    // free of DOUBLE-PRECISION accumulation noise; expecteds are sorted to match.
    let row = |c: &[&str]| c.iter().map(|s| s.to_string()).collect::<Vec<String>>();

    // ROLLUP(i_category, i_class) over ss_net_profit (whole-number profits ⇒ exact):
    // Electronics/classX = 12+8+6 = 26; Books/classY = 20+10 = 30; + subtotals + 56.
    let mut exp_rollup = vec![
        row(&["Electronics", "classX", "26"]),
        row(&["Books", "classY", "30"]),
        row(&["Electronics", "NULL", "26"]),
        row(&["Books", "NULL", "30"]),
        row(&["NULL", "NULL", "56"]),
    ];
    exp_rollup.sort();
    assert_eq!(
        canon_rows(&client, "select i_category, i_class, sum(ss_net_profit) as profit from store_sales, item where ss_item_sk = i_item_sk group by rollup(i_category, i_class) order by i_category, i_class").await,
        exp_rollup,
        "ROLLUP profit: per-class rows + category subtotals + grand total"
    );

    // CUBE(d_year, i_category) over ss_ext_sales_price: 2000/Elec=39.98+29.50=69.48,
    // 2000/Books=49.95+42.00=91.95, 1999/Elec=19.99; all cross-margins + grand 181.42.
    let mut exp_cube = vec![
        row(&["2000", "Electronics", "69.48"]),
        row(&["2000", "Books", "91.95"]),
        row(&["1999", "Electronics", "19.99"]),
        row(&["2000", "NULL", "161.43"]),
        row(&["1999", "NULL", "19.99"]),
        row(&["NULL", "Electronics", "89.47"]),
        row(&["NULL", "Books", "91.95"]),
        row(&["NULL", "NULL", "181.42"]),
    ];
    exp_cube.sort();
    assert_eq!(
        canon_rows(&client, "select d_year, i_category, sum(ss_ext_sales_price) as rev from store_sales, item, date_dim where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk group by cube(d_year, i_category) order by d_year, i_category").await,
        exp_cube,
        "CUBE revenue: all (year, category) margins + subtotals + grand total"
    );

    // rank() window: per-category brand revenue rank. Electronics brandA
    // (39.98+19.99=59.97) rank 1, brandB (29.50) rank 2; Books brandC (49.95) rank 1,
    // brandD (42.00) rank 2.
    let mut exp_rank = vec![
        row(&["Electronics", "brandA", "59.97", "1"]),
        row(&["Electronics", "brandB", "29.5", "2"]),
        row(&["Books", "brandC", "49.95", "1"]),
        row(&["Books", "brandD", "42", "2"]),
    ];
    exp_rank.sort();
    assert_eq!(
        canon_rows(&client, "select i_category, i_brand, sum(ss_ext_sales_price) as rev, rank() over (partition by i_category order by sum(ss_ext_sales_price) desc) as rnk from store_sales, item where ss_item_sk = i_item_sk group by i_category, i_brand order by i_category, rnk").await,
        exp_rank,
        "rank() window: per-category brand revenue ranking"
    );
    eprintln!("✓ value-correctness: ROLLUP / CUBE / rank() window anchored");

    // Window FRAME-UNIT correctness (RANGE vs GROUPS) on a key WITH TIES — a
    // frame that silently executed as ROWS would give wrong numbers here.
    // ORDER BY ss_quantity is [1,1,2,3,5] (the two q=1 rows carry profits 8 and
    // 6). BETWEEN 1 PRECEDING AND CURRENT ROW over sum(ss_net_profit):
    //   RANGE  — value window [q-1, q]:  q1→8+6=14, q2→8+6+12=26, q3→12+10=22, q5→20
    //   GROUPS — whole peer groups:      q1→14,      q2→26,        q3→22,       q5→10+20=30
    // The two units agree everywhere except q=5 (20 vs 30) — proof the declared
    // unit is honoured, not collapsed. (ROWS is order-sensitive on ties, so its
    // per-row values are non-deterministic here; the ROWS frame is anchored by
    // the deterministic `win_running` ratchet query above.)
    let mut exp_range = vec![
        row(&["1", "14"]),
        row(&["1", "14"]),
        row(&["2", "26"]),
        row(&["3", "22"]),
        row(&["5", "20"]),
    ];
    exp_range.sort();
    assert_eq!(
        canon_rows(&client, "select ss_quantity, sum(ss_net_profit) over (order by ss_quantity range between 1 preceding and current row) as s from store_sales").await,
        exp_range,
        "RANGE frame: value-window sums over a tied ORDER BY key"
    );

    let mut exp_groups = vec![
        row(&["1", "14"]),
        row(&["1", "14"]),
        row(&["2", "26"]),
        row(&["3", "22"]),
        row(&["5", "30"]),
    ];
    exp_groups.sort();
    assert_eq!(
        canon_rows(&client, "select ss_quantity, sum(ss_net_profit) over (order by ss_quantity groups between 1 preceding and current row) as s from store_sales").await,
        exp_groups,
        "GROUPS frame: peer-group sums over a tied ORDER BY key"
    );

    assert_ne!(
        exp_range, exp_groups,
        "RANGE and GROUPS must differ on ties — else the frame unit was collapsed to ROWS"
    );
    eprintln!("✓ value-correctness: RANGE vs GROUPS frame units honoured on ties");
    // Anchored accuracy phase (TD-182 P1) — same server, same client.
    tpcds_accuracy_anchored_phase(&client).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// ADR-040 accuracy ratchet (TD-182 P1): full-row anchors for the query shapes
// NOT already anchored in the conformance test above (ROLLUP / CUBE / rank /
// RANGE-GROUPS / INTERSECT live there). Hand-derived over the 5-row star seed:
//   SS1 2000-11-01 i1(Elec,brandA,11) cust1 ext 39.98 profit 12
//   SS2 2000-11-01 i2(Elec,brandB,12) cust2 ext 29.50 profit 8
//   SS3 2000-11-15 i3(Books,brandC,21) cust1 ext 49.95 profit 20
//   SS4 2000-12-10 i4(Books,brandD,22) cust3 ext 42.00 profit 10
//   SS5 1999-11-05 i1(Elec,brandA,11) cust1 ext 19.99 profit 6
// ─────────────────────────────────────────────────────────────────────────────
fn tpcds_expected() -> Vec<(&'static str, Vec<Vec<&'static str>>)> {
    vec![
        // q42: manufact 100 + moy 11 + year 2000 → SS1 only.
        ("q42", vec![vec!["2000", "1", "Electronics", "39.98"]]),
        // q52: moy 11 + year 2000 → SS1/SS2/SS3 (canon_rows sorts: brand_id ASC).
        (
            "q52",
            vec![
                vec!["2000", "11", "brandA", "39.98"],
                vec!["2000", "12", "brandB", "29.5"],
                vec!["2000", "21", "brandC", "49.95"],
            ],
        ),
        // q55: single-brand (manufact 100) revenue = SS1.
        ("q55", vec![vec!["11", "brandA", "39.98"]]),
        // q3: moy filter only (NO d_year) → SS1 (2000) AND SS5 (1999).
        (
            "q3",
            vec![
                vec!["1999", "11", "brandA", "19.99"],
                vec!["2000", "11", "brandA", "39.98"],
            ],
        ),
        // q98: per-item revenue + ratio to class total. classX = 39.98+29.50 =
        // 69.48 → item1 57.54 / item2 42.46; classY = 49.95+42.00 = 91.95 →
        // item3 54.32 / item4 45.68.
        (
            "q98",
            vec![
                vec![
                    "ITEM001",
                    "Electronics",
                    "classX",
                    "19.99",
                    "39.98",
                    "57.54",
                ],
                vec!["ITEM002", "Electronics", "classX", "29.5", "29.5", "42.46"],
                vec!["ITEM003", "Books", "classY", "9.99", "49.95", "54.32"],
                vec!["ITEM004", "Books", "classY", "14", "42", "45.68"],
            ],
        ),
        // win_running: per-day revenue + running total in date order. d_date
        // renders as a day-number (1999-11-05 = 10900 …) on this route — DATE
        // column-fidelity is a known rendering follow-up; the daily/running
        // VALUE columns are the assertion.
        (
            "win_running",
            vec![
                vec!["10900", "19.99", "19.99"],
                vec!["11262", "69.48", "89.47"],
                vec!["11276", "49.95", "139.42"],
                vec!["11301", "42", "181.42"],
            ],
        ),
        // cte KNOWN-BAD (TD-185a): `WITH` falls to the pgwire dispatch
        // fallthrough and silently returns ZERO rows. Correct answer:
        // Books 91.95, Electronics 89.47 (rev DESC). Pinned wrong; auto-trips
        // when the WITH-dispatch fix lands on this branch.
        ("cte-KNOWN-BAD", vec![]),
        // count_distinct: Electronics buyers {1,2}; Books buyers {1,3}.
        (
            "count_distinct",
            vec![vec!["Books", "2"], vec!["Electronics", "2"]],
        ),
        // correlated: items priced above their category average
        // (Elec avg 24.745 → item2 29.5; Books avg 11.995 → item4 14).
        ("correlated", vec![vec!["2", "29.5"], vec!["4", "14"]]),
        // case_agg: profit>10 hi / ≤10 lo — Electronics {12,8,6} → 1/2;
        // Books {20,10} → 1/1 (10 counts as lo).
        (
            "case_agg",
            vec![vec!["Books", "1", "1"], vec!["Electronics", "1", "2"]],
        ),
        // EXCEPT: Electronics buyers {1,2} minus Books buyers {1,3} → {2}.
        ("except", vec![vec!["2"]]),
    ]
}

/// ADR-040 accuracy ratchet (TD-182 P1): every anchored TPC-DS query must
/// produce its hand-derived full result set over the seed data. Complements
/// the ROLLUP/CUBE/rank/RANGE-GROUPS/INTERSECT anchors in the conformance
/// test above — a wrong-but-clean answer can no longer pass.
/// Anchored accuracy phase — CALLED from tpcds_pgwire_conformance_inner
/// so both phases share ONE server boot (the global WAL manifest is a
/// process-wide set-once; a second boot in the same process silently
/// reuses the first's manifest pointed at a deleted tempdir).
async fn tpcds_accuracy_anchored_phase(client: &tokio_postgres::Client) {
    for (name, ddl) in SCHEMA {
        let _ = client
            .simple_query(&format!("DROP TABLE IF EXISTS {name}"))
            .await;
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("CREATE {name}: {}", explain_err(&e)));
    }
    for sql in DATA {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("INSERT failed: {e}"));
    }
    for (name, _) in SCHEMA {
        client
            .simple_query(&format!("ALTER TABLE {name} MATERIALIZE"))
            .await
            .unwrap_or_else(|e| panic!("MATERIALIZE {name}: {}", explain_err(&e)));
    }

    let expected = tpcds_expected();
    let by_id: std::collections::HashMap<&str, String> = tpcds_queries().into_iter().collect();
    let mut accurate = 0;
    let mut failures = Vec::new();
    for (id, want) in &expected {
        let is_known_bad = id.ends_with("-KNOWN-BAD");
        let lookup_id = id.strip_suffix("-KNOWN-BAD").unwrap_or(id);
        let sql = by_id
            .get(lookup_id)
            .unwrap_or_else(|| panic!("{lookup_id} missing from tpcds_queries()"));
        let got = canon_rows(client, sql).await;
        let want: Vec<Vec<String>> = want
            .iter()
            .map(|row| row.iter().map(|c| c.to_string()).collect())
            .collect();
        if is_known_bad {
            if got == *want {
                // Still returns the pinned wrong answer — known-bad holds.
                eprintln!("  ✓ {id} still KNOWN-BAD (pinned wrong answer)");
                continue;
            }
            panic!(
                "{id}: KNOWN-BAD anchor tripped — the engine's answer CHANGED (got \\
                 {got:?}, pinned {want:?}). If the result is now the CORRECT \\
                 {{Books 91.95, Electronics 89.47}} set, update the anchor, move \\
                 {lookup_id} into the anchored set, and close TD-185a on this branch."
            );
        }
        if got == want {
            accurate += 1;
            eprintln!("  ✓ {id} accurate ({} rows)", got.len());
        } else {
            failures.push(format!("{id}: want {want:?}, got {got:?}"));
        }
    }

    eprintln!(
        "\n=== TPC-DS accuracy: {}/{} anchored queries accurate (ratchet {}) ===",
        accurate,
        expected.len(),
        TPCDS_ACCURATE_RATCHET
    );
    for f in &failures {
        eprintln!("  ✗ {f}");
    }
    assert!(
        failures.is_empty(),
        "TPC-DS anchored accuracy failures: {}",
        failures.join("; ")
    );
    assert!(
        accurate >= TPCDS_ACCURATE_RATCHET,
        "TPC-DS accuracy regressed: {accurate} < ratchet {TPCDS_ACCURATE_RATCHET}"
    );
}
