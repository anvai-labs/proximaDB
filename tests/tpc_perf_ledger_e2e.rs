//! TPC-H / TPC-DS performance evidence-ledger harness (TD-OLAP-4, P0b skeleton).
//!
//! Promotes the TPC suites from count-only *conformance* (see
//! `tests/{tpch,tpcds}_pgwire_e2e.rs`) toward *measured* evidence (ADR-052
//! invariant 4): per query × route × temperature it records wall latency and
//! the per-query `observability::io_trace` snapshot (bytes read, ranged GETs,
//! footer hits, per-engine compute_ms), on BOTH routes — native/Volcano
//! (pre-MATERIALIZE) and DataFusion-on-Parquet (post-MATERIALIZE) — into a
//! versioned JSON ledger artifact.
//!
//! Methodology (pinned per TD-OLAP-4; regimes are reported separately, never
//! averaged):
//! - storage regime: `local-tempdir` (file:// object store on local disk).
//!   Object-store regimes are a separate ledger section when they land.
//! - datagen: `synthetic-tpc-shaped` — deterministic, seeded, TPC-schema
//!   row-ratio-scaled with key skew, referential integrity, and fact tables
//!   CLUSTERED by their date column (uniform data defeats zone-map pruning —
//!   see TPC_PERF_GATE_EVIDENCE_2026_07_04). It is NOT audited dbgen/dsdgen
//!   output; no TPC-official claim may cite this ledger.
//! - temperature: `first` vs `repeat` against a warm process (true cold-cache
//!   runs need a server restart per query — a future slice).
//! - scale: `TPC_PERF_SCALE` (default 0.001 ≈ 9k TPC-H rows; SF1 ≡ 1.0).
//!   Runs whose estimated peak exceeds 8 GiB fail before data generation unless
//!   `TPC_PERF_ALLOW_LARGE_SCALE=I_ACCEPT_ISOLATED_OOM` is set and the process
//!   is inside a finite cgroup that leaves at least 16 GiB for the host. Use
//!   `scripts/run_tpc_perf_ledger.sh`; never launch SF-sized runs directly.
//!
//! `#[ignore]` — advisory, never a merge gate. Run on demand:
//!
//!   TPC_PERF_SCALE=0.001 cargo test --test tpc_perf_ledger_e2e -- --ignored --nocapture
//!
//! Output: `TPC_PERF_LEDGER_OUT` (default `target/tpc-perf-ledger/ledger.json`).
//!
//! Schemas and query texts are copied verbatim from the conformance suites
//! (`tests/tpch_pgwire_e2e.rs`, `tests/tpcds_pgwire_e2e.rs`) — the repo's
//! integration tests are self-contained by convention; keep them in sync.

use std::net::TcpListener;
#[cfg(not(feature = "duckdb"))]
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::observability::io_trace::{self, IoTraceSnapshot};
use tempfile::TempDir;
use tokio::time::sleep;
use tokio_postgres::{Client, SimpleQueryMessage};

/// One billing snapshot per query, pushed by the collecting observer at
/// io_trace scope close (pgwire wraps each statement in its own scope).
static CAPTURE: Mutex<Vec<IoTraceSnapshot>> = Mutex::new(Vec::new());

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
        // ADR-059: the catalog metadata_url MUST be per-tempdir — the default
        // ("file://./metadata") is a FIXED relative path that persists across
        // test runs, carrying stale table locations from previous tempdirs.
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
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

        // Override the process billing observer (installed by db.start()) with
        // a collector that captures each query's snapshot for this harness.
        io_trace::set_billing_observer(Some(Box::new(|snap: &IoTraceSnapshot, _tenant| {
            CAPTURE.lock().expect("capture lock").push(snap.clone());
        })));

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ if Instant::now() > deadline => anyhow::bail!("REST not ready"),
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

    async fn shutdown(mut self) {
        if let Some(mut db) = self.db.take() {
            let _ = db.shutdown().await;
        }
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

// ---------------------------------------------------------------------------
// Deterministic seeded generator — synthetic-tpc-shaped (NOT dbgen/dsdgen).
// SplitMix64 keeps every run reproducible; key skew is power-law-ish so join
// and filter selectivities are non-uniform like real TPC data.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    /// Power-law-skewed key in `1..=n` (quadratic bias toward low keys).
    fn skewed_key(&mut self, n: u64) -> u64 {
        let u = self.next() as f64 / u64::MAX as f64;
        ((u * u * n as f64) as u64).min(n.saturating_sub(1)) + 1
    }
    fn pick<'a>(&mut self, choices: &[&'a str]) -> &'a str {
        choices[self.below(choices.len() as u64) as usize]
    }
    fn date(&mut self) -> String {
        let y = 1992 + self.below(7);
        let m = 1 + self.below(12);
        let d = 1 + self.below(28);
        format!("DATE '{y}-{m:02}-{d:02}'")
    }
    fn money(&mut self, max: u64) -> String {
        format!("{}.{:02}", self.below(max), self.below(100))
    }
}

fn chunked_inserts(table: &str, cols: &str, values: Vec<String>, chunk: usize) -> Vec<String> {
    values
        .chunks(chunk)
        .map(|c| format!("INSERT INTO {table} ({cols}) VALUES {}", c.join(", ")))
        .collect()
}

/// Scale a canonical TPC row count, with a floor so every table joins.
fn scaled(base: u64, scale: f64, floor: u64) -> u64 {
    ((base as f64 * scale) as u64).max(floor)
}

const SAFE_ESTIMATED_PEAK_MIB: u64 = 8 * 1024;
const REQUIRED_HOST_HEADROOM_MIB: u64 = 16 * 1024;
const LARGE_SCALE_ACK: &str = "I_ACCEPT_ISOLATED_OOM";
// Calibrated conservatively from the 2026-09-15 SF1 incident. The generator,
// pgwire load path, native tables, and materialized parquet can coexist during
// a run, so source-row text size alone substantially understates peak RSS.
const ESTIMATED_BYTES_PER_SOURCE_ROW: f64 = 16_384.0;
const TPCH_SF1_ROWS: u64 = 8_660_030;
const TPCDS_SF1_ROWS: u64 = 5_722_000;

#[derive(Debug, PartialEq)]
struct PerfRunPreflight {
    estimated_peak_mib: u64,
    cgroup_limit_mib: Option<u64>,
    host_total_mib: Option<u64>,
}

fn estimated_peak_mib(scale: f64, benchmark_filter: Option<&str>) -> u64 {
    let base_rows = match benchmark_filter {
        Some("tpch") => TPCH_SF1_ROWS,
        Some("tpcds") => TPCDS_SF1_ROWS,
        // The suites run sequentially, so the larger suite determines peak.
        _ => TPCH_SF1_ROWS.max(TPCDS_SF1_ROWS),
    };
    ((base_rows as f64 * scale * ESTIMATED_BYTES_PER_SOURCE_ROW) / (1024.0 * 1024.0)).ceil() as u64
}

fn validate_perf_run(
    scale: f64,
    benchmark_filter: Option<&str>,
    acknowledgement: Option<&str>,
    cgroup_limit_mib: Option<u64>,
    host_total_mib: Option<u64>,
) -> Result<PerfRunPreflight, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("TPC_PERF_SCALE must be a finite number greater than zero".to_string());
    }

    let estimated_peak_mib = estimated_peak_mib(scale, benchmark_filter);
    let preflight = PerfRunPreflight {
        estimated_peak_mib,
        cgroup_limit_mib,
        host_total_mib,
    };
    if estimated_peak_mib <= SAFE_ESTIMATED_PEAK_MIB {
        return Ok(preflight);
    }
    if acknowledgement != Some(LARGE_SCALE_ACK) {
        return Err(format!(
            "estimated peak {estimated_peak_mib} MiB exceeds the unisolated {SAFE_ESTIMATED_PEAK_MIB} MiB ceiling; run scripts/run_tpc_perf_ledger.sh and set TPC_PERF_ALLOW_LARGE_SCALE={LARGE_SCALE_ACK}"
        ));
    }

    let limit = cgroup_limit_mib.ok_or_else(|| {
        "large-scale TPC runs require a finite cgroup memory.max; use scripts/run_tpc_perf_ledger.sh"
            .to_string()
    })?;
    let host_total = host_total_mib.ok_or_else(|| {
        "large-scale TPC runs require readable host memory capacity for the headroom check"
            .to_string()
    })?;
    if limit >= host_total || host_total - limit < REQUIRED_HOST_HEADROOM_MIB {
        return Err(format!(
            "cgroup limit {limit} MiB leaves less than {REQUIRED_HOST_HEADROOM_MIB} MiB host headroom (host {host_total} MiB)"
        ));
    }
    Ok(preflight)
}

fn current_cgroup_memory_limit_mib() -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim_start_matches('/');
    let path = std::path::Path::new("/sys/fs/cgroup")
        .join(relative)
        .join("memory.max");
    let raw = std::fs::read_to_string(path).ok()?;
    let bytes = raw.trim().parse::<u64>().ok()?;
    Some(bytes / (1024 * 1024))
}

fn host_total_memory_mib() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo.lines().find_map(|line| {
        line.strip_prefix("MemTotal:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
    })?;
    Some(kib / 1024)
}

// --- TPC-H (schema copied from tests/tpch_pgwire_e2e.rs) -------------------

const TPCH_SCHEMA: &[(&str, &str)] = &[
    (
        "region",
        "CREATE TABLE region (r_regionkey INT PRIMARY KEY, r_name VARCHAR, r_comment VARCHAR)",
    ),
    (
        "nation",
        "CREATE TABLE nation (n_nationkey INT PRIMARY KEY, n_name VARCHAR, n_regionkey INT, n_comment VARCHAR)",
    ),
    (
        "supplier",
        "CREATE TABLE supplier (s_suppkey INT PRIMARY KEY, s_name VARCHAR, s_address VARCHAR, s_nationkey INT, s_phone VARCHAR, s_acctbal DOUBLE PRECISION, s_comment VARCHAR)",
    ),
    (
        "part",
        "CREATE TABLE part (p_partkey INT PRIMARY KEY, p_name VARCHAR, p_mfgr VARCHAR, p_brand VARCHAR, p_type VARCHAR, p_size INT, p_container VARCHAR, p_retailprice DOUBLE PRECISION, p_comment VARCHAR)",
    ),
    (
        "partsupp",
        "CREATE TABLE partsupp (ps_partkey INT, ps_suppkey INT, ps_availqty INT, ps_supplycost DOUBLE PRECISION, ps_comment VARCHAR)",
    ),
    (
        "customer",
        "CREATE TABLE customer (c_custkey INT PRIMARY KEY, c_name VARCHAR, c_address VARCHAR, c_nationkey INT, c_phone VARCHAR, c_acctbal DOUBLE PRECISION, c_mktsegment VARCHAR, c_comment VARCHAR)",
    ),
    (
        "orders",
        "CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_orderstatus VARCHAR, o_totalprice DOUBLE PRECISION, o_orderdate DATE, o_orderpriority VARCHAR, o_clerk VARCHAR, o_shippriority INT, o_comment VARCHAR)",
    ),
    (
        "lineitem",
        "CREATE TABLE lineitem (l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DOUBLE PRECISION, l_extendedprice DOUBLE PRECISION, l_discount DOUBLE PRECISION, l_tax DOUBLE PRECISION, l_returnflag VARCHAR, l_linestatus VARCHAR, l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR, l_shipmode VARCHAR, l_comment VARCHAR)",
    ),
];

/// 5 regions / 25 nations mapped like the standard, including every nation
/// name the 22 queries reference by constant.
const REGIONS: &[&str] = &["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"];
const NATIONS: &[(&str, u64)] = &[
    ("ALGERIA", 0),
    ("ARGENTINA", 1),
    ("BRAZIL", 1),
    ("CANADA", 1),
    ("EGYPT", 4),
    ("ETHIOPIA", 0),
    ("FRANCE", 3),
    ("GERMANY", 3),
    ("INDIA", 2),
    ("INDONESIA", 2),
    ("IRAN", 4),
    ("IRAQ", 4),
    ("JAPAN", 2),
    ("JORDAN", 4),
    ("KENYA", 0),
    ("MOROCCO", 0),
    ("MOZAMBIQUE", 0),
    ("PERU", 1),
    ("CHINA", 2),
    ("ROMANIA", 3),
    ("SAUDI ARABIA", 4),
    ("VIETNAM", 2),
    ("RUSSIA", 3),
    ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
];

fn gen_tpch(scale: f64) -> Vec<String> {
    let mut rng = Rng::new(0x07C0_FFEE_5EED);
    let n_supp = scaled(10_000, scale, 10);
    let n_part = scaled(200_000, scale, 20);
    let n_cust = scaled(150_000, scale, 15);
    let n_ord = scaled(1_500_000, scale, 150);

    let types = [
        "PROMO BRUSHED STEEL",
        "STANDARD POLISHED TIN",
        "SMALL PLATED COPPER",
        "MEDIUM POLISHED BRASS",
        "ECONOMY ANODIZED NICKEL",
    ];
    let containers = [
        "SM CASE", "SM BOX", "MED BOX", "MED BAG", "LG BOX", "WRAP PKG",
    ];
    let segments = [
        "BUILDING",
        "AUTOMOBILE",
        "MACHINERY",
        "HOUSEHOLD",
        "FURNITURE",
    ];
    let priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"];
    let shipmodes = ["MAIL", "SHIP", "AIR", "AIR REG", "TRUCK", "RAIL", "FOB"];
    let instructs = [
        "DELIVER IN PERSON",
        "NONE",
        "TAKE BACK RETURN",
        "COLLECT COD",
    ];
    let name_words = ["forest", "green", "sky", "metal", "ivory", "rose", "navy"];
    let phone_codes = ["13", "31", "23", "29", "30", "18", "17", "27", "10"];

    let mut out = Vec::new();
    out.extend(chunked_inserts(
        "region",
        "r_regionkey, r_name, r_comment",
        REGIONS
            .iter()
            .enumerate()
            .map(|(i, r)| format!("({i}, '{r}', 'rc{i}')"))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "nation",
        "n_nationkey, n_name, n_regionkey, n_comment",
        NATIONS
            .iter()
            .enumerate()
            .map(|(i, (n, r))| format!("({i}, '{n}', {r}, 'nc{i}')"))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "supplier",
        "s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment",
        (1..=n_supp)
            .map(|k| {
                let nat = rng.below(25);
                let comment = if rng.below(20) == 0 {
                    "Customer Complaints"
                } else {
                    "sc"
                };
                format!(
                    "({k}, 'Supplier#{k}', 'addr{k}', {nat}, '{}-{k:03}', {}, '{comment}')",
                    rng.pick(&phone_codes),
                    rng.money(10_000)
                )
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "part",
        "p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment",
        (1..=n_part)
            .map(|k| {
                format!(
                    "({k}, '{} part {k}', 'Mfgr#{}', 'Brand#{}{}', '{}', {}, '{}', {}, 'pc{k}')",
                    rng.pick(&name_words),
                    1 + rng.below(5),
                    1 + rng.below(5),
                    1 + rng.below(5),
                    rng.pick(&types),
                    1 + rng.below(50),
                    rng.pick(&containers),
                    rng.money(2_000)
                )
            })
            .collect(),
        200,
    ));
    // 4 suppliers per part, like the standard 800k/200k ratio.
    out.extend(chunked_inserts(
        "partsupp",
        "ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment",
        (1..=n_part)
            .flat_map(|p| {
                (0..4)
                    .map(|_| {
                        format!(
                            "({p}, {}, {}, {}, 'psc')",
                            rng.skewed_key(n_supp),
                            1 + rng.below(9_999),
                            rng.money(1_000)
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "customer",
        "c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment",
        (1..=n_cust)
            .map(|k| {
                let comment = if rng.below(15) == 0 {
                    "special requests noted"
                } else {
                    "cc"
                };
                format!(
                    "({k}, 'Customer#{k}', 'caddr{k}', {}, '{}-{k:03}', {}, '{}', '{comment}')",
                    rng.below(25),
                    rng.pick(&phone_codes),
                    rng.money(10_000),
                    rng.pick(&segments)
                )
            })
            .collect(),
        200,
    ));
    // Fact tables are CLUSTERED by their date column before insertion.
    // Uniform per-row dates give every row group whole-domain min/max, so
    // zone-map pruning provably cannot skip (measured 0% in
    // docs/_internal/status/TPC_PERF_GATE_EVIDENCE_2026_07_04.adoc). Real TPC
    // data is time-ordered; sorting restores that property so row groups
    // carry tight bounds. The `DATE 'YYYY-MM-DD'` literal format sorts
    // lexicographically.
    let mut lineitems: Vec<(String, String)> = Vec::new();
    let mut orders: Vec<(String, String)> = (1..=n_ord)
        .map(|o| {
            let lines = 1 + rng.below(7);
            for l in 1..=lines {
                let ship = rng.date();
                let row = format!(
                    "({o}, {}, {}, {l}, {}, {}, 0.0{}, 0.0{}, '{}', '{}', {ship}, {}, {}, '{}', '{}', 'lc')",
                    rng.skewed_key(n_part),
                    rng.skewed_key(n_supp),
                    1 + rng.below(50),
                    rng.money(10_000),
                    rng.below(11),
                    rng.below(9),
                    rng.pick(&["N", "R", "A"]),
                    rng.pick(&["O", "F"]),
                    rng.date(),
                    rng.date(),
                    rng.pick(&instructs),
                    rng.pick(&shipmodes),
                );
                lineitems.push((ship, row));
            }
            let odate = rng.date();
            let row = format!(
                "({o}, {}, '{}', {}, {odate}, '{}', 'Clerk#{}', 0, 'oc')",
                rng.skewed_key(n_cust),
                rng.pick(&["O", "F", "P"]),
                rng.money(100_000),
                rng.pick(&priorities),
                1 + rng.below(100)
            );
            (odate, row)
        })
        .collect();
    orders.sort_by(|a, b| a.0.cmp(&b.0));
    lineitems.sort_by(|a, b| a.0.cmp(&b.0));
    out.extend(chunked_inserts(
        "orders",
        "o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment",
        orders.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "lineitem",
        "l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment",
        lineitems.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    out
}

// --- TPC-DS star subset (schema copied from tests/tpcds_pgwire_e2e.rs) -----

const TPCDS_SCHEMA: &[(&str, &str)] = &[
    (
        "date_dim",
        "CREATE TABLE date_dim (d_date_sk INT PRIMARY KEY, d_date DATE, d_year INT, d_moy INT, d_qoy INT, d_dom INT, d_dow INT, d_month_seq INT, d_week_seq INT, d_day_name VARCHAR, d_quarter_name VARCHAR)",
    ),
    (
        "item",
        "CREATE TABLE item (i_item_sk INT PRIMARY KEY, i_item_id VARCHAR, i_item_desc VARCHAR, i_brand_id INT, i_brand VARCHAR, i_class_id INT, i_class VARCHAR, i_category_id INT, i_category VARCHAR, i_manufact_id INT, i_manufact VARCHAR, i_current_price DOUBLE PRECISION, i_wholesale_cost DOUBLE PRECISION, i_product_name VARCHAR, i_color VARCHAR, i_units VARCHAR, i_size VARCHAR, i_manager_id INT)",
    ),
    (
        "store",
        "CREATE TABLE store (s_store_sk INT PRIMARY KEY, s_store_id VARCHAR, s_store_name VARCHAR, s_state VARCHAR, s_county VARCHAR, s_city VARCHAR, s_zip VARCHAR, s_gmt_offset DOUBLE PRECISION, s_market_id INT, s_number_employees INT, s_company_name VARCHAR, s_company_id INT)",
    ),
    (
        "customer",
        "CREATE TABLE customer (c_customer_sk INT PRIMARY KEY, c_customer_id VARCHAR, c_first_name VARCHAR, c_last_name VARCHAR, c_current_addr_sk INT, c_birth_country VARCHAR, c_birth_year INT, c_birth_month INT, c_birth_day INT, c_preferred_cust_flag VARCHAR, c_current_cdemo_sk INT, c_current_hdemo_sk INT, c_salutation VARCHAR, c_login VARCHAR, c_email_address VARCHAR)",
    ),
    (
        "customer_address",
        "CREATE TABLE customer_address (ca_address_sk INT PRIMARY KEY, ca_state VARCHAR, ca_city VARCHAR, ca_zip VARCHAR, ca_country VARCHAR, ca_gmt_offset DOUBLE PRECISION, ca_county VARCHAR)",
    ),
    (
        "customer_demographics",
        "CREATE TABLE customer_demographics (cd_demo_sk INT PRIMARY KEY, cd_gender VARCHAR, cd_marital_status VARCHAR, cd_education_status VARCHAR, cd_purchase_estimate INT, cd_credit_rating VARCHAR, cd_dep_count INT, cd_dep_employed_count INT, cd_dep_college_count INT)",
    ),
    (
        "household_demographics",
        "CREATE TABLE household_demographics (hd_demo_sk INT PRIMARY KEY, hd_income_band_sk INT, hd_buy_potential VARCHAR, hd_dep_count INT, hd_vehicle_count INT)",
    ),
    (
        "income_band",
        "CREATE TABLE income_band (ib_income_band_sk INT PRIMARY KEY, ib_lower_bound INT, ib_upper_bound INT)",
    ),
    (
        "promotion",
        "CREATE TABLE promotion (p_promo_sk INT PRIMARY KEY, p_channel_dmail VARCHAR, p_channel_email VARCHAR, p_channel_event VARCHAR, p_channel_tv VARCHAR)",
    ),
    (
        "warehouse",
        "CREATE TABLE warehouse (w_warehouse_sk INT PRIMARY KEY, w_warehouse_name VARCHAR, w_state VARCHAR, w_warehouse_sq_ft INT, w_city VARCHAR, w_county VARCHAR, w_country VARCHAR)",
    ),
    (
        "reason",
        "CREATE TABLE reason (r_reason_sk INT PRIMARY KEY, r_reason_desc VARCHAR)",
    ),
    (
        "time_dim",
        "CREATE TABLE time_dim (t_time_sk INT PRIMARY KEY, t_time INT, t_hour INT, t_minute INT, t_meal_time VARCHAR)",
    ),
    (
        "ship_mode",
        "CREATE TABLE ship_mode (sm_ship_mode_sk INT PRIMARY KEY, sm_type VARCHAR, sm_carrier VARCHAR)",
    ),
    (
        "web_site",
        "CREATE TABLE web_site (web_site_sk INT PRIMARY KEY, web_site_id VARCHAR, web_name VARCHAR, web_company_name VARCHAR)",
    ),
    (
        "call_center",
        "CREATE TABLE call_center (cc_call_center_sk INT PRIMARY KEY, cc_call_center_id VARCHAR, cc_name VARCHAR, cc_manager VARCHAR, cc_county VARCHAR)",
    ),
    (
        "store_sales",
        // TD-OLAP-6: star-schema fact tables key on an INT date surrogate, so
        // the first-DATE-column heuristic finds nothing — declare the cluster
        // key explicitly so sort-on-materialize gives row groups tight
        // ss_sold_date_sk windows (the runtime join filter's prune target).
        "CREATE TABLE store_sales (ss_sold_date_sk INT, ss_sold_time_sk INT, ss_item_sk INT, ss_store_sk INT, ss_customer_sk INT, ss_addr_sk INT, ss_cdemo_sk INT, ss_hdemo_sk INT, ss_promo_sk INT, ss_ticket_number INT, ss_quantity INT, ss_wholesale_cost DOUBLE PRECISION, ss_list_price DOUBLE PRECISION, ss_sales_price DOUBLE PRECISION, ss_coupon_amt DOUBLE PRECISION, ss_ext_sales_price DOUBLE PRECISION, ss_ext_wholesale_cost DOUBLE PRECISION, ss_ext_list_price DOUBLE PRECISION, ss_ext_discount_amt DOUBLE PRECISION, ss_ext_tax DOUBLE PRECISION, ss_net_paid DOUBLE PRECISION, ss_net_paid_inc_tax DOUBLE PRECISION, ss_net_profit DOUBLE PRECISION) WITH (cluster_key = 'ss_sold_date_sk')",
    ),
    (
        "store_returns",
        "CREATE TABLE store_returns (sr_returned_date_sk INT, sr_item_sk INT, sr_customer_sk INT, sr_store_sk INT, sr_ticket_number INT, sr_return_quantity INT, sr_return_amt DOUBLE PRECISION, sr_net_loss DOUBLE PRECISION, sr_reason_sk INT)",
    ),
    (
        "catalog_sales",
        "CREATE TABLE catalog_sales (cs_sold_date_sk INT, cs_sold_time_sk INT, cs_ship_date_sk INT, cs_item_sk INT, cs_call_center_sk INT, cs_warehouse_sk INT, cs_catalog_page_sk INT, cs_bill_customer_sk INT, cs_ship_customer_sk INT, cs_bill_addr_sk INT, cs_ship_addr_sk INT, cs_bill_cdemo_sk INT, cs_bill_hdemo_sk INT, cs_promo_sk INT, cs_order_number INT, cs_quantity INT, cs_wholesale_cost DOUBLE PRECISION, cs_list_price DOUBLE PRECISION, cs_sales_price DOUBLE PRECISION, cs_coupon_amt DOUBLE PRECISION, cs_ext_sales_price DOUBLE PRECISION, cs_ext_wholesale_cost DOUBLE PRECISION, cs_ext_list_price DOUBLE PRECISION, cs_ext_discount_amt DOUBLE PRECISION, cs_ext_ship_cost DOUBLE PRECISION, cs_net_paid DOUBLE PRECISION, cs_net_paid_inc_tax DOUBLE PRECISION, cs_net_profit DOUBLE PRECISION, cs_ship_mode_sk INT)",
    ),
    (
        "catalog_returns",
        "CREATE TABLE catalog_returns (cr_returned_date_sk INT, cr_item_sk INT, cr_order_number INT, cr_call_center_sk INT, cr_returning_customer_sk INT, cr_returning_addr_sk INT, cr_return_quantity INT, cr_return_amount DOUBLE PRECISION, cr_refunded_cash DOUBLE PRECISION, cr_reversed_charge DOUBLE PRECISION, cr_store_credit DOUBLE PRECISION, cr_net_loss DOUBLE PRECISION, cr_return_amt_inc_tax DOUBLE PRECISION)",
    ),
    (
        "web_sales",
        "CREATE TABLE web_sales (ws_sold_date_sk INT, ws_sold_time_sk INT, ws_ship_date_sk INT, ws_item_sk INT, ws_web_site_sk INT, ws_web_page_sk INT, ws_warehouse_sk INT, ws_bill_customer_sk INT, ws_ship_customer_sk INT, ws_bill_addr_sk INT, ws_ship_addr_sk INT, ws_bill_cdemo_sk INT, ws_ship_hdemo_sk INT, ws_order_number INT, ws_quantity INT, ws_wholesale_cost DOUBLE PRECISION, ws_list_price DOUBLE PRECISION, ws_sales_price DOUBLE PRECISION, ws_coupon_amt DOUBLE PRECISION, ws_ext_sales_price DOUBLE PRECISION, ws_ext_wholesale_cost DOUBLE PRECISION, ws_ext_list_price DOUBLE PRECISION, ws_ext_discount_amt DOUBLE PRECISION, ws_ext_ship_cost DOUBLE PRECISION, ws_net_paid DOUBLE PRECISION, ws_net_paid_inc_tax DOUBLE PRECISION, ws_net_profit DOUBLE PRECISION, ws_promo_sk INT, ws_ship_mode_sk INT)",
    ),
    (
        "web_returns",
        "CREATE TABLE web_returns (wr_returned_date_sk INT, wr_item_sk INT, ws_order_number INT, wr_order_number INT, wr_web_page_sk INT, wr_returning_customer_sk INT, wr_returning_addr_sk INT, wr_refunded_cdemo_sk INT, wr_returning_cdemo_sk INT, wr_refunded_addr_sk INT, wr_reason_sk INT, wr_return_quantity INT, wr_return_amt DOUBLE PRECISION, wr_refunded_cash DOUBLE PRECISION, cr_reversed_charge DOUBLE PRECISION, wr_fee DOUBLE PRECISION, wr_net_loss DOUBLE PRECISION)",
    ),
    (
        "inventory",
        "CREATE TABLE inventory (inv_date_sk INT, inv_item_sk INT, inv_warehouse_sk INT, inv_quantity_on_hand INT)",
    ),
];

fn gen_tpcds(scale: f64) -> Vec<String> {
    let mut rng = Rng::new(0x0D50_FFEE_5EED);
    let n_item = scaled(18_000, scale, 20);
    let n_cust = scaled(100_000, scale, 10);
    let n_sales = scaled(2_880_000, scale, 300);
    let n_store = 12u64;
    let n_warehouse = 5u64;
    let n_reason = 35u64;
    let n_call_center = 6u64;
    // Three years of dates, 1998-2000 (queries filter d_year=1998..2002).
    let mut dates = Vec::new();
    let mut sk = 23_000u64;
    let day_names = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    for y in [1998u64, 1999, 2000] {
        for m in 1..=12u64 {
            for d in 1..=28u64 {
                sk += 1;
                let dow = (sk + 5) % 7; // Monday=0 at sk=24500ish
                let dn = day_names[(dow) as usize % 7];
                let qoy = m.div_ceil(3);
                let week_seq = (y - 1998) * 52 + m * 4 + d / 7;
                let month_seq = (y - 1998) * 12 + m;
                dates.push(format!(
                    "({sk}, DATE '{y}-{m:02}-{d:02}', {y}, {m}, {qoy}, {d}, {dow}, {month_seq}, {week_seq}, '{dn}', '{y}Q{qoy}')"
                ));
            }
        }
    }
    let n_dates = dates.len() as u64;
    let first_sk = 23_001u64;

    let categories = [
        ("Electronics", 1u64),
        ("Books", 2),
        ("Home", 3),
        ("Sports", 4),
        ("Music", 5),
    ];
    let states = [
        "CA", "NY", "TX", "WA", "IL", "GA", "TN", "FL", "OH", "MI", "VA", "OR",
    ];
    let counties = [
        "Williamson County",
        "Franklin Parish",
        "Bronx County",
        "Orange County",
    ];

    let mut out = Vec::new();
    out.extend(chunked_inserts(
        "date_dim",
        "d_date_sk, d_date, d_year, d_moy, d_qoy, d_dom, d_dow, d_month_seq, d_week_seq, d_day_name, d_quarter_name",
        dates,
        200,
    ));
    out.extend(chunked_inserts(
        "item",
        "i_item_sk, i_item_id, i_item_desc, i_brand_id, i_brand, i_class_id, i_class, i_category_id, i_category, i_manufact_id, i_manufact, i_current_price, i_wholesale_cost, i_product_name, i_color, i_units, i_size, i_manager_id",
        (1..=n_item)
            .map(|k| {
                let (cat, cat_id) = categories[rng.below(5) as usize];
                let brand = 10 * cat_id + 1 + rng.below(5);
                let manufact_id = 100 + rng.below(10);
                let price = rng.money(100);
                let wcost = rng.money(80);
                format!(
                    "({k}, 'ITEM{k:05}', 'desc{k}', {brand}, 'brand{brand}', {cat_id}, 'class{}', {cat_id}, '{cat}', {manufact_id}, 'mfg{manufact_id}', {price}, {wcost}, 'Product{k}', '{}', '{}', '{}', {})",
                    1 + rng.below(3),
                    rng.pick(&["powder", "khaki", "brown", "slate", "floral", "spring"]),
                    rng.pick(&["Ounce", "Box", "Pound", "Each", "Dozen"]),
                    rng.pick(&["medium", "small", "large", "petite"]),
                    1 + rng.below(10),
                )
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "store",
        "s_store_sk, s_store_id, s_store_name, s_state, s_county, s_city, s_zip, s_gmt_offset, s_market_id, s_number_employees, s_company_name, s_company_id",
        (1..=n_store)
            .map(|k| {
                let st = states[(k as usize - 1) % states.len()];
                format!(
                    "({k}, 'STORE{k:02}', 'Store {k}', '{st}', '{}', 'Fairview', '{:05}', -{}.0, {}, {}, 'Company{}', {})",
                    counties[(k as usize - 1) % counties.len()],
                    10_000 + k * 137,
                    5 + (k % 3),
                    1 + (k % 5),
                    200 + (k as i64 * 13) % 100,
                    k,
                    k,
                )
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "customer_address",
        "ca_address_sk, ca_state, ca_city, ca_zip, ca_country, ca_gmt_offset, ca_county",
        (1..=n_cust)
            .map(|k| {
                format!(
                    "({k}, '{}', 'City{}', '{:05}', 'United States', -{}.0, '{}')",
                    rng.pick(&states),
                    k,
                    10_000 + rng.below(89_999),
                    5 + rng.below(4),
                    counties[rng.below(counties.len() as u64) as usize],
                )
            })
            .collect(),
        200,
    ));
    // Demographics: small dimension tables with deterministic small row counts.
    let n_cdemo = 100u64;
    out.extend(chunked_inserts(
        "customer_demographics",
        "cd_demo_sk, cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating, cd_dep_count, cd_dep_employed_count, cd_dep_college_count",
        (1..=n_cdemo)
            .map(|k| {
                format!(
                    "({k}, '{}', '{}', '{}', {}, '{}', {}, {}, {})",
                    rng.pick(&["M", "F"]),
                    rng.pick(&["M", "S", "W", "D"]),
                    rng.pick(&["College", "Advanced Degree", "2 yr Degree", "4 yr Degree", "Unknown"]),
                    1000 + rng.below(9000),
                    rng.pick(&["Good", "Bad", "High Risk"]),
                    rng.below(10),
                    rng.below(5),
                    rng.below(5),
                )
            })
            .collect(),
        200,
    ));
    let n_hdemo = 100u64;
    let n_income_band = 20u64;
    out.extend(chunked_inserts(
        "household_demographics",
        "hd_demo_sk, hd_income_band_sk, hd_buy_potential, hd_dep_count, hd_vehicle_count",
        (1..=n_hdemo)
            .map(|k| {
                format!(
                    "({k}, {}, '{}', {}, {})",
                    1 + rng.below(n_income_band),
                    rng.pick(&[">10000", "5000-10000", "Unknown", "0-1000"]),
                    rng.below(10),
                    rng.below(5),
                )
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "income_band",
        "ib_income_band_sk, ib_lower_bound, ib_upper_bound",
        (1..=n_income_band)
            .map(|k| format!("({k}, {}, {})", k * 10000, k * 10000 + 10000))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "promotion",
        "p_promo_sk, p_channel_dmail, p_channel_email, p_channel_event, p_channel_tv",
        (1..=50u64)
            .map(|k| {
                format!(
                    "({k}, '{}', '{}', '{}', '{}')",
                    rng.pick(&["Y", "N"]),
                    rng.pick(&["Y", "N"]),
                    rng.pick(&["Y", "N"]),
                    rng.pick(&["Y", "N"]),
                )
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "warehouse",
        "w_warehouse_sk, w_warehouse_name, w_state, w_warehouse_sq_ft, w_city, w_county, w_country",
        (1..=n_warehouse)
            .map(|k| {
                format!(
                    "({k}, 'Warehouse{}', '{}', {}, 'City{}', 'County{}', 'United States')",
                    k,
                    rng.pick(&states),
                    100_000 + k * 5000,
                    k,
                    k,
                )
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "reason",
        "r_reason_sk, r_reason_desc",
        (1..=n_reason)
            .map(|k| format!("({k}, 'reason {}')", k))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "time_dim",
        "t_time_sk, t_time, t_hour, t_minute, t_meal_time",
        (0..86_400u64)
            .step_by(60)
            .map(|secs| {
                let hour = secs / 3600;
                let minute = (secs % 3600) / 60;
                let meal = if (8..10).contains(&hour) {
                    "breakfast"
                } else if (12..14).contains(&hour) {
                    "lunch"
                } else if (18..21).contains(&hour) {
                    "dinner"
                } else {
                    "night"
                };
                format!("({secs}, {secs}, {hour}, {minute}, '{meal}')")
            })
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "ship_mode",
        "sm_ship_mode_sk, sm_type, sm_carrier",
        (1..=5u64)
            .map(|k| format!("({k}, 'type{k}', 'CARRIER{k}')"))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "web_site",
        "web_site_sk, web_site_id, web_name, web_company_name",
        (1..=6u64)
            .map(|k| format!("({k}, 'WS{k:02}', 'WebSite{k}', 'pri')"))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "call_center",
        "cc_call_center_sk, cc_call_center_id, cc_name, cc_manager, cc_county",
        (1..=n_call_center)
            .map(|k| format!("({k}, 'CC{k:02}', 'CallCenter{k}', 'Manager{k}', 'County{k}')"))
            .collect(),
        200,
    ));
    out.extend(chunked_inserts(
        "customer",
        "c_customer_sk, c_customer_id, c_first_name, c_last_name, c_current_addr_sk, c_birth_country, c_birth_year, c_birth_month, c_birth_day, c_preferred_cust_flag, c_current_cdemo_sk, c_current_hdemo_sk, c_salutation, c_login, c_email_address",
        (1..=n_cust)
            .map(|k| {
                format!(
                    "({k}, 'CUST{k:05}', 'fn{k}', 'ln{k}', {}, '{}', {}, {}, {}, '{}', {}, {}, '{}', 'login{k}', 'email{k}')",
                    1 + rng.below(n_cust),
                    rng.pick(&["CANADA", "MEXICO", "BRAZIL", "JAPAN"]),
                    1950 + rng.below(50),
                    1 + rng.below(12),
                    1 + rng.below(28),
                    rng.pick(&["Y", "N"]),
                    1 + rng.below(n_cdemo),
                    1 + rng.below(n_hdemo),
                    rng.pick(&["Mr.", "Ms.", "Dr."]),
                )
            })
            .collect(),
        200,
    ));
    // store_sales — clustered by ss_sold_date_sk. The generator produces a
    // fully-populated fact row so every TPC-DS query has its columns.
    let mut sales: Vec<(u64, String)> = (1..=n_sales)
        .map(|t| {
            let qty = 1 + rng.below(100);
            let price = 1 + rng.below(100);
            let wcost = rng.money(80);
            let list_price = price + 10; // list > sales
            let ext_sales = qty * price;
            let ext_wcost = qty * (wcost.parse::<f64>().unwrap_or(0.0) as u64 + 1);
            let ext_list = qty * list_price;
            let coupon = rng.below(20);
            let discount = rng.below(10);
            let ext_disc = ext_sales * discount / 100;
            let tax = ext_sales * 8 / 100;
            let net_paid = ext_sales - ext_disc;
            let net_profit = net_paid as i64 - ext_wcost as i64;
            let date_sk = first_sk + rng.below(n_dates);
            let row = format!(
                "({date_sk}, {}, {}, {}, {}, {}, {}, {}, {}, {t}, {qty}, {wcost}, {list_price}, {price}.00, {coupon}.00, {ext_sales}.00, {ext_wcost}.00, {ext_list}.00, {ext_disc}.00, {tax}.00, {net_paid}.00, {}, {net_profit})",
                rng.skewed_key(n_item),
                1 + rng.below(n_store),
                rng.skewed_key(n_cust),
                1 + rng.below(n_cust),
                1 + rng.below(n_cdemo),
                1 + rng.below(n_hdemo),
                1 + rng.below(50),
                rng.below(86400),
                net_paid + tax,
            );
            (date_sk, row)
        })
        .collect();
    sales.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "store_sales",
        "ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_store_sk, ss_customer_sk, ss_addr_sk, ss_cdemo_sk, ss_hdemo_sk, ss_promo_sk, ss_ticket_number, ss_quantity, ss_wholesale_cost, ss_list_price, ss_sales_price, ss_coupon_amt, ss_ext_sales_price, ss_ext_wholesale_cost, ss_ext_list_price, ss_ext_discount_amt, ss_ext_tax, ss_net_paid, ss_net_paid_inc_tax, ss_net_profit",
        sales.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    // store_returns — a fraction of store_sales get returned.
    let n_sr = scaled(288_000, scale, 30);
    let mut sreturns: Vec<(u64, String)> = (1..=n_sr)
        .map(|t| {
            let date_sk = first_sk + rng.below(n_dates);
            let row = format!(
                "({date_sk}, {}, {}, {}, {t}, {}, {}.00, {}.00, {})",
                rng.skewed_key(n_item),
                rng.skewed_key(n_cust),
                1 + rng.below(n_store),
                1 + rng.below(5),
                rng.below(100),
                rng.below(50),
                1 + rng.below(35),
            );
            (date_sk, row)
        })
        .collect();
    sreturns.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "store_returns",
        "sr_returned_date_sk, sr_item_sk, sr_customer_sk, sr_store_sk, sr_ticket_number, sr_return_quantity, sr_return_amt, sr_net_loss, sr_reason_sk",
        sreturns.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    // catalog_sales — similar structure to store_sales but with shipping.
    let n_cs = scaled(1_440_000, scale, 150);
    let mut csales: Vec<(u64, String)> = (1..=n_cs)
        .map(|t| {
            let qty = 1 + rng.below(100);
            let price = 1 + rng.below(100);
            let date_sk = first_sk + rng.below(n_dates);
            let ship_sk = first_sk + rng.below(n_dates);
            let ext_sales = qty * price;
            let wcost = rng.money(80);
            let list_p = price + 10;
            let time_sk = rng.below(86400);
            let item = rng.skewed_key(n_item);
            let cc = 1 + rng.below(n_call_center);
            let wh = 1 + rng.below(n_warehouse);
            let page = 1 + rng.below(100);
            let bc = rng.skewed_key(n_cust);
            let sc = rng.skewed_key(n_cust);
            let ba = 1 + rng.below(n_cust);
            let sa = 1 + rng.below(n_cust);
            let cd = 1 + rng.below(n_cdemo);
            let hd = 1 + rng.below(n_hdemo);
            let promo = 1 + rng.below(50);
            let ship_cost = ext_sales * 5 / 100;
            let paid_tax = ext_sales + ext_sales * 8 / 100;
            let profit = ext_sales * 8 / 100;
            let sm = 1 + rng.below(5);
            let row = format!(
                "({date_sk}, {time_sk}, {ship_sk}, {item}, {cc}, {wh}, {page}, {bc}, {sc}, {ba}, {sa}, {cd}, {hd}, {promo}, {t}, {qty}, {wcost}, {list_p}.00, {price}.00, 0.00, {ext_sales}.00, 0.00, {ext_sales}.00, 0.00, {ship_cost}.00, {ext_sales}.00, {paid_tax}.00, {profit}.00, {sm})"
            );
            (date_sk, row)
        })
        .collect();
    csales.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "catalog_sales",
        "cs_sold_date_sk, cs_sold_time_sk, cs_ship_date_sk, cs_item_sk, cs_call_center_sk, cs_warehouse_sk, cs_catalog_page_sk, cs_bill_customer_sk, cs_ship_customer_sk, cs_bill_addr_sk, cs_ship_addr_sk, cs_bill_cdemo_sk, cs_bill_hdemo_sk, cs_promo_sk, cs_order_number, cs_quantity, cs_wholesale_cost, cs_list_price, cs_sales_price, cs_coupon_amt, cs_ext_sales_price, cs_ext_wholesale_cost, cs_ext_list_price, cs_ext_discount_amt, cs_ext_ship_cost, cs_net_paid, cs_net_paid_inc_tax, cs_net_profit, cs_ship_mode_sk",
        csales.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    // catalog_returns
    let n_cr = scaled(144_000, scale, 15);
    let mut creturns: Vec<(u64, String)> = (1..=n_cr)
        .map(|t| {
            let date_sk = first_sk + rng.below(n_dates);
            let row = format!(
                "({date_sk}, {}, {t}, {}, {}, {}, {}, {}.00, {}.00, {}.00, {}.00, {}.00, {}.00)",
                rng.skewed_key(n_item),
                1 + rng.below(n_call_center),
                rng.skewed_key(n_cust),
                1 + rng.below(n_cust),
                rng.below(10),
                rng.below(50),
                rng.below(100),
                rng.below(50),
                rng.below(50),
                rng.below(50),
                rng.below(100),
            );
            (date_sk, row)
        })
        .collect();
    creturns.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "catalog_returns",
        "cr_returned_date_sk, cr_item_sk, cr_order_number, cr_call_center_sk, cr_returning_customer_sk, cr_returning_addr_sk, cr_return_quantity, cr_return_amount, cr_refunded_cash, cr_reversed_charge, cr_store_credit, cr_net_loss, cr_return_amt_inc_tax",
        creturns.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    // web_sales
    let n_ws = scaled(720_000, scale, 75);
    let mut wsales: Vec<(u64, String)> = (1..=n_ws)
        .map(|t| {
            let qty = 1 + rng.below(100);
            let price = 1 + rng.below(100);
            let date_sk = first_sk + rng.below(n_dates);
            let ship_sk = first_sk + rng.below(n_dates);
            let ext_sales = qty * price;
            let time_sk = rng.below(86400);
            let item = rng.skewed_key(n_item);
            let ws_site = 1 + rng.below(6);
            let wp = 1 + rng.below(100);
            let wh = 1 + rng.below(n_warehouse);
            let bc = rng.skewed_key(n_cust);
            let sc = rng.skewed_key(n_cust);
            let ba = 1 + rng.below(n_cust);
            let sa = 1 + rng.below(n_cust);
            let cd = 1 + rng.below(n_cdemo);
            let hd = 1 + rng.below(n_hdemo);
            let wcost = rng.money(80);
            let list_p = price + 10;
            let ship_cost = ext_sales * 5 / 100;
            let paid_tax = ext_sales + ext_sales * 8 / 100;
            let profit = ext_sales * 8 / 100;
            let promo = 1 + rng.below(50);
            let sm = 1 + rng.below(5);
            let row = format!(
                "({date_sk}, {time_sk}, {ship_sk}, {item}, {ws_site}, {wp}, {wh}, {bc}, {sc}, {ba}, {sa}, {cd}, {hd}, {t}, {qty}, {wcost}, {list_p}.00, {price}.00, 0.00, {ext_sales}.00, 0.00, {ext_sales}.00, 0.00, {ship_cost}.00, {ext_sales}.00, {paid_tax}.00, {profit}.00, {promo}, {sm})"
            );
            (date_sk, row)
        })
        .collect();
    wsales.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "web_sales",
        "ws_sold_date_sk, ws_sold_time_sk, ws_ship_date_sk, ws_item_sk, ws_web_site_sk, ws_web_page_sk, ws_warehouse_sk, ws_bill_customer_sk, ws_ship_customer_sk, ws_bill_addr_sk, ws_ship_addr_sk, ws_bill_cdemo_sk, ws_ship_hdemo_sk, ws_order_number, ws_quantity, ws_wholesale_cost, ws_list_price, ws_sales_price, ws_coupon_amt, ws_ext_sales_price, ws_ext_wholesale_cost, ws_ext_list_price, ws_ext_discount_amt, ws_ext_ship_cost, ws_net_paid, ws_net_paid_inc_tax, ws_net_profit, ws_promo_sk, ws_ship_mode_sk",
        wsales.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    // web_returns
    let n_wr = scaled(72_000, scale, 8);
    let mut wreturns: Vec<(u64, String)> = (1..=n_wr)
        .map(|t| {
            let date_sk = first_sk + rng.below(n_dates);
            let item = rng.skewed_key(n_item);
            let wp = 1 + rng.below(100);
            let cust = rng.skewed_key(n_cust);
            let addr = 1 + rng.below(n_cust);
            let cd1 = 1 + rng.below(n_cdemo);
            let cd2 = 1 + rng.below(n_cdemo);
            let raddr = 1 + rng.below(n_cust);
            let reason = 1 + rng.below(35);
            let qty = rng.below(10);
            let amt = rng.below(100);
            let cash = rng.below(50);
            let reversed = rng.below(50);
            let fee = rng.below(50);
            let loss = rng.below(50);
            let row = format!(
                "({date_sk}, {item}, {t}, {t}, {wp}, {cust}, {addr}, {cd1}, {cd2}, {raddr}, {reason}, {qty}, {amt}.00, {cash}.00, {reversed}.00, {fee}.00, {loss}.00)"
            );
            (date_sk, row)
        })
        .collect();
    wreturns.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "web_returns",
        "wr_returned_date_sk, wr_item_sk, ws_order_number, wr_order_number, wr_web_page_sk, wr_returning_customer_sk, wr_returning_addr_sk, wr_refunded_cdemo_sk, wr_returning_cdemo_sk, wr_refunded_addr_sk, wr_reason_sk, wr_return_quantity, wr_return_amt, wr_refunded_cash, cr_reversed_charge, wr_fee, wr_net_loss",
        wreturns.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    // inventory
    let n_inv = scaled(78_000, scale, 100);
    let mut inv: Vec<(u64, String)> = (1..=n_inv)
        .map(|_| {
            let date_sk = first_sk + rng.below(n_dates);
            let row = format!(
                "({date_sk}, {}, {}, {})",
                rng.skewed_key(n_item),
                1 + rng.below(n_warehouse),
                rng.below(1000),
            );
            (date_sk, row)
        })
        .collect();
    inv.sort_by_key(|(sk, _)| *sk);
    out.extend(chunked_inserts(
        "inventory",
        "inv_date_sk, inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand",
        inv.into_iter().map(|(_, row)| row).collect(),
        200,
    ));
    out
}

// --- Queries (copied verbatim from the conformance suites) -----------------

fn tpch_queries() -> Vec<(&'static str, String)> {
    vec![
        ("Q1", "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice*(1-l_discount)) as sum_disc_price, sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem where l_shipdate <= DATE '1998-09-01' group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus".to_string()),
        ("Q2", "select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from part, supplier, partsupp, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 15 and p_type like '%STEEL' and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'AMERICA' and ps_supplycost = (select min(ps_supplycost) from partsupp, supplier, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'AMERICA') order by s_acctbal desc, n_name, s_name, p_partkey".to_string()),
        ("Q3", "select l_orderkey, sum(l_extendedprice*(1-l_discount)) as revenue, o_orderdate, o_shippriority from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < DATE '1995-03-15' and l_shipdate > DATE '1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate".to_string()),
        ("Q4", "select o_orderpriority, count(*) as order_count from orders where o_orderdate >= DATE '1993-07-01' and o_orderdate < DATE '1993-10-01' and exists (select * from lineitem where l_orderkey = o_orderkey and l_commitdate < l_receiptdate) group by o_orderpriority order by o_orderpriority".to_string()),
        ("Q5", "select n_name, sum(l_extendedprice*(1-l_discount)) as revenue from customer, orders, lineitem, supplier, nation, region where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'AMERICA' and o_orderdate >= DATE '1994-01-01' and o_orderdate < DATE '1995-01-01' group by n_name order by revenue desc".to_string()),
        ("Q6", "select sum(l_extendedprice*l_discount) as revenue from lineitem where l_shipdate >= DATE '1994-01-01' and l_shipdate < DATE '1995-01-01' and l_discount between 0.05 and 0.07 and l_quantity < 24".to_string()),
        ("Q7", "select supp_nation, cust_nation, l_year, sum(volume) as revenue from (select n1.n_name as supp_nation, n2.n_name as cust_nation, extract(year from l_shipdate) as l_year, l_extendedprice*(1-l_discount) as volume from supplier, lineitem, orders, customer, nation n1, nation n2 where s_suppkey = l_suppkey and o_orderkey = l_orderkey and c_custkey = o_custkey and s_nationkey = n1.n_nationkey and c_nationkey = n2.n_nationkey and ((n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE') or (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')) and l_shipdate between DATE '1995-01-01' and DATE '1996-12-31') as shipping group by supp_nation, cust_nation, l_year order by supp_nation, cust_nation, l_year".to_string()),
        ("Q8", "select o_year, sum(case when nation = 'GERMANY' then volume else 0 end) / sum(volume) as mkt_share from (select extract(year from o_orderdate) as o_year, l_extendedprice*(1-l_discount) as volume, n2.n_name as nation from part, supplier, lineitem, orders, customer, nation n1, nation n2, region where p_partkey = l_partkey and s_suppkey = l_suppkey and l_orderkey = o_orderkey and o_custkey = c_custkey and c_nationkey = n1.n_nationkey and n1.n_regionkey = r_regionkey and r_name = 'EUROPE' and s_nationkey = n2.n_nationkey and o_orderdate between DATE '1995-01-01' and DATE '1996-12-31' and p_type = 'STANDARD POLISHED TIN') as all_nations group by o_year order by o_year".to_string()),
        ("Q9", "select nation, o_year, sum(amount) as sum_profit from (select n_name as nation, extract(year from o_orderdate) as o_year, l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity as amount from part, supplier, lineitem, partsupp, orders, nation where s_suppkey = l_suppkey and ps_suppkey = l_suppkey and ps_partkey = l_partkey and p_partkey = l_partkey and o_orderkey = l_orderkey and s_nationkey = n_nationkey and p_name like '%green%') as profit group by nation, o_year order by nation, o_year desc".to_string()),
        ("Q10", "select c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment from customer, orders, lineitem, nation where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= DATE '1993-10-01' and o_orderdate < DATE '1994-01-01' and l_returnflag = 'R' and c_nationkey = n_nationkey group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by revenue desc".to_string()),
        ("Q11", "select ps_partkey, sum(ps_supplycost*ps_availqty) as value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' group by ps_partkey having sum(ps_supplycost*ps_availqty) > (select sum(ps_supplycost*ps_availqty) * 0.0001 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY') order by value desc".to_string()),
        ("Q12", "select l_shipmode, sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count, sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count from orders, lineitem where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP') and l_commitdate < l_receiptdate and l_shipdate < l_commitdate and l_receiptdate >= DATE '1994-01-01' and l_receiptdate < DATE '1995-01-01' group by l_shipmode order by l_shipmode".to_string()),
        ("Q13", "select c_count, count(*) as custdist from (select c_custkey, count(o_orderkey) as c_count from customer left outer join orders on c_custkey = o_custkey and o_comment not like '%special%requests%' group by c_custkey) as c_orders group by c_count order by custdist desc, c_count desc".to_string()),
        ("Q14", "select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice*(1-l_discount) else 0 end) / sum(l_extendedprice*(1-l_discount)) as promo_revenue from lineitem, part where l_partkey = p_partkey and l_shipdate >= DATE '1995-09-01' and l_shipdate < DATE '1995-10-01'".to_string()),
        ("Q15", "select s_suppkey, s_name, s_address, s_phone, total_revenue from supplier, (select l_suppkey as supplier_no, sum(l_extendedprice*(1-l_discount)) as total_revenue from lineitem where l_shipdate >= DATE '1996-01-01' and l_shipdate < DATE '1996-04-01' group by l_suppkey) as revenue0 where s_suppkey = supplier_no order by s_suppkey".to_string()),
        ("Q16", "select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt from partsupp, part where p_partkey = ps_partkey and p_brand <> 'Brand#45' and p_type not like 'MEDIUM POLISHED%' and p_size in (49, 14, 23, 45, 19, 3, 36, 9) group by p_brand, p_type, p_size order by supplier_cnt desc, p_brand, p_type, p_size".to_string()),
        ("Q17", "select sum(l_extendedprice) / 7.0 as avg_yearly from lineitem, part where p_partkey = l_partkey and p_brand = 'Brand#23' and p_container = 'MED BOX' and l_quantity < (select 0.2 * avg(l_quantity) from lineitem where l_partkey = p_partkey)".to_string()),
        ("Q18", "select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) from customer, orders, lineitem where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300) and c_custkey = o_custkey and o_orderkey = l_orderkey group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice order by o_totalprice desc, o_orderdate".to_string()),
        ("Q19", "select sum(l_extendedprice*(1-l_discount)) as revenue from lineitem, part where (p_partkey = l_partkey and p_brand = 'Brand#12' and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') and l_quantity >= 1 and l_quantity <= 11 and p_size between 1 and 5 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON') or (p_partkey = l_partkey and p_brand = 'Brand#23' and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') and l_quantity >= 10 and l_quantity <= 20 and p_size between 1 and 10 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON')".to_string()),
        ("Q20", "select s_name, s_address from supplier, nation where s_suppkey in (select ps_suppkey from partsupp where ps_partkey in (select p_partkey from part where p_name like 'forest%') and ps_availqty > (select 0.5 * sum(l_quantity) from lineitem where l_partkey = ps_partkey and l_suppkey = ps_suppkey and l_shipdate >= DATE '1994-01-01' and l_shipdate < DATE '1995-01-01')) and s_nationkey = n_nationkey and n_name = 'CANADA' order by s_name".to_string()),
        ("Q21", "select s_name, count(*) as numwait from supplier, lineitem l1, orders, nation where s_suppkey = l1.l_suppkey and o_orderkey = l1.l_orderkey and o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate and exists (select * from lineitem l2 where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey) and not exists (select * from lineitem l3 where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey and l3.l_receiptdate > l3.l_commitdate) and s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA' group by s_name order by numwait desc, s_name".to_string()),
        ("Q22", "select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from (select substring(c_phone from 1 for 2) as cntrycode, c_acctbal from customer where substring(c_phone from 1 for 2) in ('13', '31', '23', '29', '30', '18', '17') and c_acctbal > (select avg(c_acctbal) from customer where c_acctbal > 0.00 and substring(c_phone from 1 for 2) in ('13', '31', '23', '29', '30', '18', '17')) and not exists (select * from orders where o_custkey = c_custkey)) as custsale group by cntrycode order by cntrycode".to_string()),
    ]
}

fn tpcds_queries() -> Vec<(&'static str, String)> {
    vec![
        // q1 — returns: customers above average return per store (CTE + correlated subquery).
        ("q1", "with customer_total_return as (select sr_customer_sk as ctr_customer_sk, sr_store_sk as ctr_store_sk, sum(sr_return_amt) as ctr_total_return from store_returns, date_dim where sr_returned_date_sk = d_date_sk and d_year = 2000 group by sr_customer_sk, sr_store_sk) select c_customer_id from customer_total_return ctr1, store, customer where ctr1.ctr_total_return > (select avg(ctr_total_return)*1.2 from customer_total_return ctr2 where ctr1.ctr_store_sk = ctr2.ctr_store_sk) and s_store_sk = ctr1.ctr_store_sk and s_state = 'TN' and ctr1.ctr_customer_sk = c_customer_sk order by c_customer_id".to_string()),
        // q3 — manufacturer revenue by year/brand.
        ("q3", "select dt.d_year, item.i_brand_id brand_id, item.i_brand brand, sum(ss_ext_sales_price) sum_agg from date_dim dt, store_sales, item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and item.i_manufact_id = 100 and dt.d_moy = 11 group by dt.d_year, item.i_brand, item.i_brand_id order by dt.d_year, sum_agg desc, brand_id".to_string()),
        // q6 — address-state count with subquery on item avg price.
        ("q6", "select a.ca_state state, count(*) cnt from customer_address a, customer c, store_sales s, date_dim d, item i where a.ca_address_sk = c.c_current_addr_sk and c.c_customer_sk = s.ss_customer_sk and s.ss_sold_date_sk = d.d_date_sk and s.ss_item_sk = i.i_item_sk and d.d_month_seq = (select distinct d_month_seq from date_dim where d_year = 2000 and d_moy = 1) and i.i_current_price > 1.2 * (select avg(j.i_current_price) from item j where j.i_category = i.i_category) group by a.ca_state having count(*) >= 1 order by cnt, a.ca_state".to_string()),
        // q7 — promo/demographic filtered averages.
        ("q7", "select i_item_id, avg(ss_quantity) agg1, avg(ss_list_price) agg2, avg(ss_coupon_amt) agg3, avg(ss_sales_price) agg4 from store_sales, customer_demographics, date_dim, item, promotion where ss_sold_date_sk = d_date_sk and ss_item_sk = i_item_sk and ss_cdemo_sk = cd_demo_sk and ss_promo_sk = p_promo_sk and cd_gender = 'M' and cd_marital_status = 'S' and cd_education_status = 'College' and (p_channel_email = 'N' or p_channel_event = 'N') and d_year = 2000 group by i_item_id order by i_item_id".to_string()),
        // q8 — store net profit by store name for a date range with a subquery on zips.
        ("q8", "select s_store_name, sum(ss_net_profit) from store_sales, date_dim, store where ss_store_sk = s_store_sk and ss_sold_date_sk = d_date_sk and d_qoy = 2 and d_year = 2000 group by s_store_name order by s_store_name".to_string()),
        // q9 — CASE buckets over quantity ranges with scalar subqueries.
        ("q9", "with b1 as (select count(*) as cnt, avg(ss_ext_discount_amt) as disc_avg, avg(ss_net_paid) as paid_avg from store_sales where ss_quantity between 1 and 20), b2 as (select count(*) as cnt, avg(ss_ext_discount_amt) as disc_avg, avg(ss_net_paid) as paid_avg from store_sales where ss_quantity between 21 and 40) select case when b1.cnt > 100 then b1.disc_avg else b1.paid_avg end as bucket1, case when b2.cnt > 100 then b2.disc_avg else b2.paid_avg end as bucket2 from b1, b2".to_string()),
        // q10 — demographics count with EXISTS.
        ("q10", "select cd_gender, cd_marital_status, cd_education_status, count(*) cnt1, cd_purchase_estimate, count(*) cnt2, cd_credit_rating, count(*) cnt3, cd_dep_count, count(*) cnt4, cd_dep_employed_count, count(*) cnt5, cd_dep_college_count, count(*) cnt6 from customer c, customer_address ca, customer_demographics where c.c_current_addr_sk = ca.ca_address_sk and cd_demo_sk = c.c_current_cdemo_sk and exists (select * from store_sales, date_dim where c.c_customer_sk = ss_customer_sk and ss_sold_date_sk = d_date_sk and d_year = 2000) group by cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating, cd_dep_count, cd_dep_employed_count, cd_dep_college_count order by cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating, cd_dep_count, cd_dep_employed_count, cd_dep_college_count".to_string()),
        // q12 — web sales revenue ratio window.
        ("q12", "select i_item_id, i_category, i_class, i_current_price, sum(ws_ext_sales_price) as itemrevenue, sum(ws_ext_sales_price)*100/sum(sum(ws_ext_sales_price)) over (partition by i_class) as revenueratio from web_sales, item, date_dim where ws_item_sk = i_item_sk and i_category in ('Sports', 'Books', 'Home') and ws_sold_date_sk = d_date_sk and d_year = 2000 group by i_item_id, i_category, i_class, i_current_price order by i_category, i_class, i_item_id, revenueratio".to_string()),
        // q14 — cross-channel sales (simplified, removing INTERSECT-on-3-channels for now — see q14b for full).
        ("q14a", "with avg_sales as (select avg(quantity*list_price) average_sales from (select ss_quantity quantity, ss_list_price list_price from store_sales, date_dim where ss_sold_date_sk = d_date_sk and d_year between 1999 and 2000 union all select cs_quantity quantity, cs_list_price list_price from catalog_sales, date_dim where cs_sold_date_sk = d_date_sk and d_year between 1999 and 2000) x) select 'store' channel, i_brand_id, i_class_id, i_category_id, sum(ss_quantity*ss_list_price) sales, count(*) number_sales from store_sales, item, date_dim where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk and d_year = 2000 and d_moy = 11 group by i_brand_id, i_class_id, i_category_id having sum(ss_quantity*ss_list_price) > (select average_sales from avg_sales)".to_string()),
        // q15 — catalog sales by zip/state/qoy.
        ("q15", "select ca_zip, sum(cs_sales_price) from catalog_sales, customer, customer_address, date_dim where cs_bill_customer_sk = c_customer_sk and c_current_addr_sk = ca_address_sk and (ca_state in ('CA','WA','GA') or cs_sales_price > 500) and cs_sold_date_sk = d_date_sk and d_qoy = 2 and d_year = 2000 group by ca_zip order by ca_zip".to_string()),
        // q17 — cross-fact stddev.
        ("q17", "select i_item_id, s_state, count(ss_quantity) as store_sales_quantitycount, avg(ss_quantity) as store_sales_quantityave, stddev_samp(ss_quantity) as store_sales_quantitystdev, stddev_samp(ss_quantity)/avg(ss_quantity) as store_sales_quantitycov, count(sr_return_quantity) as store_returns_quantitycount, avg(sr_return_quantity) as store_returns_quantityave, stddev_samp(sr_return_quantity) as store_returns_quantitystdev, stddev_samp(sr_return_quantity)/avg(sr_return_quantity) as store_returns_quantitycov, count(cs_quantity) as catalog_sales_quantitycount, avg(cs_quantity) as catalog_sales_quantityave, stddev_samp(cs_quantity) as catalog_sales_quantitystdev, stddev_samp(cs_quantity)/avg(cs_quantity) as catalog_sales_quantitycov from store_sales, store_returns, catalog_sales, date_dim d1, store, item where d1.d_date_sk = ss_sold_date_sk and i_item_sk = ss_item_sk and s_store_sk = ss_store_sk and ss_customer_sk = sr_customer_sk and ss_item_sk = sr_item_sk and ss_ticket_number = sr_ticket_number and sr_customer_sk = cs_bill_customer_sk and sr_item_sk = cs_item_sk group by i_item_id, s_state order by i_item_id, s_state".to_string()),
        // q19 — brand/manufacturer revenue.
        ("q19", "select i_brand_id brand_id, i_brand brand, i_manufact_id, i_manufact, sum(ss_ext_sales_price) ext_price from date_dim, store_sales, item, customer, customer_address, store where d_date_sk = ss_sold_date_sk and ss_item_sk = i_item_sk and d_moy = 11 and d_year = 2000 and ss_customer_sk = c_customer_sk and c_current_addr_sk = ca_address_sk and ss_store_sk = s_store_sk group by i_brand, i_brand_id, i_manufact_id, i_manufact order by ext_price desc, i_brand, i_brand_id, i_manufact_id, i_manufact".to_string()),
        // q20 — catalog revenue ratio window.
        ("q20", "select i_item_id, i_category, i_class, i_current_price, sum(cs_ext_sales_price) as itemrevenue, sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over (partition by i_class) as revenueratio from catalog_sales, item, date_dim where cs_item_sk = i_item_sk and i_category in ('Sports','Books','Home') and cs_sold_date_sk = d_date_sk and d_year = 2000 group by i_item_id, i_category, i_class, i_current_price order by i_category, i_class, i_item_id, revenueratio".to_string()),
        // q22 — inventory rollup.
        ("q22", "select i_product_name, i_brand, i_class, i_category, avg(inv_quantity_on_hand) qoh from inventory, date_dim, item where inv_date_sk = d_date_sk and inv_item_sk = i_item_sk and d_month_seq between 12 and 23 group by rollup(i_product_name, i_brand, i_class, i_category) order by qoh, i_product_name, i_brand, i_class, i_category".to_string()),
        // q23a — frequent items sales.
        ("q23a", "with frequent_ss_items as (select i_item_sk item_sk, d_date solddate, count(*) cnt from store_sales, date_dim, item where ss_sold_date_sk = d_date_sk and ss_item_sk = i_item_sk and d_year in (1998, 1999, 2000) group by i_item_sk, d_date having count(*) > 2), max_store_sales as (select max(csales) tpcds_cmax from (select c_customer_sk, sum(ss_quantity*ss_sales_price) csales from store_sales, customer, date_dim where ss_customer_sk = c_customer_sk and ss_sold_date_sk = d_date_sk and d_year in (1998, 1999, 2000) group by c_customer_sk) t1) select sum(sales) from (select cs_quantity*cs_list_price sales from catalog_sales, date_dim where d_year = 1999 and cs_sold_date_sk = d_date_sk and cs_item_sk in (select item_sk from frequent_ss_items)) t2".to_string()),
        // q24 — store returns with demographics (simplified).
        ("q24a", "with ssales as (select c_last_name, c_first_name, s_store_name, sum(ss_net_paid) netpaid from store_sales, store_returns, store, item, customer where ss_ticket_number = sr_ticket_number and ss_item_sk = sr_item_sk and ss_customer_sk = c_customer_sk and ss_item_sk = i_item_sk and ss_store_sk = s_store_sk group by c_last_name, c_first_name, s_store_name) select c_last_name, c_first_name, s_store_name, sum(netpaid) paid from ssales group by c_last_name, c_first_name, s_store_name having sum(netpaid) > (select 0.05*avg(netpaid) from ssales) order by c_last_name, c_first_name, s_store_name".to_string()),
        // q25 — item/store profit and loss across fact tables.
        ("q25", "select i_item_id, s_store_id, s_store_name, sum(ss_net_profit) as store_sales_profit, sum(sr_net_loss) as store_returns_loss, sum(cs_net_profit) as catalog_sales_profit from store_sales, store_returns, catalog_sales, date_dim d1, date_dim d2, store, item where d1.d_moy = 4 and d1.d_year = 2000 and d1.d_date_sk = ss_sold_date_sk and i_item_sk = ss_item_sk and s_store_sk = ss_store_sk and ss_customer_sk = sr_customer_sk and ss_item_sk = sr_item_sk and ss_ticket_number = sr_ticket_number and sr_returned_date_sk = d2.d_date_sk and sr_customer_sk = cs_bill_customer_sk and sr_item_sk = cs_item_sk group by i_item_id, s_store_id, s_store_name order by i_item_id, s_store_id, s_store_name".to_string()),
        // q26 — catalog sales + promo/demographics.
        ("q26", "select i_item_id, avg(cs_quantity) agg1, avg(cs_list_price) agg2, avg(cs_coupon_amt) agg3, avg(cs_sales_price) agg4 from catalog_sales, customer_demographics, date_dim, item, promotion where cs_sold_date_sk = d_date_sk and cs_item_sk = i_item_sk and cs_bill_cdemo_sk = cd_demo_sk and cs_promo_sk = p_promo_sk and cd_gender = 'M' and cd_marital_status = 'S' and cd_education_status = 'College' and (p_channel_email = 'N' or p_channel_event = 'N') and d_year = 2000 group by i_item_id order by i_item_id".to_string()),
        // q27 — store sales rollup with demographics.
        ("q27", "select i_item_id, s_state, grouping(s_state) g_state, avg(ss_quantity) agg1, avg(ss_list_price) agg2, avg(ss_coupon_amt) agg3, avg(ss_sales_price) agg4 from store_sales, customer_demographics, date_dim, store, item where ss_sold_date_sk = d_date_sk and ss_item_sk = i_item_sk and ss_store_sk = s_store_sk and ss_cdemo_sk = cd_demo_sk and cd_gender = 'M' and cd_marital_status = 'S' and cd_education_status = 'College' and d_year = 2000 group by rollup(i_item_id, s_state) order by i_item_id, s_state".to_string()),
        // q28 — bucketed count/avg/various (derived table product).
        ("q28", "select * from (select avg(ss_list_price) B1_LP, count(ss_list_price) B1_CNT, count(distinct ss_list_price) B1_CNTD from store_sales where ss_quantity between 0 and 5 and (ss_list_price between 8 and 18 or ss_coupon_amt between 459 and 1459 or ss_wholesale_cost between 57 and 77)) B1".to_string()),
        // q29 — item/store/store-returns/catalog quantities.
        ("q29", "select i_item_id, s_store_id, s_store_name, sum(ss_quantity) as store_sales_quantity, sum(sr_return_quantity) as store_returns_quantity, sum(cs_quantity) as catalog_sales_quantity from store_sales, store_returns, catalog_sales, date_dim d1, date_dim d2, date_dim d3, store, item where d1.d_moy = 9 and d1.d_year = 1999 and d1.d_date_sk = ss_sold_date_sk and i_item_sk = ss_item_sk and s_store_sk = ss_store_sk and ss_customer_sk = sr_customer_sk and ss_item_sk = sr_item_sk and ss_ticket_number = sr_ticket_number and sr_returned_date_sk = d2.d_date_sk and sr_customer_sk = cs_bill_customer_sk and sr_item_sk = cs_item_sk and cs_sold_date_sk = d3.d_date_sk group by i_item_id, s_store_id, s_store_name order by i_item_id, s_store_id, s_store_name".to_string()),
        // q32 — excess discount (correlated subquery).
        ("q32", "select sum(cs_ext_discount_amt) as excess_discount from catalog_sales, item, date_dim where i_manufact_id = 100 and i_item_sk = cs_item_sk and d_date_sk = cs_sold_date_sk and cs_ext_discount_amt > (select 1.3 * avg(cs_ext_discount_amt) from catalog_sales, date_dim where cs_item_sk = i_item_sk and d_date_sk = cs_sold_date_sk)".to_string()),
        // q33 — union-all sales by manufacturer for a category.
        ("q33", "with ss as (select i_manufact_id, sum(ss_ext_sales_price) total_sales from store_sales, date_dim, item where i_manufact_id in (select i_manufact_id from item where i_category in ('Electronics')) and ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk and d_year = 1999 group by i_manufact_id), cs as (select i_manufact_id, sum(cs_ext_sales_price) total_sales from catalog_sales, date_dim, item where i_manufact_id in (select i_manufact_id from item where i_category in ('Electronics')) and cs_item_sk = i_item_sk and cs_sold_date_sk = d_date_sk and d_year = 1999 group by i_manufact_id) select i_manufact_id, sum(total_sales) total_sales from (select * from ss union all select * from cs) tmp1 group by i_manufact_id order by total_sales".to_string()),
        // q34 — ticket count by demographics.
        ("q34", "select c_last_name, c_first_name, c_salutation, c_preferred_cust_flag, ss_ticket_number, cnt from (select ss_ticket_number, ss_customer_sk, count(*) cnt from store_sales, date_dim, store, household_demographics where ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk and ss_hdemo_sk = hd_demo_sk and d_year in (1998, 1999, 2000) group by ss_ticket_number, ss_customer_sk) dn, customer where ss_customer_sk = c_customer_sk order by c_last_name, c_first_name, c_salutation, c_preferred_cust_flag desc, ss_ticket_number".to_string()),
        // q35 — demographics with EXISTS store_sales/web_sales/catalog_sales.
        ("q35", "select ca_state, cd_gender, cd_marital_status, cd_dep_count, count(*) cnt1, min(cd_dep_count), max(cd_dep_count), avg(cd_dep_count), cd_dep_employed_count, count(*) cnt2, min(cd_dep_employed_count), max(cd_dep_employed_count), avg(cd_dep_employed_count), cd_dep_college_count, count(*) cnt3, min(cd_dep_college_count), max(cd_dep_college_count), avg(cd_dep_college_count) from customer c, customer_address ca, customer_demographics where c.c_current_addr_sk = ca.ca_address_sk and cd_demo_sk = c.c_current_cdemo_sk and exists (select * from store_sales, date_dim where c.c_customer_sk = ss_customer_sk and ss_sold_date_sk = d_date_sk and d_year = 2000 and d_qoy < 4) group by ca_state, cd_gender, cd_marital_status, cd_dep_count, cd_dep_employed_count, cd_dep_college_count order by ca_state, cd_gender, cd_marital_status, cd_dep_count, cd_dep_employed_count, cd_dep_college_count".to_string()),
        // q36 — gross margin rank window with rollup.
        ("q36", "select sum(ss_net_profit)/sum(ss_ext_sales_price) as gross_margin, i_category, i_class, grouping(i_category)+grouping(i_class) as lochierarchy, rank() over (partition by grouping(i_category)+grouping(i_class), case when grouping(i_class) = 0 then i_category end order by sum(ss_net_profit)/sum(ss_ext_sales_price) asc) as rank_within_parent from store_sales, date_dim d1, item, store where d1.d_year = 2000 and d1.d_date_sk = ss_sold_date_sk and i_item_sk = ss_item_sk and s_store_sk = ss_store_sk group by rollup(i_category, i_class) order by lochierarchy desc, case when lochierarchy = 0 then i_category end, rank_within_parent".to_string()),
        // q38 — INTERSECT across store/catalog/web sales.
        ("q38", "select count(*) from (select distinct c_last_name, c_first_name, d_date from store_sales, date_dim, customer where ss_sold_date_sk = d_date_sk and ss_customer_sk = c_customer_sk intersect select distinct c_last_name, c_first_name, d_date from catalog_sales, date_dim, customer where cs_sold_date_sk = d_date_sk and cs_bill_customer_sk = c_customer_sk intersect select distinct c_last_name, c_first_name, d_date from web_sales, date_dim, customer where ws_sold_date_sk = d_date_sk and ws_bill_customer_sk = c_customer_sk) hot_cust".to_string()),
        // q42 — year/category revenue.
        ("q42", "select dt.d_year, item.i_category_id, item.i_category, sum(ss_ext_sales_price) as revenue from date_dim dt, store_sales, item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and item.i_manufact_id = 100 and dt.d_moy = 11 and dt.d_year = 2000 group by dt.d_year, item.i_category_id, item.i_category order by revenue desc, dt.d_year".to_string()),
        // q43 — store sales by day-name (CASE + window).
        ("q43", "select s_store_name, s_store_id, sum(case when (d_day_name='Sunday') then ss_sales_price else null end) sun_sales, sum(case when (d_day_name='Monday') then ss_sales_price else null end) mon_sales, sum(case when (d_day_name='Tuesday') then ss_sales_price else null end) tue_sales, sum(case when (d_day_name='Wednesday') then ss_sales_price else null end) wed_sales, sum(case when (d_day_name='Thursday') then ss_sales_price else null end) thu_sales, sum(case when (d_day_name='Friday') then ss_sales_price else null end) fri_sales, sum(case when (d_day_name='Saturday') then ss_sales_price else null end) sat_sales from date_dim, store_sales, store where d_date_sk = ss_sold_date_sk and s_store_sk = ss_store_sk and d_year = 2000 group by s_store_name, s_store_id order by s_store_name, s_store_id".to_string()),
        // q45 — web sales by zip/city.
        ("q45", "select ca_zip, ca_city, sum(ws_sales_price) from web_sales, customer, customer_address, date_dim, item where ws_bill_customer_sk = c_customer_sk and c_current_addr_sk = ca_address_sk and ws_item_sk = i_item_sk and ws_sold_date_sk = d_date_sk and d_qoy = 2 and d_year = 2000 group by ca_zip, ca_city order by ca_zip, ca_city".to_string()),
        // q46 — store sales with household demographics and city.
        ("q46", "with dn as (select ss_ticket_number, ss_customer_sk, ca_city as bought_city, sum(ss_coupon_amt) as amt, sum(ss_net_profit) as profit from store_sales, date_dim, store, household_demographics, customer_address where ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk and ss_hdemo_sk = hd_demo_sk and ss_addr_sk = ca_address_sk and (hd_dep_count = 4 or hd_vehicle_count = 3) and d_year in (1998, 1999, 2000) group by ss_ticket_number, ss_customer_sk, ca_city) select c_last_name, c_first_name, bought_city, ss_ticket_number, amt, profit from dn, customer where ss_customer_sk = c_customer_sk order by c_last_name, c_first_name, bought_city, ss_ticket_number".to_string()),
        // q48 — store sales sum quantity by demographics/address.
        ("q48", "select sum(ss_quantity) from store_sales, store, customer_demographics, customer_address, date_dim where s_store_sk = ss_store_sk and ss_sold_date_sk = d_date_sk and d_year = 2000 and cd_demo_sk = ss_cdemo_sk and cd_marital_status = 'M' and cd_education_status = '4 yr Degree' and ss_sales_price between 100.00 and 150.00 and ss_addr_sk = ca_address_sk and ca_country = 'United States' and ca_state in ('CO', 'OH', 'TX')".to_string()),
        // q50 — store returns by store with return-date lag buckets.
        ("q50", "select s_store_name, s_state, sum(case when (sr_returned_date_sk - ss_sold_date_sk <= 30) then 1 else 0 end) as d30, sum(case when (sr_returned_date_sk - ss_sold_date_sk > 30) and (sr_returned_date_sk - ss_sold_date_sk <= 60) then 1 else 0 end) as d31_60, sum(case when (sr_returned_date_sk - ss_sold_date_sk > 60) then 1 else 0 end) as d60plus from store_sales, store_returns, store where ss_ticket_number = sr_ticket_number and ss_item_sk = sr_item_sk and ss_customer_sk = sr_customer_sk and ss_store_sk = s_store_sk group by s_store_name, s_state order by s_store_name, s_state".to_string()),
        // q52 — brand revenue by year/month.
        ("q52", "select dt.d_year, item.i_brand_id as brand_id, item.i_brand as brand, sum(ss_ext_sales_price) as ext_price from date_dim dt, store_sales, item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and dt.d_moy = 11 and dt.d_year = 2000 group by dt.d_year, item.i_brand, item.i_brand_id order by dt.d_year, ext_price desc, brand_id".to_string()),
        // q53 — window avg of manufacturer sales with CASE.
        ("q53", "select * from (select i_manufact_id, sum(ss_sales_price) sum_sales, avg(sum(ss_sales_price)) over (partition by i_manufact_id) avg_quarterly_sales from item, store_sales, date_dim, store where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk group by i_manufact_id, d_qoy) tmp1 where case when avg_quarterly_sales > 0 then abs(sum_sales - avg_quarterly_sales)/avg_quarterly_sales else null end > 0.1 order by avg_quarterly_sales, sum_sales, i_manufact_id".to_string()),
        // q55 — single-brand revenue.
        ("q55", "select i_brand_id as brand_id, i_brand as brand, sum(ss_ext_sales_price) as ext_price from date_dim, store_sales, item where store_sales.ss_sold_date_sk = date_dim.d_date_sk and store_sales.ss_item_sk = item.i_item_sk and i_manufact_id = 100 and d_moy = 11 and d_year = 2000 group by i_brand, i_brand_id order by ext_price desc, i_brand_id".to_string()),
        // q56 — union-all sales by item across channels.
        ("q56", "with ss as (select i_item_id, sum(ss_ext_sales_price) total_sales from store_sales, date_dim, item where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk and d_year = 2000 group by i_item_id), cs as (select i_item_id, sum(cs_ext_sales_price) total_sales from catalog_sales, date_dim, item where cs_item_sk = i_item_sk and cs_sold_date_sk = d_date_sk and d_year = 2000 group by i_item_id) select i_item_id, sum(total_sales) total_sales from (select * from ss union all select * from cs) tmp1 group by i_item_id order by total_sales, i_item_id".to_string()),
        // q59 — store sales day-of-week sums by store/week.
        ("q59", "with wss as (select d_week_seq, ss_store_sk, sum(case when (d_day_name='Sunday') then ss_sales_price else null end) sun_sales, sum(case when (d_day_name='Monday') then ss_sales_price else null end) mon_sales, sum(case when (d_day_name='Tuesday') then ss_sales_price else null end) tue_sales from store_sales, date_dim where d_date_sk = ss_sold_date_sk group by d_week_seq, ss_store_sk) select s_store_name, s_store_id, d_week_seq from wss, store where ss_store_sk = s_store_sk order by s_store_name, s_store_id, d_week_seq".to_string()),
        // q60 — union-all sales by item across channels (Music category).
        ("q60", "with ss as (select i_item_id, sum(ss_ext_sales_price) total_sales from store_sales, date_dim, item where i_category in ('Music') and ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk and d_year = 1999 group by i_item_id), cs as (select i_item_id, sum(cs_ext_sales_price) total_sales from catalog_sales, date_dim, item where i_category in ('Music') and cs_item_sk = i_item_sk and cs_sold_date_sk = d_date_sk and d_year = 1999 group by i_item_id) select i_item_id, sum(total_sales) total_sales from (select * from ss union all select * from cs) tmp1 group by i_item_id order by i_item_id, total_sales".to_string()),
        // q61 — promotion percentage.
        ("q61", "select promotions, total, cast(promotions as decimal(15,4))/cast(total as decimal(15,4))*100 from (select sum(ss_ext_sales_price) promotions from store_sales, store, promotion, date_dim, customer, customer_address, item where ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk and ss_promo_sk = p_promo_sk and ss_customer_sk = c_customer_sk and ca_address_sk = c_current_addr_sk and ss_item_sk = i_item_sk and d_year = 1999 and d_moy = 11) promotional_sales, (select sum(ss_ext_sales_price) total from store_sales, store, date_dim, customer, customer_address, item where ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk and ss_customer_sk = c_customer_sk and ca_address_sk = c_current_addr_sk and ss_item_sk = i_item_sk and d_year = 1999 and d_moy = 11) all_sales".to_string()),
        // q63 — window avg monthly sales with CASE.
        ("q63", "select * from (select i_manager_id, sum(ss_sales_price) sum_sales, avg(sum(ss_sales_price)) over (partition by i_manager_id) avg_monthly_sales from item, store_sales, date_dim, store where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk group by i_manager_id, d_moy) tmp1 where case when avg_monthly_sales > 0 then abs(sum_sales - avg_monthly_sales)/avg_monthly_sales else null end > 0.1 order by i_manager_id, avg_monthly_sales, sum_sales".to_string()),
        // q65 — store revenue vs average.
        ("q65", "select s_store_name, i_item_desc, sc.revenue, i_current_price, i_wholesale_cost, i_brand from store, item, (select ss_store_sk, avg(revenue) as ave from (select ss_store_sk, ss_item_sk, sum(ss_sales_price) as revenue from store_sales, date_dim where ss_sold_date_sk = d_date_sk group by ss_store_sk, ss_item_sk) sa group by ss_store_sk) sb, (select ss_store_sk, ss_item_sk, sum(ss_sales_price) as revenue from store_sales, date_dim where ss_sold_date_sk = d_date_sk group by ss_store_sk, ss_item_sk) sc where sb.ss_store_sk = sc.ss_store_sk and sc.revenue <= 0.1 * sb.ave and s_store_sk = sc.ss_store_sk and i_item_sk = sc.ss_item_sk order by s_store_name, i_item_desc".to_string()),
        // q67 — rollup over store sales.
        ("q67", "select * from (select i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id, sumsales, rank() over (partition by i_category order by sumsales desc) rk from (select i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id, sum(coalesce(ss_sales_price*ss_quantity, 0)) sumsales from store_sales, date_dim, store, item where ss_sold_date_sk = d_date_sk and ss_item_sk = i_item_sk and ss_store_sk = s_store_sk group by rollup(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id)) dw1) dw2 order by i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id, sumsales, rk".to_string()),
        // q69 — demographics with EXISTS + NOT EXISTS.
        ("q69", "select cd_gender, cd_marital_status, cd_education_status, count(*) cnt1, cd_purchase_estimate, count(*) cnt2, cd_credit_rating, count(*) cnt3 from customer c, customer_address ca, customer_demographics where c.c_current_addr_sk = ca.ca_address_sk and ca_state in ('KY', 'GA', 'NM') and cd_demo_sk = c.c_current_cdemo_sk and exists (select * from store_sales, date_dim where c.c_customer_sk = ss_customer_sk and ss_sold_date_sk = d_date_sk and d_year = 2000) and not exists (select * from web_sales, date_dim where c.c_customer_sk = ws_bill_customer_sk and ws_sold_date_sk = d_date_sk and d_year = 2000) group by cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating order by cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate, cd_credit_rating".to_string()),
        // q70 — store sales rollup by state/county with window.
        ("q70", "select sum(ss_net_profit) as total_sum, s_state, s_county, grouping(s_state)+grouping(s_county) as lochierarchy, rank() over (partition by grouping(s_state)+grouping(s_county), case when grouping(s_county) = 0 then s_state end order by sum(ss_net_profit) desc) as rank_within_parent from store_sales, date_dim d1, store where d1.d_date_sk = ss_sold_date_sk and s_store_sk = ss_store_sk group by rollup(s_state, s_county) order by lochierarchy desc, case when lochierarchy = 0 then s_state end, rank_within_parent".to_string()),
        // q76 — NULL-key sales across channels (UNION ALL).
        ("q76", "select channel, col_name, d_year, d_qoy, i_category, count(*) sales_cnt, sum(ext_sales_price) sales_amt from (select 'store' as channel, 'ss_store_sk' col_name, d_year, d_qoy, i_category, ss_ext_sales_price ext_sales_price from store_sales, item, date_dim where ss_store_sk is null and ss_sold_date_sk = d_date_sk and ss_item_sk = i_item_sk union all select 'web' as channel, 'ws_ship_customer_sk' col_name, d_year, d_qoy, i_category, ws_ext_sales_price ext_sales_price from web_sales, item, date_dim where ws_ship_customer_sk is null and ws_sold_date_sk = d_date_sk and ws_item_sk = i_item_sk union all select 'catalog' as channel, 'cs_ship_addr_sk' col_name, d_year, d_qoy, i_category, cs_ext_sales_price ext_sales_price from catalog_sales, item, date_dim where cs_ship_addr_sk is null and cs_sold_date_sk = d_date_sk and cs_item_sk = i_item_sk) foo group by channel, col_name, d_year, d_qoy, i_category order by channel, col_name, d_year, d_qoy, i_category".to_string()),
        // q79 — store sales ticket summary with demographics.
        ("q79", "select c_last_name, c_first_name, ss_ticket_number, amt, profit from (select ss_ticket_number, ss_customer_sk, sum(ss_coupon_amt) amt, sum(ss_net_profit) profit from store_sales, date_dim, store, household_demographics where ss_sold_date_sk = d_date_sk and ss_store_sk = s_store_sk and ss_hdemo_sk = hd_demo_sk and (hd_dep_count = 6 or hd_vehicle_count > 2) and d_year in (1998, 1999, 2000) group by ss_ticket_number, ss_customer_sk) ms, customer where ss_customer_sk = c_customer_sk order by c_last_name, c_first_name, profit".to_string()),
        // q82 — inventory items by price/manufacturer.
        ("q82", "select i_item_id, i_current_price from item, inventory, date_dim, store_sales where i_current_price between 62 and 92 and inv_item_sk = i_item_sk and d_date_sk = inv_date_sk and i_manufact_id in (100, 101, 102) and inv_quantity_on_hand between 100 and 500 and ss_item_sk = i_item_sk group by i_item_id, i_current_price order by i_item_id".to_string()),
        // q87 — EXCEPT across store/catalog/web sales.
        ("q87", "select count(*) from ((select distinct c_last_name, c_first_name, d_date from store_sales, date_dim, customer where ss_sold_date_sk = d_date_sk and ss_customer_sk = c_customer_sk) except (select distinct c_last_name, c_first_name, d_date from catalog_sales, date_dim, customer where cs_sold_date_sk = d_date_sk and cs_bill_customer_sk = c_customer_sk)) cool_cust".to_string()),
        // q93 — store sales minus returns by reason.
        ("q93", "select ss_customer_sk, sum(act_sales) sumsales from (select ss_item_sk, ss_ticket_number, ss_customer_sk, case when sr_return_quantity is not null then (ss_quantity-sr_return_quantity)*ss_sales_price else (ss_quantity*ss_sales_price) end act_sales from store_sales left outer join store_returns on (sr_item_sk = ss_item_sk and sr_ticket_number = ss_ticket_number), reason where sr_reason_sk = r_reason_sk) t group by ss_customer_sk order by sumsales, ss_customer_sk".to_string()),
        // q97 — FULL OUTER JOIN store-only/catalog-only/store-and-catalog.
        ("q97", "with ssci as (select ss_customer_sk customer_sk, ss_item_sk item_sk from store_sales, date_dim where ss_sold_date_sk = d_date_sk group by ss_customer_sk, ss_item_sk), csci as (select cs_bill_customer_sk customer_sk, cs_item_sk item_sk from catalog_sales, date_dim where cs_sold_date_sk = d_date_sk group by cs_bill_customer_sk, cs_item_sk) select sum(case when ssci.customer_sk is not null and csci.customer_sk is null then 1 else 0 end) store_only, sum(case when ssci.customer_sk is null and csci.customer_sk is not null then 1 else 0 end) catalog_only, sum(case when ssci.customer_sk is not null and csci.customer_sk is not null then 1 else 0 end) store_and_catalog from ssci full outer join csci on (ssci.customer_sk = csci.customer_sk and ssci.item_sk = csci.item_sk)".to_string()),
        // q98 — window revenue ratio (store).
        ("q98", "select i_item_id, i_category, i_class, i_current_price, sum(ss_ext_sales_price) as itemrevenue, sum(ss_ext_sales_price)*100/sum(sum(ss_ext_sales_price)) over (partition by i_class) as revenueratio from store_sales, item, date_dim where ss_item_sk = i_item_sk and i_category in ('Sports', 'Books', 'Home') and ss_sold_date_sk = d_date_sk and d_year = 2000 group by i_item_id, i_category, i_class, i_current_price order by i_category, i_class, i_item_id, revenueratio".to_string()),
        // --- Conformance feature-coverage queries (original 16, kept for ratchet). ---
        ("win_rank", "select i_category, i_brand, sum(ss_ext_sales_price) as rev, rank() over (partition by i_category order by sum(ss_ext_sales_price) desc) as rnk from store_sales, item where ss_item_sk = i_item_sk group by i_category, i_brand order by i_category, rnk".to_string()),
        ("win_running", "select d_date, sum(ss_ext_sales_price) as daily, sum(sum(ss_ext_sales_price)) over (order by d_date rows between unbounded preceding and current row) as running from store_sales, date_dim where ss_sold_date_sk = d_date_sk group by d_date order by d_date".to_string()),
        ("rollup", "select i_category, i_class, sum(ss_net_profit) as profit from store_sales, item where ss_item_sk = i_item_sk group by rollup(i_category, i_class) order by i_category, i_class".to_string()),
        ("grouping_sets", "select i_category, i_brand, sum(ss_quantity) as qty from store_sales, item where ss_item_sk = i_item_sk group by grouping sets ((i_category), (i_brand), ()) order by i_category, i_brand".to_string()),
        ("cube", "select d_year, i_category, sum(ss_ext_sales_price) as rev from store_sales, item, date_dim where ss_item_sk = i_item_sk and ss_sold_date_sk = d_date_sk group by cube(d_year, i_category) order by d_year, i_category".to_string()),
        ("intersect", "select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Electronics' intersect select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Books'".to_string()),
        ("except", "select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Electronics' except select ss_customer_sk from store_sales, item where ss_item_sk = i_item_sk and i_category = 'Books'".to_string()),
        ("cte", "with cat_rev as (select i_category, sum(ss_ext_sales_price) as rev from store_sales, item where ss_item_sk = i_item_sk group by i_category) select i_category, rev from cat_rev where rev > 10 order by rev desc".to_string()),
        ("count_distinct", "select i_category, count(distinct ss_customer_sk) as buyers from store_sales, item where ss_item_sk = i_item_sk group by i_category having count(distinct ss_customer_sk) >= 1 order by buyers desc, i_category".to_string()),
        ("correlated", "select i_item_sk, i_current_price from item where i_current_price > (select avg(i_current_price) from item i2 where i2.i_category = item.i_category) order by i_item_sk".to_string()),
        ("case_agg", "select i_category, sum(case when ss_net_profit > 10 then 1 else 0 end) as hi, sum(case when ss_net_profit <= 10 then 1 else 0 end) as lo from store_sales, item where ss_item_sk = i_item_sk group by i_category order by i_category".to_string()),
    ]
}

// --- Ledger shape ----------------------------------------------------------

#[derive(serde::Serialize)]
struct Methodology {
    engine: &'static str,
    engine_version: &'static str,
    git_sha: Option<String>,
    scale: f64,
    storage_regime: &'static str,
    datagen: &'static str,
    temperature_semantics: &'static str,
    routes: [&'static str; 2],
}

#[derive(serde::Serialize)]
struct LedgerRecord {
    benchmark: String,
    query: String,
    /// Harness sweep label (`native` / `datafusion` / `duckdb`). Serialized as
    /// `sweep`: the flattened snapshot also carries a `route` key (the STAMPED
    /// `(shape_class, backend)` tuple), and duplicate JSON keys made parsers
    /// silently drop this label. `sweep` = which phase ran the query; `route`
    /// = what the router actually stamped/served.
    #[serde(rename = "sweep")]
    route: String,
    temperature: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    rows: usize,
    wall_ms: u128,
    #[serde(flatten)]
    snapshot: IoTraceSnapshot,
}

#[derive(serde::Serialize)]
struct Ledger {
    methodology: Methodology,
    records: Vec<LedgerRecord>,
}

/// Run one query: clear the capture, time the client round-trip, drain the
/// per-query billing snapshot (the observer fires server-side at scope close,
/// so poll briefly for it).
async fn measure(
    client: &Client,
    sql: &str,
) -> Result<(usize, u128, IoTraceSnapshot), (u128, String)> {
    CAPTURE.lock().expect("lock").clear();
    let t0 = Instant::now();
    // Optional per-query wall budget (`TPC_PERF_QUERY_TIMEOUT_MS`, unset/0 =
    // off): a pathological query becomes an ERR record + a server-side cancel
    // instead of wedging the whole sweep behind one Volcano grind. The cancel
    // matters — dropping the query future alone leaves the connection busy and
    // the NEXT query would queue behind the still-running one.
    let budget_ms: u64 = std::env::var("TPC_PERF_QUERY_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let res = if budget_ms == 0 {
        client.simple_query(sql).await
    } else {
        match tokio::time::timeout(Duration::from_millis(budget_ms), client.simple_query(sql)).await
        {
            Ok(r) => r,
            Err(_elapsed) => {
                let _ = client
                    .cancel_token()
                    .cancel_query(tokio_postgres::NoTls)
                    .await;
                return Err((
                    t0.elapsed().as_millis(),
                    format!("harness timeout after {budget_ms} ms (query cancelled)"),
                ));
            }
        }
    };
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

#[allow(clippy::too_many_arguments)]
fn push_record(
    out: &mut Vec<LedgerRecord>,
    benchmark: &str,
    query: &str,
    route: &str,
    temperature: &str,
    result: Result<(usize, u128, IoTraceSnapshot), (u128, String)>,
) {
    let rec = match result {
        Ok((rows, wall_ms, snapshot)) => LedgerRecord {
            benchmark: benchmark.into(),
            query: query.into(),
            route: route.into(),
            temperature: temperature.into(),
            ok: true,
            error: None,
            rows,
            wall_ms,
            snapshot,
        },
        Err((wall_ms, err)) => LedgerRecord {
            benchmark: benchmark.into(),
            query: query.into(),
            route: route.into(),
            temperature: temperature.into(),
            ok: false,
            error: Some(err),
            rows: 0,
            wall_ms,
            snapshot: IoTraceSnapshot::default(),
        },
    };
    // Live progress + crash-safe partial ledger. The harness previously went
    // silent between seeding and the final write, holding every record in
    // memory — a wedged/killed run could neither name the culprit query nor
    // keep any completed measurement. Now each record prints as it lands and
    // appends to `<out>.partial.jsonl`, so a run that never completes still
    // leaves the full trail up to the query it died on.
    match &rec.error {
        None => eprintln!(
            "[{benchmark}] {query} {route}/{temperature} {} ms rows={}",
            rec.wall_ms, rec.rows
        ),
        Some(e) => eprintln!(
            "[{benchmark}] {query} {route}/{temperature} {} ms ERR: {e}",
            rec.wall_ms
        ),
    }
    append_partial_record(&ledger_out_path(), &rec);
    out.push(rec);
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

/// The final ledger path (`TPC_PERF_LEDGER_OUT`, defaulted) — shared by the
/// end-of-run write and the incremental partial sidecar.
fn ledger_out_path() -> String {
    std::env::var("TPC_PERF_LEDGER_OUT")
        .unwrap_or_else(|_| "target/tpc-perf-ledger/ledger.json".to_string())
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

/// ADR-059: query `xcatalog.table_routing` for each table's materialized-parquet
/// `location` (the catalog-introspection `location` column), so the in-process
/// DuckDB engine reads the SAME parquet DataFusion reads (apples-to-apples).
#[cfg(feature = "duckdb")]
async fn query_parquet_locations(client: &Client, table_names: &[&str]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Ok(msgs) = client
        .simple_query("SELECT * FROM xcatalog.table_routing")
        .await
    else {
        return out;
    };
    for m in msgs {
        if let SimpleQueryMessage::Row(row) = m {
            // table_name = col 2 (table_catalog, table_schema, table_name);
            // location = col 12 (the last, added by ADR-059).
            let name = row.get(2).unwrap_or_default().to_string();
            let loc = row.get(12).unwrap_or_default().to_string();
            // Only the benchmark's tables (filter out system/internal tables
            // whose locations may be invalid → would break the DuckDB session).
            if !loc.is_empty() && table_names.contains(&name.as_str()) {
                out.push((name, loc));
            }
        }
    }
    out
}

/// ADR-059: run one query on the in-process DuckDB engine over the materialized
/// parquet, io-traced (compute_ms). Mirrors `measure` (clear CAPTURE → run →
/// drain) but via `execute_sql_with_backend(DuckDbCompat)` in
/// `io_trace::instrument` instead of pgwire.
#[cfg(feature = "duckdb")]
async fn measure_duckdb_inprocess(
    sql: &str,
    parquet_tables: &[(String, String)],
) -> Result<(usize, u128, IoTraceSnapshot), (u128, String)> {
    use proximadb::query::execution::engine::{QueryExecutionContext, execute_sql_with_backend};
    use proximadb::query::table_write_plan::ComputeBackend;
    CAPTURE.lock().expect("lock").clear();
    let ctx = QueryExecutionContext {
        parquet_tables: parquet_tables.to_vec(),
        ..Default::default()
    };
    let t0 = Instant::now();
    // instrument sets the IO_TRACE scope; DuckDbLocalEngine records compute_ms;
    // instrument emits the snapshot → billing observer → CAPTURE.
    let sql_owned = sql.to_string();
    let res = io_trace::instrument(None, "duckdb".to_string(), async move {
        execute_sql_with_backend(ComputeBackend::DuckDbCompat, &sql_owned, ctx).await
    })
    .await;
    let wall_ms = t0.elapsed().as_millis();
    // Drain CAPTURE (same 60×5ms poll as `measure`).
    let mut snap = IoTraceSnapshot::default();
    for _ in 0..60 {
        if let Some(s) = CAPTURE.lock().expect("lock").pop() {
            snap = s;
            break;
        }
        sleep(Duration::from_millis(5)).await;
    }
    match res {
        Ok(r) => Ok((r.rows.len(), wall_ms, snap)),
        Err(e) => Err((wall_ms, e.to_string())),
    }
}

/// DuckDB external baseline (TD-OLAP-4 "external baselines"). When `DUCKDB_BIN`
/// is set, load the SAME synthetic-tpc data (the `schema` CREATE TABLEs + the
/// `inserts` — both DuckDB-compatible standard SQL) into a persistent temp
/// DuckDB file, then time each query through the DuckDB CLI. Records
/// `route:"duckdb"` with no IoTrace (out-of-process — the wall_ms comparison is
/// the signal). **No-op when `DUCKDB_BIN` is unset** (the default — the advisory
/// ledger stays ProximaDB-self). Mirrors `tests/clickbench_ledger_e2e.rs`. No
/// new dependency (uses `std::process` + the operator-provided binary); the
/// result cache (#708) is default-OFF so ProximaDB's latency is already fair.
///
/// Strip a terminal `WITH (…) ` storage-parameter clause from a `CREATE TABLE`
/// DDL. ProximaDB accepts `WITH (cluster_key = '<col>')` (the TD-OLAP-6
/// sort-on-materialize hint); DuckDB's parser rejects it. The clause is always
/// at the end of the DDL and at paren depth 0 (after the column-list `)`), so we
/// locate the first depth-0 `WITH (` and drop from there to the end. A `WITH`
/// inside the column list (depth > 0) is left untouched. DuckDB builds its own
/// zone maps on load, so the cluster key is irrelevant to the wall-time baseline.
#[cfg(not(feature = "duckdb"))]
fn strip_duckdb_incompatible_storage_param(ddl: &str) -> String {
    let lower = ddl.to_ascii_lowercase();
    let Some(rel) = lower.find(" with ") else {
        return ddl.to_string();
    };
    // Only a depth-0 WITH (after the column list closes) is the storage param.
    let before = &ddl[..rel];
    let depth: i32 = before.matches('(').count() as i32 - before.matches(')').count() as i32;
    if depth != 0 {
        return ddl.to_string();
    }
    let after = lower[rel..].trim_start();
    if !after.starts_with("with (") && !after.starts_with("with(") {
        return ddl.to_string();
    }
    before.trim_end().to_string()
}

#[cfg(not(feature = "duckdb"))]
fn run_duckdb_baseline(
    benchmark: &str,
    schema: &[(&str, &str)],
    inserts: &[String],
    queries: &[(&'static str, String)],
    out: &mut Vec<LedgerRecord>,
) {
    let Ok(duckdb) = std::env::var("DUCKDB_BIN") else {
        return; // no binary ⇒ skip (default)
    };
    eprintln!("[{benchmark}] · DuckDB baseline (DUCKDB_BIN={duckdb})");

    // Loader SQL: schema.1 is the full `CREATE TABLE` DDL; inserts are standard
    // `INSERT INTO … VALUES`. DuckDB accepts both — EXCEPT ProximaDB's
    // `WITH (cluster_key = …)` storage param (the TD-OLAP-6 sort-on-materialize
    // hint), which DuckDB's parser rejects ("WITH clause is not supported for
    // tables"). Strip it for the baseline: DuckDB builds its own zone maps on
    // load, so the declared cluster key is irrelevant to the wall-time comparison.
    let mut loader = String::new();
    for (_, ddl) in schema {
        loader.push_str(&strip_duckdb_incompatible_storage_param(ddl));
        loader.push_str(";\n");
    }
    for ins in inserts {
        loader.push_str(ins);
        loader.push_str(";\n");
    }

    // Load once into a persistent temp DuckDB file (per-query spawns read it,
    // so the data isn't reloaded per query).
    let Ok(tmp) = tempfile::tempdir() else {
        return;
    };
    let db_path = tmp.path().join(format!("{benchmark}.duckdb"));

    // Write loader SQL to a file (more reliable than stdin pipe — the pipe can
    // close before DuckDB finishes reading, silently truncating the load).
    let loader_path = tmp.path().join("loader.sql");
    if std::fs::write(&loader_path, &loader).is_err() {
        eprintln!("[{benchmark}] · DuckDB loader write failed");
        return;
    }

    // Load: `duckdb db.duckdb < loader.sql` (stdin redirected from the file).
    let loader_file = match std::fs::File::open(&loader_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let load_output = Command::new(&duckdb)
        .arg(&db_path)
        .stdin(Stdio::from(loader_file))
        .output();
    if let Ok(o) = &load_output
        && !o.status.success()
    {
        let stderr = String::from_utf8_lossy(&o.stderr);
        eprintln!(
            "[{benchmark}] · DuckDB load FAILED: {}",
            stderr.lines().next().unwrap_or("?")
        );
        return; // don't run queries against a failed load
    }

    // Verify: confirm data actually loaded (COUNT on the first table).
    let first_table = schema[0].0;
    let verify = Command::new(&duckdb)
        .arg("-csv")
        .arg(&db_path)
        .arg("-c")
        .arg(format!("SELECT count(*) FROM {first_table}"))
        .output();
    let row_count = verify
        .as_ref()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|l| l.trim().chars().all(|c| c.is_ascii_digit()))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "??".to_string());
    eprintln!("[{benchmark}] · DuckDB loaded: {row_count} rows in {first_table}");

    // Per query: time the CLI round-trip. Rows aren't parsed (DuckDB CLI output
    // shape varies); the comparison signal is wall_ms, recorded honestly with
    // ok/error per the methodology.
    for (id, sql) in queries {
        let t0 = Instant::now();
        let res = Command::new(&duckdb)
            .arg(&db_path)
            .arg("-c")
            .arg(sql)
            .output();
        let wall_ms = t0.elapsed().as_millis();
        let result = match res {
            Ok(o) if o.status.success() => Ok((0, wall_ms, IoTraceSnapshot::default())),
            Ok(o) => Err((
                wall_ms,
                String::from_utf8_lossy(&o.stderr).trim().to_string(),
            )),
            Err(e) => Err((wall_ms, format!("duckdb spawn: {e}"))),
        };
        push_record(out, benchmark, id, "duckdb", "first", result);
    }
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

/// Measure every query on both routes, first + repeat run each.
async fn run_benchmark(
    client: &Client,
    benchmark: &str,
    schema: &[(&str, &str)],
    inserts: Vec<String>,
    queries: Vec<(&'static str, String)>,
    out: &mut Vec<LedgerRecord>,
) {
    let n_inserts = inserts.len();
    seed(client, schema, inserts).await;
    eprintln!(
        "[{benchmark}] seeded ({n_inserts} insert batches across {} tables)",
        schema.len()
    );

    // Native/Volcano route (pre-MATERIALIZE).
    // `TPC_PERF_SKIP_NATIVE=1` skips this route — useful when iterating on the
    // DataFusion (post-MATERIALIZE) or DuckDB routes in isolation, since the
    // row-at-a-time Volcano path is the slow floor (pathological on
    // subquery-heavy shapes like TPC-H Q2). Default: run it (full comparison).
    let skip_native = std::env::var("TPC_PERF_SKIP_NATIVE").is_ok();
    if !skip_native {
        for (id, sql) in &queries {
            for temperature in ["first", "repeat"] {
                let r = measure(client, sql).await;
                push_record(out, benchmark, id, "native", temperature, r);
            }
        }
    } else {
        eprintln!("[{benchmark}] · skipping native/Volcano route (TPC_PERF_SKIP_NATIVE)");
    }

    // Flip to parquet-backed → DataFusion route.
    for (name, _) in schema {
        if let Err(e) = client
            .simple_query(&format!("ALTER TABLE {name} MATERIALIZE"))
            .await
        {
            eprintln!("[{benchmark}] · MATERIALIZE {name}: {}", explain_err(&e));
        }
    }

    // DataFusion route (post-MATERIALIZE).
    for (id, sql) in &queries {
        for temperature in ["first", "repeat"] {
            let r = measure(client, sql).await;
            push_record(out, benchmark, id, "datafusion", temperature, r);
        }
    }

    // ADR-059: DuckDB in-process route (post-MATERIALIZE, same parquet, io-traced).
    // Reads the SAME materialized parquet DataFusion reads (apples-to-apples), via
    // the in-process DuckDbLocalEngine, io-traced (compute_ms). The CLI fallback
    // (after shutdown, below) runs only when the `duckdb` feature is OFF.
    #[cfg(feature = "duckdb")]
    {
        let table_names: Vec<&str> = schema.iter().map(|(n, _)| *n).collect();
        let parquet_tables = query_parquet_locations(client, &table_names).await;
        if parquet_tables.is_empty() {
            eprintln!("[{benchmark}] · DuckDB in-process SKIPPED (no parquet locations)");
        } else {
            eprintln!(
                "[{benchmark}] · DuckDB in-process baseline ({} parquet tables)",
                parquet_tables.len()
            );
            for (id, sql) in &queries {
                let r = measure_duckdb_inprocess(sql, &parquet_tables).await;
                push_record(out, benchmark, id, "duckdb", "first", r);
            }
        }
    }
}

async fn connect(server: &PgServer) -> Client {
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// 8 MB stack — same fix as tpch_pgwire_e2e.rs (planner recursion on deep plans).
#[test]
#[ignore = "perf evidence-ledger harness (TD-OLAP-4) — advisory; run with --ignored --nocapture"]
fn tpc_perf_ledger() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(tpc_perf_ledger_inner());
}

async fn tpc_perf_ledger_inner() {
    let scale_raw = std::env::var("TPC_PERF_SCALE").unwrap_or_else(|_| "0.001".to_string());
    let scale: f64 = scale_raw
        .parse()
        .unwrap_or_else(|_| panic!("TPC_PERF_SCALE must be numeric, got {scale_raw:?}"));
    // Iteration filters (diagnostic / gate-comparison runs): restrict to one
    // benchmark and/or a comma-separated query-id list. Unset ⇒ the full
    // advisory ledger. Used by the TD-OLAP-3 baseline-vs-gate re-measure.
    let benchmark_filter = std::env::var("TPC_PERF_BENCHMARK").ok();
    let preflight = validate_perf_run(
        scale,
        benchmark_filter.as_deref(),
        std::env::var("TPC_PERF_ALLOW_LARGE_SCALE").ok().as_deref(),
        current_cgroup_memory_limit_mib(),
        host_total_memory_mib(),
    )
    .unwrap_or_else(|message| panic!("TPC performance preflight rejected the run: {message}"));
    let query_filter: Option<Vec<String>> = std::env::var("TPC_PERF_QUERIES")
        .ok()
        .map(|s| s.split(',').map(|q| q.trim().to_string()).collect());
    let filtered = benchmark_filter.is_some() || query_filter.is_some();
    let keep_queries = |queries: Vec<(&'static str, String)>| -> Vec<(&'static str, String)> {
        match &query_filter {
            Some(ids) => queries
                .into_iter()
                .filter(|(id, _)| ids.iter().any(|q| q == id))
                .collect(),
            None => queries,
        }
    };
    apply_ledger_env_defaults("TPC_PERF_QUERY_TIMEOUT_MS");
    eprintln!(
        "=== tpc-perf-ledger harness (TPC_PERF_SCALE={scale}, estimated_peak={} MiB, cgroup_limit={:?} MiB) ===",
        preflight.estimated_peak_mib, preflight.cgroup_limit_mib
    );
    // Fresh partial sidecar per run — stale lines from a prior run must not
    // mix into this run's crash-safe trail.
    let _ = std::fs::remove_file(format!("{}.partial.jsonl", ledger_out_path()));

    let mut records = Vec::new();

    // Fresh server per benchmark: TPC-H and TPC-DS both define `customer`.
    if benchmark_filter.as_deref().is_none_or(|b| b == "tpch") {
        let server = PgServer::start().await.expect("server start (tpch)");
        let client = connect(&server).await;
        run_benchmark(
            &client,
            "tpch",
            TPCH_SCHEMA,
            gen_tpch(scale),
            keep_queries(tpch_queries()),
            &mut records,
        )
        .await;
        server.shutdown().await;
        #[cfg(not(feature = "duckdb"))]
        {
            run_duckdb_baseline(
                "tpch",
                TPCH_SCHEMA,
                &gen_tpch(scale),
                &keep_queries(tpch_queries()),
                &mut records,
            );
        }
    }
    if benchmark_filter.as_deref().is_none_or(|b| b == "tpcds") {
        let server = PgServer::start().await.expect("server start (tpcds)");
        let client = connect(&server).await;
        run_benchmark(
            &client,
            "tpcds",
            TPCDS_SCHEMA,
            gen_tpcds(scale),
            keep_queries(tpcds_queries()),
            &mut records,
        )
        .await;
        server.shutdown().await;
        #[cfg(not(feature = "duckdb"))]
        {
            run_duckdb_baseline(
                "tpcds",
                TPCDS_SCHEMA,
                &gen_tpcds(scale),
                &keep_queries(tpcds_queries()),
                &mut records,
            );
        }
    }

    // Console summary: per benchmark × route, pass count + medians. Native and
    // DataFusion report the `repeat` (warm) temperature; DuckDB is out-of-process
    // (loaded once, no warm/cold distinction) so it records only `first` — match
    // that temperature or the DuckDB row silently shows 0/0.
    for benchmark in ["tpch", "tpcds"] {
        for route in ["native", "datafusion", "duckdb"] {
            let temp = if route == "duckdb" { "first" } else { "repeat" };
            let mut rows: Vec<&LedgerRecord> = records
                .iter()
                .filter(|r| r.benchmark == benchmark && r.route == route && r.temperature == temp)
                .collect();
            rows.sort_by_key(|r| r.wall_ms);
            let ok = rows.iter().filter(|r| r.ok).count();
            let median = rows.get(rows.len() / 2).map(|r| r.wall_ms).unwrap_or(0);
            let bytes: u64 = rows.iter().map(|r| r.snapshot.bytes_read).sum();
            eprintln!(
                "[{benchmark}/{route}] ok {ok}/{} · median {temp} wall {median} ms · total bytes_read {bytes}",
                rows.len()
            );
        }
    }

    let ledger = Ledger {
        methodology: Methodology {
            engine: "proximadb",
            engine_version: env!("CARGO_PKG_VERSION"),
            git_sha: std::env::var("GITHUB_SHA").ok(),
            scale,
            storage_regime: "local-tempdir",
            datagen: "synthetic-tpc-shaped (seeded, deterministic, date-clustered facts, non-audited)",
            temperature_semantics: "first vs repeat against a warm process (not cold page cache)",
            routes: [
                "native (Volcano, pre-MATERIALIZE)",
                "datafusion (Parquet, post-MATERIALIZE)",
            ],
        },
        records,
    };

    let out_path = ledger_out_path();
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(dir).expect("create ledger dir");
    }
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&ledger).expect("serialize ledger"),
    )
    .expect("write ledger");
    eprintln!("ledger written: {out_path}");

    // Advisory skeleton: assert only harness integrity, never perf numbers.
    // Skipped for filtered diagnostic runs — the fixed count is only meaningful
    // for the full ledger. DuckDB baseline records (route "duckdb") are excluded
    // from the count — they are an optional external baseline.
    if !filtered {
        // Uniqueness invariant (the real check): one record per
        // (benchmark, query, route, temperature). The DuckDB baseline
        // (DUCKDB_BIN set) adds a variable record count depending on load
        // success, so a fixed magic number no longer holds — check uniqueness,
        // and that the native+datafusion baseline is always fully measured.
        use std::collections::HashSet;
        let mut seen: HashSet<(&str, &str, &str, &str)> = HashSet::new();
        for r in &ledger.records {
            assert!(
                seen.insert((
                    r.benchmark.as_str(),
                    r.query.as_str(),
                    r.route.as_str(),
                    r.temperature.as_str()
                )),
                "duplicate (benchmark, query, route, temperature) record"
            );
        }
        // Dynamic: TPC-DS query count grows with coverage (#855); TPC-H is the
        // fixed 22. Both routes × both temperatures must be measured per query
        // (unless `TPC_PERF_SKIP_NATIVE` drops the Volcano route for focused
        // DF/DuckDB iteration — then only the datafusion route is expected).
        let n_queries = 22 + tpcds_queries().len();
        let n_routes = if std::env::var("TPC_PERF_SKIP_NATIVE").is_ok() {
            1 // datafusion only
        } else {
            2 // native + datafusion
        };
        let baseline = ledger
            .records
            .iter()
            .filter(|r| r.route != "duckdb")
            .count();
        assert_eq!(
            baseline,
            n_queries * n_routes * 2, /* first + repeat */
            "native+datafusion baseline incomplete (expected one record per query x route x temperature)"
        );
    }
}

#[test]
fn perf_preflight_accepts_default_scale_without_cgroup() {
    let result = validate_perf_run(0.001, Some("tpch"), None, None, Some(128 * 1024))
        .expect("the default diagnostic scale must remain easy to run");
    assert!(result.estimated_peak_mib < SAFE_ESTIMATED_PEAK_MIB);
}

#[test]
fn perf_preflight_rejects_sf1_without_explicit_acknowledgement() {
    let error = validate_perf_run(1.0, Some("tpch"), None, Some(64 * 1024), Some(128 * 1024))
        .expect_err("SF1 must fail closed without the exact acknowledgement");
    assert!(error.contains("TPC_PERF_ALLOW_LARGE_SCALE"));
}

#[test]
fn perf_preflight_rejects_large_run_in_unbounded_cgroup() {
    let error = validate_perf_run(
        1.0,
        Some("tpch"),
        Some(LARGE_SCALE_ACK),
        None,
        Some(128 * 1024),
    )
    .expect_err("an acknowledgement alone must never bypass resource isolation");
    assert!(error.contains("finite cgroup memory.max"));
}

#[test]
fn perf_preflight_rejects_limit_that_starves_host() {
    let error = validate_perf_run(
        1.0,
        Some("tpch"),
        Some(LARGE_SCALE_ACK),
        Some(120 * 1024),
        Some(128 * 1024),
    )
    .expect_err("the benchmark cgroup must reserve memory for CI and the host");
    assert!(error.contains("host headroom"));
}

#[test]
fn perf_preflight_accepts_isolated_large_run_with_headroom() {
    let result = validate_perf_run(
        1.0,
        Some("tpch"),
        Some(LARGE_SCALE_ACK),
        Some(64 * 1024),
        Some(128 * 1024),
    )
    .expect("a bounded SF1 run with substantial host headroom is allowed");
    assert!(result.estimated_peak_mib > SAFE_ESTIMATED_PEAK_MIB);
}

#[test]
fn perf_preflight_rejects_non_finite_or_non_positive_scale() {
    for scale in [0.0, -0.1, f64::INFINITY, f64::NAN] {
        assert!(validate_perf_run(scale, None, None, None, None).is_err());
    }
}

/// TD-OLAP-6 regression: MATERIALIZE must publish CLUSTERED row groups.
/// With sort-on-materialize (heuristic key = first DATE column) and a small
/// row-group cap, each row group's date bounds must be tight and the group
/// windows monotone — the property zone-map / runtime-filter pruning needs.
/// (Origin: the v2 evidence diagnostic that root-caused the order-scrambling
/// snapshot scan — every group spanned min=8766..max=9066 before the sort.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn materialize_publishes_clustered_row_groups() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    // Force multiple row groups at this small scale (writer default 65,536).
    // Safe under nextest's process-per-test isolation.
    unsafe { std::env::set_var("PROXIMADB_MATERIALIZE_ROW_GROUP_ROWS", "4096") };

    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;

    let _ = client.simple_query("DROP TABLE IF EXISTS diag").await;
    client
        .simple_query("CREATE TABLE diag (d DATE, x INT)")
        .await
        .expect("create");
    // 20k rows, strictly date-ordered: 2000 rows per month, Jan..Oct 1994.
    let rows: Vec<String> = (0..20_000)
        .map(|i| {
            let month = 1 + i / 2_000;
            let day = 1 + (i % 2_000) % 28;
            format!("(DATE '1994-{month:02}-{day:02}', {i})")
        })
        .collect();
    for sql in chunked_inserts("diag", "d, x", rows, 200) {
        client.simple_query(&sql).await.expect("insert");
    }
    client
        .simple_query("ALTER TABLE diag MATERIALIZE")
        .await
        .expect("materialize");

    // Inspect every published parquet footer: per-row-group bounds for column
    // `d` (Date32 → Int32 days) must form tight, monotonically increasing,
    // near-disjoint windows — NOT the whole-domain span the unsorted snapshot
    // produced before sort-on-materialize.
    let mut found = 0;
    for entry in walk(server._tmp.path()) {
        if entry.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let file = std::fs::File::open(&entry).expect("open parquet");
        let reader = SerializedFileReader::new(file).expect("footer");
        let meta = reader.metadata();
        assert!(
            meta.num_row_groups() >= 4,
            "expected multiple row groups (cap 4096 over ~20k rows), got {}",
            meta.num_row_groups()
        );
        let mut windows: Vec<(i32, i32)> = Vec::new();
        for i in 0..meta.num_row_groups() {
            let rg = meta.row_group(i);
            for col in rg.columns() {
                if col.column_path().string() == "d"
                    && let Some(parquet::file::statistics::Statistics::Int32(s)) = col.statistics()
                {
                    let (min, max) = (
                        *s.min_opt().expect("min stat"),
                        *s.max_opt().expect("max stat"),
                    );
                    eprintln!("  rg{i} d=[{min}, {max}] rows={}", rg.num_rows());
                    windows.push((min, max));
                }
            }
        }
        assert_eq!(windows.len(), meta.num_row_groups(), "d stats per group");
        // Monotone windows (adjacent groups may share the boundary day); the
        // whole-domain-in-every-group failure mode must not recur.
        for pair in windows.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "row-group windows must be sorted/disjoint: {pair:?}"
            );
        }
        let full_domain = (windows.first().unwrap().0, windows.last().unwrap().1);
        assert!(
            windows.iter().filter(|w| **w == full_domain).count() <= 1,
            "clustering collapsed: multiple groups span the full domain {windows:?}"
        );
        found += 1;
    }
    assert!(found > 0, "no parquet files found under the server tempdir");
    server.shutdown().await;
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
    }
    out
}
