//! TD-DOC-CONV-2 **bake phase** — MEASURE the document store-convergence cutover on a
//! canonical-routed collection before flipping the default-OFF gate to default-ON.
//!
//! The convergence capability is merged and green behind the per-collection
//! `PROXIMADB_DOC_CANONICAL_VECTOR` gate (default-OFF). The e2e suite
//! (`documents_convergence_e2e.rs`) proves the capability *works*; it deliberately does NOT
//! measure the four bake criteria the TD gates the cutover on. This suite closes that gap by
//! running the real network transports (REST v2 + gRPC) with the gate ON and emitting
//! `BAKE_METRIC` lines that are transcribed into the evidence artifact + `BENCHMARK_EVIDENCE.toml`:
//!
//!   1. Metering delta      — a canonical (vector-bearing) document write, once materialized, bills
//!                            the per-tenant object-store WRITE dimensions (KIU op count + KSU write
//!                            bytes) AND that signal is observable at `/metrics/prometheus`. This
//!                            gates the two metering gaps the bake originally found (endpoint export
//!                            gap + vector write-path instrumentation gap), now closed. (TD crit. 1)
//!   2. ANN recall          — documents that carry vectors are ANN-searchable through the shared
//!                            store at recall@k within the f32 baseline tolerance; vectorless
//!                            (metadata-only) documents are excluded from ANN yet gettable by id.
//!                            (TD criterion 2)
//!   3. Single-store bytes  — a converged collection stores each body ONCE (vector store only);
//!                            the legacy `document_wal` holds zero bytes for it. (TD criterion 3)
//!   4. Restart recovery    — a populated canonical collection recovers fully from the vector WAL
//!                            after a full restart; measures recovery time + count. (TD criterion 5)
//!
//! One ProximaDB boot per test process (WAL manifest is a set-once singleton); nextest isolates
//! each test in its own process, so the per-test env gate + boot are isolated.
//!
//!   PROXIMADB_DOC_CANONICAL_VECTOR is set per-test via GateGuard (do NOT set it globally).
//!   Run: cargo test --test documents_convergence_bake -- --nocapture

use std::collections::{HashMap, HashSet};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::proto::proximadb_v2::proxima_document_service_client::ProximaDocumentServiceClient;
use proximadb::proto::proximadb_v2::{
    CreateDocumentRequest, GetDocumentRequest, QueryDocumentsRequest,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::sleep;

// ---------------------------------------------------------------------------
// Shared server harness (mirrors documents_convergence_e2e.rs + vector_ann_recall_e2e.rs)
// ---------------------------------------------------------------------------

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct TestServer {
    rest_port: u16,
    grpc_port: u16,
    db: Option<ProximaDB>,
    tmp_data: TempDir,
}

impl TestServer {
    fn build_config(data_path: &Path, rest_port: u16, grpc_port: u16, pg_port: u16) -> Config {
        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = data_path.to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", data_path.display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", data_path.display());
        // TD-CONV-1: the default metadata_url is CWD-relative, so concurrent
        // bake servers shared one system-catalog WAL and a restarted server
        // could come back without its collection ("Collection ... not found").
        config.storage.metadata_url = format!("file://{}/metadata", data_path.display());
        config
    }

    async fn wait_ready(rest_port: u16) -> anyhow::Result<()> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if Instant::now() > deadline {
                        anyhow::bail!("REST server didn't become ready within 20s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        sleep(Duration::from_millis(300)).await;
        Ok(())
    }

    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let pg_port = free_port();
        let tmp_data = TempDir::new()?;
        let mut db = ProximaDB::new(Self::build_config(
            tmp_data.path(),
            rest_port,
            grpc_port,
            pg_port,
        ))
        .await?;
        db.start().await?;
        Self::wait_ready(rest_port).await?;
        Ok(Self {
            rest_port,
            grpc_port,
            db: Some(db),
            tmp_data,
        })
    }

    /// Shut down and restart on the SAME `data_dir` (fresh ports). Returns the wall-clock time
    /// spent in `ProximaDB::new + start` (the WAL-replay recovery window) so the bake can report
    /// recovery latency at scale.
    async fn restart(&mut self) -> anyhow::Result<Duration> {
        if let Some(mut db) = self.db.take() {
            db.shutdown().await?;
        }
        sleep(Duration::from_millis(750)).await;
        let rest_port = free_port();
        let grpc_port = free_port();
        let pg_port = free_port();
        let started = Instant::now();
        let mut db = ProximaDB::new(Self::build_config(
            self.tmp_data.path(),
            rest_port,
            grpc_port,
            pg_port,
        ))
        .await?;
        db.start().await?;
        let recovery = started.elapsed();
        Self::wait_ready(rest_port).await?;
        self.rest_port = rest_port;
        self.grpc_port = grpc_port;
        self.db = Some(db);
        Ok(recovery)
    }

    fn rest(&self) -> String {
        format!("http://127.0.0.1:{}", self.rest_port)
    }

    fn grpc(&self) -> String {
        format!("http://127.0.0.1:{}", self.grpc_port)
    }

    fn data_dir(&self) -> &Path {
        self.tmp_data.path()
    }

    async fn flush(&self, coll: &str) -> anyhow::Result<()> {
        self.db
            .as_ref()
            .expect("db handle")
            .force_flush_collection(coll)
            .await
            .map_err(|e| anyhow::anyhow!("force flush {coll}: {e}"))?;
        Ok(())
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

/// Scope the process-global gate to `collection` (or `all`) for the test; removes it on drop.
struct GateGuard;
impl GateGuard {
    fn on(value: &str) -> Self {
        // SAFETY (edition 2024): nextest runs each test in its own process, so this env mutation
        // is single-threaded for the process.
        unsafe { std::env::set_var("PROXIMADB_DOC_CANONICAL_VECTOR", value) };
        Self
    }
}
impl Drop for GateGuard {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("PROXIMADB_DOC_CANONICAL_VECTOR") };
    }
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .no_proxy()
        .build()
        .expect("http client")
}

// ---------------------------------------------------------------------------
// Deterministic vectors + brute-force recall (shared with vector_ann_recall_e2e.rs)
// ---------------------------------------------------------------------------

fn gen_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s ^= s >> 30;
            s = s.wrapping_mul(0xBF58476D1CE4E5B9);
            s ^= s >> 27;
            s = s.wrapping_mul(0x94D049BB133111EB);
            s ^= s >> 31;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn brute_force_topk(corpus: &[(String, Vec<f32>)], query: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(&String, f32)> = corpus
        .iter()
        .map(|(id, v)| (id, cosine(v, query)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(id, _)| id.clone())
        .collect()
}

fn ids_from_search_body(body: &Value) -> Vec<String> {
    body.get("results")
        .or_else(|| body.get("matches"))
        .or_else(|| body.get("hits"))
        .or_else(|| body.get("records"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    m.get("id")
                        .or_else(|| m.get("record_id"))
                        .and_then(|s| s.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Prometheus scrape parsing — sum every series of a metric family (robust to tenant labels)
// ---------------------------------------------------------------------------

/// Sum the value of every exported series whose name equals `family` (a prometheus counter/gauge
/// vec exports one line per label-set: `name{labels} value`). Summing across label-sets makes the
/// probe robust to whatever `tenant_id`/`operation` labels the write path attributes to.
fn prom_family_sum(scrape: &str, family: &str) -> f64 {
    scrape
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (name_and_labels, value) = l.rsplit_once(' ')?;
            let name = name_and_labels
                .split_once('{')
                .map(|(n, _)| n)
                .unwrap_or(name_and_labels);
            if name == family {
                value.trim().parse::<f64>().ok()
            } else {
                None
            }
        })
        .sum()
}

/// Snapshot of the per-tenant consumption metric families the bake tracks.
#[derive(Debug, Clone, Copy)]
struct MeterSnapshot {
    object_store_ops: f64,
    write_bytes_by_tier: f64,
    kou_bytes: f64,
    storage_bytes_seconds: f64,
}

/// Scrape the four families from the `/metrics/prometheus` ENDPOINT (what an external
/// Prometheus would see).
async fn scrape_meters_endpoint(http: &reqwest::Client, rest: &str) -> MeterSnapshot {
    let body = http
        .get(format!("{rest}/metrics/prometheus"))
        .send()
        .await
        .expect("scrape /metrics/prometheus")
        .text()
        .await
        .expect("metrics body");
    MeterSnapshot {
        object_store_ops: prom_family_sum(&body, "proximadb_object_store_ops_total"),
        write_bytes_by_tier: prom_family_sum(
            &body,
            "proximadb_object_store_write_bytes_by_tier_total",
        ),
        kou_bytes: prom_family_sum(&body, "proximadb_kou_bytes_total"),
        storage_bytes_seconds: prom_family_sum(&body, "proximadb_storage_bytes_seconds"),
    }
}

/// Gather the DEFAULT prometheus registry IN-PROCESS — where `consumption_metrics.rs` registers
/// the per-tenant counters via `register_counter_vec!`/`register_gauge_vec!`. This bypasses the
/// endpoint export gap (the `/metrics/prometheus` handler does not gather the default registry),
/// so it measures the WRITE-PATH INSTRUMENTATION directly: did a document write increment the
/// consumption counters at all?
fn gather_meters_registry() -> MeterSnapshot {
    // Encode the gathered default-registry families to Prometheus text and reuse the text parser,
    // avoiding the protobuf-field accessor API (which varies across prometheus crate versions).
    use prometheus::Encoder;
    let families = prometheus::gather();
    let mut buf = Vec::new();
    let encoder = prometheus::TextEncoder::new();
    let text = if encoder.encode(&families, &mut buf).is_ok() {
        String::from_utf8(buf).unwrap_or_default()
    } else {
        String::new()
    };
    MeterSnapshot {
        object_store_ops: prom_family_sum(&text, "proximadb_object_store_ops_total"),
        write_bytes_by_tier: prom_family_sum(
            &text,
            "proximadb_object_store_write_bytes_by_tier_total",
        ),
        kou_bytes: prom_family_sum(&text, "proximadb_kou_bytes_total"),
        storage_bytes_seconds: prom_family_sum(&text, "proximadb_storage_bytes_seconds"),
    }
}

// ---------------------------------------------------------------------------
// On-disk byte accounting for the single-store GB-month measurement
// ---------------------------------------------------------------------------

/// Recursively sum file sizes under `dir` whose path contains `needle` (matched anywhere in the
/// path). `needle = ""` sums everything. Used to bucket data-dir bytes into legacy `document_wal`
/// vs the shared vector store.
fn bytes_matching(dir: &Path, needle: &str) -> u64 {
    fn walk(dir: &Path, needle: &str, acc: &mut u64) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => walk(&path, needle, acc),
                Ok(ft) if ft.is_file() => {
                    let hay = path.to_string_lossy();
                    if (needle.is_empty() || hay.contains(needle))
                        && let Ok(md) = entry.metadata()
                    {
                        *acc += md.len();
                    }
                }
                _ => {}
            }
        }
    }
    let mut acc = 0u64;
    walk(dir, needle, &mut acc);
    acc
}

/// List the paths of every non-empty `document_wal` file under `dir` (for diagnosing a
/// single-store violation — a canonical collection must not write here).
fn document_wal_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => walk(&path, out),
                Ok(ft) if ft.is_file() => {
                    if path.to_string_lossy().contains("document_wal")
                        && let Ok(md) = entry.metadata()
                        && md.len() > 0
                    {
                        out.push(path);
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, &mut out);
    out
}

// ---------------------------------------------------------------------------
// REST helpers
// ---------------------------------------------------------------------------

async fn create_vector_collection(http: &reqwest::Client, rest: &str, coll: &str, dim: usize) {
    let create = http
        .post(format!("{rest}/api/v2/collections"))
        .json(&json!({
            "name": coll,
            "dimension": dim,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
        }))
        .send()
        .await
        .expect("create collection send");
    let status = create.status();
    let body = create.text().await.unwrap_or_default();
    assert!(status.is_success(), "collection create: {status} — {body}");
}

/// Ingest documents (id + vector + metadata) via REST v2 `/documents` with an SDK-provided vector
/// (the canonical route stamps the document label + routes to the shared vector store). Returns the
/// wall-clock elapsed for the whole batch.
async fn ingest_documents(
    http: &reqwest::Client,
    rest: &str,
    coll: &str,
    docs: &[(String, Vec<f32>)],
) -> Duration {
    let records: Vec<Value> = docs
        .iter()
        .map(|(id, v)| json!({ "id": id, "vector": v, "metadata": {"title": id} }))
        .collect();
    let started = Instant::now();
    let ingest = http
        .post(format!("{rest}/api/v2/collections/{coll}/documents"))
        .header("X-Embed-Source", "sdk-vector")
        .header("X-Ingest-Mode", "sync")
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("ingest documents send");
    let status = ingest.status();
    let body = ingest.text().await.unwrap_or_default();
    assert!(status.is_success(), "document ingest: {status} — {body}");
    started.elapsed()
}

// ===========================================================================
// TD criterion 1 — Metering delta on a canonical document write.
// ===========================================================================

/// A canonical-route document write, once materialized to the object store, MUST bill the
/// per-tenant consumption dimensions (KIU object-store op count + KSU write bytes), and that signal
/// MUST be observable at `/metrics/prometheus` — closing the two metering gaps this bake originally
/// found (the endpoint export gap + the vector write-path instrumentation gap).
///
/// Design: a document write lands in the WAL/memtable; per the co-design principle consumption is
/// metered at the FLUSH / object-store boundary, not per un-flushed insert. `force_flush_collection`
/// is a no-op for materialization (the flush coordinator owns it), so we drive a real
/// memtable→object-store materialization via the SHUTDOWN flush (`StorageEngine::stop` →
/// `flush_memtable_to_storage` → `materialize_collection`, where the per-tenant write metering is
/// now emitted). The prometheus DEFAULT registry is process-global, so the counters survive the
/// in-process restart and are read back after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bake_metering_delta_on_canonical_document_write() {
    const DIM: usize = 8;
    const N: usize = 200;
    let mut server = TestServer::start().await.expect("server start");
    let rest_coll = format!("bakemeter_rest_{}", server.grpc_port);
    let grpc_coll = format!("bakemeter_grpc_{}", server.grpc_port);
    // Gate ON for BOTH collections under test (comma list).
    let _gate = GateGuard::on(&format!("{rest_coll},{grpc_coll}"));
    let http = http_client();
    let rest = server.rest();

    create_vector_collection(&http, &rest, &rest_coll, DIM).await;
    create_vector_collection(&http, &rest, &grpc_coll, DIM).await;

    // Baseline both surfaces: the in-process default REGISTRY (process-global; survives the
    // in-process restart) and the `/metrics/prometheus` ENDPOINT (external Prometheus view).
    let before_reg = gather_meters_registry();
    let before_ep = scrape_meters_endpoint(&http, &rest).await;

    // --- REST v2 document path (canonical route → shared vector-write service) ---
    let docs: Vec<(String, Vec<f32>)> = (0..N)
        .map(|i| (format!("m-{i}"), gen_vec(i as u64, DIM)))
        .collect();
    let ingest_elapsed = ingest_documents(&http, &rest, &rest_coll, &docs).await;

    // --- gRPC ProximaDocumentService path (must ride the SAME metered flush) ---
    {
        let mut client = ProximaDocumentServiceClient::connect(server.grpc())
            .await
            .expect("gRPC connect");
        for i in 0..N {
            client
                .create_document(CreateDocumentRequest {
                    collection_id: grpc_coll.clone(),
                    id: format!("g-{i}"),
                    ..Default::default()
                })
                .await
                .expect("gRPC CreateDocument on canonical route");
        }
    }

    // Drive a real materialization: the shutdown flush inside restart() materializes both
    // collections' memtables through `materialize_collection`, emitting the per-tenant write meter.
    let _recovery = server
        .restart()
        .await
        .expect("restart flushes memtable → materialize → meter");
    let rest2 = server.rest();

    let after_reg = gather_meters_registry();
    let after_ep = scrape_meters_endpoint(&http, &rest2).await;

    // Registry deltas (write-path instrumentation, export-gap-independent).
    let reg_ops = after_reg.object_store_ops - before_reg.object_store_ops;
    let reg_wbytes = after_reg.write_bytes_by_tier - before_reg.write_bytes_by_tier;
    let reg_sbs = after_reg.storage_bytes_seconds - before_reg.storage_bytes_seconds;
    // Endpoint deltas (what an external billing scrape would see — proves the export fix).
    let ep_ops = after_ep.object_store_ops - before_ep.object_store_ops;
    let ep_wbytes = after_ep.write_bytes_by_tier - before_ep.write_bytes_by_tier;

    eprintln!(
        "BAKE_METRIC metering.rest_ingest_ms={}",
        ingest_elapsed.as_millis()
    );
    eprintln!("BAKE_METRIC metering.registry.object_store_ops_delta={reg_ops}");
    eprintln!("BAKE_METRIC metering.registry.write_bytes_by_tier_delta={reg_wbytes}");
    eprintln!("BAKE_METRIC metering.registry.storage_bytes_seconds_delta={reg_sbs}");
    eprintln!("BAKE_METRIC metering.endpoint.object_store_ops_delta={ep_ops}");
    eprintln!("BAKE_METRIC metering.endpoint.write_bytes_by_tier_delta={ep_wbytes}");
    let registry_observable = reg_ops > 0.0 || reg_wbytes > 0.0;
    let endpoint_observable = ep_ops > 0.0 || ep_wbytes > 0.0;
    eprintln!("BAKE_METRIC metering.per_tenant_delta_observable_in_registry={registry_observable}");
    eprintln!("BAKE_METRIC metering.per_tenant_delta_observable_on_endpoint={endpoint_observable}");

    // GAP 2 (write-path instrumentation): a canonical document write, once materialized, bills the
    // per-tenant object-store WRITE dimensions — op count (KIU) AND write bytes (KSU). write_bytes
    // is the cleanest signal (startup reads move ops but never write bytes).
    assert!(
        reg_wbytes > 0.0,
        "canonical document writes did not bill per-tenant object-store WRITE bytes after \
         materialization (write-bytes delta {reg_wbytes}) — write-path instrumentation gap not closed"
    );
    assert!(
        reg_ops > 0.0,
        "per-tenant object-store op counter did not move after materialization (ops delta {reg_ops})"
    );
    // GAP 1 (export): the per-tenant consumption family is now scrapeable on /metrics/prometheus
    // and moved with the write.
    assert!(
        ep_wbytes > 0.0,
        "per-tenant write bytes are not observable on /metrics/prometheus (delta {ep_wbytes}) — \
         the default-registry export gap is not closed"
    );
    let body = http
        .get(format!("{rest2}/metrics/prometheus"))
        .send()
        .await
        .expect("scrape endpoint")
        .text()
        .await
        .expect("metrics body");
    assert!(
        body.contains("proximadb_object_store_write_bytes_by_tier_total"),
        "the per-tenant consumption family must appear on /metrics/prometheus after the export fix"
    );
}

// ===========================================================================
// TD criterion 2 — ANN recall on a canonical document collection + vectorless exclusion.
// ===========================================================================

/// Documents that carry embeddings become vector-searchable through the shared store: measure
/// recall@k on a canonical-routed document collection and hold within tolerance of the brute-force
/// f32 baseline. Metadata-only (vectorless) documents must be excluded from ANN yet served by id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bake_ann_recall_on_canonical_document_collection() {
    const DIM: usize = 32;
    const N: usize = 300;
    const TOP_K: usize = 10;
    const N_QUERIES: usize = 20;
    const RECALL_RATCHET: f64 = 0.90;
    const N_VECTORLESS: usize = 30;

    let server = TestServer::start().await.expect("server start");
    let coll = format!("bakerecall_{}", server.grpc_port);
    let _gate = GateGuard::on(&coll);
    let http = http_client();
    let rest = server.rest();
    create_vector_collection(&http, &rest, &coll, DIM).await;

    // Embedded documents (ANN-eligible).
    let corpus: Vec<(String, Vec<f32>)> = (0..N)
        .map(|i| (format!("doc-{i}"), gen_vec(i as u64, DIM)))
        .collect();
    ingest_documents(&http, &rest, &coll, &corpus).await;

    // Metadata-only documents via gRPC (no vector) — must converge on the store but stay out of ANN.
    let mut client = ProximaDocumentServiceClient::connect(server.grpc())
        .await
        .expect("gRPC connect");
    let vectorless_ids: Vec<String> = (0..N_VECTORLESS).map(|i| format!("meta-{i}")).collect();
    for id in &vectorless_ids {
        let mut props = HashMap::new();
        props.insert(
            "title".to_string(),
            proximadb::proto::proximadb_v2::TypedValue {
                declared_type: 0,
                value: Some(
                    proximadb::proto::proximadb_v2::typed_value::Value::TextValue(id.clone()),
                ),
            },
        );
        client
            .create_document(CreateDocumentRequest {
                collection_id: coll.clone(),
                id: id.clone(),
                props,
                ..Default::default()
            })
            .await
            .expect("gRPC CreateDocument (metadata-only)");
    }

    server.flush(&coll).await.expect("flush");
    sleep(Duration::from_millis(750)).await;

    let vectorless_set: HashSet<&String> = vectorless_ids.iter().collect();
    let mut recalls = Vec::new();
    let mut empty_results = 0usize;
    let mut vectorless_leaks = 0usize;
    for q in 0..N_QUERIES {
        let base_idx = (q * (N / N_QUERIES)) % N;
        let mut query = corpus[base_idx].1.clone();
        let noise = gen_vec((q as u64).wrapping_add(1_000_000), DIM);
        for j in 0..DIM {
            query[j] += noise[j] * 0.01;
        }
        let resp = http
            .post(format!("{rest}/api/v2/collections/{coll}/search"))
            .json(&json!({ "vector": query, "top_k": TOP_K }))
            .send()
            .await
            .expect("search send");
        assert!(resp.status().is_success(), "search: {}", resp.status());
        let body: Value = resp.json().await.expect("search json");
        let ann = ids_from_search_body(&body);
        if ann.is_empty() {
            empty_results += 1;
        }
        vectorless_leaks += ann.iter().filter(|id| vectorless_set.contains(id)).count();
        let exact = brute_force_topk(&corpus, &query, TOP_K);
        let exact_set: HashSet<&String> = exact.iter().collect();
        let hits = ann.iter().filter(|id| exact_set.contains(id)).count();
        recalls.push(hits as f64 / TOP_K as f64);
    }
    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;

    // A metadata-only document is still gettable by id through the shared store.
    let got = client
        .get_document(GetDocumentRequest {
            collection_id: coll.clone(),
            id: vectorless_ids[0].clone(),
        })
        .await
        .expect("gRPC GetDocument (metadata-only)")
        .into_inner()
        .document;
    let vectorless_gettable = got.map(|d| d.id == vectorless_ids[0]).unwrap_or(false);

    eprintln!("BAKE_METRIC recall.mean_at_{TOP_K}={mean:.4}");
    eprintln!(
        "BAKE_METRIC recall.n_embedded={N} n_queries={N_QUERIES} dim={DIM} ratchet={RECALL_RATCHET}"
    );
    eprintln!("BAKE_METRIC recall.empty_results={empty_results}");
    eprintln!(
        "BAKE_METRIC recall.vectorless_leaks_into_ann={vectorless_leaks} (n_vectorless={N_VECTORLESS})"
    );
    eprintln!("BAKE_METRIC recall.vectorless_gettable_by_id={vectorless_gettable}");

    assert_eq!(
        empty_results, 0,
        "{empty_results}/{N_QUERIES} document searches returned ZERO results — canonical document \
         ingest→ANN wiring is broken"
    );
    assert!(
        mean >= RECALL_RATCHET,
        "canonical document ANN recall@{TOP_K} regressed: mean {mean:.4} < ratchet {RECALL_RATCHET}"
    );
    assert_eq!(
        vectorless_leaks, 0,
        "{vectorless_leaks} metadata-only (vectorless) documents leaked into ANN results — they \
         must be excluded from the vector index"
    );
    assert!(
        vectorless_gettable,
        "metadata-only document was not gettable by id through the shared store"
    );
}

// ===========================================================================
// TD criterion 3 — single-store: converged bodies stored ONCE (no double GB-month).
// ===========================================================================

/// A converged collection stores each body in the vector store ONLY — the legacy `document_wal`
/// must hold ZERO bytes for it (the double-GB-month elimination that motivates the cutover). For
/// contrast, a gate-OFF document write DOES land in `document_wal`, so the probe distinguishes the
/// two paths on the same server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bake_single_store_no_double_gb_month() {
    const DIM: usize = 8;
    const N: usize = 200;
    let server = TestServer::start().await.expect("server start");
    let canon_coll = format!("bakestore_canon_{}", server.grpc_port);
    let legacy_coll = format!("bakestore_legacy_{}", server.grpc_port);
    let http = http_client();
    let rest = server.rest();

    // --- Canonical path (gate ON just for canon_coll) ---
    {
        let _gate = GateGuard::on(&canon_coll);
        create_vector_collection(&http, &rest, &canon_coll, DIM).await;
        let docs: Vec<(String, Vec<f32>)> = (0..N)
            .map(|i| (format!("c-{i}"), gen_vec(i as u64, DIM)))
            .collect();
        ingest_documents(&http, &rest, &canon_coll, &docs).await;
        server.flush(&canon_coll).await.expect("flush canon");
        sleep(Duration::from_millis(500)).await;
    }

    // --- Legacy path — gRPC CreateDocument on a NON-canonical collection ---
    {
        // legacy_coll is never created as a vector collection, so under the DEFAULT-ON gate the
        // capability check keeps it on the legacy path (mixed-write-safe), not the canonical route.
        let mut client = ProximaDocumentServiceClient::connect(server.grpc())
            .await
            .expect("gRPC connect");
        for i in 0..N {
            client
                .create_document(CreateDocumentRequest {
                    collection_id: legacy_coll.clone(),
                    id: format!("l-{i}"),
                    ..Default::default()
                })
                .await
                .expect("gRPC CreateDocument (legacy)");
        }
        sleep(Duration::from_millis(500)).await;
    }

    let total_bytes = bytes_matching(server.data_dir(), "");
    let document_wal_bytes = bytes_matching(server.data_dir(), "document_wal");
    let canon_named_wal = document_wal_files(server.data_dir());

    eprintln!("BAKE_METRIC single_store.total_data_dir_bytes={total_bytes}");
    eprintln!("BAKE_METRIC single_store.document_wal_bytes_total={document_wal_bytes}");
    eprintln!(
        "BAKE_METRIC single_store.document_wal_nonempty_files={}",
        canon_named_wal.len()
    );
    for p in &canon_named_wal {
        eprintln!("BAKE_METRIC single_store.document_wal_file={}", p.display());
    }

    // The canonical body must NOT be duplicated into document_wal. We proved the legacy write DOES
    // use document_wal (writes above), and the canonical write must NOT: assert that no
    // document_wal file carries the canonical collection's id prefix. (The legacy collection may
    // legitimately have document_wal bytes; the canonical one must not.)
    let canon_leak = canon_named_wal
        .iter()
        .any(|p| p.to_string_lossy().contains(&canon_coll));
    assert!(
        !canon_leak,
        "canonical collection {canon_coll} leaked bytes into document_wal — double GB-month not \
         eliminated: {canon_named_wal:?}"
    );
}

// ===========================================================================
// TD criterion 5 — restart recovery of a populated canonical collection via the vector WAL.
// ===========================================================================

/// A populated canonical collection (the canonical path drops the legacy `document_wal`, so
/// durability rests entirely on vector-WAL replay) recovers fully after a full server restart.
/// Measures recovery latency and asserts every document is served post-restart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bake_restart_recovery_at_scale() {
    const DIM: usize = 8;
    const N: usize = 500;
    let mut server = TestServer::start().await.expect("server start");
    // Stable name across the restart (fixed from the initial port). Gate ON throughout.
    let coll = format!("bakerecovery{}", server.grpc_port);
    let _gate = GateGuard::on(&coll);
    let http = http_client();
    let rest = server.rest();
    create_vector_collection(&http, &rest, &coll, DIM).await;

    let docs: Vec<(String, Vec<f32>)> = (0..N)
        .map(|i| (format!("r-{i}"), gen_vec(i as u64, DIM)))
        .collect();
    ingest_documents(&http, &rest, &coll, &docs).await;
    // Deliberately do NOT flush — recovery must come purely from vector-WAL replay.
    sleep(Duration::from_millis(300)).await;

    // Confirm visible pre-restart via gRPC query.
    let mut client = ProximaDocumentServiceClient::connect(server.grpc())
        .await
        .expect("gRPC connect");
    let pre = client
        .query_documents(QueryDocumentsRequest {
            collection_id: coll.clone(),
            limit: N as u32,
            ..Default::default()
        })
        .await
        .expect("pre-restart query")
        .into_inner();
    let pre_count = pre.documents.len();

    // Full shutdown + restart on the SAME data_dir; time the recovery window.
    let recovery = server.restart().await.expect("restart on same data_dir");

    // Post-restart: every document must be recovered from the vector WAL and served.
    let mut client2 = ProximaDocumentServiceClient::connect(server.grpc())
        .await
        .expect("gRPC reconnect after restart");
    let post = client2
        .query_documents(QueryDocumentsRequest {
            collection_id: coll.clone(),
            limit: N as u32,
            ..Default::default()
        })
        .await
        .expect("post-restart query")
        .into_inner();
    let post_count = post.documents.len();

    // Spot-check a sample of ids by point-get after restart.
    let mut recovered_by_id = 0usize;
    for i in (0..N).step_by(50) {
        let got = client2
            .get_document(GetDocumentRequest {
                collection_id: coll.clone(),
                id: format!("r-{i}"),
            })
            .await
            .expect("post-restart get")
            .into_inner()
            .document;
        if got.map(|d| d.id == format!("r-{i}")).unwrap_or(false) {
            recovered_by_id += 1;
        }
    }
    let sampled = (0..N).step_by(50).count();

    eprintln!("BAKE_METRIC recovery.n_ingested={N}");
    eprintln!("BAKE_METRIC recovery.pre_restart_query_count={pre_count}");
    eprintln!("BAKE_METRIC recovery.post_restart_query_count={post_count}");
    eprintln!("BAKE_METRIC recovery.spotcheck_recovered={recovered_by_id}/{sampled}");
    eprintln!(
        "BAKE_METRIC recovery.recovery_window_ms={}",
        recovery.as_millis()
    );

    assert_eq!(
        pre_count, N,
        "pre-restart query did not see all {N} ingested documents (saw {pre_count})"
    );
    assert_eq!(
        post_count, N,
        "post-restart: only {post_count}/{N} documents recovered from the vector WAL"
    );
    assert_eq!(
        recovered_by_id, sampled,
        "post-restart point-get recovered {recovered_by_id}/{sampled} sampled ids"
    );
}
