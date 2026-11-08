//! TD-OBJSTORE-2 process-boundary recovery proof.
//!
//! Runs the real server three times against one S3/ADLS object-store prefix,
//! with a brand-new local `server.data_dir` each time:
//!
//! 1. CREATE + INSERT, then SIGKILL (catalog + data WAL must already be durable).
//! 2. Recover catalog + replay WAL, verify search, then SIGINT (flush SST).
//! 3. Recover on another empty local disk and verify cold SST search.
//!
//! Run through `scripts/prove_object_store_durability.sh`; the test is ignored
//! by default because it requires cloud credentials and
//! `PROXIMADB_OBJECT_STORE_URL=s3://...` or `adls://...`.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};

mod support;

fn write_config(path: &Path, data_dir: &Path, root: &str, ports: [u16; 4]) {
    let [rest, grpc, flight, pg] = ports;
    let config = format!(
        r#"
[server]
node_id = "td-objstore-2"
bind_address = "127.0.0.1"
port = {rest}
data_dir = "{}"

[server.tenant]
mode = "single_tenant"
default_tenant = "default"

[storage]
metadata_url = "{root}/metadata"
mmap_enabled = false

[[storage.storage_locations]]
url = "{root}/collections"
weight = 1
tags = ["td-objstore-2", "durable"]

[storage.optimization]
enable_mmap = false

[storage.wal_config]
write_buffer_directory = "{root}/wal"
enable_wal = true
sync_mode = "PerBatch"

[storage.sst_config]
data_directory = "{root}/collections"
mmap_enabled = false

[storage.viper_config]
data_directory = "{root}/collections"

[api]
rest_port = {rest}
grpc_port = {grpc}
arrow_flight_port = {flight}
pg_port = {pg}
unified_mode = false

[monitoring]
metrics_enabled = false
log_level = "info"
"#,
        data_dir.display()
    );
    std::fs::write(path, config).expect("write process config");
}

struct ServerProcess {
    child: Child,
    base_url: String,
}

impl ServerProcess {
    async fn start(root: &str, data_dir: &Path, config_path: &Path) -> anyhow::Result<Self> {
        Self::start_with_env(root, data_dir, config_path, &[]).await
    }

    /// Like [`start`], but injects extra environment variables into the spawned server
    /// process. Used to arm the `PROXIMADB_TEST_RECOVERY_CRASH_POINT` seam for a specific
    /// boot when reproducing recovery crash windows (TD-OBJSTORE-4 S3).
    async fn start_with_env(
        root: &str,
        data_dir: &Path,
        config_path: &Path,
        extra_env: &[(&str, &str)],
    ) -> anyhow::Result<Self> {
        let port_reservation = support::reserve_loopback_ports::<4>()?;
        let ports = port_reservation.ports();
        write_config(config_path, data_dir, root, ports);
        let mut command = Command::new(env!("CARGO_BIN_EXE_proximadb-server"));
        command
            .arg("--config")
            .arg(config_path)
            .env("RUST_LOG", "info")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        for (key, value) in extra_env {
            command.env(key, value);
        }
        drop(port_reservation);
        let child = command.spawn()?;
        let server = Self {
            child,
            base_url: format!("http://127.0.0.1:{}", ports[0]),
        };
        server.wait_ready().await?;
        Ok(server)
    }

    async fn wait_ready(&self) -> anyhow::Result<()> {
        let http = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()?;
        let deadline = Instant::now() + Duration::from_secs(45);
        loop {
            if let Ok(response) = http.get(format!("{}/health", self.base_url)).send().await
                && response.status().is_success()
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                anyhow::bail!("server at {} did not become healthy", self.base_url);
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    fn crash(mut self) -> anyhow::Result<()> {
        self.child.kill()?;
        self.child.wait()?;
        Ok(())
    }

    fn graceful(mut self) -> anyhow::Result<()> {
        let status = Command::new("kill")
            .args(["-INT", &self.child.id().to_string()])
            .status()?;
        if !status.success() {
            anyhow::bail!("failed to send SIGINT to server pid {}", self.child.id());
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait()? {
                if !status.success() {
                    anyhow::bail!("server exited unsuccessfully during graceful stop: {status}");
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                anyhow::bail!("server did not complete graceful shutdown in 30s");
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

async fn create_and_insert(base: &str, collection: &str, engine: &str) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    let response = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": collection,
            "dimension": 8,
            "engine": engine,
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "create failed: {status} {body}");

    let response = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({
            "records": [
                { "id": "durable-0", "vector": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] },
                { "id": "durable-1", "vector": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] }
            ]
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "insert failed: {status} {body}");
    Ok(())
}

async fn assert_recovered(base: &str, collection: &str, phase: &str) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    let response = http
        .get(format!("{base}/api/v2/collections/{collection}"))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(
        status.is_success(),
        "{phase}: catalog did not recover collection: {status} {body}"
    );

    let response = http
        .get(format!(
            "{base}/api/v2/collections/{collection}/records/durable-0"
        ))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(
        status.is_success(),
        "{phase}: recovered point read failed: {status} {body}"
    );
    let record: Value = serde_json::from_str(&body)?;
    anyhow::ensure!(
        record.get("id").and_then(Value::as_str) == Some("durable-0"),
        "{phase}: point read returned the wrong record: {body}"
    );

    let response = http
        .post(format!("{base}/api/v2/collections/{collection}/search"))
        .json(&json!({
            "vector": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "top_k": 2
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(
        status.is_success(),
        "{phase}: search failed: {status} {body}"
    );
    let json: Value = serde_json::from_str(&body)?;
    let ids = json
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| row.get("id").and_then(Value::as_str))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.contains(&"durable-0"),
        "{phase}: durable-0 absent after recovery: {body}"
    );
    Ok(())
}

/// TD-OBJSTORE-1 batch 3: seed a dim-1 "zero-vector" KV collection.
///
/// anvaiops uses `vector: [0.0]` rows as a durable KV store (API keys, tenant
/// registry, billing). The engine skips embedding for zero-vector rows and
/// serves them by id. `rec-zero` is the KV record under test; `rec-ctrl` is a
/// non-zero dim-1 control that isolates the zero-vector-ness as the variable.
async fn create_and_insert_kv(base: &str, collection: &str, engine: &str) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    let response = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": collection,
            "dimension": 1,
            "engine": engine,
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "kv create failed: {status} {body}");

    let response = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({
            "records": [
                { "id": "rec-ctrl", "vector": [0.5], "props": { "kind": "control" } },
                { "id": "rec-zero", "vector": [0.0], "props": { "kind": "api_key", "secret": "sk-anvaiops" } }
            ]
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "kv insert failed: {status} {body}");
    // Both records must exist immediately after seed (matches the TD: the KV
    // record is queryable at seed time; it only 404s after restart).
    for id in ["rec-ctrl", "rec-zero"] {
        let r = http
            .get(format!(
                "{base}/api/v2/collections/{collection}/records/{id}"
            ))
            .send()
            .await?;
        anyhow::ensure!(
            r.status().is_success(),
            "kv seed: {id} not readable at seed time: {}",
            r.status()
        );
    }
    Ok(())
}

/// Assert the dim-1 zero-vector KV record survived restart. The TD symptom is a
/// 404 on `GET .../records/rec-zero` (and `record_count=0`) after the VM restart.
async fn assert_kv_recovered(base: &str, collection: &str, phase: &str) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    // Collection must recover from the catalog.
    let response = http
        .get(format!("{base}/api/v2/collections/{collection}"))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(
        status.is_success(),
        "{phase}: catalog did not recover kv collection: {status} {body}"
    );

    // Each record must be readable by id after restart (the actual TD symptom).
    for id in ["rec-ctrl", "rec-zero"] {
        let response = http
            .get(format!(
                "{base}/api/v2/collections/{collection}/records/{id}"
            ))
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        anyhow::ensure!(
            status.is_success(),
            "{phase}: kv record {id} gone after restart ({status}): {body}"
        );
        let record: Value = serde_json::from_str(&body)?;
        anyhow::ensure!(
            record.get("id").and_then(Value::as_str) == Some(id),
            "{phase}: kv point read returned the wrong record: {body}"
        );
    }
    Ok(())
}

/// TD-OBJSTORE-1 batch 3 regression: a dim-1 zero-vector KV record written to an
/// object-store backend, then lost after a crash+restart. Reproduces on
/// `adls://`/`s3://`/`gs://` (whole-object-overwrite WAL); passes trivially on
/// `file://`. Runs against Azurite/MinIO/fake-gcs via
/// `scripts/run_cloud_emulator_tests.sh` or a real cloud
/// `PROXIMADB_OBJECT_STORE_URL` (TD-OBJSTORE-5 tiers).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn zero_vector_kv_record_survives_restart() -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-1-kv/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("kv_{}", uuid::Uuid::new_v4().simple());

    let vm1 = tmp.path().join("vm1");
    let vm2 = tmp.path().join("vm2");
    for dir in [&vm1, &vm2] {
        std::fs::create_dir_all(dir)?;
    }

    // Phase 1: seed the KV records, then SIGKILL before any flush (WAL-only).
    let first = ServerProcess::start(&root, &vm1, &tmp.path().join("vm1.toml")).await?;
    create_and_insert_kv(&first.base_url, &collection, "sst").await?;
    first.crash()?;

    // Phase 2: restart on a fresh local disk → WAL replay from object store.
    let wal_replay = ServerProcess::start(&root, &vm2, &tmp.path().join("vm2.toml")).await?;
    assert_kv_recovered(&wal_replay.base_url, &collection, "KV WAL replay").await?;
    wal_replay.crash()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL plus S3/ADLS credentials"]
async fn catalog_wal_and_sst_survive_fresh_local_disks() -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://") || base.starts_with("adls://"),
        "test URL must use s3:// or adls:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-2/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let sst_collection = format!("td_objstore_sst_{}", uuid::Uuid::new_v4().simple());
    let helix_collection = format!("td_objstore_helix_{}", uuid::Uuid::new_v4().simple());

    let vm1 = tmp.path().join("vm1");
    let vm2 = tmp.path().join("vm2");
    let vm3 = tmp.path().join("vm3");
    for dir in [&vm1, &vm2, &vm3] {
        std::fs::create_dir_all(dir)?;
    }

    let first = ServerProcess::start(&root, &vm1, &tmp.path().join("vm1.toml")).await?;
    create_and_insert(&first.base_url, &sst_collection, "sst").await?;
    create_and_insert(&first.base_url, &helix_collection, "helix").await?;
    first.crash()?;

    let wal_replay = ServerProcess::start(&root, &vm2, &tmp.path().join("vm2.toml")).await?;
    assert_recovered(&wal_replay.base_url, &sst_collection, "SST WAL replay").await?;
    assert_recovered(&wal_replay.base_url, &helix_collection, "HELIX WAL replay").await?;
    wal_replay.graceful()?; // materializes SST and reclaims flushed WAL

    let cold_sst = ServerProcess::start(&root, &vm3, &tmp.path().join("vm3.toml")).await?;
    assert_recovered(&cold_sst.base_url, &sst_collection, "cold SST").await?;
    assert_recovered(&cold_sst.base_url, &helix_collection, "cold HELIX").await?;
    cold_sst.crash()?;
    Ok(())
}

/// Deterministic splitmix64 (no `rand` dev-dep) — reproducible seed vectors.
fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit_vec(seed: &mut u64, dim: usize) -> Vec<f32> {
    let raw: Vec<f64> = (0..dim)
        .map(|_| (splitmix64(seed) as f64 / u64::MAX as f64) * 2.0 - 1.0)
        .collect();
    let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    raw.iter().map(|x| (x / norm) as f32).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// TD-OBJSTORE-5 S3 (ADR-063 D8 QA tier): cold-read recall ratchet on the object
/// store. Seeds a deterministic corpus, flushes (graceful stop), restarts on a
/// FRESH local disk, and asserts recall@5 vs local brute force. The storage
/// backend must be recall-NEUTRAL: this proves the quantized cascade's ranged
/// reads + footer parse are byte-correct over the real cloud wire API — any
/// recall delta vs the file:// baseline is a read-path bug, not a tuning knob.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn cold_recall_ratchet_survives_restart() -> anyhow::Result<()> {
    const DIM: usize = 8;
    const N: usize = 64;
    const K: usize = 5;
    const PROBES: usize = 8;
    const MIN_RECALL: f32 = 0.9;

    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-5-recall/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("recall_{}", uuid::Uuid::new_v4().simple());
    let vm1 = tmp.path().join("vm1");
    let vm2 = tmp.path().join("vm2");
    for dir in [&vm1, &vm2] {
        std::fs::create_dir_all(dir)?;
    }

    let mut seed = 0x5EED_5EED_5EED_5EEDu64;
    let corpus: Vec<Vec<f32>> = (0..N).map(|_| unit_vec(&mut seed, DIM)).collect();

    // Phase 1: seed the corpus, then GRACEFUL stop (SIGINT flushes the segment) —
    // this ratchet targets the COLD read path, not WAL replay.
    let first = ServerProcess::start(&root, &vm1, &tmp.path().join("vm1.toml")).await?;
    {
        let http = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(30))
            .build()?;
        let response = http
            .post(format!("{}/api/v2/collections", first.base_url))
            .json(&json!({
                "name": collection,
                "dimension": DIM,
                "engine": "sst",
                "distance_metric": "cosine",
                "canonical_embedding_precision": "fp32",
                "enable_proxima_record": false
            }))
            .send()
            .await?;
        anyhow::ensure!(response.status().is_success(), "recall create failed");
        let records: Vec<Value> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| json!({ "id": format!("vec-{i}"), "vector": v }))
            .collect();
        let response = http
            .post(format!(
                "{}/api/v2/collections/{collection}/records/batch",
                first.base_url
            ))
            .json(&json!({ "records": records }))
            .send()
            .await?;
        anyhow::ensure!(response.status().is_success(), "recall insert failed");
    }
    first.graceful()?;

    // Phase 2: fresh local disk → cold read from the object store only.
    let cold = ServerProcess::start(&root, &vm2, &tmp.path().join("vm2.toml")).await?;
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    let mut total_hits = 0usize;
    for p in 0..PROBES {
        // Probe = perturbed copy of every 8th corpus vector (deterministic).
        let base_vec = &corpus[p * (N / PROBES)];
        let mut pseed = 0xABCD_EF01_2345_6789u64 ^ (p as u64);
        let noise = unit_vec(&mut pseed, DIM);
        let probe: Vec<f32> = base_vec
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + 0.05 * n)
            .collect();

        // Local brute-force top-K by cosine.
        let mut scored: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(&probe, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let truth: Vec<String> = scored[..K]
            .iter()
            .map(|(i, _)| format!("vec-{i}"))
            .collect();

        let response = http
            .post(format!(
                "{}/api/v2/collections/{collection}/search",
                cold.base_url
            ))
            .json(&json!({ "vector": probe, "top_k": K }))
            .send()
            .await?;
        anyhow::ensure!(response.status().is_success(), "recall search failed");
        let body: Value = serde_json::from_str(&response.text().await?)?;
        let got: Vec<&str> = body
            .get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|row| row.get("id").and_then(Value::as_str))
            .collect();
        total_hits += truth.iter().filter(|t| got.contains(&t.as_str())).count();
    }
    let recall = total_hits as f32 / (PROBES * K) as f32;
    cold.crash()?;
    anyhow::ensure!(
        recall >= MIN_RECALL,
        "cold recall@{K} on the object store = {recall:.3} < {MIN_RECALL} — \
         backend must be recall-neutral; a delta vs file:// is a read-path bug"
    );
    Ok(())
}

/// TD-OBJSTORE-4 defect-6 redux (task: normal-flush string-strip): a GRACEFUL
/// flush on a cloud base must persist the segment INTO the object store — and
/// must NOT write it to a literal local `adls:...` directory (the
/// URL-as-local-path artifact). #1061 fixed the RECOVERY staging path only;
/// the normal flush retained the false-success class (masked by WAL replay).
/// RED on that state: no segment blob in the store + artifact dir on disk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn graceful_flush_persists_segment_to_object_store() -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-flush-strip/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("flushseg_{}", uuid::Uuid::new_v4().simple());
    let vm1 = tmp.path().join("vm1");
    std::fs::create_dir_all(&vm1)?;

    // The URL-as-local-path artifact appears under the server process CWD.
    let scheme_dir =
        std::path::PathBuf::from(format!("{}:", base.split("://").next().unwrap_or("adls")));
    let artifact_preexisted = scheme_dir.exists();

    // Seed dim-8 records, then SIGINT: the graceful stop flushes the segment.
    let first = ServerProcess::start(&root, &vm1, &tmp.path().join("vm1.toml")).await?;
    create_and_insert(&first.base_url, &collection, "sst").await?;
    first.graceful()?;

    // 1) The flushed segment must exist IN the object store under the
    //    collection data prefix (production FileSystem, same emulator env).
    let factory = std::sync::Arc::new(
        proximadb::storage::persistence::filesystem::FilesystemFactory::create(
            proximadb::storage::persistence::filesystem::FilesystemConfig::default(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("filesystem factory: {e}"))?,
    );
    let collections_prefix = format!("{root}/collections");
    let fs = factory
        .get_filesystem(&collections_prefix)
        .map_err(|e| anyhow::anyhow!("get_filesystem: {e}"))?;
    // LIST the whole collections prefix (flat keyspace) and look for a
    // flushed vector segment object under a .../data/ key.
    let entries = fs
        .list(&collections_prefix)
        .await
        .map_err(|e| anyhow::anyhow!("LIST {collections_prefix}: {e}"))?;
    let segment_blobs: Vec<&str> = entries
        .iter()
        .map(|e| e.url.as_str())
        .filter(|u| u.contains("/data/") && (u.ends_with(".pax") || u.ends_with(".sst")))
        .collect();
    anyhow::ensure!(
        !segment_blobs.is_empty(),
        "graceful flush must persist a segment INTO the object store under \
         {collections_prefix}/**/data/ — found none (URLs: {:?}). The segment \
         was written to a literal local path instead (defect-6 class).",
        entries
            .iter()
            .map(|e| e.url.as_str())
            .take(10)
            .collect::<Vec<_>>()
    );

    // 2) No URL-as-local-path artifact directory may appear.
    anyhow::ensure!(
        artifact_preexisted || !scheme_dir.exists(),
        "flush created a literal local '{}' directory — the staging URL was \
         string-stripped into a local path (defect-6 class)",
        scheme_dir.display()
    );
    Ok(())
}

/// Cloud-compaction e2e (TD-OBJSTORE-4 staged-I/O follow-up; QA tier): armed
/// per-collection compaction (WLP-2 tags) on an object-store base must merge
/// flushed segments through the staged write — a false-success compaction here
/// is the data-loss shape that deletes the INPUT segments (the only copies).
/// Three boots: seed+flush ×2 (the second shutdown flush trips l0_threshold:2
/// and ENQUEUES the armed compaction — whether it completes before process
/// exit is a race this test does not depend on), then a fresh-disk boot
/// proving every record from BOTH batches is still readable whether or not the
/// compaction ran. Also asserts no URL-as-local-path artifact.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn armed_compaction_on_object_store_preserves_all_records() -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-compaction/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("compact_{}", uuid::Uuid::new_v4().simple());
    let dirs: Vec<_> = (0..3).map(|i| tmp.path().join(format!("vm{i}"))).collect();
    for d in &dirs {
        std::fs::create_dir_all(d)?;
    }
    let scheme_dir =
        std::path::PathBuf::from(format!("{}:", base.split("://").next().unwrap_or("adls")));
    let artifact_preexisted = scheme_dir.exists();

    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;

    // Boot 1: create with ARMED compaction (per-collection WLP-2 override) and
    // a low L0 threshold, seed batch A, graceful stop (flush #1).
    let b1 = ServerProcess::start(&root, &dirs[0], &tmp.path().join("vm0.toml")).await?;
    let response = http
        .post(format!("{}/api/v2/collections", b1.base_url))
        .json(&json!({
            "name": collection,
            "dimension": 8,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false,
            "tags": ["compaction:on", "l0_threshold:2"]
        }))
        .send()
        .await?;
    anyhow::ensure!(response.status().is_success(), "compaction create failed");
    let batch_a: Vec<Value> = (0..8)
        .map(|i| json!({ "id": format!("a-{i}"), "vector": vec![0.1_f32 * (i as f32 + 1.0); 8] }))
        .collect();
    let response = http
        .post(format!(
            "{}/api/v2/collections/{collection}/records/batch",
            b1.base_url
        ))
        .json(&json!({ "records": batch_a }))
        .send()
        .await?;
    anyhow::ensure!(response.status().is_success(), "batch A insert failed");
    b1.graceful()?;

    // Boot 2: seed batch B, graceful stop (flush #2 → l0_threshold:2 trips →
    // the armed compaction is ENQUEUED on the flush path; completion before
    // process exit is not guaranteed — the assertions below hold either way).
    let b2 = ServerProcess::start(&root, &dirs[1], &tmp.path().join("vm1.toml")).await?;
    let batch_b: Vec<Value> = (0..8)
        .map(|i| json!({ "id": format!("b-{i}"), "vector": vec![0.05_f32 * (i as f32 + 1.0); 8] }))
        .collect();
    let response = http
        .post(format!(
            "{}/api/v2/collections/{collection}/records/batch",
            b2.base_url
        ))
        .json(&json!({ "records": batch_b }))
        .send()
        .await?;
    anyhow::ensure!(response.status().is_success(), "batch B insert failed");
    b2.graceful()?;

    // Boot 3: fresh disk — every record from BOTH batches must be readable
    // (compacted or not, correctness holds; a false-success compaction that
    // deleted its inputs would lose batch A here).
    let b3 = ServerProcess::start(&root, &dirs[2], &tmp.path().join("vm2.toml")).await?;
    for id in (0..8)
        .map(|i| format!("a-{i}"))
        .chain((0..8).map(|i| format!("b-{i}")))
    {
        let response = http
            .get(format!(
                "{}/api/v2/collections/{collection}/records/{id}",
                b3.base_url
            ))
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "record {id} unreadable after armed compaction + restart ({})",
            response.status()
        );
    }
    b3.crash()?;

    anyhow::ensure!(
        artifact_preexisted || !scheme_dir.exists(),
        "compaction created a literal local '{}' directory (URL-as-local-path)",
        scheme_dir.display()
    );
    Ok(())
}

/// TD-TIER-1 SC5: tier-mix at rest — the SERVER write path, not a probe PUT.
///
/// Creates a collection with `default_access_tier: "cool"`, drives two
/// flush-at-shutdown cycles, then a third boot runs the operator COMPACT
/// (synchronous — it awaits quiescence, so the L1 publish is deterministic
/// rather than riding the shutdown enqueue race) and seeds one more L0 that
/// stays un-compacted. The final at-rest set is at least one tiered L1 (the
/// collection's tier — tiered objects are COMPACTION output only) beside at
/// least one untiered L0 (the account default: L0s are compacted away within
/// minutes, so tiering them would only pay the cold tier's minimum duration).
///
/// What is proven in-process: the plumbing end-to-end (create → admission →
/// flush plan → `enqueue_due_compaction` → tiered publish) produced an L1
/// object and kept an L0 beside it, and every record survives restarts.
/// What is NOT provable in-process: the tier VALUE on the objects —
/// `object_store` 0.13 cannot surface a tier on read and our `FileMetadata`
/// `storage_class` is write-side only. When
/// `$PROXIMADB_TIER_ROUNDTRIP_MANIFEST` is set (set by
/// `scripts/run_cloud_emulator_tests.sh`), the test writes the L0/L1 object
/// inventories there for the out-of-band az/aws read-backs (blobTier=Cool /
/// StorageClass=STANDARD_IA on every L1, account default on every L0).
/// Segment names carry their level (`L0_…`, `L1_…` — `FilenameCodec`), so the
/// CLI read-back classifies without another surface.
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO/LocalStack, gs://fake-gcs, or real cloud)"]
#[tokio::test]
async fn compaction_output_carries_the_collection_tier_end_to_end() -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-tier1-tier-roundtrip/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("tiermix_{}", uuid::Uuid::new_v4().simple());
    let dirs: Vec<_> = (0..3).map(|i| tmp.path().join(format!("vm{i}"))).collect();
    for d in &dirs {
        std::fs::create_dir_all(d)?;
    }
    let scheme_dir =
        std::path::PathBuf::from(format!("{}:", base.split("://").next().unwrap_or("adls")));
    let artifact_preexisted = scheme_dir.exists();

    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;

    async fn seed_batch(
        http: &Client,
        base_url: &str,
        collection: &str,
        prefix: &str,
    ) -> anyhow::Result<()> {
        let records: Vec<Value> = (0..8)
            .map(|i| {
                json!({ "id": format!("{prefix}-{i}"), "vector": vec![0.1_f32 * (i as f32 + 1.0); 8] })
            })
            .collect();
        let response = http
            .post(format!(
                "{base_url}/api/v2/collections/{collection}/records/batch"
            ))
            .json(&json!({ "records": records }))
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "batch {prefix} insert failed ({})",
            response.status()
        );
        Ok(())
    }
    async fn read_all(http: &Client, base_url: &str, collection: &str) -> anyhow::Result<()> {
        for prefix in ["a", "b", "c"] {
            for i in 0..8 {
                let id = format!("{prefix}-{i}");
                let response = http
                    .get(format!(
                        "{base_url}/api/v2/collections/{collection}/records/{id}"
                    ))
                    .send()
                    .await?;
                anyhow::ensure!(
                    response.status().is_success(),
                    "record {id} unreadable ({})",
                    response.status()
                );
            }
        }
        Ok(())
    }

    // Boot 1: create with an explicit tier + ARMED compaction; flush #1.
    let b1 = ServerProcess::start(&root, &dirs[0], &tmp.path().join("vm0.toml")).await?;
    let response = http
        .post(format!("{}/api/v2/collections", b1.base_url))
        .json(&json!({
            "name": collection,
            "dimension": 8,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false,
            "default_access_tier": "cool",
            "tags": ["compaction:on", "l0_threshold:2"]
        }))
        .send()
        .await?;
    anyhow::ensure!(
        response.status().is_success(),
        "tiered create failed ({}) — admission must accept cool on a tiered object store",
        response.status()
    );
    seed_batch(&http, &b1.base_url, &collection, "a").await?;
    b1.graceful()?;

    // Boot 2: flush #2 (shutdown) trips l0_threshold:2 and enqueues the armed
    // compaction. Shutdown does NOT drain the compaction queue (Compaction::stop
    // drops still-queued tasks), so whether this pass completes before exit is a
    // race this test must not depend on — the next boot compacts explicitly.
    let b2 = ServerProcess::start(&root, &dirs[1], &tmp.path().join("vm1.toml")).await?;
    seed_batch(&http, &b2.base_url, &collection, "b").await?;
    b2.graceful()?;

    // Boot 3: the operator COMPACT (TD-COMPACT-14, synchronous — it awaits
    // quiescence) deterministically publishes the L1 from whatever L0s exist
    // (two if boot-2's raced shutdown task was dropped; none due if it ran —
    // either way at least one L1 exists afterward). Then one more flush leaves
    // a single L0 below the threshold at the account default, and all records
    // must survive compaction and restarts.
    let b3 = ServerProcess::start(&root, &dirs[2], &tmp.path().join("vm2.toml")).await?;
    let response = http
        .post(format!(
            "{}/api/v2/collections/{collection}/compact",
            b3.base_url
        ))
        .json(&json!({ "full": false }))
        .send()
        .await?;
    anyhow::ensure!(
        response.status().is_success(),
        "operator compact failed ({})",
        response.status()
    );
    seed_batch(&http, &b3.base_url, &collection, "c").await?;
    read_all(&http, &b3.base_url, &collection).await?;
    b3.graceful()?;

    // Structural tier-mix proof: at rest there is at least one compacted (L1+)
    // object AND at least one un-compacted L0 — the mix the out-of-band
    // read-backs assert per class.
    let blobs = list_segment_blobs(&root).await?;
    let classify = |url: &str| -> Option<bool> {
        // Some(true) = compacted (L1+), Some(false) = L0, None = unrecognized.
        let name = url.rsplit('/').next().unwrap_or(url);
        if name.starts_with("L0_") {
            Some(false)
        } else if name
            .strip_prefix('L')
            .and_then(|rest| rest.split_once('_'))
            .is_some_and(|(digits, _)| {
                !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
            })
        {
            Some(true)
        } else {
            None
        }
    };
    let mut l0_objects = Vec::new();
    let mut l1_objects = Vec::new();
    for url in &blobs {
        match classify(url) {
            Some(false) => l0_objects.push(url.clone()),
            Some(true) => l1_objects.push(url.clone()),
            None => anyhow::bail!("unclassifiable segment object under the test root: {url}"),
        }
    }
    anyhow::ensure!(
        !l1_objects.is_empty(),
        "no compacted (L1+) object exists under {root} — the synchronous operator compact did not publish"
    );
    anyhow::ensure!(
        !l0_objects.is_empty(),
        "no un-compacted L0 object exists under {root} — the untiered half of the tier mix is missing"
    );

    // Hand the inventory to the out-of-band tier read-back (az/aws CLI).
    if let Ok(manifest_path) = std::env::var("PROXIMADB_TIER_ROUNDTRIP_MANIFEST") {
        std::fs::write(
            &manifest_path,
            serde_json::json!({
                "root": root,
                "collection": collection,
                "l0": l0_objects,
                "l1": l1_objects,
            })
            .to_string(),
        )?;
    }

    anyhow::ensure!(
        artifact_preexisted || !scheme_dir.exists(),
        "test created a literal local '{}' directory (URL-as-local-path)",
        scheme_dir.display()
    );
    Ok(())
}

/// LIST the collections prefix (flat keyspace) and return the materialized vector
/// segment blobs (`.pax`/`.sst` under a `.../data/` key). With a unique per-test
/// `root`, every such blob belongs to this test's single collection, so the count
/// is a structural duplicate detector independent of any read-path de-duplication.
async fn list_segment_blobs(root: &str) -> anyhow::Result<Vec<String>> {
    let factory = std::sync::Arc::new(
        proximadb::storage::persistence::filesystem::FilesystemFactory::create(
            proximadb::storage::persistence::filesystem::FilesystemConfig::default(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("filesystem factory: {e}"))?,
    );
    let collections_prefix = format!("{root}/collections");
    let fs = factory
        .get_filesystem(&collections_prefix)
        .map_err(|e| anyhow::anyhow!("get_filesystem: {e}"))?;
    let entries = fs
        .list(&collections_prefix)
        .await
        .map_err(|e| anyhow::anyhow!("LIST {collections_prefix}: {e}"))?;
    Ok(entries
        .into_iter()
        .map(|e| e.url)
        .filter(|u| u.contains("/data/") && (u.ends_with(".pax") || u.ends_with(".sst")))
        .collect())
}

/// TD-OBJSTORE-4 S3 double-replay crash-window idempotency. Reproduces a crash in
/// the post-materialization recovery-retirement sequence (via the
/// `PROXIMADB_TEST_RECOVERY_CRASH_POINT` seam), then a clean restart over the same
/// object store, and proves that re-processing the still-present WAL is idempotent:
/// the data survives and **exactly one** materialized segment exists (no duplicate
/// segment, no duplicate records).
///
/// Two crash windows, two idempotency mechanisms, both must hold:
/// * `after_materialize` (W1): segment committed but manifest not yet flushed → the
///   restart re-replays and hits `write_if_absent` → `AlreadyExists` (the ADR-063
///   commit-record).
/// * `after_mark_flushed` (W2): manifest flushed but WAL not yet deleted → the
///   restart skips the batch via the durable skip-list and retires the stray WAL.
async fn run_double_replay(crash_point: &str) -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-double-replay/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("dblreplay_{}", uuid::Uuid::new_v4().simple());
    let dirs: Vec<_> = (0..3).map(|i| tmp.path().join(format!("vm{i}"))).collect();
    for d in &dirs {
        std::fs::create_dir_all(d)?;
    }

    // Boot 1: seed dim-8 records, then SIGKILL before any flush (WAL-only durability).
    let b1 = ServerProcess::start(&root, &dirs[0], &tmp.path().join("vm0.toml")).await?;
    create_and_insert(&b1.base_url, &collection, "sst").await?;
    b1.crash()?;

    // Boot 2: startup recovery materializes the segment, then the seam stops at the
    // crash point — leaving the WAL present. This is the exact crash-window state.
    let b2 = ServerProcess::start_with_env(
        &root,
        &dirs[1],
        &tmp.path().join("vm1.toml"),
        &[("PROXIMADB_TEST_RECOVERY_CRASH_POINT", crash_point)],
    )
    .await?;
    assert_recovered(&b2.base_url, &collection, &format!("{crash_point} boot 2")).await?;
    let after_boot2 = list_segment_blobs(&root).await?;
    anyhow::ensure!(
        after_boot2.len() == 1,
        "{crash_point}: recovery must materialize exactly one segment, got {after_boot2:?}"
    );
    b2.crash()?;

    // Boot 3: clean restart re-processes the still-present WAL. Re-materialization
    // must be idempotent — no duplicate segment, no duplicate records.
    let b3 = ServerProcess::start(&root, &dirs[2], &tmp.path().join("vm2.toml")).await?;
    assert_recovered(&b3.base_url, &collection, &format!("{crash_point} boot 3")).await?;
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    for id in ["durable-0", "durable-1"] {
        let r = http
            .get(format!(
                "{}/api/v2/collections/{collection}/records/{id}",
                b3.base_url
            ))
            .send()
            .await?;
        anyhow::ensure!(
            r.status().is_success(),
            "{crash_point}: {id} not readable after double replay ({})",
            r.status()
        );
    }
    // Core idempotency invariant: STILL exactly one materialized segment — the
    // re-replay hit AlreadyExists / the durable skip-list, not a duplicate write.
    let after_boot3 = list_segment_blobs(&root).await?;
    anyhow::ensure!(
        after_boot3.len() == 1,
        "{crash_point}: double replay created a DUPLICATE segment — expected 1, got {after_boot3:?}"
    );
    b3.crash()?;
    Ok(())
}

/// W1: crash after the `write_if_absent` segment commit, before manifest mark-flushed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn double_replay_after_materialize_is_idempotent() -> anyhow::Result<()> {
    run_double_replay("after_materialize").await
}

/// W2: crash after manifest mark-flushed, before WAL retirement.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn double_replay_after_mark_flushed_is_idempotent() -> anyhow::Result<()> {
    run_double_replay("after_mark_flushed").await
}

/// TD-OBJSTORE-4 S3 recovery-compaction-suppression: a recovery flush that
/// materializes an orphan WAL range with per-collection compaction ARMED must NOT
/// compact inline until the WAL is retired (`suppress_compaction_until_wal_retired`,
/// set in `recovery_manager.rs`, honored at `sst/flush/mod.rs`). Otherwise inline
/// compaction could consume the deterministic `L0_recovery` commit-record segment
/// before WAL retirement — and a crash there would re-replay and duplicate records.
///
/// Boot 1 flushes batch A into a normal segment (WAL A retired, one L0 segment). Boot
/// 2 seeds batch B WAL-only and SIGKILLs. Boot 3 startup recovery materializes batch B
/// into an `L0_recovery` segment; with two L0 segments now present and `l0_threshold:2`
/// armed, compaction WOULD run inline — suppression must instead leave the `L0_recovery`
/// segment intact and lose no records from either batch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires PROXIMADB_OBJECT_STORE_URL (adls://Azurite, s3://MinIO, gs://fake-gcs, or real cloud)"]
async fn recovery_suppresses_compaction_until_wal_retired() -> anyhow::Result<()> {
    let base = std::env::var("PROXIMADB_OBJECT_STORE_URL")?;
    anyhow::ensure!(
        base.starts_with("s3://")
            || base.starts_with("adls://")
            || base.starts_with("az://")
            || base.starts_with("gs://"),
        "test URL must use s3://, adls://, az:// or gs:// (got {base})"
    );
    let root = format!(
        "{}/td-objstore-recovery-compaction/{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = tempfile::tempdir()?;
    let collection = format!("recomp_{}", uuid::Uuid::new_v4().simple());
    let dirs: Vec<_> = (0..3).map(|i| tmp.path().join(format!("vm{i}"))).collect();
    for d in &dirs {
        std::fs::create_dir_all(d)?;
    }
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;

    // Boot 1: create with ARMED compaction (l0_threshold:2), seed batch A, graceful
    // flush — one L0 segment, WAL A retired.
    let b1 = ServerProcess::start(&root, &dirs[0], &tmp.path().join("vm0.toml")).await?;
    let response = http
        .post(format!("{}/api/v2/collections", b1.base_url))
        .json(&json!({
            "name": collection,
            "dimension": 8,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false,
            "tags": ["compaction:on", "l0_threshold:2"]
        }))
        .send()
        .await?;
    anyhow::ensure!(response.status().is_success(), "compaction create failed");
    let batch_a: Vec<Value> = (0..8)
        .map(|i| json!({ "id": format!("a-{i}"), "vector": vec![0.1_f32 * (i as f32 + 1.0); 8] }))
        .collect();
    let response = http
        .post(format!(
            "{}/api/v2/collections/{collection}/records/batch",
            b1.base_url
        ))
        .json(&json!({ "records": batch_a }))
        .send()
        .await?;
    anyhow::ensure!(response.status().is_success(), "batch A insert failed");
    b1.graceful()?;

    // Boot 2: seed batch B WAL-only, SIGKILL before any flush.
    let b2 = ServerProcess::start(&root, &dirs[1], &tmp.path().join("vm1.toml")).await?;
    let batch_b: Vec<Value> = (0..8)
        .map(|i| json!({ "id": format!("b-{i}"), "vector": vec![0.05_f32 * (i as f32 + 1.0); 8] }))
        .collect();
    let response = http
        .post(format!(
            "{}/api/v2/collections/{collection}/records/batch",
            b2.base_url
        ))
        .json(&json!({ "records": batch_b }))
        .send()
        .await?;
    anyhow::ensure!(response.status().is_success(), "batch B insert failed");
    b2.crash()?;

    // Boot 3: startup recovery materializes batch B into an `L0_recovery` segment with
    // compaction armed. Suppression must keep that segment and every record.
    let b3 = ServerProcess::start(&root, &dirs[2], &tmp.path().join("vm2.toml")).await?;
    for id in (0..8)
        .map(|i| format!("a-{i}"))
        .chain((0..8).map(|i| format!("b-{i}")))
    {
        let response = http
            .get(format!(
                "{}/api/v2/collections/{collection}/records/{id}",
                b3.base_url
            ))
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "record {id} unreadable after recovery with armed compaction ({})",
            response.status()
        );
    }
    // The deterministic recovery segment must still be present — inline compaction was
    // suppressed and did not consume it before WAL retirement.
    let segments = list_segment_blobs(&root).await?;
    anyhow::ensure!(
        segments.iter().any(|u| u.contains("L0_recovery")),
        "recovery segment was consumed by inline compaction — suppression failed \
         (segments: {segments:?})"
    );
    b3.crash()?;
    Ok(())
}
