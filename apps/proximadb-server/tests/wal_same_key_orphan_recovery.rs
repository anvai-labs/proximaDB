//! TD-OBJSTORE-4 — same-key orphan ordering gate (the one remaining adversarial
//! recovery gate after #1140 closed the compaction-suppression + double-replay
//! gates).
//!
//! Proves the invariant at
//! `src/storage/persistence/write_ahead_log/recovery_manager.rs:586-600` —
//! "_latest mutation wins in durable token order_": the recovery replay iterates
//! WAL objects sorted by token epoch/sequence and `latest_by_oid` keeps the last
//! per OID. Concretely: two same-key records (old V1 + newer V2) written then
//! SIGKILLed before the async manifest pointer flush land as **unmanifested
//! orphans**; on restart the LIST-authority recovery discovers both via LIST
//! (not the manifest) and resolves to V2 — the orphan is not lost, and the newer
//! mutation prevails.
//!
//! Uses a local `file://` store (the live ADLS/S3/GCS proof is TD-OBJSTORE-2;
//! the recovery LIST-authority + same-key resolution code path is identical).
//! The scenario-specific `ServerProcess` harness stays local because Rust
//! integration tests are separate crates; loopback-port reservation is shared
//! through `tests/support`.
//!
//! Run: `cargo nextest run --test wal_same_key_orphan_recovery -- --ignored`

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};

mod support;

/// Local-`file://` config (no cloud creds): the WAL + SST persist on the same
/// volume across restarts, so the stranded orphan WAL batches survive the crash.
fn write_config(path: &Path, data_dir: &Path, root: &str, ports: [u16; 4]) {
    let [rest, grpc, flight, pg] = ports;
    let config = format!(
        r#"
[server]
node_id = "td-objstore-4-same-key"
bind_address = "127.0.0.1"
port = {rest}
data_dir = "{data_dir}"

[server.tenant]
mode = "single_tenant"
default_tenant = "default"

[storage]
metadata_url = "{root}/metadata"
mmap_enabled = false

[[storage.storage_locations]]
url = "{root}/collections"
weight = 1
tags = ["same-key-gate"]

[storage.optimization]
enable_mmap = false

[storage.wal_config]
write_buffer_directory = "{root}/wal"
enable_wal = true
sync_mode = "PerBatch"

[storage.sst_config]
data_directory = "{root}/collections"
mmap_enabled = false

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
        data_dir = data_dir.display()
    );
    std::fs::write(path, config).expect("write process config");
}

struct ServerProcess {
    child: Child,
    base_url: String,
}

impl ServerProcess {
    async fn start(root: &str, data_dir: &Path, config_path: &Path) -> anyhow::Result<Self> {
        let port_reservation = support::reserve_loopback_ports::<4>()?;
        let ports = port_reservation.ports();
        write_config(config_path, data_dir, root, ports);
        let mut command = Command::new(env!("CARGO_BIN_EXE_proximadb-server"));
        command
            .arg("--config")
            .arg(config_path)
            .env("RUST_LOG", "warn")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
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
        let deadline = Instant::now() + Duration::from_secs(60);
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

    /// SIGKILL — strands unflushed WAL batches as orphans (the manifest pointer
    /// append is async/batched at ~100ms; the PerBatch-fsynced `.bcwal` data
    /// survives the kill, the manifest entry does not).
    fn crash(mut self) -> anyhow::Result<()> {
        self.child.kill()?;
        self.child.wait()?;
        Ok(())
    }
}

async fn http_client() -> anyhow::Result<Client> {
    Ok(Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?)
}

async fn create_collection(base: &str, collection: &str, dim: usize) -> anyhow::Result<()> {
    let http = http_client().await?;
    let response = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": collection,
            "dimension": dim,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    // 409 (already exists) is fine — the collection may persist across boots.
    anyhow::ensure!(
        status.is_success() || status.as_u16() == 409,
        "create failed: {status} {body}"
    );
    Ok(())
}

async fn insert_vec(
    base: &str,
    collection: &str,
    id: &str,
    vector: Vec<f32>,
) -> anyhow::Result<()> {
    let http = http_client().await?;
    let response = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({ "records": [ { "id": id, "vector": vector } ] }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "insert failed: {status} {body}");
    Ok(())
}

/// GET a record by id; return its vector (parsed as f64). `None` if the record
/// is absent (recovery did not surface it).
async fn get_vec(base: &str, collection: &str, id: &str) -> anyhow::Result<Option<Vec<f64>>> {
    let http = http_client().await?;
    let response = http
        .get(format!(
            "{base}/api/v2/collections/{collection}/records/{id}"
        ))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Ok(None);
    }
    let record: Value = serde_json::from_str(&body)?;
    let vec = record
        .get("vector")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>());
    Ok(vec)
}

/// TD-OBJSTORE-4 same-key orphan ordering: two same-key records (old V1 + newer
/// V2), then SIGKILL before the async manifest flush ⇒ both are unmanifested
/// orphans. On restart, LIST-authority recovery discovers both and resolves to
/// the newer V2 — the orphan is not lost, and the latest mutation wins.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "spawns the server binary — run with --ignored"]
async fn same_key_orphan_recovery_keeps_latest_mutation() -> anyhow::Result<()> {
    let base = tempfile::tempdir()?;
    let root = format!("file://{}/store", base.path().display());
    let data_dir = base.path().join("data");
    let cfg = base.path().join("server.toml");

    let dim = 4;
    let v_old: Vec<f32> = vec![0.1; dim]; // V1
    let v_new: Vec<f32> = vec![0.9; dim]; // V2 (newer — must win)

    // Boot 1: create + insert the SAME key twice (V1 then the newer V2), then
    // SIGKILL immediately so neither batch's manifest pointer has landed.
    let s1 = ServerProcess::start(&root, &data_dir, &cfg).await?;
    create_collection(&s1.base_url, "sk", dim).await?;
    insert_vec(&s1.base_url, "sk", "k", v_old.clone()).await?;
    insert_vec(&s1.base_url, "sk", "k", v_new.clone()).await?;
    s1.crash()?;

    // Boot 2 (recovery): LIST-authority discovers both orphan batches and
    // resolves the same-key pair in durable token order ⇒ the newer V2.
    let s2 = ServerProcess::start(&root, &data_dir, &cfg).await?;
    let got = get_vec(&s2.base_url, "sk", "k")
        .await?
        .expect("recovered record must be point-readable");
    // The latest mutation (V2) must win; the older V1 must not survive beside it.
    let got_f32: Vec<f32> = got.iter().map(|x| *x as f32).collect();
    assert_eq!(
        got_f32, v_new,
        "same-key recovery must keep the NEWER mutation (V2); got {got:?}"
    );
    Ok(())
}
