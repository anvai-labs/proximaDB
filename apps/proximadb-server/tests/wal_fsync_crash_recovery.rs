//! ADR-069 S5 / TD-WAL-1 S5 — `PerBatch` fsync crash-consistency proof (RPO 0).
//!
//! Asserts the WAL `PerBatch` fsync-on-commit contract (ADR-069 decision D5):
//! after writing N `PerBatch`-synced batches and then SIGKILL-ing the server
//! mid-stream, a restart on the SAME local `file://` volume replays ALL N
//! fsynced batches — RPO 0 for every acknowledged, fsync'd batch.
//!
//! The existing WAL recovery tests use `drop(db)` / `db.stop()` (cooperative
//! shutdown), which drains in-flight I/O before exit and therefore does NOT
//! exercise the torn-page fsync path. Only a real SIGKILL of the server
//! subprocess simulates the power-loss / node-reap event that ADR-069 D5
//! elevates from "primitive exists" to "tested contract".
//!
//! Test shape (N=4 guaranteed fsynced batches + 1 in-flight crash bait):
//! 1. Boot the server against a fresh `file://` data_dir with
//!    `sync_mode = "PerBatch"`.
//! 2. Create a collection; insert N=4 distinct record batches via separate
//!    batch POSTs. Each batch's HTTP 200 returns ONLY after the `PerBatch`
//!    fsync lands on the local WAL file (`batch_sync_coordinator.rs:272`
//!    `sync_file`, triggered at `write_ahead_log/mod.rs:2163-2167`).
//! 3. Fire a 5th "crash-bait" batch and immediately SIGKILL the server
//!    WITHOUT awaiting the response — the fsync for this batch is genuinely
//!    in flight at crash time.
//! 4. Boot a fresh server pointing at the SAME `file://` volume (the
//!    reattached-PersistentVolume scenario from ADR-069 §1).
//! 5. Assert every record from all N=4 fsynced batches is readable by id
//!    (RPO 0). The 5th in-flight batch is best-effort per ADR-069 D5: it
//!    MAY or MAY NOT have fsync'd before the kill — we do not assert it
//!    either way, keeping the test deterministic.
//!
//! This test reuses the `ServerProcess` subprocess harness pattern established
//! by `object_store_restart_recovery.rs`. Loopback-port reservation is shared
//! through `tests/support`; the scenario-specific config and lifecycle helpers
//! stay local because Rust integration tests are separate crates.
//!
//! Run via:
//!   cargo nextest run --test wal_fsync_crash_recovery -- --ignored
//!   cargo test --test wal_fsync_crash_recovery -- --ignored --nocapture

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};

mod support;

/// Write a `file://`-backed server config with `sync_mode = "PerBatch"` — the
/// ADR-069 D5 contract under test. The SAME `root` (storage URL) + `data_dir`
/// are reused across both boots: this is the reattached-volume scenario, where
/// the killed server's local WAL files MUST still be present for the restart's
/// WAL replay to find them.
fn write_config(path: &Path, data_dir: &Path, root: &str, ports: [u16; 4]) {
    let [rest, grpc, flight, pg] = ports;
    let config = format!(
        r#"
[server]
node_id = "td-wal-1-s5"
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
tags = ["td-wal-1-s5", "local-fsync"]

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

    /// SIGKILL the server — the crash primitive. Simulates a node-reap /
    /// power-loss event with no cooperative drain of in-flight I/O. This is
    /// the ONLY way to exercise the torn-page fsync path; `drop(db)` /
    /// `db.stop()` hide it behind a clean shutdown.
    fn crash(mut self) -> anyhow::Result<()> {
        self.child.kill()?;
        self.child.wait()?;
        Ok(())
    }

    /// SIGTERM + wait — clean shutdown for the verification server.
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

/// Create a dim-8 SST collection. Idempotent semantics are not required — the
/// collection name is unique per test invocation.
async fn create_collection(base: &str, collection: &str) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    let response = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": collection,
            "dimension": 8,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "create failed: {status} {body}");
    Ok(())
}

/// Insert one batch of records and return ONLY after the server ACKs the
/// batch. For `sync_mode = "PerBatch"`, the ACK is gated on the per-batch
/// fsync (`batch_sync_coordinator.rs:272` `sync_file` via
/// `write_ahead_log/mod.rs:2163-2167`) — so a 200 here means the batch is
/// durable on the local disk and MUST survive a subsequent SIGKILL.
async fn insert_batch(
    base: &str,
    collection: &str,
    batch_id: usize,
    records: &[(String, Vec<f32>)],
) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    let payload: Vec<Value> = records
        .iter()
        .map(|(id, vec)| json!({ "id": id, "vector": vec }))
        .collect();
    let response = http
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({ "records": payload }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(
        status.is_success(),
        "insert batch {batch_id} failed: {status} {body}"
    );
    Ok(())
}

/// Assert every record id is readable post-restart. Each GET must return 200
/// and the correct id — proving the WAL replay recovered the fsync'd batch.
/// The collection must also recover from the catalog (which itself must have
/// survived the SIGKILL on the same `file://` volume).
async fn assert_records_readable(
    base: &str,
    collection: &str,
    phase: &str,
    ids: &[&str],
) -> anyhow::Result<()> {
    let http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?;
    // Collection must recover from the catalog first.
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

    for id in ids {
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
            "{phase}: record {id} not recovered (PerBatch fsync contract violated): {status} {body}"
        );
        let record: Value = serde_json::from_str(&body)?;
        anyhow::ensure!(
            record.get("id").and_then(Value::as_str) == Some(*id),
            "{phase}: point read for {id} returned the wrong record: {body}"
        );
    }
    Ok(())
}

/// ADR-069 S5 / TD-WAL-1 S5: the `PerBatch` fsync crash-consistency contract.
///
/// Writes N=4 distinct `PerBatch`-synced record batches to a local `file://`
/// WAL, fires a 5th crash-bait batch, SIGKILLs the server, restarts on the
/// SAME volume, and asserts all N=4 fsynced batches replay (RPO 0). The 5th
/// batch is the in-flight best-effort case documented in ADR-069 D5 — not
/// asserted, since whether its fsync beat the kill is nondeterministic.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "spawns the proximadb-server subprocess; run with `--ignored`"]
async fn perbatch_fsync_survives_sigkill_rpo_zero() -> anyhow::Result<()> {
    // One tempdir backs BOTH boots — the "reattached persistent volume"
    // scenario from ADR-069 §1. The killed server's WAL files MUST still be
    // here for the restart to replay them.
    let tmp = tempfile::tempdir()?;
    let storage_root = format!("file://{}", tmp.path().join("storage").display());
    let data_dir = tmp.path().join("server_data");
    std::fs::create_dir_all(&data_dir)?;
    let config_a = tmp.path().join("server_a.toml");
    let config_b = tmp.path().join("server_b.toml");
    let collection = format!("fsync_crash_{}", uuid::Uuid::new_v4().simple());

    // Distinct, deterministic id + vector per record so each is unambiguously
    // recovered (no cross-batch collisions). N=4 batches × 3 records = 12
    // guaranteed-survivor records.
    const N_BATCHES: usize = 4;
    const RECORDS_PER_BATCH: usize = 3;
    let batches: Vec<Vec<(String, Vec<f32>)>> = (0..N_BATCHES)
        .map(|b| {
            (0..RECORDS_PER_BATCH)
                .map(|i| {
                    (
                        format!("batch{b}-rec{i}"),
                        vec![b as f32, i as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    )
                })
                .collect()
        })
        .collect();

    // ---- Phase 1: boot, seed N fsynced batches, fire crash bait, SIGKILL ----
    let server = ServerProcess::start(&storage_root, &data_dir, &config_a).await?;
    create_collection(&server.base_url, &collection).await?;

    // Each awaited insert returns ONLY after `PerBatch` fsync → guaranteed
    // RPO 0 for these N batches even under a hard SIGKILL.
    for (b, records) in batches.iter().enumerate() {
        insert_batch(&server.base_url, &collection, b, records).await?;
    }

    // Fire the crash-bait batch WITHOUT awaiting — genuinely in-flight at kill
    // time. Best-effort per ADR-069 D5; not asserted below.
    let bait_url = format!(
        "{}/api/v2/collections/{collection}/records/batch",
        server.base_url
    );
    let bait_body = serde_json::to_string(&json!({
        "records": [{ "id": "crashbait-0", "vector": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0] }]
    }))?;
    let _bait = tokio::spawn(async move {
        let http = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(30))
            .build()?;
        let _ = http
            .post(&bait_url)
            .header("content-type", "application/json")
            .body(bait_body)
            .send()
            .await;
        Ok::<_, anyhow::Error>(())
    });
    // Give the request a moment to reach the server and be mid-fsync, so the
    // crash genuinely catches a write in flight (the power-loss scenario).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // SIGKILL — simulates power loss / node reap with no cooperative drain.
    server.crash()?;

    // ---- Phase 2: restart on the SAME volume, assert RPO 0 ----------------
    let restarted = ServerProcess::start(&storage_root, &data_dir, &config_b).await?;

    // The N=4 fsynced batches MUST all survive the crash (RPO 0 contract).
    let guaranteed_ids: Vec<&str> = batches
        .iter()
        .flatten()
        .map(|(id, _)| id.as_str())
        .collect();
    assert_records_readable(
        &restarted.base_url,
        &collection,
        "post-SIGKILL WAL replay",
        &guaranteed_ids,
    )
    .await?;

    restarted.graceful()?;
    Ok(())
}
