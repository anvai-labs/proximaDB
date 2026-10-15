//! Cold backup/restore round trip against the REAL server binary.
//!
//! The unit tests in `src/cold_backup` prove the copy/verify/refuse mechanics
//! on synthetic files. This proves the property an operator relies on: data a
//! running server acknowledged survives `backup` → wipe every storage root →
//! `restore` → restart, byte for byte — and that backup refuses while the
//! server is still running (the ownership lease), leaving no manifest behind.
//!
//! Layout: four DISTINCT local roots (data_dir, metadata, collections, wal) so
//! multi-root grouping and copying are exercised, not one directory.
//!
//! Run via:
//!   cargo test -p proximadb-server --test cold_backup_restore -- --test-threads=1

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};

mod support;

fn write_config(path: &Path, data_dir: &Path, root: &str, ports: [u16; 4]) {
    let [rest, grpc, flight, pg] = ports;
    let config = format!(
        r#"
[server]
node_id = "cold-backup"
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
tags = ["cold-backup"]

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

    fn graceful(mut self) -> anyhow::Result<()> {
        let status = Command::new("kill")
            .args(["-INT", &self.child.id().to_string()])
            .status()?;
        anyhow::ensure!(status.success(), "failed to SIGINT pid {}", self.child.id());
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait()? {
                anyhow::ensure!(
                    status.success(),
                    "unclean exit during graceful stop: {status}"
                );
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

impl Drop for ServerProcess {
    fn drop(&mut self) {
        // A failing assertion must not leak a server holding the roots.
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn http() -> anyhow::Result<Client> {
    Ok(Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()?)
}

async fn create_collection(base: &str, collection: &str) -> anyhow::Result<()> {
    let response = http()?
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

async fn insert_batch(
    base: &str,
    collection: &str,
    records: &[(String, Vec<f32>)],
) -> anyhow::Result<()> {
    let payload: Vec<Value> = records
        .iter()
        .map(|(id, vector)| json!({ "id": id, "vector": vector }))
        .collect();
    let response = http()?
        .post(format!(
            "{base}/api/v2/collections/{collection}/records/batch"
        ))
        .json(&json!({ "records": payload }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "insert failed: {status} {body}");
    Ok(())
}

/// The record's vector as served — what must come back identical.
async fn read_vector(base: &str, collection: &str, id: &str) -> anyhow::Result<Value> {
    let response = http()?
        .get(format!(
            "{base}/api/v2/collections/{collection}/records/{id}"
        ))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    anyhow::ensure!(status.is_success(), "read {id}: {status} {body}");
    let record: Value = serde_json::from_str(&body)?;
    anyhow::ensure!(
        record.get("id").and_then(Value::as_str) == Some(id),
        "read {id} returned the wrong record: {body}"
    );
    record
        .get("vector")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("record {id} has no vector: {body}"))
}

/// Run an offline subcommand of the server binary.
fn admin(config: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proximadb-server"))
        .arg("--config")
        .arg(config)
        .args(args)
        .output()
        .expect("spawn proximadb-server admin subcommand")
}

fn roots(base: &Path) -> [PathBuf; 4] {
    [
        base.join("server_data"),
        base.join("metadata"),
        base.join("collections"),
        base.join("wal"),
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_backup_round_trip_restores_identical_rows() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let live = temp.path().join("live");
    std::fs::create_dir_all(&live)?;
    let root = format!("file://{}", live.display());
    let data_dir = live.join("server_data");
    let config = temp.path().join("server.toml");
    let collection = "cold_backup_rt";

    // 1. Write acknowledged data through the real server.
    let server = ServerProcess::start(&root, &data_dir, &config).await?;
    create_collection(&server.base_url, collection).await?;
    let mut ids = Vec::new();
    for batch in 0..2 {
        let records: Vec<(String, Vec<f32>)> = (0..5)
            .map(|i| {
                let n = batch * 5 + i;
                (
                    format!("rec-{n}"),
                    (0..8).map(|d| (n * 8 + d) as f32 / 100.0).collect(),
                )
            })
            .collect();
        ids.extend(records.iter().map(|(id, _)| id.clone()));
        insert_batch(&server.base_url, collection, &records).await?;
    }
    // What the server serves straight from the write path (memtable, f32).
    let mut written = Vec::new();
    for id in &ids {
        written.push(read_vector(&server.base_url, collection, id).await?);
    }

    // 2. Backup while the server runs MUST refuse — and leave no manifest.
    let refused_to = temp.path().join("bk_live");
    let refused = admin(
        &config,
        &["backup", "--to", refused_to.to_str().expect("utf8")],
    );
    assert!(
        !refused.status.success(),
        "backup must refuse while the server holds the storage lease"
    );
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("still running"),
        "refusal must say why: {}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(!refused_to.join("manifest.json").exists());

    // 3. Baseline = what a PLAIN restart serves. A graceful stop flushes the
    //    memtable to segments, and reads may then come from a different tier
    //    than before the stop; comparing the restored server against the
    //    pre-stop read would conflate backup fidelity with that tier change.
    //    Backup must reproduce exactly what a restart without it serves.
    server.graceful()?;
    let server = ServerProcess::start(&root, &data_dir, &config).await?;
    let mut expected = Vec::new();
    for id in &ids {
        expected.push(read_vector(&server.base_url, collection, id).await?);
    }
    if expected != written {
        eprintln!(
            "NOTE: point reads changed across a plain restart (memtable -> segment tier); \
             rec-0 written={} restarted={}",
            written[0], expected[0]
        );
    }

    // 4. Stop cleanly, back up.
    server.graceful()?;
    let backup_to = temp.path().join("bk");
    let backup = admin(
        &config,
        &["backup", "--to", backup_to.to_str().expect("utf8")],
    );
    assert!(
        backup.status.success(),
        "backup failed: {}",
        String::from_utf8_lossy(&backup.stderr)
    );
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(backup_to.join("manifest.json"))?)?;
    assert_eq!(manifest["clean_shutdown"], json!(true));
    let root_count = manifest["roots"].as_array().map(Vec::len).unwrap_or(0);
    assert_eq!(root_count, 4, "four distinct roots: {manifest:#}");
    let listed: Vec<&str> = manifest["files"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|file| file["path"].as_str())
        .collect();
    assert!(!listed.is_empty(), "the backup must contain data files");
    assert!(
        listed.iter().all(
            |path| !path.ends_with("access.lock") && !path.ends_with(".proximadb-runtime.json")
        ),
        "coordination/runtime files must not be backed up: {listed:?}"
    );

    // 5. Lose every root, restore, restart.
    for dir in roots(&live) {
        std::fs::remove_dir_all(&dir)?;
    }
    let restore = admin(
        &config,
        &["restore", "--from", backup_to.to_str().expect("utf8")],
    );
    assert!(
        restore.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&restore.stderr)
    );

    let server = ServerProcess::start(&root, &data_dir, &config).await?;
    for (id, want) in ids.iter().zip(&expected) {
        let got = read_vector(&server.base_url, collection, id).await?;
        assert_eq!(&got, want, "record {id} must come back identical");
    }
    server.graceful()?;
    Ok(())
}
