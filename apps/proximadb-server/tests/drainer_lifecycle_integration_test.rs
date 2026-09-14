//! Exercise critical-drainer supervision through the actual server executable.
#![cfg(unix)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};

mod support;

const TOPIC: &str = "embed-ingest";

fn queue_config(root: &Path) -> QueueConfig {
    QueueConfig {
        root: format!("file://{}", root.display()),
        topics: HashMap::from([(TOPIC.to_string(), TopicConfig::default())]),
        ..Default::default()
    }
}

struct ServerProcess {
    child: Child,
    log: PathBuf,
}

impl ServerProcess {
    fn start(dir: &Path, queue_root: &Path) -> anyhow::Result<Self> {
        let reservation = support::reserve_loopback_ports::<4>()?;
        let [rest, grpc, flight, pg] = reservation.ports();
        let config_path = dir.join("server.toml");
        std::fs::write(
            &config_path,
            format!(
                r#"
[server]
bind_address = "127.0.0.1"
port = {rest}
data_dir = "{data}"
[storage]
metadata_url = "file://{data}/metadata"
[[storage.storage_locations]]
url = "file://{data}/collections"
weight = 1
tags = ["drainer-lifecycle"]
[storage.wal_config]
write_buffer_directory = "file://{data}/wal"
[api]
rest_port = {rest}
grpc_port = {grpc}
arrow_flight_port = {flight}
pg_port = {pg}
unified_mode = false
[queue]
root = "file://{queue}"
drainer_partitions = "0..16"
[monitoring]
metrics_enabled = false
log_level = "info"
"#,
                data = dir.display(),
                queue = queue_root.display(),
            ),
        )?;
        let log = dir.join("server.log");
        let output = std::fs::File::create(&log)?;
        let mut command = Command::new(env!("CARGO_BIN_EXE_proximadb-server"));
        command
            .current_dir(dir)
            .args([
                "--config",
                config_path.to_str().expect("UTF-8 fixture path"),
            ])
            .env("RUST_LOG", "info")
            // Scope overrides to the owned subprocess, never the test process.
            .env_remove("PROXIMADB_QUEUE_ROOT")
            .env_remove("PROXIMADB_QUEUE_OBJECT_ARCHIVE")
            .env_remove("PROXIMADB_EMBED_DRAINER_PARTITIONS")
            .env_remove("PROXIMADB_PGWIRE_AUTH")
            .stdout(Stdio::from(output.try_clone()?))
            .stderr(Stdio::from(output));
        drop(reservation);
        Ok(Self {
            child: command.spawn()?,
            log,
        })
    }

    fn output(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    async fn wait_for_log(&mut self, text: &str) -> anyhow::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if self.output().contains(text) {
                return Ok(());
            }
            anyhow::ensure!(
                self.child.try_wait()?.is_none() && Instant::now() < deadline,
                "server did not reach {text:?}: {}",
                self.output()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn wait_for_exit(&mut self) -> anyhow::Result<ExitStatus> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "server remained alive after terminal drainer failure or shutdown: {}",
                self.output()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[tokio::test]
async fn terminal_drainer_failure_exits_server_and_preserves_unacked_work() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let root = temp.path().join("queue");
    let config = queue_config(&root);
    let queue = QueueClient::open(config.clone()).await?;
    // Valid JSON, rejected before any embedding/provider I/O. Malformed JSON
    // is deliberately skipped by the drainer and would not test termination.
    let payload = serde_json::to_vec(&serde_json::json!({
        "target_collection": "lifecycle-test",
        "tenant_id": "different-tenant",
        "embedding_route_identity": {"kind": "bge-small"},
        "expected_dimension": 384,
        "records": []
    }))?;
    let receipt = queue
        .producer()
        .send(Message::new(TOPIC, "tenant-a", payload))
        .await?;
    queue.shutdown().await?;
    drop(queue);

    let mut server = ServerProcess::start(temp.path(), &root)?;
    server.wait_for_log("embedding drainer task failed").await?;
    let status = server.wait_for_exit().await?;
    anyhow::ensure!(
        !status.success(),
        "terminal worker failure must exit nonzero"
    );
    anyhow::ensure!(
        server.output().contains("envelope tenant"),
        "expected injected drainer failure: {}",
        server.output()
    );
    anyhow::ensure!(
        server
            .output()
            .contains("Critical embedding drainer stopped")
            && server.output().contains("Server shutdown complete"),
        "supervisor must trigger awaited shutdown, not an unrelated startup error: {}",
        server.output()
    );
    anyhow::ensure!(
        !temp.path().join(".proximadb-runtime.json").exists(),
        "awaited shutdown must clear runtime ownership"
    );

    let recovered = QueueClient::open(config).await?;
    let consumer = recovered.consumer("embed-drainer");
    consumer.subscribe(TOPIC, &[receipt.partition]).await?;
    let batch = consumer.poll(1, Duration::ZERO).await?;
    let preserved = batch.len() == 1 && batch[0].message_id == receipt.message_id;
    consumer.shutdown().await?;
    recovered.shutdown().await?;
    anyhow::ensure!(
        preserved,
        "terminal failure must not ACK or discard queued work"
    );
    Ok(())
}

#[tokio::test]
async fn healthy_drainer_stays_alive_until_signal() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let mut server = ServerProcess::start(temp.path(), &temp.path().join("queue"))?;
    server
        .wait_for_log("ProximaDB server started successfully")
        .await?;
    tokio::time::sleep(Duration::from_secs(2)).await;
    anyhow::ensure!(
        server.child.try_wait()?.is_none(),
        "healthy drainer must keep serving"
    );
    let sent = Command::new("kill")
        .args(["-TERM", &server.child.id().to_string()])
        .status()?;
    anyhow::ensure!(sent.success(), "signal owned server process");
    anyhow::ensure!(
        server.wait_for_exit().await?.success(),
        "normal shutdown succeeds"
    );
    Ok(())
}
