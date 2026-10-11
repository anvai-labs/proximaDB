//! Exercise critical-drainer supervision through the actual server executable.
#![cfg(unix)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};

#[path = "../src/runtime_state.rs"]
mod runtime_state;
#[path = "../src/shutdown.rs"]
mod shutdown;
mod support;

const TOPIC: &str = "embed-ingest";

#[tokio::test]
async fn incomplete_shutdown_retains_ownership_until_drainer_and_storage_finish()
-> anyhow::Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let temp = tempfile::tempdir()?;
    let owner = runtime_state::RuntimeStateWriter::start(temp.path(), None);
    let release = tokio::sync::Notify::new();
    let effect = release.notified();
    tokio::pin!(effect);
    let attempts = AtomicUsize::new(0);
    let storage_stopped = AtomicBool::new(false);
    let retry_observed = tokio::sync::Notify::new();
    let finish = shutdown::finish(owner, async || {
        if attempts.fetch_add(1, Ordering::SeqCst) > 0 {
            retry_observed.notify_one();
        }
        // Model the existing database contract with a controlled in-flight
        // effect. Shrink only the test's grace period, not production's five seconds.
        if tokio::time::timeout(Duration::from_millis(10), &mut effect)
            .await
            .is_err()
        {
            return (Err(anyhow::anyhow!("drainer still in flight")), true);
        }
        storage_stopped.store(true, Ordering::SeqCst);
        (Ok(()), false)
    });
    tokio::pin!(finish);
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            biased;
            _ = &mut finish => anyhow::bail!(
                "incomplete drainer shutdown must not complete the process lifecycle"
            ),
            _ = retry_observed.notified() => Ok(()),
        }
    })
    .await??;
    anyhow::ensure!(attempts.load(Ordering::SeqCst) >= 2, "must retry shutdown");
    anyhow::ensure!(!storage_stopped.load(Ordering::SeqCst));
    let state = runtime_state::read_state(temp.path())
        .ok_or_else(|| anyhow::anyhow!("in-flight shutdown lost runtime ownership"))?;
    anyhow::ensure!(state.phase == runtime_state::Phase::Stopping);
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), &mut finish).await??;
    anyhow::ensure!(storage_stopped.load(Ordering::SeqCst));
    anyhow::ensure!(runtime_state::read_state(temp.path()).is_none());
    Ok(())
}

#[tokio::test]
async fn completed_shutdown_failure_is_returned_without_retry() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let owner = runtime_state::RuntimeStateWriter::start(temp.path(), None);
    let mut attempts = 0;
    let result = shutdown::finish(owner, async || {
        attempts += 1;
        (Err(anyhow::anyhow!("terminal shutdown failure")), false)
    })
    .await;
    anyhow::ensure!(
        result.is_err(),
        "shutdown failure must produce a failing exit"
    );
    anyhow::ensure!(attempts == 1, "completed failures must not be retried");
    anyhow::ensure!(runtime_state::read_state(temp.path()).is_none());
    Ok(())
}

#[tokio::test]
async fn incomplete_shutdown_retry_preserves_the_eventual_failure() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let owner = runtime_state::RuntimeStateWriter::start(temp.path(), None);
    let mut attempts = 0;
    let result = shutdown::finish(owner, async || {
        attempts += 1;
        match attempts {
            1 => (Err(anyhow::anyhow!("drainer still in flight")), true),
            _ => (Err(anyhow::anyhow!("eventual shutdown failure")), false),
        }
    })
    .await;
    anyhow::ensure!(attempts == 2);
    anyhow::ensure!(result.unwrap_err().to_string() == "eventual shutdown failure");
    anyhow::ensure!(runtime_state::read_state(temp.path()).is_none());
    Ok(())
}

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
