//! ADR-095 OLTP durability against the REAL server binary (TD-PGOLTP-1/2).
//!
//! Every durable root lives on one "volume" directory, as it would on an
//! EBS / Azure Disk / PD mount; the writer-epoch authority lives OUTSIDE it, as
//! it would in an object store. Scenarios:
//!
//! 1. crash matrix — a process killed right after a COMMIT's group frame is
//!    durable recovers the whole transaction; one killed mid-write of that frame
//!    boots cleanly with none of it, and earlier commits intact;
//! 2. reattach — host A is frozen (SIGSTOP), its volume is "moved" (copied) to
//!    host B, B boots and wins the next epoch; when A resumes, its watchdog sees
//!    the newer epoch and A exits 75 without serving another write;
//! 3. refusal — a durable root off the volume, or no explicit control URL,
//!    refuses to boot and names why.
//!
//! Run via:
//!   cargo nextest run -p proximadb-server --test oltp_reattach_e2e

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

mod support;

const FAULT: &str = "PROXIMADB_TEST_TXN_COMMIT_FAULT";
const FENCED_EXIT_CODE: i32 = 75;

struct Layout {
    _tmp: tempfile::TempDir,
    base: PathBuf,
    control: PathBuf,
}

impl Layout {
    fn new() -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path().to_path_buf();
        let control = base.join("control");
        std::fs::create_dir_all(&control).expect("control dir");
        Self {
            _tmp: tmp,
            base,
            control,
        }
    }

    fn host_dir(&self, name: &str) -> PathBuf {
        let dir = self.base.join(name);
        std::fs::create_dir_all(&dir).expect("host dir");
        dir
    }
}

struct HostOptions<'a> {
    volume: &'a Path,
    control_url: Option<String>,
    cwd: &'a Path,
    env: Vec<(&'static str, String)>,
    /// Put the WAL write buffer OFF the volume (refusal scenario).
    wal_buffer_override: Option<PathBuf>,
}

fn write_config(path: &Path, ports: [u16; 4], opts: &HostOptions<'_>) {
    let [rest, grpc, flight, pg] = ports;
    let vol = opts.volume.display();
    let wal_buffer = opts
        .wal_buffer_override
        .clone()
        .unwrap_or_else(|| opts.volume.join("wal"));
    let control = opts
        .control_url
        .as_ref()
        .map(|url| format!("control_url = \"{url}\"\n"))
        .unwrap_or_default();
    let config = format!(
        r#"
[server]
node_id = "oltp-reattach"
bind_address = "127.0.0.1"
port = {rest}
data_dir = "{vol}/data"

[server.tenant]
mode = "single_tenant"
default_tenant = "default"

[storage]
metadata_url = "file://{vol}/metadata"
mmap_enabled = false

[[storage.storage_locations]]
url = "file://{vol}/collections"
weight = 1
tags = ["oltp"]

[storage.optimization]
enable_mmap = false

[storage.wal_config]
write_buffer_directory = "{wal_buffer}"
enable_wal = true
sync_mode = "PerBatch"

[storage.sst_config]
data_directory = "{vol}/collections"
mmap_enabled = false

[storage.viper_config]
data_directory = "{vol}/collections"

[api]
rest_port = {rest}
grpc_port = {grpc}
arrow_flight_port = {flight}
pg_port = {pg}
unified_mode = false

[monitoring]
metrics_enabled = false
log_level = "info"

[oltp]
atomic_commit = true

[oltp.durable_volume]
root = "{vol}"
deployment_id = "e2e"
fence_check_interval_ms = 200
{control}"#,
        wal_buffer = wal_buffer.display(),
    );
    std::fs::write(path, config).expect("write config");
}

struct Host {
    child: Child,
    rest_port: u16,
    pg_port: u16,
}

impl Host {
    fn spawn(opts: &HostOptions<'_>, stderr: Stdio) -> anyhow::Result<Self> {
        let reservation = support::reserve_loopback_ports::<4>()?;
        let ports = reservation.ports();
        let config_path = opts.cwd.join("proximadb.toml");
        write_config(&config_path, ports, opts);
        let mut command = Command::new(env!("CARGO_BIN_EXE_proximadb-server"));
        command
            .arg("--config")
            .arg(&config_path)
            .env("RUST_LOG", "info")
            .current_dir(opts.cwd)
            .envs(opts.env.iter().map(|(k, v)| (*k, v.as_str())))
            .stdout(Stdio::null())
            .stderr(stderr);
        drop(reservation);
        Ok(Self {
            child: command.spawn()?,
            rest_port: ports[0],
            pg_port: ports[3],
        })
    }

    async fn start(opts: &HostOptions<'_>) -> anyhow::Result<Self> {
        let host = Self::spawn(opts, Stdio::inherit())?;
        host.wait_ready().await?;
        Ok(host)
    }

    async fn wait_ready(&self) -> anyhow::Result<()> {
        let http = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()?;
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Ok(r) = http
                .get(format!("http://127.0.0.1:{}/health", self.rest_port))
                .send()
                .await
                && r.status().is_success()
            {
                return Ok(());
            }
            anyhow::ensure!(Instant::now() < deadline, "server did not become healthy");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn client(&self) -> anyhow::Result<tokio_postgres::Client> {
        let conn_str = format!(
            "host=127.0.0.1 port={} user=anonymous dbname=proximadb sslmode=disable",
            self.pg_port
        );
        let (client, conn) = tokio_postgres::connect(&conn_str, tokio_postgres::NoTls).await?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        Ok(client)
    }

    fn signal(&self, sig: &str) -> anyhow::Result<()> {
        let status = Command::new("kill")
            .args([sig, &self.child.id().to_string()])
            .status()?;
        anyhow::ensure!(status.success(), "kill {sig} failed");
        Ok(())
    }

    fn crash(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Wait for the process to exit on its own; its exit code.
    fn wait_exit(&mut self, within: Duration) -> anyhow::Result<Option<i32>> {
        let deadline = Instant::now() + within;
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status.code());
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "process did not exit within {within:?}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.signal("-CONT");
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

async fn ids(client: &tokio_postgres::Client, table: &str) -> anyhow::Result<Vec<i64>> {
    let mut ids: Vec<i64> = client
        .simple_query(&format!("SELECT id FROM {table}"))
        .await?
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(row) => row.get(0).and_then(|v| v.parse().ok()),
            _ => None,
        })
        .collect();
    ids.sort_unstable();
    Ok(ids)
}

fn run(test: impl std::future::Future<Output = anyhow::Result<()>>) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(test)
}

fn file_url(path: &Path) -> String {
    format!("file://{}", path.display())
}

#[test]
fn commit_crash_matrix_is_all_or_nothing_e2e() -> anyhow::Result<()> {
    run(async {
        let layout = Layout::new();
        let volume = layout.host_dir("vol");
        let cwd = layout.host_dir("host-a");
        let opts = |env: Vec<(&'static str, String)>| HostOptions {
            volume: &volume,
            control_url: Some(file_url(&layout.control)),
            cwd: &cwd,
            env,
            wal_buffer_override: None,
        };

        // Setup on a clean boot: with atomic commit on, autocommit statements
        // also take the atomic append, so they must not run under a fault.
        let host = Host::start(&opts(vec![])).await?;
        let client = host.client().await?;
        client
            .simple_query("CREATE TABLE acct (id BIGINT PRIMARY KEY, label VARCHAR)")
            .await?;
        client
            .simple_query("INSERT INTO acct (id, label) VALUES (1, 'autocommit')")
            .await?;
        host.crash();

        // Crash point 1: killed right AFTER the group frame is durable, BEFORE
        // it is applied in memory.
        let host = Host::start(&opts(vec![(FAULT, "abort_after_append".into())])).await?;
        let client = host.client().await?;
        let commit = client
            .simple_query(
                "BEGIN; INSERT INTO acct (id, label) VALUES (2, 'b'); \
                 INSERT INTO acct (id, label) VALUES (3, 'c'); COMMIT",
            )
            .await;
        assert!(commit.is_err(), "the server aborts at the crash point");
        host.crash();

        let host = Host::start(&opts(vec![])).await?;
        assert_eq!(
            ids(&host.client().await?, "acct").await?,
            vec![1, 2, 3],
            "a COMMIT whose group frame was durable is recovered WHOLE"
        );
        host.crash();

        // Crash point 2: killed MID-WRITE of the group frame (torn append).
        let host = Host::start(&opts(vec![(FAULT, "torn_append".into())])).await?;
        let commit = host
            .client()
            .await?
            .simple_query(
                "BEGIN; INSERT INTO acct (id, label) VALUES (4, 'd'); \
                 INSERT INTO acct (id, label) VALUES (5, 'e'); COMMIT",
            )
            .await;
        assert!(commit.is_err());
        host.crash();

        let host = Host::start(&opts(vec![])).await?;
        assert_eq!(
            ids(&host.client().await?, "acct").await?,
            vec![1, 2, 3],
            "a torn COMMIT leaves NONE of its rows, and earlier commits intact"
        );
        let quarantined = std::fs::read_dir(volume.join("data/pgwire"))?
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".torn-"));
        assert!(quarantined, "the dropped torn tail is kept beside the WAL");
        host.crash();
        Ok(())
    })
}

#[test]
fn reattached_volume_fences_the_previous_owner_e2e() -> anyhow::Result<()> {
    run(async {
        let layout = Layout::new();
        let vol_a = layout.host_dir("vol-a");
        let cwd_a = layout.host_dir("host-a");
        let cwd_b = layout.host_dir("host-b");
        let control_url = file_url(&layout.control);

        let mut host_a = Host::start(&HostOptions {
            volume: &vol_a,
            control_url: Some(control_url.clone()),
            cwd: &cwd_a,
            env: vec![],
            wal_buffer_override: None,
        })
        .await?;
        let client_a = host_a.client().await?;
        client_a
            .simple_query("CREATE TABLE acct (id BIGINT PRIMARY KEY, label VARCHAR)")
            .await?;
        client_a
            .simple_query(
                "BEGIN; INSERT INTO acct (id, label) VALUES (1, 'a'); \
                 INSERT INTO acct (id, label) VALUES (2, 'b'); COMMIT",
            )
            .await?;

        // Host A loses its machine without dying (partition / pause). The
        // orchestrator moves the volume to host B: modelled as a copy taken
        // while A is frozen, so B sees exactly what A made durable.
        host_a.signal("-STOP")?;
        let vol_b = layout.base.join("vol-b");
        let status = Command::new("cp")
            .arg("-a")
            .arg(&vol_a)
            .arg(&vol_b)
            .status()?;
        anyhow::ensure!(status.success(), "copy volume");

        let host_b = Host::start(&HostOptions {
            volume: &vol_b,
            control_url: Some(control_url.clone()),
            cwd: &cwd_b,
            env: vec![],
            wal_buffer_override: None,
        })
        .await?;
        let client_b = host_b.client().await?;
        assert_eq!(
            ids(&client_b, "acct").await?,
            vec![1, 2],
            "B serves A's commits"
        );

        // A wakes up: its watchdog sees B's newer epoch and exits WITHOUT
        // serving another write or flushing.
        host_a.signal("-CONT")?;
        let code = host_a.wait_exit(Duration::from_secs(10))?;
        assert_eq!(code, Some(FENCED_EXIT_CODE), "the old owner self-fences");
        assert!(
            host_a.client().await.is_err(),
            "the fenced owner no longer accepts connections"
        );

        // B keeps owning and writing.
        client_b
            .simple_query("BEGIN; INSERT INTO acct (id, label) VALUES (3, 'c'); COMMIT")
            .await?;
        assert_eq!(ids(&client_b, "acct").await?, vec![1, 2, 3]);
        host_b.crash();
        Ok(())
    })
}

#[test]
fn boot_refuses_off_volume_roots_and_implicit_control_e2e() -> anyhow::Result<()> {
    let layout = Layout::new();
    let volume = layout.host_dir("vol");
    let cwd = layout.host_dir("host");
    let outside = layout.base.join("off-volume-wal");

    let mut cases = vec![
        (
            HostOptions {
                volume: &volume,
                control_url: Some(file_url(&layout.control)),
                cwd: &cwd,
                env: vec![],
                wal_buffer_override: Some(outside.clone()),
            },
            "WalWriteBuffer",
        ),
        (
            HostOptions {
                volume: &volume,
                control_url: None,
                cwd: &cwd,
                env: vec![],
                wal_buffer_override: None,
            },
            "control_url",
        ),
    ];
    for (opts, expected) in cases.drain(..) {
        let mut host = Host::spawn(&opts, Stdio::piped())?;
        let code = host.wait_exit(Duration::from_secs(60))?;
        let mut stderr = String::new();
        if let Some(mut pipe) = host.child.stderr.take() {
            use std::io::Read;
            pipe.read_to_string(&mut stderr)?;
        }
        assert_ne!(code, Some(0), "boot must be refused ({expected})");
        assert!(
            stderr.contains(expected),
            "refusal must name '{expected}'; stderr was:\n{stderr}"
        );
    }
    Ok(())
}
