//! Iceberg interop e2e: materialize a table over pgwire, then prove ProximaDB wrote a
//! spec-shaped Iceberg v3 table to the warehouse (a `v*.metadata.json` referencing an Avro
//! manifest list + manifest), so external Iceberg engines can read it.
//!
//!   cargo test --test iceberg_interop_e2e -- --nocapture

use std::net::TcpListener;
use std::path::{Path as FsPath, PathBuf};
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

struct PgServer {
    pg_port: u16,
    data_dir: PathBuf,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl PgServer {
    async fn start() -> anyhow::Result<Self> {
        let pg_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
        let tmp = TempDir::new()?;
        let data_dir = tmp.path().to_path_buf();
        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = data_dir.clone();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", data_dir.display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", data_dir.display());
        // TD-CONV-1: anchor the system catalog in this server's temp dir, not the
        // shared CWD-relative `file://./metadata` default (cross-run DDL replay).
        config.storage.metadata_url = format!("file://{}", data_dir.join("metadata").display());
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
            data_dir,
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

/// Recursively collect every file path under `dir`.
fn walk(dir: &FsPath, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn materialize_emits_readable_iceberg_v3_table() {
    let server = PgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let _ = client.simple_query("DROP TABLE IF EXISTS region").await;
    client
        .simple_query(
            "CREATE TABLE region (r_regionkey INT PRIMARY KEY, r_name VARCHAR, r_comment VARCHAR)",
        )
        .await
        .expect("create");
    client
        .simple_query("INSERT INTO region (r_regionkey, r_name, r_comment) VALUES (0, 'AMERICA', 'rc0'), (1, 'EUROPE', 'rc1')")
        .await
        .expect("insert");
    client
        .simple_query("ALTER TABLE region MATERIALIZE")
        .await
        .expect("materialize");
    // Materialize publishes the Iceberg snapshot best-effort after the layout flip.
    sleep(Duration::from_millis(300)).await;

    // Walk the warehouse tree and classify the Iceberg artifacts.
    let warehouse = server.data_dir.join("warehouse");
    let mut files = Vec::new();
    walk(&warehouse, &mut files);
    let names: Vec<String> = files
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();
    eprintln!("warehouse files: {names:?}");

    let metadata_json: Vec<&PathBuf> = files
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".metadata.json"))
        })
        .collect();
    let has_manifest_list = names.iter().any(|n| n.ends_with("-list.avro"));
    let has_manifest = names.iter().any(|n| n.ends_with("-m0.avro"));
    let has_data = names.iter().any(|n| n.ends_with(".parquet"));

    assert!(has_data, "materialize wrote a Parquet data file");
    assert!(
        !metadata_json.is_empty(),
        "materialize wrote an Iceberg TableMetadata json; files={names:?}"
    );
    assert!(
        has_manifest_list,
        "wrote an Avro manifest list; files={names:?}"
    );
    assert!(has_manifest, "wrote an Avro manifest; files={names:?}");

    // The metadata is Iceberg format-version 3 and references a manifest list.
    let meta_bytes = std::fs::read(metadata_json[0]).expect("read metadata.json");
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes).expect("parse metadata.json");
    assert_eq!(meta["format-version"], 3, "Iceberg v3 (ProximaDB default)");
    let snapshots = meta["snapshots"].as_array().expect("snapshots array");
    assert_eq!(snapshots.len(), 1, "one snapshot");
    assert!(
        snapshots[0]["manifest-list"].is_string(),
        "snapshot points at a manifest list"
    );
    eprintln!("✓ materialize emitted a readable Iceberg v3 table");
}
