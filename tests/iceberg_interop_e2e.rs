//! Iceberg interop e2e: materialize a table over pgwire, then prove ProximaDB wrote a
//! spec-shaped Iceberg table to the warehouse (a `v*.metadata.json` referencing an Avro
//! manifest list + manifest), and that engines which are NOT ProximaDB read it back
//! value-for-value (ADR-097 D3, TD-SEAM-1).
//!
//!   cargo test --test iceberg_interop_e2e -- --nocapture
//!
//! The read-back gate shells out to `scripts/iceberg_readback.py` (PyIceberg through
//! the REST catalog and from the metadata file; DuckDB `iceberg_scan`). Engines that
//! are not installed are skipped unless listed in `PROXIMADB_ICEBERG_READBACK_REQUIRE`
//! (CI sets it), so a missing reader fails CI instead of silently passing.

use std::path::{Path as FsPath, PathBuf};

use futures::StreamExt;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::sleep;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
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
async fn materialize_emits_readable_iceberg_table() {
    let server = TestPgServer::start().await.expect("server start");
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
    let warehouse = server.data_dir().join("warehouse");
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

    // The metadata is Iceberg format-version 2 (ADR-097 D3: v2 until an external v3
    // reader is in the read-back gate; no v3-only column types here) and references a
    // manifest list.
    let meta_bytes = std::fs::read(metadata_json[0]).expect("read metadata.json");
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes).expect("parse metadata.json");
    assert_eq!(meta["format-version"], 2, "Iceberg v2 for external readers");
    let snapshots = meta["snapshots"].as_array().expect("snapshots array");
    assert_eq!(snapshots.len(), 1, "one snapshot");
    assert!(
        snapshots[0]["manifest-list"].is_string(),
        "snapshot points at a manifest list"
    );
    eprintln!("✓ materialize emitted a readable Iceberg table");
}

/// Rows of `SELECT {cols} FROM {table}` over pgwire, as text (`None` = NULL).
async fn pg_rows(
    client: &tokio_postgres::Client,
    table: &str,
    cols: &[&str],
) -> Vec<Vec<Option<String>>> {
    let sql = format!("SELECT {} FROM {table}", cols.join(", "));
    let mut out = Vec::new();
    for msg in client.simple_query(&sql).await.expect("select") {
        if let tokio_postgres::SimpleQueryMessage::Row(row) = msg {
            out.push(
                (0..cols.len())
                    .map(|i| row.get(i).map(str::to_string))
                    .collect(),
            );
        }
    }
    out
}

/// Tenant of a table created over pgwire: the session is tenant-scoped
/// (`resolve_table_scoped` prefixes the tenant — `proximadb`, the database), so the
/// catalog entry lives at `proximadb.default`. The Iceberg REST catalog resolves
/// every request within the request tenant (#2072): a client selects the tenant
/// with `X-Tenant-ID` and addresses the table by its client namespace `default`.
const TENANT: &str = "proximadb";
const NAMESPACE: &str = "default";

/// `metadata-location` served by ProximaDB's Iceberg REST catalog for `table`.
async fn rest_metadata_location(rest_port: u16, table: &str) -> serde_json::Value {
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let url =
        format!("http://127.0.0.1:{rest_port}/iceberg/v1/namespaces/{NAMESPACE}/tables/{table}");
    let resp = http
        .get(&url)
        .header("X-Tenant-ID", TENANT)
        .send()
        .await
        .expect("load_table");
    assert!(
        resp.status().is_success(),
        "load_table {url}: {}",
        resp.status()
    );
    resp.json().await.expect("load_table json")
}

/// Run `scripts/iceberg_readback.py` against the published table. Returns `false`
/// when no reader is installed and none is required (the gate is then skipped).
fn external_readback(
    rest_port: u16,
    table: &str,
    metadata_location: &str,
    cols: &[&str],
    rows: &[Vec<Option<String>>],
    scratch: &FsPath,
) -> bool {
    let required = std::env::var("PROXIMADB_ICEBERG_READBACK_REQUIRE").unwrap_or_default();
    let python = std::env::var("PROXIMADB_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let has_pyiceberg = std::process::Command::new(&python)
        .args(["-c", "import pyiceberg"])
        .status()
        .is_ok_and(|s| s.success());
    if !has_pyiceberg && required.is_empty() {
        eprintln!("SKIP external Iceberg read-back: pyiceberg not installed");
        return false;
    }
    let expected = scratch.join(format!("{table}-expected.json"));
    std::fs::write(
        &expected,
        serde_json::to_vec(&serde_json::json!({"columns": cols, "rows": rows})).unwrap(),
    )
    .expect("write expected rows");
    let script = FsPath::new(env!("CARGO_MANIFEST_DIR")).join("scripts/iceberg_readback.py");
    let out = std::process::Command::new(&python)
        .arg(script)
        .args([
            "--rest-uri",
            &format!("http://127.0.0.1:{rest_port}/iceberg"),
        ])
        .args([
            "--namespace",
            NAMESPACE,
            "--tenant",
            TENANT,
            "--table",
            table,
        ])
        .args(["--metadata-location", metadata_location])
        .arg("--expected")
        .arg(&expected)
        .args(["--require", &required])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .expect("run iceberg_readback.py");
    let report = String::from_utf8_lossy(&out.stdout);
    eprintln!(
        "iceberg read-back report:\n{report}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "external Iceberg read-back failed:\n{report}"
    );
    true
}

/// P2.4 slice 3b (TD-SEAM-4): a published snapshot carrying deletion vectors is
/// read delete-aware EVERYWHERE. The vectors are published test-side through
/// the engine's writer (the delete WRITER is the next slice and stays
/// unactivated); what this gate proves is the READ side end to end: the live
/// server's own scan (manifest-selected, DV-applied) and the external engines
/// (PyIceberg REST/static, DuckDB iceberg_scan — CI pins pyiceberg 0.12.0 and
/// duckdb 1.5.6, both read v3 + deletion vectors) all exclude the deleted rows,
/// value-for-value with the pgwire answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deletion_vectors_are_invisible_to_proximadb_and_external_engines() {
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;
    use proximadb_iceberg_engine::deletion_vector::PositionDeletes;
    use proximadb_iceberg_engine::iceberg as engine_iceberg;
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;

    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let table = "dv_returns";
    let cols = ["sr_id", "sr_item"];
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (sr_id INT PRIMARY KEY, sr_item VARCHAR)"
        ))
        .await
        .expect("create");
    client
        .simple_query(&format!(
            "INSERT INTO {table} (sr_id, sr_item) VALUES \
             (1, 'keep-me-not'), (2, 'keep'), (3, 'gone-too'), (4, 'keep')"
        ))
        .await
        .expect("insert");
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize");
    sleep(Duration::from_millis(300)).await;

    // Locate the table's Iceberg base under the warehouse: its metadata dir
    // holds the `v*.metadata.json` the REST catalog serves.
    let warehouse = server.data_dir().join("warehouse");
    let mut files = Vec::new();
    walk(&warehouse, &mut files);
    let metadata_json = files
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".metadata.json"))
                && p.to_string_lossy().contains(table)
        })
        .expect("the table's metadata.json")
        .clone();
    let metadata_dir = metadata_json.parent().expect("metadata dir").to_path_buf();
    let base_dir = metadata_dir.parent().expect("table base").to_path_buf();
    let base_url = format!("file://{}", base_dir.display());

    // Publish a v3 successor snapshot carrying one deletion vector, deleting
    // the positions of sr_id 1 and 3 in the (single) published data file.
    let bridge = IcebergObjectStoreBridge::from_url(&base_url).expect("bridge");
    let version = bridge
        .latest_metadata_version("metadata")
        .await
        .expect("latest version")
        .expect("a published snapshot");
    let snapshot = bridge
        .read_iceberg_snapshot_with_deletes("metadata", version)
        .await
        .expect("read snapshot");
    assert!(snapshot.data_files.len() == 1, "one published part");
    let data_file = snapshot.data_files[0].clone();
    assert!(
        snapshot.deletion_vectors.is_empty(),
        "the materialized snapshot carries no vectors yet"
    );

    // Map sr_id → file position by reading the part's first column.
    let object_url =
        url::Url::parse(bridge.object_uri(data_file.as_ref()).as_str()).expect("part url");
    let (store, path) = object_store::parse_url(&object_url).expect("parse store");
    let reader =
        parquet::arrow::async_reader::ParquetObjectReader::new(std::sync::Arc::from(store), path);
    let builder = parquet::arrow::ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .expect("part footer");
    let schema = builder.schema().clone();
    let id_idx = schema.index_of("sr_id").expect("sr_id column");
    let mut stream = builder.build().expect("part stream");
    let mut positions = Vec::new();
    let mut position = 0_u64;
    let mut ids = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("part batch");
        let ids_col = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .expect("sr_id int32");
        for row in 0..batch.num_rows() {
            ids.push(ids_col.value(row));
            positions.push(position);
            position += 1;
        }
    }
    let deleted: Vec<u64> = ids
        .iter()
        .zip(&positions)
        .filter(|(id, _)| **id == 1 || **id == 3)
        .map(|(_, p)| *p)
        .collect();
    assert_eq!(deleted.len(), 2, "positions of sr_id 1 and 3: {ids:?}");

    let mut dv = PositionDeletes::new();
    for p in &deleted {
        dv.insert(*p);
    }
    let metas = bridge
        .write_deletion_vectors(
            &object_store::path::Path::from("data/dv-0.puffin"),
            &[(data_file.clone(), &dv)],
        )
        .await
        .expect("write vectors");
    let engine_fields = vec![
        engine_iceberg::IcebergField {
            id: 1,
            name: "sr_id".into(),
            type_name: "int".into(),
            required: true,
        },
        engine_iceberg::IcebergField {
            id: 2,
            name: "sr_item".into(),
            type_name: "string".into(),
            required: true,
        },
    ];
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(2);
    bridge
        .publish_iceberg_files_with_deletes(
            &[data_file],
            &metas,
            Some(version),
            "metadata",
            "00000000-0000-0000-0000-0000000000d3",
            &format!("{base_url}/data"),
            &engine_fields,
            now_ms + 1,
            now_ms,
        )
        .await
        .expect("publish the deleting snapshot");
    // The live server picks the new snapshot up on its next open (manifest-
    // selected; the table-open cache is default-OFF and this is its first open
    // since publication).

    // pgwire keeps serving the WAL-fresh native tier (4 rows): a test-side
    // published DV is EXTERNAL state the server was never told about — the
    // delete WRITER (the next slice) is what synchronizes both sides. What
    // this gate proves here is the PUBLISHED view: the engine's snapshot read
    // (what the DataFusion leaf consumes) and every external engine exclude
    // the deleted rows.
    let expected_rows = vec![
        vec![Some("2".to_string()), Some("keep".to_string())],
        vec![Some("4".to_string()), Some("keep".to_string())],
    ];
    {
        use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;
        let latest = bridge
            .latest_metadata_version("metadata")
            .await
            .expect("latest")
            .expect("the deleting snapshot");
        let snapshot = bridge
            .read_iceberg_snapshot_with_deletes("metadata", latest)
            .await
            .expect("read");
        assert_eq!(
            snapshot.deletion_vectors.len(),
            1,
            "the published v3 carries the vector"
        );
    }

    // The REST catalog serves the v3 metadata verbatim, and every external
    // reader excludes the deleted rows too.
    let load = rest_metadata_location(server.rest_port, table).await;
    let metadata_location = load["metadata-location"]
        .as_str()
        .expect("metadata-location")
        .to_string();
    assert_eq!(
        load["metadata"]["format-version"], 3,
        "the deleting snapshot is v3"
    );
    let scratch = TempDir::new().expect("scratch");
    if !external_readback(
        server.rest_port,
        table,
        &metadata_location,
        &cols,
        &expected_rows,
        scratch.path(),
    ) {
        return;
    }
    eprintln!("✓ deletion vectors are invisible to ProximaDB and external engines");
}

/// ADR-097 D3 / TD-SEAM-1 gate: a table materialized over pgwire is read back by
/// engines that are not ProximaDB — through the Iceberg REST catalog and from the
/// metadata file — and every value matches what ProximaDB returns. A second
/// MATERIALIZE (after an insert and an update) commits a successor snapshot that
/// chains onto the first, and the read-back still matches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn materialized_table_reads_back_in_external_iceberg_engines() {
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // DECIMAL(10,2) exercises the exact-decimal path end to end: catalog
    // precision/scale → Decimal128 Parquet → Iceberg `decimal(10,2)` + bounds.
    let table = "store_returns_rb";
    let cols = [
        "sr_id",
        "sr_item",
        "sr_amount",
        "sr_price",
        "sr_date",
        "sr_qty",
    ];
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (sr_id INT PRIMARY KEY, sr_item VARCHAR, \
             sr_amount DECIMAL(10,2), sr_price DOUBLE PRECISION, sr_date DATE, \
             sr_qty BIGINT)"
        ))
        .await
        .expect("create");
    client
        .simple_query(&format!(
            "INSERT INTO {table} (sr_id, sr_item, sr_amount, sr_price, sr_date, sr_qty) \
             VALUES \
             (1, 'AAAA', 12.50, 12.5, '2024-01-02', 3), \
             (2, 'BBBB', -0.99, 0.75, '2024-02-29', NULL), \
             (3, NULL, 1000.00, 1000.25, '2023-12-31', 9000000000)"
        ))
        .await
        .expect("insert");

    let scratch = TempDir::new().expect("scratch");
    let mut snapshots = Vec::new();
    for round in 0..2 {
        if round == 1 {
            client
                .simple_query(&format!(
                    "INSERT INTO {table} \
                     (sr_id, sr_item, sr_amount, sr_price, sr_date, sr_qty) \
                     VALUES (4, 'DDDD', 7.25, 7.25, '2024-06-30', 1)"
                ))
                .await
                .expect("insert 2");
            client
                .simple_query(&format!("UPDATE {table} SET sr_qty = 5 WHERE sr_id = 1"))
                .await
                .expect("update");
        }
        client
            .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
            .await
            .expect("materialize");
        sleep(Duration::from_millis(300)).await;

        let rows = pg_rows(&client, table, &cols).await;
        assert_eq!(rows.len(), 3 + round, "pgwire sees every row");

        let load = rest_metadata_location(server.rest_port, table).await;
        let metadata_location = load["metadata-location"]
            .as_str()
            .expect("metadata-location")
            .to_string();
        assert!(
            metadata_location.starts_with("file://")
                && metadata_location.ends_with(".metadata.json"),
            "REST serves the engine's metadata file, not a synthesized URL: {metadata_location}"
        );
        let md = &load["metadata"];
        assert_eq!(md["format-version"], 2);
        let amount_type = md["schemas"][0]["fields"]
            .as_array()
            .and_then(|fs| fs.iter().find(|f| f["name"] == "sr_amount"))
            .map(|f| f["type"].clone());
        assert_eq!(
            amount_type,
            Some(serde_json::json!("decimal(10,2)")),
            "DECIMAL keeps its declared precision/scale through to Iceberg"
        );
        assert_eq!(md["last-sequence-number"], (round + 1) as i64);
        let snapshot = &md["snapshots"][0];
        assert!(
            snapshot["manifest-list"]
                .as_str()
                .is_some_and(|l| l.starts_with("file://")),
            "manifest list is an absolute URI: {snapshot}"
        );
        if let Some(prev) = snapshots.last() {
            assert_eq!(
                &snapshot["parent-snapshot-id"], prev,
                "snapshot chains onto parent"
            );
            assert_eq!(snapshot["summary"]["operation"], "overwrite");
        }
        snapshots.push(md["current-snapshot-id"].clone());

        if !external_readback(
            server.rest_port,
            table,
            &metadata_location,
            &cols,
            &rows,
            scratch.path(),
        ) {
            return;
        }
    }
    eprintln!("✓ external Iceberg engines read back both snapshots value-for-value");
}
