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

/// Every file name under the server's published warehouse — the
/// artifact-assertion view shared by the live-delete gates.
fn warehouse_file_names(server: &TestPgServer) -> Vec<String> {
    let warehouse = server.data_dir().join("warehouse");
    let mut files = Vec::new();
    walk(&warehouse, &mut files);
    files
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect()
}

/// The table's (k, v) rows over pgwire, sorted — the WAL-view assertion
/// shared by the live-delete gates.
async fn sorted_kv_rows(client: &tokio_postgres::Client, table: &str) -> Vec<(i32, String)> {
    let rows = pg_rows(client, table, &["k", "v"]).await;
    let mut out: Vec<(i32, String)> = rows
        .iter()
        .map(|r| {
            (
                r[0].as_deref()
                    .and_then(|v| v.parse::<i32>().ok())
                    .unwrap_or(0),
                r[1].clone().unwrap_or_default(),
            )
        })
        .collect();
    out.sort();
    out
}

/// The gates' shared setup: a two-row (k, v) table — `(1,'keep')` and
/// `(2,'kill-me')` — published by one MATERIALIZE.
async fn create_materialized_kv(client: &tokio_postgres::Client, table: &str) {
    for sql in [
        format!("CREATE TABLE {table} (k INT PRIMARY KEY, v VARCHAR)"),
        format!("INSERT INTO {table} (k, v) VALUES (1, 'keep'), (2, 'kill-me')"),
        format!("ALTER TABLE {table} MATERIALIZE"),
    ] {
        client
            .simple_query(&sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    }
    sleep(Duration::from_millis(300)).await;
}

/// The gates' shared live delete of the `kill-me` row.
async fn delete_kill_me(client: &tokio_postgres::Client, table: &str) {
    client
        .simple_query(&format!("DELETE FROM {table} WHERE v = 'kill-me'"))
        .await
        .expect("delete");
    sleep(Duration::from_millis(300)).await;
}

/// The WAL view (pgwire, sorted) equals the model.
async fn assert_wal_matches(
    client: &tokio_postgres::Client,
    table: &str,
    model: &std::collections::BTreeMap<i32, String>,
    label: &str,
) {
    let got = sorted_kv_rows(client, table).await;
    let want: Vec<(i32, String)> = model.iter().map(|(k, v)| (*k, v.clone())).collect();
    assert_eq!(got, want, "{label}: the WAL view must match the model");
}

/// The external-reader leg shared by the live-delete gates and the v3
/// read-back test: resolve the table's current metadata location over the
/// REST catalog and read the (k, v) rows back through a pinned external
/// reader into `target/<scratch>`. Returns false (the leg skipped) only when
/// no reader is installed locally and none is required — CI sets the
/// REQUIRE env so the leg binds there.
async fn external_leg_kv(
    rest_port: u16,
    table: &str,
    rows: &[(&str, &str)],
    scratch: &str,
) -> bool {
    let load = rest_metadata_location(rest_port, table).await;
    let location = load["metadata-location"]
        .as_str()
        .expect("metadata-location")
        .to_string();
    let scratch = FsPath::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(scratch);
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let expected: Vec<Vec<Option<String>>> = rows
        .iter()
        .map(|(k, v)| vec![Some((*k).to_string()), Some((*v).to_string())])
        .collect();
    external_readback(
        rest_port,
        table,
        &location,
        &["k", "v"],
        &expected,
        &scratch,
    )
}

/// The live-delete publication artifact exists (round-1 lesson: assert the
/// fence FIRED, not just the outcome — a silent no-op passes outcome-only
/// gates).
fn assert_live_delete_artifact(server: &TestPgServer) {
    let names = warehouse_file_names(server);
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("dv-live-") && n.ends_with(".puffin")),
        "the live delete published a deletion-vector Puffin: {names:?}"
    );
}

async fn connect_pg(server: &TestPgServer) -> tokio_postgres::Client {
    let (client, connection) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("pgwire connection");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn materialize_emits_readable_iceberg_table() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

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

    // This legacy caller still selects Iceberg format-version 2. ADR-097's
    // v3-first activation is a separate gate after reader compatibility, not a
    // decision to develop new v2 features. The separate deletion test proves
    // external v3 readers; this fixture checks the legacy metadata/manifest chain.
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

/// P2.4 slice 3b (TD-SEAM-4): the PUBLISHED view of a snapshot carrying
/// deletion vectors is read delete-aware. The vectors are published test-side
/// through the engine's writer (the delete WRITER is the next slice and stays
/// unactivated); what this gate proves is that the published v3 snapshot
/// serves its post-delete rows — through the engine's snapshot read (the same
/// read the DataFusion leaf consumes; the leaf and the DuckDB view are
/// unit-tested over a real open) and through the external engines (PyIceberg
/// REST/static, DuckDB iceberg_scan — CI pins pyiceberg 0.12.0 and duckdb
/// 1.5.6, both read v3 + deletion vectors, and requires all three). Pgwire
/// aggregate paths are asserted too: cache/fallback/merge must preserve published
/// visibility or explicitly refuse when native and published ordering is unknown.
#[test]
fn deletion_vectors_are_invisible_to_proximadb_and_external_engines() {
    // Configure a child before its runtime starts: no process-global environment
    // mutation racing other tests or server workers. Exercise both caches and a
    // low native-provider cap through the real pgwire facade.
    if std::env::var_os("PROXIMADB_DV_FACADE_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "deletion_vectors_are_invisible_to_proximadb_and_external_engines",
                "--nocapture",
            ])
            .env("PROXIMADB_DV_FACADE_CHILD", "1")
            .env("PROXIMADB_QUERY_RESULT_CACHE", "1")
            .env("PROXIMADB_OLAP_TABLE_CACHE", "1")
            .env("PROXIMADB_NATIVE_TABLE_PROVIDER", "1")
            .env("PROXIMADB_NATIVE_TABLE_ROW_CAP", "1")
            .env("PROXIMADB_NATIVE_TABLE_LAYER_A", "0")
            .env("PROXIMADB_NATIVE_ROUTE", "1")
            .status()
            .expect("launch isolated facade regression");
        assert!(status.success(), "deletion-vector child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_deletion_vector_readers());
}

async fn check_deletion_vector_readers() {
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;
    use proximadb_iceberg_engine::deletion_vector::PositionDeletes;
    use proximadb_iceberg_engine::iceberg as engine_iceberg;
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;

    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

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

    client
        .simple_query("CREATE TABLE dv_native (id INT PRIMARY KEY)")
        .await
        .unwrap();
    client
        .simple_query("INSERT INTO dv_native (id) VALUES (1), (2)")
        .await
        .unwrap();
    let count_sql = format!("SELECT COUNT(*) FROM {table}");
    let count = |messages: Vec<tokio_postgres::SimpleQueryMessage>| {
        messages
            .into_iter()
            .find_map(|message| match message {
                tokio_postgres::SimpleQueryMessage::Row(row) => row.get(0).map(str::to_string),
                _ => None,
            })
            .expect("aggregate row")
    };
    assert_eq!(count(client.simple_query(&count_sql).await.unwrap()), "4");

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
    // The workspace uses this object_store adapter until its AsyncFileReader
    // replacement lands (same compatibility boundary as the production reader).
    #[allow(deprecated)]
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
            std::slice::from_ref(&data_file),
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
    // No WAL write occurred between these identical aggregate queries. A cache
    // keyed by WAL LSN alone would still return four, and native COUNT shortcuts
    // would trust the pre-delete footer. Published visibility must win.
    assert_eq!(count(client.simple_query(&count_sql).await.unwrap()), "2");
    let mixed_error = client
        .simple_query("SELECT COUNT(*) FROM dv_returns d JOIN dv_native n ON d.sr_id = n.id")
        .await
        .expect_err("row-cap fallback must not change published visibility");
    assert!(
        pg_server::db_err(&mixed_error).contains("cannot fall back to native storage"),
        "{mixed_error:?}"
    );

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

    // Exercise the actual DataFusion provider, not just the manifest decoder.
    let ctx = datafusion::prelude::SessionContext::new();
    proximadb::datafusion::engine_adapters::register_object_store_parquet_location(
        &ctx,
        "dv_published",
        &format!("{base_url}/data"),
        None,
        proximadb_data_model::StatsTrust::Trusted,
    )
    .await
    .expect("register deleting snapshot");
    let actual = ctx
        .sql("SELECT sr_id FROM dv_published ORDER BY sr_id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = actual
        .iter()
        .flat_map(|batch| {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            (0..batch.num_rows())
                .map(|i| col.value(i))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(ids, vec![2, 4]);

    // Exercise our optional DuckDB comparison route as well as the external
    // iceberg_scan reader. They have different file-selection adapters.
    #[cfg(feature = "duckdb")]
    {
        use proximadb::query::execution::{
            duckdb_engine::DuckDbLocalEngine,
            engine::{ExecutionEngine, QueryExecutionContext},
        };
        // Publish a separate fixture with a consistently encoded root. Changing
        // only an existing manifest's URL spelling is not a supported alias.
        let encoded_tmp = TempDir::new().unwrap();
        let encoded_dir = encoded_tmp.path().join("encoded root");
        std::fs::create_dir_all(encoded_dir.join("data")).unwrap();
        std::fs::copy(
            base_dir.join(data_file.as_ref()),
            encoded_dir.join(data_file.as_ref()),
        )
        .unwrap();
        let encoded_url = url::Url::from_directory_path(&encoded_dir)
            .unwrap()
            .to_string();
        let encoded_bridge = IcebergObjectStoreBridge::from_url(&encoded_url).unwrap();
        let encoded_deletes = encoded_bridge
            .write_deletion_vectors(
                &object_store::path::Path::from("data/dv.puffin"),
                &[(data_file.clone(), &dv)],
            )
            .await
            .unwrap();
        encoded_bridge
            .publish_iceberg_files_with_deletes(
                std::slice::from_ref(&data_file),
                &encoded_deletes,
                None,
                "metadata",
                "00000000-0000-0000-0000-0000000000e3",
                &format!("{}data", encoded_url),
                &engine_fields,
                now_ms + 10,
                now_ms,
            )
            .await
            .unwrap();
        for location in [format!("{base_url}/data"), format!("{encoded_url}data")] {
            for (sql, expected) in [
                (
                    "SELECT CAST(sr_id AS BIGINT) FROM t ORDER BY sr_id",
                    vec![2, 4],
                ),
                (
                    "SELECT CAST(sr_id AS BIGINT) FROM t WHERE sr_id > 2 LIMIT 1",
                    vec![4],
                ),
                ("SELECT COUNT(*) FROM t", vec![2]),
            ] {
                let result = DuckDbLocalEngine
                    .execute_sql(
                        sql,
                        QueryExecutionContext {
                            parquet_tables: vec![("t".into(), location.clone())],
                            ..Default::default()
                        },
                    )
                    .await
                    .expect("DuckDB comparison view");
                let values: Vec<i64> = result
                    .rows
                    .iter()
                    .map(|row| match row.first() {
                        Some(proximadb_records::ProximaValue::Int64(value)) => *value,
                        other => panic!("expected BIGINT, got {other:?}"),
                    })
                    .collect();
                assert_eq!(values, expected, "{sql}");
            }
        }
    }

    // Republishing unsynchronized native state must not resurrect hidden rows.
    let err = client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect_err("refuse native rewrite of externally deleted snapshot");
    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("deletion vectors"),
        "{err:?}"
    );
    assert_eq!(
        bridge.latest_metadata_version("metadata").await.unwrap(),
        Some(version + 1)
    );

    // A native update predating a newer external DV publication must not be
    // appended back onto that snapshot by the default-on WAL-tail merge.
    client
        .simple_query("UPDATE dv_returns SET sr_item = 'updated-before-delete' WHERE sr_id = 1")
        .await
        .unwrap();
    bridge
        .publish_iceberg_files_with_deletes(
            std::slice::from_ref(&data_file),
            &metas,
            Some(version + 1),
            "metadata",
            "00000000-0000-0000-0000-0000000000d3",
            &format!("{base_url}/data"),
            &engine_fields,
            now_ms + 2,
            now_ms + 1,
        )
        .await
        .expect("publish DV after native update");
    let merge_error = client
        .simple_query(&count_sql)
        .await
        .expect_err("unordered delta cannot resurrect DV-hidden rows");
    assert!(
        pg_server::db_err(&merge_error).contains("nonempty WAL delta cannot be ordered"),
        "{merge_error:?}"
    );

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
    let client = connect_pg(&server).await;

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

/// TD-USUB-10 3c rung 2 — a **plain v3 table through the rung-1 knob**: the
/// first publication commits format-version 3 with row lineage, no deletes, no
/// `_ns` columns — and pinned external readers (pyiceberg static+REST; DuckDB
/// when installed) read the values back. The knob runs in a child process
/// (the same isolation pattern as the DV test) so no process-global env leaks.
#[test]
fn v3_default_first_publication_reads_back_externally() {
    if std::env::var_os("PROXIMADB_V3_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "v3_default_first_publication_reads_back_externally",
                "--nocapture",
            ])
            .env("PROXIMADB_V3_CHILD", "1")
            .env("PROXIMADB_ICEBERG_V3_DEFAULT", "1")
            .status()
            .expect("launch isolated v3-default child");
        assert!(status.success(), "v3-default child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_v3_default_publication());
}

async fn check_v3_default_publication() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = "v3_plain";
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (k INT PRIMARY KEY, v VARCHAR)"
        ))
        .await
        .expect("create");
    client
        .simple_query(&format!(
            "INSERT INTO {table} (k, v) VALUES (1, 'one'), (2, 'two'), (3, 'three')"
        ))
        .await
        .expect("insert");
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize");
    sleep(Duration::from_millis(300)).await;

    // 1. The published metadata is v3 with row lineage allocated.
    let warehouse = server.data_dir().join("warehouse");
    let mut files = Vec::new();
    walk(&warehouse, &mut files);
    let metadata_json: Vec<&PathBuf> = files
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".metadata.json"))
        })
        .filter(|p| {
            // this test's table only (the server owns a fresh tempdir, but the
            // filter keeps the walk honest if a fixture ever shares it)
            std::fs::read_to_string(p)
                .unwrap_or_default()
                .contains("v3_plain")
        })
        .collect();
    assert!(
        !metadata_json.is_empty(),
        "the v3_plain table published metadata.json; files walked={:?}",
        files.len()
    );
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(metadata_json[0]).expect("read metadata.json"))
            .expect("parse metadata.json");
    assert_eq!(
        meta["format-version"], 3,
        "the knob made the FIRST publication v3"
    );
    assert!(
        meta["next-row-id"].as_i64().is_some_and(|v| v > 0),
        "row lineage allocated: next-row-id={} (was allocated, not defaulted)",
        meta["next-row-id"]
    );

    // 2. A second publication (append) chains onto the parent and stays v3.
    client
        .simple_query(&format!("INSERT INTO {table} (k, v) VALUES (4, 'four')"))
        .await
        .expect("insert 2");
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize 2");
    sleep(Duration::from_millis(300)).await;

    // 3. The SECOND metadata document chains correctly (the ladder's
    // sequence/parent-chain proof): parent-snapshot-id is the first
    // publication's snapshot, the sequence advanced, and row lineage grew.
    let mut files2 = Vec::new();
    walk(&warehouse, &mut files2);
    let mut v3_docs: Vec<serde_json::Value> = files2
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".metadata.json"))
        })
        .filter_map(|p| -> Option<serde_json::Value> {
            let bytes = std::fs::read(p).ok()?;
            serde_json::from_slice(&bytes).ok()
        })
        .filter(|m: &serde_json::Value| m.to_string().contains("v3_plain"))
        .collect();
    assert_eq!(
        v3_docs.len(),
        2,
        "two metadata versions after two publications"
    );
    v3_docs.sort_by_key(|m| m["last-updated-ms"].as_i64().unwrap_or(0));
    let (first_doc, second_doc) = (&v3_docs[0], &v3_docs[1]);
    let parent_snapshot = first_doc["current-snapshot-id"].clone();
    let second_snapshot = second_doc["snapshots"]
        .as_array()
        .and_then(|snaps| snaps.last())
        .cloned()
        .expect("the second snapshot is listed");
    assert_eq!(
        second_snapshot["parent-snapshot-id"], parent_snapshot,
        "the second snapshot chains onto the first"
    );
    assert!(
        second_doc["last-sequence-number"].as_i64().unwrap_or(0)
            > first_doc["last-sequence-number"].as_i64().unwrap_or(0),
        "the sequence number advanced"
    );
    assert!(
        second_doc["next-row-id"].as_i64().unwrap_or(0)
            > first_doc["next-row-id"].as_i64().unwrap_or(0),
        "row lineage grew with the appended rows"
    );
    assert_eq!(second_doc["format-version"], 3, "the successor stays v3");

    // 4. pgwire sees every value after two publications.
    let got = sorted_kv_rows(&client, &table).await;
    assert_eq!(
        got,
        vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
            (4, "four".to_string()),
        ],
        "pgwire sees all four rows after two publications"
    );

    // 4. A pinned external reader reads the knob-published v3 table back.
    // The reader gate is skipped (returns false) when no reader is installed
    // locally and none is REQUIRED — CI sets the REQUIRE env so the gate is
    // binding there, matching the sibling tests' pattern.
    if !external_leg_kv(
        server.rest_port,
        &table,
        &[("1", "one"), ("2", "two"), ("3", "three"), ("4", "four")],
        "iceberg-v3-plain",
    )
    .await
    {
        eprintln!(
            "SKIP external read-back (no reader installed): v3 metadata + lineage assertions still ran"
        );
        return;
    }
    eprintln!("✓ v3-default first publication: v3 + row lineage + external read-back");
}

/// TD-USUB-10 3c writer, gate 1 of the visibility fence contract
/// (docs/12-design/PUBLICATION_WAL_VISIBILITY_FENCE_2026_10_27.adoc): a live
/// DELETE over pgwire crosses the fence — the row is hidden from pgwire (the
/// phase-2 leg: the WAL hides it regardless) AND from an external reader (the
/// fence-specific leg: the DV-carrying successor snapshot served bare). The
/// publisher is default-OFF; this test arms it in an isolated child.
#[test]
fn live_delete_publishes_and_both_views_hide_the_row() {
    if std::env::var_os("PROXIMADB_LIVE_DEL_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "live_delete_publishes_and_both_views_hide_the_row",
                "--nocapture",
            ])
            .env("PROXIMADB_LIVE_DEL_CHILD", "1")
            .env("PROXIMADB_LIVE_DELETE_PUBLISHER", "1")
            .status()
            .expect("launch isolated live-delete child");
        assert!(status.success(), "live-delete child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_live_delete_gate1());
}

async fn check_live_delete_gate1() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = "live_del";
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (k INT PRIMARY KEY, v VARCHAR)"
        ))
        .await
        .expect("create");
    client
        .simple_query(&format!(
            "INSERT INTO {table} (k, v) VALUES (1, 'keep'), (2, 'kill-me'), (3, 'also-keep')"
        ))
        .await
        .expect("insert");
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize");
    sleep(Duration::from_millis(300)).await;

    // The fence: the live delete (publisher armed) publishes a DV successor.
    delete_kill_me(&client, table).await;

    // Phase-2 leg: pgwire hides the row (WAL-authoritative regardless).
    let k1 = sorted_kv_rows(&client, &table).await;
    assert_eq!(
        k1,
        vec![(1, "keep".to_string()), (3, "also-keep".to_string())],
        "pgwire (WAL view) hides the deleted row"
    );

    // Round-1 review (blocker 1): assert the PUBLICATION ARTIFACTS locally —
    // the fence actually fired, not a silent no-op: a new metadata version and
    // the live-delete Puffin exist under the table's metadata/data dirs.
    assert_live_delete_artifact(&server);
    let names = warehouse_file_names(&server);
    assert!(
        names
            .iter()
            .filter(|n| n.ends_with(".metadata.json"))
            .count()
            >= 2,
        "the live delete published a successor metadata version: {names:?}"
    );

    // Fence-specific leg: the external reader serves the DV-carrying successor
    // bare — the deleted row is gone from the PUBLISHED view too.
    if !external_leg_kv(
        server.rest_port,
        &table,
        &[("1", "keep"), ("3", "also-keep")],
        "iceberg-live-del",
    )
    .await
    {
        eprintln!("SKIP external leg (no reader installed locally); CI requires it");
        return;
    }
    eprintln!("✓ live delete crossed the fence: both views hide the row");
}

/// TD-USUB-10 3c writer, gate 2 of the fence contract: the fence HOLDS across
/// a successor — after the live delete publishes the DV snapshot, a subsequent
/// append + re-publication carries the vectors forward; the deleted row stays
/// hidden in the newest snapshot (guarantee 1: no resurrection).
#[test]
fn live_delete_survives_a_successor_publication() {
    if std::env::var_os("PROXIMADB_LIVE_DEL2_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "live_delete_survives_a_successor_publication",
                "--nocapture",
            ])
            .env("PROXIMADB_LIVE_DEL2_CHILD", "1")
            .env("PROXIMADB_LIVE_DELETE_PUBLISHER", "1")
            .status()
            .expect("launch isolated live-delete-2 child");
        assert!(status.success(), "live-delete-2 child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_live_delete_gate2());
}

async fn check_live_delete_gate2() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = "live_del2";
    create_materialized_kv(&client, table).await;

    delete_kill_me(&client, table).await;

    // The successor publication: an insert + re-MATERIALIZE. Review round 3
    // (finding 5): plan_append refuses a DV-carrying parent, so this
    // re-MATERIALIZE takes the REWRITE path — the relaxed guard admits it
    // (its capture LSN is at or above the recorded fence LSN, so the
    // tombstone is visible to the dead-filtered native scan) and the row
    // stays dead by native filtering, not by DV carry-through. The
    // no-resurrection OUTCOME across a successor is what this gate pins;
    // the append-carry branch itself is exercised by the bulk-lane suite.
    client
        .simple_query(&format!("INSERT INTO {table} (k, v) VALUES (3, 'later')"))
        .await
        .expect("insert 2");
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize 2");
    sleep(Duration::from_millis(300)).await;

    // Round-1 review (minor 1): prove THIS test's live publication fired
    // (the dv-live Puffin exists) so the carry assertion below is
    // non-vacuous — without it, gate 2 passes WAL-only even if the
    // publisher never publishes.
    assert_live_delete_artifact(&server);

    let got = sorted_kv_rows(&client, &table).await;
    assert_eq!(
        got,
        vec![(1, "keep".to_string()), (3, "later".to_string())],
        "the deleted row stays dead across the successor publication"
    );

    // Round-4 review (minor 2): the gate's property is about the PUBLISHED
    // view — the WAL view hides the row regardless of any resurrection, so
    // the leg above alone cannot fail on the defect this gate exists to
    // pin. Read the successor snapshot back through the external reader.
    if !external_leg_kv(
        server.rest_port,
        &table,
        &[("1", "keep"), ("3", "later")],
        "iceberg-live-del2",
    )
    .await
    {
        eprintln!("SKIP external leg (no reader installed locally); CI requires it");
        return;
    }
    eprintln!("✓ gate 2: the successor's published view keeps the row dead");
}

/// Gate 3 of the fence contract (race), forced deterministically: the
/// debug-only competitor hook commits a legitimate no-change successor from
/// the same parent and flips the layout BEFORE the publisher's CAS — the
/// state a concurrent publisher that completed first leaves. The publisher's
/// REAL CAS then fires against the taken slot and must LOSE it (the engine's
/// AlreadyExists/"moved past" path — the retry's error detection, exercised
/// for real), and the bounded retry must re-read the new parent, re-map
/// positions, and still close the fence. The WAL view is asserted directly;
/// the published view is inferred from the ARTIFACTS (>= 3 metadata versions
/// — materialize, competitor, final — plus the dv-live Puffin) — no sleeps,
/// no true concurrency.
#[test]
fn live_delete_loses_a_cas_race_and_the_retry_closes_the_fence() {
    if std::env::var_os("PROXIMADB_LIVE_DEL3_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "live_delete_loses_a_cas_race_and_the_retry_closes_the_fence",
                "--nocapture",
            ])
            .env("PROXIMADB_LIVE_DEL3_CHILD", "1")
            .env("PROXIMADB_LIVE_DELETE_PUBLISHER", "1")
            .env("PROXIMADB_TEST_FENCE_INJECT_COMPETITOR", "1")
            .status()
            .expect("launch isolated gate-3 child");
        assert!(status.success(), "gate-3 child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_live_delete_gate3());
}

async fn check_live_delete_gate3() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = "live_del3";
    create_materialized_kv(&client, table).await;

    delete_kill_me(&client, table).await;

    // The retry path ran: the competitor and the final successor are BOTH
    // committed after the materialize (>= 3 metadata versions total).
    assert_live_delete_artifact(&server);
    let names = warehouse_file_names(&server);
    assert!(
        names
            .iter()
            .filter(|n| n.ends_with(".metadata.json"))
            .count()
            >= 3,
        "the competitor AND the retried successor committed: {names:?}"
    );

    let got = sorted_kv_rows(&client, &table).await;
    assert_eq!(
        got,
        vec![(1, "keep".to_string())],
        "the fence closes across the lost race: the row stays hidden"
    );
}

/// Gate 5 of the fence contract (crash-gap): the test fault lands the
/// successor CAS and skips the layout flip — the state a pod crash between
/// the two leaves. The wedge must be FAIL-CLOSED (reads refuse rather than
/// serve wrong data), and a later MATERIALIZE must ADOPT the orphaned
/// publication (its dv-live Puffin proves it is ours; its tombstones are
/// durable by the contract's phase ordering) and heal the table.
#[test]
fn crash_gap_is_fail_closed_and_materialize_adopts() {
    if std::env::var_os("PROXIMADB_LIVE_DEL4_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "crash_gap_is_fail_closed_and_materialize_adopts",
                "--nocapture",
            ])
            .env("PROXIMADB_LIVE_DEL4_CHILD", "1")
            .env("PROXIMADB_LIVE_DELETE_PUBLISHER", "1")
            .env("PROXIMADB_TEST_FENCE_CRASH_BEFORE_FLIP", "1")
            .status()
            .expect("launch isolated gate-5 child");
        assert!(status.success(), "gate-5 child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_live_delete_gate5());
}

async fn check_live_delete_gate5() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = "live_del4";
    create_materialized_kv(&client, table).await;

    // The publication lands; the fault skips the flip (the crash window).
    delete_kill_me(&client, table).await;
    assert_live_delete_artifact(&server);

    // The wedge is fail-closed: the layout still records the pre-delete
    // publication, the delta is nonempty (the tombstone), and the base
    // carries DVs — the OLAP read REFUSES rather than guess.
    let err = client
        .simple_query(&format!("SELECT COUNT(*) FROM {table}"))
        .await
        .expect_err("the crash-gap wedge must refuse reads");
    assert!(
        format!("{err:?}").contains("deletion-vector")
            || format!("{err:?}").contains("deletion vectors"),
        "the refusal names the unordered-delta condition: {err:?}"
    );

    // Recovery: a MATERIALIZE adopts the orphaned publication (the dv-live
    // Puffin proves it is ours) and heals the table.
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize adopts the orphaned publication");
    sleep(Duration::from_millis(300)).await;

    let got = sorted_kv_rows(&client, &table).await;
    assert_eq!(
        got,
        vec![(1, "keep".to_string())],
        "adoption heals the wedge and the row stays dead"
    );
}

/// Runs `body` in an isolated child process with `envs` set (the publisher's
/// knobs are process-wide). Returns in the parent after asserting the child
/// passed; runs the async body on a fresh runtime in the child.
fn run_isolated<F, Fut>(test: &str, child_var: &str, envs: &[(&str, &str)], body: F)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    if std::env::var_os(child_var).is_none() {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", test, "--nocapture"])
            .env(child_var, "1");
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let status = cmd.status().expect("launch isolated child");
        assert!(status.success(), "{test} child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(body());
}

/// Gate 6 of the fence contract, as relaxed by the ordered-DV-base rule: after
/// a live delete closes the fence, the DV-carrying base IS the layout's
/// recorded publication, so later WAL writes are MERGED over it (no refusal,
/// no MATERIALIZE needed) — an insert appears and a delete of a base row
/// disappears through the merge.
#[test]
fn ordered_dv_base_serves_the_merge_across_the_fence() {
    run_isolated(
        "ordered_dv_base_serves_the_merge_across_the_fence",
        "PROXIMADB_LIVE_DEL5_CHILD",
        &[("PROXIMADB_LIVE_DELETE_PUBLISHER", "1")],
        || check_live_delete_gate6(true),
    );
}

/// The kill switch (`PROXIMADB_OLAP_ORDERED_DV_BASE=0`) restores the original
/// unconditional refusal of a non-empty delta against a DV base.
#[test]
fn ordered_dv_base_kill_switch_restores_the_refusal() {
    run_isolated(
        "ordered_dv_base_kill_switch_restores_the_refusal",
        "PROXIMADB_LIVE_DEL6_CHILD",
        &[
            ("PROXIMADB_LIVE_DELETE_PUBLISHER", "1"),
            ("PROXIMADB_OLAP_ORDERED_DV_BASE", "0"),
        ],
        || check_live_delete_gate6(false),
    );
}

async fn check_live_delete_gate6(ordered: bool) {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = if ordered { "live_del5" } else { "live_del6" };
    create_materialized_kv(&client, table).await;
    delete_kill_me(&client, table).await;
    assert_live_delete_artifact(&server);
    // Empty delta against the DV base: served bare, the row hidden. The
    // AGGREGATE routes the query to the DataFusion/OLAP path (the delta
    // merge); a plain projection can take the native/WAL scan, which never
    // consults the DV base.
    assert_eq!(
        scalar_count(&client, table).await,
        1,
        "the bare DV base hides the deleted row (delta-merge path)"
    );

    // A post-fence WAL write makes the delta non-empty.
    client
        .simple_query(&format!("INSERT INTO {table} (k, v) VALUES (3, 'later')"))
        .await
        .expect("insert");
    if ordered {
        // The base is the recorded publication: the merge serves it.
        assert_eq!(
            scalar_count(&client, table).await,
            2,
            "the insert is merged over the ordered DV base, no MATERIALIZE"
        );
        // A delete of a BASE row is suppressed through the merge as well.
        client
            .simple_query(&format!("DELETE FROM {table} WHERE k = 1"))
            .await
            .expect("delete a base row");
        assert_eq!(
            scalar_count(&client, table).await,
            1,
            "a base-row delete is suppressed through the merge"
        );
        let got = sorted_kv_rows(&client, table).await;
        assert_eq!(got, vec![(3, "later".to_string())], "the WAL view agrees");
    } else {
        let err = client
            .simple_query(&format!("SELECT COUNT(*) FROM {table}"))
            .await
            .expect_err("with the kill switch, a non-empty delta against a DV base refuses");
        assert!(
            format!("{err:?}").contains("deletion-vector"),
            "the refusal names the unordered-delta condition: {err:?}"
        );
    }

    // The next full publication reconciles either way.
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("materialize reconciles");
    sleep(Duration::from_millis(300)).await;
    let want = if ordered { 1 } else { 2 };
    assert_eq!(
        scalar_count(&client, table).await,
        want,
        "reconciliation serves the full correct state"
    );
}

/// COUNT(*) over pgwire as an i64 — the OLAP-shaped probe gate 6 uses to
/// force the DataFusion/delta-merge path. A malformed result PANICS here
/// (naming the real defect) rather than returning a sentinel.
async fn scalar_count(client: &tokio_postgres::Client, table: &str) -> i64 {
    let rows = pg_rows(client, table, &["COUNT(*)"]).await;
    let raw = rows
        .first()
        .and_then(|r| r.first())
        .and_then(|v| v.as_deref())
        .unwrap_or_else(|| panic!("COUNT(*) returned no scalar for {table}"));
    raw.trim()
        .parse::<i64>()
        .unwrap_or_else(|_| panic!("COUNT(*) returned a non-integer: {raw:?}"))
}

/// A tiny deterministic LCG so the soak's operation mix is reproducible
/// (no flaky randomness; every run issues the same sequence).
struct SoakRng(u64);
impl SoakRng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Soak evidence for the activation flip (the contract's "six gates plus
/// soak"), with the publisher ARMED and a seeded LCG driving the mix:
/// - Phase A, interleaved inserts/deletes with a MATERIALIZE every 4th
///   round: the WAL view equals the model after every round, the OLAP view
///   reconciles at every MATERIALIZE, and ZERO publications happen (the
///   completeness precondition refuses every DV-only publication while an
///   insert is pending — TD-LIVEDEL-1's regression baseline, pinned);
/// - Phase B, a delete chain right after a reconciling MATERIALIZE: every
///   delete publishes (one dv-live Puffin AND one metadata version each)
///   and the OLAP view tracks it with no MATERIALIZE;
/// - Phase C, an insert on top of the DV-carrying base: the OLAP read merges
///   it over the base (the base IS the recorded publication), and a
///   MATERIALIZE over the DV parent (the relaxed entry guard) reconciles
///   both views — no resurrection.
#[test]
fn soak_mixed_workload_holds_the_fence() {
    if std::env::var_os("PROXIMADB_SOAK_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "soak_mixed_workload_holds_the_fence",
                "--nocapture",
            ])
            .env("PROXIMADB_SOAK_CHILD", "1")
            .env("PROXIMADB_LIVE_DELETE_PUBLISHER", "1")
            .status()
            .expect("launch isolated soak child");
        assert!(status.success(), "soak child failed: {status}");
        return;
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_soak_mixed_workload());
}

async fn check_soak_mixed_workload() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;

    let table = "soak";
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (k INT PRIMARY KEY, v VARCHAR)"
        ))
        .await
        .expect("create");
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("initial materialize");
    sleep(Duration::from_millis(300)).await;

    let mut rng = SoakRng(0x5EED_2026_1028);
    let mut model: std::collections::BTreeMap<i32, String> = Default::default();
    let mut next_key: i32 = 1;
    let mut live_deletes = 0usize;
    let mut materializes = 1usize;

    for round in 0..40u64 {
        // Seeded insert batch (1..=3 new keys).
        let inserts = 1 + rng.below(3);
        for _ in 0..inserts {
            let k = next_key;
            next_key += 1;
            let v = format!("v{k}");
            client
                .simple_query(&format!("INSERT INTO {table} (k, v) VALUES ({k}, '{v}')"))
                .await
                .expect("insert");
            model.insert(k, v);
        }

        // Seeded delete batch (0..=2 live keys, when any exist).
        let deletes = rng.below(3);
        for _ in 0..deletes {
            if model.is_empty() {
                break;
            }
            let idx = rng.below(model.len() as u64) as usize;
            let k = *model.keys().nth(idx).expect("indexed key");
            client
                .simple_query(&format!("DELETE FROM {table} WHERE k = {k}"))
                .await
                .expect("delete");
            model.remove(&k);
            live_deletes += 1;
        }

        // The WAL view equals the model after every round.
        assert_wal_matches(&client, table, &model, &format!("round {round}")).await;
        // Every 4th round: reconcile and assert BOTH views.
        if round % 4 == 3 {
            client
                .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
                .await
                .expect("materialize");
            materializes += 1;
            sleep(Duration::from_millis(200)).await;
            let n = scalar_count(&client, &table).await;
            assert_eq!(
                n,
                model.len() as i64,
                "round {round}: the OLAP view must reconcile to the model"
            );
        }
    }

    // Phase B — the live fence itself. Phase A's interleaved inserts make
    // the completeness precondition refuse every DV-only publication (a
    // pending insert is in neither the parent's files nor a DV), so the
    // fence is exercised here in its eligible shape: right after a
    // reconciling MATERIALIZE, a chain of deletes with no other writes.
    // Each delete must publish (one dv-live Puffin per delete) and the
    // OLAP view must track the model WITHOUT another MATERIALIZE.
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("reconciling materialize");
    sleep(Duration::from_millis(300)).await;
    assert_eq!(
        scalar_count(&client, &table).await as usize,
        model.len(),
        "phase B starts reconciled"
    );
    let count_files = |server: &TestPgServer, pred: fn(&str) -> bool| {
        warehouse_file_names(server)
            .iter()
            .filter(|n| pred(n))
            .count()
    };
    let is_puffin: fn(&str) -> bool = |n| n.starts_with("dv-live-") && n.ends_with(".puffin");
    let is_version: fn(&str) -> bool = |n| n.ends_with(".metadata.json");
    // Phase A published nothing (TD-LIVEDEL-1's regression baseline): every
    // delete there had a pending insert in the changed-set.
    assert_eq!(
        count_files(&server, is_puffin),
        0,
        "phase A: interleaved inserts refuse every DV-only publication"
    );
    let puffins_before = count_files(&server, is_puffin);
    let versions_before = count_files(&server, is_version);
    let chain = model.len().min(5);
    assert!(chain >= 3, "phase A left enough rows for a delete chain");
    for step in 0..chain {
        let k = *model.keys().next().expect("a live key");
        client
            .simple_query(&format!("DELETE FROM {table} WHERE k = {k}"))
            .await
            .expect("chained delete");
        model.remove(&k);
        sleep(Duration::from_millis(200)).await;
        assert_eq!(
            count_files(&server, is_puffin),
            puffins_before + step + 1,
            "phase B step {step}: the delete wrote its own Puffin"
        );
        assert_eq!(
            count_files(&server, is_version),
            versions_before + step + 1,
            "phase B step {step}: the delete committed its own metadata version"
        );
        assert_eq!(
            scalar_count(&client, &table).await as usize,
            model.len(),
            "phase B step {step}: the OLAP view tracks the delete with no MATERIALIZE"
        );
        assert_wal_matches(&client, table, &model, &format!("phase B step {step}")).await;
    }

    // Phase C — an insert on top of the DV-carrying base: the delta is now
    // non-empty against a base with vectors, and the base IS the recorded
    // publication, so the OLAP read merges it; a MATERIALIZE over the DV
    // parent (the relaxed entry guard) then reconciles both views — the
    // vector-carrying reconciliation path.
    let k = next_key;
    client
        .simple_query(&format!("INSERT INTO {table} (k, v) VALUES ({k}, 'v{k}')"))
        .await
        .expect("phase C insert");
    model.insert(k, format!("v{k}"));
    assert_wal_matches(&client, table, &model, "phase C after insert").await;
    // The base is the recorded publication (ordered DV base): the insert is
    // merged over it, no refusal and no MATERIALIZE needed.
    assert_eq!(
        scalar_count(&client, &table).await as usize,
        model.len(),
        "phase C: the insert is merged over the ordered DV base"
    );
    client
        .simple_query(&format!("ALTER TABLE {table} MATERIALIZE"))
        .await
        .expect("phase C: MATERIALIZE over the DV parent");
    materializes += 1;
    sleep(Duration::from_millis(300)).await;
    assert_eq!(
        scalar_count(&client, &table).await as usize,
        model.len(),
        "phase C: the OLAP view reconciles over the DV parent"
    );
    assert_wal_matches(&client, table, &model, "phase C after MATERIALIZE").await;
    eprintln!(
        "soak ok: phase A {live_deletes} deletes under interleaved writes (publication refused \
         by the completeness precondition, both views exact); phase B {chain} chained deletes \
         -> {chain} publications, OLAP view live without MATERIALIZE; phase C DV-base \
         reconciliation; {materializes} materializes"
    );
}

/// TD-USUB-10 / TD-LIVEDEL-1: with `PROXIMADB_MATERIALIZE_INCREMENTAL_DV`, a
/// MATERIALIZE after an UPDATE and a DELETE republishes INCREMENTALLY through
/// the real pgwire facade: the published file is carried (not rewritten), the
/// update's after-image lands in a new part, and both old row versions are
/// hidden by one `dv-mat-*` deletion vector. Both views and the external
/// engines agree; a later INSERT is merged over the DV-carrying base (the
/// ordered-DV-base relaxation, #2277); an insert-only MATERIALIZE then
/// appends while carrying the vectors.
#[test]
fn incremental_materialize_publishes_updates_and_deletes_as_vectors() {
    run_isolated(
        "incremental_materialize_publishes_updates_and_deletes_as_vectors",
        "PROXIMADB_INCR_MAT_CHILD",
        &[
            ("PROXIMADB_MATERIALIZE_INCREMENTAL_DV", "1"),
            ("PROXIMADB_PARQUET_WRITE_BLOOM_MIN_ROWS", "1"),
        ],
        check_incremental_materialize,
    );
}

async fn check_incremental_materialize() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;
    let table = "incr_mat";
    let exec = |sql: String| {
        let client = &client;
        async move {
            client
                .simple_query(&sql)
                .await
                .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        }
    };
    exec(format!(
        "CREATE TABLE {table} (k INT PRIMARY KEY, v VARCHAR)"
    ))
    .await;
    // Ten rows: one update plus one delete is 20% — exactly the compaction
    // bound, so the republication stays incremental.
    let mut model: std::collections::BTreeMap<i32, String> = Default::default();
    for k in 1..=10 {
        exec(format!("INSERT INTO {table} (k, v) VALUES ({k}, 'v{k}')")).await;
        model.insert(k, format!("v{k}"));
    }
    exec(format!("ALTER TABLE {table} MATERIALIZE")).await;
    sleep(Duration::from_millis(300)).await;
    let parquet = |server: &TestPgServer| -> Vec<String> {
        warehouse_file_names(server)
            .into_iter()
            .filter(|n| n.ends_with(".parquet"))
            .collect()
    };
    let first = parquet(&server);
    assert_eq!(first.len(), 1, "one published part: {first:?}");

    exec(format!("UPDATE {table} SET v = 'updated' WHERE k = 1")).await;
    model.insert(1, "updated".to_string());
    exec(format!("DELETE FROM {table} WHERE k = 2")).await;
    model.remove(&2);
    exec(format!("ALTER TABLE {table} MATERIALIZE")).await;
    sleep(Duration::from_millis(300)).await;

    let second = parquet(&server);
    assert!(
        second.contains(&first[0]),
        "the published part is CARRIED, not rewritten: {second:?}"
    );
    assert_eq!(
        second.len(),
        2,
        "carried + one after-image part: {second:?}"
    );
    let minted: Vec<String> = warehouse_file_names(&server)
        .into_iter()
        .filter(|n| n.starts_with("dv-mat-") && n.ends_with(".puffin"))
        .collect();
    assert_eq!(minted.len(), 1, "one dv-mat Puffin: {minted:?}");
    assert_eq!(scalar_count(&client, table).await as usize, model.len());
    assert_wal_matches(&client, table, &model, "after the incremental publication").await;

    // A later write is MERGED over the DV-carrying base (no refusal).
    exec(format!("INSERT INTO {table} (k, v) VALUES (11, 'v11')")).await;
    model.insert(11, "v11".to_string());
    assert_eq!(
        scalar_count(&client, table).await as usize,
        model.len(),
        "the insert is merged over the incremental (DV-carrying) base"
    );

    // An insert-only MATERIALIZE appends and carries the vectors.
    exec(format!("ALTER TABLE {table} MATERIALIZE")).await;
    sleep(Duration::from_millis(300)).await;
    let third = parquet(&server);
    assert!(
        second.iter().all(|p| third.contains(p)),
        "the insert-only republication appends: {third:?}"
    );
    assert_eq!(scalar_count(&client, table).await as usize, model.len());
    assert_wal_matches(&client, table, &model, "after the append").await;

    let rows: Vec<(String, String)> = model
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let expected: Vec<(&str, &str)> = rows.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    if !external_leg_kv(server.rest_port, table, &expected, "iceberg-incr-mat").await {
        eprintln!("SKIP external leg (no reader installed locally); CI requires it");
        return;
    }
    eprintln!(
        "✓ incremental MATERIALIZE: carried parts + dv-mat vectors, both views + external agree"
    );
}

/// Soak evidence for flipping `PROXIMADB_MATERIALIZE_INCREMENTAL_DV` on: a
/// seeded LCG drives 40 rounds of inserts, updates and deletes with a
/// MATERIALIZE every 3rd round (incremental under the 20% bound, compacting
/// past it). After every round the WAL view equals the model; after every
/// MATERIALIZE, and between them over the DV-carrying base, the OLAP view
/// agrees on the row count, the key sum and the count of updated values;
/// after every MATERIALIZE the `dv-mat` Puffins on disk are EXACTLY those the
/// current and previous snapshots name (superseded ones collected); the
/// external engines read the final state.
#[test]
fn soak_incremental_materialize_holds_both_views() {
    run_isolated(
        "soak_incremental_materialize_holds_both_views",
        "PROXIMADB_INCR_SOAK_CHILD",
        &[
            ("PROXIMADB_MATERIALIZE_INCREMENTAL_DV", "1"),
            ("PROXIMADB_PARQUET_WRITE_BLOOM_MIN_ROWS", "1"),
        ],
        check_soak_incremental_materialize,
    );
}

/// The OLAP view's `(COUNT(*), SUM(k), COUNT of updated values)`; updates
/// write `a{round}` values, which sort below every inserted `v{k}`.
async fn olap_fingerprint(client: &tokio_postgres::Client, table: &str) -> (i64, i64, i64) {
    let int = |raw: Option<&Option<String>>| -> i64 {
        raw.and_then(|v| v.as_deref())
            .expect("a non-NULL aggregate")
            .trim()
            .parse()
            .expect("integer aggregate")
    };
    let totals = pg_rows(client, table, &["COUNT(*)", "SUM(k)"]).await;
    let updated = pg_rows(client, &format!("{table} WHERE v < 'b'"), &["COUNT(*)"]).await;
    (
        int(totals[0].first()),
        int(totals[0].get(1)),
        int(updated[0].first()),
    )
}

/// The `dv-mat-*` Puffin file names on disk under the table's base, and the
/// ones its latest and previous snapshots name (read through the engine).
async fn dv_mat_puffins(
    server: &TestPgServer,
    table: &str,
) -> (
    std::collections::BTreeSet<String>,
    std::collections::BTreeSet<String>,
) {
    use proximadb_iceberg_engine::IcebergObjectStoreBridge;
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;

    let mut files = Vec::new();
    walk(&server.data_dir().join("warehouse"), &mut files);
    let metadata_dir = files
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".metadata.json"))
                && p.to_string_lossy().contains(table)
        })
        .and_then(|p| p.parent())
        .expect("the table's metadata dir")
        .to_path_buf();
    let base_dir = metadata_dir.parent().expect("table base");
    let name_of = |p: &std::path::Path| p.file_name().and_then(|n| n.to_str()).map(String::from);
    let on_disk = files
        .iter()
        .filter(|p| p.starts_with(base_dir))
        .filter_map(|p| name_of(p))
        .filter(|n| n.starts_with("dv-mat-") && n.ends_with(".puffin"))
        .collect();
    let bridge = IcebergObjectStoreBridge::from_url(&format!("file://{}", base_dir.display()))
        .expect("bridge");
    let latest = bridge
        .latest_metadata_version("metadata")
        .await
        .expect("metadata log")
        .expect("a published snapshot");
    let mut referenced = std::collections::BTreeSet::new();
    for version in [latest.saturating_sub(1), latest] {
        let Ok(snapshot) = bridge
            .read_iceberg_snapshot_files("metadata", version)
            .await
        else {
            continue; // v0 / a pre-history version
        };
        for v in snapshot.deletion_vectors {
            let name = v.puffin_file.filename().map(String::from);
            if let Some(name) = name.filter(|n| n.starts_with("dv-mat-")) {
                referenced.insert(name);
            }
        }
    }
    (on_disk, referenced)
}

fn model_fingerprint(model: &std::collections::BTreeMap<i32, String>) -> (i64, i64, i64) {
    (
        model.len() as i64,
        model.keys().map(|k| *k as i64).sum(),
        model.values().filter(|v| v.as_str() < "b").count() as i64,
    )
}

async fn check_soak_incremental_materialize() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect_pg(&server).await;
    let table = "incr_soak";
    let exec = |sql: String| {
        let client = &client;
        async move {
            client
                .simple_query(&sql)
                .await
                .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        }
    };
    exec(format!(
        "CREATE TABLE {table} (k INT PRIMARY KEY, v VARCHAR)"
    ))
    .await;
    let mut model: std::collections::BTreeMap<i32, String> = Default::default();
    // A base large enough that most deltas stay under the 20% bound.
    for k in 1..=40 {
        exec(format!("INSERT INTO {table} (k, v) VALUES ({k}, 'v{k}')")).await;
        model.insert(k, format!("v{k}"));
    }
    exec(format!("ALTER TABLE {table} MATERIALIZE")).await;
    sleep(Duration::from_millis(300)).await;

    let mut rng = SoakRng(0x1C4E_2026_1029);
    let mut next_key: i32 = 41;
    let (mut updates, mut deletes, mut materializes) = (0usize, 0usize, 1usize);
    let mut max_dv_mat = 0usize;
    let mut minted_any = false;
    for round in 0..40u64 {
        for _ in 0..rng.below(3) {
            let k = next_key;
            next_key += 1;
            exec(format!("INSERT INTO {table} (k, v) VALUES ({k}, 'v{k}')")).await;
            model.insert(k, format!("v{k}"));
        }
        for _ in 0..rng.below(3) {
            let idx = rng.below(model.len() as u64) as usize;
            let k = *model.keys().nth(idx).expect("indexed key");
            let v = format!("a{round}");
            exec(format!("UPDATE {table} SET v = '{v}' WHERE k = {k}")).await;
            model.insert(k, v);
            updates += 1;
        }
        for _ in 0..rng.below(2) {
            if model.len() <= 10 {
                break;
            }
            let idx = rng.below(model.len() as u64) as usize;
            let k = *model.keys().nth(idx).expect("indexed key");
            exec(format!("DELETE FROM {table} WHERE k = {k}")).await;
            model.remove(&k);
            deletes += 1;
        }
        assert_wal_matches(&client, table, &model, &format!("round {round}")).await;
        // Between MATERIALIZEs the OLAP view merges the delta over the
        // DV-carrying base (the ordered-DV-base read path).
        assert_eq!(
            olap_fingerprint(&client, table).await,
            model_fingerprint(&model),
            "round {round}: the OLAP view merged over the DV base"
        );
        if round % 3 == 2 {
            exec(format!("ALTER TABLE {table} MATERIALIZE")).await;
            materializes += 1;
            sleep(Duration::from_millis(200)).await;
            assert_eq!(
                olap_fingerprint(&client, table).await,
                model_fingerprint(&model),
                "round {round}: the OLAP view reconciles after MATERIALIZE"
            );
            // Exact: the dv-mat Puffins on disk are the ones the current or
            // the previous snapshot names — superseded ones are collected.
            let (on_disk, referenced) = dv_mat_puffins(&server, table).await;
            max_dv_mat = max_dv_mat.max(on_disk.len());
            minted_any |= !on_disk.is_empty();
            assert_eq!(
                on_disk, referenced,
                "round {round}: dv-mat Puffins on disk == those V and V-1 name"
            );
        }
    }
    assert!(minted_any, "the soak exercised the incremental path");

    let rows: Vec<(String, String)> = model
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let expected: Vec<(&str, &str)> = rows.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let external = external_leg_kv(server.rest_port, table, &expected, "iceberg-incr-soak").await;
    eprintln!(
        "incremental soak ok: {updates} updates, {deletes} deletes, {materializes} materializes, \
         <= {max_dv_mat} dv-mat Puffins on disk (== those V and V-1 name); external leg {}",
        if external {
            "checked"
        } else {
            "SKIPPED (no reader locally)"
        }
    );
}
