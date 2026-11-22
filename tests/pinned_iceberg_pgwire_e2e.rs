#![cfg(all(feature = "datafusion-integration", unix))]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Decimal128Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use object_store::path::Path;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use proximadb::observability::io_trace;
use proximadb_iceberg_engine::{
    IcebergObjectStoreBridge, deletion_vector::PositionDeletes, iceberg::IcebergField,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio_postgres::{Client, NoTls, SimpleQueryMessage};

#[path = "common/pg_server.rs"]
mod pg_server;

#[derive(Deserialize)]
struct TpcFixture {
    suite: String,
    tables: Vec<TpcTable>,
    queries: Vec<TpcQuery>,
}

#[derive(Deserialize)]
struct TpcTable {
    name: String,
    rows: u64,
    schema_ipc_base64: String,
    iceberg: IcebergInput,
    json: Option<NdjsonInput>,
    delta: Option<DeltaInput>,
}

#[derive(Deserialize)]
struct IcebergInput {
    location: String,
    metadata_location: String,
}

#[derive(Deserialize)]
struct NdjsonInput {
    path: String,
    manifest_sha256: Option<String>,
}

#[derive(Deserialize)]
struct DeltaInput {
    path: String,
    version: u64,
}

#[derive(Deserialize)]
struct TpcQuery {
    id: usize,
    sql: String,
    sql_sha256: String,
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn columns_for_tpc_table(table: &TpcTable) -> anyhow::Result<String> {
    let bytes = STANDARD.decode(&table.schema_ipc_base64)?;
    let schema = arrow::ipc::convert::try_schema_from_ipc_buffer(&bytes)?;
    let mut columns = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let sql_type = match field.data_type() {
            DataType::Int32 => "INTEGER".into(),
            DataType::Int64 => "BIGINT".into(),
            DataType::Float64 => "DOUBLE PRECISION".into(),
            DataType::Utf8 | DataType::LargeUtf8 => "TEXT".into(),
            DataType::Date32 => "DATE".into(),
            DataType::Decimal128(p, s) => format!("DECIMAL({p},{s})"),
            other => anyhow::bail!(
                "{}: unsupported TPC fixture field type {other}",
                field.name()
            ),
        };
        columns.push(format!("{} {sql_type}", quote_identifier(field.name())));
    }
    Ok(columns.join(", "))
}

struct ResetTraceObserver;

impl Drop for ResetTraceObserver {
    fn drop(&mut self) {
        io_trace::set_trace_observer(None);
    }
}

fn capture_pinned_traces() -> (
    std::sync::mpsc::Receiver<io_trace::IoTraceSnapshot>,
    ResetTraceObserver,
) {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    io_trace::set_trace_observer(Some(Box::new(move |trace, _tenant| {
        if !trace.pinned_binds.is_empty() {
            let _ = sender.try_send(trace.clone());
        }
    })));
    (receiver, ResetTraceObserver)
}

async fn wait_for_failed_query_trace(client: &Client) -> anyhow::Result<()> {
    // The client can return on ErrorResponse before the server publishes the
    // trace and sends ReadyForQuery. A successful command on the same
    // connection waits behind that work and has no pinned binds of its own.
    tokio::time::timeout(Duration::from_secs(5), client.simple_query("SELECT 1")).await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires an external fixture from scripts/bench/build_open_table_tpc_fixture.py"]
async fn pinned_external_full_tpc_sql() -> anyhow::Result<()> {
    use std::io::Write;

    let root = std::path::PathBuf::from(std::env::var("TPC_OPEN_FIXTURE")?);
    let fixture: TpcFixture = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let (table_count, query_count) = match fixture.suite.as_str() {
        "tpch" => (8, 22),
        "tpcds" => (24, 99),
        suite => anyhow::bail!("unknown fixture suite {suite}"),
    };
    anyhow::ensure!(
        fixture.tables.len() == table_count,
        "unexpected table inventory"
    );
    anyhow::ensure!(
        fixture
            .queries
            .iter()
            .map(|query| query.id)
            .eq(1..=query_count),
        "unexpected query inventory"
    );
    let format = std::env::var("TPC_OPEN_FORMAT").unwrap_or_else(|_| "iceberg".to_owned());
    anyhow::ensure!(
        matches!(format.as_str(), "iceberg" | "ndjson" | "delta"),
        "expected TPC_OPEN_FORMAT=iceberg, ndjson, or delta"
    );
    let mut output = std::fs::File::create_new(std::env::var("TPC_OPEN_RESULTS")?)?;
    let server = pg_server::TestPgServer::start().await?;
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), NoTls).await?;
    tokio::spawn(conn);
    for table in &fixture.tables {
        let name = quote_identifier(&table.name);
        let columns = columns_for_tpc_table(table)?;
        let options = if format == "ndjson" {
            let input = table
                .json
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing NDJSON fixture"))?;
            let path = url::Url::from_file_path(root.join(&input.path))
                .map_err(|_| anyhow::anyhow!("expected an absolute NDJSON fixture path"))?;
            let location = quote_literal(path.as_str());
            let digest = quote_literal(
                input
                    .manifest_sha256
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("missing NDJSON fixture manifest digest"))?,
            );
            format!("format='json', external_location={location}, ndjson_manifest_sha256={digest}")
        } else if format == "delta" {
            let input = table
                .delta
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing Delta fixture"))?;
            let path = url::Url::from_file_path(root.join(&input.path))
                .map_err(|_| anyhow::anyhow!("expected an absolute Delta fixture path"))?;
            let location = quote_literal(path.as_str());
            let version = input.version;
            format!("format='delta', external_location={location}, delta_version='{version}'")
        } else {
            let location = quote_literal(&table.iceberg.location);
            let metadata = quote_literal(&table.iceberg.metadata_location);
            format!(
                "format='iceberg', external_location={location}, iceberg_metadata_location={metadata}"
            )
        };
        client
            .simple_query(&format!(
                "CREATE TABLE {name} ({columns}) WITH (authority='external', {options})"
            ))
            .await
            .map_err(|e| anyhow::anyhow!("register {name}: {}", pg_error(&e)))?;
        let counts = client
            .simple_query(&format!("SELECT count(*) FROM {name}"))
            .await
            .map_err(|e| anyhow::anyhow!("count {name}: {}", pg_error(&e)))?;
        let actual = counts.iter().find_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get(0).and_then(|value| value.parse::<u64>().ok()),
            _ => None,
        });
        anyhow::ensure!(
            actual == Some(table.rows),
            "{name}: row count is {actual:?}, expected {}",
            table.rows
        );
    }
    let mut failures = 0;
    let (trace_rx, _reset_trace) = capture_pinned_traces();
    let mut trace_rx = Some(trace_rx);
    for query in fixture.queries {
        let sql = std::fs::read_to_string(root.join(&query.sql))?;
        let digest = hex::encode(Sha256::digest(sql.as_bytes()));
        anyhow::ensure!(
            digest == query.sql_sha256,
            "fixture SQL digest differs for Q{}",
            query.id
        );
        let start = std::time::Instant::now();
        let result =
            tokio::time::timeout(Duration::from_secs(120), client.simple_query(&sql)).await;
        let mut record = serde_json::json!({
            "suite": fixture.suite,
            "format": format,
            "id": query.id,
            "sql_sha256": digest,
            "elapsed_ms": start.elapsed().as_millis() as u64,
        });
        let trace_sync = match &result {
            Ok(Err(error)) if error.as_db_error().is_some() && trace_rx.is_some() => {
                wait_for_failed_query_trace(&client).await
            }
            Ok(Err(error)) if error.as_db_error().is_none() => Err(anyhow::anyhow!(
                "pgwire connection failed before trace completion"
            )),
            Err(_) => Err(anyhow::anyhow!(
                "client deadline expired before trace completion"
            )),
            _ => Ok(()),
        };
        if let Err(error) = trace_sync {
            trace_rx = None;
            record["trace_error"] = serde_json::json!(error.to_string());
        }
        if let Some(trace_rx) = &trace_rx {
            match trace_rx.try_recv() {
                Ok(trace) => {
                    record["trace"] = serde_json::json!({
                        "setup_ms": trace.setup_ms,
                        "session_ms": trace.session_ms,
                        "open_ms": trace.open_ms,
                        "plan_ms": trace.plan_ms,
                        "compute_ms": trace.compute_ms,
                        "emit_ms": trace.emit_ms,
                        "pinned_binds": trace.pinned_binds,
                    });
                }
                Err(error) if matches!(result, Ok(Ok(_))) => {
                    failures += 1;
                    record["trace_error"] =
                        serde_json::json!(format!("missing binding trace: {error}"));
                }
                Err(_) => {}
            }
        } else if record.get("trace_error").is_none() {
            record["trace_error"] =
                serde_json::json!("trace capture disabled after an earlier query");
        }
        match result {
            Ok(Ok(messages)) => {
                let columns = messages.iter().find_map(|msg| match msg {
                    SimpleQueryMessage::RowDescription(columns) => Some(columns.len()),
                    _ => None,
                });
                record["columns"] = serde_json::json!(columns);
                let rows: Vec<Vec<Option<&str>>> = messages
                    .iter()
                    .filter_map(|msg| match msg {
                        SimpleQueryMessage::Row(row) => {
                            Some((0..row.len()).map(|i| row.get(i)).collect())
                        }
                        _ => None,
                    })
                    .collect();
                record["rows"] = serde_json::json!(rows);
            }
            Ok(Err(err)) => {
                failures += 1;
                record["error"] = serde_json::json!(pg_error(&err));
            }
            Err(_) => {
                failures += 1;
                record["error"] = serde_json::json!("120 second client deadline");
                client.cancel_token().cancel_query(NoTls).await?;
            }
        }
        writeln!(output, "{record}")?;
    }
    anyhow::ensure!(
        failures == 0,
        "{failures}/{query_count} queries had execution or trace failures; see TPC_OPEN_RESULTS"
    );
    Ok(())
}

fn pg_error(error: &tokio_postgres::Error) -> String {
    error.as_db_error().map_or_else(
        || error.to_string(),
        |error| format!("[{}] {}", error.code().code(), error.message()),
    )
}

async fn write_fixture(root: &str) -> (String, Bytes) {
    let bridge = IcebergObjectStoreBridge::from_url(root).unwrap();
    let parquet = Path::from("data/rows.parquet");
    let id = |name, dt, id: &str| {
        Field::new(name, dt, false)
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_owned(), id.to_owned())])
    };
    let schema = Arc::new(Schema::new(vec![
        id("old_value", DataType::Int64, "7"),
        id("amount", DataType::Decimal128(12, 2), "8"),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![10, 11, 12])),
            Arc::new(
                Decimal128Array::from(vec![12345, 999, -123])
                    .with_precision_and_scale(12, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    bridge
        .object_store()
        .put(&parquet, Bytes::from(bytes))
        .await
        .unwrap();

    let mut delete = PositionDeletes::new();
    delete.insert(1);
    let vectors = bridge
        .write_deletion_vectors(
            &Path::from("data/delete.puffin"),
            &[(parquet.clone(), &delete)],
        )
        .await
        .unwrap();
    let version = bridge
        .publish_iceberg_files_with_deletes(
            &[parquet],
            &vectors,
            None,
            "metadata",
            "eeff0011-2233-4455-6677-8899aabbccdd",
            root,
            &[
                IcebergField {
                    id: 7,
                    name: "value".into(),
                    type_name: "long".into(),
                    required: true,
                },
                IcebergField {
                    id: 8,
                    name: "amount".into(),
                    type_name: "decimal(12,2)".into(),
                    required: true,
                },
            ],
            7,
            1_700_000_000_000,
        )
        .await
        .unwrap();
    let uri = bridge.metadata_file_uri("metadata", version);
    let mut doc: serde_json::Value = serde_json::from_slice(
        &bridge
            .object_store()
            .get(&Path::from(uri.strip_prefix(&format!("{root}/")).unwrap()))
            .await
            .unwrap(),
    )
    .unwrap();
    doc["properties"]["converted_delta_version"] = serde_json::json!("42");
    let bytes = Bytes::from(serde_json::to_vec(&doc).unwrap());
    let producer = Path::from("metadata/00007-producer.metadata.json");
    bridge
        .object_store()
        .put(&producer, bytes.clone())
        .await
        .unwrap();
    (bridge.object_uri(producer.as_ref()), bytes)
}

async fn create(
    client: &Client,
    name: &str,
    root: &str,
    metadata_uri: &str,
    version: u64,
) -> Result<Vec<SimpleQueryMessage>, tokio_postgres::Error> {
    client.simple_query(&format!(
        "CREATE TABLE {name} (value BIGINT, amount DECIMAL(12,2)) WITH (         format='iceberg', authority='external', external_location='{root}',          iceberg_metadata_location='{metadata_uri}', converted_delta_version='{version}')"
    )).await
}

async fn create_ndjson(
    client: &Client,
    name: &str,
    allowed: &std::path::Path,
    ranges: &[&[u8]],
) -> (std::path::PathBuf, std::path::PathBuf) {
    let file = allowed.join(format!("{name}.jsonl"));
    let manifest_file = allowed.join(format!("{name}.jsonl.manifest.json"));
    let mut size = 0;
    let entries: Vec<_> = ranges
        .iter()
        .map(|range| {
            size += range.len();
            serde_json::json!({"end": size, "sha256": hex::encode(Sha256::digest(range))})
        })
        .collect();
    let manifest =
        serde_json::to_vec(&serde_json::json!({"version": 1, "size": size, "ranges": entries}))
            .unwrap();
    std::fs::write(&file, ranges.concat()).unwrap();
    std::fs::write(&manifest_file, &manifest).unwrap();
    let digest = hex::encode(Sha256::digest(manifest));
    client
        .simple_query(&format!(
            "CREATE TABLE {name} (value BIGINT NOT NULL, amount DECIMAL(12,2), day DATE) \
         WITH (format='json', authority='external', external_location='file://{}', \
         ndjson_manifest_sha256='{digest}')",
            file.display(),
        ))
        .await
        .unwrap();
    (file, manifest_file)
}

async fn create_delta(client: &Client, name: &str, root: &std::path::Path, version: u64) {
    client
        .simple_query(&format!(
            "CREATE TABLE {name} (id BIGINT) WITH (format='delta', authority='external', \
         external_location='file://{}', delta_version='{version}')",
            root.display(),
        ))
        .await
        .unwrap();
}

fn delta_rows(messages: &[SimpleQueryMessage]) -> Vec<i64> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => Some(row.get(0).unwrap().parse().unwrap()),
            _ => None,
        })
        .collect()
}

fn write_delta_commit(root: &std::path::Path, version: u64, actions: &[serde_json::Value]) {
    let log = root.join("_delta_log");
    std::fs::create_dir_all(&log).unwrap();
    let content: String = actions.iter().map(|value| format!("{value}\n")).collect();
    std::fs::write(log.join(format!("{version:020}.json")), content).unwrap();
}

fn write_delta_fixture(root: &std::path::Path) {
    use serde_json::json;

    std::fs::create_dir_all(root).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
    let physical_path = root.join("data.parquet");
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&physical_path).unwrap(),
        schema.clone(),
        None,
    )
    .unwrap();
    writer
        .write(
            &RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![10, 11, 12]))])
                .unwrap(),
        )
        .unwrap();
    writer.close().unwrap();
    let schema_string = json!({"type":"struct","fields":[{"name":"id","type":"long","nullable":true,"metadata":{}}]});
    let add = json!({"path": "data.parquet", "size": std::fs::metadata(&physical_path).unwrap().len(),
        "partitionValues": {}, "modificationTime": 1, "dataChange": true, "stats": "{\"numRecords\":3}"});
    write_delta_commit(
        root,
        0,
        &[
            json!({"protocol": {"minReaderVersion":3,"minWriterVersion":7,
            "readerFeatures":["deletionVectors"],"writerFeatures":["deletionVectors"]}}),
            json!({"metaData":{"id":"60cf3037-2476-4baa-a76a-d52553dfc500","format":{"provider":"parquet","options":{}},
            "schemaString":schema_string.to_string(),"partitionColumns":[],"configuration":{"delta.enableDeletionVectors":"true"}}}),
            json!({"add":add}),
        ],
    );
    let mut updated = add;
    updated["deletionVector"] = write_delta_delete(root);
    write_delta_commit(
        root,
        1,
        &[
            json!({"remove":{"path":"data.parquet","deletionTimestamp":2,"dataChange":true}}),
            json!({"add":updated}),
        ],
    );
    updated["path"] = json!("https://example.invalid/outside.parquet?X-Amz-Signature=fixture");
    write_delta_commit(root, 2, &[json!({"add":updated})]);
}

fn write_delta_delete(root: &std::path::Path) -> serde_json::Value {
    use delta_kernel::actions::deletion_vector_writer::{
        KernelDeletionVector, StreamingDeletionVectorWriter,
    };
    let mut bytes = Vec::new();
    let mut writer = StreamingDeletionVectorWriter::new(&mut bytes);
    let mut vector = KernelDeletionVector::new();
    vector.add_deleted_row_indexes([0u64]);
    let descriptor = writer.write_deletion_vector(vector).unwrap();
    writer.finalize().unwrap();
    std::fs::write(
        root.join("deletion_vector_00000000-0000-0000-0000-000000000000.bin"),
        &bytes,
    )
    .unwrap();
    serde_json::json!({
        "storageType":"u", "pathOrInlineDv":"00000000000000000000",
        "offset":descriptor.offset,"sizeInBytes":descriptor.size_in_bytes,"cardinality":descriptor.cardinality,
    })
}

// A mismatched physical field must fail when selected. A scan of another
// column can skip decoding it while still resolving the requested mapping.
fn write_mapped_projection_fixture(root: &std::path::Path, partitioned: bool) {
    std::fs::create_dir_all(root).unwrap();
    let data = root.join("data.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("col-id", DataType::Int64, true),
        Field::new("col-payload", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![7, 8])),
            Arc::new(arrow_array::StringArray::from(vec!["abc", "def"])),
        ],
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&data).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    use serde_json::json;

    let mut fields = vec![
        json!({"name":"id","type":"long","nullable":true,"metadata":{"delta.columnMapping.id":1,"delta.columnMapping.physicalName":"col-id"}}),
        json!({"name":"payload","type":"long","nullable":true,"metadata":{"delta.columnMapping.id":2,"delta.columnMapping.physicalName":"col-payload"}}),
    ];
    let mut configuration =
        json!({"delta.columnMapping.mode":"name","delta.columnMapping.maxColumnId":"2"});
    let mut add = json!({"path":"data.parquet","size":std::fs::metadata(data).unwrap().len(),
        "modificationTime":1,"partitionValues":{},"dataChange":true});
    let (protocol, partition_columns) = if partitioned {
        fields.insert(1, json!({"name":"region","type":"long","nullable":true,"metadata":{"delta.columnMapping.id":3,"delta.columnMapping.physicalName":"col-region"}}));
        configuration["delta.columnMapping.maxColumnId"] = json!("3");
        configuration["delta.enableDeletionVectors"] = json!("true");
        add["partitionValues"] = json!({"col-region":"40"});
        add["deletionVector"] = write_delta_delete(root);
        (
            json!({"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["columnMapping","deletionVectors"],"writerFeatures":["columnMapping","deletionVectors"]}),
            json!(["region"]),
        )
    } else {
        (
            json!({"minReaderVersion":2,"minWriterVersion":5}),
            json!([]),
        )
    };
    let schema_string = json!({"type":"struct","fields":fields});
    write_delta_commit(
        root,
        0,
        &[
            json!({"protocol":protocol}),
            json!({"metaData":{"id":"78742617-28e7-4dc9-87d6-e82b85ea0a65","format":{"provider":"parquet","options":{}},
                "schemaString":schema_string.to_string(),"partitionColumns":partition_columns,"configuration":configuration}}),
            json!({"add":add}),
        ],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the delta-rs retained-checkpoint fixture"]
async fn pinned_delta_checkpoint_replay() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(std::env::var("TPC_DELTA_CHECKPOINT_FIXTURE")?);
    let server = pg_server::TestPgServer::start().await?;
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), NoTls).await?;
    tokio::spawn(conn);
    for (version, expected) in [(2, &[4, 5][..]), (3, &[4, 5, 6][..])] {
        let name = format!("checkpoint_v{version}");
        create_delta(&client, &name, &root, version).await;
        let rows = client
            .simple_query(&format!("SELECT id FROM {name} ORDER BY id"))
            .await
            .map_err(|e| anyhow::anyhow!(pg_error(&e)))?;
        assert_eq!(delta_rows(&rows), expected);
    }
    create_delta(&client, "expired_delta", &root, 0).await;
    let error = client
        .simple_query("SELECT id FROM expired_delta")
        .await
        .unwrap_err();
    assert!(pg_error(&error).contains("cannot open Delta version 0"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_reads_only_the_authorized_pinned_snapshot() {
    let allowed = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let table = allowed.path().join("producer");
    std::fs::create_dir(&table).unwrap();
    let root = format!("file://{}", table.display());
    let authorized = format!("file://{}", allowed.path().display());
    unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &authorized) };
    let (uri, metadata) = write_fixture(&root).await;

    let server = pg_server::TestPgServer::start().await.unwrap();
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), NoTls)
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });

    create(&client, "pinned_read", &root, &uri, 42)
        .await
        .unwrap();
    let (http, base) = server.rest_client().unwrap();
    let collection = http
        .get(format!("{base}/api/v2/collections/pinned_read"))
        .send()
        .await
        .unwrap();
    assert_eq!(collection.status(), reqwest::StatusCode::NOT_FOUND);
    let msgs = client
        .simple_query("SELECT value, amount FROM pinned_read ORDER BY value")
        .await
        .unwrap();
    let rows: Vec<_> = msgs
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => Some((row.get(0).unwrap(), row.get(1).unwrap())),
            _ => None,
        })
        .collect();
    assert_eq!(rows, vec![("10", "123.45"), ("12", "-1.23")]);
    let ddl_error = client
        .simple_query("ALTER TABLE pinned_read ADD COLUMN extra BIGINT")
        .await
        .unwrap_err();
    assert!(pg_error(&ddl_error).contains("read-only"), "{ddl_error:?}");
    let ordered = client
        .simple_query("SELECT amount, value FROM pinned_read ORDER BY 2 DESC LIMIT 1")
        .await
        .unwrap();
    assert!(ordered.iter().any(|msg| matches!(
        msg,
        SimpleQueryMessage::Row(row) if row.get(0) == Some("-1.23") && row.get(1) == Some("12")
    )));
    let rows = client
        .simple_query(
            "SELECT CASE WHEN EXISTS(SELECT 1 FROM pinned_read WHERE value = 10) \
             THEN 9 ELSE 0 END AS value FROM \"pinned_read\" WHERE value = 10 \
             UNION ALL SELECT value FROM pinned_read WHERE value = 12 ORDER BY value",
        )
        .await
        .unwrap();
    let values: Vec<_> = rows
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get(0),
            _ => None,
        })
        .collect();
    assert_eq!(values, ["9", "12"]);
    let error = client
        .simple_query(
            "SELECT value FROM pinned_read \
             WHERE EXISTS (SELECT 1 FROM timeseries_range('x', 0, 1))",
        )
        .await
        .unwrap_err();
    assert!(
        pg_error(&error).contains("pinned external reads require only unqualified pinned tables")
    );
    let statement = client
        .prepare_typed(
            "SELECT value AS Renamed FROM pinned_read WHERE value > $1",
            &[tokio_postgres::types::Type::INT8],
        )
        .await
        .unwrap();
    assert_eq!(statement.columns()[0].name(), "renamed");
    let rows = client.query(&statement, &[&10i64]).await.unwrap();
    assert_eq!(
        rows.iter()
            .map(|row| row.get::<_, i64>(0))
            .collect::<Vec<_>>(),
        [12]
    );

    let delta = allowed.path().join("delta");
    write_delta_fixture(&delta);
    create_delta(&client, "delta_v0", &delta, 0).await;
    create_delta(&client, "delta_v1", &delta, 1).await;
    create_delta(&client, "delta_bad_path", &delta, 2).await;
    let case_fold = std::env::var_os("PROXIMADB_IDENT_CASE_FOLD");
    unsafe { std::env::set_var("PROXIMADB_IDENT_CASE_FOLD", "0") };
    create_delta(&client, "\"MixedPinned\"", &delta, 0).await;
    create_delta(&client, "mixedpinned", &delta, 1).await;
    unsafe {
        match case_fold {
            Some(value) => std::env::set_var("PROXIMADB_IDENT_CASE_FOLD", value),
            None => std::env::remove_var("PROXIMADB_IDENT_CASE_FOLD"),
        }
    }
    for (sql, expected) in [
        ("SELECT id FROM delta_v0 ORDER BY id", &[10, 11, 12][..]),
        ("SELECT id FROM delta_v1 ORDER BY id", &[11, 12][..]),
        (
            "SELECT id FROM \"MixedPinned\" EXCEPT SELECT id FROM MixedPinned ORDER BY id",
            &[10][..],
        ),
        (
            "SELECT d.id FROM delta_v1 d JOIN pinned_read i ON d.id = i.value ORDER BY d.id",
            &[12][..],
        ),
        ("SELECT count(*) FROM delta_v1", &[2][..]),
    ] {
        let rows = client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {}", pg_error(&e)));
        assert_eq!(delta_rows(&rows), expected, "{sql}");
    }
    for (sql, name, expected) in [
        (
            "SELECT id AS Renamed FROM \"MixedPinned\" WHERE id = 10",
            "renamed",
            10i64,
        ),
        (
            "SELECT id AS Renamed FROM mixedpinned WHERE id = 11 AND EXISTS (SELECT 1 FROM \"MixedPinned\")",
            "renamed",
            11i64,
        ),
        ("SELECT count(*) FROM \"MixedPinned\"", "count(*)", 3i64),
    ] {
        let prepared = client.prepare(sql).await.unwrap();
        assert_eq!(prepared.columns()[0].name(), name);
        assert_eq!(
            client
                .query_one(&prepared, &[])
                .await
                .unwrap()
                .get::<_, i64>(0),
            expected
        );
    }
    let wildcard = client
        .prepare("SELECT * FROM MixedPinned WHERE id = 10")
        .await
        .unwrap();
    assert_eq!(wildcard.columns()[0].name(), "id");
    assert!(client.query(&wildcard, &[]).await.unwrap().is_empty());
    {
        let (traces, _reset_trace) = capture_pinned_traces();
        let error = client
            .simple_query("SELECT id FROM delta_bad_path LIMIT 1")
            .await
            .unwrap_err();
        assert!(pg_error(&error).contains("invalid relative Delta data path"));
        wait_for_failed_query_trace(&client).await.unwrap();
        let failed = traces.try_recv().expect("trace for failed Delta bind");
        assert_eq!(failed.pinned_binds.len(), 1);
        let delta = &failed.pinned_binds["delta"];
        assert_eq!(
            (delta.attempts, delta.failures, delta.bound_splits),
            (1, 1, 0)
        );
        assert!(failed.setup_ms >= delta.elapsed_us / 1000);

        client
            .simple_query("SELECT value FROM pinned_read LIMIT 1")
            .await
            .unwrap();
        let succeeded = traces
            .try_recv()
            .expect("trace for successful Iceberg bind");
        assert_eq!(succeeded.pinned_binds.len(), 1);
        let iceberg = &succeeded.pinned_binds["iceberg"];
        assert_eq!((iceberg.attempts, iceberg.failures), (1, 0));
        assert!(traces.try_recv().is_err());
    }
    let error = client
        .simple_query("INSERT INTO delta_v1 VALUES (1)")
        .await
        .unwrap_err();
    assert!(pg_error(&error).contains("read-only"));

    let projected = allowed.path().join("delta_projection");
    write_mapped_projection_fixture(&projected, false);
    client.simple_query(&format!(
        "CREATE TABLE delta_projection (id BIGINT, payload BIGINT) WITH (format='delta', authority='external', \
         external_location='file://{}', delta_version='0')", projected.display()
    )).await.unwrap();
    for (sql, expected) in [
        ("SELECT id FROM delta_projection ORDER BY id", &[7, 8][..]),
        ("SELECT count(*) FROM delta_projection", &[2][..]),
    ] {
        let rows = client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {}", pg_error(&e)));
        assert_eq!(delta_rows(&rows), expected);
    }
    assert!(
        client
            .simple_query("SELECT payload FROM delta_projection")
            .await
            .is_err()
    );

    let partitioned = allowed.path().join("delta_partition_projection");
    write_mapped_projection_fixture(&partitioned, true);
    client.simple_query(&format!(
        "CREATE TABLE delta_partition_projection (id BIGINT, region BIGINT, payload BIGINT) \
         WITH (format='delta', authority='external', external_location='file://{}', delta_version='0')",
        partitioned.display(),
    )).await.unwrap();
    let rows = client
        .simple_query("SELECT region, id FROM delta_partition_projection")
        .await
        .unwrap_or_else(|e| panic!("{}", pg_error(&e)));
    let values: Vec<_> = rows
        .iter()
        .filter_map(|message| match message {
            SimpleQueryMessage::Row(row) => Some((row.get(0), row.get(1))),
            _ => None,
        })
        .collect();
    assert_eq!(values, [(Some("40"), Some("8"))]);
    for (sql, expected) in [
        ("SELECT region FROM delta_partition_projection", 40),
        ("SELECT count(*) FROM delta_partition_projection", 1),
    ] {
        let rows = client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {}", pg_error(&e)));
        assert_eq!(delta_rows(&rows), [expected]);
    }

    let cancel_root = allowed.path().join("delta_cancel");
    std::fs::create_dir_all(cancel_root.join("_delta_log")).unwrap();
    std::fs::copy(delta.join("data.parquet"), cancel_root.join("data.parquet")).unwrap();
    let initial = "_delta_log/00000000000000000000.json";
    std::fs::copy(delta.join(initial), cancel_root.join(initial)).unwrap();
    let change = serde_json::json!({"commitInfo":{"operation":"WRITE"}});
    for version in 1..=1536 {
        write_delta_commit(&cancel_root, version, std::slice::from_ref(&change));
    }
    create_delta(&client, "delta_cancel", &cancel_root, 1536).await;
    let query = client.simple_query("SELECT count(*) FROM delta_cancel");
    tokio::pin!(query);
    let error = tokio::select! {
        result = &mut query => result.expect_err("Delta replay completed before the cancel window"),
        _ = tokio::time::sleep(Duration::from_millis(40)) => {
            client.cancel_token().cancel_query(NoTls).await.unwrap();
            tokio::time::timeout(Duration::from_secs(5), &mut query)
                .await.expect("Delta replay did not cancel").unwrap_err()
        }
    };
    assert_eq!(
        error.as_db_error().unwrap().code().code(),
        "57014",
        "{error:?}"
    );
    let rows = client
        .simple_query("SELECT count(*) FROM delta_v1")
        .await
        .unwrap();
    assert_eq!(delta_rows(&rows), [2]);

    let first = br#"{"value":10,"amount":123.45,"day":"2000-02-29"}
"#;
    let second = br#"{"value":12,"day":null}
"#;
    let (json_file, json_manifest) =
        create_ndjson(&client, "pinned_json", allowed.path(), &[first, second]).await;
    let msgs = client.simple_query(
        "SELECT j.amount, j.day FROM pinned_json j JOIN pinned_read i ON j.value = i.value ORDER BY j.value"
    ).await.unwrap();
    let values: Vec<_> = msgs
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => Some((row.get(0), row.get(1))),
            _ => None,
        })
        .collect();
    assert_eq!(values, [(Some("123.45"), Some("2000-02-29")), (None, None)]);
    let rows = client
        .simple_query(
            "SELECT d.id FROM delta_v1 d JOIN pinned_json j ON d.id = j.value \
             JOIN pinned_read i ON d.id = i.value ORDER BY d.id",
        )
        .await
        .unwrap();
    assert_eq!(delta_rows(&rows), [12]);
    let err = client
        .simple_query("INSERT INTO pinned_json (value) VALUES (20)")
        .await
        .unwrap_err();
    assert!(pg_error(&err).contains("read-only"), "{err:?}");
    create_ndjson(
        &client,
        "pinned_inexact_json",
        allowed.path(),
        &[b"{\"value\":1,\"amount\":0.001}\n"],
    )
    .await;
    let err = client
        .simple_query("SELECT count(*) FROM pinned_inexact_json")
        .await
        .unwrap_err();
    assert!(pg_error(&err).contains("represented exactly"), "{err:?}");

    let mut changed = [first.as_slice(), second.as_slice()].concat();
    changed[9] = b'9';
    std::fs::write(json_file, changed).unwrap();
    let err = client
        .simple_query("SELECT count(*) FROM pinned_json")
        .await
        .unwrap_err();
    assert!(pg_error(&err).contains("range SHA-256 mismatch"), "{err:?}");
    std::fs::write(json_manifest, b"{}").unwrap();
    let err = client
        .simple_query("SELECT count(*) FROM pinned_json")
        .await
        .unwrap_err();
    assert!(
        pg_error(&err).contains("manifest SHA-256 mismatch"),
        "{err:?}"
    );

    let err = client
        .simple_query("INSERT INTO pinned_read (value, amount) VALUES (20, 1.00)")
        .await
        .unwrap_err();
    assert!(
        err.as_db_error().unwrap().message().contains("read-only"),
        "{err:?}"
    );

    create(&client, "pinned_wrong_version", &root, &uri, 43)
        .await
        .unwrap();
    let err = client
        .simple_query("SELECT value FROM pinned_wrong_version")
        .await
        .unwrap_err();
    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("expected UniForm Delta version 43"),
        "{err:?}"
    );

    let external_metadata = outside.path().join("external.metadata.json");
    std::fs::write(&external_metadata, metadata).unwrap();
    std::os::unix::fs::symlink(
        &external_metadata,
        table.join("metadata/escape.metadata.json"),
    )
    .unwrap();
    create(
        &client,
        "pinned_escape",
        &root,
        &format!("{root}/metadata/escape.metadata.json"),
        42,
    )
    .await
    .unwrap();
    let err = client
        .simple_query("SELECT value FROM pinned_escape")
        .await
        .unwrap_err();
    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("confined local external file store"),
        "{err:?}"
    );

    let parquet = table.join("data/rows.parquet");
    let external_parquet = outside.path().join("rows.parquet");
    std::fs::rename(&parquet, &external_parquet).unwrap();
    std::os::unix::fs::symlink(&external_parquet, &parquet).unwrap();
    let err = client
        .simple_query("SELECT count(*) FROM pinned_read")
        .await
        .unwrap_err();
    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("confined local external file store"),
        "{err:?}"
    );
}
