use std::sync::Arc;

use object_store::memory::InMemory;
use proximadb_object_store::ProximaObjectStore;
use proximadb_records::{ProximaRecord, ProximaTreeNode, ProximaValue};
use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;

use super::*;

const META: &str = "t/metadata";

fn bridge() -> IcebergObjectStoreBridge {
    IcebergObjectStoreBridge::new(ProximaObjectStore::new(Arc::new(InMemory::new())))
        .with_root_url("memory:///")
}

fn fields() -> Vec<iceberg::IcebergField> {
    vec![iceberg::IcebergField {
        id: 1,
        name: "age".into(),
        type_name: "long".into(),
        required: false,
    }]
}

async fn part(b: &IcebergObjectStoreBridge, key: &str, ages: &[i64]) -> Path {
    let records: Vec<ProximaRecord> = ages
        .iter()
        .map(|a| {
            let mut r = ProximaRecord {
                oid: format!("r{a}"),
                ..Default::default()
            };
            r.props.insert(
                "age".into(),
                ProximaTreeNode::Value(ProximaValue::Int64(*a)),
            );
            r
        })
        .collect();
    let path = Path::from(key);
    b.write_records_to_parquet(&path, &records, None)
        .await
        .expect("write part");
    path
}

/// A published file the way MATERIALIZE writes it: the hidden `__proxima_oid`
/// column rides every row (P2.4 slice 2a). The engine's plain
/// `write_records_to_parquet` does NOT append it — the system-column
/// conversion does.
async fn part_with_oid(b: &IcebergObjectStoreBridge, key: &str, ages: &[i64]) -> Path {
    let records: Vec<ProximaRecord> = ages
        .iter()
        .map(|a| {
            let mut r = ProximaRecord {
                oid: format!("r{a}"),
                ..Default::default()
            };
            r.props.insert(
                "age".into(),
                ProximaTreeNode::Value(ProximaValue::Int64(*a)),
            );
            r
        })
        .collect();
    let schema = proximadb_storage_common::proxima_arrow::infer_proxima_schema_strict(&records)
        .expect("infer schema");
    let batch = proximadb_storage_common::proxima_arrow::proxima_records_to_record_batch_with_system_columns(&records, &schema)
        .expect("batch with oid column");
    let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
        std::slice::from_ref(&batch),
        batch.schema(),
        None,
    )
    .expect("parquet bytes");
    let path = Path::from(key);
    b.store
        .put_if_absent(&path, bytes::Bytes::from(bytes))
        .await
        .expect("put part");
    path
}

fn deletes(positions: &[u64]) -> PositionDeletes {
    let mut d = PositionDeletes::new();
    for p in positions {
        d.insert(*p);
    }
    d
}

async fn publish(
    b: &IcebergObjectStoreBridge,
    files: &[Path],
    dvs: &[iceberg::DeleteFileMeta],
    parent: Option<u64>,
    snapshot_id: i64,
) -> Result<u64, StorageError> {
    b.publish_iceberg_files_with_deletes(
        files,
        dvs,
        parent,
        META,
        "uuid-dv",
        "memory:///t",
        &fields(),
        snapshot_id,
        1_700_000_000_000,
    )
    .await
}

#[tokio::test]
async fn a_snapshot_with_deletion_vectors_reads_back_with_them() {
    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[10, 11, 12, 13]).await;
    let c = part(&b, "t/data/c.parquet", &[20, 21]).await;
    let (da, dc) = (deletes(&[1, 3]), deletes(&[0]));
    let dvs = b
        .write_deletion_vectors(
            &Path::from("t/data/dv-1.puffin"),
            &[(a.clone(), &da), (c.clone(), &dc)],
        )
        .await
        .expect("write vectors");
    assert_eq!(dvs.len(), 2);
    assert_eq!(
        dvs[0].referenced_data_file,
        b.object_uri("t/data/a.parquet")
    );

    let version = publish(&b, &[a.clone(), c.clone()], &dvs, None, 7)
        .await
        .expect("publish");
    let snapshot = b
        .read_iceberg_snapshot_with_deletes(META, version)
        .await
        .expect("read");
    assert_eq!(snapshot.data_files, vec![a.clone(), c.clone()]);
    assert_eq!(snapshot.deletion_vectors.len(), 2);
    for (vector, (file, expect)) in snapshot.deletion_vectors.iter().zip([(&a, &da), (&c, &dc)]) {
        assert_eq!(&vector.data_file, file);
        assert_eq!(vector.puffin_file, Path::from("t/data/dv-1.puffin"));
        assert_eq!(vector.cardinality, expect.cardinality());
        let read = b.read_deletion_vector(vector).await.expect("read vector");
        assert_eq!(read.positions(), expect.positions());
    }

    // The data-only entry point refuses rather than return the deleted rows.
    let err = b
        .read_iceberg_snapshot(META, version)
        .await
        .expect_err("a reader that cannot apply deletes");
    assert!(err.to_string().contains("deletion vectors"), "{err}");
    let err = b
        .read_iceberg_table(META, version)
        .await
        .expect_err("the bridge trait read too");
    assert!(err.to_string().contains("deletion vectors"), "{err}");
}

#[tokio::test]
async fn a_snapshot_without_deletes_reads_as_before() {
    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[1, 2]).await;
    let version = publish(&b, std::slice::from_ref(&a), &[], None, 1)
        .await
        .expect("publish");
    assert_eq!(
        b.read_iceberg_snapshot(META, version).await.expect("read"),
        vec![a.clone()]
    );
    let snapshot = b
        .read_iceberg_snapshot_with_deletes(META, version)
        .await
        .expect("read");
    assert!(snapshot.deletion_vectors.is_empty());
}

#[tokio::test]
async fn invalid_deletion_vectors_are_refused_at_publish() {
    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[1, 2]).await;
    let stray = part(&b, "t/data/stray.parquet", &[3]).await;
    let one = deletes(&[0]);
    let for_stray = b
        .write_deletion_vectors(&Path::from("t/data/dv-s.puffin"), &[(stray, &one)])
        .await
        .expect("write");
    assert!(
        publish(&b, std::slice::from_ref(&a), &for_stray, None, 1)
            .await
            .is_err(),
        "a vector for a file not in the snapshot"
    );
    let twice = b
        .write_deletion_vectors(
            &Path::from("t/data/dv-2.puffin"),
            &[(a.clone(), &one), (a.clone(), &one)],
        )
        .await
        .expect("write");
    assert!(
        publish(&b, std::slice::from_ref(&a), &twice, None, 1)
            .await
            .is_err(),
        "two vectors for one file"
    );
    let v2 = b
        .write_deletion_vectors(&Path::from("t/data/dv-3.puffin"), &[(a.clone(), &one)])
        .await
        .expect("write");
    assert!(
        b.publish_files_inner(
            std::slice::from_ref(&a),
            &v2,
            None,
            META,
            "uuid-dv",
            "memory:///t",
            &fields(),
            1,
            0,
            iceberg::FormatVersion::V2,
        )
        .await
        .is_err(),
        "format v2 cannot carry deletion vectors"
    );
    // Nothing was committed by the refused publishes.
    assert_eq!(b.latest_metadata_version(META).await.expect("log"), None);
}

#[tokio::test]
async fn a_damaged_vector_is_an_error_not_an_empty_set() {
    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[1, 2, 3]).await;
    let d = deletes(&[2]);
    let dvs = b
        .write_deletion_vectors(&Path::from("t/data/dv.puffin"), &[(a.clone(), &d)])
        .await
        .expect("write");
    let version = publish(&b, &[a], &dvs, None, 3).await.expect("publish");
    let snapshot = b
        .read_iceberg_snapshot_with_deletes(META, version)
        .await
        .expect("read");
    let mut vector = snapshot.deletion_vectors[0].clone();
    vector.cardinality += 1;
    assert!(
        b.read_deletion_vector(&vector).await.is_err(),
        "cardinality"
    );
    vector.cardinality -= 1;
    vector.content_offset += 1;
    assert!(b.read_deletion_vector(&vector).await.is_err(), "range");
}

/// P2.4 slice 3b: the trait view (`read_iceberg_snapshot_files`) maps the
/// engine's vectors — including the Puffin file's size, which a re-publisher
/// needs to re-emit the delete-manifest entry — and `publish_iceberg_files_
/// deleting` carries those view entries into a successor snapshot whose
/// read-back keeps every deleted position deleted (no resurrection).
#[tokio::test]
async fn carried_deletion_vectors_survive_a_successor_publication() {
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;

    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[10, 11, 12, 13]).await;
    let d = deletes(&[1, 3]);
    let dvs = b
        .write_deletion_vectors(&Path::from("t/data/dv-1.puffin"), &[(a.clone(), &d)])
        .await
        .expect("write vectors");
    let v1 = publish(&b, std::slice::from_ref(&a), &dvs, None, 11)
        .await
        .expect("publish v1");

    // The trait view round-trips the vectors (bridge-relative keys + sizes).
    let view = b.read_iceberg_snapshot_files(META, v1).await.expect("view");
    assert_eq!(view.data_files, vec![a.clone()]);
    assert_eq!(view.deletion_vectors.len(), 1);
    let carried = view.deletion_vectors[0].clone();
    assert_eq!(carried.data_file, a);
    assert_eq!(carried.puffin_file, Path::from("t/data/dv-1.puffin"));
    assert_eq!(carried.cardinality, 2);
    assert!(
        carried.puffin_file_size_in_bytes > 0,
        "the re-publish entry needs the Puffin file's size"
    );

    // Carry the view entries into a successor that also adds a new part.
    let extra = part(&b, "t/data/extra.parquet", &[30]).await;
    let view_fields = vec![
        proximadb_storage_common::object_store_bridge::IcebergSnapshotField {
            id: 1,
            name: "age".into(),
            type_name: "long".into(),
            required: false,
        },
    ];
    let v2 = b
        .publish_iceberg_files_deleting(
            &[a.clone(), extra.clone()],
            &[carried],
            Some(v1),
            META,
            "uuid-dv",
            "memory:///t",
            &view_fields,
            12,
            1_700_000_000_001,
        )
        .await
        .expect("publish v2 carrying deletes");

    let successor = b
        .read_iceberg_snapshot_with_deletes(META, v2)
        .await
        .expect("read v2");
    let mut files = successor.data_files.clone();
    files.sort_by(|x, y| x.as_ref().cmp(y.as_ref()));
    assert_eq!(files, vec![a.clone(), extra]);
    assert_eq!(successor.deletion_vectors.len(), 1, "deletes carried");
    let read_back = b
        .read_deletion_vector(&successor.deletion_vectors[0])
        .await
        .expect("carried vector decodes");
    assert_eq!(read_back.positions(), vec![1, 3]);

    // Each snapshot serves its OWN view: v1 still names only its part with its
    // original vector — the successor's carry does not rewrite history.
    let v1_engine = b
        .read_iceberg_snapshot_with_deletes(META, v1)
        .await
        .expect("v1 engine view");
    assert_eq!(v1_engine.data_files, vec![a.clone()]);
    assert_eq!(v1_engine.deletion_vectors.len(), 1);
    let v1_positions = b
        .read_deletion_vector(&v1_engine.deletion_vectors[0])
        .await
        .expect("v1 vector decodes");
    assert_eq!(v1_positions.positions(), vec![1, 3]);
}

/// The never-downgrade FLOOR never upgrades either: a v2 table whose publisher
/// keeps requesting v2 stays v2 across successors (v3 activation is separately
/// gated on lineage + reader compatibility + external delete read-back).
#[tokio::test]
async fn a_v2_table_stays_v2_without_an_explicit_upgrade() {
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;

    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[1, 2]).await;
    let v1 = b
        .publish_iceberg_files(
            std::slice::from_ref(&a),
            None,
            META,
            "uuid-v2",
            "memory:///t",
            &fields(),
            21,
            1_700_000_000_000,
            iceberg::FormatVersion::V2,
        )
        .await
        .expect("publish v1 as v2");
    let v2 = b
        .publish_iceberg_files(
            std::slice::from_ref(&a),
            Some(v1),
            META,
            "uuid-v2",
            "memory:///t",
            &fields(),
            22,
            1_700_000_000_001,
            iceberg::FormatVersion::V2,
        )
        .await
        .expect("publish successor as v2");
    for version in [v1, v2] {
        let bytes = b.read_table_metadata(META, version).await.expect("meta");
        let meta: serde_json::Value = serde_json::from_slice(&bytes).expect("metadata json");
        assert_eq!(
            meta["format-version"], 2,
            "v2 stays v2 without an explicit upgrade (v{version})"
        );
    }
}

/// TD-USUB-10 3c rung 1: a **v2 parent** that carries deletion vectors into its
/// successor is upgraded to v3 ATOMICALLY — one metadata CAS commits the new
/// snapshot AND the format-version advance. The upgrade must preserve the
/// parent chain, the schema's field ids, and the carried data file's path;
/// new files gain row lineage, the carried v2-era file legitimately has none.
#[tokio::test]
async fn a_v2_parent_upgrades_to_v3_atomically_when_it_carries_deletes() {
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge as _;

    let b = bridge();
    let a = part(&b, "t/data/a.parquet", &[10, 11, 12, 13]).await;
    // The parent commits as v2 (no _ns columns, no deletes, no v3 request).
    let parent = b
        .publish_iceberg_files(
            std::slice::from_ref(&a),
            None,
            META,
            "uuid-upg",
            "memory:///t",
            &fields(),
            100,
            100,
            iceberg::FormatVersion::V2,
        )
        .await
        .expect("publish the v2 parent");
    let parent_json = b
        .read_table_metadata(META, parent)
        .await
        .expect("parent metadata");
    let parent_doc: serde_json::Value = serde_json::from_slice(&parent_json).expect("parent json");
    assert_eq!(
        parent_doc["format-version"], 2,
        "precondition: parent is v2"
    );

    // A deletion vector for the v2-era file, carried into a successor append.
    let d = deletes(&[2]);
    let dvs = b
        .write_deletion_vectors(&Path::from("t/data/dv-upg.puffin"), &[(a.clone(), &d)])
        .await
        .expect("write vectors");
    // The engine writes DeleteFileMeta; the trait carries the view struct.
    // One vector, one puffin file: mirror it field by field.
    let carried: Vec<proximadb_storage_common::object_store_bridge::IcebergSnapshotDeletion> = dvs
        .iter()
        .map(
            |m| proximadb_storage_common::object_store_bridge::IcebergSnapshotDeletion {
                data_file: a.clone(),
                puffin_file: Path::from("t/data/dv-upg.puffin"),
                content_offset: m.content_offset as u64,
                content_size_in_bytes: m.content_size_in_bytes as u64,
                puffin_file_size_in_bytes: m.file_size_in_bytes as u64,
                cardinality: m.record_count as u64,
            },
        )
        .collect();
    // The trait's deleting publish takes the bridge view descriptors.
    let view_fields = vec![
        proximadb_storage_common::object_store_bridge::IcebergSnapshotField {
            id: 1,
            name: "age".into(),
            type_name: "long".into(),
            required: false,
        },
    ];
    let extra = part(&b, "t/data/extra-upg.parquet", &[40]).await;
    let child = b
        .publish_iceberg_files_deleting(
            &[a.clone(), extra.clone()],
            &carried,
            Some(parent),
            META,
            "uuid-upg",
            "memory:///t",
            &view_fields,
            200,
            200,
        )
        .await
        .expect("publish the carrying successor");

    let child_json = b
        .read_table_metadata(META, child)
        .await
        .expect("child metadata");
    let child_doc: serde_json::Value = serde_json::from_slice(&child_json).expect("child json");
    assert_eq!(
        child_doc["format-version"], 3,
        "deletes force the v3 upgrade"
    );
    // Atomicity means the SAME commit advanced the version and the snapshot:
    // the child's snapshot chain names the parent's snapshot, and no version
    // exists between them.
    let parent_snapshot_id = parent_doc["current-snapshot-id"].clone();
    let child_snapshot = child_doc["snapshots"]
        .as_array()
        .expect("child snapshots")
        .iter()
        .find(|snap| snap["snapshot-id"] == child_doc["current-snapshot-id"])
        .expect("the current snapshot is listed");
    assert_eq!(
        child_snapshot["parent-snapshot-id"], parent_snapshot_id,
        "the upgrade commit chains directly onto the parent snapshot"
    );
    assert_eq!(
        child_doc["current-snapshot-id"], 200,
        "the child's own snapshot"
    );
    // Field ids survive the upgrade unchanged.
    let parent_ids: Vec<i64> = parent_doc["schemas"][0]["fields"]
        .as_array()
        .expect("parent fields")
        .iter()
        .map(|f| f["id"].as_i64().expect("id"))
        .collect();
    let child_ids: Vec<i64> = child_doc["schemas"][0]["fields"]
        .as_array()
        .expect("child fields")
        .iter()
        .map(|f| f["id"].as_i64().expect("id"))
        .collect();
    assert_eq!(
        parent_ids, child_ids,
        "field ids are preserved by the upgrade"
    );
    assert!(
        child_doc.get("next-row-id").is_some(),
        "v3 allocates row lineage"
    );

    // Versions are dense (successor = parent + 1) and this was ONE commit:
    // the child is exactly the parent's successor, pinning one CAS per publish.
    assert_eq!(
        child,
        parent + 1,
        "the upgrade is a single successor commit"
    );

    // The carried v2-era data file is still referenced at its original path,
    // and the deleted position stays deleted through the upgraded child —
    // not just as carried metadata but to the row-reading entry point.
    let view = b
        .read_iceberg_snapshot_files(META, child)
        .await
        .expect("child view");
    assert!(view.data_files.contains(&a), "carried file keeps its path");
    assert_eq!(view.deletion_vectors.len(), 1);
    let snapshot = b
        .read_iceberg_snapshot_with_deletes(META, child)
        .await
        .expect("read the upgraded child");
    let vector = &snapshot.deletion_vectors[0];
    assert_eq!(&vector.data_file, &a);
    let read = b.read_deletion_vector(vector).await.expect("read vector");
    assert_eq!(
        read.positions(),
        &[2],
        "position 2 is hidden on the v3 child"
    );
}

/// TD-USUB-10 3c (live writer, foundation): map deleted OIDs to their
/// per-file row positions in a PUBLISHED snapshot. The hidden `__proxima_oid`
/// column is written with every published file (P2.4 slice 2a); the delete
/// writer keys on oids and the Puffin DV keys on positions — this is the
/// bridge. Reuses the immutable `OidPositionResolver` primitive.
#[tokio::test]
async fn deleted_oids_map_to_published_positions() {
    let b = bridge();
    // Two files with distinct oids; positions are per-file write order.
    let a = part_with_oid(&b, "t/data/map-a.parquet", &[10, 11, 12]).await;
    let c = part_with_oid(&b, "t/data/map-c.parquet", &[20, 21]).await;
    let _version = b
        .publish_iceberg_files(
            &[a.clone(), c.clone()],
            None,
            META,
            "uuid-map",
            "memory:///t",
            &fields(),
            42,
            42,
            iceberg::FormatVersion::V3,
        )
        .await
        .expect("publish");

    // part() writes oid = r{age}; the mapping is per file, position = write
    // order within that file.
    let mapping = b
        .published_positions_of(&[a.clone(), c.clone()], &["r11", "r20", "r99"])
        .await
        .expect("map oids to positions");
    assert_eq!(
        mapping.per_file.len(),
        2,
        "one entry per file that holds a deleted oid"
    );
    assert_eq!(
        mapping.oids_missing,
        vec!["r99".to_string()],
        "a never-published oid is surfaced top-level, never silently skipped"
    );
    let by_file: std::collections::HashMap<_, _> = mapping
        .per_file
        .into_iter()
        .map(|m| (m.file.clone(), m))
        .collect();
    let in_a = by_file.get(&a).expect("r11 lives in map-a");
    assert_eq!(in_a.positions, &[1], "r11 is the second row of map-a");
    assert_eq!(in_a.oids_found, vec!["r11".to_string()]);
    let in_c = by_file.get(&c).expect("r20 lives in map-c");
    assert_eq!(in_c.positions, &[0], "r20 is the first row of map-c");
}

/// Round-1 review: a file written WITHOUT the hidden oid column (a pre-2a
/// part carried forward by an append) must be an ERROR naming the file, not a
/// panic on a 0-column batch.
#[tokio::test]
async fn a_file_without_the_oid_column_is_an_error_not_a_panic() {
    let b = bridge();
    let legacy = part(&b, "t/data/legacy.parquet", &[1, 2]).await; // no oid column
    let err = b
        .published_positions_of(std::slice::from_ref(&legacy), &["r1"])
        .await
        .expect_err("pre-oid-column files must fail closed");
    assert!(
        err.to_string().contains("__proxima_oid"),
        "the error names the missing column: {err}"
    );
    assert!(
        err.to_string().contains("legacy"),
        "the error names the file: {err}"
    );
}

/// Round-1 review: positions must hold across ROW GROUPS — production
/// MATERIALIZE files are multi-row-group, and a basis bug would bite exactly
/// there. Write a two-row-group file and map an oid in the second group.
#[tokio::test]
async fn positions_span_row_groups() {
    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use std::sync::Arc;

    let b = bridge();
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("__proxima_oid", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("age", arrow_schema::DataType::Int64, false),
    ]));
    let b1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["r1", "r2"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
        ],
    )
    .expect("batch 1");
    let b2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["r3", "r4"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![3, 4])) as ArrayRef,
        ],
    )
    .expect("batch 2");
    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
        &[b1, b2],
        schema,
        Some(props),
    )
    .expect("two-row-group parquet");
    let path = Path::from("t/data/multi-rg.parquet");
    b.store
        .put_if_absent(&path, bytes::Bytes::from(bytes))
        .await
        .expect("put");

    let mapping = b
        .published_positions_of(std::slice::from_ref(&path), &["r3"])
        .await
        .expect("map across row groups");
    assert_eq!(mapping.oids_missing, Vec::<String>::new());
    assert_eq!(mapping.per_file.len(), 1);
    assert_eq!(
        mapping.per_file[0].positions,
        &[2],
        "r3 is the first row of the SECOND row group — file position 2"
    );
}
