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
async fn pinned_producer_metadata_is_independent_of_the_managed_tip() {
    let b = bridge().with_root_url("memory://warehouse");
    let a = part(&b, "t/data/a.parquet", &[10, 11]).await;
    let deletion = deletes(&[1]);
    let dv = b
        .write_deletion_vectors(&Path::from("t/data/dv.puffin"), &[(a.clone(), &deletion)])
        .await
        .unwrap();
    let version = publish(&b, std::slice::from_ref(&a), &dv, None, 7)
        .await
        .unwrap();
    let committer = metadata::MetadataCommitter::new(b.store.clone(), META);
    let mut doc: serde_json::Value =
        serde_json::from_slice(&committer.read_metadata(version).await.unwrap()).unwrap();
    doc["table-uuid"] = serde_json::json!("eeff0011-2233-4455-6677-8899aabbccdd");
    doc["location"] = serde_json::json!(b.object_uri("t"));
    doc["properties"]["converted_delta_version"] = serde_json::json!("42");
    let key = Path::from("t/metadata/00000-producer.metadata.json");
    b.store
        .put(&key, Bytes::from(serde_json::to_vec(&doc).unwrap()))
        .await
        .unwrap();
    let metadata_uri = b.object_uri(key.as_ref());
    let table = Path::from("t");

    let newer = part(&b, "t/data/newer.parquet", &[20]).await;
    publish(&b, &[newer], &[], Some(version), 8).await.unwrap();
    let pinned = b
        .read_pinned_iceberg_snapshot(&table, &metadata_uri, Some(42))
        .await
        .unwrap();
    assert_eq!(pinned.snapshot_id, 7);
    assert_eq!(
        pinned.table_uuid.to_string(),
        "eeff0011-2233-4455-6677-8899aabbccdd"
    );
    assert_eq!(pinned.format_version, 3);
    assert_eq!(pinned.converted_delta_version, Some(42));
    assert_eq!(pinned.plan.files.data_files, vec![a]);
    assert_eq!(pinned.plan.fields[0].id, 1);
    let [vector] = pinned.plan.files.deletion_vectors.as_slice() else {
        panic!("expected one deletion vector");
    };
    assert_eq!(
        b.read_deletion_vector(vector).await.unwrap().positions(),
        deletion.positions()
    );

    for (case, bad_list) in [
        "memory://warehouse/t-other/snap.avro",
        "memory://warehouse/t/../t-other/snap.avro",
        "memory://warehouse/t/%2e%2e/snap.avro",
        "memory://another/t/snap.avro",
    ]
    .into_iter()
    .enumerate()
    {
        doc["snapshots"][0]["manifest-list"] = serde_json::json!(bad_list);
        let malformed_key = Path::from(format!("t/metadata/case-{case}.metadata.json"));
        b.store
            .put(
                &malformed_key,
                Bytes::from(serde_json::to_vec(&doc).unwrap()),
            )
            .await
            .unwrap();
        let metadata_uri = b.object_uri(malformed_key.as_ref());
        let err = b
            .read_pinned_iceberg_snapshot(&table, &metadata_uri, Some(43))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("expected UniForm Delta version 43"),
            "{err}"
        );
        let err = b
            .read_pinned_iceberg_snapshot(&table, &metadata_uri, Some(42))
            .await
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("unsupported Iceberg URI") || message.contains("pinned Iceberg key"),
            "{err}"
        );
    }
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
    let b = bridge().with_root_url("memory://warehouse");
    let table = Path::from("t");
    let location = b.object_uri(table.as_ref());
    let uuid = "eeff0011-2233-4455-6677-8899aabbccdd";
    let a = part(&b, "t/data/a.parquet", &[1, 2]).await;
    let v1 = b
        .publish_iceberg_files(
            std::slice::from_ref(&a),
            None,
            META,
            uuid,
            &location,
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
            uuid,
            &location,
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
    let uri = b.metadata_file_uri(META, v1);
    let pinned = b
        .read_pinned_iceberg_snapshot(&table, &uri, None)
        .await
        .unwrap();
    assert_eq!(pinned.format_version, 2);
    assert_eq!(pinned.snapshot_id, 21);
    assert_eq!(pinned.plan.files.data_files, vec![a]);
    assert!(pinned.plan.files.deletion_vectors.is_empty());
    assert!(pinned.converted_delta_version.is_none());
    let err = b
        .read_pinned_iceberg_snapshot(&table, &uri, Some(42))
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("expected UniForm Delta version 42"),
        "{err}"
    );
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

/// One file with an explicit, NULLABLE `__proxima_oid` column — a foreign
/// writer's shape (ours are non-null) — so position alignment over null cells
/// and duplicate oids can be pinned.
async fn part_with_raw_oids(
    b: &IcebergObjectStoreBridge,
    key: &str,
    oids: &[Option<&str>],
) -> Path {
    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use std::sync::Arc;

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("__proxima_oid", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("age", arrow_schema::DataType::Int64, false),
    ]));
    let ages: Vec<i64> = (0..oids.len() as i64).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(oids.to_vec())) as ArrayRef,
            Arc::new(Int64Array::from(ages)) as ArrayRef,
        ],
    )
    .expect("batch");
    let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
        &[batch],
        schema,
        None,
    )
    .expect("parquet");
    let path = Path::from(key);
    b.store
        .put_if_absent(&path, bytes::Bytes::from(bytes))
        .await
        .expect("put");
    path
}

/// The incremental MATERIALIZE marks EVERY old position of a changed oid, so
/// the exact core must (a) return every occurrence, not the first, and
/// (b) count null cells so later positions stay aligned with the file. The
/// first-occurrence resolver behind `published_positions_of` gets both
/// wrong on this file.
#[tokio::test]
async fn exact_positions_return_every_occurrence_aligned_over_nulls() {
    let b = bridge();
    let path = part_with_raw_oids(
        &b,
        "t/data/exact.parquet",
        &[Some("r1"), None, Some("r2"), Some("r1")],
    )
    .await;
    let exact = b
        .oid_positions_exact(std::slice::from_ref(&path), &["r1", "r2", "r9"])
        .await
        .expect("exact positions");
    assert_eq!(
        exact,
        vec![(path.clone(), vec![0, 2, 3])],
        "r1 at 0 AND 3, r2 at 2 (the null at 1 still counts), r9 nowhere"
    );
    let none = b
        .oid_positions_exact(std::slice::from_ref(&path), &["r9"])
        .await
        .expect("no matches");
    assert!(
        none.is_empty(),
        "a file holding none of the oids is omitted"
    );
}

#[tokio::test]
async fn exact_positions_fail_closed_on_a_file_without_the_oid_column() {
    let b = bridge();
    let legacy = part(&b, "t/data/legacy-exact.parquet", &[1, 2]).await;
    let err = b
        .oid_positions_exact(std::slice::from_ref(&legacy), &["r1"])
        .await
        .expect_err("pre-oid-column files must fail closed");
    assert!(err.to_string().contains("__proxima_oid"), "{err}");
    assert!(err.to_string().contains("legacy-exact"), "{err}");
}

/// The port seam round-trips: positions written through the trait come back
/// through the trait, and each vector names the caller's bridge-relative data
/// file (mapped back by URI, not by output order).
#[tokio::test]
async fn the_deletion_vector_seam_round_trips_positions() {
    use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;

    let b = bridge();
    let a = Path::from("t/data/seam-a.parquet");
    let c = Path::from("t/data/seam-c.parquet");
    let written = b
        .write_deletion_vector_positions(
            &Path::from("t/data/dv-mat-seam.puffin"),
            &[(c.clone(), vec![7, 1]), (a.clone(), vec![0, 3, 3])],
        )
        .await
        .expect("write through the seam");
    assert_eq!(written.len(), 2);
    for vector in &written {
        let positions = b
            .read_deletion_vector_positions(vector)
            .await
            .expect("read through the seam");
        let want: Vec<u64> = if vector.data_file == a {
            vec![0, 3]
        } else {
            vec![1, 7]
        };
        assert_eq!(positions, want, "{:?}", vector.data_file);
        assert_eq!(vector.cardinality, want.len() as u64);
        assert_eq!(vector.puffin_file, Path::from("t/data/dv-mat-seam.puffin"));
    }
}

#[tokio::test]
async fn pinned_reader_projects_field_ids_and_applies_physical_deletes() {
    use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};

    let b = bridge().with_root_url("memory://warehouse");
    let table = Path::from("t");
    let old = Path::from("t/data/old.parquet");
    let current = Path::from("t/data/current.parquet");
    let no_ids = Path::from("t/data/without-ids.parquet");
    let field = |name, data_type, id: i32| {
        Field::new(name, data_type, false)
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
    };
    let old_file = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            field("old_name", DataType::Int32, 7),
            // The current name has been reused for a different field ID.
            field("value", DataType::Utf8, 9),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![10, 11, 12])),
            Arc::new(StringArray::from(vec!["old-a", "deleted", "old-c"])),
        ],
    )
    .unwrap();
    let new_file = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            field("extra", DataType::Int64, 12),
            field("label", DataType::Utf8, 9),
            field("value", DataType::Int64, 7),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![100])),
            Arc::new(StringArray::from(vec!["current"])),
            Arc::new(Int64Array::from(vec![20])),
        ],
    )
    .unwrap();
    let file_without_ids = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![99]))],
    )
    .unwrap();
    for (path, batch) in [
        (&old, old_file),
        (&current, new_file),
        (&no_ids, file_without_ids),
    ] {
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        b.store.put(path, Bytes::from(bytes)).await.unwrap();
    }

    let dv = b
        .write_deletion_vectors(
            &Path::from("t/data/filter.puffin"),
            &[(old.clone(), &deletes(&[1]))],
        )
        .await
        .unwrap();
    let version = publish(&b, &[old, current], &dv, None, 7).await.unwrap();
    let committer = metadata::MetadataCommitter::new(b.store.clone(), META);
    let mut doc: serde_json::Value =
        serde_json::from_slice(&committer.read_metadata(version).await.unwrap()).unwrap();
    doc["table-uuid"] = serde_json::json!("eeff0011-2233-4455-6677-8899aabbccdd");
    doc["location"] = serde_json::json!(b.object_uri("t"));
    doc["schemas"][0]["fields"] = serde_json::json!([
        {"id": 9, "name": "label", "type": "string", "required": true},
        {"id": 12, "name": "extra", "type": "long", "required": false},
        {"id": 7, "name": "value", "type": "long", "required": true},
    ]);
    let key = Path::from("t/metadata/00007-producer.metadata.json");
    b.store
        .put(&key, Bytes::from(serde_json::to_vec(&doc).unwrap()))
        .await
        .unwrap();
    let reader = b
        .open_pinned_iceberg_reader(&table, &b.object_uri(key.as_ref()), None)
        .await
        .unwrap();
    let batches: Vec<_> = reader
        .scan(Some(&[2, 0, 1]), 1)
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        assert_eq!(batch.schema().field(0).name(), "value");
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let labels = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let extra = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        rows.extend((0..batch.num_rows()).map(|i| {
            (
                values.value(i),
                labels.value(i),
                (!extra.is_null(i)).then(|| extra.value(i)),
            )
        }));
    }
    rows.sort_unstable();
    assert_eq!(
        rows,
        vec![
            (10, "old-a", None),
            (12, "old-c", None),
            (20, "current", Some(100))
        ]
    );
    let count_batches: Vec<_> = reader
        .scan(Some(&[]), 1)
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert!(count_batches.iter().all(|batch| batch.num_columns() == 0));
    assert_eq!(
        count_batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        3
    );

    // An older file cannot stand in for an omitted partition value or default.
    for (name, column_override) in [
        ("default", serde_json::json!({"initial-default": 50})),
        ("required", serde_json::json!({"required": true})),
    ] {
        let mut unsupported = doc.clone();
        unsupported["schemas"][0]["fields"][1]
            .as_object_mut()
            .unwrap()
            .extend(column_override.as_object().unwrap().clone());
        let key = Path::from(format!("t/metadata/{name}.metadata.json"));
        b.store
            .put(&key, Bytes::from(serde_json::to_vec(&unsupported).unwrap()))
            .await
            .unwrap();
        let reader = b
            .open_pinned_iceberg_reader(&table, &b.object_uri(key.as_ref()), None)
            .await
            .unwrap();
        let err = reader
            .scan(None, 2)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("missing field ID 12 requires a value"),
            "{err}"
        );
    }

    let mut identity_partitioned = doc.clone();
    identity_partitioned["partition-specs"][0]["fields"] = serde_json::json!([
        {"source-id": 12, "field-id": 1000, "name": "extra", "transform": "identity"}
    ]);
    let key = Path::from("t/metadata/identity.metadata.json");
    b.store
        .put(
            &key,
            Bytes::from(serde_json::to_vec(&identity_partitioned).unwrap()),
        )
        .await
        .unwrap();
    let reader = b
        .open_pinned_iceberg_reader(&table, &b.object_uri(key.as_ref()), None)
        .await
        .unwrap();
    let err = reader
        .scan(None, 2)
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("missing field ID 12 requires a value"),
        "{err}"
    );

    let bad_version = publish(&b, &[no_ids], &[], Some(version), 8).await.unwrap();
    let bad: serde_json::Value =
        serde_json::from_slice(&committer.read_metadata(bad_version).await.unwrap()).unwrap();
    doc["snapshots"] = bad["snapshots"].clone();
    doc["current-snapshot-id"] = bad["current-snapshot-id"].clone();
    let key = Path::from("t/metadata/no-ids.metadata.json");
    b.store
        .put(&key, Bytes::from(serde_json::to_vec(&doc).unwrap()))
        .await
        .unwrap();
    let reader = b
        .open_pinned_iceberg_reader(&table, &b.object_uri(key.as_ref()), None)
        .await
        .unwrap();
    let err = reader
        .scan(None, 2)
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Parquet column value has no field ID"),
        "{err}"
    );
}
