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
    let v1 = publish(&b, &[a.clone()], &dvs, None, 11)
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
