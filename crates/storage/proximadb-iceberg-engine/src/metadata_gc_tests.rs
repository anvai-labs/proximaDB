//! TD-SPINE-3 acceptance through the real bridge (in-memory store): the
//! generation window is the only thing that decides what metadata survives.

use std::collections::BTreeSet;
use std::sync::Arc;

use futures::StreamExt;
use object_store::memory::InMemory;
use proximadb_object_store::ProximaObjectStore;
use proximadb_records::{ProximaRecord, ProximaTreeNode, ProximaValue};
use proximadb_storage_common::object_store_bridge::ObjectStoreBridge;

use super::*;
use crate::deletion_vector::PositionDeletes;

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

/// Publish `versions` MATERIALIZE-style generations: version `v` is an
/// overwrite listing the shared base file plus its own part `g{v}`; version 2
/// also carries a deletion vector on the base (a delete manifest `-m1`).
async fn publish_generations(b: &IcebergObjectStoreBridge, versions: u64) -> Vec<Path> {
    let base = part(b, "t/data/base.parquet", &[1, 2, 3]).await;
    let mut parts = Vec::new();
    let mut parent = None;
    for v in 0..versions {
        let own = part(b, &format!("t/data/g{v}.parquet"), &[100 + v as i64]).await;
        parts.push(own.clone());
        let files = vec![base.clone(), own];
        let committed = if v == 2 {
            let mut d = PositionDeletes::new();
            d.insert(0);
            let dvs = b
                .write_deletion_vectors(&Path::from("t/data/dv-2.puffin"), &[(base.clone(), &d)])
                .await
                .expect("write dv");
            b.publish_iceberg_files_with_deletes(
                &files,
                &dvs,
                parent,
                META,
                "uuid-gc",
                "memory:///t",
                &fields(),
                1000 + v as i64,
                1_700_000_000_000 + v as i64,
            )
            .await
        } else {
            b.publish_iceberg_files(
                &files,
                parent,
                META,
                "uuid-gc",
                "memory:///t",
                &fields(),
                1000 + v as i64,
                1_700_000_000_000 + v as i64,
                iceberg::FormatVersion::V3,
            )
            .await
        }
        .expect("publish generation");
        assert_eq!(committed, v);
        parent = Some(committed);
    }
    parts
}

/// File names directly under the metadata prefix.
async fn metadata_names(b: &IcebergObjectStoreBridge) -> BTreeSet<String> {
    b.store
        .list(Some(&Path::from(META)))
        .await
        .expect("list")
        .into_iter()
        .filter_map(|m| m.location.filename().map(String::from))
        .collect()
}

fn generation_names(versions: impl Iterator<Item = u64>, dv_version: u64) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for v in versions {
        let sid = 1000 + v;
        out.insert(format!("v{v:020}.metadata.json"));
        out.insert(format!("v{v:020}-{sid}-m0.avro"));
        out.insert(format!("snap-{sid}-v{v:020}-list.avro"));
        if v == dv_version {
            out.insert(format!("v{v:020}-{sid}-m1.avro"));
        }
    }
    out
}

async fn read_rows(b: &IcebergObjectStoreBridge, version: u64) -> Result<usize, StorageError> {
    let snapshot = b.read_iceberg_snapshot_with_deletes(META, version).await?;
    let mut rows = 0;
    for file in &snapshot.data_files {
        let mut stream = b
            .read_parquet_batches(file, Arc::new(arrow_schema::Schema::empty()), 1024, None)
            .await?;
        while let Some(batch) = stream.next().await {
            rows += batch?.num_rows();
        }
    }
    for dv in &snapshot.deletion_vectors {
        rows -= b.read_deletion_vector(dv).await?.cardinality() as usize;
    }
    Ok(rows)
}

#[tokio::test]
async fn gc_keeps_the_tip_and_the_window_and_collects_everything_older() {
    let b = bridge();
    publish_generations(&b, 6).await;
    assert_eq!(metadata_names(&b).await, generation_names(0..6, 2));

    // committed v5, window 3 → v3, v4, v5 retained.
    let report = b.gc_metadata(META, 5, 3).await.expect("gc");
    assert_eq!(report.skipped, None);
    assert_eq!(report.failed, 0);
    assert_eq!(metadata_names(&b).await, generation_names(3..6, 2));
    // v0, v1: 3 objects each; v2: 4 (its delete manifest).
    assert_eq!(report.deleted, 10);

    // A reader pinned to the OLDEST retained generation still reads, through
    // its own metadata, manifest list and manifests.
    assert_eq!(read_rows(&b, 3).await.expect("pinned v3 reads"), 4);
    // Negative control: the collected generation really is gone.
    assert!(read_rows(&b, 2).await.is_err(), "v2 must be collected");

    // Idempotent and clock-free: a second pass over the same state deletes
    // nothing more.
    let again = b.gc_metadata(META, 5, 3).await.expect("gc again");
    assert_eq!(again.deleted, 0);
}

#[tokio::test]
async fn gc_never_deletes_newer_versions_or_in_flight_manifests() {
    let b = bridge();
    publish_generations(&b, 7).await;
    // An in-flight publisher's manifest for the slot after the tip: written
    // before its metadata CAS, so no document reaches it yet.
    let in_flight = format!("v{:020}-9999-m0.avro", 7);
    b.store
        .put(
            &Path::from(format!("{META}/{in_flight}")),
            bytes::Bytes::from_static(b"in-flight"),
        )
        .await
        .expect("in-flight manifest");
    // A slow MATERIALIZE that committed v4 collects with ITS window (v2..v4);
    // v5 and v6, committed by others meanwhile, are newer and untouched.
    let report = b.gc_metadata(META, 4, 3).await.expect("gc");
    assert_eq!(report.skipped, None);
    let mut expected = generation_names(2..7, 2);
    expected.insert(in_flight);
    assert_eq!(metadata_names(&b).await, expected);
}

#[tokio::test]
async fn an_unreadable_retained_version_skips_the_whole_pass() {
    let b = bridge();
    publish_generations(&b, 5).await;
    let before = metadata_names(&b).await;
    // Corrupt a RETAINED version's manifest list: reachability can no longer
    // be proven, so nothing may be deleted.
    b.store
        .put(
            &Path::from(format!("{META}/snap-1003-v{:020}-list.avro", 3)),
            bytes::Bytes::from_static(b"not avro"),
        )
        .await
        .expect("corrupt");
    let report = b.gc_metadata(META, 4, 3).await.expect("gc");
    assert!(report.skipped.is_some(), "{report:?}");
    assert_eq!(report.deleted, 0);
    assert_eq!(metadata_names(&b).await, before);
}

#[tokio::test]
async fn a_manifest_a_retained_version_still_reaches_is_kept() {
    let b = bridge();
    publish_generations(&b, 4).await;
    // Rewrite v3's metadata so its snapshot points at v0's manifest list
    // (a foreign writer re-using an old manifest). v0's list and manifest
    // are below the floor by name, but reachable: both must survive.
    let mut doc: serde_json::Value =
        serde_json::from_slice(&b.read_table_metadata(META, 3).await.expect("v3")).expect("json");
    let v0_list = b.object_uri(&format!("{META}/snap-1000-v{:020}-list.avro", 0));
    doc["snapshots"][0]["manifest-list"] = serde_json::json!(v0_list);
    b.store
        .put(
            &Path::from(format!("{META}/v{:020}.metadata.json", 3)),
            bytes::Bytes::from(doc.to_string()),
        )
        .await
        .expect("rewrite v3");
    let report = b.gc_metadata(META, 3, 2).await.expect("gc");
    assert_eq!(report.skipped, None);
    let names = metadata_names(&b).await;
    assert!(names.contains(&format!("snap-1000-v{:020}-list.avro", 0)));
    assert!(names.contains(&format!("v{:020}-1000-m0.avro", 0)));
    // The unreachable old documents still go.
    assert!(!names.contains(&format!("v{:020}.metadata.json", 0)));
}

#[tokio::test]
async fn the_metadata_log_holds_exactly_the_window_and_every_entry_reads() {
    let b = bridge();
    publish_generations(&b, 6).await;
    // Default window (3): the tip's log names exactly v3 and v4.
    let tip: serde_json::Value =
        serde_json::from_slice(&b.read_table_metadata(META, 5).await.expect("v5")).expect("json");
    let log: Vec<String> = tip["metadata-log"]
        .as_array()
        .expect("metadata-log")
        .iter()
        .filter_map(|e| e["metadata-file"].as_str().map(String::from))
        .collect();
    assert_eq!(
        log,
        vec![b.metadata_file_uri(META, 3), b.metadata_file_uri(META, 4)]
    );
    // After GC every entry the log lists is still readable (none names a
    // collected snapshot).
    b.gc_metadata(META, 5, 3).await.expect("gc");
    for v in [3, 4] {
        assert!(read_rows(&b, v).await.is_ok(), "log entry v{v} must read");
    }
}

#[tokio::test]
async fn previous_versions_max_only_lowers_the_log_bound() {
    let b = bridge();
    publish_generations(&b, 2).await;
    // Set the property on the tip (v1) to 1 and 100 in turn; the successor's
    // log is min(window - 1, property) and the property is carried forward.
    for (property, expected_len) in [("1", 1usize), ("100", 2)] {
        let tip = b
            .latest_metadata_version(META)
            .await
            .expect("tip")
            .expect("some");
        let mut doc: serde_json::Value =
            serde_json::from_slice(&b.read_table_metadata(META, tip).await.expect("tip doc"))
                .expect("json");
        doc["properties"][retention::PREVIOUS_VERSIONS_MAX_PROPERTY] = serde_json::json!(property);
        b.store
            .put(
                &Path::from(format!("{META}/v{tip:020}.metadata.json")),
                bytes::Bytes::from(doc.to_string()),
            )
            .await
            .expect("set property");
        let own = part(&b, &format!("t/data/p{tip}.parquet"), &[7]).await;
        let next = b
            .publish_iceberg_files(
                &[own],
                Some(tip),
                META,
                "uuid-gc",
                "memory:///t",
                &fields(),
                2000 + tip as i64,
                1_800_000_000_000,
                iceberg::FormatVersion::V3,
            )
            .await
            .expect("publish successor");
        let child: serde_json::Value =
            serde_json::from_slice(&b.read_table_metadata(META, next).await.expect("child"))
                .expect("json");
        assert_eq!(
            child["metadata-log"].as_array().map(Vec::len),
            Some(expected_len),
            "property {property}"
        );
        assert_eq!(
            child["properties"][retention::PREVIOUS_VERSIONS_MAX_PROPERTY],
            serde_json::json!(property)
        );
    }
}
