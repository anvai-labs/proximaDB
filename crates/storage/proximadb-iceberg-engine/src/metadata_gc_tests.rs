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
async fn v2_head_discovers_only_the_retained_generation_window() {
    let b = bridge().with_root_url("memory://warehouse");
    let table_root = b.object_uri("t");
    let file = part(&b, "t/data/part.parquet", &[1]).await;
    for version in 0u64..5 {
        b.publish_iceberg_files(
            std::slice::from_ref(&file),
            version.checked_sub(1),
            META,
            "uuid-gc",
            &table_root,
            &fields(),
            1000 + version as i64,
            1_700_000_000_000 + version as i64,
            iceberg::FormatVersion::V2,
        )
        .await
        .unwrap();
        let head: serde_json::Value =
            serde_json::from_slice(&b.read_table_metadata(META, version).await.unwrap()).unwrap();
        let expected: Vec<_> = (version.saturating_sub(2)..=version)
            .map(|v| 1000 + v as i64)
            .collect();
        for key in ["snapshots", "snapshot-log"] {
            let actual: Vec<_> = head[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|snapshot| snapshot["snapshot-id"].as_i64().unwrap())
                .collect();
            assert_eq!(actual, expected, "{key} at v{version}");
        }
        let report = b.gc_metadata(META, version, 3).await.unwrap();
        assert_eq!(report.skipped, None);
        assert_eq!(
            metadata_names(&b).await,
            generation_names(version.saturating_sub(2)..=version, u64::MAX)
        );
    }
}

#[tokio::test]
async fn v2_history_starts_over_at_a_schema_change() {
    let b = bridge().with_root_url("memory://warehouse");
    let table_root = b.object_uri("t");
    let file = part(&b, "t/data/part.parquet", &[1]).await;
    let mut schema = fields();
    for (version, expected) in [(0u64, vec![1000]), (1, vec![1001]), (2, vec![1001, 1002])] {
        if version == 1 {
            schema.push(iceberg::IcebergField {
                id: 2,
                name: "name".into(),
                type_name: "string".into(),
                required: false,
            });
        }
        b.publish_iceberg_files(
            std::slice::from_ref(&file),
            version.checked_sub(1),
            META,
            "uuid-gc",
            &table_root,
            &schema,
            1000 + version as i64,
            1_700_000_000_000 + version as i64,
            iceberg::FormatVersion::V2,
        )
        .await
        .unwrap();
        let head: serde_json::Value =
            serde_json::from_slice(&b.read_table_metadata(META, version).await.unwrap()).unwrap();
        let pins: Vec<_> = head["snapshots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["snapshot-id"].as_i64().unwrap())
            .collect();
        assert_eq!(pins, expected);
    }
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

/// Review M1: a publisher whose parent read predates a GC can win the
/// create-only CAS on a COLLECTED slot below the tip. That must surface as a
/// conflict (the caller never flips its catalog layout to it), the tip must
/// not move, and the orphaned document must be removed.
#[tokio::test]
async fn a_stale_publisher_winning_a_collected_slot_is_a_conflict() {
    let b = bridge();
    publish_generations(&b, 6).await;
    // GC collected v1's slot while the stale publisher still holds parent v0
    // (its parent read happened before the GC).
    b.store
        .delete(&Path::from(format!("{META}/v{:020}.metadata.json", 1)))
        .await
        .expect("collect v1");
    let own = part(&b, "t/data/stale.parquet", &[9]).await;
    let err = b
        .publish_iceberg_files(
            &[own],
            Some(0),
            META,
            "uuid-gc",
            "memory:///t",
            &fields(),
            5000,
            1_900_000_000_000,
            iceberg::FormatVersion::V3,
        )
        .await
        .expect_err("a below-tip slot win is a conflict");
    assert!(matches!(err, StorageError::AlreadyExists(_)), "{err:?}");
    assert_eq!(
        b.latest_metadata_version(META).await.expect("tip"),
        Some(5),
        "the tip does not move"
    );
    assert!(
        !metadata_names(&b)
            .await
            .contains(&format!("v{:020}.metadata.json", 1)),
        "the orphaned below-tip document is removed"
    );
    // Control: an ordinary successor of the tip still commits.
    let next = part(&b, "t/data/next.parquet", &[10]).await;
    let v = b
        .publish_iceberg_files(
            &[next],
            Some(5),
            META,
            "uuid-gc",
            "memory:///t",
            &fields(),
            6000,
            1_900_000_000_001,
            iceberg::FormatVersion::V3,
        )
        .await
        .expect("successor of the tip commits");
    assert_eq!(v, 6);
}

/// Review M2: a LIST that omits a version inside the window (here the
/// committed one) cannot prove what that version reaches, so nothing is
/// collected.
#[tokio::test]
async fn a_listing_missing_a_window_version_skips_gc() {
    for missing in [5u64, 4] {
        let b = bridge();
        publish_generations(&b, 6).await;
        b.store
            .delete(&Path::from(format!("{META}/v{missing:020}.metadata.json")))
            .await
            .expect("hide a window version");
        let before = metadata_names(&b).await;
        let report = b.gc_metadata(META, 5, 3).await.expect("gc");
        assert!(report.skipped.is_some(), "missing v{missing}: {report:?}");
        assert_eq!(report.deleted, 0);
        assert_eq!(metadata_names(&b).await, before);
    }
}

/// An in-memory store that fails GETs and/or deletes of one exact path.
#[derive(Debug)]
struct FaultStore {
    inner: Arc<InMemory>,
    fail_get: std::sync::Mutex<Option<Path>>,
    fail_delete: std::sync::Mutex<Option<Path>>,
}

impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FaultStore")
    }
}

fn injected(what: &str) -> object_store::Error {
    object_store::Error::Generic {
        store: "FaultStore",
        source: Box::new(std::io::Error::other(format!("injected {what} failure"))),
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: object_store::PutPayload,
        options: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(path, payload, options).await
    }
    async fn put_multipart_opts(
        &self,
        p: &Path,
        o: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(p, o).await
    }
    async fn get_opts(
        &self,
        p: &Path,
        o: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        if self
            .fail_get
            .lock()
            .map(|g| g.as_ref() == Some(p))
            .unwrap_or(false)
        {
            return Err(injected("GET"));
        }
        self.inner.get_opts(p, o).await
    }
    fn delete_stream(
        &self,
        paths: futures::stream::BoxStream<'static, object_store::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
        let fail = self.fail_delete.lock().ok().and_then(|g| g.clone());
        let inner = self.inner.clone();
        Box::pin(paths.then(move |p| {
            let fail = fail.clone();
            let inner = inner.clone();
            async move {
                let p = p?;
                if fail.as_ref() == Some(&p) {
                    return Err(injected("DELETE"));
                }
                object_store::ObjectStoreExt::delete(inner.as_ref(), &p).await?;
                Ok(p)
            }
        }))
    }
    fn list(
        &self,
        p: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(p)
    }
    async fn list_with_delimiter(
        &self,
        p: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(p).await
    }
    async fn copy_opts(
        &self,
        a: &Path,
        b: &Path,
        o: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(a, b, o).await
    }
}

fn fault_bridge() -> (IcebergObjectStoreBridge, Arc<FaultStore>) {
    let store = Arc::new(FaultStore {
        inner: Arc::new(InMemory::new()),
        fail_get: std::sync::Mutex::new(None),
        fail_delete: std::sync::Mutex::new(None),
    });
    let b = IcebergObjectStoreBridge::new(ProximaObjectStore::new(store.clone()))
        .with_root_url("memory:///");
    (b, store)
}

fn meta_path(v: u64) -> Path {
    Path::from(format!("{META}/v{v:020}.metadata.json"))
}

async fn stale_publish(b: &IcebergObjectStoreBridge, parent: u64) -> Result<u64, StorageError> {
    let own = part(b, &format!("t/data/stale-{parent}.parquet"), &[9]).await;
    b.publish_iceberg_files(
        &[own],
        Some(parent),
        META,
        "uuid-gc",
        "memory:///t",
        &fields(),
        5000,
        1_900_000_000_000,
        iceberg::FormatVersion::V3,
    )
    .await
}

/// Review LOW (a): the stale publisher lands in a freed slot AND the next
/// version is also collected (missing) — still a conflict, never a commit.
#[tokio::test]
async fn a_below_tip_win_whose_successor_is_also_collected_is_a_conflict() {
    let b = bridge();
    publish_generations(&b, 6).await;
    for v in [1, 2] {
        b.store.delete(&meta_path(v)).await.expect("collect");
    }
    let err = stale_publish(&b, 0).await.expect_err("conflict");
    assert!(matches!(err, StorageError::AlreadyExists(_)), "{err:?}");
    assert_eq!(b.latest_metadata_version(META).await.expect("tip"), Some(5));
    assert!(
        !metadata_names(&b)
            .await
            .contains(&format!("v{:020}.metadata.json", 1))
    );
}

/// Review LOW (b): a transient GET error on the next version proves nothing,
/// so the commit is kept (never deleted as an orphan) — same rule as a
/// failed LIST.
#[tokio::test]
async fn a_transient_successor_read_error_keeps_the_commit() {
    let (b, store) = fault_bridge();
    publish_generations(&b, 6).await;
    b.store.delete(&meta_path(1)).await.expect("collect v1");
    *store.fail_get.lock().expect("lock") = Some(meta_path(2));
    let v = stale_publish(&b, 0)
        .await
        .expect("kept on an unproven read");
    assert_eq!(v, 1);
    assert!(
        metadata_names(&b)
            .await
            .contains(&format!("v{:020}.metadata.json", 1))
    );
    assert_eq!(
        below_tip_evidence(&b, META, 1, None).await,
        TipEvidence::Unknown
    );
    // Once the read succeeds, the same state is positively a below-tip win.
    *store.fail_get.lock().expect("lock") = None;
    assert_eq!(
        below_tip_evidence(&b, META, 1, None).await,
        TipEvidence::NotBuiltOn { latest: 5 }
    );
}

/// Review MED (c): the orphan delete fails, so the below-tip document stays.
/// The publisher still sees a conflict, and the shared evidence rule that
/// MATERIALIZE's `check_commit` uses classifies the leftover as NOT on the
/// line of history (so it is never `Landed`, never flipped).
#[tokio::test]
async fn a_failed_orphan_delete_still_never_reads_as_landed() {
    let (b, store) = fault_bridge();
    publish_generations(&b, 6).await;
    b.store.delete(&meta_path(1)).await.expect("collect v1");
    *store.fail_delete.lock().expect("lock") = Some(meta_path(1));
    let err = stale_publish(&b, 0).await.expect_err("conflict");
    assert!(matches!(err, StorageError::AlreadyExists(_)), "{err:?}");
    assert!(
        metadata_names(&b)
            .await
            .contains(&format!("v{:020}.metadata.json", 1)),
        "the injected failure left the orphan in place"
    );
    assert_eq!(
        below_tip_evidence(&b, META, 1, None).await,
        TipEvidence::NotBuiltOn { latest: 5 }
    );
    // Controls: a version on the line of history is BuiltOn, the tip AtTip.
    assert_eq!(
        below_tip_evidence(&b, META, 4, None).await,
        TipEvidence::BuiltOn
    );
    assert_eq!(
        below_tip_evidence(&b, META, 5, None).await,
        TipEvidence::AtTip
    );
}
