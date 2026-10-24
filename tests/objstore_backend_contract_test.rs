//! TD-OBJSTORE-5 S2: per-primitive contract tests against the PRODUCTION root
//! `FileSystem` backends (src/storage/persistence/filesystem/*) over the real
//! emulator wire APIs. These pin the exact object-store semantics the recovery
//! stack depends on (ADR-063 D4/D5):
//!
//! * PUT → prefix-LIST sees the object (LIST is the discovery authority).
//! * `exists()` on a directory PREFIX is FALSE while LIST is positive — the
//!   flat-keyspace fact that made prefix-`exists()`-before-LIST silently drop
//!   durable WAL batches (TD-OBJSTORE-1 batch 3). If a backend ever changes
//!   this, the forbidden-pattern rationale changes with it.
//! * LIST returns ALL objects across pages (the production GCS backend once
//!   silently truncated at page 1).
//!
//! Emulator-gated (`#[ignore]` + env skip), same pattern as the
//! `put_with_tier_*` tier tests; run via `scripts/run_cloud_emulator_tests.sh`.

use std::sync::Arc;

use proximadb::storage::persistence::filesystem::{
    FilesystemConfig, FilesystemError, FilesystemFactory,
};

async fn factory() -> Arc<FilesystemFactory> {
    Arc::new(
        FilesystemFactory::create(FilesystemConfig::default())
            .await
            .expect("filesystem factory"),
    )
}

/// PUT under a fresh prefix → LIST(prefix) must see it; exists(prefix) must be
/// FALSE (flat keyspace: HEAD on a prefix key matches nothing) while the child
/// object exists and LIST proves it.
async fn contract_prefix_semantics(base: &str) {
    let run = uuid::Uuid::new_v4().simple().to_string();
    let prefix = format!("{base}/contract/{run}/wal");
    let key = format!("{prefix}/8dWbZDDUPo.bcwal");

    let factory = factory().await;
    let fs = factory.get_filesystem(&key).expect("backend for key");
    fs.write(&key, b"batch-bytes", None)
        .await
        .expect("PUT object");

    // LIST authority: the child is discoverable under the prefix.
    let listed = fs.list(&prefix).await.expect("LIST prefix");
    assert!(
        listed.iter().any(|e| e.url.ends_with("8dWbZDDUPo.bcwal")),
        "PUT object must be visible to prefix LIST; got {:?}",
        listed.iter().map(|e| e.url.as_str()).collect::<Vec<_>>()
    );

    // Flat-keyspace fact: exists() on the PREFIX is false (HEAD-on-key), even
    // though LIST is positive — the reason prefix-exists() gates are forbidden.
    let prefix_exists = fs.exists(&prefix).await.unwrap_or(false);
    assert!(
        !prefix_exists,
        "exists({prefix}) returned true — flat-keyspace prefix semantics \
         changed; revisit ADR-063 D5's forbidden-pattern rationale"
    );

    // And exists() on the exact KEY is true (it is a real object).
    assert!(
        fs.exists(&key).await.expect("exists(key)"),
        "exists() on the exact object key must be true"
    );

    let _ = fs.delete(&key).await;
}

/// Publish enough bytes to require multiple multipart chunks, then prove the
/// object visible through the production backend is byte-for-byte identical.
async fn contract_local_file_multipart_publish(base: &str) {
    const BYTES: usize = 17 * 1024 * 1024 + 19;

    let run = uuid::Uuid::new_v4().simple().to_string();
    let key = format!("{base}/contract/{run}/multi-part.pax");
    let scratch = tempfile::tempdir().expect("scratch directory");
    let local_path = scratch.path().join("segment.pax");
    let payload = (0..BYTES)
        .map(|index| ((index * 31 + 7) % 251) as u8)
        .collect::<Vec<_>>();
    tokio::fs::write(&local_path, &payload)
        .await
        .expect("write local source");

    let factory = factory().await;
    let fs = factory.get_filesystem(&key).expect("backend for key");
    assert!(
        fs.supports_bounded_local_file_write(),
        "{base} must advertise bounded local-file publication before spill admission"
    );
    let written = fs
        .write_local_file(&key, &local_path, None)
        .await
        .expect("multipart publish local file");
    assert_eq!(written, BYTES as u64);
    assert_eq!(fs.read(&key).await.expect("read published object"), payload);

    let _ = fs.delete(&key).await;
}

#[tokio::test]
#[ignore = "needs Azurite — set PROXIMADB_AZURE_EMULATOR=1 with Azurite running"]
async fn azure_prefix_list_and_exists_contract() {
    if std::env::var("PROXIMADB_AZURE_EMULATOR").is_err() {
        eprintln!("skip: set PROXIMADB_AZURE_EMULATOR=1 with Azurite running");
        return;
    }
    contract_prefix_semantics("az://proximadb-test").await;
}

#[tokio::test]
#[ignore = "needs Azurite — set PROXIMADB_AZURE_EMULATOR=1 with Azurite running"]
async fn azure_local_file_multipart_publish_contract() {
    if std::env::var("PROXIMADB_AZURE_EMULATOR").is_err() {
        eprintln!("skip: set PROXIMADB_AZURE_EMULATOR=1 with Azurite running");
        return;
    }
    contract_local_file_multipart_publish("az://proximadb-test").await;
}

#[tokio::test]
#[ignore = "needs MinIO — set AWS_ENDPOINT to the emulator"]
async fn s3_prefix_list_and_exists_contract() {
    if std::env::var("AWS_ENDPOINT").is_err() {
        eprintln!("skip: set AWS_ENDPOINT (MinIO) to run");
        return;
    }
    contract_prefix_semantics("s3://proximadb-test").await;
}

#[tokio::test]
#[ignore = "needs MinIO — set AWS_ENDPOINT to the emulator"]
async fn s3_local_file_multipart_publish_contract() {
    if std::env::var("AWS_ENDPOINT").is_err() {
        eprintln!("skip: set AWS_ENDPOINT (MinIO) to run");
        return;
    }
    contract_local_file_multipart_publish("s3://proximadb-test").await;
}

// GCS is best-effort (documented fake-gcs/object_store incompatibilities; never a
// gate — TD-OBJSTORE-5 nightly tier). The backend needs `PROXIMADB_GCS_ENDPOINT`
// + `PROXIMADB_GCS_ANONYMOUS=1` to register against fake-gcs; if it does not
// register, warn and pass rather than block.
#[tokio::test]
#[ignore = "needs fake-gcs — set PROXIMADB_GCS_ENDPOINT + PROXIMADB_GCS_ANONYMOUS=1"]
async fn gcs_prefix_list_and_exists_contract_best_effort() {
    if std::env::var("PROXIMADB_GCS_ENDPOINT").is_err() {
        eprintln!("skip: set PROXIMADB_GCS_ENDPOINT (fake-gcs) to run");
        return;
    }
    let factory = factory().await;
    if factory.get_filesystem("gs://proximadb-test/probe").is_err() {
        eprintln!("::warning:: GCS backend did not register (best-effort tier) — skipping");
        return;
    }
    contract_prefix_semantics("gs://proximadb-test").await;
}

#[tokio::test]
#[ignore = "needs fake-gcs — set PROXIMADB_GCS_ENDPOINT + PROXIMADB_GCS_ANONYMOUS=1"]
async fn gcs_local_file_resumable_publish_contract_best_effort() {
    if std::env::var("PROXIMADB_GCS_ENDPOINT").is_err() {
        eprintln!("skip: set PROXIMADB_GCS_ENDPOINT (fake-gcs) to run");
        return;
    }
    let factory = factory().await;
    let Ok(fs) = factory.get_filesystem("gs://proximadb-test/probe") else {
        eprintln!("::warning:: GCS backend did not register (best-effort tier) — skipping");
        return;
    };
    assert!(
        fs.supports_bounded_local_file_write(),
        "GCS spill publication must advertise only its bounded resumable path"
    );
    contract_local_file_multipart_publish("gs://proximadb-test").await;
}

/// GCS pagination contract: with the page size pinned to 2, five objects must
/// ALL come back — the production backend once read a single `list_objects`
/// page and ignored `next_page_token`, silently truncating LIST-authority
/// discovery (TD-OBJSTORE-4 S1).
#[tokio::test]
#[ignore = "needs fake-gcs — set PROXIMADB_GCS_ENDPOINT + PROXIMADB_GCS_ANONYMOUS=1"]
async fn gcs_multi_page_list_returns_all_objects_best_effort() {
    if std::env::var("PROXIMADB_GCS_ENDPOINT").is_err() {
        eprintln!("skip: set PROXIMADB_GCS_ENDPOINT (fake-gcs) to run");
        return;
    }
    // SAFETY-free env set: test-scoped page size for the pagination loop.
    unsafe { std::env::set_var("PROXIMADB_GCS_LIST_PAGE_SIZE", "2") };

    let run = uuid::Uuid::new_v4().simple().to_string();
    let prefix = format!("gs://proximadb-test/contract-paging/{run}");
    let factory = factory().await;
    let Ok(fs) = factory.get_filesystem(&prefix) else {
        eprintln!("::warning:: GCS backend did not register (best-effort tier) — skipping");
        unsafe { std::env::remove_var("PROXIMADB_GCS_LIST_PAGE_SIZE") };
        return;
    };
    for i in 0..5 {
        fs.write(&format!("{prefix}/obj-{i}.bin"), b"x", None)
            .await
            .expect("PUT paged object");
    }
    let listed = fs.list(&prefix).await.expect("LIST paged prefix");
    unsafe { std::env::remove_var("PROXIMADB_GCS_LIST_PAGE_SIZE") };
    assert_eq!(
        listed.len(),
        5,
        "multi-page LIST must return all 5 objects (page size 2); got {:?}",
        listed.iter().map(|e| e.url.as_str()).collect::<Vec<_>>()
    );
    for i in 0..5 {
        let _ = fs.delete(&format!("{prefix}/obj-{i}.bin")).await;
    }
}

/// `write_if_absent` is the recovery COMMIT primitive (ADR-063 D4/D5, TD-OBJSTORE-4
/// S2/S3): the first create wins, and a second create on the SAME key must fail
/// `AlreadyExists` WITHOUT clobbering the committed bytes. A backend that silently
/// overwrote would make crash-window materialization non-idempotent — a re-replay
/// after a crash would greenwash a duplicate commit over the real one. This pins the
/// exact put-if-absent semantics per cloud backend (S3 `PutMode::Create`, Azure
/// `PutMode::Create`, GCS `if_generation_match=0`).
///
/// `strict` gates the collision assertion. Azurite and MinIO enforce put-if-absent,
/// so Azure/S3 assert it as a hard gate. fake-gcs, however, does NOT enforce
/// `ifGenerationMatch=0` (a documented emulator gap — our backend sets the
/// precondition and maps 409/412 to `AlreadyExists`, see `gcs_store.rs`), so the GCS
/// tier runs best-effort: it warns rather than fails when the emulator lets the
/// colliding create through. Real-GCS put-if-absent conformance is validated in the
/// nightly tier (TD-OBJSTORE-5), never against fake-gcs.
async fn contract_write_if_absent_semantics(base: &str, strict: bool) {
    let run = uuid::Uuid::new_v4().simple().to_string();
    let key = format!("{base}/contract-cas/{run}/L0_recovery.pax");

    let factory = factory().await;
    let fs = factory.get_filesystem(&key).expect("backend for key");

    // First conditional create commits the object.
    fs.write_if_absent(&key, b"first-commit", None)
        .await
        .expect("first conditional create must succeed");

    // Second conditional create on the same key must be rejected, not overwrite.
    match fs.write_if_absent(&key, b"second-commit", None).await {
        // Conformant: the first-committed bytes must survive the rejected create.
        Err(FilesystemError::AlreadyExists(_)) => {
            assert_eq!(
                fs.read(&key).await.expect("read back the committed object"),
                b"first-commit",
                "first-committed bytes must survive the rejected second create"
            );
        }
        // Best-effort tier (GCS/fake-gcs): the emulator did not enforce the
        // precondition. Our backend sets `ifGenerationMatch=0`; warn, do not gate.
        Ok(()) if !strict => {
            eprintln!(
                "::warning:: {base}: emulator did not enforce put-if-absent on \
                 write_if_absent (fake-gcs lacks ifGenerationMatch=0 enforcement); \
                 backend sets the precondition — real-backend conformance runs in the \
                 nightly tier"
            );
        }
        other => {
            panic!("backend must reject the colliding create with AlreadyExists; got {other:?}")
        }
    }

    let _ = fs.delete(&key).await;
}

#[tokio::test]
#[ignore = "needs Azurite — set PROXIMADB_AZURE_EMULATOR=1 with Azurite running"]
async fn azure_write_if_absent_contract() {
    if std::env::var("PROXIMADB_AZURE_EMULATOR").is_err() {
        eprintln!("skip: set PROXIMADB_AZURE_EMULATOR=1 with Azurite running");
        return;
    }
    contract_write_if_absent_semantics("az://proximadb-test", true).await;
}

#[tokio::test]
#[ignore = "needs MinIO — set AWS_ENDPOINT to the emulator"]
async fn s3_write_if_absent_contract() {
    if std::env::var("AWS_ENDPOINT").is_err() {
        eprintln!("skip: set AWS_ENDPOINT (MinIO) to run");
        return;
    }
    contract_write_if_absent_semantics("s3://proximadb-test", true).await;
}

// GCS is best-effort (documented fake-gcs/object_store incompatibilities; never a
// gate — TD-OBJSTORE-5 nightly tier). If the backend does not register against
// fake-gcs, warn and pass rather than block.
#[tokio::test]
#[ignore = "needs fake-gcs — set PROXIMADB_GCS_ENDPOINT + PROXIMADB_GCS_ANONYMOUS=1"]
async fn gcs_write_if_absent_contract_best_effort() {
    if std::env::var("PROXIMADB_GCS_ENDPOINT").is_err() {
        eprintln!("skip: set PROXIMADB_GCS_ENDPOINT (fake-gcs) to run");
        return;
    }
    let factory = factory().await;
    if factory.get_filesystem("gs://proximadb-test/probe").is_err() {
        eprintln!("::warning:: GCS backend did not register (best-effort tier) — skipping");
        return;
    }
    contract_write_if_absent_semantics("gs://proximadb-test", false).await;
}

/// A missing object is `NotFound` — the variant callers use to tell "absent"
/// from "unreachable", as on the local backend. Reporting it as `Network`
/// made the record spill refuse every fresh partition on object storage (its
/// first discovery reads a purge-obligation marker that does not exist yet).
/// A delete of a missing object may be a no-op (S3 answers 204) or
/// `NotFound`; it must not be `Network`.
async fn contract_missing_object_is_not_found(base: &str) {
    let run = uuid::Uuid::new_v4().simple().to_string();
    let key = format!("{base}/contract/{run}/never-written.json");
    let factory = factory().await;
    let fs = factory.get_filesystem(&key).expect("backend for key");
    match fs.read(&key).await {
        Err(FilesystemError::NotFound(_)) => {}
        other => panic!("read of a missing object: expected NotFound, got {other:?}"),
    }
    match fs.read_range(&key, 0, 4).await {
        Err(FilesystemError::NotFound(_)) => {}
        other => panic!("ranged read of a missing object: expected NotFound, got {other:?}"),
    }
    match fs.delete(&key).await {
        Ok(()) | Err(FilesystemError::NotFound(_)) => {}
        other => panic!("delete of a missing object: expected Ok or NotFound, got {other:?}"),
    }
    assert!(!fs.exists(&key).await.expect("exists answers"));
    assert!(
        fs.list(&format!("{base}/contract/{run}/"))
            .await
            .expect("an empty prefix lists")
            .is_empty()
    );
}

#[tokio::test]
#[ignore = "needs Azurite — set PROXIMADB_AZURE_EMULATOR=1 with Azurite running"]
async fn azure_missing_object_is_not_found_contract() {
    if std::env::var("PROXIMADB_AZURE_EMULATOR").is_err() {
        eprintln!("skip: set PROXIMADB_AZURE_EMULATOR=1 with Azurite running");
        return;
    }
    contract_missing_object_is_not_found("az://proximadb-test").await;
}

#[tokio::test]
#[ignore = "needs MinIO — set AWS_ENDPOINT to the emulator"]
async fn s3_missing_object_is_not_found_contract() {
    if std::env::var("AWS_ENDPOINT").is_err() {
        eprintln!("skip: set AWS_ENDPOINT (MinIO) to run");
        return;
    }
    contract_missing_object_is_not_found("s3://proximadb-test").await;
}

#[tokio::test]
#[ignore = "needs fake-gcs — set PROXIMADB_GCS_ENDPOINT + PROXIMADB_GCS_ANONYMOUS=1"]
async fn gcs_missing_object_is_not_found_contract_best_effort() {
    if std::env::var("PROXIMADB_GCS_ENDPOINT").is_err() {
        eprintln!("skip: set PROXIMADB_GCS_ENDPOINT (fake-gcs) to run");
        return;
    }
    let factory = factory().await;
    if factory.get_filesystem("gs://proximadb-test/probe").is_err() {
        eprintln!("::warning:: GCS backend did not register (best-effort tier) — skipping");
        return;
    }
    contract_missing_object_is_not_found("gs://proximadb-test").await;
}
