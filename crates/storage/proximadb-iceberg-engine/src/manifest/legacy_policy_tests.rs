use super::*;
use object_store::memory::InMemory;
use std::sync::Arc;
use std::time::Duration;

const PREFIX: &str = "data/t/ns/migration";

fn legacy(store: ProximaObjectStore) -> ManifestCommitter {
    ManifestCommitter::new(store, PREFIX).with_legacy_encoding(LegacyEncoding::GenerationPrefixed)
}

#[tokio::test]
async fn legacy_retention_is_disabled_without_deleting_history() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store);
    super::tests::seed_fenced(&c, 6, 5).await;
    assert!(c.prune_retention(2, Duration::ZERO).await.is_err());
    for v in 0..6 {
        assert_eq!(c.read_fenced(v).await.unwrap().0, 5);
    }
}

#[tokio::test]
async fn legacy_pruned_log_blocks_even_current_parent_writes() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store.clone());
    super::tests::seed_fenced(&c, 6, 5).await;
    store.delete(&c.manifest_path(0)).await.unwrap();
    assert!(matches!(
        c.commit_fenced(Some(5), 5, Bytes::new()).await,
        Err(StorageError::TransactionCommitFailed(_))
    ));
    assert_eq!(c.read_fenced(5).await.unwrap().0, 5);
    assert_eq!(c.latest_version().await.unwrap(), Some(5));
}

#[tokio::test]
async fn legacy_plain_and_fenced_formats_are_declared_not_guessed() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let plain = ManifestCommitter::new(store.clone(), "data/t/ns/plain");
    let bytes = Bytes::from_static(br#"{"checkpoint":"source-cursor"}"#);
    plain.commit(None, bytes.clone()).await.unwrap();
    assert_eq!(plain.read_fenced(0).await.unwrap(), (0, bytes));
    assert!(plain.commit_fenced(Some(0), 5, Bytes::new()).await.is_err());
    let fenced = legacy(store);
    assert!(
        fenced
            .commit(None, Bytes::from_static(b"ambiguous plain"))
            .await
            .is_err()
    );
    fenced.commit_fenced(None, 5, Bytes::new()).await.unwrap();
    assert_eq!(fenced.read_fenced(0).await.unwrap(), (5, Bytes::new()));
}

#[tokio::test]
async fn legacy_migration_preserves_tip_generation_and_retained_payloads() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store.clone());
    super::tests::seed_fenced(&c, 6, 5).await;
    store.delete(&c.manifest_path(0)).await.unwrap();
    let migrated = c.migrate_versioned(5).await.unwrap();
    assert_eq!(migrated.latest_version().await.unwrap(), Some(5));
    for v in 1..6 {
        assert_eq!(
            migrated.read_fenced(v).await.unwrap(),
            (5, Bytes::from_static(b"g"))
        );
    }
    assert_eq!(
        migrated
            .commit_fenced(Some(5), 6, Bytes::from_static(b"successor"))
            .await
            .unwrap(),
        CommitOutcome::Committed(6)
    );
    assert_eq!(
        migrated.prune_retention(2, Duration::ZERO).await.unwrap(),
        4
    );
    // Migration never deletes source records; explicit old-format inspection remains possible.
    assert!(
        store
            .get(&Path::from(format!(
                "{PREFIX}/v00000000000000000005.manifest"
            )))
            .await
            .is_ok()
    );
    let reopened = ManifestCommitter::open_versioned(store, PREFIX)
        .await
        .unwrap();
    assert_eq!(reopened.read_fenced(6).await.unwrap().0, 6);
}

#[tokio::test]
async fn legacy_migration_rejects_stale_expected_tip_before_creating_marker() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store.clone());
    super::tests::seed_fenced(&c, 2, 5).await;
    assert!(c.migrate_versioned(0).await.is_err());
    assert!(matches!(
        store
            .get(&Path::from(format!("{PREFIX}/_publication.format")))
            .await,
        Err(StorageError::NotFound(_))
    ));
    assert_eq!(legacy(store).latest_version().await.unwrap(), Some(1));
}

#[tokio::test]
async fn legacy_nested_and_noncanonical_names_do_not_count_as_history() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store.clone());
    super::tests::seed_fenced(&c, 2, 5).await;
    store.delete(&c.manifest_path(0)).await.unwrap();
    for suffix in [
        "nested/v00000000000000000000.manifest",
        "v0.manifest",
        "nested/v00000000000000000999.manifest",
    ] {
        store
            .put(
                &Path::from(format!("{PREFIX}/{suffix}")),
                Bytes::from_static(b"noise"),
            )
            .await
            .unwrap();
    }
    assert_eq!(c.latest_version().await.unwrap(), Some(1));
    assert!(c.commit_fenced(Some(1), 5, Bytes::new()).await.is_err());
}

#[tokio::test]
async fn legacy_empty_log_pruning_is_disabled() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store.clone());
    assert!(matches!(
        c.prune_retention(2, Duration::ZERO).await,
        Err(StorageError::TransactionCommitFailed(_))
    ));
    assert!(
        store
            .list(Some(&Path::from(PREFIX)))
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn legacy_interior_gap_blocks_current_parent_without_hiding_reads() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = legacy(store.clone());
    super::tests::seed_fenced(&c, 4, 5).await;
    store.delete(&c.manifest_path(1)).await.unwrap();
    assert!(matches!(
        c.commit_fenced(Some(3), 6, Bytes::new()).await,
        Err(StorageError::TransactionCommitFailed(_))
    ));
    assert_eq!(c.read_fenced(0).await.unwrap().0, 5);
    assert_eq!(c.read_fenced(3).await.unwrap().0, 5);
    assert!(matches!(
        store.get(&c.manifest_path(4)).await,
        Err(StorageError::NotFound(_))
    ));
}

#[tokio::test]
async fn legacy_truncated_declared_generation_fails_before_migration_marker() {
    for len in 0..8 {
        let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
        let c = legacy(store.clone());
        let source = Bytes::from(vec![0x41; len]);
        store
            .put(&c.manifest_path(0), source.clone())
            .await
            .unwrap();
        assert!(matches!(
            c.read_fenced(0).await,
            Err(StorageError::Corruption(_))
        ));
        assert!(matches!(
            c.commit_fenced(Some(0), 5, Bytes::new()).await,
            Err(StorageError::Corruption(_))
        ));
        assert_eq!(c.read_manifest(0).await.unwrap(), source);
        assert!(matches!(
            c.migrate_versioned(0).await,
            Err(StorageError::Corruption(_))
        ));
        assert!(matches!(
            store
                .get(&Path::from(format!("{PREFIX}/_publication.format")))
                .await,
            Err(StorageError::NotFound(_))
        ));
        assert_eq!(
            store.list(Some(&Path::from(PREFIX))).await.unwrap().len(),
            1
        );
    }
}

#[tokio::test]
async fn legacy_migration_repeat_after_advancement_preserves_current_authority() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let source = legacy(store.clone());
    super::tests::seed_fenced(&source, 3, 5).await;
    let migrated = source.migrate_versioned(2).await.unwrap();
    migrated
        .commit_fenced(Some(2), 6, Bytes::from_static(b"advanced"))
        .await
        .unwrap();
    let head_path = Path::from(format!("{PREFIX}/_publication.head"));
    let current_head = store.get(&head_path).await.unwrap();
    let repeated = legacy(store.clone()).migrate_versioned(2).await.unwrap();
    assert_eq!(repeated.latest_version().await.unwrap(), Some(3));
    assert_eq!(
        repeated.read_fenced(3).await.unwrap(),
        (6, Bytes::from_static(b"advanced"))
    );
    assert_eq!(store.get(&head_path).await.unwrap(), current_head);
    assert!(legacy(store.clone()).migrate_versioned(3).await.is_err());
    assert!(
        ManifestCommitter::new(store.clone(), PREFIX)
            .migrate_versioned(2)
            .await
            .is_err()
    );
    assert_eq!(store.get(&head_path).await.unwrap(), current_head);
}

#[tokio::test]
async fn legacy_migrated_log_routes_existing_and_reopened_serving_handles() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let existing = legacy(store.clone());
    super::tests::seed_fenced(&existing, 2, 5).await;
    legacy(store.clone()).migrate_versioned(1).await.unwrap();
    assert_eq!(
        existing
            .commit_fenced(Some(1), 6, Bytes::from_static(b"existing"))
            .await
            .unwrap(),
        CommitOutcome::Committed(2)
    );
    let reopened = ManifestCommitter::new(store.clone(), PREFIX);
    assert_eq!(
        reopened.read_fenced(2).await.unwrap(),
        (6, Bytes::from_static(b"existing"))
    );
    assert_eq!(
        reopened
            .commit_fenced(Some(2), 6, Bytes::from_static(b"reopened"))
            .await
            .unwrap(),
        CommitOutcome::Committed(3)
    );
    assert_eq!(existing.latest_version().await.unwrap(), Some(3));
    assert!(matches!(
        store.get(&existing.manifest_path(2)).await,
        Err(StorageError::NotFound(_))
    ));
    assert!(matches!(
        store.get(&existing.manifest_path(3)).await,
        Err(StorageError::NotFound(_))
    ));
}

#[tokio::test]
async fn legacy_plain_migration_preserves_long_payload_and_zero_generation() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let c = ManifestCommitter::new(store.clone(), PREFIX);
    let payload = Bytes::from_static(br#"{"checkpoint":"source-cursor"}"#);
    c.commit(None, payload.clone()).await.unwrap();
    let migrated = c.migrate_versioned(0).await.unwrap();
    assert_eq!(migrated.read_fenced(0).await.unwrap(), (0, payload.clone()));
    assert_eq!(
        migrated
            .commit_fenced(Some(0), 7, Bytes::new())
            .await
            .unwrap(),
        CommitOutcome::Committed(1)
    );
    assert_eq!(migrated.read_fenced(0).await.unwrap(), (0, payload.clone()));
    assert_eq!(
        store.get(&migrated.manifest_path(0)).await.unwrap(),
        payload
    );
}
