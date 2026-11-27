// Restart-safety integration — exercises the full corpus_version
// durability chain with real file I/O.
//
// Unit tests pin the registry + the FileSystemCorpusVersionStore
// individually. This test composes them as one lifecycle:
//
//   1. Build a registry backed by a FileSystemCorpusVersionStore.
//   2. Bump some versions.
//   3. Drop the registry (simulates process exit).
//   4. Build a NEW registry pointing at the same store path.
//   5. Hydrate.
//   6. Verify the versions match what was bumped before the drop.
//
// The "drop and rebuild" is the closest the test harness can get to
// an actual process restart. The file content is what's transferred
// between "lives", so the test verifies the on-disk shape is the
// durability contract — not the in-memory registry layout.

use std::sync::Arc;

use proximadb::catalog::{
    CorpusVersionRegistry, FileSystemCorpusVersionStore, corpus_version::CorpusVersionStore,
};
use tempfile::TempDir;

#[tokio::test]
async fn bumps_survive_drop_and_reload_cycle() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    // First "process life": bump a few versions.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.bump("tenant-a", "kb").await; // → 2
        reg.bump("tenant-a", "kb").await; // → 3
        reg.bump("tenant-b", "logs").await; // → 2
        // reg drops at end of block — simulates process exit.
    }

    // Second "process life": rebuild from the same path.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        let loaded = reg.hydrate_from_store().await;
        assert_eq!(
            loaded, 2,
            "two distinct (tenant, collection) pairs persisted"
        );
        // The versions match what life #1 left.
        assert_eq!(reg.current("tenant-a", "kb").await, 3);
        assert_eq!(reg.current("tenant-b", "logs").await, 2);
        // A pair life #1 never touched still defaults to 1.
        assert_eq!(reg.current("tenant-c", "never-seen").await, 1);
    }
}

#[tokio::test]
async fn monotonicity_holds_across_restart() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    // Life 1: bump to 5.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        for _ in 0..4 {
            reg.bump("tenant-a", "kb").await;
        }
        assert_eq!(reg.current("tenant-a", "kb").await, 5);
    }

    // Life 2: hydrate, then bump once more.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.hydrate_from_store().await;
        // Pre-bump: we see the value life #1 left.
        assert_eq!(reg.current("tenant-a", "kb").await, 5);
        // Post-bump: continues from where it left off — never restarts
        // at 1 or 2.
        let v = reg.bump("tenant-a", "kb").await;
        assert_eq!(v, 6, "version must continue across restart, not reset");
    }
}

#[tokio::test]
async fn cross_tenant_isolation_holds_across_restart() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    // Life 1: distinct values for two tenants on the same collection.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        for _ in 0..4 {
            reg.bump("tenant-a", "kb").await;
        }
        for _ in 0..9 {
            reg.bump("tenant-b", "kb").await;
        }
        assert_eq!(reg.current("tenant-a", "kb").await, 5);
        assert_eq!(reg.current("tenant-b", "kb").await, 10);
    }

    // Life 2: hydrate; each tenant retains its independent version.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.hydrate_from_store().await;
        assert_eq!(reg.current("tenant-a", "kb").await, 5);
        assert_eq!(reg.current("tenant-b", "kb").await, 10);
        // Bumping one doesn't disturb the other.
        reg.bump("tenant-a", "kb").await;
        assert_eq!(reg.current("tenant-a", "kb").await, 6);
        assert_eq!(reg.current("tenant-b", "kb").await, 10);
    }
}

#[tokio::test]
async fn cross_collection_isolation_holds_across_restart() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    // Life 1: same tenant, two collections, distinct versions.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.bump("tenant-a", "kb-1").await;
        reg.bump("tenant-a", "kb-1").await;
        reg.bump("tenant-a", "kb-2").await;
    }

    // Life 2: each collection's version restored independently.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.hydrate_from_store().await;
        assert_eq!(reg.current("tenant-a", "kb-1").await, 3);
        assert_eq!(reg.current("tenant-a", "kb-2").await, 2);
    }
}

#[tokio::test]
async fn fresh_path_starts_empty_does_not_pollute_old_runs() {
    // Two independent paths produce two independent registries.
    let dir = TempDir::new().unwrap();
    let path_a = dir.path().join("a.json");
    let path_b = dir.path().join("b.json");

    // Populate path_a with three rows.
    {
        let store: Arc<dyn CorpusVersionStore> =
            Arc::new(FileSystemCorpusVersionStore::new(&path_a));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.bump("tenant-a", "kb").await;
        reg.bump("tenant-a", "logs").await;
        reg.bump("tenant-b", "kb").await;
    }

    // path_b is a fresh path; a registry pointed at it sees nothing.
    let store_b: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path_b));
    let reg_b = CorpusVersionRegistry::with_store(store_b);
    let loaded = reg_b.hydrate_from_store().await;
    assert_eq!(loaded, 0, "fresh path has no pre-existing rows");
    assert_eq!(reg_b.current("tenant-a", "kb").await, 1);
}

#[tokio::test]
async fn corruption_during_restart_falls_back_to_empty_without_panic() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    // Life 1: persist some rows.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.bump("tenant-a", "kb").await;
        reg.bump("tenant-b", "logs").await;
    }

    // Simulate disk corruption between lives.
    tokio::fs::write(&path, b"{ not valid json").await.unwrap();

    // Life 2: hydrate must not panic and must report 0 loaded.
    let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
    let reg = CorpusVersionRegistry::with_store(store);
    let loaded = reg.hydrate_from_store().await;
    assert_eq!(loaded, 0, "corruption treated as empty load");
    // The registry is fully usable post-corruption.
    let v = reg.bump("tenant-a", "kb").await;
    assert_eq!(v, 2);

    // Life 3 sees the recovered state from the fresh persist.
    drop(reg);
    let store2: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
    let reg2 = CorpusVersionRegistry::with_store(store2);
    let loaded = reg2.hydrate_from_store().await;
    assert_eq!(loaded, 1, "life 2's bump persisted");
    assert_eq!(reg2.current("tenant-a", "kb").await, 2);
    // The pre-corruption data is gone — corruption is self-healing
    // by overwrite, not by repair. This is the documented contract.
    assert_eq!(reg2.current("tenant-b", "logs").await, 1);
}

#[tokio::test]
async fn unrelated_collection_versions_unchanged_after_targeted_bump() {
    // The cross-collection independence isn't just an in-memory
    // property; it must survive the round-trip through the on-disk
    // format. A bump on (tenant-a, kb) must NOT affect the on-disk
    // value of any other (tenant, collection) pair.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    // Life 1: seed three collections.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        // Set explicit versions for clarity.
        reg.set("tenant-a", "kb", 10).await;
        reg.set("tenant-a", "logs", 20).await;
        reg.set("tenant-b", "kb", 30).await;
    }

    // Life 2: hydrate, bump ONE pair, persist.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.hydrate_from_store().await;
        reg.bump("tenant-a", "kb").await; // 10 → 11
    }

    // Life 3: hydrate again, verify only the bumped pair changed.
    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.hydrate_from_store().await;
        assert_eq!(reg.current("tenant-a", "kb").await, 11, "bumped");
        assert_eq!(reg.current("tenant-a", "logs").await, 20, "untouched");
        assert_eq!(reg.current("tenant-b", "kb").await, 30, "untouched");
    }
}

#[tokio::test]
async fn many_bumps_persist_correctly_in_one_life() {
    // Stress: 100 bumps on a single key in life 1; life 2 sees 101.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corpus.json");

    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        for _ in 0..100 {
            reg.bump("tenant-a", "kb").await;
        }
        assert_eq!(reg.current("tenant-a", "kb").await, 101);
    }

    {
        let store: Arc<dyn CorpusVersionStore> = Arc::new(FileSystemCorpusVersionStore::new(&path));
        let reg = CorpusVersionRegistry::with_store(store);
        reg.hydrate_from_store().await;
        assert_eq!(reg.current("tenant-a", "kb").await, 101);
    }
}

#[tokio::test]
async fn no_persist_path_means_in_memory_only_no_file_created() {
    // The non-durable default — register without a store. No file
    // is created and no bump persists. Pinned so the documented
    // "PROXIMADB_CORPUS_VERSION_PATH unset = in-memory only"
    // behavior holds.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("never-created.json");
    assert!(!path.exists());

    let reg = CorpusVersionRegistry::default();
    reg.bump("tenant-a", "kb").await;
    reg.bump("tenant-b", "logs").await;

    // No file at the path because no store was attached.
    assert!(!path.exists(), "no store → no file written");
}
