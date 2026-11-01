#[cfg(test)]
mod tests {
    use crate::storage::engines::sst::{SstConfig, SstEngine};
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use crate::storage::trait_components::path_resolver::ConfigFallbackResolver;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use std::sync::Arc;

    /// TD-SSTCLEAN-1: DROP COLLECTION's SST cleanup must delete exactly the
    /// dropped collection's segment files and leave every other collection's
    /// files untouched. The unfixed code listed the filesystem ROOT and deleted
    /// by bare basename, so it never reached the collection's own files (a
    /// silent leak) and was one flat layout away from deleting another
    /// collection's files. This test pins the isolation property both ways:
    /// the dropped collection's files are gone AND the surviving collection's
    /// files are all still present.
    #[tokio::test]
    async fn cleanup_collection_files_deletes_only_the_dropped_collection() {
        let base = tempfile::tempdir().expect("unique tempdir");
        let root = base.path().to_path_buf();

        let mut fs_config = crate::storage::persistence::filesystem::FilesystemConfig::default();
        fs_config.local = Some(
            crate::storage::persistence::filesystem::local::LocalConfig {
                root_dir: Some(root.clone()),
                ..Default::default()
            },
        );
        let filesystem = Arc::new(FilesystemFactory::create(fs_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());
        let engine = SstEngine::new_with_config(SstConfig::default(), filesystem, distance_compute)
            .await
            .unwrap()
            .with_path_resolver(Arc::new(ConfigFallbackResolver::new(format!(
                "file://{}",
                root.display()
            ))));

        // Plant segment files for two collections under each one's `…/data`
        // prefix — the extensions the readers recognize (.sst/.proximablock/
        // .pax) plus the ancillary .bloom/.meta/.idx this cleanup has always
        // claimed to remove.
        let segment_files = [
            "L0_20261031T000000_aa00.sst",
            "L0_20261031T000000_aa00.bloom",
            "L0_20261031T000000_aa00.meta",
            "L0_20261031T000000_aa00.idx",
            "L00_20261031T000001_bb11.pax",
        ];
        for collection in ["dropped_collection", "surviving_collection"] {
            let data_dir = root.join(collection).join("data");
            std::fs::create_dir_all(&data_dir).unwrap();
            for file in segment_files {
                std::fs::write(data_dir.join(file), b"payload").unwrap();
            }
        }

        engine
            .cleanup_collection_files("dropped_collection")
            .await
            .unwrap();

        // The dropped collection's segment files are gone.
        let dropped_dir = root.join("dropped_collection").join("data");
        for file in segment_files {
            assert!(
                !dropped_dir.join(file).exists(),
                "dropped collection's {file} must be deleted"
            );
        }

        // The surviving collection's files are ALL still present.
        let surviving_dir = root.join("surviving_collection").join("data");
        for file in segment_files {
            assert!(
                surviving_dir.join(file).exists(),
                "surviving collection's {file} must NOT be deleted by dropping another collection"
            );
        }
    }

    #[tokio::test]
    async fn test_collection_stats() {
        let engine = create_test_engine().await;
        let stats = engine.collection_stats("test_collection").unwrap();

        assert_eq!(stats["collection_id"], "test_collection");
        assert_eq!(stats["engine"], "sst");
    }

    #[tokio::test]
    async fn test_collection_metadata() {
        let engine = create_test_engine().await;
        let metadata = engine.collection_metadata("test_collection").unwrap();

        assert_eq!(metadata["collection_id"], "test_collection");
        assert_eq!(metadata["engine"], "sst");
        assert_eq!(metadata["storage_format"], "sstable");
    }

    #[tokio::test]
    async fn test_get_collection_storage_url() {
        let engine = create_test_engine().await;
        let url = engine
            .get_collection_storage_url("test_collection")
            .await
            .unwrap();

        assert!(url.contains("test_collection"));
    }

    #[tokio::test]
    async fn test_collection_size_info() {
        let engine = create_test_engine().await;
        let size_info = engine.get_collection_size("test_collection").await.unwrap();

        assert_eq!(size_info.collection_id, "test_collection");
        assert_eq!(size_info.total_size_bytes, 0); // No files in test
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }
}
