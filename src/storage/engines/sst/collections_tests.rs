#[cfg(test)]
mod tests {
    use crate::storage::engines::sst::{SstConfig, SstEngine};
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use std::sync::Arc;

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
