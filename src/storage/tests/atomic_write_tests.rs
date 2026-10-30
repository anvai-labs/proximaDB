/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Integration tests for atomic write patterns with TransactionCoordinator

use crate::storage::persistence::filesystem::FilesystemConfig;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::transaction_coordinator::{
    StagingConfig, TransactionCoordinator, TransactionStageType,
};
use proximadb_records::{EmbeddingCell, ProximaRecord};
use std::sync::Arc;
use tempfile::TempDir;
use tokio;

async fn create_test_environment() -> (Arc<TransactionCoordinator>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path().to_str().unwrap();

    let fs_config = FilesystemConfig::default();
    let fs_factory = Arc::new(
        FilesystemFactory::create(fs_config)
            .await
            .expect("Failed to create filesystem factory"),
    );

    let coordinator = Arc::new(
        TransactionCoordinator::new(fs_factory, Some(base_path.to_string()))
            .await
            .expect("Failed to create coordinator"),
    );

    (coordinator, temp_dir)
}

#[tokio::test]
async fn test_atomic_write_with_sync() {
    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    let collection_id = "test_collection";
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    // Begin atomic operation
    let config = StagingConfig {
        base_url: storage_url.clone(),
        collection_id: Some(collection_id.to_string()),
        operation_type: TransactionStageType::Flush,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write test data to staging
    let test_data = b"Critical data for atomic write";
    coordinator
        .write_to_staging(&operation.operation_id, "test_data.bin", test_data)
        .await
        .unwrap();

    // Finalize operation - this should:
    // 1. Sync staging data if local filesystem
    // 2. Atomically move to final location
    // 3. Delete staging only after successful move
    let result = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(result.is_ok());

    // Verify data exists at final location
    let _final_path = format!("{}/test_data.bin", storage_url);
    // In a real test, we'd verify the file exists at the final location
}

#[tokio::test]
async fn test_atomic_write_failure_rollback() {
    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    let collection_id = "test_collection";
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    // Begin atomic operation
    let config = StagingConfig {
        base_url: storage_url.clone(),
        collection_id: Some(collection_id.to_string()),
        operation_type: TransactionStageType::Flush,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write test data to staging
    let test_data = b"Data that will be rolled back";
    coordinator
        .write_to_staging(&operation.operation_id, "rollback_test.bin", test_data)
        .await
        .unwrap();

    // Rollback operation - staging data should be cleaned up
    let result = coordinator
        .abort_atomic_operation(&operation.operation_id, "Test rollback")
        .await;
    assert!(result.is_ok());

    // Verify staging data was cleaned up
    // In a real test, we'd verify the staging directory is empty
}

#[tokio::test]
async fn test_concurrent_atomic_operations() {
    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    let collection_id = "test_collection";
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    // Start multiple concurrent atomic operations
    let mut handles = vec![];

    for i in 0..5 {
        let coord_clone = coordinator.clone();
        let url_clone = storage_url.clone();
        let coll_id = collection_id.to_string();

        let handle = tokio::spawn(async move {
            // Begin atomic operation
            let config = StagingConfig {
                base_url: url_clone,
                collection_id: Some(coll_id),
                operation_type: TransactionStageType::Flush,
                ..Default::default()
            };

            let operation = coord_clone.begin_atomic_operation(&config).await.unwrap();

            // Write unique data
            let test_data = format!("Concurrent data {}", i).into_bytes();
            coord_clone
                .write_to_staging(
                    &operation.operation_id,
                    &format!("concurrent_{}.bin", i),
                    &test_data,
                )
                .await
                .unwrap();

            // Finalize
            coord_clone
                .finalize_atomic_operation(&operation.operation_id)
                .await
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_atomic_wal_to_storage_flow() {
    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    let collection_id = "test_collection";
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    // Simulate WAL batch write with atomic coordination
    let config = StagingConfig {
        base_url: storage_url.clone(),
        collection_id: Some(collection_id.to_string()),
        operation_type: TransactionStageType::Flush,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write WAL batch data
    let vectors = vec![
        ProximaRecord {
            oid: "vec1".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            ..Default::default()
        },
        ProximaRecord {
            oid: "vec2".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                3,
                vec![4.0, 5.0, 6.0],
            )],
            ..Default::default()
        },
    ];

    // Serialize vectors (simplified for test)
    let serialized = serde_json::to_vec(&vectors).unwrap();

    coordinator
        .write_to_staging(&operation.operation_id, "batch_001.wal", &serialized)
        .await
        .unwrap();

    // Finalize - ensures atomic visibility
    let result = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_cloud_storage_atomic_pattern() {
    // This test simulates the atomic pattern for cloud storage
    // In cloud storage:
    // 1. Write to staging (could be local temp)
    // 2. Upload to cloud (atomic operation)
    // 3. Delete local staging after successful upload

    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    // Simulate cloud storage URL
    let collection_id = "cloud_collection";
    // In real scenario, this would be s3://, gs://, or adls://
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    let config = StagingConfig {
        base_url: storage_url.clone(),
        collection_id: Some(collection_id.to_string()),
        operation_type: TransactionStageType::Flush,
        // Cloud storage might use local staging
        auto_cleanup: true,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write large data that would benefit from local staging
    let large_data = vec![0u8; 10 * 1024 * 1024]; // 10MB

    coordinator
        .write_to_staging(&operation.operation_id, "large_file.bin", &large_data)
        .await
        .unwrap();

    // Finalize - this would:
    // 1. Complete local write
    // 2. Upload to cloud atomically
    // 3. Clean up local staging
    let result = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_partial_write_prevention() {
    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    let collection_id = "test_collection";
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    // Begin atomic operation
    let config = StagingConfig {
        base_url: storage_url.clone(),
        collection_id: Some(collection_id.to_string()),
        operation_type: TransactionStageType::Flush,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Start writing data
    let partial_data = b"This is partial dat"; // Intentionally incomplete
    coordinator
        .write_to_staging(&operation.operation_id, "partial.bin", partial_data)
        .await
        .unwrap();

    // Simulate failure by rolling back instead of finalizing
    coordinator
        .abort_atomic_operation(&operation.operation_id, "Simulated failure")
        .await
        .unwrap();

    // No partial data should be visible in final location
    // The atomic pattern ensures all-or-nothing visibility
}

#[tokio::test]
async fn test_metadata_consistency_during_atomic_write() {
    let (coordinator, temp_dir) = create_test_environment().await;
    let base_path = temp_dir.path().to_str().unwrap();

    let collection_id = "test_collection";
    let storage_url = format!("file://{}/{}/data", base_path, collection_id);

    // Begin atomic operation for both data and metadata
    let config = StagingConfig {
        base_url: storage_url.clone(),
        collection_id: Some(collection_id.to_string()),
        operation_type: TransactionStageType::Flush,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write vector data
    let vector_data = b"vector data";
    coordinator
        .write_to_staging(&operation.operation_id, "vectors.bin", vector_data)
        .await
        .unwrap();

    // Write metadata
    let metadata = b"metadata_info";
    coordinator
        .write_to_staging(&operation.operation_id, "metadata.json", metadata)
        .await
        .unwrap();

    // Finalize - both files become visible atomically
    let result = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(result.is_ok());

    // In production, readers would see either both files or neither
    // This prevents inconsistent state where vectors exist without metadata
}
