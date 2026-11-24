//! Comprehensive tests for TransactionCoordinator
//!
//! These tests ensure the TransactionCoordinator correctly handles:
//! - Atomic operations across storage systems
//! - Transaction management
//! - Staging operations
//! - Abort/rollback scenarios
//! - Multi-operation coordination

use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::transaction_coordinator::{
    StagingConfig, TransactionCoordinator, TransactionStageType, TransactionState,
    TransactionalOperationStatus,
};
use std::sync::Arc;

#[tokio::test]
async fn test_atomic_coordinator_creation() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());

    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .expect("Failed to create coordinator");

    // Should be created successfully
    let active_ops = coordinator.list_active_operations().await;
    assert_eq!(
        active_ops.len(),
        0,
        "Should start with no active operations"
    );
}

#[tokio::test]
async fn test_begin_atomic_operation() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        operation_type: TransactionStageType::Flush,
        collection_id: Some("test_collection".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: None,
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator
        .begin_atomic_operation(&config)
        .await
        .expect("Failed to begin atomic operation");

    assert!(!operation.operation_id.is_empty());
    match operation.status {
        TransactionalOperationStatus::Preparing | TransactionalOperationStatus::Staging => {}
        _ => panic!("Operation should be Preparing or Staging"),
    }
}

#[tokio::test]
async fn test_write_to_staging() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    // Begin operation
    let config = StagingConfig {
        operation_type: TransactionStageType::Flush,
        collection_id: Some("staging_test".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: None,
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write data to staging
    let test_data = b"test data for atomic operation";
    coordinator
        .write_to_staging(&operation.operation_id, "test.parquet", test_data)
        .await
        .expect("Failed to write to staging");

    // Verify operation status updated
    let status = coordinator
        .get_operation_status(&operation.operation_id)
        .await
        .expect("Operation status not found");

    // Status should have progressed beyond Preparing
    match status {
        TransactionalOperationStatus::Staging | TransactionalOperationStatus::Finalizing => {}
        _ => panic!("Operation should be in Staging or Finalizing state"),
    }
}

#[tokio::test]
async fn test_finalize_atomic_operation() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    // Begin operation
    let config = StagingConfig {
        operation_type: TransactionStageType::Flush,
        collection_id: Some("finalize_test".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: Some("final".to_string()),
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write test data
    let test_data = b"finalized data";
    coordinator
        .write_to_staging(&operation.operation_id, "final.parquet", test_data)
        .await
        .unwrap();

    // Finalize operation
    coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await
        .expect("Failed to finalize operation");

    // After finalization, operation should be removed from active operations
    let status = coordinator
        .get_operation_status(&operation.operation_id)
        .await;
    assert!(
        status.is_none(),
        "Operation should be removed after finalization"
    );

    // Verify the file was moved to final location
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let fs = filesystem_factory
        .get_filesystem(&operation.final_url)
        .unwrap();
    let final_file_path = format!("{}/final.parquet", operation.final_url);
    assert!(
        fs.exists(&final_file_path).await.unwrap(),
        "File should exist in final location"
    );
}

#[tokio::test]
async fn test_abort_atomic_operation() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    // Begin operation
    let config = StagingConfig {
        operation_type: TransactionStageType::Flush,
        collection_id: Some("abort_test".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: None,
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write test data
    coordinator
        .write_to_staging(&operation.operation_id, "abort.parquet", b"abort data")
        .await
        .unwrap();

    // Abort operation
    coordinator
        .abort_atomic_operation(&operation.operation_id, "Test abort")
        .await
        .expect("Failed to abort operation");

    // Check status - operation might be removed after abort
    let status = coordinator
        .get_operation_status(&operation.operation_id)
        .await;

    // After abort, the operation might be cleaned up (None) or marked as Failed
    match status {
        Some(TransactionalOperationStatus::Failed(reason)) => {
            assert_eq!(reason, "Test abort");
        }
        None => {
            // Operation was cleaned up after abort - this is also valid
        }
        Some(status) => panic!(
            "Operation should be Failed or cleaned up after abort, got: {:?}",
            status
        ),
    }
}

#[tokio::test]
async fn test_transaction_lifecycle() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    // Begin transaction
    let tx_handle = coordinator
        .begin_transaction("test_transaction", vec![])
        .await
        .expect("Failed to begin transaction");

    // Prepare phase
    let can_commit = tx_handle
        .prepare()
        .await
        .expect("Failed to prepare transaction");
    assert!(can_commit, "Transaction should be able to commit");

    // Commit transaction
    tx_handle
        .commit()
        .await
        .expect("Failed to commit transaction");

    // Check state
    let state = coordinator
        .get_transaction_state("test_transaction")
        .await
        .expect("Failed to get transaction state");

    match state {
        TransactionState::Committed => {}
        _ => panic!("Transaction should be Committed"),
    }
}

#[tokio::test]
async fn test_transaction_rollback() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory.clone(),
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    // Create a test file first
    let test_file_path = format!(
        "{}/test_rollback_file.txt",
        temp_dir.path().to_str().unwrap()
    );
    let fs = filesystem_factory
        .get_filesystem(&format!("file://{}", temp_dir.path().to_str().unwrap()))
        .unwrap();
    fs.write(&test_file_path, b"test content", None)
        .await
        .unwrap();

    // Begin transaction with participant
    let tx_handle = coordinator
        .begin_transaction("rollback_test", vec!["test_participant".to_string()])
        .await
        .expect("Failed to begin transaction");

    // Register rollback action
    tx_handle
        .register_rollback(
            "test_participant",
            crate::storage::transaction_coordinator::RollbackAction::DeleteFile {
                path: format!("file://{}", test_file_path),
            },
        )
        .await
        .expect("Failed to register rollback");

    // Rollback transaction
    tx_handle
        .rollback()
        .await
        .expect("Failed to rollback transaction");

    // After rollback, the transaction might be removed or in Aborted state
    match coordinator.get_transaction_state("rollback_test").await {
        Ok(TransactionState::Aborted) | Ok(TransactionState::Aborting) => {
            // Transaction is in expected state
        }
        Err(_) => {
            // Transaction was removed after rollback - this is also valid
        }
        Ok(state) => panic!(
            "Transaction should be Aborted or removed after rollback, got: {:?}",
            state
        ),
    }

    // Verify the file was deleted as part of rollback
    assert!(
        !fs.exists(&test_file_path).await.unwrap(),
        "File should be deleted after rollback"
    );
}

#[tokio::test]
async fn test_concurrent_operations() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = Arc::new(
        TransactionCoordinator::new(
            filesystem_factory,
            Some(temp_dir.path().to_str().unwrap().to_string()),
        )
        .await
        .unwrap(),
    );

    // Start multiple concurrent operations
    let mut handles = vec![];

    for i in 0..5 {
        let coord_clone = coordinator.clone();
        let base_path = temp_dir.path().to_str().unwrap().to_string();

        let handle = tokio::spawn(async move {
            let config = StagingConfig {
                operation_type: TransactionStageType::Flush,
                collection_id: Some(format!("concurrent_{}", i)),
                base_url: base_path,
                custom_staging_dir: None,
                auto_cleanup: true,
                max_orphaned_age_hours: 24,
                ..Default::default()
            };

            let operation = coord_clone
                .begin_atomic_operation(&config)
                .await
                .expect("Failed to begin operation");

            // Write data
            let data = format!("data for operation {}", i);
            coord_clone
                .write_to_staging(&operation.operation_id, "data.txt", data.as_bytes())
                .await
                .expect("Failed to write");

            // Finalize
            coord_clone
                .finalize_atomic_operation(&operation.operation_id)
                .await
                .expect("Failed to finalize");

            operation.operation_id
        });

        handles.push(handle);
    }

    // Wait for all operations
    let mut operation_ids = vec![];
    for handle in handles {
        let op_id = handle.await.expect("Task failed");
        operation_ids.push(op_id);
    }

    // All operations should have unique IDs
    operation_ids.sort();
    operation_ids.dedup();
    assert_eq!(
        operation_ids.len(),
        5,
        "All operations should have unique IDs"
    );
}

#[tokio::test]
async fn test_cleanup_orphaned_operations() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    // Create some operations but don't finalize them
    for i in 0..3 {
        let config = StagingConfig {
            operation_type: TransactionStageType::Flush,
            collection_id: Some(format!("orphaned_{}", i)),
            base_url: temp_dir.path().to_str().unwrap().to_string(),
            custom_staging_dir: None,
            auto_cleanup: false, // Disable auto cleanup for testing
            max_orphaned_age_hours: 0,
            ..Default::default()
        };

        let operation = coordinator.begin_atomic_operation(&config).await.unwrap();
        coordinator
            .write_to_staging(&operation.operation_id, "orphan.data", b"orphaned")
            .await
            .unwrap();
    }

    // The operations are still active, so cleanup won't remove them
    // This is by design - cleanup only removes truly orphaned operations
    // (those not in the active list, usually from crashes)

    // For this test, let's verify that active operations exist
    let active = coordinator.list_active_operations().await;
    assert_eq!(active.len(), 3, "Should have 3 active operations");

    // Now abort one operation to test cleanup
    let first_op = active[0].operation_id.clone();
    coordinator
        .abort_atomic_operation(&first_op, "test cleanup")
        .await
        .unwrap();

    // After abort, we should have 2 active operations
    let active = coordinator.list_active_operations().await;
    assert_eq!(
        active.len(),
        2,
        "Should have 2 active operations after abort"
    );
}

#[tokio::test]
async fn test_flush_stage_atomic_operations() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let coordinator = Arc::new(coordinator);

    // Generic flush-staging coverage (previously reached through the VIPER
    // wrapper, removed with the engine per ADR-093).
    let config = StagingConfig {
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        collection_id: Some("flush_collection".to_string()),
        operation_type: TransactionStageType::Flush,
        auto_cleanup: true,
        skip_uuid_subdir: true,
        ..Default::default()
    };
    let operation = coordinator
        .begin_atomic_operation(&config)
        .await
        .expect("Failed to begin flush");

    coordinator
        .write_to_staging(
            &operation.operation_id,
            "vectors/data.parquet",
            b"mock parquet data",
        )
        .await
        .expect("Failed to write parquet");

    coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await
        .expect("Failed to finalize flush");
}

#[tokio::test]
async fn test_staging_operation_type_variants() {
    // Test all TransactionStageType variants and their staging_dir_name() method
    let flush_type = TransactionStageType::Flush;
    assert_eq!(flush_type.staging_dir_name(), "__flush");

    let compaction_type = TransactionStageType::Compaction;
    assert_eq!(compaction_type.staging_dir_name(), "__compact");

    let metadata_type = TransactionStageType::Metadata;
    assert_eq!(metadata_type.staging_dir_name(), "__metadata_info");

    let wal_type = TransactionStageType::Wal;
    assert_eq!(wal_type.staging_dir_name(), "__wal");

    let transaction_type = TransactionStageType::Transaction;
    assert_eq!(transaction_type.staging_dir_name(), "__transaction");

    let custom_type = TransactionStageType::Custom("custom_staging".to_string());
    assert_eq!(custom_type.staging_dir_name(), "custom_staging");
}

#[tokio::test]
async fn test_staging_config_default() {
    let config = StagingConfig::default();
    assert_eq!(config.base_url, "file://./data");
    assert!(config.collection_id.is_none());
    assert_eq!(config.operation_type, TransactionStageType::Flush);
    assert!(config.custom_staging_dir.is_none());
    assert!(config.auto_cleanup);
    assert_eq!(config.max_orphaned_age_hours, 24);
}

#[tokio::test]
async fn test_staging_config_with_custom_staging_dir() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        operation_type: TransactionStageType::Custom("my_custom_op".to_string()),
        collection_id: Some("test_collection".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: Some("__custom_dir".to_string()),
        auto_cleanup: false,
        max_orphaned_age_hours: 48,
        ..Default::default()
    };

    let operation = coordinator
        .begin_atomic_operation(&config)
        .await
        .expect("Failed to begin atomic operation with custom staging dir");

    assert!(!operation.operation_id.is_empty());
}

#[tokio::test]
async fn test_operation_without_collection_id() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        operation_type: TransactionStageType::Metadata,
        collection_id: None, // No collection ID
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: None,
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator
        .begin_atomic_operation(&config)
        .await
        .expect("Failed to begin atomic operation without collection ID");

    assert!(!operation.operation_id.is_empty());
}

#[tokio::test]
async fn test_invalid_operation_id_handling() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let fake_operation_id = "non_existent_operation_id".to_string();

    // Try to write to staging with invalid operation ID
    let write_result = coordinator
        .write_to_staging(&fake_operation_id, "test_file.txt", b"test data")
        .await;

    assert!(
        write_result.is_err(),
        "Should fail with invalid operation ID"
    );

    // Try to finalize with invalid operation ID
    let finalize_result = coordinator
        .finalize_atomic_operation(&fake_operation_id)
        .await;
    assert!(
        finalize_result.is_err(),
        "Should fail with invalid operation ID"
    );

    // Try to abort with invalid operation ID (should succeed silently - idempotent operation)
    let abort_result = coordinator
        .abort_atomic_operation(&fake_operation_id, "Invalid ID test")
        .await;
    assert!(
        abort_result.is_ok(),
        "Abort should be idempotent and not fail for non-existent operations"
    );
}

#[tokio::test]
async fn test_double_finalize_operation() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        operation_type: TransactionStageType::Flush,
        collection_id: Some("test_collection".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: None,
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write some data
    coordinator
        .write_to_staging(&operation.operation_id, "test_file.txt", b"test data")
        .await
        .unwrap();

    // First finalize should succeed
    let first_finalize = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(first_finalize.is_ok(), "First finalize should succeed");

    // Second finalize should fail
    let second_finalize = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(second_finalize.is_err(), "Second finalize should fail");
}

#[tokio::test]
async fn test_write_after_finalize() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        operation_type: TransactionStageType::Flush,
        collection_id: Some("test_collection".to_string()),
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        custom_staging_dir: None,
        auto_cleanup: true,
        max_orphaned_age_hours: 24,
        ..Default::default()
    };

    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write some data
    coordinator
        .write_to_staging(&operation.operation_id, "test_file.txt", b"test data")
        .await
        .unwrap();

    // Finalize the operation
    coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await
        .unwrap();

    // Try to write after finalize - should fail
    let write_after_finalize = coordinator
        .write_to_staging(&operation.operation_id, "another_file.txt", b"more data")
        .await;

    assert!(
        write_after_finalize.is_err(),
        "Write after finalize should fail"
    );
}

#[tokio::test]
async fn test_operation_status_transitions() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        ..StagingConfig::default()
    };
    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Check initial status
    let initial_status = coordinator
        .get_operation_status(&operation.operation_id)
        .await
        .unwrap();
    assert!(matches!(
        initial_status,
        TransactionalOperationStatus::Preparing | TransactionalOperationStatus::Staging
    ));

    // Write data (should keep it in staging)
    coordinator
        .write_to_staging(&operation.operation_id, "test_file.txt", b"test data")
        .await
        .unwrap();

    let staging_status = coordinator
        .get_operation_status(&operation.operation_id)
        .await
        .unwrap();
    assert!(matches!(
        staging_status,
        TransactionalOperationStatus::Staging
    ));

    // Finalize (should mark as completed)
    coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await
        .unwrap();

    let final_status = coordinator
        .get_operation_status(&operation.operation_id)
        .await;
    // After finalization, the operation might be cleaned up, so it might not exist
    // This tests both successful finalization and cleanup behavior
    assert!(
        final_status.is_none()
            || matches!(
                final_status.unwrap(),
                TransactionalOperationStatus::Completed
            )
    );
}

#[tokio::test]
async fn test_empty_data_write() {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let config = StagingConfig {
        base_url: temp_dir.path().to_str().unwrap().to_string(),
        ..StagingConfig::default()
    };
    let operation = coordinator.begin_atomic_operation(&config).await.unwrap();

    // Write empty data
    let write_result = coordinator
        .write_to_staging(&operation.operation_id, "empty_file.txt", b"")
        .await;
    assert!(write_result.is_ok(), "Should be able to write empty data");

    // Finalize with empty data
    let finalize_result = coordinator
        .finalize_atomic_operation(&operation.operation_id)
        .await;
    assert!(
        finalize_result.is_ok(),
        "Should be able to finalize with empty data"
    );
}

#[tokio::test]
#[ignore = "Temporarily disabled - causes segfault in tarpaulin"]
async fn test_wal_atomic_operations() {
    // Use timestamp-based directory names to avoid URL parsing issues
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_dir = tempfile::Builder::new()
        .prefix(&format!("atomic_test_{}_", timestamp))
        .tempdir_in("/tmp")
        .unwrap();
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await.unwrap());
    let coordinator = TransactionCoordinator::new(
        filesystem_factory,
        Some(temp_dir.path().to_str().unwrap().to_string()),
    )
    .await
    .unwrap();

    let wal_ops = crate::storage::transaction_coordinator::WalTransactionalOperations::new(
        Arc::new(coordinator),
    );

    // Test WAL segment rotation
    let operation = wal_ops
        .begin_segment_rotation(temp_dir.path().to_str().unwrap())
        .await
        .expect("Failed to begin WAL rotation");

    // Write segment data
    let segment_data = b"WAL segment data";
    wal_ops
        .write_segment_to_staging(&operation.operation_id, "segment_001", segment_data)
        .await
        .expect("Failed to write segment");

    // Finalize rotation
    wal_ops
        .finalize_rotation(&operation.operation_id)
        .await
        .expect("Failed to finalize WAL rotation");
}
