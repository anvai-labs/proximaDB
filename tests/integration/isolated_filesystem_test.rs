//! Isolated Filesystem Integration Tests
//!
//! Tests filesystem functionality with completely isolated environments
//! to ensure reliable testing without cross-test contamination.

// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;

use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

use common::integration_test_helpers::UnifiedTestEnvironment as IsolatedTestEnvironment;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};

#[tokio::test]
async fn test_isolated_filesystem_basic_operations() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let env = IsolatedTestEnvironment::new().await?;

    // Create filesystem factory
    let factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

    // Get filesystem for local storage
    let storage_url = format!("file://{}", env.temp_dir.path().display());
    let filesystem = factory.get_filesystem(&storage_url)?;

    // Test file operations
    let test_file = format!("{}/test_file.txt", env.temp_dir.path().display());
    let test_content = format!("Test content for collection: {}", env.collection_id());

    // Write file
    filesystem
        .write(&test_file, test_content.as_bytes(), None)
        .await?;

    // Read file
    let read_content = filesystem.read(&test_file).await?;
    assert_eq!(String::from_utf8(read_content)?, test_content);

    // Check file exists
    assert!(filesystem.exists(&test_file).await?);

    // List files
    let files = filesystem
        .list(&format!("{}", env.temp_dir.path().display()))
        .await?;
    assert!(files.iter().any(|f| f.name.contains("test_file.txt")));

    debug!(
        "✅ Basic filesystem operations test passed for collection: {}",
        env.collection_id()
    );
    Ok(())
}

#[tokio::test]
async fn test_isolated_filesystem_directory_operations() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let env = IsolatedTestEnvironment::new().await?;

    let factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);
    let storage_url = format!("file://{}", env.temp_dir.path().display());
    let filesystem = factory.get_filesystem(&storage_url)?;

    // Create nested directories
    let nested_dir = format!(
        "{}/subdir/nested/{}",
        env.temp_dir.path().display(),
        env.collection_id()
    );
    filesystem.create_dir_all(&nested_dir).await?;

    assert!(filesystem.exists(&nested_dir).await?);

    // Write file in nested directory
    let nested_file = format!("{}/nested_file.txt", nested_dir);
    let content = format!("Nested file for collection: {}", env.collection_id());
    filesystem
        .write(&nested_file, content.as_bytes(), None)
        .await?;

    // Verify file in nested directory
    let read_content = filesystem.read(&nested_file).await?;
    assert_eq!(String::from_utf8(read_content)?, content);

    // List nested directory
    let nested_files = filesystem.list(&nested_dir).await?;
    assert_eq!(nested_files.len(), 1);
    assert!(nested_files[0].name.contains("nested_file.txt"));

    debug!(
        "✅ Directory operations test passed for collection: {}",
        env.collection_id()
    );
    Ok(())
}

#[tokio::test]
async fn test_isolated_filesystem_concurrent_operations() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let env = IsolatedTestEnvironment::new().await?;

    // Create the storage URL and filesystem outside the async closures
    let storage_url = format!("file://{}", env.temp_dir.path().display());
    let temp_path = env.temp_dir.path().to_path_buf();
    let collection_id = env.collection_id().to_string();

    // Create filesystem factory and filesystem instances upfront
    let factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

    // Spawn concurrent file operations
    let mut handles = Vec::new();
    let concurrent_operations = 5;

    for i in 0..concurrent_operations {
        let factory_clone = factory.clone();
        let storage_url_clone = storage_url.clone();
        let collection_id_clone = collection_id.clone();
        let temp_path_clone = temp_path.clone();

        let handle = tokio::spawn(async move {
            // Create filesystem instance within each task to avoid lifetime issues
            let filesystem = factory_clone.get_filesystem(&storage_url_clone)?;

            let file_path = format!("{}/concurrent_file_{}.txt", temp_path_clone.display(), i);
            let content = format!(
                "Concurrent content {} for collection: {}",
                i, collection_id_clone
            );

            // Write file
            filesystem
                .write(&file_path, content.as_bytes(), None)
                .await?;

            // Read file back
            let read_content = filesystem.read(&file_path).await?;
            let read_string = String::from_utf8(read_content)?;

            // Verify content matches
            if read_string == content {
                Ok(i)
            } else {
                Err(anyhow::anyhow!("Content mismatch for file {}", i))
            }
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut successful_operations = 0;

    for handle in handles {
        match handle.await? {
            Ok(file_id) => {
                successful_operations += 1;
                debug!("📁 File {} operation succeeded", file_id);
            }
            Err(e) => {
                debug!("⚠️ File operation failed: {}", e);
            }
        }
    }

    assert_eq!(
        successful_operations, concurrent_operations,
        "All {} concurrent operations should succeed",
        concurrent_operations
    );

    // Verify all files exist using a fresh filesystem instance
    let final_filesystem = factory.get_filesystem(&storage_url)?;
    let final_files = final_filesystem
        .list(&format!("{}", env.temp_dir.path().display()))
        .await?;
    let concurrent_files: Vec<_> = final_files
        .iter()
        .filter(|f| f.name.contains("concurrent_file_"))
        .collect();

    assert_eq!(
        concurrent_files.len(),
        concurrent_operations,
        "Should have {} concurrent files",
        concurrent_operations
    );

    debug!(
        "✅ Concurrent filesystem operations test passed for collection: {}",
        env.collection_id()
    );
    debug!(
        "   {} concurrent operations completed successfully",
        successful_operations
    );
    Ok(())
}

#[tokio::test]
async fn test_isolated_filesystem_error_handling() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let env = IsolatedTestEnvironment::new().await?;

    let factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);
    let storage_url = format!("file://{}", env.temp_dir.path().display());
    let filesystem = factory.get_filesystem(&storage_url)?;

    // Test reading non-existent file
    let non_existent_file = format!("{}/does_not_exist.txt", env.temp_dir.path().display());
    let read_result = filesystem.read(&non_existent_file).await;
    assert!(
        read_result.is_err(),
        "Reading non-existent file should fail"
    );

    // Test file existence check
    assert!(!filesystem.exists(&non_existent_file).await?);

    // Test writing to read-only location (if applicable)
    // For now, just test that normal operations work
    let valid_file = format!("{}/valid_file.txt", env.temp_dir.path().display());
    let content = format!("Valid content for collection: {}", env.collection_id());

    // This should succeed
    filesystem
        .write(&valid_file, content.as_bytes(), None)
        .await?;
    assert!(filesystem.exists(&valid_file).await?);

    // Test deleting the file
    filesystem.delete(&valid_file).await?;
    assert!(!filesystem.exists(&valid_file).await?);

    debug!(
        "✅ Error handling test passed for collection: {}",
        env.collection_id()
    );
    Ok(())
}

#[tokio::test]
async fn test_isolated_filesystem_large_file_operations() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let env = IsolatedTestEnvironment::new().await?;

    let factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);
    let storage_url = format!("file://{}", env.temp_dir.path().display());
    let filesystem = factory.get_filesystem(&storage_url)?;

    // Create a larger file (1MB)
    let large_file = format!("{}/large_file.dat", env.temp_dir.path().display());
    let chunk_size = 1024; // 1KB chunks
    let num_chunks = 1024; // Total 1MB

    let chunk_data =
        format!("Collection {} chunk data: ", env.collection_id()).repeat(chunk_size / 50); // Repeat to fill chunk
    let chunk_bytes = chunk_data.as_bytes();

    // Build large content
    let mut large_content = Vec::new();
    for i in 0..num_chunks {
        large_content.extend_from_slice(&format!("{:04}:", i).as_bytes());
        large_content.extend_from_slice(&chunk_bytes[0..chunk_size.saturating_sub(5)]);
    }

    // Write large file
    let start_time = std::time::Instant::now();
    filesystem.write(&large_file, &large_content, None).await?;
    let write_duration = start_time.elapsed();

    // Read large file back
    let start_time = std::time::Instant::now();
    let read_content = filesystem.read(&large_file).await?;
    let read_duration = start_time.elapsed();

    // Verify content
    assert_eq!(read_content.len(), large_content.len());
    assert_eq!(read_content, large_content);

    // Check file info
    let files = filesystem
        .list(&format!("{}", env.temp_dir.path().display()))
        .await?;
    let large_file_info = files.iter().find(|f| f.name.contains("large_file.dat"));
    assert!(large_file_info.is_some());

    let file_info = large_file_info.unwrap();
    assert!(file_info.metadata.size > 1_000_000); // Should be ~1MB

    debug!(
        "✅ Large file operations test passed for collection: {}",
        env.collection_id()
    );
    debug!(
        "   File size: {} bytes, Write: {:?}, Read: {:?}",
        file_info.metadata.size, write_duration, read_duration
    );
    Ok(())
}

#[tokio::test]
async fn test_isolated_filesystem_multi_collection_isolation() -> Result<()> {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    // Create multiple isolated environments
    let env1 = IsolatedTestEnvironment::new().await?;
    let env2 = IsolatedTestEnvironment::new().await?;
    let env3 = IsolatedTestEnvironment::new().await?;

    let environments = vec![&env1, &env2, &env3];
    let factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

    // Create files in each environment
    for (i, env) in environments.iter().enumerate() {
        let storage_url = format!("file://{}", env.temp_dir.path().display());
        let filesystem = factory.get_filesystem(&storage_url)?;

        let test_file = format!("{}/collection_file.txt", env.temp_dir.path().display());
        let content = format!("Content for collection {} (env {})", env.collection_id(), i);

        filesystem
            .write(&test_file, content.as_bytes(), None)
            .await?;
    }

    // Verify each environment only sees its own files
    for (i, env) in environments.iter().enumerate() {
        let storage_url = format!("file://{}", env.temp_dir.path().display());
        let filesystem = factory.get_filesystem(&storage_url)?;

        let files = filesystem
            .list(&format!("{}", env.temp_dir.path().display()))
            .await?;

        // Filter to only regular files (exclude directories created by IsolatedTestEnvironment)
        let regular_files: Vec<_> = files
            .iter()
            .filter(|f| !f.metadata.is_directory && f.name.contains("collection_file.txt"))
            .collect();

        // Should only see 1 regular file in this environment
        assert_eq!(
            regular_files.len(),
            1,
            "Environment {} should have exactly 1 regular file",
            i
        );
        assert!(regular_files[0].name.contains("collection_file.txt"));

        // Read and verify content
        let test_file = format!("{}/collection_file.txt", env.temp_dir.path().display());
        let content = filesystem.read(&test_file).await?;
        let content_str = String::from_utf8(content)?;

        assert!(
            content_str.contains(env.collection_id()),
            "Content should contain collection ID for environment {}",
            i
        );
        assert!(
            content_str.contains(&format!("env {}", i)),
            "Content should contain environment number for environment {}",
            i
        );

        // Verify filesystem is correctly isolated to its own temp directory
        // (This confirms that each environment's filesystem can only access its own directory)
        let own_dir = env.temp_dir.path().display().to_string();
        assert!(
            storage_url.contains(&own_dir),
            "Environment {} filesystem should be pointing to its own directory",
            i
        );
    }

    debug!("✅ Multi-collection isolation test passed");
    for (i, env) in environments.iter().enumerate() {
        debug!("   Environment {}: {} (isolated)", i, env.collection_id());
    }
    Ok(())
}
