// Example to verify SST engine file writing
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, SqlValue, StorageAssignment, StorageConfig, VectorRecord,
};
use proximadb::storage::engines::factory::StorageEngineFactory;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::traits::FlushParameters;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    println!("🚀 Testing SST Engine File Writing");

    // Create temp directory
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap().to_string();
    println!("📁 Using directory: {}", base_path);

    // Create filesystem
    let filesystem = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

    // Create SST engine using factory
    let engine = StorageEngineFactory::create_sst_async().await?;

    println!("✅ Created SST engine");

    // Create test vectors
    let mut vectors = Vec::new();
    for i in 0..10 {
        let mut metadata = HashMap::new();
        metadata.insert(
            "index".to_string(),
            SqlValue {
                value: Some(
                    proximadb::proto::proximadb_v1::sql_value::Value::NumberValue(i as f64),
                ),
            },
        );

        vectors.push(VectorRecord {
            id: format!("vec_{}", i),
            vector: vec![i as f32; 128],
            metadata,
            timestamp: Some(i as i64),
            updated_at: None,
            expires_at: None,
            version: None,
            source: None,
        });
    }

    println!("📊 Created {} test vectors", vectors.len());

    // Create collection config
    let collection = Collection {
        storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
        id: "test_collection".to_string(),
        config: Some(CollectionConfig {
            name: "test_collection".to_string(),
            dimension: 128,
            storage_config: Some(StorageConfig::default()),
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            primary_path: base_path.clone(),
            base_location: base_path.clone(),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Flush vectors
    println!("💾 Flushing vectors...");
    let flush_params = FlushParameters {
        collection_id: Some("test_collection".to_string()),
        vector_records: vectors
            .into_iter()
            .map(proximadb::proto::defaults::vector_record_to_proxima_record)
            .collect(),
        force: true,
        synchronous: true,
        collection_config: Some(collection.clone()),
        ..Default::default()
    };

    let result = engine.do_flush(&flush_params).await?;

    println!("✅ Flush result:");
    println!("   - Success: {}", result.success);
    println!("   - Entries flushed: {:?}", result.entries_flushed);
    println!("   - Bytes written: {:?}", result.bytes_written);
    println!("   - Files created: {:?}", result.files_created);

    // Check what files were created
    let data_path = format!("{}/test_collection/data", base_path);
    println!("\n📂 Checking directory: {}", data_path);

    let fs = filesystem.get_filesystem(&format!("file://{}", data_path))?;
    let files = fs.list(&format!("file://{}", data_path)).await?;

    println!("📁 Files in directory:");
    if files.is_empty() {
        println!("   ❌ No files found!");
    } else {
        for file in &files {
            println!(
                "   - {} ({} bytes, dir={})",
                file.name, file.metadata.size, file.metadata.is_directory
            );
        }
    }

    // Check for SST files specifically
    let sst_files: Vec<_> = files.iter().filter(|f| f.name.ends_with(".sst")).collect();

    if sst_files.is_empty() {
        println!("\n❌ No .sst files created!");

        // Check staging directory
        let staging_path = format!("{}/test_collection/data/__staging", base_path);
        println!("\n🔍 Checking staging directory: {}", staging_path);
        if let Ok(staging_files) = fs.list(&format!("file://{}", staging_path)).await {
            if !staging_files.is_empty() {
                println!("⚠️  Found {} files in staging:", staging_files.len());
                for file in &staging_files {
                    println!("   - {} ({} bytes)", file.name, file.metadata.size);
                }
            } else {
                println!("   No files in staging either");
            }
        }
    } else {
        println!("\n✅ Found {} SST files:", sst_files.len());
        for file in sst_files {
            println!("   - {}", file.name);
        }
    }

    // Now try to search to verify data can be read
    println!("\n🔍 Testing search to verify data can be read...");

    use proximadb::core::search::SearchParams;
    use proximadb::storage::traits::StorageQueryContext;
    use proximadb::storage::traits::StorageQueryMetadata;

    let mut search_params = SearchParams::default();
    search_params.vector = Some(vec![0.0; 128]);
    search_params.top_k = Some(5);
    let search_params = Arc::new(search_params);

    let mut metadata = StorageQueryMetadata::default();
    metadata.collection_id = "test_collection".to_string();
    metadata.dimension = 128;
    metadata.storage_path = base_path.clone();

    let ctx = StorageQueryContext {
        search_params,
        collection: Arc::new(collection),
        metadata,
        user_context: None,
        tenant_context: None,
    };

    match engine.search_vectors_unified(&ctx).await {
        Ok(results) => {
            if results.is_empty() {
                println!("   ⚠️  Search returned no results");
            } else {
                println!("   ✅ Search returned {} results", results.len());
                for (i, result) in results.iter().take(3).enumerate() {
                    println!(
                        "      #{}: {} (score: {:.4})",
                        i + 1,
                        result.id,
                        result.score
                    );
                }
            }
        }
        Err(e) => {
            println!("   ❌ Search failed: {}", e);
        }
    }

    // Keep temp directory info for manual inspection
    let path = temp_dir.path().to_path_buf();
    println!("\n📁 Temporary directory: {}", path.display());
    println!(
        "   Run this to inspect: ls -la {}/test_collection/data/",
        path.display()
    );

    // Keep the temp directory alive by not dropping it immediately
    std::mem::forget(temp_dir);

    Ok(())
}
