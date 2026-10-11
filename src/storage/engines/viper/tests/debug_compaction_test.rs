//! Debug test for VIPER compaction flow
//! Tests each step: insert -> flush -> read parquet -> compact -> read compacted

use anyhow::Result;
use std::sync::Arc;
use tempfile::TempDir;
use tracing::{debug, info};

use crate::storage::engines::viper::{ViperEngine, ViperEngineConfig};
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::traits::{EngineFilesystemAccess, FlushParameters, UnifiedStorageFormat};
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTree, ProximaTreeNode};
use proximadb_storage_common::storage_path::StoragePath;

/// Helper to read and debug parquet file contents
async fn debug_parquet_file(file_path: &str, label: &str) -> Result<()> {
    use arrow_array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    debug!("\n🔍 DEBUG: {} - Reading {}", label, file_path);

    let fs = FilesystemFactory::create(Default::default()).await?;
    let filesystem = fs.get_filesystem(file_path)?;

    match filesystem.read(file_path).await {
        Ok(data) => {
            debug!("  ✅ File exists, size: {} bytes", data.len());

            // Parse with arrow reader
            match ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(data)) {
                Ok(builder) => {
                    debug!("  📋 Schema: {:?}", builder.schema());

                    let reader = builder.build()?;
                    let mut total_rows = 0;
                    let mut batch_count = 0;

                    for batch_result in reader {
                        match batch_result {
                            Ok(batch) => {
                                batch_count += 1;
                                let rows = batch.num_rows();
                                total_rows += rows;

                                debug!("  📊 Batch {}: {} rows", batch_count, rows);

                                // Print column info
                                for (i, field) in batch.schema().fields().iter().enumerate() {
                                    let column = batch.column(i);
                                    debug!(
                                        "    - Column '{}': type={:?}, null_count={}",
                                        field.name(),
                                        field.data_type(),
                                        column.null_count()
                                    );
                                }

                                // Print first few IDs if available
                                if let Some(id_column) = batch.column_by_name("id")
                                    && let Some(id_array) = id_column
                                        .as_any()
                                        .downcast_ref::<arrow_array::StringArray>(
                                    )
                                {
                                    debug!("    📝 First few IDs:");
                                    for i in 0..std::cmp::min(5, id_array.len()) {
                                        if id_array.is_valid(i) {
                                            debug!("      [{}]: {}", i, id_array.value(i));
                                        } else {
                                            debug!("      [{}]: NULL", i);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("  ❌ Error reading batch: {}", e);
                            }
                        }
                    }

                    debug!("  📊 Total: {} batches, {} rows", batch_count, total_rows);

                    if total_rows == 0 {
                        debug!("  ⚠️ WARNING: File contains NO DATA!");
                    }
                }
                Err(e) => {
                    debug!("  ❌ Failed to create parquet reader: {}", e);
                }
            }
        }
        Err(e) => {
            debug!("  ❌ Failed to read file: {}", e);
        }
    }

    Ok(())
}

/// Create test collection with storage assignment
fn create_test_collection(
    collection_id: &str,
    base_path: &str,
) -> crate::proto::proximadb_v1::Collection {
    use crate::proto::proximadb_v1::{Collection, CollectionConfig, StorageAssignment};

    Collection {
        id: collection_id.to_string(),
        config: Some(CollectionConfig {
            name: collection_id.to_string(),
            embedding_config: None,
            dimension: 128,
            distance_metric: Some(0), // Cosine
            storage_engine: Some(1),  // VIPER
            filterable_columns: vec![],
            index_configs: vec![],
            quantization: None,
            storage_config: None,
            primary_index: Some(String::new()),
            auto_index_selection: Some(false),
            description: None,
            tags: vec![],
            owner: None,
            embedding_models: vec![],
            record_schema: None,
            enable_proxima_record: None,
            text_columns: vec![],
            text_storage_configs: vec![],
            enable_dual_use_embeddings: None,
            canonical_embedding_precision: None,
            permitted_principals: vec![],
            index_policy: None,
            pax_vector_quant: None,
        }),
        stats: None,
        created_at: chrono::Utc::now().timestamp(),
        updated_at: chrono::Utc::now().timestamp(),
        storage_assignment: Some(StorageAssignment {
            primary_path: format!("file://{}", base_path),
            backup_paths: vec![],
            engine: 1,
            engine_config: std::collections::HashMap::new(),
            base_location: format!("file://{}", base_path),
            assigned_at: chrono::Utc::now().timestamp(),
            ..Default::default()
        }),
    }
}

/// Create test vector
fn create_test_vector(id: &str, dimension: usize) -> ProximaRecord {
    let mut props = ProximaTree::new();
    props.insert(
        "test_key".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("test_value".to_string())),
    );
    let values: Vec<f32> = (0..dimension)
        .map(|i| (i as f32) / (dimension as f32))
        .collect();
    ProximaRecord {
        oid: id.to_string(),
        embeddings: vec![EmbeddingCell {
            model_id: "default".to_string(),
            modality: "vector".to_string(),
            dim: dimension as u32,
            values: proximadb_records::EmbeddingValues::Fp32(values),
            ..Default::default()
        }],
        props,
        record_version: 1,
        created_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        updated_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        ..Default::default()
    }
}

#[tokio::test]
async fn test_viper_flush_and_compaction_debug() -> Result<()> {
    // Initialize hardware capabilities for testing
    let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

    debug!("\n🚀 Starting VIPER debug test");

    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();

    debug!("📂 Test directory: {}", base_path);

    // Create config
    let mut config = ViperEngineConfig::default();
    config.enable_ml_clustering = false;
    config.flush_size_bytes = Some(512 * 1024); // 512KB

    // Create engine
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);
    let engine = ViperEngine::from_core_config(
        crate::core::config::ViperConfig::default(),
        filesystem_factory,
    )
    .await?;

    let collection_id = "debug_test";

    // Set up storage assignment
    use tokio::fs;
    let data_dir = StoragePath::collection_data_path(base_path, collection_id);
    fs::create_dir_all(&data_dir).await?;
    let temp_dir = StoragePath::data_file_path(base_path, collection_id, "___temp");
    fs::create_dir_all(&temp_dir).await?;

    // Storage assignment is now handled internally by CollectionService
    // when a collection is created. For test purposes, we just ensure
    // the directory structure exists.
    let wal_dir = format!("{}/{}/write_buffer", base_path, collection_id);
    fs::create_dir_all(&wal_dir).await?;

    let data_url = format!(
        "file://{}",
        StoragePath::collection_data_path(base_path, collection_id)
    );
    debug!("📍 Data directory: {}", data_url);

    // Step 1: Create and flush vectors
    debug!("\n📝 Step 1: Creating and flushing vectors");

    let mut vectors = Vec::new();
    for i in 0..10 {
        vectors.push(create_test_vector(&format!("vec_{}", i), 128));
    }

    let flush_params = FlushParameters {
        collection_id: Some(collection_id.to_string()),
        force: true,
        synchronous: true,
        vector_records: vectors.into_iter().map(|v: ProximaRecord| v).collect(),
        batch_ids: vec![],
        hints: std::collections::HashMap::new(),
        timeout_ms: None,
        trigger_compaction: false,
        collection_config: Some(create_test_collection(collection_id, base_path)),
        estimated_size: 1024, // Estimated size for 10 vectors
    };

    let flush_result = engine.do_flush(&flush_params).await?;
    info!(
        "✅ Flush complete: {} files created, {} entries flushed",
        flush_result.files_created.unwrap_or(0),
        flush_result.entries_flushed.unwrap_or(0)
    );

    // Step 2: List and inspect flushed files
    debug!("\n📋 Step 2: Listing flushed files");

    let fs = engine.get_filesystem_factory().get_filesystem(&data_url)?;
    let entries = fs.list(&data_url).await?;

    let mut parquet_files = Vec::new();
    for entry in entries {
        if entry.name.ends_with(".parquet") && !entry.metadata.is_directory {
            debug!("  📄 Found: {}", entry.name);
            parquet_files.push(entry.url.clone());
        } else if entry.metadata.is_directory {
            debug!("  📁 Directory: {}", entry.name);

            // Check inside directories
            if !entry.name.starts_with("__") {
                let sub_entries = fs.list(&entry.url).await?;
                for sub_entry in sub_entries {
                    if sub_entry.name.ends_with(".parquet") {
                        debug!("    📄 Found in {}: {}", entry.name, sub_entry.name);
                        parquet_files.push(sub_entry.url.clone());
                    }
                }
            }
        }
    }

    // Step 3: Debug each parquet file
    debug!("\n📊 Step 3: Inspecting flushed parquet files");
    for (i, file) in parquet_files.iter().enumerate() {
        debug_parquet_file(file, &format!("Flushed file {}", i)).await?;
    }

    // Step 4: Run compaction
    debug!("\n🗜️ Step 4: Running compaction_info");

    let compact_params = crate::storage::traits::CompactionParameters {
        collection_id: Some(collection_id.to_string()),
        force: true,
        synchronous: true,
        hints: std::collections::HashMap::new(),
        timeout_ms: None,
        priority: crate::storage::traits::OperationPriority::Medium,
        collection_config: Some(create_test_collection(collection_id, base_path)),
        estimated_input_size: 10240, // Estimated size for the data being compacted
    };

    let compact_result = engine.do_compact(&compact_params).await?;
    info!(
        "✅ Compaction complete: {} input files, {} output files, {} entries processed",
        compact_result.input_files.unwrap_or(0),
        compact_result.output_files.unwrap_or(0),
        compact_result.entries_processed.unwrap_or(0)
    );

    // Step 5: List files after compaction
    debug!("\n📋 Step 5: Listing files after compaction_info");

    let entries_after = fs.list(&data_url).await?;
    let mut compacted_files = Vec::new();

    for entry in entries_after {
        if entry.name.ends_with(".parquet") && !entry.metadata.is_directory {
            debug!("  📄 Found: {}", entry.name);
            if entry.name.contains("compacted") {
                compacted_files.push(entry.url.clone());
            }
        }
    }

    // Step 6: Debug compacted files
    debug!("\n📊 Step 6: Inspecting compacted parquet files");
    for (i, file) in compacted_files.iter().enumerate() {
        debug_parquet_file(file, &format!("Compacted file {}", i)).await?;
    }

    // Step 7: Try to search
    debug!("\n🔍 Step 7: Testing search on compacted data");

    let storage_url = format!(
        "file://{}",
        StoragePath::collection_data_path(base_path, collection_id)
    );
    let search_results = engine
        .search_vectors(collection_id, &storage_url, &vec![0.5; 128], 10)
        .await?;

    debug!("🔍 Search returned {} results", search_results.len());
    for (i, search_result) in search_results.iter().enumerate() {
        for result in &search_result.results {
            debug!("  [{}] ID: {}, Score: {:?}", i, result.id, result.score);
        }
    }

    Ok(())
}
