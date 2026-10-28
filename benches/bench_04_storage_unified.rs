// Consolidated Storage Engine Benchmarks
// Combines: compression, search, lifecycle, and cross-engine comparisons

mod common;
use common::benchmark_utils::{STANDARD_BATCH_SIZES, print_system_info};

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use proximadb::core::search::{ComparisonOperator, FilterExpression, SearchParams};
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, CompressionAlgorithm, StorageAssignment, StorageConfig,
    VectorRecord,
};
use proximadb::storage::engines::factory::StorageEngineFactory;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::traits::{FlushParameters, StorageQueryContext, StorageQueryMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Configurable base URL for storage benchmarks
/// Can be changed to test different storage backends:
/// - Local: "file:///tmp/proximadb-bench"
/// - S3: "s3://my-bucket/proximadb-bench"
/// - Azure: "azure://mycontainer/proximadb-bench"
/// - GCS: "gs://my-bucket/proximadb-bench"
const BENCHMARK_BASE_URL: &str = "file:///tmp/proximadb-bench";

fn get_base_path() -> String {
    if BENCHMARK_BASE_URL.starts_with("file://") {
        BENCHMARK_BASE_URL.trim_start_matches("file://").to_string()
    } else {
        BENCHMARK_BASE_URL.to_string()
    }
}

/// Initialize hardware features for benchmarking
fn init_hardware() {
    // Hardware detection is automatically done by the system
}

/// Helper to measure directory size using filesystem API
async fn measure_directory_size_async(
    path: &str,
    filesystem_factory: &FilesystemFactory,
) -> anyhow::Result<u64> {
    // Only add file:// prefix if it's not already a URL
    let fs_url = if path.starts_with("s3://")
        || path.starts_with("gs://")
        || path.starts_with("azure://")
        || path.starts_with("wasbs://")
        || path.starts_with("file://")
    {
        path.to_string()
    } else {
        format!("file://{}", path)
    };

    let fs = filesystem_factory.get_filesystem(&fs_url)?;

    // For listing, we need to use the path without the scheme
    let list_path = if let Some(stripped) = path.strip_prefix("file://") {
        stripped
    } else if path.contains("://") {
        // For other schemes, extract path after ://
        path.split("://").nth(1).unwrap_or(path)
    } else {
        path
    };

    let entries = fs.list(list_path).await?;

    let mut total_size = 0u64;
    for entry in entries {
        if entry.metadata.is_directory {
            // For recursive calls, construct the proper path
            let subdir_path = if list_path.ends_with('/') {
                format!("{}{}", list_path, entry.name)
            } else {
                format!("{}/{}", list_path, entry.name)
            };
            total_size += Box::pin(measure_directory_size_async(
                &subdir_path,
                filesystem_factory,
            ))
            .await?;
        } else {
            total_size += entry.metadata.size;
        }
    }

    Ok(total_size)
}

/// Synchronous wrapper for directory size measurement
fn measure_directory_size(path: &str, runtime: &tokio::runtime::Runtime) -> std::io::Result<u64> {
    // Check if path exists first
    if !std::path::Path::new(path).exists() {
        return Ok(0);
    }

    runtime.block_on(async {
        let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        measure_directory_size_async(path, &fs_factory)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    })
}

/// Generate normalized random vector for testing
fn generate_random_vector(dimension: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut vec: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Normalize for cosine similarity
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut vec {
            *val /= norm;
        }
    }
    vec
}

/// Generate vectors with controlled similarity ranges for realistic benchmarking
/// Creates clusters with varying distances from cluster centers
fn generate_test_vectors(count: usize, dimension: usize) -> Vec<VectorRecord> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42); // Deterministic for reproducibility

    // Create 10 cluster centers
    let num_clusters = 10;
    let mut cluster_centers: Vec<Vec<f32>> = Vec::new();

    for _ in 0..num_clusters {
        let mut center = vec![0.0f32; dimension];
        for val in &mut center {
            *val = rng.gen_range(-1.0..1.0);
        }
        // Normalize
        let norm: f32 = center.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut center {
                *val /= norm;
            }
        }
        cluster_centers.push(center);
    }

    (0..count)
        .map(|i| {
            let cluster_id = i % num_clusters;
            let center = &cluster_centers[cluster_id];

            // Create vector at varying distances from cluster center
            // Distance varies based on position within cluster
            let position_in_cluster = (i / num_clusters) as f32;
            let vectors_per_cluster = (count / num_clusters).max(1) as f32; // Avoid division by zero
            let noise_level = ((position_in_cluster / vectors_per_cluster) * 0.5)
                .min(0.5)
                .max(0.0); // 0.0 to 0.5

            let mut vector = center.clone();

            // Add controlled noise (creates similarity range 0.5-1.0)
            if noise_level > 0.0 {
                for val in &mut vector {
                    *val += rng.gen_range(-noise_level..noise_level);
                }
            }

            // Normalize
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut vector {
                    *val /= norm;
                }
            }

            let mut metadata = HashMap::new();

            // Category matches cluster
            metadata.insert(
                "category".to_string(),
                proximadb::proto::proximadb_v1::SqlValue {
                    value: Some(
                        proximadb::proto::proximadb_v1::sql_value::Value::StringValue(format!(
                            "cat_{}",
                            cluster_id
                        )),
                    ),
                },
            );

            // Price varies widely to ensure filter selectivity
            // Use modulo to create price ranges that overlap with filter conditions
            metadata.insert(
                "price".to_string(),
                proximadb::proto::proximadb_v1::SqlValue {
                    value: Some(
                        proximadb::proto::proximadb_v1::sql_value::Value::NumberValue(
                            (i as f64 * 10.0) % 1000.0, // Creates prices 0-990 across all categories
                        ),
                    ),
                },
            );

            // Tags for multi-value filtering
            metadata.insert(
                "tags".to_string(),
                proximadb::proto::proximadb_v1::SqlValue {
                    value: Some(
                        proximadb::proto::proximadb_v1::sql_value::Value::StringValue(format!(
                            "tag_{},tag_{}",
                            cluster_id % 5,
                            i % 3
                        )),
                    ),
                },
            );

            VectorRecord {
                id: format!("vec_{}", i),
                vector,
                metadata,
                ..Default::default()
            }
        })
        .collect()
}

// ============================================================================
// BENCHMARK 1: Unified Compression + Search Test
// ============================================================================
fn bench_compression_with_search(c: &mut Criterion) {
    print_system_info("Storage Engine Unified Benchmarks");
    init_hardware();

    let dimension = 768;
    let count = 128;

    eprintln!("\n📊 UNIFIED COMPRESSION + SEARCH BENCHMARK");
    eprintln!("   Organized by compression level → Compare all engines per compression");
    eprintln!("   Dimension: {}, Vectors: {}", dimension, count);

    let vectors = generate_test_vectors(count, dimension);
    // Use the first vector as query for deterministic results
    let query_vector = vectors
        .first()
        .map(|v| v.vector.clone())
        .unwrap_or_else(|| generate_random_vector(dimension));

    // Debug: Show we're searching for the first vector
    eprintln!("\n   🔍 Query vector: Using vec_0 (first stored vector) for deterministic results");
    eprintln!(
        "   📊 First 5 values of query vector: {:?}",
        &query_vector[..5.min(query_vector.len())]
    );
    if let Some(first_vec) = vectors.first() {
        eprintln!("   📝 Full first vector record:");
        eprintln!("      ID: {}", first_vec.id);
        eprintln!("      Vector dimensions: {}", first_vec.vector.len());
        eprintln!(
            "      First 5 values: {:?}",
            &first_vec.vector[..5.min(first_vec.vector.len())]
        );
        eprintln!("      Metadata: {:?}", first_vec.metadata);
    }

    let uncompressed_size = count * dimension * std::mem::size_of::<f32>();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    // Compression algorithms to test (OUTER LOOP)
    let compressions = vec![("none", 0), ("zstd", 1), ("lz4", 2), ("snappy", 3)];

    // REORGANIZED: Iterate by compression first, then engines
    for (compress_name, compress_value) in &compressions {
        eprintln!("\n═══════════════════════════════════════════════════════════════════════════");
        eprintln!(
            "🔧 COMPRESSION: {} - Testing all engines",
            compress_name.to_uppercase()
        );
        eprintln!("═══════════════════════════════════════════════════════════════════════════");
        eprintln!(
            "\n{:<8} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Engine", "Compress", "Size(MB)", "Ratio%", "Flush(ms)", "Pure(ms)", "Filter(ms)"
        );
        eprintln!("{}", "-".repeat(80));

        // Test each engine with this compression (INNER LOOP)
        // Create fresh engines for each compression to avoid state contamination
        #[allow(unused_mut)]
        let mut engine_names = vec!["sst", "viper", "nova", "helix"];
        #[cfg(feature = "experimental-engines")]
        {
            engine_names.push("swift");
            engine_names.push("raptor");
        }

        for engine_name in engine_names {
            // Create fresh engine for each test to avoid state contamination between compressions
            let engine: std::sync::Arc<dyn proximadb::storage::traits::UnifiedStorageEngine> =
                match engine_name {
                    "sst" => StorageEngineFactory::create_sst().unwrap(),
                    "viper" => StorageEngineFactory::create_viper().unwrap(),
                    "nova" => StorageEngineFactory::create_nova().unwrap(),
                    "helix" => StorageEngineFactory::create_helix().unwrap(),
                    #[cfg(feature = "experimental-engines")]
                    "swift" => StorageEngineFactory::create_swift().unwrap(),
                    #[cfg(feature = "experimental-engines")]
                    "raptor" => StorageEngineFactory::create_raptor().unwrap(),
                    _ => continue,
                };
            // Collection ID format: {engine}-{compression} (use hyphen for URL compatibility)
            let collection_id = format!("{}-{}", engine_name, compress_name);
            // Base path: engines will append collection_id to this
            let base_path = get_base_path();

            // Clean previous run using filesystem API
            let collection_path = format!("{}/{}", base_path, collection_id);
            runtime.block_on(async {
                let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
                    .await
                    .ok()?;
                let fs = fs_factory
                    .get_filesystem(&format!("file://{}", &collection_path))
                    .ok()?;
                let _ = fs.remove_dir_all(&collection_path).await;
                Some(())
            });

            // Step 1: Flush data with compression
            eprintln!(
                "  Flushing {} vectors (dim={}) with {} compression to {}...",
                count, dimension, compress_name, collection_path
            );
            eprintln!("    📁 Data directory: {}", collection_path);

            // Calculate expected size
            let expected_bytes = count * dimension * std::mem::size_of::<f32>();
            eprintln!(
                "    Expected raw size: {} bytes ({:.2} MB)",
                expected_bytes,
                expected_bytes as f64 / 1_048_576.0
            );

            let flush_start = std::time::Instant::now();
            let flush_result = runtime.block_on(async {
                let mut storage_config = StorageConfig::default();
                storage_config.compression = Some(*compress_value);

                // Define filterable columns for metadata schema
                let filterable_columns = vec![
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "category".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableString
                                as i32,
                        indexed: true,
                        supports_range: false,
                        estimated_cardinality: Some(10), // We have 10 categories
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "price".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableFloat
                                as i32,
                        indexed: true,
                        supports_range: true,
                        estimated_cardinality: None,
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "tags".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableString
                                as i32,
                        indexed: true,
                        supports_range: false,
                        estimated_cardinality: Some(15), // Combinations of tags
                    },
                ];

                let collection = Collection {
                    storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
                    id: collection_id.clone(),
                    config: Some(CollectionConfig {
                        name: collection_id.clone(),
                        dimension: dimension as u32,
                        storage_config: Some(storage_config),
                        filterable_columns,
                        ..Default::default()
                    }),
                    storage_assignment: Some(StorageAssignment {
                        primary_path: base_path.clone(),
                        base_location: base_path.clone(),
                        ..Default::default()
                    }),
                    ..Default::default()
                };

                let params = FlushParameters {
                    collection_id: Some(collection_id.clone()),
                    vector_records: vectors
                        .iter()
                        .cloned()
                        .map(proximadb::proto::defaults::vector_record_to_proxima_record)
                        .collect(),
                    force: true,
                    synchronous: true,
                    collection_config: Some(collection),
                    ..Default::default()
                };

                engine.flush(params).await
            });
            let flush_time_ms = flush_start.elapsed().as_millis();

            // Validate flush result
            let flush_result = match flush_result {
                Ok(result) => {
                    // Validate that data was actually written
                    if result.entries_flushed.unwrap_or(0) == 0 {
                        eprintln!(
                            "    ⚠️  WARNING: No vectors written for {} with {}",
                            engine_name, compress_name
                        );
                    }
                    if result.bytes_written.unwrap_or(0) == 0 {
                        eprintln!(
                            "    ⚠️  WARNING: No bytes written for {} with {}",
                            engine_name, compress_name
                        );
                    }
                    let vectors_written = result.entries_flushed.unwrap_or(0);
                    let bytes_written = result.bytes_written.unwrap_or(0);
                    eprintln!(
                        "    ✓ Flushed {} vectors, {} bytes written in {}ms",
                        vectors_written, bytes_written, flush_time_ms
                    );

                    // Validate size is reasonable
                    if bytes_written < (expected_bytes as u64 / 100) {
                        eprintln!(
                            "    ⚠️  SUSPICIOUSLY SMALL: {} bytes is < 1% of expected {} bytes",
                            bytes_written, expected_bytes
                        );
                    }
                    Ok(result)
                }
                Err(e) => Err(e),
            };

            // Check if flush failed or wrote no data
            let flush_success = match &flush_result {
                Err(e) => {
                    eprintln!(
                        "    ⚠️  Flush failed for {} with {}: {:?}",
                        engine_name, compress_name, e
                    );
                    false
                }
                Ok(result) if result.entries_flushed.unwrap_or(0) == 0 => {
                    eprintln!(
                        "    ⚠️  Flush succeeded but wrote 0 vectors for {} with {}, skipping search",
                        engine_name, compress_name
                    );
                    false
                }
                Ok(_) => true,
            };

            if !flush_success {
                // Print skipped status in results table
                eprintln!(
                    "{:<8} {:<8} {:>10} {:>8} {:>7} {:>10} {:>10} {:>10}  ⛔ SKIPPED",
                    engine_name, compress_name, "FAILED", "N/A", "N/A", flush_time_ms, "N/A", "N/A"
                );
                eprintln!();
                // Clean up any partial data
                runtime.block_on(async {
                    let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
                        .await
                        .ok()?;
                    let fs = fs_factory
                        .get_filesystem(&format!("file://{}", &collection_path))
                        .ok()?;
                    let _ = fs.remove_dir_all(&collection_path).await;
                    Some(())
                });
                continue;
            }

            // Verify files were actually created and list them
            let (files_created, file_details) = runtime
                .block_on(async {
                    let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
                        .await
                        .ok()?;
                    let fs = fs_factory
                        .get_filesystem(&format!("file://{}", &collection_path))
                        .ok()?;
                    let entries = fs.list(&collection_path).await.ok()?;

                    let mut details = Vec::new();
                    for entry in &entries {
                        let size = if entry.metadata.is_directory {
                            "DIR".to_string()
                        } else {
                            format!("{} bytes", entry.metadata.size)
                        };
                        details.push(format!("      {} ({})", entry.name, size));
                    }

                    Some((entries.len(), details))
                })
                .unwrap_or((0, Vec::new()));

            if files_created == 0 {
                eprintln!(
                    "    ⚠️  WARNING: No files created after flush for {} with {}",
                    engine_name, compress_name
                );
            } else {
                eprintln!("    ✓ Created {} files/directories:", files_created);
                for detail in &file_details {
                    eprintln!("{}", detail);
                }
            }

            // Measure storage metrics using filesystem API
            let size_bytes =
                measure_directory_size(&collection_path, &runtime).unwrap_or_else(|e| {
                    eprintln!("    ⚠️  Failed to measure directory size: {:?}", e);
                    0
                });
            let size_mb = size_bytes as f64 / 1_048_576.0;
            // Compression ratio: 1 - (compressed/uncompressed)
            // Standard definition: higher is better, negative means expansion
            let compression_ratio = 1.0 - (size_bytes as f64 / uncompressed_size as f64);
            let compression_ratio_percent = compression_ratio * 100.0;

            // Step 2: Pure vector search benchmark
            let mut pure_group =
                c.benchmark_group(format!("pure_{}-{}", engine_name, compress_name));
            pure_group.measurement_time(Duration::from_secs(3));
            pure_group.sample_size(15);
            pure_group.warm_up_time(Duration::from_secs(1));

            let mut pure_time_ms = 0u128;
            let query_clone = query_vector.clone();
            let compress_val = *compress_value; // Capture for closure

            // Track if we've logged results for this engine+compression (only log once)
            let mut pure_results_logged = false;

            pure_group.bench_function("search", |b| {
                b.iter(|| {
                    let start = std::time::Instant::now();
                    let result: Result<Vec<proximadb::core::search::results::OptimizedSearchRecord>, _> = runtime.block_on(async {
                        let search_params = Arc::new(SearchParams {
                            vector: Some(query_clone.clone()),
                            top_k: Some(10),
                            filters: None,
                            filter_expression: None,
                            ..Default::default()
                        });

                        // Search needs same collection config as flush for proper data loading
                        let mut storage_config = StorageConfig::default();
                        storage_config.compression = Some(compress_val);

                        // Same filterable columns as flush
                        let filterable_columns = vec![
                            proximadb::proto::proximadb_v1::FilterableColumnSpec {
                                name: "category".to_string(),
                                data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableString as i32,
                                indexed: true,
                                supports_range: false,
                                estimated_cardinality: Some(10),
                            },
                            proximadb::proto::proximadb_v1::FilterableColumnSpec {
                                name: "price".to_string(),
                                data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableFloat as i32,
                                indexed: true,
                                supports_range: true,
                                estimated_cardinality: None,
                            },
                            proximadb::proto::proximadb_v1::FilterableColumnSpec {
                                name: "tags".to_string(),
                                data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableString as i32,
                                indexed: true,
                                supports_range: false,
                                estimated_cardinality: Some(15),
                            },
                        ];

                        let collection = Arc::new(Collection {
                            storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
                            id: collection_id.clone(),
                            config: Some(CollectionConfig {
                                name: collection_id.clone(),
                                dimension: dimension as u32,
                                storage_config: Some(storage_config),
                                filterable_columns,
                                ..Default::default()
                            }),
                            storage_assignment: Some(StorageAssignment {
                                primary_path: base_path.clone(),
                                base_location: base_path.clone(),
                                ..Default::default()
                            }),
                            ..Default::default()
                        });

                        // Configure metadata with proper collection_id for SST path resolution
                        let mut metadata = StorageQueryMetadata::default();
                        metadata.collection_id = collection_id.clone();
                        metadata.dimension = dimension;
                        metadata.storage_path = base_path.clone();

                        let ctx = StorageQueryContext {
                            search_params,
                            collection,
                            metadata,
                            user_context: None,
                            tenant_context: None,
                        };

                        engine.search_vectors_unified(&ctx).await
                    });
                    pure_time_ms = start.elapsed().as_millis();

                    // Validate and log search results ONCE per engine+compression
                    if !pure_results_logged {
                        if let Ok(ref results) = result {
                            if results.is_empty() {
                                eprintln!("    ⚠️  Pure search: NO RESULTS for {} (expected vec_0)", engine_name);
                                eprintln!("       Query: {:?}", &query_clone[..5.min(query_clone.len())]);
                            } else {
                                eprintln!("    ✅ Pure search: Found {} results for {} in {}ms", results.len(), engine_name, pure_time_ms);
                                // Print first 3 results with scores
                                for (i, r) in results.iter().take(3).enumerate() {
                                    eprintln!("       [{}] ID={}, score={:.4}, sim={:.4}",
                                        i+1, r.id, r.score, r.similarity.unwrap_or(0.0));
                                }
                            }
                        } else if let Err(ref e) = result {
                            eprintln!("    ❌ Pure search FAILED for {}: {:?}", engine_name, e);
                        }
                        pure_results_logged = true;
                    }

                    black_box(result)
                })
            });
            pure_group.finish();

            // Step 3: Metadata-filtered search benchmark
            let mut filtered_group =
                c.benchmark_group(format!("filter_{}-{}", engine_name, compress_name));
            filtered_group.measurement_time(Duration::from_secs(3));
            filtered_group.sample_size(15);
            filtered_group.warm_up_time(Duration::from_secs(1));

            let mut filter_time_ms = 0u128;
            let query_clone = query_vector.clone();
            let compress_val = *compress_value; // Capture for closure

            // Track if we've logged results for this engine+compression (only log once)
            let mut filter_results_logged = false;

            filtered_group.bench_function("search", |b| {
                b.iter(|| {
                    let start = std::time::Instant::now();
                    let result: Result<Vec<proximadb::core::search::results::OptimizedSearchRecord>, _> = runtime.block_on(async {
                        // Filter: category="cat_5" AND price < 500
                        let filter_expr = FilterExpression::And(vec![
                            FilterExpression::Comparison {
                                field: "category".to_string(),
                                operator: ComparisonOperator::Equals,
                                value: serde_json::Value::String("cat_5".to_string()),
                            },
                            FilterExpression::Comparison {
                                field: "price".to_string(),
                                operator: ComparisonOperator::LessThan,
                                value: serde_json::Value::Number(serde_json::Number::from(500)),
                            },
                        ]);

                        let search_params = Arc::new(SearchParams {
                            vector: Some(query_clone.clone()),
                            top_k: Some(10),
                            filter_expression: Some(filter_expr),
                            filters: None,
                            ..Default::default()
                        });

                        // Search needs same collection config as flush for proper data loading
                        let mut storage_config = StorageConfig::default();
                        storage_config.compression = Some(compress_val);

                        // Same filterable columns as flush
                        let filterable_columns = vec![
                            proximadb::proto::proximadb_v1::FilterableColumnSpec {
                                name: "category".to_string(),
                                data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableString as i32,
                                indexed: true,
                                supports_range: false,
                                estimated_cardinality: Some(10),
                            },
                            proximadb::proto::proximadb_v1::FilterableColumnSpec {
                                name: "price".to_string(),
                                data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableFloat as i32,
                                indexed: true,
                                supports_range: true,
                                estimated_cardinality: None,
                            },
                            proximadb::proto::proximadb_v1::FilterableColumnSpec {
                                name: "tags".to_string(),
                                data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableString as i32,
                                indexed: true,
                                supports_range: false,
                                estimated_cardinality: Some(15),
                            },
                        ];

                        let collection = Arc::new(Collection {
                            storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
                            id: collection_id.clone(),
                            config: Some(CollectionConfig {
                                name: collection_id.clone(),
                                dimension: dimension as u32,
                                storage_config: Some(storage_config),
                                filterable_columns,
                                ..Default::default()
                            }),
                            storage_assignment: Some(StorageAssignment {
                                primary_path: base_path.clone(),
                                base_location: base_path.clone(),
                                ..Default::default()
                            }),
                            ..Default::default()
                        });

                        // Configure metadata with proper collection_id for SST path resolution
                        let mut metadata = StorageQueryMetadata::default();
                        metadata.collection_id = collection_id.clone();
                        metadata.dimension = dimension;
                        metadata.storage_path = base_path.clone();

                        let ctx = StorageQueryContext {
                            search_params,
                            collection,
                            metadata,
                            user_context: None,
                            tenant_context: None,
                        };

                        engine.search_vectors_unified(&ctx).await
                    });
                    filter_time_ms = start.elapsed().as_millis();

                    // Validate and log filtered search results ONCE per engine+compression
                    if !filter_results_logged {
                        if let Ok(ref results) = result {
                            if results.is_empty() {
                                eprintln!("    ⚠️  Filtered search: NO RESULTS for {} (expected vectors with category=cat_5, price<500)", engine_name);
                            } else {
                                eprintln!("    ✅ Filtered search: Found {} results for {} in {}ms", results.len(), engine_name, filter_time_ms);
                                // Print first 3 results with scores
                                for (i, r) in results.iter().take(3).enumerate() {
                                    eprintln!("       [{}] ID={}, score={:.4}, sim={:.4}",
                                        i+1, r.id, r.score, r.similarity.unwrap_or(0.0));
                                }
                            }
                        } else if let Err(ref e) = result {
                            eprintln!("    ❌ Filtered search FAILED for {}: {:?}", engine_name, e);
                        }
                        filter_results_logged = true;
                    }

                    black_box(result)
                })
            });
            filtered_group.finish();

            // Print results with validation status
            if size_bytes == 0 && files_created == 0 {
                eprintln!(
                    "{:<8} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10}  ❌ NO DATA",
                    engine_name,
                    compress_name,
                    "NO DATA",
                    "N/A",
                    flush_time_ms,
                    pure_time_ms,
                    filter_time_ms
                );
            } else {
                eprintln!(
                    "{:<8} {:<8} {:>10.2} {:>9.1}% {:>10} {:>10} {:>10}",
                    engine_name,
                    compress_name,
                    size_mb,
                    compression_ratio_percent,
                    flush_time_ms,
                    pure_time_ms,
                    filter_time_ms
                );
            }

            // Clean up immediately after each test using filesystem API
            runtime.block_on(async {
                let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
                    .await
                    .ok()?;
                let fs = fs_factory
                    .get_filesystem(&format!("file://{}", base_path))
                    .ok()?;
                let _ = fs.remove_dir_all(&base_path).await;
                Some(())
            });
        }
        eprintln!(); // Blank line between engines
    }

    // Final cleanup using filesystem API
    eprintln!("\n✅ Benchmark complete, all test data cleaned up");
    runtime.block_on(async {
        let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
            .await
            .ok()?;
        // Clean up all engine_compression directories
        let base_path = get_base_path();
        let fs = fs_factory
            .get_filesystem(&format!("file://{}", base_path))
            .ok()?;
        // Remove all test directories
        for engine in ["sst", "viper", "nova", "swift", "raptor", "helix"] {
            for compression in ["none", "zstd", "lz4", "snappy"] {
                let test_path = format!("{}/{}-{}", base_path, engine, compression);
                let _ = fs.remove_dir_all(&test_path).await;
            }
        }
        Some(())
    });
}

// ============================================================================
// BENCHMARK 2: Engine Creation and Lifecycle
// ============================================================================
fn bench_engine_lifecycle(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("engine_lifecycle");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(15);
    group.warm_up_time(Duration::from_secs(1));

    let _runtime = tokio::runtime::Runtime::new().unwrap();

    // Benchmark engine creation
    group.bench_function("sst_create", |b| {
        b.iter(|| {
            let engine = StorageEngineFactory::create_sst();
            black_box(engine)
        })
    });

    group.bench_function("viper_create", |b| {
        b.iter(|| {
            let engine = StorageEngineFactory::create_viper();
            black_box(engine)
        })
    });

    group.bench_function("nova_create", |b| {
        b.iter(|| {
            let engine = StorageEngineFactory::create_nova();
            black_box(engine)
        })
    });

    #[cfg(feature = "experimental-engines")]
    group.bench_function("swift_create", |b| {
        b.iter(|| {
            let engine = StorageEngineFactory::create_swift();
            black_box(engine)
        })
    });

    #[cfg(feature = "experimental-engines")]
    group.bench_function("raptor_create", |b| {
        b.iter(|| {
            let engine = StorageEngineFactory::create_raptor();
            black_box(engine)
        })
    });

    group.bench_function("helix_create", |b| {
        b.iter(|| {
            let engine = StorageEngineFactory::create_helix();
            black_box(engine)
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 3: Large-Scale Search Performance
// ============================================================================
fn bench_large_scale_search(c: &mut Criterion) {
    init_hardware();

    eprintln!("\n🔍 LARGE-SCALE SEARCH BENCHMARK");
    eprintln!("   Testing search performance at different scales with compression");
    eprintln!("   Collection ID Format: {{engine}}-{{compression}}-{{batchsize}}");
    eprintln!("   Engines: sst, viper, nova, swift, raptor, helix");
    eprintln!("   Compressions: none, zstd, lz4, snappy, gzip");
    eprintln!("   Batch Sizes: 128, 512, 1024");
    eprintln!("   Total Combinations: 105 (7 engines × 5 compressions × 3 batch sizes)");

    let dimension = 768;
    let runtime = tokio::runtime::Runtime::new().unwrap();

    // Test different scales (reduced ~10x for OOM prevention)
    // Use standard batch sizes for scales
    let scales = vec![
        ("128", STANDARD_BATCH_SIZES[0]),  // Small batch
        ("512", STANDARD_BATCH_SIZES[1]),  // Medium batch
        ("1024", STANDARD_BATCH_SIZES[2]), // Large batch
    ];

    // Test with different compression methods
    let compressions = vec![
        ("none", CompressionAlgorithm::CompressionNone),
        ("zstd", CompressionAlgorithm::CompressionZstd),
        ("lz4", CompressionAlgorithm::CompressionLz4),
        ("snappy", CompressionAlgorithm::CompressionSnappy),
        ("gzip", CompressionAlgorithm::CompressionGzip),
    ];

    let mut test_num = 0;
    let total_tests = 105; // 7 engines × 5 compressions × 3 batch sizes

    for (batch_size_name, count) in scales {
        eprintln!("\n   === Batch Size: {} vectors ===", count);

        let vectors = generate_test_vectors(count, dimension);
        let query = generate_random_vector(dimension);

        // Test all storage engines - stateless engines get collection info from parameters
        #[allow(unused_mut)]
        let mut engines: Vec<(
            &str,
            std::sync::Arc<dyn proximadb::storage::traits::UnifiedStorageEngine>,
        )> = vec![
            ("sst", StorageEngineFactory::create_sst().unwrap()),
            ("viper", StorageEngineFactory::create_viper().unwrap()),
            ("nova", StorageEngineFactory::create_nova().unwrap()),
            ("helix", StorageEngineFactory::create_helix().unwrap()),
        ];
        #[cfg(feature = "experimental-engines")]
        {
            engines.push(("swift", StorageEngineFactory::create_swift().unwrap()));
            engines.push(("raptor", StorageEngineFactory::create_raptor().unwrap()));
        }

        for (engine_name, engine) in engines {
            for (compress_name, compression) in &compressions {
                test_num += 1;
                eprintln!(
                    "   Test {}/{}: {}-{}-{}",
                    test_num, total_tests, engine_name, compress_name, batch_size_name
                );

                // Use {engine}-{compression}-{batchsize} format for collection ID
                let collection_id =
                    format!("{}-{}-{}", engine_name, compress_name, batch_size_name);
                let base_path = get_base_path(); // Don't include collection_id in base_path

                // Clean and load data using filesystem API
                let collection_path = format!("{}/{}", base_path, collection_id);
                runtime.block_on(async {
                    let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
                        .await
                        .ok()?;
                    let fs = fs_factory
                        .get_filesystem(&format!("file://{}", collection_path))
                        .ok()?;
                    let _ = fs.remove_dir_all(&collection_path).await;
                    Some(())
                });

                // Capture compression value to avoid borrow issues
                let compress_val = *compression;

                runtime.block_on(async {
                    let collection = Collection {
                        storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
                        id: collection_id.clone(),
                        config: Some(CollectionConfig {
                            name: collection_id.clone(),
                            dimension: dimension as u32,
                            storage_config: Some(StorageConfig {
                                compression: Some(compress_val as i32),
                                ..Default::default()
                            }),
                            ..Default::default()
                        }),
                        storage_assignment: Some(StorageAssignment {
                            primary_path: base_path.clone(),
                            base_location: base_path.clone(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    };

                    let params = FlushParameters {
                        collection_id: Some(collection_id.clone()),
                        vector_records: vectors
                            .iter()
                            .cloned()
                            .map(proximadb::proto::defaults::vector_record_to_proxima_record)
                            .collect(),
                        force: true,
                        synchronous: true,
                        collection_config: Some(collection),
                        ..Default::default()
                    };

                    let _ = engine.flush(params).await;
                });

                // Benchmark search
                let mut group = c.benchmark_group(format!(
                    "search_{}-{}-{}",
                    engine_name, compress_name, batch_size_name
                ));
                group.measurement_time(Duration::from_secs(5));
                group.sample_size(15);
                group.warm_up_time(Duration::from_secs(1));

                let query_clone = query.clone();
                group.bench_function("top10", |b| {
                    b.iter(|| {
                        runtime.block_on(async {
                            let search_params = Arc::new(SearchParams {
                                vector: Some(query_clone.clone()),
                                top_k: Some(10),
                                filters: None,
                                filter_expression: None,
                                ..Default::default()
                            });

                            let collection = Arc::new(Collection {
                                storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
                                id: collection_id.clone(),
                                config: Some(CollectionConfig {
                                    name: collection_id.clone(),
                                    dimension: dimension as u32,
                                    storage_config: Some(StorageConfig {
                                        compression: Some(compress_val as i32),
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                }),
                                storage_assignment: Some(StorageAssignment {
                                    primary_path: base_path.clone(),
                                    base_location: base_path.clone(),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            });

                            // Configure metadata with proper collection_id for SST path resolution
                            let mut metadata = StorageQueryMetadata::default();
                            metadata.collection_id = collection_id.clone(); // Specific to benchmark run
                            metadata.dimension = dimension;
                            metadata.storage_path = base_path.clone(); // Consistent base path

                            let ctx = StorageQueryContext {
                                search_params,
                                collection,
                                metadata,
                                user_context: None,
                                tenant_context: None,
                            };

                            let result = engine.search_vectors_unified(&ctx).await;
                            black_box(result)
                        })
                    })
                });

                group.finish();

                // Clean up using filesystem API
                let collection_path = format!("{}/{}", base_path, collection_id);
                runtime.block_on(async {
                    let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
                        .await
                        .ok()?;
                    let fs = fs_factory
                        .get_filesystem(&format!("file://{}", collection_path))
                        .ok()?;
                    let _ = fs.remove_dir_all(&collection_path).await;
                    Some(())
                });
            }
        }
    }

    // Final cleanup using filesystem API
    runtime.block_on(async {
        let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
            .await
            .ok()?;
        let base_path = get_base_path();
        let fs = fs_factory
            .get_filesystem(&format!("file://{}", base_path))
            .ok()?;

        // Clean up all {engine}-{compression}-{batchsize} directories
        for engine in ["sst", "viper", "nova", "swift", "raptor", "helix"] {
            for compression in ["none", "zstd", "lz4", "snappy", "gzip"] {
                for batch_size in ["128", "512", "1024"] {
                    let test_path =
                        format!("{}/{}-{}-{}", base_path, engine, compression, batch_size);
                    let _ = fs.remove_dir_all(&test_path).await;
                }
            }
        }
        Some(())
    });
}

// ============================================================================
// BENCHMARK 4: Insertion Performance
// ============================================================================
fn bench_insertion_performance(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("insertion_performance");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(15);
    group.warm_up_time(Duration::from_secs(1));

    let dimension = 768;
    let count = 128;
    let vectors = generate_test_vectors(count, dimension);
    let runtime = tokio::runtime::Runtime::new().unwrap();

    // Test each engine - stateless engines get collection info from parameters
    #[allow(unused_mut)]
    let mut engines: Vec<(
        &str,
        std::sync::Arc<dyn proximadb::storage::traits::UnifiedStorageEngine>,
    )> = vec![
        ("sst", StorageEngineFactory::create_sst().unwrap()),
        ("viper", StorageEngineFactory::create_viper().unwrap()),
        ("nova", StorageEngineFactory::create_nova().unwrap()),
    ];
    #[cfg(feature = "experimental-engines")]
    {
        engines.push(("swift", StorageEngineFactory::create_swift().unwrap()));
    }

    for (engine_name, engine) in engines {
        group.bench_function(engine_name, |b| {
            b.iter(|| {
                runtime.block_on(async {
                    let collection_id = format!("insert-{}", engine_name);
                    let base_path = format!("{}/insert/{}", get_base_path(), engine_name);

                    // Clean before each iteration using filesystem API
                    let fs_factory =
                        match FilesystemFactory::create(FilesystemConfig::default()).await {
                            Ok(f) => f,
                            Err(_) => {
                                return Err(anyhow::anyhow!("Failed to create filesystem factory"));
                            }
                        };
                    let fs = fs_factory.get_filesystem(&format!("file://{}", base_path))?;
                    let _ = fs.remove_dir_all(&base_path).await;

                    let collection = Collection {
                        storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
                        id: collection_id.clone(),
                        config: Some(CollectionConfig {
                            name: collection_id.clone(),
                            dimension: dimension as u32,
                            ..Default::default()
                        }),
                        storage_assignment: Some(StorageAssignment {
                            primary_path: base_path.clone(),
                            base_location: base_path.clone(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    };

                    let params = FlushParameters {
                        collection_id: Some(collection_id),
                        vector_records: vectors
                            .iter()
                            .cloned()
                            .map(proximadb::proto::defaults::vector_record_to_proxima_record)
                            .collect(),
                        force: true,
                        synchronous: true,
                        collection_config: Some(collection),
                        ..Default::default()
                    };

                    let result = engine.flush(params).await;

                    // Clean after using filesystem API
                    let _ = fs.remove_dir_all(&base_path).await;

                    black_box(result)
                })
            })
        });
    }

    group.finish();

    // Final cleanup using filesystem API
    runtime.block_on(async {
        let fs_factory = FilesystemFactory::create(FilesystemConfig::default())
            .await
            .ok()?;
        let cleanup_path = format!("{}/insert", get_base_path());
        let fs = fs_factory
            .get_filesystem(&format!("file://{}", cleanup_path))
            .ok()?;
        let _ = fs.remove_dir_all(&cleanup_path).await;
        Some(())
    });
}

// Configure and run all benchmarks with custom settings
// Will stop when either condition is met:
// - 30 samples collected OR
// - 5 seconds of measurement time elapsed
criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(15)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_compression_with_search,
              bench_engine_lifecycle,
              bench_large_scale_search,
              bench_insertion_performance
}

criterion_main!(benches);
