//! Performance benchmark integration tests

// use super::common::*; // Removed - common module deleted
use anyhow::Result;
use tracing::{debug, error, info, warn};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tonic::transport::Channel;
use proximadb::proto::proximadb_v1::proxima_db_client::ProximaDbClient;
use proximadb::proto::proximadb_v1::{
    CollectionRequest, CollectionOperation, VectorBatchRequest, VectorOperation,
    VectorSearchRequest, SearchQuery, SearchParameters, MetadataFilter, MetadataValue,
    metadata_value, FilterableColumnSpec, FilterableDataType, IncludeFields,
};
use tonic::Request;

#[cfg(test)]
mod performance_tests {
    use super::*;
    // use crate::measure_performance; // Removed - from common module

    #[tokio::test]
    async fn test_vector_insertion_performance() -> Result<()> {
        // init_test_env(); // Removed - from common module

        let collection_id = format!("test_collection_{}", uuid::Uuid::new_v4());
        let test_cases = vec![
            ("small_batch", 10, 384),
            ("medium_batch", 100, 384),
            ("large_batch", 1000, 384),
            ("high_dimension", 100, 1536),
        ];

        for (test_name, batch_size, dimension) in test_cases {
            let (batch, measurement) =
                measure_performance!(&format!("Vector creation - {}", test_name), batch_size, {
                    create_test_vector_batch(collection_id.clone(), batch_size, dimension)
                });

            assert_eq!(batch.len(), batch_size);

            // Performance expectations - adjust based on actual operation (just creating vectors)
            // Vector creation should be very fast
            match test_name {
                "small_batch" => assert!(measurement.throughput > 10000.0, "Small batch throughput: {}", measurement.throughput),
                "medium_batch" => assert!(measurement.throughput > 5000.0, "Medium batch throughput: {}", measurement.throughput),
                "large_batch" => assert!(measurement.throughput > 1000.0, "Large batch throughput: {}", measurement.throughput),
                "high_dimension" => assert!(measurement.throughput > 500.0, "High dimension throughput: {}", measurement.throughput),
                _ => {}
            }
        }

        debug!("✅ Vector insertion performance test completed");
        Ok(())
    }

    #[tokio::test]
    async fn test_distance_calculation_performance() -> Result<()> {
        // init_test_env(); // Removed - from common module

        let dimensions = vec![384, 768, 1536];
        let num_calculations = 1000;

        for dim in dimensions {
            let vec1 = vec![0.5; dim];
            let vec2 = vec![0.7; dim];

            let (similarities, measurement) = measure_performance!(
                &format!("Cosine similarity - {}D", dim),
                num_calculations,
                {
                    let mut results = Vec::new();
                    for _ in 0..num_calculations {
                        let calculator = proximadb::compute::distance_computation::create_distance_calculator(
                            proximadb::compute::distance_computation::DistanceMetric::Cosine,
                        );
                        let sim = calculator.distance(&vec1, &vec2);
                        results.push(sim);
                    }
                    results
                }
            );

            assert_eq!(similarities.len(), num_calculations);

            // All distances should be valid (0 to 2 for cosine distance)
            for distance in similarities {
                assert!(distance >= 0.0 && distance <= 2.0);
            }

            // Performance expectations (calculations per second)
            match dim {
                384 => assert!(measurement.throughput > 10000.0),
                768 => assert!(measurement.throughput > 5000.0),
                1536 => assert!(measurement.throughput > 2000.0),
                _ => {}
            }
        }

        debug!("✅ Distance calculation performance test completed");
        Ok(())
    }

    #[tokio::test]
    async fn test_metadata_processing_performance() -> Result<()> {
        // init_test_env(); // Removed - from common module

        let record_counts = vec![100, 500, 1000, 5000];
        let filterable_columns = vec!["category", "author", "doc_type", "year"];

        for count in record_counts {
            // Generate test records with extensive metadata
            let records: Vec<HashMap<String, serde_json::Value>> = (0..count)
                .map(|i| {
                    let mut meta = HashMap::new();
                    // Filterable fields
                    meta.insert(
                        "category".to_string(),
                        serde_json::Value::String(format!("category_{}", i % 10)),
                    );
                    meta.insert(
                        "author".to_string(),
                        serde_json::Value::String(format!("author_{}", i % 50)),
                    );
                    meta.insert(
                        "doc_type".to_string(),
                        serde_json::Value::String("research_paper".to_string()),
                    );
                    meta.insert(
                        "year".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(2020 + (i % 5))),
                    );

                    // Extra metadata fields
                    meta.insert(
                        "title".to_string(),
                        serde_json::Value::String(format!("Document Title {}", i)),
                    );
                    meta.insert(
                        "abstract".to_string(),
                        serde_json::Value::String(format!("Abstract for document {}", i)),
                    );
                    meta.insert(
                        "keywords".to_string(),
                        serde_json::Value::Array(vec![
                            serde_json::Value::String("keyword1".to_string()),
                            serde_json::Value::String("keyword2".to_string()),
                        ]),
                    );
                    meta.insert(
                        "citation_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(i % 100)),
                    );
                    meta.insert(
                        "download_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(i * 10)),
                    );
                    meta.insert(
                        "quality_score".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(0.5 + (i % 50) as f64 / 100.0).unwrap(),
                        ),
                    );

                    meta
                })
                .collect();

            // Test metadata transformation performance
            let (transformed, measurement) = measure_performance!(
                &format!("Metadata transformation - {} records", count),
                count,
                {
                    let mut filterable_batch = Vec::new();
                    let mut extra_meta_batch = Vec::new();

                    for record in &records {
                        let mut filterable = HashMap::new();
                        let mut extra_meta = HashMap::new();

                        for (key, value) in record {
                            if filterable_columns.contains(&key.as_str()) {
                                filterable.insert(key.clone(), value.clone());
                            } else {
                                extra_meta.insert(key.clone(), value.clone());
                            }
                        }

                        filterable_batch.push(filterable);
                        extra_meta_batch.push(extra_meta);
                    }

                    (filterable_batch, extra_meta_batch)
                }
            );

            let (filterable_batch, extra_meta_batch) = transformed;
            assert_eq!(filterable_batch.len(), count);
            assert_eq!(extra_meta_batch.len(), count);

            // Performance expectations (records per second)
            match count {
                100 => assert!(measurement.throughput > 1000.0),
                500 => assert!(measurement.throughput > 500.0),
                1000 => assert!(measurement.throughput > 300.0),
                5000 => assert!(measurement.throughput > 100.0),
                _ => {}
            }
        }

        debug!("✅ Metadata processing performance test completed");
        Ok(())
    }

    #[tokio::test]
    async fn test_search_performance_real_data() -> Result<()> {
        // init_test_env(); // Removed - from common module

        // Note: This test assumes ProximaDB server is already running on port 5679
        // In CI/CD, start the server before running tests:
        // cargo run --release --bin proximadb-server &

        // Connect to gRPC server with retry logic
        let mut retries = 5;
        let channel = loop {
            match Channel::from_static("http://127.0.0.1:5679")
                .connect()
                .await 
            {
                Ok(ch) => break ch,
                Err(e) if retries > 0 => {
                    debug!("⏳ Waiting for server to start... (retries left: {})", retries);
                    retries -= 1;
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(e) => return Err(anyhow::anyhow!("Failed to connect to gRPC server: {}", e)),
            }
        };
        let mut client = ProximaDbClient::new(channel);

        // Create test collection
        let collection_name = format!("test_collection_{}", uuid::Uuid::new_v4());
        let dimension = 384;
        
        let create_request = Request::new(CollectionRequest {
            operation: Some(CollectionOperation::Create(
                proximadb::proto::proximadb_v1::CreateCollectionRequest {
                    config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                        name: collection_name.clone(),
                        dimension,
                        distance_metric: proximadb::proto::proximadb_v1::DistanceMetric::Cosine as i32,
                        storage_engine: proximadb::proto::proximadb_v1::StorageEngine::Sst as i32,
                        primary_indexing_algorithm: proximadb::proto::proximadb_v1::IndexingAlgorithm::Hnsw as i32,
                        filterable_columns: vec![
                            FilterableColumnSpec {
                                name: "category".to_string(),
                                data_type: FilterableDataType::FilterableString as i32,
                                indexed: true,
                                supports_range: false,
                                estimated_cardinality: None,
                compression: None,
                optimization_hints: None,
                        encoding_hint: None,
                    },
                        ],
                        index_configs: vec![],
                        quantization: None,
                        primary_index: "default".to_string(),
                        auto_index_selection: false,
                        description: Some("Performance test collection".to_string()),
                        tags: vec!["perf_test".to_string()],
                        owner: Some("test_user".to_string()),
                    }),
                },
            )),
        });
        
        let create_response = client.collection_operation(create_request).await?;
        assert!(create_response.into_inner().success);
        debug!("✅ Created collection: {}", collection_name);

        // Test different dataset sizes
        let dataset_sizes = vec![1000, 5000, 10000];
        
        for dataset_size in dataset_sizes {
            debug!("\n📊 Testing with {} vectors...", dataset_size);
            
            // Insert vectors in batches
            let batch_size = 1000;
            let num_batches = (dataset_size + batch_size - 1) / batch_size;
            
            let insert_start = Instant::now();
            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = ((batch_idx + 1) * batch_size).min(dataset_size);
                let batch_count = end_idx - start_idx;
                
                // Create vector batch
                let mut vector_ops = Vec::new();
                for i in start_idx..end_idx {
                    let vector_data = vec![i as f32 / count as f32; dimension];
                    let mut metadata = HashMap::new();
                    metadata.insert(
                        "category".to_string(),
                        MetadataValue {
                            value: Some(metadata_value::Value::StringValue(
                                format!("cat_{}", i % 10)
                            )),
                        },
                    );
                    
                    vector_ops.push(VectorOperation::Insert(
                        proximadb::proto::proximadb_v1::VectorInsert {
                            id: Some(format!("vec_{:06}", i)),
                            vector: vector_data,
                            metadata,
                        },
                    ));
                }
                
                let insert_request = Request::new(VectorBatchRequest {
                    collection_name: collection_name.clone(),
                    operations: vector_ops,
                });
                
                let response = client.vector_batch_operation(insert_request).await?;
                assert!(response.into_inner().success);
            }
            let insert_duration = insert_start.elapsed();
            debug!("  Inserted {} vectors in {:.2}s ({:.0} vectors/sec)", 
                dataset_size, 
                insert_duration.as_secs_f64(),
                dataset_size as f64 / insert_duration.as_secs_f64()
            );
            
            // Wait for vectors to be flushed from memtable to storage
            tokio::time::sleep(Duration::from_secs(2)).await;
            
            // Test search performance - first without filter (full scan)
            let query_vector = vec![0.5; dimension];
            
            let search_start = Instant::now();
            let search_request = Request::new(VectorSearchRequest {
                collection_id: collection_name.clone(),
                queries: vec![SearchQuery {
                    vector: query_vector.clone(),
                    metadata_filter: None,
                    namespace: None,
                }],
                top_k: 10,
                distance_metric_override: Some(proximadb::proto::proximadb_v1::DistanceMetric::Cosine as i32),
                search_params: Some(SearchParameters {
                    ef_search: Some(50),
                    max_connections: None,
                    num_probes: None,
                    quantization: None,
                    rerank_k: None,
                }),
                include_fields: Some(IncludeFields {
                    vector: false,
                    metadata: true,
                    score: true,
                    rank: false,
                }),
            });
            
            let search_response = client.vector_search(search_request).await?;
            let full_scan_duration = search_start.elapsed();
            let full_scan_results = search_response.into_inner().results.len();
            
            debug!("  Full scan search: {:.2}ms for {} results", 
                full_scan_duration.as_secs_f64() * 1000.0,
                full_scan_results
            );
            
            // Test search with filter (10% selectivity)
            let filtered_search_start = Instant::now();
            let filtered_search_request = Request::new(VectorSearchRequest {
                collection_id: collection_name.clone(),
                queries: vec![SearchQuery {
                    vector: query_vector.clone(),
                    metadata_filter: Some(MetadataFilter {
                        field: "category".to_string(),
                        operator: proximadb::proto::proximadb_v1::FilterOperator::Eq as i32,
                        value: Some(MetadataValue {
                            value: Some(metadata_value::Value::StringValue(
                                "cat_1".to_string()
                            )),
                        }),
                    }),
                    namespace: None,
                }],
                top_k: 10,
                distance_metric_override: Some(proximadb::proto::proximadb_v1::DistanceMetric::Cosine as i32),
                search_params: Some(SearchParameters {
                    ef_search: Some(50),
                    max_connections: None,
                    num_probes: None,
                    quantization: None,
                    rerank_k: None,
                }),
                include_fields: Some(IncludeFields {
                    vector: false,
                    metadata: true,
                    score: true,
                    rank: false,
                }),
            });
            
            let filtered_response = client.vector_search(filtered_search_request).await?;
            let filtered_duration = filtered_search_start.elapsed();
            let filtered_results = filtered_response.into_inner().results.len();
            
            debug!("  Filtered search (10% selectivity): {:.2}ms for {} results", 
                filtered_duration.as_secs_f64() * 1000.0,
                filtered_results
            );
            
            // Calculate speedup
            let speedup = full_scan_duration.as_secs_f64() / filtered_duration.as_secs_f64();
            debug!("  Filtered search speedup: {:.1}x", speedup);
            
            // For datasets >= 10k vectors, filtered search should be significantly faster
            if dataset_size >= 10000 {
                assert!(
                    speedup > 2.0,
                    "Expected filtered search speedup > 2.0x for {} vectors, but got {:.2}x",
                    dataset_size,
                    speedup
                );
            }
        }
        
        // Clean up - delete collection
        let delete_request = Request::new(CollectionRequest {
            operation: Some(CollectionOperation::Delete(
                proximadb::proto::proximadb_v1::DeleteCollectionRequest {
                    name: collection_name.clone(),
                },
            )),
        });
        client.collection_operation(delete_request).await?;
        
        debug!("\n✅ Search performance test with real data completed");
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_operations_performance() -> Result<()> {
        // init_test_env(); // Removed - from common module

        let collection_id = format!("test_collection_{}", uuid::Uuid::new_v4());
        let num_threads = 4;
        let operations_per_thread = 100;

        // Test concurrent vector creation
        let start = Instant::now();
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let collection_id = collection_id.clone();
                tokio::spawn(async move {
                    let mut results = Vec::new();
                    for i in 0..operations_per_thread {
                        let vector_id = format!("thread_{}_vector_{}", thread_id, i);
                        let vector_record =
                            create_test_vector_record(vector_id, collection_id.clone(), 384);
                        results.push(vector_record);
                    }
                    results
                })
            })
            .collect();

        // Wait for all threads to complete
        let mut all_results = Vec::new();
        for handle in handles {
            let thread_results = handle.await.unwrap();
            all_results.extend(thread_results);
        }

        let duration = start.elapsed();
        let total_operations = num_threads * operations_per_thread;
        let throughput = total_operations as f64 / duration.as_secs_f64();

        assert_eq!(all_results.len(), total_operations);

        debug!("📊 Concurrent operations:");
        debug!("   Threads: {}", num_threads);
        debug!("   Total operations: {}", total_operations);
        debug!("   Duration: {:.2}ms", duration.as_secs_f64() * 1000.0);
        debug!("   Throughput: {:.1} ops/sec", throughput);

        // Should achieve reasonable concurrent throughput
        assert!(throughput > 1000.0);

        debug!("✅ Concurrent operations performance test completed");
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_usage_estimation() -> Result<()> {
        // init_test_env(); // Removed - from common module

        let vector_counts = vec![1000, 5000, 10000];
        let dimension = 384;

        for count in vector_counts {
            let vectors =
                // create_test_vector_batch functionality needs to be reimplemented
                let _batch_id = format!("batch_{}", uuid::Uuid::new_v4());

            // Estimate memory usage
            let vector_size_bytes = dimension * 4; // 4 bytes per f32
            let metadata_size_estimate = 500; // Rough estimate per record
            let record_size_estimate = vector_size_bytes + metadata_size_estimate;
            let total_size_mb = (count * record_size_estimate) as f64 / 1024.0 / 1024.0;

            debug!("📊 Memory estimation for {} vectors:", count);
            debug!("   Vector size: {} bytes", vector_size_bytes);
            debug!("   Estimated record size: {} bytes", record_size_estimate);
            debug!("   Total estimated size: {:.2} MB", total_size_mb);

            // Verify reasonable memory usage
            assert!(total_size_mb < 1000.0); // Should be under 1GB for test data
            assert_eq!(vectors.len(), count);

            // Verify vector dimensions
            for vector in &vectors {
                assert_eq!(vector.vector.len(), dimension);
            }
        }

        debug!("✅ Memory usage estimation test completed");
        Ok(())
    }
}
