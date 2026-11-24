// Import the common test helpers
// ORPHANED TEST (not in any compile target — see TD-ORPHAN-1): references the
// VIPER engine removed by ADR-093 stage 3 / TD-VIPER-1 S3. Unreachable code;
// cleanup of the file itself belongs to the orphan-file program, not the
// engine-removal PR.
#[path = "../common/mod.rs"]
mod common;

//! Unified Search Performance Benchmarks
//!
//! This module benchmarks the unified search implementation across:
//! - WAL + Storage search with deduplication
//! - Different collection sizes
//! - Various distance metrics
//! - Metadata filtering performance
//! - Streaming vs batch search

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use proximadb::proto::proximadb_v1::{
    VectorRecord, MetadataItem, StorageEngine,
};
use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::results::InternalSearchResult;
use proximadb::services::VectorOperationsService;
use proximadb::services::collection_service::CollectionService;
use proximadb::storage::engines::viper::ViperEngine;
use proximadb::storage::engines::sst::LsmTree;
use proximadb::storage::memtable::implementations::GlobalPartitionedMemtable;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use std::sync::Arc;
use tracing::{info, debug};

/// Performance measurement result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub duration_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub items_processed: usize,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub collection_sizes: Vec<usize>,
    pub vector_dimensions: Vec<usize>,
    pub search_k_values: Vec<usize>,
    pub num_queries: usize,
    pub enable_metadata_filter: bool,
    pub flush_percentage: f32, // Percentage of data to keep in WAL
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            collection_sizes: vec![1000, 10000, 100000],
            vector_dimensions: vec![128, 512, 1536],
            search_k_values: vec![10, 50, 100],
            num_queries: 100,
            enable_metadata_filter: true,
            flush_percentage: 0.3, // 30% in WAL, 70% flushed
        }
    }
}

/// Unified search benchmark suite
pub struct UnifiedSearchBenchmark {
    direct_service: Arc<VectorOperationsService>,
    collection_service: Arc<CollectionService>,
    config: BenchmarkConfig,
}

impl UnifiedSearchBenchmark {
    /// Create new benchmark suite
    pub async fn new(config: BenchmarkConfig) -> Result<Self> {
        // Initialize services
        let filesystem_factory = Arc::new(FilesystemFactory::new_with_default());
        let global_memtable = Arc::new(GlobalPartitionedMemtable::new());
        let collection_service = Arc::new(CollectionService::new(
            None, // metadata_store
            filesystem_factory.clone(),
        ));
        
        // Create VectorOperationsService using test utilities
        let direct_service = Arc::new(
            tests::common::integration_test_helpers::create_test_vector_operations_service()
                .await
                .expect("Failed to create VectorOperationsService")
        );
        
        Ok(Self {
            direct_service,
            collection_service,
            config,
        })
    }
    
    /// Run all benchmarks
    pub async fn run_all(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        info!("🚀 Starting Unified Search Benchmarks");
        
        // Benchmark different scenarios
        for collection_size in &self.config.collection_sizes {
            for dimension in &self.config.vector_dimensions {
                for k in &self.config.search_k_values {
                    // Create test collection
                    let collection_id = format!("bench_{}_{}", collection_size, dimension);
                    self.setup_test_collection(&collection_id, *collection_size, *dimension).await?;
                    
                    // Benchmark scenarios
                    results.extend(self.benchmark_pure_wal_search(&collection_id, *dimension, *k).await?);
                    results.extend(self.benchmark_pure_storage_search(&collection_id, *dimension, *k).await?);
                    results.extend(self.benchmark_unified_search(&collection_id, *dimension, *k).await?);
                    
                    if self.config.enable_metadata_filter {
                        results.extend(self.benchmark_filtered_search(&collection_id, *dimension, *k).await?);
                    }
                    
                    // Cleanup
                    self.cleanup_collection(&collection_id).await?;
                }
            }
        }
        
        // Print summary
        self.print_summary(&results);
        
        Ok(results)
    }
    
    /// Setup test collection with data
    async fn setup_test_collection(
        &self,
        collection_id: &str,
        size: usize,
        dimension: usize,
    ) -> Result<()> {
        info!("📦 Setting up collection {} with {} vectors of dimension {}", 
              collection_id, size, dimension);
        
        // Create collection
        self.collection_service.create_collection(
            collection_id.to_string(),
            Some(vec![]), // filterable_columns
            dimension as i32,
            StorageEngine::Viper as i32,
            None, // config
        ).await?;
        
        // Insert vectors in batches
        let batch_size = 1000;
        let num_batches = (size + batch_size - 1) / batch_size;
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = ((batch_idx + 1) * batch_size).min(size);
            let batch_vectors = self.generate_test_vectors(collection_id, start, end, dimension);
            
            // Insert through VectorOperationsService
            self.direct_service.insert_vectors_batch(
                collection_id,
                batch_vectors,
                None, // assignment_info
            ).await?;
        }
        
        // Flush some data to storage based on config
        let flush_count = (size as f32 * self.config.flush_percentage) as usize;
        if flush_count > 0 {
            self.direct_service.manual_flush(collection_id, Some(flush_count)).await?;
        }
        
        debug!("✅ Collection setup complete: {} in WAL, {} in storage", 
               size - flush_count, flush_count);
        
        Ok(())
    }
    
    /// Generate test vectors
    fn generate_test_vectors(
        &self,
        collection_id: &str,
        start: usize,
        end: usize,
        dimension: usize,
    ) -> Vec<VectorRecord> {
        (start..end).map(|i| {
            VectorRecord {
                id: Some(format!("vec_{,
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            distance: None,
            rank: None,
            score: None,
        }", i)),
                vector: vec![i as f32 / 1000.0; dimension], // Simple pattern for reproducibility
                metadata: vec![
                    MetadataItem {
                        key: "category".to_string(),
                        value: Some(proximadb::proto::proximadb_v1::metadata_item::Value::StringValue(format!("cat_{}", i % 10))),
                    },
                    MetadataItem {
                        key: "score".to_string(),
                        value: Some(proximadb::proto::proximadb_v1::metadata_item::Value::StringValue((i % 100).to_string())),
                    },
                ],
                timestamp: chrono::Utc::now().timestamp() as u32,
                updated_at: Some(chrono::Utc::now().timestamp() as u32),
                expires_at: None,
                version: Some(1),
                rank: None,
                score: None,
                distance: None,
            }
        }).collect()
    }
    
    /// Benchmark pure WAL search
    async fn benchmark_pure_wal_search(
        &self,
        collection_id: &str,
        dimension: usize,
        k: usize,
    ) -> Result<Vec<BenchmarkResult>> {
        info!("🔍 Benchmarking pure WAL search: k={}", k);
        
        let mut latencies = Vec::new();
        let query_vector = vec![0.5; dimension];
        
        let start = Instant::now();
        
        for _ in 0..self.config.num_queries {
            let query_start = Instant::now();
            
            // Search only WAL (unflushed data)
            let results = self.direct_service.search_vectors_wal_only(
                collection_id,
                &query_vector,
                k,
                DistanceMetric::Cosine,
            ).await?;
            
            let query_duration = query_start.elapsed();
            latencies.push(query_duration.as_secs_f64() * 1000.0);
            
            assert!(!results.is_empty(), "WAL search returned no results");
        }
        
        let total_duration = start.elapsed();
        
        Ok(vec![self.calculate_stats(
            format!("WAL_Search_k{}_d{}", k, dimension),
            latencies,
            total_duration,
            self.config.num_queries,
        )])
    }
    
    /// Benchmark pure storage search
    async fn benchmark_pure_storage_search(
        &self,
        collection_id: &str,
        dimension: usize,
        k: usize,
    ) -> Result<Vec<BenchmarkResult>> {
        info!("🔍 Benchmarking pure storage search: k={}", k);
        
        let mut latencies = Vec::new();
        let query_vector = vec![0.5; dimension];
        
        let start = Instant::now();
        
        for _ in 0..self.config.num_queries {
            let query_start = Instant::now();
            
            // Search only storage engines
            let (viper_results, lsm_results) = tokio::try_join!(
                self.direct_service.search_viper_engine_enhanced(
                    collection_id,
                    &query_vector,
                    k,
                    DistanceMetric::Cosine,
                    None, // metadata_filters
                    false, // include_vectors
                    false, // include_metadata
                ),
                self.direct_service.search_sst_engine_enhanced(
                    collection_id,
                    &query_vector,
                    k,
                    DistanceMetric::Cosine,
                    None, // metadata_filters
                    false, // include_vectors
                    false, // include_metadata
                )
            )?;
            
            let query_duration = query_start.elapsed();
            latencies.push(query_duration.as_secs_f64() * 1000.0);
            
            assert!(viper_results.len() + lsm_results.len() > 0, "Storage search returned no results");
        }
        
        let total_duration = start.elapsed();
        
        Ok(vec![self.calculate_stats(
            format!("Storage_Search_k{}_d{}", k, dimension),
            latencies,
            total_duration,
            self.config.num_queries,
        )])
    }
    
    /// Benchmark unified search
    async fn benchmark_unified_search(
        &self,
        collection_id: &str,
        dimension: usize,
        k: usize,
    ) -> Result<Vec<BenchmarkResult>> {
        info!("🔍 Benchmarking unified search: k={}", k);
        
        let mut latencies = Vec::new();
        let query_vector = vec![0.5; dimension];
        
        let start = Instant::now();
        
        for _ in 0..self.config.num_queries {
            let query_start = Instant::now();
            
            let results = self.direct_service.search_vectors_unified(
                collection_id,
                &query_vector,
                k,
                DistanceMetric::Cosine,
                None, // search_params
                None, // metadata_filters
                false, // include_vectors
                false, // include_metadata
            ).await?;
            
            let query_duration = query_start.elapsed();
            latencies.push(query_duration.as_secs_f64() * 1000.0);
            
            assert_eq!(results.len(), k.min(results.len()), "Unified search returned wrong number of results");
        }
        
        let total_duration = start.elapsed();
        
        Ok(vec![self.calculate_stats(
            format!("Unified_Search_k{}_d{}", k, dimension),
            latencies,
            total_duration,
            self.config.num_queries,
        )])
    }
    
    /// Benchmark filtered search
    async fn benchmark_filtered_search(
        &self,
        collection_id: &str,
        dimension: usize,
        k: usize,
    ) -> Result<Vec<BenchmarkResult>> {
        info!("🔍 Benchmarking filtered search: k={}", k);
        
        let mut latencies = Vec::new();
        let query_vector = vec![0.5; dimension];
        
        // Create metadata filter
        let mut metadata_filters = HashMap::new();
        metadata_filters.insert("category".to_string(), serde_json::Value::String("cat_5".to_string()));
        
        let start = Instant::now();
        
        for _ in 0..self.config.num_queries {
            let query_start = Instant::now();
            
            let results = self.direct_service.search_vectors_unified(
                collection_id,
                &query_vector,
                k,
                DistanceMetric::Cosine,
                None, // search_params
                Some(&metadata_filters),
                false, // include_vectors
                false, // include_metadata
            ).await?;
            
            let query_duration = query_start.elapsed();
            latencies.push(query_duration.as_secs_f64() * 1000.0);
        }
        
        let total_duration = start.elapsed();
        
        Ok(vec![self.calculate_stats(
            format!("Filtered_Search_k{}_d{}", k, dimension),
            latencies,
            total_duration,
            self.config.num_queries,
        )])
    }
    
    /// Calculate statistics from latencies
    fn calculate_stats(
        &self,
        operation: String,
        mut latencies: Vec<f64>,
        total_duration: Duration,
        items_processed: usize,
    ) -> BenchmarkResult {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50_idx = latencies.len() / 2;
        let p95_idx = (latencies.len() as f64 * 0.95) as usize;
        let p99_idx = (latencies.len() as f64 * 0.99) as usize;
        
        BenchmarkResult {
            operation,
            duration_ms: total_duration.as_secs_f64() * 1000.0,
            throughput_ops_per_sec: items_processed as f64 / total_duration.as_secs_f64(),
            latency_p50_ms: latencies[p50_idx],
            latency_p95_ms: latencies[p95_idx.min(latencies.len() - 1)],
            latency_p99_ms: latencies[p99_idx.min(latencies.len() - 1)],
            items_processed,
        }
    }
    
    /// Cleanup test collection
    async fn cleanup_collection(&self, collection_id: &str) -> Result<()> {
        // In production, would delete collection
        // For benchmarks, we might want to keep for analysis
        Ok(())
    }
    
    /// Print benchmark summary
    fn print_summary(&self, results: &[BenchmarkResult]) {
        debug!("\n📊 Unified Search Benchmark Results");
        debug!("═══════════════════════════════════════════════════════════════════════");
        debug!("{:<30} {:>10} {:>10} {:>10} {:>10} {:>10}", 
                 "Operation", "Throughput", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Total (ms)");
        debug!("───────────────────────────────────────────────────────────────────────");
        
        for result in results {
            debug!("{:<30} {:>10.1} {:>10.2} {:>10.2} {:>10.2} {:>10.1}",
                     result.operation,
                     result.throughput_ops_per_sec,
                     result.latency_p50_ms,
                     result.latency_p95_ms,
                     result.latency_p99_ms,
                     result.duration_ms);
        }
        
        debug!("═══════════════════════════════════════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unified_search_benchmark_small() -> Result<()> {
        // Initialize logging - respects RUST_LOG environment variable
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
            )
            .try_init();
        
        // Small benchmark for CI/CD
        let config = BenchmarkConfig {
            collection_sizes: vec![1000],
            vector_dimensions: vec![128],
            search_k_values: vec![10],
            num_queries: 10,
            enable_metadata_filter: true,
            flush_percentage: 0.5,
        };
        
        let benchmark = UnifiedSearchBenchmark::new(config).await?;
        let results = benchmark.run_all().await?;
        
        // Verify we got results
        assert!(!results.is_empty());
        
        // Basic performance assertions
        for result in &results {
            assert!(result.throughput_ops_per_sec > 0.0);
            assert!(result.latency_p50_ms > 0.0);
            assert!(result.latency_p95_ms >= result.latency_p50_ms);
            assert!(result.latency_p99_ms >= result.latency_p95_ms);
        }
        
        Ok(())
    }
    
    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored
    async fn test_unified_search_benchmark_full() -> Result<()> {
        // Initialize logging - respects RUST_LOG environment variable
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
            )
            .try_init();
        
        // Full benchmark
        let config = BenchmarkConfig::default();
        let benchmark = UnifiedSearchBenchmark::new(config).await?;
        let results = benchmark.run_all().await?;
        
        // Save results to file
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write("unified_search_benchmark_results.json", json)?;
        
        Ok(())
    }
}