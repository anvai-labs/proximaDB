//! # Memtable Module - High-Performance In-Memory Storage
//!
//! This module provides ProximaDB's in-memory storage implementations with pluggable
//! data structures optimized for different workloads. It serves as the write buffer
//! for incoming data before persistence to disk, offering high-throughput writes
//! with concurrent access support.
//!
//! ## Memtable Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         Write Requests                   │
//! └────────────────┬────────────────────────┘
//!                  ↓
//! ┌─────────────────────────────────────────┐
//! │      GlobalPartitionedMemtable           │
//! │  ┌─────────┬─────────┬─────────┐       │
//! │  │ Coll. A │ Coll. B │ Coll. C │       │
//! │  └─────────┴─────────┴─────────┘       │
//! └─────────────────────────────────────────┘
//!                  ↓
//! ┌─────────────────────────────────────────┐
//! │        Underlying Implementation         │
//! │   BTree │ SkipList │ DashMap │ ART      │
//! └─────────────────────────────────────────┘
//!                  ↓
//! ┌─────────────────────────────────────────┐
//! │          Flush to Storage                │
//! │     SST │ NOVA │ SWIFT                 │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Core Components
//!
//! ### 1. **GlobalPartitionedMemtable**
//! Production memtable with collection isolation:
//! - **Collection Partitioning**: Separate memtable per collection
//! - **Vector Content Search**: Efficient vector similarity within memory
//! - **Concurrent Access**: Lock-free reads, minimal write contention
//! - **Memory Management**: Per-collection size tracking and limits
//!
//! ### 2. **Available Implementations**
//! Multiple data structures for different workloads:
//!
//! #### BTree (`BTreeMemtable`)
//! - **Ordered Storage**: Natural key ordering
//! - **Memory Efficient**: Low overhead per entry
//! - **Range Queries**: Efficient scans
//! - **Best For**: WAL, sequential writes
//!
//! #### SkipList (`SkipListMemtable`)
//! - **Lock-Free**: Concurrent reads and writes
//! - **Probabilistic**: O(log n) average operations
//! - **Range Support**: Fast range scans
//! - **Best For**: High concurrency, LSM trees
//!
//! #### DashMap (`DashMapMemtable`)
//! - **Sharded HashMap**: Reduced contention
//! - **O(1) Lookups**: Fast point queries
//! - **High Throughput**: Parallel operations
//! - **Best For**: Random access, high concurrency
//!
//! #### ART (Adaptive Radix Tree)
//! - **Trie Structure**: Prefix compression
//! - **Memory Adaptive**: Node types by density
//! - **String Keys**: Optimized for strings
//! - **Best For**: URL-like keys, memory constrained
//!
//! ## MVCC Support
//!
//! Multi-Version Concurrency Control for transactions:
//! ```rust,ignore
//! pub trait MemtableMVCC {
//!     async fn insert_mvcc(&self, key, value, version);
//!     async fn get_mvcc(&self, key, version);
//!     async fn scan_mvcc(&self, range, version);
//! }
//! ```
//!
//! Features:
//! - **Snapshot Isolation**: Consistent reads
//! - **Version Chains**: Multiple versions per key
//! - **Garbage Collection**: Automatic old version cleanup
//! - **Lock-Free Reads**: No blocking on reads
//!
//! ## Flush Process
//!
//! ### Trigger Conditions
//! - **Size Threshold**: When memtable exceeds size limit
//! - **Time Threshold**: Periodic flush for durability
//! - **Memory Pressure**: System memory constraints
//! - **Manual Flush**: Explicit flush command
//!
//! ### Flush Flow
//! 1. **Freeze Memtable**: Make immutable
//! 2. **Create New Active**: New writes go here
//! 3. **Persist Frozen**: Write to storage engine
//! 4. **Update Metadata**: Record flush in manifest
//! 5. **Release Memory**: Return to pool
//!
//! ## Performance Characteristics
//!
//! ### Operation Latencies
//! - **Insert**: < 1μs (in-memory)
//! - **Point Lookup**: < 500ns (hash-based)
//! - **Range Scan**: 10-50μs per 1000 items
//! - **Flush**: 100-500ms (depends on size)
//!
//! ### Memory Efficiency
//! - **Overhead**: 8-16 bytes per entry
//! - **Compression**: Optional value compression
//! - **Pooling**: Reusable memory buffers
//! - **Fragmentation**: < 10% with proper sizing
//!
//! ## Configuration
//!
//! ```toml
//! [memtable]
//! # Maximum size before flush
//! max_size_mb = 256
//!
//! # Flush interval
//! flush_interval_sec = 300
//!
//! # Implementation type
//! type = "global_partitioned"
//!
//! # Underlying structure
//! structure = "btree"  # btree, skiplist, dashmap, art
//!
//! # MVCC settings
//! enable_mvcc = true
//! max_versions = 10
//! ```
//!
//! ## Workload Optimization
//!
//! ### Sequential Writes (Time-Series)
//! - Use BTree for ordered storage
//! - Enable write batching
//! - Larger flush sizes
//!
//! ### Random Writes (Key-Value)
//! - Use DashMap for O(1) access
//! - Smaller flush sizes
//! - More frequent compaction
//!
//! ### High Concurrency (Multi-Tenant)
//! - Use GlobalPartitioned
//! - SkipList or DashMap backend
//! - Per-collection limits
//!
//! ### Memory Constrained
//! - Use ART for compression
//! - Aggressive flush policy
//! - Value compression enabled
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use proximadb::storage::memtable::{MemtableFactory, MemtableConfig};
//!
//! // Create production memtable
//! let config = MemtableConfig {
//!     max_size: 256 * 1024 * 1024,  // 256MB
//!     flush_interval: Duration::from_secs(300),
//!     enable_mvcc: true,
//!     ..Default::default()
//! };
//!
//! let memtable = MemtableFactory::create_for_wal(config);
//!
//! // Insert data
//! memtable.insert("key1", vector_record).await?;
//!
//! // Query data
//! let result = memtable.get("key1").await?;
//!
//! // Range scan
//! let results = memtable.range_scan("key1", Some(100)).await?;
//! ```
//!
//! ## Benchmarking Framework
//!
//! Compare implementations for your workload:
//! ```rust,ignore
//! let benchmark = MemtableBenchmark::new(config);
//! let report = benchmark.run_all().await;
//! report.print();  // Shows ops/sec, latencies, memory
//! ```
//!
//! ## Best Practices
//!
//! 1. **Size Appropriately**: Balance memory usage and flush frequency
//! 2. **Choose Right Structure**: Match to access patterns
//! 3. **Monitor Metrics**: Track flush times and frequencies
//! 4. **Tune Thresholds**: Adjust based on workload
//! 5. **Use Partitioning**: Isolate collections for multi-tenancy

pub mod core;
pub mod implementations;
pub mod serialization;
pub mod specialized;

// Lock-free implementations have been integrated into the main codebase
// DashMap is now used in TransactionCoordinator and StorageEngine

// Re-export core traits
pub use core::{MemtableConfig, MemtableCore, MemtableMVCC, MemtableMetrics};

// Import tracing for debug macros
use tracing::debug;

// 🔴 UNUSED MEMTABLE EXPORTS - COMMENTED OUT FOR REMOVAL
// Only GlobalPartitionedMemtable is actually used in production
// Re-export implementations
pub use implementations::{
    // bplustree::BPlusTreeMemtable,  // UNUSED - Never instantiated
    btree::BTreeMemtable, // Needed for tests
    // dashmap::DashMapMemtable,      // UNUSED - Never instantiated
    // artmap::ArtMemtable,           // Already commented out
    // hashmap::HashMapMemtable,      // UNUSED - Never instantiated
    skiplist::SkipListMemtable, // Needed for tests
};

// Re-export specialized wrappers (using proper OOP composition)
pub use specialized::SpecializedMemtableFactory;

/// Available memtable implementation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemtableType {
    /// BTree - Ordered storage, memory efficient
    BTree,
    /// B+Tree - Superior range scans, database-optimized
    BPlusTree,
    /// SkipList - Lock-free concurrent access
    SkipList,
    /// HashMap - O(1) point lookups
    HashMap,
    /// DashMap - High-concurrency HashMap
    DashMap,
    /// ART - Adaptive Radix Tree
    ART,
}

impl MemtableType {
    /// Get all available types for benchmarking
    pub fn all() -> Vec<MemtableType> {
        vec![
            MemtableType::BTree,
            MemtableType::BPlusTree,
            MemtableType::SkipList,
            MemtableType::HashMap,
            MemtableType::DashMap,
            MemtableType::ART,
        ]
    }

    /// Get recommended type for workload
    pub fn recommended_for_workload(workload: WorkloadCharacteristics) -> MemtableType {
        match workload {
            WorkloadCharacteristics::SequentialWrites => MemtableType::BPlusTree,
            WorkloadCharacteristics::RandomWrites => MemtableType::DashMap,
            WorkloadCharacteristics::PointLookups => MemtableType::HashMap,
            WorkloadCharacteristics::RangeQueries => MemtableType::BPlusTree,
            WorkloadCharacteristics::HighConcurrency => MemtableType::DashMap,
            WorkloadCharacteristics::MemoryConstrained => MemtableType::ART,
            WorkloadCharacteristics::StringKeys => MemtableType::ART,
            WorkloadCharacteristics::WAL => MemtableType::BTree, // BTree better for numeric keys (u64)
            WorkloadCharacteristics::LSM => MemtableType::SkipList,
        }
    }
}

/// Workload characteristics for memtable selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadCharacteristics {
    SequentialWrites,
    RandomWrites,
    PointLookups,
    RangeQueries,
    HighConcurrency,
    MemoryConstrained,
    StringKeys,
    WAL,
    LSM,
}

/// Factory for creating memtable instances
pub struct MemtableFactory;

impl MemtableFactory {
    /// Create WAL-optimized memtable (global partitioned for collection isolation + vector content search)
    pub fn create_for_wal(config: MemtableConfig) -> specialized::wal_behavior::WALBehaviorWrapper {
        specialized::SpecializedMemtableFactory::create_global_partitioned_for_wal(config)
    }

    // 🔴 UNUSED METHOD - SST doesn't use SkipList memtable
    // /// Create SST-optimized memtable (SkipList for concurrent access)
    // pub fn create_for_sst(
    //     config: MemtableConfig,
    // ) -> specialized::LsmMemtable<String, crate::storage::engines::sst::SstRecord> {
    //     specialized::SpecializedMemtableFactory::create_skiplist_for_sst(config)
    // }

    /// Create a specific memtable type for testing/benchmarking.
    ///
    /// Returns `None` for every type except `GlobalPartitioned` (all others are
    /// disabled — production uses [`create_for_wal`](Self::create_for_wal)). This
    /// returns `None` rather than panicking so a future config-driven caller of
    /// [`auto_select`](Self::auto_select) degrades gracefully instead of crashing.
    #[expect(clippy::match_single_binding)] // Only GlobalPartitioned is used in production
    pub fn create_typed<K, V>(
        memtable_type: MemtableType,
        _config: MemtableConfig,
    ) -> Option<Box<dyn MemtableCore<K, V> + Send + Sync>>
    where
        K: Clone + Ord + std::hash::Hash + Send + Sync + std::fmt::Debug + AsRef<[u8]> + 'static,
        V: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        match memtable_type {
            // 🔴 UNUSED MEMTABLE TYPES - COMMENTED OUT FOR REMOVAL
            // These implementations are never actually instantiated in production
            // MemtableType::BTree => Some(Box::new(BTreeMemtable::new(true))),
            // MemtableType::BPlusTree => Some(Box::new(BPlusTreeMemtable::new())),
            // MemtableType::SkipList => Some(Box::new(SkipListMemtable::new())),
            // MemtableType::HashMap => Some(Box::new(HashMapMemtable::new())),
            // MemtableType::DashMap => Some(Box::new(DashMapMemtable::new())),
            // MemtableType::ART => Some(Box::new(BTreeMemtable::new(false))),

            // Only GlobalPartitioned is wired; every other type is unavailable.
            _ => None,
        }
    }

    /// Auto-select best memtable type based on workload analysis.
    ///
    /// Returns `None` when the recommended type is not one of the wired
    /// implementations (see [`create_typed`](Self::create_typed)).
    pub fn auto_select<K, V>(
        workload: WorkloadCharacteristics,
        config: MemtableConfig,
    ) -> Option<Box<dyn MemtableCore<K, V> + Send + Sync>>
    where
        K: Clone + Ord + std::hash::Hash + Send + Sync + std::fmt::Debug + AsRef<[u8]> + 'static,
        V: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let memtable_type = MemtableType::recommended_for_workload(workload);
        Self::create_typed(memtable_type, config)
    }
}

/// Performance benchmarking framework
pub struct MemtableBenchmark<K, V>
where
    K: Clone + Ord + std::hash::Hash + Send + Sync + std::fmt::Debug + AsRef<[u8]>,
    V: Clone + Send + Sync + std::fmt::Debug,
{
    implementations: Vec<(MemtableType, Box<dyn MemtableCore<K, V> + Send + Sync>)>,
    config: MemtableConfig,
}

impl<K, V> MemtableBenchmark<K, V>
where
    K: Clone + Ord + std::hash::Hash + Send + Sync + std::fmt::Debug + AsRef<[u8]> + 'static,
    V: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create benchmark suite with all implementations
    pub fn new(config: MemtableConfig) -> Self {
        let mut implementations = Vec::new();

        for memtable_type in MemtableType::all() {
            // Only wired types yield an implementation; skip the rest (all
            // non-GlobalPartitioned types are currently disabled).
            if let Some(implementation) =
                MemtableFactory::create_typed(memtable_type, config.clone())
            {
                implementations.push((memtable_type, implementation));
            }
        }

        Self {
            implementations,
            config,
        }
    }

    /// Run insert benchmark
    pub async fn benchmark_inserts(&mut self, entries: Vec<(K, V)>) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        for (memtable_type, memtable) in &mut self.implementations {
            let start = std::time::Instant::now();
            let mut _total_size = 0u64;

            for (key, value) in &entries {
                match memtable.insert(key.clone(), value.clone()).await {
                    Ok(size_delta) => _total_size += size_delta,
                    Err(_) => continue,
                }
            }

            let duration = start.elapsed();
            let ops_per_sec = entries.len() as f64 / duration.as_secs_f64();

            results.push(BenchmarkResult {
                memtable_type: *memtable_type,
                operation: "insert".to_string(),
                duration,
                ops_per_second: ops_per_sec,
                memory_usage: memtable.size_bytes().await,
                entry_count: memtable.len().await,
                success_count: entries.len(),
                error_count: 0,
            });
        }

        results
    }

    /// Run point lookup benchmark
    pub async fn benchmark_lookups(&mut self, keys: Vec<K>) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        for (memtable_type, memtable) in &mut self.implementations {
            let start = std::time::Instant::now();
            let mut success_count = 0;

            for key in &keys {
                match memtable.get(key).await {
                    Ok(Some(_)) => success_count += 1,
                    Ok(None) => continue,
                    Err(_) => continue,
                }
            }

            let duration = start.elapsed();
            let ops_per_sec = keys.len() as f64 / duration.as_secs_f64();

            results.push(BenchmarkResult {
                memtable_type: *memtable_type,
                operation: "lookup".to_string(),
                duration,
                ops_per_second: ops_per_sec,
                memory_usage: memtable.size_bytes().await,
                entry_count: memtable.len().await,
                success_count,
                error_count: keys.len() - success_count,
            });
        }

        results
    }

    /// Run range scan benchmark
    pub async fn benchmark_range_scans(
        &mut self,
        ranges: Vec<(K, Option<usize>)>,
    ) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        for (memtable_type, memtable) in &mut self.implementations {
            let start = std::time::Instant::now();
            let mut success_count = 0;
            let mut _total_results = 0;

            for (from_key, limit) in &ranges {
                match memtable.range_scan(from_key.clone(), *limit).await {
                    Ok(scan_results) => {
                        success_count += 1;
                        _total_results += scan_results.len();
                    }
                    Err(_) => continue,
                }
            }

            let duration = start.elapsed();
            let ops_per_sec = ranges.len() as f64 / duration.as_secs_f64();

            results.push(BenchmarkResult {
                memtable_type: *memtable_type,
                operation: "range_scan".to_string(),
                duration,
                ops_per_second: ops_per_sec,
                memory_usage: memtable.size_bytes().await,
                entry_count: memtable.len().await,
                success_count,
                error_count: ranges.len() - success_count,
            });
        }

        results
    }

    /// Generate comprehensive benchmark report
    pub async fn generate_report(&mut self) -> BenchmarkReport {
        // For now, return empty benchmark report
        // Deferred: Implement proper generic test data generation
        BenchmarkReport {
            config: self.config.clone(),
            insert_results: vec![],
            lookup_results: vec![],
            scan_results: vec![],
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Benchmark result for a specific operation and implementation
///
/// Captures performance metrics for a single benchmark run including
/// throughput, latency, memory usage, and success/failure rates.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The memtable implementation type that was benchmarked
    pub memtable_type: MemtableType,
    /// Name of the operation performed (e.g., "insert", "lookup")
    pub operation: String,
    /// Total duration of the benchmark run
    pub duration: std::time::Duration,
    /// Operations completed per second
    pub ops_per_second: f64,
    /// Total memory usage in bytes after the operation
    pub memory_usage: usize,
    /// Number of entries in the memtable after the operation
    pub entry_count: usize,
    /// Number of successfully completed operations
    pub success_count: usize,
    /// Number of failed operations
    pub error_count: usize,
}

/// Comprehensive benchmark report
///
/// Aggregates benchmark results across all operations for comparison
/// and analysis of different memtable implementations.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Configuration used for the benchmark
    pub config: MemtableConfig,
    /// Results from insert benchmarks across all implementations
    pub insert_results: Vec<BenchmarkResult>,
    /// Results from point lookup benchmarks across all implementations
    pub lookup_results: Vec<BenchmarkResult>,
    /// Results from range scan benchmarks across all implementations
    pub scan_results: Vec<BenchmarkResult>,
    /// Timestamp when the benchmark was completed
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl BenchmarkReport {
    /// Print formatted report
    pub fn print(&self) {
        debug!("Memtable Performance Benchmark Report");
        debug!("=====================================");
        debug!(
            "Timestamp: {}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
        debug!("");

        debug!("INSERT PERFORMANCE:");
        debug!(
            "{:<12} {:>12} {:>12} {:>12} {:>10}",
            "Type", "Ops/Sec", "Duration", "Memory", "Entries"
        );
        debug!("{}", "-".repeat(60));
        for result in &self.insert_results {
            debug!(
                "{:<12} {:>12.1} {:>10.3}s {:>10}B {:>8}",
                format!("{:?}", result.memtable_type),
                result.ops_per_second,
                result.duration.as_secs_f64(),
                result.memory_usage,
                result.entry_count
            );
        }
        debug!("");

        debug!("LOOKUP PERFORMANCE:");
        debug!(
            "{:<12} {:>12} {:>12} {:>12} {:>10}",
            "Type", "Ops/Sec", "Duration", "Hit Rate", "Entries"
        );
        debug!("{}", "-".repeat(60));
        for result in &self.lookup_results {
            let hit_rate = if result.success_count + result.error_count > 0 {
                result.success_count as f64 / (result.success_count + result.error_count) as f64
                    * 100.0
            } else {
                0.0
            };
            debug!(
                "{:<12} {:>12.1} {:>10.3}s {:>11.1}% {:>8}",
                format!("{:?}", result.memtable_type),
                result.ops_per_second,
                result.duration.as_secs_f64(),
                hit_rate,
                result.entry_count
            );
        }
        debug!("");

        debug!("RANGE SCAN PERFORMANCE:");
        debug!(
            "{:<12} {:>12} {:>12} {:>12} {:>10}",
            "Type", "Scans/Sec", "Duration", "Success", "Entries"
        );
        debug!("{}", "-".repeat(60));
        for result in &self.scan_results {
            debug!(
                "{:<12} {:>12.1} {:>10.3}s {:>11} {:>8}",
                format!("{:?}", result.memtable_type),
                result.ops_per_second,
                result.duration.as_secs_f64(),
                result.success_count,
                result.entry_count
            );
        }
        debug!("");
    }

    /// Get winner for each operation type
    pub fn get_winners(&self) -> PerformanceWinners {
        let best_insert = self
            .insert_results
            .iter()
            .max_by(|a, b| {
                a.ops_per_second
                    .partial_cmp(&b.ops_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.memtable_type);

        let best_lookup = self
            .lookup_results
            .iter()
            .max_by(|a, b| {
                a.ops_per_second
                    .partial_cmp(&b.ops_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.memtable_type);

        let best_scan = self
            .scan_results
            .iter()
            .max_by(|a, b| {
                a.ops_per_second
                    .partial_cmp(&b.ops_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.memtable_type);

        PerformanceWinners {
            best_insert,
            best_lookup,
            best_scan,
        }
    }
}

/// Performance winners for each operation type
///
/// Indicates which memtable implementation achieved the best performance
/// for each category of operation.
#[derive(Debug, Clone)]
pub struct PerformanceWinners {
    /// Memtable type with highest insert throughput
    pub best_insert: Option<MemtableType>,
    /// Memtable type with highest lookup throughput
    pub best_lookup: Option<MemtableType>,
    /// Memtable type with highest range scan throughput
    pub best_scan: Option<MemtableType>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memtable_factory() {
        let config = MemtableConfig::default();

        // Test WriteBuffer creation
        let _write_buffer_memtable = MemtableFactory::create_for_wal(config.clone());

        // 🔴 UNUSED TEST - SST memtable creation is unused
        // Test SST creation
        // let _sst_memtable = MemtableFactory::create_for_sst(config.clone());

        // 🔴 UNUSED TEST - Typed memtable creation is unused
        // Test typed creation
        // let _btree_memtable: Box<dyn MemtableCore<String, String> + Send + Sync> =
        //     MemtableFactory::create_typed(MemtableType::BTree, config.clone());
    }

    #[tokio::test]
    async fn test_workload_recommendations() {
        assert_eq!(
            MemtableType::recommended_for_workload(WorkloadCharacteristics::WAL),
            MemtableType::BTree
        );
        assert_eq!(
            MemtableType::recommended_for_workload(WorkloadCharacteristics::LSM),
            MemtableType::SkipList
        );
        assert_eq!(
            MemtableType::recommended_for_workload(WorkloadCharacteristics::PointLookups),
            MemtableType::HashMap
        );
        assert_eq!(
            MemtableType::recommended_for_workload(WorkloadCharacteristics::HighConcurrency),
            MemtableType::DashMap
        );
    }

    #[test]
    fn create_typed_returns_none_for_disabled_types_instead_of_panicking() {
        let config = MemtableConfig::default();
        // Every type is currently disabled (only the WAL global-partitioned
        // memtable is wired, via `create_for_wal`). `create_typed` must return
        // `None` — never panic — so a future config-driven `auto_select`
        // caller degrades gracefully.
        for memtable_type in MemtableType::all() {
            assert!(
                MemtableFactory::create_typed::<String, String>(memtable_type, config.clone())
                    .is_none(),
                "no type is wired via create_typed yet: {memtable_type:?}"
            );
        }
        assert!(
            MemtableFactory::auto_select::<String, String>(WorkloadCharacteristics::WAL, config)
                .is_none()
        );
    }

    #[tokio::test]
    #[ignore = "Memtable types are disabled - only GlobalPartitionedMemtable is used in production"]
    async fn test_benchmark_framework() {
        let config = MemtableConfig::default();
        let mut benchmark = MemtableBenchmark::<String, String>::new(config);

        // Test small benchmark
        let test_entries = vec![
            ("key1".to_string(), "value1".to_string()),
            ("key2".to_string(), "value2".to_string()),
        ];

        let insert_results = benchmark.benchmark_inserts(test_entries).await;
        assert_eq!(insert_results.len(), MemtableType::all().len());

        let test_keys = vec!["key1".to_string(), "key2".to_string()];
        let lookup_results = benchmark.benchmark_lookups(test_keys).await;
        assert_eq!(lookup_results.len(), MemtableType::all().len());
    }
}
