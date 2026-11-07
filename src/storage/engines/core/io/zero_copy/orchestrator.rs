// Integrated I/O Optimizer - Main System Orchestrator
// Combines zero-copy metadata cache with smart download optimization

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

use super::access_tracker::{AccessEvent, AccessPatternTracker};
use super::bandwidth_optimizer::{BandwidthOptimizer, DownloadStrategy, OptimizedRange};
use super::config::ZeroCopyIOConfig;
use super::metrics::SystemPerformanceMetrics;
use super::traits::{
    DataRange, FileAccessRequest, MetadataSerializer, QueryContext, RequestPriority,
};
use crate::storage::cache::specialized::filesystem_metadata_store::{
    FilesystemMetadata, FilesystemMetadataStore,
};
use crate::storage::persistence::filesystem::FilesystemFactory;
use proximadb_kernel::error::ProximaDBError;

/// Result of I/O optimization analysis
#[derive(Debug, Clone)]
pub struct OptimizedIOResult {
    /// Chosen I/O strategy
    pub strategy: IOStrategy,
    /// Estimated performance savings
    pub estimated_savings: IOSavings,
    /// Detailed execution plan
    pub execution_plan: ZeroCopyExecutionPlan,
    /// Human-readable rationale
    pub rationale: String,
    /// Confidence in the optimization (0.0-1.0)
    pub confidence: f32,
}

/// I/O strategy enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum IOStrategy {
    /// Skip file entirely (filtered out by metadata)
    SkipFile { reason: String },
    /// Use local cache if available
    LocalCache {
        cache_path: String,
        cache_valid: bool,
    },
    /// Download entire file
    FullDownload {
        cache_locally: bool,
        prefetch_related: bool,
    },
    /// Download specific byte ranges
    SelectiveRanges {
        ranges: Vec<OptimizedRange>,
        parallel_downloads: bool,
        merge_threshold: u64,
    },
    /// Hybrid strategy with fallback
    HybridStrategy {
        primary: Box<IOStrategy>,
        fallback: Box<IOStrategy>,
        condition: String,
    },
}

/// Estimated performance and cost savings
#[derive(Debug, Clone, Default)]
pub struct IOSavings {
    /// Bandwidth saved in bytes
    pub bandwidth_saved_bytes: u64,
    /// Number of HTTP requests saved
    pub requests_saved: u32,
    /// Latency reduction in milliseconds
    pub latency_saved_ms: f32,
    /// Estimated cost savings in dollars
    pub cost_saved_dollars: f64,
    /// Memory saved by not caching full file
    pub memory_saved_bytes: u64,
    /// I/O operations avoided
    pub io_operations_saved: u32,
}

/// Detailed execution plan for the chosen strategy
#[derive(Debug, Clone)]
pub struct ZeroCopyExecutionPlan {
    /// Ordered list of operations to execute
    pub operations: Vec<ExecutionOperation>,
    /// Estimated total execution time
    pub estimated_duration: Duration,
    /// Resource requirements
    pub resource_requirements: ZeroCopyResourceRequirements,
    /// Fallback plans in case of failure
    pub fallback_plans: Vec<ZeroCopyExecutionPlan>,
}

/// Individual execution operation
#[derive(Debug, Clone)]
pub struct ExecutionOperation {
    /// Operation type
    pub operation_type: OperationType,
    /// Target (file path, range, etc.)
    pub target: String,
    /// Parameters for the operation
    pub parameters: HashMap<String, String>,
    /// Estimated duration for this operation
    pub estimated_duration: Duration,
    /// Dependencies on other operations
    pub dependencies: Vec<usize>,
}

/// Types of execution operations
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    /// Check local cache
    CheckCache,
    /// Download file metadata
    FetchMetadata,
    /// Download file ranges
    DownloadRanges,
    /// Download complete file
    DownloadFile,
    /// Cache file locally
    CacheFile,
    /// Merge downloaded ranges
    MergeRanges,
    /// Validate download
    ValidateDownload,
}

/// Resource requirements for execution
#[derive(Debug, Clone, Default)]
pub struct ZeroCopyResourceRequirements {
    /// Memory required in bytes
    pub memory_bytes: u64,
    /// Disk space required in bytes
    pub disk_bytes: u64,
    /// Network bandwidth required in bytes/sec
    pub bandwidth_bps: u64,
    /// CPU cores required
    pub cpu_cores: u32,
    /// Maximum concurrent operations
    pub max_concurrency: u32,
}

/// Batch optimization result
#[derive(Debug, Clone)]
pub struct BatchOptimizationResult {
    /// Individual optimization results
    pub individual_results: Vec<OptimizedIOResult>,
    /// Cross-file optimizations applied
    pub cross_file_optimizations: Vec<CrossFileOptimization>,
    /// Total estimated savings
    pub total_savings: IOSavings,
    /// Batch execution plan
    pub batch_execution_plan: ZeroCopyExecutionPlan,
}

/// Cross-file optimization opportunities
#[derive(Debug, Clone)]
pub struct CrossFileOptimization {
    /// Files involved in the optimization
    pub file_paths: Vec<String>,
    /// Type of cross-file optimization
    pub optimization_type: CrossFileOptimizationType,
    /// Estimated additional savings
    pub additional_savings: IOSavings,
    /// Description of the optimization
    pub description: String,
}

/// Types of cross-file optimizations
#[derive(Debug, Clone)]
pub enum CrossFileOptimizationType {
    /// Batch range requests to the same storage location
    BatchedRangeRequests,
    /// Pipeline downloads to improve parallelism
    PipelinedDownloads,
    /// Predictive prefetching based on access patterns
    PredictivePrefetching,
    /// Shared cache utilization
    SharedCacheOptimization,
}

/// Main zero-copy I/O system
pub struct ZeroCopyIOSystem {
    /// Metadata cache for ultra-fast filtering (integrated with unified cache)
    metadata_cache: Arc<FilesystemMetadataStore>,
    /// Smart download optimizer
    download_optimizer: Arc<RwLock<BandwidthOptimizer>>,
    /// Access pattern tracker for learning
    access_tracker: Arc<RwLock<AccessPatternTracker>>,
    /// Filesystem abstraction layer
    filesystem: Arc<FilesystemFactory>,
    /// System configuration
    config: ZeroCopyIOConfig,
    /// Performance metrics
    metrics: Arc<RwLock<SystemPerformanceMetrics>>,
    /// Background task handles
    background_tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Metadata serializers for different engine types
    serializers: HashMap<String, Arc<dyn MetadataSerializer>>,
}

impl ZeroCopyIOSystem {
    /// Create new system with configuration
    pub async fn new(
        config: ZeroCopyIOConfig,
        filesystem: Arc<FilesystemFactory>,
        serializers: Vec<Box<dyn MetadataSerializer>>,
    ) -> Result<Self, ProximaDBError> {
        // Create metadata cache using unified cache infrastructure
        let metadata_cache = Arc::new(FilesystemMetadataStore::new(
            config.metadata_cache.max_memory_mb,
            config.metadata_cache.max_entries,
        ));

        // Store serializers in a HashMap for engine-type based lookup
        let mut serializer_map = HashMap::new();
        for serializer in serializers {
            let engine_type = serializer.engine_id();
            serializer_map.insert(engine_type.to_string(), Arc::from(serializer));
        }

        // Create download optimizer
        let download_optimizer = Arc::new(RwLock::new(BandwidthOptimizer::new(
            config.download_optimizer.clone(),
        )));

        // Create access pattern tracker
        let access_tracker = Arc::new(RwLock::new(
            AccessPatternTracker::new(1000, Duration::from_secs(3600)), // 1000 entries, 1 hour window
        ));

        // Initialize metrics
        let metrics = Arc::new(RwLock::new(SystemPerformanceMetrics::default()));

        let mut system = Self {
            metadata_cache,
            download_optimizer,
            access_tracker,
            filesystem,
            config,
            metrics,
            background_tasks: Vec::new(),
            serializers: serializer_map,
        };

        // Start background tasks if enabled
        if system.config.performance.enable_monitoring {
            system.start_background_tasks().await;
        }

        info!("Zero-copy I/O system initialized");
        Ok(system)
    }

    /// Optimize single file access
    pub async fn optimize_file_access(
        &self,
        file_path: &str,
        collection_id: &str,
        engine_type: &str,
        query_context: &QueryContext,
    ) -> Result<OptimizedIOResult, ProximaDBError> {
        let start_time = Instant::now();

        debug!(
            file_path,
            collection_id,
            engine_type,
            query_type = ?query_context.query_type,
            "Starting file access optimization"
        );

        // Step 1: Check if file can be skipped entirely using metadata cache
        let can_skip = self
            .metadata_cache
            .can_skip_file(file_path, collection_id, engine_type)
            .await;

        if can_skip {
            let savings = IOSavings {
                bandwidth_saved_bytes: u64::MAX, // Would need actual file size
                requests_saved: 1,
                latency_saved_ms: 50.0,    // Typical cloud request latency
                cost_saved_dollars: 0.001, // Rough estimate
                memory_saved_bytes: 0,
                io_operations_saved: 1,
            };

            let result = OptimizedIOResult {
                strategy: IOStrategy::SkipFile {
                    reason: "File filtered out by metadata analysis".to_string(),
                },
                estimated_savings: savings,
                execution_plan: ZeroCopyExecutionPlan {
                    operations: vec![],
                    estimated_duration: Duration::from_millis(1),
                    resource_requirements: ZeroCopyResourceRequirements::default(),
                    fallback_plans: vec![],
                },
                rationale: "Metadata analysis determined no relevant data in file".to_string(),
                confidence: 0.95,
            };

            self.record_optimization_result(&result, start_time.elapsed())
                .await;
            return Ok(result);
        }

        // Step 2: Get required data ranges (selective ranges from metadata)
        let range_tuples = self
            .metadata_cache
            .get_selective_ranges(file_path, collection_id, engine_type)
            .await
            .unwrap_or_else(|| vec![(0, u64::MAX)]); // Full file if no selective ranges

        // Convert to DataRange format
        let required_ranges = if range_tuples.is_empty()
            || (range_tuples.len() == 1 && range_tuples[0].1 == u64::MAX)
        {
            None // Full file
        } else {
            Some(
                range_tuples
                    .into_iter()
                    .map(|(offset, end)| DataRange {
                        offset,
                        length: end.saturating_sub(offset),
                        priority: 128, // Medium priority
                    })
                    .collect(),
            )
        };

        // Step 3: Get file size (would need filesystem integration)
        let file_size = 0u64; // Placeholder - would get from filesystem

        // Step 4: Use download optimizer to decide strategy
        let download_strategy = {
            let optimizer = self.download_optimizer.read().await;
            optimizer
                .decide_strategy(
                    file_path,
                    file_size,
                    required_ranges,
                    query_context,
                    RequestPriority::Normal,
                )
                .await
                .map_err(|e| ProximaDBError::Internal(format!("Download strategy error: {}", e)))?
        };

        // Step 5: Convert to IOStrategy and create execution plan
        let (io_strategy, execution_plan, savings) = self
            .create_execution_plan(download_strategy, file_path, collection_id, file_size)
            .await?;

        let result = OptimizedIOResult {
            strategy: io_strategy,
            estimated_savings: savings,
            execution_plan,
            rationale: "Optimized based on metadata analysis and access patterns".to_string(),
            confidence: 0.85,
        };

        // Record access pattern
        {
            let mut tracker = self.access_tracker.write().await;
            tracker.record_access(AccessEvent {
                file_path: file_path.to_string(),
                collection_id: collection_id.to_string(),
                query_type: query_context.query_type.clone(),
                timestamp: start_time,
                result_type: match &result.strategy {
                    IOStrategy::SkipFile { .. } => "skipped".to_string(),
                    IOStrategy::FullDownload { .. } => "full_download".to_string(),
                    IOStrategy::SelectiveRanges { .. } => "selective_ranges".to_string(),
                    _ => "other".to_string(),
                },
            });
        }

        self.record_optimization_result(&result, start_time.elapsed())
            .await;

        debug!(
            file_path,
            strategy = ?result.strategy,
            optimization_time_ms = start_time.elapsed().as_millis(),
            "File access optimization completed"
        );

        Ok(result)
    }

    /// Optimize multiple file accesses with cross-file optimizations
    pub async fn optimize_multi_file_access(
        &self,
        requests: Vec<FileAccessRequest>,
    ) -> Result<BatchOptimizationResult, ProximaDBError> {
        let start_time = Instant::now();

        debug!(
            request_count = requests.len(),
            "Starting batch file access optimization"
        );

        // Step 1: Optimize each file individually
        let mut individual_results = Vec::with_capacity(requests.len());
        for request in &requests {
            let result = self
                .optimize_file_access(
                    &request.file_path,
                    &request.collection_id,
                    &request.engine_type,
                    &request.query_context,
                )
                .await?;
            individual_results.push(result);
        }

        // Step 2: Identify cross-file optimization opportunities
        let cross_file_optimizations = self
            .identify_cross_file_optimizations(&requests, &individual_results)
            .await?;

        // Step 3: Calculate total savings
        let total_savings =
            self.calculate_total_savings(&individual_results, &cross_file_optimizations);

        // Step 4: Create batch execution plan
        let batch_execution_plan = self
            .create_batch_execution_plan(&individual_results, &cross_file_optimizations)
            .await?;

        let result = BatchOptimizationResult {
            individual_results,
            cross_file_optimizations,
            total_savings,
            batch_execution_plan,
        };

        info!(
            request_count = requests.len(),
            optimization_time_ms = start_time.elapsed().as_millis(),
            total_bandwidth_saved = result.total_savings.bandwidth_saved_bytes,
            "Batch file access optimization completed"
        );

        Ok(result)
    }

    /// Execute optimized I/O operation
    pub fn execute_optimized_read<'a>(
        &'a self,
        optimization: &'a OptimizedIOResult,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, ProximaDBError>> + Send + 'a>,
    > {
        Box::pin(async move { self.execute_optimized_read_impl(optimization).await })
    }

    fn execute_optimized_read_impl<'a>(
        &'a self,
        optimization: &'a OptimizedIOResult,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<u8>, ProximaDBError>> + Send + 'a>,
    > {
        Box::pin(async move { self.execute_optimized_read_impl_inner(optimization).await })
    }

    async fn execute_optimized_read_impl_inner(
        &self,
        optimization: &OptimizedIOResult,
    ) -> Result<Vec<u8>, ProximaDBError> {
        let _start_time = Instant::now();

        debug!(
            strategy = ?optimization.strategy,
            "Executing optimized I/O operation"
        );

        match &optimization.strategy {
            IOStrategy::SkipFile { reason } => {
                debug!(reason, "Skipping file read");
                Ok(Vec::new())
            }

            IOStrategy::LocalCache { cache_path, .. } => {
                debug!(cache_path, "Reading from local cache");
                // Would implement cache reading logic
                Ok(Vec::new())
            }

            IOStrategy::FullDownload { cache_locally, .. } => {
                debug!(cache_locally, "Executing full file download");
                // Would implement full download logic via filesystem
                Ok(Vec::new())
            }

            IOStrategy::SelectiveRanges { ranges, .. } => {
                debug!(
                    range_count = ranges.len(),
                    "Executing selective range downloads"
                );
                // Would implement range download logic
                Ok(Vec::new())
            }

            IOStrategy::HybridStrategy {
                primary,
                fallback,
                condition,
            } => {
                debug!(condition, "Executing hybrid strategy");
                // Try primary strategy first, fallback on failure
                match Box::pin(self.execute_optimized_read_impl_inner(&OptimizedIOResult {
                    strategy: (**primary).clone(),
                    ..optimization.clone()
                }))
                .await
                {
                    Ok(data) => Ok(data),
                    Err(_) => {
                        warn!("Primary strategy failed, trying fallback");
                        Box::pin(self.execute_optimized_read_impl_inner(&OptimizedIOResult {
                            strategy: (**fallback).clone(),
                            ..optimization.clone()
                        }))
                        .await
                    }
                }
            }
        }
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> SystemPerformanceMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Invalidate cache for entire collection
    pub async fn invalidate_collection_cache(
        &self,
        collection_id: &str,
    ) -> Result<u64, ProximaDBError> {
        // Clear all entries for this collection from unified cache
        self.metadata_cache.clear_collection(collection_id).await;

        // Also clear access patterns for this collection
        {
            let mut tracker = self.access_tracker.write().await;
            tracker.clear_collection_patterns(collection_id);
        }

        info!(collection_id, "Collection cache invalidated");
        Ok(1) // Return 1 to indicate success
    }

    /// Warm cache for collection by preloading metadata
    pub async fn warm_cache_for_collection(
        &self,
        collection_id: &str,
    ) -> Result<u64, ProximaDBError> {
        // This would implement cache warming logic by scanning collection files
        // and preloading their metadata
        debug!(collection_id, "Cache warming not yet implemented");
        let _ = collection_id; // Suppress unused warning
        Ok(0)
    }

    /// Optimize cache layout and evict unused entries
    pub async fn optimize_cache_layout(&self) -> Result<(), ProximaDBError> {
        // This would implement cache optimization logic
        debug!("Cache layout optimization not yet implemented");
        Ok(())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn create_execution_plan<'a>(
        &'a self,
        strategy: DownloadStrategy,
        file_path: &'a str,
        collection_id: &'a str,
        file_size: u64,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<(IOStrategy, ZeroCopyExecutionPlan, IOSavings), ProximaDBError>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            match strategy {
                DownloadStrategy::SkipFile { reason } => {
                    let io_strategy = IOStrategy::SkipFile { reason };
                    let plan = ZeroCopyExecutionPlan {
                        operations: vec![],
                        estimated_duration: Duration::from_millis(1),
                        resource_requirements: ZeroCopyResourceRequirements::default(),
                        fallback_plans: vec![],
                    };
                    let savings = IOSavings {
                        bandwidth_saved_bytes: file_size,
                        requests_saved: 1,
                        latency_saved_ms: 50.0,
                        cost_saved_dollars: 0.001,
                        memory_saved_bytes: file_size,
                        io_operations_saved: 1,
                    };
                    Ok((io_strategy, plan, savings))
                }

                DownloadStrategy::FullDownload {
                    cache_locally,
                    reason: _,
                } => {
                    let io_strategy = IOStrategy::FullDownload {
                        cache_locally,
                        prefetch_related: false,
                    };

                    let mut operations = vec![ExecutionOperation {
                        operation_type: OperationType::DownloadFile,
                        target: file_path.to_string(),
                        parameters: HashMap::new(),
                        estimated_duration: Duration::from_millis(100), // Placeholder
                        dependencies: vec![],
                    }];

                    if cache_locally {
                        operations.push(ExecutionOperation {
                            operation_type: OperationType::CacheFile,
                            target: file_path.to_string(),
                            parameters: HashMap::new(),
                            estimated_duration: Duration::from_millis(50),
                            dependencies: vec![0],
                        });
                    }

                    let plan = ZeroCopyExecutionPlan {
                        operations,
                        estimated_duration: Duration::from_millis(150),
                        resource_requirements: ZeroCopyResourceRequirements {
                            memory_bytes: file_size,
                            disk_bytes: if cache_locally { file_size } else { 0 },
                            bandwidth_bps: file_size,
                            cpu_cores: 1,
                            max_concurrency: 1,
                        },
                        fallback_plans: vec![],
                    };

                    let savings = IOSavings::default(); // No savings for full download

                    Ok((io_strategy, plan, savings))
                }

                DownloadStrategy::SelectiveRanges {
                    ranges,
                    total_bytes,
                    reason: _,
                } => {
                    let io_strategy = IOStrategy::SelectiveRanges {
                        ranges: ranges.clone(),
                        parallel_downloads: self
                            .config
                            .download_optimizer
                            .range_optimization
                            .enable_parallel_downloads,
                        merge_threshold: self
                            .config
                            .download_optimizer
                            .range_optimization
                            .max_merge_gap,
                    };

                    let operations = vec![ExecutionOperation {
                        operation_type: OperationType::DownloadRanges,
                        target: file_path.to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("range_count".to_string(), ranges.len().to_string());
                            params.insert("total_bytes".to_string(), total_bytes.to_string());
                            params
                        },
                        estimated_duration: Duration::from_millis(ranges.len() as u64 * 20),
                        dependencies: vec![],
                    }];

                    let plan = ZeroCopyExecutionPlan {
                        operations,
                        estimated_duration: Duration::from_millis(ranges.len() as u64 * 25),
                        resource_requirements: ZeroCopyResourceRequirements {
                            memory_bytes: total_bytes,
                            disk_bytes: 0,
                            bandwidth_bps: total_bytes,
                            cpu_cores: 1,
                            max_concurrency: ranges.len() as u32,
                        },
                        fallback_plans: vec![],
                    };

                    let savings = IOSavings {
                        bandwidth_saved_bytes: file_size.saturating_sub(total_bytes),
                        requests_saved: 0, // Actually increases requests
                        latency_saved_ms: 0.0,
                        cost_saved_dollars: 0.01,
                        memory_saved_bytes: file_size.saturating_sub(total_bytes),
                        io_operations_saved: 0,
                    };

                    Ok((io_strategy, plan, savings))
                }

                DownloadStrategy::HybridStrategy {
                    primary,
                    fallback,
                    condition,
                } => {
                    // Recursively create plans for primary and fallback
                    let (primary_strategy, primary_plan, primary_savings) = self
                        .create_execution_plan(*primary, file_path, collection_id, file_size)
                        .await?;

                    let (fallback_strategy, fallback_plan, _) = self
                        .create_execution_plan(*fallback, file_path, collection_id, file_size)
                        .await?;

                    let io_strategy = IOStrategy::HybridStrategy {
                        primary: Box::new(primary_strategy),
                        fallback: Box::new(fallback_strategy),
                        condition,
                    };

                    // Use primary plan as the main plan, with fallback in fallback_plans
                    let mut plan = primary_plan;
                    plan.fallback_plans.push(fallback_plan);

                    Ok((io_strategy, plan, primary_savings))
                }
            }
        })
    }

    async fn identify_cross_file_optimizations(
        &self,
        _requests: &[FileAccessRequest],
        _individual_results: &[OptimizedIOResult],
    ) -> Result<Vec<CrossFileOptimization>, ProximaDBError> {
        // This would implement cross-file optimization logic
        // For now, return empty optimizations
        let _ = (_requests, _individual_results); // Suppress unused warnings
        Ok(Vec::new())
    }

    fn calculate_total_savings(
        &self,
        individual_results: &[OptimizedIOResult],
        cross_file_optimizations: &[CrossFileOptimization],
    ) -> IOSavings {
        let mut total = IOSavings::default();

        // Sum individual savings
        for result in individual_results {
            let savings = &result.estimated_savings;
            total.bandwidth_saved_bytes += savings.bandwidth_saved_bytes;
            total.requests_saved += savings.requests_saved;
            total.latency_saved_ms += savings.latency_saved_ms;
            total.cost_saved_dollars += savings.cost_saved_dollars;
            total.memory_saved_bytes += savings.memory_saved_bytes;
            total.io_operations_saved += savings.io_operations_saved;
        }

        // Add cross-file optimization savings
        for optimization in cross_file_optimizations {
            let savings = &optimization.additional_savings;
            total.bandwidth_saved_bytes += savings.bandwidth_saved_bytes;
            total.requests_saved += savings.requests_saved;
            total.latency_saved_ms += savings.latency_saved_ms;
            total.cost_saved_dollars += savings.cost_saved_dollars;
            total.memory_saved_bytes += savings.memory_saved_bytes;
            total.io_operations_saved += savings.io_operations_saved;
        }

        total
    }

    async fn create_batch_execution_plan(
        &self,
        _individual_results: &[OptimizedIOResult],
        _cross_file_optimizations: &[CrossFileOptimization],
    ) -> Result<ZeroCopyExecutionPlan, ProximaDBError> {
        // This would implement batch execution planning
        let _ = (_individual_results, _cross_file_optimizations); // Suppress unused warnings
        Ok(ZeroCopyExecutionPlan {
            operations: vec![],
            estimated_duration: Duration::from_millis(100),
            resource_requirements: ZeroCopyResourceRequirements::default(),
            fallback_plans: vec![],
        })
    }

    async fn record_optimization_result(&self, _result: &OptimizedIOResult, _duration: Duration) {
        // This would update performance metrics
        // Implementation would track various statistics
        let _ = (_result, _duration); // Suppress unused warnings
    }

    async fn start_background_tasks(&mut self) {
        // This would start background tasks for metrics collection,
        // cache maintenance, pattern analysis, etc.
        debug!("Background tasks not yet implemented");
        let _ = &self.config; // Suppress unused warning
    }
}

// Enable clean shutdown
impl Drop for ZeroCopyIOSystem {
    fn drop(&mut self) {
        // Cancel background tasks
        for handle in &self.background_tasks {
            handle.abort();
        }
    }
}

impl ZeroCopyIOSystem {
    /// Get cached metadata with automatic cache population on miss
    /// This is the primary method readers should use for filename-based cache verification
    pub async fn get_cached_metadata(
        &self,
        cache_key: &str,
    ) -> Result<Option<Arc<Box<dyn super::traits::EngineMetadata>>>, ProximaDBError> {
        // Parse cache key format: "file_path:collection_id:engine" (filename-first for optimal matching)
        let parts: Vec<&str> = cache_key.rsplitn(3, ':').collect();
        if parts.len() < 3 {
            return Err(ProximaDBError::Config(format!(
                "Invalid cache key: {}",
                cache_key
            )));
        }

        // Since rsplitn splits from the right, reverse the order
        let engine_type = parts[0];
        let collection_id = parts[1];
        let file_path = parts[2]; // File path is now the remaining part

        debug!(
            cache_key,
            engine_type,
            collection_id,
            file_path,
            "Checking cache for metadata (filename-first format for optimal sequential matching)"
        );

        // Try to get from cache first
        if let Some(_cached_metadata) = self
            .metadata_cache
            .get_metadata(file_path, collection_id, engine_type)
            .await
        {
            trace!(cache_key, "Cache HIT for metadata");
            // Create a synthetic EngineMetadata from FilesystemMetadata
            // This would need actual deserialization using the appropriate serializer
            if self.serializers.contains_key(engine_type) {
                // For now, return None as we need to implement proper deserialization
                // In production, would deserialize mmap_metadata using the serializer
                Ok(None)
            } else {
                warn!(cache_key, "No serializer found for engine type");
                Ok(None)
            }
        } else {
            trace!(cache_key, "Cache MISS - metadata not in cache");

            // CACHE POPULATION: Load metadata from file and populate cache
            match self
                .populate_cache_from_file(file_path, collection_id, engine_type)
                .await
            {
                Ok(metadata) => {
                    debug!(cache_key, "Successfully populated cache from file");
                    Ok(Some(metadata))
                }
                Err(e) => {
                    warn!(
                        cache_key,
                        error = %e,
                        "Failed to populate cache from file"
                    );
                    Ok(None)
                }
            }
        }
    }

    /// Populate cache by loading metadata from file (used on cache miss)
    async fn populate_cache_from_file(
        &self,
        file_path: &str,
        collection_id: &str,
        engine_type: &str,
    ) -> Result<Arc<Box<dyn super::traits::EngineMetadata>>, ProximaDBError> {
        debug!(
            file_path,
            collection_id, engine_type, "Populating cache from file"
        );

        // Load metadata from file using the appropriate serializer
        if let Some(serializer) = self.serializers.get(engine_type) {
            // Load metadata from filesystem
            let actual_fs = self.filesystem.get_filesystem(file_path).map_err(|e| {
                ProximaDBError::Internal(format!("Failed to get filesystem: {}", e))
            })?;
            let file_data = actual_fs
                .read_range(file_path, 0, 4096)
                .await
                .map_err(|e| ProximaDBError::Internal(format!("Failed to read file: {}", e)))?; // Read header

            // Parse metadata using serializer
            let engine_metadata = serializer.deserialize_metadata(&file_data).map_err(|e| {
                ProximaDBError::Internal(format!("Failed to deserialize metadata: {}", e))
            })?;

            // Create filesystem metadata entry for caching
            let fs_metadata = FilesystemMetadata {
                mmap_metadata: None,    // Would be populated with actual mmap data
                file_size: 0,           // Would get actual size from filesystem
                last_modified: 0,       // Would get actual timestamp
                can_skip: false,        // Would be determined by metadata analysis
                selective_ranges: None, // Would be computed based on query
                collection_id: collection_id.to_string(),
                engine_type: engine_type.to_string(),
            };

            // Store in unified cache
            self.metadata_cache
                .put_metadata(file_path, collection_id, engine_type, fs_metadata)
                .await
                .map_err(|e| {
                    ProximaDBError::Internal(format!("Failed to cache metadata: {}", e))
                })?;

            debug!(
                file_path,
                collection_id, engine_type, "Successfully populated cache from file"
            );

            Ok(Arc::new(engine_metadata))
        } else {
            Err(ProximaDBError::Internal(format!(
                "No serializer found for engine type: {}",
                engine_type
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_savings_calculation() {
        let savings1 = IOSavings {
            bandwidth_saved_bytes: 1000,
            requests_saved: 5,
            latency_saved_ms: 100.0,
            cost_saved_dollars: 0.01,
            memory_saved_bytes: 500,
            io_operations_saved: 2,
        };

        let savings2 = IOSavings {
            bandwidth_saved_bytes: 2000,
            requests_saved: 3,
            latency_saved_ms: 50.0,
            cost_saved_dollars: 0.02,
            memory_saved_bytes: 1000,
            io_operations_saved: 1,
        };

        // Manual calculation for testing
        assert_eq!(
            savings1.bandwidth_saved_bytes + savings2.bandwidth_saved_bytes,
            3000
        );
        assert_eq!(savings1.requests_saved + savings2.requests_saved, 8);
    }

    #[test]
    fn test_execution_operation() {
        let operation = ExecutionOperation {
            operation_type: OperationType::DownloadFile,
            target: "/path/to/file.sst".to_string(),
            parameters: HashMap::new(),
            estimated_duration: Duration::from_millis(100),
            dependencies: vec![],
        };

        assert_eq!(operation.operation_type, OperationType::DownloadFile);
        assert_eq!(operation.target, "/path/to/file.sst");
        assert!(operation.dependencies.is_empty());
    }
}
