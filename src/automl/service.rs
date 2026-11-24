// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Core AutoML Service Implementation

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::metrics::{CollectionMetrics, UnifiedMetricsCollector};
// AutoMLConfig moved to proximadb-automl crate (TD-DECOMP-15); re-export for existing callers.
pub use proximadb_automl::AutoMLConfig;

/// Optimization request
#[derive(Debug, Clone)]
pub struct OptimizationRequest {
    /// Identifier of the collection to optimize
    pub collection_id: String,
    /// Category of optimization to apply
    pub optimization_type: OptimizationType,
    /// How urgently the optimization should be scheduled
    pub urgency: OptimizationUrgency,
    /// Snapshot of current performance and resource state used to guide optimization
    pub context: OptimizationContext,
}

/// Types of optimization
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Choose the most appropriate ANN index algorithm
    IndexSelection,
    /// Choose the best quantization scheme for memory/accuracy trade-off
    QuantizationLevel,
    /// Switch to a storage engine better suited to the workload
    EngineSelection,
    /// Tune cache size and eviction policy
    CacheConfiguration,
    /// Tune algorithm-specific parameters (e.g. HNSW `M`, `ef_search`)
    HyperparameterTuning,
    /// Re-configure the system to match an observed workload shift
    WorkloadAdaptation,
}

/// Optimization urgency levels
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationUrgency {
    /// Performance degradation detected; immediate action required
    Critical,
    /// Significant improvement opportunity identified
    High,
    /// Part of the regular periodic optimization cycle
    Normal,
    /// Minor improvements possible; low priority
    Low,
}

/// Optimization context with relevant metrics
#[derive(Debug, Clone)]
pub struct OptimizationContext {
    /// Current query performance numbers at the time the request was created
    pub current_performance: AutomlPerformanceMetrics,
    /// Characterisation of the workload mix seen by the collection
    pub workload_characteristics: WorkloadCharacteristics,
    /// System resource utilisation at the time the request was created
    pub resource_usage: AutomlResourceUsage,
}

/// Backwards-compat alias for [`AutomlPerformanceMetrics`].
pub type PerformanceMetrics = AutomlPerformanceMetrics;

/// Performance metrics
#[derive(Debug, Clone)]
pub struct AutomlPerformanceMetrics {
    /// Median (P50) query end-to-end latency in milliseconds
    pub query_latency_p50: f64,
    /// 99th-percentile query end-to-end latency in milliseconds
    pub query_latency_p99: f64,
    /// Observed query throughput in queries per second
    pub throughput_qps: f64,
    /// Fraction of queries that completed without error (0.0–1.0)
    pub success_rate: f64,
}

/// Workload characteristics
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Ratio of read operations to write operations (e.g. 5.0 = five reads per write)
    pub read_write_ratio: f64,
    /// Typical complexity level of queries observed on this collection
    pub query_complexity: QueryComplexity,
    /// Dominant data-access pattern observed on this collection
    pub access_pattern: AccessPattern,
    /// Rate at which new vectors are being added (vectors per second)
    pub data_growth_rate: f64,
}

/// Query complexity levels
#[derive(Debug, Clone, PartialEq)]
pub enum QueryComplexity {
    /// Simple point (k-NN) queries with no additional filters
    Simple,
    /// Range or pre-filter queries that narrow the candidate set before ANN search
    Medium,
    /// Multi-index or join-style queries combining vector and metadata predicates
    Complex,
}

/// Data access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    /// No temporal or spatial locality; accesses are uniformly distributed
    Random,
    /// Accesses follow a sequential scan order through the collection
    Sequential,
    /// Recently inserted vectors are accessed more frequently than older ones
    Temporal,
    /// A small subset of vectors accounts for the majority of accesses
    Hotspot,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct AutomlResourceUsage {
    /// CPU utilisation as a percentage (0.0–100.0)
    pub cpu_usage_percent: f64,
    /// Current resident memory consumption in MB
    pub memory_usage_mb: u64,
    /// Disk I/O throughput in MB/s
    pub disk_io_mb_per_sec: f64,
    /// Network I/O throughput in MB/s
    pub network_io_mb_per_sec: f64,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Unique identifier for this optimization run
    pub request_id: String,
    /// `true` if the optimization applied at least one improvement
    pub success: bool,
    /// List of individual metric improvements achieved by this run
    pub improvements: Vec<Improvement>,
    /// Wall-clock time (milliseconds) taken to complete the optimization
    pub execution_time_ms: u64,
    /// Human-readable error description if `success` is `false`
    pub error: Option<String>,
}

/// Performance improvement details
#[derive(Debug, Clone)]
pub struct Improvement {
    /// Name of the metric that was improved (e.g. `"query_latency"`, `"memory_usage"`)
    pub metric: String,
    /// Metric value before the optimization was applied
    pub before: f64,
    /// Metric value after the optimization was applied
    pub after: f64,
    /// Relative improvement as a percentage: `(before - after) / before * 100`
    pub improvement_percent: f64,
}

/// AutoML Service
pub struct AutoMLService {
    #[allow(dead_code)]
    config: AutoMLConfig,
    #[allow(dead_code)]
    metrics_collector: Arc<UnifiedMetricsCollector>,
    optimization_queue: Arc<RwLock<Vec<OptimizationRequest>>>,
    active_optimizations: Arc<RwLock<HashMap<String, OptimizationRequest>>>,
    optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,
    optimization_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl AutoMLService {
    /// Create a new AutoML service
    pub async fn new(config: AutoMLConfig) -> Result<Self> {
        let metrics_collector = Arc::new(UnifiedMetricsCollector::new());

        Ok(Self {
            config,
            metrics_collector,
            optimization_queue: Arc::new(RwLock::new(Vec::new())),
            active_optimizations: Arc::new(RwLock::new(HashMap::new())),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            shutdown_tx: Arc::new(RwLock::new(None)),
            optimization_handle: Arc::new(RwLock::new(None)),
        })
    }

    /// Start the AutoML service
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("AutoML service is disabled");
            return Ok(());
        }

        info!("Starting AutoML service");

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        let service = self.clone();
        let handle = tokio::spawn(async move {
            let mut optimization_interval = interval(Duration::from_secs(
                service.config.optimization_interval_secs,
            ));

            loop {
                tokio::select! {
                    _ = optimization_interval.tick() => {
                        if let Err(e) = service.run_optimization_cycle().await {
                            warn!("Optimization cycle failed: {}", e);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("AutoML service shutting down");
                        break;
                    }
                }
            }
        });

        // Store shutdown channel and handle
        *self.shutdown_tx.write().await = Some(shutdown_tx);
        *self.optimization_handle.write().await = Some(handle);

        Ok(())
    }

    /// Stop the AutoML service
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping AutoML service");

        // Send shutdown signal
        if let Some(tx) = &*self.shutdown_tx.read().await {
            let _ = tx.send(()).await;
        }

        // Wait for optimization task to complete with timeout
        if let Some(handle) = self.optimization_handle.write().await.take() {
            match tokio::time::timeout(Duration::from_secs(30), handle).await {
                Ok(Ok(())) => {
                    info!("AutoML service stopped successfully");
                }
                Ok(Err(e)) => {
                    warn!("AutoML service task panicked: {:?}", e);
                }
                Err(_) => {
                    warn!("AutoML service shutdown timed out after 30s");
                }
            }
        }

        // Clear shutdown channel
        *self.shutdown_tx.write().await = None;

        Ok(())
    }

    /// Run an optimization cycle
    async fn run_optimization_cycle(&self) -> Result<()> {
        debug!("Running optimization cycle");

        // Collect current metrics
        let collections = self.identify_optimization_candidates().await?;

        // Queue optimization requests
        let mut queue = self.optimization_queue.write().await;
        for (collection_id, urgency) in collections {
            let request = self
                .create_optimization_request(collection_id, urgency)
                .await?;
            queue.push(request);
        }

        // Process optimization queue
        self.process_optimization_queue().await?;

        Ok(())
    }

    /// Identify collections that need optimization by querying the metrics collector
    async fn identify_optimization_candidates(&self) -> Result<Vec<(String, OptimizationUrgency)>> {
        // Query system-level metrics snapshot
        let system_metrics = self.metrics_collector.current_metrics().await;

        // The unified metrics collector currently exposes only system-level query metrics.
        // Until per-collection snapshots are restored, avoid fabricating candidates here.
        if system_metrics.query.total_queries == 0 {
            return Ok(Vec::new());
        }

        Ok(Vec::new())
    }

    /// Evaluate optimization urgency based on metrics
    #[allow(dead_code)]
    async fn evaluate_optimization_urgency(
        &self,
        metrics: &CollectionMetrics,
    ) -> Result<OptimizationUrgency> {
        // Use real metrics from the collection: average search latency in ms
        let avg_latency = if metrics.total_searches > 0 {
            metrics.avg_search_latency_us / 1000.0 // Convert us to ms
        } else {
            0.0 // No queries yet, low urgency
        };

        if avg_latency > 1000.0 {
            Ok(OptimizationUrgency::Critical)
        } else if avg_latency > 500.0 {
            Ok(OptimizationUrgency::High)
        } else if avg_latency > 200.0 {
            Ok(OptimizationUrgency::Normal)
        } else {
            Ok(OptimizationUrgency::Low)
        }
    }

    /// Create an optimization request for a collection using live metrics
    async fn create_optimization_request(
        &self,
        collection_id: String,
        urgency: OptimizationUrgency,
    ) -> Result<OptimizationRequest> {
        // Collect current performance metrics from the metrics collector
        let system_metrics = self.metrics_collector.current_metrics().await;

        let latency_p99 = if system_metrics.query.p99_latency_ms > 0.0 {
            system_metrics.query.p99_latency_ms
        } else {
            150.0
        };
        let latency_p50 = if latency_p99 > 0.0 {
            latency_p99 / 3.0
        } else {
            50.0
        };
        let throughput = if system_metrics.query.total_queries > 0 {
            system_metrics.query.total_queries as f64
        } else {
            1000.0
        };
        let success_rate = if system_metrics.query.total_queries > 0 {
            let failures = system_metrics
                .query
                .failed_queries
                .min(system_metrics.query.total_queries);
            1.0 - (failures as f64 / system_metrics.query.total_queries as f64)
        } else {
            0.99
        };

        let performance = AutomlPerformanceMetrics {
            query_latency_p50: latency_p50,
            query_latency_p99: latency_p99,
            throughput_qps: throughput,
            success_rate,
        };

        let workload = WorkloadCharacteristics {
            read_write_ratio: 5.0,
            query_complexity: QueryComplexity::Medium,
            access_pattern: AccessPattern::Random,
            data_growth_rate: 100.0,
        };

        let resources = AutomlResourceUsage {
            cpu_usage_percent: system_metrics.cpu_usage as f64,
            memory_usage_mb: system_metrics.memory_used_bytes / (1024 * 1024),
            disk_io_mb_per_sec: 0.0,
            network_io_mb_per_sec: 0.0,
        };

        let context = OptimizationContext {
            current_performance: performance,
            workload_characteristics: workload,
            resource_usage: resources,
        };

        // Determine optimization type based on urgency and context
        let optimization_type = self.select_optimization_type(&context, &urgency).await?;

        Ok(OptimizationRequest {
            collection_id,
            optimization_type,
            urgency,
            context,
        })
    }

    /// Select the most appropriate optimization type
    async fn select_optimization_type(
        &self,
        _context: &OptimizationContext,
        urgency: &OptimizationUrgency,
    ) -> Result<OptimizationType> {
        // Simple heuristic for now
        match urgency {
            OptimizationUrgency::Critical => Ok(OptimizationType::IndexSelection),
            OptimizationUrgency::High => Ok(OptimizationType::QuantizationLevel),
            OptimizationUrgency::Normal => Ok(OptimizationType::HyperparameterTuning),
            OptimizationUrgency::Low => Ok(OptimizationType::CacheConfiguration),
        }
    }

    /// Process the optimization queue
    async fn process_optimization_queue(&self) -> Result<()> {
        let mut queue = self.optimization_queue.write().await;
        let active = self.active_optimizations.read().await;

        // Process up to max_concurrent_optimizations
        while !queue.is_empty() && active.len() < self.config.max_concurrent_optimizations {
            if let Some(request) = queue.pop() {
                // Spawn optimization task
                let service = self.clone();
                let request_clone = request.clone();

                tokio::spawn(async move {
                    if let Err(e) = service.execute_optimization(request_clone).await {
                        warn!("Optimization failed: {}", e);
                    }
                });
            }
        }

        Ok(())
    }

    /// Execute an optimization request
    async fn execute_optimization(&self, request: OptimizationRequest) -> Result<()> {
        let start_time = Instant::now();
        let request_id = uuid::Uuid::new_v4().to_string();

        // Add to active optimizations
        {
            let mut active = self.active_optimizations.write().await;
            active.insert(request_id.clone(), request.clone());
        }

        // Execute optimization based on type
        let result = match request.optimization_type {
            OptimizationType::IndexSelection => self.optimize_index_selection(&request).await,
            OptimizationType::QuantizationLevel => self.optimize_quantization(&request).await,
            OptimizationType::EngineSelection => self.optimize_engine_selection(&request).await,
            OptimizationType::CacheConfiguration => self.optimize_cache_config(&request).await,
            OptimizationType::HyperparameterTuning => self.tune_hyperparameters(&request).await,
            OptimizationType::WorkloadAdaptation => self.adapt_to_workload(&request).await,
        };

        // Create optimization result
        let optimization_result = match result {
            Ok(improvements) => OptimizationResult {
                request_id: request_id.clone(),
                success: true,
                improvements,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                error: None,
            },
            Err(e) => OptimizationResult {
                request_id: request_id.clone(),
                success: false,
                improvements: Vec::new(),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                error: Some(e.to_string()),
            },
        };

        // Remove from active and add to history
        {
            let mut active = self.active_optimizations.write().await;
            active.remove(&request_id);
        }

        {
            let mut history = self.optimization_history.write().await;
            history.push(optimization_result);

            // Keep only last 100 results
            if history.len() > 100 {
                let drain_count = history.len() - 100;
                history.drain(0..drain_count);
            }
        }

        Ok(())
    }

    // Optimization implementations

    async fn optimize_index_selection(
        &self,
        request: &OptimizationRequest,
    ) -> Result<Vec<Improvement>> {
        let ctx = &request.context;
        let latency = ctx.current_performance.query_latency_p50;
        let memory_mb = ctx.resource_usage.memory_usage_mb;

        // Decision logic based on workload characteristics:
        // - HNSW for collections < 1M vectors (best recall, moderate memory)
        // - IVF for collections > 1M with high recall needs (partitioned search)
        // - Flat for collections < 10K (brute-force is fast enough)
        let recommendation = if memory_mb < 100 {
            "flat" // Small dataset, brute-force is optimal
        } else if memory_mb < 2048 {
            "hnsw" // Medium dataset, HNSW provides best recall/latency trade-off
        } else {
            "ivf" // Large dataset, IVF partitions reduce search space
        };

        let estimated_improvement = match recommendation {
            "hnsw" if latency > 50.0 => (latency - 10.0).max(5.0) / latency * 100.0,
            "ivf" if latency > 100.0 => (latency - 30.0).max(10.0) / latency * 100.0,
            "flat" if latency > 5.0 => (latency - 2.0).max(1.0) / latency * 100.0,
            _ => 5.0, // Minimal improvement expected
        };

        info!(
            "AutoML index recommendation for {}: {} (estimated {:.1}% latency improvement)",
            request.collection_id, recommendation, estimated_improvement
        );

        Ok(vec![Improvement {
            metric: "query_latency".to_string(),
            before: latency,
            after: latency * (1.0 - estimated_improvement / 100.0),
            improvement_percent: estimated_improvement,
        }])
    }

    async fn optimize_quantization(
        &self,
        request: &OptimizationRequest,
    ) -> Result<Vec<Improvement>> {
        let ctx = &request.context;
        let memory_mb = ctx.resource_usage.memory_usage_mb as f64;

        // Quantization decision based on memory pressure:
        // - FP32 for < 100MB (no quantization needed)
        // - INT8 for 100MB-1GB (8x compression, <5% recall loss)
        // - PQ for > 1GB (32x+ compression, <10% recall loss)
        let (recommendation, compression_ratio) = if memory_mb < 100.0 {
            ("fp32", 1.0)
        } else if memory_mb < 1024.0 {
            ("int8", 4.0) // 4x compression (32-bit -> 8-bit)
        } else {
            ("pq8", 8.0) // 8x compression with product quantization
        };

        let memory_after = memory_mb / compression_ratio;
        let improvement = (memory_mb - memory_after) / memory_mb * 100.0;

        info!(
            "AutoML quantization recommendation for {}: {} ({:.1}% memory reduction)",
            request.collection_id, recommendation, improvement
        );

        Ok(vec![Improvement {
            metric: "memory_usage".to_string(),
            before: memory_mb,
            after: memory_after,
            improvement_percent: improvement,
        }])
    }

    async fn optimize_engine_selection(
        &self,
        request: &OptimizationRequest,
    ) -> Result<Vec<Improvement>> {
        let ctx = &request.context;
        let rw_ratio = ctx.workload_characteristics.read_write_ratio;
        let throughput = ctx.current_performance.throughput_qps;

        // Engine selection based on workload:
        // - SST for write-heavy (rw_ratio < 2.0) - LSM-tree optimized for writes
        // - NOVA for read-heavy analytics (rw_ratio > 10.0) - Parquet columnar
        // - NOVA for mixed with predicate pushdown needs
        // - HELIX for high-dimensional data
        let (recommendation, estimated_throughput_gain) = if rw_ratio < 2.0 {
            ("SST", 1.3) // 30% throughput improvement for write-heavy
        } else if rw_ratio > 10.0 {
            ("NOVA", 1.5) // 50% improvement for analytics
        } else {
            ("NOVA", 1.2) // 20% improvement for mixed
        };

        let after = throughput * estimated_throughput_gain;
        let improvement = (after - throughput) / throughput * 100.0;

        info!(
            "AutoML engine recommendation for {}: {} ({:.1}% throughput improvement)",
            request.collection_id, recommendation, improvement
        );

        Ok(vec![Improvement {
            metric: "throughput".to_string(),
            before: throughput,
            after,
            improvement_percent: improvement,
        }])
    }

    async fn optimize_cache_config(
        &self,
        request: &OptimizationRequest,
    ) -> Result<Vec<Improvement>> {
        let ctx = &request.context;
        let access_pattern = &ctx.workload_characteristics.access_pattern;

        // Cache tuning based on access patterns:
        // - Hotspot: Increase cache size, use LRU
        // - Temporal: Use time-based eviction, warm recent data
        // - Sequential: Use read-ahead prefetching
        // - Random: Standard LRU with moderate cache size
        let (estimated_hit_rate_before, estimated_hit_rate_after) = match access_pattern {
            AccessPattern::Hotspot => (0.6, 0.92), // Hot data stays in cache
            AccessPattern::Temporal => (0.5, 0.85), // Recent data cached well
            AccessPattern::Sequential => (0.4, 0.75), // Prefetch helps
            AccessPattern::Random => (0.3, 0.55),  // Limited improvement
        };

        let improvement = (estimated_hit_rate_after - estimated_hit_rate_before)
            / estimated_hit_rate_before
            * 100.0;

        info!(
            "AutoML cache recommendation for {}: {:?} pattern, {:.1}% hit rate improvement",
            request.collection_id, access_pattern, improvement
        );

        Ok(vec![Improvement {
            metric: "cache_hit_rate".to_string(),
            before: estimated_hit_rate_before,
            after: estimated_hit_rate_after,
            improvement_percent: improvement,
        }])
    }

    async fn tune_hyperparameters(
        &self,
        request: &OptimizationRequest,
    ) -> Result<Vec<Improvement>> {
        let ctx = &request.context;
        let latency_p50 = ctx.current_performance.query_latency_p50;
        let latency_p99 = ctx.current_performance.query_latency_p99;
        let complexity = &ctx.workload_characteristics.query_complexity;

        // Hyperparameter tuning based on latency distribution:
        // - If p99/p50 ratio is high (tail latency), tune HNSW ef_search
        // - If overall latency is high, tune number of probes (IVF) or search depth
        let tail_ratio = if latency_p50 > 0.0 {
            latency_p99 / latency_p50
        } else {
            1.0
        };

        let estimated_improvement = match (complexity, tail_ratio > 5.0) {
            (QueryComplexity::Complex, true) => 25.0, // High tail latency + complex queries
            (QueryComplexity::Complex, false) => 15.0,
            (QueryComplexity::Medium, true) => 20.0,
            (QueryComplexity::Medium, false) => 10.0,
            (QueryComplexity::Simple, _) => 5.0,
        };

        info!(
            "AutoML hyperparameter tuning for {}: p99/p50 ratio={:.1}, estimated {:.1}% improvement",
            request.collection_id, tail_ratio, estimated_improvement
        );

        Ok(vec![Improvement {
            metric: "overall_performance".to_string(),
            before: latency_p50,
            after: latency_p50 * (1.0 - estimated_improvement / 100.0),
            improvement_percent: estimated_improvement,
        }])
    }

    async fn adapt_to_workload(&self, request: &OptimizationRequest) -> Result<Vec<Improvement>> {
        let ctx = &request.context;
        let rw_ratio = ctx.workload_characteristics.read_write_ratio;
        let growth_rate = ctx.workload_characteristics.data_growth_rate;
        let throughput = ctx.current_performance.throughput_qps;

        // Workload adaptation: adjust engine configuration based on evolving patterns
        // - High growth rate: prepare for scaling (increase batch sizes, compaction thresholds)
        // - Shifting read/write balance: re-evaluate engine choice
        let estimated_improvement = if growth_rate > 0.1 {
            12.0 // High growth, proactive adaptation helps
        } else if rw_ratio > 5.0 {
            8.0 // Read-heavy, optimize caches and indexes
        } else {
            5.0 // Stable workload, minor tuning
        };

        let after = throughput * (1.0 + estimated_improvement / 100.0);

        info!(
            "AutoML workload adaptation for {}: growth_rate={:.2}, rw_ratio={:.1}",
            request.collection_id, growth_rate, rw_ratio
        );

        Ok(vec![Improvement {
            metric: "adaptive_performance".to_string(),
            before: throughput,
            after,
            improvement_percent: estimated_improvement,
        }])
    }
}

impl Clone for AutoMLService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_collector: self.metrics_collector.clone(),
            optimization_queue: self.optimization_queue.clone(),
            active_optimizations: self.active_optimizations.clone(),
            optimization_history: self.optimization_history.clone(),
            shutdown_tx: Arc::new(RwLock::new(None)), // Don't clone shutdown channel
            optimization_handle: Arc::new(RwLock::new(None)), // Don't clone handle
        }
    }
}
