//! Metrics and monitoring for HELIX engine
//!
//! Comprehensive metrics for tracking HELIX performance,
//! clustering quality, and optimization opportunities.

use lazy_static::lazy_static;
use prometheus::{
    CounterVec, GaugeVec, HistogramVec, register_counter_vec, register_gauge_vec,
    register_histogram_vec,
};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

fn register_histogram_vec_safe(name: &str, help: &str, labels: &[&str]) -> Option<HistogramVec> {
    match register_histogram_vec!(name, help, labels) {
        Ok(metric) => Some(metric),
        Err(error) => {
            warn!(metric = name, error = %error, "HELIX histogram metric disabled");
            None
        }
    }
}

fn register_counter_vec_safe(name: &str, help: &str, labels: &[&str]) -> Option<CounterVec> {
    // Disable-on-clash semantics preserved via the shared or_none helper.
    proximadb_metrics::register_counter_vec_or_none(name, help, labels)
}

fn register_gauge_vec_safe(name: &str, help: &str, labels: &[&str]) -> Option<GaugeVec> {
    proximadb_metrics::register_gauge_vec_or_none(name, help, labels)
}

lazy_static! {
    /// Compaction metrics
    static ref COMPACTION_DURATION: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_compaction_duration_seconds",
        "Time spent in compaction by level",
        &["level"]
    );

    static ref COMPACTION_BYTES_WRITTEN: Option<CounterVec> = register_counter_vec_safe(
        "proximadb_helix_compaction_bytes_written",
        "Bytes written during compaction",
        &["level"]
    );

    static ref COMPACTION_CLUSTERING_QUALITY: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_compaction_clustering_quality",
        "Clustering quality score after compaction",
        &["level"]
    );

    /// Query metrics
    static ref QUERY_PRUNING_RATIO: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_query_pruning_ratio",
        "Ratio of SSTables pruned during query",
        &["collection"]
    );

    static ref PROXIMA_BLOCKS_SCANNED: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_proximablocks_scanned",
        "Number of Proxima blocks scanned per query",
        &["collection"]
    );

    static ref HILBERT_RANGE_EFFICIENCY: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_hilbert_range_efficiency",
        "Efficiency of Hilbert range pruning",
        &["collection"]
    );

    /// PCA metrics
    static ref PCA_MODEL_VERSION: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_pca_model_version",
        "Current PCA model version",
        &["collection"]
    );

    static ref PCA_PROJECTION_LATENCY: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_pca_projection_latency_us",
        "PCA projection latency in microseconds",
        &["collection"]
    );

    static ref PCA_MODEL_DRIFT_SCORE: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_pca_model_drift_score",
        "PCA model drift from training distribution",
        &["collection"]
    );

    /// Storage metrics
    static ref SSTABLE_COUNT: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_sstable_count",
        "Number of SSTables by level",
        &["level", "collection"]
    );

    static ref SSTABLE_SIZE_BYTES: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_sstable_size_bytes",
        "Total size of SSTables by level",
        &["level", "collection"]
    );

    static ref BLOOM_FILTER_HITS: Option<CounterVec> = register_counter_vec_safe(
        "proximadb_helix_bloom_filter_hits",
        "Bloom filter hit count",
        &["type", "collection"]
    );

    /// Liquid clustering metrics
    static ref LIQUID_CLUSTERING_QUALITY: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_liquid_clustering_quality",
        "Liquid clustering quality score",
        &["collection"]
    );

    static ref HOT_REGIONS_COUNT: Option<GaugeVec> = register_gauge_vec_safe(
        "proximadb_helix_hot_regions_count",
        "Number of hot regions identified",
        &["collection"]
    );

    static ref LIQUID_REORGANIZATIONS: Option<CounterVec> = register_counter_vec_safe(
        "proximadb_helix_liquid_reorganizations_total",
        "Total number of liquid clustering reorganizations",
        &["collection"]
    );

    /// Progressive search metrics
    static ref PROGRESSIVE_SEARCH_STAGES: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_progressive_search_stages",
        "Number of stages used in progressive search",
        &["collection"]
    );

    static ref QUANTIZATION_STAGE_LATENCY: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_quantization_stage_latency_ms",
        "Latency of each quantization stage",
        &["stage", "collection"]
    );

    /// Zone map metrics
    static ref ZONE_MAP_PRUNING_RATIO: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_zone_map_pruning_ratio",
        "Ratio of blocks pruned by zone maps",
        &["collection"]
    );

    static ref DIMENSION_SELECTIVITY: Option<HistogramVec> = register_histogram_vec_safe(
        "proximadb_helix_dimension_selectivity",
        "Selectivity per dimension",
        &["dimension", "collection"]
    );
}

/// HELIX engine metrics aggregator
pub struct HelixMetrics {
    collection: String,
    query_count: AtomicU64,
    compaction_count: AtomicU64,
    flush_count: AtomicU64,
}

impl HelixMetrics {
    pub fn new(collection: String) -> Self {
        Self {
            collection,
            query_count: AtomicU64::new(0),
            compaction_count: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
        }
    }

    /// Record compaction metrics
    pub fn record_compaction(
        &self,
        level: usize,
        duration_secs: f64,
        bytes_written: u64,
        clustering_quality: f32,
    ) {
        if let Some(metric) = COMPACTION_DURATION.as_ref() {
            metric
                .with_label_values(&[&level.to_string()])
                .observe(duration_secs);
        }

        if let Some(metric) = COMPACTION_BYTES_WRITTEN.as_ref() {
            metric
                .with_label_values(&[&level.to_string()])
                .inc_by(bytes_written);
        }

        if let Some(metric) = COMPACTION_CLUSTERING_QUALITY.as_ref() {
            metric
                .with_label_values(&[&level.to_string()])
                .set(clustering_quality as f64);
        }
    }

    /// Record query metrics
    pub fn record_query(&self, pruning_ratio: f32, blocks_scanned: usize, stages_used: usize) {
        if let Some(metric) = QUERY_PRUNING_RATIO.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .observe(pruning_ratio as f64);
        }

        if let Some(metric) = PROXIMA_BLOCKS_SCANNED.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .observe(blocks_scanned as f64);
        }

        if let Some(metric) = PROGRESSIVE_SEARCH_STAGES.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .observe(stages_used as f64);
        }
    }

    /// Record PCA metrics
    pub fn record_pca(&self, model_version: u32, projection_latency_us: u64, drift_score: f32) {
        if let Some(metric) = PCA_MODEL_VERSION.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .set(model_version as f64);
        }

        if let Some(metric) = PCA_PROJECTION_LATENCY.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .observe(projection_latency_us as f64);
        }

        if let Some(metric) = PCA_MODEL_DRIFT_SCORE.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .set(drift_score as f64);
        }
    }

    /// Record storage metrics
    pub fn record_storage(&self, level: usize, sstable_count: usize, total_size_bytes: u64) {
        if let Some(metric) = SSTABLE_COUNT.as_ref() {
            metric
                .with_label_values(&[&level.to_string(), &self.collection])
                .set(sstable_count as f64);
        }

        if let Some(metric) = SSTABLE_SIZE_BYTES.as_ref() {
            metric
                .with_label_values(&[&level.to_string(), &self.collection])
                .set(total_size_bytes as f64);
        }
    }

    /// Record bloom filter hit
    pub fn record_bloom_hit(&self, filter_type: &str) {
        if let Some(metric) = BLOOM_FILTER_HITS.as_ref() {
            metric
                .with_label_values(&[filter_type, &self.collection])
                .inc();
        }
    }

    /// Record liquid clustering metrics
    pub fn record_liquid_clustering(&self, quality: f32, hot_regions: usize, reorganized: bool) {
        if let Some(metric) = LIQUID_CLUSTERING_QUALITY.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .set(quality as f64);
        }

        if let Some(metric) = HOT_REGIONS_COUNT.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .set(hot_regions as f64);
        }

        if reorganized {
            if let Some(metric) = LIQUID_REORGANIZATIONS.as_ref() {
                metric.with_label_values(&[&self.collection]).inc();
            }
        }
    }

    /// Record quantization stage latency
    pub fn record_quantization_stage(&self, stage: &str, latency_ms: u64) {
        if let Some(metric) = QUANTIZATION_STAGE_LATENCY.as_ref() {
            metric
                .with_label_values(&[stage, &self.collection])
                .observe(latency_ms as f64);
        }
    }

    /// Record zone map metrics
    pub fn record_zone_map_pruning(&self, pruning_ratio: f32) {
        if let Some(metric) = ZONE_MAP_PRUNING_RATIO.as_ref() {
            metric
                .with_label_values(&[&self.collection])
                .observe(pruning_ratio as f64);
        }
    }

    /// Record dimension selectivity
    pub fn record_dimension_selectivity(&self, dimension: usize, selectivity: f32) {
        if let Some(metric) = DIMENSION_SELECTIVITY.as_ref() {
            metric
                .with_label_values(&[&dimension.to_string(), &self.collection])
                .observe(selectivity as f64);
        }
    }

    /// Get query throughput
    pub async fn query_throughput(&self) -> f64 {
        let count = self.query_count.load(Ordering::Relaxed);
        // Would need time tracking for actual throughput
        count as f64
    }

    /// Increment query count
    pub async fn inc_query_count(&self) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment compaction count
    pub async fn inc_compaction_count(&self) {
        self.compaction_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment flush count
    pub async fn inc_flush_count(&self) {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Backwards-compat alias for [`HelixPerformanceMonitor`].
pub type PerformanceMonitor = HelixPerformanceMonitor;

/// Performance monitor for tracking optimization opportunities
pub struct HelixPerformanceMonitor {
    metrics: Arc<HelixMetrics>,
    thresholds: HelixPerformanceThresholds,
}

/// Backwards-compat alias for [`HelixPerformanceThresholds`].
pub type PerformanceThresholds = HelixPerformanceThresholds;

#[derive(Debug, Clone)]
pub struct HelixPerformanceThresholds {
    pub max_pruning_ratio: f32,
    pub min_clustering_quality: f32,
    pub max_pca_drift: f32,
    pub max_query_latency_ms: u64,
}

impl Default for HelixPerformanceThresholds {
    fn default() -> Self {
        Self {
            max_pruning_ratio: 0.8,      // Alert if pruning > 80%
            min_clustering_quality: 0.6, // Alert if quality < 60%
            max_pca_drift: 0.3,          // Alert if drift > 30%
            max_query_latency_ms: 100,   // Alert if query > 100ms
        }
    }
}

impl HelixPerformanceMonitor {
    pub fn new(metrics: Arc<HelixMetrics>) -> Self {
        Self {
            metrics,
            thresholds: HelixPerformanceThresholds::default(),
        }
    }

    /// Check for performance issues
    pub async fn check_performance(&self) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();

        // Check clustering quality
        // In real implementation, would read from metrics

        // Check PCA drift
        // In real implementation, would read from metrics

        // Check query latency
        // In real implementation, would read from metrics

        alerts
    }

    /// Generate optimization recommendations
    pub async fn get_recommendations(&self) -> Vec<HelixOptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze metrics and generate recommendations
        let query_throughput = self.metrics.query_throughput().await;

        if query_throughput > 1000.0 {
            recommendations.push(HelixOptimizationRecommendation {
                category: "Caching".to_string(),
                suggestion: "Consider increasing block cache size".to_string(),
                expected_improvement: "20-30% latency reduction".to_string(),
            });
        }

        recommendations
    }
}

#[derive(Debug)]
pub struct PerformanceAlert {
    pub severity: AlertSeverity,
    pub category: String,
    pub message: String,
}

#[derive(Debug)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Backwards-compat alias for [`HelixOptimizationRecommendation`].
pub type OptimizationRecommendation = HelixOptimizationRecommendation;

#[derive(Debug)]
pub struct HelixOptimizationRecommendation {
    pub category: String,
    pub suggestion: String,
    pub expected_improvement: String,
}

/// Dashboard metrics for visualization
pub struct DashboardMetrics {
    pub query_latency_p50: f64,
    pub query_latency_p99: f64,
    pub pruning_efficiency: f32,
    pub clustering_quality: f32,
    pub hot_regions: usize,
    pub cache_hit_rate: f32,
    pub sstable_count: HashMap<usize, usize>,
    pub storage_size_gb: f64,
}

impl DashboardMetrics {
    pub async fn collect(collection: &str) -> Self {
        // In real implementation, would query Prometheus
        Self {
            query_latency_p50: 10.0,
            query_latency_p99: 50.0,
            pruning_efficiency: 0.85,
            clustering_quality: 0.75,
            hot_regions: 5,
            cache_hit_rate: 0.65,
            sstable_count: vec![(0, 4), (1, 2), (2, 1)].into_iter().collect(),
            storage_size_gb: 1.5,
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            r#"
HELIX Engine Performance Summary
================================
Query Performance:
  - P50 Latency: {:.2}ms
  - P99 Latency: {:.2}ms
  - Cache Hit Rate: {:.1}%

Clustering:
  - Quality Score: {:.1}%
  - Hot Regions: {}
  - Pruning Efficiency: {:.1}%

Storage:
  - Total SSTables: {}
  - Storage Size: {:.2}GB
  - Level Distribution: {:?}
"#,
            self.query_latency_p50,
            self.query_latency_p99,
            self.cache_hit_rate * 100.0,
            self.clustering_quality * 100.0,
            self.hot_regions,
            self.pruning_efficiency * 100.0,
            self.sstable_count.values().sum::<usize>(),
            self.storage_size_gb,
            self.sstable_count
        )
    }
}

use std::collections::HashMap;
