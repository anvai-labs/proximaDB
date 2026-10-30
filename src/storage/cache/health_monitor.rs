//! Cache monitoring and observability dashboard

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use crate::metrics::CacheMetricsSnapshot;
use crate::storage::cache::config::CacheConfig;
use crate::storage::cache::{CacheType, CrossCacheOrchestrator};

/// Cache monitoring dashboard for real-time insights
///
/// Provides comprehensive monitoring, alerting, and profiling capabilities
/// for the cache system with historical metrics tracking.
pub struct CacheMonitoringDashboard {
    /// Cache orchestrator reference
    orchestrator: Arc<CrossCacheOrchestrator>,

    /// Configuration
    config: Arc<CacheConfig>,

    /// Historical metrics
    history: Arc<RwLock<MetricsHistory>>,

    /// CacheHealthAlert manager
    alert_manager: Arc<AlertManager>,

    /// Performance profiler
    _profiler: Option<Arc<PerformanceProfiler>>,
}

/// Historical metrics storage
///
/// Stores time-series data points with a maximum size limit.
#[derive(Debug, Clone, Default)]
struct MetricsHistory {
    /// Time series data points
    time_series: Vec<CacheHealthMetricsSnapshot>,

    /// Maximum history size
    max_size: usize,
}

/// Point-in-time metrics snapshot
///
/// Captures cache metrics and system resource usage at a specific moment in time.
#[derive(Debug, Clone)]
pub struct CacheHealthMetricsSnapshot {
    /// When the snapshot was taken
    pub timestamp: SystemTime,
    /// Cache performance metrics
    pub cache_metrics: CacheMetricsSnapshot,
    /// Memory pressure (0.0-1.0)
    pub memory_pressure: f64,
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,
    /// I/O wait time (0.0-1.0)
    pub io_wait: f64,
}

/// CacheHealthAlert manager for threshold monitoring
///
/// Monitors cache metrics and triggers alerts when thresholds are exceeded.
pub struct AlertManager {
    /// Active alerts
    active_alerts: Arc<RwLock<Vec<CacheHealthAlert>>>,

    /// CacheHealthAlert thresholds from config
    _thresholds: AlertThresholds,

    /// CacheHealthAlert handlers
    _handlers: Vec<Box<dyn AlertHandler + Send + Sync>>,
}

/// Individual alert
///
/// Represents a cache-related alert triggered when a threshold is exceeded.
#[derive(Debug, Clone)]
pub struct CacheHealthAlert {
    /// Unique alert identifier
    pub id: String,
    /// CacheHealthAlert severity level
    pub severity: AlertSeverity,
    /// Human-readable alert message
    pub message: String,
    /// When the alert was triggered
    pub triggered_at: SystemTime,
    /// Which cache type triggered the alert
    pub cache_type: Option<CacheType>,
    /// Current metric value
    pub metric_value: f64,
    /// Threshold that was exceeded
    pub threshold: f64,
}

/// CacheHealthAlert severity levels
///
/// Indicates the urgency and impact of an alert.
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    /// Informational alert (no action required)
    Info,
    /// Warning alert (monitoring recommended)
    Warning,
    /// Critical alert (immediate action required)
    Critical,
}

/// CacheHealthAlert thresholds configuration
///
/// Defines threshold values for triggering cache alerts.
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Minimum acceptable hit rate (0.0-1.0)
    _min_hit_rate: f64,
    /// Maximum memory usage (0.0-1.0)
    _max_memory_usage: f64,
    /// Maximum eviction rate (0.0-1.0)
    _max_eviction_rate: f64,
    /// Maximum cascade size
    _max_cascade_size: usize,
    /// Maximum prefetch queue size
    _max_prefetch_queue: usize,
}

/// CacheHealthAlert handler trait
///
/// Defines the interface for handling cache alerts.
trait AlertHandler: Send + Sync {
    /// Handle an alert
    #[allow(dead_code)]
    fn handle_alert(&self, alert: &CacheHealthAlert);
}

/// Performance profiler for detailed analysis
///
/// Profiles cache operations for performance analysis and optimization.
pub struct PerformanceProfiler {
    /// Profiling data
    profiles: Arc<RwLock<Vec<PerformanceProfile>>>,

    /// Sampling configuration (0.0-1.0)
    sampling_rate: f64,

    /// Output path for profiles
    _output_path: String,
}

/// Performance profile data point
///
/// Records timing and metadata for a single cache operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceProfile {
    /// Timestamp in milliseconds
    timestamp_ms: u64,
    /// Operation name
    operation: String,
    /// Duration in milliseconds
    duration_ms: f64,
    /// Cache type
    cache_type: CacheType,
    /// Cache tier
    tier: String,
    /// Whether it was a cache hit
    hit: bool,
    /// Size of the value
    value_size: usize,
}

impl CacheMonitoringDashboard {
    /// Create new monitoring dashboard
    pub fn new(orchestrator: Arc<CrossCacheOrchestrator>, config: Arc<CacheConfig>) -> Self {
        let alert_manager = Arc::new(AlertManager::new(&config.monitoring.alert_thresholds));

        let profiler = if config.monitoring.enable_profiling {
            Some(Arc::new(PerformanceProfiler::new(
                config.monitoring.trace_sampling_rate,
                config
                    .monitoring
                    .profile_output_path
                    .clone()
                    .unwrap_or_else(|| "/tmp/cache_profiles".to_string()),
            )))
        } else {
            None
        };

        Self {
            orchestrator,
            config,
            history: Arc::new(RwLock::new(MetricsHistory {
                time_series: Vec::new(),
                max_size: 1000,
            })),
            alert_manager,
            _profiler: profiler,
        }
    }

    /// Start monitoring loop
    pub async fn start(&self) {
        let interval = Duration::from_secs(self.config.monitoring.metrics_interval_seconds);
        let orchestrator = self.orchestrator.clone();
        let history = self.history.clone();
        let alert_manager = self.alert_manager.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);

            loop {
                interval.tick().await;

                // Collect metrics - convert from CacheMetrics to CacheMetricsSnapshot
                let cache_metrics = orchestrator.metrics();
                use crate::metrics::cache::{CoordinationMetrics, EvictionMetrics, MemoryMetrics};
                let metrics = CacheMetricsSnapshot {
                    overall_hit_rate: cache_metrics.hit_rate(),
                    l1_metrics: cache_metrics
                        .tier_metrics(crate::storage::cache::backend::CacheTier::L1),
                    l2_metrics: cache_metrics
                        .tier_metrics(crate::storage::cache::backend::CacheTier::L2),
                    l3_metrics: cache_metrics
                        .tier_metrics(crate::storage::cache::backend::CacheTier::L3),
                    memory_usage: MemoryMetrics {
                        total_allocated_bytes: cache_metrics.total_allocated_bytes(),
                        used_bytes: cache_metrics.used_bytes(),
                        fragmentation_ratio: cache_metrics.fragmentation_ratio(),
                        allocation_failures: cache_metrics.allocation_failures(),
                    },
                    eviction_metrics: EvictionMetrics {
                        total_evictions: cache_metrics.total_evictions(),
                        lru_evictions: cache_metrics.lru_evictions(),
                        lfu_evictions: cache_metrics.lfu_evictions(),
                        arc_evictions: cache_metrics.arc_evictions(),
                        memory_pressure_evictions: cache_metrics.memory_pressure_evictions(),
                        ttl_evictions: cache_metrics.ttl_evictions(),
                    },
                    coordination_metrics: CoordinationMetrics {
                        cross_cache_hits: cache_metrics.cross_cache_hits(),
                        prefetch_success_rate: cache_metrics.prefetch_success_rate(),
                        invalidation_cascades: cache_metrics.invalidation_cascades(),
                        memory_rebalances: cache_metrics.memory_rebalances(),
                    },
                    last_updated: SystemTime::now(),
                };

                // Create snapshot
                let snapshot = CacheHealthMetricsSnapshot {
                    timestamp: SystemTime::now(),
                    cache_metrics: metrics.clone(),
                    memory_pressure: Self::get_memory_pressure(),
                    cpu_usage: Self::get_cpu_usage(),
                    io_wait: Self::get_io_wait(),
                };

                // Store in history
                {
                    let mut hist = history.write().await;
                    hist.time_series.push(snapshot.clone());
                    if hist.time_series.len() > hist.max_size {
                        hist.time_series.remove(0);
                    }
                }

                // Check alerts
                alert_manager
                    .check_alerts(&metrics, &config.monitoring.alert_thresholds)
                    .await;
            }
        });
    }

    /// Get current dashboard state
    pub async fn get_dashboard_state(&self) -> CacheHealthDashboardState {
        let history = self.history.read().await;
        let alerts = self.alert_manager.active_alerts.read().await;
        // Convert from CacheMetrics to CacheMetricsSnapshot
        let cache_metrics = self.orchestrator.metrics();
        use crate::metrics::cache::{CoordinationMetrics, EvictionMetrics, MemoryMetrics};
        let metrics = CacheMetricsSnapshot {
            overall_hit_rate: cache_metrics.hit_rate() * 100.0,
            l1_metrics: cache_metrics.tier_metrics(crate::storage::cache::backend::CacheTier::L1),
            l2_metrics: cache_metrics.tier_metrics(crate::storage::cache::backend::CacheTier::L2),
            l3_metrics: cache_metrics.tier_metrics(crate::storage::cache::backend::CacheTier::L3),
            memory_usage: MemoryMetrics {
                total_allocated_bytes: cache_metrics.total_allocated_bytes(),
                used_bytes: cache_metrics.used_bytes(),
                fragmentation_ratio: cache_metrics.fragmentation_ratio(),
                allocation_failures: cache_metrics.allocation_failures(),
            },
            eviction_metrics: EvictionMetrics {
                total_evictions: cache_metrics.total_evictions(),
                lru_evictions: cache_metrics.lru_evictions(),
                lfu_evictions: cache_metrics.lfu_evictions(),
                arc_evictions: cache_metrics.arc_evictions(),
                memory_pressure_evictions: cache_metrics.memory_pressure_evictions(),
                ttl_evictions: cache_metrics.ttl_evictions(),
            },
            coordination_metrics: CoordinationMetrics {
                cross_cache_hits: cache_metrics.cross_cache_hits(),
                prefetch_success_rate: cache_metrics.prefetch_success_rate(),
                invalidation_cascades: cache_metrics.invalidation_cascades(),
                memory_rebalances: cache_metrics.memory_rebalances(),
            },
            last_updated: SystemTime::now(),
        };

        CacheHealthDashboardState {
            current_metrics: metrics.clone(),
            active_alerts: alerts.clone(),
            recent_history: history
                .time_series
                .iter()
                .rev()
                .take(100)
                .cloned()
                .collect(),
            cache_status: self.get_cache_status().await,
            optimization_suggestions: self.get_optimization_suggestions(&metrics).await,
        }
    }

    /// Get cache status for each cache type
    async fn get_cache_status(&self) -> HashMap<String, CacheStatus> {
        let mut status = HashMap::new();

        status.insert(
            "vector_data".to_string(),
            CacheStatus {
                enabled: self.config.vector_data.enabled,
                memory_allocated_mb: self.config.global.total_memory_mb
                    * self.config.vector_data.memory_percentage as usize
                    / 100,
                entries: 0, // Would get from actual cache
                hit_rate: 0.0,
            },
        );

        status.insert(
            "query_result".to_string(),
            CacheStatus {
                enabled: self.config.query_result.enabled,
                memory_allocated_mb: self.config.global.total_memory_mb
                    * self.config.query_result.memory_percentage as usize
                    / 100,
                entries: 0,
                hit_rate: 0.0,
            },
        );

        status
    }

    /// Get optimization suggestions based on metrics
    async fn get_optimization_suggestions(&self, metrics: &CacheMetricsSnapshot) -> Vec<String> {
        let mut suggestions = Vec::new();

        if metrics.overall_hit_rate < 0.5 {
            suggestions.push("Consider increasing cache memory allocation".to_string());
        }

        if metrics.l1_metrics.hit_rate < 0.7 && self.config.global.enable_tiered_storage {
            suggestions.push("L1 hit rate is low, consider adjusting tier thresholds".to_string());
        }

        if metrics.eviction_metrics.memory_pressure_evictions
            > metrics.eviction_metrics.total_evictions / 2
        {
            suggestions
                .push("High memory pressure evictions, increase total memory budget".to_string());
        }

        if metrics.coordination_metrics.prefetch_success_rate < 0.3 {
            suggestions
                .push("Low prefetch success rate, consider tuning pattern analysis".to_string());
        }

        suggestions
    }

    // System metrics collection (simplified)
    fn get_memory_pressure() -> f64 {
        // Would use actual system metrics
        0.5
    }

    fn get_cpu_usage() -> f64 {
        // Would use actual system metrics
        0.3
    }

    fn get_io_wait() -> f64 {
        // Would use actual system metrics
        0.1
    }
}

/// Dashboard state for API responses
///
/// Aggregates current metrics, alerts, and historical data for dashboard display.
///
/// Backwards-compat alias for [`CacheHealthDashboardState`].
pub type DashboardState = CacheHealthDashboardState;

#[derive(Debug, Clone)]
pub struct CacheHealthDashboardState {
    /// Current cache metrics snapshot
    pub current_metrics: CacheMetricsSnapshot,
    /// Currently active alerts
    pub active_alerts: Vec<CacheHealthAlert>,
    /// Historical metrics data
    pub recent_history: Vec<CacheHealthMetricsSnapshot>,
    /// Per-cache status information
    pub cache_status: HashMap<String, CacheStatus>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
}

/// Individual cache status
///
/// Status information for a single cache instance.
#[derive(Debug, Clone)]
pub struct CacheStatus {
    /// Whether the cache is enabled
    pub enabled: bool,
    /// Memory allocated in MB
    pub memory_allocated_mb: usize,
    /// Number of entries
    pub entries: usize,
    /// Current hit rate (0.0-1.0)
    pub hit_rate: f64,
}

impl AlertManager {
    pub fn new(thresholds: &crate::storage::cache::config::AlertThresholds) -> Self {
        Self {
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            _thresholds: AlertThresholds {
                _min_hit_rate: thresholds.min_hit_rate,
                _max_memory_usage: thresholds.max_memory_usage,
                _max_eviction_rate: thresholds.max_eviction_rate,
                _max_cascade_size: thresholds.max_cascade_size,
                _max_prefetch_queue: thresholds.max_prefetch_queue,
            },
            _handlers: Vec::new(),
        }
    }

    pub async fn get_active_alerts(&self) -> Vec<CacheHealthAlert> {
        self.active_alerts.read().await.clone()
    }

    pub async fn check_alerts(
        &self,
        metrics: &CacheMetricsSnapshot,
        thresholds: &crate::storage::cache::config::AlertThresholds,
    ) {
        let mut alerts = self.active_alerts.write().await;
        alerts.clear();

        // Check hit rate
        if metrics.overall_hit_rate < thresholds.min_hit_rate {
            alerts.push(CacheHealthAlert {
                id: "low_hit_rate".to_string(),
                severity: AlertSeverity::Warning,
                message: format!(
                    "Cache hit rate {} below threshold {}",
                    metrics.overall_hit_rate, thresholds.min_hit_rate
                ),
                triggered_at: SystemTime::now(),
                cache_type: None,
                metric_value: metrics.overall_hit_rate,
                threshold: thresholds.min_hit_rate,
            });
        }

        // Check memory usage
        let memory_usage = metrics.memory_usage.used_bytes as f64
            / metrics.memory_usage.total_allocated_bytes as f64;
        if memory_usage > thresholds.max_memory_usage {
            alerts.push(CacheHealthAlert {
                id: "high_memory_usage".to_string(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "Memory usage {} exceeds threshold {}",
                    memory_usage, thresholds.max_memory_usage
                ),
                triggered_at: SystemTime::now(),
                cache_type: None,
                metric_value: memory_usage,
                threshold: thresholds.max_memory_usage,
            });
        }
    }
}

impl PerformanceProfiler {
    pub fn new(sampling_rate: f64, output_path: String) -> Self {
        Self {
            profiles: Arc::new(RwLock::new(Vec::new())),
            sampling_rate,
            _output_path: output_path,
        }
    }

    pub async fn record_operation(
        &self,
        operation: String,
        cache_type: CacheType,
        tier: String,
        duration_ms: f64,
        hit: bool,
        value_size: usize,
    ) {
        // Sample based on rate
        if rand::random::<f64>() > self.sampling_rate {
            return;
        }

        let profile = PerformanceProfile {
            timestamp_ms: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            operation,
            duration_ms,
            cache_type,
            tier,
            hit,
            value_size,
        };

        let mut profiles = self.profiles.write().await;
        profiles.push(profile);

        // Periodically write to disk
        if profiles.len() >= 1000 {
            self.flush_to_disk(&profiles).await;
            profiles.clear();
        }
    }

    async fn flush_to_disk(&self, profiles: &[PerformanceProfile]) {
        // Would write to actual file
        let _json = serde_json::to_string_pretty(profiles).unwrap_or_else(|_| "[]".to_string());
        // std::fs::write(&self.output_path, json).ok();
    }
}

use rand;
