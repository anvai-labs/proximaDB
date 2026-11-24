//! Operation Metrics - Monitoring and observability for background operations
//!
//! This module provides comprehensive metrics for flush, compaction, and re-quantization
//! operations across all storage engines, enabling performance optimization and monitoring.

use crate::{OperationStatistics, OperationType};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Backwards-compat alias for [`StorageOperationMetrics`].
pub type OperationMetrics = StorageOperationMetrics;

/// Operation metrics for monitoring and optimization
pub struct StorageOperationMetrics {
    /// Operation counts by type and collection
    #[allow(dead_code)]
    operation_counts: Arc<RwLock<HashMap<String, HashMap<OperationType, u64>>>>,

    /// Duration statistics for performance analysis
    duration_stats: Arc<RwLock<HashMap<OperationType, DurationStats>>>,

    /// Success/failure rates
    success_rates: Arc<RwLock<HashMap<OperationType, SuccessRate>>>,

    /// Resource utilization during operations
    #[allow(dead_code)]
    resource_usage: Arc<RwLock<ResourceUsageStats>>,
}

/// Duration statistics for operations
#[derive(Debug, Clone, Default)]
struct DurationStats {
    total_duration: std::time::Duration,
    count: u64,
    min_duration: Option<std::time::Duration>,
    max_duration: Option<std::time::Duration>,
}

/// Success rate tracking
#[derive(Debug, Clone, Default)]
struct SuccessRate {
    total_attempts: u64,
    successful_attempts: u64,
}

/// Resource utilization statistics
#[derive(Debug, Clone, Default)]
struct ResourceUsageStats {
    #[allow(dead_code)]
    peak_memory_usage: u64,
    #[allow(dead_code)]
    peak_cpu_usage: f64,
    #[allow(dead_code)]
    peak_io_throughput: u64,
    #[allow(dead_code)]
    concurrent_operations_peak: usize,
}

impl StorageOperationMetrics {
    /// Create new operation metrics tracker
    pub fn new() -> Self {
        info!("📊 Initializing StorageOperationMetrics");

        Self {
            operation_counts: Arc::new(RwLock::new(HashMap::new())),
            duration_stats: Arc::new(RwLock::new(HashMap::new())),
            success_rates: Arc::new(RwLock::new(HashMap::new())),
            resource_usage: Arc::new(RwLock::new(ResourceUsageStats::default())),
        }
    }

    /// Record operation execution with timing and outcome
    pub async fn record_operation(
        &self,
        operation_type: OperationType,
        duration: std::time::Duration,
        success: bool,
    ) {
        // Update duration statistics
        {
            let mut duration_stats = self.duration_stats.write().await;
            let stats = duration_stats.entry(operation_type).or_default();

            stats.total_duration += duration;
            stats.count += 1;

            if stats.min_duration.is_none() || Some(duration) < stats.min_duration {
                stats.min_duration = Some(duration);
            }

            if stats.max_duration.is_none() || Some(duration) > stats.max_duration {
                stats.max_duration = Some(duration);
            }
        }

        // Update success rates
        {
            let mut success_rates = self.success_rates.write().await;
            let rate = success_rates.entry(operation_type).or_default();

            rate.total_attempts += 1;
            if success {
                rate.successful_attempts += 1;
            }
        }

        info!(
            "📊 Recorded {} operation: duration={:?}, success={}",
            operation_type_name(operation_type),
            duration,
            success
        );
    }

    /// Get average duration for operation type
    pub async fn get_average_duration(
        &self,
        operation_type: OperationType,
    ) -> Option<std::time::Duration> {
        let duration_stats = self.duration_stats.read().await;

        if let Some(stats) = duration_stats.get(&operation_type) {
            if stats.count > 0 {
                Some(stats.total_duration / stats.count as u32)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get success rate for operation type
    pub async fn get_success_rate(&self, operation_type: OperationType) -> f64 {
        let success_rates = self.success_rates.read().await;

        if let Some(rate) = success_rates.get(&operation_type) {
            if rate.total_attempts > 0 {
                rate.successful_attempts as f64 / rate.total_attempts as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Get comprehensive operation statistics
    pub async fn get_operation_statistics(&self) -> OperationStatistics {
        let duration_stats = self.duration_stats.read().await;
        let success_rates = self.success_rates.read().await;

        let total_operations: u64 = duration_stats.values().map(|s| s.count).sum();
        let successful_operations: u64 =
            success_rates.values().map(|r| r.successful_attempts).sum();
        let failed_operations: u64 = success_rates
            .values()
            .map(|r| r.total_attempts - r.successful_attempts)
            .sum();

        let average_duration_ms = if total_operations > 0 {
            let total_duration: std::time::Duration =
                duration_stats.values().map(|s| s.total_duration).sum();
            total_duration.as_secs_f64() * 1000.0 / total_operations as f64
        } else {
            0.0
        };

        let operations_by_type = duration_stats
            .iter()
            .map(|(op_type, stats)| (*op_type, stats.count))
            .collect();

        OperationStatistics {
            total_operations,
            successful_operations,
            failed_operations,
            average_duration_ms,
            operations_by_type,
        }
    }

    /// Generate Prometheus-compatible metrics
    pub async fn generate_prometheus_metrics(&self) -> Vec<String> {
        let mut metrics = Vec::new();

        // Add operation counts
        let duration_stats = self.duration_stats.read().await;
        for (op_type, stats) in duration_stats.iter() {
            metrics.push(format!(
                "proximadb_operations_total{{operation=\"{}\"}} {}",
                operation_type_name(*op_type),
                stats.count
            ));

            if stats.count > 0 {
                let avg_duration_ms =
                    stats.total_duration.as_secs_f64() * 1000.0 / stats.count as f64;
                metrics.push(format!(
                    "proximadb_operation_duration_ms{{operation=\"{}\"}} {:.2}",
                    operation_type_name(*op_type),
                    avg_duration_ms
                ));
            }
        }

        // Add success rates
        let success_rates = self.success_rates.read().await;
        for (op_type, rate) in success_rates.iter() {
            if rate.total_attempts > 0 {
                let success_rate = rate.successful_attempts as f64 / rate.total_attempts as f64;
                metrics.push(format!(
                    "proximadb_operation_success_rate{{operation=\"{}\"}} {:.3}",
                    operation_type_name(*op_type),
                    success_rate
                ));
            }
        }

        metrics
    }
}

impl Default for StorageOperationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall flush status for monitoring
#[derive(Debug, Clone)]
pub struct FlushStatus {
    pub sst_active_flushes: usize,
    pub helix_active_flushes: usize,
    pub total_flushes_completed: u64,
    pub average_flush_time_ms: f64,
}

/// Helper function for operation type names
fn operation_type_name(op: OperationType) -> &'static str {
    match op {
        OperationType::Flush => "flush",
        OperationType::MinorCompaction => "minor_compaction",
        OperationType::MajorCompaction => "major_compaction",
        OperationType::Requantization => "requantization",
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_recording() {
        let metrics = StorageOperationMetrics::new();

        // Record some operations
        metrics
            .record_operation(
                OperationType::Flush,
                std::time::Duration::from_millis(150),
                true,
            )
            .await;

        metrics
            .record_operation(
                OperationType::MinorCompaction,
                std::time::Duration::from_millis(2500),
                true,
            )
            .await;

        // Verify metrics
        let avg_flush = metrics.get_average_duration(OperationType::Flush).await;
        assert!(avg_flush.is_some());
        assert_eq!(avg_flush.unwrap().as_millis(), 150);

        let success_rate = metrics.get_success_rate(OperationType::Flush).await;
        assert_eq!(success_rate, 1.0);
    }

    #[tokio::test]
    async fn test_prometheus_metrics_generation() {
        let metrics = StorageOperationMetrics::new();

        // Record some test data
        metrics
            .record_operation(
                OperationType::Flush,
                std::time::Duration::from_millis(100),
                true,
            )
            .await;
        metrics
            .record_operation(
                OperationType::Flush,
                std::time::Duration::from_millis(200),
                true,
            )
            .await;

        let prometheus_metrics = metrics.generate_prometheus_metrics().await;

        assert!(!prometheus_metrics.is_empty());
        assert!(
            prometheus_metrics
                .iter()
                .any(|m| m.contains("proximadb_operations_total"))
        );
        assert!(
            prometheus_metrics
                .iter()
                .any(|m| m.contains("proximadb_operation_duration_ms"))
        );
    }
}
