//! # Workload Router
//!
//! Intelligent query routing based on workload characteristics.
//! Routes queries to OLTP (SST) or OLAP based on:
//! - Query pattern analysis
//! - Estimated row count
//! - Presence of aggregations
//! - Historical query patterns (adaptive learning)

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::RwLock;
use tracing::debug;

use super::replication::ReplicationCoordinator;

/// Workload type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Online Transaction Processing - point queries, small transactions
    OLTP,
    /// Online Analytical Processing - aggregations, large scans
    OLAP,
    /// Hybrid - could go either way
    Hybrid,
}

impl WorkloadType {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkloadType::OLTP => "OLTP",
            WorkloadType::OLAP => "OLAP",
            WorkloadType::Hybrid => "HYBRID",
        }
    }
}

/// Query characteristics for routing decisions
#[derive(Debug, Clone, Default)]
pub struct QueryCharacteristics {
    /// Estimated number of rows to process
    pub estimated_rows: Option<usize>,
    /// Has aggregation functions (SUM, COUNT, AVG, etc.)
    pub has_aggregation: bool,
    /// Has GROUP BY clause
    pub has_group_by: bool,
    /// Has ORDER BY clause
    pub has_order_by: bool,
    /// Number of tables/joins involved
    pub join_count: usize,
    /// Is a point lookup (WHERE id = ?)
    pub is_point_lookup: bool,
    /// Is a range scan (WHERE x BETWEEN ? AND ?)
    pub is_range_scan: bool,
    /// Is a full table scan
    pub is_full_scan: bool,
    /// Tables accessed
    pub tables: Vec<String>,
    /// Query complexity score (0-100)
    pub complexity_score: u8,
}

impl QueryCharacteristics {
    /// Create characteristics for a point lookup
    pub fn point_lookup(table: &str) -> Self {
        Self {
            estimated_rows: Some(1),
            is_point_lookup: true,
            tables: vec![table.to_string()],
            complexity_score: 5,
            ..Default::default()
        }
    }

    /// Create characteristics for an aggregation query
    pub fn aggregation(table: &str, estimated_rows: usize) -> Self {
        Self {
            estimated_rows: Some(estimated_rows),
            has_aggregation: true,
            has_group_by: true,
            tables: vec![table.to_string()],
            complexity_score: 70,
            ..Default::default()
        }
    }

    /// Create characteristics for a range scan
    pub fn range_scan(table: &str, estimated_rows: usize) -> Self {
        Self {
            estimated_rows: Some(estimated_rows),
            is_range_scan: true,
            tables: vec![table.to_string()],
            complexity_score: 40,
            ..Default::default()
        }
    }

    /// Create characteristics for a full scan
    pub fn full_scan(table: &str, estimated_rows: usize) -> Self {
        Self {
            estimated_rows: Some(estimated_rows),
            is_full_scan: true,
            tables: vec![table.to_string()],
            complexity_score: 80,
            ..Default::default()
        }
    }
}

/// Backwards-compat alias for [`LegacyHtapRoutingDecision`].
pub type RoutingDecision = LegacyHtapRoutingDecision;

/// Routing decision with explanation
#[derive(Debug, Clone)]
pub struct LegacyHtapRoutingDecision {
    /// Workload type classification
    pub workload_type: WorkloadType,
    /// Should use OLAP store
    pub use_olap: bool,
    /// Reason for the decision
    pub reason: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Estimated cost on OLTP
    pub oltp_cost_estimate: u64,
    /// Estimated cost on OLAP
    pub olap_cost_estimate: u64,
}

impl LegacyHtapRoutingDecision {
    /// Create an OLTP routing decision
    pub fn oltp(reason: impl Into<String>) -> Self {
        Self {
            workload_type: WorkloadType::OLTP,
            use_olap: false,
            reason: reason.into(),
            confidence: 0.9,
            oltp_cost_estimate: 1,
            olap_cost_estimate: 10,
        }
    }

    /// Create an OLAP routing decision
    pub fn olap(reason: impl Into<String>) -> Self {
        Self {
            workload_type: WorkloadType::OLAP,
            use_olap: true,
            reason: reason.into(),
            confidence: 0.9,
            oltp_cost_estimate: 100,
            olap_cost_estimate: 1,
        }
    }

    /// Create a hybrid routing decision (defaults to OLTP)
    pub fn hybrid(reason: impl Into<String>, prefer_olap: bool) -> Self {
        Self {
            workload_type: WorkloadType::Hybrid,
            use_olap: prefer_olap,
            reason: reason.into(),
            confidence: 0.5,
            oltp_cost_estimate: 10,
            olap_cost_estimate: 10,
        }
    }
}

/// Router configuration
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Threshold for switching to OLAP (estimated rows)
    pub olap_row_threshold: usize,
    /// Always use OLAP for aggregations
    pub olap_for_aggregations: bool,
    /// Always use OLAP for full scans
    pub olap_for_full_scans: bool,
    /// Minimum complexity score for OLAP
    pub olap_complexity_threshold: u8,
    /// Enable adaptive learning
    pub adaptive_learning: bool,
    /// Maximum staleness tolerance for OLAP queries (ms)
    pub max_staleness_ms: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            olap_row_threshold: 10_000,
            olap_for_aggregations: true,
            olap_for_full_scans: true,
            olap_complexity_threshold: 60,
            adaptive_learning: true,
            max_staleness_ms: 5_000,
        }
    }
}

/// Query execution history for adaptive learning
#[derive(Debug, Clone, Default)]
struct QueryHistory {
    /// Number of times routed to OLTP
    oltp_count: u64,
    /// Number of times routed to OLAP
    olap_count: u64,
    /// Total OLTP execution time in ms
    oltp_time_ms: u64,
    /// Total OLAP execution time in ms
    olap_time_ms: u64,
}

/// Workload router for HTAP query routing
pub struct WorkloadRouter {
    /// Configuration
    config: RouterConfig,

    /// Replication coordinator for freshness checks
    replication: Option<Arc<ReplicationCoordinator>>,

    /// Query execution history (for adaptive learning)
    query_history: RwLock<HashMap<String, QueryHistory>>,

    /// Total queries routed to OLTP
    oltp_queries: AtomicU64,

    /// Total queries routed to OLAP
    olap_queries: AtomicU64,
}

impl WorkloadRouter {
    /// Create a new workload router
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            replication: None,
            query_history: RwLock::new(HashMap::new()),
            oltp_queries: AtomicU64::new(0),
            olap_queries: AtomicU64::new(0),
        }
    }

    /// Create with replication coordinator
    pub fn with_replication(
        config: RouterConfig,
        replication: Arc<ReplicationCoordinator>,
    ) -> Self {
        Self {
            config,
            replication: Some(replication),
            query_history: RwLock::new(HashMap::new()),
            oltp_queries: AtomicU64::new(0),
            olap_queries: AtomicU64::new(0),
        }
    }

    /// Route a query based on its characteristics
    pub async fn route(
        &self,
        query_id: &str,
        chars: &QueryCharacteristics,
    ) -> LegacyHtapRoutingDecision {
        // First, check if OLAP is even available and fresh enough
        if let Some(replication) = &self.replication {
            // Check freshness for all tables
            for table in &chars.tables {
                if !replication.can_use_olap(table, true).await {
                    return LegacyHtapRoutingDecision::oltp(
                        "OLAP not fresh enough for required tables",
                    );
                }
            }
        }

        // Point lookups always go to OLTP
        if chars.is_point_lookup {
            self.oltp_queries.fetch_add(1, Ordering::Relaxed);
            return LegacyHtapRoutingDecision::oltp("Point lookup - optimized for OLTP");
        }

        // Aggregations with GROUP BY go to OLAP
        if chars.has_aggregation && chars.has_group_by && self.config.olap_for_aggregations {
            self.olap_queries.fetch_add(1, Ordering::Relaxed);
            return LegacyHtapRoutingDecision::olap(
                "Aggregation with GROUP BY - optimized for OLAP",
            );
        }

        // Full table scans go to OLAP
        if chars.is_full_scan && self.config.olap_for_full_scans {
            self.olap_queries.fetch_add(1, Ordering::Relaxed);
            return LegacyHtapRoutingDecision::olap("Full table scan - optimized for OLAP");
        }

        // Check row count threshold
        if let Some(rows) = chars.estimated_rows
            && rows >= self.config.olap_row_threshold
        {
            self.olap_queries.fetch_add(1, Ordering::Relaxed);
            return LegacyHtapRoutingDecision::olap(format!(
                "Large result set ({} rows >= {} threshold)",
                rows, self.config.olap_row_threshold
            ));
        }

        // Check complexity score
        if chars.complexity_score >= self.config.olap_complexity_threshold {
            self.olap_queries.fetch_add(1, Ordering::Relaxed);
            return LegacyHtapRoutingDecision::olap(format!(
                "High complexity score ({} >= {})",
                chars.complexity_score, self.config.olap_complexity_threshold
            ));
        }

        // Check adaptive learning history
        if self.config.adaptive_learning
            && let Some(decision) = self.check_history(query_id).await
        {
            if decision.use_olap {
                self.olap_queries.fetch_add(1, Ordering::Relaxed);
            } else {
                self.oltp_queries.fetch_add(1, Ordering::Relaxed);
            }
            return decision;
        }

        // Default to OLTP for uncertain cases
        self.oltp_queries.fetch_add(1, Ordering::Relaxed);
        LegacyHtapRoutingDecision::oltp("Default routing - no clear OLAP indicators")
    }

    /// Check historical performance for adaptive routing
    async fn check_history(&self, query_id: &str) -> Option<LegacyHtapRoutingDecision> {
        let history = self.query_history.read().await;

        if let Some(h) = history.get(query_id) {
            // Need at least some data points
            if h.oltp_count + h.olap_count < 5 {
                return None;
            }

            // Compare average execution times — `u64::MAX` sentinel means
            // "no observations, treat as worst-case" so the comparator
            // prefers the other branch.
            let oltp_avg = h.oltp_time_ms.checked_div(h.oltp_count).unwrap_or(u64::MAX);
            let olap_avg = h.olap_time_ms.checked_div(h.olap_count).unwrap_or(u64::MAX);

            if olap_avg < oltp_avg / 2 {
                return Some(LegacyHtapRoutingDecision::olap(format!(
                    "Adaptive learning: OLAP {}ms vs OLTP {}ms",
                    olap_avg, oltp_avg
                )));
            } else if oltp_avg < olap_avg / 2 {
                return Some(LegacyHtapRoutingDecision::oltp(format!(
                    "Adaptive learning: OLTP {}ms vs OLAP {}ms",
                    oltp_avg, olap_avg
                )));
            }
        }

        None
    }

    /// Record query execution for adaptive learning
    pub async fn record_execution(&self, query_id: &str, used_olap: bool, execution_time_ms: u64) {
        if !self.config.adaptive_learning {
            return;
        }

        let mut history = self.query_history.write().await;
        let entry = history.entry(query_id.to_string()).or_default();

        if used_olap {
            entry.olap_count += 1;
            entry.olap_time_ms += execution_time_ms;
        } else {
            entry.oltp_count += 1;
            entry.oltp_time_ms += execution_time_ms;
        }

        debug!(
            "Recorded {} execution for query {}: {}ms",
            if used_olap { "OLAP" } else { "OLTP" },
            query_id,
            execution_time_ms
        );
    }

    /// Get routing statistics
    pub fn stats(&self) -> LegacyHtapRouterStats {
        LegacyHtapRouterStats {
            oltp_queries: self.oltp_queries.load(Ordering::Relaxed),
            olap_queries: self.olap_queries.load(Ordering::Relaxed),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Clear history (for testing)
    #[cfg(test)]
    pub async fn clear_history(&self) {
        let mut history = self.query_history.write().await;
        history.clear();
    }
}

impl Default for WorkloadRouter {
    fn default() -> Self {
        Self::new(RouterConfig::default())
    }
}

/// Backwards-compat alias for [`LegacyHtapRouterStats`].
pub type RouterStats = LegacyHtapRouterStats;

/// Router statistics
#[derive(Debug, Clone, Default)]
pub struct LegacyHtapRouterStats {
    /// Total queries routed to OLTP
    pub oltp_queries: u64,
    /// Total queries routed to OLAP
    pub olap_queries: u64,
}

impl LegacyHtapRouterStats {
    /// Get percentage routed to OLAP
    pub fn olap_percentage(&self) -> f64 {
        let total = self.oltp_queries + self.olap_queries;
        if total == 0 {
            0.0
        } else {
            (self.olap_queries as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_route_point_lookup() {
        let router = WorkloadRouter::new(RouterConfig::default());
        let chars = QueryCharacteristics::point_lookup("users");

        let decision = router.route("q1", &chars).await;

        assert_eq!(decision.workload_type, WorkloadType::OLTP);
        assert!(!decision.use_olap);
        assert!(decision.reason.contains("Point lookup"));
    }

    #[tokio::test]
    async fn test_route_aggregation() {
        let router = WorkloadRouter::new(RouterConfig::default());
        let chars = QueryCharacteristics::aggregation("orders", 100_000);

        let decision = router.route("q2", &chars).await;

        assert_eq!(decision.workload_type, WorkloadType::OLAP);
        assert!(decision.use_olap);
        assert!(decision.reason.contains("Aggregation"));
    }

    #[tokio::test]
    async fn test_route_full_scan() {
        let router = WorkloadRouter::new(RouterConfig::default());
        let chars = QueryCharacteristics::full_scan("products", 50_000);

        let decision = router.route("q3", &chars).await;

        assert_eq!(decision.workload_type, WorkloadType::OLAP);
        assert!(decision.use_olap);
    }

    #[tokio::test]
    async fn test_route_by_row_count() {
        let router = WorkloadRouter::new(RouterConfig::default());

        // Below threshold - OLTP
        let chars_small = QueryCharacteristics {
            estimated_rows: Some(5_000),
            is_range_scan: true,
            ..Default::default()
        };
        let decision_small = router.route("q4", &chars_small).await;
        assert_eq!(decision_small.workload_type, WorkloadType::OLTP);

        // Above threshold - OLAP
        let chars_large = QueryCharacteristics {
            estimated_rows: Some(50_000),
            is_range_scan: true,
            ..Default::default()
        };
        let decision_large = router.route("q5", &chars_large).await;
        assert_eq!(decision_large.workload_type, WorkloadType::OLAP);
    }

    #[tokio::test]
    async fn test_adaptive_learning() {
        let router = WorkloadRouter::new(RouterConfig {
            adaptive_learning: true,
            ..Default::default()
        });

        let chars = QueryCharacteristics {
            estimated_rows: Some(5_000), // Below threshold
            ..Default::default()
        };

        // Record several OLAP executions with better performance
        for _ in 0..10 {
            router.record_execution("q6", true, 10).await; // OLAP: 10ms
            router.record_execution("q6", false, 100).await; // OLTP: 100ms
        }

        // Should now route to OLAP based on history
        let decision = router.route("q6", &chars).await;
        assert!(decision.reason.contains("Adaptive learning"));
        assert!(decision.use_olap);
    }

    #[tokio::test]
    async fn test_router_stats() {
        let router = WorkloadRouter::new(RouterConfig::default());

        // Route several queries
        router
            .route("q1", &QueryCharacteristics::point_lookup("t"))
            .await;
        router
            .route("q2", &QueryCharacteristics::point_lookup("t"))
            .await;
        router
            .route("q3", &QueryCharacteristics::aggregation("t", 100_000))
            .await;

        let stats = router.stats();
        assert_eq!(stats.oltp_queries, 2);
        assert_eq!(stats.olap_queries, 1);
        assert!((stats.olap_percentage() - 33.33).abs() < 1.0);
    }

    #[test]
    fn test_routing_decision_builders() {
        let oltp = LegacyHtapRoutingDecision::oltp("Test OLTP");
        assert!(!oltp.use_olap);
        assert_eq!(oltp.workload_type, WorkloadType::OLTP);

        let olap = LegacyHtapRoutingDecision::olap("Test OLAP");
        assert!(olap.use_olap);
        assert_eq!(olap.workload_type, WorkloadType::OLAP);

        let hybrid = LegacyHtapRoutingDecision::hybrid("Test Hybrid", true);
        assert!(hybrid.use_olap);
        assert_eq!(hybrid.workload_type, WorkloadType::Hybrid);
    }
}
