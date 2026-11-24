//! # RDBMS Store
//!
//! Combines SST (OLTP row store) and a columnar OLAP store for HTAP workloads.
//!
//! ## HTAP Architecture (inspired by TiDB)
//!
//! - **SST Engine**: Row-oriented storage for OLTP (point queries, transactions)
//! - **Columnar engine**: Column-oriented storage for OLAP (analytics, aggregations)
//! - **Async Replication**: Changes flow from SST to the column store with configurable lag

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use crate::storage::traits::UnifiedStorageFormat;

use super::super::traits::{ModelType, StoreCapabilities};

/// Configuration for the RDBMS store
#[derive(Debug, Clone)]
pub struct RDBMSStoreConfig {
    /// Maximum replication lag in nanoseconds before triggering sync
    pub max_replication_lag_ns: i64,
    /// Enable B-tree indexes
    pub enable_btree_indexes: bool,
    /// Enable constraint enforcement
    pub enable_constraints: bool,
    /// OLAP query threshold (rows) - queries above this use the column store
    pub olap_query_threshold: usize,
}

impl Default for RDBMSStoreConfig {
    fn default() -> Self {
        Self {
            max_replication_lag_ns: 1_000_000_000, // 1 second
            enable_btree_indexes: true,
            enable_constraints: true,
            olap_query_threshold: 10_000,
        }
    }
}

/// Query type for routing to appropriate engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Point query or small scan - use SST
    OLTP,
    /// Analytics/aggregation - use the column store
    OLAP,
    /// Unknown - analyze query to decide
    Unknown,
}

/// RDBMSStore implements HTAP with SST (OLTP) + columnar (OLAP) separation
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────┐
/// │                    RDBMSStore                            │
/// │  ┌───────────────────────────────────────────────────┐  │
/// │  │                Query Router                        │  │
/// │  │  - Analyze query type (OLTP vs OLAP)              │  │
/// │  │  - Route to appropriate engine                     │  │
/// │  └───────────────────────────────────────────────────┘  │
/// │                  │                   │                   │
/// │    ┌─────────────▼─────────┐   ┌────▼──────────────┐   │
/// │    │     SST Engine        │   │  Columnar Engine   │   │
/// │    │   (OLTP - Row Store)  │   │ (OLAP - Columnar) │   │
/// │    │  - Point queries      │   │ - Analytics        │   │
/// │    │  - Transactions       │   │ - Aggregations     │   │
/// │    │  - Low latency        │   │ - High throughput  │   │
/// │    └─────────────┬─────────┘   └────────────────────┘   │
/// │                  │                       ▲               │
/// │                  │   Async Replication   │               │
/// │                  └───────────────────────┘               │
/// └─────────────────────────────────────────────────────────┘
/// ```
pub struct RDBMSStore {
    /// Row-oriented engine for OLTP (SST)
    row_store: Option<Arc<dyn UnifiedStorageFormat>>,
    /// Column-oriented engine for OLAP
    column_store: Option<Arc<dyn UnifiedStorageFormat>>,
    /// Current replication lag in nanoseconds
    replication_lag_ns: AtomicI64,
    /// Configuration
    config: RDBMSStoreConfig,
}

impl RDBMSStore {
    /// Create a new RDBMSStore with the given configuration
    pub fn new(config: RDBMSStoreConfig) -> Self {
        Self {
            row_store: None,
            column_store: None,
            replication_lag_ns: AtomicI64::new(0),
            config,
        }
    }

    /// Set the SST engine for OLTP
    pub fn with_row_store(mut self, engine: Arc<dyn UnifiedStorageFormat>) -> Self {
        self.row_store = Some(engine);
        self
    }

    /// Set the columnar engine for OLAP
    pub fn with_column_store(mut self, engine: Arc<dyn UnifiedStorageFormat>) -> Self {
        self.column_store = Some(engine);
        self
    }

    /// Get store capabilities
    pub fn capabilities(&self) -> StoreCapabilities {
        StoreCapabilities {
            model_type: ModelType::Relational,
            supports_transactions: true,      // SST supports transactions
            supports_secondary_indexes: true, // B-tree indexes
            supports_acid: true,
            supports_streaming: true,
            max_recommended_records: Some(1_000_000_000), // 1B rows
            description: "HTAP relational storage with SST (OLTP) + columnar (OLAP)".to_string(),
        }
    }

    /// Determine query type for routing
    pub fn analyze_query(&self, estimated_rows: Option<usize>) -> QueryType {
        // Simple heuristic: if query touches many rows, use OLAP
        if let Some(rows) = estimated_rows
            && rows > self.config.olap_query_threshold
        {
            return QueryType::OLAP;
        }
        QueryType::OLTP
    }

    /// Get current replication lag
    pub fn replication_lag_ns(&self) -> i64 {
        self.replication_lag_ns.load(Ordering::Relaxed)
    }

    /// Check if OLAP store is sufficiently caught up
    pub fn is_olap_fresh(&self) -> bool {
        self.replication_lag_ns() <= self.config.max_replication_lag_ns
    }

    /// Get the row store (SST) engine
    pub fn row_store(&self) -> Option<&Arc<dyn UnifiedStorageFormat>> {
        self.row_store.as_ref()
    }

    /// Get the column store engine
    pub fn column_store(&self) -> Option<&Arc<dyn UnifiedStorageFormat>> {
        self.column_store.as_ref()
    }

    /// Get the primary engine for writes (row store)
    pub fn primary_engine(&self) -> Option<&Arc<dyn UnifiedStorageFormat>> {
        self.row_store.as_ref().or(self.column_store.as_ref())
    }

    /// Route to appropriate engine based on query type
    pub fn route_engine(&self, query_type: QueryType) -> Option<&Arc<dyn UnifiedStorageFormat>> {
        match query_type {
            QueryType::OLTP => self.row_store.as_ref().or(self.column_store.as_ref()),
            QueryType::OLAP => {
                if self.is_olap_fresh() {
                    self.column_store.as_ref().or(self.row_store.as_ref())
                } else {
                    // Fall back to row store if OLAP is too far behind
                    self.row_store.as_ref().or(self.column_store.as_ref())
                }
            }
            QueryType::Unknown => {
                // Default to row store
                self.row_store.as_ref().or(self.column_store.as_ref())
            }
        }
    }

    /// Get configuration
    pub fn config(&self) -> &RDBMSStoreConfig {
        &self.config
    }

    /// Check if store is operational
    pub fn is_operational(&self) -> bool {
        self.row_store.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdbms_store_config_default() {
        let config = RDBMSStoreConfig::default();
        assert_eq!(config.max_replication_lag_ns, 1_000_000_000);
        assert!(config.enable_btree_indexes);
        assert!(config.enable_constraints);
    }

    #[test]
    fn test_rdbms_store_capabilities() {
        let store = RDBMSStore::new(RDBMSStoreConfig::default());
        let caps = store.capabilities();

        assert_eq!(caps.model_type, ModelType::Relational);
        assert!(caps.supports_transactions);
        assert!(caps.supports_acid);
    }

    #[test]
    fn test_query_type_routing() {
        let store = RDBMSStore::new(RDBMSStoreConfig::default());

        // Small queries go to OLTP
        assert_eq!(store.analyze_query(Some(100)), QueryType::OLTP);

        // Large queries go to OLAP
        assert_eq!(store.analyze_query(Some(50_000)), QueryType::OLAP);

        // Unknown defaults to OLTP
        assert_eq!(store.analyze_query(None), QueryType::OLTP);
    }
}
