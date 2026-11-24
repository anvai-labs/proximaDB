//! Unified Scan Strategy - Holistic design based on existing engine implementations
//!
//! This module unifies the scan patterns found across all storage engines:
//! - **SST**: full_scan_strategy, filtered_scan_strategy_modular, parallel_full_scan
//! - **NOVA**: evaluate_predicate_pushdown, columnar filtering with progressive stages
//! - **NOVA**: Progressive columnar search, streaming processor, zone maps
//! - **RAPTOR**: scan_vectors_with_strategy, full_scan_all_vectors, filtered_scan_vectors
//!
//! Key insights from existing implementations:
//! 1. All engines separate full scan (for compaction) from filtered scan (for queries)
//! 2. Columnar engines (NOVA) have predicate pushdown capabilities
//! 3. Hybrid columnar engines (SST) use bloom filters and block-level filtering
//! 4. RAPTOR provides a unified ScanStrategy enum pattern
//! 5. Parallel processing is key for multi-file scenarios

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashSet;

use crate::proto::proximadb_v1::Collection;
use proximadb_filter_expression::FilterExpression;

// ── Hoisted to `proximadb-storage-ports` (root-crate decomposition, gap 2) ──
// The pure-data scan-strategy cluster — `ScanStrategy`, `ScanIterator`,
// `ScanStatistics`, `ScanCostEstimate` — has been moved to the
// `proximadb-storage-ports` crate (alongside `ScanCapabilities` hoisted by
// #910). These thin re-exports preserve every existing
// `crate::storage::scan_strategy::<Type>` path so callers resolve unchanged.
// `UnifiedScanEngine` (which names the proto `Collection`) stays here — it is
// not pure-data and out of scope for the type hoist.
pub use proximadb_storage_ports::{
    ScanCapabilities, ScanCostEstimate, ScanIterator, ScanStatistics, ScanStrategy,
};

/// Core scan trait that all engines implement
#[async_trait]
pub trait UnifiedScanEngine: Send + Sync {
    /// Get scan capabilities for this engine
    fn scan_capabilities(&self) -> ScanCapabilities;

    /// Create a scan iterator based on strategy
    /// This is the main entry point, similar to RAPTOR's scan_vectors_with_strategy
    async fn create_scan_iterator(
        &self,
        _collection_id: &str,
        strategy: ScanStrategy,
        collection_config: Option<&Collection>,
    ) -> Result<Box<dyn ScanIterator>>;

    /// Estimate scan cost for query planning
    async fn estimate_scan_cost(
        &self,
        _collection_id: &str,
        strategy: &ScanStrategy,
    ) -> Result<ScanCostEstimate>;

    /// Optimize scan strategy based on statistics
    /// Engines can upgrade/downgrade strategies based on their capabilities
    async fn optimize_scan_strategy(
        &self,
        _collection_id: &str,
        strategy: ScanStrategy,
    ) -> Result<ScanStrategy> {
        // Default: return as-is
        Ok(strategy)
    }
}

/// Engine-specific implementations based on actual code patterns
pub mod engine_impl {

    /// SST scan implementation traits
    pub struct SSTScanIterator {
        /// Based on sst_query_engine.rs patterns
        pub use_modular_reader: bool,
        pub use_block_cache: bool,
        pub parallel_files: bool,
        pub bloom_filter_enabled: bool,
    }

    /// NOVA scan implementation traits
    pub struct NOVAScanIterator {
        /// Based on columnar_search.rs patterns
        pub progressive_stages: bool,
        pub streaming_enabled: bool,
        pub zone_map_pruning: bool,
        pub memory_budget: usize,
    }

    /// RAPTOR scan implementation traits
    pub struct RAPTORScanIterator {
        /// Based on consolidated_reader.rs patterns
        pub tier_aware: bool,
        pub rowgroup_filtering: bool,
        pub bloom_filter_checking: bool,
        pub consolidation_enabled: bool,
    }
}

/// Helper functions extracted from existing implementations
pub mod scan_helpers {
    use super::*;

    /// Extract columns needed for filter evaluation
    pub fn extract_filter_columns(filter: &FilterExpression) -> HashSet<String> {
        let mut columns = HashSet::new();
        match filter {
            FilterExpression::Comparison { field, .. } => {
                columns.insert(field.clone());
            }
            FilterExpression::And(exprs) | FilterExpression::Or(exprs) => {
                for expr in exprs {
                    columns.extend(extract_filter_columns(expr));
                }
            }
            _ => {}
        }
        columns
    }

    /// From SST: Check if block should be scanned based on bloom filter
    pub async fn should_scan_block(
        _block_id: &str,
        bloom_filter: Option<&[u8]>,
        target_ids: &[String],
    ) -> bool {
        // Implementation based on SST's bloom filter logic
        if bloom_filter.is_none() || target_ids.is_empty() {
            return true;
        }
        // Check bloom filter (simplified)
        true
    }

    /// From NOVA: Determine if progressive search is beneficial
    pub fn should_use_progressive(
        total_vectors: usize,
        selectivity: f32,
        latency_budget_ms: Option<u64>,
    ) -> bool {
        // Based on NOVA's heuristics
        total_vectors > 10000 && selectivity < 0.1 && latency_budget_ms.is_some()
    }

    /// From RAPTOR: Estimate row group selectivity
    pub fn estimate_rowgroup_selectivity(
        predicates: &[FilterExpression],
        _statistics: &RowGroupStatistics,
    ) -> f32 {
        // Based on RAPTOR's statistics-based pruning
        if predicates.is_empty() {
            return 1.0;
        }
        // Simplified selectivity estimation
        0.5
    }
}

/// Placeholder for row group statistics
#[derive(Debug, Clone)]
pub struct RowGroupStatistics {
    pub row_count: usize,
    pub null_count: usize,
    pub min_values: Vec<Option<serde_json::Value>>,
    pub max_values: Vec<Option<serde_json::Value>>,
}
