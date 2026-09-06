//! Search module for ProximaDB storage-aware search implementations

// Foundation search-result / filter-value types extracted to the
// `proximadb-search-types` crate (root-crate decomposition, Slice D / D1).
// Re-exported here so every existing `crate::core::search::{results,
// bounded_queue, sql_value_filter, json_value_serde, json_comparison}::*`
// path resolves unchanged. The non-extractable orchestration types
// (`UnifiedSearchParams`, the search engines) remain in this module.
pub use proximadb_search_types::{
    block_prune::BlockPruneMode, bounded_queue, compiled_filter::CompiledGlobalFilter,
    json_comparison, json_value_serde, results, sql_value_filter,
};

pub use proximadb_cross_modal_fusion::cross_modal_fusion;
pub mod engine_benchmarks;
pub mod filter_contract; // Filter contracts for hybrid search (Issue #38, SB-08)
pub mod filter_pushdown_engine;
pub mod fusion_route;
pub mod hybrid;
pub mod index_based_filter;
pub mod integrated_search_optimization;
pub mod merge;
pub mod metadata_filter_pushdown;
pub mod multi_tier_deduplication;
pub use proximadb_mvcc_resolution::mvcc_resolution;
pub mod progressive_quantization;
pub mod progressive_search_pipeline;
pub mod queries;
pub mod query_preprocessing;
pub mod rank;
pub mod search_interface;
pub mod smart_execution_strategy;
pub mod strategies;

#[cfg(test)]
mod early_termination_tests;
#[cfg(test)]
mod optimization_tests;

#[cfg(test)]
use std::collections::HashMap;

pub use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

// ── Hoisted to `proximadb-search-types` (root-crate decomposition, gap 1) ──
// The `UnifiedSearchParams` cluster — `UnifiedSearchParams`, `SearchParams`
// alias, `ProgressiveRecalls`, `SearchMode`, `SearchEffort`,
// `VectorFreshnessMode`, `HybridSearchMode`, `BlockPruneConfig`, and
// `FilterOptimizationHints` — has been moved to the `proximadb-search-types`
// crate. These re-exports preserve every existing
// `crate::core::search::<Type>` path so callers resolve unchanged. The
// conversion-function modules (`protocol_conversions`, `filter_extraction`)
// stay in this module — they reference proto types not needed in the data
// struct.
pub use proximadb_search_types::search_params::{
    BlockPruneConfig, DEFAULT_ADAPTIVE_VECTOR_THRESHOLD, FilterOptimizationHints, HybridSearchMode,
    LexicalQueryMode, ProgressiveRecalls, SearchEffort, SearchMode, UnifiedSearchParams,
    VectorFreshnessMode,
};
/// Backward-compatibility alias kept here so the long-standing
/// `crate::core::search::SearchParams` path continues to resolve.
pub type SearchParams = UnifiedSearchParams;

// Re-export main types
pub use multi_tier_deduplication::{
    DataFreshnessTier, DeduplicationStats, DeduplicationStorageEngine, MetadataFilter,
    MultiTierDeduplicator, TieredSearchCandidate,
};

// Filter types are already defined above, no need to re-export
pub use results::{
    EngineStats, OptimizedSearchRecord, QuantizationInfo, SearchDebugInfo, SearchResultSet,
};
// NOTE: Proto types (SearchResult, SearchVectorRecord) should NOT be re-exported here.
// They belong in the API layer only. Services should use OptimizedSearchRecord
// and convert to proto types at the API boundary.
pub use search_interface::{
    CollectionConfig, ColumnData, FilterableColumn, IntegratedSearchOptimizer, OptimizationHint,
    SearchPlan, StorageInfo, UnifiedSearchEngine,
};

// Provide a distinct export for the advanced optimizer to avoid name collisions
pub use integrated_search_optimization::{AdvancedSearchOptimizer, SearchOptimizer};

// Re-export search strategy pattern (Phase 2 of ISP compliance)
pub use strategies::{
    AdaptiveSearchStrategy, ApproximateSearchStrategy, CandidateProvider, ExactSearchStrategy,
    ScoredCandidate, SearchContext, SearchContextImpl, SearchCostEstimate, SearchStrategy,
    SearchStrategyRegistry,
};

/// Protocol Filter Conversion Utilities
///
/// This module provides conversion functions from protocol-specific filter types
/// to the unified FilterExpression type for consistent handling across all APIs
pub mod protocol_conversions {
    use crate::core::search::{ComparisonOperator, FilterExpression};
    use serde_json::Value;

    /// Convert gRPC proto MetadataFilter to unified FilterExpression
    /// Used by gRPC handlers to convert incoming proto filters
    pub fn from_proto_metadata_filter(
        proto_filter: &crate::proto::proximadb_v1::MetadataFilter,
    ) -> Result<FilterExpression, String> {
        if proto_filter.clauses.is_empty() {
            return Ok(FilterExpression::And(vec![]));
        }

        let conditions: Result<Vec<FilterExpression>, String> = proto_filter
            .clauses
            .iter()
            .map(|condition| {
                let field = condition.field.clone();
                let value = match &condition.value {
                    Some(v) => {
                        // Create parent FilterClause to use custom serde implementation
                        let filter_clause = crate::proto::proximadb_v1::FilterClause {
                            field: condition.field.clone(),
                            op: condition.op,
                            value: Some(v.clone()),
                        };
                        serde_json::to_value(&filter_clause).map_err(|e| e.to_string())?
                    }
                    None => return Err("Missing value in filter condition".to_string()),
                };

                let operator =
                    match crate::proto::proximadb_v1::ComparisonOp::try_from(condition.op) {
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Eq) => {
                            ComparisonOperator::Equals
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Ne) => {
                            ComparisonOperator::NotEquals
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Gt) => {
                            ComparisonOperator::GreaterThan
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Gte) => {
                            ComparisonOperator::GreaterThanOrEqual
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Lt) => {
                            ComparisonOperator::LessThan
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Lte) => {
                            ComparisonOperator::LessThanOrEqual
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::In) => ComparisonOperator::In,
                        Ok(crate::proto::proximadb_v1::ComparisonOp::NotIn) => {
                            ComparisonOperator::NotIn
                        }
                        Ok(crate::proto::proximadb_v1::ComparisonOp::Contains) => {
                            ComparisonOperator::Contains
                        }
                        _ => {
                            return Err(format!(
                                "Unknown proto filter operation: {}",
                                condition.op
                            ));
                        }
                    };

                Ok(FilterExpression::Comparison {
                    field,
                    operator,
                    value,
                })
            })
            .collect();

        match conditions {
            Ok(conds) => {
                if conds.len() == 1 {
                    conds.into_iter().next().ok_or_else(|| {
                        "Internal error: expected one converted filter condition".to_string()
                    })
                } else {
                    // Use AND logic by default for multiple conditions
                    Ok(FilterExpression::And(conds))
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Convert REST JSON filter to unified FilterExpression
    /// Used by REST handlers to convert JSON filter objects
    pub fn from_rest_json_filter(
        json_filter: &serde_json::Value,
    ) -> Result<FilterExpression, String> {
        match json_filter {
            Value::Object(obj) => {
                // Handle different REST filter formats
                if let Some(conditions) = obj.get("conditions") {
                    // Array of conditions with logic operator
                    if let Value::Array(cond_array) = conditions {
                        let logic = obj.get("logic").and_then(|v| v.as_str()).unwrap_or("and");
                        let expressions: Result<Vec<FilterExpression>, String> =
                            cond_array.iter().map(parse_rest_condition).collect();

                        match expressions {
                            Ok(exprs) => {
                                if logic == "or" {
                                    Ok(FilterExpression::Or(exprs))
                                } else {
                                    Ok(FilterExpression::And(exprs))
                                }
                            }
                            Err(e) => Err(e),
                        }
                    } else {
                        Err("conditions must be an array".to_string())
                    }
                } else {
                    // Single condition object
                    parse_rest_condition(json_filter)
                }
            }
            _ => Err("Filter must be an object".to_string()),
        }
    }

    /// Parse a single REST condition into FilterExpression
    fn parse_rest_condition(condition: &Value) -> Result<FilterExpression, String> {
        if let Value::Object(obj) = condition {
            let field = obj
                .get("field")
                .and_then(|v| v.as_str())
                .ok_or("Missing field name")?
                .to_string();

            let operator = obj
                .get("operator")
                .and_then(|v| v.as_str())
                .ok_or("Missing operator")?;

            let value = obj.get("value").ok_or("Missing value")?.clone();

            let op = match operator {
                "eq" | "equals" => ComparisonOperator::Equals,
                "ne" | "not_equals" => ComparisonOperator::NotEquals,
                "gt" | "greater_than" => ComparisonOperator::GreaterThan,
                "gte" | "greater_than_or_equal" => ComparisonOperator::GreaterThanOrEqual,
                "lt" | "less_than" => ComparisonOperator::LessThan,
                "lte" | "less_than_or_equal" => ComparisonOperator::LessThanOrEqual,
                "in" => ComparisonOperator::In,
                "not_in" => ComparisonOperator::NotIn,
                "contains" => ComparisonOperator::Contains,
                "starts_with" => ComparisonOperator::StartsWith,
                "ends_with" => ComparisonOperator::EndsWith,
                "between" => ComparisonOperator::Between,
                "is_null" => ComparisonOperator::IsNull,
                "is_not_null" => ComparisonOperator::IsNotNull,
                "like" => ComparisonOperator::Like,
                _ => return Err(format!("Unknown operator: {}", operator)),
            };

            Ok(FilterExpression::Comparison {
                field,
                operator: op,
                value,
            })
        } else {
            Err("Condition must be an object".to_string())
        }
    }

    /// Convert legacy HashMap filters to unified FilterExpression
    /// Used for backward compatibility with existing filter formats
    pub fn from_legacy_hashmap_filter(
        filters: &std::collections::HashMap<String, serde_json::Value>,
    ) -> FilterExpression {
        if filters.is_empty() {
            return FilterExpression::And(vec![]);
        }

        let conditions: Vec<FilterExpression> = filters
            .iter()
            .map(|(key, value)| FilterExpression::Comparison {
                field: key.clone(),
                operator: ComparisonOperator::Equals,
                value: serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
            })
            .collect();

        if conditions.len() == 1 {
            conditions
                .into_iter()
                .next()
                .unwrap_or(FilterExpression::And(Vec::new()))
        } else {
            FilterExpression::And(conditions)
        }
    }

    /// Convert v1 simple filters (map<string, SqlValue>) to FilterExpression
    pub fn from_v1_simple_filters(
        filters: &std::collections::HashMap<String, crate::proto::proximadb_v1::SqlValue>,
    ) -> Result<FilterExpression, String> {
        if filters.is_empty() {
            return Ok(FilterExpression::And(vec![]));
        }

        let conditions: Result<Vec<FilterExpression>, String> = filters
            .iter()
            .map(|(field, sql_value)| {
                // The canonical FILTER LOWERING (same function the stored
                // side renders through) — a literal built any other way can
                // never equal the stored rendering.
                static NULL_SENTINEL: crate::proto::proximadb_v1::sql_value::Value =
                    crate::proto::proximadb_v1::sql_value::Value::NullValue(0);
                let value = proximadb_search_types::sql_value_filter::sql_val_to_json(
                    sql_value.value.as_ref().unwrap_or(&NULL_SENTINEL),
                );

                Ok(FilterExpression::Comparison {
                    field: field.clone(),
                    operator: ComparisonOperator::Equals,
                    value,
                })
            })
            .collect();

        let conditions = conditions?;
        if conditions.len() == 1 {
            conditions.into_iter().next().ok_or_else(|| {
                "Internal error: expected one v1 simple filter condition".to_string()
            })
        } else {
            Ok(FilterExpression::And(conditions))
        }
    }

    /// Convert v1 metadata filters (MetadataFilter from entity.proto) to FilterExpression  
    pub fn from_v1_metadata_filter(
        metadata_filter: &crate::proto::proximadb_v1::MetadataFilter,
    ) -> Result<FilterExpression, String> {
        if metadata_filter.clauses.is_empty() {
            return Ok(FilterExpression::And(vec![]));
        }

        let conditions: Result<Vec<FilterExpression>, String> = metadata_filter
            .clauses
            .iter()
            .map(|clause| {
                let value = match &clause.value {
                    Some(crate::proto::proximadb_v1::filter_clause::Value::StringValue(s)) => {
                        serde_json::Value::String(s.clone())
                    }
                    Some(crate::proto::proximadb_v1::filter_clause::Value::IntValue(i)) => {
                        serde_json::json!(*i)
                    }
                    Some(crate::proto::proximadb_v1::filter_clause::Value::DoubleValue(d)) => {
                        serde_json::json!(*d)
                    }
                    Some(crate::proto::proximadb_v1::filter_clause::Value::BoolValue(b)) => {
                        serde_json::Value::Bool(*b)
                    }
                    None => serde_json::Value::Null,
                };

                let operator = match crate::proto::proximadb_v1::ComparisonOp::try_from(clause.op) {
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Eq) => ComparisonOperator::Equals,
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Ne) => {
                        ComparisonOperator::NotEquals
                    }
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Gt) => {
                        ComparisonOperator::GreaterThan
                    }
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Gte) => {
                        ComparisonOperator::GreaterThanOrEqual
                    }
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Lt) => {
                        ComparisonOperator::LessThan
                    }
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Lte) => {
                        ComparisonOperator::LessThanOrEqual
                    }
                    Ok(crate::proto::proximadb_v1::ComparisonOp::In) => ComparisonOperator::In,
                    Ok(crate::proto::proximadb_v1::ComparisonOp::NotIn) => {
                        ComparisonOperator::NotIn
                    }
                    Ok(crate::proto::proximadb_v1::ComparisonOp::Contains) => {
                        ComparisonOperator::Contains
                    }
                    _ => return Err(format!("Unsupported comparison operator: {}", clause.op)),
                };

                Ok(FilterExpression::Comparison {
                    field: clause.field.clone(),
                    operator,
                    value,
                })
            })
            .collect();

        let conditions = conditions?;
        match crate::proto::proximadb_v1::LogicalOp::try_from(metadata_filter.op) {
            Ok(crate::proto::proximadb_v1::LogicalOp::And) => {
                if conditions.len() == 1 {
                    conditions.into_iter().next().ok_or_else(|| {
                        "Internal error: expected one metadata filter condition".to_string()
                    })
                } else {
                    Ok(FilterExpression::And(conditions))
                }
            }
            Ok(crate::proto::proximadb_v1::LogicalOp::Or) => {
                if conditions.len() == 1 {
                    conditions.into_iter().next().ok_or_else(|| {
                        "Internal error: expected one metadata filter condition".to_string()
                    })
                } else {
                    Ok(FilterExpression::Or(conditions))
                }
            }
            _ => {
                // Default to AND for unspecified
                if conditions.len() == 1 {
                    conditions.into_iter().next().ok_or_else(|| {
                        "Internal error: expected one metadata filter condition".to_string()
                    })
                } else {
                    Ok(FilterExpression::And(conditions))
                }
            }
        }
    }
}

/// Centralized metadata filter extraction utilities
pub mod filter_extraction {
    use crate::core::search::{ComparisonOperator, FilterExpression};
    use std::collections::{HashMap, HashSet};

    /// Extract simple equality conditions from filter expressions
    ///
    /// This extracts field/value pairs from FilterExpression for efficient metadata filtering
    /// Used consistently across SST, VIPER, and Write Buffer engines
    ///
    /// # Examples
    /// ```rust,ignore
    /// use proximadb::core::search::{FilterExpression, ComparisonOperator};
    /// use proximadb::core::search::filter_extraction::extract_metadata_conditions;
    ///
    /// let filter = FilterExpression::Comparison {
    ///     field: "batch".to_string(),
    ///     operator: ComparisonOperator::Equals,
    ///     value: serde_json::json!(2),
    /// };
    /// let conditions = extract_metadata_conditions(&filter);
    /// assert_eq!(conditions.get(key), Some(&serde_json::json!(2)));
    /// ```
    pub fn extract_metadata_conditions(
        filter_expr: &FilterExpression,
    ) -> HashMap<String, serde_json::Value> {
        let mut conditions = HashMap::new();
        extract_conditions_recursive(filter_expr, &mut conditions);
        conditions
    }

    /// Recursively extract metadata conditions from filter expressions
    fn extract_conditions_recursive(
        expr: &FilterExpression,
        conditions: &mut HashMap<String, serde_json::Value>,
    ) {
        match expr {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                // Only extract equality conditions for metadata filtering
                if matches!(operator, ComparisonOperator::Equals) {
                    conditions.insert(field.clone(), value.clone());
                }
            }
            FilterExpression::And(exprs) => {
                // For AND expressions, extract all conditions
                for expr in exprs {
                    extract_conditions_recursive(expr, conditions);
                }
            }
            FilterExpression::Or(_) | FilterExpression::Not(_) => {
                // OR and NOT expressions are too complex for simple metadata filtering
                // These will be handled by full expression evaluation
            }
        }
    }

    /// Extract column names referenced in a filter expression
    ///
    /// Used for determining which columns need to be loaded/indexed
    ///
    /// # Examples
    /// ```rust,ignore
    /// use proximadb::core::search::{FilterExpression, ComparisonOperator};
    /// use proximadb::core::search::filter_extraction::extract_filter_columns;
    ///
    /// let filter = FilterExpression::And(vec![
    ///     FilterExpression::Comparison {
    ///         field: "batch".to_string(),
    ///         operator: ComparisonOperator::Equals,
    ///         value: serde_json::json!(1),
    ///     },
    ///     FilterExpression::Comparison {
    ///         field: "category".to_string(),
    ///         operator: ComparisonOperator::Equals,
    ///         value: serde_json::json!("A"),
    ///     },
    /// ]);
    /// let columns = extract_filter_columns(&filter);
    /// assert!(columns.contains("batch"));
    /// assert!(columns.contains("category"));
    /// ```
    pub fn extract_filter_columns(expr: &FilterExpression) -> HashSet<String> {
        let mut columns = HashSet::new();
        extract_columns_recursive(expr, &mut columns);
        columns
    }

    /// Recursively extract column names from filter expressions
    fn extract_columns_recursive(expr: &FilterExpression, columns: &mut HashSet<String>) {
        match expr {
            FilterExpression::Comparison { field, .. } => {
                columns.insert(field.clone());
            }
            FilterExpression::And(exprs) | FilterExpression::Or(exprs) => {
                for expr in exprs {
                    extract_columns_recursive(expr, columns);
                }
            }
            FilterExpression::Not(expr) => {
                extract_columns_recursive(expr, columns);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── VectorFreshnessMode (Phase 5, Slice 5.1) ────────────────────────

    #[test]
    fn vector_freshness_mode_defaults_to_strong() {
        assert_eq!(VectorFreshnessMode::default(), VectorFreshnessMode::Strong);
    }

    #[test]
    fn unified_search_params_default_freshness_is_strong() {
        let params = UnifiedSearchParams::default();
        // Field unset on default — the safe default is provided by the accessor.
        assert!(params.freshness_mode.is_none());
        assert_eq!(
            params.effective_freshness_mode(),
            VectorFreshnessMode::Strong
        );
    }

    #[test]
    fn vector_freshness_mode_strong_requires_delta_merge() {
        assert!(VectorFreshnessMode::Strong.requires_delta_merge());
        assert!(
            VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 5_000,
            }
            .requires_delta_merge()
        );
        assert!(!VectorFreshnessMode::StaleOk.requires_delta_merge());
    }

    #[test]
    fn vector_freshness_mode_explain_label_is_stable() {
        assert_eq!(VectorFreshnessMode::Strong.explain_label(), "strong");
        assert_eq!(
            VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 1_000,
            }
            .explain_label(),
            "bounded_stale"
        );
        assert_eq!(VectorFreshnessMode::StaleOk.explain_label(), "stale_ok");
    }

    #[test]
    fn vector_freshness_mode_round_trips_through_json() {
        for mode in [
            VectorFreshnessMode::Strong,
            VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 2_500,
            },
            VectorFreshnessMode::StaleOk,
        ] {
            let json = serde_json::to_string(&mode).expect("serialize");
            let decoded: VectorFreshnessMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(decoded, mode);
        }
    }

    // ── should_scan_delta decision logic (Slice 5.3) ─────────────────────

    #[test]
    fn should_scan_delta_skips_for_stale_ok_regardless_of_lsns() {
        // StaleOk MUST never trigger a scan, even when the WAL has
        // newer data than the directory watermark.
        assert!(!VectorFreshnessMode::StaleOk.should_scan_delta(/*now*/ 100, /*wm*/ 10));
        assert!(!VectorFreshnessMode::StaleOk.should_scan_delta(0, 0));
        assert!(!VectorFreshnessMode::StaleOk.should_scan_delta(u64::MAX, 0));
    }

    #[test]
    fn should_scan_delta_skips_strong_when_watermark_matches_or_exceeds_lsn() {
        // Watermark already covers all committed writes — nothing to merge.
        assert!(!VectorFreshnessMode::Strong.should_scan_delta(/*now*/ 50, /*wm*/ 50));
        assert!(!VectorFreshnessMode::Strong.should_scan_delta(/*now*/ 50, /*wm*/ 60));
    }

    #[test]
    fn should_scan_delta_triggers_strong_when_wal_has_newer_records() {
        assert!(VectorFreshnessMode::Strong.should_scan_delta(/*now*/ 100, /*wm*/ 50));
        assert!(VectorFreshnessMode::Strong.should_scan_delta(/*now*/ 1, /*wm*/ 0));
    }

    #[test]
    fn should_scan_delta_scans_strong_when_lsn_tracking_is_zero() {
        // Reconciled 2026-05-28: when `current_lsn == 0` the global manifest
        // LSN allocator hasn't been advanced (e.g. v2 INSERT path skips the
        // manifest::append_* call). Returning false here would silently
        // hide memtable records from search. Strong/BoundedStale must scan;
        // StaleOk continues to skip.
        assert!(VectorFreshnessMode::Strong.should_scan_delta(/*now*/ 0, /*wm*/ 0));
        assert!(
            VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 5_000,
            }
            .should_scan_delta(0, 0)
        );
        assert!(!VectorFreshnessMode::StaleOk.should_scan_delta(0, 0));
    }

    #[test]
    fn should_scan_delta_treats_bounded_stale_like_strong_when_ns_unset() {
        // LSN-only entry point passes ns=0/0 → bound check is skipped,
        // BoundedStale falls back to Strong's behaviour.
        let mode = VectorFreshnessMode::BoundedStale {
            max_staleness_ms: 5_000,
        };
        assert!(mode.should_scan_delta(100, 50));
        assert!(!mode.should_scan_delta(50, 50));
    }

    // ── Slice 5.10 — should_scan_delta_with_time (BoundedStale bound) ────

    const MS_NS: i64 = 1_000_000;

    #[test]
    fn time_bound_stale_ok_always_skips_regardless_of_lsn_or_time() {
        let now = 10_000 * MS_NS;
        assert!(!VectorFreshnessMode::StaleOk.should_scan_delta_with_time(100, 50, 0, now));
        assert!(!VectorFreshnessMode::StaleOk.should_scan_delta_with_time(100, 50, now - 1, now));
    }

    #[test]
    fn time_bound_strong_ignores_time_and_uses_lsn_only() {
        // Even when the watermark is very recent, Strong still scans
        // when the WAL has newer LSNs.
        let now = 10_000 * MS_NS;
        let watermark_ns = now - 100 * MS_NS; // 100ms ago — very fresh
        assert!(VectorFreshnessMode::Strong.should_scan_delta_with_time(
            100,
            50,
            watermark_ns,
            now
        ));
        // And skips when LSN already covers (independent of time).
        assert!(!VectorFreshnessMode::Strong.should_scan_delta_with_time(
            50,
            50,
            watermark_ns,
            now
        ));
    }

    #[test]
    fn time_bound_bounded_stale_skips_within_bound() {
        // Watermark 2s ago, bound 5s → within bound, skip scan.
        let now = 10_000 * MS_NS;
        let watermark_ns = now - 2_000 * MS_NS;
        let mode = VectorFreshnessMode::BoundedStale {
            max_staleness_ms: 5_000,
        };
        assert!(!mode.should_scan_delta_with_time(100, 50, watermark_ns, now));
    }

    #[test]
    fn time_bound_bounded_stale_scans_beyond_bound() {
        // Watermark 10s ago, bound 5s → beyond bound, scan.
        let now = 10_000 * MS_NS;
        let watermark_ns = now - 10_000 * MS_NS;
        let mode = VectorFreshnessMode::BoundedStale {
            max_staleness_ms: 5_000,
        };
        assert!(mode.should_scan_delta_with_time(100, 50, watermark_ns, now));
    }

    #[test]
    fn time_bound_bounded_stale_skips_when_lsn_already_covers_regardless_of_age() {
        // LSN already covers — no scan needed even if the directory is
        // ancient. (Otherwise we'd scan an empty delta repeatedly.)
        let now = 10_000 * MS_NS;
        let watermark_ns = now - 1_000_000 * MS_NS; // ~16 minutes ago
        let mode = VectorFreshnessMode::BoundedStale {
            max_staleness_ms: 5_000,
        };
        assert!(!mode.should_scan_delta_with_time(50, 50, watermark_ns, now));
    }

    #[test]
    fn time_bound_bounded_stale_with_unset_ns_falls_back_to_lsn_only() {
        // watermark_ns == 0 means "time unknown" — conservatively scan
        // when LSNs disagree. Matches the writer's current placeholder.
        let mode = VectorFreshnessMode::BoundedStale {
            max_staleness_ms: 5_000,
        };
        let now = 10_000 * MS_NS;
        assert!(mode.should_scan_delta_with_time(100, 50, 0, now));
        // And when both are 0 (lsn-only entry-point shape), still scans
        // on LSN advance.
        assert!(mode.should_scan_delta_with_time(100, 50, 0, 0));
    }

    #[test]
    fn search_params_freshness_mode_round_trips_through_json() {
        let params = UnifiedSearchParams {
            freshness_mode: Some(VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 1_000,
            }),
            ..UnifiedSearchParams::default()
        };
        let json = serde_json::to_string(&params).expect("serialize");
        let decoded: UnifiedSearchParams = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            decoded.effective_freshness_mode(),
            VectorFreshnessMode::BoundedStale {
                max_staleness_ms: 1_000,
            }
        );
    }

    #[test]
    fn test_search_params_default() {
        let params = UnifiedSearchParams::default();

        // Core defaults
        assert!(params.query_vectors.is_none());
        assert!(params.vector.is_none());
        assert_eq!(params.top_k, Some(10));
        assert_eq!(
            params.distance_metric,
            Some(crate::compute::distance_computation::DistanceMetric::Cosine)
        );
        assert!(params.filter_expression.is_none());
        assert!(params.filters.is_none());
        assert_eq!(params.accuracy_threshold, Some(0.95));
        assert_eq!(params.include_expired, Some(false));
        assert_eq!(params.timeout_ms, Some(5000));
        assert_eq!(params.enable_two_stage, Some(true));

        // Execution pipeline defaults (disabled by default)
        assert_eq!(params.enable_vectorized_execution, Some(false));
        assert_eq!(params.enable_parallel_morsels, Some(false));
        assert_eq!(params.enable_pipeline_execution, Some(false));

        // Optimization defaults
        assert_eq!(params.enable_clustering_hint, Some(true));
        assert_eq!(params.enable_metadata_filtering_hint, Some(true));

        // Search mode defaults to cost-adaptive (TD-165) — exact when a full segment
        // scan is cheap, approximate otherwise. `Exact` is now an explicit opt-in.
        assert_eq!(
            params.search_mode,
            SearchMode::Adaptive {
                threshold: DEFAULT_ADAPTIVE_VECTOR_THRESHOLD
            }
        );
        // Hybrid mode defaults to VectorOnly
        assert_eq!(params.hybrid_mode, HybridSearchMode::VectorOnly);
    }

    #[test]
    fn test_search_params_with_filters() {
        let mut filters = HashMap::new();
        filters.insert("category".to_string(), serde_json::json!("electronics"));
        filters.insert("price".to_string(), serde_json::json!(99.99));

        let params = UnifiedSearchParams::default().with_simple_filters(filters);

        // Filter expression should be set
        assert!(params.filter_expression.is_some());

        // With 2 filters, the expression should be an And with 2 Comparison children
        match params.filter_expression.as_ref() {
            Some(FilterExpression::And(conditions)) => {
                assert_eq!(conditions.len(), 2);
                // Each child should be a Comparison with Equals operator
                for cond in conditions {
                    match cond {
                        FilterExpression::Comparison { operator, .. } => {
                            assert_eq!(*operator, ComparisonOperator::Equals);
                        }
                        _ => panic!("Expected Comparison inside And"),
                    }
                }
            }
            _ => panic!("Expected And expression with 2 conditions"),
        }

        // Empty filters should not set filter_expression
        let params_empty = UnifiedSearchParams::default().with_simple_filters(HashMap::new());
        assert!(params_empty.filter_expression.is_none());
    }

    #[test]
    fn test_search_mode_variants() {
        // Default is cost-adaptive (TD-165), not strict Exact.
        let default = SearchMode::default();
        assert_eq!(
            default,
            SearchMode::Adaptive {
                threshold: DEFAULT_ADAPTIVE_VECTOR_THRESHOLD
            }
        );
        assert!(!default.is_exact());

        // Approximate with auto nprobe
        let approx = SearchMode::approximate();
        assert_eq!(approx, SearchMode::Approximate { nprobe: None });
        assert!(!approx.is_exact());

        // Approximate with explicit nprobe
        let approx_np = SearchMode::approximate_with_nprobe(16);
        assert_eq!(approx_np, SearchMode::Approximate { nprobe: Some(16) });
        assert!(!approx_np.is_exact());

        // Adaptive with default threshold
        let adaptive = SearchMode::adaptive();
        assert_eq!(adaptive, SearchMode::Adaptive { threshold: 10_000 });
        assert!(!adaptive.is_exact());

        // effective_nprobe: Exact searches all partitions
        assert_eq!(SearchMode::Exact.effective_nprobe(100, 50_000), 100);

        // effective_nprobe: Approximate with explicit nprobe
        assert_eq!(
            SearchMode::Approximate { nprobe: Some(8) }.effective_nprobe(100, 50_000),
            8
        );

        // effective_nprobe: Approximate auto = sqrt(num_partitions), at least 3
        let auto_nprobe = SearchMode::Approximate { nprobe: None }.effective_nprobe(100, 50_000);
        assert_eq!(auto_nprobe, 10); // sqrt(100) = 10

        // effective_nprobe: Adaptive below threshold uses exact
        assert_eq!(
            SearchMode::Adaptive { threshold: 10_000 }.effective_nprobe(100, 5_000),
            100
        );

        // effective_nprobe: Adaptive above threshold uses approximate
        let adaptive_above =
            SearchMode::Adaptive { threshold: 10_000 }.effective_nprobe(100, 50_000);
        assert_eq!(adaptive_above, 10); // sqrt(100) = 10
    }

    #[test]
    fn test_search_mode_to_search_effort() {
        // Exact maps to Exact effort (keeps the index recall-maximizing default).
        assert_eq!(
            SearchMode::Exact.to_search_effort(),
            Some(SearchEffort::Exact)
        );

        // Approximate forwards the explicit/auto nprobe as the effort hint.
        assert_eq!(
            SearchMode::Approximate { nprobe: Some(8) }.to_search_effort(),
            Some(SearchEffort::Approximate { hint: Some(8) })
        );
        assert_eq!(
            SearchMode::Approximate { nprobe: None }.to_search_effort(),
            Some(SearchEffort::Approximate { hint: None })
        );

        // Adaptive yields no per-query override (index self-adapts to N).
        assert_eq!(
            SearchMode::Adaptive { threshold: 10_000 }.to_search_effort(),
            None
        );
    }

    #[test]
    fn test_search_effort_hnsw_ef_override() {
        let top_k = 10;

        // Exact keeps the index default (no override) so the warm path is
        // byte-identical to pre-knob behavior.
        assert_eq!(SearchEffort::Exact.hnsw_ef_override(top_k), None);

        // Explicit hint is used directly as ef, floored at top_k.
        assert_eq!(
            SearchEffort::Approximate { hint: Some(32) }.hnsw_ef_override(top_k),
            Some(32)
        );
        // A hint below top_k is floored to top_k (never return fewer than k).
        assert_eq!(
            SearchEffort::Approximate { hint: Some(3) }.hnsw_ef_override(top_k),
            Some(10)
        );

        // Auto-approximate uses a recall-trading ef below the exact ceiling.
        let auto = SearchEffort::Approximate { hint: None }
            .hnsw_ef_override(top_k)
            .unwrap();
        assert!(
            auto >= top_k && auto < 500,
            "auto ef {auto} should trade recall, below the 500 clamp"
        );
    }

    #[test]
    fn test_search_effort_ivf_nprobe() {
        let nlist = 100;
        // Exact probes all partitions.
        assert_eq!(SearchEffort::Exact.ivf_nprobe(nlist), 100);
        // Explicit hint is clamped to [1, nlist].
        assert_eq!(
            SearchEffort::Approximate { hint: Some(8) }.ivf_nprobe(nlist),
            8
        );
        assert_eq!(
            SearchEffort::Approximate { hint: Some(999) }.ivf_nprobe(nlist),
            100
        );
        assert_eq!(
            SearchEffort::Approximate { hint: Some(0) }.ivf_nprobe(nlist),
            1
        );
        // Auto = sqrt(nlist).
        assert_eq!(
            SearchEffort::Approximate { hint: None }.ivf_nprobe(nlist),
            10
        );
    }

    #[test]
    fn test_hybrid_search_mode_variants() {
        // Default is VectorOnly
        let default = HybridSearchMode::default();
        assert_eq!(default, HybridSearchMode::VectorOnly);

        // All variants can be constructed
        let vector_only = HybridSearchMode::VectorOnly;
        let keyword_only = HybridSearchMode::KeywordOnly;
        let hybrid = HybridSearchMode::Hybrid;
        let hybrid_custom = HybridSearchMode::HybridCustom { rrf_k: 120 };

        // Verify they are distinct
        assert_ne!(vector_only, keyword_only);
        assert_ne!(keyword_only, hybrid);
        assert_ne!(hybrid, hybrid_custom);

        // Verify custom parameter is stored
        match hybrid_custom {
            HybridSearchMode::HybridCustom { rrf_k } => assert_eq!(rrf_k, 120),
            _ => panic!("Expected HybridCustom variant"),
        }
    }

    #[test]
    fn test_block_prune_config_default() {
        let config = BlockPruneConfig::default();

        assert!(!config.force_exact);
        assert!(matches!(config.mode, BlockPruneMode::Sqrt));
        assert!((config.ratio - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.min_keep, 1);
        assert_eq!(config.max_keep, 0);
        assert!(config.min_blocks_override.is_none());

        // Verify other modes can be constructed
        let ratio_mode = BlockPruneMode::Ratio;
        let fixed_mode = BlockPruneMode::Fixed(50);
        assert!(matches!(ratio_mode, BlockPruneMode::Ratio));
        match fixed_mode {
            BlockPruneMode::Fixed(n) => assert_eq!(n, 50),
            _ => panic!("Expected Fixed variant"),
        }
    }

    #[test]
    fn test_filter_expression_construction() {
        // Comparison expression
        let comparison = FilterExpression::Comparison {
            field: "status".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!("active"),
        };
        match &comparison {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "status");
                assert_eq!(*operator, ComparisonOperator::Equals);
                assert_eq!(*value, serde_json::json!("active"));
            }
            _ => panic!("Expected Comparison"),
        }

        // AND expression
        let and_expr = FilterExpression::And(vec![
            FilterExpression::Comparison {
                field: "age".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: serde_json::json!(18),
            },
            FilterExpression::Comparison {
                field: "age".to_string(),
                operator: ComparisonOperator::LessThan,
                value: serde_json::json!(65),
            },
        ]);
        match &and_expr {
            FilterExpression::And(exprs) => assert_eq!(exprs.len(), 2),
            _ => panic!("Expected And"),
        }

        // OR expression
        let or_expr = FilterExpression::Or(vec![
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::json!("A"),
            },
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::json!("B"),
            },
        ]);
        match &or_expr {
            FilterExpression::Or(exprs) => assert_eq!(exprs.len(), 2),
            _ => panic!("Expected Or"),
        }

        // NOT expression
        let not_expr = FilterExpression::Not(Box::new(FilterExpression::Comparison {
            field: "deleted".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(true),
        }));
        assert!(matches!(not_expr, FilterExpression::Not(_)));

        // Verify all comparison operators
        let operators = vec![
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqual,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqual,
            ComparisonOperator::In,
            ComparisonOperator::NotIn,
            ComparisonOperator::Contains,
            ComparisonOperator::StartsWith,
            ComparisonOperator::EndsWith,
            ComparisonOperator::Between,
            ComparisonOperator::IsNull,
            ComparisonOperator::IsNotNull,
            ComparisonOperator::Like,
        ];
        assert_eq!(operators.len(), 15, "Expected 15 comparison operators");

        // Verify PartialEq works for filter expressions
        let expr1 = FilterExpression::Comparison {
            field: "x".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(1),
        };
        let expr2 = FilterExpression::Comparison {
            field: "x".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(1),
        };
        assert_eq!(expr1, expr2);
    }
}
