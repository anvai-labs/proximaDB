//! Canonical transport-neutral search DTOs for v2 and internal callers.
//!
//! These are the search-side companions to [`crate::rich_record`] (the write
//! DTOs). They live in the platform `proximadb-runtime` crate so that a
//! transport-neutral read port (`RecordSearchPort`) has a self-contained
//! contract — without forcing adapters (Arrow Flight, REST, gRPC) to import
//! root-internal module paths. The root `services::operations::vectors::legacy`
//! module re-exports them so existing callers compile unchanged (mirrors the
//! TD-104 `rich_record` relocation).

use std::collections::HashMap;

use proximadb_data_model::ProximaValue;
use proximadb_filter_expression::FilterExpression;
use proximadb_search_types::PredicateShortfall;
use proximadb_search_types::sql_value_filter::proxima_value_to_filter_literal;

/// Canonical rich search request for v2 and internal callers.
#[derive(Debug, Clone)]
pub struct RichSearchRequest {
    pub collection_id: String,
    pub query_vector: Vec<f32>,
    pub top_k: u32,
    pub filters: Vec<RichFilterCondition>,
}

/// Canonical rich search response for v2 and internal callers.
#[derive(Debug, Clone, Default)]
pub struct RichSearchResponse {
    pub results: Vec<RichSearchResult>,
    pub total_found: i64,
    pub collection_id: Option<String>,
    /// TD-064(a): predicate-aware shortfall — `Some(...)` when a filtered
    /// search returned fewer than the requested `top_k` after the
    /// WAL+AXIS+storage merge. First-class and always-on (NOT debug-gated):
    /// a silent `<top_k` under a tenant/RLS filter is fail-open, so the
    /// client must be able to tell "fewer than k match my filter" from "the
    /// engine returned my full top-k". Recomputed authoritatively against
    /// the final merged result so an AXIS-stage false positive is cleared.
    pub predicate_shortfall: Option<PredicateShortfall>,
}

#[derive(Debug, Clone)]
pub struct RichSearchResult {
    pub id: String,
    pub score: f64,
    pub similarity: Option<f32>,
    pub vector: Vec<f32>,
    pub props: HashMap<String, ProximaValue>,
    pub version: Option<u32>,
    pub timestamp: Option<i64>,
    pub source: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RichFilterCondition {
    pub field: String,
    pub operator: RichFilterOperator,
    pub value: ProximaValue,
    pub value_upper: Option<ProximaValue>,
    pub value_list: Vec<ProximaValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RichFilterOperator {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Between,
    In,
    NotIn,
    Contains,
    StartsWith,
    EndsWith,
}

/// Lower rich filter conditions to the canonical [`FilterExpression`] — the
/// single lowering shared by v2 REST handlers and internal callers (ADR-094:
/// moved here so the converged handlers are root-independent).
pub fn rich_filters_to_filter_expression(
    filters: &[RichFilterCondition],
) -> Option<FilterExpression> {
    use proximadb_filter_expression::ComparisonOperator;

    let mut conditions: Vec<FilterExpression> = Vec::new();
    for filter in filters {
        let field = filter.field.clone();
        match filter.operator {
            RichFilterOperator::Between => {
                conditions.push(FilterExpression::Comparison {
                    field: field.clone(),
                    operator: ComparisonOperator::GreaterThanOrEqual,
                    value: proxima_value_to_filter_literal(&filter.value),
                });
                if let Some(upper) = &filter.value_upper {
                    conditions.push(FilterExpression::Comparison {
                        field,
                        operator: ComparisonOperator::LessThanOrEqual,
                        value: proxima_value_to_filter_literal(upper),
                    });
                }
            }
            RichFilterOperator::In | RichFilterOperator::NotIn => {
                let values = if filter.value_list.is_empty() {
                    match &filter.value {
                        proximadb_data_model::ProximaValue::Array(values) => values.clone(),
                        value => vec![value.clone()],
                    }
                } else {
                    filter.value_list.clone()
                };
                let array = serde_json::Value::Array(
                    values.iter().map(proxima_value_to_filter_literal).collect(),
                );
                conditions.push(FilterExpression::Comparison {
                    field,
                    operator: if matches!(filter.operator, RichFilterOperator::In) {
                        ComparisonOperator::In
                    } else {
                        ComparisonOperator::NotIn
                    },
                    value: array,
                });
            }
            operator => {
                let operator = match operator {
                    RichFilterOperator::Eq => ComparisonOperator::Equals,
                    RichFilterOperator::Ne => ComparisonOperator::NotEquals,
                    RichFilterOperator::Gt => ComparisonOperator::GreaterThan,
                    RichFilterOperator::Gte => ComparisonOperator::GreaterThanOrEqual,
                    RichFilterOperator::Lt => ComparisonOperator::LessThan,
                    RichFilterOperator::Lte => ComparisonOperator::LessThanOrEqual,
                    RichFilterOperator::Contains => ComparisonOperator::Contains,
                    RichFilterOperator::StartsWith => ComparisonOperator::StartsWith,
                    RichFilterOperator::EndsWith => ComparisonOperator::EndsWith,
                    RichFilterOperator::Between
                    | RichFilterOperator::In
                    | RichFilterOperator::NotIn => unreachable!("handled above"),
                };
                conditions.push(FilterExpression::Comparison {
                    field,
                    operator,
                    value: proxima_value_to_filter_literal(&filter.value),
                });
            }
        }
    }

    match conditions.len() {
        0 => None,
        1 => conditions.into_iter().next(),
        _ => Some(FilterExpression::And(conditions)),
    }
}

