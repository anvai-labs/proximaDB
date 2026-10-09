//! Unified Filter Evaluator Module
//!
//! This module provides a comprehensive, thread-safe filter evaluation system
//! that can be used across all storage engines (SST, VIPER, HELIX, etc.).
//!
//! Features:
//! - Thread-safe evaluation with Send+Sync support
//! - All comparison operators from FilterExpression
//! - Automatic type conversions between String and serde_json::Value
//! - Performance optimizations with caching
//! - Consistent behavior across all engines

use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::search::sql_value_filter::evaluate_filter_resolved;
use proximadb_filter_expression::FilterExpression;

/// Thread-safe filter evaluator that can be shared across async tasks.
///
/// Holds the `FilterExpression` directly and evaluates every representation
/// through the single canonical operator-semantics seam
/// (`sql_value_filter::evaluate_filter_resolved`), so this engine-facing
/// evaluator behaves identically to the v2 / json / SqlValue paths. `In`/`NotIn`
/// list literals supplied as comma-separated strings are normalized up front
/// (preserving the prior convenience).
#[derive(Clone)]
pub struct UnifiedFilterEvaluator {
    filter: Arc<FilterExpression>,
}

impl UnifiedFilterEvaluator {
    /// Create a new evaluator from a FilterExpression
    pub fn new(expr: Option<&FilterExpression>) -> Result<Option<Self>> {
        Ok(expr.map(|expr| Self {
            filter: Arc::new(normalize_in_values(expr)),
        }))
    }

    /// Evaluate the filter against metadata with string values
    pub fn evaluate_strings(&self, metadata: &HashMap<String, String>) -> bool {
        let json_metadata = strings_to_json_map(metadata);
        evaluate_filter_resolved(&self.filter, &|field| json_metadata.get(field).cloned())
    }

    /// Evaluate the filter against metadata with JSON values
    pub fn evaluate(&self, metadata: &HashMap<String, Value>) -> bool {
        evaluate_filter_resolved(&self.filter, &|field| metadata.get(field).cloned())
    }

    /// Evaluate against proto metadata items
    pub fn evaluate_proto(&self, metadata: &[crate::proto::proximadb_v1::MetadataItem]) -> bool {
        let json_metadata = crate::core::proto_metadata_helper::proto_metadata_to_json(metadata);
        evaluate_filter_resolved(&self.filter, &|field| json_metadata.get(field).cloned())
    }

    /// Create a thread-safe closure for use in parallel operations
    pub fn as_closure(&self) -> Arc<dyn Fn(&HashMap<String, String>) -> bool + Send + Sync> {
        let evaluator = self.clone();
        Arc::new(move |metadata| evaluator.evaluate_strings(metadata))
    }

    /// Create a thread-safe closure for JSON metadata
    pub fn as_json_closure(&self) -> Arc<dyn Fn(&HashMap<String, Value>) -> bool + Send + Sync> {
        let evaluator = self.clone();
        Arc::new(move |metadata| evaluator.evaluate(metadata))
    }
}

/// Parse a string-valued metadata map into JSON values (JSON literal when
/// parseable, otherwise a JSON string) so it can flow through the canonical seam.
fn strings_to_json_map(metadata: &HashMap<String, String>) -> HashMap<String, Value> {
    metadata
        .iter()
        .map(|(k, v)| {
            let value = serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.clone()));
            (k.clone(), value)
        })
        .collect()
}

/// Normalize `In`/`NotIn` comparison literals supplied as a comma-separated
/// string into a JSON array (e.g. `"a, b"` → `["a","b"]`), preserving the prior
/// `CompiledFilter` convenience. All other expressions pass through unchanged.
fn normalize_in_values(expr: &FilterExpression) -> FilterExpression {
    use proximadb_filter_expression::ComparisonOperator;
    match expr {
        FilterExpression::And(exprs) => {
            FilterExpression::And(exprs.iter().map(normalize_in_values).collect())
        }
        FilterExpression::Or(exprs) => {
            FilterExpression::Or(exprs.iter().map(normalize_in_values).collect())
        }
        FilterExpression::Not(inner) => FilterExpression::Not(Box::new(normalize_in_values(inner))),
        FilterExpression::Comparison {
            field,
            operator,
            value,
        } => {
            let value = match operator {
                ComparisonOperator::In | ComparisonOperator::NotIn => match value {
                    Value::String(s) => Value::Array(
                        s.split(',')
                            .map(|item| Value::String(item.trim().to_string()))
                            .collect(),
                    ),
                    other => other.clone(),
                },
                _ => value.clone(),
            };
            FilterExpression::Comparison {
                field: field.clone(),
                operator: operator.clone(),
                value,
            }
        }
    }
}

/// Helper function to create a filter closure from an optional FilterExpression
/// This is the primary API for storage engines to use
pub fn create_filter_fn(
    expr: Option<&FilterExpression>,
) -> Option<Arc<dyn Fn(&HashMap<String, String>) -> bool + Send + Sync>> {
    UnifiedFilterEvaluator::new(expr)
        .ok()
        .flatten()
        .map(|evaluator| evaluator.as_closure())
}

/// Create a filter closure for JSON metadata
pub fn create_json_filter_fn(
    expr: Option<&FilterExpression>,
) -> Option<Arc<dyn Fn(&HashMap<String, Value>) -> bool + Send + Sync>> {
    UnifiedFilterEvaluator::new(expr)
        .ok()
        .flatten()
        .map(|evaluator| evaluator.as_json_closure())
}

/// Direct evaluation function for backward compatibility
pub fn evaluate_filter(expr: &FilterExpression, metadata: &HashMap<String, Value>) -> bool {
    evaluate_filter_resolved(&normalize_in_values(expr), &|field| {
        metadata.get(field).cloned()
    })
}

/// Evaluate filter with string metadata
pub fn evaluate_filter_strings(
    expr: &FilterExpression,
    metadata: &HashMap<String, String>,
) -> bool {
    UnifiedFilterEvaluator::new(Some(expr))
        .ok()
        .flatten()
        .is_some_and(|evaluator| evaluator.evaluate_strings(metadata))
}

/// Check if a field is filterable based on collection configuration
pub fn is_filterable_field(field: &str, filterable_columns: &[String]) -> bool {
    filterable_columns.contains(&field.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_filter_expression::ComparisonOperator;
    use serde_json::json;

    #[test]
    fn test_equals_filter() {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), json!("Alice"));
        metadata.insert("age".to_string(), json!(30));

        let expr = FilterExpression::Comparison {
            field: "name".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("Alice"),
        };

        assert!(evaluate_filter(&expr, &metadata));
    }

    #[test]
    fn test_numeric_comparison() {
        let mut metadata = HashMap::new();
        metadata.insert("score".to_string(), json!(85));

        let expr = FilterExpression::Comparison {
            field: "score".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: json!(80),
        };

        assert!(evaluate_filter(&expr, &metadata));
    }

    #[test]
    fn test_between_filter() {
        let mut metadata = HashMap::new();
        metadata.insert("age".to_string(), json!(25));

        let expr = FilterExpression::Comparison {
            field: "age".to_string(),
            operator: ComparisonOperator::Between,
            value: json!([20, 30]),
        };

        assert!(evaluate_filter(&expr, &metadata));
    }

    #[test]
    fn test_like_pattern() {
        let mut metadata = HashMap::new();
        metadata.insert("email".to_string(), json!("user@example.com"));

        let expr = FilterExpression::Comparison {
            field: "email".to_string(),
            operator: ComparisonOperator::Like,
            value: json!("%@example.%"),
        };

        assert!(evaluate_filter(&expr, &metadata));
    }

    #[test]
    fn test_thread_safe_evaluator() {
        let expr = FilterExpression::Comparison {
            field: "status".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("active"),
        };

        let evaluator = UnifiedFilterEvaluator::new(Some(&expr)).unwrap().unwrap();
        let closure = evaluator.as_closure();

        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), "active".to_string());

        assert!(closure(&metadata));
    }

    #[test]
    fn test_complex_and_filter() {
        let mut metadata = HashMap::new();
        metadata.insert("age".to_string(), json!(25));
        metadata.insert("city".to_string(), json!("NYC"));

        let expr = FilterExpression::And(vec![
            FilterExpression::Comparison {
                field: "age".to_string(),
                operator: ComparisonOperator::GreaterThanOrEqual,
                value: json!(21),
            },
            FilterExpression::Comparison {
                field: "city".to_string(),
                operator: ComparisonOperator::In,
                value: json!(["NYC", "LA", "SF"]),
            },
        ]);

        assert!(evaluate_filter(&expr, &metadata));
    }
}
