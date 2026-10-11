//! Pure document-query adaptation helpers shared across query surfaces.

use proximadb_data_model::DataModel;
use proximadb_document_query::PathFilter;
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_proto::proximadb_v1::{
    DocFilterCondition, DocFilterOperator, DocumentFilter, SqlObject, SqlValue,
    sql_value::Value as SqlValueVariant,
};
use proximadb_query_filter::{FilterOperator, FilterValue};

use crate::UnifiedRecord;

/// Convert a protobuf `SqlObject` into JSON.
pub fn sql_object_to_json(obj: &proximadb_proto::proximadb_v1::SqlObject) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (key, value) in &obj.fields {
        map.insert(key.clone(), sql_value_to_json(value));
    }
    serde_json::Value::Object(map)
}

use proximadb_records::conversions::sql_value_to_json;

/// Convert query-IR path filters into the protobuf `DocumentFilter` contract.
pub fn convert_path_filters_to_document_filter(filters: &[PathFilter]) -> Option<DocumentFilter> {
    if filters.is_empty() {
        return None;
    }

    let conditions: Vec<DocFilterCondition> = filters
        .iter()
        .map(|pf| {
            let operator = match pf.operator {
                FilterOperator::Eq => DocFilterOperator::Eq,
                FilterOperator::Ne => DocFilterOperator::Ne,
                FilterOperator::Gt => DocFilterOperator::Gt,
                FilterOperator::Gte => DocFilterOperator::Gte,
                FilterOperator::Lt => DocFilterOperator::Lt,
                FilterOperator::Lte => DocFilterOperator::Lte,
                FilterOperator::In => DocFilterOperator::In,
                FilterOperator::NotIn => DocFilterOperator::NotIn,
                FilterOperator::Contains => DocFilterOperator::Contains,
                FilterOperator::StartsWith => DocFilterOperator::Regex,
                FilterOperator::EndsWith => DocFilterOperator::Regex,
                FilterOperator::Exists => DocFilterOperator::Exists,
                FilterOperator::Type => DocFilterOperator::Type,
            };

            DocFilterCondition {
                path: pf.path.clone(),
                operator: operator.into(),
                value: Some(convert_filter_value_to_sql(&pf.value)),
                values: vec![],
            }
        })
        .collect();

    Some(DocumentFilter {
        conditions,
        or_filters: vec![],
        and_filters: vec![],
    })
}

/// Convert unified `FilterValue` into protobuf `SqlValue`.
pub fn convert_filter_value_to_sql(value: &FilterValue) -> SqlValue {
    match value {
        FilterValue::String(s) => SqlValue {
            value: Some(SqlValueVariant::StringValue(s.clone())),
        },
        FilterValue::Number(n) => SqlValue {
            value: Some(SqlValueVariant::NumberValue(*n)),
        },
        FilterValue::Bool(b) => SqlValue {
            value: Some(SqlValueVariant::BoolValue(*b)),
        },
        FilterValue::Null => SqlValue {
            value: Some(SqlValueVariant::NullValue(0)),
        },
        FilterValue::Array(arr) => {
            let sql_arr = proximadb_proto::proximadb_v1::SqlArray {
                values: arr.iter().map(convert_filter_value_to_sql).collect(),
            };
            SqlValue {
                value: Some(SqlValueVariant::ArrayValue(sql_arr)),
            }
        }
    }
}

/// Lower a protobuf `DocumentFilter` to the shared `FilterExpression` algebra.
///
/// This is the Phase 4 lowering bridge from the document-specific wire format
/// into the plan-level algebra used by `Operator::Filter` in `MultiModelPlan`.
/// The output can be pushed into a `Scan` operator's `filter` field or wrapped
/// in an explicit `Filter` operator for cross-model plan composition.
///
/// Semantics:
/// - `conditions` list is treated as implicit AND (all must match).
/// - `or_filters` are recursively lowered and combined with `FilterExpression::Or`.
/// - `and_filters` are recursively lowered and combined with `FilterExpression::And`.
/// - A mix of conditions + or_filters + and_filters is wrapped in an outer AND.
/// - Returns `None` for an empty filter (no conditions, no sub-filters).
pub fn document_filter_to_filter_expression(filter: &DocumentFilter) -> Option<FilterExpression> {
    let mut parts: Vec<FilterExpression> = Vec::new();

    // Lower each condition to a Comparison leaf.
    if !filter.conditions.is_empty() {
        let condition_exprs: Vec<FilterExpression> = filter
            .conditions
            .iter()
            .filter_map(doc_filter_condition_to_expression)
            .collect();
        match condition_exprs.len() {
            0 => {}
            1 => parts.push(condition_exprs.into_iter().next().unwrap()),
            _ => parts.push(FilterExpression::And(condition_exprs)),
        }
    }

    // Recursively lower OR sub-filters.
    if !filter.or_filters.is_empty() {
        let or_exprs: Vec<FilterExpression> = filter
            .or_filters
            .iter()
            .filter_map(document_filter_to_filter_expression)
            .collect();
        if !or_exprs.is_empty() {
            parts.push(FilterExpression::Or(or_exprs));
        }
    }

    // Recursively lower AND sub-filters.
    if !filter.and_filters.is_empty() {
        let and_exprs: Vec<FilterExpression> = filter
            .and_filters
            .iter()
            .filter_map(document_filter_to_filter_expression)
            .collect();
        if !and_exprs.is_empty() {
            parts.push(FilterExpression::And(and_exprs));
        }
    }

    match parts.len() {
        0 => None,
        1 => Some(parts.into_iter().next().unwrap()),
        _ => Some(FilterExpression::And(parts)),
    }
}

/// Lower one `DocFilterCondition` to a `FilterExpression::Comparison`.
fn doc_filter_condition_to_expression(cond: &DocFilterCondition) -> Option<FilterExpression> {
    let op = DocFilterOperator::try_from(cond.operator).unwrap_or(DocFilterOperator::Eq);

    let shared_op = match op {
        DocFilterOperator::Eq | DocFilterOperator::Unspecified => ComparisonOperator::Equals,
        DocFilterOperator::Ne => ComparisonOperator::NotEquals,
        DocFilterOperator::Gt => ComparisonOperator::GreaterThan,
        DocFilterOperator::Gte => ComparisonOperator::GreaterThanOrEqual,
        DocFilterOperator::Lt => ComparisonOperator::LessThan,
        DocFilterOperator::Lte => ComparisonOperator::LessThanOrEqual,
        DocFilterOperator::In => ComparisonOperator::In,
        DocFilterOperator::NotIn => ComparisonOperator::NotIn,
        DocFilterOperator::Contains | DocFilterOperator::Fulltext => ComparisonOperator::Contains,
        DocFilterOperator::Regex => ComparisonOperator::Like,
        DocFilterOperator::Exists => ComparisonOperator::IsNotNull,
        // Type check: approximate as Equals with the type string as value.
        DocFilterOperator::Type => ComparisonOperator::Equals,
    };

    // For IN/NOT_IN, use the multi-value list; otherwise use the single
    // value. Literals lower through the FILTER LOWERING (int-array bytes),
    // NOT the rendering converter (base64 bytes) — the lowering-vs-rendering
    // distinction the records doc forbids conflating; evaluators render the
    // stored side with the same lowering.
    let literal = proximadb_search_types::sql_value_filter::sql_value_to_filter_literal;
    let value = if !cond.values.is_empty() {
        let arr: Vec<serde_json::Value> = cond.values.iter().map(&literal).collect();
        serde_json::Value::Array(arr)
    } else {
        cond.value
            .as_ref()
            .map(literal)
            .unwrap_or(serde_json::Value::Null)
    };

    Some(FilterExpression::Comparison {
        field: cond.path.clone(),
        operator: shared_op,
        value,
    })
}

/// Build a unified record from a document query result.
pub fn build_document_query_record(id: &str, document: &SqlObject) -> UnifiedRecord {
    UnifiedRecord {
        id: id.to_string(),
        source_model: DataModel::Document,
        data: sql_object_to_json(document),
        score: None,
        metadata: std::collections::HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
    use proximadb_proto::proximadb_v1::{
        DocFilterCondition, DocFilterOperator, SqlObject, sql_value::Value,
    };

    #[test]
    fn sql_value_to_json_primitives() {
        let null_val = SqlValue {
            value: Some(Value::NullValue(0)),
        };
        assert_eq!(sql_value_to_json(&null_val), serde_json::Value::Null);

        let bool_val = SqlValue {
            value: Some(Value::BoolValue(true)),
        };
        assert_eq!(sql_value_to_json(&bool_val), serde_json::Value::Bool(true));

        let int_val = SqlValue {
            value: Some(Value::Int64Value(42)),
        };
        assert_eq!(sql_value_to_json(&int_val), serde_json::json!(42));

        let str_val = SqlValue {
            value: Some(Value::StringValue("hello".to_string())),
        };
        assert_eq!(sql_value_to_json(&str_val), serde_json::json!("hello"));
    }

    #[test]
    fn sql_object_to_json_recurses_nested_values() {
        let obj = SqlObject {
            fields: std::collections::HashMap::from([
                (
                    "name".to_string(),
                    SqlValue {
                        value: Some(Value::StringValue("alice".to_string())),
                    },
                ),
                (
                    "age".to_string(),
                    SqlValue {
                        value: Some(Value::Int64Value(42)),
                    },
                ),
            ]),
        };

        assert_eq!(
            sql_object_to_json(&obj),
            serde_json::json!({"name": "alice", "age": 42})
        );
    }

    #[test]
    fn path_filters_convert_to_document_filter_contract() {
        let filters = vec![PathFilter {
            path: "$.category".to_string(),
            operator: FilterOperator::Eq,
            value: FilterValue::String("electronics".to_string()),
        }];

        let filter = convert_path_filters_to_document_filter(&filters).expect("filter");
        assert_eq!(filter.conditions.len(), 1);
        assert_eq!(filter.conditions[0].path, "$.category");
    }

    #[test]
    fn filter_value_array_converts_to_sql_array() {
        let value = FilterValue::Array(vec![
            FilterValue::String("a".to_string()),
            FilterValue::Number(2.0),
        ]);
        let sql = convert_filter_value_to_sql(&value);
        match sql.value {
            Some(SqlValueVariant::ArrayValue(arr)) => assert_eq!(arr.values.len(), 2),
            other => panic!("expected array value, got {:?}", other),
        }
    }

    #[test]
    fn build_document_query_record_preserves_document_shape() {
        let document = SqlObject {
            fields: std::collections::HashMap::from([(
                "title".to_string(),
                SqlValue {
                    value: Some(Value::StringValue("Doc".to_string())),
                },
            )]),
        };

        let record = build_document_query_record("doc_1", &document);
        assert_eq!(record.id, "doc_1");
        assert_eq!(record.source_model, DataModel::Document);
        assert_eq!(record.data["title"], "Doc");
        assert!(record.score.is_none());
    }

    fn make_cond(path: &str, op: DocFilterOperator, val: &str) -> DocFilterCondition {
        DocFilterCondition {
            path: path.to_string(),
            operator: op as i32,
            value: Some(SqlValue {
                value: Some(Value::StringValue(val.to_string())),
            }),
            values: vec![],
        }
    }

    #[test]
    fn document_filter_single_eq_lowers_to_comparison() {
        let filter = DocumentFilter {
            conditions: vec![make_cond("$.status", DocFilterOperator::Eq, "active")],
            or_filters: vec![],
            and_filters: vec![],
        };
        let expr = document_filter_to_filter_expression(&filter).expect("non-empty filter");
        match expr {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "$.status");
                assert_eq!(operator, ComparisonOperator::Equals);
                assert_eq!(value, serde_json::json!("active"));
            }
            other => panic!("expected Comparison, got {:?}", other),
        }
    }

    #[test]
    fn document_filter_multiple_conditions_become_and() {
        let filter = DocumentFilter {
            conditions: vec![
                make_cond("$.a", DocFilterOperator::Eq, "1"),
                make_cond("$.b", DocFilterOperator::Gt, "2"),
            ],
            or_filters: vec![],
            and_filters: vec![],
        };
        let expr = document_filter_to_filter_expression(&filter).expect("non-empty filter");
        match expr {
            FilterExpression::And(parts) => assert_eq!(parts.len(), 2),
            other => panic!("expected And, got {:?}", other),
        }
    }

    #[test]
    fn document_filter_or_sub_filters_become_or_expression() {
        let filter = DocumentFilter {
            conditions: vec![],
            or_filters: vec![
                DocumentFilter {
                    conditions: vec![make_cond("$.x", DocFilterOperator::Eq, "p")],
                    or_filters: vec![],
                    and_filters: vec![],
                },
                DocumentFilter {
                    conditions: vec![make_cond("$.x", DocFilterOperator::Eq, "q")],
                    or_filters: vec![],
                    and_filters: vec![],
                },
            ],
            and_filters: vec![],
        };
        let expr = document_filter_to_filter_expression(&filter).expect("non-empty filter");
        match expr {
            FilterExpression::Or(parts) => assert_eq!(parts.len(), 2),
            other => panic!("expected Or, got {:?}", other),
        }
    }

    #[test]
    fn document_filter_exists_lowers_to_is_not_null() {
        let filter = DocumentFilter {
            conditions: vec![DocFilterCondition {
                path: "$.email".to_string(),
                operator: DocFilterOperator::Exists as i32,
                value: None,
                values: vec![],
            }],
            or_filters: vec![],
            and_filters: vec![],
        };
        let expr = document_filter_to_filter_expression(&filter).expect("non-empty filter");
        match expr {
            FilterExpression::Comparison { operator, .. } => {
                assert_eq!(operator, ComparisonOperator::IsNotNull);
            }
            other => panic!("expected Comparison, got {:?}", other),
        }
    }

    #[test]
    fn empty_document_filter_produces_none() {
        let filter = DocumentFilter {
            conditions: vec![],
            or_filters: vec![],
            and_filters: vec![],
        };
        assert!(document_filter_to_filter_expression(&filter).is_none());
    }
}
