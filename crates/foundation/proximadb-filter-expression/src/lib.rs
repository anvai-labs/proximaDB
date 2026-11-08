pub mod index_algorithm;
/// Advanced metadata query engine with runtime evaluation and regex caching.
pub mod metadata_query;
pub mod normalize;
pub mod query_params;
pub mod search_query;

pub use index_algorithm::IndexAlgorithmConfig;
pub use normalize::normalize_field;
pub use query_params::Params;
pub use search_query::SearchQuery;

/// Complex filter expression for advanced metadata filtering.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FilterExpression {
    /// Single comparison operation.
    Comparison {
        /// Metadata field name to compare.
        field: String,
        /// Comparison operator to apply.
        operator: ComparisonOperator,
        /// Value to compare against.
        value: serde_json::Value,
    },
    /// Logical AND of multiple expressions.
    And(Vec<FilterExpression>),
    /// Logical OR of multiple expressions.
    Or(Vec<FilterExpression>),
    /// Logical NOT of an expression.
    Not(Box<FilterExpression>),
}

/// Comparison operators for metadata filtering.
///
/// Canonical vocabulary: the union of this enum and the former
/// `metadata_query::ComparisonOperator`, which had drifted apart (same
/// intent, diverged variant names and variant sets). The legacy
/// `metadata_query` wire names still deserialize via `serde(alias)`, so
/// persisted/parsed payloads from either vocabulary remain readable
/// (mixed-read-safe); serialization always emits the canonical names.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ComparisonOperator {
    /// Equal to (legacy `metadata_query` wire name: `Equal`).
    #[serde(alias = "Equal")]
    Equals,
    /// Not equal to (legacy `metadata_query` wire name: `NotEqual`).
    #[serde(alias = "NotEqual")]
    NotEquals,
    /// Greater than.
    GreaterThan,
    /// Greater than or equal to.
    GreaterThanOrEqual,
    /// Less than.
    LessThan,
    /// Less than or equal to.
    LessThanOrEqual,
    /// Value is in the provided list.
    In,
    /// Value is not in the provided list.
    NotIn,
    /// String contains substring.
    Contains,
    /// String starts with prefix.
    StartsWith,
    /// String ends with suffix.
    EndsWith,
    /// Value is between two bounds (inclusive).
    Between,
    /// Value is null.
    IsNull,
    /// Value is not null.
    IsNotNull,
    /// SQL-style LIKE pattern matching (supports `%` and `_` wildcards).
    Like,
    /// Field exists and is not null.
    Exists,
    /// Field is absent or null.
    NotExists,
    /// Regular expression match.
    Regex,
}

/// Convert a [`FilterExpression`] into AXIS index metadata filters
/// (conjunction semantics).
///
/// This is the single canonical converter — previously stamped as private
/// `convert_filter_to_index` copies in six engines, which drifted: only the
/// SST copy mapped `Contains`/`StartsWith`/`EndsWith`/`Like`/`Between`, so
/// the other engines silently lost index pushdown for those operators and
/// fell back to post-filtering. The SST superset is canonical; this crate
/// is the natural home because it already depends on
/// `proximadb-index-traits` (the reverse would be a dependency cycle).
///
/// Semantics (unchanged from the engine copies): `And` recurses into its
/// operands; `Or`/`Not` have no index representation and contribute
/// nothing, so callers must keep post-filtering the full expression;
/// operators without a direct index equivalent (`IsNull`, `IsNotNull`)
/// are likewise skipped.
pub fn to_index_metadata_filters(
    filter_expression: Option<&FilterExpression>,
) -> Vec<proximadb_index_traits::IndexMetadataFilter> {
    use proximadb_index_traits::{IndexFilterOperator, IndexMetadataFilter};

    let Some(filter) = filter_expression else {
        return Vec::new();
    };

    let mut index_filters = Vec::new();

    match filter {
        FilterExpression::Comparison {
            field,
            operator,
            value,
        } => {
            let index_operator = match operator {
                ComparisonOperator::Equals => IndexFilterOperator::Equals,
                ComparisonOperator::NotEquals => IndexFilterOperator::NotEquals,
                ComparisonOperator::GreaterThan => IndexFilterOperator::GreaterThan,
                ComparisonOperator::GreaterThanOrEqual => IndexFilterOperator::GreaterThanOrEqual,
                ComparisonOperator::LessThan => IndexFilterOperator::LessThan,
                ComparisonOperator::LessThanOrEqual => IndexFilterOperator::LessThanOrEqual,
                ComparisonOperator::In => IndexFilterOperator::In,
                ComparisonOperator::NotIn => IndexFilterOperator::NotIn,
                ComparisonOperator::Contains => IndexFilterOperator::Contains,
                ComparisonOperator::StartsWith => IndexFilterOperator::StartsWith,
                ComparisonOperator::EndsWith => IndexFilterOperator::EndsWith,
                ComparisonOperator::Like => IndexFilterOperator::Like,
                ComparisonOperator::Between => IndexFilterOperator::Between,
                _ => {
                    tracing::debug!(
                        "Operator {:?} not directly supported by index, will use post-filtering",
                        operator
                    );
                    return index_filters;
                }
            };

            index_filters.push(IndexMetadataFilter {
                field: field.clone(),
                operator: index_operator,
                value: value.clone(),
            });
        }
        FilterExpression::And(filters) => {
            for f in filters {
                index_filters.extend(to_index_metadata_filters(Some(f)));
            }
        }
        FilterExpression::Or(_) | FilterExpression::Not(_) => {
            // OR and NOT are not directly supported by the index; the
            // caller keeps post-filtering the full expression.
            tracing::debug!("OR/NOT filters not supported by index, will use post-filtering");
        }
    }

    index_filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_expression_is_structural() {
        let expr = FilterExpression::Comparison {
            field: "score".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: serde_json::json!(0.8),
        };

        match expr {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "score");
                assert_eq!(operator, ComparisonOperator::GreaterThan);
                assert_eq!(value, serde_json::json!(0.8));
            }
            _ => panic!("expected comparison expression"),
        }
    }

    #[test]
    fn nested_boolean_filter_expressions_round_trip_shape() {
        let expr = FilterExpression::And(vec![
            FilterExpression::Comparison {
                field: "tenant".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::json!("acme"),
            },
            FilterExpression::Not(Box::new(FilterExpression::Comparison {
                field: "deleted".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::json!(true),
            })),
        ]);

        match expr {
            FilterExpression::And(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(parts[1], FilterExpression::Not(_)));
            }
            _ => panic!("expected and expression"),
        }
    }

    fn comparison(
        field: &str,
        op: ComparisonOperator,
        value: serde_json::Value,
    ) -> FilterExpression {
        FilterExpression::Comparison {
            field: field.to_string(),
            operator: op,
            value,
        }
    }

    #[test]
    fn to_index_metadata_filters_maps_all_thirteen_operators() {
        use proximadb_index_traits::IndexFilterOperator as Op;

        let cases: Vec<(ComparisonOperator, Op)> = vec![
            (ComparisonOperator::Equals, Op::Equals),
            (ComparisonOperator::NotEquals, Op::NotEquals),
            (ComparisonOperator::GreaterThan, Op::GreaterThan),
            (
                ComparisonOperator::GreaterThanOrEqual,
                Op::GreaterThanOrEqual,
            ),
            (ComparisonOperator::LessThan, Op::LessThan),
            (ComparisonOperator::LessThanOrEqual, Op::LessThanOrEqual),
            (ComparisonOperator::In, Op::In),
            (ComparisonOperator::NotIn, Op::NotIn),
            // The five that only the SST copy used to map (redundancy
            // pass 3): losing these meant silent post-filter fallback.
            (ComparisonOperator::Contains, Op::Contains),
            (ComparisonOperator::StartsWith, Op::StartsWith),
            (ComparisonOperator::EndsWith, Op::EndsWith),
            (ComparisonOperator::Like, Op::Like),
            (ComparisonOperator::Between, Op::Between),
        ];

        for (i, (op, want)) in cases.into_iter().enumerate() {
            let filters =
                to_index_metadata_filters(Some(&comparison("f", op.clone(), serde_json::json!(i))));
            assert_eq!(filters.len(), 1, "{op:?} must push down");
            assert_eq!(filters[0].operator, want);
            assert_eq!(filters[0].field, "f");
        }
    }

    #[test]
    fn to_index_metadata_filters_and_recurses_or_and_not_contribute_nothing() {
        let expr = FilterExpression::And(vec![
            comparison("a", ComparisonOperator::Equals, serde_json::json!(1)),
            FilterExpression::Or(vec![
                comparison("b", ComparisonOperator::Equals, serde_json::json!(2)),
                comparison("c", ComparisonOperator::Equals, serde_json::json!(3)),
            ]),
            FilterExpression::Not(Box::new(comparison(
                "d",
                ComparisonOperator::Equals,
                serde_json::json!(4),
            ))),
        ]);

        // The Or/Not branches contribute nothing by design; callers keep
        // post-filtering the full expression.
        let filters = to_index_metadata_filters(Some(&expr));
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].field, "a");
    }

    #[test]
    fn to_index_metadata_filters_none_and_unsupported_operators() {
        assert!(to_index_metadata_filters(None).is_empty());
        assert!(
            to_index_metadata_filters(Some(&comparison(
                "x",
                ComparisonOperator::IsNull,
                serde_json::json!(null),
            )))
            .is_empty()
        );
    }
}
