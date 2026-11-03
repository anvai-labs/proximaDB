//! AXIS hybrid query construction utilities.
//!
//! Provides conversion from ProximaDB filter expressions to AXIS hybrid queries,
//! supporting both vector similarity search and metadata filtering.

use anyhow::{Result, anyhow};

/// Build an AXIS hybrid query from ProximaDB search parameters.
///
/// This function converts ProximaDB's filter expression format into AXIS's
/// hybrid query format, separating ID filters from metadata filters for
/// optimized query execution.
///
/// # Arguments
///
/// * `collection_id` - The target collection identifier
/// * `search_params` - Search parameters including filter expressions and vector data
///
/// # Returns
///
/// A `HybridQuery` suitable for submission to the AXIS index manager
///
/// # Errors
///
/// Returns an error if:
/// - ID filters use non-string values
/// - The filter expression contains unsupported operators (OR, NOT)
pub fn build_axis_hybrid_query(
    collection_id: &str,
    search_params: &crate::core::search::SearchParams,
) -> Result<crate::index::axis::management::manager::HybridQuery> {
    build_axis_hybrid_query_with_mode(
        collection_id,
        search_params,
        crate::index::axis::management::manager::AnnFilteringMode::default(),
    )
}

/// Like `build_axis_hybrid_query` but with an explicit ADR-011 filtering mode.
pub fn build_axis_hybrid_query_with_mode(
    collection_id: &str,
    search_params: &crate::core::search::SearchParams,
    ann_filtering_mode: crate::index::axis::management::manager::AnnFilteringMode,
) -> Result<crate::index::axis::management::manager::HybridQuery> {
    build_axis_hybrid_query_with_policy(
        collection_id,
        search_params,
        ann_filtering_mode,
        None,
        None,
    )
}

/// Like `build_axis_hybrid_query_with_mode`, but carries the ADR-011 policy
/// inputs through to AXIS so the manager can select PreFilter / Inline /
/// PostFilter from the same policy contract used by the planner.
pub fn build_axis_hybrid_query_with_policy(
    collection_id: &str,
    search_params: &crate::core::search::SearchParams,
    ann_filtering_mode: crate::index::axis::management::manager::AnnFilteringMode,
    ann_filtering_policy: Option<proximadb_catalog::AnnFilteringPolicy>,
    estimated_selectivity: Option<f64>,
) -> Result<crate::index::axis::management::manager::HybridQuery> {
    use crate::core::search::{ComparisonOperator, FilterExpression};
    use crate::index::axis::management::manager::{
        FilterOperator, HybridQuery, MetadataFilter, VectorQuery,
    };

    fn flatten_filter_expression(
        expr: &FilterExpression,
        metadata_filters: &mut Vec<MetadataFilter>,
        id_filters: &mut Vec<String>,
    ) -> Result<()> {
        match expr {
            FilterExpression::And(parts) => {
                for part in parts {
                    flatten_filter_expression(part, metadata_filters, id_filters)?;
                }
                Ok(())
            }
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                let axis_operator = match operator {
                    ComparisonOperator::Equals => FilterOperator::Equals,
                    ComparisonOperator::NotEquals => FilterOperator::NotEquals,
                    ComparisonOperator::GreaterThan => FilterOperator::GreaterThan,
                    ComparisonOperator::GreaterThanOrEqual => FilterOperator::GreaterThanOrEqual,
                    ComparisonOperator::LessThan => FilterOperator::LessThan,
                    ComparisonOperator::LessThanOrEqual => FilterOperator::LessThanOrEqual,
                    ComparisonOperator::In => FilterOperator::In,
                    ComparisonOperator::NotIn => FilterOperator::NotIn,
                    ComparisonOperator::Contains => FilterOperator::Contains,
                    ComparisonOperator::StartsWith => FilterOperator::StartsWith,
                    ComparisonOperator::EndsWith => FilterOperator::EndsWith,
                    ComparisonOperator::Between => FilterOperator::Between,
                    ComparisonOperator::IsNull => FilterOperator::IsNull,
                    ComparisonOperator::IsNotNull => FilterOperator::IsNotNull,
                    ComparisonOperator::Like => FilterOperator::Like,
                    // Presence tests lower onto their identical null-test
                    // predicates (absent and present-and-null are conflated).
                    ComparisonOperator::NotExists => FilterOperator::IsNull,
                    ComparisonOperator::Exists => FilterOperator::IsNotNull,
                    // No AXIS lowering for Regex. Unlike
                    // to_index_metadata_filters (whose consumers re-apply the
                    // full expression as a residual), the AXIS leg has NO
                    // residual re-filter — skipping the condition here would
                    // silently drop it. Decline the whole AXIS lowering
                    // instead (same contract as Or/Not below): callers either
                    // fall back to the storage leg, which evaluates the full
                    // filter, or fail closed.
                    ComparisonOperator::Regex => {
                        return Err(anyhow!(
                            "Regex filters are not lowerable to AXIS (no residual re-filter on this path)"
                        ));
                    }
                };

                if field == "id"
                    && matches!(axis_operator, FilterOperator::Equals | FilterOperator::In)
                {
                    match axis_operator {
                        FilterOperator::Equals => {
                            let vector_id = value.as_str().ok_or_else(|| {
                                anyhow!("id equality filters must use string values")
                            })?;
                            id_filters.push(vector_id.to_string());
                        }
                        FilterOperator::In => {
                            let values = value.as_array().ok_or_else(|| {
                                anyhow!("id IN filters must use an array of strings")
                            })?;
                            for item in values {
                                let vector_id = item.as_str().ok_or_else(|| {
                                    anyhow!("id IN filters must use an array of strings")
                                })?;
                                id_filters.push(vector_id.to_string());
                            }
                        }
                        _ => {}
                    }
                } else {
                    metadata_filters.push(MetadataFilter {
                        field: field.clone(),
                        operator: axis_operator,
                        value: value.clone(),
                    });
                }

                Ok(())
            }
            FilterExpression::Or(_) | FilterExpression::Not(_) => Err(anyhow!(
                "AXIS hybrid query builder currently supports conjunctive filters only"
            )),
        }
    }

    fn normalize_selectivity(value: f64) -> Option<f64> {
        value.is_finite().then(|| value.clamp(0.0, 1.0))
    }

    let vector_query = if let Some(vectors) = &search_params.query_vectors {
        vectors.first().map(|vector| VectorQuery::Dense {
            vector: vector.clone(),
            similarity_threshold: 0.0,
        })
    } else {
        search_params
            .vector
            .clone()
            .map(|vector| VectorQuery::Dense {
                vector,
                similarity_threshold: 0.0,
            })
    };

    let mut metadata_filters = Vec::new();
    let mut id_filters = Vec::new();

    if let Some(filter_expression) = &search_params.filter_expression {
        flatten_filter_expression(filter_expression, &mut metadata_filters, &mut id_filters)?;
    } else if let Some(filters) = &search_params.filters {
        for (field, value) in filters {
            metadata_filters.push(MetadataFilter {
                field: field.clone(),
                operator: FilterOperator::Equals,
                value: value.clone(),
            });
        }
    }

    let hinted_selectivity = estimated_selectivity
        .and_then(normalize_selectivity)
        .or_else(|| {
            search_params
                .runtime_hints
                .as_ref()
                .and_then(|hints| hints.expected_selectivity)
                .and_then(normalize_selectivity)
        });
    let ann_filtering_policy =
        ann_filtering_policy.or_else(|| hinted_selectivity.map(|_| Default::default()));

    Ok(HybridQuery {
        collection_id: collection_id.to_string(),
        vector_query,
        metadata_filters,
        id_filters,
        top_k: search_params.top_k.unwrap_or(10) as usize,
        include_expired: search_params.include_expired.unwrap_or(false),
        ann_filtering_mode,
        ann_filtering_policy,
        estimated_selectivity: hinted_selectivity,
        // Map the accuracy-vs-latency knob onto the AXIS query so the warm
        // HNSW/IVF path honors it on this builder path too (mirrors the
        // SST search-engine direct construction).
        search_effort: search_params.search_mode.to_search_effort(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::search::{ComparisonOperator, FilterExpression, SearchParams};

    #[test]
    fn test_build_axis_hybrid_query_basic() {
        let search_params = SearchParams {
            vector: Some(vec![1.0, 2.0, 3.0]),
            top_k: Some(5),
            ..Default::default()
        };

        let result = build_axis_hybrid_query("test_collection", &search_params).unwrap();

        assert_eq!(result.collection_id, "test_collection");
        assert_eq!(result.top_k, 5);
        assert!(result.vector_query.is_some());
    }

    #[test]
    fn test_build_axis_hybrid_query_with_filter() {
        let filter = FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!("test"),
        };

        let search_params = SearchParams {
            vector: Some(vec![1.0, 2.0, 3.0]),
            filter_expression: Some(filter),
            ..Default::default()
        };

        let result = build_axis_hybrid_query("test_collection", &search_params).unwrap();

        assert_eq!(result.metadata_filters.len(), 1);
        assert_eq!(result.metadata_filters[0].field, "category");
    }

    #[test]
    fn test_build_axis_hybrid_query_id_filter() {
        let filter = FilterExpression::Comparison {
            field: "id".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!("vec123"),
        };

        let search_params = SearchParams {
            vector: Some(vec![1.0, 2.0, 3.0]),
            filter_expression: Some(filter),
            ..Default::default()
        };

        let result = build_axis_hybrid_query("test_collection", &search_params).unwrap();

        assert_eq!(result.id_filters.len(), 1);
        assert_eq!(result.id_filters[0], "vec123");
        assert!(result.metadata_filters.is_empty());
    }

    #[test]
    fn test_build_axis_hybrid_query_carries_runtime_selectivity_hint() {
        let search_params = SearchParams {
            vector: Some(vec![1.0, 2.0, 3.0]),
            runtime_hints: Some(crate::query::query_optimizer::FilterOptimizationHints {
                expected_selectivity: Some(0.8),
                preferred_index: None,
                allow_parallel: true,
            }),
            ..Default::default()
        };

        let result = build_axis_hybrid_query_with_mode(
            "test_collection",
            &search_params,
            crate::index::axis::management::manager::AnnFilteringMode::Inline,
        )
        .unwrap();

        assert!(result.ann_filtering_policy.is_some());
        assert_eq!(result.estimated_selectivity, Some(0.8));
    }

    #[test]
    fn test_build_axis_hybrid_query_clamps_explicit_selectivity() {
        let search_params = SearchParams {
            vector: Some(vec![1.0, 2.0, 3.0]),
            ..Default::default()
        };

        let result = build_axis_hybrid_query_with_policy(
            "test_collection",
            &search_params,
            crate::index::axis::management::manager::AnnFilteringMode::Inline,
            Some(proximadb_catalog::AnnFilteringPolicy::default()),
            Some(1.7),
        )
        .unwrap();

        assert!(result.ann_filtering_policy.is_some());
        assert_eq!(result.estimated_selectivity, Some(1.0));
    }
}
