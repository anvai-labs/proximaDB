//! Parser for converting gRPC metadata filters to logical metadata queries
//!
//! Supports both simple string-based filters and advanced JSON-based logical operators.

use anyhow::{Context, Result};
use serde_json::{Value as JsonValue, json};
use std::collections::HashMap;

use crate::core::{ComparisonOperator, FieldQuery, MetadataQuery};

/// Parse metadata filters from gRPC request into logical metadata query
pub fn parse_metadata_query(
    metadata_filter: &HashMap<String, String>,
) -> Result<Option<MetadataQuery>> {
    if metadata_filter.is_empty() {
        return Ok(None);
    }

    // Check if there's a special "__logical_query" or "$logical" field containing JSON
    if let Some(logical_query_json) = metadata_filter
        .get("__logical_query")
        .or_else(|| metadata_filter.get("$logical"))
    {
        return parse_json_logical_query(logical_query_json);
    }

    // Check for logical operator fields
    let and_queries = parse_operator_queries(metadata_filter, "__and")?;
    let or_queries = parse_operator_queries(metadata_filter, "__or")?;
    let not_query = parse_not_query(metadata_filter)?;

    // If we have logical operators, construct a logical query
    if !and_queries.is_empty() || !or_queries.is_empty() || not_query.is_some() {
        return construct_logical_query(and_queries, or_queries, not_query, metadata_filter);
    }

    // Fall back to simple equality filters
    parse_simple_filters(metadata_filter)
}

/// Parse a complete JSON logical query
fn parse_json_logical_query(json_str: &str) -> Result<Option<MetadataQuery>> {
    let query_value: JsonValue = serde_json::from_str(json_str)
        .with_context(|| format!("Invalid JSON in logical query: {}", json_str))?;

    parse_json_query_value(&query_value)
}

/// Parse a JSON value into a metadata query
fn parse_json_query_value(value: &JsonValue) -> Result<Option<MetadataQuery>> {
    match value {
        JsonValue::Object(obj) => {
            if let Some(and_array) = obj.get("$and").or_else(|| obj.get("and")) {
                parse_json_and(and_array)
            } else if let Some(or_array) = obj.get("$or").or_else(|| obj.get("or")) {
                parse_json_or(or_array)
            } else if let Some(not_obj) = obj.get("$not").or_else(|| obj.get("not")) {
                parse_json_not(not_obj)
            } else if let Some(field_name) = obj.keys().next() {
                // Field-level query
                if let Some(field_value) = obj.get(field_name) {
                    parse_json_field_query(field_name, field_value)
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Parse JSON AND operation
fn parse_json_and(and_array: &JsonValue) -> Result<Option<MetadataQuery>> {
    if let JsonValue::Array(queries) = and_array {
        let mut parsed_queries = Vec::new();
        for query_value in queries {
            if let Some(query) = parse_json_query_value(query_value)? {
                parsed_queries.push(query);
            }
        }

        if parsed_queries.is_empty() {
            Ok(None)
        } else {
            Ok(Some(MetadataQuery::And(parsed_queries)))
        }
    } else {
        Err(anyhow::anyhow!("AND operation must be an array"))
    }
}

/// Parse JSON OR operation
fn parse_json_or(or_array: &JsonValue) -> Result<Option<MetadataQuery>> {
    if let JsonValue::Array(queries) = or_array {
        let mut parsed_queries = Vec::new();
        for query_value in queries {
            if let Some(query) = parse_json_query_value(query_value)? {
                parsed_queries.push(query);
            }
        }

        if parsed_queries.is_empty() {
            Ok(None)
        } else {
            Ok(Some(MetadataQuery::Or(parsed_queries)))
        }
    } else {
        Err(anyhow::anyhow!("OR operation must be an array"))
    }
}

/// Parse JSON NOT operation
fn parse_json_not(not_obj: &JsonValue) -> Result<Option<MetadataQuery>> {
    if let Some(inner_query) = parse_json_query_value(not_obj)? {
        Ok(Some(MetadataQuery::Not(Box::new(inner_query))))
    } else {
        Ok(None)
    }
}

/// Parse JSON field query with operators
fn parse_json_field_query(
    field_name: &str,
    field_value: &JsonValue,
) -> Result<Option<MetadataQuery>> {
    match field_value {
        JsonValue::Object(operators) => {
            // Field with operators like {"price": {"$gt": 100, "$lt": 500}}
            let mut queries = Vec::new();

            for (op, value) in operators {
                let operator = match op.as_str() {
                    "$eq" | "eq" => ComparisonOperator::Equals,
                    "$ne" | "ne" => ComparisonOperator::NotEquals,
                    "$gt" | "gt" => ComparisonOperator::GreaterThan,
                    "$gte" | "gte" => ComparisonOperator::GreaterThanOrEqual,
                    "$lt" | "lt" => ComparisonOperator::LessThan,
                    "$lte" | "lte" => ComparisonOperator::LessThanOrEqual,
                    "$contains" | "contains" => ComparisonOperator::Contains,
                    "$startsWith" | "startsWith" => ComparisonOperator::StartsWith,
                    "$endsWith" | "endsWith" => ComparisonOperator::EndsWith,
                    "$in" | "in" => ComparisonOperator::In,
                    "$nin" | "nin" => ComparisonOperator::NotIn,
                    "$exists" | "exists" => ComparisonOperator::Exists,
                    "$nexists" | "nexists" => ComparisonOperator::NotExists,
                    "$regex" | "regex" => ComparisonOperator::Regex,
                    _ => {
                        tracing::warn!("Unknown operator: {}", op);
                        continue;
                    }
                };

                queries.push(MetadataQuery::Field(FieldQuery {
                    field: field_name.to_string(),
                    operator,
                    value: value.clone(),
                }));
            }

            if queries.is_empty() {
                Ok(None)
            } else if queries.len() == 1 {
                let only_query = queries.into_iter().next();
                Ok(only_query)
            } else {
                Ok(Some(MetadataQuery::And(queries)))
            }
        }
        _ => {
            // Simple equality: {"category": "electronics"}
            Ok(Some(MetadataQuery::field_eq(
                field_name,
                field_value.clone(),
            )))
        }
    }
}

/// Parse logical operator queries from special fields
fn parse_operator_queries(
    metadata_filter: &HashMap<String, String>,
    operator: &str,
) -> Result<Vec<MetadataQuery>> {
    let mut queries = Vec::new();

    // Look for numbered operator fields: __and_1, __and_2, etc.
    for (key, value) in metadata_filter {
        if key.starts_with(operator) && key.len() > operator.len() {
            // Parse the value as JSON
            match serde_json::from_str::<JsonValue>(value) {
                Ok(json_value) => {
                    if let Some(query) = parse_json_query_value(&json_value)? {
                        queries.push(query);
                    }
                }
                Err(_) => {
                    // If not JSON, treat as simple field=value
                    if let Some(field_name) = key.strip_prefix(&format!("{}_", operator)) {
                        queries.push(MetadataQuery::field_eq(field_name, json!(value)));
                    }
                }
            }
        }
    }

    Ok(queries)
}

/// Parse NOT query from special field
fn parse_not_query(metadata_filter: &HashMap<String, String>) -> Result<Option<MetadataQuery>> {
    if let Some(not_value) = metadata_filter
        .get("__not")
        .or_else(|| metadata_filter.get("$not"))
    {
        match serde_json::from_str::<JsonValue>(not_value) {
            Ok(json_value) => {
                if let Some(inner_query) = parse_json_query_value(&json_value)? {
                    Ok(Some(MetadataQuery::Not(Box::new(inner_query))))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    } else {
        Ok(None)
    }
}

/// Construct logical query from parsed components
fn construct_logical_query(
    and_queries: Vec<MetadataQuery>,
    or_queries: Vec<MetadataQuery>,
    not_query: Option<MetadataQuery>,
    metadata_filter: &HashMap<String, String>,
) -> Result<Option<MetadataQuery>> {
    let mut all_queries = Vec::new();

    // Add AND queries
    if !and_queries.is_empty() {
        all_queries.push(MetadataQuery::And(and_queries));
    }

    // Add OR queries
    if !or_queries.is_empty() {
        all_queries.push(MetadataQuery::Or(or_queries));
    }

    // Add NOT query
    if let Some(not_q) = not_query {
        all_queries.push(not_q);
    }

    // Add simple equality filters (excluding special operator fields)
    let simple_queries = parse_simple_filters_excluding_operators(metadata_filter)?;
    if let Some(simple_query) = simple_queries {
        all_queries.push(simple_query);
    }

    // Combine all queries with AND
    if all_queries.is_empty() {
        Ok(None)
    } else if all_queries.len() == 1 {
        let only_query = all_queries.into_iter().next();
        Ok(only_query)
    } else {
        Ok(Some(MetadataQuery::And(all_queries)))
    }
}

/// Parse simple equality filters
fn parse_simple_filters(
    metadata_filter: &HashMap<String, String>,
) -> Result<Option<MetadataQuery>> {
    let mut queries = Vec::new();

    for (key, value) in metadata_filter {
        // Try to parse value as JSON, fall back to string
        let json_value = match serde_json::from_str::<JsonValue>(value) {
            Ok(json_val) => json_val,
            Err(_) => JsonValue::String(value.clone()),
        };

        queries.push(MetadataQuery::field_eq(key, json_value));
    }

    if queries.is_empty() {
        Ok(None)
    } else if queries.len() == 1 {
        let only_query = queries.into_iter().next();
        Ok(only_query)
    } else {
        Ok(Some(MetadataQuery::And(queries)))
    }
}

/// Parse simple filters excluding operator fields
fn parse_simple_filters_excluding_operators(
    metadata_filter: &HashMap<String, String>,
) -> Result<Option<MetadataQuery>> {
    let mut queries = Vec::new();

    for (key, value) in metadata_filter {
        // Skip operator fields
        if key.starts_with("__") {
            continue;
        }

        // Try to parse value as JSON, fall back to string
        let json_value = match serde_json::from_str::<JsonValue>(value) {
            Ok(json_val) => json_val,
            Err(_) => JsonValue::String(value.clone()),
        };

        queries.push(MetadataQuery::field_eq(key, json_value));
    }

    if queries.is_empty() {
        Ok(None)
    } else if queries.len() == 1 {
        let only_query = queries.into_iter().next();
        Ok(only_query)
    } else {
        Ok(Some(MetadataQuery::And(queries)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::panic)] // Test panic for assertion failure
    fn test_simple_equality_parsing() {
        let mut filter = HashMap::new();
        filter.insert("category".to_string(), "electronics".to_string());
        filter.insert("brand".to_string(), "TechCorp".to_string());

        let query = parse_metadata_query(&filter).unwrap().unwrap();

        // Should create AND query with two equality conditions
        match query {
            MetadataQuery::And(queries) => {
                assert_eq!(queries.len(), 2);
            }
            _ => panic!("Expected AND query"),
        }
    }

    #[test]
    #[allow(clippy::panic)] // Test panic for assertion failure
    fn test_json_logical_query_parsing() {
        let mut filter = HashMap::new();
        filter.insert(
            "__logical_query".to_string(),
            r#"
        {
            "and": [
                {"category": "electronics"},
                {"price": {"$lt": 300}}
            ]
        }
        "#
            .to_string(),
        );

        let query = parse_metadata_query(&filter).unwrap().unwrap();

        match query {
            MetadataQuery::And(queries) => {
                assert_eq!(queries.len(), 2);
            }
            _ => panic!("Expected AND query"),
        }
    }

    #[test]
    #[allow(clippy::panic)] // Test panic for assertion failure
    fn test_operator_field_parsing() {
        let mut filter = HashMap::new();
        filter.insert(
            "__and_1".to_string(),
            r#"{"category": "electronics"}"#.to_string(),
        );
        filter.insert(
            "__and_2".to_string(),
            r#"{"price": {"$lt": 300}}"#.to_string(),
        );

        let query = parse_metadata_query(&filter).unwrap().unwrap();

        match query {
            MetadataQuery::And(queries) => {
                assert!(!queries.is_empty());
            }
            _ => panic!("Expected AND query"),
        }
    }

    #[test]
    #[allow(clippy::panic)] // Test panic for assertion failure
    fn test_complex_query_parsing() {
        let mut filter = HashMap::new();
        filter.insert(
            "__logical_query".to_string(),
            r#"
        {
            "or": [
                {
                    "and": [
                        {"category": "electronics"},
                        {"price": {"$lt": 500}}
                    ]
                },
                {
                    "and": [
                        {"category": "books"},
                        {"rating": {"$gte": 4.0}}
                    ]
                }
            ]
        }
        "#
            .to_string(),
        );

        let query = parse_metadata_query(&filter).unwrap().unwrap();

        match query {
            MetadataQuery::Or(queries) => {
                assert_eq!(queries.len(), 2);
            }
            _ => panic!("Expected OR query"),
        }
    }
}
