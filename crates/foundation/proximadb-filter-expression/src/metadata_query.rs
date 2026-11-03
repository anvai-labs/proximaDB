//! Advanced metadata query engine with logical operators
//!
//! Supports complex metadata filtering with AND, OR, NOT operations
//! and various comparison operators for flexible search queries.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Metadata query expression supporting logical and comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataQuery {
    /// Single field comparison
    Field(FieldQuery),
    /// Logical AND operation
    And(Vec<MetadataQuery>),
    /// Logical OR operation
    Or(Vec<MetadataQuery>),
    /// Logical NOT operation
    Not(Box<MetadataQuery>),
    /// Always true (matches all)
    All,
    /// Always false (matches none)
    None,
}

/// Field-specific query with comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldQuery {
    /// Metadata field name
    pub field: String,
    /// Comparison operation
    pub operator: ComparisonOperator,
    /// Expected value
    pub value: JsonValue,
}

pub use crate::ComparisonOperator;

/// Metadata query engine for evaluating complex queries
#[derive(Debug)]
pub struct MetadataQueryEngine {
    /// Pre-compiled regex patterns for performance
    regex_cache: HashMap<String, regex::Regex>,
}

impl MetadataQueryEngine {
    /// Create a new metadata query engine
    pub fn new() -> Self {
        Self {
            regex_cache: HashMap::new(),
        }
    }

    /// Evaluate a metadata query against a record's metadata
    pub fn evaluate(
        &mut self,
        query: &MetadataQuery,
        metadata: &HashMap<String, JsonValue>,
    ) -> Result<bool> {
        match query {
            MetadataQuery::Field(field_query) => self.evaluate_field_query(field_query, metadata),
            MetadataQuery::And(queries) => {
                for sub_query in queries {
                    if !self.evaluate(sub_query, metadata)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            MetadataQuery::Or(queries) => {
                for sub_query in queries {
                    if self.evaluate(sub_query, metadata)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            MetadataQuery::Not(query) => Ok(!self.evaluate(query, metadata)?),
            MetadataQuery::All => Ok(true),
            MetadataQuery::None => Ok(false),
        }
    }

    fn evaluate_field_query(
        &mut self,
        field_query: &FieldQuery,
        metadata: &HashMap<String, JsonValue>,
    ) -> Result<bool> {
        let field_value = metadata.get(&field_query.field);

        match field_query.operator {
            ComparisonOperator::Equals => Ok(field_value == Some(&field_query.value)),
            ComparisonOperator::NotEquals => Ok(field_value != Some(&field_query.value)),
            ComparisonOperator::LessThan => {
                self.compare_numeric_or_string(field_value, &field_query.value, |a, b| a < b)
            }
            ComparisonOperator::LessThanOrEqual => {
                self.compare_numeric_or_string(field_value, &field_query.value, |a, b| a <= b)
            }
            ComparisonOperator::GreaterThan => {
                self.compare_numeric_or_string(field_value, &field_query.value, |a, b| a > b)
            }
            ComparisonOperator::GreaterThanOrEqual => {
                self.compare_numeric_or_string(field_value, &field_query.value, |a, b| a >= b)
            }
            ComparisonOperator::Contains => {
                self.string_operation(field_value, &field_query.value, |text, pattern| {
                    text.contains(pattern)
                })
            }
            ComparisonOperator::StartsWith => {
                self.string_operation(field_value, &field_query.value, |text, pattern| {
                    text.starts_with(pattern)
                })
            }
            ComparisonOperator::EndsWith => {
                self.string_operation(field_value, &field_query.value, |text, pattern| {
                    text.ends_with(pattern)
                })
            }
            ComparisonOperator::In => self.array_operation(field_value, &field_query.value, true),
            ComparisonOperator::NotIn => {
                self.array_operation(field_value, &field_query.value, false)
            }
            ComparisonOperator::Exists => Ok(field_value.is_some_and(|v| !v.is_null())),
            ComparisonOperator::NotExists => Ok(field_value.is_none_or(|v| v.is_null())),
            ComparisonOperator::Regex => {
                self.regex_operation(&field_query.field, field_value, &field_query.value)
            }
            ComparisonOperator::Between => {
                // Canonical Between: value carries the inclusive [lo, hi]
                // bounds as a two-element JSON array.
                if let Some(actual) = field_value
                    && let JsonValue::Array(bounds) = &field_query.value
                    && bounds.len() == 2
                {
                    let ge =
                        self.compare_numeric_or_string(Some(actual), &bounds[0], |a, b| a >= b)?;
                    let le =
                        self.compare_numeric_or_string(Some(actual), &bounds[1], |a, b| a <= b)?;
                    Ok(ge && le)
                } else {
                    Ok(false)
                }
            }
            ComparisonOperator::IsNull => Ok(field_value.is_none_or(|v| v.is_null())),
            ComparisonOperator::IsNotNull => Ok(field_value.is_some_and(|v| !v.is_null())),
            ComparisonOperator::Like => {
                // SQL-style LIKE: `%` matches any run, `_` any single char,
                // everything else literal — translated to an anchored regex.
                self.string_operation(field_value, &field_query.value, |text, pattern| {
                    let mut src = String::with_capacity(pattern.len() * 2);
                    src.push('^');
                    for ch in pattern.chars() {
                        match ch {
                            '%' => src.push_str(".*"),
                            '_' => src.push('.'),
                            c => src.push_str(&regex::escape(&c.to_string())),
                        }
                    }
                    src.push('$');
                    regex::Regex::new(&src)
                        .map(|re| re.is_match(text))
                        .unwrap_or(false)
                })
            }
        }
    }

    fn compare_numeric_or_string<F>(
        &self,
        field_value: Option<&JsonValue>,
        expected: &JsonValue,
        compare_fn: F,
    ) -> Result<bool>
    where
        F: Fn(f64, f64) -> bool + Clone,
    {
        match field_value {
            Some(actual) => {
                if let (Some(a), Some(b)) = (actual.as_f64(), expected.as_f64()) {
                    Ok(compare_fn(a, b))
                } else if let (Some(a), Some(b)) = (actual.as_i64(), expected.as_i64()) {
                    Ok(compare_fn(a as f64, b as f64))
                } else if let (Some(a), Some(b)) = (actual.as_str(), expected.as_str()) {
                    let result = match std::cmp::PartialOrd::partial_cmp(a, b) {
                        Some(std::cmp::Ordering::Less) => compare_fn(0.0, 1.0),
                        Some(std::cmp::Ordering::Equal) => compare_fn(0.0, 0.0),
                        Some(std::cmp::Ordering::Greater) => compare_fn(1.0, 0.0),
                        None => false,
                    };
                    Ok(result)
                } else {
                    Ok(false)
                }
            }
            None => Ok(false),
        }
    }

    fn string_operation<F>(
        &self,
        field_value: Option<&JsonValue>,
        pattern: &JsonValue,
        string_fn: F,
    ) -> Result<bool>
    where
        F: Fn(&str, &str) -> bool,
    {
        match (field_value, pattern.as_str()) {
            (Some(JsonValue::String(text)), Some(pattern_str)) => {
                Ok(string_fn(text.as_str(), pattern_str))
            }
            _ => Ok(false),
        }
    }

    fn array_operation(
        &self,
        field_value: Option<&JsonValue>,
        array_value: &JsonValue,
        should_contain: bool,
    ) -> Result<bool> {
        match (field_value, array_value) {
            (Some(value), JsonValue::Array(array)) => {
                let contains = array.contains(value);
                Ok(if should_contain { contains } else { !contains })
            }
            _ => Ok(false),
        }
    }

    fn regex_operation(
        &mut self,
        field_name: &str,
        field_value: Option<&JsonValue>,
        pattern: &JsonValue,
    ) -> Result<bool> {
        match (field_value, pattern.as_str()) {
            (Some(JsonValue::String(text)), Some(pattern_str)) => {
                let regex = if let Some(cached_regex) = self.regex_cache.get(pattern_str) {
                    cached_regex.clone()
                } else {
                    let compiled_regex = regex::Regex::new(pattern_str).with_context(|| {
                        format!(
                            "Invalid regex pattern for field {}: {}",
                            field_name, pattern_str
                        )
                    })?;
                    self.regex_cache
                        .insert(pattern_str.to_string(), compiled_regex.clone());
                    compiled_regex
                };

                Ok(regex.is_match(text.as_str()))
            }
            _ => Ok(false),
        }
    }
}

impl Default for MetadataQueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing metadata queries
pub struct MetadataQueryBuilder {
    query: MetadataQuery,
}

impl MetadataQueryBuilder {
    pub fn new() -> Self {
        Self {
            query: MetadataQuery::All,
        }
    }

    pub fn field_equals(mut self, field: &str, value: JsonValue) -> Self {
        let field_query = MetadataQuery::Field(FieldQuery {
            field: field.to_string(),
            operator: ComparisonOperator::Equals,
            value,
        });
        self.query = self.combine_with_and(field_query);
        self
    }

    pub fn field_compare(
        mut self,
        field: &str,
        operator: ComparisonOperator,
        value: JsonValue,
    ) -> Self {
        let field_query = MetadataQuery::Field(FieldQuery {
            field: field.to_string(),
            operator,
            value,
        });
        self.query = self.combine_with_and(field_query);
        self
    }

    pub fn and(queries: Vec<MetadataQuery>) -> MetadataQuery {
        match queries.len() {
            0 => MetadataQuery::All,
            1 => queries.into_iter().next().unwrap_or(MetadataQuery::All),
            _ => MetadataQuery::And(queries),
        }
    }

    pub fn or(queries: Vec<MetadataQuery>) -> MetadataQuery {
        match queries.len() {
            0 => MetadataQuery::None,
            1 => queries.into_iter().next().unwrap_or(MetadataQuery::None),
            _ => MetadataQuery::Or(queries),
        }
    }

    pub fn not_condition(query: MetadataQuery) -> MetadataQuery {
        MetadataQuery::Not(Box::new(query))
    }

    pub fn build(self) -> MetadataQuery {
        self.query
    }

    fn combine_with_and(&self, new_query: MetadataQuery) -> MetadataQuery {
        match &self.query {
            MetadataQuery::All => new_query,
            existing => MetadataQuery::And(vec![existing.clone(), new_query]),
        }
    }
}

impl Default for MetadataQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataQuery {
    pub fn field_eq(field: &str, value: JsonValue) -> Self {
        Self::Field(FieldQuery {
            field: field.to_string(),
            operator: ComparisonOperator::Equals,
            value,
        })
    }

    pub fn field_exists(field: &str) -> Self {
        Self::Field(FieldQuery {
            field: field.to_string(),
            operator: ComparisonOperator::Exists,
            value: JsonValue::Null,
        })
    }

    pub fn field_contains(field: &str, substring: &str) -> Self {
        Self::Field(FieldQuery {
            field: field.to_string(),
            operator: ComparisonOperator::Contains,
            value: JsonValue::String(substring.to_string()),
        })
    }

    pub fn field_range(field: &str, min: f64, max: f64) -> Self {
        Self::And(vec![
            Self::Field(FieldQuery {
                field: field.to_string(),
                operator: ComparisonOperator::GreaterThanOrEqual,
                value: JsonValue::from(min),
            }),
            Self::Field(FieldQuery {
                field: field.to_string(),
                operator: ComparisonOperator::LessThanOrEqual,
                value: JsonValue::from(max),
            }),
        ])
    }

    pub fn field_in(field: &str, values: Vec<JsonValue>) -> Self {
        Self::Field(FieldQuery {
            field: field.to_string(),
            operator: ComparisonOperator::In,
            value: JsonValue::Array(values),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_metadata() -> HashMap<String, JsonValue> {
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), json!("electronics"));
        metadata.insert("price".to_string(), json!(99.99));
        metadata.insert(
            "tags".to_string(),
            json!(["computer", "gaming", "portable"]),
        );
        metadata.insert("brand".to_string(), json!("TechCorp"));
        metadata.insert("year".to_string(), json!(2023));
        metadata.insert(
            "description".to_string(),
            json!("Gaming electronics for professionals"),
        );
        metadata
    }

    #[test]
    fn test_simple_equality() {
        let mut engine = MetadataQueryEngine::new();
        let metadata = create_test_metadata();

        let query = MetadataQuery::field_eq("category", json!("electronics"));
        assert!(engine.evaluate(&query, &metadata).unwrap());

        let query = MetadataQuery::field_eq("category", json!("books"));
        assert!(!engine.evaluate(&query, &metadata).unwrap());
    }

    #[test]
    fn test_numeric_comparisons() {
        let mut engine = MetadataQueryEngine::new();
        let metadata = create_test_metadata();

        let query = MetadataQuery::Field(FieldQuery {
            field: "price".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: json!(50.0),
        });
        assert!(engine.evaluate(&query, &metadata).unwrap());

        let query = MetadataQuery::Field(FieldQuery {
            field: "year".to_string(),
            operator: ComparisonOperator::LessThan,
            value: json!(2025),
        });
        assert!(engine.evaluate(&query, &metadata).unwrap());
    }

    #[test]
    fn test_logical_operators() {
        let mut engine = MetadataQueryEngine::new();
        let metadata = create_test_metadata();

        let query = MetadataQuery::And(vec![
            MetadataQuery::field_eq("category", json!("electronics")),
            MetadataQuery::Field(FieldQuery {
                field: "price".to_string(),
                operator: ComparisonOperator::LessThan,
                value: json!(150.0),
            }),
        ]);
        assert!(engine.evaluate(&query, &metadata).unwrap());

        let query = MetadataQuery::Or(vec![
            MetadataQuery::field_eq("category", json!("books")),
            MetadataQuery::field_eq("category", json!("electronics")),
        ]);
        assert!(engine.evaluate(&query, &metadata).unwrap());

        let query = MetadataQuery::Not(Box::new(MetadataQuery::field_eq(
            "category",
            json!("books"),
        )));
        assert!(engine.evaluate(&query, &metadata).unwrap());
    }

    #[test]
    fn test_field_range() {
        let mut engine = MetadataQueryEngine::new();
        let metadata = create_test_metadata();

        let query = MetadataQuery::field_range("price", 50.0, 200.0);
        assert!(engine.evaluate(&query, &metadata).unwrap());

        let query = MetadataQuery::field_range("price", 200.0, 300.0);
        assert!(!engine.evaluate(&query, &metadata).unwrap());
    }

    #[test]
    fn test_query_builder() {
        let mut engine = MetadataQueryEngine::new();
        let metadata = create_test_metadata();

        let query = MetadataQueryBuilder::new()
            .field_equals("category", json!("electronics"))
            .build();

        assert!(engine.evaluate(&query, &metadata).unwrap());
    }

    #[test]
    fn string_array_existence_and_regex_operators_cover_success_and_miss_paths() {
        let mut engine = MetadataQueryEngine::default();
        let mut metadata = create_test_metadata();
        metadata.insert("nullable".to_string(), JsonValue::Null);

        for (field, operator, value) in [
            (
                "description",
                ComparisonOperator::Contains,
                json!("electronics"),
            ),
            ("brand", ComparisonOperator::StartsWith, json!("Tech")),
            ("brand", ComparisonOperator::EndsWith, json!("Corp")),
        ] {
            let query = MetadataQuery::Field(FieldQuery {
                field: field.to_string(),
                operator,
                value,
            });
            assert!(engine.evaluate(&query, &metadata).unwrap());
        }

        let contains = MetadataQuery::field_contains("description", "professionals");
        assert!(engine.evaluate(&contains, &metadata).unwrap());
        let non_string_contains = MetadataQuery::field_contains("price", "99");
        assert!(!engine.evaluate(&non_string_contains, &metadata).unwrap());

        let in_query =
            MetadataQuery::field_in("category", vec![json!("books"), json!("electronics")]);
        assert!(engine.evaluate(&in_query, &metadata).unwrap());
        let not_in_query = MetadataQuery::Field(FieldQuery {
            field: "category".to_string(),
            operator: ComparisonOperator::NotIn,
            value: json!(["books", "music"]),
        });
        assert!(engine.evaluate(&not_in_query, &metadata).unwrap());
        let malformed_in_query = MetadataQuery::Field(FieldQuery {
            field: "category".to_string(),
            operator: ComparisonOperator::In,
            value: json!("electronics"),
        });
        assert!(!engine.evaluate(&malformed_in_query, &metadata).unwrap());

        assert!(
            engine
                .evaluate(&MetadataQuery::field_exists("brand"), &metadata)
                .unwrap()
        );
        assert!(
            !engine
                .evaluate(&MetadataQuery::field_exists("nullable"), &metadata)
                .unwrap()
        );
        assert!(
            engine
                .evaluate(
                    &MetadataQuery::Field(FieldQuery {
                        field: "missing".to_string(),
                        operator: ComparisonOperator::NotExists,
                        value: JsonValue::Null,
                    }),
                    &metadata,
                )
                .unwrap()
        );

        let regex_query = MetadataQuery::Field(FieldQuery {
            field: "brand".to_string(),
            operator: ComparisonOperator::Regex,
            value: json!("^Tech.*rp$"),
        });
        assert!(engine.evaluate(&regex_query, &metadata).unwrap());
        assert!(engine.evaluate(&regex_query, &metadata).unwrap());

        let regex_non_string_pattern = MetadataQuery::Field(FieldQuery {
            field: "brand".to_string(),
            operator: ComparisonOperator::Regex,
            value: json!(42),
        });
        assert!(
            !engine
                .evaluate(&regex_non_string_pattern, &metadata)
                .unwrap()
        );
        let invalid_regex = MetadataQuery::Field(FieldQuery {
            field: "brand".to_string(),
            operator: ComparisonOperator::Regex,
            value: json!("["),
        });
        let err = engine.evaluate(&invalid_regex, &metadata).unwrap_err();
        assert!(err.to_string().contains("Invalid regex pattern"));
    }

    #[test]
    fn comparison_builder_and_logical_edge_cases_cover_all_branches() {
        let mut engine = MetadataQueryEngine::new();
        let metadata = create_test_metadata();

        for (field, operator, value, expected) in [
            (
                "year",
                ComparisonOperator::LessThanOrEqual,
                json!(2023),
                true,
            ),
            (
                "year",
                ComparisonOperator::GreaterThanOrEqual,
                json!(2023),
                true,
            ),
            ("brand", ComparisonOperator::LessThan, json!("Zoo"), true),
            (
                "brand",
                ComparisonOperator::GreaterThan,
                json!("Apple"),
                true,
            ),
            (
                "brand",
                ComparisonOperator::LessThanOrEqual,
                json!("TechCorp"),
                true,
            ),
            (
                "brand",
                ComparisonOperator::GreaterThanOrEqual,
                json!("TechCorp"),
                true,
            ),
            (
                "price",
                ComparisonOperator::GreaterThan,
                json!("bad"),
                false,
            ),
            ("missing", ComparisonOperator::GreaterThan, json!(1), false),
            (
                "category",
                ComparisonOperator::NotEquals,
                json!("books"),
                true,
            ),
            (
                "category",
                ComparisonOperator::NotEquals,
                json!("electronics"),
                false,
            ),
        ] {
            let query = MetadataQuery::Field(FieldQuery {
                field: field.to_string(),
                operator,
                value,
            });
            assert_eq!(engine.evaluate(&query, &metadata).unwrap(), expected);
        }

        assert!(engine.evaluate(&MetadataQuery::All, &metadata).unwrap());
        assert!(!engine.evaluate(&MetadataQuery::None, &metadata).unwrap());
        assert!(
            engine
                .evaluate(&MetadataQueryBuilder::and(Vec::new()), &metadata)
                .unwrap()
        );
        assert!(
            !engine
                .evaluate(&MetadataQueryBuilder::or(Vec::new()), &metadata)
                .unwrap()
        );

        let single = MetadataQuery::field_eq("category", json!("electronics"));
        assert_eq!(
            MetadataQueryBuilder::and(vec![single.clone()]),
            single.clone()
        );
        assert_eq!(
            MetadataQueryBuilder::or(vec![single.clone()]),
            single.clone()
        );
        assert!(matches!(
            MetadataQueryBuilder::and(vec![single.clone(), MetadataQuery::All]),
            MetadataQuery::And(_)
        ));
        assert!(matches!(
            MetadataQueryBuilder::or(vec![single.clone(), MetadataQuery::None]),
            MetadataQuery::Or(_)
        ));
        assert!(matches!(
            MetadataQueryBuilder::not_condition(single.clone()),
            MetadataQuery::Not(_)
        ));

        let built = MetadataQueryBuilder::default()
            .field_equals("category", json!("electronics"))
            .field_compare("price", ComparisonOperator::LessThanOrEqual, json!(99.99))
            .build();
        assert!(matches!(built, MetadataQuery::And(_)));
        assert!(engine.evaluate(&built, &metadata).unwrap());
    }
}
