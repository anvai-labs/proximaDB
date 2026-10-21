/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Metadata validation for ProximaRecord
//!
//! Provides validation of metadata fields in canonical records to prevent
//! SQL injection and ensure data integrity.

use super::type_validators::{
    BinaryValidator, JsonValidator, TypeValidationResult, ValidationError,
    contains_sql_injection_pattern,
};
use once_cell::sync::Lazy;
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};
use regex::Regex;
use std::collections::HashMap;
use tracing::{debug, warn};

/// Maximum allowed string length in metadata (configurable)
const DEFAULT_MAX_STRING_LENGTH: usize = 65536; // 64KB

/// Maximum allowed binary size in metadata (configurable)
const DEFAULT_MAX_BINARY_SIZE: usize = 1024 * 1024; // 1MB

/// Maximum allowed JSON nesting depth
const DEFAULT_MAX_JSON_DEPTH: usize = 32;

/// Maximum number of metadata fields
const DEFAULT_MAX_METADATA_FIELDS: usize = 1000;

/// Configuration for metadata validation
#[derive(Debug, Clone)]
pub struct MetadataValidationConfig {
    /// Enable SQL injection detection
    pub check_sql_injection: bool,
    /// Maximum string length
    pub max_string_length: usize,
    /// Maximum binary size
    pub max_binary_size: usize,
    /// Maximum JSON nesting depth
    pub max_json_depth: usize,
    /// Maximum number of metadata fields
    pub max_metadata_fields: usize,
    /// Enable strict mode (reject any suspicious patterns)
    pub strict_mode: bool,
}

impl Default for MetadataValidationConfig {
    fn default() -> Self {
        Self {
            check_sql_injection: true,
            max_string_length: DEFAULT_MAX_STRING_LENGTH,
            max_binary_size: DEFAULT_MAX_BINARY_SIZE,
            max_json_depth: DEFAULT_MAX_JSON_DEPTH,
            max_metadata_fields: DEFAULT_MAX_METADATA_FIELDS,
            strict_mode: false,
        }
    }
}

impl MetadataValidationConfig {
    /// Create a strict validation config
    pub fn strict() -> Self {
        Self {
            check_sql_injection: true,
            max_string_length: 8192, // 8KB
            max_binary_size: 65536,  // 64KB
            max_json_depth: 16,
            max_metadata_fields: 100,
            strict_mode: true,
        }
    }

    /// Create a permissive validation config (only basic checks)
    pub fn permissive() -> Self {
        Self {
            check_sql_injection: true,
            max_string_length: 1024 * 1024,    // 1MB
            max_binary_size: 10 * 1024 * 1024, // 10MB
            max_json_depth: 64,
            max_metadata_fields: 10000,
            strict_mode: false,
        }
    }
}

/// Metadata validator for ProximaRecord metadata fields
#[derive(Debug, Clone)]
pub struct MetadataValidator {
    /// Validation configuration (field limits, injection checks, nesting depth)
    config: MetadataValidationConfig,
    /// Validator for binary metadata values (size limits, magic byte checks)
    binary_validator: BinaryValidator,
    /// Validator for nested JSON metadata values (depth limits)
    #[allow(dead_code)]
    json_validator: JsonValidator,
}

impl Default for MetadataValidator {
    fn default() -> Self {
        Self::new(MetadataValidationConfig::default())
    }
}

impl MetadataValidator {
    /// Create a new metadata validator with the given configuration
    pub fn new(config: MetadataValidationConfig) -> Self {
        Self {
            binary_validator: BinaryValidator::new(config.max_binary_size),
            json_validator: JsonValidator::with_max_depth(config.max_json_depth),
            config,
        }
    }

    /// Validate a single ProximaRecord's metadata
    ///
    /// Returns Ok(()) if valid, or a vector of validation errors
    pub fn validate_record_metadata(
        &self,
        record: &ProximaRecord,
    ) -> Result<(), Vec<(String, ValidationError)>> {
        let mut errors = Vec::new();

        // Check metadata field count
        if record.props.len() > self.config.max_metadata_fields {
            errors.push((
                "_metadata".to_string(),
                ValidationError::LengthExceeded {
                    actual: record.props.len(),
                    max: self.config.max_metadata_fields,
                },
            ));
        }

        // Validate each metadata field
        for (key, value) in &record.props {
            // Validate key
            if let Err(e) = self.validate_field_key(key) {
                errors.push((key.clone(), e));
            }

            // Validate value
            if let Err(e) = self.validate_tree_node(key, value, 0) {
                errors.push((key.clone(), e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate a metadata field key
    fn validate_field_key(&self, key: &str) -> TypeValidationResult {
        // Key length check
        if key.len() > 256 {
            return Err(ValidationError::LengthExceeded {
                actual: key.len(),
                max: 256,
            });
        }

        // Empty key check
        if key.is_empty() {
            return Err(ValidationError::InvalidFormat {
                type_name: "metadata_key".to_string(),
                value: "empty key not allowed".to_string(),
            });
        }

        // SQL injection check on key
        if self.config.check_sql_injection && contains_sql_injection_pattern(key) {
            return Err(ValidationError::SecurityViolation {
                reason: format!("Potential SQL injection detected in metadata key: {}", key),
            });
        }

        Ok(())
    }

    /// Validate a ProximaTreeNode recursively.
    fn validate_tree_node(
        &self,
        field_name: &str,
        value: &ProximaTreeNode,
        depth: usize,
    ) -> TypeValidationResult {
        // Check nesting depth
        if depth > self.config.max_json_depth {
            return Err(ValidationError::JsonDepthExceeded {
                depth,
                max: self.config.max_json_depth,
            });
        }

        match value {
            ProximaTreeNode::Value(value) => self.validate_proxima_value(field_name, value, depth),
            ProximaTreeNode::Object(tree) => {
                for (key, value) in tree {
                    self.validate_field_key(key)?;
                    let nested_field = format!("{}.{}", field_name, key);
                    self.validate_tree_node(&nested_field, value, depth + 1)?;
                }
                Ok(())
            }
        }
    }

    /// Validate a ProximaValue recursively.
    fn validate_proxima_value(
        &self,
        field_name: &str,
        value: &ProximaValue,
        depth: usize,
    ) -> TypeValidationResult {
        if depth > self.config.max_json_depth {
            return Err(ValidationError::JsonDepthExceeded {
                depth,
                max: self.config.max_json_depth,
            });
        }

        match value {
            ProximaValue::String(s) | ProximaValue::Symbol(s) | ProximaValue::Decimal(s) => {
                self.validate_string_value(field_name, s)
            }
            ProximaValue::Binary(bytes) | ProximaValue::BinaryVector(bytes) => {
                self.binary_validator.validate(bytes)
            }
            ProximaValue::Array(values) => {
                for (idx, item) in values.iter().enumerate() {
                    let nested_field = format!("{}[{}]", field_name, idx);
                    self.validate_proxima_value(&nested_field, item, depth + 1)?;
                }
                Ok(())
            }
            ProximaValue::Map(values) | ProximaValue::Struct(values) => {
                for (key, value) in values {
                    self.validate_field_key(key)?;
                    let nested_field = format!("{}.{}", field_name, key);
                    self.validate_proxima_value(&nested_field, value, depth + 1)?;
                }
                Ok(())
            }
            ProximaValue::Float16(n) | ProximaValue::Float32(n) => {
                self.validate_finite_number(*n as f64)
            }
            ProximaValue::Float64(n) => self.validate_finite_number(*n),
            ProximaValue::DenseVector(values) => {
                for value in values {
                    self.validate_finite_number(*value as f64)?;
                }
                Ok(())
            }
            ProximaValue::SparseVector { values, .. } => {
                for value in values {
                    self.validate_finite_number(*value as f64)?;
                }
                Ok(())
            }
            ProximaValue::Json(value) | ProximaValue::Jsonb(value) => {
                self.validate_json_value(field_name, value, depth)
            }
            ProximaValue::Boolean(_)
            | ProximaValue::Int8(_)
            | ProximaValue::Int16(_)
            | ProximaValue::Int32(_)
            | ProximaValue::Int64(_)
            | ProximaValue::UInt8(_)
            | ProximaValue::UInt16(_)
            | ProximaValue::UInt32(_)
            | ProximaValue::UInt64(_)
            | ProximaValue::Date(_)
            | ProximaValue::Time(_, _)
            | ProximaValue::Timestamp(_, _)
            | ProximaValue::TimestampTz(_, _)
            | ProximaValue::Uuid(_)
            | ProximaValue::ULID(_)
            | ProximaValue::Null => Ok(()),
        }
    }

    fn validate_json_value(
        &self,
        field_name: &str,
        value: &serde_json::Value,
        depth: usize,
    ) -> TypeValidationResult {
        if depth > self.config.max_json_depth {
            return Err(ValidationError::JsonDepthExceeded {
                depth,
                max: self.config.max_json_depth,
            });
        }

        match value {
            serde_json::Value::String(s) => self.validate_string_value(field_name, s),
            serde_json::Value::Number(n) => {
                if let Some(value) = n.as_f64() {
                    self.validate_finite_number(value)
                } else {
                    Ok(())
                }
            }
            serde_json::Value::Array(values) => {
                for (idx, item) in values.iter().enumerate() {
                    let nested_field = format!("{}[{}]", field_name, idx);
                    self.validate_json_value(&nested_field, item, depth + 1)?;
                }
                Ok(())
            }
            serde_json::Value::Object(values) => {
                for (key, item) in values {
                    self.validate_field_key(key)?;
                    let nested_field = format!("{}.{}", field_name, key);
                    self.validate_json_value(&nested_field, item, depth + 1)?;
                }
                Ok(())
            }
            serde_json::Value::Bool(_) | serde_json::Value::Null => Ok(()),
        }
    }

    fn validate_finite_number(&self, value: f64) -> TypeValidationResult {
        if value.is_nan() {
            return Err(ValidationError::InvalidFormat {
                type_name: "number".to_string(),
                value: "NaN not allowed".to_string(),
            });
        }
        if value.is_infinite() {
            return Err(ValidationError::InvalidFormat {
                type_name: "number".to_string(),
                value: "Infinity not allowed".to_string(),
            });
        }
        Ok(())
    }

    /// Validate a string value for SQL injection and length
    fn validate_string_value(&self, field_name: &str, value: &str) -> TypeValidationResult {
        // Length check
        if value.len() > self.config.max_string_length {
            return Err(ValidationError::LengthExceeded {
                actual: value.len(),
                max: self.config.max_string_length,
            });
        }

        // SQL injection check
        if self.config.check_sql_injection && contains_sql_injection_pattern(value) {
            warn!(
                "SQL injection pattern detected in field '{}': truncated value '{}'",
                field_name,
                value.chars().take(100).collect::<String>()
            );
            return Err(ValidationError::SecurityViolation {
                reason: format!("Potential SQL injection detected in field '{}'", field_name),
            });
        }

        Ok(())
    }

    /// Validate a batch of ProximaRecords
    ///
    /// Returns a map of record IDs to their validation errors
    pub fn validate_batch(
        &self,
        records: &[ProximaRecord],
    ) -> HashMap<String, Vec<(String, ValidationError)>> {
        let mut result = HashMap::new();

        for record in records {
            if let Err(errors) = self.validate_record_metadata(record) {
                result.insert(record.oid.clone(), errors);
            }
        }

        result
    }

    /// Quick check if a batch has any validation errors
    pub fn is_batch_valid(&self, records: &[ProximaRecord]) -> bool {
        for record in records {
            if self.validate_record_metadata(record).is_err() {
                return false;
            }
        }
        true
    }
}

/// Collection name validator
///
/// Validates collection names to prevent injection attacks and ensure
/// valid naming conventions.
#[derive(Debug, Clone)]
pub struct CollectionNameValidator {
    /// Maximum collection name length
    max_length: usize,
    /// Minimum collection name length
    min_length: usize,
}

/// Valid collection name pattern: alphanumeric, underscore, hyphen
/// Must start with a letter or underscore
static COLLECTION_NAME_PATTERN: Lazy<Regex> = Lazy::new(|| {
    #[allow(clippy::panic)] // Intentional: compile-time invariant check
    Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
        .unwrap_or_else(|_| panic!("Invalid collection name regex"))
});

/// Reserved collection names
static RESERVED_NAMES: Lazy<Vec<&str>> = Lazy::new(|| {
    vec![
        "system",
        "admin",
        "config",
        "metadata",
        "internal",
        "null",
        "undefined",
        "true",
        "false",
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "table",
        "database",
        "schema",
        "index",
    ]
});

impl Default for CollectionNameValidator {
    fn default() -> Self {
        Self {
            max_length: 256,
            min_length: 1,
        }
    }
}

impl CollectionNameValidator {
    /// Create a new collection name validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom length limits
    pub fn with_length_limits(min: usize, max: usize) -> Self {
        Self {
            max_length: max,
            min_length: min,
        }
    }

    /// Validate a collection name
    pub fn validate(&self, name: &str) -> TypeValidationResult {
        // Length checks
        if name.len() < self.min_length {
            return Err(ValidationError::InvalidFormat {
                type_name: "collection_name".to_string(),
                value: format!("Name too short (min {} characters)", self.min_length),
            });
        }

        if name.len() > self.max_length {
            return Err(ValidationError::LengthExceeded {
                actual: name.len(),
                max: self.max_length,
            });
        }

        // Pattern check
        if !COLLECTION_NAME_PATTERN.is_match(name) {
            return Err(ValidationError::PatternMismatch {
                pattern: "Collection names must start with a letter or underscore and contain only alphanumeric characters, underscores, or hyphens".to_string(),
            });
        }

        // Reserved name check
        let lower_name = name.to_lowercase();
        if RESERVED_NAMES.contains(&lower_name.as_str()) {
            return Err(ValidationError::SecurityViolation {
                reason: format!("'{}' is a reserved collection name", name),
            });
        }

        // SQL injection check
        if contains_sql_injection_pattern(name) {
            return Err(ValidationError::SecurityViolation {
                reason: "Potential SQL injection detected in collection name".to_string(),
            });
        }

        debug!("Collection name '{}' passed validation", name);
        Ok(())
    }
}

/// Convenience function to validate a single record's metadata
pub fn validate_record_metadata(
    record: &ProximaRecord,
) -> Result<(), Vec<(String, ValidationError)>> {
    MetadataValidator::default().validate_record_metadata(record)
}

/// Convenience function to validate a collection name
pub fn validate_collection_name(name: &str) -> TypeValidationResult {
    CollectionNameValidator::default().validate(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(id: &str, metadata: HashMap<String, ProximaValue>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            props: metadata
                .into_iter()
                .map(|(key, value)| (key, ProximaTreeNode::Value(value)))
                .collect(),
            ..ProximaRecord::default()
        }
    }

    fn string_value(s: &str) -> ProximaValue {
        ProximaValue::String(s.to_string())
    }

    #[test]
    fn test_valid_metadata() {
        let validator = MetadataValidator::default();
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), string_value("value1"));
        metadata.insert("key2".to_string(), string_value("value2"));

        let record = create_test_record("test-1", metadata);
        assert!(validator.validate_record_metadata(&record).is_ok());
    }

    #[test]
    fn test_sql_injection_in_value() {
        let validator = MetadataValidator::default();
        let mut metadata = HashMap::new();
        metadata.insert("evil".to_string(), string_value("'; DROP TABLE users; --"));

        let record = create_test_record("test-1", metadata);
        let result = validator.validate_record_metadata(&record);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|(_, e)| matches!(e, ValidationError::SecurityViolation { .. }))
        );
    }

    #[test]
    fn test_sql_injection_in_key() {
        let validator = MetadataValidator::default();
        let mut metadata = HashMap::new();
        metadata.insert("SELECT * FROM users".to_string(), string_value("value"));

        let record = create_test_record("test-1", metadata);
        let result = validator.validate_record_metadata(&record);
        assert!(result.is_err());
    }

    #[test]
    fn test_string_too_long() {
        let config = MetadataValidationConfig {
            max_string_length: 10,
            ..Default::default()
        };
        let validator = MetadataValidator::new(config);
        let mut metadata = HashMap::new();
        metadata.insert(
            "key".to_string(),
            string_value("this string is too long for validation"),
        );

        let record = create_test_record("test-1", metadata);
        let result = validator.validate_record_metadata(&record);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|(_, e)| matches!(e, ValidationError::LengthExceeded { .. }))
        );
    }

    #[test]
    fn test_too_many_fields() {
        let config = MetadataValidationConfig {
            max_metadata_fields: 2,
            ..Default::default()
        };
        let validator = MetadataValidator::new(config);
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), string_value("value1"));
        metadata.insert("key2".to_string(), string_value("value2"));
        metadata.insert("key3".to_string(), string_value("value3"));

        let record = create_test_record("test-1", metadata);
        let result = validator.validate_record_metadata(&record);
        assert!(result.is_err());
    }

    #[test]
    fn test_collection_name_valid() {
        let validator = CollectionNameValidator::default();
        assert!(validator.validate("my_collection").is_ok());
        assert!(validator.validate("Collection123").is_ok());
        assert!(validator.validate("_private").is_ok());
        assert!(validator.validate("test-collection").is_ok());
    }

    #[test]
    fn test_collection_name_invalid() {
        let validator = CollectionNameValidator::default();

        // Cannot start with number
        assert!(validator.validate("123collection").is_err());

        // Cannot contain spaces
        assert!(validator.validate("my collection").is_err());

        // Cannot contain special characters
        assert!(validator.validate("my@collection").is_err());
    }

    #[test]
    fn test_collection_name_reserved() {
        let validator = CollectionNameValidator::default();
        assert!(validator.validate("system").is_err());
        assert!(validator.validate("SELECT").is_err());
        assert!(validator.validate("drop").is_err());
    }

    #[test]
    fn test_collection_name_sql_injection() {
        let validator = CollectionNameValidator::default();
        assert!(validator.validate("users; DROP TABLE users").is_err());
    }

    #[test]
    fn test_batch_validation() {
        let validator = MetadataValidator::default();

        let mut good_metadata = HashMap::new();
        good_metadata.insert("key".to_string(), string_value("value"));
        let good_record = create_test_record("good", good_metadata);

        let mut bad_metadata = HashMap::new();
        bad_metadata.insert("key".to_string(), string_value("'; DROP TABLE users;"));
        let bad_record = create_test_record("bad", bad_metadata);

        let errors = validator.validate_batch(&[good_record, bad_record]);
        assert_eq!(errors.len(), 1);
        assert!(errors.contains_key("bad"));
    }
}
