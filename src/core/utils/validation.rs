//! # Validation Utilities Module
//!
//! This module provides comprehensive validation functions for various data types
//! used throughout ProximaDB. It consolidates validation logic that was previously
//! scattered across different modules, ensuring consistent error handling and
//! validation rules.
//!
//! ## Categories of Validation
//!
//! - **ID Validation**: Vector IDs, collection names, field names
//! - **Range Validation**: Numeric ranges, dimensions, limits
//! - **Format Validation**: String formats, naming conventions
//! - **Semantic Validation**: Business logic validation
//!
//! ## Design Principles
//!
//! 1. **Fail Fast**: Validate early to catch errors at boundaries
//! 2. **Descriptive Errors**: Provide clear error messages with context
//! 3. **Performance**: Minimal overhead for hot path validations
//! 4. **Consistency**: Same validation rules across all components

use anyhow::{Result, anyhow, bail};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

/// ## ID Validation
///
/// ### Vector ID Validation
///
/// Validates that a vector ID meets ProximaDB's requirements:
/// - Non-empty
/// - Maximum length of 256 characters
/// - Contains only alphanumeric, underscore, hyphen, or dot
/// - Does not start with a number
///
/// #### Arguments
/// * `id` - The vector ID to validate
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_vector_id(id: &str) -> Result<()> {
    if id.is_empty() {
        bail!("Vector ID cannot be empty");
    }

    if id.len() > 256 {
        bail!(
            "Vector ID exceeds maximum length of 256 characters: {}",
            id.len()
        );
    }

    // Check first character
    if id.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        bail!("Vector ID cannot start with a number: {}", id);
    }

    // Check allowed characters
    static VALID_ID_REGEX: Lazy<Option<Regex>> =
        Lazy::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_.-]*$").ok());
    let valid_id_regex = VALID_ID_REGEX
        .as_ref()
        .ok_or_else(|| anyhow!("Internal error: vector ID regex failed to initialize"))?;

    if !valid_id_regex.is_match(id) {
        bail!(
            "Vector ID contains invalid characters. Only alphanumeric, underscore, hyphen, and dot are allowed: {}",
            id
        );
    }

    Ok(())
}

/// ### Collection Name Validation
///
/// Validates collection names according to ProximaDB naming rules:
/// - 3-64 characters long
/// - Starts with a letter
/// - Contains only lowercase letters, numbers, and underscores
///
/// #### Arguments
/// * `name` - The collection name to validate
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_collection_name(name: &str) -> Result<()> {
    if name.len() < 3 || name.len() > 64 {
        bail!(
            "Collection name must be between 3 and 64 characters: {}",
            name.len()
        );
    }

    static COLLECTION_NAME_REGEX: Lazy<Option<Regex>> =
        Lazy::new(|| Regex::new(r"^[a-z][a-z0-9_]*$").ok());
    let collection_name_regex = COLLECTION_NAME_REGEX
        .as_ref()
        .ok_or_else(|| anyhow!("Internal error: collection name regex failed to initialize"))?;

    if !collection_name_regex.is_match(name) {
        bail!(
            "Collection name must start with a lowercase letter and contain only lowercase letters, numbers, and underscores: {}",
            name
        );
    }

    // Check for reserved names
    static RESERVED_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
        [
            "system", "admin", "config", "test", "temp", "tmp", "internal",
        ]
        .iter()
        .cloned()
        .collect()
    });

    if RESERVED_NAMES.contains(name) {
        bail!("Collection name '{}' is reserved", name);
    }

    Ok(())
}

/// ### Field Name Validation
///
/// Validates metadata field names:
/// - Non-empty
/// - Maximum 128 characters
/// - No special characters except underscore
/// - Cannot start with double underscore (reserved for system fields)
///
/// #### Arguments
/// * `field` - The field name to validate
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_field_name(field: &str) -> Result<()> {
    if field.is_empty() {
        bail!("Field name cannot be empty");
    }

    if field.len() > 128 {
        bail!(
            "Field name exceeds maximum length of 128 characters: {}",
            field.len()
        );
    }

    if field.starts_with("__") {
        bail!(
            "Field names starting with '__' are reserved for system use: {}",
            field
        );
    }

    static FIELD_NAME_REGEX: Lazy<Option<Regex>> =
        Lazy::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").ok());
    let field_name_regex = FIELD_NAME_REGEX
        .as_ref()
        .ok_or_else(|| anyhow!("Internal error: field name regex failed to initialize"))?;

    if !field_name_regex.is_match(field) {
        bail!(
            "Field name contains invalid characters. Only alphanumeric and underscore are allowed: {}",
            field
        );
    }

    Ok(())
}

/// ## Range Validation
///
/// ### Dimension Validation
///
/// Validates vector dimensions are within acceptable ranges:
/// - Minimum: 1
/// - Maximum: 65536 (configurable)
/// - Must be positive
///
/// #### Arguments
/// * `dimension` - The dimension to validate
/// * `max_allowed` - Optional maximum allowed dimension (defaults to 65536)
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_dimension(dimension: usize, max_allowed: Option<usize>) -> Result<()> {
    if dimension == 0 {
        bail!("Vector dimension must be positive");
    }

    let max = max_allowed.unwrap_or(65536);
    if dimension > max {
        bail!(
            "Vector dimension {} exceeds maximum allowed {}",
            dimension,
            max
        );
    }

    Ok(())
}

/// ### Batch Size Validation
///
/// Validates batch sizes for bulk operations:
/// - Minimum: 1
/// - Maximum: 10000 (configurable)
///
/// #### Arguments
/// * `batch_size` - The batch size to validate
/// * `max_allowed` - Optional maximum batch size (defaults to 10000)
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_batch_size(batch_size: usize, max_allowed: Option<usize>) -> Result<()> {
    if batch_size == 0 {
        bail!("Batch size must be at least 1");
    }

    let max = max_allowed.unwrap_or(10000);
    if batch_size > max {
        bail!("Batch size {} exceeds maximum allowed {}", batch_size, max);
    }

    Ok(())
}

/// ### Top-K Validation
///
/// Validates top-k parameter for search queries:
/// - Minimum: 1
/// - Maximum: 1000 (configurable)
///
/// #### Arguments
/// * `k` - The top-k value to validate
/// * `max_allowed` - Optional maximum k (defaults to 1000)
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_top_k(k: usize, max_allowed: Option<usize>) -> Result<()> {
    if k == 0 {
        bail!("Top-k must be at least 1");
    }

    let max = max_allowed.unwrap_or(1000);
    if k > max {
        bail!("Top-k {} exceeds maximum allowed {}", k, max);
    }

    Ok(())
}

/// ### Score Range Validation
///
/// Validates similarity scores are within expected ranges:
/// - For cosine similarity: [-1, 1]
/// - For distance metrics: [0, ∞) but typically bounded
///
/// #### Arguments
/// * `score` - The score to validate
/// * `metric_type` - Type of distance/similarity metric
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_score(score: f32, metric_type: &str) -> Result<()> {
    if !score.is_finite() {
        bail!("Score must be finite, got: {}", score);
    }

    match metric_type.to_lowercase().as_str() {
        "cosine" | "cosine_similarity" if !(-1.0..=1.0).contains(&score) => {
            bail!(
                "Cosine similarity score must be in range [-1, 1], got: {}",
                score
            );
        }
        "euclidean" | "l2" | "manhattan" | "hamming" if score < 0.0 => {
            bail!("Distance score must be non-negative, got: {}", score);
        }
        _ => {
            // Unknown metric (or in-range), just check for finite
        }
    }

    Ok(())
}

/// ## Format Validation
///
/// ### JSON Path Validation
///
/// Validates JSON path expressions for metadata filtering:
/// - Must start with $ or field name
/// - Valid operators: ., [], *
///
/// #### Arguments
/// * `path` - The JSON path to validate
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_json_path(path: &str) -> Result<()> {
    if path.is_empty() {
        bail!("JSON path cannot be empty");
    }

    // Basic validation - can be extended based on actual JSON path library used
    static JSON_PATH_REGEX: Lazy<Option<Regex>> = Lazy::new(|| {
        Regex::new(r"^(\$\.)?[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*(\[[0-9]+\])?$").ok()
    });
    let json_path_regex = JSON_PATH_REGEX
        .as_ref()
        .ok_or_else(|| anyhow!("Internal error: JSON path regex failed to initialize"))?;

    if !json_path_regex.is_match(path) {
        bail!("Invalid JSON path format: {}", path);
    }

    Ok(())
}

/// ### URL Validation
///
/// Validates URLs for remote storage or API endpoints:
/// - Must be valid URL format
/// - Supported schemes: http, https, s3, file
///
/// #### Arguments
/// * `url` - The URL to validate
/// * `allowed_schemes` - Optional list of allowed schemes
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_url(url: &str, allowed_schemes: Option<&[&str]>) -> Result<()> {
    use url::Url;

    let parsed = Url::parse(url).map_err(|e| anyhow!("Invalid URL format: {}", e))?;

    let default_schemes = vec!["http", "https", "s3", "file"];
    let schemes = allowed_schemes.unwrap_or(&default_schemes);

    if !schemes.contains(&parsed.scheme()) {
        bail!(
            "URL scheme '{}' not allowed. Supported: {:?}",
            parsed.scheme(),
            schemes
        );
    }

    Ok(())
}

/// ## Semantic Validation
///
/// ### Distance Metric Validation
///
/// Validates that a distance metric is supported:
///
/// #### Arguments
/// * `metric` - The distance metric name
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_distance_metric(metric: &str) -> Result<()> {
    static VALID_METRICS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
        [
            "cosine",
            "euclidean",
            "dot_product",
            "manhattan",
            "hamming",
            "jaccard",
            "chebyshev",
            "canberra",
            "minkowski",
            "angular",
            "bray_curtis",
            "hellinger",
        ]
        .iter()
        .cloned()
        .collect()
    });

    let metric_lower = metric.to_lowercase();
    if !VALID_METRICS.contains(metric_lower.as_str()) {
        bail!(
            "Unknown distance metric '{}'. Supported metrics: {:?}",
            metric,
            VALID_METRICS.iter().collect::<Vec<_>>()
        );
    }

    Ok(())
}

/// ### Storage Engine Validation
///
/// Validates that a storage engine name is valid:
///
/// #### Arguments
/// * `engine` - The storage engine name
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_storage_engine(engine: &str) -> Result<()> {
    static VALID_ENGINES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
        ["sst", "raptor", "nova", "swift", "prism", "helix", "tst"]
            .iter()
            .cloned()
            .collect()
    });

    let engine_lower = engine.to_lowercase();
    if !VALID_ENGINES.contains(engine_lower.as_str()) {
        bail!(
            "Unknown storage engine '{}'. Supported engines: {:?}",
            engine,
            VALID_ENGINES.iter().collect::<Vec<_>>()
        );
    }

    Ok(())
}

/// ### Quantization Level Validation
///
/// Validates quantization levels:
///
/// #### Arguments
/// * `level` - The quantization level name
///
/// #### Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_quantization_level(level: &str) -> Result<()> {
    static VALID_LEVELS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
        [
            "binary", "int4", "int8", "pq4", "pq8", "pq16", "fp16", "fp32", "none",
        ]
        .iter()
        .cloned()
        .collect()
    });

    let level_lower = level.to_lowercase();
    if !VALID_LEVELS.contains(level_lower.as_str()) {
        bail!(
            "Unknown quantization level '{}'. Supported levels: {:?}",
            level,
            VALID_LEVELS.iter().collect::<Vec<_>>()
        );
    }

    Ok(())
}

/// ## Composite Validation
///
/// ### Validate Insert Request
///
/// Comprehensive validation for vector insert requests:
///
/// #### Arguments
/// * `id` - Vector ID
/// * `vector` - Vector data
/// * `dimension` - Expected dimension
/// * `collection_name` - Target collection
///
/// #### Returns
/// * Ok(()) if all validations pass
pub fn validate_insert_request(
    id: &str,
    vector: &[f32],
    dimension: usize,
    collection_name: &str,
) -> Result<()> {
    validate_vector_id(id)?;
    validate_collection_name(collection_name)?;
    validate_dimension(dimension, None)?;

    // Validate vector
    crate::core::utils::vector_ops::validate_vector(vector, Some(dimension))?;

    Ok(())
}

/// ### Validate Search Request
///
/// Comprehensive validation for search requests:
///
/// #### Arguments
/// * `vector` - Query vector
/// * `dimension` - Expected dimension
/// * `top_k` - Number of results
/// * `metric` - Distance metric
///
/// #### Returns
/// * Ok(()) if all validations pass
pub fn validate_search_request(
    vector: &[f32],
    dimension: usize,
    top_k: usize,
    metric: &str,
) -> Result<()> {
    validate_dimension(dimension, None)?;
    validate_top_k(top_k, None)?;
    validate_distance_metric(metric)?;

    // Validate vector
    crate::core::utils::vector_ops::validate_vector(vector, Some(dimension))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_id_validation() {
        assert!(validate_vector_id("valid_id_123").is_ok());
        assert!(validate_vector_id("").is_err());
        assert!(validate_vector_id("123_starts_with_number").is_err());
        assert!(validate_vector_id("has spaces").is_err());
    }

    #[test]
    fn test_collection_name_validation() {
        assert!(validate_collection_name("my_collection").is_ok());
        assert!(validate_collection_name("ab").is_err()); // Too short
        assert!(validate_collection_name("MixedCase").is_err());
        assert!(validate_collection_name("system").is_err()); // Reserved
    }

    #[test]
    fn test_dimension_validation() {
        assert!(validate_dimension(128, None).is_ok());
        assert!(validate_dimension(0, None).is_err());
        assert!(validate_dimension(100000, None).is_err());
    }

    #[test]
    fn test_distance_metric_validation() {
        assert!(validate_distance_metric("cosine").is_ok());
        assert!(validate_distance_metric("EUCLIDEAN").is_ok());
        assert!(validate_distance_metric("unknown_metric").is_err());
    }
}
