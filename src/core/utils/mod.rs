//! # Core Utilities Module
//!
//! This module contains common utility functions and helpers that are used across
//! multiple components of ProximaDB. It consolidates previously duplicated code
//! into reusable, well-documented, and tested utilities.
//!
//! ## Submodules
//!
//! - **metadata_conversions**: Conversions between different metadata representations
//! - **vector_ops**: Common vector operations and transformations
//! - **validation**: Input validation and sanitization utilities
//! - **config_utils**: Configuration parsing and validation helpers
//!
//! ## Design Principles
//!
//! 1. **Zero-cost abstractions**: Utilities should have minimal runtime overhead
//! 2. **Type safety**: Strong typing to catch errors at compile time
//! 3. **Documentation**: Every public function must be thoroughly documented
//! 4. **Testing**: Comprehensive unit tests for all utility functions
//! 5. **Performance**: SIMD and other optimizations where applicable
//!
//! ## Migration Guide
//!
//! When refactoring existing code to use these utilities:
//!
//! 1. Search for duplicate implementations of the same logic
//! 2. Replace with calls to the appropriate utility function
//! 3. Update imports to use `crate::core::utils::{module}::{function}`
//! 4. Run tests to ensure functionality is preserved
//! 5. Remove the old duplicate implementation

pub mod metadata_conversions;
pub mod validation;
pub mod vector_ops;

// Re-export commonly used functions for convenience
pub use metadata_conversions::{
    filter_metadata, json_to_metadata_item, json_to_proto_metadata, merge_metadata,
    proto_metadata_to_json,
};

pub use vector_ops::{
    cosine_similarity, dot_product, mean, normalize_l2, resize_vector, standard_deviation,
    validate_vector,
};

pub use validation::{
    validate_batch_size, validate_collection_name, validate_dimension, validate_distance_metric,
    validate_field_name, validate_storage_engine, validate_top_k, validate_vector_id,
};

/// ASCII-case-insensitive find returning a byte offset into the ORIGINAL
/// string — offsets found in a `to_uppercase()` copy can slice the original
/// mid-character (uppercase changes UTF-8 byte lengths, e.g. 'ﬀ' 3→2 bytes).
/// Shared by the pgwire protocol parser and the multi-model SQL builders.
pub fn find_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

/// [`find_ascii_ci`], but matches only OUTSIDE single/double-quoted
/// segments (quote-doubling aware) — SET-style name/value splits must not
/// fire inside a quoted identifier such as "time TO live".
pub fn find_ascii_ci_outside_quotes(haystack: &str, needle: &str) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let mut quote: Option<u8> = None;
    let mut i = 0;
    while i < bytes.len() {
        match quote {
            Some(q) => {
                if bytes[i] == q {
                    // doubled quote stays inside the literal
                    if i + 1 < bytes.len() && bytes[i + 1] == q {
                        i += 2;
                        continue;
                    }
                    quote = None;
                }
                i += 1;
            }
            None => {
                if bytes[i] == b'\'' || bytes[i] == b'"' {
                    quote = Some(bytes[i]);
                    i += 1;
                    continue;
                }
                if i + needle.len() <= bytes.len()
                    && bytes[i..i + needle.len()].eq_ignore_ascii_case(needle.as_bytes())
                {
                    return Some(i);
                }
                i += 1;
            }
        }
    }
    None
}
