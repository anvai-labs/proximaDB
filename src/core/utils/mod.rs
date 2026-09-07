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
                // Line comments: an apostrophe INSIDE '-- don't' must not
                // open quote state (it swallowed every target after it).
                if bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'-' {
                    i += 2;
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                    continue;
                }
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

/// Shared f64→f32 narrowing guard: the narrowing can overflow to inf
/// (1e300), and non-finite components must never dispatch to the distance
/// kernels. Call sites migrate here incrementally so the policy has one
/// auditable implementation.
pub fn finite_f32(value: f64) -> Option<f32> {
    let narrowed = value as f32;
    narrowed.is_finite().then_some(narrowed)
}

#[cfg(test)]
mod tests {
    use super::finite_f32;

    #[test]
    fn finite_f32_rejects_non_finite_and_overflowed_values() {
        assert_eq!(finite_f32(1.25), Some(1.25));
        assert_eq!(finite_f32(f32::MAX as f64), Some(f32::MAX));
        assert_eq!(finite_f32(f64::NAN), None);
        assert_eq!(finite_f32(f64::INFINITY), None);
        assert_eq!(finite_f32(1e300), None);
    }
}

/// Bind placeholders ('$N' and bare '?') cannot resolve during EXPLAIN —
/// they are not catalog targets.
fn is_bind_placeholder(value: &str) -> bool {
    value.starts_with('$') || value == "?"
}

pub(crate) fn collect_quoted_first_args(sql: &str, function_name: &str, targets: &mut Vec<String>) {
    // ASCII-case search on the original (shared helper — offsets in a
    // to_uppercase copy can slice mid-character).
    let mut search_start = 0;

    // Quote-aware: names inside string literals or longer identifiers
    // must not mint catalog targets (find_ascii_ci matched both).
    while let Some(relative_pos) = find_ascii_ci_outside_quotes(&sql[search_start..], function_name)
    {
        let name_start = search_start + relative_pos;
        let after_name = name_start + function_name.len();
        let Some(open_relative) = sql[after_name..].find('(') else {
            break;
        };
        let mut arg_start = after_name + open_relative + 1;
        while let Some(ch) = sql[arg_start..].chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            arg_start += ch.len_utf8();
        }
        // Quoted first args may contain delimiter characters. Match the
        // fusion parser's SQL quote-doubling rules before considering the
        // simpler unquoted form.
        let opening_quote = sql.as_bytes().get(arg_start).copied();
        if matches!(opening_quote, Some(b'\'' | b'"')) {
            let quote = opening_quote.unwrap_or_default();
            let value_start = arg_start + 1;
            let mut scan = value_start;
            let mut closed = None;
            let bytes = sql.as_bytes();
            while scan < bytes.len() {
                if bytes[scan] == quote {
                    if scan + 1 < bytes.len() && bytes[scan + 1] == quote {
                        scan += 2;
                        continue;
                    }
                    closed = Some(scan);
                    break;
                }
                scan += 1;
            }
            if let Some(close) = closed {
                let escaped = if quote == b'\'' { "''" } else { "\"\"" };
                let replacement = if quote == b'\'' { "'" } else { "\"" };
                let value = sql[value_start..close]
                    .replace(escaped, replacement)
                    .trim()
                    .to_string();
                // '$'-binds skip on the QUOTED arm too (consistent with
                // the unquoted arm below).
                if !value.is_empty()
                    && !is_bind_placeholder(&value)
                    && !targets.iter().any(|existing| existing == &value)
                {
                    targets.push(value.clone());
                }
                search_start = close + 1;
                continue;
            }
        } else {
            // Unquoted first args are targets too; '$'-prefixed bind params
            // cannot be resolved during EXPLAIN and are deliberately skipped.
            let candidate_end = sql[arg_start..]
                .find([',', ')'])
                .map(|rel| arg_start + rel)
                .unwrap_or(sql.len());
            let candidate = sql[arg_start..candidate_end].trim();
            if !candidate.is_empty()
                && !is_bind_placeholder(candidate)
                && !targets.iter().any(|existing| existing == candidate)
            {
                targets.push(candidate.to_string());
            }
            search_start = candidate_end;
            continue;
        }
        search_start = after_name;
    }
}
