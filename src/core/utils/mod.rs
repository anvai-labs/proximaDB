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
/// Case-insensitive `IF NOT EXISTS` prefix strip with a word boundary
/// (whitespace OR an opening quote — `EXISTS"logs"` parses under the
/// pinned GenericDialect). Returns (had_prefix, rest).
pub fn strip_if_not_exists(input: &str) -> (bool, &str) {
    let Some(head) = input.get(.."IF NOT EXISTS".len()) else {
        return (false, input);
    };
    if !head.eq_ignore_ascii_case("IF NOT EXISTS") {
        return (false, input);
    }
    let next = input.as_bytes().get("IF NOT EXISTS".len());
    match next {
        None => (true, ""),
        Some(b) if b.is_ascii_whitespace() || *b == b'"' => {
            (true, input["IF NOT EXISTS".len()..].trim_start())
        }
        _ => (false, input),
    }
}

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
                // PostgreSQL block comments may nest. Ignore every nested
                // comment body so function-like text there cannot become an
                // EXPLAIN catalog target (and an apostrophe inside one must
                // not open quote state).
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    i += 2;
                    let mut depth = 1usize;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                            depth += 1;
                            i += 2;
                        } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                            depth -= 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    continue;
                }
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
    use super::{collect_quoted_first_args, finite_f32, inject_graph_target_into_cypher};

    #[test]
    fn finite_f32_rejects_non_finite_and_overflowed_values() {
        assert_eq!(finite_f32(1.25), Some(1.25));
        assert_eq!(finite_f32(f32::MAX as f64), Some(f32::MAX));
        assert_eq!(finite_f32(f64::NAN), None);
        assert_eq!(finite_f32(f64::INFINITY), None);
        assert_eq!(finite_f32(1e300), None);
    }

    #[test]
    fn target_scanner_requires_a_complete_function_call_name() {
        let mut targets = Vec::new();
        collect_quoted_first_args(
            "SELECT * FROM NOT_VECTOR_SEARCH('wrong', [1.0])",
            "VECTOR_SEARCH",
            &mut targets,
        );
        collect_quoted_first_args(
            "SELECT * FROM πVECTOR_SEARCH('unicode_wrong', [1.0])",
            "VECTOR_SEARCH",
            &mut targets,
        );
        collect_quoted_first_args(
            "/* outer /* VECTOR_SEARCH('block_wrong') */ comment */",
            "VECTOR_SEARCH",
            &mut targets,
        );
        collect_quoted_first_args(
            "SELECT VECTOR_SEARCH + other_call('also_wrong')",
            "VECTOR_SEARCH",
            &mut targets,
        );
        collect_quoted_first_args(
            "SELECT * FROM VECTOR_SEARCH ('right', [1.0])",
            "VECTOR_SEARCH",
            &mut targets,
        );

        assert_eq!(targets, ["right"]);
    }

    #[test]
    fn graph_target_injection_ignores_clause_text_inside_literals() {
        let cypher = "MATCH (n {note: ' FROM archived RETURN value'}) RETURN n";
        assert_eq!(
            inject_graph_target_into_cypher("social", cypher),
            "MATCH (n {note: ' FROM archived RETURN value'}) FROM social RETURN n"
        );
    }

    #[test]
    fn graph_target_injection_ignores_cypher_names_and_comments() {
        let quoted_name = "MATCH (n:` FROM archive RETURN value`) RETURN n";
        assert_eq!(
            inject_graph_target_into_cypher("social", quoted_name),
            "MATCH (n:` FROM archive RETURN value`) FROM social RETURN n"
        );

        let line_comment = "MATCH (n) // FROM archive RETURN value\nRETURN n";
        assert_eq!(
            inject_graph_target_into_cypher("social", line_comment),
            "MATCH (n) // FROM archive RETURN value\nFROM social RETURN n"
        );
    }
}

/// Bind placeholders ('$N' and bare '?') cannot resolve during EXPLAIN —
/// they are not catalog targets.
fn is_bind_placeholder(value: &str) -> bool {
    value.starts_with('$') || value == "?"
}

/// Extension functions whose FIRST argument is a catalog target
/// (collection/namespace) — the EXPLAIN authority scanner's list.
/// GRAPH_QUERY is EXCLUDED (its first arg is Cypher, never a target).
/// Keep in sync with the fusion parser's FUNCTION_NAMES registry —
/// derive from it when the terminal scanner home lands.
pub const CATALOG_FIRST_ARG_FUNCTIONS: [&str; 6] = [
    "VECTOR_SEARCH",
    "DOCUMENT_QUERY",
    "LOGS",
    "METRICS",
    "TRACES",
    "RERANK",
];

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
        // Keep searching after every rejected occurrence. SQL identifiers may
        // contain ASCII letters, digits, underscores, and dollar signs, so a
        // match within NOT_VECTOR_SEARCH or VECTOR_SEARCH_V2 is not this
        // function. Likewise, only whitespace may separate the name and `(`;
        // scanning forward to an unrelated call misattributes its first arg.
        search_start = after_name;
        // The ONE shared byte class — a divergent char-class here
        // tokenized '·LOGS' differently from the keyword scanners.
        let is_identifier_char = |ch: char| !ch.is_ascii() || is_identifier_byte(ch as u8);
        if sql[..name_start]
            .chars()
            .next_back()
            .is_some_and(is_identifier_char)
            || sql[after_name..]
                .chars()
                .next()
                .is_some_and(is_identifier_char)
        {
            continue;
        }

        let mut open = after_name;
        while let Some(ch) = sql[open..].chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            open += ch.len_utf8();
        }
        if sql.as_bytes().get(open) != Some(&b'(') {
            continue;
        }

        let mut arg_start = open + 1;
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
                // No trim: the runtime parser's unquote preserves inner
                // padding — EXPLAIN must resolve the SAME name execution
                // uses (' ops ' stayed untrimmed at runtime).
                let value = sql[value_start..close].replace(escaped, replacement);
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
        // Only reachable for an unterminated literal — resume past the
        // name to avoid an infinite loop.
        search_start = after_name;
    }
}

/// Find a whitespace-delimited Cypher keyword sequence outside strings,
/// backtick-quoted symbolic names, and comments. Cypher line comments use
/// `//`, and escaped quotes use backslashes, so the SQL-oriented scanner above
/// is deliberately not reused here.
pub fn find_cypher_clause(haystack: &str, keywords: &[&str]) -> Option<usize> {
    let first = keywords.first()?.as_bytes();
    let bytes = haystack.as_bytes();
    let mut quote: Option<u8> = None;
    let mut i = 0usize;

    while i < bytes.len() {
        if let Some(q) = quote {
            if bytes[i] == b'\\' && q != b'`' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if bytes[i] == q {
                if i + 1 < bytes.len() && bytes[i + 1] == q {
                    i += 2;
                    continue;
                }
                quote = None;
            }
            i += 1;
            continue;
        }

        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if matches!(bytes[i], b'\'' | b'"' | b'`') {
            quote = Some(bytes[i]);
            i += 1;
            continue;
        }

        let starts_at_boundary = i == 0 || bytes[i - 1].is_ascii_whitespace();
        if starts_at_boundary
            && i + first.len() <= bytes.len()
            && bytes[i..i + first.len()].eq_ignore_ascii_case(first)
        {
            let mut end = i + first.len();
            let mut matched = true;
            for keyword in &keywords[1..] {
                let whitespace_start = end;
                while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                    end += 1;
                }
                let keyword = keyword.as_bytes();
                if end == whitespace_start
                    || end + keyword.len() > bytes.len()
                    || !bytes[end..end + keyword.len()].eq_ignore_ascii_case(keyword)
                {
                    matched = false;
                    break;
                }
                end += keyword.len();
            }
            if matched && (end == bytes.len() || bytes[end].is_ascii_whitespace()) {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

pub fn inject_graph_target_into_cypher(graph: &str, cypher: &str) -> String {
    let graph = graph.trim();
    let cypher = cypher.trim().trim_end_matches(';').trim();

    if graph.is_empty() || graph == "default" {
        return cypher.to_string();
    }

    if find_cypher_clause(cypher, &["FROM"]).is_some() {
        return cypher.to_string();
    }

    let insertion_index = [
        find_cypher_clause(cypher, &["WHERE"]),
        find_cypher_clause(cypher, &["RETURN"]),
        find_cypher_clause(cypher, &["ORDER", "BY"]),
        find_cypher_clause(cypher, &["LIMIT"]),
        find_cypher_clause(cypher, &["SKIP"]),
    ]
    .into_iter()
    .flatten()
    .min();

    if let Some(index) = insertion_index {
        format!("{}FROM {} {}", &cypher[..index], graph, &cypher[index..])
    } else {
        format!("{} FROM {}", cypher, graph)
    }
}

/// The ONE identifier-continuation byte class ('$' and non-ASCII are
/// identifier bytes — Postgres identifiers may contain dollar signs and
/// Unicode letters). Shared by the keyword-boundary checks and the
/// identifier validators.
pub fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || b >= 0x80
}
