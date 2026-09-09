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
                // Backticks quote identifiers (GenericDialect) — text
                // inside them is NOT code (keywords/apostrophes/parens
                // in a `col` gutted the predicate).
                if matches!(bytes[i], b'\'' | b'"' | b'`') {
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
/// Skip leading whitespace and SQL comments (line + nested block) so a
/// keyword check can see through `TABLE /* v2 */ IF NOT EXISTS`.
pub fn skip_leading_ws_and_comments(input: &str) -> &str {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    loop {
        // Unicode whitespace anywhere whitespace is skipped.
        while let Some((idx, ch)) = input[i..].char_indices().next() {
            if ch.is_whitespace() {
                i += idx + ch.len_utf8();
            } else {
                break;
            }
        }
        if bytes.get(i) == Some(&b'-') && bytes.get(i + 1) == Some(&b'-') {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes.get(i) == Some(&b'/') && bytes.get(i + 1) == Some(&b'*') {
            let comment_start = i;
            i += 2;
            let mut depth = 1usize;
            while i < bytes.len() && depth > 0 {
                if bytes.get(i) == Some(&b'/') && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if bytes.get(i) == Some(&b'*') && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth > 0 {
                return &input[comment_start..];
            }
            continue;
        }
        return &input[i..];
    }
}

/// Decode ONE identifier: strip a MATCHED outer delimiter pair (double
/// quote or backtick — content of the other style is untouched), collapse
/// doubled delimiter escapes, and perform no dot-splitting. Borrow when no
/// escape needs allocation.
pub fn decode_identifier(ident: &str) -> std::borrow::Cow<'_, str> {
    let bytes = ident.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' || first == b'`') && first == last {
            let inner = &ident[1..ident.len() - 1];
            let doubled = if first == b'"' { "\"\"" } else { "``" };
            if inner.contains(doubled) {
                let delimiter = if first == b'"' { "\"" } else { "`" };
                return std::borrow::Cow::Owned(inner.replace(doubled, delimiter));
            }
            return std::borrow::Cow::Borrowed(inner);
        }
    }
    std::borrow::Cow::Borrowed(ident)
}

/// Discover catalog-backed relations through the pinned SQL parser's AST.
/// This is deliberately shared by REST and the unified-query port so EXPLAIN
/// reports the same storage authority on both surfaces. Invalid SQL yields no
/// authority metadata; supported extension-function targets are visited in the
/// same parsed tree.
pub(crate) fn collect_sql_catalog_targets(sql: &str, targets: &mut Vec<String>) {
    use core::ops::ControlFlow;
    use sqlparser::ast::{
        Expr, FromTable, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName, Query,
        SetExpr, Statement, TableFactor, TableObject, Visit, Visitor,
    };
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;
    use sqlparser::tokenizer::Token;

    let dialect = GenericDialect;
    let statements = match Parser::parse_sql(&dialect, sql) {
        Ok(statements) => statements,
        // sqlparser 0.59 models PostgreSQL's `TABLE relation` as a query body,
        // but its statement dispatcher does not accept TABLE at the top level.
        // Retrying through the query parser preserves AST validation and also
        // covers TABLE operands inside set expressions without a raw scanner.
        Err(_) => {
            let Ok(mut parser) = Parser::new(&dialect).try_with_sql(sql) else {
                return;
            };
            let Ok(query) = parser.parse_query() else {
                return;
            };
            if !parser.consume_token(&Token::EOF) {
                return;
            }
            vec![Statement::Query(query)]
        }
    };

    struct RelationCollector<'a> {
        targets: &'a mut Vec<String>,
        cte_scopes: Vec<CteScope>,
        cte_children: std::collections::HashMap<usize, (usize, usize)>,
        query_restorations: Vec<Option<(usize, std::collections::HashSet<String>)>>,
    }

    struct CteScope {
        aliases: Vec<String>,
        visible: std::collections::HashSet<String>,
    }

    impl RelationCollector<'_> {
        fn identifier_key(ident: &Ident) -> String {
            if ident.quote_style.is_none() {
                ident.value.to_lowercase()
            } else {
                ident.value.clone()
            }
        }

        fn is_cte_reference(&self, relation: &ObjectName) -> bool {
            let [part] = relation.0.as_slice() else {
                return false;
            };
            let Some(ident) = part.as_ident() else {
                return false;
            };
            let key = Self::identifier_key(ident);
            self.cte_scopes
                .iter()
                .rev()
                .any(|scope| scope.visible.contains(&key))
        }

        fn push_relation(&mut self, relation: &ObjectName) {
            if self.is_cte_reference(relation) {
                return;
            }
            let components = relation
                .0
                .iter()
                .map(|part| part.as_ident().map(|ident| ident.value.as_str()))
                .collect::<Option<Vec<_>>>();
            if let Some(components) = components
                && !components.is_empty()
            {
                self.targets.push(components.join("."));
            }
        }

        fn push_relation_unconditionally(&mut self, relation: &ObjectName) {
            let components = relation
                .0
                .iter()
                .map(|part| part.as_ident().map(|ident| ident.value.as_str()))
                .collect::<Option<Vec<_>>>();
            if let Some(components) = components
                && !components.is_empty()
            {
                self.targets.push(components.join("."));
            }
        }

        fn push_table_factor_unconditionally(&mut self, factor: &TableFactor) {
            if let TableFactor::Table {
                name, args: None, ..
            } = factor
            {
                self.push_relation_unconditionally(name);
            }
        }

        fn is_catalog_function(name: &ObjectName) -> bool {
            let Some(name) = name.0.last().and_then(|part| part.as_ident()) else {
                return false;
            };
            matches!(
                name.value.to_ascii_uppercase().as_str(),
                "VECTOR_SEARCH" | "DOCUMENT_QUERY" | "LOGS" | "METRICS" | "TRACES" | "RERANK"
            )
        }

        fn function_arg_expr(arg: &FunctionArg) -> &FunctionArgExpr {
            match arg {
                FunctionArg::Named { arg, .. }
                | FunctionArg::ExprNamed { arg, .. }
                | FunctionArg::Unnamed(arg) => arg,
            }
        }

        fn function_target(args: &[FunctionArg]) -> Option<String> {
            let FunctionArgExpr::Expr(expr) = Self::function_arg_expr(args.first()?) else {
                return None;
            };
            let value = match expr {
                Expr::Value(value) => value.value.clone().into_string(),
                Expr::Identifier(ident) => Some(ident.value.clone()),
                Expr::CompoundIdentifier(parts) => Some(
                    parts
                        .iter()
                        .map(|ident| ident.value.as_str())
                        .collect::<Vec<_>>()
                        .join("."),
                ),
                _ => None,
            }?;
            (!value.is_empty() && !value.starts_with('$') && value != "?").then_some(value)
        }

        fn push_catalog_function(&mut self, name: &ObjectName, args: &[FunctionArg]) {
            if Self::is_catalog_function(name)
                && let Some(target) = Self::function_target(args)
            {
                self.targets.push(target);
            }
        }

        fn push_table_shorthands(&mut self, body: &SetExpr) {
            match body {
                SetExpr::Table(table) => {
                    let Some(table_name) = &table.table_name else {
                        return;
                    };
                    if table.schema_name.is_none()
                        && self
                            .cte_scopes
                            .iter()
                            .rev()
                            .any(|scope| scope.visible.contains(&table_name.to_lowercase()))
                    {
                        return;
                    }
                    let target = table
                        .schema_name
                        .as_ref()
                        .map(|schema| format!("{schema}.{table_name}"))
                        .unwrap_or_else(|| table_name.clone());
                    self.targets.push(target);
                }
                SetExpr::SetOperation { left, right, .. } => {
                    self.push_table_shorthands(left);
                    self.push_table_shorthands(right);
                }
                _ => {}
            }
        }
    }

    impl Visitor for RelationCollector<'_> {
        type Break = ();

        fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
            // A CTE definition sees prior aliases in declaration order, plus
            // itself only for WITH RECURSIVE. The containing query's main body
            // sees the complete alias list. Query-address registration lets
            // the generic visitor apply that scope while it enters each CTE
            // child without reimplementing AST traversal.
            let query_key = query as *const Query as usize;
            let restoration =
                self.cte_children
                    .remove(&query_key)
                    .and_then(|(scope_index, visible_count)| {
                        let scope = self.cte_scopes.get_mut(scope_index)?;
                        let full_visibility = std::mem::take(&mut scope.visible);
                        scope.visible = scope.aliases.iter().take(visible_count).cloned().collect();
                        Some((scope_index, full_visibility))
                    });
            self.query_restorations.push(restoration);

            let scope_index = self.cte_scopes.len();
            let aliases: Vec<String> = query
                .with
                .as_ref()
                .map(|with| {
                    with.cte_tables
                        .iter()
                        .map(|cte| Self::identifier_key(&cte.alias.name))
                        .collect()
                })
                .unwrap_or_default();
            if let Some(with) = &query.with {
                for (cte_index, cte) in with.cte_tables.iter().enumerate() {
                    let visible_count = cte_index + usize::from(with.recursive);
                    let child_key = cte.query.as_ref() as *const Query as usize;
                    self.cte_children
                        .insert(child_key, (scope_index, visible_count));
                }
            }
            self.cte_scopes.push(CteScope {
                visible: aliases.iter().cloned().collect(),
                aliases,
            });
            self.push_table_shorthands(&query.body);
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<Self::Break> {
            self.cte_scopes.pop();
            if let Some(Some((scope_index, full_visibility))) = self.query_restorations.pop()
                && let Some(scope) = self.cte_scopes.get_mut(scope_index)
            {
                scope.visible = full_visibility;
            }
            ControlFlow::Continue(())
        }

        fn pre_visit_table_factor(
            &mut self,
            table_factor: &TableFactor,
        ) -> ControlFlow<Self::Break> {
            if let TableFactor::Table { name, args, .. } = table_factor {
                if let Some(args) = args {
                    self.push_catalog_function(name, &args.args);
                } else {
                    self.push_relation(name);
                }
            } else if let TableFactor::Function { name, args, .. } = table_factor {
                self.push_catalog_function(name, args);
            }
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
            if let Expr::Function(function) = expr
                && let FunctionArguments::List(args) = &function.args
            {
                self.push_catalog_function(&function.name, &args.args);
            }
            ControlFlow::Continue(())
        }

        fn pre_visit_statement(&mut self, statement: &Statement) -> ControlFlow<Self::Break> {
            match statement {
                Statement::Insert(insert) => {
                    if let TableObject::TableName(name) = &insert.table {
                        self.push_relation_unconditionally(name);
                    }
                }
                Statement::Update { table, .. } => {
                    self.push_table_factor_unconditionally(&table.relation);
                }
                Statement::Delete(delete) => {
                    let from = match &delete.from {
                        FromTable::WithFromKeyword(from) | FromTable::WithoutKeyword(from) => from,
                    };
                    for table in from {
                        self.push_table_factor_unconditionally(&table.relation);
                    }
                }
                Statement::Merge { table, .. } => {
                    self.push_table_factor_unconditionally(table);
                }
                _ => {}
            }
            ControlFlow::Continue(())
        }
    }

    let mut collector = RelationCollector {
        targets,
        cte_scopes: Vec::new(),
        cte_children: std::collections::HashMap::new(),
        query_restorations: Vec::new(),
    };
    let _ = statements.visit(&mut collector);
}

/// Case-insensitive `IF NOT EXISTS` prefix strip with a word boundary
/// (whitespace OR an opening quote — `EXISTS"logs"` parses under the
/// pinned GenericDialect). Returns (had_prefix, rest).
pub fn strip_if_not_exists(input: &str) -> (bool, &str) {
    // Comments and arbitrary SQL whitespace may separate each keyword.
    // Require trivia between words so identifier prefixes such as IFNOT and
    // NOTEXISTS never match.
    fn take_word<'a>(value: &'a str, word: &str) -> Option<&'a str> {
        let head = value.get(..word.len())?;
        head.eq_ignore_ascii_case(word)
            .then(|| &value[word.len()..])
    }
    fn separated(value: &str) -> Option<&str> {
        let remainder = skip_leading_ws_and_comments(value);
        (remainder.len() < value.len()).then_some(remainder)
    }

    let cleaned = skip_leading_ws_and_comments(input);
    let Some(after_if) = take_word(cleaned, "IF").and_then(separated) else {
        return (false, input);
    };
    let Some(after_not) = take_word(after_if, "NOT").and_then(separated) else {
        return (false, input);
    };
    let Some(rest) = take_word(after_not, "EXISTS") else {
        return (false, input);
    };
    let operand = skip_leading_ws_and_comments(rest);
    if operand.len() < rest.len() {
        return (true, operand);
    }
    let next = rest.chars().next();
    match next {
        None => (true, ""),
        // No operand decoding here: identifier decoding belongs to the
        // CONSUMERS — extract_identifier handles quotes AND preserves the
        // AS-clause tail the rank-profile parser needs; clean_identifier
        // strips both delimiters at the pgwire create path. Decoding in
        // the strip truncated doubled-quote escapes and dropped the tail
        // (rounds 45-47 churn).
        // Char-based whitespace (NBSP included) — a bare >= 0x80 byte test
        // treated é as a separator while identifier scanning treats it as text.
        Some(ch) if ch.is_whitespace() => (true, operand),
        // Quote-ADJACENT operand (IF NOT EXISTS"logs" parses under the
        // pinned dialect): strip, rest undecoded (consumers decode).
        // Paren-adjacent too (IF NOT EXISTS(x INT) — the paren is the
        // column-list opener, never identifier text; the fall-through
        // minted a collection named 'if').
        Some('"' | '`' | '(') => (true, rest),
        Some(_) => (false, input),
    }
}

/// [`find_ascii_ci_outside_quotes`], additionally excluding text nested in
/// parentheses, brackets, or braces. This is the appropriate scanner for
/// top-level SQL clauses: for example, `WHERE` inside `FILTER (WHERE ...)`
/// is not the statement's predicate clause.
fn at_top_level_impl(
    haystack: &str,
    needle: &str,
    skip_quote_at: Option<usize>,
    complete_scan: bool,
) -> (Option<usize>, Option<usize>, bool) {
    let bytes = haystack.as_bytes();
    let mut quote: Option<(u8, usize)> = None;
    let mut nesting = Vec::new();
    let mut found = None;
    let mut i = 0usize;
    while i < bytes.len() {
        match quote {
            Some((q, _opened_at)) => {
                if bytes[i] == q {
                    if i + 1 < bytes.len() && bytes[i + 1] == q {
                        i += 2;
                        continue;
                    }
                    quote = None;
                }
                i += 1;
            }
            None => {
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    i += 2;
                    let mut comment_depth = 1usize;
                    while i < bytes.len() && comment_depth > 0 {
                        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                            comment_depth += 1;
                            i += 2;
                        } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                            comment_depth -= 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    if comment_depth > 0 {
                        return (None, None, true);
                    }
                    continue;
                }
                if bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'-' {
                    i += 2;
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                    continue;
                }
                // Backticks quote identifiers (GenericDialect) — text
                // inside them is NOT code (keywords/apostrophes/parens
                // in a `col` gutted the predicate).
                if matches!(bytes[i], b'\'' | b'"' | b'`') && skip_quote_at != Some(i) {
                    quote = Some((bytes[i], i));
                    i += 1;
                    continue;
                }
                match bytes[i] {
                    b'(' | b'[' | b'{' => {
                        nesting.push(bytes[i]);
                        i += 1;
                        continue;
                    }
                    close @ (b')' | b']' | b'}') => {
                        let expected_open = match close {
                            b')' => b'(',
                            b']' => b'[',
                            _ => b'{',
                        };
                        if nesting.pop() != Some(expected_open) {
                            return (None, None, true);
                        }
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
                if nesting.is_empty()
                    && found.is_none()
                    && i + needle.len() <= bytes.len()
                    && bytes[i..i + needle.len()].eq_ignore_ascii_case(needle.as_bytes())
                {
                    if !complete_scan {
                        return (Some(i), None, false);
                    }
                    found = Some(i);
                }
                i += 1;
            }
        }
    }
    // EOF in quote mode: report the opener so the caller can retry once
    // with it treated as a literal byte (a stray apostrophe must not
    // swallow the rest of the query).
    (found, quote.map(|(_, at)| at), !nesting.is_empty())
}

pub fn find_ascii_ci_at_top_level(haystack: &str, needle: &str) -> Option<usize> {
    // Unterminated-quote tolerance, EXACTLY-ONCE retry: a stray apostrophe
    // or dollar-quote body must not swallow the rest of the query (LIMIT
    // inside $$ don't panic $$ streamed the whole table). The retry is
    // driven by the tracked EOF-in-quote opener — retrying EVERY quote
    // byte matched keywords inside properly-terminated literals (wrong
    // results) and made needle-absent scans quadratic (15k quotes → 15k
    // rescans).
    let (found, unterminated, _) = at_top_level_impl(haystack, needle, None, false);
    if found.is_some() {
        return found;
    }
    match unterminated {
        Some(open) => at_top_level_impl(haystack, needle, Some(open), false).0,
        None => None,
    }
}

/// A full top-level keyword scan with malformed-structure reporting.
/// Fail-closed fallback paths must distinguish a genuinely absent clause from
/// one hidden by an unterminated quote, comment, or nesting delimiter, and must
/// validate syntax after an early keyword too. Ordinary tolerant lookup above
/// retains its at-most-once retry; checked lookup is strict and scans once.
pub fn find_ascii_ci_at_top_level_checked(
    haystack: &str,
    needle: &str,
) -> Result<Option<usize>, &'static str> {
    let (found, unterminated, malformed) = at_top_level_impl(haystack, needle, None, true);
    if malformed || unterminated.is_some() {
        Err("unterminated SQL quote, comment, or mismatched nesting delimiter")
    } else {
        Ok(found)
    }
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
    use super::{skip_leading_ws_and_comments, strip_if_not_exists};

    #[test]
    fn strip_if_not_exists_is_comment_and_ws_tolerant() {
        // Comment BEFORE the prefix and AFTER it (rounds 43-44); operand
        // decoding belongs to the CONSUMERS (extract_identifier /
        // clean_identifier) — the strip hands back the raw rest.
        assert_eq!(
            strip_if_not_exists("/* v2 */ IF NOT EXISTS docs"),
            (true, "docs")
        );
        assert_eq!(
            strip_if_not_exists("IF NOT EXISTS /* v2 */ docs"),
            (true, "docs")
        );
        // Word boundary: no separator, no strip.
        assert_eq!(
            strip_if_not_exists("IF NOT EXISTSx"),
            (false, "IF NOT EXISTSx")
        );
        assert_eq!(
            strip_if_not_exists("IF NOT EXISTSécole"),
            (false, "IF NOT EXISTSécole")
        );
        assert_eq!(strip_if_not_exists("IF\nNOT\tEXISTS docs"), (true, "docs"));
        assert_eq!(
            strip_if_not_exists("IF/* one */NOT /* two */ EXISTS docs"),
            (true, "docs")
        );
        assert_eq!(
            strip_if_not_exists("IF NOT EXISTS-- note\nlogs"),
            (true, "logs")
        );
        // Quoted operands pass through raw (consumers decode).
        assert_eq!(
            strip_if_not_exists("IF NOT EXISTS `logs` (id INT)"),
            (true, "`logs` (id INT)")
        );
    }

    #[test]
    fn skip_leading_ws_and_comments_handles_nested_blocks() {
        assert_eq!(
            skip_leading_ws_and_comments("  -- note\n  /* a /* b */ */ t"),
            "t"
        );
        assert_eq!(skip_leading_ws_and_comments("plain"), "plain");

        // An unterminated comment is invalid input, not ignorable trivia.
        // Preserve it so callers fail closed instead of mistaking the
        // remainder for an absent identifier.
        assert_eq!(
            skip_leading_ws_and_comments("  /* unterminated"),
            "/* unterminated"
        );
        assert_eq!(
            strip_if_not_exists("IF NOT EXISTS /* unterminated"),
            (true, "/* unterminated")
        );
    }

    use super::{collect_sql_catalog_targets, finite_f32, inject_graph_target_into_cypher};

    #[test]
    fn sql_target_scanner_uses_ast_relation_context() {
        let mut targets = Vec::new();
        for sql in [
            r#"SELECT*FROM "schema"."orders", extra"#,
            r#"INSERT INTO"events(2026)"(id) VALUES (1) ON CONFLICT DO UPDATE SET id = 2"#,
            r#"SELECT EXTRACT(YEAR FROM created_at) FROM "left"JOIN "right" ON true"#,
            r#"SELECT * FROM "tenant" . "spaced""#,
            r#"SELECT * FROM "commented"/*c*/."table""#,
            "SELECT * FROM orders FOR UPDATE",
            "SELECT * FROM TRACES('ops')",
            r#"SELECT * FROM "TRACES""#,
            "WITH scoped AS (SELECT * FROM base) SELECT * FROM scoped",
            "WITH first AS (SELECT * FROM later), later AS (SELECT * FROM base) SELECT * FROM first",
            "WITH x AS (SELECT * FROM x) SELECT * FROM x",
            "WITH RECURSIVE r AS (SELECT * FROM seed UNION ALL SELECT * FROM r) SELECT * FROM r",
            r#"WITH "foo" AS (SELECT * FROM quoted_base) SELECT * FROM foo"#,
            r#"WITH bar AS (SELECT * FROM plain_base) SELECT * FROM "bar""#,
            r#"WITH "Caps" AS (SELECT * FROM caps_base) SELECT * FROM caps"#,
            "WITH write_target AS (SELECT * FROM write_seed) INSERT INTO write_target VALUES (1)",
            "TABLE shorthand",
            "TABLE public.qualified_shorthand",
            "SELECT $$ harmless TRACES('secret') $$ AS note FROM actual",
        ] {
            collect_sql_catalog_targets(sql, &mut targets);
        }
        targets.sort();
        targets.dedup();
        assert_eq!(
            targets,
            [
                "TRACES",
                "actual",
                "base",
                "caps",
                "caps_base",
                "commented.table",
                "events(2026)",
                "extra",
                "later",
                "left",
                "ops",
                "orders",
                "plain_base",
                "public.qualified_shorthand",
                "quoted_base",
                "right",
                "schema.orders",
                "seed",
                "shorthand",
                "tenant.spaced",
                "write_seed",
                "write_target",
                "x",
            ]
        );
        assert!(!targets.contains(&"created_at".to_string()));
        assert!(!targets.contains(&"id".to_string()));
        assert!(!targets.contains(&"scoped".to_string()));
        assert!(!targets.contains(&"first".to_string()));
        assert!(!targets.contains(&"r".to_string()));
        assert!(!targets.contains(&"foo".to_string()));
        assert!(!targets.contains(&"bar".to_string()));
        assert!(!targets.contains(&"secret".to_string()));
    }

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
        collect_sql_catalog_targets(
            r#"SELECT * FROM NOT_VECTOR_SEARCH('wrong', [1.0]);
               SELECT * FROM πVECTOR_SEARCH('unicode_wrong', [1.0]);
               /* outer /* VECTOR_SEARCH('block_wrong') */ comment */
               SELECT VECTOR_SEARCH + other_call('also_wrong') FROM actual;
               SELECT * FROM VECTOR_SEARCH ('right', [1.0]);
               SELECT * FROM TRACES /* outer /* nested */ comment */ ( /* arg */ 'commented');
               SELECT * FROM TRACES -- call trivia
                 ( -- arg trivia
                 'line-commented')"#,
            &mut targets,
        );
        targets.sort();
        targets.dedup();
        assert_eq!(targets, ["actual", "commented", "line-commented", "right"]);
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
