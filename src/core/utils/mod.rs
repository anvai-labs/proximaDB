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

const MAX_SQL_AUTHORITY_BYTES: usize = 1_048_576;

/// Discover catalog-backed relations through the pinned SQL parser's AST.
/// This is deliberately shared by REST and the unified-query port so EXPLAIN
/// reports the same storage authority on both surfaces. Invalid SQL yields no
/// authority metadata; supported extension-function targets are visited in the
/// same parsed tree.
pub(crate) fn collect_sql_catalog_targets(sql: &str, targets: &mut Vec<String>) {
    use core::ops::ControlFlow;
    use sqlparser::ast::{
        Expr, FromTable, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident,
        ObjectName, PipeOperator, Query, SetExpr, Statement, TableFactor, TableObject, Visit,
        Visitor,
    };
    use sqlparser::dialect::GenericDialect;
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::Parser;
    use sqlparser::tokenizer::{Location, Token, TokenWithSpan, Tokenizer};

    if sql.len() > MAX_SQL_AUTHORITY_BYTES {
        return;
    }

    let dialect = GenericDialect;
    let Ok(tokenized) = Tokenizer::new(&dialect, sql)
        .with_unescape(false)
        .tokenize_with_location()
    else {
        return;
    };

    // The derived AST visitor is recursive. Bound adversarially deep/large
    // inputs before parsing or visiting them so catalog introspection cannot
    // exhaust a server thread stack. The 256-way TABLE regression remains
    // comfortably inside these limits.
    const MAX_SQL_AUTHORITY_TOKENS: usize = 16_384;
    const MAX_SQL_AUTHORITY_NESTING: usize = 128;
    const MAX_SQL_AUTHORITY_SET_OPERATIONS: usize = 512;
    let mut significant_count = 0usize;
    let mut nesting = 0usize;
    let mut set_operations = 0usize;
    for entry in &tokenized {
        if matches!(entry.token, Token::Whitespace(_)) {
            continue;
        }
        significant_count += 1;
        if significant_count > MAX_SQL_AUTHORITY_TOKENS {
            return;
        }
        match entry.token {
            Token::LParen => {
                nesting += 1;
                if nesting > MAX_SQL_AUTHORITY_NESTING {
                    return;
                }
            }
            Token::RParen => nesting = nesting.saturating_sub(1),
            Token::Word(ref word)
                if matches!(
                    word.keyword,
                    Keyword::UNION | Keyword::INTERSECT | Keyword::EXCEPT | Keyword::MINUS
                ) =>
            {
                set_operations += 1;
                if set_operations > MAX_SQL_AUTHORITY_SET_OPERATIONS {
                    return;
                }
            }
            _ => {}
        }
    }

    // sqlparser 0.59's SetExpr::Table discards identifier quote style, and its
    // direct query parser over-consumes two tokens after an unqualified name.
    // Normalize only TABLE query-body positions, then run the normal full SQL
    // parser. Only the TABLE keyword's source span is replaced; every other
    // source byte remains untouched, including eager-decoded E/U& literals.
    // Full reparsing remains the syntax authority.
    fn normalize_table_query_bodies(
        sql: &str,
        tokenized: &[TokenWithSpan],
    ) -> Option<(String, usize)> {
        let tokens = tokenized
            .iter()
            .map(|entry| entry.token.clone())
            .collect::<Vec<_>>();
        let significant = tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| !matches!(token, Token::Whitespace(_)))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let mut rewrite = Vec::new();
        let mut statement_has_insert = false;
        let mut statement_has_into = false;

        for (position, token_index) in significant.iter().copied().enumerate() {
            match &tokens[token_index] {
                Token::SemiColon => {
                    statement_has_insert = false;
                    statement_has_into = false;
                }
                Token::Word(word) if word.keyword == Keyword::INSERT => {
                    statement_has_insert = true;
                }
                Token::Word(word) if word.keyword == Keyword::INTO && statement_has_insert => {
                    statement_has_into = true;
                }
                _ => {}
            }
            let Token::Word(word) = &tokens[token_index] else {
                continue;
            };
            if word.keyword != Keyword::TABLE {
                continue;
            }
            let previous = position
                .checked_sub(1)
                .and_then(|index| significant.get(index))
                .map(|index| &tokens[*index]);
            let previous_keyword = previous.and_then(|token| match token {
                Token::Word(word) => Some(word.keyword),
                _ => None,
            });
            let after_quantifier =
                matches!(previous_keyword, Some(Keyword::ALL | Keyword::DISTINCT))
                    && position
                        .checked_sub(2)
                        .and_then(|index| significant.get(index))
                        .is_some_and(|index| {
                            matches!(
                                &tokens[*index],
                                Token::Word(word)
                                    if matches!(
                                        word.keyword,
                                        Keyword::UNION
                                            | Keyword::INTERSECT
                                            | Keyword::EXCEPT
                                            | Keyword::MINUS
                                    )
                            )
                        });
            let after_by_name = previous_keyword == Some(Keyword::NAME)
                && position
                    .checked_sub(2)
                    .and_then(|index| significant.get(index))
                    .is_some_and(|index| {
                        matches!(
                            &tokens[*index],
                            Token::Word(word) if word.keyword == Keyword::BY
                        )
                    })
                && position
                    .checked_sub(3)
                    .and_then(|index| significant.get(index))
                    .is_some_and(|index| {
                        matches!(
                            &tokens[*index],
                            Token::Word(word)
                                if matches!(
                                    word.keyword,
                                    Keyword::UNION
                                        | Keyword::INTERSECT
                                        | Keyword::EXCEPT
                                        | Keyword::MINUS
                                        | Keyword::ALL
                                        | Keyword::DISTINCT
                                )
                        )
                    })
                && (position
                    .checked_sub(3)
                    .and_then(|index| significant.get(index))
                    .is_some_and(|index| {
                        matches!(
                            &tokens[*index],
                            Token::Word(word)
                                if matches!(
                                    word.keyword,
                                    Keyword::UNION
                                        | Keyword::INTERSECT
                                        | Keyword::EXCEPT
                                        | Keyword::MINUS
                                )
                        )
                    })
                    || position
                        .checked_sub(4)
                        .and_then(|index| significant.get(index))
                        .is_some_and(|index| {
                            matches!(
                                &tokens[*index],
                                Token::Word(word)
                                    if matches!(
                                        word.keyword,
                                        Keyword::UNION
                                            | Keyword::INTERSECT
                                            | Keyword::EXCEPT
                                            | Keyword::MINUS
                                    )
                            )
                        }));
            let insert_query_body = statement_has_insert && statement_has_into;
            let query_body_position = previous.is_none()
                || matches!(
                    previous,
                    Some(Token::SemiColon | Token::LParen | Token::RParen)
                )
                || matches!(
                    previous_keyword,
                    Some(
                        Keyword::AS
                            | Keyword::UNION
                            | Keyword::INTERSECT
                            | Keyword::EXCEPT
                            | Keyword::MINUS
                    )
                )
                || after_quantifier
                || after_by_name
                || insert_query_body;
            if !query_body_position {
                continue;
            }

            // Match sqlparser's TABLE grammar exactly: one Word, optionally
            // followed by `. Word`, and then a query-boundary token. This is
            // what prevents `TABLE orders garbage` from becoming a SELECT
            // with an accidentally accepted alias.
            let first_name_position = position + 1;
            let Some(first_name_index) = significant.get(first_name_position) else {
                continue;
            };
            if !matches!(tokens[*first_name_index], Token::Word(_)) {
                continue;
            }
            let mut after_name_position = first_name_position + 1;
            if significant
                .get(after_name_position)
                .is_some_and(|index| matches!(tokens[*index], Token::Period))
            {
                let Some(qualified_name_index) = significant.get(after_name_position + 1) else {
                    continue;
                };
                if !matches!(tokens[*qualified_name_index], Token::Word(_)) {
                    continue;
                }
                after_name_position += 2;
            }
            let boundary = significant
                .get(after_name_position)
                .map(|index| &tokens[*index]);
            let valid_boundary = boundary.is_none()
                || matches!(
                    boundary,
                    Some(Token::SemiColon | Token::RParen | Token::VerticalBarRightAngleBracket)
                )
                || matches!(
                        boundary,
                        Some(Token::Word(word))
                            if matches!(
                                word.keyword,
                                Keyword::UNION
                                    | Keyword::INTERSECT
                                    | Keyword::EXCEPT
                                    | Keyword::MINUS
                                    | Keyword::ORDER
                                    | Keyword::LIMIT
                                    | Keyword::OFFSET
                                    | Keyword::FETCH
                                    | Keyword::FOR
                                    | Keyword::FORMAT
                                    | Keyword::ON
                                    | Keyword::RETURNING
                                    | Keyword::SETTINGS
                            )
                );
            if valid_boundary {
                rewrite.push(token_index);
            }
        }

        if rewrite.is_empty() {
            return None;
        }
        let spans = rewrite
            .iter()
            .map(|token_index| tokenized.get(*token_index).map(|entry| entry.span))
            .collect::<Option<Vec<_>>>()?;
        let mut requested_locations = spans
            .iter()
            .flat_map(|span| [span.start, span.end])
            .collect::<Vec<_>>();
        requested_locations.sort_unstable();
        requested_locations.dedup();
        let mut requested_offsets = vec![None; requested_locations.len()];
        let mut requested_index = 0usize;
        let mut record_offset = |location: Location, byte_offset: usize| {
            if requested_locations.get(requested_index) == Some(&location) {
                requested_offsets[requested_index] = Some(byte_offset);
                requested_index += 1;
            }
        };
        let mut location = Location::new(1, 1);
        record_offset(location, 0);
        for (byte_index, ch) in sql.char_indices() {
            if ch == '\n' {
                location.line += 1;
                location.column = 1;
            } else {
                location.column += 1;
            }
            record_offset(location, byte_index + ch.len_utf8());
        }
        drop(record_offset);
        let byte_offset = |location: Location| {
            let index = requested_locations.binary_search(&location).ok()?;
            requested_offsets.get(index).copied().flatten()
        };
        let mut rewritten = String::with_capacity(sql.len());
        let mut cursor = 0usize;
        let rewrite_count = rewrite.len();
        for span in spans {
            let start = byte_offset(span.start)?;
            let end = byte_offset(span.end)?;
            if start < cursor || end < start {
                return None;
            }
            rewritten.push_str(sql.get(cursor..start)?);
            rewritten.push_str("SELECT * FROM");
            cursor = end;
        }
        rewritten.push_str(sql.get(cursor..)?);
        Some((rewritten, rewrite_count))
    }

    fn direct_table_count(body: &SetExpr) -> usize {
        let mut count = 0usize;
        let mut pending = vec![body];
        while let Some(body) = pending.pop() {
            match body {
                SetExpr::Table(_) => count += 1,
                SetExpr::SetOperation { left, right, .. } => {
                    pending.push(right);
                    pending.push(left);
                }
                // Nested Query nodes receive their own visitor callback.
                SetExpr::Query(_) => {}
                _ => {}
            }
        }
        count
    }

    struct TableCounter(usize);
    impl Visitor for TableCounter {
        type Break = ();

        fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
            self.0 += direct_table_count(&query.body);
            ControlFlow::Continue(())
        }
    }

    let original = Parser::parse_sql(&dialect, sql);
    let normalized = normalize_table_query_bodies(sql, &tokenized);
    let statements = match original {
        Ok(statements) => {
            let mut counter = TableCounter(0);
            let _ = statements.visit(&mut counter);
            if counter.0 == 0 {
                statements
            } else {
                let Some((rewritten, rewrite_count)) = normalized else {
                    return;
                };
                if rewrite_count != counter.0 {
                    return;
                }
                let Ok(statements) = Parser::parse_sql(&dialect, &rewritten) else {
                    return;
                };
                statements
            }
        }
        Err(_) => {
            let Some((rewritten, _)) = normalized else {
                return;
            };
            let Ok(statements) = Parser::parse_sql(&dialect, &rewritten) else {
                return;
            };
            statements
        }
    };

    struct RelationCollector<'a> {
        targets: &'a mut Vec<String>,
        cte_scopes: Vec<CteScope>,
        cte_children: std::collections::HashMap<usize, (usize, usize)>,
        query_restorations: Vec<Option<(usize, usize)>>,
    }

    struct CteScope {
        declaration_index: std::collections::HashMap<String, usize>,
        visible_count: usize,
    }

    impl RelationCollector<'_> {
        fn identifier_key(ident: &Ident) -> String {
            if ident.quote_style.is_none() {
                ident.value.to_ascii_lowercase()
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
            self.cte_scopes.iter().rev().any(|scope| {
                scope
                    .declaration_index
                    .get(&key)
                    .is_some_and(|index| *index < scope.visible_count)
            })
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
            name.quote_style.is_none()
                && matches!(
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

        fn push_function(&mut self, function: &Function) {
            if let FunctionArguments::List(args) = &function.args {
                self.push_catalog_function(&function.name, &args.args);
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
                        let full_visibility =
                            std::mem::replace(&mut scope.visible_count, visible_count);
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
                visible_count: aliases.len(),
                declaration_index: aliases
                    .into_iter()
                    .enumerate()
                    .map(|(index, alias)| (alias, index))
                    .collect(),
            });
            for operator in &query.pipe_operators {
                if let PipeOperator::Call { function, .. } = operator {
                    self.push_function(function);
                }
            }
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<Self::Break> {
            self.cte_scopes.pop();
            if let Some(Some((scope_index, full_visibility))) = self.query_restorations.pop()
                && let Some(scope) = self.cte_scopes.get_mut(scope_index)
            {
                scope.visible_count = full_visibility;
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
                Statement::Insert(insert) => match &insert.table {
                    TableObject::TableName(name) => self.push_relation_unconditionally(name),
                    TableObject::TableFunction(function) => self.push_function(function),
                },
                Statement::Update { table, .. } => {
                    self.push_table_factor_unconditionally(&table.relation);
                }
                Statement::Delete(delete) => {
                    if delete.tables.is_empty() {
                        let from = match &delete.from {
                            FromTable::WithFromKeyword(from) | FromTable::WithoutKeyword(from) => {
                                from
                            }
                        };
                        for table in from {
                            self.push_table_factor_unconditionally(&table.relation);
                        }
                    } else {
                        for table in &delete.tables {
                            self.push_relation_unconditionally(table);
                        }
                    }
                }
                Statement::Merge { table, .. } => {
                    self.push_table_factor_unconditionally(table);
                }
                Statement::Call(function) => {
                    self.push_function(function);
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

    use super::{
        MAX_SQL_AUTHORITY_BYTES, collect_sql_catalog_targets, finite_f32,
        inject_graph_target_into_cypher,
    };

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
            r#"WITH É AS (SELECT * FROM unicode_seed) SELECT * FROM "é""#,
            "WITH write_target AS (SELECT * FROM write_seed) INSERT INTO write_target VALUES (1)",
            "INSERT INTO insertion_sink TABLE insertion_source",
            "INSERT INTO returning_sink TABLE returning_source RETURNING id",
            "INSERT INTO conflict_sink TABLE conflict_source ON CONFLICT DO NOTHING",
            "TABLE shorthand",
            "TABLE public.qualified_shorthand",
            "TABLE terminated;",
            r#"TABLE "CaseTable""#,
            r#"WITH caps_table AS (SELECT * FROM caps_table_seed) TABLE "CapsTable""#,
            r#"WITH "QuotedTable" AS (SELECT * FROM quoted_table_seed) TABLE "QuotedTable""#,
            r#"SELECT * FROM union_base UNION ALL TABLE "UnionTable""#,
            "SELECT 'O''Brien' FROM literal_base UNION ALL TABLE literal_other",
            r#"SELECT E'O\'Brien' FROM e_base UNION ALL TABLE e_other"#,
            r#"SELECT U&'d\0061t' FROM u_base UNION ALL TABLE u_other"#,
            r#"SELECT * FROM "a""b" UNION ALL TABLE identifier_other"#,
            "TABLE pipe_source |> CALL TRACES('pipe_table')",
            "SELECT * FROM by_base UNION BY NAME TABLE by_source",
            "SELECT * FROM by_all_base UNION ALL BY NAME TABLE by_all_source",
            "SELECT * FROM by_distinct_base UNION DISTINCT BY NAME TABLE by_distinct_source",
            "SELECT $$ harmless TRACES('secret') $$ AS note FROM actual",
            "CALL TRACES('called')",
            r#"CALL "TRACES"('quoted_call')"#,
            r#"SELECT * FROM "TRACES"('quoted_function')"#,
        ] {
            collect_sql_catalog_targets(sql, &mut targets);
        }
        targets.sort();
        targets.dedup();
        assert_eq!(
            targets,
            [
                "CapsTable",
                "CaseTable",
                "TRACES",
                "UnionTable",
                "a\"b",
                "actual",
                "base",
                "by_all_base",
                "by_all_source",
                "by_base",
                "by_distinct_base",
                "by_distinct_source",
                "by_source",
                "called",
                "caps",
                "caps_base",
                "caps_table_seed",
                "commented.table",
                "conflict_sink",
                "conflict_source",
                "e_base",
                "e_other",
                "events(2026)",
                "extra",
                "identifier_other",
                "insertion_sink",
                "insertion_source",
                "later",
                "left",
                "literal_base",
                "literal_other",
                "ops",
                "orders",
                "pipe_source",
                "pipe_table",
                "plain_base",
                "public.qualified_shorthand",
                "quoted_base",
                "quoted_table_seed",
                "returning_sink",
                "returning_source",
                "right",
                "schema.orders",
                "seed",
                "shorthand",
                "tenant.spaced",
                "terminated",
                "u_base",
                "u_other",
                "unicode_seed",
                "union_base",
                "write_seed",
                "write_target",
                "x",
                "é",
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
        assert!(!targets.contains(&"quoted_call".to_string()));
        assert!(!targets.contains(&"quoted_function".to_string()));

        for malformed in ["TABLE guarded garbage", "TABLE guarded; SELECT"] {
            let mut trailing = Vec::new();
            collect_sql_catalog_targets(malformed, &mut trailing);
            assert!(trailing.is_empty(), "{malformed}: {trailing:?}");
        }
    }

    #[test]
    fn delete_target_authority_distinguishes_multi_table_sources() {
        let mut targets = Vec::new();
        // GenericDialect rejects MySQL's multi-target form before traversal;
        // keep the defensive Delete.tables branch for any future dialect
        // broadening, without claiming unsupported syntax is accepted here.
        collect_sql_catalog_targets("DELETE t FROM c JOIN t ON true", &mut targets);
        assert!(targets.is_empty());

        collect_sql_catalog_targets("WITH t AS (SELECT * FROM seed) DELETE FROM t", &mut targets);
        targets.sort();
        targets.dedup();
        assert_eq!(targets, ["seed", "t"]);
    }

    #[test]
    fn cte_authority_scope_handles_large_flat_with_list() {
        const CTE_COUNT: usize = 512;
        let definitions = (0..CTE_COUNT)
            .map(|index| format!("c{index} AS (SELECT * FROM base{index})"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("WITH {definitions} SELECT * FROM c511");
        let mut targets = Vec::new();
        collect_sql_catalog_targets(&sql, &mut targets);
        targets.sort();
        targets.dedup();

        assert_eq!(targets.len(), CTE_COUNT);
        assert!(targets.contains(&"base0".to_string()));
        assert!(targets.contains(&"base511".to_string()));
        assert!(!targets.contains(&"c511".to_string()));
    }

    #[test]
    fn table_authority_normalization_handles_large_set_chain() {
        const TABLE_COUNT: usize = 256;
        let sql = (0..TABLE_COUNT)
            .map(|index| format!("TABLE table{index}"))
            .collect::<Vec<_>>()
            .join(" UNION ALL ");
        let mut targets = Vec::new();
        collect_sql_catalog_targets(&sql, &mut targets);
        targets.sort();
        targets.dedup();

        assert_eq!(targets.len(), TABLE_COUNT);
        assert!(targets.contains(&"table0".to_string()));
        assert!(targets.contains(&"table255".to_string()));
    }

    #[test]
    fn sql_authority_scanner_bounds_adversarial_ast_shapes() {
        let oversized_set_chain = (0..600)
            .map(|index| format!("TABLE table{index}"))
            .collect::<Vec<_>>()
            .join(" UNION ALL ");
        let mut targets = Vec::new();
        collect_sql_catalog_targets(&oversized_set_chain, &mut targets);
        assert!(targets.is_empty());

        let deeply_nested = format!("{}SELECT * FROM hidden{}", "(".repeat(129), ")".repeat(129));
        collect_sql_catalog_targets(&deeply_nested, &mut targets);
        assert!(targets.is_empty());

        let oversized_literal = format!(
            "TABLE visible UNION ALL SELECT '{}' FROM hidden",
            "x".repeat(MAX_SQL_AUTHORITY_BYTES)
        );
        collect_sql_catalog_targets(&oversized_literal, &mut targets);
        assert!(targets.is_empty());
    }

    #[test]
    fn catalog_functions_are_collected_from_bare_function_ast_slots() {
        let mut targets = Vec::new();
        for sql in [
            "CALL TRACES('called')",
            "SELECT * FROM source |> CALL TRACES('piped')",
        ] {
            collect_sql_catalog_targets(sql, &mut targets);
        }
        targets.sort();
        targets.dedup();
        assert_eq!(targets, ["called", "piped", "source"]);

        // GenericDialect rejects ClickHouse's TableObject::TableFunction
        // syntax. The visitor still covers that AST slot defensively if the
        // collector's configured dialect expands later.
        targets.clear();
        collect_sql_catalog_targets(
            "INSERT INTO TABLE FUNCTION TRACES('inserted') VALUES (1)",
            &mut targets,
        );
        assert!(targets.is_empty());
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
