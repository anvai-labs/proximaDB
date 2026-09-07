//! # Federated Parser
//!
//! Extends SQL with multi-model query capabilities.
//!
//! ## Supported Extensions
//!
//! - **VECTOR_SEARCH(collection, query_vector, top_k)**: Similarity search
//! - **GRAPH_QUERY('cypher_query')**: Graph traversal via Cypher
//! - **DOCUMENT_QUERY(collection, filter)**: Document queries
//! - **RERANK(collection, query_text, query_vector, k, rank_profile)**:
//!   multi-phase rank pipeline (R-7c.4c parser slice). Returns
//!   `(id, score, phase, match_features, summary_features)` rows.
//!   Planner + executor wiring deferred to R-7c.4c.1 — this slice
//!   only establishes the extension surface.
//! - **LOGS(namespace)**: Observability log queries
//! - **METRICS(namespace)**: Observability metric queries
//! - **TRACES(namespace)**: Observability trace queries
//! - **<->** operator: Vector distance (pgvector compatible)

use anyhow::Result;
use std::collections::HashMap;

/// Type of query being executed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    /// Pure SQL query (RDBMS)
    Sql,
    /// Vector similarity search
    VectorSearch,
    /// Graph traversal query
    GraphQuery,
    /// Document query
    DocumentQuery,
    /// Multi-phase rank pipeline (R-7c.4c). RERANK(...) SRF — drives
    /// first/second/global rank phases against the per-process
    /// `RankServices`. Returns one row per scored hit.
    RerankSearch,
    /// Observability log query
    LogQuery,
    /// Observability metric query
    MetricQuery,
    /// Observability trace query
    TraceQuery,
    /// Cross-model federated query
    Federated,
}

/// Parsed vector argument for VECTOR_SEARCH and distance operators
#[derive(Debug, Clone, PartialEq)]
pub enum VectorQuery {
    /// Literal vector value from SQL, e.g. `[0.1, 0.2]`
    Literal(Vec<f32>),
    /// Raw SQL expression or column reference, e.g. `u.embedding`
    Expression(String),
}

/// SQL extension type detected in query
#[derive(Debug, Clone, PartialEq)]
pub enum SqlExtension {
    /// VECTOR_SEARCH(collection, vector, top_k)
    VectorSearch {
        collection: String,
        query_vector: VectorQuery,
        top_k: usize,
    },
    /// GRAPH_QUERY('cypher')
    GraphQuery { cypher: String },
    /// DOCUMENT_QUERY(collection, filter)
    DocumentQuery {
        collection: String,
        filter: Option<String>,
    },
    /// RERANK(collection, query_text, query_vector, k, rank_profile)
    /// — multi-phase rank pipeline. R-7c.4c. Positional args:
    /// 1. collection (required): collection name
    /// 2. query_text (required): text for BM25 / cross-encoder reranking
    /// 3. query_vector (required): retrieval vector — `[…]` literal
    ///    or column reference
    /// 4. k (optional, default 10): result count after global phase
    /// 5. rank_profile (optional): named profile in the ProfileRegistry;
    ///    absent means retrieval-only (mirrors the REST handler's
    ///    `rank_profile: None` path)
    RerankSearch {
        collection: String,
        query_text: String,
        query_vector: VectorQuery,
        k: usize,
        rank_profile: Option<String>,
    },
    /// LOGS(namespace)
    Logs { namespace: String },
    /// METRICS(namespace)
    Metrics { namespace: String },
    /// TRACES(namespace)
    Traces { namespace: String },
    /// Vector distance operator <->
    VectorDistance {
        left_column: String,
        right_literal: String,
    },
}

/// Parsed federated query
#[derive(Debug, Clone)]
pub struct FederatedQuery {
    /// Original SQL text
    pub sql: String,
    /// Primary query type
    pub query_type: QueryType,
    /// Detected SQL extensions
    pub extensions: Vec<SqlExtension>,
    /// Source-order byte positions for each detected extension
    pub extension_positions: Vec<usize>,
    /// Source aliases for each detected extension when present
    pub extension_aliases: Vec<Option<String>>,
    /// Target tables/collections
    pub targets: Vec<QueryTarget>,
    /// Extracted parameters
    pub parameters: HashMap<String, String>,
    /// Whether this is a cross-model join
    pub is_cross_model_join: bool,
}

/// Target of a query operation
#[derive(Debug, Clone)]
pub struct QueryTarget {
    /// Name of the table/collection
    pub name: String,
    /// Alias if specified
    pub alias: Option<String>,
    /// Model type for this target
    pub model_type: TargetModelType,
}

/// Model type for a query target
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetModelType {
    /// RDBMS table
    Table,
    /// Vector collection
    VectorCollection,
    /// Document collection
    DocumentCollection,
    /// Graph
    Graph,
    /// Observability namespace
    Observability,
    /// Unknown (needs resolution)
    Unknown,
}

/// Federated parser for SQL with multi-model extensions
pub struct FederatedParser {
    /// Extension keywords to detect
    extension_keywords: Vec<&'static str>,
}

impl FederatedParser {
    /// Create a new federated parser
    pub fn new() -> Self {
        Self {
            extension_keywords: vec![
                "VECTOR_SEARCH",
                "GRAPH_QUERY",
                "DOCUMENT_QUERY",
                "LOGS",
                "METRICS",
                "TRACES",
                "<->",
                "::vector",
            ],
        }
    }

    /// Get supported SQL extensions
    pub fn supported_extensions(&self) -> &[&'static str] {
        &self.extension_keywords
    }

    /// Parse a SQL query with extensions
    pub fn parse(&self, sql: &str) -> Result<FederatedQuery> {
        let sql_upper = sql.to_ascii_uppercase();
        let mut extensions = Vec::new();
        let mut extension_positions = Vec::new();
        let mut extension_aliases = Vec::new();
        let mut query_type = QueryType::Sql;
        let mut is_cross_model_join = false;

        for (extension, position, alias) in self.parse_function_extensions(sql) {
            query_type = Self::query_type_for_extension(&extension);
            extensions.push(extension);
            extension_positions.push(position);
            extension_aliases.push(alias);
        }

        // Detect vector distance operator <->
        if let Some((ext, position)) = self.parse_vector_distance(sql) {
            extensions.push(ext);
            extension_positions.push(position);
            extension_aliases.push(None);
            if query_type == QueryType::Sql {
                query_type = QueryType::VectorSearch;
            }
        }

        // Parse FROM clause targets, then drop the function-call
        // pseudo-targets (`RERANK('docs', ...)`, `VECTOR_SEARCH(...)`,
        // etc.). Each function-call SRF already lives in `extensions`
        // — leaving its raw FROM token in `targets` would make the
        // optimizer's `ordered_query_sources` emit both an Extension
        // sub-plan AND a Scan over a bogus collection name like
        // `"RERANK('docs',"`, then HashJoin them. R-7c.4c.1.
        let targets: Vec<_> = self
            .parse_from_targets(sql)
            .into_iter()
            .filter(|t| {
                let upper = t.name.to_uppercase();
                !upper.starts_with("VECTOR_SEARCH")
                    && !upper.starts_with("GRAPH_QUERY")
                    && !upper.starts_with("DOCUMENT_QUERY")
                    && !upper.starts_with("RERANK")
                    && !upper.starts_with("LOGS")
                    && !upper.starts_with("METRICS")
                    && !upper.starts_with("TRACES")
            })
            .collect();
        let real_target_count = targets.len();

        // Detect cross-model joins (only when we have multiple real tables or extensions)
        if extensions.len() > 1 || (!extensions.is_empty() && real_target_count > 1) {
            is_cross_model_join = true;
            query_type = QueryType::Federated;
        }

        // Check for JOIN with different model types
        if sql_upper.contains("JOIN") && sql_upper.contains("LATERAL") && !extensions.is_empty() {
            is_cross_model_join = true;
            query_type = QueryType::Federated;
        }

        Ok(FederatedQuery {
            sql: sql.to_string(),
            query_type,
            extensions,
            extension_positions,
            extension_aliases,
            targets,
            parameters: HashMap::new(),
            is_cross_model_join,
        })
    }

    fn query_type_for_extension(extension: &SqlExtension) -> QueryType {
        match extension {
            SqlExtension::VectorSearch { .. } | SqlExtension::VectorDistance { .. } => {
                QueryType::VectorSearch
            }
            SqlExtension::GraphQuery { .. } => QueryType::GraphQuery,
            SqlExtension::RerankSearch { .. } => QueryType::RerankSearch,
            SqlExtension::DocumentQuery { .. } => QueryType::DocumentQuery,
            SqlExtension::Logs { .. } => QueryType::LogQuery,
            SqlExtension::Metrics { .. } => QueryType::MetricQuery,
            SqlExtension::Traces { .. } => QueryType::TraceQuery,
        }
    }

    fn parse_function_extensions(&self, sql: &str) -> Vec<(SqlExtension, usize, Option<String>)> {
        // RERANK comes before LOGS/METRICS/TRACES so a query like
        // `SELECT * FROM RERANK(...)` is matched as a single token
        // rather than colliding with a partial prefix.
        const FUNCTION_NAMES: [&str; 7] = [
            "VECTOR_SEARCH",
            "GRAPH_QUERY",
            "DOCUMENT_QUERY",
            "RERANK",
            "LOGS",
            "METRICS",
            "TRACES",
        ];

        let sql_upper = sql.to_ascii_uppercase();
        let mut search_from = 0;
        let mut extensions = Vec::new();

        while search_from < sql_upper.len() {
            let next_match = FUNCTION_NAMES
                .iter()
                .filter_map(|function_name| {
                    let needle = format!("{}(", function_name);
                    sql_upper[search_from..]
                        .find(&needle)
                        .map(|relative| (*function_name, search_from + relative))
                })
                .min_by_key(|(_, position)| *position);

            let Some((function_name, position)) = next_match else {
                break;
            };

            if let Some((args, end_position)) =
                self.extract_function_args_at(sql, position, function_name)
            {
                if let Some(extension) = self.parse_function_extension(function_name, args) {
                    extensions.push((
                        extension,
                        position,
                        self.parse_extension_alias(sql, end_position),
                    ));
                }
                search_from = end_position;
            } else {
                search_from = position + function_name.len();
            }
        }

        extensions
    }

    fn parse_function_extension(&self, function_name: &str, args: &str) -> Option<SqlExtension> {
        match function_name {
            "VECTOR_SEARCH" => self.parse_vector_search_args(args),
            "GRAPH_QUERY" => self.parse_graph_query_args(args),
            "DOCUMENT_QUERY" => self.parse_document_query_args(args),
            "RERANK" => self.parse_rerank_args(args),
            "LOGS" => self.parse_logs_query_args(args),
            "METRICS" => self.parse_metrics_query_args(args),
            "TRACES" => self.parse_traces_query_args(args),
            _ => None,
        }
    }

    /// Parse `RERANK(collection, query_text, query_vector, k, rank_profile)`
    /// argument list. Positional args 1–3 are required; `k` defaults
    /// to 10; `rank_profile` defaults to absent (retrieval-only path).
    /// R-7c.4c.
    fn parse_rerank_args(&self, args: &str) -> Option<SqlExtension> {
        let parts = self.split_function_args(args);
        if parts.len() < 3 {
            return None;
        }
        let collection = Self::unquote_sql_string(&parts[0]);
        let query_text = Self::unquote_sql_string(&parts[1]);
        let query_vector = self.parse_vector_argument(&parts[2])?;
        let k = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
        // Empty rank_profile string → None (retrieval-only path); a
        // non-empty string → Some(name). Mirrors the REST DTO
        // `rank_profile: Option<String>` contract.
        let rank_profile = parts.get(4).and_then(|s| {
            let unquoted = Self::unquote_sql_string(s);
            if unquoted.is_empty() {
                None
            } else {
                Some(unquoted)
            }
        });
        Some(SqlExtension::RerankSearch {
            collection,
            query_text,
            query_vector,
            k,
            rank_profile,
        })
    }

    /// Parse VECTOR_SEARCH(collection, vector, top_k)
    fn parse_vector_search_args(&self, args: &str) -> Option<SqlExtension> {
        let parts = self.split_function_args(args);
        if parts.len() < 2 {
            return None;
        }

        let collection = Self::unquote_sql_string(&parts[0]);
        let query_vector = self.parse_vector_argument(&parts[1])?;
        let top_k = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

        Some(SqlExtension::VectorSearch {
            collection,
            query_vector,
            top_k,
        })
    }

    /// Parse GRAPH_QUERY('cypher')
    fn parse_graph_query_args(&self, args: &str) -> Option<SqlExtension> {
        let cypher = Self::unquote_sql_string(args);
        Some(SqlExtension::GraphQuery { cypher })
    }

    /// Parse DOCUMENT_QUERY(collection, filter)
    fn parse_document_query_args(&self, args: &str) -> Option<SqlExtension> {
        let parts = self.split_function_args(args);
        let collection = Self::unquote_sql_string(parts.first()?);
        let filter = parts.get(1).map(|s| Self::unquote_sql_string(s));
        Some(SqlExtension::DocumentQuery { collection, filter })
    }

    /// Parse LOGS(namespace)
    fn parse_logs_query_args(&self, args: &str) -> Option<SqlExtension> {
        let namespace = Self::unquote_sql_string(args);
        Some(SqlExtension::Logs { namespace })
    }

    /// Parse METRICS(namespace)
    fn parse_metrics_query_args(&self, args: &str) -> Option<SqlExtension> {
        let namespace = Self::unquote_sql_string(args);
        Some(SqlExtension::Metrics { namespace })
    }

    /// Parse TRACES(namespace)
    fn parse_traces_query_args(&self, args: &str) -> Option<SqlExtension> {
        let namespace = Self::unquote_sql_string(args);
        Some(SqlExtension::Traces { namespace })
    }

    /// Parse vector distance operator <->
    fn parse_vector_distance(&self, sql: &str) -> Option<(SqlExtension, usize)> {
        if !sql.contains("<->") {
            return None;
        }

        // Find <-> and extract left/right operands
        if let Some(pos) = sql.find("<->") {
            // Simple extraction - find word before and after
            let before = sql[..pos].trim();
            let after = sql[pos + 3..].trim();

            // Get last word before <->
            let left_column = before
                .split_whitespace()
                .last()
                .unwrap_or("embedding")
                .to_string();

            // A missing/unparseable RHS is an ERROR, not a silent empty
            // vector (a 0-dimension query dispatched to the kernels).
            let right_literal = Self::parse_vector_distance_rhs(after)?;

            return Some((
                SqlExtension::VectorDistance {
                    left_column,
                    right_literal,
                },
                pos,
            ));
        }
        None
    }

    fn parse_vector_distance_rhs(after_operator: &str) -> Option<String> {
        let trimmed = after_operator.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(rest) = trimmed.strip_prefix('\'') {
            let mut in_doubled_pair = false;
            for (idx, ch) in rest.char_indices() {
                if in_doubled_pair {
                    // second char of a doubled close-quote — the pair is
                    // literal content, NOT the close.
                    in_doubled_pair = false;
                    continue;
                }
                // POSTGRES dialect: a DOUBLED close-quote is an escaped
                // quote inside the literal; backslash is a literal.
                if ch == '\'' && rest[(idx + ch.len_utf8())..].starts_with('\'') {
                    in_doubled_pair = true;
                    continue;
                }
                if ch == '\'' {
                    let literal_end = 1 + idx + ch.len_utf8();
                    let mut end = literal_end;
                    let suffix = trimmed[literal_end..].trim_start();
                    if suffix.to_ascii_lowercase().starts_with("::vector") {
                        end = literal_end + (trimmed[literal_end..].len() - suffix.len());
                        let cast_suffix = &trimmed[end..];
                        let cast_len = Self::vector_cast_len(cast_suffix).unwrap_or(0);
                        end += cast_len;
                    }
                    return Some(trimmed[..end].trim().to_string());
                }
            }
            return None;
        }

        if trimmed.starts_with('[') {
            let mut depth = 0;
            for (idx, ch) in trimmed.char_indices() {
                match ch {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            let mut end = idx + ch.len_utf8();
                            let suffix = trimmed[end..].trim_start();
                            if suffix.to_ascii_lowercase().starts_with("::vector") {
                                end += trimmed[end..].len() - suffix.len();
                                let cast_len = Self::vector_cast_len(&trimmed[end..]).unwrap_or(0);
                                end += cast_len;
                            }
                            return Some(trimmed[..end].trim().to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        trimmed.split_whitespace().next().map(str::to_string)
    }

    fn vector_cast_len(value: &str) -> Option<usize> {
        let lower = value.to_ascii_lowercase();
        let vector = lower.strip_prefix("::vector")?;
        let mut len = "::vector".len();
        let rest = vector.trim_start();
        len += vector.len() - rest.len();
        if rest.starts_with('(') {
            let close = rest.find(')')?;
            len += close + 1;
        }
        Some(len)
    }

    /// Parse FROM clause to extract table targets
    fn parse_from_targets(&self, sql: &str) -> Vec<QueryTarget> {
        let mut targets = Vec::new();
        let upper = sql.to_ascii_uppercase();

        // Find FROM clause
        if let Some(from_pos) = upper.find("FROM") {
            let after_from = &sql[from_pos + 4..];

            // Find end of FROM clause (WHERE, JOIN, ORDER, GROUP, LIMIT, etc.)
            let end_keywords = ["WHERE", "JOIN", "ORDER", "GROUP", "LIMIT", "HAVING", ";"];
            let mut end_pos = after_from.len();
            for keyword in end_keywords {
                if let Some(pos) = after_from.to_ascii_uppercase().find(keyword)
                    && pos < end_pos
                {
                    end_pos = pos;
                }
            }

            let from_clause = after_from[..end_pos].trim();

            // Check if this is a function call (contains parentheses at start)
            let from_upper = from_clause.to_ascii_uppercase();
            let is_function_call = from_upper.starts_with("VECTOR_SEARCH")
                || from_upper.starts_with("GRAPH_QUERY")
                || from_upper.starts_with("DOCUMENT_QUERY")
                || from_upper.starts_with("LOGS(")
                || from_upper.starts_with("METRICS(")
                || from_upper.starts_with("TRACES(");

            // If it's a function call, don't split by comma
            if is_function_call {
                return targets; // Return empty - no real table targets
            }

            // Split by comma for multiple tables (respecting parentheses depth)
            let table_refs = self.split_respecting_parens(from_clause);
            for table_ref in table_refs {
                let parts: Vec<&str> = table_ref.split_whitespace().collect();
                if !parts.is_empty() {
                    let name = parts[0].to_string();
                    let alias = if parts.len() > 1 && !parts[1].eq_ignore_ascii_case("AS") {
                        Some(parts[1].to_string())
                    } else if parts.len() > 2 {
                        Some(parts[2].to_string())
                    } else {
                        None
                    };

                    targets.push(QueryTarget {
                        name,
                        alias,
                        model_type: TargetModelType::Unknown,
                    });
                }
            }
        }

        targets
    }

    /// Split a string by commas, respecting parentheses depth
    fn split_respecting_parens(&self, s: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for c in s.chars() {
            match c {
                '(' => {
                    depth += 1;
                    current.push(c);
                }
                ')' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    if !current.trim().is_empty() {
                        result.push(current.trim().to_string());
                    }
                    current = String::new();
                }
                _ => {
                    current.push(c);
                }
            }
        }

        if !current.trim().is_empty() {
            result.push(current.trim().to_string());
        }

        result
    }

    /// Split function arguments while respecting quotes, brackets, and parentheses.
    fn split_function_args(&self, s: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut paren_depth = 0;
        let mut bracket_depth = 0;
        let mut in_quote = None;

        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if let Some(quote) = in_quote {
                current.push(c);
                // POSTGRES dialect (standard_conforming_strings=on): the
                // only in-literal escape is a DOUBLED close-quote (both
                // chars stay); backslash is a LITERAL character.
                if c == quote {
                    if chars.peek() == Some(&quote) {
                        chars.next();
                        current.push(quote);
                        continue;
                    }
                    in_quote = None;
                }
                continue;
            }

            match c {
                '\'' | '"' => {
                    in_quote = Some(c);
                    current.push(c);
                }
                '(' => {
                    paren_depth += 1;
                    current.push(c);
                }
                ')' => {
                    paren_depth -= 1;
                    current.push(c);
                }
                '[' => {
                    bracket_depth += 1;
                    current.push(c);
                }
                ']' => {
                    bracket_depth -= 1;
                    current.push(c);
                }
                ',' if paren_depth == 0 && bracket_depth == 0 => {
                    if !current.trim().is_empty() {
                        result.push(current.trim().to_string());
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        if !current.trim().is_empty() {
            result.push(current.trim().to_string());
        }

        result
    }

    fn extract_function_args_at<'a>(
        &self,
        sql: &'a str,
        start: usize,
        function_name: &str,
    ) -> Option<(&'a str, usize)> {
        let content = &sql[start + function_name.len() + 1..];
        let mut depth = 1;
        let mut in_quote = None;

        let mut in_doubled_pair = false;
        for (i, c) in content.char_indices() {
            if let Some(quote) = in_quote {
                if in_doubled_pair {
                    // second char of a doubled close-quote — literal content
                    in_doubled_pair = false;
                    continue;
                }
                // POSTGRES dialect: doubled close-quote stays in-quote;
                // backslash is a literal.
                if c == quote {
                    if content[(i + c.len_utf8())..].starts_with(quote) {
                        in_doubled_pair = true;
                        continue;
                    }
                    in_quote = None;
                }
                continue;
            }

            match c {
                '\'' | '"' => in_quote = Some(c),
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        let end_position = start + function_name.len() + 2 + i;
                        return Some((&content[..i], end_position));
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn unquote_sql_string(s: &str) -> String {
        let trimmed = s.trim();
        if trimmed.len() >= 2 {
            let first = trimmed.as_bytes()[0];
            let last = trimmed.as_bytes()[trimmed.len() - 1];
            if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
                let inner = &trimmed[1..trimmed.len() - 1];
                let escaped = if first == b'\'' { "''" } else { "\"\"" };
                let quote = if first == b'\'' { "'" } else { "\"" };
                return inner.replace(escaped, quote);
            }
        }
        trimmed.to_string()
    }

    fn parse_extension_alias(&self, sql: &str, mut position: usize) -> Option<String> {
        const RESERVED_TOKENS: [&str; 15] = [
            "JOIN", "LEFT", "RIGHT", "INNER", "FULL", "CROSS", "LATERAL", "ON", "WHERE", "ORDER",
            "GROUP", "LIMIT", "HAVING", "UNION", "EXCEPT",
        ];

        while let Some(ch) = sql[position..].chars().next() {
            if ch.is_whitespace() {
                position += ch.len_utf8();
            } else {
                break;
            }
        }

        if position >= sql.len() {
            return None;
        }

        let mut remainder = sql[position..].trim_start();
        let mut token = Self::parse_alias_token(remainder)?;
        if token.eq_ignore_ascii_case("AS") {
            remainder = &remainder[token.len()..];
            remainder = remainder.trim_start();
            token = Self::parse_alias_token(remainder)?;
        }

        if !Self::is_quoted_identifier(token)
            && RESERVED_TOKENS.contains(&token.to_ascii_uppercase().as_str())
        {
            return None;
        }

        Some(token.to_string())
    }

    fn parse_alias_token(input: &str) -> Option<&str> {
        match input.chars().next()? {
            '"' => Self::parse_quoted_identifier_token(input),
            'A'..='Z' | 'a'..='z' | '_' => {
                let token_end = input
                    .char_indices()
                    .find(|(_, ch)| !matches!(ch, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_'))
                    .map(|(idx, _)| idx)
                    .unwrap_or(input.len());
                Some(&input[..token_end])
            }
            _ => None,
        }
    }

    fn parse_quoted_identifier_token(input: &str) -> Option<&str> {
        let mut chars = input.char_indices().peekable();
        let (_, first) = chars.next()?;
        if first != '"' {
            return None;
        }

        while let Some((idx, ch)) = chars.next() {
            if ch != '"' {
                continue;
            }

            if matches!(chars.peek(), Some((_, '"'))) {
                chars.next();
                continue;
            }

            return Some(&input[..idx + ch.len_utf8()]);
        }

        None
    }

    fn is_quoted_identifier(token: &str) -> bool {
        token.len() >= 2 && token.starts_with('"') && token.ends_with('"')
    }

    fn parse_vector_argument(&self, arg: &str) -> Option<VectorQuery> {
        let trimmed = arg.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(literal) = Self::parse_vector_literal(trimmed) {
            return Some(VectorQuery::Literal(literal));
        }

        Some(VectorQuery::Expression(trimmed.to_string()))
    }

    fn parse_vector_literal(raw: &str) -> Option<Vec<f32>> {
        let trimmed = raw.trim();
        let without_cast = Self::strip_vector_cast(trimmed);
        let unquoted = without_cast.trim_matches('\'').trim_matches('"').trim();

        if !(unquoted.starts_with('[') && unquoted.ends_with(']')) {
            return None;
        }

        let inner = &unquoted[1..unquoted.len() - 1];
        if inner.trim().is_empty() {
            // An empty vector is not a queryable literal (the optimizer twin
            // and the JSON twins reject it — do not dispatch a 0-dimension query).
            return None;
        }

        inner
            .split(',')
            .map(|value| {
                value
                    .trim()
                    .parse::<f32>()
                    .ok()
                    .filter(|component| component.is_finite())
            })
            .collect()
    }

    fn strip_vector_cast(value: &str) -> &str {
        let trimmed = value.trim();
        // Cheap gate (the lowercase scan allocated a full copy of large
        // literals per parse — see the optimizer sibling).
        if !trimmed.contains("::") {
            return trimmed;
        }
        let lower = trimmed.to_ascii_lowercase();
        let Some(cast_start) = lower.rfind("::vector") else {
            return trimmed;
        };
        let suffix = lower[cast_start + "::vector".len()..].trim();
        if suffix.is_empty()
            || (suffix.starts_with('(')
                && suffix.ends_with(')')
                && suffix[1..suffix.len() - 1]
                    .chars()
                    .all(|ch| ch.is_ascii_digit()))
        {
            trimmed[..cast_start].trim()
        } else {
            trimmed
        }
    }

    /// Resolve model types for targets using catalog
    pub fn resolve_model_types(&self, query: &mut FederatedQuery, catalog: &impl ModelTypeCatalog) {
        for target in &mut query.targets {
            target.model_type = catalog.get_model_type(&target.name);
        }
    }
}

impl Default for FederatedParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for resolving model types from catalog
pub trait ModelTypeCatalog {
    fn get_model_type(&self, name: &str) -> TargetModelType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_sql() {
        let parser = FederatedParser::new();
        let query = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();
        assert_eq!(query.query_type, QueryType::Sql);
        assert!(query.extensions.is_empty());
        assert_eq!(query.targets.len(), 1);
        assert_eq!(query.targets[0].name, "users");
    }

    #[test]
    fn test_parse_vector_search() {
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM VECTOR_SEARCH('embeddings', '[0.1,0.2]', 10)")
            .unwrap();
        assert_eq!(query.query_type, QueryType::VectorSearch);
        assert_eq!(query.extensions.len(), 1);
        match &query.extensions[0] {
            SqlExtension::VectorSearch {
                collection,
                query_vector,
                top_k,
            } => {
                assert_eq!(collection, "embeddings");
                assert_eq!(*query_vector, VectorQuery::Literal(vec![0.1, 0.2]));
                assert_eq!(*top_k, 10);
            }
            _ => panic!("Expected VectorSearch extension"),
        }
    }

    #[test]
    fn test_parse_vector_search_with_expression() {
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM VECTOR_SEARCH('embeddings', u.preference_vector, 5)")
            .unwrap();

        match &query.extensions[0] {
            SqlExtension::VectorSearch { query_vector, .. } => {
                assert_eq!(
                    *query_vector,
                    VectorQuery::Expression("u.preference_vector".to_string())
                );
            }
            _ => panic!("Expected VectorSearch extension"),
        }
    }

    // ---------------- R-7c.4c: RERANK() parser ----------------

    #[test]
    fn test_parse_rerank_with_all_args() {
        let parser = FederatedParser::new();
        let query = parser
            .parse(
                "SELECT * FROM RERANK('docs', 'laptop computer', '[0.1,0.2,0.3]', 50, 'semantic_plus_ce')",
            )
            .unwrap();
        assert_eq!(query.query_type, QueryType::RerankSearch);
        assert_eq!(query.extensions.len(), 1);
        match &query.extensions[0] {
            SqlExtension::RerankSearch {
                collection,
                query_text,
                query_vector,
                k,
                rank_profile,
            } => {
                assert_eq!(collection, "docs");
                assert_eq!(query_text, "laptop computer");
                assert_eq!(*query_vector, VectorQuery::Literal(vec![0.1, 0.2, 0.3]));
                assert_eq!(*k, 50);
                assert_eq!(rank_profile.as_deref(), Some("semantic_plus_ce"));
            }
            _ => panic!("Expected RerankSearch extension"),
        }
    }

    #[test]
    fn test_parse_rerank_defaults_k_to_ten() {
        // Omitting `k` and `rank_profile` → k=10, rank_profile=None.
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM RERANK('docs', 'q', '[0.5]')")
            .unwrap();
        match &query.extensions[0] {
            SqlExtension::RerankSearch {
                k, rank_profile, ..
            } => {
                assert_eq!(*k, 10);
                assert!(rank_profile.is_none());
            }
            _ => panic!("Expected RerankSearch extension"),
        }
    }

    #[test]
    fn test_parse_rerank_empty_profile_treated_as_none() {
        // An empty string profile name → None (retrieval-only path).
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM RERANK('docs', 'q', '[0.5]', 5, '')")
            .unwrap();
        match &query.extensions[0] {
            SqlExtension::RerankSearch { rank_profile, .. } => {
                assert!(rank_profile.is_none());
            }
            _ => panic!("Expected RerankSearch extension"),
        }
    }

    #[test]
    fn test_parse_rerank_rejects_missing_required_args() {
        // Only 2 args (collection + query_text), no query_vector →
        // the extension fails to parse, so query falls back to
        // QueryType::Sql with no extensions detected.
        let parser = FederatedParser::new();
        let query = parser.parse("SELECT * FROM RERANK('docs', 'q')").unwrap();
        assert_eq!(query.query_type, QueryType::Sql);
        assert!(query.extensions.is_empty());
    }

    #[test]
    fn test_parse_rerank_with_query_vector_column_reference() {
        // Production callers pass an expression (e.g. an embedded
        // table-column reference) rather than a literal vector. The
        // parser must accept it as VectorQuery::Expression.
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM RERANK('docs', 'q', u.embedding, 20, 'p')")
            .unwrap();
        match &query.extensions[0] {
            SqlExtension::RerankSearch {
                query_vector,
                k,
                rank_profile,
                ..
            } => {
                assert_eq!(*query_vector, VectorQuery::Expression("u.embedding".into()));
                assert_eq!(*k, 20);
                assert_eq!(rank_profile.as_deref(), Some("p"));
            }
            _ => panic!("Expected RerankSearch extension"),
        }
    }

    #[test]
    fn test_parse_rerank_query_type_routes_through_extension_mapping() {
        // The extension-to-QueryType mapping must point RERANK at
        // QueryType::RerankSearch — verified end-to-end through the
        // parser entry point.
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM RERANK('docs', 'q', '[0.5]', 5, 'p')")
            .unwrap();
        assert_eq!(query.query_type, QueryType::RerankSearch);
    }

    #[test]
    fn test_parse_graph_query() {
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM GRAPH_QUERY('MATCH (a)-[:KNOWS]->(b) RETURN b.name')")
            .unwrap();
        assert_eq!(query.query_type, QueryType::GraphQuery);
        assert_eq!(query.extensions.len(), 1);
        match &query.extensions[0] {
            SqlExtension::GraphQuery { cypher } => {
                assert!(cypher.contains("MATCH"));
                assert!(cypher.contains("KNOWS"));
            }
            _ => panic!("Expected GraphQuery extension"),
        }
    }

    #[test]
    fn postgres_string_dialect_is_pinned() {
        // The ONLY in-literal escape is quote-doubling; backslash is a
        // LITERAL (standard_conforming_strings=on). A trailing backslash
        // does not escape the close-quote (rounds 23/24/28 churned this).
        let parser = FederatedParser::new();
        let path_arg = parser.parse("SELECT * FROM LOGS('C:\\dir\\') ORDER BY ts LIMIT 5");
        // The extension is recognized and the namespace keeps the
        // backslashes (no phantom literal swallowing ORDER BY/LIMIT).
        let query = path_arg.expect("trailing backslash does not break parsing");
        assert!(
            matches!(&query.extensions[0], SqlExtension::Logs { namespace }
                if namespace == "C:\\dir\\"),
            "namespace must keep the backslashes exactly: {:?}",
            query.extensions[0]
        );

        // Doubled quotes stay INSIDE the literal (one arg, decoded).
        let doubled = parser
            .parse("SELECT * FROM LOGS('team''s-prod')")
            .expect("doubled quote is one literal");
        assert!(
            matches!(&doubled.extensions[0], SqlExtension::Logs { namespace }
                if namespace == "team's-prod")
        );

        assert_eq!(
            FederatedParser::parse_vector_distance_rhs("'[0.1]''tag'::vector LIMIT 1"),
            Some("'[0.1]''tag'::vector".to_string()),
            "a doubled quote pair must not close the RHS literal early"
        );
    }

    #[test]
    fn sql_string_escapes_are_decoded_for_extension_arguments() {
        let parser = FederatedParser::new();

        let vector = parser
            .parse("SELECT * FROM VECTOR_SEARCH('team''s-vectors', '[0.1]', 1)")
            .unwrap();
        assert!(matches!(
            &vector.extensions[0],
            SqlExtension::VectorSearch { collection, .. } if collection == "team's-vectors"
        ));

        let document = parser
            .parse("SELECT * FROM DOCUMENT_QUERY('team''s-docs', 'owner = \"O''Brien\"')")
            .unwrap();
        assert!(matches!(
            &document.extensions[0],
            SqlExtension::DocumentQuery {
                collection,
                filter: Some(filter),
            } if collection == "team's-docs" && filter == "owner = \"O'Brien\""
        ));

        let graph = parser
            .parse("SELECT * FROM GRAPH_QUERY('MATCH (n) RETURN ''label''')")
            .unwrap();
        assert!(matches!(
            &graph.extensions[0],
            SqlExtension::GraphQuery { cypher } if cypher == "MATCH (n) RETURN 'label'"
        ));

        let logs = parser
            .parse("SELECT * FROM LOGS('team''s-production')")
            .unwrap();
        assert!(matches!(
            &logs.extensions[0],
            SqlExtension::Logs { namespace } if namespace == "team's-production"
        ));
    }

    #[test]
    fn vector_literals_reject_non_finite_components() {
        assert!(FederatedParser::parse_vector_literal("'[NaN]'").is_none());
        assert!(FederatedParser::parse_vector_literal("'[inf]'").is_none());
        assert!(FederatedParser::parse_vector_literal("'[-inf]'").is_none());
        assert!(FederatedParser::parse_vector_literal("'[1e300]'").is_none());
    }

    #[test]
    fn test_parse_vector_distance() {
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM products ORDER BY embedding <-> '[0.1,0.2]'::vector LIMIT 10")
            .unwrap();
        assert_eq!(query.query_type, QueryType::VectorSearch);
        assert_eq!(query.extensions.len(), 1);
        match &query.extensions[0] {
            SqlExtension::VectorDistance {
                left_column,
                right_literal,
            } => {
                assert_eq!(left_column, "embedding");
                assert_eq!(right_literal, "'[0.1,0.2]'::vector");
            }
            _ => panic!("Expected VectorDistance extension"),
        }
    }

    #[test]
    fn test_parse_pgvector_distance_with_spaced_literal_and_dimension() {
        let parser = FederatedParser::new();
        let query = parser
            .parse(
                "SELECT id FROM memories ORDER BY embedding <-> '[0.1, 0.2, -0.3]'::vector(3) LIMIT 5",
            )
            .unwrap();

        assert_eq!(query.query_type, QueryType::VectorSearch);
        match &query.extensions[0] {
            SqlExtension::VectorDistance {
                left_column,
                right_literal,
            } => {
                assert_eq!(left_column, "embedding");
                assert_eq!(right_literal, "'[0.1, 0.2, -0.3]'::vector(3)");
            }
            other => panic!("expected vector distance, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_vector_search_accepts_pgvector_cast_dimension() {
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM VECTOR_SEARCH('memories', '[0.1, 0.2]'::vector(2), 3)")
            .unwrap();

        match &query.extensions[0] {
            SqlExtension::VectorSearch {
                collection,
                query_vector,
                top_k,
            } => {
                assert_eq!(collection, "memories");
                assert_eq!(query_vector, &VectorQuery::Literal(vec![0.1, 0.2]));
                assert_eq!(*top_k, 3);
            }
            other => panic!("expected vector search, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_logs_query() {
        let parser = FederatedParser::new();
        let query = parser
            .parse("SELECT * FROM LOGS('production') WHERE timestamp > now() - interval '1h'")
            .unwrap();
        assert_eq!(query.query_type, QueryType::LogQuery);
        match &query.extensions[0] {
            SqlExtension::Logs { namespace } => {
                assert_eq!(namespace, "production");
            }
            _ => panic!("Expected Logs extension"),
        }
    }

    #[test]
    fn test_parse_traces_query() {
        let parser = FederatedParser::new();
        let query = parser.parse("SELECT * FROM TRACES('production')").unwrap();
        assert_eq!(query.query_type, QueryType::TraceQuery);
        match &query.extensions[0] {
            SqlExtension::Traces { namespace } => {
                assert_eq!(namespace, "production");
            }
            other => panic!("Expected Traces extension, got {other:?}"),
        }
        assert!(query.targets.is_empty());
    }

    #[test]
    fn test_parse_cross_model_join() {
        let parser = FederatedParser::new();
        let query = parser.parse(
            "SELECT u.*, v.similar FROM users u JOIN LATERAL VECTOR_SEARCH('embeddings', u.vector, 10) v ON true"
        ).unwrap();
        assert_eq!(query.query_type, QueryType::Federated);
        assert!(query.is_cross_model_join);
    }

    #[test]
    fn test_parse_multiple_function_sources_preserves_order_and_aliases() {
        let parser = FederatedParser::new();
        let query = parser
            .parse(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') p, DOCUMENT_QUERY('right_profiles') q JOIN LATERAL VECTOR_SEARCH('products', q.document.embedding, 1) v ON true",
            )
            .expect("parser should preserve repeated function-backed sources");

        assert_eq!(query.extensions.len(), 3);
        assert_eq!(
            query.extension_aliases,
            vec![
                Some("p".to_string()),
                Some("q".to_string()),
                Some("v".to_string())
            ]
        );
        assert!(query.extension_positions.windows(2).all(|w| w[0] < w[1]));

        match &query.extensions[0] {
            SqlExtension::DocumentQuery { collection, .. } => {
                assert_eq!(collection, "left_profiles");
            }
            other => panic!("expected first extension to be left document query, got {other:?}"),
        }
        match &query.extensions[1] {
            SqlExtension::DocumentQuery { collection, .. } => {
                assert_eq!(collection, "right_profiles");
            }
            other => panic!("expected second extension to be right document query, got {other:?}"),
        }
        match &query.extensions[2] {
            SqlExtension::VectorSearch { query_vector, .. } => {
                assert_eq!(
                    query_vector,
                    &VectorQuery::Expression("q.document.embedding".to_string())
                );
            }
            other => panic!("expected third extension to be vector search, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_multiple_function_sources_preserves_quoted_aliases() {
        let parser = FederatedParser::new();
        let query = parser
            .parse(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') \"LeftAlias\", DOCUMENT_QUERY('right_profiles') AS \"RightAlias\" JOIN LATERAL VECTOR_SEARCH('products', \"RightAlias\".document.embedding, 1) v ON true",
            )
            .expect("parser should preserve quoted function-backed source aliases");

        assert_eq!(query.extensions.len(), 3);
        assert_eq!(
            query.extension_aliases,
            vec![
                Some("\"LeftAlias\"".to_string()),
                Some("\"RightAlias\"".to_string()),
                Some("v".to_string())
            ]
        );

        match &query.extensions[2] {
            SqlExtension::VectorSearch { query_vector, .. } => {
                assert_eq!(
                    query_vector,
                    &VectorQuery::Expression("\"RightAlias\".document.embedding".to_string())
                );
            }
            other => panic!("expected third extension to be vector search, got {other:?}"),
        }
    }

    #[test]
    fn test_supported_extensions() {
        let parser = FederatedParser::new();
        let extensions = parser.supported_extensions();
        assert!(extensions.contains(&"VECTOR_SEARCH"));
        assert!(extensions.contains(&"GRAPH_QUERY"));
        assert!(extensions.contains(&"LOGS"));
        assert!(extensions.contains(&"TRACES"));
    }

    #[test]
    fn test_parse_cross_modal_document_vector_graph_query() {
        let parser = FederatedParser::new();
        let query = parser
            .parse(
                "SELECT d.id, v.score, g.path \
                 FROM DOCUMENT_QUERY('agent_docs', '$.role = \"planner\"') d \
                 JOIN LATERAL VECTOR_SEARCH('agent_vectors', d.document.embedding, 5) v ON true \
                 JOIN LATERAL GRAPH_QUERY('MATCH (s:Symbol)-[:CALLS]->(t) RETURN t') g ON true",
            )
            .expect("cross-modal query should parse");

        assert_eq!(query.query_type, QueryType::Federated);
        assert!(query.is_cross_model_join);
        assert_eq!(query.extensions.len(), 3);
        assert_eq!(
            query.extension_aliases,
            vec![
                Some("d".to_string()),
                Some("v".to_string()),
                Some("g".to_string())
            ]
        );

        match &query.extensions[0] {
            SqlExtension::DocumentQuery { collection, filter } => {
                assert_eq!(collection, "agent_docs");
                assert_eq!(filter.as_deref(), Some("$.role = \"planner\""));
            }
            other => panic!("expected document query, got {other:?}"),
        }
        match &query.extensions[1] {
            SqlExtension::VectorSearch {
                collection,
                query_vector,
                top_k,
            } => {
                assert_eq!(collection, "agent_vectors");
                assert_eq!(
                    query_vector,
                    &VectorQuery::Expression("d.document.embedding".to_string())
                );
                assert_eq!(*top_k, 5);
            }
            other => panic!("expected vector search, got {other:?}"),
        }
        match &query.extensions[2] {
            SqlExtension::GraphQuery { cypher } => {
                assert_eq!(cypher, "MATCH (s:Symbol)-[:CALLS]->(t) RETURN t");
            }
            other => panic!("expected graph query, got {other:?}"),
        }
    }
}
