//! # Prepared Statement Implementation
//!
//! Core types and cache implementation for prepared statements.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use thiserror::Error;
use tracing::{debug, info};

use crate::query::federated::{
    CrossModelOptimizer, FederatedParser, FederatedQuery, FederatedQueryPlan,
};

/// Unique identifier for a prepared statement
pub type PreparedStatementId = String;

/// Error types for prepared statement operations
#[derive(Debug, Error)]
pub enum PreparedStatementError {
    /// Statement not found in cache
    #[error("Prepared statement not found: {0}")]
    NotFound(PreparedStatementId),

    /// Failed to parse the SQL query
    #[error("Failed to parse SQL: {0}")]
    ParseError(String),

    /// Failed to optimize the query plan
    #[error("Failed to optimize query: {0}")]
    OptimizationError(String),

    /// Invalid parameter binding
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Parameter count mismatch
    #[error("Expected {expected} parameters, got {actual}")]
    ParameterCountMismatch {
        /// Number of parameters expected
        expected: usize,
        /// Number of parameters provided
        actual: usize,
    },

    /// Cache is full
    #[error("Statement cache is full (max: {0})")]
    CacheFull(usize),

    /// Statement has expired
    #[error("Prepared statement has expired: {0}")]
    Expired(PreparedStatementId),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for prepared statement operations
pub type PreparedResult<T> = Result<T, PreparedStatementError>;

/// Configuration for the prepared statement cache
#[derive(Debug, Clone)]
pub struct PreparedStatementConfig {
    /// Maximum number of cached statements (default: 1000)
    pub max_statements: usize,
    /// Default TTL for statements (default: 1 hour)
    pub default_ttl: Duration,
    /// Enable automatic cleanup of expired statements (default: true)
    pub enable_cleanup: bool,
    /// Cleanup interval (default: 5 minutes)
    pub cleanup_interval: Duration,
}

impl Default for PreparedStatementConfig {
    fn default() -> Self {
        Self {
            max_statements: 1000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            enable_cleanup: true,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Parameter value that can be bound to a prepared statement
#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Null value
    Null,
    /// Vector (for VECTOR_SEARCH)
    Vector(Vec<f32>),
    /// JSON value
    Json(serde_json::Value),
}

impl From<&str> for ParameterValue {
    fn from(s: &str) -> Self {
        ParameterValue::String(s.to_string())
    }
}

impl From<String> for ParameterValue {
    fn from(s: String) -> Self {
        ParameterValue::String(s)
    }
}

impl From<i64> for ParameterValue {
    fn from(i: i64) -> Self {
        ParameterValue::Int(i)
    }
}

impl From<f64> for ParameterValue {
    fn from(f: f64) -> Self {
        ParameterValue::Float(f)
    }
}

impl From<bool> for ParameterValue {
    fn from(b: bool) -> Self {
        ParameterValue::Bool(b)
    }
}

impl From<Vec<f32>> for ParameterValue {
    fn from(v: Vec<f32>) -> Self {
        ParameterValue::Vector(v)
    }
}

impl From<serde_json::Value> for ParameterValue {
    fn from(v: serde_json::Value) -> Self {
        ParameterValue::Json(v)
    }
}

/// Single-quote SQL text with '' escaping — the ONE quoting rule shared by
/// every parameter splice (String/Vector/Json) and the literal path in
/// `unified_query_port_impl` (the direct and literal paths must agree on
/// quoting for federated equality to be possible at all).
pub(crate) fn escape_sql_text(value: &str) -> String {
    value.replace('\'', "''")
}

pub(crate) fn sql_quote(value: &str) -> String {
    format!("'{}'", escape_sql_text(value))
}

impl ParameterValue {
    /// Convert to SQL string representation
    pub fn to_sql_string(&self) -> String {
        match self {
            ParameterValue::String(s) => sql_quote(s),
            ParameterValue::Int(i) => i.to_string(),
            // Non-finite floats have no SQL literal — bare 'NaN'/'inf'
            // text is a parse error on Postgres-style engines ("column
            // does not exist"). NULL is valid everywhere and, per SQL
            // 3VL, compares to nothing — matching the canonical JSON
            // rendering's null.
            ParameterValue::Float(f) if !f.is_finite() => "NULL".to_string(),
            ParameterValue::Float(f) => f.to_string(),
            ParameterValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            ParameterValue::Null => "NULL".to_string(),
            // Quoted JSON-array TEXT (brackets kept — both internal
            // vector parsers require '[' after quote-stripping; non-finite
            // elements splice as their text inside the quotes — valid SQL,
            // needs a cast to compare).
            ParameterValue::Vector(v) => {
                let formatted: Vec<String> = v.iter().map(|f| f.to_string()).collect();
                sql_quote(&format!("[{}]", formatted.join(",")))
            }
            // JSON splices QUOTED: bare '[0,1,2]' / '{"x":1}' is a parse
            // error on every engine (the text may still need a cast to
            // compare — see the TD-tracked federated literal dialect gap).
            ParameterValue::Json(v) => sql_quote(&v.to_string()),
        }
    }
}

/// Parameter binding specification
#[derive(Debug, Clone)]
pub struct ParameterBinding {
    /// Parameter index (1-based, matching $1, $2, etc.)
    pub index: usize,
    /// Parameter name (optional, for named parameters)
    pub name: Option<String>,
    /// Expected type hint (optional)
    pub type_hint: Option<String>,
    /// Position in the original SQL
    pub position: usize,
}

/// A prepared statement with parsed query and optimized plan
#[derive(Debug)]
pub struct PreparedStatement {
    /// Unique statement ID
    pub id: PreparedStatementId,
    /// Original SQL with parameter placeholders
    pub original_sql: String,
    /// Parsed federated query (Arc for sharing without Clone)
    pub parsed_query: Arc<FederatedQuery>,
    /// Optimized query plan (Arc for sharing without Clone)
    pub optimized_plan: Arc<FederatedQueryPlan>,
    /// Parameter bindings extracted from the SQL
    pub parameter_bindings: Vec<ParameterBinding>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Number of times this statement has been executed
    pub execution_count: u64,
}

impl Clone for PreparedStatement {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            original_sql: self.original_sql.clone(),
            parsed_query: Arc::clone(&self.parsed_query),
            optimized_plan: Arc::clone(&self.optimized_plan),
            parameter_bindings: self.parameter_bindings.clone(),
            created_at: self.created_at,
            execution_count: self.execution_count,
        }
    }
}

impl PreparedStatement {
    /// Create a new prepared statement
    pub fn new(
        id: PreparedStatementId,
        sql: &str,
        parser: &FederatedParser,
        optimizer: &CrossModelOptimizer,
    ) -> PreparedResult<Self> {
        // Extract parameter bindings from the SQL
        let (parameter_bindings, normalized_sql) = Self::extract_parameters(sql)?;

        // Parse the query
        let parsed_query = parser
            .parse(&normalized_sql)
            .map_err(|e| PreparedStatementError::ParseError(e.to_string()))?;

        // Optimize the query plan
        let optimized_plan = optimizer
            .optimize(&parsed_query)
            .map_err(|e| PreparedStatementError::OptimizationError(e.to_string()))?;

        Ok(Self {
            id,
            original_sql: sql.to_string(),
            parsed_query: Arc::new(parsed_query),
            optimized_plan: Arc::new(optimized_plan),
            parameter_bindings,
            created_at: Instant::now(),
            execution_count: 0,
        })
    }

    /// Extract parameter placeholders ($1, $2, etc.) from SQL
    fn extract_parameters(sql: &str) -> PreparedResult<(Vec<ParameterBinding>, String)> {
        let mut bindings = Vec::new();
        let mut normalized = String::with_capacity(sql.len());
        let mut chars = sql.char_indices().peekable();
        let mut in_string = false;
        let mut string_char = '"';

        while let Some((pos, c)) = chars.next() {
            // Track string literals to avoid matching $ inside them
            if c == '\'' || c == '"' {
                if !in_string {
                    in_string = true;
                    string_char = c;
                } else if c == string_char {
                    in_string = false;
                }
                normalized.push(c);
                continue;
            }

            if in_string {
                normalized.push(c);
                continue;
            }

            // Check for parameter placeholder
            if c == '$' {
                // Collect the parameter number
                let mut num_str = String::new();
                while let Some(&(_, next_c)) = chars.peek() {
                    if next_c.is_ascii_digit() {
                        num_str.push(next_c);
                        chars.next();
                    } else {
                        break;
                    }
                }

                if !num_str.is_empty() {
                    let index: usize = num_str.parse().map_err(|_| {
                        PreparedStatementError::InvalidParameter(format!(
                            "Invalid parameter index: ${}",
                            num_str
                        ))
                    })?;

                    bindings.push(ParameterBinding {
                        index,
                        name: None,
                        type_hint: None,
                        position: pos,
                    });

                    // Keep the placeholder in normalized SQL for now
                    normalized.push('$');
                    normalized.push_str(&num_str);
                } else {
                    normalized.push(c);
                }
            } else {
                normalized.push(c);
            }
        }

        // Sort bindings by index and validate
        bindings.sort_by_key(|b| b.index);

        // Check for gaps in parameter indices
        let mut seen_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for binding in &bindings {
            seen_indices.insert(binding.index);
        }

        if !seen_indices.is_empty() {
            // TD-007: unwrap_or with safe default - 0 for empty index set
            let max_index = *seen_indices.iter().max().unwrap_or(&0);
            for i in 1..=max_index {
                if !seen_indices.contains(&i) {
                    return Err(PreparedStatementError::InvalidParameter(format!(
                        "Missing parameter ${}",
                        i
                    )));
                }
            }
        }

        // Deduplicate bindings (same parameter can appear multiple times)
        let mut unique_bindings: Vec<ParameterBinding> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for binding in bindings {
            if !seen.contains(&binding.index) {
                seen.insert(binding.index);
                unique_bindings.push(binding);
            }
        }

        Ok((unique_bindings, normalized))
    }

    /// Substitute parameters into the SQL
    pub fn substitute_parameters(&self, params: &[ParameterValue]) -> PreparedResult<String> {
        if params.len() != self.parameter_bindings.len() {
            return Err(PreparedStatementError::ParameterCountMismatch {
                expected: self.parameter_bindings.len(),
                actual: params.len(),
            });
        }

        let mut result = self.original_sql.clone();

        // Sort bindings by position in reverse order to avoid index shifting
        let mut bindings_with_params: Vec<_> = self.parameter_bindings.iter().enumerate().collect();
        bindings_with_params.sort_by_key(|b| std::cmp::Reverse(b.1.position));

        // Replace from the end to avoid position shifts
        for (param_idx, _binding) in &bindings_with_params {
            let param_value = &params[*param_idx];
            let placeholder = format!("${}", *param_idx + 1);
            result = result.replace(&placeholder, &param_value.to_sql_string());
        }

        Ok(result)
    }

    /// Get the number of expected parameters
    pub fn parameter_count(&self) -> usize {
        self.parameter_bindings.len()
    }
}

/// Cached statement with TTL tracking
#[derive(Debug)]
pub struct CachedStatement {
    /// The prepared statement
    pub statement: PreparedStatement,
    /// Time-to-live for this statement
    pub ttl: Duration,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access count for LRU eviction
    pub access_count: u64,
}

impl CachedStatement {
    /// Check if this statement has expired
    pub fn is_expired(&self) -> bool {
        self.last_accessed.elapsed() > self.ttl
    }

    /// Update last access time
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Thread-safe cache for prepared statements
pub struct PreparedStatementCache {
    /// The statement cache using DashMap for concurrent access
    cache: DashMap<PreparedStatementId, CachedStatement>,
    /// Parser for SQL queries
    parser: Arc<FederatedParser>,
    /// Query optimizer
    optimizer: Arc<CrossModelOptimizer>,
    /// Configuration
    config: PreparedStatementConfig,
    /// Statement ID counter for generating unique IDs
    next_id: std::sync::atomic::AtomicU64,
}

impl PreparedStatementCache {
    /// Create a new prepared statement cache
    pub fn new(config: PreparedStatementConfig) -> Self {
        Self {
            cache: DashMap::new(),
            parser: Arc::new(FederatedParser::new()),
            optimizer: Arc::new(CrossModelOptimizer::new()),
            config,
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create a new cache with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PreparedStatementConfig::default())
    }

    /// Generate a unique statement ID
    fn generate_id(&self) -> PreparedStatementId {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("stmt_{:016x}", id)
    }

    /// Prepare a SQL statement and cache it
    pub fn prepare(&self, sql: &str) -> PreparedResult<PreparedStatementId> {
        self.prepare_with_ttl(sql, self.config.default_ttl)
    }

    /// Prepare a SQL statement with a custom TTL
    pub fn prepare_with_ttl(
        &self,
        sql: &str,
        ttl: Duration,
    ) -> PreparedResult<PreparedStatementId> {
        // Check cache size
        if self.cache.len() >= self.config.max_statements {
            // Try to evict expired statements first
            self.cleanup_expired();

            // If still full, return error
            if self.cache.len() >= self.config.max_statements {
                return Err(PreparedStatementError::CacheFull(
                    self.config.max_statements,
                ));
            }
        }

        let id = self.generate_id();
        let statement = PreparedStatement::new(id.clone(), sql, &self.parser, &self.optimizer)?;

        let cached = CachedStatement {
            statement,
            ttl,
            last_accessed: Instant::now(),
            access_count: 0,
        };

        self.cache.insert(id.clone(), cached);

        info!(
            statement_id = %id,
            parameter_count = self.cache.get(&id).map_or(0, |s| s.statement.parameter_count()),
            "Prepared statement cached"
        );

        Ok(id)
    }

    /// Get a prepared statement by ID
    pub fn get(&self, id: &str) -> PreparedResult<PreparedStatement> {
        let mut entry = self
            .cache
            .get_mut(id)
            .ok_or_else(|| PreparedStatementError::NotFound(id.to_string()))?;

        // Check expiration
        if entry.is_expired() {
            drop(entry);
            self.cache.remove(id);
            return Err(PreparedStatementError::Expired(id.to_string()));
        }

        // Update access tracking
        entry.touch();

        Ok(entry.statement.clone())
    }

    /// Execute a prepared statement with parameters
    pub fn execute_sql(&self, id: &str, params: &[ParameterValue]) -> PreparedResult<String> {
        let mut entry = self
            .cache
            .get_mut(id)
            .ok_or_else(|| PreparedStatementError::NotFound(id.to_string()))?;

        // Check expiration
        if entry.is_expired() {
            drop(entry);
            self.cache.remove(id);
            return Err(PreparedStatementError::Expired(id.to_string()));
        }

        // Update access tracking
        entry.touch();
        let statement = &entry.statement;

        // Substitute parameters and return the final SQL
        statement.substitute_parameters(params)
    }

    /// Delete a prepared statement
    pub fn drop_statement(&self, id: &str) -> PreparedResult<()> {
        self.cache
            .remove(id)
            .map(|_| {
                debug!(statement_id = %id, "Prepared statement dropped");
            })
            .ok_or_else(|| PreparedStatementError::NotFound(id.to_string()))
    }

    /// Check if a statement exists
    pub fn exists(&self, id: &str) -> bool {
        self.cache.contains_key(id)
    }

    /// Get the number of cached statements
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Cleanup expired statements
    pub fn cleanup_expired(&self) -> usize {
        let expired_ids: Vec<PreparedStatementId> = self
            .cache
            .iter()
            .filter(|entry| entry.value().is_expired())
            .map(|entry| entry.key().clone())
            .collect();

        let count = expired_ids.len();

        for id in expired_ids {
            self.cache.remove(&id);
        }

        if count > 0 {
            debug!(
                expired_count = count,
                "Cleaned up expired prepared statements"
            );
        }

        count
    }

    /// Clear all cached statements
    pub fn clear(&self) {
        let count = self.cache.len();
        self.cache.clear();
        info!(cleared_count = count, "Cleared all prepared statements");
    }

    /// Get cache statistics
    pub fn stats(&self) -> PreparedStatementCacheStats {
        let mut total_executions = 0u64;
        let mut oldest_access = Instant::now();
        let mut total_access_count = 0u64;

        for entry in &self.cache {
            total_executions += entry.statement.execution_count;
            total_access_count += entry.access_count;
            if entry.last_accessed < oldest_access {
                oldest_access = entry.last_accessed;
            }
        }

        PreparedStatementCacheStats {
            cached_statements: self.cache.len(),
            max_statements: self.config.max_statements,
            total_executions,
            total_access_count,
            oldest_statement_age_secs: if self.cache.is_empty() {
                0
            } else {
                oldest_access.elapsed().as_secs()
            },
        }
    }

    /// Get the optimized query plan for a prepared statement
    pub fn get_plan(&self, id: &str) -> PreparedResult<Arc<FederatedQueryPlan>> {
        let entry = self
            .cache
            .get(id)
            .ok_or_else(|| PreparedStatementError::NotFound(id.to_string()))?;

        if entry.is_expired() {
            drop(entry);
            self.cache.remove(id);
            return Err(PreparedStatementError::Expired(id.to_string()));
        }

        Ok(Arc::clone(&entry.statement.optimized_plan))
    }

    /// Get the parsed query for a prepared statement
    pub fn get_parsed_query(&self, id: &str) -> PreparedResult<Arc<FederatedQuery>> {
        let entry = self
            .cache
            .get(id)
            .ok_or_else(|| PreparedStatementError::NotFound(id.to_string()))?;

        if entry.is_expired() {
            drop(entry);
            self.cache.remove(id);
            return Err(PreparedStatementError::Expired(id.to_string()));
        }

        Ok(Arc::clone(&entry.statement.parsed_query))
    }
}

impl Default for PreparedStatementCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct PreparedStatementCacheStats {
    /// Number of currently cached statements
    pub cached_statements: usize,
    /// Maximum allowed statements
    pub max_statements: usize,
    /// Total number of statement executions
    pub total_executions: u64,
    /// Total access count across all statements
    pub total_access_count: u64,
    /// Age of the oldest statement in seconds
    pub oldest_statement_age_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_extraction() {
        let sql = "SELECT * FROM VECTOR_SEARCH($1, $2, 10) WHERE id = $3";
        let (bindings, _normalized) = PreparedStatement::extract_parameters(sql)
            .expect("parameter extraction should succeed for valid SQL");

        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0].index, 1);
        assert_eq!(bindings[1].index, 2);
        assert_eq!(bindings[2].index, 3);
    }

    #[test]
    fn test_parameter_extraction_duplicates() {
        let sql = "SELECT * FROM t WHERE a = $1 AND b = $1 AND c = $2";
        let (bindings, _normalized) = PreparedStatement::extract_parameters(sql)
            .expect("parameter extraction should succeed for valid SQL with duplicates");

        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].index, 1);
        assert_eq!(bindings[1].index, 2);
    }

    #[test]
    fn test_parameter_extraction_in_string() {
        let sql = "SELECT * FROM t WHERE name = 'test$1' AND id = $1";
        let (bindings, _normalized) = PreparedStatement::extract_parameters(sql)
            .expect("parameter extraction should succeed for valid SQL with string literals");

        // Should only find the $1 outside the string
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].index, 1);
    }

    #[test]
    fn test_parameter_gap_error() {
        let sql = "SELECT * FROM t WHERE a = $1 AND b = $3";
        let result = PreparedStatement::extract_parameters(sql);

        assert!(result.is_err());
        match result {
            Err(PreparedStatementError::InvalidParameter(msg)) => {
                assert!(msg.contains("Missing parameter $2"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_parameter_value_to_sql() {
        assert_eq!(
            ParameterValue::String("test".into()).to_sql_string(),
            "'test'"
        );
        assert_eq!(ParameterValue::Int(42).to_sql_string(), "42");
        assert_eq!(ParameterValue::Float(3.5).to_sql_string(), "3.5");
        assert_eq!(ParameterValue::Bool(true).to_sql_string(), "true");
        assert_eq!(ParameterValue::Null.to_sql_string(), "NULL");
        assert_eq!(
            ParameterValue::Vector(vec![0.1, 0.2]).to_sql_string(),
            "'[0.1,0.2]'"
        );
    }

    #[test]
    fn test_cache_prepare_and_get() {
        let cache = PreparedStatementCache::with_defaults();

        let id = cache
            .prepare("SELECT * FROM test WHERE id = $1")
            .expect("prepare should succeed for valid SQL");
        assert!(cache.exists(&id));

        let statement = cache
            .get(&id)
            .expect("get should succeed for existing statement");
        assert_eq!(statement.parameter_count(), 1);
    }

    #[test]
    fn test_cache_execute_sql() {
        let cache = PreparedStatementCache::with_defaults();

        let id = cache
            .prepare("SELECT * FROM test WHERE id = $1 AND name = $2")
            .expect("prepare should succeed for valid SQL");

        let sql = cache
            .execute_sql(
                &id,
                &[
                    ParameterValue::Int(42),
                    ParameterValue::String("test".into()),
                ],
            )
            .expect("execute_sql should succeed with valid parameters");

        assert!(sql.contains("42"));
        assert!(sql.contains("'test'"));
    }

    #[test]
    fn test_cache_parameter_mismatch() {
        let cache = PreparedStatementCache::with_defaults();

        let id = cache
            .prepare("SELECT * FROM test WHERE id = $1 AND name = $2")
            .expect("prepare should succeed for valid SQL");

        let result = cache.execute_sql(&id, &[ParameterValue::Int(42)]);

        assert!(result.is_err());
        match result {
            Err(PreparedStatementError::ParameterCountMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected ParameterCountMismatch error"),
        }
    }

    #[test]
    fn test_cache_drop_statement() {
        let cache = PreparedStatementCache::with_defaults();

        let id = cache
            .prepare("SELECT * FROM test")
            .expect("prepare should succeed for valid SQL");
        assert!(cache.exists(&id));

        cache
            .drop_statement(&id)
            .expect("drop_statement should succeed for existing statement");
        assert!(!cache.exists(&id));
    }

    #[test]
    fn test_cache_not_found() {
        let cache = PreparedStatementCache::with_defaults();

        let result = cache.get("nonexistent");
        assert!(matches!(result, Err(PreparedStatementError::NotFound(_))));
    }

    #[test]
    fn test_cache_stats() {
        let cache = PreparedStatementCache::with_defaults();

        cache
            .prepare("SELECT 1")
            .expect("prepare should succeed for valid SQL");
        cache
            .prepare("SELECT 2")
            .expect("prepare should succeed for valid SQL");

        let stats = cache.stats();
        assert_eq!(stats.cached_statements, 2);
    }

    #[test]
    fn test_cache_clear() {
        let cache = PreparedStatementCache::with_defaults();

        cache
            .prepare("SELECT 1")
            .expect("prepare should succeed for valid SQL");
        cache
            .prepare("SELECT 2")
            .expect("prepare should succeed for valid SQL");
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_ttl_expiration() {
        let config = PreparedStatementConfig {
            default_ttl: Duration::from_millis(1),
            ..Default::default()
        };
        let cache = PreparedStatementCache::new(config);

        let id = cache
            .prepare("SELECT 1")
            .expect("prepare should succeed for valid SQL");

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        let result = cache.get(&id);
        assert!(matches!(result, Err(PreparedStatementError::Expired(_))));
    }
}
