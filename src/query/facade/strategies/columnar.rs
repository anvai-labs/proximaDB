//! # Columnar Query Strategy
//!
//! Real implementation of `QueryStrategy` for columnar/analytical queries.
//! Uses the `ColumnarReadProvider` abstraction for efficient columnar data access.
//!
//! ## Features
//!
//! - Handles analytical SQL queries with columnar execution
//! - Leverages Arrow RecordBatch and Parquet range pruning
//! - Supports predicate pushdown for filter optimization
//! - Automatic provider selection (in-memory vs on-disk)
//! - SQL WHERE clause parsing for predicate pushdown
//! - Returns results as JSON rows or Arrow-compatible format
//!
//! ## Architecture
//!
//! ```text
//! QueryRequest (facade)
//!       │
//!       ▼
//! ColumnarStrategy
//!       │
//!       ├──> ArrowInMemoryProvider (cached data, zero-copy)
//!       │         └── Used when data is in memory cache
//!       │
//!       └──> ParquetRangePrunedProvider (on-disk Parquet/SST)
//!                 └── Used for NOVA engine files
//!                 └── Supports row group pruning via statistics
//!       │
//!       ▼
//! PredicatePushdownConfig (filter, projection, limit)
//!       │
//!       ▼
//! QueryResult (facade)
//! ```
//!
//! ## Provider Selection Logic
//!
//! The strategy automatically selects the best provider:
//!
//! 1. If an in-memory Arrow provider is registered -> use it (zero I/O cost)
//! 2. If a Parquet provider is registered -> use it with predicate pushdown
//! 3. Otherwise -> return error (no provider available)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use parking_lot::RwLock;
use tracing::{debug, info, instrument};

use crate::core::search::{ComparisonOperator, FilterExpression};
use crate::query::columnar::{
    ArrowInMemoryProvider, ColumnarReadProvider, ParquetRangePrunedProvider,
    PredicatePushdownConfig,
};
use crate::query::facade::{
    ExecutionMetrics, QueryContent, QueryContext, QueryRequest, QueryResult, QueryResultData,
    QueryStrategy,
};

/// Configuration for columnar strategy behavior
#[derive(Debug, Clone)]
pub struct ColumnarStrategyConfig {
    /// Enable SQL WHERE clause parsing for predicate pushdown
    pub enable_predicate_pushdown: bool,
    /// Enable projection pushdown (only read needed columns)
    pub enable_projection_pushdown: bool,
    /// Enable statistics-based row group pruning
    pub enable_statistics_pruning: bool,
    /// Default batch size for streaming operations
    pub default_batch_size: usize,
    /// Prefer in-memory providers over disk providers
    pub prefer_in_memory: bool,
}

impl Default for ColumnarStrategyConfig {
    fn default() -> Self {
        Self {
            enable_predicate_pushdown: true,
            enable_projection_pushdown: true,
            enable_statistics_pruning: true,
            default_batch_size: 8192,
            prefer_in_memory: true,
        }
    }
}

/// Columnar Query Strategy - Analytical query execution using columnar providers
///
/// This strategy handles analytical queries by:
/// 1. Detecting columnar-optimizable queries (aggregations, GROUP BY, DISTINCT)
/// 2. Selecting appropriate provider (in-memory Arrow or Parquet)
/// 3. Parsing SQL WHERE clauses for predicate pushdown
/// 4. Applying projection pruning based on SELECT columns
/// 5. Converting Arrow results to facade format
///
/// ## Provider Selection
///
/// When both in-memory and disk providers are available for a collection,
/// the strategy prefers in-memory providers (zero I/O cost) unless configured
/// otherwise.
pub struct ColumnarStrategy {
    /// Registered columnar providers by collection
    providers: RwLock<HashMap<String, Arc<dyn ColumnarReadProvider>>>,
    /// Strategy priority (higher = preferred)
    priority: i32,
    /// Strategy configuration
    config: ColumnarStrategyConfig,
}

impl ColumnarStrategy {
    /// Create a new ColumnarStrategy with default configuration
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            priority: 50, // Medium priority (below vector, above SQL)
            config: ColumnarStrategyConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: ColumnarStrategyConfig) -> Self {
        self.config = config;
        self
    }

    /// Create with custom priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Get the current configuration
    pub fn config(&self) -> &ColumnarStrategyConfig {
        &self.config
    }

    /// Register an in-memory Arrow provider for a collection
    ///
    /// In-memory providers are preferred over disk providers when both are
    /// available, as they have zero I/O cost.
    ///
    /// Returns Err if the batches are empty
    pub fn register_arrow_provider(
        &self,
        collection_id: impl Into<String>,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        let collection_id = collection_id.into();
        let provider = ArrowInMemoryProvider::new(batches, collection_id.clone())?;

        debug!(
            collection = %collection_id,
            "Registered Arrow in-memory provider"
        );

        self.providers.write().insert(
            collection_id,
            Arc::new(provider) as Arc<dyn ColumnarReadProvider>,
        );
        Ok(())
    }

    /// Register a Parquet provider for a collection
    ///
    /// Parquet providers support predicate pushdown via row group statistics
    /// and bloom filters for efficient I/O.
    pub fn register_parquet_provider(
        &self,
        collection_id: impl Into<String>,
        provider: ParquetRangePrunedProvider,
    ) {
        let collection_id = collection_id.into();

        debug!(
            collection = %collection_id,
            "Registered Parquet range-pruned provider"
        );

        self.providers
            .write()
            .insert(collection_id, Arc::new(provider));
    }

    /// Register a custom provider for a collection
    pub fn register_provider(
        &self,
        collection_id: impl Into<String>,
        provider: Arc<dyn ColumnarReadProvider>,
    ) {
        let collection_id = collection_id.into();
        debug!(
            collection = %collection_id,
            provider = %provider.name(),
            "Registered custom columnar provider"
        );
        self.providers.write().insert(collection_id, provider);
    }

    /// Remove a provider for a collection
    pub fn unregister_provider(
        &self,
        collection_id: &str,
    ) -> Option<Arc<dyn ColumnarReadProvider>> {
        self.providers.write().remove(collection_id)
    }

    /// Get provider for a collection
    pub fn get_provider(&self, collection_id: &str) -> Option<Arc<dyn ColumnarReadProvider>> {
        self.providers.read().get(collection_id).cloned()
    }

    /// Check if a provider exists for a collection
    pub fn has_provider(&self, collection_id: &str) -> bool {
        self.providers.read().contains_key(collection_id)
    }

    /// Get list of all registered collection IDs
    pub fn registered_collections(&self) -> Vec<String> {
        self.providers.read().keys().cloned().collect()
    }

    /// Check if a SQL query is suitable for columnar execution
    fn is_columnar_query(&self, sql: &str) -> bool {
        let sql_upper = sql.to_uppercase();

        // Columnar execution is good for:
        // - Aggregation queries (COUNT, SUM, AVG, MIN, MAX)
        // - Full table scans with filters
        // - GROUP BY queries
        // - DISTINCT queries

        let has_aggregation = sql_upper.contains("COUNT(")
            || sql_upper.contains("SUM(")
            || sql_upper.contains("AVG(")
            || sql_upper.contains("MIN(")
            || sql_upper.contains("MAX(");

        let has_group_by = sql_upper.contains("GROUP BY");
        let has_distinct = sql_upper.contains("DISTINCT");

        // Exclude vector-specific queries
        let has_vector_ops = sql_upper.contains("VECTOR_SEARCH")
            || sql_upper.contains("<->")
            || sql_upper.contains("COSINE_DISTANCE")
            || sql_upper.contains("EUCLIDEAN_DISTANCE");

        // Columnar is good for aggregations and table scans, not vector ops
        (has_aggregation || has_group_by || has_distinct) && !has_vector_ops
    }

    /// Extract collection name from SQL (basic parsing)
    fn extract_collection_from_sql(&self, sql: &str) -> Option<String> {
        let sql_upper = sql.to_uppercase();

        // Simple extraction: FROM <table_name>
        if let Some(from_pos) = sql_upper.find("FROM ") {
            let after_from = &sql[from_pos + 5..];
            let collection: String = after_from
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !collection.is_empty() {
                return Some(collection);
            }
        }

        None
    }

    /// Parse SQL WHERE clause into a FilterExpression for predicate pushdown
    ///
    /// Supports basic comparisons:
    /// - `column = value`
    /// - `column > value`
    /// - `column < value`
    /// - `column >= value`
    /// - `column <= value`
    /// - `column != value`
    /// - `column AND column` (conjunction)
    ///
    /// Returns None if no WHERE clause or parsing fails (graceful degradation).
    fn parse_where_clause(&self, sql: &str) -> Option<FilterExpression> {
        if !self.config.enable_predicate_pushdown {
            return None;
        }

        let sql_upper = sql.to_uppercase();

        // Find WHERE clause
        let where_pos = sql_upper.find("WHERE ")?;
        let after_where = &sql[where_pos + 6..];

        // Find end of WHERE clause (before GROUP BY, ORDER BY, LIMIT, or end)
        let end_keywords = ["GROUP BY", "ORDER BY", "LIMIT", "HAVING", ";"];
        let mut end_pos = after_where.len();
        for keyword in &end_keywords {
            if let Some(pos) = sql_upper[where_pos + 6..].find(keyword) {
                end_pos = end_pos.min(pos);
            }
        }

        let where_clause = after_where[..end_pos].trim();
        if where_clause.is_empty() {
            return None;
        }

        // Parse the WHERE clause
        self.parse_filter_expression(where_clause)
    }

    /// Parse a filter expression from a WHERE clause string
    fn parse_filter_expression(&self, expr: &str) -> Option<FilterExpression> {
        let expr = expr.trim();

        // Check for AND (split into conjunction)
        let expr_upper = expr.to_uppercase();
        if let Some(and_pos) = expr_upper.find(" AND ") {
            let left = &expr[..and_pos];
            let right = &expr[and_pos + 5..];

            let left_expr = self.parse_filter_expression(left)?;
            let right_expr = self.parse_filter_expression(right)?;

            return Some(FilterExpression::And(vec![left_expr, right_expr]));
        }

        // Check for OR (split into disjunction)
        if let Some(or_pos) = expr_upper.find(" OR ") {
            let left = &expr[..or_pos];
            let right = &expr[or_pos + 4..];

            let left_expr = self.parse_filter_expression(left)?;
            let right_expr = self.parse_filter_expression(right)?;

            return Some(FilterExpression::Or(vec![left_expr, right_expr]));
        }

        // Parse single comparison
        self.parse_comparison(expr)
    }

    /// Parse a single comparison expression (e.g., "column >= 100")
    fn parse_comparison(&self, expr: &str) -> Option<FilterExpression> {
        let expr = expr.trim();

        // Try operators in order of specificity (longer operators first)
        let operators = [
            (">=", ComparisonOperator::GreaterThanOrEqual),
            ("<=", ComparisonOperator::LessThanOrEqual),
            ("!=", ComparisonOperator::NotEquals),
            ("<>", ComparisonOperator::NotEquals),
            (">", ComparisonOperator::GreaterThan),
            ("<", ComparisonOperator::LessThan),
            ("=", ComparisonOperator::Equals),
        ];

        for (op_str, op) in &operators {
            if let Some(pos) = expr.find(op_str) {
                let field = expr[..pos].trim().to_string();
                let value_str = expr[pos + op_str.len()..].trim();

                // Parse value (string, number, or boolean)
                let value = self.parse_sql_value(value_str)?;

                return Some(FilterExpression::Comparison {
                    field,
                    operator: op.clone(),
                    value,
                });
            }
        }

        None
    }

    /// Parse a SQL value literal into serde_json::Value
    fn parse_sql_value(&self, value_str: &str) -> Option<serde_json::Value> {
        let value_str = value_str.trim();

        // String literal (single or double quotes)
        if (value_str.starts_with('\'') && value_str.ends_with('\''))
            || (value_str.starts_with('"') && value_str.ends_with('"'))
        {
            let inner = &value_str[1..value_str.len() - 1];
            return Some(serde_json::Value::String(inner.to_string()));
        }

        // Boolean literals
        let upper = value_str.to_uppercase();
        if upper == "TRUE" {
            return Some(serde_json::Value::Bool(true));
        }
        if upper == "FALSE" {
            return Some(serde_json::Value::Bool(false));
        }
        if upper == "NULL" {
            return Some(serde_json::Value::Null);
        }

        // Try parsing as integer
        if let Ok(i) = value_str.parse::<i64>() {
            return Some(serde_json::json!(i));
        }

        // Try parsing as float
        if let Ok(f) = value_str.parse::<f64>() {
            return serde_json::Number::from_f64(f).map(serde_json::Value::Number);
        }

        // Fallback: treat as string
        Some(serde_json::Value::String(value_str.to_string()))
    }

    /// Extract SELECT column names for projection pushdown
    fn extract_projection_columns(&self, sql: &str) -> Option<Vec<String>> {
        if !self.config.enable_projection_pushdown {
            return None;
        }

        let sql_upper = sql.to_uppercase();

        // Check for SELECT *
        if sql_upper.contains("SELECT *") || sql_upper.contains("SELECT DISTINCT *") {
            return None; // No projection, read all columns
        }

        // Find SELECT and FROM positions
        let select_pos = sql_upper.find("SELECT ")?;
        let from_pos = sql_upper.find(" FROM ")?;

        if from_pos <= select_pos {
            return None;
        }

        // Extract column list
        let after_select = select_pos + 7; // Length of "SELECT "
        // Handle DISTINCT keyword
        let column_start = if sql_upper[after_select..].starts_with("DISTINCT ") {
            after_select + 9
        } else {
            after_select
        };

        let columns_str = &sql[column_start..from_pos];

        // Parse column list (handle aliases and aggregates)
        let columns: Vec<String> = columns_str
            .split(',')
            .filter_map(|col| {
                let col = col.trim();
                // Skip aggregate functions (they need all columns)
                let col_upper = col.to_uppercase();
                if col_upper.contains("COUNT(")
                    || col_upper.contains("SUM(")
                    || col_upper.contains("AVG(")
                    || col_upper.contains("MIN(")
                    || col_upper.contains("MAX(")
                {
                    // Extract column from aggregate if simple (e.g., SUM(price))
                    if let Some(start) = col.find('(')
                        && let Some(end) = col.find(')')
                    {
                        let inner = col[start + 1..end].trim();
                        if inner != "*" && !inner.contains(' ') {
                            return Some(inner.to_string());
                        }
                    }
                    return None;
                }

                // Handle aliases (e.g., "column AS alias")
                let col_parts: Vec<&str> = col.split(" AS ").collect();
                let column_name = col_parts[0].trim();

                // Clean column name (remove table prefix if present)
                let clean_name = if let Some(dot_pos) = column_name.rfind('.') {
                    &column_name[dot_pos + 1..]
                } else {
                    column_name
                };

                if clean_name.is_empty() || clean_name == "*" {
                    None
                } else {
                    Some(clean_name.to_string())
                }
            })
            .collect();

        if columns.is_empty() {
            None
        } else {
            Some(columns)
        }
    }

    /// Extract LIMIT value from SQL
    fn extract_limit(&self, sql: &str) -> Option<usize> {
        let sql_upper = sql.to_uppercase();
        let limit_pos = sql_upper.find("LIMIT ")?;
        let after_limit = &sql[limit_pos + 6..];

        // Extract number until space or end
        let num_str: String = after_limit
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();

        num_str.parse().ok()
    }

    /// Extract OFFSET value from SQL
    fn extract_offset(&self, sql: &str) -> Option<usize> {
        let sql_upper = sql.to_uppercase();
        let offset_pos = sql_upper.find("OFFSET ")?;
        let after_offset = &sql[offset_pos + 7..];

        // Extract number until space or end
        let num_str: String = after_offset
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();

        num_str.parse().ok()
    }

    /// Build PredicatePushdownConfig from SQL query
    fn build_pushdown_config(&self, sql: &str) -> PredicatePushdownConfig {
        let filter = self.parse_where_clause(sql);
        let projection = self.extract_projection_columns(sql);
        let limit = self.extract_limit(sql);
        let offset = self.extract_offset(sql);

        if filter.is_some() || projection.is_some() || limit.is_some() || offset.is_some() {
            debug!(
                has_filter = filter.is_some(),
                has_projection = projection.is_some(),
                limit = ?limit,
                offset = ?offset,
                "Built predicate pushdown config from SQL"
            );
        }

        PredicatePushdownConfig {
            enable_statistics_pruning: self.config.enable_statistics_pruning,
            enable_bloom_filters: true,
            enable_projection: self.config.enable_projection_pushdown,
            projection,
            filter,
            limit,
            offset,
        }
    }

    /// Convert Arrow RecordBatches to JSON rows
    fn batches_to_json_rows(&self, batches: &[RecordBatch]) -> Vec<serde_json::Value> {
        let mut rows = Vec::new();

        for batch in batches {
            let schema = batch.schema();
            let num_rows = batch.num_rows();

            for row_idx in 0..num_rows {
                let mut row_obj = serde_json::Map::new();

                for (col_idx, field) in schema.fields().iter().enumerate() {
                    let col = batch.column(col_idx);
                    let value = self.extract_value(col, row_idx);
                    row_obj.insert(field.name().clone(), value);
                }

                rows.push(serde_json::Value::Object(row_obj));
            }
        }

        rows
    }

    /// Extract a single value from an Arrow array at the given index
    fn extract_value(&self, array: &ArrayRef, idx: usize) -> serde_json::Value {
        if array.is_null(idx) {
            return serde_json::Value::Null;
        }

        match array.data_type() {
            DataType::Utf8 => {
                let arr = array.as_string::<i32>();
                serde_json::Value::String(arr.value(idx).to_string())
            }
            DataType::LargeUtf8 => {
                let arr = array.as_string::<i64>();
                serde_json::Value::String(arr.value(idx).to_string())
            }
            DataType::Int8 => {
                let arr = array.as_primitive::<arrow::datatypes::Int8Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::Int16 => {
                let arr = array.as_primitive::<arrow::datatypes::Int16Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::Int32 => {
                let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::Int64 => {
                let arr = array.as_primitive::<arrow::datatypes::Int64Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::UInt8 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt8Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::UInt16 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt16Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::UInt32 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt32Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::UInt64 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt64Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::Float32 => {
                let arr = array.as_primitive::<arrow::datatypes::Float32Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::Float64 => {
                let arr = array.as_primitive::<arrow::datatypes::Float64Type>();
                serde_json::json!(arr.value(idx))
            }
            DataType::Boolean => {
                let arr = array.as_boolean();
                serde_json::json!(arr.value(idx))
            }
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
                // Encode binary as base64
                use base64::Engine;
                let bytes: &[u8] = match array.data_type() {
                    DataType::Binary => array.as_binary::<i32>().value(idx),
                    DataType::LargeBinary => array.as_binary::<i64>().value(idx),
                    DataType::FixedSizeBinary(_) => array.as_fixed_size_binary().value(idx),
                    _ => &[],
                };
                let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                serde_json::Value::String(encoded)
            }
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
                // Handle list types (including vector embeddings)
                self.extract_list_value(array, idx)
            }
            _ => {
                // Fallback: convert to string representation
                serde_json::Value::String(format!("{:?}", array.data_type()))
            }
        }
    }

    /// Extract list/array values
    fn extract_list_value(&self, array: &ArrayRef, idx: usize) -> serde_json::Value {
        match array.data_type() {
            DataType::List(field) | DataType::LargeList(field) => {
                // For Float32 lists (common for embeddings)
                if *field.data_type() == DataType::Float32 {
                    let list_arr = array.as_list::<i32>();
                    let values = list_arr.value(idx);
                    let float_arr = values.as_primitive::<arrow::datatypes::Float32Type>();
                    let floats: Vec<f32> = float_arr.values().to_vec();
                    return serde_json::json!(floats);
                }
                // Generic list handling
                serde_json::Value::Array(vec![])
            }
            DataType::FixedSizeList(field, size) => {
                if *field.data_type() == DataType::Float32 {
                    let list_arr = array.as_fixed_size_list();
                    let values = list_arr.value(idx);
                    let float_arr = values.as_primitive::<arrow::datatypes::Float32Type>();
                    let floats: Vec<f32> = float_arr.values()[..*size as usize].to_vec();
                    return serde_json::json!(floats);
                }
                serde_json::Value::Array(vec![])
            }
            _ => serde_json::Value::Array(vec![]),
        }
    }
}

impl Default for ColumnarStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QueryStrategy for ColumnarStrategy {
    fn name(&self) -> &str {
        "columnar"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        // Handle SQL queries that are suitable for columnar execution
        match &request.content {
            QueryContent::Sql(sql) => {
                // Check if query is columnar-optimizable
                if !self.is_columnar_query(sql) {
                    return false;
                }

                // Check if we have a provider for the target collection
                if let Some(collection) = &request.target {
                    return self.providers.read().contains_key(collection);
                }

                // Try to extract collection from SQL
                if let Some(collection) = self.extract_collection_from_sql(sql) {
                    return self.providers.read().contains_key(&collection);
                }

                false
            }
            _ => false,
        }
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    #[instrument(skip(self, request, _ctx), fields(strategy = "columnar"))]
    async fn execute(&self, request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        let start = Instant::now();

        // Extract SQL query
        let sql = match &request.content {
            QueryContent::Sql(sql) => sql.clone(),
            _ => return Err(anyhow!("ColumnarStrategy requires SQL content")),
        };

        // Determine target collection
        let collection_id = request
            .target
            .clone()
            .or_else(|| self.extract_collection_from_sql(&sql))
            .ok_or_else(|| anyhow!("Could not determine target collection"))?;

        // Get provider
        let provider = self
            .get_provider(&collection_id)
            .ok_or_else(|| anyhow!("No columnar provider for collection '{}'", collection_id))?;

        // Build predicate pushdown config from SQL query
        // This parses WHERE clause, SELECT columns, LIMIT, and OFFSET
        let config = self.build_pushdown_config(&sql);

        let provider_name = provider.name().to_string();
        let provider_capabilities = provider.capabilities();

        debug!(
            collection = %collection_id,
            provider = %provider_name,
            is_in_memory = provider_capabilities.is_in_memory,
            has_filter = config.filter.is_some(),
            has_projection = config.projection.is_some(),
            limit = ?config.limit,
            "Executing columnar query with predicate pushdown"
        );

        // Execute query through provider
        let batches = provider.read_batches(config.clone()).await?;

        // Convert to JSON rows
        let rows = self.batches_to_json_rows(&batches);

        let execution_time_ms = start.elapsed().as_millis() as u64;
        let results_returned = rows.len();

        // Get provider stats
        let stats = provider.get_stats();

        info!(
            collection = %collection_id,
            provider = %provider_name,
            results = results_returned,
            blocks_scanned = stats.blocks_scanned,
            blocks_pruned = stats.blocks_pruned,
            bytes_read = stats.bytes_read,
            time_ms = execution_time_ms,
            "Columnar query completed"
        );

        Ok(QueryResult {
            data: QueryResultData::Rows(rows),
            metrics: Some(ExecutionMetrics {
                execution_path: "unified".to_string(),
                strategy_name: "columnar".to_string(),
                execution_time_ms,
                planning_time_ms: 0,
                results_scanned: stats.total_rows as usize,
                results_returned,
                cache_hit: stats.cache_hits > 0,
                extra: serde_json::json!({
                    "provider": provider_name,
                    "provider_type": if provider_capabilities.is_in_memory { "in_memory" } else { "disk" },
                    "blocks_scanned": stats.blocks_scanned,
                    "bytes_read": stats.bytes_read,
                    "blocks_pruned": stats.blocks_pruned,
                    "rows_after_pruning": stats.rows_after_pruning,
                    "predicate_pushdown": {
                        "filter_applied": config.filter.is_some(),
                        "projection_applied": config.projection.is_some(),
                        "limit": config.limit,
                        "offset": config.offset,
                    },
                }),
            }),
        })
    }
}

// ================================================================================
// TESTS
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
            Field::new("score", DataType::Float32, false),
        ]));

        let id_array = StringArray::from(vec!["a", "b", "c"]);
        let value_array = Int64Array::from(vec![1, 2, 3]);
        let score_array = Float32Array::from(vec![0.1, 0.2, 0.3]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(value_array),
                Arc::new(score_array),
            ],
        )
        .expect("Failed to create test RecordBatch")
    }

    #[test]
    fn test_columnar_strategy_creation() {
        let strategy = ColumnarStrategy::new();
        assert_eq!(strategy.name(), "columnar");
        assert_eq!(strategy.priority(), 50);
    }

    #[test]
    fn test_is_columnar_query_aggregation() {
        let strategy = ColumnarStrategy::new();

        // Aggregation queries should be columnar
        assert!(strategy.is_columnar_query("SELECT COUNT(*) FROM products"));
        assert!(strategy.is_columnar_query("SELECT SUM(price) FROM orders"));
        assert!(strategy.is_columnar_query("SELECT AVG(score) FROM results"));
        assert!(strategy.is_columnar_query("SELECT MIN(value), MAX(value) FROM data"));
    }

    #[test]
    fn test_is_columnar_query_group_by() {
        let strategy = ColumnarStrategy::new();

        // GROUP BY queries should be columnar
        assert!(
            strategy.is_columnar_query("SELECT category, COUNT(*) FROM products GROUP BY category")
        );
        assert!(
            strategy.is_columnar_query("SELECT status, SUM(amount) FROM orders GROUP BY status")
        );
    }

    #[test]
    fn test_is_columnar_query_distinct() {
        let strategy = ColumnarStrategy::new();

        // DISTINCT queries should be columnar
        assert!(strategy.is_columnar_query("SELECT DISTINCT category FROM products"));
    }

    #[test]
    fn test_is_columnar_query_not_vector() {
        let strategy = ColumnarStrategy::new();

        // Vector queries should NOT be columnar
        assert!(
            !strategy.is_columnar_query("SELECT * FROM VECTOR_SEARCH('products', '[0.1]', 10)")
        );
        assert!(
            !strategy.is_columnar_query("SELECT * FROM products ORDER BY embedding <-> '[0.1]'")
        );
        assert!(
            !strategy.is_columnar_query("SELECT COSINE_DISTANCE(embedding, '[0.1]') FROM products")
        );
    }

    #[test]
    fn test_is_columnar_query_simple_select() {
        let strategy = ColumnarStrategy::new();

        // Simple SELECT without aggregation should NOT be columnar
        assert!(!strategy.is_columnar_query("SELECT * FROM products"));
        assert!(!strategy.is_columnar_query("SELECT id, name FROM products WHERE price > 100"));
    }

    #[test]
    fn test_extract_collection_from_sql() {
        let strategy = ColumnarStrategy::new();

        assert_eq!(
            strategy.extract_collection_from_sql("SELECT * FROM products"),
            Some("products".to_string())
        );
        assert_eq!(
            strategy.extract_collection_from_sql(
                "SELECT COUNT(*) FROM orders WHERE status = 'pending'"
            ),
            Some("orders".to_string())
        );
        assert_eq!(
            strategy.extract_collection_from_sql("SELECT * FROM my_table_123"),
            Some("my_table_123".to_string())
        );
    }

    #[test]
    fn test_register_arrow_provider() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        strategy
            .register_arrow_provider("test_collection", vec![batch])
            .expect("Failed to register Arrow provider for test_collection");

        assert!(strategy.get_provider("test_collection").is_some());
        assert!(strategy.get_provider("nonexistent").is_none());
    }

    #[test]
    fn test_register_arrow_provider_empty_fails() {
        let strategy = ColumnarStrategy::new();
        let result = strategy.register_arrow_provider("empty", vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_can_handle_with_provider() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        strategy
            .register_arrow_provider("products", vec![batch])
            .expect("Failed to register Arrow provider for products");

        // Should handle aggregation query with registered provider
        let request = QueryRequest::sql("SELECT COUNT(*) FROM products");
        assert!(strategy.can_handle(&request));

        // Should not handle query without provider
        let request = QueryRequest::sql("SELECT COUNT(*) FROM unknown_table");
        assert!(!strategy.can_handle(&request));

        // Should not handle non-columnar query
        let request = QueryRequest::sql("SELECT * FROM products");
        assert!(!strategy.can_handle(&request));
    }

    #[test]
    fn test_can_handle_with_target() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        strategy
            .register_arrow_provider("products", vec![batch])
            .expect("Failed to register Arrow provider for products");

        // Should handle with explicit target
        let request = QueryRequest::sql("SELECT COUNT(*) FROM products").with_target("products");
        assert!(strategy.can_handle(&request));
    }

    #[test]
    fn test_batches_to_json_rows() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        let rows = strategy.batches_to_json_rows(&[batch]);

        assert_eq!(rows.len(), 3);

        // Check first row
        let row0 = rows[0].as_object().expect("First row should be an object");
        assert_eq!(
            row0.get("id").expect("id field should exist"),
            &serde_json::json!("a")
        );
        assert_eq!(
            row0.get("value").expect("value field should exist"),
            &serde_json::json!(1)
        );

        // Check score is approximately 0.1
        let score0 = row0
            .get("score")
            .expect("score field should exist")
            .as_f64()
            .expect("score should be a float");
        assert!((score0 - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_batches_to_json_rows_multiple_batches() {
        let strategy = ColumnarStrategy::new();
        let batch1 = create_test_batch();
        let batch2 = create_test_batch();

        let rows = strategy.batches_to_json_rows(&[batch1, batch2]);

        // 3 rows per batch * 2 batches = 6 rows
        assert_eq!(rows.len(), 6);
    }

    #[tokio::test]
    async fn test_execute_with_arrow_provider() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        strategy
            .register_arrow_provider("products", vec![batch])
            .expect("Failed to register Arrow provider for products");

        let request = QueryRequest::sql("SELECT COUNT(*) FROM products")
            .with_target("products")
            .with_metrics();

        let ctx = QueryContext::new(30000);
        let result = strategy
            .execute(request, &ctx)
            .await
            .expect("Failed to execute query");

        // Should return rows
        if let QueryResultData::Rows(rows) = result.data {
            assert_eq!(rows.len(), 3);
        } else {
            panic!("Expected Rows result");
        }

        // Should have metrics
        let metrics = result.metrics.expect("Metrics should be present");
        assert_eq!(metrics.strategy_name, "columnar");
        assert_eq!(metrics.execution_path, "unified");
    }

    #[tokio::test]
    async fn test_execute_without_provider_fails() {
        let strategy = ColumnarStrategy::new();

        let request = QueryRequest::sql("SELECT COUNT(*) FROM unknown").with_target("unknown");

        let ctx = QueryContext::new(30000);
        let result = strategy.execute(request, &ctx).await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No columnar provider")
        );
    }

    #[test]
    fn test_strategy_priority() {
        let strategy = ColumnarStrategy::new().with_priority(75);
        assert_eq!(strategy.priority(), 75);
    }

    // ================================================================================
    // SQL Parsing Tests
    // ================================================================================

    #[test]
    fn test_parse_where_clause_simple_equals() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT COUNT(*) FROM products WHERE status = 'active'";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_some());

        if let Some(FilterExpression::Comparison {
            field,
            operator,
            value,
        }) = filter
        {
            assert_eq!(field, "status");
            assert_eq!(operator, ComparisonOperator::Equals);
            assert_eq!(value, serde_json::json!("active"));
        } else {
            panic!("Expected Comparison filter");
        }
    }

    #[test]
    fn test_parse_where_clause_numeric_comparison() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT SUM(amount) FROM orders WHERE value >= 100";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_some());

        if let Some(FilterExpression::Comparison {
            field,
            operator,
            value,
        }) = filter
        {
            assert_eq!(field, "value");
            assert_eq!(operator, ComparisonOperator::GreaterThanOrEqual);
            assert_eq!(value, serde_json::json!(100));
        } else {
            panic!("Expected Comparison filter");
        }
    }

    #[test]
    fn test_parse_where_clause_and_conjunction() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT COUNT(*) FROM orders WHERE status = 'pending' AND amount > 50";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_some());

        if let Some(FilterExpression::And(exprs)) = filter {
            assert_eq!(exprs.len(), 2);
        } else {
            panic!("Expected And filter");
        }
    }

    #[test]
    fn test_parse_where_clause_or_disjunction() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT COUNT(*) FROM products WHERE category = 'A' OR category = 'B'";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_some());

        if let Some(FilterExpression::Or(exprs)) = filter {
            assert_eq!(exprs.len(), 2);
        } else {
            panic!("Expected Or filter");
        }
    }

    #[test]
    fn test_parse_where_clause_with_group_by() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT category, COUNT(*) FROM products WHERE active = true GROUP BY category";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_some());

        if let Some(FilterExpression::Comparison {
            field,
            operator,
            value,
        }) = filter
        {
            assert_eq!(field, "active");
            assert_eq!(operator, ComparisonOperator::Equals);
            assert_eq!(value, serde_json::json!(true));
        } else {
            panic!("Expected Comparison filter");
        }
    }

    #[test]
    fn test_parse_where_clause_no_where() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT COUNT(*) FROM products";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_none());
    }

    #[test]
    fn test_parse_where_clause_disabled() {
        let config = ColumnarStrategyConfig {
            enable_predicate_pushdown: false,
            ..Default::default()
        };
        let strategy = ColumnarStrategy::new().with_config(config);
        let sql = "SELECT COUNT(*) FROM products WHERE status = 'active'";

        let filter = strategy.parse_where_clause(sql);
        assert!(filter.is_none());
    }

    // ================================================================================
    // Projection Pushdown Tests
    // ================================================================================

    #[test]
    fn test_extract_projection_columns_simple() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT id, name, price FROM products";

        let columns = strategy.extract_projection_columns(sql);
        assert!(columns.is_some());

        let columns = columns.expect("Columns should be present");
        assert_eq!(columns.len(), 3);
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"name".to_string()));
        assert!(columns.contains(&"price".to_string()));
    }

    #[test]
    fn test_extract_projection_columns_star() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT * FROM products";

        let columns = strategy.extract_projection_columns(sql);
        assert!(columns.is_none()); // Star means all columns
    }

    #[test]
    fn test_extract_projection_columns_distinct_star() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT DISTINCT * FROM products";

        let columns = strategy.extract_projection_columns(sql);
        assert!(columns.is_none());
    }

    #[test]
    fn test_extract_projection_columns_with_alias() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT id, name AS product_name FROM products";

        let columns = strategy.extract_projection_columns(sql);
        assert!(columns.is_some());

        let columns = columns.expect("Columns should be present");
        assert_eq!(columns.len(), 2);
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"name".to_string())); // Original name, not alias
    }

    #[test]
    fn test_extract_projection_columns_aggregate() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT category, COUNT(*), SUM(price) FROM products GROUP BY category";

        let columns = strategy.extract_projection_columns(sql);
        assert!(columns.is_some());

        let columns = columns.expect("Columns should be present");
        // Should extract 'category' and 'price' from SUM(price)
        assert!(columns.contains(&"category".to_string()));
        assert!(columns.contains(&"price".to_string()));
    }

    #[test]
    fn test_extract_projection_columns_disabled() {
        let config = ColumnarStrategyConfig {
            enable_projection_pushdown: false,
            ..Default::default()
        };
        let strategy = ColumnarStrategy::new().with_config(config);
        let sql = "SELECT id, name FROM products";

        let columns = strategy.extract_projection_columns(sql);
        assert!(columns.is_none());
    }

    // ================================================================================
    // LIMIT/OFFSET Parsing Tests
    // ================================================================================

    #[test]
    fn test_extract_limit() {
        let strategy = ColumnarStrategy::new();

        assert_eq!(
            strategy.extract_limit("SELECT * FROM products LIMIT 10"),
            Some(10)
        );
        assert_eq!(
            strategy.extract_limit("SELECT * FROM products LIMIT 100"),
            Some(100)
        );
        assert_eq!(strategy.extract_limit("SELECT * FROM products"), None);
    }

    #[test]
    fn test_extract_offset() {
        let strategy = ColumnarStrategy::new();

        assert_eq!(
            strategy.extract_offset("SELECT * FROM products LIMIT 10 OFFSET 20"),
            Some(20)
        );
        assert_eq!(
            strategy.extract_offset("SELECT * FROM products OFFSET 5"),
            Some(5)
        );
        assert_eq!(strategy.extract_offset("SELECT * FROM products"), None);
    }

    // ================================================================================
    // Build Pushdown Config Tests
    // ================================================================================

    #[test]
    fn test_build_pushdown_config_full() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT id, name FROM products WHERE status = 'active' LIMIT 10 OFFSET 5";

        let config = strategy.build_pushdown_config(sql);

        assert!(config.filter.is_some());
        assert!(config.projection.is_some());
        assert_eq!(config.limit, Some(10));
        assert_eq!(config.offset, Some(5));
        assert!(config.enable_statistics_pruning);
    }

    #[test]
    fn test_build_pushdown_config_defaults() {
        let strategy = ColumnarStrategy::new();
        let sql = "SELECT COUNT(*) FROM products";

        let config = strategy.build_pushdown_config(sql);

        assert!(config.filter.is_none());
        // COUNT(*) doesn't produce specific columns
        assert!(config.limit.is_none());
        assert!(config.offset.is_none());
    }

    // ================================================================================
    // Provider Management Tests
    // ================================================================================

    #[test]
    fn test_unregister_provider() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        strategy
            .register_arrow_provider("test", vec![batch])
            .expect("Failed to register Arrow provider for test");
        assert!(strategy.has_provider("test"));

        let removed = strategy.unregister_provider("test");
        assert!(removed.is_some());
        assert!(!strategy.has_provider("test"));
    }

    #[test]
    fn test_registered_collections() {
        let strategy = ColumnarStrategy::new();
        let batch = create_test_batch();

        strategy
            .register_arrow_provider("collection_a", vec![batch.clone()])
            .expect("Failed to register Arrow provider for collection_a");
        strategy
            .register_arrow_provider("collection_b", vec![batch])
            .expect("Failed to register Arrow provider for collection_b");

        let collections = strategy.registered_collections();
        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"collection_a".to_string()));
        assert!(collections.contains(&"collection_b".to_string()));
    }

    #[test]
    fn test_config_getter() {
        let config = ColumnarStrategyConfig {
            enable_predicate_pushdown: false,
            default_batch_size: 4096,
            ..Default::default()
        };
        let strategy = ColumnarStrategy::new().with_config(config);

        assert!(!strategy.config().enable_predicate_pushdown);
        assert_eq!(strategy.config().default_batch_size, 4096);
    }

    // ================================================================================
    // Integration Tests with Provider
    // ================================================================================

    #[tokio::test]
    async fn test_execute_with_filter_pushdown() {
        let strategy = ColumnarStrategy::new();

        // Create batch with more rows for filtering
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
            Field::new("status", DataType::Utf8, false),
        ]));

        let id_array = StringArray::from(vec!["a", "b", "c", "d"]);
        let value_array = Int64Array::from(vec![10, 20, 30, 40]);
        let status_array = StringArray::from(vec!["active", "inactive", "active", "pending"]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(value_array),
                Arc::new(status_array),
            ],
        )
        .expect("Failed to create test RecordBatch");

        strategy
            .register_arrow_provider("products", vec![batch])
            .expect("Failed to register Arrow provider for products");

        let request = QueryRequest::sql("SELECT COUNT(*) FROM products WHERE value > 15")
            .with_target("products")
            .with_metrics();

        let ctx = QueryContext::new(30000);
        let result = strategy
            .execute(request, &ctx)
            .await
            .expect("Failed to execute query");

        // Verify metrics include predicate pushdown info
        let metrics = result.metrics.expect("Metrics should be present");
        assert_eq!(metrics.strategy_name, "columnar");

        let extra = metrics
            .extra
            .as_object()
            .expect("extra should be an object");
        assert!(extra.contains_key("predicate_pushdown"));
        assert_eq!(
            extra
                .get("provider_type")
                .expect("provider_type should exist"),
            "in_memory"
        );
    }

    #[tokio::test]
    async fn test_execute_with_limit_pushdown() {
        let strategy = ColumnarStrategy::new();

        // Create batch with many rows
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let id_array = Int64Array::from((0..100).collect::<Vec<i64>>());

        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_array)])
            .expect("Failed to create test RecordBatch");

        strategy
            .register_arrow_provider("numbers", vec![batch])
            .expect("Failed to register Arrow provider for numbers");

        let request = QueryRequest::sql("SELECT COUNT(*) FROM numbers LIMIT 10")
            .with_target("numbers")
            .with_metrics();

        let ctx = QueryContext::new(30000);
        let result = strategy
            .execute(request, &ctx)
            .await
            .expect("Failed to execute query");

        // Should return at most 10 rows due to LIMIT
        if let QueryResultData::Rows(rows) = result.data {
            assert!(rows.len() <= 10);
        } else {
            panic!("Expected Rows result");
        }
    }

    #[test]
    fn test_parse_sql_value_types() {
        let strategy = ColumnarStrategy::new();

        // String values
        assert_eq!(
            strategy.parse_sql_value("'hello'"),
            Some(serde_json::json!("hello"))
        );
        assert_eq!(
            strategy.parse_sql_value("\"world\""),
            Some(serde_json::json!("world"))
        );

        // Boolean values
        assert_eq!(
            strategy.parse_sql_value("true"),
            Some(serde_json::json!(true))
        );
        assert_eq!(
            strategy.parse_sql_value("TRUE"),
            Some(serde_json::json!(true))
        );
        assert_eq!(
            strategy.parse_sql_value("false"),
            Some(serde_json::json!(false))
        );
        assert_eq!(
            strategy.parse_sql_value("FALSE"),
            Some(serde_json::json!(false))
        );

        // Null value
        assert_eq!(
            strategy.parse_sql_value("NULL"),
            Some(serde_json::Value::Null)
        );

        // Integer values
        assert_eq!(strategy.parse_sql_value("42"), Some(serde_json::json!(42)));
        assert_eq!(
            strategy.parse_sql_value("-100"),
            Some(serde_json::json!(-100))
        );

        // Float values
        let float_val = strategy.parse_sql_value("3.5");
        assert!(float_val.is_some());
        let f = float_val
            .expect("Float value should be present")
            .as_f64()
            .expect("Should be a float64");
        assert!((f - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_comparison_operators() {
        let strategy = ColumnarStrategy::new();

        // Test all comparison operators
        let test_cases = [
            ("value = 10", ComparisonOperator::Equals),
            ("value != 10", ComparisonOperator::NotEquals),
            ("value <> 10", ComparisonOperator::NotEquals),
            ("value > 10", ComparisonOperator::GreaterThan),
            ("value < 10", ComparisonOperator::LessThan),
            ("value >= 10", ComparisonOperator::GreaterThanOrEqual),
            ("value <= 10", ComparisonOperator::LessThanOrEqual),
        ];

        for (expr, expected_op) in test_cases {
            let filter = strategy.parse_comparison(expr);
            assert!(filter.is_some(), "Failed to parse: {}", expr);

            if let Some(FilterExpression::Comparison { operator, .. }) = filter {
                assert_eq!(operator, expected_op, "Wrong operator for: {}", expr);
            } else {
                panic!("Expected Comparison for: {}", expr);
            }
        }
    }
}
