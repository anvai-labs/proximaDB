//! Root-crate implementation of `UnifiedQueryPort`.
//!
//! Wraps `QueryFacadeAdapter` (for query execution) and `PreparedStatementCache`
//! (for parse-once-execute-many) so `proximadb-api`'s multimodal REST handlers
//! can delegate to real business logic without importing root-crate concrete types.
//!
//! Phase 9.9: this impl unblocks all nine `/api/v1/unified/*` endpoints in
//! `crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_runtime::UnifiedQueryPort;
use tracing::{debug, info};

use crate::catalog::CatalogManager;
use crate::query::authority_context::{AuthoritySource, resolve_catalog_authority_context};
use crate::query::explain::StorageAuthorityExplanation;
use crate::query::multimodal::plan::PlanContext;
use crate::query::unified::uql::{
    ComparisonOperator, Condition, SelectStatement, UQLParser, UQLStatement, Value,
};
use crate::query::{
    ParameterValue, PreparedStatementCache, PreparedStatementConfig, PreparedStatementError,
    QueryFacadeAdapter,
};

// ── Conversion helpers ────────────────────────────────────────────────────────

fn proxima_value_to_param(value: &ProximaValue) -> ParameterValue {
    match value {
        ProximaValue::String(s) | ProximaValue::Symbol(s) => ParameterValue::String(s.clone()),
        ProximaValue::Int8(v) => ParameterValue::Int(*v as i64),
        ProximaValue::Int16(v) => ParameterValue::Int(*v as i64),
        ProximaValue::Int32(v) => ParameterValue::Int(*v as i64),
        ProximaValue::Int64(v) => ParameterValue::Int(*v),
        ProximaValue::UInt8(v) => ParameterValue::Int(*v as i64),
        ProximaValue::UInt16(v) => ParameterValue::Int(*v as i64),
        ProximaValue::UInt32(v) => ParameterValue::Int(*v as i64),
        ProximaValue::UInt64(v) => i64::try_from(*v)
            .map(ParameterValue::Int)
            .unwrap_or_else(|_| ParameterValue::String(v.to_string())),
        ProximaValue::Float16(v) | ProximaValue::Float32(v) => ParameterValue::Float(*v as f64),
        ProximaValue::Float64(v) => ParameterValue::Float(*v),
        ProximaValue::Boolean(v) => ParameterValue::Bool(*v),
        ProximaValue::DenseVector(values) => ParameterValue::Vector(values.clone()),
        ProximaValue::Json(value) | ProximaValue::Jsonb(value) => {
            ParameterValue::Json(value.clone())
        }
        ProximaValue::Array(values) => ParameterValue::Json(serde_json::Value::Array(
            values.iter().map(proxima_value_to_json).collect(),
        )),
        ProximaValue::Map(values) | ProximaValue::Struct(values) => {
            ParameterValue::Json(serde_json::Value::Object(
                values
                    .iter()
                    .map(|(key, value)| (key.clone(), proxima_value_to_json(value)))
                    .collect(),
            ))
        }
        ProximaValue::Null => ParameterValue::Null,
        // Typed exotics (Binary/Uuid/ULID/temporals/SparseVector/Decimal)
        // lower through the ONE shared filter spelling — the old Rust-Debug
        // strings ("Binary([1, 2, 3])") could never equal a stored value's
        // rendering, so such parameters silently matched nothing. Structured
        // values map to their JSON TEXT as a String param so the direct
        // path and the literal path render byte-identical SQL text
        // (to_sql_string quotes both String and Json params).
        other => match proxima_value_to_json(other) {
            serde_json::Value::String(s) => ParameterValue::String(s),
            json => ParameterValue::String(json.to_string()),
        },
    }
}

/// Thin alias over the ONE shared ProximaValue→JSON filter spelling
/// (search-types) — literals bound into federated SQL must render exactly
/// what the filter evaluator renders on the stored side, or equality
/// parameters silently match nothing.
fn proxima_value_to_json(value: &ProximaValue) -> serde_json::Value {
    proximadb_search_types::sql_value_filter::proxima_value_to_json(value)
}

fn proxima_values_to_params(values: Option<Vec<ProximaValue>>) -> Vec<ParameterValue> {
    values
        .unwrap_or_default()
        .iter()
        .map(proxima_value_to_param)
        .collect()
}

fn proxima_value_to_f32_vector(value: &ProximaValue) -> Option<Vec<f32>> {
    match value {
        ProximaValue::DenseVector(values) => Some(values.clone()),
        ProximaValue::Array(values) => values
            .iter()
            .map(|value| match value {
                ProximaValue::Float16(v) | ProximaValue::Float32(v) => Some(*v),
                ProximaValue::Float64(v) => Some(*v as f32),
                ProximaValue::Int8(v) => Some(*v as f32),
                ProximaValue::Int16(v) => Some(*v as f32),
                ProximaValue::Int32(v) => Some(*v as f32),
                ProximaValue::Int64(v) => Some(*v as f32),
                ProximaValue::UInt8(v) => Some(*v as f32),
                ProximaValue::UInt16(v) => Some(*v as f32),
                ProximaValue::UInt32(v) => Some(*v as f32),
                ProximaValue::UInt64(v) => Some(*v as f32),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn proxima_value_to_sql_literal(value: &ProximaValue) -> Result<String> {
    match value {
        ProximaValue::String(value) | ProximaValue::Symbol(value) => Ok(sql_quote(value)),
        ProximaValue::Int8(value) => Ok(value.to_string()),
        ProximaValue::Int16(value) => Ok(value.to_string()),
        ProximaValue::Int32(value) => Ok(value.to_string()),
        ProximaValue::Int64(value) => Ok(value.to_string()),
        ProximaValue::UInt8(value) => Ok(value.to_string()),
        ProximaValue::UInt16(value) => Ok(value.to_string()),
        ProximaValue::UInt32(value) => Ok(value.to_string()),
        ProximaValue::UInt64(value) => Ok(value.to_string()),
        // Non-finite floats have no SQL literal (bare 'NaN' text is a
        // parse error) — NULL, matching the canonical JSON rendering.
        ProximaValue::Float16(value) | ProximaValue::Float32(value) => Ok(if value.is_finite() {
            value.to_string()
        } else {
            "NULL".to_string()
        }),
        ProximaValue::Float64(value) => Ok(if value.is_finite() {
            value.to_string()
        } else {
            "NULL".to_string()
        }),
        ProximaValue::Boolean(value) => Ok(if *value { "TRUE" } else { "FALSE" }.to_string()),
        ProximaValue::DenseVector(values) => Ok(sql_quote(&serde_json::to_string(values)?)),
        ProximaValue::Array(values) => {
            if let Some(vector) = proxima_value_to_f32_vector(value) {
                Ok(sql_quote(&serde_json::to_string(&vector)?))
            } else {
                Ok(sql_quote(&serde_json::to_string(&proxima_value_to_json(
                    &ProximaValue::Array(values.clone()),
                ))?))
            }
        }
        ProximaValue::Json(value) | ProximaValue::Jsonb(value) => {
            Ok(sql_quote(&serde_json::to_string(value)?))
        }
        ProximaValue::Map(_) | ProximaValue::Struct(_) => Ok(sql_quote(&serde_json::to_string(
            &proxima_value_to_json(value),
        )?)),
        ProximaValue::Null => Ok("NULL".to_string()),
        // Typed exotics: the shared filter spelling (dashed Uuid, int-array
        // Binary, canonical sparse object) — the Debug fallback could never
        // equal a stored rendering. Strings come out BARE (double-quoting
        // the JSON text embedded the quotes in the literal itself). KNOWN
        // GAPS (tracked in TD-PROTO-2, dialect-dependent): temporals splice
        // as quoted epoch-numbers, and Binary/SparseVector splice as quoted
        // JSON text — parseable SQL on every engine, but equality against a
        // native binary/timestamp column needs the per-dialect literal form
        // (e.g. ISO-8601 text for Postgres-style engines).
        other => match proxima_value_to_json(other) {
            serde_json::Value::String(s) => Ok(sql_quote(&s)),
            json => Ok(sql_quote(&json.to_string())),
        },
    }
}

fn bind_federated_sql_parameters(query: &str, parameters: &[ProximaValue]) -> Result<String> {
    if parameters.is_empty() {
        return Ok(query.to_string());
    }

    let mut bound = String::with_capacity(query.len());
    let mut chars = query.chars().peekable();
    let mut in_single_quote = false;
    let mut param_index = 0usize;

    while let Some(ch) = chars.next() {
        match ch {
            '\'' => {
                bound.push(ch);
                if in_single_quote && chars.peek() == Some(&'\'') {
                    if let Some(escaped) = chars.next() {
                        bound.push(escaped);
                    }
                } else {
                    in_single_quote = !in_single_quote;
                }
            }
            '?' if !in_single_quote => {
                let value = parameters.get(param_index).ok_or_else(|| {
                    anyhow!(
                        "federated query has more placeholders than provided parameters: missing parameter {}",
                        param_index + 1
                    )
                })?;
                bound.push_str(&proxima_value_to_sql_literal(value)?);
                param_index += 1;
            }
            _ => bound.push(ch),
        }
    }

    if param_index != parameters.len() {
        return Err(anyhow!(
            "federated query received {} parameters but only used {} placeholders",
            parameters.len(),
            param_index
        ));
    }

    Ok(bound)
}

fn value_to_filter_literal(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(format!("\"{}\"", value.replace('"', "\\\""))),
        Value::Integer(value) => Some(value.to_string()),
        Value::Number(value) => Some(value.to_string()),
        Value::Boolean(value) => Some(value.to_string()),
        Value::Null => Some("null".to_string()),
        _ => None,
    }
}

fn comparison_operator_to_filter(operator: &ComparisonOperator) -> Option<&'static str> {
    match operator {
        ComparisonOperator::Eq => Some("="),
        ComparisonOperator::Ne => Some("!="),
        ComparisonOperator::Lt => Some("<"),
        ComparisonOperator::Lte => Some("<="),
        ComparisonOperator::Gt => Some(">"),
        ComparisonOperator::Gte => Some(">="),
        ComparisonOperator::Like | ComparisonOperator::Contains => Some("CONTAINS"),
        _ => None,
    }
}

fn document_filter_from_select(select: &SelectStatement) -> Result<String> {
    let Some(where_clause) = &select.where_clause else {
        return Ok(String::new());
    };

    if where_clause.logic != crate::query::unified::uql::LogicOperator::And {
        return Err(anyhow!(
            "UQL document lowering currently supports AND filters only"
        ));
    }

    let mut parts = Vec::new();
    for condition in &where_clause.conditions {
        match condition {
            Condition::JsonPath {
                path,
                operator,
                value,
            }
            | Condition::Comparison {
                field: path,
                operator,
                value,
            } => {
                let op = comparison_operator_to_filter(operator).ok_or_else(|| {
                    anyhow!(
                        "UQL document lowering does not support operator {:?}",
                        operator
                    )
                })?;
                let value = value_to_filter_literal(value).ok_or_else(|| {
                    anyhow!("UQL document lowering only supports scalar filter values")
                })?;
                let field = path.strip_prefix("$.").unwrap_or(path);
                parts.push(format!("{field} {op} {value}"));
            }
            other => {
                return Err(anyhow!(
                    "UQL document lowering does not support condition {:?}",
                    other
                ));
            }
        }
    }

    Ok(parts.join(" AND "))
}

fn uql_to_federated_sql(
    query: &str,
    parameters: &[ProximaValue],
    request_limit: Option<u32>,
) -> Result<Option<String>> {
    let mut parser = UQLParser::new();
    let statement = match parser.parse(query) {
        Ok(statement) => statement,
        Err(_) => return Ok(None),
    };

    let select = match statement {
        UQLStatement::Select(select) => select,
        UQLStatement::Explain(inner) => match *inner {
            UQLStatement::Select(select) => select,
            _ => {
                return Err(anyhow!(
                    "UQL EXPLAIN lowering currently supports SELECT statements only"
                ));
            }
        },
        UQLStatement::MultiModal(_) => {
            return Err(anyhow!(
                "UQL MULTIMODAL lowering is not yet wired to FederatedQueryContext"
            ));
        }
    };

    let limit = request_limit.or(select.limit).unwrap_or(10);
    match select.from.model {
        proximadb_data_model::DataModel::Vector => {
            let query_param = select
                .where_clause
                .as_ref()
                .and_then(|where_clause| {
                    where_clause
                        .conditions
                        .iter()
                        .find_map(|condition| match condition {
                            Condition::VectorSimilar { query_param, .. }
                            | Condition::VectorDistance { query_param, .. } => Some(*query_param),
                            _ => None,
                        })
                })
                .ok_or_else(|| {
                    anyhow!(
                        "UQL vector queries require VECTOR_SIMILAR(...) or VECTOR_DISTANCE(...)"
                    )
                })?;
            let vector = parameters
                .get(query_param)
                .and_then(proxima_value_to_f32_vector)
                .ok_or_else(|| {
                    anyhow!(
                        "UQL vector query parameter ${} must be a numeric vector",
                        query_param + 1
                    )
                })?;
            if vector.is_empty() {
                return Err(anyhow!("UQL vector query parameter cannot be empty"));
            }
            let vector_json = serde_json::to_string(&vector)?;
            Ok(Some(format!(
                "SELECT * FROM VECTOR_SEARCH({}, {}, {})",
                sql_quote(&select.from.collection),
                sql_quote(&vector_json),
                limit
            )))
        }
        proximadb_data_model::DataModel::Document => {
            let filter = document_filter_from_select(&select)?;
            Ok(Some(format!(
                "SELECT * FROM DOCUMENT_QUERY({}, {}) LIMIT {}",
                sql_quote(&select.from.collection),
                sql_quote(&filter),
                limit
            )))
        }
        proximadb_data_model::DataModel::Observability => Ok(Some(format!(
            "SELECT * FROM LOGS({}) LIMIT {}",
            sql_quote(&select.from.collection),
            limit
        ))),
        proximadb_data_model::DataModel::Graph => Err(anyhow!(
            "UQL graph SELECT lowering requires GRAPH_QUERY(...) support; use federated GRAPH_QUERY SQL for now"
        )),
        other => Err(anyhow!(
            "UQL lowering does not support data model {:?} through the federated executor",
            other
        )),
    }
}

// ── Port implementation ───────────────────────────────────────────────────────

/// Implementation of `UnifiedQueryPort` backed by root-crate services.
///
/// Created once at server startup and injected into `UnifiedQueryRestState`.
pub struct UnifiedQueryPortImpl {
    adapter: Arc<QueryFacadeAdapter>,
    cache: Arc<PreparedStatementCache>,
    catalog_manager: Option<Arc<CatalogManager>>,
}

impl UnifiedQueryPortImpl {
    /// Create with a pre-built adapter and a default prepared-statement cache.
    pub fn new(adapter: Arc<QueryFacadeAdapter>) -> Self {
        Self {
            adapter,
            cache: Arc::new(PreparedStatementCache::new(
                PreparedStatementConfig::default(),
            )),
            catalog_manager: None,
        }
    }

    /// Create with a custom prepared-statement cache.
    pub fn with_cache(
        adapter: Arc<QueryFacadeAdapter>,
        cache: Arc<PreparedStatementCache>,
    ) -> Self {
        Self {
            adapter,
            cache,
            catalog_manager: None,
        }
    }

    /// Attach xCatalog so port-backed EXPLAIN can expose planner-native authority metadata.
    pub fn with_catalog_manager(mut self, catalog_manager: Arc<CatalogManager>) -> Self {
        self.catalog_manager = Some(catalog_manager);
        self
    }

    /// Serialize a `QueryResult` to `serde_json::Value`, applying an optional row limit.
    fn result_to_json(
        result: crate::query::facade::QueryResult,
        limit: Option<u32>,
    ) -> Result<serde_json::Value> {
        let limit = limit.unwrap_or(u32::MAX) as usize;
        use crate::query::facade::QueryResultData;
        let rows: Vec<serde_json::Value> = match result.data {
            QueryResultData::Rows(rows) => rows.into_iter().take(limit).collect(),
            QueryResultData::VectorResults(matches) => matches
                .into_iter()
                .take(limit)
                .map(|m| {
                    serde_json::json!({
                        "id": m.record.oid,
                        "score": m.score,
                        "rank": m.rank,
                    })
                })
                .collect(),
            QueryResultData::Graph(gr) => {
                let _ = limit; // apply below after conversion
                vec![serde_json::to_value(&gr).unwrap_or(serde_json::Value::Null)]
            }
            QueryResultData::Empty => vec![],
        };
        let metrics = serde_json::to_value(&result.metrics).unwrap_or(serde_json::Value::Null);
        Ok(serde_json::json!({
            "records": rows,
            "total_count": rows.len(),  // post-limit count; accurate for page consumers
            "metrics": metrics,
        }))
    }

    async fn explain_storage_authority_from_catalog(
        &self,
        collection: Option<&str>,
        query: &str,
    ) -> Result<Option<StorageAuthorityExplanation>> {
        let Some(catalog_manager) = &self.catalog_manager else {
            return Ok(None);
        };

        let mut context = PlanContext::default();
        let mut targets = explain_catalog_targets(query);
        if let Some(collection) = collection
            && !collection.trim().is_empty()
        {
            targets.insert(0, collection.trim().to_string());
        }
        targets.sort();
        targets.dedup();

        for target in targets {
            match resolve_catalog_authority_context(
                catalog_manager,
                AuthoritySource::new(target.clone(), "relational"),
            )
            .await
            {
                Ok(resolved) => context.resolved_objects.push(resolved),
                Err(err) => {
                    debug!(
                        "port-backed EXPLAIN storage authority unavailable for '{}': {}",
                        target, err
                    );
                }
            }
        }

        Ok(StorageAuthorityExplanation::from_plan_context(&context))
    }
}

fn explain_catalog_targets(sql: &str) -> Vec<String> {
    let mut targets = Vec::new();
    let normalized = sql.replace(['\n', '\t', ',', '(', ')'], " ");
    let tokens: Vec<&str> = normalized.split_whitespace().collect();

    for window in tokens.windows(2) {
        if let [keyword, target] = window {
            let keyword = keyword.trim_matches('"').to_ascii_uppercase();
            if matches!(keyword.as_str(), "FROM" | "JOIN" | "INTO" | "UPDATE")
                && !target.starts_with('$')
            {
                targets.push(target.trim_matches('"').trim_end_matches(';').to_string());
            }
        }
    }

    for function in [
        "VECTOR_SEARCH",
        "DOCUMENT_QUERY",
        "GRAPH_QUERY",
        "LOGS",
        "METRICS",
    ] {
        let needle = format!("{function}(");
        let mut search_from = 0;
        while let Some(offset) = sql[search_from..].to_ascii_uppercase().find(&needle) {
            let start = search_from + offset + needle.len();
            let Some(rest) = sql.get(start..) else {
                break;
            };
            let candidate = rest
                .split([',', ')'])
                .next()
                .unwrap_or_default()
                .trim()
                .trim_matches('\'')
                .trim_matches('"');
            if !candidate.is_empty() && !candidate.starts_with('$') {
                targets.push(candidate.to_string());
            }
            search_from = start;
        }
    }

    targets
        .into_iter()
        .filter(|target| {
            let upper = target.to_ascii_uppercase();
            !matches!(
                upper.as_str(),
                "SELECT" | "WHERE" | "ON" | "AS" | "LATERAL" | "UNNEST"
            )
        })
        .collect()
}

#[async_trait]
impl UnifiedQueryPort for UnifiedQueryPortImpl {
    async fn execute_unified_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        _collection: Option<String>,
        limit: Option<u32>,
    ) -> Result<serde_json::Value> {
        if query.trim().is_empty() {
            return Err(anyhow!("query cannot be empty"));
        }
        debug!(
            "execute_unified_query: {}",
            query.chars().take(120).collect::<String>()
        );
        let parameters = parameters.unwrap_or_default();
        let federated_query = match uql_to_federated_sql(&query, &parameters, limit)
            .with_context(|| format!("UQL lowering failed for query '{}'", query))?
        {
            Some(lowered) => lowered,
            None => bind_federated_sql_parameters(&query, &parameters)?,
        };
        let result = self
            .adapter
            .federated_query(&federated_query)
            .await
            .context("federated_query failed")?;
        Self::result_to_json(result, limit)
    }

    async fn execute_multi_model_query(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // Convert the JSON multi-model request to a federated SQL string.
        // The root-crate logic is in multimodal_query::convert_multi_model_to_sql.
        // We replicate a simplified version here so we don't import the handler module.
        let sql = json_to_multi_model_sql(&request).unwrap_or_else(|| {
            // Fallback: treat "query" field as raw SQL, or use a SELECT 1.
            request
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("SELECT 1")
                .to_string()
        });
        info!(
            "execute_multi_model_query SQL: {}",
            &sql[..sql.len().min(200)]
        );
        let result = self
            .adapter
            .federated_query(&sql)
            .await
            .context("multi-model federated_query failed")?;
        Self::result_to_json(result, None)
    }

    async fn execute_federated_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
    ) -> Result<serde_json::Value> {
        if query.trim().is_empty() {
            return Err(anyhow!("query cannot be empty"));
        }
        let parameters = parameters.unwrap_or_default();
        let query = bind_federated_sql_parameters(&query, &parameters)?;
        debug!(
            "execute_federated_query: {}",
            query.chars().take(120).collect::<String>()
        );
        let result = self
            .adapter
            .federated_query(&query)
            .await
            .context("federated_query failed")?;
        Self::result_to_json(result, None)
    }

    async fn execute_distributed_query(
        &self,
        request: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let query = request
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("distributed query request must have a 'query' field"))?
            .to_string();
        let limit = request
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|l| l as u32);
        debug!(
            "execute_distributed_query: {}",
            query.chars().take(120).collect::<String>()
        );
        let result = self
            .adapter
            .distributed_query(&query)
            .await
            .context("distributed_query failed")?;
        Self::result_to_json(result, limit)
    }

    async fn explain_unified_query(
        &self,
        query: String,
        collection: Option<String>,
    ) -> Result<serde_json::Value> {
        let explain = self.adapter.explain(&query).context("explain failed")?;
        let mut value =
            serde_json::to_value(&explain).context("failed to serialize explain result")?;
        let storage_authority = self
            .explain_storage_authority_from_catalog(collection.as_deref(), &query)
            .await?;
        if let Some(storage_authority) = storage_authority
            && let serde_json::Value::Object(ref mut object) = value
        {
            object.insert(
                "storage_authority".to_string(),
                serde_json::to_value(storage_authority)
                    .context("failed to serialize storage authority")?,
            );
        }
        Ok(value)
    }

    async fn prepare_statement(
        &self,
        _name: Option<String>,
        query: String,
        _cache_results: bool,
        ttl_seconds: Option<u64>,
    ) -> Result<String> {
        let ttl = Duration::from_secs(ttl_seconds.unwrap_or(3600));
        self.cache
            .prepare_with_ttl(&query, ttl)
            .map_err(|e| anyhow!("prepare_statement failed: {}", e))
    }

    async fn execute_prepared(
        &self,
        statement_id: String,
        parameters: Option<Vec<ProximaValue>>,
        _collection: Option<String>,
    ) -> Result<serde_json::Value> {
        let params = proxima_values_to_params(parameters);
        let sql = self
            .cache
            .execute_sql(&statement_id, &params)
            .map_err(|e| match e {
                PreparedStatementError::NotFound(_) => {
                    anyhow!("prepared statement not found: {}", statement_id)
                }
                PreparedStatementError::Expired(_) => {
                    anyhow!("prepared statement expired: {}", statement_id)
                }
                other => anyhow!("prepared statement error: {}", other),
            })?;
        let result = self
            .adapter
            .federated_query(&sql)
            .await
            .context("execute_prepared federated_query failed")?;
        Self::result_to_json(result, None)
    }

    async fn delete_prepared(&self, statement_id: String) -> Result<()> {
        self.cache
            .drop_statement(&statement_id)
            .map_err(|e| anyhow!("delete_prepared failed: {}", e))
    }

    async fn get_prepared_stats(&self, _statement_ids: Vec<String>) -> Result<serde_json::Value> {
        let stats = self.cache.stats();
        Ok(serde_json::json!({
            "cached_statements": stats.cached_statements,
            "max_statements": stats.max_statements,
            "total_executions": stats.total_executions,
            "total_access_count": stats.total_access_count,
            "oldest_statement_age_secs": stats.oldest_statement_age_secs,
        }))
    }
}

// ── Multi-model JSON → SQL conversion ────────────────────────────────────────

/// Convert a multi-model query JSON to a federated SQL string.
///
/// Mirrors the logic in `src/network/rest/canonical/multimodal_query::convert_multi_model_to_sql`.
/// Returns `None` when the request cannot be converted.
fn json_to_multi_model_sql(req: &serde_json::Value) -> Option<String> {
    let components = req.get("components")?.as_array()?;
    if components.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    for component in components {
        let ctype = component.get("component_type")?.as_str()?;
        let config = component.get("config").cloned().unwrap_or_default();
        let sql_part = match ctype {
            "vector" => {
                let collection = config
                    .get("collection")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                let query_vec = config
                    .get("query_vector")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_f64())
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default();
                let top_k = config.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10);
                format!(
                    "SELECT * FROM VECTOR_SEARCH('{}', '[{}]', {})",
                    collection, query_vec, top_k
                )
            }
            "document" => {
                let collection = config
                    .get("collection")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                let filter = config
                    .get("filter")
                    .and_then(|v| v.as_str())
                    .unwrap_or("1=1");
                format!(
                    "SELECT * FROM DOCUMENT_QUERY('{}', '{}')",
                    collection, filter
                )
            }
            "graph" => {
                let cypher = config
                    .get("cypher")
                    .and_then(|v| v.as_str())
                    .unwrap_or("MATCH (n) RETURN n LIMIT 10");
                format!("SELECT * FROM GRAPH_QUERY('{}')", cypher)
            }
            "observability" => {
                let namespace = config
                    .get("namespace")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                format!("SELECT * FROM LOGS('{}')", namespace)
            }
            _ => continue,
        };
        parts.push(sql_part);
    }

    if parts.is_empty() {
        return None;
    }

    // Single component: use directly; multiple: UNION ALL
    if parts.len() == 1 {
        Some(parts.remove(0))
    } else {
        Some(parts.join(" UNION ALL "))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::TableIdentifier;
    use crate::query::authority_context::{AuthoritySource, resolved_object_from_catalog_schema};
    use crate::query::multimodal::plan::ResolvedAuthorityMode;
    use proximadb_catalog::{CatalogPhysicalFormat, CatalogStorageLayout, CatalogTableSchema};

    #[test]
    fn test_proxima_value_to_param_string() {
        let value = ProximaValue::String("hello".into());
        assert!(
            matches!(proxima_value_to_param(&value), ParameterValue::String(s) if s == "hello")
        );
    }

    #[test]
    fn test_proxima_value_to_param_int() {
        let value = ProximaValue::Int64(42);
        assert!(matches!(
            proxima_value_to_param(&value),
            ParameterValue::Int(42)
        ));
    }

    #[test]
    fn test_proxima_value_to_param_composites() {
        let value = ProximaValue::Array(vec![ProximaValue::Int64(1), ProximaValue::Int64(2)]);
        assert!(matches!(
            proxima_value_to_param(&value),
            ParameterValue::Json(_)
        ));
    }

    #[test]
    fn test_uql_vector_select_lowers_to_federated_vector_search() {
        let sql = uql_to_federated_sql(
            "SELECT * FROM vectors.products WHERE VECTOR_SIMILAR(embedding, ?, 0.8) LIMIT 7",
            &[ProximaValue::Array(vec![
                ProximaValue::Float64(0.1),
                ProximaValue::Float64(0.2),
                ProximaValue::Float64(0.3),
            ])],
            None,
        )
        .expect("lowering should succeed")
        .expect("query should lower");

        assert_eq!(
            sql,
            "SELECT * FROM VECTOR_SEARCH('products', '[0.1,0.2,0.3]', 7)"
        );
    }

    #[test]
    fn test_uql_vector_select_uses_request_limit_override() {
        let sql = uql_to_federated_sql(
            "SELECT * FROM vectors.products WHERE VECTOR_SIMILAR(embedding, ?, 0.8) LIMIT 7",
            &[ProximaValue::DenseVector(vec![0.1, 0.2])],
            Some(3),
        )
        .expect("lowering should succeed")
        .expect("query should lower");

        assert_eq!(
            sql,
            "SELECT * FROM VECTOR_SEARCH('products', '[0.1,0.2]', 3)"
        );
    }

    #[test]
    fn test_uql_document_select_lowers_to_document_query() {
        let sql = uql_to_federated_sql(
            "SELECT * FROM docs.orders WHERE $.status = 'pending' LIMIT 5",
            &[],
            None,
        )
        .expect("lowering should succeed")
        .expect("query should lower");

        assert_eq!(
            sql,
            "SELECT * FROM DOCUMENT_QUERY('orders', 'status = \"pending\"') LIMIT 5"
        );
    }

    #[test]
    fn test_non_uql_query_is_left_for_federated_sql() {
        assert!(
            uql_to_federated_sql(
                "SELECT * FROM VECTOR_SEARCH('products', '[0.1]', 10)",
                &[],
                None
            )
            .expect("non-UQL parse errors should not fail")
            .is_none()
        );
    }

    #[test]
    fn test_bind_federated_sql_parameters_vector_and_limit() {
        let sql = bind_federated_sql_parameters(
            "SELECT * FROM VECTOR_SEARCH('products', ?, ?)",
            &[
                ProximaValue::DenseVector(vec![0.1, 0.2, 0.3]),
                ProximaValue::Int64(5),
            ],
        )
        .expect("parameters should bind");

        assert_eq!(
            sql,
            "SELECT * FROM VECTOR_SEARCH('products', '[0.1,0.2,0.3]', 5)"
        );
    }

    #[test]
    fn test_bind_federated_sql_parameters_ignores_question_marks_in_strings() {
        let sql = bind_federated_sql_parameters(
            "SELECT * FROM DOCUMENT_QUERY('docs', 'title = \"why?\" AND status = ?') WHERE id = ?",
            &[ProximaValue::String("doc-1".to_string())],
        )
        .expect("only the placeholder outside the quoted filter should bind");

        assert_eq!(
            sql,
            "SELECT * FROM DOCUMENT_QUERY('docs', 'title = \"why?\" AND status = ?') WHERE id = 'doc-1'"
        );
    }

    #[test]
    fn test_bind_federated_sql_parameters_rejects_missing_parameter() {
        let error = bind_federated_sql_parameters(
            "SELECT * FROM VECTOR_SEARCH('products', ?, ?)",
            &[ProximaValue::DenseVector(vec![0.1])],
        )
        .expect_err("missing placeholder parameter should fail");

        assert!(error.to_string().contains("missing parameter 2"));
    }

    #[test]
    fn test_bind_federated_sql_parameters_rejects_unused_parameter() {
        let error = bind_federated_sql_parameters(
            "SELECT * FROM VECTOR_SEARCH('products', '[0.1]', 1)",
            &[ProximaValue::Int64(1)],
        )
        .expect_err("unused parameter should fail");

        assert!(error.to_string().contains("only used 0 placeholders"));
    }

    #[test]
    fn test_proxima_value_to_param_null() {
        let value = ProximaValue::Null;
        assert!(matches!(
            proxima_value_to_param(&value),
            ParameterValue::Null
        ));
    }

    #[test]
    fn test_json_to_multi_model_sql_vector() {
        let req = serde_json::json!({
            "components": [
                {
                    "component_type": "vector",
                    "config": {
                        "collection": "embeddings",
                        "query_vector": [0.1, 0.2, 0.3],
                        "top_k": 5
                    }
                }
            ]
        });
        let sql = json_to_multi_model_sql(&req).unwrap();
        assert!(sql.contains("VECTOR_SEARCH('embeddings'"));
        assert!(sql.contains(", 5)"));
    }

    #[test]
    fn test_json_to_multi_model_sql_empty_components() {
        let req = serde_json::json!({ "components": [] });
        assert!(json_to_multi_model_sql(&req).is_none());
    }

    #[test]
    fn test_json_to_multi_model_sql_no_components_field() {
        let req = serde_json::json!({ "query": "SELECT 1" });
        assert!(json_to_multi_model_sql(&req).is_none());
    }

    #[test]
    fn test_explain_catalog_targets_extracts_from_sql_and_functions() {
        let targets = explain_catalog_targets(
            "SELECT * FROM default.docs d JOIN graph.edges e ON d.id = e.src \
             UNION ALL SELECT * FROM VECTOR_SEARCH('vectors', '[0.1]', 10)",
        );

        assert!(targets.contains(&"default.docs".to_string()));
        assert!(targets.contains(&"graph.edges".to_string()));
        assert!(targets.contains(&"vectors".to_string()));
    }

    #[test]
    fn test_resolved_object_from_catalog_schema_preserves_external_policy_boundary() {
        let table_id = TableIdentifier::new(vec!["lake".to_string()], "docs".to_string());
        let mut schema = CatalogTableSchema::new("docs");
        schema.storage_layouts = vec![CatalogStorageLayout::external_authoritative(
            "iceberg",
            CatalogPhysicalFormat::Iceberg,
            "s3://warehouse/docs",
        )];

        let object = resolved_object_from_catalog_schema(
            AuthoritySource::new("lake.docs", "document"),
            &table_id,
            &schema,
        );

        assert_eq!(
            object.authority,
            ResolvedAuthorityMode::ExternalAuthoritative
        );
        assert!(object.requires_policy_boundary());
        assert_eq!(object.storage_layouts[0].physical_format, "Iceberg");
    }
}
