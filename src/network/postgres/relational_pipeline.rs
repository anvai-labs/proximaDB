//! Bridge between the pgwire surface and the new relational
//! pipeline (algebra → planner → executor → engine). This is the
//! S5c integration point — when
//! `PROXIMADB_PGWIRE_RELATIONAL_PIPELINE=1` is set in the
//! environment, SELECT statements on tables that exist in the
//! global [`InMemoryRelationalEngine`] route through this module
//! instead of the legacy `QueryLowering` → vector-collection
//! path.
//!
//! Scope:
//!
//! - Read-only path (SELECT only). INSERT/UPDATE/DELETE on engine
//!   tables stay on the legacy path until a follow-up slice wires
//!   the write side.
//! - Tables are discovered from the engine's catalog
//!   ([`InMemoryRelationalEngine::schema_of`]).
//! - If the SQL doesn't lower cleanly (e.g. it's a pg-specific
//!   query like `SELECT current_schema()`), [`try_run_select`]
//!   returns `None` so the caller falls through to the legacy
//!   path.
//! - The engine is bootstrapped lazily once per process. When
//!   `PROXIMADB_RELATIONAL_BOOTSTRAP_DEMO_TABLE=1` is also set,
//!   a `demo_users(id BIGINT PRIMARY KEY, name TEXT, age INT)`
//!   table is seeded with 3 rows so operators can demonstrate
//!   the new path end-to-end via psql.

use crate::query::execution::{
    ExecutionControls, ExecutionPipelineResult, NativeVolcanoEngine, normalize_table_key,
};
#[cfg(feature = "datafusion-integration")]
use crate::query::execution::{QueryExecutionContext, execute_sql_with_backend};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use proximadb_data_model::{ProximaType, ProximaValue, StatsTrust};
use proximadb_functions::builtins;
use proximadb_relational_algebra::TableId;
use proximadb_relational_engine::{
    EngineReaderFactory, InMemoryRelationalEngine, RelationalWriter,
};
use proximadb_relational_executor::{ExecError, NodeMetric, ReaderFactory};
use proximadb_relational_frontend::{CatalogLookup, LoweringDecline, lower_sql};
use proximadb_relational_planner::{
    CapabilityResolver, PhysicalPlan, Planner, StaticCapabilities, explain_physical,
    measure_geometry, stack_probe,
};
use proximadb_relational_reader::{ReaderCapabilities, ReaderError, RelationalReader, ScanContext};
use proximadb_relational_types::{ColumnInfo, Expr, ExprError, RelationalRow, RelationalSchema};
use sqlparser::ast::{
    BinaryOperator, Expr as SqlExpr, GroupByExpr, LimitClause, OrderByKind, Query as SqlQuery,
    SelectItem, SetExpr, SetOperator, Statement, TableFactor, TableWithJoins, Value,
};
use sqlparser::dialect::{GenericDialect, PostgreSqlDialect};
use sqlparser::parser::Parser;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::search::VectorFreshnessMode;
use crate::query::cache::invalidation_coordinator::CacheInvalidationCoordinator;
use crate::query::cache::query_result_cache::{QueryKey, QueryResultCache, StructuralKey};
use crate::services::dml::DmlService;
use crate::storage::tenant::context::TenantContext;
use proximadb_runtime::{OwnedPortIdentity, PortIdentity};
use proximadb_vector_query::{VectorSearchExpr, VectorSearchParams};

use super::types::PgType;

// =========================================================================
// Canonical pgvector SELECT lowering
// =========================================================================

/// Logical vector-search shape lowered from PostgreSQL's pgvector operators.
///
/// This lives at the query boundary, not in `protocol.rs`: pgwire supplies SQL
/// and identity, while the query plane owns parsing, semantics, and execution.
/// The private type deliberately exposes no second public vector API.
#[derive(Debug, Clone, PartialEq)]
struct PgVectorSelectPlan {
    search: VectorSearchExpr,
    output: Vec<PgVectorOutputColumn>,
}

#[derive(Debug, Clone, PartialEq)]
enum PgVectorOutputColumn {
    Id(String),
    Distance(String),
    Metadata { source: String, output: String },
}

impl PgVectorOutputColumn {
    fn schema_column(&self) -> ColumnInfo {
        match self {
            Self::Id(name) => ColumnInfo::new(name, ProximaType::String, false),
            Self::Distance(name) => ColumnInfo::new(name, ProximaType::Float64, false),
            Self::Metadata { output, .. } => ColumnInfo::new(output, ProximaType::Jsonb, false),
        }
    }

    fn field_description(&self) -> super::types::FieldDescription {
        match self {
            Self::Id(name) => super::types::FieldDescription::new(name, PgType::Text),
            Self::Distance(name) => super::types::FieldDescription::new(name, PgType::Float8),
            Self::Metadata { output, .. } => {
                super::types::FieldDescription::new(output, PgType::Jsonb)
            }
        }
    }
}

/// Lower the supported pgvector nearest-neighbour SELECT into one typed plan.
/// `Ok(None)` means the SQL is not a pgvector operator query. Once an operator
/// is present, malformed or unsupported semantics are errors; they must never
/// fall through to a string parser or silently widen a metadata predicate.
fn lower_pgvector_select(sql: &str) -> Result<Option<PgVectorSelectPlan>, String> {
    let statements = match Parser::parse_sql(&PostgreSqlDialect {}, sql) {
        Ok(statements) => statements,
        Err(_) => return Ok(None),
    };
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let Some(metric) = pgvector_ordering_metric(query)? else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err("pgvector operators require a simple SELECT body".to_string());
    };
    let [from] = select.from.as_slice() else {
        return Err("pgvector operators require exactly one source collection".to_string());
    };
    if !from.joins.is_empty() {
        return Err(
            "pgvector operator JOINs must use vector_search(...) in the relational pipeline"
                .to_string(),
        );
    }
    let TableFactor::Table {
        name,
        args: None,
        with_hints,
        ..
    } = &from.relation
    else {
        return Err("pgvector operators require a collection table source".to_string());
    };
    if !with_hints.is_empty() {
        return Err("table hints are not supported for pgvector search".to_string());
    }

    let Some(order_by) = query.order_by.as_ref() else {
        return Err("pgvector nearest-neighbour SELECT requires ORDER BY".to_string());
    };
    let OrderByKind::Expressions(order_exprs) = &order_by.kind else {
        return Err("ORDER BY ALL is not valid for pgvector search".to_string());
    };
    let [order] = order_exprs.as_slice() else {
        return Err(
            "pgvector search requires exactly one distance expression in ORDER BY".to_string(),
        );
    };
    if order.options.asc == Some(false) {
        return Err("pgvector nearest-neighbour ORDER BY must be ascending".to_string());
    }

    let SqlExpr::BinaryOp { left, op, right } = &order.expr else {
        return Err("ORDER BY must contain a pgvector distance operator".to_string());
    };
    debug_assert_eq!(pgvector_metric(op), Some(metric));
    let vector_column = sql_column_name(left)
        .ok_or_else(|| "pgvector distance left operand must be a vector column".to_string())?;
    let query_vector = parse_vector_expr(right)?;
    if query_vector.is_empty() {
        return Err("pgvector query vector must not be empty".to_string());
    }

    let top_k = match query.limit_clause.as_ref() {
        None => 10,
        Some(LimitClause::LimitOffset {
            limit: Some(limit),
            offset: None,
            limit_by,
        }) if limit_by.is_empty() => parse_positive_usize(limit, "LIMIT")?,
        Some(_) => {
            return Err("pgvector search supports LIMIT without OFFSET or LIMIT BY".to_string());
        }
    };
    let top_k = u32::try_from(top_k)
        .map_err(|_| "pgvector LIMIT exceeds the supported u32 result cardinality".to_string())?;

    let filter = select
        .selection
        .as_ref()
        .map(lower_vector_filter)
        .transpose()?;
    let output = lower_pgvector_projection(&select.projection, metric)?;

    Ok(Some(PgVectorSelectPlan {
        search: VectorSearchExpr {
            collection: name.to_string(),
            vector_column: Some(vector_column),
            query_vector,
            top_k,
            threshold: None,
            metric,
            filter,
            params: VectorSearchParams::default(),
        },
        output,
    }))
}

/// Return the metric only when the parsed query's ordering expression is a
/// pgvector operator. Literal/comment text is deliberately irrelevant to this
/// routing decision.
fn pgvector_ordering_metric(
    query: &SqlQuery,
) -> Result<Option<proximadb_distance_types::DistanceMetric>, String> {
    let Some(order_by) = query.order_by.as_ref() else {
        return Ok(None);
    };
    let OrderByKind::Expressions(order_exprs) = &order_by.kind else {
        return Ok(None);
    };
    let metrics = order_exprs
        .iter()
        .filter_map(|order| match &order.expr {
            SqlExpr::BinaryOp { op, .. } => pgvector_metric(op),
            _ => None,
        })
        .collect::<Vec<_>>();
    if metrics.is_empty() {
        return Ok(None);
    }
    if order_exprs.len() != 1 || metrics.len() != 1 {
        return Err(
            "pgvector search requires exactly one distance expression in ORDER BY".to_string(),
        );
    }
    Ok(metrics.first().copied())
}

fn pgvector_metric(operator: &BinaryOperator) -> Option<proximadb_distance_types::DistanceMetric> {
    match operator {
        BinaryOperator::LtDashGt => Some(proximadb_distance_types::DistanceMetric::L2),
        BinaryOperator::Spaceship => Some(proximadb_distance_types::DistanceMetric::Cosine),
        BinaryOperator::Custom(operator) if operator == "<#>" => {
            Some(proximadb_distance_types::DistanceMetric::InnerProduct)
        }
        _ => None,
    }
}

fn lower_pgvector_projection(
    projection: &[SelectItem],
    order_metric: proximadb_distance_types::DistanceMetric,
) -> Result<Vec<PgVectorOutputColumn>, String> {
    if matches!(projection, [SelectItem::Wildcard(_)]) {
        return Ok(vec![
            PgVectorOutputColumn::Id("id".to_string()),
            PgVectorOutputColumn::Distance("distance".to_string()),
            PgVectorOutputColumn::Metadata {
                source: "metadata".to_string(),
                output: "metadata".to_string(),
            },
        ]);
    }

    projection
        .iter()
        .map(|item| {
            let (expr, alias) = match item {
                SelectItem::UnnamedExpr(expr) => (expr, None),
                SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.as_str())),
                _ => return Err(format!("unsupported pgvector projection: {item}")),
            };
            if let Some(column) = sql_column_name(expr) {
                return match column.to_ascii_lowercase().as_str() {
                    "id" => Ok(PgVectorOutputColumn::Id(
                        alias.unwrap_or(column.as_str()).to_string(),
                    )),
                    "payload" | "metadata" => Ok(PgVectorOutputColumn::Metadata {
                        source: column.clone(),
                        output: alias.unwrap_or(column.as_str()).to_string(),
                    }),
                    _ => Err(format!("unsupported pgvector projection column: {column}")),
                };
            }
            let SqlExpr::BinaryOp { op, .. } = uncast(expr) else {
                return Err(format!("unsupported pgvector projection: {item}"));
            };
            if pgvector_metric(op) != Some(order_metric) {
                return Err("projected vector distance must use the ORDER BY metric".to_string());
            }
            Ok(PgVectorOutputColumn::Distance(
                alias.unwrap_or("distance").to_string(),
            ))
        })
        .collect()
}

pub(super) fn describe_pgvector_select(
    sql: &str,
) -> Result<Option<Vec<super::types::FieldDescription>>, String> {
    let statements = match Parser::parse_sql(&PostgreSqlDialect {}, sql) {
        Ok(statements) => statements,
        Err(_) => return Ok(None),
    };
    let [Statement::Query(query)] = statements.as_slice() else {
        return Ok(None);
    };
    let Some(metric) = pgvector_ordering_metric(query)? else {
        return Ok(None);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err("pgvector operators require a simple SELECT body".to_string());
    };
    let Some(order_by) = query.order_by.as_ref() else {
        return Err("pgvector nearest-neighbour SELECT requires ORDER BY".to_string());
    };
    let OrderByKind::Expressions(order_exprs) = &order_by.kind else {
        return Err("ORDER BY ALL is not valid for pgvector search".to_string());
    };
    let [order] = order_exprs.as_slice() else {
        return Err("pgvector search requires one distance ORDER BY".to_string());
    };
    let SqlExpr::BinaryOp { op, .. } = &order.expr else {
        return Err("ORDER BY must contain a pgvector distance operator".to_string());
    };
    debug_assert_eq!(pgvector_metric(op), Some(metric));
    Ok(Some(
        lower_pgvector_projection(&select.projection, metric)?
            .iter()
            .map(PgVectorOutputColumn::field_description)
            .collect(),
    ))
}

fn sql_column_name(expr: &SqlExpr) -> Option<String> {
    match expr {
        SqlExpr::Identifier(ident) => Some(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => parts.last().map(|ident| ident.value.clone()),
        SqlExpr::Nested(expr) => sql_column_name(expr),
        _ => None,
    }
}

fn uncast(expr: &SqlExpr) -> &SqlExpr {
    match expr {
        SqlExpr::Cast { expr, .. } | SqlExpr::Nested(expr) => uncast(expr),
        _ => expr,
    }
}

fn sql_string(expr: &SqlExpr) -> Option<&str> {
    let SqlExpr::Value(value) = uncast(expr) else {
        return None;
    };
    match &value.value {
        Value::SingleQuotedString(value)
        | Value::EscapedStringLiteral(value)
        | Value::DoubleQuotedString(value) => Some(value.as_str()),
        _ => None,
    }
}

fn parse_vector_expr(expr: &SqlExpr) -> Result<Vec<f32>, String> {
    let literal = sql_string(expr)
        .ok_or_else(|| "pgvector query operand must be a bound or literal vector".to_string())?;
    let values = literal
        .trim()
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| "pgvector literal must use [x,y,...] syntax".to_string())?;
    values
        .split(',')
        .map(|value| {
            let parsed = value
                .trim()
                .parse::<f32>()
                .map_err(|error| format!("invalid pgvector component `{value}`: {error}"))?;
            if !parsed.is_finite() {
                return Err(format!(
                    "invalid pgvector component `{value}`: value must be finite"
                ));
            }
            Ok(parsed)
        })
        .collect()
}

fn parse_positive_usize(expr: &SqlExpr, label: &str) -> Result<usize, String> {
    let value = match uncast(expr) {
        SqlExpr::Value(value) => match &value.value {
            Value::Number(value, _) | Value::SingleQuotedString(value) => value.as_str(),
            _ => return Err(format!("{label} must be a positive integer")),
        },
        _ => return Err(format!("{label} must be a positive integer")),
    };
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{label} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(parsed)
}

fn lower_vector_filter(
    expr: &SqlExpr,
) -> Result<proximadb_filter_expression::FilterExpression, String> {
    use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

    let SqlExpr::BinaryOp { left, op, right } = uncast(expr) else {
        return Err(format!("unsupported vector-search filter: {expr}"));
    };
    match op {
        BinaryOperator::And => Ok(FilterExpression::And(vec![
            lower_vector_filter(left)?,
            lower_vector_filter(right)?,
        ])),
        BinaryOperator::Or => Ok(FilterExpression::Or(vec![
            lower_vector_filter(left)?,
            lower_vector_filter(right)?,
        ])),
        BinaryOperator::Eq
        | BinaryOperator::NotEq
        | BinaryOperator::Gt
        | BinaryOperator::GtEq
        | BinaryOperator::Lt
        | BinaryOperator::LtEq => {
            let field = vector_filter_field(left)
                .ok_or_else(|| format!("unsupported vector-search filter field: {left}"))?;
            let value = vector_filter_value(right)
                .ok_or_else(|| format!("unsupported vector-search filter value: {right}"))?;
            let operator = match op {
                BinaryOperator::Eq => ComparisonOperator::Equals,
                BinaryOperator::NotEq => ComparisonOperator::NotEquals,
                BinaryOperator::Gt => ComparisonOperator::GreaterThan,
                BinaryOperator::GtEq => ComparisonOperator::GreaterThanOrEqual,
                BinaryOperator::Lt => ComparisonOperator::LessThan,
                BinaryOperator::LtEq => ComparisonOperator::LessThanOrEqual,
                _ => return Err(format!("unsupported vector-search filter: {expr}")),
            };
            Ok(FilterExpression::Comparison {
                field,
                operator,
                value,
            })
        }
        _ => Err(format!("unsupported vector-search filter: {expr}")),
    }
}

fn vector_filter_field(expr: &SqlExpr) -> Option<String> {
    if let Some(field) = sql_column_name(expr) {
        return Some(field);
    }
    if let SqlExpr::Function(function) = uncast(expr) {
        let name = function.name.to_string().to_ascii_lowercase();
        if matches!(
            name.rsplit('.').next(),
            Some("json_extract_text" | "json_extract_path_text")
        ) && let sqlparser::ast::FunctionArguments::List(arguments) = &function.args
            && arguments.clauses.is_empty()
            && arguments.args.len() >= 2
        {
            let mut args = arguments.args.iter().filter_map(|argument| match argument {
                sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
                    expr,
                )) => Some(expr),
                _ => None,
            });
            let container = sql_column_name(args.next()?)?;
            let path = args.map(sql_string).collect::<Option<Vec<_>>>()?;
            if !path.is_empty() {
                return Some(format!("{container}.{}", path.join(".")));
            }
        }
    }
    let SqlExpr::BinaryOp { left, op, right } = uncast(expr) else {
        return None;
    };
    if !matches!(op, BinaryOperator::LongArrow) {
        return None;
    }
    let container = sql_column_name(left)?;
    if !matches!(
        container.to_ascii_lowercase().as_str(),
        "payload" | "metadata"
    ) {
        return None;
    }
    sql_string(right).map(|field| format!("{container}.{field}"))
}

fn vector_filter_value(expr: &SqlExpr) -> Option<serde_json::Value> {
    let SqlExpr::Value(value) = uncast(expr) else {
        return None;
    };
    match &value.value {
        Value::SingleQuotedString(value)
        | Value::EscapedStringLiteral(value)
        | Value::DoubleQuotedString(value) => Some(serde_json::Value::String(value.clone())),
        Value::Number(value, _) => value
            .parse::<i64>()
            .map(serde_json::Value::from)
            .or_else(|_| value.parse::<f64>().map(serde_json::Value::from))
            .ok(),
        Value::Boolean(value) => Some(serde_json::Value::Bool(*value)),
        Value::Null => Some(serde_json::Value::Null),
        _ => None,
    }
}

async fn execute_pgvector_plan(
    plan: PgVectorSelectPlan,
    dml: Option<&Arc<DmlService>>,
    vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
    identity: PortIdentity<'_>,
    tenant_context: Option<&TenantContext>,
    controls: ExecutionControls,
) -> Result<ExecutionPipelineResult, String> {
    controls
        .check_cancelled()
        .map_err(|error| error.to_string())?;
    // Catalog tables search their authoritative TableRecordStore primary. Native
    // collections search VectorOps. The catalog existence check is the one route
    // decision; there is no dual read and no empty-result fallback between stores.
    let results = if let Some(dml) = dml {
        let vector_column =
            plan.search.vector_column.as_deref().ok_or_else(|| {
                "pgvector table search requires a named vector column".to_string()
            })?;
        let requested_metadata: Vec<String> = plan
            .output
            .iter()
            .filter_map(|column| match column {
                PgVectorOutputColumn::Metadata { source, .. }
                    if !source.eq_ignore_ascii_case("metadata") =>
                {
                    Some(source.clone())
                }
                _ => None,
            })
            .collect();
        let metadata_columns = (!plan.output.iter().any(|column| {
            matches!(column, PgVectorOutputColumn::Metadata { source, .. } if source.eq_ignore_ascii_case("metadata"))
        }))
        .then_some(requested_metadata.as_slice());
        match dml
            .search_table_vectors_exact(
                &plan.search.collection,
                vector_column,
                &plan.search.query_vector,
                plan.search.top_k as usize,
                plan.search.filter.as_ref(),
                plan.search.metric,
                metadata_columns,
                tenant_context,
                identity,
                &controls,
            )
            .await
            .map_err(|error| format!("catalog-table vector search failed: {error}"))?
        {
            Some(results) => results,
            None => vector_ops
                .unified_search_native(&plan.search, identity)
                .await
                .map_err(|error| format!("native collection vector search failed: {error}"))?,
        }
    } else {
        vector_ops
            .unified_search_native(&plan.search, identity)
            .await
            .map_err(|error| format!("native collection vector search failed: {error}"))?
    };

    let schema = RelationalSchema::new(
        plan.output
            .iter()
            .map(PgVectorOutputColumn::schema_column)
            .collect(),
    );
    let rows = results
        .into_iter()
        .map(|record| {
            plan.output
                .iter()
                .map(|column| match column {
                    PgVectorOutputColumn::Id(_) => Ok(ProximaValue::String(record.id.clone())),
                    PgVectorOutputColumn::Distance(_) => {
                        pgvector_distance(&record, plan.search.metric).map(ProximaValue::Float64)
                    }
                    PgVectorOutputColumn::Metadata { source, .. }
                        if source.eq_ignore_ascii_case("metadata") =>
                    {
                        serde_json::to_value(&record.metadata)
                            .map(ProximaValue::Jsonb)
                            .map_err(|error| format!("vector metadata conversion failed: {error}"))
                    }
                    PgVectorOutputColumn::Metadata { source, .. } => Ok(record
                        .metadata
                        .get(source)
                        .cloned()
                        .unwrap_or(ProximaValue::Null)),
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    ExecutionPipelineResult { schema, rows }
        .enforce_row_limit(controls.max_rows)
        .map_err(|error| error.to_string())
}

fn pgvector_distance(
    record: &proximadb_search_types::results::OptimizedSearchRecord,
    metric: proximadb_distance_types::DistanceMetric,
) -> Result<f64, String> {
    let expected = match metric {
        proximadb_distance_types::DistanceMetric::L2 => {
            proximadb_distance_kernel::DistanceMetric::Euclidean
        }
        proximadb_distance_types::DistanceMetric::Cosine => {
            proximadb_distance_kernel::DistanceMetric::Cosine
        }
        proximadb_distance_types::DistanceMetric::InnerProduct => {
            proximadb_distance_kernel::DistanceMetric::DotProduct
        }
        proximadb_distance_types::DistanceMetric::L1 => {
            proximadb_distance_kernel::DistanceMetric::Manhattan
        }
    };
    let semantic = record.semantic_similarity.as_ref().ok_or_else(|| {
        format!(
            "vector search result '{}' omitted the requested {metric} distance",
            record.id
        )
    })?;
    if semantic.metric != expected {
        return Err(format!(
            "vector search result '{}' used {:?}, expected {:?}",
            record.id, semantic.metric, expected
        ));
    }
    Ok(semantic.distance as f64)
}

// =========================================================================
// Process-wide engine
// =========================================================================

/// Process-wide [`InMemoryRelationalEngine`] used by the new
/// pipeline. Initialised on first access. **Single-process only**:
/// real distribution will replace this with a SharedServices-owned
/// engine; for the MVP wire-in (S5c) this static keeps the
/// integration self-contained.
pub static GLOBAL_ENGINE: Lazy<Arc<InMemoryRelationalEngine>> = Lazy::new(|| {
    let engine = InMemoryRelationalEngine::new();
    if std::env::var("PROXIMADB_RELATIONAL_BOOTSTRAP_DEMO_TABLE").is_ok() {
        bootstrap_demo_table(&engine);
    }
    engine
});

fn bootstrap_demo_table(engine: &Arc<InMemoryRelationalEngine>) {
    let schema = RelationalSchema::new(vec![
        ColumnInfo::new("id", ProximaType::Int64, false),
        ColumnInfo::new("name", ProximaType::String, true),
        ColumnInfo::new("age", ProximaType::Int32, true),
    ]);
    if engine.create_table("demo_users", schema, vec![0]).is_ok() {
        let _ = engine.insert_rows(
            "demo_users",
            vec![
                vec![
                    ProximaValue::Int64(1),
                    ProximaValue::String("alice".into()),
                    ProximaValue::Int32(30),
                ],
                vec![
                    ProximaValue::Int64(2),
                    ProximaValue::String("bob".into()),
                    ProximaValue::Int32(25),
                ],
                vec![
                    ProximaValue::Int64(3),
                    ProximaValue::String("carol".into()),
                    ProximaValue::Int32(40),
                ],
            ],
        );
    }
}

// =========================================================================
// Process-wide OLAP result cache (default-OFF; ADR-051 D2 / mandate #16)
// =========================================================================

/// Master switch for the pgwire OLAP result cache. **Default-OFF.** Set
/// `PROXIMADB_QUERY_RESULT_CACHE` to a truthy token (`1`/`true`/`on`/`yes`,
/// case-insensitive) to enable structurally tenant-keyed caching of OLAP SELECT
/// results with Strong LSN-pinned freshness. Pure + split out so it is
/// unit-testable without mutating the process environment.
fn olap_result_cache_on(var: Option<&str>) -> bool {
    match var.map(str::trim) {
        Some(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"),
        None => false,
    }
}

fn olap_result_cache_enabled() -> bool {
    olap_result_cache_on(
        std::env::var("PROXIMADB_QUERY_RESULT_CACHE")
            .ok()
            .as_deref(),
    )
}

/// Process-wide OLAP result cache, keyed structurally on
/// `(tenant, namespace, query)`. Shared across every pgwire + REST/gRPC
/// connection (mirrors [`GLOBAL_ENGINE`]'s singleton pattern).
pub static GLOBAL_OLAP_RESULT_CACHE: Lazy<Arc<QueryResultCache<ExecutionPipelineResult>>> =
    Lazy::new(|| Arc::new(QueryResultCache::with_defaults()));

/// Process-wide invalidation coordinator with the OLAP result cache attached,
/// so writes/DDL drop tenant-scoped entries in one fan-out call (mandate #16b).
///
/// Only the `result_cache` arm is attached, deliberately:
/// - **PlanCache** is already invalidated *lazily* — every
///   `invalidate_collection` call bumps `CorpusVersionRegistry`, and `PlanCache`
///   consults that version on lookup → stale entries already miss. An eager
///   `with_plan_cache` arm would only free memory slightly sooner.
/// - **BatchGroupCache** has no production instance (test-only) and is keyed on
///   `(batch_id, group_id)`, not `(tenant, collection)` — there's no
///   `(tenant, collection) → batch_id` mapping to drive it. Wire it when a live
///   batch registry lands.
pub static GLOBAL_OLAP_CACHE_COORDINATOR: Lazy<Arc<CacheInvalidationCoordinator>> =
    Lazy::new(|| {
        Arc::new(
            CacheInvalidationCoordinator::empty()
                .with_result_cache(GLOBAL_OLAP_RESULT_CACHE.clone()),
        )
    });

/// Returns the shared OLAP result cache only when the feature is enabled
/// (default-OFF). Callers pass `.as_deref()` of this as the `result_cache`
/// argument to [`try_run_select`].
pub fn olap_result_cache() -> Option<Arc<QueryResultCache<ExecutionPipelineResult>>> {
    if olap_result_cache_enabled() {
        Some(GLOBAL_OLAP_RESULT_CACHE.clone())
    } else {
        None
    }
}

/// The shared invalidation coordinator (always reachable; its result-cache arm
/// is a no-op when the flag is off because nothing was ever inserted).
pub fn olap_cache_coordinator() -> Arc<CacheInvalidationCoordinator> {
    GLOBAL_OLAP_CACHE_COORDINATOR.clone()
}

/// Tenant-scoped invalidation of the OLAP result cache after a write/DDL. Drops
/// every cached entry registered under `(tenant, normalize_table_key(table))`.
/// The table name is normalized here (matching the read-side dependency keys,
/// which are `snapshot.tables` keys) so callers can pass the raw parsed name.
/// Default-OFF: a cheap no-op when the cache flag is unset. Call from the
/// pgwire INSERT/UPDATE/DELETE/DDL success paths (mandate #16b).
pub async fn invalidate_olap_result_cache_for(tenant: &str, table: &str) {
    if olap_result_cache().is_some() {
        let normalized = normalize_table_key(table);
        olap_cache_coordinator()
            .invalidate_collection(tenant, &normalized, &[])
            .await;
    }
}

/// Cheap canonical-WAL LSN read — the Strong-freshness anchor (ADR-051 D2 /
/// ADR-046). Returns `0` when WAL-manifest tracking is unavailable, which makes
/// Strong lookups bypass (never serve) — the safe default. Same source as the
/// vector-search path (`services::operations::vectors::legacy`).
async fn current_canonical_lsn() -> u64 {
    match crate::storage::persistence::write_ahead_log::manifest::get_service() {
        Some(svc) => svc.current_lsn().await,
        None => 0,
    }
}

/// Stamp a freshly-executed OLAP result into the result cache under its
/// structural key, with the LSN it was computed at. No-op when the cache is
/// disabled or LSN tracking is unavailable (`current_lsn == 0` → never
/// Strong-eligible, so don't waste a slot).
fn populate_olap_result_cache(
    cache: Option<&QueryResultCache<ExecutionPipelineResult>>,
    skey: &StructuralKey,
    lsn: u64,
    result: ExecutionPipelineResult,
    deps_table_keys: &HashMap<String, PreparedTable>,
) {
    let Some(cache) = cache else {
        return;
    };
    let computed_at_lsn = (lsn != 0).then_some(lsn);
    let deps: Vec<String> = deps_table_keys.keys().cloned().collect();
    if let Err(e) = cache.insert_fresh(skey.clone(), result, deps, computed_at_lsn) {
        tracing::debug!(
            target: "proximadb::pgwire::result_cache",
            error = ?e,
            "failed to cache OLAP result"
        );
    }
}

// =========================================================================
// Public entry point
// =========================================================================

/// Try to run `sql` through the new pipeline.
///
/// Returns:
///
/// - `None` — SQL didn't lower cleanly
///   (the caller should fall through to the legacy SQL path).
/// - `Some(Ok(result))` — pipeline executed; caller should emit
///   the result to the pgwire client.
/// - `Some(Err(msg))` — pipeline reached execution and failed;
///   caller should report a pgwire `ERROR` to the client.
///
/// ADR-018 Phase 2: New pipeline is enabled by default for SELECT queries
/// that can be lowered by the relational frontend. This provides proper
/// multi-column ORDER BY support and other relational features.
/// Set PROXIMADB_PGWIRE_RELATIONAL_PIPELINE=0 to disable and force legacy path.
pub async fn try_run_select(
    sql: &str,
    dml: Option<&Arc<DmlService>>,
    #[cfg_attr(not(feature = "datafusion-integration"), allow(unused_variables))]
    vector_ops: Option<Arc<dyn proximadb_runtime::VectorOpsPort>>,
    #[cfg_attr(not(feature = "datafusion-integration"), allow(unused_variables))] graph_ops: Option<
        Arc<dyn proximadb_graph_query::service::GraphQueryReadService>,
    >,
    identity: PortIdentity<'_>,
    controls: ExecutionControls,
    // pgwire passes `true`: only joins/GROUP BY/aggregates/set-ops engage the
    // real-data relational engine, leaving simple SELECTs on pgwire's hardened
    // legacy path. Off-pgwire callers (gRPC/REST `ExecuteQuery`, TD-121) have no
    // such legacy single-table path, so they pass `false` to route *any*
    // resolvable relational SELECT through this tenant-scoped pipeline; queries
    // whose tables don't resolve return `None` and fall back to the caller's
    // vector/graph engine.
    require_engagement: bool,
    // Namespace / schema (pgwire `search_path`) — a structural key component so
    // identical SQL under different namespaces never collides.
    namespace: Option<&str>,
    // Shared OLAP result cache (default-OFF — `None` when
    // `PROXIMADB_QUERY_RESULT_CACHE` is unset). A hit short-circuits routing +
    // execution; a miss falls through normally and the result is stamped on the
    // real-data success paths.
    result_cache: Option<&QueryResultCache<ExecutionPipelineResult>>,
) -> Option<Result<ExecutionPipelineResult, String>> {
    let identity = identity.into_owned();
    let tenant = identity.tenant_id.as_deref();
    // Governed identities must reach DML's canonical row-policy seam. The
    // process-local demo engine, subject-agnostic result cache, and bare-Parquet
    // engines cannot currently apply that predicate and are therefore
    // ineligible rather than silently widening access.
    #[cfg(feature = "abac-policy")]
    let relational_abac_required =
        dml.is_some_and(|service| service.relational_abac_required(identity.as_borrowed()));
    #[cfg(not(feature = "abac-policy"))]
    let relational_abac_required = false;
    // TD-064: the connection's tenant scopes every snapshot read to the tenant's
    // record partition (carried into the SnapshotCatalog → DmlTableReader).
    let tenant_ctx: Option<TenantContext> = tenant.filter(|t| !t.is_empty()).map(|tenant| {
        let mut context = TenantContext::for_tenant_id(tenant);
        context.tenant_stable_id = identity.tenant_stable_id;
        context
    });

    // TD-XMODAL-4: pgvector operators are a logical vector source in this
    // pipeline, not a protocol-level string-dispatch escape hatch. Parse them
    // before relational catalog lowering: a native vector collection need not
    // also masquerade as a relational table. Once an operator is recognized,
    // every lowering/execution failure is surfaced instead of falling through
    // to the legacy single-table parser or returning an empty success.
    match lower_pgvector_select(sql) {
        Ok(Some(plan)) => {
            let Some(vector_ops) = vector_ops else {
                return Some(Err("vector search service is unavailable".to_string()));
            };
            return Some(
                execute_pgvector_plan(
                    plan,
                    dml,
                    vector_ops,
                    identity.as_borrowed(),
                    tenant_ctx.as_ref(),
                    controls,
                )
                .await,
            );
        }
        Ok(None) => {}
        Err(error) => return Some(Err(error)),
    }

    // ADR-018 Phase 2: Allow opting out with explicit "0" value
    if std::env::var("PROXIMADB_PGWIRE_RELATIONAL_PIPELINE")
        .ok()
        .as_deref()
        == Some("0")
    {
        tracing::debug!(
            target: "proximadb::pgwire::new_pipeline",
            "PROXIMADB_PGWIRE_RELATIONAL_PIPELINE=0; skipping new pipeline"
        );
        return None;
    }

    // OLAP result cache lookup (ADR-051 D2 / mandate #16c). Strong freshness:
    // a hit is served ONLY when the entry was computed at the current canonical
    // WAL LSN, so any write since → guaranteed miss → read-after-write correct.
    // `cache_ctx` carries the (key, lsn) pair so the real-data success paths
    // below can stamp the result without recomputing it.
    let cache_ctx: Option<(StructuralKey, u64)> = if relational_abac_required {
        None
    } else if let Some(cache) = result_cache {
        let current_lsn = current_canonical_lsn().await;
        let skey = StructuralKey::new(
            tenant.unwrap_or(""),
            namespace.unwrap_or(""),
            QueryKey::from_sql(sql),
        );
        if let Some(cached) = cache.get_fresh(&skey, &VectorFreshnessMode::Strong, current_lsn) {
            tracing::debug!(
                target: "proximadb::pgwire::result_cache",
                tenant = tenant.unwrap_or(""),
                lsn = current_lsn,
                "OLAP result cache hit — short-circuiting route + execution"
            );
            return Some(Ok(cached.result.clone()));
        }
        Some((skey, current_lsn))
    } else {
        None
    };

    // TD-OLAP-4 result-path probe: time ALL of `try_run_select` before execution
    // — including the path-1 engine-catalog `lower_sql` attempt below, which a
    // DataFusion-routed query pays and fails before reaching the path-2 setup.
    // The earlier per-query floor decomposition attributed ~0 ms to the path-2
    // setup alone; starting the clock here tests whether the ~46 ms remainder is
    // the (previously untimed) path-1 lowering attempt.
    let setup_start = std::time::Instant::now();
    // 1) Existing in-memory-engine path (preserves the demo table and any
    //    engine-resident tables). Only engages when the SQL lowers against the
    //    engine's own catalog; a catalog miss falls through to the real-data
    //    path below.
    let engine = GLOBAL_ENGINE.clone();
    if !relational_abac_required && let Ok(logical) = lower_sql(sql, &EngineCatalog(engine.clone()))
    {
        let factory = EngineReaderFactory::new(engine);
        let resolver = StaticCapabilities {
            caps: ReaderCapabilities::full(),
            pk_columns: Vec::new(),
            secondary_columns: Vec::new(),
        };
        return Some(run_plan(&factory, resolver, logical, controls).await);
    }

    // 2) Real-data path (gated, additive). Engage ONLY for queries the legacy
    //    single-table path can't serve — joins / GROUP BY / aggregates / HAVING
    //    / set-ops — leaving simple SELECTs on the (hardened) legacy path.
    let dml = dml?;
    let statements = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return None;
    };
    // Engaged = a relational shape the legacy single-table path can't serve
    // (joins / GROUP BY / aggregates / set-ops). pgwire keeps simple SELECTs on
    // its legacy path (`require_engagement = true`); off-pgwire callers route any
    // resolvable relational SELECT here (`require_engagement = false`).
    if require_engagement && !query_engages_relational_engine(query) {
        return None;
    }

    // Pre-resolve every referenced table's SCHEMA from xCatalog (the sync
    // `CatalogLookup`/`ReaderFactory` traits can't await). Rows are fetched
    // lazily per scan in `DmlTableReader::open`, with the executor's
    // projection/predicate/limit pushed into storage.
    let mut names = Vec::new();
    let cte_names = collect_table_names(query, &mut names);
    let mut tables: HashMap<String, PreparedTable> = HashMap::new();
    // P1: per-table Parquet location (object-store backed), populated only under the
    // `datafusion-integration` feature so the OLAP DataFusion route is never taken
    // (nor advertised) when the build can't honor it.
    #[cfg(feature = "datafusion-integration")]
    let mut parquet_loc_by_key: HashMap<String, String> = HashMap::new();
    // ADR-058 D5 / §9.A: per-table footer-stats trust, parallel to
    // `parquet_loc_by_key`. Drives whether the native route may answer
    // COUNT/MIN/MAX from the parquet FOOTER (Trusted) or must value-scan
    // (Untrusted — external/federated writer's footer is not trusted).
    #[cfg(feature = "datafusion-integration")]
    let mut parquet_trust_by_key: HashMap<String, StatsTrust> = HashMap::new();
    // ADR-025: per-table OLAP read-merge params (snapshot_lsn + PK column) for the
    // opted-in parquet-backed tables; empty ⇒ all reads stay on the bare Parquet path.
    #[cfg(feature = "datafusion-integration")]
    let mut olap_delta_tables: HashMap<
        String,
        crate::query::execution::olap_delta_merge::OlapDeltaTable,
    > = HashMap::new();
    // TD-OLAP-1 Test 2.1: per-table PAX-backed flag (has ProximaBlock format layout)
    let mut pax_tables: std::collections::HashSet<String> = std::collections::HashSet::new();
    for raw in &names {
        let key = normalize_table_key(raw);
        // TD-185a: a WITH-defined name is resolved by the SQL engine, not the
        // catalog — it must never fail snapshot preparation.
        if cte_names.contains(&key) {
            continue;
        }
        if tables.contains_key(&key) {
            continue;
        }
        match dml.resolve_relational_schema(raw, tenant).await {
            Ok(catalog_schema) => {
                #[cfg(feature = "datafusion-integration")]
                if let Some(location) = catalog_table_is_parquet_backed(&catalog_schema) {
                    parquet_loc_by_key.insert(key.clone(), location);
                    parquet_trust_by_key.insert(key.clone(), parquet_stats_trust(&catalog_schema));
                    if let Some(params) = olap_delta_table_params(&catalog_schema) {
                        olap_delta_tables.insert(key.clone(), params);
                    }
                }
                // TD-OLAP-1 Test 2.1: Track PAX-backed tables from catalog signals
                if catalog_table_is_pax_backed(&catalog_schema) {
                    pax_tables.insert(key.clone());
                }
                tables.insert(key, PreparedTable::from_catalog(raw, &catalog_schema));
            }
            Err(e) => {
                tracing::debug!(
                    target: "proximadb::pgwire::new_pipeline",
                    "relational schema resolve for `{raw}` failed: {e}; falling through to legacy"
                );
                return None;
            }
        }
    }

    // Course-correction §5: compute the per-query Parquet-backed signal — only ever
    // true under `datafusion-integration`, where the DataFusion destination is
    // compiled — and feed it into the canonical `ComputeScheduler` so the ROUTE
    // DECISION and the physical DISPATCH come from ONE place. Mixed Parquet+native
    // queries report `false` (cross-engine join is a later phase) → Volcano.
    #[cfg(feature = "datafusion-integration")]
    let parquet_backed = !relational_abac_required
        && !tables.is_empty()
        && tables.keys().all(|k| parquet_loc_by_key.contains_key(k));
    #[cfg(not(feature = "datafusion-integration"))]
    let parquet_backed = false;

    // TD-OLAP-1 Test 2.1: compute the per-query PAX-backed signal from catalog
    // PAX format detection. A table is PAX-backed if it has ANY storage layout
    // with ProximaBlock format (the native vector+relational hybrid format).
    // Mixed PAX+non-PAX queries report `false` (cross-engine join is later phase).
    let pax_backed = !relational_abac_required
        && !tables.is_empty()
        && tables.keys().all(|k| pax_tables.contains(k));

    // C4 Phase-2b: refine the shape-class with real fan-out / cardinality for
    // hot Parquet-backed tables, peeked from the in-memory stat cache warmed by
    // table opens (zero route-time I/O — a cold/never-scanned table stays
    // `Unknown`, backward-compatible). Finer classes let the cost model
    // discriminate within the OLAP-over-Parquet band where a Native↔DataFusion
    // override is freshness-safe.
    #[cfg(feature = "datafusion-integration")]
    let (partition_fanout, cardinality) = if parquet_backed {
        let locations: Vec<String> = parquet_loc_by_key.values().cloned().collect();
        crate::query::route_cost_model::classify_table_shapes(&locations)
    } else {
        (Default::default(), Default::default())
    };
    #[cfg(not(feature = "datafusion-integration"))]
    let (partition_fanout, cardinality) = (Default::default(), Default::default());

    // AST cardinality fallback: when the footer-warmed stat is `Unknown` (native
    // storage, or a cold Parquet table never yet scanned), derive a cheap
    // syntax-only hint (LIMIT / scalar aggregate) so the cost-model shape-class
    // still discriminates instead of collapsing to the coarse class. Pure AST
    // walk — zero route-time I/O (co-design P5).
    let cardinality = if cardinality == crate::query::compute_scheduler::CardinalityClass::Unknown {
        query_cardinality_hint(query)
    } else {
        cardinality
    };

    let operation_class = query_operation_class(query);
    // TD-EXEC-2 Slice 3: the geometry tier — bucketed depth × breaker bands
    // estimated from the AST, refining the cost-model class so EWMA cells
    // accumulate per shape⊕geometry (the dimension TD-OLAP-15 measured engines
    // win/lose on).
    let geometry = query_geometry_class(query);
    let shape = crate::query::compute_scheduler::QueryShape {
        engages_relational: true,
        parquet_backed,
        // TD-OLAP-1 Test 2.1: Route flip from catalog signals (no longer hard-coded)
        pax_backed,
        partition_fanout,
        cardinality,
        operation_class,
        geometry,
        join_bearing: query_join_bearing(query),
    };
    // TD-ROUTE-3: read the env master switches ONCE here (both are OnceLock-cached)
    // and thread them into the decision so `route_select_advised` — not the
    // dispatch — decides whether a native/duckdb primary is engaged. The decision
    // core stays env-free / unit-testable.
    let route_flags = crate::query::compute_scheduler::RouteFlags {
        native_route: crate::query::execution::native_engine::native_route_enabled(),
        // `duckdb_engine` compiles only under the `duckdb` feature; without it the
        // switch is inert (false), matching the absent DuckDB dispatch arm.
        #[cfg(feature = "duckdb")]
        duckdb_route: crate::query::execution::duckdb_engine::duckdb_route_enabled(),
        #[cfg(not(feature = "duckdb"))]
        duckdb_route: false,
    };
    let decision = crate::query::compute_scheduler::ComputeScheduler::new().route_select_advised(
        shape,
        // C4: observe-mode advisory from the trace-driven cost model — augments
        // the reason for telemetry/EXPLAIN, never changes the backend.
        Some(&crate::query::route_cost_model::GLOBAL_ROUTE_COST_MODEL),
        route_flags,
    );
    tracing::debug!(
        target: "proximadb::compute_route",
        backend = ?decision.backend,
        workload = ?decision.workload_profile,
        source = decision.source.as_str(),
        reason = %decision.reason,
        "{}",
        decision.explain_line()
    );

    // P1 dispatch driven by the AUTHORITATIVE scheduler decision (TD-ROUTE-3).
    // `route_select_advised` has already folded eligibility (ADR-058) + the env
    // master switches + any cost-model override into `decision.backend`, so this
    // block does NOT re-decide — it is a thin `match decision.backend` over the
    // parquet-OLAP engines, each PRIMARY owning its own fall-through to the shared
    // DataFusion floor. The DataFusion arm exists only under the feature (and
    // `parquet_backed`), so a default build never enters it and stays Volcano.
    //
    // The outer guard restricts entry to the three parquet-OLAP backends; any
    // other `decision.backend` (never produced for a parquet-OLAP shape) falls
    // through to the Volcano path below — exactly as before.
    #[cfg(feature = "datafusion-integration")]
    if !relational_abac_required
        && parquet_backed
        && matches!(
            decision.backend,
            crate::query::table_write_plan::ComputeBackend::DataFusionLocal
                | crate::query::table_write_plan::ComputeBackend::Native
                | crate::query::table_write_plan::ComputeBackend::DuckDbCompat
        )
    {
        // Shared prep for the DuckDB primary + the DataFusion floor: per-table
        // Parquet locations and (ADR-058 D5 / §9.A) per-table footer-stats trust,
        // keyed by table_name (matching `parquet_tables`) and derived from
        // `parquet_trust_by_key` (keyed by the canonical table key). Drives adapter
        // elision gating.
        let parquet_tables: Vec<(String, String)> = tables
            .iter()
            .map(|(k, t)| (t.table_name.clone(), parquet_loc_by_key[k].clone()))
            .collect();
        let parquet_table_trust: HashMap<String, StatsTrust> = tables
            .iter()
            .filter_map(|(k, t)| {
                parquet_trust_by_key
                    .get(k)
                    .map(|tr| (t.table_name.clone(), *tr))
            })
            .collect();

        // PRIMARY dispatch — a thin match over the authoritative backend. Each arm
        // attempts its engine and RETURNS on success; on decline it falls through
        // to the shared DataFusion floor below (which re-stamps the io_trace route
        // so cost cells attribute to the engine that actually served). Eligibility
        // + enablement were decided upstream, so reaching an arm IS the
        // authorization to attempt it — no re-decision here (the TD-ROUTE-3 fix).
        match decision.backend {
            // TD-OLAP-4 "favor native by operation": the native vectorized engine
            // over the SAME external parquet, for the footer-elidable `COUNT(*)` /
            // `MIN` / `MAX` (MetadataElidable) and narrow unfiltered scalar
            // aggregates (ScalarAggregate) it measurably wins on. Its own fine
            // plan-shape gates (single-scan / width / trust) live inside
            // `try_native_over_parquet`; any decline falls to the floor.
            crate::query::table_write_plan::ComputeBackend::Native => {
                let native_snapshot = SnapshotCatalog {
                    dml: dml.clone(),
                    tables: tables.clone(),
                    tenant: tenant_ctx.clone(),
                    identity: identity.clone(),
                };
                // Setup (planning/route) time is attributed here; the native engine
                // records its own `native-vectorized` compute sample inside the run.
                crate::observability::io_trace::record_setup_ms(
                    setup_start.elapsed().as_millis() as u64
                );
                if let Some(result) = try_native_over_parquet(
                    sql,
                    &native_snapshot,
                    &parquet_loc_by_key,
                    &parquet_trust_by_key,
                )
                .await
                {
                    tracing::debug!(
                        target: "proximadb::compute_route",
                        "native-over-parquet served as PRIMARY (op={:?}); DataFusion floor bypassed",
                        operation_class
                    );
                    if let Some((skey, lsn)) = &cache_ctx {
                        populate_olap_result_cache(
                            result_cache,
                            skey,
                            *lsn,
                            result.clone(),
                            &native_snapshot.tables,
                        );
                    }
                    return Some(Ok(result));
                }
                // Native declined → fall through to the DataFusion floor below.
            }
            // ADR-059: DuckDB-Local as PRIMARY for the join/agg-shaped parquet route
            // (the measured 100–700× DataFusion join-plan gap, TD-OLAP-15/TD-OLAP-2).
            // The remaining gate is the ADR-025 TRUST probe: DuckDB reads bare
            // parquet and cannot reconcile a post-snapshot WAL delta, so it serves
            // only when every referenced table's delta is EMPTY (the delta-merging
            // DataFusion floor serves everything else; a probe error fails closed to
            // the floor). The probe is the TD-OLAP-17 per-collection high-water
            // check DataFusion itself pays at table open, so the common no-writes
            // case costs no scan.
            #[cfg(feature = "duckdb")]
            crate::query::table_write_plan::ComputeBackend::DuckDbCompat => {
                let duckdb_delta_clean = {
                    use crate::query::execution::olap_delta_merge::OlapDeltaSource;
                    let mut clean = true;
                    for (tkey, params) in &olap_delta_tables {
                        match dml
                            .changed_oids_since(tkey, params.snapshot_lsn, tenant)
                            .await
                        {
                            Ok(oids) if oids.is_empty() => {}
                            _ => {
                                clean = false;
                                break;
                            }
                        }
                    }
                    clean
                };
                if duckdb_delta_clean {
                    let duck_context = QueryExecutionContext {
                        parquet_tables: parquet_tables.clone(),
                        parquet_table_trust: parquet_table_trust.clone(),
                        identity: identity.clone(),
                        controls: controls.clone(),
                        ..Default::default()
                    };
                    crate::observability::io_trace::record_setup_ms(
                        setup_start.elapsed().as_millis() as u64,
                    );
                    match execute_sql_with_backend(
                        crate::query::table_write_plan::ComputeBackend::DuckDbCompat,
                        sql,
                        duck_context,
                    )
                    .await
                    {
                        Ok(result) => {
                            crate::observability::io_trace::record_route(
                                &crate::query::route_cost_model::shape_class(&shape),
                                "DuckDbCompat",
                            );
                            tracing::debug!(
                                target: "proximadb::compute_route",
                                "duckdb-local served as PRIMARY (join/agg parquet shape); \
                                 DataFusion floor bypassed"
                            );
                            if let Some((skey, lsn)) = &cache_ctx {
                                populate_olap_result_cache(
                                    result_cache,
                                    skey,
                                    *lsn,
                                    result.clone(),
                                    &tables,
                                );
                            }
                            return Some(Ok(result));
                        }
                        Err(e) => {
                            tracing::debug!(
                                target: "proximadb::compute_route",
                                "duckdb-local PRIMARY declined ({e}); DataFusion floor serves"
                            );
                        }
                    }
                }
                // Not delta-clean or DuckDB declined → fall through to the floor.
            }
            // DataFusionLocal (or a DuckDbCompat decision in a non-`duckdb` build) →
            // the floor serves directly, no primary attempt.
            _ => {}
        }

        // === DataFusion floor (shared) ===
        // Reached directly for a DataFusionLocal decision, or by fall-through when a
        // Native/DuckDB PRIMARY declined. This IS the correctness floor for the
        // OLAP-over-parquet route.
        let context = QueryExecutionContext {
            parquet_tables,
            parquet_table_trust,
            vector_ops,
            graph_ops,
            identity: identity.clone(),
            // ADR-025: reconcile opted-in parquet-backed tables with their
            // post-snapshot WAL delta at scan time. `None` (no opted-in table)
            // keeps the legacy bare-Parquet read (default-OFF).
            olap_delta: if olap_delta_tables.is_empty() {
                None
            } else {
                Some(crate::query::execution::olap_delta_merge::OlapDeltaConfig {
                    source: dml.clone(),
                    tables: olap_delta_tables,
                })
            },
            // Full ANSI SQL over pgwire: when the shared relational frontend can't
            // yet lower a query (e.g. typed DATE literals, certain subquery/window
            // shapes), execute it through DataFusion's own ANSI SQL planner over the
            // same registered Parquet tables. The query still routes through pgwire →
            // ComputeScheduler → the DataFusion engine; only the lowering frontend
            // differs. The shared logical plane stays the fast path where it works.
            allow_engine_sql_fallback: true,
            controls: controls.clone(),
        };
        // ADR-030 / TD-158: time the DataFusion execution so the always-on billing
        // observer can attribute KRU to this engine at scope close. `record_compute_ms`
        // no-ops outside an `io_trace` scope.
        crate::observability::io_trace::record_setup_ms(setup_start.elapsed().as_millis() as u64);
        let started = std::time::Instant::now();
        let floor_backend = crate::query::table_write_plan::ComputeBackend::DataFusionLocal;
        // Fold-attribution correctness (TD-ROUTE-3): if a Native/DuckDB PRIMARY was
        // decided but declined above, the floor is about to serve — re-stamp the
        // route to DataFusion ("last write wins") so the declined engine's cost
        // cells are not poisoned with DataFusion's measured quantities. A direct
        // DataFusionLocal decision was already stamped by `finalize_route`, so the
        // guard skips the redundant re-stamp.
        if !matches!(
            decision.backend,
            crate::query::table_write_plan::ComputeBackend::DataFusionLocal
        ) {
            crate::observability::io_trace::record_route(
                &crate::query::route_cost_model::shape_class(&shape),
                &crate::query::compute_scheduler::backend_label(&floor_backend),
            );
        }
        let engine_result = execute_sql_with_backend(floor_backend.clone(), sql, context).await;
        crate::observability::io_trace::record_compute_ms(
            &crate::query::compute_scheduler::backend_label(&floor_backend),
            started.elapsed().as_millis() as u64,
        );
        // TD-OLAP-4 (engine dimension): optional native SHADOW — run the same query
        // on the native vectorized engine over the same parquet, purely to record a
        // `native-vectorized` compute sample next to DataFusion's for the trace.
        // Default-OFF, benchmark-only; the result is discarded (DataFusion's stands)
        // and any failure is swallowed, so it never affects correctness or latency
        // of the served query.
        if crate::query::execution::native_engine::native_shadow_enabled() {
            let probe_snapshot = SnapshotCatalog {
                dml: dml.clone(),
                tables: tables.clone(),
                tenant: tenant_ctx.clone(),
                identity: identity.clone(),
            };
            shadow_probe_native_parquet(
                sql,
                &probe_snapshot,
                &parquet_loc_by_key,
                &parquet_trust_by_key,
            )
            .await;
        }
        return Some(match engine_result {
            Ok(result) => {
                if let Some((skey, lsn)) = &cache_ctx {
                    populate_olap_result_cache(result_cache, skey, *lsn, result.clone(), &tables);
                }
                Ok(result)
            }
            Err(e) => Err(e.to_string()),
        });
    }

    let snapshot = SnapshotCatalog {
        dml: dml.clone(),
        tables,
        tenant: tenant_ctx,
        identity,
    };
    // Lower + plan via the shared planning path (so EXPLAIN discloses exactly this
    // plan). `None` → lowering declined → fall through to legacy. From the planned
    // physical onward, errors are real and surface to the client.
    let physical = match plan_over_snapshot(sql, &snapshot)? {
        Ok(p) => p,
        Err(e) => return Some(Err(e)),
    };
    crate::observability::io_trace::record_setup_ms(setup_start.elapsed().as_millis() as u64);
    match execute_physical(physical, &snapshot, controls).await {
        Ok(result) => {
            if let Some((skey, lsn)) = &cache_ctx {
                populate_olap_result_cache(
                    result_cache,
                    skey,
                    *lsn,
                    result.clone(),
                    &snapshot.tables,
                );
            }
            Some(Ok(result))
        }
        Err(e) => Some(Err(e)),
    }
}

/// Plan + build + drain an executor for `logical` against `factory`, using
/// `resolver` for capability/PK-lookup planning (engine path → `StaticCapabilities`,
/// real-data path → `SnapshotCapabilities`).
async fn run_plan<F: ReaderFactory, R: CapabilityResolver>(
    factory: &F,
    resolver: R,
    logical: proximadb_relational_algebra::LogicalNode,
    controls: ExecutionControls,
) -> Result<ExecutionPipelineResult, String> {
    let planner = Planner::new(resolver);
    let physical = plan_instrumented(&planner, logical)?;
    execute_physical(physical, factory, controls).await
}

/// Plan `logical` and record the TD-EXEC-2 Slice-1 observe-only trace fields into
/// the active io_trace scope: planning wall ms, the planner recursion's measured
/// stack high-water mark (via `stack_probe`), and the served plan's geometry
/// vector (`measure_geometry`, iterative — safe on any depth). This is the
/// plan→execute seam the calibration ledger reads; it makes no decision and
/// changes no behavior, and every record call no-ops outside an io_trace scope.
fn plan_instrumented<R: CapabilityResolver>(
    planner: &Planner<R>,
    logical: proximadb_relational_algebra::LogicalNode,
) -> Result<PhysicalPlan, String> {
    let plan_start = std::time::Instant::now();
    let (planned, stack_hwm) = stack_probe::probe(|| planner.plan(logical));
    let physical = planned.map_err(|e| format!("plan: {e}"))?;
    crate::observability::io_trace::record_plan_ms(plan_start.elapsed().as_millis() as u64);
    if stack_hwm > 0 {
        crate::observability::io_trace::record_stack_hwm(stack_hwm);
    }
    let g = measure_geometry(&physical);
    let ops: Vec<(&str, u64)> = g
        .op_histogram
        .iter()
        .map(|(kind, count)| (kind.as_str(), u64::from(*count)))
        .collect();
    crate::observability::io_trace::record_plan_geometry(
        g.max_depth as u64,
        g.node_count as u64,
        g.leaf_count as u64,
        g.max_fanout as u64,
        u64::from(g.blocking_count),
        &ops,
    );
    Ok(physical)
}

/// Build + open + drain an executor for an already-planned `physical` plan.
/// Shared tail of `run_plan` and the real-data path so EXPLAIN (which plans
/// without executing) and execution run the SAME `PhysicalPlan`.
async fn execute_physical<F: ReaderFactory>(
    physical: PhysicalPlan,
    factory: &F,
    controls: ExecutionControls,
) -> Result<ExecutionPipelineResult, String> {
    if plan_has_raw_cross_join(&physical) {
        return Err(
            "0A000: native execution declined an unnormalized cross join; use an explicit equi-join or a vectorized backend"
                .to_string(),
        );
    }
    // ADR-030 / TD-158: feed the per-query I/O trace the native engine's compute
    // time so the always-on billing observer can emit KRU(tenant, "native") at
    // scope close. `record_compute_ms` no-ops outside an `io_trace` scope.
    let started = std::time::Instant::now();
    let result = NativeVolcanoEngine::execute_physical(physical, factory, controls)
        .await
        .map_err(|e| e.to_string());
    crate::observability::io_trace::record_compute_ms(
        "native",
        started.elapsed().as_millis() as u64,
    );
    result
}

/// Like [`execute_physical`] but meters every operator (EXPLAIN ANALYZE): returns the
/// result plus per-operator [`NodeMetric`]s in pre-order (aligned with
/// `explain_physical`'s line order). The executor tree is dropped before reading the
/// metrics so each `MeteredExec` has flushed its counters.
async fn execute_physical_metered<F: ReaderFactory>(
    physical: PhysicalPlan,
    factory: &F,
) -> Result<(ExecutionPipelineResult, Vec<NodeMetric>), String> {
    if plan_has_raw_cross_join(&physical) {
        return Err(
            "0A000: native execution declined an unnormalized cross join; use an explicit equi-join or a vectorized backend"
                .to_string(),
        );
    }
    NativeVolcanoEngine::execute_physical_metered(physical, factory, ExecutionControls::default())
        .await
        .map_err(|e| e.to_string())
}

/// Availability boundary for TD-REL-LOWER-2: after planner normalization,
/// Volcano must never enumerate a raw cross product. Equi predicates should
/// already have become inner/hash joins; remaining CROSS nodes fail loudly.
fn plan_has_raw_cross_join(plan: &PhysicalPlan) -> bool {
    match plan {
        PhysicalPlan::Join {
            left, right, kind, ..
        } => {
            *kind == proximadb_relational_algebra::JoinKind::Cross
                || plan_has_raw_cross_join(left)
                || plan_has_raw_cross_join(right)
        }
        PhysicalPlan::Filter { input, .. }
        | PhysicalPlan::Project { input, .. }
        | PhysicalPlan::Aggregate { input, .. }
        | PhysicalPlan::Sort { input, .. }
        | PhysicalPlan::Limit { input, .. }
        | PhysicalPlan::Distinct { input, .. }
        | PhysicalPlan::AssertMaxOneRow { input } => plan_has_raw_cross_join(input),
        PhysicalPlan::Union { inputs, .. } => inputs.iter().any(plan_has_raw_cross_join),
        PhysicalPlan::SetOp { left, right, .. } => {
            plan_has_raw_cross_join(left) || plan_has_raw_cross_join(right)
        }
        PhysicalPlan::Scan { .. } | PhysicalPlan::Values { .. } => false,
    }
}

/// Lower + plan a SELECT over an already-prepared `SnapshotCatalog` to a Volcano
/// `PhysicalPlan`. The single planning path shared by execution
/// ([`try_run_select`]) and `EXPLAIN` ([`explain_select_route_with_catalog`]) so
/// the disclosed plan is exactly the one that runs. `None` = lowering declined
/// (fall through to legacy); `Some(Err)` = a real planning error.
fn plan_over_snapshot(
    sql: &str,
    snapshot: &SnapshotCatalog,
) -> Option<Result<PhysicalPlan, String>> {
    let logical = match lower_sql(sql, snapshot) {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(
                target: "proximadb::pgwire::new_pipeline",
                "lower_sql declined `{sql}` over real data: {e}; falling through to legacy"
            );
            return None;
        }
    };
    // Per-table PK ordinals so the planner can rewrite PK-equality scans to
    // ScanAccess::PkLookup (single-column PK only — see PreparedTable).
    let pk_by_table: HashMap<String, Vec<usize>> = snapshot
        .tables
        .iter()
        .map(|(key, prepared)| (key.clone(), prepared.pk_columns.clone()))
        .collect();
    // TD-127: per-table secondary-indexed columns so the planner can rewrite a
    // non-PK equality / IN-list to ScanAccess::SecondaryLookup.
    let secondary_by_table: HashMap<String, Vec<String>> = snapshot
        .tables
        .iter()
        .map(|(key, prepared)| (key.clone(), prepared.secondary_columns.clone()))
        .collect();
    let resolver = SnapshotCapabilities {
        pk_by_table,
        secondary_by_table,
    };
    let planner = Planner::new(resolver);
    Some(plan_instrumented(&planner, logical))
}

// =========================================================================
// TD-OLAP-4 native SHADOW probe (engine dimension). Default-OFF, benchmark-only:
// run the SAME parquet SELECT on the native vectorized engine alongside DataFusion
// to record a `native-vectorized` compute sample in the io-trace. The native
// result is discarded (DataFusion's is authoritative) and all failures are
// swallowed, so the probe never changes the served result or fails a query.
// =========================================================================

/// Column names emitted by the plan's sole `Scan` node, or `None` if the plan has
/// zero or more than one scan (multi-table / no-table). Used to build the native
/// parquet reader's projection so it reads the same columns DataFusion does.
#[cfg(feature = "datafusion-integration")]
fn single_scan_columns(plan: &PhysicalPlan) -> Option<Vec<String>> {
    fn collect(plan: &PhysicalPlan, count: &mut usize, out: &mut Option<Vec<String>>) {
        match plan {
            PhysicalPlan::Scan { output_schema, .. } => {
                *count += 1;
                *out = Some(
                    output_schema
                        .columns
                        .iter()
                        .map(|c| c.name.clone())
                        .collect(),
                );
            }
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Project { input, .. }
            | PhysicalPlan::Aggregate { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Distinct { input, .. }
            | PhysicalPlan::AssertMaxOneRow { input } => collect(input, count, out),
            PhysicalPlan::Join { left, right, .. } => {
                collect(left, count, out);
                collect(right, count, out);
            }
            PhysicalPlan::Union { inputs, .. } => {
                for i in inputs {
                    collect(i, count, out);
                }
            }
            _ => {}
        }
    }
    let mut count = 0usize;
    let mut out = None;
    collect(plan, &mut count, &mut out);
    (count == 1).then_some(out).flatten()
}

/// Is this an unfiltered, ungrouped `COUNT(*)` (possibly under a `Limit`) directly
/// over a `Scan`? Then the answer is the sum of the parquet/PAX footer row counts —
/// a metadata read, no column scan. (`scan_cols` can't detect this: the relational
/// `Scan.output_schema` is full-width, so COUNT(*) would otherwise read all columns.)
#[cfg(feature = "datafusion-integration")]
fn is_pure_count_star(plan: &PhysicalPlan) -> bool {
    use proximadb_relational_algebra::AggregateExpr;
    match plan {
        // A `COUNT(*)` lowers to `Project(Aggregate(Scan))` (the Project names/orders
        // the output) — recurse through both Project and Limit to the Aggregate.
        PhysicalPlan::Limit { input, .. } | PhysicalPlan::Project { input, .. } => {
            is_pure_count_star(input)
        }
        PhysicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
            having,
            ..
        } => {
            group_by.is_empty()
                && having.is_none()
                && !aggregates.is_empty()
                && aggregates.iter().all(|a| {
                    matches!(
                        &a.agg,
                        AggregateExpr::Count {
                            arg: None,
                            distinct: false
                        }
                    )
                })
                // MUST be an UNFILTERED scan: a pushed-down `Scan.predicate`
                // (`WHERE AdvEngineID<>0`, `WHERE URL LIKE …`) means the count is
                // over the FILTERED rows — footer elision would ignore the filter
                // and return the wrong (whole-table) count.
                && matches!(input.as_ref(), PhysicalPlan::Scan { predicate: None, .. })
        }
        _ => false,
    }
}

/// TD-OLAP-4 metadata elision: if the plan is an UNGROUPED, UNFILTERED aggregate
/// over a `Scan` whose aggregates are all `COUNT(*)` / `COUNT(col)` / `MIN(col)` /
/// `MAX(col)` on NUMERIC columns, build the answer row directly from the parquet
/// FOOTER (row count + per-column min/max/null_count) — zero column I/O, flat cost
/// (matching DataFusion's `AggregateStatistics`). Returns `None` (→ scan) for
/// SUM/AVG/DISTINCT, non-column args, string columns (truncated stats), or any
/// column lacking full footer coverage.
#[cfg(feature = "datafusion-integration")]
fn elidable_aggregate_batch(
    plan: &PhysicalPlan,
    table: &crate::datafusion::engine_adapters::ObjectStoreParquetTable,
    file_schema: &arrow::datatypes::SchemaRef,
) -> Option<arrow_array::RecordBatch> {
    use arrow_array::{ArrayRef, Float64Array, Int64Array};
    use proximadb_relational_algebra::AggregateExpr;
    use proximadb_relational_types::Expr;

    fn find_agg(p: &PhysicalPlan) -> Option<&PhysicalPlan> {
        match p {
            PhysicalPlan::Project { input, .. } | PhysicalPlan::Limit { input, .. } => {
                find_agg(input)
            }
            PhysicalPlan::Aggregate { .. } => Some(p),
            _ => None,
        }
    }
    let PhysicalPlan::Aggregate {
        input,
        group_by,
        aggregates,
        having,
        ..
    } = find_agg(plan)?
    else {
        return None;
    };
    if !group_by.is_empty()
        || having.is_some()
        || !matches!(
            input.as_ref(),
            PhysicalPlan::Scan {
                predicate: None,
                ..
            }
        )
    {
        return None;
    }
    // Resolve a catalog column name to the FILE column (case-insensitive — CamelCase
    // DDL over a lowercased parquet) → (parquet-stats key, Arrow type).
    let resolve = |name: &str| -> Option<(String, arrow::datatypes::DataType)> {
        file_schema
            .fields()
            .iter()
            .find(|f| f.name().eq_ignore_ascii_case(name))
            .map(|f| (f.name().clone(), f.data_type().clone()))
    };
    let rows = table.estimated_rows()?;
    let mut fields: Vec<arrow::datatypes::Field> = Vec::with_capacity(aggregates.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(aggregates.len());
    for na in aggregates {
        let (dt, arr): (arrow::datatypes::DataType, ArrayRef) = match &na.agg {
            AggregateExpr::Count {
                arg: None,
                distinct: false,
            } => (
                arrow::datatypes::DataType::Int64,
                std::sync::Arc::new(Int64Array::from(vec![rows as i64])),
            ),
            AggregateExpr::Count {
                arg: Some(Expr::Column(c)),
                distinct: false,
            } => {
                let (cn, _) = resolve(&c.name)?;
                let (_, _, nulls) = table.aggregate_numeric_bounds(&cn)?;
                (
                    arrow::datatypes::DataType::Int64,
                    std::sync::Arc::new(Int64Array::from(vec![rows.saturating_sub(nulls) as i64])),
                )
            }
            AggregateExpr::Min {
                arg: Expr::Column(c),
            }
            | AggregateExpr::Max {
                arg: Expr::Column(c),
            } => {
                let (cn, dt) = resolve(&c.name)?;
                let (lo, hi, _) = table.aggregate_numeric_bounds(&cn)?;
                let v = if matches!(na.agg, AggregateExpr::Max { .. }) {
                    hi
                } else {
                    lo
                };
                let arr = arrow::compute::cast(&Float64Array::from(vec![v]), &dt).ok()?;
                (dt, arr)
            }
            // SUM/AVG/DISTINCT/STRING_AGG/non-column → not footer-elidable.
            _ => return None,
        };
        fields.push(arrow::datatypes::Field::new(&na.name, dt, true));
        arrays.push(arr);
    }
    arrow_array::RecordBatch::try_new(
        std::sync::Arc::new(arrow::datatypes::Schema::new(fields)),
        arrays,
    )
    .ok()
}

/// Does any `Scan` in the plan carry a pushed-down `predicate`? Native's scan does
/// NOT apply `Scan.predicate` (no predicate pushdown yet), so it would count/aggregate
/// the UNFILTERED rows — a wrong result. Filtered scans route to DataFusion (which
/// applies the predicate + prunes row groups) until native gains predicate pushdown.
#[cfg(feature = "datafusion-integration")]
fn plan_scan_has_predicate(plan: &PhysicalPlan) -> bool {
    match plan {
        PhysicalPlan::Scan { predicate, .. } => predicate.is_some(),
        PhysicalPlan::Filter { input, .. }
        | PhysicalPlan::Project { input, .. }
        | PhysicalPlan::Aggregate { input, .. }
        | PhysicalPlan::Sort { input, .. }
        | PhysicalPlan::Limit { input, .. }
        | PhysicalPlan::Distinct { input, .. }
        | PhysicalPlan::AssertMaxOneRow { input } => plan_scan_has_predicate(input),
        PhysicalPlan::Join { left, right, .. } => {
            plan_scan_has_predicate(left) || plan_scan_has_predicate(right)
        }
        PhysicalPlan::Union { inputs, .. } => inputs.iter().any(plan_scan_has_predicate),
        _ => false,
    }
}

/// Does the plan contain an `Aggregate` with a non-empty `GROUP BY`? Native's
/// `HashAggregate` is in-memory with no spilling, so a high-cardinality group-by
/// (e.g. `GROUP BY UserID` over 100M rows) balloons the hash table to tens of GB
/// and OOMs. Until a bounded/spilling aggregate lands, the shadow declines grouped
/// aggregates — they route to DataFusion (its partitioned/spilling aggregation
/// handles them), which is itself the dispatch rule the trace confirms.
#[cfg(feature = "datafusion-integration")]
fn plan_has_grouped_aggregate(plan: &PhysicalPlan) -> bool {
    match plan {
        PhysicalPlan::Aggregate {
            input, group_by, ..
        } => !group_by.is_empty() || plan_has_grouped_aggregate(input),
        PhysicalPlan::Filter { input, .. }
        | PhysicalPlan::Project { input, .. }
        | PhysicalPlan::Sort { input, .. }
        | PhysicalPlan::Limit { input, .. }
        | PhysicalPlan::Distinct { input, .. }
        | PhysicalPlan::AssertMaxOneRow { input } => plan_has_grouped_aggregate(input),
        PhysicalPlan::Join { left, right, .. } => {
            plan_has_grouped_aggregate(left) || plan_has_grouped_aggregate(right)
        }
        PhysicalPlan::Union { inputs, .. } => inputs.iter().any(plan_has_grouped_aggregate),
        _ => false,
    }
}

/// Run the native vectorized engine over the same external parquet DataFusion just
/// served, purely to record a per-engine compute sample. Single-table parquet only
/// (MVP); declines silently on anything native can't yet serve. Thin wrapper over
/// [`try_native_over_parquet`] that discards the result — the sample it records via
/// the io-trace is the whole point; DataFusion's answer is the one served.
#[cfg(feature = "datafusion-integration")]
async fn shadow_probe_native_parquet(
    sql: &str,
    snapshot: &SnapshotCatalog,
    parquet_loc_by_key: &HashMap<String, String>,
    parquet_trust_by_key: &HashMap<String, StatsTrust>,
) {
    let _ = try_native_over_parquet(sql, snapshot, parquet_loc_by_key, parquet_trust_by_key).await;
}

/// Serve `sql` on the native vectorized engine over the SAME external parquet
/// DataFusion would read, returning `Some(result)` when native handled it as the
/// PRIMARY backend (its result is wire-identical to DataFusion's for the routed
/// shapes) or `None` to route to DataFusion. This is the single native-over-parquet
/// code path shared by the production route arm ([`try_run_select`], gated on
/// `native_route_enabled`) and the benchmark shadow probe (gated on
/// `native_shadow_enabled`) — every eligibility guard and source selection lives
/// here so the two callers can never diverge (TD-OLAP-4 "favor native by
/// operation").
///
/// Eligibility (any failure → `None`, DataFusion serves): single-table parquet,
/// a single-scan plan, no grouped aggregate (native's non-spilling HashAggregate
/// OOMs on high-cardinality `GROUP BY`), no filter predicate (native has no
/// predicate pushdown, so it would wrongly aggregate unfiltered rows), and a scan
/// no wider than the width cap. The served shapes are exactly the measured native
/// wins: footer-elidable `COUNT(*)`/`MIN`/`MAX` and narrow unfiltered scalar
/// aggregates.
#[cfg(feature = "datafusion-integration")]
async fn try_native_over_parquet(
    sql: &str,
    snapshot: &SnapshotCatalog,
    parquet_loc_by_key: &HashMap<String, String>,
    parquet_trust_by_key: &HashMap<String, StatsTrust>,
) -> Option<ExecutionPipelineResult> {
    // Single-table parquet only — joins are a separately-gated native path.
    if parquet_loc_by_key.len() != 1 {
        return None;
    }
    let table_key = parquet_loc_by_key.keys().next()?;
    let location = parquet_loc_by_key.values().next()?;
    // ADR-058 D5 / §9.A: footer-stats elision (below) is only sound when the
    // writer is authoritative. External/federated parquet (FederatedRead /
    // ExternalAuthoritative / ImportedSnapshot) ⇒ Untrusted ⇒ the
    // COUNT/MIN/MAX-from-footer paths are skipped and the query is served by
    // the trust-safe value-scan. Materialized parquet (ProjectionPublication,
    // which `ALTER TABLE … MATERIALIZE` produces) is ProximaDB-generated ⇒
    // Trusted ⇒ elision stays enabled.
    let stats_trust = parquet_trust_by_key
        .get(table_key)
        .copied()
        .unwrap_or(StatsTrust::Untrusted);
    let sqlp = &sql[..sql.len().min(70)];
    // Lower the SAME SQL to the relational physical plan (decline → skip).
    let physical = match plan_over_snapshot(sql, snapshot) {
        Some(Ok(p)) => p,
        _ => {
            tracing::debug!(target: "proximadb::native_route", "native SKIP plan-declined: {sqlp}");
            return None;
        }
    };
    let Some(scan_cols) = single_scan_columns(&physical) else {
        tracing::debug!(target: "proximadb::native_route", "native SKIP not-single-scan: {sqlp}");
        return None;
    };
    // Skip grouped aggregates: native's non-spilling HashAggregate OOMs on
    // high-cardinality GROUP BY at scale. These route to DataFusion.
    // Opt-in gate (PROXIMADB_NATIVE_ALLOW_GROUPED_AGG): allow grouped aggregates
    // when the caller knows the group cardinality is safe (e.g., SF=0.01 with
    // low-cardinality GROUP BY like Q1's l_returnflag/l_linestatus → 8 groups).
    if plan_has_grouped_aggregate(&physical)
        && std::env::var("PROXIMADB_NATIVE_ALLOW_GROUPED_AGG").is_err()
    {
        tracing::debug!(target: "proximadb::native_route", "native SKIP grouped-aggregate: {sqlp}");
        return None;
    }
    // Skip filtered scans: native does not apply a pushed-down `Scan.predicate`,
    // so it would (wrongly) aggregate the unfiltered rows. Route to DataFusion.
    if plan_scan_has_predicate(&physical) {
        tracing::debug!(target: "proximadb::native_route", "native SKIP filtered-scan (no native predicate pushdown): {sqlp}");
        return None;
    }
    tracing::debug!(target: "proximadb::native_route", "native RUN: {sqlp}");
    // Open the same parquet DataFusion read (table-OPEN cache is warm from its run).
    let table = match crate::datafusion::engine_adapters::ObjectStoreParquetTable::open(location)
        .await
    {
        Ok(t) => t,
        Err(e) => {
            tracing::debug!(target: "proximadb::native_route", %e, "native: table open declined");
            return None;
        }
    };
    let (store, files, file_schema) = table.native_scan_inputs();
    // Metadata elision (TD-OLAP-4): unfiltered MIN/MAX/COUNT answered from the
    // parquet FOOTER — zero column I/O, flat cost. The single biggest native win at
    // scale (q07 MIN/MAX: a full-column vectorized scan → a footer read). Emits the
    // pre-computed row and bypasses the scan + HashAggregate entirely.
    // ADR-058 D5 / §9.A: footer-stats elision is TRUST-GATED — only Trusted
    // (ProximaDB-owned) parquet may answer from the footer; Untrusted
    // (external/federated) writer footers may be stale/absent ⇒ fall through to
    // the trust-safe value-scan below.
    if stats_trust == StatsTrust::Trusted
        && let Some(batch) = elidable_aggregate_batch(&physical, &table, &file_schema)
    {
        tracing::debug!(target: "proximadb::native_route", "native ELIDE (footer stats, trusted): {sqlp}");
        let src = Box::new(
            crate::query::execution::native_parquet_scan::StatsAggregateOperator::new(batch),
        );
        return native_result(
            crate::query::execution::native_engine::run_native_source_only(src).await,
            sqlp,
        );
    }
    // Unfiltered COUNT(*): parquet/PAX carry the row count in the FOOTER, so elide
    // the scan entirely — read footers only, emit the count (a metadata op, not a
    // scan). Detected from the PLAN, not `scan_cols`: the relational Scan schema is
    // full-width, so COUNT(*) would otherwise read all 105 columns. Trust-gated
    // (§9.A): the footer row-count is only authoritative for Trusted parquet.
    let source: Box<dyn proximadb_execution_contracts::ExecutionOperator> =
        if stats_trust == StatsTrust::Trusted && is_pure_count_star(&physical) {
            Box::new(
                crate::query::execution::native_parquet_scan::ParquetScanOperator::new_count_only(
                    store, files,
                ),
            )
        } else {
            // The relational Scan schema is full-width (no projection pushdown yet), so
            // `scan_cols` reads every column the scan declares. Native has neither
            // projection nor predicate pushdown, so a wide scan reads all columns
            // row-wise — impractically slow at scale. Cap it: route wide scans to
            // DataFusion (which prunes columns + row groups). This IS the dispatch rule
            // the trace confirms; narrow scans still sample native. (Lifting the cap
            // needs projection/predicate pushdown — the medium-tier native enabler.)
            const NATIVE_SCAN_WIDTH_CAP: usize = 8;
            if scan_cols.len() > NATIVE_SCAN_WIDTH_CAP {
                tracing::debug!(
                    target: "proximadb::native_route",
                    cols = scan_cols.len(),
                    "native SKIP wide-scan (no pushdown): {sqlp}"
                );
                return None;
            }
            // Map the plan's Scan columns → parquet leaf indices by name; build the
            // native scan output schema from the FILE fields (exact Arrow types). Sort
            // ascending so the reader's file-order `ProjectionMask::roots` matches the
            // operator schema — when the plan preserves declared column order (the
            // common case) native aligns; otherwise the native lowering declines and no
            // sample is recorded.
            let mut paired: Vec<(usize, arrow::datatypes::Field)> =
                Vec::with_capacity(scan_cols.len());
            for name in &scan_cols {
                // Match case-insensitively: the catalog/DDL may declare CamelCase
                // columns over a lowercased parquet (e.g. ClickBench). Ordinals still
                // align positionally — native ops reference columns by ordinal, not name.
                let Some((idx, field)) = file_schema
                    .fields()
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name().eq_ignore_ascii_case(name))
                else {
                    return None; // column absent from file (schema drift) → never risk a sample
                };
                paired.push((idx, field.as_ref().clone()));
            }
            paired.sort_by_key(|(i, _)| *i);
            let projection: Vec<usize> = paired.iter().map(|(i, _)| *i).collect();
            let out_schema = Arc::new(arrow::datatypes::Schema::new(
                paired.into_iter().map(|(_, f)| f).collect::<Vec<_>>(),
            ));
            Box::new(
                crate::query::execution::native_parquet_scan::ParquetScanOperator::new(
                    store,
                    files,
                    Some(projection),
                    out_schema,
                ),
            )
        };
    native_result(
        crate::query::execution::native_engine::run_native_over_parquet(&physical, source).await,
        sqlp,
    )
}

/// Interpret a native-engine run result for [`try_native_over_parquet`]: `Ok(Some)`
/// → native served it (return the result); `Ok(None)` → native declined the shape;
/// `Err` → native errored. In both non-served cases the caller routes to DataFusion.
#[cfg(feature = "datafusion-integration")]
fn native_result(
    run: Result<Option<ExecutionPipelineResult>, crate::query::execution::engine::ExecutionError>,
    sqlp: &str,
) -> Option<ExecutionPipelineResult> {
    match run {
        Ok(Some(result)) => {
            tracing::debug!(target: "proximadb::native_route", "native served: {sqlp}");
            Some(result)
        }
        Ok(None) => {
            tracing::debug!(target: "proximadb::native_route", "native declined (shape/align): {sqlp}");
            None
        }
        Err(e) => {
            tracing::debug!(target: "proximadb::native_route", %e, "native errored: {sqlp}");
            None
        }
    }
}

// =========================================================================
// P1: DataFusion OLAP route over Parquet-backed tables (course-correction §6 P1)
// Gated on `datafusion-integration`; when the feature is off none of this is
// compiled and `try_run_select` behaves exactly as before (Volcano only).
// =========================================================================

/// If `schema` is backed by an external Parquet file (object store / open format),
/// return its location URI. Detects the `CatalogStorageLayout` the route branches on:
/// `physical_format == Parquet` + a read-federation/external authority + a location.
#[cfg(feature = "datafusion-integration")]
fn catalog_table_is_parquet_backed(
    schema: &proximadb_catalog::CatalogTableSchema,
) -> Option<String> {
    use proximadb_catalog::{CatalogAuthorityMode, CatalogPhysicalFormat};
    schema.storage_layouts.iter().find_map(|layout| {
        let format_ok = matches!(layout.physical_format, CatalogPhysicalFormat::Parquet);
        let authority_ok = matches!(
            layout.authority,
            CatalogAuthorityMode::FederatedRead
                | CatalogAuthorityMode::ExternalAuthoritative
                | CatalogAuthorityMode::ImportedSnapshot
                | CatalogAuthorityMode::ExportedPublication
                | CatalogAuthorityMode::ProjectionPublication
        );
        if format_ok && authority_ok {
            layout.location.clone()
        } else {
            None
        }
    })
}

/// Check if a catalog table is PAX-backed (has ProximaBlock format storage layout).
///
/// Returns `true` if the table has ANY storage layout with `ProximaBlock` format,
/// `false` otherwise. This is the PAX analog of `catalog_table_is_parquet_backed`
/// but returns a bool instead of an optional location (PAX location is derived
/// via `DrPathBuilder`, not catalog.location).
pub(crate) fn catalog_table_is_pax_backed(schema: &proximadb_catalog::CatalogTableSchema) -> bool {
    use proximadb_catalog::CatalogPhysicalFormat;
    schema
        .storage_layouts
        .iter()
        .any(|layout| matches!(layout.physical_format, CatalogPhysicalFormat::ProximaBlock))
}

/// Derive a parquet table's [`StatsTrust`] from its catalog authority. The ONLY
/// untrusted writers are external sources ProximaDB did not generate —
/// `FederatedRead` / `ExternalAuthoritative` / `ImportedSnapshot` (their parquet
/// footer stats may be stale or absent ⇒ [`StatsTrust::Untrusted`]). Everything
/// ProximaDB generated — materialized parquet (`ProjectionPublication`, the form
/// `ALTER TABLE … MATERIALIZE` produces), `ExportedPublication`, the WAL-owned
/// authorities — is [`StatsTrust::Trusted`] (ProximaDB wrote the footer).
fn parquet_stats_trust(schema: &proximadb_catalog::CatalogTableSchema) -> StatsTrust {
    use proximadb_catalog::CatalogAuthorityMode;
    let external = schema
        .storage_layouts
        .iter()
        .filter(|l| {
            matches!(
                l.physical_format,
                proximadb_catalog::CatalogPhysicalFormat::Parquet
            )
        })
        .any(|l| {
            matches!(
                l.authority,
                CatalogAuthorityMode::FederatedRead
                    | CatalogAuthorityMode::ExternalAuthoritative
                    | CatalogAuthorityMode::ImportedSnapshot
            )
        });
    if external {
        StatsTrust::Untrusted
    } else {
        StatsTrust::Trusted
    }
}

/// Master switch for the ADR-025 OLAP read-merge. **Default-ON** (ADR-025 PR3):
/// once a table is materialized, OLAP `SELECT`s reconcile its cold Parquet base
/// with the authoritative post-snapshot WAL delta so `DELETE`/`UPDATE`/`INSERT`
/// written after `MATERIALIZE` are reflected. Read-mostly tables (nothing changed
/// since the snapshot) take the empty-delta fast path — a bare Parquet read — so
/// default-ON is correctness- and cost-neutral for them.
///
/// Kill-switch: set `PROXIMADB_OLAP_DELTA_MERGE` to a falsy token
/// (`0`/`false`/`off`/`no`, case-insensitive) to force the legacy bare-Parquet
/// read engine-wide; individual tables can still opt back in via the
/// `olap_delta_merge` storage-layout property. Any other value — or leaving it
/// unset — keeps the merge on.
#[cfg(feature = "datafusion-integration")]
fn olap_delta_merge_enabled() -> bool {
    olap_delta_merge_on(std::env::var("PROXIMADB_OLAP_DELTA_MERGE").ok().as_deref())
}

/// Pure kill-switch policy for [`olap_delta_merge_enabled`], split out so it is
/// unit-testable without mutating the process environment: `None` (unset) or any
/// non-falsy value ⇒ merge ON; only an explicit falsy token turns it OFF.
#[cfg(feature = "datafusion-integration")]
fn olap_delta_merge_on(var: Option<&str>) -> bool {
    match var {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

/// Resolve the ADR-025 read-merge parameters for a parquet-backed table, or `None`
/// when it is ineligible (no single-column PK, no recorded `snapshot_lsn`, or not
/// opted in). Ineligible tables fall back to the bare Parquet read.
#[cfg(feature = "datafusion-integration")]
fn olap_delta_table_params(
    schema: &proximadb_catalog::CatalogTableSchema,
) -> Option<crate::query::execution::olap_delta_merge::OlapDeltaTable> {
    use proximadb_catalog::{CatalogAuthorityMode, CatalogPhysicalFormat};
    // Single-column PK required: the merge recomputes each base row's canonical oid
    // from its PK column. Keyless heaps fall back to the bare Parquet read.
    let pk_column = match schema.primary_key.as_slice() {
        [pk] => pk.clone(),
        _ => return None,
    };
    let layout = schema.storage_layouts.iter().find(|l| {
        matches!(l.physical_format, CatalogPhysicalFormat::Parquet)
            && matches!(
                l.authority,
                CatalogAuthorityMode::FederatedRead
                    | CatalogAuthorityMode::ExternalAuthoritative
                    | CatalogAuthorityMode::ImportedSnapshot
                    | CatalogAuthorityMode::ExportedPublication
                    | CatalogAuthorityMode::ProjectionPublication
            )
    })?;
    let opted_in = olap_delta_merge_enabled()
        || matches!(
            layout
                .properties
                .get("olap_delta_merge")
                .map(String::as_str),
            Some("1" | "true" | "on" | "yes")
        );
    if !opted_in {
        return None;
    }
    let snapshot_lsn: u64 = layout.properties.get("snapshot_lsn")?.parse().ok()?;
    Some(crate::query::execution::olap_delta_merge::OlapDeltaTable {
        snapshot_lsn,
        pk_column,
    })
}

/// Sync `CatalogLookup` wrapper for the process-wide engine.
struct EngineCatalog(Arc<InMemoryRelationalEngine>);

impl CatalogLookup for EngineCatalog {
    fn lookup_table(&self, name: &str) -> Option<std::sync::Arc<RelationalSchema>> {
        self.0.schema_of(name).map(std::sync::Arc::new)
    }
}

/// Per-table schema/PK cache for a query, pre-resolved from xCatalog.
#[derive(Clone, Debug)]
struct PreparedTable {
    table_name: String,
    /// Shared per-connection snapshot: readers clone the Arc, not the whole
    /// column list, on every open (the schema Vec was deep-cloned per
    /// reader-open and per planner lookup before).
    schema: std::sync::Arc<RelationalSchema>,
    pk_columns: Vec<usize>,
    /// TD-127: names of this table's single-column non-PK secondary indexes
    /// (from `schema_secondary_index_columns`). Threaded into the planner's
    /// `CapabilityResolver::secondary_index_columns` so an equality / IN on
    /// one of these columns can rewrite to `ScanAccess::SecondaryLookup`.
    secondary_columns: Vec<String>,
}

impl PreparedTable {
    fn from_catalog(name: &str, schema: &proximadb_catalog::CatalogTableSchema) -> Self {
        let cols: Vec<ColumnInfo> = schema
            .columns
            .iter()
            .map(|c| ColumnInfo::new(&c.name, c.data_type.clone(), c.nullable))
            .collect();
        let mut pk = Vec::new();
        for (i, c) in schema.columns.iter().enumerate() {
            if schema.primary_key.contains(&c.name) {
                pk.push(i);
            }
        }
        Self {
            table_name: name.to_string(),
            schema: std::sync::Arc::new(RelationalSchema::new(cols)),
            pk_columns: pk,
            secondary_columns: crate::services::record_store::schema_secondary_index_columns(
                schema,
            ),
        }
    }
}

struct SnapshotCatalog {
    dml: Arc<DmlService>,
    tables: HashMap<String, PreparedTable>,
    /// TD-064: connection tenant scope threaded into every snapshot reader so
    /// relational scans/point-lookups read the tenant's record partition.
    tenant: Option<TenantContext>,
    /// ADR-087 canonical caller identity. Owned because readers outlive the
    /// protocol stack frame; projected back to PortIdentity only at the DML
    /// enforcement boundary.
    identity: OwnedPortIdentity,
}

impl CatalogLookup for SnapshotCatalog {
    fn lookup_table(&self, name: &str) -> Option<std::sync::Arc<RelationalSchema>> {
        // Refcount bump — the per-lookup schema deep clone is gone.
        self.tables
            .get(&normalize_table_key(name))
            .map(|t| std::sync::Arc::clone(&t.schema))
    }
}

impl ReaderFactory for SnapshotCatalog {
    fn open_reader(&self, table: &TableId) -> Result<Box<dyn RelationalReader>, ExecError> {
        let key = normalize_table_key(&table.name);
        let prepared = self
            .tables
            .get(&key)
            .ok_or_else(|| ExecError::Internal(format!("table not snapshotted: {}", table.name)))?;
        Ok(Box::new(DmlTableReader {
            dml: self.dml.clone(),
            table_name: prepared.table_name.clone(),
            full_schema: std::sync::Arc::clone(&prepared.schema),
            pk_columns: prepared.pk_columns.clone(),
            tenant: self.tenant.clone(),
            identity: self.identity.clone(),
            open_state: None,
        }))
    }
}

/// `CapabilityResolver` for the real-data pipeline; mirrors the canonical
/// [`StaticCapabilities`]. Supplies per-table single-column PK ordinals so the
/// planner can rewrite a PK-equality scan to `ScanAccess::PkLookup`.
///
/// Route metadata for the PK-lookup it enables: `workload_profile=Oltp`,
/// `authority_mode=ProximaAuthoritative`, `policy_boundary=engine-enforced`,
/// `freshness=latest-committed` (NativeRecord — not an external open-format
/// route). ADR-018 Phase 2 (pgwire SQL parity); TD-076.
struct SnapshotCapabilities {
    pk_by_table: HashMap<String, Vec<usize>>,
    /// TD-127: per-table secondary-indexed column names. Empty / absent =
    /// no secondary index → planner keeps a full scan + filter.
    secondary_by_table: HashMap<String, Vec<String>>,
}

impl CapabilityResolver for SnapshotCapabilities {
    fn capabilities(&self, _table: &TableId) -> ReaderCapabilities {
        // Unchanged pushdown behavior (projection/predicate). PK-lookup,
        // PK-batch (TD-128) and secondary-lookup (TD-127) are advertised here
        // but gated per-table by `primary_key` / `secondary_index_columns`
        // below, so a table without the relevant index keeps a full scan.
        ReaderCapabilities::full()
            .with_pk_lookup_batch(true)
            .with_secondary_lookup(true)
    }

    fn primary_key(&self, table: &TableId) -> Vec<usize> {
        self.pk_by_table
            .get(&normalize_table_key(&table.name))
            .cloned()
            .unwrap_or_default()
    }

    fn secondary_index_columns(&self, table: &TableId) -> Vec<String> {
        self.secondary_by_table
            .get(&normalize_table_key(&table.name))
            .cloned()
            .unwrap_or_default()
    }
}

// =========================================================================
// DmlTableReader — lazy reader that pushes projection/predicate/limit into
// the DmlService scan (only matching, projected rows are materialized).
// =========================================================================

/// Per-scan state captured at `open`: the projected output schema + the rows
/// already fetched (predicate-filtered + projected + limited at the store).
struct ReaderOpenState {
    output_schema: Arc<RelationalSchema>,
    rows: Vec<RelationalRow>,
    cursor: usize,
}

struct DmlTableReader {
    dml: Arc<DmlService>,
    table_name: String,
    full_schema: Arc<RelationalSchema>,
    /// Single-column PK ordinal(s) for the PK-lookup arity check (empty when the
    /// planner won't pick PkLookup for this table).
    pk_columns: Vec<usize>,
    /// TD-064: connection tenant scope for this reader's record partition.
    tenant: Option<TenantContext>,
    /// ADR-087 whole caller identity retained to the enforcement seam.
    identity: OwnedPortIdentity,
    open_state: Option<ReaderOpenState>,
}

impl DmlTableReader {
    /// Resolve the projected output schema from `ScanContext::projection`
    /// (`None` = full schema), mirroring the `VecReader` contract.
    fn resolve_output_schema(
        &self,
        projection: &Option<Vec<String>>,
    ) -> Result<Arc<RelationalSchema>, ReaderError> {
        let Some(names) = projection else {
            return Ok(Arc::clone(&self.full_schema));
        };
        let mut cols = Vec::with_capacity(names.len());
        for n in names {
            let (_idx, info) = self
                .full_schema
                .column_by_name(n)
                .ok_or_else(|| ReaderError::InvalidProjection(n.clone()))?;
            cols.push(info.clone());
        }
        Ok(Arc::new(RelationalSchema::new(cols)))
    }
}

#[async_trait]
impl RelationalReader for DmlTableReader {
    fn name(&self) -> &'static str {
        "dml_relational"
    }

    fn capabilities(&self) -> ReaderCapabilities {
        // Honors projection + predicate + limit + single-column PK lookup
        // (see `lookup_pk`), discrete PK batch (TD-128, `lookup_pk_batch`) and
        // OLTP secondary lookup (TD-127, `lookup_secondary`). The planner gates
        // each per-table via the resolver's `primary_key` /
        // `secondary_index_columns`, so these flags are informational here.
        ReaderCapabilities::full()
            .with_pk_lookup(true)
            .with_pk_lookup_batch(true)
            .with_secondary_lookup(true)
    }

    fn schema(&self) -> &RelationalSchema {
        match &self.open_state {
            Some(state) => &state.output_schema,
            None => &self.full_schema,
        }
    }

    async fn open(&mut self, ctx: &ScanContext) -> Result<(), ReaderError> {
        let output_schema = self.resolve_output_schema(&ctx.projection)?;
        // Predicate ordinals bind to the FULL table schema; evaluate before
        // projection (same contract as VecReader). Type-check up-front.
        if let Some(pred) = &ctx.predicate {
            pred.type_check(&self.full_schema)
                .map_err(ReaderError::PredicateEval)?;
        }
        let predicate: Option<Expr> = ctx.predicate.clone();
        // ADR-043 Invariant 1 — fail-loud predicate: evaluate against the real
        // builtin registry and PROPAGATE any eval error (unknown function, cast,
        // arithmetic, type) instead of coercing it to `false`. The empty registry
        // (`NoFunctions`) and the error-swallowing `matches!(…, Ok(Boolean(true)))`
        // here were a silent-data-loss seam: a pushed-down predicate the native
        // engine could not evaluate dropped every row and presented as a clean empty
        // result. `scan_table_relational` now surfaces the captured error; a query
        // native cannot serve fails loudly and the OLAP route answers it (ADR-039).
        let row_pred = move |full_row: &[ProximaValue]| -> Result<bool, ExprError> {
            match &predicate {
                Some(expr) => match expr.eval(&full_row.to_vec(), builtins())? {
                    ProximaValue::Boolean(b) => Ok(b),
                    // SQL three-valued logic: NULL (and any non-boolean predicate
                    // result) is not TRUE, so the row is excluded — this is a
                    // *value*, not an error, and is the correct filter semantics.
                    _ => Ok(false),
                },
                None => Ok(true),
            }
        };
        let row_pred_ref: Option<
            &(dyn Fn(&[ProximaValue]) -> Result<bool, ExprError> + Send + Sync),
        > = if ctx.predicate.is_some() {
            Some(&row_pred)
        } else {
            None
        };
        let limit = ctx.limit.map(|l| l as usize);
        let (_schema, rows) = self
            .dml
            .scan_table_relational(
                &self.table_name,
                ctx.projection.as_deref(),
                row_pred_ref,
                limit,
                self.tenant.as_ref(),
                self.identity.as_borrowed(),
            )
            .await
            .map_err(|e| ReaderError::Storage(e.to_string()))?;
        self.open_state = Some(ReaderOpenState {
            output_schema,
            rows,
            cursor: 0,
        });
        Ok(())
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ReaderError> {
        let state = self.open_state.as_mut().ok_or(ReaderError::NotOpen)?;
        if state.cursor >= state.rows.len() {
            return Ok(None);
        }
        let row = state.rows[state.cursor].clone();
        state.cursor += 1;
        Ok(Some(row))
    }

    /// OLTP point-read fast path: resolve a single row by primary key via
    /// `DmlService::point_lookup_relational`. Returns the FULL row (the executor
    /// re-applies projection from `schema()`, which is the full schema on the
    /// PkLookup path — `open` is not called). Single-column PK only.
    /// ADR-018 Phase 2 (pgwire SQL parity); TD-076.
    async fn lookup_pk(&self, key: &[ProximaValue]) -> Result<Option<RelationalRow>, ReaderError> {
        if key.len() != 1 {
            return Err(ReaderError::PkArityMismatch {
                expected: self.pk_columns.len().max(1),
                actual: key.len(),
            });
        }
        // Convert the PK value to the stored `record.oid` string (the same
        // canonical text form used for int/string/uuid keys). SQL NULL → no row.
        let Some(key_str) = text_encode(&key[0]) else {
            return Ok(None);
        };
        tracing::debug!(
            target: "proximadb::pgwire::new_pipeline",
            access_path = "PkLookup",
            table = %self.table_name,
            "relational PK point lookup"
        );
        self.dml
            .point_lookup_relational(&self.table_name, &key_str, self.tenant.as_ref())
            .await
            .map_err(|e| ReaderError::Storage(e.to_string()))
    }

    /// TD-128: discrete multi-key OLTP point-read (`PK IN (...)` / eq-OR).
    /// Forwards to [`DmlService::point_lookup_batch_relational`], which reuses
    /// the same `get_by_key` path (dead-record-filtered) as the single-key
    /// fast-path. Single-column PK only; SQL NULL keys can't match a stored oid
    /// and are skipped. Returns the FULL candidate rows (the executor narrows).
    async fn lookup_pk_batch(
        &self,
        keys: &[Vec<ProximaValue>],
    ) -> Result<Vec<RelationalRow>, ReaderError> {
        let mut key_strs = Vec::with_capacity(keys.len());
        for key in keys {
            if key.len() != 1 {
                return Err(ReaderError::PkArityMismatch {
                    expected: 1,
                    actual: key.len(),
                });
            }
            if let Some(s) = text_encode(&key[0]) {
                key_strs.push(s);
            }
        }
        tracing::debug!(
            target: "proximadb::pgwire::new_pipeline",
            access_path = "PkLookupBatch",
            table = %self.table_name,
            keys = key_strs.len(),
            "relational PK batch point lookup"
        );
        self.dml
            .point_lookup_batch_relational(&self.table_name, &key_strs, self.tenant.as_ref())
            .await
            .map_err(|e| ReaderError::Storage(e.to_string()))
    }

    /// TD-127: OLTP secondary-index point-read. Forwards to
    /// [`DmlService::secondary_lookup_relational`], which probes the store's
    /// single-column hash index and re-materializes each live candidate via
    /// `get_by_key` (dead-record-filtered). `Ok(None)` (no built index, or all
    /// probe values NULL) → the executor falls back to a full scan + the
    /// residual filter; `Ok(Some(rows))` → FULL candidate rows (the executor
    /// narrows and the residual filter re-checks — the index only narrows).
    async fn lookup_secondary(
        &self,
        column: &str,
        values: &[ProximaValue],
    ) -> Result<Option<Vec<RelationalRow>>, ReaderError> {
        let mut value_strs = Vec::with_capacity(values.len());
        for v in values {
            if let Some(s) = text_encode(v) {
                value_strs.push(s);
            }
        }
        if value_strs.is_empty() {
            // No probe-able (non-NULL) values — let the executor scan + filter.
            return Ok(None);
        }
        tracing::debug!(
            target: "proximadb::pgwire::new_pipeline",
            access_path = "SecondaryLookup",
            table = %self.table_name,
            column = %column,
            "relational secondary index point lookup"
        );
        self.dml
            .secondary_lookup_relational(
                &self.table_name,
                column,
                &value_strs,
                self.tenant.as_ref(),
            )
            .await
            .map_err(|e| ReaderError::Storage(e.to_string()))
    }

    async fn close(&mut self) -> Result<(), ReaderError> {
        self.open_state = None;
        Ok(())
    }
}

// =========================================================================
// PgType / text encoding helpers
// =========================================================================

/// Map a [`ProximaType`] to the closest [`PgType`] for
/// `RowDescription` emission. Unknowns map to `Text` so clients
/// can still display values.
pub fn pg_type_for(t: &ProximaType) -> PgType {
    match t {
        ProximaType::Boolean => PgType::Bool,
        ProximaType::Int8 | ProximaType::Int16 => PgType::Int2,
        ProximaType::Int32 => PgType::Int4,
        ProximaType::Int64 => PgType::Int8,
        ProximaType::UInt8 | ProximaType::UInt16 => PgType::Int4,
        ProximaType::UInt32 | ProximaType::UInt64 => PgType::Int8,
        ProximaType::Float16 | ProximaType::Float32 => PgType::Float4,
        ProximaType::Float64 => PgType::Float8,
        ProximaType::String | ProximaType::Symbol => PgType::Text,
        ProximaType::Binary => PgType::Bytea,
        ProximaType::Date => PgType::Date,
        ProximaType::Timestamp(_) => PgType::Timestamp,
        ProximaType::TimestampTz(_) => PgType::Timestamptz,
        ProximaType::Json => PgType::Json,
        ProximaType::Jsonb => PgType::Jsonb,
        ProximaType::Uuid | ProximaType::ULID => PgType::Uuid,
        _ => PgType::Text,
    }
}

/// Text-encode a [`ProximaValue`] for the pgwire text format.
/// Returns `None` for SQL `NULL` (the caller emits the pgwire
/// null sentinel `-1` length).
pub fn text_encode(v: &ProximaValue) -> Option<String> {
    use ProximaValue as V;
    Some(match v {
        V::Null => return None,
        V::Boolean(b) => (if *b { "t" } else { "f" }).into(),
        V::Int8(x) => x.to_string(),
        V::Int16(x) => x.to_string(),
        V::Int32(x) => x.to_string(),
        V::Int64(x) => x.to_string(),
        V::UInt8(x) => x.to_string(),
        V::UInt16(x) => x.to_string(),
        V::UInt32(x) => x.to_string(),
        V::UInt64(x) => x.to_string(),
        V::Float16(x) | V::Float32(x) => x.to_string(),
        V::Float64(x) => x.to_string(),
        V::Decimal(s) => s.clone(),
        V::String(s) | V::Symbol(s) => s.clone(),
        V::Binary(b) | V::BinaryVector(b) => {
            let mut s = String::with_capacity(2 + b.len() * 2);
            s.push_str("\\x");
            for byte in b {
                s.push_str(&format!("{:02x}", byte));
            }
            s
        }
        V::Date(d) => d.to_string(),
        V::Time(t, _) | V::Timestamp(t, _) | V::TimestampTz(t, _) => t.to_string(),
        V::Uuid(u) | V::ULID(u) => format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            u[0],
            u[1],
            u[2],
            u[3],
            u[4],
            u[5],
            u[6],
            u[7],
            u[8],
            u[9],
            u[10],
            u[11],
            u[12],
            u[13],
            u[14],
            u[15]
        ),
        V::Json(j) | V::Jsonb(j) => j.to_string(),
        V::Array(values) => {
            // Element NULL renders as the empty field in a Postgres array
            // literal (same convention as the legacy simple-query encoder).
            let parts = values
                .iter()
                .map(text_encode)
                .map(Option::unwrap_or_default)
                .collect::<Vec<_>>();
            format!("{{{}}}", parts.join(","))
        }
        V::Map(value) | V::Struct(value) => {
            serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
        }
        V::DenseVector(values) => {
            let parts = values.iter().map(ToString::to_string).collect::<Vec<_>>();
            format!("[{}]", parts.join(","))
        }
        V::SparseVector { indices, values } => serde_json::json!({
            "indices": indices,
            "values": values,
        })
        .to_string(),
    })
}

// =========================================================================
// Read-route EXPLAIN surface (course-correction §5 / ADR-004)
// =========================================================================

/// Compute the read-route decision for a `SELECT` without executing it.
///
/// Returns `None` when `sql` is not a single relational query (so the caller has
/// no route to report). Catalog-free in P0 — routes on query shape only via the
/// canonical [`crate::query::compute_scheduler::ComputeScheduler`]; P1 adds the
/// Parquet-backed signal that flips the OLAP arm to DataFusion.
pub fn classify_select_route(
    sql: &str,
) -> Option<crate::query::compute_scheduler::SelectRouteDecision> {
    let statements = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return None;
    };
    let engages = query_engages_relational_engine(query);
    Some(
        crate::query::compute_scheduler::ComputeScheduler::new().route_select_advised(
            crate::query::compute_scheduler::QueryShape {
                engages_relational: engages,
                parquet_backed: false,
                cardinality: query_cardinality_hint(query),
                join_bearing: query_join_bearing(query),
                ..Default::default()
            },
            Some(&crate::query::route_cost_model::GLOBAL_ROUTE_COST_MODEL),
            // `parquet_backed: false` here — the env master switches gate only the
            // parquet-OLAP primary arm, so flags are inert; default is exact.
            crate::query::compute_scheduler::RouteFlags::default(),
        ),
    )
}

/// Serializable read-route explanation for `EXPLAIN SELECT` (JSON over Jsonb).
///
/// Field vocabulary mirrors `table_write_plan::RouteDecisionMetadata` so read and
/// write EXPLAIN share one contract (ADR-004 unified EXPLAIN).
/// Native-lowering disclosure for the `EXPLAIN` trace (ADR-064 / TD-TRACE-1).
/// Layered onto [`SelectRouteExplanation`] so a query that declines native
/// lowering surfaces the real reason rather than silence (or a text-guessed
/// mask). Default/empty for queries that lowered or routed elsewhere.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LoweringTrace {
    /// Named lowering rules that fired. Populated once the ADR-064 lowering
    /// rule-seam lands (Phase 2 / TD-REL-LOWER-5); empty today.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rules_fired: Vec<String>,
    /// Structured `{node, why}` decline reasons when native lowering could not
    /// serve the shape. Empty when the query lowered (or was not routed native).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub declines: Vec<LoweringDecline>,
}

impl LoweringTrace {
    /// True when there is nothing to disclose (skips the field in EXPLAIN JSON).
    fn is_empty(&self) -> bool {
        self.rules_fired.is_empty() && self.declines.is_empty()
    }
}

/// One physical-plan operator's metered execution actuals for the structured
/// `EXPLAIN ANALYZE` surface (TD-TRACE-1 Slice 2). Pre-order, index-aligned with
/// `physical_plan`. Mirrors the io_trace `ExecOpTrace` snapshot shape but is the
/// EXPLAIN JSON view. `bytes`/`spill` are `None`/`false` for the row-oriented
/// native Volcano executor (the DataFusion adapter, Slice 3, fills them).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecOp {
    /// Operator keyword (matches the `physical_plan` line at the same index).
    pub op: String,
    /// Rows fed into this operator = sum of its direct children's `rows_out`.
    pub rows_in: u64,
    /// Rows this operator emitted.
    pub rows_out: u64,
    /// Exclusive (self) milliseconds — inclusive minus children.
    pub ms_self: u64,
    /// Bytes processed when the engine tracks it; `None` for native.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Whether this operator spilled to disk; always `false` for native.
    pub spill: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SelectRouteExplanation {
    /// Selected physical engine label (e.g. `Native(Volcano)`, `DataFusionLocal`).
    pub compute_route: String,
    /// Workload classification (`Oltp`/`Olap`).
    pub workload_profile: String,
    /// Control-plane routing decision — no durable authority is moved.
    pub authority_mode: String,
    /// Routing granularity — one engine per query plan, never per row.
    pub policy_boundary: String,
    /// Freshness guarantee of the selected route.
    pub freshness_sla: String,
    /// Human-readable reason for the choice.
    pub reason: String,
    /// Structured, axis-tagged routing reasons (ADR-058
    /// eligibility→selection→trust) behind the chosen route (ADR-064 /
    /// TD-TRACE-1). Always non-empty for a real route; the machine-readable
    /// record the cost model and tooling read instead of parsing `reason`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub routing_reasons: Vec<crate::query::compute_scheduler::RouteReason>,
    /// Native-lowering trace: which rules fired and, on a decline, the REAL
    /// structured `{node, why}` reason — so a declining query discloses *why* in
    /// EXPLAIN instead of the pgwire path re-guessing it from SQL text
    /// (ADR-064 Decision 3 / TD-TRACE-1). Omitted when empty.
    #[serde(skip_serializing_if = "LoweringTrace::is_empty")]
    pub lowering: LoweringTrace,
    /// Typed read-route contract that future split-aware DataFusion/Ballista
    /// execution will consume. Kept nested so existing top-level fields remain
    /// stable while ADR-004 diagnostics converge on `RoutedReadPlan`.
    pub read_route: crate::query::read_route::ReadRouteExplanation,
    /// Structural disclosure of the planned physical plan (one string per
    /// operator, indented), when the query engages the native (Volcano) PATH B
    /// engine. `None` for simple/legacy SELECTs and non-native routes. No cost
    /// estimates — the planner is rule-based (ADR-004; CBO is a follow-up).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_plan: Option<Vec<String>>,
    /// EXPLAIN ANALYZE only: rows the executed plan actually returned. `None` for
    /// plain EXPLAIN (the query is not executed) and non-native routes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_rows: Option<u64>,
    /// EXPLAIN ANALYZE only: wall-clock microseconds to execute the whole plan
    /// (build + open + drain). `None` for plain EXPLAIN. Whole-query granularity —
    /// per-operator timing is a follow-up (no Volcano instrumentation yet).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_elapsed_us: Option<u64>,
    /// ADR-004 / TD-174 (gap #4): stats-backed estimated selectivity of the
    /// `WHERE` clause — the product of per-`col = ?` equality selectivities
    /// (`1/ndistinct` from the resident HLL distinct count, ADR-037 producer).
    /// A 0..1 fraction the AnvaiOps cost consumer prices. `None` (omitted) when
    /// the query has no equality predicate, targets multiple tables, or the
    /// collection has no resident statistics yet — correct-or-absent, never a
    /// guessed value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_selectivity: Option<f64>,
    /// Estimated surviving rows = `record_count × estimated_selectivity`, when
    /// both the selectivity and the resident `record_count` are known. `None`
    /// (omitted) otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_rows: Option<u64>,
    /// EXPLAIN ANALYZE only: per-operator execution actuals in pre-order
    /// (TD-TRACE-1 Slice 2) — `{op, rows_in, rows_out, ms_self, bytes, spill}`,
    /// index-aligned with `physical_plan`. Empty (and omitted) for plain EXPLAIN
    /// and non-native routes; the structured counterpart of the per-op text
    /// annotations on `physical_plan`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub exec: Vec<ExecOp>,
}

/// Build the `EXPLAIN SELECT` payload. `Err` if `sql` is not a routable single
/// relational query.
/// Render a route decision to the serializable EXPLAIN payload (single source so the
/// catalog-free and catalog-aware EXPLAIN variants stay consistent).
fn decision_to_explanation(
    decision: &crate::query::compute_scheduler::SelectRouteDecision,
) -> SelectRouteExplanation {
    let read_route = decision.routed_read_plan().route_explanation();
    SelectRouteExplanation {
        compute_route: decision.compute_route_label(),
        workload_profile: format!("{:?}", decision.workload_profile),
        authority_mode: read_route.authority_mode.clone(),
        policy_boundary: read_route.policy_boundary.clone(),
        freshness_sla: read_route.freshness_sla.clone(),
        reason: decision.reason.clone(),
        routing_reasons: decision.reasons.clone(),
        lowering: LoweringTrace::default(),
        read_route,
        // Populated by the catalog-aware EXPLAIN once the plan is built; the
        // catalog-free route disclosure has no plan to render. ANALYZE metrics are
        // filled only when the catalog-aware ANALYZE path actually executes.
        physical_plan: None,
        execution_rows: None,
        execution_elapsed_us: None,
        estimated_selectivity: None,
        estimated_rows: None,
        exec: Vec::new(),
    }
}

/// Collect the LHS column names of top-level `col = <literal>` equality predicates
/// in a single-table SELECT's `WHERE` clause (AND-chained). Non-equality
/// predicates, OR branches, and `col = col` (no literal) are ignored — only the
/// shapes statistics can estimate. Returns an empty Vec for anything else.
fn equality_predicate_columns(query: &SqlQuery) -> Vec<String> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Vec::new();
    };
    let Some(selection) = select.selection.as_ref() else {
        return Vec::new();
    };
    let mut cols = Vec::new();
    collect_eq_columns(selection, &mut cols);
    cols
}

fn collect_eq_columns(expr: &SqlExpr, out: &mut Vec<String>) {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_eq_columns(left, out);
            collect_eq_columns(right, out);
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            // `col = <literal>`: a bare identifier on one side, a literal value on
            // the other (either order). `col = col` has no literal → not estimable.
            if let (Some(col), true) = (ident_name(left), is_literal(right)) {
                out.push(col);
            } else if let (Some(col), true) = (ident_name(right), is_literal(left)) {
                out.push(col);
            }
        }
        _ => {}
    }
}

fn ident_name(expr: &SqlExpr) -> Option<String> {
    match expr {
        SqlExpr::Identifier(id) => Some(id.value.clone()),
        // `t.col` → the bare column name (stats are keyed by column).
        SqlExpr::CompoundIdentifier(parts) => parts.last().map(|p| p.value.clone()),
        _ => None,
    }
}

fn is_literal(expr: &SqlExpr) -> bool {
    matches!(expr, SqlExpr::Value(_))
}

/// Stats-backed selectivity + row estimate for a single-table SELECT's equality
/// `WHERE` predicates. `eq_sel(field) -> Option<f64>` returns the per-field
/// equality selectivity (DI seam for testing; production passes the registry).
/// Returns `(selectivity, estimated_rows)` only when at least one predicate has a
/// resident statistic — else `None` (caller leaves EXPLAIN unchanged). Multiple
/// predicates multiply (independence assumption, standard CBO).
fn stats_filter_estimate(
    cols: &[String],
    record_count: Option<u64>,
    eq_sel: impl Fn(&str) -> Option<f64>,
) -> Option<(f64, Option<u64>)> {
    let mut selectivity = 1.0_f64;
    let mut any = false;
    for col in cols {
        if let Some(s) = eq_sel(col) {
            selectivity *= s;
            any = true;
        }
    }
    if !any {
        return None;
    }
    let rows = record_count.map(|n| (n as f64 * selectivity).ceil() as u64);
    Some((selectivity, rows))
}

/// Prepare an engaging SELECT for EXPLAIN (cold path): resolve table schemas, build
/// the snapshot, then lower + plan via the SAME [`plan_over_snapshot`] execution
/// uses. Returns the snapshot+plan so EXPLAIN can render the tree and (for ANALYZE)
/// execute it via [`execute_physical`]. `None` when a schema can't be resolved,
/// lowering declines, or planning errors — EXPLAIN then falls back to route-only.
async fn prepare_select_plan(
    sql: &str,
    query: &SqlQuery,
    dml: &Arc<DmlService>,
    identity: PortIdentity<'_>,
) -> Option<(SnapshotCatalog, PhysicalPlan)> {
    let snapshot = build_snapshot(query, dml, identity).await?;
    match plan_over_snapshot(sql, &snapshot)? {
        Ok(physical) => Some((snapshot, physical)),
        Err(_) => None,
    }
}

/// Resolve every table a SELECT references into a [`SnapshotCatalog`] under the
/// connection tenant. `None` if any referenced schema can't be resolved. Split
/// out of [`prepare_select_plan`] so the EXPLAIN path can build the snapshot once
/// and, on a lowering decline, re-lower to disclose the REAL structured reason
/// (ADR-064 / TD-TRACE-1) instead of leaving EXPLAIN silent.
async fn build_snapshot(
    query: &SqlQuery,
    dml: &Arc<DmlService>,
    identity: PortIdentity<'_>,
) -> Option<SnapshotCatalog> {
    let tenant = identity.tenant_id;
    let mut names = Vec::new();
    let cte_names = collect_table_names(query, &mut names);
    let mut tables: HashMap<String, PreparedTable> = HashMap::new();
    for raw in &names {
        let key = normalize_table_key(raw);
        // TD-185a: WITH-defined names resolve by the SQL engine, not the catalog.
        if cte_names.contains(&key) {
            continue;
        }
        if tables.contains_key(&key) {
            continue;
        }
        // Resolve under the connection tenant so EXPLAIN finds the tenant's
        // schema row (and, for ANALYZE, reads the tenant's partition).
        let schema = dml.resolve_relational_schema(raw, tenant).await.ok()?;
        tables.insert(key, PreparedTable::from_catalog(raw, &schema));
    }
    Some(SnapshotCatalog {
        dml: dml.clone(),
        tables,
        tenant: tenant.filter(|t| !t.is_empty()).map(|tenant| {
            let mut context = TenantContext::for_tenant_id(tenant);
            context.tenant_stable_id = identity.tenant_stable_id;
            context
        }),
        // EXPLAIN ANALYZE executes the query, so it retains the exact same full
        // identity and composition semantics as normal execution.
        identity: identity.into_owned(),
    })
}

/// Catalog-free `EXPLAIN SELECT` route (shape only). Reports the Volcano/Native route;
/// use [`explain_select_route_with_catalog`] when a `DmlService` is available so
/// Parquet-backed tables disclose the DataFusion route they actually take.
pub fn explain_select_route(sql: &str) -> Result<SelectRouteExplanation, String> {
    let decision =
        classify_select_route(sql).ok_or_else(|| "not a routable relational SELECT".to_string())?;
    Ok(decision_to_explanation(&decision))
}

/// Catalog-aware `EXPLAIN SELECT`: resolves referenced tables so the disclosed route
/// matches what [`try_run_select`] executes. Discloses the planned physical plan but
/// does NOT execute the query. Under `datafusion-integration`, an all-Parquet-backed
/// query discloses `DataFusionLocal`; otherwise it discloses the Volcano/Native route.
pub async fn explain_select_route_with_catalog(
    sql: &str,
    dml: &Arc<DmlService>,
    identity: PortIdentity<'_>,
) -> Result<SelectRouteExplanation, String> {
    route_and_plan_select(sql, dml, false, identity).await
}

/// Catalog-aware `EXPLAIN ANALYZE SELECT`: like [`explain_select_route_with_catalog`]
/// but ALSO executes the planned native (Volcano) plan and attaches the measured
/// `execution_rows` + `execution_elapsed_us`. SELECT is read-only, so executing it is
/// side-effect free (matches Postgres EXPLAIN ANALYZE). Execution errors propagate.
pub async fn explain_analyze_select_with_catalog(
    sql: &str,
    dml: &Arc<DmlService>,
    identity: PortIdentity<'_>,
) -> Result<SelectRouteExplanation, String> {
    route_and_plan_select(sql, dml, true, identity).await
}

/// Shared core for catalog-aware `EXPLAIN [ANALYZE] SELECT`. Routes the query, then for
/// an engaging native plan renders the physical tree; when `analyze`, also executes it
/// (timed) and records actual rows + wall-clock. The plan is built ONCE via
/// [`prepare_select_plan`] and (for ANALYZE) executed via the same [`execute_physical`]
/// path real queries use — so the disclosed plan and the measured run never diverge.
async fn route_and_plan_select(
    sql: &str,
    dml: &Arc<DmlService>,
    analyze: bool,
    identity: PortIdentity<'_>,
) -> Result<SelectRouteExplanation, String> {
    let tenant = identity.tenant_id;
    let statements =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| format!("parse: {e}"))?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return Err("not a routable relational SELECT".to_string());
    };
    let engages = query_engages_relational_engine(query);

    #[allow(unused_mut)]
    let mut parquet_backed = false;
    // Locations of the all-Parquet table set, captured during the route check so the
    // EXPLAIN split disclosure can reopen exactly the objects the executor scans.
    #[cfg(feature = "datafusion-integration")]
    let mut parquet_locations: Vec<String> = Vec::new();
    #[cfg(feature = "datafusion-integration")]
    {
        let mut names = Vec::new();
        let cte_names = collect_table_names(query, &mut names);
        if !names.is_empty() {
            let mut all_parquet = true;
            let mut locations = Vec::with_capacity(names.len());
            for raw in &names {
                // TD-185a: WITH-defined names resolve by the SQL engine.
                if cte_names.contains(&normalize_table_key(raw)) {
                    continue;
                }
                match dml.resolve_relational_schema(raw, tenant).await {
                    Ok(schema) => match catalog_table_is_parquet_backed(&schema) {
                        Some(location) => locations.push(location),
                        None => {
                            all_parquet = false;
                            break;
                        }
                    },
                    _ => {
                        all_parquet = false;
                        break;
                    }
                }
            }
            parquet_backed = all_parquet;
            if all_parquet {
                parquet_locations = locations;
            }
        }
    }
    let decision = crate::query::compute_scheduler::ComputeScheduler::new().route_select_advised(
        crate::query::compute_scheduler::QueryShape {
            engages_relational: engages,
            parquet_backed,
            cardinality: query_cardinality_hint(query),
            join_bearing: query_join_bearing(query),
            ..Default::default()
        },
        Some(&crate::query::route_cost_model::GLOBAL_ROUTE_COST_MODEL),
        // EXPLAIN must disclose the SAME backend the real path picks, so thread the
        // real env master switches (TD-ROUTE-3), not a default.
        crate::query::compute_scheduler::RouteFlags {
            native_route: crate::query::execution::native_engine::native_route_enabled(),
            #[cfg(feature = "duckdb")]
            duckdb_route: crate::query::execution::duckdb_engine::duckdb_route_enabled(),
            #[cfg(not(feature = "duckdb"))]
            duckdb_route: false,
        },
    );
    let mut explanation = decision_to_explanation(&decision);

    // ADR-004 / TD-174 (gap #4): disclose stats-backed selectivity for a
    // single-table SELECT with `col = ?` equality predicate(s). Resolve the table
    // to the canonical collection id (the key the flush path stamps stats under),
    // multiply the resident per-field equality selectivities, and estimate
    // surviving rows from the resident record_count. Best-effort and
    // correct-or-absent: any unresolved table / absent statistics leaves the
    // fields `None` (omitted from EXPLAIN) rather than guessing.
    {
        let mut table_names = Vec::new();
        collect_table_names(query, &mut table_names);
        if table_names.len() == 1 {
            let cols = equality_predicate_columns(query);
            if !cols.is_empty()
                && let Some(collection_id) =
                    dml.resolve_collection_id(&table_names[0], tenant).await
            {
                let registry = crate::core::statistics::statistics_registry();
                let record_count = registry.envelope(&collection_id).map(|e| e.record_count);
                if let Some((selectivity, rows)) = stats_filter_estimate(&cols, record_count, |f| {
                    registry.equality_selectivity(&collection_id, f)
                }) {
                    explanation.estimated_selectivity = Some(selectivity);
                    explanation.estimated_rows = rows;
                }
            }
        }
    }
    // For the DataFusion route, disclose the concrete Parquet row-group split
    // inventory (partition count + footer row/byte estimates) in place of the
    // conservative whole-collection placeholder, by reading the same object-store
    // tables the executor will scan. Best-effort: any open failure leaves the
    // conservative summary intact rather than failing EXPLAIN.
    #[cfg(feature = "datafusion-integration")]
    if matches!(
        decision.backend,
        crate::query::table_write_plan::ComputeBackend::DataFusionLocal
    ) && let Some(summary) = parquet_split_summary(&parquet_locations).await
    {
        explanation.read_route = decision
            .routed_read_plan_with_splits(summary)
            .route_explanation();
    }
    // Disclose the planned physical plan for native (Volcano) PATH B queries — the same
    // plan execution runs (via the shared `prepare_select_plan` / `execute_physical`).
    if engages
        && matches!(
            decision.backend,
            crate::query::table_write_plan::ComputeBackend::Native
        )
    {
        match prepare_select_plan(sql, query, dml, identity).await {
            Some((snapshot, physical)) => {
                let base_lines = explain_physical(&physical);
                if analyze {
                    // EXPLAIN ANALYZE: run the query (read-only) and record actuals —
                    // whole-query totals plus per-operator rows/time annotated onto the
                    // plan lines (metrics are pre-order aligned with `base_lines`).
                    let started = std::time::Instant::now();
                    let (result, node_metrics) =
                        execute_physical_metered(physical, &snapshot).await?;
                    explanation.execution_rows = Some(result.rows.len() as u64);
                    explanation.execution_elapsed_us = Some(started.elapsed().as_micros() as u64);
                    // TD-TRACE-1 Slice 2: the structured per-operator vector, plus
                    // the same data pushed into the active io_trace scope (neutral
                    // primitive tuples, like `record_plan_geometry`) so the cost
                    // model / billing see per-op detail. No-ops if no scope active.
                    let exec = build_exec_ops(&node_metrics);
                    let exec_tuples: Vec<crate::observability::io_trace::ExecOpSample> = exec
                        .iter()
                        .map(|e| {
                            (
                                e.op.as_str(),
                                e.rows_in,
                                e.rows_out,
                                e.ms_self,
                                e.bytes,
                                e.spill,
                            )
                        })
                        .collect();
                    crate::observability::io_trace::record_exec_vector(&exec_tuples);
                    explanation.exec = exec;
                    explanation.physical_plan =
                        Some(annotate_plan_lines(base_lines, &node_metrics));
                } else {
                    explanation.physical_plan = Some(base_lines);
                }
            }
            None => {
                // No native plan was produced. If the tables resolve but native
                // LOWERING declined, disclose the REAL structured `{node, why}`
                // reason (ADR-064 / TD-TRACE-1) — the same decline the legacy path
                // otherwise re-guesses from SQL text. A re-lower here is cheap and
                // EXPLAIN is a cold, diagnostic path (never the hot query path).
                if let Some(snapshot) = build_snapshot(query, dml, identity).await
                    && let Err(e) = lower_sql(sql, &snapshot)
                {
                    explanation.lowering.declines.push(e.decline());
                }
            }
        }
    }
    Ok(explanation)
}

/// Open the object-store Parquet tables backing a DataFusion-routed query and sum
/// their row-group split inventory into a [`crate::query::read_route::ReadSplitSummary`]
/// for EXPLAIN. Returns `None` (caller keeps the conservative whole-collection
/// summary) when there are no locations or any table fails to open — split
/// disclosure is diagnostic and must never make EXPLAIN itself fail.
#[cfg(feature = "datafusion-integration")]
async fn parquet_split_summary(
    locations: &[String],
) -> Option<crate::query::read_route::ReadSplitSummary> {
    use crate::datafusion::engine_adapters::ObjectStoreParquetTable;
    if locations.is_empty() {
        return None;
    }
    let mut partitions = 0usize;
    let (mut rows, mut bytes) = (0u64, 0u64);
    let (mut any_rows, mut any_bytes) = (false, false);
    for location in locations {
        let table = ObjectStoreParquetTable::open(location).await.ok()?;
        let splits = table.split_count();
        let est_rows = table.estimated_rows();
        partitions += splits;
        if let Some(r) = est_rows {
            rows = rows.saturating_add(r);
            any_rows = true;
        }
        if let Some(b) = table.estimated_bytes() {
            bytes = bytes.saturating_add(b);
            any_bytes = true;
        }
        // EXPLAIN opens the footer anyway — warm the route-time shape cache so
        // the next SELECT's route decision can classify this location's fan-out
        // / cardinality without a cold read (co-design: zero extra I/O).
        crate::query::route_cost_model::record_table_shape_stat(location, splits as u32, est_rows);
    }
    Some(crate::query::read_route::ReadSplitSummary::row_groups(
        partitions,
        any_rows.then_some(rows),
        any_bytes.then_some(bytes),
    ))
}

/// Append per-operator actuals to each physical-plan line. Metrics are pre-order
/// aligned with the lines (both walk the plan parent-first, left-to-right); if the
/// counts ever disagree, leave the lines unannotated (whole-query metrics still
/// reported) rather than mislabel. `time` is inclusive of children (Postgres "actual
/// time"); `self` subtracts the direct children's inclusive time — the operator's own
/// cost (the headline "which node is actually slow").
/// Build the structured per-operator `exec[]` vector (TD-TRACE-1 Slice 2) from the
/// metered `NodeMetric`s. Pre-order, index-aligned with the plan lines: `rows_out`
/// is the operator's emitted rows, `rows_in` the sum of its direct children's
/// `rows_out` (via `child_input_rows`), `ms_self` the exclusive time in
/// milliseconds (via `self_times`, matching the `compute_ms` billing unit).
/// `bytes`/`spill` are `None`/`false` for the row-oriented native executor (the
/// DataFusion adapter fills them in a later slice).
fn build_exec_ops(metrics: &[NodeMetric]) -> Vec<ExecOp> {
    let self_ns = proximadb_relational_executor::self_times(metrics);
    let rows_in = proximadb_relational_executor::child_input_rows(metrics);
    metrics
        .iter()
        .enumerate()
        .map(|(i, m)| ExecOp {
            op: m.label.clone(),
            rows_in: rows_in[i],
            rows_out: m.rows,
            ms_self: (self_ns[i] / 1_000_000) as u64,
            bytes: None,
            spill: false,
        })
        .collect()
}

fn annotate_plan_lines(lines: Vec<String>, metrics: &[NodeMetric]) -> Vec<String> {
    if lines.len() != metrics.len() {
        return lines;
    }
    let self_ns = proximadb_relational_executor::self_times(metrics);
    lines
        .into_iter()
        .zip(metrics)
        .zip(self_ns)
        .map(|((line, m), self_ns)| {
            format!(
                "{line} (actual rows={} time={}us self={}us)",
                m.rows,
                m.elapsed_ns / 1000,
                self_ns / 1000
            )
        })
        .collect()
}

#[cfg(test)]
mod route_explain_tests {
    use super::*;

    #[test]
    fn native_containment_rejects_any_remaining_cross_join() {
        let values = || PhysicalPlan::Values {
            rows: Vec::new(),
            output_schema: RelationalSchema::new(Vec::new()),
        };
        let raw_cross = PhysicalPlan::Join {
            left: Box::new(values()),
            right: Box::new(values()),
            kind: proximadb_relational_algebra::JoinKind::Cross,
            on: None,
            strategy: proximadb_relational_algebra::JoinStrategy::NestedLoop,
        };
        assert!(plan_has_raw_cross_join(&raw_cross));

        let inner = PhysicalPlan::Join {
            left: Box::new(values()),
            right: Box::new(values()),
            kind: proximadb_relational_algebra::JoinKind::Inner,
            on: None,
            strategy: proximadb_relational_algebra::JoinStrategy::NestedLoop,
        };
        assert!(!plan_has_raw_cross_join(&inner));
    }

    #[test]
    fn olap_select_explains_as_olap() {
        let expl = explain_select_route("SELECT service, count(*) FROM events GROUP BY service")
            .expect("routable");
        assert_eq!(expl.workload_profile, "Olap");
        // P0 invariant: OLAP shape still executes on Volcano.
        assert_eq!(expl.compute_route, "Native(Volcano)");
        assert_eq!(expl.freshness_sla, "synchronous");
        assert_eq!(expl.read_route.selected_backend, "Native");
        assert_eq!(expl.read_route.split_strategy, "whole_collection");
    }

    #[test]
    fn simple_select_explains_as_oltp() {
        let expl =
            explain_select_route("SELECT id, name FROM users WHERE id = 1").expect("routable");
        assert_eq!(expl.workload_profile, "Oltp");
        assert_eq!(expl.compute_route, "Native(Volcano)");
    }

    #[test]
    fn non_select_is_not_routable() {
        assert!(explain_select_route("INSERT INTO t VALUES (1)").is_err());
    }

    #[test]
    fn catalog_free_route_carries_no_physical_plan() {
        // The catalog-free route disclosure has no planner output to render, so
        // `physical_plan` stays None and is omitted from the JSON. The physical
        // plan is only attached by the catalog-aware variant (covered e2e).
        let expl = explain_select_route("SELECT service, count(*) FROM events GROUP BY service")
            .expect("routable");
        assert!(expl.physical_plan.is_none());
        let json = serde_json::to_string(&expl).unwrap();
        assert!(
            !json.contains("physical_plan"),
            "None physical_plan is skipped in JSON: {json}"
        );
    }

    #[test]
    fn explain_carries_structured_routing_reasons() {
        // ADR-064 / TD-TRACE-1: EXPLAIN discloses the structured, axis-tagged
        // routing reasons (not just the flat `reason` string) so tooling/cost
        // model read them without parsing prose.
        use crate::query::compute_scheduler::RouteAxis;
        let expl =
            explain_select_route("SELECT id, name FROM users WHERE id = 1").expect("routable");
        assert!(
            !expl.routing_reasons.is_empty(),
            "routing_reasons must be populated"
        );
        assert!(
            expl.routing_reasons
                .iter()
                .any(|r| matches!(r.axis, RouteAxis::Selection)),
            "a Selection reason is always present: {:?}",
            expl.routing_reasons
        );
        // The structured reasons serialize into the EXPLAIN JSON.
        let json = serde_json::to_string(&expl).unwrap();
        assert!(
            json.contains("routing_reasons") && json.contains("Selection"),
            "routing_reasons surface in EXPLAIN JSON: {json}"
        );
        // No native lowering was attempted (catalog-free), so no declines and the
        // empty `lowering` trace is omitted from the JSON.
        assert!(expl.lowering.is_empty());
        assert!(
            !json.contains("lowering"),
            "empty lowering trace is skipped: {json}"
        );
    }

    #[test]
    fn route_only_explanation_omits_analyze_metrics() {
        // Plain (non-ANALYZE) route disclosure never executes, so the ANALYZE metric
        // fields stay None and are omitted from the JSON. They are populated only by
        // the catalog-aware ANALYZE path that actually runs the plan (covered e2e).
        let expl = explain_select_route("SELECT service, count(*) FROM events GROUP BY service")
            .expect("routable");
        assert!(expl.execution_rows.is_none() && expl.execution_elapsed_us.is_none());
        // TD-TRACE-1 Slice 2: the per-op exec[] vector is ANALYZE-only too.
        assert!(expl.exec.is_empty());
        let json = serde_json::to_string(&expl).unwrap();
        assert!(
            !json.contains("execution_rows")
                && !json.contains("execution_elapsed_us")
                && !json.contains("\"exec\""),
            "None ANALYZE metrics (incl. exec[]) are skipped in JSON: {json}"
        );
    }

    #[test]
    fn build_exec_ops_maps_metered_metrics_pre_order() {
        // Aggregate(1 child, 1 row, 5ms incl) over Scan(0 children, 10 rows, 2ms):
        //   Aggregate: rows_in=10 (Scan's rows_out), rows_out=1, ms_self=5-2=3
        //   Scan:      rows_in=0 (leaf),               rows_out=10, ms_self=2
        let metrics = vec![
            NodeMetric {
                label: "Aggregate".into(),
                arity: 1,
                rows: 1,
                elapsed_ns: 5_000_000,
            },
            NodeMetric {
                label: "Scan".into(),
                arity: 0,
                rows: 10,
                elapsed_ns: 2_000_000,
            },
        ];
        let exec = build_exec_ops(&metrics);
        assert_eq!(exec.len(), 2);
        assert_eq!(exec[0].op, "Aggregate");
        assert_eq!(exec[0].rows_in, 10);
        assert_eq!(exec[0].rows_out, 1);
        assert_eq!(exec[0].ms_self, 3);
        assert_eq!(exec[1].op, "Scan");
        assert_eq!(exec[1].rows_in, 0);
        assert_eq!(exec[1].rows_out, 10);
        assert_eq!(exec[1].ms_self, 2);
        // Native never tracks per-op bytes and never spills.
        assert!(exec.iter().all(|e| e.bytes.is_none() && !e.spill));
    }

    #[test]
    fn explain_surfaces_cost_model_advisory_once_warm() {
        // C4 slice 3: once the trace-driven cost model has warmed history for a
        // shape-class, the EXPLAIN reason discloses its (observe-mode) advisory.
        use crate::observability::io_trace::IoTraceSnapshot;
        use crate::query::route_cost_model::GLOBAL_ROUTE_COST_MODEL;
        let snap = IoTraceSnapshot {
            get_ops: 5,
            bytes_read: 1 << 20,
            ..Default::default()
        };
        // The GROUP BY query classifies as "olap/native"; warm Native there.
        for _ in 0..3 {
            GLOBAL_ROUTE_COST_MODEL.observe_by_label("olap/native", "Native(Volcano)", &snap);
        }
        // Publish the frozen consult table so the advisory is visible this query
        // (production recomputes debounced every N observations; tests force it).
        GLOBAL_ROUTE_COST_MODEL.recompute();
        let expl = explain_select_route("SELECT service, count(*) FROM events GROUP BY service")
            .expect("routable");
        // Observe-mode: the chosen engine is unchanged ...
        assert_eq!(expl.compute_route, "Native(Volcano)");
        // ... but the cost-model advisory is now visible in the EXPLAIN reason.
        assert!(
            expl.reason.contains("cost-model"),
            "advisory surfaced in EXPLAIN reason: {}",
            expl.reason
        );
    }
}

// =========================================================================
// Engagement gate + table-name collection (over the parsed AST)
// =========================================================================

/// True iff the query uses a feature the legacy single-table path can't serve
/// (join / GROUP BY / HAVING / aggregate projection / set-op / WITH), so it
/// should be routed to the algebra engine over real data. Simple single-table
/// SELECTs return false and stay on the legacy path.
fn query_engages_relational_engine(query: &SqlQuery) -> bool {
    // TD-185a: a WITH (CTE) header is itself beyond the legacy single-table
    // path — the CTE name is not a catalog table, so the legacy FROM-token
    // walk can only misparse it. Engage the relational/OLAP route; the
    // frontend's CTE decline then falls the query through to the DataFusion
    // floor (which plans CTEs, recursive included, natively).
    query.with.is_some() || set_expr_engages(&query.body)
}

/// TD-OLAP-4 operation dimension: classify the SELECT's OLAP operation from the AST
/// (syntax-only, zero route-time I/O). Feeds the cost-model shape-class so per-engine
/// samples accumulate per operation — the geometry the shadow ledger showed engines
/// win/lose on. Priority: grouped > string-heavy > metadata-elidable > scalar-agg.
fn query_operation_class(query: &SqlQuery) -> crate::query::compute_scheduler::OperationClass {
    use crate::query::compute_scheduler::OperationClass;
    let SetExpr::Select(select) = query.body.as_ref() else {
        return OperationClass::Other;
    };
    let has_group_by = match &select.group_by {
        GroupByExpr::All(_) => true,
        GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
    };
    if has_group_by {
        return OperationClass::Grouped;
    }
    // A LIKE/regex predicate → native can't push it down → DataFusion class.
    if select.selection.as_ref().is_some_and(expr_is_string_heavy) {
        return OperationClass::StringHeavy;
    }
    let has_agg = select.projection.iter().any(select_item_has_aggregate);
    if has_agg && select.having.is_none() {
        // Unfiltered + every projection a COUNT/MIN/MAX → footer-elidable.
        if select.selection.is_none()
            && select
                .projection
                .iter()
                .all(select_item_is_elidable_aggregate)
        {
            return OperationClass::MetadataElidable;
        }
        return OperationClass::ScalarAggregate;
    }
    OperationClass::Other
}

/// TD-ROUTE-1 eligibility signal (ADR-058: eligibility precedes selection):
/// does the plan anywhere contain a JOIN — top-level, inside a derived table /
/// nested join / CTE, or implied by a subquery PATH B lowers to a
/// semi/anti/left join? The Native arm cannot serve a join-bearing plan over
/// Parquet: Volcano *does* have hash/nested-loop join executors, but no
/// external-parquet scan is wired for it, and the vectorized
/// native-over-parquet path is single-scan-only — a flipped plan would be
/// served by the DataFusion floor while `finalize_route` stamps Native into
/// the io_trace (poisoning the cost model's EWMA cells), and `EXPLAIN ANALYZE`
/// would execute a diverging plan. [`query_operation_class`] cannot carry the
/// signal — a `JOIN … GROUP BY` classifies as `Grouped` (the FROM clause is
/// never inspected there) — hence a dedicated shape bit. Fail-closed: set-op /
/// non-`SELECT` bodies count as join-bearing.
fn query_join_bearing(query: &SqlQuery) -> bool {
    // Generic expression traversal covers scalar/IN/EXISTS subqueries in
    // projection, WHERE, HAVING, GROUP BY, ORDER BY, and other query clauses.
    // Each such subquery is hoisted or lowered to a join-shaped plan.
    if ast_has_subquery(query) {
        return true;
    }
    // A CTE can carry the join the outer body then scans.
    if let Some(with) = &query.with
        && with
            .cte_tables
            .iter()
            .any(|cte| query_join_bearing(&cte.query))
    {
        return true;
    }
    set_expr_join_bearing(query.body.as_ref())
}

fn ast_has_subquery<V: sqlparser::ast::Visit>(node: &V) -> bool {
    use std::ops::ControlFlow;

    matches!(
        sqlparser::ast::visit_expressions(node, |expr| {
            if matches!(
                expr,
                SqlExpr::InSubquery { .. } | SqlExpr::Exists { .. } | SqlExpr::Subquery(_)
            ) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }),
        ControlFlow::Break(())
    )
}

fn set_expr_join_bearing(body: &SetExpr) -> bool {
    match body {
        SetExpr::Query(q) => query_join_bearing(q),
        SetExpr::Select(select) => {
            // Top-level: multi-table FROM (comma join) or explicit JOINs.
            if select.from.len() > 1 || select.from.iter().any(|t| !t.joins.is_empty()) {
                return true;
            }
            // A single FROM factor can still hide a join: a derived table
            // (`FROM (SELECT … JOIN …) t`) or a nested join (`FROM (a JOIN b)`).
            if select
                .from
                .iter()
                .any(|t| table_factor_join_bearing(&t.relation))
            {
                return true;
            }
            // WHERE `IN`/`EXISTS` subqueries lower to Semi/Anti joins, and a
            // projection scalar subquery is hoisted into a LEFT JOIN (PATH B) —
            // join-bearing plans regardless of the FROM clause.
            select.selection.as_ref().is_some_and(where_has_subquery)
                || select.projection.iter().any(select_item_has_subquery)
        }
        // Set-ops / anything else: fail closed.
        _ => true,
    }
}

fn table_factor_join_bearing(relation: &TableFactor) -> bool {
    match relation {
        TableFactor::Derived { subquery, .. } => query_join_bearing(subquery),
        TableFactor::NestedJoin { .. } => true,
        _ => false,
    }
}

/// Does the predicate use a string-matching construct (`LIKE`/`ILIKE`/`SIMILAR TO`
/// or a `regexp_*` function)? Recursive over boolean structure.
fn expr_is_string_heavy(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Like { .. } | SqlExpr::ILike { .. } | SqlExpr::SimilarTo { .. } => true,
        SqlExpr::Function(f) => {
            let n = f.name.to_string().to_ascii_lowercase();
            matches!(
                n.rsplit('.').next().unwrap_or(&n),
                "regexp_replace" | "regexp_match" | "regexp_matches" | "regexp_like"
            )
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_is_string_heavy(left) || expr_is_string_heavy(right)
        }
        SqlExpr::UnaryOp { expr, .. } | SqlExpr::Nested(expr) | SqlExpr::Cast { expr, .. } => {
            expr_is_string_heavy(expr)
        }
        _ => false,
    }
}

/// Is this projection item a `COUNT`/`MIN`/`MAX` aggregate (footer-elidable, unlike
/// `SUM`/`AVG` which need column data)?
fn select_item_is_elidable_aggregate(item: &SelectItem) -> bool {
    let expr = match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => e,
        _ => return false,
    };
    if let SqlExpr::Function(f) = expr {
        let n = f.name.to_string().to_ascii_lowercase();
        matches!(n.rsplit('.').next().unwrap_or(&n), "count" | "min" | "max")
    } else {
        false
    }
}

/// A cheap, syntax-only cardinality hint from the parsed AST — the fallback
/// when the footer-warmed [`crate::query::route_cost_model::classify_table_shapes`]
/// stat is `Unknown` (native storage, or a cold Parquet table never yet scanned),
/// so the cost-model shape-class still discriminates instead of collapsing to
/// the coarse class. Returns [`CardinalityClass::Small`] only when the query
/// shape GUARANTEES a tiny result, else `Unknown` (never over-claims). Pure AST
/// walk — zero route-time I/O (co-design P5: I/O round-trips, not CPU, dominate;
/// the route path must not probe storage).
fn query_cardinality_hint(query: &SqlQuery) -> crate::query::compute_scheduler::CardinalityClass {
    use crate::query::compute_scheduler::CardinalityClass;
    // sqlparser 0.59 carries LIMIT on the Query as a `LimitClause` enum (standard
    // `LIMIT N [OFFSET]` vs MySQL `LIMIT offset, N`). A literal renders to a bare
    // number string; a placeholder / expression doesn't parse → stay Unknown
    // (never over-claim). Version-agnostic (no `Value::Number` probe).
    if let Some(rows) = query
        .limit_clause
        .as_ref()
        .and_then(limit_expr)
        .and_then(|e| e.to_string().parse::<u64>().ok())
    {
        return CardinalityClass::from_estimate(Some(rows));
    }
    cardinality_hint_body(&query.body)
}

/// The limit `Expr` from either `LimitClause` variant (`None` for `LIMIT ALL`).
fn limit_expr(clause: &sqlparser::ast::LimitClause) -> Option<&SqlExpr> {
    use sqlparser::ast::LimitClause;
    match clause {
        LimitClause::LimitOffset { limit, .. } => limit.as_ref(),
        LimitClause::OffsetCommaLimit { limit, .. } => Some(limit),
    }
}

fn cardinality_hint_body(body: &SetExpr) -> crate::query::compute_scheduler::CardinalityClass {
    use crate::query::compute_scheduler::CardinalityClass;
    match body {
        SetExpr::Select(select) => {
            // A scalar aggregate (aggregate projection, no GROUP BY / HAVING)
            // yields exactly one row.
            let has_agg = select.projection.iter().any(select_item_has_aggregate);
            let has_group_by = match &select.group_by {
                GroupByExpr::All(_) => true,
                GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
            };
            if has_agg && !has_group_by && select.having.is_none() {
                return CardinalityClass::Small;
            }
            CardinalityClass::Unknown
        }
        // A parenthesized subquery may carry its own LIMIT; a set-op branch's
        // LIMIT does not bound the whole result, so be conservative elsewhere.
        SetExpr::Query(q) => {
            if let Some(rows) = q
                .limit_clause
                .as_ref()
                .and_then(limit_expr)
                .and_then(|e| e.to_string().parse::<u64>().ok())
            {
                return CardinalityClass::from_estimate(Some(rows));
            }
            cardinality_hint_body(&q.body)
        }
        _ => CardinalityClass::Unknown,
    }
}

/// TD-EXEC-2 Slice 3: estimate the plan-geometry bands (depth × pipeline
/// breakers) from the AST — a pure syntax walk, zero route-time I/O (co-design
/// P5), computed BEFORE lowering because the route decision precedes the
/// physical plan. Key consistency is by construction: `finalize_route` stamps
/// the geometry-refined shape-class into the io_trace, so the consult key and
/// the EWMA fold key are the same string even when the estimate is imperfect.
/// The *measured* `plan_depth`/`plan_blocking` recorded per query by
/// `plan_instrumented` (TD-EXEC-2 Slice 1) sit beside the stamped class in the
/// ledger, exposing the prediction error for band refitting.
fn query_geometry_class(query: &SqlQuery) -> crate::query::compute_scheduler::GeometryClass {
    let (depth, blocking) = geometry_of_query(query);
    crate::query::compute_scheduler::GeometryClass::from_estimate(depth, blocking)
}

/// Estimated (longest root→leaf operator chain, pipeline-breaker count) for a
/// query. Mirrors what lowering emits: the body's plan plus `Sort` for ORDER BY
/// (a breaker, per `plan_geometry::OpKind::is_blocking`) and `Limit`.
fn geometry_of_query(query: &SqlQuery) -> (u32, u32) {
    let (mut depth, mut blocking) = geometry_of_body(&query.body);
    if query.order_by.is_some() {
        depth += 1; // Sort
        blocking += 1;
    }
    if query.limit_clause.is_some() {
        depth += 1; // Limit
    }
    (depth, blocking)
}

fn geometry_of_body(body: &SetExpr) -> (u32, u32) {
    match body {
        SetExpr::Select(select) => {
            // The deepest FROM chain feeds the plan's leaf slot; additional
            // comma-separated FROM items lower as cross joins above it.
            let mut depth = 1u32; // Scan/Values floor (also covers empty FROM)
            let mut blocking = 0u32;
            let mut first = true;
            for twj in &select.from {
                let (d, b) = geometry_of_table_with_joins(twj);
                blocking += b;
                if first {
                    depth = d;
                    first = false;
                } else {
                    depth = depth.max(d) + 1; // implicit cross join
                    blocking += 1;
                }
            }
            if select.selection.is_some() {
                depth += 1; // Filter
            }
            let has_group_by = match &select.group_by {
                GroupByExpr::All(_) => true,
                GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
            };
            let has_agg = select.projection.iter().any(select_item_has_aggregate);
            if has_group_by || has_agg {
                depth += 1; // Aggregate
                blocking += 1;
            }
            if select.having.is_some() {
                depth += 1; // post-aggregate Filter
            }
            // v2 (measured 2026-07-13): predicate/projection subqueries lower
            // to HOISTED join branches (semi/anti/left) — the deepest path may
            // run through the branch, and each hoist is a breaker. v1 ignored
            // them; TPC-H Q22 measured mxhi against an sxlo estimate.
            let mut subqueries: Vec<&SqlQuery> = Vec::new();
            if let Some(sel) = &select.selection {
                geometry_subqueries_of_expr(sel, &mut subqueries);
            }
            if let Some(having) = &select.having {
                geometry_subqueries_of_expr(having, &mut subqueries);
            }
            for item in &select.projection {
                if let SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } = item
                {
                    geometry_subqueries_of_expr(e, &mut subqueries);
                }
            }
            for sub in subqueries {
                let (sd, sb) = geometry_of_query(sub);
                depth = depth.max(sd) + 1; // hoisted Join above the deeper side
                blocking += sb + 1; // branch breakers + the join itself
            }
            // v2: a window function lowers to a Window node plus the sort it
            // implies — two extra levels (measured: win_rank/win_running
            // banded mxhi against an sxhi estimate; depth-only divergence, so
            // no breaker is added — mirrors `measure_logical_geometry`, which
            // labels window without counting it as blocking).
            if select.projection.iter().any(select_item_has_window) {
                depth += 2;
            }
            depth += 1; // Project
            if select.distinct.is_some() {
                depth += 1; // Distinct
            }
            (depth, blocking)
        }
        SetExpr::Query(q) => geometry_of_query(q),
        SetExpr::SetOperation {
            op, left, right, ..
        } => {
            let (dl, bl) = geometry_of_body(left);
            let (dr, br) = geometry_of_body(right);
            match op {
                // UNION streams through a concat node.
                SetOperator::Union => (dl.max(dr) + 1, bl + br),
                // v2: EXCEPT/INTERSECT lower to distinct-aggregates on BOTH
                // sides under an anti/semi join plus alias/projection wrappers
                // — not a single set node (measured: both banded mxhi against
                // an sxlo estimate; the DF lowering adds ~5 levels and 3
                // breakers: two side aggregates + the join).
                _ => (dl.max(dr) + 5, bl + br + 3),
            }
        }
        _ => (1, 0),
    }
}

/// One FROM item: the join chain lowers left-deep, so each JOIN adds one level
/// above its deeper input and counts as one pipeline breaker.
fn geometry_of_table_with_joins(twj: &TableWithJoins) -> (u32, u32) {
    let (mut depth, mut blocking) = geometry_of_table_factor(&twj.relation);
    for join in &twj.joins {
        let (d, b) = geometry_of_table_factor(&join.relation);
        depth = depth.max(d) + 1; // Join node above the deeper side
        blocking += b + 1; // the Join itself is a breaker
    }
    (depth, blocking)
}

fn geometry_of_table_factor(tf: &TableFactor) -> (u32, u32) {
    match tf {
        TableFactor::Derived { subquery, .. } => geometry_of_query(subquery),
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => geometry_of_table_with_joins(table_with_joins),
        _ => (1, 0), // named table → Scan leaf
    }
}

/// Collect the subqueries in an expression tree (scalar `(SELECT …)`,
/// `EXISTS`, `IN (SELECT …)`) — the shapes lowering hoists into join
/// branches. Traversal mirrors [`where_has_subquery`].
fn geometry_subqueries_of_expr<'a>(expr: &'a SqlExpr, out: &mut Vec<&'a SqlQuery>) {
    match expr {
        SqlExpr::Subquery(q) => out.push(q),
        SqlExpr::Exists { subquery, .. } => out.push(subquery),
        SqlExpr::InSubquery { subquery, .. } => out.push(subquery),
        SqlExpr::BinaryOp { left, right, .. } => {
            geometry_subqueries_of_expr(left, out);
            geometry_subqueries_of_expr(right, out);
        }
        SqlExpr::UnaryOp { expr, .. } | SqlExpr::Nested(expr) | SqlExpr::Cast { expr, .. } => {
            geometry_subqueries_of_expr(expr, out)
        }
        _ => {}
    }
}

/// True if a projection item applies a window function (`… OVER (…)`).
fn select_item_has_window(item: &SelectItem) -> bool {
    let expr = match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => e,
        _ => return false,
    };
    expr_has_window(expr)
}

fn expr_has_window(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Function(f) => f.over.is_some(),
        SqlExpr::BinaryOp { left, right, .. } => expr_has_window(left) || expr_has_window(right),
        SqlExpr::UnaryOp { expr, .. } | SqlExpr::Nested(expr) | SqlExpr::Cast { expr, .. } => {
            expr_has_window(expr)
        }
        _ => false,
    }
}

fn set_expr_engages(body: &SetExpr) -> bool {
    match body {
        SetExpr::SetOperation { .. } => true,
        SetExpr::Query(q) => set_expr_engages(&q.body),
        SetExpr::Select(select) => {
            let has_join =
                select.from.len() > 1 || select.from.iter().any(|twj| !twj.joins.is_empty());
            let has_group_by = match &select.group_by {
                GroupByExpr::All(_) => true,
                GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
            };
            let has_aggregate = select.projection.iter().any(select_item_has_aggregate);
            // A WHERE subquery (IN/EXISTS) lowers to a Semi/Anti join — a shape the
            // legacy single-table path can't serve — so engage PATH B. If it turns out
            // unliftable (correlated/NOT IN), lowering declines and the caller falls
            // through to legacy anyway, so engaging on ANY subquery is safe.
            let has_where_subquery = select.selection.as_ref().is_some_and(where_has_subquery);
            // A scalar subquery in the projection list (e.g.
            // `SELECT a, (SELECT max(x) FROM t)`) is hoisted by PATH B into a LEFT
            // JOIN; the legacy single-table path can't serve it. Same safety as the
            // WHERE case — an unliftable shape declines back to legacy.
            let has_projection_subquery = select.projection.iter().any(select_item_has_subquery);
            // A derived table (subquery in FROM, e.g. `FROM (SELECT … row_number()
            // OVER …) t WHERE rn = 1`, the TSBS last-point shape) is not something
            // the legacy single-table path can serve — engage the relational/OLAP
            // route so it reaches DataFusion.
            let has_derived = select.from.iter().any(table_with_joins_has_derived);
            // TD-183: a JSON-path extraction (`->`/`->>`/`json_extract_path_text`,
            // lowered to `JSON_EXTRACT[_TEXT]` calls by the translator) in a plain
            // projection or WHERE must engage the relational/OLAP route. Otherwise the
            // non-engaging SELECT falls through to store-type dispatch, is misread as a
            // *document* query because the SQL text contains `JSON_EXTRACT`, and is
            // served by the document handler against an empty collection → 0 rows.
            // Aggregated forms (e.g. d09 GROUP BY) already engage via `has_group_by`.
            let has_json_extract = select.projection.iter().any(select_item_has_json_extract)
                || select.selection.as_ref().is_some_and(expr_has_json_extract);
            // TD-XMODAL-4 S1: a table-valued function in FROM (a `TableFactor::Table`
            // carrying `args`, e.g. `vector_search('c','[..]',k)` / `timeseries_range`
            // / `graph_traverse`) is a DataFusion UDTF — the legacy single-table path
            // can't serve it, so engage the relational/OLAP route to reach DataFusion
            // where the UDTFs are registered. This is what makes a STANDALONE
            // `SELECT * FROM vector_search(...)` resolve over pgwire (before, only a
            // JOIN engaged). Default-safe: a plain table has `args: None`.
            let has_table_function = select
                .from
                .iter()
                .any(|twj| matches!(&twj.relation, TableFactor::Table { args: Some(_), .. }));
            has_join
                || has_group_by
                || select.having.is_some()
                || has_aggregate
                || has_where_subquery
                || has_projection_subquery
                || has_derived
                || has_json_extract
                || has_table_function
        }
        _ => false,
    }
}

/// True if a projected item applies a JSON-path extraction function. Mirrors
/// [`select_item_has_aggregate`]; used by the engagement gate (TD-183).
fn select_item_has_json_extract(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
            expr_has_json_extract(expr)
        }
        _ => false,
    }
}

/// True if `expr` contains a `JSON_EXTRACT` / `JSON_EXTRACT_TEXT` /
/// `json_extract_path_text` call anywhere in its tree. The extraction is always the
/// outermost function or is wrapped in Cast/Nested/Unary/Binary (e.g.
/// `(JSON_EXTRACT_TEXT(doc,'price'))::int > 8`), so walking those node kinds — the
/// same set [`expr_has_aggregate`] walks — is sufficient.
fn expr_has_json_extract(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Function(f) => {
            let name = f.name.to_string().to_ascii_lowercase();
            matches!(
                name.rsplit('.').next().unwrap_or(&name),
                "json_extract" | "json_extract_text" | "json_extract_path_text"
            )
        }
        SqlExpr::Nested(inner) => expr_has_json_extract(inner),
        SqlExpr::UnaryOp { expr, .. } => expr_has_json_extract(expr),
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_has_json_extract(left) || expr_has_json_extract(right)
        }
        SqlExpr::Cast { expr, .. } => expr_has_json_extract(expr),
        _ => false,
    }
}

/// True if a WHERE expression contains a subquery (`IN (…)`, `EXISTS`, or a scalar
/// subquery) anywhere in its tree — used by the engagement gate to route subquery
/// predicates onto the relational (Semi/Anti) path.
fn where_has_subquery(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::InSubquery { .. } | SqlExpr::Exists { .. } | SqlExpr::Subquery(_) => true,
        SqlExpr::BinaryOp { left, right, .. } => {
            where_has_subquery(left) || where_has_subquery(right)
        }
        SqlExpr::UnaryOp { expr, .. } | SqlExpr::Nested(expr) => where_has_subquery(expr),
        _ => false,
    }
}

/// True if a projection item's expression contains a subquery (scalar `(SELECT …)`,
/// `EXISTS`, or `IN (SELECT …)`) anywhere in its tree — routes a SELECT whose output
/// list carries a subquery onto PATH B, which hoists it into a join. Reuses the
/// general subquery-detection of [`where_has_subquery`].
fn select_item_has_subquery(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
            where_has_subquery(e)
        }
        _ => false,
    }
}

fn select_item_has_aggregate(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
            expr_has_aggregate(expr)
        }
        _ => false,
    }
}

fn expr_has_aggregate(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::Function(f) => {
            let name = f.name.to_string().to_ascii_lowercase();
            matches!(
                name.rsplit('.').next().unwrap_or(&name),
                "count"
                    | "sum"
                    | "avg"
                    | "min"
                    | "max"
                    | "stddev"
                    | "stddev_pop"
                    | "stddev_samp"
                    | "variance"
                    | "var_pop"
                    | "var_samp"
            )
        }
        SqlExpr::Nested(inner) => expr_has_aggregate(inner),
        SqlExpr::UnaryOp { expr, .. } => expr_has_aggregate(expr),
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_has_aggregate(left) || expr_has_aggregate(right)
        }
        SqlExpr::Cast { expr, .. } => expr_has_aggregate(expr),
        _ => false,
    }
}

/// Collect every catalog table referenced by `query`, returning the set of
/// CTE-local names declared by any (possibly nested) `WITH` clause.
///
/// TD-185a: base tables inside CTE bodies must resolve into the snapshot
/// (the executor/DATAFusion floor needs their schemas), while a FROM
/// reference to a CTE name must NOT be resolved as a catalog table — before
/// this, the CTE name was collected as a table, schema resolution failed,
/// and the whole WITH query silently bounced to the legacy single-table
/// path (the recursive-CTE empty-result bug).
fn collect_table_names(
    query: &SqlQuery,
    out: &mut Vec<String>,
) -> std::collections::HashSet<String> {
    let mut cte_names = std::collections::HashSet::new();
    collect_query_tables(query, out, &mut cte_names);
    cte_names
}

fn collect_query_tables(
    query: &SqlQuery,
    out: &mut Vec<String>,
    cte_names: &mut std::collections::HashSet<String>,
) {
    // Nested WITH scopes add their CTE names to the shared set. SQL shadowing
    // semantics would confine a CTE name to its own scope; the shared set is
    // wider, and only in the pathological case of a real table shadowed by a
    // same-named CTE outside its scope — acceptable for snapshot preparation.
    if let Some(with) = &query.with {
        for cte in &with.cte_tables {
            cte_names.insert(normalize_table_key(&cte.alias.name.value));
            collect_query_tables(&cte.query, out, cte_names);
        }
    }
    collect_from_set_expr(&query.body, out, cte_names);
}

fn collect_from_set_expr(
    body: &SetExpr,
    out: &mut Vec<String>,
    cte_names: &mut std::collections::HashSet<String>,
) {
    match body {
        SetExpr::Select(select) => {
            for twj in &select.from {
                collect_from_table_with_joins(twj, out, cte_names);
            }
            // Also descend into WHERE subqueries (IN/EXISTS lower to Semi/Anti joins
            // whose right side scans the subquery's tables) so those tables get
            // prepared in the snapshot — otherwise the subquery fails to lower and
            // the query silently falls through to the legacy path.
            if let Some(where_expr) = &select.selection {
                collect_subquery_tables_in_expr(where_expr, out, cte_names);
            }
        }
        SetExpr::Query(q) => collect_query_tables(q, out, cte_names),
        SetExpr::SetOperation { left, right, .. } => {
            collect_from_set_expr(left, out, cte_names);
            collect_from_set_expr(right, out, cte_names);
        }
        _ => {}
    }
}

/// Collect table names referenced by subqueries inside a WHERE expression
/// (`IN (…)`, `EXISTS`, scalar subqueries), recursing through boolean structure.
fn collect_subquery_tables_in_expr(
    expr: &SqlExpr,
    out: &mut Vec<String>,
    cte_names: &mut std::collections::HashSet<String>,
) {
    match expr {
        SqlExpr::InSubquery { subquery, .. } | SqlExpr::Exists { subquery, .. } => {
            collect_query_tables(subquery, out, cte_names);
        }
        SqlExpr::Subquery(q) => collect_query_tables(q, out, cte_names),
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_subquery_tables_in_expr(left, out, cte_names);
            collect_subquery_tables_in_expr(right, out, cte_names);
        }
        SqlExpr::UnaryOp { expr, .. } | SqlExpr::Nested(expr) => {
            collect_subquery_tables_in_expr(expr, out, cte_names);
        }
        _ => {}
    }
}

/// True if the FROM item (or any of its joins) is a derived table — a subquery in
/// FROM. Such a shape must engage the relational/OLAP route (the legacy single-table
/// path can't serve it).
fn table_with_joins_has_derived(twj: &TableWithJoins) -> bool {
    matches!(twj.relation, TableFactor::Derived { .. })
        || twj
            .joins
            .iter()
            .any(|j| matches!(j.relation, TableFactor::Derived { .. }))
}

fn collect_from_table_with_joins(
    twj: &TableWithJoins,
    out: &mut Vec<String>,
    cte_names: &mut std::collections::HashSet<String>,
) {
    collect_from_table_factor(&twj.relation, out, cte_names);
    for join in &twj.joins {
        collect_from_table_factor(&join.relation, out, cte_names);
    }
}

fn collect_from_table_factor(
    factor: &TableFactor,
    out: &mut Vec<String>,
    cte_names: &mut std::collections::HashSet<String>,
) {
    match factor {
        // A `FROM name(args)` item is a table-valued function (cross-modal source:
        // vector_search / timeseries_range / graph_traverse), NOT a catalog table.
        // Collecting it as a table name makes catalog resolution decline the query to
        // the legacy path AND breaks the parquet-backed route check. Skip it here and
        // let the DataFusion `ctx.sql` fallback (which has the UDTFs registered) resolve it.
        // Only a plain `name` (args: None) is a catalog table. A `FROM name(args)` item is a
        // table-valued function (cross-modal source: vector_search / timeseries_range /
        // graph_traverse) — it matches neither arm below, falls through to `_`, and is left for
        // the DataFusion `ctx.sql` fallback (where the UDTFs are registered) to resolve.
        // TD-185a: a reference to a WITH-defined name resolves by the SQL
        // engine, not the catalog — the match guard skips it so it can't fail
        // snapshot preparation.
        TableFactor::Table {
            name, args: None, ..
        } if !cte_names.contains(&normalize_table_key(&name.to_string())) => {
            out.push(name.to_string());
        }
        TableFactor::Table { args: None, .. } => {}
        TableFactor::Derived { subquery, .. } => collect_query_tables(subquery, out, cte_names),
        _ => {}
    }
}

// =========================================================================
// PgType / text encoding helpers
// =========================================================================

/// Map a [`ProximaType`] to the closest [`PgType`] for
/// `RowDescription` emission. Unknowns map to `Text` so clients
/// can still display values.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pgvector_operator_lowers_to_typed_vector_plan() {
        let plan = lower_pgvector_select(
            "SELECT id, payload FROM public.memories \
             WHERE payload->>'type' = 'fact' AND priority >= 3 \
             ORDER BY embedding <-> '[0.1, -0.2, 0.3]'::vector LIMIT 7",
        )
        .expect("vector SQL must parse")
        .expect("vector operator must engage");

        assert_eq!(plan.search.collection, "public.memories");
        assert_eq!(plan.search.vector_column.as_deref(), Some("embedding"));
        assert_eq!(plan.search.query_vector, vec![0.1, -0.2, 0.3]);
        assert_eq!(plan.search.top_k, 7);
        assert_eq!(
            plan.search.metric,
            proximadb_distance_types::DistanceMetric::L2
        );
        assert_eq!(
            plan.output,
            vec![
                PgVectorOutputColumn::Id("id".to_string()),
                PgVectorOutputColumn::Metadata {
                    source: "payload".to_string(),
                    output: "payload".to_string(),
                },
            ]
        );
        assert_eq!(
            plan.search.filter,
            Some(proximadb_filter_expression::FilterExpression::And(vec![
                proximadb_filter_expression::FilterExpression::Comparison {
                    field: "payload.type".to_string(),
                    operator: proximadb_filter_expression::ComparisonOperator::Equals,
                    value: serde_json::json!("fact"),
                },
                proximadb_filter_expression::FilterExpression::Comparison {
                    field: "priority".to_string(),
                    operator: proximadb_filter_expression::ComparisonOperator::GreaterThanOrEqual,
                    value: serde_json::json!(3),
                },
            ]))
        );
    }

    #[test]
    fn pgvector_operator_detection_uses_the_ast_not_literal_or_comment_text() {
        for sql in [
            "SELECT '<->' AS operator_name",
            "SELECT 1 /* <=> is documentation, not an expression */",
            "SELECT '<#>' AS operator_name ORDER BY operator_name",
        ] {
            assert_eq!(
                lower_pgvector_select(sql),
                Ok(None),
                "non-operator SQL must stay on the relational path: {sql}"
            );
            assert!(
                matches!(describe_pgvector_select(sql), Ok(None)),
                "describe must make the same AST-based routing decision: {sql}"
            );
        }
    }

    #[test]
    fn pgvector_filter_accepts_translator_normalized_json_path_ast() {
        let plan = lower_pgvector_select(
            "SELECT id FROM memories \
             WHERE JSON_EXTRACT_TEXT(payload, 'type') = 'fact' \
             ORDER BY embedding <-> '[0.1, 0.2]'::vector LIMIT 5",
        )
        .expect("normalized vector SQL must parse")
        .expect("vector operator must engage");

        assert_eq!(
            plan.search.filter,
            Some(proximadb_filter_expression::FilterExpression::Comparison {
                field: "payload.type".to_string(),
                operator: proximadb_filter_expression::ComparisonOperator::Equals,
                value: serde_json::json!("fact"),
            })
        );
    }

    #[test]
    fn pgvector_operator_preserves_each_metric() {
        for (operator, expected) in [
            ("<->", proximadb_distance_types::DistanceMetric::L2),
            ("<=>", proximadb_distance_types::DistanceMetric::Cosine),
            (
                "<#>",
                proximadb_distance_types::DistanceMetric::InnerProduct,
            ),
        ] {
            let sql = format!(
                "SELECT id FROM docs ORDER BY embedding {operator} '[1,2]'::vector LIMIT 2"
            );
            let plan = lower_pgvector_select(&sql)
                .expect("vector SQL must parse")
                .expect("vector operator must engage");
            assert_eq!(plan.search.metric, expected, "operator {operator}");
        }
    }

    #[test]
    fn pgvector_distance_uses_metric_tagged_distance_not_normalized_score() {
        let record = proximadb_search_types::results::OptimizedSearchRecord {
            id: "doc-1".to_string(),
            score: 0.9,
            semantic_similarity: Some(proximadb_distance_kernel::SimilarityResult::new(
                6.0,
                proximadb_distance_kernel::DistanceMetric::DotProduct,
            )),
            ..Default::default()
        };

        assert_eq!(
            pgvector_distance(
                &record,
                proximadb_distance_types::DistanceMetric::InnerProduct
            ),
            Ok(-6.0),
            "<#> exposes negative inner product, not normalized similarity"
        );
        assert!(
            pgvector_distance(&record, proximadb_distance_types::DistanceMetric::L2).is_err(),
            "a distance computed under another metric must fail closed"
        );
    }

    #[test]
    fn pgvector_projection_is_sql_shaped_and_rejects_metric_drift() {
        let plan = lower_pgvector_select(
            "SELECT id, embedding <=> '[1,2]'::vector AS cosine_distance, payload \
             FROM docs ORDER BY embedding <=> '[1,2]'::vector LIMIT 2",
        )
        .expect("vector SQL must parse")
        .expect("vector operator must engage");
        assert_eq!(
            plan.output,
            vec![
                PgVectorOutputColumn::Id("id".to_string()),
                PgVectorOutputColumn::Distance("cosine_distance".to_string()),
                PgVectorOutputColumn::Metadata {
                    source: "payload".to_string(),
                    output: "payload".to_string(),
                },
            ]
        );

        let error = lower_pgvector_select(
            "SELECT embedding <#> '[1,2]'::vector AS wrong FROM docs \
             ORDER BY embedding <=> '[1,2]'::vector LIMIT 2",
        )
        .expect_err("projection and ordering metrics must agree");
        assert!(error.contains("must use the ORDER BY metric"), "{error}");
    }

    #[test]
    fn pgvector_projection_alias_does_not_replace_its_catalog_source() {
        let plan = lower_pgvector_select(
            "SELECT payload AS body FROM docs \
             ORDER BY embedding <-> '[1,2]'::vector LIMIT 2",
        )
        .expect("vector SQL must parse")
        .expect("vector operator must engage");

        assert_eq!(
            plan.output,
            vec![PgVectorOutputColumn::Metadata {
                source: "payload".to_string(),
                output: "body".to_string(),
            }]
        );
    }

    #[test]
    fn pgvector_operator_rejects_predicates_it_cannot_push_down() {
        let error = lower_pgvector_select(
            "SELECT id FROM docs WHERE lower(category) = 'fact' \
             ORDER BY embedding <-> '[1,2]'::vector LIMIT 2",
        )
        .expect_err("unsupported filter must not widen to an unfiltered search");

        assert!(
            error.contains("unsupported vector-search filter"),
            "{error}"
        );
    }

    #[test]
    fn pgvector_limit_cannot_wrap_the_canonical_top_k() {
        let error = lower_pgvector_select(
            "SELECT id FROM docs ORDER BY embedding <-> '[1,2]'::vector LIMIT 4294967296",
        )
        .expect_err("LIMIT must not truncate while lowering into the canonical u32 top_k");
        assert!(error.contains("u32 result cardinality"), "{error}");
    }

    #[test]
    fn pgvector_query_vector_rejects_non_finite_components() {
        let error = lower_pgvector_select(
            "SELECT id FROM docs ORDER BY embedding <-> '[NaN,2]'::vector LIMIT 2",
        )
        .expect_err("non-finite components must not enter distance or heap ordering");
        assert!(error.contains("must be finite"), "{error}");
    }

    fn gate(sql: &str) -> bool {
        let statements = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
        match statements.as_slice() {
            [Statement::Query(query)] => query_engages_relational_engine(query),
            _ => panic!("expected a single SELECT statement"),
        }
    }

    /// TD-XMODAL-4 S1: a STANDALONE table-valued function in FROM (a DataFusion
    /// UDTF like `vector_search`/`timeseries_range`/`graph_traverse`) must engage the
    /// relational route so it reaches DataFusion where the UDTF is registered — this
    /// is what makes `SELECT * FROM vector_search(...)` resolve over pgwire (before,
    /// only a JOIN engaged; a lone UDTF FROM fell through to the legacy path).
    #[test]
    fn standalone_table_function_from_engages() {
        assert!(
            gate("SELECT * FROM vector_search('c', '[0.1,0.2,0.3]', 5)"),
            "lone vector_search UDTF must engage the relational route"
        );
        assert!(
            gate("SELECT id, score FROM vector_search('c', '[0.1]', 10) WHERE score > 0.5"),
            "UDTF FROM engages even without a join"
        );
        assert!(
            gate("SELECT * FROM timeseries_range('c', 0, 100)"),
            "other UDTFs engage the same way"
        );
        // Default-safe: a plain single-table SELECT does NOT engage via this signal
        // (it has `args: None`) — it stays on the hardened legacy OLTP path.
        assert!(!gate("SELECT id, name FROM users WHERE id = 1"));
    }

    /// ADR-058 D5 / §9.A: the stats-trust derivation is the load-bearing security
    /// mapping behind the native route's footer-elision gate — an authority that
    /// wrongly classifies external parquet as `Trusted` would re-open the
    /// wrong-`COUNT`/`MIN`/`MAX`-on-external-parquet hole #860 closed. This ratchet
    /// locks the mapping: external parquet authorities ⇒ `Untrusted`; everything
    /// ProximaDB authored ⇒ `Trusted`; and the gate is parquet-footer-specific (an
    /// external authority on a non-parquet layout is NOT a footer-trust concern).
    #[cfg(feature = "datafusion-integration")]
    #[test]
    fn parquet_stats_trust_maps_external_authorities_to_untrusted() {
        use proximadb_catalog::{
            CatalogAuthorityMode, CatalogPhysicalFormat, CatalogStorageLayout, CatalogTableSchema,
        };
        let layout = |authority, physical_format| CatalogStorageLayout {
            authority,
            physical_format,
            ..Default::default()
        };
        let schema_with = |layouts: Vec<CatalogStorageLayout>| {
            let mut s = CatalogTableSchema::new("t");
            s.storage_layouts = layouts;
            s
        };

        // External / federated / imported PARQUET → Untrusted (footer not ours).
        for authority in [
            CatalogAuthorityMode::FederatedRead,
            CatalogAuthorityMode::ExternalAuthoritative,
            CatalogAuthorityMode::ImportedSnapshot,
        ] {
            let s = schema_with(vec![layout(authority, CatalogPhysicalFormat::Parquet)]);
            assert_eq!(
                parquet_stats_trust(&s),
                StatsTrust::Untrusted,
                "{authority:?} parquet must be Untrusted — its footer stats are not ProximaDB's"
            );
        }

        // ProximaDB-authored PARQUET (incl. the `MATERIALIZE` form) → Trusted.
        for authority in [
            CatalogAuthorityMode::ProjectionPublication,
            CatalogAuthorityMode::ExportedPublication,
        ] {
            let s = schema_with(vec![layout(authority, CatalogPhysicalFormat::Parquet)]);
            assert_eq!(
                parquet_stats_trust(&s),
                StatsTrust::Trusted,
                "{authority:?} parquet is ProximaDB-authored ⇒ footer stats are authoritative"
            );
        }

        // The gate is parquet-footer-specific: an external authority on a
        // NON-parquet layout must not taint trust (no parquet footer is trusted).
        let s = schema_with(vec![layout(
            CatalogAuthorityMode::FederatedRead,
            CatalogPhysicalFormat::ProximaBlock,
        )]);
        assert_eq!(
            parquet_stats_trust(&s),
            StatsTrust::Trusted,
            "an external authority on a non-parquet layout is not a footer-trust concern"
        );

        // No storage layouts at all ⇒ nothing external ⇒ Trusted (fail-open is safe
        // here: with no parquet layout the elision path is never reached).
        assert_eq!(
            parquet_stats_trust(&CatalogTableSchema::new("t")),
            StatsTrust::Trusted
        );
    }

    #[test]
    fn operation_class_classifies_olap_shapes() {
        use crate::query::compute_scheduler::OperationClass;
        let op = |sql: &str| {
            let s = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
            match s.as_slice() {
                [Statement::Query(q)] => query_operation_class(q),
                _ => panic!("one SELECT"),
            }
        };
        // Unfiltered COUNT(*) / MIN / MAX → footer-elidable (native wins).
        assert_eq!(
            op("SELECT COUNT(*) FROM hits"),
            OperationClass::MetadataElidable
        );
        assert_eq!(
            op("SELECT MIN(eventdate), MAX(eventdate) FROM hits"),
            OperationClass::MetadataElidable
        );
        // SUM/AVG need column data → scalar aggregate.
        assert_eq!(
            op("SELECT SUM(advengineid), AVG(resolutionwidth) FROM hits"),
            OperationClass::ScalarAggregate
        );
        // A plain (non-string) filter is still a scalar aggregate.
        assert_eq!(
            op("SELECT COUNT(*) FROM hits WHERE advengineid <> 0"),
            OperationClass::ScalarAggregate
        );
        // LIKE predicate → string-heavy (routes to DataFusion).
        assert_eq!(
            op("SELECT COUNT(*) FROM hits WHERE url LIKE '%google%'"),
            OperationClass::StringHeavy
        );
        // GROUP BY → grouped.
        assert_eq!(
            op("SELECT regionid, COUNT(*) FROM hits GROUP BY regionid"),
            OperationClass::Grouped
        );
    }

    #[test]
    fn geometry_class_bands_separate_scan_from_join_shapes() {
        use crate::query::compute_scheduler::{BlockingBand, DepthBand, GeometryClass};
        let geom = |sql: &str| {
            let s = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
            match s.as_slice() {
                [Statement::Query(q)] => query_geometry_class(q),
                _ => panic!("one SELECT"),
            }
        };
        // Plain filtered scan (scan→filter→project) — shallow, no breakers.
        assert_eq!(
            geom("SELECT id FROM t WHERE x = 1"),
            GeometryClass::Known {
                depth: DepthBand::Shallow,
                blocking: BlockingBand::Zero,
            }
        );
        // ClickBench posture: single-table aggregate — shallow, low breakers
        // (the shapes TD-OLAP-15 measured native winning).
        assert_eq!(
            geom("SELECT COUNT(*) FROM hits"),
            GeometryClass::Known {
                depth: DepthBand::Shallow,
                blocking: BlockingBand::Low,
            }
        );
        // TPC-H posture: join chain + GROUP BY + ORDER BY — high breakers
        // (the shapes TD-OLAP-15 measured DataFusion winning).
        let tpch_ish = geom(
            "SELECT c.name, SUM(o.total) FROM customer c \
             JOIN orders o ON c.id = o.cid \
             JOIN lineitem l ON o.id = l.oid \
             WHERE o.d > 5 GROUP BY c.name ORDER BY 2 DESC LIMIT 10",
        );
        assert_eq!(
            tpch_ish,
            GeometryClass::Known {
                depth: DepthBand::Mid,
                blocking: BlockingBand::High,
            }
        );
        // A derived-table subquery deepens the leaf; set-ops sum branch breakers.
        let nested = geom(
            "SELECT a FROM (SELECT a FROM u JOIN v ON u.i = v.i \
             GROUP BY a ORDER BY a) t WHERE a > 0 ORDER BY a",
        );
        assert!(
            matches!(
                nested,
                GeometryClass::Known {
                    depth: DepthBand::Mid | DepthBand::Deep,
                    blocking: BlockingBand::High,
                }
            ),
            "nested join+agg+two sorts must band mid/deep × high: {nested:?}"
        );
    }

    #[test]
    fn geometry_estimator_v2_covers_the_measured_gaps() {
        use crate::query::compute_scheduler::{BlockingBand, DepthBand, GeometryClass};
        let raw = |sql: &str| {
            let s = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
            match s.as_slice() {
                [Statement::Query(q)] => geometry_of_query(q),
                _ => panic!("one SELECT"),
            }
        };
        let geom = |sql: &str| {
            let (d, b) = raw(sql);
            GeometryClass::from_estimate(d, b)
        };
        // Gap 1 — predicate subqueries (TPC-H Q22 shape: v1 banded sxlo,
        // measured mxhi). Each subquery is a hoisted join branch.
        let q22ish = geom(
            "SELECT cntrycode, COUNT(*) FROM customer              WHERE c_acctbal > (SELECT AVG(c_acctbal) FROM customer WHERE c_acctbal > 0)              AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)              GROUP BY cntrycode ORDER BY cntrycode",
        );
        assert_eq!(
            q22ish,
            GeometryClass::Known {
                depth: DepthBand::Mid,
                blocking: BlockingBand::High,
            },
            "Q22 shape must band mxhi (measured 10x5)"
        );
        // Gap 2 — window functions add Window + implied-sort levels
        // (depth-only; window is not a breaker in the OpKind vocabulary).
        let (d_plain, b_plain) = raw("SELECT x FROM t");
        let (d_win, b_win) = raw("SELECT rank() OVER (ORDER BY x) FROM t");
        assert_eq!(d_win, d_plain + 2, "window adds two levels");
        assert_eq!(b_win, b_plain, "window adds no breaker");
        // Gap 3 — EXCEPT/INTERSECT lower to side-aggregates + join, unlike
        // UNION's streaming concat (measured: simple set-ops banded mxhi).
        let except = geom("SELECT a FROM t EXCEPT SELECT a FROM u");
        assert_eq!(
            except,
            GeometryClass::Known {
                depth: DepthBand::Mid,
                blocking: BlockingBand::High,
            },
            "EXCEPT must band mid x high"
        );
        let union = geom("SELECT a FROM t UNION ALL SELECT a FROM u");
        assert_eq!(
            union,
            GeometryClass::Known {
                depth: DepthBand::Shallow,
                blocking: BlockingBand::Zero,
            },
            "UNION ALL stays a streaming concat"
        );
    }

    #[cfg(feature = "datafusion-integration")]
    fn scan_fixture(cols: &[&str]) -> PhysicalPlan {
        PhysicalPlan::Scan {
            table: TableId::new("hits"),
            output_schema: RelationalSchema::new(
                cols.iter()
                    .map(|c| ColumnInfo::new(*c, ProximaType::Int64, false))
                    .collect(),
            ),
            projection: None,
            predicate: None,
            limit: None,
            access: proximadb_relational_planner::ScanAccess::FullScan,
        }
    }

    #[cfg(feature = "datafusion-integration")]
    #[test]
    fn single_scan_columns_detects_sole_scan_and_rejects_multi_or_none() {
        // Bare scan → its projected column names.
        assert_eq!(
            single_scan_columns(&scan_fixture(&["k", "x"])),
            Some(vec!["k".to_string(), "x".to_string()])
        );
        // Recurses through an intermediate op (Limit) to the sole scan.
        let limited = PhysicalPlan::Limit {
            input: Box::new(scan_fixture(&["k", "x"])),
            limit: Some(10),
            offset: 0,
        };
        assert_eq!(
            single_scan_columns(&limited),
            Some(vec!["k".to_string(), "x".to_string()])
        );
        // Two scans (Union) → None: the native shadow is single-table only.
        let two = PhysicalPlan::Union {
            inputs: vec![scan_fixture(&["k"]), scan_fixture(&["x"])],
            all: true,
        };
        assert_eq!(single_scan_columns(&two), None);
        // No scan at all → None.
        let no_scan = PhysicalPlan::Values {
            rows: vec![],
            output_schema: RelationalSchema::new(vec![]),
        };
        assert_eq!(single_scan_columns(&no_scan), None);
    }

    #[test]
    fn gate_engages_for_join_groupby_aggregate_and_union() {
        // Queries the legacy single-table path can't serve → route to PATH B.
        assert!(gate(
            "SELECT emp.name FROM emp JOIN dept ON emp.dept_id = dept.id"
        ));
        assert!(gate("SELECT status, COUNT(*) FROM inv GROUP BY status"));
        assert!(gate("SELECT COUNT(*) AS n FROM inv"));
        assert!(gate(
            "SELECT status FROM inv GROUP BY status HAVING COUNT(*) > 1"
        ));
        assert!(gate("SELECT id FROM a UNION SELECT id FROM b"));
        assert!(gate("SELECT SUM(qty) AS s FROM inv"));
        // WHERE subqueries (IN/EXISTS) lower to Semi/Anti joins → engage PATH B,
        // even on a single FROM table (incl. when ANDed with a plain predicate).
        assert!(gate("SELECT id FROM inv WHERE id IN (SELECT id FROM dept)"));
        assert!(gate(
            "SELECT id FROM inv WHERE EXISTS (SELECT id FROM dept)"
        ));
        assert!(gate(
            "SELECT id FROM inv WHERE qty > 5 AND id IN (SELECT id FROM dept)"
        ));
    }

    #[test]
    fn gate_skips_simple_single_table_selects() {
        // These stay on the (hardened) legacy path — gate must NOT engage.
        assert!(!gate("SELECT * FROM inv"));
        assert!(!gate(
            "SELECT id FROM inv WHERE status = 'active' OR qty >= 30"
        ));
        assert!(!gate("SELECT id FROM inv ORDER BY qty LIMIT 5"));
        assert!(!gate("SELECT id, status FROM inv WHERE id = 'i1'"));
    }

    /// ADR-025 PR3: the OLAP read-merge is default-ON, with an explicit falsy
    /// `PROXIMADB_OLAP_DELTA_MERGE` token as the engine-wide kill-switch. Tests the
    /// pure policy directly so it never has to mutate the process environment.
    #[cfg(feature = "datafusion-integration")]
    #[test]
    fn olap_delta_merge_default_on_with_explicit_killswitch() {
        // Default-ON: unset and any non-falsy value enable the merge.
        assert!(olap_delta_merge_on(None));
        for on in ["1", "true", "on", "yes", "anything", ""] {
            assert!(olap_delta_merge_on(Some(on)), "`{on}` must keep merge ON");
        }
        // Kill-switch: explicit falsy tokens (case-insensitive, trimmed) disable it.
        for off in ["0", "false", "off", "no", "OFF", " False ", "No"] {
            assert!(
                !olap_delta_merge_on(Some(off)),
                "`{off}` must disable merge"
            );
        }
    }

    fn card_hint(sql: &str) -> crate::query::compute_scheduler::CardinalityClass {
        let statements = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
        match statements.as_slice() {
            [Statement::Query(query)] => query_cardinality_hint(query),
            _ => panic!("expected a single SELECT statement"),
        }
    }

    #[test]
    fn cardinality_hint_small_for_limit_and_scalar_aggregate() {
        use crate::query::compute_scheduler::CardinalityClass;
        // LIMIT n bounds the result (a literal renders to a bare number string).
        assert_eq!(
            card_hint("SELECT * FROM inv LIMIT 10"),
            CardinalityClass::Small
        );
        // A large LIMIT still buckets via from_estimate (≤1M → Medium).
        assert_eq!(
            card_hint("SELECT * FROM inv LIMIT 100000"),
            CardinalityClass::Medium
        );
        // A scalar aggregate (no GROUP BY / HAVING) → exactly one row → Small.
        assert_eq!(
            card_hint("SELECT COUNT(*) FROM inv"),
            CardinalityClass::Small
        );
        assert_eq!(
            card_hint("SELECT SUM(qty), AVG(qty) FROM inv"),
            CardinalityClass::Small
        );
        // GROUP BY defeats the scalar-aggregate signal (could be many groups).
        assert_eq!(
            card_hint("SELECT status, COUNT(*) FROM inv GROUP BY status"),
            CardinalityClass::Unknown
        );
        // A plain unbounded scan → Unknown (never over-claim).
        assert_eq!(card_hint("SELECT * FROM inv"), CardinalityClass::Unknown);
        // A non-literal LIMIT expression doesn't parse to a count → Unknown.
        assert_eq!(
            card_hint("SELECT * FROM inv LIMIT 1 + 1"),
            CardinalityClass::Unknown
        );
    }

    #[test]
    fn collect_table_names_covers_join_and_normalizes() {
        let statements = Parser::parse_sql(
            &GenericDialect {},
            "SELECT * FROM Emp JOIN \"dept\" ON Emp.d = dept.id",
        )
        .expect("parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };
        let mut names = Vec::new();
        collect_table_names(query, &mut names);
        let keys: Vec<String> = names.iter().map(|n| normalize_table_key(n)).collect();
        assert!(keys.contains(&"emp".to_string()), "got {keys:?}");
        assert!(keys.contains(&"dept".to_string()), "got {keys:?}");
    }

    fn parse_query(sql: &str) -> SqlQuery {
        let statements = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };
        (**query).clone()
    }

    #[test]
    fn join_bearing_detects_joins_at_every_nesting_level() {
        // TD-ROUTE-1 follow-up (review of #923): the bit must see joins the
        // top-level FROM walk misses, or the override poisons Native's cost
        // cells (the DataFusion floor serves the query while the io_trace
        // stamps Native) and EXPLAIN ANALYZE diverges.
        let jb = |sql: &str| query_join_bearing(&parse_query(sql));

        // Top-level: explicit JOIN and comma join.
        assert!(jb("SELECT * FROM a JOIN b ON a.id = b.id"));
        assert!(jb("SELECT * FROM a, b WHERE a.id = b.id"));
        // Derived table hiding a join.
        assert!(jb(
            "SELECT count(*) FROM (SELECT a.id FROM a JOIN b ON a.id = b.id) t"
        ));
        // CTE carrying the join the body then scans.
        assert!(jb(
            "WITH x AS (SELECT a.id FROM a JOIN b ON a.id = b.id) SELECT * FROM x"
        ));
        // WHERE IN / EXISTS lower to semi/anti joins.
        assert!(jb("SELECT * FROM a WHERE id IN (SELECT id FROM b)"));
        assert!(jb(
            "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.id = a.id)"
        ));
        // Projection scalar subquery hoists into a LEFT JOIN.
        assert!(jb("SELECT id, (SELECT max(x) FROM b) FROM a"));
        // Subqueries outside SELECT/WHERE are still hoisted into join-shaped
        // plans and must not remain eligible for Native cost overrides.
        assert!(jb(
            "SELECT a.id FROM a GROUP BY a.id, (SELECT max(id) FROM b)"
        ));
        assert!(jb("SELECT count(*) FROM a HAVING EXISTS (SELECT 1 FROM b)"));
        assert!(jb("SELECT id FROM a ORDER BY (SELECT max(id) FROM b)"));
        // Set-op body: fail closed.
        assert!(jb("SELECT id FROM a UNION ALL SELECT id FROM b"));

        // Join-free shapes stay eligible for a Native flip.
        assert!(!jb("SELECT * FROM a WHERE id = 1"));
        assert!(!jb("SELECT status, count(*) FROM a GROUP BY status"));
        assert!(!jb(
            "SELECT status, count(*) FROM a GROUP BY status HAVING count(*) > 1 ORDER BY status"
        ));
        assert!(!jb(
            "SELECT count(*) FROM (SELECT id FROM a WHERE id > 5) t"
        ));
        assert!(!jb("WITH x AS (SELECT id FROM a) SELECT * FROM x"));
    }

    #[test]
    fn equality_predicate_columns_extracts_and_chained_literals_only() {
        // AND-chained equalities → both columns; `t.col` → bare name; literal on
        // either side works.
        let q = parse_query("SELECT * FROM t WHERE status = 'open' AND 7 = t.priority");
        let mut cols = equality_predicate_columns(&q);
        cols.sort();
        assert_eq!(cols, vec!["priority".to_string(), "status".to_string()]);

        // No WHERE → empty.
        assert!(equality_predicate_columns(&parse_query("SELECT * FROM t")).is_empty());
        // Non-equality (range) and col=col (no literal) → ignored.
        assert!(
            equality_predicate_columns(&parse_query("SELECT * FROM t WHERE a > 1 AND b = c"))
                .is_empty()
        );
        // OR is not AND-chained equality → ignored (not estimable here).
        assert!(
            equality_predicate_columns(&parse_query("SELECT * FROM t WHERE a = 1 OR b = 2"))
                .is_empty()
        );
    }

    #[test]
    fn stats_filter_estimate_multiplies_and_estimates_rows() {
        let cols = vec!["a".to_string(), "b".to_string(), "unknown".to_string()];
        // a: 1/4, b: 1/5, unknown: no stat. product = 0.05; rows = ceil(1000*0.05)=50.
        let (sel, rows) = stats_filter_estimate(&cols, Some(1000), |f| match f {
            "a" => Some(0.25),
            "b" => Some(0.2),
            _ => None,
        })
        .expect("at least one field has a stat");
        assert!((sel - 0.05).abs() < 1e-9, "selectivity {sel}");
        assert_eq!(rows, Some(50));

        // No field has a stat → None (EXPLAIN left unchanged, never guessed).
        assert!(stats_filter_estimate(&cols, Some(1000), |_| None).is_none());
        // Stat but no record_count → selectivity present, rows None.
        let (sel2, rows2) =
            stats_filter_estimate(&["a".to_string()], None, |_| Some(0.1)).expect("has stat");
        assert!((sel2 - 0.1).abs() < 1e-9);
        assert_eq!(rows2, None);
    }

    #[test]
    fn collect_table_names_descends_into_where_subqueries() {
        let statements = Parser::parse_sql(
            &GenericDialect {},
            "SELECT ename FROM emp WHERE dept_id IN (SELECT id FROM dept WHERE dname = 'eng')",
        )
        .expect("parse");
        let [Statement::Query(query)] = statements.as_slice() else {
            panic!("expected query");
        };
        let mut names = Vec::new();
        collect_table_names(query, &mut names);
        let keys: Vec<String> = names.iter().map(|n| normalize_table_key(n)).collect();
        // Both the outer table AND the subquery table must be collected so the
        // snapshot prepares `dept` (else the Semi-join subquery can't lower).
        assert!(keys.contains(&"emp".to_string()), "got {keys:?}");
        assert!(keys.contains(&"dept".to_string()), "got {keys:?}");
    }

    #[test]
    fn collect_table_names_treats_cte_names_as_engine_local_not_catalog_tables() {
        // TD-185a: the CTE name must not be collected as a catalog table (that
        // resolution failure bounced WITH queries to the legacy path), while the
        // base tables INSIDE the CTE body must be collected for the snapshot.
        let query = parse_query(
            "WITH reach(id) AS (SELECT 1 UNION SELECT k.k_person2 FROM knows k \
             JOIN reach r ON k.k_person1 = r.id) \
             SELECT count(distinct id) FROM reach",
        );
        let mut names = Vec::new();
        let cte_names = collect_table_names(&query, &mut names);
        let keys: Vec<String> = names.iter().map(|n| normalize_table_key(n)).collect();
        assert!(
            keys.contains(&"knows".to_string()),
            "CTE body base table must be collected, got {keys:?}"
        );
        assert!(
            !keys.iter().any(|k| k == "reach"),
            "CTE name must not be collected as a catalog table, got {keys:?}"
        );
        assert!(
            cte_names.contains("reach"),
            "got {cte_keys:?}",
            cte_keys = cte_names
        );
    }

    #[test]
    fn with_header_always_engages_the_relational_engine() {
        // TD-185a: a WITH header is beyond the legacy single-table path even
        // when the outer body is a plain single-table scan over the CTE — the
        // legacy FROM-token walk can only misparse the CTE name.
        assert!(query_engages_relational_engine(&parse_query(
            "WITH x AS (SELECT id FROM t) SELECT id FROM x"
        )));
        assert!(query_engages_relational_engine(&parse_query(
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 9) \
             SELECT sum(n) FROM r"
        )));
    }

    #[test]
    fn normalize_table_key_strips_qualifier_and_quotes() {
        assert_eq!(normalize_table_key("Inv"), "inv");
        assert_eq!(normalize_table_key("public.Orders"), "orders");
        assert_eq!(normalize_table_key("\"Mixed\""), "mixed");
    }

    #[test]
    fn snapshot_capabilities_advertises_single_col_pk_only() {
        let mut pk_by_table = HashMap::new();
        pk_by_table.insert("users".to_string(), vec![0]); // single-col PK at ordinal 0
        pk_by_table.insert("edges".to_string(), Vec::new()); // composite/no-PK → none
        let mut secondary_by_table = HashMap::new();
        secondary_by_table.insert("users".to_string(), vec!["name".to_string()]);
        let resolver = SnapshotCapabilities {
            pk_by_table,
            secondary_by_table,
        };

        // Single-col PK table → planner can pick PkLookup (name normalized).
        assert_eq!(resolver.primary_key(&TableId::new("users")), vec![0]);
        assert_eq!(resolver.primary_key(&TableId::new("USERS")), vec![0]);
        // Composite/no-PK and unknown tables → empty → planner keeps full scan.
        assert!(resolver.primary_key(&TableId::new("edges")).is_empty());
        assert!(resolver.primary_key(&TableId::new("unknown")).is_empty());
        // Pushdown capabilities unchanged (pk_lookup gated per-table by primary_key).
        assert!(resolver.capabilities(&TableId::new("users")).pk_lookup);
        // TD-128/TD-127: batch + secondary advertised (gated per-table below).
        assert!(
            resolver
                .capabilities(&TableId::new("users"))
                .pk_lookup_batch
        );
        assert!(
            resolver
                .capabilities(&TableId::new("users"))
                .secondary_lookup
        );
        // TD-127: secondary-indexed columns surfaced per-table (name-normalized);
        // unknown tables → empty → planner keeps a full scan.
        assert_eq!(
            resolver.secondary_index_columns(&TableId::new("USERS")),
            vec!["name".to_string()]
        );
        assert!(
            resolver
                .secondary_index_columns(&TableId::new("edges"))
                .is_empty()
        );
    }

    #[test]
    fn text_encode_null_is_none_and_exotic_types_render() {
        // SQL NULL → None (the caller emits the pgwire `-1` null sentinel).
        // This is the core of the pgwire NULL-rendering fix.
        assert_eq!(text_encode(&ProximaValue::Null), None);

        // Scalars render as before.
        assert_eq!(
            text_encode(&ProximaValue::String("alice".into())),
            Some("alice".to_string())
        );
        assert_eq!(
            text_encode(&ProximaValue::Boolean(false)),
            Some("f".to_string())
        );

        // Exotic types now render explicitly rather than as a Debug fallback.
        assert_eq!(
            text_encode(&ProximaValue::Array(vec![
                ProximaValue::Int32(1),
                ProximaValue::Int32(2),
            ])),
            Some("{1,2}".to_string())
        );
        // A NULL element inside an array uses the empty-field convention.
        assert_eq!(
            text_encode(&ProximaValue::Array(vec![
                ProximaValue::Int32(1),
                ProximaValue::Null,
            ])),
            Some("{1,}".to_string())
        );

        // UUID renders dashed (Postgres format), not raw hex.
        let uuid = ProximaValue::Uuid([
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
            0x00, 0x00,
        ]);
        let rendered = text_encode(&uuid).expect("uuid must encode");
        assert_eq!(rendered, "550e8400-e29b-41d4-a716-446655440000");
        assert!(
            rendered.contains('-'),
            "UUID must be dashed (Postgres format)"
        );
    }
}
