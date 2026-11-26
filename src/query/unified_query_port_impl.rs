//! Root-crate implementation of `UnifiedQueryPort`.
//!
//! Wraps `QueryFacadeAdapter` (for query execution) and `PreparedStatementCache`
//! (for parse-once-execute-many) so `proximadb-api`'s multimodal REST handlers
//! can delegate to real business logic without importing root-crate concrete types.
//!
//! Phase 9.9: this impl unblocks all nine `/api/v1/unified/*` endpoints in
//! `crates/platform/proximadb-api/src/rest/canonical/multimodal_query.rs`.
//!
//! **One decision path (ADR-098 D5 / TD-SPINE-6):** plain relational SQL
//! delegates to [`QueryFacadeAdapter::execute_sql_via_authority`] — the same
//! EXPLAIN/write/`try_run_select` body (whose SELECT arm is the
//! `ComputeScheduler::route_select` decision) that serves pgwire and
//! `api_handlers.execute_sql`. SQL with multi-model extensions keeps its
//! federated lowering (`adapter.federated_query`): those queries carry no
//! resolvable relational tables, so the scheduler decision has nothing to
//! decide — engine selection for vector/graph/document search happens inside
//! the engine layer the facade strategies drive. The discriminator is the
//! adapter's ONE `should_use_federated_request`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_runtime::{
    InvalidQueryInput, MissingQueryIdentity, OwnedPortIdentity, PortIdentity,
    QueryDeadlineExceeded, UnifiedQueryPort, UnifiedQueryPortContext,
};
// The FILTER-lowering spelling (int-array binary, ns temporals) — named
// for what it IS, mirroring the SqlValue twin sql_value_to_filter_literal;
// the canonical (base64) renderer lives in records under a different name.
use proximadb_search_types::search_params::VectorFreshnessMode;
use proximadb_search_types::sql_value_filter::proxima_value_to_filter_literal as proxima_filter_literal;
use tracing::{debug, info};

use crate::catalog::CatalogManager;
use crate::query::authority_context::{AuthoritySource, resolve_catalog_authority_context};
use crate::query::explain::StorageAuthorityExplanation;
use crate::query::multimodal::plan::PlanContext;
use crate::query::prepared::statement::{
    escape_sql_text, float_sql_literal, sql_quote, vector_literal_text,
};
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
        // Json/Array/Map flow through the exotic catch-all below — one
        // spelling of 'structured value → JSON-text param' (the deleted
        // per-variant arms re-derived it and drifted from the catch-all).
        // A JSON-null DOCUMENT is the SQL NULL param (3VL), not the string
        // 'null' the catch-all would splice — matching From<Value>'s arm.
        ProximaValue::Json(v) | ProximaValue::Jsonb(v) if v.is_null() => ParameterValue::Null,
        ProximaValue::Null => ParameterValue::Null,
        // Typed exotics (Binary/Uuid/ULID/temporals/SparseVector/Decimal)
        // lower through the ONE shared filter spelling — the old Rust-Debug
        // strings ("Binary([1, 2, 3])") could never equal a stored value's
        // rendering, so such parameters silently matched nothing. Structured
        // values map to their JSON TEXT as a String param — with
        // to_sql_string quoting both String and Json params, the direct and
        // literal paths agree on QUOTING (numeric-array text and
        // UInt64>i64::MAX spellings still differ between them; the literal
        // path's f32 vector coercion is pre-existing).
        other => match proxima_filter_literal(other) {
            serde_json::Value::String(text) => ParameterValue::String(text),
            // JSON scalars keep their SEMANTIC param type — bool/number
            // splice BARE like the From<Value> route (a String param would
            // quote them into string-vs-numeric no-matches).
            serde_json::Value::Bool(b) => ParameterValue::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ParameterValue::Int(i)
                } else {
                    ParameterValue::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            json => ParameterValue::String(json.to_string()),
        },
    }
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
        // One shared non-finite rule; the f32-width arms render NATIVE
        // precision (widening to f64 changes the literal: 0.1f32 ->
        // 0.10000000149011612, silently unmatching equality filters).
        ProximaValue::Float16(value) | ProximaValue::Float32(value) => Ok(if value.is_finite() {
            value.to_string()
        } else {
            "NULL".to_string()
        }),
        ProximaValue::Float64(value) => Ok(float_sql_literal(*value)),
        ProximaValue::Boolean(value) => Ok(if *value { "TRUE" } else { "FALSE" }.to_string()),
        ProximaValue::DenseVector(values) => vector_to_sql_literal(values),
        ProximaValue::Array(_) => match proxima_value_to_f32_vector(value) {
            Some(vector) => vector_to_sql_literal(&vector),
            None => exotic_literal(value),
        },
        // Json/Jsonb and Map/Struct flow through the exotic catch-all:
        // their literals must render exactly what the filter evaluator
        // renders on the stored side (a root-string Json lowers to BARE
        // text there — the old serde_json spelling embedded the quotes
        // and could never match).
        ProximaValue::Null => Ok("NULL".to_string()),
        other => exotic_literal(other),
    }
}

/// Structured/exotic literals — the filter-lowering spelling, QUOTED: these
/// must render exactly what the filter evaluator renders on the stored side
/// (a root-string Json lowers to BARE text there; the old serde_json
/// spelling embedded the quotes and could never match). KNOWN GAPS
/// (tracked in TD-PROTO-2, dialect-dependent): temporals splice as quoted
/// epoch-numbers, and Binary/SparseVector as quoted JSON text — parseable
/// SQL on every engine, but equality against a native binary/timestamp
/// column needs the per-dialect literal form (ISO-8601 text for
/// Postgres-style engines).
fn exotic_literal(value: &ProximaValue) -> Result<String> {
    // ONE splice definition: delegate to ParameterValue::Json's arm (the
    // shape rule — null→SQL NULL, scalars bare, strings/containers
    // quoted — lived here as a byte-identical third copy).
    Ok(ParameterValue::Json(proxima_filter_literal(value)).to_sql_string())
}

fn vector_to_sql_literal(values: &[f32]) -> Result<String> {
    if values.iter().any(|component| !component.is_finite()) {
        return Err(anyhow!("query vector components must be finite"));
    }
    Ok(sql_quote(&vector_literal_text(values)))
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
            let Some(query_param) = select.where_clause.as_ref().and_then(|where_clause| {
                where_clause
                    .conditions
                    .iter()
                    .find_map(|condition| match condition {
                        Condition::VectorSimilar { query_param, .. }
                        | Condition::VectorDistance { query_param, .. } => Some(*query_param),
                        _ => None,
                    })
            }) else {
                // TD-SPINE-6: a SELECT the UQL grammar parses but that carries
                // no vector condition is NOT a UQL claim — the UQL grammar
                // overlaps plain SQL and an un-prefixed FROM defaults to the
                // Vector model. Claim nothing (Ok(None)) so the caller routes
                // the text as plain relational SQL through the one planner
                // seam; the pre-convergence behavior for this shape was a
                // hard error, so nothing working changes.
                return Ok(None);
            };
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
            let vector_literal = vector_to_sql_literal(&vector)?;
            Ok(Some(format!(
                "SELECT * FROM VECTOR_SEARCH({}, {}, {})",
                sql_quote(&select.from.collection),
                vector_literal,
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
    /// Client-facing prepared-statement handles: a random id per statement,
    /// owned by the tenant that prepared it. Only the owner can execute,
    /// delete or count it; any other id reads as not found.
    prepared: Arc<dashmap::DashMap<String, PreparedHandle>>,
}

/// A client-facing prepared statement: its owner, the shared cache's id, and
/// the owner's execution count.
#[derive(Debug, Clone)]
struct PreparedHandle {
    owner: String,
    cache_id: String,
    executions: u64,
}

/// The owner key for prepared-statement handles: the caller's EXPLICIT
/// identity tenant when it carries one (ADR-098 D5 — identity as data), else
/// the task-local request scope in force (the fallback for callers that
/// cannot construct an identity, e.g. bare-mounted test routers), else none —
/// no scope owns nothing.
fn prepared_owner(identity_tenant: Option<&str>) -> Option<String> {
    if let Some(tenant) = identity_tenant.filter(|tenant| !tenant.is_empty()) {
        return Some(tenant.to_string());
    }
    match crate::query::request_scope::current_scope()? {
        crate::query::request_scope::RequestScope::Tenant(tenant) => Some(tenant),
        crate::query::request_scope::RequestScope::Internal(_) => Some("_internal".to_string()),
    }
}

/// The identity the ONE SQL authority executes as (MED-3, round-1 review):
///
/// * the caller's EXPLICIT identity tenant wins when it carries one;
/// * else a `Tenant` request scope in force is **derived into the identity**
///   (an anonymous-identity call inside `with_request_tenant(t)` runs fully
///   scoped as `t` — it can never execute unscoped, so the documented
///   task-local fallback no longer opens a shared unscoped namespace);
/// * else an `Internal` scope stays in force via the authority's
///   `with_identity_scope(None)` (explicitly-internal execution keeps its
///   designed unscoped-collection semantics);
/// * else **fail closed** with the typed [`MissingQueryIdentity`] (401 at the
///   HTTP surfaces — TD-SPINE-6 acceptance: a request without identity fails
///   closed on every surface).
fn effective_identity_or_fail(identity: &PortIdentity<'_>) -> Result<OwnedPortIdentity> {
    if identity.tenant_id.is_some_and(|tenant| !tenant.is_empty()) {
        return Ok(identity.into_owned());
    }
    match crate::query::request_scope::current_scope() {
        Some(crate::query::request_scope::RequestScope::Tenant(tenant)) => {
            let mut derived = identity.into_owned();
            derived.tenant_id = Some(tenant);
            Ok(derived)
        }
        // Internal scope: keep the (empty) identity — the authority's
        // `with_identity_scope(None)` leaves the internal scope in force.
        Some(crate::query::request_scope::RequestScope::Internal(_)) => Ok(identity.into_owned()),
        None => Err(MissingQueryIdentity(
            "unified query has no request identity; the caller must act as a tenant",
        )
        .into()),
    }
}

/// MED-4 (round-1 review) — dated behavior note 2026-11-24: **plain SQL on
/// the unified query port is READ-ONLY.** The port surfaces are query
/// surfaces (`UnifiedQueryPort`, MCP `search`, `/api/v2/query`,
/// `/api/v2/unified/*`); writes are served by the guarded write surfaces —
/// `/api/v2/sql` (single-statement validation + 1 MiB cap + deadline),
/// gRPC `ExecuteQuery`, and pgwire. A DDL/DML statement on a port surface is
/// rejected as a client error naming the policy, never silently executed or
/// silently lost. Round 2 (2026-11-25): reads = a single PURE query
/// (`SELECT` / `WITH…SELECT` / `VALUES`, no write anywhere in the
/// tree — see [`query_is_pure_read`]); EVERY `EXPLAIN` form (including
/// `EXPLAIN SELECT`) is rejected with a message naming EXPLAIN, and the
/// accepted read then runs through the authority's read-only entry.
fn ensure_plain_sql_is_read(sql: &str) -> Result<()> {
    use sqlparser::ast::Statement as SqlStatement;
    use sqlparser::dialect::PostgreSqlDialect;
    use sqlparser::parser::Parser;

    // PostgreSqlDialect (MED-1, round 2): the relational pipeline's own
    // lowering (`lower_pgvector_select`) parses with it — the GenericDialect
    // does not tokenize `<->`, which rejected valid KNN reads.
    let statements = Parser::parse_sql(&PostgreSqlDialect {}, sql)
        .map_err(|error| InvalidQueryInput(format!("cannot parse statement: {error}")))?;
    let (is_read, explain_shaped) = match statements.as_slice() {
        // Exactly one statement — the port, like /api/v2/sql, is a
        // single-statement surface. HIGH-1 (round 2): read-vs-write is
        // decided on the AST, never on a text substring or a parser flag —
        // `sqlparser`'s `analyze` flag is false for EVERY parenthesized
        // option list (`(ANALYZE)`, `(FORMAT JSON, ANALYZE)`, `(ANALYZE
        // off)`, lowercase…), which let `EXPLAIN (ANALYZE) <DML>` through
        // the flag-based rule while the authority's explainer executed the
        // write. EXPLAIN is no longer advertised on the port at all (L7):
        // the authority's `EXPLAIN SELECT` path is a pre-existing 500, so
        // the honest contract is a clear rejection naming EXPLAIN.
        [SqlStatement::Query(query)] => (query_is_pure_read(query), false),
        [SqlStatement::Explain { .. }] => (false, true),
        _ => (false, false),
    };
    if is_read {
        return Ok(());
    }
    let message = if explain_shaped {
        "EXPLAIN is not served on the read-only unified query port; use /api/v2/sql (REST), gRPC ExecuteQuery, or pgwire"
    } else {
        "plain SQL on the unified query port is read-only (pure SELECT); writes are served by /api/v2/sql (REST), gRPC ExecuteQuery, or pgwire"
    };
    Err(InvalidQueryInput(message.to_string()).into())
}

/// A `Query` is a pure read only when NOTHING in it can execute a write
/// (HIGH-1, round 2): data-modifying set-expression bodies
/// (`SetExpr::Insert/Update/Delete/Merge`, including parenthesized and
/// set-operation positions like `SELECT 1 UNION (DELETE … RETURNING *)`),
/// data-modifying CTEs (`WITH del AS (DELETE …) SELECT …`), and
/// `SELECT … INTO` all reject.
///
/// The walk is the sqlparser AST visitor, so EVERY nested `Query` is checked
/// — CTE bodies, parenthesized bodies, set-operation operands, derived
/// tables in FROM, and expression subqueries alike — and ANY nested
/// non-query `Statement` (the payload of `SetExpr::Insert/Update/Delete/
/// Merge`) fails the read. No position in the tree is exempt.
fn query_is_pure_read(query: &sqlparser::ast::Query) -> bool {
    use sqlparser::ast::{Query, Statement, Visit, Visitor};
    use std::ops::ControlFlow;

    struct WriteFinder;
    impl Visitor for WriteFinder {
        type Break = ();
        fn pre_visit_statement(&mut self, statement: &Statement) -> ControlFlow<()> {
            // The walk starts at a `Query`, so any `Statement` reached is a
            // nested statement body — only a nested query is a read.
            if matches!(statement, Statement::Query(_)) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        }
        fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
            if set_expr_is_pure_read(query.body.as_ref()) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        }
    }
    query.visit(&mut WriteFinder).is_continue()
}

fn set_expr_is_pure_read(body: &sqlparser::ast::SetExpr) -> bool {
    use sqlparser::ast::SetExpr;
    match body {
        SetExpr::Select(select) => select.into.is_none(), // SELECT … INTO writes
        SetExpr::Values(_) | SetExpr::Table(_) => true,
        SetExpr::Query(inner) => query_is_pure_read(inner),
        SetExpr::SetOperation { left, right, .. } => {
            set_expr_is_pure_read(left) && set_expr_is_pure_read(right)
        }
        SetExpr::Insert(_) | SetExpr::Update(_) | SetExpr::Delete(_) | SetExpr::Merge(_) => false,
    }
}

/// L4 (round 2): the port surfaces carry /api/v2/sql's query-size contract —
/// a plain-SQL body is capped at 1 MiB.
const MAX_PORT_QUERY_BYTES: usize = 1024 * 1024;

fn ensure_query_size(sql: &str) -> Result<()> {
    if sql.len() > MAX_PORT_QUERY_BYTES {
        return Err(InvalidQueryInput(format!(
            "query exceeds the {MAX_PORT_QUERY_BYTES}-byte limit"
        ))
        .into());
    }
    Ok(())
}

/// The OLAP result cache the delegated scheduler path may consult, per the
/// requested freshness (mandate #16c): a Strong read BYPASSES the cache
/// lookup — it always computes fresh; weaker modes hand the cache through
/// (hits are LSN-pinned inside, never staler than the requested mode).
///
/// LOW-5 (round-1 review), decided 2026-11-24: no live surface exposes a
/// freshness knob, so every port call defaults to `Strong` (bypass) — which
/// is exactly these surfaces' PRE-PR behavior (they never consulted the OLAP
/// result cache). `UnifiedQueryPortContext::with_freshness` is the seam for
/// a future surface-level knob; wiring one is deliberately NOT invented
/// here.
fn olap_cache_for_freshness(
    enabled_cache: Option<
        Arc<
            crate::query::cache::query_result_cache::QueryResultCache<
                crate::query::execution::ExecutionPipelineResult,
            >,
        >,
    >,
    freshness: &VectorFreshnessMode,
) -> Option<
    Arc<
        crate::query::cache::query_result_cache::QueryResultCache<
            crate::query::execution::ExecutionPipelineResult,
        >,
    >,
> {
    match freshness {
        // Mandate #16c: gate the lookup on `!Strong`. (Mutation-proven
        // 2026-11-24: removing this arm fails
        // `olap_cache_policy_strong_bypasses_the_result_cache` in the
        // default env.)
        VectorFreshnessMode::Strong => None,
        _ => enabled_cache,
    }
}

/// Whether `sql` is text the PORT itself synthesized from a UQL or
/// multi-model lowering (trusted extension SQL) versus RAW user text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortSqlOrigin {
    /// Synthesized by `uql_to_federated_sql` / `json_to_multi_model_sql`:
    /// genuine extension SQL (`SELECT * FROM VECTOR_SEARCH(...)` etc.) —
    /// takes the federated branch directly, keeping its `{id, score, rank}`
    /// rendering.
    PortLowered,
    /// Verbatim user text (bound parameters spliced in): MUST go through the
    /// ONE SQL authority FIRST. The authority itself falls back to the
    /// federated facade for genuine extension SQL — but only AFTER
    /// `try_run_select` declines, so plain SQL that merely CONTAINS an
    /// extension keyword (in a string literal or a comment) executes as the
    /// plain SQL it is (HIGH-1, round-1 review: the raw-substring
    /// discriminator pre-routing here silently dropped such writes and
    /// mis-executed such reads).
    RawUserText,
}

/// Run `sql` through the ONE SQL authority when it is raw user text, or
/// through the federated lowering when it is port-synthesized extension SQL —
/// and render either into the port's JSON envelope. This is the single
/// planner seam every port SQL entry funnels through (ADR-098 D5 /
/// TD-SPINE-6). Raw text is classified read-only on the AST (HIGH-1/2,
/// round 2) and executes through `execute_sql_read_only_via_authority`,
/// which structurally cannot reach the EXPLAIN-DML explainer or the write
/// dispatch.
///
/// Round 3 (L3): the whole execution runs under
/// [`proximadb_runtime::PORT_QUERY_DEADLINE`] — enforced HERE, at the one
/// seam, so every surface over the port carries the same deadline and maps
/// the typed [`QueryDeadlineExceeded`] to its timeout status (504 over HTTP).
/// The deadline stops waiting (the future is dropped); it does not cancel
/// work already handed to spawned or blocking tasks.
async fn run_sql_via_one_planner(
    adapter: &Arc<QueryFacadeAdapter>,
    sql: &str,
    collection: Option<String>,
    identity: PortIdentity<'_>,
    freshness: &VectorFreshnessMode,
    limit: Option<u32>,
    origin: PortSqlOrigin,
) -> Result<serde_json::Value> {
    let deadline = proximadb_runtime::PORT_QUERY_DEADLINE;
    tokio::time::timeout(
        deadline,
        run_sql_via_one_planner_within_deadline(
            adapter, sql, collection, identity, freshness, limit, origin,
        ),
    )
    .await
    .map_err(|_| anyhow::Error::new(QueryDeadlineExceeded(deadline)))?
}

async fn run_sql_via_one_planner_within_deadline(
    adapter: &Arc<QueryFacadeAdapter>,
    sql: &str,
    collection: Option<String>,
    identity: PortIdentity<'_>,
    freshness: &VectorFreshnessMode,
    limit: Option<u32>,
    origin: PortSqlOrigin,
) -> Result<serde_json::Value> {
    // L4: the size cap applies to every text the port executes (the bound /
    // lowered text — each entry also checks the RAW input before lowering).
    ensure_query_size(sql)?;
    if origin == PortSqlOrigin::PortLowered {
        // Port-synthesized extension SQL (VECTOR_SEARCH/GRAPH_QUERY/
        // DOCUMENT_QUERY/LOGS/METRICS/RERANK): keep the federated lowering +
        // its result rendering. No relational table resolves for these, so
        // the ComputeScheduler decision has nothing to decide; engine
        // selection happens inside the engine layer behind the facade
        // strategies. L2 (round 2): the fail-closed gate is typed here too —
        // `FederatedQueryContext` denies a scope-less request with an
        // untyped (500-class) error.
        effective_identity_or_fail(&identity)?;
        let result = adapter
            .federated_query(sql)
            .await
            .context("federated_query failed")?;
        return UnifiedQueryPortImpl::result_to_json(result, limit);
    }
    // RAW user text: AST read classification (HIGH-1, round 2 — never a
    // substring or a parser flag), identity resolution per MED-3 (explicit
    // tenant, else scope-derived, else internal in force, else 401), then
    // the READ-ONLY authority entry (HIGH-1: structurally cannot reach the
    // EXPLAIN-DML explainer or the write dispatch — only `try_run_select`,
    // the `ComputeScheduler::route_select` decision, and the facade fallback
    // which promotes GENUINE extension SQL to the federated executor).
    ensure_plain_sql_is_read(sql)?;
    let effective = effective_identity_or_fail(&identity)?;
    let result_cache = olap_cache_for_freshness(
        crate::network::postgres::relational_pipeline::olap_result_cache(),
        freshness,
    );
    let started = std::time::Instant::now();
    let mut result = adapter
        .execute_sql_read_only_via_authority(
            sql.to_string(),
            collection,
            effective.as_borrowed(),
            result_cache.as_deref(),
        )
        .await
        // A typed client rejection (e.g. the pgvector ORDER BY KNN shape this
        // path cannot serve) keeps its own actionable message; anything else
        // gains the authority context.
        .map_err(|error| {
            if error.downcast_ref::<InvalidQueryInput>().is_some() {
                error
            } else {
                error.context("sql authority execution failed")
            }
        })?;
    // LOW-4: the authority body does not time itself (the runtime HANDLER
    // wrapper does); time the delegated call here so the port's metrics are
    // real.
    let elapsed_ms = started.elapsed().as_millis() as u64;
    if result.execution_time_ms < elapsed_ms {
        result.execution_time_ms = elapsed_ms;
    }
    sql_result_to_port_json(result, limit)
}

/// Render a `SqlExecutionResult` (the shared authority's typed shape) into
/// the port's JSON envelope (`records` / `total_count` / `metrics`) — the
/// shape the canonical `/sql` handler, `/api/v2/query` and MCP read.
fn sql_result_to_port_json(
    result: proximadb_runtime::SqlExecutionResult,
    limit: Option<u32>,
) -> Result<serde_json::Value> {
    let limit = limit.unwrap_or(u32::MAX) as usize;
    let rows: Vec<serde_json::Value> = result
        .rows
        .iter()
        .take(limit)
        .map(|row| {
            let mut object = serde_json::Map::with_capacity(row.len());
            for (index, value) in row.iter().enumerate() {
                let column = result
                    .columns
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("column_{index}"));
                object.insert(
                    column,
                    proximadb_records::conversions::proxima_to_json(value),
                );
            }
            serde_json::Value::Object(object)
        })
        .collect();
    let total_count = match result.rows_affected {
        // The affected count for DDL/DML (rows is empty; limit is a read
        // notion); the post-limit row count for reads — accurate for page
        // consumers, matching the legacy envelope's semantics.
        Some(affected) => affected,
        None => rows.len() as u64,
    };
    Ok(serde_json::json!({
        "records": rows,
        // Post-limit count for reads; the affected count for writes.
        "total_count": total_count,
        "metrics": {
            "rows_scanned": result.rows_scanned,
            "rows_affected": result.rows_affected.unwrap_or(0),
            "execution_time_ms": result.execution_time_ms,
        },
    }))
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
            prepared: Arc::new(dashmap::DashMap::new()),
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
            prepared: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// The shared cache id of `statement_id` when the caller owns it.
    fn owned_cache_id(&self, statement_id: &str, identity_tenant: Option<&str>) -> Option<String> {
        let owner = prepared_owner(identity_tenant)?;
        let handle = self.prepared.get(statement_id)?;
        (handle.owner == owner).then(|| handle.cache_id.clone())
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
    crate::core::utils::collect_sql_catalog_targets(sql, &mut targets);

    targets.sort();
    targets.dedup();
    targets
}

#[async_trait]
impl UnifiedQueryPort for UnifiedQueryPortImpl {
    async fn execute_unified_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        limit: Option<u32>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        if query.trim().is_empty() {
            return Err(anyhow!("query cannot be empty"));
        }
        // L4 (round 3): cap the RAW input before UQL lowering / binding.
        ensure_query_size(&query)?;
        debug!(
            "execute_unified_query: {}",
            query.chars().take(120).collect::<String>()
        );
        let parameters = parameters.unwrap_or_default();
        let (sql, origin) = match uql_to_federated_sql(&query, &parameters, limit)
            .with_context(|| format!("UQL lowering failed for query '{}'", query))?
        {
            // Port-lowered extension SQL is trusted to take the federated
            // branch; raw user text (parse failed / no UQL claim) goes to the
            // authority first (HIGH-1).
            Some(lowered) => (lowered, PortSqlOrigin::PortLowered),
            None => (
                bind_federated_sql_parameters(&query, &parameters)?,
                PortSqlOrigin::RawUserText,
            ),
        };
        run_sql_via_one_planner(
            &self.adapter,
            &sql,
            collection,
            context.identity,
            &context.freshness,
            limit,
            origin,
        )
        .await
    }

    async fn execute_multi_model_query(
        &self,
        request: serde_json::Value,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        // Convert the JSON multi-model request to a federated SQL string.
        // (Formerly a twin of the deleted root multimodal_query module's
        // convert_multi_model_to_sql; this is now the only copy.)
        // L4 (round 3): cap the RAW request before lowering it to SQL.
        ensure_query_size(&request.to_string())?;
        let (sql, origin) = multi_model_sql_from_request(&request)?;
        // chars().take — a byte-indexed slice can land mid-character in
        // the user-controlled cypher/collection text now spliced verbatim.
        info!(
            "execute_multi_model_query SQL: {}",
            sql.chars().take(200).collect::<String>()
        );
        run_sql_via_one_planner(
            &self.adapter,
            &sql,
            None,
            context.identity,
            &context.freshness,
            None,
            origin,
        )
        .await
    }

    async fn execute_federated_query(
        &self,
        query: String,
        parameters: Option<Vec<ProximaValue>>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        if query.trim().is_empty() {
            return Err(anyhow!("query cannot be empty"));
        }
        // L4 (round 3): cap the RAW input before parameter binding.
        ensure_query_size(&query)?;
        let parameters = parameters.unwrap_or_default();
        let query = bind_federated_sql_parameters(&query, &parameters)?;
        debug!(
            "execute_federated_query: {}",
            query.chars().take(120).collect::<String>()
        );
        // HIGH-1: even the federated surface sends RAW user text to the
        // authority first — the authority's fallback promotes GENUINE
        // extension SQL to the federated executor after the scheduler
        // declines, so behavior for real federated queries is unchanged,
        // while plain SQL carrying extension-looking literals/comments
        // executes as the plain SQL it is.
        run_sql_via_one_planner(
            &self.adapter,
            &query,
            None,
            context.identity,
            &context.freshness,
            None,
            PortSqlOrigin::RawUserText,
        )
        .await
    }

    async fn execute_distributed_query(
        &self,
        request: serde_json::Value,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        let query = request
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                anyhow::Error::new(InvalidQueryInput(
                    "distributed query request must have a 'query' field".to_string(),
                ))
            })?
            .to_string();
        ensure_query_size(&query)?;
        let limit = request
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|l| l as u32);
        debug!(
            "execute_distributed_query: {}",
            query.chars().take(120).collect::<String>()
        );
        // MED-2 (round 2): plain SQL on the distributed surface routes
        // through the SAME read-only seam as every other port entry. The old
        // path force-routed to the DistributedQueryCoordinator, which silently
        // returned EMPTY rows for plain SQL (no local resolution, no error).
        // The explicit `force_path = "distributed"` placement remains
        // reachable through `QueryFacadeAdapter::distributed_query` for
        // embedders; the port no longer invokes it (documented in the TD).
        run_sql_via_one_planner(
            &self.adapter,
            &query,
            None,
            context.identity,
            &context.freshness,
            limit,
            PortSqlOrigin::RawUserText,
        )
        .await
    }

    async fn explain_unified_query(
        &self,
        query: String,
        collection: Option<String>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        // LOW-1 (round-1 review): the context is USED — the explain runs in
        // the caller's identity scope so any tenant-scoped resolution inside
        // acts as the caller. (`adapter.explain` is a heuristic plan sketch,
        // not an execution — the freshness mode has nothing to gate and is
        // deliberately ignored; see the LOW-5 note on
        // `olap_cache_for_freshness`.)
        let scope_tenant = context.identity.tenant_id;
        crate::query::request_scope::with_identity_scope(scope_tenant, async {
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
        })
        .await
    }

    async fn prepare_statement(
        &self,
        _name: Option<String>,
        query: String,
        _cache_results: bool,
        ttl_seconds: Option<u64>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<String> {
        let owner = prepared_owner(context.identity.tenant_id).ok_or(MissingQueryIdentity(
            "prepare_statement has no request identity; the caller must act as a tenant",
        ))?;
        // L4 (round 3): the cap applies at prepare time too.
        ensure_query_size(&query)?;
        let ttl = Duration::from_secs(ttl_seconds.unwrap_or(3600));
        let cache_id = self
            .cache
            .prepare_with_ttl(&query, ttl)
            // L2 (round 2): a text the prepared-statement cache cannot parse
            // or optimize is a caller error (400), not a 500.
            .map_err(|e| InvalidQueryInput(format!("prepare_statement failed: {e}")))?;
        let statement_id = format!("stmt_{}", uuid::Uuid::new_v4().simple());
        self.prepared.insert(
            statement_id.clone(),
            PreparedHandle {
                owner,
                cache_id,
                executions: 0,
            },
        );
        Ok(statement_id)
    }

    async fn execute_prepared(
        &self,
        statement_id: String,
        parameters: Option<Vec<ProximaValue>>,
        collection: Option<String>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        let params = proxima_values_to_params(parameters);
        let cache_id = self
            .owned_cache_id(&statement_id, context.identity.tenant_id)
            .ok_or_else(|| anyhow!("prepared statement not found: {}", statement_id))?;
        let sql = self
            .cache
            .execute_sql(&cache_id, &params)
            .map_err(|e| match e {
                PreparedStatementError::NotFound(_) => {
                    anyhow!("prepared statement not found: {}", statement_id)
                }
                PreparedStatementError::Expired(_) => {
                    anyhow!("prepared statement expired: {}", statement_id)
                }
                other => anyhow!("prepared statement error: {}", other),
            })?;
        // The prepared text is user-authored: raw user text (HIGH-1).
        let result = run_sql_via_one_planner(
            &self.adapter,
            &sql,
            collection,
            context.identity,
            &context.freshness,
            None,
            PortSqlOrigin::RawUserText,
        )
        .await?;
        if let Some(mut handle) = self.prepared.get_mut(&statement_id) {
            handle.executions += 1;
        }
        Ok(result)
    }

    async fn delete_prepared(
        &self,
        statement_id: String,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<()> {
        let cache_id = self
            .owned_cache_id(&statement_id, context.identity.tenant_id)
            .ok_or_else(|| {
                anyhow!("delete_prepared failed: prepared statement not found: {statement_id}")
            })?;
        self.prepared.remove(&statement_id);
        // The shared cache may hold the same text for another handle; dropping
        // it there is safe only when no other handle refers to it.
        if !self
            .prepared
            .iter()
            .any(|entry| entry.value().cache_id == cache_id)
        {
            let _ = self.cache.drop_statement(&cache_id);
        }
        Ok(())
    }

    async fn get_prepared_stats(
        &self,
        _statement_ids: Vec<String>,
        context: UnifiedQueryPortContext<'_>,
    ) -> Result<serde_json::Value> {
        // Counts cover only the caller's own statements.
        let owner = prepared_owner(context.identity.tenant_id);
        let (statements, executions) = self
            .prepared
            .iter()
            .filter(|entry| Some(&entry.value().owner) == owner.as_ref())
            .fold((0u64, 0u64), |(n, e), entry| {
                (n + 1, e + entry.value().executions)
            });
        let stats = self.cache.stats();
        Ok(serde_json::json!({
            "cached_statements": statements,
            "max_statements": stats.max_statements,
            "total_executions": executions,
        }))
    }
}

/// The SQL to run and whether the PORT synthesized it (component lowering)
/// versus it being raw user text (the `query` passthrough) — HIGH-1.
fn multi_model_sql_from_request(request: &serde_json::Value) -> Result<(String, PortSqlOrigin)> {
    match json_to_multi_model_sql(request)
        .map_err(|error| InvalidQueryInput(error.to_string()))?
    {
        Some(sql) => Ok((sql, PortSqlOrigin::PortLowered)),
        // None unambiguously means 'not a multi-model request' (the
        // callee errs on every components-present shape — empty,
        // malformed, unknown — since round 41). The raw-query fallback
        // must still be explicit and typed; invalid JSON must never become
        // a successful broad `SELECT 1`.
        None => request
            .get("query")
            .and_then(serde_json::Value::as_str)
            .filter(|query| !query.trim().is_empty())
            .map(|query| (query.to_string(), PortSqlOrigin::RawUserText))
            .ok_or_else(|| {
                InvalidQueryInput(
                    "multi-model request without components must include a non-empty string 'query' field"
                        .to_string(),
                )
                .into()
            }),
    }
}

// ── Multi-model JSON → SQL conversion ────────────────────────────────────────

/// Convert a multi-model query JSON to a federated SQL string.
///
/// (The root canonical multimodal_query module this mirrored was a dead twin and has
/// been deleted; this is now the only copy.)
/// Returns `None` when no component request is present and fails closed when
/// a supplied component is malformed.
fn json_to_multi_model_sql(req: &serde_json::Value) -> Result<Option<String>> {
    let Some(components_value) = req.get("components") else {
        return Ok(None);
    };
    let components = components_value
        .as_array()
        .ok_or_else(|| anyhow!("components must be an array"))?;
    if components.is_empty() {
        // Err, not Ok(None): None must unambiguously mean 'not a
        // multi-model request' — the caller-side components-present
        // guard (raw-JSON re-inspection) was dropped once by a
        // restructure and returned SELECT-1 garbage.
        return Err(anyhow!(
            "multi-model request contained no components that could be lowered"
        ));
    }
    let mut parts = Vec::new();
    let empty_config = serde_json::Map::new();
    for (component_index, component) in components.iter().enumerate() {
        let ctype = component
            .get("component_type")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                anyhow!("components[{component_index}].component_type must be a string")
            })?;
        let config = match component.get("config") {
            Some(value) => value
                .as_object()
                .ok_or_else(|| anyhow!("components[{component_index}].config must be an object"))?,
            None => &empty_config,
        };
        let sql_part = match ctype {
            "vector" => {
                let collection = config
                    .get("collection")
                    .map(|v| {
                        v.as_str().ok_or_else(|| {
                            anyhow!(
                                "components[{component_index}].config.collection must be a string"
                            )
                        })
                    })
                    .transpose()?
                    .ok_or_else(|| {
                        anyhow!("components[{component_index}].config.collection is required")
                    })?;
                let query_values = config
                    .get("query_vector")
                    .ok_or_else(|| {
                        anyhow!("components[{component_index}].config.query_vector is required")
                    })?
                    .as_array()
                    .ok_or_else(|| {
                        anyhow!(
                            "components[{component_index}].config.query_vector must be an array"
                        )
                    })?;
                if query_values.is_empty() {
                    return Err(anyhow!(
                        "components[{component_index}].config.query_vector must be non-empty"
                    ));
                }
                // Validate then render through the ONE vector-text home
                // (f64 Display splices a different literal than the REST
                // twin for values outside exact f32 range).
                let f32_vec: Result<Vec<f32>> = query_values
                    .iter()
                    .enumerate()
                    .map(|(value_index, value)| {
                        let number = value.as_f64().ok_or_else(|| {
                            anyhow!(
                                "components[{component_index}].config.query_vector[{value_index}] must be numeric"
                            )
                        })?;
                        let f = crate::core::utils::finite_f32(number)
                        .ok_or_else(|| {
                            anyhow!(
                                "components[{component_index}].config.query_vector[{value_index}] must be a finite f32"
                            )
                        })?;
                        Ok(f)
                    })
                    .collect();
                let query_vec = vector_literal_text(&f32_vec?);
                // Typed: a string top_k silently baked the default.
                let top_k = config
                    .get("top_k")
                    .map(|v| {
                        v.as_u64().ok_or_else(|| {
                            anyhow!(
                                "components[{component_index}].config.top_k must be a non-negative integer"
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or(10);
                format!(
                    "SELECT * FROM VECTOR_SEARCH('{}', '{}', {})",
                    escape_sql_text(collection),
                    query_vec,
                    top_k
                )
            }
            "document" => {
                let collection = config
                    .get("collection")
                    .map(|v| {
                        v.as_str().ok_or_else(|| {
                            anyhow!(
                                "components[{component_index}].config.collection must be a string"
                            )
                        })
                    })
                    .transpose()?
                    .ok_or_else(|| {
                        anyhow!("components[{component_index}].config.collection is required")
                    })?;
                let filter = config
                    .get("filter")
                    .map(|v| {
                        v.as_str().ok_or_else(|| {
                            anyhow!("components[{component_index}].config.filter must be a string")
                        })
                    })
                    .transpose()?
                    .unwrap_or("1=1");
                format!(
                    "SELECT * FROM DOCUMENT_QUERY('{}', '{}')",
                    escape_sql_text(collection),
                    escape_sql_text(filter)
                )
            }
            "graph" => {
                // Typed (the silent-default class).
                let cypher = config
                    .get("cypher")
                    .map(|v| {
                        v.as_str().ok_or_else(|| {
                            anyhow!("components[{component_index}].config.cypher must be a string")
                        })
                    })
                    .transpose()?
                    .unwrap_or("MATCH (n) RETURN n LIMIT 10");
                // Honor config.graph like the v1 twin — without the
                // injection the query silently targets the DEFAULT graph.
                let graph = config
                    .get("graph")
                    .map(|v| {
                        v.as_str().ok_or_else(|| {
                            anyhow!("components[{component_index}].config.graph must be a string")
                        })
                    })
                    .transpose()?
                    .unwrap_or("default");
                let cypher = crate::core::utils::inject_graph_target_into_cypher(graph, cypher);
                format!("SELECT * FROM GRAPH_QUERY('{}')", escape_sql_text(&cypher))
            }
            // 'log'/'metric' are the v1 REST twin's component vocabulary
            // for the same observability arms — accepting them here stops
            // the silent `_ => continue` → 'SELECT 1' fallback returning a
            // 200-OK wrong result for v1-shaped requests.
            "observability" | "log" | "metric" => {
                let namespace = config
                    .get("namespace")
                    .map(|v| {
                        v.as_str().ok_or_else(|| {
                            anyhow!(
                                "components[{component_index}].config.namespace must be a string"
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or("default");
                let table = match ctype {
                    "metric" => "METRICS",
                    _ => "LOGS",
                };
                format!("SELECT * FROM {table}('{}')", escape_sql_text(namespace))
            }
            unknown => {
                return Err(anyhow!(
                    "components[{component_index}].component_type '{unknown}' is unsupported"
                ));
            }
        };
        parts.push(sql_part);
    }

    // parts cannot be empty: a present components array with zero
    // entries errs above, and every present component either errs or
    // pushes (the dead Ok(None) arm here was the ambiguous-None trap).

    // Single component: use directly; multiple: UNION ALL
    if parts.len() == 1 {
        Ok(Some(parts.remove(0)))
    } else {
        Ok(Some(parts.join(" UNION ALL ")))
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
    fn olap_cache_policy_strong_bypasses_the_result_cache() {
        // MED-2 (round-1 review): the gate is pinned IN THE DEFAULT ENV —
        // with an INJECTED enabled cache, Strong must STILL yield None
        // (removing the Strong arm makes this return Some and FAILS this
        // test), and a weaker mode hands the very same cache through.
        // No env flag involved: the policy is pure over its input.
        let cache =
            Arc::new(crate::query::cache::query_result_cache::QueryResultCache::with_defaults());
        assert!(
            olap_cache_for_freshness(Some(cache.clone()), &VectorFreshnessMode::Strong).is_none(),
            "Strong must bypass the result-cache lookup even when the cache is enabled"
        );
        assert!(
            olap_cache_for_freshness(Some(cache.clone()), &VectorFreshnessMode::StaleOk)
                .as_ref()
                .is_some_and(|handed| Arc::ptr_eq(handed, &cache)),
            "a weaker mode must hand the SAME cache through"
        );
    }

    #[test]
    fn plain_sql_read_classifier_accepts_reads_and_rejects_writes() {
        // MED-4: the port's plain-SQL surface is read-only. Reads pass…
        for read in [
            "SELECT id, score FROM orders WHERE id = 'a'",
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
            // A trailing comment does not change the statement class.
            "SELECT 1 -- uses <-> operator",
            // MED-1 (round 2): the PostgreSqlDialect tokenizes `<->`, so the
            // KNN WHERE-form parses (the GenericDialect rejected it).
            "SELECT id FROM c WHERE embedding <-> '[1.0, 0.0]' < 1.5",
            "SELECT * FROM c ORDER BY embedding <-> '[1.0, 0.0]' LIMIT 3",
            "SELECT * FROM c ORDER BY vector <-> '[1,0]'::vector LIMIT 2",
            // Positive controls for the nested-write walk: nested READS pass.
            "SELECT * FROM (SELECT id FROM t) AS d",
            "SELECT * FROM t WHERE id IN (SELECT id FROM u) UNION SELECT 'x'",
            "WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a) SELECT * FROM b",
        ] {
            ensure_plain_sql_is_read(read)
                .unwrap_or_else(|error| panic!("read must pass the classifier: {read} — {error}"));
        }
        // …and extension-looking LITERALS do not either (HIGH-1: classification
        // is syntactic, not substring).
        ensure_plain_sql_is_read(
            "SELECT id FROM t WHERE note = 'LOGS(app) literal' OR note LIKE '%VECTOR_SEARCH(%'",
        )
        .expect("literal-containing SELECT is a read");

        // Writes are rejected — including writes whose payloads contain
        // extension keywords (the HIGH-1 silent-write-loss probes).
        for write in [
            "CREATE TABLE t (id TEXT)",
            "INSERT INTO t (id) VALUES ('a')",
            "UPDATE t SET id = 'b'",
            "DELETE FROM t",
            "DROP TABLE t",
            "INSERT INTO probe VALUES ('p', 'logged VECTOR_SEARCH(v, [0.1], 10)')",
            "UPDATE t SET note = 'GRAPH_QUERY(MATCH (n) RETURN n)'",
            // Multi-statement text is not a single read.
            "SELECT 1; SELECT 2",
        ] {
            let error = ensure_plain_sql_is_read(write)
                .expect_err("a write must be rejected by the read-only classifier");
            assert!(
                error.to_string().contains("read-only"),
                "the rejection must name the policy: {write} — {error}"
            );
        }
    }

    /// HIGH-1 (round 2): every EXPLAIN form is rejected — the flag-based rule
    /// let ALL parenthesized ANALYZE options through (sqlparser sets
    /// `analyze` only for the bare keyword), executing the write.
    #[test]
    fn every_explain_form_is_rejected_by_the_read_only_classifier() {
        for explain in [
            "EXPLAIN SELECT 1 FROM t",
            "EXPLAIN ANALYZE INSERT INTO t (id) VALUES ('a')",
            // The round-2 bypass shapes: parenthesized option lists.
            "EXPLAIN (ANALYZE) INSERT INTO t (id) VALUES ('a')",
            "EXPLAIN (FORMAT JSON, ANALYZE) INSERT INTO t (id) VALUES ('a')",
            "EXPLAIN (ANALYZE, VERBOSE) DELETE FROM t",
            "EXPLAIN (ANALYZE off) DELETE FROM t",
            "EXPLAIN (analyze) UPDATE t SET id = 'b'",
            "EXPLAIN (x ANALYZE) DELETE FROM t",
            "EXPLAIN (ANALYZE true) DELETE FROM t",
        ] {
            let error = ensure_plain_sql_is_read(explain)
                .expect_err(&format!("every EXPLAIN form must be rejected: {explain}"));
            assert!(
                error.to_string().contains("EXPLAIN"),
                "the rejection must name EXPLAIN: {explain} — {error}"
            );
        }
        // `EXPLAIN DESCRIBE <DML>` does not parse under the PostgreSqlDialect
        // ("Explain must be root of the plan") — a fail-closed parse
        // rejection, which is equally a non-read.
        assert!(
            ensure_plain_sql_is_read("EXPLAIN DESCRIBE INSERT INTO t (id) VALUES ('a')").is_err()
        );
        // `DESCRIBE <DML>` (an EXPLAIN alias) is rejected too.
        assert!(ensure_plain_sql_is_read("DESCRIBE INSERT INTO t (id) VALUES ('a')").is_err());
        assert!(ensure_plain_sql_is_read("DESCRIBE DELETE FROM t").is_err());
    }

    /// HIGH-1 (round 2): a Query is a pure read only when NOTHING in it can
    /// execute a write — data-modifying set-expr bodies (parenthesized or in
    /// set operations), data-modifying CTEs, and SELECT … INTO all reject.
    #[test]
    fn query_shaped_writes_are_rejected_by_the_read_only_classifier() {
        for write_shaped in [
            // A set-expression body that IS a write.
            "(DELETE FROM t RETURNING *)",
            "(INSERT INTO t (id) VALUES ('a') RETURNING *)",
            // A write in a set-operation position.
            "SELECT 1 UNION (DELETE FROM t RETURNING *)",
            "SELECT 1 UNION ALL (UPDATE t SET id = 'b' RETURNING *)",
            // A data-modifying CTE.
            "WITH del AS (DELETE FROM t RETURNING *) SELECT * FROM del",
            "WITH ins AS (INSERT INTO t (id) VALUES ('a') RETURNING *) SELECT * FROM ins",
            "WITH x AS (SELECT 1) DELETE FROM t",
            "(WITH x AS (SELECT 1) DELETE FROM t)",
            // SELECT … INTO creates a table.
            "SELECT id INTO new_table FROM t",
            // Writes nested below the top level: a derived table in FROM, an
            // expression subquery, and a set-operation operand inside a CTE.
            "SELECT * FROM (DELETE FROM t RETURNING *) AS d",
            "SELECT * FROM t WHERE id IN (SELECT id FROM (DELETE FROM t RETURNING id) AS d)",
            "WITH x AS (SELECT 1 UNION (UPDATE t SET id = 'b' RETURNING 1)) SELECT * FROM x",
            "SELECT * FROM (SELECT id INTO t2 FROM t) AS d",
        ] {
            let error = ensure_plain_sql_is_read(write_shaped).expect_err(&format!(
                "a query-shaped write must be rejected: {write_shaped}"
            ));
            assert!(
                error.to_string().contains("read-only"),
                "the rejection must name the policy: {write_shaped} — {error}"
            );
        }
    }

    /// HIGH-2 (round 2): `parse_explain_kind` parses the option list —
    /// `(ANALYZE off)` is NOT analyze (PostgreSQL semantics; the old
    /// substring check executed a write the user asked to only plan).
    #[test]
    fn parse_explain_kind_parses_the_option_list() {
        fn parse_explain_kind(query: &str) -> Option<(bool, &str)> {
            crate::query::sql_frontend::parse_explain_kind(query)
                .unwrap_or_else(|error| panic!("{query}: {error}"))
        }
        assert_eq!(
            parse_explain_kind("EXPLAIN (ANALYZE) DELETE FROM t"),
            Some((true, "DELETE FROM t"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN (FORMAT JSON, ANALYZE) DELETE FROM t"),
            Some((true, "DELETE FROM t"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN (analyze, verbose) DELETE FROM t"),
            Some((true, "DELETE FROM t"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN (ANALYZE off) DELETE FROM t"),
            Some((false, "DELETE FROM t"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN (ANALYZE false) DELETE FROM t"),
            Some((false, "DELETE FROM t"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN (ANALYZE 0) DELETE FROM t"),
            Some((false, "DELETE FROM t"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN (VERBOSE) SELECT 1"),
            Some((false, "SELECT 1"))
        );
        assert_eq!(
            parse_explain_kind("EXPLAIN ANALYZE SELECT 1"),
            Some((true, "SELECT 1"))
        );
        assert_eq!(parse_explain_kind("SELECT 1"), None);
    }

    #[test]
    fn effective_identity_prefers_explicit_then_derives_scope() {
        // MED-3: explicit identity tenant wins…
        let explicit = effective_identity_or_fail(&PortIdentity::for_tenant("acme"))
            .expect("explicit identity resolves");
        assert_eq!(explicit.tenant_id.as_deref(), Some("acme"));

        // …a Tenant scope is DERIVED into an anonymous identity (asserted in
        // the async test below — task-locals need a lazy async block)…

        // …and neither identity nor scope fails closed with the TYPED error.
        let error = effective_identity_or_fail(&PortIdentity::anonymous())
            .expect_err("anonymous + scope-less must fail closed");
        assert!(
            error
                .downcast_ref::<proximadb_runtime::MissingQueryIdentity>()
                .is_some(),
            "the gate must be typed MissingQueryIdentity: {error:#}"
        );
    }

    #[tokio::test]
    async fn effective_identity_derives_the_scope_tenant() {
        // (The async block is REQUIRED — `std::future::ready(expr)` evaluates
        // `expr` eagerly, OUTSIDE the task-local scope.)
        let derived = crate::query::request_scope::with_request_tenant("scoped", async {
            effective_identity_or_fail(&PortIdentity::anonymous())
        })
        .await
        .expect("scope-derived identity resolves");
        assert_eq!(derived.tenant_id.as_deref(), Some("scoped"));

        // An explicit identity inside a DIFFERENT scope still wins.
        let explicit_wins = crate::query::request_scope::with_request_tenant("outer", async {
            effective_identity_or_fail(&PortIdentity::for_tenant("acme"))
        })
        .await
        .expect("explicit identity resolves");
        assert_eq!(explicit_wins.tenant_id.as_deref(), Some("acme"));
    }

    #[test]
    fn sql_result_renders_the_port_envelope() {
        use proximadb_data_model::ProximaValue as V;
        let result = proximadb_runtime::SqlExecutionResult {
            columns: vec!["id".to_string(), "score".to_string()],
            column_types: vec!["TEXT".to_string(), "FLOAT".to_string()],
            rows: vec![
                vec![V::String("a".into()), V::Float64(1.5)],
                vec![V::String("b".into()), V::Float64(2.5)],
            ],
            rows_affected: None,
            rows_scanned: 2,
            execution_time_ms: 3,
        };
        let json = sql_result_to_port_json(result, None).expect("render");
        let records = json["records"].as_array().expect("records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["id"], serde_json::json!("a"));
        assert_eq!(records[0]["score"], serde_json::json!(1.5));
        assert_eq!(json["total_count"], serde_json::json!(2));

        // The read limit truncates both records and the post-limit count.
        let json = sql_result_to_port_json(
            proximadb_runtime::SqlExecutionResult {
                columns: vec!["id".to_string()],
                column_types: vec!["TEXT".to_string()],
                rows: vec![vec![V::String("a".into())], vec![V::String("b".into())]],
                rows_affected: None,
                rows_scanned: 2,
                execution_time_ms: 0,
            },
            Some(1),
        )
        .expect("render");
        assert_eq!(json["records"].as_array().expect("records").len(), 1);
        assert_eq!(json["total_count"], serde_json::json!(1));

        // A write renders an empty records list with the affected count.
        let json = sql_result_to_port_json(
            proximadb_runtime::SqlExecutionResult {
                columns: vec![],
                column_types: vec![],
                rows: vec![],
                rows_affected: Some(7),
                rows_scanned: 7,
                execution_time_ms: 1,
            },
            None,
        )
        .expect("render");
        assert_eq!(json["records"].as_array().expect("records").len(), 0);
        assert_eq!(json["total_count"], serde_json::json!(7));
    }

    #[tokio::test]
    async fn prepared_owner_prefers_the_explicit_identity_tenant() {
        // Explicit identity tenant wins (ADR-098 D5: identity as data), even
        // over a task-local scope in force around the call. (The async block
        // is REQUIRED: `std::future::ready(expr)` evaluates `expr` eagerly,
        // OUTSIDE the task-local scope.)
        assert_eq!(prepared_owner(Some("acme")), Some("acme".to_string()));
        let scoped = crate::query::request_scope::with_request_tenant(
            "outer",
            crate::query::request_scope::with_request_tenant("inner", async {
                prepared_owner(Some("acme"))
            }),
        )
        .await;
        assert_eq!(scoped, Some("acme".to_string()));

        // Fallback: no identity → the scope in force (task-local).
        let fallback = crate::query::request_scope::with_request_tenant("scoped", async {
            prepared_owner(None)
        })
        .await;
        assert_eq!(fallback, Some("scoped".to_string()));
        // ...and no scope at all → None (no scope owns nothing).
        assert_eq!(prepared_owner(None), None);
    }

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
        // Structured values flow through the catch-all as their JSON-text
        // String param (the deleted per-variant Json arms re-derived it).
        let value = ProximaValue::Array(vec![ProximaValue::Int64(1), ProximaValue::Int64(2)]);
        assert!(matches!(
            proxima_value_to_param(&value),
            ParameterValue::String(s) if s == "[1,2]"
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
    fn uql_vector_select_rejects_non_finite_components() {
        let error = uql_to_federated_sql(
            "SELECT * FROM vectors.products WHERE VECTOR_SIMILAR(embedding, ?, 0.8)",
            &[ProximaValue::DenseVector(vec![f32::NAN])],
            None,
        )
        .expect_err("non-finite UQL vector must fail closed");

        assert!(error.to_string().contains("finite"));
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
    fn plain_relational_select_is_not_a_uql_claim() {
        // TD-SPINE-6: the UQL grammar overlaps plain SQL (an un-prefixed FROM
        // defaults to the Vector model). A SELECT with no vector condition
        // must claim NOTHING (Ok(None)) so the text routes as plain SQL
        // through the one planner seam — the pre-convergence behavior was a
        // hard "UQL vector queries require VECTOR_SIMILAR" error.
        for plain in [
            "SELECT id, score FROM orders ORDER BY id",
            "SELECT * FROM orders WHERE id = 'a' LIMIT 5",
            "SELECT * FROM vectors.products",
        ] {
            assert!(
                uql_to_federated_sql(plain, &[], None)
                    .expect("plain relational SELECT must not error at UQL lowering")
                    .is_none(),
                "plain SELECT must not be claimed as UQL: {plain}"
            );
        }

        // A REAL vector claim still lowers (and still validates params).
        let claimed = uql_to_federated_sql(
            "SELECT * FROM vectors.p WHERE VECTOR_SIMILAR(embedding, ?, 0.8) LIMIT 7",
            &[ProximaValue::DenseVector(vec![0.1, 0.2])],
            None,
        )
        .expect("vector UQL must lower")
        .expect("vector UQL must be claimed");
        assert!(claimed.contains("VECTOR_SEARCH('p'"));
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
    fn bind_federated_sql_parameters_rejects_non_finite_vectors() {
        let error = bind_federated_sql_parameters(
            "SELECT * FROM VECTOR_SEARCH('products', ?, 5)",
            &[ProximaValue::DenseVector(vec![f32::NAN])],
        )
        .expect_err("non-finite vectors must fail closed");

        assert!(error.to_string().contains("finite"));
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
    fn explain_skips_function_call_positions() {
        // Paren-bearing tokens are function-call positions, not catalog
        // targets — and a subquery's closing paren must NOT lose its
        // target.
        for query in [
            "SELECT * FROM TRACES('ops') WHERE EXISTS (SELECT 1 FROM orders)",
            "SELECT * FROM TRACES ('ops') WHERE EXISTS (SELECT 1 FROM orders)",
            "SELECT * FROM TRACES ( 'ops') WHERE EXISTS (SELECT 1 FROM orders)",
        ] {
            let targets = explain_catalog_targets(query);
            assert!(!targets.iter().any(|t| t == "TRACES"), "got {targets:?}");
            assert!(targets.iter().any(|t| t == "orders"), "got {targets:?}");
            assert!(targets.iter().any(|t| t == "ops"), "got {targets:?}");
        }

        let targets = explain_catalog_targets(
            r#"INSERT INTO orders(id) VALUES (1); INSERT INTO "events(2026)"(id) VALUES (1); SELECT 1 FROM (SELECT * FROM "archive")"#,
        );
        assert!(targets.iter().any(|t| t == "orders"), "got {targets:?}");
        assert!(
            targets.iter().any(|t| t == "events(2026)"),
            "got {targets:?}"
        );
        assert!(targets.iter().any(|t| t == "archive"), "got {targets:?}");
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
        let sql = json_to_multi_model_sql(&req).unwrap().unwrap();
        // STRICT vector pin: the splice is QUOTED (the bare '[...]'
        // spelling was the round-15/20 churn — substring pins passed
        // under both).
        assert_eq!(
            sql,
            "SELECT * FROM VECTOR_SEARCH('embeddings', '[0.1,0.2,0.3]', 5)"
        );
    }

    #[test]
    fn json_to_multi_model_sql_escapes_every_text_argument() {
        let req = serde_json::json!({
            "components": [
                {
                    "component_type": "vector",
                    "config": {"collection": "team's-vectors", "query_vector": [0.1]}
                },
                {
                    "component_type": "document",
                    "config": {"collection": "team's-docs", "filter": "owner = \"O'Brien\""}
                },
                {
                    "component_type": "graph",
                    "config": {"cypher": "MATCH (n) RETURN 'label'"}
                },
                {
                    "component_type": "observability",
                    "config": {"namespace": "team's-production"}
                }
            ]
        });

        let sql = json_to_multi_model_sql(&req).unwrap().unwrap();
        assert!(sql.contains("VECTOR_SEARCH('team''s-vectors'"));
        assert!(sql.contains("DOCUMENT_QUERY('team''s-docs', 'owner = \"O''Brien\"')"));
        assert!(sql.contains("GRAPH_QUERY('MATCH (n) RETURN ''label''')"));
        assert!(sql.contains("LOGS('team''s-production')"));
    }

    #[test]
    fn json_to_multi_model_sql_rejects_non_numeric_vector_elements() {
        let req = serde_json::json!({
            "components": [{
                "component_type": "vector",
                "config": {"collection": "vectors", "query_vector": [0.1, "bad", 0.3]}
            }]
        });

        let error = json_to_multi_model_sql(&req).expect_err("invalid vector must fail closed");
        assert!(error.to_string().contains("query_vector[1]"));
    }

    #[test]
    fn json_to_multi_model_sql_requires_document_collection() {
        let req = serde_json::json!({
            "components": [{
                "component_type": "document",
                "config": {"filter": "active = true"}
            }]
        });
        let error = json_to_multi_model_sql(&req).expect_err("collection must be explicit");
        assert!(
            error
                .to_string()
                .contains("components[0].config.collection is required")
        );
    }

    #[test]
    fn json_to_multi_model_sql_uses_executable_document_match_all() {
        let req = serde_json::json!({
            "components": [{
                "component_type": "document",
                "config": {"collection": "docs"}
            }]
        });
        let sql = json_to_multi_model_sql(&req)
            .expect("conversion should succeed")
            .expect("components produce SQL");
        assert!(sql.contains("DOCUMENT_QUERY('docs', '1=1')"));
    }

    #[test]
    fn json_to_multi_model_sql_rejects_values_outside_f32_range() {
        let req = serde_json::json!({
            "components": [{
                "component_type": "vector",
                "config": {"collection": "vectors", "query_vector": [1e300]}
            }]
        });

        let error = json_to_multi_model_sql(&req).expect_err("infinite f32 must fail closed");
        assert!(error.to_string().contains("finite f32"));
    }

    #[test]
    fn test_json_to_multi_model_sql_empty_components() {
        // Err (not Ok(None)) since round 41 — None must mean 'not a
        // multi-model request' only.
        let req = serde_json::json!({"components": []});
        let err = json_to_multi_model_sql(&req).unwrap_err();
        assert!(err.to_string().contains("no components"));
    }

    #[test]
    fn json_to_multi_model_sql_rejects_non_object_config() {
        for component_type in ["graph", "log", "metric"] {
            for config in [
                serde_json::json!([]),
                serde_json::json!(42),
                serde_json::Value::Null,
            ] {
                let req = serde_json::json!({
                    "components": [{"component_type": component_type, "config": config}]
                });
                let error = json_to_multi_model_sql(&req)
                    .expect_err("a supplied component config must be an object");
                assert!(error.to_string().contains("config must be an object"));
            }
        }
    }

    #[test]
    fn test_json_to_multi_model_sql_no_components_field() {
        let req = serde_json::json!({ "query": "SELECT 1" });
        assert!(json_to_multi_model_sql(&req).unwrap().is_none());
        let (sql, origin) = multi_model_sql_from_request(&req).unwrap();
        assert_eq!(sql, "SELECT 1");
        assert_eq!(origin, PortSqlOrigin::RawUserText);
    }

    #[test]
    fn multi_model_sql_requires_typed_non_empty_raw_query() {
        for req in [
            serde_json::json!({}),
            serde_json::json!({"query": 42}),
            serde_json::json!({"query": null}),
            serde_json::json!({"query": "  "}),
        ] {
            let error = multi_model_sql_from_request(&req)
                .expect_err("missing or malformed raw query must fail closed");
            assert!(error.downcast_ref::<InvalidQueryInput>().is_some());
        }
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

        let quoted = explain_catalog_targets(
            r#"SELECT * FROM "team""logs"; SELECT * FROM "tenant,west"; SELECT * FROM "archive data"; SELECT * FROM "events(2026)"; SELECT * FROM "schema"."table""#,
        );
        assert!(quoted.contains(&"team\"logs".to_string()), "got {quoted:?}");
        assert!(
            quoted.contains(&"tenant,west".to_string()),
            "got {quoted:?}"
        );
        assert!(
            quoted.contains(&"archive data".to_string()),
            "got {quoted:?}"
        );
        assert!(
            quoted.contains(&"events(2026)".to_string()),
            "got {quoted:?}"
        );
        assert!(
            quoted.contains(&"schema.table".to_string()),
            "got {quoted:?}"
        );

        let quoted_keyword =
            explain_catalog_targets(r#"SELECT 1 AS "FROM" FROM actual; SELECT * FROM "SELECT""#);
        assert!(quoted_keyword.contains(&"actual".to_string()));
        assert!(quoted_keyword.contains(&"SELECT".to_string()));

        let quoted_function_arg = explain_catalog_targets("SELECT * FROM TRACES('SELECT')");
        assert_eq!(quoted_function_arg, ["SELECT"]);

        let adjacent = explain_catalog_targets(
            r#"SELECT*FROM "orders"; INSERT INTO"events(2026)"(id) VALUES (1)"#,
        );
        assert!(adjacent.contains(&"orders".to_string()), "got {adjacent:?}");
        assert!(
            adjacent.contains(&"events(2026)".to_string()),
            "got {adjacent:?}"
        );
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
