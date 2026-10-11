//! SQL → LogicalNode frontend for ProximaDB's relational query path
//! (ADR-019 L1 / S5a). Parses SQL with sqlparser, walks the AST,
//! and emits a [`LogicalNode`] from `proximadb-relational-algebra`.
//! The planner and executor crates take it from there.
//!
//! Scope (MVP):
//!
//! - `SELECT [DISTINCT] <projection> FROM <table>|<join> [WHERE …]
//!   [GROUP BY …] [HAVING …] [ORDER BY …] [LIMIT n OFFSET m]`
//! - `INNER JOIN` and `LEFT JOIN`
//! - `UNION [ALL]` set operations
//! - Top-level `VALUES`
//! - Expressions: Identifier / CompoundIdentifier (table.col),
//!   numeric / string / boolean / NULL literals, BinaryOp,
//!   UnaryOp, IsNull / IsNotNull, Between, InList, Like / ILike,
//!   Aggregate calls (COUNT/SUM/AVG/MIN/MAX), wildcard `*`.
//!
//! Out of scope for this slice — explicitly returned as
//! [`FrontendError::Unsupported`]:
//!
//! - Subqueries (need `AlgebraExpr::Sub` lowering).
//! - CTEs (parsed but rejected — Phase 3).
//! - Window functions, CASE/COALESCE/NULLIF expressions.
//! - DML (INSERT/UPDATE/DELETE) — separate write-path slice.
//! - DDL (CREATE TABLE / ALTER TABLE) — separate.
//!
//! Schema resolution: callers provide a [`CatalogLookup`] so we
//! don't depend on the runtime catalog crate from here.

use proximadb_data_model::{ProximaType, ProximaValue, TimeUnit};
use proximadb_relational_algebra::{
    AggregateExpr, JoinKind, JoinStrategy, LogicalNode, NamedAggregate, NamedExpr, SetOpKind,
    SortKey, TableId,
};
use proximadb_relational_types::{
    BinaryOp, ColumnInfo, ColumnRef, Expr, RelationalSchema, UnaryOp,
};
use sqlparser::ast::{
    BinaryOperator, DataType as SqlDataType, Distinct, Expr as SqlExpr, FunctionArg,
    FunctionArgExpr, FunctionArguments, GroupByExpr, JoinConstraint, JoinOperator, LimitClause,
    ObjectName, OrderByKind, Query as SqlQuery, Select as SqlSelect, SelectItem, SetExpr,
    SetOperator, SetQuantifier, Statement, TableFactor, UnaryOperator, Value as SqlValue,
    ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use thiserror::Error;

// =========================================================================
// Errors
// =========================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum FrontendError {
    #[error("SQL parse error: {0}")]
    Parse(String),

    #[error("table not found: {0}")]
    TableNotFound(String),

    #[error("column not found: {0}")]
    ColumnNotFound(String),

    #[error("ambiguous column reference: {0}")]
    AmbiguousColumn(String),

    #[error("unsupported SQL feature: {0}")]
    Unsupported(String),

    #[error("invalid literal: {0}")]
    InvalidLiteral(String),

    #[error("expected exactly one statement, got {0}")]
    StatementCount(usize),

    #[error("type error: {0}")]
    Type(String),
}

/// Machine-readable reason a query shape could not be lowered to a native
/// `LogicalNode`. `node` is the declining construct class (a coarse label used
/// to group declines in the ADR-064 query trace); `why` is the specific,
/// human-readable limitation (the error's `Display`).
///
/// Produced by [`FrontendError::decline`]. Structuring the existing error
/// message (rather than surfacing a flat string, or dropping it entirely) lets
/// `EXPLAIN` disclose the *real* reason a query declined instead of the pgwire
/// path re-guessing it from the raw SQL text (ADR-064 Decision 3 / TD-TRACE-1).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LoweringDecline {
    /// Coarse construct class of the decline (`subquery`, `having`, `join`,
    /// `cte`, `group_by`, `distinct`, `window`, `table`, `column`, `literal`,
    /// `type`, `statement`, `parse`, or `unsupported`).
    pub node: String,
    /// The specific limitation — the error's `Display` string.
    pub why: String,
}

impl FrontendError {
    /// Structured `{node, why}` view of this error for the ADR-064 query trace /
    /// `EXPLAIN` surface. `why` is the `Display` string this error already
    /// carries; `node` is a coarse construct class (see
    /// [`classify_unsupported`] for the `Unsupported` case). Cheap: one small
    /// static label plus the already-owned message.
    pub fn decline(&self) -> LoweringDecline {
        let node = match self {
            FrontendError::Parse(_) => "parse",
            FrontendError::TableNotFound(_) => "table",
            FrontendError::ColumnNotFound(_) | FrontendError::AmbiguousColumn(_) => "column",
            FrontendError::Unsupported(msg) => classify_unsupported(msg),
            FrontendError::InvalidLiteral(_) => "literal",
            FrontendError::StatementCount(_) => "statement",
            FrontendError::Type(_) => "type",
        };
        LoweringDecline {
            node: node.to_string(),
            why: self.to_string(),
        }
    }
}

/// Classify an `Unsupported` feature message into a coarse construct class for
/// the query trace. Substring match over the stable message literals this crate
/// emits (e.g. `"correlated scalar subquery"`, `"HAVING with aggregate calls"`,
/// `"EXISTS / IN subquery in expression position"`). Best-effort labelling —
/// the authoritative reason is always the full `why` string, never the label.
fn classify_unsupported(msg: &str) -> &'static str {
    if msg.contains("subquery") || msg.contains("EXISTS") || msg.contains("IN subquery") {
        "subquery"
    } else if msg.contains("HAVING") {
        "having"
    } else if msg.contains("JOIN") || msg.contains("join") {
        "join"
    } else if msg.contains("CTE") || msg.contains("WITH") {
        "cte"
    } else if msg.contains("GROUP BY") {
        "group_by"
    } else if msg.contains("DISTINCT") {
        "distinct"
    } else if msg.contains("window") || msg.contains("WINDOW") {
        "window"
    } else {
        "unsupported"
    }
}

// =========================================================================
// Catalog lookup trait
// =========================================================================

/// Resolve a table name to its schema. The frontend stays pure
/// — no catalog crate dependency — by taking this trait object.
///
/// Returns a shared `Arc` so hot implementors (per-connection snapshot
/// catalogs) can serve lookups by refcount instead of deep-cloning the
/// schema's column list per call.
pub trait CatalogLookup {
    fn lookup_table(&self, name: &str) -> Option<std::sync::Arc<RelationalSchema>>;
}

/// Simple in-memory catalog for tests and standalone use.
#[derive(Debug, Default, Clone)]
pub struct InMemoryCatalog {
    tables: std::collections::HashMap<String, std::sync::Arc<RelationalSchema>>,
}

impl InMemoryCatalog {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, name: impl Into<String>, schema: RelationalSchema) {
        self.tables.insert(name.into(), std::sync::Arc::new(schema));
    }
}

impl CatalogLookup for InMemoryCatalog {
    fn lookup_table(&self, name: &str) -> Option<std::sync::Arc<RelationalSchema>> {
        if let Some(s) = self.tables.get(name) {
            return Some(std::sync::Arc::clone(s));
        }
        // TD-OLAP-18: quoted spellings arrive as `"Name"` (ObjectName::to_string
        // keeps the quotes) — they match the declared name case-exactly only.
        let trimmed = name.trim_matches('"');
        if name != trimmed {
            return self.tables.get(trimmed).cloned();
        }
        // Unquoted spellings fold: a unique case-insensitive match resolves.
        if !ident_case_fold_enabled() {
            return None;
        }
        let mut hit: Option<&std::sync::Arc<RelationalSchema>> = None;
        for (k, v) in &self.tables {
            if k.eq_ignore_ascii_case(trimmed) {
                if hit.is_some() {
                    return None; // ambiguous across case-variants — stay strict
                }
                hit = Some(v);
            }
        }
        hit.cloned()
    }
}

/// TD-OLAP-18: ANSI/PostgreSQL identifier case-folding. Unquoted identifiers
/// resolve case-insensitively (an exact/declared-case match always wins first);
/// quoted identifiers stay case-exact. The fold only turns previously-failing
/// lookups into successes, so it defaults ON; `PROXIMADB_IDENT_CASE_FOLD=0`
/// restores the legacy case-exact resolution. Consulted only on the exact-miss
/// cold path, so it costs nothing on today's working queries.
pub fn ident_case_fold_enabled() -> bool {
    match std::env::var("PROXIMADB_IDENT_CASE_FOLD") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

// =========================================================================
// Entry point
// =========================================================================

/// Parse one SQL statement and lower it to a [`LogicalNode`].
/// Errors if the input has zero or multiple statements.
pub fn lower_sql(sql: &str, catalog: &dyn CatalogLookup) -> Result<LogicalNode, FrontendError> {
    let statements = Parser::parse_sql(&GenericDialect {}, sql)
        .map_err(|e| FrontendError::Parse(e.to_string()))?;
    if statements.len() != 1 {
        return Err(FrontendError::StatementCount(statements.len()));
    }
    lower_statement(&statements[0], catalog)
}

fn lower_statement(
    stmt: &Statement,
    catalog: &dyn CatalogLookup,
) -> Result<LogicalNode, FrontendError> {
    match stmt {
        Statement::Query(q) => lower_query(q, catalog),
        Statement::Insert(_) | Statement::Update { .. } | Statement::Delete(_) => Err(
            FrontendError::Unsupported("DML (INSERT/UPDATE/DELETE)".into()),
        ),
        other => Err(FrontendError::Unsupported(format!(
            "statement kind {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

// =========================================================================
// Query (top-level): SELECT / VALUES / UNION
// =========================================================================

fn lower_query(
    query: &SqlQuery,
    catalog: &dyn CatalogLookup,
) -> Result<LogicalNode, FrontendError> {
    if query.with.is_some() {
        return Err(FrontendError::Unsupported("WITH (CTE)".into()));
    }
    let mut body = lower_set_expr(&query.body, catalog)?;
    // ORDER BY (only at top of the query)
    if let Some(order_by) = &query.order_by {
        body = apply_order_by(body, order_by, catalog)?;
    }
    // LIMIT / OFFSET
    if let Some(limit_clause) = &query.limit_clause {
        body = apply_limit(body, limit_clause)?;
    }
    Ok(body)
}

fn lower_set_expr(
    set: &SetExpr,
    catalog: &dyn CatalogLookup,
) -> Result<LogicalNode, FrontendError> {
    match set {
        SetExpr::Select(s) => lower_select(s, catalog),
        SetExpr::Query(q) => lower_query(q, catalog),
        SetExpr::Values(v) => lower_values(v),
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            let l = lower_set_expr(left, catalog)?;
            let r = lower_set_expr(right, catalog)?;
            let all = matches!(set_quantifier, SetQuantifier::All);
            match op {
                SetOperator::Union => Ok(LogicalNode::Union {
                    inputs: vec![l, r],
                    all,
                }),
                SetOperator::Intersect => Ok(LogicalNode::SetOp {
                    op: SetOpKind::Intersect,
                    left: Box::new(l),
                    right: Box::new(r),
                    all,
                }),
                SetOperator::Except => Ok(LogicalNode::SetOp {
                    op: SetOpKind::Except,
                    left: Box::new(l),
                    right: Box::new(r),
                    all,
                }),
                other => Err(FrontendError::Unsupported(format!(
                    "set operator {:?}",
                    other
                ))),
            }
        }
        other => Err(FrontendError::Unsupported(format!(
            "set-expr {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

fn lower_values(values: &sqlparser::ast::Values) -> Result<LogicalNode, FrontendError> {
    let mut rows: Vec<Vec<Expr>> = Vec::with_capacity(values.rows.len());
    // VALUES rows live in an empty-schema scope; expressions must
    // be self-contained (literal-only at MVP).
    let empty_scope = Scope::empty();
    let mut col_types: Vec<ProximaType> = Vec::new();
    for (i, row) in values.rows.iter().enumerate() {
        let mut lowered = Vec::with_capacity(row.len());
        for (j, e) in row.iter().enumerate() {
            let expr = lower_expr_sealed(e, &empty_scope)?;
            if i == 0 {
                col_types.push(expr.result_type());
            }
            lowered.push(expr);
            let _ = j;
        }
        rows.push(lowered);
    }
    let cols: Vec<ColumnInfo> = col_types
        .into_iter()
        .enumerate()
        .map(|(i, ty)| ColumnInfo {
            name: format!("column{}", i + 1),
            ty,
            nullable: true,
        })
        .collect();
    Ok(LogicalNode::Values {
        rows,
        output_schema: RelationalSchema::new(cols),
    })
}

// =========================================================================
// SELECT
// =========================================================================

fn lower_select(
    select: &SqlSelect,
    catalog: &dyn CatalogLookup,
) -> Result<LogicalNode, FrontendError> {
    // 1) FROM (single, with explicit joins, or comma-separated). Empty FROM is
    //    unsupported.
    if select.from.is_empty() {
        return Err(FrontendError::Unsupported("SELECT without FROM".into()));
    }
    let (mut plan, mut scope) = lower_from(&select.from, select.selection.as_ref(), catalog)?;

    // 2) WHERE — lift uncorrelated IN / EXISTS / NOT EXISTS subqueries that appear
    //    as top-level AND-conjuncts into Semi/Anti joins; the remaining conjuncts
    //    become a Filter. A subquery that can't be lifted (correlated, NOT IN,
    //    multi-column, or nested under OR/NOT) stays in the filter predicate, where
    //    `lower_expr` rejects it — so the whole query falls through to the legacy
    //    path rather than silently mis-evaluating.
    if let Some(where_expr) = &select.selection {
        let mut filter_conjuncts: Vec<&SqlExpr> = Vec::new();
        for conj in flatten_sql_and(where_expr) {
            match lower_subquery_join_parts(conj, &scope, catalog)? {
                Some((kind, on, right)) => {
                    // Semi/Anti emit the LEFT schema only, so `scope` is unchanged.
                    plan = LogicalNode::Join {
                        left: Box::new(plan),
                        right: Box::new(right),
                        kind,
                        on,
                        strategy: JoinStrategy::Auto,
                    };
                }
                None => filter_conjuncts.push(conj),
            }
        }
        if let Some((first, rest)) = filter_conjuncts.split_first() {
            // Stage 2: a scalar subquery in a WHERE comparison (`a > (SELECT …)`)
            // is hoisted into a `LEFT JOIN ON TRUE` appended BELOW the Filter, so
            // the predicate references the post-join column. `base_width` is the
            // current plan width (Semi/Anti lifts above emit left-only, so it
            // equals the base relation width).
            let base_width = plan.output_schema().columns.len();
            let mut ctx = LoweringCtx::new(catalog, base_width);
            let mut predicate = lower_expr(first, &scope, &mut ctx)?;
            for conj in rest {
                predicate = Expr::bin(
                    BinaryOp::And,
                    predicate,
                    lower_expr(conj, &scope, &mut ctx)?,
                );
            }
            for hoisted in ctx.hoisted.drain(..) {
                plan = LogicalNode::Join {
                    left: Box::new(plan),
                    right: Box::new(hoisted.plan),
                    kind: JoinKind::Left,
                    on: hoisted.on,
                    strategy: JoinStrategy::Auto,
                };
            }
            plan = LogicalNode::Filter {
                input: Box::new(plan),
                predicate,
            };
        }
    }

    // 3) GROUP BY + aggregates (split projection into group_by
    //    and aggregates, leaving non-aggregate columns to be
    //    treated as group keys).
    let group_keys = match &select.group_by {
        GroupByExpr::All(_) => {
            return Err(FrontendError::Unsupported("GROUP BY ALL".into()));
        }
        GroupByExpr::Expressions(exprs, _) => exprs.clone(),
    };
    let has_group_by = !group_keys.is_empty();
    let has_aggregate_in_projection = select.projection.iter().any(projection_contains_aggregate);
    let has_having = select.having.is_some();
    let needs_aggregate = has_group_by || has_aggregate_in_projection || has_having;

    if needs_aggregate {
        // 4a) Group-by expressions become NamedExprs (their
        //     pre-aggregate ordinals are their position in the
        //     group_by Vec; the post-aggregate scope places them
        //     first, followed by aggregate slots).
        let group_by: Vec<NamedExpr> = group_keys
            .iter()
            .map(|g| -> Result<NamedExpr, FrontendError> {
                let expr = lower_expr_sealed(g, &scope)?;
                let name = projection_alias_for_expr(g);
                Ok(NamedExpr {
                    name: name.unwrap_or_else(|| "group_key".into()),
                    expr,
                })
            })
            .collect::<Result<_, _>>()?;
        // 4b) Project + aggregate extraction.
        let (post_agg_outputs, mut aggregates) =
            lower_projection_with_aggregates(&select.projection, &scope, &group_by)?;
        // 4c) HAVING.
        //   - Plain HAVING over group keys keeps the cheap `Aggregate.having` path.
        //   - A HAVING that references an AGGREGATE and/or a scalar SUBQUERY
        //     (TPC-H Q11: `having sum(x) > (select sum(x)*k from …)`) is lowered as
        //     a POST-AGGREGATE Filter: its aggregates are extracted INTO the
        //     Aggregate (`lower_having`), any scalar subquery is hoisted as a
        //     `LEFT JOIN ON TRUE` above it, and the predicate — over the combined
        //     [group keys | aggregates | hoisted] schema — becomes a Filter. The
        //     Project outputs are unaffected (they index the group-key + projection
        //     aggregate slots, which keep their ordinals).
        match &select.having {
            Some(expr) if expr_contains_aggregate(expr) || expr_contains_subquery(expr) => {
                let group_count = group_by.len();
                // Reserve the HAVING aggregate slots ahead of any hoisted scalar
                // column so their ordinals never collide.
                let base_width = group_count + aggregates.len() + count_aggregates(expr);
                let mut ctx = LoweringCtx::new(catalog, base_width);
                let having_expr = lower_having(
                    expr,
                    &scope,
                    &group_by,
                    group_count,
                    &mut aggregates,
                    &mut ctx,
                )?;
                plan = LogicalNode::Aggregate {
                    input: Box::new(plan),
                    group_by,
                    aggregates,
                    having: None,
                };
                for hoisted in ctx.hoisted.drain(..) {
                    plan = LogicalNode::Join {
                        left: Box::new(plan),
                        right: Box::new(hoisted.plan),
                        kind: JoinKind::Left,
                        on: hoisted.on,
                        strategy: JoinStrategy::Auto,
                    };
                }
                plan = LogicalNode::Filter {
                    input: Box::new(plan),
                    predicate: having_expr,
                };
            }
            Some(expr) => {
                let post_agg_scope = post_aggregate_scope(&group_by, &aggregates);
                let having = Some(lower_expr_sealed(expr, &post_agg_scope)?);
                plan = LogicalNode::Aggregate {
                    input: Box::new(plan),
                    group_by,
                    aggregates,
                    having,
                };
            }
            None => {
                plan = LogicalNode::Aggregate {
                    input: Box::new(plan),
                    group_by,
                    aggregates,
                    having: None,
                };
            }
        }
        plan = LogicalNode::Project {
            input: Box::new(plan),
            outputs: post_agg_outputs,
        };
    } else {
        // Lower the projection through a hoisting context so a scalar subquery in
        // an output expression (e.g. `SELECT name, (SELECT max(t) FROM o) FROM u`)
        // is rewritten to a `LEFT JOIN ON TRUE` over an `AssertMaxOneRow`-guarded
        // subplan. `base_width` is the current plan width (which already includes
        // any scalar columns a WHERE subquery appended), so projection-hoisted
        // columns append after all existing columns and ordinals never shift.
        let base_width = plan.output_schema().columns.len();
        let mut ctx = LoweringCtx::new(catalog, base_width);
        let projection_items = lower_projection_items(&select.projection, &scope, &mut ctx)?;
        // Append each hoisted subquery as a LEFT JOIN onto the base plan, in
        // allocation order, BEFORE the Project — the projection's column refs
        // point at the post-join ordinals.
        for hoisted in ctx.hoisted.drain(..) {
            plan = LogicalNode::Join {
                left: Box::new(plan),
                right: Box::new(hoisted.plan),
                kind: JoinKind::Left,
                on: hoisted.on,
                strategy: JoinStrategy::Auto,
            };
        }
        plan = LogicalNode::Project {
            input: Box::new(plan),
            outputs: projection_items,
        };
    }

    // 5) DISTINCT
    if let Some(distinct) = select.distinct.as_ref() {
        match distinct {
            Distinct::Distinct => {
                plan = LogicalNode::Distinct {
                    input: Box::new(plan),
                };
            }
            Distinct::On(_) => {
                return Err(FrontendError::Unsupported("DISTINCT ON".into()));
            }
        }
    }

    // 6) Update scope to post-projection schema (subsequent
    //    ORDER BY references use the projected schema, mirroring
    //    SQL semantics).
    scope = Scope::from_schema(&plan.output_schema());
    let _ = scope; // ORDER BY is applied at the Query layer.

    Ok(plan)
}

/// Flatten a WHERE expression into its top-level `AND` conjuncts (descending through
/// parenthesised `Nested` wrappers). Disjunctions / other operators are returned whole
/// — only conjuncts at the AND-top-level are individually liftable into Semi/Anti joins.
fn flatten_sql_and(expr: &SqlExpr) -> Vec<&SqlExpr> {
    fn rec<'a>(e: &'a SqlExpr, out: &mut Vec<&'a SqlExpr>) {
        match e {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                rec(left, out);
                rec(right, out);
            }
            SqlExpr::Nested(inner) => rec(inner, out),
            other => out.push(other),
        }
    }
    let mut out = Vec::new();
    rec(expr, &mut out);
    out
}

/// A column reference split into an optional table/alias qualifier and the bare
/// column name (`n1.n_nationkey` → `(Some("n1"), "n_nationkey")`).
type ColIdent = (Option<String>, String);

/// Order comma-separated FROM entries so the left-deep CROSS chain has an
/// equi-join key at every level. Greedy connectivity walk over the equi-graph
/// derived from the WHERE clause: start from the first entry and always append
/// the lowest-index remaining entry that shares an equi-key with an
/// already-placed entry — so a FROM order that is already fully connected is
/// reproduced unchanged. Returns the identity order when there are fewer than
/// three entries, no WHERE clause, or a disconnected component remains (a
/// genuine cross product). Reordering only ever helps: inner joins commute and
/// column refs resolve by name, so a correct result never changes.
fn comma_join_order(scopes: &[&Scope], where_clause: Option<&SqlExpr>) -> Vec<usize> {
    let n = scopes.len();
    if n < 3 {
        return (0..n).collect();
    }
    let Some(where_expr) = where_clause else {
        return (0..n).collect();
    };
    let mut adj = vec![vec![false; n]; n];
    for conj in flatten_sql_and(where_expr) {
        if let Some((a, b)) = equi_column_pair(conj)
            && let (Some(i), Some(j)) = (entry_of_column(&a, scopes), entry_of_column(&b, scopes))
            && i != j
        {
            adj[i][j] = true;
            adj[j][i] = true;
        }
    }
    let mut placed = vec![false; n];
    let mut order = Vec::with_capacity(n);
    placed[0] = true;
    order.push(0);
    while order.len() < n {
        // Lowest-index unplaced entry connected to any placed entry; if the
        // graph is disconnected, fall back to the lowest-index unplaced entry
        // (that component genuinely cross-joins — same as before).
        let next = (0..n)
            .find(|&j| !placed[j] && (0..n).any(|i| placed[i] && adj[i][j]))
            .or_else(|| (0..n).find(|&j| !placed[j]));
        let pick = next.expect("order.len() < n implies an unplaced entry");
        placed[pick] = true;
        order.push(pick);
    }
    order
}

/// If `expr` is `col = col` between two column identifiers, return both parts.
fn equi_column_pair(expr: &SqlExpr) -> Option<(ColIdent, ColIdent)> {
    let SqlExpr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return None;
    };
    Some((column_ident(left)?, column_ident(right)?))
}

/// A bare or qualified column identifier as `(table?, column)`; `None` for any
/// non-identifier expression (literals, arithmetic, function calls).
fn column_ident(expr: &SqlExpr) -> Option<ColIdent> {
    match expr {
        SqlExpr::Identifier(id) => Some((None, id.value.clone())),
        SqlExpr::CompoundIdentifier(parts) if parts.len() >= 2 => Some((
            Some(parts[parts.len() - 2].value.clone()),
            parts[parts.len() - 1].value.clone(),
        )),
        SqlExpr::Nested(inner) => column_ident(inner),
        _ => None,
    }
}

/// Which FROM entry (index into `scopes`) owns a column identifier — qualified
/// by table/alias match, unqualified by *unique* column-name ownership. `None`
/// when the column resolves to no entry or is ambiguous across entries (the
/// connectivity edge is simply dropped — conservative, never incorrect).
fn entry_of_column(ident: &ColIdent, scopes: &[&Scope]) -> Option<usize> {
    let (table, col) = ident;
    match table {
        Some(t) => scopes.iter().position(|s| {
            s.columns
                .iter()
                .any(|c| &c.table == t && &c.column.name == col)
        }),
        None => {
            let mut found = None;
            for (i, s) in scopes.iter().enumerate() {
                if s.columns.iter().any(|c| &c.column.name == col) {
                    if found.is_some() {
                        return None; // ambiguous across entries — drop the edge
                    }
                    found = Some(i);
                }
            }
            found
        }
    }
}

/// If `conj` is a liftable subquery predicate, lower it to the parts of a Semi/Anti
/// join `(kind, on, right_plan)` to wrap the current plan. Returns `Ok(None)` when
/// `conj` is not a liftable subquery (a normal predicate, or a subquery shape we
/// don't decorrelate); the caller then leaves it in the `Filter` predicate.
///
/// Two paths:
/// - **Uncorrelated** — `lower_query` builds the subquery's scope from its OWN `FROM`
///   only, so the subquery lowers in isolation. `expr IN (SELECT …)` (single- and
///   multi-column) → Semi/AntiNullAware with an equi ON; `EXISTS` → keyless Semi;
///   `NOT EXISTS` → keyless Anti.
/// - **Correlated** — when isolation lowering fails (the subquery references an outer
///   column), [`decorrelate_subquery`] lifts the correlation/outer predicate into the
///   join ON (evaluated over the combined outer++inner row) and keeps inner-local
///   predicates as the inner Filter. Correlated `EXISTS`/`NOT EXISTS` and single-column
///   `IN`/`NOT IN` are supported; any other shape declines to legacy.
fn lower_subquery_join_parts(
    conj: &SqlExpr,
    scope: &Scope,
    catalog: &dyn CatalogLookup,
) -> Result<Option<(JoinKind, Option<Expr>, LogicalNode)>, FrontendError> {
    match conj {
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            // IN → Semi; NOT IN → AntiNullAware (NULL-correct via the executor's
            // three-valued ON evaluation).
            let kind = if *negated {
                JoinKind::AntiNullAware
            } else {
                JoinKind::Semi
            };
            // Lower the subquery in isolation; correlated/unsupported → fall through
            // to the correlated path below.
            let Ok(sub) = lower_query(subquery, catalog) else {
                return lower_correlated_in(expr, subquery, kind, scope, catalog);
            };
            let sub_schema = sub.output_schema();
            // Outer operands: a row constructor `(a, b, …) [NOT] IN (…)` is a tuple;
            // a scalar `x [NOT] IN (…)` is a single operand. Their count must match the
            // subquery's column count, else decline (→ legacy).
            let outer_exprs: Vec<&SqlExpr> = match expr.as_ref() {
                SqlExpr::Tuple(items) => items.iter().collect(),
                single => vec![single],
            };
            if outer_exprs.len() != sub_schema.columns.len() {
                return Ok(None);
            }
            // Build the equi ON = AND over `outer_i = subcol_i`. Each subquery column
            // sits at `outer_width + i` in the combined left++right row (Scope::concat
            // offsets right ordinals by the left width).
            let base = scope.columns.len();
            let mut on: Option<Expr> = None;
            for (i, oexpr) in outer_exprs.iter().enumerate() {
                let outer = lower_expr_sealed(oexpr, scope)?;
                let sub_col = &sub_schema.columns[i];
                let sub_ref = ColumnRef {
                    name: sub_col.name.clone(),
                    ordinal: base + i,
                    ty: sub_col.ty.clone(),
                    nullable: sub_col.nullable,
                };
                let eq = Expr::bin(BinaryOp::Eq, outer, Expr::column(sub_ref));
                on = Some(match on {
                    Some(prev) => Expr::bin(BinaryOp::And, prev, eq),
                    None => eq,
                });
            }
            Ok(Some((kind, on, sub)))
        }
        SqlExpr::Exists { subquery, negated } => {
            // EXISTS → Semi; NOT EXISTS → Anti.
            let kind = if *negated {
                JoinKind::Anti
            } else {
                JoinKind::Semi
            };
            // Uncorrelated: the subquery lowers in isolation; keyless Semi/Anti
            // (any matching inner row qualifies the outer row).
            if let Ok(sub) = lower_query(subquery, catalog) {
                return Ok(Some((kind, None, sub)));
            }
            // Correlated: decorrelate — lift the correlation/outer predicate to the
            // join ON (evaluated over the combined outer++inner row), keep inner-local
            // predicates as the inner Filter. Unsupported shape → decline to legacy.
            match decorrelate_subquery(subquery, scope, catalog)? {
                Some(d) => Ok(Some((kind, Some(d.correlation), d.inner))),
                None => Ok(None),
            }
        }
        // Everything else is a normal predicate handled by the Filter.
        _ => Ok(None),
    }
}

/// A correlated subquery rewritten for a Semi/Anti join: the inner relation
/// (Scan + inner-local Filter) and the correlation predicate lifted to the join
/// `ON` (resolved over the COMBINED `outer ++ inner` row).
struct DecorrelatedSubquery {
    /// Inner relation with any inner-local WHERE predicates applied. Its rows are
    /// the subquery table's full columns (no projection — Semi/Anti emit the left
    /// side only, and `lower_correlated_in` resolves the IN column by ordinal).
    inner: LogicalNode,
    /// Inner table scope (ordinals 0..inner_width, standalone).
    inner_scope: Scope,
    /// Correlation predicate over the combined `outer ++ inner` row (outer columns
    /// at `0..outer_width`, inner columns at `outer_width..`).
    correlation: Expr,
}

/// Attempt to decorrelate a simple correlated subquery against `outer_scope`.
/// Returns `Ok(None)` when the subquery isn't a shape we can safely decorrelate
/// (the caller declines to the legacy path).
///
/// Supported shape: `SELECT … FROM <one table> WHERE <AND-conjuncts>` with NO
/// joins / GROUP BY / HAVING / DISTINCT / set-op / ORDER BY / LIMIT, where at
/// least one WHERE conjunct references an outer column (the correlation) and
/// every conjunct resolves against either the inner table alone (→ inner Filter)
/// or the combined outer+inner scope (→ join ON). Correlation under OR, or an
/// unresolvable conjunct, declines.
fn decorrelate_subquery(
    subquery: &SqlQuery,
    outer_scope: &Scope,
    catalog: &dyn CatalogLookup,
) -> Result<Option<DecorrelatedSubquery>, FrontendError> {
    // Reject query-level decoration we can't reason about under decorrelation.
    if subquery.with.is_some() || subquery.order_by.is_some() || subquery.limit_clause.is_some() {
        return Ok(None);
    }
    let SetExpr::Select(select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    // A single base table, no grouping/having/distinct, and a WHERE to split.
    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Ok(None);
    }
    let has_group_by = match &select.group_by {
        GroupByExpr::All(_) => true,
        GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
    };
    if has_group_by || select.having.is_some() || select.distinct.is_some() {
        return Ok(None);
    }
    let Some(where_expr) = &select.selection else {
        return Ok(None);
    };
    let (inner_plan, inner_scope) = lower_table_factor(&select.from[0].relation, catalog)?;
    let combined = outer_scope.concat(&inner_scope);

    // Split the WHERE conjuncts: those that resolve against the inner table alone
    // stay as the inner Filter; those that need the outer scope are correlation
    // predicates lifted to the join ON. An unresolvable conjunct declines.
    let mut inner_local: Option<Expr> = None;
    let mut correlation: Option<Expr> = None;
    for conj in flatten_sql_and(where_expr) {
        if let Ok(local) = lower_expr_sealed(conj, &inner_scope) {
            inner_local = Some(match inner_local {
                Some(prev) => Expr::bin(BinaryOp::And, prev, local),
                None => local,
            });
        } else if let Ok(corr) = lower_expr_sealed(conj, &combined) {
            correlation = Some(match correlation {
                Some(prev) => Expr::bin(BinaryOp::And, prev, corr),
                None => corr,
            });
        } else {
            // References neither resolvable inner nor combined columns (or is
            // ambiguous across them) → don't risk a wrong rewrite.
            return Ok(None);
        }
    }
    // No correlation conjunct → not actually correlated (and `lower_query` already
    // failed for some other reason) → decline.
    let Some(correlation) = correlation else {
        return Ok(None);
    };
    let inner = match inner_local {
        Some(predicate) => LogicalNode::Filter {
            input: Box::new(inner_plan),
            predicate,
        },
        None => inner_plan,
    };
    Ok(Some(DecorrelatedSubquery {
        inner,
        inner_scope,
        correlation,
    }))
}

/// Lower a correlated `expr [NOT] IN (SELECT col FROM … WHERE <correlation>)` to a
/// Semi/AntiNullAware join. The subquery must project exactly one plain column and
/// the outer operand must be scalar (not a row constructor); otherwise decline.
fn lower_correlated_in(
    expr: &SqlExpr,
    subquery: &SqlQuery,
    kind: JoinKind,
    outer_scope: &Scope,
    catalog: &dyn CatalogLookup,
) -> Result<Option<(JoinKind, Option<Expr>, LogicalNode)>, FrontendError> {
    // Correlated multi-column IN is out of scope for MVP.
    if matches!(expr, SqlExpr::Tuple(_)) {
        return Ok(None);
    }
    let Some(d) = decorrelate_subquery(subquery, outer_scope, catalog)? else {
        return Ok(None);
    };
    // The subquery must project exactly one plain column; resolve it against the
    // inner scope, then offset into the combined row.
    let Some(in_col) = single_projected_column(subquery, &d.inner_scope)? else {
        return Ok(None);
    };
    let combined_in_col = ColumnRef {
        ordinal: outer_scope.columns.len() + in_col.ordinal,
        ..in_col
    };
    let outer = lower_expr_sealed(expr, outer_scope)?;
    let in_eq = Expr::bin(BinaryOp::Eq, outer, Expr::column(combined_in_col));
    // ON = correlation AND (outer = inner_col).
    let on = Expr::bin(BinaryOp::And, d.correlation, in_eq);
    Ok(Some((kind, Some(on), d.inner)))
}

/// If a subquery's SELECT list is exactly one plain column reference, resolve it
/// against `inner_scope` and return its `ColumnRef`. Returns `Ok(None)` for any
/// other projection shape (wildcard, expression, multiple items, aggregate).
fn single_projected_column(
    subquery: &SqlQuery,
    inner_scope: &Scope,
) -> Result<Option<ColumnRef>, FrontendError> {
    let SetExpr::Select(select) = subquery.body.as_ref() else {
        return Ok(None);
    };
    let [item] = select.projection.as_slice() else {
        return Ok(None);
    };
    let col_expr = match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => e,
        _ => return Ok(None),
    };
    match col_expr {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => Ok(Some(
            lower_expr_sealed(col_expr, inner_scope).and_then(|e| match e {
                Expr::Column(c) => Ok(c),
                _ => Err(FrontendError::Unsupported(
                    "non-column IN projection".into(),
                )),
            })?,
        )),
        _ => Ok(None),
    }
}

fn projection_contains_aggregate(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
            expr_contains_aggregate(e)
        }
        _ => false,
    }
}

fn expr_contains_aggregate(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Function(f) => is_aggregate_function_name(&f.name),
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_contains_aggregate(left) || expr_contains_aggregate(right)
        }
        SqlExpr::UnaryOp { expr, .. } => expr_contains_aggregate(expr),
        SqlExpr::IsNull(e)
        | SqlExpr::IsNotNull(e)
        | SqlExpr::Nested(e)
        | SqlExpr::IsTrue(e)
        | SqlExpr::IsFalse(e)
        | SqlExpr::IsNotTrue(e)
        | SqlExpr::IsNotFalse(e) => expr_contains_aggregate(e),
        _ => false,
    }
}

fn is_aggregate_function_name(name: &ObjectName) -> bool {
    let n = name.to_string().to_uppercase();
    // Builtin aggregates, plus any aggregate registered in the shared function registry
    // (so a custom UDAF like `product` is routed to the aggregate path, not scalar lowering).
    matches!(n.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX")
        || proximadb_functions::builtins()
            .lookup_aggregate(&n)
            .is_some()
}

fn projection_alias_for_expr(e: &SqlExpr) -> Option<String> {
    match e {
        SqlExpr::Identifier(id) => Some(id.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => parts.last().map(|i| i.value.clone()),
        _ => None,
    }
}

// =========================================================================
// FROM clause + joins
// =========================================================================

/// Lower a FROM clause (one entry, explicit joins, or comma-separated) into a
/// plan + combined scope. Comma-separated FROM is an implicit cross join:
/// `FROM a, b, c` becomes a left-deep chain of CROSS joins, with the equality
/// predicates in WHERE acting as the join conditions (the optimizer rewrites
/// Cross+Filter into hash joins). This is the classic ANSI/TPC-H join spelling.
///
/// Lower every top-level entry once, then ORDER the comma entries by equi-join
/// connectivity (`comma_join_order`, using `where_clause`) before building the
/// chain: a FROM-order accident (a pair with no direct equi-key placed adjacent —
/// e.g. `part, supplier`, joined only transitively through `lineitem` in TPC-H
/// Q9) would otherwise leave a raw cross product that the native route declines.
/// Comma/inner joins commute and column refs resolve by name against the
/// reordered scope, so this preserves semantics; it is a no-op when the FROM
/// order is already fully connected or when connectivity can't be resolved.
///
/// Shared by `lower_select` and the correlated-scalar-subquery decorrelator
/// (TD-REL-LOWER-5), so a correlated subquery may have a multi-table body.
/// `from` must be non-empty (callers check).
fn lower_from(
    from: &[sqlparser::ast::TableWithJoins],
    where_clause: Option<&SqlExpr>,
    catalog: &dyn CatalogLookup,
) -> Result<(LogicalNode, Scope), FrontendError> {
    let lowered = from
        .iter()
        .map(|twj| lower_table_with_joins(twj, catalog))
        .collect::<Result<Vec<_>, _>>()?;
    let order = comma_join_order(
        &lowered.iter().map(|(_, s)| s).collect::<Vec<_>>(),
        where_clause,
    );
    let mut lowered: Vec<Option<(LogicalNode, Scope)>> = lowered.into_iter().map(Some).collect();
    let mut order_iter = order.into_iter();
    let first = order_iter
        .next()
        .expect("FROM is non-empty (callers check)");
    let (mut plan, mut scope) = lowered[first].take().expect("each index taken once");
    for idx in order_iter {
        let (right, right_scope) = lowered[idx].take().expect("each index taken once");
        scope = scope.concat(&right_scope);
        plan = LogicalNode::Join {
            left: Box::new(plan),
            right: Box::new(right),
            kind: JoinKind::Cross,
            on: None,
            strategy: JoinStrategy::Auto,
        };
    }
    Ok((plan, scope))
}

fn lower_table_with_joins(
    twj: &sqlparser::ast::TableWithJoins,
    catalog: &dyn CatalogLookup,
) -> Result<(LogicalNode, Scope), FrontendError> {
    let (mut plan, mut scope) = lower_table_factor(&twj.relation, catalog)?;
    for j in &twj.joins {
        let (right, right_scope) = lower_table_factor(&j.relation, catalog)?;
        let kind = match &j.join_operator {
            // The `OUTER` keyword is optional SQL noise: sqlparser parses bare
            // `JOIN`/`LEFT JOIN`/`RIGHT JOIN` as `Join`/`Left`/`Right`, distinct
            // from the `INNER`/`LEFT OUTER`/`RIGHT OUTER` spellings — both map to
            // the same `JoinKind`. (`FULL JOIN` is folded into `FullOuter`.)
            JoinOperator::Inner(_) | JoinOperator::Join(_) => JoinKind::Inner,
            JoinOperator::LeftOuter(_) | JoinOperator::Left(_) => JoinKind::Left,
            JoinOperator::RightOuter(_) | JoinOperator::Right(_) => JoinKind::Right,
            JoinOperator::FullOuter(_) => JoinKind::Full,
            JoinOperator::CrossJoin(_) => JoinKind::Cross,
            other => {
                return Err(FrontendError::Unsupported(format!(
                    "join operator {:?}",
                    std::mem::discriminant(other)
                )));
            }
        };
        // Merge scopes: the combined relation has columns from
        // left followed by columns from right.
        let combined = scope.concat(&right_scope);
        let on = match &j.join_operator {
            JoinOperator::Inner(c)
            | JoinOperator::Join(c)
            | JoinOperator::LeftOuter(c)
            | JoinOperator::Left(c)
            | JoinOperator::RightOuter(c)
            | JoinOperator::Right(c)
            | JoinOperator::FullOuter(c) => match c {
                JoinConstraint::On(expr) => Some(lower_expr_sealed(expr, &combined)?),
                JoinConstraint::Using(_) => {
                    return Err(FrontendError::Unsupported("USING constraint".into()));
                }
                JoinConstraint::Natural => {
                    return Err(FrontendError::Unsupported("NATURAL JOIN".into()));
                }
                JoinConstraint::None => None,
            },
            JoinOperator::CrossJoin(_) => None,
            _ => None,
        };
        plan = LogicalNode::Join {
            left: Box::new(plan),
            right: Box::new(right),
            kind,
            on,
            strategy: JoinStrategy::Auto,
        };
        scope = combined;
    }
    Ok((plan, scope))
}

fn lower_table_factor(
    tf: &TableFactor,
    catalog: &dyn CatalogLookup,
) -> Result<(LogicalNode, Scope), FrontendError> {
    match tf {
        // A table-valued function `FROM name(args)` (a cross-modal source such as
        // vector_search / timeseries_range / graph_traverse) is not a catalog table.
        // Decline it cleanly so the pgwire path falls back to DataFusion's SQL planner
        // (where the UDTF is registered) instead of mis-lowering it as a bare scan with
        // its args dropped.
        TableFactor::Table { args: Some(_), .. } => {
            Err(FrontendError::Unsupported("table-valued function".into()))
        }
        TableFactor::Table { name, alias, .. } => {
            let table_name = name.to_string();
            let schema = catalog
                .lookup_table(&table_name)
                .ok_or_else(|| FrontendError::TableNotFound(table_name.clone()))?;
            // Scope names carry no quote characters: a quoted `FROM "CaseTbl"`
            // serializes with its quotes in `table_name`, but qualified column
            // references compare against the bare identifier value.
            let scope_name = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| table_name.trim_matches('"').to_string());
            let scope = Scope::from_table(&scope_name, &schema);
            let plan = LogicalNode::Scan {
                table: TableId::new(table_name),
                // The plan IR owns its schema; one clone at plan-build time
                // (the trait's Arc already saved the per-lookup deep clone).
                table_schema: (*schema).clone(),
                projected_columns: None,
                predicate: None,
            };
            Ok((plan, scope))
        }
        TableFactor::Derived {
            subquery, alias, ..
        } => {
            let inner = lower_query(subquery, catalog)?;
            let schema = inner.output_schema();
            let scope_name = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| "subquery".into());
            let scope = Scope::from_table(&scope_name, &schema);
            Ok((inner, scope))
        }
        other => Err(FrontendError::Unsupported(format!(
            "table factor {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

// =========================================================================
// Projection lowering
// =========================================================================

fn lower_projection_items(
    items: &[SelectItem],
    scope: &Scope,
    ctx: &mut LoweringCtx,
) -> Result<Vec<NamedExpr>, FrontendError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            SelectItem::UnnamedExpr(e) => {
                let expr = lower_expr(e, scope, ctx)?;
                let name =
                    projection_alias_for_expr(e).unwrap_or_else(|| auto_column_name(out.len()));
                out.push(NamedExpr { name, expr });
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                let e = lower_expr(expr, scope, ctx)?;
                out.push(NamedExpr {
                    name: alias.value.clone(),
                    expr: e,
                });
            }
            SelectItem::Wildcard(_) => {
                // SELECT * → expand to every column in the BASE scope only.
                // `scope` reflects the base relation (hoisted scalar-subquery
                // columns are appended AFTER projection lowering), so `*` never
                // exposes them.
                for col in scope.all_columns() {
                    out.push(NamedExpr {
                        name: col.name.clone(),
                        expr: Expr::column(col.clone()),
                    });
                }
            }
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(FrontendError::Unsupported("qualified wildcard".into()));
            }
        }
    }
    Ok(out)
}

fn auto_column_name(idx: usize) -> String {
    format!("col{}", idx + 1)
}

// =========================================================================
// Expression lowering
// =========================================================================

/// Threaded through expression lowering to collect scalar subqueries that must
/// be hoisted into `LEFT JOIN ON TRUE` relations. The caller (e.g.
/// [`lower_projection_items`]) drains [`LoweringCtx::hoisted`] after lowering and
/// appends each as a `LEFT JOIN` onto the outer plan.
///
/// In positions where a subquery is not supported (VALUES rows, ORDER BY,
/// HAVING, JOIN ON, function arguments, the GROUP BY/aggregate projection path,
/// `CREATE FUNCTION` bodies) the position is *sealed* — `catalog` is `None` and a
/// subquery errors with [`FrontendError::Unsupported`]. Those callers use the
/// [`lower_expr_sealed`] wrapper.
struct LoweringCtx<'a> {
    /// `Some` when scalar subqueries are allowed here; `None` seals the position.
    catalog: Option<&'a dyn CatalogLookup>,
    /// Column width of the base relation, frozen BEFORE any hoist. Hoisted
    /// subquery columns are appended AFTER these, so base ordinals never shift.
    base_width: usize,
    /// Subquery relations to `LEFT JOIN` onto the base relation, in allocation
    /// order. Each is an `AssertMaxOneRow`-guarded plan (uncorrelated scalar).
    hoisted: Vec<HoistedSub>,
}

/// A scalar subquery hoisted out of an expression, to be `LEFT JOIN`ed onto the
/// outer plan.
struct HoistedSub {
    /// The relation to join (an `AssertMaxOneRow` for an uncorrelated scalar).
    plan: LogicalNode,
    /// Join predicate. `None` = ON TRUE (uncorrelated scalar: zero rows → NULL,
    /// one row → value).
    on: Option<Expr>,
}

impl<'a> LoweringCtx<'a> {
    fn new(catalog: &'a dyn CatalogLookup, base_width: usize) -> Self {
        Self {
            catalog: Some(catalog),
            base_width,
            hoisted: Vec::new(),
        }
    }

    /// A position where subqueries are disallowed.
    fn sealed() -> Self {
        Self {
            catalog: None,
            base_width: 0,
            hoisted: Vec::new(),
        }
    }

    /// Ordinal the NEXT hoisted column will occupy in the combined row: base
    /// width plus the total width of all subqueries hoisted so far.
    fn next_ordinal(&self) -> usize {
        self.base_width
            + self
                .hoisted
                .iter()
                .map(|h| h.plan.output_schema().columns.len())
                .sum::<usize>()
    }
}

/// Lower an expression in a position where subqueries are not allowed. Builds a
/// sealed [`LoweringCtx`]; a subquery encountered here errors `Unsupported`.
fn lower_expr_sealed(expr: &SqlExpr, scope: &Scope) -> Result<Expr, FrontendError> {
    let mut ctx = LoweringCtx::sealed();
    let lowered = lower_expr(expr, scope, &mut ctx)?;
    debug_assert!(ctx.hoisted.is_empty(), "sealed lowering must not hoist");
    lowered_no_hoist(lowered, &ctx)
}

/// Guard: a sealed context must never have accumulated a hoist (the subquery arm
/// rejects before pushing). Returns the expression unchanged.
fn lowered_no_hoist(expr: Expr, ctx: &LoweringCtx) -> Result<Expr, FrontendError> {
    if ctx.hoisted.is_empty() {
        Ok(expr)
    } else {
        Err(FrontendError::Unsupported(
            "subquery not allowed in this position".into(),
        ))
    }
}

/// Lower a scalar subquery `(SELECT col FROM …)` appearing inside an expression.
///
/// - **Uncorrelated** — lowers in isolation; hoisted into `ctx` as an
///   `AssertMaxOneRow`-guarded relation that the caller `LEFT JOIN`s ON TRUE
///   (zero rows → NULL, one row → value, two+ → error). Returns a `ColumnRef` to
///   its single output column.
/// - **Correlated** — when isolation lowering fails, [`lower_correlated_scalar_aggregate`]
///   decorrelates a single-key aggregate subquery to a `LEFT JOIN` over a `GROUP BY`.
fn lower_scalar_subquery(
    q: &SqlQuery,
    outer_scope: &Scope,
    ctx: &mut LoweringCtx,
) -> Result<Expr, FrontendError> {
    let Some(catalog) = ctx.catalog else {
        return Err(FrontendError::Unsupported(
            "subquery not allowed in this position".into(),
        ));
    };
    // Uncorrelated: lowers from its own FROM only.
    if let Ok(sub_plan) = lower_query(q, catalog) {
        let sub_schema = sub_plan.output_schema();
        if sub_schema.columns.len() != 1 {
            return Err(FrontendError::Unsupported(
                "scalar subquery must return exactly one column".into(),
            ));
        }
        let col = sub_schema.columns[0].clone();
        let ordinal = ctx.next_ordinal();
        ctx.hoisted.push(HoistedSub {
            plan: LogicalNode::AssertMaxOneRow {
                input: Box::new(sub_plan),
            },
            on: None,
        });
        return Ok(Expr::column(ColumnRef {
            name: col.name,
            ordinal,
            ty: col.ty,
            // Empty subquery → LEFT JOIN null-pads → the value is always possibly NULL.
            nullable: true,
        }));
    }
    // Correlated scalar aggregate → LEFT JOIN + GROUP BY.
    lower_correlated_scalar_aggregate(q, outer_scope, ctx, catalog)
}

/// Decorrelate a correlated scalar aggregate subquery
/// (`(SELECT max(t) FROM o WHERE o.k = u.k)`) into a `LEFT JOIN` over a `GROUP BY`:
/// `u LEFT JOIN (SELECT k, max(t) FROM o [WHERE inner-local] GROUP BY k) agg ON u.k = agg.k`.
/// A non-matching outer row gets a NULL aggregate, which is the correct SQL result
/// for `max/min/sum/avg` over an empty set. `COUNT` is declined (its empty-set result
/// is 0, not NULL — the LEFT JOIN can't express that). Any non-conforming shape
/// (multiple/zero correlation keys, non-equi or non-column correlation, non-aggregate
/// projection, joins/grouping/etc.) returns `Unsupported` → the query declines to legacy.
fn lower_correlated_scalar_aggregate(
    q: &SqlQuery,
    outer_scope: &Scope,
    ctx: &mut LoweringCtx,
    catalog: &dyn CatalogLookup,
) -> Result<Expr, FrontendError> {
    let unsupported = || FrontendError::Unsupported("correlated scalar subquery".into());
    if q.with.is_some() || q.order_by.is_some() || q.limit_clause.is_some() {
        return Err(unsupported());
    }
    let SetExpr::Select(select) = q.body.as_ref() else {
        return Err(unsupported());
    };
    if select.from.is_empty() {
        return Err(unsupported());
    }
    // The subquery body may be MULTI-TABLE (TPC-H Q2: `min(ps_supplycost) from
    // partsupp, supplier, nation, region where …`). It lowers exactly like the
    // main FROM — a connectivity-ordered cross chain the planner equi-fies — via
    // the shared `lower_from`. The single correlation equi is still split out
    // below; inner join conditions and inner-local filters fold into the Filter.
    let has_group_by = match &select.group_by {
        GroupByExpr::All(_) => true,
        GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
    };
    if has_group_by || select.having.is_some() || select.distinct.is_some() {
        return Err(unsupported());
    }
    let Some(where_expr) = &select.selection else {
        return Err(unsupported());
    };
    // Projection must be exactly one aggregate call.
    let [item] = select.projection.as_slice() else {
        return Err(unsupported());
    };
    let (proj_sql, alias) = match item {
        SelectItem::UnnamedExpr(e) => (e, None),
        SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.clone())),
        _ => return Err(unsupported()),
    };
    // The projection is a single aggregate, optionally wrapped in scalar
    // arithmetic over constants / outer columns (e.g. `0.2 * avg(x)` in TPC-H
    // Q17, `sum(x)/7.0`). Decorrelate the one aggregate as usual; the wrapper is
    // re-applied to its post-join value column at the end (`build_scalar_over_agg`).
    let f = single_aggregate_call(proj_sql).ok_or_else(unsupported)?;
    let (inner_plan, inner_scope) = lower_from(&select.from, Some(where_expr), catalog)?;
    let agg = lower_aggregate_call(f, &inner_scope)?;
    // The count-bug: COUNT over an empty correlated group must be 0, but the
    // LEFT JOIN + GROUP BY rewrite yields NULL. Decline COUNT.
    if matches!(agg, AggregateExpr::Count { .. }) {
        return Err(unsupported());
    }
    // Split WHERE: inner-local predicates → Filter; correlation equis
    // `inner_col = outer_col` → the GROUP BY keys / join keys. N keys are supported
    // (TPC-H Q20's inner scalar correlates on TWO keys: `l_partkey = ps_partkey AND
    // l_suppkey = ps_suppkey`).
    let mut inner_local: Option<Expr> = None;
    let mut group_keys: Vec<(ColumnRef, ColumnRef)> = Vec::new(); // (inner_key, outer_key)
    for conj in flatten_sql_and(where_expr) {
        if let Ok(local) = lower_expr_sealed(conj, &inner_scope) {
            inner_local = Some(match inner_local {
                Some(prev) => Expr::bin(BinaryOp::And, prev, local),
                None => local,
            });
            continue;
        }
        // Correlation conjunct — must be a column-to-column equality with one side
        // in the inner scope and the other in the outer scope.
        let SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = strip_nested(conj)
        else {
            return Err(unsupported());
        };
        group_keys.push(resolve_correlation_pair(
            left,
            right,
            &inner_scope,
            outer_scope,
        )?);
    }
    if group_keys.is_empty() {
        return Err(unsupported()); // not correlated → not this path
    }
    let inner = match inner_local {
        Some(predicate) => LogicalNode::Filter {
            input: Box::new(inner_plan),
            predicate,
        },
        None => inner_plan,
    };
    // GROUP BY every inner correlation key; the aggregate is the scalar value. The
    // aggregate output schema is [group_key_0 .. group_key_{n-1}, agg_value].
    let agg_name = alias.unwrap_or_else(|| f.name.to_string().to_lowercase());
    let plan = LogicalNode::Aggregate {
        input: Box::new(inner),
        group_by: group_keys
            .iter()
            .map(|(inner_key, _)| NamedExpr {
                name: inner_key.name.clone(),
                expr: Expr::column(inner_key.clone()),
            })
            .collect(),
        aggregates: vec![NamedAggregate {
            name: agg_name.clone(),
            agg: agg.clone(),
        }],
        having: None,
    };
    // In the combined row after the LEFT JOIN, the N group keys sit at `base .. base+n`
    // and the aggregate value at `base + n` (group keys precede the aggregate slot).
    // The join ON is the conjunction of every `outer_key_i = inner_key_i` pair.
    let base = ctx.next_ordinal();
    let on = group_keys
        .iter()
        .enumerate()
        .map(|(i, (inner_key, outer_key))| {
            Expr::bin(
                BinaryOp::Eq,
                Expr::column(outer_key.clone()),
                Expr::column(ColumnRef {
                    name: inner_key.name.clone(),
                    ordinal: base + i,
                    ty: inner_key.ty.clone(),
                    nullable: true,
                }),
            )
        })
        .reduce(|acc, e| Expr::bin(BinaryOp::And, acc, e))
        .expect("group_keys is non-empty (checked above)");
    let value_ty = agg.result_type();
    ctx.hoisted.push(HoistedSub { plan, on: Some(on) });
    let agg_col = Expr::column(ColumnRef {
        name: agg_name,
        ordinal: base + group_keys.len(),
        ty: value_ty,
        nullable: true,
    });
    // Re-apply the scalar arithmetic that wrapped the aggregate (a no-op when the
    // projection was a bare aggregate), with the aggregate replaced by its
    // post-join value column. Non-aggregate leaves (literals, outer columns)
    // resolve against the outer scope — valid because the LEFT JOIN places the
    // outer columns first, so their ordinals are unchanged in the combined row.
    build_scalar_over_agg(proj_sql, &agg_col, outer_scope)
}

/// The single aggregate call inside a scalar-subquery projection, which may be
/// wrapped in scalar arithmetic (`0.2 * avg(x)`, `sum(x) / 7.0`). Returns `None`
/// when the projection has zero or more than one aggregate (the correlated
/// scalar-aggregate decorrelation handles exactly one).
fn single_aggregate_call(sql: &SqlExpr) -> Option<&sqlparser::ast::Function> {
    fn collect<'a>(sql: &'a SqlExpr, out: &mut Vec<&'a sqlparser::ast::Function>) {
        match sql {
            SqlExpr::Function(f) if is_aggregate_function_name(&f.name) => out.push(f),
            SqlExpr::BinaryOp { left, right, .. } => {
                collect(left, out);
                collect(right, out);
            }
            SqlExpr::UnaryOp { expr, .. } => collect(expr, out),
            SqlExpr::Nested(e) => collect(e, out),
            _ => {}
        }
    }
    let mut v = Vec::new();
    collect(sql, &mut v);
    (v.len() == 1).then(|| v[0])
}

/// Rebuild the scalar arithmetic that wrapped a correlated scalar aggregate,
/// substituting the (single) aggregate call with `agg_col` — the decorrelated
/// aggregate's post-join value column. Arithmetic-spine nodes recurse; any other
/// leaf (a literal or an outer column) lowers against `outer_scope`. There is
/// exactly one aggregate (guaranteed by [`single_aggregate_call`]), so matching
/// any aggregate function here yields `agg_col`.
fn build_scalar_over_agg(
    sql: &SqlExpr,
    agg_col: &Expr,
    outer_scope: &Scope,
) -> Result<Expr, FrontendError> {
    match sql {
        SqlExpr::Function(f) if is_aggregate_function_name(&f.name) => Ok(agg_col.clone()),
        SqlExpr::BinaryOp { left, op, right } => Ok(Expr::bin(
            lower_binary_op(op)?,
            build_scalar_over_agg(left, agg_col, outer_scope)?,
            build_scalar_over_agg(right, agg_col, outer_scope)?,
        )),
        SqlExpr::UnaryOp { op, expr } => {
            let inner = build_scalar_over_agg(expr, agg_col, outer_scope)?;
            match lower_unary_op(op)? {
                Some(u) => Ok(Expr::UnaryOp {
                    op: u,
                    expr: Box::new(inner),
                }),
                None => Ok(inner),
            }
        }
        SqlExpr::Nested(e) => build_scalar_over_agg(e, agg_col, outer_scope),
        other => lower_expr_sealed(other, outer_scope),
    }
}

/// Resolve a correlation equality's two operands into `(inner_key, outer_key)`
/// column refs: one side must be a plain column of the inner scope, the other a
/// plain column of the outer scope. Errors otherwise (non-column side, or both
/// sides on the same side).
fn resolve_correlation_pair(
    a: &SqlExpr,
    b: &SqlExpr,
    inner_scope: &Scope,
    outer_scope: &Scope,
) -> Result<(ColumnRef, ColumnRef), FrontendError> {
    let unsupported = || FrontendError::Unsupported("correlated scalar subquery key".into());
    let as_col = |e: &SqlExpr, scope: &Scope| -> Option<ColumnRef> {
        match lower_expr_sealed(e, scope) {
            Ok(Expr::Column(c)) => Some(c),
            _ => None,
        }
    };
    // a ∈ inner, b ∈ outer
    if let (Some(inner), Some(outer)) = (as_col(a, inner_scope), as_col(b, outer_scope)) {
        return Ok((inner, outer));
    }
    // a ∈ outer, b ∈ inner
    if let (Some(outer), Some(inner)) = (as_col(a, outer_scope), as_col(b, inner_scope)) {
        return Ok((inner, outer));
    }
    Err(unsupported())
}

/// Strip parenthesised `Nested` wrappers from an expression.
fn strip_nested(expr: &SqlExpr) -> &SqlExpr {
    match expr {
        SqlExpr::Nested(inner) => strip_nested(inner),
        other => other,
    }
}

fn lower_expr(expr: &SqlExpr, scope: &Scope, ctx: &mut LoweringCtx) -> Result<Expr, FrontendError> {
    match expr {
        SqlExpr::Nested(inner) => lower_expr(inner, scope, ctx),
        SqlExpr::Identifier(id) => scope
            .resolve_unqualified(&id.value, id.quote_style.is_some())
            .map(Expr::column),
        SqlExpr::CompoundIdentifier(parts) => {
            if parts.len() < 2 {
                return Err(FrontendError::ColumnNotFound(
                    parts
                        .iter()
                        .map(|i| i.value.clone())
                        .collect::<Vec<_>>()
                        .join("."),
                ));
            }
            let table = &parts[0].value;
            let column = &parts[1].value;
            scope
                .resolve_qualified(table, column, parts[1].quote_style.is_some())
                .map(Expr::column)
        }
        SqlExpr::Value(v) => Ok(Expr::literal(lower_value(&v.value)?)),
        // `DATE 'YYYY-MM-DD'` typed literal → the canonical `ProximaValue::Date`
        // (days since the Unix epoch), matching the write path's encoding so a
        // `col < DATE '…'` predicate compares correctly against a stored Date32
        // column. TPC-H/TPC-DS date filters (Q1/Q3/Q5/Q10/Q12/… ) declined here
        // before this. Other typed strings (TIME/TIMESTAMP/INTERVAL) stay
        // unsupported for now — a scoped follow-up.
        SqlExpr::TypedString(ts) if matches!(ts.data_type, SqlDataType::Date) => {
            let raw = ts.value.value.clone().into_string().ok_or_else(|| {
                FrontendError::InvalidLiteral(format!("DATE literal: {:?}", ts.value.value))
            })?;
            let days = proximadb_data_model::parse_iso_date_to_days(&raw)
                .ok_or_else(|| FrontendError::InvalidLiteral(format!("DATE '{raw}'")))?;
            Ok(Expr::literal(ProximaValue::Date(days)))
        }
        SqlExpr::TypedString(ts) => Err(FrontendError::Unsupported(format!(
            "typed string literal: {}",
            ts.data_type
        ))),
        SqlExpr::BinaryOp { left, op, right } => {
            let l = lower_expr(left, scope, ctx)?;
            let r = lower_expr(right, scope, ctx)?;
            Ok(Expr::bin(lower_binary_op(op)?, l, r))
        }
        SqlExpr::UnaryOp { op, expr } => {
            let inner = lower_expr(expr, scope, ctx)?;
            match lower_unary_op(op)? {
                Some(uop) => Ok(Expr::unary(uop, inner)),
                None => Ok(inner), // unary +
            }
        }
        SqlExpr::IsNull(e) => Ok(Expr::IsNull {
            expr: Box::new(lower_expr(e, scope, ctx)?),
            not: false,
        }),
        SqlExpr::IsNotNull(e) => Ok(Expr::IsNull {
            expr: Box::new(lower_expr(e, scope, ctx)?),
            not: true,
        }),
        SqlExpr::Between {
            expr,
            low,
            high,
            negated,
        } => Ok(Expr::Between {
            expr: Box::new(lower_expr(expr, scope, ctx)?),
            low: Box::new(lower_expr(low, scope, ctx)?),
            high: Box::new(lower_expr(high, scope, ctx)?),
            not: *negated,
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => Ok(Expr::In {
            expr: Box::new(lower_expr(expr, scope, ctx)?),
            list: list
                .iter()
                .map(|e| lower_expr(e, scope, ctx))
                .collect::<Result<Vec<_>, _>>()?,
            not: *negated,
        }),
        SqlExpr::Like {
            expr,
            pattern,
            negated,
            ..
        } => Ok(Expr::Like {
            expr: Box::new(lower_expr(expr, scope, ctx)?),
            pattern: Box::new(lower_expr(pattern, scope, ctx)?),
            not: *negated,
            case_insensitive: false,
        }),
        SqlExpr::ILike {
            expr,
            pattern,
            negated,
            ..
        } => Ok(Expr::Like {
            expr: Box::new(lower_expr(expr, scope, ctx)?),
            pattern: Box::new(lower_expr(pattern, scope, ctx)?),
            not: *negated,
            case_insensitive: true,
        }),
        SqlExpr::Function(f) => lower_scalar_function(f, scope, ctx),
        // CASE — both the searched form (`CASE WHEN cond THEN r ...`) and the simple
        // form (`CASE op WHEN v THEN r ...`, lowered to `op = v` branch conditions).
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            let mut branches = Vec::with_capacity(conditions.len());
            for cw in conditions {
                let cond = match operand {
                    Some(op) => Expr::bin(
                        BinaryOp::Eq,
                        lower_expr(op, scope, ctx)?,
                        lower_expr(&cw.condition, scope, ctx)?,
                    ),
                    None => lower_expr(&cw.condition, scope, ctx)?,
                };
                branches.push((cond, lower_expr(&cw.result, scope, ctx)?));
            }
            let otherwise = match else_result {
                Some(e) => Some(Box::new(lower_expr(e, scope, ctx)?)),
                None => None,
            };
            Ok(Expr::Case {
                branches,
                otherwise,
            })
        }
        SqlExpr::Subquery(q) => lower_scalar_subquery(q, scope, ctx),
        SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => Err(FrontendError::Unsupported(
            "EXISTS / IN subquery in expression position".into(),
        )),
        // CAST — the single, typed home for casts (ADR-043 Inv 2). The pgwire translator
        // no longer string-strips cast suffixes, so an explicit `x::T` reaches here. Lower
        // to one typed `Expr::Cast`, IDENTITY-FOLDED to zero-copy when the inner expression
        // already has the target type (the redundant casts the old strip list silently
        // dropped) — using the pure `Expr::result_type()` inference.
        SqlExpr::Cast {
            expr, data_type, ..
        } => {
            let inner = lower_expr(expr, scope, ctx)?;
            let target = sql_type_to_proxima(data_type)?;
            if inner.result_type() == target {
                Ok(inner)
            } else {
                Ok(Expr::Cast {
                    expr: Box::new(inner),
                    ty: target,
                })
            }
        }
        // Special-syntax ANSI functions → their registry FuncCall names, so they
        // answer on the native/Volcano route too (TD-OLAP-5 P1b; previously these
        // AST shapes were Unsupported and only ran via the DataFusion fallback).
        //
        // SUBSTRING(x FROM s [FOR n]) / SUBSTRING(x, s[, n]) → substr(x, s[, n]).
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let mut args = vec![lower_expr(expr, scope, ctx)?];
            match substring_from {
                Some(from) => args.push(lower_expr(from, scope, ctx)?),
                // SUBSTRING(x FOR n) means from the start.
                None => args.push(Expr::Literal {
                    value: ProximaValue::Int64(1),
                    ty: ProximaType::Int64,
                }),
            }
            if let Some(len) = substring_for {
                args.push(lower_expr(len, scope, ctx)?);
            }
            Ok(registry_func_call("substr", args))
        }
        // EXTRACT(field FROM x) → date_part('field', x).
        SqlExpr::Extract { field, expr, .. } => {
            let field_name = field.to_string().to_ascii_lowercase();
            let args = vec![
                Expr::Literal {
                    value: ProximaValue::String(field_name),
                    ty: ProximaType::String,
                },
                lower_expr(expr, scope, ctx)?,
            ];
            Ok(registry_func_call("date_part", args))
        }
        // TRIM([BOTH|LEADING|TRAILING] [chars FROM] x) → trim/btrim/ltrim/rtrim.
        SqlExpr::Trim {
            expr,
            trim_where,
            trim_what,
            trim_characters,
        } => {
            use sqlparser::ast::TrimWhereField;
            let name = match trim_where {
                Some(TrimWhereField::Leading) => "ltrim",
                Some(TrimWhereField::Trailing) => "rtrim",
                Some(TrimWhereField::Both) | None => "btrim",
            };
            let mut args = vec![lower_expr(expr, scope, ctx)?];
            if let Some(what) = trim_what {
                args.push(lower_expr(what, scope, ctx)?);
            } else if let Some(chars) = trim_characters
                && let Some(first) = chars.first()
            {
                args.push(lower_expr(first, scope, ctx)?);
            }
            Ok(registry_func_call(name, args))
        }
        // POSITION(sub IN str) → strpos(str, sub) — note the argument swap.
        SqlExpr::Position { expr, r#in } => {
            let sub = lower_expr(expr, scope, ctx)?;
            let s = lower_expr(r#in, scope, ctx)?;
            Ok(registry_func_call("strpos", vec![s, sub]))
        }
        other => Err(FrontendError::Unsupported(format!(
            "expression {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

/// Build a lowered [`Expr::FuncCall`] against a shared-registry function,
/// resolving the declared return type exactly like `lower_scalar_function`.
fn registry_func_call(name: &str, args: Vec<Expr>) -> Expr {
    let return_ty = proximadb_functions::builtins()
        .lookup_scalar(name)
        .map(|d| d.signature.return_ty.clone())
        .unwrap_or(ProximaType::String);
    Expr::FuncCall {
        name: name.to_string(),
        args,
        return_ty,
    }
}

/// Map a SQL `CAST` target type to the canonical [`ProximaType`]. Covers the standard
/// scalar types (including every suffix the old translator strip list handled, so deleting
/// it does not regress those casts). Postgres byte-width convention: `int4`/`int` →
/// `Int32`, `int8`/`bigint` → `Int64`. Unknown targets are `Unsupported` so the query
/// declines to the existing fallback rather than guessing.
fn sql_type_to_proxima(dt: &SqlDataType) -> Result<ProximaType, FrontendError> {
    use SqlDataType as D;
    Ok(match dt {
        D::Char(_) | D::Varchar(_) | D::Text => ProximaType::String,
        D::TinyInt(_) => ProximaType::Int8,
        D::SmallInt(_) => ProximaType::Int16,
        D::Int(_) | D::Int4(_) | D::Integer(_) => ProximaType::Int32,
        D::Int8(_) | D::BigInt(_) => ProximaType::Int64,
        D::Float(_) | D::Float4 | D::Real => ProximaType::Float32,
        D::Float8 | D::Double(_) => ProximaType::Float64,
        D::Bool | D::Boolean => ProximaType::Boolean,
        D::Date => ProximaType::Date,
        D::Timestamp(_, _) => ProximaType::Timestamp(TimeUnit::Microsecond),
        D::JSON => ProximaType::Json,
        D::JSONB => ProximaType::Jsonb,
        other => {
            return Err(FrontendError::Unsupported(format!("CAST to {other}")));
        }
    })
}

fn lower_binary_op(op: &BinaryOperator) -> Result<BinaryOp, FrontendError> {
    Ok(match op {
        BinaryOperator::Plus => BinaryOp::Plus,
        BinaryOperator::Minus => BinaryOp::Minus,
        BinaryOperator::Multiply => BinaryOp::Mul,
        BinaryOperator::Divide => BinaryOp::Div,
        BinaryOperator::Modulo => BinaryOp::Mod,
        BinaryOperator::Eq => BinaryOp::Eq,
        BinaryOperator::NotEq => BinaryOp::NotEq,
        BinaryOperator::Lt => BinaryOp::Lt,
        BinaryOperator::LtEq => BinaryOp::LtEq,
        BinaryOperator::Gt => BinaryOp::Gt,
        BinaryOperator::GtEq => BinaryOp::GtEq,
        BinaryOperator::And => BinaryOp::And,
        BinaryOperator::Or => BinaryOp::Or,
        BinaryOperator::StringConcat => BinaryOp::Concat,
        other => {
            return Err(FrontendError::Unsupported(format!("binary op {:?}", other)));
        }
    })
}

/// Lower a unary operator. SQL `+` is a no-op (returns
/// `Ok(None)`); callers strip it. Unsupported operators return
/// the matching error.
fn lower_unary_op(op: &UnaryOperator) -> Result<Option<UnaryOp>, FrontendError> {
    Ok(match op {
        UnaryOperator::Not => Some(UnaryOp::Not),
        UnaryOperator::Minus => Some(UnaryOp::Neg),
        UnaryOperator::Plus => None, // identity
        other => {
            return Err(FrontendError::Unsupported(format!("unary op {:?}", other)));
        }
    })
}

fn lower_value(v: &SqlValue) -> Result<ProximaValue, FrontendError> {
    match v {
        SqlValue::Number(s, _) => {
            // Try i64 first, then f64.
            if let Ok(i) = s.parse::<i64>() {
                return Ok(ProximaValue::Int64(i));
            }
            if let Ok(f) = s.parse::<f64>() {
                return Ok(ProximaValue::Float64(f));
            }
            Err(FrontendError::InvalidLiteral(s.clone()))
        }
        SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
            Ok(ProximaValue::String(s.clone()))
        }
        SqlValue::Boolean(b) => Ok(ProximaValue::Boolean(*b)),
        SqlValue::Null => Ok(ProximaValue::Null),
        other => Err(FrontendError::Unsupported(format!("literal {:?}", other))),
    }
}

// =========================================================================
// Projection with aggregates
// =========================================================================
//
// To avoid carrying aggregate placeholders in `Expr` (which would
// create a cycle with the algebra crate), we lower the projection
// in a single pass that recognises aggregate function calls at
// the *projection level* (i.e. as the top-level node of a
// SelectItem), produces a NamedAggregate, and substitutes the
// projection's Expr with a column reference into the aggregate
// slot. This is the SQL standard's pattern for simple aggregate
// projections (`SELECT col, COUNT(*), SUM(x) FROM …`). Complex
// nested cases (aggregates inside arithmetic) are Phase 3.
fn lower_projection_with_aggregates(
    items: &[SelectItem],
    scope: &Scope,
    group_by: &[NamedExpr],
) -> Result<(Vec<NamedExpr>, Vec<NamedAggregate>), FrontendError> {
    let group_count = group_by.len();
    let mut outputs = Vec::new();
    let mut aggregates = Vec::new();
    for item in items {
        let (sql_expr, alias) = match item {
            SelectItem::UnnamedExpr(e) => (e.clone(), None),
            SelectItem::ExprWithAlias { expr, alias } => (expr.clone(), Some(alias.value.clone())),
            SelectItem::Wildcard(_) => {
                for col in scope.all_columns() {
                    outputs.push(NamedExpr {
                        name: col.name.clone(),
                        expr: Expr::column(col.clone()),
                    });
                }
                continue;
            }
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(FrontendError::Unsupported("qualified wildcard".into()));
            }
        };
        match &sql_expr {
            SqlExpr::Function(f) if is_aggregate_function_name(&f.name) => {
                let agg = lower_aggregate_call(f, scope)?;
                let slot = group_count + aggregates.len();
                let name = alias.unwrap_or_else(|| f.name.to_string().to_lowercase());
                let result_ty = agg.result_type();
                aggregates.push(NamedAggregate {
                    name: name.clone(),
                    agg,
                });
                // ColumnRef references the post-aggregate slot.
                // Crucially, the ref's `name` matches the slot's
                // declared name so name-based projection-pushdown
                // rebinding (planner Phase 3 rule) works.
                outputs.push(NamedExpr {
                    name: name.clone(),
                    expr: Expr::column(ColumnRef {
                        name,
                        ordinal: slot,
                        ty: result_ty,
                        nullable: true,
                    }),
                });
            }
            // Aggregate(s) nested inside an arithmetic projection expression
            // (`sum(x)/sum(y)`, `100.0*sum(case…)/sum(y)` — the ratio /
            // market-share shapes, TPC-H Q8/Q14). Hoist each nested aggregate to
            // its own Aggregate slot and rebuild the surrounding expression as a
            // post-aggregate Project expression. (A *bare* top-level aggregate is
            // handled by the arm above; this is the "Phase 3" nested case.)
            _ if expr_contains_aggregate(&sql_expr) => {
                let name = alias
                    .or_else(|| projection_alias_for_expr(&sql_expr))
                    .unwrap_or_else(|| auto_column_name(outputs.len()));
                let output_expr = lower_agg_projection_expr(
                    &sql_expr,
                    scope,
                    group_by,
                    group_count,
                    &mut aggregates,
                )?;
                outputs.push(NamedExpr {
                    name,
                    expr: output_expr,
                });
            }
            _ => {
                // Aggregate-path projection: scalar subqueries here would need to
                // hoist a join under the Aggregate, which is out of scope for MVP
                // (a GROUP BY query with a scalar subquery declines to legacy).
                let expr = lower_expr_sealed(&sql_expr, scope)?;
                let name = alias
                    .or_else(|| projection_alias_for_expr(&sql_expr))
                    .unwrap_or_else(|| auto_column_name(outputs.len()));
                // A non-aggregate projected column must be a GROUP BY key. The
                // Aggregate node places group keys FIRST (ordinals
                // 0..group_count) ahead of the aggregate slots, so rebind the
                // reference to its group-key slot instead of leaving the
                // pre-aggregate ordinal — which would otherwise point into an
                // aggregate column (wrong type/value) once the grouped column
                // isn't the table's first column.
                let ty = expr.result_type();
                let output_expr = match group_by.iter().position(|g| g.expr == expr) {
                    // Reference the group-key slot by the AGGREGATE OUTPUT's
                    // declared name (the group-by key's name), NOT the projection
                    // ALIAS. The alias belongs only on the enclosing `NamedExpr`
                    // (the Project renames the column); the inner `ColumnRef` must
                    // name the slot it indexes so name-based rebind in
                    // `push_projections` resolves it. With the alias on the inner
                    // ref, `SELECT g AS a … ORDER BY a` produced a `ColumnRef`
                    // named `a` over an Aggregate output named `g`, and the
                    // planner rebind failed (`column 'a' not in narrowed schema`,
                    // TD-REL-LOWER-3 q3/q52).
                    Some(slot) => Expr::column(ColumnRef {
                        name: group_by[slot].name.clone(),
                        ordinal: slot,
                        ty,
                        nullable: true,
                    }),
                    None => expr,
                };
                outputs.push(NamedExpr {
                    name,
                    expr: output_expr,
                });
            }
        }
    }
    Ok((outputs, aggregates))
}

/// Lower a projection expression that CONTAINS one or more aggregate calls nested
/// inside arithmetic (the "aggregate ratio / market-share" shapes — TPC-H Q8
/// `sum(case…)/sum(vol)`, Q14 `100.0*sum(case…)/sum(…)`). Each nested aggregate is
/// extracted into its own Aggregate slot (appended to `aggregates`) and replaced
/// by a `ColumnRef` to that slot; a sub-expression with no aggregate is lowered
/// directly (a scalar over group keys / literals) with its group-key column refs
/// remapped to the post-aggregate group slots. The result is an expression over
/// the post-aggregate schema, suitable as a `Project` output above the `Aggregate`.
///
/// Handles the arithmetic spine (`BinaryOp` / `UnaryOp` / parentheses) that carries
/// the aggregates. An aggregate nested inside a shape not covered here (CASE/CAST/
/// scalar-function at the projection top level) still declines to legacy, exactly
/// as before — no regression, just not yet newly supported.
fn lower_agg_projection_expr(
    sql: &SqlExpr,
    scope: &Scope,
    group_by: &[NamedExpr],
    group_count: usize,
    aggregates: &mut Vec<NamedAggregate>,
) -> Result<Expr, FrontendError> {
    // A subtree with no aggregate is a scalar over group keys / literals: lower it
    // as-is, then point any group-key column at its post-aggregate slot (group keys
    // occupy ordinals 0..group_count ahead of the aggregate slots).
    if !expr_contains_aggregate(sql) {
        let lowered = lower_expr_sealed(sql, scope)?;
        return Ok(remap_group_key_refs(lowered, group_by));
    }
    match sql {
        SqlExpr::Nested(inner) => {
            lower_agg_projection_expr(inner, scope, group_by, group_count, aggregates)
        }
        SqlExpr::Function(f) if is_aggregate_function_name(&f.name) => {
            let agg = lower_aggregate_call(f, scope)?;
            let slot = group_count + aggregates.len();
            let ty = agg.result_type();
            // Internal, unique slot name so the post-aggregate scope and the
            // planner's name-based projection-pushdown rebind stay consistent
            // (`ColumnRef.name` must equal the Aggregate output slot's name).
            let name = format!("__agg_{slot}");
            aggregates.push(NamedAggregate {
                name: name.clone(),
                agg,
            });
            Ok(Expr::column(ColumnRef {
                name,
                ordinal: slot,
                ty,
                nullable: true,
            }))
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let l = lower_agg_projection_expr(left, scope, group_by, group_count, aggregates)?;
            let r = lower_agg_projection_expr(right, scope, group_by, group_count, aggregates)?;
            Ok(Expr::BinaryOp {
                op: lower_binary_op(op)?,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => {
            let inner = lower_agg_projection_expr(expr, scope, group_by, group_count, aggregates)?;
            match lower_unary_op(op)? {
                Some(u) => Ok(Expr::UnaryOp {
                    op: u,
                    expr: Box::new(inner),
                }),
                None => Ok(inner), // unary plus is identity
            }
        }
        _ => Err(FrontendError::Unsupported(
            "aggregate nested inside an unsupported projection expression".into(),
        )),
    }
}

/// Count the aggregate function calls in an expression, over the SAME shapes
/// [`lower_having`] extracts (arithmetic spine + bare aggregate; NOT descending
/// into subqueries — their inner aggregates are separate). Used to RESERVE the
/// HAVING aggregate slot space ahead of any hoisted scalar-subquery column, so a
/// subquery hoisted before a later-extracted aggregate never collides with it.
fn count_aggregates(sql: &SqlExpr) -> usize {
    match sql {
        SqlExpr::Function(f) if is_aggregate_function_name(&f.name) => 1,
        SqlExpr::Nested(inner) | SqlExpr::UnaryOp { expr: inner, .. } => count_aggregates(inner),
        SqlExpr::BinaryOp { left, right, .. } => count_aggregates(left) + count_aggregates(right),
        _ => 0,
    }
}

/// True if an expression carries a subquery (scalar `(SELECT …)`, `EXISTS`, or
/// `IN (SELECT …)`) in its arithmetic/boolean spine — gates the HAVING lowering
/// onto the post-aggregate `Filter` path.
fn expr_contains_subquery(sql: &SqlExpr) -> bool {
    match sql {
        SqlExpr::Subquery(_) | SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => true,
        SqlExpr::Nested(e) | SqlExpr::UnaryOp { expr: e, .. } => expr_contains_subquery(e),
        SqlExpr::BinaryOp { left, right, .. } => {
            expr_contains_subquery(left) || expr_contains_subquery(right)
        }
        _ => false,
    }
}

/// Lower a HAVING predicate that may reference aggregate calls and/or an
/// (uncorrelated) scalar subquery — TPC-H Q11:
/// `HAVING sum(x) > (SELECT sum(x)*k FROM …)`. Mirrors [`lower_agg_projection_expr`]'s
/// extraction over the arithmetic/boolean spine, but ALSO hoists scalar subqueries:
/// - an aggregate call is extracted into `aggregates` (post-aggregate slot,
///   `__agg_<slot>`) and replaced by a `ColumnRef` to that slot;
/// - a scalar subquery is lowered + hoisted onto `ctx` (a `LEFT JOIN ON TRUE`
///   above the Aggregate, via the shared [`lower_scalar_subquery`]) and replaced by
///   a `ColumnRef` to the hoisted column;
/// - an aggregate-free / subquery-free leaf is a scalar over group keys / literals
///   (`lower_expr_sealed` + [`remap_group_key_refs`]).
///
/// The result is a boolean expression over the combined
/// `[group keys | aggregates | hoisted scalars]` schema, applied as a
/// post-aggregate `Filter`. The caller RESERVES the aggregate slot space
/// (`ctx.base_width = group_count + aggregates.len() + count_aggregates(having)`)
/// so hoisted-scalar ordinals never overlap a later-extracted aggregate slot.
fn lower_having(
    sql: &SqlExpr,
    scope: &Scope,
    group_by: &[NamedExpr],
    group_count: usize,
    aggregates: &mut Vec<NamedAggregate>,
    ctx: &mut LoweringCtx,
) -> Result<Expr, FrontendError> {
    match sql {
        SqlExpr::Nested(inner) => {
            lower_having(inner, scope, group_by, group_count, aggregates, ctx)
        }
        // Uncorrelated → hoisted `LEFT JOIN ON TRUE` over `AssertMaxOneRow`;
        // correlated → decorrelated. Reuses the shared scalar-subquery helper.
        SqlExpr::Subquery(q) => lower_scalar_subquery(q, scope, ctx),
        SqlExpr::Function(f) if is_aggregate_function_name(&f.name) => {
            let agg = lower_aggregate_call(f, scope)?;
            let slot = group_count + aggregates.len();
            let ty = agg.result_type();
            let name = format!("__agg_{slot}");
            aggregates.push(NamedAggregate {
                name: name.clone(),
                agg,
            });
            Ok(Expr::column(ColumnRef {
                name,
                ordinal: slot,
                ty,
                nullable: true,
            }))
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let l = lower_having(left, scope, group_by, group_count, aggregates, ctx)?;
            let r = lower_having(right, scope, group_by, group_count, aggregates, ctx)?;
            Ok(Expr::BinaryOp {
                op: lower_binary_op(op)?,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => {
            let inner = lower_having(expr, scope, group_by, group_count, aggregates, ctx)?;
            match lower_unary_op(op)? {
                Some(u) => Ok(Expr::UnaryOp {
                    op: u,
                    expr: Box::new(inner),
                }),
                None => Ok(inner),
            }
        }
        // A subtree with neither aggregate nor subquery: a scalar over group keys /
        // literals — lower directly, remapping group-key refs to their post-agg slot.
        _ => {
            let lowered = lower_expr_sealed(sql, scope)?;
            Ok(remap_group_key_refs(lowered, group_by))
        }
    }
}

/// Remap every `Expr::Column` in `expr` that references a GROUP BY key to that
/// key's post-aggregate slot (`ordinal` = its position in `group_by`, `name` =
/// the group-key's declared name). Non-group columns and literals pass through
/// unchanged. Used for the aggregate-free sub-expressions of a nested-aggregate
/// projection (e.g. the `yr` in `sum(vol) + yr`), so they index the Aggregate
/// output rather than a stale pre-aggregate ordinal.
fn remap_group_key_refs(expr: Expr, group_by: &[NamedExpr]) -> Expr {
    match &expr {
        Expr::Column(c) => {
            if let Some(slot) = group_by.iter().position(|g| g.expr == expr) {
                return Expr::column(ColumnRef {
                    name: group_by[slot].name.clone(),
                    ordinal: slot,
                    ty: c.ty.clone(),
                    nullable: c.nullable,
                });
            }
            expr
        }
        _ => map_expr_children(expr, &mut |child| remap_group_key_refs(child, group_by)),
    }
}

/// Structural map over an expression's immediate child expressions, rebuilding
/// the node with each child passed through `f`. Leaves (Column/Literal) are
/// returned unchanged. Keeps recursive rewrites (e.g. [`remap_group_key_refs`])
/// from having to enumerate every operator variant at each call site.
fn map_expr_children(expr: Expr, f: &mut dyn FnMut(Expr) -> Expr) -> Expr {
    match expr {
        Expr::Column(_) | Expr::Literal { .. } => expr,
        Expr::Cast { expr, ty } => Expr::Cast {
            expr: Box::new(f(*expr)),
            ty,
        },
        Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
            op,
            left: Box::new(f(*left)),
            right: Box::new(f(*right)),
        },
        Expr::UnaryOp { op, expr } => Expr::UnaryOp {
            op,
            expr: Box::new(f(*expr)),
        },
        Expr::IsNull { expr, not } => Expr::IsNull {
            expr: Box::new(f(*expr)),
            not,
        },
        Expr::Between {
            expr,
            low,
            high,
            not,
        } => Expr::Between {
            expr: Box::new(f(*expr)),
            low: Box::new(f(*low)),
            high: Box::new(f(*high)),
            not,
        },
        Expr::In { expr, list, not } => Expr::In {
            expr: Box::new(f(*expr)),
            list: list.into_iter().map(&mut *f).collect(),
            not,
        },
        Expr::Like {
            expr,
            pattern,
            not,
            case_insensitive,
        } => Expr::Like {
            expr: Box::new(f(*expr)),
            pattern: Box::new(f(*pattern)),
            not,
            case_insensitive,
        },
        Expr::Case {
            branches,
            otherwise,
        } => Expr::Case {
            branches: branches.into_iter().map(|(w, t)| (f(w), f(t))).collect(),
            otherwise: otherwise.map(|e| Box::new(f(*e))),
        },
        Expr::Coalesce(items) => Expr::Coalesce(items.into_iter().map(&mut *f).collect()),
        Expr::NullIf { left, right } => Expr::NullIf {
            left: Box::new(f(*left)),
            right: Box::new(f(*right)),
        },
        Expr::FuncCall {
            name,
            args,
            return_ty,
        } => Expr::FuncCall {
            name,
            args: args.into_iter().map(&mut *f).collect(),
            return_ty,
        },
    }
}

/// Build a [`Scope`] representing the columns visible AFTER an
/// Aggregate node: group_by columns first (in declared order),
/// followed by aggregate-result slots. Used to lower HAVING
/// expressions over the post-aggregate schema.
/// followed by aggregate-result slots. Used to lower HAVING
/// expressions over the post-aggregate schema.
fn post_aggregate_scope(group_by: &[NamedExpr], aggregates: &[NamedAggregate]) -> Scope {
    let mut columns = Vec::with_capacity(group_by.len() + aggregates.len());
    let mut ordinal = 0;
    for g in group_by {
        columns.push(ScopeColumn {
            table: String::new(),
            column: ColumnInfo {
                name: g.name.clone(),
                ty: g.expr.result_type(),
                nullable: true,
            },
            ordinal,
        });
        ordinal += 1;
    }
    for a in aggregates {
        columns.push(ScopeColumn {
            table: String::new(),
            column: ColumnInfo {
                name: a.name.clone(),
                ty: a.agg.result_type(),
                nullable: !matches!(a.agg, AggregateExpr::Count { .. }),
            },
            ordinal,
        });
        ordinal += 1;
    }
    Scope { columns }
}

/// Lower a non-aggregate SQL function call `f(args)` to [`Expr::FuncCall`].
///
/// Reaching here means the function is in **scalar position** (aggregate calls are
/// handled by [`lower_aggregate_call`] in the SELECT/HAVING aggregate path), so a bare
/// aggregate name here is a misuse and is rejected. Scalar functions resolve their
/// `return_ty` from the shared builtin registry — the *same* registry the Volcano
/// executor dispatches through, so definition and lowering stay single-sourced. Unknown
/// names still lower (the DataFusion path may serve them via its own function set, and the
/// Volcano path raises a clear `UnknownFunction` at execution rather than at parse).
fn lower_scalar_function(
    f: &sqlparser::ast::Function,
    scope: &Scope,
    ctx: &mut LoweringCtx,
) -> Result<Expr, FrontendError> {
    // TD-187: a windowed call (`… OVER (…)`) is NOT a plain scalar function —
    // lowering it here would silently DROP the window spec (partition/order/
    // frame) and produce a plain `Expr::FuncCall`, which the DataFusion-side
    // fallback either mis-evaluates (for a name that happens to also be a
    // registered scalar builtin) or declines on (for LAG/LEAD/RANK/etc.,
    // which aren't registered as scalars) — decline unconditionally instead,
    // every window function needs DataFusion's real window executor, and
    // `ctx.sql()` reaches it once this frontend backs off.
    if f.over.is_some() {
        return Err(FrontendError::Unsupported(
            "window function (requires DataFusion's window executor)".into(),
        ));
    }

    let raw_name = f.name.to_string();
    if matches!(
        raw_name.to_uppercase().as_str(),
        "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"
    ) {
        return Err(FrontendError::Unsupported(
            "aggregate function in non-aggregate position".into(),
        ));
    }

    let args = match &f.args {
        FunctionArguments::List(list) => list
            .args
            .iter()
            .map(|fa| match fa {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => lower_expr(e, scope, ctx),
                _ => Err(FrontendError::Unsupported(
                    "unsupported scalar function argument".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        FunctionArguments::None => Vec::new(),
        FunctionArguments::Subquery(_) => {
            return Err(FrontendError::Unsupported(
                "subquery as function argument".into(),
            ));
        }
    };

    // COALESCE / NULLIF have NULL-aware semantics (short-circuit / equality-to-NULL)
    // that a generic eager `FuncCall` can't express, so lower them to their dedicated
    // `Expr` variants (the evaluator handles the three-valued logic natively).
    match raw_name.to_uppercase().as_str() {
        "COALESCE" => return Ok(Expr::Coalesce(args)),
        "NULLIF" => {
            let [left, right] = <[Expr; 2]>::try_from(args).map_err(|_| {
                FrontendError::Unsupported("NULLIF requires exactly 2 arguments".into())
            })?;
            return Ok(Expr::NullIf {
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        _ => {}
    }

    // Resolve the declared return type from the shared registry; unknown functions get a
    // permissive placeholder (the evaluator ignores `return_ty`, and the DataFusion path
    // re-derives types from Arrow — see `Expr::FuncCall` eval).
    let return_ty = proximadb_functions::builtins()
        .lookup_scalar(&raw_name)
        .map(|d| d.signature.return_ty.clone())
        .unwrap_or(ProximaType::String);

    Ok(Expr::FuncCall {
        name: raw_name.to_ascii_lowercase(),
        args,
        return_ty,
    })
}

fn lower_aggregate_call(
    f: &sqlparser::ast::Function,
    scope: &Scope,
) -> Result<AggregateExpr, FrontendError> {
    // TD-187: an aggregate-named function used as a WINDOW aggregate
    // (`avg(x) OVER (...)`) is not a plain GROUP BY aggregate — lowering it
    // here would drop the window spec and (best case) fail an unrelated
    // scope check, or (worst case) silently produce a whole-table aggregate
    // instead of a per-partition one. Decline unconditionally so it falls
    // through to DataFusion's real window executor, same reasoning as the
    // scalar-function guard in `lower_scalar_function`.
    if f.over.is_some() {
        return Err(FrontendError::Unsupported(
            "window aggregate (requires DataFusion's window executor)".into(),
        ));
    }
    let name = f.name.to_string().to_uppercase();
    // Detect DISTINCT and the argument(s).
    let (args, distinct) = match &f.args {
        FunctionArguments::List(list) => {
            let distinct = matches!(
                list.duplicate_treatment,
                Some(sqlparser::ast::DuplicateTreatment::Distinct)
            );
            let args = list
                .args
                .iter()
                .map(|fa| match fa {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => {
                        Ok(Some(lower_expr_sealed(e, scope)?))
                    }
                    FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => Ok(None),
                    FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => Ok(None),
                    FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
                        Err(FrontendError::Unsupported("named function arg".into()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            (args, distinct)
        }
        FunctionArguments::None => (Vec::new(), false),
        FunctionArguments::Subquery(_) => {
            return Err(FrontendError::Unsupported("subquery in aggregate".into()));
        }
    };
    Ok(match name.as_str() {
        "COUNT" => match args.as_slice() {
            [None] => AggregateExpr::Count {
                arg: None,
                distinct: false,
            },
            [Some(e)] => AggregateExpr::Count {
                arg: Some(e.clone()),
                distinct,
            },
            [] => AggregateExpr::Count {
                arg: None,
                distinct: false,
            },
            _ => {
                return Err(FrontendError::Unsupported(
                    "COUNT with multiple args".into(),
                ));
            }
        },
        "SUM" => match args.as_slice() {
            [Some(e)] => AggregateExpr::Sum {
                arg: e.clone(),
                distinct,
            },
            _ => {
                return Err(FrontendError::Unsupported(
                    "SUM with wrong arg count".into(),
                ));
            }
        },
        "AVG" => match args.as_slice() {
            [Some(e)] => AggregateExpr::Avg {
                arg: e.clone(),
                distinct,
            },
            _ => {
                return Err(FrontendError::Unsupported(
                    "AVG with wrong arg count".into(),
                ));
            }
        },
        "MIN" => match args.as_slice() {
            [Some(e)] => AggregateExpr::Min { arg: e.clone() },
            _ => {
                return Err(FrontendError::Unsupported(
                    "MIN with wrong arg count".into(),
                ));
            }
        },
        "MAX" => match args.as_slice() {
            [Some(e)] => AggregateExpr::Max { arg: e.clone() },
            _ => {
                return Err(FrontendError::Unsupported(
                    "MAX with wrong arg count".into(),
                ));
            }
        },
        // A non-builtin aggregate: resolve it against the shared registry (the same one the
        // Volcano executor accumulates through, and the DataFusion AggregateUDF adapter binds).
        // Unknown names still error clearly.
        other => {
            if let Some(def) = proximadb_functions::builtins().lookup_aggregate(other) {
                let lowered = args
                    .into_iter()
                    .map(|a| {
                        a.ok_or_else(|| {
                            FrontendError::Unsupported(
                                "wildcard argument in custom aggregate".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(AggregateExpr::Custom {
                    name: other.to_ascii_lowercase(),
                    args: lowered,
                    distinct,
                    return_ty: def.signature.return_ty.clone(),
                });
            }
            return Err(FrontendError::Unsupported(format!(
                "aggregate function {}",
                other
            )));
        }
    })
}

// =========================================================================
// ORDER BY + LIMIT
// =========================================================================

fn apply_order_by(
    plan: LogicalNode,
    order_by: &sqlparser::ast::OrderBy,
    _catalog: &dyn CatalogLookup,
) -> Result<LogicalNode, FrontendError> {
    let exprs = match &order_by.kind {
        OrderByKind::Expressions(e) => e,
        OrderByKind::All(_) => {
            return Err(FrontendError::Unsupported("ORDER BY ALL".into()));
        }
    };
    let scope = Scope::from_schema(&plan.output_schema());
    let mut keys = Vec::with_capacity(exprs.len());
    for o in exprs {
        // ORDER BY resolves against the post-projection scope, which carries no
        // table qualifiers (`Scope::from_schema` — the Project erased them). A
        // qualified sort key (`ORDER BY t.col` / `alias.col`) that names a
        // projected output column therefore resolves by its bare column name
        // (ANSI: sort keys designate output columns). Without this fallback the
        // whole query declines to the legacy single-table path (TD-REL-LOWER-1).
        let expr = match (lower_expr_sealed(&o.expr, &scope), &o.expr) {
            (Ok(e), _) => e,
            (Err(FrontendError::ColumnNotFound(_)), SqlExpr::CompoundIdentifier(parts))
                if !parts.is_empty() =>
            {
                Expr::column(scope.resolve_unqualified(
                    &parts[parts.len() - 1].value,
                    parts[parts.len() - 1].quote_style.is_some(),
                )?)
            }
            (Err(e), _) => return Err(e),
        };
        let descending = matches!(o.options.asc, Some(false));
        let nulls_first = match o.options.nulls_first {
            Some(b) => b,
            None => descending, // ASC → NULLS LAST, DESC → NULLS FIRST
        };
        keys.push(SortKey {
            expr,
            descending,
            nulls_first,
        });
    }
    Ok(LogicalNode::Sort {
        input: Box::new(plan),
        keys,
    })
}

fn apply_limit(plan: LogicalNode, lc: &LimitClause) -> Result<LogicalNode, FrontendError> {
    match lc {
        LimitClause::LimitOffset {
            limit,
            offset,
            limit_by,
        } => {
            if !limit_by.is_empty() {
                return Err(FrontendError::Unsupported("LIMIT BY".into()));
            }
            let limit_val = match limit {
                Some(SqlExpr::Value(ValueWithSpan {
                    value: SqlValue::Number(s, _),
                    ..
                })) => Some(
                    s.parse::<u64>()
                        .map_err(|e| FrontendError::InvalidLiteral(format!("LIMIT: {e}")))?,
                ),
                Some(_) => {
                    return Err(FrontendError::Unsupported("LIMIT with non-literal".into()));
                }
                None => None,
            };
            let offset_val = match offset {
                Some(off) => match &off.value {
                    SqlExpr::Value(ValueWithSpan {
                        value: SqlValue::Number(s, _),
                        ..
                    }) => s
                        .parse::<u64>()
                        .map_err(|e| FrontendError::InvalidLiteral(format!("OFFSET: {e}")))?,
                    _ => {
                        return Err(FrontendError::Unsupported("OFFSET with non-literal".into()));
                    }
                },
                None => 0,
            };
            Ok(LogicalNode::Limit {
                input: Box::new(plan),
                limit: limit_val,
                offset: offset_val,
            })
        }
        LimitClause::OffsetCommaLimit { offset, limit } => {
            let offset_val = parse_literal_u64(offset)?;
            let limit_val = parse_literal_u64(limit)?;
            Ok(LogicalNode::Limit {
                input: Box::new(plan),
                limit: Some(limit_val),
                offset: offset_val,
            })
        }
    }
}

fn parse_literal_u64(e: &SqlExpr) -> Result<u64, FrontendError> {
    match e {
        SqlExpr::Value(ValueWithSpan {
            value: SqlValue::Number(s, _),
            ..
        }) => s
            .parse::<u64>()
            .map_err(|err| FrontendError::InvalidLiteral(err.to_string())),
        _ => Err(FrontendError::Unsupported(
            "LIMIT/OFFSET with non-literal".into(),
        )),
    }
}

// =========================================================================
// Scope: resolves identifier references to column ordinals
// =========================================================================

/// Tracks the columns in the current relation. Builds an ordinal
/// index from `(table_alias?, column_name)` to `ColumnRef`.
#[derive(Debug, Clone)]
struct Scope {
    columns: Vec<ScopeColumn>,
}

#[derive(Debug, Clone)]
struct ScopeColumn {
    table: String,
    column: ColumnInfo,
    /// Position in the combined relation (used as the `ordinal`
    /// in resulting `ColumnRef`s).
    ordinal: usize,
}

impl Scope {
    fn empty() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    fn from_table(table_alias: &str, schema: &RelationalSchema) -> Self {
        Self {
            columns: schema
                .columns
                .iter()
                .enumerate()
                .map(|(i, c)| ScopeColumn {
                    table: table_alias.to_string(),
                    column: c.clone(),
                    ordinal: i,
                })
                .collect(),
        }
    }

    fn from_schema(schema: &RelationalSchema) -> Self {
        // No table alias context (e.g. post-projection scope).
        Self {
            columns: schema
                .columns
                .iter()
                .enumerate()
                .map(|(i, c)| ScopeColumn {
                    table: String::new(),
                    column: c.clone(),
                    ordinal: i,
                })
                .collect(),
        }
    }

    fn concat(&self, other: &Scope) -> Scope {
        let mut combined = self.clone();
        let offset = combined.columns.len();
        for sc in &other.columns {
            combined.columns.push(ScopeColumn {
                table: sc.table.clone(),
                column: sc.column.clone(),
                ordinal: sc.ordinal + offset,
            });
        }
        combined
    }

    fn all_columns(&self) -> Vec<ColumnRef> {
        self.columns
            .iter()
            .map(|sc| ColumnRef {
                name: sc.column.name.clone(),
                ordinal: sc.ordinal,
                ty: sc.column.ty.clone(),
                nullable: sc.column.nullable,
            })
            .collect()
    }

    /// Resolve an unqualified column reference. Exact (declared-case) match
    /// wins; when the reference was UNQUOTED and nothing matched exactly, a
    /// unique case-insensitive match resolves (TD-OLAP-18 ANSI folding).
    fn resolve_unqualified(&self, name: &str, quoted: bool) -> Result<ColumnRef, FrontendError> {
        match self.resolve_unqualified_exact(name) {
            Err(FrontendError::ColumnNotFound(_)) if !quoted && ident_case_fold_enabled() => {
                self.resolve_unqualified_folded(name)
            }
            other => other,
        }
    }

    /// Case-insensitive resolution for an unquoted reference that had no exact
    /// match. Distinct case-variant column names (`Foo` AND `foo` in scope) are
    /// ambiguous — resolution stays strict rather than guessing.
    fn resolve_unqualified_folded(&self, name: &str) -> Result<ColumnRef, FrontendError> {
        let mut declared: Option<&str> = None;
        for c in &self.columns {
            if c.column.name.eq_ignore_ascii_case(name) {
                match declared {
                    Some(d) if d != c.column.name => {
                        return Err(FrontendError::AmbiguousColumn(name.into()));
                    }
                    _ => declared = Some(&c.column.name),
                }
            }
        }
        match declared {
            Some(d) => self.resolve_unqualified_exact(d),
            None => Err(FrontendError::ColumnNotFound(name.into())),
        }
    }

    fn resolve_unqualified_exact(&self, name: &str) -> Result<ColumnRef, FrontendError> {
        let mut hits: Vec<&ScopeColumn> = self
            .columns
            .iter()
            .filter(|c| c.column.name == name)
            .collect();
        match hits.len() {
            0 => Err(FrontendError::ColumnNotFound(name.into())),
            1 => Ok(ColumnRef {
                name: hits[0].column.name.clone(),
                ordinal: hits[0].ordinal,
                ty: hits[0].column.ty.clone(),
                nullable: hits[0].column.nullable,
            }),
            _ => {
                // Multiple hits — ambiguous only if the names AND
                // tables differ; if they all come from the same
                // table, return the first (post-projection scope).
                hits.sort_by_key(|c| &c.table);
                hits.dedup_by_key(|c| c.table.clone());
                if hits.len() == 1 {
                    Ok(ColumnRef {
                        name: hits[0].column.name.clone(),
                        ordinal: hits[0].ordinal,
                        ty: hits[0].column.ty.clone(),
                        nullable: hits[0].column.nullable,
                    })
                } else {
                    Err(FrontendError::AmbiguousColumn(name.into()))
                }
            }
        }
    }

    /// Resolve a qualified `table.column` reference. Exact match wins; when the
    /// COLUMN part was unquoted and nothing matched exactly, the pair resolves
    /// case-insensitively if unique (TD-OLAP-18 ANSI folding — the table/alias
    /// part always folds, matching `normalize_table_key`'s table semantics).
    fn resolve_qualified(
        &self,
        table: &str,
        column: &str,
        column_quoted: bool,
    ) -> Result<ColumnRef, FrontendError> {
        if let Some(c) = self
            .columns
            .iter()
            .find(|c| c.table == table && c.column.name == column)
        {
            return Ok(ColumnRef {
                name: c.column.name.clone(),
                ordinal: c.ordinal,
                ty: c.column.ty.clone(),
                nullable: c.column.nullable,
            });
        }
        if !column_quoted && ident_case_fold_enabled() {
            let mut hit: Option<&ScopeColumn> = None;
            for c in &self.columns {
                if c.table.eq_ignore_ascii_case(table) && c.column.name.eq_ignore_ascii_case(column)
                {
                    match hit {
                        Some(h) if h.column.name != c.column.name => {
                            return Err(FrontendError::AmbiguousColumn(format!(
                                "{table}.{column}"
                            )));
                        }
                        _ => hit = Some(c),
                    }
                }
            }
            if let Some(c) = hit {
                return Ok(ColumnRef {
                    name: c.column.name.clone(),
                    ordinal: c.ordinal,
                    ty: c.column.ty.clone(),
                    nullable: c.column.nullable,
                });
            }
        }
        Err(FrontendError::ColumnNotFound(format!("{table}.{column}")))
    }
}

// =========================================================================
// CREATE FUNCTION body lowering (F5)
// =========================================================================

/// Lower a SQL-language function body — a scalar expression over the function's parameters —
/// into an engine-neutral [`Expr`]. Each parameter is exposed as a column (parameter `i` →
/// ordinal `i`) so the body references it by name; the resulting `Expr` feeds
/// `proximadb_functions::sql_bodied_scalar` to register a `CREATE FUNCTION` as a registry
/// function that runs on both engines (F5).
///
/// Uses the SAME `lower_expr` the SELECT path uses, so a function body supports the full scalar
/// grammar (arithmetic, builtins, CASE, …). Parse errors and references to unknown parameters
/// return a clear [`FrontendError`].
pub fn lower_function_body(
    body_sql: &str,
    params: &[(String, ProximaType)],
) -> Result<Expr, FrontendError> {
    let columns: Vec<ColumnInfo> = params
        .iter()
        .map(|(name, ty)| ColumnInfo::new(name.clone(), ty.clone(), true))
        .collect();
    let schema = RelationalSchema::new(columns);
    let scope = Scope::from_schema(&schema);

    let mut parser = Parser::new(&GenericDialect {})
        .try_with_sql(body_sql)
        .map_err(|e| FrontendError::Parse(format!("function body: {e}")))?;
    let expr = parser
        .parse_expr()
        .map_err(|e| FrontendError::Parse(format!("function body: {e}")))?;
    lower_expr_sealed(&expr, &scope)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn users_schema() -> RelationalSchema {
        RelationalSchema::new(vec![
            ColumnInfo::new("id", ProximaType::Int64, false),
            ColumnInfo::new("name", ProximaType::String, true),
            ColumnInfo::new("age", ProximaType::Int32, true),
        ])
    }

    fn orders_schema() -> RelationalSchema {
        RelationalSchema::new(vec![
            ColumnInfo::new("oid", ProximaType::Int64, false),
            ColumnInfo::new("uid", ProximaType::Int64, false),
            ColumnInfo::new("total", ProximaType::Float64, false),
        ])
    }

    fn catalog() -> InMemoryCatalog {
        let mut c = InMemoryCatalog::new();
        c.register("users", users_schema());
        c.register("orders", orders_schema());
        c
    }

    fn lower(sql: &str) -> LogicalNode {
        lower_sql(sql, &catalog()).unwrap_or_else(|e| panic!("lower failed: {e:?}"))
    }

    #[test]
    fn correlated_scalar_aggregate_allows_arithmetic_wrapper() {
        // TD-REL-LOWER-8 (TPC-H Q17): a correlated scalar aggregate wrapped in
        // scalar arithmetic (`0.5 * avg(total)`) must decorrelate — previously
        // only a BARE aggregate lowered; the `k *` wrapper declined.
        let wrapped = lower_sql(
            "SELECT id FROM users WHERE age < \
             (SELECT 0.5 * avg(total) FROM orders WHERE orders.uid = users.id)",
            &catalog(),
        );
        assert!(
            wrapped.is_ok(),
            "wrapped correlated scalar agg must lower: {wrapped:?}"
        );

        // No regression: the bare form still lowers.
        assert!(
            lower_sql(
                "SELECT id FROM users WHERE age < \
                 (SELECT avg(total) FROM orders WHERE orders.uid = users.id)",
                &catalog()
            )
            .is_ok()
        );

        // A non-aggregate scalar subquery projection is still not this path
        // (declines or takes the uncorrelated path) — the wrapper must contain
        // exactly one aggregate. `count` over an empty correlated group is still
        // declined (NULL-vs-0 bug).
        assert!(
            lower_sql(
                "SELECT id FROM users WHERE age < \
                 (SELECT count(total) FROM orders WHERE orders.uid = users.id)",
                &catalog()
            )
            .is_err(),
            "correlated COUNT still declines (empty-group NULL-vs-0)"
        );
    }

    fn parse_where(sql: &str) -> Option<SqlExpr> {
        let stmts = Parser::parse_sql(&GenericDialect {}, sql).expect("parse");
        match &stmts[0] {
            Statement::Query(q) => match &*q.body {
                SetExpr::Select(s) => s.selection.clone(),
                _ => None,
            },
            _ => None,
        }
    }

    #[test]
    fn comma_join_order_places_hub_between_disconnected_tables() {
        // TD-REL-LOWER-7 (Q9 shape): `a` and `b` share no direct equi-key; both
        // join only to the hub. FROM order [a, b, hub] would cross-join a×b, so
        // the reorder must move the hub between them → [a, hub, b], every
        // adjacency sharing an equi-key.
        let schema = |cols: &[&str]| {
            RelationalSchema::new(
                cols.iter()
                    .map(|c| ColumnInfo::new(*c, ProximaType::Int64, true))
                    .collect(),
            )
        };
        let a = Scope::from_table("a", &schema(&["a_id", "a_href"]));
        let b = Scope::from_table("b", &schema(&["b_id", "b_href"]));
        let hub = Scope::from_table("hub", &schema(&["h_id", "h_val"]));

        let scopes = [&a, &b, &hub];
        let w = parse_where(
            "SELECT 1 FROM a, b, hub WHERE a.a_href = hub.h_id AND b.b_href = hub.h_id",
        );
        assert_eq!(
            comma_join_order(&scopes, w.as_ref()),
            vec![0, 2, 1],
            "hub moves between the two otherwise-disconnected tables"
        );

        // An already fully-connected FROM order is reproduced unchanged.
        let scopes2 = [&a, &hub, &b];
        let w2 = parse_where(
            "SELECT 1 FROM a, hub, b WHERE a.a_href = hub.h_id AND b.b_href = hub.h_id",
        );
        assert_eq!(
            comma_join_order(&scopes2, w2.as_ref()),
            vec![0, 1, 2],
            "already-connected order is a no-op"
        );

        // No WHERE ⇒ identity (nothing to connect on).
        assert_eq!(comma_join_order(&scopes, None), vec![0, 1, 2]);
    }

    /// A `DATE 'YYYY-MM-DD'` typed literal lowers to the canonical
    /// `ProximaValue::Date` (days since the Unix epoch) — the same encoding the
    /// write path stores — so `col < DATE '…'` compares correctly against a
    /// Date32 column. Was `Unsupported("typed string literal")` before, which
    /// declined ~8 TPC-H native queries (Q1/Q3/Q5/Q10/Q12/…) at lowering.
    #[test]
    fn typed_date_literal_lowers_to_canonical_days() {
        let mut cat = InMemoryCatalog::new();
        cat.register(
            "events",
            RelationalSchema::new(vec![
                ColumnInfo::new("id", ProximaType::Int64, false),
                ColumnInfo::new("ts", ProximaType::Date, true),
            ]),
        );
        let plan = lower_sql("SELECT id FROM events WHERE ts < DATE '1995-03-15'", &cat)
            .expect("DATE literal must lower");
        // Reach the Filter predicate and confirm the RHS literal.
        fn find_date_literal(node: &LogicalNode) -> Option<i32> {
            fn in_expr(e: &Expr) -> Option<i32> {
                match e {
                    Expr::Literal {
                        value: ProximaValue::Date(d),
                        ..
                    } => Some(*d),
                    Expr::BinaryOp { left, right, .. } => in_expr(left).or_else(|| in_expr(right)),
                    Expr::Cast { expr, .. } | Expr::UnaryOp { expr, .. } => in_expr(expr),
                    _ => None,
                }
            }
            match node {
                LogicalNode::Filter { predicate, .. } => in_expr(predicate),
                LogicalNode::Project { input, .. } => find_date_literal(input),
                _ => None,
            }
        }
        assert_eq!(
            find_date_literal(&plan),
            Some(9204),
            "DATE '1995-03-15' must be 9204 days since epoch"
        );
        // A malformed date fails loud, not silently.
        assert!(matches!(
            lower_sql("SELECT id FROM events WHERE ts < DATE '1995-13-99'", &cat),
            Err(FrontendError::InvalidLiteral(_))
        ));
        // A non-DATE typed literal is still declined with its type named.
        assert!(matches!(
            lower_sql(
                "SELECT id FROM events WHERE ts < TIMESTAMP '1995-03-15 00:00:00'",
                &cat
            ),
            Err(FrontendError::Unsupported(_))
        ));
    }

    // --- Semi/Anti via uncorrelated IN / EXISTS / NOT EXISTS subqueries --------

    /// Strip the outer `Project` to reach the join/filter the WHERE produced.
    fn under_project(plan: LogicalNode) -> LogicalNode {
        match plan {
            LogicalNode::Project { input, .. } => *input,
            other => other,
        }
    }

    #[test]
    fn in_subquery_lowers_to_semi_join_with_equi_on() {
        // users.id(0) IN (SELECT uid FROM orders) → Semi join, ON id = orders.uid.
        // The subquery's lone column sits at ordinal 3 (users width = 3).
        let plan = under_project(lower(
            "SELECT name FROM users WHERE id IN (SELECT uid FROM orders)",
        ));
        match plan {
            LogicalNode::Join {
                kind: JoinKind::Semi,
                on: Some(Expr::BinaryOp { left, op, right }),
                ..
            } => {
                assert_eq!(op, BinaryOp::Eq);
                assert!(matches!(*left, Expr::Column(c) if c.ordinal == 0));
                assert!(matches!(*right, Expr::Column(c) if c.ordinal == 3));
            }
            other => panic!("expected Semi join with equi ON, got {other:?}"),
        }
    }

    #[test]
    fn exists_lowers_to_keyless_semi_join() {
        let plan = under_project(lower(
            "SELECT name FROM users WHERE EXISTS (SELECT oid FROM orders)",
        ));
        assert!(
            matches!(
                plan,
                LogicalNode::Join {
                    kind: JoinKind::Semi,
                    on: None,
                    ..
                }
            ),
            "EXISTS → keyless Semi join: {plan:?}"
        );
    }

    #[test]
    fn not_exists_lowers_to_keyless_anti_join() {
        let plan = under_project(lower(
            "SELECT name FROM users WHERE NOT EXISTS (SELECT oid FROM orders)",
        ));
        assert!(
            matches!(
                plan,
                LogicalNode::Join {
                    kind: JoinKind::Anti,
                    on: None,
                    ..
                }
            ),
            "NOT EXISTS → keyless Anti join: {plan:?}"
        );
    }

    #[test]
    fn where_subquery_and_plain_predicate_compose() {
        // `age > 25 AND id IN (subquery)` → Filter(age>25) over Semi join.
        let plan = under_project(lower(
            "SELECT name FROM users WHERE age > 25 AND id IN (SELECT uid FROM orders)",
        ));
        match plan {
            LogicalNode::Filter { input, predicate } => {
                assert!(matches!(
                    predicate,
                    Expr::BinaryOp {
                        op: BinaryOp::Gt,
                        ..
                    }
                ));
                assert!(matches!(
                    *input,
                    LogicalNode::Join {
                        kind: JoinKind::Semi,
                        ..
                    }
                ));
            }
            other => panic!("expected Filter over Semi join, got {other:?}"),
        }
    }

    #[test]
    fn not_in_subquery_lowers_to_null_aware_anti_join() {
        // `x NOT IN (subquery)` → AntiNullAware join with the equi ON (NULL-correct
        // via the executor's three-valued evaluation).
        let plan = under_project(lower(
            "SELECT name FROM users WHERE id NOT IN (SELECT uid FROM orders)",
        ));
        match plan {
            LogicalNode::Join {
                kind: JoinKind::AntiNullAware,
                on: Some(Expr::BinaryOp { op, .. }),
                ..
            } => assert_eq!(op, BinaryOp::Eq),
            other => panic!("expected AntiNullAware join with equi ON, got {other:?}"),
        }
    }

    #[test]
    fn correlated_in_subquery_is_decorrelated() {
        // (Was `correlated_subquery_is_declined` before Stage 3.) The subquery
        // references the outer `users.age` → can't lower in isolation → the
        // decorrelator lifts `total = users.age` and the IN-equality into the Semi
        // join ON. Correlated IN is now supported rather than declined.
        let plan = under_project(lower(
            "SELECT name FROM users WHERE id IN (SELECT uid FROM orders WHERE total = users.age)",
        ));
        assert!(
            matches!(
                plan,
                LogicalNode::Join {
                    kind: JoinKind::Semi,
                    on: Some(Expr::BinaryOp {
                        op: BinaryOp::And,
                        ..
                    }),
                    ..
                }
            ),
            "correlated IN decorrelates to a Semi join with AND ON, got {plan:?}"
        );
    }

    #[test]
    fn in_subquery_arity_mismatch_is_declined() {
        // A scalar operand against a 2-column subquery is an arity mismatch → decline
        // (falls through to legacy). Multi-column IN needs a matching tuple operand.
        assert!(
            lower_sql(
                "SELECT name FROM users WHERE id IN (SELECT oid, uid FROM orders)",
                &catalog()
            )
            .is_err()
        );
    }

    #[test]
    fn multi_column_in_lowers_to_semi_with_and_on() {
        // `(a, b) IN (SELECT x, y …)` → Semi join, ON = (a = x AND b = y).
        match under_project(lower(
            "SELECT name FROM users WHERE (id, age) IN (SELECT oid, uid FROM orders)",
        )) {
            LogicalNode::Join {
                kind: JoinKind::Semi,
                on:
                    Some(Expr::BinaryOp {
                        op: BinaryOp::And, ..
                    }),
                ..
            } => {}
            other => panic!("expected Semi join with AND ON, got {other:?}"),
        }
        // `(a, b) NOT IN (…)` → AntiNullAware with the same AND ON.
        match under_project(lower(
            "SELECT name FROM users WHERE (id, age) NOT IN (SELECT oid, uid FROM orders)",
        )) {
            LogicalNode::Join {
                kind: JoinKind::AntiNullAware,
                on:
                    Some(Expr::BinaryOp {
                        op: BinaryOp::And, ..
                    }),
                ..
            } => {}
            other => panic!("expected AntiNullAware join with AND ON, got {other:?}"),
        }
    }

    // --- Stage 1: uncorrelated scalar subquery in projection -------------------

    #[test]
    fn scalar_subquery_in_projection_lowers_to_left_join_over_assert() {
        // `SELECT name, (SELECT max(total) FROM orders) AS m FROM users`
        // → Project over `LEFT JOIN ON TRUE` whose right is an AssertMaxOneRow.
        // users width = 3, so the hoisted scalar column sits at ordinal 3.
        let plan = lower("SELECT name, (SELECT max(total) FROM orders) AS m FROM users");
        let LogicalNode::Project { input, outputs } = plan else {
            panic!("expected outer Project");
        };
        // The `m` output references the hoisted column at ordinal 3.
        let m = outputs.iter().find(|o| o.name == "m").expect("m output");
        assert!(
            matches!(&m.expr, Expr::Column(c) if c.ordinal == 3),
            "m should reference ordinal 3, got {:?}",
            m.expr
        );
        match *input {
            LogicalNode::Join {
                kind: JoinKind::Left,
                on: None,
                left,
                right,
                ..
            } => {
                assert!(
                    matches!(*left, LogicalNode::Scan { .. }),
                    "left is base Scan"
                );
                assert!(
                    matches!(*right, LogicalNode::AssertMaxOneRow { .. }),
                    "right is AssertMaxOneRow, got {right:?}"
                );
            }
            other => panic!("expected LEFT JOIN ON TRUE over AssertMaxOneRow, got {other:?}"),
        }
    }

    #[test]
    fn select_star_does_not_expose_scalar_subquery_column() {
        // `SELECT *, (SELECT …) AS m` → the `*` expands ONLY the 3 base columns
        // (ordinals 0,1,2); the hoisted column is appended after lowering and is
        // never enumerated by the wildcard. Total outputs = 3 base + 1 explicit.
        let LogicalNode::Project { outputs, .. } =
            lower("SELECT *, (SELECT max(total) FROM orders) AS m FROM users")
        else {
            panic!("expected Project");
        };
        assert_eq!(
            outputs.len(),
            4,
            "3 base columns + 1 explicit subquery column"
        );
        // The wildcard-expanded columns keep base ordinals 0,1,2.
        let base: Vec<usize> = outputs
            .iter()
            .take(3)
            .filter_map(|o| match &o.expr {
                Expr::Column(c) => Some(c.ordinal),
                _ => None,
            })
            .collect();
        assert_eq!(base, vec![0, 1, 2], "star expands base columns only");
    }

    #[test]
    fn two_scalar_subqueries_get_distinct_appended_ordinals() {
        // Two scalar subqueries in one SELECT append left-to-right at ordinals
        // 3 and 4 (after the 3 base columns). Guards the ordinal accounting.
        let LogicalNode::Project { outputs, .. } = lower(
            "SELECT (SELECT max(total) FROM orders) AS a, \
                    (SELECT min(total) FROM orders) AS b FROM users",
        ) else {
            panic!("expected Project");
        };
        let a = outputs.iter().find(|o| o.name == "a").expect("a");
        let b = outputs.iter().find(|o| o.name == "b").expect("b");
        assert!(matches!(&a.expr, Expr::Column(c) if c.ordinal == 3));
        assert!(matches!(&b.expr, Expr::Column(c) if c.ordinal == 4));
    }

    #[test]
    fn scalar_subquery_with_multiple_columns_declines() {
        // A scalar subquery must return exactly one column.
        assert!(
            lower_sql(
                "SELECT (SELECT oid, uid FROM orders) FROM users",
                &catalog()
            )
            .is_err()
        );
    }

    #[test]
    fn correlated_scalar_aggregate_lowers_to_left_join_over_group_by() {
        // `SELECT name, (SELECT max(total) FROM orders WHERE orders.uid = users.id) AS m`
        // → Project over `u LEFT JOIN (orders GROUP BY uid → [uid, max]) ON u.id = agg.uid`.
        // users width 3 → agg key at ordinal 3, agg value (m) at ordinal 4.
        let plan = lower(
            "SELECT name, (SELECT max(total) FROM orders WHERE orders.uid = users.id) AS m FROM users",
        );
        let LogicalNode::Project { input, outputs } = plan else {
            panic!("expected Project");
        };
        let m = outputs.iter().find(|o| o.name == "m").expect("m output");
        assert!(
            matches!(&m.expr, Expr::Column(c) if c.ordinal == 4),
            "m references the aggregate value at ordinal 4, got {:?}",
            m.expr
        );
        match *input {
            LogicalNode::Join {
                kind: JoinKind::Left,
                on:
                    Some(Expr::BinaryOp {
                        op: BinaryOp::Eq, ..
                    }),
                right,
                ..
            } => assert!(
                matches!(*right, LogicalNode::Aggregate { .. }),
                "right side is the GROUP BY aggregate, got {right:?}"
            ),
            other => panic!("expected LEFT JOIN with equi ON over Aggregate, got {other:?}"),
        }
    }

    #[test]
    fn two_key_correlated_scalar_aggregate_groups_and_joins_on_both_keys() {
        // TD-REL-LOWER-5 slice 3 (TPC-H Q20): a correlated scalar aggregate with TWO
        // correlation keys (`orders.uid = users.id AND orders.oid = users.age`) →
        // LEFT JOIN over `orders GROUP BY uid, oid` with the join ON = both key
        // equalities ANDed. Previously declined at the single-key MVP guard. This is
        // the root cause of Q20 (masked upstream as "IN in expression position").
        let plan = lower(
            "SELECT id FROM users WHERE age > \
             (SELECT avg(total) FROM orders WHERE orders.uid = users.id AND orders.oid = users.age)",
        );
        let LogicalNode::Project { input, .. } = plan else {
            panic!("expected Project");
        };
        let LogicalNode::Filter { input, .. } = *input else {
            panic!("expected Filter over the LEFT JOIN, got {input:?}");
        };
        let LogicalNode::Join {
            kind: JoinKind::Left,
            on: Some(on),
            right,
            ..
        } = *input
        else {
            panic!("expected LEFT JOIN over the decorrelated aggregate");
        };
        // The join ON conjoins BOTH key equalities.
        assert!(
            matches!(
                on,
                Expr::BinaryOp {
                    op: BinaryOp::And,
                    ..
                }
            ),
            "2-key correlation joins on both keys ANDed, got {on:?}"
        );
        let LogicalNode::Aggregate { group_by, .. } = *right else {
            panic!("join right must be the GROUP BY aggregate");
        };
        assert_eq!(group_by.len(), 2, "GROUP BY both inner correlation keys");
    }

    #[test]
    fn multi_table_correlated_scalar_aggregate_lowers_to_left_join_over_grouped_inner_join() {
        // TD-REL-LOWER-5 slice 1 (TPC-H Q2): a correlated scalar aggregate whose
        // body is MULTI-TABLE — `min(cost) FROM ps, s WHERE ps_sk = sk AND
        // ps_pk = p.pk` — decorrelates. The inner join conditions + inner-local
        // filters fold into the inner Filter (the ONE correlation equi `ps_pk =
        // pk` becomes the group/join key), so the aggregate groups the joined
        // inner body. Previously declined at the single-table guard.
        let mut c = InMemoryCatalog::new();
        c.register(
            "p",
            RelationalSchema::new(vec![
                ColumnInfo::new("pk", ProximaType::Int64, false),
                ColumnInfo::new("budget", ProximaType::Float64, true),
            ]),
        );
        c.register(
            "ps",
            RelationalSchema::new(vec![
                ColumnInfo::new("ps_pk", ProximaType::Int64, false),
                ColumnInfo::new("ps_sk", ProximaType::Int64, false),
                ColumnInfo::new("cost", ProximaType::Float64, false),
            ]),
        );
        c.register(
            "s",
            RelationalSchema::new(vec![ColumnInfo::new("sk", ProximaType::Int64, false)]),
        );

        let plan = lower_sql(
            "SELECT pk FROM p WHERE budget = \
             (SELECT min(cost) FROM ps, s WHERE ps_sk = sk AND ps_pk = pk)",
            &c,
        )
        .expect("multi-table correlated scalar aggregate must lower");

        // Project over Filter over `p LEFT JOIN Aggregate(...) ON p.pk = grp_key`.
        let filter = under_project(plan);
        let LogicalNode::Filter { input, .. } = filter else {
            panic!("expected Filter, got {filter:?}");
        };
        let LogicalNode::Join {
            kind: JoinKind::Left,
            on: Some(Expr::BinaryOp {
                op: BinaryOp::Eq, ..
            }),
            right,
            ..
        } = *input
        else {
            panic!("expected LEFT JOIN with equi ON over the decorrelated aggregate");
        };
        // The aggregate's input is the MULTI-TABLE inner body (a Join under the
        // inner Filter) — the crux of this slice.
        let LogicalNode::Aggregate {
            input: agg_input,
            group_by,
            ..
        } = *right
        else {
            panic!("right side must be the GROUP BY aggregate");
        };
        assert_eq!(group_by.len(), 1, "single correlation group key");
        let inner_has_join = |mut node: &LogicalNode| loop {
            match node {
                LogicalNode::Filter { input, .. } => node = input,
                LogicalNode::Join { .. } => break true,
                _ => break false,
            }
        };
        assert!(
            inner_has_join(&agg_input),
            "the decorrelated aggregate must group a multi-table (joined) inner body, got {agg_input:?}"
        );

        // No regression: the single-table body still lowers.
        assert!(
            lower_sql(
                "SELECT id FROM users WHERE age < \
                 (SELECT avg(total) FROM orders WHERE orders.uid = users.id)",
                &catalog()
            )
            .is_ok()
        );
    }

    #[test]
    fn correlated_scalar_count_declines_count_bug() {
        // COUNT over an empty correlated group must be 0, but the LEFT JOIN + GROUP
        // BY rewrite yields NULL — so correlated COUNT declines to legacy.
        assert!(
            lower_sql(
                "SELECT (SELECT count(*) FROM orders WHERE orders.uid = users.id) FROM users",
                &catalog()
            )
            .is_err()
        );
    }

    #[test]
    fn correlated_scalar_non_aggregate_declines() {
        // A correlated scalar subquery that is NOT a single aggregate (a plain
        // column with LIMIT-style semantics) is out of scope → declines.
        assert!(
            lower_sql(
                "SELECT (SELECT total FROM orders WHERE orders.uid = users.id) FROM users",
                &catalog()
            )
            .is_err()
        );
    }

    // --- Stage 2: uncorrelated scalar subquery in WHERE ------------------------

    #[test]
    fn where_scalar_subquery_lowers_to_filter_over_left_join() {
        // `WHERE age > (SELECT max(total) FROM orders)` → Filter over a
        // `LEFT JOIN ON TRUE` whose right is AssertMaxOneRow. The predicate
        // references the hoisted column at ordinal 3 (users width = 3).
        let plan = under_project(lower(
            "SELECT name FROM users WHERE age > (SELECT max(total) FROM orders)",
        ));
        let LogicalNode::Filter { input, predicate } = plan else {
            panic!("expected Filter");
        };
        // The predicate's right operand is the hoisted scalar column at ordinal 3.
        match predicate {
            Expr::BinaryOp {
                op: BinaryOp::Gt,
                right,
                ..
            } => {
                assert!(matches!(*right, Expr::Column(c) if c.ordinal == 3));
            }
            other => panic!("expected `age > col(3)`, got {other:?}"),
        }
        match *input {
            LogicalNode::Join {
                kind: JoinKind::Left,
                on: None,
                right,
                ..
            } => assert!(matches!(*right, LogicalNode::AssertMaxOneRow { .. })),
            other => panic!("expected LEFT JOIN ON TRUE over AssertMaxOneRow, got {other:?}"),
        }
    }

    #[test]
    fn where_and_projection_scalar_subqueries_get_disjoint_ordinals() {
        // A scalar subquery in WHERE (appended at ordinal 3) and one in the
        // projection (appended at ordinal 4) must not collide.
        let LogicalNode::Project { outputs, input } = lower(
            "SELECT (SELECT min(total) FROM orders) AS p FROM users \
             WHERE age > (SELECT max(total) FROM orders)",
        ) else {
            panic!("expected Project");
        };
        let p = outputs.iter().find(|o| o.name == "p").expect("p");
        assert!(
            matches!(&p.expr, Expr::Column(c) if c.ordinal == 4),
            "projection scalar should be at ordinal 4 (after base 0..2 + WHERE scalar at 3), got {:?}",
            p.expr
        );
        // The WHERE scalar's join sits below the Filter, the projection scalar's
        // above it — two nested LEFT JOINs total.
        assert!(matches!(
            *input,
            LogicalNode::Join {
                kind: JoinKind::Left,
                ..
            }
        ));
    }

    // --- Stage 3: correlated subqueries via decorrelation ----------------------

    #[test]
    fn correlated_exists_lowers_to_semi_with_correlation_on() {
        // `EXISTS (SELECT 1 FROM orders WHERE orders.uid = users.id)` → Semi join
        // with the correlation lifted to ON. users width 3, so orders.uid sits at
        // combined ordinal 4 (3 + uid's inner ordinal 1) and users.id at 0.
        let plan = under_project(lower(
            "SELECT name FROM users WHERE EXISTS \
             (SELECT 1 FROM orders WHERE orders.uid = users.id)",
        ));
        match plan {
            LogicalNode::Join {
                kind: JoinKind::Semi,
                on:
                    Some(Expr::BinaryOp {
                        op: BinaryOp::Eq,
                        left,
                        right,
                    }),
                right: inner,
                ..
            } => {
                let ords = [
                    match *left {
                        Expr::Column(c) => c.ordinal,
                        _ => usize::MAX,
                    },
                    match *right {
                        Expr::Column(c) => c.ordinal,
                        _ => usize::MAX,
                    },
                ];
                assert!(
                    ords.contains(&0) && ords.contains(&4),
                    "correlation references outer users.id(0) and inner orders.uid(4): {ords:?}"
                );
                assert!(
                    matches!(*inner, LogicalNode::Scan { .. }),
                    "right is the bare inner Scan"
                );
            }
            other => panic!("expected correlated Semi join with equi ON, got {other:?}"),
        }
    }

    #[test]
    fn correlated_exists_splits_inner_local_predicate_into_filter() {
        // `total > 100` is inner-local (→ inner Filter); `orders.uid = users.id` is
        // the correlation (→ join ON).
        let plan = under_project(lower(
            "SELECT name FROM users WHERE EXISTS \
             (SELECT 1 FROM orders WHERE total > 100 AND orders.uid = users.id)",
        ));
        match plan {
            LogicalNode::Join {
                kind: JoinKind::Semi,
                on: Some(_),
                right: inner,
                ..
            } => assert!(
                matches!(*inner, LogicalNode::Filter { .. }),
                "inner-local predicate becomes a Filter, got {inner:?}"
            ),
            other => panic!("expected Semi join over inner Filter, got {other:?}"),
        }
    }

    #[test]
    fn correlated_not_exists_lowers_to_anti() {
        let plan = under_project(lower(
            "SELECT name FROM users WHERE NOT EXISTS \
             (SELECT 1 FROM orders WHERE orders.uid = users.id)",
        ));
        assert!(matches!(
            plan,
            LogicalNode::Join {
                kind: JoinKind::Anti,
                on: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn correlated_in_lowers_to_semi_with_correlation_and_in_equality() {
        // `id IN (SELECT uid FROM orders WHERE orders.total > users.age)` → Semi
        // join, ON = (orders.total > users.age) AND (users.id = orders.uid).
        let plan = under_project(lower(
            "SELECT name FROM users WHERE id IN \
             (SELECT uid FROM orders WHERE orders.total > users.age)",
        ));
        assert!(
            matches!(
                plan,
                LogicalNode::Join {
                    kind: JoinKind::Semi,
                    on: Some(Expr::BinaryOp {
                        op: BinaryOp::And,
                        ..
                    }),
                    ..
                }
            ),
            "expected Semi join with AND ON (correlation AND in-equality)"
        );
    }

    #[test]
    fn correlated_not_in_lowers_to_null_aware_anti() {
        let plan = under_project(lower(
            "SELECT name FROM users WHERE id NOT IN \
             (SELECT uid FROM orders WHERE orders.total > users.age)",
        ));
        assert!(matches!(
            plan,
            LogicalNode::Join {
                kind: JoinKind::AntiNullAware,
                on: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn correlated_subquery_with_inner_join_declines() {
        // The subquery is not a single-table shape → decorrelation declines → the
        // whole query falls through to legacy (errors here).
        assert!(
            lower_sql(
                "SELECT name FROM users WHERE EXISTS \
                 (SELECT 1 FROM orders JOIN users u2 ON orders.uid = u2.id \
                  WHERE orders.uid = users.id)",
                &catalog()
            )
            .is_err()
        );
    }

    #[test]
    fn correlated_multi_column_in_declines() {
        // Correlated multi-column IN is out of scope → decline.
        assert!(
            lower_sql(
                "SELECT name FROM users WHERE (id, age) IN \
                 (SELECT uid, oid FROM orders WHERE orders.total > users.age)",
                &catalog()
            )
            .is_err()
        );
    }

    #[test]
    fn select_all_from_table() {
        let plan = lower("SELECT * FROM users");
        // Project(Scan)
        match plan {
            LogicalNode::Project { input, outputs } => {
                assert_eq!(outputs.len(), 3);
                assert!(matches!(*input, LogicalNode::Scan { .. }));
            }
            _ => panic!("expected Project, got {plan:?}"),
        }
    }

    #[test]
    fn select_with_where() {
        let plan = lower("SELECT id FROM users WHERE age > 25");
        // Project(Filter(Scan))
        match plan {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Filter {
                    input: _,
                    predicate,
                } => {
                    assert!(matches!(
                        predicate,
                        Expr::BinaryOp {
                            op: BinaryOp::Gt,
                            ..
                        }
                    ));
                }
                other => panic!("expected Filter, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn select_count_star() {
        // COUNT(*) with no GROUP BY → Project(Aggregate(Scan))
        let plan = lower("SELECT COUNT(*) FROM users");
        match plan {
            LogicalNode::Project { input, outputs } => {
                assert_eq!(outputs.len(), 1);
                match *input {
                    LogicalNode::Aggregate {
                        aggregates,
                        group_by,
                        ..
                    } => {
                        assert!(group_by.is_empty());
                        assert_eq!(aggregates.len(), 1);
                        assert!(matches!(
                            aggregates[0].agg,
                            AggregateExpr::Count {
                                arg: None,
                                distinct: false
                            }
                        ));
                    }
                    other => panic!("expected Aggregate, got {other:?}"),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn select_sum_with_group_by() {
        let plan = lower("SELECT age, SUM(id) FROM users GROUP BY age");
        match plan {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Aggregate {
                    group_by,
                    aggregates,
                    ..
                } => {
                    assert_eq!(group_by.len(), 1);
                    assert_eq!(aggregates.len(), 1);
                    assert!(matches!(aggregates[0].agg, AggregateExpr::Sum { .. }));
                }
                other => panic!("expected Aggregate, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn having_with_aggregate_lowers_to_filter_over_aggregate() {
        // TD-REL-LOWER-5 slice 3 (Q11): `HAVING sum(total) > 100` — an aggregate
        // in HAVING — is extracted into the Aggregate and applied as a
        // POST-AGGREGATE Filter (was declined: "HAVING with aggregate calls").
        let plan = lower("SELECT uid, sum(total) FROM orders GROUP BY uid HAVING sum(total) > 100");
        let LogicalNode::Project { input, .. } = plan else {
            panic!("expected Project");
        };
        let LogicalNode::Filter { input, predicate } = *input else {
            panic!("expected HAVING as a post-aggregate Filter, got {input:?}");
        };
        // Filter compares the extracted HAVING aggregate slot to the literal.
        assert!(
            matches!(
                predicate,
                Expr::BinaryOp {
                    op: BinaryOp::Gt,
                    ..
                }
            ),
            "HAVING predicate is `agg > lit`, got {predicate:?}"
        );
        let LogicalNode::Aggregate {
            aggregates, having, ..
        } = *input
        else {
            panic!("Filter input must be the Aggregate");
        };
        assert!(
            having.is_none(),
            "HAVING moved to the Filter, not the field"
        );
        assert_eq!(
            aggregates.len(),
            2,
            "projection sum + extracted HAVING sum (dedup is a later optimization)"
        );
    }

    #[test]
    fn having_with_scalar_subquery_hoists_left_join_over_aggregate() {
        // Q11 shape: `HAVING sum(total) > (SELECT sum(total)*k FROM orders)` — an
        // aggregate compared to an uncorrelated scalar subquery. The subquery is
        // hoisted as a LEFT JOIN ON TRUE above the Aggregate; the predicate refs
        // the extracted agg slot and the hoisted scalar column.
        let plan = lower(
            "SELECT uid, sum(total) FROM orders GROUP BY uid \
             HAVING sum(total) > (SELECT sum(total) * 2 FROM orders)",
        );
        let LogicalNode::Project { input, .. } = plan else {
            panic!("expected Project");
        };
        let LogicalNode::Filter { input, .. } = *input else {
            panic!("expected post-aggregate Filter");
        };
        let LogicalNode::Join {
            kind: JoinKind::Left,
            on: None,
            left,
            right,
            ..
        } = *input
        else {
            panic!("expected LEFT JOIN ON TRUE for the hoisted scalar subquery, got {input:?}");
        };
        assert!(
            matches!(*left, LogicalNode::Aggregate { .. }),
            "join left is the Aggregate"
        );
        assert!(
            matches!(*right, LogicalNode::AssertMaxOneRow { .. }),
            "join right is the AssertMaxOneRow-guarded uncorrelated scalar"
        );
    }

    #[test]
    fn plain_having_over_group_key_keeps_aggregate_having_field() {
        // No-regression: a HAVING with neither aggregate nor subquery
        // (`HAVING age > 30`, over the group key) keeps the cheap
        // `Aggregate.having` path — no Filter, no restructure.
        let plan = lower("SELECT age, count(*) FROM users GROUP BY age HAVING age > 30");
        let LogicalNode::Project { input, .. } = plan else {
            panic!("expected Project");
        };
        let LogicalNode::Aggregate { having, .. } = *input else {
            panic!("expected Aggregate directly under Project (no Filter)");
        };
        assert!(
            having.is_some(),
            "plain HAVING stays on the Aggregate.having field"
        );
    }

    #[test]
    fn select_group_by() {
        let plan = lower("SELECT age FROM users GROUP BY age");
        match plan {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Aggregate { group_by, .. } => {
                    assert_eq!(group_by.len(), 1);
                }
                other => panic!("expected Aggregate, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn select_group_key_projection_rebinds_to_post_aggregate_ordinal() {
        // `name` is column ordinal 1 in `users`. After GROUP BY, the Aggregate
        // node places group keys FIRST, so the projected `name` must reference
        // the post-aggregate group slot (ordinal 0) — NOT its pre-aggregate
        // ordinal 1, which now holds the COUNT result (Int64). Regression for a
        // type-mismatch that surfaced whenever the grouped column isn't the
        // table's first column.
        let plan = lower("SELECT name, COUNT(*) FROM users GROUP BY name");
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        assert_eq!(outputs.len(), 2);
        match &outputs[0].expr {
            Expr::Column(c) => {
                assert_eq!(c.ordinal, 0, "group key rebinds to post-agg slot 0");
                assert_eq!(c.ty, ProximaType::String);
            }
            other => panic!("expected group-key column ref, got {other:?}"),
        }
        match &outputs[1].expr {
            Expr::Column(c) => assert_eq!(c.ordinal, 1, "COUNT slot follows the group keys"),
            other => panic!("expected aggregate column ref, got {other:?}"),
        }
    }

    #[test]
    fn select_aggregate_arithmetic_hoists_nested_aggregates() {
        // TD-REL-LOWER-4: an aggregate nested inside projection arithmetic
        // (`sum(id) / sum(age)`, the ratio / market-share shape) must lower to
        // a `Project(BinaryOp(Div, __agg_0, __agg_1))` over an `Aggregate` with
        // TWO extracted aggregate slots — not decline with "aggregate function
        // in non-aggregate position". Each hoisted ColumnRef names its slot
        // (`__agg_<slot>`) at the post-aggregate ordinal so the planner's
        // name-based projection-pushdown rebind resolves it.
        let plan = lower("SELECT sum(id) / sum(age) FROM users");
        let LogicalNode::Project { outputs, input } = plan else {
            panic!("expected Project");
        };
        assert_eq!(outputs.len(), 1);
        match &outputs[0].expr {
            Expr::BinaryOp {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                match (&**left, &**right) {
                    (Expr::Column(l), Expr::Column(r)) => {
                        // Global aggregate (no GROUP BY): slots are ordinals 0,1.
                        assert_eq!(l.ordinal, 0);
                        assert_eq!(r.ordinal, 1);
                        assert_eq!(l.name, "__agg_0");
                        assert_eq!(r.name, "__agg_1");
                    }
                    other => panic!("expected two aggregate column refs, got {other:?}"),
                }
            }
            other => panic!("expected BinaryOp(Div) over aggregates, got {other:?}"),
        }
        match *input {
            LogicalNode::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                assert_eq!(group_by.len(), 0, "no GROUP BY");
                assert_eq!(aggregates.len(), 2, "both sums extracted to slots");
                assert!(matches!(aggregates[0].agg, AggregateExpr::Sum { .. }));
                assert!(matches!(aggregates[1].agg, AggregateExpr::Sum { .. }));
                assert_eq!(aggregates[0].name, "__agg_0");
                assert_eq!(aggregates[1].name, "__agg_1");
            }
            other => panic!("expected Aggregate, got {other:?}"),
        }
    }

    #[test]
    fn select_with_inner_join() {
        let plan = lower("SELECT users.id FROM users INNER JOIN orders ON users.id = orders.uid");
        match plan {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Join { kind, .. } => {
                    assert_eq!(kind, JoinKind::Inner);
                }
                other => panic!("expected Join, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn select_unqualified_columns_across_bare_join() {
        // Bare `JOIN` (no INNER keyword) must lower as INNER JOIN, and
        // unqualified column refs resolve against the merged scope: `name` is
        // unique to users (ordinal 1), `total` unique to orders (combined
        // ordinal 3+2=5). Regression: bare `JOIN` was previously rejected as an
        // unsupported join operator, forcing a fall-through to the legacy path.
        let plan = lower("SELECT name, total FROM users JOIN orders ON users.id = orders.uid");
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        let ords: Vec<(String, usize)> = outputs
            .iter()
            .map(|o| match &o.expr {
                Expr::Column(c) => (o.name.clone(), c.ordinal),
                other => panic!("expected column, got {other:?}"),
            })
            .collect();
        // users = [id@0, name@1, age@2]; orders = [oid@3, uid@4, total@5].
        assert_eq!(
            ords,
            vec![("name".to_string(), 1), ("total".to_string(), 5)]
        );
    }

    #[test]
    fn bare_left_and_right_join_lower_to_outer_join_kinds() {
        // Bare `LEFT JOIN` / `RIGHT JOIN` (no `OUTER`) parse to sqlparser's
        // `Left`/`Right` variants (distinct from `LeftOuter`/`RightOuter`) and
        // must lower to the same JoinKind. Regression: the bare forms were
        // previously rejected as unsupported → PATH B fell through to legacy.
        for (sql, expected) in [
            (
                "SELECT name, total FROM users LEFT JOIN orders ON users.id = orders.uid",
                JoinKind::Left,
            ),
            (
                "SELECT name, total FROM users RIGHT JOIN orders ON users.id = orders.uid",
                JoinKind::Right,
            ),
        ] {
            let plan = lower(sql);
            let LogicalNode::Project { input, .. } = plan else {
                panic!("expected Project for `{sql}`");
            };
            match *input {
                LogicalNode::Join { kind, .. } => assert_eq!(kind, expected, "for `{sql}`"),
                other => panic!("expected Join for `{sql}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn select_distinct() {
        let plan = lower("SELECT DISTINCT id FROM users");
        match plan {
            LogicalNode::Distinct { .. } => {}
            _ => panic!("expected Distinct, got {plan:?}"),
        }
    }

    #[test]
    fn select_order_by_limit_offset() {
        // NB: MVP limit — ORDER BY resolves against the
        // post-projection schema, so the sort key must appear in
        // the projection list. ORDER BY on non-projected columns
        // is a Phase 3 enhancement.
        let plan = lower("SELECT id FROM users ORDER BY id DESC LIMIT 10 OFFSET 5");
        match plan {
            LogicalNode::Limit { limit, offset, .. } => {
                assert_eq!(limit, Some(10));
                assert_eq!(offset, 5);
            }
            _ => panic!("expected Limit, got {plan:?}"),
        }
    }

    #[test]
    fn union_all_combines_selects() {
        let plan = lower("SELECT id FROM users UNION ALL SELECT oid FROM orders");
        match plan {
            LogicalNode::Union { inputs, all } => {
                assert!(all);
                assert_eq!(inputs.len(), 2);
            }
            _ => panic!("expected Union, got {plan:?}"),
        }
    }

    #[test]
    fn case_coalesce_nullif_lower_to_expr_variants() {
        fn proj0(sql: &str) -> Expr {
            match lower(sql) {
                LogicalNode::Project { outputs, .. } => outputs[0].expr.clone(),
                other => panic!("expected Project, got {other:?}"),
            }
        }
        // Searched CASE and simple CASE both lower to Expr::Case.
        assert!(matches!(
            proj0("SELECT CASE WHEN age > 30 THEN 1 ELSE 0 END FROM users"),
            Expr::Case { .. }
        ));
        assert!(matches!(
            proj0("SELECT CASE age WHEN 30 THEN 1 ELSE 0 END FROM users"),
            Expr::Case { .. }
        ));
        // COALESCE / NULLIF lower to their dedicated NULL-aware variants, not FuncCall.
        assert!(matches!(
            proj0("SELECT COALESCE(name, 'x') FROM users"),
            Expr::Coalesce(_)
        ));
        assert!(matches!(
            proj0("SELECT NULLIF(age, 0) FROM users"),
            Expr::NullIf { .. }
        ));
        // NULLIF arity is enforced.
        assert!(lower_sql("SELECT NULLIF(age) FROM users", &catalog()).is_err());
    }

    #[test]
    fn intersect_and_except_lower_to_setop() {
        // INTERSECT → SetOp{Intersect}; EXCEPT ALL → SetOp{Except, all}.
        match lower("SELECT id FROM users INTERSECT SELECT oid FROM orders") {
            LogicalNode::SetOp { op, all, .. } => {
                assert_eq!(op, SetOpKind::Intersect);
                assert!(!all);
            }
            other => panic!("expected SetOp(Intersect), got {other:?}"),
        }
        match lower("SELECT id FROM users EXCEPT ALL SELECT oid FROM orders") {
            LogicalNode::SetOp { op, all, .. } => {
                assert_eq!(op, SetOpKind::Except);
                assert!(all);
            }
            other => panic!("expected SetOp(Except, all), got {other:?}"),
        }
    }

    #[test]
    fn values_emits_values_node() {
        let plan = lower("VALUES (1, 'a'), (2, 'b')");
        match plan {
            LogicalNode::Values { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("expected Values, got {plan:?}"),
        }
    }

    #[test]
    fn unknown_table_errors() {
        let err = lower_sql("SELECT * FROM nope", &catalog()).unwrap_err();
        assert!(matches!(err, FrontendError::TableNotFound(_)));
    }

    #[test]
    fn unknown_column_errors() {
        let err = lower_sql("SELECT bogus FROM users", &catalog()).unwrap_err();
        assert!(matches!(err, FrontendError::ColumnNotFound(_)));
    }

    #[test]
    fn dml_is_rejected() {
        let err = lower_sql("INSERT INTO users VALUES (1, 'x', 30)", &catalog()).unwrap_err();
        assert!(matches!(err, FrontendError::Unsupported(_)));
    }

    #[test]
    fn subquery_under_or_is_rejected() {
        // Only TOP-LEVEL AND-conjuncts lift into Semi/Anti joins. A subquery under
        // OR isn't a standalone conjunct → not lifted → lower_expr rejects it (a
        // disjunctive membership test can't be expressed as a filtering join), so
        // the query falls through to the legacy path.
        let err = lower_sql(
            "SELECT id FROM users WHERE age > 1 OR id IN (SELECT uid FROM orders)",
            &catalog(),
        )
        .unwrap_err();
        assert!(matches!(err, FrontendError::Unsupported(_)));
    }

    #[test]
    fn decline_structures_unsupported_reason_for_trace() {
        // ADR-064 / TD-TRACE-1: a decline must carry a structured {node, why}
        // (the real reason), not be dropped so the pgwire path re-guesses it.
        // Subquery-in-expression-position (the Q18/Q20 shape) → node="subquery".
        let err = lower_sql(
            "SELECT id FROM users WHERE age > 1 OR id IN (SELECT uid FROM orders)",
            &catalog(),
        )
        .unwrap_err();
        let d = err.decline();
        assert_eq!(d.node, "subquery", "unexpected node for {err:?}");
        assert_eq!(d.why, err.to_string(), "why must be the Display reason");
        assert!(
            !d.why.contains("comma-separated"),
            "must not be a masked guess"
        );

        // Node classification covers the other declining families the arc hit.
        assert_eq!(
            FrontendError::Unsupported(
                "HAVING with aggregate calls (use a sub-select for now)".into()
            )
            .decline()
            .node,
            "having"
        );
        assert_eq!(
            FrontendError::Unsupported("correlated scalar subquery".into())
                .decline()
                .node,
            "subquery"
        );
        assert_eq!(
            FrontendError::TableNotFound("orders".into()).decline().node,
            "table"
        );
    }

    #[test]
    fn is_null_lowers_correctly() {
        let plan = lower("SELECT id FROM users WHERE name IS NULL");
        match plan {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Filter { predicate, .. } => {
                    assert!(matches!(predicate, Expr::IsNull { not: false, .. }));
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn between_lowers_correctly() {
        let plan = lower("SELECT id FROM users WHERE age BETWEEN 18 AND 65");
        match plan {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Filter { predicate, .. } => {
                    assert!(matches!(predicate, Expr::Between { not: false, .. }));
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn scalar_function_lowers_to_funccall_with_registry_return_type() {
        // F1b: a scalar function in projection position now lowers to `Expr::FuncCall`
        // (previously rejected as "aggregate function in non-aggregate position"). The
        // `return_ty` is resolved from the shared builtin registry: `upper`→String,
        // `abs`→Float64.
        let plan = lower("SELECT upper(name), abs(age) FROM users");
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        assert_eq!(outputs.len(), 2);
        match &outputs[0].expr {
            Expr::FuncCall {
                name,
                args,
                return_ty,
            } => {
                assert_eq!(name, "upper");
                assert_eq!(args.len(), 1);
                assert_eq!(*return_ty, ProximaType::String);
            }
            other => panic!("expected FuncCall upper, got {other:?}"),
        }
        match &outputs[1].expr {
            Expr::FuncCall {
                name, return_ty, ..
            } => {
                assert_eq!(name, "abs");
                assert_eq!(*return_ty, ProximaType::Float64);
            }
            other => panic!("expected FuncCall abs, got {other:?}"),
        }
    }

    #[test]
    fn aggregate_in_projection_arithmetic_lowers_but_unsupported_nesting_declines() {
        // TD-REL-LOWER-4: an aggregate nested in projection ARITHMETIC is valid
        // SQL (a global aggregate expression) and now lowers — previously it was
        // rejected as "aggregate function in non-aggregate position".
        let plan = lower_sql("SELECT sum(age) + 1 FROM users", &catalog());
        assert!(plan.is_ok(), "sum(age) + 1 must lower: {plan:?}");

        // But an aggregate nested inside a CASE / scalar function at the
        // projection top level is not yet supported and must still decline
        // (graceful fall-through to legacy), not silently mislower.
        let err = lower_sql(
            "SELECT case when sum(age) > 0 then 1 else 0 end FROM users",
            &catalog(),
        )
        .unwrap_err();
        assert!(matches!(err, FrontendError::Unsupported(_)));
    }

    #[test]
    fn cast_lowers_typed_and_identity_folds() {
        // ADR-043 Inv 2: a real conversion lowers to a typed `Expr::Cast`.
        // age (Int32) AS bigint → Cast { ty: Int64 }.
        let plan = lower("SELECT CAST(age AS bigint) FROM users");
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        match &outputs[0].expr {
            Expr::Cast { ty, .. } => assert_eq!(*ty, ProximaType::Int64),
            other => panic!("expected Cast to Int64, got {other:?}"),
        }
        // Identity-fold (zero-copy): name (String) AS varchar already has the target type,
        // so it folds to the bare column — no `Cast` node emitted.
        let plan = lower("SELECT CAST(name AS varchar) FROM users");
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        assert!(
            matches!(&outputs[0].expr, Expr::Column(_)),
            "identity cast should fold to the column (zero-copy), got {:?}",
            outputs[0].expr
        );
    }

    #[test]
    fn unknown_scalar_function_still_lowers() {
        // Unknown functions lower permissively (the DataFusion path may serve them; the
        // Volcano path raises UnknownFunction at execution, not at parse). Placeholder
        // return type is String.
        let plan = lower("SELECT some_udf(name) FROM users");
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        match &outputs[0].expr {
            Expr::FuncCall { name, .. } => assert_eq!(name, "some_udf"),
            other => panic!("expected FuncCall, got {other:?}"),
        }
    }

    #[test]
    fn window_function_declines_instead_of_dropping_the_over_clause() {
        // TD-187: lowering a windowed call here (rather than declining) would
        // silently DROP the `OVER (...)` clause and produce a plain scalar/
        // aggregate `Expr` — which either mis-evaluates (for a name that
        // happens to also be a registered scalar/aggregate builtin) or
        // declines further downstream by luck (for names like LAG that
        // aren't registered). Both `lower_scalar_function` (LAG/LEAD/RANK/…)
        // and `lower_aggregate_call` (AVG/SUM/COUNT/MIN/MAX used as a window
        // aggregate) must decline unconditionally whenever `f.over.is_some()`,
        // so the caller always falls through to DataFusion's real window
        // executor via the `ctx.sql()` fallback.
        assert!(
            lower_sql(
                "SELECT id, age - lag(age) OVER (ORDER BY id) AS delta FROM users",
                &catalog()
            )
            .is_err(),
            "a bare LAG window call must decline, not silently drop OVER"
        );
        assert!(
            lower_sql(
                "SELECT id, avg(age) OVER (PARTITION BY id ORDER BY id) AS moving_avg FROM users",
                &catalog()
            )
            .is_err(),
            "an aggregate-named function used as a window aggregate must decline too"
        );
        // No regression: the same functions used WITHOUT a window clause still lower.
        assert!(
            lower_sql("SELECT id, avg(age) FROM users GROUP BY id", &catalog()).is_ok(),
            "a genuine GROUP BY aggregate (no OVER) must keep lowering"
        );
    }

    #[test]
    fn custom_aggregate_lowers_to_aggregate_expr_custom() {
        // F3: a non-builtin aggregate registered in the shared registry (`product`) lowers to
        // AggregateExpr::Custom carrying the registry's return type, instead of being rejected.
        let plan = lower("SELECT product(age) FROM users");
        let LogicalNode::Project { input, .. } = plan else {
            panic!("expected Project");
        };
        let LogicalNode::Aggregate {
            aggregates,
            group_by,
            ..
        } = *input
        else {
            panic!("expected Aggregate");
        };
        assert!(group_by.is_empty());
        assert_eq!(aggregates.len(), 1);
        match &aggregates[0].agg {
            AggregateExpr::Custom {
                name,
                args,
                return_ty,
                ..
            } => {
                assert_eq!(name, "product");
                assert_eq!(args.len(), 1);
                assert_eq!(*return_ty, ProximaType::Float64);
            }
            other => panic!("expected Custom, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_column_errors_in_join() {
        // both tables have a column named `id`? users has id, orders has oid.
        // Construct an actual ambiguity: users.id vs another table with id.
        let mut c = catalog();
        c.register(
            "people",
            RelationalSchema::new(vec![ColumnInfo::new("id", ProximaType::Int64, false)]),
        );
        let err = lower_sql(
            "SELECT id FROM users INNER JOIN people ON users.id = people.id",
            &c,
        )
        .unwrap_err();
        assert!(matches!(err, FrontendError::AmbiguousColumn(_)));
    }

    #[test]
    fn lower_function_body_lowers_param_expression() {
        // F5: `CREATE FUNCTION double(x BIGINT) AS 'x * 2'` body → BinaryOp(Mul, Col(0), Lit).
        let body = lower_function_body("x * 2", &[("x".to_string(), ProximaType::Int64)]).unwrap();
        match body {
            Expr::BinaryOp { op, left, right } => {
                assert_eq!(op, BinaryOp::Mul);
                assert!(matches!(*left, Expr::Column(ColumnRef { ordinal: 0, .. })));
                assert!(matches!(*right, Expr::Literal { .. }));
            }
            other => panic!("expected BinaryOp, got {other:?}"),
        }
    }

    #[test]
    fn lower_function_body_supports_builtins_and_rejects_unknown_param() {
        // body may call builtins (resolved at execution against the registry) ...
        assert!(lower_function_body("abs(x)", &[("x".to_string(), ProximaType::Float64)]).is_ok());
        // ... and a reference to an undeclared parameter errors clearly.
        let err =
            lower_function_body("y + 1", &[("x".to_string(), ProximaType::Int64)]).unwrap_err();
        assert!(matches!(err, FrontendError::ColumnNotFound(_)));
    }

    // --- Special-syntax ANSI functions → registry FuncCall (TD-OLAP-5 P1b) ----

    fn first_output_expr(plan: LogicalNode) -> Expr {
        let LogicalNode::Project { outputs, .. } = plan else {
            panic!("expected Project");
        };
        outputs[0].expr.clone()
    }

    #[test]
    fn substring_from_for_lowers_to_substr() {
        let expr = first_output_expr(lower("SELECT SUBSTRING(name FROM 1 FOR 2) FROM users"));
        match expr {
            Expr::FuncCall {
                name,
                args,
                return_ty,
            } => {
                assert_eq!(name, "substr");
                assert_eq!(args.len(), 3);
                assert_eq!(return_ty, ProximaType::String);
            }
            other => panic!("expected substr FuncCall, got {other:?}"),
        }
        // SUBSTRING(x FOR n) implies FROM 1.
        let expr = first_output_expr(lower("SELECT SUBSTRING(name FOR 2) FROM users"));
        match expr {
            Expr::FuncCall { name, args, .. } => {
                assert_eq!(name, "substr");
                assert!(matches!(
                    args[1],
                    Expr::Literal {
                        value: ProximaValue::Int64(1),
                        ..
                    }
                ));
            }
            other => panic!("expected substr FuncCall, got {other:?}"),
        }
    }

    #[test]
    fn extract_lowers_to_date_part() {
        let expr = first_output_expr(lower("SELECT EXTRACT(YEAR FROM name) FROM users"));
        match expr {
            Expr::FuncCall {
                name,
                args,
                return_ty,
            } => {
                assert_eq!(name, "date_part");
                assert_eq!(return_ty, ProximaType::Float64);
                assert!(matches!(
                    &args[0],
                    Expr::Literal {
                        value: ProximaValue::String(field),
                        ..
                    } if field == "year"
                ));
            }
            other => panic!("expected date_part FuncCall, got {other:?}"),
        }
    }

    #[test]
    fn trim_variants_lower_to_trim_family() {
        for (sql, want) in [
            ("SELECT TRIM(name) FROM users", "btrim"),
            ("SELECT TRIM(LEADING 'x' FROM name) FROM users", "ltrim"),
            ("SELECT TRIM(TRAILING 'x' FROM name) FROM users", "rtrim"),
            ("SELECT TRIM(BOTH 'x' FROM name) FROM users", "btrim"),
        ] {
            let expr = first_output_expr(lower(sql));
            match expr {
                Expr::FuncCall { name, .. } => assert_eq!(name, want, "for {sql}"),
                other => panic!("expected {want} FuncCall for {sql}, got {other:?}"),
            }
        }
    }

    #[test]
    fn position_lowers_to_strpos_with_swapped_args() {
        let expr = first_output_expr(lower("SELECT POSITION('a' IN name) FROM users"));
        match expr {
            Expr::FuncCall { name, args, .. } => {
                assert_eq!(name, "strpos");
                // strpos(string, substring): the IN operand comes first.
                assert!(matches!(args[0], Expr::Column(_)), "haystack first");
                assert!(
                    matches!(&args[1], Expr::Literal { value: ProximaValue::String(sub), .. } if sub == "a")
                );
            }
            other => panic!("expected strpos FuncCall, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod td_rel_lower_order_by {
    use super::*;
    fn catalog() -> InMemoryCatalog {
        let mut c = InMemoryCatalog::new();
        c.register(
            "users",
            RelationalSchema::new(vec![
                ColumnInfo::new("id", ProximaType::Int64, false),
                ColumnInfo::new("name", ProximaType::String, true),
            ]),
        );
        c.register(
            "orders",
            RelationalSchema::new(vec![
                ColumnInfo::new("oid", ProximaType::Int64, false),
                ColumnInfo::new("uid", ProximaType::Int64, false),
                ColumnInfo::new("total", ProximaType::Float64, false),
            ]),
        );
        c
    }

    /// TD-REL-LOWER-1 root cause: a qualified ORDER BY key over an aliased
    /// join declined (`ColumnNotFound("u.name")`) because the post-projection
    /// scope carries no table qualifiers — sending the whole query down the
    /// legacy single-table fallthrough. It must lower.
    #[test]
    fn order_by_qualified_key_over_aliased_join_lowers() {
        for sql in [
            "SELECT u.name, sum(o.total) FROM users u JOIN orders o ON u.id = o.uid GROUP BY u.name ORDER BY u.name",
            "SELECT u.name, sum(o.total) FROM users u, orders o WHERE u.id = o.uid GROUP BY u.name ORDER BY u.name",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.uid ORDER BY u.name",
            "SELECT users.name FROM users ORDER BY users.name",
        ] {
            let plan = lower_sql(sql, &catalog())
                .unwrap_or_else(|e| panic!("must lower, declined: {e:?}\n  {sql}"));
            assert!(
                matches!(plan, LogicalNode::Sort { .. }),
                "outermost Sort for {sql}"
            );
        }
    }

    /// The fallback resolves by the BARE column name — a qualified key whose
    /// column does not exist in the output still errors (never silently binds).
    #[test]
    fn order_by_qualified_key_missing_column_still_errors() {
        let err = lower_sql(
            "SELECT u.name FROM users u ORDER BY u.no_such_column",
            &catalog(),
        )
        .expect_err("unknown sort column must decline");
        assert!(
            matches!(err, FrontendError::ColumnNotFound(_)),
            "got {err:?}"
        );
    }
}
