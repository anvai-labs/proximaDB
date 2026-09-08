//! SQL frontend parser: wraps sqlparser-rs and produces the internal AST.

use anyhow::{Result, anyhow};
use sqlparser::ast::{
    BinaryOperator, ConflictTarget, CreateTableOptions, Cte as SqlCte, Expr as SqlExpr,
    FunctionArg, FunctionArgExpr, GroupByExpr, Join as SqlJoin, JoinConstraint, JoinOperator,
    OnConflictAction, OnInsert, OrderByExpr as SqlOrderByExpr, Query as SqlQuery,
    Select as SqlSelect, SelectItem, SetExpr, SetOperator as SqlSetOperator, SqlOption, Statement,
    TableFactor, TableWithJoins, UnaryOperator, Value, With as SqlWith,
};
use sqlparser::ast::{CreateIndex, IndexOption};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::collections::HashMap;

// DML types for INSERT/UPDATE/DELETE
use crate::services::dml::{
    ComparisonOperator as DmlComparisonOperator, Condition, DmlStatement, LogicalOperator,
    SqlValueLiteral, WhereClause,
};

use crate::query::table_write_plan::{
    ConflictPolicy, CopyIntoPlan, DistributionMode, LogicalTableRef, ReadSource, SnapshotRef,
    WriteMode,
};

use crate::query::ast::{
    BinaryOp, Cte, Expr, Join, JoinType, Literal, OrderByExpr, ProjectionItem, Query, Select,
    SetOp, TableRef, UnaryOp,
};

/// Validate that a transport request contains exactly one syntactically valid
/// SQL statement without lowering or executing it.
///
/// Network RPCs are deliberately single-statement. Keeping this validation on
/// the shared frontend parser prevents REST/gRPC adapters from growing their
/// own quote/semicolon splitters while still allowing every statement kind the
/// execution authority may handle (SELECT, DDL, DML, EXPLAIN, extensions).
pub fn validate_single_statement(sql: &str) -> Result<()> {
    let statements = Parser::parse_sql(&GenericDialect, sql)
        .map_err(|error| anyhow!("SQL parsing failed: {error}"))?;
    match statements.len() {
        1 => Ok(()),
        0 => Err(anyhow!("No SQL statement found")),
        count => Err(anyhow!(
            "Exactly one SQL statement is required, found {count}"
        )),
    }
}

/// Detect an `EXPLAIN [ANALYZE] <inner>` wrapper.
///
/// Returns `Some((is_analyze, inner))` when `query` begins with `EXPLAIN`,
/// where `inner` is the wrapped statement text with the `EXPLAIN [ANALYZE]`
/// (or `EXPLAIN (… ANALYZE …)`) prefix stripped, and `is_analyze` is true when
/// the EXPLAIN requested ANALYZE. Returns `None` when `query` is not an EXPLAIN.
///
/// This is the single source of truth for EXPLAIN-prefix parsing shared by the
/// ROOT handler and the `QueryFacadeAdapter` SQL path (TD-104 / seam S1).
pub fn parse_explain_kind(query: &str) -> Option<(bool, &str)> {
    let upper = query.to_ascii_uppercase();
    if !upper.starts_with("EXPLAIN") {
        return None;
    }
    let after_explain = query["EXPLAIN".len()..].trim_start();
    let after_upper = after_explain.to_ascii_uppercase();

    if after_upper.starts_with("ANALYZE") {
        let inner = after_explain["ANALYZE".len()..].trim_start();
        return Some((true, inner));
    }

    if after_explain.starts_with('(') {
        let close = after_explain.find(')')?;
        let options = &after_explain[1..close];
        let is_analyze = options.to_ascii_uppercase().contains("ANALYZE");
        let rest = after_explain[close + 1..].trim_start();
        return Some((is_analyze, rest));
    }

    Some((false, after_explain))
}

/// Best-effort output-column label for a single SELECT projection item: the
/// explicit alias when present, else the trailing identifier of the expression
/// (`a.attname` → `attname`), else the raw expression text. Used to mirror the
/// client's expected result shape for catalog introspection.
fn projection_label(item: &SelectItem) -> String {
    match item {
        SelectItem::ExprWithAlias { alias, .. } => alias.value.clone(),
        SelectItem::UnnamedExpr(expr) => {
            let text = expr.to_string();
            text.rsplit_once('.')
                .map(|(_, last)| last.trim().to_string())
                .unwrap_or(text)
        }
        SelectItem::Wildcard(_) => "*".to_string(),
        _ => item.to_string(),
    }
}

/// Parse a single SELECT's projection list and return the output-column labels
/// the client expects (explicit alias, else a label derived from the
/// expression). Returns `None` for anything that isn't a single SELECT, for any
/// projection containing a wildcard (whose width we cannot determine from our
/// side), or on parse failure — callers fall back to substring-based dispatch
/// and the width-safety net. This lets catalog introspection answer psql-style
/// queries (`SELECT ... AS "Name" ... FROM pg_catalog.pg_class ...`) with a
/// result whose columns match the client's SELECT list by construction.
pub fn parse_select_aliases(sql: &str) -> Option<Vec<String>> {
    let statements = Parser::parse_sql(&GenericDialect, sql).ok()?;
    let [statement] = statements.as_slice() else {
        return None;
    };
    let Statement::Query(query) = statement else {
        return None;
    };
    let SetExpr::Select(select) = &*query.body else {
        return None;
    };
    if select
        .projection
        .iter()
        .any(|item| matches!(item, SelectItem::Wildcard(_)))
    {
        return None;
    }
    Some(select.projection.iter().map(projection_label).collect())
}

/// SQL frontend parser that converts SQL text into internal AST nodes
pub struct SqlFrontendParser {
    dialect: GenericDialect,
}

impl SqlFrontendParser {
    /// Create a new SQL frontend parser.
    pub fn new() -> Self {
        Self {
            dialect: GenericDialect {},
        }
    }

    /// Parse SQL text into the internal AST.
    pub fn parse(&self, sql: &str) -> Result<Query> {
        // Parse SQL using sqlparser-rs
        let statements = Parser::parse_sql(&self.dialect, sql)
            .map_err(|e| anyhow!("SQL parsing failed: {}", e))?;

        if statements.is_empty() {
            return Err(anyhow!("No SQL statements found"));
        }

        if statements.len() > 1 {
            return Err(anyhow!(
                "Multiple statements not supported, found {}",
                statements.len()
            ));
        }

        // Convert the first statement to internal AST
        let statement = &statements[0];
        self.convert_statement(statement)
    }

    fn convert_statement(&self, statement: &Statement) -> Result<Query> {
        match statement {
            Statement::Query(query) => self.convert_query(query),
            Statement::Insert { .. } => Err(anyhow!(
                "INSERT statements are parsed but not yet executed. Use REST/gRPC API for insertions."
            )),
            Statement::Update { .. } => Err(anyhow!(
                "UPDATE statements are parsed but not yet executed. Use REST/gRPC API for updates."
            )),
            Statement::Delete { .. } => Err(anyhow!(
                "DELETE statements are parsed but not yet executed. Use REST/gRPC API for deletions."
            )),
            _ => Err(anyhow!(
                "Only SELECT queries are currently supported. For DML operations (INSERT/UPDATE/DELETE), use REST/gRPC API."
            )),
        }
    }

    fn convert_query(&self, query: &SqlQuery) -> Result<Query> {
        // Handle WITH (CTE)
        if let Some(with) = &query.with {
            let ctes = self.convert_with(with)?;
            let inner = self.convert_query_no_with(query)?; // convert body
            return Ok(Query::With {
                ctes,
                query: Box::new(inner),
            });
        }
        self.convert_query_no_with(query)
    }

    /// Convert query without handling WITH (handled by caller)
    fn convert_query_no_with(&self, query: &SqlQuery) -> Result<Query> {
        match &*query.body {
            SetExpr::Select(select) => Ok(Query::Select(self.convert_select(select, query)?)),
            SetExpr::SetOperation {
                left,
                op,
                right,
                set_quantifier,
            } => {
                let (set_op, all) = match op {
                    SqlSetOperator::Union => (
                        SetOp::Union,
                        matches!(set_quantifier, sqlparser::ast::SetQuantifier::All),
                    ),
                    SqlSetOperator::Intersect => (
                        SetOp::Intersect,
                        matches!(set_quantifier, sqlparser::ast::SetQuantifier::All),
                    ),
                    SqlSetOperator::Except | SqlSetOperator::Minus => (
                        SetOp::Except,
                        matches!(set_quantifier, sqlparser::ast::SetQuantifier::All),
                    ),
                };
                let left_q = self.convert_setexpr(left)?;
                let right_q = self.convert_setexpr(right)?;
                Ok(Query::Set {
                    left: Box::new(left_q),
                    op: set_op,
                    all,
                    right: Box::new(right_q),
                })
            }
            other => Err(anyhow!("Unsupported query body: {:?}", other)),
        }
    }

    fn convert_setexpr(&self, expr: &SetExpr) -> Result<Query> {
        match expr {
            SetExpr::Select(sel) => {
                // Minimal wrapper Query for nested select
                Ok(Query::Select(self.convert_select(
                    sel,
                    &SqlQuery {
                        with: None,
                        body: Box::new(SetExpr::Select(sel.clone())),
                        order_by: None,
                        limit_clause: None,
                        fetch: None,
                        locks: vec![],
                        for_clause: None,
                        settings: None,
                        format_clause: None,
                        pipe_operators: vec![],
                    },
                )?))
            }
            SetExpr::SetOperation {
                left,
                op,
                right,
                set_quantifier,
            } => {
                let (set_op, all) = match op {
                    SqlSetOperator::Union => (
                        SetOp::Union,
                        matches!(set_quantifier, sqlparser::ast::SetQuantifier::All),
                    ),
                    SqlSetOperator::Intersect => (
                        SetOp::Intersect,
                        matches!(set_quantifier, sqlparser::ast::SetQuantifier::All),
                    ),
                    SqlSetOperator::Except | SqlSetOperator::Minus => (
                        SetOp::Except,
                        matches!(set_quantifier, sqlparser::ast::SetQuantifier::All),
                    ),
                };
                let left_q = self.convert_setexpr(left)?;
                let right_q = self.convert_setexpr(right)?;
                Ok(Query::Set {
                    left: Box::new(left_q),
                    op: set_op,
                    all,
                    right: Box::new(right_q),
                })
            }
            _ => Err(anyhow!("Unsupported set expression: {:?}", expr)),
        }
    }

    fn convert_with(&self, with: &SqlWith) -> Result<Vec<Cte>> {
        let mut ctes = Vec::new();
        for SqlCte { alias, query, .. } in &with.cte_tables {
            let name = alias.name.value.clone();
            let q = self.convert_query(query)?;
            ctes.push(Cte {
                name,
                query: Box::new(q),
            });
        }
        Ok(ctes)
    }

    fn convert_select(&self, select: &SqlSelect, query: &SqlQuery) -> Result<Select> {
        // Convert projection
        let projection = select
            .projection
            .iter()
            .map(|item| self.convert_select_item_with_alias(item))
            .collect::<Result<Vec<ProjectionItem>>>()?;

        // Convert FROM clause
        let from = select
            .from
            .iter()
            .map(|table_with_joins| self.convert_table_with_joins(table_with_joins))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        // Convert JOINs (handled in convert_table_with_joins)
        let joins = select
            .from
            .iter()
            .flat_map(|table_with_joins| &table_with_joins.joins)
            .map(|join| self.convert_join(join))
            .collect::<Result<Vec<_>>>()?;

        // Convert WHERE clause
        let selection = select
            .selection
            .as_ref()
            .map(|expr| self.convert_expr(expr))
            .transpose()?;

        // Convert GROUP BY
        let group_by = match &select.group_by {
            sqlparser::ast::GroupByExpr::All(_) => vec![], // Handle GROUP BY ALL (PostgreSQL extension)
            sqlparser::ast::GroupByExpr::Expressions(exprs, _modifiers) => exprs
                .iter()
                .map(|expr| self.convert_expr(expr))
                .collect::<Result<Vec<_>>>()?,
        };

        // Convert HAVING
        let having = select
            .having
            .as_ref()
            .map(|expr| self.convert_expr(expr))
            .transpose()?;

        // Convert ORDER BY
        let order_by = if let Some(order_by_clause) = &query.order_by {
            match &order_by_clause.kind {
                sqlparser::ast::OrderByKind::Expressions(exprs) => exprs
                    .iter()
                    .map(|order_expr| self.convert_order_by_expr(order_expr))
                    .collect::<Result<Vec<_>>>()?,
                sqlparser::ast::OrderByKind::All(_) => vec![],
            }
        } else {
            vec![]
        };

        // Convert LIMIT and OFFSET (sqlparser 0.59 uses LimitClause enum)
        let (limit, offset) = if let Some(lc) = &query.limit_clause {
            match lc {
                sqlparser::ast::LimitClause::LimitOffset {
                    limit: lim,
                    offset: off,
                    ..
                } => {
                    let limit_val = lim.as_ref().and_then(|expr| {
                        if let SqlExpr::Value(value_with_span) = expr
                            && let Value::Number(n, _) = &value_with_span.value
                        {
                            return n.parse::<u64>().ok();
                        }
                        None
                    });
                    let offset_val = off.as_ref().and_then(|off_expr| {
                        if let SqlExpr::Value(value_with_span) = &off_expr.value
                            && let Value::Number(n, _) = &value_with_span.value
                        {
                            return n.parse::<u64>().ok();
                        }
                        None
                    });
                    (limit_val, offset_val)
                }
                sqlparser::ast::LimitClause::OffsetCommaLimit {
                    offset: off,
                    limit: lim,
                } => {
                    let limit_val = if let SqlExpr::Value(value_with_span) = lim {
                        if let Value::Number(n, _) = &value_with_span.value {
                            n.parse::<u64>().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let offset_val = if let SqlExpr::Value(value_with_span) = off {
                        if let Value::Number(n, _) = &value_with_span.value {
                            n.parse::<u64>().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    (limit_val, offset_val)
                }
            }
        } else {
            (None, None)
        };

        Ok(Select {
            projection,
            from,
            joins,
            selection,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn convert_select_item_with_alias(&self, item: &SelectItem) -> Result<ProjectionItem> {
        match item {
            SelectItem::UnnamedExpr(expr) => Ok(ProjectionItem {
                expr: self.convert_expr(expr)?,
                alias: None,
            }),
            SelectItem::ExprWithAlias { expr, alias } => Ok(ProjectionItem {
                expr: self.convert_expr(expr)?,
                alias: Some(alias.value.clone()),
            }),
            SelectItem::Wildcard(_) => Ok(ProjectionItem {
                expr: Expr::Identifier("*".to_string()),
                alias: None,
            }),
            _ => Err(anyhow!("Unsupported select item: {:?}", item)),
        }
    }

    fn convert_table_with_joins(&self, table_with_joins: &TableWithJoins) -> Result<Vec<TableRef>> {
        let tables = vec![self.convert_table_factor(&table_with_joins.relation)?];

        // Note: Joins are handled separately in convert_select
        // This just returns the main table references

        Ok(tables)
    }

    fn convert_table_factor(&self, table_factor: &TableFactor) -> Result<TableRef> {
        match table_factor {
            TableFactor::Table { name, alias, .. } => {
                let table_name = name.to_string();
                let alias_name = alias.as_ref().map(|a| a.name.value.clone());

                Ok(TableRef {
                    name: Some(table_name),
                    subquery: None,
                    alias: alias_name,
                })
            }
            TableFactor::Derived {
                subquery, alias, ..
            } => {
                let converted_subquery = self.convert_query(subquery)?;
                let alias_name = alias.as_ref().map(|a| a.name.value.clone());

                Ok(TableRef {
                    name: None,
                    subquery: Some(Box::new(converted_subquery)),
                    alias: alias_name,
                })
            }
            _ => Err(anyhow!("Unsupported table factor: {:?}", table_factor)),
        }
    }

    fn convert_join(&self, join: &SqlJoin) -> Result<Join> {
        let kind = match &join.join_operator {
            JoinOperator::Join(_) => JoinType::Inner, // Default JOIN is treated as INNER
            JoinOperator::Inner(_) => JoinType::Inner,
            JoinOperator::Left(_) | JoinOperator::LeftOuter(_) => JoinType::LeftOuter,
            JoinOperator::Right(_) | JoinOperator::RightOuter(_) => JoinType::RightOuter,
            JoinOperator::FullOuter(_) => JoinType::FullOuter,
            JoinOperator::CrossJoin(_) => JoinType::Cross,
            _ => return Err(anyhow!("Unsupported join type: {:?}", join.join_operator)),
        };

        let right = self.convert_table_factor(&join.relation)?;

        let on = match &join.join_operator {
            JoinOperator::Join(constraint)
            | JoinOperator::Inner(constraint)
            | JoinOperator::Left(constraint)
            | JoinOperator::LeftOuter(constraint)
            | JoinOperator::Right(constraint)
            | JoinOperator::RightOuter(constraint)
            | JoinOperator::FullOuter(constraint) => match constraint {
                JoinConstraint::On(expr) => Some(self.convert_expr(expr)?),
                JoinConstraint::Using(_) => return Err(anyhow!("USING clause not supported")),
                JoinConstraint::Natural => None,
                JoinConstraint::None => None,
            },
            JoinOperator::CrossJoin(_) => None, // Cross join has no ON condition
            _ => None,
        };

        Ok(Join {
            join_type: kind,
            right_table: right,
            on_condition: on,
        })
    }

    fn convert_expr(&self, expr: &SqlExpr) -> Result<Expr> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(Expr::Identifier(ident.value.clone())),

            SqlExpr::Value(value_with_span) => match &value_with_span.value {
                Value::Placeholder(ph) => Ok(Expr::Param(ph.clone())),
                _ => Ok(Expr::Literal(self.convert_value(&value_with_span.value)?)),
            },

            SqlExpr::BinaryOp { left, op, right } => {
                let left_expr = Box::new(self.convert_expr(left)?);
                let right_expr = Box::new(self.convert_expr(right)?);
                let binary_op = self.convert_binary_op(op)?;

                Ok(Expr::Binary {
                    left: left_expr,
                    op: binary_op,
                    right: right_expr,
                })
            }

            SqlExpr::UnaryOp { op, expr } => {
                let converted_expr = Box::new(self.convert_expr(expr)?);
                let unary_op = self.convert_unary_op(op)?;

                Ok(Expr::Unary {
                    op: unary_op,
                    expr: converted_expr,
                })
            }

            SqlExpr::Function(func) => {
                let name = func.name.to_string();
                // In sqlparser 0.59, args is FunctionArguments enum
                let arg_list = match &func.args {
                    sqlparser::ast::FunctionArguments::List(func_arg_list) => &func_arg_list.args,
                    _ => return Err(anyhow!("Unsupported function argument type")),
                };
                let args = arg_list
                    .iter()
                    .map(|arg| self.convert_function_arg(arg))
                    .collect::<Result<Vec<_>>>()?;

                // Check if it's an aggregate function
                if self.is_aggregate_function(&name) {
                    Ok(Expr::AggCall { name, args })
                } else {
                    // Normalize function name for SKS detection
                    let upper = name.to_uppercase();
                    match upper.as_str() {
                        "SIMILAR" => {
                            // Expected: SIMILAR(field, query, metric?, threshold?)
                            if args.len() < 2 {
                                return Err(anyhow!(
                                    "SIMILAR(field, query, metric?, threshold?) requires at least 2 arguments"
                                ));
                            }

                            // Validate field identifier
                            let field = match &args[0] {
                                Expr::Identifier(s) => s.clone(),
                                Expr::FuncCall { .. } | Expr::AggCall { .. } => {
                                    return Err(anyhow!(
                                        "SIMILAR: first argument must be a column identifier (e.g., embedding)"
                                    ));
                                }
                                _ => {
                                    return Err(anyhow!(
                                        "SIMILAR: unsupported first argument; expected identifier (e.g., embedding)"
                                    ));
                                }
                            };

                            // Validate query expression is present (allow literal/param/identifier)
                            let query_expr = Box::new(args[1].clone());
                            if matches!(args[1], Expr::Literal(Literal::Null)) {
                                return Err(anyhow!("SIMILAR: query argument cannot be NULL"));
                            }

                            // Optional metric validation
                            let metric = if args.len() >= 3 {
                                match &args[2] {
                                    Expr::Literal(Literal::String(s)) => {
                                        let m = s.to_lowercase();
                                        match m.as_str() {
                                            "cosine" | "dot" | "euclidean" => Some(m),
                                            _ => {
                                                return Err(anyhow!(
                                                    "SIMILAR: unsupported metric '{}'. Use 'cosine', 'dot', or 'euclidean'",
                                                    s
                                                ));
                                            }
                                        }
                                    }
                                    _ => {
                                        return Err(anyhow!(
                                            "SIMILAR: metric must be a string literal (e.g., 'cosine')"
                                        ));
                                    }
                                }
                            } else {
                                None
                            };

                            // Optional threshold validation
                            let threshold = if args.len() >= 4 {
                                match &args[3] {
                                    Expr::Literal(Literal::Number(n)) => {
                                        if n < &0.0 {
                                            return Err(anyhow!("SIMILAR: threshold must be ≥ 0"));
                                        }
                                        Some(*n)
                                    }
                                    _ => {
                                        return Err(anyhow!(
                                            "SIMILAR: threshold must be numeric (e.g., 0.75)"
                                        ));
                                    }
                                }
                            } else {
                                None
                            };

                            // Extra arguments guard
                            if args.len() > 4 {
                                return Err(anyhow!(
                                    "SIMILAR: too many arguments; expected up to 4"
                                ));
                            }

                            Ok(Expr::SksSimilar {
                                field,
                                query: query_expr,
                                metric,
                                threshold,
                            })
                        }
                        "FOLLOW" => {
                            // Expected: FOLLOW(start, edge_type, max_depth)
                            if args.len() != 3 {
                                return Err(anyhow!(
                                    "FOLLOW(start, edge, max_depth) requires exactly 3 arguments"
                                ));
                            }

                            // start node id (string or identifier)
                            let start = match &args[0] {
                                Expr::Literal(Literal::String(_)) | Expr::Identifier(_) => {
                                    Box::new(args[0].clone())
                                }
                                _ => {
                                    return Err(anyhow!(
                                        "FOLLOW: start must be an identifier or string literal (node id)"
                                    ));
                                }
                            };

                            // edge type (string or identifier)
                            let edge = match &args[1] {
                                Expr::Literal(Literal::String(s)) => s.clone(),
                                Expr::Identifier(s) => s.clone(),
                                _ => {
                                    return Err(anyhow!(
                                        "FOLLOW: edge must be a string literal or identifier (edge type)"
                                    ));
                                }
                            };

                            // depth must be positive integer
                            let max_depth = match &args[2] {
                                Expr::Literal(Literal::Number(n)) => {
                                    if n < &1.0 {
                                        return Err(anyhow!("FOLLOW: max_depth must be ≥ 1"));
                                    }
                                    *n as u32
                                }
                                _ => {
                                    return Err(anyhow!(
                                        "FOLLOW: max_depth must be a number (e.g., 2)"
                                    ));
                                }
                            };

                            Ok(Expr::SksFollow {
                                start,
                                edge,
                                max_depth,
                            })
                        }
                        "GEO_DISTANCE" => {
                            // GEO_DISTANCE(lat1, lon1, lat2, lon2) returns distance in km
                            if args.len() != 4 {
                                return Err(anyhow!(
                                    "GEO_DISTANCE(lat1, lon1, lat2, lon2) requires exactly 4 arguments"
                                ));
                            }
                            Ok(Expr::GeoDistance {
                                lat1: Box::new(args[0].clone()),
                                lon1: Box::new(args[1].clone()),
                                lat2: Box::new(args[2].clone()),
                                lon2: Box::new(args[3].clone()),
                            })
                        }
                        "GEO_WITHIN_DISTANCE" | "GEO_NEAR" => {
                            // GEO_WITHIN_DISTANCE(lat, lon, center_lat, center_lon, radius, unit?)
                            if args.len() < 5 || args.len() > 6 {
                                return Err(anyhow!(
                                    "GEO_WITHIN_DISTANCE(lat, lon, center_lat, center_lon, radius, unit?) requires 5-6 arguments"
                                ));
                            }
                            let unit = if args.len() == 6 {
                                match &args[5] {
                                    Expr::Literal(Literal::String(s)) => {
                                        let u = s.to_lowercase();
                                        match u.as_str() {
                                            "km" | "kilometers" => Some("km".to_string()),
                                            "mi" | "miles" => Some("mi".to_string()),
                                            "m" | "meters" => Some("m".to_string()),
                                            _ => {
                                                return Err(anyhow!(
                                                    "GEO_WITHIN_DISTANCE: unit must be 'km', 'mi', or 'm'"
                                                ));
                                            }
                                        }
                                    }
                                    _ => {
                                        return Err(anyhow!(
                                            "GEO_WITHIN_DISTANCE: unit must be a string literal"
                                        ));
                                    }
                                }
                            } else {
                                Some("km".to_string()) // Default to km
                            };
                            Ok(Expr::GeoWithinDistance {
                                lat: Box::new(args[0].clone()),
                                lon: Box::new(args[1].clone()),
                                center_lat: Box::new(args[2].clone()),
                                center_lon: Box::new(args[3].clone()),
                                radius: Box::new(args[4].clone()),
                                unit,
                            })
                        }
                        "GEO_WITHIN_BOX" | "GEO_BBOX" => {
                            // GEO_WITHIN_BOX(lat, lon, sw_lat, sw_lon, ne_lat, ne_lon)
                            if args.len() != 6 {
                                return Err(anyhow!(
                                    "GEO_WITHIN_BOX(lat, lon, sw_lat, sw_lon, ne_lat, ne_lon) requires exactly 6 arguments"
                                ));
                            }
                            Ok(Expr::GeoWithinBox {
                                lat: Box::new(args[0].clone()),
                                lon: Box::new(args[1].clone()),
                                sw_lat: Box::new(args[2].clone()),
                                sw_lon: Box::new(args[3].clone()),
                                ne_lat: Box::new(args[4].clone()),
                                ne_lon: Box::new(args[5].clone()),
                            })
                        }
                        "GEO_POINT" => {
                            // GEO_POINT(lat, lon) creates a geo point
                            if args.len() != 2 {
                                return Err(anyhow!(
                                    "GEO_POINT(lat, lon) requires exactly 2 arguments"
                                ));
                            }
                            Ok(Expr::GeoPoint {
                                lat: Box::new(args[0].clone()),
                                lon: Box::new(args[1].clone()),
                            })
                        }
                        _ => Ok(Expr::FuncCall { name, args }),
                    }
                }
            }

            SqlExpr::CompoundIdentifier(idents) => {
                let combined = idents
                    .iter()
                    .map(|i| i.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                Ok(Expr::Identifier(combined))
            }

            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                case_token: _,
                end_token: _,
            } => {
                // Convert CASE expression
                let lowered_operand = if let Some(op) = operand {
                    Some(Box::new(self.convert_expr(op)?))
                } else {
                    None
                };

                let mut lowered_conditions = Vec::new();
                // In sqlparser 0.59, conditions is Vec<CaseWhen>
                for case_when in conditions {
                    let when_expr = self.convert_expr(&case_when.condition)?;
                    let then_expr = self.convert_expr(&case_when.result)?;
                    lowered_conditions.push((when_expr, then_expr));
                }

                let lowered_else_expr = if let Some(el) = else_result {
                    Some(Box::new(self.convert_expr(el)?))
                } else {
                    None
                };

                Ok(Expr::Case {
                    operand: lowered_operand,
                    conditions: lowered_conditions,
                    else_expr: lowered_else_expr,
                })
            }

            SqlExpr::Subquery(subquery) => {
                // Convert subquery expression
                let converted_subquery = self.convert_query(subquery)?;
                Ok(Expr::Subquery(Box::new(converted_subquery)))
            }

            SqlExpr::InSubquery {
                expr,
                subquery,
                negated,
            } => {
                // Convert IN (SELECT ...) expression as a binary operation
                let left_expr = Box::new(self.convert_expr(expr)?);
                let subquery_expr =
                    Box::new(Expr::Subquery(Box::new(self.convert_query(subquery)?)));

                // Create a binary expression with IN operator
                Ok(Expr::Binary {
                    left: left_expr,
                    op: if *negated {
                        BinaryOp::NotIn
                    } else {
                        BinaryOp::In
                    },
                    right: subquery_expr,
                })
            }

            SqlExpr::Array(sqlparser::ast::Array { elem, named }) => {
                // Convert array literal [0.1, 0.2, ...] to Expr::Array
                let converted_elements: Result<Vec<Expr>> =
                    elem.iter().map(|e| self.convert_expr(e)).collect();

                Ok(Expr::Array {
                    elem: converted_elements?,
                    named: *named,
                })
            }

            // EXISTS (SELECT ...) / NOT EXISTS (SELECT ...)
            SqlExpr::Exists { subquery, negated } => {
                let converted_subquery = self.convert_query(subquery)?;
                Ok(Expr::Exists {
                    subquery: Box::new(converted_subquery),
                    negated: *negated,
                })
            }

            // expr LIKE pattern / expr NOT LIKE pattern
            SqlExpr::Like {
                negated,
                expr,
                pattern,
                ..
            } => {
                let left_expr = Box::new(self.convert_expr(expr)?);
                let right_expr = Box::new(self.convert_expr(pattern)?);
                Ok(Expr::Binary {
                    left: left_expr,
                    op: if *negated {
                        BinaryOp::NotLike
                    } else {
                        BinaryOp::Like
                    },
                    right: right_expr,
                })
            }

            // Case-insensitive LIKE (ILIKE) - treat as regular LIKE for now
            SqlExpr::ILike {
                negated,
                expr,
                pattern,
                ..
            } => {
                let left_expr = Box::new(self.convert_expr(expr)?);
                let right_expr = Box::new(self.convert_expr(pattern)?);
                Ok(Expr::Binary {
                    left: left_expr,
                    op: if *negated {
                        BinaryOp::NotLike
                    } else {
                        BinaryOp::Like
                    },
                    right: right_expr,
                })
            }

            // expr BETWEEN low AND high
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let converted_expr = Box::new(self.convert_expr(expr)?);
                let converted_low = Box::new(self.convert_expr(low)?);
                let converted_high = Box::new(self.convert_expr(high)?);
                Ok(Expr::Between {
                    expr: converted_expr,
                    low: converted_low,
                    high: converted_high,
                    negated: *negated,
                })
            }

            // expr IS NULL
            SqlExpr::IsNull(expr) => {
                let converted_expr = Box::new(self.convert_expr(expr)?);
                Ok(Expr::IsNull {
                    expr: converted_expr,
                    negated: false,
                })
            }

            // expr IS NOT NULL
            SqlExpr::IsNotNull(expr) => {
                let converted_expr = Box::new(self.convert_expr(expr)?);
                Ok(Expr::IsNull {
                    expr: converted_expr,
                    negated: true,
                })
            }

            // expr IN (val1, val2, ...)
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => {
                let converted_expr = Box::new(self.convert_expr(expr)?);
                let converted_list: Result<Vec<Expr>> =
                    list.iter().map(|e| self.convert_expr(e)).collect();
                Ok(Expr::InList {
                    expr: converted_expr,
                    list: converted_list?,
                    negated: *negated,
                })
            }

            // Parenthesized expression (nested)
            SqlExpr::Nested(inner) => self.convert_expr(inner),

            _ => Err(anyhow!("Unsupported expression: {:?}", expr)),
        }
    }

    fn convert_value(&self, value: &Value) -> Result<Literal> {
        match value {
            Value::Number(n, _) => {
                // Try to parse as integer first, then as float
                if let Ok(int_val) = n.parse::<i64>() {
                    Ok(Literal::Number(int_val as f64))
                } else if let Ok(float_val) = n.parse::<f64>() {
                    Ok(Literal::Number(float_val))
                } else {
                    Err(anyhow!("Invalid number: {}", n))
                }
            }
            Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
                Ok(Literal::String(s.clone()))
            }
            Value::Boolean(b) => Ok(Literal::Bool(*b)),
            Value::Null => Ok(Literal::Null),
            _ => Err(anyhow!("Unsupported literal value: {:?}", value)),
        }
    }

    fn convert_binary_op(&self, op: &BinaryOperator) -> Result<BinaryOp> {
        match op {
            BinaryOperator::Eq => Ok(BinaryOp::Eq),
            BinaryOperator::NotEq => Ok(BinaryOp::Ne),
            BinaryOperator::Lt => Ok(BinaryOp::Lt),
            BinaryOperator::LtEq => Ok(BinaryOp::Le),
            BinaryOperator::Gt => Ok(BinaryOp::Gt),
            BinaryOperator::GtEq => Ok(BinaryOp::Ge),
            BinaryOperator::And => Ok(BinaryOp::And),
            BinaryOperator::Or => Ok(BinaryOp::Or),
            BinaryOperator::Plus => Ok(BinaryOp::Add),
            BinaryOperator::Minus => Ok(BinaryOp::Sub),
            BinaryOperator::Multiply => Ok(BinaryOp::Mul),
            BinaryOperator::Divide => Ok(BinaryOp::Div),
            BinaryOperator::Modulo => Ok(BinaryOp::Mod),
            _ => Err(anyhow!("Unsupported binary operator: {:?}", op)),
        }
    }

    fn convert_unary_op(&self, op: &UnaryOperator) -> Result<UnaryOp> {
        match op {
            UnaryOperator::Not => Ok(UnaryOp::Not),
            UnaryOperator::Minus => Ok(UnaryOp::Neg),
            _ => Err(anyhow!("Unsupported unary operator: {:?}", op)),
        }
    }

    fn convert_function_arg(&self, arg: &FunctionArg) -> Result<Expr> {
        match arg {
            FunctionArg::Named { .. } => Err(anyhow!("Named function arguments not supported")),
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => self.convert_expr(expr),
            FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                Ok(Expr::Identifier("*".to_string()))
            }
            _ => Err(anyhow!("Unsupported function argument: {:?}", arg)),
        }
    }

    fn convert_order_by_expr(&self, order_expr: &SqlOrderByExpr) -> Result<OrderByExpr> {
        let expr = self.convert_expr(&order_expr.expr)?;
        // In sqlparser 0.59, asc is in options.asc
        let asc = order_expr.options.asc.unwrap_or(true); // Default to ascending

        Ok(OrderByExpr { expr, asc })
    }

    fn is_aggregate_function(&self, name: &str) -> bool {
        matches!(
            name.to_lowercase().as_str(),
            "count" | "sum" | "avg" | "min" | "max" | "stddev" | "variance"
        )
    }

    // ========================
    // DML Statement Parsing
    // ========================

    /// Parse SQL text and return a DML statement if it's INSERT/UPDATE/DELETE
    pub fn parse_dml(&self, sql: &str) -> Result<Option<DmlStatement>> {
        let statements = Parser::parse_sql(&self.dialect, sql)
            .map_err(|e| anyhow!("SQL parsing failed: {}", e))?;

        if statements.is_empty() {
            return Err(anyhow!("No SQL statements found"));
        }

        if statements.len() > 1 {
            return Err(anyhow!(
                "Multiple statements not supported, found {}",
                statements.len()
            ));
        }

        let statement = &statements[0];
        self.try_convert_dml(statement)
    }

    /// Parse a `SELECT` statement and return its `WHERE` clause as the same
    /// faithful boolean [`WhereClause`] IR that UPDATE/DELETE use — so the
    /// pgwire relational SELECT path can push OR / mixed-AND-OR / grouped
    /// predicates into the record scan instead of degrading to a full scan.
    ///
    /// Returns `Ok(None)` when the statement is a well-formed SELECT with no
    /// `WHERE`. Returns `Err` when the text is not a single SELECT, fails to
    /// parse, or its `WHERE` contains an expression the DML converter rejects;
    /// the caller treats `Err` as "fall back to the legacy string-predicate
    /// path" (over-inclusive full scan), NOT as a query error.
    pub fn parse_select_where_clause(&self, sql: &str) -> Result<Option<WhereClause>> {
        let statements = Parser::parse_sql(&self.dialect, sql)
            .map_err(|e| anyhow!("SQL parsing failed: {}", e))?;
        let [statement] = statements.as_slice() else {
            return Err(anyhow!(
                "expected exactly one statement, found {}",
                statements.len()
            ));
        };
        let Statement::Query(query) = statement else {
            return Err(anyhow!("statement is not a SELECT query"));
        };
        let SetExpr::Select(select) = &*query.body else {
            return Err(anyhow!("query body is not a simple SELECT"));
        };
        match &select.selection {
            None => Ok(None),
            Some(expr) => Ok(Some(self.convert_where_clause_dml(expr)?)),
        }
    }

    /// Try to convert a statement to DML, returning None for SELECT queries
    fn try_convert_dml(&self, statement: &Statement) -> Result<Option<DmlStatement>> {
        match statement {
            Statement::Insert(insert) => Ok(Some(self.convert_insert(insert)?)),
            Statement::Update {
                table,
                assignments,
                selection,
                ..
            } => Ok(Some(self.convert_update(table, assignments, selection)?)),
            Statement::Delete(delete) => Ok(Some(self.convert_delete(delete)?)),
            Statement::Query(_) => Ok(None), // SELECT query, not DML
            _ => Err(anyhow!("Unsupported statement type for DML")),
        }
    }

    /// Convert INSERT statement to DmlStatement
    fn convert_insert(&self, insert: &sqlparser::ast::Insert) -> Result<DmlStatement> {
        // Get table name
        let table_name = unquote_object_name(&insert.table.to_string());

        // Get column names
        let columns: Vec<String> = insert
            .columns
            .iter()
            .map(|c| unquote_identifier_text(&c.to_string()))
            .collect();

        let Some(source) = &insert.source else {
            return Err(anyhow!("INSERT requires VALUES or SELECT source"));
        };

        if !matches!(&*source.body, SetExpr::Values(_)) {
            if insert.on.is_some() {
                return Err(anyhow!(
                    "INSERT ... SELECT with ON CONFLICT is not supported yet"
                ));
            }
            let target = LogicalTableRef::new(table_name.clone());
            let source = self
                .simple_catalog_table_source(source)
                .unwrap_or_else(|| ReadSource::QuerySql(source.to_string()));
            let plan = CopyIntoPlan {
                source,
                target,
                write_mode: if insert.overwrite {
                    WriteMode::OverwriteTable
                } else {
                    WriteMode::Append
                },
                conflict_policy: if insert.overwrite {
                    ConflictPolicy::Upsert
                } else {
                    ConflictPolicy::Error
                },
                distribution: DistributionMode::Auto,
            };
            return if insert.overwrite {
                Ok(DmlStatement::InsertOverwrite { plan, columns })
            } else {
                Ok(DmlStatement::InsertSelect { plan, columns })
            };
        }

        let values = self.extract_values_from_source(source)?;

        if let Some(on_insert) = &insert.on {
            let (conflict_columns, update_assignments) =
                self.convert_insert_conflict_clause(on_insert)?;
            return Ok(DmlStatement::Upsert {
                table_name,
                columns,
                values,
                conflict_columns,
                update_assignments,
            });
        }

        Ok(DmlStatement::Insert {
            table_name,
            columns,
            values,
        })
    }

    /// Extract values from INSERT source (VALUES clause)
    fn extract_values_from_source(
        &self,
        source: &sqlparser::ast::Query,
    ) -> Result<Vec<Vec<SqlValueLiteral>>> {
        match &*source.body {
            SetExpr::Values(values) => values
                .rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|expr| self.convert_expr_to_dml_literal(expr))
                        .collect()
                })
                .collect(),
            _ => Err(anyhow!("INSERT source must be VALUES clause")),
        }
    }

    fn simple_catalog_table_source(&self, source: &SqlQuery) -> Option<ReadSource> {
        if source.with.is_some()
            || source.order_by.is_some()
            || source.limit_clause.is_some()
            || source.fetch.is_some()
            || !source.locks.is_empty()
            || source.for_clause.is_some()
            || source.settings.is_some()
            || source.format_clause.is_some()
            || !source.pipe_operators.is_empty()
        {
            return None;
        }

        let SetExpr::Select(select) = &*source.body else {
            return None;
        };
        if select.distinct.is_some()
            || select.top.is_some()
            || select.into.is_some()
            || select.prewhere.is_some()
            || select.selection.is_some()
            || select.having.is_some()
            || select.qualify.is_some()
            || select.connect_by.is_some()
            || !select.lateral_views.is_empty()
            || !select.cluster_by.is_empty()
            || !select.distribute_by.is_empty()
            || !select.sort_by.is_empty()
            || !select.named_window.is_empty()
            || select.value_table_mode.is_some()
        {
            return None;
        }
        if !matches!(&select.group_by, GroupByExpr::Expressions(exprs, modifiers) if exprs.is_empty() && modifiers.is_empty())
        {
            return None;
        }
        if !matches!(select.projection.as_slice(), [SelectItem::Wildcard(_)]) {
            return None;
        }

        let [from] = select.from.as_slice() else {
            return None;
        };
        if !from.joins.is_empty() {
            return None;
        }
        let TableFactor::Table {
            name,
            alias: None,
            args: None,
            with_hints,
            version: None,
            with_ordinality: false,
            partitions,
            json_path: None,
            sample: None,
            index_hints,
        } = &from.relation
        else {
            return None;
        };
        if !with_hints.is_empty() || !partitions.is_empty() || !index_hints.is_empty() {
            return None;
        }

        let mut parts = unquote_object_name(&name.to_string())
            .split('.')
            .map(|part| part.to_string())
            .collect::<Vec<_>>();
        let table_name = parts.pop()?;
        Some(ReadSource::CatalogTable {
            table: LogicalTableRef {
                namespace: parts,
                name: table_name,
            },
            snapshot: SnapshotRef::Latest,
        })
    }

    fn convert_insert_conflict_clause(
        &self,
        on_insert: &OnInsert,
    ) -> Result<(Vec<String>, Vec<(String, SqlValueLiteral)>)> {
        match on_insert {
            OnInsert::OnConflict(on_conflict) => {
                let conflict_columns = match &on_conflict.conflict_target {
                    Some(ConflictTarget::Columns(columns)) => columns
                        .iter()
                        .map(|column| unquote_identifier_text(&column.to_string()))
                        .collect(),
                    Some(ConflictTarget::OnConstraint(name)) => {
                        vec![unquote_object_name(&name.to_string())]
                    }
                    None => Vec::new(),
                };

                let update_assignments = match &on_conflict.action {
                    OnConflictAction::DoNothing => Vec::new(),
                    OnConflictAction::DoUpdate(update) => update
                        .assignments
                        .iter()
                        .map(|assignment| {
                            Ok((
                                self.assignment_target_to_string(&assignment.target)?,
                                self.convert_expr_to_dml_literal(&assignment.value)?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?,
                };

                Ok((conflict_columns, update_assignments))
            }
            OnInsert::DuplicateKeyUpdate(assignments) => {
                let update_assignments = assignments
                    .iter()
                    .map(|assignment| {
                        Ok((
                            self.assignment_target_to_string(&assignment.target)?,
                            self.convert_expr_to_dml_literal(&assignment.value)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((Vec::new(), update_assignments))
            }
            _ => Err(anyhow!(
                "Unsupported INSERT conflict clause: {:?}",
                on_insert
            )),
        }
    }

    /// Convert expression to SQL value literal (for DML)
    fn convert_expr_to_dml_literal(&self, expr: &SqlExpr) -> Result<SqlValueLiteral> {
        match expr {
            SqlExpr::Value(value_with_span) => {
                // In sqlparser 0.59, Value is wrapped in ValueWithSpan
                self.convert_value_to_dml_literal(&value_with_span.value)
            }
            SqlExpr::Array(arr) => {
                let elements: Result<Vec<SqlValueLiteral>> = arr
                    .elem
                    .iter()
                    .map(|e| self.convert_expr_to_dml_literal(e))
                    .collect();
                Ok(SqlValueLiteral::Array(elements?))
            }
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => {
                // Handle negative numbers
                match self.convert_expr_to_dml_literal(expr)? {
                    SqlValueLiteral::Integer(i) => Ok(SqlValueLiteral::Integer(-i)),
                    SqlValueLiteral::Float(f) => Ok(SqlValueLiteral::Float(-f)),
                    _ => Err(anyhow!("Unary minus only supported for numbers")),
                }
            }
            SqlExpr::Function(func) => {
                let name = func.name.to_string();
                // Extract function args based on sqlparser 0.59 API
                let args = self.extract_function_args_dml(&func.args)?;
                Ok(SqlValueLiteral::Function { name, args })
            }
            SqlExpr::Cast { expr, .. } => self.convert_expr_to_dml_literal(expr),
            SqlExpr::TypedString(typed) => {
                // Typed string literals: DATE '1995-02-15', TIMESTAMP '...',
                // TIME '...'. Carry the inner string; the target column's declared
                // type drives coercion, and ISO-8601 values sort chronologically as
                // strings. (TPC-H is dense with DATE literals in INSERT ... VALUES.)
                self.convert_value_to_dml_literal(&typed.value.value)
            }
            SqlExpr::Identifier(ident) => {
                // Could be DEFAULT or a column reference
                if ident.value.eq_ignore_ascii_case("DEFAULT") {
                    Ok(SqlValueLiteral::Default)
                } else {
                    Ok(SqlValueLiteral::Column(ident.value.clone()))
                }
            }
            SqlExpr::CompoundIdentifier(parts) => Ok(SqlValueLiteral::Column(
                parts
                    .iter()
                    .map(|part| unquote_identifier_text(&part.to_string()))
                    .collect::<Vec<_>>()
                    .join("."),
            )),
            _ => Err(anyhow!("Unsupported expression in VALUES: {:?}", expr)),
        }
    }

    /// Extract function arguments for DML
    fn extract_function_args_dml(
        &self,
        args: &sqlparser::ast::FunctionArguments,
    ) -> Result<Vec<SqlValueLiteral>> {
        use sqlparser::ast::FunctionArguments;
        match args {
            FunctionArguments::List(list) => list
                .args
                .iter()
                .filter_map(|arg| match arg {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => {
                        self.convert_expr_to_dml_literal(e).ok()
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(Ok)
                .collect(),
            FunctionArguments::None => Ok(Vec::new()),
            FunctionArguments::Subquery(_) => Err(anyhow!("Subquery function args not supported")),
        }
    }

    /// Convert sqlparser Value to SqlValueLiteral (for DML)
    fn convert_value_to_dml_literal(&self, value: &Value) -> Result<SqlValueLiteral> {
        match value {
            Value::Null => Ok(SqlValueLiteral::Null),
            Value::Boolean(b) => Ok(SqlValueLiteral::Boolean(*b)),
            Value::Number(n, _) => {
                // Try to parse as integer first, then float
                if let Ok(i) = n.parse::<i64>() {
                    Ok(SqlValueLiteral::Integer(i))
                } else if let Ok(f) = n.parse::<f64>() {
                    Ok(SqlValueLiteral::Float(f))
                } else {
                    Err(anyhow!("Invalid number: {}", n))
                }
            }
            Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
                Ok(SqlValueLiteral::String(s.clone()))
            }
            Value::HexStringLiteral(h) => {
                // Simple hex decode
                let bytes: Vec<u8> = h
                    .as_bytes()
                    .chunks(2)
                    .filter_map(|chunk| {
                        if chunk.len() == 2 {
                            u8::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok()
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(SqlValueLiteral::Binary(bytes))
            }
            _ => Err(anyhow!("Unsupported value type: {:?}", value)),
        }
    }

    /// Convert UPDATE statement to DmlStatement
    fn convert_update(
        &self,
        table: &sqlparser::ast::TableWithJoins,
        assignments: &[sqlparser::ast::Assignment],
        selection: &Option<SqlExpr>,
    ) -> Result<DmlStatement> {
        // Get table name
        let table_name = match &table.relation {
            TableFactor::Table { name, .. } => unquote_object_name(&name.to_string()),
            _ => return Err(anyhow!("UPDATE requires a table name")),
        };

        // Convert assignments
        let mut update_assignments = Vec::new();
        for assignment in assignments {
            let column = self.assignment_target_to_string(&assignment.target)?;
            let value = self.convert_expr_to_dml_literal(&assignment.value)?;
            update_assignments.push((column, value));
        }

        // Convert WHERE clause
        let where_clause = if let Some(selection) = selection {
            Some(self.convert_where_clause_dml(selection)?)
        } else {
            None
        };

        Ok(DmlStatement::Update {
            table_name,
            assignments: update_assignments,
            where_clause,
        })
    }

    /// Convert assignment target to string
    fn assignment_target_to_string(
        &self,
        target: &sqlparser::ast::AssignmentTarget,
    ) -> Result<String> {
        use sqlparser::ast::AssignmentTarget;
        match target {
            AssignmentTarget::ColumnName(names) => {
                // ObjectName is a newtype around Vec<Ident>, access inner with .0
                Ok(names
                    .0
                    .iter()
                    .map(|n| unquote_identifier_text(&n.to_string()))
                    .collect::<Vec<_>>()
                    .join("."))
            }
            AssignmentTarget::Tuple(cols) => {
                // For tuple assignment, join column names
                Ok(cols
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", "))
            }
        }
    }

    /// Convert DELETE statement to DmlStatement
    fn convert_delete(&self, delete: &sqlparser::ast::Delete) -> Result<DmlStatement> {
        use sqlparser::ast::FromTable;

        // Get table name from FROM clause
        let table_name = match &delete.from {
            FromTable::WithFromKeyword(tables) => {
                if let Some(first) = tables.first() {
                    match &first.relation {
                        TableFactor::Table { name, .. } => unquote_object_name(&name.to_string()),
                        _ => return Err(anyhow!("DELETE requires a table name")),
                    }
                } else {
                    return Err(anyhow!("DELETE requires FROM clause"));
                }
            }
            FromTable::WithoutKeyword(tables) => {
                if let Some(first) = tables.first() {
                    match &first.relation {
                        TableFactor::Table { name, .. } => unquote_object_name(&name.to_string()),
                        _ => return Err(anyhow!("DELETE requires a table name")),
                    }
                } else {
                    return Err(anyhow!("DELETE requires FROM clause"));
                }
            }
        };

        // Convert WHERE clause
        let where_clause = if let Some(selection) = &delete.selection {
            Some(self.convert_where_clause_dml(selection)?)
        } else {
            None
        };

        Ok(DmlStatement::Delete {
            table_name,
            where_clause,
        })
    }

    /// Convert WHERE clause expression to WhereClause (for DML)
    fn convert_where_clause_dml(&self, expr: &SqlExpr) -> Result<WhereClause> {
        let operator = self.determine_logical_operator_dml(expr);
        let conditions = self.flatten_conditions_for_op(expr, operator)?;
        Ok(WhereClause {
            conditions,
            operator,
        })
    }

    /// Flatten a boolean WHERE expression into the operands of `target_op`,
    /// preserving any differently-combined subtree as a single
    /// `Condition::Nested`. This keeps the boolean tree faithful — e.g.
    /// `A OR (B AND C)` → `[A, Nested{And,[B,C]}]` — instead of flattening every
    /// AND/OR into one list (which silently mis-evaluated mixed clauses).
    fn flatten_conditions_for_op(
        &self,
        expr: &SqlExpr,
        target_op: LogicalOperator,
    ) -> Result<Vec<Condition>> {
        match expr {
            SqlExpr::Nested(inner) => self.flatten_conditions_for_op(inner, target_op),
            SqlExpr::BinaryOp { left, op, right }
                if Self::binary_op_matches_logical(op, target_op) =>
            {
                let mut conditions = self.flatten_conditions_for_op(left, target_op)?;
                conditions.extend(self.flatten_conditions_for_op(right, target_op)?);
                Ok(conditions)
            }
            _ => Ok(vec![self.expr_to_condition_dml(expr)?]),
        }
    }

    /// Convert one `SqlExpr` into a single `Condition` — a `Nested` node for an
    /// AND/OR subtree, a leaf otherwise.
    fn expr_to_condition_dml(&self, expr: &SqlExpr) -> Result<Condition> {
        match expr {
            SqlExpr::Nested(inner) => self.expr_to_condition_dml(inner),
            SqlExpr::BinaryOp {
                op: BinaryOperator::And,
                ..
            } => Ok(Condition::Nested {
                conditions: self.flatten_conditions_for_op(expr, LogicalOperator::And)?,
                operator: LogicalOperator::And,
            }),
            SqlExpr::BinaryOp {
                op: BinaryOperator::Or,
                ..
            } => Ok(Condition::Nested {
                conditions: self.flatten_conditions_for_op(expr, LogicalOperator::Or)?,
                operator: LogicalOperator::Or,
            }),
            SqlExpr::BinaryOp { left, op, right } => {
                let column = self.expr_to_column_name_dml(left)?;
                let operator = self.convert_comparison_op_dml(op)?;
                let value = self.convert_expr_to_dml_literal(right)?;
                Ok(Condition::Comparison {
                    column,
                    operator,
                    value,
                })
            }
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => {
                let column = self.expr_to_column_name_dml(expr)?;
                let values = list
                    .iter()
                    .map(|e| self.convert_expr_to_dml_literal(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Condition::In {
                    column,
                    values,
                    negated: *negated,
                })
            }
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let column = self.expr_to_column_name_dml(expr)?;
                Ok(Condition::Between {
                    column,
                    low: self.convert_expr_to_dml_literal(low)?,
                    high: self.convert_expr_to_dml_literal(high)?,
                    negated: *negated,
                })
            }
            SqlExpr::IsNull(inner) => Ok(Condition::IsNull {
                column: self.expr_to_column_name_dml(inner)?,
                negated: false,
            }),
            SqlExpr::IsNotNull(inner) => Ok(Condition::IsNull {
                column: self.expr_to_column_name_dml(inner)?,
                negated: true,
            }),
            SqlExpr::Like {
                expr,
                pattern,
                negated,
                ..
            } => Ok(Condition::Like {
                column: self.expr_to_column_name_dml(expr)?,
                pattern: self.extract_like_pattern(pattern)?,
                negated: *negated,
            }),
            _ => Err(anyhow!("Unsupported WHERE expression: {:?}", expr)),
        }
    }

    fn binary_op_matches_logical(op: &BinaryOperator, logical: LogicalOperator) -> bool {
        matches!(
            (op, logical),
            (BinaryOperator::And, LogicalOperator::And) | (BinaryOperator::Or, LogicalOperator::Or)
        )
    }

    /// Extract LIKE pattern string
    fn extract_like_pattern(&self, pattern: &SqlExpr) -> Result<String> {
        match pattern {
            SqlExpr::Value(value_with_span) => match &value_with_span.value {
                Value::SingleQuotedString(s) => Ok(s.clone()),
                Value::DoubleQuotedString(s) => Ok(s.clone()),
                _ => Err(anyhow!("LIKE pattern must be a string")),
            },
            _ => Err(anyhow!("LIKE pattern must be a string literal")),
        }
    }

    /// Extract column name from expression (for DML)
    fn expr_to_column_name_dml(&self, expr: &SqlExpr) -> Result<String> {
        match expr {
            SqlExpr::Identifier(ident) => Ok(ident.value.clone()),
            SqlExpr::CompoundIdentifier(parts) => Ok(parts
                .iter()
                .map(|p| p.value.clone())
                .collect::<Vec<_>>()
                .join(".")),
            _ => Err(anyhow!("Expected column name, got {:?}", expr)),
        }
    }

    /// Convert comparison operator (for DML)
    fn convert_comparison_op_dml(&self, op: &BinaryOperator) -> Result<DmlComparisonOperator> {
        match op {
            BinaryOperator::Eq => Ok(DmlComparisonOperator::Equal),
            BinaryOperator::NotEq => Ok(DmlComparisonOperator::NotEqual),
            BinaryOperator::Lt => Ok(DmlComparisonOperator::LessThan),
            BinaryOperator::LtEq => Ok(DmlComparisonOperator::LessThanOrEqual),
            BinaryOperator::Gt => Ok(DmlComparisonOperator::GreaterThan),
            BinaryOperator::GtEq => Ok(DmlComparisonOperator::GreaterThanOrEqual),
            _ => Err(anyhow!("Not a comparison operator: {:?}", op)),
        }
    }

    /// Determine the logical operator from an expression (for DML)
    fn determine_logical_operator_dml(&self, expr: &SqlExpr) -> LogicalOperator {
        match expr {
            SqlExpr::BinaryOp {
                op: BinaryOperator::Or,
                ..
            } => LogicalOperator::Or,
            _ => LogicalOperator::And, // Default to AND
        }
    }
}

impl Default for SqlFrontendParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ddl_alter_table_set_data_type_supports_jsonb() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_ddl("ALTER TABLE demo ALTER COLUMN payload SET DATA TYPE JSONB;")
            .expect("expected ddl parse to succeed")
            .expect("expected alter table ddl");

        if let DdlStatement::AlterTable {
            table_name,
            changes,
            ..
        } = statement
        {
            assert_eq!(table_name, "demo");
            assert_eq!(changes.len(), 1);

            match &changes[0] {
                AlterTableChange::ChangeType {
                    column_name,
                    new_type,
                } => {
                    assert_eq!(column_name, "payload");
                    assert!(matches!(new_type, SqlDataType::Jsonb));
                }
                _ => panic!("expected change type for JSONB"),
            }
        } else {
            panic!("expected alter table statement");
        }
    }

    #[test]
    fn parse_ddl_alter_table_normalizes_quoted_identifiers() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_ddl(
                "ALTER TABLE \"Tenant One\".\"Event Store\" \
                 ALTER COLUMN \"Payload Field\" SET DATA TYPE JSONB;",
            )
            .expect("expected ddl parse to succeed")
            .expect("expected alter table ddl");

        match statement {
            DdlStatement::AlterTable {
                table_name,
                changes,
            } => {
                assert_eq!(table_name, "Tenant One.Event Store");
                assert_eq!(changes.len(), 1);
                match &changes[0] {
                    AlterTableChange::ChangeType {
                        column_name,
                        new_type,
                    } => {
                        assert_eq!(column_name, "Payload Field");
                        assert!(matches!(new_type, SqlDataType::Jsonb));
                    }
                    other => panic!("expected ChangeType, got {other:?}"),
                }
            }
            other => panic!("expected AlterTable, got {other:?}"),
        }

        let statement = parser
            .parse_ddl(
                "ALTER TABLE \"Tenant One\".\"Event Store\" \
                 RENAME COLUMN \"Payload Field\" TO \"Payload Body\";",
            )
            .expect("expected ddl parse to succeed")
            .expect("expected alter table ddl");

        match statement {
            DdlStatement::AlterTable {
                table_name,
                changes,
            } => {
                assert_eq!(table_name, "Tenant One.Event Store");
                assert_eq!(changes.len(), 1);
                match &changes[0] {
                    AlterTableChange::RenameColumn { old_name, new_name } => {
                        assert_eq!(old_name, "Payload Field");
                        assert_eq!(new_name, "Payload Body");
                    }
                    other => panic!("expected RenameColumn, got {other:?}"),
                }
            }
            other => panic!("expected AlterTable, got {other:?}"),
        }

        let statement = parser
            .parse_ddl(
                "ALTER TABLE \"Tenant One\".\"Event Store\" \
                 ADD CONSTRAINT \"Unique Payload\" UNIQUE (\"Payload Field\");",
            )
            .expect("expected ddl parse to succeed")
            .expect("expected alter table ddl");

        match statement {
            DdlStatement::AlterTable {
                table_name,
                changes,
            } => {
                assert_eq!(table_name, "Tenant One.Event Store");
                assert_eq!(changes.len(), 1);
                match &changes[0] {
                    AlterTableChange::AddConstraint {
                        constraint_name,
                        constraint: TableConstraint::Unique { columns },
                    } => {
                        assert_eq!(constraint_name.as_deref(), Some("Unique Payload"));
                        assert_eq!(columns, &vec!["Payload Field".to_string()]);
                    }
                    other => panic!("expected UNIQUE AddConstraint, got {other:?}"),
                }
            }
            other => panic!("expected AlterTable, got {other:?}"),
        }

        let statement = parser
            .parse_ddl(
                "ALTER TABLE \"Tenant One\".\"Event Store\" \
                 ADD CONSTRAINT \"FK Payload Parent\" FOREIGN KEY (\"Payload Field\") \
                 REFERENCES \"Tenant One\".\"Parent Store\" (\"Parent Id\");",
            )
            .expect("expected ddl parse to succeed")
            .expect("expected alter table ddl");

        match statement {
            DdlStatement::AlterTable {
                table_name,
                changes,
            } => {
                assert_eq!(table_name, "Tenant One.Event Store");
                assert_eq!(changes.len(), 1);
                match &changes[0] {
                    AlterTableChange::AddConstraint {
                        constraint_name,
                        constraint:
                            TableConstraint::ForeignKey {
                                columns,
                                references_table,
                                references_columns,
                                ..
                            },
                    } => {
                        assert_eq!(constraint_name.as_deref(), Some("FK Payload Parent"));
                        assert_eq!(columns, &vec!["Payload Field".to_string()]);
                        assert_eq!(references_table, "Tenant One.Parent Store");
                        assert_eq!(references_columns, &vec!["Parent Id".to_string()]);
                    }
                    other => panic!("expected FOREIGN KEY AddConstraint, got {other:?}"),
                }
            }
            other => panic!("expected AlterTable, got {other:?}"),
        }

        let statement = parser
            .parse_ddl(
                "ALTER TABLE \"Tenant One\".\"Event Store\" \
                 DROP CONSTRAINT \"Unique Payload\";",
            )
            .expect("expected ddl parse to succeed")
            .expect("expected alter table ddl");

        match statement {
            DdlStatement::AlterTable {
                table_name,
                changes,
            } => {
                assert_eq!(table_name, "Tenant One.Event Store");
                assert_eq!(changes.len(), 1);
                match &changes[0] {
                    AlterTableChange::DropConstraint { constraint_name } => {
                        assert_eq!(constraint_name, "Unique Payload");
                    }
                    other => panic!("expected DropConstraint, got {other:?}"),
                }
            }
            other => panic!("expected AlterTable, got {other:?}"),
        }
    }

    #[test]
    fn parse_ddl_create_table_supports_jsonb() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_ddl("CREATE TABLE demo (payload JSONB);")
            .expect("expected ddl parse to succeed")
            .expect("expected create table ddl");

        if let DdlStatement::CreateTable {
            table_name,
            columns,
            ..
        } = statement
        {
            assert_eq!(table_name, "demo");
            assert_eq!(columns.len(), 1);
            assert_eq!(columns[0].name, "payload");
            assert!(matches!(columns[0].data_type, SqlDataType::Jsonb));
        } else {
            panic!("expected create table statement");
        }
    }

    #[test]
    fn parse_ddl_create_table_supports_jsonb_with_additional_columns() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_ddl("CREATE TABLE demo (id INT, payload JSONB);")
            .expect("expected ddl parse to succeed")
            .expect("expected create table ddl");

        if let DdlStatement::CreateTable {
            table_name,
            columns,
            ..
        } = statement
        {
            assert_eq!(table_name, "demo");
            assert_eq!(columns.len(), 2);
            assert_eq!(columns[0].name, "id");
            assert_eq!(columns[1].name, "payload");
            assert!(matches!(columns[1].data_type, SqlDataType::Jsonb));
        } else {
            panic!("expected create table statement");
        }
    }

    #[test]
    fn parse_ddl_create_table_lowers_catalog_options_and_table_primary_key() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_ddl(
                "CREATE TABLE IF NOT EXISTS \"agent_store\" (
                    \"record_id\" TEXT NOT NULL,
                    \"payload\" JSONB NOT NULL DEFAULT '{}'::jsonb,
                    \"embedding\" VECTOR(384),
                    PRIMARY KEY (\"record_id\")
                ) WITH (
                    storage_engine = 'VIPER',
                    layout = 'columnar',
                    xcatalog_namespace = 'agentic.demo',
                    schema_kind = 'agentic_mixed'
                );",
            )
            .expect("expected ddl parse to succeed")
            .expect("expected create table ddl");

        if let DdlStatement::CreateTable {
            table_name,
            columns,
            if_not_exists,
            properties,
            ..
        } = statement
        {
            assert_eq!(table_name, "agent_store");
            assert!(if_not_exists);
            assert_eq!(
                properties.get("storage_engine").map(String::as_str),
                Some("VIPER")
            );
            assert_eq!(
                properties.get("layout").map(String::as_str),
                Some("columnar")
            );
            assert_eq!(
                properties.get("xcatalog_namespace").map(String::as_str),
                Some("agentic.demo")
            );
            let record_id = columns
                .iter()
                .find(|column| column.name == "record_id")
                .expect("record_id column should parse");
            assert!(record_id.primary_key);
            assert!(!record_id.nullable);
            assert!(matches!(
                columns
                    .iter()
                    .find(|column| column.name == "payload")
                    .expect("payload column")
                    .data_type,
                SqlDataType::Jsonb
            ));
            assert!(matches!(
                columns
                    .iter()
                    .find(|column| column.name == "embedding")
                    .expect("embedding column")
                    .data_type,
                SqlDataType::Vector { dimension: 384 }
            ));
        } else {
            panic!("expected create table statement");
        }
    }

    #[test]
    fn parse_ddl_create_table_supports_tpcc_constraints() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_ddl(
                "CREATE TABLE customer (
                    c_w_id int NOT NULL,
                    c_d_id int NOT NULL,
                    c_id int NOT NULL,
                    c_credit char(2) NOT NULL,
                    c_since timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (c_w_id, c_d_id) REFERENCES district (d_w_id, d_id) ON DELETE CASCADE,
                    PRIMARY KEY (c_w_id, c_d_id, c_id),
                    UNIQUE (c_w_id, c_d_id, c_id)
                );",
            )
            .expect("expected tpcc ddl parse to succeed")
            .expect("expected create table ddl");

        let DdlStatement::CreateTable {
            table_name,
            columns,
            constraints,
            ..
        } = statement
        else {
            panic!("expected create table statement");
        };

        assert_eq!(table_name, "customer");
        assert!(matches!(
            columns
                .iter()
                .find(|column| column.name == "c_credit")
                .expect("char column")
                .data_type,
            SqlDataType::Varchar {
                max_length: Some(2)
            }
        ));
        assert_eq!(
            columns
                .iter()
                .filter(|column| column.primary_key)
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            vec!["c_w_id", "c_d_id", "c_id"]
        );
        assert!(constraints.iter().any(|constraint| matches!(
            constraint,
            TableConstraint::ForeignKey {
                columns,
                references_table,
                references_columns,
                ..
            } if columns == &vec!["c_w_id".to_string(), "c_d_id".to_string()]
                && references_table == "district"
                && references_columns == &vec!["d_w_id".to_string(), "d_id".to_string()]
        )));
        assert!(constraints.iter().any(|constraint| matches!(
            constraint,
            TableConstraint::Unique { columns }
                if columns == &vec![
                    "c_w_id".to_string(),
                    "c_d_id".to_string(),
                    "c_id".to_string()
                ]
        )));
    }

    #[test]
    fn parse_dml_insert_supports_jsonb_cast_default_and_vector_literal() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml(
                "INSERT INTO \"agent_store\" (\"record_id\", payload, embedding)
                 VALUES ('r1', '{}'::jsonb, '[0.1, 0.2]'::vector(2)),
                        ('r2', DEFAULT, '[0.3, 0.4]'::vector);",
            )
            .expect("expected dml parse to succeed")
            .expect("expected insert dml");

        match statement {
            DmlStatement::Insert {
                table_name,
                columns,
                values,
            } => {
                assert_eq!(table_name, "agent_store");
                assert_eq!(columns, vec!["record_id", "payload", "embedding"]);
                assert_eq!(values.len(), 2);
                assert!(matches!(values[0][1], SqlValueLiteral::String(_)));
                assert!(matches!(values[1][1], SqlValueLiteral::Default));
                assert!(matches!(values[0][2], SqlValueLiteral::String(_)));
            }
            other => panic!("expected insert statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_upsert_preserves_conflict_columns_and_update_assignments() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml(
                "INSERT INTO \"agent_store\" (\"record_id\", payload, embedding)
                 VALUES ('r1', '{\"kind\":\"memory\"}'::jsonb, '[0.1, 0.2]'::vector(2))
                 ON CONFLICT (\"record_id\") DO UPDATE
                 SET \"payload\" = excluded.payload,
                     \"embedding\" = excluded.embedding;",
            )
            .expect("expected dml parse to succeed")
            .expect("expected upsert dml");

        match statement {
            DmlStatement::Upsert {
                table_name,
                columns,
                conflict_columns,
                update_assignments,
                ..
            } => {
                assert_eq!(table_name, "agent_store");
                assert_eq!(columns, vec!["record_id", "payload", "embedding"]);
                assert_eq!(conflict_columns, vec!["record_id"]);
                assert_eq!(update_assignments.len(), 2);
                assert_eq!(update_assignments[0].0, "payload");
                assert!(matches!(
                    &update_assignments[0].1,
                    SqlValueLiteral::Column(column) if column == "excluded.payload"
                ));
                assert_eq!(update_assignments[1].0, "embedding");
                assert!(matches!(
                    &update_assignments[1].1,
                    SqlValueLiteral::Column(column) if column == "excluded.embedding"
                ));
            }
            other => panic!("expected upsert statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_upsert_do_nothing_preserves_conflict_columns() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml(
                "INSERT INTO agent_store (record_id, payload)
                 VALUES ('r1', DEFAULT)
                 ON CONFLICT (record_id) DO NOTHING;",
            )
            .expect("expected dml parse to succeed")
            .expect("expected upsert dml");

        match statement {
            DmlStatement::Upsert {
                conflict_columns,
                update_assignments,
                ..
            } => {
                assert_eq!(conflict_columns, vec!["record_id"]);
                assert!(update_assignments.is_empty());
            }
            other => panic!("expected upsert statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_insert_select_lowers_to_copy_plan() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml(
                "INSERT INTO facts (id, payload)
                 SELECT id, payload FROM staging WHERE tenant_id = 'acme';",
            )
            .expect("expected dml parse to succeed")
            .expect("expected insert-select dml");

        match statement {
            DmlStatement::InsertSelect { plan, columns } => {
                assert_eq!(columns, vec!["id", "payload"]);
                assert_eq!(plan.target.name, "facts");
                assert!(matches!(
                    plan.write_mode,
                    crate::query::table_write_plan::WriteMode::Append
                ));
                match plan.source {
                    crate::query::table_write_plan::ReadSource::QuerySql(sql) => {
                        assert!(sql.contains("SELECT id, payload FROM staging"));
                    }
                    other => panic!("expected query source, got {other:?}"),
                }
            }
            other => panic!("expected insert-select statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_simple_insert_select_star_lowers_to_catalog_table_source() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml("INSERT INTO facts SELECT * FROM staging;")
            .expect("expected dml parse to succeed")
            .expect("expected insert-select dml");

        match statement {
            DmlStatement::InsertSelect { plan, columns } => {
                assert!(columns.is_empty());
                assert_eq!(plan.target.name, "facts");
                match plan.source {
                    crate::query::table_write_plan::ReadSource::CatalogTable {
                        table,
                        snapshot,
                    } => {
                        assert_eq!(table.name, "staging");
                        assert!(table.namespace.is_empty());
                        assert!(matches!(
                            snapshot,
                            crate::query::table_write_plan::SnapshotRef::Latest
                        ));
                    }
                    other => panic!("expected catalog-table source, got {other:?}"),
                }
            }
            other => panic!("expected insert-select statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_insert_overwrite_select_lowers_to_overwrite_plan() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml("INSERT OVERWRITE facts SELECT * FROM staging;")
            .expect("expected dml parse to succeed")
            .expect("expected insert-overwrite dml");

        match statement {
            DmlStatement::InsertOverwrite { plan, columns } => {
                assert!(columns.is_empty());
                assert_eq!(plan.target.name, "facts");
                assert!(matches!(
                    plan.write_mode,
                    crate::query::table_write_plan::WriteMode::OverwriteTable
                ));
            }
            other => panic!("expected insert-overwrite statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_update_supports_quoted_catalog_names_jsonb_and_vector_literal() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml(
                "UPDATE \"agent_store\"
                 SET \"payload\" = '{\"kind\":\"updated\"}'::jsonb,
                     \"embedding\" = '[0.1, 0.2]'::vector(2)
                 WHERE \"record_id\" = 'r1';",
            )
            .expect("expected dml parse to succeed")
            .expect("expected update dml");

        match statement {
            DmlStatement::Update {
                table_name,
                assignments,
                where_clause: Some(where_clause),
            } => {
                assert_eq!(table_name, "agent_store");
                assert_eq!(assignments.len(), 2);
                assert_eq!(assignments[0].0, "payload");
                assert!(matches!(assignments[0].1, SqlValueLiteral::String(_)));
                assert_eq!(assignments[1].0, "embedding");
                assert!(matches!(assignments[1].1, SqlValueLiteral::String(_)));
                match &where_clause.conditions[0] {
                    Condition::Comparison { column, value, .. } => {
                        assert_eq!(column, "record_id");
                        assert!(matches!(value, SqlValueLiteral::String(id) if id == "r1"));
                    }
                    other => panic!("expected comparison condition, got {other:?}"),
                }
            }
            other => panic!("expected update statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_delete_preserves_catalog_primary_key_column() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml("DELETE FROM agent_store WHERE record_id IN ('r1', 'r2');")
            .expect("expected dml parse to succeed")
            .expect("expected delete dml");

        match statement {
            DmlStatement::Delete {
                table_name,
                where_clause: Some(where_clause),
            } => {
                assert_eq!(table_name, "agent_store");
                match &where_clause.conditions[0] {
                    Condition::In {
                        column,
                        values,
                        negated,
                    } => {
                        assert_eq!(column, "record_id");
                        assert_eq!(values.len(), 2);
                        assert!(!negated);
                    }
                    other => panic!("expected IN condition, got {other:?}"),
                }
            }
            other => panic!("expected delete statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_dml_delete_unquotes_catalog_table_name() {
        let parser = SqlFrontendParser::new();

        let statement = parser
            .parse_dml("DELETE FROM \"agent_store\" WHERE record_id = 'r1';")
            .expect("expected dml parse to succeed")
            .expect("expected delete dml");

        match statement {
            DmlStatement::Delete { table_name, .. } => {
                assert_eq!(table_name, "agent_store");
            }
            other => panic!("expected delete statement, got {other:?}"),
        }
    }

    #[test]
    fn parse_select_where_clause_preserves_or_and_nested_grouping() {
        let parser = SqlFrontendParser::new();

        // Flat OR: faithful top-level Or operator + two conditions.
        let wc = parser
            .parse_select_where_clause("SELECT id FROM inv WHERE a = 1 OR b = 2")
            .expect("parse")
            .expect("has WHERE");
        assert!(matches!(wc.operator, LogicalOperator::Or));
        assert_eq!(wc.conditions.len(), 2);

        // Mixed AND/OR must NOT flatten: `a AND (b OR c)` →
        // top-level And of [Comparison(a), Nested{Or,[b,c]}].
        let wc = parser
            .parse_select_where_clause("SELECT id FROM inv WHERE a = 1 AND (b = 2 OR c = 3)")
            .expect("parse")
            .expect("has WHERE");
        assert!(matches!(wc.operator, LogicalOperator::And));
        assert_eq!(wc.conditions.len(), 2);
        assert!(matches!(wc.conditions[0], Condition::Comparison { .. }));
        match &wc.conditions[1] {
            Condition::Nested {
                conditions,
                operator,
            } => {
                assert!(matches!(operator, LogicalOperator::Or));
                assert_eq!(conditions.len(), 2);
            }
            other => panic!("expected nested OR subtree, got {other:?}"),
        }
    }

    #[test]
    fn parse_select_where_clause_handles_no_where_and_non_select() {
        let parser = SqlFrontendParser::new();

        // Well-formed SELECT with no WHERE → Ok(None) (scan all).
        assert!(
            parser
                .parse_select_where_clause("SELECT id FROM inv")
                .expect("parse")
                .is_none()
        );

        // Non-SELECT statement → Err so the caller falls back to the legacy
        // string-predicate path rather than treating it as a query error.
        assert!(
            parser
                .parse_select_where_clause("UPDATE inv SET status = 'x' WHERE id = '1'")
                .is_err()
        );
    }

    #[test]
    fn parse_ddl_create_index_supports_default_btree() {
        let parser = SqlFrontendParser::new();

        let cases = [
            (
                "CREATE INDEX demo_payload_idx ON demo (payload);",
                "demo_payload_idx",
                "demo",
                vec!["payload".to_string()],
                false,
            ),
            (
                "CREATE INDEX IF NOT EXISTS \"Payload Index\" ON \"Tenant One\".\"Event Store\" (\"Payload Field\");",
                "Payload Index",
                "Tenant One.Event Store",
                vec!["Payload Field".to_string()],
                true,
            ),
        ];

        for (
            sql,
            expected_index_name,
            expected_table_name,
            expected_columns,
            expected_if_not_exists,
        ) in cases
        {
            let statement = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"))
                .expect("expected create index ddl");

            if let DdlStatement::CreateIndex {
                index_name,
                table_name,
                columns,
                index_type,
                if_not_exists,
                ..
            } = statement
            {
                assert_eq!(
                    index_name, expected_index_name,
                    "index mismatch for `{sql}`"
                );
                assert_eq!(
                    table_name, expected_table_name,
                    "table mismatch for `{sql}`"
                );
                assert_eq!(columns, expected_columns, "columns mismatch for `{sql}`");
                assert!(matches!(index_type, IndexType::BTree));
                assert_eq!(
                    if_not_exists, expected_if_not_exists,
                    "IF NOT EXISTS mismatch for `{sql}`"
                );
            } else {
                panic!("expected create index statement");
            }
        }
    }

    #[test]
    fn parse_ddl_create_index_supports_gin_and_hnsw() {
        let parser = SqlFrontendParser::new();

        let gin = parser
            .parse_ddl("CREATE INDEX idx_payload ON agent_store USING GIN (payload);")
            .expect("gin parse should succeed")
            .expect("expected create index");
        let hnsw = parser
            .parse_ddl("CREATE INDEX idx_embedding ON agent_store USING HNSW (embedding);")
            .expect("hnsw parse should succeed")
            .expect("expected create index");

        assert!(matches!(
            gin,
            DdlStatement::CreateIndex {
                index_type: IndexType::Gin,
                ..
            }
        ));
        assert!(matches!(
            hnsw,
            DdlStatement::CreateIndex {
                index_type: IndexType::Hnsw { .. },
                ..
            }
        ));
    }

    #[test]
    fn parse_ddl_drop_table_supports_table_name() {
        let parser = SqlFrontendParser::new();

        let cases = [
            ("DROP TABLE demo;", "demo", false, false),
            (
                "DROP TABLE IF EXISTS \"Tenant One\".\"Event Store\";",
                "Tenant One.Event Store",
                true,
                false,
            ),
        ];

        for (sql, expected_table_name, expected_if_exists, expected_purge) in cases {
            let statement = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"))
                .expect("expected drop table ddl");

            if let DdlStatement::DropTable {
                table_name,
                if_exists,
                purge,
            } = statement
            {
                assert_eq!(
                    table_name, expected_table_name,
                    "table mismatch for `{sql}`"
                );
                assert_eq!(
                    if_exists, expected_if_exists,
                    "IF EXISTS mismatch for `{sql}`"
                );
                assert_eq!(purge, expected_purge, "PURGE mismatch for `{sql}`");
            } else {
                panic!("expected drop table statement");
            }
        }
    }

    #[test]
    fn parse_ddl_create_namespace_routes_through_pre_parser() {
        let parser = SqlFrontendParser::new();

        let cases = [
            ("CREATE NAMESPACE analytics", vec!["analytics"], false),
            (
                "CREATE NAMESPACE IF NOT EXISTS acmecorp.analytics;",
                vec!["acmecorp", "analytics"],
                true,
            ),
            (
                "create namespace if not exists \"Tenant One\".\"Event Store\"",
                vec!["Tenant One", "Event Store"],
                true,
            ),
        ];

        for (sql, expected_namespace, expected_if_not_exists) in cases {
            let statement = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"))
                .expect("expected create namespace ddl");

            match statement {
                DdlStatement::CreateNamespace {
                    namespace,
                    if_not_exists,
                    properties,
                } => {
                    assert_eq!(
                        namespace, expected_namespace,
                        "namespace mismatch for `{sql}`"
                    );
                    assert_eq!(
                        if_not_exists, expected_if_not_exists,
                        "IF NOT EXISTS mismatch for `{sql}`"
                    );
                    assert!(properties.is_empty());
                }
                other => panic!("expected CreateNamespace for `{sql}`, got {:?}", other),
            }
        }
    }

    #[test]
    fn parse_ddl_drop_namespace_routes_through_pre_parser() {
        let parser = SqlFrontendParser::new();

        let cases = [
            ("DROP NAMESPACE analytics", vec!["analytics"], false, false),
            (
                "DROP NAMESPACE IF EXISTS acmecorp.analytics CASCADE;",
                vec!["acmecorp", "analytics"],
                true,
                true,
            ),
            (
                "drop namespace if exists \"Tenant One\".\"Event Store\" restrict",
                vec!["Tenant One", "Event Store"],
                true,
                false,
            ),
        ];

        for (sql, expected_namespace, expected_if_exists, expected_cascade) in cases {
            let statement = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"))
                .expect("expected drop namespace ddl");

            match statement {
                DdlStatement::DropNamespace {
                    namespace,
                    if_exists,
                    cascade,
                } => {
                    assert_eq!(
                        namespace, expected_namespace,
                        "namespace mismatch for `{sql}`"
                    );
                    assert_eq!(
                        if_exists, expected_if_exists,
                        "IF EXISTS mismatch for `{sql}`"
                    );
                    assert_eq!(cascade, expected_cascade, "CASCADE mismatch for `{sql}`");
                }
                other => panic!("expected DropNamespace for `{sql}`, got {:?}", other),
            }
        }
    }

    #[test]
    fn parse_ddl_alter_table_materialize_routes_through_pre_parser() {
        let parser = SqlFrontendParser::new();

        for sql in [
            "ALTER TABLE inv MATERIALIZE",
            "alter table \"inv\" materialize;",
            "ALTER TABLE \"Tenant One\".\"Event Store\" MATERIALIZE",
            "ALTER TABLE \"tenant.with.dot\".\"events.with.dot\" MATERIALIZE",
        ] {
            let statement = parser
                .parse_ddl(sql)
                .expect("expected ddl parse to succeed")
                .expect("expected materialize ddl");
            match statement {
                DdlStatement::MaterializeTable { name } => {
                    let expected = if sql.contains("Tenant One") {
                        "Tenant One.Event Store"
                    } else if sql.contains("tenant.with.dot") {
                        "tenant.with.dot.events.with.dot"
                    } else {
                        "inv"
                    };
                    assert_eq!(name, expected, "table mismatch for `{sql}`");
                }
                other => panic!("expected MaterializeTable, got {:?}", other),
            }
        }

        // Not a MATERIALIZE statement → the pre-parser declines (sqlparser handles it).
        assert!(matches!(
            parser.parse_ddl("DROP TABLE inv;"),
            Ok(Some(DdlStatement::DropTable { .. }))
        ));
        // Malformed MATERIALIZE (extra tokens) → pre-parser declines, sqlparser then errors.
        assert!(parser.parse_ddl("ALTER TABLE inv MATERIALIZE NOW").is_err());
    }

    #[test]
    fn parse_ddl_drop_index_supports_drop_index() {
        let parser = SqlFrontendParser::new();

        let cases = [
            (
                "DROP INDEX demo_payload_idx ON demo;",
                "demo_payload_idx",
                "demo",
                false,
            ),
            (
                "DROP INDEX IF EXISTS \"Payload Index\" ON \"Tenant One\".\"Event Store\";",
                "Payload Index",
                "Tenant One.Event Store",
                true,
            ),
        ];

        for (sql, expected_index_name, expected_table_name, expected_if_exists) in cases {
            let statement = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"))
                .expect("expected drop index ddl");

            if let DdlStatement::DropIndex {
                index_name,
                table_name,
                if_exists,
            } = statement
            {
                assert_eq!(
                    index_name, expected_index_name,
                    "index mismatch for `{sql}`"
                );
                assert_eq!(
                    table_name, expected_table_name,
                    "table mismatch for `{sql}`"
                );
                assert_eq!(
                    if_exists, expected_if_exists,
                    "IF EXISTS mismatch for `{sql}`"
                );
            } else {
                panic!("expected drop index statement");
            }
        }
    }

    #[test]
    fn parse_ddl_unsupported_statements_still_fail_fast() {
        let parser = SqlFrontendParser::new();
        let unsupported = [
            (
                "CREATE TABLE demo AS SELECT * FROM events;",
                "CREATE TABLE with query/LIKE/CLONE clauses is not supported",
                true,
            ),
            ("DROP TABLE;", "sql parser error", false),
            (
                "CREATE TABLE demo LIKE users;",
                "CREATE TABLE with query/LIKE/CLONE clauses is not supported",
                true,
            ),
            (
                "CREATE TABLE demo CLONE users;",
                "CREATE TABLE with query/LIKE/CLONE clauses is not supported",
                true,
            ),
            ("DROP TABLESPACE sample_space;", "sql parser error", false),
            ("DROP VIEW demo;", "Unsupported DROP object type", false),
            (
                "DROP INDEX demo_payload_idx;",
                "DROP INDEX requires a table name",
                true,
            ),
        ];

        for (sql, expected, exact) in unsupported {
            let err = parser
                .parse_ddl(sql)
                .expect_err("expected unsupported ddl to be rejected");
            let err_msg = err.to_string();
            if exact {
                assert_eq!(err_msg, expected, "unexpected parse error for `{sql}`");
            } else {
                assert!(
                    err_msg.contains(expected),
                    "unexpected parse error for `{sql}`: {err_msg}"
                );
            }
        }

        let non_ddl = ["INSERT INTO users (id) VALUES (1);", "SELECT * FROM users;"];

        for sql in non_ddl {
            assert!(
                parser.parse_ddl(sql).is_ok_and(|s| s.is_none()),
                "expected `{sql}` to parse as non-DDL (`None`)"
            );
        }
    }

    #[test]
    fn parse_ddl_promote_props_key_basic_types() {
        let parser = SqlFrontendParser::new();

        let cases: &[(&str, &str, &str, &str)] = &[
            (
                "ALTER TABLE events PROMOTE PROPS KEY user_id TYPE BIGINT",
                "events",
                "user_id",
                "BigInt",
            ),
            (
                "ALTER TABLE events PROMOTE PROPS KEY label TYPE TEXT;",
                "events",
                "label",
                "Text",
            ),
            (
                "ALTER TABLE logs PROMOTE PROPS KEY score TYPE FLOAT",
                "logs",
                "score",
                "Float",
            ),
            (
                "ALTER TABLE docs PROMOTE PROPS KEY meta TYPE JSONB",
                "docs",
                "meta",
                "Jsonb",
            ),
            (
                "ALTER TABLE \"Tenant One\".\"Event Store\" PROMOTE PROPS KEY payload TYPE JSONB",
                "Tenant One.Event Store",
                "payload",
                "Jsonb",
            ),
            (
                "ALTER TABLE \"tenant.with.dot\".\"events.with.dot\" PROMOTE PROPS KEY payload TYPE JSONB",
                "tenant.with.dot.events.with.dot",
                "payload",
                "Jsonb",
            ),
        ];

        for (sql, expected_table, expected_key, expected_type_fragment) in cases {
            let result = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"));

            let stmt = result.expect("expected DdlStatement");
            match stmt {
                crate::services::ddl::DdlStatement::AlterTable {
                    table_name,
                    changes,
                } => {
                    assert_eq!(table_name, *expected_table, "table mismatch for `{sql}`");
                    assert_eq!(changes.len(), 1, "expected exactly one change");
                    match &changes[0] {
                        crate::services::ddl::AlterTableChange::PromotePropsKey {
                            key,
                            column_type,
                            comment,
                        } => {
                            assert_eq!(key, expected_key, "key mismatch for `{sql}`");
                            assert!(
                                format!("{:?}", column_type).contains(expected_type_fragment),
                                "type mismatch for `{sql}`: got {:?}",
                                column_type
                            );
                            assert!(comment.is_none());
                        }
                        other => panic!("expected PromotePropsKey, got {:?}", other),
                    }
                }
                other => panic!("expected AlterTable, got {:?}", other),
            }
        }
    }

    #[test]
    fn parse_ddl_set_table_option_props_auto_promotion() {
        let parser = SqlFrontendParser::new();

        let cases = [
            (
                "ALTER TABLE events SET (props_auto_promotion = 'enabled')",
                "events",
                "props_auto_promotion",
                "enabled",
            ),
            (
                "ALTER TABLE logs SET (props_auto_promotion = 'disabled');",
                "logs",
                "props_auto_promotion",
                "disabled",
            ),
        ];

        for (sql, expected_table, expected_key, expected_value) in cases {
            let stmt = parser
                .parse_ddl(sql)
                .unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"))
                .expect("expected DdlStatement");

            match stmt {
                crate::services::ddl::DdlStatement::AlterTable {
                    table_name,
                    changes,
                } => {
                    assert_eq!(table_name, expected_table, "table mismatch for `{sql}`");
                    assert_eq!(changes.len(), 1, "expected one change for `{sql}`");
                    match &changes[0] {
                        crate::services::ddl::AlterTableChange::SetTableOption { key, value } => {
                            assert_eq!(key, expected_key, "key mismatch for `{sql}`");
                            assert_eq!(value, expected_value, "value mismatch for `{sql}`");
                        }
                        other => panic!("expected SetTableOption, got {:?}", other),
                    }
                }
                other => panic!("expected AlterTable, got {:?}", other),
            }
        }
    }

    #[test]
    fn parse_ddl_promote_props_key_varchar() {
        let parser = SqlFrontendParser::new();
        let sql = "ALTER TABLE users PROMOTE PROPS KEY email TYPE VARCHAR(255);";
        let stmt = parser.parse_ddl(sql).unwrap().unwrap();
        match stmt {
            crate::services::ddl::DdlStatement::AlterTable { changes, .. } => match &changes[0] {
                crate::services::ddl::AlterTableChange::PromotePropsKey {
                    key,
                    column_type,
                    ..
                } => {
                    assert_eq!(key, "email");
                    assert!(
                        matches!(
                            column_type,
                            crate::services::ddl::SqlDataType::Varchar { .. }
                        ),
                        "expected Varchar, got {:?}",
                        column_type
                    );
                }
                other => panic!("expected PromotePropsKey, got {:?}", other),
            },
            other => panic!("expected AlterTable, got {:?}", other),
        }
    }
}

// ========================
// DDL Statement Parsing
// ========================

use crate::services::ddl::{
    AlterTableChange, ColumnDefinition, ColumnPosition, DdlStatement, IndexType, SqlDataType,
    TableConstraint,
};
use proximadb_data_model::ProximaType;

impl SqlFrontendParser {
    /// Parse SQL text and return a DDL statement if it's CREATE/ALTER/DROP
    pub fn parse_ddl(&self, sql: &str) -> Result<Option<DdlStatement>> {
        // Pre-parse: intercept ProximaDB-specific DDL that sqlparser does not understand.
        // Pattern: ALTER TABLE <name> PROMOTE PROPS KEY <key> TYPE <type>[;]
        if let Some(result) = self.try_parse_promote_props_key(sql)? {
            return Ok(Some(result));
        }
        // Pattern: CREATE RANK PROFILE [IF NOT EXISTS] <ident> AS '<toml>'
        if let Some(result) = try_parse_create_rank_profile(sql)? {
            return Ok(Some(result));
        }
        // Pattern: DROP RANK PROFILE [IF EXISTS] <ident>
        if let Some(result) = try_parse_drop_rank_profile(sql)? {
            return Ok(Some(result));
        }
        // Pattern: CREATE [OR REPLACE] FUNCTION <name>(params) RETURNS <ty> AS '<body>' (F5)
        if let Some(result) = try_parse_create_function(sql)? {
            return Ok(Some(result));
        }
        // Pattern: CREATE NAMESPACE [IF NOT EXISTS] <qualified_namespace>
        if let Some(result) = try_parse_create_namespace(sql)? {
            return Ok(Some(result));
        }
        // Pattern: DROP NAMESPACE [IF EXISTS] <qualified_namespace> [CASCADE|RESTRICT]
        if let Some(result) = try_parse_drop_namespace(sql)? {
            return Ok(Some(result));
        }
        // Pattern: ALTER TABLE <name> MATERIALIZE (warehouse publish → Parquet-backed)
        if let Some(result) = self.try_parse_materialize_table(sql)? {
            return Ok(Some(result));
        }

        let statements = Parser::parse_sql(&self.dialect, sql)
            .map_err(|e| anyhow!("SQL parsing failed: {}", e))?;

        if statements.is_empty() {
            return Err(anyhow!("No SQL statements found"));
        }

        if statements.len() > 1 {
            return Err(anyhow!(
                "Multiple statements not supported, found {}",
                statements.len()
            ));
        }

        let statement = &statements[0];
        self.try_convert_ddl(statement)
    }

    /// Intercept `ALTER TABLE <name> MATERIALIZE` before sqlparser (MATERIALIZE is a
    /// ProximaDB extension sqlparser does not understand). Publishes the table's
    /// current rows as a Parquet snapshot and marks it Parquet-backed for the OLAP
    /// route. Returns `Ok(Some(..))` on match, `Ok(None)` otherwise.
    fn try_parse_materialize_table(&self, sql: &str) -> Result<Option<DdlStatement>> {
        let normalised = sql.trim().trim_end_matches(';').trim();
        let upper = normalised.to_uppercase();

        // Fast path: skip anything that is not a MATERIALIZE statement.
        if !upper.contains("MATERIALIZE") {
            return Ok(None);
        }

        // Expected exactly: ALTER TABLE <name> MATERIALIZE
        let tokens = split_sql_words_respecting_quotes(normalised, "ALTER TABLE MATERIALIZE")?;
        if tokens.len() != 4 {
            return Ok(None);
        }
        let t = |i: usize| tokens.get(i).map(|s| s.to_uppercase());
        if t(0).as_deref() != Some("ALTER")
            || t(1).as_deref() != Some("TABLE")
            || t(3).as_deref() != Some("MATERIALIZE")
        {
            return Ok(None);
        }

        let name = unquote_object_name(&tokens[2]);
        if name.is_empty() {
            return Ok(None);
        }
        Ok(Some(DdlStatement::MaterializeTable { name }))
    }

    /// Intercept `ALTER TABLE <name> PROMOTE PROPS KEY <key> TYPE <type>` before sqlparser.
    /// Returns `Ok(Some(...))` when the pattern matches, `Ok(None)` when it does not.
    fn try_parse_promote_props_key(&self, sql: &str) -> Result<Option<DdlStatement>> {
        // Normalise: collapse whitespace, strip trailing semicolon.
        let normalised = sql.trim().trim_end_matches(';').trim();
        let upper = normalised.to_uppercase();

        // Fast path: skip anything that is not an ALTER TABLE … PROMOTE … statement.
        if !upper.contains("PROMOTE") {
            return Ok(None);
        }

        // Expected token sequence (case-insensitive):
        //   ALTER TABLE <name> PROMOTE PROPS KEY <key> TYPE <type…>
        let tokens =
            split_sql_words_respecting_quotes(normalised, "ALTER TABLE PROMOTE PROPS KEY")?;

        // Need at least 9 tokens: ALTER TABLE name PROMOTE PROPS KEY key TYPE type
        if tokens.len() < 9 {
            return Ok(None);
        }

        let t = |i: usize| tokens.get(i).map(|s| s.to_uppercase());

        if t(0).as_deref() != Some("ALTER")
            || t(1).as_deref() != Some("TABLE")
            || t(3).as_deref() != Some("PROMOTE")
            || t(4).as_deref() != Some("PROPS")
            || t(5).as_deref() != Some("KEY")
            || t(7).as_deref() != Some("TYPE")
        {
            return Ok(None);
        }

        let table_name = unquote_object_name(&tokens[2]);
        let key = unquote_identifier_text(&tokens[6]);
        // Remaining tokens after TYPE form the type string (e.g. "VARCHAR(255)").
        let type_str = tokens[8..].join(" ");

        // Parse the type via a dummy CREATE TABLE so we reuse convert_data_type.
        let dummy_sql = format!("CREATE TABLE _proxima_dummy (x {type_str});");
        let dummy_stmts = Parser::parse_sql(&self.dialect, &dummy_sql)
            .map_err(|e| anyhow!("Invalid type '{}' in PROMOTE PROPS KEY: {}", type_str, e))?;

        let column_type = match dummy_stmts.first() {
            Some(Statement::CreateTable(ct)) => {
                let col = ct
                    .columns
                    .first()
                    .ok_or_else(|| anyhow!("Internal: dummy table has no columns"))?;
                self.convert_data_type(&col.data_type)?
            }
            _ => return Err(anyhow!("Internal: dummy CREATE TABLE parse failed")),
        };

        Ok(Some(DdlStatement::AlterTable {
            table_name,
            changes: vec![AlterTableChange::PromotePropsKey {
                key,
                column_type,
                comment: None,
            }],
        }))
    }

    /// Try to convert a statement to DDL, returning None for non-DDL statements
    fn try_convert_ddl(&self, statement: &Statement) -> Result<Option<DdlStatement>> {
        use sqlparser::ast::{AlterTableOperation, ColumnOption, ObjectType as SqlObjectType};

        match statement {
            Statement::AlterTable {
                name,
                if_exists: _,
                only: _,
                operations,
                ..
            } => {
                let table_name = unquote_object_name(&name.to_string());
                let mut changes = Vec::new();

                for op in operations {
                    match op {
                        AlterTableOperation::AddColumn { column_def, .. } => {
                            let col = self.convert_column_def(column_def)?;
                            changes.push(AlterTableChange::AddColumn(col));
                        }
                        AlterTableOperation::DropColumn { column_names, .. } => {
                            // Take the first column name (most ALTER DROP COLUMN has single column)
                            if let Some(col_name) = column_names.first() {
                                changes.push(AlterTableChange::DropColumn(
                                    unquote_identifier_text(&col_name.to_string()),
                                ));
                            }
                        }
                        AlterTableOperation::RenameColumn {
                            old_column_name,
                            new_column_name,
                        } => {
                            changes.push(AlterTableChange::RenameColumn {
                                old_name: unquote_identifier_text(&old_column_name.to_string()),
                                new_name: unquote_identifier_text(&new_column_name.to_string()),
                            });
                        }
                        AlterTableOperation::AlterColumn { column_name, op } => {
                            use sqlparser::ast::AlterColumnOperation;
                            let column_name = unquote_identifier_text(&column_name.to_string());
                            match op {
                                AlterColumnOperation::SetNotNull => {
                                    changes.push(AlterTableChange::SetNullable {
                                        column_name: column_name.clone(),
                                        nullable: false,
                                    });
                                }
                                AlterColumnOperation::DropNotNull => {
                                    changes.push(AlterTableChange::SetNullable {
                                        column_name: column_name.clone(),
                                        nullable: true,
                                    });
                                }
                                AlterColumnOperation::SetDefault { value } => {
                                    changes.push(AlterTableChange::SetDefault {
                                        column_name: column_name.clone(),
                                        default_value: Some(format!("{}", value)),
                                    });
                                }
                                AlterColumnOperation::DropDefault => {
                                    changes.push(AlterTableChange::SetDefault {
                                        column_name: column_name.clone(),
                                        default_value: None,
                                    });
                                }
                                AlterColumnOperation::SetDataType { data_type, .. } => {
                                    let new_type = self.convert_data_type(data_type)?;
                                    changes.push(AlterTableChange::ChangeType {
                                        column_name: column_name.clone(),
                                        new_type,
                                    });
                                }
                                _ => {
                                    return Err(anyhow!(
                                        "Unsupported ALTER COLUMN operation: {:?}",
                                        op
                                    ));
                                }
                            }
                        }
                        AlterTableOperation::AddConstraint { constraint, .. } => {
                            use sqlparser::ast::TableConstraint as SqlConstraint;
                            match constraint {
                                SqlConstraint::Unique { name, columns, .. } => {
                                    let cols: Vec<String> = columns
                                        .iter()
                                        .map(|c| unquote_identifier_text(&c.to_string()))
                                        .collect();
                                    changes.push(AlterTableChange::AddConstraint {
                                        constraint_name: name
                                            .as_ref()
                                            .map(|n| unquote_identifier_text(&n.to_string())),
                                        constraint: TableConstraint::Unique { columns: cols },
                                    });
                                }
                                SqlConstraint::Check { name, expr, .. } => {
                                    changes.push(AlterTableChange::AddConstraint {
                                        constraint_name: name
                                            .as_ref()
                                            .map(|n| unquote_identifier_text(&n.to_string())),
                                        constraint: TableConstraint::Check {
                                            expression: format!("{}", expr),
                                        },
                                    });
                                }
                                SqlConstraint::ForeignKey {
                                    name,
                                    columns,
                                    foreign_table,
                                    referred_columns,
                                    on_delete,
                                    on_update,
                                    ..
                                } => {
                                    let cols: Vec<String> = columns
                                        .iter()
                                        .map(|c| unquote_identifier_text(&c.to_string()))
                                        .collect();
                                    let ref_cols: Vec<String> = referred_columns
                                        .iter()
                                        .map(|c| unquote_identifier_text(&c.to_string()))
                                        .collect();
                                    changes.push(AlterTableChange::AddConstraint {
                                        constraint_name: name
                                            .as_ref()
                                            .map(|n| unquote_identifier_text(&n.to_string())),
                                        constraint: TableConstraint::ForeignKey {
                                            columns: cols,
                                            references_table: unquote_object_name(
                                                &foreign_table.to_string(),
                                            ),
                                            references_columns: ref_cols,
                                            on_delete: on_delete.map(map_referential_action),
                                            on_update: on_update.map(map_referential_action),
                                        },
                                    });
                                }
                                _ => {
                                    return Err(anyhow!("Unsupported constraint type"));
                                }
                            }
                        }
                        AlterTableOperation::DropConstraint { name, .. } => {
                            changes.push(AlterTableChange::DropConstraint {
                                constraint_name: unquote_identifier_text(&name.to_string()),
                            });
                        }
                        AlterTableOperation::ChangeColumn {
                            old_name,
                            new_name,
                            data_type,
                            options,
                            ..
                        } => {
                            // MySQL-style CHANGE COLUMN (rename + type change)
                            let old_name = unquote_identifier_text(&old_name.to_string());
                            let new_name = unquote_identifier_text(&new_name.to_string());
                            if old_name != new_name {
                                changes.push(AlterTableChange::RenameColumn {
                                    old_name: old_name.clone(),
                                    new_name: new_name.clone(),
                                });
                            }
                            let new_type = self.convert_data_type(data_type)?;
                            changes.push(AlterTableChange::ChangeType {
                                column_name: new_name.clone(),
                                new_type,
                            });
                            // Handle FIRST/AFTER positioning
                            for opt in options {
                                if let ColumnOption::CharacterSet(cs) = opt {
                                    // Check for FIRST/AFTER in the raw string
                                    let cs_str = cs.to_string().to_uppercase();
                                    if cs_str == "FIRST" {
                                        changes.push(AlterTableChange::MoveColumn {
                                            column_name: new_name.clone(),
                                            position: ColumnPosition::First,
                                        });
                                    }
                                }
                            }
                        }
                        AlterTableOperation::SetOptionsParens { options } => {
                            // ALTER TABLE <name> SET (key = 'value', ...)
                            use sqlparser::ast::SqlOption as SqlOpt;
                            for opt in options {
                                if let SqlOpt::KeyValue { key, value } = opt {
                                    let value_str = match value {
                                        sqlparser::ast::Expr::Value(v) => match &v.value {
                                            sqlparser::ast::Value::SingleQuotedString(s)
                                            | sqlparser::ast::Value::DoubleQuotedString(s) => {
                                                s.clone()
                                            }
                                            other => format!("{other}"),
                                        },
                                        other => format!("{other}"),
                                    };
                                    changes.push(AlterTableChange::SetTableOption {
                                        key: key.to_string(),
                                        value: value_str,
                                    });
                                }
                            }
                        }
                        _ => {
                            return Err(anyhow!("Unsupported ALTER TABLE operation: {:?}", op));
                        }
                    }
                }

                Ok(Some(DdlStatement::AlterTable {
                    table_name,
                    changes,
                }))
            }
            Statement::CreateTable(create_table) => {
                if create_table.query.is_some()
                    || create_table.like.is_some()
                    || create_table.clone.is_some()
                {
                    return Err(anyhow!(
                        "CREATE TABLE with query/LIKE/CLONE clauses is not supported"
                    ));
                }

                let table_name = unquote_object_name(&create_table.name.to_string());
                let if_not_exists = create_table.if_not_exists;
                let mut columns = create_table
                    .columns
                    .iter()
                    .map(|col| self.convert_column_def(col))
                    .collect::<Result<Vec<_>>>()?;
                let constraints = apply_table_constraints(&mut columns, &create_table.constraints)?;
                let properties = table_options_to_properties(&create_table.table_options);

                Ok(Some(DdlStatement::CreateTable {
                    table_name,
                    columns,
                    constraints,
                    if_not_exists,
                    properties,
                }))
            }
            Statement::CreateIndex(create_index) => {
                let index_name = create_index
                    .name
                    .as_ref()
                    .map(|name| name.to_string())
                    .ok_or_else(|| anyhow!("CREATE INDEX requires an index name"))?;
                let index_name = unquote_identifier_text(&index_name);
                let table_name = unquote_object_name(&create_index.table_name.to_string());
                let columns = create_index
                    .columns
                    .iter()
                    .map(|col| unquote_identifier_text(&col.column.to_string()))
                    .collect::<Vec<_>>();
                let index_type = self.parse_index_type(create_index)?;
                let if_not_exists = create_index.if_not_exists;

                Ok(Some(DdlStatement::CreateIndex {
                    index_name,
                    table_name,
                    columns,
                    index_type,
                    if_not_exists,
                }))
            }
            Statement::Drop {
                object_type,
                if_exists,
                names,
                purge,
                table,
                ..
            } => match object_type {
                SqlObjectType::Table => {
                    let table_name = names
                        .first()
                        .ok_or_else(|| anyhow!("DROP TABLE requires a table name"))?
                        .to_string();
                    let table_name = unquote_object_name(&table_name);
                    Ok(Some(DdlStatement::DropTable {
                        table_name,
                        if_exists: *if_exists,
                        purge: *purge,
                    }))
                }
                SqlObjectType::Index => {
                    let index_name = names
                        .first()
                        .ok_or_else(|| anyhow!("DROP INDEX requires an index name"))?
                        .to_string();
                    let index_name = unquote_identifier_text(&index_name);
                    let table_name = table
                        .as_ref()
                        .map(|table_name| table_name.to_string())
                        .ok_or_else(|| anyhow!("DROP INDEX requires a table name"))?;
                    let table_name = unquote_object_name(&table_name);

                    Ok(Some(DdlStatement::DropIndex {
                        index_name,
                        table_name,
                        if_exists: *if_exists,
                    }))
                }
                _ => Err(anyhow!("Unsupported DROP object type: {:?}", object_type)),
            },
            Statement::Query(_) => Ok(None), // SELECT query, not DDL
            Statement::Insert(_) | Statement::Update { .. } | Statement::Delete(_) => Ok(None), // DML
            _ => Ok(None),
        }
    }

    fn parse_index_type(&self, create_index: &CreateIndex) -> Result<IndexType> {
        let explicit_using = create_index.using.as_ref().cloned().or_else(|| {
            create_index.index_options.iter().find_map(|opt| {
                if let IndexOption::Using(index_type) = opt {
                    Some(index_type.clone())
                } else {
                    None
                }
            })
        });

        let index_type = explicit_using.unwrap_or(sqlparser::ast::IndexType::BTree);
        let index_type = match index_type {
            sqlparser::ast::IndexType::BTree => IndexType::BTree,
            sqlparser::ast::IndexType::Hash => IndexType::Hash,
            sqlparser::ast::IndexType::GIN => IndexType::Gin,
            sqlparser::ast::IndexType::Custom(name)
                if name.value.eq_ignore_ascii_case("fulltext") =>
            {
                IndexType::FullText
            }
            sqlparser::ast::IndexType::Custom(name) if name.value.eq_ignore_ascii_case("gin") => {
                IndexType::Gin
            }
            sqlparser::ast::IndexType::Custom(name) if name.value.eq_ignore_ascii_case("hnsw") => {
                IndexType::Hnsw {
                    m: None,
                    ef_construction: None,
                }
            }
            sqlparser::ast::IndexType::Custom(name) if name.value.eq_ignore_ascii_case("ivf") => {
                IndexType::Ivf { nlist: None }
            }
            sqlparser::ast::IndexType::Custom(name) => {
                return Err(anyhow!("Unsupported CREATE INDEX USING {}", name.value));
            }
            other => return Err(anyhow!("Unsupported CREATE INDEX USING {}", other)),
        };

        Ok(index_type)
    }

    /// Convert sqlparser column definition to DDL ColumnDefinition
    fn convert_column_def(&self, col_def: &sqlparser::ast::ColumnDef) -> Result<ColumnDefinition> {
        use sqlparser::ast::ColumnOption;

        let name = col_def.name.value.clone();
        let data_type = self.convert_data_type(&col_def.data_type)?;
        let mut nullable = true;
        let mut default_value = None;
        let mut comment = None;
        let mut primary_key = false;

        for option in &col_def.options {
            match &option.option {
                ColumnOption::NotNull => {
                    nullable = false;
                }
                ColumnOption::Null => {
                    nullable = true;
                }
                ColumnOption::Default(expr) => {
                    default_value = Some(format!("{}", expr));
                }
                ColumnOption::Comment(c) => {
                    comment = Some(c.clone());
                }
                ColumnOption::Unique {
                    is_primary: true, ..
                } => {
                    primary_key = true;
                    nullable = false;
                }
                _ => {}
            }
        }

        Ok(ColumnDefinition {
            name,
            data_type,
            nullable,
            default_value,
            comment,
            primary_key,
        })
    }

    /// Convert sqlparser DataType to DDL SqlDataType
    fn convert_data_type(&self, dt: &sqlparser::ast::DataType) -> Result<SqlDataType> {
        use sqlparser::ast::DataType as SqlDt;

        match dt {
            SqlDt::Boolean | SqlDt::Bool => Ok(SqlDataType::Boolean),
            SqlDt::TinyInt(_) => Ok(SqlDataType::TinyInt),
            SqlDt::SmallInt(_) => Ok(SqlDataType::SmallInt),
            SqlDt::Int(_) | SqlDt::Integer(_) => Ok(SqlDataType::Int),
            SqlDt::BigInt(_) => Ok(SqlDataType::BigInt),
            SqlDt::Real | SqlDt::Float(_) => Ok(SqlDataType::Float),
            SqlDt::Double(_) | SqlDt::DoublePrecision => Ok(SqlDataType::Double),
            SqlDt::Decimal(info) | SqlDt::Numeric(info) => {
                use sqlparser::ast::ExactNumberInfo;
                match info {
                    ExactNumberInfo::PrecisionAndScale(p, s) => Ok(SqlDataType::Decimal {
                        precision: *p as u32,
                        scale: *s as u32,
                    }),
                    ExactNumberInfo::Precision(p) => Ok(SqlDataType::Decimal {
                        precision: *p as u32,
                        scale: 0,
                    }),
                    ExactNumberInfo::None => Ok(SqlDataType::Decimal {
                        precision: 38,
                        scale: 9,
                    }),
                }
            }
            SqlDt::Varchar(info)
            | SqlDt::CharVarying(info)
            | SqlDt::Char(info)
            | SqlDt::Character(info) => {
                use sqlparser::ast::CharacterLength;
                let max_length = match info {
                    Some(CharacterLength::IntegerLength { length, .. }) => Some(*length as u32),
                    _ => None,
                };
                Ok(SqlDataType::Varchar { max_length })
            }
            SqlDt::Text => Ok(SqlDataType::Text),
            SqlDt::Binary(_) | SqlDt::Varbinary(_) => Ok(SqlDataType::Binary),
            SqlDt::Blob(_) => Ok(SqlDataType::Blob),
            SqlDt::Date => Ok(SqlDataType::Date),
            SqlDt::Time(_, _) => Ok(SqlDataType::Time),
            SqlDt::Timestamp(_, tz_info) => {
                use sqlparser::ast::TimezoneInfo;
                match tz_info {
                    TimezoneInfo::WithTimeZone | TimezoneInfo::Tz => Ok(SqlDataType::TimestampTz),
                    _ => Ok(SqlDataType::Timestamp),
                }
            }
            SqlDt::Uuid => Ok(SqlDataType::Uuid),
            SqlDt::JSON => Ok(SqlDataType::Json),
            SqlDt::JSONB => Ok(SqlDataType::Jsonb),
            SqlDt::Custom(name, modifiers) => {
                let type_name = name.to_string().to_uppercase();
                match type_name.as_str() {
                    "VECTOR" => {
                        // Parse dimension from modifiers: VECTOR(768)
                        let dimension = modifiers
                            .first()
                            .and_then(|m| m.to_string().parse::<u32>().ok())
                            .unwrap_or(0);
                        Ok(SqlDataType::Vector { dimension })
                    }
                    "SPARSE_VECTOR" => {
                        let dimension = modifiers
                            .first()
                            .and_then(|m| m.to_string().parse::<u32>().ok())
                            .unwrap_or(0);
                        Ok(SqlDataType::SparseVector { dimension })
                    }
                    "BINARY_VECTOR" => {
                        let dimension = modifiers
                            .first()
                            .and_then(|m| m.to_string().parse::<u32>().ok())
                            .unwrap_or(0);
                        Ok(SqlDataType::BinaryVector { dimension })
                    }
                    _ => Err(anyhow!("Unsupported custom type: {}", type_name)),
                }
            }
            _ => Err(anyhow!("Unsupported data type: {:?}", dt)),
        }
    }
}

/// Map sqlparser's referential action onto the catalog's, so parsed
/// `ON DELETE`/`ON UPDATE` actions survive into the table constraint (TD-110).
fn map_referential_action(
    action: sqlparser::ast::ReferentialAction,
) -> proximadb_catalog::ReferentialAction {
    use proximadb_catalog::ReferentialAction as Cat;
    use sqlparser::ast::ReferentialAction as Sql;
    match action {
        Sql::Restrict => Cat::Restrict,
        Sql::Cascade => Cat::Cascade,
        Sql::SetNull => Cat::SetNull,
        Sql::NoAction => Cat::NoAction,
        Sql::SetDefault => Cat::SetDefault,
    }
}

fn apply_table_constraints(
    columns: &mut [ColumnDefinition],
    constraints: &[sqlparser::ast::TableConstraint],
) -> Result<Vec<TableConstraint>> {
    use sqlparser::ast::TableConstraint as SqlConstraint;

    let mut table_constraints = Vec::new();

    for constraint in constraints {
        match constraint {
            SqlConstraint::PrimaryKey { columns: pk, .. } => {
                for ident in pk {
                    let name = unquote_identifier_text(&ident.to_string());
                    if let Some(column) = columns.iter_mut().find(|column| column.name == name) {
                        column.primary_key = true;
                        column.nullable = false;
                    } else {
                        return Err(anyhow!("PRIMARY KEY references unknown column {}", name));
                    }
                }
            }
            SqlConstraint::Unique {
                columns: unique, ..
            } => {
                let unique_columns = unique
                    .iter()
                    .map(|ident| {
                        let name = unquote_identifier_text(&ident.to_string());
                        if columns.iter().any(|column| column.name == name) {
                            Ok(name)
                        } else {
                            Err(anyhow!("UNIQUE references unknown column {}", name))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                table_constraints.push(TableConstraint::Unique {
                    columns: unique_columns,
                });
            }
            SqlConstraint::Check { expr, .. } => {
                table_constraints.push(TableConstraint::Check {
                    expression: format!("{}", expr),
                });
            }
            SqlConstraint::ForeignKey {
                columns: fk,
                foreign_table,
                referred_columns,
                on_delete,
                on_update,
                ..
            } => {
                let fk_columns = fk
                    .iter()
                    .map(|ident| {
                        let name = unquote_identifier_text(&ident.to_string());
                        if columns.iter().any(|column| column.name == name) {
                            Ok(name)
                        } else {
                            Err(anyhow!("FOREIGN KEY references unknown column {}", name))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let references_columns = referred_columns
                    .iter()
                    .map(|ident| unquote_identifier_text(&ident.to_string()))
                    .collect::<Vec<_>>();

                table_constraints.push(TableConstraint::ForeignKey {
                    columns: fk_columns,
                    references_table: unquote_object_name(&foreign_table.to_string()),
                    references_columns,
                    on_delete: on_delete.map(map_referential_action),
                    on_update: on_update.map(map_referential_action),
                });
            }
            _ => {}
        }
    }

    Ok(table_constraints)
}

fn table_options_to_properties(options: &CreateTableOptions) -> HashMap<String, String> {
    let mut properties = HashMap::new();
    let options = match options {
        CreateTableOptions::With(options)
        | CreateTableOptions::Options(options)
        | CreateTableOptions::Plain(options)
        | CreateTableOptions::TableProperties(options) => options,
        CreateTableOptions::None => return properties,
    };

    for option in options {
        match option {
            SqlOption::KeyValue { key, value } => {
                properties.insert(
                    key.value.to_ascii_lowercase(),
                    sql_option_value_to_string(value),
                );
            }
            SqlOption::Ident(ident) => {
                properties.insert(ident.value.to_ascii_lowercase(), "true".to_string());
            }
            SqlOption::Comment(comment) => {
                properties.insert("comment".to_string(), comment.to_string());
            }
            _ => {}
        }
    }

    properties
}

fn sql_option_value_to_string(value: &SqlExpr) -> String {
    match value {
        SqlExpr::Value(value) => match &value.value {
            Value::SingleQuotedString(value) | Value::DoubleQuotedString(value) => value.clone(),
            Value::Boolean(value) => value.to_string(),
            Value::Number(value, _) => value.clone(),
            _ => value.value.to_string(),
        },
        SqlExpr::Identifier(ident) => ident.value.clone(),
        _ => value.to_string().trim_matches('\'').to_string(),
    }
}

pub fn unquote_object_name(value: &str) -> String {
    parse_qualified_identifier_parts(value, "qualified identifier")
        .map(|parts| parts.join("."))
        .unwrap_or_else(|_| {
            value
                .split('.')
                .map(unquote_identifier_text)
                .collect::<Vec<_>>()
                .join(".")
        })
}

pub fn unquote_identifier_text(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1].replace("\"\"", "\"")
    } else {
        trimmed.to_string()
    }
}

fn consume_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let trimmed = input.trim_start();
    if trimmed.len() < keyword.len() {
        return None;
    }

    let (candidate, rest) = trimmed.split_at(keyword.len());
    if !candidate.eq_ignore_ascii_case(keyword) {
        return None;
    }
    if rest.chars().next().is_some_and(|ch| !ch.is_whitespace()) {
        return None;
    }
    Some(rest)
}

fn parse_qualified_identifier_parts(input: &str, statement_name: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = input.trim().chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            '.' if !in_quotes => {
                let part = current.trim();
                if part.is_empty() {
                    return Err(anyhow!(
                        "{statement_name} contains an empty identifier part"
                    ));
                }
                parts.push(part.to_string());
                current.clear();
            }
            ch if ch.is_whitespace() && !in_quotes => {
                return Err(anyhow!(
                    "{statement_name} expects a single qualified namespace identifier"
                ));
            }
            _ => current.push(ch),
        }
    }

    if in_quotes {
        return Err(anyhow!(
            "{statement_name} has an unterminated quoted identifier"
        ));
    }

    let part = current.trim();
    if part.is_empty() {
        return Err(anyhow!("{statement_name} requires a namespace name"));
    }
    parts.push(part.to_string());
    Ok(parts)
}

fn try_parse_create_namespace(sql: &str) -> Result<Option<DdlStatement>> {
    let normalised = sql.trim().trim_end_matches(';').trim();
    let Some(rest) = consume_keyword(normalised, "CREATE") else {
        return Ok(None);
    };
    let Some(mut rest) = consume_keyword(rest, "NAMESPACE") else {
        return Ok(None);
    };

    let mut if_not_exists = false;
    if let Some(after_if) = consume_keyword(rest, "IF")
        && let Some(after_not) = consume_keyword(after_if, "NOT")
        && let Some(after_exists) = consume_keyword(after_not, "EXISTS")
    {
        if_not_exists = true;
        rest = after_exists;
    }

    let namespace = parse_qualified_identifier_parts(rest, "CREATE NAMESPACE")?;
    Ok(Some(DdlStatement::CreateNamespace {
        namespace,
        if_not_exists,
        properties: HashMap::new(),
    }))
}

fn split_sql_words_respecting_quotes(input: &str, statement_name: &str) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = input.trim().chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                current.push(ch);
                if in_quotes && chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ch if ch.is_whitespace() && !in_quotes => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }

    if in_quotes {
        return Err(anyhow!(
            "{statement_name} has an unterminated quoted identifier"
        ));
    }
    if !current.is_empty() {
        words.push(current);
    }
    Ok(words)
}

fn try_parse_drop_namespace(sql: &str) -> Result<Option<DdlStatement>> {
    let normalised = sql.trim().trim_end_matches(';').trim();
    let Some(rest) = consume_keyword(normalised, "DROP") else {
        return Ok(None);
    };
    let Some(mut rest) = consume_keyword(rest, "NAMESPACE") else {
        return Ok(None);
    };

    let mut if_exists = false;
    if let Some(after_if) = consume_keyword(rest, "IF")
        && let Some(after_exists) = consume_keyword(after_if, "EXISTS")
    {
        if_exists = true;
        rest = after_exists;
    }

    let mut words = split_sql_words_respecting_quotes(rest, "DROP NAMESPACE")?;
    if words.is_empty() {
        return Err(anyhow!("DROP NAMESPACE requires a namespace name"));
    }

    let mut cascade = false;
    if let Some(last) = words.last() {
        if last.eq_ignore_ascii_case("CASCADE") {
            cascade = true;
            words.pop();
        } else if last.eq_ignore_ascii_case("RESTRICT") {
            words.pop();
        }
    }

    if words.len() != 1 {
        return Err(anyhow!(
            "DROP NAMESPACE expects a single qualified namespace identifier"
        ));
    }

    let namespace = parse_qualified_identifier_parts(&words[0], "DROP NAMESPACE")?;
    Ok(Some(DdlStatement::DropNamespace {
        namespace,
        if_exists,
        cascade,
    }))
}

// =========================================================================
// CREATE / DROP RANK PROFILE (commit 4/5 of R-7c production wiring)
// =========================================================================

/// Match `CREATE RANK PROFILE [IF NOT EXISTS] <ident> AS '<toml-body>'`. The
/// body is a SQL string literal (single-quoted, with `''` as an escaped single
/// quote). Returns `Ok(None)` when the prefix doesn't match so the caller can
/// fall through to sqlparser-rs.
pub(crate) fn try_parse_create_rank_profile(sql: &str) -> Result<Option<DdlStatement>> {
    let normalised = sql.trim().trim_end_matches(';').trim();
    let upper = normalised.to_ascii_uppercase();
    if !upper.starts_with("CREATE RANK PROFILE") {
        return Ok(None);
    }
    let after_prefix = normalised["CREATE RANK PROFILE".len()..].trim_start();

    let (if_not_exists, after_optional) = strip_if_not_exists(after_prefix);
    let (name, after_name) = extract_identifier(after_optional)?;

    let after_name = after_name.trim_start();
    let after_as = match after_name
        .get(..2)
        .map(|s| s.eq_ignore_ascii_case("AS"))
        .unwrap_or(false)
        && after_name
            .chars()
            .nth(2)
            .map(|c| c.is_whitespace())
            .unwrap_or(false)
    {
        true => after_name[2..].trim_start(),
        false => {
            return Err(anyhow!(
                "CREATE RANK PROFILE '{}': expected 'AS' before TOML body",
                name
            ));
        }
    };

    let spec_toml = extract_single_quoted_literal(after_as).map_err(|e| {
        anyhow!(
            "CREATE RANK PROFILE '{}': expected single-quoted TOML body after AS: {}",
            name,
            e
        )
    })?;

    Ok(Some(DdlStatement::CreateRankProfile {
        name,
        spec_toml,
        if_not_exists,
    }))
}

/// Match `DROP RANK PROFILE [IF EXISTS] <ident>`. Returns `Ok(None)` when the
/// prefix doesn't match so the caller can fall through to sqlparser-rs.
pub(crate) fn try_parse_drop_rank_profile(sql: &str) -> Result<Option<DdlStatement>> {
    let normalised = sql.trim().trim_end_matches(';').trim();
    let upper = normalised.to_ascii_uppercase();
    if !upper.starts_with("DROP RANK PROFILE") {
        return Ok(None);
    }
    let after_prefix = normalised["DROP RANK PROFILE".len()..].trim_start();
    let (if_exists, after_optional) = strip_if_exists(after_prefix);
    let (name, rest) = extract_identifier(after_optional)?;

    if !rest.trim().is_empty() {
        return Err(anyhow!(
            "DROP RANK PROFILE '{}': unexpected trailing text: '{}'",
            name,
            rest.trim()
        ));
    }
    Ok(Some(DdlStatement::DropRankProfile { name, if_exists }))
}

/// Map a SQL type name (as written in a `CREATE FUNCTION` signature) to a [`ProximaType`].
/// Covers the common scalar types; `DOUBLE PRECISION` and other multi-word names arrive joined.
fn parse_function_param_type(s: &str) -> Result<ProximaType> {
    Ok(match s.trim().to_ascii_uppercase().as_str() {
        "BOOLEAN" | "BOOL" => ProximaType::Boolean,
        "TINYINT" | "INT1" => ProximaType::Int8,
        "SMALLINT" | "INT2" => ProximaType::Int16,
        "INT" | "INTEGER" | "INT4" => ProximaType::Int32,
        "BIGINT" | "INT8" => ProximaType::Int64,
        "REAL" | "FLOAT4" => ProximaType::Float32,
        "DOUBLE" | "DOUBLE PRECISION" | "FLOAT" | "FLOAT8" => ProximaType::Float64,
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => ProximaType::String,
        "JSON" | "JSONB" => ProximaType::Json,
        other => return Err(anyhow!("unsupported function type '{}'", other)),
    })
}

/// Match `CREATE [OR REPLACE] FUNCTION <name>(p1 t1, p2 t2, ...) RETURNS <ty> AS '<expr>'` (F5).
/// The body is a SQL string literal (a scalar expression over the parameters). Returns `Ok(None)`
/// when the prefix doesn't match so the caller falls through to sqlparser. Parameterized types
/// (e.g. `VARCHAR(255)`) are not supported in this first cut.
pub(crate) fn try_parse_create_function(sql: &str) -> Result<Option<DdlStatement>> {
    let normalised = sql.trim().trim_end_matches(';').trim();
    let upper = normalised.to_ascii_uppercase();
    let (or_replace, after_create) = if upper.starts_with("CREATE OR REPLACE FUNCTION") {
        (true, &normalised["CREATE OR REPLACE FUNCTION".len()..])
    } else if upper.starts_with("CREATE FUNCTION") {
        (false, &normalised["CREATE FUNCTION".len()..])
    } else {
        return Ok(None);
    };

    let after = after_create.trim_start();

    // Parameter list: name ( p1 t1, p2 t2, ... ) — simple types only (no inner parens). The
    // name is the text up to '(' (which may abut the name with no space).
    let open = after
        .find('(')
        .ok_or_else(|| anyhow!("CREATE FUNCTION: expected '(' before the parameter list"))?;
    let name = after[..open].trim().to_string();
    if name.is_empty() {
        return Err(anyhow!("CREATE FUNCTION: missing function name"));
    }
    let close_rel = after[open..]
        .find(')')
        .ok_or_else(|| anyhow!("CREATE FUNCTION '{}': expected ')' after parameters", name))?;
    let close = open + close_rel;
    let mut params: Vec<(String, ProximaType)> = Vec::new();
    for part in after[open + 1..close].split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut it = part.splitn(2, char::is_whitespace);
        let pname = it.next().unwrap_or("").trim().to_string();
        let ptype = it.next().ok_or_else(|| {
            anyhow!(
                "CREATE FUNCTION '{}': parameter '{}' is missing a type",
                name,
                pname
            )
        })?;
        params.push((pname, parse_function_param_type(ptype)?));
    }

    // RETURNS <ty> AS '<body>'
    let rest = after[close + 1..].trim_start();
    if !rest.to_ascii_uppercase().starts_with("RETURNS") {
        return Err(anyhow!(
            "CREATE FUNCTION '{}': expected RETURNS <type>",
            name
        ));
    }
    let after_returns = rest["RETURNS".len()..].trim_start();
    let as_pos = after_returns
        .to_ascii_uppercase()
        .find(" AS ")
        .ok_or_else(|| anyhow!("CREATE FUNCTION '{}': expected AS '<body>'", name))?;
    let return_ty = parse_function_param_type(after_returns[..as_pos].trim())?;
    let body =
        extract_single_quoted_literal(after_returns[as_pos + 4..].trim_start()).map_err(|e| {
            anyhow!(
                "CREATE FUNCTION '{}': expected single-quoted body after AS: {}",
                name,
                e
            )
        })?;

    Ok(Some(DdlStatement::CreateFunction {
        name,
        params,
        return_ty,
        body,
        or_replace,
    }))
}

pub(crate) fn strip_if_not_exists(input: &str) -> (bool, &str) {
    let upper = input.to_ascii_uppercase();
    if upper.starts_with("IF NOT EXISTS")
        && input
            .chars()
            .nth("IF NOT EXISTS".len())
            .map(|c| c.is_whitespace())
            .unwrap_or(false)
    {
        (true, input["IF NOT EXISTS".len()..].trim_start())
    } else {
        (false, input)
    }
}

fn strip_if_exists(input: &str) -> (bool, &str) {
    let upper = input.to_ascii_uppercase();
    if upper.starts_with("IF EXISTS")
        && input
            .chars()
            .nth("IF EXISTS".len())
            .map(|c| c.is_whitespace())
            .unwrap_or(false)
    {
        (true, input["IF EXISTS".len()..].trim_start())
    } else {
        (false, input)
    }
}

/// Extract a SQL identifier (`name` or `"name"`) from the head of `input` and
/// return it alongside the remainder. The unquoted identifier is returned with
/// `"` escapes unescaped (`""` → `"`).
fn extract_identifier(input: &str) -> Result<(String, &str)> {
    let input = input.trim_start();
    if input.is_empty() {
        return Err(anyhow!("expected identifier"));
    }
    if let Some(rest) = input.strip_prefix('"') {
        let end = rest
            .find('"')
            .ok_or_else(|| anyhow!("unterminated quoted identifier"))?;
        let name = rest[..end].replace("\"\"", "\"");
        Ok((name, &rest[end + 1..]))
    } else {
        let end = input
            .find(|c: char| c.is_whitespace())
            .unwrap_or(input.len());
        let name = input[..end].to_string();
        if name.is_empty() {
            return Err(anyhow!("expected identifier"));
        }
        Ok((name, &input[end..]))
    }
}

/// Extract a single-quoted SQL string literal from the head of `input`.
/// `''` inside the literal escapes a single quote.
fn extract_single_quoted_literal(input: &str) -> Result<String> {
    let input = input.trim_start();
    if !input.starts_with('\'') {
        return Err(anyhow!("missing opening quote"));
    }
    let bytes = input.as_bytes();
    let mut i = 1; // skip opening quote
    let mut out = String::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\'' {
            // Check for escaped single quote
            if i + 1 < bytes.len() && bytes[i + 1] as char == '\'' {
                out.push('\'');
                i += 2;
                continue;
            }
            // End of literal — anything after the closing quote (besides
            // optional whitespace + trailing semicolon) is an error.
            let trailing = input[i + 1..].trim();
            if !trailing.is_empty() {
                return Err(anyhow!(
                    "unexpected text after string literal: '{}'",
                    trailing
                ));
            }
            return Ok(out);
        }
        out.push(c);
        i += 1;
    }
    Err(anyhow!("unterminated string literal"))
}

#[cfg(test)]
mod rank_profile_ddl_tests {
    use super::*;

    fn assert_create_profile(sql: &str, expected_name: &str, expected_toml: &str, ine: bool) {
        let stmt = try_parse_create_rank_profile(sql).unwrap().unwrap();
        match stmt {
            DdlStatement::CreateRankProfile {
                name,
                spec_toml,
                if_not_exists,
            } => {
                assert_eq!(name, expected_name);
                assert_eq!(spec_toml, expected_toml);
                assert_eq!(if_not_exists, ine);
            }
            other => panic!("expected CreateRankProfile, got {:?}", other),
        }
    }

    fn assert_drop_profile(sql: &str, expected_name: &str, ie: bool) {
        let stmt = try_parse_drop_rank_profile(sql).unwrap().unwrap();
        match stmt {
            DdlStatement::DropRankProfile { name, if_exists } => {
                assert_eq!(name, expected_name);
                assert_eq!(if_exists, ie);
            }
            other => panic!("expected DropRankProfile, got {:?}", other),
        }
    }

    #[test]
    fn create_rank_profile_minimal_form() {
        assert_create_profile(
            "CREATE RANK PROFILE basic AS '[first_phase]\nexpression = \"1.0\"'",
            "basic",
            "[first_phase]\nexpression = \"1.0\"",
            false,
        );
    }

    #[test]
    fn create_rank_profile_with_if_not_exists() {
        assert_create_profile(
            "CREATE RANK PROFILE IF NOT EXISTS basic AS '[first_phase]\nexpression = \"1.0\"'",
            "basic",
            "[first_phase]\nexpression = \"1.0\"",
            true,
        );
    }

    #[test]
    fn create_rank_profile_handles_trailing_semicolon() {
        assert_create_profile(
            "CREATE RANK PROFILE basic AS '[first_phase]\nexpression = \"1.0\"';",
            "basic",
            "[first_phase]\nexpression = \"1.0\"",
            false,
        );
    }

    #[test]
    fn create_rank_profile_escapes_single_quotes_in_body() {
        let sql = "CREATE RANK PROFILE x AS 'don''t'";
        let stmt = try_parse_create_rank_profile(sql).unwrap().unwrap();
        match stmt {
            DdlStatement::CreateRankProfile { spec_toml, .. } => {
                assert_eq!(spec_toml, "don't");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn create_rank_profile_accepts_quoted_identifier() {
        assert_create_profile(
            "CREATE RANK PROFILE \"My Profile\" AS 'body'",
            "My Profile",
            "body",
            false,
        );
    }

    #[test]
    fn create_rank_profile_returns_none_for_unrelated_sql() {
        assert!(
            try_parse_create_rank_profile("SELECT * FROM docs")
                .unwrap()
                .is_none()
        );
        assert!(
            try_parse_create_rank_profile("CREATE TABLE t (id INT)")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn create_rank_profile_rejects_missing_as() {
        let err = try_parse_create_rank_profile("CREATE RANK PROFILE basic 'body'").unwrap_err();
        assert!(err.to_string().contains("expected 'AS'"));
    }

    #[test]
    fn create_rank_profile_rejects_unterminated_string() {
        let err = try_parse_create_rank_profile("CREATE RANK PROFILE basic AS 'no closing quote")
            .unwrap_err();
        assert!(err.to_string().contains("unterminated"));
    }

    #[test]
    fn drop_rank_profile_minimal_form() {
        assert_drop_profile("DROP RANK PROFILE basic", "basic", false);
    }

    #[test]
    fn drop_rank_profile_with_if_exists() {
        assert_drop_profile("DROP RANK PROFILE IF EXISTS basic", "basic", true);
    }

    #[test]
    fn drop_rank_profile_handles_trailing_semicolon() {
        assert_drop_profile("DROP RANK PROFILE basic;", "basic", false);
    }

    #[test]
    fn drop_rank_profile_rejects_trailing_garbage() {
        let err = try_parse_drop_rank_profile("DROP RANK PROFILE basic EXTRA").unwrap_err();
        assert!(err.to_string().contains("unexpected trailing text"));
    }

    #[test]
    fn drop_rank_profile_returns_none_for_unrelated_sql() {
        assert!(
            try_parse_drop_rank_profile("DROP TABLE t")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parse_ddl_routes_create_rank_profile_through_pre_parser() {
        let parser = SqlFrontendParser::new();
        let stmt = parser
            .parse_ddl("CREATE RANK PROFILE basic AS '[first_phase]\nexpression = \"1.0\"'")
            .unwrap()
            .expect("parse should succeed");
        assert!(matches!(stmt, DdlStatement::CreateRankProfile { .. }));
    }

    #[test]
    fn parse_ddl_routes_drop_rank_profile_through_pre_parser() {
        let parser = SqlFrontendParser::new();
        let stmt = parser
            .parse_ddl("DROP RANK PROFILE basic")
            .unwrap()
            .expect("parse should succeed");
        assert!(matches!(stmt, DdlStatement::DropRankProfile { .. }));
    }

    #[test]
    fn parse_create_function_lowers_signature_and_body() {
        // F5 slice 4: CREATE FUNCTION text → DdlStatement::CreateFunction.
        let stmt =
            try_parse_create_function("CREATE FUNCTION double(x BIGINT) RETURNS BIGINT AS 'x * 2'")
                .unwrap()
                .unwrap();
        match stmt {
            DdlStatement::CreateFunction {
                name,
                params,
                return_ty,
                body,
                or_replace,
            } => {
                assert_eq!(name, "double");
                assert_eq!(params, vec![("x".to_string(), ProximaType::Int64)]);
                assert_eq!(return_ty, ProximaType::Int64);
                assert_eq!(body, "x * 2");
                assert!(!or_replace);
            }
            other => panic!("expected CreateFunction, got {:?}", other),
        }
    }

    #[test]
    fn parse_create_or_replace_function_multi_param() {
        let stmt = try_parse_create_function(
            "CREATE OR REPLACE FUNCTION wsum(a DOUBLE, b DOUBLE) RETURNS DOUBLE AS 'a * 2 + b'",
        )
        .unwrap()
        .unwrap();
        let DdlStatement::CreateFunction {
            params,
            or_replace,
            body,
            ..
        } = stmt
        else {
            panic!("expected CreateFunction");
        };
        assert!(or_replace);
        assert_eq!(params.len(), 2);
        assert_eq!(params[1], ("b".to_string(), ProximaType::Float64));
        assert_eq!(body, "a * 2 + b");
    }

    #[test]
    fn parse_create_function_falls_through_for_non_function() {
        assert!(try_parse_create_function("SELECT 1").unwrap().is_none());
        assert!(
            try_parse_create_function("CREATE TABLE t (id INT)")
                .unwrap()
                .is_none()
        );
    }
}

#[cfg(test)]
mod parse_explain_kind_tests {
    use super::parse_explain_kind;

    #[test]
    fn parse_explain_kind_detects_bare_analyze() {
        let (is_analyze, inner) =
            parse_explain_kind("EXPLAIN ANALYZE INSERT INTO t SELECT * FROM s;").unwrap();
        assert!(is_analyze);
        assert_eq!(inner, "INSERT INTO t SELECT * FROM s;");
    }

    #[test]
    fn parse_explain_kind_detects_parenthesized_analyze() {
        let (is_analyze, inner) =
            parse_explain_kind("EXPLAIN (ANALYZE, VERBOSE) INSERT INTO t SELECT * FROM s;")
                .unwrap();
        assert!(is_analyze);
        assert_eq!(inner, "INSERT INTO t SELECT * FROM s;");
    }

    #[test]
    fn parse_explain_kind_plain_explain_returns_false() {
        let (is_analyze, inner) =
            parse_explain_kind("EXPLAIN INSERT INTO t SELECT * FROM s;").unwrap();
        assert!(!is_analyze);
        assert_eq!(inner, "INSERT INTO t SELECT * FROM s;");
    }

    #[test]
    fn parse_explain_kind_non_explain_returns_none() {
        assert!(parse_explain_kind("SELECT * FROM t").is_none());
    }
}
