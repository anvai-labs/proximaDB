//! # P4 — Shared Logical Lowering: `LogicalNode` → DataFusion `LogicalPlan`
//!
//! Lowers ProximaDB's relational algebra (`proximadb_relational_algebra::LogicalNode`,
//! produced by the shared frontend that also feeds the Volcano executor) into a DataFusion
//! `LogicalPlan`. This discharges the course-correction §5 "shared logical plane, split
//! physical plane" guarantee in code: SQL and (later) the PySpark-style DataFrame API lower
//! to ONE algebra, and the router picks the physical engine — Volcano runs the `LogicalNode`
//! directly, DataFusion runs the lowered `LogicalPlan`.
//!
//! ## Scope
//! Covers the OLAP shapes the P1 route targets:
//! `Scan / Filter / Project / Aggregate / Sort / Limit / Distinct / Union` and
//! `Join` (Inner/Left/Right/Full/Cross via `join_on`/`cross_join`, plus
//! Semi/Anti via `LeftSemi`/`LeftAnti` — the decorrelated IN/EXISTS targets),
//! `SetOp` (`INTERSECT`/`EXCEPT`, `ALL` preserving multiset dups),
//! `Values` (inline literal rows, aliased to the algebra's output names) and
//! `WITH` CTEs (`CteBind`/`CteRef`, resolved by inlining the body at each ref),
//! with `Expr` translation for `Column / Literal / BinaryOp / UnaryOp / IsNull /
//! Cast / Between / In / Like / Case / Coalesce / NullIf` and the common scalar
//! functions `UPPER/LOWER/LENGTH/ABS/CEIL/FLOOR/SQRT/CONCAT`. Anything else
//! (null-aware anti join / `NOT IN`, scalar-subquery `AssertMaxOneRow` guard;
//! uncommon/variadic `FuncCall`s; `StringAgg/Custom` aggregates) returns
//! [`DataFusionError::NotImplemented`] so the caller keeps the existing
//! `ctx.sql(...)` path for those — additive, never wrong.

use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{
    Between, Cast, Expr, JoinType, Like, LogicalPlan, LogicalPlanBuilder, Operator, binary_expr,
    col, lit,
};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;

use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_relational_algebra::{
    AggregateExpr, JoinKind, LogicalNode, NamedAggregate, SetOpKind,
};
use proximadb_relational_types::{BinaryOp as RBinOp, Expr as RExpr, UnaryOp as RUnOp};

fn unsupported(what: impl Into<String>) -> DataFusionError {
    DataFusionError::NotImplemented(format!("logical_lowering: {}", what.into()))
}

/// Lower a relational `LogicalNode` to a DataFusion `LogicalPlan`, resolving `Scan` leaves
/// against the tables registered in `ctx`.
pub async fn lower_logical_node(ctx: &SessionContext, node: &LogicalNode) -> DFResult<LogicalPlan> {
    // Resolve CTEs first (inline each `CteRef` with its bound body) so `lower`
    // only ever sees a `CteBind`/`CteRef`-free tree.
    let inlined = inline_ctes(node, &[])?;
    lower(ctx, &inlined).await
}

/// Resolve `WITH` CTEs by inlining: substitute every `CteRef` with a clone of its
/// bound body, producing a `CteBind`/`CteRef`-free tree. This is the "single-use
/// always inline" cut — DataFusion's optimizer can re-share identical subtrees, and
/// any unbound reference returns `NotImplemented` so the caller keeps the `ctx.sql`
/// fallback. `env` is the lexical binding stack (innermost last); a CTE body is
/// resolved in the scope where it is bound, so later/inner CTEs may reference earlier
/// ones and inner names shadow outer ones.
fn inline_ctes(node: &LogicalNode, env: &[(String, LogicalNode)]) -> DFResult<LogicalNode> {
    let recur = |n: &LogicalNode| inline_ctes(n, env);
    Ok(match node {
        LogicalNode::CteBind { name, body, usages } => {
            // Resolve the body in the CURRENT scope, then bind it for `usages`.
            let resolved_body = inline_ctes(body, env)?;
            let mut inner_env = env.to_vec();
            inner_env.push((name.clone(), resolved_body));
            inline_ctes(usages, &inner_env)?
        }
        LogicalNode::CteRef { name, .. } => env
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, body)| body.clone())
            .ok_or_else(|| unsupported(format!("unbound CTE reference {name}")))?,
        // Leaves: no `LogicalNode` children to descend into.
        LogicalNode::Scan { .. } | LogicalNode::Values { .. } => node.clone(),
        LogicalNode::Filter { input, predicate } => LogicalNode::Filter {
            input: Box::new(recur(input)?),
            predicate: predicate.clone(),
        },
        LogicalNode::Project { input, outputs } => LogicalNode::Project {
            input: Box::new(recur(input)?),
            outputs: outputs.clone(),
        },
        LogicalNode::Join {
            left,
            right,
            kind,
            on,
            strategy,
        } => LogicalNode::Join {
            left: Box::new(recur(left)?),
            right: Box::new(recur(right)?),
            kind: *kind,
            on: on.clone(),
            strategy: *strategy,
        },
        LogicalNode::Aggregate {
            input,
            group_by,
            aggregates,
            having,
        } => LogicalNode::Aggregate {
            input: Box::new(recur(input)?),
            group_by: group_by.clone(),
            aggregates: aggregates.clone(),
            having: having.clone(),
        },
        LogicalNode::Sort { input, keys } => LogicalNode::Sort {
            input: Box::new(recur(input)?),
            keys: keys.clone(),
        },
        LogicalNode::Limit {
            input,
            limit,
            offset,
        } => LogicalNode::Limit {
            input: Box::new(recur(input)?),
            limit: *limit,
            offset: *offset,
        },
        LogicalNode::Distinct { input } => LogicalNode::Distinct {
            input: Box::new(recur(input)?),
        },
        LogicalNode::AssertMaxOneRow { input } => LogicalNode::AssertMaxOneRow {
            input: Box::new(recur(input)?),
        },
        LogicalNode::Union { inputs, all } => LogicalNode::Union {
            inputs: inputs
                .iter()
                .map(|n| inline_ctes(n, env))
                .collect::<DFResult<Vec<_>>>()?,
            all: *all,
        },
        LogicalNode::SetOp {
            op,
            left,
            right,
            all,
        } => LogicalNode::SetOp {
            op: *op,
            left: Box::new(recur(left)?),
            right: Box::new(recur(right)?),
            all: *all,
        },
    })
}

// Boxed recursion: an `async fn` calling itself needs an explicit boxed future.
fn lower<'a>(
    ctx: &'a SessionContext,
    node: &'a LogicalNode,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = DFResult<LogicalPlan>> + Send + 'a>> {
    Box::pin(async move {
        match node {
            LogicalNode::Scan { table, .. } => {
                // Resolve the registered table provider by name and build a scan over it.
                // Tables register under their normalized key (lowercase, unquoted), so a
                // differently-cased or quoted query spelling retries folded (TD-OLAP-18).
                let (provider, scan_name) = match ctx.table_provider(table.name.as_str()).await {
                    Ok(p) => (p, table.name.clone()),
                    Err(e) => {
                        let folded = table.name.trim_matches('"').to_ascii_lowercase();
                        if folded == table.name {
                            return Err(e);
                        }
                        let p = ctx.table_provider(folded.as_str()).await.map_err(|_| e)?;
                        (p, folded)
                    }
                };
                let source = datafusion::datasource::provider_as_source(provider);
                LogicalPlanBuilder::scan(scan_name.as_str(), source, None)?.build()
            }
            LogicalNode::Filter { input, predicate } => {
                let input = lower(ctx, input).await?;
                LogicalPlanBuilder::from(input)
                    .filter(lower_expr(predicate)?)?
                    .build()
            }
            LogicalNode::Project { input, outputs } => {
                let input = lower(ctx, input).await?;
                let exprs = outputs
                    .iter()
                    .map(|o| Ok(lower_expr(&o.expr)?.alias(&o.name)))
                    .collect::<DFResult<Vec<Expr>>>()?;
                LogicalPlanBuilder::from(input).project(exprs)?.build()
            }
            LogicalNode::Aggregate {
                input,
                group_by,
                aggregates,
                having,
            } => {
                let input = lower(ctx, input).await?;
                let groups = group_by
                    .iter()
                    .map(|g| lower_expr(&g.expr))
                    .collect::<DFResult<Vec<Expr>>>()?;
                let aggs = aggregates
                    .iter()
                    .map(lower_aggregate)
                    .collect::<DFResult<Vec<Expr>>>()?;
                let mut builder = LogicalPlanBuilder::from(input).aggregate(groups, aggs)?;
                if let Some(h) = having {
                    builder = builder.filter(lower_expr(h)?)?;
                }
                builder.build()
            }
            LogicalNode::Sort { input, keys } => {
                let input = lower(ctx, input).await?;
                let sorts = keys
                    .iter()
                    .map(|k| Ok(lower_expr(&k.expr)?.sort(!k.descending, k.nulls_first)))
                    .collect::<DFResult<Vec<_>>>()?;
                LogicalPlanBuilder::from(input).sort(sorts)?.build()
            }
            LogicalNode::Limit {
                input,
                limit,
                offset,
            } => {
                let input = lower(ctx, input).await?;
                let fetch = limit.map(|l| l as usize);
                LogicalPlanBuilder::from(input)
                    .limit(*offset as usize, fetch)?
                    .build()
            }
            LogicalNode::Join {
                left,
                right,
                kind,
                on,
                strategy: _, // physical hint; DataFusion picks its own join algorithm
            } => {
                let left_plan = lower(ctx, left).await?;
                let right_plan = lower(ctx, right).await?;
                let join_type = match kind {
                    JoinKind::Inner => JoinType::Inner,
                    JoinKind::Left => JoinType::Left,
                    JoinKind::Right => JoinType::Right,
                    JoinKind::Full => JoinType::Full,
                    // CROSS JOIN carries no ON predicate.
                    JoinKind::Cross => {
                        return LogicalPlanBuilder::from(left_plan)
                            .cross_join(right_plan)?
                            .build();
                    }
                    // Semi/Anti come from decorrelated IN / EXISTS / NOT EXISTS
                    // subqueries (correlated EXISTS/IN decorrelation is live in the
                    // relational engine). They carry the correlation/IN-key ON
                    // predicate and project only the left schema → lower directly to
                    // DataFusion's LeftSemi / LeftAnti instead of the ctx.sql fallback.
                    JoinKind::Semi | JoinKind::Anti => {
                        let semi_join_type = if matches!(kind, JoinKind::Semi) {
                            JoinType::LeftSemi
                        } else {
                            JoinType::LeftAnti
                        };
                        let predicate = on.as_ref().ok_or_else(|| {
                            unsupported("semi/anti join without ON predicate (use ctx.sql path)")
                        })?;
                        return LogicalPlanBuilder::from(left_plan)
                            .join_on(right_plan, semi_join_type, [lower_expr(predicate)?])?
                            .build();
                    }
                    // Null-aware anti join is the `NOT IN (subquery)` target, whose
                    // SQL three-valued logic (a NULL in the right relation makes
                    // `NOT IN` yield no rows) DataFusion's LeftAnti does NOT
                    // implement. Keep it on the `ctx.sql` fallback (correct, never wrong).
                    JoinKind::AntiNullAware => {
                        return Err(unsupported("null-aware anti join (use ctx.sql path)"));
                    }
                };
                match on {
                    // A join condition the lowering can't translate errors out → the
                    // caller falls back to `ctx.sql` (additive, never wrong).
                    Some(predicate) => LogicalPlanBuilder::from(left_plan)
                        .join_on(right_plan, join_type, [lower_expr(predicate)?])?
                        .build(),
                    // No ON predicate on an inner-family join == a cross join.
                    None => LogicalPlanBuilder::from(left_plan)
                        .cross_join(right_plan)?
                        .build(),
                }
            }
            LogicalNode::Distinct { input } => {
                let input = lower(ctx, input).await?;
                LogicalPlanBuilder::from(input).distinct()?.build()
            }
            LogicalNode::Union { inputs, all } => {
                let mut it = inputs.iter();
                let first = match it.next() {
                    Some(n) => lower(ctx, n).await?,
                    None => return Err(unsupported("empty Union")),
                };
                let mut builder = LogicalPlanBuilder::from(first);
                for node in it {
                    let plan = lower(ctx, node).await?;
                    // `all` selects UNION ALL vs UNION (distinct). All inputs share
                    // a schema (algebra invariant), so DataFusion's union accepts them.
                    builder = if *all {
                        builder.union(plan)?
                    } else {
                        builder.union_distinct(plan)?
                    };
                }
                builder.build()
            }
            LogicalNode::SetOp {
                op,
                left,
                right,
                all,
            } => {
                let left_plan = lower(ctx, left).await?;
                let right_plan = lower(ctx, right).await?;
                // Both inputs share a schema (algebra invariant); `all` selects
                // multiset (preserve dups) vs set semantics. `intersect`/`except`
                // are static assoc fns returning the finished `LogicalPlan`.
                match op {
                    SetOpKind::Intersect => {
                        LogicalPlanBuilder::intersect(left_plan, right_plan, *all)
                    }
                    SetOpKind::Except => LogicalPlanBuilder::except(left_plan, right_plan, *all),
                }
            }
            LogicalNode::Values {
                rows,
                output_schema,
            } => {
                if rows.is_empty() {
                    return Err(unsupported("empty Values"));
                }
                // Lower each literal row to DataFusion exprs.
                let lowered_rows = rows
                    .iter()
                    .map(|row| row.iter().map(lower_expr).collect::<DFResult<Vec<_>>>())
                    .collect::<DFResult<Vec<_>>>()?;
                // DataFusion names VALUES columns `column1`, `column2`, … (Postgres
                // convention). Re-alias to the algebra's `output_schema` names so
                // downstream column references resolve.
                let aliases = output_schema
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(i, c)| col(format!("column{}", i + 1)).alias(&c.name))
                    .collect::<Vec<Expr>>();
                LogicalPlanBuilder::values(lowered_rows)?
                    .project(aliases)?
                    .build()
            }
            // `inline_ctes` (run by `lower_logical_node` before `lower`) strips these,
            // so they are unreachable here — kept as a defensive guard in case `lower`
            // is ever reached on an un-inlined tree.
            LogicalNode::CteBind { .. } => Err(unsupported("CteBind (should be inlined)")),
            LogicalNode::CteRef { .. } => Err(unsupported("CteRef (should be inlined)")),
            // Scalar-subquery cardinality guard — DataFusion serves this via the
            // `ctx.sql` fallback until it is lowered on the shared path. Explicit arm
            // (not a wildcard) so the next new LogicalNode variant forces a deliberate
            // decision here, not silent rot.
            LogicalNode::AssertMaxOneRow { .. } => Err(unsupported("AssertMaxOneRow")),
        }
    })
}

/// Translate a named aggregate to a DataFusion aggregate `Expr`, aliased to its output name.
fn lower_aggregate(named: &NamedAggregate) -> DFResult<Expr> {
    use datafusion::functions_aggregate::expr_fn::{avg, count, count_distinct, max, min, sum};
    let expr = match &named.agg {
        AggregateExpr::Count { arg, distinct } => match (arg, distinct) {
            (None, _) => count(lit(1i64)), // COUNT(*) — count of a constant
            (Some(a), true) => count_distinct(lower_expr(a)?),
            (Some(a), false) => count(lower_expr(a)?),
        },
        AggregateExpr::Sum { arg, distinct } => {
            let e = lower_expr(arg)?;
            if *distinct {
                return Err(unsupported("SUM(DISTINCT)"));
            }
            sum(e)
        }
        AggregateExpr::Avg { arg, distinct } => {
            let e = lower_expr(arg)?;
            if *distinct {
                return Err(unsupported("AVG(DISTINCT)"));
            }
            avg(e)
        }
        AggregateExpr::Min { arg } => min(lower_expr(arg)?),
        AggregateExpr::Max { arg } => max(lower_expr(arg)?),
        AggregateExpr::StringAgg { .. } => return Err(unsupported("StringAgg")),
        // F3b: a registry aggregate (e.g. `product`) binds to its DataFusion AggregateUDF.
        AggregateExpr::Custom {
            name,
            args,
            distinct,
            ..
        } => {
            if *distinct {
                return Err(unsupported(format!("{name}(DISTINCT)")));
            }
            match proximadb_functions::builtins().lookup_aggregate(name) {
                Some(def) => {
                    let udf = std::sync::Arc::new(super::registry_udf::proxima_aggregate_udf(def));
                    let lowered = args.iter().map(lower_expr).collect::<DFResult<Vec<_>>>()?;
                    udf.call(lowered)
                }
                None => return Err(unsupported(format!("custom aggregate {name}"))),
            }
        }
    };
    Ok(expr.alias(&named.name))
}

/// Translate a relational `Expr` to a DataFusion `Expr` (first-slice subset).
fn lower_expr(e: &RExpr) -> DFResult<Expr> {
    Ok(match e {
        // `Column::new_unqualified` keeps the DECLARED name verbatim. `col()`
        // would re-parse it as a SQL identifier and fold unquoted names to
        // lowercase (DataFusion planner semantics), so a CamelCase Parquet
        // column (`AdvEngineID`) would never resolve (TD-OLAP-18).
        RExpr::Column(c) => Expr::Column(datafusion::common::Column::new_unqualified(&c.name)),
        RExpr::Literal { value, .. } => lit(proxima_value_to_scalar(value)?),
        RExpr::BinaryOp { op, left, right } => {
            let l = lower_expr(left)?;
            let r = lower_expr(right)?;
            binary_expr(l, lower_binary_op(op)?, r)
        }
        RExpr::UnaryOp { op, expr } => {
            let inner = lower_expr(expr)?;
            match op {
                RUnOp::Not => !inner,
                RUnOp::Neg => Expr::Negative(Box::new(inner)),
            }
        }
        RExpr::IsNull { expr, not } => {
            let inner = lower_expr(expr)?;
            if *not {
                inner.is_not_null()
            } else {
                inner.is_null()
            }
        }
        RExpr::Cast { expr, ty } => Expr::Cast(Cast::new(
            Box::new(lower_expr(expr)?),
            proxima_to_arrow(ty)?,
        )),
        RExpr::Between {
            expr,
            low,
            high,
            not,
        } => Expr::Between(Between::new(
            Box::new(lower_expr(expr)?),
            *not,
            Box::new(lower_expr(low)?),
            Box::new(lower_expr(high)?),
        )),
        RExpr::In { expr, list, not } => {
            let items = list.iter().map(lower_expr).collect::<DFResult<Vec<_>>>()?;
            Expr::InList(datafusion::logical_expr::expr::InList::new(
                Box::new(lower_expr(expr)?),
                items,
                *not,
            ))
        }
        RExpr::Like {
            expr,
            pattern,
            not,
            case_insensitive,
        } => Expr::Like(Like::new(
            *not,
            Box::new(lower_expr(expr)?),
            Box::new(lower_expr(pattern)?),
            None, // no custom ESCAPE
            *case_insensitive,
        )),
        RExpr::Case {
            branches,
            otherwise,
        } => {
            use datafusion::logical_expr::Case;
            // Searched CASE (`CASE WHEN cond THEN result ...`) — operand is None.
            let when_then = branches
                .iter()
                .map(|(c, r)| Ok((Box::new(lower_expr(c)?), Box::new(lower_expr(r)?))))
                .collect::<DFResult<Vec<_>>>()?;
            let else_expr = otherwise
                .as_ref()
                .map(|e| lower_expr(e).map(Box::new))
                .transpose()?;
            Expr::Case(Case::new(None, when_then, else_expr))
        }
        RExpr::Coalesce(args) => {
            let lowered = args.iter().map(lower_expr).collect::<DFResult<Vec<_>>>()?;
            datafusion::functions::core::expr_fn::coalesce(lowered)
        }
        RExpr::NullIf { left, right } => {
            datafusion::functions::core::expr_fn::nullif(lower_expr(left)?, lower_expr(right)?)
        }
        RExpr::FuncCall { name, args, .. } => {
            let lowered = args.iter().map(lower_expr).collect::<DFResult<Vec<_>>>()?;
            lower_scalar_function(name, lowered)?
        }
    })
}

/// Map a builtin scalar-function call (case-insensitive name) to a DataFusion
/// scalar `Expr`. Covers the high-frequency analytical builtins with predictable
/// arities; unknown names or unsupported arities return `NotImplemented` so the
/// caller keeps the `ctx.sql` fallback (which has DataFusion's full function set).
fn lower_scalar_function(name: &str, args: Vec<Expr>) -> DFResult<Expr> {
    use datafusion::functions::expr_fn as f;
    fn one(name: &str, mut args: Vec<Expr>) -> DFResult<Expr> {
        match args.len() {
            1 => args
                .pop()
                .ok_or_else(|| unsupported(format!("{name} expects 1 arg"))),
            n => Err(unsupported(format!("{name} expects 1 arg, got {n}"))),
        }
    }
    Ok(match name.to_ascii_lowercase().as_str() {
        "upper" | "ucase" => f::upper(one(name, args)?),
        "lower" | "lcase" => f::lower(one(name, args)?),
        "length" | "char_length" | "character_length" => f::char_length(one(name, args)?),
        "abs" => f::abs(one(name, args)?),
        "ceil" | "ceiling" => f::ceil(one(name, args)?),
        "floor" => f::floor(one(name, args)?),
        "sqrt" => f::sqrt(one(name, args)?),
        "concat" => f::concat(args), // variadic
        "substr" | "substring" => datafusion::functions::unicode::substr().call(args),
        // JSON extraction is a registry function (for native + frontend typing), but on the
        // OLAP side it has dedicated Arrow-vectorized UDFs (`udf.rs`). Bind THOSE here — NOT
        // the engine-neutral kernel via the F2 fallback below, whose untyped (0-arity)
        // signature would shadow the typed `(Utf8,Utf8)` UDF and break coercion in comparison
        // contexts (ADR-043 Inv 3; mirrors the `DATAFUSION_NATIVE_SCALARS` skip-list).
        "json_extract" => {
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                std::sync::Arc::new(super::udf::json_extract_udf()),
                args,
            ))
        }
        "json_extract_text" => {
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                std::sync::Arc::new(super::udf::json_extract_text_udf()),
                args,
            ))
        }
        "json_extract_path_text" => {
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                std::sync::Arc::new(super::udf::json_extract_path_text_udf()),
                args,
            ))
        }
        // F2 fallback: a registry function DataFusion lacks natively (custom functions; and via
        // F4/F5 vector distances + user CREATE FUNCTIONs). Bind its engine-neutral kernel as a
        // ScalarUDF and call it. Variadic registry funcs aren't fixed-arity-adaptable here.
        other => match proximadb_functions::builtins().lookup_scalar(other) {
            Some(def) if !def.signature.variadic => {
                let udf = std::sync::Arc::new(super::registry_udf::proxima_scalar_udf(def));
                Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                    udf, args,
                ))
            }
            _ => return Err(unsupported(format!("function {other}"))),
        },
    })
}

/// `ProximaType` → Arrow `DataType` for `CAST` targets (inverse of the pgwire
/// route's `arrow_type_to_proxima`). Types without a direct Arrow scalar mapping
/// return `NotImplemented`, so a cast to them keeps the `ctx.sql` fallback.
fn proxima_to_arrow(ty: &ProximaType) -> DFResult<arrow_schema::DataType> {
    use ProximaType as P;
    use arrow_schema::DataType as D;
    Ok(match ty {
        P::Boolean => D::Boolean,
        P::Int8 => D::Int8,
        P::Int16 => D::Int16,
        P::Int32 => D::Int32,
        P::Int64 => D::Int64,
        P::UInt8 => D::UInt8,
        P::UInt16 => D::UInt16,
        P::UInt32 => D::UInt32,
        P::UInt64 => D::UInt64,
        P::Float32 => D::Float32,
        P::Float64 => D::Float64,
        P::String => D::Utf8,
        P::Binary => D::Binary,
        P::Date => D::Date32,
        other => return Err(unsupported(format!("cast to {other:?}"))),
    })
}

fn lower_binary_op(op: &RBinOp) -> DFResult<Operator> {
    Ok(match op {
        RBinOp::Plus => Operator::Plus,
        RBinOp::Minus => Operator::Minus,
        RBinOp::Mul => Operator::Multiply,
        RBinOp::Div => Operator::Divide,
        RBinOp::Mod => Operator::Modulo,
        RBinOp::Eq => Operator::Eq,
        RBinOp::NotEq => Operator::NotEq,
        RBinOp::Lt => Operator::Lt,
        RBinOp::LtEq => Operator::LtEq,
        RBinOp::Gt => Operator::Gt,
        RBinOp::GtEq => Operator::GtEq,
        RBinOp::And => Operator::And,
        RBinOp::Or => Operator::Or,
        RBinOp::Concat => return Err(unsupported("|| (concat)")),
    })
}

/// Convert a `ProximaValue` literal to a DataFusion `ScalarValue` (common scalar types).
pub(crate) fn proxima_value_to_scalar(v: &ProximaValue) -> DFResult<ScalarValue> {
    Ok(match v {
        ProximaValue::Null => ScalarValue::Null,
        ProximaValue::Boolean(b) => ScalarValue::Boolean(Some(*b)),
        ProximaValue::Int8(x) => ScalarValue::Int8(Some(*x)),
        ProximaValue::Int16(x) => ScalarValue::Int16(Some(*x)),
        ProximaValue::Int32(x) => ScalarValue::Int32(Some(*x)),
        ProximaValue::Int64(x) => ScalarValue::Int64(Some(*x)),
        ProximaValue::UInt8(x) => ScalarValue::UInt8(Some(*x)),
        ProximaValue::UInt16(x) => ScalarValue::UInt16(Some(*x)),
        ProximaValue::UInt32(x) => ScalarValue::UInt32(Some(*x)),
        ProximaValue::UInt64(x) => ScalarValue::UInt64(Some(*x)),
        ProximaValue::Float32(x) => ScalarValue::Float32(Some(*x)),
        ProximaValue::Float64(x) => ScalarValue::Float64(Some(*x)),
        ProximaValue::String(s) | ProximaValue::Symbol(s) => ScalarValue::Utf8(Some(s.clone())),
        other => return Err(unsupported(format!("literal {other:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use proximadb_data_model::ProximaType;
    use proximadb_relational_algebra::{
        JoinKind, JoinStrategy, LogicalNode, NamedExpr, SortKey, TableId,
    };
    use proximadb_relational_types::{ColumnInfo, ColumnRef, RelationalSchema};
    use std::sync::Arc;

    fn colref(name: &str, ordinal: usize, ty: ProximaType) -> ColumnRef {
        ColumnRef {
            name: name.to_string(),
            ordinal,
            ty,
            nullable: false,
        }
    }

    async fn ctx_with_t() -> SessionContext {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 10.0])),
            ],
        )
        .unwrap();
        let mem = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(mem)).unwrap();
        ctx
    }

    fn scan_t() -> LogicalNode {
        LogicalNode::Scan {
            table: TableId::new("t"),
            table_schema: RelationalSchema::new(vec![
                ColumnInfo::new("k", ProximaType::String, false),
                ColumnInfo::new("x", ProximaType::Float64, false),
            ]),
            projected_columns: None,
            predicate: None,
        }
    }

    #[tokio::test]
    async fn lowers_scan_filter_project() {
        let ctx = ctx_with_t().await;
        // SELECT k FROM t WHERE x > 2.0
        let node = LogicalNode::Project {
            input: Box::new(LogicalNode::Filter {
                input: Box::new(scan_t()),
                predicate: RExpr::BinaryOp {
                    op: RBinOp::Gt,
                    left: Box::new(RExpr::Column(colref("x", 1, ProximaType::Float64))),
                    right: Box::new(RExpr::Literal {
                        value: ProximaValue::Float64(2.0),
                        ty: ProximaType::Float64,
                    }),
                },
            }),
            outputs: vec![NamedExpr {
                name: "k".to_string(),
                expr: RExpr::Column(colref("k", 0, ProximaType::String)),
            }],
        };
        let plan = lower_logical_node(&ctx, &node).await.unwrap();
        let batches = ctx
            .execute_logical_plan(plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2); // x=3.0 (a) and x=10.0 (b)
    }

    #[tokio::test]
    async fn lowers_inner_join_on() {
        // t(k,x) ⋈ u(j,y) ON k = j  (distinct join-key names avoid ambiguity).
        let ctx = SessionContext::new();
        let t_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("x", DataType::Float64, false),
        ]));
        let t_batch = RecordBatch::try_new(
            t_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
                Arc::new(Float64Array::from(vec![1.0, 3.0, 10.0])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "t",
            Arc::new(MemTable::try_new(t_schema, vec![vec![t_batch]]).unwrap()),
        )
        .unwrap();
        let u_schema = Arc::new(Schema::new(vec![
            Field::new("j", DataType::Utf8, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let u_batch = RecordBatch::try_new(
            u_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "b"])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .unwrap();
        ctx.register_table(
            "u",
            Arc::new(MemTable::try_new(u_schema, vec![vec![u_batch]]).unwrap()),
        )
        .unwrap();

        let scan_u = LogicalNode::Scan {
            table: TableId::new("u"),
            table_schema: RelationalSchema::new(vec![
                ColumnInfo::new("j", ProximaType::String, false),
                ColumnInfo::new("y", ProximaType::Float64, false),
            ]),
            projected_columns: None,
            predicate: None,
        };
        let node = LogicalNode::Join {
            left: Box::new(scan_t()),
            right: Box::new(scan_u),
            kind: JoinKind::Inner,
            on: Some(RExpr::BinaryOp {
                op: RBinOp::Eq,
                left: Box::new(RExpr::Column(colref("k", 0, ProximaType::String))),
                right: Box::new(RExpr::Column(colref("j", 0, ProximaType::String))),
            }),
            strategy: JoinStrategy::Auto,
        };
        let plan = lower_logical_node(&ctx, &node).await.unwrap();
        let batches = ctx
            .execute_logical_plan(plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // a⋈a (t has 2 a-rows × u has 1 a-row = 2) + b⋈b (1 × 2 = 2) = 4.
        assert_eq!(total, 4);
    }

    #[tokio::test]
    async fn lowers_semi_and_anti_join() {
        // t(k,x) = {a,a,b}; u(j) = {a}. SEMI t ON k=j keeps t rows with a match
        // (the two a-rows → 2); ANTI keeps t rows with NO match (the b-row → 1).
        // These are the decorrelated IN / NOT EXISTS lowering targets — they now
        // lower to DataFusion LeftSemi/LeftAnti instead of the ctx.sql fallback.
        let ctx = ctx_with_t().await; // registers t(k,x) = a/a/b
        let u_schema = Arc::new(Schema::new(vec![Field::new("j", DataType::Utf8, false)]));
        let u_batch = RecordBatch::try_new(
            u_schema.clone(),
            vec![Arc::new(StringArray::from(vec!["a"]))],
        )
        .unwrap();
        ctx.register_table(
            "u",
            Arc::new(MemTable::try_new(u_schema, vec![vec![u_batch]]).unwrap()),
        )
        .unwrap();

        let scan_u = LogicalNode::Scan {
            table: TableId::new("u"),
            table_schema: RelationalSchema::new(vec![ColumnInfo::new(
                "j",
                ProximaType::String,
                false,
            )]),
            projected_columns: None,
            predicate: None,
        };
        let on = RExpr::BinaryOp {
            op: RBinOp::Eq,
            left: Box::new(RExpr::Column(colref("k", 0, ProximaType::String))),
            right: Box::new(RExpr::Column(colref("j", 0, ProximaType::String))),
        };
        let mk = |kind| LogicalNode::Join {
            left: Box::new(scan_t()),
            right: Box::new(scan_u.clone()),
            kind,
            on: Some(on.clone()),
            strategy: JoinStrategy::Auto,
        };

        let semi_plan = lower_logical_node(&ctx, &mk(JoinKind::Semi)).await.unwrap();
        let semi_rows: usize = ctx
            .execute_logical_plan(semi_plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(semi_rows, 2, "SEMI keeps the two matching a-rows");

        let anti_plan = lower_logical_node(&ctx, &mk(JoinKind::Anti)).await.unwrap();
        let anti_rows: usize = ctx
            .execute_logical_plan(anti_plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(anti_rows, 1, "ANTI keeps the non-matching b-row");

        // Null-aware anti (NOT IN) must remain on the ctx.sql fallback (errors here).
        assert!(
            lower_logical_node(&ctx, &mk(JoinKind::AntiNullAware))
                .await
                .is_err(),
            "null-aware anti must NOT be lowered (correctness: NOT IN three-valued logic)"
        );
    }

    #[tokio::test]
    async fn lowers_cast_between_in_like() {
        // SELECT k FROM t
        //  WHERE x BETWEEN 1.0 AND 3.0  AND k IN ('a')  AND k LIKE 'a%'
        //    AND CAST(x AS BIGINT) = 1
        // t = (a,1),(a,3),(b,10). Between→(a,1),(a,3); IN('a')→both; LIKE 'a%'→both;
        // CAST(x AS Int64)=1 → only (a,1). Expect 1 row.
        let ctx = ctx_with_t().await;
        let str_lit = |s: &str| RExpr::Literal {
            value: ProximaValue::String(s.to_string()),
            ty: ProximaType::String,
        };
        let f64_lit = |v: f64| RExpr::Literal {
            value: ProximaValue::Float64(v),
            ty: ProximaType::Float64,
        };
        let and = |l: RExpr, r: RExpr| RExpr::BinaryOp {
            op: RBinOp::And,
            left: Box::new(l),
            right: Box::new(r),
        };
        let kcol = || RExpr::Column(colref("k", 0, ProximaType::String));
        let xcol = || RExpr::Column(colref("x", 1, ProximaType::Float64));

        let predicate = and(
            and(
                and(
                    RExpr::Between {
                        expr: Box::new(xcol()),
                        low: Box::new(f64_lit(1.0)),
                        high: Box::new(f64_lit(3.0)),
                        not: false,
                    },
                    RExpr::In {
                        expr: Box::new(kcol()),
                        list: vec![str_lit("a")],
                        not: false,
                    },
                ),
                RExpr::Like {
                    expr: Box::new(kcol()),
                    pattern: Box::new(str_lit("a%")),
                    not: false,
                    case_insensitive: false,
                },
            ),
            RExpr::BinaryOp {
                op: RBinOp::Eq,
                left: Box::new(RExpr::Cast {
                    expr: Box::new(xcol()),
                    ty: ProximaType::Int64,
                }),
                right: Box::new(RExpr::Literal {
                    value: ProximaValue::Int64(1),
                    ty: ProximaType::Int64,
                }),
            },
        );
        let node = LogicalNode::Filter {
            input: Box::new(scan_t()),
            predicate,
        };
        let plan = lower_logical_node(&ctx, &node).await.unwrap();
        let batches = ctx
            .execute_logical_plan(plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1); // only (a, 1)
    }

    fn project_k() -> LogicalNode {
        LogicalNode::Project {
            input: Box::new(scan_t()),
            outputs: vec![NamedExpr {
                name: "k".to_string(),
                expr: RExpr::Column(colref("k", 0, ProximaType::String)),
            }],
        }
    }

    async fn row_count(ctx: &SessionContext, node: &LogicalNode) -> usize {
        let plan = lower_logical_node(ctx, node).await.unwrap();
        let batches = ctx
            .execute_logical_plan(plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        batches.iter().map(|b| b.num_rows()).sum()
    }

    #[tokio::test]
    async fn lowers_distinct() {
        // SELECT DISTINCT k FROM t  →  t.k = {a,a,b}  →  {a,b} = 2 rows.
        let ctx = ctx_with_t().await;
        let node = LogicalNode::Distinct {
            input: Box::new(project_k()),
        };
        assert_eq!(row_count(&ctx, &node).await, 2);
    }

    #[tokio::test]
    async fn lowers_scalar_functions() {
        // SELECT UPPER(k), LENGTH(k), ABS(x), CONCAT(k,'!') FROM t — executes over 3 rows.
        let ctx = ctx_with_t().await;
        let func = |name: &str, args: Vec<RExpr>| RExpr::FuncCall {
            name: name.to_string(),
            args,
            return_ty: ProximaType::String,
        };
        let k = || RExpr::Column(colref("k", 0, ProximaType::String));
        let node = LogicalNode::Project {
            input: Box::new(scan_t()),
            outputs: vec![
                NamedExpr {
                    name: "u".to_string(),
                    expr: func("upper", vec![k()]),
                },
                NamedExpr {
                    name: "len".to_string(),
                    expr: func("length", vec![k()]),
                },
                NamedExpr {
                    name: "a".to_string(),
                    expr: func(
                        "abs",
                        vec![RExpr::Column(colref("x", 1, ProximaType::Float64))],
                    ),
                },
                NamedExpr {
                    name: "c".to_string(),
                    expr: func(
                        "concat",
                        vec![
                            k(),
                            RExpr::Literal {
                                value: ProximaValue::String("!".to_string()),
                                ty: ProximaType::String,
                            },
                        ],
                    ),
                },
            ],
        };
        assert_eq!(row_count(&ctx, &node).await, 3);
    }

    #[tokio::test]
    async fn lowers_union_all_and_distinct() {
        let ctx = ctx_with_t().await;
        // (SELECT k FROM t) UNION ALL (SELECT k FROM t) → 3 + 3 = 6.
        let union_all = LogicalNode::Union {
            inputs: vec![project_k(), project_k()],
            all: true,
        };
        assert_eq!(row_count(&ctx, &union_all).await, 6);
        // …UNION (distinct) → {a,b} = 2.
        let union_distinct = LogicalNode::Union {
            inputs: vec![project_k(), project_k()],
            all: false,
        };
        assert_eq!(row_count(&ctx, &union_distinct).await, 2);
    }

    // SELECT k FROM t WHERE k = <val>  — a filtered projection of just `k`,
    // schema-matching `project_k()` so set ops can combine them.
    fn project_k_where(val: &str) -> LogicalNode {
        LogicalNode::Project {
            input: Box::new(LogicalNode::Filter {
                input: Box::new(scan_t()),
                predicate: RExpr::BinaryOp {
                    op: RBinOp::Eq,
                    left: Box::new(RExpr::Column(colref("k", 0, ProximaType::String))),
                    right: Box::new(RExpr::Literal {
                        value: ProximaValue::String(val.to_string()),
                        ty: ProximaType::String,
                    }),
                },
            }),
            outputs: vec![NamedExpr {
                name: "k".to_string(),
                expr: RExpr::Column(colref("k", 0, ProximaType::String)),
            }],
        }
    }

    #[tokio::test]
    async fn lowers_intersect_and_except() {
        // left  = SELECT k FROM t            → {a, a, b}
        // right = SELECT k FROM t WHERE k='a' → {a, a}
        let ctx = ctx_with_t().await;
        let set_op = |op, all| LogicalNode::SetOp {
            op,
            left: Box::new(project_k()),
            right: Box::new(project_k_where("a")),
            all,
        };

        // INTERSECT (distinct): {a,b} ∩ {a} = {a} → 1 row.
        assert_eq!(
            row_count(&ctx, &set_op(SetOpKind::Intersect, false)).await,
            1
        );
        // INTERSECT ALL: multiset min → a:min(2,2)=2, b:min(1,0)=0 → {a,a} → 2 rows.
        assert_eq!(
            row_count(&ctx, &set_op(SetOpKind::Intersect, true)).await,
            2
        );
        // EXCEPT (distinct): {a,b} − {a} = {b} → 1 row.
        assert_eq!(row_count(&ctx, &set_op(SetOpKind::Except, false)).await, 1);
        // EXCEPT ALL: multiset diff → a:max(2−2,0)=0, b:max(1−0,0)=1 → {b} → 1 row.
        assert_eq!(row_count(&ctx, &set_op(SetOpKind::Except, true)).await, 1);
    }

    #[tokio::test]
    async fn lowers_values() {
        // VALUES (1,'a'), (2,'b'), (3,'a')  with output names (n, s).
        let ctx = ctx_with_t().await; // ctx needs no table — VALUES is self-contained.
        let int_lit = |n: i64| RExpr::Literal {
            value: ProximaValue::Int64(n),
            ty: ProximaType::Int64,
        };
        let str_lit = |s: &str| RExpr::Literal {
            value: ProximaValue::String(s.to_string()),
            ty: ProximaType::String,
        };
        let values = || LogicalNode::Values {
            rows: vec![
                vec![int_lit(1), str_lit("a")],
                vec![int_lit(2), str_lit("b")],
                vec![int_lit(3), str_lit("a")],
            ],
            output_schema: RelationalSchema::new(vec![
                ColumnInfo::new("n", ProximaType::Int64, false),
                ColumnInfo::new("s", ProximaType::String, false),
            ]),
        };
        // Bare VALUES → 3 rows.
        assert_eq!(row_count(&ctx, &values()).await, 3);
        // Filtering on the aliased column `s` proves the output_schema names carry
        // through (DataFusion's default column1/column2 are re-aliased to n/s).
        let filtered = LogicalNode::Filter {
            input: Box::new(values()),
            predicate: RExpr::BinaryOp {
                op: RBinOp::Eq,
                left: Box::new(RExpr::Column(colref("s", 1, ProximaType::String))),
                right: Box::new(str_lit("a")),
            },
        };
        assert_eq!(row_count(&ctx, &filtered).await, 2); // rows 1 and 3
    }

    #[tokio::test]
    async fn lowers_cte_inlined_at_each_ref() {
        // WITH cte AS (SELECT k FROM t WHERE k='a')        -- body = {a, a}
        //   (SELECT k FROM cte) UNION ALL (SELECT k FROM cte)
        // Inlining the body at BOTH refs → 2 + 2 = 4 rows.
        let ctx = ctx_with_t().await;
        let cte_schema =
            || RelationalSchema::new(vec![ColumnInfo::new("k", ProximaType::String, false)]);
        let cte_ref = || LogicalNode::CteRef {
            name: "cte".to_string(),
            output_schema: cte_schema(),
        };
        let node = LogicalNode::CteBind {
            name: "cte".to_string(),
            body: Box::new(project_k_where("a")),
            usages: Box::new(LogicalNode::Union {
                inputs: vec![cte_ref(), cte_ref()],
                all: true,
            }),
        };
        assert_eq!(row_count(&ctx, &node).await, 4);

        // An unbound CTE reference must error so the caller keeps the ctx.sql fallback.
        assert!(
            lower_logical_node(&ctx, &cte_ref()).await.is_err(),
            "unbound CteRef must NOT lower"
        );
    }

    #[tokio::test]
    async fn lowers_case_coalesce_nullif() {
        // SELECT
        //   CASE WHEN x > 5 THEN 'big' ELSE 'small' END AS bucket,
        //   COALESCE(NULLIF(k, 'a'), 'was_a')          AS c
        // FROM t  — executes over all 3 rows; success proves the exprs lower.
        let ctx = ctx_with_t().await;
        let bucket = RExpr::Case {
            branches: vec![(
                RExpr::BinaryOp {
                    op: RBinOp::Gt,
                    left: Box::new(RExpr::Column(colref("x", 1, ProximaType::Float64))),
                    right: Box::new(RExpr::Literal {
                        value: ProximaValue::Float64(5.0),
                        ty: ProximaType::Float64,
                    }),
                },
                RExpr::Literal {
                    value: ProximaValue::String("big".to_string()),
                    ty: ProximaType::String,
                },
            )],
            otherwise: Some(Box::new(RExpr::Literal {
                value: ProximaValue::String("small".to_string()),
                ty: ProximaType::String,
            })),
        };
        let c = RExpr::Coalesce(vec![
            RExpr::NullIf {
                left: Box::new(RExpr::Column(colref("k", 0, ProximaType::String))),
                right: Box::new(RExpr::Literal {
                    value: ProximaValue::String("a".to_string()),
                    ty: ProximaType::String,
                }),
            },
            RExpr::Literal {
                value: ProximaValue::String("was_a".to_string()),
                ty: ProximaType::String,
            },
        ]);
        let node = LogicalNode::Project {
            input: Box::new(scan_t()),
            outputs: vec![
                NamedExpr {
                    name: "bucket".to_string(),
                    expr: bucket,
                },
                NamedExpr {
                    name: "c".to_string(),
                    expr: c,
                },
            ],
        };
        assert_eq!(row_count(&ctx, &node).await, 3);
    }

    #[tokio::test]
    async fn lowers_aggregate_sort() {
        let ctx = ctx_with_t().await;
        // SELECT k, sum(x) AS s FROM t GROUP BY k ORDER BY k
        let agg = LogicalNode::Aggregate {
            input: Box::new(scan_t()),
            group_by: vec![NamedExpr {
                name: "k".to_string(),
                expr: RExpr::Column(colref("k", 0, ProximaType::String)),
            }],
            aggregates: vec![NamedAggregate {
                name: "s".to_string(),
                agg: AggregateExpr::Sum {
                    arg: RExpr::Column(colref("x", 1, ProximaType::Float64)),
                    distinct: false,
                },
            }],
            having: None,
        };
        let node = LogicalNode::Sort {
            input: Box::new(agg),
            keys: vec![SortKey {
                expr: RExpr::Column(colref("k", 0, ProximaType::String)),
                descending: false,
                nulls_first: false,
            }],
        };
        let plan = lower_logical_node(&ctx, &node).await.unwrap();
        let batches = ctx
            .execute_logical_plan(plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2); // groups: a, b
    }

    #[tokio::test]
    async fn unsupported_shape_errors_so_caller_falls_back() {
        // A still-unsupported expr (FuncCall) must make lowering error so the P1
        // route keeps the `ctx.sql` fallback. (Joins/Distinct/Union/Case now lower.)
        let ctx = ctx_with_t().await;
        let node = LogicalNode::Project {
            input: Box::new(scan_t()),
            outputs: vec![NamedExpr {
                name: "f".to_string(),
                expr: RExpr::FuncCall {
                    name: "some_udf".to_string(),
                    args: vec![RExpr::Column(colref("k", 0, ProximaType::String))],
                    return_ty: ProximaType::String,
                },
            }],
        };
        assert!(lower_logical_node(&ctx, &node).await.is_err());
    }
}
