//! TD-PGEXT-1: parse-only helpers for `Describe(statement)`.
//!
//! The extended query protocol makes Describe the authority for a statement's
//! result columns: Execute suppresses its own RowDescription once Describe has
//! spoken, so Describe must report exactly the columns Execute will stream —
//! a wrong non-empty description is worse than an empty one.
//!
//! `PostgresProtocolHandler::describe_select_shape` owns the routing (it reuses
//! the dispatcher's own predicates and the DML service's column resolution for
//! the legacy single-table reader). This module covers the one shape it can
//! describe from the SQL text alone: a relational-pipeline SELECT whose
//! projection is plain columns and `count()`. It mirrors the relational
//! frontend's output-naming rules (`proximadb-relational-frontend`):
//!
//! * an alias wins;
//! * a column reference is named as written (`ident.value`, no case folding —
//!   the frontend does not fold either);
//! * an aggregate is named by its function name, lowercased (`count`).
//!
//! Anything else (literals, expressions, other aggregates, `*`, joins) is
//! `None`: the caller keeps the previous empty description rather than guess.

use sqlparser::ast::{
    Expr as SqlExpr, FunctionArguments, ObjectNamePart, SelectItem, SetExpr, Statement, TableFactor,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// One output column of a describable relational-pipeline SELECT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PipelineColumn {
    /// A column of the FROM table: `name` is the output name the frontend
    /// emits, `source` the referenced column (its type comes from the table).
    Column { name: String, source: String },
    /// `count(*)` / `count(col)` / `count(DISTINCT col)`: always `int8`.
    Count { name: String },
}

/// A describable relational-pipeline SELECT: its single FROM table and its
/// output columns, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipelineProjection {
    pub(crate) table: String,
    pub(crate) columns: Vec<PipelineColumn>,
}

/// Resolve the output shape of a one-table SELECT whose projection is plain
/// columns and `count()`. Parses with the same dialect the relational pipeline
/// uses. `None` for every other shape.
pub(crate) fn pipeline_projection(sql: &str) -> Option<PipelineProjection> {
    let statements = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return None;
    };
    if query.with.is_some() {
        return None;
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    let [from] = select.from.as_slice() else {
        return None;
    };
    if !from.joins.is_empty() {
        return None;
    }
    // A table-valued function (`FROM vector_search(...)`) has its own shape.
    let TableFactor::Table {
        name, args: None, ..
    } = &from.relation
    else {
        return None;
    };
    let table = name
        .0
        .iter()
        .map(|part| match part {
            ObjectNamePart::Identifier(ident) => Some(ident.value.clone()),
            ObjectNamePart::Function(_) => None,
        })
        .collect::<Option<Vec<_>>>()?
        .join(".");

    let mut columns = Vec::with_capacity(select.projection.len());
    for item in &select.projection {
        let (expr, alias) = match item {
            SelectItem::UnnamedExpr(expr) => (expr, None),
            SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.clone())),
            _ => return None,
        };
        columns.push(pipeline_column(expr, alias)?);
    }
    Some(PipelineProjection { table, columns })
}

fn pipeline_column(expr: &SqlExpr, alias: Option<String>) -> Option<PipelineColumn> {
    match expr {
        SqlExpr::Identifier(ident) => Some(PipelineColumn::Column {
            name: alias.unwrap_or_else(|| ident.value.clone()),
            source: ident.value.clone(),
        }),
        SqlExpr::CompoundIdentifier(parts) => {
            let source = parts.last()?.value.clone();
            Some(PipelineColumn::Column {
                name: alias.unwrap_or_else(|| source.clone()),
                source,
            })
        }
        SqlExpr::Function(function) => {
            let [ObjectNamePart::Identifier(fname)] = function.name.0.as_slice() else {
                return None;
            };
            if !fname.value.eq_ignore_ascii_case("count")
                || function.over.is_some()
                || function.filter.is_some()
                || !matches!(function.args, FunctionArguments::List(_))
            {
                return None;
            }
            Some(PipelineColumn::Count {
                name: alias.unwrap_or_else(|| function.name.to_string().to_lowercase()),
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(name: &str, source: &str) -> PipelineColumn {
        PipelineColumn::Column {
            name: name.to_string(),
            source: source.to_string(),
        }
    }

    fn count(name: &str) -> PipelineColumn {
        PipelineColumn::Count {
            name: name.to_string(),
        }
    }

    #[test]
    fn describes_columns_and_count_with_frontend_names() {
        let cases: Vec<(&str, Vec<PipelineColumn>)> = vec![
            ("SELECT count(*) FROM t", vec![count("count")]),
            ("SELECT COUNT(*) FROM t", vec![count("count")]),
            ("SELECT count(id) AS n FROM t", vec![count("n")]),
            ("SELECT count(DISTINCT id) FROM t", vec![count("count")]),
            (
                "SELECT name, count(*) FROM t GROUP BY name",
                vec![col("name", "name"), count("count")],
            ),
            (
                "SELECT t.name AS label, count(*) FROM t GROUP BY t.name",
                vec![col("label", "name"), count("count")],
            ),
            // The frontend names a column as written (no case folding).
            (
                "SELECT NAME, count(*) FROM t GROUP BY NAME",
                vec![col("NAME", "NAME"), count("count")],
            ),
        ];
        for (sql, expected) in cases {
            let described = pipeline_projection(sql).unwrap_or_else(|| panic!("{sql}"));
            assert_eq!(described.table, "t", "{sql}");
            assert_eq!(described.columns, expected, "{sql}");
        }
    }

    #[test]
    fn leaves_shapes_it_cannot_know_undescribed() {
        for sql in [
            "SELECT * FROM t",
            "SELECT sum(id) FROM t",
            "SELECT max(id), count(*) FROM t",
            "SELECT 1 FROM t",
            "SELECT id + 1 FROM t",
            "SELECT count(*) OVER () FROM t",
            "SELECT a.id FROM t a JOIN u b ON a.id = b.id",
            "SELECT id FROM t UNION SELECT id FROM u",
            "WITH c AS (SELECT id FROM t) SELECT id FROM c",
            "SELECT id FROM (SELECT id FROM t) s",
            "SELECT count(*) FROM vector_search('t', '[1,2]', 5)",
            "INSERT INTO t VALUES (1)",
            "not sql",
        ] {
            assert_eq!(pipeline_projection(sql), None, "{sql}");
        }
    }
}
