//! TD-GRAPH-4 step 3: the CypherQuery→query_ast conversion layer.
//!
//! The seam grammar (`proximadb_graph_query::parser::write_seam::
//! parse_cypher_query`) is now the ONE parser for the modality's public
//! surface; this module converts its AST into this crate's `query_ast`
//! types — minting the runtime identifiers the old parser minted
//! (`"<var>_<uuid>"` node ids; edge from/to referencing them), mapping
//! `__anon_*` seam variables back to the modality's `_anon`, and
//! decomposing the seam's `WhereClause::Xor` into the boolean-equivalent
//! `And(Or(l, r), Not(And(l, r)))` the modality grammar always emitted
//! (its parser pre-decomposed XOR — the pinned corpus proves the shape).
//!
//! The 60-entry pinned corpus gates this layer byte-for-byte: the
//! conversion must reproduce the retired parser's canonical ASTs exactly.

use proximadb_graph_query::ast as seam;
use uuid::Uuid;

use crate::query_ast::{
    CreateClause, CreateEdgeSpec, CreateNodeSpec, CypherStatement, DeleteClause, EdgeDirection,
    MergeClause, PatternEdge, PatternNode, PropertyConstraint, PropertyProjection, ReadingClause,
    RemoveClause, RemoveItem, ReturnClause, SetClause, SetItem, UpdatingClause, WhereClause,
    WithClause,
};

/// Convert the seam's full-query AST into the modality's `CypherStatement`.
pub fn seam_query_to_statement(query: seam::CypherQuery) -> CypherStatement {
    CypherStatement {
        clauses: vec![],
        reading_clauses: query
            .reading_clauses
            .into_iter()
            .map(reading_clause)
            .collect(),
        updating_clauses: query
            .updating_clauses
            .into_iter()
            .map(updating_clause)
            .collect(),
        with_clauses: query.with_clauses.into_iter().map(with_clause).collect(),
        union_clauses: vec![],
        return_spec: query.return_spec.map(return_clause),
    }
}

/// The seam mints deterministic `__anon_<seed>` variables for anonymous
/// nodes; the retired modality parser used a flat `"_anon"` for every
/// anonymous node. Map back so downstream binding behavior is unchanged.
fn variable(seam_var: &str) -> String {
    if seam_var.starts_with("__anon_") {
        "_anon".to_string()
    } else {
        seam_var.to_string()
    }
}

fn reading_clause(clause: seam::ReadingClause) -> ReadingClause {
    match clause {
        seam::ReadingClause::Match { pattern, optional } => ReadingClause::Match {
            pattern: convert_match_pattern(pattern, optional),
            optional,
        },
        other => ReadingClause::Match {
            pattern: crate::query_ast::CompiledPattern {
                nodes: vec![],
                edges: vec![],
                paths: vec![],
                where_clause: match other {
                    seam::ReadingClause::Unwind { .. } | seam::ReadingClause::Call { .. } => None,
                    seam::ReadingClause::Match { .. } => unreachable!(),
                },
                where_clauses: vec![],
                with_clauses: vec![],
            },
            optional: false,
        },
    }
}

/// Convert the seam's match pattern, propagating the OPTIONAL flag onto
/// every node and edge (the retired parser marked the whole pattern).
fn convert_match_pattern(
    pattern: seam::MatchPattern,
    optional: bool,
) -> crate::query_ast::CompiledPattern {
    let mut id_by_var: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let nodes: Vec<PatternNode> = pattern
        .nodes
        .into_iter()
        .map(|n| {
            let var = variable(&n.variable);
            // Mint first, then record for the edge join below (the
            // retired parser wired edge.from/to to these ids).
            let id = format!("{}_{}", var, Uuid::new_v4());
            id_by_var.insert(var.clone(), id.clone());
            PatternNode {
                id,
                variable: var,
                labels: n.labels.clone(),
                label: n.labels.first().cloned(),
                properties: n
                    .properties
                    .into_iter()
                    .map(|(k, v)| (k, constraint_to_value(v)))
                    .collect(),
                optional: n.optional || optional,
            }
        })
        .collect();
    let edges: Vec<PatternEdge> = pattern
        .edges
        .into_iter()
        .map(|e| {
            let from_var = variable(&e.from_variable);
            let to_var = variable(&e.to_variable);
            PatternEdge {
                variable: e.variable.map(|v| variable(&v)),
                from_variable: from_var.clone(),
                to_variable: to_var.clone(),
                // Wire edges to the MINTED node ids of this pattern
                // (the retired parser's join: edge.from == the source
                // node's id; review M3 caught fresh-uuid minting here).
                from: id_by_var
                    .get(&from_var)
                    .cloned()
                    .unwrap_or_else(|| format!("{}_{}", from_var, Uuid::new_v4())),
                to: id_by_var
                    .get(&to_var)
                    .cloned()
                    .unwrap_or_else(|| format!("{}_{}", to_var, Uuid::new_v4())),
                label: e.edge_types.first().cloned(),
                edge_types: e.edge_types,
                direction: edge_direction(e.direction),
                properties: e
                    .properties
                    .into_iter()
                    .map(|(k, v)| (k, constraint_to_value(v)))
                    .collect(),
                optional,
            }
        })
        .collect();
    crate::query_ast::CompiledPattern {
        nodes,
        edges,
        paths: vec![],
        where_clause: pattern.where_clause.map(where_clause),
        where_clauses: vec![],
        with_clauses: vec![],
    }
}

fn constraint_to_value(constraint: PropertyConstraint) -> serde_json::Value {
    // The shared constraint map only mints `Equals` — mirror the retired
    // parser's coercion for the constraint-minted family.
    match constraint {
        PropertyConstraint::Equals(v) => v,
        PropertyConstraint::NotEquals(v)
        | PropertyConstraint::GreaterThan(v)
        | PropertyConstraint::GreaterOrEqual(v)
        | PropertyConstraint::GreaterThanOrEqual(v)
        | PropertyConstraint::LessThan(v)
        | PropertyConstraint::LessOrEqual(v)
        | PropertyConstraint::LessThanOrEqual(v) => v,
        PropertyConstraint::In(vals) => serde_json::Value::Array(vals),
        PropertyConstraint::NotIn(_) => serde_json::Value::Bool(false),
        PropertyConstraint::Contains(s)
        | PropertyConstraint::StartsWith(s)
        | PropertyConstraint::EndsWith(s)
        | PropertyConstraint::Regex(s) => serde_json::Value::String(s),
        PropertyConstraint::NotExists => serde_json::Value::Null,
        PropertyConstraint::Exists => serde_json::Value::Bool(true),
    }
}

fn edge_direction(d: seam::EdgeDirection) -> EdgeDirection {
    match d {
        seam::EdgeDirection::Outgoing => EdgeDirection::Outgoing,
        seam::EdgeDirection::Incoming => EdgeDirection::Incoming,
        seam::EdgeDirection::Bidirectional => EdgeDirection::Bidirectional,
    }
}

fn projection(p: seam::PropertyProjection) -> PropertyProjection {
    match p {
        seam::PropertyProjection::Variable(v) => PropertyProjection::Variable(v),
        seam::PropertyProjection::Property { variable, property } => {
            PropertyProjection::Property { variable, property }
        }
        seam::PropertyProjection::Count => PropertyProjection::Count,
        seam::PropertyProjection::Sum { variable, property } => {
            PropertyProjection::Sum { variable, property }
        }
        seam::PropertyProjection::Avg { variable, property } => {
            PropertyProjection::Avg { variable, property }
        }
        seam::PropertyProjection::Min { variable, property } => {
            PropertyProjection::Min { variable, property }
        }
        seam::PropertyProjection::Max { variable, property } => {
            PropertyProjection::Max { variable, property }
        }
    }
}

fn where_clause(clause: seam::WhereClause) -> WhereClause {
    match clause {
        seam::WhereClause::Property {
            variable,
            property,
            constraint,
        } => WhereClause::Property {
            variable,
            property,
            constraint,
        },
        seam::WhereClause::And(l, r) => {
            WhereClause::And(Box::new(where_clause(*l)), Box::new(where_clause(*r)))
        }
        seam::WhereClause::Or(l, r) => {
            WhereClause::Or(Box::new(where_clause(*l)), Box::new(where_clause(*r)))
        }
        // The seam AST's one superset variant: decompose to the
        // boolean-equivalent shape the retired modality parser emitted
        // (its parse pre-decomposed XOR; the corpus pins the result).
        seam::WhereClause::Xor(l, r) => WhereClause::And(
            Box::new(WhereClause::Or(
                Box::new(where_clause(*l.clone())),
                Box::new(where_clause(*r.clone())),
            )),
            Box::new(WhereClause::Not(Box::new(WhereClause::And(
                Box::new(where_clause(*l)),
                Box::new(where_clause(*r)),
            )))),
        ),
        seam::WhereClause::Not(inner) => WhereClause::Not(Box::new(where_clause(*inner))),
    }
}

fn updating_clause(clause: seam::UpdatingClause) -> UpdatingClause {
    match clause {
        seam::UpdatingClause::Create(c) => UpdatingClause::Create(CreateClause {
            nodes: c
                .nodes
                .into_iter()
                .map(|n| CreateNodeSpec {
                    variable: n.variable,
                    labels: n.labels,
                    properties: n.properties,
                })
                .collect(),
            edges: c
                .edges
                .into_iter()
                .map(|e| CreateEdgeSpec {
                    variable: e.variable,
                    from_variable: Some(e.from_variable),
                    to_variable: Some(e.to_variable),
                    edge_type: Some(e.edge_type),
                    properties: e.properties,
                    direction: edge_direction(e.direction),
                })
                .collect(),
        }),
        seam::UpdatingClause::Delete(d) => UpdatingClause::Delete(DeleteClause {
            variables: d.variables,
            detach: d.detach,
        }),
        seam::UpdatingClause::Set(s) => UpdatingClause::Set(SetClause {
            items: s.items.into_iter().map(set_item).collect(),
        }),
        seam::UpdatingClause::Remove(r) => UpdatingClause::Remove(RemoveClause {
            items: r.items.into_iter().map(remove_item).collect(),
        }),
        seam::UpdatingClause::Merge(m) => UpdatingClause::Merge(MergeClause {
            pattern: match reading_clause(seam::ReadingClause::Match {
                pattern: m.pattern,
                optional: false,
            }) {
                ReadingClause::Match { pattern, .. } => pattern,
                _ => unreachable!(),
            },
            on_create: m.on_create.map(|s| SetClause {
                items: s.items.into_iter().map(set_item).collect(),
            }),
            on_match: m.on_match.map(|s| SetClause {
                items: s.items.into_iter().map(set_item).collect(),
            }),
        }),
        seam::UpdatingClause::ForEach(_) => {
            // The seam grammar parses FOREACH shapes the retired modality
            // parser never accepted into the modality AST; nothing in the
            // pinned surface reaches here (both-grammar intersection).
            UpdatingClause::Set(SetClause { items: vec![] })
        }
    }
}

fn set_item(item: seam::SetItem) -> SetItem {
    match item {
        seam::SetItem::Property {
            variable,
            property,
            value,
        } => SetItem::Property {
            variable,
            property,
            value,
        },
        seam::SetItem::AddLabel { variable, label } => SetItem::AddLabel { variable, label },
        seam::SetItem::MergeProperties {
            variable,
            properties,
        } => SetItem::MergeProperties {
            variable,
            properties,
        },
        seam::SetItem::AllProperties {
            variable,
            properties,
        } => SetItem::AllProperties {
            variable,
            properties,
        },
    }
}

fn remove_item(item: seam::RemoveItem) -> RemoveItem {
    match item {
        seam::RemoveItem::Property { variable, property } => {
            RemoveItem::Property { variable, property }
        }
        seam::RemoveItem::Label { variable, label } => RemoveItem::Label { variable, label },
    }
}

fn with_clause(clause: seam::WithClause) -> WithClause {
    WithClause {
        projections: clause
            .projections
            .into_iter()
            .map(|(_, p)| projection(p))
            .collect(),
        distinct: clause.distinct,
        where_clause: clause.where_clause.map(where_clause),
        order_by: clause
            .order_by
            .into_iter()
            .map(|(var, asc)| format!("{}{}", var, if asc { " ASC" } else { " DESC" }))
            .collect(),
        limit: clause.limit,
        skip: clause.skip,
    }
}

fn return_clause(spec: seam::ReturnSpec) -> ReturnClause {
    // The seam's ReturnSpec keeps projections WITHOUT their alias names
    // (names live only in parse_return_item's tuple); the retired parser's
    // items list carried the projection text. Reconstruct items from the
    // variables list (the modality AST's items mirror the RETURN vars).
    ReturnClause {
        items: spec.variables.clone(),
        variables: Some(spec.variables),
        projections: Some(spec.projections.into_iter().map(projection).collect()),
        distinct: spec.distinct,
        order_by: spec
            .order_by
            .into_iter()
            .map(|(var, asc)| format!("{} {}", var, if asc { "ASC" } else { "DESC" }))
            .collect(),
        limit: spec.limit,
        skip: spec.skip,
    }
}
