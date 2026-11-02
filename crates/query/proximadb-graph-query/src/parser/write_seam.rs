//! Cypher write-clause seam — the INTENTIONAL future surface for full-query
//! parsing in this crate (owner decision 2026-11-01: retain and build upon).
//!
//! Today the live entry of this crate is [`super::QueryParser`] (the read
//! path: MATCH/WHERE/RETURN → `CompiledPattern`, used by the SQL-lowering
//! pipeline). The full-Cypher grammar below — CREATE/MERGE/DELETE/SET/REMOVE/
//! WITH/OPTIONAL MATCH over `CypherQuery` — is **not yet wired into any
//! runtime path**; its contract is pinned by the ported tests at the bottom
//! of [`super`]. It exists as the convergence target for the *other* live
//! Cypher parser (`proximadb-graph`'s `cypher_parser`, which serves the graph
//! API path and drift-diverged from this grammar — see the dedup program's
//! wave-3 notes): new full-query parsing work belongs HERE, behind this seam,
//! so the two grammars converge on one implementation instead of forking
//! again. Do not delete; wire up or extend deliberately.

use super::*;

pub(super) fn parse_property_value_map(
    input: &str,
) -> IResult<&str, HashMap<String, serde_json::Value>> {
    // Shared lexeme (TD-GRAPH-4 step 1): the superset behavior — quoted
    // keys, permissive spacing — with existing inputs parsing identically.
    lexemes::property_value_map(input)
}

// Parse a CREATE node specification (n:Label {props})
pub(super) fn parse_create_node_spec(input: &str) -> IResult<&str, CreateNodeSpec> {
    map(
        delimited(
            char('('),
            tuple((
                preceded(multispace0, opt(any_identifier)), // Optional variable
                // Labels: :A or :A:B (multi-label — the modality grammar's
                // shape; previously single-only, which left the seam
                // internally inconsistent: MERGE multi-label parsed via the
                // shared node pattern while CREATE did not)
                many0(preceded(char(':'), any_identifier)),
                opt(preceded(multispace0, parse_property_value_map)), // Optional properties
                multispace0,
            )),
            char(')'),
        ),
        |(variable, labels, properties_opt, _)| CreateNodeSpec {
            variable,
            labels,
            properties: properties_opt.unwrap_or_default(),
        },
    )(input)
}

// Parse a CREATE edge specification (a)-[r:TYPE {props}]->(b)
pub(super) fn parse_create_edge_pattern(
    input: &str,
) -> IResult<&str, (CreateNodeSpec, CreateEdgeSpec, CreateNodeSpec)> {
    map(
        tuple((
            parse_create_node_spec,
            preceded(multispace0, parse_create_edge_spec),
            preceded(multispace0, parse_create_node_spec),
        )),
        |(from_node, (edge_var, edge_type, edge_props, _direction), to_node)| {
            let from_var = from_node
                .variable
                .clone()
                .unwrap_or_else(|| "_from".to_string());
            let to_var = to_node
                .variable
                .clone()
                .unwrap_or_else(|| "_to".to_string());

            let edge = CreateEdgeSpec {
                variable: edge_var,
                from_variable: from_var,
                to_variable: to_var,
                edge_type: edge_type.unwrap_or_default(),
                properties: edge_props,
            };

            (from_node, edge, to_node)
        },
    )(input)
}

// Parse edge spec for CREATE: -[r:TYPE {props}]->
pub(super) fn parse_create_edge_spec(input: &str) -> IResult<&str, CreateEdgeSpecParts> {
    map(
        tuple((
            parse_edge_direction_left,
            delimited(
                char('['),
                tuple((
                    opt(any_identifier), // Variable
                    // Edge type — SINGLE type (the modality grammar's
                    // create-edge shape; review R3-F3: alternation here
                    // was a seam superset silently dropping the extra
                    // types — the read path keeps the full list, CREATE
                    // matches the modality and rejects alternation).
                    opt(preceded(char(':'), any_identifier)),
                    opt(preceded(multispace0, parse_property_value_map)), // Properties
                )),
                char(']'),
            ),
            parse_edge_direction_right,
        )),
        |(left_arrow, (variable, edge_type, properties_opt), right_arrow)| {
            let direction = match (left_arrow, right_arrow) {
                (false, true) => EdgeDirection::Outgoing,
                (true, false) => EdgeDirection::Incoming,
                _ => EdgeDirection::Bidirectional,
            };
            (
                variable,
                edge_type,
                properties_opt.unwrap_or_default(),
                direction,
            )
        },
    )(input)
}

// Parse CREATE clause
pub(super) fn parse_create_clause(input: &str) -> IResult<&str, CreateClause> {
    preceded(
        tag_no_case("CREATE"),
        preceded(
            multispace1,
            map(
                separated_list1(
                    delimited(multispace0, char(','), multispace0),
                    alt((
                        // Edge pattern: (a)-[r:TYPE]->(b)
                        map(parse_create_edge_pattern, |(from, edge, to)| {
                            (vec![from, to], vec![edge])
                        }),
                        // Simple node: (n:Label {props})
                        map(parse_create_node_spec, |node| (vec![node], vec![])),
                    )),
                ),
                |patterns| {
                    let mut nodes = Vec::new();
                    let mut edges = Vec::new();
                    for (mut ns, mut es) in patterns {
                        nodes.append(&mut ns);
                        edges.append(&mut es);
                    }
                    CreateClause { nodes, edges }
                },
            ),
        ),
    )(input)
}

// Parse DELETE clause
pub(super) fn parse_delete_clause(input: &str) -> IResult<&str, DeleteClause> {
    map(
        tuple((
            opt(preceded(tag_no_case("DETACH"), multispace1)),
            preceded(
                tag_no_case("DELETE"),
                preceded(
                    multispace1,
                    separated_list1(
                        delimited(multispace0, char(','), multispace0),
                        any_identifier,
                    ),
                ),
            ),
        )),
        |(detach_opt, variables)| DeleteClause {
            variables,
            detach: detach_opt.is_some(),
        },
    )(input)
}

// Parse SET item: n.prop = value OR n:Label OR n = {props} OR n += {props}
pub(super) fn parse_set_item(input: &str) -> IResult<&str, SetItem> {
    alt((
        // Property assignment: n.prop = value
        map(
            tuple((
                any_identifier,
                char('.'),
                any_identifier,
                delimited(multispace0, char('='), multispace0),
                property_value,
            )),
            |(variable, _, property, _, value)| SetItem::Property {
                variable,
                property,
                value,
            },
        ),
        // Add label: n:Label
        map(
            tuple((any_identifier, char(':'), any_identifier)),
            |(variable, _, label)| SetItem::AddLabel { variable, label },
        ),
        // Merge properties: n += {props}
        map(
            tuple((
                any_identifier,
                delimited(multispace0, tag("+="), multispace0),
                parse_property_value_map,
            )),
            |(variable, _, properties)| SetItem::MergeProperties {
                variable,
                properties,
            },
        ),
        // Replace all properties: n = {props}
        map(
            tuple((
                any_identifier,
                delimited(multispace0, char('='), multispace0),
                parse_property_value_map,
            )),
            |(variable, _, properties)| SetItem::AllProperties {
                variable,
                properties,
            },
        ),
    ))(input)
}

// Parse SET clause
pub(super) fn parse_set_clause(input: &str) -> IResult<&str, SetClause> {
    preceded(
        tag_no_case("SET"),
        preceded(
            multispace1,
            map(
                separated_list1(
                    delimited(multispace0, char(','), multispace0),
                    parse_set_item,
                ),
                |items| SetClause { items },
            ),
        ),
    )(input)
}

// Parse REMOVE item: n.prop OR n:Label
pub(super) fn parse_remove_item(input: &str) -> IResult<&str, RemoveItem> {
    alt((
        // Remove property: n.prop
        map(
            tuple((any_identifier, char('.'), any_identifier)),
            |(variable, _, property)| RemoveItem::Property { variable, property },
        ),
        // Remove label: n:Label
        map(
            tuple((any_identifier, char(':'), any_identifier)),
            |(variable, _, label)| RemoveItem::Label { variable, label },
        ),
    ))(input)
}

// Parse REMOVE clause
pub(super) fn parse_remove_clause(input: &str) -> IResult<&str, RemoveClause> {
    preceded(
        tag_no_case("REMOVE"),
        preceded(
            multispace1,
            map(
                separated_list1(
                    delimited(multispace0, char(','), multispace0),
                    parse_remove_item,
                ),
                |items| RemoveClause { items },
            ),
        ),
    )(input)
}

// Parse MERGE clause
pub(super) fn parse_merge_clause(input: &str) -> IResult<&str, MergeClause> {
    map(
        tuple((
            preceded(
                tag_no_case("MERGE"),
                preceded(
                    multispace1,
                    map(
                        separated_list1(
                            delimited(multispace0, char(','), multispace0),
                            alt((
                                map(parse_path_segment, |(from, edge, to)| {
                                    (vec![from, to], vec![edge])
                                }),
                                map(parse_node_pattern, |node| (vec![node], vec![])),
                            )),
                        ),
                        |patterns| {
                            let mut nodes = Vec::new();
                            let mut edges = Vec::new();
                            for (mut ns, mut es) in patterns {
                                nodes.append(&mut ns);
                                edges.append(&mut es);
                            }
                            MatchPattern {
                                nodes,
                                edges,
                                paths: Vec::new(),
                                where_clause: None,
                            }
                        },
                    ),
                ),
            ),
            // ON CREATE / ON MATCH — case-insensitive and spacing-
            // permissive (the modality grammar's shape; review R3-F1: the
            // frozen single-space case-sensitive tags never landed the
            // claimed parity).
            opt(preceded(
                tuple((
                    multispace1,
                    tag_no_case("ON"),
                    multispace1,
                    tag_no_case("CREATE"),
                    multispace1,
                )),
                parse_set_clause,
            )),
            opt(preceded(
                tuple((
                    multispace1,
                    tag_no_case("ON"),
                    multispace1,
                    tag_no_case("MATCH"),
                    multispace1,
                )),
                parse_set_clause,
            )),
        )),
        |(pattern, on_create, on_match)| MergeClause {
            pattern,
            on_create,
            on_match,
        },
    )(input)
}

// Parse OPTIONAL MATCH clause
pub(super) fn parse_optional_match_clause(
    input: &str,
) -> IResult<&str, (Vec<NodePattern>, Vec<EdgePattern>)> {
    map(
        preceded(
            tag_no_case("OPTIONAL"),
            preceded(multispace1, parse_match_clause),
        ),
        |(nodes, edges)| {
            // Mark all nodes and edges as optional
            let optional_nodes = nodes
                .into_iter()
                .map(|mut n| {
                    n.optional = true;
                    n
                })
                .collect();
            let optional_edges = edges
                .into_iter()
                .map(|mut e| {
                    e.optional = true;
                    e
                })
                .collect();
            (optional_nodes, optional_edges)
        },
    )(input)
}

// Parse WITH clause
pub(super) fn parse_with_clause(input: &str) -> IResult<&str, WithClause> {
    preceded(
        tag_no_case("WITH"),
        map(
            tuple((
                parse_distinct,
                preceded(
                    multispace1,
                    separated_list1(
                        delimited(multispace0, char(','), multispace0),
                        parse_return_item,
                    ),
                ),
                opt(parse_order_by),
                opt(parse_skip),
                opt(parse_limit),
                opt(preceded(multispace1, parse_where_clause)),
            )),
            |(distinct, items, order_by_opt, skip_opt, limit_opt, where_opt)| WithClause {
                projections: items,
                distinct,
                order_by: order_by_opt.unwrap_or_default(),
                limit: limit_opt,
                skip: skip_opt,
                where_clause: where_opt,
            },
        ),
    )(input)
}

/// Parse a full Cypher query with all clause types.
///
/// `pub` as the crate's declared convergence surface (TD-GRAPH-4): the
/// modality crate's corpus pin (and, per the TD's convergence design, the
/// clause ports that follow) drive the grammar through this entry — never
/// through the private combinators below, which stay `pub(super)`.
///
/// NOTE: this is a nom parser and does NOT require eof — `Ok((rest, _))`
/// with a non-empty `rest` means the input was NOT fully parsed and must
/// be treated as a rejection. Callers MUST check the remainder (the corpus
/// pin's `rest.trim().is_empty()` assertion is the standard).
pub fn parse_cypher_query(input: &str) -> IResult<&str, CypherQuery> {
    map(
        tuple((
            // Reading clauses (MATCH, OPTIONAL MATCH)
            many0(preceded(
                multispace0,
                alt((
                    // OPTIONAL MATCH
                    map(parse_optional_match_clause, |(nodes, edges)| {
                        ReadingClause::Match {
                            pattern: MatchPattern {
                                nodes,
                                edges,
                                paths: Vec::new(),
                                where_clause: None,
                            },
                            optional: true,
                        }
                    }),
                    // Regular MATCH with optional WHERE
                    map(
                        tuple((
                            parse_match_clause,
                            opt(preceded(multispace1, parse_where_clause)),
                        )),
                        |((nodes, edges), where_opt)| ReadingClause::Match {
                            pattern: MatchPattern {
                                nodes,
                                edges,
                                paths: Vec::new(),
                                where_clause: where_opt,
                            },
                            optional: false,
                        },
                    ),
                )),
            )),
            // WITH clauses
            many0(preceded(multispace0, parse_with_clause)),
            // Updating clauses (CREATE, DELETE, SET, REMOVE, MERGE)
            many0(preceded(
                multispace0,
                alt((
                    map(parse_create_clause, UpdatingClause::Create),
                    map(parse_delete_clause, UpdatingClause::Delete),
                    map(parse_set_clause, UpdatingClause::Set),
                    map(parse_remove_clause, UpdatingClause::Remove),
                    map(parse_merge_clause, UpdatingClause::Merge),
                )),
            )),
            // Optional RETURN clause
            opt(preceded(multispace0, parse_return_clause)),
        )),
        |(reading_clauses, with_clauses, updating_clauses, return_spec)| CypherQuery {
            reading_clauses,
            updating_clauses,
            with_clauses,
            union_clauses: Vec::new(), // TD-019: UNION clause support
            return_spec,
        },
    )(input)
}
