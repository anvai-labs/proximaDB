use anyhow::{Result, anyhow};
use proximadb_data_model::DataModel;
use proximadb_graph_query::ast::{
    CompiledPattern, PropertyConstraint, PropertyProjection, WhereClause,
};
pub use proximadb_graph_query::declarative::{LoweredGraphQuery, SupportedGraphQueryDescriptor};
use proximadb_graph_query::parser::QueryParser;
use proximadb_multimodel_query::{ModelOperation, QueryComponent};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct ParsedGraphQuery {
    graph_id: String,
    normalized_query: String,
    compiled: CompiledPattern,
    output_columns: Vec<String>,
}

impl ParsedGraphQuery {
    pub fn graph_id(&self) -> &str {
        &self.graph_id
    }

    pub fn normalized_query(&self) -> &str {
        &self.normalized_query
    }

    pub fn compiled(&self) -> &CompiledPattern {
        &self.compiled
    }

    pub fn output_columns(&self) -> &[String] {
        &self.output_columns
    }

    pub fn uses_legacy_node_rows(&self) -> bool {
        if self.compiled.return_spec.projections.len() != 1 {
            return false;
        }

        match &self.compiled.return_spec.projections[0] {
            PropertyProjection::Variable(variable) => self
                .compiled
                .nodes
                .iter()
                .any(|node| node.variable == *variable),
            _ => false,
        }
    }
}

pub fn lower_supported_graph_query(
    query: &str,
    request_target: Option<&str>,
    default_graph: Option<&str>,
) -> Result<LoweredGraphQuery> {
    describe_supported_graph_query(query, request_target, default_graph)
}

/// Lower a supported declarative graph query into the unified query IR.
pub fn lower_supported_graph_query_expr(
    query: &str,
    request_target: Option<&str>,
    default_graph: Option<&str>,
) -> Result<LoweredGraphQuery> {
    lower_supported_graph_query(query, request_target, default_graph)
}

/// Lower a supported declarative graph query into a graph query component.
pub fn lower_supported_graph_query_component(
    query: &str,
    request_target: Option<&str>,
    default_graph: Option<&str>,
) -> Result<QueryComponent> {
    Ok(QueryComponent {
        model: DataModel::Graph,
        operation: ModelOperation::GraphQuery(lower_supported_graph_query_expr(
            query,
            request_target,
            default_graph,
        )?),
        filters: Vec::new(),
        dependencies: Vec::new(),
    })
}

pub fn describe_supported_graph_query(
    query: &str,
    request_target: Option<&str>,
    default_graph: Option<&str>,
) -> Result<SupportedGraphQueryDescriptor> {
    let parsed = parse_supported_graph_query(query, request_target, default_graph)?;
    let uses_legacy_node_rows = parsed.uses_legacy_node_rows();
    let output_columns = if uses_legacy_node_rows {
        vec![
            "node_id".to_string(),
            "label".to_string(),
            "properties".to_string(),
        ]
    } else {
        parsed.output_columns().to_vec()
    };

    Ok(SupportedGraphQueryDescriptor {
        graph_name: parsed.graph_id().to_string(),
        normalized_query: parsed.normalized_query().to_string(),
        output_columns,
        uses_legacy_node_rows,
        max_depth: parsed.compiled().edges.len() as u32,
    })
}

pub fn parse_supported_graph_query(
    query: &str,
    request_target: Option<&str>,
    default_graph: Option<&str>,
) -> Result<ParsedGraphQuery> {
    let (normalized_query, from_graph) = strip_from_clause(query)?;

    let request_target = request_target.filter(|target| !target.trim().is_empty());
    if let (Some(target), Some(from_graph)) = (request_target, from_graph.as_deref())
        && target != from_graph
    {
        return Err(anyhow!(
            "Graph query target conflict: request targets graph '{}' but query specifies FROM {}",
            target,
            from_graph
        ));
    }

    let graph_id = request_target
        .or(from_graph.as_deref())
        .or(default_graph)
        .unwrap_or("default")
        .to_string();

    let parser = QueryParser::new();
    let compiled = parser.parse(&normalized_query).map_err(|error| {
        anyhow!(
            "Unsupported graph query syntax for the facade/federated subset: {}",
            error
        )
    })?;

    validate_compiled_query(&compiled, &normalized_query)?;

    Ok(ParsedGraphQuery {
        graph_id,
        normalized_query,
        output_columns: compiled.return_spec.variables.clone(),
        compiled,
    })
}

fn strip_from_clause(query: &str) -> Result<(String, Option<String>)> {
    let trimmed = query.trim().trim_end_matches(';').trim();

    // WHITESPACE-CLASS, QUOTE-AWARE keyword scan — the injector preserves
    // the caller's newline/tab padding before FROM, and a FROM inside a
    // Cypher string literal (its own tests use exactly that shape) must
    // not hijack the split.
    let bytes = trimmed.as_bytes();
    let mut quote: Option<u8> = None;
    let mut i = 0usize;
    let mut from_pos: Option<usize> = None;
    while i < bytes.len() {
        match quote {
            Some(q) => {
                // Backslash-escaped quote (the injector's scanner handles
                // it; without it the literal closed early and FROM was
                // never found or mis-split).
                if bytes[i] == b'\\' && q != b'`' && i + 1 < bytes.len() {
                    i += 2;
                    continue;
                }
                if bytes[i] == q {
                    if i + 1 < bytes.len() && bytes[i + 1] == q {
                        i += 2;
                        continue;
                    }
                    quote = None;
                }
                i += 1;
            }
            None => {
                // Cypher line comments (//), block comments, and backtick
                // symbolic names — a FROM inside any of them hijacked the
                // split (the injector's scanner handles all three; this
                // crate cannot import the root's shared home — the
                // terminal-scanner TD item).
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    i += 2;
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                    continue;
                }
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    // NESTED block comments (Postgres allows them; the
                    // injector's scanner counts depth).
                    i += 2;
                    let mut depth = 1usize;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                            depth += 1;
                            i += 2;
                        } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                            depth -= 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    continue;
                }
                if bytes[i] == b'\'' || bytes[i] == b'"' || bytes[i] == b'`' {
                    quote = Some(bytes[i]);
                    i += 1;
                    continue;
                }
                let boundary_before = i == 0 || bytes[i - 1].is_ascii_whitespace();
                if boundary_before
                    && i + 4 <= bytes.len()
                    && bytes[i..i + 4].eq_ignore_ascii_case(b"FROM")
                {
                    let after = i + 4;
                    let boundary_after = after >= bytes.len() || bytes[after].is_ascii_whitespace();
                    if boundary_after {
                        from_pos = Some(i);
                        break;
                    }
                }
                i += 1;
            }
        }
    }
    let Some(from_pos) = from_pos else {
        return Ok((trimmed.to_string(), None));
    };

    let before_from = trimmed[..from_pos].trim_end();
    let after_from = trimmed[from_pos + 4..].trim_start();
    let graph_name_len = after_from
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .count();

    if graph_name_len == 0 {
        return Err(anyhow!("Expected graph name after FROM in graph query"));
    }

    let graph_name = after_from[..graph_name_len].to_string();
    let remainder = after_from[graph_name_len..].trim_start();
    let normalized_query = if remainder.is_empty() {
        before_from.to_string()
    } else {
        format!("{} {}", before_from, remainder)
    };

    Ok((normalized_query, Some(graph_name)))
}

fn validate_compiled_query(compiled: &CompiledPattern, normalized_query: &str) -> Result<()> {
    let upper = normalized_query.to_uppercase();
    for unsupported in [
        "OPTIONAL MATCH",
        "CREATE ",
        "DELETE ",
        "MERGE ",
        " SET ",
        " REMOVE ",
        " WITH ",
        " UNION ",
    ] {
        if upper.contains(unsupported) {
            return Err(anyhow!(
                "Unsupported graph query clause '{}' in facade/federated graph subset",
                unsupported.trim()
            ));
        }
    }

    if compiled.nodes.is_empty() {
        return Err(anyhow!(
            "Graph query must contain at least one node pattern"
        ));
    }

    if !compiled.paths.is_empty() {
        return Err(anyhow!(
            "Explicit path bindings are not supported in the facade/federated graph subset"
        ));
    }

    if compiled.edges.is_empty() {
        if compiled.nodes.len() != 1 {
            return Err(anyhow!(
                "Disconnected graph matches are not supported; use a single node pattern or a linear traversal chain"
            ));
        }
    } else {
        if compiled.nodes.len() != compiled.edges.len() + 1 {
            return Err(anyhow!(
                "Only linear graph traversal chains are supported in the facade/federated graph subset"
            ));
        }

        for (index, edge) in compiled.edges.iter().enumerate() {
            if compiled.nodes[index].variable != edge.from_variable
                || compiled.nodes[index + 1].variable != edge.to_variable
            {
                return Err(anyhow!(
                    "Only linear graph traversal chains are supported in the facade/federated graph subset"
                ));
            }

            if edge
                .properties
                .values()
                .any(|constraint| !matches!(constraint, PropertyConstraint::Equals(_)))
            {
                return Err(anyhow!(
                    "Edge property maps only support equality constraints in the facade/federated graph subset"
                ));
            }
        }
    }

    let mut known_variables = HashSet::new();
    for node in &compiled.nodes {
        if node.optional {
            return Err(anyhow!(
                "OPTIONAL MATCH is not supported in the facade/federated graph subset"
            ));
        }
        if !known_variables.insert(node.variable.clone()) {
            return Err(anyhow!(
                "Repeated node variable '{}' is not supported in the facade/federated graph subset",
                node.variable
            ));
        }
        if node
            .properties
            .values()
            .any(|constraint| !matches!(constraint, PropertyConstraint::Equals(_)))
        {
            return Err(anyhow!(
                "Node property maps only support equality constraints in the facade/federated graph subset"
            ));
        }
    }

    for edge in &compiled.edges {
        if edge.optional {
            return Err(anyhow!(
                "OPTIONAL MATCH is not supported in the facade/federated graph subset"
            ));
        }
        if let Some(variable) = &edge.variable
            && !known_variables.insert(variable.clone())
        {
            return Err(anyhow!(
                "Repeated edge variable '{}' is not supported in the facade/federated graph subset",
                variable
            ));
        }
    }

    for clause in &compiled.where_clauses {
        validate_where_clause(clause, &known_variables)?;
    }

    for projection in &compiled.return_spec.projections {
        match projection {
            PropertyProjection::Variable(variable) => {
                if !known_variables.contains(variable) {
                    return Err(anyhow!(
                        "Graph query RETURN references unknown variable '{}'",
                        variable
                    ));
                }
            }
            PropertyProjection::Property { variable, .. } => {
                if !known_variables.contains(variable) {
                    return Err(anyhow!(
                        "Graph query RETURN references unknown variable '{}'",
                        variable
                    ));
                }
            }
            PropertyProjection::Count
            | PropertyProjection::Sum { .. }
            | PropertyProjection::Avg { .. }
            | PropertyProjection::Min { .. }
            | PropertyProjection::Max { .. } => {
                return Err(anyhow!(
                    "Aggregations are not supported in the facade/federated graph subset"
                ));
            }
        }
    }

    for (order_key, _) in &compiled.return_spec.order_by {
        if !compiled
            .return_spec
            .variables
            .iter()
            .any(|column| column == order_key)
        {
            return Err(anyhow!(
                "ORDER BY '{}' must reference a returned column alias in the facade/federated graph subset",
                order_key
            ));
        }
    }

    Ok(())
}

fn validate_where_clause(clause: &WhereClause, known_variables: &HashSet<String>) -> Result<()> {
    match clause {
        WhereClause::Property {
            variable,
            constraint,
            ..
        } => {
            if !known_variables.contains(variable) {
                return Err(anyhow!(
                    "Graph query WHERE references unknown variable '{}'",
                    variable
                ));
            }

            if !is_supported_constraint(constraint) {
                return Err(anyhow!(
                    "WHERE clause contains an unsupported constraint in the facade/federated graph subset"
                ));
            }
        }
        WhereClause::And(left, right) | WhereClause::Or(left, right) => {
            validate_where_clause(left, known_variables)?;
            validate_where_clause(right, known_variables)?;
        }
        WhereClause::Not(inner) => validate_where_clause(inner, known_variables)?,
    }

    Ok(())
}

fn is_supported_constraint(constraint: &PropertyConstraint) -> bool {
    matches!(
        constraint,
        PropertyConstraint::Equals(_)
            | PropertyConstraint::NotEquals(_)
            | PropertyConstraint::GreaterThan(_)
            | PropertyConstraint::GreaterThanOrEqual(_)
            | PropertyConstraint::GreaterOrEqual(_)
            | PropertyConstraint::LessThan(_)
            | PropertyConstraint::LessThanOrEqual(_)
            | PropertyConstraint::LessOrEqual(_)
            | PropertyConstraint::Contains(_)
            | PropertyConstraint::StartsWith(_)
            | PropertyConstraint::EndsWith(_)
            | PropertyConstraint::Exists
            | PropertyConstraint::NotExists
            | PropertyConstraint::In(_)
            | PropertyConstraint::NotIn(_)
            | PropertyConstraint::Regex(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_supported_graph_query_extracts_from_clause() {
        let parsed = parse_supported_graph_query(
            "MATCH (n:Person) FROM social RETURN n.name AS name",
            None,
            None,
        )
        .expect("parse graph query");

        assert_eq!(parsed.graph_id(), "social");
        assert_eq!(
            parsed.normalized_query(),
            "MATCH (n:Person) RETURN n.name AS name"
        );
        assert_eq!(parsed.output_columns(), &["name".to_string()]);
    }

    #[test]
    fn describe_supported_graph_query_legacy_rows_and_depth() {
        let descriptor = describe_supported_graph_query(
            "MATCH (n:Person)-[:KNOWS]->(m:Person) RETURN m",
            None,
            Some("social"),
        )
        .expect("describe graph query");

        assert_eq!(descriptor.graph_id(), "social");
        assert_eq!(
            descriptor.normalized_query(),
            "MATCH (n:Person)-[:KNOWS]->(m:Person) RETURN m"
        );
        assert_eq!(
            descriptor.output_columns(),
            &[
                "node_id".to_string(),
                "label".to_string(),
                "properties".to_string()
            ]
        );
        assert!(descriptor.uses_legacy_node_rows());
        assert_eq!(descriptor.max_depth(), 1);
    }

    #[test]
    fn parse_supported_graph_query_rejects_unknown_return_variable() {
        let error = parse_supported_graph_query(
            "MATCH (n:Person) RETURN m.name AS name",
            None,
            Some("social"),
        )
        .expect_err("unknown variable should fail validation");

        assert!(error.to_string().contains("unknown variable 'm'"));
    }
}
