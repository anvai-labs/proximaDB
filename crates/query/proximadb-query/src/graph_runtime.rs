use anyhow::{Result, anyhow};
use proximadb_graph_query::ast::{
    CompiledPattern, EdgeDirection, EdgePattern, NodePattern, PropertyConstraint,
    PropertyProjection, WhereClause,
};
pub use proximadb_graph_query::declarative::graph_query_row_id;
use proximadb_graph_query::service::GraphQueryReadService;
use proximadb_proto::proximadb_v1::{
    Edge, EdgeQuery, Node, NodeQuery, PropertyArray, PropertyFilter, PropertyFilterOperator,
    PropertyObject, PropertyValue, VectorData, property_value,
};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::graph_lowering::{LoweredGraphQuery, ParsedGraphQuery, parse_supported_graph_query};

#[derive(Debug, Default, Clone, Copy)]
pub struct GraphExecutionStats {
    pub rows_returned: usize,
    pub matched_nodes: usize,
    pub matched_edges: usize,
}

#[derive(Debug, Clone)]
pub struct ExecutedGraphQuery {
    pub rows: Vec<Value>,
    pub stats: GraphExecutionStats,
}

#[derive(Debug, Clone)]
pub struct LoweredGraphQueryResult {
    pub rows: Vec<Value>,
    pub stats: GraphExecutionStats,
}

pub type GraphQueryRuntimeResult = LoweredGraphQueryResult;

#[derive(Debug, Clone)]
enum BoundValue {
    Node(Arc<Node>),
    Edge(Arc<Edge>),
}

impl BoundValue {
    fn as_node(&self) -> Option<&Arc<Node>> {
        match self {
            Self::Node(node) => Some(node),
            Self::Edge(_) => None,
        }
    }

    fn identity(&self) -> (&'static str, &str) {
        match self {
            Self::Node(node) => ("node", &node.id),
            Self::Edge(edge) => ("edge", &edge.id),
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::Node(node) => node_to_json(node),
            Self::Edge(edge) => edge_to_json(edge),
        }
    }
}

type BindingRow = HashMap<String, BoundValue>;

pub async fn discover_default_graph_id(graph_ops: &dyn GraphQueryReadService) -> Option<String> {
    let graphs = graph_ops.list_graphs().await.ok()?;
    if graphs.is_empty() {
        None
    } else if graphs.iter().any(|graph| graph == "default") {
        Some("default".to_string())
    } else {
        graphs.into_iter().next()
    }
}

pub async fn execute_supported_graph_query(
    graph_ops: &dyn GraphQueryReadService,
    parsed: &ParsedGraphQuery,
) -> Result<ExecutedGraphQuery> {
    execute_supported_graph_query_with_start_nodes(graph_ops, parsed, None).await
}

pub async fn execute_lowered_graph_query(
    graph_ops: &dyn GraphQueryReadService,
    lowered: &LoweredGraphQuery,
) -> Result<LoweredGraphQueryResult> {
    execute_lowered_graph_query_with_start_nodes(graph_ops, lowered, None).await
}

pub async fn execute_lowered_graph_query_with_start_nodes(
    graph_ops: &dyn GraphQueryReadService,
    lowered: &LoweredGraphQuery,
    start_node_ids: Option<&[String]>,
) -> Result<LoweredGraphQueryResult> {
    let parsed = parse_supported_graph_query(
        lowered.normalized_query(),
        Some(lowered.graph_id()),
        Some(lowered.graph_id()),
    )?;
    let executed = if let Some(start_node_ids) = start_node_ids {
        execute_supported_graph_query_with_start_nodes(graph_ops, &parsed, Some(start_node_ids))
            .await?
    } else {
        execute_supported_graph_query(graph_ops, &parsed).await?
    };
    let rows = executed
        .rows
        .into_iter()
        .map(|row| {
            shape_graph_query_row(
                row,
                lowered.uses_legacy_node_rows(),
                lowered.output_columns(),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LoweredGraphQueryResult {
        rows,
        stats: executed.stats,
    })
}

/// Execute a lowered declarative graph query through the shared graph subset.
///
/// Returned rows are already shaped into the canonical outward contract:
/// scalar projections preserve their declared columns, while whole-node
/// projections use the legacy `node_id`/`label`/`properties` envelope.
pub async fn execute_graph_query_expr(
    graph_ops: &dyn GraphQueryReadService,
    expr: &LoweredGraphQuery,
) -> Result<LoweredGraphQueryResult> {
    execute_lowered_graph_query(graph_ops, expr).await
}

/// Execute a lowered declarative graph query optionally constraining the first
/// bound node variable to explicit node IDs.
pub async fn execute_graph_query_expr_with_start_nodes(
    graph_ops: &dyn GraphQueryReadService,
    expr: &LoweredGraphQuery,
    start_node_ids: Option<&[String]>,
) -> Result<LoweredGraphQueryResult> {
    execute_lowered_graph_query_with_start_nodes(graph_ops, expr, start_node_ids).await
}

/// Hard cap on variable-length `*min..max` expansion depth. Per-source work is already
/// bounded to O(V + E) by the visited set; this backstops a user-supplied `max` far larger
/// than any real graph diameter (e.g. `*1..1000000`) so a single query cannot iterate the
/// depth loop pathologically.
const MAX_VARIABLE_LENGTH_DEPTH: u32 = 64;

pub async fn execute_supported_graph_query_with_start_nodes(
    graph_ops: &dyn GraphQueryReadService,
    parsed: &ParsedGraphQuery,
    start_node_ids: Option<&[String]>,
) -> Result<ExecutedGraphQuery> {
    let compiled = parsed.compiled();
    let start_pattern = &compiled.nodes[0];
    let mut bindings =
        initial_bindings(graph_ops, parsed.graph_id(), start_pattern, start_node_ids).await?;

    for (edge_pattern, next_node_pattern) in
        compiled.edges.iter().zip(compiled.nodes.iter().skip(1))
    {
        let mut next_bindings = Vec::new();

        for binding in bindings {
            let Some(current_node) = binding
                .get(&edge_pattern.from_variable)
                .and_then(BoundValue::as_node)
                .cloned()
            else {
                return Err(anyhow!(
                    "Traversal variable '{}' is not bound to a node",
                    edge_pattern.from_variable
                ));
            };

            match edge_pattern.var_length {
                None => {
                    // Fixed single hop: bind both the traversed edge and the adjacent node.
                    let candidate_edges = query_candidate_edges(
                        graph_ops,
                        parsed.graph_id(),
                        &current_node.id,
                        edge_pattern,
                    )
                    .await?;

                    for edge in candidate_edges {
                        let next_node_id = resolve_adjacent_node_id(
                            &current_node.id,
                            edge.as_ref(),
                            edge_pattern.direction,
                        );
                        let Some(next_node) =
                            graph_ops.get_node(parsed.graph_id(), &next_node_id).await?
                        else {
                            continue;
                        };

                        if !matches_node_pattern(next_node.as_ref(), next_node_pattern) {
                            continue;
                        }

                        let mut next_binding = binding.clone();
                        let edge_value = BoundValue::Edge(edge.clone());
                        if let Some(edge_var) = &edge_pattern.variable {
                            if !binding_is_compatible(next_binding.get(edge_var), &edge_value) {
                                continue;
                            }
                            next_binding.insert(edge_var.clone(), edge_value);
                        } else {
                            let synthetic_key = format!("_edge_{}", edge.id);
                            next_binding.insert(synthetic_key, edge_value);
                        }

                        let next_node_value = BoundValue::Node(next_node.clone());
                        if !binding_is_compatible(
                            next_binding.get(&next_node_pattern.variable),
                            &next_node_value,
                        ) {
                            continue;
                        }
                        next_binding.insert(next_node_pattern.variable.clone(), next_node_value);
                        next_bindings.push(next_binding);
                    }
                }
                Some((min_hops, max_hops)) => {
                    // Variable-length `*min..max`: bounded BFS from the current node,
                    // emitting one binding per DISTINCT node reachable in [min, max] hops
                    // that matches the terminal pattern. This is *reachability* semantics —
                    // the cost-bounded default for impact / lineage / ring tracing — rather
                    // than per-path multiplicity. Each node is expanded at most once, so the
                    // work is O(V + E) per source regardless of `max`; MAX_VARIABLE_LENGTH_DEPTH
                    // backstops pathological graphs. Intermediate nodes are unconstrained (only
                    // the edge type filters each hop); the terminal pattern gates emission.
                    let max_depth = max_hops.min(MAX_VARIABLE_LENGTH_DEPTH) as usize;
                    let min_depth = (min_hops.max(1)) as usize;

                    let mut visited: HashSet<String> = HashSet::new();
                    visited.insert(current_node.id.clone());
                    let mut emitted: HashSet<String> = HashSet::new();
                    let mut frontier = vec![current_node.clone()];

                    for depth in 1..=max_depth {
                        let mut next_frontier = Vec::new();
                        for node in &frontier {
                            let candidate_edges = query_candidate_edges(
                                graph_ops,
                                parsed.graph_id(),
                                &node.id,
                                edge_pattern,
                            )
                            .await?;
                            for edge in candidate_edges {
                                let adj_id = resolve_adjacent_node_id(
                                    &node.id,
                                    edge.as_ref(),
                                    edge_pattern.direction,
                                );
                                let Some(adj_node) =
                                    graph_ops.get_node(parsed.graph_id(), &adj_id).await?
                                else {
                                    continue;
                                };

                                if depth >= min_depth
                                    && matches_node_pattern(adj_node.as_ref(), next_node_pattern)
                                    && emitted.insert(adj_id.clone())
                                {
                                    let next_node_value = BoundValue::Node(adj_node.clone());
                                    if binding_is_compatible(
                                        binding.get(&next_node_pattern.variable),
                                        &next_node_value,
                                    ) {
                                        let mut next_binding = binding.clone();
                                        next_binding.insert(
                                            next_node_pattern.variable.clone(),
                                            next_node_value,
                                        );
                                        next_bindings.push(next_binding);
                                    }
                                }

                                if depth < max_depth && visited.insert(adj_id.clone()) {
                                    next_frontier.push(adj_node.clone());
                                }
                            }
                        }
                        frontier = next_frontier;
                        if frontier.is_empty() {
                            break;
                        }
                    }
                }
            }
        }

        bindings = next_bindings;
        if bindings.is_empty() {
            break;
        }
    }

    bindings.retain(|binding| {
        compiled
            .where_clauses
            .iter()
            .all(|clause| matches_where_clause(binding, clause))
    });

    let matched_nodes = bindings
        .iter()
        .flat_map(|binding| binding.values())
        .filter_map(|value| match value {
            BoundValue::Node(node) => Some(node.id.clone()),
            BoundValue::Edge(_) => None,
        })
        .collect::<HashSet<_>>()
        .len();

    let matched_edges = bindings
        .iter()
        .flat_map(|binding| binding.values())
        .filter_map(|value| match value {
            BoundValue::Edge(edge) => Some(edge.id.clone()),
            BoundValue::Node(_) => None,
        })
        .collect::<HashSet<_>>()
        .len();

    let projected_rows = bindings
        .into_iter()
        .map(|binding| project_row(&binding, compiled))
        .collect::<Result<Vec<_>>>()?;
    let rows = apply_row_modifiers(projected_rows, compiled)?;

    Ok(ExecutedGraphQuery {
        stats: GraphExecutionStats {
            rows_returned: rows.len(),
            matched_nodes,
            matched_edges,
        },
        rows,
    })
}

pub fn shape_graph_query_row(
    row: Value,
    uses_legacy_node_rows: bool,
    output_columns: &[String],
) -> Result<Value> {
    if uses_legacy_node_rows {
        materialize_legacy_graph_query_row(row)
    } else {
        retain_graph_query_output_columns(row, output_columns)
    }
}

pub fn legacy_graph_row_to_node(row: &Value) -> Result<Arc<Node>> {
    let Value::Object(columns) = row else {
        return Err(anyhow!(
            "Legacy graph row projection expected an object row"
        ));
    };

    let id = columns
        .get("node_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("Legacy graph row is missing 'node_id'"))?
        .to_string();
    let labels = columns
        .get("label")
        .and_then(Value::as_str)
        .map(|label| vec![label.to_string()])
        .unwrap_or_default();
    let properties = columns
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(key, value)| (key.clone(), json_value_to_property_value(value)))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();

    Ok(Arc::new(Node {
        id,
        labels,
        properties,
        embedding: None,
        created_at_ms: 0,
        updated_at_ms: 0,
    }))
}

async fn initial_bindings(
    graph_ops: &dyn GraphQueryReadService,
    graph_id: &str,
    start_pattern: &NodePattern,
    start_node_ids: Option<&[String]>,
) -> Result<Vec<BindingRow>> {
    let start_nodes = if let Some(start_node_ids) = start_node_ids {
        query_candidate_start_nodes(graph_ops, graph_id, start_pattern, start_node_ids).await?
    } else {
        query_candidate_nodes(graph_ops, graph_id, start_pattern).await?
    };

    Ok(start_nodes
        .into_iter()
        .map(|node| {
            let mut row = BindingRow::new();
            row.insert(start_pattern.variable.clone(), BoundValue::Node(node));
            row
        })
        .collect())
}

fn retain_graph_query_output_columns(row: Value, output_columns: &[String]) -> Result<Value> {
    let Value::Object(object) = row else {
        return Err(anyhow!("Graph query row projection expected an object row"));
    };

    if output_columns.is_empty() {
        return Ok(Value::Object(object));
    }

    let projected = output_columns
        .iter()
        .filter_map(|column| {
            object
                .get(column)
                .cloned()
                .map(|value| (column.clone(), value))
        })
        .collect::<serde_json::Map<_, _>>();
    Ok(Value::Object(projected))
}

fn materialize_legacy_graph_query_row(row: Value) -> Result<Value> {
    let Value::Object(columns) = row else {
        return Err(anyhow!(
            "Legacy graph row materialization expected an object row"
        ));
    };
    let Some(node_value) = columns.values().next() else {
        return Err(anyhow!(
            "Legacy graph row materialization expected a projected node value"
        ));
    };
    let Value::Object(node) = node_value else {
        return Err(anyhow!(
            "Legacy graph row materialization expected a node object"
        ));
    };

    let node_id = node.get("id").cloned().unwrap_or(Value::Null);
    let label = node
        .get("labels")
        .and_then(|labels| labels.as_array())
        .and_then(|labels| labels.first())
        .cloned()
        .unwrap_or(Value::Null);
    let properties = node
        .get("properties")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    Ok(serde_json::json!({
        "node_id": node_id,
        "label": label,
        "properties": properties,
    }))
}

async fn query_candidate_nodes(
    graph_ops: &dyn GraphQueryReadService,
    graph_id: &str,
    pattern: &NodePattern,
) -> Result<Vec<Arc<Node>>> {
    if let Some(id) = extract_identity_constraint(pattern.properties.get("id"))
        && let Some(node) = graph_ops.get_node(graph_id, &id).await?
    {
        if matches_node_pattern(node.as_ref(), pattern) {
            return Ok(vec![node]);
        }
        return Ok(Vec::new());
    }

    let filters = pattern
        .properties
        .iter()
        .filter_map(|(key, constraint)| property_filter_from_constraint(key, constraint))
        .collect::<Vec<_>>();

    let query = NodeQuery {
        graph_id: graph_id.to_string(),
        labels: pattern.labels.clone(),
        filters,
        limit: None,
        offset: None,
        continuation_token: None,
    };

    let mut nodes = graph_ops.query_nodes(graph_id, query).await?;
    nodes.retain(|node| matches_node_pattern(node.as_ref(), pattern));
    Ok(nodes)
}

async fn query_candidate_start_nodes(
    graph_ops: &dyn GraphQueryReadService,
    graph_id: &str,
    pattern: &NodePattern,
    start_node_ids: &[String],
) -> Result<Vec<Arc<Node>>> {
    let mut nodes = Vec::new();
    let mut seen = HashSet::new();

    for node_id in start_node_ids {
        if !seen.insert(node_id.clone()) {
            continue;
        }

        let Some(node) = graph_ops.get_node(graph_id, node_id).await? else {
            continue;
        };

        if matches_node_pattern(node.as_ref(), pattern) {
            nodes.push(node);
        }
    }

    Ok(nodes)
}

async fn query_candidate_edges(
    graph_ops: &dyn GraphQueryReadService,
    graph_id: &str,
    current_node_id: &str,
    pattern: &EdgePattern,
) -> Result<Vec<Arc<Edge>>> {
    let filters = pattern
        .properties
        .iter()
        .filter_map(|(key, constraint)| property_filter_from_constraint(key, constraint))
        .collect::<Vec<_>>();

    let mut collected = Vec::new();
    let mut seen = HashSet::new();

    let mut push_edges = |edges: Vec<Arc<Edge>>| {
        for edge in edges {
            if seen.insert(edge.id.clone()) {
                collected.push(edge);
            }
        }
    };

    match pattern.direction {
        EdgeDirection::Outgoing => {
            push_edges(
                graph_ops
                    .query_edges(
                        graph_id,
                        EdgeQuery {
                            graph_id: graph_id.to_string(),
                            from_node_id: Some(current_node_id.to_string()),
                            to_node_id: None,
                            edge_types: pattern.edge_types.clone(),
                            filters: filters.clone(),
                            limit: None,
                            offset: None,
                            continuation_token: None,
                        },
                    )
                    .await?,
            );
        }
        EdgeDirection::Incoming => {
            push_edges(
                graph_ops
                    .query_edges(
                        graph_id,
                        EdgeQuery {
                            graph_id: graph_id.to_string(),
                            from_node_id: None,
                            to_node_id: Some(current_node_id.to_string()),
                            edge_types: pattern.edge_types.clone(),
                            filters: filters.clone(),
                            limit: None,
                            offset: None,
                            continuation_token: None,
                        },
                    )
                    .await?,
            );
        }
        EdgeDirection::Bidirectional => {
            push_edges(
                graph_ops
                    .query_edges(
                        graph_id,
                        EdgeQuery {
                            graph_id: graph_id.to_string(),
                            from_node_id: Some(current_node_id.to_string()),
                            to_node_id: None,
                            edge_types: pattern.edge_types.clone(),
                            filters: filters.clone(),
                            limit: None,
                            offset: None,
                            continuation_token: None,
                        },
                    )
                    .await?,
            );
            push_edges(
                graph_ops
                    .query_edges(
                        graph_id,
                        EdgeQuery {
                            graph_id: graph_id.to_string(),
                            from_node_id: None,
                            to_node_id: Some(current_node_id.to_string()),
                            edge_types: pattern.edge_types.clone(),
                            filters,
                            limit: None,
                            offset: None,
                            continuation_token: None,
                        },
                    )
                    .await?,
            );
        }
    }

    collected.retain(|edge| matches_edge_pattern(current_node_id, edge.as_ref(), pattern));
    Ok(collected)
}

fn property_filter_from_constraint(
    key: &str,
    constraint: &PropertyConstraint,
) -> Option<PropertyFilter> {
    if matches!(key, "id" | "labels" | "type" | "source" | "target") {
        return None;
    }

    let PropertyConstraint::Equals(value) = constraint else {
        return None;
    };
    let value = json_to_property_value(value)?;

    Some(PropertyFilter {
        key: key.to_string(),
        operator: PropertyFilterOperator::Equals as i32,
        value: Some(value),
    })
}

fn extract_identity_constraint(constraint: Option<&PropertyConstraint>) -> Option<String> {
    let PropertyConstraint::Equals(value) = constraint? else {
        return None;
    };

    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn json_to_property_value(value: &Value) -> Option<PropertyValue> {
    let value = match value {
        Value::Null => return None,
        Value::Bool(value) => property_value::Value::BoolValue(*value),
        Value::Number(value) => {
            if let Some(number) = value.as_i64() {
                property_value::Value::IntValue(number)
            } else {
                property_value::Value::DoubleValue(value.as_f64()?)
            }
        }
        Value::String(value) => property_value::Value::StringValue(value.clone()),
        Value::Array(values) => {
            let values = values
                .iter()
                .filter_map(json_to_property_value)
                .collect::<Vec<_>>();
            property_value::Value::ArrayValue(PropertyArray { values })
        }
        Value::Object(values) => {
            let fields = values
                .iter()
                .filter_map(|(key, value)| {
                    json_to_property_value(value).map(|value| (key.clone(), value))
                })
                .collect::<HashMap<_, _>>();
            property_value::Value::ObjectValue(PropertyObject { fields })
        }
    };

    Some(PropertyValue { value: Some(value) })
}

fn property_value_to_json(value: &PropertyValue) -> Value {
    match &value.value {
        Some(property_value::Value::StringValue(value)) => Value::String(value.clone()),
        Some(property_value::Value::IntValue(value)) => Value::Number((*value).into()),
        Some(property_value::Value::DoubleValue(value)) => {
            serde_json::Number::from_f64(*value).map_or(Value::Null, Value::Number)
        }
        Some(property_value::Value::BoolValue(value)) => Value::Bool(*value),
        Some(property_value::Value::BytesValue(bytes)) => {
            use base64::Engine;
            Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        Some(property_value::Value::ArrayValue(values)) => Value::Array(
            values
                .values
                .iter()
                .map(property_value_to_json)
                .collect::<Vec<_>>(),
        ),
        Some(property_value::Value::ObjectValue(object)) => Value::Object(
            object
                .fields
                .iter()
                .map(|(key, value)| (key.clone(), property_value_to_json(value)))
                .collect::<Map<_, _>>(),
        ),
        Some(property_value::Value::VectorValue(vector)) => Value::Array(
            vector
                .values
                .iter()
                .map(|value| Value::from(*value))
                .collect::<Vec<_>>(),
        ),
        None => Value::Null,
    }
}

fn json_value_to_property_value(value: &Value) -> PropertyValue {
    let value = match value {
        Value::Null => None,
        Value::Bool(value) => Some(property_value::Value::BoolValue(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(property_value::Value::IntValue)
            .or_else(|| value.as_f64().map(property_value::Value::DoubleValue)),
        Value::String(value) => Some(property_value::Value::StringValue(value.clone())),
        Value::Array(values) => {
            if values.iter().all(Value::is_number) {
                Some(property_value::Value::VectorValue(VectorData {
                    values: values
                        .iter()
                        .filter_map(|value| value.as_f64().map(|value| value as f32))
                        .collect(),
                }))
            } else {
                Some(property_value::Value::ArrayValue(PropertyArray {
                    values: values.iter().map(json_value_to_property_value).collect(),
                }))
            }
        }
        Value::Object(values) => Some(property_value::Value::ObjectValue(PropertyObject {
            fields: values
                .iter()
                .map(|(key, value)| (key.clone(), json_value_to_property_value(value)))
                .collect(),
        })),
    };

    PropertyValue { value }
}

fn matches_node_pattern(node: &Node, pattern: &NodePattern) -> bool {
    if !pattern
        .labels
        .iter()
        .all(|label| node.labels.iter().any(|candidate| candidate == label))
    {
        return false;
    }

    pattern.properties.iter().all(|(key, constraint)| {
        let actual = resolve_node_property(node, key);
        matches_constraint(&actual, constraint)
    })
}

fn matches_edge_pattern(current_node_id: &str, edge: &Edge, pattern: &EdgePattern) -> bool {
    let direction_matches = match pattern.direction {
        EdgeDirection::Outgoing => edge.from_node_id == current_node_id,
        EdgeDirection::Incoming => edge.to_node_id == current_node_id,
        EdgeDirection::Bidirectional => {
            edge.from_node_id == current_node_id || edge.to_node_id == current_node_id
        }
    };

    direction_matches
        && (pattern.edge_types.is_empty()
            || pattern
                .edge_types
                .iter()
                .any(|edge_type| edge_type == &edge.edge_type))
        && pattern.properties.iter().all(|(key, constraint)| {
            let actual = resolve_edge_property(edge, key);
            matches_constraint(&actual, constraint)
        })
}

fn resolve_adjacent_node_id(
    current_node_id: &str,
    edge: &Edge,
    direction: EdgeDirection,
) -> String {
    match direction {
        EdgeDirection::Outgoing => edge.to_node_id.clone(),
        EdgeDirection::Incoming => edge.from_node_id.clone(),
        EdgeDirection::Bidirectional => {
            if edge.from_node_id == current_node_id {
                edge.to_node_id.clone()
            } else {
                edge.from_node_id.clone()
            }
        }
    }
}

fn binding_is_compatible(existing: Option<&BoundValue>, candidate: &BoundValue) -> bool {
    existing.is_none_or(|current| current.identity() == candidate.identity())
}

fn matches_where_clause(binding: &BindingRow, clause: &WhereClause) -> bool {
    match clause {
        WhereClause::Property {
            variable,
            property,
            constraint,
        } => binding.get(variable).is_some_and(|value| {
            matches_constraint(&resolve_entity_property(value, property), constraint)
        }),
        WhereClause::And(left, right) => {
            matches_where_clause(binding, left) && matches_where_clause(binding, right)
        }
        WhereClause::Or(left, right) => {
            matches_where_clause(binding, left) || matches_where_clause(binding, right)
        }
        WhereClause::Xor(left, right) => {
            matches_where_clause(binding, left) ^ matches_where_clause(binding, right)
        }
        WhereClause::Not(inner) => !matches_where_clause(binding, inner),
    }
}

fn matches_constraint(actual: &Value, constraint: &PropertyConstraint) -> bool {
    match constraint {
        PropertyConstraint::Equals(expected) => actual == expected,
        PropertyConstraint::NotEquals(expected) => actual != expected,
        PropertyConstraint::GreaterThan(expected) => {
            compare_json_values(actual, expected) == Some(Ordering::Greater)
        }
        PropertyConstraint::GreaterThanOrEqual(expected)
        | PropertyConstraint::GreaterOrEqual(expected) => matches!(
            compare_json_values(actual, expected),
            Some(Ordering::Greater | Ordering::Equal)
        ),
        PropertyConstraint::LessThan(expected) => {
            compare_json_values(actual, expected) == Some(Ordering::Less)
        }
        PropertyConstraint::LessThanOrEqual(expected)
        | PropertyConstraint::LessOrEqual(expected) => matches!(
            compare_json_values(actual, expected),
            Some(Ordering::Less | Ordering::Equal)
        ),
        PropertyConstraint::In(values) => values.contains(actual),
        PropertyConstraint::NotIn(values) => !values.contains(actual),
        PropertyConstraint::Contains(expected) => actual
            .as_str()
            .is_some_and(|value| value.contains(expected)),
        PropertyConstraint::StartsWith(expected) => actual
            .as_str()
            .is_some_and(|value| value.starts_with(expected)),
        PropertyConstraint::EndsWith(expected) => actual
            .as_str()
            .is_some_and(|value| value.ends_with(expected)),
        PropertyConstraint::Regex(pattern) => actual
            .as_str()
            .and_then(|value| {
                regex::Regex::new(pattern)
                    .ok()
                    .map(|regex| regex.is_match(value))
            })
            .unwrap_or(false),
        PropertyConstraint::Exists => !actual.is_null(),
        PropertyConstraint::NotExists => actual.is_null(),
    }
}

fn compare_json_values(left: &Value, right: &Value) -> Option<Ordering> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => left.as_f64()?.partial_cmp(&right.as_f64()?),
        (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
        (Value::Bool(left), Value::Bool(right)) => Some(left.cmp(right)),
        (Value::Null, Value::Null) => Some(Ordering::Equal),
        (Value::Null, _) => Some(Ordering::Less),
        (_, Value::Null) => Some(Ordering::Greater),
        _ => Some(left.to_string().cmp(&right.to_string())),
    }
}

fn resolve_entity_property(entity: &BoundValue, property: &str) -> Value {
    match entity {
        BoundValue::Node(node) => resolve_node_property(node, property),
        BoundValue::Edge(edge) => resolve_edge_property(edge, property),
    }
}

fn resolve_node_property(node: &Node, property: &str) -> Value {
    match property {
        "id" => Value::String(node.id.clone()),
        "labels" => Value::Array(
            node.labels
                .iter()
                .map(|label| Value::String(label.clone()))
                .collect::<Vec<_>>(),
        ),
        property => node
            .properties
            .get(property)
            .map_or(Value::Null, property_value_to_json),
    }
}

fn resolve_edge_property(edge: &Edge, property: &str) -> Value {
    match property {
        "id" => Value::String(edge.id.clone()),
        "source" | "from" | "from_node_id" => Value::String(edge.from_node_id.clone()),
        "target" | "to" | "to_node_id" => Value::String(edge.to_node_id.clone()),
        "type" | "edge_type" => Value::String(edge.edge_type.clone()),
        "weight" => edge
            .weight
            .and_then(serde_json::Number::from_f64)
            .map_or(Value::Null, Value::Number),
        property => edge
            .properties
            .get(property)
            .map_or(Value::Null, property_value_to_json),
    }
}

fn project_row(binding: &BindingRow, compiled: &CompiledPattern) -> Result<Value> {
    let mut row = Map::new();

    for (column, projection) in compiled
        .return_spec
        .variables
        .iter()
        .zip(compiled.return_spec.projections.iter())
    {
        let value = match projection {
            PropertyProjection::Variable(variable) => binding
                .get(variable)
                .map(BoundValue::to_json)
                .ok_or_else(|| {
                    anyhow!(
                        "Graph query projection references unknown variable '{}'",
                        variable
                    )
                })?,
            PropertyProjection::Property { variable, property } => binding
                .get(variable)
                .map(|value| resolve_entity_property(value, property))
                .ok_or_else(|| {
                    anyhow!(
                        "Graph query projection references unknown variable '{}'",
                        variable
                    )
                })?,
            PropertyProjection::Count
            | PropertyProjection::Sum { .. }
            | PropertyProjection::Avg { .. }
            | PropertyProjection::Min { .. }
            | PropertyProjection::Max { .. } => {
                return Err(anyhow!(
                    "Aggregations are not supported in the facade/federated graph subset"
                ));
            }
        };

        row.insert(column.clone(), value);
    }

    Ok(Value::Object(row))
}

fn apply_row_modifiers(mut rows: Vec<Value>, compiled: &CompiledPattern) -> Result<Vec<Value>> {
    if compiled.return_spec.distinct {
        let mut seen = HashSet::new();
        rows.retain(|row| {
            let key = serde_json::to_string(row).unwrap_or_default();
            seen.insert(key)
        });
    }

    if !compiled.return_spec.order_by.is_empty() {
        rows.sort_by(|left, right| {
            let left = left.as_object();
            let right = right.as_object();

            for (column, ascending) in &compiled.return_spec.order_by {
                let left_value = left.and_then(|row| row.get(column)).unwrap_or(&Value::Null);
                let right_value = right
                    .and_then(|row| row.get(column))
                    .unwrap_or(&Value::Null);
                let ordering =
                    compare_json_values(left_value, right_value).unwrap_or(Ordering::Equal);
                let ordering = if *ascending {
                    ordering
                } else {
                    ordering.reverse()
                };
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }

            Ordering::Equal
        });
    }

    let skip = compiled.return_spec.skip.unwrap_or(0);
    let limit = compiled.return_spec.limit.unwrap_or(usize::MAX);

    Ok(rows.into_iter().skip(skip).take(limit).collect())
}

fn node_to_json(node: &Node) -> Value {
    Value::Object(Map::from_iter([
        ("id".to_string(), Value::String(node.id.clone())),
        (
            "labels".to_string(),
            Value::Array(
                node.labels
                    .iter()
                    .map(|label| Value::String(label.clone()))
                    .collect::<Vec<_>>(),
            ),
        ),
        (
            "properties".to_string(),
            Value::Object(
                node.properties
                    .iter()
                    .map(|(key, value)| (key.clone(), property_value_to_json(value)))
                    .collect::<Map<_, _>>(),
            ),
        ),
    ]))
}

fn edge_to_json(edge: &Edge) -> Value {
    Value::Object(Map::from_iter([
        ("id".to_string(), Value::String(edge.id.clone())),
        (
            "source".to_string(),
            Value::String(edge.from_node_id.clone()),
        ),
        ("target".to_string(), Value::String(edge.to_node_id.clone())),
        ("type".to_string(), Value::String(edge.edge_type.clone())),
        (
            "weight".to_string(),
            edge.weight
                .and_then(serde_json::Number::from_f64)
                .map_or(Value::Null, Value::Number),
        ),
        (
            "properties".to_string(),
            Value::Object(
                edge.properties
                    .iter()
                    .map(|(key, value)| (key.clone(), property_value_to_json(value)))
                    .collect::<Map<_, _>>(),
            ),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use proximadb_graph_query::service::GraphQueryResult as QueryResult;

    #[derive(Default)]
    struct MockGraphReadService {
        graphs: Vec<String>,
        nodes_by_graph: HashMap<String, HashMap<String, Arc<Node>>>,
        edges_by_graph: HashMap<String, Vec<Arc<Edge>>>,
    }

    impl MockGraphReadService {
        fn with_graphs(graphs: &[&str]) -> Self {
            Self {
                graphs: graphs.iter().map(|graph| graph.to_string()).collect(),
                ..Self::default()
            }
        }

        fn insert_node(&mut self, graph_id: &str, node: Node) {
            self.nodes_by_graph
                .entry(graph_id.to_string())
                .or_default()
                .insert(node.id.clone(), Arc::new(node));
        }

        fn insert_edge(&mut self, graph_id: &str, edge: Edge) {
            self.edges_by_graph
                .entry(graph_id.to_string())
                .or_default()
                .push(Arc::new(edge));
        }
    }

    #[async_trait]
    impl GraphQueryReadService for MockGraphReadService {
        async fn list_graphs(&self) -> QueryResult<Vec<String>> {
            Ok(self.graphs.clone())
        }

        async fn get_node(&self, graph_id: &str, node_id: &str) -> QueryResult<Option<Arc<Node>>> {
            Ok(self
                .nodes_by_graph
                .get(graph_id)
                .and_then(|nodes| nodes.get(node_id))
                .cloned())
        }

        async fn query_nodes(
            &self,
            graph_id: &str,
            query: NodeQuery,
        ) -> QueryResult<Vec<Arc<Node>>> {
            let nodes = self
                .nodes_by_graph
                .get(graph_id)
                .into_iter()
                .flat_map(|nodes| nodes.values())
                .filter(|node| {
                    query.labels.is_empty()
                        || query
                            .labels
                            .iter()
                            .all(|label| node.labels.iter().any(|candidate| candidate == label))
                })
                .filter(|node| {
                    query.filters.iter().all(|filter| {
                        node.properties
                            .get(&filter.key)
                            .zip(filter.value.as_ref())
                            .is_some_and(|(actual, expected)| actual.value == expected.value)
                    })
                })
                .cloned()
                .collect();
            Ok(nodes)
        }

        async fn query_edges(
            &self,
            graph_id: &str,
            query: EdgeQuery,
        ) -> QueryResult<Vec<Arc<Edge>>> {
            let edges = self
                .edges_by_graph
                .get(graph_id)
                .into_iter()
                .flat_map(|edges| edges.iter())
                .filter(|edge| {
                    query
                        .from_node_id
                        .as_ref()
                        .is_none_or(|from_node_id| &edge.from_node_id == from_node_id)
                })
                .filter(|edge| {
                    query
                        .to_node_id
                        .as_ref()
                        .is_none_or(|to_node_id| &edge.to_node_id == to_node_id)
                })
                .filter(|edge| {
                    query.edge_types.is_empty()
                        || query
                            .edge_types
                            .iter()
                            .any(|edge_type| &edge.edge_type == edge_type)
                })
                .cloned()
                .collect();
            Ok(edges)
        }
    }

    fn pv_string(value: &str) -> PropertyValue {
        PropertyValue {
            value: Some(property_value::Value::StringValue(value.to_string())),
        }
    }

    fn seed_graph() -> MockGraphReadService {
        let mut service = MockGraphReadService::with_graphs(&["social"]);

        service.insert_node(
            "social",
            Node {
                id: "alice".to_string(),
                labels: vec!["Person".to_string()],
                properties: HashMap::from([("name".to_string(), pv_string("Alice"))]),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        );
        service.insert_node(
            "social",
            Node {
                id: "bob".to_string(),
                labels: vec!["Person".to_string()],
                properties: HashMap::from([("name".to_string(), pv_string("Bob"))]),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        );
        service.insert_node(
            "social",
            Node {
                id: "acme".to_string(),
                labels: vec!["Company".to_string()],
                properties: HashMap::from([("name".to_string(), pv_string("Acme"))]),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        );
        service.insert_edge(
            "social",
            Edge {
                id: "edge_knows".to_string(),
                from_node_id: "alice".to_string(),
                to_node_id: "bob".to_string(),
                edge_type: "KNOWS".to_string(),
                properties: HashMap::new(),
                weight: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        );
        service.insert_edge(
            "social",
            Edge {
                id: "edge_works_at".to_string(),
                from_node_id: "bob".to_string(),
                to_node_id: "acme".to_string(),
                edge_type: "WORKS_AT".to_string(),
                properties: HashMap::new(),
                weight: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        );

        service
    }

    #[test]
    fn property_filter_from_constraint_skips_special_and_non_equality_constraints() {
        assert!(
            property_filter_from_constraint(
                "id",
                &PropertyConstraint::Equals(Value::String("alice".to_string()))
            )
            .is_none()
        );
        assert!(
            property_filter_from_constraint(
                "name",
                &PropertyConstraint::GreaterThan(Value::String("Alice".to_string()))
            )
            .is_none()
        );

        let filter = property_filter_from_constraint(
            "name",
            &PropertyConstraint::Equals(Value::String("Alice".to_string())),
        )
        .expect("equality constraint should lower to property filter");

        assert_eq!(filter.key, "name");
        assert_eq!(filter.operator, PropertyFilterOperator::Equals as i32);
        assert_eq!(
            property_value_to_json(filter.value.as_ref().expect("filter value")),
            Value::String("Alice".to_string())
        );
    }

    #[tokio::test]
    async fn execute_supported_graph_query_multi_hop_projection() {
        let service = seed_graph();
        let parsed = crate::graph_lowering::parse_supported_graph_query(
            "MATCH (a:Person {id: \"alice\"})-[:KNOWS]->(b)-[:WORKS_AT]->(c:Company) RETURN b.name AS colleague, c.name AS company",
            None,
            Some("social"),
        )
        .expect("parse graph query");

        let executed = execute_supported_graph_query(&service, &parsed)
            .await
            .expect("execute graph query");

        assert_eq!(executed.stats.rows_returned, 1);
        assert_eq!(executed.stats.matched_nodes, 3);
        assert_eq!(executed.stats.matched_edges, 2);
        assert_eq!(
            executed.rows,
            vec![serde_json::json!({
                "colleague": "Bob",
                "company": "Acme"
            })]
        );
    }

    // A linear chain n0 -> n1 -> n2 -> n3 -> n4 over edge type LINK, for exercising
    // variable-length `*min..max` traversal.
    fn seed_chain() -> MockGraphReadService {
        let mut service = MockGraphReadService::with_graphs(&["chain"]);
        for id in ["n0", "n1", "n2", "n3", "n4"] {
            service.insert_node(
                "chain",
                Node {
                    id: id.to_string(),
                    labels: vec!["N".to_string()],
                    properties: HashMap::new(),
                    embedding: None,
                    created_at_ms: 0,
                    updated_at_ms: 0,
                },
            );
        }
        for (from, to) in [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")] {
            service.insert_edge(
                "chain",
                Edge {
                    id: format!("{from}->{to}"),
                    from_node_id: from.to_string(),
                    to_node_id: to.to_string(),
                    edge_type: "LINK".to_string(),
                    properties: HashMap::new(),
                    weight: None,
                    created_at_ms: 0,
                    updated_at_ms: 0,
                },
            );
        }
        service
    }

    async fn chain_ids(service: &MockGraphReadService, query: &str) -> Vec<String> {
        let parsed = crate::graph_lowering::parse_supported_graph_query(query, None, Some("chain"))
            .expect("parse graph query");
        let executed = execute_supported_graph_query(service, &parsed)
            .await
            .expect("execute graph query");
        let mut ids = executed
            .rows
            .iter()
            .filter_map(|row| row.get("id").and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn variable_length_path_expands_every_depth_in_range() {
        let service = seed_chain();

        // `*1..3` from n0 must reach n1 (depth 1), n2 (depth 2), n3 (depth 3) — not n4 (depth 4).
        // Before the fix this returned only n1 (depth 1).
        assert_eq!(
            chain_ids(
                &service,
                "MATCH (a:N {id: \"n0\"})-[:LINK*1..3]->(x:N) RETURN x.id AS id",
            )
            .await,
            vec!["n1", "n2", "n3"],
        );

        // `*1..1` is a plain single hop.
        assert_eq!(
            chain_ids(
                &service,
                "MATCH (a:N {id: \"n0\"})-[:LINK*1..1]->(x:N) RETURN x.id AS id",
            )
            .await,
            vec!["n1"],
        );

        // The lower bound excludes the direct neighbour: `*2..4` starts at depth 2.
        assert_eq!(
            chain_ids(
                &service,
                "MATCH (a:N {id: \"n0\"})-[:LINK*2..4]->(x:N) RETURN x.id AS id",
            )
            .await,
            vec!["n2", "n3", "n4"],
        );
    }

    #[tokio::test]
    async fn anonymous_chained_pattern_traverses_two_explicit_hops() {
        let service = seed_chain();
        // Two explicit hops through an anonymous intermediate node `()` must reach n2.
        // Before the fix, anonymous nodes failed to parse and the query returned nothing.
        assert_eq!(
            chain_ids(
                &service,
                "MATCH (a:N {id: \"n0\"})-[:LINK]->()-[:LINK]->(x:N) RETURN x.id AS id",
            )
            .await,
            vec!["n2"],
        );
    }

    #[tokio::test]
    async fn execute_lowered_graph_query_materializes_legacy_rows() {
        let service = seed_graph();
        let lowered = crate::graph_lowering::lower_supported_graph_query(
            "MATCH (n:Person {id: \"alice\"}) RETURN n",
            None,
            Some("social"),
        )
        .expect("lower graph query");

        let executed = execute_lowered_graph_query(&service, &lowered)
            .await
            .expect("execute lowered graph query");

        assert_eq!(
            executed.rows,
            vec![serde_json::json!({
                "node_id": "alice",
                "label": "Person",
                "properties": { "name": "Alice" }
            })]
        );
    }

    #[tokio::test]
    async fn execute_lowered_graph_query_with_start_nodes_preserves_projection() {
        let service = seed_graph();
        let lowered = crate::graph_lowering::lower_supported_graph_query(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name AS neighbor",
            None,
            Some("social"),
        )
        .expect("lower graph query");

        let start_node_ids = vec!["alice".to_string()];
        let executed =
            execute_lowered_graph_query_with_start_nodes(&service, &lowered, Some(&start_node_ids))
                .await
                .expect("execute bound lowered graph query");

        assert_eq!(
            executed.rows,
            vec![serde_json::json!({
                "neighbor": "Bob"
            })]
        );
    }

    #[tokio::test]
    async fn discover_default_graph_id_prefers_default_graph() {
        let service = MockGraphReadService::with_graphs(&["social", "default"]);
        let discovered = discover_default_graph_id(&service).await;
        assert_eq!(discovered.as_deref(), Some("default"));
    }

    #[tokio::test]
    async fn discover_default_graph_id_returns_none_when_empty() {
        let service = MockGraphReadService::default();
        let discovered = discover_default_graph_id(&service).await;
        assert!(discovered.is_none());
    }

    #[test]
    fn legacy_graph_row_to_node_preserves_properties() {
        let row = serde_json::json!({
            "node_id": "alice",
            "label": "Person",
            "properties": {
                "name": "Alice",
                "embedding": [0.1, 0.2]
            }
        });

        let node = legacy_graph_row_to_node(&row).expect("legacy row should convert");
        assert_eq!(node.id, "alice");
        assert_eq!(node.labels, vec!["Person".to_string()]);
        assert!(node.properties.contains_key("name"));
        assert!(node.properties.contains_key("embedding"));
    }

    #[test]
    fn shape_graph_query_row_retains_declared_columns_only() {
        let row = serde_json::json!({
            "neighbor": "Bob",
            "company": "Acme"
        });

        let shaped = shape_graph_query_row(row, false, &["neighbor".to_string()])
            .expect("row projection should succeed");

        assert_eq!(shaped, serde_json::json!({ "neighbor": "Bob" }));
    }

    #[test]
    fn graph_query_row_id_falls_back_across_supported_shapes() {
        assert_eq!(
            graph_query_row_id(&serde_json::json!({ "node_id": "alice" }), 0),
            "alice"
        );
        assert_eq!(
            graph_query_row_id(&serde_json::json!({ "id": "edge_1" }), 1),
            "edge_1"
        );
        assert_eq!(
            graph_query_row_id(&serde_json::json!({ "node": { "id": "nested" } }), 2),
            "nested"
        );
        assert_eq!(
            graph_query_row_id(&serde_json::json!({ "name": "anonymous" }), 3),
            "graph_row_3"
        );
    }

    #[test]
    fn legacy_graph_row_to_node_requires_node_id() {
        let error = legacy_graph_row_to_node(&serde_json::json!({
            "label": "Person",
            "properties": { "name": "Alice" }
        }))
        .expect_err("legacy row without node_id should fail");

        assert!(error.to_string().contains("missing 'node_id'"));
    }
}
