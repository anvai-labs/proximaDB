//! Phase D: MultiModelPlan operator dispatch — spec §7.
//!
//! Routes `Operator::CrossModelJoin` → [`MshjExecutor`] and
//! `Operator::HybridTraverse` → [`HybridTraverseExecutor`].
//!
//! All other operators are passed through as no-ops in this phase; they are
//! handled by the storage-engine pipeline in `compute::pipeline_executor`.

use std::{cmp::Ordering, collections::BTreeMap, sync::Arc};

use anyhow::{Result, anyhow};
use proximadb_data_model::DataModel;
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_multimodel_plan::{AggregateExpression, AggregateFunction, MultiModelPlan, Operator};
use serde_json::{Map, Value};

use crate::operators::{
    hybrid_traverse::{AnnSeedProvider, GraphNeighbourProvider, HybridTraverseExecutor},
    mshj::MshjExecutor,
};

/// A row-oriented data source injected into the executor for Scan dispatch.
pub trait PlanDataSource: Send + Sync {
    fn scan(&self, model: DataModel, limit: Option<usize>) -> Result<Vec<Value>>;
}

/// Execution context for one plan run.
pub struct PlanExecutionContext {
    /// Source for Scan-dispatched rows (left/right sides of joins).
    pub data_source: Arc<dyn PlanDataSource>,
    /// ANN seed provider wired to HybridTraverse + VectorTopK.
    pub ann_provider: Option<Arc<dyn AnnSeedProvider>>,
    /// Graph neighbour provider wired to HybridTraverse.
    pub graph_provider: Option<Arc<dyn GraphNeighbourProvider>>,
    /// Query vector carried from a preceding VectorTopK op.
    current_query_vector: Vec<f32>,
    /// Rows produced by the last operator (pipeline accumulator).
    current_rows: Vec<Value>,
}

impl PlanExecutionContext {
    pub fn new(data_source: Arc<dyn PlanDataSource>) -> Self {
        Self {
            data_source,
            ann_provider: None,
            graph_provider: None,
            current_query_vector: Vec::new(),
            current_rows: Vec::new(),
        }
    }

    pub fn with_ann(mut self, ann: Arc<dyn AnnSeedProvider>) -> Self {
        self.ann_provider = Some(ann);
        self
    }

    pub fn with_graph(mut self, graph: Arc<dyn GraphNeighbourProvider>) -> Self {
        self.graph_provider = Some(graph);
        self
    }
}

/// Per-operator execution statistics.
#[derive(Debug, Clone, Default)]
pub struct OperatorStats {
    pub operator_index: usize,
    pub rows_in: usize,
    pub rows_out: usize,
    /// Microseconds spent in this operator.
    pub elapsed_us: u64,
}

/// Result of executing a full plan.
#[derive(Debug, Clone, Default)]
pub struct PlanExecutionResult {
    pub rows: Vec<Value>,
    pub operator_stats: Vec<OperatorStats>,
}

/// Phase D plan executor — dispatches operators to concrete executors.
pub struct PlanExecutor;

impl PlanExecutor {
    /// Execute `plan` given the injected `ctx` providers, returning all rows.
    pub fn execute(
        plan: &MultiModelPlan,
        ctx: &mut PlanExecutionContext,
    ) -> Result<PlanExecutionResult> {
        let mut stats_vec = Vec::with_capacity(plan.len());

        for (idx, operator) in plan.operators.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let rows_in = ctx.current_rows.len();

            Self::dispatch(idx, operator, ctx)?;

            stats_vec.push(OperatorStats {
                operator_index: idx,
                rows_in,
                rows_out: ctx.current_rows.len(),
                elapsed_us: t0.elapsed().as_micros() as u64,
            });
        }

        Ok(PlanExecutionResult {
            rows: std::mem::take(&mut ctx.current_rows),
            operator_stats: stats_vec,
        })
    }

    fn dispatch(idx: usize, operator: &Operator, ctx: &mut PlanExecutionContext) -> Result<()> {
        match operator {
            // ── Phase D: Cross-modal Multi-Stage Hash Join ─────────────────────
            Operator::CrossModelJoin {
                left_modality,
                right_modality,
                condition,
            } => {
                let left = ctx.data_source.scan(*left_modality, None)?;
                let right = ctx.data_source.scan(*right_modality, None)?;
                let (joined, _stats) = MshjExecutor::join(&left, &right, condition)?;
                ctx.current_rows = joined.into_iter().map(|r| r.record).collect();
            }

            // ── Phase D: Hybrid graph + vector traversal ───────────────────────
            Operator::HybridTraverse { edge_pattern } => {
                let ann = ctx
                    .ann_provider
                    .as_ref()
                    .ok_or_else(|| anyhow!("HybridTraverse requires an AnnSeedProvider"))?
                    .clone();
                let graph = ctx
                    .graph_provider
                    .as_ref()
                    .ok_or_else(|| anyhow!("HybridTraverse requires a GraphNeighbourProvider"))?
                    .clone();

                let beam_width = 10.max(ctx.current_rows.len());
                let top_k = beam_width;
                let executor = HybridTraverseExecutor::new(beam_width, top_k);
                let (nodes, _tstat) = executor.traverse(
                    &ctx.current_query_vector,
                    edge_pattern,
                    ann.as_ref(),
                    graph.as_ref(),
                )?;
                ctx.current_rows = nodes
                    .into_iter()
                    .map(|n| {
                        serde_json::json!({
                            "id": n.id,
                            "vector_score": n.vector_score,
                            "hop_depth": n.hop_depth,
                            "payload": n.payload,
                        })
                    })
                    .collect();
            }

            // ── VectorTopK: seed the query vector for downstream HybridTraverse ─
            Operator::VectorTopK {
                query_vector, k, ..
            } => {
                ctx.current_query_vector = query_vector.clone();
                if let Some(ann) = &ctx.ann_provider {
                    let seeds = ann.find_seeds(query_vector, *k)?;
                    ctx.current_rows = seeds
                        .into_iter()
                        .map(|(id, score)| serde_json::json!({ "id": id, "score": score }))
                        .collect();
                }
                // If no ANN provider, leave rows unchanged (plan validation catches this).
            }

            // ── Passthrough operators (storage-engine pipeline handles these) ───
            Operator::Scan {
                source,
                filter,
                columns,
                ..
            } => {
                // The data model name drives the scan; full engine dispatch is in
                // `compute::pipeline_executor`. Here we emit a placeholder scan.
                let model = Self::data_model_from_source(source);
                ctx.current_rows = ctx.data_source.scan(model, None)?;
                if let Some(filter) = filter {
                    ctx.current_rows
                        .retain(|row| Self::row_matches(row, filter));
                }
                if let Some(columns) = columns {
                    ctx.current_rows = ctx
                        .current_rows
                        .iter()
                        .map(|row| Self::project_row(row, columns))
                        .collect();
                }
            }

            Operator::Filter { expression } => {
                ctx.current_rows
                    .retain(|row| Self::row_matches(row, expression));
            }

            Operator::Project { columns } => {
                ctx.current_rows = ctx
                    .current_rows
                    .iter()
                    .map(|row| Self::project_row(row, columns))
                    .collect();
            }

            Operator::Limit { n, offset } => {
                let rows = std::mem::take(&mut ctx.current_rows);
                ctx.current_rows = rows.into_iter().skip(*offset).take(*n).collect();
            }

            Operator::Sort {
                column,
                ascending,
                limit,
            } => {
                Self::sort_rows(&mut ctx.current_rows, column, *ascending);
                if let Some(limit) = limit {
                    ctx.current_rows.truncate(*limit);
                }
            }

            Operator::TopK { k, sort_column } => {
                Self::sort_rows(&mut ctx.current_rows, sort_column, false);
                ctx.current_rows.truncate(*k);
            }

            Operator::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                ctx.current_rows = Self::aggregate_rows(&ctx.current_rows, group_by, aggregates)?;
            }

            // All other operators are no-ops in Phase D (handled by PipelineExecutor)
            _ => {
                tracing::trace!(
                    "PlanExecutor: operator {} ({:?}) is a passthrough in Phase D",
                    idx,
                    std::mem::discriminant(operator)
                );
            }
        }
        Ok(())
    }

    fn sort_rows(rows: &mut [Value], column: &str, ascending: bool) {
        rows.sort_by(|left, right| {
            let ordering = Self::compare_json_values(
                Self::field_value(left, column),
                Self::field_value(right, column),
            );
            if ascending {
                ordering
            } else {
                ordering.reverse()
            }
        });
    }

    fn aggregate_rows(
        rows: &[Value],
        group_by: &[String],
        aggregates: &[AggregateExpression],
    ) -> Result<Vec<Value>> {
        let mut groups: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
        for row in rows {
            let key = Self::group_key(row, group_by)?;
            groups.entry(key).or_default().push(row);
        }

        let mut out = Vec::with_capacity(groups.len());
        for (_key, group_rows) in groups {
            let mut row = Map::new();
            if let Some(first) = group_rows.first() {
                for column in group_by {
                    if let Some(value) = Self::field_value(first, column) {
                        row.insert(column.clone(), value.clone());
                    }
                }
            }

            for aggregate in aggregates {
                let name = aggregate
                    .alias
                    .clone()
                    .unwrap_or_else(|| Self::aggregate_column_name(aggregate));
                let value = Self::evaluate_aggregate(&group_rows, aggregate)?;
                row.insert(name, value);
            }

            out.push(Value::Object(row));
        }
        Ok(out)
    }

    fn group_key(row: &Value, group_by: &[String]) -> Result<String> {
        let values: Vec<Value> = group_by
            .iter()
            .map(|column| {
                Self::field_value(row, column)
                    .cloned()
                    .unwrap_or(Value::Null)
            })
            .collect();
        serde_json::to_string(&values).map_err(Into::into)
    }

    fn aggregate_column_name(aggregate: &AggregateExpression) -> String {
        let function = match aggregate.function {
            AggregateFunction::Count => "count",
            AggregateFunction::Sum => "sum",
            AggregateFunction::Avg => "avg",
            AggregateFunction::Min => "min",
            AggregateFunction::Max => "max",
            AggregateFunction::StdDev => "stddev",
            AggregateFunction::Variance => "variance",
            AggregateFunction::ArrayAgg => "array_agg",
        };
        format!("{}_{}", function, aggregate.column)
    }

    fn evaluate_aggregate(rows: &[&Value], aggregate: &AggregateExpression) -> Result<Value> {
        match aggregate.function {
            AggregateFunction::Count => {
                if aggregate.distinct {
                    let mut values = BTreeMap::new();
                    for row in rows {
                        if let Some(value) = Self::field_value(row, &aggregate.column) {
                            values.insert(serde_json::to_string(value)?, ());
                        }
                    }
                    Ok(serde_json::json!(values.len()))
                } else {
                    let count = rows
                        .iter()
                        .filter(|row| {
                            Self::field_value(row, &aggregate.column)
                                .is_some_and(|value| !value.is_null())
                        })
                        .count();
                    Ok(serde_json::json!(count))
                }
            }
            AggregateFunction::Sum => {
                let sum: f64 = rows
                    .iter()
                    .filter_map(|row| Self::field_value(row, &aggregate.column)?.as_f64())
                    .sum();
                Ok(serde_json::json!(sum))
            }
            AggregateFunction::Avg => {
                let values: Vec<f64> = rows
                    .iter()
                    .filter_map(|row| Self::field_value(row, &aggregate.column)?.as_f64())
                    .collect();
                let avg = if values.is_empty() {
                    Value::Null
                } else {
                    serde_json::json!(values.iter().sum::<f64>() / values.len() as f64)
                };
                Ok(avg)
            }
            AggregateFunction::Min => Ok(rows
                .iter()
                .filter_map(|row| Self::field_value(row, &aggregate.column).cloned())
                .min_by(|left, right| Self::compare_json_values(Some(left), Some(right)))
                .unwrap_or(Value::Null)),
            AggregateFunction::Max => Ok(rows
                .iter()
                .filter_map(|row| Self::field_value(row, &aggregate.column).cloned())
                .max_by(|left, right| Self::compare_json_values(Some(left), Some(right)))
                .unwrap_or(Value::Null)),
            AggregateFunction::ArrayAgg => {
                let mut values: Vec<Value> = rows
                    .iter()
                    .filter_map(|row| Self::field_value(row, &aggregate.column).cloned())
                    .collect();
                if aggregate.distinct {
                    values
                        .sort_by(|left, right| Self::compare_json_values(Some(left), Some(right)));
                    values.dedup();
                }
                Ok(Value::Array(values))
            }
            AggregateFunction::StdDev | AggregateFunction::Variance => Err(anyhow!(
                "{:?} aggregate is not implemented in row executor yet",
                aggregate.function
            )),
        }
    }

    fn project_row(row: &Value, columns: &[String]) -> Value {
        let mut out = Map::new();
        for column in columns {
            if let Some(value) = Self::field_value(row, column) {
                out.insert(column.clone(), value.clone());
            }
        }
        Value::Object(out)
    }

    fn row_matches(row: &Value, expression: &FilterExpression) -> bool {
        match expression {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => Self::compare_value(Self::field_value(row, field), operator, value),
            FilterExpression::And(expressions) => {
                expressions.iter().all(|expr| Self::row_matches(row, expr))
            }
            FilterExpression::Or(expressions) => {
                expressions.iter().any(|expr| Self::row_matches(row, expr))
            }
            FilterExpression::Not(expression) => !Self::row_matches(row, expression),
        }
    }

    fn field_value<'a>(row: &'a Value, field: &str) -> Option<&'a Value> {
        // TD-PROTO-2 follow-up: delegate to the shared object-only dot-path
        // walker; identical semantics.
        proximadb_search_types::sql_value_filter::json_get_path(row, field.split('.'))
    }

    fn compare_value(
        field_value: Option<&Value>,
        operator: &ComparisonOperator,
        expected: &Value,
    ) -> bool {
        match operator {
            ComparisonOperator::IsNull => field_value.is_none_or(Value::is_null),
            ComparisonOperator::IsNotNull => field_value.is_some_and(|value| !value.is_null()),
            ComparisonOperator::Equals => field_value == Some(expected),
            ComparisonOperator::NotEquals => field_value != Some(expected),
            ComparisonOperator::GreaterThan => {
                Self::compare_ordering(field_value, expected, |ord| ord.is_gt())
            }
            ComparisonOperator::GreaterThanOrEqual => {
                Self::compare_ordering(field_value, expected, |ord| ord.is_ge())
            }
            ComparisonOperator::LessThan => {
                Self::compare_ordering(field_value, expected, |ord| ord.is_lt())
            }
            ComparisonOperator::LessThanOrEqual => {
                Self::compare_ordering(field_value, expected, |ord| ord.is_le())
            }
            ComparisonOperator::In => match (field_value, expected) {
                (Some(value), Value::Array(values)) => values.contains(value),
                _ => false,
            },
            ComparisonOperator::NotIn => match (field_value, expected) {
                (Some(value), Value::Array(values)) => !values.contains(value),
                _ => true,
            },
            ComparisonOperator::Contains => match (field_value, expected) {
                (Some(Value::String(text)), Value::String(needle)) => text.contains(needle),
                (Some(Value::Array(values)), needle) => values.contains(needle),
                _ => false,
            },
            ComparisonOperator::StartsWith => match (field_value, expected) {
                (Some(Value::String(text)), Value::String(prefix)) => text.starts_with(prefix),
                _ => false,
            },
            ComparisonOperator::EndsWith => match (field_value, expected) {
                (Some(Value::String(text)), Value::String(suffix)) => text.ends_with(suffix),
                _ => false,
            },
            ComparisonOperator::Between => match (field_value, expected) {
                (Some(value), Value::Array(bounds)) if bounds.len() == 2 => {
                    Self::compare_ordering(Some(value), &bounds[0], |ord| ord.is_ge())
                        && Self::compare_ordering(Some(value), &bounds[1], |ord| ord.is_le())
                }
                _ => false,
            },
            ComparisonOperator::Like => match (field_value, expected) {
                (Some(Value::String(text)), Value::String(pattern)) => {
                    Self::matches_like_pattern(text, pattern)
                }
                _ => false,
            },
        }
    }

    fn compare_ordering<F>(field_value: Option<&Value>, expected: &Value, predicate: F) -> bool
    where
        F: FnOnce(Ordering) -> bool,
    {
        let Some(field_value) = field_value else {
            return false;
        };
        let ordering = match (field_value, expected) {
            (Value::Number(left), Value::Number(right)) => left
                .as_f64()
                .zip(right.as_f64())
                .and_then(|(left, right)| left.partial_cmp(&right)),
            (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
            _ => None,
        };
        ordering.is_some_and(predicate)
    }

    fn compare_json_values(left: Option<&Value>, right: Option<&Value>) -> Ordering {
        match (left, right) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(Value::Number(left)), Some(Value::Number(right))) => left
                .as_f64()
                .zip(right.as_f64())
                .and_then(|(left, right)| left.partial_cmp(&right))
                .unwrap_or(Ordering::Equal),
            (Some(Value::String(left)), Some(Value::String(right))) => left.cmp(right),
            (Some(Value::Bool(left)), Some(Value::Bool(right))) => left.cmp(right),
            (Some(left), Some(right)) => left.to_string().cmp(&right.to_string()),
        }
    }

    fn matches_like_pattern(text: &str, pattern: &str) -> bool {
        if pattern == "%" {
            return true;
        }

        let starts_with_wildcard = pattern.starts_with('%');
        let ends_with_wildcard = pattern.ends_with('%');
        let needle = pattern.trim_matches('%');

        match (starts_with_wildcard, ends_with_wildcard) {
            (true, true) => text.contains(needle),
            (true, false) => text.ends_with(needle),
            (false, true) => text.starts_with(needle),
            (false, false) => text == needle,
        }
    }

    fn data_model_from_source(source: &str) -> DataModel {
        match source.to_lowercase().as_str() {
            "vector" | "vectors" => DataModel::Vector,
            "document" | "documents" => DataModel::Document,
            "graph" | "graphs" => DataModel::Graph,
            "timeseries" | "ts" => DataModel::TimeSeries,
            "log" | "logs" | "observability" => DataModel::Observability,
            "event" | "events" => DataModel::Event,
            _ => DataModel::Vector,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (TDD first)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_data_model::DataModel as PlanDataModel;
    use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
    use proximadb_multimodel_plan::{
        AggregateExpression, AggregateFunction, EdgePattern, JoinCondition, PlanContext,
        TraversalDirection, VectorMetric,
    };

    // ── Mock data source ─────────────────────────────────────────────────────

    struct FixedDataSource {
        vector_rows: Vec<Value>,
        document_rows: Vec<Value>,
    }

    impl PlanDataSource for FixedDataSource {
        fn scan(&self, model: DataModel, _limit: Option<usize>) -> Result<Vec<Value>> {
            Ok(match model {
                proximadb_data_model::DataModel::Vector => self.vector_rows.clone(),
                proximadb_data_model::DataModel::Document => self.document_rows.clone(),
                _ => Vec::new(),
            })
        }
    }

    // ── Mock ANN provider ────────────────────────────────────────────────────

    struct FixedAnn {
        seeds: Vec<(String, f32)>,
    }

    impl AnnSeedProvider for FixedAnn {
        fn find_seeds(&self, _query: &[f32], _k: usize) -> Result<Vec<(String, f32)>> {
            Ok(self.seeds.clone())
        }
    }

    // ── Mock graph provider ──────────────────────────────────────────────────

    struct EmptyGraph;

    impl GraphNeighbourProvider for EmptyGraph {
        fn neighbours(
            &self,
            _node_id: &str,
        ) -> Result<Vec<(String, Option<String>, TraversalDirection)>> {
            Ok(Vec::new())
        }
    }

    fn make_context(vector_rows: Vec<Value>, document_rows: Vec<Value>) -> PlanExecutionContext {
        let src = Arc::new(FixedDataSource {
            vector_rows,
            document_rows,
        });
        PlanExecutionContext::new(src)
    }

    fn plan_ctx() -> PlanContext {
        PlanContext::default()
    }

    // ── Test: empty plan returns empty result ────────────────────────────────

    #[test]
    fn test_empty_plan_returns_empty_result() {
        let plan = MultiModelPlan::new(vec![], plan_ctx());
        let mut ctx = make_context(vec![], vec![]);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();
        assert!(result.rows.is_empty());
        assert!(result.operator_stats.is_empty());
    }

    // ── Test: CrossModelJoin dispatches to MshjExecutor ─────────────────────

    #[test]
    fn test_cross_model_join_dispatches_to_mshj() {
        let vector_rows = vec![
            serde_json::json!({ "id": "v1", "embedding_ref": "e1" }),
            serde_json::json!({ "id": "v2", "embedding_ref": "e2" }),
        ];
        let doc_rows = vec![
            serde_json::json!({ "doc_id": "e1", "title": "Alpha" }),
            serde_json::json!({ "doc_id": "e3", "title": "Gamma" }), // no match
        ];

        let plan = MultiModelPlan::new(
            vec![Operator::CrossModelJoin {
                left_modality: PlanDataModel::Vector,
                right_modality: PlanDataModel::Document,
                condition: JoinCondition::On("embedding_ref".to_string(), "doc_id".to_string()),
            }],
            plan_ctx(),
        );

        let mut ctx = make_context(vector_rows, doc_rows);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();

        // Only v1/e1 match; v2 has no doc, e3 has no vector
        assert_eq!(result.rows.len(), 1, "expected exactly one joined row");
        let row = &result.rows[0];
        assert_eq!(row["id"], "v1");
        assert_eq!(row["title"], "Alpha");

        // Operator stats recorded
        assert_eq!(result.operator_stats.len(), 1);
        assert_eq!(result.operator_stats[0].rows_out, 1);
    }

    // ── Test: HybridTraverse dispatches to HybridTraverseExecutor ───────────

    #[test]
    fn test_hybrid_traverse_dispatches_to_executor() {
        let ann = Arc::new(FixedAnn {
            seeds: vec![("node_a".to_string(), 0.95), ("node_b".to_string(), 0.80)],
        });
        let graph = Arc::new(EmptyGraph);

        let edge_pattern = EdgePattern {
            edge_type: None,
            min_hops: 0,
            max_hops: Some(2),
            direction: TraversalDirection::Both,
        };

        let plan = MultiModelPlan::new(
            vec![
                Operator::VectorTopK {
                    query_vector: vec![1.0, 0.0, 0.0],
                    k: 5,
                    metric: VectorMetric::Cosine,
                    predicate: None,
                },
                Operator::HybridTraverse { edge_pattern },
            ],
            plan_ctx(),
        );

        let mut ctx = make_context(vec![], vec![]).with_ann(ann).with_graph(graph);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();

        // Seeds node_a and node_b returned; EmptyGraph adds no further nodes
        assert_eq!(result.rows.len(), 2);
        let ids: Vec<&str> = result
            .rows
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"node_a"));
        assert!(ids.contains(&"node_b"));

        // Two operators, both recorded
        assert_eq!(result.operator_stats.len(), 2);
    }

    // ── Test: HybridTraverse without provider returns an error ───────────────

    #[test]
    fn test_hybrid_traverse_without_provider_errors() {
        let edge_pattern = EdgePattern {
            edge_type: None,
            min_hops: 0,
            max_hops: Some(1),
            direction: TraversalDirection::Both,
        };

        let plan = MultiModelPlan::new(vec![Operator::HybridTraverse { edge_pattern }], plan_ctx());

        let mut ctx = make_context(vec![], vec![]); // no ANN provider
        let err = PlanExecutor::execute(&plan, &mut ctx).unwrap_err();
        assert!(err.to_string().contains("AnnSeedProvider"));
    }

    // ── Test: Limit operator slices rows ────────────────────────────────────

    #[test]
    fn test_limit_operator_slices_rows() {
        let vector_rows: Vec<Value> = (0..10)
            .map(|i| serde_json::json!({ "id": format!("v{i}") }))
            .collect();

        let plan = MultiModelPlan::new(
            vec![
                Operator::Scan {
                    source: "vector".to_string(),
                    data_model: PlanDataModel::Vector,
                    filter: None,
                    columns: None,
                },
                Operator::Limit { n: 3, offset: 2 },
            ],
            plan_ctx(),
        );

        let mut ctx = make_context(vector_rows, vec![]);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0]["id"], "v2"); // offset 2
    }

    #[test]
    fn test_filter_operator_applies_json_row_predicates() {
        let vector_rows = vec![
            serde_json::json!({
                "id": "v1",
                "tenant": "acme",
                "score": 0.91,
                "payload": { "kind": "checkpoint" }
            }),
            serde_json::json!({
                "id": "v2",
                "tenant": "acme",
                "score": 0.42,
                "payload": { "kind": "event" }
            }),
            serde_json::json!({
                "id": "v3",
                "tenant": "globex",
                "score": 0.99,
                "payload": { "kind": "checkpoint" }
            }),
        ];

        let plan = MultiModelPlan::new(
            vec![
                Operator::Scan {
                    source: "vector".to_string(),
                    data_model: PlanDataModel::Vector,
                    filter: None,
                    columns: None,
                },
                Operator::Filter {
                    expression: FilterExpression::And(vec![
                        FilterExpression::Comparison {
                            field: "tenant".to_string(),
                            operator: ComparisonOperator::Equals,
                            value: serde_json::json!("acme"),
                        },
                        FilterExpression::Comparison {
                            field: "score".to_string(),
                            operator: ComparisonOperator::GreaterThan,
                            value: serde_json::json!(0.5),
                        },
                        FilterExpression::Comparison {
                            field: "payload.kind".to_string(),
                            operator: ComparisonOperator::Equals,
                            value: serde_json::json!("checkpoint"),
                        },
                    ]),
                },
            ],
            plan_ctx(),
        );

        let mut ctx = make_context(vector_rows, vec![]);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["id"], "v1");
        assert_eq!(result.operator_stats[1].rows_in, 3);
        assert_eq!(result.operator_stats[1].rows_out, 1);
    }

    #[test]
    fn dot_path_lookup_remains_object_only() {
        let row = serde_json::json!({
            "payload": {"kind": "checkpoint"},
            "items": [{"kind": "must-not-resolve"}],
            "scalar": "value"
        });

        assert_eq!(
            PlanExecutor::field_value(&row, "payload.kind").and_then(Value::as_str),
            Some("checkpoint")
        );
        assert!(PlanExecutor::field_value(&row, "items.0.kind").is_none());
        assert!(PlanExecutor::field_value(&row, "scalar.child").is_none());
    }

    #[test]
    fn test_scan_filter_and_project_shape_rows() {
        let vector_rows = vec![
            serde_json::json!({
                "id": "v1",
                "tenant": "acme",
                "score": 0.91,
                "discard": "hidden"
            }),
            serde_json::json!({
                "id": "v2",
                "tenant": "globex",
                "score": 0.99,
                "discard": "hidden"
            }),
        ];

        let plan = MultiModelPlan::new(
            vec![
                Operator::Scan {
                    source: "vector".to_string(),
                    data_model: PlanDataModel::Vector,
                    filter: Some(FilterExpression::Comparison {
                        field: "tenant".to_string(),
                        operator: ComparisonOperator::Equals,
                        value: serde_json::json!("acme"),
                    }),
                    columns: Some(vec!["id".to_string(), "score".to_string()]),
                },
                Operator::Project {
                    columns: vec!["id".to_string()],
                },
            ],
            plan_ctx(),
        );

        let mut ctx = make_context(vector_rows, vec![]);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();

        assert_eq!(result.rows, vec![serde_json::json!({ "id": "v1" })]);
        assert_eq!(result.operator_stats[0].rows_in, 0);
        assert_eq!(result.operator_stats[0].rows_out, 1);
        assert_eq!(result.operator_stats[1].rows_in, 1);
        assert_eq!(result.operator_stats[1].rows_out, 1);
    }

    #[test]
    fn test_sort_and_topk_rank_json_rows() {
        let vector_rows = vec![
            serde_json::json!({ "id": "v1", "score": 0.20 }),
            serde_json::json!({ "id": "v2", "score": 0.95 }),
            serde_json::json!({ "id": "v3", "score": 0.70 }),
            serde_json::json!({ "id": "v4" }),
        ];

        let plan = MultiModelPlan::new(
            vec![
                Operator::Scan {
                    source: "vector".to_string(),
                    data_model: PlanDataModel::Vector,
                    filter: None,
                    columns: None,
                },
                Operator::Sort {
                    column: "score".to_string(),
                    ascending: true,
                    limit: Some(3),
                },
                Operator::TopK {
                    k: 2,
                    sort_column: "score".to_string(),
                },
            ],
            plan_ctx(),
        );

        let mut ctx = make_context(vector_rows, vec![]);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();

        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0]["id"], "v2");
        assert_eq!(result.rows[1]["id"], "v3");
        assert_eq!(result.operator_stats[1].rows_out, 3);
        assert_eq!(result.operator_stats[2].rows_out, 2);
    }

    #[test]
    fn test_aggregate_groups_and_computes_basic_functions() {
        let vector_rows = vec![
            serde_json::json!({ "tenant": "acme", "score": 0.20, "kind": "event" }),
            serde_json::json!({ "tenant": "acme", "score": 0.80, "kind": "event" }),
            serde_json::json!({ "tenant": "globex", "score": 0.50, "kind": "checkpoint" }),
        ];

        let plan = MultiModelPlan::new(
            vec![
                Operator::Scan {
                    source: "vector".to_string(),
                    data_model: PlanDataModel::Vector,
                    filter: None,
                    columns: None,
                },
                Operator::Aggregate {
                    group_by: vec!["tenant".to_string()],
                    aggregates: vec![
                        AggregateExpression {
                            function: AggregateFunction::Count,
                            column: "score".to_string(),
                            alias: Some("row_count".to_string()),
                            distinct: false,
                        },
                        AggregateExpression {
                            function: AggregateFunction::Avg,
                            column: "score".to_string(),
                            alias: Some("avg_score".to_string()),
                            distinct: false,
                        },
                        AggregateExpression {
                            function: AggregateFunction::ArrayAgg,
                            column: "kind".to_string(),
                            alias: Some("kinds".to_string()),
                            distinct: true,
                        },
                    ],
                    alias: None,
                },
                Operator::Sort {
                    column: "tenant".to_string(),
                    ascending: true,
                    limit: None,
                },
            ],
            plan_ctx(),
        );

        let mut ctx = make_context(vector_rows, vec![]);
        let result = PlanExecutor::execute(&plan, &mut ctx).unwrap();

        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0]["tenant"], "acme");
        assert_eq!(result.rows[0]["row_count"], 2);
        assert_eq!(result.rows[0]["avg_score"], 0.5);
        assert_eq!(result.rows[0]["kinds"], serde_json::json!(["event"]));
        assert_eq!(result.rows[1]["tenant"], "globex");
        assert_eq!(result.rows[1]["row_count"], 1);
        assert_eq!(result.operator_stats[1].rows_in, 3);
        assert_eq!(result.operator_stats[1].rows_out, 2);
    }
}
