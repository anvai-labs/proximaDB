/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Execution Planner for Multi-Model Queries
//!
//! Analyzes Query AST and generates optimized ExecutionPlan by selecting
//! appropriate execution strategies (VectorOnly, GraphOnly, Hybrid).

use super::{
    ExecutionOperation, ExecutionPlan, ExecutionStrategy, FusionStrategy, ProjectionTransform,
    SeedingStrategy,
};
use crate::query::ast::{Expr, Query};
use crate::services::operations::vectors::VectorOperationsService;
use crate::storage::cache::orchestrator::CrossCacheOrchestrator;
use anyhow::{Result, anyhow};
use proximadb_graph_query::service::GraphExecutionService;
use std::sync::Arc;

fn default_hybrid_fusion_weights() -> Vec<f64> {
    crate::core::config::HybridRuntimeConfig::default()
        .fusion_weights
        .unwrap_or_else(|| vec![0.6, 0.4])
}

/// Execution planner that transforms AST into ExecutionPlan
pub struct ExecutionPlanner {
    #[allow(dead_code)]
    vector_service: Arc<VectorOperationsService>,
    #[allow(dead_code)]
    graph_service: Arc<dyn GraphExecutionService>,
    cost_model: CostModel,
    params: Option<Vec<crate::proto::proximadb_v1::SqlValue>>,
    seeding_strategy: SeedingStrategy,
    fusion_weights: Option<Vec<f64>>,
    #[allow(dead_code)]
    cache_orchestrator: Option<Arc<CrossCacheOrchestrator>>,
}

impl ExecutionPlanner {
    /// Create new execution planner with service integrations
    pub fn new(
        vector_service: Arc<VectorOperationsService>,
        graph_service: Arc<dyn GraphExecutionService>,
    ) -> Self {
        Self {
            vector_service,
            graph_service,
            cost_model: CostModel::new(),
            params: None,
            seeding_strategy: SeedingStrategy::Average,
            fusion_weights: Some(default_hybrid_fusion_weights()),
            cache_orchestrator: None,
        }
    }

    /// Create with cache orchestrator for intelligent caching
    pub fn with_cache(
        vector_service: Arc<VectorOperationsService>,
        graph_service: Arc<dyn GraphExecutionService>,
        cache_orchestrator: Arc<CrossCacheOrchestrator>,
    ) -> Self {
        Self {
            vector_service,
            graph_service,
            cost_model: CostModel::new(),
            params: None,
            seeding_strategy: SeedingStrategy::Average,
            fusion_weights: Some(default_hybrid_fusion_weights()),
            cache_orchestrator: Some(cache_orchestrator),
        }
    }

    /// Create with bound parameters
    pub fn with_params(
        vector_service: Arc<VectorOperationsService>,
        graph_service: Arc<dyn GraphExecutionService>,
        params: Option<Vec<crate::proto::proximadb_v1::SqlValue>>,
    ) -> Self {
        let mut p = Self::new(vector_service, graph_service);
        p.params = params;
        p
    }

    pub fn set_seeding_strategy(&mut self, strategy: SeedingStrategy) {
        self.seeding_strategy = strategy;
    }

    pub fn set_fusion_weights(&mut self, weights: Option<Vec<f64>>) {
        self.fusion_weights = weights;
    }

    /// Create execution plan from Query AST
    pub fn create_plan(&self, query: &Query) -> Result<ExecutionPlan> {
        match query {
            Query::Select(select) => self.plan_select(select),
            Query::Set {
                left, right, all, ..
            } => self.plan_union(left, right, *all),
            _ => Err(anyhow!("Unsupported query type for execution planning")),
        }
    }

    fn plan_select(&self, select: &crate::query::ast::Select) -> Result<ExecutionPlan> {
        // 1. Determine execution strategy
        let strategy = self.detect_strategy(select);

        // 2. Generate operations
        let mut operations = Vec::new();

        // Add model-specific retrieval operations
        match strategy {
            ExecutionStrategy::VectorOnly => {
                operations.push(self.plan_vector_search(select)?);
            }
            ExecutionStrategy::GraphOnly => {
                operations.push(self.plan_graph_traversal(select)?);
            }
            ExecutionStrategy::Hybrid => {
                // For hybrid, we might run both or sequential depending on seeding
                operations.extend(self.plan_hybrid_operations(select)?);
            }
            ExecutionStrategy::Relational => {
                return Err(anyhow!("Relational execution not yet supported"));
            }
        }

        // Add common post-processing operations
        if !select.projection.is_empty() {
            operations.push(self.plan_projection(select)?);
        }

        if !select.group_by.is_empty() {
            operations.push(self.plan_aggregation(select)?);
        }

        // 3. Estimate cost
        let estimated_cost = self.cost_model.estimate(&operations);

        Ok(ExecutionPlan::runtime(
            strategy,
            operations,
            estimated_cost,
            self.identify_optimizations(select),
            self.generate_hints(select),
            self.seeding_strategy.clone(),
            select.limit.map(|l| l as usize),
            select.offset.map(|o| o as usize),
        ))
    }

    fn detect_strategy(&self, select: &crate::query::ast::Select) -> ExecutionStrategy {
        // TableRef no longer carries an explicit model discriminator. Until that
        // annotation is reintroduced through lowering, use the canonical
        // expression capabilities to detect modality-specific execution.
        let has_vector = select
            .projection
            .iter()
            .any(|p| matches!(p.expr, Expr::SksSimilar { .. }));
        let has_graph = select
            .projection
            .iter()
            .any(|p| matches!(p.expr, Expr::SksFollow { .. } | Expr::SksAssemble { .. }));

        if has_vector && has_graph {
            ExecutionStrategy::Hybrid
        } else if has_graph {
            ExecutionStrategy::GraphOnly
        } else if has_vector {
            ExecutionStrategy::VectorOnly
        } else {
            ExecutionStrategy::Relational
        }
    }

    fn plan_vector_search(&self, select: &crate::query::ast::Select) -> Result<ExecutionOperation> {
        // Extract collection ID
        let collection_id = select
            .from
            .first()
            .and_then(|t| t.name.clone())
            .ok_or_else(|| anyhow!("Vector search requires a collection"))?;

        // Extract query vector and parameters from SksSimilar expr
        let mut query_vector = None;
        let top_k = select.limit.unwrap_or(10) as usize;
        let mut distance_metric = "cosine".to_string();

        for p in &select.projection {
            if let Expr::SksSimilar { query, metric, .. } = &p.expr {
                if let Expr::Literal(crate::query::ast::Literal::String(vec_str)) = query.as_ref() {
                    query_vector = Some(self.parse_vector_literal(vec_str)?);
                }
                if let Some(m) = metric {
                    distance_metric = m.clone();
                }
            }
        }

        Ok(ExecutionOperation::VectorQuery {
            collection_id,
            query_vector,
            filters: None, // AST Expr → FilterExpression conversion deferred (TD-048)
            top_k,
            distance_metric,
        })
    }

    fn plan_graph_traversal(
        &self,
        select: &crate::query::ast::Select,
    ) -> Result<ExecutionOperation> {
        let graph_id = select
            .from
            .first()
            .and_then(|t| t.name.clone())
            .ok_or_else(|| anyhow!("Graph traversal requires a graph name"))?;

        Ok(ExecutionOperation::GraphTraversal {
            graph_id,
            start_nodes: vec![], // To be resolved from filters or parameters
            edge_types: vec![],
            max_depth: 3,
            filters: None, // AST Expr → FilterExpression conversion deferred (TD-048)
            vector_target_collection: None,
        })
    }

    fn plan_hybrid_operations(
        &self,
        select: &crate::query::ast::Select,
    ) -> Result<Vec<ExecutionOperation>> {
        Ok(vec![
            self.plan_vector_search(select)?,
            self.plan_graph_traversal(select)?,
            ExecutionOperation::Fusion {
                strategy: FusionStrategy::ReciprocalRankFusion { k: 60.0 },
                weights: self
                    .fusion_weights
                    .clone()
                    .unwrap_or_else(default_hybrid_fusion_weights),
            },
        ])
    }

    fn plan_projection(&self, select: &crate::query::ast::Select) -> Result<ExecutionOperation> {
        let mut columns = Vec::new();
        let mut transformations = Vec::new();

        for p in &select.projection {
            let col_name = p
                .alias
                .clone()
                .unwrap_or_else(|| format!("col_{}", columns.len()));
            columns.push(col_name);

            match &p.expr {
                Expr::Identifier(name) => {
                    transformations.push(ProjectionTransform::ExtractMetadata {
                        field: name.clone(),
                    });
                }
                Expr::SksSimilar { .. } => {
                    transformations.push(ProjectionTransform::SimilarityScore);
                }
                _ => {}
            }
        }

        Ok(ExecutionOperation::Project {
            columns,
            transformations,
        })
    }

    fn plan_aggregation(&self, select: &crate::query::ast::Select) -> Result<ExecutionOperation> {
        Ok(ExecutionOperation::Aggregate {
            group_keys: select.group_by.iter().map(|e| format!("{:?}", e)).collect(),
            aggs: vec![], // To be extracted from projection
            having: None, // AST Expr → FilterExpression conversion deferred (TD-048)
        })
    }

    fn plan_union(&self, _left: &Query, _right: &Query, all: bool) -> Result<ExecutionPlan> {
        Ok(ExecutionPlan::runtime(
            ExecutionStrategy::Relational,
            vec![ExecutionOperation::Union { all }],
            100.0,
            vec![],
            vec![],
            SeedingStrategy::None,
            None,
            None,
        ))
    }

    fn parse_vector_literal(&self, s: &str) -> Result<Vec<f32>> {
        // Basic parser for "[0.1, 0.2, ...]" — non-finite components
        // reject (the ONE filtered policy; an inf/NaN here reaches the
        // distance kernels).
        let trimmed = s.trim_matches(|c| c == '[' || c == ']' || c == ' ');
        trimmed
            .split(',')
            .map(|v| {
                let f = v.trim().parse::<f32>().map_err(|e| anyhow!(e))?;
                if !f.is_finite() {
                    return Err(anyhow!("query vector components must be finite"));
                }
                Ok(f)
            })
            .collect()
    }

    fn identify_optimizations(&self, _select: &crate::query::ast::Select) -> Vec<String> {
        vec!["HashMap Metadata Filtering (O(1))".to_string()]
    }

    fn generate_hints(&self, _select: &crate::query::ast::Select) -> Vec<String> {
        vec!["Use index-backed retrieval where possible".to_string()]
    }
}

/// Simple cost model for execution planning
struct CostModel;

impl CostModel {
    fn new() -> Self {
        Self
    }

    fn estimate(&self, operations: &[ExecutionOperation]) -> f64 {
        operations.iter().map(|op| self.cost_for_op(op)).sum()
    }

    fn cost_for_op(&self, op: &ExecutionOperation) -> f64 {
        match op {
            ExecutionOperation::VectorQuery { top_k, .. } => *top_k as f64 * 0.1,
            ExecutionOperation::GraphTraversal { max_depth, .. } => (*max_depth as f64).powi(2),
            ExecutionOperation::Project { columns, .. } => columns.len() as f64 * 0.01,
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hybrid_fusion_weights_follow_runtime_config() {
        assert_eq!(default_hybrid_fusion_weights(), vec![0.6, 0.4]);
    }
}
