//! AQL source for the agent-memory read surface (TD-100, P1).
//!
//! This source does NOT introduce a new storage path. Per the Convergence
//! Gate (CLAUDE.md) it converges on existing surfaces, citing
//! `ADR-022-agent-memory-layer` + `AGENT_MEMORY_LAYER_HLD_2026_05_30`:
//!
//! * scoped semantic retrieval → `VectorOperationsService::unified_search_v1`
//!   with a pushed-down `FilterExpression` (tenant / session scope);
//! * lexical retrieval → `ProductionHybridBackend::bm25_search` (best-effort);
//! * fusion → `ResultFuser` (Reciprocal Rank Fusion), the same fuser the
//!   cross-model runtime uses;
//! * typed-memory scoping → `ProximaRecord.memory_type` (Memanto, TD-055),
//!   applied as a post-filter over the scoped candidate set;
//! * audit → the existing `AuditFrame` / `AuditContext` machinery.
//!
//! Scope safety: tenant/session are pushed into the semantic leg. The lexical
//! leg carries no metadata, so the fused set is restricted to the scoped
//! semantic candidate ids before returning — BM25 can only reorder memories
//! that scoped retrieval already surfaced. Lexical-only recall within scope
//! requires BM25 metadata scoping and is tracked as a follow-up (TD-102).
//!
//! The query embedding is supplied by the caller as an `AqlValue::Vector`
//! (mem0 and other agent callers already compute embeddings client-side);
//! embedding text inline is a follow-up, consistent with `VectorAqlSource`.

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use proximadb_data_model::{DataModel as PqDataModel, MemoryType};
use proximadb_kernel::error::{ProximaDBError, QueryError};
use proximadb_query::fusion::{ResultFuser, SubQueryResult};
use proximadb_query::results::UnifiedRecord;
use proximadb_query_fusion::FusionStrategy;

use crate::core::search::{ComparisonOperator, FilterExpression};
use crate::network::rest::canonical::rank::HybridSearchBackend;
use crate::network::rest::canonical::rank_backend::ProductionHybridBackend;
use crate::query::aql::sources::vector::VectorAqlSource;
use crate::query::aql::{
    AqlFrom, AqlPredicate, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditFrame,
    AuditOp, DataModel, Result,
};
use crate::services::VectorOperationsService;

/// Retrieve this multiple of `top_k` per leg so fusion/post-filter has headroom.
const RECALL_POOL_MULTIPLIER: usize = 5;
const DEFAULT_TOP_K: u32 = 10;
/// Reciprocal Rank Fusion constant (standard default).
const RRF_K: u32 = 60;

/// Semantic-leg recall pool: `top_k * RECALL_POOL_MULTIPLIER` (headroom for
/// fusion / post-filter), at least `top_k` and at least 1.
/// Clamped to the server cap (`max_top_k()`): the pool is an internal
/// oversample, so an at-cap `top_k` must not trip the cap that
/// `unified_search_v1` enforces (TD-SECSCAN-1 §5). `top_k` itself is checked
/// against the cap before this is computed.
fn recall_pool_size(top_k: u32) -> usize {
    (top_k as usize)
        .saturating_mul(RECALL_POOL_MULTIPLIER)
        .max(top_k as usize)
        .min(proximadb_search_types::top_k_limit::max_top_k())
        .max(1)
}

/// Scope + query parameters extracted from the AQL predicate tree.
#[derive(Default, Debug, PartialEq)]
struct MemoryScope {
    collection: String,
    tenant_id: Option<String>,
    session_id: Option<String>,
    memory_type: Option<MemoryType>,
    query_text: Option<String>,
    query_vector: Vec<f32>,
    top_k: u32,
}

/// Agent-memory read source. Composes scoped semantic + lexical retrieval,
/// RRF fusion, typed-memory scoping, and audit-frame emission.
pub struct MemoryAqlSource {
    vector_ops: Arc<VectorOperationsService>,
    lexical: Option<Arc<ProductionHybridBackend>>,
}

impl MemoryAqlSource {
    pub fn new(vector_ops: Arc<VectorOperationsService>) -> Self {
        Self {
            vector_ops,
            lexical: None,
        }
    }

    /// Wire the optional lexical (BM25) leg. When absent, the source runs
    /// semantic-only (still scoped, fused-trivially, audited).
    pub fn with_lexical_backend(mut self, backend: Arc<ProductionHybridBackend>) -> Self {
        self.lexical = Some(backend);
        self
    }

    /// Walk the predicate tree collecting scope + query parameters.
    fn extract_scope(query: &AqlQuery) -> MemoryScope {
        let mut scope = MemoryScope {
            collection: "default".to_string(),
            top_k: DEFAULT_TOP_K,
            ..Default::default()
        };

        if let AqlFrom::Source { name, .. } = &query.from {
            scope.collection = name.clone();
        }

        if let Some(pred) = &query.where_clause.predicate {
            Self::walk_predicate(pred, &mut scope);
        }

        scope
    }

    fn walk_predicate(pred: &AqlPredicate, scope: &mut MemoryScope) {
        match pred {
            AqlPredicate::Equals { field, value } => match (field.as_str(), value) {
                ("tenant_id", AqlValue::String(s)) => scope.tenant_id = Some(s.clone()),
                ("session_id" | "props.session_id", AqlValue::String(s)) => {
                    scope.session_id = Some(s.clone())
                }
                ("embedding" | "vector" | "query_vector", AqlValue::Vector(v)) => {
                    scope.query_vector = v.clone()
                }
                _ => {}
            },
            AqlPredicate::SemanticMatch { query, top_k, .. } => {
                if !query.is_empty() {
                    scope.query_text = Some(query.clone());
                }
                if *top_k > 0 {
                    scope.top_k = *top_k;
                }
            }
            AqlPredicate::TypeMatch { memory_type } => scope.memory_type = Some(*memory_type),
            AqlPredicate::And { lhs, rhs } | AqlPredicate::Or { lhs, rhs } => {
                Self::walk_predicate(lhs, scope);
                Self::walk_predicate(rhs, scope);
            }
            AqlPredicate::Not { inner } => Self::walk_predicate(inner, scope),
            _ => {}
        }
    }

    /// Build the tenant/session scope filter pushed into the semantic leg.
    /// `memory_type` is NOT pushed (no `FilterExpression` operator for it yet);
    /// it is enforced as a post-filter over the scoped candidate set.
    fn build_scope_filter(scope: &MemoryScope) -> (Option<FilterExpression>, Vec<String>) {
        let mut parts = Vec::new();
        let mut pushed = Vec::new();
        // The tenant predicate is mandatory for isolation; only a non-empty
        // tenant produces it. `execute` fail-closes before calling this, so a
        // missing/blank tenant never reaches a real search — this is the
        // belt-and-suspenders so the builder never emits `tenant_id = ""`.
        if let Some(tenant) = &scope.tenant_id
            && !tenant.trim().is_empty()
        {
            parts.push(FilterExpression::Comparison {
                field: "tenant_id".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::String(tenant.clone()),
            });
            pushed.push(format!("tenant_id = {tenant}"));
        }
        if let Some(session) = &scope.session_id {
            // Flat metadata key from props flattening (`proxima_tree_to_value_map`):
            // a top-level prop `session_id` is the key "session_id", not
            // "props.session_id" (TD-103). The dotted form silently matched nothing.
            parts.push(FilterExpression::Comparison {
                field: "session_id".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::String(session.clone()),
            });
            pushed.push(format!("session_id = {session}"));
        }
        match parts.len() {
            0 => (None, pushed),
            1 => (parts.into_iter().next(), pushed),
            _ => (Some(FilterExpression::And(parts)), pushed),
        }
    }

    /// Canonical lowercase string for a `MemoryType` (matches serde rename).
    fn memory_type_str(m: MemoryType) -> String {
        serde_json::to_value(m)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| format!("{m:?}").to_lowercase())
    }

    /// Pure assembly step (no I/O): fuse the legs, restrict to the scoped
    /// semantic candidate set, apply the `memory_type` post-filter, and trim to
    /// `top_k`. Extracted so it can be unit-tested deterministically.
    fn assemble_rows(
        row_by_id: &HashMap<String, HashMap<String, AqlValue>>,
        semantic_ids: &HashSet<String>,
        sub_results: Vec<SubQueryResult>,
        scope: &MemoryScope,
    ) -> Result<(Vec<HashMap<String, AqlValue>>, Vec<String>)> {
        let fuser = ResultFuser::new(FusionStrategy::ReciprocalRankFusion { k: RRF_K });
        let fused = fuser
            .fuse(
                sub_results,
                &FusionStrategy::ReciprocalRankFusion { k: RRF_K },
            )
            .map_err(|e| ProximaDBError::Query(QueryError::VectorSearch(e.to_string())))?;

        let mut filters_post = Vec::new();
        let memory_type_str = scope.memory_type.map(|m| {
            let s = Self::memory_type_str(m);
            filters_post.push(format!("memory_type = {s}"));
            s
        });

        let mut rows = Vec::new();
        for rec in fused.records {
            // Scope safety: only emit memories scoped retrieval surfaced.
            if !semantic_ids.contains(&rec.id) {
                continue;
            }
            let Some(mut row) = row_by_id.get(&rec.id).cloned() else {
                continue;
            };
            // Typed-memory post-filter (Memanto).
            if let Some(expected) = &memory_type_str {
                let matches = row
                    .get("memory_type")
                    .or_else(|| row.get("type"))
                    .is_some_and(|v| matches!(v, AqlValue::String(s) if s == expected));
                if !matches {
                    continue;
                }
            }
            // Surface the fused score.
            if let Some(score) = rec.score {
                row.insert("score".to_string(), AqlValue::Float(score));
            }
            rows.push(row);
            if rows.len() >= scope.top_k as usize {
                break;
            }
        }
        Ok((rows, filters_post))
    }
}

#[async_trait]
impl AqlSource for MemoryAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Vector
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let scope = Self::extract_scope(query);
        // Fail-closed: a memory read MUST be tenant-scoped. Refuse rather than
        // run an unscoped search that could surface another tenant's memories.
        // The tenant predicate is the load-bearing isolation control for the
        // shared-collection model, so it must always be present and non-empty.
        match &scope.tenant_id {
            Some(t) if !t.trim().is_empty() => {}
            _ => {
                return Err(ProximaDBError::Query(QueryError::VectorSearch(
                    "agent-memory search requires a non-empty tenant scope".to_string(),
                )));
            }
        }
        // The client's top_k is checked against the shared cap BEFORE the
        // internal oversample, so over-cap is a 400 and at-cap is served.
        proximadb_search_types::top_k_limit::check_top_k(u64::from(scope.top_k))
            .map_err(|e| ProximaDBError::InvalidInput(e.to_string()))?;
        let recall_k = recall_pool_size(scope.top_k);
        let (scope_filter, filters_pushed) = Self::build_scope_filter(&scope);

        // ---- Semantic leg: scoped vector search (scope pushed down) ----------
        let start = Instant::now();
        let search_results = self
            .vector_ops
            .unified_search_v1_for_tenant(
                &scope.collection,
                ctx.request_tenant.as_deref(),
                scope.query_vector.clone(),
                recall_k,
                scope_filter,
                None,
                None,
                None,
                proximadb_tenant::AuthClass::Anonymous,
            )
            .await
            .map_err(VectorAqlSource::vector_search_error)?;
        let semantic_wall_us = start.elapsed().as_micros() as u64;

        // Build (id -> row) plus the ordered semantic candidate set. Rows carry
        // metadata so memory_type / projection work downstream.
        let mut row_by_id: HashMap<String, HashMap<String, AqlValue>> = HashMap::new();
        let mut semantic_ids: HashSet<String> = HashSet::new();
        let mut semantic_records: Vec<UnifiedRecord> = Vec::new();
        if let Some(batch) = search_results.first() {
            for res in &batch.results {
                let mut row = HashMap::new();
                row.insert("id".to_string(), AqlValue::String(res.id.clone()));
                row.insert("score".to_string(), AqlValue::Float(res.score));
                for (k, v) in &res.metadata {
                    if let Some(val) = &v.value {
                        row.insert(k.clone(), VectorAqlSource::sql_data_to_aql(val));
                    }
                }
                semantic_records.push(UnifiedRecord {
                    id: res.id.clone(),
                    source_model: PqDataModel::Vector,
                    data: serde_json::Value::Null,
                    score: Some(res.score),
                    metadata: HashMap::new(),
                });
                semantic_ids.insert(res.id.clone());
                row_by_id.insert(res.id.clone(), row);
            }
        }
        let semantic_scanned = semantic_records.len() as u64;

        ctx.push_frame(AuditFrame {
            frame_id: 0,
            source: self.model(),
            op: AuditOp::VectorSearch {
                collection: scope.collection.clone(),
                top_k: scope.top_k,
                metric: "Cosine".to_string(),
            },
            filters_pushed: filters_pushed.clone(),
            filters_post: Vec::new(),
            records_scanned: semantic_scanned,
            records_returned: semantic_scanned,
            wall_time_us: semantic_wall_us,
            error: None,
            redaction_count: 0,
        });

        // ---- Lexical leg: best-effort BM25 (reorders scoped candidates) ------
        let mut sub_results = vec![SubQueryResult {
            source_model: PqDataModel::Vector,
            records: semantic_records,
            total_count: Some(semantic_scanned),
            execution_time_us: semantic_wall_us,
            records_scanned: semantic_scanned,
            records_returned: semantic_scanned,
        }];

        if let (Some(backend), Some(text)) = (&self.lexical, &scope.query_text) {
            let lex_start = Instant::now();
            match backend.bm25_search(&scope.collection, text).await {
                Ok(hits) => {
                    let lexical_records: Vec<UnifiedRecord> = hits
                        .into_iter()
                        .map(|hit| UnifiedRecord {
                            id: hit.doc_id,
                            source_model: PqDataModel::Document,
                            data: serde_json::Value::Null,
                            score: Some(hit.score),
                            metadata: HashMap::new(),
                        })
                        .collect();
                    let lex_count = lexical_records.len() as u64;
                    let lex_wall = lex_start.elapsed().as_micros() as u64;
                    sub_results.push(SubQueryResult {
                        source_model: PqDataModel::Document,
                        records: lexical_records,
                        total_count: Some(lex_count),
                        execution_time_us: lex_wall,
                        records_scanned: lex_count,
                        records_returned: lex_count,
                    });
                    ctx.push_frame(AuditFrame {
                        frame_id: 0,
                        source: DataModel::Document,
                        op: AuditOp::Scan {
                            source: format!("bm25:{}", scope.collection),
                        },
                        filters_pushed: Vec::new(),
                        filters_post: Vec::new(),
                        records_scanned: lex_count,
                        records_returned: lex_count,
                        wall_time_us: lex_wall,
                        error: None,
                        redaction_count: 0,
                    });
                }
                // BM25 is best-effort: a missing index degrades to semantic-only.
                Err(e) => tracing::debug!("memory lexical leg skipped: {e}"),
            }
        }

        // ---- Fuse (RRF), restrict to scoped set, type post-filter ------------
        let (rows, filters_post) =
            Self::assemble_rows(&row_by_id, &semantic_ids, sub_results, &scope)?;
        let returned = rows.len() as u64;

        let frame_id = ctx.push_frame(AuditFrame {
            frame_id: 0,
            source: self.model(),
            op: AuditOp::VectorSearch {
                collection: scope.collection.clone(),
                top_k: scope.top_k,
                metric: "rrf".to_string(),
            },
            filters_pushed,
            filters_post,
            records_scanned: semantic_scanned,
            records_returned: returned,
            wall_time_us: start.elapsed().as_micros() as u64,
            error: None,
            redaction_count: 0,
        });

        Ok(AqlResult { rows, frame_id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-SECSCAN-1 §5 follow-up: the 5x recall oversample feeds
    /// `unified_search_v1`, which enforces the server cap. An at-cap `top_k`
    /// must yield a pool within the cap (clamped), never one that trips it.
    #[test]
    fn recall_pool_stays_within_the_top_k_cap() {
        crate::services::top_k_env_test_support::with_max_top_k_override("10000", || {
            let cap = proximadb_search_types::top_k_limit::max_top_k();
            assert_eq!(recall_pool_size(10_000), cap);
            assert_eq!(recall_pool_size(u32::MAX), cap);
            assert_eq!(recall_pool_size(4), 4 * RECALL_POOL_MULTIPLIER);
            assert_eq!(recall_pool_size(0), 1);
        });
    }

    fn vector_pred(v: Vec<f32>) -> AqlPredicate {
        AqlPredicate::Equals {
            field: "embedding".to_string(),
            value: AqlValue::Vector(v),
        }
    }

    fn and(lhs: AqlPredicate, rhs: AqlPredicate) -> AqlPredicate {
        AqlPredicate::And {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    fn query_with(pred: AqlPredicate, collection: &str) -> AqlQuery {
        AqlQuery {
            find: crate::query::aql::AqlFind {
                projections: vec![crate::query::aql::AqlProjection {
                    field: "*".to_string(),
                    alias: None,
                }],
            },
            from: AqlFrom::Source {
                name: collection.to_string(),
                alias: None,
            },
            where_clause: crate::query::aql::AqlWhere {
                predicate: Some(pred),
            },
        }
    }

    #[test]
    fn extract_scope_pulls_tenant_session_type_and_vector() {
        let pred = and(
            and(
                AqlPredicate::Equals {
                    field: "tenant_id".to_string(),
                    value: AqlValue::String("acme".to_string()),
                },
                AqlPredicate::Equals {
                    field: "props.session_id".to_string(),
                    value: AqlValue::String("sess-1".to_string()),
                },
            ),
            and(
                AqlPredicate::TypeMatch {
                    memory_type: MemoryType::Fact,
                },
                vector_pred(vec![0.1, 0.2, 0.3]),
            ),
        );
        let scope = MemoryAqlSource::extract_scope(&query_with(pred, "mem"));
        assert_eq!(scope.collection, "mem");
        assert_eq!(scope.tenant_id.as_deref(), Some("acme"));
        assert_eq!(scope.session_id.as_deref(), Some("sess-1"));
        assert_eq!(scope.memory_type, Some(MemoryType::Fact));
        assert_eq!(scope.query_vector, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn build_scope_filter_ands_tenant_and_session() {
        let scope = MemoryScope {
            tenant_id: Some("acme".to_string()),
            session_id: Some("sess-1".to_string()),
            ..Default::default()
        };
        let (filter, pushed) = MemoryAqlSource::build_scope_filter(&scope);
        assert_eq!(pushed.len(), 2);
        match filter {
            Some(FilterExpression::And(parts)) => {
                assert_eq!(parts.len(), 2);
                // TD-103: fields must be the FLAT metadata keys (props flattened
                // verbatim), not the dotted `props.session_id` which silently
                // matched nothing.
                let fields: Vec<&str> = parts
                    .iter()
                    .filter_map(|p| match p {
                        FilterExpression::Comparison { field, .. } => Some(field.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(fields.contains(&"tenant_id"), "fields: {fields:?}");
                assert!(fields.contains(&"session_id"), "fields: {fields:?}");
                assert!(
                    !fields.iter().any(|f| f.contains('.')),
                    "no dotted field names: {fields:?}"
                );
            }
            other => panic!("expected And of 2, got {other:?}"),
        }
    }

    #[test]
    fn build_scope_filter_none_when_unscoped() {
        let (filter, pushed) = MemoryAqlSource::build_scope_filter(&MemoryScope::default());
        assert!(filter.is_none());
        assert!(pushed.is_empty());
    }

    #[test]
    fn build_scope_filter_skips_blank_tenant() {
        // A present-but-blank tenant must NOT produce a `tenant_id = ""`
        // predicate (which would match nothing / leak). `execute` fail-closes
        // on this case; the builder is the belt-and-suspenders.
        let scope = MemoryScope {
            tenant_id: Some("   ".to_string()),
            ..Default::default()
        };
        let (filter, pushed) = MemoryAqlSource::build_scope_filter(&scope);
        assert!(filter.is_none(), "blank tenant must not yield a predicate");
        assert!(pushed.is_empty());
    }

    #[test]
    fn scope_filter_always_carries_tenant_for_valid_scope() {
        // Tenant-only scope (no session) must still emit the tenant predicate —
        // tenant isolation is never silently dropped.
        let scope = MemoryScope {
            tenant_id: Some("acme".to_string()),
            ..Default::default()
        };
        let (filter, pushed) = MemoryAqlSource::build_scope_filter(&scope);
        assert_eq!(pushed, vec!["tenant_id = acme".to_string()]);
        match filter {
            Some(FilterExpression::Comparison { field, value, .. }) => {
                assert_eq!(field, "tenant_id");
                assert_eq!(value, serde_json::Value::String("acme".to_string()));
            }
            other => panic!("expected single tenant Comparison, got {other:?}"),
        }
    }

    // ── assembly: scope restriction + type post-filter + top_k ───────────────

    fn row(id: &str, mtype: Option<&str>) -> HashMap<String, AqlValue> {
        let mut r = HashMap::new();
        r.insert("id".to_string(), AqlValue::String(id.to_string()));
        if let Some(t) = mtype {
            r.insert("memory_type".to_string(), AqlValue::String(t.to_string()));
        }
        r
    }

    fn urec(id: &str, score: f64, model: PqDataModel) -> UnifiedRecord {
        UnifiedRecord {
            id: id.to_string(),
            source_model: model,
            data: serde_json::Value::Null,
            score: Some(score),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn assemble_restricts_to_scoped_set_and_filters_type() {
        // Scoped semantic set: m1(fact), m2(decision), m3(fact).
        let mut row_by_id = HashMap::new();
        row_by_id.insert("m1".to_string(), row("m1", Some("fact")));
        row_by_id.insert("m2".to_string(), row("m2", Some("decision")));
        row_by_id.insert("m3".to_string(), row("m3", Some("fact")));
        let semantic_ids: HashSet<String> =
            ["m1", "m2", "m3"].iter().map(|s| s.to_string()).collect();

        let semantic = SubQueryResult {
            source_model: PqDataModel::Vector,
            records: vec![
                urec("m1", 0.9, PqDataModel::Vector),
                urec("m2", 0.8, PqDataModel::Vector),
                urec("m3", 0.7, PqDataModel::Vector),
            ],
            total_count: Some(3),
            execution_time_us: 0,
            records_scanned: 3,
            records_returned: 3,
        };
        // Lexical leg references an out-of-scope id (mX) which must be dropped.
        let lexical = SubQueryResult {
            source_model: PqDataModel::Document,
            records: vec![
                urec("m3", 5.0, PqDataModel::Document),
                urec("mX", 4.0, PqDataModel::Document),
            ],
            total_count: Some(2),
            execution_time_us: 0,
            records_scanned: 2,
            records_returned: 2,
        };

        let scope = MemoryScope {
            memory_type: Some(MemoryType::Fact),
            top_k: 10,
            ..Default::default()
        };

        let (rows, filters_post) = MemoryAqlSource::assemble_rows(
            &row_by_id,
            &semantic_ids,
            vec![semantic, lexical],
            &scope,
        )
        .expect("assemble");

        let ids: HashSet<String> = rows
            .iter()
            .filter_map(|r| match r.get("id") {
                Some(AqlValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        // Only fact-typed, in-scope memories survive; decision(m2) + out-of-scope(mX) dropped.
        assert_eq!(ids, ["m1", "m3"].iter().map(|s| s.to_string()).collect());
        assert_eq!(filters_post, vec!["memory_type = fact".to_string()]);
    }

    #[test]
    fn assemble_respects_top_k() {
        let mut row_by_id = HashMap::new();
        let mut ids = HashSet::new();
        let mut recs = Vec::new();
        for i in 0..5 {
            let id = format!("m{i}");
            row_by_id.insert(id.clone(), row(&id, None));
            ids.insert(id.clone());
            recs.push(urec(&id, 1.0 - i as f64 * 0.1, PqDataModel::Vector));
        }
        let semantic = SubQueryResult {
            source_model: PqDataModel::Vector,
            records: recs,
            total_count: Some(5),
            execution_time_us: 0,
            records_scanned: 5,
            records_returned: 5,
        };
        let scope = MemoryScope {
            top_k: 2,
            ..Default::default()
        };
        let (rows, _) = MemoryAqlSource::assemble_rows(&row_by_id, &ids, vec![semantic], &scope)
            .expect("assemble");
        assert_eq!(rows.len(), 2);
    }
}
