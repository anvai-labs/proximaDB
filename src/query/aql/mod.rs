//! # Agentic Query Language (AQL) — TD-050.
//!
//! Implementation of the auditable query layer aligned with the RUBICON
//! architecture (arXiv:2604.21413, Stonebraker 2026).
//!
//! RUBICON replaces opaque LLM reasoning chains with explicit, auditable
//! query plans expressed in AQL. ProximaDB uses AQL as a structured
//! intermediate representation that guarantees a structured audit trail
//! for every cross-model operation.

use crate::query::unified::ast::{JoinType as UnifiedJoinType, MultiModelQuery, QueryComponent};
use proximadb_data_model::MemoryType;
use proximadb_kernel::error::ProximaDBError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub mod executor;
pub mod sources;

#[cfg(test)]
mod executor_test;

pub type Result<T> = std::result::Result<T, ProximaDBError>;

/// Fields whose vector-valued `Equals` carries the search query vector. The
/// vector / memory sources consume it; the executor strips it from the WHERE
/// post-filter (result rows carry no such field) and rejects it under `Not` /
/// `Or` (`executor::post_filter_predicate`).
pub(crate) fn is_query_vector_field(field: &str) -> bool {
    matches!(field, "embedding" | "vector" | "query_vector")
}
pub type DataModel = crate::query::unified::ast::DataModel;

// ---------------------------------------------------------------------------
// AQL AST
// ---------------------------------------------------------------------------

/// Top-level AQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqlQuery {
    pub find: AqlFind,
    pub from: AqlFrom,
    pub where_clause: AqlWhere,
}

/// Head of the AQL query, defining the projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqlFind {
    pub projections: Vec<AqlProjection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqlProjection {
    pub field: String,
    pub alias: Option<String>,
}

/// Data source definition, supporting single sources, joins, and multi-model lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AqlFrom {
    /// A single named source (e.g., a collection or graph).
    Source { name: String, alias: Option<String> },
    /// A join between two AQL sources.
    Join {
        left: Box<AqlFrom>,
        right: Box<AqlFrom>,
        on: AqlPredicate,
        join_type: JoinType,
    },
    /// A list of independent or chained sources (maps to MultiModelQuery).
    MultiSource { sources: Vec<AqlSourceSpec> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqlSourceSpec {
    pub name: String,
    pub model: DataModel,
    pub alias: Option<String>,
    pub dependencies: Vec<AqlDependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqlDependency {
    pub source_index: usize,
    pub on_field: String,
    pub join_type: JoinType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Semi,
    Anti,
    Semantic,
}
/// The filter part of an AQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AqlWhere {
    pub predicate: Option<AqlPredicate>,
}

/// Predicates for filtering and join conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AqlPredicate {
    Equals {
        field: String,
        value: AqlValue,
    },
    GreaterThan {
        field: String,
        value: AqlValue,
    },
    LessThan {
        field: String,
        value: AqlValue,
    },
    Contains {
        field: String,
        value: AqlValue,
    },
    And {
        lhs: Box<AqlPredicate>,
        rhs: Box<AqlPredicate>,
    },
    Or {
        lhs: Box<AqlPredicate>,
        rhs: Box<AqlPredicate>,
    },
    Not {
        inner: Box<AqlPredicate>,
    },
    /// Semantic similarity filter (vector search).
    SemanticMatch {
        field: String,
        query: String,
        threshold: f32,
        top_k: u32,
    },
    /// High-fidelity memory type filter (Memanto — TD-055).
    TypeMatch {
        memory_type: MemoryType,
    },
}

/// Literal values in AQL predicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AqlValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Vector(Vec<f32>),
    /// ISO-8601 date string or days since epoch.
    Date(String),
    /// ISO-8601 timestamp string with offset.
    TimestampTz(String),
    /// Structured JSON data.
    Json(serde_json::Value),
    /// Binary JSON data for faster access.
    Jsonb(serde_json::Value),
    Null,
}

// ---------------------------------------------------------------------------
// Audit Trail (The load-bearing change)
// ---------------------------------------------------------------------------

/// A complete auditable trace of an AQL query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    pub query_id: Uuid,
    pub started_at_ms: i64,
    pub finished_at_ms: i64,
    pub plan: AqlQuery,
    pub frames: Vec<AuditFrame>,
    pub outcome: AuditOutcome,
}

/// A single step in the execution of an AQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFrame {
    pub frame_id: u64,
    pub source: DataModel,
    pub op: AuditOp,
    pub filters_pushed: Vec<String>, // Serialized predicates
    pub filters_post: Vec<String>,
    pub records_scanned: u64,
    pub records_returned: u64,
    pub wall_time_us: u64,
    pub error: Option<String>,
    pub redaction_count: u32,
}

/// Operations captured in the audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOp {
    VectorSearch {
        collection: String,
        top_k: u32,
        metric: String,
    },
    GraphTraversal {
        graph_id: String,
        depth: u32,
        algorithm: String,
    },
    DocumentQuery {
        collection: String,
    },
    Join {
        join_type: JoinType,
        left_frame: u64,
        right_frame: u64,
    },
    Scan {
        source: String,
    },
    /// Type-filtered retrieval step.
    TypeMatch {
        memory_type: MemoryType,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    PartialFailure { failed_frames: Vec<u64> },
    Failure { reason: String },
}

/// Context passed through the executor to collect audit frames.
pub struct AuditContext {
    pub query_id: Uuid,
    pub frames: Vec<AuditFrame>,
    pub next_frame_id: u64,
}

impl Default for AuditContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditContext {
    pub fn new() -> Self {
        Self {
            query_id: Uuid::new_v4(),
            frames: Vec::new(),
            next_frame_id: 1,
        }
    }

    pub fn push_frame(&mut self, mut frame: AuditFrame) -> u64 {
        let id = self.next_frame_id;
        frame.frame_id = id;
        self.frames.push(frame);
        self.next_frame_id += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// Source Wrapper Trait
// ---------------------------------------------------------------------------

/// Trait for data sources that can resolve AQL operations.
#[async_trait::async_trait]
pub trait AqlSource: Send + Sync {
    /// The data model this wrapper resolves.
    fn model(&self) -> DataModel;

    /// Execute the wrapped operation and emit an audit frame.
    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult>;
}

// ---------------------------------------------------------------------------
// Conversion from MultiModelQuery (TD-050 Phase 3)
// ---------------------------------------------------------------------------

impl AqlQuery {
    pub fn from_multi_model(q: &MultiModelQuery) -> Self {
        let sources = q
            .components
            .iter()
            .map(AqlSourceSpec::from_component)
            .collect();

        Self {
            find: AqlFind {
                projections: vec![AqlProjection {
                    field: "*".to_string(),
                    alias: None,
                }],
            },
            from: AqlFrom::MultiSource { sources },
            where_clause: AqlWhere { predicate: None },
        }
    }
}

impl AqlSourceSpec {
    pub fn from_component(c: &QueryComponent) -> Self {
        let name = c.target_collection().unwrap_or("default").to_string();
        let model = c.model;

        let dependencies = c
            .dependencies
            .iter()
            .map(|d| AqlDependency {
                source_index: d.component_index,
                on_field: d.join_field.clone(),
                join_type: JoinType::from_unified(&d.join_type),
            })
            .collect();

        Self {
            name,
            model,
            alias: None,
            dependencies,
        }
    }
}

impl JoinType {
    pub fn from_unified(jt: &UnifiedJoinType) -> Self {
        match jt {
            UnifiedJoinType::Inner => JoinType::Inner,
            UnifiedJoinType::LeftOuter => JoinType::Left,
            UnifiedJoinType::Semi => JoinType::Semi,
            UnifiedJoinType::Anti => JoinType::Anti,
            UnifiedJoinType::Semantic { .. } => JoinType::Semantic,
        }
    }
}

/// Result from an AQL source execution.
pub struct AqlResult {
    /// The resulting rows (columns -> values).
    pub rows: Vec<HashMap<String, AqlValue>>,
    /// ID of the audit frame generated for this result.
    pub frame_id: u64,
}
