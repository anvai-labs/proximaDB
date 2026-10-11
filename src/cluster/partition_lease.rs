//! Generation-fenced **partition leases** — Phase 7 of the read-heavy
//! system-catalog redesign.
//!
//! The catalog's global write authority is the object-store generation fence
//! (Phase 6a/6b): a single highest-generation pod publishes DDL, and stale
//! writers are fenced. That is correct but coarse — *every* write serializes
//! through one pod. A partition lease makes it **per-partition**: a peer acquires
//! a durable, generation-fenced lease over one `(tenant, collection)` and then
//! serves that partition's writes locally, contention-free, exactly like
//! CockroachDB's per-range leaseholder or Spanner's per-directory leader. The
//! object-store fence stays the *correctness* authority (no two pods can hold the
//! same partition); the lease is the *latency* optimization (no global CAS per
//! write while the lease is valid). Cross-partition DDL still falls back to the
//! global catalog fence.
//!
//! ## Mechanism (reuses the Phase-6a substrate)
//!
//! Each `(tenant, collection)` has its own fenced manifest log under
//! `{prefix}/{tenant}/{collection}/_manifests/` (the `prefix` is rooted at the
//! operator control plane via [`DrPathBuilder::operator_subprefix`]). The lease
//! body is committed with [`ManifestCommitter::commit_fenced`]: the **version**
//! CAS (`put_if_absent` on the successor slot) makes acquisition atomic, and the
//! **generation** header fences a stale owner — a writer carrying a generation
//! below the one a newer pod has committed is rejected *before* it can clobber
//! the lease. Two pods contending therefore converge to exactly one owner; the
//! loser re-reads and observes who won.
//!
//! ## Lifecycle
//!
//! - **Acquire (fresh / takeover-of-expired):** read the current lease; if none
//!   or expired, commit a new lease at `generation + 1` (strictly outranking the
//!   prior, so the expired owner's renewal is fenced). A live lease held by
//!   another pod yields [`LeaseOutcome::Held`].
//! - **Renew:** the holder re-commits at the *same* generation with a fresh
//!   expiry. If a takeover has happened in the meantime, the renew is
//!   [`LeaseOutcome::Fenced`]/`Held` — the stale owner learns it must step down.
//! - **Expiry / handoff:** a lease that is not renewed expires at its
//!   `expires_at_ms`; the next acquirer takes it over with a higher generation.
//!
//! Clocks are passed in explicitly (`now_ms`) so expiry is deterministic in
//! tests and so a deployment can choose its time source; like the queue leases,
//! a few seconds of clock skew only shifts handoff timing, never correctness
//! (the fence, not the clock, guarantees single-ownership).

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{OnceLock, RwLock as StdRwLock};
use std::time::Duration;

use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use proximadb_iceberg_engine::manifest::{CommitOutcome, ManifestCommitter};
use proximadb_object_store::ProximaObjectStore;

use crate::cluster::primary_pod_registry::{
    AssignmentReason, PrimaryPodRegistry, WriteRoutingDecision, consult_for_write,
};

////////////////////////////////////////////////////////////////////////////////
// Unified Resource Key (Type-Agnostic Core)
////////////////////////////////////////////////////////////////////////////////

/// Resource type discriminator — enables routing to correct strategy
/// and determines how `resource_id` is interpreted.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Vector collections (HNSW/IVF indexes)
    Collection,
    /// Relational tables (warehouse DataFusion/Polars)
    Table,
    /// Relational schemas (DDL namespace)
    Schema,
    /// Graph databases
    Graph,
    /// Graph nodes (individual vertex locks)
    GraphNode,
    /// Graph edges (individual edge locks)
    GraphEdge,
    /// Document collections
    Document,
    /// Document (individual document locks)
    Doc,
    /// ML models
    Model,
    /// Specific model versions
    ModelVersion,
    /// ML experiments
    Experiment,
    /// ML experiment runs
    ExperimentRun,
    /// Feature sets
    FeatureSet,
}

impl ResourceType {
    /// Legacy collection type (for backward compatibility)
    pub const LEGACY_COLLECTION: &'static str = "collection";

    /// Human-readable name for logging
    pub fn name(&self) -> &str {
        match self {
            ResourceType::Collection => "collection",
            ResourceType::Table => "table",
            ResourceType::Schema => "schema",
            ResourceType::Graph => "graph",
            ResourceType::GraphNode => "graph_node",
            ResourceType::GraphEdge => "graph_edge",
            ResourceType::Document => "document",
            ResourceType::Doc => "doc",
            ResourceType::Model => "model",
            ResourceType::ModelVersion => "model_version",
            ResourceType::Experiment => "experiment",
            ResourceType::ExperimentRun => "experiment_run",
            ResourceType::FeatureSet => "feature_set",
        }
    }
}

/// Resource identifier — interpretation varies by ResourceType
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceIdentifier {
    /// Single identifier (collection_id, table_name, model_id, etc.)
    Single(String),

    /// Composite identifier (schema.table, graph:node, model:version, etc.)
    Composite(Vec<String>),

    /// Hierarchical identifier (for nested resources like schema.table.partition)
    Hierarchical { parent: String, child: String },
}

impl ResourceIdentifier {
    /// Create a single identifier
    pub fn single(id: impl Into<String>) -> Self {
        ResourceIdentifier::Single(id.into())
    }

    /// Create a composite identifier from parts
    pub fn composite(parts: Vec<String>) -> Self {
        ResourceIdentifier::Composite(parts)
    }

    /// Create a hierarchical identifier
    pub fn hierarchical(parent: impl Into<String>, child: impl Into<String>) -> Self {
        ResourceIdentifier::Hierarchical {
            parent: parent.into(),
            child: child.into(),
        }
    }

    /// Flatten to a string path for manifest storage
    pub fn to_path(&self) -> String {
        match self {
            ResourceIdentifier::Single(s) => encode_path_component(s),
            ResourceIdentifier::Composite(parts) => parts
                .iter()
                .map(|part| encode_path_component(part))
                .collect::<Vec<_>>()
                .join("/"),
            ResourceIdentifier::Hierarchical { parent, child } => {
                format!(
                    "{}/{}",
                    encode_path_component(parent),
                    encode_path_component(child)
                )
            }
        }
    }
}

fn encode_path_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

fn decode_path_component(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'%' {
            let hi = *bytes.get(idx + 1)?;
            let lo = *bytes.get(idx + 2)?;
            let hex = [hi, lo];
            let hex = std::str::from_utf8(&hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            idx += 3;
        } else {
            out.push(bytes[idx]);
            idx += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Unified key for all lease-protected resources across all modalities.
///
/// This key structure is type-agnostic at the lease layer but enables
/// type-specific routing via `resource_type`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceKey {
    /// Tenant owning the resource.
    pub tenant_id: String,

    /// Namespace (optional — None for global resources)
    pub namespace_id: Option<String>,

    /// Resource type discriminator.
    pub resource_type: ResourceType,

    /// Resource identifier.
    pub resource_id: ResourceIdentifier,
}

impl ResourceKey {
    /// Create a new resource key.
    pub fn new(
        tenant_id: impl Into<String>,
        namespace_id: Option<String>,
        resource_type: ResourceType,
        resource_id: ResourceIdentifier,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id,
            resource_type,
            resource_id,
        }
    }

    /// Create a legacy collection key (for backward compatibility).
    pub fn legacy_collection(
        tenant_id: impl Into<String>,
        collection_id: impl Into<String>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id: None,
            resource_type: ResourceType::Collection,
            resource_id: ResourceIdentifier::single(collection_id),
        }
    }

    /// Create a table key (schema.table).
    pub fn table(
        tenant_id: impl Into<String>,
        schema_name: impl Into<String>,
        table_name: impl Into<String>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id: None,
            resource_type: ResourceType::Table,
            resource_id: ResourceIdentifier::composite(vec![schema_name.into(), table_name.into()]),
        }
    }

    /// Create a schema key.
    pub fn schema(tenant_id: impl Into<String>, schema_name: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id: None,
            resource_type: ResourceType::Schema,
            resource_id: ResourceIdentifier::single(schema_name),
        }
    }

    /// Create a graph key.
    pub fn graph(tenant_id: impl Into<String>, graph_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id: None,
            resource_type: ResourceType::Graph,
            resource_id: ResourceIdentifier::single(graph_id),
        }
    }

    /// Create a model key.
    pub fn model(tenant_id: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id: None,
            resource_type: ResourceType::Model,
            resource_id: ResourceIdentifier::single(model_name),
        }
    }

    /// Create an experiment key.
    pub fn experiment(tenant_id: impl Into<String>, experiment_name: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            namespace_id: None,
            resource_type: ResourceType::Experiment,
            resource_id: ResourceIdentifier::single(experiment_name),
        }
    }

    /// Flatten to a path string for manifest storage.
    /// Format: `{tenant_id}/{resource_type}/{resource_path}` or `{tenant_id}/{namespace}/{resource_type}/{resource_path}`
    pub fn to_path(&self) -> String {
        let tenant = encode_path_component(&self.tenant_id);
        if let Some(ns) = &self.namespace_id {
            format!(
                "{}/{}/{}/{}",
                tenant,
                encode_path_component(ns),
                self.resource_type.name(),
                self.resource_id.to_path()
            )
        } else {
            format!(
                "{}/{}/{}",
                tenant,
                self.resource_type.name(),
                self.resource_id.to_path()
            )
        }
    }

    /// Parse a path string back into a ResourceKey (for migration).
    /// Handles both formats: {tenant_id}/{resource_type}/{resource_path} and
    /// {tenant_id}/{namespace}/{resource_type}/{resource_path}.
    pub fn from_path(path: &str) -> Option<Self> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 3 {
            return None;
        }

        // Try to detect format by checking if the second part is a known resource type
        let (tenant_id, namespace_id, resource_type_idx) = match parts.get(1) {
            Some(
                &("collection" | "table" | "schema" | "graph" | "graph_node" | "graph_edge"
                | "document" | "doc" | "model" | "model_version" | "experiment"
                | "experiment_run" | "feature_set"),
            ) => {
                // Format: {tenant_id}/{resource_type}/{resource_path}
                (parts.first()?, None, 1)
            }
            Some(&ns) => {
                // Format: {tenant_id}/{namespace}/{resource_type}/{resource_path}
                (parts.first()?, Some(decode_path_component(ns)?), 2)
            }
            None => return None,
        };

        let resource_type = match *parts.get(resource_type_idx)? {
            "collection" => ResourceType::Collection,
            "table" => ResourceType::Table,
            "schema" => ResourceType::Schema,
            "graph" => ResourceType::Graph,
            "graph_node" => ResourceType::GraphNode,
            "graph_edge" => ResourceType::GraphEdge,
            "document" => ResourceType::Document,
            "doc" => ResourceType::Doc,
            "model" => ResourceType::Model,
            "model_version" => ResourceType::ModelVersion,
            "experiment" => ResourceType::Experiment,
            "experiment_run" => ResourceType::ExperimentRun,
            "feature_set" => ResourceType::FeatureSet,
            _ => return None,
        };

        // Remaining parts form the resource_id
        let resource_parts: Vec<String> = parts
            .iter()
            .skip(resource_type_idx + 1)
            .map(|s| decode_path_component(s))
            .collect::<Option<Vec<_>>>()?;
        let resource_id = if resource_parts.len() == 1 {
            ResourceIdentifier::single(resource_parts.into_iter().next()?)
        } else {
            ResourceIdentifier::composite(resource_parts)
        };

        Some(ResourceKey {
            tenant_id: decode_path_component(tenant_id)?,
            namespace_id,
            resource_type,
            resource_id,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////
// Resource Strategy Pattern (Type-Specific Behavior)
////////////////////////////////////////////////////////////////////////////////

/// Strategy for handling a specific resource type.
///
/// This trait enables type-specific behavior while keeping the lease mechanism
/// type-agnostic. Different resource types have different:
/// - Lease TTL requirements (latency vs safety tradeoffs)
/// - Partitioning strategies (how to subdivide resources)
/// - Conflict resolution (fencing vs optimistic concurrency)
pub trait ResourceStrategy: Send + Sync {
    /// Strategy name for telemetry/debugging.
    fn name(&self) -> &str;

    /// The resource type this strategy handles.
    fn resource_type(&self) -> ResourceType {
        ResourceType::Collection // Default
    }

    /// Lease TTL for this resource type (in seconds).
    ///
    /// Different workloads have different latency requirements:
    /// - Vector collections: 60s (frequent writes, low latency)
    /// - Relational tables: 120s (DDL is less frequent, longer TTL reduces overhead)
    /// - ML models: 300s (model updates are rare, longer TTL is acceptable)
    fn lease_ttl_secs(&self) -> u64 {
        60 // Default: 60 seconds
    }

    /// Whether this resource type supports partitioning.
    ///
    /// Resources that support partitioning can be subdivided for parallel writes:
    /// - Collections: partition by shard ID
    /// - Tables: partition by partition ID
    /// - Graphs: partition by split
    fn supports_partitioning(&self) -> bool {
        false
    }

    /// Generate a resource-specific key from generic components.
    ///
    /// This method interprets the `components` slice based on the resource type:
    /// - Collection: [collection_id]
    /// - Table: [schema_name, table_name]
    /// - Graph: [graph_id]
    ///
    /// Returns [`LeaseError::InvalidKey`] when `components` does not satisfy the
    /// required shape for this resource type (e.g. an empty slice for a Collection).
    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError>;

    /// Validate that an operation is allowed on this resource type.
    ///
    /// This enables type-specific validation logic.
    fn validate_operation(&self, _operation: &ResourceOperation) -> Result<(), LeaseError> {
        Ok(()) // Default: no validation
    }

    /// Whether this resource type supports hierarchical locking.
    ///
    /// Hierarchical locking enables locking at different levels:
    /// - Schema → Table → Partition → Record
    fn supports_hierarchical_locking(&self) -> bool {
        false
    }

    /// Get the parent key for hierarchical locking (if supported).
    fn parent_key(&self, _key: &ResourceKey) -> Option<ResourceKey> {
        None
    }
}

/// Standard resource operations for validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceOperation {
    // DDL operations
    Create,
    Drop,
    Alter,
    // DML operations
    Read,
    Write,
    // Schema-level operations
    SchemaChange,
    // Admin operations
    Admin,
    // ML-specific operations
    ModelRegister,
    ModelDeploy,
    ExperimentLog,
    MetricWrite,
    FeatureWrite,
    TrainingRun,
    Inference,
}

/// Lease strategy errors.
#[derive(Debug, thiserror::Error)]
pub enum LeaseError {
    #[error("Operation {operation:?} not allowed on resource type {resource_type}")]
    InvalidOperation {
        resource_type: String,
        operation: String,
    },
    #[error("Invalid key for resource type: {reason}")]
    InvalidKey { reason: String },
}

////////////////////////////////////////////////////////////////////////////////
// Built-in Strategy Implementations
////////////////////////////////////////////////////////////////////////////////

/// Strategy for vector collections.
#[derive(Debug, Clone)]
pub struct CollectionStrategy;

impl ResourceStrategy for CollectionStrategy {
    fn name(&self) -> &str {
        "collection"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Collection
    }

    fn lease_ttl_secs(&self) -> u64 {
        60 // Vector workloads have frequent writes
    }

    fn supports_partitioning(&self) -> bool {
        true // Collections can be sharded
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "Collection key requires at least collection_id".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Collection,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            ResourceOperation::SchemaChange => Err(LeaseError::InvalidOperation {
                resource_type: "collection".to_string(),
                operation: "SchemaChange".to_string(),
            }),
            _ => Ok(()),
        }
    }
}

/// Strategy for relational tables.
#[derive(Debug, Clone)]
pub struct TableStrategy;

impl ResourceStrategy for TableStrategy {
    fn name(&self) -> &str {
        "table"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Table
    }

    fn lease_ttl_secs(&self) -> u64 {
        120 // DDL is less frequent, can afford longer TTL
    }

    fn supports_partitioning(&self) -> bool {
        true // Tables can be partitioned
    }

    fn supports_hierarchical_locking(&self) -> bool {
        true // Schema → Table → Partition hierarchy
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.len() < 2 {
            return Err(LeaseError::InvalidKey {
                reason: "Table key requires schema_name and table_name".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Table,
            resource_id: ResourceIdentifier::composite(vec![
                components[0].clone(),
                components[1].clone(),
            ]),
        })
    }

    fn parent_key(&self, key: &ResourceKey) -> Option<ResourceKey> {
        // Parent of a table is its schema
        match &key.resource_id {
            ResourceIdentifier::Composite(parts) if !parts.is_empty() => Some(ResourceKey {
                tenant_id: key.tenant_id.clone(),
                namespace_id: key.namespace_id.clone(),
                resource_type: ResourceType::Schema,
                resource_id: ResourceIdentifier::single(parts[0].clone()),
            }),
            _ => None,
        }
    }

    fn validate_operation(&self, _operation: &ResourceOperation) -> Result<(), LeaseError> {
        // All operations are valid on tables
        Ok(())
    }
}

/// Strategy for relational schemas.
#[derive(Debug, Clone)]
pub struct SchemaStrategy;

impl ResourceStrategy for SchemaStrategy {
    fn name(&self) -> &str {
        "schema"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Schema
    }

    fn lease_ttl_secs(&self) -> u64 {
        300 // Schema changes are rare, use longer TTL
    }

    fn supports_partitioning(&self) -> bool {
        false // Schemas are not partitioned
    }

    fn supports_hierarchical_locking(&self) -> bool {
        true // Schema is the root of the hierarchy
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "Schema key requires at least schema_name".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Schema,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            ResourceOperation::Write | ResourceOperation::Read => {
                Err(LeaseError::InvalidOperation {
                    resource_type: "schema".to_string(),
                    operation: format!("{:?}", operation),
                })
            }
            _ => Ok(()),
        }
    }
}

/// Strategy for graph databases.
#[derive(Debug, Clone)]
pub struct GraphStrategy;

impl ResourceStrategy for GraphStrategy {
    fn name(&self) -> &str {
        "graph"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Graph
    }

    fn lease_ttl_secs(&self) -> u64 {
        60 // Graph workloads have frequent writes
    }

    fn supports_partitioning(&self) -> bool {
        true // Graphs can be split
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "Graph key requires at least graph_id".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Graph,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }

    fn validate_operation(&self, _operation: &ResourceOperation) -> Result<(), LeaseError> {
        // All operations are valid on graphs
        Ok(())
    }
}

/// Strategy for individual graph nodes (for DML-level locking).
#[derive(Debug, Clone)]
pub struct GraphNodeStrategy;

impl ResourceStrategy for GraphNodeStrategy {
    fn name(&self) -> &str {
        "graph_node"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::GraphNode
    }

    fn lease_ttl_secs(&self) -> u64 {
        10 // Node locks are short-lived
    }

    fn supports_partitioning(&self) -> bool {
        false // Individual nodes are not partitioned
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.len() < 2 {
            return Err(LeaseError::InvalidKey {
                reason: "GraphNode key requires graph_id and node_id".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::GraphNode,
            resource_id: ResourceIdentifier::composite(vec![
                components[0].clone(),
                components[1].clone(),
            ]),
        })
    }
}

/// Strategy for individual graph edges (for DML-level locking).
#[derive(Debug, Clone)]
pub struct GraphEdgeStrategy;

impl ResourceStrategy for GraphEdgeStrategy {
    fn name(&self) -> &str {
        "graph_edge"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::GraphEdge
    }

    fn lease_ttl_secs(&self) -> u64 {
        10 // Edge locks are short-lived
    }

    fn supports_partitioning(&self) -> bool {
        false // Individual edges are not partitioned
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.len() < 2 {
            return Err(LeaseError::InvalidKey {
                reason: "GraphEdge key requires graph_id and edge_id".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::GraphEdge,
            resource_id: ResourceIdentifier::composite(vec![
                components[0].clone(),
                components[1].clone(),
            ]),
        })
    }
}

/// Strategy for document collections.
#[derive(Debug, Clone)]
pub struct DocumentStrategy;

impl ResourceStrategy for DocumentStrategy {
    fn name(&self) -> &str {
        "document"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Document
    }

    fn lease_ttl_secs(&self) -> u64 {
        60 // Document workloads have frequent writes
    }

    fn supports_partitioning(&self) -> bool {
        true // Document collections can be sharded
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "Document key requires at least collection_id".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Document,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }
}

/// Strategy for ML models.
#[derive(Debug, Clone)]
pub struct ModelStrategy;

impl ResourceStrategy for ModelStrategy {
    fn name(&self) -> &str {
        "model"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Model
    }

    fn lease_ttl_secs(&self) -> u64 {
        300 // Model updates are rare
    }

    fn supports_partitioning(&self) -> bool {
        false // Models are not partitioned
    }

    fn supports_hierarchical_locking(&self) -> bool {
        true // Model → ModelVersion hierarchy
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "Model key requires at least model_name".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Model,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }

    fn parent_key(&self, key: &ResourceKey) -> Option<ResourceKey> {
        // ModelVersion parent is the Model
        if key.resource_type == ResourceType::ModelVersion {
            match &key.resource_id {
                ResourceIdentifier::Composite(parts) if !parts.is_empty() => {
                    return Some(ResourceKey {
                        tenant_id: key.tenant_id.clone(),
                        namespace_id: key.namespace_id.clone(),
                        resource_type: ResourceType::Model,
                        resource_id: ResourceIdentifier::single(parts[0].clone()),
                    });
                }
                _ => {}
            }
        }
        None
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            // ML-specific operations valid on models
            ResourceOperation::ModelRegister | ResourceOperation::ModelDeploy => Ok(()),
            // DDL-like operations
            ResourceOperation::Create | ResourceOperation::Drop | ResourceOperation::Admin => {
                Ok(())
            }
            // Read operations for model metadata
            ResourceOperation::Read => Ok(()),
            // Other operations not valid on models
            _ => Err(LeaseError::InvalidOperation {
                resource_type: "model".to_string(),
                operation: format!("{:?}", operation),
            }),
        }
    }
}

/// Strategy for ML model versions.
#[derive(Debug, Clone)]
pub struct ModelVersionStrategy;

impl ResourceStrategy for ModelVersionStrategy {
    fn name(&self) -> &str {
        "model_version"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::ModelVersion
    }

    fn lease_ttl_secs(&self) -> u64 {
        300 // Model version updates are rare
    }

    fn supports_partitioning(&self) -> bool {
        false // Model versions are not partitioned
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.len() < 2 {
            return Err(LeaseError::InvalidKey {
                reason: "ModelVersion key requires model_name and version".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::ModelVersion,
            resource_id: ResourceIdentifier::composite(vec![
                components[0].clone(),
                components[1].clone(),
            ]),
        })
    }

    fn parent_key(&self, key: &ResourceKey) -> Option<ResourceKey> {
        // Parent of a model version is its model
        match &key.resource_id {
            ResourceIdentifier::Composite(parts) if !parts.is_empty() => Some(ResourceKey {
                tenant_id: key.tenant_id.clone(),
                namespace_id: key.namespace_id.clone(),
                resource_type: ResourceType::Model,
                resource_id: ResourceIdentifier::single(parts[0].clone()),
            }),
            _ => None,
        }
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            // Model-specific operations
            ResourceOperation::ModelDeploy | ResourceOperation::Inference => Ok(()),
            // Read operations
            ResourceOperation::Read => Ok(()),
            // Admin operations
            ResourceOperation::Admin => Ok(()),
            // Other operations not valid on model versions
            _ => Err(LeaseError::InvalidOperation {
                resource_type: "model_version".to_string(),
                operation: format!("{:?}", operation),
            }),
        }
    }
}

/// Strategy for ML experiments.
#[derive(Debug, Clone)]
pub struct ExperimentStrategy;

impl ResourceStrategy for ExperimentStrategy {
    fn name(&self) -> &str {
        "experiment"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::Experiment
    }

    fn lease_ttl_secs(&self) -> u64 {
        120 // Experiment updates are moderately frequent
    }

    fn supports_partitioning(&self) -> bool {
        false // Experiments are not partitioned
    }

    fn supports_hierarchical_locking(&self) -> bool {
        true // Experiment → ExperimentRun hierarchy
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "Experiment key requires at least experiment_name".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::Experiment,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }

    fn parent_key(&self, key: &ResourceKey) -> Option<ResourceKey> {
        // ExperimentRun parent is the Experiment
        if key.resource_type == ResourceType::ExperimentRun {
            match &key.resource_id {
                ResourceIdentifier::Composite(parts) if !parts.is_empty() => {
                    return Some(ResourceKey {
                        tenant_id: key.tenant_id.clone(),
                        namespace_id: key.namespace_id.clone(),
                        resource_type: ResourceType::Experiment,
                        resource_id: ResourceIdentifier::single(parts[0].clone()),
                    });
                }
                _ => {}
            }
        }
        None
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            // ML-specific operations valid on experiments
            ResourceOperation::ExperimentLog | ResourceOperation::MetricWrite => Ok(()),
            // DDL-like operations
            ResourceOperation::Create | ResourceOperation::Drop | ResourceOperation::Admin => {
                Ok(())
            }
            // Read operations
            ResourceOperation::Read => Ok(()),
            // Other operations not valid on experiments
            _ => Err(LeaseError::InvalidOperation {
                resource_type: "experiment".to_string(),
                operation: format!("{:?}", operation),
            }),
        }
    }
}

/// Strategy for ML experiment runs.
#[derive(Debug, Clone)]
pub struct ExperimentRunStrategy;

impl ResourceStrategy for ExperimentRunStrategy {
    fn name(&self) -> &str {
        "experiment_run"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::ExperimentRun
    }

    fn lease_ttl_secs(&self) -> u64 {
        60 // Experiment run updates are frequent
    }

    fn supports_partitioning(&self) -> bool {
        false // Experiment runs are not partitioned
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.len() < 2 {
            return Err(LeaseError::InvalidKey {
                reason: "ExperimentRun key requires experiment_name and run_id".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::ExperimentRun,
            resource_id: ResourceIdentifier::composite(vec![
                components[0].clone(),
                components[1].clone(),
            ]),
        })
    }

    fn parent_key(&self, key: &ResourceKey) -> Option<ResourceKey> {
        // Parent of an experiment run is its experiment
        match &key.resource_id {
            ResourceIdentifier::Composite(parts) if !parts.is_empty() => Some(ResourceKey {
                tenant_id: key.tenant_id.clone(),
                namespace_id: key.namespace_id.clone(),
                resource_type: ResourceType::Experiment,
                resource_id: ResourceIdentifier::single(parts[0].clone()),
            }),
            _ => None,
        }
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            // ML-specific operations valid on experiment runs
            ResourceOperation::TrainingRun | ResourceOperation::MetricWrite => Ok(()),
            // Write operations for logging
            ResourceOperation::Write => Ok(()),
            // Read operations
            ResourceOperation::Read => Ok(()),
            // Other operations not valid on experiment runs
            _ => Err(LeaseError::InvalidOperation {
                resource_type: "experiment_run".to_string(),
                operation: format!("{:?}", operation),
            }),
        }
    }
}

/// Strategy for feature sets.
#[derive(Debug, Clone)]
pub struct FeatureSetStrategy;

impl ResourceStrategy for FeatureSetStrategy {
    fn name(&self) -> &str {
        "feature_set"
    }

    fn resource_type(&self) -> ResourceType {
        ResourceType::FeatureSet
    }

    fn lease_ttl_secs(&self) -> u64 {
        120 // Feature set updates are moderately frequent
    }

    fn supports_partitioning(&self) -> bool {
        false // Feature sets are not partitioned
    }

    fn make_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        components: &[String],
    ) -> Result<ResourceKey, LeaseError> {
        if components.is_empty() {
            return Err(LeaseError::InvalidKey {
                reason: "FeatureSet key requires at least feature_set_name".to_string(),
            });
        }
        Ok(ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(|s| s.to_string()),
            resource_type: ResourceType::FeatureSet,
            resource_id: ResourceIdentifier::single(components[0].clone()),
        })
    }

    fn validate_operation(&self, operation: &ResourceOperation) -> Result<(), LeaseError> {
        match operation {
            // ML-specific operations valid on feature sets
            ResourceOperation::FeatureWrite => Ok(()),
            // DDL-like operations
            ResourceOperation::Create
            | ResourceOperation::Alter
            | ResourceOperation::Drop
            | ResourceOperation::Admin => Ok(()),
            // Read operations
            ResourceOperation::Read => Ok(()),
            // Other operations not valid on feature sets
            _ => Err(LeaseError::InvalidOperation {
                resource_type: "feature_set".to_string(),
                operation: format!("{:?}", operation),
            }),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// DML-Level Locking (Hierarchical & Coordinator-Driven)
////////////////////////////////////////////////////////////////////////////////

/// Lock levels (hierarchical): Schema → Table → Partition → Record
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LockLevel {
    /// Schema-level lock (for DDL operations on schemas)
    Schema,
    /// Table-level lock (for bulk operations like TRUNCATE, bulk load)
    Table,
    /// Partition-level lock (for partition-specific operations)
    Partition,
    /// Record-level lock (for individual row/document operations)
    Record,
}

impl LockLevel {
    /// Human-readable name
    pub fn name(&self) -> &str {
        match self {
            LockLevel::Schema => "schema",
            LockLevel::Table => "table",
            LockLevel::Partition => "partition",
            LockLevel::Record => "record",
        }
    }

    /// Whether this level can lock a parent level (for hierarchical locking)
    pub fn can_lock_parent(&self) -> bool {
        matches!(
            self,
            LockLevel::Table | LockLevel::Partition | LockLevel::Record
        )
    }

    /// Whether this level can lock a child level
    pub fn can_lock_child(&self) -> bool {
        matches!(
            self,
            LockLevel::Schema | LockLevel::Table | LockLevel::Partition
        )
    }

    /// Get the parent level (if any)
    pub fn parent(&self) -> Option<LockLevel> {
        match self {
            LockLevel::Schema => None,
            LockLevel::Table => Some(LockLevel::Schema),
            LockLevel::Partition => Some(LockLevel::Table),
            LockLevel::Record => Some(LockLevel::Partition),
        }
    }
}

/// Lock intent for compatibility checking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LockIntent {
    /// Shared read lock (multiple readers can hold)
    Read,
    /// Exclusive write lock (only one holder)
    Write,
    /// DDL/schema change lock (exclusive, blocks all others)
    Schema,
}

impl LockIntent {
    /// Human-readable name
    pub fn name(&self) -> &str {
        match self {
            LockIntent::Read => "read",
            LockIntent::Write => "write",
            LockIntent::Schema => "schema",
        }
    }

    /// Whether this intent is compatible with another intent.
    ///
    /// MVP DML locking uses the existing single-holder object-store lease, so
    /// every durable DML lock is exclusive. Shared read locks need a separate
    /// multi-holder durable record and are intentionally not claimed here.
    pub fn is_compatible_with(&self, _other: &LockIntent) -> bool {
        false
    }
}

/// Scope of a DML lock (what is being locked).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DmlLockScope {
    /// Schema-level lock
    Schema { schema_name: String },
    /// Table-level lock
    Table {
        schema_name: String,
        table_name: String,
    },
    /// Partition-level lock
    Partition {
        schema_name: String,
        table_name: String,
        partition_id: String,
    },
    /// Record-level lock
    Record {
        schema_name: String,
        table_name: String,
        key: String,
    },
}

impl DmlLockScope {
    /// Get the lock level for this scope
    pub fn level(&self) -> LockLevel {
        match self {
            DmlLockScope::Schema { .. } => LockLevel::Schema,
            DmlLockScope::Table { .. } => LockLevel::Table,
            DmlLockScope::Partition { .. } => LockLevel::Partition,
            DmlLockScope::Record { .. } => LockLevel::Record,
        }
    }

    /// Get the parent scope (if any)
    pub fn parent(&self) -> Option<DmlLockScope> {
        match self {
            DmlLockScope::Schema { .. } => None,
            DmlLockScope::Table { schema_name, .. } => Some(DmlLockScope::Schema {
                schema_name: schema_name.clone(),
            }),
            DmlLockScope::Partition {
                schema_name,
                table_name,
                ..
            } => Some(DmlLockScope::Table {
                schema_name: schema_name.clone(),
                table_name: table_name.clone(),
            }),
            DmlLockScope::Record {
                schema_name,
                table_name,
                ..
            } => Some(DmlLockScope::Table {
                schema_name: schema_name.clone(),
                table_name: table_name.clone(),
            }),
        }
    }

    /// Check if this scope is an ancestor of another scope in the hierarchy.
    ///
    /// Hierarchy: Schema → Table → Partition → Record
    ///
    /// A Schema is an ancestor of Table/Partition/Record in the same schema.
    /// A Table is an ancestor of Partition/Record in the same table.
    /// A Partition is an ancestor of Records in the same partition (conceptually).
    pub fn is_ancestor_of(&self, other: &DmlLockScope) -> bool {
        // Same schema is the base requirement for any ancestor relationship
        if self.schema_name() != other.schema_name() {
            return false;
        }

        match (self, other) {
            // Schema is ancestor of all non-schema scopes in the same schema
            (DmlLockScope::Schema { .. }, DmlLockScope::Table { .. }) => true,
            (DmlLockScope::Schema { .. }, DmlLockScope::Partition { .. }) => true,
            (DmlLockScope::Schema { .. }, DmlLockScope::Record { .. }) => true,

            // Table is ancestor of partition/record in the same table
            (
                DmlLockScope::Table {
                    schema_name: s1,
                    table_name: t1,
                },
                DmlLockScope::Partition {
                    schema_name: s2,
                    table_name: t2,
                    ..
                },
            ) => s1 == s2 && t1 == t2,
            (
                DmlLockScope::Table {
                    schema_name: s1,
                    table_name: t1,
                },
                DmlLockScope::Record {
                    schema_name: s2,
                    table_name: t2,
                    ..
                },
            ) => s1 == s2 && t1 == t2,

            // Partition is ancestor of records in the same partition
            // (Note: Record scope doesn't carry partition_id, so we treat Table as the direct parent)
            (DmlLockScope::Partition { .. }, DmlLockScope::Record { .. }) => false,

            // Same scope is not ancestor of itself
            _ => false,
        }
    }

    /// Check if this scope is a descendant of another scope in the hierarchy.
    ///
    /// This is the inverse of `is_ancestor_of`.
    pub fn is_descendant_of(&self, other: &DmlLockScope) -> bool {
        other.is_ancestor_of(self)
    }

    /// Get the schema name (if applicable)
    pub fn schema_name(&self) -> Option<&str> {
        match self {
            DmlLockScope::Schema { schema_name }
            | DmlLockScope::Table { schema_name, .. }
            | DmlLockScope::Partition { schema_name, .. }
            | DmlLockScope::Record { schema_name, .. } => Some(schema_name),
        }
    }

    /// Get the table name (if applicable)
    pub fn table_name(&self) -> Option<&str> {
        match self {
            DmlLockScope::Table { table_name, .. }
            | DmlLockScope::Partition { table_name, .. }
            | DmlLockScope::Record { table_name, .. } => Some(table_name),
            DmlLockScope::Schema { .. } => None,
        }
    }

    /// Convert to a string key for in-memory lock registry
    pub fn to_key(&self) -> String {
        match self {
            DmlLockScope::Schema { schema_name } => {
                format!("schema:{}", encode_path_component(schema_name))
            }
            DmlLockScope::Table {
                schema_name,
                table_name,
            } => format!(
                "table:{}.{}",
                encode_path_component(schema_name),
                encode_path_component(table_name)
            ),
            DmlLockScope::Partition {
                schema_name,
                table_name,
                partition_id,
            } => format!(
                "partition:{}.{}.{}",
                encode_path_component(schema_name),
                encode_path_component(table_name),
                encode_path_component(partition_id)
            ),
            DmlLockScope::Record {
                schema_name,
                table_name,
                key,
            } => format!(
                "record:{}.{}.{}",
                encode_path_component(schema_name),
                encode_path_component(table_name),
                encode_path_component(key)
            ),
        }
    }

    fn overlaps(&self, other: &DmlLockScope) -> bool {
        let same_schema = self.schema_name() == other.schema_name();
        if !same_schema {
            return false;
        }

        match (self.table_name(), other.table_name()) {
            (None, _) | (_, None) => true,
            (Some(left_table), Some(right_table)) if left_table != right_table => false,
            (Some(_), Some(_)) => match (self, other) {
                (DmlLockScope::Table { .. }, _) | (_, DmlLockScope::Table { .. }) => true,
                (
                    DmlLockScope::Partition {
                        partition_id: left, ..
                    },
                    DmlLockScope::Partition {
                        partition_id: right,
                        ..
                    },
                ) => left == right,
                (
                    DmlLockScope::Record { key: left, .. },
                    DmlLockScope::Record { key: right, .. },
                ) => left == right,
                // Record scopes do not currently carry partition identity, so a
                // partition-vs-record overlap is conservatively treated as a table conflict.
                _ => true,
            },
        }
    }
}

/// Outcome of a DML lock acquisition attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockOutcome {
    /// Lock was acquired successfully
    Acquired { lease: PartitionLease },
    /// Lock is held by another pod
    Held { holder: String, expires_at: i64 },
    /// Lock request conflicted with existing lock
    Conflict,
    /// This pod was fenced (newer generation owns the lock)
    Fenced { latest: PartitionLease },
}

/// Prometheus label for a [`LockOutcome`] (A9). Kept low-cardinality:
/// `acquired` / `conflict` / `held` / `fenced`.
fn dml_lock_outcome_label(outcome: &LockOutcome) -> &'static str {
    match outcome {
        LockOutcome::Acquired { .. } => "acquired",
        LockOutcome::Conflict => "conflict",
        LockOutcome::Held { .. } => "held",
        LockOutcome::Fenced { .. } => "fenced",
    }
}

////////////////////////////////////////////////////////////////////////////////
// Lease Types
////////////////////////////////////////////////////////////////////////////////

/// Current wall-clock milliseconds since the Unix epoch.
fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Error returned by [`validate_fencing`] when a write carries a fencing
/// generation older than the current durable lease generation.
///
/// This is the storage-boundary split-brain defense (A6): a writer that
/// acquired a lease at `write_generation` but has since been displaced (a
/// takeover committed a strictly-higher generation) must be rejected
/// *before* it commits, so two pods can never both believe they own a
/// resource.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "stale fencing generation: write generation {write_generation} is older than current durable generation {current_generation}"
)]
pub struct FencingError {
    /// Generation the writer is carrying (from the DML lock guard /
    /// `FlushParameters::fencing_generation`).
    pub write_generation: u64,
    /// Generation currently published on the durable lease log for the
    /// resource.
    pub current_generation: u64,
}

/// Validate a write's fencing generation against the current durable lease
/// generation (A6 boundary contract).
///
/// The DML write request carries the guard's generation on
/// `WriteIntent::fencing_generation` (populated via
/// `DmlLockGuard::lease_generation()`). A storage writer that consumes it
/// MUST call this before committing the mutation:
///
/// - `write_generation = None` → legacy / unfenced write: **allowed**
///   (`Ok`). The fence is opt-in per write so non-DML writes are
///   unaffected.
/// - `write_generation = Some(g)` with `g >= current_generation` → the
///   write's lease is current or newer: **allowed** (`Ok`).
/// - `write_generation = Some(g)` with `g < current_generation` → a
///   takeover has displaced this writer: **rejected** (`Err`).
///
/// `current_generation` is the generation read from the durable lease log
/// for the resource at commit time. Wiring that read into each storage
/// engine's flush path (and threading the token from `WriteIntent` through
/// the WAL intermediate to the writer) is the follow-up enforcement TD;
/// this function is the pure, unit-tested contract they call.
pub fn validate_fencing(
    write_generation: Option<u64>,
    current_generation: u64,
) -> std::result::Result<(), FencingError> {
    match write_generation {
        None => Ok(()),
        Some(g) if g < current_generation => Err(FencingError {
            write_generation: g,
            current_generation,
        }),
        Some(_) => Ok(()),
    }
}

/// A durable lease granting **one** pod write authority over a single resource
/// until `expires_at_ms`.
///
/// **Mixed-read-safe migration**: Old-format leases (with `tenant_id` and
/// `collection_id` only) are automatically migrated to the new `resource_key`
/// format on read. New leases always use `resource_key`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartitionLease {
    /// Unified resource key (new format). Present in all new leases.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_key: Option<ResourceKey>,

    /// Legacy: Tenant owning the partition (kept for backward compatibility).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,

    /// Legacy: Collection (the partition) within the tenant (kept for backward compatibility).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_id: Option<String>,

    /// Pod that holds the lease (opaque identifier, matching the registry's
    /// `PrimaryPod.pod` convention).
    pub holder_pod: String,

    /// Monotonic fencing generation. A takeover commits a strictly-higher
    /// generation so the displaced owner's next write is fenced.
    pub generation: u64,

    /// Wall-clock ms when the lease was last (re)acquired.
    pub acquired_at_ms: i64,

    /// Wall-clock ms at which the lease lapses if not renewed.
    pub expires_at_ms: i64,

    /// Format version for migration detection.
    /// - None or 0: legacy format (tenant_id + collection_id only)
    /// - 1: unified resource key format
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format_version: Option<u32>,

    /// Tombstone flag (F7): `true` when the prior holder explicitly
    /// relinquished the lease via [`PartitionLeaseStore::release_with_key`].
    ///
    /// A released lease is free for the taking, but it still occupies a
    /// generation slot in the fence log so the next acquirer claims a
    /// strictly-higher generation (preserving the monotonic fence). Old
    /// readers default this to `false` (mixed-read-safe).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub released: bool,
}

impl PartitionLease {
    /// Create a new lease with a unified resource key.
    pub fn with_key(
        key: ResourceKey,
        holder_pod: impl Into<String>,
        generation: u64,
        acquired_at_ms: i64,
        expires_at_ms: i64,
    ) -> Self {
        Self {
            resource_key: Some(key),
            tenant_id: None,
            collection_id: None,
            holder_pod: holder_pod.into(),
            generation,
            acquired_at_ms,
            expires_at_ms,
            format_version: Some(1),
            released: false,
        }
    }

    /// Create a legacy-format lease (for backward compatibility in tests/migration).
    pub fn legacy(
        tenant_id: impl Into<String>,
        collection_id: impl Into<String>,
        holder_pod: impl Into<String>,
        generation: u64,
        acquired_at_ms: i64,
        expires_at_ms: i64,
    ) -> Self {
        Self {
            resource_key: None,
            tenant_id: Some(tenant_id.into()),
            collection_id: Some(collection_id.into()),
            holder_pod: holder_pod.into(),
            generation,
            acquired_at_ms,
            expires_at_ms,
            format_version: None,
            released: false,
        }
    }

    /// Ensure this lease is migrated to the new format.
    /// If it's legacy format, creates a new ResourceKey and returns a migrated copy.
    pub fn ensure_migrated(&self) -> Cow<'_, PartitionLease> {
        if self.resource_key.is_some() || self.format_version == Some(1) {
            // Already migrated
            return Cow::Borrowed(self);
        }

        // Migrate from legacy format — only when the legacy identity is
        // present. A malformed legacy lease (no identity) is left as-is
        // rather than panicking.
        let (Some(tenant_id), Some(collection_id)) =
            (self.tenant_id.as_deref(), self.collection_id.as_deref())
        else {
            return Cow::Borrowed(self);
        };

        let migrated = Self {
            resource_key: Some(ResourceKey::legacy_collection(tenant_id, collection_id)),
            tenant_id: Some(tenant_id.to_string()),
            collection_id: Some(collection_id.to_string()),
            holder_pod: self.holder_pod.clone(),
            generation: self.generation,
            acquired_at_ms: self.acquired_at_ms,
            expires_at_ms: self.expires_at_ms,
            format_version: Some(1),
            released: self.released,
        };

        Cow::Owned(migrated)
    }

    /// Get the resource key, migrating if necessary.
    pub fn key(&self) -> Cow<'_, ResourceKey> {
        let migrated = self.ensure_migrated();
        match &migrated {
            Cow::Borrowed(lease) => match lease.resource_key.as_ref() {
                // Normal path: the lease carries its key.
                Some(k) => Cow::Borrowed(k),
                // Malformed lease with no key and no migratable identity:
                // synthesize an empty legacy key rather than panic.
                None => Cow::Owned(ResourceKey::legacy_collection(
                    lease.tenant_id.as_deref().unwrap_or(""),
                    lease.collection_id.as_deref().unwrap_or(""),
                )),
            },
            Cow::Owned(lease) => match lease.resource_key.as_ref() {
                Some(k) => Cow::Owned(k.clone()),
                None => Cow::Owned(ResourceKey::legacy_collection(
                    lease.tenant_id.as_deref().unwrap_or(""),
                    lease.collection_id.as_deref().unwrap_or(""),
                )),
            },
        }
    }

    /// Get the tenant ID (from resource_key if available, otherwise from legacy field).
    pub fn get_tenant_id(&self) -> String {
        self.ensure_migrated().key().tenant_id.clone()
    }

    /// Get the resource ID (collection_id in legacy format, or from resource_key).
    pub fn get_resource_id(&self) -> String {
        let key = self.key();
        match &key.resource_id {
            ResourceIdentifier::Single(s) => s.clone(),
            ResourceIdentifier::Composite(parts) => parts.join("."),
            ResourceIdentifier::Hierarchical { parent, child } => format!("{}.{}", parent, child),
        }
    }

    /// Whether the lease has lapsed at `now_ms`.
    pub fn is_expired(&self, now_ms: i64) -> bool {
        now_ms >= self.expires_at_ms
    }

    /// Whether `pod` holds this lease and it is still valid at `now_ms`.
    pub fn is_valid_for(&self, pod: &str, now_ms: i64) -> bool {
        self.holder_pod == pod && !self.is_expired(now_ms)
    }

    /// Whether this is a legacy-format lease (before migration).
    pub fn is_legacy(&self) -> bool {
        self.resource_key.is_none() && self.format_version.is_none()
    }

    /// Whether this is a new-format lease (with ResourceKey).
    pub fn is_new_format(&self) -> bool {
        self.resource_key.is_some() || self.format_version == Some(1)
    }
}

/// Outcome of an acquire/renew attempt against the durable lease log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseOutcome {
    /// This pod now holds the lease (freshly acquired, took over an expired one,
    /// or renewed its own).
    Acquired(PartitionLease),
    /// Another pod holds a still-valid lease — this pod is **not** the owner.
    Held {
        /// The valid lease currently held by the other pod.
        by: PartitionLease,
    },
    /// **Fenced**: a newer generation already owns the partition (this pod lost
    /// the version CAS, or carried a stale generation). The latest durable lease
    /// is returned so the caller can route to / step down for the real owner.
    Fenced {
        /// The lease that actually won.
        latest: PartitionLease,
    },
}

/// Object-store home for generation-fenced partition leases. One fenced manifest
/// log per `(tenant, collection)` under `{prefix}/{tenant}/{collection}/_manifests/`.
///
/// `prefix` must be rooted at the operator control plane (the lease registry is
/// control-plane metadata — *who owns what*, never tenant data), e.g.
/// `DrPathBuilder::operator_subprefix("leases")`.
pub struct PartitionLeaseStore {
    store: ProximaObjectStore,
    prefix: String,
}

impl PartitionLeaseStore {
    /// Build from an already-open store and the operator-rooted lease prefix.
    pub fn new(store: ProximaObjectStore, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        while prefix.ends_with('/') {
            prefix.pop();
        }
        Self { store, prefix }
    }

    /// Build from an object-store base URL (e.g. `s3://bucket`, `memory:///`).
    pub fn from_url(base_url: &str, prefix: impl Into<String>) -> Result<Self> {
        let store = ProximaObjectStore::from_url(base_url)
            .with_context(|| format!("opening object store at {base_url}"))?;
        Ok(Self::new(store, prefix))
    }

    /// The fenced manifest committer for one partition's lease log.
    fn committer(&self, tenant_id: &str, collection_id: &str) -> ManifestCommitter {
        ManifestCommitter::new(
            self.store.clone(),
            format!("{}/{tenant_id}/{collection_id}/_manifests", self.prefix),
        )
        .with_legacy_encoding(
            proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
        )
    }

    /// Read the current lease for a partition with its pointer version + fencing
    /// generation, or `None` if the partition has never been leased.
    pub async fn read(
        &self,
        tenant_id: &str,
        collection_id: &str,
    ) -> Result<Option<(u64, PartitionLease)>> {
        let committer = self.committer(tenant_id, collection_id);
        match committer
            .latest_version()
            .await
            .with_context(|| format!("reading lease log {tenant_id}/{collection_id}"))?
        {
            Some(version) => {
                let (_generation, bytes) =
                    committer.read_fenced(version).await.with_context(|| {
                        format!("reading lease {tenant_id}/{collection_id}@{version}")
                    })?;
                let lease: PartitionLease = serde_json::from_slice(&bytes)
                    .with_context(|| format!("decoding lease {tenant_id}/{collection_id}"))?;
                Ok(Some((version, lease)))
            }
            None => Ok(None),
        }
    }

    /// Acquire — or renew, or take over an expired — the lease on
    /// `(tenant_id, collection_id)` for `holder_pod`, valid for `lease_ms` from
    /// `now_ms`.
    ///
    /// **Legacy compatibility**: This method converts to ResourceKey and delegates
    /// to `acquire_with_key`. New code should use `acquire_with_key` directly.
    ///
    /// Returns [`LeaseOutcome::Held`] when a *live* lease belongs to another pod,
    /// [`LeaseOutcome::Fenced`] when this attempt lost the fenced CAS to a
    /// concurrent acquirer, and [`LeaseOutcome::Acquired`] when this pod holds it.
    pub async fn acquire(
        &self,
        tenant_id: &str,
        collection_id: &str,
        holder_pod: &str,
        now_ms: i64,
        lease_ms: i64,
    ) -> Result<LeaseOutcome> {
        self.acquire_via_key(tenant_id, collection_id, holder_pod, now_ms, lease_ms)
            .await
    }

    ////////////////////////////////////////////////////////////////////////////////
    // ResourceKey-based methods (multi-modality support)
    ////////////////////////////////////////////////////////////////////////////////

    /// The fenced manifest committer for a resource identified by ResourceKey.
    ///
    /// Collection routing is **total, never silently divergent** (CLAUDE.md
    /// mandate #8, mixed-read-safe migration). A collection's lease lives on the
    /// legacy `{tenant}/{collection}/_manifests` log so the old
    /// `acquire(tenant, collection)` API and the new `ResourceKey` API address
    /// the SAME log — two independent fences for one collection is split brain.
    /// The legacy log is keyed by `(tenant, collection)` only, so the canonical
    /// collection shape is namespace-less + single-component. Any other
    /// collection-domain key (namespaced, or composite/multi-part) would land on
    /// a *different* `to_path()` log and fork the fence; reject it **loudly**
    /// rather than diverge silently. No current caller produces such a key — this
    /// guards against one being introduced. Non-collection resource types
    /// (table/schema/graph/…) are genuinely distinct resources and keep their own
    /// `to_path()` log.
    fn committer_for_key(&self, key: &ResourceKey) -> Result<ManifestCommitter> {
        if key.resource_type == ResourceType::Collection {
            let single = match &key.resource_id {
                ResourceIdentifier::Single(id) => Some(id.as_str()),
                ResourceIdentifier::Composite(parts) if parts.len() == 1 => Some(parts[0].as_str()),
                _ => None,
            };
            return match (key.namespace_id.as_deref(), single) {
                (None, Some(collection_id)) => Ok(self.committer(&key.tenant_id, collection_id)),
                _ => anyhow::bail!(
                    "ambiguous collection lease key (tenant={}, namespace={:?}, id={:?}): a \
                     collection lease must address the legacy \
                     `{{tenant}}/{{collection}}/_manifests` log; a namespaced or composite \
                     collection key would fork the generation fence",
                    key.tenant_id,
                    key.namespace_id,
                    key.resource_id
                ),
            };
        }

        let path = format!("{}/{}", self.prefix, key.to_path());
        Ok(
            ManifestCommitter::new(self.store.clone(), path).with_legacy_encoding(
                proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
            ),
        )
    }

    /// Read the current lease for a resource by its ResourceKey.
    /// Handles both legacy and new-format leases (mixed-read-safe).
    pub async fn read_key(&self, key: &ResourceKey) -> Result<Option<(u64, PartitionLease)>> {
        let committer = self.committer_for_key(key)?;
        let key_desc = format!("{}:{}", key.tenant_id, key.resource_type.name());
        match committer
            .latest_version()
            .await
            .with_context(|| format!("reading lease log {key_desc}"))?
        {
            Some(version) => {
                let (_generation, bytes) = committer
                    .read_fenced(version)
                    .await
                    .with_context(|| format!("reading lease {key_desc}@{version}"))?;
                let lease: PartitionLease = serde_json::from_slice(&bytes)
                    .with_context(|| format!("decoding lease {key_desc}"))?;
                Ok(Some((version, lease)))
            }
            None => Ok(None),
        }
    }

    /// Acquire — or renew, or take over an expired — the lease on a resource
    /// identified by `key` for `holder_pod`, valid for `lease_ms` from `now_ms`.
    ///
    /// This is the multi-modality entry point that works for all resource types
    /// (collections, tables, graphs, ML models, etc.).
    ///
    /// Returns [`LeaseOutcome::Held`] when a *live* lease belongs to another pod,
    /// [`LeaseOutcome::Fenced`] when this attempt lost the fenced CAS to a
    /// concurrent acquirer, and [`LeaseOutcome::Acquired`] when this pod holds it.
    pub async fn acquire_with_key(
        &self,
        key: &ResourceKey,
        holder_pod: &str,
        now_ms: i64,
        lease_ms: i64,
    ) -> Result<LeaseOutcome> {
        let committer = self.committer_for_key(key)?;
        let key_desc = format!("{}:{}", key.tenant_id, key.resource_type.name());

        // Decide the parent version + the generation to claim, from the current
        // durable state.
        let (parent, generation) = match self.read_key(key).await? {
            // Fresh resource — generation 1 (manifest versions start at 0).
            None => (None, 1),
            Some((version, lease)) => {
                // A released tombstone (F7) means the prior holder explicitly
                // relinquished: the resource is free. Claim a strictly-higher
                // generation so the monotonic fence stays intact (the tombstone
                // occupies `lease.generation`; an object-store delete would have
                // reset it to 1 and collapsed the fence).
                if lease.released {
                    // `saturating_add` rather than `+` to honor the no-panic
                    // policy (matches the manifest version counter's `checked_add`);
                    // u64 generation overflow is unreachable in practice.
                    (Some(version), lease.generation.saturating_add(1))
                } else if lease.holder_pod.as_str() == holder_pod {
                    // We already own it → renew at the SAME generation
                    (Some(version), lease.generation)
                } else if lease.is_expired(now_ms) {
                    // Take over a dead owner → strictly-higher generation
                    (Some(version), lease.generation.saturating_add(1))
                } else {
                    // A live lease belongs to someone else — we are not the owner.
                    return Ok(LeaseOutcome::Held { by: lease });
                }
            }
        };

        let lease = PartitionLease::with_key(
            key.clone(),
            holder_pod,
            generation,
            now_ms,
            now_ms.saturating_add(lease_ms),
        );
        let payload = serde_json::to_vec(&lease).context("encoding partition lease")?;

        match committer
            .commit_fenced(parent, generation, bytes::Bytes::from(payload))
            .await
            .with_context(|| format!("committing lease {key_desc}"))?
        {
            CommitOutcome::Committed(_) => Ok(LeaseOutcome::Acquired(lease)),
            // Lost the version CAS or was fenced by a higher generation — re-read
            // to report who actually owns the resource now.
            CommitOutcome::Conflict { .. } => match self.read_key(key).await? {
                Some((_, latest)) => Ok(LeaseOutcome::Fenced { latest }),
                // Extremely unlikely (the conflicting object vanished) — surface
                // our own attempt as the latest so the caller does not own it.
                None => Ok(LeaseOutcome::Fenced { latest: lease }),
            },
        }
    }

    /// Begin a new writer process incarnation. Unlike lease renewal, this always
    /// advances the durable generation, including when the same pod id held the
    /// previous lease. That makes `{generation, process-local sequence}` a total
    /// order across process restarts without adding I/O per WAL batch.
    pub async fn begin_writer_incarnation_with_key(
        &self,
        key: &ResourceKey,
        holder_pod: &str,
        now_ms: i64,
        lease_ms: i64,
    ) -> Result<LeaseOutcome> {
        let committer = self.committer_for_key(key)?;
        let (parent, generation) = match self.read_key(key).await? {
            None => (None, 1),
            Some((version, lease)) => {
                if !lease.released && !lease.is_expired(now_ms) && lease.holder_pod != holder_pod {
                    return Ok(LeaseOutcome::Held { by: lease });
                }
                (Some(version), lease.generation.saturating_add(1))
            }
        };
        let lease = PartitionLease::with_key(
            key.clone(),
            holder_pod,
            generation,
            now_ms,
            now_ms.saturating_add(lease_ms),
        );
        let payload = serde_json::to_vec(&lease).context("encoding writer incarnation lease")?;
        match committer
            .commit_fenced(parent, generation, bytes::Bytes::from(payload))
            .await?
        {
            CommitOutcome::Committed(_) => Ok(LeaseOutcome::Acquired(lease)),
            CommitOutcome::Conflict { .. } => match self.read_key(key).await? {
                Some((_, latest)) => Ok(LeaseOutcome::Fenced { latest }),
                None => Ok(LeaseOutcome::Fenced { latest: lease }),
            },
        }
    }

    /// Legacy compatibility: convert (tenant_id, collection_id) to ResourceKey
    /// and delegate to acquire_with_key.
    async fn acquire_via_key(
        &self,
        tenant_id: &str,
        collection_id: &str,
        holder_pod: &str,
        now_ms: i64,
        lease_ms: i64,
    ) -> Result<LeaseOutcome> {
        let key = ResourceKey::legacy_collection(tenant_id, collection_id);
        self.acquire_with_key(&key, holder_pod, now_ms, lease_ms)
            .await
    }

    /// Explicitly relinquish a lease (F7) by publishing a **released
    /// tombstone** at a strictly-higher generation.
    ///
    /// This is the fast-path counterpart to natural TTL expiry: instead of
    /// waiting for `expires_at_ms`, the holder publishes a tombstone so the
    /// next acquirer can take over immediately. The tombstone occupies a
    /// generation slot (committed at `lease.generation + 1`), which is
    /// load-bearing: any in-flight write still carrying the old
    /// `lease.generation` is fenced, and the monotonic-generation invariant
    /// survives. Object-store *deleting* the manifest is deliberately avoided
    /// — it would make the next acquire see an empty log and claim generation
    /// 1, collapsing the fence.
    ///
    /// Best-effort and fail-open: if the holder does not match, the lease is
    /// already released, or the object-store CAS/errors, this returns `Ok`
    /// and the lease simply expires at its TTL (the crash-safe fallback).
    pub async fn release_with_key(
        &self,
        key: &ResourceKey,
        holder_pod: &str,
        now_ms: i64,
    ) -> Result<()> {
        let key_desc = format!("{}:{}", key.tenant_id, key.resource_type.name());

        // Read the current durable state. A read failure is fail-open: leave
        // the lease to expire at TTL rather than surfacing a transient error.
        let Some((version, lease)) = (match self.read_key(key).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    target: "partition_lease_release",
                    resource = %key_desc,
                    error = %e,
                    "release read failed; falling back to TTL expiry"
                );
                return Ok(());
            }
        }) else {
            // Nothing durable to release.
            return Ok(());
        };

        // Only the current holder may release, and only a live (non-tombstoned)
        // lease. Anything else is a stale no-op.
        if lease.holder_pod != holder_pod || lease.released {
            return Ok(());
        }

        // Publish the tombstone at generation+1 so the holder's own in-flight
        // writes at `lease.generation` are fenced (strict less-than).
        // `saturating_add` honors the no-panic policy (u64 overflow unreachable).
        let new_generation = lease.generation.saturating_add(1);
        let tombstone = PartitionLease {
            released: true,
            generation: new_generation,
            acquired_at_ms: now_ms,
            expires_at_ms: now_ms,
            ..lease.clone()
        };
        let payload = serde_json::to_vec(&tombstone).context("encoding release tombstone")?;

        match self
            .committer_for_key(key)?
            .commit_fenced(Some(version), new_generation, bytes::Bytes::from(payload))
            .await
        {
            // Tombstone published, or we lost the CAS to a concurrent takeover —
            // either way there is nothing left for us to release.
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::warn!(
                    target: "partition_lease_release",
                    resource = %key_desc,
                    error = %e,
                    "durable release commit failed; falling back to TTL expiry"
                );
                Ok(())
            }
        }
    }
}

/// Ties the durable [`PartitionLeaseStore`] to the in-memory
/// [`PrimaryPodRegistry`] that [`consult_for_write`] reads on the hot path.
///
/// The registry stays the fast, lock-free routing cache; this manager keeps it
/// *truthful* against the durable lease — it assigns the binding to this pod on a
/// successful acquire, and **steps down** (re-points the binding at the new
/// owner, never unassigns) when a renewal reveals the lease was lost. Re-pointing
/// rather than clearing is the load-bearing safety property: an unassigned
/// binding makes `consult_for_write` return `Allow`, which would let a displaced
/// pod accept writes — a split brain. Pointing at the new owner instead makes it
/// return `Misrouted`, so the displaced pod fails closed.
pub struct PartitionLeaseManager {
    store: Arc<PartitionLeaseStore>,
    registry: Arc<PrimaryPodRegistry>,
    self_pod_id: String,
    lease_ms: i64,
    /// Strategy registry for type-specific behavior (TTL, partitioning, etc.)
    strategies: DashMap<ResourceType, Arc<dyn ResourceStrategy>>,
    /// Full-key leases held by this pod that are not represented in the legacy
    /// `(tenant, collection)` primary-pod registry.
    held_resource_keys: DashMap<String, ResourceKey>,
    /// Incarnation bumps already performed by this process. A new manager on
    /// restart starts empty and therefore performs a fresh durable CAS bump.
    writer_incarnations: DashMap<String, (u64, i64)>,
}

fn global_manager_slot() -> &'static StdRwLock<Option<Arc<PartitionLeaseManager>>> {
    static SLOT: OnceLock<StdRwLock<Option<Arc<PartitionLeaseManager>>>> = OnceLock::new();
    SLOT.get_or_init(|| StdRwLock::new(None))
}

pub fn install_global_partition_lease_manager(manager: Arc<PartitionLeaseManager>) {
    if let Ok(mut slot) = global_manager_slot().write() {
        *slot = Some(manager);
    }
}

pub fn global_partition_lease_manager() -> Option<Arc<PartitionLeaseManager>> {
    global_manager_slot()
        .read()
        .ok()
        .and_then(|slot| slot.clone())
}

impl PartitionLeaseManager {
    /// Build a manager for `self_pod_id` issuing leases valid for `lease_ms`.
    pub fn new(
        store: Arc<PartitionLeaseStore>,
        registry: Arc<PrimaryPodRegistry>,
        self_pod_id: impl Into<String>,
        lease_ms: i64,
    ) -> Self {
        Self {
            store,
            registry,
            self_pod_id: self_pod_id.into(),
            lease_ms,
            strategies: DashMap::new(),
            held_resource_keys: DashMap::new(),
            writer_incarnations: DashMap::new(),
        }
    }

    /// Register a strategy for a resource type.
    ///
    /// This enables type-specific behavior (TTL, partitioning, etc.).
    /// Multiple strategies can be registered; the strategy for a given
    /// resource type is looked up on each acquire/renew.
    pub fn register_strategy(&self, strategy: Arc<dyn ResourceStrategy>) {
        self.strategies.insert(strategy.resource_type(), strategy);
    }

    /// Get the TTL for a resource type (from strategy or default).
    fn get_ttl_for_resource(&self, resource_type: &ResourceType) -> i64 {
        self.strategies
            .get(resource_type)
            .map(|s| s.lease_ttl_secs() as i64 * 1000) // Convert seconds to ms
            .unwrap_or(self.lease_ms) // Default to configured lease_ms
    }

    /// Get a strategy by resource type (if registered).
    pub fn get_strategy(&self, resource_type: &ResourceType) -> Option<Arc<dyn ResourceStrategy>> {
        self.strategies.get(resource_type).map(|s| s.clone())
    }

    /// This pod's identity.
    pub fn self_pod_id(&self) -> &str {
        &self.self_pod_id
    }

    /// Attempt to become (or stay) the primary for `(tenant, collection)` at
    /// `now_ms`. On success the registry binding points at this pod; otherwise it
    /// is pointed at the actual owner so the write path fails closed. Returns
    /// whether this pod now owns the partition.
    pub async fn acquire(&self, tenant_id: &str, collection_id: &str, now_ms: i64) -> Result<bool> {
        let outcome = self
            .store
            .acquire(
                tenant_id,
                collection_id,
                &self.self_pod_id,
                now_ms,
                self.lease_ms,
            )
            .await?;
        Ok(self.reconcile(tenant_id, collection_id, outcome))
    }

    /// Ensure this pod owns `(tenant, collection)` with at most one object-store
    /// round-trip per pod per collection (lease = latency optimization).
    ///
    /// Fast path: if the shared registry already binds the partition to this pod,
    /// the renew loop keeps the durable lease warm, so return `true` without I/O.
    /// Slow path (no binding, or a foreign owner): [`acquire`] (CAS) + reconcile,
    /// which repoints the shared registry at the true owner so `consult_for_write`
    /// is durably backed rather than empty-after-restart. Returns whether this pod
    /// owns the partition. Callers fail-open on `Err` (transient object-store
    /// blip), matching the lease stack's bootstrap posture; the storage-write
    /// fence (A6) is the boundary backstop for the residual handoff race.
    pub async fn ensure_owned(
        &self,
        tenant_id: &str,
        collection_id: &str,
        now_ms: i64,
    ) -> Result<bool> {
        if self
            .registry
            .lookup(tenant_id, collection_id)
            .map(|binding| binding.pod == self.self_pod_id)
            .unwrap_or(false)
        {
            return Ok(true);
        }
        self.acquire(tenant_id, collection_id, now_ms).await
    }

    /// Ensure this process has a durable, incarnation-unique epoch for the
    /// canonical collection UUID and publish it to the WAL token provider.
    pub async fn begin_writer_incarnation(
        &self,
        tenant_id: &str,
        canonical_collection_id: &str,
        now_ms: i64,
    ) -> Result<bool> {
        let cache_key = format!("{tenant_id}/{canonical_collection_id}");
        if let Some(state) = self.writer_incarnations.get(&cache_key)
            && now_ms < state.1
        {
            crate::storage::persistence::write_ahead_log::RecoveryTokenProvider::global()
                .register_incarnation(tenant_id, canonical_collection_id, state.0, state.1);
            return Ok(true);
        }

        let key = ResourceKey::legacy_collection(tenant_id, canonical_collection_id);
        let outcome = self
            .store
            .begin_writer_incarnation_with_key(&key, &self.self_pod_id, now_ms, self.lease_ms)
            .await?;
        match outcome {
            LeaseOutcome::Acquired(lease) => {
                self.writer_incarnations
                    .insert(cache_key, (lease.generation, lease.expires_at_ms));
                crate::storage::persistence::write_ahead_log::RecoveryTokenProvider::global()
                    .register_incarnation(
                        tenant_id,
                        canonical_collection_id,
                        lease.generation,
                        lease.expires_at_ms,
                    );
                Ok(true)
            }
            LeaseOutcome::Held { .. } | LeaseOutcome::Fenced { .. } => Ok(false),
        }
    }

    /// Read the current durable leaseholder for `(tenant, collection)` — the
    /// storage-write fence's authority. Returns the lease record (manifest
    /// version + generation + holder) or `None` if the partition has never been
    /// leased. This is the read half of the A6 boundary fence (the write half is
    /// [`validate_fencing`]); it delegates to the durable lease log, not the
    /// in-memory registry, so it reflects ground truth across pods.
    pub async fn current_holder(
        &self,
        tenant_id: &str,
        collection_id: &str,
    ) -> Result<Option<(u64, PartitionLease)>> {
        self.store.read(tenant_id, collection_id).await
    }

    /// A6 storage-write fence decision: is THIS pod fenced out of writing
    /// `(tenant, collection)` to shared storage at `now_ms`?
    ///
    /// Returns `Ok(true)` iff a *live* lease (not released, not expired) is held by a
    /// **different** pod — i.e. a takeover has displaced this pod, so a flush from
    /// it would publish stale data and must be rejected. Durable read errors are
    /// propagated so the storage fence can fail closed when enforcement is enabled.
    ///
    /// NOTE: this is the fence *primitive*. Wiring it at the live storage-write
    /// boundary is gated on the flush-path convergence (the server vector-flush
    /// path is currently stubbed; see the A6 TD / LEASE_FENCE_REREVIEW doc).
    pub async fn is_fenced_out(
        &self,
        tenant_id: &str,
        collection_id: &str,
        now_ms: i64,
    ) -> Result<bool> {
        match self.current_holder(tenant_id, collection_id).await {
            Ok(Some((_version, lease))) => Ok(!lease.released
                && !lease.is_expired(now_ms)
                && lease.holder_pod != self.self_pod_id),
            Ok(None) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Acquire a lease for a resource identified by its ResourceKey.
    ///
    /// This is the multi-modality entry point that uses the registered strategy
    /// to determine the appropriate TTL for the resource type.
    ///
    /// Returns whether this pod now owns the resource.
    pub async fn acquire_with_key(&self, key: &ResourceKey, now_ms: i64) -> Result<bool> {
        let ttl_ms = self.get_ttl_for_resource(&key.resource_type);
        let outcome = self
            .store
            .acquire_with_key(key, &self.self_pod_id, now_ms, ttl_ms)
            .await?;

        // A legacy collection key reconciles into the (tenant, collection)
        // registry; everything else is tracked by full ResourceKey. Match the
        // single-component id directly (no-panic policy — fall through to the
        // generalized path rather than `unreachable!` if the shape ever differs).
        if Self::is_legacy_collection_key(key)
            && let ResourceIdentifier::Single(collection_id) = &key.resource_id
        {
            return Ok(self.reconcile(&key.tenant_id, collection_id, outcome));
        }

        let identity = Self::resource_key_identity(key);
        match outcome {
            LeaseOutcome::Acquired(_) => {
                self.held_resource_keys.insert(identity, key.clone());
                Ok(true)
            }
            LeaseOutcome::Held { .. } | LeaseOutcome::Fenced { .. } => {
                self.held_resource_keys.remove(&identity);
                Ok(false)
            }
        }
    }

    /// Track a resource lease this pod holds so [`renew_held`] keeps it warm.
    ///
    /// `DmlLockService` acquires the durable lease via the store directly (it
    /// needs the `LeaseOutcome`/generation to build its guard), bypassing
    /// [`acquire_with_key`] — the only other path that populates the renewal set.
    /// Without this registration a DML lock's durable lease silently lapses at
    /// its TTL while still held, opening a takeover-eligible split-brain window.
    /// The matching un-track happens in [`release_with_key`], which removes the
    /// same identity.
    pub fn track_held_key(&self, key: &ResourceKey) {
        self.held_resource_keys
            .insert(Self::resource_key_identity(key), key.clone());
    }

    /// Explicitly relinquish a generalized resource lease (F7): drop the
    /// in-memory held-key entry and publish a durable released-tombstone
    /// (best-effort; TTL expiry remains the crash-safe fallback). Returns
    /// whether this pod held the lease.
    ///
    /// Legacy `(tenant, collection)` leases are released via the same store
    /// primitive but are not tracked in `held_resource_keys`, so only the
    /// durable tombstone is published for them.
    pub async fn release_with_key(&self, key: &ResourceKey, now_ms: i64) -> Result<bool> {
        let identity = Self::resource_key_identity(key);
        let was_held = self.held_resource_keys.remove(&identity).is_some();
        // Best-effort durable tombstone; the store method is fail-open to TTL.
        self.store
            .release_with_key(key, &self.self_pod_id, now_ms)
            .await?;
        Ok(was_held)
    }

    /// Renew every lease this pod believes it holds. Legacy collection leases
    /// are renewed from the primary-pod registry; generalized resource leases
    /// are renewed from the full `ResourceKey` registry.
    pub async fn renew_held(&self, now_ms: i64) -> Result<usize> {
        let mut still_owned = 0;
        for (tenant_id, collection_id, binding) in self.registry.list() {
            if binding.pod != self.self_pod_id {
                continue; // not ours to renew
            }
            match self
                .store
                .acquire(
                    &tenant_id,
                    &collection_id,
                    &self.self_pod_id,
                    now_ms,
                    self.lease_ms,
                )
                .await
            {
                Ok(outcome) => {
                    if self.reconcile(&tenant_id, &collection_id, outcome) {
                        still_owned += 1;
                    }
                }
                // A transient object-store error: keep the binding and retry next
                // tick rather than spuriously step down (the lease has not lapsed
                // from our side yet).
                Err(e) => {
                    tracing::warn!(
                        tenant = %tenant_id,
                        collection = %collection_id,
                        error = %e,
                        "partition-lease renewal failed; retrying next interval"
                    );
                    crate::metrics::dml_lock_metrics::record_renewal_failure("collection");
                    still_owned += 1;
                }
            }
        }

        let incarnations = self
            .writer_incarnations
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect::<Vec<_>>();
        for (identity, (_epoch, _expiry)) in incarnations {
            let Some((tenant_id, collection_id)) = identity.split_once('/') else {
                continue;
            };
            let key = ResourceKey::legacy_collection(tenant_id, collection_id);
            match self
                .store
                .acquire_with_key(&key, &self.self_pod_id, now_ms, self.lease_ms)
                .await
            {
                Ok(LeaseOutcome::Acquired(lease)) => {
                    self.writer_incarnations
                        .insert(identity.clone(), (lease.generation, lease.expires_at_ms));
                    crate::storage::persistence::write_ahead_log::RecoveryTokenProvider::global()
                        .register_incarnation(
                            tenant_id,
                            collection_id,
                            lease.generation,
                            lease.expires_at_ms,
                        );
                    still_owned += 1;
                }
                Ok(LeaseOutcome::Held { .. } | LeaseOutcome::Fenced { .. }) => {
                    self.writer_incarnations.remove(&identity);
                    let _ = crate::storage::persistence::write_ahead_log::RecoveryTokenProvider::global()
                        .expire(collection_id);
                }
                Err(error) => {
                    // Keep the last durable expiry. Allocation fails closed once
                    // it passes, even if renewal keeps returning transient errors.
                    tracing::warn!(resource = %identity, %error, "writer incarnation renewal failed");
                }
            }
        }

        let held_keys = self
            .held_resource_keys
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect::<Vec<_>>();
        for (identity, key) in held_keys {
            let ttl_ms = self.get_ttl_for_resource(&key.resource_type);
            match self
                .store
                .acquire_with_key(&key, &self.self_pod_id, now_ms, ttl_ms)
                .await
            {
                Ok(LeaseOutcome::Acquired(_)) => {
                    still_owned += 1;
                }
                Ok(LeaseOutcome::Held { .. } | LeaseOutcome::Fenced { .. }) => {
                    self.held_resource_keys.remove(&identity);
                }
                Err(e) => {
                    tracing::warn!(
                        resource = %identity,
                        error = %e,
                        "resource lease renewal failed; retrying next interval"
                    );
                    crate::metrics::dml_lock_metrics::record_renewal_failure(
                        key.resource_type.name(),
                    );
                    still_owned += 1;
                }
            }
        }
        Ok(still_owned)
    }

    fn is_legacy_collection_key(key: &ResourceKey) -> bool {
        key.resource_type == ResourceType::Collection
            && key.namespace_id.is_none()
            && matches!(key.resource_id, ResourceIdentifier::Single(_))
    }

    fn resource_key_identity(key: &ResourceKey) -> String {
        key.to_path()
    }

    /// Fold a durable [`LeaseOutcome`] into the routing registry. Returns whether
    /// this pod owns the partition after reconciliation.
    fn reconcile(&self, tenant_id: &str, collection_id: &str, outcome: LeaseOutcome) -> bool {
        match outcome {
            LeaseOutcome::Acquired(_) => {
                self.registry.assign(
                    tenant_id,
                    collection_id,
                    self.self_pod_id.as_str(),
                    AssignmentReason::Failover,
                );
                true
            }
            // Someone else owns it (live lease, or we were fenced): reflect the
            // real owner so consult_for_write returns Misrouted, not Allow.
            LeaseOutcome::Held { by } | LeaseOutcome::Fenced { latest: by } => {
                self.registry.assign(
                    tenant_id,
                    collection_id,
                    by.holder_pod,
                    AssignmentReason::CatalogReplay,
                );
                false
            }
        }
    }

    /// Spawn a background task that renews this pod's held leases every
    /// `interval`. Returns the [`JoinHandle`](tokio::task::JoinHandle); the caller
    /// **owns** it and must `abort()` on shutdown (it is a cooperative tokio task,
    /// not an OS thread). Pick `interval <= lease_ms / 2` so a held lease never
    /// lapses between renewals.
    pub fn spawn_renew_loop(
        self: Arc<Self>,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await; // consume the immediate first tick
            loop {
                ticker.tick().await;
                if let Err(e) = self.renew_held(now_millis()).await {
                    tracing::warn!(error = %e, "partition-lease renew loop pass failed");
                }
            }
        })
    }

    /// Prune the stale tail of every lease manifest log this pod owns. Best-effort and
    /// idempotent: a transient object-store error is logged and the pass continues, and
    /// a log visited twice in one pass (a collection acquired via both the legacy and
    /// `ResourceKey` APIs shares one manifest log) is a harmless no-op the second time.
    /// See [`ManifestCommitter::prune_retention`] for the safety invariants.
    /// Unmigrated legacy logs are rejected without deletion and logged; migration
    /// is an explicit operator action, never performed by this maintenance loop.
    pub async fn prune_held(&self, keep_k: usize, min_age: std::time::Duration) -> Result<usize> {
        let mut total = 0usize;

        // Legacy (tenant, collection) leases this pod owns.
        for (tenant_id, collection_id, binding) in self.registry.list() {
            if binding.pod != self.self_pod_id {
                continue;
            }
            let committer = self.store.committer(&tenant_id, &collection_id);
            match committer.prune_retention(keep_k, min_age).await {
                Ok(n) => total += n,
                Err(e) => tracing::warn!(
                    tenant = %tenant_id,
                    collection = %collection_id,
                    error = %e,
                    "partition-lease manifest prune failed; retrying next interval"
                ),
            }
        }

        // Full-key leases (tables / graphs / models / collection-via-ResourceKey) this
        // pod owns. Collect first so no DashMap shard read-guard is held across an await.
        let held: Vec<ResourceKey> = self
            .held_resource_keys
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        for key in held {
            let path = key.to_path();
            let committer = match self.store.committer_for_key(&key) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        key = %path,
                        error = %e,
                        "prune: skipping unaddressable held key"
                    );
                    continue;
                }
            };
            match committer.prune_retention(keep_k, min_age).await {
                Ok(n) => total += n,
                Err(e) => tracing::warn!(
                    key = %path,
                    error = %e,
                    "partition-lease manifest prune failed; retrying next interval"
                ),
            }
        }

        Ok(total)
    }

    /// Spawn a low-priority background task that prunes the stale tail of each lease
    /// manifest log this pod owns, every `interval`. Drop the returned [`JoinHandle`]
    /// as a temporary so the task runs detached — matches [`spawn_renew_loop`].
    ///
    /// Only explicitly migrated versioned logs can prune. Their current lookup
    /// uses the authoritative head, independent of history count. Legacy history
    /// continues growing until migration; this loop must not reopen legacy slots.
    /// See [`ManifestCommitter::prune_retention`] for the safety invariants.
    ///
    /// **Scope / safety.** Only logs for resources this pod owns are pruned
    /// (`binding.pod == self_pod_id`, same gate as [`renew_held`]); the held set is
    /// re-read fresh each tick, so a key lost between ticks is not pruned again.
    /// Versioned pruning deletes only archives below the observed head, never its
    /// authority or current payload. Merely preserving the maximum legacy slot
    /// would NOT fence delayed writers; legacy pruning is therefore disabled.
    /// Detached, like the renew loop: process exit
    /// tears the runtime down and aborts the task; a graceful SIGTERM flush races only
    /// idempotent manifest deletes.
    pub fn spawn_prune_loop(
        self: Arc<Self>,
        interval: std::time::Duration,
        keep_k: usize,
        min_age: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await; // consume the immediate first tick
            loop {
                ticker.tick().await;
                if let Err(e) = self.prune_held(keep_k, min_age).await {
                    tracing::warn!(error = %e, "partition-lease prune loop pass failed");
                }
            }
        })
    }
}

/// Lease-aware write gate: `consult_for_write` over a registry the
/// [`PartitionLeaseManager`] keeps truthful. A thin convenience wrapper so call
/// sites read intent-first; the actual gating is unchanged (the registry binding
/// is the source of truth, now durably backed by the lease).
pub fn consult_for_write_leased(
    registry: &PrimaryPodRegistry,
    self_pod_id: &str,
    tenant_id: &str,
    collection_id: &str,
) -> WriteRoutingDecision {
    consult_for_write(registry, self_pod_id, tenant_id, collection_id)
}

////////////////////////////////////////////////////////////////////////////////
// DML Lock Service (Hierarchical Locking for Fine-Grained DML)
////////////////////////////////////////////////////////////////////////////////

/// Cluster-local typed error: a DML lock could not be acquired (conflict /
/// held / fenced). Kept cluster-local (no `core::errors` dependency — layering)
/// and mapped to `ProximaDBError::DmlLockConflict` at the services layer so
/// protocol handlers can recover it uniformly via an anyhow chain-walk.
#[derive(Debug, thiserror::Error)]
pub enum DmlLockAcquireError {
    #[error("DML lock conflict on {resource}")]
    Conflict { resource: String },
    #[error("DML lock held by {holder} on {resource}")]
    Held { resource: String, holder: String },
    #[error("DML lock fenced by {holder} on {resource}")]
    Fenced { resource: String, holder: String },
}

impl DmlLockAcquireError {
    /// `(resource, holder?)` for mapping into a typed protocol error.
    pub fn resource_holder(&self) -> (String, Option<String>) {
        match self {
            Self::Conflict { resource } => (resource.clone(), None),
            Self::Held { resource, holder } | Self::Fenced { resource, holder } => {
                (resource.clone(), Some(holder.clone()))
            }
        }
    }
}

/// Active DML lock (held by a pod for a specific scope with an intent).
#[derive(Debug, Clone)]
pub struct ActiveLock {
    pub tenant_id: String,
    pub namespace_id: Option<String>,
    pub scope: DmlLockScope,
    pub pod_id: String,
    pub intent: LockIntent,
    pub acquired_at_ms: i64,
    pub expires_at_ms: i64,
}

/// DML lock service for fine-grained locking (record-level, table-level, etc.).
///
/// This service provides hierarchical locking with intent compatibility checking.
/// It works with the PartitionLeaseManager to acquire durable leases for locks.
pub struct DmlLockService {
    lease_manager: Arc<PartitionLeaseManager>,
    self_pod_id: String,

    /// In-memory registry of active DML locks (fast path).
    /// Key is the scope's serialized form.
    active_locks: Arc<RwLock<HashMap<String, ActiveLock>>>,

    /// Shutdown signal receiver for the reconciliation loop.
    shutdown_rx: Option<tokio::sync::watch::Receiver<bool>>,

    /// Shutdown signal sender for the reconciliation loop.
    shutdown_tx: Option<tokio::sync::watch::Sender<bool>>,
}

/// Guard returned for a successfully acquired DML lock.
///
/// The guard owns the local lock registration. Call [`Self::release`] for an
/// explicit async release. `Drop` performs a best-effort local cleanup on the
/// current tokio runtime; durable object-store release is still a follow-up
/// gate, so the durable lease remains crash-safe by expiring naturally.
pub struct DmlLockGuard {
    service: Arc<DmlLockService>,
    tenant_id: String,
    namespace_id: Option<String>,
    scope: DmlLockScope,
    intent: LockIntent,
    lease_generation: u64,
    released: bool,
}

impl DmlLockGuard {
    /// Durable fencing generation associated with this lock acquisition.
    pub fn lease_generation(&self) -> u64 {
        self.lease_generation
    }

    /// Scope protected by this guard.
    pub fn scope(&self) -> &DmlLockScope {
        &self.scope
    }

    /// Intent protected by this guard.
    pub fn intent(&self) -> &LockIntent {
        &self.intent
    }

    /// Explicitly release the local in-memory lock registration.
    pub async fn release(mut self) {
        self.release_inner().await;
    }

    async fn release_inner(&mut self) {
        if self.released {
            return;
        }
        self.service
            .release_full(&self.tenant_id, self.namespace_id.as_deref(), &self.scope)
            .await;
        self.released = true;
    }
}

impl Drop for DmlLockGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;

        let service = self.service.clone();
        let tenant_id = self.tenant_id.clone();
        let namespace_id = self.namespace_id.clone();
        let scope = self.scope.clone();
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(async move {
                service
                    .release_full(&tenant_id, namespace_id.as_deref(), &scope)
                    .await;
            });
        }
    }
}

impl DmlLockService {
    /// Create a new DML lock service.
    pub fn new(lease_manager: Arc<PartitionLeaseManager>, self_pod_id: impl Into<String>) -> Self {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        Self {
            lease_manager,
            self_pod_id: self_pod_id.into(),
            active_locks: Arc::new(RwLock::new(HashMap::new())),
            shutdown_rx: Some(shutdown_rx),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Start the background reconciliation loop.
    ///
    /// This spawns a background task that periodically scans the in-memory
    /// lock registry and removes expired locks. The loop runs every
    /// `reconciliation_interval_ms` milliseconds.
    ///
    /// The loop can be stopped by calling `shutdown()` or by dropping the service.
    pub fn spawn_reconciliation_loop(
        &self,
        reconciliation_interval_ms: u64,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let active_locks = self.active_locks.clone();
        let mut shutdown_rx = self.shutdown_rx.as_ref()?.clone();

        Some(tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(reconciliation_interval_ms));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::reconcile_expired_locks(&active_locks).await;
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                }
            }
        }))
    }

    /// Reconcile expired locks from the in-memory registry.
    ///
    /// This is called periodically by the background reconciliation loop.
    /// It scans all locks and removes those that have expired.
    async fn reconcile_expired_locks(active_locks: &Arc<RwLock<HashMap<String, ActiveLock>>>) {
        let now_ms = now_millis();
        let mut locks = active_locks.write().await;
        let initial_count = locks.len();

        // Remove expired locks
        locks.retain(|_, lock| !lock.is_expired(now_ms));

        let removed = initial_count - locks.len();
        if removed > 0 {
            tracing::debug!(
                target: "dml_lock_reconciliation",
                removed,
                remaining = locks.len(),
                "Reconciled expired DML locks"
            );
        }
    }

    /// Shutdown the reconciliation loop.
    ///
    /// This signals the background task to stop gracefully.
    pub fn shutdown(&self) {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(true);
        }
    }

    /// Acquire a DML lock for the given scope and intent.
    ///
    /// Returns `LockOutcome::Acquired` if the lock was acquired,
    /// `LockOutcome::Conflict` if the lock conflicts with an existing lock,
    /// `LockOutcome::Held` if the lock is held by another pod,
    /// or `LockOutcome::Fenced` if this pod was fenced.
    pub async fn acquire_dml_lock(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        scope: &DmlLockScope,
        intent: LockIntent,
        now_ms: i64,
    ) -> Result<LockOutcome> {
        let started = std::time::Instant::now();
        // Resolve the canonical resource key up front so it labels the metrics
        // on every path (including the in-memory fast-path block) and is reused
        // for the durable acquire.
        let resource_key = self.scope_to_resource_key(tenant_id, namespace_id, scope);
        let resource_type = resource_key.resource_type.name();
        let scope_key = Self::scoped_lock_key(tenant_id, namespace_id, scope);
        let namespace = namespace_id.map(str::to_string);

        // Check compatibility with existing locks in-memory. A conflict/held
        // decision is captured and reported after the loop (single metric site)
        // rather than early-returned, so every outcome is observed.
        let mut blocked: Option<LockOutcome> = None;
        {
            let locks = self.active_locks.read().await;
            for existing in locks.values() {
                if existing.is_expired(now_ms)
                    || existing.tenant_id != tenant_id
                    || existing.namespace_id != namespace
                    || !existing.scope.overlaps(scope)
                {
                    continue;
                }

                if existing.pod_id == self.self_pod_id && existing.scope == *scope {
                    continue;
                }

                let held_elsewhere = existing.pod_id != self.self_pod_id;
                let held_outcome = || {
                    if held_elsewhere {
                        LockOutcome::Held {
                            holder: existing.pod_id.clone(),
                            expires_at: existing.expires_at_ms,
                        }
                    } else {
                        LockOutcome::Conflict
                    }
                };

                // Hierarchical locking enforcement: ancestor/descendant conflicts.
                // A schema lock blocks all table/partition/record locks in that
                // schema; a table lock blocks all partition/record locks in that
                // table. This applies regardless of intent.
                if scope.is_ancestor_of(&existing.scope) || scope.is_descendant_of(&existing.scope)
                {
                    blocked = Some(held_outcome());
                    break;
                }

                if !intent.is_compatible_with(&existing.intent) {
                    blocked = Some(held_outcome());
                    break;
                }
            }
        }
        if let Some(outcome) = blocked {
            crate::metrics::dml_lock_metrics::record_acquisition(
                dml_lock_outcome_label(&outcome),
                resource_type,
                tenant_id,
                started.elapsed(),
            );
            return Ok(outcome);
        }

        // Acquire durable lease for this lock
        let ttl_ms = 10_000; // DML locks have short TTL (10s)
        let lease_outcome = self
            .lease_manager
            .store
            .acquire_with_key(&resource_key, &self.self_pod_id, now_ms, ttl_ms)
            .await?;

        let outcome = match lease_outcome {
            LeaseOutcome::Acquired(lease) => {
                // Register in-memory lock
                let mut locks = self.active_locks.write().await;
                locks.insert(
                    scope_key,
                    ActiveLock {
                        tenant_id: tenant_id.to_string(),
                        namespace_id: namespace,
                        scope: scope.clone(),
                        pod_id: self.self_pod_id.clone(),
                        intent: intent.clone(),
                        acquired_at_ms: now_ms,
                        expires_at_ms: now_ms.saturating_add(ttl_ms),
                    },
                );
                crate::metrics::dml_lock_metrics::inc_held(resource_type);
                // Register the durable lease in the manager's renewal set so the
                // renew loop keeps it warm — we acquired via the store directly
                // (above) to get the generation, bypassing the manager's
                // `acquire_with_key`, which is the only other path that tracks it.
                self.lease_manager.track_held_key(&resource_key);
                LockOutcome::Acquired { lease }
            }
            LeaseOutcome::Held { by } => LockOutcome::Held {
                holder: by.holder_pod,
                expires_at: by.expires_at_ms,
            },
            LeaseOutcome::Fenced { latest } => LockOutcome::Fenced { latest },
        };
        crate::metrics::dml_lock_metrics::record_acquisition(
            dml_lock_outcome_label(&outcome),
            resource_type,
            tenant_id,
            started.elapsed(),
        );
        Ok(outcome)
    }

    /// Acquire a DML lock and return a guard that owns local release.
    pub async fn acquire_dml_lock_guard(
        self: &Arc<Self>,
        tenant_id: &str,
        namespace_id: Option<&str>,
        scope: DmlLockScope,
        intent: LockIntent,
        now_ms: i64,
    ) -> Result<DmlLockGuard> {
        // Compute the resource label up front so the error arms can use it
        // without moving `scope` (which the Acquired arm consumes).
        let resource = scope.to_key();
        match self
            .acquire_dml_lock(tenant_id, namespace_id, &scope, intent.clone(), now_ms)
            .await?
        {
            LockOutcome::Acquired { lease } => Ok(DmlLockGuard {
                service: self.clone(),
                tenant_id: tenant_id.to_string(),
                namespace_id: namespace_id.map(str::to_string),
                scope,
                intent,
                lease_generation: lease.generation,
                released: false,
            }),
            // Typed conflict errors (wrapped in anyhow) — mapped to
            // ProximaDBError::DmlLockConflict at the services layer.
            LockOutcome::Conflict => Err(DmlLockAcquireError::Conflict { resource }.into()),
            LockOutcome::Held { holder, .. } => {
                Err(DmlLockAcquireError::Held { resource, holder }.into())
            }
            LockOutcome::Fenced { latest } => Err(DmlLockAcquireError::Fenced {
                resource,
                holder: latest.holder_pod,
            }
            .into()),
        }
    }

    /// Release a DML lock.
    ///
    /// This removes the lock from the in-memory registry. The durable lease
    /// will expire naturally after its TTL.
    pub async fn release_dml_lock(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        scope: &DmlLockScope,
    ) {
        let scope_key = Self::scoped_lock_key(tenant_id, namespace_id, scope);
        let mut locks = self.active_locks.write().await;
        locks.remove(&scope_key);
    }

    /// Full release (F7): deregister the in-memory lock AND publish a
    /// best-effort durable released-tombstone, so the next acquirer need
    /// not wait for TTL. The durable step is fail-open to TTL expiry.
    pub async fn release_full(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        scope: &DmlLockScope,
    ) {
        self.release_dml_lock(tenant_id, namespace_id, scope).await;
        let key = self.scope_to_resource_key(tenant_id, namespace_id, scope);
        crate::metrics::dml_lock_metrics::dec_held(key.resource_type.name());
        // Best-effort; `release_with_key` is fail-open to TTL internally.
        let _ = self
            .lease_manager
            .release_with_key(&key, now_millis())
            .await;
    }

    /// Release all DML locks held by this pod.
    pub async fn release_all_locks(&self) {
        let mut locks = self.active_locks.write().await;
        locks.retain(|_, lock| lock.pod_id != self.self_pod_id);
    }

    /// Convert a DML lock scope to a ResourceKey for lease acquisition.
    fn scope_to_resource_key(
        &self,
        tenant_id: &str,
        namespace_id: Option<&str>,
        scope: &DmlLockScope,
    ) -> ResourceKey {
        // For DML locks, we use a special resource type based on the scope level
        let resource_type = match scope {
            DmlLockScope::Schema { .. } => ResourceType::Schema,
            DmlLockScope::Table { .. } => ResourceType::Table,
            DmlLockScope::Partition { .. } => ResourceType::Table, // Partitions use Table type
            DmlLockScope::Record { .. } => ResourceType::Table,    // Records use Table type
        };

        let resource_id = match scope {
            DmlLockScope::Schema { schema_name } => ResourceIdentifier::single(schema_name.clone()),
            DmlLockScope::Table {
                schema_name,
                table_name,
            } => ResourceIdentifier::composite(vec![schema_name.clone(), table_name.clone()]),
            DmlLockScope::Partition {
                schema_name,
                table_name,
                partition_id,
            } => ResourceIdentifier::composite(vec![
                schema_name.clone(),
                table_name.clone(),
                partition_id.clone(),
            ]),
            DmlLockScope::Record {
                schema_name,
                table_name,
                key,
            } => ResourceIdentifier::composite(vec![
                schema_name.clone(),
                table_name.clone(),
                key.clone(),
            ]),
        };

        ResourceKey {
            tenant_id: tenant_id.to_string(),
            namespace_id: namespace_id.map(str::to_string),
            resource_type,
            resource_id,
        }
    }

    fn scoped_lock_key(
        tenant_id: &str,
        namespace_id: Option<&str>,
        scope: &DmlLockScope,
    ) -> String {
        match namespace_id {
            Some(namespace_id) => format!(
                "tenant:{}/namespace:{}/{}",
                encode_path_component(tenant_id),
                encode_path_component(namespace_id),
                scope.to_key()
            ),
            None => format!(
                "tenant:{}/{}",
                encode_path_component(tenant_id),
                scope.to_key()
            ),
        }
    }
}

impl ActiveLock {
    /// Check if this lock has expired at the given time.
    pub fn is_expired(&self, now_ms: i64) -> bool {
        now_ms >= self.expires_at_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    const PREFIX: &str = "_operator/leases";
    const LEASE_MS: i64 = 10_000;

    /// Two lease stores over ONE shared backing object store = two pods.
    fn shared_backing() -> Arc<dyn object_store::ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn store(backing: &Arc<dyn object_store::ObjectStore>) -> PartitionLeaseStore {
        PartitionLeaseStore::new(ProximaObjectStore::new(backing.clone()), PREFIX)
    }

    fn test_manager(
        backing: &Arc<dyn object_store::ObjectStore>,
        pod: &str,
    ) -> PartitionLeaseManager {
        PartitionLeaseManager::new(
            Arc::new(store(backing)),
            Arc::new(PrimaryPodRegistry::new()),
            pod,
            LEASE_MS,
        )
    }

    #[tokio::test]
    async fn writer_incarnation_bumps_epoch_after_same_pod_restart() -> Result<()> {
        let backing = shared_backing();
        let first_process = test_manager(&backing, "stable-pod-id");
        assert!(
            first_process
                .begin_writer_incarnation("tenant", "canonical-uuid", 1)
                .await?
        );
        let (_, first) = store(&backing)
            .read("tenant", "canonical-uuid")
            .await?
            .expect("first incarnation lease");

        // A fresh manager represents a process restart but deliberately keeps
        // the pod identity. It must bump, not renew/reuse, the generation.
        let restarted_process = test_manager(&backing, "stable-pod-id");
        assert!(
            restarted_process
                .begin_writer_incarnation("tenant", "canonical-uuid", 2)
                .await?
        );
        let (_, second) = store(&backing)
            .read("tenant", "canonical-uuid")
            .await?
            .expect("second incarnation lease");
        assert!(second.generation > first.generation);
        Ok(())
    }

    /// `validate_fencing` is the storage-boundary contract (A6). None =
    /// unfenced/legacy write, always allowed.
    #[test]
    fn validate_fencing_none_is_unfenced() {
        assert!(validate_fencing(None, 5).is_ok());
        assert!(validate_fencing(None, 0).is_ok());
    }

    /// A write carrying the current (or newer) generation is allowed.
    #[test]
    fn validate_fencing_current_or_newer_allowed() {
        assert!(validate_fencing(Some(5), 5).is_ok()); // equal
        assert!(validate_fencing(Some(7), 5).is_ok()); // newer
    }

    /// A write carrying a stale generation (a takeover displaced it) is rejected.
    #[test]
    fn validate_fencing_stale_rejected() {
        match validate_fencing(Some(3), 5) {
            Err(FencingError {
                write_generation,
                current_generation,
            }) => {
                assert_eq!(write_generation, 3);
                assert_eq!(current_generation, 5);
            }
            other => panic!("expected FencingError for stale write, got {other:?}"),
        }
    }

    /// A fresh resource (current generation 0) allows any Some(g) write.
    #[test]
    fn validate_fencing_fresh_resource_allows_all() {
        assert!(validate_fencing(Some(0), 0).is_ok());
        assert!(validate_fencing(Some(1), 0).is_ok());
    }

    /// A6 storage-write fence primitive (#346): `is_fenced_out` reflects durable
    /// cross-pod ownership and flips on a lease takeover. Two managers over one
    /// shared backing = two pods; the holder is never fenced, a non-holder always
    /// is, and after the lease lapses + the other pod takes over the verdict
    /// inverts — the exact signal the flush boundary uses to reject a displaced
    /// pod's stale write.
    #[tokio::test]
    async fn is_fenced_out_reflects_cross_pod_takeover() -> Result<()> {
        let backing = shared_backing();
        let mgr_a = test_manager(&backing, "A");
        let mgr_b = test_manager(&backing, "B");

        // No lease yet → fail-open: nobody is fenced.
        assert!(!mgr_a.is_fenced_out("t", "c", 0).await?);
        assert!(!mgr_b.is_fenced_out("t", "c", 0).await?);

        // A acquires (t,c): A (holder) is not fenced; B (different pod) IS fenced
        // out while A's lease is live.
        assert!(mgr_a.acquire("t", "c", 0).await?);
        assert!(
            !mgr_a.is_fenced_out("t", "c", 1).await?,
            "the leaseholder is never fenced out of its own partition"
        );
        assert!(
            mgr_b.is_fenced_out("t", "c", 1).await?,
            "a non-holder is fenced while another pod holds a live lease"
        );

        // A's lease lapses; B takes over. The fence inverts: displaced A is now
        // fenced, new holder B is not.
        let after = LEASE_MS + 1;
        assert!(mgr_b.acquire("t", "c", after).await?);
        assert!(
            mgr_a.is_fenced_out("t", "c", after + 1).await?,
            "displaced A is fenced out after B's takeover (would resurrect stale data)"
        );
        assert!(
            !mgr_b.is_fenced_out("t", "c", after + 1).await?,
            "new holder B is not fenced"
        );
        Ok(())
    }

    /// A6 cross-pod integration: the storage-write fence ADAPTER over a real lease
    /// manager, driven through the exact decision the flush coordinator uses
    /// (`evaluate_fence`), rejects a displaced pod's flush when enforcement is ON
    /// and is byte-identical (Proceed) when OFF. This is the boundary the convergence
    /// wires: pod A holds, B takes over, A's flush at the boundary is REJECTED (ON)
    /// / allowed (OFF).
    #[tokio::test]
    async fn a6_fence_rejects_displaced_pod_flush_when_on_byte_identical_when_off() -> Result<()> {
        use crate::network::storage_write_fence::LeaseStorageWriteFence;
        use crate::storage::write_fence::{FenceDecision, StorageWriteFence, evaluate_fence};

        let backing = shared_backing();
        let mgr_a = Arc::new(test_manager(&backing, "A"));
        let mgr_b = Arc::new(test_manager(&backing, "B"));

        // Pod A's flush boundary consults a fence backed by A's lease manager.
        let fence_a: Arc<dyn StorageWriteFence> =
            Arc::new(LeaseStorageWriteFence::new(mgr_a.clone()));

        // A holds the lease → A's flush proceeds whether enforcement is on or off.
        assert!(mgr_a.acquire("t", "c", 0).await?);
        assert_eq!(
            evaluate_fence(true, Some(&fence_a), Some("t"), "c", 1).await,
            FenceDecision::Proceed,
            "holder A flushes freely while it owns the lease"
        );

        // B takes over after A's lease lapses → A is now the stale, displaced pod.
        let after = LEASE_MS + 1;
        assert!(mgr_b.acquire("t", "c", after).await?);

        // ON ⇒ reject A's flush (would resurrect stale/tombstoned data);
        // OFF ⇒ proceed (byte-identical to the pre-fence path).
        assert_eq!(
            evaluate_fence(true, Some(&fence_a), Some("t"), "c", after + 1).await,
            FenceDecision::Fenced,
            "displaced A's flush is rejected when fencing is ON"
        );
        assert_eq!(
            evaluate_fence(false, Some(&fence_a), Some("t"), "c", after + 1).await,
            FenceDecision::Proceed,
            "displaced A's flush proceeds (byte-identical) when fencing is OFF"
        );
        Ok(())
    }

    /// F7: explicit release publishes a released tombstone at generation+1,
    /// not an object-store delete (which would reset the fence to 1).
    #[tokio::test]
    async fn release_publishes_higher_generation_tombstone() -> Result<()> {
        let backing = shared_backing();
        let s = store(&backing);
        let key = ResourceKey::table("t1", "public", "users");

        let LeaseOutcome::Acquired(lease) = s.acquire_with_key(&key, "A", 0, 10_000).await? else {
            panic!("A should acquire");
        };
        assert_eq!(lease.generation, 1);
        assert!(!lease.released);

        s.release_with_key(&key, "A", 100).await?;

        let (_, durable) = s
            .read_key(&key)
            .await?
            .expect("a tombstone must remain on the log");
        assert!(durable.released, "tombstone must be marked released");
        assert_eq!(durable.generation, 2, "tombstone sits at generation+1");
        Ok(())
    }

    /// F7: acquiring after a release must NOT reset the generation to 1 — it
    /// claims generation+1 over the tombstone, preserving the monotonic fence.
    #[tokio::test]
    async fn acquire_after_release_keeps_monotonic_generation() -> Result<()> {
        let backing = shared_backing();
        let s = store(&backing);
        let key = ResourceKey::table("t1", "public", "users");

        s.acquire_with_key(&key, "A", 0, 10_000).await?;
        s.release_with_key(&key, "A", 100).await?;

        let LeaseOutcome::Acquired(lease) = s.acquire_with_key(&key, "B", 200, 10_000).await?
        else {
            panic!("B should acquire the released resource");
        };
        assert_eq!(
            lease.generation, 3,
            "generation must stay monotonic (tombstone@2 → acquire@3), not reset to 1"
        );
        assert!(!lease.released);
        Ok(())
    }

    #[tokio::test]
    async fn partition_serving_handles_continue_after_explicit_manifest_migration() -> Result<()> {
        let backing = shared_backing();
        let serving = store(&backing);
        let key = ResourceKey::table("t1", "public", "migration_test");
        assert!(matches!(
            serving.acquire_with_key(&key, "A", 0, 10_000).await?,
            LeaseOutcome::Acquired(_)
        ));
        // Explicit cutover while all source writers/pruners are quiescent.
        serving
            .committer_for_key(&key)?
            .migrate_versioned(0)
            .await?;
        let (_, before) = serving.read_key(&key).await?.expect("migrated lease");
        assert_eq!(before.holder_pod, "A");
        serving.release_with_key(&key, "A", 100).await?;
        let reopened = store(&backing);
        let LeaseOutcome::Acquired(after) =
            reopened.acquire_with_key(&key, "B", 200, 10_000).await?
        else {
            panic!("successor must acquire released versioned lease");
        };
        assert_eq!(after.holder_pod, "B");
        assert_eq!(after.generation, 3);
        Ok(())
    }

    /// F7: release by a pod that does not hold the lease is a stale no-op —
    /// the live lease is untouched (no tombstone, holder unchanged).
    #[tokio::test]
    async fn release_by_non_holder_is_noop() -> Result<()> {
        let backing = shared_backing();
        let s = store(&backing);
        let key = ResourceKey::table("t1", "public", "users");

        s.acquire_with_key(&key, "A", 0, 10_000).await?;
        s.release_with_key(&key, "B", 100).await?; // B does not hold it

        let (_, durable) = s.read_key(&key).await?.expect("lease present");
        assert!(!durable.released, "non-holder release must not tombstone");
        assert_eq!(durable.holder_pod, "A");
        Ok(())
    }

    /// A fresh partition: one pod acquires generation 1; a second pod racing the
    /// same fresh partition is fenced (the version CAS admits exactly one), and
    /// it learns the real owner.
    #[tokio::test]
    async fn two_pods_contend_exactly_one_owner() -> Result<()> {
        let backing = shared_backing();
        let pod_a = store(&backing);
        let pod_b = store(&backing);

        let a = pod_a.acquire("t", "c", "A", 0, LEASE_MS).await?;
        let LeaseOutcome::Acquired(lease_a) = a else {
            panic!("pod A should acquire the fresh partition, got {a:?}");
        };
        assert_eq!(lease_a.holder_pod, "A");
        assert_eq!(lease_a.generation, 1);

        // Pod B, racing the same fresh partition, read None too and tries gen 1 —
        // it loses the version CAS and is fenced, learning A is the owner.
        let b = pod_b.acquire("t", "c", "B", 0, LEASE_MS).await?;
        match b {
            LeaseOutcome::Fenced { latest } | LeaseOutcome::Held { by: latest } => {
                assert_eq!(latest.holder_pod, "A");
            }
            LeaseOutcome::Acquired(_) => panic!("two owners — fence failed"),
        }
        Ok(())
    }

    /// A live lease held by another pod yields `Held` (not a takeover).
    #[tokio::test]
    async fn live_lease_blocks_other_pod() -> Result<()> {
        let backing = shared_backing();
        let pod_a = store(&backing);
        let pod_b = store(&backing);

        assert!(matches!(
            pod_a.acquire("t", "c", "A", 0, LEASE_MS).await?,
            LeaseOutcome::Acquired(_)
        ));
        // B at a time well inside A's lease window.
        match pod_b.acquire("t", "c", "B", 1_000, LEASE_MS).await? {
            LeaseOutcome::Held { by } => assert_eq!(by.holder_pod, "A"),
            other => panic!("expected Held by A, got {other:?}"),
        }
        Ok(())
    }

    /// The holder renews at the same generation with a fresh expiry.
    #[tokio::test]
    async fn holder_renews_extends_expiry_same_generation() -> Result<()> {
        let backing = shared_backing();
        let pod_a = store(&backing);

        let LeaseOutcome::Acquired(l1) = pod_a.acquire("t", "c", "A", 0, LEASE_MS).await? else {
            panic!("acquire");
        };
        let LeaseOutcome::Acquired(l2) = pod_a.acquire("t", "c", "A", 5_000, LEASE_MS).await?
        else {
            panic!("renew");
        };
        assert_eq!(
            l2.generation, l1.generation,
            "renewal must not bump generation"
        );
        assert!(
            l2.expires_at_ms > l1.expires_at_ms,
            "renewal extends expiry"
        );
        Ok(())
    }

    /// P1a: the background renew loop (spawned in production by SharedServices)
    /// keeps a held lease alive past its TTL. Without it the lease would lapse
    /// and the leaseholder model the write-gate (421) + DML locks rely on
    /// silently degrades.
    #[tokio::test]
    async fn renew_loop_keeps_held_lease_past_ttl() -> Result<()> {
        let backing = shared_backing();
        let store_handle = Arc::new(store(&backing));
        // Short TTL (100ms) so the test is fast.
        let mgr = Arc::new(PartitionLeaseManager::new(
            store_handle.clone(),
            Arc::new(PrimaryPodRegistry::new()),
            "pod-A",
            100,
        ));
        let key = ResourceKey::legacy_collection("t1", "c1");
        assert!(mgr.acquire_with_key(&key, now_millis()).await?);

        // Renew every 40ms (≤ TTL/2), fire-and-forget like production.
        let renew_handle = mgr.clone().spawn_renew_loop(Duration::from_millis(40));

        // Sleep well past the TTL. Without renewal the lease would be expired;
        // with the renew loop it must stay held by pod-A at the same generation.
        tokio::time::sleep(Duration::from_millis(300)).await;

        let (_, lease) = store_handle
            .read_key(&key)
            .await?
            .expect("lease still present");
        assert_eq!(lease.holder_pod, "pod-A");
        assert!(
            !lease.is_expired(now_millis()),
            "renewal must push expiry forward past now"
        );
        assert_eq!(
            lease.generation, 1,
            "renewal keeps generation (no takeover)"
        );

        renew_handle.abort();
        Ok(())
    }

    /// Owner death (lease expiry) → fenced handoff: a new pod takes over with a
    /// strictly-higher generation, and the dead owner's later renewal is rejected.
    #[tokio::test]
    async fn expired_lease_handoff_fences_dead_owner() -> Result<()> {
        let backing = shared_backing();
        let pod_a = store(&backing);
        let pod_b = store(&backing);

        let LeaseOutcome::Acquired(la) = pod_a.acquire("t", "c", "A", 0, LEASE_MS).await? else {
            panic!("A acquire");
        };
        assert_eq!(la.generation, 1);

        // A "dies"; time advances past expiry. B takes over.
        let after_expiry = LEASE_MS + 1;
        let LeaseOutcome::Acquired(lb) =
            pod_b.acquire("t", "c", "B", after_expiry, LEASE_MS).await?
        else {
            panic!("B takeover");
        };
        assert_eq!(lb.holder_pod, "B");
        assert_eq!(lb.generation, 2, "takeover outranks the dead owner");

        // A "comes back" and tries to renew — it must NOT regain ownership.
        match pod_a
            .acquire("t", "c", "A", after_expiry + 1, LEASE_MS)
            .await?
        {
            LeaseOutcome::Held { by } => assert_eq!(by.holder_pod, "B"),
            other => panic!("stale owner A must be rejected, got {other:?}"),
        }
        Ok(())
    }

    /// Legacy collection keys must share the exact same manifest log as the old
    /// `(tenant, collection)` API until a dedicated path migration lands.
    #[tokio::test]
    async fn legacy_collection_key_uses_legacy_manifest_path() -> Result<()> {
        let backing = shared_backing();
        let store = store(&backing);
        let key = ResourceKey::legacy_collection("tenant-a", "collection-a");

        let legacy = store
            .acquire("tenant-a", "collection-a", "pod-a", 0, LEASE_MS)
            .await?;
        assert!(matches!(legacy, LeaseOutcome::Acquired(_)));

        let keyed = store
            .acquire_with_key(&key, "pod-b", 1_000, LEASE_MS)
            .await?;
        match keyed {
            LeaseOutcome::Held { by } => {
                assert_eq!(by.holder_pod, "pod-a");
                assert_eq!(by.key().as_ref(), &key);
            }
            other => panic!("keyed acquire must contend with legacy path, got {other:?}"),
        }

        let read_by_key = store.read_key(&key).await?.expect("lease by key");
        assert_eq!(read_by_key.1.holder_pod, "pod-a");
        Ok(())
    }

    /// Collection routing is TOTAL: a namespaced collection key would land on a
    /// different `to_path()` log than the legacy `(tenant, collection)` writer —
    /// a forked generation fence = cross-version split brain. It must be rejected
    /// LOUDLY, never silently routed to a divergent log. (No current caller
    /// builds such a key; this guards against one being introduced.)
    #[tokio::test]
    async fn namespaced_collection_key_is_rejected_not_forked() -> Result<()> {
        let backing = shared_backing();
        let store = store(&backing);
        let ns_key = ResourceKey::new(
            "tenant-a",
            Some("ns-x".to_string()),
            ResourceType::Collection,
            ResourceIdentifier::single("collection-a".to_string()),
        );

        // Every routing path (read + acquire) must error rather than diverge.
        assert!(
            store.read_key(&ns_key).await.is_err(),
            "namespaced collection key must be rejected, not routed to a forked log"
        );
        assert!(
            store
                .acquire_with_key(&ns_key, "pod-a", 0, LEASE_MS)
                .await
                .is_err(),
            "acquire on a namespaced collection key must error, not fork the fence"
        );
        Ok(())
    }

    /// A composite-of-one collection key reduces to the canonical legacy shape
    /// and MUST contend on the SAME log as the legacy `(tenant, collection)`
    /// writer — totality, not a separate `to_path()` log.
    #[tokio::test]
    async fn composite_one_collection_key_uses_legacy_log() -> Result<()> {
        let backing = shared_backing();
        let store = store(&backing);

        let legacy = store
            .acquire("tenant-a", "collection-a", "pod-a", 0, LEASE_MS)
            .await?;
        assert!(matches!(legacy, LeaseOutcome::Acquired(_)));

        let composite_key = ResourceKey::new(
            "tenant-a",
            None,
            ResourceType::Collection,
            ResourceIdentifier::composite(vec!["collection-a".to_string()]),
        );
        match store
            .acquire_with_key(&composite_key, "pod-b", 1_000, LEASE_MS)
            .await?
        {
            LeaseOutcome::Held { by } => assert_eq!(by.holder_pod, "pod-a"),
            other => panic!("composite-of-one must contend with legacy log, got {other:?}"),
        }
        Ok(())
    }

    /// Generalized resource keys must be path-safe: separators inside tenant,
    /// namespace, or resource components are encoded, not treated as hierarchy.
    #[test]
    fn resource_key_path_encodes_components() {
        let key = ResourceKey::new(
            "tenant/a",
            Some("ns/b".to_string()),
            ResourceType::Table,
            ResourceIdentifier::composite(vec!["schema/c".to_string(), "table d".to_string()]),
        );

        let path = key.to_path();
        assert_eq!(path, "tenant%2Fa/ns%2Fb/table/schema%2Fc/table%20d");

        let decoded = ResourceKey::from_path(&path).expect("decode path");
        assert_eq!(decoded, key);
    }

    /// Manager integration: acquisition assigns the registry binding; a lost
    /// lease steps the binding down to the new owner (consult_for_write flips
    /// Allow → Misrouted), so exactly one pod is ever writable across a handoff.
    #[tokio::test]
    async fn manager_acquire_assigns_and_steps_down() -> Result<()> {
        let backing = shared_backing();
        let reg_a = Arc::new(PrimaryPodRegistry::new());
        let reg_b = Arc::new(PrimaryPodRegistry::new());
        let mgr_a =
            PartitionLeaseManager::new(Arc::new(store(&backing)), reg_a.clone(), "A", LEASE_MS);
        let mgr_b =
            PartitionLeaseManager::new(Arc::new(store(&backing)), reg_b.clone(), "B", LEASE_MS);

        // A acquires → A is the owner; A's gate Allows, B's gate Misroutes to A.
        assert!(mgr_a.acquire("t", "c", 0).await?);
        assert_eq!(
            consult_for_write(&reg_a, "A", "t", "c"),
            WriteRoutingDecision::Allow
        );
        assert!(!mgr_b.acquire("t", "c", 1_000).await?);
        assert_eq!(
            consult_for_write(&reg_b, "B", "t", "c"),
            WriteRoutingDecision::Misrouted {
                target_pod: "A".to_string()
            }
        );

        // A's lease lapses; B takes over.
        let after = LEASE_MS + 1;
        assert!(mgr_b.acquire("t", "c", after).await?);
        assert_eq!(
            consult_for_write(&reg_b, "B", "t", "c"),
            WriteRoutingDecision::Allow
        );

        // A's renew pass discovers it lost the lease and steps down — its gate now
        // Misroutes to B (it does NOT fall back to Allow, which would split-brain).
        assert_eq!(
            mgr_a.renew_held(after + 1).await?,
            0,
            "A owns nothing after step-down"
        );
        assert_eq!(
            consult_for_write(&reg_a, "A", "t", "c"),
            WriteRoutingDecision::Misrouted {
                target_pod: "B".to_string()
            }
        );
        Ok(())
    }

    /// `ensure_owned` is the lease-on-write entry point: it acquires on a miss
    /// (making the shared registry truthful), is idempotent on the fast-path when
    /// already held, and steps a non-owner down to the true owner so the gate
    /// Misroutes instead of admitting a split-brain write.
    #[tokio::test]
    async fn ensure_owned_idempotent_and_steps_down() -> Result<()> {
        let backing = shared_backing();
        let reg_a = Arc::new(PrimaryPodRegistry::new());
        let reg_b = Arc::new(PrimaryPodRegistry::new());
        let mgr_a =
            PartitionLeaseManager::new(Arc::new(store(&backing)), reg_a.clone(), "A", LEASE_MS);
        let mgr_b =
            PartitionLeaseManager::new(Arc::new(store(&backing)), reg_b.clone(), "B", LEASE_MS);

        // A's first write: ensure_owned acquires (registry miss) → A owns → Allow.
        assert!(mgr_a.ensure_owned("t", "c", 0).await?);
        assert_eq!(
            consult_for_write(&reg_a, "A", "t", "c"),
            WriteRoutingDecision::Allow
        );
        // A's subsequent write: fast-path (already bound to self) → still owns.
        assert!(mgr_a.ensure_owned("t", "c", 1_000).await?);

        // B's first write to the same collection: ensure_owned acquires → A holds
        // the live lease → B does NOT own it, and B's shared registry is repointed
        // to A so its gate Misroutes (no split brain).
        assert!(!mgr_b.ensure_owned("t", "c", 2_000).await?);
        assert_eq!(
            consult_for_write(&reg_b, "B", "t", "c"),
            WriteRoutingDecision::Misrouted {
                target_pod: "A".to_string()
            }
        );
        Ok(())
    }

    /// A6 fence primitive: `is_fenced_out` (via `current_holder`) detects that a
    /// live lease held by another pod fences this pod out of writing to shared
    /// storage, and fails open when there is no live lease.
    #[tokio::test]
    async fn is_fenced_out_detects_takeover() -> Result<()> {
        let backing = shared_backing();
        let reg_a = Arc::new(PrimaryPodRegistry::new());
        let reg_b = Arc::new(PrimaryPodRegistry::new());
        let mgr_a = PartitionLeaseManager::new(Arc::new(store(&backing)), reg_a, "A", LEASE_MS);
        let mgr_b = PartitionLeaseManager::new(Arc::new(store(&backing)), reg_b, "B", LEASE_MS);

        assert!(mgr_a.acquire("t", "c", 0).await?);
        // A holds the live lease → A is not fenced; B (a different pod) IS fenced
        // out and must not flush.
        assert!(!mgr_a.is_fenced_out("t", "c", 1).await?);
        assert!(mgr_b.is_fenced_out("t", "c", 1).await?);
        // No lease for an unrelated collection → fail-open (not fenced).
        assert!(!mgr_b.is_fenced_out("t", "other", 1).await?);
        // Once A's lease expires with no renewal it is no longer live → B is no
        // longer fenced (it could legitimately take over).
        assert!(!mgr_b.is_fenced_out("t", "c", LEASE_MS + 1).await?);
        // current_holder still surfaces the (now-stale) record for inspection.
        let held = mgr_b.current_holder("t", "c").await?;
        assert_eq!(held.map(|(_, l)| l.holder_pod), Some("A".to_string()));
        Ok(())
    }

    /// Single-pod / no contention: the same pod acquires and renews indefinitely,
    /// staying the owner — the lease layer is inert overhead-wise and never
    /// fences itself.
    #[tokio::test]
    async fn single_pod_keeps_its_lease() -> Result<()> {
        let backing = shared_backing();
        let reg = Arc::new(PrimaryPodRegistry::new());
        let mgr =
            PartitionLeaseManager::new(Arc::new(store(&backing)), reg.clone(), "solo", LEASE_MS);

        assert!(mgr.acquire("t", "c", 0).await?);
        // Renew across several intervals — always retains ownership.
        for tick in 1..=5 {
            assert_eq!(mgr.renew_held(tick * 1_000).await?, 1);
        }
        assert_eq!(
            consult_for_write(&reg, "solo", "t", "c"),
            WriteRoutingDecision::Allow
        );
        Ok(())
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Strategy Pattern Tests (Phase 2)
    ////////////////////////////////////////////////////////////////////////////////

    /// Collection strategy uses 60s TTL by default.
    #[tokio::test]
    async fn collection_strategy_default_ttl() -> Result<()> {
        let strategy = CollectionStrategy;
        assert_eq!(strategy.lease_ttl_secs(), 60);
        assert_eq!(strategy.resource_type(), ResourceType::Collection);
        assert!(strategy.supports_partitioning());
        Ok(())
    }

    /// Table strategy uses 120s TTL for DSL operations.
    #[tokio::test]
    async fn table_strategy_longer_ttl() -> Result<()> {
        let strategy = TableStrategy;
        assert_eq!(strategy.lease_ttl_secs(), 120);
        assert_eq!(strategy.resource_type(), ResourceType::Table);
        assert!(strategy.supports_partitioning());
        assert!(strategy.supports_hierarchical_locking());
        Ok(())
    }

    /// Model strategy uses 300s TTL for rare model updates.
    #[tokio::test]
    async fn model_strategy_long_ttl() -> Result<()> {
        let strategy = ModelStrategy;
        assert_eq!(strategy.lease_ttl_secs(), 300);
        assert_eq!(strategy.resource_type(), ResourceType::Model);
        assert!(!strategy.supports_partitioning());
        assert!(strategy.supports_hierarchical_locking());
        Ok(())
    }

    /// Graph node/edge strategies use short 10s TTL for DML-level locking.
    #[tokio::test]
    async fn graph_node_edge_short_ttl() -> Result<()> {
        let node_strategy = GraphNodeStrategy;
        let edge_strategy = GraphEdgeStrategy;

        assert_eq!(node_strategy.lease_ttl_secs(), 10);
        assert_eq!(node_strategy.resource_type(), ResourceType::GraphNode);
        assert!(!node_strategy.supports_partitioning());

        assert_eq!(edge_strategy.lease_ttl_secs(), 10);
        assert_eq!(edge_strategy.resource_type(), ResourceType::GraphEdge);
        Ok(())
    }

    /// Manager uses strategy TTL when strategy is registered.
    #[tokio::test]
    async fn manager_uses_strategy_ttl() -> Result<()> {
        let backing = shared_backing();
        let reg = Arc::new(PrimaryPodRegistry::new());
        let mgr = Arc::new(PartitionLeaseManager::new(
            Arc::new(store(&backing)),
            reg.clone(),
            "solo",
            999_999, // Default TTL (unused when strategy registered)
        ));

        // Register collection strategy (60s TTL)
        mgr.register_strategy(Arc::new(CollectionStrategy));

        // Acquire with ResourceKey - should use strategy TTL
        let key = ResourceKey::legacy_collection("tenant1", "collection1");
        assert!(mgr.acquire_with_key(&key, 0).await?);

        // Verify the lease was created with strategy TTL (60s = 60000ms)
        let stored = mgr.store.read_key(&key).await?.unwrap();
        let lease = stored.1;
        let expected_expiry = 60_000; // 60 seconds
        assert_eq!(lease.expires_at_ms, expected_expiry);
        Ok(())
    }

    /// Non-collection resource leases are renewed by full ResourceKey, not by
    /// the legacy primary-pod registry.
    #[tokio::test]
    async fn manager_renews_generalized_resource_keys() -> Result<()> {
        let backing = shared_backing();
        let reg = Arc::new(PrimaryPodRegistry::new());
        let mgr = Arc::new(PartitionLeaseManager::new(
            Arc::new(store(&backing)),
            reg,
            "solo",
            LEASE_MS,
        ));

        mgr.register_strategy(Arc::new(TableStrategy));
        let key = ResourceKey::table("tenant1", "public", "users");
        assert!(mgr.acquire_with_key(&key, 0).await?);
        assert_eq!(mgr.renew_held(10_000).await?, 1);

        let stored = mgr.store.read_key(&key).await?.unwrap();
        let lease = stored.1;
        assert_eq!(lease.holder_pod, "solo");
        assert_eq!(lease.expires_at_ms, 130_000);
        Ok(())
    }

    /// A DML lock's durable lease is registered in the manager's renewal set
    /// (via `track_held_key`), so the renew loop keeps it warm. Before this fix
    /// `DmlLockService` acquired via the store directly — bypassing the manager's
    /// `held_resource_keys` — so `renew_held` returned 0 and the lease silently
    /// lapsed at the 10s TTL while still held (a takeover-eligible split-brain
    /// window for any DML lock held longer than the TTL).
    #[tokio::test]
    async fn dml_lock_durable_lease_is_tracked_and_renewed() -> Result<()> {
        let backing = shared_backing();
        let manager = Arc::new(test_manager(&backing, "pod-1"));
        let lock_service = DmlLockService::new(manager.clone(), "pod-1".to_string());

        let scope = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let acquired = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Write, 0)
            .await?;
        assert!(matches!(acquired, LockOutcome::Acquired { .. }));

        // The renew loop must now see the DML lease as held-by-us and renew it.
        let renewed = manager.renew_held(1_000).await?;
        assert_eq!(renewed, 1, "DML lease must be in the manager's renewal set");

        // Renewal pushed the durable expiry past the original 10s TTL window.
        let key = ResourceKey::table("tenant1", "public", "users");
        let stored = manager.store.read_key(&key).await?.expect("lease present");
        assert_eq!(stored.1.holder_pod, "pod-1");
        assert!(
            stored.1.expires_at_ms > 10_000,
            "renewal at t=1000 must push expiry beyond the initial TTL (got {})",
            stored.1.expires_at_ms
        );

        // After full release the key leaves the renewal set (no renew leak).
        lock_service.release_full("tenant1", None, &scope).await;
        assert_eq!(
            manager.renew_held(2_000).await?,
            0,
            "released DML lease must no longer be renewed"
        );
        Ok(())
    }

    /// Table strategy creates composite keys for schema.table.
    #[tokio::test]
    async fn table_strategy_composite_key() -> Result<()> {
        let strategy = TableStrategy;
        let key = strategy
            .make_key(
                "tenant1",
                None,
                &["schema1".to_string(), "table1".to_string()],
            )
            .expect("valid key");

        assert_eq!(key.tenant_id, "tenant1");
        assert_eq!(key.resource_type, ResourceType::Table);
        match key.resource_id {
            ResourceIdentifier::Composite(ref parts) => {
                assert_eq!(parts, &["schema1".to_string(), "table1".to_string()]);
            }
            _ => panic!("Expected composite identifier"),
        }

        // Verify path serialization (uses "/" for object storage path segments)
        assert_eq!(key.to_path(), "tenant1/table/schema1/table1");
        Ok(())
    }

    /// Schema strategy validates operations correctly.
    #[tokio::test]
    async fn schema_strategy_validates_operations() -> Result<()> {
        let strategy = SchemaStrategy;

        // SchemaChange is valid on schemas
        assert!(
            strategy
                .validate_operation(&ResourceOperation::SchemaChange)
                .is_ok()
        );

        // Read/Write are invalid on schemas (schema-level only)
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Read)
                .is_err()
        );
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Write)
                .is_err()
        );

        Ok(())
    }

    /// Graph hierarchy: Graph → GraphNode/GraphEdge.
    #[tokio::test]
    async fn graph_hierarchy_keys() -> Result<()> {
        let graph_strategy = GraphStrategy;
        let node_strategy = GraphNodeStrategy;
        let edge_strategy = GraphEdgeStrategy;

        let graph_key = graph_strategy
            .make_key("tenant1", None, &["graph1".to_string()])
            .expect("valid key");
        assert_eq!(graph_key.resource_type, ResourceType::Graph);

        let node_key = node_strategy
            .make_key(
                "tenant1",
                None,
                &["graph1".to_string(), "node1".to_string()],
            )
            .expect("valid key");
        assert_eq!(node_key.resource_type, ResourceType::GraphNode);
        match node_key.resource_id {
            ResourceIdentifier::Composite(parts) => {
                assert_eq!(parts, vec!["graph1".to_string(), "node1".to_string()]);
            }
            _ => panic!("Expected composite identifier"),
        }

        let edge_key = edge_strategy
            .make_key(
                "tenant1",
                None,
                &["graph1".to_string(), "edge1".to_string()],
            )
            .expect("valid key");
        assert_eq!(edge_key.resource_type, ResourceType::GraphEdge);

        Ok(())
    }

    /// Model → ModelVersion hierarchy.
    #[tokio::test]
    async fn model_version_hierarchy() -> Result<()> {
        let model_strategy = ModelStrategy;
        let version_strategy = ModelVersionStrategy;

        let model_key = model_strategy
            .make_key("tenant1", None, &["model1".to_string()])
            .expect("valid key");
        assert_eq!(model_key.resource_type, ResourceType::Model);

        let version_key = version_strategy
            .make_key("tenant1", None, &["model1".to_string(), "v1.0".to_string()])
            .expect("valid key");
        assert_eq!(version_key.resource_type, ResourceType::ModelVersion);
        match version_key.resource_id {
            ResourceIdentifier::Composite(ref parts) => {
                assert_eq!(parts, &["model1".to_string(), "v1.0".to_string()]);
            }
            _ => panic!("Expected composite identifier"),
        }

        // Verify parent key lookup
        let parent = model_strategy.parent_key(&version_key);
        assert!(parent.is_some());
        let parent = parent.unwrap();
        assert_eq!(parent.resource_type, ResourceType::Model);
        match parent.resource_id {
            ResourceIdentifier::Single(s) => {
                assert_eq!(s, "model1");
            }
            _ => panic!("Expected single identifier"),
        }

        Ok(())
    }

    /// Table → Schema hierarchy.
    #[tokio::test]
    async fn table_schema_hierarchy() -> Result<()> {
        let table_strategy = TableStrategy;

        let table_key = table_strategy
            .make_key(
                "tenant1",
                None,
                &["public".to_string(), "users".to_string()],
            )
            .expect("valid key");

        // Verify parent key lookup
        let parent = table_strategy.parent_key(&table_key);
        assert!(parent.is_some());
        let parent = parent.unwrap();
        assert_eq!(parent.resource_type, ResourceType::Schema);
        match parent.resource_id {
            ResourceIdentifier::Single(s) => {
                assert_eq!(s, "public");
            }
            _ => panic!("Expected single identifier"),
        }

        Ok(())
    }

    /// DML lock intent compatibility: MVP durable DML locks are exclusive.
    #[tokio::test]
    async fn dml_lock_exclusive_intent_compatibility() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        let scope = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };

        // First lock should succeed.
        let result1 = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Read, 0)
            .await?;
        assert!(matches!(result1, LockOutcome::Acquired { .. }));

        // Same-pod exact-scope reacquire renews the durable lease.
        let result2 = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Read, 100)
            .await?;
        assert!(matches!(result2, LockOutcome::Acquired { .. }));

        // A broader schema lock conflicts with the table lock.
        let schema_scope = DmlLockScope::Schema {
            schema_name: "public".to_string(),
        };
        let result3 = lock_service
            .acquire_dml_lock("tenant1", None, &schema_scope, LockIntent::Write, 200)
            .await?;
        assert!(matches!(result3, LockOutcome::Conflict));

        lock_service.release_dml_lock("tenant1", None, &scope).await;

        // Now Write should succeed
        let result5 = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Write, 400)
            .await?;
        assert!(matches!(result5, LockOutcome::Acquired { .. }));

        Ok(())
    }

    /// DML lock hierarchical locking: Schema → Table → Partition.
    #[tokio::test]
    async fn dml_lock_hierarchy() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        let schema_scope = DmlLockScope::Schema {
            schema_name: "public".to_string(),
        };
        let table_scope = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let partition_scope = DmlLockScope::Partition {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
            partition_id: "p1".to_string(),
        };

        // Acquire schema lock
        let schema_lock = lock_service
            .acquire_dml_lock("tenant1", None, &schema_scope, LockIntent::Write, 0)
            .await?;
        assert!(matches!(schema_lock, LockOutcome::Acquired { .. }));

        // Table lock should conflict with schema Write lock
        let table_lock = lock_service
            .acquire_dml_lock("tenant1", None, &table_scope, LockIntent::Read, 100)
            .await?;
        assert!(matches!(table_lock, LockOutcome::Conflict));

        // Partition lock also conflicts with schema lock.
        let partition_lock = lock_service
            .acquire_dml_lock("tenant1", None, &partition_scope, LockIntent::Write, 200)
            .await?;
        assert!(matches!(partition_lock, LockOutcome::Conflict));

        Ok(())
    }

    /// DML lock expiration and renewal.
    #[tokio::test]
    async fn dml_lock_expiration() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        let scope = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };

        // Acquire lock at time 0
        let result1 = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Write, 0)
            .await?;
        assert!(matches!(result1, LockOutcome::Acquired { .. }));

        // At time 500 (before 10s TTL), renewal by same pod should work
        let result2 = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Write, 500)
            .await?;
        assert!(matches!(result2, LockOutcome::Acquired { .. }));

        // At time 15000 (after 10s TTL), lock should be expired
        let result3 = lock_service
            .acquire_dml_lock("tenant1", None, &scope, LockIntent::Write, 15000)
            .await?;
        // Should succeed because the old lease expired
        assert!(matches!(result3, LockOutcome::Acquired { .. }));

        Ok(())
    }

    /// DML lock release_all clears all locks for a pod.
    #[tokio::test]
    async fn dml_lock_release_all() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        let scope1 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let scope2 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "orders".to_string(),
        };

        // Acquire two locks
        lock_service
            .acquire_dml_lock("tenant1", None, &scope1, LockIntent::Write, 0)
            .await?;
        lock_service
            .acquire_dml_lock("tenant1", None, &scope2, LockIntent::Write, 100)
            .await?;

        // Verify locks are held
        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 2);
        drop(locks);

        // Release all locks
        lock_service.release_all_locks().await;

        // Verify all locks are cleared
        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 0);

        Ok(())
    }

    /// DML lock guards own explicit local release.
    #[tokio::test]
    async fn dml_lock_guard_releases_local_registry() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = Arc::new(DmlLockService::new(Arc::new(manager), "pod-1".to_string()));

        let scope = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let guard = lock_service
            .acquire_dml_lock_guard("tenant1", Some("public"), scope, LockIntent::Write, 0)
            .await?;
        assert_eq!(guard.lease_generation(), 1);

        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 1);
        drop(locks);

        guard.release().await;
        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 0);

        Ok(())
    }

    /// DML lock scope key generation is consistent.
    #[tokio::test]
    async fn dml_lock_scope_key_consistency() -> Result<()> {
        let scope1 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let scope2 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let scope3 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "orders".to_string(),
        };

        // Same scope should produce same key
        assert_eq!(scope1.to_key(), scope2.to_key());
        // Different scope should produce different key
        assert_ne!(scope1.to_key(), scope3.to_key());

        Ok(())
    }

    /// DML lock reconciliation removes expired locks.
    #[tokio::test]
    async fn dml_lock_reconciliation_removes_expired() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        let scope1 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };
        let scope2 = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "orders".to_string(),
        };

        // Manually insert locks with different expiry times
        let now_ms = now_millis();
        let mut locks = lock_service.active_locks.write().await;
        locks.insert(
            DmlLockService::scoped_lock_key("tenant1", None, &scope1),
            ActiveLock {
                tenant_id: "tenant1".to_string(),
                namespace_id: None,
                scope: scope1.clone(),
                pod_id: "pod-1".to_string(),
                intent: LockIntent::Write,
                acquired_at_ms: 0,
                expires_at_ms: now_ms.saturating_sub(1),
            },
        );
        locks.insert(
            DmlLockService::scoped_lock_key("tenant1", None, &scope2),
            ActiveLock {
                tenant_id: "tenant1".to_string(),
                namespace_id: None,
                scope: scope2.clone(),
                pod_id: "pod-1".to_string(),
                intent: LockIntent::Write,
                acquired_at_ms: 0,
                expires_at_ms: now_ms.saturating_add(60_000),
            },
        );
        drop(locks);

        // Verify 2 locks
        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 2);
        drop(locks);

        // Reconcile at time 10s (first lock expired, second still valid)
        DmlLockService::reconcile_expired_locks(&lock_service.active_locks).await;

        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 1);
        assert!(locks.contains_key(&DmlLockService::scoped_lock_key("tenant1", None, &scope2)));
        assert!(!locks.contains_key(&DmlLockService::scoped_lock_key("tenant1", None, &scope1)));

        Ok(())
    }

    /// DML lock reconciliation loop runs periodically.
    #[tokio::test]
    async fn dml_lock_reconciliation_loop() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        let scope = DmlLockScope::Table {
            schema_name: "public".to_string(),
            table_name: "users".to_string(),
        };

        // Insert a lock that will expire soon
        let mut locks = lock_service.active_locks.write().await;
        locks.insert(
            DmlLockService::scoped_lock_key("tenant1", None, &scope),
            ActiveLock {
                tenant_id: "tenant1".to_string(),
                namespace_id: None,
                scope: scope.clone(),
                pod_id: "pod-1".to_string(),
                intent: LockIntent::Write,
                acquired_at_ms: 0,
                expires_at_ms: 100, // Expires at 100ms
            },
        );
        drop(locks);

        // Start reconciliation loop with 50ms interval
        let handle = lock_service
            .spawn_reconciliation_loop(50)
            .expect("reconciliation handle");

        // Wait for at least one reconciliation cycle
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Verify lock was removed
        let locks = lock_service.active_locks.read().await;
        assert_eq!(locks.len(), 0);

        // Shutdown the loop
        lock_service.shutdown();

        // Wait for the task to finish
        let _ = tokio::time::timeout(Duration::from_millis(100), handle).await;

        Ok(())
    }

    /// DML lock shutdown signals the reconciliation loop to stop.
    #[tokio::test]
    async fn dml_lock_shutdown_stops_reconciliation() -> Result<()> {
        let backing = shared_backing();
        let manager = test_manager(&backing, "test-pod");
        let lock_service = DmlLockService::new(Arc::new(manager), "pod-1".to_string());

        // Start reconciliation loop with 1s interval
        let handle = lock_service
            .spawn_reconciliation_loop(1000)
            .expect("reconciliation handle");

        // Shutdown immediately
        lock_service.shutdown();

        // Verify the task stopped quickly
        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        assert!(
            result.is_ok(),
            "Reconciliation loop should stop on shutdown"
        );

        Ok(())
    }

    /// ML Model strategy validates model-specific operations.
    #[tokio::test]
    async fn ml_model_strategy_validates_operations() -> Result<()> {
        let strategy = ModelStrategy;

        // Model-specific operations should be valid
        assert!(
            strategy
                .validate_operation(&ResourceOperation::ModelRegister)
                .is_ok()
        );
        assert!(
            strategy
                .validate_operation(&ResourceOperation::ModelDeploy)
                .is_ok()
        );

        // DDL-like operations should be valid
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Create)
                .is_ok()
        );
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Read)
                .is_ok()
        );

        // Write operations should NOT be valid on models
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Write)
                .is_err()
        );

        Ok(())
    }

    /// ML ExperimentRun strategy validates training run operations.
    #[tokio::test]
    async fn ml_experiment_run_validates_training_operations() -> Result<()> {
        let strategy = ExperimentRunStrategy;

        // Training-specific operations should be valid
        assert!(
            strategy
                .validate_operation(&ResourceOperation::TrainingRun)
                .is_ok()
        );
        assert!(
            strategy
                .validate_operation(&ResourceOperation::MetricWrite)
                .is_ok()
        );

        // Write operations for logging should be valid
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Write)
                .is_ok()
        );

        // Schema change should NOT be valid on experiment runs
        assert!(
            strategy
                .validate_operation(&ResourceOperation::SchemaChange)
                .is_err()
        );

        Ok(())
    }

    /// ML FeatureSet strategy validates feature write operations.
    #[tokio::test]
    async fn ml_feature_set_validates_feature_operations() -> Result<()> {
        let strategy = FeatureSetStrategy;

        // Feature-specific operations should be valid
        assert!(
            strategy
                .validate_operation(&ResourceOperation::FeatureWrite)
                .is_ok()
        );

        // DDL operations should be valid
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Create)
                .is_ok()
        );
        assert!(
            strategy
                .validate_operation(&ResourceOperation::Alter)
                .is_ok()
        );

        // Training operations should NOT be valid on feature sets
        assert!(
            strategy
                .validate_operation(&ResourceOperation::TrainingRun)
                .is_err()
        );

        Ok(())
    }

    /// ML Model hierarchy: Model → ModelVersion.
    #[tokio::test]
    async fn ml_model_hierarchy_keys() -> Result<()> {
        let model_strategy = ModelStrategy;
        let version_strategy = ModelVersionStrategy;

        let model_key = model_strategy
            .make_key("tenant1", None, &["my_model".to_string()])
            .expect("valid key");
        let version_key = version_strategy
            .make_key(
                "tenant1",
                None,
                &["my_model".to_string(), "v1.0".to_string()],
            )
            .expect("valid key");
        assert_eq!(model_key.resource_type, ResourceType::Model);

        // Verify model version parent is the model
        let parent = version_strategy.parent_key(&version_key);
        assert!(parent.is_some());
        let parent = parent.unwrap();
        assert_eq!(parent.resource_type, ResourceType::Model);
        match parent.resource_id {
            ResourceIdentifier::Single(s) => {
                assert_eq!(s, "my_model");
            }
            _ => panic!("Expected single identifier"),
        }

        Ok(())
    }

    /// ML Experiment hierarchy: Experiment → ExperimentRun.
    #[tokio::test]
    async fn ml_experiment_hierarchy_keys() -> Result<()> {
        let exp_strategy = ExperimentStrategy;
        let run_strategy = ExperimentRunStrategy;

        let exp_key = exp_strategy
            .make_key("tenant1", None, &["exp1".to_string()])
            .expect("valid key");
        let run_key = run_strategy
            .make_key("tenant1", None, &["exp1".to_string(), "run123".to_string()])
            .expect("valid key");
        assert_eq!(exp_key.resource_type, ResourceType::Experiment);

        // Verify experiment run parent is the experiment
        let parent = run_strategy.parent_key(&run_key);
        assert!(parent.is_some());
        let parent = parent.unwrap();
        assert_eq!(parent.resource_type, ResourceType::Experiment);
        match parent.resource_id {
            ResourceIdentifier::Single(s) => {
                assert_eq!(s, "exp1");
            }
            _ => panic!("Expected single identifier"),
        }

        Ok(())
    }

    /// ML operations use appropriate TTLs.
    #[tokio::test]
    async fn ml_strategies_use_appropriate_ttls() -> Result<()> {
        let model_strategy = ModelStrategy;
        let version_strategy = ModelVersionStrategy;
        let exp_strategy = ExperimentStrategy;
        let run_strategy = ExperimentRunStrategy;
        let feature_strategy = FeatureSetStrategy;

        // Models have long TTL (300s) - rare updates
        assert_eq!(model_strategy.lease_ttl_secs(), 300);
        assert_eq!(version_strategy.lease_ttl_secs(), 300);

        // Experiments have moderate TTL (120s) - moderately frequent updates
        assert_eq!(exp_strategy.lease_ttl_secs(), 120);
        assert_eq!(feature_strategy.lease_ttl_secs(), 120);

        // Experiment runs have shorter TTL (60s) - frequent updates during training
        assert_eq!(run_strategy.lease_ttl_secs(), 60);

        Ok(())
    }

    /// ML resource keys serialize correctly for storage.
    #[tokio::test]
    async fn ml_resource_keys_serialize_correctly() -> Result<()> {
        let model_strategy = ModelStrategy;
        let exp_strategy = ExperimentStrategy;

        let model_key = model_strategy
            .make_key("tenant1", Some("ml-namespace"), &["my_model".to_string()])
            .expect("valid key");
        let exp_key = exp_strategy
            .make_key("tenant1", None, &["exp1".to_string()])
            .expect("valid key");

        // Model with namespace: {tenant}/{namespace}/{type}/{name}
        assert_eq!(model_key.to_path(), "tenant1/ml-namespace/model/my_model");

        // Experiment without namespace: {tenant}/{type}/{name}
        assert_eq!(exp_key.to_path(), "tenant1/experiment/exp1");

        Ok(())
    }
}
