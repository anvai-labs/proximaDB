// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! The reference tool catalog + backend trait (the agent-facing catalog reference-MCP design).
//!
//! Each tool is a thin **projection of an existing engine surface** — catalog
//! introspection, the statistics envelope, the unified EXPLAIN
//! (ADR-004), REST v2 hybrid search, and the ADR-022 agent-state services. The
//! surface is **generic and unprivileged**: no pricing, rate card, per-account
//! entitlement, PII redaction, or governance policy — those belong to AnvaiOps's
//! governed gateway (ADR-0021), which *composes* this reference surface. The
//! unprivileged invariant is enforced by the `unprivileged_*` guard test.

use crate::protocol::{Tool, error_code};
use async_trait::async_trait;
use serde_json::{Value, json};

/// An error from a backend projection, mapped to a JSON-RPC error.
#[derive(Debug, Clone)]
pub struct BackendError {
    pub code: i32,
    pub message: String,
}

impl BackendError {
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: error_code::INVALID_PARAMS,
            message: message.into(),
        }
    }
    /// The caller's arguments are invalid (JSON-RPC `INVALID_PARAMS`).
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: error_code::INVALID_PARAMS,
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: error_code::INTERNAL_ERROR,
            message: message.into(),
        }
    }
}

/// The capabilities the reference MCP server projects. An adapter implements this
/// over the engine's existing REST/gRPC surfaces; the protocol layer never
/// touches the engine directly. All methods are tenant-scoped by the adapter
/// (structural isolation), but carry **no** per-account entitlement (AnvaiOps).
#[async_trait]
pub trait ReferenceBackend: Send + Sync {
    /// Catalog introspection: list collections (xcatalog / information_schema).
    async fn list_collections(&self, args: &Value) -> Result<Value, BackendError>;
    /// Catalog introspection: describe one collection's schema/indexes.
    async fn describe(&self, args: &Value) -> Result<Value, BackendError>;
    /// The statistics envelope (units only) for a collection.
    async fn stats(&self, args: &Value) -> Result<Value, BackendError>;
    /// The ADR-004 unified EXPLAIN for a query (selectivity/rows/cost).
    async fn explain(&self, args: &Value) -> Result<Value, BackendError>;
    /// REST v2 hybrid search.
    async fn search(&self, args: &Value) -> Result<Value, BackendError>;
    /// ADR-022 agent memory (store/get/list).
    async fn memory(&self, args: &Value) -> Result<Value, BackendError>;
    /// ADR-022 agent checkpoint (save/restore).
    async fn checkpoint(&self, args: &Value) -> Result<Value, BackendError>;
    /// ADR-022 agent event (append/list).
    async fn event(&self, args: &Value) -> Result<Value, BackendError>;
}

/// Stable tool names (the conformance contract for the boundary object).
pub mod tool_name {
    pub const LIST_COLLECTIONS: &str = "list_collections";
    pub const DESCRIBE: &str = "describe";
    pub const STATS: &str = "stats";
    pub const EXPLAIN: &str = "explain";
    pub const SEARCH: &str = "search";
    pub const MEMORY: &str = "memory";
    pub const CHECKPOINT: &str = "checkpoint";
    pub const EVENT: &str = "event";
}

fn collection_arg_schema(extra_desc: &str) -> Value {
    json!({
        "type": "object",
        "required": ["collection_id"],
        "properties": {
            "collection_id": { "type": "string", "description": format!("Collection name or id.{extra_desc}") }
        },
        "additionalProperties": true
    })
}

/// The reference tool catalog returned by `tools/list`.
pub fn reference_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: tool_name::LIST_COLLECTIONS.to_string(),
            description: "List collections visible to the caller (catalog introspection)."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1 },
                    "offset": { "type": "integer", "minimum": 0 }
                },
                "additionalProperties": true
            }),
        },
        Tool {
            name: tool_name::DESCRIBE.to_string(),
            description: "Describe a collection's schema, indexes, and capabilities.".to_string(),
            input_schema: collection_arg_schema(""),
        },
        Tool {
            name: tool_name::STATS.to_string(),
            description: "Return the statistics envelope (units and distributions only) for a \
                          collection — record/field counts, cardinality, distributions, and \
                          per-modality summaries. The engine attests a freshness fact (a \
                          watermark)."
                .to_string(),
            input_schema: collection_arg_schema(""),
        },
        Tool {
            name: tool_name::EXPLAIN.to_string(),
            description: "Explain a query: the unified plan with estimated selectivity, rows, \
                          and cost, plus stats freshness."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["collection_id", "query"],
                "properties": {
                    "collection_id": { "type": "string" },
                    "query": { "type": "object", "description": "The query to explain." }
                },
                "additionalProperties": true
            }),
        },
        Tool {
            name: tool_name::SEARCH.to_string(),
            description: "Hybrid search over a collection (vector + filter + text).".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["collection_id"],
                "properties": {
                    "collection_id": { "type": "string" },
                    "query": { "type": "object" },
                    "k": { "type": "integer", "minimum": 1 }
                },
                "additionalProperties": true
            }),
        },
        Tool {
            name: tool_name::MEMORY.to_string(),
            description: "Agent memory: semantic search (read) over stored memory entries (ADR-022). Writing memory is not yet available over this surface."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["collection_id", "query", "tenant_id"],
                "properties": {
                    "collection_id": { "type": "string", "description": "Memory collection to search." },
                    "query": { "type": "string", "description": "Query text (embedded, then vector-searched)." },
                    "tenant_id": { "type": "string", "description": "Tenant scope — required (fail-closed isolation)." },
                    "session_id": { "type": "string", "description": "Optional session scope narrowing." },
                    "actor": { "type": "string", "description": "Optional actor scope narrowing." },
                    "k": { "type": "integer", "minimum": 1, "description": "Max hits to return (default 10)." }
                },
                "additionalProperties": true
            }),
        },
        Tool {
            name: tool_name::CHECKPOINT.to_string(),
            description: "Agent checkpoint: save, get, or list agent state for a thread (ADR-022). \
                          Event-sourced — `save` appends and `get` returns the latest checkpoint \
                          unless `checkpoint_id` is given."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["thread_id", "tenant_id"],
                "properties": {
                    "op": { "type": "string", "enum": ["save", "get", "list"], "description": "Operation (default `save`)." },
                    "thread_id": { "type": "string", "description": "Agent thread this checkpoint belongs to." },
                    "tenant_id": { "type": "string", "description": "Tenant scope — required (fail-closed isolation)." },
                    "checkpoint_ns": { "type": "string", "description": "Optional namespace within the thread (default `default`)." },
                    "checkpoint_id": { "type": "string", "description": "On `save`, use this id instead of minting one. On `get`, select this checkpoint instead of the latest." },
                    "parent_checkpoint_id": { "type": "string", "description": "Parent checkpoint, recording lineage." },
                    "checkpoint": { "description": "State payload — required for `save`." },
                    "metadata": { "type": "object", "description": "Optional checkpoint metadata." },
                    "writes": { "type": "array", "description": "Optional pending channel writes." },
                    "limit": { "type": "integer", "minimum": 1, "description": "Max entries for `list` (default 20)." }
                },
                "additionalProperties": true
            }),
        },
        Tool {
            name: tool_name::EVENT.to_string(),
            description: "Agent event log: append an event to the ADR-022 auditable trail."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["event_type"],
                "properties": {
                    "entity_id": { "type": "string", "description": "Entity/stream the event belongs to. Falls back to `collection_id` if omitted." },
                    "collection_id": { "type": "string", "description": "Used as the entity/stream when `entity_id` is absent." },
                    "event_type": { "type": "string", "description": "Event type/name." },
                    "data": { "description": "Event payload (any JSON value)." },
                    "metadata": { "type": "object", "description": "Optional key/value metadata." },
                    "causation_id": { "type": "string", "description": "Optional id of the causing event." }
                },
                "additionalProperties": true
            }),
        },
    ]
}

/// Route a `tools/call` to the backing projection. Unknown tool → method-not-found.
pub async fn dispatch_tool(
    backend: &dyn ReferenceBackend,
    name: &str,
    args: &Value,
) -> Result<Value, BackendError> {
    match name {
        tool_name::LIST_COLLECTIONS => backend.list_collections(args).await,
        tool_name::DESCRIBE => backend.describe(args).await,
        tool_name::STATS => backend.stats(args).await,
        tool_name::EXPLAIN => backend.explain(args).await,
        tool_name::SEARCH => backend.search(args).await,
        tool_name::MEMORY => backend.memory(args).await,
        tool_name::CHECKPOINT => backend.checkpoint(args).await,
        tool_name::EVENT => backend.event(args).await,
        other => Err(BackendError {
            code: error_code::METHOD_NOT_FOUND,
            message: format!("unknown tool: {other}"),
        }),
    }
}
