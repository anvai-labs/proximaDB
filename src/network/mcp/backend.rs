// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! In-process `ReferenceBackend` (ADR-037 Decision 5).
//!
//! Projects the reference MCP tools onto the engine's existing services by
//! calling them **directly** (no REST/gRPC self-hop): catalog list/describe via
//! [`ApiHandlersPort`], the statistics envelope via the resident
//! `statistics_registry`, and explain/search via the [`UnifiedQueryPort`] (which
//! already returns `serde_json::Value`). The `event` tool appends to the ADR-022
//! auditable `EventLogEngine`, and `memory` runs a scoped semantic search (read)
//! over agent memory via a `VectorMemoryStore` — both shared from
//! `SharedServices`. `checkpoint` is event-sourced over the same `EventLogEngine`
//! via `AgentCheckpointStore` (TD-MCP-1 phase MCP-1d). Memory *store*/put remains
//! LLM-gated (TD-101) and is not offered on this surface — the tool description
//! says so rather than fabricating.

use crate::core::statistics::{StatisticsSummary, statistics_registry};
use crate::proto::proximadb_v1::{CollectionOperation, CollectionRequest};
use crate::services::VectorOperationsService;
use crate::services::agent_checkpoint::{AgentCheckpointStore, CheckpointPut, CheckpointScope};
use crate::services::agent_memory::{
    EmbeddingServiceEmbedder, ExtractedFact, MemoryStore, MemoryWriteScope, VectorMemoryStore,
};
use crate::storage::engines::eventlog::{Event, EventLogEngine};
use async_trait::async_trait;
use proximadb_data_model::MemoryType;
use proximadb_mcp::{BackendError, ReferenceBackend};
use proximadb_runtime::{ApiHandlersPort, UnifiedQueryPort};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

/// A `ReferenceBackend` that calls the engine's in-process services directly.
pub struct EngineBackend {
    api_handlers: Arc<dyn ApiHandlersPort>,
    query: Option<Arc<dyn UnifiedQueryPort>>,
    /// ADR-022 auditable event log — backs the `event` tool. `None` disables it
    /// (the engine failed to initialize), and `event` reports so explicitly.
    event_log: Option<Arc<EventLogEngine>>,
    /// Vector operations service — backs the `memory` search (read) tool via a
    /// `VectorMemoryStore`. `None` disables it.
    vector_ops: Option<Arc<VectorOperationsService>>,
    /// Tenant the MCP surface operates as. Stage 1 uses a single configured
    /// tenant (or the port default when `None`); per-request tenant scoping from
    /// MCP auth is a later refinement.
    tenant: Option<String>,
}

impl EngineBackend {
    pub fn new(
        api_handlers: Arc<dyn ApiHandlersPort>,
        query: Option<Arc<dyn UnifiedQueryPort>>,
        event_log: Option<Arc<EventLogEngine>>,
        vector_ops: Option<Arc<VectorOperationsService>>,
        tenant: Option<String>,
    ) -> Self {
        Self {
            api_handlers,
            query,
            event_log,
            vector_ops,
            tenant,
        }
    }

    fn tenant(&self) -> Option<&str> {
        self.tenant.as_deref()
    }

    fn collection_request(
        op: CollectionOperation,
        collection_id: Option<String>,
    ) -> CollectionRequest {
        CollectionRequest {
            operation: op as i32,
            collection_id,
            collection_config: None,
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        }
    }
}

fn require_str(args: &Value, key: &str) -> Result<String, BackendError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| BackendError::not_found(format!("missing required string arg `{key}`")))
}

/// The `query` arg as a query string (a string is used verbatim; any other
/// non-null value is serialized — the engine routes/parses it).
fn query_string(args: &Value) -> Result<String, BackendError> {
    match args.get("query") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(v) if !v.is_null() => Ok(v.to_string()),
        _ => Err(BackendError::not_found(
            "missing required `query` (a query string)".to_string(),
        )),
    }
}

#[async_trait]
impl ReferenceBackend for EngineBackend {
    async fn list_collections(&self, _args: &Value) -> Result<Value, BackendError> {
        let req = Self::collection_request(CollectionOperation::CollectionList, None);
        let resp = self
            .api_handlers
            .handle_collection_operation_for_tenant(req, self.tenant())
            .await
            .map_err(|e| BackendError::internal(format!("list collections failed: {e}")))?;
        let collections: Vec<Value> = resp
            .collections
            .into_iter()
            .map(|c| {
                let cfg = c.config.unwrap_or_default();
                let stats = c.stats.unwrap_or_default();
                let id = if c.id.is_empty() { cfg.name.clone() } else { c.id };
                json!({ "collection_id": id, "name": cfg.name, "record_count": stats.vector_count.max(0) })
            })
            .collect();
        Ok(json!({ "collections": collections }))
    }

    async fn describe(&self, args: &Value) -> Result<Value, BackendError> {
        let id = require_str(args, "collection_id")?;
        let req = Self::collection_request(CollectionOperation::CollectionGet, Some(id.clone()));
        let resp = self
            .api_handlers
            .handle_collection_operation_for_tenant(req, self.tenant())
            .await
            .map_err(|e| BackendError::internal(format!("describe failed: {e}")))?;
        let c = resp
            .collection
            .ok_or_else(|| BackendError::not_found(format!("collection `{id}` not found")))?;
        let cfg = c.config.unwrap_or_default();
        let stats = c.stats.unwrap_or_default();
        Ok(json!({
            "collection_id": if c.id.is_empty() { id } else { c.id },
            "name": cfg.name,
            "dimension": cfg.dimension,
            "record_count": stats.vector_count.max(0),
            "storage_size_bytes": stats.data_size_bytes.max(0),
            "index_size_bytes": stats.index_size_bytes.max(0),
        }))
    }

    async fn stats(&self, args: &Value) -> Result<Value, BackendError> {
        let id = require_str(args, "collection_id")?;
        // Resident envelope from the flush/compaction write boundary; an empty
        // (epoch-watermark) envelope when no snapshot has populated one yet.
        let envelope = statistics_registry()
            .envelope(&id)
            .unwrap_or_else(|| StatisticsSummary::new(&id).to_envelope());
        serde_json::to_value(&envelope)
            .map_err(|e| BackendError::internal(format!("serialize envelope failed: {e}")))
    }

    async fn explain(&self, args: &Value) -> Result<Value, BackendError> {
        let id = require_str(args, "collection_id")?;
        let q = query_string(args)?;
        let port = self
            .query
            .as_ref()
            .ok_or_else(|| BackendError::internal("query facade not available".to_string()))?;
        // The MCP session acts as its configured tenant (default when none).
        let tenant = self
            .tenant
            .clone()
            .unwrap_or_else(|| proximadb_tenant::DEFAULT_TENANT.to_string());
        crate::query::request_scope::with_request_tenant(
            tenant,
            port.explain_unified_query(q, Some(id)),
        )
        .await
        .map_err(|e| BackendError::internal(format!("explain failed: {e}")))
    }

    async fn search(&self, args: &Value) -> Result<Value, BackendError> {
        let id = require_str(args, "collection_id")?;
        let q = query_string(args)?;
        let limit = args.get("k").and_then(Value::as_u64).map(|k| k as u32);
        let port = self
            .query
            .as_ref()
            .ok_or_else(|| BackendError::internal("query facade not available".to_string()))?;
        let tenant = self
            .tenant
            .clone()
            .unwrap_or_else(|| proximadb_tenant::DEFAULT_TENANT.to_string());
        crate::query::request_scope::with_request_tenant(
            tenant,
            port.execute_unified_query(q, None, Some(id), limit),
        )
        .await
        .map_err(|e| BackendError::internal(format!("search failed: {e}")))
    }

    async fn memory(&self, args: &Value) -> Result<Value, BackendError> {
        // MCP-1b: semantic search (read) over agent memory via `VectorMemoryStore`
        // (embeds the query, scoped vector search). Store/put is LLM-gated
        // (TD-101) and remains unwired.
        let vector_ops = self.vector_ops.as_ref().ok_or_else(|| {
            BackendError::internal("vector operations service unavailable".to_string())
        })?;
        let collection = require_str(args, "collection_id")?;
        let query = require_str(args, "query")?;
        // Fail-closed: memory search MUST be tenant-scoped (shared-collection
        // isolation), mirroring `MemoryAqlSource`.
        let tenant_id = args
            .get("tenant_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.tenant.clone())
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| {
                BackendError::not_found(
                    "memory search requires a non-empty `tenant_id` (fail-closed isolation)"
                        .to_string(),
                )
            })?;
        let session_id = args
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let actor = args
            .get("actor")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let k = args.get("k").and_then(Value::as_u64).unwrap_or(10);
        // Shared server top_k cap (TD-SECSCAN-1 §5): reject at the boundary as
        // invalid params (client error) before embedding the query; the search
        // authority (`unified_search_v1`) enforces it again.
        proximadb_search_types::top_k_limit::check_top_k(k)
            .map_err(|e| BackendError::invalid_params(e.to_string()))?;
        let k = k as usize;

        let store = VectorMemoryStore::new(vector_ops.clone(), Arc::new(EmbeddingServiceEmbedder));
        let scope = MemoryWriteScope {
            collection,
            tenant_id,
            actor,
            session_id,
        };
        let fact = ExtractedFact {
            text: query,
            memory_type: MemoryType::Fact,
        };
        let hits = store
            .retrieve_similar(&scope, &fact, k)
            .await
            .map_err(|e| BackendError::internal(format!("memory search failed: {e}")))?;
        let items: Vec<Value> = hits
            .into_iter()
            .map(|h| {
                json!({
                    "id": h.id,
                    "text": h.text,
                    "score": h.score,
                    "memory_type": h.memory_type.map(|t| format!("{t:?}")),
                })
            })
            .collect();
        Ok(json!({ "hits": items }))
    }
    /// MCP-1d: agent checkpoint save/get/list, event-sourced over the ADR-022
    /// event log (`AgentCheckpointStore`). Tenant scope is fail-closed, matching
    /// the `memory` tool.
    async fn checkpoint(&self, args: &Value) -> Result<Value, BackendError> {
        let log = self.event_log.as_ref().ok_or_else(|| {
            BackendError::internal(
                "checkpoint unavailable (EventLogEngine failed to initialize)".to_string(),
            )
        })?;
        let tenant_id = args
            .get("tenant_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.tenant.clone())
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| {
                BackendError::not_found(
                    "checkpoint requires a non-empty `tenant_id` (fail-closed isolation)"
                        .to_string(),
                )
            })?;
        let thread_id = require_str(args, "thread_id")?;
        let checkpoint_ns = args
            .get("checkpoint_ns")
            .and_then(Value::as_str)
            .map(str::to_string);
        let scope = CheckpointScope::new(tenant_id, checkpoint_ns, thread_id);
        let store = AgentCheckpointStore::new(log.clone());
        let op = args.get("op").and_then(Value::as_str).unwrap_or("save");

        let bad = |e: anyhow::Error| BackendError::not_found(e.to_string());
        match op {
            "save" | "put" => {
                let checkpoint = args.get("checkpoint").cloned().ok_or_else(|| {
                    BackendError::not_found("`save` requires a `checkpoint` payload".to_string())
                })?;
                let put = CheckpointPut {
                    checkpoint_id: args
                        .get("checkpoint_id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    parent_checkpoint_id: args
                        .get("parent_checkpoint_id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    checkpoint,
                    metadata: args.get("metadata").cloned().unwrap_or(Value::Null),
                    writes: args
                        .get("writes")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default(),
                };
                let saved = store.save(&scope, put).await.map_err(bad)?;
                Ok(json!({
                    "checkpoint_id": saved.checkpoint_id,
                    "config": saved.config(&scope),
                }))
            }
            "get" | "restore" => {
                let id = args.get("checkpoint_id").and_then(Value::as_str);
                match store.get(&scope, id).await.map_err(bad)? {
                    Some(c) => Ok(json!({
                        "found": true,
                        "checkpoint_id": c.checkpoint_id,
                        "parent_checkpoint_id": c.parent_checkpoint_id,
                        "checkpoint": c.checkpoint,
                        "metadata": c.metadata,
                        "writes": c.writes,
                        "config": c.config(&scope),
                    })),
                    None => Ok(json!({ "found": false })),
                }
            }
            "list" => {
                let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
                let items: Vec<Value> = store
                    .list(&scope, limit)
                    .await
                    .map_err(bad)?
                    .into_iter()
                    .map(|c| {
                        json!({
                            "checkpoint_id": c.checkpoint_id,
                            "parent_checkpoint_id": c.parent_checkpoint_id,
                            "sequence": c.sequence,
                        })
                    })
                    .collect();
                Ok(json!({ "checkpoints": items }))
            }
            other => Err(BackendError::not_found(format!(
                "unknown checkpoint op `{other}` (expected save|get|list)"
            ))),
        }
    }
    async fn event(&self, args: &Value) -> Result<Value, BackendError> {
        let log = self.event_log.as_ref().ok_or_else(|| {
            BackendError::internal(
                "event log unavailable (EventLogEngine failed to initialize)".to_string(),
            )
        })?;
        // The entity/stream the event belongs to (accept `entity_id`, else `collection_id`).
        let entity_id =
            require_str(args, "entity_id").or_else(|_| require_str(args, "collection_id"))?;
        let event_type = require_str(args, "event_type")?;
        let data = args.get("data").cloned().unwrap_or(Value::Null);
        let metadata = match args.get("metadata") {
            Some(Value::Object(m)) => m.clone().into_iter().collect::<HashMap<String, Value>>(),
            _ => HashMap::new(),
        };
        let causation_id = args
            .get("causation_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let event = Event {
            sequence: 0, // assigned by append_event
            entity_id,
            event_type,
            data,
            timestamp: chrono::Utc::now(),
            causation_id,
            metadata,
        };
        let appended = log
            .append_event(event)
            .await
            .map_err(|e| BackendError::internal(format!("append_event failed: {e}")))?;
        Ok(json!({
            "sequence": appended.sequence,
            "entity_id": appended.entity_id,
            "event_type": appended.event_type,
            "timestamp": appended.timestamp.to_rfc3339(),
        }))
    }
}
