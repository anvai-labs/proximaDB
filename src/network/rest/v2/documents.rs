/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! `POST /api/v2/collections/{collection_id}/documents`
//!
//! Text-only ingest endpoint. Accepts records without vector fields; the
//! server populates them via the in-process EmbeddingService singleton
//! (see `proximadb_embedding::EmbeddingService::global`).
//!
//! Reuses [`ProximaFlightService::embed_text_only_records`] so the REST and
//! Arrow Flight paths share one dispatch implementation.

use axum::{
    Json,
    extract::{Extension, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::api_handlers::RichRecordBatchRequest;
use crate::errors::{ApiError, ApiResult};
use crate::network::arrow_ipc::ProximaFlightService;
use crate::network::auth::middleware::DataPlaneCapability;
use crate::network::middleware::tenant::TenantContext;
use crate::network::rest::canonical::handlers::AppState;
use crate::services::{
    WriteDurabilityRequirement, WriteIntent, WriteLaneRouter, WriteOperationKind,
};

// ── Request / response DTOs ────────────────────────────────────────────────

/// A single record submitted for native embedding + indexing.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct IngestDocument {
    pub id: String,
    /// Raw text content. Required when `X-Embed-Source: native` (default);
    /// optional when the client also supplied a vector.
    #[serde(default)]
    pub text: Option<String>,
    /// Optional client-provided vector. When present, the server skips
    /// embedding for this record. Use case: SDK that already embedded
    /// locally (legacy path).
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    /// Arbitrary metadata fields. Stored as ProximaRecord props.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct IngestDocumentsRequest {
    pub records: Vec<IngestDocument>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct IngestedRecord {
    pub id: String,
    pub dim: u32,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct IngestDocumentsResponse {
    pub mode: String,
    pub records: Vec<IngestedRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
}

// ── Handler ────────────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/api/v2/collections/{collection_id}/documents",
    tag = "Documents",
    operation_id = "ingestDocuments",
    summary = "Ingest documents for native server-side embedding.",
    description = "Canonical document-ingest surface (ADR-041, spec-driven-primary). Body \
        `{records:[{id,text,metadata}]}`; the server embeds `text` natively under \
        `X-Embed-Source=native` (default) when no per-record `vector` is supplied. \
        `X-Tenant-ID` scopes records to a tenant; `X-Ingest-Mode` carries the billing mode.",
    params(
        ("collection_id" = String, Path, description = "Target collection name/ID."),
        // X-Embed-Source is read from the HeaderMap in the handler (not a typed
        // extractor), so it isn't declared as a utoipa param — declaring it as
        // Option<String> trips E0599 (Option<String>: Display) in codegen. It's
        // documented in the operation description + carried per-call by the SDK
        // header-passthrough (ADR-041 P2).
    ),
    request_body = IngestDocumentsRequest,
    responses(
        (status = 200, description = "Ingested.", body = IngestDocumentsResponse),
        (status = 400, description = "Invalid request.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 404, description = "Collection not found.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn ingest_documents(
    Path(collection): Path<String>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    capability: Option<Extension<DataPlaneCapability>>,
    headers: HeaderMap,
    Json(request): Json<IngestDocumentsRequest>,
) -> Response {
    match ingest_documents_inner(
        collection,
        state,
        tenant,
        capability.map(|ext| ext.0),
        headers,
        request,
    )
    .await
    {
        Ok((status, response)) => (status, Json(response)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn ingest_documents_inner(
    collection: String,
    state: AppState,
    tenant: TenantContext,
    capability: Option<DataPlaneCapability>,
    headers: HeaderMap,
    request: IngestDocumentsRequest,
) -> ApiResult<(StatusCode, IngestDocumentsResponse)> {
    if collection.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection name is required".to_string(),
        ));
    }
    if request.records.is_empty() {
        return Err(ApiError::InvalidArgument(
            "At least one record is required".to_string(),
        ));
    }
    if let Some(capability) = capability.as_ref() {
        capability
            .ensure_record_count(request.records.len())
            .map_err(ApiError::InvalidArgument)?;
    }

    let tenant_id = tenant.tenant_id;

    let mode = headers
        .get("X-Ingest-Mode")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("sync")
        .to_lowercase();
    if mode != "sync" && mode != "async" {
        return Err(ApiError::InvalidArgument(format!(
            "X-Ingest-Mode must be 'sync' or 'async', got {mode:?}"
        )));
    }
    if let Some(capability) = capability.as_ref()
        && let Some(capability_mode) = capability.mode.as_deref()
        && capability_mode != mode
    {
        return Err(ApiError::InvalidArgument(format!(
            "Capability token mode {capability_mode:?} does not match X-Ingest-Mode {mode:?}"
        )));
    }

    let embed_source = headers
        .get("X-Embed-Source")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("native")
        .to_lowercase();
    if embed_source != "native" && embed_source != "sdk-vector" {
        return Err(ApiError::InvalidArgument(format!(
            "X-Embed-Source must be 'native' or 'sdk-vector', got {embed_source:?}"
        )));
    }

    info!(
        collection = %collection,
        tenant = %tenant_id,
        mode = %mode,
        embed_source = %embed_source,
        records = request.records.len(),
        "v3 documents: ingesting"
    );

    // Build ProximaRecords. Reserved column names (text, metadata, etc.) get
    // hoisted into props so the codec-shared embed_text_only_records dispatch
    // can read them out the same way the Arrow Flight path does.
    //
    // ADR-009 convergence: when the canonical-vector document route is enabled for this
    // collection, stamp the same `document` label + `_document_collection` prop that the
    // DocumentService (gRPC) surface writes, so a REST-v2-ingested doc rebuilds cleanly on a
    // gRPC read (cross-surface visibility). Gate OFF ⇒ the REST v2 record shape is unchanged.
    let stamp_document_facade =
        crate::storage::document::service::doc_canonical_vector_enabled(&collection);
    let mut records: Vec<ProximaRecord> = Vec::with_capacity(request.records.len());
    for doc in request.records {
        let now_ns = chrono::Utc::now()
            .timestamp_millis()
            .saturating_mul(1_000_000);

        let mut props = HashMap::new();
        if let Some(text) = doc.text.as_ref() {
            props.insert(
                "text".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(text.clone())),
            );
        }
        for (k, v) in doc.metadata.into_iter() {
            props.insert(k, ProximaTreeNode::Value(json_to_proxima_value(v)));
        }
        // #950: make the tenant queryable. `record.tenant_id` (the isolation
        // field, stamped below) is invisible to metadata filters and record
        // GET, which read `props` — so a tenant that lives only in the field
        // never matches a `tenant_id` filter. The server value overwrites any
        // client-supplied `tenant_id` metadata (server-authoritative).
        if !tenant_id.is_empty() {
            props.insert(
                "tenant_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(tenant_id.clone())),
            );
        }

        let embeddings = match doc.vector {
            Some(values) if !values.is_empty() => vec![EmbeddingCell {
                model_id: "sdk-provided".to_string(),
                modality: "dense_vector".to_string(),
                dim: values.len() as u32,
                values: proximadb_records::EmbeddingValues::Fp32(values),
                ..Default::default()
            }],
            _ => vec![],
        };

        let mut record = ProximaRecord {
            oid: doc.id.clone(),
            local_id: Some(doc.id),
            tenant_id: tenant_id.clone(),
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            origin: Some("v3_documents".to_string()),
            props,
            embeddings,
            ..ProximaRecord::default()
        };
        if stamp_document_facade {
            record
                .labels
                .insert(proximadb_document::DOCUMENT_RECORD_LABEL);
            record.props.insert(
                proximadb_document::DOCUMENT_COLLECTION_PROP.to_string(),
                ProximaTreeNode::Value(ProximaValue::String(collection.clone())),
            );
        }
        records.push(record);
    }

    // #951: resolve the target collection ONCE, up front. The ack contract is
    // "2xx means the records will be readable": a write must never be
    // accepted against a collection that cannot hold it (missing, dimension
    // 0, or a mismatched dimension) and then silently dropped by the batch
    // processor after the response has gone out.
    let tenant_ctx = state
        .collection_service
        .load_tenant_context(Some(&tenant_id))
        .map_err(|e| ApiError::InvalidArgument(format!("tenant resolution failed: {e}")))?;
    let existing_dimension: Option<u32> = state
        .collection_service
        .get_collection_with_tenant_context(&collection, tenant_ctx.as_ref())
        .await
        .map_err(|e| ApiError::Internal(format!("collection lookup failed: {e}")))?
        .map(|col| col.config.as_ref().map(|c| c.dimension).unwrap_or(0));

    // A dimension-0 collection can never accept a record — every insert is
    // dropped by dimension validation after the ack. Fail loudly instead of
    // 200-acking into a black hole (#951 symptom A).
    if existing_dimension == Some(0) {
        return Err(ApiError::InvalidArgument(format!(
            "collection '{collection}' exists with dimension 0 and cannot store records; \
             recreate it with the embedding dimension (or drop it and let this endpoint \
             auto-create it at the embed dimension)"
        )));
    }

    // SDK-provided vectors: the dimension is known before any embedding
    // dispatch, so mismatches are rejected here with a 400 instead of being
    // dropped post-ack by the batch processor (#951 symptom B).
    if let Some(expected) = existing_dimension {
        for record in &records {
            if let Some(cell) = record.embeddings.first()
                && cell.dim != expected
            {
                return Err(ApiError::InvalidArgument(format!(
                    "record '{}' has dimension {} but collection '{}' expects dimension {}",
                    record.oid, cell.dim, collection, expected
                )));
            }
        }
    }

    // Phase 2H: async mode + queue available → enqueue to embed-ingest
    // topic and return 202 fast. Inline embedding only runs on the sync
    // path or as a degradation when no queue is configured.
    //
    // #951: only fast-ack onto the queue when the target collection already
    // exists — a missing collection falls through to the inline path below,
    // which embeds first and can therefore auto-create the collection at the
    // real embedding dimension before inserting.
    if mode == "async"
        && embed_source == "native"
        && let Some(expected_dimension) = existing_dimension
        && records.iter().all(|record| record.embeddings.is_empty())
        && let Some(queue) = state.queue_client.clone()
    {
        let embedding_route_identity =
            admit_native_async(&records, &collection, &tenant_id, expected_dimension)?;
        let producer = queue.producer();
        // Build the EmbedIngestPayload directly from `records`'
        // props (where the text was hoisted during the build-loop
        // above). The drainer (Phase 2G) deserializes this exact
        // shape; the contract is frozen in
        // `services::embedding_drainer::EmbedIngestPayload`.
        let drainer_records: Vec<crate::services::EmbedIngestRecord> = records
            .iter()
            .map(|r| crate::services::EmbedIngestRecord {
                oid: r.oid.clone(),
                text: extract_text(r).unwrap_or_default(),
                // #950: carry the record's props (tenant_id stamp + user
                // metadata) through the queue envelope so the drainer-side
                // insert restores them. Previously this was an empty map, so
                // every queued record lost its metadata AND its queryable
                // tenant stamp.
                metadata: envelope_metadata(r),
            })
            .collect();
        let payload = crate::services::EmbedIngestPayload {
            target_collection: collection.clone(),
            tenant_id: tenant_id.clone(),
            embedding_route_identity,
            expected_dimension,
            records: drainer_records,
        };
        let payload_bytes = serde_json::to_vec(&payload)
            .map_err(|e| ApiError::Internal(format!("queue payload serialize: {e}")))?;
        let msg = proximadb_queue::Message::new(
            crate::services::EMBED_INGEST_TOPIC,
            tenant_id.clone(),
            payload_bytes,
        );
        match producer.send(msg).await {
            Ok(_receipt) => {
                let response = IngestDocumentsResponse {
                    mode: "async".to_string(),
                    records: records
                        .iter()
                        .map(|r| IngestedRecord {
                            id: r.oid.clone(),
                            dim: expected_dimension,
                        })
                        .collect(),
                    retry_after_ms: None,
                };
                return Ok((StatusCode::ACCEPTED, response));
            }
            Err(proximadb_queue::QueueError::Backpressure {
                pct,
                retry_after_ms,
            }) => {
                warn!(pct, retry_after_ms, "v3 async ingest: queue backpressure");
                // ResourceExhausted maps to HTTP 503 in the
                // existing ApiError IntoResponse; for queue
                // backpressure the convention is 429 — the response
                // body carries `retry_after_ms` so the SDK knows
                // how long to back off. Translating to 429
                // semantically is a follow-up touch on the
                // IntoResponse impl.
                return Err(ApiError::ResourceExhausted(format!(
                    "queue at {pct:.0}% capacity; retry after {retry_after_ms}ms"
                )));
            }
            Err(e) => {
                warn!(error = %e, "v3 async ingest: queue.send failed");
                return Err(ApiError::Internal(format!("queue.send: {e}")));
            }
        }
        // No queue configured — fall through to inline embed below.
    }

    // Dispatch text-only records through the shared embedding helper. Records
    // that already carry an SDK-provided vector skip the dispatch entirely.
    if embed_source == "native" {
        ProximaFlightService::embed_text_only_records(&mut records, Some(&tenant_id))
            .await
            .map_err(|e| {
                warn!(error = %e, "v3 documents: embedding dispatch failed");
                ApiError::Internal(format!("embedding dispatch failed: {e}"))
            })?;
    }

    // Reject any records still missing an embedding (caller asked for
    // sdk-vector but didn't provide one, or the dispatcher couldn't extract
    // text). Better to fail loudly than to silently drop to defaults.
    let missing: Vec<&str> = records
        .iter()
        .filter(|r| r.embeddings.is_empty())
        .map(|r| r.oid.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(ApiError::InvalidArgument(format!(
            "{} record(s) have no vector and no text to embed: {}",
            missing.len(),
            missing.join(", ")
        )));
    }

    // #951: post-embed the dimension of every record is known. Validate it
    // against the collection (400 on mismatch, never accept-then-drop), or —
    // when the collection doesn't exist — auto-create it at the embedding
    // dimension the handler just produced.
    let embed_dim = records
        .first()
        .and_then(|r| r.embeddings.first())
        .map(|cell| cell.dim)
        .unwrap_or(0);
    if let Some(mismatch) = records.iter().find(|r| {
        r.embeddings
            .first()
            .is_some_and(|cell| cell.dim != embed_dim)
    }) {
        return Err(ApiError::InvalidArgument(format!(
            "record '{}' has dimension {} but the batch embeds at dimension {embed_dim}; \
             a single /documents batch must be dimensionally uniform",
            mismatch.oid,
            mismatch.embeddings.first().map(|c| c.dim).unwrap_or(0),
        )));
    }
    match existing_dimension {
        Some(expected) => {
            if embed_dim != expected {
                return Err(ApiError::InvalidArgument(format!(
                    "records embed at dimension {embed_dim} but collection '{collection}' \
                     expects dimension {expected}"
                )));
            }
        }
        None => {
            auto_create_collection_at_dim(&state, &collection, embed_dim, &tenant_id).await?;
            info!(
                collection = %collection,
                dimension = embed_dim,
                tenant = %tenant_id,
                "v3 documents: auto-created missing collection at embed dimension"
            );
        }
    }

    let inserted_ids: Vec<IngestedRecord> = records
        .iter()
        .map(|r| IngestedRecord {
            id: r.oid.clone(),
            dim: r.embeddings.first().map(|e| e.dim).unwrap_or(0),
        })
        .collect();
    let row_count = records.len() as u64;

    // Async ack path: in Phase 1 the EmbeddingService dispatch is in-process
    // and synchronous from the request handler's perspective, so we don't yet
    // get a true sub-10ms ack from `mode=async`. Still set the response code
    // to 202 so clients build against the right contract; the latency
    // optimization (WAL pending_embed flag + background drain) lands when
    // the drainer task is registered in BackgroundMaintenanceManager.
    let intent = WriteIntent::new(&collection, WriteOperationKind::Insert)
        .with_durability(WriteDurabilityRequirement::WalRequired)
        .with_row_count_hint(row_count);
    let lane = WriteLaneRouter::new().route(&intent);
    debug!(
        collection_id = %collection,
        write_lane = ?lane.lane,
        guards = ?lane.required_guards,
        "v3 documents write-lane decision"
    );

    let batch_request = RichRecordBatchRequest {
        collection_id: collection.clone(),
        records,
    };

    match state
        .record_ops
        .handle_record_batch_for_tenant(batch_request, Some(&tenant_id))
        .await
    {
        // #951: `handle_record_batch_for_tenant` reports most failures as
        // `Ok(result)` with `success == false` (NOT_FOUND, schema/dimension
        // validation, WAL-lane rejection). Acking those with a 2xx is the
        // accept-then-drop contract violation — surface a real HTTP error.
        Ok(resp) if !resp.success => Err(batch_failure_to_api_error(&collection, &resp)),
        Ok(_resp) => {
            let status = if mode == "async" {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            };
            let response = IngestDocumentsResponse {
                mode,
                records: inserted_ids,
                retry_after_ms: None,
            };
            Ok((status, response))
        }
        // ADR-069 S4: a WalBackpressure in the error chain maps to a retryable
        // status (429 / RESOURCE_EXHAUSTED); anything else stays a 500. NOTE: the
        // primary batch path folds WAL-lane rejections into Ok(success=false)
        // (#951), so this arm sees backpressure only on Err-propagating inserts —
        // end-to-end 429 for the batch path is the TD-WAL-1 S4 residual.
        Err(e) => Err(crate::errors::api_error_from_write_error("Insert failed", e)),
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Validate all facts needed to make a durable native-embedding job
/// admissible before returning 202. The returned credential-free identity is
/// serialized into the queue payload; the drainer accepts credential rotation
/// but rejects any model, endpoint, or geometry drift before inference.
fn admit_native_async(
    records: &[ProximaRecord],
    collection: &str,
    tenant_id: &str,
    expected_dimension: u32,
) -> ApiResult<proximadb_embedding::config::EmbedRouteIdentity> {
    let missing_text: Vec<&str> = records
        .iter()
        .filter(|record| {
            extract_text(record)
                .as_deref()
                .is_none_or(|text| text.trim().is_empty())
        })
        .map(|record| record.oid.as_str())
        .collect();
    if !missing_text.is_empty() {
        return Err(ApiError::InvalidArgument(format!(
            "{} record(s) have no text to embed: {}",
            missing_text.len(),
            missing_text.join(", ")
        )));
    }

    let embed_service = proximadb_embedding::EmbeddingService::try_global().ok_or_else(|| {
        ApiError::Internal("native embedding service is not initialized".to_string())
    })?;
    let route = embed_service.resolve_route(tenant_id);
    let route_dimension = u32::try_from(route.dimension()).map_err(|_| {
        ApiError::Internal(format!(
            "embedding route dimension {} exceeds the supported range",
            route.dimension()
        ))
    })?;
    if route_dimension != expected_dimension {
        return Err(ApiError::InvalidArgument(format!(
            "native embedding route produces dimension {route_dimension} but collection \
             '{collection}' expects dimension {expected_dimension}"
        )));
    }
    Ok((&route).into())
}

/// #951: create a missing target collection at the embedding dimension the
/// handler just produced, through the same unified create path the v2
/// `POST /collections` handler uses (full registration + tenant tagging —
/// never the half-registered dimension-0 entity a bare catalog write leaves
/// behind).
async fn auto_create_collection_at_dim(
    state: &AppState,
    collection: &str,
    dimension: u32,
    tenant_id: &str,
) -> ApiResult<()> {
    use crate::proto::proximadb_v1::{
        CollectionConfig, CollectionOperation, CollectionRequest, DistanceMetric,
    };

    if dimension == 0 {
        return Err(ApiError::InvalidArgument(format!(
            "cannot auto-create collection '{collection}' at dimension 0"
        )));
    }

    let collection_config = CollectionConfig {
        name: collection.to_string(),
        dimension,
        distance_metric: Some(DistanceMetric::Cosine as i32),
        enable_proxima_record: Some(true),
        ..Default::default()
    };
    let create_request = CollectionRequest {
        operation: CollectionOperation::CollectionCreate as i32,
        collection_id: Some(collection.to_string()),
        collection_config: Some(collection_config),
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    match state
        .api_handlers
        .handle_collection_operation_for_tenant(create_request, Some(tenant_id))
        .await
    {
        Ok(resp) if resp.success => Ok(()),
        Ok(resp) => Err(ApiError::Internal(format!(
            "auto-create of collection '{collection}' at dimension {dimension} failed: {}",
            resp.error_code
                .unwrap_or_else(|| "unknown error".to_string())
        ))),
        Err(e) => Err(ApiError::Internal(format!(
            "auto-create of collection '{collection}' at dimension {dimension} failed: {e}"
        ))),
    }
}

/// #951: map a non-success `BatchOperationResult` onto an honest HTTP error.
/// The batch layer folds validation failures into `Ok(result)` shapes; a 2xx
/// for any of them is an accept-then-drop ack violation.
fn batch_failure_to_api_error(
    collection: &str,
    resp: &crate::services::operations::BatchOperationResult,
) -> ApiError {
    let detail = if resp.errors.is_empty() {
        "insert failed".to_string()
    } else {
        resp.errors.join("; ")
    };
    match resp.error_code.as_deref() {
        Some("NOT_FOUND") => ApiError::NotFound(format!("Collection '{collection}' not found")),
        Some("SCHEMA_VALIDATION_FAILED") | Some("INSERT_CONFLICT") => {
            ApiError::InvalidArgument(detail)
        }
        // ADR-069 S4: write-admission backpressure — the batch was shed at the
        // critical watermark (`WAL_BACKPRESSURE`, raised inside the write path) or
        // the write lane rejected it pre-write (`WAL_LANE_REJECTED`, #951). Both
        // are *retryable*: surface 429 / RESOURCE_EXHAUSTED so the client backs off
        // instead of treating a shed write as a hard 400/500.
        Some("WAL_BACKPRESSURE") | Some("WAL_LANE_REJECTED") => ApiError::ResourceExhausted(detail),
        // Dimension violations surface as RECORD_INSERT_FAILED with the
        // validator's message — a client error, not a server fault.
        _ if detail.contains("dimension") => ApiError::InvalidArgument(detail),
        _ => ApiError::Internal(detail),
    }
}

/// #950: project a record's props into the queue envelope's string metadata
/// map so the drainer-side insert restores them (the server tenant stamp
/// included; `text` rides in the dedicated field). Values are stringified —
/// the envelope's frozen shape carries `HashMap<String, String>` — which is
/// lossy for non-string types but strictly better than the previous behavior
/// of dropping every prop on the queued path.
fn envelope_metadata(record: &ProximaRecord) -> HashMap<String, String> {
    record
        .props
        .iter()
        .filter(|(k, _)| k.as_str() != "text")
        .filter_map(|(k, v)| {
            let ProximaTreeNode::Value(value) = v else {
                return None;
            };
            let s = match value {
                ProximaValue::String(s) => s.clone(),
                ProximaValue::Boolean(b) => b.to_string(),
                ProximaValue::Int64(i) => i.to_string(),
                ProximaValue::Float64(f) => f.to_string(),
                ProximaValue::Json(j) => j.to_string(),
                _ => return None,
            };
            Some((k.clone(), s))
        })
        .collect()
}

/// Extract a record's text from `props["text"]` if present. The async
/// queue producer needs this to populate `EmbedIngestRecord::text` for
/// the drainer to embed downstream.
fn extract_text(record: &ProximaRecord) -> Option<String> {
    let value = record.props.get("text")?;
    if let ProximaTreeNode::Value(ProximaValue::String(s)) = value {
        Some(s.clone())
    } else {
        None
    }
}

fn json_to_proxima_value(value: serde_json::Value) -> ProximaValue {
    match value {
        // ProximaValue has no Null variant — represent JSON null as empty
        // string so the metadata key survives without crashing the type
        // system. Callers can skip null fields earlier if they prefer.
        serde_json::Value::Null => ProximaValue::String(String::new()),
        serde_json::Value::Bool(b) => ProximaValue::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ProximaValue::Int64(i)
            } else if let Some(f) = n.as_f64() {
                ProximaValue::Float64(f)
            } else {
                ProximaValue::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => ProximaValue::String(s),
        // Arrays and objects get preserved as full JSON so downstream
        // query/filter layers retain structural fidelity.
        v @ (serde_json::Value::Array(_) | serde_json::Value::Object(_)) => ProximaValue::Json(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::operations::BatchOperationResult;
    use proximadb_embedding::config::{EmbedRoute, EmbeddingConfig};
    use proximadb_embedding::scheduler::EmbedSchedulerConfig;

    fn record_with_props(entries: &[(&str, ProximaValue)]) -> ProximaRecord {
        ProximaRecord {
            oid: "r1".to_string(),
            props: entries
                .iter()
                .map(|(k, v)| (k.to_string(), ProximaTreeNode::Value(v.clone())))
                .collect(),
            ..ProximaRecord::default()
        }
    }

    /// #950: the queue envelope must carry the tenant stamp and user metadata
    /// (stringified), with `text` excluded (it rides the dedicated field).
    #[test]
    fn envelope_metadata_carries_tenant_and_user_props_but_not_text() {
        let record = record_with_props(&[
            ("text", ProximaValue::String("body".to_string())),
            ("tenant_id", ProximaValue::String("demo1".to_string())),
            ("kind", ProximaValue::String("doc".to_string())),
            ("rank", ProximaValue::Int64(7)),
            ("hot", ProximaValue::Boolean(true)),
        ]);
        let metadata = envelope_metadata(&record);
        assert_eq!(metadata.get("tenant_id").map(String::as_str), Some("demo1"));
        assert_eq!(metadata.get("kind").map(String::as_str), Some("doc"));
        assert_eq!(metadata.get("rank").map(String::as_str), Some("7"));
        assert_eq!(metadata.get("hot").map(String::as_str), Some("true"));
        assert!(
            !metadata.contains_key("text"),
            "text rides the envelope's dedicated field, not metadata"
        );
    }

    /// #951: non-success batch results map onto honest HTTP errors instead of
    /// the former blanket 2xx ack.
    #[test]
    fn batch_failure_maps_to_client_or_server_errors() {
        let not_found = BatchOperationResult::failure("gone".to_string(), "NOT_FOUND".to_string());
        assert!(matches!(
            batch_failure_to_api_error("docs", &not_found),
            ApiError::NotFound(_)
        ));

        let dim = BatchOperationResult::failure(
            "Record at index 0 has dimension 384 but collection 'docs' expects dimension 1"
                .to_string(),
            "RECORD_INSERT_FAILED".to_string(),
        );
        assert!(matches!(
            batch_failure_to_api_error("docs", &dim),
            ApiError::InvalidArgument(_)
        ));

        let schema = BatchOperationResult::failure(
            "Schema validation failed: bad column".to_string(),
            "SCHEMA_VALIDATION_FAILED".to_string(),
        );
        assert!(matches!(
            batch_failure_to_api_error("docs", &schema),
            ApiError::InvalidArgument(_)
        ));

        let opaque =
            BatchOperationResult::failure("disk on fire".to_string(), "IO_ERROR".to_string());
        assert!(matches!(
            batch_failure_to_api_error("docs", &opaque),
            ApiError::Internal(_)
        ));
    }

    fn ensure_embedding_singleton() {
        if proximadb_embedding::EmbeddingService::try_global().is_some() {
            return;
        }
        let _ = proximadb_embedding::EmbeddingService::initialize(
            EmbeddingConfig {
                route: EmbedRoute::BgeSmall,
            },
            EmbedSchedulerConfig::default(),
        );
    }

    #[test]
    fn async_native_admission_rejects_route_dimension_mismatch() {
        ensure_embedding_singleton();
        let record = record_with_props(&[(
            "text",
            ProximaValue::String("dimensionally checked before enqueue".to_string()),
        )]);

        let result = admit_native_async(&[record], "docs", "tenant-admission", 1024);

        assert!(matches!(result, Err(ApiError::InvalidArgument(_))));
    }

    #[test]
    fn async_native_admission_rejects_blank_text() {
        ensure_embedding_singleton();
        let record = record_with_props(&[("text", ProximaValue::String("   ".to_string()))]);

        let result = admit_native_async(&[record], "docs", "tenant-admission", 384);

        assert!(matches!(result, Err(ApiError::InvalidArgument(_))));
    }
}
