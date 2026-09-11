/*
 * Copyright 2025 ProximaDB
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

//! OpenAPI spec-from-code aggregation (TD-126 Phase 1).
//!
//! The publishable REST contract (`docs/openapi/proximadb-openapi.yaml`) is
//! generated from the annotated axum handlers rather than hand-maintained. This
//! removes one of the three hand-kept copies of the contract (handlers ↔ spec ↔
//! SDKs) and makes server↔spec divergence impossible: a CI drift gate
//! regenerates the document and fails if it differs from the committed copy
//! (see `tests/openapi_spec_gen.rs` and the `openapi-spec` CI job).
//!
//! ## What is generated vs. supplemented (Phase 1 scope)
//!
//! The **core v2 surface** — collection lifecycle, schema, records (insert /
//! scan / get / delete / search), and the AQL/UQL query facade — is generated
//! directly from `#[utoipa::path]` + `#[derive(ToSchema)]` annotations on the
//! live handlers in `src/network/rest/v2/`. That is the spec-from-code half and
//! is what the drift gate guards.
//!
//! The remaining published surfaces — `/health*`, `/api/v2/_meta/capabilities`,
//! the graph surface (handlers live in the `proximadb-api` crate and take
//! private DTOs), and the free-form `hybrid` / `document-collections` /
//! `observability` passthrough endpoints — are carried verbatim from a small,
//! checked-in supplement fragment (`src/network/rest/openapi_supplement.yaml`) and merged
//! on top of the generated document. These are tracked for code-annotation in
//! later TD-126 increments (their handlers either live in another crate or
//! return dynamic `serde_json::Value`); keeping them in the supplement ensures
//! the published spec does not regress the operations the SDK contract gates
//! depend on.
//!
//! The generated document is the source of truth for the core surface; the
//! supplement is the source of truth for the not-yet-annotated surfaces. The
//! drift gate covers the merged result, so neither half can silently diverge.

use serde_json::Value;
use utoipa::OpenApi;

use crate::network::rest::v2::{
    collections, documents, entities, graphs, model_registry, query, records, schema, sql,
};

/// Canonical ProximaDB error envelope (`{ error: { type, message, code } }`).
///
/// `request_id` (also returned in the `X-Request-ID` response header) is present
/// whenever the request passed through the request-id middleware; quote it in
/// bug reports.
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

/// Inner body of [`ErrorResponse`].
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct ErrorBody {
    /// Stable machine-readable error code (snake_case).
    #[schema(example = "collection_not_found")]
    pub r#type: String,
    pub message: String,
    /// HTTP status code.
    pub code: i32,
    /// Correlation id (matches the X-Request-ID header).
    pub request_id: Option<String>,
    /// Optional structured context (e.g. migration hints).
    #[schema(value_type = Option<Object>)]
    pub details: Option<Value>,
}

/// Query facade result. Implementations return records, total_count, metrics,
/// plan, or diagnostics depending on language and endpoint, so the body is a
/// free-form JSON object.
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
#[schema(value_type = Object)]
pub struct QueryResponse(pub Value);

/// Aggregated OpenAPI document for the spec-from-code core surface.
///
/// Lists every annotated v2 handler and the components they reference. The
/// info / servers / security / tag metadata mirrors the published contract.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "ProximaDB REST API",
        version = "0.2.0",
        description = "SDK-facing ProximaDB REST contract. This specification is v2-first and \
centers on ProximaRecord, typed schemas, record search, and collection \
lifecycle operations. Legacy v1 compatibility routes are intentionally not \
part of this publishable SDK surface.",
    ),
    servers(
        (url = "http://localhost:5678", description = "Local ProximaDB server"),
    ),
    paths(
        collections::create_collection_v2,
        collections::list_collections_v2,
        collections::get_collection_v2,
        // NOTE: get_collection_statistics_v2 is intentionally NOT in the
        // published OpenAPI spec — it lives under `/api/v2/_diagnostics/...`,
        // and `_diagnostics` routes (route-health / recall-tune / recluster) are
        // excluded by convention. The route stays registered (rest/v2/mod.rs);
        // the agent-facing stats surface is the reference MCP `stats` tool.
        collections::delete_collection_v2,
        schema::get_schema,
        schema::update_schema,
        records::insert_records,
        records::scan_records,
        records::get_record_v2,
        records::delete_record_v2,
        records::search_with_typed_filters,
        query::execute_query,
        query::explain_query,
        sql::execute_sql,
        graphs::fusion_search_v2,
        graphs::impact_analysis_v2,
        entities::upsert_entity_v2,
        entities::get_entity_v2,
        entities::delete_entity_v2,
        entities::search_entities_v2,
        // Canonical document-ingest surface (ADR-041). Native server-side embedding.
        documents::ingest_documents,
        model_registry::create_model_registry,
        model_registry::list_model_registries,
        model_registry::get_model_registry,
        model_registry::apply_model_registry_mutation,
        model_registry::resolve_model_alias,
    ),
    components(
        schemas(
            ErrorResponse,
            ErrorBody,
            QueryResponse,
            sql::SqlRequest,
            sql::SqlResponse,
            entities::UpsertEntityRequest,
            entities::UpsertEntityResponse,
            entities::EntityDto,
            entities::SearchEntitiesRequest,
            entities::SearchEntitiesResponse,
            entities::EntitySearchResult,
            // Canonical document-ingest surface (ADR-041).
            documents::IngestDocument,
            documents::IngestDocumentsRequest,
            documents::IngestedRecord,
            documents::IngestDocumentsResponse,
            model_registry::CreateModelRegistryRequest,
            model_registry::ModelRegistryRecordResponse,
            model_registry::ListModelRegistriesResponse,
            model_registry::ApplyModelRegistryMutationRequest,
            model_registry::ResolveModelAliasRequest,
            model_registry::ResolvedEmbeddingModelResponse,
            model_registry::ResolvedModelBindingResponse,
        ),
    ),
    tags(
        (name = "Collections", description = "Collection lifecycle."),
        (name = "Schema", description = "Typed schema management."),
        (name = "Records", description = "ProximaRecord write / read / scan."),
        (name = "Search", description = "Vector similarity + typed-filter search."),
        (name = "Query", description = "AQL / UQL query facade."),
        (name = "SQL", description = "Authenticated single-statement SQL transport."),
        (name = "Entities", description = "Entity orchestration (graph node + embeddings + provenance)."),
        (name = "Documents", description = "Native-embedding document ingest."),
        (name = "Model Registry", description = "Tenant-scoped embedding-model lifecycle backed by xCatalog."),
    ),
)]
pub struct ApiDoc;

/// The committed supplement fragment carrying the surfaces that are not yet
/// annotated in code (see the module docs for scope). Checked in alongside the
/// generated spec so the generator is hermetic.
const SUPPLEMENT_YAML: &str = include_str!("openapi_supplement.yaml");

/// Deep-merge `overlay` into `base` (objects merge key-wise; arrays / scalars in
/// `overlay` replace `base`). Used to fold the supplement onto the generated
/// document.
fn merge_into(base: &mut Value, overlay: &Value) {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (key, overlay_val) in overlay_map {
                merge_into(
                    base_map.entry(key.clone()).or_insert(Value::Null),
                    overlay_val,
                );
            }
        }
        (base_slot, overlay_val) => {
            *base_slot = overlay_val.clone();
        }
    }
}

/// Build the merged OpenAPI document (generated core ⊕ supplement) as JSON.
fn openapi_document() -> Result<Value, String> {
    let mut doc: Value = serde_json::to_value(ApiDoc::openapi())
        .map_err(|e| format!("serialize generated OpenAPI document: {e}"))?;

    // Default top-level security for the whole surface. utoipa applies
    // `security([])` per-operation; the published contract declares a default
    // bearer scheme at the document root with explicit `security: []` opt-outs
    // on the public probes (carried by the supplement).
    if let Value::Object(map) = &mut doc {
        map.insert(
            "security".to_string(),
            serde_json::json!([{ "bearerAuth": [] }]),
        );
        // bearerAuth security scheme.
        let components = map
            .entry("components".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if let Value::Object(components_map) = components {
            let schemes = components_map
                .entry("securitySchemes".to_string())
                .or_insert_with(|| serde_json::json!({}));
            merge_into(
                schemes,
                &serde_json::json!({
                    "bearerAuth": { "type": "http", "scheme": "bearer", "bearerFormat": "JWT" }
                }),
            );
        }
    }

    let supplement: Value = serde_yaml::from_str(SUPPLEMENT_YAML)
        .map_err(|e| format!("parse openapi_supplement.yaml: {e}"))?;
    merge_into(&mut doc, &supplement);

    // Formalize the tenant selector as an OPTIONAL `X-Tenant-ID` header on EVERY operation
    // (generated core ⊕ supplement) — one place, zero per-handler churn, and it sidesteps the
    // `Option<String>: Display` codegen error that blocks a per-handler `params(...)` header.
    // Isolation is structural on the server (the tenant is scoped into the storage key); this
    // header only SELECTS the tenant when there is no authenticated context.
    inject_tenant_header(&mut doc);
    Ok(doc)
}

/// Inject the optional `X-Tenant-ID` header parameter into every operation of the merged document.
///
/// Runs AFTER the supplement merge so BOTH the generated core paths and the supplement-carried
/// paths (`/health*`, hybrid, observability, …) carry it. Idempotent — never double-adds. Applied
/// document-wide because tenant selection is uniform across the REST surface (mirrors the
/// document-root `security` injection above).
fn inject_tenant_header(doc: &mut Value) {
    const METHODS: [&str; 7] = ["get", "put", "post", "delete", "patch", "options", "head"];
    let header = serde_json::json!({
        "name": "X-Tenant-ID",
        "in": "header",
        "required": false,
        "description": "Optional explicit tenant selector. Applied only when there is no \
            authenticated tenant context — a JWT tenant claim takes precedence, and a header that \
            disagrees with the authenticated tenant is rejected. When absent, a single-tenant \
            deployment selects its configured default tenant; a multi-tenant deployment rejects \
            the request. Tenant isolation is structural on the server; this header only selects \
            the tenant.",
        "schema": { "type": "string" }
    });
    let Some(paths) = doc.get_mut("paths").and_then(Value::as_object_mut) else {
        return;
    };
    for path_item in paths.values_mut() {
        let Some(item) = path_item.as_object_mut() else {
            continue;
        };
        for method in METHODS {
            let Some(op) = item.get_mut(method).and_then(Value::as_object_mut) else {
                continue;
            };
            let params = op
                .entry("parameters".to_string())
                .or_insert_with(|| serde_json::json!([]));
            if let Some(arr) = params.as_array_mut() {
                let already = arr.iter().any(|p| {
                    p.get("name").and_then(Value::as_str) == Some("X-Tenant-ID")
                        && p.get("in").and_then(Value::as_str) == Some("header")
                });
                if !already {
                    arr.push(header.clone());
                }
            }
        }
    }
}

/// Render the merged OpenAPI document to canonical YAML (the on-disk form of
/// `docs/openapi/proximadb-openapi.yaml`).
pub fn openapi_yaml() -> Result<String, String> {
    let doc = openapi_document()?;
    serde_yaml::to_string(&doc).map_err(|e| format!("serialize OpenAPI document to YAML: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_builds_and_carries_core_operations() {
        let doc = openapi_document().expect("openapi document builds");
        let paths = doc
            .get("paths")
            .and_then(Value::as_object)
            .expect("paths object present");

        // A few spec-from-code core operations are present with the right ids.
        let create = &paths["/api/v2/collections"]["post"]["operationId"];
        assert_eq!(create, "createCollection");
        let search = &paths["/api/v2/collections/{collection_id}/search"]["post"]["operationId"];
        assert_eq!(search, "searchRecords");

        // Supplement-carried operations survive the merge.
        assert!(paths.contains_key("/health/live"));
        assert!(paths.contains_key("/api/v2/graphs"));
    }

    #[test]
    fn every_operation_carries_the_optional_tenant_header() {
        let doc = openapi_document().expect("openapi document builds");
        let paths = doc
            .get("paths")
            .and_then(Value::as_object)
            .expect("paths object present");
        const METHODS: [&str; 7] = ["get", "put", "post", "delete", "patch", "options", "head"];

        let mut checked = 0usize;
        for (route, item) in paths {
            let Some(item) = item.as_object() else {
                continue;
            };
            for method in METHODS {
                let Some(op) = item.get(method).and_then(Value::as_object) else {
                    continue;
                };
                let params = op
                    .get("parameters")
                    .and_then(Value::as_array)
                    .unwrap_or_else(|| panic!("{method} {route} should carry parameters"));
                let tenant = params.iter().find(|p| {
                    p.get("name").and_then(Value::as_str) == Some("X-Tenant-ID")
                        && p.get("in").and_then(Value::as_str) == Some("header")
                });
                let tenant =
                    tenant.unwrap_or_else(|| panic!("{method} {route} missing X-Tenant-ID header"));
                assert_eq!(
                    tenant.get("required").and_then(Value::as_bool),
                    Some(false),
                    "{method} {route}: X-Tenant-ID must be optional"
                );
                let description = tenant
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                assert!(
                    description.contains("single-tenant")
                        && description.contains("multi-tenant")
                        && description.contains("rejected"),
                    "{method} {route}: tenant default must be deployment-qualified"
                );
                checked += 1;
            }
        }
        // Covers the generated core (20 ops) + supplement operations.
        assert!(checked >= 20, "expected many operations, checked {checked}");
    }
}
