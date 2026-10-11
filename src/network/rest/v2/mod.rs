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

//! REST API v2 - ProximaRecord and typed schema support
//!
//! This module provides the v2 REST API handlers for ProximaDB, introducing
//! ProximaRecord as the primary record type with full type system support.
//!
//! ## New Endpoints
//!
//! - `POST /api/v2/collections` - Create collection with schema
//! - `PUT /api/v2/collections/{id}/schema` - Update collection schema
//! - `GET /api/v2/collections/{id}/schema` - Get collection schema
//! - `POST /api/v2/collections/{collection}/records/batch` - Insert ProximaRecords
//! - `DELETE /api/v2/collections/{collection}/records/{id}` - Delete a ProximaRecord
//! - `POST /api/v2/collections/{collection}/search` - Search with typed filters
//! - `POST /api/v2/query` - Execute AQL/UQL through the shared query facade
//! - `POST /api/v2/sql` - Execute one SQL statement through the shared SQL authority
//!
//! ## Key Features
//!
//! - **Typed Fields**: Full support for TEXT, INTEGER, FLOAT, DECIMAL, UUID, etc.
//! - **Schema Enforcement**: Strict, Flexible, or Hybrid modes
//! - **TEXT Column Storage**: Dedicated columnar storage for large text fields
//! - **Typed Filtering**: Range, equality, and CONTAINS filters with type safety
//!
//! ## Migration from v1
//!
//! The v2 API is designed for gradual migration. Collections can enable
//! ProximaRecord support via the `enable_proxima_record` flag, allowing
//! mixed v1/v2 operations during transition.

pub mod collections;
pub mod discovery;
pub mod documents;
pub mod entities;
pub mod external_collection;
pub mod graphs;
pub mod model_registry;
pub mod query;
pub mod records;
pub mod schema;
pub mod sql;
pub mod timeseries;

pub use collections::*;
pub use query::*;
pub use records::*;
pub use schema::*;

use axum::Router;

use super::canonical::handlers::AppState;

/// Create the v2 API router with all endpoints
///
/// This router provides the v2 REST API endpoints for ProximaRecord operations.
/// It should be nested under `/api/v2` in the main application router.
///
/// ## Endpoints
///
/// ### Collections
/// - `POST /collections` - Create collection with schema
/// - `GET /collections` - List collections with pagination
/// - `GET /collections/:collection_id` - Get collection details
///
/// ### Schema
/// - `GET /collections/:collection_id/schema` - Get collection schema
/// - `PUT /collections/:collection_id/schema` - Update collection schema
///
/// ### Records
/// - `POST /collections/:collection_id/records/batch` - Batch insert ProximaRecords
/// - `GET /collections/:collection_id/records/:record_id` - Get single record
/// - `POST /collections/:collection_id/search` - Search with typed filters
///
/// ### Query
/// - `POST /query` - Execute AQL/UQL through UnifiedQueryPort
/// - `POST /query/explain` - Explain AQL/UQL through UnifiedQueryPort
/// - `POST /sql` - Execute one authenticated SQL statement through ApiHandlersPort
pub fn create_v2_router() -> Router<AppState> {
    use axum::routing::{delete, get, post, put};

    let router = Router::new()
        // Collection operations with schema support
        .route("/collections", post(collections::create_collection_v2))
        .route("/collections", get(collections::list_collections_v2))
        .route(
            "/collections/{collection_id}",
            get(collections::get_collection_v2).delete(collections::delete_collection_v2),
        )
        // Entity operations (orchestration facade over graph+vector+document)
        .route(
            "/collections/{collection_id}/entities",
            post(entities::upsert_entity_v2),
        )
        .route(
            "/collections/{collection_id}/entities/search",
            post(entities::search_entities_v2),
        )
        .route(
            "/collections/{collection_id}/entities/{entity_id}",
            get(entities::get_entity_v2).delete(entities::delete_entity_v2),
        )
        // Schema management - separate routes for GET and PUT
        .route(
            "/collections/{collection_id}/schema",
            get(schema::get_schema),
        )
        .route(
            "/collections/{collection_id}/schema",
            put(schema::update_schema),
        )
        // Record operations
        .route(
            "/collections/{collection_id}/records/batch",
            post(records::insert_records),
        )
        // TD-099: paginated scan — SHIPPED (storage-engine delegation landed in
        // c8050ab1a: handler drives `handle_record_scan_for_tenant` → the shared
        // WAL/memtable scan, cursor-paginated; see records.rs + services/scan_cursor.rs).
        .route(
            "/collections/{collection_id}/records/scan",
            post(records::scan_records),
        )
        .route(
            "/collections/{collection_id}/records/{record_id}",
            get(records::get_record_v2).delete(records::delete_record_v2),
        )
        // Search with typed filters
        .route(
            "/collections/{collection_id}/search",
            post(records::search_with_typed_filters),
        )
        // CDC change-feed: row-level changes since an LSN cursor (unified across surfaces).
        .route(
            "/collections/{collection_id}/changes",
            get(records::get_changes),
        )
        // Document ingest (text-only / native server-side embedding). Folded in
        // from the former v3 surface (v3 is removed; this is the canonical route).
        .route(
            "/collections/{collection_id}/documents",
            post(documents::ingest_documents),
        )
        // Query facade operations
        .route("/query", post(query::execute_query))
        .route("/query/explain", post(query::explain_query))
        // Canonical HTTP/JSON SQL adapter. This owns transport validation and
        // result shaping only; execution is the same shared authority as gRPC.
        .route("/sql", post(sql::execute_sql))
        // Tenant-scoped xCatalog embedding-model lifecycle.
        .route(
            "/model-registries",
            post(model_registry::create_model_registry).get(model_registry::list_model_registries),
        )
        .route(
            "/model-registries/{name}",
            get(model_registry::get_model_registry),
        )
        .route(
            "/model-registries/{name}/mutations",
            post(model_registry::apply_model_registry_mutation),
        )
        .route(
            "/model-registries/{name}/resolve",
            post(model_registry::resolve_model_alias),
        )
        // Time-series surface (TD-TS-1) — native TST engine over the SDK contract.
        .route(
            "/timeseries/collections",
            post(timeseries::create_timeseries_collection),
        )
        .route(
            "/timeseries/collections",
            get(timeseries::list_timeseries_collections),
        )
        .route(
            "/timeseries/collections/{collection_id}",
            delete(timeseries::delete_timeseries_collection),
        )
        .route(
            "/timeseries/collections/{collection_id}/ingest",
            post(timeseries::ingest_timeseries),
        )
        .route(
            "/timeseries/collections/{collection_id}/query",
            post(timeseries::query_timeseries),
        )
        .route(
            "/timeseries/collections/{collection_id}/aggregate",
            post(timeseries::aggregate_timeseries),
        )
        // Cross-modal fusion seam — graph instance (TD-137): vector seed → graph
        // expand → calibrated fuse-by-oid.
        .route(
            "/graphs/{graph_id}/fusion-search",
            post(graphs::fusion_search_v2),
        )
        // TD-131 — graph impact analysis (forward/backward blast radius).
        .route(
            "/graphs/{graph_id}/impact-analysis",
            post(graphs::impact_analysis_v2),
        )
        // Phase 8 (F1) — Continuous Discovery jobs (experimental).
        .route(
            "/collections/{collection_id}/discovery-jobs",
            post(discovery::create_discovery_job_v2),
        )
        .route(
            "/collections/{collection_id}/discovery-jobs",
            get(discovery::list_discovery_jobs_v2),
        )
        .route(
            "/collections/{collection_id}/discovery-jobs/{job_id}",
            get(discovery::get_discovery_job_v2),
        )
        // Phase 8 (F5) — External Collections: index external lake data un-copied.
        .route(
            "/external-collections",
            post(external_collection::register_external_collection_v2)
                .get(external_collection::list_external_collections_v2),
        )
        .route(
            "/external-collections/{id}",
            get(external_collection::get_external_collection_v2),
        )
        .route(
            "/external-collections/{id}/build",
            post(external_collection::build_external_collection_v2),
        )
        .route(
            "/external-collections/{id}/search",
            post(external_collection::search_external_collection_v2),
        )
        .route(
            "/external-collections/{id}/refresh",
            post(external_collection::refresh_external_collection_v2),
        )
        // Diagnostics — experimental capability contract endpoints.
        // Namespaced under `_diagnostics` while the shape stabilizes;
        // promotion to `/collections/:id/route-health` is intentional
        // future work, not an oversight.
        .route(
            "/_diagnostics/collections/{collection_id}/route-health",
            get(collections::get_collection_route_health_v2),
        )
        // ADR-037 (TD-174) — agent-facing statistics envelope: the
        // modality-neutral, units-only boundary object the agent catalog
        // consumes. Read from the resident summary maintained at the
        // flush/compaction write boundary (never a corpus scan).
        .route(
            "/_diagnostics/collections/{collection_id}/statistics",
            get(collections::get_collection_statistics_v2),
        );
    // Adaptive HNSW retune. AXIS-only — the handler and this route are
    // elided when the `axis` feature is off. POST resolves
    // DriftKind::EfSearchOnly in-place via
    // AxisManager::apply_hnsw_ef_hot_swap; reports
    // DriftKind::EfConstructionOrM cases as "rebuild required"
    // (operator must run /recluster — separate slice).
    #[cfg(feature = "axis")]
    let router = router.route(
        "/_diagnostics/collections/{collection_id}/recall-tune",
        axum::routing::post(collections::post_collection_recall_tune_v2),
    );
    router
        // Recall-aware HNSW rebuild. POST reads every record for the
        // collection, runs the advisor at the live N, and atomically
        // swaps in a new HNSW graph sized for the recall_target tag.
        // This is the rebuild_required arm of the recall-drift
        // workflow — the only path that resolves m / ef_construction
        // drift (hot-swap handles ef_search alone).
        .route(
            "/_diagnostics/collections/{collection_id}/recluster",
            axum::routing::post(collections::post_collection_recluster_v2),
        )
        // Phase 8 (F4a) — single-node collection suspend/resume (TD-094).
        .route(
            "/collections/{collection_id}/suspend",
            post(collections::post_collection_suspend_v2),
        )
        .route(
            "/collections/{collection_id}/resume",
            post(collections::post_collection_resume_v2),
        )
        // Capability negotiation: SDKs call this once to discover the server's
        // supported features + limits instead of hard-coding assumptions.
        .route("/_meta/capabilities", get(capabilities))
}

/// `GET /api/v2/_meta/capabilities` — advertise API version, features, and
/// limits so clients can negotiate behaviour. Static today; feature flags can
/// be threaded from `AppState` as they become conditional.
pub async fn capabilities() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "api_version": env!("CARGO_PKG_VERSION"),
        "surface": "v2",
        "features": [
            "collections", "records", "typed_search", "query_uql", "query_explain",
            "graphs", "hybrid_search", "document_collections", "observability",
            "index_configs", "tags", "request_id", "timeseries", "model_registry"
        ],
        "limits": {
            "max_batch_records": 10000,
            "max_request_size_mb": 64
        },
        "error_envelope": {
            "shape": "{ error: { type, message, code, request_id? } }",
            "request_id_header": "X-Request-ID"
        }
    }))
}
