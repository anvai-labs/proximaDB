//! # Canonical REST API handlers (port-backed `/api/v2` surface)
//!
//! The canonical, port-backed REST router builders + handlers that serve
//! `/api/v2/*` — graph, document, observability, multimodal/unified query, and
//! hybrid search. Each router is constructed with a platform port
//! (`GraphPort`, `DocumentPort`, …) and wired into the live router at
//! `src/network/rest/canonical/handlers.rs` (`create_*_router`).
//!
//! Historically this module was named `rest::v1` (it reuses v1 proto *request*
//! types internally), which led to it being mistaken for the deprecated
//! `/api/v1` surface — it is **not**; it is the live canonical surface. Renamed
//! to `rest::canonical` to make that unambiguous (TD-V1SUNSET-1 follow-up).
//!
//! The deprecation-header machinery (for residual `/api/v1`-shaped traffic) is
//! centralized in `crate::rest::deprecation` (TD-V1SUNSET-1 step 2) and
//! re-exported below.

pub mod analytics;
pub mod catalog;
pub mod document;
pub mod entities;
pub mod graph;
pub mod hybrid;
pub mod multimodal_query;
pub mod observability;

// Deprecation-header utilities live in `crate::rest::deprecation` (a sibling
// module) and are re-exported here so the adapters' `super::with_v1_compatibility_headers`
// call sites resolve. See TD-V1SUNSET-1 step 2.
pub use crate::rest::deprecation::{
    REST_V1_DEPRECATION_MESSAGE, REST_V1_REPLACEMENT_SURFACE,
    add_compatibility_deprecation_headers, add_rest_v1_deprecation_headers,
    apply_v1_deprecation_headers, with_v1_compatibility_headers,
};

// Re-exports — handler types
pub use analytics::{AnalyticsHandler, AqlHandler};
pub use catalog::{CatalogHandler, CollectionHandler};
pub use document::{DocumentHandler, DocumentQueryHandler};
pub use entities::{EntityHandler, VectorHandler};
pub use graph::{GraphHandler, GraphTraversalHandler};
pub use hybrid::{HybridSearchHandler, ProgressiveSearchHandler};
pub use observability::{LogsHandler, MetricsHandler};

// Re-exports — state types
pub use analytics::AnalyticsRestState;
pub use document::DocumentRestState;
pub use graph::GraphRestState;
pub use hybrid::HybridRestState;
pub use multimodal_query::UnifiedQueryRestState;
pub use observability::ObservabilityRestState;

// Re-exports — router builders
pub use analytics::create_analytics_router;
pub use document::create_document_router;
pub use graph::create_graph_router;
pub use hybrid::{create_health_router, create_hybrid_search_router, execute_sql};
pub use multimodal_query::create_multimodal_router;
pub use observability::create_observability_router;

// Re-exports — handler functions (for direct registration in existing root-crate routers)
pub use catalog::{collection_operation, delete_collection, get_collection, list_collections};
pub use hybrid::{liveness_check, readiness_check};
