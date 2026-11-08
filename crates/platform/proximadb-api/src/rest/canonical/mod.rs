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

pub mod document;
pub mod entities;
pub mod graph;
pub mod graph_tenant_scope;
pub mod hybrid;
pub mod multimodal_query;
pub mod observability;

/// Shared default for pagination `limit` fields across canonical REST endpoints.
pub(crate) const DEFAULT_LIMIT: u32 = 100;

/// Serde-compatible accessor for [`DEFAULT_LIMIT`] (serde `default = "..."` needs a fn path).
pub(crate) fn default_limit() -> u32 {
    DEFAULT_LIMIT
}

// Deprecation-header utilities live in `crate::rest::deprecation` (a sibling
// module) and are re-exported here so the adapters' `super::with_v1_compatibility_headers`
// call sites resolve. See TD-V1SUNSET-1 step 2.
pub use crate::rest::deprecation::{
    REST_V1_DEPRECATION_MESSAGE, REST_V1_REPLACEMENT_SURFACE,
    add_compatibility_deprecation_headers, add_rest_v1_deprecation_headers,
    apply_v1_deprecation_headers, with_v1_compatibility_headers,
};

// Re-exports — state types
pub use document::DocumentRestState;
pub use graph::GraphRestState;
pub use hybrid::HybridRestState;
pub use multimodal_query::UnifiedQueryRestState;
pub use observability::ObservabilityRestState;

// Re-exports — router builders
pub use document::create_document_router;
pub use graph::create_graph_router;
pub use hybrid::{create_health_router, create_hybrid_search_router, execute_sql};
pub use multimodal_query::create_multimodal_router;
pub use observability::create_observability_router;

// Re-exports — handler functions (for direct registration in existing root-crate routers)
pub use hybrid::{liveness_check, readiness_check};

#[cfg(test)]
mod pagination_contract_tests {
    use super::{document::DocumentQueryParams, observability::LogQueryRequest};
    use axum::{extract::Query, http::Uri};
    use serde_json::json;

    #[test]
    fn missing_limits_default_and_explicit_limits_are_preserved() {
        for (limit, expected) in [
            (None, 100),
            (Some(0), 0),
            (Some(17), 17),
            (Some(u32::MAX), u32::MAX),
        ] {
            let (uri, payload) = match limit {
                None => ("/documents".to_string(), json!({})),
                Some(limit) => (format!("/documents?limit={limit}"), json!({"limit": limit})),
            };
            let uri: Uri = uri.parse().unwrap();
            let document = Query::<DocumentQueryParams>::try_from_uri(&uri).unwrap().0;
            let logs: LogQueryRequest = serde_json::from_value(payload).unwrap();
            assert_eq!(document.limit, expected);
            assert_eq!(logs.limit, expected);
        }
    }

    #[test]
    fn invalid_limits_are_rejected_instead_of_defaulted_or_truncated() {
        for payload in [
            json!({"limit": null}),
            json!({"limit": -1}),
            json!({"limit": u64::from(u32::MAX) + 1}),
            json!({"limit": "100"}),
            json!({"limit": 1.5}),
        ] {
            assert!(serde_json::from_value::<LogQueryRequest>(payload).is_err());
        }
        // Documents use query strings; a numeric text value such as "100" is valid.
        for limit in ["", "-1", "4294967296", "invalid", "1.5", "null"] {
            let uri: Uri = format!("/documents?limit={limit}").parse().unwrap();
            assert!(Query::<DocumentQueryParams>::try_from_uri(&uri).is_err());
        }
    }
}
