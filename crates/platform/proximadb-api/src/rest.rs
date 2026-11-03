//! # REST API Handlers
//!
//! HTTP/JSON API handlers via Axum framework.

pub mod canonical;
pub mod deprecation;
pub mod errors;
pub mod state;
pub mod v2;

use serde::Serialize;

// Re-export errors and state for consumers
pub use errors::{RestError, RestResult};
pub use state::{RestAppState, TenantContext};

// Re-export canonical REST handler + state types
pub use canonical::{
    DocumentHandler, DocumentQueryHandler, EntityHandler, GraphHandler, GraphTraversalHandler,
    HybridSearchHandler, LogsHandler, MetricsHandler, ProgressiveSearchHandler,
    UnifiedQueryRestState, VectorHandler,
};
pub use canonical::{DocumentRestState, GraphRestState, HybridRestState, ObservabilityRestState};

// Re-export canonical REST router builders
pub use canonical::{
    create_document_router, create_graph_router, create_health_router, create_hybrid_search_router,
    create_multimodal_router, create_observability_router,
};

/// REST API request context
#[derive(Debug, Clone)]
pub struct RestRequest {
    pub path: String,
    pub method: String,
    pub body: Option<serde_json::Value>,
    pub headers: Vec<(String, String)>,
}

/// REST API response
#[derive(Debug, Clone, Serialize)]
pub struct RestResponse {
    pub status: u16,
    pub body: serde_json::Value,
}

/// REST API handler
pub struct RestApiHandler {
    // Service dependencies will be added here
}

impl RestApiHandler {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for RestApiHandler {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: Move REST handlers from src/network/rest and src/api_handlers

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rest_request_carries_path_method_headers_and_optional_body() {
        let request = RestRequest {
            path: "/v1/collections".to_string(),
            method: "POST".to_string(),
            body: Some(json!({"name": "c1"})),
            headers: vec![("tenant-id".to_string(), "tenant-a".to_string())],
        };

        assert_eq!(request.path, "/v1/collections");
        assert_eq!(request.method, "POST");
        assert_eq!(request.body, Some(json!({"name": "c1"})));
        assert_eq!(
            request.headers,
            vec![("tenant-id".to_string(), "tenant-a".to_string())]
        );
    }

    #[test]
    fn rest_response_serializes_status_and_json_body() {
        let response = RestResponse {
            status: 202,
            body: json!({"accepted": true}),
        };

        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json, json!({"status": 202, "body": {"accepted": true}}));
    }

    #[test]
    fn rest_handler_default_matches_new() {
        let _from_new = RestApiHandler::new();
        let _from_default = RestApiHandler::default();
    }
}
