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

//! Request ID Middleware
//!
//! This middleware extracts or generates request IDs for all incoming requests
//! and adds them to the response headers for client correlation.
//!
//! ## Features
//!
//! - Extracts `X-Request-ID` from incoming request headers
//! - Generates a UUID v4 if header is not present
//! - Stores the request ID as a request extension for handler access
//! - Adds `X-Request-ID` to response headers
//! - Creates a tracing span for the request

use axum::{
    http::{HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use std::task::{Context, Poll};
use tower::{Layer, Service};

use proximadb_kernel::uuid::Uuid;

/// The header name for request ID
pub const X_REQUEST_ID: &str = "x-request-id";

/// Request ID extension stored in request extensions
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

impl RequestId {
    /// Create a new request ID from a string
    pub fn new(id: String) -> Self {
        Self(id)
    }

    /// Generate a new random request ID
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Get the request ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Middleware function that extracts or generates request ID
///
/// This can be used with `axum::middleware::from_fn`:
///
/// ```rust,ignore
/// use axum::middleware;
/// use crate::network::middleware::request_id::request_id_middleware;
///
/// let app = Router::new()
///     .route("/api/v1/...", ...)
///     .layer(middleware::from_fn(request_id_middleware));
/// ```
pub async fn request_id_middleware(mut request: axum::extract::Request, next: Next) -> Response {
    // Extract or generate request ID
    let request_id = request
        .headers()
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map_or_else(RequestId::generate, |s| RequestId::new(s.to_string()));

    // Store in request extensions for handler access
    request.extensions_mut().insert(request_id.clone());

    // Create a tracing span for this request
    let span = tracing::info_span!(
        "http_request",
        request_id = %request_id,
        method = %request.method(),
        uri = %request.uri().path()
    );

    // Execute the request within the span
    let _guard = span.enter();

    // Scope the proximadb-api task-local for the whole request so error
    // envelopes (`RestError`/`AgenticApiError::into_response`) carry the SAME
    // id we advertise in the `X-Request-ID` response header below — no handler
    // signature changes required.
    let mut response = proximadb_api_error::REQUEST_ID
        .scope(request_id.0.clone(), next.run(request))
        .await;

    // Add request ID to response headers
    if let Ok(header_value) = HeaderValue::from_str(&request_id.0) {
        response.headers_mut().insert(X_REQUEST_ID, header_value);
    }

    response
}

/// Tower Layer for request ID middleware
///
/// This layer can be used with `tower::ServiceBuilder`:
///
/// ```rust,ignore
/// use tower::ServiceBuilder;
/// use crate::network::middleware::request_id::RequestIdLayer;
///
/// let service = ServiceBuilder::new()
///     .layer(RequestIdLayer::new())
///     .service(my_service);
/// ```
#[derive(Clone, Default)]
pub struct RequestIdLayer;

impl RequestIdLayer {
    /// Create a new request ID layer
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdService { inner }
    }
}

/// Tower Service that adds request ID to requests
#[derive(Clone)]
pub struct RequestIdService<S> {
    inner: S,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for RequestIdService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send,
    ReqBody: Send + 'static,
    ResBody: Default + Send + 'static,
{
    type Response = Response<ResBody>;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<ReqBody>) -> Self::Future {
        // Extract or generate request ID
        let request_id = request
            .headers()
            .get(X_REQUEST_ID)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map_or_else(RequestId::generate, |s| RequestId::new(s.to_string()));

        // Store in request extensions
        request.extensions_mut().insert(request_id.clone());

        let request_id_string = request_id.0.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let mut response = inner.call(request).await?;

            // Add request ID to response headers
            if let Ok(header_value) = HeaderValue::from_str(&request_id_string) {
                response.headers_mut().insert(X_REQUEST_ID, header_value);
            }

            Ok(response)
        })
    }
}

/// Helper trait for extracting RequestId from request extensions
pub trait RequestIdExt {
    /// Get the request ID from the request extensions
    fn request_id(&self) -> Option<&RequestId>;
}

impl<B> RequestIdExt for Request<B> {
    fn request_id(&self) -> Option<&RequestId> {
        self.extensions().get::<RequestId>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_generate() {
        let id = RequestId::generate();
        assert_eq!(id.as_str().len(), 36); // UUID format
    }

    #[test]
    fn test_request_id_new() {
        let id = RequestId::new("custom-id-123".to_string());
        assert_eq!(id.as_str(), "custom-id-123");
    }

    #[test]
    fn test_request_id_display() {
        let id = RequestId::new("test-id".to_string());
        assert_eq!(format!("{}", id), "test-id");
    }
}
