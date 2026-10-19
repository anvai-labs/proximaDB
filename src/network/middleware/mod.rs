/*
 * Copyright 2025 Vijaykumar Singh
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

//! HTTP middleware for ProximaDB
//!
//! This module provides various middleware layers for HTTP request processing,
//! including authentication, rate limiting, CORS, timeout, and metrics collection.
//!
//! # Security Middleware
//!
//! All security middleware is designed with defense-in-depth principles:
//! - **CORS**: Whitelist-based origin control (default: deny all cross-origin)
//! - **Rate Limiting**: Token bucket with per-IP and global limits
//! - **Authentication**: JWT and API key validation
//! - **Timeout**: Request timeout enforcement
//!
//! # Usage
//!
//! ```rust,ignore
//! use proximadb::network::middleware::{CorsConfig, RateLimitConfig, TimeoutConfig};
//!
//! // Production security configuration
//! let cors = CorsConfig::production()
//!     .allow_origin("https://app.example.com");
//! // Request rate limiting on the REST surface is mounted via the env gate
//! // PROXIMADB_REST_RATE_LIMIT_RPM (TD-RATE-1; unset = no layer). It is NOT
//! // configured by constructing a RateLimitConfig here.
//! let timeout = TimeoutConfig::default(); // 30 second timeout
//! ```

pub mod auth;
pub mod auth_failclosed;
pub mod claim_metrics;
pub use proximadb_network_middleware::backpressure;
pub use proximadb_network_middleware::cors;
pub mod metrics;
pub use proximadb_network_middleware::rate_limit;
pub mod request_id;
pub mod request_scope;
pub mod tenant;
pub use proximadb_network_middleware::timeout;
pub mod tls;

pub use auth::{AuthLayer, MiddlewareAuthConfig, UserInfo};
pub use backpressure::{BackpressureConfig, create_concurrency_limit_layer};
pub use cors::{CorsConfig, CorsConfigError, create_cors_layer};
pub use rate_limit::RateLimitConfig;
pub use request_id::{
    RequestId, RequestIdExt, RequestIdLayer, X_REQUEST_ID, request_id_middleware,
};
pub use tenant::{
    TenantContext, TenantContextExt, TenantExtractor, TenantExtractorConfig, TenantIdSource,
    X_TENANT_ID, create_tenant_extractor, tenant_middleware,
};
pub use timeout::{TimeoutConfig, create_timeout_layer};
pub use tls::{
    TlsAuthenticatedUser, TlsAuthenticatedUserExt, TlsCertErrorResponse, TlsClientCertConfig,
    TlsClientCertLayer, TlsClientCertState, matches_cn_pattern, tls_client_cert_middleware,
};
