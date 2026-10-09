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

//! Unified Authentication Middleware for ProximaDB REST API

use crate::network::auth::{
    AuthError, AuthResult, AuthService, Permission, PermissionContext, ResourceType,
};
use crate::network::tls::ClientCertificateInfo;
use crate::security::{AuthenticationData, SecurityCoordinator, UnifiedUserContext};
use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{Json, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Authentication middleware state
pub struct AuthMiddlewareState {
    /// Shared authentication service instance
    pub auth_service: Arc<AuthService>,
}

/// Authentication error response returned as JSON
#[derive(Debug, Serialize)]
pub struct AuthErrorResponse {
    /// Error type identifier
    pub error: String,
    /// Human-readable error description
    pub message: String,
    /// HTTP status code
    pub code: u16,
}

/// Narrow data-plane capability carried by an authenticated JWT.
///
/// This keeps ProximaDB generic: an external control plane can issue scoped
/// route tokens, while ProximaDB only validates transport, collection,
/// operation, byte, and record limits from additive JWT claims.
#[derive(Debug, Clone)]
pub struct DataPlaneCapability {
    pub capability_type: String,
    pub collection: Option<String>,
    pub operation: Option<String>,
    pub protocol: Option<String>,
    pub mode: Option<String>,
    pub scopes: Vec<String>,
    pub max_records: Option<u64>,
    pub max_bytes: Option<u64>,
}

impl DataPlaneCapability {
    pub fn from_user_context(user_context: &UnifiedUserContext) -> Option<Self> {
        let capability_type = user_context.metadata.get("capability_type")?.clone();
        Some(Self {
            capability_type,
            collection: user_context.metadata.get("collection").cloned(),
            operation: user_context.metadata.get("operation").cloned(),
            protocol: user_context.metadata.get("protocol").cloned(),
            mode: user_context.metadata.get("mode").cloned(),
            scopes: user_context
                .metadata
                .get("scopes")
                .map(|scopes| {
                    scopes
                        .split_whitespace()
                        .map(ToOwned::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            max_records: user_context
                .metadata
                .get("max_records")
                .and_then(|value| value.parse::<u64>().ok()),
            max_bytes: user_context
                .metadata
                .get("max_bytes")
                .and_then(|value| value.parse::<u64>().ok()),
        })
    }

    pub fn ensure_record_count(&self, count: usize) -> Result<(), String> {
        if let Some(max_records) = self.max_records
            && count as u64 > max_records
        {
            return Err(format!(
                "Request has {} records, exceeding capability limit {}",
                count, max_records
            ));
        }
        Ok(())
    }

    fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|candidate| candidate == scope)
    }
}

/// Unified auth middleware using SecurityCoordinator
pub async fn auth_middleware_unified(
    State(security_coordinator): State<Arc<SecurityCoordinator>>,
    mut request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<AuthErrorResponse>)> {
    let path = request.uri().path();

    if should_skip_auth(path) {
        return Ok(next.run(request).await);
    }

    let auth_header = extract_auth_header(&request)?;
    let auth_data = map_header_to_auth_data(&auth_header)?;

    let user_context = security_coordinator
        .authenticate_request(auth_data)
        .await
        .map_err(|e| {
            (
                StatusCode::UNAUTHORIZED,
                Json(AuthErrorResponse {
                    error: "authentication_failed".to_string(),
                    message: format!("{}", e),
                    code: 401,
                }),
            )
        })?;

    if let Some(capability) = DataPlaneCapability::from_user_context(&user_context) {
        validate_rest_data_plane_capability(&capability, &request)?;
        request.extensions_mut().insert(capability);
    }
    request.extensions_mut().insert(user_context);
    Ok(next.run(request).await)
}

/// Parallel port-based auth middleware (ADR-016 / Task #69 step 3).
///
/// Behaves like `auth_middleware_unified` but talks to a
/// `proximadb_runtime::SecurityPort` trait object instead of the
/// concrete `SecurityCoordinator`, and stores the port-level
/// `PortUserContext` in `request.extensions` instead of the rich
/// root-crate `UnifiedUserContext`.  Consumers that have migrated to
/// the port projection can mount this middleware; the legacy
/// `auth_middleware_unified` remains the production path until step 4
/// (per-consumer migrations) completes.
///
/// `DataPlaneCapability` handling is deliberately omitted here — the
/// capability lookup currently requires `UnifiedUserContext`-typed
/// info that's lost in the port projection.  Step 4 either extends
/// the port surface for capability extraction or leaves callers needing
/// capability on the legacy middleware.
///
/// # Wiring status (2026-05-26 audit, Tier 1.3 of pre-release plan)
///
/// **This middleware is NOT wired into any production route as of
/// 2026-05-26.** It exists as the parallel-shortcut foundation for
/// ADR-016 step 4 (per-consumer `UnifiedUserContext → PortUserContext`
/// migrations). Step 4 is intentionally demand-blocked: don't wire
/// this middleware in until a real cross-crate consumer (typically a
/// `proximadb-api` gRPC handler that needs port-typed auth context)
/// surfaces. See `docs/12-design/adr/ADR-016-port-user-context.adoc`
/// for the migration plan and
/// `docs/_internal/status/PRE_RELEASE_FOUNDATIONS_2026_05_26.adoc`
/// for the deferral rationale.
///
/// Existing routes use `auth_middleware_unified` (just above) which
/// stores the rich `UnifiedUserContext` directly. That path is the
/// production surface for v0.3.
pub async fn auth_middleware_unified_port(
    State(security_port): State<Arc<dyn proximadb_runtime::SecurityPort>>,
    mut request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<AuthErrorResponse>)> {
    let path = request.uri().path();

    if should_skip_auth(path) {
        return Ok(next.run(request).await);
    }

    let auth_header = extract_auth_header(&request)?;
    let credential = map_header_to_port_credential(&auth_header)?;

    let port_context = security_port.authenticate(credential).await.map_err(|e| {
        (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "authentication_failed".to_string(),
                message: format!("{}", e),
                code: 401,
            }),
        )
    })?;

    request.extensions_mut().insert(port_context);
    Ok(next.run(request).await)
}

/// Map an HTTP `Authorization` header value to a port-level credential.
/// Mirrors `map_header_to_auth_data` but emits `PortAuthCredential`
/// instead of the root-crate `AuthenticationData` enum.
fn map_header_to_port_credential(
    auth_header: &str,
) -> Result<proximadb_runtime::PortAuthCredential, (StatusCode, Json<AuthErrorResponse>)> {
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        return Ok(proximadb_runtime::PortAuthCredential::Jwt(
            token.to_string(),
        ));
    }
    if let Some(key) = auth_header.strip_prefix("API-Key ") {
        return Ok(proximadb_runtime::PortAuthCredential::ApiKey(
            key.to_string(),
        ));
    }
    if let Some(key) = auth_header.strip_prefix("Api-Key ") {
        return Ok(proximadb_runtime::PortAuthCredential::ApiKey(
            key.to_string(),
        ));
    }
    Ok(proximadb_runtime::PortAuthCredential::ApiKey(
        auth_header.to_string(),
    ))
}

/// Middleware to authenticate and authorize requests
pub async fn auth_middleware(
    State(auth_state): State<AuthMiddlewareState>,
    mut request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<AuthErrorResponse>)> {
    let uri = request.uri();
    let method = request.method();
    let path = uri.path();

    debug!("Processing auth for {} {}", method, path);

    // Skip authentication for health endpoints (configurable)
    if should_skip_auth(path) {
        debug!("Skipping auth for health endpoint: {}", path);
        return Ok(next.run(request).await);
    }

    // Extract authorization header
    let auth_header = extract_auth_header(&request)?;

    // Authenticate the request
    let auth_result = auth_state
        .auth_service
        .authenticate(&auth_header)
        .await
        .map_err(auth_error_to_response)?;

    // Check basic permissions for the endpoint
    let required_permission = determine_required_permission(method.as_str(), path);
    if let Some(permission) = required_permission {
        let context = build_permission_context(path, &auth_result);

        auth_state
            .auth_service
            .rbac
            .check_permission_with_context(&auth_result.user_id, permission, &context)
            .await
            .map_err(auth_error_to_response)?;
    }

    // Add auth result to request extensions for use by handlers
    request.extensions_mut().insert(auth_result);

    Ok(next.run(request).await)
}

/// Extract authorization header from request
fn extract_auth_header(request: &Request) -> Result<String, (StatusCode, Json<AuthErrorResponse>)> {
    let auth_header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                Json(AuthErrorResponse {
                    error: "missing_authorization".to_string(),
                    message: "Authorization header is required".to_string(),
                    code: 401,
                }),
            )
        })?;

    let auth_str = auth_header.to_str().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(AuthErrorResponse {
                error: "invalid_authorization_header".to_string(),
                message: "Authorization header contains invalid characters".to_string(),
                code: 400,
            }),
        )
    })?;

    Ok(auth_str.to_string())
}

fn map_header_to_auth_data(
    auth_header: &str,
) -> Result<AuthenticationData, (StatusCode, Json<AuthErrorResponse>)> {
    // TD-ABAC-6: the shared credential parser. This was a verbatim copy of gRPC's
    // `auth_data_from_headers` and Arrow's `auth_data_from_metadata` (same Bearer
    // / API-Key / raw logic); all three now share one parser.
    Ok(crate::security::request_identity::parse_authorization(
        auth_header,
    ))
}

fn validate_rest_data_plane_capability(
    capability: &DataPlaneCapability,
    request: &Request,
) -> Result<(), (StatusCode, Json<AuthErrorResponse>)> {
    if capability.protocol.as_deref() != Some("rest") {
        return Err(authz_response(
            StatusCode::FORBIDDEN,
            "capability_protocol_mismatch",
            "Capability token is not valid for REST",
        ));
    }

    let path = request.uri().path();
    let method = request.method().as_str();
    let operation = infer_data_plane_operation(method, path).ok_or_else(|| {
        authz_response(
            StatusCode::FORBIDDEN,
            "capability_endpoint_denied",
            "Capability token is not valid for this endpoint",
        )
    })?;

    if capability.operation.as_deref() != Some(operation) {
        return Err(authz_response(
            StatusCode::FORBIDDEN,
            "capability_operation_mismatch",
            "Capability token operation does not match the REST endpoint",
        ));
    }

    if let Some(collection) = extract_collection_from_path(path)
        && capability.collection.as_deref() != Some(collection)
    {
        return Err(authz_response(
            StatusCode::FORBIDDEN,
            "capability_collection_mismatch",
            "Capability token collection does not match the REST endpoint",
        ));
    }

    let scope_ok = match operation {
        "ingest" => capability.has_scope("records:write"),
        "search" => capability.has_scope("search:execute") || capability.has_scope("records:read"),
        _ => false,
    };
    if !scope_ok {
        return Err(authz_response(
            StatusCode::FORBIDDEN,
            "capability_scope_missing",
            "Capability token lacks the required scope",
        ));
    }

    if let Some(max_bytes) = capability.max_bytes
        && let Some(content_length) = request
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        && content_length > max_bytes
    {
        return Err(authz_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "capability_byte_limit_exceeded",
            "Request body exceeds the capability byte limit",
        ));
    }

    Ok(())
}

fn infer_data_plane_operation(method: &str, path: &str) -> Option<&'static str> {
    if method == "POST" && (path.ends_with("/documents") || path.ends_with("/records/batch")) {
        return Some("ingest");
    }
    if method == "POST" && (path.ends_with("/search") || path.ends_with("/query")) {
        return Some("search");
    }
    None
}

fn extract_collection_from_path(path: &str) -> Option<&str> {
    let mut segments = path.split('/').filter(|segment| !segment.is_empty());
    while let Some(segment) = segments.next() {
        if segment == "collections" {
            return segments.next().filter(|value| !value.is_empty());
        }
    }
    None
}

fn authz_response(
    status: StatusCode,
    error: &str,
    message: &str,
) -> (StatusCode, Json<AuthErrorResponse>) {
    (
        status,
        Json(AuthErrorResponse {
            error: error.to_string(),
            message: message.to_string(),
            code: status.as_u16(),
        }),
    )
}

/// Determine if authentication should be skipped for this path
fn should_skip_auth(path: &str) -> bool {
    // Health endpoints
    if path.starts_with("/health") {
        return true;
    }

    // Auth endpoints (login, etc.)
    if path.starts_with("/auth/") {
        return true;
    }

    // Public documentation endpoints
    if path.starts_with("/docs") || path.starts_with("/openapi") {
        return true;
    }

    false
}

/// Determine required permission based on HTTP method and path
fn determine_required_permission(method: &str, path: &str) -> Option<Permission> {
    match path {
        // Vector endpoints (check before general collection endpoints)
        path if path.contains("/vectors") => match method {
            "GET" => Some(Permission::ReadVectors),
            "POST" => {
                if path.contains("/search") {
                    Some(Permission::SearchVectors)
                } else {
                    Some(Permission::InsertVectors)
                }
            }
            "PUT" | "PATCH" => Some(Permission::UpdateVectors),
            "DELETE" => Some(Permission::DeleteVectors),
            _ => None,
        },

        // Graph endpoints
        path if path.contains("/graph") => match method {
            "GET" => Some(Permission::ReadGraphRelations),
            "POST" => Some(Permission::CreateGraphRelations),
            "DELETE" => Some(Permission::DeleteGraphRelations),
            _ => None,
        },

        // SQL query endpoints
        path if path.contains("/sql") || path.contains("/query") => {
            Some(Permission::ExecuteSqlQueries)
        }

        // System endpoints
        path if path.starts_with("/system") => match method {
            "GET" => {
                if path.contains("/metrics") {
                    Some(Permission::ViewSystemMetrics)
                } else {
                    Some(Permission::ViewSystemHealth)
                }
            }
            "POST" | "PUT" | "PATCH" => Some(Permission::ConfigureSystem),
            _ => None,
        },

        // Admin endpoints
        path if path.starts_with("/admin") => {
            if path.contains("/users") {
                Some(Permission::ManageUsers)
            } else if path.contains("/roles") {
                Some(Permission::ManageRoles)
            } else if path.contains("/api-keys") {
                Some(Permission::ManageApiKeys)
            } else if path.contains("/audit") {
                Some(Permission::ViewAuditLogs)
            } else {
                Some(Permission::ConfigureSystem)
            }
        }

        // Collection endpoints (checked last as they're more general)
        path if path.starts_with("/collections") => match method {
            "GET" => {
                if path.ends_with("/collections") {
                    Some(Permission::ListCollections)
                } else {
                    Some(Permission::ReadCollectionMetadata)
                }
            }
            "POST" => Some(Permission::CreateCollection),
            "PUT" | "PATCH" => Some(Permission::UpdateCollectionMetadata),
            "DELETE" => Some(Permission::DeleteCollection),
            _ => None,
        },

        _ => None, // No specific permission required
    }
}

/// Build permission context from request path and auth result
fn build_permission_context(path: &str, auth_result: &AuthResult) -> PermissionContext {
    let resource_type = if path.starts_with("/collections") {
        ResourceType::Collection
    } else if path.contains("/vectors") {
        ResourceType::Vector
    } else if path.contains("/graph") {
        ResourceType::Graph
    } else if path.starts_with("/system") || path.starts_with("/admin") {
        ResourceType::System
    } else if path.contains("/users") {
        ResourceType::User
    } else if path.contains("/roles") {
        ResourceType::Role
    } else {
        ResourceType::System
    };

    // Extract resource ID from path if possible
    let resource_id = extract_resource_id(path);

    PermissionContext {
        resource_type,
        resource_id,
        tenant_id: auth_result.tenant_id.clone(),
        metadata: std::collections::HashMap::new(),
    }
}

/// Extract resource ID from URL path
fn extract_resource_id(path: &str) -> Option<String> {
    let path_segments: Vec<&str> = path.split('/').collect();

    // Look for patterns like /collections/{id}, /vectors/{id}, etc.
    for (i, segment) in path_segments.iter().enumerate() {
        if matches!(*segment, "collections" | "vectors" | "users" | "roles")
            && let Some(id) = path_segments.get(i + 1)
            && !id.is_empty()
            && *id != "search"
            && *id != "bulk"
        {
            return Some(id.to_string());
        }
    }

    None
}

/// Convert auth error to HTTP response
fn auth_error_to_response(error: AuthError) -> (StatusCode, Json<AuthErrorResponse>) {
    match error {
        AuthError::AuthenticationFailed(msg) => (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "authentication_failed".to_string(),
                message: msg,
                code: 401,
            }),
        ),
        AuthError::AuthorizationDenied(permission) => (
            StatusCode::FORBIDDEN,
            Json(AuthErrorResponse {
                error: "authorization_denied".to_string(),
                message: format!("Permission required: {:?}", permission),
                code: 403,
            }),
        ),
        AuthError::InvalidToken(msg) => (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "invalid_token".to_string(),
                message: msg,
                code: 401,
            }),
        ),
        AuthError::TokenExpired => (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "token_expired".to_string(),
                message: "Token has expired".to_string(),
                code: 401,
            }),
        ),
        AuthError::InvalidCredentials => (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "invalid_credentials".to_string(),
                message: "Invalid credentials provided".to_string(),
                code: 401,
            }),
        ),
        AuthError::UserNotFound(user_id) => (
            StatusCode::NOT_FOUND,
            Json(AuthErrorResponse {
                error: "user_not_found".to_string(),
                message: format!("User not found: {}", user_id),
                code: 404,
            }),
        ),
        AuthError::RoleNotFound(role) => (
            StatusCode::NOT_FOUND,
            Json(AuthErrorResponse {
                error: "role_not_found".to_string(),
                message: format!("Role not found: {}", role),
                code: 404,
            }),
        ),
    }
}

/// Extension trait to extract auth result from request
pub trait RequestAuthExt {
    /// Extract the authentication result from request extensions
    fn auth_result(&self) -> Option<&AuthResult>;
    /// Extract the authenticated user ID
    fn user_id(&self) -> Option<&str>;
    /// Extract the tenant ID for multi-tenant requests
    fn tenant_id(&self) -> Option<&str>;
}

impl RequestAuthExt for Request {
    fn auth_result(&self) -> Option<&AuthResult> {
        self.extensions().get::<AuthResult>()
    }

    fn user_id(&self) -> Option<&str> {
        self.auth_result().map(|auth| auth.user_id.as_str())
    }

    fn tenant_id(&self) -> Option<&str> {
        self.auth_result()
            .and_then(|auth| auth.tenant_id.as_deref())
    }
}

/// Optional authorization check for specific operations within handlers
pub async fn check_permission(
    auth_service: &AuthService,
    auth_result: &AuthResult,
    permission: Permission,
    resource_type: ResourceType,
    resource_id: Option<String>,
) -> Result<(), AuthError> {
    let context = PermissionContext {
        resource_type,
        resource_id,
        tenant_id: auth_result.tenant_id.clone(),
        metadata: std::collections::HashMap::new(),
    };

    auth_service
        .rbac
        .check_permission_with_context(&auth_result.user_id, permission, &context)
        .await
}

// ============================================================================
// mTLS (Mutual TLS) Authentication Middleware
// ============================================================================

/// mTLS configuration for certificate-based authentication
#[derive(Debug, Clone, Deserialize)]
pub struct MtlsConfig {
    /// Require client certificates
    pub enabled: bool,
    /// Allowed CN patterns (supports wildcards like "*.example.com")
    pub allowed_cn_patterns: Vec<String>,
    /// Map certificate CNs to user IDs
    pub cn_to_user_mapping: std::collections::HashMap<String, String>,
    /// Default roles for mTLS authenticated users
    pub default_roles: Vec<String>,
}

impl Default for MtlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_cn_patterns: vec![],
            cn_to_user_mapping: std::collections::HashMap::new(),
            default_roles: vec!["reader".to_string()],
        }
    }
}

/// mTLS authentication state shared with the middleware layer
pub struct MtlsAuthState {
    /// mTLS configuration
    pub config: MtlsConfig,
    /// Shared authentication service for permission resolution
    pub auth_service: Arc<AuthService>,
}

/// Authenticated user from mTLS
#[derive(Debug, Clone)]
pub struct MtlsAuthenticatedUser {
    /// User ID derived from certificate CN
    pub user_id: String,
    /// Certificate common name
    pub common_name: String,
    /// Organization from certificate
    pub organization: Option<String>,
    /// Certificate fingerprint (SHA-256)
    pub certificate_fingerprint: String,
    /// Roles assigned to this user
    pub roles: Vec<String>,
    /// Authentication method
    pub auth_method: String,
}

/// Middleware for mTLS (client certificate) authentication
///
/// This middleware extracts client certificate information from TLS connections
/// and authenticates users based on their certificate's Common Name (CN).
///
/// ## Usage
///
/// ```rust,ignore
/// use axum::Router;
/// use axum::middleware;
///
/// let mtls_config = MtlsConfig {
///     enabled: true,
///     allowed_cn_patterns: vec!["*.mycompany.com".to_string()],
///     ..Default::default()
/// };
///
/// let app = Router::new()
///     .route("/api/v1/secure", get(handler))
///     .layer(middleware::from_fn_with_state(mtls_state, mtls_auth_middleware));
/// ```
pub async fn mtls_auth_middleware(
    State(mtls_state): State<Arc<MtlsAuthState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<AuthErrorResponse>)> {
    // Skip if mTLS is not enabled
    if !mtls_state.config.enabled {
        return Ok(next.run(request).await);
    }

    let path = request.uri().path();

    // Skip auth for health endpoints
    if should_skip_auth(path) {
        return Ok(next.run(request).await);
    }

    // Extract client certificate from request extensions
    // The certificate is set by the TLS layer when using mTLS
    let client_cert_info = request.extensions().get::<ClientCertificateInfo>().cloned();

    let cert_info = client_cert_info.ok_or_else(|| {
        warn!("mTLS authentication failed: no client certificate provided");
        (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "client_certificate_required".to_string(),
                message: "Client certificate is required for mTLS authentication".to_string(),
                code: 401,
            }),
        )
    })?;

    // Check if certificate is valid
    if !cert_info.is_valid {
        warn!("mTLS authentication failed: certificate expired or not yet valid");
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "certificate_invalid".to_string(),
                message: "Client certificate is expired or not yet valid".to_string(),
                code: 401,
            }),
        ));
    }

    // Extract CN from certificate
    let common_name = cert_info.common_name.clone().ok_or_else(|| {
        warn!("mTLS authentication failed: certificate has no CN");
        (
            StatusCode::UNAUTHORIZED,
            Json(AuthErrorResponse {
                error: "certificate_missing_cn".to_string(),
                message: "Client certificate must have a Common Name (CN)".to_string(),
                code: 401,
            }),
        )
    })?;

    // Validate CN against allowed patterns
    if !mtls_state.config.allowed_cn_patterns.is_empty() {
        let cn_allowed = mtls_state
            .config
            .allowed_cn_patterns
            .iter()
            .any(|pattern| matches_cn_pattern(&common_name, pattern));

        if !cn_allowed {
            warn!(
                "mTLS authentication failed: CN '{}' not in allowed patterns",
                common_name
            );
            return Err((
                StatusCode::FORBIDDEN,
                Json(AuthErrorResponse {
                    error: "certificate_cn_not_allowed".to_string(),
                    message: format!(
                        "Certificate CN '{}' is not in the allowed list",
                        common_name
                    ),
                    code: 403,
                }),
            ));
        }
    }

    // Map CN to user ID
    let user_id = mtls_state
        .config
        .cn_to_user_mapping
        .get(&common_name)
        .cloned()
        .unwrap_or_else(|| {
            // Default: use CN as user ID
            common_name.clone()
        });

    // Create authenticated user
    let authenticated_user = MtlsAuthenticatedUser {
        user_id: user_id.clone(),
        common_name: common_name.clone(),
        organization: cert_info.organization.clone(),
        certificate_fingerprint: cert_info.fingerprint.clone(),
        roles: mtls_state.config.default_roles.clone(),
        auth_method: "mtls".to_string(),
    };

    info!(
        "mTLS authentication successful: user_id={}, cn={}, fingerprint={}...",
        user_id,
        common_name,
        &cert_info.fingerprint[..16.min(cert_info.fingerprint.len())]
    );

    // Add authenticated user to request extensions
    request.extensions_mut().insert(authenticated_user);

    Ok(next.run(request).await)
}

/// Check if a Common Name matches a pattern (supports wildcards)
///
/// Patterns:
/// - Exact match: "client.example.com" matches "client.example.com"
/// - Wildcard: "*.example.com" matches "client.example.com"
/// - Star: "*" matches any CN
pub fn matches_cn_pattern(cn: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(suffix) = pattern.strip_prefix("*.") {
        // Wildcard pattern - CN must end with the suffix and have at least one character before
        cn.ends_with(suffix) && cn.len() > suffix.len() + 1
    } else {
        // Exact match
        cn == pattern
    }
}

/// Combined authentication middleware that tries mTLS first, then falls back to token auth
///
/// This middleware provides flexibility for services that want to accept both
/// mTLS and token-based authentication.
pub async fn hybrid_auth_middleware(
    State((mtls_state, security_coordinator)): State<(
        Option<Arc<MtlsAuthState>>,
        Arc<SecurityCoordinator>,
    )>,
    mut request: Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<AuthErrorResponse>)> {
    let path = request.uri().path();

    // Skip auth for health endpoints
    if should_skip_auth(path) {
        return Ok(next.run(request).await);
    }

    // First, try mTLS authentication if configured
    if let Some(ref mtls_state) = mtls_state
        && mtls_state.config.enabled
        && let Some(cert_info) = request.extensions().get::<ClientCertificateInfo>().cloned()
        && cert_info.is_valid
        && let Some(cn) = &cert_info.common_name
    {
        // CN is valid, check against allowed patterns
        let cn_allowed = mtls_state.config.allowed_cn_patterns.is_empty()
            || mtls_state
                .config
                .allowed_cn_patterns
                .iter()
                .any(|p| matches_cn_pattern(cn, p));

        if cn_allowed {
            let user_id = mtls_state
                .config
                .cn_to_user_mapping
                .get(cn)
                .cloned()
                .unwrap_or_else(|| cn.clone());

            let authenticated_user = MtlsAuthenticatedUser {
                user_id,
                common_name: cn.clone(),
                organization: cert_info.organization.clone(),
                certificate_fingerprint: cert_info.fingerprint.clone(),
                roles: mtls_state.config.default_roles.clone(),
                auth_method: "mtls".to_string(),
            };

            request.extensions_mut().insert(authenticated_user);
            return Ok(next.run(request).await);
        }
    }

    // Fall back to token-based authentication
    let auth_header = extract_auth_header(&request)?;
    let auth_data = map_header_to_auth_data(&auth_header)?;

    let user_context = security_coordinator
        .authenticate_request(auth_data)
        .await
        .map_err(|e| {
            (
                StatusCode::UNAUTHORIZED,
                Json(AuthErrorResponse {
                    error: "authentication_failed".to_string(),
                    message: format!("{}", e),
                    code: 401,
                }),
            )
        })?;

    request.extensions_mut().insert(user_context);
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::body::to_bytes;
    use axum::extract::Extension;
    use axum::extract::Request;
    use axum::middleware;
    use axum::routing::get;
    use std::collections::HashMap;
    use tower::ServiceExt;

    use crate::network::auth::config::JwtAlgorithm;
    use crate::network::auth::jwt::JwtService;
    use crate::security::auth_service::{
        ApiKeyInfo, AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig,
    };
    use crate::security::rbac_service::RBACConfig;
    use crate::security::security_coordinator::{ComplianceConfig, TlsConfig};
    use crate::security::{AuditConfig, SecurityConfig, SecurityCoordinator, SecurityMode};

    #[test]
    fn test_should_skip_auth() {
        assert!(should_skip_auth("/health"));
        assert!(should_skip_auth("/health/ready"));
        assert!(should_skip_auth("/auth/login"));
        assert!(should_skip_auth("/docs"));
        assert!(!should_skip_auth("/collections"));
        assert!(!should_skip_auth("/api/collections"));
    }

    #[test]
    fn test_determine_required_permission() {
        // Collection endpoints
        assert_eq!(
            determine_required_permission("GET", "/collections"),
            Some(Permission::ListCollections)
        );
        assert_eq!(
            determine_required_permission("POST", "/collections"),
            Some(Permission::CreateCollection)
        );
        assert_eq!(
            determine_required_permission("DELETE", "/collections/test"),
            Some(Permission::DeleteCollection)
        );

        // Vector endpoints
        assert_eq!(
            determine_required_permission("POST", "/collections/test/vectors/search"),
            Some(Permission::SearchVectors)
        );
        assert_eq!(
            determine_required_permission("POST", "/collections/test/vectors"),
            Some(Permission::InsertVectors)
        );

        // System endpoints
        assert_eq!(
            determine_required_permission("GET", "/system/metrics"),
            Some(Permission::ViewSystemMetrics)
        );
        assert_eq!(
            determine_required_permission("POST", "/system/config"),
            Some(Permission::ConfigureSystem)
        );
    }

    #[test]
    fn test_extract_resource_id() {
        assert_eq!(
            extract_resource_id("/collections/test-collection"),
            Some("test-collection".to_string())
        );
        assert_eq!(
            extract_resource_id("/collections/test-collection/vectors"),
            Some("test-collection".to_string())
        );
        assert_eq!(extract_resource_id("/collections"), None);
        assert_eq!(
            extract_resource_id("/collections/test/vectors/search"),
            Some("test".to_string())
        );
    }

    fn build_security_config_with_api_key() -> SecurityConfig {
        let mut api_keys = HashMap::new();
        api_keys.insert(
            "test-key".to_string(),
            ApiKeyInfo {
                user_id: "user1".to_string(),
                tenant_id: None,
                permissions: vec!["search".to_string()],
                roles: vec![],
                created_at: None,
                expires_at: None,
                rate_limit_per_minute: None,
                ip_restrictions: vec![],
            },
        );

        SecurityConfig {
            enabled: true,
            mode: SecurityMode::Development,
            authentication: AuthenticationConfig {
                enabled: true,
                methods: vec![AuthenticationMethod::ApiKey],
                require_authentication: true,
                default_session_timeout_minutes: 60,
                api_keys,
                scram_users: HashMap::new(),
                jwt: JwtConfig {
                    enabled: false,
                    secret: "dev-secret".to_string(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: "test".to_string(),
                    audience: "test".to_string(),
                    algorithm: "HS256".to_string(),
                },
                sso: SSOConfig {
                    enabled: false,
                    providers: vec![],
                    token_cache_ttl_minutes: 5,
                },
                mtls: MtlsConfig::default(),
                oidc: None,
                audit_fail_closed: false,
            },
            rbac: RBACConfig::default(),
            audit: AuditConfig::default(),
            tls: TlsConfig {
                enabled: false,
                require_client_certificates: false,
                cert_file: None,
                key_file: None,
                ca_file: None,
            },
            compliance: ComplianceConfig {
                frameworks: vec![],
                data_residency: None,
                encryption_at_rest: false,
                encryption_in_transit: false,
            },
            encryption: crate::security::EncryptionConfig::default(),
            key_store: crate::security::KeyStoreConfig::default(),
            tenant: Default::default(),
            pgwire: crate::security::security_coordinator::PgwireSecurityConfig::default(),
        }
    }

    fn build_security_config_with_jwt() -> SecurityConfig {
        SecurityConfig {
            enabled: true,
            mode: SecurityMode::Development,
            authentication: AuthenticationConfig {
                enabled: true,
                methods: vec![AuthenticationMethod::JWT],
                require_authentication: true,
                default_session_timeout_minutes: 60,
                api_keys: HashMap::new(),
                scram_users: std::collections::HashMap::new(),
                jwt: JwtConfig {
                    enabled: true,
                    secret: "dev-jwt-secret".to_string(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: "proximadb-dev".to_string(),
                    audience: "proximadb-clients".to_string(),
                    algorithm: "HS256".to_string(),
                },
                sso: SSOConfig {
                    enabled: false,
                    providers: vec![],
                    token_cache_ttl_minutes: 5,
                },
                mtls: MtlsConfig::default(),
                oidc: None,
                audit_fail_closed: false,
            },
            rbac: RBACConfig::default(),
            audit: AuditConfig::default(),
            tls: TlsConfig {
                enabled: false,
                require_client_certificates: false,
                cert_file: None,
                key_file: None,
                ca_file: None,
            },
            compliance: ComplianceConfig {
                frameworks: vec![],
                data_residency: None,
                encryption_at_rest: false,
                encryption_in_transit: false,
            },
            encryption: crate::security::EncryptionConfig::default(),
            key_store: crate::security::KeyStoreConfig::default(),
            tenant: Default::default(),
            pgwire: crate::security::security_coordinator::PgwireSecurityConfig::default(),
        }
    }

    #[tokio::test]
    async fn auth_middleware_unified_allows_valid_api_key() {
        let coordinator = Arc::new(
            SecurityCoordinator::from_config(build_security_config_with_api_key())
                .await
                .expect("Failed to create SecurityCoordinator from config"),
        );

        let request = Request::builder()
            .uri("/api/v1/search")
            .header("Authorization", "Api-Key test-key")
            .body(Body::empty())
            .expect("Failed to build test request");

        let app = Router::new()
            .route(
                "/api/v1/search",
                get(
                    |Extension(ctx): Extension<crate::security::UnifiedUserContext>| async move {
                        if ctx.user_id == "user1" {
                            "ok"
                        } else {
                            "forbidden"
                        }
                    },
                ),
            )
            .layer(middleware::from_fn_with_state(
                coordinator.clone(),
                auth_middleware_unified,
            ));

        let response = app
            .oneshot(request)
            .await
            .expect("Failed to send test request");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn auth_middleware_unified_rejects_missing_header() {
        let coordinator = Arc::new(
            SecurityCoordinator::from_config(build_security_config_with_api_key())
                .await
                .expect("Failed to create SecurityCoordinator from config"),
        );

        let request = Request::builder()
            .uri("/api/v1/search")
            .body(Body::empty())
            .expect("Failed to build test request");

        let app = Router::new()
            .route("/api/v1/search", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                coordinator.clone(),
                auth_middleware_unified,
            ));

        let response = app
            .oneshot(request)
            .await
            .expect("Failed to send test request");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        assert!(String::from_utf8_lossy(&body).contains("Authorization header is required"));
    }

    #[tokio::test]
    async fn auth_middleware_unified_accepts_jwt() {
        // Build a JWT that matches the security configuration
        let cfg = build_security_config_with_jwt();
        let jwt_service = JwtService::new(crate::network::auth::config::JwtConfig {
            secret: Some(cfg.authentication.jwt.secret.clone()),
            expiration_secs: cfg.authentication.jwt.access_token_expiration_minutes * 60,
            refresh_expiration_secs: cfg.authentication.jwt.refresh_token_expiration_days
                * 24
                * 3600,
            issuer: cfg.authentication.jwt.issuer.clone(),
            audience: cfg.authentication.jwt.audience.clone(),
            algorithm: JwtAlgorithm::HS256,
        })
        .expect("Failed to create JWT service");

        let token_pair = jwt_service
            .generate_token_pair("jwt-user", None, vec!["reader".to_string()])
            .await
            .expect("Failed to generate JWT token pair");

        let coordinator = Arc::new(
            SecurityCoordinator::from_config(cfg)
                .await
                .expect("Failed to create SecurityCoordinator from config"),
        );

        let request = Request::builder()
            .uri("/api/v1/search")
            .header(
                "Authorization",
                format!("Bearer {}", token_pair.access_token),
            )
            .body(Body::empty())
            .expect("Failed to build test request");

        let app = Router::new()
            .route(
                "/api/v1/search",
                get(
                    |Extension(ctx): Extension<crate::security::UnifiedUserContext>| async move {
                        format!("hello {}", ctx.user_id)
                    },
                ),
            )
            .layer(middleware::from_fn_with_state(
                coordinator.clone(),
                auth_middleware_unified,
            ));

        let response = app
            .oneshot(request)
            .await
            .expect("Failed to send test request");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        assert!(String::from_utf8_lossy(&body).contains("jwt-user"));
    }

    #[tokio::test]
    async fn auth_middleware_unified_rejects_invalid_jwt() {
        let cfg = build_security_config_with_jwt();
        let coordinator = Arc::new(
            SecurityCoordinator::from_config(cfg)
                .await
                .expect("Failed to create SecurityCoordinator from config"),
        );

        let request = Request::builder()
            .uri("/api/v1/search")
            .header("Authorization", "Bearer invalid-token")
            .body(Body::empty())
            .expect("Failed to build test request");

        let app = Router::new()
            .route("/api/v1/search", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                coordinator.clone(),
                auth_middleware_unified,
            ));

        let response = app
            .oneshot(request)
            .await
            .expect("Failed to send test request");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_middleware_unified_rejects_expired_jwt() {
        let cfg = build_security_config_with_jwt();
        let secret = cfg.authentication.jwt.secret.clone();

        // Build an already-expired token that matches audience/issuer
        let now = chrono::Utc::now().timestamp();
        let claims = crate::network::auth::jwt::Claims {
            sub: "jwt-user".to_string(),
            iat: now - 300,
            exp: now - 60,
            nbf: now - 300,
            iss: cfg.authentication.jwt.issuer.clone(),
            aud: cfg.authentication.jwt.audience.clone(),
            jti: "expired-token".to_string(),
            tenant_id: None,
            roles: vec!["reader".to_string()],
            typ: crate::network::auth::jwt::TokenType::Access,
            // New control-plane capability fields — defaulted in this
            // test because expiry is the only thing under test.
            capability_type: None,
            collection: None,
            operation: None,
            protocol: None,
            mode: None,
            scopes: Vec::new(),
            max_records: None,
            max_bytes: None,
            tier: None,
            route_visibility: None,
            metering_required: None,
        };
        let expired_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("Failed to encode expired JWT token");

        let coordinator = Arc::new(
            SecurityCoordinator::from_config(cfg)
                .await
                .expect("Failed to create SecurityCoordinator from config"),
        );

        let request = Request::builder()
            .uri("/api/v1/search")
            .header("Authorization", format!("Bearer {}", expired_token))
            .body(Body::empty())
            .expect("Failed to build test request");

        let app = Router::new()
            .route("/api/v1/search", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                coordinator.clone(),
                auth_middleware_unified,
            ));

        let response = app
            .oneshot(request)
            .await
            .expect("Failed to send test request");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_middleware_unified_rejects_jwt_with_wrong_issuer() {
        // Coordinator expects issuer proximadb-dev; we mint a token with a mismatched issuer
        let cfg = build_security_config_with_jwt();
        let bad_jwt_service = JwtService::new(crate::network::auth::config::JwtConfig {
            secret: Some(cfg.authentication.jwt.secret.clone()),
            expiration_secs: cfg.authentication.jwt.access_token_expiration_minutes * 60,
            refresh_expiration_secs: cfg.authentication.jwt.refresh_token_expiration_days
                * 24
                * 3600,
            issuer: "unexpected-issuer".to_string(),
            audience: cfg.authentication.jwt.audience.clone(),
            algorithm: JwtAlgorithm::HS256,
        })
        .expect("Failed to create JWT service with wrong issuer");
        let bad_token = bad_jwt_service
            .generate_token_pair("jwt-user", None, vec!["reader".to_string()])
            .await
            .expect("Failed to generate JWT token with wrong issuer")
            .access_token;

        let coordinator = Arc::new(
            SecurityCoordinator::from_config(cfg)
                .await
                .expect("Failed to create SecurityCoordinator from config"),
        );

        let request = Request::builder()
            .uri("/api/v1/search")
            .header("Authorization", format!("Bearer {}", bad_token))
            .body(Body::empty())
            .expect("Failed to build test request");

        let app = Router::new()
            .route("/api/v1/search", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                coordinator.clone(),
                auth_middleware_unified,
            ));

        let response = app
            .oneshot(request)
            .await
            .expect("Failed to send test request");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// ADR-016 / Task #69 step 3: lock in the port-credential header
    /// parsing rules so future step-4 migrations can rely on the
    /// contract.  Mirrors the existing `map_header_to_auth_data`
    /// semantics from the legacy middleware (Bearer, API-Key, Api-Key,
    /// raw-as-ApiKey fallback).
    mod port_credential_parsing {
        use super::super::map_header_to_port_credential;
        use proximadb_runtime::PortAuthCredential;

        fn assert_jwt(header: &str, expected: &str) {
            match map_header_to_port_credential(header).expect("ok") {
                PortAuthCredential::Jwt(t) => assert_eq!(t, expected),
                PortAuthCredential::ApiKey(k) => panic!("expected Jwt, got ApiKey({})", k),
            }
        }

        fn assert_api_key(header: &str, expected: &str) {
            match map_header_to_port_credential(header).expect("ok") {
                PortAuthCredential::ApiKey(k) => assert_eq!(k, expected),
                PortAuthCredential::Jwt(t) => panic!("expected ApiKey, got Jwt({})", t),
            }
        }

        #[test]
        fn bearer_prefix_emits_jwt() {
            assert_jwt("Bearer abc.def.ghi", "abc.def.ghi");
        }

        #[test]
        fn api_key_caps_prefix_emits_api_key() {
            assert_api_key("API-Key secret123", "secret123");
        }

        #[test]
        fn api_key_titlecase_prefix_emits_api_key() {
            // Same semantics as the legacy `map_header_to_auth_data`.
            assert_api_key("Api-Key secret123", "secret123");
        }

        #[test]
        fn raw_value_falls_back_to_api_key() {
            // Mirrors the legacy "treat raw value as API key" rule.
            assert_api_key("rawkey-no-prefix", "rawkey-no-prefix");
        }

        #[test]
        fn empty_string_is_api_key_with_empty_value() {
            // The legacy parser also accepts "" as ApiKey("") — keeping
            // the contract identical so the parallel port middleware
            // doesn't diverge from the legacy one on edge cases.
            assert_api_key("", "");
        }
    }
}
