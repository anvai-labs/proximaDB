//! Shared gRPC authentication and data-plane capability enforcement.
//!
//! The transport layer authenticates every generic gRPC request when enterprise
//! security is enabled. Narrow operator-issued capability tokens are then
//! constrained to the V2 record data-plane RPCs; body-aware checks for
//! collection, record, and byte limits are applied inside `ProximaRecordService`
//! after protobuf decoding.

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tonic::codegen::http::{self, HeaderMap, Request as HttpRequest, Response as HttpResponse};
use tonic::{Code, Request, Status};
use tower::{Layer, Service};

use proximadb_tenant::identity_trust::{
    AuthenticatedTenantBinding, HeaderTrustPolicy, ResolvedTenantAssertion, TenantAssertionError,
    resolve_tenant_assertion,
};

use crate::network::auth::middleware::DataPlaneCapability;
use crate::security::{AuthenticationData, SecurityCoordinator, UnifiedUserContext};

/// Adds the deployment-wide tenant resolution mode to every gRPC request.
/// Authentication is an independent optional layer, so this layer is always
/// installed even in development deployments with security disabled.
#[derive(Clone)]
pub struct GrpcTenantModeLayer {
    mode: proximadb_tenant::TenantDeploymentMode,
}

impl GrpcTenantModeLayer {
    pub fn new(mode: proximadb_tenant::TenantDeploymentMode) -> Self {
        Self { mode }
    }
}

#[derive(Clone)]
pub struct GrpcTenantModeService<S> {
    inner: S,
    mode: proximadb_tenant::TenantDeploymentMode,
}

impl<S> Layer<S> for GrpcTenantModeLayer {
    type Service = GrpcTenantModeService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GrpcTenantModeService {
            inner,
            mode: self.mode.clone(),
        }
    }
}

impl<S, B> Service<HttpRequest<B>> for GrpcTenantModeService<S>
where
    S: Service<HttpRequest<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: HttpRequest<B>) -> Self::Future {
        request.extensions_mut().insert(self.mode.clone());
        self.inner.call(request)
    }
}

#[derive(Debug, Clone)]
pub struct GrpcAuthContext {
    pub user_context: UnifiedUserContext,
    pub capability: Option<DataPlaneCapability>,
    /// The request tenant as resolved by the ONE shared trust primitive
    /// (`proximadb_tenant::resolve_tenant_assertion`, TD-TENANT-1): the
    /// credential binding, an accepted `x-tenant-id` assertion, or a
    /// permitted gateway delegation. `None` = no tenant asserted or bound.
    pub resolved_tenant: Option<String>,
    /// TD-TENANT-1 item 3: the tenant's ADR-0083 stable u64 id (the account u32
    /// widened) — the ABAC binding-filter key. Stamped from `resolved_tenant`
    /// via the wired `TenantStableIdResolver`; `None` when no resolver is wired,
    /// no tenant resolved, or the tenant is unminted (fail-closed deny).
    pub tenant_stable_id: Option<u64>,
}

#[derive(Clone)]
pub struct GrpcAuthLayer {
    security_coordinator: Arc<SecurityCoordinator>,
    header_trust: HeaderTrustPolicy,
    /// ADR-0053 W8: trust policy for the `x-tenant-tier` entitlement claim.
    tier_header_trust: HeaderTrustPolicy,
    /// TD-TENANT-1 item 3: resolver that stamps `tenant_stable_id` on each
    /// request's `GrpcAuthContext` for ABAC. `None` when no catalog resolver is
    /// wired (gRPC ABAC inert — the same default REST had before PR1).
    stable_id_resolver: Option<Arc<dyn proximadb_tenant::TenantStableIdResolver>>,
}

impl GrpcAuthLayer {
    pub fn new(security_coordinator: Arc<SecurityCoordinator>) -> Self {
        Self {
            security_coordinator,
            header_trust: HeaderTrustPolicy::default(),
            tier_header_trust: HeaderTrustPolicy::default(),
            stable_id_resolver: None,
        }
    }

    /// Set the deployment's bare `x-tenant-id` trust policy (TD-TENANT-1).
    pub fn with_header_trust(mut self, header_trust: HeaderTrustPolicy) -> Self {
        self.header_trust = header_trust;
        self
    }

    /// Set the deployment's `x-tenant-tier` claim trust policy (ADR-0053 W8).
    pub fn with_tier_header_trust(mut self, tier_header_trust: HeaderTrustPolicy) -> Self {
        self.tier_header_trust = tier_header_trust;
        self
    }

    /// TD-TENANT-1 item 3: wire the tenant-stable-id resolver so each request's
    /// `GrpcAuthContext` carries `tenant_stable_id` for ABAC enforcement.
    pub fn with_stable_id_resolver(
        mut self,
        resolver: Arc<dyn proximadb_tenant::TenantStableIdResolver>,
    ) -> Self {
        self.stable_id_resolver = Some(resolver);
        self
    }
}

#[derive(Clone)]
pub struct GrpcAuthService<S> {
    inner: S,
    security_coordinator: Arc<SecurityCoordinator>,
    header_trust: HeaderTrustPolicy,
    tier_header_trust: HeaderTrustPolicy,
    stable_id_resolver: Option<Arc<dyn proximadb_tenant::TenantStableIdResolver>>,
}

impl<S> Layer<S> for GrpcAuthLayer {
    type Service = GrpcAuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GrpcAuthService {
            inner,
            security_coordinator: self.security_coordinator.clone(),
            header_trust: self.header_trust,
            tier_header_trust: self.tier_header_trust,
            stable_id_resolver: self.stable_id_resolver.clone(),
        }
    }
}

impl<S, B> Service<HttpRequest<B>> for GrpcAuthService<S>
where
    S: Service<HttpRequest<B>, Response = HttpResponse<tonic::body::Body>, Error = Infallible>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = HttpResponse<tonic::body::Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: HttpRequest<B>) -> Self::Future {
        let security_coordinator = self.security_coordinator.clone();
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        let header_trust = self.header_trust;
        let tier_header_trust = self.tier_header_trust;
        let stable_id_resolver = self.stable_id_resolver.clone();

        Box::pin(async move {
            match authenticate_http_request(
                &security_coordinator,
                &mut request,
                header_trust,
                tier_header_trust,
                stable_id_resolver,
            )
            .await
            {
                Ok(()) => inner.call(request).await,
                Err(status) => Ok(status_to_http_response(status)),
            }
        })
    }
}

pub async fn authenticate_http_request<B>(
    security_coordinator: &SecurityCoordinator,
    request: &mut HttpRequest<B>,
    header_trust: HeaderTrustPolicy,
    tier_header_trust: HeaderTrustPolicy,
    stable_id_resolver: Option<Arc<dyn proximadb_tenant::TenantStableIdResolver>>,
) -> Result<(), Status> {
    let path = request.uri().path();
    if is_arrow_flight_path(path) {
        return Ok(());
    }

    let auth_data = auth_data_from_headers(request.headers())?;
    let user_context = security_coordinator
        .authenticate_request(auth_data)
        .await
        .map_err(|error| Status::unauthenticated(format!("authentication failed: {error}")))?;

    let capability = DataPlaneCapability::from_user_context(&user_context);
    if let Some(capability) = capability.as_ref() {
        validate_capability_for_grpc_path(capability, path)?;
        validate_tenant_metadata(&user_context, request.headers())?;
    }

    // TD-TENANT-1: reconcile the asserted `x-tenant-id` against the
    // credential's tenant binding through the ONE shared primitive (same call
    // REST / Arrow Flight / pgwire make). This also closes the pre-existing
    // gRPC gap where a NON-capability credential's metadata mismatch was
    // silently ignored instead of rejected.
    // TD-TENANT-3: the shared claim vocabulary, canonical name only.
    let request_headers = request.headers();
    let asserted = proximadb_tenant::tenant_claim(|name| {
        request_headers
            .get(name)
            .and_then(|value| value.to_str().ok())
    })
    .map(|hit| hit.value);
    let binding = user_context
        .tenant_id
        .as_ref()
        .map(|tenant_id| AuthenticatedTenantBinding {
            tenant_id: tenant_id.clone(),
            is_gateway_principal: user_context.is_gateway_principal(),
        });
    let resolved_tenant = match resolve_tenant_assertion(asserted, binding.as_ref(), header_trust) {
        Ok(
            ResolvedTenantAssertion::Asserted(tenant) | ResolvedTenantAssertion::Credential(tenant),
        ) => Some(tenant),
        Ok(ResolvedTenantAssertion::NoTenant) => None,
        Err(error @ TenantAssertionError::Mismatch { .. }) => {
            tracing::warn!(
                target: "proximadb::tenant_audit",
                surface = "grpc",
                %error,
                "rejected x-tenant-id: does not match authenticated tenant binding"
            );
            return Err(Status::permission_denied(error.to_string()));
        }
        Err(error @ TenantAssertionError::UnauthenticatedAssertionRejected { .. }) => {
            tracing::warn!(
                target: "proximadb::tenant_audit",
                surface = "grpc",
                policy = %header_trust,
                %error,
                "rejected bare x-tenant-id without authenticated tenant binding"
            );
            return Err(Status::permission_denied(error.to_string()));
        }
    };

    // Open-core cache tier hook: record the tenant's tier from `x-tenant-tier`
    // metadata (control-plane supplied, opaque id) for the cache policy, before
    // `user_context` is moved into the auth context.
    //
    // ADR-0053 W8: the claim is gated by `tier_header_trust`. A rejected claim
    // is DROPPED (warned once per tenant) — never a gRPC error, since the claim
    // is an entitlement hint and the request proceeds at the default tier.
    // Note: this code runs post-authenticate, so a binding is structurally
    // always present here — `AuthenticatedOnly` can never drop a gRPC claim;
    // only `GatewayOnly` (non-gateway principal) can.
    //
    // The stamp lands on the ACTING tenant (resolved_tenant — the delegated
    // tenant under gateway delegation) falling back to the credential tenant,
    // matching REST's `context.tenant_id` keying.
    let raw_tier_claim = proximadb_tenant::tier_claim(|name| {
        request_headers
            .get(name)
            .and_then(|value| value.to_str().ok())
    });
    let tenant_for_tier = resolved_tenant
        .clone()
        .or_else(|| user_context.tenant_id.clone());
    let tier_claim = match proximadb_tenant::resolve_tier_claim(
        raw_tier_claim,
        binding.as_ref(),
        tier_header_trust,
    ) {
        Ok(tier) => tier,
        Err(rejection) => {
            let tenant = tenant_for_tier.clone().unwrap_or_default();
            crate::network::middleware::tenant::warn_tier_claim_dropped(
                "grpc", &tenant, &rejection,
            );
            None
        }
    };

    // TD-TENANT-1 item 3: resolve the tenant's stable u64 id (the account u32
    // widened) for ABAC. None when no resolver is wired, no tenant resolved, or
    // the tenant is unminted (fail-closed deny).
    let tenant_stable_id = resolved_tenant
        .as_deref()
        .and_then(|t| stable_id_resolver.as_ref().and_then(|r| r.stable_id_of(t)));

    request.extensions_mut().insert(GrpcAuthContext {
        user_context,
        capability,
        resolved_tenant,
        tenant_stable_id,
    });

    if let (Some(tenant), Some(tier)) = (tenant_for_tier, tier_claim) {
        crate::services::record_store::set_tenant_tier(tenant, tier);
    }
    Ok(())
}

pub fn data_plane_capability<T>(request: &Request<T>) -> Option<DataPlaneCapability> {
    request
        .extensions()
        .get::<GrpcAuthContext>()
        .and_then(|context| context.capability.clone())
}

pub fn tenant_id<T>(request: &Request<T>) -> Option<String> {
    if let Some(context) = request.extensions().get::<GrpcAuthContext>() {
        // The auth layer already reconciled assertion vs binding under the
        // deployment policy (TD-TENANT-1); its verdict is authoritative.
        return context
            .resolved_tenant
            .clone()
            .or_else(|| context.user_context.tenant_id.clone());
    }
    // No auth layer on this mount (dev / embedded): bare metadata is the only
    // signal — the legacy Open behavior. Strict policies require the layer.
    tenant_id_from_metadata(request.metadata())
}

/// Resolve and validate the request tenant under the deployment mode installed
/// by [`GrpcTenantModeLayer`]. Missing identity fails closed in multi-tenant
/// mode; an absent layer preserves embedded single-tenant compatibility.
pub fn resolved_tenant_id<T>(request: &Request<T>) -> Result<String, Status> {
    let fallback_mode = proximadb_tenant::TenantDeploymentMode::single_tenant_default();
    let mode = request
        .extensions()
        .get::<proximadb_tenant::TenantDeploymentMode>()
        .unwrap_or(&fallback_mode);

    proximadb_tenant::resolve_request_tenant_for_mode(tenant_id(request).as_deref(), mode).map_err(
        |error| match error {
            proximadb_tenant::ResolveRequestTenantError::MissingTenant => {
                Status::unauthenticated(error.to_string())
            }
            proximadb_tenant::ResolveRequestTenantError::InvalidTenant(_) => {
                Status::invalid_argument(error.to_string())
            }
        },
    )
}

/// Acting principal (user id) for within-tenant row-level RBAC
/// (`permitted_principals`, TD-134). Extracted from the authenticated
/// [`GrpcAuthContext`]; `None` when the request is unauthenticated (e.g. dev /
/// embedded with no auth layer) ⇒ structural isolation only (no per-record
/// filtering) — the same default the fusion seam applies for `principal: None`.
pub fn user_id<T>(request: &Request<T>) -> Option<String> {
    request
        .extensions()
        .get::<GrpcAuthContext>()
        .map(|context| context.user_context.user_id.clone())
        .filter(|id| !id.is_empty())
}

/// TD-TENANT-1 item 3: the request's tenant stable u64 id (the ABAC binding-
/// filter key), stamped by the gRPC auth layer from the resolved tenant via the
/// wired `TenantStableIdResolver`. `None` when no resolver is wired, no tenant
/// is resolved, or the tenant is unminted (fail-closed deny).
pub fn tenant_stable_id<T>(request: &Request<T>) -> Option<u64> {
    request
        .extensions()
        .get::<GrpcAuthContext>()
        .and_then(|context| context.tenant_stable_id)
}

/// L0.3 (ADR-090): the ONE construction of a port identity from gRPC auth
/// state. Composes the three extraction primitives above — it deliberately
/// re-derives NOTHING, so it cannot drift from them — and owns the single
/// auth-class rule for this surface: gRPC subjects come only from the
/// authenticated [`GrpcAuthContext`] (there is no assertion path), so subject
/// presence decides `Authenticated` vs `Anonymous`.
///
/// Replaces the four hand-built `PortIdentity { .. }` literals in
/// `grpc/v2/record_service.rs`, each of which repeated that ternary inline
/// (ADR-090 appendix A gap 13).
pub fn port_identity<T>(
    request: &Request<T>,
) -> Result<proximadb_runtime::OwnedPortIdentity, Status> {
    let tenant_id = resolved_tenant_id(request)?;
    let subject = user_id(request);
    let tenant_stable_id = tenant_stable_id(request);
    let auth_class = if subject.is_some() {
        proximadb_tenant::AuthClass::Authenticated
    } else {
        proximadb_tenant::AuthClass::Anonymous
    };
    Ok(proximadb_runtime::OwnedPortIdentity {
        tenant_id: Some(tenant_id),
        subject,
        tenant_stable_id,
        auth_class,
    })
}

pub fn enforce_data_plane_request<T>(
    request: &Request<T>,
    operation: &str,
    collection: &str,
    record_count: usize,
    byte_count: Option<u64>,
) -> Result<(), Status> {
    if let Some(capability) = request
        .extensions()
        .get::<GrpcAuthContext>()
        .and_then(|context| context.capability.as_ref())
    {
        validate_data_plane_capability(
            capability,
            operation,
            collection,
            record_count,
            byte_count,
        )?;
    }
    Ok(())
}

pub fn validate_data_plane_capability(
    capability: &DataPlaneCapability,
    operation: &str,
    collection: &str,
    record_count: usize,
    byte_count: Option<u64>,
) -> Result<(), Status> {
    ensure_protocol(capability, "grpc")?;
    ensure_operation(capability, operation)?;
    ensure_collection(capability, collection)?;
    ensure_scope(capability, operation)?;
    capability
        .ensure_record_count(record_count)
        .map_err(Status::resource_exhausted)?;
    if let Some(max_bytes) = capability.max_bytes
        && let Some(byte_count) = byte_count
        && byte_count > max_bytes
    {
        return Err(Status::resource_exhausted(format!(
            "Request has {byte_count} bytes, exceeding capability limit {max_bytes}"
        )));
    }
    Ok(())
}

fn auth_data_from_headers(headers: &HeaderMap) -> Result<AuthenticationData, Status> {
    let auth_header = headers
        .get(http::header::AUTHORIZATION)
        .ok_or_else(|| Status::unauthenticated("authorization metadata is required"))?
        .to_str()
        .map_err(|_| Status::invalid_argument("authorization metadata is not valid ASCII"))?;

    // TD-ABAC-6: the shared credential parser. This was a verbatim copy of
    // Arrow's `auth_data_from_metadata` and REST's `map_header_to_auth_data`
    // (same Bearer / API-Key / raw logic); all three now share one parser.
    Ok(crate::security::request_identity::parse_authorization(
        auth_header,
    ))
}

fn validate_capability_for_grpc_path(
    capability: &DataPlaneCapability,
    path: &str,
) -> Result<(), Status> {
    ensure_protocol(capability, "grpc")?;
    let operation = infer_grpc_data_plane_operation(path).ok_or_else(|| {
        Status::permission_denied("capability token is not valid for this gRPC method")
    })?;
    ensure_operation(capability, operation)?;
    ensure_scope(capability, operation)?;
    Ok(())
}

fn validate_tenant_metadata(
    user_context: &UnifiedUserContext,
    headers: &HeaderMap,
) -> Result<(), Status> {
    let token_tenant = user_context
        .tenant_id
        .as_deref()
        .ok_or_else(|| Status::permission_denied("capability token requires tenant_id"))?;
    // TD-TENANT-3: same vocabulary as the assertion path above, so the
    // capability check and the trust gate can never read different headers.
    if let Some(header_tenant) =
        proximadb_tenant::tenant_claim(|name| headers.get(name).and_then(|v| v.to_str().ok()))
            .map(|hit| hit.value)
        && header_tenant != token_tenant
    {
        return Err(Status::permission_denied(
            "x-tenant-id does not match capability tenant_id",
        ));
    }
    Ok(())
}

fn ensure_protocol(capability: &DataPlaneCapability, expected: &str) -> Result<(), Status> {
    if capability.protocol.as_deref() != Some(expected) {
        return Err(Status::permission_denied(format!(
            "capability token is not valid for {expected}"
        )));
    }
    Ok(())
}

fn ensure_operation(capability: &DataPlaneCapability, expected: &str) -> Result<(), Status> {
    if capability.operation.as_deref() != Some(expected) {
        return Err(Status::permission_denied(
            "capability operation does not match gRPC method",
        ));
    }
    Ok(())
}

fn ensure_collection(capability: &DataPlaneCapability, expected: &str) -> Result<(), Status> {
    if capability.collection.as_deref() != Some(expected) {
        return Err(Status::permission_denied(
            "capability collection does not match gRPC request",
        ));
    }
    Ok(())
}

fn ensure_scope(capability: &DataPlaneCapability, operation: &str) -> Result<(), Status> {
    let allowed = match operation {
        "ingest" => capability
            .scopes
            .iter()
            .any(|scope| scope == "records:write"),
        "search" => capability
            .scopes
            .iter()
            .any(|scope| scope == "search:execute" || scope == "records:read"),
        _ => false,
    };
    if !allowed {
        return Err(Status::permission_denied(
            "capability token lacks the required scope",
        ));
    }
    Ok(())
}

fn infer_grpc_data_plane_operation(path: &str) -> Option<&'static str> {
    match path {
        "/proximadb.v2.ProximaRecordService/InsertRecords"
        | "/proximadb.v2.ProximaRecordService/UpsertRecords"
        | "/proximadb.v2.ProximaRecordService/UpdateRecords"
        | "/proximadb.v2.ProximaRecordService/DeleteRecords"
        | "/proximadb.v2.ProximaRecordService/BatchWriteStream" => Some("ingest"),
        "/proximadb.v2.ProximaRecordService/Search"
        | "/proximadb.v2.ProximaRecordService/SearchStream" => Some("search"),
        _ => None,
    }
}

fn is_arrow_flight_path(path: &str) -> bool {
    path.starts_with("/arrow.flight.protocol.FlightService/")
}

fn tenant_id_from_metadata(metadata: &tonic::metadata::MetadataMap) -> Option<String> {
    // TD-TENANT-3: the shared claim vocabulary, canonical name only.
    proximadb_tenant::tenant_claim(|name| metadata.get(name).and_then(|value| value.to_str().ok()))
        .map(|hit| hit.value.to_owned())
}

// `http::Response::Builder::body` only fails if a header is invalid.
// All headers above are static strings or grpc-status digits, and the
// message has had embedded newlines stripped. The `expect` documents
// the static-construction invariant.
#[allow(clippy::expect_used)]
fn status_to_http_response(status: Status) -> HttpResponse<tonic::body::Body> {
    let code = status.code();
    let message = status.message().replace('\n', " ");
    HttpResponse::builder()
        .status(http::StatusCode::OK)
        .header(http::header::CONTENT_TYPE, "application/grpc")
        .header("grpc-status", grpc_status_code(code))
        .header("grpc-message", message)
        .body(tonic::body::Body::empty())
        .expect("gRPC status response should be valid")
}

fn grpc_status_code(code: Code) -> &'static str {
    match code {
        Code::Ok => "0",
        Code::Cancelled => "1",
        Code::Unknown => "2",
        Code::InvalidArgument => "3",
        Code::DeadlineExceeded => "4",
        Code::NotFound => "5",
        Code::AlreadyExists => "6",
        Code::PermissionDenied => "7",
        Code::ResourceExhausted => "8",
        Code::FailedPrecondition => "9",
        Code::Aborted => "10",
        Code::OutOfRange => "11",
        Code::Unimplemented => "12",
        Code::Internal => "13",
        Code::Unavailable => "14",
        Code::DataLoss => "15",
        Code::Unauthenticated => "16",
    }
}

#[cfg(test)]
mod tests {
    // --- L0.3 port_identity spec (ADR-090) -------------------------------

    fn authed_request(user: &str, stable: Option<u64>) -> Request<()> {
        let mut request = Request::new(());
        let mut user_context = proximadb_tenant::UnifiedUserContext::anonymous();
        user_context.user_id = user.to_string();
        request.extensions_mut().insert(GrpcAuthContext {
            user_context,
            capability: None,
            resolved_tenant: Some("tenant-a".to_string()),
            tenant_stable_id: stable,
        });
        request
    }

    /// Subject present ⇒ Authenticated, with tenant + stable id mapped through
    /// the SAME primitives the individual helpers expose.
    #[test]
    fn port_identity_authenticated_when_subject_present() {
        let request = authed_request("alice", Some(7));
        let identity = port_identity(&request).expect("identity");
        assert_eq!(identity.subject.as_deref(), Some("alice"));
        assert_eq!(identity.tenant_stable_id, Some(7));
        assert!(identity.tenant_id.is_some());
        assert!(matches!(
            identity.auth_class,
            proximadb_tenant::AuthClass::Authenticated
        ));
    }

    /// Empty user id ⇒ no subject ⇒ Anonymous (gRPC has no assertion path;
    /// the class comes ONLY from authenticated context presence).
    #[test]
    fn port_identity_anonymous_when_subject_absent() {
        let request = authed_request("", None);
        let identity = port_identity(&request).expect("identity");
        assert_eq!(identity.subject, None);
        assert_eq!(identity.tenant_stable_id, None);
        assert!(matches!(
            identity.auth_class,
            proximadb_tenant::AuthClass::Anonymous
        ));
    }

    use super::*;
    use std::collections::HashMap;

    use crate::network::auth::JwtService;
    use crate::network::auth::config::JwtAlgorithm;
    use crate::network::auth::jwt::{Claims, TokenType};
    use crate::security::auth_service::{
        AuthenticationConfig, AuthenticationMethod, JwtConfig, MtlsConfig, SSOConfig,
    };
    use crate::security::rbac_service::RBACConfig;
    use crate::security::security_coordinator::{ComplianceConfig, TlsConfig};
    use crate::security::{AuditConfig, SecurityConfig, SecurityMode};

    #[test]
    fn resolved_tenant_id_defaults_in_single_tenant_mode() {
        // A bare gRPC/Flight request (no auth context, no `x-tenant-id` metadata)
        // resolves to the ONE canonical default — the same bucket REST and pgwire
        // use — instead of the old `""` that split it from REST's `"default"`.
        let mut req = Request::new(());
        req.extensions_mut()
            .insert(proximadb_tenant::TenantDeploymentMode::single_tenant_default());
        assert_eq!(tenant_id(&req), None);
        assert_eq!(
            resolved_tenant_id(&req).unwrap(),
            proximadb_tenant::DEFAULT_TENANT
        );
    }

    #[test]
    fn resolved_tenant_id_rejects_missing_tenant_in_multi_tenant_mode() {
        let mut req = Request::new(());
        req.extensions_mut()
            .insert(proximadb_tenant::TenantDeploymentMode::MultiTenant);

        let status = resolved_tenant_id(&req).unwrap_err();
        assert_eq!(status.code(), Code::Unauthenticated);
        assert_eq!(
            status.message(),
            "tenant id is required in multi-tenant mode"
        );
    }

    #[test]
    fn resolved_tenant_id_accepts_explicit_tenant_in_multi_tenant_mode() {
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("x-tenant-id", "tenant-a".parse().unwrap());
        req.extensions_mut()
            .insert(proximadb_tenant::TenantDeploymentMode::MultiTenant);

        assert_eq!(resolved_tenant_id(&req).unwrap(), "tenant-a");
    }

    fn security_config() -> SecurityConfig {
        SecurityConfig {
            enabled: true,
            mode: SecurityMode::Development,
            authentication: AuthenticationConfig {
                enabled: true,
                methods: vec![AuthenticationMethod::JWT],
                require_authentication: true,
                default_session_timeout_minutes: 60,
                api_keys: HashMap::new(),
                scram_users: HashMap::new(),
                jwt: JwtConfig {
                    enabled: true,
                    secret: "dev-jwt-secret".to_string(),
                    access_token_expiration_minutes: 15,
                    refresh_token_expiration_days: 7,
                    issuer: "operator-control-plane".to_string(),
                    audience: "proximadb-data-plane".to_string(),
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

    fn capability_token(protocol: &str, operation: &str, collection: &str) -> String {
        capability_token_for_tenant(protocol, operation, collection, "tenant-a")
    }

    fn capability_token_for_tenant(
        protocol: &str,
        operation: &str,
        collection: &str,
        tenant: &str,
    ) -> String {
        let now = chrono::Utc::now().timestamp();
        let claims = Claims {
            sub: "key-a".to_string(),
            iat: now - 10,
            exp: now + 300,
            nbf: now - 10,
            iss: "operator-control-plane".to_string(),
            aud: "proximadb-data-plane".to_string(),
            jti: "capability-token".to_string(),
            tenant_id: Some(tenant.to_string()),
            roles: vec!["data_plane".to_string()],
            typ: TokenType::Access,
            capability_type: Some("operator.data-plane.capability.v1".to_string()),
            collection: Some(collection.to_string()),
            operation: Some(operation.to_string()),
            protocol: Some(protocol.to_string()),
            mode: Some("async".to_string()),
            scopes: match operation {
                "search" => vec!["records:read".to_string(), "search:execute".to_string()],
                _ => vec!["records:write".to_string(), "metering:emit".to_string()],
            },
            max_records: Some(10),
            max_bytes: Some(4096),
            tier: Some("business".to_string()),
            route_visibility: Some("public".to_string()),
            metering_required: Some(true),
        };
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret("dev-jwt-secret".as_bytes()),
        )
        .expect("capability JWT should encode")
    }

    async fn coordinator() -> SecurityCoordinator {
        SecurityCoordinator::from_config(security_config())
            .await
            .expect("security coordinator should initialize")
    }

    #[tokio::test]
    async fn grpc_auth_accepts_capability_for_v2_search() {
        let token = capability_token("grpc", "search", "example_knowledge");
        let mut request = HttpRequest::builder()
            .uri("/proximadb.v2.ProximaRecordService/Search")
            .header("authorization", format!("Bearer {token}"))
            .header("x-tenant-id", "tenant-a")
            .body(())
            .expect("request should build");

        authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect("valid gRPC capability should authenticate");

        let context = request
            .extensions()
            .get::<GrpcAuthContext>()
            .expect("auth context should be inserted");
        assert_eq!(context.user_context.tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(
            context
                .capability
                .as_ref()
                .and_then(|cap| cap.protocol.as_deref()),
            Some("grpc")
        );
    }

    #[tokio::test]
    async fn grpc_auth_rejects_capability_for_wrong_protocol() {
        let token = capability_token("rest", "search", "example_knowledge");
        let mut request = HttpRequest::builder()
            .uri("/proximadb.v2.ProximaRecordService/Search")
            .header("authorization", format!("Bearer {token}"))
            .header("x-tenant-id", "tenant-a")
            .body(())
            .expect("request should build");

        let status = authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect_err("REST capability must not authorize generic gRPC");
        assert_eq!(status.code(), Code::PermissionDenied);
    }

    #[tokio::test]
    async fn grpc_auth_rejects_capability_for_non_data_plane_method() {
        let token = capability_token("grpc", "search", "example_knowledge");
        let mut request = HttpRequest::builder()
            .uri("/proximadb.v1.CollectionService/ListCollections")
            .header("authorization", format!("Bearer {token}"))
            .header("x-tenant-id", "tenant-a")
            .body(())
            .expect("request should build");

        let status = authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect_err("capability must not authorize catalog methods");
        assert_eq!(status.code(), Code::PermissionDenied);
    }

    #[test]
    fn data_plane_body_validation_rejects_collection_and_limit_mismatches() {
        let user_context = UnifiedUserContext {
            user_id: "key-a".to_string(),
            tenant_id: Some("tenant-a".to_string()),
            roles: vec!["data_plane".to_string()],
            effective_permissions: Default::default(),
            auth_method: crate::security::UnifiedAuthMethod::JWT,
            session_id: "session-a".to_string(),
            expires_at: None,
            created_at: chrono::Utc::now(),
            metadata: HashMap::from([
                (
                    "capability_type".to_string(),
                    "operator.data-plane.capability.v1".to_string(),
                ),
                ("collection".to_string(), "example_knowledge".to_string()),
                ("operation".to_string(), "ingest".to_string()),
                ("protocol".to_string(), "grpc".to_string()),
                ("scopes".to_string(), "records:write".to_string()),
                ("max_records".to_string(), "2".to_string()),
                ("max_bytes".to_string(), "100".to_string()),
            ]),
        };
        let capability =
            DataPlaneCapability::from_user_context(&user_context).expect("capability should parse");

        validate_data_plane_capability(&capability, "ingest", "example_knowledge", 2, Some(100))
            .expect("matching request should pass");
        assert_eq!(
            validate_data_plane_capability(&capability, "ingest", "other", 1, Some(1))
                .expect_err("collection mismatch should fail")
                .code(),
            Code::PermissionDenied
        );
        assert_eq!(
            validate_data_plane_capability(&capability, "ingest", "example_knowledge", 3, Some(1))
                .expect_err("record limit should fail")
                .code(),
            Code::ResourceExhausted
        );
        assert_eq!(
            validate_data_plane_capability(
                &capability,
                "ingest",
                "example_knowledge",
                1,
                Some(101)
            )
            .expect_err("byte limit should fail")
            .code(),
            Code::ResourceExhausted
        );
    }

    #[tokio::test]
    async fn arrow_flight_paths_are_left_to_flight_auth() {
        let mut request = HttpRequest::builder()
            .uri("/arrow.flight.protocol.FlightService/DoPut")
            .body(())
            .expect("request should build");

        authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect("Flight service performs its own auth");
        assert!(request.extensions().get::<GrpcAuthContext>().is_none());
    }

    #[tokio::test]
    async fn grpc_auth_rejects_tenant_header_mismatch() {
        let token = capability_token("grpc", "search", "example_knowledge");
        let mut request = HttpRequest::builder()
            .uri("/proximadb.v2.ProximaRecordService/Search")
            .header("authorization", format!("Bearer {token}"))
            .header("x-tenant-id", "tenant-b")
            .body(())
            .expect("request should build");

        let status = authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect_err("tenant header mismatch must fail");
        assert_eq!(status.code(), Code::PermissionDenied);
    }

    #[tokio::test]
    async fn grpc_auth_accepts_normal_jwt_without_data_plane_capability() {
        let jwt_service = JwtService::new(crate::network::auth::config::JwtConfig {
            secret: Some("dev-jwt-secret".to_string()),
            expiration_secs: 900,
            refresh_expiration_secs: 86400,
            issuer: "operator-control-plane".to_string(),
            audience: "proximadb-data-plane".to_string(),
            algorithm: JwtAlgorithm::HS256,
        })
        .expect("jwt service should initialize");
        let token = jwt_service
            .generate_token_pair(
                "admin-user",
                Some("tenant-a".to_string()),
                vec!["admin".to_string()],
            )
            .await
            .expect("jwt should generate")
            .access_token;
        let mut request = HttpRequest::builder()
            .uri("/proximadb.v1.CollectionService/ListCollections")
            .header("authorization", format!("Bearer {token}"))
            .body(())
            .expect("request should build");

        authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect("normal JWT should authenticate");
        assert!(
            request
                .extensions()
                .get::<GrpcAuthContext>()
                .and_then(|context| context.capability.as_ref())
                .is_none()
        );
    }

    async fn normal_jwt(tenant: Option<&str>) -> String {
        let jwt_service = JwtService::new(crate::network::auth::config::JwtConfig {
            secret: Some("dev-jwt-secret".to_string()),
            expiration_secs: 900,
            refresh_expiration_secs: 86400,
            issuer: "operator-control-plane".to_string(),
            audience: "proximadb-data-plane".to_string(),
            algorithm: JwtAlgorithm::HS256,
        })
        .expect("jwt service should initialize");
        jwt_service
            .generate_token_pair(
                "admin-user",
                tenant.map(str::to_string),
                vec!["admin".to_string()],
            )
            .await
            .expect("jwt should generate")
            .access_token
    }

    /// TD-TENANT-1 gap closure: a NON-capability credential asserting a
    /// different tenant via `x-tenant-id` was previously silently ignored
    /// (the authenticated tenant just won). It is now a PERMISSION_DENIED
    /// in every policy mode — the same masquerade rejection REST applies.
    #[tokio::test]
    async fn grpc_auth_rejects_normal_jwt_tenant_metadata_mismatch() {
        let token = normal_jwt(Some("tenant-a")).await;
        let mut request = HttpRequest::builder()
            .uri("/proximadb.v1.CollectionService/ListCollections")
            .header("authorization", format!("Bearer {token}"))
            .header("x-tenant-id", "tenant-b")
            .body(())
            .expect("request should build");

        let status = authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect_err("normal-JWT tenant metadata mismatch must fail");
        assert_eq!(status.code(), Code::PermissionDenied);
    }

    /// A credential with NO tenant binding asserting a tenant via metadata:
    /// accepted under `open` (resolved tenant = the assertion), rejected
    /// under `authenticated-only`.
    #[tokio::test]
    async fn grpc_auth_applies_policy_to_unbound_credential_assertions() {
        let token = normal_jwt(None).await;
        let build = |token: &str| {
            HttpRequest::builder()
                .uri("/proximadb.v1.CollectionService/ListCollections")
                .header("authorization", format!("Bearer {token}"))
                .header("x-tenant-id", "demo1")
                .body(())
                .expect("request should build")
        };

        let mut open_request = build(&token);
        authenticate_http_request(
            &coordinator().await,
            &mut open_request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect("open policy accepts the assertion");
        assert_eq!(
            open_request
                .extensions()
                .get::<GrpcAuthContext>()
                .and_then(|context| context.resolved_tenant.as_deref()),
            Some("demo1")
        );

        let mut strict_request = build(&token);
        let status = authenticate_http_request(
            &coordinator().await,
            &mut strict_request,
            HeaderTrustPolicy::AuthenticatedOnly,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect_err("strict policy rejects the unbound assertion");
        assert_eq!(status.code(), Code::PermissionDenied);
    }

    // ── Tier-claim gate (ADR-0053 W8) ───────────────────────────────────

    /// A capability token bound to `tenant` (the stock `capability_token`
    /// hardcodes `tenant-a` — these tests need unique tenants because
    /// `TENANT_TIERS` is a process-global insert-only registry).
    fn tier_token(tenant: &str) -> String {
        capability_token_for_tenant("grpc", "search", "example_knowledge", tenant)
    }

    /// A gRPC request carrying an `x-tenant-tier` claim, authenticated with
    /// the plain (non-gateway) data-plane capability token bound to `tenant`.
    fn tier_claim_request(tenant: &str, claim: &str) -> HttpRequest<()> {
        let token = tier_token(tenant);
        HttpRequest::builder()
            .uri("/proximadb.v2.ProximaRecordService/Search")
            .header("authorization", format!("Bearer {token}"))
            .header("x-tenant-id", tenant)
            .header("x-tenant-tier", claim)
            .body(())
            .expect("request should build")
    }

    #[tokio::test]
    async fn grpc_tier_claim_open_stamps_acting_tenant() {
        let mut request = tier_claim_request("tier-grpc-open-tenant", "pro");
        authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::Open,
            None,
        )
        .await
        .expect("authenticated request");
        assert_eq!(
            crate::services::record_store::tenant_tier("tier-grpc-open-tenant").as_deref(),
            Some("pro")
        );
    }

    #[tokio::test]
    async fn grpc_tier_claim_gateway_only_drops_non_gateway_principal() {
        let mut request = tier_claim_request("tier-grpc-strict-tenant", "enterprise");
        authenticate_http_request(
            &coordinator().await,
            &mut request,
            HeaderTrustPolicy::Open,
            HeaderTrustPolicy::GatewayOnly,
            None,
        )
        .await
        .expect("the dropped claim must not fail the request");
        assert_eq!(
            crate::services::record_store::tenant_tier("tier-grpc-strict-tenant"),
            None,
            "a plain data-plane credential must not self-assign a tier"
        );
    }
}
